//! Unified recall service used by both CLI and MCP.
//! Implements the full pipeline: waterfall search + cross-validation.

use crate::config::ReinConfig;
use crate::embed::EmbedCache;
use crate::store::SqliteStore;
use crate::sync::{auto_memory::AutoMemoryScanner, supermemory::SupermemoryClient, validate};
use crate::types::Embedder as _;
use crate::types::{Memory, MemoryStore, ReinResult};

/// A recalled memory with score and confidence.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RecallResult {
    pub memory: Memory,
    pub score: f32,
    pub confidence: f32,
    pub sources_hit: usize,
    pub evidence_count: usize,
    pub evidence_preview: Vec<String>,
    /// v0.26 Cap C: condensed LLM summary surfaced when the underlying
    /// memory is in the `Cold` tier AND `[ars].cold_archive_enabled = true`
    /// AND the row carries a non-NULL `archival_summary` at the current
    /// `ARCHIVAL_SUMMARY_VERSION`. `None` for Hot/Warm tiers, when the
    /// feature flag is off, or when the stored summary is at a stale
    /// version (the worker will regenerate it on the next pass).
    ///
    /// Clients (MCP / REST / GUI) MUST continue to render `memory.content`
    /// when this is `None` — the summary is an enhancement, never a
    /// substitute. Non-cold memories deliberately surface `None` even when
    /// the column has data; only the cold tier benefits from the condensed
    /// view (per contract §2.6).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub archival_summary: Option<String>,
}

/// v0.26 Cap C: populate `RecallResult.archival_summary` when the gate
/// permits — separated as a pure helper so tests can drive it without a
/// full `SqliteStore` dance. Mirror of the gate condition in contract §2.6.
///
/// Returns `Some(summary)` iff:
/// - `cold_archive_enabled = true`
/// - `memory.tier == MemoryTier::Cold`
/// - `memory.archival_summary.is_some()`
/// - `memory.archival_summary_version == Some(ARCHIVAL_SUMMARY_VERSION)`
///
/// All other inputs (warm/hot tiers, feature off, stale version, missing
/// summary) → `None`. Pure function.
pub(crate) fn maybe_archival_summary_for_recall(
    cold_archive_enabled: bool,
    memory: &Memory,
) -> Option<String> {
    if !cold_archive_enabled {
        return None;
    }
    if memory.tier != crate::types::MemoryTier::Cold {
        return None;
    }
    let summary = memory.archival_summary.as_ref()?;
    let version = memory.archival_summary_version?;
    if version != crate::ops::cold_archive_summary::ARCHIVAL_SUMMARY_VERSION {
        return None;
    }
    Some(summary.clone())
}

fn sort_recall_results(results: &mut [RecallResult]) {
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.memory.support_count.cmp(&a.memory.support_count))
            .then_with(|| {
                b.memory
                    .source_diversity
                    .partial_cmp(&a.memory.source_diversity)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| {
                b.confidence
                    .partial_cmp(&a.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| b.memory.updated_at.cmp(&a.memory.updated_at))
    });
}

fn apply_ars_dynamic_fusion_scores(
    fused: Vec<(String, f32)>,
    fts_norm_log: &std::collections::HashMap<String, f32>,
    vec_norm_log: &std::collections::HashMap<String, f32>,
    kg_norm_log: &std::collections::HashMap<String, f32>,
    episode_norm_log: &std::collections::HashMap<String, f32>,
    memory_map: &std::collections::HashMap<String, Memory>,
    weights: Option<crate::search::alpha_optimizer::ShadowFusionWeights>,
    runtime_adoption_weight: f64,
) -> Vec<(String, f32)> {
    let Some(weights) = weights.map(|w| w.normalized_or_default()) else {
        return fused;
    };

    // v0.28.7+ audit M-6 fix: outer-blend the ARS simplex score against the
    // route-aware `legacy_score` by `runtime_adoption_weight`. Pre-fix the
    // simplex score replaced `legacy_score` wholesale whenever `weights` was
    // Some, which silently nuked route-specific alpha (most visibly
    // ExactKeyword's alpha=0.85 BM25-heavy signal — it became indistinguishable
    // from the generic simplex). The inner trust blend in
    // `ready_shadow_fusion_weights_for_recall::effective_simplex` blends
    // weights against a static prior; this OUTER blend smooths the
    // simplex-vs-legacy transition so a barely-promoted canary
    // (adoption=0.05) does not lose 95% of the route-specific signal in one
    // step. `adoption=0` reproduces pre-canary behavior exactly; `adoption=1`
    // reproduces the pre-fix wholesale-simplex behavior.
    let adoption = runtime_adoption_weight.clamp(0.0, 1.0) as f32;
    let legacy_share = 1.0 - adoption;

    let mut rescored: Vec<(String, f32, usize)> = fused
        .into_iter()
        .enumerate()
        .map(|(idx, (id, legacy_score))| {
            let Some(memory) = memory_map.get(&id) else {
                return (id, legacy_score, idx);
            };
            let score = weights.bm25 * fts_norm_log.get(&id).copied().unwrap_or(0.0) as f64
                + weights.vec * vec_norm_log.get(&id).copied().unwrap_or(0.0) as f64
                + weights.kg * kg_norm_log.get(&id).copied().unwrap_or(0.0) as f64
                + weights.episode * episode_norm_log.get(&id).copied().unwrap_or(0.0) as f64
                + weights.support
                    * crate::search::alpha_optimizer::support_signal(memory.support_count)
                + weights.diversity
                    * crate::search::alpha_optimizer::diversity_signal(memory.source_diversity);
            let simplex = if score.is_finite() {
                score as f32
            } else {
                legacy_score
            };
            let blended = adoption * simplex + legacy_share * legacy_score;
            let score = if blended.is_finite() {
                blended
            } else {
                legacy_score
            };
            (id, score, idx)
        })
        .collect();
    rescored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.2.cmp(&b.2))
    });
    rescored
        .into_iter()
        .map(|(id, score, _idx)| (id, score))
        .collect()
}

fn ready_shadow_fusion_weights_for_recall(
    state: &crate::store::adaptive::AdaptiveState,
    config: &ReinConfig,
    query_type: &str,
    cluster_id: Option<u32>,
    runtime_adoption_weight: f64,
) -> Option<crate::search::alpha_optimizer::ShadowFusionWeights> {
    let production_canary = runtime_adoption_weight > f64::EPSILON;
    if !config.adaptive.enabled
        || !config.ars.acceleration.enabled
        || config.ars.acceleration.shadow_only
        || !production_canary
    {
        return None;
    }
    let entry = state.get_shadow_fusion_weights(
        query_type,
        cluster_id,
        config.adaptive.min_samples_alpha,
    )?;
    let static_prior = crate::search::alpha_optimizer::ShadowFusionWeights::default();
    let effective = crate::ops::ars_tuning::effective_simplex(
        [
            static_prior.bm25,
            static_prior.vec,
            static_prior.kg,
            static_prior.episode,
            static_prior.support,
            static_prior.diversity,
        ],
        [
            entry.weights.bm25,
            entry.weights.vec,
            entry.weights.kg,
            entry.weights.episode,
            entry.weights.support,
            entry.weights.diversity,
        ],
        crate::ops::ars_tuning::TrustInputs {
            enabled: config.ars.acceleration.enabled,
            production_canary,
            runtime_adoption_weight,
            human_count: entry.sample_count as u64,
            llm_count: 0,
            llm_reliability: 0.0,
            calibration: 1.0,
            stability: 1.0,
            drift_alert: false,
            prior_strength: config.adaptive.shrinkage_prior,
            max_trust: 0.85,
        },
    );
    Some(crate::search::alpha_optimizer::ShadowFusionWeights {
        bm25: effective[0],
        vec: effective[1],
        kg: effective[2],
        episode: effective[3],
        support: effective[4],
        diversity: effective[5],
    })
}

fn collapse_results_to_canonicals(
    store: &SqliteStore,
    results: Vec<RecallResult>,
) -> ReinResult<Vec<RecallResult>> {
    let mut ordered_ids = Vec::new();
    let mut meta: std::collections::HashMap<String, (f32, f32, usize)> =
        std::collections::HashMap::new();
    let mut fallback: std::collections::HashMap<String, Memory> = std::collections::HashMap::new();
    let mut passthrough = Vec::new();

    for result in results {
        if result.memory.id.starts_with("sm:") || result.memory.id.starts_with("auto:") {
            passthrough.push(result);
            continue;
        }
        let canonical_id = store.canonical_id_for(&result.memory.id)?;
        if !meta.contains_key(&canonical_id) {
            ordered_ids.push(canonical_id.clone());
        }
        meta.entry(canonical_id.clone())
            .and_modify(|entry| {
                entry.0 = entry.0.max(result.score);
                entry.1 = entry.1.max(result.confidence);
                entry.2 = entry.2.max(result.sources_hit);
            })
            .or_insert((result.score, result.confidence, result.sources_hit));
        fallback.entry(canonical_id).or_insert(result.memory);
    }

    let mut canonical_map: std::collections::HashMap<String, Memory> = store
        .get_batch(&ordered_ids)
        .into_iter()
        .map(|memory| (memory.id.clone(), memory))
        .collect();

    let mut collapsed = Vec::new();
    for canonical_id in ordered_ids {
        if let Some((score, confidence, sources_hit)) = meta.remove(&canonical_id) {
            if let Some(memory) = canonical_map
                .remove(&canonical_id)
                .or_else(|| fallback.remove(&canonical_id))
            {
                collapsed.push(RecallResult {
                    memory,
                    score,
                    confidence,
                    sources_hit,
                    evidence_count: 0,
                    evidence_preview: vec![],
                    archival_summary: None,
                });
            }
        }
    }
    collapsed.extend(passthrough);
    Ok(collapsed)
}

fn build_evidence_preview(
    store: &SqliteStore,
    memory: &Memory,
    preview_limit: usize,
) -> (usize, Vec<String>) {
    if memory.id.starts_with("sm:") || memory.id.starts_with("auto:") {
        return (0, vec![]);
    }

    let total = memory.support_count.saturating_sub(1) as usize;
    if total == 0 || preview_limit == 0 {
        return (total, vec![]);
    }

    let evidence = store
        .list_memory_evidence(&memory.id, preview_limit.saturating_add(1))
        .unwrap_or_default();
    let preview = evidence
        .into_iter()
        .filter(|item| item.memory_id.as_deref() != Some(memory.id.as_str()))
        .take(preview_limit)
        .map(|item| format!("[{}] {}", item.source_topic, item.summary))
        .collect();

    (total, preview)
}

fn enrich_results_with_evidence(
    store: &SqliteStore,
    results: &mut [RecallResult],
    preview_limit: usize,
) {
    for result in results {
        let (evidence_count, evidence_preview) =
            build_evidence_preview(store, &result.memory, preview_limit);
        result.evidence_count = evidence_count;
        result.evidence_preview = evidence_preview;
    }
}

fn apply_evidence_rerank(
    store: &SqliteStore,
    query: &str,
    results: &mut [RecallResult],
    evidence_limit: usize,
) {
    for result in results {
        if result.memory.support_count <= 1
            || (result.confidence >= 0.85 && result.sources_hit >= 2)
        {
            continue;
        }
        let evidence = store
            .list_memory_evidence(&result.memory.id, evidence_limit.saturating_add(1))
            .unwrap_or_default();
        let best_sim = evidence
            .into_iter()
            .filter(|item| item.memory_id.as_deref() != Some(result.memory.id.as_str()))
            .map(|item| {
                crate::extract::similarity(query, &item.summary)
                    .max(crate::extract::similarity(query, &item.content))
            })
            .fold(0.0f32, f32::max);

        if best_sim > 0.0 {
            let support_scale = (result.memory.support_count.min(4) as f32 - 1.0).max(0.0) / 3.0;
            result.score += 0.08 * best_sim * support_scale;
        }
    }
}

fn matches_external_filters(
    memory: &Memory,
    topic: Option<&str>,
    keyword: Option<&str>,
    time_from: Option<chrono::DateTime<chrono::Utc>>,
    time_to: Option<chrono::DateTime<chrono::Utc>>,
) -> bool {
    if let Some(topic) = topic {
        if memory.topic != topic {
            return false;
        }
    }
    if let Some(from) = time_from {
        if memory.created_at < from {
            return false;
        }
    }
    if let Some(to) = time_to {
        if memory.created_at > to {
            return false;
        }
    }
    if let Some(keyword) = keyword {
        let keyword_lower = keyword.to_lowercase();
        if !memory
            .keywords
            .iter()
            .any(|value| value.to_lowercase().contains(&keyword_lower))
            && !memory.content.to_lowercase().contains(&keyword_lower)
        {
            return false;
        }
    }
    true
}

/// Full recall pipeline: waterfall search + optional cross-validation.
///
/// This is sync-safe: embedding uses reqwest::blocking if needed.
pub fn recall(
    store: &SqliteStore,
    config: &ReinConfig,
    query: &str,
    topic: Option<&str>,
    keyword: Option<&str>,
    limit: usize,
) -> ReinResult<Vec<RecallResult>> {
    recall_temporal(
        store, config, query, topic, keyword, limit, None, None, None, false,
    )
}

/// Fast recall: local-only search (FTS + HNSW + KG + linear reranker).
/// Skips expansion, LLM reranker, and Supermemory. ~50-100ms latency.
/// Designed for proxy mode and hook_prompt where latency is critical.
pub fn recall_fast(
    store: &SqliteStore,
    config: &ReinConfig,
    query: &str,
    topic: Option<&str>,
    keyword: Option<&str>,
    limit: usize,
) -> ReinResult<Vec<RecallResult>> {
    recall_temporal(
        store,
        config,
        query,
        topic,
        keyword,
        limit,
        None,
        None,
        Some(false),
        true,
    )
}

/// Full recall pipeline with optional temporal filtering.
/// `expand_override`: Some(true) forces expansion, Some(false) disables, None uses config.
/// `fast`: if true, skips expansion, LLM reranker, and Supermemory cross-validation.
pub fn recall_temporal(
    store: &SqliteStore,
    config: &ReinConfig,
    query: &str,
    topic: Option<&str>,
    keyword: Option<&str>,
    limit: usize,
    time_from: Option<chrono::DateTime<chrono::Utc>>,
    time_to: Option<chrono::DateTime<chrono::Utc>>,
    expand_override: Option<bool>,
    fast: bool,
) -> ReinResult<Vec<RecallResult>> {
    recall_temporal_with_request_id(
        store,
        config,
        query,
        topic,
        keyword,
        limit,
        time_from,
        time_to,
        expand_override,
        fast,
        None,
    )
}

pub fn recall_temporal_with_request_id(
    store: &SqliteStore,
    config: &ReinConfig,
    query: &str,
    topic: Option<&str>,
    keyword: Option<&str>,
    limit: usize,
    time_from: Option<chrono::DateTime<chrono::Utc>>,
    time_to: Option<chrono::DateTime<chrono::Utc>>,
    expand_override: Option<bool>,
    fast: bool,
    request_id: Option<String>,
) -> ReinResult<Vec<RecallResult>> {
    let _span = tracing::info_span!("recall", query_len = query.len()).entered();
    let total_start = std::time::Instant::now();

    // === Query classification (FT-3: autonomous retrieval routing) ===
    let strategy = crate::search::classify::classify(query, time_from.is_some(), time_to.is_some());
    tracing::debug!(query_type = %strategy.query_type, "query classified");

    // Auto-inject temporal bounds for temporal queries
    let (time_from, time_to) =
        if strategy.force_temporal && time_from.is_none() && time_to.is_none() {
            if let Some(days) = strategy.temporal_days_back {
                let from = chrono::Utc::now() - chrono::Duration::days(days);
                (Some(from), Some(chrono::Utc::now()))
            } else {
                (time_from, time_to)
            }
        } else {
            (time_from, time_to)
        };

    // Apply limit multiplier from strategy
    let effective_limit = (limit as f32 * strategy.limit_multiplier) as usize;

    // === Early-launch: Supermemory search (200-500ms network I/O) ===
    // Skip in fast mode (proxy/hook_prompt) — store-local only.
    let sm_enabled = !fast && config.sync.supermemory_enabled;
    let am_enabled = !fast && config.sync.auto_memory_enabled;
    let sm_api_key = config.sync.api_key.clone();
    let sm_endpoint = config.sync.endpoint.clone();
    let q_sm = query.to_string();
    let sm_handle = if sm_enabled {
        sm_api_key.map(|api_key| {
            std::thread::spawn(move || {
                let client = SupermemoryClient::new(api_key, sm_endpoint);
                // Reuse existing tokio runtime if available, else create a temporary one
                let result = if let Ok(handle) = tokio::runtime::Handle::try_current() {
                    std::thread::scope(|_| handle.block_on(client.search(&q_sm, effective_limit)))
                } else {
                    let rt = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .ok();
                    rt.map(|rt| rt.block_on(client.search(&q_sm, effective_limit)))
                        .unwrap_or_default()
                };
                result
            })
        })
    } else {
        None
    };

    // === Phase 1a: FTS search with ORIGINAL query (fast, ~1ms) ===
    let (mut fts_results, mut fts_scores, strong_signal) = if strategy.skip_fts {
        (
            vec![],
            std::collections::HashMap::<String, f32>::new(),
            false,
        )
    } else {
        let fts_start = std::time::Instant::now();
        let (results, ranked) = try_tantivy_then_fts5(store, query, topic, effective_limit * 2)?;
        let scores: std::collections::HashMap<String, f32> = ranked.into_iter().collect();
        let ranked_vec: Vec<(String, f32)> = scores.iter().map(|(k, v)| (k.clone(), *v)).collect();
        let ss = crate::search::rerank_llm::detect_strong_signal_with_thresholds(
            &ranked_vec,
            config.search.strong_signal_ratio,
            config.search.strong_signal_single,
        );
        tracing::debug!(
            elapsed_ms = fts_start.elapsed().as_millis() as u64,
            hits = scores.len(),
            "fts search (original)"
        );
        if ss {
            tracing::info!("strong BM25 signal — will skip expansion + LLM reranker");
        }
        (results, scores, ss)
    };

    // === Query expansion: launch AFTER strong signal check to avoid unnecessary LLM calls ===
    let should_expand = if fast {
        false // Fast mode: no expansion
    } else {
        match expand_override {
            Some(true) => true,
            Some(false) => false,
            None => {
                !strong_signal
                    && strategy.query_type != crate::search::classify::QueryType::ExactKeyword
            }
        }
    };
    let adaptive_max = match strategy.query_type {
        crate::search::classify::QueryType::Temporal => Some(1),
        crate::search::classify::QueryType::Episodic => Some(2),
        _ => None,
    };
    let expand_config = config.clone();
    let expand_query_str = query.to_string();
    let cancel_expand = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let cancel_expand_clone = cancel_expand.clone();
    let expand_handle = if should_expand {
        Some(std::thread::spawn(move || {
            if cancel_expand_clone.load(std::sync::atomic::Ordering::Relaxed) {
                return vec![];
            }
            // Checks the cancel flag at pre-LLM and post-LLM boundaries so the
            // strong-signal bypass can short-circuit the expansion thread even
            // if it started between `spawn` and that bypass decision.
            crate::search::expand::expand_query_cancellable(
                &expand_config,
                &expand_query_str,
                adaptive_max,
                Some(&cancel_expand_clone),
            )
        }))
    } else {
        None
    };

    // === Phase 1b: Vec + KG search with ORIGINAL query ===
    //
    // Speculative execution for Vec search:
    //   - Cache HIT (any mode) → sync HNSW search on the main thread (~5ms, no extra connection)
    //   - Cache MISS + fast mode → skip (no remote API calls in fast mode)
    //   - Cache MISS + no strong FTS signal → background thread does API call while KG runs
    //   - Cache MISS + strong FTS signal → skip embedding API (FTS already dominant)
    //   - skip_vec strategy → skip entirely

    // Step 1: check embedding cache on the main thread (fast, no I/O).
    // Done even in fast mode so cached embeddings get HNSW search without any API call.
    let maybe_cached_emb: Option<Vec<f32>> = if !strategy.skip_vec {
        let model = config.embedding_model();
        crate::embed::EmbedCache::get(store.conn(), query, &model)
            .ok()
            .flatten()
    } else {
        None
    };

    // Step 2: launch Vec search — sync for cache hit, background thread for miss
    enum VecSearchState {
        Skip,
        Sync(Vec<(String, f32)>),
        Thread(std::thread::JoinHandle<Vec<(String, f32)>>),
    }
    let vec_state: VecSearchState = if strategy.skip_vec {
        VecSearchState::Skip
    } else if let Some(emb) = maybe_cached_emb {
        // Cache hit: synchronous HNSW search (no new connection, no API call)
        let vec_start = std::time::Instant::now();
        let results = vec_search_direct(store, &emb, topic, effective_limit, Some(config));
        tracing::debug!(
            elapsed_ms = vec_start.elapsed().as_millis() as u64,
            hits = results.len(),
            "vector search (cache hit, sync)"
        );
        VecSearchState::Sync(results)
    } else if !fast && !strong_signal {
        // Cache miss + normal mode + no strong FTS signal: background thread overlaps with KG.
        //
        // v0.22 P1 pool path: if the store has a pool attached AND we are
        // executing inside a Tokio runtime, check out a conn from the pool
        // instead of opening a fresh `SqliteStore::new(db_path)`. This
        // elides the per-channel schema init (~1-2ms) and the embedding-
        // model check. Falls back to the pre-v0.22 `SqliteStore::new` path
        // when either condition is not met, so existing callers without a
        // pool keep working exactly as before (serial fallback per I4).
        let vec_db_path = store.db_path().to_path_buf();
        let vec_config = config.clone();
        let vec_query_str = query.to_string();
        let vec_topic_str = topic.map(|s| s.to_string());
        let vec_model = config.embedding_model();
        let vec_dims = config.embedding.dimensions;
        let vec_limit = effective_limit;
        let vec_pool = store.pool().cloned();
        VecSearchState::Thread(std::thread::spawn(move || {
            // Pool path: non-blocking `try_get` — on saturation fall
            // through to the pre-v0.22 fresh-conn path rather than
            // queueing on the semaphore and eating the per-channel
            // budget (spec I1: pool checkout must not degrade recall
            // semantics under load).
            if let Some(pool) = vec_pool.as_ref() {
                if let Some(guard) = pool.try_get() {
                    let (conn, detached) = guard.detach();
                    let s = SqliteStore::from_conn(conn, vec_db_path.clone(), vec_dims);
                    let result = try_vector_search(
                        &s,
                        &vec_config,
                        &vec_query_str,
                        vec_topic_str.as_deref(),
                        vec_limit,
                    );
                    let conn_back = s.into_conn();
                    detached.put_back(conn_back);
                    return result;
                }
                tracing::debug!("vec channel pool saturated; falling back to SqliteStore::new");
            }
            // Fallback: the pre-v0.22 path. Unchanged behavior.
            let Ok(s) = SqliteStore::new(&vec_db_path, &vec_model, vec_dims) else {
                return vec![];
            };
            try_vector_search(
                &s,
                &vec_config,
                &vec_query_str,
                vec_topic_str.as_deref(),
                vec_limit,
            )
        }))
    } else {
        // Fast mode (cache miss) or strong FTS signal: skip remote API call
        if !fast {
            tracing::debug!("strong FTS signal + embedding cache miss — skipping vec API search");
        }
        VecSearchState::Skip
    };

    // Step 3: KG search.
    // - fast mode: synchronous on main thread using existing store (no extra connection cost)
    // - normal mode: background thread parallel with Vec; timeout budget measured from spawn
    enum KgState {
        Sync(
            std::collections::HashMap<String, f32>,
            std::collections::HashMap<String, f32>,
        ),
        Thread(
            std::time::Instant,
            std::sync::mpsc::Receiver<(
                std::collections::HashMap<String, f32>,
                std::collections::HashMap<String, f32>,
            )>,
        ),
    }

    fn run_kg_search(
        s: &SqliteStore,
        query: &str,
        effective_limit: usize,
        is_episodic: bool,
        time_from: Option<chrono::DateTime<chrono::Utc>>,
        time_to: Option<chrono::DateTime<chrono::Utc>>,
    ) -> (
        std::collections::HashMap<String, f32>,
        std::collections::HashMap<String, f32>,
    ) {
        let mut kg_scores: std::collections::HashMap<String, f32> =
            std::collections::HashMap::new();
        let seed_concepts = s.search_all_concepts(query, 5).unwrap_or_default();
        let concept_results =
            crate::search::kg_search::search_concepts_ranked_from(&seed_concepts, effective_limit);
        let seed_ids: Vec<String> = seed_concepts.iter().map(|c| c.id.clone()).collect();
        let bfs_expanded = if !seed_ids.is_empty() {
            crate::search::kg_search::bfs_expand_memories_by_id(s, &seed_ids, 2, effective_limit)
        } else {
            vec![]
        };
        for (id, score) in concept_results.into_iter().chain(bfs_expanded) {
            let entry = kg_scores.entry(id).or_default();
            *entry = entry.max(score);
        }
        let episode_scores = if is_episodic || time_from.is_some() || time_to.is_some() {
            collect_episode_memory_scores(s, query, effective_limit, time_from, time_to)
        } else {
            std::collections::HashMap::new()
        };
        (kg_scores, episode_scores)
    }

    let kg_is_episodic = strategy.query_type == crate::search::classify::QueryType::Episodic;
    let is_memory_db = store.db_path().to_str() == Some(":memory:");
    let kg_state: KgState = if fast || is_memory_db {
        // Fast mode or in-memory DB: run synchronously on main thread.
        // In-memory DBs cannot be shared across threads (each new connection gets an empty DB).
        let kg_start = std::time::Instant::now();
        let (kg_scores, episode_scores) = run_kg_search(
            store,
            query,
            effective_limit,
            kg_is_episodic,
            time_from,
            time_to,
        );
        tracing::debug!(
            elapsed_ms = kg_start.elapsed().as_millis() as u64,
            kg_hits = kg_scores.len(),
            "kg search (sync)"
        );
        KgState::Sync(kg_scores, episode_scores)
    } else {
        // Normal mode: background thread, budget measured from spawn time.
        //
        // v0.22 P1 pool path: mirror the Vec-channel treatment above —
        // when a pool is attached AND we are inside a Tokio runtime,
        // checkout a conn instead of opening a fresh SqliteStore. Elides
        // schema-init + embedding-model check per channel per recall.
        // Fallback to pre-v0.22 SqliteStore::new path otherwise.
        let kg_db_path = store.db_path().to_path_buf();
        let kg_query_str = query.to_string();
        let kg_model = config.embedding_model();
        let kg_dims = config.embedding.dimensions;
        let kg_effective_limit = effective_limit;
        let kg_time_from = time_from;
        let kg_time_to = time_to;
        let kg_pool = store.pool().cloned();
        let (kg_tx, kg_rx) = std::sync::mpsc::channel();
        let kg_spawn_time = std::time::Instant::now();
        std::thread::spawn(move || {
            let empty = || {
                (
                    std::collections::HashMap::<String, f32>::new(),
                    std::collections::HashMap::<String, f32>::new(),
                )
            };
            // Pool path (non-blocking — see Vec channel for rationale).
            if let Some(pool) = kg_pool.as_ref() {
                if let Some(guard) = pool.try_get() {
                    let (conn, detached) = guard.detach();
                    let s = SqliteStore::from_conn(conn, kg_db_path.clone(), kg_dims);
                    let result = run_kg_search(
                        &s,
                        &kg_query_str,
                        kg_effective_limit,
                        kg_is_episodic,
                        kg_time_from,
                        kg_time_to,
                    );
                    let conn_back = s.into_conn();
                    detached.put_back(conn_back);
                    let _ = kg_tx.send(result);
                    return;
                }
                tracing::debug!("kg channel pool saturated; falling back to SqliteStore::new");
            }
            // Fallback path (unchanged pre-v0.22 behavior).
            let Ok(s) = SqliteStore::new(&kg_db_path, &kg_model, kg_dims) else {
                let _ = kg_tx.send(empty());
                return;
            };
            let result = run_kg_search(
                &s,
                &kg_query_str,
                kg_effective_limit,
                kg_is_episodic,
                kg_time_from,
                kg_time_to,
            );
            let _ = kg_tx.send(result);
        });
        KgState::Thread(kg_spawn_time, kg_rx)
    };

    // Step 4: collect Vec results (join background thread if it was launched)
    let mut vec_scores: std::collections::HashMap<String, f32> = match vec_state {
        VecSearchState::Skip => std::collections::HashMap::new(),
        VecSearchState::Sync(results) => results.into_iter().collect(),
        VecSearchState::Thread(h) => {
            let join_start = std::time::Instant::now();
            let results = h.join().unwrap_or_else(|e| {
                tracing::error!(?e, "vector search thread panicked");
                vec![]
            });
            tracing::debug!(
                elapsed_ms = join_start.elapsed().as_millis() as u64,
                hits = results.len(),
                "vector search join (cache-miss path, overlapped with KG)"
            );
            results.into_iter().collect()
        }
    };

    // Step 5: join KG results.
    // Budget is computed from spawn time so the timeout is meaningful even if Step 4 was slow.
    let (mut kg_scores, mut episode_scores) = match kg_state {
        KgState::Sync(kg, ep) => (kg, ep),
        KgState::Thread(spawn_time, rx) => {
            let elapsed = spawn_time.elapsed();
            let budget = std::time::Duration::from_millis(80).saturating_sub(elapsed);
            rx.recv_timeout(budget).unwrap_or_else(|_| {
                tracing::warn!(
                    kg_elapsed_ms = elapsed.as_millis() as u64,
                    budget_ms = budget.as_millis() as u64,
                    "KG search budget exhausted (time measured from spawn), using empty results"
                );
                (
                    std::collections::HashMap::new(),
                    std::collections::HashMap::new(),
                )
            })
        }
    };
    tracing::debug!(
        kg_hits = kg_scores.len(),
        ep_hits = episode_scores.len(),
        "kg search joined (parallel with Vec)"
    );

    // === Phase 2: Join expansion thread, search with expanded queries, merge ===
    // Skip entirely if strong signal detected (BM25 top1 is dominant, expansion won't help)
    let expanded_queries = if strong_signal {
        tracing::info!("strong signal — skipping expanded query searches");
        // Signal the expansion thread to skip the LLM API call if it hasn't started yet.
        cancel_expand.store(true, std::sync::atomic::Ordering::Relaxed);
        drop(expand_handle);
        vec![]
    } else {
        expand_handle
            .and_then(|h| h.join().ok())
            .unwrap_or_default()
    };
    // Filter out expanded queries too similar to original OR to each other
    // (Jaccard word overlap > 0.8). Previously the filter was only vs. the
    // original, so two identical LLM variants would both survive and each
    // trigger a fresh embedding batch — B5 #21.
    let deduped_queries: Vec<&String> = {
        let mut kept: Vec<&String> = Vec::new();
        for eq in &expanded_queries {
            if word_jaccard(query, eq) > 0.8 {
                continue;
            }
            let too_similar_to_sibling = kept.iter().any(|prev| word_jaccard(prev, eq) > 0.8);
            if !too_similar_to_sibling {
                kept.push(eq);
            }
        }
        kept
    };
    if deduped_queries.len() < expanded_queries.len() {
        tracing::debug!(
            before = expanded_queries.len(),
            after = deduped_queries.len(),
            "filtered similar expanded queries"
        );
    }
    if !deduped_queries.is_empty() {
        tracing::debug!(
            count = deduped_queries.len(),
            "merging expanded query results"
        );

        // FTS: per-query (Tantivy is local, already fast)
        for eq in &deduped_queries {
            if !strategy.skip_fts {
                if let Ok((results, ranked)) =
                    try_tantivy_then_fts5(store, eq, topic, effective_limit * 2)
                {
                    for (id, score) in ranked {
                        let entry = fts_scores.entry(id).or_insert(f32::MIN);
                        *entry = entry.max(score);
                    }
                    for m in results {
                        if !fts_results.iter().any(|r: &Memory| r.id == m.id) {
                            fts_results.push(m);
                        }
                    }
                }
            }
        }

        // Vec: BATCH embed all expanded queries in one API call
        if !strategy.skip_vec {
            let eq_strs: Vec<&str> = deduped_queries.iter().map(|s| s.as_str()).collect();
            let batch_results =
                try_vector_search_batch(store, config, &eq_strs, topic, effective_limit);
            for (id, score) in batch_results {
                let entry = vec_scores.entry(id).or_insert(f32::MIN);
                *entry = entry.max(score);
            }
        }

        // KG: per expanded query.
        // In-memory DBs cannot be opened from a new thread (each gets its own empty DB),
        // so we run sequentially on the main thread in that case.
        // For file-backed DBs with 2-3 expanded queries, parallel threads cut latency from
        // O(n * kg_time) to O(kg_time).
        if is_memory_db {
            for eq in &deduped_queries {
                let (kg, ep) = run_kg_search(
                    store,
                    eq,
                    effective_limit,
                    kg_is_episodic,
                    time_from,
                    time_to,
                );
                for (id, score) in kg {
                    let entry = kg_scores.entry(id).or_default();
                    *entry = entry.max(score);
                }
                for (id, score) in ep {
                    let entry = episode_scores.entry(id).or_default();
                    *entry = entry.max(score);
                }
            }
        } else {
            // v0.22 P1 pool path (same rationale as Vec / normal-mode KG above).
            let shared_pool = store.pool().cloned();
            let kg_handles: Vec<_> = deduped_queries
                .iter()
                .map(|eq| {
                    let eq_str = (*eq).clone();
                    let db_path = store.db_path().to_path_buf();
                    let model = config.embedding_model();
                    let dims = config.embedding.dimensions;
                    let limit = effective_limit;
                    let is_ep = kg_is_episodic;
                    let t_from = time_from;
                    let t_to = time_to;
                    let pool = shared_pool.clone();
                    std::thread::spawn(move || {
                        let empty = || {
                            (
                                std::collections::HashMap::<String, f32>::new(),
                                std::collections::HashMap::<String, f32>::new(),
                            )
                        };
                        if let Some(pool) = pool.as_ref() {
                            if let Some(guard) = pool.try_get() {
                                let (conn, detached) = guard.detach();
                                let s = SqliteStore::from_conn(conn, db_path.clone(), dims);
                                let result = run_kg_search(&s, &eq_str, limit, is_ep, t_from, t_to);
                                let conn_back = s.into_conn();
                                detached.put_back(conn_back);
                                return result;
                            }
                            tracing::debug!(
                                "expanded kg pool saturated; falling back to SqliteStore::new"
                            );
                        }
                        let Ok(s) = SqliteStore::new(&db_path, &model, dims) else {
                            return empty();
                        };
                        run_kg_search(&s, &eq_str, limit, is_ep, t_from, t_to)
                    })
                })
                .collect();
            for handle in kg_handles {
                let (kg, ep) = handle.join().unwrap_or_else(|e| {
                    tracing::error!(?e, "Phase 2 KG thread panicked");
                    (
                        std::collections::HashMap::new(),
                        std::collections::HashMap::new(),
                    )
                });
                for (id, score) in kg {
                    let entry = kg_scores.entry(id).or_default();
                    *entry = entry.max(score);
                }
                for (id, score) in ep {
                    let entry = episode_scores.entry(id).or_default();
                    *entry = entry.max(score);
                }
            }
        }
    }

    // Convert to ranked vecs for fusion — MUST sort by descending score
    // because RRF uses list position (rank), not score value.
    let mut fts_ranked: Vec<(String, f32)> = fts_scores.into_iter().collect();
    fts_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut vec_ranked: Vec<(String, f32)> = vec_scores.into_iter().collect();
    vec_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    // Batch-fetch topics for KG and episode results to avoid N+1 queries
    let kg_episode_ids: Vec<String> = if topic.is_some() {
        kg_scores
            .iter()
            .chain(episode_scores.iter())
            .map(|(id, _)| id.clone())
            .collect()
    } else {
        vec![]
    };
    let kg_episode_topic_map = batch_topic_map(store, &kg_episode_ids);
    let mut kg_ranked: Vec<(String, f32)> = kg_scores
        .into_iter()
        .filter(|(id, _)| matches_topic_from_map(&kg_episode_topic_map, id, topic))
        .collect();
    kg_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    kg_ranked.truncate(effective_limit);
    let mut episode_ranked: Vec<(String, f32)> = episode_scores
        .into_iter()
        .filter(|(id, _)| matches_topic_from_map(&kg_episode_topic_map, id, topic))
        .collect();
    episode_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    episode_ranked.truncate(effective_limit);

    let use_kg = !kg_ranked.is_empty();
    let use_episode = !episode_ranked.is_empty();

    // === Path quality gating (Wang 2025: weakest-link phenomenon) ===
    // Skip a path if it returned no results — avoids empty/broken paths degrading fusion.
    // Note: scores are rank-encoded (0, -1, -2...) so we gate on result count, not score value.
    let use_fts = !fts_ranked.is_empty();
    let use_vec = !vec_ranked.is_empty();

    // Collect vector-only IDs before moving vec_ranked into fusion
    let vec_ids: Vec<String> = if use_vec {
        vec_ranked.iter().map(|(id, _)| id.clone()).collect()
    } else {
        vec![]
    };
    let episode_ids: Vec<String> = if use_episode {
        episode_ranked.iter().map(|(id, _)| id.clone()).collect()
    } else {
        vec![]
    };

    // === Score fusion (RRF or Convex Combination) ===
    // Only include paths that passed quality gating
    let fts_for_fusion = if use_fts { fts_ranked } else { vec![] };
    let vec_for_fusion = if use_vec { vec_ranked } else { vec![] };

    // === Adaptive alpha (M2): read from AdaptiveState if available ===
    // Use the dominant cluster from vector-search top candidate as a proxy for the
    // current query's semantic neighborhood, enabling per-cluster alpha lookup.
    //
    // v0.28.7+ audit M-8 R4 P2 #1 — read `cluster_id` from
    // `adaptive_state_snapshot.memory_clusters` (the SAME atomic
    // source as `cluster_version`), NOT from the `memories.cluster_id`
    // SQL column. M4 reclustering writes the SQL column FIRST and
    // saves the snapshot's incremented `cluster_version` LAST; the
    // window between is a mixed-source race (new SQL cluster_id, old
    // snapshot cluster_version) that would make learn-time treat
    // every recall in the window as stale-versioned and drop them
    // back to the derived-bucket fallback — defeating M-8's
    // alignment exactly during live reclustering. Reading both
    // cluster_id and cluster_version from the same snapshot blob
    // guarantees they belong to the same atomic epoch.
    //
    // SQL-column fallback retained for adaptive-disabled deployments
    // (snapshot is None there) and for the rare path where the
    // top-vec-hit memory isn't yet in the snapshot's
    // `memory_clusters` map (e.g., a memory just inserted but not
    // yet swept into the cluster index).
    let adaptive_state_snapshot = if config.adaptive.enabled {
        crate::store::adaptive::AdaptiveState::restore_snapshot(store.conn())
    } else {
        None
    };
    // R10 P2 (2026-05-04): split the cluster-id resolution so the
    // event payload only stamps `query_cluster_id_at_recall` when the
    // id came from `adaptive_state_snapshot.memory_clusters`. The
    // event ALSO logs `cluster_version_at_recall` from the same
    // snapshot, so a SQL-fallback id (read from `memories.cluster_id`
    // when the snapshot's `memory_clusters` map doesn't yet cover the
    // top-vec hit — e.g., a freshly-inserted memory) is NOT atomic
    // with that version. Pre-R10 the event recorded the SQL id
    // alongside the snapshot's version, and learn-time's
    // `top_vec_hit_cluster` saw a version match (snapshot version
    // equals the current cluster_version) and HONORED the SQL id —
    // potentially bucketing scoped weights under a stale or
    // reassigned cluster label. Read-time alpha / shadow-fusion
    // lookups still consult the SQL fallback (best-effort bucket
    // for live serving), but the event payload's recorded id falls
    // back to None when the snapshot-source path didn't fire,
    // forcing learn-time to re-derive the bucket from the candidate
    // payload — which IS the post-recluster truth a fresh read would
    // see.
    let query_cluster_id_from_snapshot: Option<u32> = vec_for_fusion.first().and_then(|(id, _)| {
        adaptive_state_snapshot
            .as_ref()
            .and_then(|s| s.memory_clusters.get(id).copied())
    });
    let query_cluster_id: Option<u32> = query_cluster_id_from_snapshot.or_else(|| {
        vec_for_fusion.first().and_then(|(id, _)| {
            store
                .conn()
                .query_row(
                    "SELECT cluster_id FROM memories WHERE id = ?1",
                    rusqlite::params![id],
                    |row| row.get::<_, Option<u32>>(0),
                )
                .ok()
                .flatten()
        })
    });
    // R13 P2 (2026-05-04): capture the top-vec memory id BEFORE
    // `vec_for_fusion` is moved into the fusion `lists` collection
    // below. Stamped into the recall_complete event payload for
    // learn-time's memory-id-remap bucket-resolution path. One String
    // clone per recall — negligible cost.
    let query_top_vec_memory_id_at_recall: Option<String> =
        vec_for_fusion.first().map(|(id, _)| id.clone());
    let query_type_label = format!("{}", strategy.query_type);
    let adaptive_alpha = adaptive_state_snapshot
        .as_ref()
        .and_then(|s| s.get_alpha(&query_type_label, query_cluster_id));
    // v0.28.7+ audit M-6: capture `runtime_adoption_weight` alongside the
    // shadow weights so `apply_ars_dynamic_fusion_scores` can outer-blend
    // simplex against legacy by the same scalar that the inner trust blend
    // uses. Default 0.0 falls cleanly into the "no canary" branch.
    let (ars_dynamic_fusion_weights, ars_runtime_adoption_weight) = adaptive_state_snapshot
        .as_ref()
        .map(|s| {
            let runtime_adoption_weight =
                crate::ops::ars_tuning::parameter_policy_recall_fusion_runtime_adoption_weight(
                    store.conn(),
                    config,
                    s,
                    &query_type_label,
                    query_cluster_id,
                );
            let weights = ready_shadow_fusion_weights_for_recall(
                s,
                config,
                &query_type_label,
                query_cluster_id,
                runtime_adoption_weight,
            );
            (weights, runtime_adoption_weight)
        })
        .unwrap_or((None, 0.0));
    let ars_dynamic_fusion_active = ars_dynamic_fusion_weights.is_some();

    // Capture per-channel scores for reranking and M2 logging.
    // Clamp negatives (rank sentinels like -1,-2) to positive via 1/(1+|rank|),
    // then max-normalize positive scores to [0,1] so CC fusion channels are comparable.
    //
    // Bug #O1 fix: use `is_sign_negative()` instead of `*s < 0.0`. In IEEE 754
    // `-0.0 < 0.0` is FALSE (negative-zero compares equal to positive-zero),
    // so when the FTS channel emits `-0.0` for the top rank the sentinel
    // conversion was being skipped — the score stayed at -0.0, then `fts_max`
    // (folded with a 0.0 seed) kept a positive max, and the first-place row
    // was never bumped to its proper rank-derived `1.0 / (1.0 + 0) = 1.0`.
    let fts_norm_log: std::collections::HashMap<String, f32> = fts_for_fusion
        .iter()
        .map(|(id, s)| {
            // Convert negative rank sentinels (-1,-2,...) to positive rank scores: 1/(1+|rank|)
            let score = if s.is_sign_negative() {
                1.0 / (1.0 + s.abs())
            } else {
                *s
            };
            (id.clone(), score)
        })
        .collect();
    let fts_max = fts_norm_log.values().copied().fold(0.0f32, f32::max);
    let fts_norm_log: std::collections::HashMap<String, f32> =
        if fts_max.is_finite() && fts_max > 1.0 {
            fts_norm_log
                .into_iter()
                .map(|(id, s)| (id, s / fts_max))
                .collect()
        } else {
            fts_norm_log
        };
    let vec_norm_log: std::collections::HashMap<String, f32> = vec_for_fusion
        .iter()
        .map(|(id, s)| {
            // Bug #O1 fix: see fts_norm_log above — `*s < 0.0` misses `-0.0`.
            let score = if s.is_sign_negative() {
                1.0 / (1.0 + s.abs())
            } else {
                *s
            };
            (id.clone(), score)
        })
        .collect();
    let vec_max = vec_norm_log.values().copied().fold(0.0f32, f32::max);
    let vec_norm_log: std::collections::HashMap<String, f32> =
        if vec_max.is_finite() && vec_max > 1.0 {
            vec_norm_log
                .into_iter()
                .map(|(id, s)| (id, s / vec_max))
                .collect()
        } else {
            vec_norm_log
        };
    // Max-normalize KG/episode channels to match the [0,1] scale of fts/vec after
    // CC normalization. Before v0.21.0 these were raw clones, so the 0.5/0.65
    // boost weights below were not scale-comparable to the CC output and could
    // dominate or vanish depending on the raw magnitude of the upstream scorers.
    let kg_norm_log: std::collections::HashMap<String, f32> = {
        let map: std::collections::HashMap<String, f32> = kg_ranked.iter().cloned().collect();
        let max = map.values().copied().fold(0.0f32, f32::max);
        if max.is_finite() && max > 1.0 {
            map.into_iter().map(|(id, s)| (id, s / max)).collect()
        } else {
            map
        }
    };
    let episode_norm_log: std::collections::HashMap<String, f32> = {
        let map: std::collections::HashMap<String, f32> = episode_ranked.iter().cloned().collect();
        let max = map.values().copied().fold(0.0f32, f32::max);
        if max.is_finite() && max > 1.0 {
            map.into_iter().map(|(id, s)| (id, s / max)).collect()
        } else {
            map
        }
    };

    let fused = if config.search.fusion_method == "cc" {
        let alpha = adaptive_alpha
            .or(strategy.cc_alpha)
            .unwrap_or(config.search.cc_alpha as f32);
        // Run CC normalization on clean vec/fts channels first, then boost with KG/episode
        let mut fused =
            crate::search::rrf::convex_combination(&fts_for_fusion, &vec_for_fusion, alpha);
        // Post-fusion KG/episode boost.
        //
        // Invariant: `alpha` controls BM25 weight vs. every other channel. The
        // KG and episode channels are "every other channel", so their
        // contribution must vary with alpha to honor route intent
        // (e.g. ExactKeyword sets alpha=0.85 specifically to suppress non-BM25
        // channels; an unscaled +0.5 KG-only boost used to override that).
        //
        // We use `2 * (1 - alpha)` — NOT `(1 - alpha)` alone — so the
        // balanced routes (Episodic/Exploratory alpha=0.5) still see the
        // full base weight: a naive `(1-alpha)` halves the boost at
        // alpha=0.5 and silently regresses episodic recall ranking. The
        // `2*` factor anchors the scaling at the pre-existing calibrated
        // point:
        //   alpha=0.5 (Episodic, Exploratory) → factor 1.0 (unchanged)
        //   alpha=0.85 (ExactKeyword)         → factor 0.30 (suppressed)
        //   alpha=0.70 (Temporal)             → factor 0.60
        //   alpha=0.40 (Preference)           → factor 1.20
        //   alpha=0.30 (Semantic)             → factor 1.40
        // Base weights (0.5 for KG, 0.65 for episode) were calibrated at
        // alpha=0.5 and still apply there today.
        if use_kg || use_episode {
            let non_bm25_factor = (2.0 * (1.0 - alpha)).max(0.0);
            let kg_weight = 0.5 * non_bm25_factor;
            let episode_weight = 0.65 * non_bm25_factor;
            for (id, kg_score) in kg_norm_log.iter() {
                if let Some(pos) = fused.iter().position(|(fid, _)| fid == id) {
                    fused[pos].1 += *kg_score * kg_weight;
                } else {
                    fused.push((id.clone(), *kg_score * kg_weight));
                }
            }
            for (id, episode_score) in episode_norm_log.iter() {
                if let Some(pos) = fused.iter().position(|(fid, _)| fid == id) {
                    fused[pos].1 += *episode_score * episode_weight;
                } else {
                    fused.push((id.clone(), *episode_score * episode_weight));
                }
            }
            // Re-sort after boost so boosted items are not stuck at the tail.
            fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }
        fused
    } else {
        let rrf_k = config.search.rrf_k as f32;
        // Map strategy alpha to RRF weights (alpha=high → FTS dominant)
        let (fts_weight, vec_weight) = if let Some(alpha) = strategy.cc_alpha {
            (alpha, 1.0 - alpha)
        } else {
            (
                config.search.rrf_fts_weight as f32,
                config.search.rrf_vec_weight as f32,
            )
        };
        let fts_empty = fts_for_fusion.is_empty();
        let vec_empty = vec_for_fusion.is_empty();
        let kg_empty = kg_ranked.is_empty();
        let ep_empty = episode_ranked.is_empty();
        let mut lists = Vec::new();
        if !fts_empty {
            lists.push((fts_for_fusion, fts_weight));
        }
        if !vec_empty {
            lists.push((vec_for_fusion, vec_weight));
        }
        if !kg_empty {
            let kg_weight = 0.3; // KG is supplementary
            lists.push((kg_ranked.clone(), kg_weight));
        }
        if !ep_empty {
            let episode_weight = match strategy.query_type {
                crate::search::classify::QueryType::Episodic => 0.45,
                crate::search::classify::QueryType::Temporal => 0.30,
                _ => 0.18,
            };
            lists.push((episode_ranked.clone(), episode_weight));
        }
        let result = crate::search::rrf::reciprocal_rank_fusion(&lists, rrf_k);
        if result.is_empty() && fts_empty && vec_empty && kg_empty && ep_empty {
            // Diagnostic: every channel returned zero candidates. Emit a single
            // structured log line so silent empty recalls are observable in ops.
            tracing::info!(
                query = %query,
                "recall: all channels returned zero candidates"
            );
        }
        result
    };

    // Build memory lookup from already-fetched results
    let mut memory_map: std::collections::HashMap<String, Memory> =
        std::collections::HashMap::new();
    for m in fts_results {
        memory_map.entry(m.id.clone()).or_insert(m);
    }

    // Batch-fetch vector-search memories not already in FTS results (avoids N+1 queries)
    let missing_ids: Vec<String> = vec_ids
        .iter()
        .filter(|id| !memory_map.contains_key(*id))
        .cloned()
        .collect();
    if !missing_ids.is_empty() {
        for m in store.get_batch(&missing_ids) {
            memory_map.entry(m.id.clone()).or_insert(m);
        }
    }

    // Batch-fetch KG-sourced memories not already in map
    let kg_ids: Vec<String> = kg_norm_log
        .keys()
        .filter(|id| !memory_map.contains_key(*id))
        .cloned()
        .collect();
    if !kg_ids.is_empty() {
        for m in store.get_batch(&kg_ids) {
            memory_map.entry(m.id.clone()).or_insert(m);
        }
    }
    let episode_missing_ids: Vec<String> = episode_ids
        .into_iter()
        .filter(|id| !memory_map.contains_key(id))
        .collect();
    if !episode_missing_ids.is_empty() {
        for m in store.get_batch(&episode_missing_ids) {
            memory_map.entry(m.id.clone()).or_insert(m);
        }
    }

    // Bug #2 (HIGH, v0.26.2): centralized "deprecated" filter on the fully-
    // assembled `memory_map`. Belt-and-suspenders behind the SQL filters in
    // `store::fts::search_fts` and `store::vec::search_vec` — required because
    // not every channel goes through SQL:
    //   - The Tantivy BM25 path in `try_tantivy_then_fts5` queries an
    //     external index that doesn't auto-prune when `apply_evolution`
    //     raw-SQL flips `status='deprecated'` (Agent C's Bug #3 in v0.26.2);
    //     until the next side-index refresh, the dead row's text still
    //     matches and `store.get(id)` happily returns the deprecated row.
    //   - The KG/episode channels resolve memory IDs through
    //     `bfs_expand_memories_by_id` / `episode.memory_ids` and `store.get`
    //     — neither of which filters by status.
    //
    // v0.26.2 R2 Codex F3: drop ONLY `Deprecated` (terminal dead rows). DO
    // NOT drop superseded rows (`superseded_by IS NOT NULL` with status
    // `Active`) — `collapse_results_to_canonicals` later maps them to the
    // live canonical successor under the canonical-first read model. Pre-R2
    // we filtered both, which silently lost queries that matched only the
    // old/evidence text whose canonical is still live.
    let live_filter_before = memory_map.len();
    memory_map.retain(|_id, m| {
        matches!(
            m.status,
            crate::types::MemoryStatus::Active | crate::types::MemoryStatus::Updated
        )
    });
    let live_filter_dropped = live_filter_before - memory_map.len();
    if live_filter_dropped > 0 {
        tracing::debug!(
            dropped = live_filter_dropped,
            kept = memory_map.len(),
            "recall: live-status filter excluded deprecated rows from memory_map"
        );
    }

    // v0.26.2 R2 Codex finding F1 (recall side): also prune `fused` so
    // dead-row IDs don't consume the `take(limit * 2)` budget below. Without
    // this, when stale Tantivy/KG/episode hits put the top-2N fused IDs all
    // on deprecated rows, the take() picks them all and `memory_map.remove`
    // silently no-ops, leaving live lower-ranked candidates behind →
    // recall returns empty despite valid matches existing further down.
    let fused: Vec<(String, f32)> = fused
        .into_iter()
        .filter(|(id, _)| memory_map.contains_key(id))
        .collect();
    let fused = apply_ars_dynamic_fusion_scores(
        fused,
        &fts_norm_log,
        &vec_norm_log,
        &kg_norm_log,
        &episode_norm_log,
        &memory_map,
        ars_dynamic_fusion_weights,
        ars_runtime_adoption_weight,
    );

    // Apply strength weighting (Ebbinghaus or KM survival curve) + temporal filter
    // Load cached per-cluster survival curves from M3 (if available)
    let mut survival_cache: std::collections::HashMap<u32, crate::search::survival::SurvivalCurve> =
        std::collections::HashMap::new();
    let mut global_prior: Option<crate::search::survival::SurvivalCurve> = None;
    if config.adaptive.enabled {
        if let Ok(mut stmt) = store
            .conn()
            .prepare("SELECT key, value FROM metadata WHERE key LIKE 'survival_curve:%'")
        {
            let _ = stmt
                .query_map([], |row| {
                    let key: String = row.get(0)?;
                    let json: String = row.get(1)?;
                    Ok((key, json))
                })
                .ok()
                .map(|rows| {
                    for row in rows.flatten() {
                        if let Some(id_str) = row.0.strip_prefix("survival_curve:") {
                            if id_str == "global" {
                                // M3 cold-start: global prior for clusters without enough data
                                if let Ok(curve) = serde_json::from_str(&row.1) {
                                    global_prior = Some(curve);
                                }
                            } else if let (Ok(cid), Ok(curve)) =
                                (id_str.parse::<u32>(), serde_json::from_str(&row.1))
                            {
                                survival_cache.insert(cid, curve);
                            }
                        }
                    }
                });
        }
    }

    // === M5: Tier filtering — exclude Cold memories unless Exploratory query ===
    //
    // Applied **before** the take/limit so Cold memories don't consume top-N
    // slots. The previous implementation retained() after take(limit * 2),
    // which meant any Cold memories surfacing in the top 2N (possible under
    // KG/episode boosts) truncated the result set instead of being replaced
    // by Warm/Hot candidates that would otherwise make the cut.
    let include_cold = strategy.query_type == crate::search::classify::QueryType::Exploratory;
    let mut cold_filtered: usize = 0;
    let fused: Vec<(String, f32)> = if include_cold {
        fused
    } else {
        fused
            .into_iter()
            .filter(|(id, _)| match memory_map.get(id) {
                Some(mem) => {
                    let is_cold = mem.tier == crate::store::tiering::MemoryTier::Cold;
                    if is_cold {
                        cold_filtered += 1;
                    }
                    !is_cold
                }
                // Unknown IDs (shouldn't happen in practice) pass through so
                // the downstream loop's memory_map.remove can no-op cleanly.
                None => true,
            })
            .collect()
    };
    if cold_filtered > 0 {
        tracing::debug!(cold_filtered, "cold tier memories excluded pre-take");
    }

    let has_temporal = time_from.is_some() || time_to.is_some();
    let take_count = if has_temporal { usize::MAX } else { limit * 2 };
    let mut local_results: Vec<(Memory, f32)> = Vec::new();
    for (id, rrf_score) in fused.into_iter().take(take_count) {
        if let Some(memory) = memory_map.remove(&id) {
            // Temporal filter: skip memories outside the requested time range
            if let Some(from) = time_from {
                if memory.created_at < from {
                    continue;
                }
            }
            if let Some(to) = time_to {
                if memory.created_at > to {
                    continue;
                }
            }
            // Use per-cluster survival curve if available (M3), else global prior, else Ebbinghaus
            let curve = memory
                .cluster_id
                .and_then(|cid| survival_cache.get(&cid))
                .or(global_prior.as_ref());
            let final_score = crate::search::scoring::apply_strength_weighting_with_curve(
                rrf_score, &memory, curve,
            );
            local_results.push((memory, final_score));
        }
    }
    local_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // === R2: Multi-feature reranking — overwrite scores so downstream ordering uses rerank ===
    if local_results.len() > 1 {
        let weights = crate::search::rerank::load_weights(store.conn());
        let importance_weight = |imp: &crate::types::Importance| -> f32 {
            match imp {
                crate::types::Importance::Critical => 1.0,
                crate::types::Importance::High => 0.8,
                crate::types::Importance::Medium => 0.6,
                crate::types::Importance::Low => 0.4,
            }
        };
        // Pre-compute lowercased query and words once for the entire loop
        let query_lower = query.to_lowercase();
        let query_words: Vec<&str> = query_lower.split_whitespace().collect();
        for (mem, score) in local_results.iter_mut() {
            let fts = fts_norm_log.get(&mem.id).copied().unwrap_or(0.0);
            let vec = vec_norm_log.get(&mem.id).copied().unwrap_or(0.0);
            let kg = kg_norm_log.get(&mem.id).copied().unwrap_or(0.0);
            let episode = episode_norm_log.get(&mem.id).copied().unwrap_or(0.0);
            // Channel coverage: how many channels found this memory (1-4)
            let channels = (if fts > 0.0 { 1 } else { 0 })
                + (if vec > 0.0 { 1 } else { 0 })
                + (if kg > 0.0 { 1 } else { 0 })
                + (if episode > 0.0 { 1 } else { 0 });
            let channel_coverage = channels.max(1) as f32 / 4.0;
            // Topic match: does memory topic appear as a word in query?
            // Word-boundary match: multi-word topics match as phrase, single-word as exact token.
            // Avoids "sql" matching "nosql" while allowing "release process" to match.
            let topic_lower = mem.topic.to_lowercase();
            let topic_match = if topic_lower.contains(' ') {
                // Multi-word topic: check if the phrase appears in the query.
                if query_lower.contains(&topic_lower) {
                    1.0
                } else {
                    0.0
                }
            } else {
                // Single-word topic: exact word match (not substring).
                if query_words.iter().any(|w| *w == topic_lower) {
                    1.0
                } else {
                    0.0
                }
            };
            let features = crate::search::rerank::RerankFeatures {
                fts_score: fts,
                vec_score: vec,
                kg_score: kg,
                episode_score: episode,
                recency_days: (chrono::Utc::now() - mem.created_at).num_hours() as f32 / 24.0,
                access_count: mem.access_count,
                strength: mem.strength as f32,
                importance_weight: importance_weight(&mem.importance),
                keyword_overlap: crate::search::rerank::compute_keyword_overlap_with_words(
                    &query_words,
                    &mem.keywords,
                    &mem.content,
                ),
                topic_match,
                brevity: 1.0 / (1.0 + mem.content.len() as f32 / 500.0),
                channel_coverage,
                canonical_support: mem.support_count as f32 / (mem.support_count as f32 + 1.0),
                source_diversity: mem.source_diversity / (mem.source_diversity + 1.0),
                usage_recency: (chrono::Utc::now() - mem.last_accessed).num_hours() as f32 / 24.0,
                connectivity: (mem.related_ids.len().min(10) as f32) / 10.0,
                concept_richness: (mem.concept_ids.len().min(5) as f32) / 5.0,
                tier_score: match mem.tier {
                    crate::store::tiering::MemoryTier::Hot => 1.0,
                    crate::store::tiering::MemoryTier::Warm => 0.5,
                    crate::store::tiering::MemoryTier::Cold => 0.0,
                },
                is_current: if mem.superseded_by.is_none() {
                    1.0
                } else {
                    0.0
                },
                // M3: cluster-level KM survival probability at current days-since-last-access.
                // Fallback to global prior, then 0.5 (neutral) when no curve exists.
                cluster_survival: mem
                    .cluster_id
                    .and_then(|cid| survival_cache.get(&cid))
                    .or(global_prior.as_ref())
                    .map(|curve| {
                        let days =
                            (chrono::Utc::now() - mem.last_accessed).num_hours() as f64 / 24.0;
                        curve.probability_at(days) as f32
                    })
                    .unwrap_or(0.5),
            };
            *score = crate::search::rerank::rerank_score(&features, &weights);
        }
        local_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    }

    // === R2+: LLM reranker — launched in background, joined after cross-validation ===
    // Skip if: strong BM25 signal, or linear rerank already shows clear separation (top1 >> top2).
    // The thread runs concurrently with AutoMemory scan + Supermemory join + cross-validate,
    // so its latency largely overlaps with work the pipeline must do anyway.
    let linear_clear = if local_results.len() >= 2 {
        let top1 = local_results[0].1;
        let top2 = local_results[1].1;
        top2 > 0.0 && top1 / top2 >= 1.5
    } else {
        false
    };
    if linear_clear {
        tracing::debug!("linear rerank scores well-separated, skipping LLM reranker");
    }
    // Snapshot positional id list and candidates for background thread.
    // `candidate_ids` lets us rebuild an id→score map after the thread joins.
    // When `llm_reranker_timeout_ms = 0`, use legacy synchronous mode (blocks in-place).
    // v0.27.1 B2: route through `resolve_llm_for("search.llm_reranker")`
    // so `[llm]` inheritance applies to the reranker-gate decision.
    // Fail-soft: an Err from the resolver (e.g. provider known but model
    // missing) acts as `Provider::None` — disables the reranker rather
    // than aborting the request.
    let reranker_resolved_provider = config
        .resolve_llm_for("search.llm_reranker")
        .map(|r| r.provider)
        .unwrap_or(crate::config::Provider::None);
    let llm_rerank_state: Option<(Vec<String>, std::sync::mpsc::Receiver<Vec<f32>>)> = if !fast
        && !strong_signal
        && !linear_clear
        && reranker_resolved_provider != crate::config::Provider::None
        && local_results.len() > 1
    {
        let candidate_ids: Vec<String> = local_results.iter().map(|(m, _)| m.id.clone()).collect();

        if config.search.llm_reranker_timeout_ms == 0 {
            // Synchronous legacy mode: block in-place, apply scores immediately
            let llm_scores =
                crate::search::rerank_llm::rerank_with_llm(config, query, &local_results);
            for (i, (_, score)) in local_results.iter_mut().enumerate() {
                if let Some(&s) = llm_scores.get(i) {
                    *score = s;
                }
            }
            local_results
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            None // no async state needed
        } else {
            let candidates_clone: Vec<(Memory, f32)> = local_results.clone();
            let config_clone = config.clone();
            let query_clone = query.to_string();
            let (tx, rx) = std::sync::mpsc::channel();
            std::thread::spawn(move || {
                let scores = crate::search::rerank_llm::rerank_with_llm(
                    &config_clone,
                    &query_clone,
                    &candidates_clone,
                );
                let _ = tx.send(scores);
            });
            tracing::debug!("llm reranker launched in background thread");
            Some((candidate_ids, rx))
        }
    } else {
        None
    };

    // === Optional keyword filter ===
    if let Some(kw) = keyword {
        let kw_lower = kw.to_lowercase();
        local_results.retain(|(m, _)| {
            m.keywords
                .iter()
                .any(|k| k.to_lowercase().contains(&kw_lower))
                || m.content.to_lowercase().contains(&kw_lower)
        });
    }

    // === Cross-validation (if enabled) ===
    // Supermemory search was launched at pipeline start (sm_handle); join it here.
    // AutoMemory is a fast local file scan.

    // am_enabled is set at pipeline start (respects fast mode)
    let am_glob = config.sync.auto_memory_glob.clone();
    let q_am = query.to_string();

    let auto_memory_results = if am_enabled {
        let scanner = AutoMemoryScanner::new(am_glob);
        scanner.scan(&q_am)
    } else {
        vec![]
    }
    .into_iter()
    .filter(|memory| matches_external_filters(memory, topic, keyword, time_from, time_to))
    .collect::<Vec<_>>();

    // Join early-launched Supermemory thread (has been running since pipeline start)
    let supermemory_results = sm_handle
        .and_then(|h| h.join().ok())
        .unwrap_or_default()
        .into_iter()
        .filter(|memory| matches_external_filters(memory, topic, keyword, time_from, time_to))
        .collect::<Vec<_>>();
    tracing::debug!(
        elapsed_ms = total_start.elapsed().as_millis() as u64,
        hits = supermemory_results.len(),
        "supermemory search (joined)"
    );

    let validated =
        validate::cross_validate(&local_results, &supermemory_results, &auto_memory_results);

    // Build final results — scores already assigned by cross_validate
    let mut results: Vec<RecallResult> = validated
        .into_iter()
        .map(|v| RecallResult {
            memory: v.memory,
            score: v.score,
            confidence: v.confidence,
            sources_hit: v.sources_hit,
            evidence_count: 0,
            evidence_preview: vec![],
            archival_summary: None,
        })
        .collect();
    results = collapse_results_to_canonicals(store, results)?;
    apply_evidence_rerank(store, query, &mut results, 3);

    sort_recall_results(&mut results);

    // === R2+ async join: apply LLM reranker scores if they arrived within budget ===
    if let Some((candidate_ids, rx)) = llm_rerank_state {
        let timeout_ms = config.search.llm_reranker_timeout_ms;
        let elapsed = total_start.elapsed();
        let budget = std::time::Duration::from_millis(timeout_ms);
        let remaining = budget.saturating_sub(elapsed);
        match rx.recv_timeout(remaining) {
            Ok(llm_scores) => {
                tracing::info!(
                    elapsed_ms = total_start.elapsed().as_millis() as u64,
                    count = llm_scores.len(),
                    "llm reranker scores arrived (async)"
                );
                // `candidate_ids` are raw IDs from before canonical collapse.
                // After collapse, result.memory.id is the canonical ID which may differ.
                // Build canonical_id → max(llm_score) so the lookup works correctly.
                // Multiple raw IDs may share a canonical — take the max score.
                let mut llm_score_map: std::collections::HashMap<String, f32> =
                    std::collections::HashMap::new();
                for (raw_id, &s) in candidate_ids.iter().zip(llm_scores.iter()) {
                    let canonical = store
                        .canonical_id_for(raw_id)
                        .unwrap_or_else(|_| raw_id.clone());
                    let entry = llm_score_map.entry(canonical).or_insert(f32::NEG_INFINITY);
                    if s > *entry {
                        *entry = s;
                    }
                }
                // Apply to final results (sm:/auto: memories have no entry — unchanged)
                for r in &mut results {
                    if let Some(&llm_s) = llm_score_map.get(&r.memory.id) {
                        r.score = llm_s;
                    }
                }
                sort_recall_results(&mut results);
            }
            Err(_) => {
                tracing::debug!(
                    elapsed_ms = total_start.elapsed().as_millis() as u64,
                    "llm reranker exceeded budget, keeping linear scores"
                );
            }
        }
    }

    // === M1: Emit recall_complete BEFORE truncation (full candidate set for counterfactual replay) ===
    if config.adaptive.enabled {
        let request_id = request_id.unwrap_or_else(|| ulid::Ulid::new().to_string());
        let episode_matches: Vec<serde_json::Value> = episode_ranked
            .iter()
            .take(5)
            .filter_map(|(memory_id, score)| {
                store.get(memory_id).ok().map(|memory| {
                    serde_json::json!({
                        "memory_id": memory.id,
                        "memory_topic": memory.topic,
                        "memory_summary": memory.summary,
                        "episode_score": score,
                    })
                })
            })
            .collect();
        let candidates: Vec<serde_json::Value> = results
            .iter()
            .filter(|r| !r.memory.id.starts_with("sm:") && !r.memory.id.starts_with("auto:"))
            .map(|r| {
                let bm25 = fts_norm_log.get(&r.memory.id).copied().unwrap_or(0.0);
                let vec = vec_norm_log.get(&r.memory.id).copied().unwrap_or(0.0);
                let kg = kg_norm_log.get(&r.memory.id).copied().unwrap_or(0.0);
                let episode = episode_norm_log.get(&r.memory.id).copied().unwrap_or(0.0);
                serde_json::json!({
                    "id": r.memory.id,
                    "bm25_norm": bm25,
                    "vec_norm": vec,
                    "kg_norm": kg,
                    "episode_norm": episode,
                    "final_score": r.score,
                    "confidence": r.confidence,
                    "sources_hit": r.sources_hit,
                    "support_count": r.memory.support_count,
                    "source_diversity": r.memory.source_diversity,
                })
            })
            .collect();
        let alpha_used = adaptive_alpha
            .or(strategy.cc_alpha)
            .unwrap_or(config.search.cc_alpha as f32);
        let _ = crate::store::adaptive::emit_event(
            store.conn(),
            crate::store::adaptive::FeedbackEvent {
                event_type: crate::store::adaptive::EventType::RecallComplete,
                request_id: Some(request_id),
                memory_id: None,
                concept_id: None,
                query: Some(query.chars().take(200).collect()),
                query_type: Some(format!("{}", strategy.query_type)),
                topic: topic.map(|t| t.to_string()),
                payload: Some(serde_json::json!({
                "candidates": candidates,
                "episode_matches": episode_matches,
                "alpha_used": alpha_used,
                "ars_dynamic_fusion": ars_dynamic_fusion_active,
                "fusion_method": &config.search.fusion_method,
                "result_count": results.len(),
                // v0.28.7+ audit M-8 R2 P2 follow-up — log the cluster id
                // production recall actually used. R10 P2 (2026-05-04)
                // tightens this to the SNAPSHOT-SOURCED id only:
                // `query_cluster_id_from_snapshot` is `Some` exactly when
                // the top-vec hit appeared in
                // `adaptive_state_snapshot.memory_clusters`, so it's
                // atomic with `cluster_version_at_recall` below. The
                // SQL-fallback path (best-effort live read from
                // `memories.cluster_id`) still serves read-time alpha
                // selection but is NOT recorded here — feeding the SQL
                // id alongside the snapshot's version would lie to
                // learn-time, allowing scoped weights to land under a
                // stale or reassigned cluster label. Learn-time at
                // `parse_candidates_from_event` reads this field back so
                // alpha / shadow-fusion bucketing matches read-time
                // exactly, immune to candidate-set collapse / filter /
                // canonical-collapse rewrites between recall and event
                // emission. When this is None, learn-time falls back to
                // deriving the bucket from the candidate payload (the
                // M-8 R3 derived path).
                "query_cluster_id_at_recall": query_cluster_id_from_snapshot,
                // v0.28.7+ audit M-8 R3 P2 follow-up — also stamp the
                // `AdaptiveState::cluster_version` (== `state.version` here;
                // see `commit_shadow_fusion_weight_replay` and snapshot
                // CAS-merge logic where cluster_version is folded). HDBSCAN
                // cluster ids are local labels that get reassigned on M4
                // recluster; learn-time MUST drop the recorded
                // `query_cluster_id_at_recall` and fall back to the
                // current-state-derived bucket when the version no longer
                // matches.
                "cluster_version_at_recall": adaptive_state_snapshot.as_ref().map(|s| s.cluster_version),
                // v0.28.7+ audit R13 P2 (2026-05-04) — stamp the
                // top-vec-hit memory id directly. Learn-time uses this
                // as the PREFERRED bucket-resolution path: looking it
                // up against the CURRENT memory_clusters returns the
                // post-recluster truth a fresh read would also see,
                // which is correct regardless of how many M4 passes
                // have fired between recall and learn-time. The
                // `cluster_version_at_recall` guard above stays only
                // as a backward-compat hook for pre-R13 events.
                //
                // Always stamped when vec_for_fusion is non-empty,
                // regardless of whether `query_cluster_id` came from
                // the snapshot or SQL fallback (the R10 discipline
                // applied to the cluster_id field, not the memory_id —
                // memory_id remap doesn't depend on snapshot atomicity
                // because we look it up against current memory_clusters
                // at learn-time).
                "query_top_vec_memory_id_at_recall": query_top_vec_memory_id_at_recall,
                })),
            },
        );
    }

    // === MMR diversity reranking (lambda < 1.0 activates diversity pressure) ===
    // Applied after M1 emission so the full candidate set is logged for counterfactual replay.
    let mmr_lambda = config.search.mmr_lambda as f32;
    if mmr_lambda < 1.0 && results.len() > limit {
        // Build candidates from the already-sorted results (sort_recall_results applied
        // multi-key ordering: score → support_count → source_diversity → confidence).
        // This preserves tie-break determinism when MMR scores are equal.
        let candidates: Vec<(Memory, f32)> = results
            .iter()
            .map(|r| (r.memory.clone(), r.score))
            .collect();
        let selected = crate::search::mmr::apply_mmr(candidates, limit, mmr_lambda);
        // Reassemble RecallResults by id lookup (preserves confidence/sources_hit)
        let mut result_map: std::collections::HashMap<String, RecallResult> = results
            .into_iter()
            .map(|r| (r.memory.id.clone(), r))
            .collect();
        results = selected
            .into_iter()
            .filter_map(|(m, _)| result_map.remove(&m.id))
            .collect();
    } else {
        // Truncate to the caller's requested limit (not effective_limit).
        results.truncate(limit);
    }
    enrich_results_with_evidence(store, &mut results, 2);

    // v0.26 Cap C: surface `archival_summary` for cold-tier memories when
    // the operator has flipped `[ars].cold_archive_enabled = true`. Gating
    // is centralised in `maybe_archival_summary_for_recall` so the
    // tier/version invariants live next to the field definition. Pure /
    // memory-local — no DB IO.
    let cold_archive_enabled = config.ars.cold_archive_enabled;
    for result in &mut results {
        if let Some(summary) =
            maybe_archival_summary_for_recall(cold_archive_enabled, &result.memory)
        {
            result.archival_summary = Some(summary);
        }
    }

    // Record recall hit (NOT access — access should only be counted when
    // the agent/user actually uses the memory, not just when it's returned).
    let recall_ids: Vec<String> = results
        .iter()
        .filter(|r| !r.memory.id.starts_with("sm:") && !r.memory.id.starts_with("auto:"))
        .map(|r| r.memory.id.clone())
        .collect();
    store.record_recall_hit(&recall_ids);

    // Periodically update quality weights (every ~50 recalls)
    let total_recalls: u64 = store.quality_metrics().map(|(_, r, _)| r).unwrap_or(0);
    if total_recalls > 0 && total_recalls.is_multiple_of(50) {
        store.update_quality_weights();
    }

    tracing::debug!(
        elapsed_ms = total_start.elapsed().as_millis() as u64,
        results = results.len(),
        "recall complete"
    );
    Ok(results)
}

/// Try vector search: check cache first, then call API if available.
/// Returns empty vec on any failure (graceful degradation).
fn try_vector_search(
    store: &SqliteStore,
    config: &ReinConfig,
    query: &str,
    topic: Option<&str>,
    limit: usize,
) -> Vec<(String, f32)> {
    let model = config.embedding_model();

    // Level 2: Check embedding cache
    if let Ok(Some(cached)) = EmbedCache::get(store.conn(), query, &model) {
        let results = vec_search_direct(store, &cached, topic, limit, Some(config));
        if !results.is_empty() {
            return results;
        }
    }

    // Level 3: Use configured embedder (Google or OMLX)
    let embedder = match crate::embed::create_embedder(config) {
        Some(e) => e,
        None => return vec![],
    };

    let embedding = match tokio::runtime::Handle::try_current() {
        Ok(handle) => tokio::task::block_in_place(|| handle.block_on(embedder.embed(query))),
        Err(_) => {
            // No tokio runtime — create a temporary one
            match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(rt) => rt.block_on(embedder.embed(query)),
                Err(_) => return vec![],
            }
        }
    };

    match embedding {
        Ok(emb) => {
            let _ = EmbedCache::put(store.conn(), query, &model, &emb);
            vec_search_direct(store, &emb, topic, limit, Some(config))
        }
        Err(e) => {
            tracing::warn!("embedding failed, falling back to FTS-only: {e}");
            vec![]
        }
    }
}

/// Batch vector search for multiple queries. Checks cache first, then batch-embeds uncached
/// queries in a single API call. Returns merged (id, score) with max score per ID.
fn try_vector_search_batch(
    store: &SqliteStore,
    config: &ReinConfig,
    queries: &[&str],
    topic: Option<&str>,
    limit: usize,
) -> std::collections::HashMap<String, f32> {
    let model = config.embedding_model();
    let mut merged: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
    let mut uncached: Vec<(usize, &str)> = Vec::new();

    // Check cache for each query
    for (i, q) in queries.iter().enumerate() {
        if let Ok(Some(cached)) = EmbedCache::get(store.conn(), q, &model) {
            for (id, score) in vec_search_direct(store, &cached, topic, limit, Some(config)) {
                let entry = merged.entry(id).or_insert(f32::MIN);
                *entry = entry.max(score);
            }
        } else {
            uncached.push((i, q));
        }
    }

    if uncached.is_empty() {
        return merged;
    }

    // Batch embed uncached queries
    let embedder = match crate::embed::create_embedder(config) {
        Some(e) => e,
        None => return merged,
    };

    let texts: Vec<&str> = uncached.iter().map(|(_, q)| *q).collect();
    let embeddings = match tokio::runtime::Handle::try_current() {
        Ok(handle) => tokio::task::block_in_place(|| handle.block_on(embedder.embed_batch(&texts))),
        Err(_) => match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(rt) => rt.block_on(embedder.embed_batch(&texts)),
            Err(_) => return merged,
        },
    };

    match embeddings {
        Ok(embs) => {
            for (emb, (_, q)) in embs.iter().zip(uncached.iter()) {
                let _ = EmbedCache::put(store.conn(), q, &model, emb);
                for (id, score) in vec_search_direct(store, emb, topic, limit, Some(config)) {
                    let entry = merged.entry(id).or_insert(f32::MIN);
                    *entry = entry.max(score);
                }
            }
        }
        Err(e) => {
            tracing::warn!("batch embedding failed: {e}");
        }
    }

    merged
}

/// Batch-fetch topic for a set of memory IDs in a single query.
/// Returns a map from memory ID to topic string.
fn batch_topic_map(
    store: &SqliteStore,
    ids: &[String],
) -> std::collections::HashMap<String, String> {
    let mut map = std::collections::HashMap::new();
    if ids.is_empty() {
        return map;
    }
    // Process in chunks to avoid SQLite parameter limits
    for chunk in ids.chunks(500) {
        let placeholders: String = (1..=chunk.len())
            .map(|i| format!("?{}", i))
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "SELECT id, topic FROM memories WHERE id IN ({})",
            placeholders
        );
        if let Ok(mut stmt) = store.conn().prepare(&sql) {
            let params: Vec<&dyn rusqlite::ToSql> =
                chunk.iter().map(|id| id as &dyn rusqlite::ToSql).collect();
            if let Ok(rows) = stmt.query_map(params.as_slice(), |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            }) {
                for row in rows.flatten() {
                    map.insert(row.0, row.1);
                }
            }
        }
    }
    map
}

/// Check if a memory matches the requested topic filter using a pre-fetched topic map.
fn matches_topic_from_map(
    topic_map: &std::collections::HashMap<String, String>,
    id: &str,
    topic: Option<&str>,
) -> bool {
    match topic {
        None => true,
        Some(t) => topic_map
            .get(id)
            .map(|mt| {
                // Compare both raw and normalized forms so user-supplied filters
                // match even after store-time topic normalization.
                mt == t
                    || crate::ops::normalize_topic_name(mt) == crate::ops::normalize_topic_name(t)
            })
            .unwrap_or(false),
    }
}

/// Rank results by position and filter by topic.
/// Scores are converted to negative rank positions for RRF. CC fusion re-normalizes via
/// min-max, so original score magnitudes are not needed.
fn rank_and_filter(
    results: Vec<(String, f32)>,
    store: &SqliteStore,
    topic: Option<&str>,
    limit: usize,
) -> Vec<(String, f32)> {
    if topic.is_none() {
        return results
            .into_iter()
            .take(limit)
            .enumerate()
            .map(|(i, (id, _))| (id, -(i as f32)))
            .collect();
    }
    let ids: Vec<String> = results.iter().map(|(id, _)| id.clone()).collect();
    let topic_map = batch_topic_map(store, &ids);
    results
        .into_iter()
        .filter(|(id, _)| matches_topic_from_map(&topic_map, id, topic))
        .take(limit)
        .enumerate()
        .map(|(i, (id, _))| (id, -(i as f32)))
        .collect()
}

/// Direct vector search using HNSW index first, falling back to sqlite-vec.
fn vec_search_direct(
    store: &SqliteStore,
    embedding: &[f32],
    topic: Option<&str>,
    limit: usize,
    config: Option<&ReinConfig>,
) -> Vec<(String, f32)> {
    // Try HNSW first (O(log n) approximate nearest neighbor)
    let hnsw_path = store.db_path().with_extension("");
    if crate::store::hnsw::HnswIndex::is_dirty(&hnsw_path) {
        // Atomically claim the dirty marker. Only one concurrent caller wins;
        // others fall through to sqlite-vec immediately (no duplicate rebuilds).
        if crate::store::hnsw::HnswIndex::take_dirty_for_rebuild(&hnsw_path) {
            tracing::info!(
                "hnsw index dirty — spawning background rebuild, using sqlite-vec for this request"
            );
            let rebuild_path = hnsw_path.clone();
            let rebuild_cfg = config.cloned().unwrap_or_else(|| {
                let mut fb = crate::config::ReinConfig::load().unwrap_or_default();
                fb.embedding.dimensions = embedding.len();
                fb
            });
            let db_path = store.db_path().to_path_buf();
            let model = rebuild_cfg.embedding_model();
            let dims = rebuild_cfg.embedding.dimensions;
            std::thread::spawn(move || {
                let rebuilt = if let Ok(s) = SqliteStore::new(&db_path, &model, dims) {
                    crate::search::warmup::populate_hnsw(&s, &rebuild_cfg)
                } else {
                    false
                };
                if rebuilt {
                    // Success: clear `.rebuilding` — index is ready
                    crate::store::hnsw::HnswIndex::clear_rebuilding(&rebuild_path);
                } else {
                    // Failed or skipped: restore `.dirty` so a future request retries
                    let dirty = crate::store::hnsw::HnswIndex::dirty_marker_path(&rebuild_path);
                    let rebuilding =
                        crate::store::hnsw::HnswIndex::rebuilding_marker_path(&rebuild_path);
                    if rebuilding.exists() {
                        let _ = std::fs::rename(&rebuilding, &dirty);
                    }
                }
            });
        } else {
            tracing::debug!("hnsw rebuild already in progress, using sqlite-vec for this request");
        }
        // Fall through to sqlite-vec fallback immediately — do not block.
        // Bug #O2: pass `topic` into the SQL so the topic filter happens on the
        // ANN scan (the in-SQL over-fetch lives in `search_vec` itself). The
        // outer `rank_and_filter` is still load-bearing for rank-encoding to
        // negative positions used by RRF; its topic post-filter is a redundant
        // belt-and-suspenders pass after the SQL filter and a no-op when SQL
        // already filtered correctly.
        return match crate::store::vec::search_vec(store.conn(), embedding, topic, limit) {
            Ok(results) => rank_and_filter(results, store, topic, limit),
            Err(_) => vec![],
        };
    }
    if let Ok(index) = crate::store::hnsw::HnswIndex::open(&hnsw_path, embedding.len()) {
        if !index.is_empty() {
            if let Ok(results) = index.search(embedding, limit * 2) {
                let filtered = rank_and_filter(results, store, topic, limit);
                if !filtered.is_empty() {
                    return filtered;
                }
            }
        }
    }

    // Fall back to sqlite-vec (brute-force O(n)).
    // Bug #O2: same fix as the early-return fallback above — push `topic` into
    // the SQL so we don't lose every top-k ANN hit when none happen to match.
    match crate::store::vec::search_vec(store.conn(), embedding, topic, limit) {
        Ok(results) => rank_and_filter(results, store, topic, limit),
        Err(_) => vec![],
    }
}

/// Try Tantivy BM25 search first, fall back to FTS5.
/// Returns (memories, ranked_ids) for use in the recall pipeline.
fn try_tantivy_then_fts5(
    store: &SqliteStore,
    query: &str,
    topic: Option<&str>,
    limit: usize,
) -> ReinResult<(Vec<Memory>, Vec<(String, f32)>)> {
    let mut memories = Vec::new();
    let mut ranked = Vec::new();
    let mut seen_ids = std::collections::HashSet::new();

    let db_path = store.db_path();
    if db_path.to_str() != Some(":memory:") {
        let dirty_path = crate::search::warmup::tantivy_dirty_path(db_path);
        if dirty_path.exists() {
            tracing::info!("tantivy index marked dirty, rebuilding before search");
            crate::search::warmup::populate_tantivy(store);
        }
        let tantivy_path = db_path.with_extension("tantivy");
        if let Ok(tantivy) = crate::store::tantivy_fts::TantivyFts::open(&tantivy_path) {
            if let Ok(results) = tantivy.search(query, topic, limit) {
                for (i, (id, score)) in results.into_iter().enumerate() {
                    if let Ok(m) = store.get(&id) {
                        let mem_id = m.id.clone();
                        if seen_ids.insert(mem_id.clone()) {
                            // Preserve original score for CC fusion; use rank for RRF
                            // Score is Tantivy BM25 relevance (positive float)
                            ranked.push((mem_id, if score > 0.0 { score } else { -(i as f32) }));
                            memories.push(m);
                        }
                    }
                }
            }
        }
    }

    // Always run FTS5 too. It complements Tantivy when side-index updates were skipped
    // or when the Tantivy index is stale.
    let fts_results = store.search_fts(query, topic, limit)?;
    for (i, m) in fts_results.into_iter().enumerate() {
        if seen_ids.insert(m.id.clone()) {
            ranked.push((m.id.clone(), -(i as f32)));
            memories.push(m);
        }
    }

    if !memories.is_empty() {
        tracing::debug!(hits = memories.len(), "tantivy+fts search");
    }
    Ok((memories, ranked))
}

/// Jaccard similarity for query dedup.
/// Uses word-level for space-separated text, falls back to character bigrams for CJK.
fn word_jaccard(a: &str, b: &str) -> f32 {
    let wa: std::collections::HashSet<&str> = a.split_whitespace().collect();
    let wb: std::collections::HashSet<&str> = b.split_whitespace().collect();
    // If both queries have ≤1 whitespace token (likely CJK), use character bigrams
    if wa.len() <= 1 && wb.len() <= 1 {
        let ca: std::collections::HashSet<(char, char)> =
            a.chars().zip(a.chars().skip(1)).collect();
        let cb: std::collections::HashSet<(char, char)> =
            b.chars().zip(b.chars().skip(1)).collect();
        let inter = ca.intersection(&cb).count() as f32;
        let union = ca.union(&cb).count() as f32;
        return if union == 0.0 { 1.0 } else { inter / union };
    }
    let inter = wa.intersection(&wb).count() as f32;
    let union = wa.union(&wb).count() as f32;
    if union == 0.0 {
        1.0
    } else {
        inter / union
    }
}

/// Search episodes and project them back to linked memory IDs.
/// This gives episodic queries a real session-level retrieval path instead of
/// only changing routing parameters.
fn collect_episode_memory_scores(
    store: &SqliteStore,
    query: &str,
    limit: usize,
    time_from: Option<chrono::DateTime<chrono::Utc>>,
    time_to: Option<chrono::DateTime<chrono::Utc>>,
) -> std::collections::HashMap<String, f32> {
    let mut memory_scores = std::collections::HashMap::new();
    let episodes = store
        .search_episodes_ranked(query, limit, time_from, time_to)
        .unwrap_or_default();

    for (episode, base_score) in episodes {
        for mem_id in &episode.memory_ids {
            let entry = memory_scores.entry(mem_id.clone()).or_insert(0.0_f32);
            *entry = entry.max(base_score);
            if let Ok(memory) = store.get(mem_id) {
                for related_id in &memory.related_ids {
                    let related = memory_scores.entry(related_id.clone()).or_insert(0.0_f32);
                    *related = related.max(base_score * 0.65);
                }
            }
        }
        for concept_id in &episode.concept_ids {
            if let Ok(Some(concept)) = store.get_concept_by_id(concept_id) {
                for mem_id in &concept.source_memory_ids {
                    let entry = memory_scores.entry(mem_id.clone()).or_insert(0.0_f32);
                    *entry = entry.max(base_score * 0.85);
                }
            }
        }
    }

    memory_scores
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Importance, MemoryLayer, MemoryStatus, MemoryTier, Source};
    use chrono::Utc;

    fn test_memory(id: &str, support_count: u32, source_diversity: f32) -> Memory {
        Memory {
            id: id.to_string(),
            layer: MemoryLayer::LTM,
            topic: "docker".to_string(),
            summary: format!("summary {id}"),
            content: format!("content {id}"),
            keywords: vec![],
            importance: Importance::High,
            source: Source::Manual,
            strength: 0.8,
            decay_lambda: 0.02,
            access_count: 0,
            superseded_by: None,
            canonical_id: Some(id.to_string()),
            support_count,
            merge_count: support_count.saturating_sub(1),
            dedup_confidence: 0.9,
            source_diversity,
            contradiction_score: 0.0,
            related_ids: vec![],
            concept_ids: vec![],
            status: MemoryStatus::Active,
            embedding: None,
            tier: MemoryTier::Warm,
            cluster_id: None,
            archival_summary: None,
            archival_summary_at: None,
            archival_summary_version: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_accessed: Utc::now(),
        }
    }

    #[test]
    fn sort_recall_results_prefers_stronger_canonical_support_on_ties() {
        let mut results = vec![
            RecallResult {
                memory: test_memory("low", 1, 1.0),
                score: 0.8,
                confidence: 0.7,
                sources_hit: 1,
                evidence_count: 0,
                evidence_preview: vec![],
                archival_summary: None,
            },
            RecallResult {
                memory: test_memory("high", 4, 2.0),
                score: 0.8,
                confidence: 0.7,
                sources_hit: 1,
                evidence_count: 0,
                evidence_preview: vec![],
                archival_summary: None,
            },
        ];

        sort_recall_results(&mut results);
        assert_eq!(results[0].memory.id, "high");
    }

    #[test]
    fn ars_dynamic_fusion_default_off_preserves_legacy_scores_and_order() {
        let fused = vec![
            ("legacy-top".to_string(), 0.9),
            ("legacy-low".to_string(), 0.1),
        ];
        let memory_map = std::collections::HashMap::from([
            ("legacy-top".to_string(), test_memory("legacy-top", 1, 1.0)),
            ("legacy-low".to_string(), test_memory("legacy-low", 9, 9.0)),
        ]);

        let actual = apply_ars_dynamic_fusion_scores(
            fused.clone(),
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
            &memory_map,
            None,
            // adoption_weight is irrelevant when weights=None — the early
            // return preserves `fused` regardless. Use 1.0 to make any
            // future regression that drops the early-return obvious.
            1.0,
        );

        assert_eq!(actual, fused);
    }

    #[test]
    fn ars_dynamic_fusion_canary_uses_support_and_diversity_dimensions() {
        let fused = vec![
            ("bm25-only".to_string(), 0.9),
            ("supported".to_string(), 0.1),
        ];
        let fts_norm_log = std::collections::HashMap::from([("bm25-only".to_string(), 1.0)]);
        let memory_map = std::collections::HashMap::from([
            ("bm25-only".to_string(), test_memory("bm25-only", 0, 0.0)),
            ("supported".to_string(), test_memory("supported", 9, 9.0)),
        ]);
        let weights = crate::search::alpha_optimizer::ShadowFusionWeights {
            bm25: 0.0,
            vec: 0.0,
            kg: 0.0,
            episode: 0.0,
            support: 0.6,
            diversity: 0.4,
        };

        let actual = apply_ars_dynamic_fusion_scores(
            fused,
            &fts_norm_log,
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
            &memory_map,
            Some(weights),
            // Full canary (adoption_weight=1.0) reproduces the pre-M-6
            // wholesale-simplex behavior this test was originally written
            // against; the M-6 outer blend is a no-op at this extreme.
            1.0,
        );

        assert_eq!(actual[0].0, "supported");
        assert!(actual[0].1 > actual[1].1);
    }

    /// v0.28.7+ audit M-6 — the audit-named regression vector. With
    /// `weights=Some` AND `runtime_adoption_weight=0`, the outer blend
    /// must collapse to pure legacy so route-specific signal
    /// (canonically: ExactKeyword's `alpha=0.85` BM25-heavy fusion) is
    /// preserved bit-for-bit during a barely-promoted-or-rolled-back
    /// canary. Pre-fix the function unconditionally replaced
    /// `legacy_score` with the simplex sum the moment `weights` was Some,
    /// nuking the route alpha. The two assertions below would both have
    /// failed against the pre-fix behavior — `actual` would have ranked
    /// "support-heavy" first and rewritten the BM25-leader's score.
    #[test]
    fn ars_dynamic_fusion_zero_adoption_preserves_route_specific_legacy_scores() {
        // Mimic an ExactKeyword route: BM25 strongly favors "bm25-leader",
        // legacy fused score reflects that.
        let fused = vec![
            ("bm25-leader".to_string(), 0.95),
            ("support-heavy".to_string(), 0.10),
        ];
        let fts_norm_log = std::collections::HashMap::from([
            ("bm25-leader".to_string(), 1.0),
            ("support-heavy".to_string(), 0.05),
        ]);
        let memory_map = std::collections::HashMap::from([
            (
                "bm25-leader".to_string(),
                test_memory("bm25-leader", 0, 0.0),
            ),
            // High support_count + diversity would let a support-weighted
            // simplex re-rank this above bm25-leader if the outer blend
            // were absent.
            (
                "support-heavy".to_string(),
                test_memory("support-heavy", 9, 9.0),
            ),
        ]);
        // Simplex weights chosen to dominate via support+diversity (the
        // pre-M-6 reproduction).
        let weights = crate::search::alpha_optimizer::ShadowFusionWeights {
            bm25: 0.0,
            vec: 0.0,
            kg: 0.0,
            episode: 0.0,
            support: 0.6,
            diversity: 0.4,
        };

        let actual_zero = apply_ars_dynamic_fusion_scores(
            fused.clone(),
            &fts_norm_log,
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
            &memory_map,
            Some(weights),
            0.0,
        );

        assert_eq!(
            actual_zero, fused,
            "adoption_weight=0 must reproduce legacy fused output exactly \
             (pre-M-6 wholesale-replace would have rewritten both scores \
             AND reordered the pair to put 'support-heavy' first, which \
             is the recall-quality regression the audit named)"
        );

        // Mid-canary (adoption=0.5) sanity: each output score is exactly
        // halfway between legacy and the simplex it would have been at
        // adoption=1.0. Asserts the blend is a true linear interpolation,
        // not a step or threshold.
        let actual_full = apply_ars_dynamic_fusion_scores(
            fused.clone(),
            &fts_norm_log,
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
            &memory_map,
            Some(weights),
            1.0,
        );
        let actual_half = apply_ars_dynamic_fusion_scores(
            fused.clone(),
            &fts_norm_log,
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
            &memory_map,
            Some(weights),
            0.5,
        );
        // For each id, find legacy + full + half and assert
        // half ≈ 0.5 * legacy + 0.5 * full.
        // (Outputs may be reordered by score, so look up by id.)
        for (id, legacy_score) in &fused {
            let full = actual_full
                .iter()
                .find(|(other, _)| other == id)
                .map(|(_, s)| *s)
                .expect("full canary preserves all ids");
            let half = actual_half
                .iter()
                .find(|(other, _)| other == id)
                .map(|(_, s)| *s)
                .expect("half canary preserves all ids");
            let expected = 0.5 * legacy_score + 0.5 * full;
            let diff = (half - expected).abs();
            assert!(
                diff < 1e-6,
                "linear blend failed for id={id}: legacy={legacy_score} full={full} \
                 half={half} expected={expected} (diff={diff})"
            );
        }
    }

    #[test]
    fn ars_dynamic_fusion_resolver_is_shadow_only_by_default() {
        let mut config = ReinConfig::default();
        config.ars.acceleration.enabled = true;
        config.ars.acceleration.shadow_only = true;
        config.adaptive.min_samples_alpha = 1;
        let mut state = crate::store::adaptive::AdaptiveState::default();
        state.learned_shadow_fusion.insert(
            "semantic".into(),
            crate::store::adaptive::LearnedShadowFusionEntry {
                weights: crate::store::adaptive::ShadowFusionWeightEntry {
                    bm25: 0.1,
                    vec: 0.2,
                    kg: 0.3,
                    episode: 0.1,
                    support: 0.2,
                    diversity: 0.1,
                },
                sample_count: 12,
                last_updated: "2026-04-30T00:00:00Z".into(),
            },
        );

        assert!(
            ready_shadow_fusion_weights_for_recall(&state, &config, "semantic", None, 1.0)
                .is_none()
        );
    }

    #[test]
    fn ars_dynamic_fusion_resolver_requires_parameter_policy_canary() {
        let mut config = ReinConfig::default();
        config.ars.acceleration.enabled = true;
        config.ars.acceleration.shadow_only = false;
        config.adaptive.min_samples_alpha = 1;
        let mut state = crate::store::adaptive::AdaptiveState::default();
        state.learned_shadow_fusion.insert(
            "semantic".into(),
            crate::store::adaptive::LearnedShadowFusionEntry {
                weights: crate::store::adaptive::ShadowFusionWeightEntry {
                    bm25: 0.1,
                    vec: 0.2,
                    kg: 0.3,
                    episode: 0.1,
                    support: 0.2,
                    diversity: 0.1,
                },
                sample_count: 12,
                last_updated: "2026-04-30T00:00:00Z".into(),
            },
        );

        assert!(
            ready_shadow_fusion_weights_for_recall(&state, &config, "semantic", None, 0.0)
                .is_none()
        );
    }

    #[test]
    fn ars_dynamic_fusion_resolver_returns_effective_weights_with_policy_canary() {
        let mut config = ReinConfig::default();
        config.ars.acceleration.enabled = true;
        config.ars.acceleration.shadow_only = false;
        config.adaptive.min_samples_alpha = 1;
        let mut state = crate::store::adaptive::AdaptiveState::default();
        state.learned_shadow_fusion.insert(
            "semantic".into(),
            crate::store::adaptive::LearnedShadowFusionEntry {
                weights: crate::store::adaptive::ShadowFusionWeightEntry {
                    bm25: 2.0,
                    vec: 2.0,
                    kg: 2.0,
                    episode: 2.0,
                    support: 1.0,
                    diversity: 1.0,
                },
                sample_count: 12,
                last_updated: "2026-04-30T00:00:00Z".into(),
            },
        );

        let weights =
            ready_shadow_fusion_weights_for_recall(&state, &config, "semantic", None, 1.0)
                .expect("non-shadow mode should expose eligible snapshot weights");
        assert!((weights.sum() - 1.0).abs() < 1e-9);
        assert!(weights.bm25 > weights.support);
    }

    #[test]
    fn external_filters_require_matching_topic_and_keyword() {
        let mut memory = test_memory("filtered", 1, 1.0);
        memory.topic = "rust".to_string();
        memory.keywords = vec!["borrow".to_string()];
        memory.content = "borrow checker details".to_string();

        assert!(matches_external_filters(
            &memory,
            Some("rust"),
            Some("borrow"),
            None,
            None,
        ));
        assert!(!matches_external_filters(
            &memory,
            Some("python"),
            Some("borrow"),
            None,
            None,
        ));
        assert!(!matches_external_filters(
            &memory,
            Some("rust"),
            Some("decorator"),
            None,
            None,
        ));
        assert!(!matches_external_filters(
            &memory,
            Some("rust"),
            Some("borrow"),
            Some(memory.created_at + chrono::Duration::days(1)),
            None,
        ));
    }

    #[test]
    fn evidence_rerank_boosts_supported_memory() {
        let store = SqliteStore::in_memory().unwrap();
        let supported_id = store.store(test_memory("supported", 3, 2.0)).unwrap();
        let unsupported_id = store.store(test_memory("unsupported", 1, 1.0)).unwrap();

        let supported = store.get(&supported_id).unwrap();
        store
            .snapshot_memory_as_evidence(
                &supported_id,
                &Memory {
                    id: "ev1".to_string(),
                    content: "database connection pool tuning".to_string(),
                    summary: "connection pool tuning".to_string(),
                    created_at: supported.created_at,
                    updated_at: supported.updated_at,
                    last_accessed: supported.last_accessed,
                    ..supported.clone()
                },
            )
            .unwrap();

        let mut results = vec![
            RecallResult {
                memory: store.get(&supported_id).unwrap(),
                score: 0.5,
                confidence: 0.7,
                sources_hit: 1,
                evidence_count: 0,
                evidence_preview: vec![],
                archival_summary: None,
            },
            RecallResult {
                memory: store.get(&unsupported_id).unwrap(),
                score: 0.52,
                confidence: 0.7,
                sources_hit: 1,
                evidence_count: 0,
                evidence_preview: vec![],
                archival_summary: None,
            },
        ];

        apply_evidence_rerank(&store, "connection pool", &mut results, 3);
        sort_recall_results(&mut results);
        assert_eq!(results[0].memory.id, supported_id);
    }

    // ── v0.26 Cap C: archival_summary surfacing ─────────────────────────

    /// Helper: build a memory with the desired tier + archival fields.
    /// All other fields use the same defaults as `test_memory` so the
    /// gate-only behavior stays the focus.
    fn cold_archive_memory(
        id: &str,
        tier: MemoryTier,
        summary: Option<&str>,
        version: Option<u32>,
    ) -> Memory {
        let mut m = test_memory(id, 1, 1.0);
        m.tier = tier;
        m.archival_summary = summary.map(|s| s.to_string());
        m.archival_summary_at = if summary.is_some() { Some(0) } else { None };
        m.archival_summary_version = version;
        m
    }

    /// Gate path 1 (canonical happy path): `cold_archive_enabled = true` +
    /// tier=Cold + summary present + version current → surface the summary.
    #[test]
    fn maybe_archival_summary_returns_summary_for_enabled_cold_current_version() {
        let memory = cold_archive_memory(
            "cold-1",
            MemoryTier::Cold,
            Some("condensed view"),
            Some(crate::ops::cold_archive_summary::ARCHIVAL_SUMMARY_VERSION),
        );
        let summary = maybe_archival_summary_for_recall(true, &memory);
        assert_eq!(summary.as_deref(), Some("condensed view"));
    }

    /// Gate path 2: feature off → always None even when the row has data.
    #[test]
    fn maybe_archival_summary_returns_none_when_feature_disabled() {
        let memory = cold_archive_memory(
            "cold-2",
            MemoryTier::Cold,
            Some("condensed view"),
            Some(crate::ops::cold_archive_summary::ARCHIVAL_SUMMARY_VERSION),
        );
        assert!(maybe_archival_summary_for_recall(false, &memory).is_none());
    }

    /// Gate path 3: tier != Cold → None even when the column has a current
    /// summary (Hot/Warm memories deliberately do NOT surface the
    /// condensed view per contract §2.6).
    #[test]
    fn maybe_archival_summary_returns_none_for_non_cold_tier() {
        for tier in [MemoryTier::Hot, MemoryTier::Warm] {
            let memory = cold_archive_memory(
                "warm-3",
                tier,
                Some("condensed view"),
                Some(crate::ops::cold_archive_summary::ARCHIVAL_SUMMARY_VERSION),
            );
            assert!(
                maybe_archival_summary_for_recall(true, &memory).is_none(),
                "tier {tier:?} must not surface archival_summary"
            );
        }
    }

    /// Gate path 4: stale version → None (worker will regenerate). Without
    /// this we'd serve a summary written under an old prompt / contract.
    #[test]
    fn maybe_archival_summary_returns_none_for_stale_version() {
        let stale_version =
            crate::ops::cold_archive_summary::ARCHIVAL_SUMMARY_VERSION.saturating_sub(1);
        let memory = cold_archive_memory(
            "cold-stale",
            MemoryTier::Cold,
            Some("old prompt output"),
            Some(stale_version),
        );
        assert!(maybe_archival_summary_for_recall(true, &memory).is_none());
    }

    /// Gate path 5: column null → None.
    #[test]
    fn maybe_archival_summary_returns_none_when_column_null() {
        let memory = cold_archive_memory("cold-null", MemoryTier::Cold, None, None);
        assert!(maybe_archival_summary_for_recall(true, &memory).is_none());
    }

    /// `RecallResult` JSON roundtrip with `archival_summary = Some`.
    /// Mirrors the GUI consumer contract.
    #[test]
    fn recall_result_serde_roundtrip_with_archival_summary() {
        let memory = test_memory("rr-1", 1, 1.0);
        let r = RecallResult {
            memory,
            score: 0.7,
            confidence: 0.9,
            sources_hit: 2,
            evidence_count: 1,
            evidence_preview: vec!["[ev] preview".to_string()],
            archival_summary: Some("compact view".to_string()),
        };
        let json = serde_json::to_string(&r).unwrap();
        assert!(
            json.contains("archival_summary"),
            "field MUST appear in wire format when populated; got {json}"
        );
        assert!(json.contains("compact view"));
    }

    /// Bug #2 (HIGH, v0.26.2) + R2 Codex F3: the centralized retain in
    /// `recall_temporal_with_request_id` must drop `Deprecated` rows
    /// (terminal dead from `apply_evolution`) but MUST keep superseded rows
    /// (`superseded_by IS NOT NULL`, status still `Active`) so
    /// `collapse_results_to_canonicals` can map them to the live canonical
    /// successor under the canonical-first read model. Dropping superseded
    /// here would silently lose queries that match only old/evidence text.
    #[test]
    fn recall_memory_map_live_filter_drops_deprecated_keeps_superseded() {
        // Mirror of the production retain — keep these in sync.
        let predicate = |m: &Memory| -> bool {
            matches!(m.status, MemoryStatus::Active | MemoryStatus::Updated)
        };

        let mut active = test_memory("active", 1, 1.0);
        active.status = MemoryStatus::Active;
        active.superseded_by = None;

        let mut updated = test_memory("updated", 1, 1.0);
        updated.status = MemoryStatus::Updated;
        updated.superseded_by = None;

        let mut deprecated = test_memory("deprecated", 1, 1.0);
        deprecated.status = MemoryStatus::Deprecated;
        deprecated.superseded_by = None;

        // The mark_superseded shape: superseded_by set, status still Active.
        // R2 F3: this row MUST pass the retain so collapse can map it.
        let mut superseded = test_memory("superseded", 1, 1.0);
        superseded.status = MemoryStatus::Active;
        superseded.superseded_by = Some("active".to_string());

        assert!(predicate(&active), "Active + superseded_by=None must pass");
        assert!(
            predicate(&updated),
            "Updated + superseded_by=None must pass"
        );
        assert!(!predicate(&deprecated), "Deprecated must be dropped");
        assert!(
            predicate(&superseded),
            "superseded row MUST pass — collapse_results_to_canonicals \
             maps it to the live canonical successor (R2 Codex F3)"
        );
    }

    /// Bug #O1 (v0.26.2): the rank-sentinel-to-positive-score normalization
    /// loop in `recall_temporal_with_request_id` (`fts_norm_log` /
    /// `vec_norm_log`) used `*s < 0.0` to detect the sentinel. In IEEE 754
    /// `-0.0 < 0.0` is **false**, so a `-0.0` sentinel (which the FTS path
    /// CAN emit when the top hit's rank position is 0) was passed through
    /// unchanged. `f32::max` then kept the positive max, max-normalization
    /// did nothing, and the first-place row stayed at -0.0 instead of being
    /// promoted to its proper rank score `1.0 / (1.0 + 0) = 1.0`.
    ///
    /// This test re-implements the production closure inline so we can
    /// assert the post-fix behavior without reaching into the giant
    /// `recall_temporal_with_request_id` body. The closure must stay in
    /// sync with the production code — see fts_norm_log/vec_norm_log.
    #[test]
    fn rank_sentinel_normalization_handles_negative_zero() {
        let normalize = |s: f32| -> f32 {
            // Mirrors the production fix: `is_sign_negative()` catches both
            // `-1.0` AND `-0.0` (the latter is what the original `< 0.0`
            // missed).
            if s.is_sign_negative() {
                1.0 / (1.0 + s.abs())
            } else {
                s
            }
        };

        // The bug case: -0.0 sentinel from rank position 0.
        let neg_zero: f32 = -0.0;
        assert!(
            neg_zero.is_sign_negative(),
            "fixture sanity: -0.0 must report as sign-negative"
        );
        // Spelled-out via partial_cmp to avoid clippy::neg_cmp_op_on_partial_ord;
        // the underlying invariant is the IEEE 754 trap: -0.0 < 0.0 is FALSE
        // (negative-zero compares Equal to positive-zero), which is exactly
        // what the original `*s < 0.0` check fell into.
        assert_eq!(
            neg_zero.partial_cmp(&0.0_f32),
            Some(std::cmp::Ordering::Equal),
            "fixture sanity: -0.0 partial-compares Equal to 0.0 (the IEEE 754 trap)"
        );
        assert_eq!(
            normalize(neg_zero),
            1.0,
            "post-fix: -0.0 sentinel must normalize to 1/(1+0) = 1.0"
        );

        // Other ranks still work: -1.0 → 0.5, -2.0 → 0.333...
        assert_eq!(normalize(-1.0), 0.5);
        assert!((normalize(-2.0) - (1.0 / 3.0)).abs() < 1e-6);

        // Positive scores pass through unchanged.
        assert_eq!(normalize(0.7), 0.7);
        assert_eq!(normalize(2.5), 2.5);

        // Positive zero passes through (not a sentinel — this is a real
        // zero score from a regular channel).
        let pos_zero: f32 = 0.0;
        assert_eq!(normalize(pos_zero), 0.0);
    }

    /// `archival_summary = None` is omitted from the wire format
    /// (`skip_serializing_if = "Option::is_none"`) so old GUI builds that
    /// don't know the field stay bit-identical.
    #[test]
    fn recall_result_serde_omits_archival_summary_when_none() {
        let memory = test_memory("rr-2", 1, 1.0);
        let r = RecallResult {
            memory,
            score: 0.7,
            confidence: 0.9,
            sources_hit: 2,
            evidence_count: 0,
            evidence_preview: vec![],
            archival_summary: None,
        };
        let json = serde_json::to_string(&r).unwrap();
        assert!(
            !json.contains("archival_summary"),
            "None field MUST be elided; got {json}"
        );
    }
}
