//! Unified recall service used by both CLI and MCP.
//! Implements the full pipeline: waterfall search + cross-validation.

use crate::config::ReinConfig;
use crate::embed::EmbedCache;
use crate::store::SqliteStore;
use crate::types::Embedder as _;
use crate::sync::{auto_memory::AutoMemoryScanner, supermemory::SupermemoryClient, validate};
use crate::types::{Memory, MemoryStore, ReinResult};

/// A recalled memory with score and confidence.
pub struct RecallResult {
    pub memory: Memory,
    pub score: f32,
    pub confidence: f32,
    pub sources_hit: usize,
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
    recall_temporal(store, config, query, topic, keyword, limit, None, None, None)
}

/// Full recall pipeline with optional temporal filtering.
/// `expand_override`: Some(true) forces expansion, Some(false) disables, None uses config.
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
) -> ReinResult<Vec<RecallResult>> {
    let _span = tracing::info_span!("recall", query_len = query.len()).entered();
    let total_start = std::time::Instant::now();

    // === Query classification (FT-3: autonomous retrieval routing) ===
    let strategy = crate::search::classify::classify(query, time_from.is_some(), time_to.is_some());
    tracing::debug!(query_type = %strategy.query_type, "query classified");

    // Auto-inject temporal bounds for temporal queries
    let (time_from, time_to) = if strategy.force_temporal && time_from.is_none() && time_to.is_none() {
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
    // Start this immediately — it runs in parallel with everything else until we join it.
    let sm_enabled = config.sync.supermemory_enabled;
    let sm_api_key = config.sync.api_key.clone();
    let sm_endpoint = config.sync.endpoint.clone();
    let q_sm = query.to_string();
    let sm_handle = if sm_enabled {
        sm_api_key.map(|api_key| std::thread::spawn(move || {
            let client = SupermemoryClient::new(api_key, sm_endpoint);
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .ok();
            rt.map(|rt| rt.block_on(client.search(&q_sm, limit)))
                .unwrap_or_default()
        }))
    } else {
        None
    };

    // === Phase 1a: FTS search with ORIGINAL query (fast, ~1ms) ===
    let (mut fts_results, mut fts_scores, strong_signal) = if strategy.skip_fts {
        (vec![], std::collections::HashMap::<String, f32>::new(), false)
    } else {
        let fts_start = std::time::Instant::now();
        let (results, ranked) = try_tantivy_then_fts5(store, query, topic, effective_limit * 2)?;
        let scores: std::collections::HashMap<String, f32> = ranked.into_iter().collect();
        let ranked_vec: Vec<(String, f32)> = scores.iter().map(|(k, v)| (k.clone(), *v)).collect();
        let ss = crate::search::rerank_llm::detect_strong_signal(&ranked_vec);
        tracing::debug!(elapsed_ms = fts_start.elapsed().as_millis() as u64, hits = scores.len(), "fts search (original)");
        if ss { tracing::info!("strong BM25 signal — will skip expansion + LLM reranker"); }
        (results, scores, ss)
    };

    // === Query expansion: launch AFTER strong signal check to avoid unnecessary LLM calls ===
    let should_expand = match expand_override {
        Some(true) => true,
        Some(false) => false,
        None => {
            !strong_signal && strategy.query_type != crate::search::classify::QueryType::ExactKeyword
        }
    };
    let adaptive_max = match strategy.query_type {
        crate::search::classify::QueryType::Temporal => Some(1),
        crate::search::classify::QueryType::Episodic => Some(2),
        _ => None,
    };
    let expand_config = config.clone();
    let expand_query_str = query.to_string();
    let expand_handle = if should_expand {
        Some(std::thread::spawn(move || {
            crate::search::expand::expand_query(&expand_config, &expand_query_str, adaptive_max)
        }))
    } else {
        None
    };

    // === Phase 1b: Vec + KG search with ORIGINAL query (runs while expansion is in flight) ===

    let mut vec_scores: std::collections::HashMap<String, f32> = if strategy.skip_vec {
        std::collections::HashMap::new()
    } else {
        let vec_start = std::time::Instant::now();
        let r: std::collections::HashMap<String, f32> = try_vector_search(store, config, query, topic, effective_limit)
            .into_iter().collect();
        tracing::debug!(elapsed_ms = vec_start.elapsed().as_millis() as u64, hits = r.len(), "vector search (original)");
        r
    };

    let mut kg_scores: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
    {
        let kg_start = std::time::Instant::now();
        let seed_concepts = store.search_all_concepts(query, 5).unwrap_or_default();
        let concept_results = crate::search::kg_search::search_concepts_ranked_from(&seed_concepts, effective_limit);
        let seed_ids: Vec<String> = seed_concepts.iter().map(|c| c.id.clone()).collect();
        let bfs_expanded = if !seed_ids.is_empty() {
            crate::search::kg_search::bfs_expand_memories_by_id(store, &seed_ids, 2, effective_limit)
        } else {
            vec![]
        };
        for (id, score) in concept_results.into_iter().chain(bfs_expanded.into_iter()) {
            let entry = kg_scores.entry(id).or_default();
            *entry = entry.max(score);
        }
        tracing::debug!(elapsed_ms = kg_start.elapsed().as_millis() as u64, hits = kg_scores.len(), "kg search (original)");
    }

    // === Phase 2: Join expansion thread, search with expanded queries, merge ===
    // Skip entirely if strong signal detected (BM25 top1 is dominant, expansion won't help)
    let expanded_queries = if strong_signal {
        tracing::info!("strong signal — skipping expanded query searches");
        drop(expand_handle); // detach thread — LLM call completes in background, result discarded
        vec![]
    } else {
        expand_handle.and_then(|h| h.join().ok()).unwrap_or_default()
    };
    // Filter out expanded queries too similar to original (Jaccard word overlap > 0.8)
    let deduped_queries: Vec<&String> = expanded_queries.iter()
        .filter(|eq| word_jaccard(query, eq) <= 0.8)
        .collect();
    if deduped_queries.len() < expanded_queries.len() {
        tracing::debug!(
            before = expanded_queries.len(),
            after = deduped_queries.len(),
            "filtered similar expanded queries"
        );
    }
    if !deduped_queries.is_empty() {
        tracing::debug!(count = deduped_queries.len(), "merging expanded query results");

        // FTS: per-query (Tantivy is local, already fast)
        for eq in &deduped_queries {
            if !strategy.skip_fts {
                if let Ok((results, ranked)) = try_tantivy_then_fts5(store, eq, topic, effective_limit * 2) {
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
            let batch_results = try_vector_search_batch(store, config, &eq_strs, topic, effective_limit);
            for (id, score) in batch_results {
                let entry = vec_scores.entry(id).or_insert(f32::MIN);
                *entry = entry.max(score);
            }
        }

        // KG: per-query (local concept FTS + BFS)
        for eq in &deduped_queries {
            let seed_concepts = store.search_all_concepts(eq, 5).unwrap_or_default();
            let concept_results = crate::search::kg_search::search_concepts_ranked_from(&seed_concepts, effective_limit);
            let seed_ids: Vec<String> = seed_concepts.iter().map(|c| c.id.clone()).collect();
            let bfs_expanded = if !seed_ids.is_empty() {
                crate::search::kg_search::bfs_expand_memories_by_id(store, &seed_ids, 2, effective_limit)
            } else {
                vec![]
            };
            for (id, score) in concept_results.into_iter().chain(bfs_expanded.into_iter()) {
                let entry = kg_scores.entry(id).or_default();
                *entry = entry.max(score);
            }
        }
    }

    // Convert to ranked vecs for fusion — MUST sort by descending score
    // because RRF uses list position (rank), not score value.
    let mut fts_ranked: Vec<(String, f32)> = fts_scores.into_iter().collect();
    fts_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut vec_ranked: Vec<(String, f32)> = vec_scores.into_iter().collect();
    vec_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut kg_ranked: Vec<(String, f32)> = kg_scores.into_iter()
        .filter(|(id, _)| matches_topic(store, id, topic))
        .collect();
    kg_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    kg_ranked.truncate(effective_limit);

    let use_kg = !kg_ranked.is_empty();

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

    // === Score fusion (RRF or Convex Combination) ===
    // Only include paths that passed quality gating
    let fts_for_fusion = if use_fts { fts_ranked } else { vec![] };
    let vec_for_fusion = if use_vec { vec_ranked } else { vec![] };

    // === Adaptive alpha (M2): read from AdaptiveState if available ===
    let adaptive_alpha = if config.adaptive.enabled {
        crate::store::adaptive::AdaptiveState::restore_snapshot(store.conn())
            .and_then(|s| {
                let qt = format!("{}", strategy.query_type);
                s.get_alpha(&qt, None) // cluster_id: None until M4 integration
            })
    } else {
        None
    };

    // Capture per-channel scores for reranking and M2 logging.
    // Clamp negatives (rank sentinels like -1,-2) to 0 so reranker features are in [0, +inf).
    let fts_norm_log: std::collections::HashMap<String, f32> = fts_for_fusion.iter()
        .map(|(id, s)| {
            // Convert negative rank sentinels (-1,-2,...) to positive rank scores: 1/(1+|rank|)
            let score = if *s < 0.0 { 1.0 / (1.0 + s.abs()) } else { *s };
            (id.clone(), score)
        }).collect();
    let vec_norm_log: std::collections::HashMap<String, f32> = vec_for_fusion.iter()
        .map(|(id, s)| {
            let score = if *s < 0.0 { 1.0 / (1.0 + s.abs()) } else { *s };
            (id.clone(), score)
        }).collect();
    let kg_norm_log: std::collections::HashMap<String, f32> = kg_ranked.iter().cloned().collect();

    let fused = if config.search.fusion_method == "cc" {
        let alpha = adaptive_alpha
            .or(strategy.cc_alpha)
            .unwrap_or(config.search.cc_alpha as f32);
        // For CC mode, boost vec scores with KG signal (2-stage fusion)
        let vec_for_cc = if use_kg {
            let mut boosted = vec_for_fusion.clone();
            for (id, kg_score) in &kg_ranked {
                if let Some(pos) = boosted.iter().position(|(vid, _)| vid == id) {
                    boosted[pos].1 = boosted[pos].1.max(*kg_score * 0.5);
                } else {
                    boosted.push((id.clone(), *kg_score * 0.5));
                }
            }
            boosted
        } else {
            vec_for_fusion
        };
        crate::search::rrf::convex_combination(&fts_for_fusion, &vec_for_cc, alpha)
    } else {
        let rrf_k = config.search.rrf_k as f32;
        // Map strategy alpha to RRF weights (alpha=high → FTS dominant)
        let (fts_weight, vec_weight) = if let Some(alpha) = strategy.cc_alpha {
            (alpha, 1.0 - alpha)
        } else {
            (config.search.rrf_fts_weight as f32, config.search.rrf_vec_weight as f32)
        };
        let mut lists = Vec::new();
        if !fts_for_fusion.is_empty() { lists.push((fts_for_fusion, fts_weight)); }
        if !vec_for_fusion.is_empty() { lists.push((vec_for_fusion, vec_weight)); }
        if !kg_ranked.is_empty() {
            let kg_weight = 0.3; // KG is supplementary
            lists.push((kg_ranked.clone(), kg_weight));
        }
        crate::search::rrf::reciprocal_rank_fusion(&lists, rrf_k)
    };

    // Build memory lookup from already-fetched results
    let mut memory_map: std::collections::HashMap<String, Memory> = std::collections::HashMap::new();
    for m in fts_results {
        memory_map.entry(m.id.clone()).or_insert(m);
    }

    // Batch-fetch vector-search memories not already in FTS results (avoids N+1 queries)
    let missing_ids: Vec<String> = vec_ids.iter()
        .filter(|id| !memory_map.contains_key(*id))
        .cloned()
        .collect();
    if !missing_ids.is_empty() {
        for m in store.get_batch(&missing_ids) {
            memory_map.entry(m.id.clone()).or_insert(m);
        }
    }

    // Batch-fetch KG-sourced memories not already in map
    let kg_ids: Vec<String> = kg_norm_log.keys()
        .filter(|id| !memory_map.contains_key(*id))
        .cloned()
        .collect();
    if !kg_ids.is_empty() {
        for m in store.get_batch(&kg_ids) {
            memory_map.entry(m.id.clone()).or_insert(m);
        }
    }

    // Apply strength weighting (Ebbinghaus or KM survival curve) + temporal filter
    // Load cached per-cluster survival curves from M3 (if available)
    let mut survival_cache: std::collections::HashMap<u32, crate::search::survival::SurvivalCurve> = std::collections::HashMap::new();
    if config.adaptive.enabled {
        if let Ok(mut stmt) = store.conn().prepare(
            "SELECT key, value FROM metadata WHERE key LIKE 'survival_curve:%'"
        ) {
            let _ = stmt.query_map([], |row| {
                let key: String = row.get(0)?;
                let json: String = row.get(1)?;
                Ok((key, json))
            }).ok().map(|rows| {
                for row in rows.flatten() {
                    if let Some(id_str) = row.0.strip_prefix("survival_curve:") {
                        if let (Ok(cid), Ok(curve)) = (id_str.parse::<u32>(), serde_json::from_str(&row.1)) {
                            survival_cache.insert(cid, curve);
                        }
                    }
                }
            });
        }
    }

    let has_temporal = time_from.is_some() || time_to.is_some();
    let take_count = if has_temporal { usize::MAX } else { limit * 2 };
    let mut local_results: Vec<(Memory, f32)> = Vec::new();
    for (id, rrf_score) in fused.into_iter().take(take_count) {
        if let Some(memory) = memory_map.remove(&id) {
            // Temporal filter: skip memories outside the requested time range
            if let Some(from) = time_from {
                if memory.created_at < from { continue; }
            }
            if let Some(to) = time_to {
                if memory.created_at > to { continue; }
            }
            // Use per-cluster survival curve if available (M3), else Ebbinghaus
            let curve = memory.cluster_id.and_then(|cid| survival_cache.get(&cid));
            let final_score = crate::search::scoring::apply_strength_weighting_with_curve(rrf_score, &memory, curve);
            local_results.push((memory, final_score));
        }
    }
    local_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // === M5: Tier filtering — exclude Cold memories unless Exploratory query ===
    let include_cold = strategy.query_type == crate::search::classify::QueryType::Exploratory;
    if !include_cold {
        let before = local_results.len();
        local_results.retain(|(mem, _)| mem.tier != "cold");
        let filtered = before - local_results.len();
        if filtered > 0 {
            tracing::debug!(filtered, "cold tier memories excluded");
        }
    }

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
        for (mem, score) in local_results.iter_mut() {
            let features = crate::search::rerank::RerankFeatures {
                fts_score: fts_norm_log.get(&mem.id).copied().unwrap_or(0.0),
                vec_score: vec_norm_log.get(&mem.id).copied().unwrap_or(0.0),
                kg_score: kg_norm_log.get(&mem.id).copied().unwrap_or(0.0),
                recency_days: (chrono::Utc::now() - mem.created_at).num_hours() as f32 / 24.0,
                access_count: mem.access_count,
                strength: mem.strength as f32,
                importance_weight: importance_weight(&mem.importance),
                keyword_overlap: crate::search::rerank::compute_keyword_overlap(query, &mem.keywords, &mem.content),
            };
            *score = crate::search::rerank::rerank_score(&features, &weights);
        }
        local_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    }

    // === R2+: LLM reranker — override linear scores with LLM judgement ===
    // Skip if: strong BM25 signal, or linear rerank already shows clear separation (top1 >> top2)
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
    if !strong_signal && !linear_clear && config.reranker_provider() != crate::config::Provider::None && local_results.len() > 1 {
        let llm_scores = crate::search::rerank_llm::rerank_with_llm(config, query, &local_results);
        for (i, (_, score)) in local_results.iter_mut().enumerate() {
            if let Some(&llm_s) = llm_scores.get(i) {
                *score = llm_s;
            }
        }
        local_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    }

    // === Optional keyword filter ===
    if let Some(kw) = keyword {
        let kw_lower = kw.to_lowercase();
        local_results.retain(|(m, _)| {
            m.keywords.iter().any(|k| k.to_lowercase().contains(&kw_lower))
                || m.content.to_lowercase().contains(&kw_lower)
        });
    }

    // === Cross-validation (if enabled) ===
    // Supermemory search was launched at pipeline start (sm_handle); join it here.
    // AutoMemory is a fast local file scan.

    let am_enabled = config.sync.auto_memory_enabled;
    let am_glob = config.sync.auto_memory_glob.clone();
    let q_am = query.to_string();

    let auto_memory_results = if am_enabled {
        let scanner = AutoMemoryScanner::new(am_glob);
        scanner.scan(&q_am)
    } else {
        vec![]
    };

    // Join early-launched Supermemory thread (has been running since pipeline start)
    let supermemory_results = sm_handle
        .and_then(|h| h.join().ok())
        .unwrap_or_default();
    tracing::debug!(elapsed_ms = total_start.elapsed().as_millis() as u64, hits = supermemory_results.len(), "supermemory search (joined)");

    let validated = validate::cross_validate(&local_results, &supermemory_results, &auto_memory_results);

    // Build final results — scores already assigned by cross_validate
    let mut results: Vec<RecallResult> = validated
        .into_iter()
        .map(|v| {
            RecallResult {
                memory: v.memory,
                score: v.score,
                confidence: v.confidence,
                sources_hit: v.sources_hit,
            }
        })
        .collect();

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    // === M1: Emit recall_complete BEFORE truncation (full candidate set for counterfactual replay) ===
    if config.adaptive.enabled {
        let request_id = ulid::Ulid::new().to_string();
        let candidates: Vec<serde_json::Value> = results.iter()
            .filter(|r| !r.memory.id.starts_with("sm:") && !r.memory.id.starts_with("auto:"))
            .map(|r| {
                let bm25 = fts_norm_log.get(&r.memory.id).copied().unwrap_or(0.0);
                let vec = vec_norm_log.get(&r.memory.id).copied().unwrap_or(0.0);
                let kg = kg_norm_log.get(&r.memory.id).copied().unwrap_or(0.0);
                serde_json::json!({
                    "id": r.memory.id,
                    "bm25_norm": bm25,
                    "vec_norm": vec,
                    "kg_norm": kg,
                    "final_score": r.score,
                })
            })
            .collect();
        let alpha_used = adaptive_alpha
            .or(strategy.cc_alpha)
            .unwrap_or(config.search.cc_alpha as f32);
        let _ = crate::store::adaptive::emit_event(store.conn(), crate::store::adaptive::FeedbackEvent {
            event_type: crate::store::adaptive::EventType::RecallComplete,
            request_id: Some(request_id),
            memory_id: None,
            concept_id: None,
            query: Some(query.chars().take(200).collect()),
            query_type: Some(format!("{}", strategy.query_type)),
            topic: topic.map(|t| t.to_string()),
            payload: Some(serde_json::json!({
                "candidates": candidates,
                "alpha_used": alpha_used,
                "fusion_method": &config.search.fusion_method,
                "result_count": results.len(),
            })),
        });
    }

    // Truncate to the caller's requested limit (not effective_limit).
    results.truncate(limit);

    // Record recall hit (NOT access — access should only be counted when
    // the agent/user actually uses the memory, not just when it's returned).
    let recall_ids: Vec<String> = results.iter()
        .filter(|r| !r.memory.id.starts_with("sm:") && !r.memory.id.starts_with("auto:"))
        .map(|r| r.memory.id.clone())
        .collect();
    store.record_recall_hit(&recall_ids);

    // Periodically update quality weights (every ~50 recalls)
    let total_recalls: u64 = store.quality_metrics().map(|(_, r, _)| r).unwrap_or(0);
    if total_recalls.is_multiple_of(50) && total_recalls > 0 {
        store.update_quality_weights();
    }

    tracing::debug!(elapsed_ms = total_start.elapsed().as_millis() as u64, results = results.len(), "recall complete");
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
        let results = vec_search_direct(store, &cached, topic, limit);
        if !results.is_empty() {
            return results;
        }
    }

    // Level 3: Use configured embedder (Google or OMLX)
    let embedder = match crate::embed::create_embedder(config) {
        Some(e) => e,
        None => return vec![],
    };

    let embedding = tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(embedder.embed(query))
    });

    match embedding {
        Ok(emb) => {
            let _ = EmbedCache::put(store.conn(), query, &model, &emb);
            vec_search_direct(store, &emb, topic, limit)
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
            for (id, score) in vec_search_direct(store, &cached, topic, limit) {
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
    let embeddings = tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(embedder.embed_batch(&texts))
    });

    match embeddings {
        Ok(embs) => {
            for (emb, (_, q)) in embs.iter().zip(uncached.iter()) {
                let _ = EmbedCache::put(store.conn(), q, &model, emb);
                for (id, score) in vec_search_direct(store, emb, topic, limit) {
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

/// Check if a memory matches the requested topic filter.
fn matches_topic(store: &SqliteStore, id: &str, topic: Option<&str>) -> bool {
    match topic {
        None => true,
        Some(t) => store.conn()
            .query_row("SELECT topic FROM memories WHERE id = ?1", rusqlite::params![id], |row| row.get::<_, String>(0))
            .map(|mem_topic| mem_topic == t)
            .unwrap_or(false),
    }
}

/// Rank results by position and filter by topic.
fn rank_and_filter(results: Vec<(String, f32)>, store: &SqliteStore, topic: Option<&str>, limit: usize) -> Vec<(String, f32)> {
    results.into_iter()
        .filter(|(id, _)| matches_topic(store, id, topic))
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
) -> Vec<(String, f32)> {
    // Try HNSW first (O(log n) approximate nearest neighbor)
    let hnsw_path = store.db_path().with_extension("");
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

    // Fall back to sqlite-vec (brute-force O(n))
    match crate::store::vec::search_vec(store.conn(), embedding, limit) {
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
    let db_path = store.db_path();
    if db_path.to_str() != Some(":memory:") {
        let tantivy_path = db_path.with_extension("tantivy");
        if let Ok(tantivy) = crate::store::tantivy_fts::TantivyFts::open(&tantivy_path) {
            if let Ok(results) = tantivy.search(query, topic, limit) {
                if !results.is_empty() {
                    // Convert tantivy results to Memory objects
                    let mut memories = Vec::new();
                    let mut ranked = Vec::new();
                    for (i, (id, score)) in results.into_iter().enumerate() {
                        if let Ok(m) = store.get(&id) {
                            // Preserve original score for CC fusion; use rank for RRF
                            // Score is Tantivy BM25 relevance (positive float)
                            ranked.push((m.id.clone(), if score > 0.0 { score } else { -(i as f32) }));
                            memories.push(m);
                        }
                    }
                    if !memories.is_empty() {
                        tracing::debug!(hits = memories.len(), "tantivy search");
                        return Ok((memories, ranked));
                    }
                }
            }
        }
    }

    // Fall back to FTS5
    let fts_results = store.search_fts(query, topic, limit)?;
    let fts_ranked: Vec<(String, f32)> = fts_results
        .iter()
        .enumerate()
        .map(|(i, m)| (m.id.clone(), -(i as f32)))
        .collect();
    Ok((fts_results, fts_ranked))
}

/// Jaccard similarity for query dedup.
/// Uses word-level for space-separated text, falls back to character bigrams for CJK.
fn word_jaccard(a: &str, b: &str) -> f32 {
    let wa: std::collections::HashSet<&str> = a.split_whitespace().collect();
    let wb: std::collections::HashSet<&str> = b.split_whitespace().collect();
    // If both queries have ≤1 whitespace token (likely CJK), use character bigrams
    if wa.len() <= 1 && wb.len() <= 1 {
        let ca: std::collections::HashSet<(char, char)> = a.chars().zip(a.chars().skip(1)).collect();
        let cb: std::collections::HashSet<(char, char)> = b.chars().zip(b.chars().skip(1)).collect();
        let inter = ca.intersection(&cb).count() as f32;
        let union = ca.union(&cb).count() as f32;
        return if union == 0.0 { 1.0 } else { inter / union };
    }
    let inter = wa.intersection(&wb).count() as f32;
    let union = wa.union(&wb).count() as f32;
    if union == 0.0 { 1.0 } else { inter / union }
}
