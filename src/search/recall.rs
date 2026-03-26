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
    let _span = tracing::info_span!("recall", query_len = query.len()).entered();
    let total_start = std::time::Instant::now();

    // === Level 1: FTS (Tantivy BM25 → FTS5 fallback) ===
    let fts_start = std::time::Instant::now();
    let (fts_results, fts_ranked) = try_tantivy_then_fts5(store, query, topic, limit * 2)?;
    tracing::debug!(elapsed_ms = fts_start.elapsed().as_millis() as u64, hits = fts_results.len(), "fts search");

    // === Level 2+3: Vector search (cached → API) ===
    let vec_start = std::time::Instant::now();
    let vec_ranked = try_vector_search(store, config, query, topic, limit);
    tracing::debug!(elapsed_ms = vec_start.elapsed().as_millis() as u64, hits = vec_ranked.len(), "vector search");

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

    let fused = if config.search.fusion_method == "cc" {
        let alpha = config.search.cc_alpha as f32;
        crate::search::rrf::convex_combination(&fts_for_fusion, &vec_for_fusion, alpha)
    } else {
        let rrf_k = config.search.rrf_k as f32;
        let fts_weight = config.search.rrf_fts_weight as f32;
        let vec_weight = config.search.rrf_vec_weight as f32;
        let mut lists = Vec::new();
        if !fts_for_fusion.is_empty() { lists.push((fts_for_fusion, fts_weight)); }
        if !vec_for_fusion.is_empty() { lists.push((vec_for_fusion, vec_weight)); }
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

    // Apply Ebbinghaus weighting
    let mut local_results: Vec<(Memory, f32)> = Vec::new();
    for (id, rrf_score) in fused.into_iter().take(limit * 2) {
        if let Some(memory) = memory_map.remove(&id) {
            let final_score = crate::search::scoring::apply_strength_weighting(rrf_score, &memory);
            local_results.push((memory, final_score));
        }
    }
    local_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // === Optional keyword filter ===
    if let Some(kw) = keyword {
        let kw_lower = kw.to_lowercase();
        local_results.retain(|(m, _)| {
            m.keywords.iter().any(|k| k.to_lowercase().contains(&kw_lower))
                || m.content.to_lowercase().contains(&kw_lower)
        });
    }

    // === Cross-validation (if enabled) ===
    // Run Supermemory + AutoMemory in parallel to reduce total latency.
    // Supermemory is 200-500ms network; AutoMemory is local file scan.

    let sm_enabled = config.sync.supermemory_enabled;
    let sm_api_key = config.sync.api_key.clone();
    let sm_endpoint = config.sync.endpoint.clone();
    let am_enabled = config.sync.auto_memory_enabled;
    let am_glob = config.sync.auto_memory_glob.clone();
    let q_sm = query.to_string();
    let q_am = query.to_string();

    let sm_start = std::time::Instant::now();

    // Spawn Supermemory search in a thread (it's async + network I/O)
    let sm_handle = if sm_enabled {
        if let Some(api_key) = sm_api_key {
            Some(std::thread::spawn(move || {
                let client = SupermemoryClient::new(api_key, sm_endpoint);
                // Build a small runtime for this thread since we can't share the main one
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .ok();
                rt.map(|rt| rt.block_on(client.search(&q_sm, limit)))
                    .unwrap_or_default()
            }))
        } else {
            None
        }
    } else {
        None
    };

    // AutoMemory runs on current thread (fast local scan)
    let auto_memory_results = if am_enabled {
        let scanner = AutoMemoryScanner::new(am_glob);
        scanner.scan(&q_am)
    } else {
        vec![]
    };

    // Join Supermemory thread
    let supermemory_results = sm_handle
        .and_then(|h| h.join().ok())
        .unwrap_or_default();
    tracing::debug!(elapsed_ms = sm_start.elapsed().as_millis() as u64, hits = supermemory_results.len(), "supermemory search");

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
    results.truncate(limit);

    // Record recall hit (NOT access — access should only be counted when
    // the agent/user actually uses the memory, not just when it's returned).
    // This separation is critical for the quality feedback loop:
    // "bad" = recalled many times but never accessed = low quality.
    let recall_ids: Vec<String> = results.iter()
        .filter(|r| !r.memory.id.starts_with("sm:") && !r.memory.id.starts_with("auto:"))
        .map(|r| r.memory.id.clone())
        .collect();
    store.record_recall_hit(&recall_ids);

    // Periodically update quality weights (every ~50 recalls)
    let total_recalls: u64 = store.quality_metrics().map(|(_, r, _)| r).unwrap_or(0);
    if total_recalls % 50 == 0 && total_recalls > 0 {
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
