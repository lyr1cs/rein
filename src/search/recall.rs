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
    // === Level 1: FTS5 (<1ms) ===
    let fts_results = store.search_fts(query, topic, limit * 2)?;
    let fts_ranked: Vec<(String, f32)> = fts_results
        .iter()
        .enumerate()
        .map(|(i, m)| (m.id.clone(), -(i as f32)))
        .collect();

    // === Level 2+3: Vector search (cached → API) ===
    let vec_ranked = try_vector_search(store, config, query, topic, limit);

    // === RRF fusion ===
    let rrf_k = config.search.rrf_k as f32;
    let fts_weight = config.search.rrf_fts_weight as f32;
    let vec_weight = config.search.rrf_vec_weight as f32;

    let mut lists = vec![(fts_ranked, fts_weight)];
    if !vec_ranked.is_empty() {
        lists.push((vec_ranked, vec_weight));
    }
    let fused = crate::search::rrf::reciprocal_rank_fusion(&lists, rrf_k);

    // Build memory lookup from already-fetched results
    let mut memory_map: std::collections::HashMap<String, Memory> = std::collections::HashMap::new();
    for m in fts_results {
        memory_map.entry(m.id.clone()).or_insert(m);
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
    let local_memories: Vec<Memory> = local_results.iter().map(|(m, _)| m.clone()).collect();
    let local_scores: std::collections::HashMap<String, f32> = local_results.iter().map(|(m, s)| (m.id.clone(), *s)).collect();

    let supermemory_results = if config.sync.supermemory_enabled {
        if let Some(ref api_key) = config.sync.api_key {
            let client = SupermemoryClient::new(api_key.clone());
            let q = query.to_string();
            tokio::task::block_in_place(move || {
                tokio::runtime::Handle::current().block_on(client.search(&q, limit))
            })
        } else {
            vec![]
        }
    } else {
        vec![]
    };

    let auto_memory_results = if config.sync.auto_memory_enabled {
        let scanner = AutoMemoryScanner::new(config.sync.auto_memory_glob.clone());
        scanner.scan(query)
    } else {
        vec![]
    };

    let validated = validate::cross_validate(&local_memories, &supermemory_results, &auto_memory_results);

    // Build final results
    let mut results: Vec<RecallResult> = validated
        .into_iter()
        .map(|v| {
            let score = local_scores.get(&v.memory.id).copied().unwrap_or(v.score);
            RecallResult {
                memory: v.memory,
                score,
                confidence: v.confidence,
                sources_hit: v.sources_hit,
            }
        })
        .collect();

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(limit);

    // Record access for returned memories
    for r in &results {
        if !r.memory.id.starts_with("sm:") && !r.memory.id.starts_with("auto:") {
            let _ = store.record_access(&r.memory.id);
        }
    }

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

/// Direct vector search using store.conn() — avoids async trait methods.
fn vec_search_direct(
    store: &SqliteStore,
    embedding: &[f32],
    topic: Option<&str>,
    limit: usize,
) -> Vec<(String, f32)> {
    match crate::store::vec::search_vec(store.conn(), embedding, limit) {
        Ok(results) => {
            let ranked: Vec<(String, f32)> = results
                .into_iter()
                .filter(|(id, _)| {
                    if let Some(t) = topic {
                        // Quick topic check via direct SQL
                        store.conn()
                            .query_row(
                                "SELECT topic FROM memories WHERE id = ?1",
                                rusqlite::params![id],
                                |row| row.get::<_, String>(0),
                            )
                            .map(|mem_topic| mem_topic == t)
                            .unwrap_or(false)
                    } else {
                        true
                    }
                })
                .enumerate()
                .map(|(i, (id, _))| (id, -(i as f32)))
                .collect();
            ranked
        }
        Err(_) => vec![],
    }
}


