use crate::types::{Embedder, Memory, MemoryStore, ReinResult};

/// Result from waterfall search with score and source info.
#[derive(Debug)]
pub struct SearchResult {
    pub memory: Memory,
    pub score: f32,
    pub source: SearchSource,
}

/// Which search backend produced the result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchSource {
    Fts,
    CachedVec,
    ApiVec,
}

/// Three-level waterfall search pipeline.
/// Level 1: FTS5 (local, <1ms)
/// Level 2: Cached vector search (local, <1ms)
/// Level 3: API embedding + vector search (~255ms)
///
/// Results are fused with RRF and weighted by Ebbinghaus strength.
pub async fn waterfall_search<S: MemoryStore, E: Embedder>(
    store: &S,
    query: &str,
    embedder: Option<&E>,
    limit: usize,
    rrf_k: f32,
    fts_weight: f32,
    vec_weight: f32,
) -> ReinResult<Vec<SearchResult>> {
    // 1. FTS5 search
    let fts_results = store.search_fts(query, None, limit).await?;
    let fts_ranked: Vec<(String, f32)> = fts_results
        .iter()
        .enumerate()
        .map(|(i, m)| (m.id.clone(), -(i as f32))) // rank-based
        .collect();

    // 2. Vector search (cached or API)
    let vec_results = if let Some(emb) = embedder {
        let embedding = emb.embed(query).await?;
        store.search_vec(&embedding, None, limit).await?
    } else {
        vec![]
    };

    // 3. RRF fusion
    let mut lists = vec![(fts_ranked, fts_weight)];
    if !vec_results.is_empty() {
        let vr: Vec<(String, f32)> = vec_results
            .iter()
            .enumerate()
            .map(|(i, m)| (m.id.clone(), -(i as f32)))
            .collect();
        lists.push((vr, vec_weight));
    }
    let fused = crate::search::rrf::reciprocal_rank_fusion(&lists, rrf_k);

    // 4. Build a lookup of all memories we already fetched
    let mut memory_map: std::collections::HashMap<String, Memory> =
        std::collections::HashMap::new();
    for m in fts_results {
        memory_map.entry(m.id.clone()).or_insert(m);
    }
    for m in vec_results {
        memory_map.entry(m.id.clone()).or_insert(m);
    }

    // 5. Apply Ebbinghaus weighting + build results
    let mut results = Vec::new();
    for (id, rrf_score) in fused.into_iter().take(limit) {
        if let Some(memory) = memory_map.remove(&id) {
            let final_score =
                crate::search::scoring::apply_strength_weighting(rrf_score, &memory);
            results.push(SearchResult {
                memory,
                score: final_score,
                source: SearchSource::Fts, // simplified for now
            });
        }
    }
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(results)
}
