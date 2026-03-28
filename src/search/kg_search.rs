//! Knowledge graph retrieval channel for the recall pipeline.
//! Searches concepts via FTS, then expands via BFS "land and expand".

use crate::store::SqliteStore;
use std::collections::{HashMap, HashSet};

/// Search concepts via FTS and return (memory_id, score) pairs.
/// Concepts link to memories via source_memory_ids — we return the
/// linked memories ranked by concept relevance.
pub fn search_concepts_ranked(
    store: &SqliteStore,
    query: &str,
    limit: usize,
) -> Vec<(String, f32)> {
    let concepts = store.search_all_concepts(query, limit * 2).unwrap_or_default();
    if concepts.is_empty() {
        return vec![];
    }

    // Collect memory IDs from concept source_memory_ids, scored by concept rank
    let mut memory_scores: HashMap<String, f32> = HashMap::new();
    for (rank, concept) in concepts.iter().enumerate() {
        let concept_score = 1.0 / (1.0 + rank as f32); // RRF-style rank score
        for mem_id in &concept.source_memory_ids {
            let entry = memory_scores.entry(mem_id.clone()).or_default();
            *entry = entry.max(concept_score); // Take best concept score per memory
        }
    }

    let mut ranked: Vec<(String, f32)> = memory_scores.into_iter().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked.truncate(limit);
    ranked
}

/// BFS "land and expand" — start from seed concept IDs, traverse links,
/// collect memory_ids from discovered concepts. Score decays with hop distance.
pub fn bfs_expand_memories(
    store: &SqliteStore,
    seed_concept_names: &[String],
    max_hops: usize,
    limit: usize,
) -> Vec<(String, f32)> {
    let mut visited: HashSet<String> = HashSet::new();
    let mut memory_scores: HashMap<String, f32> = HashMap::new();
    let mut frontier: Vec<(String, usize)> = Vec::new(); // (concept_id, hop_depth)

    // Resolve concept names to IDs and seed the frontier
    for name in seed_concept_names {
        // Search across all memoirs for this concept name
        if let Ok(concepts) = store.search_all_concepts(name, 1) {
            if let Some(c) = concepts.first() {
                if visited.insert(c.id.clone()) {
                    frontier.push((c.id.clone(), 0));
                    // Seed concepts get score 1.0
                    for mem_id in &c.source_memory_ids {
                        memory_scores.entry(mem_id.clone()).or_insert(1.0);
                    }
                }
            }
        }
    }

    // BFS traversal
    let mut i = 0;
    while i < frontier.len() {
        let (concept_id, depth) = frontier[i].clone();
        i += 1;

        if depth >= max_hops {
            continue;
        }

        // Get outgoing links, filter expired/future-dated links
        let now = chrono::Utc::now();
        let links = store.get_links_from(&concept_id).unwrap_or_default();
        for link in links {
            // Skip links outside their temporal validity window
            if let Some(valid_from) = link.valid_from {
                if now < valid_from { continue; }
            }
            if let Some(valid_until) = link.valid_until {
                if now > valid_until { continue; }
            }
            if visited.insert(link.target_id.clone()) {
                let hop_score = 1.0 / (1.0 + (depth + 1) as f32); // Decay with distance
                frontier.push((link.target_id.clone(), depth + 1));

                // Get target concept's memories
                if let Ok(Some(target)) = store.get_concept_by_id(&link.target_id) {
                    for mem_id in &target.source_memory_ids {
                        let entry = memory_scores.entry(mem_id.clone()).or_default();
                        *entry = entry.max(hop_score);
                    }
                }
            }
        }
    }

    let mut ranked: Vec<(String, f32)> = memory_scores.into_iter().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked.truncate(limit);
    ranked
}
