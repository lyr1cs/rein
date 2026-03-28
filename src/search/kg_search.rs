//! Knowledge graph retrieval channel for the recall pipeline.
//! Searches concepts via FTS, then expands via BFS "land and expand".

use crate::store::SqliteStore;
use crate::types::Concept;
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
    search_concepts_ranked_from(&concepts, limit)
}

/// Score memory IDs from pre-fetched concepts (avoids redundant FTS query).
pub fn search_concepts_ranked_from(
    concepts: &[Concept],
    limit: usize,
) -> Vec<(String, f32)> {
    if concepts.is_empty() {
        return vec![];
    }

    let mut memory_scores: HashMap<String, f32> = HashMap::new();
    for (rank, concept) in concepts.iter().enumerate() {
        let concept_score = 1.0 / (1.0 + rank as f32);
        for mem_id in &concept.source_memory_ids {
            let entry = memory_scores.entry(mem_id.clone()).or_default();
            *entry = entry.max(concept_score);
        }
    }

    let mut ranked: Vec<(String, f32)> = memory_scores.into_iter().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked.truncate(limit);
    ranked
}

/// BFS "land and expand" from seed concept IDs (preferred — avoids name collisions).
pub fn bfs_expand_memories_by_id(
    store: &SqliteStore,
    seed_concept_ids: &[String],
    max_hops: usize,
    limit: usize,
) -> Vec<(String, f32)> {
    let mut visited: HashSet<String> = HashSet::new();
    let mut memory_scores: HashMap<String, f32> = HashMap::new();
    let mut frontier: Vec<(String, usize)> = Vec::new();

    // Seed from concept IDs directly
    for cid in seed_concept_ids {
        if visited.insert(cid.clone()) {
            frontier.push((cid.clone(), 0));
            if let Ok(Some(c)) = store.get_concept_by_id(cid) {
                for mem_id in &c.source_memory_ids {
                    memory_scores.entry(mem_id.clone()).or_insert(1.0);
                }
            }
        }
    }

    bfs_core(store, &mut visited, &mut memory_scores, &mut frontier, max_hops);

    let mut ranked: Vec<(String, f32)> = memory_scores.into_iter().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked.truncate(limit);
    ranked
}

/// BFS "land and expand" from seed concept names (legacy, used by tests).
pub fn bfs_expand_memories(
    store: &SqliteStore,
    seed_concept_names: &[String],
    max_hops: usize,
    limit: usize,
) -> Vec<(String, f32)> {
    let mut visited: HashSet<String> = HashSet::new();
    let mut memory_scores: HashMap<String, f32> = HashMap::new();
    let mut frontier: Vec<(String, usize)> = Vec::new();

    for name in seed_concept_names {
        if let Ok(concepts) = store.search_all_concepts(name, 1) {
            if let Some(c) = concepts.first() {
                if visited.insert(c.id.clone()) {
                    frontier.push((c.id.clone(), 0));
                    for mem_id in &c.source_memory_ids {
                        memory_scores.entry(mem_id.clone()).or_insert(1.0);
                    }
                }
            }
        }
    }

    bfs_core(store, &mut visited, &mut memory_scores, &mut frontier, max_hops);

    let mut ranked: Vec<(String, f32)> = memory_scores.into_iter().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked.truncate(limit);
    ranked
}

/// Shared BFS traversal core with temporal link filtering.
fn bfs_core(
    store: &SqliteStore,
    visited: &mut HashSet<String>,
    memory_scores: &mut HashMap<String, f32>,
    frontier: &mut Vec<(String, usize)>,
    max_hops: usize,
) {
    let mut i = 0;
    while i < frontier.len() {
        let (concept_id, depth) = frontier[i].clone();
        i += 1;

        if depth >= max_hops {
            continue;
        }

        let now = chrono::Utc::now();
        let links = store.get_links_from(&concept_id).unwrap_or_default();
        for link in links {
            if let Some(valid_from) = link.valid_from {
                if now < valid_from { continue; }
            }
            if let Some(valid_until) = link.valid_until {
                if now > valid_until { continue; }
            }
            if visited.insert(link.target_id.clone()) {
                let hop_score = 1.0 / (1.0 + (depth + 1) as f32);
                frontier.push((link.target_id.clone(), depth + 1));

                if let Ok(Some(target)) = store.get_concept_by_id(&link.target_id) {
                    for mem_id in &target.source_memory_ids {
                        let entry = memory_scores.entry(mem_id.clone()).or_default();
                        *entry = entry.max(hop_score);
                    }
                }
            }
        }
    }
}
