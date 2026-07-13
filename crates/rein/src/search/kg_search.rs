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
    let concepts = store
        .search_all_concepts(query, limit * 2)
        .unwrap_or_default();
    search_concepts_ranked_from(&concepts, limit)
}

/// Score memory IDs from pre-fetched concepts (avoids redundant FTS query).
pub fn search_concepts_ranked_from(concepts: &[Concept], limit: usize) -> Vec<(String, f32)> {
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
    bfs_expand_memories_by_id_at(store, seed_concept_ids, max_hops, limit, chrono::Utc::now())
}

/// Deterministic form of production BFS. It applies the exact inclusive
/// `valid_from <= at <= valid_until` predicate at a caller-provided instant.
pub(crate) fn bfs_expand_memories_by_id_at(
    store: &SqliteStore,
    seed_concept_ids: &[String],
    max_hops: usize,
    limit: usize,
    evaluation_at: chrono::DateTime<chrono::Utc>,
) -> Vec<(String, f32)> {
    let mut visited: HashSet<String> = HashSet::new();
    let mut memory_scores: HashMap<String, f32> = HashMap::new();
    let mut frontier: Vec<(String, usize)> = Vec::new();
    for cid in seed_concept_ids {
        if visited.insert(cid.clone()) {
            frontier.push((cid.clone(), 0));
            if let Ok(Some(concept)) = store.get_concept_by_id(cid) {
                for memory_id in &concept.source_memory_ids {
                    memory_scores.entry(memory_id.clone()).or_insert(1.0);
                }
            }
        }
    }
    bfs_core(
        store,
        &mut visited,
        &mut memory_scores,
        &mut frontier,
        max_hops,
        evaluation_at,
    );
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

    bfs_core(
        store,
        &mut visited,
        &mut memory_scores,
        &mut frontier,
        max_hops,
        chrono::Utc::now(),
    );

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
    temporal_at: chrono::DateTime<chrono::Utc>,
) {
    let mut i = 0;
    while i < frontier.len() {
        let (concept_id, depth) = frontier[i].clone();
        i += 1;

        if depth >= max_hops {
            continue;
        }

        let links = store.get_links_from(&concept_id).unwrap_or_default();
        for link in links {
            let temporally_valid = link
                .valid_from
                .is_none_or(|valid_from| temporal_at >= valid_from)
                && link
                    .valid_until
                    .is_none_or(|valid_until| temporal_at <= valid_until);
            if !temporally_valid {
                continue;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_time_bfs_matches_production_before_inside_and_after_validity_window() {
        let store = SqliteStore::in_memory().unwrap();
        let valid_from = chrono::DateTime::parse_from_rfc3339("2026-07-13T10:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        let valid_until = chrono::DateTime::parse_from_rfc3339("2026-07-13T11:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        let now = valid_from.to_rfc3339();
        store
            .conn()
            .execute(
                "INSERT INTO memoirs (id, name, created_at, updated_at) \
                 VALUES ('m', 'm', ?1, ?1)",
                rusqlite::params![&now],
            )
            .unwrap();
        for (id, memory_ids) in [
            ("seed", "[]"),
            ("timeless", "[\"timeless-memory\"]"),
            ("bounded", "[\"bounded-memory\"]"),
        ] {
            store
                .conn()
                .execute(
                    "INSERT INTO concepts \
                     (id, memoir_id, name, definition, source_memory_ids, created_at, updated_at) \
                     VALUES (?1, 'm', ?1, ?1, ?2, ?3, ?3)",
                    rusqlite::params![id, memory_ids, &now],
                )
                .unwrap();
        }
        store
            .conn()
            .execute(
                "INSERT INTO concept_links \
                 (id, source_id, target_id, relation, created_at, valid_from, valid_until) \
                 VALUES ('timeless-link', 'seed', 'timeless', 'related_to', ?1, NULL, NULL)",
                rusqlite::params![&now],
            )
            .unwrap();
        store
            .conn()
            .execute(
                "INSERT INTO concept_links \
                 (id, source_id, target_id, relation, created_at, valid_from, valid_until) \
                 VALUES ('bounded-link', 'seed', 'bounded', 'related_to', ?1, ?2, ?3)",
                rusqlite::params![&now, valid_from.to_rfc3339(), valid_until.to_rfc3339()],
            )
            .unwrap();

        let ids_at = |at| {
            bfs_expand_memories_by_id_at(&store, &["seed".to_string()], 2, 10, at)
                .into_iter()
                .map(|(memory_id, _score)| memory_id)
                .collect::<Vec<_>>()
        };
        let before = ids_at(valid_from - chrono::Duration::seconds(1));
        let inside = ids_at(valid_from + chrono::Duration::minutes(30));
        let after = ids_at(valid_until + chrono::Duration::seconds(1));

        for ids in [&before, &inside, &after] {
            assert!(ids.contains(&"timeless-memory".to_string()));
        }
        assert!(!before.contains(&"bounded-memory".to_string()));
        assert!(inside.contains(&"bounded-memory".to_string()));
        assert!(!after.contains(&"bounded-memory".to_string()));
    }
}
