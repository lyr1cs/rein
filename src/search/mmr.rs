//! Maximal Marginal Relevance (MMR) diversity selection for recall results.
//!
//! MMR greedily selects results that are both relevant and diverse.
//! At each step, the next candidate is chosen to maximise:
//!
//!   score(d) = λ · relevance(d) − (1 − λ) · max_sim(d, selected)
//!
//! Similarity is approximated via topic equality and keyword overlap — no
//! embedding calls required, so it adds zero latency to the hot path.
//!
//! λ = 1.0 → pure relevance order (MMR off)
//! λ = 0.0 → pure diversity (ignores scores)
//! λ = 0.3 → strong diversity pressure while keeping top results relevant

use crate::types::Memory;

/// Apply MMR selection to re-order `candidates` (sorted by descending score)
/// and return at most `limit` results.
///
/// `lambda` controls the relevance-diversity tradeoff:
/// - 0.0: maximise diversity only
/// - 1.0: original score order (no diversity reranking)
/// - 0.3 (default): meaningful diversity without sacrificing top relevance
///
/// When `lambda == 1.0` or `candidates.len() <= limit`, returns the first
/// `limit` candidates unchanged — zero overhead for the common case.
pub fn apply_mmr(candidates: Vec<(Memory, f32)>, limit: usize, lambda: f32) -> Vec<(Memory, f32)> {
    if candidates.is_empty() || limit == 0 {
        return vec![];
    }
    // Fast path: diversity reranking disabled or nothing to rerank
    if lambda >= 1.0 || candidates.len() <= limit {
        return candidates.into_iter().take(limit).collect();
    }

    // Normalise scores once against the full candidate list so `lambda` is a
    // stable relevance-diversity knob throughout the greedy loop.
    //
    // Fold against NEG_INFINITY so all-negative score sets (RRF rank
    // sentinels `-0, -1, -2, ...`) produce the real max, not 0.0. Previously
    // starting from 0.0 could snap max_score to 0 → divide-by-near-zero →
    // normalised relevance collapsed to 0 → MMR degenerated to pure
    // diversity regardless of lambda. When every score is non-positive, we
    // shift to a >=0 range so the normalisation is well-defined.
    let raw_max = candidates
        .iter()
        .map(|(_, s)| *s)
        .fold(f32::NEG_INFINITY, f32::max);
    let raw_min = candidates
        .iter()
        .map(|(_, s)| *s)
        .fold(f32::INFINITY, f32::min);
    let shift = if raw_max <= 0.0 { -raw_min } else { 0.0 };
    let max_score = (raw_max + shift).max(1e-6);

    let mut selected: Vec<(Memory, f32)> = Vec::with_capacity(limit);
    // (memory, raw_score, fixed_normalised_relevance)
    let mut remaining: Vec<(Memory, f32, f32)> = candidates
        .into_iter()
        .map(|(m, s)| (m, s, (s + shift) / max_score))
        .collect();

    while selected.len() < limit && !remaining.is_empty() {
        let best_idx = remaining
            .iter()
            .enumerate()
            .map(|(i, (cand, _score, rel))| {
                let max_sim = if selected.is_empty() {
                    0.0
                } else {
                    selected
                        .iter()
                        .map(|(sel, _)| topic_keyword_similarity(cand, sel))
                        .fold(0.0f32, f32::max)
                };
                let mmr_score = lambda * rel - (1.0 - lambda) * max_sim;
                (i, mmr_score)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        let (m, s, _) = remaining.remove(best_idx);
        selected.push((m, s));
    }

    selected
}

/// Approximate semantic similarity between two memories using topic equality
/// and keyword overlap.  Returns a value in [0.0, 1.0].
///
/// - Same topic → 1.0 (identical concept cluster)
/// - Same topic prefix (first word/segment) → 0.6
/// - Keyword Jaccard overlap → scaled similarity
fn topic_keyword_similarity(a: &Memory, b: &Memory) -> f32 {
    let ta = a.topic.to_lowercase();
    let tb = b.topic.to_lowercase();

    if ta == tb {
        return 1.0;
    }

    // Shared first topic segment (e.g. "rein-v0-15" vs "rein-v0-14")
    let prefix_sim = topic_prefix_similarity(&ta, &tb);

    // Keyword Jaccard overlap
    let kw_sim = if a.keywords.is_empty() || b.keywords.is_empty() {
        0.0
    } else {
        let set_a: std::collections::HashSet<String> =
            a.keywords.iter().map(|k| k.to_lowercase()).collect();
        let set_b: std::collections::HashSet<String> =
            b.keywords.iter().map(|k| k.to_lowercase()).collect();
        let inter = set_a.intersection(&set_b).count();
        let union = set_a.union(&set_b).count();
        if union == 0 {
            0.0
        } else {
            inter as f32 / union as f32
        }
    };

    prefix_sim.max(kw_sim)
}

/// Return a similarity score [0, 0.7] based on how much of the topic prefix
/// the two strings share (split on `-`, `_`, `/`, or whitespace).
fn topic_prefix_similarity(a: &str, b: &str) -> f32 {
    let seg_a: Vec<&str> = a
        .split(['-', '_', '/', ' '])
        .filter(|s| !s.is_empty())
        .collect();
    let seg_b: Vec<&str> = b
        .split(['-', '_', '/', ' '])
        .filter(|s| !s.is_empty())
        .collect();
    if seg_a.is_empty() || seg_b.is_empty() {
        return 0.0;
    }
    let shared = seg_a
        .iter()
        .zip(seg_b.iter())
        .take_while(|(a, b)| a == b)
        .count();
    let max_len = seg_a.len().max(seg_b.len());
    // Cap at 0.7: never reach 1.0 here since equal topics are handled above
    (shared as f32 / max_len as f32) * 0.7
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::tiering::MemoryTier;
    use crate::types::{Importance, Memory, MemoryLayer, MemoryStatus, Source};

    fn make_memory(id: &str, topic: &str, keywords: &[&str]) -> Memory {
        Memory {
            id: id.to_string(),
            topic: topic.to_string(),
            summary: String::new(),
            content: String::new(),
            keywords: keywords.iter().map(|s| s.to_string()).collect(),
            importance: Importance::Medium,
            layer: MemoryLayer::LTM,
            status: MemoryStatus::Active,
            source: Source::Manual,
            strength: 1.0,
            decay_lambda: 0.02,
            access_count: 0,
            superseded_by: None,
            canonical_id: None,
            support_count: 1,
            merge_count: 0,
            dedup_confidence: 1.0,
            source_diversity: 1.0,
            contradiction_score: 0.0,
            related_ids: vec![],
            concept_ids: vec![],
            embedding: None,
            tier: MemoryTier::Warm,
            cluster_id: None,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            last_accessed: chrono::Utc::now(),
        }
    }

    #[test]
    fn test_mmr_lambda_1_is_identity() {
        let mems: Vec<(Memory, f32)> = vec![
            (make_memory("m1", "rust", &["ownership"]), 0.9),
            (make_memory("m2", "rust", &["borrowing"]), 0.8),
            (make_memory("m3", "python", &["decorators"]), 0.7),
        ];
        let result = apply_mmr(mems.clone(), 2, 1.0);
        assert_eq!(result[0].0.id, "m1");
        assert_eq!(result[1].0.id, "m2");
    }

    #[test]
    fn test_mmr_promotes_diverse_result() {
        // m1 and m2 are same topic "rust", m3 is "python"
        // With diversity pressure, m3 should be preferred over m2 for second slot
        let mems: Vec<(Memory, f32)> = vec![
            (make_memory("m1", "rust", &["ownership", "memory"]), 0.9),
            (make_memory("m2", "rust", &["ownership", "borrowing"]), 0.85),
            (make_memory("m3", "python", &["decorators", "async"]), 0.8),
        ];
        let result = apply_mmr(mems, 2, 0.3);
        assert_eq!(result[0].0.id, "m1", "top result should still be m1");
        assert_eq!(result[1].0.id, "m3", "second slot should be diverse m3");
    }

    #[test]
    fn test_mmr_fewer_than_limit() {
        let mems: Vec<(Memory, f32)> = vec![(make_memory("m1", "rust", &[]), 0.9)];
        let result = apply_mmr(mems, 10, 0.3);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_topic_prefix_similarity() {
        assert!(topic_prefix_similarity("rein-v0-15", "rein-v0-14") > 0.0);
        assert!(topic_prefix_similarity("rein-v0-15", "rein-v0-14") < 1.0);
        assert_eq!(topic_prefix_similarity("rust", "python"), 0.0);
    }
}
