// Use jaccard (symmetric) for cross-source validation, not containment (asymmetric).
// Containment is good for dedup (short text subset of long), but bad for corroboration
// where we need genuine independent confirmation, not subset overlap.
use crate::extract::dedup::jaccard_similarity;
use crate::types::MemoryTier;
use crate::types::Memory;

/// Cross-validation result wrapping a memory with confidence score.
pub struct ValidatedResult {
    pub memory: Memory,
    pub score: f32,
    pub confidence: f32,
    pub sources_hit: usize,
}

/// Cross-validate results from multiple sources.
/// Matches results by keyword/content overlap (Jaccard > 0.3).
/// Confidence is based on how many sources agree.
/// `local_results` are (Memory, score) pairs from RRF fusion.
/// External-only results are always ranked below local results.
pub fn cross_validate(
    local_results: &[(Memory, f32)],
    supermemory_results: &[Memory],
    auto_memory_results: &[Memory],
) -> Vec<ValidatedResult> {
    let mut validated = Vec::new();

    // Start with local results as base (already scored by RRF + Ebbinghaus)
    for (local, local_score) in local_results {
        let mut sources_hit = 1; // local always counts

        if has_matching_result(local, supermemory_results) {
            sources_hit += 1;
        }

        if has_matching_result(local, auto_memory_results) {
            sources_hit += 1;
        }

        let confidence = confidence_from_sources(sources_hit);

        validated.push(ValidatedResult {
            memory: local.clone(),
            score: *local_score, // preserve RRF score
            confidence,
            sources_hit,
        });
    }

    // Extract just the memories for matching
    let local_memories: Vec<&Memory> = local_results.iter().map(|(m, _)| m).collect();

    // Add unique results from supermemory not in local.
    // Score them LOWER than local results so they supplement, not replace.
    let local_min_score = local_results.iter()
        .map(|(_, s)| *s)
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0.0);
    // External results get half the minimum local score (always ranked below local)
    let external_score = (local_min_score * 0.5).max(0.001);

    for sm in supermemory_results {
        if !has_matching_result_refs(sm, &local_memories) {
            let mut sources_hit = 1;
            if has_matching_result(sm, auto_memory_results) {
                sources_hit += 1;
            }
            validated.push(ValidatedResult {
                memory: sm.clone(),
                score: external_score,
                confidence: confidence_from_sources(sources_hit),
                sources_hit,
            });
        }
    }

    // Add unique results from auto-memory not in local or supermemory
    for am in auto_memory_results {
        if !has_matching_result_refs(am, &local_memories)
            && !has_matching_result(am, supermemory_results)
        {
            validated.push(ValidatedResult {
                memory: am.clone(),
                score: external_score,
                confidence: confidence_from_sources(1),
                sources_hit: 1,
            });
        }
    }

    validated
}

fn confidence_from_sources(sources_hit: usize) -> f32 {
    match sources_hit {
        n if n >= 3 => 0.95,
        2 => 0.85,
        1 => 0.62,
        _ => 0.0,
    }
}

/// Check if a memory has a matching result in another source.
/// Matching = Jaccard similarity on content words > 0.3.
fn has_matching_result(memory: &Memory, candidates: &[Memory]) -> bool {
    candidates
        .iter()
        .any(|c| jaccard_similarity(&memory.content, &c.content) > 0.3)
}

/// Same as has_matching_result but for a slice of references.
fn has_matching_result_refs(memory: &Memory, candidates: &[&Memory]) -> bool {
    candidates
        .iter()
        .any(|c| jaccard_similarity(&memory.content, &c.content) > 0.3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Importance, MemoryLayer, MemoryStatus, Source};

    fn make_memory(id: &str, content: &str) -> Memory {
        Memory {
            id: id.to_string(),
            layer: MemoryLayer::LTM,
            topic: "test".to_string(),
            summary: content.chars().take(50).collect(),
            content: content.to_string(),
            keywords: vec![],
            importance: Importance::Medium,
            source: Source::Manual,
            strength: 1.0,
            decay_lambda: 0.0,
            access_count: 0,
            superseded_by: None,
            related_ids: vec![],
            concept_ids: vec![],
            status: MemoryStatus::default(),
            embedding: None,
            tier: MemoryTier::Warm,
            cluster_id: None,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            last_accessed: chrono::Utc::now(),
        }
    }

    #[test]
    fn test_confidence_three_sources() {
        // Same content in all three sources -> high confidence
        let content = "rust pattern matching error handling options results";
        let local = vec![(make_memory("local-1", content), 0.5)];
        let supermemory = vec![make_memory("sm-1", content)];
        let auto = vec![make_memory("auto-1", content)];

        let results = cross_validate(&local, &supermemory, &auto);
        assert_eq!(results.len(), 1);
        assert!((results[0].confidence - 0.95).abs() < f32::EPSILON);
        assert_eq!(results[0].sources_hit, 3);
    }

    #[test]
    fn test_confidence_two_sources() {
        let content = "rust pattern matching error handling options results";
        let local = vec![(make_memory("local-1", content), 0.5)];
        let supermemory = vec![make_memory("sm-1", content)];
        let auto: Vec<Memory> = vec![];

        let results = cross_validate(&local, &supermemory, &auto);
        assert_eq!(results.len(), 1);
        assert!((results[0].confidence - 0.85).abs() < f32::EPSILON);
        assert_eq!(results[0].sources_hit, 2);
    }

    #[test]
    fn test_confidence_one_source() {
        let local = vec![(make_memory("local-1", "unique local content only here"), 0.5)];
        let supermemory: Vec<Memory> = vec![];
        let auto: Vec<Memory> = vec![];

        let results = cross_validate(&local, &supermemory, &auto);
        assert_eq!(results.len(), 1);
        assert!((results[0].confidence - 0.62).abs() < f32::EPSILON);
        assert_eq!(results[0].sources_hit, 1);
    }

    #[test]
    fn test_no_match_across_sources() {
        // Completely different content in each source -> each gets 0.62
        // External-only results should have lower score than local
        let local = vec![(make_memory("local-1", "alpha beta gamma delta epsilon"), 0.5)];
        let supermemory = vec![make_memory("sm-1", "one two three four five six")];
        let auto = vec![make_memory("auto-1", "rouge bleu vert jaune orange")];

        let results = cross_validate(&local, &supermemory, &auto);
        assert_eq!(results.len(), 3);
        for r in &results {
            assert!((r.confidence - 0.62).abs() < f32::EPSILON);
            assert_eq!(r.sources_hit, 1);
        }
        // Local result should have higher score than external results
        assert!(results[0].score > results[1].score, "local should rank above supermemory");
        assert!(results[0].score > results[2].score, "local should rank above auto-memory");
    }
}
