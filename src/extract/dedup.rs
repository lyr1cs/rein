use std::collections::HashSet;

use crate::store::SqliteStore;
use crate::types::{MemoryStore, ReinResult};

/// Normalize text for similarity comparison: lowercase + strip punctuation.
fn normalize_tokens(text: &str) -> HashSet<String> {
    text.split_whitespace()
        .map(|t| t.to_lowercase().chars().filter(|c| c.is_alphanumeric()).collect::<String>())
        .filter(|t| !t.is_empty())
        .collect()
}

/// Jaccard similarity between two texts (token-level, punctuation-stripped).
pub fn jaccard_similarity(a: &str, b: &str) -> f32 {
    let set_a = normalize_tokens(a);
    let set_b = normalize_tokens(b);
    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();
    if union == 0 {
        return 0.0;
    }
    intersection as f32 / union as f32
}

/// Containment similarity: what fraction of the shorter text is covered by the longer.
/// Better than Jaccard for dedup — a short summary of a longer text scores high.
pub fn containment_similarity(a: &str, b: &str) -> f32 {
    let set_a = normalize_tokens(a);
    let set_b = normalize_tokens(b);
    let smaller = set_a.len().min(set_b.len());
    if smaller == 0 {
        return 0.0;
    }
    set_a.intersection(&set_b).count() as f32 / smaller as f32
}

/// Combined similarity: max of Jaccard and Containment.
/// Use this for dedup decisions — catches both paraphrases and subsets.
pub fn similarity(a: &str, b: &str) -> f32 {
    jaccard_similarity(a, b).max(containment_similarity(a, b))
}

/// What to do when storing a potentially duplicate memory.
pub enum DedupAction {
    /// No duplicate found, create new memory.
    CreateNew,
    /// Similar content within time window, merge into existing memory.
    MergeInto(String),
    /// Similar content but older than time window, supersede old memory.
    Supersede(String),
    /// Gray zone (0.5 <= sim < threshold): needs LLM judgment.
    /// Falls back to CreateNew if LLM unavailable.
    GrayZone(String, f32),
}

/// Check for duplicate memories using FTS search and Jaccard similarity.
///
/// Given the store, a topic, and content text, search existing memories in that
/// topic using FTS. For the best match, compute Jaccard similarity.
/// - If > threshold and time diff < time_window_days -> MergeInto(id)
/// - If > threshold and time diff >= time_window_days -> Supersede(id)
/// - Otherwise -> CreateNew
pub fn check_dedup(
    store: &SqliteStore,
    topic: &str,
    content: &str,
    similarity_threshold: f32,
    time_window_days: i64,
) -> ReinResult<DedupAction> {
    // Extract key tokens from content for FTS query (take first few words)
    let query_tokens: Vec<&str> = content.split_whitespace().take(20).collect();
    if query_tokens.is_empty() {
        return Ok(DedupAction::CreateNew);
    }
    let query = query_tokens.join(" ");

    // Search existing memories in this topic
    let candidates = store.search_fts(&query, Some(topic), 10)?;

    if candidates.is_empty() {
        return Ok(DedupAction::CreateNew);
    }

    // Find best match by Jaccard similarity
    let mut best_sim = 0.0f32;
    let mut best_memory = None;

    for candidate in &candidates {
        let sim = similarity(content, &candidate.content);
        if sim > best_sim {
            best_sim = sim;
            best_memory = Some(candidate);
        }
    }

    // M6: Randomized threshold exploration (5% of the time, offset threshold by ±0.1)
    // This creates A/B test data for causal inference on optimal thresholds.
    let (effective_threshold, is_exploration) = m6_explore_threshold(similarity_threshold);

    if best_sim > effective_threshold {
        if let Some(memory) = best_memory {
            let age_days = (chrono::Utc::now() - memory.created_at).num_days();
            // Log exploration outcome for M6 learning
            if is_exploration {
                m6_log_outcome(store, best_sim, effective_threshold, similarity_threshold, true);
            }
            if age_days < time_window_days {
                return Ok(DedupAction::MergeInto(memory.id.clone()));
            } else {
                return Ok(DedupAction::Supersede(memory.id.clone()));
            }
        }
    }

    // Gray zone: 0.5 <= sim < threshold — flag for LLM dedup if available
    if best_sim >= 0.5 {
        if let Some(memory) = best_memory {
            if is_exploration {
                m6_log_outcome(store, best_sim, effective_threshold, similarity_threshold, false);
            }
            return Ok(DedupAction::GrayZone(memory.id.clone(), best_sim));
        }
    }

    // Log exploration non-match for control group
    if is_exploration && best_sim > 0.3 {
        m6_log_outcome(store, best_sim, effective_threshold, similarity_threshold, false);
    }

    Ok(DedupAction::CreateNew)
}

/// M6: Randomized threshold exploration.
/// With 5% probability, offset the threshold by a random amount in [-0.1, +0.1].
/// Returns (effective_threshold, is_exploration).
fn m6_explore_threshold(base_threshold: f32) -> (f32, bool) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let count = COUNTER.fetch_add(1, Ordering::Relaxed);
    // Deterministic pseudo-random: explore on ~5% of calls
    let hash = count.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(0x517cc1b727220a95);
    let explore = (hash % 20) == 0; // 5% probability

    if !explore {
        return (base_threshold, false);
    }

    // Random offset in [-0.1, +0.1]
    let offset_bits = ((hash >> 16) % 201) as f32 / 1000.0 - 0.1; // [-0.100, +0.100]
    let effective = (base_threshold + offset_bits).clamp(0.30, 0.95);
    (effective, true)
}

/// M6: Log threshold exploration outcome as feedback event.
fn m6_log_outcome(store: &SqliteStore, sim: f32, used_threshold: f32, base_threshold: f32, was_dedup: bool) {
    let payload = serde_json::json!({
        "similarity": sim,
        "threshold_used": used_threshold,
        "threshold_base": base_threshold,
        "offset": used_threshold - base_threshold,
        "was_dedup": was_dedup,
    });
    let _ = crate::store::adaptive::emit_event(store.conn(), crate::store::adaptive::FeedbackEvent {
        event_type: crate::store::adaptive::EventType::ParamUpdate,
        request_id: None,
        memory_id: None,
        concept_id: None,
        query: Some(format!("m6_explore:{sim:.3}")),
        query_type: Some("threshold_exploration".to_string()),
        topic: None,
        payload: Some(payload),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jaccard_identical() {
        let text = "the quick brown fox jumps over the lazy dog";
        assert!((jaccard_similarity(text, text) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_jaccard_disjoint() {
        let a = "alpha beta gamma delta";
        let b = "one two three four";
        assert!((jaccard_similarity(a, b) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_jaccard_partial() {
        // 2 shared tokens out of 6 unique tokens
        let a = "apple banana cherry";
        let b = "apple banana date";
        let sim = jaccard_similarity(a, b);
        // intersection = {apple, banana} = 2, union = {apple, banana, cherry, date} = 4
        // 2/4 = 0.5
        assert!((sim - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_jaccard_empty() {
        assert!((jaccard_similarity("", "") - 0.0).abs() < f32::EPSILON);
        assert!((jaccard_similarity("hello", "") - 0.0).abs() < f32::EPSILON);
        assert!((jaccard_similarity("", "world") - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_jaccard_strips_punctuation() {
        // "pool" vs "pool." should match after stripping punctuation
        let a = "database connection pool";
        let b = "database connection pool.";
        assert!((jaccard_similarity(a, b) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_containment_subset() {
        // Short text is fully contained in longer text
        let long = "Fixed OOM bug by closing database connection pool properly";
        let short = "Fixed OOM bug by closing database connection pool";
        let sim = containment_similarity(long, short);
        assert!(sim > 0.95, "containment should be ~1.0, got {sim}");
    }

    #[test]
    fn test_containment_disjoint() {
        let a = "alpha beta gamma";
        let b = "one two three";
        assert!((containment_similarity(a, b) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_similarity_picks_best() {
        // Jaccard is low (0.65) but containment is high (1.0)
        let a = "Fixed OOM bug by closing database connection pool properly. The issue was that connections were not being released back to the pool after query execution.";
        let b = "Fixed OOM bug by closing database connection pool. Connections were not released back after query.";
        let sim = similarity(a, b);
        assert!(sim > 0.70, "similarity should be > 0.70 (containment dominates), got {sim}");
    }
}
