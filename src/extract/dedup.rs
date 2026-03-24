use std::collections::HashSet;

use crate::store::SqliteStore;
use crate::types::{MemoryStore, ReinResult};

/// Jaccard similarity between two texts (token-level).
pub fn jaccard_similarity(a: &str, b: &str) -> f32 {
    let set_a: HashSet<&str> = a.split_whitespace().collect();
    let set_b: HashSet<&str> = b.split_whitespace().collect();
    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();
    if union == 0 {
        return 0.0;
    }
    intersection as f32 / union as f32
}

/// What to do when storing a potentially duplicate memory.
pub enum DedupAction {
    /// No duplicate found, create new memory.
    CreateNew,
    /// Similar content within time window, merge into existing memory.
    MergeInto(String),
    /// Similar content but older than time window, supersede old memory.
    Supersede(String),
}

/// Check for duplicate memories using FTS search and Jaccard similarity.
///
/// Given the store, a topic, and content text, search existing memories in that
/// topic using FTS. For the best match, compute Jaccard similarity.
/// - If > threshold and time diff < time_window_days -> MergeInto(id)
/// - If > threshold and time diff >= time_window_days -> Supersede(id)
/// - Otherwise -> CreateNew
pub async fn check_dedup(
    store: &SqliteStore,
    topic: &str,
    content: &str,
    similarity_threshold: f32,
    time_window_days: i64,
) -> ReinResult<DedupAction> {
    // Extract key tokens from content for FTS query (take first few words)
    let query_tokens: Vec<&str> = content.split_whitespace().take(10).collect();
    if query_tokens.is_empty() {
        return Ok(DedupAction::CreateNew);
    }
    let query = query_tokens.join(" ");

    // Search existing memories in this topic
    let candidates = store.search_fts(&query, Some(topic), 10).await?;

    if candidates.is_empty() {
        return Ok(DedupAction::CreateNew);
    }

    // Find best match by Jaccard similarity
    let mut best_sim = 0.0f32;
    let mut best_memory = None;

    for candidate in &candidates {
        let sim = jaccard_similarity(content, &candidate.content);
        if sim > best_sim {
            best_sim = sim;
            best_memory = Some(candidate);
        }
    }

    if best_sim > similarity_threshold {
        if let Some(memory) = best_memory {
            let age_days = (chrono::Utc::now() - memory.created_at).num_days();
            if age_days < time_window_days {
                return Ok(DedupAction::MergeInto(memory.id.clone()));
            } else {
                return Ok(DedupAction::Supersede(memory.id.clone()));
            }
        }
    }

    Ok(DedupAction::CreateNew)
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
}
