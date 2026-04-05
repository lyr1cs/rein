//! Multi-feature reranker for post-fusion result ordering.
//! Weights are learned from M1/M2 feedback data; defaults are hand-tuned.

use serde::{Deserialize, Serialize};

/// Retrieval and metadata features extracted from each candidate memory after fusion.
#[derive(Debug, Clone)]
pub struct RerankFeatures {
    /// Normalized BM25 score [0,1]
    pub fts_score: f32,
    /// Normalized vector similarity [0,1]
    pub vec_score: f32,
    /// KG channel score [0,1]
    pub kg_score: f32,
    /// Episode/session channel score [0,1]
    pub episode_score: f32,
    /// Days since memory creation
    pub recency_days: f32,
    /// Times previously accessed
    pub access_count: u32,
    /// Current memory strength [0,1]
    pub strength: f32,
    /// Importance weight: Critical=1.0, High=0.8, Medium=0.6, Low=0.4
    pub importance_weight: f32,
    /// Fraction of query keywords found in memory keywords+content [0,1]
    pub keyword_overlap: f32,
    // --- New features (v0.9.1) ---
    /// Whether memory topic exactly matches a query word [0 or 1]
    pub topic_match: f32,
    /// Normalized content length: shorter = more precise. 1/(1 + chars/500)
    pub brevity: f32,
    /// Number of channels that found this memory (1-3), normalized to [0.33, 1.0]
    pub channel_coverage: f32,
    /// Days since last accessed (freshness of usage, not creation)
    pub usage_recency: f32,
    // --- Adaptive engine features (v0.9.2) ---
    /// Number of related memories (graph connectivity). Normalized: min(related, 10) / 10
    pub connectivity: f32,
    /// Number of linked concepts. Normalized: min(concepts, 5) / 5
    pub concept_richness: f32,
    /// Tier score: hot=1.0, warm=0.5, cold=0.0
    pub tier_score: f32,
    /// Whether memory has been superseded (0 or 1, penalty for outdated)
    pub is_current: f32,
}

/// Learned weights for the linear scoring model (17 features).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankWeights {
    pub w_fts: f32,
    pub w_vec: f32,
    pub w_kg: f32,
    #[serde(default = "default_w_episode")]
    pub w_episode: f32,
    pub w_recency: f32,
    pub w_access: f32,
    pub w_strength: f32,
    pub w_importance: f32,
    pub w_keyword: f32,
    #[serde(default = "default_w_topic_match")]
    pub w_topic_match: f32,
    #[serde(default = "default_w_brevity")]
    pub w_brevity: f32,
    #[serde(default = "default_w_channel_coverage")]
    pub w_channel_coverage: f32,
    #[serde(default = "default_w_usage_recency")]
    pub w_usage_recency: f32,
    #[serde(default = "default_w_small")]
    pub w_connectivity: f32,
    #[serde(default = "default_w_small")]
    pub w_concept_richness: f32,
    #[serde(default = "default_w_small")]
    pub w_tier_score: f32,
    #[serde(default = "default_w_small")]
    pub w_is_current: f32,
}

fn default_w_topic_match() -> f32 {
    0.03
}
fn default_w_episode() -> f32 {
    0.07
}
fn default_w_brevity() -> f32 {
    0.02
}
fn default_w_channel_coverage() -> f32 {
    0.04
}
fn default_w_usage_recency() -> f32 {
    0.03
}
fn default_w_small() -> f32 {
    0.02
}

/// Hand-tuned default weights (sum to 1.0). Retrieval signals dominate.
pub fn default_weights() -> RerankWeights {
    RerankWeights {
        w_fts: 0.16,
        w_vec: 0.16,
        w_kg: 0.08,
        w_episode: 0.07,
        w_recency: 0.07,
        w_access: 0.05,
        w_strength: 0.07,
        w_importance: 0.05,
        w_keyword: 0.05,
        w_topic_match: 0.03,
        w_brevity: 0.02,
        w_channel_coverage: 0.04,
        w_usage_recency: 0.03,
        w_connectivity: 0.03,
        w_concept_richness: 0.03,
        w_tier_score: 0.03,
        w_is_current: 0.03,
    }
}

/// Compute a weighted linear rerank score. Higher is better.
///
/// Recency is converted to a decay factor: `1.0 / (1.0 + days / 7.0)` so recent
/// memories score higher. Access count is capped at 20 and normalized to [0,1].
/// Final score is clamped to [0, 2] to allow strong multi-signal matches to
/// exceed 1.0.
pub fn rerank_score(f: &RerankFeatures, w: &RerankWeights) -> f32 {
    let recency_factor = 1.0 / (1.0 + f.recency_days / 7.0);
    let access_factor = (f.access_count.min(20) as f32) / 20.0;

    let usage_recency_factor = 1.0 / (1.0 + f.usage_recency / 14.0);

    let score = w.w_fts * f.fts_score
        + w.w_vec * f.vec_score
        + w.w_kg * f.kg_score
        + w.w_episode * f.episode_score
        + w.w_recency * recency_factor
        + w.w_access * access_factor
        + w.w_strength * f.strength
        + w.w_importance * f.importance_weight
        + w.w_keyword * f.keyword_overlap
        + w.w_topic_match * f.topic_match
        + w.w_brevity * f.brevity
        + w.w_channel_coverage * f.channel_coverage
        + w.w_usage_recency * usage_recency_factor
        + w.w_connectivity * f.connectivity
        + w.w_concept_richness * f.concept_richness
        + w.w_tier_score * f.tier_score
        + w.w_is_current * f.is_current;

    score.clamp(0.0, 2.0)
}

/// Compute fraction of query words that appear in memory keywords or content.
///
/// Query is lowercased and split on whitespace. Each query word is checked against
/// the keyword list (case-insensitive exact match) and the content (case-insensitive
/// substring). Returns 0.0 if the query has no words.
pub fn compute_keyword_overlap(
    query: &str,
    memory_keywords: &[String],
    memory_content: &str,
) -> f32 {
    let query_lower = query.to_lowercase();
    let words: Vec<&str> = query_lower.split_whitespace().collect();
    if words.is_empty() {
        return 0.0;
    }

    let keywords_lower: Vec<String> = memory_keywords.iter().map(|k| k.to_lowercase()).collect();
    let content_lower = memory_content.to_lowercase();

    let matches = words
        .iter()
        .filter(|w| keywords_lower.iter().any(|k| k == *w) || content_lower.contains(*w))
        .count();

    matches as f32 / words.len() as f32
}

/// Like [`compute_keyword_overlap`] but accepts pre-split, pre-lowercased query words
/// to avoid re-lowercasing the query on every call in a reranking loop.
pub fn compute_keyword_overlap_with_words(
    query_words: &[&str],
    memory_keywords: &[String],
    memory_content: &str,
) -> f32 {
    if query_words.is_empty() {
        return 0.0;
    }
    let keywords_lower: Vec<String> = memory_keywords.iter().map(|k| k.to_lowercase()).collect();
    let content_lower = memory_content.to_lowercase();
    let matches = query_words
        .iter()
        .filter(|w| keywords_lower.iter().any(|k| k == **w) || content_lower.contains(**w))
        .count();
    matches as f32 / query_words.len() as f32
}

/// Load rerank weights from the metadata table, falling back to defaults.
pub fn load_weights(conn: &rusqlite::Connection) -> RerankWeights {
    let result: Result<String, _> = conn.query_row(
        "SELECT value FROM metadata WHERE key = 'rerank_weights'",
        rusqlite::params![],
        |row| row.get(0),
    );
    match result {
        Ok(json) => serde_json::from_str(&json).unwrap_or_else(|_| default_weights()),
        Err(_) => default_weights(),
    }
}

/// Persist rerank weights to the metadata table.
pub fn save_weights(conn: &rusqlite::Connection, weights: &RerankWeights) {
    let json = serde_json::to_string(weights).expect("RerankWeights serialization cannot fail");
    if let Err(e) = conn.execute(
        "INSERT INTO metadata (key, value) VALUES ('rerank_weights', ?1)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        rusqlite::params![json],
    ) {
        tracing::warn!("save_weights failed: {}", e);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rerank_score_positive() {
        let f = RerankFeatures {
            fts_score: 0.8,
            vec_score: 0.6,
            kg_score: 0.4,
            episode_score: 0.0,
            recency_days: 1.0,
            access_count: 5,
            strength: 0.9,
            importance_weight: 1.0,
            keyword_overlap: 0.5,
            topic_match: 1.0,
            brevity: 0.5,
            channel_coverage: 0.67,
            usage_recency: 2.0,
            connectivity: 0.3,
            concept_richness: 0.4,
            tier_score: 1.0,
            is_current: 1.0,
        };
        let w = default_weights();
        let score = rerank_score(&f, &w);
        assert!(score > 0.0);
    }

    #[test]
    fn test_higher_features_higher_score() {
        let w = default_weights();
        let high = RerankFeatures {
            fts_score: 1.0,
            vec_score: 1.0,
            kg_score: 1.0,
            episode_score: 0.8,
            recency_days: 0.5,
            access_count: 10,
            strength: 1.0,
            importance_weight: 1.0,
            keyword_overlap: 1.0,
            topic_match: 1.0,
            brevity: 1.0,
            channel_coverage: 1.0,
            usage_recency: 0.5,
            connectivity: 0.8,
            concept_richness: 1.0,
            tier_score: 1.0,
            is_current: 1.0,
        };
        let low = RerankFeatures {
            fts_score: 0.1,
            vec_score: 0.1,
            kg_score: 0.0,
            episode_score: 0.0,
            recency_days: 30.0,
            access_count: 0,
            strength: 0.3,
            importance_weight: 0.4,
            keyword_overlap: 0.0,
            topic_match: 0.0,
            brevity: 0.2,
            channel_coverage: 0.33,
            usage_recency: 30.0,
            connectivity: 0.0,
            concept_richness: 0.0,
            tier_score: 0.0,
            is_current: 0.0,
        };
        assert!(rerank_score(&high, &w) > rerank_score(&low, &w));
    }

    #[test]
    fn test_keyword_overlap() {
        let overlap = compute_keyword_overlap(
            "rust memory system",
            &["rust".to_string(), "performance".to_string()],
            "memory management in rust",
        );
        assert!(overlap > 0.5); // "rust" and "memory" match
    }

    #[test]
    fn test_keyword_overlap_empty_query() {
        let overlap = compute_keyword_overlap("", &["rust".to_string()], "some content");
        assert_eq!(overlap, 0.0);
    }

    #[test]
    fn test_keyword_overlap_no_matches() {
        let overlap = compute_keyword_overlap(
            "quantum physics",
            &["rust".to_string()],
            "memory management",
        );
        assert_eq!(overlap, 0.0);
    }

    #[test]
    fn test_score_clamped() {
        // Even with all max features, score should not exceed 2.0
        let f = RerankFeatures {
            fts_score: 1.0,
            vec_score: 1.0,
            kg_score: 1.0,
            episode_score: 1.0,
            recency_days: 0.0,
            access_count: 20,
            strength: 1.0,
            importance_weight: 1.0,
            keyword_overlap: 1.0,
            topic_match: 1.0,
            brevity: 1.0,
            channel_coverage: 1.0,
            usage_recency: 0.0,
            connectivity: 1.0,
            concept_richness: 1.0,
            tier_score: 1.0,
            is_current: 1.0,
        };
        let w = default_weights();
        let score = rerank_score(&f, &w);
        assert!(score <= 2.0);
    }

    #[test]
    fn test_default_weights_sum() {
        let w = default_weights();
        let sum = w.w_fts
            + w.w_vec
            + w.w_kg
            + w.w_episode
            + w.w_recency
            + w.w_access
            + w.w_strength
            + w.w_importance
            + w.w_keyword
            + w.w_topic_match
            + w.w_brevity
            + w.w_channel_coverage
            + w.w_usage_recency
            + w.w_connectivity
            + w.w_concept_richness
            + w.w_tier_score
            + w.w_is_current;
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Default weights should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn test_persistence_roundtrip() {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute(
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)",
            [],
        )
        .unwrap();

        let w = default_weights();
        save_weights(&conn, &w);
        let loaded = load_weights(&conn);

        assert!((loaded.w_fts - w.w_fts).abs() < 1e-6);
        assert!((loaded.w_vec - w.w_vec).abs() < 1e-6);
        assert!((loaded.w_kg - w.w_kg).abs() < 1e-6);
        assert!((loaded.w_episode - w.w_episode).abs() < 1e-6);
    }

    #[test]
    fn test_load_weights_fallback() {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute(
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)",
            [],
        )
        .unwrap();

        // No weights saved — should return defaults
        let loaded = load_weights(&conn);
        let defaults = default_weights();
        assert!((loaded.w_fts - defaults.w_fts).abs() < 1e-6);
    }

    #[test]
    fn test_weight_stability_no_drift() {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute_batch("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
            .unwrap();

        let initial = default_weights();
        save_weights(&conn, &initial);

        // 10 load/save roundtrips should not drift
        for _ in 0..10 {
            let loaded = load_weights(&conn);
            save_weights(&conn, &loaded);
        }

        let final_w = load_weights(&conn);
        let tolerance = 1e-6;
        assert!(
            (final_w.w_fts - initial.w_fts).abs() < tolerance,
            "w_fts drifted"
        );
        assert!(
            (final_w.w_vec - initial.w_vec).abs() < tolerance,
            "w_vec drifted"
        );
        assert!(
            (final_w.w_kg - initial.w_kg).abs() < tolerance,
            "w_kg drifted"
        );
        assert!(
            (final_w.w_episode - initial.w_episode).abs() < tolerance,
            "w_episode drifted"
        );
        assert!(
            (final_w.w_recency - initial.w_recency).abs() < tolerance,
            "w_recency drifted"
        );
        assert!(
            (final_w.w_access - initial.w_access).abs() < tolerance,
            "w_access drifted"
        );
        assert!(
            (final_w.w_strength - initial.w_strength).abs() < tolerance,
            "w_strength drifted"
        );
        assert!(
            (final_w.w_importance - initial.w_importance).abs() < tolerance,
            "w_importance drifted"
        );
        assert!(
            (final_w.w_keyword - initial.w_keyword).abs() < tolerance,
            "w_keyword drifted"
        );
        assert!(
            (final_w.w_topic_match - initial.w_topic_match).abs() < tolerance,
            "w_topic_match drifted"
        );
        assert!(
            (final_w.w_brevity - initial.w_brevity).abs() < tolerance,
            "w_brevity drifted"
        );
        assert!(
            (final_w.w_channel_coverage - initial.w_channel_coverage).abs() < tolerance,
            "w_channel_coverage drifted"
        );
        assert!(
            (final_w.w_usage_recency - initial.w_usage_recency).abs() < tolerance,
            "w_usage_recency drifted"
        );
        assert!(
            (final_w.w_connectivity - initial.w_connectivity).abs() < tolerance,
            "w_connectivity drifted"
        );
        assert!(
            (final_w.w_concept_richness - initial.w_concept_richness).abs() < tolerance,
            "w_concept_richness drifted"
        );
        assert!(
            (final_w.w_tier_score - initial.w_tier_score).abs() < tolerance,
            "w_tier_score drifted"
        );
        assert!(
            (final_w.w_is_current - initial.w_is_current).abs() < tolerance,
            "w_is_current drifted"
        );
    }
}
