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
    /// Canonical support signal: support_count / (support_count + 1)
    pub canonical_support: f32,
    /// Canonical source diversity signal: diversity / (diversity + 1)
    pub source_diversity: f32,
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
    // --- M3 survival feature (v0.17) ---
    /// Cluster-level Kaplan-Meier survival probability at current days-since-last-access.
    /// Ranges [0, 1]; 0.5 when no cluster curve is available (neutral fallback).
    /// Distinct from `strength` (individual decay): captures whether memories in
    /// this semantic cluster tend to remain relevant at this age.
    pub cluster_survival: f32,
}

/// Learned weights for the linear scoring model (19 features).
///
/// ## Replay-safety watermarks (v0.25.2)
///
/// `last_access_event_id` and `last_recall_event_id` are the highest
/// `feedback_events.id` from the `recall_access` and `recall_complete`
/// streams whose gradient effect is **already durable in this row**.
/// They are persisted atomically with the weights via the same
/// `UPDATE metadata SET value = ?` statement.
///
/// The reranker consumer loop uses these watermarks to filter peeked
/// events strictly by `id > last_*_event_id`. If the per-consumer
/// `commit_offset` call fails after `save_weights` succeeds, the next
/// peek re-surfaces the same events, and the watermark filter drops
/// them — preserving "exactly-once gradient application" without
/// folding the consumer into a single transaction.
///
/// The weights table effectively owns the source-of-truth watermark;
/// `consumer_offsets` is a redundant best-effort cache (kept for
/// compatibility / observability). Replay-drain on startup is
/// implicit: `peek_events` already filters by `consumer_offsets`, then
/// the watermark filter further drops any event with `id <=
/// last_*_event_id`. Composition gives the effective watermark
/// `id > max(consumer_offsets, weights)`.
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
    #[serde(default = "default_w_small")]
    pub w_canonical_support: f32,
    #[serde(default = "default_w_small")]
    pub w_source_diversity: f32,
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
    /// M3 cluster survival weight. Defaults to 0.02.
    #[serde(default = "default_w_small")]
    pub w_cluster_survival: f32,

    /// v0.25.2 replay-safety watermark for the `reranker_weights`
    /// consumer (`recall_access` event stream). Highest event id whose
    /// gradient effect is durable in this row. `0` on fresh install /
    /// pre-v0.25.2 snapshots.
    #[serde(default)]
    pub last_access_event_id: i64,

    /// v0.25.2 replay-safety watermark for the
    /// `reranker_weights_recall` consumer (`recall_complete` event
    /// stream). Highest event id whose candidate-feature contribution
    /// is durable in this row. `0` on fresh install / pre-v0.25.2
    /// snapshots.
    #[serde(default)]
    pub last_recall_event_id: i64,
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
        w_fts: 0.14,
        w_vec: 0.14,
        w_kg: 0.08,
        w_episode: 0.07,
        w_recency: 0.07,
        w_access: 0.05,
        w_strength: 0.05,
        w_importance: 0.05,
        w_keyword: 0.05,
        w_topic_match: 0.03,
        w_brevity: 0.02,
        w_channel_coverage: 0.04,
        w_canonical_support: 0.02,
        w_source_diversity: 0.02,
        w_usage_recency: 0.03,
        w_connectivity: 0.03,
        w_concept_richness: 0.03,
        w_tier_score: 0.03,
        w_is_current: 0.03,
        w_cluster_survival: 0.02,
        last_access_event_id: 0,
        last_recall_event_id: 0,
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
        + w.w_canonical_support * f.canonical_support
        + w.w_source_diversity * f.source_diversity
        + w.w_usage_recency * usage_recency_factor
        + w.w_connectivity * f.connectivity
        + w.w_concept_richness * f.concept_richness
        + w.w_tier_score * f.tier_score
        + w.w_is_current * f.is_current
        + w.w_cluster_survival * f.cluster_survival;

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
    let weights = match result {
        Ok(json) => serde_json::from_str(&json).unwrap_or_else(|_| default_weights()),
        Err(_) => default_weights(),
    };
    // Sanity check: weights should not be wildly out of range (catch corruption)
    let sum: f32 = weights.w_fts
        + weights.w_vec
        + weights.w_kg
        + weights.w_episode
        + weights.w_recency
        + weights.w_access
        + weights.w_strength
        + weights.w_importance
        + weights.w_keyword
        + weights.w_topic_match
        + weights.w_brevity
        + weights.w_channel_coverage
        + weights.w_canonical_support
        + weights.w_source_diversity
        + weights.w_usage_recency
        + weights.w_connectivity
        + weights.w_concept_richness
        + weights.w_tier_score
        + weights.w_is_current
        + weights.w_cluster_survival;
    if sum <= 0.0 || sum > 5.0 || sum.is_nan() {
        tracing::warn!("rerank weights sum is {sum:.3} (out of valid range), using defaults");
        return default_weights();
    }
    weights
}

/// Persist rerank weights to the metadata table.
///
/// Convenience shim that delegates to [`save_weights_cas`] with the
/// `expected_*` watermarks taken from `weights` itself — i.e. it
/// trusts the caller to have set the new watermark fields to the
/// values it just observed under load. New code feeding the consumer
/// loop should prefer [`save_weights_cas`] directly so the
/// observed-vs-write watermark transition is explicit AND so the
/// boolean return value can gate downstream `commit_offset` calls.
///
/// The bool result is intentionally discarded here; legacy callers
/// (`adaptive_status` snapshot serialization, default-only test code)
/// don't have a downstream offset to gate.
pub fn save_weights(conn: &rusqlite::Connection, weights: &RerankWeights) {
    let _ = save_weights_cas(
        conn,
        weights,
        weights.last_access_event_id,
        weights.last_recall_event_id,
    );
}

/// v0.25.2: Persist rerank weights with CAS protection over both
/// per-stream watermarks (`recall_access` + `recall_complete`).
///
/// `expected_access_id` / `expected_recall_id` are the watermarks the
/// caller observed when it [`load_weights`]'d. The UPDATE only succeeds
/// when the row's current watermarks still match — otherwise a
/// concurrent worker has already incorporated newer events and our
/// gradient step is stale. On a CAS miss we log + skip (no panic);
/// the next pipeline tick will re-load and re-attempt.
///
/// On first run the row is absent → `INSERT` (the legacy
/// `ON CONFLICT(key) DO UPDATE` path is gone deliberately so the CAS
/// invariant can't be silently bypassed by concurrent inserts; if two
/// fresh-install workers race the loser's INSERT errors with UNIQUE
/// and we log + skip the same way).
///
/// `COALESCE(json_extract(...), -1)` handles two migration cases:
/// (a) pre-v0.25.2 rows whose JSON lacks the watermark fields — the
/// caller's observed `0` matches `-1` → predicate fails on first hit,
/// then succeeds after the next load surfaces the newly-written `0`.
/// To unblock that boot transition the first save after upgrade
/// always passes `expected_*=0` (consumer loads via [`load_weights`]
/// which `#[serde(default)]`s the absent fields to `0`).
/// Returns `true` when the weights row was durably written (CAS hit
/// or first-time insert), `false` when the CAS predicate missed (a
/// concurrent worker raced us) or any underlying SQL error occurred.
///
/// **Callers MUST gate their `commit_offset` on the return value** —
/// advancing the consumer offset after a CAS miss permanently drops
/// events whose gradient never made it into the weights row. Codex
/// R1 (v0.25.2) caught this regression in `run_reranker_weight_learning`.
#[must_use = "the caller must gate commit_offset on the CAS result; \
              advancing the consumer offset after a CAS miss permanently \
              drops the events that were peeked but not persisted"]
pub fn save_weights_cas(
    conn: &rusqlite::Connection,
    weights: &RerankWeights,
    expected_access_id: i64,
    expected_recall_id: i64,
) -> bool {
    let json = serde_json::to_string(weights).expect("RerankWeights serialization cannot fail");

    // CAS UPDATE: only succeeds when both stored watermarks equal what
    // we observed at load time. Atomic with the weights write because
    // it's a single statement on a single row.
    let updated = match conn.execute(
        "UPDATE metadata
            SET value = ?1
          WHERE key = 'rerank_weights'
            AND COALESCE(json_extract(value, '$.last_access_event_id'), 0) = ?2
            AND COALESCE(json_extract(value, '$.last_recall_event_id'), 0) = ?3",
        rusqlite::params![json, expected_access_id, expected_recall_id],
    ) {
        Ok(rows) => rows,
        Err(e) => {
            // Codex R3 G7: when the stored row holds malformed JSON,
            // SQLite's `json_extract` raises an error and our CAS write
            // can never succeed — the row stays stuck forever and
            // reranker learning livelocks even though `load_weights`
            // already silently falls back to defaults on read. Detect
            // the malformed-JSON failure mode and recover by deleting
            // the corrupt row, then fall through to the first-install
            // INSERT path below (treat-as-fresh contract is consistent
            // with what `load_weights` already returns to consumers).
            let err_str = e.to_string();
            let is_malformed_json = err_str.contains("malformed JSON")
                || err_str.contains("malformed json");
            if is_malformed_json {
                tracing::warn!(
                    error = %e,
                    "save_weights CAS hit malformed JSON in stored row; \
                     deleting corrupt row + retrying as first-install INSERT"
                );
                if let Err(del_err) = conn.execute(
                    "DELETE FROM metadata WHERE key = 'rerank_weights'",
                    [],
                ) {
                    tracing::warn!(
                        error = %del_err,
                        "save_weights recovery DELETE failed; row stays corrupt"
                    );
                    return false;
                }
                0 // fall through to !exists → INSERT branch
            } else {
                tracing::warn!("save_weights CAS update failed: {}", e);
                return false;
            }
        }
    };

    if updated > 0 {
        return true; // happy path
    }

    // No row updated. Either the row doesn't exist yet (first save) or
    // a concurrent worker bumped the watermark.
    let exists: bool = conn
        .query_row(
            "SELECT COUNT(*) > 0 FROM metadata WHERE key = 'rerank_weights'",
            [],
            |r| r.get(0),
        )
        .unwrap_or(false);

    if !exists {
        // First-time insert. Use INSERT (not ON CONFLICT) so a race
        // with another fresh-install worker fails loudly via UNIQUE
        // constraint rather than silently last-writes-wins.
        return match conn.execute(
            "INSERT INTO metadata (key, value) VALUES ('rerank_weights', ?1)",
            rusqlite::params![&serde_json::to_string(weights)
                .expect("RerankWeights serialization cannot fail")],
        ) {
            Ok(_) => true,
            Err(e) => {
                tracing::warn!("save_weights initial insert failed: {}", e);
                false
            }
        };
    }

    // Row exists but our CAS predicate missed → another worker won the
    // race. Log + skip; next pipeline tick will re-load and re-try.
    tracing::warn!(
        expected_access_id,
        expected_recall_id,
        "save_weights CAS predicate missed (concurrent writer); skipping cycle"
    );
    false
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
            canonical_support: 0.5,
            source_diversity: 0.5,
            usage_recency: 2.0,
            connectivity: 0.3,
            concept_richness: 0.4,
            tier_score: 1.0,
            is_current: 1.0,
            cluster_survival: 0.8,
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
            canonical_support: 0.8,
            source_diversity: 0.7,
            usage_recency: 0.5,
            connectivity: 0.8,
            concept_richness: 1.0,
            tier_score: 1.0,
            is_current: 1.0,
            cluster_survival: 0.8,
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
            canonical_support: 0.0,
            source_diversity: 0.0,
            usage_recency: 30.0,
            connectivity: 0.0,
            concept_richness: 0.0,
            tier_score: 0.0,
            is_current: 0.0,
            cluster_survival: 0.2,
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
            canonical_support: 1.0,
            source_diversity: 1.0,
            usage_recency: 0.0,
            connectivity: 1.0,
            concept_richness: 1.0,
            tier_score: 1.0,
            is_current: 1.0,
            cluster_survival: 1.0,
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
            + w.w_canonical_support
            + w.w_source_diversity
            + w.w_usage_recency
            + w.w_connectivity
            + w.w_concept_richness
            + w.w_tier_score
            + w.w_is_current
            + w.w_cluster_survival;
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
            (final_w.w_canonical_support - initial.w_canonical_support).abs() < tolerance,
            "w_canonical_support drifted"
        );
        assert!(
            (final_w.w_source_diversity - initial.w_source_diversity).abs() < tolerance,
            "w_source_diversity drifted"
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

    // ── v0.25.2 watermark + CAS tests ────────────────────────────────

    /// Helper: create an in-memory connection with the metadata table.
    fn ws_conn() -> rusqlite::Connection {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute_batch("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
            .unwrap();
        conn
    }

    /// Read the persisted watermarks from the row using `json_extract`,
    /// matching the same predicate the CAS UPDATE uses. This exercises
    /// the "atomic with weights" guarantee — the test does not parse
    /// the row through `RerankWeights` to avoid hiding any
    /// serialization-vs-storage gap.
    fn read_persisted_watermarks(conn: &rusqlite::Connection) -> (Option<i64>, Option<i64>) {
        conn.query_row(
            "SELECT json_extract(value, '$.last_access_event_id'),
                    json_extract(value, '$.last_recall_event_id')
               FROM metadata WHERE key = 'rerank_weights'",
            [],
            |row| Ok((row.get::<_, Option<i64>>(0)?, row.get::<_, Option<i64>>(1)?)),
        )
        .unwrap_or((None, None))
    }

    #[test]
    fn reranker_save_weights_writes_event_id_atomically() {
        // v0.25.2: `save_weights_cas` writes the weights and BOTH
        // watermarks in the same single-row UPDATE. After the save
        // returns, the persisted JSON row must contain the new
        // `last_access_event_id` AND `last_recall_event_id` AND the
        // new weight values — there is no observable in-between state.
        let conn = ws_conn();
        let mut w = default_weights();
        w.w_fts = 0.99;
        w.last_access_event_id = 42;
        w.last_recall_event_id = 17;

        // First save (row absent → INSERT path).
        assert!(save_weights_cas(&conn, &w, 0, 0), "first install should succeed");

        let (acc, rec) = read_persisted_watermarks(&conn);
        assert_eq!(acc, Some(42), "last_access_event_id missing from row");
        assert_eq!(rec, Some(17), "last_recall_event_id missing from row");

        let loaded = load_weights(&conn);
        // Sanity: weights and watermarks survive the roundtrip.
        // (`load_weights` falls back to defaults if `sum > 5.0` — the
        // bumped `w_fts = 0.99` keeps the sum well within range.)
        assert!((loaded.w_fts - 0.99).abs() < 1e-6);
        assert_eq!(loaded.last_access_event_id, 42);
        assert_eq!(loaded.last_recall_event_id, 17);
    }

    #[test]
    fn reranker_cas_rejects_stale_observed_watermark() {
        // v0.25.2: complementary to the consumer-loop replay-safety
        // test in `ops/adaptive::test_reranker_replay_safe_when_commit_offset_lost`.
        // Here we exercise the row-level invariant directly: once the
        // watermark in the row has been bumped to N, any caller that
        // observed an older watermark and tries to write must be
        // rejected by the CAS predicate. This is the storage-side
        // half of the protection; the peek-side filter is the
        // higher-level half.
        let conn = ws_conn();
        let mut w = default_weights();
        assert!(save_weights_cas(&conn, &w, 0, 0), "first install should succeed"); // watermarks = 0

        // Pretend cycle 1 absorbed events through id=5 on both streams.
        w.w_fts = 0.20;
        w.last_access_event_id = 5;
        w.last_recall_event_id = 5;
        assert!(save_weights_cas(&conn, &w, 0, 0), "cycle 1 CAS should hit (expected=0,0 matches)");
        let after_cycle1 = read_persisted_watermarks(&conn);
        assert_eq!(after_cycle1, (Some(5), Some(5)));

        // Cycle 2 starts with a stale view (commit_offset failed last
        // cycle, peek re-surfaces events 1..=5). A naive caller that
        // observed watermark=0 from a stale `consumer_offsets` and
        // tried to re-apply with `expected=(0,0)` MUST be rejected by
        // the CAS predicate → row unchanged.
        let stale_attempt_ftsf = 0.55_f32;
        let mut stale = w.clone();
        stale.w_fts = stale_attempt_ftsf;
        // Note: the stale caller does NOT bump the watermark (it
        // thinks it's the first-ever apply), so it passes
        // `expected=(0, 0)`.
        assert!(
            !save_weights_cas(&conn, &stale, 0, 0),
            "stale CAS should report miss so the caller can refrain from advancing offset"
        );

        // The CAS predicate misses → row still has cycle-1 weights.
        let row_value: String = conn
            .query_row(
                "SELECT value FROM metadata WHERE key = 'rerank_weights'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&row_value).unwrap();
        let w_fts = parsed.get("w_fts").and_then(|v| v.as_f64()).unwrap();
        assert!(
            (w_fts - 0.20).abs() < 1e-6,
            "stale CAS write must be rejected; w_fts in row = {w_fts} (expected 0.20)"
        );
    }

    #[test]
    fn reranker_concurrent_save_cas_protects() {
        // v0.25.2: simulate two workers who both load weights at the
        // same time (both observe watermark=(0, 0)), each compute a
        // gradient step, then both attempt to save with their own
        // bumped watermark. Worker A writes first (CAS hits); Worker
        // B's CAS predicate misses (the row's watermark is no longer
        // 0). Worker B's write is dropped silently — the spec says
        // "log + skip the cycle (do not panic)".
        let conn = ws_conn();
        let initial = default_weights();
        assert!(save_weights_cas(&conn, &initial, 0, 0), "first install should succeed");

        // Worker A: observed (0, 0), bumps to (10, 20), updates w_fts.
        let mut worker_a = initial.clone();
        worker_a.w_fts = 0.30;
        worker_a.last_access_event_id = 10;
        worker_a.last_recall_event_id = 20;
        assert!(save_weights_cas(&conn, &worker_a, 0, 0), "worker A CAS should hit");

        let after_a = read_persisted_watermarks(&conn);
        assert_eq!(after_a, (Some(10), Some(20)), "worker A must commit first");

        // Worker B: observed (0, 0) at the same time as A, computed
        // its own gradient step, attempts to commit. CAS predicate
        // checks against the row's CURRENT watermarks (10, 20) ≠ B's
        // expected (0, 0) → rejection.
        let mut worker_b = initial.clone();
        worker_b.w_fts = 0.99;
        worker_b.last_access_event_id = 7;
        worker_b.last_recall_event_id = 11;
        assert!(
            !save_weights_cas(&conn, &worker_b, 0, 0),
            "worker B CAS should miss so it knows not to advance its consumer offsets"
        );

        // Row must still reflect Worker A's write — Worker B silently
        // skipped its cycle.
        let row_value: String = conn
            .query_row(
                "SELECT value FROM metadata WHERE key = 'rerank_weights'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&row_value).unwrap();
        let w_fts = parsed.get("w_fts").and_then(|v| v.as_f64()).unwrap();
        assert!(
            (w_fts - 0.30).abs() < 1e-6,
            "worker B's stale CAS must be rejected; w_fts = {w_fts} (expected 0.30)"
        );

        let after_b = read_persisted_watermarks(&conn);
        assert_eq!(
            after_b,
            (Some(10), Some(20)),
            "Worker B must not have bumped the watermark"
        );
    }
}
