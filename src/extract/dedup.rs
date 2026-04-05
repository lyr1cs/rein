use std::collections::HashSet;

use crate::store::SqliteStore;
use crate::types::{Memory, MemoryStore, ReinResult};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LexicalDedupScore {
    pub jaccard: f32,
    pub containment: f32,
    pub score: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CandidateScore {
    pub memory_id: String,
    pub lexical: LexicalDedupScore,
    pub topic_variant_match: bool,
    pub cluster_match: bool,
    pub recency_days: i64,
    pub final_score: f32,
}

/// Normalize text for similarity comparison: lowercase + strip punctuation.
fn normalize_tokens(text: &str) -> HashSet<String> {
    text.split_whitespace()
        .map(|t| {
            t.to_lowercase()
                .chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
        })
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

pub fn lexical_score(a: &str, b: &str) -> LexicalDedupScore {
    let jaccard = jaccard_similarity(a, b);
    let containment = containment_similarity(a, b);
    LexicalDedupScore {
        jaccard,
        containment,
        score: jaccard.max(containment),
    }
}

pub fn normalize_topic_key(topic: &str) -> String {
    let mut normalized = String::with_capacity(topic.len());
    let mut prev_sep = false;
    for ch in topic.trim().to_lowercase().chars() {
        if ch.is_alphanumeric() {
            normalized.push(ch);
            prev_sep = false;
        } else if !prev_sep && !normalized.is_empty() {
            normalized.push('-');
            prev_sep = true;
        }
    }
    normalized.trim_matches('-').to_string()
}

pub fn topics_are_variants(left: &str, right: &str) -> bool {
    normalize_topic_key(left) == normalize_topic_key(right)
}

pub fn score_candidate(
    topic: &str,
    content: &str,
    candidate: &Memory,
    cluster_id: Option<u32>,
) -> CandidateScore {
    let lexical = lexical_score(content, &candidate.content);
    let topic_variant_match = topics_are_variants(topic, &candidate.topic);
    let cluster_match = cluster_id.is_some() && candidate.cluster_id == cluster_id;
    let recency_days = (chrono::Utc::now() - candidate.created_at).num_days();
    let mut final_score = lexical.score;
    if topic_variant_match {
        final_score += 0.05;
    }
    if cluster_match {
        final_score += 0.05;
    }
    CandidateScore {
        memory_id: candidate.id.clone(),
        lexical,
        topic_variant_match,
        cluster_match,
        recency_days,
        final_score: final_score.clamp(0.0, 1.0),
    }
}

pub fn gray_zone_lower_bound(best_sim: f32, llm_budget_available: bool) -> f32 {
    if (0.35..0.50).contains(&best_sim) && llm_budget_available {
        0.35
    } else {
        0.50
    }
}

fn candidate_topics(store: &SqliteStore, topic: &str) -> ReinResult<Vec<String>> {
    let normalized = normalize_topic_key(topic);
    let mut topics = vec![topic.to_string()];
    for existing in store.list_topics()? {
        if existing != topic && normalize_topic_key(&existing) == normalized {
            topics.push(existing);
        }
    }
    Ok(topics)
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

    let mut seen = HashSet::new();
    let mut candidates = Vec::new();
    for candidate_topic in candidate_topics(store, topic)? {
        for memory in store
            .search_fts(&query, Some(&candidate_topic), 8)?
            .into_iter()
            .filter(|m| m.superseded_by.is_none())
        {
            if seen.insert(memory.id.clone()) {
                candidates.push(memory);
            }
        }
    }

    if candidates.is_empty() {
        return Ok(DedupAction::CreateNew);
    }

    // Find best match by Jaccard similarity
    let mut best_sim = 0.0f32;
    let mut best_memory = None;

    for candidate in &candidates {
        let score = score_candidate(topic, content, candidate, None);
        if score.final_score > best_sim {
            best_sim = score.final_score;
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
                m6_log_outcome(
                    store,
                    best_sim,
                    effective_threshold,
                    similarity_threshold,
                    true,
                );
            }
            if age_days < time_window_days {
                return Ok(DedupAction::MergeInto(memory.id.clone()));
            } else {
                return Ok(DedupAction::Supersede(memory.id.clone()));
            }
        }
    }

    // Gray zone: 0.5 <= sim < threshold — flag for LLM dedup if available
    // M6 LLM budget: extend gray zone down to 0.35 when budget allows (directed exploration)
    // Check sim range first to avoid wasting budget on non-candidates
    let gray_floor = gray_zone_lower_bound(best_sim, m6_has_llm_budget(store));
    if best_sim >= gray_floor {
        if let Some(memory) = best_memory {
            if is_exploration {
                m6_log_outcome(
                    store,
                    best_sim,
                    effective_threshold,
                    similarity_threshold,
                    false,
                );
            }
            return Ok(DedupAction::GrayZone(memory.id.clone(), best_sim));
        }
    }

    // Log exploration non-match for control group
    if is_exploration && best_sim > 0.3 {
        m6_log_outcome(
            store,
            best_sim,
            effective_threshold,
            similarity_threshold,
            false,
        );
    }

    Ok(DedupAction::CreateNew)
}

/// M6: LLM judgment budget — allow up to 10 LLM dedup calls per hour.
/// Uses metadata table for cross-process budget sharing (multiple hook processes
/// may call check_dedup concurrently).
fn m6_has_llm_budget(store: &SqliteStore) -> bool {
    const MAX_LLM_CALLS_PER_HOUR: i64 = 10;

    let now_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let current_hour = now_secs / 3600;

    let conn = store.conn();

    // Atomic increment-and-read in a single SQL statement.
    // Resets counter when the hour changes. Returns the new call count.
    let new_calls: i64 = conn.query_row(
        "INSERT INTO metadata (key, value)
         VALUES ('m6_llm_budget', json_object('hour', ?1, 'calls', 1))
         ON CONFLICT(key) DO UPDATE SET value = CASE
           WHEN CAST(json_extract(value, '$.hour') AS INTEGER) = ?1
           THEN json_object('hour', ?1, 'calls', CAST(json_extract(value, '$.calls') AS INTEGER) + 1)
           ELSE json_object('hour', ?1, 'calls', 1)
         END
         RETURNING CAST(json_extract(value, '$.calls') AS INTEGER)",
        rusqlite::params![current_hour],
        |row| row.get(0),
    ).unwrap_or(MAX_LLM_CALLS_PER_HOUR + 1);

    new_calls <= MAX_LLM_CALLS_PER_HOUR
}

/// M6: Randomized threshold exploration.
/// With 5% probability, offset the threshold by a random amount in [-0.1, +0.1].
/// Returns (effective_threshold, is_exploration).
fn m6_explore_threshold(base_threshold: f32) -> (f32, bool) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let count = COUNTER.fetch_add(1, Ordering::Relaxed);
    // Deterministic pseudo-random: explore on ~5% of calls
    let hash = count
        .wrapping_mul(0x9e3779b97f4a7c15)
        .wrapping_add(0x517cc1b727220a95);
    let explore = hash.is_multiple_of(20); // 5% probability

    if !explore {
        return (base_threshold, false);
    }

    // Random offset in [-0.1, +0.1]
    let offset_bits = ((hash >> 16) % 201) as f32 / 1000.0 - 0.1; // [-0.100, +0.100]
    let effective = (base_threshold + offset_bits).clamp(0.30, 0.95);
    (effective, true)
}

/// M6: Log threshold exploration outcome as feedback event.
fn m6_log_outcome(
    store: &SqliteStore,
    sim: f32,
    used_threshold: f32,
    base_threshold: f32,
    was_dedup: bool,
) {
    let payload = serde_json::json!({
        "similarity": sim,
        "threshold_used": used_threshold,
        "threshold_base": base_threshold,
        "offset": used_threshold - base_threshold,
        "was_dedup": was_dedup,
    });
    let _ = crate::store::adaptive::emit_event(
        store.conn(),
        crate::store::adaptive::FeedbackEvent {
            event_type: crate::store::adaptive::EventType::ParamUpdate,
            request_id: None,
            memory_id: None,
            concept_id: None,
            query: Some(format!("m6_explore:{sim:.3}")),
            query_type: Some("threshold_exploration".to_string()),
            topic: None,
            payload: Some(payload),
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Importance, MemoryLayer, MemoryStatus, MemoryTier, Source};
    use chrono::Utc;

    fn test_memory(topic: &str, content: &str) -> Memory {
        Memory {
            id: ulid::Ulid::new().to_string(),
            layer: MemoryLayer::LTM,
            topic: topic.to_string(),
            summary: content.chars().take(32).collect(),
            content: content.to_string(),
            keywords: vec![],
            importance: Importance::High,
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
            status: MemoryStatus::Active,
            embedding: None,
            tier: MemoryTier::Warm,
            cluster_id: Some(7),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_accessed: Utc::now(),
        }
    }

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
        assert!(
            sim > 0.70,
            "similarity should be > 0.70 (containment dominates), got {sim}"
        );
    }

    #[test]
    fn test_topic_variant_match() {
        assert!(topics_are_variants(
            "Docker Deployment",
            "docker-deployment"
        ));
        assert!(topics_are_variants(
            "docker_deployment",
            "docker deployment"
        ));
        assert!(!topics_are_variants("Docker Deployment", "CP2K"));
    }

    #[test]
    fn test_score_candidate_boosts_variant_and_cluster() {
        let candidate = test_memory("docker-deployment", "Use docker compose for local stack");
        let scored = score_candidate(
            "Docker Deployment",
            "Use docker compose for the local stack",
            &candidate,
            Some(7),
        );
        assert!(scored.topic_variant_match);
        assert!(scored.cluster_match);
        assert!(scored.final_score >= scored.lexical.score);
    }

    #[test]
    fn test_gray_zone_lower_bound() {
        assert!((gray_zone_lower_bound(0.42, true) - 0.35).abs() < f32::EPSILON);
        assert!((gray_zone_lower_bound(0.42, false) - 0.50).abs() < f32::EPSILON);
        assert!((gray_zone_lower_bound(0.60, true) - 0.50).abs() < f32::EPSILON);
    }
}
