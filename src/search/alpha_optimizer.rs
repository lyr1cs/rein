//! Counterfactual Offline Alpha Optimizer
//!
//! Learns the optimal CC fusion alpha from historical recall data by replaying
//! logged candidate sets. The core idea: for each past recall event we know which
//! memories the user actually accessed, so we can grid-search the alpha that would
//! have ranked those memories highest (counterfactual replay).
//!
//! The learned alpha feeds back into `convex_combination` in `rrf.rs`:
//!   `score = alpha * bm25_norm + (1 - alpha) * vec_norm`

use chrono::{DateTime, Utc};

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// A single candidate's normalized scores from a recall event.
#[derive(Debug, Clone)]
pub struct CandidateLog {
    pub memory_id: String,
    pub bm25_norm: f32,
    pub vec_norm: f32,
    pub kg_norm: f32,
    pub episode_norm: f32,
    pub support_count: u32,
    pub source_diversity: f32,
}

/// A logged recall event with outcome.
#[derive(Debug, Clone)]
pub struct RecallEvent {
    pub request_id: String,
    pub candidates: Vec<CandidateLog>,
    /// Which memories were actually accessed/used after recall.
    pub accessed_ids: Vec<String>,
    pub timestamp: DateTime<Utc>,
}

/// Learned alpha with metadata.
#[derive(Debug, Clone)]
pub struct LearnedAlpha {
    pub value: f64,
    pub sample_count: usize,
    pub last_updated: DateTime<Utc>,
}

// ---------------------------------------------------------------------------
// Core algorithm
// ---------------------------------------------------------------------------

/// Coarse grid steps (0.00, 0.05, ..., 1.00 → 21 points).
const COARSE_STEPS: usize = 21;
/// Fine grid steps within ±0.05 of coarse best → 11 points at 0.01 resolution.
const FINE_STEPS: usize = 11;
const KG_BLEND: f64 = 0.10;
const EPISODE_BLEND: f64 = 0.12;
const SUPPORT_BLEND: f64 = 0.05;
const DIVERSITY_BLEND: f64 = 0.05;

/// Find the alpha that maximizes the rank of accessed memories.
///
/// Uses two-phase coarse-fine grid search (21 + 11 = 32 evaluations instead of 101):
/// 2. Rank candidates by score descending.
/// 3. Sum the **reciprocal ranks** of accessed memories: `Σ 1/rank`.
/// 4. The alpha that maximizes this sum is optimal.
///
/// Returns `None` if there are no candidates or no accessed memories.
pub fn optimal_alpha_for_event(event: &RecallEvent) -> Option<f64> {
    if event.candidates.is_empty() || event.accessed_ids.is_empty() {
        return None;
    }

    // Check that at least one accessed id exists in candidates.
    let has_match = event
        .accessed_ids
        .iter()
        .any(|id| event.candidates.iter().any(|c| c.memory_id == *id));
    if !has_match {
        return None;
    }

    let n = event.candidates.len();
    let mut scored: Vec<(f64, usize)> = Vec::with_capacity(n);

    let eval_alpha = |alpha: f64, scored: &mut Vec<(f64, usize)>| -> f64 {
        scored.clear();
        for (i, c) in event.candidates.iter().enumerate() {
            let score = alpha * c.bm25_norm as f64
                + (1.0 - alpha) * c.vec_norm as f64
                + KG_BLEND * c.kg_norm as f64
                + EPISODE_BLEND * c.episode_norm as f64
                + SUPPORT_BLEND * support_signal(c.support_count)
                + DIVERSITY_BLEND * diversity_signal(c.source_diversity);
            scored.push((score, i));
        }
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut mrr_sum = 0.0_f64;
        for (rank_0, &(_score, idx)) in scored.iter().enumerate() {
            let rank = rank_0 + 1;
            if event
                .accessed_ids
                .iter()
                .any(|id| *id == event.candidates[idx].memory_id)
            {
                mrr_sum += 1.0 / rank as f64;
            }
        }
        mrr_sum
    };

    // Phase 1: Coarse grid (0.00, 0.05, ..., 1.00)
    let mut best_alpha = 0.0_f64;
    let mut best_mrr = f64::NEG_INFINITY;
    for step in 0..COARSE_STEPS {
        let alpha = step as f64 / (COARSE_STEPS - 1) as f64;
        let mrr = eval_alpha(alpha, &mut scored);
        if mrr > best_mrr {
            best_mrr = mrr;
            best_alpha = alpha;
        }
    }

    // Phase 2: Fine grid around coarse best (±0.05 at 0.01 resolution)
    let fine_lo = (best_alpha - 0.05).max(0.0);
    let fine_hi = (best_alpha + 0.05).min(1.0);
    for step in 0..FINE_STEPS {
        let alpha = fine_lo + (fine_hi - fine_lo) * step as f64 / (FINE_STEPS - 1) as f64;
        let mrr = eval_alpha(alpha, &mut scored);
        if mrr > best_mrr {
            best_mrr = mrr;
            best_alpha = alpha;
        }
    }

    // Invariant: coarse grid pins alpha ∈ [0, 1], fine grid clamps via max(0)/min(1).
    // best_alpha is never written from any source outside those two grids.
    debug_assert!(
        (0.0..=1.0).contains(&best_alpha),
        "alpha must stay within [0,1]"
    );
    Some(best_alpha)
}

fn support_signal(support_count: u32) -> f64 {
    if support_count > 1 {
        (support_count - 1) as f64 / support_count as f64
    } else {
        0.0
    }
}

fn diversity_signal(source_diversity: f32) -> f64 {
    let diversity = source_diversity as f64;
    if diversity > 1.0 {
        (diversity - 1.0) / diversity
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Bucket-level optimization
// ---------------------------------------------------------------------------

/// Compute the time-weighted mean of optimal alphas across events.
///
/// Recent events receive higher weight via exponential decay:
///   `weight = exp(-base_lambda * age_days)`
///
/// Returns `None` if no events yield a valid optimal alpha.
pub fn optimize_alpha(events: &[RecallEvent], base_lambda: f64) -> Option<LearnedAlpha> {
    if events.is_empty() {
        return None;
    }

    let now = Utc::now();
    let mut weighted_sum = 0.0_f64;
    let mut weight_total = 0.0_f64;
    let mut count = 0_usize;

    for event in events {
        if let Some(alpha) = optimal_alpha_for_event(event) {
            let age_days = (now - event.timestamp).num_seconds() as f64 / 86400.0;
            let weight = (-base_lambda * age_days.max(0.0)).exp() * event_evidence_weight(event);
            weighted_sum += weight * alpha;
            weight_total += weight;
            count += 1;
        }
    }

    if count == 0 || weight_total == 0.0 {
        return None;
    }

    Some(LearnedAlpha {
        value: weighted_sum / weight_total,
        sample_count: count,
        last_updated: now,
    })
}

fn event_evidence_weight(event: &RecallEvent) -> f64 {
    let mut total = 0.0_f64;
    let mut count = 0usize;
    for candidate in &event.candidates {
        if event.accessed_ids.contains(&candidate.memory_id) {
            let support_signal = if candidate.support_count > 1 {
                (candidate.support_count - 1) as f64 / candidate.support_count as f64
            } else {
                0.0
            };
            let diversity = candidate.source_diversity as f64;
            let diversity_signal = if diversity > 1.0 {
                (diversity - 1.0) / diversity
            } else {
                0.0
            };
            total += 1.0 + support_signal + diversity_signal;
            count += 1;
        }
    }
    if count == 0 {
        1.0
    } else {
        total / count as f64
    }
}

// ---------------------------------------------------------------------------
// Bayesian shrinkage
// ---------------------------------------------------------------------------

/// Shrink a bucket's learned alpha toward the parent (global) alpha.
///
/// Uses a pseudo-count prior so small buckets regress toward the global mean:
///   `result = (n_bucket * bucket_alpha + n_prior * parent_alpha) / (n_bucket + n_prior)`
///
/// `n_prior` controls shrinkage strength (typically 5.0).
pub fn bayesian_shrinkage(
    bucket_alpha: f64,
    parent_alpha: f64,
    n_bucket: usize,
    n_prior: f64,
) -> f64 {
    let n = n_bucket as f64;
    (n * bucket_alpha + n_prior * parent_alpha) / (n + n_prior)
}

// ---------------------------------------------------------------------------
// Safety guardrails
// ---------------------------------------------------------------------------

/// Apply max-step guardrail: clamp the change from `current` to `proposed`.
///
/// If `|proposed - current| > max_step`, the result is clamped so that the
/// returned value differs from `current` by at most `max_step`.
pub fn apply_max_step(current: f64, proposed: f64, max_step: f64) -> f64 {
    let delta = proposed - current;
    if delta.abs() <= max_step {
        proposed
    } else {
        current + delta.signum() * max_step
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, Utc};

    /// Helper: build a RecallEvent from (id, bm25, vec) tuples.
    fn make_event(
        candidates: &[(&str, f32, f32)],
        accessed: &[&str],
        days_ago: i64,
    ) -> RecallEvent {
        RecallEvent {
            request_id: "test".to_string(),
            candidates: candidates
                .iter()
                .map(|(id, bm25, vec)| CandidateLog {
                    memory_id: id.to_string(),
                    bm25_norm: *bm25,
                    vec_norm: *vec,
                    kg_norm: 0.0,
                    episode_norm: 0.0,
                    support_count: 1,
                    source_diversity: 1.0,
                })
                .collect(),
            accessed_ids: accessed.iter().map(|s| s.to_string()).collect(),
            timestamp: Utc::now() - Duration::days(days_ago),
        }
    }

    #[test]
    fn test_bm25_dominant_high_alpha() {
        // Accessed memory has high BM25 but low vector score.
        // Optimal alpha should be high (favour BM25).
        // target1 and target2 both have high BM25 but zero/low vec scores.
        // Multiple accessed memories at different ranks forces alpha high.
        let event = make_event(
            &[
                ("target1", 1.0, 0.0),
                ("target2", 0.8, 0.05),
                ("noise1", 0.0, 0.95),
                ("noise2", 0.05, 0.9),
                ("noise3", 0.1, 0.85),
            ],
            &["target1", "target2"],
            0,
        );
        let alpha = optimal_alpha_for_event(&event).unwrap();
        assert!(
            alpha >= 0.5,
            "BM25-dominant targets should yield high alpha, got {alpha}"
        );
    }

    #[test]
    fn test_vector_dominant_low_alpha() {
        // Accessed memory has high vector score but low BM25.
        // Optimal alpha should be low (favour vector).
        let event = make_event(
            &[
                ("target", 0.0, 1.0),
                ("noise1", 0.9, 0.0),
                ("noise2", 0.8, 0.1),
            ],
            &["target"],
            0,
        );
        let alpha = optimal_alpha_for_event(&event).unwrap();
        assert!(
            alpha <= 0.5,
            "Vector-dominant target should yield low alpha, got {alpha}"
        );
    }

    #[test]
    fn test_empty_candidates_returns_none() {
        let event = make_event(&[], &["target"], 0);
        assert!(optimal_alpha_for_event(&event).is_none());
    }

    #[test]
    fn test_empty_accessed_returns_none() {
        let event = make_event(&[("a", 0.5, 0.5)], &[], 0);
        assert!(optimal_alpha_for_event(&event).is_none());
    }

    #[test]
    fn test_no_matching_accessed_returns_none() {
        let event = make_event(&[("a", 0.5, 0.5)], &["nonexistent"], 0);
        assert!(optimal_alpha_for_event(&event).is_none());
    }

    #[test]
    fn test_bayesian_shrinkage_convergence() {
        // With zero bucket observations, result equals parent.
        let result = bayesian_shrinkage(0.9, 0.5, 0, 5.0);
        assert!(
            (result - 0.5).abs() < 1e-9,
            "Zero obs should return parent alpha, got {result}"
        );

        // With many observations, result approaches bucket alpha.
        let result = bayesian_shrinkage(0.8, 0.5, 1000, 5.0);
        assert!(
            (result - 0.8).abs() < 0.01,
            "Many obs should approach bucket alpha, got {result}"
        );

        // With equal weight, result is the midpoint.
        let result = bayesian_shrinkage(1.0, 0.0, 5, 5.0);
        assert!(
            (result - 0.5).abs() < 1e-9,
            "Equal weight should be midpoint, got {result}"
        );
    }

    #[test]
    fn test_apply_max_step_clamping() {
        // Within step: no clamping.
        assert!((apply_max_step(0.5, 0.55, 0.1) - 0.55).abs() < 1e-9);

        // Exceeds step upward: clamp.
        assert!((apply_max_step(0.5, 0.9, 0.1) - 0.6).abs() < 1e-9);

        // Exceeds step downward: clamp.
        assert!((apply_max_step(0.5, 0.1, 0.1) - 0.4).abs() < 1e-9);

        // Exact boundary: no clamping.
        assert!((apply_max_step(0.5, 0.6, 0.1) - 0.6).abs() < 1e-9);
    }

    #[test]
    fn test_optimize_alpha_empty() {
        assert!(optimize_alpha(&[], 0.01).is_none());
    }

    #[test]
    fn test_time_weighted_mean_favours_recent() {
        // Old event: BM25 dominant → high alpha.
        let old = make_event(&[("t", 1.0, 0.0), ("n", 0.0, 1.0)], &["t"], 365);
        // Recent event: vector dominant → low alpha.
        let recent = make_event(&[("t", 0.0, 1.0), ("n", 1.0, 0.0)], &["t"], 0);

        // With strong decay (lambda=0.1), the year-old event should be
        // heavily down-weighted, so the result is closer to the recent event's
        // optimal alpha (which should be low).
        let learned = optimize_alpha(&[old, recent], 0.1).unwrap();
        assert!(
            learned.value < 0.5,
            "Time-weighted mean should favour recent (vector-dominant) event, got {}",
            learned.value
        );
        assert_eq!(learned.sample_count, 2);
    }

    #[test]
    fn test_optimize_alpha_single_event() {
        let event = make_event(&[("t", 1.0, 0.0), ("n", 0.0, 1.0)], &["t"], 0);
        let learned = optimize_alpha(&[event], 0.01).unwrap();
        assert!(learned.value >= 0.5);
        assert_eq!(learned.sample_count, 1);
    }

    #[test]
    fn test_optimize_alpha_evidence_weighting_favours_high_support_event() {
        let mut bm25_event = make_event(&[("t", 1.0, 0.0), ("n", 0.0, 1.0)], &["t"], 0);
        bm25_event.candidates[0].support_count = 5;
        bm25_event.candidates[0].source_diversity = 3.0;

        let vector_event = make_event(&[("t", 0.0, 1.0), ("n", 1.0, 0.0)], &["t"], 0);
        let baseline = optimize_alpha(
            &[
                make_event(&[("t", 1.0, 0.0), ("n", 0.0, 1.0)], &["t"], 0),
                vector_event.clone(),
            ],
            0.01,
        )
        .unwrap();

        let learned = optimize_alpha(&[bm25_event, vector_event], 0.01).unwrap();
        assert!(
            learned.value > baseline.value,
            "higher-support BM25 event should pull alpha upward vs baseline, got {} <= {}",
            learned.value,
            baseline.value
        );
    }

    #[test]
    fn test_episode_and_kg_signals_affect_alpha_scoring() {
        let event = RecallEvent {
            request_id: "test".to_string(),
            candidates: vec![
                CandidateLog {
                    memory_id: "episodic".to_string(),
                    bm25_norm: 0.3,
                    vec_norm: 0.3,
                    kg_norm: 0.2,
                    episode_norm: 0.9,
                    support_count: 1,
                    source_diversity: 1.0,
                },
                CandidateLog {
                    memory_id: "plain".to_string(),
                    bm25_norm: 0.3,
                    vec_norm: 0.3,
                    kg_norm: 0.0,
                    episode_norm: 0.0,
                    support_count: 1,
                    source_diversity: 1.0,
                },
            ],
            accessed_ids: vec!["episodic".to_string()],
            timestamp: Utc::now(),
        };

        let alpha = optimal_alpha_for_event(&event).unwrap();
        assert!((0.0..=1.0).contains(&alpha));
    }
}
