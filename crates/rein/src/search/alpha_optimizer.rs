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

/// Shadow multi-signal fusion weights for v0.28 acceleration experiments.
///
/// This is deliberately a pure offline helper: production recall still uses
/// scalar alpha until a later activation slice wires these weights in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShadowFusionWeights {
    pub bm25: f64,
    pub vec: f64,
    pub kg: f64,
    pub episode: f64,
    pub support: f64,
    pub diversity: f64,
}

impl Default for ShadowFusionWeights {
    fn default() -> Self {
        Self {
            bm25: 0.45,
            vec: 0.45,
            kg: 0.04,
            episode: 0.03,
            support: 0.02,
            diversity: 0.01,
        }
    }
}

impl ShadowFusionWeights {
    pub fn normalized_or_default(self) -> Self {
        let mut values = self.as_array();
        for value in &mut values {
            if !value.is_finite() || *value < 0.0 {
                *value = 0.0;
            }
        }
        let sum: f64 = values.iter().sum();
        if !sum.is_finite() || sum <= f64::EPSILON {
            return Self::default();
        }
        for value in &mut values {
            *value /= sum;
        }
        Self::from_array(values)
    }

    pub fn sum(self) -> f64 {
        self.bm25 + self.vec + self.kg + self.episode + self.support + self.diversity
    }

    fn as_array(self) -> [f64; SHADOW_DIMENSIONS] {
        [
            self.bm25,
            self.vec,
            self.kg,
            self.episode,
            self.support,
            self.diversity,
        ]
    }

    fn from_array(values: [f64; SHADOW_DIMENSIONS]) -> Self {
        Self {
            bm25: values[0],
            vec: values[1],
            kg: values[2],
            episode: values[3],
            support: values[4],
            diversity: values[5],
        }
    }
}

/// Learned shadow weights with metadata. Not persisted in v0.28's first S3
/// slice; callers can inspect it in tests/replay only.
#[derive(Debug, Clone)]
pub struct LearnedShadowWeights {
    pub weights: ShadowFusionWeights,
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
const SHADOW_DIMENSIONS: usize = 6;

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
    let mut worst_mrr = f64::INFINITY;
    for step in 0..COARSE_STEPS {
        let alpha = step as f64 / (COARSE_STEPS - 1) as f64;
        let mrr = eval_alpha(alpha, &mut scored);
        if mrr > best_mrr {
            best_mrr = mrr;
            best_alpha = alpha;
        }
        if mrr < worst_mrr {
            worst_mrr = mrr;
        }
    }

    // If every alpha in the coarse grid produced an identical MRR the event
    // carries no preference for any alpha (zero-variance candidate set).
    // Previously the strict `>` comparison picked whichever alpha was tried
    // first — always 0.0, biasing the learned mean toward pure-vector.
    // Returning None here makes the event invisible to `optimize_alpha`'s
    // weighted mean, so the existing prior in the cluster/global hierarchy
    // is preserved instead of being dragged by a signal-free event.
    if (best_mrr - worst_mrr).abs() < 1e-12 {
        return None;
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
    if diversity.is_finite() && diversity > 1.0 {
        (diversity - 1.0) / diversity
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Shadow multi-weight optimization
// ---------------------------------------------------------------------------

/// Score a candidate with normalized shadow fusion weights.
pub fn score_candidate_with_shadow_weights(
    candidate: &CandidateLog,
    weights: ShadowFusionWeights,
) -> f64 {
    let weights = weights.normalized_or_default().as_array();
    let features = shadow_features(candidate);
    weights
        .iter()
        .zip(features.iter())
        .map(|(weight, feature)| weight * feature)
        .sum()
}

/// Find the per-event shadow weight vector that best ranks accessed memories.
///
/// The first v0.28 slice keeps this intentionally conservative: evaluate the
/// six one-hot signal dimensions and average tied winners. Later slices can
/// replace this with a continuous optimizer without changing the caller shape.
pub fn optimal_shadow_weights_for_event(event: &RecallEvent) -> Option<ShadowFusionWeights> {
    if event.candidates.is_empty() || event.accessed_ids.is_empty() {
        return None;
    }
    if !has_matching_accessed_candidate(event) || !has_shadow_feature_variance(event) {
        return None;
    }

    let mut scores = [0.0_f64; SHADOW_DIMENSIONS];
    let mut best = f64::NEG_INFINITY;
    let mut worst = f64::INFINITY;
    for dimension in 0..SHADOW_DIMENSIONS {
        let mut values = [0.0_f64; SHADOW_DIMENSIONS];
        values[dimension] = 1.0;
        let weights = ShadowFusionWeights::from_array(values);
        let mrr = shadow_reciprocal_rank_sum(event, weights);
        scores[dimension] = mrr;
        best = best.max(mrr);
        worst = worst.min(mrr);
    }

    if !best.is_finite() || (best - worst).abs() < 1e-12 {
        return None;
    }

    let mut winners = [0.0_f64; SHADOW_DIMENSIONS];
    for (idx, score) in scores.iter().enumerate() {
        if (*score - best).abs() < 1e-12 {
            winners[idx] = 1.0;
        }
    }
    Some(ShadowFusionWeights::from_array(winners).normalized_or_default())
}

/// Compute a time-weighted shadow weight vector and shrink it toward a parent
/// prior. This is pure replay math; it does not mutate adaptive state.
pub fn optimize_shadow_weights(
    events: &[RecallEvent],
    base_lambda: f64,
    parent_prior: ShadowFusionWeights,
    n_prior: f64,
) -> Option<LearnedShadowWeights> {
    if events.is_empty() {
        return None;
    }

    let now = Utc::now();
    let mut weighted = [0.0_f64; SHADOW_DIMENSIONS];
    let mut weight_total = 0.0_f64;
    let mut count = 0_usize;
    for event in events {
        let Some(weights) = optimal_shadow_weights_for_event(event) else {
            continue;
        };
        let age_days = (now - event.timestamp).num_seconds() as f64 / 86400.0;
        let event_weight = (-base_lambda * age_days.max(0.0)).exp() * event_evidence_weight(event);
        for (target, value) in weighted.iter_mut().zip(weights.as_array()) {
            *target += event_weight * value;
        }
        weight_total += event_weight;
        count += 1;
    }

    if count == 0 || !weight_total.is_finite() || weight_total <= f64::EPSILON {
        return None;
    }

    for value in &mut weighted {
        *value /= weight_total;
    }
    let bucket = ShadowFusionWeights::from_array(weighted).normalized_or_default();
    Some(LearnedShadowWeights {
        weights: shrink_shadow_weights_toward_parent(bucket, parent_prior, count, n_prior),
        sample_count: count,
        last_updated: now,
    })
}

pub fn shrink_shadow_weights_toward_parent(
    bucket_weights: ShadowFusionWeights,
    parent_prior: ShadowFusionWeights,
    n_bucket: usize,
    n_prior: f64,
) -> ShadowFusionWeights {
    let bucket = bucket_weights.normalized_or_default().as_array();
    let parent = parent_prior.normalized_or_default().as_array();
    let n = n_bucket as f64;
    let prior = if n_prior.is_finite() && n_prior > 0.0 {
        n_prior
    } else {
        0.0
    };
    if n <= f64::EPSILON && prior <= f64::EPSILON {
        return ShadowFusionWeights::from_array(parent);
    }

    let denom = n + prior;
    let mut values = [0.0_f64; SHADOW_DIMENSIONS];
    for idx in 0..SHADOW_DIMENSIONS {
        values[idx] = (n * bucket[idx] + prior * parent[idx]) / denom;
    }
    ShadowFusionWeights::from_array(values).normalized_or_default()
}

fn has_matching_accessed_candidate(event: &RecallEvent) -> bool {
    event.accessed_ids.iter().any(|id| {
        event
            .candidates
            .iter()
            .any(|candidate| candidate.memory_id == *id)
    })
}

fn has_shadow_feature_variance(event: &RecallEvent) -> bool {
    let Some(first) = event.candidates.first().map(shadow_features) else {
        return false;
    };
    event.candidates.iter().skip(1).any(|candidate| {
        shadow_features(candidate)
            .iter()
            .zip(first.iter())
            .any(|(left, right)| (left - right).abs() > 1e-12)
    })
}

fn shadow_reciprocal_rank_sum(event: &RecallEvent, weights: ShadowFusionWeights) -> f64 {
    let mut scored = Vec::with_capacity(event.candidates.len());
    for (idx, candidate) in event.candidates.iter().enumerate() {
        scored.push((score_candidate_with_shadow_weights(candidate, weights), idx));
    }
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut mrr_sum = 0.0_f64;
    let mut group_start = 0_usize;
    while group_start < scored.len() {
        let score = scored[group_start].0;
        let mut group_end = group_start + 1;
        while group_end < scored.len() && (scored[group_end].0 - score).abs() < 1e-12 {
            group_end += 1;
        }

        let avg_reciprocal_rank = (group_start..group_end)
            .map(|rank_0| 1.0 / (rank_0 + 1) as f64)
            .sum::<f64>()
            / (group_end - group_start) as f64;
        for &(_score, idx) in &scored[group_start..group_end] {
            if event
                .accessed_ids
                .iter()
                .any(|id| *id == event.candidates[idx].memory_id)
            {
                mrr_sum += avg_reciprocal_rank;
            }
        }
        group_start = group_end;
    }
    mrr_sum
}

fn shadow_features(candidate: &CandidateLog) -> [f64; SHADOW_DIMENSIONS] {
    [
        finite_nonnegative(candidate.bm25_norm as f64),
        finite_nonnegative(candidate.vec_norm as f64),
        finite_nonnegative(candidate.kg_norm as f64),
        finite_nonnegative(candidate.episode_norm as f64),
        support_signal(candidate.support_count),
        diversity_signal(candidate.source_diversity),
    ]
}

fn finite_nonnegative(value: f64) -> f64 {
    if value.is_finite() && value >= 0.0 {
        value
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
            total += 1.0 + support_signal + diversity_signal(candidate.source_diversity);
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
        // Event where BM25 and vector differ (so alpha is *meaningful*), and
        // the accessed candidate additionally carries KG + episode signals.
        // The learned alpha must still fall inside [0, 1] — the accessory
        // signals shift the ranking but do not push alpha out of bounds.
        let event = RecallEvent {
            request_id: "test".to_string(),
            candidates: vec![
                CandidateLog {
                    memory_id: "episodic".to_string(),
                    bm25_norm: 0.8,
                    vec_norm: 0.2,
                    kg_norm: 0.2,
                    episode_norm: 0.9,
                    support_count: 1,
                    source_diversity: 1.0,
                },
                CandidateLog {
                    memory_id: "plain".to_string(),
                    bm25_norm: 0.2,
                    vec_norm: 0.8,
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

    #[test]
    fn test_zero_variance_event_returns_none() {
        // All candidates identical on alpha-sensitive signals → the learned
        // alpha would be meaningless noise. optimal_alpha_for_event must
        // return None so the weighted mean in optimize_alpha skips the event
        // instead of biasing toward the first grid point evaluated.
        let event = RecallEvent {
            request_id: "tied".to_string(),
            candidates: vec![
                CandidateLog {
                    memory_id: "a".to_string(),
                    bm25_norm: 0.5,
                    vec_norm: 0.5,
                    kg_norm: 0.0,
                    episode_norm: 0.0,
                    support_count: 1,
                    source_diversity: 1.0,
                },
                CandidateLog {
                    memory_id: "b".to_string(),
                    bm25_norm: 0.5,
                    vec_norm: 0.5,
                    kg_norm: 0.0,
                    episode_norm: 0.0,
                    support_count: 1,
                    source_diversity: 1.0,
                },
            ],
            accessed_ids: vec!["a".to_string()],
            timestamp: Utc::now(),
        };

        assert_eq!(optimal_alpha_for_event(&event), None);
    }

    fn shadow_event() -> RecallEvent {
        RecallEvent {
            request_id: "shadow".to_string(),
            candidates: vec![
                CandidateLog {
                    memory_id: "target".to_string(),
                    bm25_norm: 0.2,
                    vec_norm: 0.2,
                    kg_norm: 1.0,
                    episode_norm: 0.1,
                    support_count: 1,
                    source_diversity: 1.0,
                },
                CandidateLog {
                    memory_id: "bm25_noise".to_string(),
                    bm25_norm: 1.0,
                    vec_norm: 0.1,
                    kg_norm: 0.0,
                    episode_norm: 0.0,
                    support_count: 1,
                    source_diversity: 1.0,
                },
                CandidateLog {
                    memory_id: "vec_noise".to_string(),
                    bm25_norm: 0.1,
                    vec_norm: 1.0,
                    kg_norm: 0.0,
                    episode_norm: 0.0,
                    support_count: 1,
                    source_diversity: 1.0,
                },
            ],
            accessed_ids: vec!["target".to_string()],
            timestamp: Utc::now(),
        }
    }

    fn l1_shadow_distance(a: ShadowFusionWeights, b: ShadowFusionWeights) -> f64 {
        (a.bm25 - b.bm25).abs()
            + (a.vec - b.vec).abs()
            + (a.kg - b.kg).abs()
            + (a.episode - b.episode).abs()
            + (a.support - b.support).abs()
            + (a.diversity - b.diversity).abs()
    }

    #[test]
    fn shadow_fusion_weights_normalize_to_finite_simplex() {
        let weights = ShadowFusionWeights {
            bm25: f64::NAN,
            vec: -1.0,
            kg: f64::INFINITY,
            episode: 2.0,
            support: 1.0,
            diversity: 0.0,
        }
        .normalized_or_default();

        assert!(weights.bm25.is_finite() && weights.bm25 >= 0.0);
        assert!(weights.vec.is_finite() && weights.vec >= 0.0);
        assert!(weights.kg.is_finite() && weights.kg >= 0.0);
        assert!(weights.episode.is_finite() && weights.episode >= 0.0);
        assert!(weights.support.is_finite() && weights.support >= 0.0);
        assert!(weights.diversity.is_finite() && weights.diversity >= 0.0);
        assert!((weights.sum() - 1.0).abs() < 1e-12);

        let fallback = ShadowFusionWeights {
            bm25: f64::NAN,
            vec: -1.0,
            kg: 0.0,
            episode: 0.0,
            support: 0.0,
            diversity: 0.0,
        }
        .normalized_or_default();
        assert_eq!(fallback, ShadowFusionWeights::default());
    }

    #[test]
    fn shadow_score_candidate_uses_all_logged_dimensions() {
        let accessed = CandidateLog {
            memory_id: "accessed".to_string(),
            bm25_norm: 0.5,
            vec_norm: 0.5,
            kg_norm: 1.0,
            episode_norm: 1.0,
            support_count: 5,
            source_diversity: 3.0,
        };
        let plain = CandidateLog {
            memory_id: "plain".to_string(),
            bm25_norm: 0.5,
            vec_norm: 0.5,
            kg_norm: 0.0,
            episode_norm: 0.0,
            support_count: 1,
            source_diversity: 1.0,
        };
        let weights = ShadowFusionWeights {
            bm25: 0.0,
            vec: 0.0,
            kg: 0.35,
            episode: 0.25,
            support: 0.25,
            diversity: 0.15,
        };

        assert!(
            score_candidate_with_shadow_weights(&accessed, weights)
                > score_candidate_with_shadow_weights(&plain, weights)
        );
    }

    #[test]
    fn shadow_score_candidate_sanitizes_nonfinite_features() {
        let candidate = CandidateLog {
            memory_id: "bad_features".to_string(),
            bm25_norm: f32::NAN,
            vec_norm: f32::INFINITY,
            kg_norm: f32::NEG_INFINITY,
            episode_norm: 0.5,
            support_count: 2,
            source_diversity: f32::INFINITY,
        };

        let score = score_candidate_with_shadow_weights(&candidate, ShadowFusionWeights::default());

        assert!(score.is_finite(), "shadow score must never produce NaN/inf");
    }

    #[test]
    fn optimal_shadow_weights_for_event_prefers_accessed_signal_dimension() {
        let prior = ShadowFusionWeights::default();
        let learned = optimal_shadow_weights_for_event(&shadow_event()).unwrap();

        assert!(
            learned.kg > prior.kg,
            "KG-dominant accessed candidate should pull more KG mass than prior, got {} <= {}",
            learned.kg,
            prior.kg
        );
    }

    #[test]
    fn optimize_shadow_weights_skips_zero_variance_events() {
        let event = RecallEvent {
            request_id: "shadow_tied".to_string(),
            candidates: vec![
                CandidateLog {
                    memory_id: "a".to_string(),
                    bm25_norm: 0.5,
                    vec_norm: 0.5,
                    kg_norm: 0.5,
                    episode_norm: 0.5,
                    support_count: 2,
                    source_diversity: 2.0,
                },
                CandidateLog {
                    memory_id: "b".to_string(),
                    bm25_norm: 0.5,
                    vec_norm: 0.5,
                    kg_norm: 0.5,
                    episode_norm: 0.5,
                    support_count: 2,
                    source_diversity: 2.0,
                },
            ],
            accessed_ids: vec!["a".to_string()],
            timestamp: Utc::now(),
        };

        assert!(
            optimize_shadow_weights(&[event], 0.01, ShadowFusionWeights::default(), 5.0).is_none()
        );
    }

    #[test]
    fn optimize_shadow_weights_shrinks_toward_parent_prior() {
        let parent = ShadowFusionWeights::default();
        let event = shadow_event();
        let raw = optimal_shadow_weights_for_event(&event).unwrap();
        let learned = optimize_shadow_weights(&[event], 0.01, parent, 5.0).unwrap();

        assert!(learned.weights.kg > parent.kg);
        assert!(
            l1_shadow_distance(learned.weights, parent) < l1_shadow_distance(raw, parent),
            "small-sample shadow weights should stay closer to parent than raw event optimum"
        );
        assert_eq!(learned.sample_count, 1);
    }

    #[test]
    fn optimize_shadow_weights_sanitizes_nonfinite_evidence_weight() {
        let parent = ShadowFusionWeights::default();
        let mut event = shadow_event();
        event.candidates[0].source_diversity = f32::INFINITY;

        let learned = optimize_shadow_weights(&[event], 0.01, parent, 5.0)
            .expect("nonfinite evidence signal should sanitize, not skip event");

        assert!(learned.weights.sum().is_finite());
        assert!((learned.weights.sum() - 1.0).abs() < 1e-12);
    }
}
