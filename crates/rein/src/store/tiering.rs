//! Hot/Warm/Cold memory tiering with adaptive quantile boundaries.
//!
//! Memories are classified into three tiers based on their access rate
//! (access_count / days_since_creation). Tier boundaries adapt to the
//! actual distribution using a streaming quantile estimator (P25/P75).
//!
//! - **Hot** (top 25%): Always in recall, highest priority.
//! - **Warm** (middle 50%): Normal recall participation.
//! - **Cold** (bottom 25%): Only searched in Exploratory queries; eligible for compressed storage.

use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

// ---------------------------------------------------------------------------
// MemoryTier
// ---------------------------------------------------------------------------

/// Memory storage tier based on access frequency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum MemoryTier {
    /// Top 25% by access rate — always in recall, highest priority.
    Hot,
    /// Middle 50% — normal recall participation.
    #[default]
    Warm,
    /// Bottom 25% — only in Exploratory queries, compressed storage.
    Cold,
}

impl std::fmt::Display for MemoryTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Hot => write!(f, "hot"),
            Self::Warm => write!(f, "warm"),
            Self::Cold => write!(f, "cold"),
        }
    }
}

impl std::str::FromStr for MemoryTier {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "hot" => Ok(Self::Hot),
            "warm" => Ok(Self::Warm),
            "cold" => Ok(Self::Cold),
            _ => Err(format!("unknown tier: {s}")),
        }
    }
}

// ---------------------------------------------------------------------------
// QuantileEstimator
// ---------------------------------------------------------------------------

/// Simple streaming quantile estimator backed by a sorted reservoir.
///
/// Maintains up to `max_samples` values. When the reservoir is full new
/// values are accepted via reservoir sampling (probability `max_samples / n`
/// where `n` is the total number of values seen so far).
///
/// Quantile queries use linear interpolation between the two nearest ranks.
pub struct QuantileEstimator {
    /// Sorted sample reservoir.
    samples: Vec<f64>,
    /// Maximum reservoir size.
    max_samples: usize,
    /// Total number of values observed (may exceed `max_samples`).
    total_seen: usize,
}

impl QuantileEstimator {
    /// Create a new estimator with the given reservoir capacity.
    ///
    /// A capacity of 1000 is usually sufficient for accurate P25/P75.
    pub fn new(max_samples: usize) -> Self {
        assert!(max_samples > 0, "max_samples must be > 0");
        Self {
            samples: Vec::with_capacity(max_samples),
            max_samples,
            total_seen: 0,
        }
    }

    /// Insert a single value into the reservoir.
    ///
    /// This path is used for numeric values that do not carry an identity;
    /// it derives its reservoir-sampling decision purely from `total_seen`
    /// and the value's own bit pattern, so identical input sequences always
    /// produce identical reservoir state (reproducibility across processes).
    pub fn add(&mut self, value: f64) {
        self.total_seen += 1;

        if self.samples.len() < self.max_samples {
            // Reservoir not yet full — always accept.
            let pos = self.samples.partition_point(|&v| v < value);
            self.samples.insert(pos, value);
        } else {
            // Deterministic reservoir sampling keyed on (total_seen, value bits).
            // Avoids a global atomic nonce so concurrent insertions no longer
            // produce non-reproducible tier boundaries.
            let key = mix_u64(self.total_seen as u64, value.to_bits());
            self.add_with_key(value, key);
        }
    }

    /// Insert a value with an explicit deterministic key (e.g. hash of a memory ID).
    ///
    /// Use this when callers have a stable identifier for each value and need
    /// reproducible reservoir membership across runs / processes.
    pub fn add_with_id(&mut self, value: f64, id: &str) {
        self.total_seen += 1;

        if self.samples.len() < self.max_samples {
            let pos = self.samples.partition_point(|&v| v < value);
            self.samples.insert(pos, value);
        } else {
            let key = hash_id(id);
            self.add_with_key(value, key);
        }
    }

    /// Shared reservoir-replacement logic given a precomputed deterministic key.
    fn add_with_key(&mut self, value: f64, key: u64) {
        let accept = (key as usize) % self.total_seen < self.max_samples;
        if accept {
            let idx = (key.rotate_left(17) as usize) % self.max_samples;
            self.samples[idx] = value;
            self.samples
                .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        }
    }

    /// Insert a batch of values.
    pub fn add_batch(&mut self, values: &[f64]) {
        for &v in values {
            self.add(v);
        }
    }

    /// Estimate the `q`-th quantile (q in \[0, 1\]).
    ///
    /// Returns `None` if the reservoir is empty.
    /// Uses linear interpolation between the two nearest ranks.
    pub fn quantile(&self, q: f64) -> Option<f64> {
        if self.samples.is_empty() {
            return None;
        }
        let q = q.clamp(0.0, 1.0);
        let n = self.samples.len();
        if n == 1 {
            return Some(self.samples[0]);
        }

        // Continuous index (0-based).
        let idx = q * (n - 1) as f64;
        let lo = idx.floor() as usize;
        let hi = lo.min(n - 2) + 1; // ensure hi < n
        let frac = idx - lo as f64;
        Some(self.samples[lo] * (1.0 - frac) + self.samples[hi] * frac)
    }

    /// Number of values currently held in the reservoir.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether the reservoir is empty.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

/// Splitmix64 mixing of two u64 inputs. Deterministic, not crypto — used only
/// to derive reservoir-sampling decisions. Results are identical across runs
/// and processes for the same inputs.
fn mix_u64(a: u64, b: u64) -> u64 {
    let mut x = a.wrapping_add(b).wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Deterministic 64-bit hash of a string identifier.
///
/// Uses `DefaultHasher` which is deterministic within a process for a given
/// input. For reservoir sampling on memory IDs this is sufficient to keep
/// tier boundaries reproducible across repeated runs with the same inputs.
fn hash_id(id: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    id.hash(&mut hasher);
    hasher.finish()
}

// ---------------------------------------------------------------------------
// TierBoundaries
// ---------------------------------------------------------------------------

/// Adaptive tier boundaries computed from the access-rate distribution.
///
/// After calling [`update`] with the full set of memory access rates the
/// thresholds are set to P25 (cold boundary) and P75 (hot boundary) of
/// that distribution. Any subsequent call to [`tier_for`] classifies a
/// single access rate into Hot / Warm / Cold.
pub struct TierBoundaries {
    estimator: QuantileEstimator,
    /// Access rate at or above which a memory is Hot (P75).
    pub hot_threshold: f64,
    /// Access rate at or below which a memory is Cold (P25).
    pub cold_threshold: f64,
}

impl TierBoundaries {
    /// Create new boundaries with default reservoir size (1000).
    pub fn new() -> Self {
        Self {
            estimator: QuantileEstimator::new(1000),
            hot_threshold: f64::MAX,
            cold_threshold: 0.0,
        }
    }

    /// Recompute boundaries from all memory access rates.
    ///
    /// This replaces the estimator contents entirely (designed to be called
    /// once per GC cycle with the full population).
    pub fn update(&mut self, access_rates: &[f64]) {
        self.estimator = QuantileEstimator::new(self.estimator.max_samples);
        self.estimator.add_batch(access_rates);

        self.cold_threshold = self.estimator.quantile(0.25).unwrap_or(0.0);
        self.hot_threshold = self.estimator.quantile(0.75).unwrap_or(f64::MAX);

        // Guard against degenerate distributions where P25 == P75.
        if (self.hot_threshold - self.cold_threshold).abs() < f64::EPSILON {
            self.hot_threshold = self.cold_threshold + 1.0;
        }
    }

    /// Determine the tier for a given access rate.
    pub fn tier_for(&self, access_rate: f64) -> MemoryTier {
        if access_rate >= self.hot_threshold {
            MemoryTier::Hot
        } else if access_rate <= self.cold_threshold {
            MemoryTier::Cold
        } else {
            MemoryTier::Warm
        }
    }

    /// Determine tier with a cluster bonus adjustment.
    ///
    /// If a memory's own rate is Warm but its cluster's average rate exceeds
    /// the hot threshold, the memory is promoted to Hot. Conversely if the
    /// cluster average is below the cold threshold the memory is demoted to
    /// Cold. This prevents isolated low-access memories in otherwise active
    /// topic clusters from being incorrectly cooled.
    pub fn tier_for_with_cluster(&self, access_rate: f64, cluster_avg_rate: f64) -> MemoryTier {
        let base = self.tier_for(access_rate);
        match base {
            MemoryTier::Warm => {
                if cluster_avg_rate >= self.hot_threshold {
                    MemoryTier::Hot
                } else if cluster_avg_rate <= self.cold_threshold {
                    MemoryTier::Cold
                } else {
                    MemoryTier::Warm
                }
            }
            // Hot and Cold are not overridden by cluster context.
            other => other,
        }
    }
}

impl Default for TierBoundaries {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// MigrationTracker
// ---------------------------------------------------------------------------

/// Tracks consecutive tier-change signals per memory to prevent flapping.
///
/// A memory must receive **two consecutive** GC-cycle signals proposing the
/// same new tier before migration proceeds. If the proposed tier changes
/// between cycles the counter resets.
pub struct MigrationTracker {
    /// memory_id -> (proposed_tier, consecutive_count)
    pending: HashMap<String, (MemoryTier, u32)>,
}

impl MigrationTracker {
    /// Create a new tracker with no pending signals.
    pub fn new() -> Self {
        Self {
            pending: HashMap::new(),
        }
    }

    /// Record a proposed tier change for `memory_id`.
    ///
    /// Returns `true` when the same tier has been proposed for **two
    /// consecutive** calls (i.e., migration should proceed).
    pub fn signal(&mut self, memory_id: &str, proposed_tier: MemoryTier) -> bool {
        let entry = self
            .pending
            .entry(memory_id.to_owned())
            .or_insert((proposed_tier, 0));

        if entry.0 == proposed_tier {
            entry.1 += 1;
        } else {
            // Different tier proposed — reset.
            entry.0 = proposed_tier;
            entry.1 = 1;
        }

        entry.1 >= 2
    }

    /// Clear tracking state for a memory (call after successful migration).
    pub fn clear(&mut self, memory_id: &str) {
        self.pending.remove(memory_id);
    }
}

impl Default for MigrationTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute the access rate for a memory.
///
/// Defined as `access_count / max(1, days_since_creation)` so that very
/// recent memories do not receive an inflated rate.
pub fn compute_access_rate(access_count: u32, created_at: chrono::DateTime<chrono::Utc>) -> f64 {
    let days = (Utc::now() - created_at).num_days().max(1) as f64;
    access_count as f64 / days
}

/// Return the set of tiers to search for a given query type.
///
/// - Exploratory queries search **all** tiers (Hot + Warm + Cold).
/// - All other queries skip Cold to keep latency low.
pub fn tiers_for_query(query_type_is_exploratory: bool) -> Vec<MemoryTier> {
    if query_type_is_exploratory {
        vec![MemoryTier::Hot, MemoryTier::Warm, MemoryTier::Cold]
    } else {
        vec![MemoryTier::Hot, MemoryTier::Warm]
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, Utc};

    // -- QuantileEstimator --------------------------------------------------

    #[test]
    fn quantile_known_data() {
        let mut est = QuantileEstimator::new(200);
        for i in 1..=100 {
            est.add(i as f64);
        }
        assert_eq!(est.len(), 100);

        let p25 = est.quantile(0.25).unwrap();
        let p75 = est.quantile(0.75).unwrap();

        // With 100 values 1..=100 the exact P25 ≈ 25.75, P75 ≈ 75.25.
        assert!((p25 - 25.75).abs() < 1.0, "P25 was {p25}");
        assert!((p75 - 75.25).abs() < 1.0, "P75 was {p75}");
    }

    #[test]
    fn quantile_boundary_values() {
        let mut est = QuantileEstimator::new(200);
        est.add(42.0);

        // Single element: every quantile should return that element.
        assert_eq!(est.quantile(0.0), Some(42.0));
        assert_eq!(est.quantile(0.5), Some(42.0));
        assert_eq!(est.quantile(1.0), Some(42.0));
    }

    #[test]
    fn quantile_empty() {
        let est = QuantileEstimator::new(10);
        assert_eq!(est.quantile(0.5), None);
        assert!(est.is_empty());
    }

    #[test]
    fn quantile_reservoir_overflow() {
        let max = 50;
        let mut est = QuantileEstimator::new(max);
        for i in 0..500 {
            est.add(i as f64);
        }
        // Reservoir should not exceed max_samples.
        assert!(est.len() <= max);
        // Should still produce a reasonable estimate.
        let median = est.quantile(0.5).unwrap();
        assert!(median > 50.0 && median < 450.0, "median was {median}");
    }

    #[test]
    fn tiering_reservoir_is_deterministic_for_same_inputs() {
        // Two independent estimators fed the exact same sequence must agree
        // on the final reservoir state — no global nonce dependence.
        let mut a = QuantileEstimator::new(50);
        let mut b = QuantileEstimator::new(50);
        for i in 0..1000 {
            let v = (i as f64) * 0.5;
            a.add(v);
            b.add(v);
        }
        assert_eq!(a.samples, b.samples, "reservoir must be deterministic");
        assert_eq!(a.total_seen, b.total_seen);
    }

    #[test]
    fn tiering_reservoir_add_with_id_is_deterministic() {
        let mut a = QuantileEstimator::new(50);
        let mut b = QuantileEstimator::new(50);
        for i in 0..500 {
            let v = (i as f64) * 0.25;
            let id = format!("mem-{i}");
            a.add_with_id(v, &id);
            b.add_with_id(v, &id);
        }
        assert_eq!(a.samples, b.samples);
    }

    #[test]
    fn quantile_add_batch() {
        let mut est = QuantileEstimator::new(200);
        let vals: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        est.add_batch(&vals);
        assert_eq!(est.len(), 100);
        let p50 = est.quantile(0.5).unwrap();
        assert!((p50 - 50.5).abs() < 1.0, "P50 was {p50}");
    }

    // -- TierBoundaries -----------------------------------------------------

    #[test]
    fn tier_boundaries_clear_tiers() {
        let mut tb = TierBoundaries::new();
        // Access rates: 0..100.
        let rates: Vec<f64> = (0..=100).map(|i| i as f64).collect();
        tb.update(&rates);

        assert_eq!(tb.tier_for(90.0), MemoryTier::Hot);
        assert_eq!(tb.tier_for(50.0), MemoryTier::Warm);
        assert_eq!(tb.tier_for(5.0), MemoryTier::Cold);
    }

    #[test]
    fn tier_boundaries_degenerate_distribution() {
        let mut tb = TierBoundaries::new();
        // All same value — should not panic or produce nonsense.
        tb.update(&[5.0; 50]);
        // hot_threshold should be adjusted so it differs from cold_threshold.
        assert!(tb.hot_threshold > tb.cold_threshold);
    }

    #[test]
    fn tier_for_with_cluster_promotes() {
        let mut tb = TierBoundaries::new();
        let rates: Vec<f64> = (0..=100).map(|i| i as f64).collect();
        tb.update(&rates);

        // 50 is Warm by itself, but cluster average 90 should promote to Hot.
        assert_eq!(tb.tier_for(50.0), MemoryTier::Warm);
        assert_eq!(tb.tier_for_with_cluster(50.0, 90.0), MemoryTier::Hot,);
    }

    #[test]
    fn tier_for_with_cluster_demotes() {
        let mut tb = TierBoundaries::new();
        let rates: Vec<f64> = (0..=100).map(|i| i as f64).collect();
        tb.update(&rates);

        // 50 is Warm by itself, cluster average 5 should demote to Cold.
        assert_eq!(tb.tier_for_with_cluster(50.0, 5.0), MemoryTier::Cold,);
    }

    #[test]
    fn tier_for_with_cluster_no_override_hot() {
        let mut tb = TierBoundaries::new();
        let rates: Vec<f64> = (0..=100).map(|i| i as f64).collect();
        tb.update(&rates);

        // Hot stays Hot regardless of cluster.
        assert_eq!(tb.tier_for_with_cluster(95.0, 5.0), MemoryTier::Hot,);
    }

    // -- MigrationTracker ---------------------------------------------------

    #[test]
    fn migration_signal_once_no_migrate() {
        let mut tracker = MigrationTracker::new();
        assert!(!tracker.signal("mem-1", MemoryTier::Cold));
    }

    #[test]
    fn migration_signal_twice_migrates() {
        let mut tracker = MigrationTracker::new();
        assert!(!tracker.signal("mem-1", MemoryTier::Cold));
        assert!(tracker.signal("mem-1", MemoryTier::Cold));
    }

    #[test]
    fn migration_signal_change_resets() {
        let mut tracker = MigrationTracker::new();
        assert!(!tracker.signal("mem-1", MemoryTier::Cold));
        // Different tier resets the counter.
        assert!(!tracker.signal("mem-1", MemoryTier::Hot));
        // First signal for Hot — not yet.
        assert!(tracker.signal("mem-1", MemoryTier::Hot));
    }

    #[test]
    fn migration_clear_resets() {
        let mut tracker = MigrationTracker::new();
        assert!(!tracker.signal("mem-1", MemoryTier::Cold));
        tracker.clear("mem-1");
        // After clear, starts fresh.
        assert!(!tracker.signal("mem-1", MemoryTier::Cold));
    }

    // -- compute_access_rate ------------------------------------------------

    #[test]
    fn access_rate_normal() {
        let created = Utc::now() - Duration::days(10);
        let rate = compute_access_rate(20, created);
        assert!((rate - 2.0).abs() < 0.01, "rate was {rate}");
    }

    #[test]
    fn access_rate_zero_days() {
        // Created just now — days_since_creation clamped to 1.
        let rate = compute_access_rate(5, Utc::now());
        assert!((rate - 5.0).abs() < 0.01, "rate was {rate}");
    }

    #[test]
    fn access_rate_zero_access() {
        let created = Utc::now() - Duration::days(30);
        let rate = compute_access_rate(0, created);
        assert!(rate.abs() < f64::EPSILON, "rate was {rate}");
    }

    // -- tiers_for_query ----------------------------------------------------

    #[test]
    fn tiers_for_normal_query() {
        let tiers = tiers_for_query(false);
        assert_eq!(tiers, vec![MemoryTier::Hot, MemoryTier::Warm]);
    }

    #[test]
    fn tiers_for_exploratory_query() {
        let tiers = tiers_for_query(true);
        assert_eq!(
            tiers,
            vec![MemoryTier::Hot, MemoryTier::Warm, MemoryTier::Cold],
        );
    }

    // -- MemoryTier Display / FromStr ---------------------------------------

    #[test]
    fn tier_display_fromstr_roundtrip() {
        for tier in &[MemoryTier::Hot, MemoryTier::Warm, MemoryTier::Cold] {
            let s = tier.to_string();
            let parsed: MemoryTier = s.parse().unwrap();
            assert_eq!(*tier, parsed);
        }
    }

    #[test]
    fn tier_fromstr_unknown() {
        let err = "unknown".parse::<MemoryTier>();
        assert!(err.is_err());
    }

    #[test]
    fn tier_serde_roundtrip() {
        let tier = MemoryTier::Hot;
        let json = serde_json::to_string(&tier).unwrap();
        assert_eq!(json, "\"hot\"");
        let back: MemoryTier = serde_json::from_str(&json).unwrap();
        assert_eq!(back, tier);
    }

    #[test]
    fn tier_default() {
        assert_eq!(MemoryTier::default(), MemoryTier::Warm);
    }
}
