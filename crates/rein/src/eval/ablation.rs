//! v0.36 #ablation-harness — multi-arm ablation analysis over eval-gate
//! scorecards.
//!
//! Each "arm" is a [`GateScorecard`] (e.g. produced by `rein-eval gate run`
//! under a different build/config/feature flag). Given N arms and a designated
//! baseline, this computes:
//!
//! * per-arm score + bootstrap 95% CI, and
//! * per-arm paired delta vs the baseline + bootstrap CI on that delta, with a
//!   `significant` flag = the delta CI excludes 0.
//!
//! All arms must share an identical `fixture_id` set (same gate, schema, and
//! corpus fingerprint — verified up front), so deltas are paired per fixture.
//!
//! Reproducibility: the bootstrap uses a SEEDED xorshift64* PRNG, not a system
//! RNG — same arms + same seed always yield the same CIs, which a gate-style
//! tool needs. The paired delta resamples one shared index vector per replicate
//! and applies it to both arms (proper paired bootstrap).

use anyhow::{bail, Result};
use serde::Serialize;
use std::collections::BTreeSet;

use crate::eval::gates::{GateScorecard, SCORECARD_SCHEMA_VERSION};

/// Deterministic xorshift64* PRNG — reproducible bootstrap without a `rand`
/// dependency. Seed is forced nonzero.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform index in `[0, n)`. `n` must be > 0.
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Nearest-rank percentile of an ascending-sorted slice. `p` in `[0, 1]`.
fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn mean_bool(hits: &[bool]) -> f64 {
    if hits.is_empty() {
        return 0.0;
    }
    hits.iter().filter(|h| **h).count() as f64 / hits.len() as f64
}

/// Point score + percentile bootstrap CI over a single arm's per-fixture hits.
pub fn bootstrap_score_ci(
    hits: &[bool],
    resamples: usize,
    confidence: f64,
    seed: u64,
) -> (f64, f64, f64) {
    let point = mean_bool(hits);
    if hits.is_empty() || resamples == 0 {
        return (point, point, point);
    }
    let n = hits.len();
    let mut rng = Rng::new(seed);
    let mut scores = Vec::with_capacity(resamples);
    for _ in 0..resamples {
        let mut hit_count = 0usize;
        for _ in 0..n {
            if hits[rng.below(n)] {
                hit_count += 1;
            }
        }
        scores.push(hit_count as f64 / n as f64);
    }
    scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let tail = (1.0 - confidence) / 2.0;
    (
        point,
        percentile_sorted(&scores, tail),
        percentile_sorted(&scores, 1.0 - tail),
    )
}

/// Paired delta (`arm - baseline`) point estimate + percentile bootstrap CI.
/// `base` and `arm` must be aligned (same fixture order, same length).
pub fn paired_delta_ci(
    base: &[bool],
    arm: &[bool],
    resamples: usize,
    confidence: f64,
    seed: u64,
) -> (f64, f64, f64) {
    let point = mean_bool(arm) - mean_bool(base);
    if base.is_empty() || resamples == 0 {
        return (point, point, point);
    }
    let n = base.len();
    let mut rng = Rng::new(seed);
    let mut deltas = Vec::with_capacity(resamples);
    for _ in 0..resamples {
        let (mut b_hits, mut a_hits) = (0usize, 0usize);
        for _ in 0..n {
            let idx = rng.below(n); // shared index → paired resample
            if base[idx] {
                b_hits += 1;
            }
            if arm[idx] {
                a_hits += 1;
            }
        }
        deltas.push((a_hits as f64 - b_hits as f64) / n as f64);
    }
    deltas.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let tail = (1.0 - confidence) / 2.0;
    (
        point,
        percentile_sorted(&deltas, tail),
        percentile_sorted(&deltas, 1.0 - tail),
    )
}

#[derive(Debug, Clone, Serialize)]
pub struct ArmStats {
    pub label: String,
    pub score: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct PairwiseDelta {
    pub arm: String,
    pub baseline: String,
    /// `arm.score - baseline.score` over the paired fixture intersection.
    pub delta: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
    /// True iff the delta CI excludes 0 (the arm differs from baseline at the
    /// chosen confidence — a real ablation effect, not noise).
    pub significant: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct AblationReport {
    pub baseline: String,
    pub paired_fixture_count: usize,
    pub resamples: usize,
    pub confidence: f64,
    pub seed: u64,
    pub arms: Vec<ArmStats>,
    pub deltas: Vec<PairwiseDelta>,
    pub note: String,
}

/// Run the ablation over `arms` (label → scorecard), paired on their shared
/// fixture-id set (all arms must match gate / schema / corpus / id-set).
/// `baseline_label` must be one of the arm labels.
pub fn run_ablation(
    arms: Vec<(String, GateScorecard)>,
    baseline_label: &str,
    resamples: usize,
    confidence: f64,
    seed: u64,
) -> Result<AblationReport> {
    if arms.len() < 2 {
        bail!("ablation needs >= 2 arms (a baseline + at least one treatment)");
    }
    if !(0.0 < confidence && confidence < 1.0) {
        // OPEN interval: `(0.0..1.0).contains` is half-open and would admit
        // confidence==0.0, which makes tail=0.5 → a zero-width CI (lo==hi==
        // median) → fake significance — the same failure the resamples==0 guard
        // blocks (codex audit v0.36).
        bail!("confidence must be in the open interval (0, 1), got {confidence}");
    }
    if resamples == 0 {
        // 0 makes the bootstrap CI collapse to the point estimate, which would
        // then flag every non-zero delta as "significant" (codex v0.36 #1).
        bail!("resamples must be >= 1");
    }
    // Arm labels must be unique — a repeated label (especially the baseline)
    // would be silently dropped from the delta set / pick an ambiguous arm.
    {
        let mut seen = std::collections::HashSet::new();
        for (l, _) in &arms {
            if !seen.insert(l.as_str()) {
                bail!("duplicate arm label `{l}` — arm labels must be unique");
            }
        }
    }
    if !arms.iter().any(|(l, _)| l == baseline_label) {
        bail!(
            "baseline `{baseline_label}` is not among the arms: {:?}",
            arms.iter().map(|(l, _)| l).collect::<Vec<_>>()
        );
    }
    // Each arm's scorecard must have unique fixture ids — duplicates would be
    // silently overwritten by the lookup map and collapsed by the intersection,
    // scoring against dropped fixtures (mirrors the gate comparator's rule).
    for (label, sc) in &arms {
        let mut seen = std::collections::HashSet::new();
        for f in &sc.per_fixture {
            if !seen.insert(f.fixture_id.as_str()) {
                bail!(
                    "arm `{label}` has duplicate fixture_id `{}` — invalid scorecard",
                    f.fixture_id
                );
            }
        }
    }

    // All arms must share the same gate IDENTITY + corpus AND carry the current
    // scorecard schema. `build_fingerprint` MAY differ — that is precisely the
    // experiment (same gate + corpus, different build / config / feature flag).
    // A different gate, a stale schema (whose empty/legacy fixture_fingerprint
    // can't be trusted), a different corpus fingerprint, or — given a matching
    // fingerprint — a different fixture-id set (a partial/truncated scorecard)
    // would all yield statistically-formatted deltas over incomparable rows.
    // These are the incompatible-input cases the gate comparator rejects.
    let ref_label = &arms[0].0;
    let ref_sc = &arms[0].1;
    let ref_ids: BTreeSet<&str> = ref_sc
        .per_fixture
        .iter()
        .map(|f| f.fixture_id.as_str())
        .collect();
    for (label, sc) in &arms {
        if sc.schema_version != SCORECARD_SCHEMA_VERSION {
            bail!(
                "arm `{label}` scorecard schema_version {} != current {} — re-run/re-baseline \
                 (a stale schema's fixture_fingerprint cannot be trusted)",
                sc.schema_version,
                SCORECARD_SCHEMA_VERSION
            );
        }
        if sc.gate_name != ref_sc.gate_name {
            bail!(
                "arm `{label}` is gate `{}` but `{ref_label}` is gate `{}` — \
                 all arms must be the same gate",
                sc.gate_name,
                ref_sc.gate_name
            );
        }
        if sc.fixture_fingerprint != ref_sc.fixture_fingerprint {
            bail!(
                "arm `{label}` ran on a different fixture corpus than `{ref_label}` \
                 (fixture_fingerprint differs) — arms must share the corpus; only \
                 build/config may differ"
            );
        }
        let ids: BTreeSet<&str> = sc
            .per_fixture
            .iter()
            .map(|f| f.fixture_id.as_str())
            .collect();
        if ids != ref_ids {
            bail!(
                "arm `{label}` has a different fixture-id set than `{ref_label}` despite a \
                 matching corpus fingerprint — partial/truncated scorecard, not a valid arm"
            );
        }
    }
    // Same corpus + identical id-sets verified above → use the full set.
    let common: Vec<String> = ref_ids.iter().map(|s| s.to_string()).collect();
    if common.is_empty() {
        bail!("arms have no fixtures — nothing to compare");
    }

    // Build aligned hit vectors per arm over the common (sorted) id list.
    let aligned: Vec<(String, Vec<bool>)> = arms
        .iter()
        .map(|(label, sc)| {
            let lookup: std::collections::HashMap<&str, bool> = sc
                .per_fixture
                .iter()
                .map(|f| (f.fixture_id.as_str(), f.hit))
                .collect();
            let hits = common
                .iter()
                .map(|id| *lookup.get(id.as_str()).unwrap_or(&false))
                .collect();
            (label.clone(), hits)
        })
        .collect();

    let arm_stats: Vec<ArmStats> = aligned
        .iter()
        .map(|(label, hits)| {
            let (score, lo, hi) = bootstrap_score_ci(hits, resamples, confidence, seed);
            ArmStats {
                label: label.clone(),
                score,
                ci_lower: lo,
                ci_upper: hi,
            }
        })
        .collect();

    let base_hits = &aligned
        .iter()
        .find(|(l, _)| l == baseline_label)
        .expect("baseline presence checked above")
        .1;

    let deltas: Vec<PairwiseDelta> = aligned
        .iter()
        .filter(|(l, _)| l != baseline_label)
        .map(|(label, hits)| {
            let (delta, lo, hi) = paired_delta_ci(base_hits, hits, resamples, confidence, seed);
            PairwiseDelta {
                arm: label.clone(),
                baseline: baseline_label.to_string(),
                delta,
                ci_lower: lo,
                ci_upper: hi,
                significant: lo > 0.0 || hi < 0.0,
            }
        })
        .collect();

    let note = format!(
        "Paired over the shared {}-fixture corpus. Each `delta` = \
         arm.score - baseline.score. For an ablation arm (a feature removed vs \
         the baseline), a NEGATIVE delta means the feature HELPED (removing it \
         lowered the score) and a positive delta means it hurt. `significant` = \
         the {:.0}% bootstrap CI excludes 0. Small corpora → wide CIs; treat as \
         directional.",
        common.len(),
        confidence * 100.0
    );

    Ok(AblationReport {
        baseline: baseline_label.to_string(),
        paired_fixture_count: common.len(),
        resamples,
        confidence,
        seed,
        arms: arm_stats,
        deltas,
        note,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::gates::{
        FixtureResult, GateScorecard, ScorecardKind, SCORECARD_SCHEMA_VERSION,
    };

    // `_arm` documents the intended arm at the call site; the scorecard's
    // `gate_name` is fixed so all arms share a gate identity (the arm LABEL is
    // passed separately to `run_ablation`).
    fn scorecard(_arm: &str, hits: &[(&str, bool)]) -> GateScorecard {
        GateScorecard {
            schema_version: SCORECARD_SCHEMA_VERSION,
            gate_name: "gate".to_string(),
            kind: ScorecardKind::Run,
            created_at: 0,
            rein_version: "test".to_string(),
            build_fingerprint: String::new(),
            fixture_fingerprint: String::new(),
            fixture_count: hits.len(),
            score: hits.iter().filter(|(_, h)| *h).count() as f64 / hits.len().max(1) as f64,
            per_fixture: hits
                .iter()
                .map(|(id, h)| FixtureResult {
                    fixture_id: id.to_string(),
                    hit: *h,
                })
                .collect(),
        }
    }

    #[test]
    fn bootstrap_ci_degenerate_for_all_hits() {
        let (score, lo, hi) = bootstrap_score_ci(&[true, true, true, true], 500, 0.95, 7);
        assert!((score - 1.0).abs() < 1e-9);
        assert!((lo - 1.0).abs() < 1e-9 && (hi - 1.0).abs() < 1e-9);
    }

    #[test]
    fn bootstrap_is_seed_reproducible() {
        let hits = [true, false, true, true, false, true, false, true];
        let a = bootstrap_score_ci(&hits, 1000, 0.95, 42);
        let b = bootstrap_score_ci(&hits, 1000, 0.95, 42);
        assert_eq!(a, b, "same seed must yield identical CI");
    }

    #[test]
    fn paired_delta_zero_for_identical_arms() {
        let hits = [true, false, true, true, false];
        let (delta, lo, hi) = paired_delta_ci(&hits, &hits, 1000, 0.95, 5);
        assert!((delta).abs() < 1e-9);
        assert!(
            (lo).abs() < 1e-9 && (hi).abs() < 1e-9,
            "identical arms → 0 delta, 0-width CI"
        );
    }

    #[test]
    fn run_ablation_pairs_on_intersection_and_flags_significance() {
        // baseline hits all 5; treatment misses 4 of 5 → large negative delta.
        let base = scorecard(
            "full",
            &[
                ("a", true),
                ("b", true),
                ("c", true),
                ("d", true),
                ("e", true),
            ],
        );
        let treat = scorecard(
            "no_kg",
            &[
                ("a", false),
                ("b", false),
                ("c", false),
                ("d", false),
                ("e", true),
            ],
        );
        let rep = run_ablation(
            vec![("full".into(), base), ("no_kg".into(), treat)],
            "full",
            2000,
            0.95,
            1,
        )
        .unwrap();
        assert_eq!(rep.paired_fixture_count, 5);
        assert_eq!(rep.arms.len(), 2);
        assert_eq!(rep.deltas.len(), 1);
        let d = &rep.deltas[0];
        assert!(d.delta < 0.0, "removing kg hurt → negative delta");
        assert!(d.significant, "4/5 drop should be significant at 95%");
    }

    #[test]
    fn run_ablation_rejects_missing_baseline_and_single_arm() {
        let sc = scorecard("only", &[("a", true)]);
        assert!(run_ablation(vec![("only".into(), sc.clone())], "only", 100, 0.95, 1).is_err());
        let sc2 = scorecard("b", &[("a", true)]);
        assert!(run_ablation(
            vec![("only".into(), sc), ("b".into(), sc2)],
            "missing",
            100,
            0.95,
            1
        )
        .is_err());
    }

    #[test]
    fn run_ablation_rejects_zero_resamples_dup_labels_and_dup_fixture_ids() {
        let a = scorecard("full", &[("a", true), ("b", false)]);
        let b = scorecard("arm", &[("a", true), ("b", true)]);
        // zero resamples → would fake significance
        assert!(run_ablation(
            vec![("full".into(), a.clone()), ("arm".into(), b.clone())],
            "full",
            0,
            0.95,
            1
        )
        .is_err());
        // confidence at either open-interval endpoint must be rejected
        // (confidence==0.0 → tail 0.5 → zero-width CI → fake significance).
        for bad in [0.0_f64, 1.0_f64] {
            assert!(
                run_ablation(
                    vec![("full".into(), a.clone()), ("arm".into(), b.clone())],
                    "full",
                    100,
                    bad,
                    1
                )
                .is_err(),
                "confidence {bad} must be rejected (open interval)"
            );
        }
        // duplicate arm label
        assert!(run_ablation(
            vec![("full".into(), a.clone()), ("full".into(), b.clone())],
            "full",
            100,
            0.95,
            1
        )
        .is_err());
        // duplicate fixture id within an arm
        let dup = scorecard("dup", &[("a", true), ("a", false), ("b", true)]);
        assert!(run_ablation(
            vec![("full".into(), a), ("dup".into(), dup)],
            "full",
            100,
            0.95,
            1
        )
        .is_err());
    }

    #[test]
    fn run_ablation_rejects_mismatched_gate_or_corpus() {
        let a = scorecard("a", &[("x", true), ("y", false)]);
        // different gate_name
        let mut b = scorecard("b", &[("x", true), ("y", true)]);
        b.gate_name = "other_gate".to_string();
        assert!(run_ablation(
            vec![("a".into(), a.clone()), ("b".into(), b)],
            "a",
            100,
            0.95,
            1
        )
        .is_err());
        // different fixture corpus (fingerprint differs)
        let mut c = scorecard("c", &[("x", true), ("y", false)]);
        c.fixture_fingerprint = "deadbeefdeadbeefdeadbeefdeadbeef".to_string();
        assert!(run_ablation(
            vec![("a".into(), a.clone()), ("c".into(), c)],
            "a",
            100,
            0.95,
            1
        )
        .is_err());
        // stale scorecard schema version
        let mut old = scorecard("old", &[("x", true), ("y", false)]);
        old.schema_version = SCORECARD_SCHEMA_VERSION - 1;
        assert!(run_ablation(
            vec![("a".into(), a.clone()), ("old".into(), old)],
            "a",
            100,
            0.95,
            1
        )
        .is_err());
        // partial scorecard: same fingerprint, fewer fixture ids
        let partial = scorecard("partial", &[("x", true)]);
        assert!(run_ablation(
            vec![("a".into(), a), ("partial".into(), partial)],
            "a",
            100,
            0.95,
            1
        )
        .is_err());
    }
}
