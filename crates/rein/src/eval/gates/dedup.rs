//! v0.33 dedup gate — pairwise duplicate-detection quality over a fixture
//! corpus of labeled text pairs.
//!
//! Hermetic + pure: scores each pair with `extract::dedup::similarity` (max of
//! Jaccard / containment over normalized tokens) and classifies it a duplicate
//! when `similarity >= DEDUP_THRESHOLD`.  No store, no config, no LLM — the
//! same input always produces the same hit, which is what a reproducible gate
//! needs.  `hit = (similarity(a, b) >= threshold) == is_duplicate`.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::config::ReinConfig;
use crate::eval::gates::{
    fixture_corpus_fingerprint, FixtureResult, Gate, GateScorecard, ScorecardKind,
    SCORECARD_SCHEMA_VERSION,
};
use crate::eval::mcnemar::one_sided_binomial_upper_bound;
use crate::extract::dedup::similarity;
use crate::store::SqliteStore;

/// Classification threshold: the merge bound used by `check_dedup`
/// (`gray_zone_lower_bound` upper).  Documented default for v0.33; production
/// calibration deferred to `docs/backlog/v0.33-eval-gate-calibration.md`.
const DEDUP_THRESHOLD: f32 = 0.50;

pub struct DedupGate;

#[derive(Debug, Deserialize)]
struct DedupFixture {
    id: String,
    text_a: String,
    text_b: String,
    is_duplicate: bool,
}

impl Gate for DedupGate {
    fn name(&self) -> &'static str {
        "dedup"
    }

    fn run(&self, _store: &SqliteStore, _config: &ReinConfig) -> Result<GateScorecard> {
        let (fixtures, fixture_fingerprint) = load_dedup_fixtures()?;
        let mut per_fixture = Vec::with_capacity(fixtures.len());
        for fx in &fixtures {
            per_fixture.push(FixtureResult {
                fixture_id: fx.id.clone(),
                hit: classify_one(fx),
            });
        }
        let hits = per_fixture.iter().filter(|f| f.hit).count() as f64;
        let total = per_fixture.len() as f64;
        let score = if total > 0.0 { hits / total } else { 0.0 };

        Ok(GateScorecard {
            schema_version: SCORECARD_SCHEMA_VERSION,
            gate_name: "dedup".to_string(),
            kind: ScorecardKind::Run,
            created_at: Utc::now().timestamp(),
            rein_version: env!("CARGO_PKG_VERSION").to_string(),
            build_fingerprint: env!("REIN_BUILD_FINGERPRINT").to_string(),
            fixture_fingerprint,
            fixture_count: per_fixture.len(),
            score,
            per_fixture,
        })
    }
}

/// `hit` = the similarity classifier agrees with the ground-truth label.
fn classify_one(fx: &DedupFixture) -> bool {
    let predicted_duplicate = similarity(&fx.text_a, &fx.text_b) >= DEDUP_THRESHOLD;
    predicted_duplicate == fx.is_duplicate
}

/// Fixture dir, resolved at RUNTIME (mirrors `recall::fixture_dir` — no
/// `env!("CARGO_MANIFEST_DIR")` literal baked into the binary).
fn fixture_dir() -> PathBuf {
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        return PathBuf::from(manifest)
            .join("tests")
            .join("fixtures")
            .join("eval_gates")
            .join("dedup");
    }
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join("crates")
        .join("rein")
        .join("tests")
        .join("fixtures")
        .join("eval_gates")
        .join("dedup")
}

/// Load all `case_*.json` dedup fixtures; returns `(fixtures, fingerprint)`
/// where the fingerprint is computed over the bytes actually read.
fn load_dedup_fixtures() -> Result<(Vec<DedupFixture>, String)> {
    let dir = fixture_dir();
    let entries =
        std::fs::read_dir(&dir).with_context(|| format!("read fixture dir {}", dir.display()))?;
    let mut paths: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            let Some(name) = p.file_name().and_then(|n| n.to_str()) else {
                return false;
            };
            name.starts_with("case_") && name.ends_with(".json") && p.is_file()
        })
        .collect();
    paths.sort();

    if paths.is_empty() {
        bail!(
            "dedup gate found no fixtures matching `case_*.json` in {}. \
             A 0-fixture scorecard silently disables the gate; fixtures live at \
             `crates/rein/tests/fixtures/eval_gates/dedup/case_*.json`. \
             Run rein-eval from the source repo (`cargo run -p rein --bin rein-eval`).",
            dir.display(),
        );
    }

    let mut fixtures = Vec::with_capacity(paths.len());
    let mut corpus: Vec<(String, Vec<u8>)> = Vec::with_capacity(paths.len());
    for path in paths {
        let bytes = std::fs::read(&path)
            .with_context(|| format!("read dedup fixture {}", path.display()))?;
        let fx: DedupFixture = serde_json::from_slice(&bytes)
            .with_context(|| format!("parse dedup fixture {}", path.display()))?;
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_string();
        corpus.push((name, bytes));
        fixtures.push(fx);
    }
    fixtures.sort_by(|a, b| a.id.cmp(&b.id));
    let fingerprint = fixture_corpus_fingerprint(corpus);
    Ok((fixtures, fingerprint))
}

// ---------------------------------------------------------------------------
// v0.36 #C2 — data-driven threshold sweep
// ---------------------------------------------------------------------------

/// The production global-fallback dedup threshold (`AdaptiveState::
/// default_global_dedup_threshold`). Surfaced in the sweep report so the
/// operator can compare the corpus-optimal point against what cold-start
/// production actually uses. Kept in sync with that constant.
pub const PRODUCTION_DEFAULT_THRESHOLD: f32 = 0.70;

/// One row of the precision/recall sweep at a fixed similarity threshold.
/// "Positive" = duplicate. Pure function of the labeled corpus, so the same
/// corpus always yields the same curve (no LLM, no store).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdStat {
    pub threshold: f32,
    pub tp: usize,
    pub fp: usize,
    pub tn: usize,
    pub false_neg: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub accuracy: f64,
}

/// False-positive evidence budget for destructive hard-merge promotion.
pub const FALSE_POSITIVE_BUDGET: f64 = 0.02;
const FALSE_POSITIVE_ALPHA: f64 = 0.05;
const ZERO_FP_REQUIRED_NEGATIVES: usize = 149;

/// False-positive safety verdict for an independently fixed threshold evaluated
/// only against a sealed negative holdout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FalsePositiveSafetyStatus {
    Ship,
    Bail,
    NoData,
}

/// Full sweep report: the per-threshold curve plus the data-derived optimum.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DedupSweepReport {
    pub fixture_count: usize,
    pub positives: usize,
    pub negatives: usize,
    pub fixture_fingerprint: String,
    /// The gate's classifier threshold today (`DEDUP_THRESHOLD`).
    pub current_gate_threshold: f32,
    /// Production cold-start global fallback (`PRODUCTION_DEFAULT_THRESHOLD`).
    pub production_default_threshold: f32,
    pub curve: Vec<ThresholdStat>,
    /// Directional zero-observed-false-positive curve point: the threshold with
    /// the highest recall among rows achieving precision == 1.0 on this corpus.
    /// This discovery-only row is not a production hard-threshold
    /// recommendation. `None` if no row reaches precision 1.0.
    pub merge_safe_optimal: Option<ThresholdStat>,
    /// Number of distinct-labeled pairs in the discovery corpus. These pairs
    /// participate in threshold selection and are not a sealed holdout.
    pub discovery_negative_count: usize,
    /// Number of distinct-labeled pairs in an untouched sealed holdout. The
    /// bundled sweep currently has no such holdout, so this is zero.
    pub sealed_negative_holdout_count: usize,
    /// Exact one-sided 95% Clopper-Pearson upper bound from sealed negatives
    /// only. `None` when no independently fixed threshold or holdout exists.
    pub false_positive_upper_95: Option<f64>,
    /// Maximum acceptable false-positive rate for a destructive merge bound.
    pub false_positive_budget: f64,
    /// Safety evidence only, not a full threshold-promotion verdict.
    pub false_positive_safety_status: FalsePositiveSafetyStatus,
    /// Human-readable sealed-holdout evidence and fail-closed action.
    pub false_positive_safety_reason: String,
    /// SECONDARY, informational only: the max-F1 point. Do NOT use this as a
    /// merge bound — it can carry false positives (here it does), i.e. data
    /// loss. Reported so the precision/recall trade-off is visible.
    pub max_f1_point: ThresholdStat,
    pub power_note: String,
}

/// Sweep similarity thresholds over `[0.30, 0.95]` step `0.05`, scoring the
/// labeled corpus at each. `similarity` is computed once per pair.
pub fn sweep_thresholds(sims: &[(f32, bool)]) -> Vec<ThresholdStat> {
    let n = sims.len();
    let mut curve = Vec::new();
    // Integer stepping avoids f32 accumulation drift: 30, 35, … 95.
    let mut step = 30u32;
    while step <= 95 {
        let threshold = step as f32 / 100.0;
        let (mut tp, mut fp, mut tn, mut false_neg) = (0usize, 0usize, 0usize, 0usize);
        for (sim, is_dup) in sims {
            match (*sim >= threshold, *is_dup) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, false) => tn += 1,
                (false, true) => false_neg += 1,
            }
        }
        let precision = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };
        let recall = if tp + false_neg > 0 {
            tp as f64 / (tp + false_neg) as f64
        } else {
            0.0
        };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        let accuracy = if n > 0 {
            (tp + tn) as f64 / n as f64
        } else {
            0.0
        };
        curve.push(ThresholdStat {
            threshold,
            tp,
            fp,
            tn,
            false_neg,
            precision,
            recall,
            f1,
            accuracy,
        });
        step += 5;
    }
    curve
}

/// Pick the optimum: max F1, tie-broken by higher accuracy then HIGHER
/// threshold (conservative — prefers the bound less likely to over-merge).
pub fn optimal_threshold(curve: &[ThresholdStat]) -> ThresholdStat {
    curve
        .iter()
        .cloned()
        .max_by(|a, b| {
            a.f1.partial_cmp(&b.f1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(
                    a.accuracy
                        .partial_cmp(&b.accuracy)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
                .then(
                    a.threshold
                        .partial_cmp(&b.threshold)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
        })
        .expect("sweep curve is never empty (fixed 0.30..=0.95 grid)")
}

/// Pick the directional zero-observed-FP point: among thresholds with
/// precision == 1.0 on this corpus, choose the highest recall and tie-break at
/// the lower threshold. This curve row is not powered safety evidence and must
/// not be used directly as a production hard auto-merge bound.
pub fn merge_safe_threshold(curve: &[ThresholdStat]) -> Option<ThresholdStat> {
    curve
        .iter()
        .filter(|s| s.fp == 0 && s.tp > 0)
        .cloned()
        .max_by(|a, b| {
            a.recall
                .partial_cmp(&b.recall)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(
                    b.threshold
                        .partial_cmp(&a.threshold)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
        })
}

#[derive(Debug)]
struct FalsePositiveSafetyAssessment {
    fixed_threshold: f32,
    sealed_negative_holdout_count: usize,
    observed_false_positives: usize,
    false_positive_upper_95: Option<f64>,
    false_positive_safety_status: FalsePositiveSafetyStatus,
    false_positive_safety_reason: String,
}

fn assess_false_positive_safety(
    fixed_threshold: f32,
    sealed_negative_sims: &[f32],
) -> FalsePositiveSafetyAssessment {
    let sealed_negative_holdout_count = sealed_negative_sims.len();
    let observed_false_positives = sealed_negative_sims
        .iter()
        .filter(|&&sim| sim > fixed_threshold)
        .count();

    if sealed_negative_holdout_count == 0 {
        return FalsePositiveSafetyAssessment {
            fixed_threshold,
            sealed_negative_holdout_count,
            observed_false_positives,
            false_positive_upper_95: None,
            false_positive_safety_status: FalsePositiveSafetyStatus::NoData,
            false_positive_safety_reason: format!(
                "NoData: sealed negative holdout is missing for fixed_threshold={fixed_threshold:.6}; false-positive safety cannot be assessed."
            ),
        };
    }

    let (failures, trials) = match (
        u32::try_from(observed_false_positives),
        u32::try_from(sealed_negative_holdout_count),
    ) {
        (Ok(failures), Ok(trials)) => (failures, trials),
        _ => {
            return FalsePositiveSafetyAssessment {
                fixed_threshold,
                sealed_negative_holdout_count,
                observed_false_positives,
                false_positive_upper_95: None,
                false_positive_safety_status: FalsePositiveSafetyStatus::Bail,
                false_positive_safety_reason:
                    "Bail: sealed negative holdout counts exceed the exact-bound API range."
                        .to_string(),
            };
        }
    };

    let Some(upper) = one_sided_binomial_upper_bound(failures, trials, FALSE_POSITIVE_ALPHA) else {
        return FalsePositiveSafetyAssessment {
            fixed_threshold,
            sealed_negative_holdout_count,
            observed_false_positives,
            false_positive_upper_95: None,
            false_positive_safety_status: FalsePositiveSafetyStatus::Bail,
            false_positive_safety_reason: format!(
                "Bail: observed={sealed_negative_holdout_count}, false_positives={observed_false_positives}; UCB=unavailable."
            ),
        };
    };

    let false_positive_safety_status = if sealed_negative_holdout_count < ZERO_FP_REQUIRED_NEGATIVES
        && observed_false_positives == 0
    {
        FalsePositiveSafetyStatus::NoData
    } else if observed_false_positives > 0 && upper > FALSE_POSITIVE_BUDGET {
        FalsePositiveSafetyStatus::Bail
    } else if sealed_negative_holdout_count >= ZERO_FP_REQUIRED_NEGATIVES
        && upper <= FALSE_POSITIVE_BUDGET
    {
        FalsePositiveSafetyStatus::Ship
    } else {
        FalsePositiveSafetyStatus::Bail
    };
    let false_positive_safety_reason = match false_positive_safety_status {
        FalsePositiveSafetyStatus::Ship => format!(
            "Ship: fixed_threshold={fixed_threshold:.6}, observed={sealed_negative_holdout_count}, false_positives={observed_false_positives}, exact one-sided 95% UCB={upper:.6} <= budget={FALSE_POSITIVE_BUDGET:.6}. This passes the sealed-holdout safety gate only."
        ),
        FalsePositiveSafetyStatus::Bail => format!(
            "Bail: fixed_threshold={fixed_threshold:.6}, observed={sealed_negative_holdout_count}, false_positives={observed_false_positives}, exact one-sided 95% UCB={upper:.6} does not establish safety at budget={FALSE_POSITIVE_BUDGET:.6}."
        ),
        FalsePositiveSafetyStatus::NoData => format!(
            "NoData: required={ZERO_FP_REQUIRED_NEGATIVES} zero-false-positive sealed negative holdout cases; observed={sealed_negative_holdout_count}, false_positives={observed_false_positives}, exact one-sided 95% UCB={upper:.6} exceeds budget={FALSE_POSITIVE_BUDGET:.6}."
        ),
    };

    FalsePositiveSafetyAssessment {
        fixed_threshold,
        sealed_negative_holdout_count,
        observed_false_positives,
        false_positive_upper_95: Some(upper),
        false_positive_safety_status,
        false_positive_safety_reason,
    }
}

/// Load the dedup corpus and run the full sweep. Pure + hermetic (same
/// fixtures → same report).
pub fn run_dedup_sweep() -> Result<DedupSweepReport> {
    let (fixtures, fixture_fingerprint) = load_dedup_fixtures()?;
    let positives = fixtures.iter().filter(|f| f.is_duplicate).count();
    let negatives = fixtures.len() - positives;
    let sims: Vec<(f32, bool)> = fixtures
        .iter()
        .map(|f| (similarity(&f.text_a, &f.text_b), f.is_duplicate))
        .collect();
    let curve = sweep_thresholds(&sims);
    let max_f1_point = optimal_threshold(&curve);
    let merge_safe_optimal = merge_safe_threshold(&curve);
    let safety = assess_false_positive_safety(PRODUCTION_DEFAULT_THRESHOLD, &[]);
    let power_note = format!(
        "The bundled fixture corpus is discovery only: it sweeps LEXICAL \
         similarity on n={} ({} duplicate / {} distinct), so its {} distinct \
         pairs are not a sealed holdout and no 95% coverage bound is reported. \
         false_positive_safety_status={:?}, fixed_threshold={:.6}, \
         sealed_negative_holdout_count={}, observed_false_positives={}, \
         false_positive_upper_95=unavailable, budget={:.6}. merge_safe_optimal \
         remains directional only. A complete promotion decision additionally \
         requires paired McNemar evidence, slice safety checks, and a pinned \
         sealed-holdout fingerprint; this report exposes no promotion Ship field.",
        fixtures.len(),
        positives,
        negatives,
        negatives,
        safety.false_positive_safety_status,
        safety.fixed_threshold,
        safety.sealed_negative_holdout_count,
        safety.observed_false_positives,
        FALSE_POSITIVE_BUDGET,
    );
    Ok(DedupSweepReport {
        fixture_count: fixtures.len(),
        positives,
        negatives,
        fixture_fingerprint,
        current_gate_threshold: DEDUP_THRESHOLD,
        production_default_threshold: PRODUCTION_DEFAULT_THRESHOLD,
        curve,
        merge_safe_optimal,
        discovery_negative_count: negatives,
        sealed_negative_holdout_count: safety.sealed_negative_holdout_count,
        false_positive_upper_95: safety.false_positive_upper_95,
        false_positive_budget: FALSE_POSITIVE_BUDGET,
        false_positive_safety_status: safety.false_positive_safety_status,
        false_positive_safety_reason: safety.false_positive_safety_reason,
        max_f1_point,
        power_note,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ReinConfig;
    use crate::store::SqliteStore;

    #[test]
    fn dedup_gate_is_not_stub() {
        assert!(!DedupGate.is_stub());
        assert_eq!(DedupGate.name(), "dedup");
    }

    // ---- v0.36 #C2 sweep ----

    #[test]
    fn sweep_produces_full_grid_030_to_095() {
        let curve = sweep_thresholds(&[(1.0, true), (0.0, false)]);
        assert_eq!(curve.len(), 14, "0.30..=0.95 step 0.05 → 14 rows");
        assert!((curve.first().unwrap().threshold - 0.30).abs() < 1e-6);
        assert!((curve.last().unwrap().threshold - 0.95).abs() < 1e-6);
    }

    #[test]
    fn sweep_metrics_correct_at_050() {
        // sims chosen so threshold 0.50 classifies all four correctly.
        let sims = [(1.0, true), (0.0, false), (0.6, true), (0.4, false)];
        let curve = sweep_thresholds(&sims);
        let row = curve
            .iter()
            .find(|r| (r.threshold - 0.50).abs() < 1e-6)
            .expect("0.50 row exists");
        assert_eq!((row.tp, row.fp, row.tn, row.false_neg), (2, 0, 2, 0));
        assert!((row.precision - 1.0).abs() < 1e-9);
        assert!((row.recall - 1.0).abs() < 1e-9);
        assert!((row.f1 - 1.0).abs() < 1e-9);
        assert!((row.accuracy - 1.0).abs() < 1e-9);
    }

    #[test]
    fn optimal_prefers_higher_threshold_on_f1_tie() {
        // Perfectly separable → F1 == 1.0 at every threshold in (0.0, 1.0].
        // Conservative tie-break must pick the HIGHEST such threshold.
        let curve = sweep_thresholds(&[(1.0, true), (0.0, false)]);
        let opt = optimal_threshold(&curve);
        assert!((opt.f1 - 1.0).abs() < 1e-9);
        assert!(
            (opt.threshold - 0.95).abs() < 1e-6,
            "tie-break should prefer the most conservative (highest) threshold, got {}",
            opt.threshold
        );
    }

    #[test]
    fn production_default_constant_documents_070() {
        // Guards against silent drift from `default_global_dedup_threshold`.
        assert!((PRODUCTION_DEFAULT_THRESHOLD - 0.70).abs() < 1e-6);
    }

    #[test]
    fn run_dedup_sweep_on_real_corpus_is_sane() {
        let report = run_dedup_sweep().expect("sweep runs on the bundled corpus");
        let value = serde_json::to_value(&report).unwrap();
        assert_eq!(report.curve.len(), 14);
        assert!(report.positives > 0 && report.negatives > 0);
        assert_eq!(report.fixture_count, report.positives + report.negatives);
        assert!(report.max_f1_point.f1 >= 0.0 && report.max_f1_point.f1 <= 1.0);
        assert!((report.current_gate_threshold - DEDUP_THRESHOLD).abs() < 1e-6);
        // If a merge-safe point exists, it must have zero false positives.
        if let Some(ref ms) = report.merge_safe_optimal {
            assert_eq!(ms.fp, 0, "merge-safe optimum must have precision 1.0");
        }
        assert_eq!(value["discovery_negative_count"], 10);
        assert_eq!(value["sealed_negative_holdout_count"], 0);
        assert!(value["false_positive_upper_95"].is_null());
        assert_eq!(value["false_positive_budget"], 0.02);
        assert_eq!(value["false_positive_safety_status"], "no_data");
        assert!(value["false_positive_safety_reason"]
            .as_str()
            .unwrap()
            .contains("sealed"));
        assert!(value.get("hard_promotion_status").is_none());
        assert!(value.get("hard_promotion_reason").is_none());
        assert!(report.power_note.contains("discovery only"));
        assert!(report.power_note.contains("McNemar"));
        assert!(report.power_note.contains("slice"));
        assert!(report.power_note.contains("fingerprint"));
    }

    #[test]
    fn fixed_threshold_ships_with_149_clean_sealed_negatives() {
        let fixed_threshold = 0.80;
        let sealed_negative_sims = vec![0.79; 149];
        let assessment = assess_false_positive_safety(fixed_threshold, &sealed_negative_sims);

        assert_eq!(assessment.fixed_threshold, fixed_threshold);
        assert_eq!(assessment.sealed_negative_holdout_count, 149);
        assert_eq!(assessment.observed_false_positives, 0);
        assert_eq!(
            assessment.false_positive_safety_status,
            FalsePositiveSafetyStatus::Ship
        );
        assert!(assessment.false_positive_upper_95.unwrap() <= 0.02);
        assert!(assessment
            .false_positive_safety_reason
            .contains("observed=149"));
    }

    #[test]
    fn fixed_threshold_is_no_data_with_148_clean_sealed_negatives() {
        let fixed_threshold = 0.80;
        let sealed_negative_sims = vec![0.79; 148];
        let assessment = assess_false_positive_safety(fixed_threshold, &sealed_negative_sims);

        assert_eq!(
            assessment.false_positive_safety_status,
            FalsePositiveSafetyStatus::NoData
        );
        assert!(assessment.false_positive_upper_95.unwrap() > 0.02);
        assert!(assessment
            .false_positive_safety_reason
            .contains("required=149"));
    }

    #[test]
    fn fixed_threshold_bails_on_observed_sealed_false_positive() {
        let fixed_threshold = 0.80;
        let mut sealed_negative_sims = vec![0.79; 149];
        sealed_negative_sims[0] = 0.81;
        let assessment = assess_false_positive_safety(fixed_threshold, &sealed_negative_sims);

        assert_eq!(assessment.observed_false_positives, 1);
        assert_eq!(
            assessment.false_positive_safety_status,
            FalsePositiveSafetyStatus::Bail
        );
        assert!(assessment.false_positive_upper_95.unwrap() > 0.02);
        assert!(assessment
            .false_positive_safety_reason
            .contains("false_positives=1"));
    }

    #[test]
    fn dedup_sweep_report_json_roundtrips_false_positive_safety_evidence() {
        let report = run_dedup_sweep().expect("sweep runs on the bundled corpus");
        let json = serde_json::to_string(&report).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        let decoded: DedupSweepReport = serde_json::from_str(&json).unwrap();
        let roundtripped = serde_json::to_value(&decoded).unwrap();

        assert_eq!(value["discovery_negative_count"], 10);
        assert_eq!(value["sealed_negative_holdout_count"], 0);
        assert!(value["false_positive_upper_95"].is_null());
        assert_eq!(value["false_positive_safety_status"], "no_data");
        assert!(value.get("hard_promotion_status").is_none());
        assert_eq!(
            roundtripped["discovery_negative_count"],
            value["discovery_negative_count"]
        );
        assert_eq!(
            roundtripped["false_positive_safety_status"],
            value["false_positive_safety_status"]
        );
    }

    #[test]
    fn merge_safe_prefers_lowest_zero_fp_threshold() {
        // Perfectly separable → precision 1.0 and recall 1.0 at every threshold;
        // the directional tie-break picks the LOWEST threshold.
        let curve = sweep_thresholds(&[(1.0, true), (0.0, false)]);
        let ms = merge_safe_threshold(&curve).expect("a zero-FP threshold exists");
        assert_eq!(ms.fp, 0);
        assert!((ms.recall - 1.0).abs() < 1e-9);
        assert!((ms.threshold - 0.30).abs() < 1e-6);
    }

    #[test]
    fn merge_safe_is_none_when_no_threshold_is_clean() {
        // A lone distinct pair with high lexical similarity: every threshold
        // that predicts it a duplicate is a false positive, and the thresholds
        // above it make no positive prediction at all → no merge-safe point.
        let curve = sweep_thresholds(&[(0.9, false)]);
        assert!(merge_safe_threshold(&curve).is_none());
    }

    #[test]
    fn merge_safe_rejects_any_observed_false_positive() {
        // A floating-point precision tolerance can hide one FP when TP is very
        // large. Candidate eligibility must use the exact confusion counts.
        let tp = 2_000_000_000usize;
        let row = ThresholdStat {
            threshold: 0.80,
            tp,
            fp: 1,
            tn: 149,
            false_neg: 0,
            precision: tp as f64 / (tp + 1) as f64,
            recall: 1.0,
            f1: 1.0,
            accuracy: 1.0,
        };

        assert!(merge_safe_threshold(&[row]).is_none());
    }

    #[test]
    fn dedup_gate_loads_fixtures() {
        let (fixtures, fingerprint) =
            load_dedup_fixtures().expect("dedup fixture dir must be readable");
        assert!(!fixtures.is_empty(), "dedup corpus is empty");
        assert_eq!(fingerprint.len(), 32, "fingerprint must be 32 hex chars");
        // Both classes present so the gate exercises true/false branches.
        assert!(
            fixtures.iter().any(|f| f.is_duplicate),
            "corpus has no duplicate-labeled pairs"
        );
        assert!(
            fixtures.iter().any(|f| !f.is_duplicate),
            "corpus has no distinct-labeled pairs"
        );
    }

    #[test]
    fn dedup_gate_run_returns_scorecard() {
        let store = SqliteStore::in_memory().unwrap();
        let config = ReinConfig::default();
        let sc = DedupGate.run(&store, &config).unwrap();
        let (fixtures, fingerprint) = load_dedup_fixtures().unwrap();
        assert_eq!(sc.gate_name, "dedup");
        assert_eq!(sc.schema_version, SCORECARD_SCHEMA_VERSION);
        assert_eq!(sc.kind, ScorecardKind::Run);
        assert_eq!(sc.fixture_count, fixtures.len());
        assert_eq!(sc.fixture_fingerprint, fingerprint);
        assert!(sc.score >= 0.0 && sc.score <= 1.0);
        // per_fixture emitted in sorted id order (stable McNemar pairing).
        let ids: Vec<String> = sc
            .per_fixture
            .iter()
            .map(|f| f.fixture_id.clone())
            .collect();
        let mut sorted = ids.clone();
        sorted.sort();
        assert_eq!(ids, sorted);
    }

    #[test]
    fn dedup_classify_exact_duplicate_and_unrelated() {
        let dup = DedupFixture {
            id: "t_dup".to_string(),
            text_a: "Connection pooling reuses open database connections".to_string(),
            text_b: "Connection pooling reuses open database connections".to_string(),
            is_duplicate: true,
        };
        assert!(
            classify_one(&dup),
            "identical text must classify as duplicate"
        );

        let distinct = DedupFixture {
            id: "t_distinct".to_string(),
            text_a: "I adopted a cat from the shelter last weekend".to_string(),
            text_b: "The Kubernetes cluster runs three worker nodes".to_string(),
            is_duplicate: false,
        };
        assert!(
            classify_one(&distinct),
            "unrelated text must classify as distinct (hit = correct negative)"
        );
    }
}
