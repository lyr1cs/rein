//! v0.33 dedup gate — pairwise duplicate-detection quality over a fixture
//! corpus of labeled text pairs.
//!
//! Hermetic + pure: scores each pair with `extract::dedup::similarity` (max of
//! Jaccard / containment over normalized tokens) and classifies it a duplicate
//! when `similarity > DEDUP_THRESHOLD`.  No store, no config, no LLM — the
//! same input always produces the same hit, which is what a reproducible gate
//! needs.  `hit = (similarity(a, b) > threshold) == is_duplicate`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::config::ReinConfig;
use crate::eval::gates::{
    fixture_corpus_fingerprint, FixtureResult, Gate, GateScorecard, ScorecardKind,
    SCORECARD_SCHEMA_VERSION,
};
use crate::eval::mcnemar::{mcnemar, one_sided_binomial_upper_bound, PairedOutcome};
use crate::extract::dedup::similarity;
use crate::store::dedup_calibration::{
    canonical_policy_digest, load_dedup_calibration, load_dedup_calibration_for_runtime,
    load_dedup_calibration_seal, save_dedup_calibration_bundle_cas, DedupCalibrationConfusion,
    DedupCalibrationLoadStatus, DedupCalibrationPolicy, DedupCalibrationProvenance,
    DedupCalibrationSeal, DedupCalibrationSlice, DedupCalibrationStatus, DedupUtilityEvidence,
};
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
    let predicted_duplicate = similarity(&fx.text_a, &fx.text_b) > DEDUP_THRESHOLD;
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
/// labeled corpus at each with the production hard-boundary rule
/// `similarity > threshold`. `similarity` is computed once per pair.
pub fn sweep_thresholds(sims: &[(f32, bool)]) -> Vec<ThresholdStat> {
    let n = sims.len();
    let mut curve = Vec::new();
    // Integer stepping avoids f32 accumulation drift: 30, 35, … 95.
    let mut step = 30u32;
    while step <= 95 {
        let threshold = step as f32 / 100.0;
        let (mut tp, mut fp, mut tn, mut false_neg) = (0usize, 0usize, 0usize, 0usize);
        for (sim, is_dup) in sims {
            match (*sim > threshold, *is_dup) {
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

// ---------------------------------------------------------------------------
// #C2 — family-disjoint sealed calibration policy
// ---------------------------------------------------------------------------

const DEDUP_CALIBRATION_HOLDOUT_FOLD: u8 = 0;
const DEDUP_CALIBRATION_FOLD_COUNT: u8 = 5;
const MIN_CALIBRATION_TRAIN_PER_CLASS: usize = 10;
const MIN_CALIBRATION_POSITIVE_HOLDOUT: usize = 149;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum DedupCalibrationLabel {
    Duplicate,
    Distinct,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DedupCalibrationCaseSource {
    ExactContentHash,
    CanonicalFamily,
    StructuralContradiction,
    StructuralNonce,
    OperatorDistinct,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DedupCalibrationSliceKind {
    OperatorDistinct,
    CanonicalFamily,
    StructuralChallenge,
}

impl DedupCalibrationSliceKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::OperatorDistinct => "operator_distinct",
            Self::CanonicalFamily => "canonical_family",
            Self::StructuralChallenge => "structural_challenge",
        }
    }

    fn has_required_counts(self, positive_count: usize, negative_count: usize) -> bool {
        match self {
            Self::OperatorDistinct => negative_count > 0,
            Self::CanonicalFamily => positive_count > 0,
            Self::StructuralChallenge => positive_count > 0 && negative_count > 0,
        }
    }
}

const REQUIRED_DEDUP_CALIBRATION_SLICE_KINDS: [DedupCalibrationSliceKind; 3] = [
    DedupCalibrationSliceKind::OperatorDistinct,
    DedupCalibrationSliceKind::CanonicalFamily,
    DedupCalibrationSliceKind::StructuralChallenge,
];

impl DedupCalibrationCaseSource {
    fn is_valid_for(self, label: DedupCalibrationLabel) -> bool {
        match label {
            DedupCalibrationLabel::Duplicate => {
                matches!(self, Self::ExactContentHash | Self::CanonicalFamily)
            }
            DedupCalibrationLabel::Distinct => matches!(
                self,
                Self::StructuralContradiction | Self::StructuralNonce | Self::OperatorDistinct
            ),
        }
    }

    fn expected_slice(self) -> DedupCalibrationSliceKind {
        match self {
            Self::CanonicalFamily => DedupCalibrationSliceKind::CanonicalFamily,
            Self::OperatorDistinct => DedupCalibrationSliceKind::OperatorDistinct,
            Self::ExactContentHash | Self::StructuralContradiction | Self::StructuralNonce => {
                DedupCalibrationSliceKind::StructuralChallenge
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DedupCalibrationCase {
    case_id: String,
    /// Independence and train/holdout splitting unit. Multiple rows in the
    /// same family collapse to one deterministic observation.
    family_id: String,
    /// Real canonical roots touched by this evidence. Canonical positives have
    /// one root; operator-distinct evidence has two. Synthetic challenges have
    /// none and remain independent by `family_id`.
    root_keys: Vec<String>,
    /// Connected-component key assigned when the corpus is sealed. ESS
    /// collapse uses this key so A-B and A-C cannot count as two independent
    /// negatives. Fold assignment remains permanently rooted in `root_keys`.
    split_group_id: String,
    similarity: f32,
    label: DedupCalibrationLabel,
    source: DedupCalibrationCaseSource,
    slice: DedupCalibrationSliceKind,
    /// Hash of the exact pair bytes or immutable external-label record. This
    /// makes a text/rubric change invalidate the corpus even when its scalar
    /// similarity happens to remain numerically equal.
    evidence_fingerprint: String,
}

fn hash_case_parts(parts: &[&[u8]]) -> String {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update((part.len() as u64).to_be_bytes());
        hasher.update(part);
    }
    format!("{:x}", hasher.finalize())
}

impl DedupCalibrationCase {
    fn from_verified_pair(
        family_domain: &str,
        immutable_family_key: &str,
        immutable_evidence_key: &str,
        mut root_keys: Vec<String>,
        left: &str,
        right: &str,
        label: DedupCalibrationLabel,
        source: DedupCalibrationCaseSource,
        slice: DedupCalibrationSliceKind,
    ) -> Option<Self> {
        if immutable_family_key.is_empty()
            || immutable_evidence_key.is_empty()
            || left.is_empty()
            || right.is_empty()
        {
            return None;
        }
        if (label == DedupCalibrationLabel::Distinct
            || source == DedupCalibrationCaseSource::CanonicalFamily)
            && left == right
        {
            return None;
        }
        let family_id = hash_case_parts(&[
            b"dedup-calibration-family-v1",
            family_domain.as_bytes(),
            immutable_family_key.as_bytes(),
        ]);
        let evidence_fingerprint = hash_case_parts(&[
            b"dedup-calibration-evidence-v1",
            family_domain.as_bytes(),
            immutable_evidence_key.as_bytes(),
            left.as_bytes(),
            right.as_bytes(),
        ]);
        root_keys.sort_unstable();
        root_keys.dedup();
        Some(Self {
            case_id: format!("dc_{}", &evidence_fingerprint[..24]),
            split_group_id: family_id.clone(),
            family_id,
            root_keys,
            similarity: similarity(left, right),
            label,
            source,
            slice,
            evidence_fingerprint,
        })
    }

    pub(crate) fn exact_content_positive(immutable_record_id: &str, content: &str) -> Option<Self> {
        Self::from_verified_pair(
            "exact_content",
            immutable_record_id,
            immutable_record_id,
            Vec::new(),
            content,
            content,
            DedupCalibrationLabel::Duplicate,
            DedupCalibrationCaseSource::ExactContentHash,
            DedupCalibrationSliceKind::StructuralChallenge,
        )
    }

    pub(crate) fn canonical_family_positive(
        canonical_root_id: &str,
        immutable_evidence_id: &str,
        left: &str,
        right: &str,
    ) -> Option<Self> {
        Self::from_verified_pair(
            "canonical_family",
            canonical_root_id,
            immutable_evidence_id,
            vec![canonical_root_id.to_string()],
            left,
            right,
            DedupCalibrationLabel::Duplicate,
            DedupCalibrationCaseSource::CanonicalFamily,
            DedupCalibrationSliceKind::CanonicalFamily,
        )
    }

    pub(crate) fn operator_distinct(
        immutable_decision_id: &str,
        left_canonical_root: &str,
        right_canonical_root: &str,
        left: &str,
        right: &str,
    ) -> Option<Self> {
        let (left_root, right_root) = if left_canonical_root <= right_canonical_root {
            (left_canonical_root, right_canonical_root)
        } else {
            (right_canonical_root, left_canonical_root)
        };
        if left_root == right_root {
            return None;
        }
        // A root's fold is permanent. Cross-fold edges are quarantined instead
        // of allowing a later connected-component expansion to move old
        // evidence between discovery and the sealed holdout.
        if dedup_calibration_fold(left_root) != dedup_calibration_fold(right_root) {
            return None;
        }
        let family_key = format!("{left_root}\0{right_root}");
        Self::from_verified_pair(
            "operator_distinct",
            &family_key,
            immutable_decision_id,
            vec![left_root.to_string(), right_root.to_string()],
            left,
            right,
            DedupCalibrationLabel::Distinct,
            DedupCalibrationCaseSource::OperatorDistinct,
            DedupCalibrationSliceKind::OperatorDistinct,
        )
    }

    pub(crate) fn structural_contradiction(
        probe_set_version: &str,
        template_id: &str,
        left: &str,
        right: &str,
    ) -> Option<Self> {
        let family_key = format!("{probe_set_version}\0{template_id}");
        Self::from_verified_pair(
            "structural_contradiction",
            &family_key,
            &family_key,
            Vec::new(),
            left,
            right,
            DedupCalibrationLabel::Distinct,
            DedupCalibrationCaseSource::StructuralContradiction,
            DedupCalibrationSliceKind::StructuralChallenge,
        )
    }

    pub(crate) fn structural_nonce(
        probe_set_version: &str,
        template_id: &str,
        left: &str,
        right: &str,
    ) -> Option<Self> {
        let family_key = format!("{probe_set_version}\0{template_id}");
        Self::from_verified_pair(
            "structural_nonce",
            &family_key,
            &family_key,
            Vec::new(),
            left,
            right,
            DedupCalibrationLabel::Distinct,
            DedupCalibrationCaseSource::StructuralNonce,
            DedupCalibrationSliceKind::StructuralChallenge,
        )
    }
}

/// Stable key split: fold 0 is the permanent sealed holdout; folds 1..4 are
/// discovery/training. Real evidence is assigned from each immutable canonical
/// root, never from the mutable membership of its connected component.
fn dedup_calibration_fold(stable_key: &str) -> u8 {
    let digest = Sha256::digest(stable_key.as_bytes());
    digest[0] % DEDUP_CALIBRATION_FOLD_COUNT
}

/// Return the permanent fold for a case. Synthetic challenges use their fixed
/// family id. Root-backed cases are accepted only when every touched root was
/// independently preassigned to the same fold.
fn dedup_calibration_case_fold(case: &DedupCalibrationCase) -> Option<u8> {
    let mut root_folds = case
        .root_keys
        .iter()
        .map(|root| dedup_calibration_fold(root));
    let first = root_folds
        .next()
        .unwrap_or_else(|| dedup_calibration_fold(&case.family_id));
    root_folds.all(|fold| fold == first).then_some(first)
}

fn root_component_id(members: &BTreeSet<String>) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"dedup-calibration-root-component-v1\0");
    for member in members {
        hasher.update((member.len() as u64).to_be_bytes());
        hasher.update(member.as_bytes());
    }
    format!("dcg_{:x}", hasher.finalize())
}

fn assign_root_component_ids(cases: &mut [DedupCalibrationCase]) {
    let mut graph: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for case in cases.iter() {
        for root in &case.root_keys {
            graph.entry(root.clone()).or_default();
        }
        for left_index in 0..case.root_keys.len() {
            for right_index in (left_index + 1)..case.root_keys.len() {
                let left = case.root_keys[left_index].clone();
                let right = case.root_keys[right_index].clone();
                graph.entry(left.clone()).or_default().insert(right.clone());
                graph.entry(right).or_default().insert(left);
            }
        }
    }

    let mut seen = BTreeSet::new();
    let mut root_to_component = BTreeMap::new();
    for start in graph.keys() {
        if seen.contains(start) {
            continue;
        }
        let mut stack = vec![start.clone()];
        let mut members = BTreeSet::new();
        while let Some(root) = stack.pop() {
            if !seen.insert(root.clone()) {
                continue;
            }
            members.insert(root.clone());
            if let Some(neighbors) = graph.get(&root) {
                stack.extend(neighbors.iter().cloned());
            }
        }
        let component_id = root_component_id(&members);
        for member in members {
            root_to_component.insert(member, component_id.clone());
        }
    }

    for case in cases {
        case.split_group_id = case
            .root_keys
            .first()
            .and_then(|root| root_to_component.get(root))
            .cloned()
            .unwrap_or_else(|| case.family_id.clone());
    }
}

#[derive(Debug, Clone)]
pub(crate) struct DedupSealedCorpus {
    cases: Vec<DedupCalibrationCase>,
    generation: u64,
    cutoff: i64,
    corpus_fingerprint: String,
}

#[derive(Debug, Clone)]
pub(crate) struct DedupCalibrationEvaluation {
    pub policy: DedupCalibrationPolicy,
    pub seal: DedupCalibrationSeal,
}

impl DedupSealedCorpus {
    /// Freeze the complete evidence set before the selector can inspect a
    /// train/holdout assignment. The returned corpus exposes no mutation API.
    pub(crate) fn seal(
        mut cases: Vec<DedupCalibrationCase>,
        generation: u64,
        cutoff: i64,
    ) -> Result<Self, String> {
        if generation == 0 || cutoff <= 0 || cases.is_empty() {
            return Err("sealed corpus requires non-zero generation/cutoff and evidence".into());
        }
        if cases.iter().any(|case| !case_is_valid(case)) {
            return Err("sealed corpus contains invalid or unverified evidence".into());
        }
        assign_root_component_ids(&mut cases);
        if cases.iter().any(|case| !case_is_valid(case)) {
            return Err("sealed corpus contains invalid root-component evidence".into());
        }
        let mut case_ids: Vec<&str> = cases.iter().map(|case| case.case_id.as_str()).collect();
        case_ids.sort_unstable();
        if case_ids.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err("sealed corpus contains duplicate case ids".into());
        }
        let corpus_fingerprint = fingerprint_cases(cases.iter());
        Ok(Self {
            cases,
            generation,
            cutoff,
            corpus_fingerprint,
        })
    }
}

const STRUCTURAL_PROBE_SET_VERSION: &str = "c2-structural-v1";
const STRUCTURAL_CHALLENGE_TEMPLATES: [(&str, &str, &str); 16] = [
    (
        "status",
        "service status is enabled",
        "service status is disabled",
    ),
    (
        "access",
        "account access is granted",
        "account access is denied",
    ),
    (
        "feature",
        "feature flag is present",
        "feature flag is absent",
    ),
    (
        "deploy",
        "deployment result is success",
        "deployment result is failure",
    ),
    (
        "invoice",
        "invoice state is paid",
        "invoice state is unpaid",
    ),
    (
        "version",
        "release version is one",
        "release version is two",
    ),
    (
        "port",
        "service port is four four three",
        "service port is two two",
    ),
    ("owner", "record owner is alice", "record owner is bob"),
    (
        "mode",
        "filesystem mode is read only",
        "filesystem mode is read write",
    ),
    (
        "policy",
        "network policy says allow",
        "network policy says deny",
    ),
    (
        "environment",
        "runtime environment is production",
        "runtime environment is staging",
    ),
    (
        "branch",
        "repository branch is main",
        "repository branch is development",
    ),
    (
        "backup",
        "backup state is complete",
        "backup state is missing",
    ),
    (
        "database",
        "database role is primary",
        "database role is replica",
    ),
    (
        "health",
        "node health is healthy",
        "node health is unhealthy",
    ),
    (
        "retention",
        "retention policy keeps data",
        "retention policy deletes data",
    ),
];
const STRUCTURAL_NONCE_TEMPLATES: [(&str, &str, &str); 4] = [
    (
        "record-id",
        "calibration record nonce alpha status stable",
        "calibration record nonce beta status stable",
    ),
    (
        "request-id",
        "request identifier red returned success",
        "request identifier blue returned success",
    ),
    (
        "tenant-id",
        "tenant marker north uses policy default",
        "tenant marker south uses policy default",
    ),
    (
        "artifact-id",
        "artifact key first has checksum valid",
        "artifact key second has checksum valid",
    ),
];

/// Resolve every revision of a canonical family to one fixed root for the
/// current sealed generation. `canonical_id_for` first collapses old decision
/// rows to the live tip; the earliest `(created_at, id)` family member then
/// remains the deterministic independence key across later tip revisions.
fn fixed_canonical_family_root(store: &SqliteStore, memory_id: &str) -> Option<String> {
    let exists = store
        .conn()
        .query_row(
            "SELECT 1 FROM memories WHERE id = ?1",
            rusqlite::params![memory_id],
            |_| Ok(()),
        )
        .is_ok();
    if !exists {
        return None;
    }
    let live_tip = store
        .canonical_id_for(memory_id)
        .ok()
        .filter(|tip| !tip.is_empty())?;
    store
        .conn()
        .query_row(
            "SELECT cs.memory_id
               FROM memory_canonical_state cs
               JOIN memories m ON m.id = cs.memory_id
              WHERE COALESCE(cs.canonical_id, cs.memory_id) = ?1
              ORDER BY m.created_at ASC, cs.memory_id ASC
              LIMIT 1",
            rusqlite::params![live_tip],
            |row| row.get::<_, String>(0),
        )
        .ok()
}

/// Build a frozen calibration corpus from explicit operator-label snapshots
/// plus a small deterministic challenge set. Auto/LLM/batch decisions are
/// excluded. The payload must contain immutable pair bytes, so later memory
/// rewrites cannot silently change a label's evidence.
pub(crate) fn build_dedup_calibration_corpus(
    store: &SqliteStore,
    generation: u64,
    cutoff: i64,
) -> std::result::Result<DedupSealedCorpus, String> {
    let mut cases = Vec::new();
    let mut stmt = store
        .conn()
        .prepare(
            "SELECT id, canonical_id, relation, payload, created_at
               FROM dedup_decisions
              WHERE operator = 'operator_label'
              ORDER BY id ASC",
        )
        .map_err(|error| error.to_string())?;
    let rows = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, Option<String>>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, Option<String>>(3)?,
                row.get::<_, String>(4)?,
            ))
        })
        .map_err(|error| error.to_string())?;
    for row in rows {
        let (decision_id, canonical_id, relation, payload, created_at) =
            row.map_err(|error| error.to_string())?;
        let created_at = chrono::DateTime::parse_from_rfc3339(&created_at)
            .map_err(|error| error.to_string())?
            .timestamp();
        if created_at > cutoff {
            continue;
        }
        let Some(payload) =
            payload.and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
        else {
            continue;
        };
        let Some(label) = payload.get("calibration_label") else {
            continue;
        };
        if label.get("version").and_then(|value| value.as_u64()) != Some(1)
            || label
                .get("operator_confirmed")
                .and_then(|value| value.as_bool())
                != Some(true)
        {
            continue;
        }
        let Some(left_id) = label.get("left_memory_id").and_then(|value| value.as_str()) else {
            continue;
        };
        let Some(right_id) = label
            .get("right_memory_id")
            .and_then(|value| value.as_str())
        else {
            continue;
        };
        let Some(left) = label.get("left_content").and_then(|value| value.as_str()) else {
            continue;
        };
        let Some(right) = label.get("right_content").and_then(|value| value.as_str()) else {
            continue;
        };
        let case = match relation.as_str() {
            "duplicate" => {
                // Positive ESS is keyed only by an immutable, explicitly
                // recorded canonical root. Falling back to the pair ids would
                // let repeated labels for one family masquerade as independent
                // samples and would accept positives with no family proof.
                if left_id == right_id {
                    continue;
                }
                let Some(root) = canonical_id.as_deref().filter(|root| !root.is_empty()) else {
                    continue;
                };
                let (Some(root), Some(left_root), Some(right_root)) = (
                    fixed_canonical_family_root(store, root),
                    fixed_canonical_family_root(store, left_id),
                    fixed_canonical_family_root(store, right_id),
                ) else {
                    continue;
                };
                if root != left_root || root != right_root {
                    continue;
                }
                DedupCalibrationCase::canonical_family_positive(&root, &decision_id, left, right)
            }
            "distinct" => {
                let (Some(left_root), Some(right_root)) = (
                    fixed_canonical_family_root(store, left_id),
                    fixed_canonical_family_root(store, right_id),
                ) else {
                    continue;
                };
                DedupCalibrationCase::operator_distinct(
                    &decision_id,
                    &left_root,
                    &right_root,
                    left,
                    right,
                )
            }
            _ => None,
        };
        if let Some(case) = case {
            cases.push(case);
        }
    }
    drop(stmt);

    for (template_id, left, right) in STRUCTURAL_CHALLENGE_TEMPLATES {
        if let Some(case) = DedupCalibrationCase::structural_contradiction(
            STRUCTURAL_PROBE_SET_VERSION,
            template_id,
            left,
            right,
        ) {
            cases.push(case);
        }
        let exact_id = format!("{STRUCTURAL_PROBE_SET_VERSION}:{template_id}:exact");
        if let Some(case) = DedupCalibrationCase::exact_content_positive(&exact_id, left) {
            cases.push(case);
        }
    }
    for (template_id, left, right) in STRUCTURAL_NONCE_TEMPLATES {
        if let Some(case) = DedupCalibrationCase::structural_nonce(
            STRUCTURAL_PROBE_SET_VERSION,
            template_id,
            left,
            right,
        ) {
            cases.push(case);
        }
    }

    DedupSealedCorpus::seal(cases, generation, cutoff)
}

/// Refresh the separate sealed calibration bundle. Corrupt/future/storage
/// states are preserved for doctor repair. Underpowered `NoData` evaluation is
/// completely read-only: it writes neither policy, seal, nor revision. Only a
/// powered, revealed `Ship`/`Bail` terminal is persisted, freezing that holdout
/// against optional stopping. Expired terminals require an explicit reset.
pub(crate) fn refresh_dedup_calibration_policy(
    store: &SqliteStore,
    configured_static_threshold: f32,
    shadow_threshold: f32,
    now: i64,
    validity_secs: i64,
) -> std::result::Result<DedupCalibrationPolicy, String> {
    let loaded = load_dedup_calibration(store.conn(), now);
    let expected_revision = match loaded.status {
        DedupCalibrationLoadStatus::Missing => {
            let seal = load_dedup_calibration_seal(store.conn(), now);
            if seal.status != DedupCalibrationLoadStatus::Missing {
                return Err(format!(
                    "dedup calibration policy is missing while seal state is {:?}: {}; atomically reset both calibration metadata rows (dedup_calibration_policy and dedup_calibration_seal) before retrying",
                    seal.status,
                    seal.error.unwrap_or_else(|| "orphaned seal row".to_string())
                ));
            }
            0
        }
        DedupCalibrationLoadStatus::Loaded => {
            let verified =
                load_dedup_calibration_for_runtime(store.conn(), now, configured_static_threshold);
            if verified.status != DedupCalibrationLoadStatus::Loaded || !verified.context_verified()
            {
                return Err(format!(
                    "dedup calibration refresh preserved unverified bundle {:?}: {}; atomically reset both calibration metadata rows (dedup_calibration_policy and dedup_calibration_seal) before retrying",
                    verified.status,
                    verified.error.unwrap_or_else(|| "no detail".to_string())
                ));
            }
            if matches!(
                verified.policy.status,
                DedupCalibrationStatus::Ship | DedupCalibrationStatus::Bail
            ) {
                return Ok(verified.policy);
            }
            verified.policy.revision
        }
        DedupCalibrationLoadStatus::Stale => {
            if matches!(
                loaded.policy.status,
                DedupCalibrationStatus::Ship | DedupCalibrationStatus::Bail
            ) {
                return Err(
                    "terminal dedup calibration generation is stale; explicit operator reset is required"
                        .to_string(),
                );
            }
            loaded.policy.revision
        }
        DedupCalibrationLoadStatus::Corrupt
        | DedupCalibrationLoadStatus::UnsupportedSchema
        | DedupCalibrationLoadStatus::FingerprintMismatch
        | DedupCalibrationLoadStatus::StorageError => {
            return Err(format!(
                "dedup calibration refresh preserved unhealthy state {:?}: {}; atomically reset both calibration metadata rows (dedup_calibration_policy and dedup_calibration_seal) before retrying",
                loaded.status,
                loaded.error.unwrap_or_else(|| "no detail".to_string())
            ));
        }
    };
    let generation = loaded.policy.sealed_generation.saturating_add(1).max(1);
    let sealed = build_dedup_calibration_corpus(store, generation, now)?;
    let mut evaluation = calibrate_dedup_policy(
        configured_static_threshold,
        shadow_threshold,
        &sealed,
        now,
        validity_secs,
    );
    if !evaluation.policy.holdout_revealed
        || evaluation.policy.status == DedupCalibrationStatus::NoData
    {
        // A persisted underpowered row becomes a stateful optional-stopping
        // oracle (and revision churn on every adaptive tick). Return the shadow
        // observation to the caller, but leave metadata byte-for-byte untouched.
        return Ok(evaluation.policy);
    }
    let revision = expected_revision.saturating_add(1);
    evaluation.policy.revision = revision;
    evaluation.seal.revision = revision;
    evaluation.seal.policy_digest = canonical_policy_digest(&evaluation.policy)?;
    match save_dedup_calibration_bundle_cas(
        store.conn(),
        &evaluation.policy,
        &evaluation.seal,
        expected_revision,
    ) {
        Ok(true) => Ok(evaluation.policy),
        Ok(false) => Err("dedup calibration bundle CAS conflict".to_string()),
        Err(error) => Err(error.to_string()),
    }
}

fn case_is_valid(case: &DedupCalibrationCase) -> bool {
    let roots_valid = match case.source {
        DedupCalibrationCaseSource::CanonicalFamily => case.root_keys.len() == 1,
        DedupCalibrationCaseSource::OperatorDistinct => case.root_keys.len() == 2,
        DedupCalibrationCaseSource::ExactContentHash
        | DedupCalibrationCaseSource::StructuralContradiction
        | DedupCalibrationCaseSource::StructuralNonce => case.root_keys.is_empty(),
    } && case.root_keys.windows(2).all(|pair| pair[0] < pair[1]);
    !case.case_id.is_empty()
        && !case.family_id.is_empty()
        && !case.split_group_id.is_empty()
        && !case.evidence_fingerprint.is_empty()
        && case.similarity.is_finite()
        && (0.0..=1.0).contains(&case.similarity)
        && case.source.is_valid_for(case.label)
        && case.slice == case.source.expected_slice()
        && roots_valid
        && dedup_calibration_case_fold(case).is_some()
}

fn fingerprint_cases<'a>(cases: impl IntoIterator<Item = &'a DedupCalibrationCase>) -> String {
    let mut rows: Vec<&DedupCalibrationCase> = cases.into_iter().collect();
    rows.sort_by(|left, right| {
        left.family_id
            .cmp(&right.family_id)
            .then(left.case_id.cmp(&right.case_id))
    });
    let mut hasher = Sha256::new();
    for case in rows {
        hasher.update(case.split_group_id.as_bytes());
        hasher.update([0]);
        hasher.update(case.family_id.as_bytes());
        hasher.update([0]);
        for root in &case.root_keys {
            hasher.update((root.len() as u64).to_be_bytes());
            hasher.update(root.as_bytes());
        }
        hasher.update([0]);
        hasher.update(case.case_id.as_bytes());
        hasher.update([0]);
        hasher.update(case.similarity.to_bits().to_be_bytes());
        hasher.update([match case.label {
            DedupCalibrationLabel::Duplicate => 1,
            DedupCalibrationLabel::Distinct => 0,
        }]);
        hasher.update([case.source as u8]);
        hasher.update(case.slice.as_str().as_bytes());
        hasher.update([0]);
        hasher.update(case.evidence_fingerprint.as_bytes());
        hasher.update([0xff]);
    }
    format!("{:x}", hasher.finalize())
}

/// Collapse repeated/root-connected rows to one deterministic observation per
/// component and label. A connected component may legitimately contribute one
/// canonical positive and one operator negative, but both share one fold. Many
/// A-B/A-C negatives can never inflate ESS beyond one component observation.
fn independent_cases(cases: &[DedupCalibrationCase]) -> (Vec<&DedupCalibrationCase>, usize) {
    let mut by_component_and_label: BTreeMap<
        (&str, DedupCalibrationLabel),
        Vec<&DedupCalibrationCase>,
    > = BTreeMap::new();
    let mut rejected = 0usize;
    for case in cases {
        if case_is_valid(case) {
            by_component_and_label
                .entry((&case.split_group_id, case.label))
                .or_default()
                .push(case);
        } else {
            rejected += 1;
        }
    }
    let mut selected = Vec::with_capacity(by_component_and_label.len());
    for mut component_rows in by_component_and_label.into_values() {
        component_rows.sort_by(|left, right| left.case_id.cmp(&right.case_id));
        let first = component_rows[0];
        // Repeated rows do not increase ESS. Keep the hardest observation for
        // the claimed label so row ordering cannot cherry-pick an optimistic
        // representative: highest similarity for Distinct, lowest for
        // Duplicate; case_id remains the deterministic tie-break.
        let representative = component_rows
            .into_iter()
            .reduce(|left, right| {
                let right_is_harder = match first.label {
                    DedupCalibrationLabel::Distinct => right.similarity > left.similarity,
                    DedupCalibrationLabel::Duplicate => right.similarity < left.similarity,
                };
                if right_is_harder {
                    right
                } else {
                    left
                }
            })
            .expect("family_rows is non-empty");
        selected.push(representative);
    }
    (selected, rejected)
}

fn calibration_provenance(cases: &[&DedupCalibrationCase]) -> DedupCalibrationProvenance {
    let structural = cases.iter().any(|case| {
        matches!(
            case.source,
            DedupCalibrationCaseSource::StructuralContradiction
                | DedupCalibrationCaseSource::StructuralNonce
        )
    });
    let operator = cases
        .iter()
        .any(|case| case.source == DedupCalibrationCaseSource::OperatorDistinct);
    match (structural, operator) {
        (true, true) => DedupCalibrationProvenance::Mixed,
        (true, false) => DedupCalibrationProvenance::StructuralAnchors,
        (false, true) => DedupCalibrationProvenance::OperatorLabels,
        (false, false) => DedupCalibrationProvenance::DiscoveryOnly,
    }
}

fn false_positive_status(
    failures: usize,
    trials: usize,
    upper: Option<f64>,
) -> DedupCalibrationStatus {
    if trials == 0 || (failures == 0 && trials < ZERO_FP_REQUIRED_NEGATIVES) {
        return DedupCalibrationStatus::NoData;
    }
    match upper {
        Some(value) if value <= FALSE_POSITIVE_BUDGET => DedupCalibrationStatus::Ship,
        Some(_) => DedupCalibrationStatus::Bail,
        None => DedupCalibrationStatus::NoData,
    }
}

/// Select a raise-only candidate from train folds and evaluate it exactly once
/// on the sealed family-disjoint holdout. The input has already been frozen by
/// proof-bearing constructors; this function never treats an executed merge or
/// adaptive fire rate as a correctness label.
pub(crate) fn calibrate_dedup_policy(
    configured_static_threshold: f32,
    shadow_threshold: f32,
    corpus: &DedupSealedCorpus,
    now: i64,
    validity_secs: i64,
) -> DedupCalibrationEvaluation {
    let static_threshold = if configured_static_threshold.is_finite()
        && (0.0..=1.0).contains(&configured_static_threshold)
    {
        configured_static_threshold
    } else {
        1.0
    };
    let shadow_threshold =
        if shadow_threshold.is_finite() && (0.0..=1.0).contains(&shadow_threshold) {
            shadow_threshold
        } else {
            static_threshold
        };
    let cases = &corpus.cases;
    let (independent, rejected_case_count) = independent_cases(cases);
    let train: Vec<&DedupCalibrationCase> = independent
        .iter()
        .copied()
        .filter(|case| dedup_calibration_case_fold(case) != Some(DEDUP_CALIBRATION_HOLDOUT_FOLD))
        .collect();
    let holdout: Vec<&DedupCalibrationCase> = independent
        .iter()
        .copied()
        .filter(|case| dedup_calibration_case_fold(case) == Some(DEDUP_CALIBRATION_HOLDOUT_FOLD))
        .collect();

    let train_positive_count = train
        .iter()
        .filter(|case| case.label == DedupCalibrationLabel::Duplicate)
        .count();
    let train_negative_count = train.len().saturating_sub(train_positive_count);
    let candidate_threshold = train
        .iter()
        .filter(|case| case.label == DedupCalibrationLabel::Distinct)
        .map(|case| case.similarity)
        .fold(static_threshold, f32::max)
        .max(static_threshold);

    let holdout_positives: Vec<&DedupCalibrationCase> = holdout
        .iter()
        .copied()
        .filter(|case| {
            case.label == DedupCalibrationLabel::Duplicate
                && case.source == DedupCalibrationCaseSource::CanonicalFamily
        })
        .collect();
    let holdout_negatives: Vec<&DedupCalibrationCase> = holdout
        .iter()
        .copied()
        .filter(|case| {
            case.label == DedupCalibrationLabel::Distinct
                && case.source == DedupCalibrationCaseSource::OperatorDistinct
        })
        .collect();
    let sealed_exact_positive_count = holdout
        .iter()
        .filter(|case| case.source == DedupCalibrationCaseSource::ExactContentHash)
        .count();
    let sealed_structural_negative_count = holdout
        .iter()
        .filter(|case| {
            matches!(
                case.source,
                DedupCalibrationCaseSource::StructuralContradiction
                    | DedupCalibrationCaseSource::StructuralNonce
            )
        })
        .count();
    let false_positive_count = holdout_negatives
        .iter()
        .filter(|case| case.similarity > candidate_threshold)
        .count();
    let true_positive_count = holdout_positives
        .iter()
        .filter(|case| case.similarity > candidate_threshold)
        .count();
    let false_negative_count = holdout_positives.len().saturating_sub(true_positive_count);
    let true_negative_count = holdout_negatives.len().saturating_sub(false_positive_count);
    let false_positive_upper_95 = match (
        u32::try_from(false_positive_count),
        u32::try_from(holdout_negatives.len()),
    ) {
        (Ok(failures), Ok(trials)) => {
            one_sided_binomial_upper_bound(failures, trials, FALSE_POSITIVE_ALPHA)
        }
        _ => None,
    };
    let safety_status = false_positive_status(
        false_positive_count,
        holdout_negatives.len(),
        false_positive_upper_95,
    );

    let paired: Vec<PairedOutcome> = holdout_positives
        .iter()
        .map(|case| PairedOutcome {
            case_id: case.case_id.clone(),
            baseline_hit: case.similarity > static_threshold,
            treatment_hit: case.similarity > candidate_threshold,
            baseline_length: 0,
            treatment_length: 0,
            treatment_summary: None,
        })
        .collect();
    let utility_result = mcnemar(&paired);
    let miss_rate_upper_95 = match u32::try_from(paired.len()) {
        Ok(trials) => {
            let misses = utility_result.b;
            one_sided_binomial_upper_bound(misses, trials, FALSE_POSITIVE_ALPHA)
        }
        Err(_) => None,
    };
    let utility_status = if paired.len() < MIN_CALIBRATION_POSITIVE_HOLDOUT {
        DedupCalibrationStatus::NoData
    } else if miss_rate_upper_95.is_some_and(|upper| upper <= FALSE_POSITIVE_BUDGET)
        && utility_result.ci_lower >= -crate::eval::gates::DEFAULT_NOISE_FLOOR
    {
        DedupCalibrationStatus::Ship
    } else {
        DedupCalibrationStatus::Bail
    };
    let utility = DedupUtilityEvidence {
        n: paired.len(),
        baseline_only_hits: utility_result.b as usize,
        candidate_only_hits: utility_result.c as usize,
        ci_lower: utility_result.ci_lower,
        miss_rate_upper_95,
        status: utility_status,
    };

    let required_slices: Vec<DedupCalibrationSlice> = REQUIRED_DEDUP_CALIBRATION_SLICE_KINDS
        .iter()
        .map(|kind| {
            let name = kind.as_str();
            let rows: Vec<&DedupCalibrationCase> = holdout
                .iter()
                .copied()
                .filter(|case| case.slice == *kind)
                .collect();
            let positive_count = rows
                .iter()
                .filter(|case| case.label == DedupCalibrationLabel::Duplicate)
                .count();
            let negative_count = rows.len().saturating_sub(positive_count);
            let has_false_positive = rows.iter().any(|case| {
                case.label == DedupCalibrationLabel::Distinct
                    && case.similarity > candidate_threshold
            });
            let has_positive_regression = rows.iter().any(|case| {
                case.label == DedupCalibrationLabel::Duplicate
                    && case.similarity > static_threshold
                    && case.similarity <= candidate_threshold
            });
            let status = if !kind.has_required_counts(positive_count, negative_count) {
                DedupCalibrationStatus::NoData
            } else if has_false_positive || has_positive_regression {
                DedupCalibrationStatus::Bail
            } else {
                DedupCalibrationStatus::Ship
            };
            DedupCalibrationSlice {
                name: name.to_string(),
                positive_count,
                negative_count,
                status,
            }
        })
        .collect();
    let slices_status = if required_slices.is_empty()
        || required_slices
            .iter()
            .any(|slice| slice.status == DedupCalibrationStatus::NoData)
    {
        DedupCalibrationStatus::NoData
    } else if required_slices
        .iter()
        .any(|slice| slice.status == DedupCalibrationStatus::Bail)
    {
        DedupCalibrationStatus::Bail
    } else {
        DedupCalibrationStatus::Ship
    };
    let train_powered = train_positive_count >= MIN_CALIBRATION_TRAIN_PER_CLASS
        && train_negative_count >= MIN_CALIBRATION_TRAIN_PER_CLASS;
    let slice_data_complete = REQUIRED_DEDUP_CALIBRATION_SLICE_KINDS
        .iter()
        .zip(required_slices.iter())
        .all(|(kind, slice)| kind.has_required_counts(slice.positive_count, slice.negative_count));
    let evidence_complete = train_powered
        && holdout_positives.len() >= MIN_CALIBRATION_POSITIVE_HOLDOUT
        && holdout_negatives.len() >= ZERO_FP_REQUIRED_NEGATIVES
        && slice_data_complete;
    let holdout_revealed = evidence_complete && rejected_case_count == 0;
    let status = if rejected_case_count > 0 {
        DedupCalibrationStatus::Bail
    } else if !evidence_complete {
        DedupCalibrationStatus::NoData
    } else if safety_status == DedupCalibrationStatus::Bail
        || utility_status == DedupCalibrationStatus::Bail
        || slices_status == DedupCalibrationStatus::Bail
    {
        DedupCalibrationStatus::Bail
    } else if safety_status == DedupCalibrationStatus::NoData
        || utility_status == DedupCalibrationStatus::NoData
        || slices_status == DedupCalibrationStatus::NoData
    {
        DedupCalibrationStatus::NoData
    } else {
        DedupCalibrationStatus::Ship
    };

    let (reported_false_positives, reported_upper, reported_utility, reported_confusion) =
        if holdout_revealed {
            (
                false_positive_count,
                false_positive_upper_95,
                utility,
                DedupCalibrationConfusion {
                    true_positives: true_positive_count,
                    false_positives: false_positive_count,
                    true_negatives: true_negative_count,
                    false_negatives: false_negative_count,
                },
            )
        } else {
            (
                0,
                None,
                DedupUtilityEvidence {
                    n: holdout_positives.len(),
                    status: DedupCalibrationStatus::NoData,
                    ..DedupUtilityEvidence::default()
                },
                DedupCalibrationConfusion::default(),
            )
        };
    let reported_slices = if holdout_revealed {
        required_slices
    } else {
        required_slices
            .into_iter()
            .map(|mut slice| {
                slice.status = DedupCalibrationStatus::NoData;
                slice
            })
            .collect()
    };

    let train_fingerprint = fingerprint_cases(train.iter().copied());
    let holdout_fingerprint = fingerprint_cases(holdout.iter().copied());
    let corpus_fingerprint = corpus.corpus_fingerprint.clone();
    let valid_until = now.saturating_add(validity_secs.max(1));
    let policy = DedupCalibrationPolicy {
        status,
        configured_static_threshold: static_threshold,
        candidate_threshold,
        shadow_threshold,
        // Shadow evaluation may reach Ship, but its raw lexical score space is
        // not the enriched score used by every production destructive path.
        // Promotion stays disabled until a representative production cohort is
        // evaluated end to end.
        effective_hard_threshold: static_threshold,
        train_positive_count,
        train_negative_count,
        sealed_positive_count: holdout_positives.len(),
        sealed_negative_count: holdout_negatives.len(),
        sealed_exact_positive_count,
        sealed_structural_negative_count,
        false_positive_count: reported_false_positives,
        holdout_revealed,
        false_positive_upper_95: reported_upper,
        utility: reported_utility,
        holdout_confusion: reported_confusion,
        required_slices: reported_slices,
        rejected_case_count,
        sealed_generation: corpus.generation,
        sealed_cutoff: corpus.cutoff,
        train_fingerprint: train_fingerprint.clone(),
        holdout_fingerprint: holdout_fingerprint.clone(),
        corpus_fingerprint: corpus_fingerprint.clone(),
        provenance: calibration_provenance(&independent),
        calibrated_at: now,
        valid_until,
        ..DedupCalibrationPolicy::default()
    };
    let seal = DedupCalibrationSeal {
        schema_version: crate::store::dedup_calibration::DEDUP_CALIBRATION_SCHEMA_VERSION,
        revision: 0,
        generation: corpus.generation,
        cutoff: corpus.cutoff,
        scale: crate::store::dedup_calibration::DedupCalibrationScale::Lexical,
        configured_static_threshold_bits: static_threshold.to_bits(),
        train_fingerprint,
        holdout_fingerprint,
        corpus_fingerprint,
        policy_digest: canonical_policy_digest(&policy)
            .expect("dedup policy canonical serialization must succeed"),
        calibrated_at: now,
        valid_until,
    };
    DedupCalibrationEvaluation { policy, seal }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ReinConfig;
    use crate::store::SqliteStore;
    use crate::types::traits::MemoryStore;

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
    fn classify_one_does_not_predict_duplicate_at_exact_threshold() {
        let fixture = DedupFixture {
            id: "equality-boundary".to_string(),
            text_a: "alpha beta".to_string(),
            text_b: "alpha gamma".to_string(),
            is_duplicate: false,
        };

        assert_eq!(
            similarity(&fixture.text_a, &fixture.text_b),
            DEDUP_THRESHOLD,
            "fixture must exercise exact threshold equality"
        );
        assert!(classify_one(&fixture));
    }

    #[test]
    fn sweep_does_not_count_exact_threshold_as_false_positive() {
        let curve = sweep_thresholds(&[(0.50, false)]);
        let row = curve
            .iter()
            .find(|row| row.threshold == 0.50)
            .expect("0.50 row exists");

        assert_eq!((row.tp, row.fp, row.tn, row.false_neg), (0, 0, 1, 0));
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

    fn calibration_case(
        family_id: String,
        similarity: f32,
        label: DedupCalibrationLabel,
        slice: DedupCalibrationSliceKind,
    ) -> DedupCalibrationCase {
        let source = match label {
            DedupCalibrationLabel::Duplicate => DedupCalibrationCaseSource::CanonicalFamily,
            DedupCalibrationLabel::Distinct => DedupCalibrationCaseSource::OperatorDistinct,
        };
        let root_keys = match label {
            DedupCalibrationLabel::Duplicate => vec![format!("canonical-root-{family_id}")],
            DedupCalibrationLabel::Distinct => vec![
                format!("distinct-left-{family_id}"),
                format!("distinct-right-{family_id}"),
            ],
        };
        let split_group_id = root_component_id(&root_keys.iter().cloned().collect());
        DedupCalibrationCase {
            case_id: format!("case-{family_id}"),
            evidence_fingerprint: format!("evidence-{family_id}"),
            family_id,
            root_keys,
            split_group_id,
            similarity,
            label,
            source,
            slice,
        }
    }

    fn families_in_fold(
        fold: u8,
        count: usize,
        prefix: &str,
        label: DedupCalibrationLabel,
    ) -> Vec<String> {
        let mut out = Vec::with_capacity(count);
        let mut index = 0usize;
        while out.len() < count {
            let family = format!("{prefix}-{index}");
            let probe = calibration_case(
                family.clone(),
                0.0,
                label,
                match label {
                    DedupCalibrationLabel::Duplicate => DedupCalibrationSliceKind::CanonicalFamily,
                    DedupCalibrationLabel::Distinct => DedupCalibrationSliceKind::OperatorDistinct,
                },
            );
            if dedup_calibration_case_fold(&probe) == Some(fold) {
                out.push(family);
            }
            index += 1;
        }
        out
    }

    fn families_in_fold_without_roots(fold: u8, count: usize, prefix: &str) -> Vec<String> {
        let mut out = Vec::with_capacity(count);
        let mut index = 0usize;
        while out.len() < count {
            let family = format!("{prefix}-{index}");
            if dedup_calibration_fold(&family) == fold {
                out.push(family);
            }
            index += 1;
        }
        out
    }

    fn roots_in_fold(fold: u8, count: usize, prefix: &str) -> Vec<String> {
        let mut out = Vec::with_capacity(count);
        let mut index = 0usize;
        while out.len() < count {
            let root = format!("{prefix}-{index}");
            if dedup_calibration_fold(&root) == fold {
                out.push(root);
            }
            index += 1;
        }
        out
    }

    fn calibrate_cases(cases: Vec<DedupCalibrationCase>) -> DedupCalibrationPolicy {
        let sealed = DedupSealedCorpus::seal(cases, 1, 999).unwrap();
        calibrate_dedup_policy(0.70, 0.40, &sealed, 1_000, 86_400).policy
    }

    #[test]
    fn sealed_holdout_never_participates_in_candidate_selection() {
        let train_family =
            families_in_fold(1, 1, "train", DedupCalibrationLabel::Distinct).remove(0);
        let holdout_family =
            families_in_fold(0, 1, "holdout", DedupCalibrationLabel::Distinct).remove(0);
        let cases = vec![
            calibration_case(
                train_family,
                0.80,
                DedupCalibrationLabel::Distinct,
                DedupCalibrationSliceKind::OperatorDistinct,
            ),
            calibration_case(
                holdout_family,
                0.95,
                DedupCalibrationLabel::Distinct,
                DedupCalibrationSliceKind::OperatorDistinct,
            ),
        ];

        let policy = calibrate_cases(cases);

        assert_eq!(policy.candidate_threshold, 0.80);
        assert_eq!(policy.status, DedupCalibrationStatus::NoData);
        assert!(!policy.holdout_revealed);
    }

    #[test]
    fn powered_family_disjoint_holdout_can_ship_shadow_candidate_only() {
        let mut cases = Vec::new();
        for family in families_in_fold(
            1,
            20,
            "canonical-train-positive",
            DedupCalibrationLabel::Duplicate,
        ) {
            cases.push(calibration_case(
                family,
                1.0,
                DedupCalibrationLabel::Duplicate,
                DedupCalibrationSliceKind::CanonicalFamily,
            ));
        }
        for family in families_in_fold(
            2,
            80,
            "operator-train-negative",
            DedupCalibrationLabel::Distinct,
        ) {
            cases.push(calibration_case(
                family,
                0.80,
                DedupCalibrationLabel::Distinct,
                DedupCalibrationSliceKind::OperatorDistinct,
            ));
        }
        for family in families_in_fold(
            0,
            149,
            "canonical-holdout-positive",
            DedupCalibrationLabel::Duplicate,
        ) {
            cases.push(calibration_case(
                family,
                1.0,
                DedupCalibrationLabel::Duplicate,
                DedupCalibrationSliceKind::CanonicalFamily,
            ));
        }
        for family in families_in_fold(
            0,
            149,
            "operator-holdout-negative",
            DedupCalibrationLabel::Distinct,
        ) {
            cases.push(calibration_case(
                family,
                0.80,
                DedupCalibrationLabel::Distinct,
                DedupCalibrationSliceKind::OperatorDistinct,
            ));
        }
        let mut structural_positive = calibration_case(
            families_in_fold_without_roots(0, 1, "structural-positive").remove(0),
            1.0,
            DedupCalibrationLabel::Duplicate,
            DedupCalibrationSliceKind::StructuralChallenge,
        );
        structural_positive.source = DedupCalibrationCaseSource::ExactContentHash;
        structural_positive.root_keys.clear();
        structural_positive.split_group_id = structural_positive.family_id.clone();
        cases.push(structural_positive);
        let mut structural_negative = calibration_case(
            families_in_fold_without_roots(0, 1, "structural-negative").remove(0),
            0.80,
            DedupCalibrationLabel::Distinct,
            DedupCalibrationSliceKind::StructuralChallenge,
        );
        structural_negative.source = DedupCalibrationCaseSource::StructuralContradiction;
        structural_negative.root_keys.clear();
        structural_negative.split_group_id = structural_negative.family_id.clone();
        cases.push(structural_negative);

        let policy = calibrate_cases(cases);

        assert_eq!(policy.status, DedupCalibrationStatus::Ship);
        assert_eq!(policy.effective_hard_threshold, 0.70);
        assert_eq!(policy.sealed_negative_count, 149);
        assert_eq!(policy.sealed_positive_count, 149);
        assert_eq!(policy.sealed_exact_positive_count, 1);
        assert_eq!(policy.sealed_structural_negative_count, 1);
        assert_eq!(policy.false_positive_count, 0);
        assert!(policy.false_positive_upper_95.unwrap() <= FALSE_POSITIVE_BUDGET);
        assert_eq!(policy.utility.status, DedupCalibrationStatus::Ship);
        assert!(!policy.train_fingerprint.is_empty());
        assert_ne!(policy.train_fingerprint, policy.holdout_fingerprint);
    }

    #[test]
    fn duplicate_family_rows_count_as_one_independent_holdout_case() {
        let family =
            families_in_fold(0, 1, "same-family", DedupCalibrationLabel::Distinct).remove(0);
        let cases = vec![
            calibration_case(
                family.clone(),
                0.60,
                DedupCalibrationLabel::Distinct,
                DedupCalibrationSliceKind::OperatorDistinct,
            ),
            DedupCalibrationCase {
                case_id: "later-row-same-family".to_string(),
                ..calibration_case(
                    family,
                    0.95,
                    DedupCalibrationLabel::Distinct,
                    DedupCalibrationSliceKind::OperatorDistinct,
                )
            },
        ];

        let policy = calibrate_cases(cases);
        assert_eq!(policy.sealed_negative_count, 1);
        assert_eq!(policy.false_positive_count, 0);
        assert_eq!(policy.status, DedupCalibrationStatus::NoData);
        assert!(!policy.holdout_revealed);
    }

    #[test]
    fn root_connected_pairs_share_one_ess_component_and_fold() {
        let shared_root = "shared-root";
        let shared_fold = dedup_calibration_fold(shared_root);
        let other_roots = roots_in_fold(shared_fold, 149, "other-root");
        let mut cases = vec![DedupCalibrationCase::canonical_family_positive(
            shared_root,
            "positive-shared-root",
            "shared canonical old snapshot",
            "shared canonical live snapshot",
        )
        .unwrap()];
        for (index, other_root) in other_roots.iter().enumerate() {
            cases.push(
                DedupCalibrationCase::operator_distinct(
                    &format!("distinct-{index}"),
                    shared_root,
                    other_root,
                    &format!("shared root snapshot {index}"),
                    &format!("other root snapshot {index}"),
                )
                .unwrap(),
            );
        }

        let sealed = DedupSealedCorpus::seal(cases, 1, 999).unwrap();
        let (independent, rejected) = independent_cases(&sealed.cases);
        assert_eq!(rejected, 0);
        assert_eq!(
            independent
                .iter()
                .filter(|case| case.source == DedupCalibrationCaseSource::OperatorDistinct)
                .count(),
            1,
            "149 negative pairs sharing one canonical root are one ESS component"
        );
        assert_eq!(
            independent
                .iter()
                .filter_map(|case| dedup_calibration_case_fold(case))
                .collect::<std::collections::HashSet<_>>()
                .len(),
            1,
            "positive and negative evidence sharing a root must not cross folds"
        );
    }

    #[test]
    fn adding_a_connected_pair_cannot_move_existing_evidence_between_folds() {
        let shared_root = "stable-shared-root";
        let stable_fold = dedup_calibration_fold(shared_root);
        let same_fold_roots = roots_in_fold(stable_fold, 2, "stable-connected-root");
        let canonical = DedupCalibrationCase::canonical_family_positive(
            shared_root,
            "stable-positive",
            "stable canonical old snapshot",
            "stable canonical live snapshot",
        )
        .unwrap();
        let first_edge = DedupCalibrationCase::operator_distinct(
            "stable-edge-one",
            shared_root,
            &same_fold_roots[0],
            "stable shared snapshot one",
            "stable peer snapshot one",
        )
        .unwrap();
        let initial =
            DedupSealedCorpus::seal(vec![canonical.clone(), first_edge.clone()], 1, 999).unwrap();
        let initial_case = initial
            .cases
            .iter()
            .find(|case| case.case_id == canonical.case_id)
            .unwrap();
        let initial_component = initial_case.split_group_id.clone();
        let initial_fold = dedup_calibration_case_fold(initial_case);

        let second_edge = DedupCalibrationCase::operator_distinct(
            "stable-edge-two",
            shared_root,
            &same_fold_roots[1],
            "stable shared snapshot two",
            "stable peer snapshot two",
        )
        .unwrap();
        let expanded =
            DedupSealedCorpus::seal(vec![canonical, first_edge, second_edge], 2, 1_000).unwrap();
        let expanded_case = expanded
            .cases
            .iter()
            .find(|case| case.case_id == initial_case.case_id)
            .unwrap();

        assert_ne!(
            expanded_case.split_group_id, initial_component,
            "the regression must actually expand the root component"
        );
        assert_eq!(initial_fold, Some(stable_fold));
        assert_eq!(
            dedup_calibration_case_fold(expanded_case),
            initial_fold,
            "component growth must not change a previously sealed case's fold"
        );
    }

    #[test]
    fn operator_distinct_quarantines_cross_fold_root_pairs() {
        let left_root = "cross-fold-left-root";
        let left_fold = dedup_calibration_fold(left_root);
        let right_fold = (left_fold + 1) % DEDUP_CALIBRATION_FOLD_COUNT;
        let right_root = roots_in_fold(right_fold, 1, "cross-fold-right-root").remove(0);

        assert!(DedupCalibrationCase::operator_distinct(
            "cross-fold-decision",
            left_root,
            &right_root,
            "left snapshot",
            "right snapshot",
        )
        .is_none());
    }

    #[test]
    fn proof_bearing_case_constructors_derive_scores_and_family_identity() {
        let canonical = DedupCalibrationCase::canonical_family_positive(
            "root-1",
            "decision-positive-1",
            "alpha beta gamma",
            "alpha beta delta",
        )
        .unwrap();
        assert_eq!(
            canonical.source,
            DedupCalibrationCaseSource::CanonicalFamily
        );
        assert_eq!(
            canonical.similarity,
            similarity("alpha beta gamma", "alpha beta delta")
        );
        assert!(!canonical.evidence_fingerprint.is_empty());
        assert!(DedupCalibrationCase::canonical_family_positive(
            "root-1",
            "decision-identical-positive",
            "identical immutable snapshot",
            "identical immutable snapshot",
        )
        .is_none());

        let distinct_roots = roots_in_fold(2, 2, "constructor-distinct-root");
        assert!(DedupCalibrationCase::operator_distinct(
            "decision-1",
            &distinct_roots[0],
            &distinct_roots[1],
            "identical",
            "identical",
        )
        .is_none());

        let structural_a = DedupCalibrationCase::structural_contradiction(
            "probe-v1",
            "status-toggle",
            "record status enabled",
            "record status disabled",
        )
        .unwrap();
        let structural_b = DedupCalibrationCase::structural_contradiction(
            "probe-v1",
            "status-toggle",
            "service status healthy",
            "service status failed",
        )
        .unwrap();
        assert_eq!(
            structural_a.family_id, structural_b.family_id,
            "one structural template is one challenge family, not N pseudo-independent nonces"
        );
    }

    #[test]
    fn exact_content_positives_do_not_count_as_powered_canonical_utility() {
        let mut cases = Vec::new();
        for family in families_in_fold_without_roots(0, 149, "exact-only") {
            let mut case = calibration_case(
                family,
                1.0,
                DedupCalibrationLabel::Duplicate,
                DedupCalibrationSliceKind::StructuralChallenge,
            );
            case.source = DedupCalibrationCaseSource::ExactContentHash;
            case.root_keys.clear();
            case.split_group_id = case.family_id.clone();
            cases.push(case);
        }
        for family in families_in_fold(
            0,
            149,
            "operator-negatives",
            DedupCalibrationLabel::Distinct,
        ) {
            cases.push(calibration_case(
                family,
                0.80,
                DedupCalibrationLabel::Distinct,
                DedupCalibrationSliceKind::OperatorDistinct,
            ));
        }

        let policy = calibrate_cases(cases);
        assert_eq!(policy.sealed_exact_positive_count, 149);
        assert_eq!(policy.sealed_positive_count, 0);
        assert_eq!(policy.utility.status, DedupCalibrationStatus::NoData);
        assert_ne!(policy.status, DedupCalibrationStatus::Ship);
    }

    #[test]
    fn zero_of_148_canonical_misses_is_still_underpowered_at_two_percent() {
        let mut cases = Vec::new();
        for family in families_in_fold(
            0,
            148,
            "canonical-underpowered",
            DedupCalibrationLabel::Duplicate,
        ) {
            cases.push(calibration_case(
                family,
                1.0,
                DedupCalibrationLabel::Duplicate,
                DedupCalibrationSliceKind::CanonicalFamily,
            ));
        }
        let policy = calibrate_cases(cases);
        assert_eq!(policy.sealed_positive_count, 148);
        assert_eq!(policy.utility.baseline_only_hits, 0);
        assert_eq!(
            policy.utility.miss_rate_upper_95, None,
            "underpowered holdout outcomes stay masked until the precommitted reveal"
        );
        assert_eq!(policy.utility.status, DedupCalibrationStatus::NoData);
    }

    #[test]
    fn sealed_corpus_rejects_duplicate_case_ids_before_split() {
        let case = DedupCalibrationCase::exact_content_positive("record-1", "same bytes").unwrap();
        assert!(DedupSealedCorpus::seal(vec![case.clone(), case], 1, 999).is_err());
    }

    #[test]
    fn structural_challenges_alone_remain_no_data_for_production_promotion() {
        let store = SqliteStore::in_memory().unwrap();
        let sealed = build_dedup_calibration_corpus(&store, 1, 999).unwrap();
        let policy = calibrate_dedup_policy(0.70, 0.40, &sealed, 1_000, 86_400).policy;

        assert_eq!(policy.sealed_negative_count, 0);
        assert_eq!(policy.sealed_positive_count, 0);
        assert!(policy.sealed_structural_negative_count > 0);
        assert!(policy.sealed_exact_positive_count > 0);
        assert_eq!(
            policy.provenance,
            DedupCalibrationProvenance::StructuralAnchors
        );
        assert_eq!(policy.status, DedupCalibrationStatus::NoData);
        assert_eq!(policy.effective_hard_threshold, 0.70);
    }

    #[test]
    fn corpus_builder_accepts_only_explicit_snapshotted_operator_labels() {
        let store = SqliteStore::in_memory().unwrap();
        let config = ReinConfig::default();
        for (id, content) in [("left", "tenant alpha"), ("right", "tenant beta")] {
            let mut memory = crate::ops::build_memory(
                &config,
                "calibration".to_string(),
                content.to_string(),
                crate::types::Importance::Medium,
                vec![],
                crate::types::Source::Manual,
            );
            memory.id = id.to_string();
            memory.created_at = chrono::DateTime::from_timestamp(800, 0).unwrap();
            store.store(memory).unwrap();
        }
        let created_at = chrono::DateTime::from_timestamp(900, 0)
            .unwrap()
            .to_rfc3339();
        let payload = serde_json::json!({
            "calibration_label": {
                "version": 1,
                "operator_confirmed": true,
                "left_memory_id": "left",
                "right_memory_id": "right",
                "left_content": "shared words with distinct tenant alpha",
                "right_content": "shared words with distinct tenant beta"
            }
        });
        for (id, operator, row_payload) in [
            ("accepted", "operator_label", Some(payload.to_string())),
            ("auto-ignored", "auto", Some(payload.to_string())),
            ("missing-proof", "operator_label", None),
        ] {
            store
                .conn()
                .execute(
                    "INSERT INTO dedup_decisions
                     (id, relation, confidence, reason, operator, reversible, novel_facts,
                      conflict_detected, payload, created_at)
                     VALUES (?1, 'distinct', 1.0, 'test', ?2, 1, '[]', 0, ?3, ?4)",
                    rusqlite::params![id, operator, row_payload, created_at],
                )
                .unwrap();
        }

        let sealed = build_dedup_calibration_corpus(&store, 1, 999).unwrap();
        let operator_cases = sealed
            .cases
            .iter()
            .filter(|case| case.source == DedupCalibrationCaseSource::OperatorDistinct)
            .count();
        assert_eq!(operator_cases, 1);
    }

    #[test]
    fn corpus_builder_requires_fixed_canonical_root_and_uses_semantic_slices() {
        let store = SqliteStore::in_memory().unwrap();
        let config = ReinConfig::default();
        let other_root_id = roots_in_fold(
            dedup_calibration_fold("old-root"),
            1,
            "other-canonical-root",
        )
        .remove(0);
        let mut old_root = crate::ops::build_memory(
            &config,
            "calibration".to_string(),
            "old canonical revision".to_string(),
            crate::types::Importance::Medium,
            vec![],
            crate::types::Source::Manual,
        );
        old_root.id = "old-root".to_string();
        old_root.created_at = chrono::DateTime::from_timestamp(700, 0).unwrap();
        let mut live_tip = crate::ops::build_memory(
            &config,
            "calibration".to_string(),
            "live canonical revision".to_string(),
            crate::types::Importance::Medium,
            vec![],
            crate::types::Source::Manual,
        );
        live_tip.id = "live-tip".to_string();
        live_tip.created_at = chrono::DateTime::from_timestamp(800, 0).unwrap();
        let mut other_root = crate::ops::build_memory(
            &config,
            "calibration".to_string(),
            "other old canonical revision".to_string(),
            crate::types::Importance::Medium,
            vec![],
            crate::types::Source::Manual,
        );
        other_root.id = other_root_id.clone();
        other_root.created_at = chrono::DateTime::from_timestamp(710, 0).unwrap();
        let mut other_tip = crate::ops::build_memory(
            &config,
            "calibration".to_string(),
            "other live canonical revision".to_string(),
            crate::types::Importance::Medium,
            vec![],
            crate::types::Source::Manual,
        );
        other_tip.id = "other-tip".to_string();
        other_tip.created_at = chrono::DateTime::from_timestamp(810, 0).unwrap();
        store.store(old_root).unwrap();
        store.store(live_tip).unwrap();
        store.store(other_root).unwrap();
        store.store(other_tip).unwrap();
        store.mark_superseded("old-root", "live-tip").unwrap();
        store.mark_superseded(&other_root_id, "other-tip").unwrap();
        let created_at = chrono::DateTime::from_timestamp(900, 0)
            .unwrap()
            .to_rfc3339();
        let insert = |id: &str,
                      relation: &str,
                      canonical_id: Option<&str>,
                      left_id: &str,
                      right_id: &str,
                      left: &str,
                      right: &str| {
            let payload = serde_json::json!({
                "calibration_label": {
                    "version": 1,
                    "operator_confirmed": true,
                    "left_memory_id": left_id,
                    "right_memory_id": right_id,
                    "left_content": left,
                    "right_content": right
                }
            });
            store
                .conn()
                .execute(
                    "INSERT INTO dedup_decisions
                     (id, canonical_id, relation, confidence, reason, operator, reversible,
                      novel_facts, conflict_detected, payload, created_at)
                     VALUES (?1, ?2, ?3, 1.0, 'test', 'operator_label', 1, '[]', 0, ?4, ?5)",
                    rusqlite::params![id, canonical_id, relation, payload.to_string(), created_at],
                )
                .unwrap();
        };

        insert(
            "positive-a",
            "duplicate",
            Some("old-root"),
            "old-root",
            "live-tip",
            "alpha beta gamma delta",
            "alpha beta gamma epsilon",
        );
        insert(
            "positive-b",
            "duplicate",
            Some("live-tip"),
            "live-tip",
            "old-root",
            "alpha beta gamma delta",
            "alpha beta gamma zeta",
        );
        insert(
            "positive-missing-root",
            "duplicate",
            None,
            "old-root",
            "live-tip",
            "one two three",
            "one two four",
        );
        insert(
            "positive-arbitrary-root",
            "duplicate",
            Some("arbitrary-root"),
            "old-root",
            "live-tip",
            "one two three",
            "one two four",
        );
        insert(
            "positive-cross-family",
            "duplicate",
            Some("old-root"),
            "old-root",
            &other_root_id,
            "one two three",
            "one two four",
        );
        insert(
            "positive-identical-bytes",
            "duplicate",
            Some("old-root"),
            "old-root",
            "live-tip",
            "identical immutable snapshot",
            "identical immutable snapshot",
        );
        insert(
            "positive-same-member",
            "duplicate",
            Some("old-root"),
            "old-root",
            "old-root",
            "same snapshot once",
            "same snapshot twice",
        );
        insert(
            "operator-distinct-old-revisions",
            "distinct",
            None,
            "old-root",
            &other_root_id,
            "shared words tenant alpha",
            "shared words tenant beta",
        );
        insert(
            "operator-distinct-live-revisions",
            "distinct",
            None,
            "live-tip",
            "other-tip",
            "shared words tenant alpha current",
            "shared words tenant beta current",
        );
        insert(
            "operator-distinct-same-family",
            "distinct",
            None,
            "old-root",
            "live-tip",
            "same family old snapshot",
            "same family live snapshot",
        );

        let sealed = build_dedup_calibration_corpus(&store, 1, 999).unwrap();
        let canonical_rows: Vec<&DedupCalibrationCase> = sealed
            .cases
            .iter()
            .filter(|case| case.source == DedupCalibrationCaseSource::CanonicalFamily)
            .collect();
        assert_eq!(
            canonical_rows.len(),
            2,
            "missing canonical root must be rejected"
        );
        assert_eq!(
            canonical_rows
                .iter()
                .map(|case| case.family_id.as_str())
                .collect::<std::collections::HashSet<_>>()
                .len(),
            1,
            "two labels for one fixed root are one independent family"
        );
        assert!(canonical_rows
            .iter()
            .all(|case| case.slice == DedupCalibrationSliceKind::CanonicalFamily));

        let operator_rows: Vec<&DedupCalibrationCase> = sealed
            .cases
            .iter()
            .filter(|case| case.source == DedupCalibrationCaseSource::OperatorDistinct)
            .collect();
        assert_eq!(operator_rows.len(), 2);
        assert!(operator_rows
            .iter()
            .all(|case| case.slice == DedupCalibrationSliceKind::OperatorDistinct));
        assert_eq!(
            operator_rows
                .iter()
                .map(|case| case.family_id.as_str())
                .collect::<std::collections::HashSet<_>>()
                .len(),
            1,
            "record revisions for the same two roots must not inflate distinct ESS or cross folds"
        );
        assert_eq!(
            operator_rows
                .iter()
                .map(|case| dedup_calibration_fold(&case.family_id))
                .collect::<std::collections::HashSet<_>>()
                .len(),
            1
        );
        let (independent, rejected) = independent_cases(&sealed.cases);
        assert_eq!(rejected, 0);
        assert_eq!(
            independent
                .iter()
                .filter(|case| case.source == DedupCalibrationCaseSource::CanonicalFamily)
                .count(),
            1,
            "positive revisions must count as one family"
        );
        assert_eq!(
            independent
                .iter()
                .filter(|case| case.source == DedupCalibrationCaseSource::OperatorDistinct)
                .count(),
            1,
            "distinct record revisions must count as one root-pair family"
        );
    }

    #[test]
    fn refresh_keeps_underpowered_no_data_completely_read_only() {
        let store = SqliteStore::in_memory().unwrap();
        let first = refresh_dedup_calibration_policy(&store, 0.70, 0.40, 1_000, 1_000)
            .expect("first refresh");
        assert_eq!(first.revision, 0);
        assert_eq!(first.status, DedupCalibrationStatus::NoData);

        let loaded = crate::store::dedup_calibration::load_dedup_calibration_for_runtime(
            store.conn(),
            1_500,
            0.70,
        );
        assert_eq!(
            loaded.status,
            crate::store::dedup_calibration::DedupCalibrationLoadStatus::Missing
        );
        assert!(!loaded.context_verified());
        assert_eq!(
            crate::store::dedup_calibration::resolve_hard_lexical_threshold(0.70, &loaded),
            0.70
        );
        let persisted_rows: usize = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM metadata WHERE key IN (?1, ?2)",
                rusqlite::params![
                    crate::store::dedup_calibration::DEDUP_CALIBRATION_METADATA_KEY,
                    crate::store::dedup_calibration::DEDUP_CALIBRATION_SEAL_METADATA_KEY
                ],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(persisted_rows, 0);

        let second = refresh_dedup_calibration_policy(&store, 0.70, 0.40, 1_600, 1_000)
            .expect("second refresh");
        assert_eq!(second.revision, 0);
        assert_eq!(second.sealed_generation, first.sealed_generation);
        let persisted_rows: usize = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM metadata WHERE key IN (?1, ?2)",
                rusqlite::params![
                    crate::store::dedup_calibration::DEDUP_CALIBRATION_METADATA_KEY,
                    crate::store::dedup_calibration::DEDUP_CALIBRATION_SEAL_METADATA_KEY
                ],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(persisted_rows, 0);
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
