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
#[derive(Debug, Clone, Serialize)]
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

/// Full sweep report: the per-threshold curve plus the data-derived optimum.
#[derive(Debug, Clone, Serialize)]
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
    /// HEADLINE recommendation for a hard auto-merge bound: the threshold with
    /// the highest recall among those achieving precision == 1.0 (zero false
    /// merges on the corpus). A false merge destroys data, while a miss falls
    /// through to the gray-zone / LLM path — so precision is the priority for a
    /// merge bound, NOT F1. `None` if no threshold reaches precision 1.0.
    pub merge_safe_optimal: Option<ThresholdStat>,
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

/// Pick the merge-safe optimum: among thresholds with precision == 1.0 (zero
/// false merges), the one with the highest recall; tie-break LOWER threshold
/// (merge as much as is provably safe). `None` if no threshold is false-merge
/// free on this corpus. This is the correct objective for a HARD auto-merge
/// bound, where a false positive is data loss.
pub fn merge_safe_threshold(curve: &[ThresholdStat]) -> Option<ThresholdStat> {
    curve
        .iter()
        .filter(|s| s.precision >= 1.0 - 1e-9 && s.tp + s.fp > 0)
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
    let power_note = format!(
        "n={} ({} duplicate / {} distinct) — power-limited; directional only. \
         This sweeps LEXICAL similarity (max Jaccard/containment) separability on \
         the corpus. Production dedup is MULTI-SIGNAL (lexical bound + embedding- \
         cosine path + gray-zone LLM verdict) and per-cluster adaptive (M6), so \
         this calibrates the lexical bound in isolation, NOT the full merge \
         decision. Threshold defaults are left UNCHANGED pending a production- \
         traffic recalibration sample.",
        fixtures.len(),
        positives,
        negatives
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
        assert_eq!(report.curve.len(), 14);
        assert!(report.positives > 0 && report.negatives > 0);
        assert_eq!(report.fixture_count, report.positives + report.negatives);
        assert!(report.max_f1_point.f1 >= 0.0 && report.max_f1_point.f1 <= 1.0);
        assert!((report.current_gate_threshold - DEDUP_THRESHOLD).abs() < 1e-6);
        // If a merge-safe point exists, it must have zero false positives.
        if let Some(ref ms) = report.merge_safe_optimal {
            assert_eq!(ms.fp, 0, "merge-safe optimum must have precision 1.0");
        }
    }

    #[test]
    fn merge_safe_prefers_lowest_zero_fp_threshold() {
        // Perfectly separable → precision 1.0 and recall 1.0 at every threshold;
        // merge-safe tie-break picks the LOWEST (merge as much as is safe).
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
