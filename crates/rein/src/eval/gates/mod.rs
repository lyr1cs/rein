//! v0.32 Trust & Measurement Phase 2 — eval gate harness.
//!
//! Public types, scorecard JSON I/O, comparison logic, and the `Gate` trait
//! shared by `recall`, `dedup`, `admission`, and `latency` gates.
//!
//! Sibling files (`recall.rs`, `dedup.rs`, `admission.rs`, `latency.rs`) carry
//! the per-gate implementations; this module owns the common surface.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::config::ReinConfig;
use crate::eval::mcnemar::{mcnemar, McNemarResult, PairedOutcome};
use crate::store::SqliteStore;

pub mod admission;
pub mod dedup;
pub mod latency;
pub mod recall;

pub const SCORECARD_SCHEMA_VERSION: u32 = 1;

/// Default noise floor for ship/bail decisions: 0.02 (2pp hit-rate move).
/// `mcnemar.ci_lower >= -NOISE_FLOOR` → Ship (non-inferior).
/// `mcnemar.ci_upper <= -NOISE_FLOOR` → Bail (clearly worse).
/// Anything else → NoData (CI straddles the noise floor — underpowered).
pub const DEFAULT_NOISE_FLOOR: f64 = 0.02;

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScorecardKind {
    Baseline,
    Run,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScorecardStatus {
    /// Current >= baseline within noise floor (McNemar non-inferiority CI passes).
    Ship,
    /// Current < baseline beyond noise floor (regression detected).
    Bail,
    /// One side missing, gate is a stub, or paired sample too small / CI straddles noise floor.
    NoData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixtureResult {
    pub fixture_id: String,
    pub hit: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateScorecard {
    pub schema_version: u32,
    pub gate_name: String,
    pub kind: ScorecardKind,
    pub created_at: i64,      // Unix seconds
    pub rein_version: String, // env!("CARGO_PKG_VERSION")
    pub fixture_count: usize,
    pub score: f64,                      // hit-rate in [0.0, 1.0]
    pub per_fixture: Vec<FixtureResult>, // ordered, parallel for paired McNemar
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateComparison {
    pub gate_name: String,
    pub baseline_scorecard: Option<GateScorecard>,
    pub current_scorecard: Option<GateScorecard>,
    pub status: ScorecardStatus,
    pub reason: String,                 // human-readable diagnostic
    pub mcnemar: Option<McNemarResult>, // None when either side is missing or stub
}

pub trait Gate: Send + Sync {
    fn name(&self) -> &'static str;
    /// Returns a `Run` scorecard. Callers that intend a baseline override `.kind` to Baseline.
    fn run(&self, store: &SqliteStore, config: &ReinConfig) -> Result<GateScorecard>;
    /// True for placeholder gates that return empty scorecards in v0.32.
    fn is_stub(&self) -> bool {
        false
    }
}

/// Compute paired McNemar for two scorecards, matched by `fixture_id`.
///
/// If `fixture_id` sets differ (e.g., baseline has fixtures the run doesn't),
/// we use the INTERSECTION of fixture_ids; the comparison's `reason` documents
/// the diff.
///
/// Classification order (early-exit):
/// 1. Presence check — either side `None` → NoData.
/// 2. Schema version check — either side != `SCORECARD_SCHEMA_VERSION` → NoData.
/// 3. Stub check — either side `fixture_count == 0` → NoData.
/// 4. Intersection emptiness — no overlapping fixture ids → NoData.
/// 5. McNemar on the intersection. `ci_lower >= -noise_floor` → Ship;
///    `ci_upper <= -noise_floor` → Bail; otherwise NoData (underpowered).
/// v0.32 R1 P3: `gate_name` is taken as an explicit parameter so the
/// comparison always carries a meaningful name even when both
/// scorecards are absent.  The earlier "guess from `baseline.gate_name`
/// or `current.gate_name`, else empty string" fallback produced
/// indistinguishable `gate_name: ""` entries in `rein-eval gate compare
/// --gate all` output when no scorecards had been generated yet.
pub fn compare_scorecards(
    gate_name: &str,
    baseline: Option<&GateScorecard>,
    current: Option<&GateScorecard>,
    noise_floor: f64,
) -> GateComparison {
    let gate_name = gate_name.to_string();

    // Rule 2 (presence): either side missing → NoData.
    let baseline = match baseline {
        Some(b) => b,
        None => {
            return GateComparison {
                gate_name,
                baseline_scorecard: None,
                current_scorecard: current.cloned(),
                status: ScorecardStatus::NoData,
                reason: "no baseline scorecard; run 'rein-eval gate baseline <name>'".to_string(),
                mcnemar: None,
            };
        }
    };
    let current = match current {
        Some(c) => c,
        None => {
            return GateComparison {
                gate_name,
                baseline_scorecard: Some(baseline.clone()),
                current_scorecard: None,
                status: ScorecardStatus::NoData,
                reason: "no current scorecard; run 'rein-eval gate run <name>'".to_string(),
                mcnemar: None,
            };
        }
    };

    // Rule 3 (schema version): mismatch on either side → NoData.
    if baseline.schema_version != SCORECARD_SCHEMA_VERSION {
        return GateComparison {
            gate_name,
            baseline_scorecard: Some(baseline.clone()),
            current_scorecard: Some(current.clone()),
            status: ScorecardStatus::NoData,
            reason: format!(
                "schema_version {} != expected {}",
                baseline.schema_version, SCORECARD_SCHEMA_VERSION
            ),
            mcnemar: None,
        };
    }
    if current.schema_version != SCORECARD_SCHEMA_VERSION {
        return GateComparison {
            gate_name,
            baseline_scorecard: Some(baseline.clone()),
            current_scorecard: Some(current.clone()),
            status: ScorecardStatus::NoData,
            reason: format!(
                "schema_version {} != expected {}",
                current.schema_version, SCORECARD_SCHEMA_VERSION
            ),
            mcnemar: None,
        };
    }

    // v0.32 R4 P2-#1: validate scorecard identity (`gate_name` + `kind`)
    // before running McNemar.  Without this, `rein-eval gate compare
    // --gate recall --baseline dedup.json` would silently produce a
    // recall-labeled ship/bail decision computed from a dedup scorecard
    // if fixture-id sets happened to overlap.  A baseline scorecard
    // must carry `kind=Baseline`; a current scorecard must carry
    // `kind=Run`; both must match the caller's expected `gate_name`.
    if baseline.gate_name != gate_name {
        return GateComparison {
            gate_name: gate_name.clone(),
            baseline_scorecard: Some(baseline.clone()),
            current_scorecard: Some(current.clone()),
            status: ScorecardStatus::NoData,
            reason: format!(
                "baseline scorecard is for gate '{}', expected '{}'",
                baseline.gate_name, gate_name
            ),
            mcnemar: None,
        };
    }
    if current.gate_name != gate_name {
        return GateComparison {
            gate_name: gate_name.clone(),
            baseline_scorecard: Some(baseline.clone()),
            current_scorecard: Some(current.clone()),
            status: ScorecardStatus::NoData,
            reason: format!(
                "current scorecard is for gate '{}', expected '{}'",
                current.gate_name, gate_name
            ),
            mcnemar: None,
        };
    }
    if baseline.kind != ScorecardKind::Baseline {
        return GateComparison {
            gate_name: gate_name.clone(),
            baseline_scorecard: Some(baseline.clone()),
            current_scorecard: Some(current.clone()),
            status: ScorecardStatus::NoData,
            reason: format!(
                "baseline argument has kind={:?} (expected Baseline) — \
                 was the file written by `rein-eval gate baseline`?",
                baseline.kind
            ),
            mcnemar: None,
        };
    }
    if current.kind != ScorecardKind::Run {
        return GateComparison {
            gate_name: gate_name.clone(),
            baseline_scorecard: Some(baseline.clone()),
            current_scorecard: Some(current.clone()),
            status: ScorecardStatus::NoData,
            reason: format!(
                "current argument has kind={:?} (expected Run) — \
                 was the file written by `rein-eval gate run`?",
                current.kind
            ),
            mcnemar: None,
        };
    }

    // v0.32 R8 P2: stale `target/eval-gates/<gate>-run.json` from a
    // previous build silently passes the gate/kind/fixture-id checks
    // because the file is gitignored and persists across local code
    // changes.  Without a freshness check `rein doctor` /
    // `rein trust-measurement` can report Ship for a revision that
    // hasn't been re-run.  Compare the run's `rein_version` against
    // the current binary's compile-time `CARGO_PKG_VERSION` — a
    // mismatch is a strong signal that the run is stale.
    //
    // Baselines are committed and intentionally older than the current
    // version (that's why they're baselines); only the run side gets
    // this check.  Operators who bump the crate version implicitly
    // force a re-run, which is the desired safety behavior.
    let expected_version = env!("CARGO_PKG_VERSION");
    if current.rein_version != expected_version {
        return GateComparison {
            gate_name: gate_name.clone(),
            baseline_scorecard: Some(baseline.clone()),
            current_scorecard: Some(current.clone()),
            status: ScorecardStatus::NoData,
            reason: format!(
                "run scorecard was generated for rein v{} but the current binary is v{}; \
                 re-run `rein-eval gate run --gate {}` before comparing",
                current.rein_version, expected_version, gate_name
            ),
            mcnemar: None,
        };
    }

    // Rule 1 (stub): either side empty → NoData.
    if baseline.fixture_count == 0 || current.fixture_count == 0 {
        return GateComparison {
            gate_name,
            baseline_scorecard: Some(baseline.clone()),
            current_scorecard: Some(current.clone()),
            status: ScorecardStatus::NoData,
            reason: "stub gate".to_string(),
            mcnemar: None,
        };
    }

    // Build set indices for symmetric difference detection.
    let baseline_ids: std::collections::HashSet<&str> = baseline
        .per_fixture
        .iter()
        .map(|f| f.fixture_id.as_str())
        .collect();
    let current_ids: std::collections::HashSet<&str> = current
        .per_fixture
        .iter()
        .map(|f| f.fixture_id.as_str())
        .collect();

    // v0.32 R7 P2: reject scorecards containing duplicate `fixture_id`
    // entries.  The `HashSet` collapses dupes silently, so a malformed
    // scorecard (e.g. a copied fixture JSON that kept the same `id`)
    // would slip through the id-set-equality gate with both sides
    // looking valid; the downstream `current_index` `HashMap` would
    // then keep only one row per duplicate-id, and McNemar would run
    // over a dropped-rows-or-duplicated-baseline-rows mix — a false
    // ship/bail call on an obviously broken corpus.  Catching it here
    // forces the operator to fix the duplicate IDs before the
    // comparison can run.
    if baseline_ids.len() != baseline.per_fixture.len() {
        return GateComparison {
            gate_name: gate_name.clone(),
            baseline_scorecard: Some(baseline.clone()),
            current_scorecard: Some(current.clone()),
            status: ScorecardStatus::NoData,
            reason: format!(
                "baseline has {} per_fixture entries but only {} unique fixture_ids — \
                 duplicate ids would silently collapse during pairing; \
                 deduplicate or rename before comparing",
                baseline.per_fixture.len(),
                baseline_ids.len()
            ),
            mcnemar: None,
        };
    }
    if current_ids.len() != current.per_fixture.len() {
        return GateComparison {
            gate_name: gate_name.clone(),
            baseline_scorecard: Some(baseline.clone()),
            current_scorecard: Some(current.clone()),
            status: ScorecardStatus::NoData,
            reason: format!(
                "current has {} per_fixture entries but only {} unique fixture_ids — \
                 duplicate ids would silently collapse during pairing; \
                 deduplicate or rename before comparing",
                current.per_fixture.len(),
                current_ids.len()
            ),
            mcnemar: None,
        };
    }

    // v0.32 R3 P2-#1: STRICT id-set equality.  Earlier we silently took the
    // intersection, which could let a partial run (e.g. half the fixtures
    // were skipped due to env / network / timing) or a renamed/added/deleted
    // fixture between baseline and current still produce a Ship via the
    // shrunken paired vector.  Now any symmetric difference forces NoData
    // with a diagnostic listing the missing ids, so the caller knows
    // exactly which fixtures need to be regenerated or renamed before a
    // ship/bail call is meaningful.
    let missing_in_current: Vec<&&str> = baseline_ids.difference(&current_ids).collect();
    let missing_in_baseline: Vec<&&str> = current_ids.difference(&baseline_ids).collect();
    if !missing_in_current.is_empty() || !missing_in_baseline.is_empty() {
        // Sort for deterministic diagnostic output.
        let mut mc: Vec<String> = missing_in_current.iter().map(|s| s.to_string()).collect();
        mc.sort();
        let mut mb: Vec<String> = missing_in_baseline.iter().map(|s| s.to_string()).collect();
        mb.sort();
        let preview = |v: &[String]| -> String {
            const N: usize = 5;
            if v.len() <= N {
                v.join(",")
            } else {
                format!("{},... (+{} more)", v[..N].join(","), v.len() - N)
            }
        };
        let reason = if !mc.is_empty() && !mb.is_empty() {
            format!(
                "fixture_id sets differ: {} missing in current ({}), {} missing in baseline ({})",
                mc.len(),
                preview(&mc),
                mb.len(),
                preview(&mb),
            )
        } else if !mc.is_empty() {
            format!(
                "fixture_id sets differ: {} missing in current ({}); re-run \
                 the gate against the same fixture corpus before comparing",
                mc.len(),
                preview(&mc),
            )
        } else {
            format!(
                "fixture_id sets differ: {} missing in baseline ({}); re-baseline \
                 to capture new fixtures before comparing",
                mb.len(),
                preview(&mb),
            )
        };
        return GateComparison {
            gate_name,
            baseline_scorecard: Some(baseline.clone()),
            current_scorecard: Some(current.clone()),
            status: ScorecardStatus::NoData,
            reason,
            mcnemar: None,
        };
    }

    // Build paired outcomes — fixture_id sets equal here, so the
    // intersection-via-loop is just a deterministic re-key on baseline
    // ordering (needed for stable McNemar reproducibility).
    let current_index: HashMap<&str, bool> = current
        .per_fixture
        .iter()
        .map(|f| (f.fixture_id.as_str(), f.hit))
        .collect();

    let mut paired = Vec::new();
    for b_fx in &baseline.per_fixture {
        if let Some(&t_hit) = current_index.get(b_fx.fixture_id.as_str()) {
            paired.push(PairedOutcome {
                case_id: b_fx.fixture_id.clone(),
                baseline_hit: b_fx.hit,
                treatment_hit: t_hit,
                baseline_length: 0,
                treatment_length: 0,
                treatment_summary: None,
            });
        }
    }

    // Defensive: should be unreachable now that ids must match — but keep
    // the empty-paired guard so the function never feeds an empty vector
    // into mcnemar() in case future refactors weaken the strict check.
    if paired.is_empty() {
        return GateComparison {
            gate_name,
            baseline_scorecard: Some(baseline.clone()),
            current_scorecard: Some(current.clone()),
            status: ScorecardStatus::NoData,
            reason: format!(
                "no overlapping fixture ids between baseline ({}) and current ({})",
                baseline.per_fixture.len(),
                current.per_fixture.len()
            ),
            mcnemar: None,
        };
    }

    // Rule 5: McNemar on the intersection.
    let result = mcnemar(&paired);
    let (status, reason) = if result.ci_lower >= -noise_floor {
        (
            ScorecardStatus::Ship,
            format!(
                "non-inferior: ci_lower={:.4} >= -noise_floor={:.4}",
                result.ci_lower, noise_floor
            ),
        )
    } else if result.ci_upper <= -noise_floor {
        (
            ScorecardStatus::Bail,
            format!(
                "regressed: ci_upper={:.4} <= -noise_floor={:.4}",
                result.ci_upper, noise_floor
            ),
        )
    } else {
        (
            ScorecardStatus::NoData,
            format!(
                "underpowered: CI=[{:.4},{:.4}] straddles -noise_floor={:.4}",
                result.ci_lower, result.ci_upper, noise_floor
            ),
        )
    };

    GateComparison {
        gate_name,
        baseline_scorecard: Some(baseline.clone()),
        current_scorecard: Some(current.clone()),
        status,
        reason,
        mcnemar: Some(result),
    }
}

/// Serialize `scorecard` to `path` as pretty JSON, creating parent dirs as needed.
pub fn write_scorecard(path: &Path, scorecard: &GateScorecard) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create parent dir for {}", path.display()))?;
        }
    }
    let body = serde_json::to_string_pretty(scorecard).context("serialize scorecard to JSON")?;
    fs::write(path, body).with_context(|| format!("write scorecard to {}", path.display()))?;
    Ok(())
}

/// v0.32 R4 P2-#2: tri-state load result that lets diagnostic surfaces
/// (`doctor`, `trust_measurement`, `rein-eval gate status`) distinguish
/// "this file doesn't exist (operator just hasn't generated it yet)"
/// from "this file exists but is corrupt (committed JSON is malformed
/// or truncated)".  Earlier callers wrapped `read_scorecard()` with
/// `.ok()` and surfaced both cases as `no_baseline` / `no_run` —
/// silently passing diagnostics over genuinely broken artifacts.
#[derive(Debug)]
pub enum ScorecardLoad {
    Loaded(GateScorecard),
    Missing,
    Corrupt(String),
}

/// Like `read_scorecard` but classifies missing-vs-corrupt without
/// requiring callers to inspect `std::io::ErrorKind`.
pub fn load_scorecard(path: &Path) -> ScorecardLoad {
    if !path.exists() {
        return ScorecardLoad::Missing;
    }
    match read_scorecard(path) {
        Ok(sc) => ScorecardLoad::Loaded(sc),
        Err(e) => ScorecardLoad::Corrupt(e.to_string()),
    }
}

/// Read and parse a scorecard JSON file.
///
/// v0.32 R1 P2-#2: this function deliberately does NOT validate
/// `schema_version` — schema-drift detection is the comparison layer's
/// responsibility (see `compare_scorecards` Rule 2, which classifies a
/// mismatched scorecard as `NoData` with a descriptive `reason`).
/// Aborting here would cause `cmd_gate_compare` /  `check_eval_gates` /
/// `trust_measurement::eval_gates()` callers (which use `.ok()` to
/// absorb errors as "treat as missing") to silently swallow schema
/// mismatches as if the scorecard were absent — losing the
/// schema-version signal.  By parsing unconditionally we let the
/// comparison layer surface the real cause.
pub fn read_scorecard(path: &Path) -> Result<GateScorecard> {
    let body = fs::read_to_string(path)
        .with_context(|| format!("read scorecard from {}", path.display()))?;
    let scorecard: GateScorecard = serde_json::from_str(&body)
        .with_context(|| format!("parse scorecard JSON from {}", path.display()))?;
    Ok(scorecard)
}

/// `repo_root/docs/eval-baselines/{gate_name}.json` — committed baseline.
pub fn baseline_path(repo_root: &Path, gate_name: &str) -> PathBuf {
    repo_root
        .join("docs/eval-baselines")
        .join(format!("{gate_name}.json"))
}

/// `target_dir/eval-gates/{gate_name}-run.json` — local run output.
pub fn run_path(target_dir: &Path, gate_name: &str) -> PathBuf {
    target_dir
        .join("eval-gates")
        .join(format!("{gate_name}-run.json"))
}

/// Registry — kept tight for the v0.32 four-gate set.
pub fn all_gates() -> Vec<Box<dyn Gate>> {
    vec![
        Box::new(recall::RecallGate),
        Box::new(dedup::DedupGate),
        Box::new(admission::AdmissionGate),
        Box::new(latency::LatencyGate),
    ]
}

pub fn gate_by_name(name: &str) -> Option<Box<dyn Gate>> {
    all_gates().into_iter().find(|g| g.name() == name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    /// Helper: build a synthetic scorecard with the given per-fixture hits.
    fn make_scorecard(
        gate_name: &str,
        kind: ScorecardKind,
        per_fixture: Vec<(&str, bool)>,
    ) -> GateScorecard {
        let per_fixture: Vec<FixtureResult> = per_fixture
            .into_iter()
            .map(|(id, hit)| FixtureResult {
                fixture_id: id.to_string(),
                hit,
            })
            .collect();
        let fixture_count = per_fixture.len();
        let hits = per_fixture.iter().filter(|f| f.hit).count() as f64;
        let total = fixture_count as f64;
        let score = if total > 0.0 { hits / total } else { 0.0 };
        GateScorecard {
            schema_version: SCORECARD_SCHEMA_VERSION,
            gate_name: gate_name.to_string(),
            kind,
            created_at: 1_700_000_000,
            // v0.32 R8 P2: tests must use the current binary version so
            // the new freshness check in compare_scorecards (Rule
            // current.rein_version == env!("CARGO_PKG_VERSION"))
            // passes for healthy test fixtures.  Tests that exercise
            // the stale-run path override this field inline after
            // creation.
            rein_version: env!("CARGO_PKG_VERSION").to_string(),
            fixture_count,
            score,
            per_fixture,
        }
    }

    #[test]
    fn compare_scorecards_returns_no_data_when_baseline_missing() {
        let current = make_scorecard(
            "recall",
            ScorecardKind::Run,
            vec![("a", true), ("b", false)],
        );
        // v0.32 R1 P3: explicit gate_name flows through compare even when
        // both sides are absent — pass "recall" here so the assertion below
        // exercises the new "name is taken from the parameter, not guessed
        // from the scorecard" contract.
        let cmp = compare_scorecards("recall", None, Some(&current), DEFAULT_NOISE_FLOOR);
        assert_eq!(cmp.status, ScorecardStatus::NoData);
        assert!(
            cmp.reason.contains("no baseline scorecard"),
            "unexpected reason: {}",
            cmp.reason
        );
        assert!(cmp.mcnemar.is_none());
        assert!(cmp.baseline_scorecard.is_none());
        assert!(cmp.current_scorecard.is_some());
        assert_eq!(cmp.gate_name, "recall");
    }

    #[test]
    fn compare_scorecards_returns_no_data_when_stub() {
        // baseline non-empty, current empty (stub side).
        let baseline = make_scorecard(
            "dedup",
            ScorecardKind::Baseline,
            vec![("a", true), ("b", true)],
        );
        let current = make_scorecard("dedup", ScorecardKind::Run, vec![]);
        let cmp = compare_scorecards(
            "dedup",
            Some(&baseline),
            Some(&current),
            DEFAULT_NOISE_FLOOR,
        );
        assert_eq!(cmp.status, ScorecardStatus::NoData);
        assert_eq!(cmp.reason, "stub gate");
        assert!(cmp.mcnemar.is_none());

        // Reverse direction: baseline empty.
        let baseline = make_scorecard("dedup", ScorecardKind::Baseline, vec![]);
        let current = make_scorecard("dedup", ScorecardKind::Run, vec![("a", true)]);
        let cmp = compare_scorecards(
            "dedup",
            Some(&baseline),
            Some(&current),
            DEFAULT_NOISE_FLOOR,
        );
        assert_eq!(cmp.status, ScorecardStatus::NoData);
        assert_eq!(cmp.reason, "stub gate");
    }

    #[test]
    fn compare_scorecards_rejects_fixture_id_set_mismatch() {
        // v0.32 R3 P2-#1: STRICT id-set equality.  baseline = [a, b, c],
        // current = [b, c, d] — the symmetric difference is non-empty
        // (`a` missing in current, `d` missing in baseline) so the
        // comparison must return NoData with a diagnostic, NOT silently
        // ship the intersection.  Earlier the comparison built the
        // intersection [b, c], saw both concordant hits, and reported
        // Ship — masking the fact that the run skipped `a` and added
        // an unmatched `d`.
        let baseline = make_scorecard(
            "recall",
            ScorecardKind::Baseline,
            vec![("a", false), ("b", true), ("c", true)],
        );
        let current = make_scorecard(
            "recall",
            ScorecardKind::Run,
            vec![("b", true), ("c", true), ("d", false)],
        );
        let cmp = compare_scorecards(
            "recall",
            Some(&baseline),
            Some(&current),
            DEFAULT_NOISE_FLOOR,
        );
        assert_eq!(cmp.status, ScorecardStatus::NoData);
        assert!(cmp.mcnemar.is_none());
        assert!(
            cmp.reason.contains("missing in current") && cmp.reason.contains('a'),
            "reason should call out the missing-in-current fixture 'a': {}",
            cmp.reason
        );
        assert!(
            cmp.reason.contains("missing in baseline") && cmp.reason.contains('d'),
            "reason should call out the missing-in-baseline fixture 'd': {}",
            cmp.reason
        );
    }

    #[test]
    fn compare_scorecards_accepts_identical_fixture_id_sets() {
        // Same shape as above but with identical id sets — McNemar must run.
        let baseline = make_scorecard(
            "recall",
            ScorecardKind::Baseline,
            vec![("a", true), ("b", true), ("c", true)],
        );
        let current = make_scorecard(
            "recall",
            ScorecardKind::Run,
            vec![("a", true), ("b", true), ("c", true)],
        );
        let cmp = compare_scorecards(
            "recall",
            Some(&baseline),
            Some(&current),
            DEFAULT_NOISE_FLOOR,
        );
        let result = cmp
            .mcnemar
            .as_ref()
            .expect("McNemar should run on equal id-set scorecards");
        assert_eq!(result.n, 3);
        assert_eq!(result.a, 3);
        assert_eq!(cmp.status, ScorecardStatus::Ship);
    }

    /// v0.32 R7 P2: duplicate `fixture_id` entries on either side must
    /// be rejected before pairing.  Earlier the HashSet collapsed dupes
    /// silently and McNemar ran over a dropped-rows/duplicated-rows mix.
    #[test]
    fn compare_scorecards_rejects_duplicate_fixture_ids_in_baseline() {
        let baseline = make_scorecard(
            "recall",
            ScorecardKind::Baseline,
            // Duplicate "a" — id-set collapses to {a, b} (size 2) but
            // per_fixture has 3 entries.
            vec![("a", true), ("a", false), ("b", true)],
        );
        let current = make_scorecard("recall", ScorecardKind::Run, vec![("a", true), ("b", true)]);
        let cmp = compare_scorecards(
            "recall",
            Some(&baseline),
            Some(&current),
            DEFAULT_NOISE_FLOOR,
        );
        assert_eq!(cmp.status, ScorecardStatus::NoData);
        assert!(cmp.mcnemar.is_none());
        assert!(
            cmp.reason.contains("baseline")
                && cmp.reason.contains("duplicate")
                && cmp.reason.contains("3")
                && cmp.reason.contains("2"),
            "diagnostic should explain the dup-id collision: {}",
            cmp.reason
        );
    }

    /// v0.32 R8 P2: a run scorecard whose `rein_version` does not
    /// match the current binary's `CARGO_PKG_VERSION` is treated as
    /// stale.  This is the safety net for `target/eval-gates/<gate>-run.json`
    /// files that survive `cargo build` rebuilds — without this gate
    /// `rein doctor` could report Ship for a revision that hasn't been
    /// re-run.
    #[test]
    fn compare_scorecards_rejects_stale_run_version() {
        let baseline = make_scorecard(
            "recall",
            ScorecardKind::Baseline,
            vec![("a", true), ("b", true)],
        );
        let mut current =
            make_scorecard("recall", ScorecardKind::Run, vec![("a", true), ("b", true)]);
        // Pretend the run was generated by an older binary.
        current.rein_version = "0.0.0-stale-test".to_string();

        let cmp = compare_scorecards(
            "recall",
            Some(&baseline),
            Some(&current),
            DEFAULT_NOISE_FLOOR,
        );
        assert_eq!(cmp.status, ScorecardStatus::NoData);
        assert!(cmp.mcnemar.is_none());
        assert!(
            cmp.reason.contains("0.0.0-stale-test") && cmp.reason.contains("re-run"),
            "diagnostic should mention the stale version and recommend re-run: {}",
            cmp.reason
        );
    }

    #[test]
    fn compare_scorecards_rejects_duplicate_fixture_ids_in_current() {
        let baseline = make_scorecard(
            "recall",
            ScorecardKind::Baseline,
            vec![("a", true), ("b", true)],
        );
        let current = make_scorecard(
            "recall",
            ScorecardKind::Run,
            vec![("a", true), ("a", false), ("b", true)],
        );
        let cmp = compare_scorecards(
            "recall",
            Some(&baseline),
            Some(&current),
            DEFAULT_NOISE_FLOOR,
        );
        assert_eq!(cmp.status, ScorecardStatus::NoData);
        assert!(cmp.mcnemar.is_none());
        assert!(
            cmp.reason.contains("current") && cmp.reason.contains("duplicate"),
            "diagnostic should explain the dup-id collision: {}",
            cmp.reason
        );
    }

    #[test]
    fn compare_scorecards_classifies_ship_when_ci_within_noise() {
        // Synthetic 100-fixture scorecards where current >= baseline.
        // Make treatment strictly better: 80 a (both hit), 0 b, 15 c (only
        // treatment hits), 5 d. diff_point = (15 - 0) / 100 = 0.15, ci_lower
        // safely above -noise_floor.
        let mut baseline_fx = Vec::new();
        let mut current_fx = Vec::new();
        for i in 0..80 {
            let id = format!("a{i}");
            baseline_fx.push((id.clone(), true));
            current_fx.push((id, true));
        }
        for i in 0..15 {
            let id = format!("c{i}");
            baseline_fx.push((id.clone(), false));
            current_fx.push((id, true));
        }
        for i in 0..5 {
            let id = format!("d{i}");
            baseline_fx.push((id.clone(), false));
            current_fx.push((id, false));
        }
        let baseline = make_scorecard(
            "recall",
            ScorecardKind::Baseline,
            baseline_fx.iter().map(|(s, h)| (s.as_str(), *h)).collect(),
        );
        let current = make_scorecard(
            "recall",
            ScorecardKind::Run,
            current_fx.iter().map(|(s, h)| (s.as_str(), *h)).collect(),
        );
        let cmp = compare_scorecards(
            "recall",
            Some(&baseline),
            Some(&current),
            DEFAULT_NOISE_FLOOR,
        );
        assert_eq!(cmp.status, ScorecardStatus::Ship, "reason: {}", cmp.reason);
        let result = cmp.mcnemar.expect("McNemar should run");
        assert!(result.ci_lower >= -DEFAULT_NOISE_FLOOR);
    }

    #[test]
    fn compare_scorecards_classifies_bail_when_ci_below_minus_noise() {
        // Synthetic case where current << baseline: 40 a, 50 b (baseline-only
        // hit), 0 c, 10 d. diff_point = (0 - 50)/100 = -0.5, far below
        // -noise_floor. ci_upper should also be << -noise_floor.
        let mut baseline_fx = Vec::new();
        let mut current_fx = Vec::new();
        for i in 0..40 {
            let id = format!("a{i}");
            baseline_fx.push((id.clone(), true));
            current_fx.push((id, true));
        }
        for i in 0..50 {
            let id = format!("b{i}");
            baseline_fx.push((id.clone(), true));
            current_fx.push((id, false));
        }
        for i in 0..10 {
            let id = format!("d{i}");
            baseline_fx.push((id.clone(), false));
            current_fx.push((id, false));
        }
        let baseline = make_scorecard(
            "recall",
            ScorecardKind::Baseline,
            baseline_fx.iter().map(|(s, h)| (s.as_str(), *h)).collect(),
        );
        let current = make_scorecard(
            "recall",
            ScorecardKind::Run,
            current_fx.iter().map(|(s, h)| (s.as_str(), *h)).collect(),
        );
        let cmp = compare_scorecards(
            "recall",
            Some(&baseline),
            Some(&current),
            DEFAULT_NOISE_FLOOR,
        );
        assert_eq!(cmp.status, ScorecardStatus::Bail, "reason: {}", cmp.reason);
        let result = cmp.mcnemar.expect("McNemar should run");
        assert!(result.ci_upper <= -DEFAULT_NOISE_FLOOR);
    }

    #[test]
    fn compare_scorecards_no_data_when_schema_mismatch() {
        let baseline = make_scorecard(
            "recall",
            ScorecardKind::Baseline,
            vec![("a", true), ("b", false)],
        );
        let mut current = make_scorecard(
            "recall",
            ScorecardKind::Run,
            vec![("a", true), ("b", false)],
        );
        current.schema_version = SCORECARD_SCHEMA_VERSION + 1;
        let cmp = compare_scorecards(
            "recall",
            Some(&baseline),
            Some(&current),
            DEFAULT_NOISE_FLOOR,
        );
        assert_eq!(cmp.status, ScorecardStatus::NoData);
        assert!(
            cmp.reason.contains("schema_version"),
            "unexpected reason: {}",
            cmp.reason
        );
        assert!(cmp.mcnemar.is_none());
    }

    #[test]
    fn write_then_read_scorecard_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nested/eval-gates/recall-run.json");
        let original = make_scorecard(
            "recall",
            ScorecardKind::Run,
            vec![("a", true), ("b", false), ("c", true)],
        );
        write_scorecard(&path, &original).expect("write succeeds");
        assert!(path.exists(), "file should exist after write");

        let loaded = read_scorecard(&path).expect("read succeeds");
        assert_eq!(loaded.schema_version, original.schema_version);
        assert_eq!(loaded.gate_name, original.gate_name);
        assert_eq!(loaded.kind, original.kind);
        assert_eq!(loaded.created_at, original.created_at);
        assert_eq!(loaded.rein_version, original.rein_version);
        assert_eq!(loaded.fixture_count, original.fixture_count);
        assert!((loaded.score - original.score).abs() < 1e-12);
        assert_eq!(loaded.per_fixture.len(), original.per_fixture.len());
        for (a, b) in loaded.per_fixture.iter().zip(original.per_fixture.iter()) {
            assert_eq!(a.fixture_id, b.fixture_id);
            assert_eq!(a.hit, b.hit);
        }
    }
}
