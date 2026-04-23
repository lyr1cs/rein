//! Scorecards (persisted paired-outcome runs) and the ship-decision policy
//! that consumes them.
//!
//! A `Scorecard` is the JSON artifact emitted by `rein-eval resummerize
//! baseline` / `run` and consumed by `rein-eval resummerize compare`. The
//! ship decision itself is a pure function of the overall McNemar result plus
//! per-category stats and two knobs that come from the harness (`noise_floor`,
//! `avg_length_ratio`).

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::mcnemar::{McNemarResult, PairedOutcome};

/// Persisted record of a full evaluation run (baseline or treatment). Written
/// to JSON by the rein-eval CLI and re-loaded for the `compare` subcommand.
///
/// `outcomes` carries `baseline_hit` / `treatment_hit` per case — for a pure
/// baseline run the `treatment_*` fields may be placeholder/unset until the
/// paired comparison is performed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scorecard {
    pub fixtures_dir: String,
    pub iterations: u32,
    pub timestamp: DateTime<Utc>,
    pub outcomes: Vec<PairedOutcome>,
    #[serde(default)]
    pub per_category: HashMap<String, CategoryStats>,
    /// Optional `case_id -> category` map populated by `baseline` and `run`
    /// from the fixture's `category` field. `compare` uses this to group
    /// joined paired outcomes when computing per-category McNemar (the
    /// pre-fixture-category fallback parsed `case_id` prefix-before-colon,
    /// which doesn't fit fixtures whose case_ids use underscores).
    /// `#[serde(default)]` keeps older scorecards readable.
    #[serde(default)]
    pub category_map: HashMap<String, String>,
    /// Version of the `KeywordOverlapHitChecker` used to produce the hit
    /// outcomes. `#[serde(default)]` → `0` means "pre-version-tracking
    /// scorecard" (the first cut of v0.23 eval, before the CJK tokenizer
    /// fix added `HIT_CHECKER_VERSION`). `compare` bails if baseline and
    /// treatment scorecards were produced under different versions, so
    /// operators can't unknowingly run McNemar across incompatible
    /// scoring methodologies. Post-fix audit M-2.
    #[serde(default)]
    pub hit_checker_version: u32,
}

/// Per-category aggregate: hit rates, mean lengths, and a category-scoped
/// McNemar result. Used by `decide_ship` for the regression bail-out check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryStats {
    pub n: u32,
    pub baseline_hit_rate: f64,
    pub treatment_hit_rate: f64,
    pub avg_baseline_length: f64,
    pub avg_treatment_length: f64,
    pub mcnemar: McNemarResult,
}

/// The ship-or-bail decision emitted by the `compare` subcommand.
#[derive(Debug, Clone, Serialize)]
pub enum ShipDecision {
    Ship {
        reason: ShipReason,
        overall: McNemarResult,
    },
    BailOut {
        reason: String,
        overall: McNemarResult,
    },
}

/// Why the treatment was accepted.
///
/// `Superior` fires only when we have a statistically significant win on
/// hit-rate. `NonInferiorAndShorter` fires when we can't distinguish hit-rates
/// (within the baseline noise floor) but the treatment is materially shorter.
#[derive(Debug, Clone, Serialize)]
pub enum ShipReason {
    Superior {
        p_value: f64,
    },
    NonInferiorAndShorter {
        avg_length_reduction_pct: f64,
        ci_lower: f64,
        noise_floor: f64,
    },
}

/// Apply the v0.23 ship policy to an overall McNemar result and per-category
/// stats. Rule ordering:
///   1. Superior — `p_value < 0.05 AND diff_point > 0`
///   2. Category regression — any category significantly worse → BailOut
///   3. NonInferiorAndShorter — CI lower bound above `-noise_floor` AND
///      treatment length is <90% of baseline
///   4. Otherwise — BailOut
///
/// `noise_floor` (δ₀) comes from baseline variance runs; it bounds how much
/// hit-rate we're willing to trade for a shorter context. `avg_length_ratio`
/// is mean(treatment_length)/mean(baseline_length) over the full set.
///
/// NaN inputs are handled conservatively: NaN-comparisons are all false, so
/// a NaN `ci_lower` naturally lands in BailOut.
pub fn decide_ship(
    overall: &McNemarResult,
    per_category: &HashMap<String, CategoryStats>,
    noise_floor: f64,
    avg_length_ratio: f64,
) -> ShipDecision {
    // 1. Superior.
    if overall.p_value < 0.05 && overall.diff_point > 0.0 {
        return ShipDecision::Ship {
            reason: ShipReason::Superior {
                p_value: overall.p_value,
            },
            overall: overall.clone(),
        };
    }

    // 2. Category regression — outranks non-inferiority ship.
    for (cat, stats) in per_category {
        if stats.mcnemar.p_value < 0.05 && stats.mcnemar.diff_point < 0.0 {
            return ShipDecision::BailOut {
                reason: format!("category {cat} regressed"),
                overall: overall.clone(),
            };
        }
    }

    // 3. Non-inferior and shorter.
    if overall.ci_lower > -noise_floor && avg_length_ratio < 0.9 {
        let avg_length_reduction_pct = (1.0 - avg_length_ratio) * 100.0;
        return ShipDecision::Ship {
            reason: ShipReason::NonInferiorAndShorter {
                avg_length_reduction_pct,
                ci_lower: overall.ci_lower,
                noise_floor,
            },
            overall: overall.clone(),
        };
    }

    // 4. Neither superior nor non-inferior-and-shorter.
    ShipDecision::BailOut {
        reason: "neither superior nor non-inferior-and-shorter".into(),
        overall: overall.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::mcnemar::McNemarResult;

    fn m(p: f64, diff: f64, ci_lower: f64, ci_upper: f64) -> McNemarResult {
        McNemarResult {
            n: 100,
            a: 50,
            b: 10,
            c: 10,
            d: 30,
            chi_squared: 0.0,
            p_value: p,
            used_exact: false,
            diff_point: diff,
            ci_lower,
            ci_upper,
        }
    }

    fn cat_stats(mcnemar: McNemarResult) -> CategoryStats {
        CategoryStats {
            n: mcnemar.n,
            baseline_hit_rate: 0.5,
            treatment_hit_rate: 0.5,
            avg_baseline_length: 1000.0,
            avg_treatment_length: 500.0,
            mcnemar,
        }
    }

    #[test]
    fn superior_decision_path() {
        // Significant win: p < 0.05, diff > 0.
        let overall = m(0.001, 0.08, 0.02, 0.14);
        let per_category = HashMap::new();
        let d = decide_ship(&overall, &per_category, 0.03, 0.95);
        match d {
            ShipDecision::Ship {
                reason: ShipReason::Superior { p_value },
                ..
            } => {
                assert!((p_value - 0.001).abs() < 1e-9);
            }
            other => panic!("expected Superior, got {other:?}"),
        }
    }

    #[test]
    fn non_inferior_and_shorter_decision_path() {
        // No significant win, but CI lower bound is above -noise_floor and
        // treatment length is materially shorter.
        let overall = m(0.3, -0.005, -0.02, 0.01);
        let per_category = HashMap::new();
        let noise_floor = 0.03;
        let length_ratio = 0.6;
        let d = decide_ship(&overall, &per_category, noise_floor, length_ratio);
        match d {
            ShipDecision::Ship {
                reason:
                    ShipReason::NonInferiorAndShorter {
                        avg_length_reduction_pct,
                        ci_lower,
                        noise_floor: nf,
                    },
                ..
            } => {
                assert!((avg_length_reduction_pct - 40.0).abs() < 1e-9);
                assert!((ci_lower - (-0.02)).abs() < 1e-9);
                assert!((nf - 0.03).abs() < 1e-9);
            }
            other => panic!("expected NonInferiorAndShorter, got {other:?}"),
        }
    }

    #[test]
    fn category_regression_bailout() {
        // Overall looks fine for non-inferiority (CI > -noise, length ratio low),
        // but one category is significantly worse → BailOut outranks the ship.
        let overall = m(0.3, -0.005, -0.02, 0.01);
        let mut per_category = HashMap::new();
        per_category.insert(
            "single_session".to_string(),
            cat_stats(m(0.01, -0.1, -0.15, -0.05)),
        );
        per_category.insert("multi_session".to_string(), cat_stats(m(0.4, 0.01, -0.03, 0.05)));
        let d = decide_ship(&overall, &per_category, 0.03, 0.6);
        match d {
            ShipDecision::BailOut { reason, .. } => {
                assert!(
                    reason.contains("single_session"),
                    "reason = {reason:?}, expected mention of single_session"
                );
                assert!(reason.contains("regressed"), "reason = {reason:?}");
            }
            other => panic!("expected BailOut for category regression, got {other:?}"),
        }
    }

    #[test]
    fn bailout_when_neither_condition_met() {
        // Not significantly better AND not materially shorter.
        let overall = m(0.3, -0.005, -0.02, 0.01);
        let per_category = HashMap::new();
        let d = decide_ship(&overall, &per_category, 0.03, 0.95);
        match d {
            ShipDecision::BailOut { reason, .. } => {
                assert_eq!(reason, "neither superior nor non-inferior-and-shorter");
            }
            other => panic!("expected generic BailOut, got {other:?}"),
        }
    }

    #[test]
    fn superior_beats_category_regression_when_overall_wins() {
        // If the overall is a clear win, we ship even if a category looks iffy —
        // rule 1 (Superior) outranks rule 2 (category regression).
        let overall = m(0.001, 0.08, 0.02, 0.14);
        let mut per_category = HashMap::new();
        per_category.insert("x".to_string(), cat_stats(m(0.01, -0.1, -0.15, -0.05)));
        let d = decide_ship(&overall, &per_category, 0.03, 0.95);
        assert!(matches!(
            d,
            ShipDecision::Ship {
                reason: ShipReason::Superior { .. },
                ..
            }
        ));
    }
}
