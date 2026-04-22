//! Paired non-inferiority test (McNemar's test) for baseline vs. treatment
//! hit-rate comparison.
//!
//! This module implements McNemar's test with:
//! - Continuity-corrected chi-squared for `b + c >= 25`
//! - Exact (two-sided) binomial p-value for small discordant counts
//! - Wald 95% CI on the hit-rate difference `(c - b) / n`
//!
//! All statistical math is inline: the Abramowitz & Stegun 7.1.26 polynomial
//! approximation for `erf` gives a standard normal CDF, which is squared into
//! a chi-squared(1) CDF via the identity `chi2_cdf(x, 1) = 2*Phi(sqrt(x)) - 1`.
//!
//! ## Phase 1 caveat
//!
//! The Wald CI is a normal-approximation interval. For small `n` a better
//! interval exists (e.g. Newcombe's paired score interval), but it is not
//! implemented here — reviewers should treat CIs at low `n` as rough.
//!
//! No external stats crate is used. See `eval/scorecard.rs` for the ship-decision
//! logic that consumes `McNemarResult`.

use serde::{Deserialize, Serialize};

/// A single paired outcome: one fixture case evaluated against both
/// `baseline` and `treatment` pipelines.
///
/// `baseline_length` / `treatment_length` are in characters (or whatever unit
/// the hit-checker's caller chose — the eval harness currently counts chars of
/// the surfaced context). They drive the `avg_length_ratio` computation in
/// `decide_ship`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairedOutcome {
    pub case_id: String,
    pub baseline_hit: bool,
    pub treatment_hit: bool,
    pub baseline_length: usize,
    pub treatment_length: usize,
}

/// Summary of a paired-comparison McNemar test.
///
/// The `(a, b, c, d)` quadruplet is the classical paired 2×2 contingency table:
/// `a` = both hit, `b` = baseline hit / treatment missed,
/// `c` = treatment hit / baseline missed, `d` = both missed.
///
/// `Deserialize` is included so scorecards round-trip cleanly through JSON
/// (they embed `McNemarResult` via `CategoryStats`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McNemarResult {
    /// Total paired cases.
    pub n: u32,
    /// Both hit.
    pub a: u32,
    /// Baseline hit, treatment missed.
    pub b: u32,
    /// Treatment hit, baseline missed.
    pub c: u32,
    /// Both missed.
    pub d: u32,
    /// Continuity-corrected chi-squared statistic. Only meaningful when
    /// `b + c >= 25`; set to 0.0 when the exact-binomial path is used.
    pub chi_squared: f64,
    /// Two-sided p-value (from chi-squared or exact binomial).
    pub p_value: f64,
    /// True when the exact-binomial path was used (`b + c < 25`).
    pub used_exact: bool,
    /// Point estimate of the hit-rate difference: `(c - b) / n`.
    pub diff_point: f64,
    /// Lower bound of the 95% Wald CI on `diff_point`.
    pub ci_lower: f64,
    /// Upper bound of the 95% Wald CI on `diff_point`.
    pub ci_upper: f64,
}

/// Run McNemar's test on a set of paired outcomes.
///
/// Uses the continuity-corrected chi-squared statistic when `b + c >= 25`,
/// and falls back to an exact two-sided binomial p-value otherwise. An empty
/// outcomes vector yields `n=0` with all zeros and a p-value of 1.0 (no
/// evidence of a difference).
pub fn mcnemar(outcomes: &[PairedOutcome]) -> McNemarResult {
    let n = outcomes.len() as u32;

    // Tally the 2×2 contingency table.
    let mut a = 0u32;
    let mut b = 0u32;
    let mut c = 0u32;
    let mut d = 0u32;
    for o in outcomes {
        match (o.baseline_hit, o.treatment_hit) {
            (true, true) => a += 1,
            (true, false) => b += 1,
            (false, true) => c += 1,
            (false, false) => d += 1,
        }
    }

    // Empty input: return a neutral result (no discordant pairs → p=1.0).
    if n == 0 {
        return McNemarResult {
            n: 0,
            a: 0,
            b: 0,
            c: 0,
            d: 0,
            chi_squared: 0.0,
            p_value: 1.0,
            used_exact: true,
            diff_point: 0.0,
            ci_lower: 0.0,
            ci_upper: 0.0,
        };
    }

    let bc_sum = b + c;
    let (chi_squared, p_value, used_exact) = if bc_sum >= 25 {
        // Continuity-corrected chi-squared: (|b-c| - 1)^2 / (b+c).
        let diff = (b as f64 - c as f64).abs();
        let chi_sq = (diff - 1.0).powi(2) / (bc_sum as f64);
        let p = chi2_sf_df1(chi_sq);
        (chi_sq, p, false)
    } else {
        // Exact two-sided binomial: p = 2 * sum_{k=0}^{min(b,c)} C(bc,k) / 2^bc,
        // capped at 1.0. Handles b+c == 0 as p = 1.0 (sum starts at k=0 -> 1).
        let p = exact_two_sided_binomial(b, c);
        (0.0, p, true)
    };

    // Point estimate: (c - b) / n.
    let n_f = n as f64;
    let diff_point = (c as f64 - b as f64) / n_f;

    // Wald 95% CI: diff +/- 1.96 * sqrt((b+c - (b-c)^2/n) / n^2).
    // Guard the variance term against numerical negatives for degenerate inputs.
    let b_minus_c = b as f64 - c as f64;
    let raw_var = bc_sum as f64 - (b_minus_c * b_minus_c) / n_f;
    let var_bounded = raw_var.max(0.0);
    let se = (var_bounded / (n_f * n_f)).sqrt();
    let margin = 1.96 * se;
    let ci_lower = diff_point - margin;
    let ci_upper = diff_point + margin;

    McNemarResult {
        n,
        a,
        b,
        c,
        d,
        chi_squared,
        p_value,
        used_exact,
        diff_point,
        ci_lower,
        ci_upper,
    }
}

/// Survival function of chi-squared(df=1) at `x`: `1 - F(x)`.
///
/// Uses the identity: if `Z ~ N(0,1)` then `Z^2 ~ chi2(1)`, so
/// `F_chi2_1(x) = 2*Phi(sqrt(x)) - 1` and thus `1 - F = 2*(1 - Phi(sqrt(x)))`.
fn chi2_sf_df1(x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    let sqrt_x = x.sqrt();
    2.0 * (1.0 - phi(sqrt_x))
}

/// Standard normal CDF `Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))`.
fn phi(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Error function via Abramowitz & Stegun 7.1.26 polynomial approximation.
///
/// Maximum absolute error is approximately 1.5e-7, which is well below any
/// tolerance we care about for ship-decision thresholds.
fn erf(x: f64) -> f64 {
    // A&S 7.1.26 coefficients.
    const P: f64 = 0.327_591_1;
    const A1: f64 = 0.254_829_592;
    const A2: f64 = -0.284_496_736;
    const A3: f64 = 1.421_413_741;
    const A4: f64 = -1.453_152_027;
    const A5: f64 = 1.061_405_429;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + P * ax);
    let poly = ((((A5 * t + A4) * t + A3) * t + A2) * t + A1) * t;
    let y = 1.0 - poly * (-ax * ax).exp();
    sign * y
}

/// Exact two-sided binomial p-value for McNemar at small `b + c`.
///
/// Computes `p = 2 * sum_{k=0}^{min(b,c)} C(bc, k) / 2^bc`, capped at 1.0.
/// Uses iterative multiplication/division to compute the binomial coefficient
/// in f64 without intermediate overflow.
fn exact_two_sided_binomial(b: u32, c: u32) -> f64 {
    let bc = b + c;
    if bc == 0 {
        // No discordant pairs — no evidence against equality.
        return 1.0;
    }
    let min_bc = b.min(c);
    let total = 2f64.powi(bc as i32);

    // Iteratively compute C(bc, k) for k = 0..=min_bc, summing along the way.
    let mut sum = 0.0f64;
    let mut coeff = 1.0f64;
    // k = 0 term
    sum += coeff;
    for k in 1..=min_bc {
        // C(bc, k) = C(bc, k-1) * (bc - k + 1) / k
        coeff *= (bc - k + 1) as f64 / k as f64;
        sum += coeff;
    }

    let p = 2.0 * sum / total;
    if p > 1.0 {
        1.0
    } else {
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outcome(id: &str, b: bool, t: bool) -> PairedOutcome {
        PairedOutcome {
            case_id: id.into(),
            baseline_hit: b,
            treatment_hit: t,
            baseline_length: 1000,
            treatment_length: 500,
        }
    }

    /// Build a paired outcomes vector from explicit `(a, b, c, d)` counts.
    fn from_counts(a: u32, b: u32, c: u32, d: u32) -> Vec<PairedOutcome> {
        let mut v = Vec::new();
        let mut i = 0u32;
        for _ in 0..a {
            v.push(outcome(&format!("a{i}"), true, true));
            i += 1;
        }
        for _ in 0..b {
            v.push(outcome(&format!("b{i}"), true, false));
            i += 1;
        }
        for _ in 0..c {
            v.push(outcome(&format!("c{i}"), false, true));
            i += 1;
        }
        for _ in 0..d {
            v.push(outcome(&format!("d{i}"), false, false));
            i += 1;
        }
        v
    }

    #[test]
    fn agresti_textbook_example() {
        // Agresti "Categorical Data Analysis" paired-approval example.
        // Baseline hits = 170, treatment hits = 173, n = 200.
        // a (both hit) = 158, b = 12, c = 15, d = 15.
        // => chi2 = (|12-15| - 1)^2 / 27 = 4/27 ~= 0.1481
        // => p ~= 0.7003
        let outcomes = from_counts(158, 12, 15, 15);
        let r = mcnemar(&outcomes);
        assert_eq!(r.n, 200);
        assert_eq!(r.a, 158);
        assert_eq!(r.b, 12);
        assert_eq!(r.c, 15);
        assert_eq!(r.d, 15);
        assert!(!r.used_exact, "b+c = 27 >= 25, should use chi-squared");
        assert!(
            (r.chi_squared - 0.148).abs() < 0.001,
            "chi^2 = {} expected ~0.148",
            r.chi_squared
        );
        assert!(
            (r.p_value - 0.7003).abs() < 0.005,
            "p = {} expected ~0.7003",
            r.p_value
        );
        // diff = (15 - 12)/200 = 0.015
        assert!((r.diff_point - 0.015).abs() < 1e-9);
        // CI contains zero (not significant).
        assert!(r.ci_lower < 0.0 && r.ci_upper > 0.0);
    }

    #[test]
    fn exact_binomial_small_discordant() {
        // b + c = 4, below the 25 threshold => exact binomial.
        // b=1, c=3: p = 2 * (C(4,0) + C(4,1)) / 2^4 = 2 * 5 / 16 = 0.625.
        let outcomes = from_counts(10, 1, 3, 10);
        let r = mcnemar(&outcomes);
        assert!(r.used_exact);
        assert_eq!(r.chi_squared, 0.0);
        assert!(
            (r.p_value - 0.625).abs() < 1e-9,
            "p = {} expected 0.625",
            r.p_value
        );
        // diff = (3-1)/24
        assert!((r.diff_point - (2.0 / 24.0)).abs() < 1e-9);
    }

    #[test]
    fn b_equals_c_gives_p_one() {
        // b = c => no evidence of difference, two-sided p should be 1.0.
        // Small discordant count → exact binomial, which caps 2*sum at 1.0.
        let outcomes = from_counts(5, 3, 3, 5);
        let r = mcnemar(&outcomes);
        assert!(r.used_exact);
        assert!(
            (r.p_value - 1.0).abs() < 1e-9,
            "expected p=1.0 when b==c, got {}",
            r.p_value
        );
        assert_eq!(r.diff_point, 0.0);
    }

    #[test]
    fn empty_outcomes_returns_zeros() {
        let r = mcnemar(&[]);
        assert_eq!(r.n, 0);
        assert_eq!(r.a, 0);
        assert_eq!(r.b, 0);
        assert_eq!(r.c, 0);
        assert_eq!(r.d, 0);
        assert_eq!(r.chi_squared, 0.0);
        assert_eq!(r.p_value, 1.0);
        assert!(r.used_exact);
        assert_eq!(r.diff_point, 0.0);
        assert_eq!(r.ci_lower, 0.0);
        assert_eq!(r.ci_upper, 0.0);
    }

    #[test]
    fn large_discordant_uses_chi_squared() {
        // b + c = 30 >= 25, should take the chi-squared path.
        let outcomes = from_counts(100, 10, 20, 100);
        let r = mcnemar(&outcomes);
        assert!(!r.used_exact);
        // (|10-20| - 1)^2 / 30 = 81/30 = 2.7
        assert!((r.chi_squared - 2.7).abs() < 1e-9);
        assert!(r.p_value > 0.0 && r.p_value < 1.0);
    }

    #[test]
    fn treatment_strongly_better_gives_small_p() {
        // Heavy treatment-favoring discordance: b=2, c=30.
        let outcomes = from_counts(50, 2, 30, 50);
        let r = mcnemar(&outcomes);
        assert!(!r.used_exact);
        assert!(
            r.p_value < 0.001,
            "expected tiny p for strong treatment win, got {}",
            r.p_value
        );
        assert!(r.diff_point > 0.0);
        // CI should be strictly positive at this signal strength.
        assert!(r.ci_lower > 0.0);
    }

    #[test]
    fn erf_sanity() {
        // A&S 7.1.26 has error <~ 1.5e-7. Spot-check known values.
        assert!((erf(0.0) - 0.0).abs() < 1e-7);
        assert!((erf(1.0) - 0.842_700_8).abs() < 1e-5);
        assert!((erf(-1.0) + 0.842_700_8).abs() < 1e-5);
        // Phi(0) == 0.5
        assert!((phi(0.0) - 0.5).abs() < 1e-7);
        // Phi(1.96) ~= 0.975
        assert!((phi(1.96) - 0.975).abs() < 1e-3);
    }

    #[test]
    fn exact_binomial_zero_discordant() {
        // b = c = 0 — purely concordant table. No evidence of difference.
        let outcomes = from_counts(50, 0, 0, 50);
        let r = mcnemar(&outcomes);
        assert!(r.used_exact);
        assert!((r.p_value - 1.0).abs() < 1e-9);
        assert_eq!(r.diff_point, 0.0);
        // Variance guard: b+c - (b-c)^2/n = 0, sqrt(0) = 0, CI = [0, 0].
        assert_eq!(r.ci_lower, 0.0);
        assert_eq!(r.ci_upper, 0.0);
    }
}
