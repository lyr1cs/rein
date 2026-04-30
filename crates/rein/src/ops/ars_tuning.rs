//! ARS dynamic parameter policy.

/// Evidence and rollout gates used to decide how far an ARS parameter may move
/// from its static bootstrap value toward a learned value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrustInputs {
    /// Master adaptive parameter flag.
    pub enabled: bool,
    /// Runtime adoption gate. Shadow-only acceleration still learns, but does
    /// not affect live decisions.
    pub production_canary: bool,
    /// Human feedback count for the bucket.
    pub human_count: u64,
    /// LLM feedback count for the bucket.
    pub llm_count: u64,
    /// Relative trust in an LLM feedback event versus a human event.
    pub llm_reliability: f64,
    /// Calibration quality in `[0, 1]`.
    pub calibration: f64,
    /// Recent value stability in `[0, 1]`.
    pub stability: f64,
    /// Hard stop when judge drift is detected.
    pub drift_alert: bool,
    /// Pseudo-count anchoring the static prior.
    pub prior_strength: f64,
    /// Per-parameter trust cap in `[0, 1]`.
    pub max_trust: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalarBounds {
    pub min: f64,
    pub max: f64,
    pub max_step: f64,
}

pub fn bounds01(max_step: f64) -> ScalarBounds {
    ScalarBounds {
        min: 0.0,
        max: 1.0,
        max_step,
    }
}

pub fn dynamic_trust(inputs: TrustInputs) -> f64 {
    if !inputs.enabled || !inputs.production_canary || inputs.drift_alert {
        return 0.0;
    }

    let llm_reliability = clamp01(inputs.llm_reliability);
    let evidence = inputs.human_count as f64 + inputs.llm_count as f64 * llm_reliability;
    if evidence <= 0.0 || !evidence.is_finite() {
        return 0.0;
    }

    let prior_strength = finite_non_negative(inputs.prior_strength);
    let evidence_ratio = if prior_strength <= 0.0 {
        1.0
    } else {
        evidence / (evidence + prior_strength)
    };
    evidence_ratio
        * clamp01(inputs.calibration)
        * clamp01(inputs.stability)
        * clamp01(inputs.max_trust)
}

pub fn effective_scalar(
    static_prior: f64,
    learned_value: f64,
    previous_effective: Option<f64>,
    bounds: ScalarBounds,
    inputs: TrustInputs,
) -> f64 {
    let min = finite_or(bounds.min, 0.0);
    let max = finite_or(bounds.max, min).max(min);
    let static_prior = finite_or(static_prior, min).clamp(min, max);
    let learned_value = finite_or(learned_value, static_prior).clamp(min, max);
    let trust = dynamic_trust(inputs);
    let blended = static_prior.mul_add(1.0 - trust, learned_value * trust);
    let stepped = match previous_effective {
        Some(previous) if bounds.max_step.is_finite() && bounds.max_step >= 0.0 => {
            let previous = previous.clamp(min, max);
            blended.clamp(previous - bounds.max_step, previous + bounds.max_step)
        }
        _ => blended,
    };

    stepped.clamp(min, max)
}

pub fn effective_simplex(
    static_prior: [f64; 6],
    learned_value: [f64; 6],
    inputs: TrustInputs,
) -> [f64; 6] {
    let static_prior = normalize_simplex(static_prior).unwrap_or([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let learned_value = normalize_simplex(learned_value).unwrap_or(static_prior);
    let trust = dynamic_trust(inputs);
    let mut blended = [0.0; 6];
    for i in 0..6 {
        blended[i] = static_prior[i].mul_add(1.0 - trust, learned_value[i] * trust);
    }
    normalize_simplex(blended).unwrap_or(static_prior)
}

pub fn llm_feedback_reliability(
    calibration: Option<&crate::store::adaptive::JudgeCalibrationState>,
) -> f64 {
    let Some(calibration) = calibration else {
        return 0.0;
    };
    if calibration.judge_drift_alert > 0 {
        return 0.0;
    }

    let mut parts = Vec::new();
    if calibration.recent_pairs_synthesis.len() >= 10 {
        parts.push(clamp01(calibration.kappa));
    }
    if calibration.recent_pairs_runtime_vs_offline.len() >= 30 {
        parts.push(clamp01(calibration.runtime_vs_offline_kappa));
    }
    if parts.is_empty() {
        0.0
    } else {
        parts.iter().sum::<f64>() / parts.len() as f64
    }
}

pub fn effective_judge_weight_decay_rate(
    static_rate: f64,
    calibration: Option<&crate::store::adaptive::JudgeCalibrationState>,
    production_canary: bool,
) -> f64 {
    let pair_count = calibration
        .map(|cal| {
            cal.recent_pairs_synthesis
                .len()
                .saturating_add(cal.recent_pairs_concept.len())
                .saturating_add(cal.recent_pairs_runtime_vs_offline.len())
        })
        .unwrap_or(0) as u64;
    let drift_alert = calibration
        .map(|cal| cal.judge_drift_alert > 0)
        .unwrap_or(false);
    if production_canary && drift_alert {
        return 0.0;
    }
    effective_scalar(
        static_rate,
        llm_feedback_reliability(calibration),
        None,
        bounds01(0.10),
        TrustInputs {
            enabled: true,
            production_canary,
            human_count: pair_count,
            llm_count: 0,
            llm_reliability: 0.0,
            calibration: 1.0,
            stability: 1.0,
            drift_alert,
            prior_strength: 30.0,
            max_trust: 0.75,
        },
    )
}

fn normalize_simplex(values: [f64; 6]) -> Option<[f64; 6]> {
    let mut sanitized = [0.0; 6];
    for (idx, value) in values.into_iter().enumerate() {
        if value.is_finite() && value > 0.0 {
            sanitized[idx] = value;
        }
    }

    let sum: f64 = sanitized.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return None;
    }

    for value in &mut sanitized {
        *value /= sum;
    }
    Some(sanitized)
}

fn clamp01(value: f64) -> f64 {
    finite_or(value, 0.0).clamp(0.0, 1.0)
}

fn finite_non_negative(value: f64) -> f64 {
    finite_or(value, 0.0).max(0.0)
}

fn finite_or(value: f64, fallback: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trust_is_zero_when_policy_is_not_allowed_to_affect_runtime() {
        let inputs = TrustInputs {
            enabled: true,
            production_canary: false,
            human_count: 100,
            llm_count: 100,
            llm_reliability: 0.8,
            calibration: 0.9,
            stability: 1.0,
            drift_alert: false,
            prior_strength: 20.0,
            max_trust: 0.8,
        };

        assert_eq!(dynamic_trust(inputs), 0.0);
        assert_eq!(effective_scalar(0.5, 0.9, None, bounds01(0.1), inputs), 0.5);
    }

    #[test]
    fn trust_discounts_llm_evidence_and_calibration() {
        let inputs = TrustInputs {
            enabled: true,
            production_canary: true,
            human_count: 10,
            llm_count: 90,
            llm_reliability: 0.25,
            calibration: 0.5,
            stability: 1.0,
            drift_alert: false,
            prior_strength: 20.0,
            max_trust: 0.8,
        };

        let trust = dynamic_trust(inputs);
        let expected_evidence = 10.0 + 90.0 * 0.25;
        let expected = (expected_evidence / (expected_evidence + 20.0)) * 0.5 * 0.8;
        assert!((trust - expected).abs() < 1e-12);
    }

    #[test]
    fn scalar_effective_value_blends_and_respects_step_cap() {
        let inputs = TrustInputs {
            enabled: true,
            production_canary: true,
            human_count: 80,
            llm_count: 0,
            llm_reliability: 0.0,
            calibration: 1.0,
            stability: 1.0,
            drift_alert: false,
            prior_strength: 20.0,
            max_trust: 1.0,
        };

        let effective = effective_scalar(0.50, 0.90, Some(0.50), bounds01(0.05), inputs);

        assert_eq!(effective, 0.55);
    }

    #[test]
    fn simplex_effective_weights_blend_and_normalize() {
        let inputs = TrustInputs {
            enabled: true,
            production_canary: true,
            human_count: 80,
            llm_count: 0,
            llm_reliability: 0.0,
            calibration: 1.0,
            stability: 1.0,
            drift_alert: false,
            prior_strength: 20.0,
            max_trust: 1.0,
        };
        let static_prior = [0.45, 0.35, 0.08, 0.04, 0.05, 0.03];
        let learned = [0.10, 0.20, 0.35, 0.15, 0.15, 0.05];

        let effective = effective_simplex(static_prior, learned, inputs);

        let sum: f64 = effective.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
        assert!(effective[2] > static_prior[2]);
        assert!(effective[0] < static_prior[0]);
    }

    #[test]
    fn judge_weight_decay_rate_moves_toward_calibrated_reliability() {
        let mut calibration = crate::store::adaptive::JudgeCalibrationState {
            kappa: 0.8,
            runtime_vs_offline_kappa: 0.6,
            ..Default::default()
        };
        for idx in 0..12 {
            calibration
                .recent_pairs_synthesis
                .push_back((true, true, idx));
        }
        for idx in 0..35 {
            calibration
                .recent_pairs_runtime_vs_offline
                .push_back((true, true, idx));
        }
        let effective = effective_judge_weight_decay_rate(0.3, Some(&calibration), true);

        assert!(effective > 0.3);
        assert!(effective < 0.7);
    }

    #[test]
    fn judge_weight_decay_rate_falls_back_static_without_canary() {
        let mut calibration = crate::store::adaptive::JudgeCalibrationState {
            kappa: 0.9,
            ..Default::default()
        };
        for idx in 0..12 {
            calibration
                .recent_pairs_synthesis
                .push_back((true, true, idx));
        }

        assert_eq!(
            effective_judge_weight_decay_rate(0.3, Some(&calibration), false),
            0.3
        );
    }

    #[test]
    fn drift_alert_zeroes_llm_reliability() {
        let mut calibration = crate::store::adaptive::JudgeCalibrationState {
            kappa: 0.9,
            runtime_vs_offline_kappa: 0.9,
            judge_drift_alert: 1,
            ..Default::default()
        };
        for idx in 0..35 {
            calibration
                .recent_pairs_runtime_vs_offline
                .push_back((true, true, idx));
        }

        assert_eq!(llm_feedback_reliability(Some(&calibration)), 0.0);
        assert!(effective_judge_weight_decay_rate(0.3, Some(&calibration), true) < 0.3);
    }
}
