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
    /// Rollout-side weight in `[0, 1]`; this makes canary adoption gradual
    /// instead of a binary jump from static priors to dynamic parameters.
    pub runtime_adoption_weight: f64,
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
    let runtime_adoption_weight = clamp01(inputs.runtime_adoption_weight);
    if runtime_adoption_weight <= f64::EPSILON {
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
        * runtime_adoption_weight
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
    let static_prior_raw = finite_or(static_prior, min);
    let trust = dynamic_trust(inputs);
    if trust <= f64::EPSILON {
        return static_prior_raw;
    }

    let static_prior = static_prior_raw.clamp(min, max);
    let learned_value = finite_or(learned_value, static_prior).clamp(min, max);
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
    if calibration.judge_drift_alert > 0
        || calibration.judge_drift_alert_synthesis > 0
        || calibration.judge_drift_alert_concept > 0
    {
        return 0.0;
    }

    let mut parts = Vec::new();
    if calibration.recent_pairs_synthesis.len() >= 10 {
        parts.push(clamp01(calibration.kappa));
    }
    if calibration.recent_pairs_runtime_vs_offline.len() >= 30 {
        parts.push(clamp01(calibration.runtime_vs_offline_kappa));
    }
    if calibration.recent_pairs_runtime_vs_offline_synthesis.len() >= 30 {
        parts.push(clamp01(calibration.runtime_vs_offline_kappa_synthesis));
    }
    if calibration.recent_pairs_runtime_vs_offline_concept.len() >= 30 {
        parts.push(clamp01(calibration.runtime_vs_offline_kappa_concept));
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
    effective_judge_weight_decay_rate_with_previous(
        static_rate,
        calibration,
        adoption_weight_from_canary(production_canary),
        None,
    )
}

pub fn effective_judge_weight_decay_rate_with_previous(
    static_rate: f64,
    calibration: Option<&crate::store::adaptive::JudgeCalibrationState>,
    runtime_adoption_weight: f64,
    previous_effective: Option<f64>,
) -> f64 {
    let production_canary = runtime_adoption_weight > f64::EPSILON;
    let pair_count = calibration
        .map(|cal| {
            cal.recent_pairs_synthesis
                .len()
                .saturating_add(cal.recent_pairs_concept.len())
                .saturating_add(cal.recent_pairs_runtime_vs_offline.len())
                .saturating_add(cal.recent_pairs_runtime_vs_offline_synthesis.len())
                .saturating_add(cal.recent_pairs_runtime_vs_offline_concept.len())
        })
        .unwrap_or(0) as u64;
    let drift_alert = calibration
        .map(|cal| {
            cal.judge_drift_alert > 0
                || cal.judge_drift_alert_synthesis > 0
                || cal.judge_drift_alert_concept > 0
        })
        .unwrap_or(false);
    if production_canary && drift_alert {
        return 0.0;
    }
    effective_scalar(
        static_rate,
        llm_feedback_reliability(calibration),
        previous_effective,
        bounds01(0.10),
        TrustInputs {
            enabled: true,
            production_canary,
            runtime_adoption_weight,
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

pub fn effective_judge_sample_rate(
    static_rate: f64,
    calibration: Option<&crate::store::adaptive::JudgeCalibrationState>,
    production_canary: bool,
    cold_start: bool,
) -> f64 {
    effective_judge_sample_rate_with_previous(
        static_rate,
        calibration,
        adoption_weight_from_canary(production_canary),
        cold_start,
        None,
    )
}

pub fn effective_judge_sample_rate_with_previous(
    static_rate: f64,
    calibration: Option<&crate::store::adaptive::JudgeCalibrationState>,
    runtime_adoption_weight: f64,
    cold_start: bool,
    previous_effective: Option<f64>,
) -> f64 {
    let max_rate = if cold_start { 1.0 } else { 0.5 };
    let static_rate = finite_non_negative(static_rate).min(max_rate);
    let production_canary = runtime_adoption_weight > f64::EPSILON;
    if !production_canary || static_rate <= f64::EPSILON {
        return static_rate;
    }

    let drift_alert = calibration
        .map(|cal| {
            cal.judge_drift_alert > 0
                || cal.judge_drift_alert_synthesis > 0
                || cal.judge_drift_alert_concept > 0
        })
        .unwrap_or(false);
    if drift_alert {
        return 0.0;
    }
    let reliability = llm_feedback_reliability(calibration);
    let pair_count = calibration_pair_count(calibration);
    let learned_rate = if cold_start {
        static_rate * (1.0 - 0.5 * reliability)
    } else {
        static_rate * (1.0 - reliability)
    };
    effective_scalar(
        static_rate,
        learned_rate,
        previous_effective,
        ScalarBounds {
            min: 0.0,
            max: max_rate,
            max_step: if cold_start { 0.10 } else { 0.05 },
        },
        TrustInputs {
            enabled: true,
            production_canary,
            runtime_adoption_weight,
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

pub fn effective_cold_start_n(
    static_n: u64,
    calibration: Option<&crate::store::adaptive::JudgeCalibrationState>,
    production_canary: bool,
) -> u64 {
    effective_cold_start_n_with_previous(
        static_n,
        calibration,
        adoption_weight_from_canary(production_canary),
        None,
    )
}

pub fn effective_cold_start_n_with_previous(
    static_n: u64,
    calibration: Option<&crate::store::adaptive::JudgeCalibrationState>,
    runtime_adoption_weight: f64,
    previous_effective: Option<f64>,
) -> u64 {
    let production_canary = runtime_adoption_weight > f64::EPSILON;
    let drift_alert = calibration
        .map(|cal| {
            cal.judge_drift_alert > 0
                || cal.judge_drift_alert_synthesis > 0
                || cal.judge_drift_alert_concept > 0
        })
        .unwrap_or(false);
    if !production_canary || drift_alert || static_n == 0 {
        return static_n;
    }

    let reliability = llm_feedback_reliability(calibration);
    let learned_n = (static_n as f64 * (1.0 - 0.7 * reliability)).round();
    effective_scalar(
        static_n as f64,
        learned_n,
        previous_effective,
        ScalarBounds {
            min: (static_n as f64).min(3.0),
            max: 50.0,
            max_step: 2.0,
        },
        TrustInputs {
            enabled: true,
            production_canary,
            runtime_adoption_weight,
            human_count: calibration_pair_count(calibration),
            llm_count: 0,
            llm_reliability: 0.0,
            calibration: 1.0,
            stability: 1.0,
            drift_alert,
            prior_strength: 20.0,
            max_trust: 0.75,
        },
    )
    .round()
    .clamp((static_n as f64).min(3.0), 50.0) as u64
}

pub fn effective_useful_rate_threshold(
    static_threshold: f64,
    observed_useful_rate: f64,
    human_count: u64,
    llm_count: u64,
    calibration: Option<&crate::store::adaptive::JudgeCalibrationState>,
    production_canary: bool,
) -> f64 {
    effective_useful_rate_threshold_with_previous(
        static_threshold,
        observed_useful_rate,
        human_count,
        llm_count,
        calibration,
        adoption_weight_from_canary(production_canary),
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn effective_useful_rate_threshold_with_previous(
    static_threshold: f64,
    observed_useful_rate: f64,
    human_count: u64,
    llm_count: u64,
    calibration: Option<&crate::store::adaptive::JudgeCalibrationState>,
    runtime_adoption_weight: f64,
    previous_effective: Option<f64>,
) -> f64 {
    let static_threshold = clamp01(finite_or(static_threshold, 0.0));
    let production_canary = runtime_adoption_weight > f64::EPSILON;
    let drift_alert = calibration
        .map(|cal| {
            cal.judge_drift_alert > 0
                || cal.judge_drift_alert_synthesis > 0
                || cal.judge_drift_alert_concept > 0
        })
        .unwrap_or(false);
    if !production_canary || drift_alert {
        return static_threshold;
    }

    effective_scalar(
        static_threshold,
        observed_useful_rate.clamp(0.35, 0.75),
        previous_effective,
        bounds01(0.05),
        TrustInputs {
            enabled: true,
            production_canary,
            runtime_adoption_weight,
            human_count,
            llm_count,
            llm_reliability: llm_feedback_reliability(calibration),
            calibration: 1.0,
            stability: 1.0,
            drift_alert,
            prior_strength: 20.0,
            max_trust: 0.75,
        },
    )
}

pub fn parameter_policy_runtime_adoption_weight(
    conn: &rusqlite::Connection,
    config: &crate::config::ReinConfig,
    state: &crate::store::adaptive::AdaptiveState,
) -> f64 {
    if !config.adaptive.enabled
        || !config.ars.acceleration.enabled
        || config.ars.acceleration.shadow_only
    {
        return 0.0;
    }
    let loaded = crate::store::ars_parameter_policy::load_parameter_policy(conn);
    if !matches!(
        loaded.status,
        crate::store::ars_parameter_policy::ArsParameterPolicyLoadStatus::Loaded
    ) {
        return 0.0;
    }
    loaded.policy.runtime_adoption_weight(state.version)
}

pub fn parameter_policy_runtime_adoption_weight_for(
    conn: &rusqlite::Connection,
    config: &crate::config::ReinConfig,
    state: &crate::store::adaptive::AdaptiveState,
    key: &str,
) -> f64 {
    if !config.adaptive.enabled
        || !config.ars.acceleration.enabled
        || config.ars.acceleration.shadow_only
    {
        return 0.0;
    }
    let loaded = crate::store::ars_parameter_policy::load_parameter_policy(conn);
    if !matches!(
        loaded.status,
        crate::store::ars_parameter_policy::ArsParameterPolicyLoadStatus::Loaded
    ) {
        return 0.0;
    }
    loaded
        .policy
        .runtime_adoption_weight_for(state.version, key)
}

pub fn parameter_policy_recall_fusion_runtime_adoption_weight(
    conn: &rusqlite::Connection,
    config: &crate::config::ReinConfig,
    state: &crate::store::adaptive::AdaptiveState,
    query_type: &str,
    cluster_id: Option<u32>,
) -> f64 {
    if !config.adaptive.enabled
        || !config.ars.acceleration.enabled
        || config.ars.acceleration.shadow_only
    {
        return 0.0;
    }
    let loaded = crate::store::ars_parameter_policy::load_parameter_policy(conn);
    if !matches!(
        loaded.status,
        crate::store::ars_parameter_policy::ArsParameterPolicyLoadStatus::Loaded
    ) {
        return 0.0;
    }
    let policy = loaded.policy;
    let query_key = crate::store::adaptive::AdaptiveState::bucket_key(query_type, None);
    let mut keys = Vec::new();
    if let Some(cluster) = cluster_id {
        let cluster_key =
            crate::store::adaptive::AdaptiveState::bucket_key(query_type, Some(cluster));
        keys.push(format!("recall_fusion:{cluster_key}"));
    }
    keys.push(format!("recall_fusion:{query_key}"));
    keys.push("recall_fusion:global".to_string());

    for key in keys {
        if policy.adoption_weights.contains_key(&key) {
            return policy.runtime_adoption_weight_for(state.version, &key);
        }
    }
    policy.runtime_adoption_weight(state.version)
}

pub fn parameter_policy_allows_runtime(
    conn: &rusqlite::Connection,
    config: &crate::config::ReinConfig,
    state: &crate::store::adaptive::AdaptiveState,
) -> bool {
    parameter_policy_runtime_adoption_weight(conn, config, state) > f64::EPSILON
}

fn adoption_weight_from_canary(production_canary: bool) -> f64 {
    if production_canary {
        1.0
    } else {
        0.0
    }
}

fn calibration_pair_count(
    calibration: Option<&crate::store::adaptive::JudgeCalibrationState>,
) -> u64 {
    calibration
        .map(|cal| {
            cal.recent_pairs_synthesis
                .len()
                .saturating_add(cal.recent_pairs_concept.len())
                .saturating_add(cal.recent_pairs_runtime_vs_offline.len())
                .saturating_add(cal.recent_pairs_runtime_vs_offline_synthesis.len())
                .saturating_add(cal.recent_pairs_runtime_vs_offline_concept.len())
                as u64
        })
        .unwrap_or(0)
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
            runtime_adoption_weight: 0.0,
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
    fn scalar_effective_value_preserves_out_of_bounds_static_when_trust_is_zero() {
        let inputs = TrustInputs {
            enabled: true,
            production_canary: false,
            runtime_adoption_weight: 0.0,
            human_count: 100,
            llm_count: 0,
            llm_reliability: 0.0,
            calibration: 1.0,
            stability: 1.0,
            drift_alert: false,
            prior_strength: 20.0,
            max_trust: 1.0,
        };

        assert_eq!(
            effective_scalar(
                0.0,
                0.9,
                None,
                ScalarBounds {
                    min: 0.1,
                    max: 1.0,
                    max_step: 0.1
                },
                inputs
            ),
            0.0
        );
    }

    #[test]
    fn trust_discounts_llm_evidence_and_calibration() {
        let inputs = TrustInputs {
            enabled: true,
            production_canary: true,
            runtime_adoption_weight: 0.25,
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
        let expected = (expected_evidence / (expected_evidence + 20.0)) * 0.5 * 0.8 * 0.25;
        assert!((trust - expected).abs() < 1e-12);
    }

    #[test]
    fn runtime_adoption_weight_slides_dynamic_trust_between_static_and_dynamic() {
        let base = TrustInputs {
            enabled: true,
            production_canary: true,
            runtime_adoption_weight: 0.0,
            human_count: 100,
            llm_count: 0,
            llm_reliability: 0.0,
            calibration: 1.0,
            stability: 1.0,
            drift_alert: false,
            prior_strength: 0.0,
            max_trust: 1.0,
        };

        assert_eq!(dynamic_trust(base), 0.0);
        assert_eq!(effective_scalar(0.2, 0.8, None, bounds01(1.0), base), 0.2);

        let quarter = TrustInputs {
            runtime_adoption_weight: 0.25,
            ..base
        };
        assert_eq!(dynamic_trust(quarter), 0.25);
        assert!((effective_scalar(0.2, 0.8, None, bounds01(1.0), quarter) - 0.35).abs() < 1e-12);
    }

    #[test]
    fn scalar_effective_value_blends_and_respects_step_cap() {
        let inputs = TrustInputs {
            enabled: true,
            production_canary: true,
            runtime_adoption_weight: 1.0,
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
            runtime_adoption_weight: 1.0,
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

    #[test]
    fn judge_sample_rate_reduces_warm_rate_when_reliable() {
        let mut calibration = crate::store::adaptive::JudgeCalibrationState {
            kappa: 0.8,
            runtime_vs_offline_kappa: 0.8,
            ..Default::default()
        };
        for idx in 0..40 {
            calibration
                .recent_pairs_runtime_vs_offline
                .push_back((true, true, idx));
        }

        let effective = effective_judge_sample_rate(0.2, Some(&calibration), true, false);

        assert!(effective < 0.2);
        assert!(effective >= 0.05);
    }

    #[test]
    fn judge_sample_rate_returns_static_without_policy_canary() {
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
            effective_judge_sample_rate(1.0, Some(&calibration), false, true),
            1.0
        );
        assert_eq!(
            effective_judge_sample_rate(0.0, Some(&calibration), false, true),
            0.0
        );
        assert_eq!(
            effective_judge_sample_rate(0.0, Some(&calibration), true, false),
            0.0
        );
    }

    #[test]
    fn cold_start_n_returns_static_without_policy_canary() {
        let mut calibration = crate::store::adaptive::JudgeCalibrationState {
            kappa: 0.9,
            ..Default::default()
        };
        for idx in 0..12 {
            calibration
                .recent_pairs_synthesis
                .push_back((true, true, idx));
        }

        assert_eq!(effective_cold_start_n(1, Some(&calibration), false), 1);
        assert_eq!(effective_cold_start_n(0, Some(&calibration), true), 0);
    }

    #[test]
    fn cold_start_n_moves_down_with_reliable_feedback() {
        let mut calibration = crate::store::adaptive::JudgeCalibrationState {
            kappa: 0.9,
            ..Default::default()
        };
        for idx in 0..20 {
            calibration
                .recent_pairs_synthesis
                .push_back((true, true, idx));
        }

        let effective = effective_cold_start_n(10, Some(&calibration), true);

        assert!(effective < 10);
        assert!(effective >= 3);
    }

    #[test]
    fn useful_rate_threshold_blends_toward_observed_rate() {
        let mut calibration = crate::store::adaptive::JudgeCalibrationState {
            kappa: 0.8,
            ..Default::default()
        };
        for idx in 0..12 {
            calibration
                .recent_pairs_synthesis
                .push_back((true, true, idx));
        }

        let effective =
            effective_useful_rate_threshold(0.5, 0.35, 20, 40, Some(&calibration), true);

        assert!(effective < 0.5);
        assert!(effective >= 0.35);
    }

    #[test]
    fn parameter_policy_runtime_adoption_weight_for_uses_scoped_weight_with_global_fallback() {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute_batch("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
            .unwrap();
        let mut policy = crate::store::ars_parameter_policy::ArsParameterPolicy {
            revision: 1,
            mode: crate::store::ars_parameter_policy::ArsParameterPolicyMode::Canary,
            disabled_reason: None,
            source_adaptive_version: 7,
            runtime_adoption_weight: 0.25,
            ..Default::default()
        };
        policy
            .adoption_weights
            .insert("recall_fusion:semantic".to_string(), 0.60);
        crate::store::ars_parameter_policy::save_parameter_policy_cas(&conn, &policy, 0).unwrap();

        let mut config = crate::config::ReinConfig::default();
        config.ars.acceleration.enabled = true;
        config.ars.acceleration.shadow_only = false;
        let state = crate::store::adaptive::AdaptiveState {
            version: 7,
            ..Default::default()
        };

        assert_eq!(
            parameter_policy_runtime_adoption_weight_for(
                &conn,
                &config,
                &state,
                "recall_fusion:semantic"
            ),
            0.60
        );
        assert_eq!(
            parameter_policy_runtime_adoption_weight_for(
                &conn,
                &config,
                &state,
                "recall_fusion:unknown"
            ),
            0.25
        );

        config.ars.acceleration.shadow_only = true;
        assert_eq!(
            parameter_policy_runtime_adoption_weight_for(
                &conn,
                &config,
                &state,
                "recall_fusion:semantic"
            ),
            0.0
        );
    }

    #[test]
    fn parameter_policy_recall_fusion_weight_falls_back_cluster_query_global() {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute_batch("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
            .unwrap();
        let mut policy = crate::store::ars_parameter_policy::ArsParameterPolicy {
            revision: 1,
            mode: crate::store::ars_parameter_policy::ArsParameterPolicyMode::Canary,
            disabled_reason: None,
            source_adaptive_version: 7,
            runtime_adoption_weight: 0.25,
            ..Default::default()
        };
        policy
            .adoption_weights
            .insert("recall_fusion:global".to_string(), 0.40);
        policy
            .adoption_weights
            .insert("recall_fusion:semantic".to_string(), 0.60);
        policy
            .adoption_weights
            .insert("recall_fusion:semantic:7".to_string(), 0.80);
        crate::store::ars_parameter_policy::save_parameter_policy_cas(&conn, &policy, 0).unwrap();

        let mut config = crate::config::ReinConfig::default();
        config.ars.acceleration.enabled = true;
        config.ars.acceleration.shadow_only = false;
        let state = crate::store::adaptive::AdaptiveState {
            version: 7,
            ..Default::default()
        };

        assert_eq!(
            parameter_policy_recall_fusion_runtime_adoption_weight(
                &conn,
                &config,
                &state,
                "Semantic",
                Some(7)
            ),
            0.80
        );
        assert_eq!(
            parameter_policy_recall_fusion_runtime_adoption_weight(
                &conn,
                &config,
                &state,
                "Semantic",
                Some(8)
            ),
            0.60
        );
        assert_eq!(
            parameter_policy_recall_fusion_runtime_adoption_weight(
                &conn, &config, &state, "Episodic", None
            ),
            0.40
        );
    }
}
