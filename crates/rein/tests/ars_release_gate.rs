use rein::config::ReinConfig;
use rein::ops::ars_release_gate::{evaluate_ars_acceleration_release_gate, ReleaseGateInput};
use rein::store::adaptive::{
    AdaptiveState, JudgeCalibrationState, LearnedShadowFusionEntry, ShadowFusionWeightEntry,
};
use rein::store::ars_parameter_policy::{
    ArsParameterPolicy, ArsParameterPolicyLoad, ArsParameterPolicyLoadStatus,
    ArsParameterPolicyMode,
};

fn missing_policy() -> ArsParameterPolicyLoad {
    ArsParameterPolicyLoad {
        policy: ArsParameterPolicy::default(),
        status: ArsParameterPolicyLoadStatus::Missing,
        error: None,
    }
}

fn loaded_canary_policy(source_adaptive_version: u64) -> ArsParameterPolicyLoad {
    ArsParameterPolicyLoad {
        policy: ArsParameterPolicy {
            revision: 3,
            mode: ArsParameterPolicyMode::Canary,
            disabled_reason: None,
            source_adaptive_version,
            runtime_adoption_weight: 1.0,
            last_event_id: 42,
            last_updated: "2026-05-01T00:00:00Z".to_string(),
            ..ArsParameterPolicy::default()
        },
        status: ArsParameterPolicyLoadStatus::Loaded,
        error: None,
    }
}

fn eligible_state(sample_count: usize) -> AdaptiveState {
    let mut state = AdaptiveState {
        version: 9,
        ..AdaptiveState::default()
    };
    state.learned_shadow_fusion.insert(
        "global".to_string(),
        LearnedShadowFusionEntry {
            weights: ShadowFusionWeightEntry {
                bm25: 0.20,
                vec: 0.25,
                kg: 0.20,
                episode: 0.15,
                support: 0.10,
                diversity: 0.10,
            },
            sample_count,
            last_updated: "2026-05-01T00:00:00Z".to_string(),
        },
    );
    state
}

fn ready_shadow_status() -> serde_json::Value {
    serde_json::json!({
        "enabled": true,
        "shadow_only": false,
        "status": "ready",
        "eligible_samples": 12,
        "min_samples": 10,
        "global": {
            "sample_count": 12,
            "last_updated": "2026-05-01T00:00:00Z",
            "weights": {
                "bm25": 0.20,
                "vec": 0.25,
                "kg": 0.20,
                "episode": 0.15,
                "support": 0.10,
                "diversity": 0.10
            }
        },
        "by_query_type": [],
        "by_cluster": []
    })
}

fn input<'a>(
    config: &'a ReinConfig,
    state: &'a AdaptiveState,
    policy: &'a ArsParameterPolicyLoad,
    shadow_fusion_status: &'a serde_json::Value,
) -> ReleaseGateInput<'a> {
    ReleaseGateInput {
        config,
        state,
        policy,
        shadow_fusion_status,
    }
}

#[test]
fn default_config_blocks_canary_and_default_on_without_mutating_defaults() {
    let config = ReinConfig::default();
    let state = AdaptiveState::default();
    let policy = missing_policy();
    let shadow_fusion_status = serde_json::json!({
        "enabled": true,
        "shadow_only": false,
        "status": "insufficient_samples",
        "eligible_samples": 0,
        "min_samples": 10,
        "global": null,
        "by_query_type": [],
        "by_cluster": []
    });

    let report = evaluate_ars_acceleration_release_gate(input(
        &config,
        &state,
        &policy,
        &shadow_fusion_status,
    ));

    assert!(!report.canary.allowed);
    assert!(report
        .canary
        .blockers
        .iter()
        .any(|b| b == "ars_parameter_policy_missing"));
    assert!(!report.default_on.allowed);
    assert!(report
        .default_on
        .blockers
        .iter()
        .any(|b| b == "default_on_requires_release_evaluation"));

    assert!(config.ars.acceleration.enabled);
    assert!(!config.ars.acceleration.shadow_only);
}

#[test]
fn canary_is_allowed_only_when_config_policy_and_adaptive_evidence_are_healthy() {
    let mut config = ReinConfig::default();
    config.ars.acceleration.enabled = true;
    config.ars.acceleration.shadow_only = false;
    config.adaptive.min_samples_alpha = 10;
    let state = eligible_state(12);
    let policy = loaded_canary_policy(state.version);
    let shadow_fusion_status = ready_shadow_status();

    let report = evaluate_ars_acceleration_release_gate(input(
        &config,
        &state,
        &policy,
        &shadow_fusion_status,
    ));

    assert!(report.canary.allowed);
    assert!(report.canary.blockers.is_empty());
    assert_eq!(report.signals.policy_mode.as_deref(), Some("canary"));
    assert!(report.signals.policy_allows_runtime);
    assert_eq!(report.signals.learned_shadow_fusion_buckets, 1);

    assert!(!report.default_on.allowed);
    assert!(report
        .default_on
        .blockers
        .iter()
        .any(|b| b == "default_on_requires_release_evaluation"));
}

#[test]
fn drift_alert_blocks_canary_even_with_loaded_canary_policy() {
    let mut config = ReinConfig::default();
    config.ars.acceleration.enabled = true;
    config.ars.acceleration.shadow_only = false;
    config.adaptive.min_samples_alpha = 10;
    let mut state = eligible_state(12);
    state.judge_calibration_state = Some(JudgeCalibrationState {
        judge_drift_alert_concept: 1,
        ..JudgeCalibrationState::default()
    });
    let policy = loaded_canary_policy(state.version);
    let shadow_fusion_status = ready_shadow_status();

    let report = evaluate_ars_acceleration_release_gate(input(
        &config,
        &state,
        &policy,
        &shadow_fusion_status,
    ));

    assert!(!report.canary.allowed);
    assert!(report
        .canary
        .blockers
        .iter()
        .any(|b| b == "judge_drift_alert"));
    assert!(report.signals.judge_drift_alert);
}
