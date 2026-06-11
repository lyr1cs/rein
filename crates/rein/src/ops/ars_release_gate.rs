//! Read-only ARS acceleration release gate report.
//!
//! This module evaluates existing config, adaptive-state, parameter-policy,
//! and adaptive-status signals. It does not refresh policy rows, commit
//! offsets, or change runtime defaults.

use serde::Serialize;
use std::collections::BTreeMap;

use crate::config::ReinConfig;
use crate::store::adaptive::AdaptiveState;
use crate::store::ars_parameter_policy::{
    ArsParameterPolicyLoad, ArsParameterPolicyLoadStatus, ArsParameterPolicyMode,
};
use crate::store::SqliteStore;

pub const ARS_ACCELERATION_RELEASE_GATE_SCHEMA_VERSION: u32 = 1;

pub struct ReleaseGateInput<'a> {
    pub config: &'a ReinConfig,
    pub state: &'a AdaptiveState,
    pub policy: &'a ArsParameterPolicyLoad,
    pub shadow_fusion_status: &'a serde_json::Value,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ReleaseGateDecision {
    pub allowed: bool,
    pub blockers: Vec<String>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ReleaseGateSignals {
    pub adaptive_enabled: bool,
    pub ars_acceleration_enabled: bool,
    pub ars_acceleration_shadow_only: bool,
    pub adaptive_version: u64,
    pub min_samples_alpha: usize,
    pub learned_shadow_fusion_buckets: usize,
    pub eligible_learned_shadow_fusion_buckets: usize,
    pub policy_status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub policy_mode: Option<String>,
    pub policy_revision: u64,
    pub policy_source_adaptive_version: u64,
    pub policy_allows_runtime: bool,
    pub runtime_adoption_weight: f64,
    pub runtime_adoption_weights: BTreeMap<String, f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub policy_error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shadow_fusion_status: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shadow_fusion_eligible_samples: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shadow_fusion_min_samples: Option<u64>,
    pub judge_drift_alert: bool,
    pub judge_calibration_pairs: usize,
    pub doctor_ars_parameter_policy_level: String,
    pub doctor_ars_parameter_policy_message: String,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ArsAccelerationReleaseGateReport {
    pub schema_version: u32,
    pub purpose: String,
    pub signals: ReleaseGateSignals,
    pub canary: ReleaseGateDecision,
    pub default_on: ReleaseGateDecision,
}

pub fn ars_acceleration_release_gate_report(
    store: &SqliteStore,
    config: &ReinConfig,
) -> ArsAccelerationReleaseGateReport {
    let state = AdaptiveState::restore_snapshot(store.conn()).unwrap_or_default();
    let policy = crate::store::ars_parameter_policy::load_parameter_policy(store.conn());
    let shadow_fusion_status = crate::ops::adaptive::shadow_fusion_status(store, config);

    evaluate_ars_acceleration_release_gate(ReleaseGateInput {
        config,
        state: &state,
        policy: &policy,
        shadow_fusion_status: &shadow_fusion_status,
    })
}

pub fn evaluate_ars_acceleration_release_gate(
    input: ReleaseGateInput<'_>,
) -> ArsAccelerationReleaseGateReport {
    let config = input.config;
    let state = input.state;
    let policy = input.policy;
    // codex R11 P2: same effective floor as the runtime read gates
    // (get_alpha / get_shadow_fusion_weights / runtime_adoption_target) —
    // with min_samples_alpha configured below 10, the gate must not report
    // a bucket as eligible that runtime serving would refuse.
    let min_samples = config.adaptive.min_samples_alpha.max(10);
    let eligible_learned_shadow_fusion_buckets = state
        .learned_shadow_fusion
        .values()
        .filter(|entry| entry.sample_count >= min_samples)
        .count();
    let policy_row_adoption_weight =
        if matches!(policy.status, ArsParameterPolicyLoadStatus::Loaded) {
            policy.policy.runtime_adoption_weight(state.version)
        } else {
            0.0
        };
    let runtime_adoption_weight = if config.adaptive.enabled
        && config.ars.acceleration.enabled
        && !config.ars.acceleration.shadow_only
    {
        policy_row_adoption_weight
    } else {
        0.0
    };
    let runtime_adoption_weights = if config.adaptive.enabled
        && config.ars.acceleration.enabled
        && !config.ars.acceleration.shadow_only
        && matches!(policy.status, ArsParameterPolicyLoadStatus::Loaded)
    {
        policy
            .policy
            .adoption_weights
            .keys()
            .map(|key| {
                (
                    key.clone(),
                    policy
                        .policy
                        .runtime_adoption_weight_for(state.version, key),
                )
            })
            .collect()
    } else {
        BTreeMap::new()
    };
    let policy_allows_runtime = runtime_adoption_weight > f64::EPSILON;
    let judge_drift_alert = state
        .judge_calibration_state
        .as_ref()
        .map(|calibration| {
            calibration.judge_drift_alert > 0
                || calibration.judge_drift_alert_synthesis > 0
                || calibration.judge_drift_alert_concept > 0
        })
        .unwrap_or(false);
    let judge_calibration_pairs = state
        .judge_calibration_state
        .as_ref()
        .map(|calibration| {
            calibration
                .recent_pairs_synthesis
                .len()
                .saturating_add(calibration.recent_pairs_concept.len())
                .saturating_add(calibration.recent_pairs_runtime_vs_offline.len())
                .saturating_add(calibration.recent_pairs_runtime_vs_offline_synthesis.len())
                .saturating_add(calibration.recent_pairs_runtime_vs_offline_concept.len())
        })
        .unwrap_or(0);

    let live_allowed = config.adaptive.enabled
        && config.ars.acceleration.enabled
        && !config.ars.acceleration.shadow_only
        && policy_allows_runtime;
    let (doctor_level, doctor_message) =
        doctor_policy_signal(policy, state, live_allowed, runtime_adoption_weight);
    let shadow_status = input
        .shadow_fusion_status
        .get("status")
        .and_then(serde_json::Value::as_str)
        .map(ToOwned::to_owned);

    let signals = ReleaseGateSignals {
        adaptive_enabled: config.adaptive.enabled,
        ars_acceleration_enabled: config.ars.acceleration.enabled,
        ars_acceleration_shadow_only: config.ars.acceleration.shadow_only,
        adaptive_version: state.version,
        min_samples_alpha: min_samples,
        learned_shadow_fusion_buckets: state.learned_shadow_fusion.len(),
        eligible_learned_shadow_fusion_buckets,
        policy_status: policy_status_name(&policy.status).to_string(),
        policy_mode: Some(policy_mode_name(policy.policy.mode).to_string()),
        policy_revision: policy.policy.revision,
        policy_source_adaptive_version: policy.policy.source_adaptive_version,
        policy_allows_runtime,
        runtime_adoption_weight,
        runtime_adoption_weights,
        policy_error: policy.error.clone(),
        shadow_fusion_status: shadow_status.clone(),
        shadow_fusion_eligible_samples: json_u64(input.shadow_fusion_status, "eligible_samples"),
        shadow_fusion_min_samples: json_u64(input.shadow_fusion_status, "min_samples"),
        judge_drift_alert,
        judge_calibration_pairs,
        doctor_ars_parameter_policy_level: doctor_level.to_string(),
        doctor_ars_parameter_policy_message: doctor_message,
    };

    let mut canary_blockers = Vec::new();
    if !config.adaptive.enabled {
        canary_blockers.push("adaptive_disabled".to_string());
    }
    if !config.ars.acceleration.enabled {
        canary_blockers.push("ars_acceleration_disabled".to_string());
    }
    if config.ars.acceleration.shadow_only {
        canary_blockers.push("ars_acceleration_shadow_only".to_string());
    }
    match policy.status {
        ArsParameterPolicyLoadStatus::Missing => {
            canary_blockers.push("ars_parameter_policy_missing".to_string());
        }
        ArsParameterPolicyLoadStatus::Corrupt
        | ArsParameterPolicyLoadStatus::UnsupportedSchema
        | ArsParameterPolicyLoadStatus::StorageError => {
            // R5 P2 fix (2026-05-04): all three "unhealthy" states
            // collapse to the same blocker — the canary cannot
            // promote against a policy row we can't safely interpret,
            // regardless of whether the cause is parse error,
            // future-schema downgrade, or transient I/O failure.
            canary_blockers.push("ars_parameter_policy_unhealthy".to_string());
        }
        ArsParameterPolicyLoadStatus::Loaded => {
            if !matches!(policy.policy.mode, ArsParameterPolicyMode::Canary) {
                canary_blockers.push("ars_parameter_policy_not_canary".to_string());
            }
            if policy.policy.source_adaptive_version > state.version {
                canary_blockers.push("ars_parameter_policy_not_current".to_string());
            }
            if policy_row_adoption_weight <= f64::EPSILON {
                canary_blockers.push("ars_parameter_policy_adoption_weight_zero".to_string());
            }
        }
    }
    if eligible_learned_shadow_fusion_buckets == 0 {
        canary_blockers.push("insufficient_learned_shadow_fusion".to_string());
    }
    if judge_drift_alert {
        canary_blockers.push("judge_drift_alert".to_string());
    }
    // v1.2 (#A12 activation prerequisite — 2026-06-02 algorithm-directions
    // recommendation 2): shadow-fusion replay readiness is a BLOCKER, not a
    // warning. The volume blockers above only prove data exists
    // (buckets/samples filled); "ready" means the counterfactual replay
    // actually computed a learnable quality report over those samples. A
    // pure volume ramp must not promote a canary whose quality machinery
    // hasn't produced a verdict ("don't ramp on volume alone").
    if shadow_status.as_deref() != Some("ready") {
        canary_blockers.push(format!(
            "shadow_fusion_replay_not_ready:{}",
            shadow_status.as_deref().unwrap_or("unknown")
        ));
    }

    let canary = ReleaseGateDecision {
        allowed: canary_blockers.is_empty(),
        blockers: canary_blockers,
        warnings: Vec::new(),
    };

    let mut default_on_blockers = vec!["default_on_requires_release_evaluation".to_string()];
    if !canary.allowed {
        default_on_blockers.push("canary_not_allowed".to_string());
    }

    ArsAccelerationReleaseGateReport {
        schema_version: ARS_ACCELERATION_RELEASE_GATE_SCHEMA_VERSION,
        purpose: "read_only_release_eval_gate_for_ars_acceleration".to_string(),
        signals,
        canary,
        default_on: ReleaseGateDecision {
            allowed: false,
            blockers: default_on_blockers,
            warnings: vec![
                "default_on_gate_is_report_only_and_does_not_change_runtime_defaults".to_string(),
            ],
        },
    }
}

fn doctor_policy_signal(
    policy: &ArsParameterPolicyLoad,
    state: &AdaptiveState,
    live_allowed: bool,
    runtime_adoption_weight: f64,
) -> (&'static str, String) {
    match policy.status {
        ArsParameterPolicyLoadStatus::Missing => (
            "ok",
            "missing policy row; dynamic ARS parameters disabled".to_string(),
        ),
        ArsParameterPolicyLoadStatus::Corrupt
        | ArsParameterPolicyLoadStatus::UnsupportedSchema
        | ArsParameterPolicyLoadStatus::StorageError => (
            "warn",
            format!(
                "policy row unhealthy; dynamic ARS parameters disabled ({})",
                policy
                    .error
                    .clone()
                    .unwrap_or_else(|| "unknown error".to_string())
            ),
        ),
        ArsParameterPolicyLoadStatus::Loaded => {
            let message = format!(
                "mode={:?} revision={} source_adaptive_version={} current_adaptive_version={} live_allowed={} runtime_adoption_weight={:.3}",
                policy.policy.mode,
                policy.policy.revision,
                policy.policy.source_adaptive_version,
                state.version,
                live_allowed,
                runtime_adoption_weight,
            );
            if matches!(policy.policy.mode, ArsParameterPolicyMode::Canary) && !live_allowed {
                ("warn", message)
            } else {
                ("ok", message)
            }
        }
    }
}

fn policy_status_name(status: &ArsParameterPolicyLoadStatus) -> &'static str {
    match status {
        ArsParameterPolicyLoadStatus::Missing => "missing",
        ArsParameterPolicyLoadStatus::Loaded => "loaded",
        ArsParameterPolicyLoadStatus::Corrupt => "corrupt",
        ArsParameterPolicyLoadStatus::UnsupportedSchema => "unsupported_schema",
        ArsParameterPolicyLoadStatus::StorageError => "storage_error",
    }
}

fn policy_mode_name(mode: ArsParameterPolicyMode) -> &'static str {
    match mode {
        ArsParameterPolicyMode::Disabled => "disabled",
        ArsParameterPolicyMode::Shadow => "shadow",
        ArsParameterPolicyMode::Canary => "canary",
    }
}

fn json_u64(value: &serde_json::Value, key: &str) -> Option<u64> {
    value.get(key).and_then(serde_json::Value::as_u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn loaded_canary_policy(
        weight: f64,
    ) -> crate::store::ars_parameter_policy::ArsParameterPolicyLoad {
        crate::store::ars_parameter_policy::ArsParameterPolicyLoad {
            policy: crate::store::ars_parameter_policy::ArsParameterPolicy {
                revision: 1,
                mode: ArsParameterPolicyMode::Canary,
                disabled_reason: None,
                source_adaptive_version: 7,
                runtime_adoption_weight: weight,
                last_updated: "2026-05-01T00:00:00Z".to_string(),
                ..Default::default()
            },
            status: ArsParameterPolicyLoadStatus::Loaded,
            error: None,
        }
    }

    fn eligible_state() -> AdaptiveState {
        let mut state = AdaptiveState {
            version: 7,
            ..Default::default()
        };
        state.learned_shadow_fusion.insert(
            "global".to_string(),
            crate::store::adaptive::LearnedShadowFusionEntry {
                weights: crate::store::adaptive::ShadowFusionWeightEntry {
                    bm25: 0.45,
                    vec: 0.45,
                    kg: 0.04,
                    episode: 0.03,
                    support: 0.02,
                    diversity: 0.01,
                },
                sample_count: 12,
                last_updated: "2026-05-01T00:00:00Z".to_string(),
            },
        );
        state
    }

    #[test]
    fn release_gate_reports_runtime_adoption_weight() {
        let mut config = ReinConfig::default();
        config.adaptive.enabled = true;
        config.ars.acceleration.enabled = true;
        config.ars.acceleration.shadow_only = false;
        config.adaptive.min_samples_alpha = 10;
        let state = eligible_state();
        let mut policy = loaded_canary_policy(0.25);
        policy
            .policy
            .adoption_weights
            .insert("recall_fusion:semantic".to_string(), 0.40);
        policy
            .policy
            .adoption_weights
            .insert("synthesis_gate".to_string(), 0.35);
        let report = evaluate_ars_acceleration_release_gate(ReleaseGateInput {
            config: &config,
            state: &state,
            policy: &policy,
            shadow_fusion_status: &serde_json::json!({
                "status": "ready",
                "eligible_samples": 12,
                "min_samples": 10
            }),
        });

        assert_eq!(report.signals.runtime_adoption_weight, 0.25);
        assert_eq!(
            report.signals.runtime_adoption_weights["recall_fusion:semantic"],
            0.40
        );
        assert_eq!(
            report.signals.runtime_adoption_weights["synthesis_gate"],
            0.35
        );
        assert!(report.signals.policy_allows_runtime);
        assert!(report.canary.allowed);
    }

    #[test]
    fn release_gate_blocks_zero_runtime_adoption_weight() {
        let mut config = ReinConfig::default();
        config.adaptive.enabled = true;
        config.ars.acceleration.enabled = true;
        config.ars.acceleration.shadow_only = false;
        config.adaptive.min_samples_alpha = 10;
        let state = eligible_state();
        let policy = loaded_canary_policy(0.0);
        let report = evaluate_ars_acceleration_release_gate(ReleaseGateInput {
            config: &config,
            state: &state,
            policy: &policy,
            shadow_fusion_status: &serde_json::json!({ "status": "ready" }),
        });

        assert!(!report.canary.allowed);
        assert!(report
            .canary
            .blockers
            .contains(&"ars_parameter_policy_adoption_weight_zero".to_string()));
    }
}
