//! Read-only ARS acceleration release gate report.
//!
//! This module evaluates existing config, adaptive-state, parameter-policy,
//! and adaptive-status signals. It does not refresh policy rows, commit
//! offsets, or change runtime defaults.

use serde::Serialize;

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

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
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

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
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
    let min_samples = config.adaptive.min_samples_alpha;
    let eligible_learned_shadow_fusion_buckets = state
        .learned_shadow_fusion
        .values()
        .filter(|entry| entry.sample_count >= min_samples)
        .count();
    let policy_allows_runtime = matches!(policy.status, ArsParameterPolicyLoadStatus::Loaded)
        && policy.policy.allows_runtime_adoption(state.version);
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
    let (doctor_level, doctor_message) = doctor_policy_signal(policy, state, live_allowed);
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
        ArsParameterPolicyLoadStatus::Corrupt | ArsParameterPolicyLoadStatus::StorageError => {
            canary_blockers.push("ars_parameter_policy_unhealthy".to_string());
        }
        ArsParameterPolicyLoadStatus::Loaded => {
            if !matches!(policy.policy.mode, ArsParameterPolicyMode::Canary) {
                canary_blockers.push("ars_parameter_policy_not_canary".to_string());
            }
            if !policy_allows_runtime {
                canary_blockers.push("ars_parameter_policy_not_current".to_string());
            }
        }
    }
    if eligible_learned_shadow_fusion_buckets == 0 {
        canary_blockers.push("insufficient_learned_shadow_fusion".to_string());
    }
    if judge_drift_alert {
        canary_blockers.push("judge_drift_alert".to_string());
    }

    let mut canary_warnings = Vec::new();
    if shadow_status.as_deref() != Some("ready") {
        canary_warnings.push(format!(
            "shadow_fusion_replay_not_ready:{}",
            shadow_status.as_deref().unwrap_or("unknown")
        ));
    }

    let canary = ReleaseGateDecision {
        allowed: canary_blockers.is_empty(),
        blockers: canary_blockers,
        warnings: canary_warnings,
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
) -> (&'static str, String) {
    match policy.status {
        ArsParameterPolicyLoadStatus::Missing => (
            "ok",
            "missing policy row; dynamic ARS parameters disabled".to_string(),
        ),
        ArsParameterPolicyLoadStatus::Corrupt | ArsParameterPolicyLoadStatus::StorageError => (
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
                "mode={:?} revision={} source_adaptive_version={} current_adaptive_version={} live_allowed={}",
                policy.policy.mode,
                policy.policy.revision,
                policy.policy.source_adaptive_version,
                state.version,
                live_allowed,
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
