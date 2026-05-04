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

// ─── v0.28.7+ audit L7 — coverage for previously-missing failure modes ───
//
// Each test below pins the gate's behavior against a specific corruption
// or race state that pre-v0.28.7 had no test coverage for. Several of
// these would have caught the `H0` / `H2` regressions if they had
// existed during the v0.28.0..v0.28.6 rollout.

fn unhealthy_policy(status: ArsParameterPolicyLoadStatus, error: &str) -> ArsParameterPolicyLoad {
    ArsParameterPolicyLoad {
        // The store layer also loads `ArsParameterPolicy::disabled(reason)`
        // here, so disabled_reason is populated. Mirror that to keep the
        // synthetic input fixture-realistic.
        policy: ArsParameterPolicy {
            mode: ArsParameterPolicyMode::Disabled,
            disabled_reason: Some(error.to_string()),
            ..ArsParameterPolicy::default()
        },
        status,
        error: Some(error.to_string()),
    }
}

#[test]
fn corrupt_policy_blocks_canary_with_unhealthy_blocker() {
    let mut config = ReinConfig::default();
    config.ars.acceleration.enabled = true;
    config.ars.acceleration.shadow_only = false;
    config.adaptive.min_samples_alpha = 10;
    let state = eligible_state(12);
    let policy = unhealthy_policy(ArsParameterPolicyLoadStatus::Corrupt, "corrupt policy row");
    let shadow_fusion_status = ready_shadow_status();

    let report = evaluate_ars_acceleration_release_gate(input(
        &config,
        &state,
        &policy,
        &shadow_fusion_status,
    ));

    assert!(
        !report.canary.allowed,
        "corrupt policy row must block canary; pre-test absence let H0-class \
         silent stalls slip past release-gate validation"
    );
    assert!(
        report
            .canary
            .blockers
            .iter()
            .any(|b| b == "ars_parameter_policy_unhealthy"),
        "expected `ars_parameter_policy_unhealthy` blocker; got {:?}",
        report.canary.blockers
    );
    assert_eq!(report.signals.policy_status, "corrupt");
    assert_eq!(
        report.signals.policy_error.as_deref(),
        Some("corrupt policy row")
    );
}

/// v0.28.7+ audit L5 R5 P2 follow-up — a future-schema policy row
/// (`UnsupportedSchema` load status) must surface the same
/// `ars_parameter_policy_unhealthy` canary blocker as `Corrupt` /
/// `StorageError`. The release gate's job is to refuse to promote
/// against a policy it can't safely interpret, regardless of WHY it
/// can't; only `doctor --fix` distinguishes the three for recovery.
#[test]
fn unsupported_schema_policy_blocks_canary_with_unhealthy_blocker() {
    let mut config = ReinConfig::default();
    config.ars.acceleration.enabled = true;
    config.ars.acceleration.shadow_only = false;
    config.adaptive.min_samples_alpha = 10;
    let state = eligible_state(12);
    let policy = unhealthy_policy(
        ArsParameterPolicyLoadStatus::UnsupportedSchema,
        "unsupported policy schema version",
    );
    let shadow_fusion_status = ready_shadow_status();

    let report = evaluate_ars_acceleration_release_gate(input(
        &config,
        &state,
        &policy,
        &shadow_fusion_status,
    ));

    assert!(
        !report.canary.allowed,
        "future-schema policy load must block canary"
    );
    assert!(
        report
            .canary
            .blockers
            .iter()
            .any(|b| b == "ars_parameter_policy_unhealthy"),
        "expected `ars_parameter_policy_unhealthy` blocker; got {:?}",
        report.canary.blockers
    );
    assert_eq!(report.signals.policy_status, "unsupported_schema");
}

#[test]
fn storage_error_policy_blocks_canary_with_unhealthy_blocker() {
    let mut config = ReinConfig::default();
    config.ars.acceleration.enabled = true;
    config.ars.acceleration.shadow_only = false;
    config.adaptive.min_samples_alpha = 10;
    let state = eligible_state(12);
    let policy = unhealthy_policy(
        ArsParameterPolicyLoadStatus::StorageError,
        "io error: disk full",
    );
    let shadow_fusion_status = ready_shadow_status();

    let report = evaluate_ars_acceleration_release_gate(input(
        &config,
        &state,
        &policy,
        &shadow_fusion_status,
    ));

    assert!(
        !report.canary.allowed,
        "storage-error policy load must block canary"
    );
    assert!(
        report
            .canary
            .blockers
            .iter()
            .any(|b| b == "ars_parameter_policy_unhealthy"),
        "expected `ars_parameter_policy_unhealthy` blocker; got {:?}",
        report.canary.blockers
    );
    assert_eq!(report.signals.policy_status, "storage_error");
}

#[test]
fn stale_source_adaptive_version_blocks_canary_with_not_current_blocker() {
    let mut config = ReinConfig::default();
    config.ars.acceleration.enabled = true;
    config.ars.acceleration.shadow_only = false;
    config.adaptive.min_samples_alpha = 10;
    let state = eligible_state(12);
    // Policy claims it was minted against adaptive_version=999, but the
    // actual state is at version=9. This is the "policy is from a future
    // schema/state we don't have yet" race that the gate must refuse to
    // promote into.
    let stale_policy = loaded_canary_policy(state.version + 1_000);
    let shadow_fusion_status = ready_shadow_status();

    let report = evaluate_ars_acceleration_release_gate(input(
        &config,
        &state,
        &stale_policy,
        &shadow_fusion_status,
    ));

    assert!(
        !report.canary.allowed,
        "stale source_adaptive_version must block canary"
    );
    assert!(
        report
            .canary
            .blockers
            .iter()
            .any(|b| b == "ars_parameter_policy_not_current"),
        "expected `ars_parameter_policy_not_current` blocker; got {:?}",
        report.canary.blockers
    );
    assert_eq!(
        report.signals.policy_source_adaptive_version,
        state.version + 1_000
    );
    assert_eq!(report.signals.adaptive_version, state.version);
}

#[test]
fn post_canary_drift_alert_synthesis_surface_blocks_canary() {
    // Mirror of `drift_alert_blocks_canary_even_with_loaded_canary_policy`
    // but exercises the SYNTHESIS-surface drift counter, not the
    // concept-summary one. Both sides participate in the audit's H2 fix
    // (`refresh_ars_parameter_policy` reads all three signals); the
    // existing test only covers concept_summary, leaving synthesis as a
    // silent-regression path until this test was added.
    let mut config = ReinConfig::default();
    config.ars.acceleration.enabled = true;
    config.ars.acceleration.shadow_only = false;
    config.adaptive.min_samples_alpha = 10;
    let mut state = eligible_state(12);
    state.judge_calibration_state = Some(JudgeCalibrationState {
        judge_drift_alert_synthesis: 1,
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

    assert!(
        !report.canary.allowed,
        "synthesis-surface drift counter must block canary post-promotion"
    );
    assert!(
        report
            .canary
            .blockers
            .iter()
            .any(|b| b == "judge_drift_alert"),
        "expected `judge_drift_alert` blocker (any-surface drift); got {:?}",
        report.canary.blockers
    );
    assert!(report.signals.judge_drift_alert);

    // Cross-check: setting BOTH per-surface counters does not
    // double-emit the blocker, and still aggregates to a single
    // `judge_drift_alert` signal.
    state.judge_calibration_state = Some(JudgeCalibrationState {
        judge_drift_alert_synthesis: 1,
        judge_drift_alert_concept: 1,
        ..JudgeCalibrationState::default()
    });
    let report_both = evaluate_ars_acceleration_release_gate(input(
        &config,
        &state,
        &policy,
        &shadow_fusion_status,
    ));
    let drift_blockers = report_both
        .canary
        .blockers
        .iter()
        .filter(|b| *b == "judge_drift_alert")
        .count();
    assert_eq!(
        drift_blockers, 1,
        "drift blocker must dedupe across surface counters; got {drift_blockers}"
    );
}

#[test]
fn gate_decision_is_invariant_under_policy_revision_changes() {
    // Concurrent CAS coverage: the policy CAS path bumps `revision` on
    // every successful write. A workload with frequent canary→shadow
    // toggles can produce a long sequence of policies that differ only
    // in `revision`. The gate's allow/deny decision must depend on
    // SEMANTIC fields (mode, source_adaptive_version, signals) — never on
    // `revision` itself. Otherwise a fast-moving CAS workload could
    // briefly flap the gate's report between "allowed" and "blocked"
    // even though no semantic state changed.
    let mut config = ReinConfig::default();
    config.ars.acceleration.enabled = true;
    config.ars.acceleration.shadow_only = false;
    config.adaptive.min_samples_alpha = 10;
    let state = eligible_state(12);
    let shadow_fusion_status = ready_shadow_status();

    // Build six revisions of the same semantic policy spanning a CAS
    // burst (low → mid → high → near-max).
    let revisions: [u64; 6] = [0, 1, 7, 1_000, 1_000_000, u64::MAX - 1];
    let mut decisions: Vec<bool> = Vec::new();
    let mut blocker_sets: Vec<Vec<String>> = Vec::new();
    for rev in revisions {
        let mut policy = loaded_canary_policy(state.version);
        policy.policy.revision = rev;
        let report = evaluate_ars_acceleration_release_gate(input(
            &config,
            &state,
            &policy,
            &shadow_fusion_status,
        ));
        // Gate must surface the input revision as-is (signal is a passthrough).
        assert_eq!(
            report.signals.policy_revision, rev,
            "policy_revision signal must passthrough the input revision"
        );
        decisions.push(report.canary.allowed);
        let mut blockers = report.canary.blockers.clone();
        blockers.sort();
        blocker_sets.push(blockers);
    }

    // All decisions identical (all true with this fixture: healthy
    // canary with sufficient evidence and no drift).
    assert!(
        decisions.iter().all(|&d| d == decisions[0]),
        "canary.allowed must be invariant across revisions; got {decisions:?}"
    );
    assert!(
        blocker_sets.iter().all(|b| b == &blocker_sets[0]),
        "canary.blockers set must be invariant across revisions; got {blocker_sets:?}"
    );
    assert!(
        decisions[0],
        "fixture is the healthy-canary path; all revisions should be allowed"
    );
}
