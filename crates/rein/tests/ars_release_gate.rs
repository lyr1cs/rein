use rein::config::ReinConfig;
use rein::judge::contract::JudgeStructuralStatus;
use rein::ops::a12_activation::{
    RecallEvalGateAttestation, RecallEvalGateReasonCode, RecallFusionCalibrationRunAttestation,
    RecallFusionCalibrationRunPhase,
};
use rein::ops::ars_release_gate::{
    evaluate_ars_acceleration_release_gate, evaluate_ars_acceleration_release_gate_with_a12,
    JudgeStructuralReleaseInput, ReleaseGateInput,
};
use rein::ops::ars_tuning::JudgeStructuralTrustContext;
use rein::store::a12_calibration::{
    A12CalibrationLoad, A12CalibrationLoadStatus, A12CalibrationPhase, A12CalibrationRunMetadata,
    A12CalibrationScope, A12CalibrationState, A12CalibrationVerdict, A12FusionSimplex,
    A12PairedTop3Stats, A12ProvenanceCounts, A12ScopeEntry, A12_CALIBRATION_SCHEMA_VERSION,
};
use rein::store::adaptive::{
    AdaptiveState, JudgeCalibrationState, LearnedShadowFusionEntry, ShadowFusionWeightEntry,
};
use rein::store::ars_parameter_policy::{
    ArsParameterPolicy, ArsParameterPolicyLoad, ArsParameterPolicyLoadStatus,
    ArsParameterPolicyMode, ArsRecallGateStatus,
};
use std::collections::BTreeMap;

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
        judge_structural: JudgeStructuralReleaseInput::default(),
    }
}

fn structural_input(status: JudgeStructuralStatus, enforce: bool) -> JudgeStructuralReleaseInput {
    JudgeStructuralReleaseInput {
        synthesis: JudgeStructuralTrustContext {
            status,
            enforce,
            gate_required: true,
        },
        concept_summary: JudgeStructuralTrustContext::default(),
    }
}

fn a12_paired(n: u32) -> A12PairedTop3Stats {
    let result = rein::eval::mcnemar::mcnemar_from_counts(n, 0, 0, 0).unwrap();
    A12PairedTop3Stats {
        n: u64::from(result.n),
        both_hit: u64::from(result.a),
        baseline_only: u64::from(result.b),
        treatment_only: u64::from(result.c),
        neither_hit: u64::from(result.d),
        chi_squared: result.chi_squared,
        p_value: result.p_value,
        diff_point: result.diff_point,
        ci_lower: result.ci_lower,
        ci_upper: result.ci_upper,
        used_exact: result.used_exact,
    }
}

fn a12_global(train_ess: u64, holdout_ess: u64, valid_until: Option<i64>) -> A12CalibrationLoad {
    let entry = A12ScopeEntry {
        scope: A12CalibrationScope::Global,
        canonical_generation: 2,
        generation_fingerprint: "generation-fingerprint".to_string(),
        source_snapshot_fingerprint: "source-snapshot-fingerprint".to_string(),
        snapshot_cutoff: 1_700_000_000,
        corpus_fingerprint: "corpus-fingerprint".to_string(),
        train_family_ess: train_ess,
        train_case_count: train_ess,
        holdout_family_ess: holdout_ess,
        simplex: A12FusionSimplex {
            bm25: 0.10,
            vector: 0.20,
            kg: 0.30,
            episode: 0.15,
            support: 0.15,
            diversity: 0.10,
        },
        verdict: A12CalibrationVerdict::Ship,
        noise_floor: 0.02,
        paired_top3: a12_paired(u32::try_from(holdout_ess).unwrap()),
        provenance: A12ProvenanceCounts {
            canonical_loo: train_ess,
            concept_loo: 0,
            episode_loo: 0,
        },
        training_fingerprint: "training-fingerprint".to_string(),
        holdout_fingerprint: "holdout-fingerprint".to_string(),
        optimizer_fingerprint: "optimizer-fingerprint".to_string(),
        evaluation_fingerprint: "evaluation-fingerprint".to_string(),
        holdout_reason: "holdout evaluated".to_string(),
        calibrated_at: 1_700_000_000,
        evaluated_at: 1_700_000_050,
        valid_until_exclusive: valid_until,
        cluster_generation: None,
        invalidation: None,
    };
    A12CalibrationLoad {
        state: A12CalibrationState {
            schema_version: A12_CALIBRATION_SCHEMA_VERSION,
            revision: 3,
            generation: 2,
            generation_fingerprint: "generation-fingerprint".to_string(),
            snapshot_cutoff: 1_700_000_000,
            corpus_fingerprint: "corpus-fingerprint".to_string(),
            cluster_generation: 4,
            scopes: BTreeMap::from([("global".to_string(), entry)]),
            created_at: 1_700_000_000,
            updated_at: 1_700_000_050,
            run: Some(A12CalibrationRunMetadata {
                phase: A12CalibrationPhase::Complete,
                source_snapshot_fingerprint: "source-snapshot-fingerprint".to_string(),
                behavior_config_fingerprint: "behavior-config-fingerprint".to_string(),
            }),
        },
        status: A12CalibrationLoadStatus::Loaded,
        error: None,
    }
}

fn auto_policy(
    state: &AdaptiveState,
    a12: &A12CalibrationLoad,
    scalar_weight: f64,
    scoped_weight: f64,
    now_millis: i64,
) -> ArsParameterPolicyLoad {
    let gate = rein::ops::a12_activation::RecallEvalGateAttestation {
        status: ArsRecallGateStatus::Ship,
        reason_code: rein::ops::a12_activation::RecallEvalGateReasonCode::Compared,
        build_fingerprint: Some(env!("REIN_BUILD_FINGERPRINT").to_string()),
        fixture_fingerprint: Some("recall-fixtures".to_string()),
        evaluated_at: Some(1_700_000_100),
        reason: "paired recall gate shipped".to_string(),
    };
    let evidence = rein::ops::a12_activation::resolve_recall_fusion_evidence(
        state, a12, 10, 0.02, now_millis, &gate,
    );
    ArsParameterPolicyLoad {
        policy: ArsParameterPolicy {
            revision: 4,
            mode: ArsParameterPolicyMode::Canary,
            disabled_reason: None,
            source_adaptive_version: state.version,
            runtime_adoption_weight: scalar_weight,
            adoption_weights: std::collections::HashMap::from([(
                "recall_fusion:global".to_string(),
                scoped_weight,
            )]),
            recall_fusion_evidence: evidence.into_iter().collect(),
            last_updated: "2026-07-13T00:00:00Z".to_string(),
            ..ArsParameterPolicy::default()
        },
        status: ArsParameterPolicyLoadStatus::Loaded,
        error: None,
    }
}

fn current_ship_gate() -> RecallEvalGateAttestation {
    RecallEvalGateAttestation {
        status: ArsRecallGateStatus::Ship,
        reason_code: RecallEvalGateReasonCode::Compared,
        build_fingerprint: Some(env!("REIN_BUILD_FINGERPRINT").to_string()),
        fixture_fingerprint: Some("recall-fixtures".to_string()),
        evaluated_at: Some(1_700_000_100),
        reason: "paired recall gate shipped".to_string(),
    }
}

fn complete_run() -> RecallFusionCalibrationRunAttestation {
    RecallFusionCalibrationRunAttestation {
        phase: RecallFusionCalibrationRunPhase::Complete,
        source_snapshot_fingerprint: "source-snapshot-fingerprint".to_string(),
        behavior_config_fingerprint: "behavior-config-fingerprint".to_string(),
    }
}

fn evaluate_with_complete_a12(
    input: ReleaseGateInput<'_>,
    a12: &A12CalibrationLoad,
    now_millis: i64,
) -> rein::ops::ars_release_gate::ArsAccelerationReleaseGateReport {
    evaluate_ars_acceleration_release_gate_with_a12(
        input,
        a12,
        &current_ship_gate(),
        Some(&complete_run()),
        now_millis,
    )
}

#[test]
fn a12_ship_with_scoped_weight_and_zero_scalar_allows_recall_canary() {
    let mut config = ReinConfig::default();
    config.ars.acceleration.enabled = true;
    config.ars.acceleration.shadow_only = false;
    config.adaptive.min_samples_alpha = 10;
    let state = AdaptiveState {
        version: 9,
        ..AdaptiveState::default()
    };
    let a12 = a12_global(12, 12, None);
    let policy = auto_policy(&state, &a12, 0.0, 0.40, 1_700_000_060_000);
    let shadow = serde_json::json!({ "status": "insufficient_samples" });

    let report = evaluate_with_complete_a12(
        input(&config, &state, &policy, &shadow),
        &a12,
        1_700_000_060_000,
    );

    assert_eq!(report.schema_version, 3);
    assert!(report.canary.allowed, "{:?}", report.canary.blockers);
    assert_eq!(report.signals.scalar_runtime_adoption_weight, 0.0);
    assert!(report.signals.recall_runtime_allowed);
    assert!(report.signals.recall_self_supervised_runtime_allowed);
    assert!(!report.signals.recall_human_runtime_allowed);
    assert!(report.signals.recall_fusion_calibration.active);
}

#[test]
fn a12_training_volume_without_holdout_stays_blocked() {
    let mut config = ReinConfig::default();
    config.adaptive.min_samples_alpha = 10;
    let state = AdaptiveState {
        version: 9,
        ..AdaptiveState::default()
    };
    let mut a12 = a12_global(12, 0, None);
    a12.state.scopes.get_mut("global").unwrap().verdict = A12CalibrationVerdict::NoData;
    let policy = auto_policy(&state, &a12, 0.0, 0.40, 1_700_000_060_000);
    let shadow = serde_json::json!({ "status": "insufficient_samples" });

    let report = evaluate_with_complete_a12(
        input(&config, &state, &policy, &shadow),
        &a12,
        1_700_000_060_000,
    );

    assert!(!report.canary.allowed);
    assert!(!report.signals.recall_runtime_allowed);
    assert!(report
        .canary
        .blockers
        .iter()
        .any(|blocker| blocker == "recall_fusion_runtime_unavailable"));
}

#[test]
fn expired_a12_scope_stays_blocked() {
    let mut config = ReinConfig::default();
    config.adaptive.min_samples_alpha = 10;
    let state = AdaptiveState {
        version: 9,
        ..AdaptiveState::default()
    };
    let a12 = a12_global(12, 12, Some(1_700_000_061_000));
    let policy = auto_policy(&state, &a12, 0.0, 0.40, 1_700_000_060_000);
    let shadow = serde_json::json!({ "status": "insufficient_samples" });

    let report = evaluate_with_complete_a12(
        input(&config, &state, &policy, &shadow),
        &a12,
        1_700_000_061_000,
    );

    assert!(!report.canary.allowed);
    assert!(!report.signals.recall_fusion_calibration.scopes[0].valid_now);
    assert!(report.signals.recall_fusion_calibration.scopes[0]
        .reason
        .contains("stale or expired"));
}

#[test]
fn fingerprint_mismatched_a12_scope_stays_blocked() {
    let mut config = ReinConfig::default();
    config.adaptive.min_samples_alpha = 10;
    let state = AdaptiveState {
        version: 9,
        ..AdaptiveState::default()
    };
    let original = a12_global(12, 12, None);
    let policy = auto_policy(&state, &original, 0.0, 0.40, 1_700_000_060_000);
    let mut current = original;
    current.state.generation = 3;
    current.state.revision = 4;
    current.state.generation_fingerprint = "new-generation-fingerprint".to_string();
    let current_entry = current.state.scopes.get_mut("global").unwrap();
    current_entry.canonical_generation = 3;
    current_entry.generation_fingerprint = "new-generation-fingerprint".to_string();
    let shadow = serde_json::json!({ "status": "insufficient_samples" });

    let report = evaluate_with_complete_a12(
        input(&config, &state, &policy, &shadow),
        &current,
        1_700_000_060_000,
    );

    assert!(!report.canary.allowed);
    assert!(!report.signals.recall_runtime_allowed);
    assert!(report.signals.recall_fusion_calibration.scopes[0]
        .reason
        .contains("fingerprint identity mismatched"));
}

#[test]
fn auto_only_recall_rejects_nonzero_scalar_runtime_adoption() {
    let mut config = ReinConfig::default();
    config.adaptive.min_samples_alpha = 10;
    let state = AdaptiveState {
        version: 9,
        ..AdaptiveState::default()
    };
    let a12 = a12_global(12, 12, None);
    let policy = auto_policy(&state, &a12, 0.25, 0.40, 1_700_000_060_000);
    let shadow = serde_json::json!({ "status": "insufficient_samples" });

    let report = evaluate_with_complete_a12(
        input(&config, &state, &policy, &shadow),
        &a12,
        1_700_000_060_000,
    );

    assert!(!report.canary.allowed);
    assert!(report
        .canary
        .blockers
        .iter()
        .any(|blocker| blocker == "automatic_recall_fusion_scalar_isolation"));
}

#[test]
fn auto_only_recall_rejects_nonrecall_scoped_scalar_adoption() {
    let mut config = ReinConfig::default();
    config.adaptive.min_samples_alpha = 10;
    let state = AdaptiveState {
        version: 9,
        ..AdaptiveState::default()
    };
    let a12 = a12_global(12, 12, None);
    let mut policy = auto_policy(&state, &a12, 0.0, 0.40, 1_700_000_060_000);
    policy
        .policy
        .adoption_weights
        .insert("synthesis_gate".to_string(), 0.25);
    let shadow = serde_json::json!({ "status": "insufficient_samples" });

    let report = evaluate_with_complete_a12(
        input(&config, &state, &policy, &shadow),
        &a12,
        1_700_000_060_000,
    );

    assert!(!report.canary.allowed);
    assert_eq!(
        report.signals.scalar_runtime_adoption_weights["synthesis_gate"],
        0.25
    );
    assert!(report
        .canary
        .blockers
        .iter()
        .any(|blocker| blocker == "automatic_recall_fusion_scalar_isolation"));
}

/// The scalar-isolation blocker deliberately does not fire when human recall
/// runtime adoption is also allowed: with live human evidence the scalar
/// scopes may legitimately be human-driven. It fires only when self-supervised
/// recall fusion is the sole runtime basis.
#[test]
fn scalar_adoption_is_not_isolated_when_human_recall_runtime_is_also_allowed() {
    let mut config = ReinConfig::default();
    config.ars.acceleration.enabled = true;
    config.ars.acceleration.shadow_only = false;
    config.adaptive.min_samples_alpha = 10;
    let state = eligible_state(12);
    let a12 = a12_global(12, 12, None);
    let policy = auto_policy(&state, &a12, 0.25, 0.40, 1_700_000_060_000);
    let shadow = ready_shadow_status();

    let report = evaluate_with_complete_a12(
        input(&config, &state, &policy, &shadow),
        &a12,
        1_700_000_060_000,
    );

    assert!(report.signals.recall_human_runtime_allowed);
    assert!(report.signals.recall_self_supervised_runtime_allowed);
    assert!(report.signals.scalar_runtime_adoption_weight > 0.0);
    assert!(!report
        .canary
        .blockers
        .iter()
        .any(|blocker| blocker == "automatic_recall_fusion_scalar_isolation"));
    assert!(report.canary.allowed, "{:?}", report.canary.blockers);
}

#[test]
fn nonrecall_scalar_scope_cannot_unlock_recall_canary() {
    let mut config = ReinConfig::default();
    config.adaptive.min_samples_alpha = 10;
    let state = AdaptiveState {
        version: 9,
        ..AdaptiveState::default()
    };
    let mut policy = loaded_canary_policy(state.version);
    policy.policy.runtime_adoption_weight = 0.0;
    policy.policy.adoption_weights =
        std::collections::HashMap::from([("synthesis_gate".to_string(), 0.75)]);
    let a12 = A12CalibrationLoad {
        state: A12CalibrationState::default(),
        status: A12CalibrationLoadStatus::Missing,
        error: None,
    };
    let shadow = serde_json::json!({ "status": "ready" });

    let report = evaluate_with_complete_a12(
        input(&config, &state, &policy, &shadow),
        &a12,
        1_700_000_060_000,
    );

    assert!(!report.canary.allowed);
    assert!(!report.signals.recall_runtime_allowed);
}

#[test]
fn failed_stale_and_mismatched_structural_health_block_without_faking_human_kappa() {
    for status in [
        JudgeStructuralStatus::Failed,
        JudgeStructuralStatus::Stale,
        JudgeStructuralStatus::FingerprintMismatch,
        JudgeStructuralStatus::Corrupt,
        JudgeStructuralStatus::Unknown,
    ] {
        let mut config = ReinConfig::default();
        config.ars.llm_judge.enabled = true;
        config.ars.llm_judge.synthesis_enabled = true;
        config.ars.llm_judge.concept_summary_enabled = false;
        let state = eligible_state(12);
        let policy = loaded_canary_policy(state.version);
        let shadow_fusion_status = ready_shadow_status();
        let report = evaluate_ars_acceleration_release_gate(ReleaseGateInput {
            config: &config,
            state: &state,
            policy: &policy,
            shadow_fusion_status: &shadow_fusion_status,
            judge_structural: structural_input(status, true),
        });

        assert!(!report.canary.allowed, "{status:?} must block promotion");
        assert!(report.canary.blockers.iter().any(|blocker| {
            blocker
                == &format!(
                    "judge_structural_unhealthy:synthesis:{}",
                    match status {
                        JudgeStructuralStatus::Failed => "failed",
                        JudgeStructuralStatus::Stale => "stale",
                        JudgeStructuralStatus::FingerprintMismatch => "fingerprint_mismatch",
                        JudgeStructuralStatus::Corrupt => "corrupt",
                        JudgeStructuralStatus::Unknown => "unknown",
                        _ => unreachable!(),
                    }
                )
        }));
        assert_eq!(report.signals.judge_human_pairs_synthesis, 0);
        assert_eq!(report.signals.judge_human_kappa_synthesis, None);
        assert_eq!(report.signals.judge_calibration_pairs, 0);
    }
}

#[test]
fn ready_structural_anchors_do_not_authorize_release_promotion() {
    let mut config = ReinConfig::default();
    config.ars.llm_judge.enabled = true;
    config.ars.llm_judge.synthesis_enabled = true;
    config.ars.llm_judge.concept_summary_enabled = false;
    let state = eligible_state(12);
    let policy = loaded_canary_policy(state.version);
    let shadow_fusion_status = ready_shadow_status();
    let report = evaluate_ars_acceleration_release_gate(ReleaseGateInput {
        config: &config,
        state: &state,
        policy: &policy,
        shadow_fusion_status: &shadow_fusion_status,
        judge_structural: structural_input(JudgeStructuralStatus::Ready, true),
    });

    assert!(!report.canary.allowed);
    assert_eq!(report.schema_version, 3);
    assert!(report
        .canary
        .blockers
        .iter()
        .any(|blocker| blocker
            == "judge_trust_disallows_promotion:synthesis:structural_anchors_ready"));
    assert_eq!(
        report.signals.judge_calibration_basis_synthesis,
        "structural_anchors"
    );
    assert_eq!(report.signals.judge_human_kappa_synthesis, None);
}

#[test]
fn monitor_failure_does_not_override_healthy_human_calibration() {
    let mut config = ReinConfig::default();
    config.ars.llm_judge.enabled = true;
    config.ars.llm_judge.synthesis_enabled = true;
    config.ars.llm_judge.concept_summary_enabled = false;
    let mut state = eligible_state(12);
    let mut calibration = JudgeCalibrationState {
        kappa: 0.9,
        ..JudgeCalibrationState::default()
    };
    for idx in 0..rein::store::adaptive::LLM_JUDGE_J3_MIN_PAIRS {
        calibration
            .recent_pairs_synthesis
            .push_back((idx % 2 == 0, idx % 2 == 0, idx as i64));
    }
    state.judge_calibration_state = Some(calibration);
    let policy = loaded_canary_policy(state.version);
    let shadow_fusion_status = ready_shadow_status();
    let report = evaluate_ars_acceleration_release_gate(ReleaseGateInput {
        config: &config,
        state: &state,
        policy: &policy,
        shadow_fusion_status: &shadow_fusion_status,
        judge_structural: structural_input(JudgeStructuralStatus::Failed, false),
    });

    assert!(
        report.canary.allowed,
        "monitor status must not be an enforce blocker"
    );
    assert!(!report
        .canary
        .blockers
        .iter()
        .any(|blocker| blocker.starts_with("judge_structural_unhealthy:")));
}

#[test]
fn human_negative_evidence_precedes_structural_failure_in_release_gate() {
    let mut config = ReinConfig::default();
    config.ars.llm_judge.enabled = true;
    config.ars.llm_judge.synthesis_enabled = true;
    config.ars.llm_judge.concept_summary_enabled = false;
    let mut state = eligible_state(12);
    let mut calibration = JudgeCalibrationState {
        kappa: 0.1,
        ..JudgeCalibrationState::default()
    };
    for idx in 0..rein::store::adaptive::LLM_JUDGE_J3_MIN_PAIRS {
        calibration
            .recent_pairs_synthesis
            .push_back((idx % 2 == 0, idx % 2 == 0, idx as i64));
    }
    state.judge_calibration_state = Some(calibration);
    let policy = loaded_canary_policy(state.version);
    let shadow_fusion_status = ready_shadow_status();
    let report = evaluate_ars_acceleration_release_gate(ReleaseGateInput {
        config: &config,
        state: &state,
        policy: &policy,
        shadow_fusion_status: &shadow_fusion_status,
        judge_structural: structural_input(JudgeStructuralStatus::Failed, true),
    });

    assert!(report
        .canary
        .blockers
        .iter()
        .any(|blocker| blocker == "judge_human_kappa_below_floor:synthesis"));
    assert!(!report
        .canary
        .blockers
        .iter()
        .any(|blocker| blocker.starts_with("judge_structural_unhealthy:synthesis")));
    assert_eq!(report.signals.judge_human_pairs_synthesis, 30);
    assert_eq!(report.signals.judge_human_kappa_synthesis, Some(0.1));
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

/// v1.2 (#A12 activation prerequisite): shadow-fusion replay readiness is a
/// BLOCKER. An otherwise-fully-healthy canary (policy canary mode + filled
/// buckets + samples over min) must NOT be allowed on volume alone when the
/// replay's quality machinery hasn't produced a "ready" verdict.
#[test]
fn replay_not_ready_blocks_canary_despite_healthy_volume() {
    let mut config = ReinConfig::default();
    config.ars.acceleration.enabled = true;
    config.ars.acceleration.shadow_only = false;
    config.adaptive.min_samples_alpha = 10;
    let state = eligible_state(12);
    let policy = loaded_canary_policy(state.version);
    let mut shadow_fusion_status = ready_shadow_status();
    shadow_fusion_status["status"] = serde_json::json!("insufficient_samples");

    let report = evaluate_ars_acceleration_release_gate(input(
        &config,
        &state,
        &policy,
        &shadow_fusion_status,
    ));

    assert!(
        !report.canary.allowed,
        "volume-only readiness must not pass the canary gate"
    );
    assert!(
        report
            .canary
            .blockers
            .iter()
            .any(|b| b == "shadow_fusion_replay_not_ready:insufficient_samples"),
        "not-ready replay must surface as a BLOCKER, got {:?}",
        report.canary.blockers
    );
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
