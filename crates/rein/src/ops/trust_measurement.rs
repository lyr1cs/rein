//! Unified Trust & Measurement snapshot for ARS rollout operations.

use serde::Serialize;

use crate::config::ReinConfig;
use crate::ops::ars_release_gate::ArsAccelerationReleaseGateReport;
use crate::ops::system_health::SystemHealthSnapshot;
use crate::store::SqliteStore;

pub const TRUST_MEASUREMENT_SCHEMA_VERSION: u32 = 3;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct MeasurementGate {
    pub name: String,
    pub status: String,
    pub signal: String,
    pub command: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct IndexConsistencyReport {
    pub status: String,
    pub active_memory_count: u64,
    pub vector_row_count: u64,
    pub missing_embeddings: u64,
    pub orphan_embeddings: u64,
    /// v0.35 T&M Phase 3 — repair advice. Each entry is an operator-facing
    /// hint derived from the counts above (e.g. "run `rein doctor --fix`
    /// to backfill missing embeddings"). Empty when `status == "ok"`. Pure
    /// semantic strings — no paths, secrets, or host-local details so the
    /// vector is safe for the public `/api/trust-measurement` route.
    #[serde(default)]
    pub repair_advice: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BackgroundObservabilityReport {
    pub status: String,
    pub adaptive_version: u64,
    pub learned_shadow_fusion_buckets: usize,
    pub feedback_event_count: u64,
    pub consumer_offset_count: u64,
    pub system: SystemHealthSnapshot,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ActiveLearningReport {
    pub status: String,
    pub llm_judge_enabled: bool,
    pub synthesis_enabled: bool,
    pub concept_summary_enabled: bool,
    pub nightly_cron_enabled: bool,
    pub sample_rate_cold_start: f64,
    pub sample_rate_warm: f64,
    pub nightly_cron_sample_rate: f64,
    /// v0.35 T&M Phase 3 — canary drift signal aggregated across the three
    /// per-surface counters
    /// (`judge_drift_alert` + `judge_drift_alert_synthesis` +
    /// `judge_drift_alert_concept`) maintained by the judge calibration
    /// state. Non-zero values indicate the runtime LLM judge's recent
    /// decisions diverged from the deterministic baseline often enough to
    /// be worth investigation. Reported here as a single observability
    /// number; the doctor surface still distinguishes per-surface.
    #[serde(default)]
    pub judge_drift_alert_total: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct JudgeHumanAgreementReport {
    pub pair_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kappa: Option<f64>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct JudgeRuntimeVsNightlyReport {
    pub pair_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kappa: Option<f64>,
    pub drift_alert_count: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct JudgePromotionDecisionReport {
    pub requested_action: String,
    pub basis: String,
    pub reason: String,
    pub promotion_allowed: bool,
    pub promotion_blocks_release: bool,
    /// Mirrors the release gate's effective blocker condition.
    pub release_blocked: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct JudgeStructuralReport {
    pub mode: String,
    pub load_status: String,
    pub status: String,
    pub requested_action: String,
    pub basis: String,
    pub reason: String,
    pub baseline_allowed: bool,
    pub configured_baseline_scale: f64,
    pub baseline_blocks_release: bool,
    pub recall_fusion_promotion: JudgePromotionDecisionReport,
    pub surface_scope_promotion: JudgePromotionDecisionReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_model_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observed_model_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_rubric_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observed_rubric_fingerprint: Option<String>,
    pub expected_probe_set_version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observed_probe_set_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_probe_at: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fresh_until: Option<i64>,
    pub is_fresh: bool,
    pub alert_count: u64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub repair_advice: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct JudgeSurfaceCalibrationReport {
    pub human: JudgeHumanAgreementReport,
    pub runtime_vs_nightly: JudgeRuntimeVsNightlyReport,
    pub structural: JudgeStructuralReport,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct JudgeCalibrationObservabilityReport {
    pub synthesis: JudgeSurfaceCalibrationReport,
    pub concept_summary: JudgeSurfaceCalibrationReport,
    pub human_runtime_watermark: i64,
    pub structural_watermark: i64,
    pub last_runtime_computed_at: i64,
    pub total_runtime_nightly_events: u64,
    pub global_drift_alert_count: u64,
}

fn judge_action_name(action: crate::judge::contract::JudgeTrustAction) -> &'static str {
    use crate::judge::contract::JudgeTrustAction;
    match action {
        JudgeTrustAction::KeepConfiguredBaseline => "keep_configured_baseline",
        JudgeTrustAction::IncreaseJudgeWeight => "increase_judge_weight",
        JudgeTrustAction::IncreaseSampleRate => "increase_sample_rate",
        JudgeTrustAction::PromoteJudgeScope => "promote_judge_scope",
        JudgeTrustAction::PromoteRecallFusion => "promote_recall_fusion",
    }
}

fn judge_basis_name(basis: crate::judge::contract::JudgeCalibrationBasis) -> &'static str {
    use crate::judge::contract::JudgeCalibrationBasis;
    match basis {
        JudgeCalibrationBasis::HumanKappa => "human_kappa",
        JudgeCalibrationBasis::StructuralAnchors => "structural_anchors",
        JudgeCalibrationBasis::ConfiguredBaseline => "configured_baseline",
        JudgeCalibrationBasis::Untrusted => "untrusted",
    }
}

fn judge_reason_name(reason: crate::judge::contract::JudgeTrustReason) -> &'static str {
    use crate::judge::contract::JudgeTrustReason;
    match reason {
        JudgeTrustReason::HumanKappaHealthy => "human_kappa_healthy",
        JudgeTrustReason::HumanKappaBelowFloor => "human_kappa_below_floor",
        JudgeTrustReason::HumanKappaMissingOrInvalid => "human_kappa_missing_or_invalid",
        JudgeTrustReason::StructuralAnchorsReady => "structural_anchors_ready",
        JudgeTrustReason::ConfiguredBaselineFallback => "configured_baseline_fallback",
        JudgeTrustReason::StructuralHealthEnforced => "structural_health_enforced",
    }
}

fn judge_promotion_report(
    requested_action: crate::judge::contract::JudgeTrustAction,
    decision: crate::judge::contract::JudgeTrustDecision,
) -> JudgePromotionDecisionReport {
    JudgePromotionDecisionReport {
        requested_action: judge_action_name(requested_action).to_string(),
        basis: judge_basis_name(decision.basis).to_string(),
        reason: judge_reason_name(decision.reason).to_string(),
        promotion_allowed: decision.action_allowed,
        promotion_blocks_release: decision.blocks_release,
        release_blocked: !decision.action_allowed || decision.blocks_release,
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct TrustMeasurementReport {
    pub schema_version: u32,
    pub purpose: String,
    pub release_gate: ArsAccelerationReleaseGateReport,
    pub eval_gates: Vec<MeasurementGate>,
    pub index_consistency: IndexConsistencyReport,
    pub dedup_calibration: crate::ops::DedupThresholdObservability,
    pub judge_calibration: JudgeCalibrationObservabilityReport,
    pub background_observability: BackgroundObservabilityReport,
    pub active_learning: ActiveLearningReport,
}

pub fn collect(store: &SqliteStore, config: &ReinConfig) -> TrustMeasurementReport {
    let state =
        crate::store::adaptive::AdaptiveState::restore_snapshot(store.conn()).unwrap_or_default();
    let system = crate::ops::system_health::collect(store, config);
    let index_consistency = collect_index_consistency(store);
    let background_status = if system.status.ok && index_consistency.status == "ok" {
        "ok"
    } else {
        "attention"
    };

    TrustMeasurementReport {
        schema_version: TRUST_MEASUREMENT_SCHEMA_VERSION,
        purpose: "trust_measurement_platform_for_ars_rollout".to_string(),
        release_gate: crate::ops::ars_release_gate::ars_acceleration_release_gate_report(
            store, config,
        ),
        eval_gates: eval_gates(),
        index_consistency,
        dedup_calibration: crate::ops::dedup_threshold_observability_from_state(
            store, config, &state,
        ),
        judge_calibration: collect_judge_calibration_observability_from_state(
            store,
            config,
            &state,
            chrono::Utc::now().timestamp(),
        ),
        background_observability: BackgroundObservabilityReport {
            status: background_status.to_string(),
            adaptive_version: state.version,
            learned_shadow_fusion_buckets: state.learned_shadow_fusion.len(),
            feedback_event_count: count_table(store, "feedback_events"),
            consumer_offset_count: count_table(store, "consumer_offsets"),
            system,
        },
        active_learning: active_learning_report(store, config),
    }
}

pub fn collect_judge_calibration_observability(
    store: &SqliteStore,
    config: &ReinConfig,
    now: i64,
) -> JudgeCalibrationObservabilityReport {
    let state =
        crate::store::adaptive::AdaptiveState::restore_snapshot(store.conn()).unwrap_or_default();
    collect_judge_calibration_observability_from_state(store, config, &state, now)
}

fn collect_judge_calibration_observability_from_state(
    store: &SqliteStore,
    config: &ReinConfig,
    state: &crate::store::adaptive::AdaptiveState,
    now: i64,
) -> JudgeCalibrationObservabilityReport {
    use crate::store::adaptive::JudgeSurface;
    use crate::store::judge_structural_calibration::JudgeStructuralCalibrationLoadStatus;

    let calibration = state.judge_calibration_state.as_ref();
    let loaded =
        crate::store::judge_structural_calibration::load_judge_structural_calibration(store.conn());
    let structural_watermark = if loaded.status == JudgeStructuralCalibrationLoadStatus::Loaded {
        loaded.state.last_event_id
    } else {
        0
    };
    let human_runtime_watermark = calibration
        .map(|value| value.last_consumed_event_id_calibration)
        .unwrap_or(0);
    let last_runtime_computed_at = calibration.map(|value| value.last_computed_at).unwrap_or(0);
    let total_runtime_nightly_events = calibration
        .map(|value| value.total_offline_cron_events)
        .unwrap_or(0);
    let global_drift_alert_count = calibration
        .map(|value| value.judge_drift_alert)
        .unwrap_or(0);

    JudgeCalibrationObservabilityReport {
        synthesis: collect_judge_surface_calibration(
            store,
            config,
            calibration,
            &loaded,
            JudgeSurface::Synthesis,
            now,
        ),
        concept_summary: collect_judge_surface_calibration(
            store,
            config,
            calibration,
            &loaded,
            JudgeSurface::ConceptSummary,
            now,
        ),
        human_runtime_watermark,
        structural_watermark,
        last_runtime_computed_at,
        total_runtime_nightly_events,
        global_drift_alert_count,
    }
}

fn collect_judge_surface_calibration(
    store: &SqliteStore,
    config: &ReinConfig,
    calibration: Option<&crate::store::adaptive::JudgeCalibrationState>,
    loaded: &crate::store::judge_structural_calibration::JudgeStructuralCalibrationLoad,
    surface: crate::store::adaptive::JudgeSurface,
    now: i64,
) -> JudgeSurfaceCalibrationReport {
    use crate::config::JudgeStructuralAnchorMode;
    use crate::judge::contract::{JudgeStructuralStatus, JudgeTrustAction};
    use crate::store::adaptive::JudgeSurface;
    use crate::store::judge_structural_calibration::JudgeStructuralCalibrationLoadStatus;

    let human = crate::ops::ars_tuning::human_judge_calibration(calibration, surface);
    let (runtime_pairs, runtime_kappa, drift_alert_count) = match (calibration, surface) {
        (Some(calibration), JudgeSurface::Synthesis) => (
            calibration.recent_pairs_runtime_vs_offline_synthesis.len(),
            calibration.runtime_vs_offline_kappa_synthesis,
            calibration.judge_drift_alert_synthesis,
        ),
        (Some(calibration), JudgeSurface::ConceptSummary) => (
            calibration.recent_pairs_runtime_vs_offline_concept.len(),
            calibration.runtime_vs_offline_kappa_concept,
            calibration.judge_drift_alert_concept,
        ),
        (None, _) => (0, 0.0, 0),
    };
    let runtime_kappa = (runtime_pairs >= crate::store::adaptive::JUDGE_DRIFT_MIN_PAIRS
        && runtime_kappa.is_finite()
        && (-1.0..=1.0).contains(&runtime_kappa))
    .then_some(runtime_kappa);
    let structural =
        crate::ops::ars_tuning::resolve_judge_structural_trust(store.conn(), config, surface, now);
    let baseline_action = JudgeTrustAction::KeepConfiguredBaseline;
    let baseline_decision = crate::ops::ars_tuning::judge_trust_decision(
        calibration,
        surface,
        structural,
        baseline_action,
    );
    let recall_fusion_action = JudgeTrustAction::PromoteRecallFusion;
    let recall_fusion_decision = crate::ops::ars_tuning::judge_trust_decision(
        calibration,
        surface,
        structural,
        recall_fusion_action,
    );
    let surface_scope_action = JudgeTrustAction::PromoteJudgeScope;
    let surface_scope_decision = crate::ops::ars_tuning::judge_trust_decision(
        calibration,
        surface,
        structural,
        surface_scope_action,
    );
    let mode = match config.ars.llm_judge.structural_anchors.mode {
        JudgeStructuralAnchorMode::Off => "off",
        JudgeStructuralAnchorMode::Monitor => "monitor",
        JudgeStructuralAnchorMode::Enforce => "enforce",
    };
    let status = match structural.status {
        JudgeStructuralStatus::Disabled => "disabled",
        JudgeStructuralStatus::Collecting => "collecting",
        JudgeStructuralStatus::Ready => "ready",
        JudgeStructuralStatus::Failed => "failed",
        JudgeStructuralStatus::Stale => "stale",
        JudgeStructuralStatus::FingerprintMismatch => "fingerprint_mismatch",
        JudgeStructuralStatus::Corrupt => "corrupt",
        JudgeStructuralStatus::Unknown => "unknown",
    };
    let basis = judge_basis_name(baseline_decision.basis);
    let reason = judge_reason_name(baseline_decision.reason);
    let load_status = match loaded.status {
        JudgeStructuralCalibrationLoadStatus::Missing => "missing",
        JudgeStructuralCalibrationLoadStatus::Loaded => "loaded",
        JudgeStructuralCalibrationLoadStatus::Corrupt => "corrupt",
        JudgeStructuralCalibrationLoadStatus::UnsupportedSchema => "unsupported_schema",
        JudgeStructuralCalibrationLoadStatus::StorageError => "storage_error",
    };
    let expected_fingerprints = (structural.gate_required
        && config.ars.llm_judge.structural_anchors.mode != JudgeStructuralAnchorMode::Off)
        .then(|| crate::ops::llm_judge_worker::structural_fingerprints_for_config(config, surface))
        .flatten();
    let (expected_model_fingerprint, expected_rubric_fingerprint) = expected_fingerprints
        .map(|(model, rubric)| (Some(model), Some(rubric)))
        .unwrap_or((None, None));
    let observed = (loaded.status == JudgeStructuralCalibrationLoadStatus::Loaded)
        .then(|| loaded.state.surface(surface));
    let observed_string = |value: &str| (!value.is_empty()).then(|| value.to_string());
    let observed_model_fingerprint =
        observed.and_then(|value| observed_string(&value.model_fingerprint));
    let observed_rubric_fingerprint =
        observed.and_then(|value| observed_string(&value.rubric_fingerprint));
    let observed_probe_set_version =
        observed.and_then(|value| observed_string(&value.probe_set_version));
    let last_probe_at =
        observed.and_then(|value| (value.last_probe_at > 0).then_some(value.last_probe_at));
    let completed_at = observed.and_then(|value| value.completed_at);
    let interval_secs =
        i64::try_from(config.ars.llm_judge.structural_anchors.interval_secs).unwrap_or(i64::MAX);
    let fresh_until =
        completed_at.map(|value| value.saturating_add(interval_secs.saturating_mul(2)));
    let alert_count = observed.map(|value| value.alert_count).unwrap_or(0);
    let repair_advice = if structural.gate_required {
        judge_structural_repair_advice(
            config.ars.llm_judge.structural_anchors.mode,
            loaded.status,
            structural.status,
        )
    } else {
        Vec::new()
    };

    JudgeSurfaceCalibrationReport {
        human: JudgeHumanAgreementReport {
            pair_count: human.pair_count,
            kappa: human
                .kappa
                .filter(|kappa| kappa.is_finite() && (-1.0..=1.0).contains(kappa)),
        },
        runtime_vs_nightly: JudgeRuntimeVsNightlyReport {
            pair_count: runtime_pairs,
            kappa: runtime_kappa,
            drift_alert_count,
        },
        structural: JudgeStructuralReport {
            mode: mode.to_string(),
            load_status: load_status.to_string(),
            status: status.to_string(),
            requested_action: judge_action_name(baseline_action).to_string(),
            basis: basis.to_string(),
            reason: reason.to_string(),
            baseline_allowed: baseline_decision.action_allowed,
            configured_baseline_scale: baseline_decision.configured_baseline_scale,
            baseline_blocks_release: baseline_decision.blocks_release,
            recall_fusion_promotion: judge_promotion_report(
                recall_fusion_action,
                recall_fusion_decision,
            ),
            surface_scope_promotion: judge_promotion_report(
                surface_scope_action,
                surface_scope_decision,
            ),
            expected_model_fingerprint,
            observed_model_fingerprint,
            expected_rubric_fingerprint,
            observed_rubric_fingerprint,
            expected_probe_set_version:
                crate::ops::llm_judge_worker::JUDGE_STRUCTURAL_PROBE_SET_VERSION.to_string(),
            observed_probe_set_version,
            last_probe_at,
            completed_at,
            fresh_until,
            is_fresh: structural.status == JudgeStructuralStatus::Ready,
            alert_count,
            repair_advice,
        },
    }
}

fn judge_structural_repair_advice(
    mode: crate::config::JudgeStructuralAnchorMode,
    load_status: crate::store::judge_structural_calibration::JudgeStructuralCalibrationLoadStatus,
    status: crate::judge::contract::JudgeStructuralStatus,
) -> Vec<String> {
    use crate::config::JudgeStructuralAnchorMode;
    use crate::judge::contract::JudgeStructuralStatus;
    use crate::store::judge_structural_calibration::JudgeStructuralCalibrationLoadStatus;

    if mode == JudgeStructuralAnchorMode::Off {
        return Vec::new();
    }
    let advice = match load_status {
        JudgeStructuralCalibrationLoadStatus::Missing => {
            "Run the judge worker to create a sealed four-probe structural calibration run. This diagnostic is read-only and does not synthesize or repair calibration evidence automatically."
        }
        JudgeStructuralCalibrationLoadStatus::Corrupt => {
            "Preserve the unreadable structural calibration row for diagnosis, resolve the storage or producer fault, then run a fresh sealed four-probe suite. This diagnostic never deletes or rewrites the row."
        }
        JudgeStructuralCalibrationLoadStatus::UnsupportedSchema => {
            "Upgrade rein before evaluating this future-schema structural calibration row. Preserve the row unchanged; this diagnostic never rewrites downgrade-owned evidence."
        }
        JudgeStructuralCalibrationLoadStatus::StorageError => {
            "Restore read access to the calibration store and retry. This diagnostic is read-only and never mutates structural calibration state."
        }
        JudgeStructuralCalibrationLoadStatus::Loaded => match status {
            JudgeStructuralStatus::Ready | JudgeStructuralStatus::Disabled => return Vec::new(),
            JudgeStructuralStatus::Collecting => {
                "Let the sealed four-probe run finish, or rerun the judge worker after the configured interval. This diagnostic does not complete probes automatically."
            }
            JudgeStructuralStatus::Failed => {
                "Investigate the judge model and rubric, then run a fresh sealed four-probe suite. Failed evidence is preserved and never promoted or repaired automatically."
            }
            JudgeStructuralStatus::Stale => {
                "Run the judge worker to refresh the expired structural anchor suite. This diagnostic reports staleness but never refreshes evidence automatically."
            }
            JudgeStructuralStatus::FingerprintMismatch => {
                "Run the judge worker with the current model, rubric, and probe set to create a matching sealed suite. Mismatched evidence is preserved unchanged."
            }
            JudgeStructuralStatus::Corrupt => {
                "Preserve the structural calibration evidence for diagnosis, then run a fresh sealed suite after resolving the fault. This diagnostic never rewrites it."
            }
            JudgeStructuralStatus::Unknown => {
                "Run the judge worker to establish a complete sealed structural calibration run. This diagnostic does not infer missing evidence automatically."
            }
        },
    };
    vec![advice.to_string()]
}

/// v0.32 (T&M Phase 2): each `MeasurementGate` is now populated from a real
/// eval-gate harness round-trip — `docs/eval-baselines/{name}.json` (baseline
/// scorecard committed to the repo) compared against
/// `target/eval-gates/{name}-run.json` (last `rein-eval gate run` artifact)
/// via paired McNemar.  When either side is missing the status falls back to
/// `no_baseline` / `no_run` / `stub` so the report still serializes cleanly
/// for the doctor + GUI consumers.
///
/// The CWD used for scorecard lookup is `env::current_dir()` — the typical
/// invocation pattern is `cd source/rein && rein trust-measurement`.  In a
/// deployed binary running far from the source repo the scorecards won't be
/// found and every gate degrades to `no_baseline`; that's the honest answer
/// because nothing has been measured there.
fn eval_gates() -> Vec<MeasurementGate> {
    use crate::eval::gates::{self, ScorecardLoad, ScorecardStatus};
    let repo_root = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    let target_dir = repo_root.join("target");

    let mut gates_out = Vec::new();
    for gate in gates::all_gates() {
        let name = gate.name().to_string();
        let baseline_path = gates::baseline_path(&repo_root, &name);
        let run_path = gates::run_path(&target_dir, &name);

        // v0.32 R4 P2-#2: corrupt scorecard short-circuits to a
        // dedicated "corrupt" status — caller knows to repair the
        // file rather than treating it as merely absent.
        //
        // v0.32 R9 P2-#1: `signal` is serialized over the PUBLIC
        // `/api/trust-measurement` route, so it must NOT include
        // absolute filesystem paths, the process CWD, or any other
        // host-local detail.  Operators who need the full path can
        // get it via `rein doctor` (local-only).  We deliberately
        // drop the underlying `read_scorecard` error message from the
        // corrupt branch for the same reason — it embeds the
        // absolute scorecard path via `with_context(...)`.
        let baseline_load = gates::load_scorecard(&baseline_path);
        let current_load = gates::load_scorecard(&run_path);
        if matches!(baseline_load, ScorecardLoad::Corrupt(_)) {
            gates_out.push(MeasurementGate {
                name: name.clone(),
                status: "corrupt".to_string(),
                signal: "baseline scorecard exists but failed to parse — run \
                         `rein doctor` for the full diagnostic"
                    .to_string(),
                command: format!("rein-eval gate baseline --gate {name}"),
            });
            continue;
        }
        if matches!(current_load, ScorecardLoad::Corrupt(_)) {
            gates_out.push(MeasurementGate {
                name: name.clone(),
                status: "corrupt".to_string(),
                signal: "run scorecard exists but failed to parse — run \
                         `rein doctor` for the full diagnostic"
                    .to_string(),
                command: format!("rein-eval gate run --gate {name}"),
            });
            continue;
        }
        let baseline = match baseline_load {
            ScorecardLoad::Loaded(sc) => Some(sc),
            _ => None,
        };
        let current = match current_load {
            ScorecardLoad::Loaded(sc) => Some(sc),
            _ => None,
        };

        let (status, signal, command) = if gate.is_stub() {
            (
                "stub".to_string(),
                "placeholder for v0.32 — full gate impl deferred to v0.33+".to_string(),
                format!("rein-eval gate run --gate {name}"),
            )
        } else if baseline.is_none() {
            (
                "no_baseline".to_string(),
                // v0.32 R9 P2-#1: generic message — see comment block above.
                "no baseline scorecard committed for this gate".to_string(),
                format!("rein-eval gate baseline --gate {name}"),
            )
        } else if current.is_none() {
            (
                "no_run".to_string(),
                // v0.32 R9 P2-#1: generic message — see comment block above.
                "baseline exists; run scorecard not yet generated for this build".to_string(),
                format!("rein-eval gate run --gate {name}"),
            )
        } else {
            let cmp = gates::compare_scorecards(
                &name,
                baseline.as_ref(),
                current.as_ref(),
                gates::DEFAULT_NOISE_FLOOR,
            );
            let status = match cmp.status {
                ScorecardStatus::Ship => "ship".to_string(),
                ScorecardStatus::Bail => "bail".to_string(),
                ScorecardStatus::NoData => "no_data".to_string(),
            };
            (
                status,
                cmp.reason,
                format!("rein-eval gate compare --gate {name}"),
            )
        };

        gates_out.push(MeasurementGate {
            name,
            status,
            signal,
            command,
        });
    }
    gates_out
}

fn collect_index_consistency(store: &SqliteStore) -> IndexConsistencyReport {
    let active_memory_count = query_count(
        store,
        "SELECT COUNT(*) FROM memories WHERE status IN ('active','updated')",
    );
    let vector_row_count = query_count(store, "SELECT COUNT(*) FROM vec_memories");
    let missing_embeddings = query_count(
        store,
        "SELECT COUNT(*)
           FROM memories m
           LEFT JOIN vec_memories v ON v.id = m.id
          WHERE m.status IN ('active','updated')
            AND v.id IS NULL",
    );
    let orphan_embeddings = query_count(
        store,
        "SELECT COUNT(*)
           FROM vec_memories v
           LEFT JOIN memories m ON m.id = v.id
          WHERE m.id IS NULL",
    );
    let status = if missing_embeddings == 0 && orphan_embeddings == 0 {
        "ok"
    } else {
        "attention"
    };
    // v0.35 T&M Phase 3 codex P2 v4: `store::migrate::reindex` SELECTs
    // every row from `memories` regardless of status, so the empty-store
    // check that gates the orphan repair advice must look at the total
    // row count (active + deprecated + ...). Querying
    // `active_memory_count` here would mis-flag a deprecated-only store
    // as un-fixable when `rein migrate --reindex` would in fact clear
    // the orphans.
    let total_memory_count = query_count(store, "SELECT COUNT(*) FROM memories");
    let repair_advice = build_index_consistency_repair_advice(
        total_memory_count,
        missing_embeddings,
        orphan_embeddings,
    );
    IndexConsistencyReport {
        status: status.to_string(),
        active_memory_count,
        vector_row_count,
        missing_embeddings,
        orphan_embeddings,
        repair_advice,
    }
}

/// v0.35 T&M Phase 3 — translate the two index-consistency counters into
/// operator-facing repair advice. Returns an empty vec when both counters
/// are zero so consumers can use `repair_advice.is_empty()` as the
/// canonical "nothing to do" signal. Strings are pure semantic — no
/// paths, no secrets, safe for the public REST surface.
fn build_index_consistency_repair_advice(
    total_memory_count: u64,
    missing_embeddings: u64,
    orphan_embeddings: u64,
) -> Vec<String> {
    let mut advice = Vec::new();
    if missing_embeddings > 0 {
        // `rein migrate --reindex` drops the old `vec_memories` table,
        // re-creates it at the current embedding dimensions, and re-embeds
        // every memory row. Documented in `store::migrate::reindex`.
        // `rein doctor --fix` is deliberately NOT cited: it only rebuilds
        // Tantivy + HNSW side indexes from existing rows.
        advice.push(format!(
            "{missing_embeddings} memories have no vector embedding row — run `rein migrate --reindex` to drop `vec_memories`, recreate it at the current embedding dimensions, and re-embed every memory (single atomic repair path)"
        ));
    }
    if orphan_embeddings > 0 {
        if total_memory_count == 0 {
            // Codex P2 v3 edge case: `store::migrate::reindex` returns
            // early when `total == 0` (no memories to embed), so it does
            // NOT drop/recreate `vec_memories`. An orphan-only state on an
            // empty store therefore can't be cleared via the public CLI.
            //
            // Codex P2 v4: the empty-store check must use the TOTAL row
            // count (any status), not active+updated, because `reindex`
            // SELECTs every row from `memories`. A deprecated-only store
            // still has rows for reindex to drop+recreate from.
            advice.push(format!(
                "{orphan_embeddings} vector rows have no matching memory AND the memories table is empty — `rein migrate --reindex` returns early on empty stores and will NOT clear these orphans. No public CLI handles this edge case; track via the project issue tracker under index consistency."
            ));
        } else {
            // Standard path: memories non-empty → `reindex` drops +
            // recreates vec_memories, atomically discarding orphans.
            advice.push(format!(
                "{orphan_embeddings} vector rows have no matching memory — run `rein migrate --reindex` (the drop-and-recreate of `vec_memories` discards rows that no longer have a corresponding memory). `rein gc` alone does NOT remove pre-existing orphans."
            ));
        }
    }
    advice
}

fn active_learning_report(store: &SqliteStore, config: &ReinConfig) -> ActiveLearningReport {
    let active = config.ars.llm_judge.enabled
        && (config.ars.llm_judge.synthesis_enabled || config.ars.llm_judge.concept_summary_enabled)
        && config.ars.llm_judge.nightly_cron.enabled;
    let judge_drift_alert_total = collect_judge_drift_alert_total(store);
    ActiveLearningReport {
        status: if active { "active" } else { "disabled" }.to_string(),
        llm_judge_enabled: config.ars.llm_judge.enabled,
        synthesis_enabled: config.ars.llm_judge.synthesis_enabled,
        concept_summary_enabled: config.ars.llm_judge.concept_summary_enabled,
        nightly_cron_enabled: config.ars.llm_judge.nightly_cron.enabled,
        sample_rate_cold_start: config.ars.llm_judge.sample_rate_cold_start,
        sample_rate_warm: config.ars.llm_judge.sample_rate_warm,
        nightly_cron_sample_rate: config.ars.llm_judge.nightly_cron.sample_rate,
        judge_drift_alert_total,
    }
}

/// v0.35 T&M Phase 3 — pull the three per-surface drift counters from the
/// judge calibration state on the AdaptiveState snapshot and aggregate
/// them into a single observability number. Returns 0 when no calibration
/// row exists yet (fresh install or LLM judge never enabled). The doctor
/// surface continues to read each counter separately for per-surface
/// diagnosis.
fn collect_judge_drift_alert_total(store: &SqliteStore) -> u64 {
    let state =
        crate::store::adaptive::AdaptiveState::restore_snapshot(store.conn()).unwrap_or_default();
    let Some(cal) = state.judge_calibration_state.as_ref() else {
        return 0;
    };
    cal.judge_drift_alert
        .saturating_add(cal.judge_drift_alert_synthesis)
        .saturating_add(cal.judge_drift_alert_concept)
}

fn count_table(store: &SqliteStore, table: &str) -> u64 {
    let sql = format!("SELECT COUNT(*) FROM {table}");
    query_count(store, &sql)
}

fn query_count(store: &SqliteStore, sql: &str) -> u64 {
    store
        .conn()
        .query_row(sql, [], |row| row.get::<_, i64>(0))
        .map(|n| n.max(0) as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ReinConfig;
    use crate::store::SqliteStore;
    use tempfile::tempdir;

    #[test]
    fn trust_measurement_report_surfaces_eval_gates_and_observability() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("memories.db");
        let mut config = ReinConfig::default();
        config.database.path = db_path.to_string_lossy().into_owned();
        // v0.28.7 H0 — defaults reverted to `false`. Explicit opt-in here so the
        // test still exercises the "report mirrors enabled-state" path.
        config.ars.llm_judge.enabled = true;
        config.ars.llm_judge.nightly_cron.enabled = true;
        let store = SqliteStore::new(&db_path, "text-embedding-3-small", 3072).unwrap();

        let state = crate::store::adaptive::AdaptiveState {
            global_dedup_threshold: 0.40,
            version: 1,
            ..Default::default()
        };
        state.save_snapshot(store.conn()).unwrap();

        let report = collect(&store, &config);
        let report_json = serde_json::to_value(&report).unwrap();

        assert_eq!(report.schema_version, TRUST_MEASUREMENT_SCHEMA_VERSION);
        assert_eq!(report.schema_version, 3);
        assert_eq!(report.judge_calibration.synthesis.human.pair_count, 0);
        assert_eq!(report.judge_calibration.synthesis.human.kappa, None);
        assert_eq!(
            report
                .judge_calibration
                .synthesis
                .runtime_vs_nightly
                .pair_count,
            0
        );
        assert_eq!(report.judge_calibration.synthesis.structural.mode, "off");
        assert_eq!(
            report.judge_calibration.synthesis.structural.status,
            "disabled"
        );
        assert_eq!(
            report.judge_calibration.synthesis.structural.basis,
            "configured_baseline"
        );
        for gate in ["recall", "dedup", "admission", "latency"] {
            assert!(
                report.eval_gates.iter().any(|item| item.name == gate),
                "missing eval gate {gate}"
            );
        }
        assert_eq!(report.index_consistency.missing_embeddings, 0);
        // v0.35 T&M Phase 3: fresh stores have no orphans / no missing,
        // so the advice vector must be empty.
        assert!(
            report.index_consistency.repair_advice.is_empty(),
            "fresh store should not emit repair advice; got {:?}",
            report.index_consistency.repair_advice
        );
        assert!(report.background_observability.system.status.ok);
        assert!(report.active_learning.llm_judge_enabled);
        assert!(report.active_learning.nightly_cron_enabled);
        // No judge calibration row yet on this fresh store.
        assert_eq!(report.active_learning.judge_drift_alert_total, 0);
        let dedup = &report_json["dedup_calibration"];
        assert_eq!(dedup["source"], "legacy_unlabeled_shadow");
        assert_eq!(dedup["static_threshold"], 0.70);
        assert_eq!(dedup["shadow_threshold"], 0.40);
        assert_eq!(dedup["hard_effective_threshold"], 0.70);
        assert_eq!(dedup["calibration"]["adaptive_enabled"], true);
        assert_eq!(dedup["calibration"]["evidence_verified"], false);
        assert_eq!(dedup["calibration"]["applied"], false);
        assert_eq!(dedup["calibration"]["reason"], "no_powered_terminal_policy");
        assert!(dedup["calibration"]["policy"]["utility"].is_object());
        assert!(dedup["calibration"]["policy"]["required_slices"].is_array());
        assert!(dedup["calibration"]["counterfactual_counts"].is_object());
        assert!(dedup["repair_advice"].is_array());
        let advice = dedup["repair_advice"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(serde_json::Value::as_str)
            .collect::<Vec<_>>()
            .join(" ");
        assert!(advice.contains("recalibrat"), "advice={advice}");
        assert!(advice.contains("shadow"), "advice={advice}");
    }

    #[test]
    fn corrupt_dedup_policy_advice_requires_atomic_reset_and_recalibration() {
        let store = SqliteStore::in_memory().unwrap();
        let config = ReinConfig::default();
        store
            .conn()
            .execute(
                "INSERT INTO metadata(key, value) VALUES (?1, ?2)",
                rusqlite::params![
                    crate::store::dedup_calibration::DEDUP_CALIBRATION_METADATA_KEY,
                    "{not json"
                ],
            )
            .unwrap();

        let report = serde_json::to_value(collect(&store, &config)).unwrap();
        let serialized = report.to_string();
        assert!(!serialized.contains("{not json"));
        assert!(!serialized.contains("/Users/"));
        assert!(!serialized.contains("/home/"));
        let advice = report["dedup_calibration"]["repair_advice"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(serde_json::Value::as_str)
            .collect::<Vec<_>>()
            .join(" ");
        assert!(advice.contains("recalibrat"), "advice={advice}");
        assert!(advice.contains("atomically"), "advice={advice}");
        assert!(!advice.contains("doctor --fix"), "advice={advice}");
        assert!(!advice.contains("automatically"), "advice={advice}");
    }

    #[test]
    fn judge_calibration_observability_keeps_human_drift_and_structural_evidence_distinct() {
        use crate::config::JudgeStructuralAnchorMode;
        use crate::store::adaptive::{
            JudgeCalibrationState, JudgeStructuralProbeKind, JudgeSurface,
        };
        use crate::store::judge_structural_calibration::{
            compare_and_swap_judge_structural_calibration, JudgeStructuralCalibrationState,
            JudgeStructuralSurfaceState,
        };

        let store = SqliteStore::in_memory().unwrap();
        let mut config = ReinConfig::default();
        config.ars.llm_judge.enabled = true;
        config.ars.llm_judge.synthesis_enabled = true;
        config.ars.llm_judge.concept_summary_enabled = true;
        config.ars.llm_judge.structural_anchors.mode = JudgeStructuralAnchorMode::Enforce;
        config.ars.llm_judge.structural_anchors.interval_secs = 100;
        config.llm.provider = "omlx".to_string();
        config.llm.omlx.model = Some("judge-observability-model".to_string());

        let mut calibration = JudgeCalibrationState {
            runtime_vs_offline_kappa_synthesis: 0.88,
            runtime_vs_offline_kappa_concept: 0.79,
            last_consumed_event_id_calibration: 77,
            total_offline_cron_events: 60,
            last_computed_at: 9_990,
            judge_drift_alert: 3,
            ..JudgeCalibrationState::default()
        };
        for index in 0_i64..30 {
            let value = index % 2 == 0;
            calibration.push_pair(JudgeSurface::Synthesis, value, value, 9_900 + index);
            calibration
                .recent_pairs_runtime_vs_offline_synthesis
                .push_back((value, value, 9_900 + index));
            calibration
                .recent_pairs_runtime_vs_offline_concept
                .push_back((value, !value, 9_900 + index));
        }
        crate::store::adaptive::AdaptiveState {
            judge_calibration_state: Some(calibration),
            version: 1,
            ..Default::default()
        }
        .save_snapshot(store.conn())
        .unwrap();

        let secret_token_hash = "s".repeat(64);
        let ready_surface = |surface| {
            let (model_fingerprint, rubric_fingerprint) =
                crate::ops::llm_judge_worker::structural_fingerprints_for_config(&config, surface)
                    .unwrap();
            JudgeStructuralSurfaceState {
                run_id: Some(format!("run-{surface:?}")),
                probe_set_version: crate::ops::llm_judge_worker::JUDGE_STRUCTURAL_PROBE_SET_VERSION
                    .to_string(),
                model_fingerprint,
                rubric_fingerprint,
                run_token_hashes: JudgeStructuralProbeKind::ALL
                    .into_iter()
                    .map(|kind| (kind, secret_token_hash.clone()))
                    .collect(),
                seen_kinds: JudgeStructuralProbeKind::ALL.into_iter().collect(),
                run_started_at: 9_900,
                last_probe_at: 9_950,
                completed_at: Some(9_950),
                status: crate::judge::contract::JudgeStructuralStatus::Ready,
                ..JudgeStructuralSurfaceState::default()
            }
        };
        let structural_state = JudgeStructuralCalibrationState {
            revision: 1,
            last_event_id: 88,
            synthesis: ready_surface(JudgeSurface::Synthesis),
            concept_summary: ready_surface(JudgeSurface::ConceptSummary),
            updated_at: 9_950,
            ..JudgeStructuralCalibrationState::default()
        };
        assert!(
            compare_and_swap_judge_structural_calibration(store.conn(), &structural_state, 0)
                .unwrap()
        );

        let report = collect_judge_calibration_observability(&store, &config, 10_000);
        assert_eq!(report.human_runtime_watermark, 77);
        assert_eq!(report.structural_watermark, 88);
        assert_eq!(report.last_runtime_computed_at, 9_990);
        assert_eq!(report.total_runtime_nightly_events, 60);
        assert_eq!(report.global_drift_alert_count, 3);
        assert_eq!(report.synthesis.human.pair_count, 30);
        assert_eq!(report.synthesis.human.kappa, Some(1.0));
        assert_eq!(report.concept_summary.human.pair_count, 0);
        assert_eq!(report.concept_summary.human.kappa, None);
        assert_eq!(report.synthesis.runtime_vs_nightly.pair_count, 30);
        assert_eq!(report.synthesis.runtime_vs_nightly.kappa, Some(0.88));
        assert_eq!(report.concept_summary.runtime_vs_nightly.kappa, Some(0.79));

        let structural = &report.synthesis.structural;
        assert_eq!(structural.load_status, "loaded");
        assert_eq!(structural.status, "ready");
        assert_eq!(structural.requested_action, "keep_configured_baseline");
        assert_eq!(structural.basis, "human_kappa");
        assert_eq!(structural.reason, "human_kappa_healthy");
        assert!(structural.baseline_allowed);
        assert_eq!(structural.configured_baseline_scale, 1.0);
        assert!(!structural.baseline_blocks_release);
        assert_eq!(
            structural.expected_model_fingerprint,
            structural.observed_model_fingerprint
        );
        assert_eq!(
            structural.expected_rubric_fingerprint,
            structural.observed_rubric_fingerprint
        );
        assert_eq!(
            structural.expected_probe_set_version,
            crate::ops::llm_judge_worker::JUDGE_STRUCTURAL_PROBE_SET_VERSION
        );
        assert_eq!(
            structural.observed_probe_set_version.as_deref(),
            Some(crate::ops::llm_judge_worker::JUDGE_STRUCTURAL_PROBE_SET_VERSION)
        );
        assert_eq!(structural.last_probe_at, Some(9_950));
        assert_eq!(structural.completed_at, Some(9_950));
        assert_eq!(structural.fresh_until, Some(10_150));
        assert!(structural.is_fresh);
        assert!(structural.repair_advice.is_empty());
        let concept_structural = &report.concept_summary.structural;
        assert_eq!(concept_structural.basis, "structural_anchors");
        assert!(concept_structural.baseline_allowed);
        assert!(!concept_structural.baseline_blocks_release);
        assert_eq!(
            concept_structural.recall_fusion_promotion.requested_action,
            "promote_recall_fusion"
        );
        assert!(!concept_structural.recall_fusion_promotion.promotion_allowed);
        assert!(concept_structural.recall_fusion_promotion.release_blocked);
        assert_eq!(
            concept_structural.surface_scope_promotion.requested_action,
            "promote_judge_scope"
        );
        assert!(!concept_structural.surface_scope_promotion.promotion_allowed);
        assert!(concept_structural.surface_scope_promotion.release_blocked);

        let serialized = serde_json::to_string(&report).unwrap();
        assert!(!serialized.contains(&secret_token_hash));
        assert!(!serialized.contains("run-Synthesis"));
    }

    #[test]
    fn judge_human_kappa_projection_omits_non_finite_or_out_of_range_values() {
        use crate::store::adaptive::{AdaptiveState, JudgeCalibrationState, JudgeSurface};

        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1.01, -1.01] {
            let store = SqliteStore::in_memory().unwrap();
            let mut config = ReinConfig::default();
            config.ars.llm_judge.enabled = true;
            config.ars.llm_judge.synthesis_enabled = true;

            let mut calibration = JudgeCalibrationState::default();
            for timestamp in 0..crate::store::adaptive::LLM_JUDGE_J3_MIN_PAIRS as i64 {
                calibration.push_pair(JudgeSurface::Synthesis, true, true, timestamp);
            }
            calibration.kappa = invalid;
            let state = AdaptiveState {
                judge_calibration_state: Some(calibration),
                ..AdaptiveState::default()
            };

            let report =
                collect_judge_calibration_observability_from_state(&store, &config, &state, 1_000);
            assert_eq!(
                report.synthesis.human.pair_count,
                crate::store::adaptive::LLM_JUDGE_J3_MIN_PAIRS
            );
            assert_eq!(report.synthesis.human.kappa, None, "invalid={invalid}");
        }
    }

    #[test]
    fn judge_structural_repair_advice_is_status_specific_read_only_and_redacted() {
        use crate::config::JudgeStructuralAnchorMode;
        use crate::store::adaptive::{JudgeStructuralProbeKind, JudgeSurface};
        use crate::store::judge_structural_calibration::{
            compare_and_swap_judge_structural_calibration, JudgeStructuralCalibrationState,
            JudgeStructuralSurfaceState, JUDGE_STRUCTURAL_CALIBRATION_METADATA_KEY,
        };

        let config_for = |mode, model: &str| {
            let mut config = ReinConfig::default();
            config.ars.llm_judge.enabled = true;
            config.ars.llm_judge.synthesis_enabled = true;
            config.ars.llm_judge.concept_summary_enabled = false;
            config.ars.llm_judge.structural_anchors.mode = mode;
            config.ars.llm_judge.structural_anchors.interval_secs = 100;
            config.llm.provider = "omlx".to_string();
            config.llm.omlx.model = Some(model.to_string());
            config
        };
        let ready_state = |config: &ReinConfig, completed_at: i64| {
            let (model_fingerprint, rubric_fingerprint) =
                crate::ops::llm_judge_worker::structural_fingerprints_for_config(
                    config,
                    JudgeSurface::Synthesis,
                )
                .unwrap();
            JudgeStructuralCalibrationState {
                revision: 1,
                synthesis: JudgeStructuralSurfaceState {
                    run_id: Some("operator-run-id-must-not-leak".to_string()),
                    probe_set_version:
                        crate::ops::llm_judge_worker::JUDGE_STRUCTURAL_PROBE_SET_VERSION.to_string(),
                    model_fingerprint,
                    rubric_fingerprint,
                    run_token_hashes: JudgeStructuralProbeKind::ALL
                        .into_iter()
                        .map(|kind| (kind, "t".repeat(64)))
                        .collect(),
                    seen_kinds: JudgeStructuralProbeKind::ALL.into_iter().collect(),
                    run_started_at: completed_at - 10,
                    last_probe_at: completed_at,
                    completed_at: Some(completed_at),
                    status: crate::judge::contract::JudgeStructuralStatus::Ready,
                    ..JudgeStructuralSurfaceState::default()
                },
                updated_at: completed_at,
                ..JudgeStructuralCalibrationState::default()
            }
        };

        for (mode, expected_status, expected_basis, expected_scale, has_advice) in [
            (
                JudgeStructuralAnchorMode::Off,
                "disabled",
                "configured_baseline",
                1.0,
                false,
            ),
            (
                JudgeStructuralAnchorMode::Monitor,
                "unknown",
                "configured_baseline",
                1.0,
                true,
            ),
            (
                JudgeStructuralAnchorMode::Enforce,
                "unknown",
                "untrusted",
                0.0,
                true,
            ),
        ] {
            let store = SqliteStore::in_memory().unwrap();
            let config = config_for(mode, "judge-model-a");
            let report = collect_judge_calibration_observability(&store, &config, 1_000);
            let structural = &report.synthesis.structural;
            assert_eq!(structural.load_status, "missing");
            assert_eq!(structural.status, expected_status);
            assert_eq!(structural.basis, expected_basis);
            assert_eq!(structural.configured_baseline_scale, expected_scale);
            assert_eq!(!structural.repair_advice.is_empty(), has_advice);
            assert!(
                report.concept_summary.structural.repair_advice.is_empty(),
                "a disabled judge surface must not request structural repair"
            );
        }

        for (raw, load_status, advice_word) in [
            (
                "{operator-secret-corrupt-payload".to_string(),
                "corrupt",
                "Preserve",
            ),
            (
                serde_json::json!({
                    "schema_version": 9999,
                    "revision": 7,
                    "operator_secret": "future-secret-must-not-leak"
                })
                .to_string(),
                "unsupported_schema",
                "Upgrade",
            ),
        ] {
            let store = SqliteStore::in_memory().unwrap();
            store
                .conn()
                .execute(
                    "INSERT INTO metadata(key, value) VALUES (?1, ?2)",
                    rusqlite::params![JUDGE_STRUCTURAL_CALIBRATION_METADATA_KEY, &raw],
                )
                .unwrap();
            let config = config_for(JudgeStructuralAnchorMode::Enforce, "judge-model-a");
            let report = collect_judge_calibration_observability(&store, &config, 1_000);
            let structural = &report.synthesis.structural;
            assert_eq!(structural.load_status, load_status);
            assert!(
                structural.repair_advice.join(" ").contains(advice_word),
                "advice={:?}",
                structural.repair_advice
            );
            let serialized = serde_json::to_string(&report).unwrap();
            assert!(!serialized.contains("operator-secret"));
            assert!(!serialized.contains("future-secret"));
            assert!(!serialized.contains(&raw));
            assert!(!serialized.contains("doctor --fix"));
        }

        let stale_store = SqliteStore::in_memory().unwrap();
        let stale_config = config_for(JudgeStructuralAnchorMode::Enforce, "judge-model-a");
        let stale_state = ready_state(&stale_config, 1_000);
        assert!(
            compare_and_swap_judge_structural_calibration(stale_store.conn(), &stale_state, 0)
                .unwrap()
        );
        let stale = collect_judge_calibration_observability(&stale_store, &stale_config, 1_201);
        assert_eq!(stale.synthesis.structural.status, "stale");
        assert!(stale
            .synthesis
            .structural
            .repair_advice
            .join(" ")
            .contains("refresh"));

        let mismatch_store = SqliteStore::in_memory().unwrap();
        let original_config = config_for(JudgeStructuralAnchorMode::Enforce, "judge-model-a");
        let mismatch_state = ready_state(&original_config, 1_000);
        assert!(compare_and_swap_judge_structural_calibration(
            mismatch_store.conn(),
            &mismatch_state,
            0
        )
        .unwrap());
        let changed_config = config_for(JudgeStructuralAnchorMode::Enforce, "judge-model-b");
        let mismatch =
            collect_judge_calibration_observability(&mismatch_store, &changed_config, 1_100);
        assert_eq!(mismatch.synthesis.structural.status, "fingerprint_mismatch");
        assert!(mismatch
            .synthesis
            .structural
            .repair_advice
            .join(" ")
            .contains("current model"));
    }

    #[test]
    fn orphan_dedup_seal_is_reported_as_half_written_bundle() {
        let store = SqliteStore::in_memory().unwrap();
        let config = ReinConfig::default();
        let now = chrono::Utc::now().timestamp();
        let seal = crate::store::dedup_calibration::DedupCalibrationSeal {
            schema_version: crate::store::dedup_calibration::DEDUP_CALIBRATION_SCHEMA_VERSION,
            revision: 1,
            generation: 1,
            cutoff: now - 1,
            scale: crate::store::dedup_calibration::DedupCalibrationScale::Lexical,
            configured_static_threshold_bits: (0.70_f32).to_bits(),
            train_fingerprint: "train".to_string(),
            holdout_fingerprint: "holdout".to_string(),
            corpus_fingerprint: "corpus".to_string(),
            policy_digest: "policy-digest".to_string(),
            calibrated_at: now - 1,
            valid_until: now + 3_600,
        };
        store
            .conn()
            .execute(
                "INSERT INTO metadata(key, value) VALUES (?1, ?2)",
                rusqlite::params![
                    crate::store::dedup_calibration::DEDUP_CALIBRATION_SEAL_METADATA_KEY,
                    serde_json::to_string(&seal).unwrap()
                ],
            )
            .unwrap();

        let report = serde_json::to_value(collect(&store, &config)).unwrap();
        let calibration = &report["dedup_calibration"]["calibration"];
        assert_eq!(calibration["load_status"], "missing");
        assert_eq!(calibration["seal_status"], "loaded");
        assert_eq!(calibration["reason"], "orphan_seal_policy_missing");
        let advice = report["dedup_calibration"]["repair_advice"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(serde_json::Value::as_str)
            .collect::<Vec<_>>()
            .join(" ");
        assert!(advice.contains("atomically"), "advice={advice}");
        assert!(advice.contains("recalibrat"), "advice={advice}");
    }

    // v0.35 T&M Phase 3 — index-consistency repair advice builder.
    #[test]
    fn repair_advice_is_empty_when_counts_are_zero() {
        let advice = build_index_consistency_repair_advice(100, 0, 0);
        assert!(advice.is_empty());
    }

    #[test]
    fn repair_advice_flags_missing_embeddings_with_migrate_reindex() {
        let advice = build_index_consistency_repair_advice(50, 3, 0);
        assert_eq!(advice.len(), 1);
        assert!(advice[0].contains("3"));
        // Codex P2 v3: `rein migrate --reindex` is the documented atomic
        // repair (drop + recreate vec_memories + re-embed everything).
        assert!(advice[0].contains("rein migrate --reindex"));
        assert!(advice[0].contains("vec_memories"));
        // Known no-ops must not appear.
        assert!(!advice[0].contains("doctor --fix"));
        assert!(!advice[0].contains("rein worker memory"));
        assert!(!advice[0].contains("rein consolidate"));
    }

    #[test]
    fn repair_advice_flags_orphan_embeddings_with_migrate_reindex_when_memories_present() {
        let advice = build_index_consistency_repair_advice(50, 0, 7);
        assert_eq!(advice.len(), 1);
        assert!(advice[0].contains("7"));
        assert!(advice[0].contains("rein migrate --reindex"));
        assert!(advice[0].contains("does NOT remove pre-existing orphans"));
    }

    #[test]
    fn repair_advice_flags_orphan_only_on_empty_store_as_unfixable_via_cli() {
        // Codex P2 v3 edge case: `reindex` returns early on empty stores
        // so following `rein migrate --reindex` here would be a no-op.
        // The advice must say so + point at the backlog.
        let advice = build_index_consistency_repair_advice(0, 0, 4);
        assert_eq!(advice.len(), 1);
        assert!(advice[0].contains("4"));
        assert!(advice[0].contains("memories table is empty"));
        assert!(advice[0].contains("returns early"));
        assert!(advice[0].contains("issue tracker"));
        // Critically, the empty-store branch must NOT direct operators
        // toward the no-op `rein migrate --reindex` invocation.
        assert!(
            advice[0].contains("will NOT clear"),
            "advice must surface the reindex no-op for empty stores: {}",
            advice[0]
        );
    }

    #[test]
    fn repair_advice_recommends_reindex_for_deprecated_only_store_with_orphans() {
        // Codex P2 v4: the empty-store check must use TOTAL memory rows,
        // not just active+updated. A deprecated-only store still has
        // rows for `reindex` to clear orphans from. Total > 0 with
        // orphans > 0 must take the standard reindex branch.
        let advice = build_index_consistency_repair_advice(5, 0, 3);
        assert_eq!(advice.len(), 1);
        assert!(advice[0].contains("3"));
        assert!(advice[0].contains("rein migrate --reindex"));
        assert!(advice[0].contains("does NOT remove pre-existing orphans"));
        // The empty-store no-op branch must NOT fire.
        assert!(!advice[0].contains("memories table is empty"));
        assert!(!advice[0].contains("returns early"));
    }

    #[test]
    fn repair_advice_emits_both_lines_when_both_counters_are_nonzero() {
        let advice = build_index_consistency_repair_advice(50, 2, 5);
        assert_eq!(advice.len(), 2);
        assert!(advice
            .iter()
            .any(|s| s.contains("rein migrate --reindex") && s.contains("2")));
        assert!(advice
            .iter()
            .any(|s| s.contains("rein migrate --reindex") && s.contains("5")));
    }

    #[test]
    fn repair_advice_strings_carry_no_paths_or_secrets() {
        // The advice vector is exposed on the public REST surface, so it
        // must not embed host-local paths, tokens, or any
        // deployment-specific identifiers. The builder strings should not
        // contain `/Users/`, `/home/`, `/var/`, or env-shaped tokens.
        for (m, miss, orph) in [(50, 1, 1), (0, 0, 1)] {
            let advice = build_index_consistency_repair_advice(m, miss, orph);
            for line in &advice {
                assert!(!line.contains("/Users/"), "leaked $HOME path: {line}");
                assert!(!line.contains("/home/"), "leaked $HOME path: {line}");
                assert!(!line.contains("/var/"), "leaked tmp/cache path: {line}");
                assert!(!line.contains("REIN_HTTP_TOKEN"), "leaked env name: {line}");
                assert!(!line.contains("GEMINI_API_KEY"), "leaked env name: {line}");
            }
        }
    }
}
