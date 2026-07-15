//! Shared, side-effect-free activation logic for A12 recall fusion.

use crate::eval::gates::{self, ScorecardLoad, ScorecardStatus};
use crate::store::a12_calibration::{
    A12CalibrationLoad, A12CalibrationLoadStatus, A12CalibrationPhase, A12CalibrationVerdict,
    A12FusionSimplex, A12ScopeEntry,
};
use crate::store::adaptive::{AdaptiveState, LearnedShadowFusionEntry};
use crate::store::ars_parameter_policy::{
    ArsParameterPolicy, ArsParameterPolicyMode, ArsRecallFusionEvidence,
    ArsRecallFusionEvidenceBasis, ArsRecallGateStatus, ARS_PARAMETER_POLICY_SCHEMA_VERSION,
};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// Absolute source/evidence root used by installed daemons that do not run
/// with a repository as their working directory. The root keeps the existing
/// `docs/eval-baselines` + `target/eval-gates` layout, so `rein-eval` can
/// produce the artifacts simply by running from this directory.
pub const REIN_EVAL_GATE_ROOT_ENV: &str = "REIN_EVAL_GATE_ROOT";

/// Immutable identity of the recall eval-gate decision used by policy refresh.
/// Runtime consumes the sealed fields from the policy and never re-reads files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RecallEvalGateReasonCode {
    Compared,
    MissingBaseline,
    MissingRun,
    CorruptBaseline,
    CorruptRun,
    ArtifactRootUnconfigured,
    ArtifactRootRelative,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RecallEvalGateAttestation {
    pub status: ArsRecallGateStatus,
    pub reason_code: RecallEvalGateReasonCode,
    pub build_fingerprint: Option<String>,
    pub fixture_fingerprint: Option<String>,
    pub evaluated_at: Option<i64>,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RecallEvalGateArtifactPaths {
    baseline: PathBuf,
    run: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecallEvalGatePathError {
    RelativeOperatorRoot,
    Unconfigured,
}

fn find_recall_eval_gate_root(start: &Path) -> Option<PathBuf> {
    start
        .ancestors()
        .find(|candidate| candidate.join("docs/eval-baselines").is_dir())
        .map(Path::to_path_buf)
}

/// Resolve scorecards without treating an arbitrary process cwd as trusted
/// evidence. An explicit absolute operator root wins. Development runs may
/// discover a checkout ancestor from cwd. Installed daemons outside a checkout
/// fail closed with a configuration hint instead of retaining a private build
/// path in the binary or reading unrelated files.
fn resolve_recall_eval_gate_artifact_paths(
    operator_root: Option<&Path>,
    current_dir: &Path,
) -> Result<RecallEvalGateArtifactPaths, RecallEvalGatePathError> {
    let root = if let Some(root) = operator_root {
        if !root.is_absolute() {
            return Err(RecallEvalGatePathError::RelativeOperatorRoot);
        }
        root.to_path_buf()
    } else if let Some(root) = find_recall_eval_gate_root(current_dir) {
        root
    } else {
        return Err(RecallEvalGatePathError::Unconfigured);
    };
    Ok(RecallEvalGateArtifactPaths {
        baseline: gates::baseline_path(&root, "recall"),
        run: gates::run_path(&root.join("target"), "recall"),
    })
}

/// Load current recall scorecards through the shared stable artifact resolver.
/// The returned attestation contains semantic identity only; neither the root
/// nor either scorecard path is retained.
pub fn current_recall_eval_gate_attestation(noise_floor: f64) -> RecallEvalGateAttestation {
    let current_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let operator_root = std::env::var_os(REIN_EVAL_GATE_ROOT_ENV).map(PathBuf::from);
    let paths = match resolve_recall_eval_gate_artifact_paths(
        operator_root.as_deref(),
        &current_dir,
    ) {
        Ok(paths) => paths,
        Err(RecallEvalGatePathError::RelativeOperatorRoot) => {
            return RecallEvalGateAttestation {
                status: ArsRecallGateStatus::NoData,
                reason_code: RecallEvalGateReasonCode::ArtifactRootRelative,
                build_fingerprint: None,
                fixture_fingerprint: None,
                evaluated_at: None,
                reason: format!(
                    "{REIN_EVAL_GATE_ROOT_ENV} must be an absolute path to an eval-gate root"
                ),
            };
        }
        Err(RecallEvalGatePathError::Unconfigured) => {
            return RecallEvalGateAttestation {
                status: ArsRecallGateStatus::NoData,
                reason_code: RecallEvalGateReasonCode::ArtifactRootUnconfigured,
                build_fingerprint: None,
                fixture_fingerprint: None,
                evaluated_at: None,
                reason: format!(
                    "recall eval-gate root is unconfigured; set {REIN_EVAL_GATE_ROOT_ENV} to an absolute source/evidence root"
                ),
            };
        }
    };
    recall_eval_gate_attestation(&paths.baseline, &paths.run, noise_floor)
}

/// Load and compare caller-supplied recall scorecard paths. Supplying paths is
/// intentional: policy refresh, release-gate tests, and alternate target dirs
/// must attest the same artifacts rather than rely on process cwd.
pub fn recall_eval_gate_attestation(
    baseline_path: &Path,
    run_path: &Path,
    noise_floor: f64,
) -> RecallEvalGateAttestation {
    let baseline_load = gates::load_scorecard(baseline_path);
    let current_load = gates::load_scorecard(run_path);

    if matches!(&baseline_load, ScorecardLoad::Corrupt(_)) {
        return RecallEvalGateAttestation {
            status: ArsRecallGateStatus::NoData,
            reason_code: RecallEvalGateReasonCode::CorruptBaseline,
            build_fingerprint: None,
            fixture_fingerprint: None,
            evaluated_at: None,
            reason: "recall baseline scorecard is corrupt".to_string(),
        };
    }
    if matches!(&current_load, ScorecardLoad::Corrupt(_)) {
        return RecallEvalGateAttestation {
            status: ArsRecallGateStatus::NoData,
            reason_code: RecallEvalGateReasonCode::CorruptRun,
            build_fingerprint: None,
            fixture_fingerprint: None,
            evaluated_at: None,
            reason: "recall run scorecard is corrupt".to_string(),
        };
    }

    let baseline = match &baseline_load {
        ScorecardLoad::Loaded(scorecard) => Some(scorecard),
        ScorecardLoad::Missing | ScorecardLoad::Corrupt(_) => None,
    };
    let current = match &current_load {
        ScorecardLoad::Loaded(scorecard) => Some(scorecard),
        ScorecardLoad::Missing | ScorecardLoad::Corrupt(_) => None,
    };
    let comparison = gates::compare_scorecards("recall", baseline, current, noise_floor);
    let status = match comparison.status {
        ScorecardStatus::Ship => ArsRecallGateStatus::Ship,
        ScorecardStatus::Bail => ArsRecallGateStatus::Bail,
        ScorecardStatus::NoData => ArsRecallGateStatus::NoData,
    };
    let reason_code = if matches!(&baseline_load, ScorecardLoad::Missing) {
        RecallEvalGateReasonCode::MissingBaseline
    } else if matches!(&current_load, ScorecardLoad::Missing) {
        RecallEvalGateReasonCode::MissingRun
    } else {
        RecallEvalGateReasonCode::Compared
    };
    RecallEvalGateAttestation {
        status,
        reason_code,
        build_fingerprint: current
            .map(|scorecard| scorecard.build_fingerprint.clone())
            .filter(|fingerprint| !fingerprint.is_empty()),
        fixture_fingerprint: current
            .map(|scorecard| scorecard.fixture_fingerprint.clone())
            .filter(|fingerprint| !fingerprint.is_empty()),
        evaluated_at: current.map(|scorecard| scorecard.created_at),
        reason: comparison.reason,
    }
}

/// Resolve every known human/A12 recall scope into a policy-ready evidence
/// record. Ineligible automatic evidence remains visible as `Static`; valid
/// human evidence is never suppressed merely because the automatic path is
/// absent or unhealthy.
pub fn resolve_recall_fusion_evidence(
    adaptive: &AdaptiveState,
    a12: &A12CalibrationLoad,
    min_samples_alpha: usize,
    expected_noise_floor: f64,
    now_millis: i64,
    recall_gate: &RecallEvalGateAttestation,
) -> BTreeMap<String, ArsRecallFusionEvidence> {
    let floor = u64::try_from(min_samples_alpha.max(10)).unwrap_or(u64::MAX);
    let mut scopes = BTreeSet::new();
    for (scope, entry) in &adaptive.learned_shadow_fusion {
        if valid_scope_key(scope)
            && u64::try_from(entry.sample_count).unwrap_or(u64::MAX) >= floor
            && human_simplex(entry).is_some()
        {
            scopes.insert(scope.clone());
        }
    }
    if a12.status == A12CalibrationLoadStatus::Loaded {
        scopes.extend(a12.state.scopes.keys().cloned());
    }

    scopes
        .into_iter()
        .map(|scope| {
            let policy_key = format!("recall_fusion:{scope}");
            let human = human_for_scope(adaptive, &scope, floor);
            let automatic = if a12.status == A12CalibrationLoadStatus::Loaded {
                a12.state.scopes.get(&scope)
            } else {
                None
            };
            let automatic_blocker = automatic.and_then(|entry| {
                automatic_ineligibility_reason(
                    adaptive,
                    a12,
                    entry,
                    floor,
                    expected_noise_floor,
                    now_millis,
                    recall_gate,
                )
            });
            let automatic_eligible = automatic.is_some() && automatic_blocker.is_none();
            let evidence = build_evidence(
                human,
                automatic,
                automatic_eligible,
                automatic_blocker,
                a12,
                recall_gate,
            );
            (policy_key, evidence)
        })
        .collect()
}

fn valid_scope_key(scope: &str) -> bool {
    if scope.is_empty() || scope.chars().any(char::is_whitespace) {
        return false;
    }
    if scope == "global" {
        return true;
    }
    match scope.split_once(':') {
        None => true,
        Some((query_type, cluster)) => {
            !query_type.is_empty()
                && !cluster.is_empty()
                && !cluster.contains(':')
                && cluster.parse::<u32>().is_ok()
        }
    }
}

fn human_simplex(entry: &LearnedShadowFusionEntry) -> Option<A12FusionSimplex> {
    // Legacy runtime accepted finite/non-finite and non-normalized persisted
    // shadow vectors, sanitizing them through `normalized_or_default` inside
    // `effective_simplex`. Preserve that byte-for-byte behavior while sealing
    // a schema-valid simplex into the policy evidence record.
    let normalized = crate::search::alpha_optimizer::ShadowFusionWeights {
        bm25: entry.weights.bm25,
        vec: entry.weights.vec,
        kg: entry.weights.kg,
        episode: entry.weights.episode,
        support: entry.weights.support,
        diversity: entry.weights.diversity,
    }
    .normalized_or_default();
    let simplex = A12FusionSimplex {
        bm25: normalized.bm25,
        vector: normalized.vec,
        kg: normalized.kg,
        episode: normalized.episode,
        support: normalized.support,
        diversity: normalized.diversity,
    };
    simplex_is_valid(simplex).then_some(simplex)
}

fn raw_human_simplex(entry: &LearnedShadowFusionEntry) -> A12FusionSimplex {
    A12FusionSimplex {
        bm25: entry.weights.bm25,
        vector: entry.weights.vec,
        kg: entry.weights.kg,
        episode: entry.weights.episode,
        support: entry.weights.support,
        diversity: entry.weights.diversity,
    }
}

fn simplex_is_valid(simplex: A12FusionSimplex) -> bool {
    let values = [
        simplex.bm25,
        simplex.vector,
        simplex.kg,
        simplex.episode,
        simplex.support,
        simplex.diversity,
    ];
    values
        .iter()
        .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
        && (values.iter().sum::<f64>() - 1.0).abs() <= 1e-6
}

fn human_for_scope<'a>(
    adaptive: &'a AdaptiveState,
    scope: &str,
    floor: u64,
) -> Option<&'a LearnedShadowFusionEntry> {
    let mut keys = Vec::with_capacity(3);
    keys.push(scope);
    if let Some((query_type, _cluster)) = scope.split_once(':') {
        keys.push(query_type);
    }
    if scope != "global" {
        keys.push("global");
    }
    keys.into_iter().find_map(|key| {
        let entry = adaptive.learned_shadow_fusion.get(key)?;
        let sample_count = u64::try_from(entry.sample_count).unwrap_or(u64::MAX);
        (sample_count >= floor && human_simplex(entry).is_some()).then_some(entry)
    })
}

fn has_judge_drift(adaptive: &AdaptiveState) -> bool {
    adaptive
        .judge_calibration_state
        .as_ref()
        .is_some_and(|calibration| {
            calibration.judge_drift_alert > 0
                || calibration.judge_drift_alert_synthesis > 0
                || calibration.judge_drift_alert_concept > 0
        })
}

fn automatic_ineligibility_reason(
    adaptive: &AdaptiveState,
    a12: &A12CalibrationLoad,
    entry: &A12ScopeEntry,
    floor: u64,
    expected_noise_floor: f64,
    now_millis: i64,
    recall_gate: &RecallEvalGateAttestation,
) -> Option<String> {
    if has_judge_drift(adaptive) {
        return Some("judge drift blocks automatic recall fusion".to_string());
    }
    if entry.verdict != A12CalibrationVerdict::Ship {
        return Some(format!("A12 holdout verdict is {:?}", entry.verdict));
    }
    if entry.train_family_ess < floor || entry.holdout_family_ess < floor {
        return Some(format!(
            "A12 family ESS below floor {floor} (train={}, holdout={})",
            entry.train_family_ess, entry.holdout_family_ess
        ));
    }
    if !entry.is_current_for_at(&a12.state, expected_noise_floor, now_millis) {
        return Some("A12 scope is stale for the active generation or noise floor".to_string());
    }
    if recall_gate.status != ArsRecallGateStatus::Ship {
        return Some(format!(
            "recall eval gate is {:?}, not Ship",
            recall_gate.status
        ));
    }
    if recall_gate.build_fingerprint.as_deref() != Some(env!("REIN_BUILD_FINGERPRINT")) {
        return Some("recall eval gate build fingerprint is not current".to_string());
    }
    if recall_gate
        .fixture_fingerprint
        .as_deref()
        .is_none_or(str::is_empty)
        || recall_gate.evaluated_at.is_none_or(|value| value <= 0)
    {
        return Some("recall eval gate attestation is incomplete".to_string());
    }
    None
}

fn build_evidence(
    human: Option<&LearnedShadowFusionEntry>,
    automatic: Option<&A12ScopeEntry>,
    automatic_eligible: bool,
    automatic_blocker: Option<String>,
    a12: &A12CalibrationLoad,
    recall_gate: &RecallEvalGateAttestation,
) -> ArsRecallFusionEvidence {
    let human_simplex = human.and_then(human_simplex);
    let human_ess = human
        .and_then(|entry| u64::try_from(entry.sample_count).ok())
        .unwrap_or(0);
    let eligible_automatic = automatic.filter(|_| automatic_eligible);
    let (basis, resolved_simplex) = match (human_simplex, eligible_automatic) {
        (Some(human), Some(automatic)) => (
            ArsRecallFusionEvidenceBasis::Blended,
            blend_simplexes(
                human,
                human_ess,
                automatic.simplex,
                automatic.train_family_ess,
            ),
        ),
        (Some(human), None) => (ArsRecallFusionEvidenceBasis::Human, human),
        (None, Some(automatic)) => (
            ArsRecallFusionEvidenceBasis::SelfSupervised,
            automatic.simplex,
        ),
        (None, None) => (
            ArsRecallFusionEvidenceBasis::Static,
            A12FusionSimplex::default(),
        ),
    };

    let reason = match basis {
        ArsRecallFusionEvidenceBasis::Human => "eligible human recall fusion".to_string(),
        ArsRecallFusionEvidenceBasis::SelfSupervised => {
            "holdout-approved self-supervised recall fusion".to_string()
        }
        ArsRecallFusionEvidenceBasis::Blended => {
            "ESS-blended human and self-supervised recall fusion".to_string()
        }
        ArsRecallFusionEvidenceBasis::Static => {
            automatic_blocker.unwrap_or_else(|| "no eligible recall-fusion evidence".to_string())
        }
    };
    let mut evidence = ArsRecallFusionEvidence {
        basis,
        resolved_simplex,
        human_ess,
        automatic_candidate_present: automatic.is_some(),
        human_simplex,
        human_runtime_adoption_weight: None,
        self_supervised_train_family_ess: 0,
        self_supervised_holdout_family_ess: 0,
        a12_generation: None,
        a12_revision: None,
        generation_fingerprint: None,
        corpus_fingerprint: None,
        optimizer_fingerprint: None,
        evaluation_fingerprint: None,
        a12_verdict: None,
        a12_noise_floor: None,
        recall_gate_status: recall_gate.status,
        recall_gate_build_fingerprint: None,
        recall_gate_fixture_fingerprint: None,
        recall_gate_evaluated_at: None,
        calibrated_at: None,
        evaluated_at: None,
        a12_valid_until_exclusive: None,
        reason,
    };
    if let Some(entry) = automatic {
        evidence.self_supervised_train_family_ess = entry.train_family_ess;
        evidence.self_supervised_holdout_family_ess = entry.holdout_family_ess;
        evidence.a12_generation = Some(a12.state.generation);
        evidence.a12_revision = Some(a12.state.revision);
        evidence.generation_fingerprint = Some(entry.generation_fingerprint.clone());
        evidence.corpus_fingerprint = Some(entry.corpus_fingerprint.clone());
        evidence.optimizer_fingerprint = Some(entry.optimizer_fingerprint.clone());
        evidence.evaluation_fingerprint = Some(entry.evaluation_fingerprint.clone());
        evidence.a12_verdict = Some(entry.verdict);
        evidence.a12_noise_floor = Some(entry.noise_floor);
        evidence.recall_gate_build_fingerprint = recall_gate.build_fingerprint.clone();
        evidence.recall_gate_fixture_fingerprint = recall_gate.fixture_fingerprint.clone();
        evidence.recall_gate_evaluated_at = recall_gate.evaluated_at;
        evidence.calibrated_at = Some(entry.calibrated_at);
        evidence.evaluated_at = Some(entry.evaluated_at);
        evidence.a12_valid_until_exclusive = entry.valid_until_exclusive;
    }
    evidence
}

fn blend_simplexes(
    human: A12FusionSimplex,
    human_ess: u64,
    automatic: A12FusionSimplex,
    automatic_ess: u64,
) -> A12FusionSimplex {
    let total = human_ess.saturating_add(automatic_ess) as f64;
    let human_weight = human_ess as f64 / total;
    let automatic_weight = automatic_ess as f64 / total;
    A12FusionSimplex {
        bm25: human.bm25 * human_weight + automatic.bm25 * automatic_weight,
        vector: human.vector * human_weight + automatic.vector * automatic_weight,
        kg: human.kg * human_weight + automatic.kg * automatic_weight,
        episode: human.episode * human_weight + automatic.episode * automatic_weight,
        support: human.support * human_weight + automatic.support * automatic_weight,
        diversity: human.diversity * human_weight + automatic.diversity * automatic_weight,
    }
}

/// Machine-readable classification of one recall-fusion scope's condition.
/// The `reason` strings next to it are for humans only: health rollups and
/// doctor attention key on this code, so reworded prose can never silently
/// mute an alert.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RecallFusionScopeHealthCode {
    /// Active on current, fully attested evidence (or on sealed human
    /// evidence exactly as the policy intended).
    Healthy,
    /// Active, but serving only the sealed human fallback because the
    /// blended automatic candidate is currently blocked — deliberately
    /// degraded, not healthy.
    HumanFallback,
    /// Benign absence: disabled, unresolved, static, zero adoption, or no
    /// evidence for this scope.
    Inactive,
    /// Evidence exists but is no longer current for the active generation,
    /// noise floor, eligibility floor, or live human state.
    Stale,
    /// The fixed-time replay validity horizon has passed.
    Expired,
    /// Sealed identity (generation/revision/fingerprints/timestamps) does not
    /// match the active evidence.
    FingerprintMismatch,
    /// Sealed and live values disagree in a way identity checks cannot
    /// explain (simplex/ESS/adoption inconsistencies).
    Tampered,
    /// The active A12 calibration run is not a completed Task-5 run.
    RunIncomplete,
    /// The recall eval gate is not current Ship evidence for this binary.
    GateNotShip,
    /// The A12 holdout verdict backing the seal is not Ship.
    HoldoutNotShip,
    /// A judge drift alert blocks automatic recall fusion.
    JudgeDrift,
}

impl RecallFusionScopeHealthCode {
    /// True for every condition that deserves operator attention. `Healthy`
    /// and `Inactive` are the only benign codes.
    pub fn is_degraded(self) -> bool {
        !matches!(self, Self::Healthy | Self::Inactive)
    }
}

/// Result consumed by online recall. A non-zero adoption always carries a
/// usable simplex; any attestation mismatch returns `None` plus zero adoption.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RuntimeRecallFusionResolution {
    pub scope_key: Option<String>,
    pub basis: ArsRecallFusionEvidenceBasis,
    pub simplex: Option<A12FusionSimplex>,
    pub adoption_weight: f64,
    pub code: RecallFusionScopeHealthCode,
    pub reason: String,
}

/// Resolve the sealed policy for one live recall without touching gate files.
/// An explicitly present specific scope is authoritative: if its attestation
/// is stale it fails closed instead of silently bypassing to a broader scope.
#[allow(clippy::too_many_arguments)]
pub fn resolve_runtime_recall_fusion(
    policy: &ArsParameterPolicy,
    config: &crate::config::ReinConfig,
    adaptive: &AdaptiveState,
    active_a12: &A12CalibrationLoad,
    query_type: &str,
    cluster_id: Option<u32>,
    min_samples_alpha: usize,
    expected_noise_floor: f64,
    now_millis: i64,
) -> RuntimeRecallFusionResolution {
    if !config.adaptive.enabled
        || !config.ars.acceleration.enabled
        || config.ars.acceleration.shadow_only
    {
        return runtime_disabled(
            None,
            ArsRecallFusionEvidenceBasis::Static,
            RecallFusionScopeHealthCode::Inactive,
            "adaptive recall-fusion activation is disabled or shadow-only",
        );
    }
    if policy.schema_version != ARS_PARAMETER_POLICY_SCHEMA_VERSION
        || policy.mode != ArsParameterPolicyMode::Canary
    {
        return runtime_disabled(
            None,
            ArsRecallFusionEvidenceBasis::Static,
            RecallFusionScopeHealthCode::Inactive,
            "policy is not a current canary",
        );
    }

    let keys = runtime_scope_keys(query_type, cluster_id);
    let selected = keys.into_iter().find(|key| {
        let has_matching_adoption = policy.adoption_weights.contains_key(key);
        match policy
            .recall_fusion_evidence
            .get(key)
            .map(|evidence| (evidence.basis, has_automatic_candidate(evidence)))
        {
            Some((
                ArsRecallFusionEvidenceBasis::Static
                | ArsRecallFusionEvidenceBasis::SelfSupervised
                | ArsRecallFusionEvidenceBasis::Blended,
                _,
            )) => true,
            Some((ArsRecallFusionEvidenceBasis::Human, true)) => true,
            Some((ArsRecallFusionEvidenceBasis::Human, false)) | None => has_matching_adoption,
        }
    });
    let evidence = selected
        .as_ref()
        .and_then(|key| policy.recall_fusion_evidence.get(key));

    match evidence.map(|value| value.basis) {
        Some(ArsRecallFusionEvidenceBasis::Static) => runtime_disabled(
            selected,
            ArsRecallFusionEvidenceBasis::Static,
            RecallFusionScopeHealthCode::Inactive,
            evidence
                .map(|value| value.reason.as_str())
                .unwrap_or("scope is explicitly static"),
        ),
        Some(ArsRecallFusionEvidenceBasis::SelfSupervised)
        | Some(ArsRecallFusionEvidenceBasis::Blended) => {
            let key = selected.expect("automatic evidence always has a selected key");
            let evidence = evidence.expect("automatic basis came from evidence");
            if let Some(blocker) = automatic_runtime_ineligibility_reason(
                policy,
                adaptive,
                active_a12,
                &key,
                evidence,
                min_samples_alpha,
                expected_noise_floor,
                now_millis,
            ) {
                if evidence.basis == ArsRecallFusionEvidenceBasis::Blended
                    && blocker.allows_human_fallback()
                    && evidence.human_simplex.is_some()
                    && evidence.human_runtime_adoption_weight.is_some()
                {
                    return resolve_runtime_sealed_human_boundary(
                        policy,
                        config,
                        adaptive,
                        active_a12,
                        &key,
                        evidence,
                        min_samples_alpha,
                    );
                }
                return runtime_disabled(
                    Some(key),
                    evidence.basis,
                    blocker.code(),
                    blocker.into_reason(),
                );
            }
            let adoption_weight = policy.runtime_adoption_weight_for(adaptive.version, &key);
            if adoption_weight <= f64::EPSILON {
                return runtime_disabled(
                    Some(key),
                    evidence.basis,
                    RecallFusionScopeHealthCode::Inactive,
                    "automatic scope has zero runtime adoption",
                );
            }
            RuntimeRecallFusionResolution {
                scope_key: Some(key),
                basis: evidence.basis,
                simplex: Some(effective_runtime_simplex(
                    config,
                    evidence.resolved_simplex,
                    evidence
                        .human_ess
                        .saturating_add(evidence.self_supervised_train_family_ess),
                    adoption_weight,
                )),
                adoption_weight,
                code: RecallFusionScopeHealthCode::Healthy,
                reason: "sealed automatic recall fusion is current".to_string(),
            }
        }
        Some(ArsRecallFusionEvidenceBasis::Human) => {
            let key = selected.expect("human evidence has a selected key");
            let evidence = evidence.expect("human basis came from evidence");
            if has_automatic_candidate(evidence) {
                resolve_runtime_sealed_human_boundary(
                    policy,
                    config,
                    adaptive,
                    active_a12,
                    &key,
                    evidence,
                    min_samples_alpha,
                )
            } else {
                resolve_runtime_human(
                    policy,
                    config,
                    adaptive,
                    query_type,
                    cluster_id,
                    min_samples_alpha,
                    Some(key),
                )
            }
        }
        None => resolve_runtime_human(
            policy,
            config,
            adaptive,
            query_type,
            cluster_id,
            min_samples_alpha,
            selected,
        ),
    }
}

/// Online A12 resolver. The pure resolver above validates the sealed policy
/// graph; this store-aware boundary additionally binds it to the SQLite state
/// and behavior-changing config that exist at the instant of live recall.
/// Ordinary corpus/config drift therefore disables stale automatic Ship
/// evidence immediately, before the next cadence refresh.
#[allow(clippy::too_many_arguments)]
pub fn resolve_runtime_recall_fusion_live(
    store: &crate::store::SqliteStore,
    policy: &ArsParameterPolicy,
    config: &crate::config::ReinConfig,
    adaptive: &AdaptiveState,
    active_a12: &A12CalibrationLoad,
    query_type: &str,
    cluster_id: Option<u32>,
    min_samples_alpha: usize,
    expected_noise_floor: f64,
    now_millis: i64,
) -> RuntimeRecallFusionResolution {
    let resolved = resolve_runtime_recall_fusion(
        policy,
        config,
        adaptive,
        active_a12,
        query_type,
        cluster_id,
        min_samples_alpha,
        expected_noise_floor,
        now_millis,
    );
    if resolved.adoption_weight <= f64::EPSILON
        || resolved.simplex.is_none()
        || !matches!(
            resolved.basis,
            ArsRecallFusionEvidenceBasis::SelfSupervised | ArsRecallFusionEvidenceBasis::Blended
        )
    {
        return resolved;
    }

    let Some(run) = active_a12
        .state
        .run
        .as_ref()
        .filter(|run| run.phase == A12CalibrationPhase::Complete)
    else {
        return live_runtime_input_blocked(
            policy,
            config,
            adaptive,
            active_a12,
            min_samples_alpha,
            resolved,
            RecallFusionScopeHealthCode::RunIncomplete,
            "active A12 run has no complete live-input attestation",
        );
    };

    let current_source_epoch =
        match crate::store::a12_calibration::load_a12_input_epoch(store.conn()) {
            Ok(epoch) => epoch,
            Err(_) => {
                return live_runtime_input_blocked(
                    policy,
                    config,
                    adaptive,
                    active_a12,
                    min_samples_alpha,
                    resolved,
                    RecallFusionScopeHealthCode::Stale,
                    "current A12 input epoch is unavailable",
                );
            }
        };
    if current_source_epoch != run.source_input_epoch {
        return live_runtime_input_blocked(
            policy,
            config,
            adaptive,
            active_a12,
            min_samples_alpha,
            resolved,
            RecallFusionScopeHealthCode::Stale,
            "current A12 input epoch drifted from the sealed Ship run",
        );
    }

    let hard_dedup_bound =
        crate::ops::effective_hard_dedup_threshold_from_conn(store.conn(), config);
    let current_behavior = match crate::ops::a12_autocalibration::a12_behavior_config_fingerprint(
        config,
        hard_dedup_bound,
        crate::ops::adaptive::A12_RECALL_TRACE_LIMIT,
        min_samples_alpha,
    ) {
        Ok(fingerprint) => fingerprint,
        Err(_) => {
            return live_runtime_input_blocked(
                policy,
                config,
                adaptive,
                active_a12,
                min_samples_alpha,
                resolved,
                RecallFusionScopeHealthCode::Stale,
                "current A12 behavior config fingerprint is unavailable",
            );
        }
    };
    if current_behavior != run.behavior_config_fingerprint {
        return live_runtime_input_blocked(
            policy,
            config,
            adaptive,
            active_a12,
            min_samples_alpha,
            resolved,
            RecallFusionScopeHealthCode::Stale,
            "current A12 behavior config drifted from the sealed Ship run",
        );
    }

    resolved
}

#[allow(clippy::too_many_arguments)]
fn live_runtime_input_blocked(
    policy: &ArsParameterPolicy,
    config: &crate::config::ReinConfig,
    adaptive: &AdaptiveState,
    active_a12: &A12CalibrationLoad,
    min_samples_alpha: usize,
    resolved: RuntimeRecallFusionResolution,
    code: RecallFusionScopeHealthCode,
    reason: &str,
) -> RuntimeRecallFusionResolution {
    let Some(policy_key) = resolved.scope_key.as_deref() else {
        return runtime_disabled(None, resolved.basis, code, reason);
    };
    if policy
        .recall_fusion_evidence
        .get(policy_key)
        .is_some_and(|evidence| evidence.basis == ArsRecallFusionEvidenceBasis::Blended)
    {
        let mut fallback = resolve_runtime_sealed_human_boundary(
            policy,
            config,
            adaptive,
            active_a12,
            policy_key,
            &policy.recall_fusion_evidence[policy_key],
            min_samples_alpha,
        );
        if fallback.adoption_weight > f64::EPSILON && fallback.simplex.is_some() {
            fallback.code = RecallFusionScopeHealthCode::HumanFallback;
            fallback.reason = format!("{reason}; serving sealed human fallback");
            return fallback;
        }
    }
    runtime_disabled(resolved.scope_key, resolved.basis, code, reason)
}

fn has_automatic_candidate(evidence: &ArsRecallFusionEvidence) -> bool {
    // Schema-2 rows written before the explicit marker existed still carried
    // the A12 pointer. Treat those rows as candidate boundaries too, so a
    // downgraded/older record cannot bypass a specific scope. Missing sealed
    // human fields then fail closed in the resolver below.
    evidence.automatic_candidate_present
        || evidence.a12_generation.is_some()
        || evidence.a12_revision.is_some()
}

#[allow(clippy::too_many_arguments)]
fn resolve_runtime_sealed_human_boundary(
    policy: &ArsParameterPolicy,
    config: &crate::config::ReinConfig,
    adaptive: &AdaptiveState,
    active_a12: &A12CalibrationLoad,
    policy_key: &str,
    evidence: &ArsRecallFusionEvidence,
    min_samples_alpha: usize,
) -> RuntimeRecallFusionResolution {
    let selected = Some(policy_key.to_string());
    if policy.source_adaptive_version != adaptive.version {
        return runtime_disabled(
            selected,
            ArsRecallFusionEvidenceBasis::Human,
            RecallFusionScopeHealthCode::Stale,
            "sealed human fallback source adaptive version is not exact",
        );
    }
    if let Some((code, reason)) =
        automatic_candidate_identity_mismatch(active_a12, policy_key, evidence)
    {
        return runtime_disabled(selected, ArsRecallFusionEvidenceBasis::Human, code, reason);
    }
    let Some(sealed_human) = evidence.human_simplex else {
        return runtime_disabled(
            selected,
            ArsRecallFusionEvidenceBasis::Human,
            RecallFusionScopeHealthCode::Tampered,
            "automatic boundary is missing its sealed human simplex",
        );
    };
    if let Some(reason) =
        sealed_human_fallback_relation_mismatch(active_a12, policy_key, evidence, sealed_human)
    {
        return runtime_disabled(
            selected,
            ArsRecallFusionEvidenceBasis::Human,
            RecallFusionScopeHealthCode::Tampered,
            reason,
        );
    }
    let Some(adoption_weight) = evidence.human_runtime_adoption_weight else {
        return runtime_disabled(
            selected,
            ArsRecallFusionEvidenceBasis::Human,
            RecallFusionScopeHealthCode::Tampered,
            "automatic boundary is missing its sealed human adoption",
        );
    };
    if !adoption_weight.is_finite()
        || !(0.0..=1.0).contains(&adoption_weight)
        || adoption_weight <= f64::EPSILON
    {
        return runtime_disabled(
            selected,
            ArsRecallFusionEvidenceBasis::Human,
            RecallFusionScopeHealthCode::Inactive,
            "automatic boundary has zero or invalid sealed human adoption",
        );
    }
    let Some(scope) = policy_key.strip_prefix("recall_fusion:") else {
        return runtime_disabled(
            selected,
            ArsRecallFusionEvidenceBasis::Human,
            RecallFusionScopeHealthCode::Tampered,
            "sealed human policy key is outside recall_fusion",
        );
    };
    let floor = u64::try_from(min_samples_alpha.max(10)).unwrap_or(u64::MAX);
    let Some(human) = human_for_scope(adaptive, scope, floor) else {
        return runtime_disabled(
            selected,
            ArsRecallFusionEvidenceBasis::Human,
            RecallFusionScopeHealthCode::Stale,
            "automatic boundary has no matching live human fallback",
        );
    };
    let human_ess = u64::try_from(human.sample_count).unwrap_or(u64::MAX);
    let Some(live_human) = human_simplex(human) else {
        return runtime_disabled(
            selected,
            ArsRecallFusionEvidenceBasis::Human,
            RecallFusionScopeHealthCode::Tampered,
            "automatic boundary has an invalid live human fallback",
        );
    };
    if evidence.human_ess != human_ess || !simplex_matches(sealed_human, live_human) {
        return runtime_disabled(
            selected,
            ArsRecallFusionEvidenceBasis::Human,
            RecallFusionScopeHealthCode::Tampered,
            "live human fallback does not match sealed simplex or ESS",
        );
    }

    RuntimeRecallFusionResolution {
        scope_key: Some(policy_key.to_string()),
        basis: ArsRecallFusionEvidenceBasis::Human,
        simplex: Some(effective_runtime_simplex(
            config,
            raw_human_simplex(human),
            human_ess,
            adoption_weight,
        )),
        adoption_weight,
        // A Blended seal served through its human fallback is deliberately
        // degraded: the automatic candidate is blocked, so operators must see
        // this scope as HumanFallback, never Healthy. A pure Human seal with
        // an automatic candidate boundary is the sealed intent and healthy.
        code: if evidence.basis == ArsRecallFusionEvidenceBasis::Blended {
            RecallFusionScopeHealthCode::HumanFallback
        } else {
            RecallFusionScopeHealthCode::Healthy
        },
        reason: "authoritative scoped human fallback".to_string(),
    }
}

fn sealed_human_fallback_relation_mismatch(
    active_a12: &A12CalibrationLoad,
    policy_key: &str,
    evidence: &ArsRecallFusionEvidence,
    sealed_human: A12FusionSimplex,
) -> Option<String> {
    match evidence.basis {
        ArsRecallFusionEvidenceBasis::Human => {
            if !simplex_matches(evidence.resolved_simplex, sealed_human) {
                return Some(
                    "sealed human fallback resolved simplex mismatched pure human evidence"
                        .to_string(),
                );
            }
        }
        ArsRecallFusionEvidenceBasis::Blended => {
            let Some(scope) = policy_key.strip_prefix("recall_fusion:") else {
                return Some("blended fallback policy key is outside recall_fusion".to_string());
            };
            let Some(entry) = active_a12.state.scopes.get(scope) else {
                return Some("blended fallback has no active A12 scope".to_string());
            };
            let expected = blend_simplexes(
                sealed_human,
                evidence.human_ess,
                entry.simplex,
                entry.train_family_ess,
            );
            if !simplex_matches(evidence.resolved_simplex, expected) {
                return Some(
                    "sealed blended fallback resolved simplex mismatched A12 and human evidence"
                        .to_string(),
                );
            }
        }
        ArsRecallFusionEvidenceBasis::Static | ArsRecallFusionEvidenceBasis::SelfSupervised => {
            return Some("sealed human fallback has an incompatible evidence basis".to_string());
        }
    }
    None
}

fn automatic_candidate_identity_mismatch(
    active_a12: &A12CalibrationLoad,
    policy_key: &str,
    evidence: &ArsRecallFusionEvidence,
) -> Option<(RecallFusionScopeHealthCode, String)> {
    if active_a12.status != A12CalibrationLoadStatus::Loaded {
        return Some((
            RecallFusionScopeHealthCode::Stale,
            "active A12 calibration is unavailable for human fallback".to_string(),
        ));
    }
    let Some(scope) = policy_key.strip_prefix("recall_fusion:") else {
        return Some((
            RecallFusionScopeHealthCode::Tampered,
            "human fallback policy key is outside recall_fusion".to_string(),
        ));
    };
    let Some(entry) = active_a12.state.scopes.get(scope) else {
        return Some((
            RecallFusionScopeHealthCode::Stale,
            "active A12 calibration has no matching fallback boundary".to_string(),
        ));
    };
    if evidence.a12_generation != Some(active_a12.state.generation)
        || evidence.a12_revision != Some(active_a12.state.revision)
        || evidence.generation_fingerprint.as_deref() != Some(entry.generation_fingerprint.as_str())
        || evidence.corpus_fingerprint.as_deref() != Some(entry.corpus_fingerprint.as_str())
        || evidence.optimizer_fingerprint.as_deref() != Some(entry.optimizer_fingerprint.as_str())
        || evidence.evaluation_fingerprint.as_deref() != Some(entry.evaluation_fingerprint.as_str())
        || evidence.a12_verdict != Some(entry.verdict)
        || evidence.self_supervised_train_family_ess != entry.train_family_ess
        || evidence.self_supervised_holdout_family_ess != entry.holdout_family_ess
        || evidence.calibrated_at != Some(entry.calibrated_at)
        || evidence.evaluated_at != Some(entry.evaluated_at)
        || evidence.a12_valid_until_exclusive != entry.valid_until_exclusive
        || evidence
            .a12_noise_floor
            .is_none_or(|noise_floor| !same_f64(noise_floor, entry.noise_floor))
    {
        return Some((
            RecallFusionScopeHealthCode::FingerprintMismatch,
            "sealed A12 candidate identity mismatched human fallback boundary".to_string(),
        ));
    }
    None
}

fn runtime_scope_keys(query_type: &str, cluster_id: Option<u32>) -> Vec<String> {
    let mut keys = Vec::with_capacity(3);
    if let Some(cluster_id) = cluster_id {
        keys.push(format!(
            "recall_fusion:{}",
            AdaptiveState::bucket_key(query_type, Some(cluster_id))
        ));
    }
    keys.push(format!(
        "recall_fusion:{}",
        AdaptiveState::bucket_key(query_type, None)
    ));
    keys.push("recall_fusion:global".to_string());
    keys
}

fn resolve_runtime_human(
    policy: &ArsParameterPolicy,
    config: &crate::config::ReinConfig,
    adaptive: &AdaptiveState,
    query_type: &str,
    cluster_id: Option<u32>,
    min_samples_alpha: usize,
    selected: Option<String>,
) -> RuntimeRecallFusionResolution {
    let adoption_weight = selected.as_ref().map_or_else(
        || policy.runtime_adoption_weight(adaptive.version),
        |key| policy.runtime_adoption_weight_for(adaptive.version, key),
    );
    if adoption_weight <= f64::EPSILON {
        return runtime_disabled(
            selected,
            ArsRecallFusionEvidenceBasis::Human,
            RecallFusionScopeHealthCode::Inactive,
            "human recall-fusion policy has zero runtime adoption",
        );
    }
    let Some(entry) = adaptive.get_shadow_fusion_weights(query_type, cluster_id, min_samples_alpha)
    else {
        return runtime_disabled(
            selected,
            ArsRecallFusionEvidenceBasis::Human,
            RecallFusionScopeHealthCode::Inactive,
            "no eligible live human recall-fusion entry",
        );
    };
    // Feed the live raw vector into the same normalization boundary used by
    // the legacy runtime. The policy evidence stores a normalized copy, but
    // normalizing that copy a second time can introduce last-bit drift and
    // perturb deterministic tie ordering.
    let learned_simplex = raw_human_simplex(entry);
    let evidence_count = u64::try_from(entry.sample_count).unwrap_or(u64::MAX);
    RuntimeRecallFusionResolution {
        scope_key: selected,
        basis: ArsRecallFusionEvidenceBasis::Human,
        simplex: Some(effective_runtime_simplex(
            config,
            learned_simplex,
            evidence_count,
            adoption_weight,
        )),
        adoption_weight,
        code: RecallFusionScopeHealthCode::Healthy,
        reason: "legacy-compatible live human recall fusion".to_string(),
    }
}

fn effective_runtime_simplex(
    config: &crate::config::ReinConfig,
    learned: A12FusionSimplex,
    evidence_count: u64,
    adoption_weight: f64,
) -> A12FusionSimplex {
    let static_prior = crate::search::alpha_optimizer::ShadowFusionWeights::default();
    let effective = crate::ops::ars_tuning::effective_simplex(
        [
            static_prior.bm25,
            static_prior.vec,
            static_prior.kg,
            static_prior.episode,
            static_prior.support,
            static_prior.diversity,
        ],
        [
            learned.bm25,
            learned.vector,
            learned.kg,
            learned.episode,
            learned.support,
            learned.diversity,
        ],
        crate::ops::ars_tuning::TrustInputs {
            enabled: config.ars.acceleration.enabled,
            production_canary: adoption_weight > f64::EPSILON,
            runtime_adoption_weight: adoption_weight,
            human_count: evidence_count,
            llm_count: 0,
            llm_reliability: 0.0,
            calibration: 1.0,
            stability: 1.0,
            drift_alert: false,
            prior_strength: config.adaptive.shrinkage_prior,
            max_trust: 0.85,
        },
    );
    A12FusionSimplex {
        bm25: effective[0],
        vector: effective[1],
        kg: effective[2],
        episode: effective[3],
        support: effective[4],
        diversity: effective[5],
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum AutomaticRuntimeBlocker {
    Eligibility(RecallFusionScopeHealthCode, String),
    FailClosed(RecallFusionScopeHealthCode, String),
}

impl AutomaticRuntimeBlocker {
    fn eligibility(code: RecallFusionScopeHealthCode, reason: impl Into<String>) -> Self {
        Self::Eligibility(code, reason.into())
    }

    fn fail_closed(code: RecallFusionScopeHealthCode, reason: impl Into<String>) -> Self {
        Self::FailClosed(code, reason.into())
    }

    fn allows_human_fallback(&self) -> bool {
        matches!(self, Self::Eligibility(..))
    }

    fn code(&self) -> RecallFusionScopeHealthCode {
        match self {
            Self::Eligibility(code, _) | Self::FailClosed(code, _) => *code,
        }
    }

    fn into_reason(self) -> String {
        match self {
            Self::Eligibility(_, reason) | Self::FailClosed(_, reason) => reason,
        }
    }
}

/// True only when an attested recall eval-gate identity is current Ship
/// evidence for this exact binary: Ship status, this build's fingerprint, a
/// non-empty fixture fingerprint, and a positive evaluation time. Shared by
/// the sealed-policy and live-attestation blockers so both enforce the same
/// definition of "current Ship evidence".
fn current_ship_gate_identity(
    status: ArsRecallGateStatus,
    build_fingerprint: Option<&str>,
    fixture_fingerprint: Option<&str>,
    evaluated_at: Option<i64>,
) -> bool {
    status == ArsRecallGateStatus::Ship
        && build_fingerprint == Some(env!("REIN_BUILD_FINGERPRINT"))
        && fixture_fingerprint.is_some_and(|fingerprint| !fingerprint.is_empty())
        && evaluated_at.is_some_and(|value| value > 0)
}

fn automatic_runtime_ineligibility_reason(
    policy: &ArsParameterPolicy,
    adaptive: &AdaptiveState,
    active_a12: &A12CalibrationLoad,
    policy_key: &str,
    evidence: &ArsRecallFusionEvidence,
    min_samples_alpha: usize,
    expected_noise_floor: f64,
    now_millis: i64,
) -> Option<AutomaticRuntimeBlocker> {
    if policy.source_adaptive_version != adaptive.version {
        return Some(AutomaticRuntimeBlocker::fail_closed(
            RecallFusionScopeHealthCode::Stale,
            "automatic policy source adaptive version is not exact",
        ));
    }
    if !policy.adoption_weights.contains_key(policy_key) {
        return Some(AutomaticRuntimeBlocker::fail_closed(
            RecallFusionScopeHealthCode::Tampered,
            "automatic policy is missing its explicit scoped adoption weight",
        ));
    }
    if active_a12.status != A12CalibrationLoadStatus::Loaded {
        return Some(AutomaticRuntimeBlocker::fail_closed(
            RecallFusionScopeHealthCode::Stale,
            "active A12 calibration is unavailable",
        ));
    }
    let Some(scope) = policy_key.strip_prefix("recall_fusion:") else {
        return Some(AutomaticRuntimeBlocker::fail_closed(
            RecallFusionScopeHealthCode::Tampered,
            "automatic policy key is outside recall_fusion",
        ));
    };
    let Some(entry) = active_a12.state.scopes.get(scope) else {
        return Some(AutomaticRuntimeBlocker::fail_closed(
            RecallFusionScopeHealthCode::Stale,
            "active A12 calibration has no matching scope",
        ));
    };
    let floor = u64::try_from(min_samples_alpha.max(10)).unwrap_or(u64::MAX);
    if entry.verdict != A12CalibrationVerdict::Ship
        || evidence.a12_verdict != Some(A12CalibrationVerdict::Ship)
    {
        return Some(AutomaticRuntimeBlocker::fail_closed(
            RecallFusionScopeHealthCode::HoldoutNotShip,
            "automatic A12 holdout attestation is not Ship",
        ));
    }
    if evidence.self_supervised_train_family_ess != entry.train_family_ess
        || evidence.self_supervised_holdout_family_ess != entry.holdout_family_ess
    {
        return Some(AutomaticRuntimeBlocker::fail_closed(
            RecallFusionScopeHealthCode::Tampered,
            "automatic A12 ESS attestation is mismatched",
        ));
    }
    if evidence.a12_generation != Some(active_a12.state.generation)
        || evidence.a12_revision != Some(active_a12.state.revision)
        || evidence.generation_fingerprint.as_deref() != Some(entry.generation_fingerprint.as_str())
        || evidence.corpus_fingerprint.as_deref() != Some(entry.corpus_fingerprint.as_str())
        || evidence.optimizer_fingerprint.as_deref() != Some(entry.optimizer_fingerprint.as_str())
        || evidence.evaluation_fingerprint.as_deref() != Some(entry.evaluation_fingerprint.as_str())
        || evidence.calibrated_at != Some(entry.calibrated_at)
        || evidence.evaluated_at != Some(entry.evaluated_at)
        || evidence.a12_valid_until_exclusive != entry.valid_until_exclusive
    {
        return Some(AutomaticRuntimeBlocker::fail_closed(
            RecallFusionScopeHealthCode::FingerprintMismatch,
            "sealed A12 pointer or fingerprint identity mismatched",
        ));
    }
    if evidence
        .a12_noise_floor
        .is_none_or(|noise_floor| !same_f64(noise_floor, entry.noise_floor))
    {
        return Some(AutomaticRuntimeBlocker::fail_closed(
            RecallFusionScopeHealthCode::FingerprintMismatch,
            "sealed A12 noise floor mismatched active evidence",
        ));
    }
    if !current_ship_gate_identity(
        evidence.recall_gate_status,
        evidence.recall_gate_build_fingerprint.as_deref(),
        evidence.recall_gate_fixture_fingerprint.as_deref(),
        evidence.recall_gate_evaluated_at,
    ) {
        return Some(AutomaticRuntimeBlocker::fail_closed(
            RecallFusionScopeHealthCode::GateNotShip,
            "sealed recall eval gate is not current Ship evidence",
        ));
    }
    if !simplex_is_valid(evidence.resolved_simplex) {
        return Some(AutomaticRuntimeBlocker::fail_closed(
            RecallFusionScopeHealthCode::Tampered,
            "sealed recall-fusion simplex is invalid",
        ));
    }
    if evidence.basis == ArsRecallFusionEvidenceBasis::SelfSupervised
        && evidence.resolved_simplex != entry.simplex
    {
        return Some(AutomaticRuntimeBlocker::fail_closed(
            RecallFusionScopeHealthCode::Tampered,
            "self-supervised simplex does not match active A12 scope",
        ));
    }
    if evidence.basis == ArsRecallFusionEvidenceBasis::Blended {
        let Some(human) = human_for_scope(adaptive, scope, floor) else {
            return Some(AutomaticRuntimeBlocker::fail_closed(
                RecallFusionScopeHealthCode::Tampered,
                "blended evidence has no matching live human scope",
            ));
        };
        let human_ess = u64::try_from(human.sample_count).unwrap_or(u64::MAX);
        let Some(human_simplex) = human_simplex(human) else {
            return Some(AutomaticRuntimeBlocker::fail_closed(
                RecallFusionScopeHealthCode::Tampered,
                "blended evidence has an invalid live human simplex",
            ));
        };
        if human_ess != evidence.human_ess
            || !simplex_matches(
                evidence.resolved_simplex,
                blend_simplexes(
                    human_simplex,
                    human_ess,
                    entry.simplex,
                    entry.train_family_ess,
                ),
            )
        {
            return Some(AutomaticRuntimeBlocker::fail_closed(
                RecallFusionScopeHealthCode::Tampered,
                "blended simplex or human ESS mismatched sealed evidence",
            ));
        }
    }
    // Integrity and sealed-relation checks above deliberately dominate the
    // rollback conditions below. An expired/stale candidate is eligible for
    // its sealed Human fallback only when every persisted attestation still
    // matches; eligibility must never mask a concurrent tamper.
    if has_judge_drift(adaptive) {
        return Some(AutomaticRuntimeBlocker::eligibility(
            RecallFusionScopeHealthCode::JudgeDrift,
            "judge drift blocks automatic recall fusion",
        ));
    }
    if entry.train_family_ess < floor || entry.holdout_family_ess < floor {
        return Some(AutomaticRuntimeBlocker::eligibility(
            RecallFusionScopeHealthCode::Stale,
            "automatic A12 ESS is below the current eligibility floor",
        ));
    }
    if !entry.matches_noise_floor(expected_noise_floor) {
        return Some(AutomaticRuntimeBlocker::eligibility(
            RecallFusionScopeHealthCode::Stale,
            "active A12 noise floor is stale for the current runtime",
        ));
    }
    if !entry.is_current_for_at(&active_a12.state, expected_noise_floor, now_millis) {
        let expired = entry
            .valid_until_exclusive
            .is_some_and(|boundary| now_millis >= boundary);
        return Some(AutomaticRuntimeBlocker::eligibility(
            if expired {
                RecallFusionScopeHealthCode::Expired
            } else {
                RecallFusionScopeHealthCode::Stale
            },
            "active A12 scope is stale or expired",
        ));
    }
    None
}

fn same_f64(left: f64, right: f64) -> bool {
    left.is_finite()
        && right.is_finite()
        && (left == right
            || (left - right).abs() <= 1e-12 * left.abs().max(right.abs()).max(f64::MIN_POSITIVE))
}

fn simplex_matches(left: A12FusionSimplex, right: A12FusionSimplex) -> bool {
    [
        (left.bm25, right.bm25),
        (left.vector, right.vector),
        (left.kg, right.kg),
        (left.episode, right.episode),
        (left.support, right.support),
        (left.diversity, right.diversity),
    ]
    .into_iter()
    .all(|(left, right)| same_f64(left, right))
}

fn runtime_disabled(
    scope_key: Option<String>,
    basis: ArsRecallFusionEvidenceBasis,
    code: RecallFusionScopeHealthCode,
    reason: impl Into<String>,
) -> RuntimeRecallFusionResolution {
    RuntimeRecallFusionResolution {
        scope_key,
        basis,
        simplex: None,
        adoption_weight: 0.0,
        code,
        reason: reason.into(),
    }
}

/// Version 2 covers two additive extensions: the typed `health_code` scope
/// classification and the per-provenance holdout diagnostics
/// (`provenance_holdout` + `provenance_direction_conflict`).
pub const RECALL_FUSION_ACTIVATION_REPORT_SCHEMA_VERSION: u32 = 2;

/// Task-5 run state projected without coupling consumers to the persisted A12
/// schema. Legacy A12 rows have no run metadata and therefore never satisfy
/// the complete-run prerequisite for automatic activation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RecallFusionCalibrationRunPhase {
    Pending,
    Complete,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RecallFusionCalibrationRunAttestation {
    pub phase: RecallFusionCalibrationRunPhase,
    pub source_snapshot_fingerprint: String,
    pub behavior_config_fingerprint: String,
}

/// Read-only projection of one recall-fusion scope. Runtime eligibility is
/// owned exclusively by [`resolve_runtime_recall_fusion`]; this DTO only
/// combines that decision with the sealed policy and active A12 evidence.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RecallFusionActivationScopeReport {
    pub scope: String,
    pub policy_key: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolved_policy_key: Option<String>,
    /// Effective basis returned by the shared runtime resolver.
    pub basis: ArsRecallFusionEvidenceBasis,
    /// Basis sealed in the exact policy scope, retained as provenance only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sealed_basis: Option<ArsRecallFusionEvidenceBasis>,
    /// Typed condition code. Health rollups and doctor attention key on this
    /// field; `reason` below is human prose only.
    pub health_code: RecallFusionScopeHealthCode,
    pub reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effective_simplex: Option<A12FusionSimplex>,
    pub adoption_weight: f64,
    pub human_ess: u64,
    pub train_family_ess: u64,
    pub holdout_family_ess: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verdict: Option<A12CalibrationVerdict>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub paired_top3: Option<crate::store::a12_calibration::A12PairedTop3Stats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provenance: Option<crate::store::a12_calibration::A12ProvenanceCounts>,
    /// Per-provenance paired holdout cells sealed with the active A12 scope.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provenance_holdout: Option<crate::store::a12_calibration::A12ProvenanceHoldoutStats>,
    /// True when two label-provenance sources pull the holdout in opposite
    /// discordant directions — the deterministic cue that a second-opinion
    /// arbiter would become worthwhile. Derived from the raw paired cells;
    /// no numeric threshold anywhere.
    pub provenance_direction_conflict: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub a12_generation: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub a12_revision: Option<u64>,
    pub source_adaptive_version: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_snapshot_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub train_case_count: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub holdout_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub corpus_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub training_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub holdout_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub optimizer_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evaluation_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recall_gate_status: Option<ArsRecallGateStatus>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recall_gate_build_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recall_gate_fixture_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recall_gate_evaluated_at: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sealed_recall_gate_status: Option<ArsRecallGateStatus>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sealed_recall_gate_build_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sealed_recall_gate_fixture_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sealed_recall_gate_evaluated_at: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub calibrated_at: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evaluated_at: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub valid_until_exclusive: Option<i64>,
    pub valid_now: bool,
    pub active: bool,
}

/// Shared read-only recall-fusion activation report used by Adaptive, the
/// release gate, Trust, and doctor. It contains semantic identifiers only and
/// never records scorecard/database paths.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RecallFusionActivationReport {
    pub schema_version: u32,
    pub activation_status: String,
    pub health_status: String,
    pub reason: String,
    pub policy_load_status: String,
    pub a12_load_status: String,
    pub policy_mode: String,
    pub current_adaptive_version: u64,
    pub source_adaptive_version: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub a12_generation: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub a12_revision: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_phase: Option<RecallFusionCalibrationRunPhase>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_snapshot_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub behavior_config_fingerprint: Option<String>,
    pub run_complete: bool,
    pub active: bool,
    pub scopes: Vec<RecallFusionActivationScopeReport>,
}

/// Project the persisted Task-5 run envelope into the shared report DTO.
/// Legacy schema-2 rows carry no run metadata and therefore never satisfy the
/// complete-run prerequisite for automatic activation.
pub fn recall_fusion_calibration_run_attestation(
    active_a12: &A12CalibrationLoad,
) -> Option<RecallFusionCalibrationRunAttestation> {
    if active_a12.status != A12CalibrationLoadStatus::Loaded {
        return None;
    }
    let run = active_a12.state.run.as_ref()?;
    Some(RecallFusionCalibrationRunAttestation {
        phase: match run.phase {
            A12CalibrationPhase::Pending => RecallFusionCalibrationRunPhase::Pending,
            A12CalibrationPhase::Complete => RecallFusionCalibrationRunPhase::Complete,
        },
        source_snapshot_fingerprint: run.source_snapshot_fingerprint.clone(),
        behavior_config_fingerprint: run.behavior_config_fingerprint.clone(),
    })
}

pub fn collect_recall_fusion_activation_report(
    store: &crate::store::SqliteStore,
    config: &crate::config::ReinConfig,
    now_millis: i64,
) -> RecallFusionActivationReport {
    let adaptive = AdaptiveState::restore_snapshot(store.conn()).unwrap_or_default();
    let policy = crate::store::ars_parameter_policy::load_parameter_policy(store.conn());
    let active_a12 = crate::store::a12_calibration::load_a12_calibration(store.conn());
    let recall_gate = current_recall_eval_gate_attestation(
        crate::store::a12_calibration::A12_DEFAULT_NOISE_FLOOR,
    );
    let run = recall_fusion_calibration_run_attestation(&active_a12);
    recall_fusion_activation_report_live(
        store,
        config,
        &adaptive,
        &policy,
        &active_a12,
        &recall_gate,
        run.as_ref(),
        now_millis,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn recall_fusion_activation_report(
    config: &crate::config::ReinConfig,
    adaptive: &AdaptiveState,
    policy: &crate::store::ars_parameter_policy::ArsParameterPolicyLoad,
    active_a12: &A12CalibrationLoad,
    current_recall_gate: &RecallEvalGateAttestation,
    calibration_run: Option<&RecallFusionCalibrationRunAttestation>,
    now_millis: i64,
) -> RecallFusionActivationReport {
    recall_fusion_activation_report_impl(
        None,
        config,
        adaptive,
        policy,
        active_a12,
        current_recall_gate,
        calibration_run,
        now_millis,
    )
}

/// Store-aware report boundary used by production doctor/Trust/release paths.
/// Its scope decisions are identical to online recall: full snapshot hashing
/// stays offline, while the live epoch and behavior fingerprint fail closed.
#[allow(clippy::too_many_arguments)]
pub fn recall_fusion_activation_report_live(
    store: &crate::store::SqliteStore,
    config: &crate::config::ReinConfig,
    adaptive: &AdaptiveState,
    policy: &crate::store::ars_parameter_policy::ArsParameterPolicyLoad,
    active_a12: &A12CalibrationLoad,
    current_recall_gate: &RecallEvalGateAttestation,
    calibration_run: Option<&RecallFusionCalibrationRunAttestation>,
    now_millis: i64,
) -> RecallFusionActivationReport {
    recall_fusion_activation_report_impl(
        Some(store),
        config,
        adaptive,
        policy,
        active_a12,
        current_recall_gate,
        calibration_run,
        now_millis,
    )
}

#[allow(clippy::too_many_arguments)]
fn recall_fusion_activation_report_impl(
    live_store: Option<&crate::store::SqliteStore>,
    config: &crate::config::ReinConfig,
    adaptive: &AdaptiveState,
    policy: &crate::store::ars_parameter_policy::ArsParameterPolicyLoad,
    active_a12: &A12CalibrationLoad,
    current_recall_gate: &RecallEvalGateAttestation,
    calibration_run: Option<&RecallFusionCalibrationRunAttestation>,
    now_millis: i64,
) -> RecallFusionActivationReport {
    let mut scope_names = BTreeSet::new();
    scope_names.extend(
        adaptive
            .learned_shadow_fusion
            .keys()
            .filter(|scope| valid_scope_key(scope))
            .cloned(),
    );
    scope_names.extend(active_a12.state.scopes.keys().cloned());
    scope_names.extend(
        policy
            .policy
            .recall_fusion_evidence
            .keys()
            .filter_map(|key| {
                key.strip_prefix("recall_fusion:")
                    .filter(|scope| valid_scope_key(scope))
                    .map(ToOwned::to_owned)
            }),
    );

    let scopes = scope_names
        .into_iter()
        .map(|scope| {
            recall_fusion_activation_scope_report(
                live_store,
                config,
                adaptive,
                policy,
                active_a12,
                current_recall_gate,
                calibration_run,
                &scope,
                now_millis,
            )
        })
        .collect::<Vec<_>>();
    let active = scopes.iter().any(|scope| scope.active);
    let activation_status = if !config.adaptive.enabled
        || !config.ars.acceleration.enabled
        || config.ars.acceleration.shadow_only
    {
        "disabled"
    } else if active {
        "active"
    } else {
        "inactive"
    };
    let health_status = recall_fusion_report_health(policy, active_a12.status, active, &scopes);
    let reason = match (activation_status, health_status) {
        ("disabled", _) => "recall-fusion activation is disabled or shadow-only by config",
        ("active", "healthy") => "at least one recall-fusion scope is active and healthy",
        ("active", _) => "recall fusion is active with degraded or incomplete evidence",
        (_, "missing" | "policy_missing" | "a12_missing") => {
            "recall-fusion calibration evidence is missing"
        }
        _ => "no recall-fusion scope passes the shared runtime activation checks",
    }
    .to_string();
    let a12_loaded = active_a12.status == A12CalibrationLoadStatus::Loaded;
    let run_complete = calibration_run.is_some_and(complete_calibration_run);

    RecallFusionActivationReport {
        schema_version: RECALL_FUSION_ACTIVATION_REPORT_SCHEMA_VERSION,
        activation_status: activation_status.to_string(),
        health_status: health_status.to_string(),
        reason,
        policy_load_status: parameter_policy_load_status_name(&policy.status),
        a12_load_status: a12_load_status_name(active_a12.status).to_string(),
        policy_mode: parameter_policy_mode_name(policy.policy.mode).to_string(),
        current_adaptive_version: adaptive.version,
        source_adaptive_version: policy.policy.source_adaptive_version,
        a12_generation: a12_loaded.then_some(active_a12.state.generation),
        a12_revision: a12_loaded.then_some(active_a12.state.revision),
        run_phase: calibration_run.map(|run| run.phase),
        source_snapshot_fingerprint: calibration_run
            .map(|run| run.source_snapshot_fingerprint.clone()),
        behavior_config_fingerprint: calibration_run
            .map(|run| run.behavior_config_fingerprint.clone()),
        run_complete,
        active,
        scopes,
    }
}

#[allow(clippy::too_many_arguments)]
fn recall_fusion_activation_scope_report(
    live_store: Option<&crate::store::SqliteStore>,
    config: &crate::config::ReinConfig,
    adaptive: &AdaptiveState,
    policy: &crate::store::ars_parameter_policy::ArsParameterPolicyLoad,
    active_a12: &A12CalibrationLoad,
    current_recall_gate: &RecallEvalGateAttestation,
    calibration_run: Option<&RecallFusionCalibrationRunAttestation>,
    scope: &str,
    now_millis: i64,
) -> RecallFusionActivationScopeReport {
    let policy_key = format!("recall_fusion:{scope}");
    let evidence = policy.policy.recall_fusion_evidence.get(&policy_key);
    let entry = (active_a12.status == A12CalibrationLoadStatus::Loaded)
        .then(|| active_a12.state.scopes.get(scope))
        .flatten();
    let resolution = if policy.status
        == crate::store::ars_parameter_policy::ArsParameterPolicyLoadStatus::Loaded
    {
        match runtime_scope_parts(scope) {
            Some((query_type, cluster_id)) => match live_store {
                Some(store) => resolve_runtime_recall_fusion_live(
                    store,
                    &policy.policy,
                    config,
                    adaptive,
                    active_a12,
                    query_type,
                    cluster_id,
                    config.adaptive.min_samples_alpha,
                    crate::store::a12_calibration::A12_DEFAULT_NOISE_FLOOR,
                    now_millis,
                ),
                None => resolve_runtime_recall_fusion(
                    &policy.policy,
                    config,
                    adaptive,
                    active_a12,
                    query_type,
                    cluster_id,
                    config.adaptive.min_samples_alpha,
                    crate::store::a12_calibration::A12_DEFAULT_NOISE_FLOOR,
                    now_millis,
                ),
            },
            None => runtime_disabled(
                Some(policy_key.clone()),
                evidence.map_or(ArsRecallFusionEvidenceBasis::Static, |value| value.basis),
                RecallFusionScopeHealthCode::Inactive,
                "scope cannot be represented by the runtime resolver",
            ),
        }
    } else {
        runtime_disabled(
            Some(policy_key.clone()),
            evidence.map_or(ArsRecallFusionEvidenceBasis::Static, |value| value.basis),
            RecallFusionScopeHealthCode::Inactive,
            format!(
                "parameter policy load status is {}",
                parameter_policy_load_status_name(&policy.status)
            ),
        )
    };
    let valid_now = entry.is_some_and(|entry| {
        entry.is_current_for_at(
            &active_a12.state,
            crate::store::a12_calibration::A12_DEFAULT_NOISE_FLOOR,
            now_millis,
        )
    });
    let exact_human = adaptive
        .learned_shadow_fusion
        .get(scope)
        .is_some_and(|entry| {
            entry.sample_count >= config.adaptive.min_samples_alpha.max(10)
                && human_simplex(entry).is_some()
        });
    let owns_resolution = resolution.scope_key.as_deref() == Some(policy_key.as_str())
        || (resolution.scope_key.is_none()
            && resolution.basis == ArsRecallFusionEvidenceBasis::Human
            && exact_human);
    let automatic_basis = matches!(
        resolution.basis,
        ArsRecallFusionEvidenceBasis::SelfSupervised | ArsRecallFusionEvidenceBasis::Blended
    );
    let live_attestation_blocker = (owns_resolution
        && automatic_basis
        && resolution.adoption_weight > f64::EPSILON
        && resolution.simplex.is_some())
    .then(|| automatic_live_attestation_blocker(evidence, current_recall_gate, calibration_run))
    .flatten();
    let active = owns_resolution
        && live_attestation_blocker.is_none()
        && resolution.adoption_weight > f64::EPSILON
        && resolution.simplex.is_some();
    let health_code = if !owns_resolution {
        RecallFusionScopeHealthCode::Inactive
    } else if let Some((code, _)) = &live_attestation_blocker {
        *code
    } else {
        resolution.code
    };
    let reason = if !owns_resolution {
        resolution.scope_key.as_ref().map_or_else(
            || "runtime fallback is not owned by this exact recall-fusion scope".to_string(),
            |resolved| format!("runtime resolves this request through broader scope {resolved}"),
        )
    } else if let Some((_, blocker)) = live_attestation_blocker {
        blocker
    } else {
        resolution.reason.clone()
    };
    RecallFusionActivationScopeReport {
        scope: scope.to_string(),
        policy_key,
        resolved_policy_key: resolution.scope_key.clone(),
        basis: resolution.basis,
        sealed_basis: evidence.map(|value| value.basis),
        health_code,
        reason,
        effective_simplex: active.then_some(resolution.simplex).flatten(),
        adoption_weight: if active {
            resolution.adoption_weight
        } else {
            0.0
        },
        human_ess: evidence.map_or_else(
            || {
                adaptive
                    .learned_shadow_fusion
                    .get(scope)
                    .and_then(|entry| u64::try_from(entry.sample_count).ok())
                    .unwrap_or(0)
            },
            |value| value.human_ess,
        ),
        train_family_ess: entry.map_or_else(
            || evidence.map_or(0, |value| value.self_supervised_train_family_ess),
            |value| value.train_family_ess,
        ),
        holdout_family_ess: entry.map_or_else(
            || evidence.map_or(0, |value| value.self_supervised_holdout_family_ess),
            |value| value.holdout_family_ess,
        ),
        verdict: entry
            .map(|value| value.verdict)
            .or_else(|| evidence.and_then(|value| value.a12_verdict)),
        paired_top3: entry.map(|value| value.paired_top3),
        provenance: entry.map(|value| value.provenance),
        provenance_holdout: entry.and_then(|value| value.provenance_holdout),
        provenance_direction_conflict: entry
            .and_then(|value| value.provenance_holdout)
            .is_some_and(|stats| stats.direction_conflict()),
        a12_generation: evidence
            .and_then(|value| value.a12_generation)
            .or_else(|| entry.map(|_| active_a12.state.generation)),
        a12_revision: evidence
            .and_then(|value| value.a12_revision)
            .or_else(|| entry.map(|_| active_a12.state.revision)),
        source_adaptive_version: policy.policy.source_adaptive_version,
        source_snapshot_fingerprint: entry
            .map(|value| value.source_snapshot_fingerprint.clone())
            .filter(|fingerprint| !fingerprint.is_empty())
            .or_else(|| calibration_run.map(|run| run.source_snapshot_fingerprint.clone())),
        train_case_count: entry.map(|value| value.train_case_count),
        holdout_reason: entry
            .map(|value| value.holdout_reason.clone())
            .filter(|reason| !reason.is_empty()),
        generation_fingerprint: evidence
            .and_then(|value| value.generation_fingerprint.clone())
            .or_else(|| entry.map(|value| value.generation_fingerprint.clone())),
        corpus_fingerprint: evidence
            .and_then(|value| value.corpus_fingerprint.clone())
            .or_else(|| entry.map(|value| value.corpus_fingerprint.clone())),
        training_fingerprint: entry.map(|value| value.training_fingerprint.clone()),
        holdout_fingerprint: entry.map(|value| value.holdout_fingerprint.clone()),
        optimizer_fingerprint: evidence
            .and_then(|value| value.optimizer_fingerprint.clone())
            .or_else(|| entry.map(|value| value.optimizer_fingerprint.clone())),
        evaluation_fingerprint: evidence
            .and_then(|value| value.evaluation_fingerprint.clone())
            .or_else(|| entry.map(|value| value.evaluation_fingerprint.clone())),
        recall_gate_status: Some(current_recall_gate.status),
        recall_gate_build_fingerprint: current_recall_gate.build_fingerprint.clone(),
        recall_gate_fixture_fingerprint: current_recall_gate.fixture_fingerprint.clone(),
        recall_gate_evaluated_at: current_recall_gate.evaluated_at,
        sealed_recall_gate_status: evidence.map(|value| value.recall_gate_status),
        sealed_recall_gate_build_fingerprint: evidence
            .and_then(|value| value.recall_gate_build_fingerprint.clone()),
        sealed_recall_gate_fixture_fingerprint: evidence
            .and_then(|value| value.recall_gate_fixture_fingerprint.clone()),
        sealed_recall_gate_evaluated_at: evidence.and_then(|value| value.recall_gate_evaluated_at),
        calibrated_at: entry
            .map(|value| value.calibrated_at)
            .or_else(|| evidence.and_then(|value| value.calibrated_at)),
        evaluated_at: entry
            .map(|value| value.evaluated_at)
            .or_else(|| evidence.and_then(|value| value.evaluated_at)),
        valid_until_exclusive: entry
            .and_then(|value| value.valid_until_exclusive)
            .or_else(|| evidence.and_then(|value| value.a12_valid_until_exclusive)),
        valid_now,
        active,
    }
}

fn runtime_scope_parts(scope: &str) -> Option<(&str, Option<u32>)> {
    match scope.split_once(':') {
        Some((query_type, cluster_id)) => Some((query_type, Some(cluster_id.parse::<u32>().ok()?))),
        None => Some((scope, None)),
    }
}

fn complete_calibration_run(run: &RecallFusionCalibrationRunAttestation) -> bool {
    run.phase == RecallFusionCalibrationRunPhase::Complete
        && !run.source_snapshot_fingerprint.is_empty()
        && !run.behavior_config_fingerprint.is_empty()
}

fn automatic_live_attestation_blocker(
    evidence: Option<&ArsRecallFusionEvidence>,
    current_gate: &RecallEvalGateAttestation,
    calibration_run: Option<&RecallFusionCalibrationRunAttestation>,
) -> Option<(RecallFusionScopeHealthCode, String)> {
    if calibration_run.is_none_or(|run| !complete_calibration_run(run)) {
        return Some((
            RecallFusionScopeHealthCode::RunIncomplete,
            "current A12 calibration run is not complete".to_string(),
        ));
    }
    if current_gate.status != ArsRecallGateStatus::Ship {
        return Some((
            RecallFusionScopeHealthCode::GateNotShip,
            format!(
                "current recall eval gate is {}",
                recall_gate_status_name(current_gate.status)
            ),
        ));
    }
    if !current_ship_gate_identity(
        current_gate.status,
        current_gate.build_fingerprint.as_deref(),
        current_gate.fixture_fingerprint.as_deref(),
        current_gate.evaluated_at,
    ) {
        return Some((
            RecallFusionScopeHealthCode::GateNotShip,
            "current recall eval gate is not current Ship evidence".to_string(),
        ));
    }
    let Some(evidence) = evidence else {
        return Some((
            RecallFusionScopeHealthCode::Tampered,
            "sealed automatic recall evidence is missing".to_string(),
        ));
    };
    // Strict exact-match on the sealed identity is fail-closed by design: a
    // newer Ship re-run of the eval gate still blocks automatic activation
    // until the next policy refresh reseals against it. The blocker self-heals
    // at that reseal; nothing here rewrites the sealed evidence.
    if evidence.recall_gate_status != current_gate.status
        || evidence.recall_gate_build_fingerprint != current_gate.build_fingerprint
        || evidence.recall_gate_fixture_fingerprint != current_gate.fixture_fingerprint
        || evidence.recall_gate_evaluated_at != current_gate.evaluated_at
    {
        return Some((
            RecallFusionScopeHealthCode::FingerprintMismatch,
            "current recall eval gate identity mismatched sealed policy evidence".to_string(),
        ));
    }
    None
}

fn recall_gate_status_name(status: ArsRecallGateStatus) -> &'static str {
    match status {
        ArsRecallGateStatus::Ship => "ship",
        ArsRecallGateStatus::Bail => "bail",
        ArsRecallGateStatus::NoData => "no_data",
    }
}

fn recall_fusion_report_health(
    policy: &crate::store::ars_parameter_policy::ArsParameterPolicyLoad,
    a12_status: A12CalibrationLoadStatus,
    active: bool,
    scopes: &[RecallFusionActivationScopeReport],
) -> &'static str {
    use crate::store::ars_parameter_policy::ArsParameterPolicyLoadStatus;

    match policy.status {
        ArsParameterPolicyLoadStatus::Missing => return "policy_missing",
        ArsParameterPolicyLoadStatus::Corrupt => return "policy_corrupt",
        ArsParameterPolicyLoadStatus::UnsupportedSchema => {
            return "policy_unsupported_schema";
        }
        ArsParameterPolicyLoadStatus::StorageError => return "policy_storage_error",
        ArsParameterPolicyLoadStatus::Loaded => {}
    }
    match a12_status {
        A12CalibrationLoadStatus::Missing => return "a12_missing",
        A12CalibrationLoadStatus::Corrupt => return "a12_corrupt",
        A12CalibrationLoadStatus::UnsupportedSchema => return "a12_unsupported_schema",
        A12CalibrationLoadStatus::StorageError => return "a12_storage_error",
        A12CalibrationLoadStatus::Loaded => {}
    }
    if scopes.is_empty() {
        return "missing";
    }
    if scopes
        .iter()
        .any(|scope| scope.verdict == Some(A12CalibrationVerdict::Bail))
    {
        "bail"
    } else if scopes
        .iter()
        .any(|scope| scope.verdict == Some(A12CalibrationVerdict::NoData))
    {
        "no_data"
    } else if !active
        && scopes.iter().all(|scope| {
            scope.basis == ArsRecallFusionEvidenceBasis::Static
                || scope.sealed_basis == Some(ArsRecallFusionEvidenceBasis::Static)
        })
    {
        "static"
    } else if scopes.iter().any(|scope| scope.health_code.is_degraded()) {
        // Keyed on the typed code, never on reason prose: a reworded resolver
        // message must not be able to silently mute the degraded rollup.
        "degraded"
    } else if active {
        "healthy"
    } else {
        "inactive"
    }
}

fn a12_load_status_name(status: A12CalibrationLoadStatus) -> &'static str {
    match status {
        A12CalibrationLoadStatus::Missing => "missing",
        A12CalibrationLoadStatus::Loaded => "loaded",
        A12CalibrationLoadStatus::Corrupt => "corrupt",
        A12CalibrationLoadStatus::UnsupportedSchema => "unsupported_schema",
        A12CalibrationLoadStatus::StorageError => "storage_error",
    }
}

fn parameter_policy_load_status_name(
    status: &crate::store::ars_parameter_policy::ArsParameterPolicyLoadStatus,
) -> String {
    use crate::store::ars_parameter_policy::ArsParameterPolicyLoadStatus;
    match status {
        ArsParameterPolicyLoadStatus::Missing => "missing",
        ArsParameterPolicyLoadStatus::Loaded => "loaded",
        ArsParameterPolicyLoadStatus::Corrupt => "corrupt",
        ArsParameterPolicyLoadStatus::UnsupportedSchema => "unsupported_schema",
        ArsParameterPolicyLoadStatus::StorageError => "storage_error",
    }
    .to_string()
}

fn parameter_policy_mode_name(mode: ArsParameterPolicyMode) -> &'static str {
    match mode {
        ArsParameterPolicyMode::Disabled => "disabled",
        ArsParameterPolicyMode::Shadow => "shadow",
        ArsParameterPolicyMode::Canary => "canary",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ReinConfig;
    use crate::eval::gates::{
        FixtureResult, GateScorecard, ScorecardKind, SCORECARD_SCHEMA_VERSION,
    };
    use crate::store::a12_calibration::{
        A12CalibrationLoad, A12CalibrationLoadStatus, A12CalibrationScope, A12CalibrationState,
        A12CalibrationVerdict, A12FusionSimplex, A12PairedTop3Stats, A12ProvenanceCounts,
        A12ScopeEntry,
    };
    use crate::store::adaptive::{
        AdaptiveState, JudgeCalibrationState, LearnedShadowFusionEntry, ShadowFusionWeightEntry,
    };
    use crate::store::ars_parameter_policy::{
        ArsParameterPolicy, ArsParameterPolicyMode, ArsRecallFusionEvidence,
        ArsRecallFusionEvidenceBasis, ArsRecallGateStatus, ARS_PARAMETER_POLICY_SCHEMA_VERSION,
    };
    use std::collections::{BTreeMap, HashMap};
    use std::path::Path;

    fn scorecard(kind: ScorecardKind, hits: &[bool]) -> GateScorecard {
        GateScorecard {
            schema_version: SCORECARD_SCHEMA_VERSION,
            gate_name: "recall".to_string(),
            kind,
            created_at: 1_700_000_000,
            rein_version: env!("CARGO_PKG_VERSION").to_string(),
            build_fingerprint: env!("REIN_BUILD_FINGERPRINT").to_string(),
            fixture_fingerprint: "fixture-fingerprint".to_string(),
            fixture_count: hits.len(),
            score: hits.iter().filter(|hit| **hit).count() as f64 / hits.len() as f64,
            per_fixture: hits
                .iter()
                .enumerate()
                .map(|(index, hit)| FixtureResult {
                    fixture_id: format!("fixture-{index}"),
                    hit: *hit,
                })
                .collect(),
        }
    }

    fn write_scorecard(path: &Path, scorecard: &GateScorecard) {
        std::fs::write(path, serde_json::to_vec(scorecard).unwrap()).unwrap();
    }

    fn gate(status: ArsRecallGateStatus) -> RecallEvalGateAttestation {
        RecallEvalGateAttestation {
            status,
            reason_code: RecallEvalGateReasonCode::Compared,
            build_fingerprint: Some(env!("REIN_BUILD_FINGERPRINT").to_string()),
            fixture_fingerprint: Some("recall-fixtures".to_string()),
            evaluated_at: Some(1_700_000_100),
            reason: "test gate".to_string(),
        }
    }

    fn runtime_config() -> ReinConfig {
        let mut config = ReinConfig::default();
        config.adaptive.enabled = true;
        config.ars.acceleration.enabled = true;
        config.ars.acceleration.shadow_only = false;
        config.adaptive.shrinkage_prior = 5.0;
        config
    }

    fn expected_runtime_simplex(
        learned: A12FusionSimplex,
        evidence_count: u64,
        adoption_weight: f64,
        config: &ReinConfig,
    ) -> A12FusionSimplex {
        let static_prior = crate::search::alpha_optimizer::ShadowFusionWeights::default();
        let effective = crate::ops::ars_tuning::effective_simplex(
            [
                static_prior.bm25,
                static_prior.vec,
                static_prior.kg,
                static_prior.episode,
                static_prior.support,
                static_prior.diversity,
            ],
            [
                learned.bm25,
                learned.vector,
                learned.kg,
                learned.episode,
                learned.support,
                learned.diversity,
            ],
            crate::ops::ars_tuning::TrustInputs {
                enabled: config.ars.acceleration.enabled,
                production_canary: adoption_weight > f64::EPSILON,
                runtime_adoption_weight: adoption_weight,
                human_count: evidence_count,
                llm_count: 0,
                llm_reliability: 0.0,
                calibration: 1.0,
                stability: 1.0,
                drift_alert: false,
                prior_strength: config.adaptive.shrinkage_prior,
                max_trust: 0.85,
            },
        );
        simplex(effective)
    }

    fn simplex(values: [f64; 6]) -> A12FusionSimplex {
        A12FusionSimplex {
            bm25: values[0],
            vector: values[1],
            kg: values[2],
            episode: values[3],
            support: values[4],
            diversity: values[5],
        }
    }

    fn assert_simplex_close(left: A12FusionSimplex, right: A12FusionSimplex) {
        for (left, right) in [
            (left.bm25, right.bm25),
            (left.vector, right.vector),
            (left.kg, right.kg),
            (left.episode, right.episode),
            (left.support, right.support),
            (left.diversity, right.diversity),
        ] {
            assert!((left - right).abs() <= 1e-12, "{left} != {right}");
        }
    }

    fn human_entry(values: [f64; 6], sample_count: usize) -> LearnedShadowFusionEntry {
        LearnedShadowFusionEntry {
            weights: ShadowFusionWeightEntry {
                bm25: values[0],
                vec: values[1],
                kg: values[2],
                episode: values[3],
                support: values[4],
                diversity: values[5],
            },
            sample_count,
            last_updated: "2026-07-13T00:00:00Z".to_string(),
        }
    }

    fn paired_ship(n: u32) -> A12PairedTop3Stats {
        let result = crate::eval::mcnemar::mcnemar_from_counts(n, 0, 0, 0).unwrap();
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

    fn a12_loaded(
        scope: A12CalibrationScope,
        values: [f64; 6],
        train_ess: u64,
        holdout_ess: u64,
        verdict: A12CalibrationVerdict,
    ) -> A12CalibrationLoad {
        let key = scope.key();
        let cluster_generation = scope.is_cluster().then_some(7);
        let entry = A12ScopeEntry {
            scope,
            canonical_generation: 11,
            generation_fingerprint: "generation-fingerprint".to_string(),
            source_snapshot_fingerprint: "source-snapshot-fingerprint".to_string(),
            snapshot_cutoff: 1_700_000_000,
            corpus_fingerprint: "corpus-fingerprint".to_string(),
            train_family_ess: train_ess,
            train_case_count: train_ess,
            holdout_family_ess: holdout_ess,
            simplex: simplex(values),
            verdict,
            noise_floor: 0.02,
            paired_top3: paired_ship(u32::try_from(holdout_ess).unwrap()),
            provenance: A12ProvenanceCounts {
                canonical_loo: train_ess,
                concept_loo: 0,
                episode_loo: 0,
            },
            provenance_holdout: None,
            training_fingerprint: "training-fingerprint".to_string(),
            holdout_fingerprint: "holdout-fingerprint".to_string(),
            optimizer_fingerprint: "optimizer-fingerprint".to_string(),
            evaluation_fingerprint: "evaluation-fingerprint".to_string(),
            holdout_reason: "holdout evaluated".to_string(),
            calibrated_at: 1_700_000_000,
            evaluated_at: 1_700_000_050,
            valid_until_exclusive: None,
            cluster_generation,
            invalidation: None,
        };
        A12CalibrationLoad {
            state: A12CalibrationState {
                schema_version: crate::store::a12_calibration::A12_CALIBRATION_SCHEMA_VERSION,
                revision: 4,
                generation: 11,
                generation_fingerprint: "generation-fingerprint".to_string(),
                snapshot_cutoff: 1_700_000_000,
                corpus_fingerprint: "corpus-fingerprint".to_string(),
                cluster_generation: 7,
                scopes: BTreeMap::from([(key, entry)]),
                created_at: 1_700_000_000,
                updated_at: 1_700_000_050,
                run: Some(crate::store::a12_calibration::A12CalibrationRunMetadata {
                    phase: crate::store::a12_calibration::A12CalibrationPhase::Complete,
                    source_input_epoch: 0,
                    source_snapshot_fingerprint: "source-snapshot-fingerprint".to_string(),
                    behavior_config_fingerprint: "behavior-config-fingerprint".to_string(),
                }),
            },
            status: A12CalibrationLoadStatus::Loaded,
            error: None,
        }
    }

    fn seal_current_live_inputs(
        store: &crate::store::SqliteStore,
        config: &ReinConfig,
        calibration: &mut A12CalibrationLoad,
    ) {
        let source =
            crate::ops::a12_autocalibration::a12_source_snapshot_fingerprint(store).unwrap();
        let source_input_epoch =
            crate::store::a12_calibration::load_a12_input_epoch(store.conn()).unwrap();
        let hard_dedup_bound =
            crate::ops::effective_hard_dedup_threshold_from_conn(store.conn(), config);
        let behavior = crate::ops::a12_autocalibration::a12_behavior_config_fingerprint(
            config,
            hard_dedup_bound,
            crate::ops::adaptive::A12_RECALL_TRACE_LIMIT,
            config.adaptive.min_samples_alpha,
        )
        .unwrap();
        let run = calibration.state.run.as_mut().unwrap();
        run.source_input_epoch = source_input_epoch;
        run.source_snapshot_fingerprint = source.clone();
        run.behavior_config_fingerprint = behavior;
        for entry in calibration.state.scopes.values_mut() {
            entry.source_snapshot_fingerprint = source.clone();
        }
    }

    fn canary_policy(
        source_adaptive_version: u64,
        runtime_adoption_weight: f64,
        adoption_weights: HashMap<String, f64>,
        evidence: BTreeMap<String, ArsRecallFusionEvidence>,
    ) -> ArsParameterPolicy {
        ArsParameterPolicy {
            schema_version: ARS_PARAMETER_POLICY_SCHEMA_VERSION,
            revision: 1,
            mode: ArsParameterPolicyMode::Canary,
            disabled_reason: None,
            source_adaptive_version,
            runtime_adoption_weight,
            adoption_weights,
            recall_fusion_evidence: evidence.into_iter().collect(),
            last_event_id: 0,
            last_updated: "2026-07-13T00:00:00Z".to_string(),
        }
    }

    fn loaded_policy(
        policy: ArsParameterPolicy,
    ) -> crate::store::ars_parameter_policy::ArsParameterPolicyLoad {
        crate::store::ars_parameter_policy::ArsParameterPolicyLoad {
            policy,
            status: crate::store::ars_parameter_policy::ArsParameterPolicyLoadStatus::Loaded,
            error: None,
        }
    }

    #[test]
    fn activation_report_projects_ship_evidence_through_runtime_resolver() {
        let config = runtime_config();
        let adaptive = AdaptiveState {
            version: 7,
            ..AdaptiveState::default()
        };
        let a12 = a12_loaded(
            A12CalibrationScope::Global,
            [0.10, 0.20, 0.30, 0.15, 0.15, 0.10],
            12,
            12,
            A12CalibrationVerdict::Ship,
        );
        let evidence = resolve_recall_fusion_evidence(
            &adaptive,
            &a12,
            10,
            0.02,
            1_700_000_060_000,
            &gate(ArsRecallGateStatus::Ship),
        );
        let policy = loaded_policy(canary_policy(
            adaptive.version,
            0.0,
            HashMap::from([("recall_fusion:global".to_string(), 0.25)]),
            evidence,
        ));
        let run = recall_fusion_calibration_run_attestation(&a12);

        let report = recall_fusion_activation_report(
            &config,
            &adaptive,
            &policy,
            &a12,
            &gate(ArsRecallGateStatus::Ship),
            run.as_ref(),
            1_700_000_060_000,
        );

        assert_eq!(report.activation_status, "active");
        assert_eq!(report.health_status, "healthy");
        assert!(report.run_complete);
        assert!(report.active);
        assert_eq!(report.source_adaptive_version, adaptive.version);
        assert_eq!(report.a12_generation, Some(11));
        assert_eq!(report.a12_revision, Some(4));
        let scope = &report.scopes[0];
        assert_eq!(scope.scope, "global");
        assert_eq!(scope.basis, ArsRecallFusionEvidenceBasis::SelfSupervised);
        assert_eq!(scope.adoption_weight, 0.25);
        assert!(scope.active);
        assert!(scope.effective_simplex.is_some());
        assert_eq!(scope.train_family_ess, 12);
        assert_eq!(scope.holdout_family_ess, 12);
        assert_eq!(scope.verdict, Some(A12CalibrationVerdict::Ship));
        assert_eq!(scope.paired_top3.as_ref().map(|stats| stats.n), Some(12));
        assert_eq!(
            scope.provenance.as_ref().map(|counts| counts.canonical_loo),
            Some(12)
        );
        assert_eq!(
            scope.training_fingerprint.as_deref(),
            Some("training-fingerprint")
        );
        assert_eq!(
            scope.holdout_fingerprint.as_deref(),
            Some("holdout-fingerprint")
        );
        assert_eq!(
            scope.recall_gate_build_fingerprint.as_deref(),
            Some(env!("REIN_BUILD_FINGERPRINT"))
        );
        assert!(scope.valid_now);
        // Rows without provenance diagnostics render without the optional
        // field and never report a conflict.
        assert_eq!(scope.provenance_holdout, None);
        assert!(!scope.provenance_direction_conflict);
        let json = serde_json::to_value(&report).unwrap();
        assert_eq!(json["schema_version"], 2);
        assert!(json["scopes"][0].get("provenance_holdout").is_none());
        assert_eq!(json["scopes"][0]["provenance_direction_conflict"], false);
    }

    #[test]
    fn activation_report_projects_provenance_disagreement_diagnostics() {
        use crate::store::a12_calibration::{A12ProvenanceHoldoutCells, A12ProvenanceHoldoutStats};

        let config = runtime_config();
        let adaptive = AdaptiveState {
            version: 7,
            ..AdaptiveState::default()
        };
        let mut a12 = a12_loaded(
            A12CalibrationScope::Global,
            [0.10, 0.20, 0.30, 0.15, 0.15, 0.10],
            12,
            12,
            A12CalibrationVerdict::Ship,
        );
        let stats = A12ProvenanceHoldoutStats {
            canonical_loo: A12ProvenanceHoldoutCells {
                family_count: 8,
                both_hit: 5,
                baseline_only: 0,
                treatment_only: 3,
                neither_hit: 0,
            },
            concept_loo: A12ProvenanceHoldoutCells {
                family_count: 4,
                both_hit: 2,
                baseline_only: 1,
                treatment_only: 0,
                neither_hit: 1,
            },
            episode_loo: A12ProvenanceHoldoutCells::default(),
        };
        a12.state
            .scopes
            .get_mut("global")
            .unwrap()
            .provenance_holdout = Some(stats);
        let evidence = resolve_recall_fusion_evidence(
            &adaptive,
            &a12,
            10,
            0.02,
            1_700_000_060_000,
            &gate(ArsRecallGateStatus::Ship),
        );
        let policy = loaded_policy(canary_policy(
            adaptive.version,
            0.0,
            HashMap::from([("recall_fusion:global".to_string(), 0.25)]),
            evidence,
        ));
        let run = recall_fusion_calibration_run_attestation(&a12);

        let report = recall_fusion_activation_report(
            &config,
            &adaptive,
            &policy,
            &a12,
            &gate(ArsRecallGateStatus::Ship),
            run.as_ref(),
            1_700_000_060_000,
        );

        assert_eq!(report.schema_version, 2);
        let scope = &report.scopes[0];
        assert_eq!(scope.provenance_holdout, Some(stats));
        assert!(scope.provenance_direction_conflict);
        let json = serde_json::to_value(&report).unwrap();
        assert_eq!(
            json["scopes"][0]["provenance_holdout"]["canonical_loo"]["treatment_only"],
            3
        );
        assert_eq!(
            json["scopes"][0]["provenance_holdout"]["concept_loo"]["baseline_only"],
            1
        );
        assert_eq!(json["scopes"][0]["provenance_direction_conflict"], true);
    }

    #[test]
    fn activation_report_fails_closed_on_expiry_and_never_exposes_paths() {
        let config = runtime_config();
        let adaptive = AdaptiveState {
            version: 7,
            ..AdaptiveState::default()
        };
        let mut a12 = a12_loaded(
            A12CalibrationScope::Global,
            [0.10, 0.20, 0.30, 0.15, 0.15, 0.10],
            12,
            12,
            A12CalibrationVerdict::Ship,
        );
        a12.state
            .scopes
            .get_mut("global")
            .unwrap()
            .valid_until_exclusive = Some(1_700_000_061_000);
        let evidence = resolve_recall_fusion_evidence(
            &adaptive,
            &a12,
            10,
            0.02,
            1_700_000_060_000,
            &gate(ArsRecallGateStatus::Ship),
        );
        let policy = loaded_policy(canary_policy(
            adaptive.version,
            0.0,
            HashMap::from([("recall_fusion:global".to_string(), 0.25)]),
            evidence,
        ));
        let run = recall_fusion_calibration_run_attestation(&a12);

        let report = recall_fusion_activation_report(
            &config,
            &adaptive,
            &policy,
            &a12,
            &gate(ArsRecallGateStatus::Ship),
            run.as_ref(),
            1_700_000_061_000,
        );
        let scope = &report.scopes[0];
        assert_eq!(report.activation_status, "inactive");
        assert_eq!(report.health_status, "degraded");
        assert!(!report.active);
        assert!(!scope.active);
        assert!(!scope.valid_now);
        assert!(scope.reason.contains("stale or expired"));
        let json = serde_json::to_string(&report).unwrap();
        assert!(!json.contains("/Users/"));
        assert!(!json.contains("docs/eval-baselines"));
        assert!(!json.contains("target/eval-gates"));
    }

    /// P3-2: an ACTIVE Blended scope that is currently serving only its
    /// sealed human fallback (automatic candidate expired) must classify as
    /// degraded `HumanFallback`, never healthy.
    #[test]
    fn active_sealed_human_fallback_for_blocked_blended_candidate_is_degraded() {
        let config = runtime_config();
        let mut adaptive = AdaptiveState {
            version: 7,
            ..AdaptiveState::default()
        };
        adaptive.learned_shadow_fusion.insert(
            "global".to_string(),
            human_entry([0.20, 0.25, 0.20, 0.15, 0.10, 0.10], 12),
        );
        let mut a12 = a12_loaded(
            A12CalibrationScope::Global,
            [0.10, 0.20, 0.30, 0.15, 0.15, 0.10],
            12,
            12,
            A12CalibrationVerdict::Ship,
        );
        a12.state
            .scopes
            .get_mut("global")
            .unwrap()
            .valid_until_exclusive = Some(1_700_000_061_000);
        let mut evidence = resolve_recall_fusion_evidence(
            &adaptive,
            &a12,
            10,
            0.02,
            1_700_000_060_000,
            &gate(ArsRecallGateStatus::Ship),
        );
        let sealed = evidence.get_mut("recall_fusion:global").unwrap();
        assert_eq!(sealed.basis, ArsRecallFusionEvidenceBasis::Blended);
        // The production sealer records the human fallback adoption; mirror it.
        sealed.human_runtime_adoption_weight = Some(0.25);
        let policy = loaded_policy(canary_policy(
            adaptive.version,
            0.0,
            HashMap::from([("recall_fusion:global".to_string(), 0.40)]),
            evidence,
        ));
        let run = recall_fusion_calibration_run_attestation(&a12);

        let report = recall_fusion_activation_report(
            &config,
            &adaptive,
            &policy,
            &a12,
            &gate(ArsRecallGateStatus::Ship),
            run.as_ref(),
            1_700_000_061_000,
        );

        let scope = &report.scopes[0];
        assert!(report.active);
        assert!(scope.active, "{}", scope.reason);
        assert_eq!(scope.basis, ArsRecallFusionEvidenceBasis::Human);
        assert_eq!(
            scope.sealed_basis,
            Some(ArsRecallFusionEvidenceBasis::Blended)
        );
        assert_eq!(
            scope.health_code,
            RecallFusionScopeHealthCode::HumanFallback
        );
        assert_eq!(report.activation_status, "active");
        assert_eq!(report.health_status, "degraded");
        let json = serde_json::to_value(&report).unwrap();
        assert_eq!(json["scopes"][0]["health_code"], "human_fallback");
    }

    #[test]
    fn recall_gate_attestation_is_no_data_when_artifacts_are_missing() {
        let temp = tempfile::tempdir().unwrap();

        let attestation = recall_eval_gate_attestation(
            &temp.path().join("baseline.json"),
            &temp.path().join("run.json"),
            0.02,
        );

        assert_eq!(attestation.status, ArsRecallGateStatus::NoData);
        assert_eq!(attestation.build_fingerprint, None);
        assert_eq!(attestation.fixture_fingerprint, None);
        assert_eq!(attestation.evaluated_at, None);
        assert!(attestation.reason.contains("baseline"));
    }

    #[test]
    fn recall_gate_path_resolver_prefers_absolute_operator_root() {
        let temp = tempfile::tempdir().unwrap();
        let operator_root = temp.path().join("operator-root");
        let cwd_root = temp.path().join("checkout");
        std::fs::create_dir_all(cwd_root.join("docs/eval-baselines")).unwrap();

        let paths = resolve_recall_eval_gate_artifact_paths(
            Some(operator_root.as_path()),
            cwd_root.as_path(),
        )
        .expect("absolute operator root must be authoritative");

        assert_eq!(
            paths.baseline,
            operator_root.join("docs/eval-baselines/recall.json")
        );
        assert_eq!(
            paths.run,
            operator_root.join("target/eval-gates/recall-run.json")
        );
    }

    #[test]
    fn recall_gate_path_resolver_discovers_checkout_ancestor() {
        let temp = tempfile::tempdir().unwrap();
        let repo_root = temp.path().join("checkout");
        let nested = repo_root.join("crates/rein");
        std::fs::create_dir_all(repo_root.join("docs/eval-baselines")).unwrap();
        std::fs::create_dir_all(&nested).unwrap();

        let paths = resolve_recall_eval_gate_artifact_paths(None, &nested)
            .expect("development checkout must be discovered from a nested cwd");

        assert_eq!(
            paths.baseline,
            repo_root.join("docs/eval-baselines/recall.json")
        );
        assert_eq!(
            paths.run,
            repo_root.join("target/eval-gates/recall-run.json")
        );
    }

    #[test]
    fn recall_gate_path_resolver_rejects_relative_operator_root() {
        let error = resolve_recall_eval_gate_artifact_paths(
            Some(Path::new("relative-checkout")),
            Path::new("/daemon-cwd"),
        )
        .unwrap_err();

        assert_eq!(error, RecallEvalGatePathError::RelativeOperatorRoot);
    }

    #[test]
    fn recall_gate_path_resolver_requires_explicit_root_outside_checkout() {
        let temp = tempfile::tempdir().unwrap();
        let error = resolve_recall_eval_gate_artifact_paths(None, temp.path()).unwrap_err();

        assert_eq!(error, RecallEvalGatePathError::Unconfigured);
    }

    #[test]
    fn recall_gate_attestation_seals_current_ship_identity() {
        let temp = tempfile::tempdir().unwrap();
        let baseline_path = temp.path().join("baseline.json");
        let run_path = temp.path().join("run.json");
        let hits = vec![true; 20];
        write_scorecard(&baseline_path, &scorecard(ScorecardKind::Baseline, &hits));
        write_scorecard(&run_path, &scorecard(ScorecardKind::Run, &hits));

        let attestation = recall_eval_gate_attestation(&baseline_path, &run_path, 0.02);

        assert_eq!(attestation.status, ArsRecallGateStatus::Ship);
        assert_eq!(
            attestation.build_fingerprint.as_deref(),
            Some(env!("REIN_BUILD_FINGERPRINT"))
        );
        assert_eq!(
            attestation.fixture_fingerprint.as_deref(),
            Some("fixture-fingerprint")
        );
        assert_eq!(attestation.evaluated_at, Some(1_700_000_000));
        assert!(!attestation.reason.is_empty());
    }

    #[test]
    fn recall_gate_attestation_seals_bail_identity() {
        let temp = tempfile::tempdir().unwrap();
        let baseline_path = temp.path().join("baseline.json");
        let run_path = temp.path().join("run.json");
        write_scorecard(
            &baseline_path,
            &scorecard(ScorecardKind::Baseline, &[true; 20]),
        );
        write_scorecard(&run_path, &scorecard(ScorecardKind::Run, &[false; 20]));

        let attestation = recall_eval_gate_attestation(&baseline_path, &run_path, 0.02);

        assert_eq!(attestation.status, ArsRecallGateStatus::Bail);
        assert_eq!(
            attestation.build_fingerprint.as_deref(),
            Some(env!("REIN_BUILD_FINGERPRINT"))
        );
        assert_eq!(
            attestation.fixture_fingerprint.as_deref(),
            Some("fixture-fingerprint")
        );
        assert_eq!(attestation.evaluated_at, Some(1_700_000_000));
        assert!(!attestation.reason.is_empty());
    }

    #[test]
    fn recall_gate_attestation_redacts_corrupt_artifact_paths_and_errors() {
        let temp = tempfile::tempdir().unwrap();
        let baseline_path = temp.path().join("private-secret-baseline.json");
        let run_path = temp.path().join("private-secret-run.json");
        std::fs::write(&baseline_path, b"{secret invalid json").unwrap();

        let corrupt_baseline = recall_eval_gate_attestation(&baseline_path, &run_path, 0.02);

        assert_eq!(
            corrupt_baseline.reason_code,
            RecallEvalGateReasonCode::CorruptBaseline
        );
        assert_eq!(
            corrupt_baseline.reason,
            "recall baseline scorecard is corrupt"
        );
        assert!(!corrupt_baseline
            .reason
            .contains(&temp.path().display().to_string()));
        assert!(!corrupt_baseline.reason.contains("private-secret"));

        write_scorecard(
            &baseline_path,
            &scorecard(ScorecardKind::Baseline, &[true; 20]),
        );
        std::fs::write(&run_path, b"{another secret invalid json").unwrap();

        let corrupt_run = recall_eval_gate_attestation(&baseline_path, &run_path, 0.02);

        assert_eq!(
            corrupt_run.reason_code,
            RecallEvalGateReasonCode::CorruptRun
        );
        assert_eq!(corrupt_run.reason, "recall run scorecard is corrupt");
        assert!(!corrupt_run
            .reason
            .contains(&temp.path().display().to_string()));
        assert!(!corrupt_run.reason.contains("private-secret"));
    }

    #[test]
    fn evidence_resolver_preserves_human_only_simplex_when_auto_is_missing() {
        let values = [0.40, 0.30, 0.10, 0.08, 0.07, 0.05];
        let state = AdaptiveState {
            learned_shadow_fusion: HashMap::from([(
                "semantic".to_string(),
                human_entry(values, 12),
            )]),
            ..Default::default()
        };
        let missing = A12CalibrationLoad {
            state: A12CalibrationState::default(),
            status: A12CalibrationLoadStatus::Missing,
            error: None,
        };

        let resolved = resolve_recall_fusion_evidence(
            &state,
            &missing,
            10,
            0.02,
            1_700_000_075_000,
            &gate(ArsRecallGateStatus::NoData),
        );
        let evidence = &resolved["recall_fusion:semantic"];

        assert_eq!(evidence.basis, ArsRecallFusionEvidenceBasis::Human);
        assert_eq!(evidence.resolved_simplex, simplex(values));
        assert_eq!(evidence.human_ess, 12);
        assert_eq!(evidence.self_supervised_train_family_ess, 0);
    }

    #[test]
    fn evidence_resolver_requires_ship_ess_current_noise_and_gate_without_drift() {
        let values = [0.35, 0.35, 0.10, 0.08, 0.07, 0.05];
        let base = a12_loaded(
            A12CalibrationScope::Global,
            values,
            20,
            20,
            A12CalibrationVerdict::Ship,
        );
        let state = AdaptiveState::default();

        let mut cases = Vec::new();
        let mut no_data = base.clone();
        no_data.state.scopes.get_mut("global").unwrap().verdict = A12CalibrationVerdict::NoData;
        cases.push((no_data, gate(ArsRecallGateStatus::Ship), 0.02));
        let mut bail = base.clone();
        bail.state.scopes.get_mut("global").unwrap().verdict = A12CalibrationVerdict::Bail;
        cases.push((bail, gate(ArsRecallGateStatus::Ship), 0.02));
        let low_ess = a12_loaded(
            A12CalibrationScope::Global,
            values,
            9,
            20,
            A12CalibrationVerdict::Ship,
        );
        cases.push((low_ess, gate(ArsRecallGateStatus::Ship), 0.02));
        cases.push((base.clone(), gate(ArsRecallGateStatus::NoData), 0.02));
        cases.push((base.clone(), gate(ArsRecallGateStatus::Bail), 0.02));
        cases.push((base.clone(), gate(ArsRecallGateStatus::Ship), 0.03));

        for (calibration, gate, noise_floor) in cases {
            let resolved = resolve_recall_fusion_evidence(
                &state,
                &calibration,
                10,
                noise_floor,
                1_700_000_075_000,
                &gate,
            );
            assert_eq!(
                resolved["recall_fusion:global"].basis,
                ArsRecallFusionEvidenceBasis::Static
            );
        }

        let drifted = AdaptiveState {
            judge_calibration_state: Some(JudgeCalibrationState {
                judge_drift_alert: 1,
                ..Default::default()
            }),
            ..Default::default()
        };
        let resolved = resolve_recall_fusion_evidence(
            &drifted,
            &base,
            10,
            0.02,
            1_700_000_075_000,
            &gate(ArsRecallGateStatus::Ship),
        );
        assert_eq!(
            resolved["recall_fusion:global"].basis,
            ArsRecallFusionEvidenceBasis::Static
        );
    }

    #[test]
    fn evidence_resolver_seals_ship_as_recall_only_self_supervised_evidence() {
        let values = [0.35, 0.35, 0.10, 0.08, 0.07, 0.05];
        let calibration = a12_loaded(
            A12CalibrationScope::QueryType {
                query_type: "semantic".to_string(),
            },
            values,
            21,
            22,
            A12CalibrationVerdict::Ship,
        );

        let resolved = resolve_recall_fusion_evidence(
            &AdaptiveState::default(),
            &calibration,
            10,
            0.02,
            1_700_000_075_000,
            &gate(ArsRecallGateStatus::Ship),
        );
        assert_eq!(resolved.len(), 1);
        assert!(resolved.keys().all(|key| key.starts_with("recall_fusion:")));
        let evidence = &resolved["recall_fusion:semantic"];
        assert_eq!(evidence.basis, ArsRecallFusionEvidenceBasis::SelfSupervised);
        assert_eq!(evidence.resolved_simplex, simplex(values));
        assert_eq!(evidence.a12_generation, Some(11));
        assert_eq!(evidence.a12_revision, Some(4));
        assert_eq!(evidence.self_supervised_train_family_ess, 21);
        assert_eq!(evidence.self_supervised_holdout_family_ess, 22);
        assert_eq!(evidence.recall_gate_status, ArsRecallGateStatus::Ship);
    }

    #[test]
    fn evidence_resolver_expires_at_millisecond_validity_boundary() {
        let mut calibration = a12_loaded(
            A12CalibrationScope::Global,
            [0.35, 0.35, 0.10, 0.08, 0.07, 0.05],
            20,
            20,
            A12CalibrationVerdict::Ship,
        );
        let boundary_ms = 1_700_000_100_000;
        calibration
            .state
            .scopes
            .get_mut("global")
            .unwrap()
            .valid_until_exclusive = Some(boundary_ms);

        let before = resolve_recall_fusion_evidence(
            &AdaptiveState::default(),
            &calibration,
            10,
            0.02,
            boundary_ms - 1,
            &gate(ArsRecallGateStatus::Ship),
        );
        assert_eq!(
            before["recall_fusion:global"].basis,
            ArsRecallFusionEvidenceBasis::SelfSupervised
        );
        assert_eq!(
            before["recall_fusion:global"].a12_valid_until_exclusive,
            Some(boundary_ms)
        );

        let expired = resolve_recall_fusion_evidence(
            &AdaptiveState::default(),
            &calibration,
            10,
            0.02,
            boundary_ms,
            &gate(ArsRecallGateStatus::Ship),
        );
        assert_eq!(
            expired["recall_fusion:global"].basis,
            ArsRecallFusionEvidenceBasis::Static
        );
    }

    #[test]
    fn evidence_resolver_keeps_a12_and_gate_evaluation_times_independent() {
        let calibration = a12_loaded(
            A12CalibrationScope::Global,
            [0.35, 0.35, 0.10, 0.08, 0.07, 0.05],
            20,
            20,
            A12CalibrationVerdict::Ship,
        );
        let mut old_gate = gate(ArsRecallGateStatus::Ship);
        old_gate.evaluated_at = Some(1_699_999_999);

        let resolved = resolve_recall_fusion_evidence(
            &AdaptiveState::default(),
            &calibration,
            10,
            0.02,
            1_700_000_075_000,
            &old_gate,
        );
        let evidence = &resolved["recall_fusion:global"];

        assert_eq!(evidence.basis, ArsRecallFusionEvidenceBasis::SelfSupervised);
        assert_eq!(evidence.calibrated_at, Some(1_700_000_000));
        assert_eq!(evidence.evaluated_at, Some(1_700_000_050));
        assert_eq!(evidence.recall_gate_evaluated_at, Some(1_699_999_999));
    }

    #[test]
    fn evidence_resolver_blends_by_human_and_training_family_ess() {
        let human = [0.60, 0.20, 0.05, 0.05, 0.05, 0.05];
        let automatic = [0.20, 0.60, 0.05, 0.05, 0.05, 0.05];
        let state = AdaptiveState {
            learned_shadow_fusion: HashMap::from([(
                "semantic".to_string(),
                human_entry(human, 10),
            )]),
            ..Default::default()
        };
        let calibration = a12_loaded(
            A12CalibrationScope::QueryType {
                query_type: "semantic".to_string(),
            },
            automatic,
            30,
            20,
            A12CalibrationVerdict::Ship,
        );

        let resolved = resolve_recall_fusion_evidence(
            &state,
            &calibration,
            10,
            0.02,
            1_700_000_075_000,
            &gate(ArsRecallGateStatus::Ship),
        );
        let evidence = &resolved["recall_fusion:semantic"];

        assert_eq!(evidence.basis, ArsRecallFusionEvidenceBasis::Blended);
        assert_eq!(evidence.human_ess, 10);
        assert_eq!(evidence.self_supervised_train_family_ess, 30);
        assert_simplex_close(
            evidence.resolved_simplex,
            simplex([0.30, 0.50, 0.05, 0.05, 0.05, 0.05]),
        );
    }

    #[test]
    fn evidence_resolver_uses_human_query_fallback_for_auto_cluster_scope() {
        let human = [0.50, 0.30, 0.05, 0.05, 0.05, 0.05];
        let automatic = [0.30, 0.50, 0.05, 0.05, 0.05, 0.05];
        let state = AdaptiveState {
            learned_shadow_fusion: HashMap::from([(
                "semantic".to_string(),
                human_entry(human, 20),
            )]),
            ..Default::default()
        };
        let calibration = a12_loaded(
            A12CalibrationScope::Cluster {
                query_type: "semantic".to_string(),
                cluster_id: 7,
            },
            automatic,
            20,
            20,
            A12CalibrationVerdict::Ship,
        );

        let resolved = resolve_recall_fusion_evidence(
            &state,
            &calibration,
            10,
            0.02,
            1_700_000_075_000,
            &gate(ArsRecallGateStatus::Ship),
        );
        let evidence = &resolved["recall_fusion:semantic:7"];

        assert_eq!(evidence.basis, ArsRecallFusionEvidenceBasis::Blended);
        assert_eq!(evidence.human_ess, 20);
        assert_simplex_close(
            evidence.resolved_simplex,
            simplex([0.40, 0.40, 0.05, 0.05, 0.05, 0.05]),
        );
    }

    #[test]
    fn runtime_resolution_preserves_legacy_human_live_lookup() {
        let config = runtime_config();
        let values = [0.40, 0.30, 0.10, 0.08, 0.07, 0.05];
        let state = AdaptiveState {
            learned_shadow_fusion: HashMap::from([(
                "semantic".to_string(),
                human_entry(values, 20),
            )]),
            version: 3,
            ..Default::default()
        };
        let policy = canary_policy(2, 0.4, HashMap::new(), BTreeMap::new());
        let missing = A12CalibrationLoad {
            state: A12CalibrationState::default(),
            status: A12CalibrationLoadStatus::Missing,
            error: None,
        };

        let resolved = resolve_runtime_recall_fusion(
            &policy,
            &config,
            &state,
            &missing,
            "semantic",
            Some(7),
            10,
            0.02,
            1_700_000_075_000,
        );

        assert_eq!(resolved.basis, ArsRecallFusionEvidenceBasis::Human);
        assert_eq!(resolved.adoption_weight, 0.4);
        assert_eq!(
            resolved.simplex,
            Some(expected_runtime_simplex(simplex(values), 20, 0.4, &config))
        );
    }

    #[test]
    fn runtime_resolution_honors_adaptive_and_acceleration_gates() {
        let state = AdaptiveState {
            learned_shadow_fusion: HashMap::from([(
                "semantic".to_string(),
                human_entry([0.40, 0.30, 0.10, 0.08, 0.07, 0.05], 20),
            )]),
            version: 3,
            ..Default::default()
        };
        let policy = canary_policy(3, 0.4, HashMap::new(), BTreeMap::new());
        let missing = A12CalibrationLoad {
            state: A12CalibrationState::default(),
            status: A12CalibrationLoadStatus::Missing,
            error: None,
        };
        let mut adaptive_disabled = runtime_config();
        adaptive_disabled.adaptive.enabled = false;
        let mut acceleration_disabled = runtime_config();
        acceleration_disabled.ars.acceleration.enabled = false;
        let mut shadow_only = runtime_config();
        shadow_only.ars.acceleration.shadow_only = true;

        for config in [adaptive_disabled, acceleration_disabled, shadow_only] {
            let resolved = resolve_runtime_recall_fusion(
                &policy,
                &config,
                &state,
                &missing,
                "semantic",
                None,
                10,
                0.02,
                1_700_000_075_000,
            );
            assert_eq!(resolved.adoption_weight, 0.0);
            assert_eq!(resolved.simplex, None);
        }
    }

    #[test]
    fn runtime_resolution_uses_sealed_auto_ship_scope() {
        let config = runtime_config();
        let values = [0.35, 0.35, 0.10, 0.08, 0.07, 0.05];
        let calibration = a12_loaded(
            A12CalibrationScope::Cluster {
                query_type: "semantic".to_string(),
                cluster_id: 7,
            },
            values,
            20,
            20,
            A12CalibrationVerdict::Ship,
        );
        let state = AdaptiveState {
            version: 5,
            ..Default::default()
        };
        let evidence = resolve_recall_fusion_evidence(
            &state,
            &calibration,
            10,
            0.02,
            1_700_000_075_000,
            &gate(ArsRecallGateStatus::Ship),
        );
        let policy = canary_policy(
            5,
            0.0,
            HashMap::from([("recall_fusion:semantic:7".to_string(), 0.25)]),
            evidence,
        );

        let resolved = resolve_runtime_recall_fusion(
            &policy,
            &config,
            &state,
            &calibration,
            "semantic",
            Some(7),
            10,
            0.02,
            1_700_000_075_000,
        );

        assert_eq!(
            resolved.scope_key.as_deref(),
            Some("recall_fusion:semantic:7")
        );
        assert_eq!(resolved.basis, ArsRecallFusionEvidenceBasis::SelfSupervised);
        assert_eq!(resolved.adoption_weight, 0.25);
        assert_eq!(
            resolved.simplex,
            Some(expected_runtime_simplex(simplex(values), 20, 0.25, &config))
        );
    }

    #[test]
    fn live_runtime_resolution_blocks_ship_after_source_snapshot_drift() {
        let store = crate::store::SqliteStore::in_memory().unwrap();
        let config = runtime_config();
        let values = [0.35, 0.35, 0.10, 0.08, 0.07, 0.05];
        let mut calibration = a12_loaded(
            A12CalibrationScope::Global,
            values,
            20,
            20,
            A12CalibrationVerdict::Ship,
        );
        seal_current_live_inputs(&store, &config, &mut calibration);
        let state = AdaptiveState {
            version: 5,
            ..Default::default()
        };
        let evidence = resolve_recall_fusion_evidence(
            &state,
            &calibration,
            10,
            0.02,
            1_700_000_075_000,
            &gate(ArsRecallGateStatus::Ship),
        );
        let policy = canary_policy(
            state.version,
            0.0,
            HashMap::from([("recall_fusion:global".to_string(), 0.25)]),
            evidence,
        );

        crate::ops::a12_autocalibration::reset_a12_full_snapshot_fingerprint_call_count();

        let current = resolve_runtime_recall_fusion_live(
            &store,
            &policy,
            &config,
            &state,
            &calibration,
            "semantic",
            None,
            10,
            0.02,
            1_700_000_075_000,
        );
        assert_eq!(current.adoption_weight, 0.25, "{}", current.reason);
        assert_eq!(
            crate::ops::a12_autocalibration::a12_full_snapshot_fingerprint_call_count(),
            0,
            "online A12 resolution must never hash the full recall snapshot"
        );

        store
            .conn()
            .execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES ('rerank_weights', ?1)",
                rusqlite::params!["{\"semantic\":0.5}"],
            )
            .unwrap();

        let drifted = resolve_runtime_recall_fusion_live(
            &store,
            &policy,
            &config,
            &state,
            &calibration,
            "semantic",
            None,
            10,
            0.02,
            1_700_000_075_000,
        );
        assert_eq!(drifted.adoption_weight, 0.0);
        assert_eq!(drifted.simplex, None);
        assert_eq!(drifted.code, RecallFusionScopeHealthCode::Stale);
        assert!(drifted.reason.contains("input epoch"));

        let policy_load = crate::store::ars_parameter_policy::ArsParameterPolicyLoad {
            policy,
            status: crate::store::ars_parameter_policy::ArsParameterPolicyLoadStatus::Loaded,
            error: None,
        };
        let run = recall_fusion_calibration_run_attestation(&calibration);
        let report = recall_fusion_activation_report_live(
            &store,
            &config,
            &state,
            &policy_load,
            &calibration,
            &gate(ArsRecallGateStatus::Ship),
            run.as_ref(),
            1_700_000_075_000,
        );
        let global = report
            .scopes
            .iter()
            .find(|scope| scope.scope == "global")
            .unwrap();
        assert!(!global.active);
        assert_eq!(global.health_code, RecallFusionScopeHealthCode::Stale);
        assert!(global.reason.contains("input epoch"));
        assert_eq!(
            crate::ops::a12_autocalibration::a12_full_snapshot_fingerprint_call_count(),
            0,
            "store-aware activation reports must use the O(1) live resolver"
        );
    }

    #[test]
    fn live_runtime_resolution_blocks_ship_after_behavior_config_drift() {
        let store = crate::store::SqliteStore::in_memory().unwrap();
        let config = runtime_config();
        let mut calibration = a12_loaded(
            A12CalibrationScope::Global,
            [0.35, 0.35, 0.10, 0.08, 0.07, 0.05],
            20,
            20,
            A12CalibrationVerdict::Ship,
        );
        seal_current_live_inputs(&store, &config, &mut calibration);
        let state = AdaptiveState {
            version: 5,
            ..Default::default()
        };
        let evidence = resolve_recall_fusion_evidence(
            &state,
            &calibration,
            10,
            0.02,
            1_700_000_075_000,
            &gate(ArsRecallGateStatus::Ship),
        );
        let policy = canary_policy(
            state.version,
            0.0,
            HashMap::from([("recall_fusion:global".to_string(), 0.25)]),
            evidence,
        );
        let mut drifted_config = config.clone();
        drifted_config.search.strong_signal_ratio += 0.01;

        let drifted = resolve_runtime_recall_fusion_live(
            &store,
            &policy,
            &drifted_config,
            &state,
            &calibration,
            "semantic",
            None,
            10,
            0.02,
            1_700_000_075_000,
        );

        assert_eq!(drifted.adoption_weight, 0.0);
        assert_eq!(drifted.simplex, None);
        assert_eq!(drifted.code, RecallFusionScopeHealthCode::Stale);
        assert!(drifted.reason.contains("behavior config"));
    }

    #[test]
    fn runtime_resolution_falls_back_from_missing_cluster_to_query_scope() {
        let config = runtime_config();
        let calibration = a12_loaded(
            A12CalibrationScope::QueryType {
                query_type: "semantic".to_string(),
            },
            [0.35, 0.35, 0.10, 0.08, 0.07, 0.05],
            20,
            20,
            A12CalibrationVerdict::Ship,
        );
        let state = AdaptiveState {
            version: 2,
            ..Default::default()
        };
        let evidence = resolve_recall_fusion_evidence(
            &state,
            &calibration,
            10,
            0.02,
            1_700_000_075_000,
            &gate(ArsRecallGateStatus::Ship),
        );
        let policy = canary_policy(
            2,
            0.0,
            HashMap::from([("recall_fusion:semantic".to_string(), 0.3)]),
            evidence,
        );

        let resolved = resolve_runtime_recall_fusion(
            &policy,
            &config,
            &state,
            &calibration,
            "semantic",
            Some(99),
            10,
            0.02,
            1_700_000_075_000,
        );

        assert_eq!(
            resolved.scope_key.as_deref(),
            Some("recall_fusion:semantic")
        );
        assert_eq!(resolved.adoption_weight, 0.3);
    }

    #[test]
    fn runtime_specific_a12_candidate_is_authoritative_human_fallback() {
        let config = runtime_config();
        // Deliberately non-normalized: the sealed boundary must compare the
        // normalized attestation while feeding this raw vector through the
        // exact legacy runtime normalization boundary.
        let human = [4.0, 3.0, 1.0, 0.8, 0.7, 0.5];
        let state = AdaptiveState {
            learned_shadow_fusion: HashMap::from([(
                "semantic".to_string(),
                human_entry(human, 20),
            )]),
            version: 6,
            ..Default::default()
        };
        let calibration = a12_loaded(
            A12CalibrationScope::Cluster {
                query_type: "semantic".to_string(),
                cluster_id: 7,
            },
            [0.35, 0.35, 0.10, 0.08, 0.07, 0.05],
            20,
            20,
            A12CalibrationVerdict::Bail,
        );
        let mut evidence = resolve_recall_fusion_evidence(
            &state,
            &calibration,
            10,
            0.02,
            1_700_000_075_000,
            &gate(ArsRecallGateStatus::Ship),
        );
        assert_eq!(
            evidence["recall_fusion:semantic:7"].basis,
            ArsRecallFusionEvidenceBasis::Human
        );
        evidence
            .get_mut("recall_fusion:semantic:7")
            .unwrap()
            .human_runtime_adoption_weight = Some(0.3);
        let policy = canary_policy(
            6,
            0.8,
            HashMap::from([("recall_fusion:semantic".to_string(), 0.3)]),
            evidence,
        );

        let resolved = resolve_runtime_recall_fusion(
            &policy,
            &config,
            &state,
            &calibration,
            "semantic",
            Some(7),
            10,
            0.02,
            1_700_000_075_000,
        );

        assert_eq!(
            resolved.scope_key.as_deref(),
            Some("recall_fusion:semantic:7")
        );
        assert_eq!(resolved.basis, ArsRecallFusionEvidenceBasis::Human);
        assert_eq!(resolved.adoption_weight, 0.3);
        assert_eq!(
            resolved.simplex,
            Some(expected_runtime_simplex(simplex(human), 20, 0.3, &config,))
        );
    }

    #[test]
    fn runtime_specific_candidate_rollback_matrix_stays_on_sealed_human_scope() {
        let config = runtime_config();
        let human = [0.40, 0.30, 0.10, 0.08, 0.07, 0.05];
        let state = AdaptiveState {
            learned_shadow_fusion: HashMap::from([(
                "semantic".to_string(),
                human_entry(human, 20),
            )]),
            version: 6,
            ..Default::default()
        };
        let base = a12_loaded(
            A12CalibrationScope::Cluster {
                query_type: "semantic".to_string(),
                cluster_id: 7,
            },
            [0.35, 0.35, 0.10, 0.08, 0.07, 0.05],
            20,
            20,
            A12CalibrationVerdict::Ship,
        );
        let now_millis = 1_700_000_075_000;
        let mut cases = Vec::new();

        let mut bail = base.clone();
        bail.state.scopes.get_mut("semantic:7").unwrap().verdict = A12CalibrationVerdict::Bail;
        cases.push(("bail", bail, gate(ArsRecallGateStatus::Ship), 0.02));

        let mut no_data = base.clone();
        no_data.state.scopes.get_mut("semantic:7").unwrap().verdict = A12CalibrationVerdict::NoData;
        cases.push(("no-data", no_data, gate(ArsRecallGateStatus::Ship), 0.02));

        let mut stale = base.clone();
        stale
            .state
            .scopes
            .get_mut("semantic:7")
            .unwrap()
            .generation_fingerprint = "stale-generation".to_string();
        cases.push(("stale", stale, gate(ArsRecallGateStatus::Ship), 0.02));

        let mut expired = base.clone();
        expired
            .state
            .scopes
            .get_mut("semantic:7")
            .unwrap()
            .valid_until_exclusive = Some(now_millis);
        cases.push(("expired", expired, gate(ArsRecallGateStatus::Ship), 0.02));

        cases.push((
            "recall-gate-no-data",
            base,
            gate(ArsRecallGateStatus::NoData),
            0.02,
        ));

        for (name, calibration, recall_gate, expected_noise_floor) in cases {
            let mut evidence = resolve_recall_fusion_evidence(
                &state,
                &calibration,
                10,
                expected_noise_floor,
                now_millis,
                &recall_gate,
            );
            let specific = evidence.get_mut("recall_fusion:semantic:7").unwrap();
            assert_eq!(
                specific.basis,
                ArsRecallFusionEvidenceBasis::Human,
                "{name}"
            );
            assert!(specific.automatic_candidate_present, "{name}");
            specific.human_runtime_adoption_weight = Some(0.3);
            let policy = canary_policy(
                state.version,
                0.8,
                HashMap::from([("recall_fusion:semantic".to_string(), 0.9)]),
                evidence,
            );

            let resolved = resolve_runtime_recall_fusion(
                &policy,
                &config,
                &state,
                &calibration,
                "semantic",
                Some(7),
                10,
                expected_noise_floor,
                now_millis,
            );

            assert_eq!(
                resolved.scope_key.as_deref(),
                Some("recall_fusion:semantic:7"),
                "{name}: {}",
                resolved.reason
            );
            assert_eq!(
                resolved.basis,
                ArsRecallFusionEvidenceBasis::Human,
                "{name}"
            );
            assert_eq!(resolved.adoption_weight, 0.3, "{name}");
            assert_eq!(
                resolved.simplex,
                Some(expected_runtime_simplex(simplex(human), 20, 0.3, &config)),
                "{name}"
            );
        }
    }

    #[test]
    fn runtime_blended_expiry_uses_sealed_human_fallback() {
        let config = runtime_config();
        let human = [0.60, 0.20, 0.05, 0.05, 0.05, 0.05];
        let state = AdaptiveState {
            learned_shadow_fusion: HashMap::from([(
                "semantic".to_string(),
                human_entry(human, 20),
            )]),
            version: 8,
            ..Default::default()
        };
        let mut calibration = a12_loaded(
            A12CalibrationScope::Cluster {
                query_type: "semantic".to_string(),
                cluster_id: 7,
            },
            [0.20, 0.60, 0.05, 0.05, 0.05, 0.05],
            20,
            20,
            A12CalibrationVerdict::Ship,
        );
        let boundary = 1_700_000_100_000;
        calibration
            .state
            .scopes
            .get_mut("semantic:7")
            .unwrap()
            .valid_until_exclusive = Some(boundary);
        let mut evidence = resolve_recall_fusion_evidence(
            &state,
            &calibration,
            10,
            0.02,
            boundary - 1,
            &gate(ArsRecallGateStatus::Ship),
        );
        let specific = evidence.get_mut("recall_fusion:semantic:7").unwrap();
        assert_eq!(specific.basis, ArsRecallFusionEvidenceBasis::Blended);
        specific.human_runtime_adoption_weight = Some(0.3);
        let policy = canary_policy(
            state.version,
            0.0,
            HashMap::from([("recall_fusion:semantic:7".to_string(), 0.5)]),
            evidence,
        );

        let resolved = resolve_runtime_recall_fusion(
            &policy,
            &config,
            &state,
            &calibration,
            "semantic",
            Some(7),
            10,
            0.02,
            boundary,
        );

        assert_eq!(
            resolved.scope_key.as_deref(),
            Some("recall_fusion:semantic:7")
        );
        assert_eq!(resolved.basis, ArsRecallFusionEvidenceBasis::Human);
        assert_eq!(resolved.adoption_weight, 0.3);
        assert_eq!(
            resolved.simplex,
            Some(expected_runtime_simplex(simplex(human), 20, 0.3, &config))
        );
    }

    #[test]
    fn runtime_resolution_zeroes_auto_on_pointer_source_gate_build_or_expiry_mismatch() {
        let config = runtime_config();
        let mut calibration = a12_loaded(
            A12CalibrationScope::Global,
            [0.35, 0.35, 0.10, 0.08, 0.07, 0.05],
            20,
            20,
            A12CalibrationVerdict::Ship,
        );
        calibration
            .state
            .scopes
            .get_mut("global")
            .unwrap()
            .valid_until_exclusive = Some(1_700_000_200_000);
        let state = AdaptiveState {
            version: 9,
            ..Default::default()
        };
        let evidence = resolve_recall_fusion_evidence(
            &state,
            &calibration,
            10,
            0.02,
            1_700_000_100_000,
            &gate(ArsRecallGateStatus::Ship),
        );
        let base_policy = canary_policy(
            9,
            0.0,
            HashMap::from([("recall_fusion:global".to_string(), 0.5)]),
            evidence,
        );

        let mut pointer_mismatch = calibration.clone();
        pointer_mismatch.state.revision += 1;
        let mut source_mismatch = base_policy.clone();
        source_mismatch.source_adaptive_version = 8;
        let mut gate_mismatch = base_policy.clone();
        gate_mismatch
            .recall_fusion_evidence
            .get_mut("recall_fusion:global")
            .unwrap()
            .recall_gate_status = ArsRecallGateStatus::Bail;
        let mut build_mismatch = base_policy.clone();
        build_mismatch
            .recall_fusion_evidence
            .get_mut("recall_fusion:global")
            .unwrap()
            .recall_gate_build_fingerprint = Some("different-build".to_string());

        let cases = [
            (&base_policy, &pointer_mismatch, 1_700_000_100_000),
            (&source_mismatch, &calibration, 1_700_000_100_000),
            (&gate_mismatch, &calibration, 1_700_000_100_000),
            (&build_mismatch, &calibration, 1_700_000_100_000),
            (&base_policy, &calibration, 1_700_000_200_000),
        ];
        for (policy, active, now_millis) in cases {
            let resolved = resolve_runtime_recall_fusion(
                policy, &config, &state, active, "semantic", None, 10, 0.02, now_millis,
            );
            assert_eq!(resolved.adoption_weight, 0.0, "{}", resolved.reason);
            assert_eq!(resolved.simplex, None);
        }
    }

    #[test]
    fn runtime_resolution_recomputes_blended_simplex_from_sealed_ess() {
        let config = runtime_config();
        let human = [0.60, 0.20, 0.05, 0.05, 0.05, 0.05];
        let automatic = [0.20, 0.60, 0.05, 0.05, 0.05, 0.05];
        let state = AdaptiveState {
            learned_shadow_fusion: HashMap::from([(
                "semantic".to_string(),
                human_entry(human, 10),
            )]),
            version: 4,
            ..Default::default()
        };
        let calibration = a12_loaded(
            A12CalibrationScope::QueryType {
                query_type: "semantic".to_string(),
            },
            automatic,
            30,
            20,
            A12CalibrationVerdict::Ship,
        );
        let evidence = resolve_recall_fusion_evidence(
            &state,
            &calibration,
            10,
            0.02,
            1_700_000_100_000,
            &gate(ArsRecallGateStatus::Ship),
        );
        let valid_policy = canary_policy(
            4,
            0.0,
            HashMap::from([("recall_fusion:semantic".to_string(), 0.2)]),
            evidence.clone(),
        );
        let valid = resolve_runtime_recall_fusion(
            &valid_policy,
            &config,
            &state,
            &calibration,
            "semantic",
            None,
            10,
            0.02,
            1_700_000_100_000,
        );
        assert_eq!(
            valid.simplex,
            Some(expected_runtime_simplex(
                simplex([0.30, 0.50, 0.05, 0.05, 0.05, 0.05]),
                40,
                0.2,
                &config,
            ))
        );

        let mut tampered = evidence;
        tampered
            .get_mut("recall_fusion:semantic")
            .unwrap()
            .resolved_simplex = simplex([0.25, 0.55, 0.05, 0.05, 0.05, 0.05]);
        let policy = canary_policy(
            4,
            0.0,
            HashMap::from([("recall_fusion:semantic".to_string(), 0.2)]),
            tampered,
        );

        let resolved = resolve_runtime_recall_fusion(
            &policy,
            &config,
            &state,
            &calibration,
            "semantic",
            None,
            10,
            0.02,
            1_700_000_100_000,
        );

        assert_eq!(resolved.adoption_weight, 0.0);
        assert_eq!(resolved.simplex, None);
        assert!(resolved.reason.contains("blended"));
    }
}
