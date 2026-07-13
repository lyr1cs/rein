//! Shared, side-effect-free activation logic for A12 recall fusion.

use crate::eval::gates::{self, ScorecardLoad, ScorecardStatus};
use crate::store::a12_calibration::{
    A12CalibrationLoad, A12CalibrationLoadStatus, A12CalibrationVerdict, A12FusionSimplex,
    A12ScopeEntry,
};
use crate::store::adaptive::{AdaptiveState, LearnedShadowFusionEntry};
use crate::store::ars_parameter_policy::{
    ArsParameterPolicy, ArsParameterPolicyMode, ArsRecallFusionEvidence,
    ArsRecallFusionEvidenceBasis, ArsRecallGateStatus, ARS_PARAMETER_POLICY_SCHEMA_VERSION,
};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

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

/// Result consumed by online recall. A non-zero adoption always carries a
/// usable simplex; any attestation mismatch returns `None` plus zero adoption.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RuntimeRecallFusionResolution {
    pub scope_key: Option<String>,
    pub basis: ArsRecallFusionEvidenceBasis,
    pub simplex: Option<A12FusionSimplex>,
    pub adoption_weight: f64,
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
            "adaptive recall-fusion activation is disabled or shadow-only",
        );
    }
    if policy.schema_version != ARS_PARAMETER_POLICY_SCHEMA_VERSION
        || policy.mode != ArsParameterPolicyMode::Canary
    {
        return runtime_disabled(
            None,
            ArsRecallFusionEvidenceBasis::Static,
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
                return runtime_disabled(Some(key), evidence.basis, blocker.into_reason());
            }
            let adoption_weight = policy.runtime_adoption_weight_for(adaptive.version, &key);
            if adoption_weight <= f64::EPSILON {
                return runtime_disabled(
                    Some(key),
                    evidence.basis,
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
            "sealed human fallback source adaptive version is not exact",
        );
    }
    if let Some(reason) = automatic_candidate_identity_mismatch(active_a12, policy_key, evidence) {
        return runtime_disabled(selected, ArsRecallFusionEvidenceBasis::Human, reason);
    }
    let Some(sealed_human) = evidence.human_simplex else {
        return runtime_disabled(
            selected,
            ArsRecallFusionEvidenceBasis::Human,
            "automatic boundary is missing its sealed human simplex",
        );
    };
    if let Some(reason) =
        sealed_human_fallback_relation_mismatch(active_a12, policy_key, evidence, sealed_human)
    {
        return runtime_disabled(selected, ArsRecallFusionEvidenceBasis::Human, reason);
    }
    let Some(adoption_weight) = evidence.human_runtime_adoption_weight else {
        return runtime_disabled(
            selected,
            ArsRecallFusionEvidenceBasis::Human,
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
            "automatic boundary has zero or invalid sealed human adoption",
        );
    }
    let Some(scope) = policy_key.strip_prefix("recall_fusion:") else {
        return runtime_disabled(
            selected,
            ArsRecallFusionEvidenceBasis::Human,
            "sealed human policy key is outside recall_fusion",
        );
    };
    let floor = u64::try_from(min_samples_alpha.max(10)).unwrap_or(u64::MAX);
    let Some(human) = human_for_scope(adaptive, scope, floor) else {
        return runtime_disabled(
            selected,
            ArsRecallFusionEvidenceBasis::Human,
            "automatic boundary has no matching live human fallback",
        );
    };
    let human_ess = u64::try_from(human.sample_count).unwrap_or(u64::MAX);
    let Some(live_human) = human_simplex(human) else {
        return runtime_disabled(
            selected,
            ArsRecallFusionEvidenceBasis::Human,
            "automatic boundary has an invalid live human fallback",
        );
    };
    if evidence.human_ess != human_ess || !simplex_matches(sealed_human, live_human) {
        return runtime_disabled(
            selected,
            ArsRecallFusionEvidenceBasis::Human,
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
) -> Option<String> {
    if active_a12.status != A12CalibrationLoadStatus::Loaded {
        return Some("active A12 calibration is unavailable for human fallback".to_string());
    }
    let Some(scope) = policy_key.strip_prefix("recall_fusion:") else {
        return Some("human fallback policy key is outside recall_fusion".to_string());
    };
    let Some(entry) = active_a12.state.scopes.get(scope) else {
        return Some("active A12 calibration has no matching fallback boundary".to_string());
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
        return Some(
            "sealed A12 candidate identity mismatched human fallback boundary".to_string(),
        );
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
            "human recall-fusion policy has zero runtime adoption",
        );
    }
    let Some(entry) = adaptive.get_shadow_fusion_weights(query_type, cluster_id, min_samples_alpha)
    else {
        return runtime_disabled(
            selected,
            ArsRecallFusionEvidenceBasis::Human,
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
    Eligibility(String),
    FailClosed(String),
}

impl AutomaticRuntimeBlocker {
    fn eligibility(reason: impl Into<String>) -> Self {
        Self::Eligibility(reason.into())
    }

    fn fail_closed(reason: impl Into<String>) -> Self {
        Self::FailClosed(reason.into())
    }

    fn allows_human_fallback(&self) -> bool {
        matches!(self, Self::Eligibility(_))
    }

    fn into_reason(self) -> String {
        match self {
            Self::Eligibility(reason) | Self::FailClosed(reason) => reason,
        }
    }
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
            "automatic policy source adaptive version is not exact",
        ));
    }
    if !policy.adoption_weights.contains_key(policy_key) {
        return Some(AutomaticRuntimeBlocker::fail_closed(
            "automatic policy is missing its explicit scoped adoption weight",
        ));
    }
    if active_a12.status != A12CalibrationLoadStatus::Loaded {
        return Some(AutomaticRuntimeBlocker::fail_closed(
            "active A12 calibration is unavailable",
        ));
    }
    let Some(scope) = policy_key.strip_prefix("recall_fusion:") else {
        return Some(AutomaticRuntimeBlocker::fail_closed(
            "automatic policy key is outside recall_fusion",
        ));
    };
    let Some(entry) = active_a12.state.scopes.get(scope) else {
        return Some(AutomaticRuntimeBlocker::fail_closed(
            "active A12 calibration has no matching scope",
        ));
    };
    let floor = u64::try_from(min_samples_alpha.max(10)).unwrap_or(u64::MAX);
    if entry.verdict != A12CalibrationVerdict::Ship
        || evidence.a12_verdict != Some(A12CalibrationVerdict::Ship)
    {
        return Some(AutomaticRuntimeBlocker::fail_closed(
            "automatic A12 holdout attestation is not Ship",
        ));
    }
    if evidence.self_supervised_train_family_ess != entry.train_family_ess
        || evidence.self_supervised_holdout_family_ess != entry.holdout_family_ess
    {
        return Some(AutomaticRuntimeBlocker::fail_closed(
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
            "sealed A12 pointer or fingerprint identity mismatched",
        ));
    }
    if evidence
        .a12_noise_floor
        .is_none_or(|noise_floor| !same_f64(noise_floor, entry.noise_floor))
    {
        return Some(AutomaticRuntimeBlocker::fail_closed(
            "sealed A12 noise floor mismatched active evidence",
        ));
    }
    if evidence.recall_gate_status != ArsRecallGateStatus::Ship
        || evidence.recall_gate_build_fingerprint.as_deref() != Some(env!("REIN_BUILD_FINGERPRINT"))
        || evidence
            .recall_gate_fixture_fingerprint
            .as_deref()
            .is_none_or(str::is_empty)
        || evidence
            .recall_gate_evaluated_at
            .is_none_or(|value| value <= 0)
    {
        return Some(AutomaticRuntimeBlocker::fail_closed(
            "sealed recall eval gate is not current Ship evidence",
        ));
    }
    if !simplex_is_valid(evidence.resolved_simplex) {
        return Some(AutomaticRuntimeBlocker::fail_closed(
            "sealed recall-fusion simplex is invalid",
        ));
    }
    if evidence.basis == ArsRecallFusionEvidenceBasis::SelfSupervised
        && evidence.resolved_simplex != entry.simplex
    {
        return Some(AutomaticRuntimeBlocker::fail_closed(
            "self-supervised simplex does not match active A12 scope",
        ));
    }
    if evidence.basis == ArsRecallFusionEvidenceBasis::Blended {
        let Some(human) = human_for_scope(adaptive, scope, floor) else {
            return Some(AutomaticRuntimeBlocker::fail_closed(
                "blended evidence has no matching live human scope",
            ));
        };
        let human_ess = u64::try_from(human.sample_count).unwrap_or(u64::MAX);
        let Some(human_simplex) = human_simplex(human) else {
            return Some(AutomaticRuntimeBlocker::fail_closed(
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
            "judge drift blocks automatic recall fusion",
        ));
    }
    if entry.train_family_ess < floor || entry.holdout_family_ess < floor {
        return Some(AutomaticRuntimeBlocker::eligibility(
            "automatic A12 ESS is below the current eligibility floor",
        ));
    }
    if !entry.matches_noise_floor(expected_noise_floor) {
        return Some(AutomaticRuntimeBlocker::eligibility(
            "active A12 noise floor is stale for the current runtime",
        ));
    }
    if !entry.is_current_for_at(&active_a12.state, expected_noise_floor, now_millis) {
        return Some(AutomaticRuntimeBlocker::eligibility(
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
    reason: impl Into<String>,
) -> RuntimeRecallFusionResolution {
    RuntimeRecallFusionResolution {
        scope_key,
        basis,
        simplex: None,
        adoption_weight: 0.0,
        reason: reason.into(),
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
                schema_version: 1,
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
                    source_snapshot_fingerprint: "source-snapshot-fingerprint".to_string(),
                    behavior_config_fingerprint: "behavior-config-fingerprint".to_string(),
                }),
            },
            status: A12CalibrationLoadStatus::Loaded,
            error: None,
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
