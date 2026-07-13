//! ARS parameter-policy activation storage.

use rusqlite::{params, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io;

pub const ARS_PARAMETER_POLICY_METADATA_KEY: &str = "ars_parameter_policy";
pub const ARS_PARAMETER_POLICY_SCHEMA_VERSION: u32 = 3;
const LEGACY_ARS_PARAMETER_POLICY_SCHEMA_VERSION: u32 = 1;
const LEGACY_A12_ARS_PARAMETER_POLICY_SCHEMA_VERSION: u32 = 2;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ArsParameterPolicyMode {
    #[default]
    Disabled,
    Shadow,
    Canary,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ArsRecallFusionEvidenceBasis {
    #[default]
    Static,
    Human,
    SelfSupervised,
    Blended,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ArsRecallGateStatus {
    Ship,
    Bail,
    #[default]
    NoData,
}

/// Sealed evidence behind one `recall_fusion:*` policy scope.
///
/// The policy owns the resolved simplex so an automatic-only scope does not
/// need to masquerade as human `AdaptiveState` feedback. Fingerprints bind the
/// entry to the immutable A12 revision and current recall-gate attestation;
/// Task 5's shared resolver revalidates them before runtime use.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ArsRecallFusionEvidence {
    #[serde(default)]
    pub basis: ArsRecallFusionEvidenceBasis,
    #[serde(default)]
    pub resolved_simplex: crate::store::a12_calibration::A12FusionSimplex,
    #[serde(default)]
    pub human_ess: u64,
    /// True when this scope had an A12 candidate at refresh time, including a
    /// candidate that was Bail, stale, expired, or blocked by the recall gate.
    /// The bit makes a more-specific human fallback authoritative at runtime.
    #[serde(default)]
    pub automatic_candidate_present: bool,
    /// Pure human simplex sealed independently of any A12 blend. This lets a
    /// blended or ineligible automatic scope fall back without consulting a
    /// broader automatic scope.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub human_simplex: Option<crate::store::a12_calibration::A12FusionSimplex>,
    /// Pure legacy-human adoption sealed before automatic overlays are
    /// applied. It is intentionally scoped evidence rather than a synthesized
    /// entry in `adoption_weights`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub human_runtime_adoption_weight: Option<f64>,
    #[serde(default)]
    pub self_supervised_train_family_ess: u64,
    #[serde(default)]
    pub self_supervised_holdout_family_ess: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub a12_generation: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub a12_revision: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation_fingerprint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub corpus_fingerprint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub optimizer_fingerprint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluation_fingerprint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub a12_verdict: Option<crate::store::a12_calibration::A12CalibrationVerdict>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub a12_noise_floor: Option<f64>,
    #[serde(default)]
    pub recall_gate_status: ArsRecallGateStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recall_gate_build_fingerprint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recall_gate_fixture_fingerprint: Option<String>,
    /// Creation time of the current-build recall gate scorecard (Unix
    /// seconds). Kept separate from A12's own evaluation time because the
    /// build gate and local calibration have no causal ordering.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recall_gate_evaluated_at: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibrated_at: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluated_at: Option<i64>,
    /// Earliest Unix millisecond at which fixed-time A12 evidence expires.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub a12_valid_until_exclusive: Option<i64>,
    #[serde(default)]
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ArsParameterPolicy {
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    #[serde(default)]
    pub revision: u64,
    #[serde(default)]
    pub mode: ArsParameterPolicyMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disabled_reason: Option<String>,
    #[serde(default)]
    pub source_adaptive_version: u64,
    /// Runtime adoption cap in `[0, 1]`. This is the rollout-side weight
    /// multiplied into dynamic trust, so canary activation slides gradually
    /// from static priors toward learned values instead of acting as a binary
    /// switch.
    #[serde(default)]
    pub runtime_adoption_weight: f64,
    /// Optional scoped rollout weights. Keys are stable policy surfaces such as
    /// `recall_fusion:semantic`, `recall_fusion:semantic:7`, or
    /// `judge_sample_rate`. Missing scopes fall back to
    /// `runtime_adoption_weight`.
    #[serde(default)]
    pub adoption_weights: HashMap<String, f64>,
    /// Evidence records exist only for recall-fusion scopes. Scalar policy
    /// surfaces deliberately have no equivalent automatic-evidence map.
    #[serde(default)]
    pub recall_fusion_evidence: HashMap<String, ArsRecallFusionEvidence>,
    #[serde(default)]
    pub last_event_id: i64,
    #[serde(default)]
    pub last_updated: String,
}

impl Default for ArsParameterPolicy {
    fn default() -> Self {
        Self {
            schema_version: ARS_PARAMETER_POLICY_SCHEMA_VERSION,
            revision: 0,
            mode: ArsParameterPolicyMode::Disabled,
            disabled_reason: Some("missing policy row".to_string()),
            source_adaptive_version: 0,
            runtime_adoption_weight: 0.0,
            adoption_weights: HashMap::new(),
            recall_fusion_evidence: HashMap::new(),
            last_event_id: 0,
            last_updated: String::new(),
        }
    }
}

impl ArsParameterPolicy {
    pub fn disabled(reason: impl Into<String>) -> Self {
        Self {
            disabled_reason: Some(reason.into()),
            ..Self::default()
        }
    }

    pub fn allows_runtime_adoption(&self, adaptive_version: u64) -> bool {
        self.runtime_adoption_weight(adaptive_version) > f64::EPSILON
    }

    pub fn runtime_adoption_weight(&self, adaptive_version: u64) -> f64 {
        if !self.can_adopt_runtime(adaptive_version) {
            return 0.0;
        }
        clamp01(self.runtime_adoption_weight)
    }

    pub fn runtime_adoption_weight_for(&self, adaptive_version: u64, key: &str) -> f64 {
        if !self.can_adopt_runtime(adaptive_version) {
            return 0.0;
        }
        self.adoption_weights
            .get(key)
            .copied()
            .map(clamp01)
            .unwrap_or_else(|| self.runtime_adoption_weight(adaptive_version))
    }

    fn can_adopt_runtime(&self, adaptive_version: u64) -> bool {
        self.schema_version == ARS_PARAMETER_POLICY_SCHEMA_VERSION
            && matches!(self.mode, ArsParameterPolicyMode::Canary)
            && self.source_adaptive_version <= adaptive_version
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ArsParameterPolicyLoadStatus {
    Missing,
    Loaded,
    /// The row exists but its bytes failed JSON parse — a genuinely
    /// malformed value, no recoverable interpretation. Safe for
    /// `doctor --fix` to delete (R5 P2 audit catch 2026-05-04).
    Corrupt,
    /// The row exists and parses as JSON but its `schema_version`
    /// field disagrees with this binary's `ARS_PARAMETER_POLICY_SCHEMA_VERSION`.
    /// In a downgrade scenario this is the older binary reading a
    /// row written by a newer one — the data is valid, the older
    /// binary just can't interpret it. `doctor --fix` MUST NOT
    /// delete this; failing closed (`Disabled` semantics until the
    /// newer binary is restored or an operator hand-edits the row)
    /// is the correct behavior.
    UnsupportedSchema,
    StorageError,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ArsParameterPolicyLoad {
    pub policy: ArsParameterPolicy,
    pub status: ArsParameterPolicyLoadStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

pub fn load_parameter_policy(conn: &rusqlite::Connection) -> ArsParameterPolicyLoad {
    let raw = match conn
        .query_row(
            "SELECT value FROM metadata WHERE key = ?1",
            params![ARS_PARAMETER_POLICY_METADATA_KEY],
            |row| row.get::<_, String>(0),
        )
        .optional()
    {
        Ok(raw) => raw,
        Err(e) => {
            return ArsParameterPolicyLoad {
                policy: ArsParameterPolicy::disabled("policy storage error"),
                status: ArsParameterPolicyLoadStatus::StorageError,
                error: Some(e.to_string()),
            };
        }
    };

    let Some(raw) = raw else {
        return ArsParameterPolicyLoad {
            policy: ArsParameterPolicy::default(),
            status: ArsParameterPolicyLoadStatus::Missing,
            error: None,
        };
    };

    // Inspect the raw JSON before typed deserialization. Future schemas may
    // add enum variants this binary cannot parse; they must still classify as
    // UnsupportedSchema so a downgraded doctor never deletes their bytes.
    let value = match serde_json::from_str::<serde_json::Value>(&raw) {
        Ok(value) => value,
        Err(error) => {
            return corrupt_parameter_policy_load(error.to_string());
        }
    };
    let detected_schema = match value.get("schema_version") {
        None => LEGACY_ARS_PARAMETER_POLICY_SCHEMA_VERSION,
        Some(schema) => match schema.as_u64().and_then(|value| u32::try_from(value).ok()) {
            Some(schema) => schema,
            None => {
                return corrupt_parameter_policy_load(
                    "policy schema_version must be a non-negative u32".to_string(),
                );
            }
        },
    };
    if detected_schema > ARS_PARAMETER_POLICY_SCHEMA_VERSION {
        return ArsParameterPolicyLoad {
            policy: ArsParameterPolicy::disabled("unsupported policy schema version"),
            status: ArsParameterPolicyLoadStatus::UnsupportedSchema,
            error: Some(format!(
                "policy schema_version={} is newer than binary schema_version={}; \
                 preserve until the newer binary is restored",
                detected_schema, ARS_PARAMETER_POLICY_SCHEMA_VERSION
            )),
        };
    }
    if !matches!(
        detected_schema,
        LEGACY_ARS_PARAMETER_POLICY_SCHEMA_VERSION
            | LEGACY_A12_ARS_PARAMETER_POLICY_SCHEMA_VERSION
            | ARS_PARAMETER_POLICY_SCHEMA_VERSION
    ) {
        return corrupt_parameter_policy_load(format!(
            "policy schema_version={} is older or invalid (supported schemas are {}, {}, and {})",
            detected_schema,
            LEGACY_ARS_PARAMETER_POLICY_SCHEMA_VERSION,
            LEGACY_A12_ARS_PARAMETER_POLICY_SCHEMA_VERSION,
            ARS_PARAMETER_POLICY_SCHEMA_VERSION
        ));
    }

    match serde_json::from_value::<ArsParameterPolicy>(value) {
        Ok(mut policy) => {
            // `#[serde(default)]` deliberately describes newly-created values,
            // not historical on-disk identity. Preserve the schema detected
            // above so legacy rows remain fail-closed until a schema-3 CAS.
            policy.schema_version = detected_schema;
            if detected_schema == LEGACY_ARS_PARAMETER_POLICY_SCHEMA_VERSION {
                policy.recall_fusion_evidence.clear();
            }
            if detected_schema != ARS_PARAMETER_POLICY_SCHEMA_VERSION {
                clamp_policy_weights(&mut policy);
                return ArsParameterPolicyLoad {
                    policy,
                    status: ArsParameterPolicyLoadStatus::Loaded,
                    error: Some(format!(
                        "legacy policy schema {detected_schema} loaded fail-closed; next policy refresh must migrate it"
                    )),
                };
            }
            // Validation must observe the raw persisted values: clamping first
            // would silently launder a non-finite or out-of-range adoption
            // weight into a healthy load instead of failing closed as Corrupt.
            if let Err(error) = validate_parameter_policy(&policy) {
                return corrupt_parameter_policy_load(error);
            }
            clamp_policy_weights(&mut policy);
            ArsParameterPolicyLoad {
                policy,
                status: ArsParameterPolicyLoadStatus::Loaded,
                error: None,
            }
        }
        Err(error) => corrupt_parameter_policy_load(error.to_string()),
    }
}

#[must_use = "callers must not assume a policy update landed after a CAS miss"]
pub fn save_parameter_policy_cas(
    conn: &rusqlite::Connection,
    policy: &ArsParameterPolicy,
    expected_revision: u64,
) -> rusqlite::Result<bool> {
    let mut policy = policy.clone();
    policy.schema_version = ARS_PARAMETER_POLICY_SCHEMA_VERSION;
    clamp_policy_weights(&mut policy);
    validate_parameter_policy(&policy).map_err(invalid_parameter_policy_error)?;
    let json = serde_json::to_string(&policy)
        .expect("ArsParameterPolicy serialization cannot fail for finite fields");

    // R6 P2 defense-in-depth (2026-05-04): the UPDATE predicate must
    // ALSO match this binary's `ARS_PARAMETER_POLICY_SCHEMA_VERSION`.
    // Pre-fix the predicate was revision-only, so if a downgraded
    // (older) binary's caller built a default policy at revision=0
    // and a future-schema row happened to have a missing or zero
    // `revision` field, the UPDATE would silently overwrite valid
    // future-schema bytes. Adding the schema-version guard makes the
    // CAS fail-closed against ANY caller — including hypothetical
    // future code paths that bypass the refresh-layer early-return —
    // and the row survives the downgrade window untouched.
    //
    // Schema 1, schema 2, and missing-schema rows are explicitly migratable to
    // schema 3. Future rows cannot satisfy the allowlist and retain their exact
    // bytes.
    let updated = conn.execute(
        "UPDATE metadata
            SET value = ?1
          WHERE key = ?2
            AND json_valid(value)
            AND COALESCE(json_extract(value, '$.revision'), 0) = ?3
            AND COALESCE(json_extract(value, '$.schema_version'), ?4) IN (?4, ?5, ?6)",
        params![
            json,
            ARS_PARAMETER_POLICY_METADATA_KEY,
            expected_revision,
            LEGACY_ARS_PARAMETER_POLICY_SCHEMA_VERSION,
            LEGACY_A12_ARS_PARAMETER_POLICY_SCHEMA_VERSION,
            ARS_PARAMETER_POLICY_SCHEMA_VERSION,
        ],
    )?;
    if updated == 1 {
        return Ok(true);
    }

    let exists: bool = conn.query_row(
        "SELECT COUNT(*) > 0 FROM metadata WHERE key = ?1",
        params![ARS_PARAMETER_POLICY_METADATA_KEY],
        |row| row.get(0),
    )?;
    if exists || expected_revision != 0 {
        return Ok(false);
    }

    match conn.execute(
        "INSERT INTO metadata (key, value) VALUES (?1, ?2)",
        params![ARS_PARAMETER_POLICY_METADATA_KEY, json],
    ) {
        Ok(1) => Ok(true),
        Ok(_) => Ok(false),
        Err(rusqlite::Error::SqliteFailure(err, _))
            if err.code == rusqlite::ErrorCode::ConstraintViolation =>
        {
            Ok(false)
        }
        Err(e) => Err(e),
    }
}

pub fn delete_parameter_policy(conn: &rusqlite::Connection) -> rusqlite::Result<usize> {
    conn.execute(
        "DELETE FROM metadata WHERE key = ?1",
        params![ARS_PARAMETER_POLICY_METADATA_KEY],
    )
}

/// Result of a `repair_corrupt_parameter_policy` call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepairCorruptOutcome {
    /// Number of rows deleted (0 if the row was no longer Corrupt at
    /// recovery time — e.g., a peer wrote a healthy row in between).
    pub deleted: usize,
    /// The corrupt-row error message captured INSIDE the transaction
    /// at delete time. Only populated when `deleted > 0`.
    pub error_at_delete: Option<String>,
    /// The status observed inside the recovery transaction; useful for
    /// log messages distinguishing "deleted" from "skipped (peer
    /// already repaired)" or "skipped (now StorageError)".
    pub observed_status: ArsParameterPolicyLoadStatus,
}

/// R10 P3 (2026-05-04): atomically check the policy row's status and
/// DELETE only if it is currently `Corrupt`. Closes a TOCTOU race
/// where a peer `doctor --fix` or `refresh_ars_parameter_policy`
/// could rewrite the row to a healthy canary state between an
/// earlier `load_parameter_policy` call and an unconditional
/// `delete_parameter_policy` — destroying valid state that just
/// landed.
///
/// The recovery wraps the load + delete in a single
/// `BEGIN IMMEDIATE` transaction (matches the resummerize / judge
/// contract pattern in this crate). `BEGIN IMMEDIATE` acquires the
/// write lock immediately, so no peer can interleave between the
/// status check and the delete.
///
/// Returns `RepairCorruptOutcome::observed_status` so callers can
/// distinguish:
/// - `Corrupt` + `deleted > 0` → row was still corrupt, recovery applied.
/// - `Loaded` / `Missing` + `deleted == 0` → peer already repaired
///   the row, recovery declined to touch it.
/// - `UnsupportedSchema` / `StorageError` + `deleted == 0` → row
///   transitioned to a different unhealthy state under the lock;
///   recovery declined per the R4/R5 "destructive only on Corrupt"
///   discipline.
#[must_use = "callers must report whether recovery actually deleted the row"]
pub fn repair_corrupt_parameter_policy(
    conn: &rusqlite::Connection,
) -> rusqlite::Result<RepairCorruptOutcome> {
    conn.execute_batch("BEGIN IMMEDIATE")?;
    let recovery: rusqlite::Result<RepairCorruptOutcome> = (|| {
        let loaded = load_parameter_policy(conn);
        if !matches!(loaded.status, ArsParameterPolicyLoadStatus::Corrupt) {
            return Ok(RepairCorruptOutcome {
                deleted: 0,
                error_at_delete: None,
                observed_status: loaded.status,
            });
        }
        let deleted = conn.execute(
            "DELETE FROM metadata WHERE key = ?1",
            params![ARS_PARAMETER_POLICY_METADATA_KEY],
        )?;
        Ok(RepairCorruptOutcome {
            deleted,
            error_at_delete: loaded.error,
            observed_status: loaded.status,
        })
    })();
    match recovery {
        Ok(outcome) => {
            conn.execute_batch("COMMIT")?;
            Ok(outcome)
        }
        Err(e) => {
            // Best-effort rollback; surface the original error.
            let _ = conn.execute_batch("ROLLBACK");
            Err(e)
        }
    }
}

fn default_schema_version() -> u32 {
    ARS_PARAMETER_POLICY_SCHEMA_VERSION
}

fn corrupt_parameter_policy_load(error: String) -> ArsParameterPolicyLoad {
    ArsParameterPolicyLoad {
        policy: ArsParameterPolicy::disabled("corrupt policy row"),
        status: ArsParameterPolicyLoadStatus::Corrupt,
        error: Some(error),
    }
}

fn invalid_parameter_policy_error(error: String) -> rusqlite::Error {
    rusqlite::Error::ToSqlConversionFailure(Box::new(io::Error::new(
        io::ErrorKind::InvalidInput,
        error,
    )))
}

fn validate_parameter_policy(policy: &ArsParameterPolicy) -> Result<(), String> {
    if policy.schema_version != ARS_PARAMETER_POLICY_SCHEMA_VERSION {
        return Err(format!(
            "policy writer requires schema_version={} (observed {})",
            ARS_PARAMETER_POLICY_SCHEMA_VERSION, policy.schema_version
        ));
    }
    if !policy.runtime_adoption_weight.is_finite()
        || !(0.0..=1.0).contains(&policy.runtime_adoption_weight)
    {
        return Err(format!(
            "policy runtime_adoption_weight must be finite and in [0, 1] (observed {})",
            policy.runtime_adoption_weight
        ));
    }
    for (key, weight) in &policy.adoption_weights {
        if !weight.is_finite() || !(0.0..=1.0).contains(weight) {
            return Err(format!(
                "policy adoption weight `{key}` must be finite and in [0, 1] (observed {weight})"
            ));
        }
    }
    for (key, evidence) in &policy.recall_fusion_evidence {
        let Some(scope) = key.strip_prefix("recall_fusion:") else {
            return Err(format!(
                "recall-fusion evidence key `{key}` is outside the recall_fusion namespace"
            ));
        };
        if scope.is_empty() || scope.chars().any(char::is_whitespace) {
            return Err(format!(
                "recall-fusion evidence key `{key}` has an invalid scope"
            ));
        }
        validate_recall_fusion_evidence(key, evidence)?;
    }
    Ok(())
}

fn validate_recall_fusion_evidence(
    key: &str,
    evidence: &ArsRecallFusionEvidence,
) -> Result<(), String> {
    let simplex = evidence.resolved_simplex;
    let values = [
        simplex.bm25,
        simplex.vector,
        simplex.kg,
        simplex.episode,
        simplex.support,
        simplex.diversity,
    ];
    if values
        .iter()
        .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        return Err(format!(
            "recall-fusion evidence `{key}` has a non-finite or out-of-range simplex"
        ));
    }
    let sum = values.iter().sum::<f64>();
    if (sum - 1.0).abs() > 1e-6 {
        return Err(format!(
            "recall-fusion evidence `{key}` simplex must sum to 1 (observed {sum})"
        ));
    }
    for (label, fingerprint) in [
        ("generation", evidence.generation_fingerprint.as_deref()),
        ("corpus", evidence.corpus_fingerprint.as_deref()),
        ("optimizer", evidence.optimizer_fingerprint.as_deref()),
        ("evaluation", evidence.evaluation_fingerprint.as_deref()),
        (
            "recall-gate build",
            evidence.recall_gate_build_fingerprint.as_deref(),
        ),
        (
            "recall-gate fixture",
            evidence.recall_gate_fixture_fingerprint.as_deref(),
        ),
    ] {
        if fingerprint.is_some_and(|value| value.trim().is_empty()) {
            return Err(format!(
                "recall-fusion evidence `{key}` has an empty {label} fingerprint"
            ));
        }
    }
    if evidence
        .a12_noise_floor
        .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value) || value == 0.0)
    {
        return Err(format!(
            "recall-fusion evidence `{key}` A12 noise floor must be finite and in (0, 1]"
        ));
    }
    if evidence.calibrated_at.is_some_and(|value| value <= 0)
        || evidence.evaluated_at.is_some_and(|value| value <= 0)
        || evidence
            .recall_gate_evaluated_at
            .is_some_and(|value| value <= 0)
    {
        return Err(format!(
            "recall-fusion evidence `{key}` timestamps must be positive"
        ));
    }
    if matches!(
        (evidence.calibrated_at, evidence.evaluated_at),
        (Some(calibrated), Some(evaluated)) if evaluated < calibrated
    ) {
        return Err(format!(
            "recall-fusion evidence `{key}` A12 evaluation predates calibration"
        ));
    }
    let evaluated_at_millis = evidence
        .evaluated_at
        .and_then(|evaluated| evaluated.checked_mul(1_000));
    if evidence.evaluated_at.is_some() && evaluated_at_millis.is_none()
        || evidence
            .a12_valid_until_exclusive
            .is_some_and(|boundary| boundary <= evaluated_at_millis.unwrap_or(i64::MAX))
    {
        return Err(format!(
            "recall-fusion evidence `{key}` validity boundary must follow evaluation"
        ));
    }

    let uses_human = matches!(
        evidence.basis,
        ArsRecallFusionEvidenceBasis::Human | ArsRecallFusionEvidenceBasis::Blended
    );
    let uses_automatic = matches!(
        evidence.basis,
        ArsRecallFusionEvidenceBasis::SelfSupervised | ArsRecallFusionEvidenceBasis::Blended
    );
    if uses_human && evidence.human_ess == 0 {
        return Err(format!(
            "recall-fusion evidence `{key}` declares human evidence with zero ESS"
        ));
    }
    if let Some(simplex) = evidence.human_simplex {
        let values = [
            simplex.bm25,
            simplex.vector,
            simplex.kg,
            simplex.episode,
            simplex.support,
            simplex.diversity,
        ];
        if values
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
            || (values.iter().sum::<f64>() - 1.0).abs() > 1e-6
        {
            return Err(format!(
                "recall-fusion evidence `{key}` has an invalid sealed human simplex"
            ));
        }
    }
    if evidence
        .human_runtime_adoption_weight
        .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
    {
        return Err(format!(
            "recall-fusion evidence `{key}` human runtime adoption weight must be finite and in [0, 1]"
        ));
    }
    if evidence.human_runtime_adoption_weight.is_some()
        && (evidence.human_simplex.is_none() || evidence.human_ess == 0)
    {
        return Err(format!(
            "recall-fusion evidence `{key}` seals human adoption without complete human evidence"
        ));
    }
    if evidence.automatic_candidate_present
        && evidence.human_ess > 0
        && (evidence.human_simplex.is_none() || evidence.human_runtime_adoption_weight.is_none())
    {
        return Err(format!(
            "recall-fusion evidence `{key}` automatic boundary is missing sealed human fallback"
        ));
    }
    if evidence.automatic_candidate_present {
        let missing_candidate_identity = evidence.a12_generation.is_none()
            || evidence.a12_revision.is_none()
            || evidence.generation_fingerprint.is_none()
            || evidence.corpus_fingerprint.is_none()
            || evidence.optimizer_fingerprint.is_none()
            || evidence.evaluation_fingerprint.is_none()
            || evidence.a12_verdict.is_none()
            || evidence.a12_noise_floor.is_none()
            || evidence.calibrated_at.is_none()
            || evidence.evaluated_at.is_none();
        if missing_candidate_identity {
            return Err(format!(
                "recall-fusion evidence `{key}` has incomplete A12 candidate identity"
            ));
        }
    }
    if uses_automatic {
        let missing_identity = evidence.a12_generation.is_none()
            || evidence.a12_revision.is_none()
            || evidence.generation_fingerprint.is_none()
            || evidence.corpus_fingerprint.is_none()
            || evidence.optimizer_fingerprint.is_none()
            || evidence.evaluation_fingerprint.is_none()
            || evidence.a12_verdict.is_none()
            || evidence.a12_noise_floor.is_none()
            || evidence.recall_gate_build_fingerprint.is_none()
            || evidence.recall_gate_fixture_fingerprint.is_none()
            || evidence.recall_gate_evaluated_at.is_none()
            || evidence.calibrated_at.is_none()
            || evidence.evaluated_at.is_none();
        if missing_identity {
            return Err(format!(
                "recall-fusion evidence `{key}` has incomplete self-supervised attestation"
            ));
        }
        if evidence.self_supervised_train_family_ess == 0
            || evidence.self_supervised_holdout_family_ess == 0
        {
            return Err(format!(
                "recall-fusion evidence `{key}` has zero self-supervised family ESS"
            ));
        }
    }
    Ok(())
}

fn clamp01(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// v0.28.7+ audit L6 — defense-in-depth cap on
/// `ArsParameterPolicy.adoption_weights`. The map is repopulated from
/// `state.learned_shadow_fusion` on every pipeline tick (see
/// `next_scoped_adoption_weights`), so its size is dominated by the
/// learned_shadow_fusion cap (4096) plus a small fixed set of global
/// keys (`synthesis_gate`, `concept_summary_gate`, `judge_sample_rate`,
/// `llm_feedback_decay`, `signal_hint_priors`). Setting the cap above
/// 4096 + 32 = 4128 leaves comfortable headroom for that arithmetic and
/// for future global keys while still bounding pathological growth from
/// an unknown insert path.
///
/// Per the audit's "defense-in-depth" framing — and per the user's
/// no-subjective-heuristics guidance — this is **warn-only**: a save
/// that exceeds the cap still lands so the operator never silently
/// loses a key the canary policy depends on. The warning is the
/// signal; truncation is left to a deliberate v0.29 schema migration if
/// the cap is ever actually approached.
pub const ADOPTION_WEIGHTS_CAP: usize = 4128;

fn clamp_policy_weights(policy: &mut ArsParameterPolicy) {
    policy.runtime_adoption_weight = clamp01(policy.runtime_adoption_weight);
    for value in policy.adoption_weights.values_mut() {
        *value = clamp01(*value);
    }
    if policy.adoption_weights.len() > ADOPTION_WEIGHTS_CAP {
        // Warn-only — see ADOPTION_WEIGHTS_CAP doc for rationale.
        // tracing::warn! emits at WARN level so the doctor / GUI can
        // surface this as an operator alert. We do NOT truncate
        // because every adoption_weights entry maps to a scope identifier
        // that the runtime trust gate consults; silently dropping one
        // would silently mute a canary scope.
        tracing::warn!(
            adoption_weights_len = policy.adoption_weights.len(),
            cap = ADOPTION_WEIGHTS_CAP,
            "ars_parameter_policy: adoption_weights size exceeds defense-in-depth cap; \
             a runaway clusterer or unknown insert path may be growing the map. \
             No keys dropped — investigate cardinality source"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::alpha_optimizer::ShadowFusionWeights;
    use crate::store::adaptive::{
        AdaptiveState, LearnedShadowFusionEntry, ShadowFusionWeightEntry,
    };

    fn conn() -> rusqlite::Connection {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute_batch("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
            .unwrap();
        conn
    }

    fn canary_policy(revision: u64) -> ArsParameterPolicy {
        ArsParameterPolicy {
            schema_version: ARS_PARAMETER_POLICY_SCHEMA_VERSION,
            revision,
            mode: ArsParameterPolicyMode::Canary,
            disabled_reason: None,
            source_adaptive_version: 7,
            runtime_adoption_weight: 1.0,
            adoption_weights: HashMap::new(),
            recall_fusion_evidence: HashMap::new(),
            last_event_id: 99,
            last_updated: "2026-05-01T00:00:00Z".to_string(),
        }
    }

    #[test]
    fn parameter_policy_schema_three_round_trips_recall_fusion_evidence() {
        let conn = conn();
        let mut policy = canary_policy(1);
        policy.recall_fusion_evidence.insert(
            "recall_fusion:semantic".to_string(),
            ArsRecallFusionEvidence {
                basis: ArsRecallFusionEvidenceBasis::Blended,
                resolved_simplex: crate::store::a12_calibration::A12FusionSimplex {
                    bm25: 0.4,
                    vector: 0.3,
                    kg: 0.1,
                    episode: 0.1,
                    support: 0.05,
                    diversity: 0.05,
                },
                human_ess: 40,
                automatic_candidate_present: true,
                human_simplex: Some(crate::store::a12_calibration::A12FusionSimplex {
                    bm25: 0.4,
                    vector: 0.3,
                    kg: 0.1,
                    episode: 0.1,
                    support: 0.05,
                    diversity: 0.05,
                }),
                human_runtime_adoption_weight: Some(0.25),
                self_supervised_train_family_ess: 80,
                self_supervised_holdout_family_ess: 20,
                a12_generation: Some(7),
                a12_revision: Some(9),
                generation_fingerprint: Some("generation-fp".to_string()),
                corpus_fingerprint: Some("corpus-fp".to_string()),
                optimizer_fingerprint: Some("optimizer-fp".to_string()),
                evaluation_fingerprint: Some("evaluation-fp".to_string()),
                a12_verdict: Some(crate::store::a12_calibration::A12CalibrationVerdict::Ship),
                a12_noise_floor: Some(0.02),
                recall_gate_status: ArsRecallGateStatus::Ship,
                recall_gate_build_fingerprint: Some("build-fp".to_string()),
                recall_gate_fixture_fingerprint: Some("fixture-fp".to_string()),
                recall_gate_evaluated_at: Some(990),
                calibrated_at: Some(1_000),
                evaluated_at: Some(1_010),
                a12_valid_until_exclusive: Some(2_000_000),
                reason: "human and holdout-approved automatic evidence".to_string(),
            },
        );

        assert!(save_parameter_policy_cas(&conn, &policy, 0).unwrap());
        let loaded = load_parameter_policy(&conn);

        assert_eq!(loaded.status, ArsParameterPolicyLoadStatus::Loaded);
        assert_eq!(
            loaded.policy.schema_version,
            ARS_PARAMETER_POLICY_SCHEMA_VERSION
        );
        assert_eq!(loaded.policy, policy);
    }

    #[test]
    fn parameter_policy_schema_one_loads_fail_closed_then_migrates_by_cas() {
        let conn = conn();
        let legacy = serde_json::json!({
            "schema_version": 1,
            "revision": 4,
            "mode": "canary",
            "source_adaptive_version": 7,
            "runtime_adoption_weight": 1.0,
            "adoption_weights": {"recall_fusion:global": 0.5},
            "last_event_id": 99,
            "last_updated": "2026-05-01T00:00:00Z"
        })
        .to_string();
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES (?1, ?2)",
            params![ARS_PARAMETER_POLICY_METADATA_KEY, legacy],
        )
        .unwrap();

        let loaded = load_parameter_policy(&conn);
        assert_eq!(loaded.status, ArsParameterPolicyLoadStatus::Loaded);
        assert_eq!(loaded.policy.schema_version, 1);
        assert_eq!(loaded.policy.runtime_adoption_weight(7), 0.0);

        let migrated = ArsParameterPolicy {
            schema_version: ARS_PARAMETER_POLICY_SCHEMA_VERSION,
            revision: 5,
            mode: ArsParameterPolicyMode::Canary,
            disabled_reason: None,
            source_adaptive_version: 7,
            runtime_adoption_weight: 0.0,
            adoption_weights: HashMap::from([("recall_fusion:global".to_string(), 0.05)]),
            recall_fusion_evidence: HashMap::new(),
            last_event_id: 99,
            last_updated: "2026-07-13T00:00:00Z".to_string(),
        };
        assert!(save_parameter_policy_cas(&conn, &migrated, 4).unwrap());
        let loaded = load_parameter_policy(&conn);
        assert_eq!(
            loaded.policy.schema_version,
            ARS_PARAMETER_POLICY_SCHEMA_VERSION
        );
        assert_eq!(loaded.policy.revision, 5);
        assert_eq!(
            loaded
                .policy
                .runtime_adoption_weight_for(7, "recall_fusion:global"),
            0.05
        );
    }

    #[test]
    fn parameter_policy_schema_two_a12_evidence_loads_fail_closed_then_migrates_to_three() {
        let conn = conn();
        let legacy = serde_json::json!({
            "schema_version": 2,
            "revision": 4,
            "mode": "canary",
            "source_adaptive_version": 7,
            "runtime_adoption_weight": 0.0,
            "adoption_weights": {"recall_fusion:global": 0.5},
            "recall_fusion_evidence": {
                "recall_fusion:global": {
                    "basis": "self_supervised",
                    "resolved_simplex": {
                        "bm25": 0.45,
                        "vector": 0.45,
                        "kg": 0.04,
                        "episode": 0.03,
                        "support": 0.02,
                        "diversity": 0.01
                    },
                    "human_ess": 0,
                    "self_supervised_train_family_ess": 20,
                    "self_supervised_holdout_family_ess": 20,
                    "a12_generation": 7,
                    "a12_revision": 9,
                    "generation_fingerprint": "generation-fp",
                    "corpus_fingerprint": "corpus-fp",
                    "optimizer_fingerprint": "optimizer-fp",
                    "evaluation_fingerprint": "evaluation-fp",
                    "a12_verdict": "ship",
                    "a12_noise_floor": 0.02,
                    "recall_gate_status": "ship",
                    "recall_gate_build_fingerprint": "build-fp",
                    "recall_gate_fixture_fingerprint": "fixture-fp",
                    "recall_gate_evaluated_at": 990,
                    "calibrated_at": 1000,
                    "evaluated_at": 1010,
                    "reason": "legacy schema-two A12 evidence"
                }
            },
            "last_event_id": 99,
            "last_updated": "2026-07-13T00:00:00Z"
        })
        .to_string();
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES (?1, ?2)",
            params![ARS_PARAMETER_POLICY_METADATA_KEY, legacy],
        )
        .unwrap();

        let loaded = load_parameter_policy(&conn);
        assert_eq!(loaded.status, ArsParameterPolicyLoadStatus::Loaded);
        assert_eq!(loaded.policy.schema_version, 2);
        assert_eq!(
            loaded
                .policy
                .runtime_adoption_weight_for(7, "recall_fusion:global"),
            0.0,
            "schema-two A12 policy must not activate without schema-three fallback seals"
        );

        let mut migrated = canary_policy(5);
        migrated.runtime_adoption_weight = 0.0;
        migrated
            .adoption_weights
            .insert("recall_fusion:global".to_string(), 0.05);
        assert!(save_parameter_policy_cas(&conn, &migrated, 4).unwrap());

        let loaded = load_parameter_policy(&conn);
        assert_eq!(loaded.status, ArsParameterPolicyLoadStatus::Loaded);
        assert_eq!(loaded.policy.schema_version, 3);
        assert_eq!(loaded.policy.revision, 5);
        assert_eq!(
            loaded
                .policy
                .runtime_adoption_weight_for(7, "recall_fusion:global"),
            0.05
        );
    }

    #[test]
    fn parameter_policy_missing_schema_is_treated_as_legacy_one_not_current_three() {
        let conn = conn();
        let legacy = serde_json::json!({
            "revision": 3,
            "mode": "canary",
            "source_adaptive_version": 7,
            "runtime_adoption_weight": 1.0,
            "adoption_weights": {},
            "last_event_id": 0,
            "last_updated": "2026-05-01T00:00:00Z"
        })
        .to_string();
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES (?1, ?2)",
            params![ARS_PARAMETER_POLICY_METADATA_KEY, legacy],
        )
        .unwrap();

        let loaded = load_parameter_policy(&conn);
        assert_eq!(loaded.status, ArsParameterPolicyLoadStatus::Loaded);
        assert_eq!(loaded.policy.schema_version, 1);
        assert_eq!(loaded.policy.runtime_adoption_weight(7), 0.0);
    }

    #[test]
    fn parameter_policy_missing_or_corrupt_loads_disabled() {
        let conn = conn();
        let missing = load_parameter_policy(&conn);
        assert_eq!(missing.status, ArsParameterPolicyLoadStatus::Missing);
        assert_eq!(missing.policy.mode, ArsParameterPolicyMode::Disabled);

        conn.execute(
            "INSERT INTO metadata (key, value) VALUES (?1, ?2)",
            params![ARS_PARAMETER_POLICY_METADATA_KEY, "{not json"],
        )
        .unwrap();

        let corrupt = load_parameter_policy(&conn);
        assert_eq!(corrupt.status, ArsParameterPolicyLoadStatus::Corrupt);
        assert_eq!(corrupt.policy.mode, ArsParameterPolicyMode::Disabled);
        assert!(corrupt.error.is_some());
    }

    #[test]
    fn parameter_policy_save_cas_rejects_stale_revision() {
        let conn = conn();
        assert!(save_parameter_policy_cas(&conn, &canary_policy(1), 0).unwrap());

        let stale = canary_policy(2);
        assert!(!save_parameter_policy_cas(&conn, &stale, 0).unwrap());

        let loaded = load_parameter_policy(&conn);
        assert_eq!(loaded.policy.revision, 1);
        assert!(save_parameter_policy_cas(&conn, &stale, 1).unwrap());
        assert_eq!(load_parameter_policy(&conn).policy.revision, 2);
    }

    /// v0.28.7+ audit L5 R6 P2 — `save_parameter_policy_cas` MUST
    /// refuse to overwrite a row whose `schema_version` is FUTURE
    /// relative to this binary's `ARS_PARAMETER_POLICY_SCHEMA_VERSION`.
    /// Pre-fix the CAS predicate was revision-only, so a future-schema
    /// row with a missing/zero `revision` field could be silently
    /// overwritten by a downgraded binary's default-policy save at
    /// `expected_revision=0`. The schema-version guard makes the CAS
    /// fail-closed regardless of revision arithmetic.
    #[test]
    fn parameter_policy_save_cas_refuses_to_overwrite_future_schema_row() {
        let conn = conn();
        // Plant a future-schema row whose `revision` is missing
        // (defaults to 0 via COALESCE), schema_version is 9999.
        let future_value = serde_json::json!({
            "schema_version": 9999,
            "mode": "canary",
            "source_adaptive_version": 0,
            "runtime_adoption_weight": 0.5,
            "adoption_weights": {},
            "recall_fusion_evidence": {
                "recall_fusion:semantic:7": {
                    "basis": "future_blended",
                    "automatic_candidate_present": true,
                    "human_simplex": {"future_axis": 1.0},
                    "human_runtime_adoption_weight": 0.4
                }
            },
            "last_event_id": 0,
            "last_updated": "2030-01-01T00:00:00Z",
            "future_field": "neat",
        })
        .to_string();
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES (?1, ?2)",
            params![ARS_PARAMETER_POLICY_METADATA_KEY, &future_value],
        )
        .unwrap();

        // A downgraded binary builds a default disabled policy and
        // calls CAS at expected_revision=0 (matches the COALESCE
        // arithmetic on the future row).
        let downgraded_default = ArsParameterPolicy::disabled("downgrade reset attempt");
        let cas_result = save_parameter_policy_cas(&conn, &downgraded_default, 0).unwrap();
        assert!(
            !cas_result,
            "CAS must reject the overwrite — the schema_version guard \
             makes the UPDATE predicate fail even though revision matches. \
             Pre-R6 fix this returned `true` and silently destroyed the \
             future-schema row's bytes."
        );

        // The original future-schema bytes must still be in the DB.
        let raw_now: String = conn
            .query_row(
                "SELECT value FROM metadata WHERE key = ?1",
                params![ARS_PARAMETER_POLICY_METADATA_KEY],
                |row| row.get(0),
            )
            .expect("future-schema row must survive the CAS call");
        assert_eq!(
            raw_now, future_value,
            "future-schema row's raw bytes must be preserved exactly across \
             a downgrade-binary CAS attempt"
        );

        // Sanity: load_parameter_policy still reports UnsupportedSchema
        // (the CAS didn't accidentally clamp/normalize the row either).
        let loaded = load_parameter_policy(&conn);
        assert_eq!(
            loaded.status,
            ArsParameterPolicyLoadStatus::UnsupportedSchema
        );
    }

    /// v0.28.7+ audit R8 P2 #1 — a future-schema row that ALSO contains
    /// fields this binary cannot deserialize (e.g., a new `mode` enum
    /// variant) MUST be classified as `UnsupportedSchema`, not
    /// `Corrupt`. Pre-R8 the typed `serde_json::from_str::<ArsParameterPolicy>`
    /// ran first and failed before the schema-version branch ever
    /// fired, so the row fell into the `Err` arm and `doctor --fix`
    /// would delete valid future canary state on a downgrade. The
    /// load path now peeks `schema_version` from the raw JSON Value
    /// before attempting the typed parse.
    #[test]
    fn parameter_policy_load_future_schema_with_unknown_mode_is_unsupported_schema_not_corrupt() {
        let conn = conn();
        let future_value = serde_json::json!({
            "schema_version": 9999,
            "revision": 5,
            "mode": "future_only_variant",
            "source_adaptive_version": 0,
            "runtime_adoption_weight": 0.5,
            "adoption_weights": {},
            "last_event_id": 0,
            "last_updated": "2030-01-01T00:00:00Z",
            "future_only_field": "neat",
        })
        .to_string();
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES (?1, ?2)",
            params![ARS_PARAMETER_POLICY_METADATA_KEY, &future_value],
        )
        .unwrap();
        let loaded = load_parameter_policy(&conn);
        assert_eq!(
            loaded.status,
            ArsParameterPolicyLoadStatus::UnsupportedSchema,
            "future-schema row with an unknown mode variant must be classified \
             UnsupportedSchema (not Corrupt) so doctor --fix preserves it on \
             downgrade. Pre-R8 fix the typed deserialize failed first and the \
             row was wrongly destroyed."
        );
        // The raw bytes must still be intact — the load path must not
        // mutate the row regardless of how it classifies it.
        let raw_now: String = conn
            .query_row(
                "SELECT value FROM metadata WHERE key = ?1",
                params![ARS_PARAMETER_POLICY_METADATA_KEY],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(raw_now, future_value);
    }

    /// v0.28.7+ audit R8 P2 #2 — a row whose JSON omits the
    /// `schema_version` field entirely (e.g., one written by an older
    /// binary before the field was introduced) must deserialize via
    /// `#[serde(default)]` to schema=1 and accept CAS UPDATE under
    /// the matching revision. Pre-R8 the CAS predicate compared
    /// `COALESCE(..., 0) = ?4=1`, treating the missing field as 0 and
    /// silently rejecting every refresh. Combined with the existing-row
    /// check that prevented INSERT, policy promotion or rollback
    /// would stall permanently for upgraded rows.
    #[test]
    fn parameter_policy_save_cas_accepts_row_with_missing_schema_version_field() {
        let conn = conn();
        // Plant a row that lacks `schema_version` entirely.
        let no_schema = serde_json::json!({
            "revision": 7,
            "mode": "canary",
            "source_adaptive_version": 5,
            "runtime_adoption_weight": 0.3,
            "adoption_weights": {},
            "last_event_id": 12,
            "last_updated": "2026-04-01T00:00:00Z",
        })
        .to_string();
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES (?1, ?2)",
            params![ARS_PARAMETER_POLICY_METADATA_KEY, &no_schema],
        )
        .unwrap();

        // Missing schema is historical schema 1, not whatever schema this
        // binary happens to write today. It loads for migration but cannot
        // activate runtime behavior.
        let loaded = load_parameter_policy(&conn);
        assert_eq!(loaded.status, ArsParameterPolicyLoadStatus::Loaded);
        assert_eq!(
            loaded.policy.schema_version,
            LEGACY_ARS_PARAMETER_POLICY_SCHEMA_VERSION
        );
        assert_eq!(loaded.policy.revision, 7);
        assert_eq!(loaded.policy.runtime_adoption_weight(5), 0.0);

        // Build a refresh against the loaded policy at revision 7 (matches
        // the row's stored revision). The CAS UPDATE predicate must accept
        // this even though the on-disk JSON omits schema_version entirely.
        let mut refreshed = loaded.policy.clone();
        refreshed.schema_version = ARS_PARAMETER_POLICY_SCHEMA_VERSION;
        refreshed.revision = 8;
        refreshed.runtime_adoption_weight = 0.6;

        assert!(
            save_parameter_policy_cas(&conn, &refreshed, 7).unwrap(),
            "CAS UPDATE must succeed against a row that omits schema_version \
             (treated as migratable legacy schema 1 by the CAS allowlist)."
        );

        // Verify the new bytes landed and now include schema_version.
        let after = load_parameter_policy(&conn);
        assert_eq!(after.policy.revision, 8);
        assert!((after.policy.runtime_adoption_weight - 0.6).abs() < f64::EPSILON);
        assert_eq!(
            after.policy.schema_version,
            ARS_PARAMETER_POLICY_SCHEMA_VERSION
        );
    }

    /// R8 P2 #2 follow-up — the COALESCE-default change MUST NOT
    /// regress the R6 future-row preservation property.  A future
    /// `schema_version=9999` row must still reject the downgrade-binary
    /// CAS even when the predicate's COALESCE default is `?4`
    /// (= current schema = 1) instead of 0, because COALESCE only
    /// substitutes when the field is missing — an explicit 9999 value
    /// is preserved by the comparison.
    #[test]
    fn parameter_policy_save_cas_still_refuses_future_schema_after_coalesce_default_change() {
        let conn = conn();
        let future_value = serde_json::json!({
            "schema_version": 9999,
            "revision": 0,
            "mode": "canary",
            "source_adaptive_version": 0,
            "runtime_adoption_weight": 0.5,
            "adoption_weights": {},
            "last_event_id": 0,
            "last_updated": "2030-01-01T00:00:00Z",
        })
        .to_string();
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES (?1, ?2)",
            params![ARS_PARAMETER_POLICY_METADATA_KEY, &future_value],
        )
        .unwrap();

        let downgraded_default = ArsParameterPolicy::disabled("downgrade reset attempt");
        let cas_result = save_parameter_policy_cas(&conn, &downgraded_default, 0).unwrap();
        assert!(
            !cas_result,
            "future-schema row preservation must hold even after the \
             COALESCE default flipped from 0 → ?4. The explicit 9999 \
             value still cannot match ?4=1 in the comparison."
        );
        let raw_now: String = conn
            .query_row(
                "SELECT value FROM metadata WHERE key = ?1",
                params![ARS_PARAMETER_POLICY_METADATA_KEY],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(raw_now, future_value);
    }

    /// R15 P2 (2026-05-04) — older or hand-edited schemas (e.g.,
    /// `schema_version: 0`) MUST be classified `Corrupt`, NOT
    /// `UnsupportedSchema`. Pre-R15 the peek check used `!=`, so a
    /// schema=0 row was wrongly preserved as future-schema and the
    /// recovery path stalled permanently. After R15 the peek uses
    /// `>` (strictly greater) and the typed-deserialize `Ok(_)` arm
    /// classifies older schemas as `Corrupt` so doctor --fix can
    /// recover them.
    #[test]
    fn parameter_policy_load_older_schema_version_is_corrupt_not_unsupported() {
        let conn = conn();
        // Plant a row with schema_version=0 — older than this
        // binary's schema=1. The JSON deserializes cleanly; the
        // schema-version branch must classify as Corrupt.
        let older_value = serde_json::json!({
            "schema_version": 0,
            "revision": 3,
            "mode": "shadow",
            "source_adaptive_version": 5,
            "runtime_adoption_weight": 0.0,
            "adoption_weights": {},
            "last_event_id": 7,
            "last_updated": "2026-04-01T00:00:00Z",
        })
        .to_string();
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES (?1, ?2)",
            params![ARS_PARAMETER_POLICY_METADATA_KEY, &older_value],
        )
        .unwrap();
        let loaded = load_parameter_policy(&conn);
        assert_eq!(
            loaded.status,
            ArsParameterPolicyLoadStatus::Corrupt,
            "older schema_version (0 < current=1) must be classified \
             Corrupt so `doctor --fix` can recover it. Pre-R15 fix the \
             peek check used `!=`, classifying older schemas as \
             UnsupportedSchema, which doctor refused to delete and \
             refresh skipped — refresh stalled forever."
        );
        // Recovery via repair helper still deletes it (the row is
        // genuinely corrupt for our purposes — schema_version older
        // than what this binary writes).
        let outcome = repair_corrupt_parameter_policy(&conn).unwrap();
        assert_eq!(outcome.deleted, 1);
        assert_eq!(
            outcome.observed_status,
            ArsParameterPolicyLoadStatus::Corrupt
        );
        // Subsequent load reports Missing — refresh path can now INSERT
        // a fresh row at current schema.
        let after = load_parameter_policy(&conn);
        assert_eq!(after.status, ArsParameterPolicyLoadStatus::Missing);
    }

    /// R10 P3 (2026-05-04) — `repair_corrupt_parameter_policy` must
    /// re-check the row's status under the write lock and decline to
    /// delete a row that has been repaired between an earlier read
    /// and this recovery call. Without the in-transaction re-check,
    /// a peer `refresh_ars_parameter_policy` tick or a concurrent
    /// `doctor --fix` could rewrite the row to a healthy canary in
    /// the gap, and the doctor would then destroy that newly-valid
    /// state.
    #[test]
    fn repair_corrupt_parameter_policy_skips_when_peer_already_repaired() {
        let conn = conn();
        // Plant a healthy canary policy directly. The repair path must
        // NOT touch this row even though a hypothetical earlier
        // `apply_local_fixes` read might have observed it as Corrupt
        // before a peer fixed it.
        assert!(save_parameter_policy_cas(&conn, &canary_policy(3), 0).unwrap());

        let outcome = repair_corrupt_parameter_policy(&conn).unwrap();
        assert_eq!(outcome.deleted, 0);
        assert_eq!(
            outcome.observed_status,
            ArsParameterPolicyLoadStatus::Loaded
        );
        assert!(outcome.error_at_delete.is_none());

        // Sanity — the canary policy is intact.
        let loaded = load_parameter_policy(&conn);
        assert_eq!(loaded.status, ArsParameterPolicyLoadStatus::Loaded);
        assert_eq!(loaded.policy.revision, 3);
        assert_eq!(loaded.policy.mode, ArsParameterPolicyMode::Canary);
    }

    /// R10 P3 (2026-05-04) — when the row is genuinely Corrupt at
    /// recovery time, `repair_corrupt_parameter_policy` deletes it
    /// and reports the error message captured INSIDE the transaction.
    #[test]
    fn repair_corrupt_parameter_policy_deletes_when_still_corrupt() {
        let conn = conn();
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES (?1, ?2)",
            params![ARS_PARAMETER_POLICY_METADATA_KEY, "{not json"],
        )
        .unwrap();

        let outcome = repair_corrupt_parameter_policy(&conn).unwrap();
        assert_eq!(outcome.deleted, 1);
        assert_eq!(
            outcome.observed_status,
            ArsParameterPolicyLoadStatus::Corrupt
        );
        assert!(outcome.error_at_delete.is_some());

        let loaded = load_parameter_policy(&conn);
        assert_eq!(loaded.status, ArsParameterPolicyLoadStatus::Missing);
    }

    /// R10 P3 (2026-05-04) — `repair_corrupt_parameter_policy` must NOT
    /// destroy a future-schema row even when called during a window
    /// the doctor's earlier read might have classified differently.
    /// The R8 P2 #1 fix made future-schema rows surface as
    /// `UnsupportedSchema` (not `Corrupt`) at the load layer, but the
    /// doctor's apply_local_fixes still passes ALL eligible rows
    /// through this helper. Belt-and-suspenders: the helper itself
    /// must observe the status under the write lock and decline.
    #[test]
    fn repair_corrupt_parameter_policy_preserves_future_schema_row() {
        let conn = conn();
        let future_value = serde_json::json!({
            "schema_version": 9999,
            "revision": 4,
            "mode": "canary",
            "source_adaptive_version": 0,
            "runtime_adoption_weight": 0.5,
            "adoption_weights": {},
            "last_event_id": 0,
            "last_updated": "2030-01-01T00:00:00Z",
        })
        .to_string();
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES (?1, ?2)",
            params![ARS_PARAMETER_POLICY_METADATA_KEY, &future_value],
        )
        .unwrap();

        let outcome = repair_corrupt_parameter_policy(&conn).unwrap();
        assert_eq!(outcome.deleted, 0);
        assert_eq!(
            outcome.observed_status,
            ArsParameterPolicyLoadStatus::UnsupportedSchema,
            "future-schema row must surface as UnsupportedSchema and \
             the helper must decline to delete it"
        );

        let raw_now: String = conn
            .query_row(
                "SELECT value FROM metadata WHERE key = ?1",
                params![ARS_PARAMETER_POLICY_METADATA_KEY],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(raw_now, future_value);
    }

    #[test]
    fn canary_policy_requires_positive_runtime_adoption_weight() {
        let mut policy = canary_policy(1);
        policy.runtime_adoption_weight = 0.0;
        assert_eq!(policy.runtime_adoption_weight(7), 0.0);
        assert!(!policy.allows_runtime_adoption(7));

        policy.runtime_adoption_weight = 0.25;
        assert_eq!(policy.runtime_adoption_weight(7), 0.25);
        assert!(policy.allows_runtime_adoption(7));
    }

    #[test]
    fn canary_policy_uses_scoped_runtime_adoption_weights_with_global_fallback() {
        let mut policy = canary_policy(1);
        policy.runtime_adoption_weight = 0.25;
        policy
            .adoption_weights
            .insert("recall_fusion:semantic".to_string(), 0.40);
        policy
            .adoption_weights
            .insert("recall_fusion:semantic:7".to_string(), 0.65);

        assert_eq!(
            policy.runtime_adoption_weight_for(7, "recall_fusion:semantic"),
            0.40
        );
        assert_eq!(
            policy.runtime_adoption_weight_for(7, "recall_fusion:semantic:7"),
            0.65
        );
        assert_eq!(
            policy.runtime_adoption_weight_for(7, "concept_summary_gate"),
            0.25
        );
        assert_eq!(
            policy.runtime_adoption_weight_for(6, "recall_fusion:semantic"),
            0.0
        );
    }

    #[test]
    fn parameter_policy_load_clamps_runtime_adoption_weight() {
        let conn = conn();
        let mut policy = canary_policy(1);
        policy.runtime_adoption_weight = 2.5;
        assert!(save_parameter_policy_cas(&conn, &policy, 0).unwrap());

        let loaded = load_parameter_policy(&conn);
        assert_eq!(loaded.policy.runtime_adoption_weight, 1.0);

        let mut policy = loaded.policy;
        policy.revision += 1;
        policy.runtime_adoption_weight = f64::NAN;
        policy
            .adoption_weights
            .insert("judge_sample_rate".to_string(), 2.0);
        policy
            .adoption_weights
            .insert("llm_feedback_decay".to_string(), f64::NAN);
        assert!(save_parameter_policy_cas(&conn, &policy, 1).unwrap());
        let loaded = load_parameter_policy(&conn).policy;
        assert_eq!(loaded.runtime_adoption_weight, 0.0);
        assert_eq!(loaded.adoption_weights["judge_sample_rate"], 1.0);
        assert_eq!(loaded.adoption_weights["llm_feedback_decay"], 0.0);
    }

    #[test]
    fn parameter_policy_rejects_invalid_recall_fusion_evidence_on_save() {
        let conn = conn();
        let mut policy = canary_policy(1);
        policy.recall_fusion_evidence.insert(
            "judge_sample_rate".to_string(),
            ArsRecallFusionEvidence {
                basis: ArsRecallFusionEvidenceBasis::Human,
                human_ess: 1,
                automatic_candidate_present: false,
                human_simplex: Some(crate::store::a12_calibration::A12FusionSimplex::default()),
                human_runtime_adoption_weight: Some(0.05),
                resolved_simplex: crate::store::a12_calibration::A12FusionSimplex::default(),
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
                recall_gate_status: ArsRecallGateStatus::NoData,
                recall_gate_build_fingerprint: None,
                recall_gate_fixture_fingerprint: None,
                recall_gate_evaluated_at: None,
                calibrated_at: None,
                evaluated_at: None,
                a12_valid_until_exclusive: None,
                reason: "human feedback".to_string(),
            },
        );

        assert!(save_parameter_policy_cas(&conn, &policy, 0).is_err());
        assert_eq!(
            load_parameter_policy(&conn).status,
            ArsParameterPolicyLoadStatus::Missing
        );
    }

    #[test]
    fn parameter_policy_invalid_current_evidence_loads_fail_closed() {
        let conn = conn();
        let invalid = serde_json::json!({
            "schema_version": ARS_PARAMETER_POLICY_SCHEMA_VERSION,
            "revision": 1,
            "mode": "canary",
            "source_adaptive_version": 7,
            "runtime_adoption_weight": 0.0,
            "adoption_weights": {"recall_fusion:global": 0.05},
            "recall_fusion_evidence": {
                "recall_fusion:global": {
                    "basis": "self_supervised",
                    "resolved_simplex": {
                        "bm25": 0.5,
                        "vector": 0.5,
                        "kg": 0.5,
                        "episode": 0.0,
                        "support": 0.0,
                        "diversity": 0.0
                    },
                    "self_supervised_train_family_ess": 10,
                    "self_supervised_holdout_family_ess": 4,
                    "a12_generation": 1,
                    "a12_revision": 1,
                    "generation_fingerprint": "generation",
                    "corpus_fingerprint": "corpus",
                    "optimizer_fingerprint": "optimizer",
                    "evaluation_fingerprint": "evaluation",
                    "a12_verdict": "ship",
                    "a12_noise_floor": 0.02,
                    "recall_gate_status": "ship",
                    "recall_gate_build_fingerprint": "build",
                    "recall_gate_fixture_fingerprint": "fixture",
                    "calibrated_at": 100,
                    "evaluated_at": 101
                }
            },
            "last_event_id": 0,
            "last_updated": "2026-07-13T00:00:00Z"
        })
        .to_string();
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES (?1, ?2)",
            params![ARS_PARAMETER_POLICY_METADATA_KEY, invalid],
        )
        .unwrap();

        let loaded = load_parameter_policy(&conn);
        assert_eq!(loaded.status, ArsParameterPolicyLoadStatus::Corrupt);
        assert_eq!(loaded.policy.runtime_adoption_weight(7), 0.0);
        assert!(loaded.error.unwrap().contains("simplex must sum to 1"));
    }

    #[test]
    fn parameter_policy_rejects_invalid_sealed_human_fallback_fields() {
        let conn = conn();
        let invalid = serde_json::json!({
            "schema_version": ARS_PARAMETER_POLICY_SCHEMA_VERSION,
            "revision": 1,
            "mode": "canary",
            "source_adaptive_version": 7,
            "runtime_adoption_weight": 0.0,
            "adoption_weights": {},
            "recall_fusion_evidence": {
                "recall_fusion:semantic:7": {
                    "basis": "human",
                    "resolved_simplex": {
                        "bm25": 0.4,
                        "vector": 0.3,
                        "kg": 0.1,
                        "episode": 0.1,
                        "support": 0.05,
                        "diversity": 0.05
                    },
                    "human_ess": 40,
                    "automatic_candidate_present": true,
                    "human_simplex": {
                        "bm25": 0.4,
                        "vector": 0.3,
                        "kg": 0.1,
                        "episode": 0.1,
                        "support": 0.05,
                        "diversity": 0.05
                    },
                    "human_runtime_adoption_weight": 2.0,
                    "self_supervised_train_family_ess": 80,
                    "self_supervised_holdout_family_ess": 20,
                    "a12_generation": 7,
                    "a12_revision": 9,
                    "generation_fingerprint": "generation",
                    "corpus_fingerprint": "corpus",
                    "optimizer_fingerprint": "optimizer",
                    "evaluation_fingerprint": "evaluation",
                    "a12_verdict": "bail",
                    "a12_noise_floor": 0.02,
                    "recall_gate_status": "no_data",
                    "calibrated_at": 100,
                    "evaluated_at": 101
                }
            },
            "last_event_id": 0,
            "last_updated": "2026-07-13T00:00:00Z"
        })
        .to_string();
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES (?1, ?2)",
            params![ARS_PARAMETER_POLICY_METADATA_KEY, invalid],
        )
        .unwrap();

        let loaded = load_parameter_policy(&conn);
        assert_eq!(loaded.status, ArsParameterPolicyLoadStatus::Corrupt);
        assert!(loaded
            .error
            .unwrap()
            .contains("human runtime adoption weight"));
    }

    #[test]
    fn policy_rollback_disables_canary_without_erasing_learned_shadow_fusion() {
        let conn = conn();
        let mut state = AdaptiveState {
            version: 7,
            ..AdaptiveState::default()
        };
        let weights = ShadowFusionWeights::default();
        state.learned_shadow_fusion.insert(
            "global".to_string(),
            LearnedShadowFusionEntry {
                weights: ShadowFusionWeightEntry {
                    bm25: weights.bm25,
                    vec: weights.vec,
                    kg: weights.kg,
                    episode: weights.episode,
                    support: weights.support,
                    diversity: weights.diversity,
                },
                sample_count: 20,
                last_updated: "2026-05-01T00:00:00Z".to_string(),
            },
        );
        state.save_snapshot(&conn).unwrap();
        assert!(save_parameter_policy_cas(&conn, &canary_policy(1), 0).unwrap());

        delete_parameter_policy(&conn).unwrap();

        let policy = load_parameter_policy(&conn);
        let restored = AdaptiveState::restore_snapshot(&conn).unwrap();
        assert_eq!(policy.policy.mode, ArsParameterPolicyMode::Disabled);
        assert!(restored.learned_shadow_fusion.contains_key("global"));
    }

    /// v0.28.7+ audit L6 — `clamp_policy_weights` warns when the
    /// `adoption_weights` map exceeds the defense-in-depth cap, but
    /// MUST NOT silently drop entries (per advisor guidance: a 0.0
    /// weight from a deliberate canary→shadow rollback is operationally
    /// valuable, so heuristic eviction was rejected). The realistic
    /// post-L6 ceiling is `LEARNED_SHADOW_FUSION_CAP + 5 globals = 4101`
    /// per pipeline tick; the cap is set at `4128` for headroom. This
    /// test forces the over-cap branch by hand-building a degenerate
    /// policy and asserts the entry count is unchanged after clamping.
    #[test]
    fn clamp_policy_weights_warns_above_cap_but_does_not_drop_entries() {
        let mut policy = canary_policy(1);
        // Fill above the defense-in-depth cap. Use a value that will
        // still pass `clamp01` so we test the cap path independently.
        for i in 0..(ADOPTION_WEIGHTS_CAP + 32) {
            policy.adoption_weights.insert(format!("scope:{i}"), 0.5);
        }
        let pre_len = policy.adoption_weights.len();
        assert!(pre_len > ADOPTION_WEIGHTS_CAP);

        clamp_policy_weights(&mut policy);

        // No silent drops — the warn is the signal, the data is preserved.
        assert_eq!(
            policy.adoption_weights.len(),
            pre_len,
            "clamp_policy_weights MUST NOT drop entries above cap; the warn \
             is the operator-visible signal but every adoption_weights key \
             maps to a scope identifier the runtime trust gate consults — \
             dropping would silently mute a canary scope"
        );
        // Sanity: clamp01 still applied to values.
        for value in policy.adoption_weights.values() {
            assert!((0.0..=1.0).contains(value));
        }
    }

    #[test]
    fn validate_rejects_non_finite_adoption_weights() {
        let mut policy = canary_policy(1);
        policy
            .adoption_weights
            .insert("recall_fusion:global".to_string(), f64::NAN);
        assert!(validate_parameter_policy(&policy).is_err());

        let mut policy = canary_policy(1);
        policy
            .adoption_weights
            .insert("recall_fusion:global".to_string(), f64::INFINITY);
        assert!(validate_parameter_policy(&policy).is_err());

        let mut policy = canary_policy(1);
        policy.runtime_adoption_weight = f64::NAN;
        assert!(validate_parameter_policy(&policy).is_err());
    }

    #[test]
    fn validate_rejects_out_of_range_adoption_weights() {
        let mut policy = canary_policy(1);
        policy
            .adoption_weights
            .insert("recall_fusion:global".to_string(), 1.5);
        assert!(validate_parameter_policy(&policy).is_err());

        let mut policy = canary_policy(1);
        policy
            .adoption_weights
            .insert("recall_fusion:global".to_string(), -0.25);
        assert!(validate_parameter_policy(&policy).is_err());
    }

    /// JSON cannot encode NaN, but `1e999` parses to +inf; a tampered or
    /// bit-rotted row must load fail-closed as Corrupt (disabled policy →
    /// static resolution), never silently sanitized to a healthy load.
    #[test]
    fn infinite_adoption_weight_row_loads_fail_closed_as_corrupt() {
        let conn = conn();
        let mut policy = canary_policy(1);
        policy
            .adoption_weights
            .insert("recall_fusion:global".to_string(), 0.4375);
        assert!(save_parameter_policy_cas(&conn, &policy, 0).unwrap());
        let raw: String = conn
            .query_row(
                "SELECT value FROM metadata WHERE key = ?1",
                params![ARS_PARAMETER_POLICY_METADATA_KEY],
                |row| row.get(0),
            )
            .unwrap();
        let tampered = raw.replace("0.4375", "1e999");
        assert_ne!(raw, tampered, "fixture weight must appear exactly once");
        conn.execute(
            "UPDATE metadata SET value = ?1 WHERE key = ?2",
            params![tampered, ARS_PARAMETER_POLICY_METADATA_KEY],
        )
        .unwrap();

        let loaded = load_parameter_policy(&conn);

        assert_eq!(loaded.status, ArsParameterPolicyLoadStatus::Corrupt);
        assert_eq!(loaded.policy.mode, ArsParameterPolicyMode::Disabled);
        assert_eq!(loaded.policy.runtime_adoption_weight, 0.0);
        assert!(loaded.error.is_some());
    }

    #[test]
    fn out_of_range_adoption_weight_row_loads_fail_closed_as_corrupt() {
        let conn = conn();
        let mut policy = canary_policy(1);
        policy
            .adoption_weights
            .insert("recall_fusion:global".to_string(), 0.4375);
        assert!(save_parameter_policy_cas(&conn, &policy, 0).unwrap());
        let raw: String = conn
            .query_row(
                "SELECT value FROM metadata WHERE key = ?1",
                params![ARS_PARAMETER_POLICY_METADATA_KEY],
                |row| row.get(0),
            )
            .unwrap();
        let tampered = raw.replace("0.4375", "1.5");
        conn.execute(
            "UPDATE metadata SET value = ?1 WHERE key = ?2",
            params![tampered, ARS_PARAMETER_POLICY_METADATA_KEY],
        )
        .unwrap();

        let loaded = load_parameter_policy(&conn);

        assert_eq!(loaded.status, ArsParameterPolicyLoadStatus::Corrupt);
        assert_eq!(loaded.policy.mode, ArsParameterPolicyMode::Disabled);
    }
}
