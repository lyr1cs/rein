//! Versioned, fail-closed policy state for destructive lexical dedup.
//!
//! Unlabeled adaptive statistics never enter this row. A policy may reach a
//! statistical `Ship` verdict and expose a candidate, but it cannot change the
//! destructive hard threshold until a representative production score-space
//! cohort is implemented. Runtime therefore remains static-only.

use rusqlite::{params, OptionalExtension};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const DEDUP_CALIBRATION_METADATA_KEY: &str = "dedup_calibration_policy";
pub const DEDUP_CALIBRATION_SEAL_METADATA_KEY: &str = "dedup_calibration_seal";
pub const DEDUP_CALIBRATION_SCHEMA_VERSION: u32 = 1;
const FALSE_POSITIVE_BUDGET: f64 = 0.02;
const MIN_ZERO_FP_SEALED_NEGATIVES: usize = 149;
pub const DEDUP_CALIBRATION_REQUIRED_SLICES: [&str; 3] = [
    "operator_distinct",
    "canonical_family",
    "structural_challenge",
];

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DedupCalibrationStatus {
    Ship,
    Bail,
    #[default]
    NoData,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DedupCalibrationScale {
    #[default]
    Lexical,
    VectorCosine,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DedupCalibrationProvenance {
    #[default]
    DiscoveryOnly,
    StructuralAnchors,
    OperatorLabels,
    Mixed,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DedupUtilityEvidence {
    pub n: usize,
    pub baseline_only_hits: usize,
    pub candidate_only_hits: usize,
    pub ci_lower: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub miss_rate_upper_95: Option<f64>,
    pub status: DedupCalibrationStatus,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DedupCalibrationSlice {
    pub name: String,
    pub positive_count: usize,
    pub negative_count: usize,
    pub status: DedupCalibrationStatus,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DedupCalibrationConfusion {
    pub true_positives: usize,
    pub false_positives: usize,
    pub true_negatives: usize,
    pub false_negatives: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DedupCalibrationPolicy {
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    #[serde(default)]
    pub revision: u64,
    #[serde(default)]
    pub status: DedupCalibrationStatus,
    #[serde(default)]
    pub scale: DedupCalibrationScale,
    #[serde(default)]
    pub configured_static_threshold: f32,
    #[serde(default)]
    pub candidate_threshold: f32,
    #[serde(default)]
    pub shadow_threshold: f32,
    #[serde(default)]
    pub effective_hard_threshold: f32,
    #[serde(default)]
    pub train_positive_count: usize,
    #[serde(default)]
    pub train_negative_count: usize,
    #[serde(default)]
    pub sealed_positive_count: usize,
    #[serde(default)]
    pub sealed_negative_count: usize,
    /// Supplemental exact-content positive challenges. They do not contribute
    /// to the powered canonical-family utility ESS.
    #[serde(default)]
    pub sealed_exact_positive_count: usize,
    /// Supplemental structural distinct challenges. They do not contribute to
    /// the operator-labeled false-positive ESS.
    #[serde(default)]
    pub sealed_structural_negative_count: usize,
    #[serde(default)]
    pub false_positive_count: usize,
    /// Whether holdout outcomes have been revealed. Underpowered `NoData`
    /// evaluations remain ephemeral and mask outcomes until the precommitted
    /// power and slice requirements are complete.
    #[serde(default)]
    pub holdout_revealed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub false_positive_upper_95: Option<f64>,
    #[serde(default)]
    pub utility: DedupUtilityEvidence,
    #[serde(default)]
    pub holdout_confusion: DedupCalibrationConfusion,
    #[serde(default)]
    pub required_slices: Vec<DedupCalibrationSlice>,
    #[serde(default)]
    pub rejected_case_count: usize,
    #[serde(default = "default_selector_version")]
    pub selector_version: u32,
    #[serde(default)]
    pub sealed_generation: u64,
    #[serde(default)]
    pub sealed_cutoff: i64,
    #[serde(default)]
    pub train_fingerprint: String,
    #[serde(default)]
    pub holdout_fingerprint: String,
    #[serde(default)]
    pub corpus_fingerprint: String,
    #[serde(default)]
    pub provenance: DedupCalibrationProvenance,
    #[serde(default)]
    pub calibrated_at: i64,
    #[serde(default)]
    pub valid_until: i64,
}

/// Independently persisted immutable-corpus manifest. Runtime activation
/// requires this row and the policy row to match exactly; a half-write, stale
/// policy, or edited fingerprint fails closed to the configured static value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DedupCalibrationSeal {
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    #[serde(default)]
    pub revision: u64,
    #[serde(default)]
    pub generation: u64,
    #[serde(default)]
    pub cutoff: i64,
    #[serde(default)]
    pub scale: DedupCalibrationScale,
    #[serde(default)]
    pub configured_static_threshold_bits: u32,
    #[serde(default)]
    pub train_fingerprint: String,
    #[serde(default)]
    pub holdout_fingerprint: String,
    #[serde(default)]
    pub corpus_fingerprint: String,
    /// SHA-256 over the canonical serialization of the complete policy. The
    /// copied corpus fields above are useful for diagnostics; this digest is
    /// the trust binding that catches mutation of every policy field.
    #[serde(default)]
    pub policy_digest: String,
    #[serde(default)]
    pub calibrated_at: i64,
    #[serde(default)]
    pub valid_until: i64,
}

impl Default for DedupCalibrationPolicy {
    fn default() -> Self {
        Self {
            schema_version: DEDUP_CALIBRATION_SCHEMA_VERSION,
            revision: 0,
            status: DedupCalibrationStatus::NoData,
            scale: DedupCalibrationScale::Lexical,
            configured_static_threshold: 0.0,
            candidate_threshold: 0.0,
            shadow_threshold: 0.0,
            effective_hard_threshold: 0.0,
            train_positive_count: 0,
            train_negative_count: 0,
            sealed_positive_count: 0,
            sealed_negative_count: 0,
            sealed_exact_positive_count: 0,
            sealed_structural_negative_count: 0,
            false_positive_count: 0,
            holdout_revealed: false,
            false_positive_upper_95: None,
            utility: DedupUtilityEvidence::default(),
            holdout_confusion: DedupCalibrationConfusion::default(),
            required_slices: Vec::new(),
            rejected_case_count: 0,
            selector_version: default_selector_version(),
            sealed_generation: 0,
            sealed_cutoff: 0,
            train_fingerprint: String::new(),
            holdout_fingerprint: String::new(),
            corpus_fingerprint: String::new(),
            provenance: DedupCalibrationProvenance::DiscoveryOnly,
            calibrated_at: 0,
            valid_until: 0,
        }
    }
}

fn default_schema_version() -> u32 {
    DEDUP_CALIBRATION_SCHEMA_VERSION
}

fn default_selector_version() -> u32 {
    1
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DedupCalibrationLoadStatus {
    Missing,
    Loaded,
    Corrupt,
    UnsupportedSchema,
    Stale,
    FingerprintMismatch,
    StorageError,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DedupCalibrationLoad {
    pub policy: DedupCalibrationPolicy,
    pub status: DedupCalibrationLoadStatus,
    /// True only when the policy and independently persisted sealed-corpus
    /// manifest matched through [`load_dedup_calibration_for_runtime`].
    context_verified: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl DedupCalibrationLoad {
    fn unhealthy(status: DedupCalibrationLoadStatus, error: impl Into<Option<String>>) -> Self {
        Self {
            policy: DedupCalibrationPolicy::default(),
            status,
            context_verified: false,
            error: error.into(),
        }
    }

    pub fn context_verified(&self) -> bool {
        self.context_verified
    }
}

fn threshold_is_valid(value: f32) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

pub(crate) fn canonical_policy_digest(policy: &DedupCalibrationPolicy) -> Result<String, String> {
    // DedupCalibrationPolicy is a struct with no maps, so serde's declared
    // field order is a deterministic canonical byte representation. Keep the
    // domain tag so this digest can never be confused with a corpus hash.
    let raw = serde_json::to_vec(policy)
        .map_err(|error| format!("failed to serialize canonical dedup policy: {error}"))?;
    let mut hasher = Sha256::new();
    hasher.update(b"rein-dedup-calibration-policy-v1\0");
    hasher.update((raw.len() as u64).to_be_bytes());
    hasher.update(raw);
    Ok(format!("{:x}", hasher.finalize()))
}

fn paired_wald_ci_lower(n: usize, baseline_only: usize, candidate_only: usize) -> Option<f64> {
    if n == 0 || baseline_only.saturating_add(candidate_only) > n {
        return None;
    }
    let n = n as f64;
    let b = baseline_only as f64;
    let c = candidate_only as f64;
    let diff = (c - b) / n;
    let raw_variance = (b + c) - (b - c).powi(2) / n;
    let standard_error = (raw_variance.max(0.0) / (n * n)).sqrt();
    Some(diff - 1.96 * standard_error)
}

fn validate_policy(policy: &DedupCalibrationPolicy) -> Result<(), String> {
    if policy.schema_version != DEDUP_CALIBRATION_SCHEMA_VERSION {
        return Err(format!(
            "schema_version={} does not match current {}",
            policy.schema_version, DEDUP_CALIBRATION_SCHEMA_VERSION
        ));
    }
    for (name, value) in [
        (
            "configured_static_threshold",
            policy.configured_static_threshold,
        ),
        ("candidate_threshold", policy.candidate_threshold),
        ("shadow_threshold", policy.shadow_threshold),
        ("effective_hard_threshold", policy.effective_hard_threshold),
    ] {
        if !threshold_is_valid(value) {
            return Err(format!("{name} must be finite and in [0,1], got {value}"));
        }
    }
    if policy.effective_hard_threshold.to_bits() != policy.configured_static_threshold.to_bits() {
        return Err(
            "effective hard threshold must remain equal to static until production-cohort calibration exists"
                .to_string(),
        );
    }
    if policy.false_positive_count > policy.sealed_negative_count {
        return Err("false_positive_count exceeds sealed_negative_count".to_string());
    }
    if policy.holdout_revealed
        && (policy.holdout_confusion.false_positives != policy.false_positive_count
            || policy
                .holdout_confusion
                .true_positives
                .saturating_add(policy.holdout_confusion.false_negatives)
                != policy.sealed_positive_count
            || policy
                .holdout_confusion
                .true_negatives
                .saturating_add(policy.holdout_confusion.false_positives)
                != policy.sealed_negative_count)
    {
        return Err("revealed holdout confusion matrix does not match sealed counts".to_string());
    }
    if !policy.utility.ci_lower.is_finite() {
        return Err("utility ci_lower must be finite".to_string());
    }
    if let Some(upper) = policy.utility.miss_rate_upper_95 {
        if !upper.is_finite() || !(0.0..=1.0).contains(&upper) {
            return Err("utility miss_rate_upper_95 must be finite and in [0,1]".to_string());
        }
    }
    if let Some(upper) = policy.false_positive_upper_95 {
        if !upper.is_finite() || !(0.0..=1.0).contains(&upper) {
            return Err("false_positive_upper_95 must be finite and in [0,1]".to_string());
        }
    }

    if policy.status == DedupCalibrationStatus::Ship {
        if !policy.holdout_revealed {
            return Err("Ship policy requires a single revealed powered holdout".into());
        }
        if policy.selector_version != default_selector_version()
            || policy.rejected_case_count > 0
            || policy.sealed_generation == 0
            || policy.sealed_cutoff <= 0
        {
            return Err("Ship policy requires current selector and zero rejected cases".into());
        }
        if policy.candidate_threshold < policy.configured_static_threshold {
            return Err("Ship policy candidate must remain at or above configured static".into());
        }
        if policy.train_fingerprint.is_empty()
            || policy.holdout_fingerprint.is_empty()
            || policy.corpus_fingerprint.is_empty()
        {
            return Err("Ship policy requires non-empty train/holdout/corpus fingerprints".into());
        }
        if policy.calibrated_at <= 0 || policy.valid_until <= policy.calibrated_at {
            return Err("Ship policy requires a bounded positive freshness window".into());
        }
        let upper = policy
            .false_positive_upper_95
            .ok_or_else(|| "Ship policy requires a false-positive upper bound".to_string())?;
        let recomputed_upper = crate::eval::mcnemar::one_sided_binomial_upper_bound(
            u32::try_from(policy.false_positive_count)
                .map_err(|_| "false-positive count exceeds exact-bound range")?,
            u32::try_from(policy.sealed_negative_count)
                .map_err(|_| "sealed-negative count exceeds exact-bound range")?,
            0.05,
        )
        .ok_or_else(|| "Ship policy false-positive bound cannot be recomputed".to_string())?;
        if policy.sealed_negative_count < MIN_ZERO_FP_SEALED_NEGATIVES
            || upper > FALSE_POSITIVE_BUDGET
            || (upper - recomputed_upper).abs() > 1.0e-12
        {
            return Err("Ship policy lacks powered sealed-negative safety".into());
        }
        let miss_upper = policy
            .utility
            .miss_rate_upper_95
            .ok_or_else(|| "Ship policy requires an exact positive-miss upper bound".to_string())?;
        let recomputed_miss_upper = crate::eval::mcnemar::one_sided_binomial_upper_bound(
            u32::try_from(policy.utility.baseline_only_hits)
                .map_err(|_| "utility miss count exceeds exact-bound range")?,
            u32::try_from(policy.utility.n)
                .map_err(|_| "utility count exceeds exact-bound range")?,
            0.05,
        )
        .ok_or_else(|| "Ship policy utility bound cannot be recomputed".to_string())?;
        let recomputed_ci_lower = paired_wald_ci_lower(
            policy.utility.n,
            policy.utility.baseline_only_hits,
            policy.utility.candidate_only_hits,
        )
        .ok_or_else(|| "Ship policy utility counts are inconsistent".to_string())?;
        if policy.utility.n != policy.sealed_positive_count
            || policy.sealed_positive_count < MIN_ZERO_FP_SEALED_NEGATIVES
            || policy.utility.candidate_only_hits != 0
            || policy.utility.status != DedupCalibrationStatus::Ship
            || miss_upper > FALSE_POSITIVE_BUDGET
            || (miss_upper - recomputed_miss_upper).abs() > 1.0e-12
            || (policy.utility.ci_lower - recomputed_ci_lower).abs() > 1.0e-12
        {
            return Err("Ship policy lacks positive utility non-inferiority evidence".into());
        }
        let mut slice_names: Vec<&str> = policy
            .required_slices
            .iter()
            .map(|slice| slice.name.as_str())
            .collect();
        slice_names.sort_unstable();
        let mut expected_names = DEDUP_CALIBRATION_REQUIRED_SLICES.to_vec();
        expected_names.sort_unstable();
        let slice_has_required_semantics = |slice: &DedupCalibrationSlice| match slice.name.as_str()
        {
            "operator_distinct" => slice.negative_count > 0,
            "canonical_family" => slice.positive_count > 0,
            "structural_challenge" => slice.positive_count > 0 && slice.negative_count > 0,
            _ => false,
        };
        if slice_names != expected_names
            || policy.required_slices.iter().any(|slice| {
                slice.name.is_empty()
                    || slice.status != DedupCalibrationStatus::Ship
                    || !slice_has_required_semantics(slice)
            })
        {
            return Err("Ship policy requires populated passing slices".into());
        }
        if !matches!(
            policy.provenance,
            DedupCalibrationProvenance::OperatorLabels | DedupCalibrationProvenance::Mixed
        ) {
            return Err("Ship requires operator-labeled distinct evidence".into());
        }
        if policy.sealed_exact_positive_count == 0 || policy.sealed_structural_negative_count == 0 {
            return Err("Ship requires populated exact and structural challenge evidence".into());
        }
    }
    Ok(())
}

fn validate_seal(seal: &DedupCalibrationSeal) -> Result<(), String> {
    if seal.schema_version != DEDUP_CALIBRATION_SCHEMA_VERSION {
        return Err("seal schema version mismatch".into());
    }
    if seal.revision == 0
        || seal.generation == 0
        || seal.cutoff <= 0
        || seal.calibrated_at <= 0
        || seal.valid_until <= seal.calibrated_at
        || seal.train_fingerprint.is_empty()
        || seal.holdout_fingerprint.is_empty()
        || seal.corpus_fingerprint.is_empty()
        || seal.policy_digest.is_empty()
    {
        return Err(
            "seal is missing revision, generation, cutoff, freshness, or fingerprints".into(),
        );
    }
    let static_threshold = f32::from_bits(seal.configured_static_threshold_bits);
    if !threshold_is_valid(static_threshold) {
        return Err("seal static threshold bits are invalid".into());
    }
    Ok(())
}

fn validate_bundle_match(
    policy: &DedupCalibrationPolicy,
    seal: &DedupCalibrationSeal,
) -> Result<(), String> {
    if seal.revision != policy.revision
        || seal.generation != policy.sealed_generation
        || seal.cutoff != policy.sealed_cutoff
        || seal.scale != policy.scale
        || seal.configured_static_threshold_bits != policy.configured_static_threshold.to_bits()
        || seal.train_fingerprint != policy.train_fingerprint
        || seal.holdout_fingerprint != policy.holdout_fingerprint
        || seal.corpus_fingerprint != policy.corpus_fingerprint
        || seal.calibrated_at != policy.calibrated_at
        || seal.valid_until != policy.valid_until
        || seal.policy_digest != canonical_policy_digest(policy)?
    {
        return Err("policy does not match independently produced sealed manifest".into());
    }
    Ok(())
}

pub fn load_dedup_calibration(conn: &rusqlite::Connection, now: i64) -> DedupCalibrationLoad {
    let raw = match conn
        .query_row(
            "SELECT value FROM metadata WHERE key = ?1",
            params![DEDUP_CALIBRATION_METADATA_KEY],
            |row| row.get::<_, String>(0),
        )
        .optional()
    {
        Ok(raw) => raw,
        Err(error) => {
            return DedupCalibrationLoad::unhealthy(
                DedupCalibrationLoadStatus::StorageError,
                Some(error.to_string()),
            )
        }
    };
    let Some(raw) = raw else {
        return DedupCalibrationLoad::unhealthy(DedupCalibrationLoadStatus::Missing, None);
    };

    if let Ok(value) = serde_json::from_str::<serde_json::Value>(&raw) {
        if let Some(schema) = value.get("schema_version").and_then(|value| value.as_u64()) {
            if schema > u64::from(DEDUP_CALIBRATION_SCHEMA_VERSION) {
                return DedupCalibrationLoad::unhealthy(
                    DedupCalibrationLoadStatus::UnsupportedSchema,
                    Some(format!(
                        "schema_version={schema} is newer than current {}",
                        DEDUP_CALIBRATION_SCHEMA_VERSION
                    )),
                );
            }
        }
    }

    let policy = match serde_json::from_str::<DedupCalibrationPolicy>(&raw) {
        Ok(policy) => policy,
        Err(error) => {
            return DedupCalibrationLoad::unhealthy(
                DedupCalibrationLoadStatus::Corrupt,
                Some(error.to_string()),
            )
        }
    };
    if let Err(error) = validate_policy(&policy) {
        return DedupCalibrationLoad::unhealthy(DedupCalibrationLoadStatus::Corrupt, Some(error));
    }
    if policy.valid_until > 0 && now > policy.valid_until {
        return DedupCalibrationLoad {
            policy,
            status: DedupCalibrationLoadStatus::Stale,
            context_verified: false,
            error: Some("calibration policy freshness window expired".to_string()),
        };
    }
    DedupCalibrationLoad {
        policy,
        status: DedupCalibrationLoadStatus::Loaded,
        context_verified: false,
        error: None,
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DedupCalibrationSealLoad {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seal: Option<DedupCalibrationSeal>,
    pub status: DedupCalibrationLoadStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

pub fn load_dedup_calibration_seal(
    conn: &rusqlite::Connection,
    now: i64,
) -> DedupCalibrationSealLoad {
    let raw = match conn
        .query_row(
            "SELECT value FROM metadata WHERE key = ?1",
            params![DEDUP_CALIBRATION_SEAL_METADATA_KEY],
            |row| row.get::<_, String>(0),
        )
        .optional()
    {
        Ok(raw) => raw,
        Err(error) => {
            return DedupCalibrationSealLoad {
                seal: None,
                status: DedupCalibrationLoadStatus::StorageError,
                error: Some(error.to_string()),
            }
        }
    };
    let Some(raw) = raw else {
        return DedupCalibrationSealLoad {
            seal: None,
            status: DedupCalibrationLoadStatus::Missing,
            error: None,
        };
    };
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(&raw) {
        if value
            .get("schema_version")
            .and_then(|value| value.as_u64())
            .is_some_and(|schema| schema > u64::from(DEDUP_CALIBRATION_SCHEMA_VERSION))
        {
            return DedupCalibrationSealLoad {
                seal: None,
                status: DedupCalibrationLoadStatus::UnsupportedSchema,
                error: Some("sealed manifest schema is newer than this binary".to_string()),
            };
        }
    }
    let seal = match serde_json::from_str::<DedupCalibrationSeal>(&raw) {
        Ok(seal) => seal,
        Err(error) => {
            return DedupCalibrationSealLoad {
                seal: None,
                status: DedupCalibrationLoadStatus::Corrupt,
                error: Some(error.to_string()),
            }
        }
    };
    if let Err(error) = validate_seal(&seal) {
        return DedupCalibrationSealLoad {
            seal: None,
            status: DedupCalibrationLoadStatus::Corrupt,
            error: Some(error),
        };
    }
    if now > seal.valid_until {
        return DedupCalibrationSealLoad {
            seal: Some(seal),
            status: DedupCalibrationLoadStatus::Stale,
            error: Some("sealed manifest freshness window expired".to_string()),
        };
    }
    DedupCalibrationSealLoad {
        seal: Some(seal),
        status: DedupCalibrationLoadStatus::Loaded,
        error: None,
    }
}

/// Activate context verification only when the policy and a separately
/// persisted immutable-corpus seal agree with the exact runtime static value.
pub fn load_dedup_calibration_for_runtime(
    conn: &rusqlite::Connection,
    now: i64,
    configured_static_threshold: f32,
) -> DedupCalibrationLoad {
    let mut loaded = load_dedup_calibration(conn, now);
    if loaded.status != DedupCalibrationLoadStatus::Loaded {
        return loaded;
    }
    let seal_loaded = load_dedup_calibration_seal(conn, now);
    if seal_loaded.status != DedupCalibrationLoadStatus::Loaded {
        loaded.status = seal_loaded.status;
        loaded.error = seal_loaded.error;
        return loaded;
    }
    let Some(seal) = seal_loaded.seal else {
        loaded.status = DedupCalibrationLoadStatus::FingerprintMismatch;
        loaded.error = Some("sealed manifest disappeared during load".to_string());
        return loaded;
    };
    let policy = &loaded.policy;
    let policy_digest = match canonical_policy_digest(policy) {
        Ok(digest) => digest,
        Err(error) => {
            loaded.status = DedupCalibrationLoadStatus::Corrupt;
            loaded.error = Some(error);
            return loaded;
        }
    };
    let mismatch = seal.revision != policy.revision
        || seal.generation != policy.sealed_generation
        || seal.cutoff != policy.sealed_cutoff
        || seal.scale != policy.scale
        || policy.scale != DedupCalibrationScale::Lexical
        || seal.configured_static_threshold_bits != configured_static_threshold.to_bits()
        || policy.configured_static_threshold.to_bits() != configured_static_threshold.to_bits()
        || seal.train_fingerprint != policy.train_fingerprint
        || seal.holdout_fingerprint != policy.holdout_fingerprint
        || seal.corpus_fingerprint != policy.corpus_fingerprint
        || seal.calibrated_at != policy.calibrated_at
        || seal.valid_until != policy.valid_until
        || seal.policy_digest != policy_digest;
    if mismatch {
        loaded.status = DedupCalibrationLoadStatus::FingerprintMismatch;
        loaded.error = Some("policy and sealed-corpus manifest do not match runtime".to_string());
        return loaded;
    }
    loaded.context_verified = true;
    loaded
}

fn validation_error(error: String) -> rusqlite::Error {
    rusqlite::Error::ToSqlConversionFailure(Box::new(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        error,
    )))
}

#[cfg(test)]
#[must_use = "callers must handle a CAS miss"]
fn save_dedup_calibration_cas(
    conn: &rusqlite::Connection,
    policy: &DedupCalibrationPolicy,
    expected_revision: u64,
) -> rusqlite::Result<bool> {
    validate_policy(policy).map_err(validation_error)?;
    if policy.revision <= expected_revision {
        return Err(validation_error(format!(
            "new revision {} must be greater than expected revision {}",
            policy.revision, expected_revision
        )));
    }
    let raw = serde_json::to_string(policy)
        .map_err(|error| rusqlite::Error::ToSqlConversionFailure(Box::new(error)))?;
    save_metadata_json_cas(
        conn,
        DEDUP_CALIBRATION_METADATA_KEY,
        &raw,
        expected_revision,
        false,
    )
}

fn save_metadata_json_cas(
    conn: &rusqlite::Connection,
    key: &str,
    raw: &str,
    expected_revision: u64,
    allow_missing_at_nonzero_revision: bool,
) -> rusqlite::Result<bool> {
    let updated = conn.execute(
        "UPDATE metadata
            SET value = ?1
          WHERE key = ?2
            AND json_valid(value)
            AND COALESCE(json_extract(value, '$.revision'), 0) = ?3
            AND COALESCE(json_extract(value, '$.schema_version'), ?4) = ?4",
        params![
            raw,
            key,
            expected_revision,
            DEDUP_CALIBRATION_SCHEMA_VERSION,
        ],
    )?;
    if updated == 1 {
        return Ok(true);
    }
    let exists: bool = conn.query_row(
        "SELECT COUNT(*) > 0 FROM metadata WHERE key = ?1",
        params![key],
        |row| row.get(0),
    )?;
    if exists || (expected_revision != 0 && !allow_missing_at_nonzero_revision) {
        return Ok(false);
    }
    match conn.execute(
        "INSERT INTO metadata(key, value) VALUES (?1, ?2)",
        params![key, raw],
    ) {
        Ok(1) => Ok(true),
        Ok(_) => Ok(false),
        Err(rusqlite::Error::SqliteFailure(error, _))
            if error.code == rusqlite::ErrorCode::ConstraintViolation =>
        {
            Ok(false)
        }
        Err(error) => Err(error),
    }
}

/// Atomically persist the immutable-corpus seal and its derived policy. The
/// seal is written first inside `BEGIN IMMEDIATE`; any CAS miss or error rolls
/// both rows back, so runtime never observes a newly active half-bundle.
#[must_use = "callers must handle a bundle CAS miss"]
pub(crate) fn save_dedup_calibration_bundle_cas(
    conn: &rusqlite::Connection,
    policy: &DedupCalibrationPolicy,
    seal: &DedupCalibrationSeal,
    expected_revision: u64,
) -> rusqlite::Result<bool> {
    if !policy.holdout_revealed || policy.status == DedupCalibrationStatus::NoData {
        return Err(validation_error(
            "underpowered or unrevealed calibration is read-only; only powered Ship/Bail terminals may be persisted"
                .to_string(),
        ));
    }
    validate_policy(policy).map_err(validation_error)?;
    validate_seal(seal).map_err(validation_error)?;
    validate_bundle_match(policy, seal).map_err(validation_error)?;
    if policy.revision <= expected_revision {
        return Err(validation_error(format!(
            "new revision {} must be greater than expected revision {}",
            policy.revision, expected_revision
        )));
    }
    let policy_raw = serde_json::to_string(policy)
        .map_err(|error| rusqlite::Error::ToSqlConversionFailure(Box::new(error)))?;
    let seal_raw = serde_json::to_string(seal)
        .map_err(|error| rusqlite::Error::ToSqlConversionFailure(Box::new(error)))?;

    conn.execute_batch("BEGIN IMMEDIATE")?;
    let write_result = (|| {
        if !save_metadata_json_cas(
            conn,
            DEDUP_CALIBRATION_SEAL_METADATA_KEY,
            &seal_raw,
            expected_revision,
            true,
        )? {
            return Ok(false);
        }
        if !save_metadata_json_cas(
            conn,
            DEDUP_CALIBRATION_METADATA_KEY,
            &policy_raw,
            expected_revision,
            true,
        )? {
            return Ok(false);
        }
        Ok(true)
    })();
    match write_result {
        Ok(true) => match conn.execute_batch("COMMIT") {
            Ok(()) => Ok(true),
            Err(error) => {
                let _ = conn.execute_batch("ROLLBACK");
                Err(error)
            }
        },
        Ok(false) => {
            let _ = conn.execute_batch("ROLLBACK");
            Ok(false)
        }
        Err(error) => {
            let _ = conn.execute_batch("ROLLBACK");
            Err(error)
        }
    }
}

/// Resolve the destructive lexical threshold.
///
/// Statistical calibration currently evaluates raw lexical similarity, while
/// the production store path applies topic/cluster bonuses to its final score.
/// Until a representative cohort is evaluated in that exact production score
/// space, promotion is disabled: even a valid `Ship` bundle remains shadow
/// evidence and runtime returns the operator-configured static boundary.
pub(crate) fn resolve_hard_lexical_threshold(
    configured_static_threshold: f32,
    _loaded: &DedupCalibrationLoad,
) -> f32 {
    if !threshold_is_valid(configured_static_threshold) {
        return 1.0;
    }
    configured_static_threshold
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ReinConfig;
    use rusqlite::Connection;

    fn policy(revision: u64) -> DedupCalibrationPolicy {
        DedupCalibrationPolicy {
            revision,
            status: DedupCalibrationStatus::Ship,
            scale: DedupCalibrationScale::Lexical,
            configured_static_threshold: 0.70,
            candidate_threshold: 0.80,
            shadow_threshold: 0.40,
            // Calibration is shadow-only until a representative production
            // score-space cohort exists. Even a statistical Ship verdict must
            // leave the destructive boundary at the operator static value.
            effective_hard_threshold: 0.70,
            train_positive_count: 200,
            train_negative_count: 600,
            sealed_positive_count: 149,
            sealed_negative_count: 149,
            sealed_exact_positive_count: 1,
            sealed_structural_negative_count: 1,
            false_positive_count: 0,
            holdout_revealed: true,
            false_positive_upper_95: crate::eval::mcnemar::one_sided_binomial_upper_bound(
                0, 149, 0.05,
            ),
            utility: DedupUtilityEvidence {
                n: 149,
                baseline_only_hits: 0,
                candidate_only_hits: 0,
                ci_lower: 0.0,
                miss_rate_upper_95: crate::eval::mcnemar::one_sided_binomial_upper_bound(
                    0, 149, 0.05,
                ),
                status: DedupCalibrationStatus::Ship,
            },
            holdout_confusion: DedupCalibrationConfusion {
                true_positives: 149,
                false_positives: 0,
                true_negatives: 149,
                false_negatives: 0,
            },
            required_slices: DEDUP_CALIBRATION_REQUIRED_SLICES
                .iter()
                .map(|name| DedupCalibrationSlice {
                    name: (*name).to_string(),
                    positive_count: 25,
                    negative_count: 75,
                    status: DedupCalibrationStatus::Ship,
                })
                .collect(),
            train_fingerprint: "train-v1".to_string(),
            holdout_fingerprint: "holdout-v1".to_string(),
            corpus_fingerprint: "corpus-v1".to_string(),
            provenance: DedupCalibrationProvenance::OperatorLabels,
            calibrated_at: 1_000,
            valid_until: 2_000,
            sealed_generation: 1,
            sealed_cutoff: 999,
            ..DedupCalibrationPolicy::default()
        }
    }

    fn seal(policy: &DedupCalibrationPolicy) -> DedupCalibrationSeal {
        DedupCalibrationSeal {
            schema_version: DEDUP_CALIBRATION_SCHEMA_VERSION,
            revision: policy.revision,
            generation: policy.sealed_generation,
            cutoff: policy.sealed_cutoff,
            scale: policy.scale,
            configured_static_threshold_bits: policy.configured_static_threshold.to_bits(),
            train_fingerprint: policy.train_fingerprint.clone(),
            holdout_fingerprint: policy.holdout_fingerprint.clone(),
            corpus_fingerprint: policy.corpus_fingerprint.clone(),
            policy_digest: canonical_policy_digest(policy).unwrap(),
            calibrated_at: policy.calibrated_at,
            valid_until: policy.valid_until,
        }
    }

    #[test]
    fn dedup_calibration_round_trip_and_cas_conflict() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);")
            .unwrap();

        assert!(save_dedup_calibration_cas(&conn, &policy(1), 0).unwrap());
        let loaded = load_dedup_calibration(&conn, 1_500);
        assert_eq!(loaded.status, DedupCalibrationLoadStatus::Loaded);
        assert_eq!(loaded.policy.revision, 1);
        assert!(!save_dedup_calibration_cas(&conn, &policy(2), 0).unwrap());
        assert!(save_dedup_calibration_cas(&conn, &policy(2), 1).unwrap());
    }

    #[test]
    fn calibration_bundle_round_trip_cas_and_half_write_recovery() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);")
            .unwrap();

        assert!(
            save_dedup_calibration_bundle_cas(&conn, &policy(1), &seal(&policy(1)), 0).unwrap()
        );
        let loaded = load_dedup_calibration_for_runtime(&conn, 1_500, 0.70);
        assert!(loaded.context_verified());
        assert_eq!(resolve_hard_lexical_threshold(0.70, &loaded), 0.70);
        assert!(
            !save_dedup_calibration_bundle_cas(&conn, &policy(2), &seal(&policy(2)), 0).unwrap()
        );
        assert!(
            save_dedup_calibration_bundle_cas(&conn, &policy(2), &seal(&policy(2)), 1).unwrap()
        );

        let half = Connection::open_in_memory().unwrap();
        half.execute_batch("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);")
            .unwrap();
        assert!(save_dedup_calibration_cas(&half, &policy(1), 0).unwrap());
        let unverified = load_dedup_calibration_for_runtime(&half, 1_500, 0.70);
        assert_eq!(unverified.status, DedupCalibrationLoadStatus::Missing);
        assert_eq!(resolve_hard_lexical_threshold(0.70, &unverified), 0.70);
        assert!(
            save_dedup_calibration_bundle_cas(&half, &policy(2), &seal(&policy(2)), 1).unwrap()
        );
        assert!(load_dedup_calibration_for_runtime(&half, 1_500, 0.70).context_verified());
    }

    #[test]
    fn bundle_save_rejects_underpowered_non_terminal_policy() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);")
            .unwrap();
        let mut no_data = policy(1);
        no_data.status = DedupCalibrationStatus::NoData;
        no_data.holdout_revealed = false;
        let result = save_dedup_calibration_bundle_cas(&conn, &no_data, &seal(&no_data), 0);
        assert!(result.is_err());
        let rows: usize = conn
            .query_row(
                "SELECT COUNT(*) FROM metadata WHERE key IN (?1, ?2)",
                rusqlite::params![
                    DEDUP_CALIBRATION_METADATA_KEY,
                    DEDUP_CALIBRATION_SEAL_METADATA_KEY
                ],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(rows, 0);
    }

    #[test]
    fn production_threshold_helper_keeps_verified_ship_bundle_shadow_only() {
        let store = crate::store::SqliteStore::in_memory().unwrap();
        let now = chrono::Utc::now().timestamp();
        let mut current = policy(1);
        current.calibrated_at = now.saturating_sub(1);
        current.sealed_cutoff = now.saturating_sub(2);
        current.valid_until = now.saturating_add(1_000);
        assert!(
            save_dedup_calibration_bundle_cas(store.conn(), &current, &seal(&current), 0).unwrap()
        );
        let mut config = ReinConfig::default();
        config.search.dedup_similarity = 0.70;
        config.adaptive.enabled = true;

        assert_eq!(
            crate::ops::effective_hard_dedup_threshold_from_conn(store.conn(), &config),
            0.70
        );
        let status = crate::ops::adaptive_status_with_config(&store, &config);
        let calibration = &status["dedup_thresholds"]["calibration"];
        assert_eq!(calibration["adaptive_enabled"], true);
        assert_eq!(calibration["evidence_verified"], true);
        assert_eq!(calibration["applied"], false);
        assert_eq!(
            calibration["reason"],
            "verified_ship_shadow_only_production_cohort_missing"
        );
        config.adaptive.enabled = false;
        assert_eq!(
            crate::ops::effective_hard_dedup_threshold_from_conn(store.conn(), &config),
            0.70
        );
    }

    #[test]
    fn bundle_cas_rolls_back_policy_when_future_seal_refuses_update() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);")
            .unwrap();
        assert!(
            save_dedup_calibration_bundle_cas(&conn, &policy(1), &seal(&policy(1)), 0).unwrap()
        );
        conn.execute(
            "UPDATE metadata SET value = ?1 WHERE key = ?2",
            rusqlite::params![
                r#"{"schema_version":99,"revision":1,"future_field":"keep"}"#,
                DEDUP_CALIBRATION_SEAL_METADATA_KEY
            ],
        )
        .unwrap();

        assert!(
            !save_dedup_calibration_bundle_cas(&conn, &policy(2), &seal(&policy(2)), 1).unwrap()
        );
        let policy_revision: u64 = conn
            .query_row(
                "SELECT json_extract(value, '$.revision') FROM metadata WHERE key = ?1",
                rusqlite::params![DEDUP_CALIBRATION_METADATA_KEY],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(
            policy_revision, 1,
            "policy update must roll back with seal CAS"
        );
        let seal_raw: String = conn
            .query_row(
                "SELECT value FROM metadata WHERE key = ?1",
                rusqlite::params![DEDUP_CALIBRATION_SEAL_METADATA_KEY],
                |row| row.get(0),
            )
            .unwrap();
        assert!(seal_raw.contains("\"schema_version\":99"));
        assert!(seal_raw.contains("future_field"));
    }

    #[test]
    fn resolver_is_static_only_even_for_verified_ship() {
        let healthy = DedupCalibrationLoad {
            policy: policy(1),
            status: DedupCalibrationLoadStatus::Loaded,
            context_verified: true,
            error: None,
        };
        assert_eq!(resolve_hard_lexical_threshold(0.70, &healthy), 0.70);

        for status in [
            DedupCalibrationLoadStatus::Missing,
            DedupCalibrationLoadStatus::Corrupt,
            DedupCalibrationLoadStatus::UnsupportedSchema,
            DedupCalibrationLoadStatus::Stale,
            DedupCalibrationLoadStatus::FingerprintMismatch,
            DedupCalibrationLoadStatus::StorageError,
        ] {
            let unhealthy = DedupCalibrationLoad {
                policy: policy(1),
                status,
                context_verified: false,
                error: None,
            };
            assert_eq!(resolve_hard_lexical_threshold(0.70, &unhealthy), 0.70);
        }

        let mut no_data = healthy.clone();
        no_data.policy.status = DedupCalibrationStatus::NoData;
        assert_eq!(resolve_hard_lexical_threshold(0.70, &no_data), 0.70);
        let mut lower = healthy.clone();
        lower.policy.effective_hard_threshold = 0.40;
        assert_eq!(resolve_hard_lexical_threshold(0.70, &lower), 0.70);
    }

    #[test]
    fn future_schema_is_preserved_and_never_activated() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);")
            .unwrap();
        conn.execute(
            "INSERT INTO metadata(key, value) VALUES (?1, ?2)",
            rusqlite::params![
                DEDUP_CALIBRATION_METADATA_KEY,
                r#"{"schema_version":99,"revision":7,"status":"future"}"#
            ],
        )
        .unwrap();

        let loaded = load_dedup_calibration(&conn, 1_500);
        assert_eq!(loaded.status, DedupCalibrationLoadStatus::UnsupportedSchema);
        assert!(!save_dedup_calibration_cas(&conn, &policy(8), 7).unwrap());
        let raw: String = conn
            .query_row(
                "SELECT value FROM metadata WHERE key = ?1",
                rusqlite::params![DEDUP_CALIBRATION_METADATA_KEY],
                |row| row.get(0),
            )
            .unwrap();
        assert!(raw.contains("\"schema_version\":99"));
    }

    #[test]
    fn corrupt_non_finite_and_fingerprint_empty_rows_fail_closed() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);")
            .unwrap();
        for raw in [
            "{not-json",
            r#"{"schema_version":1,"revision":1,"status":"ship","scale":"lexical","configured_static_threshold":0.7,"candidate_threshold":0.8,"shadow_threshold":0.4,"effective_hard_threshold":0.8,"train_fingerprint":"","holdout_fingerprint":"h","corpus_fingerprint":"c","calibrated_at":1000,"valid_until":2000}"#,
        ] {
            conn.execute(
                "INSERT INTO metadata(key, value) VALUES (?1, ?2) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                rusqlite::params![DEDUP_CALIBRATION_METADATA_KEY, raw],
            )
            .unwrap();
            let loaded = load_dedup_calibration(&conn, 1_500);
            assert_eq!(loaded.status, DedupCalibrationLoadStatus::Corrupt);
            assert_eq!(resolve_hard_lexical_threshold(0.70, &loaded), 0.70);
        }
    }

    #[test]
    fn stale_or_vector_policy_never_changes_lexical_hard_threshold() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);")
            .unwrap();
        assert!(save_dedup_calibration_cas(&conn, &policy(1), 0).unwrap());
        let stale = load_dedup_calibration(&conn, 2_001);
        assert_eq!(stale.status, DedupCalibrationLoadStatus::Stale);
        assert_eq!(resolve_hard_lexical_threshold(0.70, &stale), 0.70);

        let mut vector = policy(2);
        vector.scale = DedupCalibrationScale::VectorCosine;
        let loaded = DedupCalibrationLoad {
            policy: vector,
            status: DedupCalibrationLoadStatus::Loaded,
            context_verified: true,
            error: None,
        };
        assert_eq!(resolve_hard_lexical_threshold(0.70, &loaded), 0.70);
    }

    #[test]
    fn fingerprint_or_static_config_mismatch_is_explicit_and_fails_closed() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);")
            .unwrap();
        assert!(
            save_dedup_calibration_bundle_cas(&conn, &policy(1), &seal(&policy(1)), 0).unwrap()
        );

        let matching = load_dedup_calibration_for_runtime(&conn, 1_500, 0.70);
        assert_eq!(matching.status, DedupCalibrationLoadStatus::Loaded);
        assert!(matching.context_verified());
        assert_eq!(resolve_hard_lexical_threshold(0.70, &matching), 0.70);

        let static_mismatch = load_dedup_calibration_for_runtime(&conn, 1_500, 0.71);
        assert_eq!(
            static_mismatch.status,
            DedupCalibrationLoadStatus::FingerprintMismatch
        );
        assert_eq!(resolve_hard_lexical_threshold(0.71, &static_mismatch), 0.71);

        conn.execute(
            "UPDATE metadata
                SET value = json_set(value, '$.holdout_fingerprint', 'changed')
              WHERE key = ?1",
            rusqlite::params![DEDUP_CALIBRATION_SEAL_METADATA_KEY],
        )
        .unwrap();
        let fingerprint_mismatch = load_dedup_calibration_for_runtime(&conn, 1_500, 0.70);
        assert_eq!(
            fingerprint_mismatch.status,
            DedupCalibrationLoadStatus::FingerprintMismatch
        );
        assert!(!fingerprint_mismatch.context_verified());
        assert_eq!(
            resolve_hard_lexical_threshold(0.70, &fingerprint_mismatch),
            0.70
        );
    }

    #[test]
    fn sealed_bundle_digest_rejects_mutation_of_any_policy_field() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);")
            .unwrap();
        let current = policy(1);
        assert!(save_dedup_calibration_bundle_cas(&conn, &current, &seal(&current), 0).unwrap());

        // `shadow_threshold` was not one of the manifest's individually copied
        // fields. A complete policy digest must still detect this mutation.
        conn.execute(
            "UPDATE metadata
                SET value = json_set(value, '$.shadow_threshold', 0.41)
              WHERE key = ?1",
            rusqlite::params![DEDUP_CALIBRATION_METADATA_KEY],
        )
        .unwrap();

        let loaded = load_dedup_calibration_for_runtime(&conn, 1_500, 0.70);
        assert_eq!(
            loaded.status,
            DedupCalibrationLoadStatus::FingerprintMismatch
        );
        assert!(!loaded.context_verified());
        assert_eq!(resolve_hard_lexical_threshold(0.70, &loaded), 0.70);
    }

    #[test]
    fn wrong_revision_is_fail_closed_and_requests_atomic_two_row_reset() {
        let store = crate::store::SqliteStore::in_memory().unwrap();
        let current = policy(1);
        assert!(
            save_dedup_calibration_bundle_cas(store.conn(), &current, &seal(&current), 0).unwrap()
        );
        store
            .conn()
            .execute(
                "UPDATE metadata SET value = json_set(value, '$.revision', 9) WHERE key = ?1",
                rusqlite::params![DEDUP_CALIBRATION_SEAL_METADATA_KEY],
            )
            .unwrap();

        let loaded = load_dedup_calibration_for_runtime(store.conn(), 1_500, 0.70);
        assert_eq!(
            loaded.status,
            DedupCalibrationLoadStatus::FingerprintMismatch
        );
        assert_eq!(resolve_hard_lexical_threshold(0.70, &loaded), 0.70);

        let error = crate::eval::gates::dedup::refresh_dedup_calibration_policy(
            &store, 0.70, 0.40, 1_500, 1_000,
        )
        .unwrap_err();
        assert!(error.contains("atomically reset both calibration metadata rows"));
    }

    #[test]
    fn seal_only_half_write_is_fail_closed_and_requests_atomic_two_row_reset() {
        let store = crate::store::SqliteStore::in_memory().unwrap();
        let raw = serde_json::to_string(&seal(&policy(1))).unwrap();
        store
            .conn()
            .execute(
                "INSERT INTO metadata(key, value) VALUES (?1, ?2)",
                rusqlite::params![DEDUP_CALIBRATION_SEAL_METADATA_KEY, raw],
            )
            .unwrap();

        let loaded = load_dedup_calibration_for_runtime(store.conn(), 1_500, 0.70);
        assert_eq!(loaded.status, DedupCalibrationLoadStatus::Missing);
        assert_eq!(resolve_hard_lexical_threshold(0.70, &loaded), 0.70);

        let error = crate::eval::gates::dedup::refresh_dedup_calibration_policy(
            &store, 0.70, 0.40, 1_500, 1_000,
        )
        .unwrap_err();
        assert!(error.contains("atomically reset both calibration metadata rows"));
    }

    #[test]
    fn adaptive_disabled_status_matches_static_only_runtime() {
        let store = crate::store::SqliteStore::in_memory().unwrap();
        let now = chrono::Utc::now().timestamp();
        let mut current = policy(1);
        current.calibrated_at = now.saturating_sub(1);
        current.sealed_cutoff = now.saturating_sub(2);
        current.valid_until = now.saturating_add(1_000);
        assert!(
            save_dedup_calibration_bundle_cas(store.conn(), &current, &seal(&current), 0).unwrap()
        );

        let mut config = ReinConfig::default();
        config.search.dedup_similarity = 0.70;
        config.adaptive.enabled = false;
        let runtime = crate::ops::effective_hard_dedup_threshold_from_conn(store.conn(), &config);
        let status = crate::ops::adaptive_status_with_config(&store, &config);

        assert_eq!(runtime, 0.70);
        assert_eq!(
            status["dedup_thresholds"]["dedup_threshold_hard_effective"].as_f64(),
            Some(f64::from(runtime))
        );
        let calibration = &status["dedup_thresholds"]["calibration"];
        assert_eq!(calibration["adaptive_enabled"], false);
        assert_eq!(calibration["evidence_verified"], true);
        assert_eq!(calibration["applied"], false);
        assert_eq!(calibration["reason"], "adaptive_disabled");
    }

    #[test]
    fn production_score_space_mismatch_cannot_activate_calibration_candidate() {
        let config = ReinConfig::default();
        let mut candidate = crate::ops::build_memory(
            &config,
            "same-topic".to_string(),
            "alpha beta gamma delta rightone righttwo".to_string(),
            crate::types::Importance::Medium,
            vec![],
            crate::types::Source::Manual,
        );
        candidate.cluster_id = Some(7);
        let incoming = "alpha beta gamma delta leftone lefttwo";
        let calibration_score = crate::extract::similarity(incoming, &candidate.content);
        let production_score =
            crate::extract::dedup::score_candidate("same-topic", incoming, &candidate, Some(7))
                .final_score;

        assert!(calibration_score < 0.70);
        assert!(production_score > 0.70);

        let verified_ship = DedupCalibrationLoad {
            policy: policy(1),
            status: DedupCalibrationLoadStatus::Loaded,
            context_verified: true,
            error: None,
        };
        assert_eq!(
            resolve_hard_lexical_threshold(0.70, &verified_ship),
            0.70,
            "raw lexical calibration cannot move a boundary shared with an enriched production score"
        );
    }

    #[test]
    fn forged_ship_statistics_and_slice_sets_are_rejected() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);")
            .unwrap();

        let mut forged_fp_bound = policy(1);
        forged_fp_bound.false_positive_upper_95 = Some(0.001);
        assert!(save_dedup_calibration_cas(&conn, &forged_fp_bound, 0).is_err());

        let mut underpowered_utility = policy(1);
        underpowered_utility.sealed_positive_count = 10;
        underpowered_utility.utility.n = 10;
        underpowered_utility.utility.miss_rate_upper_95 = Some(0.001);
        underpowered_utility.holdout_confusion.true_positives = 10;
        assert!(save_dedup_calibration_cas(&conn, &underpowered_utility, 0).is_err());

        let mut caller_selected_slices = policy(1);
        caller_selected_slices
            .required_slices
            .retain(|slice| slice.name == "operator_distinct");
        assert!(save_dedup_calibration_cas(&conn, &caller_selected_slices, 0).is_err());

        let mut structural_only = policy(1);
        structural_only.provenance = DedupCalibrationProvenance::StructuralAnchors;
        assert!(save_dedup_calibration_cas(&conn, &structural_only, 0).is_err());

        let valid = policy(1);
        let mut mismatched_seal = seal(&valid);
        mismatched_seal.corpus_fingerprint = "different-corpus".to_string();
        assert!(save_dedup_calibration_bundle_cas(&conn, &valid, &mismatched_seal, 0).is_err());
    }

    #[test]
    fn resolver_revalidates_even_a_manually_constructed_loaded_policy() {
        let mut forged = policy(1);
        forged.false_positive_upper_95 = Some(0.001);
        let loaded = DedupCalibrationLoad {
            policy: forged,
            status: DedupCalibrationLoadStatus::Loaded,
            context_verified: true,
            error: None,
        };

        assert_eq!(resolve_hard_lexical_threshold(0.70, &loaded), 0.70);
    }
}
