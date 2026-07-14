//! Dedicated durable state for A12 self-supervised recall calibration.
//!
//! This row is intentionally separate from [`super::adaptive::AdaptiveState`]:
//! a binary that does not understand a newer A12 schema must not deserialize a
//! large adaptive snapshot and erase automatic-calibration evidence as a side
//! effect of an unrelated save.

use rusqlite::{params, OptionalExtension};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

/// Versioned active-pointer row. Revision payloads live under
/// [`A12_CALIBRATION_REVISION_KEY_PREFIX`] and are immutable.
pub const A12_CALIBRATION_METADATA_KEY: &str = "a12_calibration_active";
pub const A12_CALIBRATION_REVISION_KEY_PREFIX: &str = "a12_calibration_revision:";
pub const A12_CALIBRATION_SCHEMA_VERSION: u32 = 2;
pub const A12_DEFAULT_NOISE_FLOOR: f64 = 0.02;
const SIMPLEX_SUM_TOLERANCE: f64 = 1e-6;
const FLOAT_COMPARISON_RELATIVE_TOLERANCE: f64 = 1e-12;

/// Holdout verdict for one independently calibrated recall-fusion scope.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum A12CalibrationVerdict {
    Ship,
    Bail,
    #[default]
    NoData,
}

/// Persisted six-dimensional recall simplex. The runtime optimizer owns its
/// in-memory representation; this type is a stable serialization boundary.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct A12FusionSimplex {
    pub bm25: f64,
    pub vector: f64,
    pub kg: f64,
    pub episode: f64,
    pub support: f64,
    pub diversity: f64,
}

impl Default for A12FusionSimplex {
    fn default() -> Self {
        Self {
            bm25: 0.45,
            vector: 0.45,
            kg: 0.04,
            episode: 0.03,
            support: 0.02,
            diversity: 0.01,
        }
    }
}

impl A12FusionSimplex {
    fn values(self) -> [f64; 6] {
        [
            self.bm25,
            self.vector,
            self.kg,
            self.episode,
            self.support,
            self.diversity,
        ]
    }

    fn validate(self) -> Result<(), String> {
        let values = self.values();
        if values
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err("A12 fusion simplex weights must be finite and in [0, 1]".to_string());
        }
        let sum = values.iter().sum::<f64>();
        if (sum - 1.0).abs() > SIMPLEX_SUM_TOLERANCE {
            return Err(format!("A12 fusion simplex must sum to 1 (observed {sum})"));
        }
        Ok(())
    }
}

/// Complete paired top-3 contingency table and McNemar projection. Keeping
/// the primitive fields avoids coupling this durable schema to eval-module
/// implementation types while preserving everything Trust/doctor must show.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct A12PairedTop3Stats {
    pub n: u64,
    pub both_hit: u64,
    pub baseline_only: u64,
    pub treatment_only: u64,
    pub neither_hit: u64,
    pub chi_squared: f64,
    pub p_value: f64,
    pub diff_point: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
    pub used_exact: bool,
}

impl A12PairedTop3Stats {
    /// Recompute every derived McNemar field from the persisted contingency
    /// table. Runtime diagnostics should prefer this projection over trusting
    /// serialized floating-point fields directly.
    pub fn recomputed_mcnemar(&self) -> Option<crate::eval::mcnemar::McNemarResult> {
        crate::eval::mcnemar::mcnemar_from_counts(
            u32::try_from(self.both_hit).ok()?,
            u32::try_from(self.baseline_only).ok()?,
            u32::try_from(self.treatment_only).ok()?,
            u32::try_from(self.neither_hit).ok()?,
        )
    }

    fn validate(self) -> Result<(), String> {
        let total = self
            .both_hit
            .checked_add(self.baseline_only)
            .and_then(|value| value.checked_add(self.treatment_only))
            .and_then(|value| value.checked_add(self.neither_hit))
            .ok_or_else(|| "A12 paired top-3 counts overflow".to_string())?;
        if total != self.n {
            return Err(format!(
                "A12 paired top-3 counts sum to {total}, expected n={}",
                self.n
            ));
        }
        let expected = self.recomputed_mcnemar().ok_or_else(|| {
            "A12 paired top-3 counts exceed the McNemar representation".to_string()
        })?;
        if self.n != u64::from(expected.n)
            || self.used_exact != expected.used_exact
            || !float_matches(self.chi_squared, expected.chi_squared)
            || !float_matches(self.p_value, expected.p_value)
            || !float_matches(self.diff_point, expected.diff_point)
            || !float_matches(self.ci_lower, expected.ci_lower)
            || !float_matches(self.ci_upper, expected.ci_upper)
        {
            return Err(
                "A12 persisted McNemar projection does not match its contingency table".to_string(),
            );
        }
        Ok(())
    }
}

fn float_matches(observed: f64, expected: f64) -> bool {
    if !observed.is_finite() || !expected.is_finite() {
        return false;
    }
    if observed == expected {
        return true;
    }
    let scale = expected.abs().max(f64::MIN_POSITIVE);
    (observed - expected).abs() <= FLOAT_COMPARISON_RELATIVE_TOLERANCE * scale
}

/// Observation-view counts retained for diagnostics. Counts may exceed family
/// ESS because canonical, concept, and episode views are averaged within one
/// family before that family contributes weight one.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct A12ProvenanceCounts {
    pub canonical_loo: u64,
    pub concept_loo: u64,
    pub episode_loo: u64,
}

/// Stable identity of a scope. [`A12CalibrationScope::key`] deliberately
/// matches the existing learned-shadow bucket convention so policy activation
/// can map it to `recall_fusion:<key>` without lossy parsing.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum A12CalibrationScope {
    Global,
    QueryType { query_type: String },
    Cluster { query_type: String, cluster_id: i64 },
}

impl A12CalibrationScope {
    pub fn key(&self) -> String {
        match self {
            Self::Global => "global".to_string(),
            Self::QueryType { query_type } => query_type.clone(),
            Self::Cluster {
                query_type,
                cluster_id,
            } => format!("{query_type}:{cluster_id}"),
        }
    }

    pub fn is_cluster(&self) -> bool {
        matches!(self, Self::Cluster { .. })
    }

    fn validate(&self) -> Result<(), String> {
        match self {
            Self::Global => Ok(()),
            Self::QueryType { query_type } => validate_query_type(query_type),
            Self::Cluster {
                query_type,
                cluster_id,
            } => {
                validate_query_type(query_type)?;
                if *cluster_id < 0 {
                    return Err("A12 cluster scope id must be non-negative".to_string());
                }
                Ok(())
            }
        }
    }
}

fn validate_query_type(query_type: &str) -> Result<(), String> {
    if query_type.is_empty() || query_type == "global" || query_type.contains(':') {
        return Err(
            "A12 query-type scope must be non-empty and cannot contain ':' or equal 'global'"
                .to_string(),
        );
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum A12ScopeInvalidationReason {
    Reclustered,
}

/// Family-level paired holdout cells for one label-provenance source.
/// Derived diagnostics only: deterministic for a fixed holdout trace and
/// deliberately excluded from every fingerprint/unchanged-skip identity.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct A12ProvenanceHoldoutCells {
    /// Families with at least one holdout view carrying this provenance.
    pub family_count: u64,
    pub both_hit: u64,
    pub baseline_only: u64,
    pub treatment_only: u64,
    pub neither_hit: u64,
}

/// Per-provenance paired holdout cells for one scope, keyed by the same
/// three label sources as [`A12ProvenanceCounts`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct A12ProvenanceHoldoutStats {
    pub canonical_loo: A12ProvenanceHoldoutCells,
    pub concept_loo: A12ProvenanceHoldoutCells,
    pub episode_loo: A12ProvenanceHoldoutCells,
}

impl A12ProvenanceHoldoutStats {
    fn cells(&self) -> [A12ProvenanceHoldoutCells; 3] {
        [self.canonical_loo, self.concept_loo, self.episode_loo]
    }

    /// True iff two label-provenance sources pull the holdout in opposite
    /// directions: one source has strictly more treatment-only families and
    /// another strictly more baseline-only families. A strict majority in
    /// either direction implies at least one discordant pair, so a single
    /// disagreement in each direction already reports a conflict — raw
    /// evidence only, no tunable rate threshold anywhere.
    pub fn direction_conflict(&self) -> bool {
        self.cells()
            .iter()
            .any(|cells| cells.treatment_only > cells.baseline_only)
            && self
                .cells()
                .iter()
                .any(|cells| cells.baseline_only > cells.treatment_only)
    }
}

/// A cluster-scope tombstone. The calibrated weights and holdout statistics
/// stay in the row for diagnostics, but runtime resolution must ignore it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct A12ScopeInvalidation {
    pub reason: A12ScopeInvalidationReason,
    pub from_cluster_generation: u64,
    pub to_cluster_generation: u64,
    pub invalidated_at: i64,
}

/// One persisted global/query/cluster calibration result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct A12ScopeEntry {
    pub scope: A12CalibrationScope,
    /// Canonical-snapshot identity copied into every scope. Validation binds
    /// these fields to the containing immutable revision, so copying an old
    /// Ship entry into a newer generation cannot make it active.
    pub canonical_generation: u64,
    pub generation_fingerprint: String,
    /// Exact local SQLite recall identity produced by Task 3. Older schema-2
    /// revisions omit this field and remain readable, but cannot satisfy the
    /// Task-5 complete-run contract.
    #[serde(default)]
    pub source_snapshot_fingerprint: String,
    pub snapshot_cutoff: i64,
    pub corpus_fingerprint: String,
    pub train_family_ess: u64,
    #[serde(default)]
    pub train_case_count: u64,
    pub holdout_family_ess: u64,
    pub simplex: A12FusionSimplex,
    pub verdict: A12CalibrationVerdict,
    /// Positive non-inferiority tolerance used when this verdict was sealed.
    /// A runtime with a different configured/default value treats the entry as
    /// stale rather than silently reinterpreting the old verdict.
    pub noise_floor: f64,
    pub paired_top3: A12PairedTop3Stats,
    pub provenance: A12ProvenanceCounts,
    /// Per-provenance paired holdout cells. Derived diagnostics; absent on
    /// rows sealed before this field existed, so it must stay optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance_holdout: Option<A12ProvenanceHoldoutStats>,
    pub training_fingerprint: String,
    pub holdout_fingerprint: String,
    pub optimizer_fingerprint: String,
    pub evaluation_fingerprint: String,
    #[serde(default)]
    pub holdout_reason: String,
    pub calibrated_at: i64,
    pub evaluated_at: i64,
    /// Earliest Unix millisecond at which fixed-time replay may diverge from
    /// production because a relative temporal window or KG validity edge
    /// changes membership. `None` means this scope observed no such boundary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_until_exclusive: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cluster_generation: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub invalidation: Option<A12ScopeInvalidation>,
}

impl A12ScopeEntry {
    pub fn matches_noise_floor(&self, expected: f64) -> bool {
        expected.is_finite() && expected > 0.0 && float_matches(self.noise_floor, expected)
    }

    /// Generation/noise/cluster identity check used by the Task 5 resolver.
    /// History loads never activate independently; callers must also require
    /// [`A12CalibrationRevisionLoad::can_activate`].
    pub fn is_current_for(&self, state: &A12CalibrationState, expected_noise_floor: f64) -> bool {
        let cluster_generation_matches = match self.scope {
            A12CalibrationScope::Cluster { .. } => {
                self.cluster_generation == Some(state.cluster_generation)
            }
            _ => self.cluster_generation.is_none(),
        };
        state.is_complete()
            && self.invalidation.is_none()
            && cluster_generation_matches
            && self.canonical_generation == state.generation
            && self.generation_fingerprint == state.generation_fingerprint
            && self.snapshot_cutoff == state.snapshot_cutoff
            && self.corpus_fingerprint == state.corpus_fingerprint
            && self.matches_noise_floor(expected_noise_floor)
    }

    /// Identity check plus the fixed-time replay validity horizon. Expiry is
    /// exclusive and fail-closed: at the first recorded transition second the
    /// sealed holdout evidence can no longer activate.
    pub fn is_current_for_at(
        &self,
        state: &A12CalibrationState,
        expected_noise_floor: f64,
        now_unix_ms: i64,
    ) -> bool {
        now_unix_ms >= 0
            && self.is_current_for(state, expected_noise_floor)
            && self
                .valid_until_exclusive
                .is_none_or(|boundary| now_unix_ms < boundary)
    }

    fn validate(&self, state: &A12CalibrationState) -> Result<(), String> {
        self.scope.validate()?;
        self.simplex.validate()?;
        self.paired_top3.validate()?;
        if self.canonical_generation != state.generation
            || self.generation_fingerprint != state.generation_fingerprint
            || self.snapshot_cutoff != state.snapshot_cutoff
            || self.corpus_fingerprint != state.corpus_fingerprint
        {
            return Err("A12 scope is not bound to its containing generation".to_string());
        }
        if !self.noise_floor.is_finite() || self.noise_floor <= 0.0 || self.noise_floor > 1.0 {
            return Err("A12 scope noise_floor must be finite and in (0, 1]".to_string());
        }
        if self.holdout_family_ess != self.paired_top3.n {
            return Err(format!(
                "A12 holdout ESS {} does not match paired n {}",
                self.holdout_family_ess, self.paired_top3.n
            ));
        }
        if let Some(stats) = self.provenance_holdout {
            for cells in [stats.canonical_loo, stats.concept_loo, stats.episode_loo] {
                let total = cells
                    .both_hit
                    .checked_add(cells.baseline_only)
                    .and_then(|sum| sum.checked_add(cells.treatment_only))
                    .and_then(|sum| sum.checked_add(cells.neither_hit));
                if total != Some(cells.family_count) {
                    return Err(
                        "A12 provenance holdout cells do not sum to their family count".to_string(),
                    );
                }
            }
        }
        if [
            self.training_fingerprint.as_str(),
            self.holdout_fingerprint.as_str(),
            self.optimizer_fingerprint.as_str(),
            self.evaluation_fingerprint.as_str(),
        ]
        .iter()
        .any(|fingerprint| fingerprint.is_empty())
        {
            return Err("A12 scope fingerprints must be non-empty".to_string());
        }
        let evaluated_at_millis = self.evaluated_at.checked_mul(1_000);
        if self.calibrated_at < 0
            || self.evaluated_at < self.calibrated_at
            || evaluated_at_millis.is_none()
            || self
                .valid_until_exclusive
                .is_some_and(|boundary| boundary <= evaluated_at_millis.unwrap_or(i64::MAX))
            || self
                .invalidation
                .is_some_and(|value| value.invalidated_at < self.evaluated_at)
        {
            return Err("A12 scope timestamps are inconsistent".to_string());
        }
        if matches!(
            self.verdict,
            A12CalibrationVerdict::Ship | A12CalibrationVerdict::Bail
        ) && (self.train_family_ess == 0 || self.holdout_family_ess == 0)
        {
            return Err("terminal A12 verdicts require non-zero train and holdout ESS".to_string());
        }
        if self.verdict == A12CalibrationVerdict::Ship
            && self.paired_top3.ci_lower < -self.noise_floor
        {
            return Err("A12 Ship verdict contradicts the McNemar lower bound".to_string());
        }
        if self.verdict == A12CalibrationVerdict::Bail
            && self.paired_top3.ci_upper > -self.noise_floor
        {
            return Err("A12 Bail verdict contradicts the McNemar upper bound".to_string());
        }

        match (&self.scope, self.cluster_generation, self.invalidation) {
            (A12CalibrationScope::Cluster { .. }, Some(generation), None)
                if generation == state.cluster_generation =>
            {
                Ok(())
            }
            (A12CalibrationScope::Cluster { .. }, Some(generation), Some(invalidation))
                if invalidation.reason == A12ScopeInvalidationReason::Reclustered
                    && invalidation.from_cluster_generation == generation
                    && invalidation.to_cluster_generation <= state.cluster_generation
                    && invalidation.to_cluster_generation > generation =>
            {
                Ok(())
            }
            (A12CalibrationScope::Cluster { .. }, None, _) => {
                Err("A12 cluster scope is missing its cluster generation".to_string())
            }
            (A12CalibrationScope::Cluster { .. }, Some(_), _) => {
                Err("A12 cluster scope does not match the active clustering generation".to_string())
            }
            (_, None, None) => Ok(()),
            (_, Some(_), _) => {
                Err("A12 global/query scopes cannot carry a cluster generation".to_string())
            }
            (_, None, Some(_)) => {
                Err("only A12 cluster scopes may carry an invalidation".to_string())
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum A12CalibrationPhase {
    Pending,
    Complete,
}

/// Cadence identity for Task 5. This is optional so revisions written by the
/// original schema-2 implementation remain readable, but a legacy revision is
/// never silently treated as a completed automatic calibration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct A12CalibrationRunMetadata {
    pub phase: A12CalibrationPhase,
    pub source_snapshot_fingerprint: String,
    pub behavior_config_fingerprint: String,
}

/// Atomic metadata envelope for one canonical-snapshot generation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct A12CalibrationState {
    pub schema_version: u32,
    pub revision: u64,
    pub generation: u64,
    pub generation_fingerprint: String,
    pub snapshot_cutoff: i64,
    pub corpus_fingerprint: String,
    pub cluster_generation: u64,
    pub scopes: BTreeMap<String, A12ScopeEntry>,
    pub created_at: i64,
    pub updated_at: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run: Option<A12CalibrationRunMetadata>,
}

impl Default for A12CalibrationState {
    fn default() -> Self {
        Self {
            schema_version: A12_CALIBRATION_SCHEMA_VERSION,
            revision: 0,
            generation: 0,
            generation_fingerprint: String::new(),
            snapshot_cutoff: 0,
            corpus_fingerprint: String::new(),
            cluster_generation: 0,
            scopes: BTreeMap::new(),
            created_at: 0,
            updated_at: 0,
            run: None,
        }
    }
}

impl A12CalibrationState {
    /// A pending revision is an immutable, empty activation barrier. Merely
    /// observing a schema-2 row without phase metadata is not enough.
    pub fn is_pending(&self) -> bool {
        self.run
            .as_ref()
            .is_some_and(|run| run.phase == A12CalibrationPhase::Pending)
    }

    /// Only an explicitly completed Task-5 revision may feed activation.
    pub fn is_complete(&self) -> bool {
        self.run
            .as_ref()
            .is_some_and(|run| run.phase == A12CalibrationPhase::Complete)
    }

    /// Earliest fixed-time replay boundary across completed scopes, in Unix
    /// milliseconds. Pending and legacy schema-2 rows have no reusable
    /// cadence horizon.
    pub fn next_expiry_unix_ms(&self) -> Option<i64> {
        if !self.is_complete() {
            return None;
        }
        self.scopes
            .values()
            .filter_map(|entry| entry.valid_until_exclusive)
            .min()
    }

    /// Mark only cluster-scoped entries unusable after M4 changes cluster ids.
    /// Global and query-type scopes remain byte-for-byte intact. The first
    /// tombstone is retained across later reclusters so diagnostics show the
    /// exact transition that invalidated the evidence.
    pub fn invalidate_cluster_scopes_after_recluster(
        &mut self,
        new_cluster_generation: u64,
        invalidated_at: i64,
    ) -> Result<usize, String> {
        if new_cluster_generation <= self.cluster_generation {
            return Err(format!(
                "new cluster generation {new_cluster_generation} must exceed current {}",
                self.cluster_generation
            ));
        }
        if invalidated_at < 0 || invalidated_at < self.updated_at {
            return Err("A12 recluster invalidation time cannot move backwards".to_string());
        }

        let previous_generation = self.cluster_generation;
        let mut invalidated = 0;
        for entry in self.scopes.values_mut() {
            if entry.scope.is_cluster() && entry.invalidation.is_none() {
                let from_cluster_generation =
                    entry.cluster_generation.unwrap_or(previous_generation);
                entry.invalidation = Some(A12ScopeInvalidation {
                    reason: A12ScopeInvalidationReason::Reclustered,
                    from_cluster_generation,
                    to_cluster_generation: new_cluster_generation,
                    invalidated_at,
                });
                invalidated += 1;
            }
        }
        self.cluster_generation = new_cluster_generation;
        self.updated_at = invalidated_at;
        Ok(invalidated)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum A12CalibrationLoadStatus {
    Missing,
    Loaded,
    Corrupt,
    UnsupportedSchema,
    StorageError,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct A12CalibrationLoad {
    pub state: A12CalibrationState,
    pub status: A12CalibrationLoadStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// One immutable revision loaded directly for diagnostics or rollback
/// analysis. A historical revision is never activation-eligible merely because
/// its payload is healthy: only the revision named by the active pointer can
/// activate.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct A12CalibrationRevisionLoad {
    pub state: A12CalibrationState,
    pub status: A12CalibrationLoadStatus,
    pub active: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl A12CalibrationRevisionLoad {
    pub fn can_activate(&self) -> bool {
        self.active && self.status == A12CalibrationLoadStatus::Loaded
    }
}

/// Compact immutable-history index used by doctor and Trust diagnostics.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct A12CalibrationHistoryEntry {
    pub row_key: String,
    pub generation: u64,
    pub revision: u64,
    pub active: bool,
    pub status: A12CalibrationLoadStatus,
    pub generation_fingerprint: String,
    pub snapshot_cutoff: i64,
    pub corpus_fingerprint: String,
    pub updated_at: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Collision-free identity for the next immutable revision. Revisions advance
/// globally even when the canonical generation changes, which makes recovery
/// possible without deleting or overwriting corrupt history rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct A12CalibrationRevisionIdentity {
    pub generation: u64,
    pub revision: u64,
}

/// Small mutable head that selects exactly one immutable revision row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct A12CalibrationActivePointer {
    schema_version: u32,
    generation: u64,
    revision: u64,
    row_key: String,
    generation_fingerprint: String,
    snapshot_cutoff: i64,
    corpus_fingerprint: String,
    state_digest: String,
    activated_at: i64,
}

#[derive(Debug)]
struct A12ActivePointerLoad {
    pointer: Option<A12CalibrationActivePointer>,
    status: A12CalibrationLoadStatus,
    error: Option<String>,
}

fn fail_closed_load(
    status: A12CalibrationLoadStatus,
    error: impl Into<Option<String>>,
) -> A12CalibrationLoad {
    A12CalibrationLoad {
        state: A12CalibrationState::default(),
        status,
        error: error.into(),
    }
}

fn fail_closed_revision_load(
    status: A12CalibrationLoadStatus,
    error: impl Into<Option<String>>,
) -> A12CalibrationRevisionLoad {
    A12CalibrationRevisionLoad {
        state: A12CalibrationState::default(),
        status,
        active: false,
        error: error.into(),
    }
}

fn validate_state(state: &A12CalibrationState) -> Result<(), String> {
    if state.schema_version != A12_CALIBRATION_SCHEMA_VERSION {
        return Err(format!(
            "unsupported A12 calibration schema {}",
            state.schema_version
        ));
    }
    if state.revision == 0 || state.generation == 0 {
        return Err("persisted A12 state requires positive revision and generation".to_string());
    }
    if state.generation_fingerprint.is_empty() || state.corpus_fingerprint.is_empty() {
        return Err("A12 generation and corpus fingerprints must be non-empty".to_string());
    }
    if state.snapshot_cutoff < 0 || state.created_at < 0 || state.updated_at < state.created_at {
        return Err("A12 state timestamps and snapshot cutoff are inconsistent".to_string());
    }
    if let Some(run) = &state.run {
        if run.source_snapshot_fingerprint.is_empty() || run.behavior_config_fingerprint.is_empty()
        {
            return Err("A12 run identities must be non-empty".to_string());
        }
        if run.phase == A12CalibrationPhase::Pending && !state.scopes.is_empty() {
            return Err("A12 pending revisions must have empty scopes".to_string());
        }
        if run.phase == A12CalibrationPhase::Complete
            && state.scopes.values().any(|entry| {
                entry.source_snapshot_fingerprint != run.source_snapshot_fingerprint
                    || entry.holdout_reason.is_empty()
            })
        {
            return Err(
                "A12 complete scopes must preserve source identity and holdout reason".to_string(),
            );
        }
    }
    for (key, entry) in &state.scopes {
        if key != &entry.scope.key() {
            return Err(format!(
                "A12 scope map key '{key}' does not match scope key '{}'",
                entry.scope.key()
            ));
        }
        entry.validate(state)?;
        if entry.evaluated_at > state.updated_at {
            return Err(format!(
                "A12 scope '{key}' was evaluated after the state update timestamp"
            ));
        }
        if entry
            .invalidation
            .is_some_and(|invalidation| invalidation.invalidated_at > state.updated_at)
        {
            return Err(format!(
                "A12 scope '{key}' was invalidated after the state update timestamp"
            ));
        }
    }
    Ok(())
}

fn parse_raw_state(raw: &str) -> A12CalibrationLoad {
    let value = match serde_json::from_str::<serde_json::Value>(raw) {
        Ok(value) => value,
        Err(error) => {
            return fail_closed_load(A12CalibrationLoadStatus::Corrupt, Some(error.to_string()));
        }
    };
    let Some(schema_version) = value
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
    else {
        return fail_closed_load(
            A12CalibrationLoadStatus::Corrupt,
            Some("A12 calibration row is missing a numeric schema_version".to_string()),
        );
    };
    if schema_version > u64::from(A12_CALIBRATION_SCHEMA_VERSION) {
        return fail_closed_load(
            A12CalibrationLoadStatus::UnsupportedSchema,
            Some(format!(
                "A12 calibration schema {schema_version} is newer than binary schema {}",
                A12_CALIBRATION_SCHEMA_VERSION
            )),
        );
    }
    if schema_version != u64::from(A12_CALIBRATION_SCHEMA_VERSION) {
        return fail_closed_load(
            A12CalibrationLoadStatus::Corrupt,
            Some(format!(
                "A12 calibration schema {schema_version} is older or invalid"
            )),
        );
    }

    match serde_json::from_value::<A12CalibrationState>(value) {
        Ok(state) => match validate_state(&state) {
            Ok(()) => A12CalibrationLoad {
                state,
                status: A12CalibrationLoadStatus::Loaded,
                error: None,
            },
            Err(error) => fail_closed_load(A12CalibrationLoadStatus::Corrupt, Some(error)),
        },
        Err(error) => fail_closed_load(A12CalibrationLoadStatus::Corrupt, Some(error.to_string())),
    }
}

fn revision_key(generation: u64, revision: u64) -> String {
    format!("{A12_CALIBRATION_REVISION_KEY_PREFIX}{generation}:{revision}")
}

fn parse_revision_key(row_key: &str) -> Option<(u64, u64)> {
    let suffix = row_key.strip_prefix(A12_CALIBRATION_REVISION_KEY_PREFIX)?;
    let mut parts = suffix.split(':');
    let generation = parts.next()?.parse().ok()?;
    let revision = parts.next()?.parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some((generation, revision))
}

fn state_digest(raw: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(raw.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn validate_active_pointer(pointer: &A12CalibrationActivePointer) -> Result<(), String> {
    if pointer.schema_version != A12_CALIBRATION_SCHEMA_VERSION {
        return Err(format!(
            "unsupported A12 active-pointer schema {}",
            pointer.schema_version
        ));
    }
    if pointer.generation == 0 || pointer.revision == 0 {
        return Err("A12 active pointer requires positive generation and revision".to_string());
    }
    if pointer.generation_fingerprint.is_empty() || pointer.corpus_fingerprint.is_empty() {
        return Err("A12 active-pointer fingerprints must be non-empty".to_string());
    }
    if pointer.snapshot_cutoff < 0 || pointer.activated_at < 0 {
        return Err("A12 active-pointer timestamps are inconsistent".to_string());
    }
    if pointer.row_key != revision_key(pointer.generation, pointer.revision) {
        return Err("A12 active pointer names an inconsistent revision row".to_string());
    }
    if pointer.state_digest.len() != 64
        || !pointer
            .state_digest
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        return Err("A12 active pointer has an invalid state digest".to_string());
    }
    Ok(())
}

fn parse_raw_active_pointer(
    raw: &str,
) -> Result<A12CalibrationActivePointer, (A12CalibrationLoadStatus, String)> {
    let value = serde_json::from_str::<serde_json::Value>(raw)
        .map_err(|error| (A12CalibrationLoadStatus::Corrupt, error.to_string()))?;
    let schema_version = value
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| {
            (
                A12CalibrationLoadStatus::Corrupt,
                "A12 active pointer is missing a numeric schema_version".to_string(),
            )
        })?;
    if schema_version > u64::from(A12_CALIBRATION_SCHEMA_VERSION) {
        return Err((
            A12CalibrationLoadStatus::UnsupportedSchema,
            format!(
                "A12 active-pointer schema {schema_version} is newer than binary schema {}",
                A12_CALIBRATION_SCHEMA_VERSION
            ),
        ));
    }
    if schema_version != u64::from(A12_CALIBRATION_SCHEMA_VERSION) {
        return Err((
            A12CalibrationLoadStatus::Corrupt,
            format!("A12 active-pointer schema {schema_version} is older or invalid"),
        ));
    }

    let pointer = serde_json::from_value::<A12CalibrationActivePointer>(value)
        .map_err(|error| (A12CalibrationLoadStatus::Corrupt, error.to_string()))?;
    validate_active_pointer(&pointer)
        .map_err(|error| (A12CalibrationLoadStatus::Corrupt, error))?;
    Ok(pointer)
}

fn read_active_pointer(conn: &rusqlite::Connection) -> A12ActivePointerLoad {
    let raw = match conn
        .query_row(
            "SELECT value FROM metadata WHERE key = ?1",
            params![A12_CALIBRATION_METADATA_KEY],
            |row| row.get::<_, String>(0),
        )
        .optional()
    {
        Ok(raw) => raw,
        Err(error) => {
            return A12ActivePointerLoad {
                pointer: None,
                status: A12CalibrationLoadStatus::StorageError,
                error: Some(error.to_string()),
            };
        }
    };
    let Some(raw) = raw else {
        return A12ActivePointerLoad {
            pointer: None,
            status: A12CalibrationLoadStatus::Missing,
            error: None,
        };
    };
    match parse_raw_active_pointer(&raw) {
        Ok(pointer) => A12ActivePointerLoad {
            pointer: Some(pointer),
            status: A12CalibrationLoadStatus::Loaded,
            error: None,
        },
        Err((status, error)) => A12ActivePointerLoad {
            pointer: None,
            status,
            error: Some(error),
        },
    }
}

fn parse_revision_row(row_key: &str, raw: &str) -> A12CalibrationLoad {
    let loaded = parse_raw_state(raw);
    if loaded.status != A12CalibrationLoadStatus::Loaded {
        return loaded;
    }
    if row_key != revision_key(loaded.state.generation, loaded.state.revision) {
        return fail_closed_load(
            A12CalibrationLoadStatus::Corrupt,
            Some("A12 immutable revision key does not match its payload".to_string()),
        );
    }
    loaded
}

fn pointer_matches_revision(
    pointer: &A12CalibrationActivePointer,
    row_key: &str,
    raw: &str,
    state: &A12CalibrationState,
) -> bool {
    pointer.row_key == row_key
        && pointer.generation == state.generation
        && pointer.revision == state.revision
        && pointer.generation_fingerprint == state.generation_fingerprint
        && pointer.snapshot_cutoff == state.snapshot_cutoff
        && pointer.corpus_fingerprint == state.corpus_fingerprint
        && pointer.activated_at == state.updated_at
        && pointer.state_digest == state_digest(raw)
}

fn load_revision_for_pointer(
    conn: &rusqlite::Connection,
    pointer: &A12CalibrationActivePointer,
) -> A12CalibrationLoad {
    let raw = match conn
        .query_row(
            "SELECT value FROM metadata WHERE key = ?1",
            params![pointer.row_key],
            |row| row.get::<_, String>(0),
        )
        .optional()
    {
        Ok(raw) => raw,
        Err(error) => {
            return fail_closed_load(
                A12CalibrationLoadStatus::StorageError,
                Some(error.to_string()),
            );
        }
    };
    let Some(raw) = raw else {
        return fail_closed_load(
            A12CalibrationLoadStatus::Corrupt,
            Some("A12 active pointer names a missing immutable revision".to_string()),
        );
    };
    let loaded = parse_revision_row(&pointer.row_key, &raw);
    if loaded.status != A12CalibrationLoadStatus::Loaded {
        return loaded;
    }
    if !pointer_matches_revision(pointer, &pointer.row_key, &raw, &loaded.state) {
        return fail_closed_load(
            A12CalibrationLoadStatus::Corrupt,
            Some("A12 active pointer does not match its immutable revision".to_string()),
        );
    }
    loaded
}

pub fn load_a12_calibration(conn: &rusqlite::Connection) -> A12CalibrationLoad {
    let pointer = read_active_pointer(conn);
    match pointer.status {
        A12CalibrationLoadStatus::Missing => A12CalibrationLoad {
            state: A12CalibrationState::default(),
            status: A12CalibrationLoadStatus::Missing,
            error: None,
        },
        A12CalibrationLoadStatus::Loaded => load_revision_for_pointer(
            conn,
            pointer
                .pointer
                .as_ref()
                .expect("loaded A12 pointer must contain its payload"),
        ),
        status => fail_closed_load(status, pointer.error),
    }
}

/// Load one immutable revision without changing or implicitly activating it.
pub fn load_a12_calibration_revision(
    conn: &rusqlite::Connection,
    generation: u64,
    revision: u64,
) -> A12CalibrationRevisionLoad {
    let row_key = revision_key(generation, revision);
    let raw = match conn
        .query_row(
            "SELECT value FROM metadata WHERE key = ?1",
            params![row_key],
            |row| row.get::<_, String>(0),
        )
        .optional()
    {
        Ok(raw) => raw,
        Err(error) => {
            return fail_closed_revision_load(
                A12CalibrationLoadStatus::StorageError,
                Some(error.to_string()),
            );
        }
    };
    let Some(raw) = raw else {
        return fail_closed_revision_load(A12CalibrationLoadStatus::Missing, None);
    };
    let loaded = parse_revision_row(&row_key, &raw);
    if loaded.status != A12CalibrationLoadStatus::Loaded {
        return fail_closed_revision_load(loaded.status, loaded.error);
    }
    let pointer = read_active_pointer(conn);
    let active = pointer.status == A12CalibrationLoadStatus::Loaded
        && pointer.pointer.as_ref().is_some_and(|pointer| {
            pointer_matches_revision(pointer, &row_key, &raw, &loaded.state)
        });
    A12CalibrationRevisionLoad {
        state: loaded.state,
        status: loaded.status,
        active,
        error: None,
    }
}

/// Enumerate every immutable revision. There is deliberately no retention cap:
/// old generation evidence and recluster tombstones remain auditable.
pub fn list_a12_calibration_history(
    conn: &rusqlite::Connection,
) -> rusqlite::Result<Vec<A12CalibrationHistoryEntry>> {
    let pointer = read_active_pointer(conn);
    let active_pointer = if pointer.status == A12CalibrationLoadStatus::Loaded {
        pointer.pointer
    } else {
        None
    };
    let prefix_len = i64::try_from(A12_CALIBRATION_REVISION_KEY_PREFIX.len())
        .expect("A12 revision metadata prefix length fits i64");
    let mut statement = conn.prepare(
        "SELECT key, value FROM metadata
         WHERE substr(key, 1, ?1) = ?2
         ORDER BY key",
    )?;
    let rows = statement.query_map(
        params![prefix_len, A12_CALIBRATION_REVISION_KEY_PREFIX],
        |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
    )?;
    let mut history = Vec::new();
    for row in rows {
        let (row_key, raw) = row?;
        let loaded = parse_revision_row(&row_key, &raw);
        let active = loaded.status == A12CalibrationLoadStatus::Loaded
            && active_pointer.as_ref().is_some_and(|pointer| {
                pointer_matches_revision(pointer, &row_key, &raw, &loaded.state)
            });
        let key_identity = parse_revision_key(&row_key).unwrap_or_default();
        let (
            generation,
            revision,
            generation_fingerprint,
            snapshot_cutoff,
            corpus_fingerprint,
            updated_at,
        ) = if loaded.status == A12CalibrationLoadStatus::Loaded {
            (
                loaded.state.generation,
                loaded.state.revision,
                loaded.state.generation_fingerprint.clone(),
                loaded.state.snapshot_cutoff,
                loaded.state.corpus_fingerprint.clone(),
                loaded.state.updated_at,
            )
        } else {
            (
                key_identity.0,
                key_identity.1,
                String::new(),
                0,
                String::new(),
                0,
            )
        };
        history.push(A12CalibrationHistoryEntry {
            row_key,
            generation,
            revision,
            active,
            status: loaded.status,
            generation_fingerprint,
            snapshot_cutoff,
            corpus_fingerprint,
            updated_at,
            error: loaded.error,
        });
    }
    history.sort_by(|left, right| {
        (left.generation, left.revision, &left.row_key).cmp(&(
            right.generation,
            right.revision,
            &right.row_key,
        ))
    });
    Ok(history)
}

/// Return a fresh immutable-row identity based on the numeric maximum revision
/// present in history. This remains usable after explicit corrupt-pointer
/// repair, when there is intentionally no active revision to increment.
pub fn next_a12_calibration_revision_identity(
    conn: &rusqlite::Connection,
    generation: u64,
) -> rusqlite::Result<A12CalibrationRevisionIdentity> {
    if generation == 0 {
        return Err(validation_error(
            "next A12 calibration identity requires a positive generation".to_string(),
        ));
    }
    let prefix_len = i64::try_from(A12_CALIBRATION_REVISION_KEY_PREFIX.len())
        .expect("A12 revision metadata prefix length fits i64");
    let mut statement = conn.prepare(
        "SELECT key FROM metadata
         WHERE substr(key, 1, ?1) = ?2",
    )?;
    let rows = statement.query_map(
        params![prefix_len, A12_CALIBRATION_REVISION_KEY_PREFIX],
        |row| row.get::<_, String>(0),
    )?;
    let mut max_revision = 0u64;
    for row_key in rows {
        if let Some((_, revision)) = parse_revision_key(&row_key?) {
            max_revision = max_revision.max(revision);
        }
    }
    let revision = max_revision.checked_add(1).ok_or_else(|| {
        validation_error("A12 calibration revision space is exhausted".to_string())
    })?;
    Ok(A12CalibrationRevisionIdentity {
        generation,
        revision,
    })
}

fn validation_error(error: String) -> rusqlite::Error {
    rusqlite::Error::ToSqlConversionFailure(Box::new(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        error,
    )))
}

fn valid_generation_successor(current: &A12CalibrationState, next: &A12CalibrationState) -> bool {
    if next.generation < current.generation
        || next.snapshot_cutoff < current.snapshot_cutoff
        || next.cluster_generation < current.cluster_generation
        || next.created_at < current.created_at
        || next.updated_at < current.updated_at
    {
        return false;
    }
    if next.generation == current.generation {
        next.generation_fingerprint == current.generation_fingerprint
            && next.snapshot_cutoff == current.snapshot_cutoff
            && next.corpus_fingerprint == current.corpus_fingerprint
            && next.created_at == current.created_at
    } else {
        next.generation_fingerprint != current.generation_fingerprint
    }
}

fn is_constraint_violation(error: &rusqlite::Error) -> bool {
    matches!(
        error,
        rusqlite::Error::SqliteFailure(error, _)
            if error.code == rusqlite::ErrorCode::ConstraintViolation
    )
}

fn finish_a12_savepoint(
    conn: &rusqlite::Connection,
    result: rusqlite::Result<bool>,
) -> rusqlite::Result<bool> {
    match result {
        Ok(true) => {
            conn.execute_batch("RELEASE SAVEPOINT a12_calibration_cas")?;
            Ok(true)
        }
        Ok(false) => {
            conn.execute_batch(
                "ROLLBACK TO SAVEPOINT a12_calibration_cas;
                 RELEASE SAVEPOINT a12_calibration_cas",
            )?;
            Ok(false)
        }
        Err(error) => {
            let _ = conn.execute_batch(
                "ROLLBACK TO SAVEPOINT a12_calibration_cas;
                 RELEASE SAVEPOINT a12_calibration_cas",
            );
            Err(error)
        }
    }
}

/// Atomically append an immutable revision and move the exact-byte active
/// pointer. A CAS miss rolls back both operations, so failed competitors never
/// leave orphan history rows.
#[must_use = "callers must handle an A12 calibration compare-and-swap miss"]
pub fn compare_and_swap_a12_calibration(
    conn: &rusqlite::Connection,
    state: &A12CalibrationState,
    expected_revision: u64,
) -> rusqlite::Result<bool> {
    validate_state(state).map_err(validation_error)?;
    if state.revision <= expected_revision {
        return Err(validation_error(format!(
            "new A12 revision {} must exceed expected revision {expected_revision}",
            state.revision
        )));
    }
    let serialized = serde_json::to_string(state)
        .map_err(|error| rusqlite::Error::ToSqlConversionFailure(Box::new(error)))?;
    let row_key = revision_key(state.generation, state.revision);
    let pointer = A12CalibrationActivePointer {
        schema_version: A12_CALIBRATION_SCHEMA_VERSION,
        generation: state.generation,
        revision: state.revision,
        row_key: row_key.clone(),
        generation_fingerprint: state.generation_fingerprint.clone(),
        snapshot_cutoff: state.snapshot_cutoff,
        corpus_fingerprint: state.corpus_fingerprint.clone(),
        state_digest: state_digest(&serialized),
        activated_at: state.updated_at,
    };
    let serialized_pointer = serde_json::to_string(&pointer)
        .map_err(|error| rusqlite::Error::ToSqlConversionFailure(Box::new(error)))?;

    conn.execute_batch("SAVEPOINT a12_calibration_cas")?;
    let operation = (|| -> rusqlite::Result<bool> {
        let current_raw = conn
            .query_row(
                "SELECT value FROM metadata WHERE key = ?1",
                params![A12_CALIBRATION_METADATA_KEY],
                |row| row.get::<_, String>(0),
            )
            .optional()?;

        if let Some(current_raw) = current_raw.as_deref() {
            let current_pointer = match parse_raw_active_pointer(current_raw) {
                Ok(pointer) => pointer,
                Err(_) => return Ok(false),
            };
            let current = load_revision_for_pointer(conn, &current_pointer);
            if current.status != A12CalibrationLoadStatus::Loaded
                || current.state.revision != expected_revision
                || !valid_generation_successor(&current.state, state)
            {
                return Ok(false);
            }
        } else if expected_revision != 0 {
            return Ok(false);
        }

        match conn.execute(
            "INSERT INTO metadata (key, value) VALUES (?1, ?2)",
            params![row_key, serialized],
        ) {
            Ok(1) => {}
            Ok(_) => return Ok(false),
            Err(error) if is_constraint_violation(&error) => return Ok(false),
            Err(error) => return Err(error),
        }

        let pointer_updated = if let Some(current_raw) = current_raw {
            conn.execute(
                "UPDATE metadata SET value = ?1 WHERE key = ?2 AND value = ?3",
                params![
                    serialized_pointer,
                    A12_CALIBRATION_METADATA_KEY,
                    current_raw
                ],
            )?
        } else {
            match conn.execute(
                "INSERT INTO metadata (key, value) VALUES (?1, ?2)",
                params![A12_CALIBRATION_METADATA_KEY, serialized_pointer],
            ) {
                Ok(updated) => updated,
                Err(error) if is_constraint_violation(&error) => return Ok(false),
                Err(error) => return Err(error),
            }
        };
        Ok(pointer_updated == 1)
    })();
    finish_a12_savepoint(conn, operation)
}

/// Outcome of an explicit operator repair. Automatic recalibration never
/// calls this path: an unhealthy row remains preserved until doctor (or an
/// equivalent explicit recovery flow) requests repair.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RepairCorruptA12CalibrationOutcome {
    pub deleted: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_at_delete: Option<String>,
    pub observed_status: A12CalibrationLoadStatus,
}

/// Atomically re-check and delete an A12 row only if it is still `Corrupt`.
/// Future-schema rows are never destructive-repair candidates.
#[must_use = "callers must report whether A12 calibration repair deleted a row"]
pub fn repair_corrupt_a12_calibration(
    conn: &rusqlite::Connection,
) -> rusqlite::Result<RepairCorruptA12CalibrationOutcome> {
    conn.execute_batch("BEGIN IMMEDIATE")?;
    let recovery: rusqlite::Result<RepairCorruptA12CalibrationOutcome> = (|| {
        let loaded = load_a12_calibration(conn);
        if loaded.status != A12CalibrationLoadStatus::Corrupt {
            return Ok(RepairCorruptA12CalibrationOutcome {
                deleted: 0,
                error_at_delete: None,
                observed_status: loaded.status,
            });
        }
        let deleted = conn.execute(
            "DELETE FROM metadata WHERE key = ?1",
            params![A12_CALIBRATION_METADATA_KEY],
        )?;
        Ok(RepairCorruptA12CalibrationOutcome {
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
        Err(error) => {
            let _ = conn.execute_batch("ROLLBACK");
            Err(error)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::{params, Connection};
    use std::collections::BTreeMap;

    fn conn() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
            .unwrap();
        conn
    }

    fn paired_top3() -> A12PairedTop3Stats {
        paired_top3_from_counts(14, 0, 4, 2)
    }

    fn paired_top3_from_counts(
        both_hit: u32,
        baseline_only: u32,
        treatment_only: u32,
        neither_hit: u32,
    ) -> A12PairedTop3Stats {
        let result = crate::eval::mcnemar::mcnemar_from_counts(
            both_hit,
            baseline_only,
            treatment_only,
            neither_hit,
        )
        .unwrap();
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

    fn simplex(seed: f64) -> A12FusionSimplex {
        A12FusionSimplex {
            bm25: 0.40 + seed,
            vector: 0.40 - seed,
            kg: 0.08,
            episode: 0.05,
            support: 0.04,
            diversity: 0.03,
        }
    }

    fn scope_entry(
        scope: A12CalibrationScope,
        cluster_generation: Option<u64>,
        generation: u64,
    ) -> A12ScopeEntry {
        let snapshot_cutoff = 900 + generation as i64;
        A12ScopeEntry {
            scope,
            canonical_generation: generation,
            generation_fingerprint: format!("generation-{generation}"),
            source_snapshot_fingerprint: format!("snapshot-{generation}"),
            snapshot_cutoff,
            corpus_fingerprint: format!("corpus-{generation}"),
            train_family_ess: 40,
            train_case_count: 40,
            holdout_family_ess: 20,
            simplex: simplex(0.0),
            verdict: A12CalibrationVerdict::Ship,
            noise_floor: A12_DEFAULT_NOISE_FLOOR,
            paired_top3: paired_top3(),
            provenance: A12ProvenanceCounts {
                canonical_loo: 40,
                concept_loo: 7,
                episode_loo: 3,
            },
            provenance_holdout: None,
            training_fingerprint: "train-fingerprint".to_string(),
            holdout_fingerprint: "holdout-fingerprint".to_string(),
            optimizer_fingerprint: "optimizer-v1".to_string(),
            evaluation_fingerprint: "evaluation-v1".to_string(),
            holdout_reason: "Ship: test holdout passed".to_string(),
            calibrated_at: 1_000,
            evaluated_at: 1_010,
            valid_until_exclusive: None,
            cluster_generation,
            invalidation: None,
        }
    }

    fn state(generation: u64, revision: u64) -> A12CalibrationState {
        let mut scopes = BTreeMap::new();
        for entry in [
            scope_entry(A12CalibrationScope::Global, None, generation),
            scope_entry(
                A12CalibrationScope::QueryType {
                    query_type: "semantic".to_string(),
                },
                None,
                generation,
            ),
            scope_entry(
                A12CalibrationScope::Cluster {
                    query_type: "semantic".to_string(),
                    cluster_id: 7,
                },
                Some(5),
                generation,
            ),
        ] {
            scopes.insert(entry.scope.key(), entry);
        }
        A12CalibrationState {
            schema_version: A12_CALIBRATION_SCHEMA_VERSION,
            revision,
            generation,
            generation_fingerprint: format!("generation-{generation}"),
            snapshot_cutoff: 900 + generation as i64,
            corpus_fingerprint: format!("corpus-{generation}"),
            cluster_generation: 5,
            scopes,
            created_at: 990,
            updated_at: 1_010,
            run: Some(A12CalibrationRunMetadata {
                phase: A12CalibrationPhase::Complete,
                source_snapshot_fingerprint: format!("snapshot-{generation}"),
                behavior_config_fingerprint: format!("behavior-{generation}"),
            }),
        }
    }

    #[test]
    fn a12_pending_revision_is_empty_and_never_complete() {
        let mut pending = state(12, 1);
        pending.scopes.clear();
        pending.run = Some(A12CalibrationRunMetadata {
            phase: A12CalibrationPhase::Pending,
            source_snapshot_fingerprint: "snapshot-12".to_string(),
            behavior_config_fingerprint: "behavior-12".to_string(),
        });

        assert!(pending.is_pending());
        assert!(!pending.is_complete());
        assert_eq!(pending.next_expiry_unix_ms(), None);

        let conn = conn();
        assert!(compare_and_swap_a12_calibration(&conn, &pending, 0).unwrap());
        let loaded = load_a12_calibration(&conn);
        assert_eq!(loaded.status, A12CalibrationLoadStatus::Loaded);
        assert!(loaded.state.is_pending());
        assert!(loaded.state.scopes.is_empty());
    }

    #[test]
    fn pending_and_legacy_schema_two_states_never_make_scope_current() {
        let complete = state(12, 1);
        let entry = complete.scopes["global"].clone();
        assert!(entry.is_current_for(&complete, A12_DEFAULT_NOISE_FLOOR));

        let mut pending = complete.clone();
        pending.run.as_mut().unwrap().phase = A12CalibrationPhase::Pending;
        pending.scopes.clear();
        assert!(!entry.is_current_for(&pending, A12_DEFAULT_NOISE_FLOOR));

        let mut legacy = complete;
        legacy.run = None;
        assert!(!entry.is_current_for(&legacy, A12_DEFAULT_NOISE_FLOOR));
    }

    #[test]
    fn a12_pending_revision_rejects_activation_scopes() {
        let mut pending = state(12, 1);
        pending.run = Some(A12CalibrationRunMetadata {
            phase: A12CalibrationPhase::Pending,
            source_snapshot_fingerprint: "snapshot-12".to_string(),
            behavior_config_fingerprint: "behavior-12".to_string(),
        });

        let error = compare_and_swap_a12_calibration(&conn(), &pending, 0).unwrap_err();

        assert!(error.to_string().contains("pending"));
    }

    #[test]
    fn legacy_schema_two_without_phase_loads_but_is_not_complete() {
        let conn = conn();
        let mut legacy = state(12, 1);
        legacy.run = None;
        let legacy_raw = serde_json::to_string(&legacy).unwrap();
        let row_key = revision_key(legacy.generation, legacy.revision);
        let pointer = A12CalibrationActivePointer {
            schema_version: A12_CALIBRATION_SCHEMA_VERSION,
            generation: legacy.generation,
            revision: legacy.revision,
            row_key: row_key.clone(),
            generation_fingerprint: legacy.generation_fingerprint.clone(),
            snapshot_cutoff: legacy.snapshot_cutoff,
            corpus_fingerprint: legacy.corpus_fingerprint.clone(),
            state_digest: state_digest(&legacy_raw),
            activated_at: legacy.updated_at,
        };
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES (?1, ?2)",
            params![row_key, legacy_raw],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES (?1, ?2)",
            params![
                A12_CALIBRATION_METADATA_KEY,
                serde_json::to_string(&pointer).unwrap()
            ],
        )
        .unwrap();

        let loaded = load_a12_calibration(&conn);

        assert_eq!(loaded.status, A12CalibrationLoadStatus::Loaded);
        assert!(!loaded.state.is_pending());
        assert!(!loaded.state.is_complete());
    }

    #[test]
    fn a12_complete_revision_reports_earliest_unix_ms_expiry() {
        let mut complete = state(12, 1);
        complete.run = Some(A12CalibrationRunMetadata {
            phase: A12CalibrationPhase::Complete,
            source_snapshot_fingerprint: "snapshot-12".to_string(),
            behavior_config_fingerprint: "behavior-12".to_string(),
        });
        complete
            .scopes
            .get_mut("global")
            .unwrap()
            .valid_until_exclusive = Some(1_200_000);
        complete
            .scopes
            .get_mut("semantic")
            .unwrap()
            .valid_until_exclusive = Some(1_100_000);

        assert!(complete.is_complete());
        assert_eq!(complete.next_expiry_unix_ms(), Some(1_100_000));
    }

    #[test]
    fn a12_scope_validity_boundary_expires_fail_closed_at_boundary() {
        let state = state(1, 1);
        let mut entry = state.scopes["global"].clone();
        entry.valid_until_exclusive = Some(1_100_000);

        assert!(entry.is_current_for_at(&state, A12_DEFAULT_NOISE_FLOOR, 1_099_999));
        assert!(!entry.is_current_for_at(&state, A12_DEFAULT_NOISE_FLOOR, 1_100_000));
    }

    fn raw(conn: &Connection) -> String {
        conn.query_row(
            "SELECT value FROM metadata WHERE key = ?1",
            params![A12_CALIBRATION_METADATA_KEY],
            |row| row.get(0),
        )
        .unwrap()
    }

    #[test]
    fn a12_calibration_round_trips_complete_scope_evidence() {
        let conn = conn();
        let expected = state(11, 1);

        assert!(compare_and_swap_a12_calibration(&conn, &expected, 0).unwrap());
        let loaded = load_a12_calibration(&conn);
        let canonical_expected: A12CalibrationState =
            serde_json::from_str(&serde_json::to_string(&expected).unwrap()).unwrap();

        assert_eq!(loaded.status, A12CalibrationLoadStatus::Loaded);
        assert_eq!(loaded.state, canonical_expected);
        assert!(
            loaded.state.scopes["global"].is_current_for(&loaded.state, A12_DEFAULT_NOISE_FLOOR)
        );
        assert_eq!(
            loaded.state.scopes["semantic:7"].paired_top3.treatment_only,
            4
        );
    }

    #[test]
    fn a12_calibration_missing_is_read_only_and_fail_closed() {
        let conn = conn();

        let loaded = load_a12_calibration(&conn);

        assert_eq!(loaded.status, A12CalibrationLoadStatus::Missing);
        assert!(loaded.state.scopes.is_empty());
        assert_eq!(
            conn.query_row("SELECT COUNT(*) FROM metadata", [], |row| row
                .get::<_, i64>(0))
                .unwrap(),
            0
        );
    }

    #[test]
    fn a12_calibration_storage_error_fails_closed() {
        let conn = Connection::open_in_memory().unwrap();

        let loaded = load_a12_calibration(&conn);

        assert_eq!(loaded.status, A12CalibrationLoadStatus::StorageError);
        assert!(loaded.state.scopes.is_empty());
        assert!(loaded.error.is_some());
    }

    #[test]
    fn repair_corrupt_a12_calibration_deletes_only_when_still_corrupt() {
        let conn = conn();
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES (?1, ?2)",
            params![A12_CALIBRATION_METADATA_KEY, "{not-json"],
        )
        .unwrap();

        let repaired = repair_corrupt_a12_calibration(&conn).unwrap();

        assert_eq!(repaired.deleted, 1);
        assert_eq!(repaired.observed_status, A12CalibrationLoadStatus::Corrupt);
        assert!(repaired.error_at_delete.is_some());
        assert_eq!(
            load_a12_calibration(&conn).status,
            A12CalibrationLoadStatus::Missing
        );
    }

    #[test]
    fn repair_corrupt_a12_calibration_preserves_future_schema_bytes() {
        let conn = conn();
        let future_schema = A12_CALIBRATION_SCHEMA_VERSION + 1;
        let future = format!(
            r#"{{
  "schema_version": {future_schema},
  "revision": 7,
  "future_field": ["preserve", "exactly"]
}}"#
        );
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES (?1, ?2)",
            params![A12_CALIBRATION_METADATA_KEY, &future],
        )
        .unwrap();

        let repaired = repair_corrupt_a12_calibration(&conn).unwrap();

        assert_eq!(repaired.deleted, 0);
        assert_eq!(
            repaired.observed_status,
            A12CalibrationLoadStatus::UnsupportedSchema
        );
        assert!(repaired.error_at_delete.is_none());
        assert_eq!(raw(&conn), future);
    }

    #[test]
    fn repair_corrupt_a12_calibration_preserves_peer_repaired_loaded_row() {
        let conn = conn();
        let healthy = state(10, 1);
        assert!(compare_and_swap_a12_calibration(&conn, &healthy, 0).unwrap());
        let before = raw(&conn);

        let repaired = repair_corrupt_a12_calibration(&conn).unwrap();

        assert_eq!(repaired.deleted, 0);
        assert_eq!(repaired.observed_status, A12CalibrationLoadStatus::Loaded);
        assert!(repaired.error_at_delete.is_none());
        assert_eq!(raw(&conn), before);
    }

    #[test]
    fn repair_keeps_corrupt_history_and_next_identity_can_recover_without_collision() {
        let conn = conn();
        let first = state(10, 1);
        assert!(compare_and_swap_a12_calibration(&conn, &first, 0).unwrap());
        conn.execute(
            "UPDATE metadata SET value = ?1 WHERE key = ?2",
            params!["{broken-revision", revision_key(10, 1)],
        )
        .unwrap();
        assert_eq!(
            load_a12_calibration(&conn).status,
            A12CalibrationLoadStatus::Corrupt
        );

        let repaired = repair_corrupt_a12_calibration(&conn).unwrap();
        assert_eq!(repaired.deleted, 1);
        assert_eq!(
            load_a12_calibration(&conn).status,
            A12CalibrationLoadStatus::Missing
        );
        assert_eq!(
            load_a12_calibration_revision(&conn, 10, 1).status,
            A12CalibrationLoadStatus::Corrupt
        );

        let next = next_a12_calibration_revision_identity(&conn, 11).unwrap();
        assert_eq!(
            next,
            A12CalibrationRevisionIdentity {
                generation: 11,
                revision: 2,
            }
        );
        let recovered = state(next.generation, next.revision);
        assert!(compare_and_swap_a12_calibration(&conn, &recovered, 0).unwrap());
        assert_eq!(load_a12_calibration(&conn).state.generation, 11);

        let history = list_a12_calibration_history(&conn).unwrap();
        assert_eq!(history.len(), 2);
        assert!(history.iter().any(|entry| {
            entry.generation == 10
                && entry.revision == 1
                && entry.status == A12CalibrationLoadStatus::Corrupt
                && !entry.active
        }));
    }

    #[test]
    fn a12_calibration_corrupt_row_fails_closed_without_overwrite() {
        let conn = conn();
        let corrupt = r#"{
  "schema_version": 1,
  "revision": 1,
  "generation": 10,
  "generation_fingerprint": "",
  "snapshot_cutoff": 900,
  "corpus_fingerprint": "corpus-10",
  "cluster_generation": 5,
  "scopes": {},
  "created_at": 1000,
  "updated_at": 999
}"#;
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES (?1, ?2)",
            params![A12_CALIBRATION_METADATA_KEY, corrupt],
        )
        .unwrap();

        let loaded = load_a12_calibration(&conn);
        assert_eq!(loaded.status, A12CalibrationLoadStatus::Corrupt);
        assert!(loaded.state.scopes.is_empty());

        let replacement = state(11, 2);
        assert!(!compare_and_swap_a12_calibration(&conn, &replacement, 1).unwrap());
        assert_eq!(raw(&conn), corrupt);
    }

    #[test]
    fn a12_calibration_forged_ship_statistics_fail_closed() {
        let conn = conn();
        let mut forged = state(10, 1);
        let global = forged.scopes.get_mut("global").unwrap();
        global.paired_top3.both_hit = 12;
        global.paired_top3.baseline_only = 2;
        global.paired_top3.diff_point = 0.1;
        // These bounds claim Ship but do not match the persisted contingency
        // table's deterministic Wald interval.
        global.paired_top3.ci_lower = -0.01;
        global.paired_top3.ci_upper = 0.21;
        let forged_raw = serde_json::to_string(&forged).unwrap();
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES (?1, ?2)",
            params![A12_CALIBRATION_METADATA_KEY, forged_raw],
        )
        .unwrap();

        let loaded = load_a12_calibration(&conn);

        assert_eq!(loaded.status, A12CalibrationLoadStatus::Corrupt);
        assert!(loaded.state.scopes.is_empty());
    }

    #[test]
    fn a12_calibration_future_schema_bytes_are_preserved() {
        let conn = conn();
        let future_schema = A12_CALIBRATION_SCHEMA_VERSION + 1;
        let future = format!(
            r#"{{
  "schema_version": {future_schema},
  "revision": 1,
  "generation": 99,
  "future_scope": {{"kind":"new_variant"}}
}}"#
        );
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES (?1, ?2)",
            params![A12_CALIBRATION_METADATA_KEY, &future],
        )
        .unwrap();

        let loaded = load_a12_calibration(&conn);
        assert_eq!(loaded.status, A12CalibrationLoadStatus::UnsupportedSchema);
        assert!(loaded.state.scopes.is_empty());

        let replacement = state(100, 2);
        assert!(!compare_and_swap_a12_calibration(&conn, &replacement, 1).unwrap());
        assert_eq!(raw(&conn), future);
    }

    #[test]
    fn a12_calibration_cas_rejects_revision_conflict() {
        let conn = conn();
        let first = state(10, 1);
        assert!(compare_and_swap_a12_calibration(&conn, &first, 0).unwrap());

        let second = state(10, 2);
        assert!(!compare_and_swap_a12_calibration(&conn, &second, 0).unwrap());
        assert_eq!(load_a12_calibration(&conn).state.revision, 1);
        assert!(compare_and_swap_a12_calibration(&conn, &second, 1).unwrap());
        assert_eq!(load_a12_calibration(&conn).state.revision, 2);
    }

    #[test]
    fn a12_calibration_newer_generation_cannot_be_replaced_by_stale_work() {
        let conn = conn();
        let first = state(10, 1);
        assert!(compare_and_swap_a12_calibration(&conn, &first, 0).unwrap());

        let stale = state(9, 2);
        assert!(!compare_and_swap_a12_calibration(&conn, &stale, 1).unwrap());

        let mut conflicting_same_generation = state(10, 2);
        conflicting_same_generation.corpus_fingerprint = "other-corpus".to_string();
        for entry in conflicting_same_generation.scopes.values_mut() {
            entry.corpus_fingerprint = "other-corpus".to_string();
        }
        assert!(!compare_and_swap_a12_calibration(&conn, &conflicting_same_generation, 1).unwrap());

        let newer = state(11, 2);
        assert!(compare_and_swap_a12_calibration(&conn, &newer, 1).unwrap());
        assert_eq!(load_a12_calibration(&conn).state.generation, 11);

        let old_retry = state(10, 3);
        assert!(!compare_and_swap_a12_calibration(&conn, &old_retry, 2).unwrap());
        assert_eq!(load_a12_calibration(&conn).state.generation, 11);
    }

    #[test]
    fn a12_generation_history_is_immutable_and_only_active_revision_can_activate() {
        let conn = conn();
        let generation_ten = state(10, 1);
        assert!(compare_and_swap_a12_calibration(&conn, &generation_ten, 0).unwrap());

        let mut tombstoned = generation_ten.clone();
        tombstoned.revision = 2;
        tombstoned
            .invalidate_cluster_scopes_after_recluster(6, 1_200)
            .unwrap();
        assert!(compare_and_swap_a12_calibration(&conn, &tombstoned, 1).unwrap());

        let mut generation_eleven = state(11, 3);
        generation_eleven.cluster_generation = 6;
        generation_eleven
            .scopes
            .get_mut("semantic:7")
            .unwrap()
            .cluster_generation = Some(6);
        generation_eleven.updated_at = 1_300;
        assert!(compare_and_swap_a12_calibration(&conn, &generation_eleven, 2).unwrap());

        let active = load_a12_calibration(&conn);
        assert_eq!(active.status, A12CalibrationLoadStatus::Loaded);
        assert_eq!(active.state.generation, 11);
        assert_eq!(active.state.revision, 3);

        let history = list_a12_calibration_history(&conn).unwrap();
        assert_eq!(history.len(), 3);
        assert_eq!(history.iter().filter(|entry| entry.active).count(), 1);
        assert!(history
            .iter()
            .any(|entry| entry.generation == 10 && entry.revision == 1 && !entry.active));
        assert!(history
            .iter()
            .any(|entry| entry.generation == 10 && entry.revision == 2 && !entry.active));
        assert!(history
            .iter()
            .any(|entry| entry.generation == 11 && entry.revision == 3 && entry.active));

        let old = load_a12_calibration_revision(&conn, 10, 1);
        assert_eq!(old.status, A12CalibrationLoadStatus::Loaded);
        assert!(!old.active);
        assert!(!old.can_activate());
        assert_eq!(old.state.scopes["global"].canonical_generation, 10);
        assert!(!old.state.scopes["global"].is_current_for(&active.state, A12_DEFAULT_NOISE_FLOOR));

        let old_tombstone = load_a12_calibration_revision(&conn, 10, 2);
        assert_eq!(old_tombstone.status, A12CalibrationLoadStatus::Loaded);
        assert!(!old_tombstone.active);
        assert!(old_tombstone.state.scopes["semantic:7"]
            .invalidation
            .is_some());

        let current = load_a12_calibration_revision(&conn, 11, 3);
        assert!(current.active);
        assert!(current.can_activate());
    }

    #[test]
    fn a12_history_is_sorted_by_numeric_generation_and_revision() {
        let conn = conn();
        let generation_two = state(2, 9);
        assert!(compare_and_swap_a12_calibration(&conn, &generation_two, 0).unwrap());
        let generation_ten = state(10, 10);
        assert!(compare_and_swap_a12_calibration(&conn, &generation_ten, 9).unwrap());

        let identities = list_a12_calibration_history(&conn)
            .unwrap()
            .into_iter()
            .map(|entry| (entry.generation, entry.revision))
            .collect::<Vec<_>>();

        assert_eq!(identities, vec![(2, 9), (10, 10)]);
    }

    #[test]
    fn copied_old_generation_ship_scope_cannot_masquerade_as_new_generation() {
        let conn = conn();
        let old = state(10, 1);
        assert!(compare_and_swap_a12_calibration(&conn, &old, 0).unwrap());

        let mut newer = state(11, 2);
        newer
            .scopes
            .insert("global".to_string(), old.scopes["global"].clone());

        assert!(compare_and_swap_a12_calibration(&conn, &newer, 1).is_err());
        assert_eq!(load_a12_calibration(&conn).state.generation, 10);
        assert_eq!(list_a12_calibration_history(&conn).unwrap().len(), 1);
    }

    #[test]
    fn persisted_noise_floor_controls_verdict_and_runtime_drift_is_explicit() {
        let conn = conn();
        let mut calibrated = state(10, 1);
        let global = calibrated.scopes.get_mut("global").unwrap();
        global.noise_floor = 0.03;
        global.holdout_family_ess = 200;
        global.paired_top3 = paired_top3_from_counts(172, 8, 12, 8);
        global.verdict = A12CalibrationVerdict::Ship;
        assert!(global.paired_top3.ci_lower < -A12_DEFAULT_NOISE_FLOOR);
        assert!(global.paired_top3.ci_lower >= -global.noise_floor);

        assert!(compare_and_swap_a12_calibration(&conn, &calibrated, 0).unwrap());
        let loaded = load_a12_calibration(&conn);
        let global = &loaded.state.scopes["global"];
        assert!(global.matches_noise_floor(0.03));
        assert!(!global.matches_noise_floor(A12_DEFAULT_NOISE_FLOOR));
        assert!(global.is_current_for(&loaded.state, 0.03));
        assert!(!global.is_current_for(&loaded.state, A12_DEFAULT_NOISE_FLOOR));
    }

    #[test]
    fn zero_noise_floor_is_rejected_before_any_revision_is_written() {
        let conn = conn();
        let mut calibrated = state(10, 1);
        calibrated.scopes.get_mut("global").unwrap().noise_floor = 0.0;

        assert!(compare_and_swap_a12_calibration(&conn, &calibrated, 0).is_err());
        assert_eq!(
            load_a12_calibration(&conn).status,
            A12CalibrationLoadStatus::Missing
        );
        assert!(list_a12_calibration_history(&conn).unwrap().is_empty());
    }

    #[test]
    fn forged_p_value_fails_closed_and_zero_case_recomputes_to_one() {
        let zero = paired_top3_from_counts(0, 0, 0, 0);
        assert_eq!(zero.recomputed_mcnemar().unwrap().p_value, 1.0);

        let conn = conn();
        let mut forged = state(10, 1);
        forged.scopes.get_mut("global").unwrap().paired_top3.p_value = 0.99;

        assert!(compare_and_swap_a12_calibration(&conn, &forged, 0).is_err());
        assert_eq!(
            load_a12_calibration(&conn).status,
            A12CalibrationLoadStatus::Missing
        );
    }

    #[test]
    fn a12_recluster_invalidates_only_cluster_scopes_and_keeps_diagnostics() {
        let mut state = state(10, 1);
        let cluster_simplex = state.scopes["semantic:7"].simplex;

        let invalidated = state
            .invalidate_cluster_scopes_after_recluster(6, 1_100)
            .unwrap();

        assert_eq!(invalidated, 1);
        assert_eq!(state.cluster_generation, 6);
        assert!(state.scopes["global"].is_current_for(&state, A12_DEFAULT_NOISE_FLOOR));
        assert!(state.scopes["semantic"].is_current_for(&state, A12_DEFAULT_NOISE_FLOOR));
        assert!(!state.scopes["semantic:7"].is_current_for(&state, A12_DEFAULT_NOISE_FLOOR));
        assert_eq!(state.scopes["semantic:7"].simplex, cluster_simplex);
        assert_eq!(
            state.scopes["semantic:7"].invalidation,
            Some(A12ScopeInvalidation {
                reason: A12ScopeInvalidationReason::Reclustered,
                from_cluster_generation: 5,
                to_cluster_generation: 6,
                invalidated_at: 1_100,
            })
        );
        assert!(state
            .invalidate_cluster_scopes_after_recluster(6, 1_101)
            .is_err());

        // A later recluster keeps the first invalidation transition readable;
        // already-invalid evidence is not counted again.
        assert_eq!(
            state
                .invalidate_cluster_scopes_after_recluster(7, 1_200)
                .unwrap(),
            0
        );
        assert_eq!(state.cluster_generation, 7);
        assert_eq!(
            state.scopes["semantic:7"]
                .invalidation
                .unwrap()
                .to_cluster_generation,
            6
        );

        let conn = conn();
        assert!(compare_and_swap_a12_calibration(&conn, &state, 0).unwrap());
        let canonical_expected: A12CalibrationState =
            serde_json::from_str(&serde_json::to_string(&state).unwrap()).unwrap();
        assert_eq!(load_a12_calibration(&conn).state, canonical_expected);
    }

    #[test]
    fn cluster_scope_currentness_fails_closed_on_unvalidated_generation_drift() {
        let state = state(10, 1);
        let mut stale_cluster = state.scopes["semantic:7"].clone();
        stale_cluster.cluster_generation = Some(4);

        assert!(!stale_cluster.is_current_for(&state, A12_DEFAULT_NOISE_FLOOR));
    }

    fn provenance_cells(
        family_count: u64,
        both_hit: u64,
        baseline_only: u64,
        treatment_only: u64,
        neither_hit: u64,
    ) -> A12ProvenanceHoldoutCells {
        A12ProvenanceHoldoutCells {
            family_count,
            both_hit,
            baseline_only,
            treatment_only,
            neither_hit,
        }
    }

    #[test]
    fn provenance_direction_conflict_requires_opposite_discordant_signs() {
        // Opposite net signs: canonical favors the treatment, concept the
        // baseline. One discordant pair in each direction already conflicts.
        let conflicting = A12ProvenanceHoldoutStats {
            canonical_loo: provenance_cells(3, 1, 0, 2, 0),
            concept_loo: provenance_cells(2, 0, 1, 0, 1),
            episode_loo: A12ProvenanceHoldoutCells::default(),
        };
        assert!(conflicting.direction_conflict());

        // Same net sign everywhere: no conflict.
        let aligned = A12ProvenanceHoldoutStats {
            canonical_loo: provenance_cells(3, 1, 0, 2, 0),
            concept_loo: provenance_cells(2, 1, 0, 1, 0),
            episode_loo: A12ProvenanceHoldoutCells::default(),
        };
        assert!(!aligned.direction_conflict());

        // A source without discordant pairs contributes no direction.
        let one_sided = A12ProvenanceHoldoutStats {
            canonical_loo: provenance_cells(3, 1, 0, 2, 0),
            concept_loo: provenance_cells(2, 1, 0, 0, 1),
            episode_loo: A12ProvenanceHoldoutCells::default(),
        };
        assert!(!one_sided.direction_conflict());

        // Balanced discordant pairs (baseline_only == treatment_only) have no
        // net sign and therefore no direction.
        let balanced = A12ProvenanceHoldoutStats {
            canonical_loo: provenance_cells(3, 1, 1, 1, 0),
            concept_loo: provenance_cells(2, 1, 0, 1, 0),
            episode_loo: A12ProvenanceHoldoutCells::default(),
        };
        assert!(!balanced.direction_conflict());
    }

    #[test]
    fn provenance_holdout_stats_round_trip_and_legacy_rows_load_none() {
        let stats = A12ProvenanceHoldoutStats {
            canonical_loo: provenance_cells(3, 1, 0, 2, 0),
            concept_loo: provenance_cells(2, 0, 1, 0, 1),
            episode_loo: A12ProvenanceHoldoutCells::default(),
        };
        let mut state = state(3, 1);
        state.scopes.get_mut("global").unwrap().provenance_holdout = Some(stats);

        let conn = conn();
        assert!(compare_and_swap_a12_calibration(&conn, &state, 0).unwrap());
        let loaded = load_a12_calibration(&conn);
        assert_eq!(loaded.status, A12CalibrationLoadStatus::Loaded);
        assert_eq!(
            loaded.state.scopes["global"].provenance_holdout,
            Some(stats)
        );

        // Rows sealed before this field existed carry no key and load None.
        let mut legacy_json = serde_json::to_value(&state.scopes["global"]).unwrap();
        assert!(legacy_json
            .as_object_mut()
            .unwrap()
            .remove("provenance_holdout")
            .is_some());
        let legacy: A12ScopeEntry = serde_json::from_value(legacy_json).unwrap();
        assert_eq!(legacy.provenance_holdout, None);
    }
}
