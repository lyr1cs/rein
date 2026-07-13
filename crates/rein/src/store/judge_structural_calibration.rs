//! Durable, versioned state for deterministic judge structural anchors.

use rusqlite::{params, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub use crate::judge::contract::JudgeStructuralStatus;
pub use crate::store::adaptive::JudgeStructuralProbeKind;
use crate::store::adaptive::JudgeSurface;

pub const JUDGE_STRUCTURAL_CALIBRATION_METADATA_KEY: &str = "judge_structural_calibration";
pub const JUDGE_STRUCTURAL_CALIBRATION_SCHEMA_VERSION: u32 = 1;

fn default_schema_version() -> u32 {
    JUDGE_STRUCTURAL_CALIBRATION_SCHEMA_VERSION
}

/// One surface's current sealed probe run. Synthesis and concept-summary use
/// independent rows inside the same CAS envelope because their rubrics differ.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JudgeStructuralSurfaceState {
    #[serde(default)]
    pub run_id: Option<String>,
    #[serde(default)]
    pub probe_set_version: String,
    #[serde(default)]
    pub model_fingerprint: String,
    #[serde(default)]
    pub rubric_fingerprint: String,
    /// Per-kind SHA-256 hashes of opaque tokens minted by the in-crate probe
    /// runner. A token exposed by one event cannot attest any other kind.
    #[serde(default)]
    pub run_token_hashes: BTreeMap<JudgeStructuralProbeKind, String>,
    #[serde(default)]
    pub seen_kinds: BTreeSet<JudgeStructuralProbeKind>,
    #[serde(default)]
    pub failed_kinds: BTreeSet<JudgeStructuralProbeKind>,
    #[serde(default)]
    pub run_started_at: i64,
    #[serde(default)]
    pub last_probe_at: i64,
    #[serde(default)]
    pub completed_at: Option<i64>,
    #[serde(default = "default_structural_status")]
    pub status: JudgeStructuralStatus,
    #[serde(default)]
    pub alert_count: u64,
}

fn default_structural_status() -> JudgeStructuralStatus {
    JudgeStructuralStatus::Unknown
}

impl Default for JudgeStructuralSurfaceState {
    fn default() -> Self {
        Self {
            run_id: None,
            probe_set_version: String::new(),
            model_fingerprint: String::new(),
            rubric_fingerprint: String::new(),
            run_token_hashes: BTreeMap::new(),
            seen_kinds: BTreeSet::new(),
            failed_kinds: BTreeSet::new(),
            run_started_at: 0,
            last_probe_at: 0,
            completed_at: None,
            status: JudgeStructuralStatus::Unknown,
            alert_count: 0,
        }
    }
}

impl JudgeStructuralSurfaceState {
    /// Project freshness without mutating durable state. A completed Ready run
    /// is fresh through exactly two configured intervals.
    pub fn status_at(&self, now: i64, interval_secs: i64) -> JudgeStructuralStatus {
        if interval_secs <= 0 || now < 0 {
            return JudgeStructuralStatus::Unknown;
        }
        if self.status != JudgeStructuralStatus::Ready {
            return self.status;
        }
        let Some(completed_at) = self.completed_at else {
            return JudgeStructuralStatus::Unknown;
        };
        if completed_at <= 0 || now < completed_at {
            return JudgeStructuralStatus::Unknown;
        }
        let freshness_window = interval_secs.saturating_mul(2);
        if now.saturating_sub(completed_at) > freshness_window {
            JudgeStructuralStatus::Stale
        } else {
            JudgeStructuralStatus::Ready
        }
    }

    /// Project freshness and bind readiness to the exact model, rubric, and
    /// probe-set fingerprints expected by the current runtime.
    pub fn status_for(
        &self,
        now: i64,
        interval_secs: i64,
        model_fingerprint: &str,
        rubric_fingerprint: &str,
        probe_set_version: &str,
    ) -> JudgeStructuralStatus {
        if matches!(
            self.status,
            JudgeStructuralStatus::Disabled
                | JudgeStructuralStatus::Corrupt
                | JudgeStructuralStatus::Unknown
        ) {
            return self.status;
        }
        if model_fingerprint.is_empty()
            || rubric_fingerprint.is_empty()
            || probe_set_version.is_empty()
            || self.model_fingerprint.is_empty()
            || self.rubric_fingerprint.is_empty()
            || self.probe_set_version.is_empty()
        {
            return JudgeStructuralStatus::Unknown;
        }
        if self.model_fingerprint != model_fingerprint
            || self.rubric_fingerprint != rubric_fingerprint
            || self.probe_set_version != probe_set_version
        {
            return JudgeStructuralStatus::FingerprintMismatch;
        }
        self.status_at(now, interval_secs)
    }
}

/// Atomic state envelope and independent replay watermark for structural
/// anchors. It never reuses the human/runtime judge-pair watermark.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JudgeStructuralCalibrationState {
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    #[serde(default)]
    pub revision: u64,
    #[serde(default)]
    pub last_event_id: i64,
    #[serde(default)]
    pub synthesis: JudgeStructuralSurfaceState,
    #[serde(default)]
    pub concept_summary: JudgeStructuralSurfaceState,
    #[serde(default)]
    pub updated_at: i64,
}

impl Default for JudgeStructuralCalibrationState {
    fn default() -> Self {
        Self {
            schema_version: JUDGE_STRUCTURAL_CALIBRATION_SCHEMA_VERSION,
            revision: 0,
            last_event_id: 0,
            synthesis: JudgeStructuralSurfaceState::default(),
            concept_summary: JudgeStructuralSurfaceState::default(),
            updated_at: 0,
        }
    }
}

impl JudgeStructuralCalibrationState {
    pub fn surface(&self, surface: JudgeSurface) -> &JudgeStructuralSurfaceState {
        match surface {
            JudgeSurface::Synthesis => &self.synthesis,
            JudgeSurface::ConceptSummary => &self.concept_summary,
        }
    }

    pub fn surface_mut(&mut self, surface: JudgeSurface) -> &mut JudgeStructuralSurfaceState {
        match surface {
            JudgeSurface::Synthesis => &mut self.synthesis,
            JudgeSurface::ConceptSummary => &mut self.concept_summary,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum JudgeStructuralCalibrationLoadStatus {
    Missing,
    Loaded,
    Corrupt,
    UnsupportedSchema,
    StorageError,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct JudgeStructuralCalibrationLoad {
    pub state: JudgeStructuralCalibrationState,
    pub status: JudgeStructuralCalibrationLoadStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

fn fail_closed_state(status: JudgeStructuralStatus) -> JudgeStructuralCalibrationState {
    JudgeStructuralCalibrationState {
        synthesis: JudgeStructuralSurfaceState {
            status,
            ..JudgeStructuralSurfaceState::default()
        },
        concept_summary: JudgeStructuralSurfaceState {
            status,
            ..JudgeStructuralSurfaceState::default()
        },
        ..JudgeStructuralCalibrationState::default()
    }
}

fn validate_surface(surface: &JudgeStructuralSurfaceState) -> Result<(), String> {
    if surface.run_started_at < 0
        || surface.last_probe_at < 0
        || surface.completed_at.is_some_and(|value| value <= 0)
    {
        return Err("structural calibration timestamps must be non-negative".to_string());
    }
    if surface.last_probe_at > 0 && surface.last_probe_at < surface.run_started_at {
        return Err("last probe timestamp precedes run start".to_string());
    }
    if surface.completed_at.is_some_and(|completed_at| {
        completed_at < surface.run_started_at || completed_at > surface.last_probe_at
    }) {
        return Err("completion timestamp falls outside the observed run".to_string());
    }
    if !surface.failed_kinds.is_subset(&surface.seen_kinds) {
        return Err("failed probe kinds must be a subset of seen kinds".to_string());
    }
    if !surface.run_token_hashes.is_empty()
        && (surface.run_token_hashes.len() != JudgeStructuralProbeKind::ALL.len()
            || JudgeStructuralProbeKind::ALL.iter().any(|kind| {
                surface
                    .run_token_hashes
                    .get(kind)
                    .is_none_or(|hash| hash.len() != 64)
            }))
    {
        return Err("probe-run token hashes must cover all four kinds".to_string());
    }
    if surface.status == JudgeStructuralStatus::Ready
        && (surface.run_id.as_deref().is_none_or(str::is_empty)
            || surface.probe_set_version.is_empty()
            || surface.model_fingerprint.is_empty()
            || surface.rubric_fingerprint.is_empty()
            || surface.run_token_hashes.len() != JudgeStructuralProbeKind::ALL.len()
            || surface.completed_at.is_none()
            || surface.seen_kinds.len() != JudgeStructuralProbeKind::ALL.len()
            || !surface.failed_kinds.is_empty())
    {
        return Err("ready structural calibration requires a complete passing run".to_string());
    }
    if surface.status == JudgeStructuralStatus::Failed && surface.failed_kinds.is_empty() {
        return Err("failed structural calibration requires a failed probe kind".to_string());
    }
    Ok(())
}

fn validate_state(state: &JudgeStructuralCalibrationState) -> Result<(), String> {
    if state.schema_version != JUDGE_STRUCTURAL_CALIBRATION_SCHEMA_VERSION {
        return Err(format!(
            "unsupported judge structural calibration schema {}",
            state.schema_version
        ));
    }
    if state.last_event_id < 0 || state.updated_at < 0 {
        return Err("state timestamps and event watermark must be non-negative".to_string());
    }
    validate_surface(&state.synthesis)?;
    validate_surface(&state.concept_summary)
}

pub fn load_judge_structural_calibration(
    conn: &rusqlite::Connection,
) -> JudgeStructuralCalibrationLoad {
    let raw = match conn
        .query_row(
            "SELECT value FROM metadata WHERE key = ?1",
            params![JUDGE_STRUCTURAL_CALIBRATION_METADATA_KEY],
            |row| row.get::<_, String>(0),
        )
        .optional()
    {
        Ok(raw) => raw,
        Err(error) => {
            return JudgeStructuralCalibrationLoad {
                state: fail_closed_state(JudgeStructuralStatus::Unknown),
                status: JudgeStructuralCalibrationLoadStatus::StorageError,
                error: Some(error.to_string()),
            };
        }
    };
    let Some(raw) = raw else {
        return JudgeStructuralCalibrationLoad {
            state: JudgeStructuralCalibrationState::default(),
            status: JudgeStructuralCalibrationLoadStatus::Missing,
            error: None,
        };
    };

    if let Ok(value) = serde_json::from_str::<serde_json::Value>(&raw) {
        if value
            .get("schema_version")
            .and_then(serde_json::Value::as_u64)
            .is_some_and(|schema| schema > u64::from(JUDGE_STRUCTURAL_CALIBRATION_SCHEMA_VERSION))
        {
            return JudgeStructuralCalibrationLoad {
                state: fail_closed_state(JudgeStructuralStatus::Unknown),
                status: JudgeStructuralCalibrationLoadStatus::UnsupportedSchema,
                error: Some("judge structural calibration row uses a future schema".to_string()),
            };
        }
    }

    match serde_json::from_str::<JudgeStructuralCalibrationState>(&raw) {
        Ok(state) => match validate_state(&state) {
            Ok(()) => JudgeStructuralCalibrationLoad {
                state,
                status: JudgeStructuralCalibrationLoadStatus::Loaded,
                error: None,
            },
            Err(error) => JudgeStructuralCalibrationLoad {
                state: fail_closed_state(JudgeStructuralStatus::Corrupt),
                status: JudgeStructuralCalibrationLoadStatus::Corrupt,
                error: Some(error),
            },
        },
        Err(error) => JudgeStructuralCalibrationLoad {
            state: fail_closed_state(JudgeStructuralStatus::Corrupt),
            status: JudgeStructuralCalibrationLoadStatus::Corrupt,
            error: Some(error.to_string()),
        },
    }
}

fn validation_error(error: String) -> rusqlite::Error {
    rusqlite::Error::ToSqlConversionFailure(Box::new(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        error,
    )))
}

#[must_use = "callers must handle a compare-and-swap miss"]
pub(crate) fn compare_and_swap_judge_structural_calibration(
    conn: &rusqlite::Connection,
    state: &JudgeStructuralCalibrationState,
    expected_revision: u64,
) -> rusqlite::Result<bool> {
    validate_state(state).map_err(validation_error)?;
    if state.revision <= expected_revision {
        return Err(validation_error(format!(
            "new revision {} must exceed expected revision {}",
            state.revision, expected_revision
        )));
    }
    let raw = serde_json::to_string(state)
        .map_err(|error| rusqlite::Error::ToSqlConversionFailure(Box::new(error)))?;
    let updated = conn.execute(
        "UPDATE metadata
            SET value = ?1
          WHERE key = ?2
            AND json_valid(value)
            AND COALESCE(json_extract(value, '$.revision'), 0) = ?3
            AND COALESCE(json_extract(value, '$.schema_version'), ?4) = ?4",
        params![
            raw,
            JUDGE_STRUCTURAL_CALIBRATION_METADATA_KEY,
            expected_revision,
            JUDGE_STRUCTURAL_CALIBRATION_SCHEMA_VERSION,
        ],
    )?;
    if updated == 1 {
        return Ok(true);
    }
    let exists: bool = conn.query_row(
        "SELECT COUNT(*) > 0 FROM metadata WHERE key = ?1",
        params![JUDGE_STRUCTURAL_CALIBRATION_METADATA_KEY],
        |row| row.get(0),
    )?;
    if exists || expected_revision != 0 {
        return Ok(false);
    }
    match conn.execute(
        "INSERT INTO metadata (key, value) VALUES (?1, ?2)",
        params![JUDGE_STRUCTURAL_CALIBRATION_METADATA_KEY, raw],
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

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::{params, Connection};

    fn conn() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
            .unwrap();
        conn
    }

    fn ready_surface(run_id: &str) -> JudgeStructuralSurfaceState {
        JudgeStructuralSurfaceState {
            run_id: Some(run_id.to_string()),
            probe_set_version: "judge-anchors-v1".to_string(),
            model_fingerprint: "model-a".to_string(),
            rubric_fingerprint: "rubric-a".to_string(),
            run_token_hashes: JudgeStructuralProbeKind::ALL
                .into_iter()
                .map(|kind| {
                    (
                        kind,
                        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
                            .to_string(),
                    )
                })
                .collect(),
            seen_kinds: JudgeStructuralProbeKind::ALL.into_iter().collect(),
            failed_kinds: Default::default(),
            run_started_at: 900,
            last_probe_at: 1_000,
            completed_at: Some(1_000),
            status: JudgeStructuralStatus::Ready,
            alert_count: 0,
        }
    }

    #[test]
    fn judge_structural_calibration_round_trips_both_surfaces() {
        let conn = conn();
        let state = JudgeStructuralCalibrationState {
            revision: 1,
            last_event_id: 42,
            synthesis: ready_surface("run-synthesis"),
            concept_summary: ready_surface("run-concept"),
            updated_at: 1_000,
            ..JudgeStructuralCalibrationState::default()
        };

        assert!(compare_and_swap_judge_structural_calibration(&conn, &state, 0).unwrap());
        let loaded = load_judge_structural_calibration(&conn);
        assert_eq!(loaded.status, JudgeStructuralCalibrationLoadStatus::Loaded);
        assert_eq!(loaded.state, state);
    }

    #[test]
    fn judge_structural_calibration_cas_rejects_stale_revision() {
        let conn = conn();
        let first = JudgeStructuralCalibrationState {
            revision: 1,
            ..JudgeStructuralCalibrationState::default()
        };
        assert!(compare_and_swap_judge_structural_calibration(&conn, &first, 0).unwrap());

        let second = JudgeStructuralCalibrationState {
            revision: 2,
            ..JudgeStructuralCalibrationState::default()
        };
        assert!(!compare_and_swap_judge_structural_calibration(&conn, &second, 0).unwrap());
        assert!(compare_and_swap_judge_structural_calibration(&conn, &second, 1).unwrap());
    }

    #[test]
    fn corrupt_row_fails_closed_without_mutation() {
        let conn = conn();
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES (?1, ?2)",
            params![JUDGE_STRUCTURAL_CALIBRATION_METADATA_KEY, "{not-json"],
        )
        .unwrap();

        let loaded = load_judge_structural_calibration(&conn);
        assert_eq!(loaded.status, JudgeStructuralCalibrationLoadStatus::Corrupt);
        assert_eq!(
            loaded.state.synthesis.status,
            JudgeStructuralStatus::Corrupt
        );
        let raw: String = conn
            .query_row(
                "SELECT value FROM metadata WHERE key = ?1",
                params![JUDGE_STRUCTURAL_CALIBRATION_METADATA_KEY],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(raw, "{not-json");
    }

    #[test]
    fn future_schema_is_preserved_and_cas_cannot_overwrite_it() {
        let conn = conn();
        let future = serde_json::json!({
            "schema_version": JUDGE_STRUCTURAL_CALIBRATION_SCHEMA_VERSION + 1,
            "revision": 0,
            "future_status": "new_variant"
        })
        .to_string();
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES (?1, ?2)",
            params![JUDGE_STRUCTURAL_CALIBRATION_METADATA_KEY, &future],
        )
        .unwrap();

        let loaded = load_judge_structural_calibration(&conn);
        assert_eq!(
            loaded.status,
            JudgeStructuralCalibrationLoadStatus::UnsupportedSchema
        );
        let replacement = JudgeStructuralCalibrationState {
            revision: 1,
            ..JudgeStructuralCalibrationState::default()
        };
        assert!(!compare_and_swap_judge_structural_calibration(&conn, &replacement, 0).unwrap());
        let raw: String = conn
            .query_row(
                "SELECT value FROM metadata WHERE key = ?1",
                params![JUDGE_STRUCTURAL_CALIBRATION_METADATA_KEY],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(raw, future);
    }

    #[test]
    fn status_at_marks_ready_state_stale_after_two_intervals() {
        let surface = ready_surface("run-1");
        assert_eq!(surface.status_at(1_199, 100), JudgeStructuralStatus::Ready);
        assert_eq!(surface.status_at(1_201, 100), JudgeStructuralStatus::Stale);
        assert_eq!(surface.status_at(1_201, 0), JudgeStructuralStatus::Unknown);
    }

    #[test]
    fn fingerprint_mismatch_invalidates_ready_state() {
        let surface = ready_surface("run-1");
        assert_eq!(
            surface.status_for(1_100, 100, "model-b", "rubric-a", "judge-anchors-v1"),
            JudgeStructuralStatus::FingerprintMismatch
        );
        assert_eq!(
            surface.status_for(1_100, 100, "model-a", "rubric-a", "judge-anchors-v1"),
            JudgeStructuralStatus::Ready
        );
    }

    #[test]
    fn fail_closed_status_precedes_fingerprint_projection() {
        let surface = JudgeStructuralSurfaceState {
            status: JudgeStructuralStatus::Corrupt,
            ..JudgeStructuralSurfaceState::default()
        };
        assert_eq!(
            surface.status_for(1_100, 100, "model-a", "rubric-a", "judge-anchors-v1"),
            JudgeStructuralStatus::Corrupt
        );
    }
}
