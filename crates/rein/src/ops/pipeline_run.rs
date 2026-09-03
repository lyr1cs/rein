//! Stage-level observability for `run_adaptive_pipeline`.
//!
//! One `metadata` row (`adaptive_pipeline_last_run`) records the most recent
//! pipeline pass: when it started, which stages ran, how long each took, and
//! how it ended. The row is rewritten after every stage, so a pass that is
//! killed mid-way is visible as `running` with a stale `started_at` and the
//! last stage that completed.
//!
//! The key is deliberately outside the A12 input-epoch trigger set
//! (`adaptive_state`, `rerank_weights`, `embedding_write_seq`,
//! `survival_curve:*`) and outside the A12 local recall snapshot identity, so
//! recording progress can never invalidate a calibration that is in flight.

use std::cell::RefCell;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::store::SqliteStore;

/// Metadata key holding the last pipeline run record.
pub const ADAPTIVE_PIPELINE_LAST_RUN_KEY: &str = "adaptive_pipeline_last_run";

/// Schema version of [`PipelineRunRecord`]. Bump on breaking layout changes.
pub const PIPELINE_RUN_SCHEMA_VERSION: u32 = 1;

/// A run that still reports `running` after this many milliseconds is
/// treated as abandoned by doctor (six hours).
pub const PIPELINE_RUN_STALE_RUNNING_MS: i64 = 6 * 60 * 60 * 1000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PipelineStageOutcome {
    Ok,
    Skipped,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PipelineRunOutcome {
    Running,
    Completed,
    Failed,
    SkippedDisabled,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PipelineStageRecord {
    pub name: String,
    pub started_at_unix_ms: i64,
    pub duration_ms: u64,
    pub outcome: PipelineStageOutcome,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PipelineRunRecord {
    pub schema_version: u32,
    pub run_id: String,
    pub pid: u32,
    pub trigger: String,
    pub started_at_unix_ms: i64,
    #[serde(default)]
    pub finished_at_unix_ms: Option<i64>,
    pub outcome: PipelineRunOutcome,
    #[serde(default)]
    pub stages: Vec<PipelineStageRecord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl PipelineRunRecord {
    /// Wall-clock length of the run in milliseconds. For a run that is still
    /// `running`, measures against `now_unix_ms`.
    pub fn elapsed_ms(&self, now_unix_ms: i64) -> i64 {
        self.finished_at_unix_ms
            .unwrap_or(now_unix_ms)
            .saturating_sub(self.started_at_unix_ms)
            .max(0)
    }

    /// Stages sorted by duration, longest first.
    pub fn slowest_stages(&self, count: usize) -> Vec<&PipelineStageRecord> {
        let mut stages: Vec<&PipelineStageRecord> = self.stages.iter().collect();
        stages.sort_by(|a, b| b.duration_ms.cmp(&a.duration_ms).then(a.name.cmp(&b.name)));
        stages.truncate(count);
        stages
    }

    /// One-line human summary used by `rein gc` and doctor.
    pub fn summary_line(&self, now_unix_ms: i64) -> String {
        let outcome = match self.outcome {
            PipelineRunOutcome::Running => "running",
            PipelineRunOutcome::Completed => "completed",
            PipelineRunOutcome::Failed => "failed",
            PipelineRunOutcome::SkippedDisabled => "skipped (adaptive disabled)",
        };
        let elapsed = self.elapsed_ms(now_unix_ms) as f64 / 1000.0;
        let slowest = self
            .slowest_stages(3)
            .iter()
            .map(|stage| format!("{} {:.1}s", stage.name, stage.duration_ms as f64 / 1000.0))
            .collect::<Vec<_>>()
            .join(", ");
        let mut line = format!("adaptive pipeline: {outcome} in {elapsed:.1}s");
        if !slowest.is_empty() {
            line.push_str(&format!(" (slowest: {slowest})"));
        }
        if let Some(error) = &self.error {
            line.push_str(&format!(" error={error}"));
        }
        line
    }
}

/// Load the last recorded run, tolerating newer schemas and extra fields.
/// Returns `None` when the row is absent or unreadable.
pub fn load_last_run(conn: &rusqlite::Connection) -> Option<PipelineRunRecord> {
    use rusqlite::OptionalExtension;
    let raw = conn
        .query_row(
            "SELECT value FROM metadata WHERE key = ?1",
            rusqlite::params![ADAPTIVE_PIPELINE_LAST_RUN_KEY],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .ok()
        .flatten()?;
    serde_json::from_str::<PipelineRunRecord>(&raw).ok()
}

fn now_unix_ms() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

/// Records stages of one pipeline pass and persists after each one.
///
/// Interior mutability lets the same recorder be shared by nested closures
/// (the A12 calibrate closure and the final-CAS review hook borrow it at the
/// same time). The pipeline is single-threaded, so `RefCell` is sufficient.
pub struct PipelineRunRecorder<'a> {
    store: &'a SqliteStore,
    record: RefCell<PipelineRunRecord>,
    /// Called every time a stage is recorded; the pipeline uses it to
    /// refresh its single-flight sentinel so a long pass is never mistaken
    /// for a dead one (codex round-19 P2).
    heartbeat: RefCell<Option<Box<dyn Fn()>>>,
}

impl<'a> PipelineRunRecorder<'a> {
    /// Start a run and persist it immediately as `running`.
    pub fn start(store: &'a SqliteStore, trigger: &str) -> Self {
        let record = PipelineRunRecord {
            schema_version: PIPELINE_RUN_SCHEMA_VERSION,
            run_id: ulid::Ulid::new().to_string(),
            pid: std::process::id(),
            trigger: trigger.to_string(),
            started_at_unix_ms: now_unix_ms(),
            finished_at_unix_ms: None,
            outcome: PipelineRunOutcome::Running,
            stages: Vec::new(),
            error: None,
        };
        let recorder = Self {
            store,
            record: RefCell::new(record),
            heartbeat: RefCell::new(None),
        };
        recorder.persist();
        recorder
    }

    /// Run `f` as a named stage, timing it and persisting the record after.
    pub fn stage<T>(&self, name: &str, f: impl FnOnce() -> T) -> T {
        let started_at = now_unix_ms();
        let clock = Instant::now();
        let _span = tracing::info_span!("pipeline_stage", stage = name).entered();
        let value = f();
        self.push_stage(name, started_at, clock, PipelineStageOutcome::Ok, None);
        value
    }

    /// Like [`Self::stage`] but records `failed` with the error text when `f`
    /// returns `Err`. The error is passed through untouched.
    pub fn stage_result<T, E: std::fmt::Display>(
        &self,
        name: &str,
        f: impl FnOnce() -> Result<T, E>,
    ) -> Result<T, E> {
        let started_at = now_unix_ms();
        let clock = Instant::now();
        let _span = tracing::info_span!("pipeline_stage", stage = name).entered();
        let result = f();
        match &result {
            Ok(_) => self.push_stage(name, started_at, clock, PipelineStageOutcome::Ok, None),
            Err(error) => self.push_stage(
                name,
                started_at,
                clock,
                PipelineStageOutcome::Failed,
                Some(error.to_string()),
            ),
        }
        result
    }

    /// Record a stage that did not run.
    pub fn skip(&self, name: &str, detail: &str) {
        let started_at = now_unix_ms();
        self.push_stage(
            name,
            started_at,
            Instant::now(),
            PipelineStageOutcome::Skipped,
            Some(detail.to_string()),
        );
    }

    /// Attach a free-form detail to the most recent stage of this name.
    pub fn annotate(&self, name: &str, detail: String) {
        let mut record = self.record.borrow_mut();
        if let Some(stage) = record.stages.iter_mut().rev().find(|s| s.name == name) {
            stage.detail = Some(detail);
        }
        drop(record);
        self.persist();
    }

    /// Mark the run finished and persist.
    pub fn finish(&self, outcome: PipelineRunOutcome, error: Option<String>) {
        {
            let mut record = self.record.borrow_mut();
            record.outcome = outcome;
            record.finished_at_unix_ms = Some(now_unix_ms());
            record.error = error;
        }
        self.persist();
        let record = self.record.borrow();
        tracing::info!(
            run_id = %record.run_id,
            trigger = %record.trigger,
            "{}",
            record.summary_line(now_unix_ms())
        );
    }

    /// Snapshot of the current record.
    pub fn record(&self) -> PipelineRunRecord {
        self.record.borrow().clone()
    }

    /// Names of stages recorded as `failed`, in order.
    pub fn failed_stage_names(&self) -> Vec<String> {
        self.record
            .borrow()
            .stages
            .iter()
            .filter(|stage| stage.outcome == PipelineStageOutcome::Failed)
            .map(|stage| stage.name.clone())
            .collect()
    }

    /// Finish as `Failed` when any recorded stage failed (naming them),
    /// otherwise as `Completed`. A run whose calibration stage errored must
    /// not read as healthy just because the snapshot saved.
    pub fn finish_from_stages(&self) {
        let failed = self.failed_stage_names();
        if failed.is_empty() {
            self.finish(PipelineRunOutcome::Completed, None);
        } else {
            self.finish(
                PipelineRunOutcome::Failed,
                Some(format!("failed stages: {}", failed.join(", "))),
            );
        }
    }

    /// Register a callback run after every recorded stage.
    pub fn set_heartbeat(&self, heartbeat: Box<dyn Fn()>) {
        *self.heartbeat.borrow_mut() = Some(heartbeat);
    }

    fn beat(&self) {
        if let Some(heartbeat) = self.heartbeat.borrow().as_ref() {
            heartbeat();
        }
    }

    fn push_stage(
        &self,
        name: &str,
        started_at_unix_ms: i64,
        clock: Instant,
        outcome: PipelineStageOutcome,
        detail: Option<String>,
    ) {
        let duration_ms = clock.elapsed().as_millis().min(u64::MAX as u128) as u64;
        tracing::info!(
            stage = name,
            duration_ms,
            outcome = ?outcome,
            detail = detail.as_deref().unwrap_or(""),
            "adaptive pipeline stage"
        );
        self.record.borrow_mut().stages.push(PipelineStageRecord {
            name: name.to_string(),
            started_at_unix_ms,
            duration_ms,
            outcome,
            detail,
        });
        self.persist();
        self.beat();
    }

    fn persist(&self) {
        let raw = match serde_json::to_string(&*self.record.borrow()) {
            Ok(raw) => raw,
            Err(error) => {
                tracing::warn!(%error, "pipeline run record could not be serialized");
                return;
            }
        };
        if let Err(error) = self.store.conn().execute(
            "INSERT INTO metadata(key, value) VALUES (?1, ?2) \
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            rusqlite::params![ADAPTIVE_PIPELINE_LAST_RUN_KEY, raw],
        ) {
            tracing::warn!(%error, "pipeline run record could not be persisted");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn epoch(store: &SqliteStore) -> u64 {
        crate::store::a12_calibration::load_a12_input_epoch(store.conn()).unwrap()
    }

    #[test]
    fn recorder_persists_running_row_at_start() {
        let store = SqliteStore::in_memory().unwrap();
        let recorder = PipelineRunRecorder::start(&store, "gc");
        let loaded = load_last_run(store.conn()).expect("row persisted at start");
        assert_eq!(loaded.outcome, PipelineRunOutcome::Running);
        assert_eq!(loaded.trigger, "gc");
        assert_eq!(loaded.schema_version, PIPELINE_RUN_SCHEMA_VERSION);
        assert!(loaded.stages.is_empty());
        assert_eq!(loaded.run_id, recorder.record().run_id);
        assert_eq!(loaded.pid, std::process::id());
    }

    #[test]
    fn recorder_persists_each_stage_and_final_outcome() {
        let store = SqliteStore::in_memory().unwrap();
        let recorder = PipelineRunRecorder::start(&store, "consolidate");
        let value = recorder.stage("m4_cluster", || 42);
        assert_eq!(value, 42);
        let after_first = load_last_run(store.conn()).unwrap();
        assert_eq!(after_first.stages.len(), 1);
        assert_eq!(after_first.stages[0].name, "m4_cluster");
        assert_eq!(after_first.stages[0].outcome, PipelineStageOutcome::Ok);
        assert_eq!(after_first.outcome, PipelineRunOutcome::Running);

        let failed: Result<(), String> =
            recorder.stage_result("a12_refresh", || Err("boom".to_string()));
        assert!(failed.is_err());
        recorder.skip("m5_tiers", "below tier_cold_start");
        recorder.annotate("m4_cluster", "clusters=3".to_string());
        recorder.finish(PipelineRunOutcome::Completed, None);

        let loaded = load_last_run(store.conn()).unwrap();
        assert_eq!(loaded.outcome, PipelineRunOutcome::Completed);
        assert!(loaded.finished_at_unix_ms.is_some());
        assert_eq!(loaded.stages.len(), 3);
        assert_eq!(loaded.stages[0].detail.as_deref(), Some("clusters=3"));
        assert_eq!(loaded.stages[1].outcome, PipelineStageOutcome::Failed);
        assert_eq!(loaded.stages[1].detail.as_deref(), Some("boom"));
        assert_eq!(loaded.stages[2].outcome, PipelineStageOutcome::Skipped);
        let summary = loaded.summary_line(now_unix_ms());
        assert!(summary.starts_with("adaptive pipeline: completed in "));
        assert!(summary.contains("slowest:"));
    }

    #[test]
    fn finish_from_stages_marks_failed_when_any_stage_failed() {
        let store = SqliteStore::in_memory().unwrap();
        let recorder = PipelineRunRecorder::start(&store, "gc");
        recorder.stage("m4_cluster", || ());
        let _: Result<(), String> = recorder.stage_result("a12_refresh", || Err("boom".into()));
        recorder.finish_from_stages();
        let loaded = load_last_run(store.conn()).unwrap();
        assert_eq!(loaded.outcome, PipelineRunOutcome::Failed);
        assert_eq!(loaded.error.as_deref(), Some("failed stages: a12_refresh"));

        let clean = PipelineRunRecorder::start(&store, "gc");
        clean.stage("m4_cluster", || ());
        clean.finish_from_stages();
        assert_eq!(
            load_last_run(store.conn()).unwrap().outcome,
            PipelineRunOutcome::Completed
        );
    }

    #[test]
    fn recorder_row_key_is_not_epoch_guarded() {
        let store = SqliteStore::in_memory().unwrap();
        let before = epoch(&store);
        let recorder = PipelineRunRecorder::start(&store, "gc");
        recorder.stage("m4_cluster", || ());
        recorder.finish(PipelineRunOutcome::Completed, None);
        assert_eq!(
            epoch(&store),
            before,
            "pipeline bookkeeping must not bump the A12 epoch"
        );
    }

    #[test]
    fn load_last_run_tolerates_future_schema_and_extra_fields() {
        let store = SqliteStore::in_memory().unwrap();
        store
            .conn()
            .execute(
                "INSERT INTO metadata(key, value) VALUES (?1, ?2)",
                rusqlite::params![
                    ADAPTIVE_PIPELINE_LAST_RUN_KEY,
                    r#"{"schema_version":2,"run_id":"r","pid":1,"trigger":"gc","started_at_unix_ms":10,"outcome":"completed","stages":[],"future_field":{"x":1}}"#
                ],
            )
            .unwrap();
        let loaded = load_last_run(store.conn()).expect("future schema still loads");
        assert_eq!(loaded.schema_version, 2);
        assert_eq!(loaded.outcome, PipelineRunOutcome::Completed);

        store
            .conn()
            .execute(
                "UPDATE metadata SET value = 'not json' WHERE key = ?1",
                rusqlite::params![ADAPTIVE_PIPELINE_LAST_RUN_KEY],
            )
            .unwrap();
        assert!(load_last_run(store.conn()).is_none());
    }

    #[test]
    fn summary_line_reports_running_elapsed_and_error() {
        let record = PipelineRunRecord {
            schema_version: 1,
            run_id: "r".into(),
            pid: 1,
            trigger: "gc".into(),
            started_at_unix_ms: 1_000,
            finished_at_unix_ms: None,
            outcome: PipelineRunOutcome::Failed,
            stages: vec![
                PipelineStageRecord {
                    name: "a".into(),
                    started_at_unix_ms: 1_000,
                    duration_ms: 500,
                    outcome: PipelineStageOutcome::Ok,
                    detail: None,
                },
                PipelineStageRecord {
                    name: "b".into(),
                    started_at_unix_ms: 1_500,
                    duration_ms: 2_500,
                    outcome: PipelineStageOutcome::Ok,
                    detail: None,
                },
            ],
            error: Some("x".into()),
        };
        let line = record.summary_line(11_000);
        assert_eq!(
            line,
            "adaptive pipeline: failed in 10.0s (slowest: b 2.5s, a 0.5s) error=x"
        );
        assert_eq!(record.slowest_stages(1)[0].name, "b");
    }
}
