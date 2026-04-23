//! Audit log for canonical resummerize runs (v0.23).
//!
//! Companion to the `resummerize_runs` schema DDL in `store/schema.rs` and the
//! `memories.needs_resummerize` / `memories.last_resummarized_at` columns added by
//! `migrate_resummerize`. A run is inserted in the `pending` status transition
//! sense (status may be any terminal state on insert — see below) when the
//! resummerize worker picks up a canonical, and later "finished" with the
//! produced output or a failure reason.
//!
//! Schema owns the table definition; this module owns the row model and the
//! helper functions that the worker, doctor, and tests use.
//!
//! Mirrors the `store/adaptive.rs` pattern: schema.rs defines the table,
//! a sibling module defines row models + helpers + unit tests against an
//! in-memory fixture.
//!
//! `count_needs_resummerize` filters by `status IN ('active', 'updated')
//! AND superseded_by IS NULL` — both states represent a live canonical
//! (`updated` is the post-merge state, auto-promoted by `store.update()`'s
//! trigger). Matches the predicate used by every other sweep selector
//! (see `ops/resummerize.rs::ELIGIBILITY_PREDICATE`, `ops/dedup.rs`
//! pending-row selector). Doctor cares about the sweep backlog, not
//! tombstoned rows. Codex round-5 H-1 + round-6 LOW.

use chrono::{DateTime, Duration, Utc};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};

/// Terminal status of a resummerize run.
///
/// * `Success` — LLM returned output, contract checks passed, persisted.
/// * `LlmError` — the LLM backend failed (timeout, transport, non-2xx, JSON parse).
/// * `ContractViolation` — output violated the Lossless Contract checks.
/// * `LengthExceeded` — output exceeded the target_bytes budget with no retry left.
/// * `Exhausted` — consecutive-failure fuse tripped; worker gives up on this
///   canonical until a new `needs_resummerize` flip (see
///   `count_recent_consecutive_failures`). Counted as a failure in
///   `recent_resummerize_failure_rate`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ResummerizeRunStatus {
    Success,
    LlmError,
    ContractViolation,
    LengthExceeded,
    Exhausted,
    /// Worker completed the LLM call but the 5-way CAS in `apply_resummerize`
    /// rejected its commit because a peer worker (or a concurrent `MergeInto`)
    /// mutated the row in the meantime. The LLM spend is sunk but the
    /// canonical is safe. Prior versions `DELETE`d the audit row on this
    /// path, which discarded the cost-observability trail and, if the
    /// delete failed, left the placeholder `llm_error` row inflating the
    /// 3-strike counter. Agent D architecture finding Q2/Q15 (v0.23.0
    /// post-ship review).
    ClaimLost,
}

impl ResummerizeRunStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::LlmError => "llm_error",
            Self::ContractViolation => "contract_violation",
            Self::LengthExceeded => "length_exceeded",
            Self::Exhausted => "exhausted",
            Self::ClaimLost => "claim_lost",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "success" => Some(Self::Success),
            "llm_error" => Some(Self::LlmError),
            "contract_violation" => Some(Self::ContractViolation),
            "length_exceeded" => Some(Self::LengthExceeded),
            "exhausted" => Some(Self::Exhausted),
            "claim_lost" => Some(Self::ClaimLost),
            _ => None,
        }
    }
}

/// Row model for `resummerize_runs`.
#[derive(Debug, Clone, Serialize)]
pub struct ResummerizeRunRow {
    pub id: String,
    pub canonical_id: String,
    pub input_evidence_count: u32,
    pub input_canonical_chars: u32,
    pub output_chars: Option<u32>,
    pub output_hash: Option<String>,
    pub target_bytes: u32,
    pub status: ResummerizeRunStatus,
    pub violations: Vec<String>,
    pub error: Option<String>,
    pub llm_backend: Option<String>,
    pub created_at: DateTime<Utc>,
    pub finished_at: Option<DateTime<Utc>>,
}

impl ResummerizeRunRow {
    /// Build a "just started, not yet finished" row for the emit-then-update
    /// worker flow in `ops/resummerize.rs`.
    ///
    /// The status defaults to `LlmError` as a tentative value: if the worker
    /// crashes between `insert_resummerize_run` and `finish_resummerize_run`,
    /// the row stays flagged as a failure rather than as a phantom success.
    /// `finish_resummerize_run` overwrites status, violations, error,
    /// output_chars, output_hash, and finished_at on terminal completion.
    pub fn starting(
        id: String,
        canonical_id: String,
        input_evidence_count: u32,
        input_canonical_chars: u32,
        target_bytes: u32,
        llm_backend: Option<String>,
        created_at: DateTime<Utc>,
    ) -> Self {
        Self {
            id,
            canonical_id,
            input_evidence_count,
            input_canonical_chars,
            output_chars: None,
            output_hash: None,
            target_bytes,
            status: ResummerizeRunStatus::LlmError,
            violations: Vec::new(),
            error: None,
            llm_backend,
            created_at,
            finished_at: None,
        }
    }
}

/// Insert a new resummerize_runs row.
///
/// Callers typically insert with the status set to the current terminal state
/// (workers may choose to emit a single row on finish, or emit-then-update via
/// `finish_resummerize_run` — both are supported; an emit-then-update flow
/// inserts with a tentative status such as `LlmError` and overwrites it on
/// success).
pub fn insert_resummerize_run(
    conn: &Connection,
    row: &ResummerizeRunRow,
) -> rusqlite::Result<()> {
    let violations_json = if row.violations.is_empty() {
        None
    } else {
        // serde_json::to_string on a Vec<String> never fails; this is the idiomatic
        // way to persist a JSON array column in rusqlite without introducing a
        // fallible error type into the signature.
        serde_json::to_string(&row.violations).ok()
    };
    conn.execute(
        "INSERT INTO resummerize_runs (
             id, canonical_id, input_evidence_count, input_canonical_chars,
             output_chars, output_hash, target_bytes, status, violations,
             error, llm_backend, created_at, finished_at
         ) VALUES (
             ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13
         )",
        rusqlite::params![
            row.id,
            row.canonical_id,
            row.input_evidence_count,
            row.input_canonical_chars,
            row.output_chars,
            row.output_hash,
            row.target_bytes,
            row.status.as_str(),
            violations_json,
            row.error,
            row.llm_backend,
            row.created_at.to_rfc3339(),
            row.finished_at.map(|t| t.to_rfc3339()),
        ],
    )?;
    Ok(())
}

/// Finalize a previously-inserted run with its terminal status and output.
///
/// * `output_hash` is owned `String` to match the caller ergonomics in
///   `ops/resummerize.rs`, where `sha256_hex(...)` returns `String`.
/// * `violations` is serialized as a JSON array when non-empty; an empty slice
///   is persisted as SQL NULL (matches the column's nullable semantics).
/// * `error` overwrites any error value set on the starting row — passing
///   `None` clears it, passing `Some(_)` sets it.
pub fn finish_resummerize_run(
    conn: &Connection,
    id: &str,
    output_chars: Option<u32>,
    output_hash: Option<String>,
    status: ResummerizeRunStatus,
    violations: &[String],
    error: Option<String>,
    finished_at: DateTime<Utc>,
) -> rusqlite::Result<()> {
    let violations_json = if violations.is_empty() {
        None
    } else {
        serde_json::to_string(violations).ok()
    };
    conn.execute(
        "UPDATE resummerize_runs
            SET output_chars = ?1,
                output_hash = ?2,
                status = ?3,
                violations = ?4,
                error = ?5,
                finished_at = ?6
          WHERE id = ?7",
        rusqlite::params![
            output_chars,
            output_hash,
            status.as_str(),
            violations_json,
            error,
            finished_at.to_rfc3339(),
            id,
        ],
    )?;
    Ok(())
}

/// Count consecutive failures for a canonical going back from its most
/// recent finished run, capped at `max`. A "failure" is any terminal status
/// other than `success` (includes `llm_error`, `contract_violation`,
/// `length_exceeded`, and `exhausted`).
///
/// The scan stops at `max` records because the caller only needs to know
/// whether the fuse has tripped — counting beyond the threshold is wasted
/// work. Unfinished rows (`finished_at IS NULL`) are ignored so mid-flight
/// workers don't self-trip the fuse.
///
/// Used by `ops/resummerize.rs` to decide whether to skip a canonical after
/// too many consecutive LLM/contract failures (spam guard).
pub fn count_recent_consecutive_failures(
    conn: &Connection,
    canonical_id: &str,
    max: usize,
) -> rusqlite::Result<usize> {
    // Agent A HIGH + Agent D Q14 fixes (post-v0.23.0):
    //
    //   (a) `llm_error` is treated as **transient** and does NOT count
    //       toward the fuse. A network blip / 429 / 5xx is not evidence
    //       that the LLM structurally can't satisfy this case, and
    //       counting it permanently strands canonicals when the API has a
    //       bad hour. A persistent API issue is instead operator-visible
    //       via `rein doctor`'s `recent_failure_rate`, which DOES count
    //       `llm_error` toward the rate metric so the signal isn't lost.
    //
    //   (b) `claim_lost` is a concurrent-write collision, not a resummerize
    //       failure; skipped entirely.
    //
    //   (c) `exhausted` marks the end of a prior generation. Prior versions
    //       treated exhausted + 3 failures as 4 strikes; after the fuse
    //       cleared the flag and `MergeInto` re-set it, the counter still
    //       saw those old rows and tripped the fuse on the very next
    //       attempt. We now BREAK on `exhausted` so the current epoch
    //       starts clean. Covered by the regression test below.
    //
    //   (d) `success` breaks (unchanged — marks a successful run; any
    //       failures before that are from an even older epoch).
    //
    //   (e) `contract_violation` and `length_exceeded` are the only two
    //       statuses that count toward the streak. Both are deterministic
    //       LLM-quality signals — "model structurally can't satisfy this
    //       case" — which is exactly what the fuse is for.
    //
    // Scan more rows than `max` because non-counting statuses (llm_error,
    // claim_lost) should be skipped rather than tripping the fuse.
    let scan_limit = (max * 4).max(32);
    let mut stmt = conn.prepare(
        "SELECT status FROM resummerize_runs
          WHERE canonical_id = ?1
            AND finished_at IS NOT NULL
          ORDER BY finished_at DESC
          LIMIT ?2",
    )?;
    let rows = stmt.query_map(
        rusqlite::params![canonical_id, scan_limit as i64],
        |row| row.get::<_, String>(0),
    )?;
    let mut consecutive = 0usize;
    for r in rows {
        let status = r?;
        match status.as_str() {
            "success" | "exhausted" => break,
            "llm_error" | "claim_lost" => continue, // transient / race — skip
            // contract_violation, length_exceeded, or any unknown new
            // status — conservative default is "counts toward fuse".
            _ => {
                consecutive += 1;
                if consecutive >= max {
                    break;
                }
            }
        }
    }
    Ok(consecutive)
}

/// Number of canonicals currently flagged as needing resummerize, restricted
/// to active non-superseded rows (matches the sweep semantics used across the
/// dedup/consolidation code path; this is what doctor surfaces as the backlog).
pub fn count_needs_resummerize(conn: &Connection) -> rusqlite::Result<u64> {
    // `status IN ('active', 'updated')`: round-5 H-1. Merged canonicals
    // are promoted from `active` to `updated`; both states are live
    // canonicals that resummerize should count.
    conn.query_row(
        "SELECT COUNT(*) FROM memories
          WHERE needs_resummerize = 1
            AND status IN ('active', 'updated')
            AND superseded_by IS NULL",
        [],
        |row| row.get::<_, i64>(0),
    )
    .map(|n| n as u64)
}

/// Recent failure rate of resummerize runs within `window`, computed over
/// finished runs only (i.e. `finished_at` is non-null).
///
/// A "failure" is `llm_error`, `contract_violation`, `length_exceeded`, or
/// `exhausted`. `claim_lost` is **excluded** — it indicates a successful LLM
/// call that lost the 5-way CAS race to a peer worker, which is the system
/// behaving as designed (concurrency-safe overwrite prevention). A high
/// `claim_lost` rate is observability-worthy as a contention signal but it
/// shouldn't page the operator as a quality regression. Returns 0.0 when
/// there are no relevant finished runs in the window (so doctor doesn't
/// page on an empty log).
///
/// The shorter name `recent_failure_rate` mirrors the doctor call site at
/// `doctor.rs::check_resummerize`; within the `resummerize_audit::` module
/// path the shorter form reads more naturally than the verbose
/// `recent_resummerize_failure_rate` originally specified.
pub fn recent_failure_rate(conn: &Connection, window: Duration) -> rusqlite::Result<f64> {
    let cutoff = (Utc::now() - window).to_rfc3339();
    // Both numerator and denominator exclude `claim_lost` so its presence
    // in the audit table doesn't shift the rate either direction.
    let total: i64 = conn.query_row(
        "SELECT COUNT(*) FROM resummerize_runs
          WHERE finished_at IS NOT NULL
            AND finished_at >= ?1
            AND status != 'claim_lost'",
        rusqlite::params![cutoff],
        |row| row.get(0),
    )?;
    if total == 0 {
        return Ok(0.0);
    }
    let failures: i64 = conn.query_row(
        "SELECT COUNT(*) FROM resummerize_runs
          WHERE finished_at IS NOT NULL
            AND finished_at >= ?1
            AND status NOT IN ('success', 'claim_lost')",
        rusqlite::params![cutoff],
        |row| row.get(0),
    )?;
    Ok(failures as f64 / total as f64)
}

/// Proportion of finished resummerize runs within `window` that ended as
/// `claim_lost`. Separate from `recent_failure_rate` which filters
/// `claim_lost` OUT (quality metric). This metric is the contention
/// signal: a high `recent_claim_lost_rate` means workers are racing on
/// the same canonicals, burning LLM budget without making progress.
///
/// Post-audit round-2 MED-2 fix. `claim_lost` is not a quality failure,
/// but without a dedicated metric it disappears from operator visibility
/// entirely once M-3 stopped folding it into `recent_failure_rate`.
/// Returns 0.0 when no runs in the window (doctor won't page on empty).
pub fn recent_claim_lost_rate(conn: &Connection, window: Duration) -> rusqlite::Result<f64> {
    let cutoff = (Utc::now() - window).to_rfc3339();
    let total: i64 = conn.query_row(
        "SELECT COUNT(*) FROM resummerize_runs
          WHERE finished_at IS NOT NULL AND finished_at >= ?1",
        rusqlite::params![cutoff],
        |row| row.get(0),
    )?;
    if total == 0 {
        return Ok(0.0);
    }
    let claim_lost: i64 = conn.query_row(
        "SELECT COUNT(*) FROM resummerize_runs
          WHERE finished_at IS NOT NULL
            AND finished_at >= ?1
            AND status = 'claim_lost'",
        rusqlite::params![cutoff],
        |row| row.get(0),
    )?;
    Ok(claim_lost as f64 / total as f64)
}

/// Number of finished resummerize runs within `window`. Used by doctor to
/// gate "failure rate" warnings on a minimum sample size — a single failure
/// out of one run shouldn't page.
///
/// Post-fix audit M-3: excludes `claim_lost` so the gate matches the
/// population used by `recent_failure_rate`. Without this alignment, a
/// burst of concurrency races (`claim_lost`) would inflate the sample
/// count past the `>= 5` threshold while simultaneously shrinking the
/// failure-rate denominator to just the real quality failures — letting
/// a single `contract_violation` alongside 4 `claim_lost` races pass
/// doctor's significance gate as "100% failure rate over 5 runs."
pub fn recent_run_count(conn: &Connection, window: Duration) -> rusqlite::Result<u64> {
    let cutoff = (Utc::now() - window).to_rfc3339();
    conn.query_row(
        "SELECT COUNT(*) FROM resummerize_runs
          WHERE finished_at IS NOT NULL
            AND finished_at >= ?1
            AND status != 'claim_lost'",
        rusqlite::params![cutoff],
        |row| row.get::<_, i64>(0),
    )
    .map(|n| n as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::schema;
    use rusqlite::Connection;

    /// Build a minimal fixture that mirrors the real schema for just the
    /// surfaces this module touches: the `memories` columns that
    /// `count_needs_resummerize` needs, plus the full `resummerize_runs` table
    /// created by running the real migration path.
    ///
    /// Rather than duplicate the DDL here (risking drift when schema.rs
    /// evolves), we spin up the real `init_schema` against an in-memory DB.
    /// That path is what every other store test uses; `adaptive.rs` predates
    /// this pattern and hand-rolls a minimal fixture, but for v0.23 migrations
    /// we want to exercise the real migrate path so the idempotency test
    /// actually tests migration idempotency — not a hand-crafted simulation.
    fn setup_db() -> Connection {
        schema::init_sqlite_vec();
        let conn = Connection::open_in_memory().unwrap();
        // 3072 is the production dim; any positive value works here since we
        // don't exercise the vec table in this module.
        schema::init_schema(&conn, 3072).unwrap();
        conn
    }

    fn insert_memory(conn: &Connection, id: &str, needs: i32) {
        conn.execute(
            "INSERT INTO memories (
                 id, layer, topic, summary, content, keywords, importance,
                 source, strength, decay_lambda, access_count, related_ids,
                 concept_ids, status, created_at, updated_at, last_accessed,
                 needs_resummerize
             ) VALUES (
                 ?1, 'LTM', 't', 's', 'c', '[]', 'medium', 'manual',
                 1.0, 0.001, 0, '[]', '[]', 'active',
                 '2026-04-22T00:00:00Z', '2026-04-22T00:00:00Z', '2026-04-22T00:00:00Z',
                 ?2
             )",
            rusqlite::params![id, needs],
        )
        .unwrap();
    }

    #[test]
    fn migrate_resummerize_is_idempotent() {
        let conn = setup_db();

        // init_schema already ran once via setup_db; run it again to verify
        // every path (including migrate_resummerize) is idempotent.
        schema::init_schema(&conn, 3072).unwrap();
        // Third pass for extra paranoia — the needs_vec_dedup migration has
        // historically been the template other migrations copy, and this
        // triple-pass is what catches accidental non-idempotency.
        schema::init_schema(&conn, 3072).unwrap();

        // The new columns must exist after migration.
        let has_needs_col: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('memories') WHERE name='needs_resummerize'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(has_needs_col, 1, "needs_resummerize column should exist");

        let has_last_col: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('memories') WHERE name='last_resummarized_at'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(
            has_last_col, 1,
            "last_resummarized_at column should exist"
        );

        // And the audit table must exist.
        let has_runs_tbl: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='resummerize_runs'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(has_runs_tbl, 1, "resummerize_runs table should exist");
    }

    #[test]
    fn insert_and_finish_round_trip_preserves_all_fields() {
        let conn = setup_db();
        insert_memory(&conn, "mem-1", 1);

        // Use the `starting` constructor to match the worker flow in
        // `ops/resummerize.rs` — the whole point of the emit-then-update
        // pattern is exercising both insert and finish.
        let created = Utc::now() - Duration::seconds(30);
        let row = ResummerizeRunRow::starting(
            "run-1".to_string(),
            "mem-1".to_string(),
            7,
            4096,
            2048,
            Some("gemini".to_string()),
            created,
        );
        insert_resummerize_run(&conn, &row).unwrap();

        // Now finalize with a contract-violation status that carries a
        // non-empty violations JSON array so we exercise both the happy path
        // and the JSON round-trip for the list column.
        let finished = Utc::now();
        let violations = vec![
            "claim_dropped:v0.21".to_string(),
            "citation_missing:doc_A".to_string(),
        ];
        finish_resummerize_run(
            &conn,
            "run-1",
            Some(1800),
            Some("deadbeef".to_string()), // sha256 hex stub
            ResummerizeRunStatus::ContractViolation,
            &violations,
            Some("timeout".to_string()),
            finished,
        )
        .unwrap();

        // Read it back. We must preserve every field including the violations
        // JSON array exactly — if serialize/deserialize drifts, downstream
        // doctor reporting shows the wrong violation.
        let (
            id,
            canonical_id,
            evidence_count,
            canonical_chars,
            output_chars,
            output_hash,
            target_bytes,
            status,
            violations_json,
            error,
            llm_backend,
            created_at_str,
            finished_at_str,
        ) = conn
            .query_row(
                "SELECT id, canonical_id, input_evidence_count, input_canonical_chars,
                        output_chars, output_hash, target_bytes, status, violations,
                        error, llm_backend, created_at, finished_at
                   FROM resummerize_runs WHERE id = 'run-1'",
                [],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, i64>(2)?,
                        row.get::<_, i64>(3)?,
                        row.get::<_, Option<i64>>(4)?,
                        row.get::<_, Option<String>>(5)?,
                        row.get::<_, i64>(6)?,
                        row.get::<_, String>(7)?,
                        row.get::<_, Option<String>>(8)?,
                        row.get::<_, Option<String>>(9)?,
                        row.get::<_, Option<String>>(10)?,
                        row.get::<_, String>(11)?,
                        row.get::<_, Option<String>>(12)?,
                    ))
                },
            )
            .unwrap();

        assert_eq!(id, "run-1");
        assert_eq!(canonical_id, "mem-1");
        assert_eq!(evidence_count, 7);
        assert_eq!(canonical_chars, 4096);
        assert_eq!(output_chars, Some(1800));
        assert_eq!(output_hash.as_deref(), Some("deadbeef"));
        assert_eq!(target_bytes, 2048);
        assert_eq!(status, "contract_violation");
        assert_eq!(error.as_deref(), Some("timeout"));
        assert_eq!(llm_backend.as_deref(), Some("gemini"));

        // Violations JSON array round-trip.
        let decoded: Vec<String> =
            serde_json::from_str(violations_json.as_deref().expect("violations set")).unwrap();
        assert_eq!(decoded, violations);

        // Timestamps preserved (we compare by parsing back to DateTime to
        // avoid flakiness around +00:00 vs Z serialization).
        let parsed_created: DateTime<Utc> = DateTime::parse_from_rfc3339(&created_at_str)
            .unwrap()
            .with_timezone(&Utc);
        assert!((parsed_created - created).num_milliseconds().abs() < 1000);
        let parsed_finished: DateTime<Utc> =
            DateTime::parse_from_rfc3339(finished_at_str.as_deref().expect("finished_at set"))
                .unwrap()
                .with_timezone(&Utc);
        assert!((parsed_finished - finished).num_milliseconds().abs() < 1000);

        // Status round-trip through the enum.
        assert_eq!(
            ResummerizeRunStatus::from_str(&status),
            Some(ResummerizeRunStatus::ContractViolation)
        );
    }

    #[test]
    fn count_needs_resummerize_reflects_flag_state() {
        let conn = setup_db();

        // Baseline: no memories → 0.
        assert_eq!(count_needs_resummerize(&conn).unwrap(), 0);

        // Insert three active memories, two flagged.
        insert_memory(&conn, "m-a", 1);
        insert_memory(&conn, "m-b", 0);
        insert_memory(&conn, "m-c", 1);
        assert_eq!(count_needs_resummerize(&conn).unwrap(), 2);

        // Flipping a flag updates the count.
        conn.execute(
            "UPDATE memories SET needs_resummerize = 0 WHERE id = 'm-a'",
            [],
        )
        .unwrap();
        assert_eq!(count_needs_resummerize(&conn).unwrap(), 1);

        // Tombstoned rows don't count even if flagged — matches the sweep's
        // WHERE clause used across the dedup/consolidation paths.
        conn.execute(
            "UPDATE memories SET status = 'deprecated' WHERE id = 'm-c'",
            [],
        )
        .unwrap();
        assert_eq!(count_needs_resummerize(&conn).unwrap(), 0);
    }

    #[test]
    fn recent_failure_rate_window_arithmetic() {
        let conn = setup_db();
        insert_memory(&conn, "mem-1", 0);

        let now = Utc::now();

        // Helper to insert a finished run at an offset.
        let insert_run = |id: &str, status: ResummerizeRunStatus, secs_ago: i64| {
            let t = now - Duration::seconds(secs_ago);
            let row = ResummerizeRunRow {
                id: id.to_string(),
                canonical_id: "mem-1".to_string(),
                input_evidence_count: 1,
                input_canonical_chars: 100,
                output_chars: Some(80),
                output_hash: Some("x".to_string()),
                target_bytes: 128,
                status,
                violations: vec![],
                error: None,
                llm_backend: Some("gemini".to_string()),
                created_at: t,
                finished_at: Some(t),
            };
            insert_resummerize_run(&conn, &row).unwrap();
        };

        // Within a 60s window: 2 success + 2 failures → 50%.
        insert_run("r1", ResummerizeRunStatus::Success, 10);
        insert_run("r2", ResummerizeRunStatus::Success, 20);
        insert_run("r3", ResummerizeRunStatus::LlmError, 30);
        insert_run("r4", ResummerizeRunStatus::ContractViolation, 40);
        // Outside the 60s window — must be excluded.
        insert_run("r5", ResummerizeRunStatus::LlmError, 3600);

        let rate = recent_failure_rate(&conn, Duration::seconds(60)).unwrap();
        assert!(
            (rate - 0.5).abs() < 1e-9,
            "expected 0.5 failure rate in 60s window, got {rate}"
        );

        // Broadening the window to include r5 pulls it in → 3/5 = 0.6.
        let rate_wide =
            recent_failure_rate(&conn, Duration::seconds(7200)).unwrap();
        assert!(
            (rate_wide - 0.6).abs() < 1e-9,
            "expected 0.6 failure rate in 2h window, got {rate_wide}"
        );

        // An unfinished run inside the window must not count (neither in
        // numerator nor denominator) — we only measure completed work.
        let pending = ResummerizeRunRow {
            id: "r6".to_string(),
            canonical_id: "mem-1".to_string(),
            input_evidence_count: 1,
            input_canonical_chars: 10,
            output_chars: None,
            output_hash: None,
            target_bytes: 64,
            status: ResummerizeRunStatus::LlmError,
            violations: vec![],
            error: Some("inflight".to_string()),
            llm_backend: Some("gemini".to_string()),
            created_at: now,
            finished_at: None,
        };
        insert_resummerize_run(&conn, &pending).unwrap();
        let rate_after = recent_failure_rate(&conn, Duration::seconds(60)).unwrap();
        assert!(
            (rate_after - 0.5).abs() < 1e-9,
            "unfinished runs must be excluded, got {rate_after}"
        );

        // Empty window → 0.0 (not NaN).
        let rate_empty = recent_failure_rate(&conn, Duration::seconds(1)).unwrap();
        assert_eq!(rate_empty, 0.0);
    }

    #[test]
    fn consecutive_failures_fuse_logic() {
        let conn = setup_db();
        insert_memory(&conn, "mem-1", 1);

        let mut at = Utc::now() - Duration::seconds(600);
        let mut push = |status: ResummerizeRunStatus| {
            at += Duration::seconds(10);
            let id = format!("run-{}", ulid::Ulid::new());
            let row = ResummerizeRunRow {
                id: id.clone(),
                canonical_id: "mem-1".to_string(),
                input_evidence_count: 1,
                input_canonical_chars: 100,
                output_chars: Some(80),
                output_hash: Some("x".to_string()),
                target_bytes: 128,
                status,
                violations: vec![],
                error: None,
                llm_backend: Some("gemini".to_string()),
                created_at: at,
                finished_at: Some(at),
            };
            insert_resummerize_run(&conn, &row).unwrap();
        };

        // Nothing yet → 0.
        assert_eq!(
            count_recent_consecutive_failures(&conn, "mem-1", 3).unwrap(),
            0
        );

        // Post-fix semantics (Agent A HIGH + Agent D Q14):
        //
        // Sequence: success, llm_error, llm_error — llm_error is treated as
        // **transient** (network blip / 429 / 5xx, not LLM-quality signal)
        // and does NOT count toward the fuse. So 0, not 2.
        push(ResummerizeRunStatus::Success);
        push(ResummerizeRunStatus::LlmError);
        push(ResummerizeRunStatus::LlmError);
        assert_eq!(
            count_recent_consecutive_failures(&conn, "mem-1", 3).unwrap(),
            0,
            "llm_error must not count toward fuse — transient errors should \
             not strand canonicals; persistent API issues are visible via \
             `recent_failure_rate` instead"
        );

        // Add a contract_violation — that DOES count.
        push(ResummerizeRunStatus::ContractViolation);
        assert_eq!(
            count_recent_consecutive_failures(&conn, "mem-1", 3).unwrap(),
            1
        );

        // Two more contract violations to trip the fuse.
        push(ResummerizeRunStatus::ContractViolation);
        push(ResummerizeRunStatus::ContractViolation);
        assert_eq!(
            count_recent_consecutive_failures(&conn, "mem-1", 3).unwrap(),
            3
        );

        // Now simulate the production path: record_exhaustion fires and
        // inserts an `exhausted` audit row + clears the flag. After
        // MergeInto re-sets needs_resummerize=1, the next worker should see
        // a CLEAN streak (epoch reset). Prior buggy behavior: streak still
        // counted those 3 contract violations → fuse trips immediately.
        push(ResummerizeRunStatus::Exhausted);
        assert_eq!(
            count_recent_consecutive_failures(&conn, "mem-1", 3).unwrap(),
            0,
            "exhausted marker must terminate the prior epoch — Agent D Q14"
        );

        // After the epoch reset, a single contract failure → streak=1.
        push(ResummerizeRunStatus::ContractViolation);
        assert_eq!(
            count_recent_consecutive_failures(&conn, "mem-1", 3).unwrap(),
            1
        );

        // claim_lost is a concurrency race, not a quality failure — must
        // not count.
        push(ResummerizeRunStatus::ClaimLost);
        push(ResummerizeRunStatus::ClaimLost);
        assert_eq!(
            count_recent_consecutive_failures(&conn, "mem-1", 3).unwrap(),
            1,
            "claim_lost must not count toward fuse — it indicates concurrency \
             loss, not LLM quality"
        );

        // Add length_exceeded → counts.
        push(ResummerizeRunStatus::LengthExceeded);
        assert_eq!(
            count_recent_consecutive_failures(&conn, "mem-1", 3).unwrap(),
            2,
            "length_exceeded must count — it's a deterministic LLM quality \
             signal, same class as contract_violation"
        );

        // A success after the partial streak still resets.
        push(ResummerizeRunStatus::Success);
        assert_eq!(
            count_recent_consecutive_failures(&conn, "mem-1", 3).unwrap(),
            0
        );

        // Unfinished rows don't trip the fuse even if they'd look like
        // failures — the worker is mid-flight.
        let inflight = ResummerizeRunRow {
            id: "inflight".to_string(),
            canonical_id: "mem-1".to_string(),
            input_evidence_count: 1,
            input_canonical_chars: 10,
            output_chars: None,
            output_hash: None,
            target_bytes: 64,
            status: ResummerizeRunStatus::ContractViolation,
            violations: vec![],
            error: None,
            llm_backend: Some("gemini".to_string()),
            created_at: Utc::now(),
            finished_at: None,
        };
        insert_resummerize_run(&conn, &inflight).unwrap();
        assert_eq!(
            count_recent_consecutive_failures(&conn, "mem-1", 3).unwrap(),
            0,
            "inflight rows must not count toward consecutive failures"
        );

        // Other canonicals' runs must not bleed across (per-canonical fuse).
        insert_memory(&conn, "mem-2", 1);
        assert_eq!(
            count_recent_consecutive_failures(&conn, "mem-2", 3).unwrap(),
            0
        );
    }

    #[test]
    fn recent_failure_rate_excludes_claim_lost() {
        // Agent D Q2/Q15 fix: `claim_lost` is the system working as designed
        // (concurrency-safe rejection of a stale CAS), not a quality
        // failure. The doctor's failure-rate metric must exclude it from
        // both numerator and denominator so a high contention period
        // doesn't page the operator with a misleading "resummerize is
        // failing" alert.
        let conn = setup_db();
        insert_memory(&conn, "mem-1", 0);
        let now = Utc::now();
        let mk = |status: ResummerizeRunStatus, id: &str| ResummerizeRunRow {
            id: id.to_string(),
            canonical_id: "mem-1".to_string(),
            input_evidence_count: 1,
            input_canonical_chars: 100,
            output_chars: Some(80),
            output_hash: Some("x".to_string()),
            target_bytes: 128,
            status,
            violations: vec![],
            error: None,
            llm_backend: Some("gemini".to_string()),
            created_at: now,
            finished_at: Some(now),
        };
        // 1 success, 1 contract_violation → without any claim_lost: 50%
        insert_resummerize_run(&conn, &mk(ResummerizeRunStatus::Success, "r1")).unwrap();
        insert_resummerize_run(&conn, &mk(ResummerizeRunStatus::ContractViolation, "r2")).unwrap();
        let rate1 = recent_failure_rate(&conn, Duration::seconds(60)).unwrap();
        assert!((rate1 - 0.5).abs() < 1e-9, "baseline 50%, got {rate1}");

        // Add 8 claim_lost rows. Old behavior would count them as failures
        // → 9/10 = 90%. New behavior excludes them entirely → still 50%.
        for i in 0..8 {
            insert_resummerize_run(
                &conn,
                &mk(ResummerizeRunStatus::ClaimLost, &format!("cl-{i}")),
            )
            .unwrap();
        }
        let rate2 = recent_failure_rate(&conn, Duration::seconds(60)).unwrap();
        assert!(
            (rate2 - 0.5).abs() < 1e-9,
            "claim_lost must not shift the failure rate, got {rate2}"
        );
    }

    #[test]
    fn finish_overwrites_starting_status_and_error() {
        let conn = setup_db();
        insert_memory(&conn, "mem-1", 1);

        // Worker flow: insert a "starting" row. Its status is the tentative
        // `LlmError` placeholder so a crash before finish doesn't falsely
        // surface as success.
        let row = ResummerizeRunRow::starting(
            "run-x".to_string(),
            "mem-1".to_string(),
            1,
            500,
            1024,
            Some("gemini".to_string()),
            Utc::now(),
        );
        assert_eq!(row.status, ResummerizeRunStatus::LlmError);
        assert!(row.error.is_none());
        assert!(row.finished_at.is_none());
        insert_resummerize_run(&conn, &row).unwrap();

        // Finish with Success and no error — finish must overwrite the
        // tentative status and leave the error column NULL.
        finish_resummerize_run(
            &conn,
            "run-x",
            Some(900),
            Some("aa".to_string()),
            ResummerizeRunStatus::Success,
            &[],
            None,
            Utc::now(),
        )
        .unwrap();

        let (status, error, finished_set): (String, Option<String>, i64) = conn
            .query_row(
                "SELECT status, error,
                        CASE WHEN finished_at IS NULL THEN 0 ELSE 1 END
                   FROM resummerize_runs WHERE id = 'run-x'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(status, "success");
        assert!(error.is_none(), "error must be cleared on success finish");
        assert_eq!(finished_set, 1, "finished_at must be set by finish call");
    }

    #[test]
    fn recent_run_count_excludes_claim_lost() {
        // Post-audit round-2 LOW #7: parallel guard for the
        // recent_run_count denominator filter. Prior to M-3 this
        // counter included claim_lost rows; doctor's "≥5 runs to warn"
        // gate could fire on a single real contract_violation alongside
        // 4 claim_lost races, producing a misleading "100% failure
        // rate" page. This test locks the filter alignment so a future
        // refactor can't silently reintroduce the mismatch.
        let conn = setup_db();
        insert_memory(&conn, "mem-rrc", 0);
        let now = Utc::now();
        let mk = |status: ResummerizeRunStatus, id: &str| ResummerizeRunRow {
            id: id.to_string(),
            canonical_id: "mem-rrc".to_string(),
            input_evidence_count: 1,
            input_canonical_chars: 100,
            output_chars: Some(80),
            output_hash: Some("x".to_string()),
            target_bytes: 128,
            status,
            violations: vec![],
            error: None,
            llm_backend: Some("gemini".to_string()),
            created_at: now,
            finished_at: Some(now),
        };

        // 2 success, 1 contract_violation → recent_run_count = 3.
        insert_resummerize_run(&conn, &mk(ResummerizeRunStatus::Success, "r1")).unwrap();
        insert_resummerize_run(&conn, &mk(ResummerizeRunStatus::Success, "r2")).unwrap();
        insert_resummerize_run(&conn, &mk(ResummerizeRunStatus::ContractViolation, "r3")).unwrap();
        assert_eq!(
            recent_run_count(&conn, Duration::seconds(60)).unwrap(),
            3,
            "baseline count"
        );

        // 5 more claim_lost rows → run_count must stay at 3, NOT jump to 8.
        for i in 0..5 {
            insert_resummerize_run(
                &conn,
                &mk(ResummerizeRunStatus::ClaimLost, &format!("cl-{i}")),
            )
            .unwrap();
        }
        assert_eq!(
            recent_run_count(&conn, Duration::seconds(60)).unwrap(),
            3,
            "claim_lost rows must be excluded from recent_run_count"
        );
    }

    #[test]
    fn recent_claim_lost_rate_tracks_contention() {
        // Post-audit round-2 MED-2: separate contention metric for
        // doctor. `recent_failure_rate` filters claim_lost OUT (quality
        // metric); `recent_claim_lost_rate` measures the ratio.
        let conn = setup_db();
        insert_memory(&conn, "mem-cl", 0);
        let now = Utc::now();
        let mk = |status: ResummerizeRunStatus, id: &str| ResummerizeRunRow {
            id: id.to_string(),
            canonical_id: "mem-cl".to_string(),
            input_evidence_count: 1,
            input_canonical_chars: 100,
            output_chars: Some(80),
            output_hash: Some("x".to_string()),
            target_bytes: 128,
            status,
            violations: vec![],
            error: None,
            llm_backend: Some("gemini".to_string()),
            created_at: now,
            finished_at: Some(now),
        };

        // Empty → 0.
        assert_eq!(
            recent_claim_lost_rate(&conn, Duration::seconds(60)).unwrap(),
            0.0
        );

        // 2 success + 2 claim_lost → rate = 0.5.
        insert_resummerize_run(&conn, &mk(ResummerizeRunStatus::Success, "s1")).unwrap();
        insert_resummerize_run(&conn, &mk(ResummerizeRunStatus::Success, "s2")).unwrap();
        insert_resummerize_run(&conn, &mk(ResummerizeRunStatus::ClaimLost, "cl1")).unwrap();
        insert_resummerize_run(&conn, &mk(ResummerizeRunStatus::ClaimLost, "cl2")).unwrap();
        let rate = recent_claim_lost_rate(&conn, Duration::seconds(60)).unwrap();
        assert!(
            (rate - 0.5).abs() < 1e-9,
            "expected 0.5 claim_lost rate, got {rate}"
        );

        // Adding a contract_violation moves the denominator to 5 →
        // rate = 2/5 = 0.4.
        insert_resummerize_run(&conn, &mk(ResummerizeRunStatus::ContractViolation, "cv")).unwrap();
        let rate2 = recent_claim_lost_rate(&conn, Duration::seconds(60)).unwrap();
        assert!(
            (rate2 - 2.0 / 5.0).abs() < 1e-9,
            "expected 0.4 (2/5) after adding a quality failure, got {rate2}"
        );
    }
}
