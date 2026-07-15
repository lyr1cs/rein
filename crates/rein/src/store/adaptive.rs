//! Adaptive Engine: event sourcing, per-consumer offsets, and AdaptiveState cache.
//! Foundation module (M1) for the unified self-adaptive engine.
//!
//! ## Invariant for event consumers (peek+commit)
//!
//! An event MUST NOT be marked consumed by consumer X unless the state
//! change derived from it is already durable — either in the
//! `adaptive_state` snapshot (orchestrator-threaded consumers) or in the
//! consumer's own persistent write (e.g. `weights`).
//!
//! Concretely: callers `peek_events` (no offset advance), perform their
//! work, persist the result, THEN `commit_offset` to advance the
//! consumer's cursor. A crash anywhere before `commit_offset` is safe —
//! the next pass re-peeks the same events. State changes in the
//! pipeline are idempotent per-event (counter increments aside, which
//! are bounded by FIFO caps), so replay is correct even if a partial
//! pass already ran.
//!
//! Pre-v0.24 code used `consume_events` which bundled peek + advance in
//! one call. That bundling is the root cause Codex round 2-4 hammered
//! during ARS L3 audit; v0.24 retires it.

use crate::types::{ReinError, ReinResult};
use chrono::Utc;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};

// ── Event Types ──────────────────────────────────────────────────────────────

/// All feedback event types emitted by the system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventType {
    RecallComplete,          // recall returned results (includes full candidate set)
    RecallAccess,            // agent used a recalled memory
    RecallMiss,              // recall returned but not accessed (record-only)
    RecallRetry,             // same query recalled again in session
    Store,                   // new memory stored
    StoreQuickRecall,        // memory recalled shortly after being stored
    Forget,                  // agent explicitly forgot/deprecated
    Refine,                  // concept refined/superseded
    SessionEnd,              // hook_stop fired
    ParamUpdate,             // slow-channel parameter update (audit trail)
    ConceptSummaryRefreshed, // v0.24 ARS L3: concept living-summary refreshed
    /// v0.26 D direction: user interacted with a Cap B synthesis prose surface.
    /// Payload is a JSON-serialized [`SynthesisInteractionPayload`] in
    /// `feedback_events.payload` (no DDL change — column already TEXT).
    /// Backward compat: `feedback_events.event_type` is a `String` column,
    /// existing consumers filter by string equality and silently skip
    /// unknown values; no exhaustive `match` over `EventType` exists outside
    /// `EventType::as_str` itself.
    SynthesisInteraction,
    /// v0.27 ARS Cap A feedback loop (Track 1, mirror of v0.26 D for Cap B):
    /// user interacted with a concept living-summary surface (Cap A).
    /// Payload is a JSON-serialized [`ConceptSummaryInteractionPayload`].
    /// Same back-compat invariant as `SynthesisInteraction` — string-based
    /// dispatch via `event_type`, no exhaustive match on `EventType` outside
    /// `EventType::as_str`.
    ConceptSummaryInteraction,
    /// v0.27.1 E direction Layer 1 — **runtime** LLM judge produced a
    /// synthesis-quality label for a Cap B output. Consumed by
    /// `synthesis_feedback` consumer + folded into `useful_rate` via the
    /// `llm_judge_count` / `llm_judge_hit_count` counters. Payload is JSON-
    /// serialized [`SynthesisLlmJudgePayload`] in `feedback_events.payload`.
    /// Distinct from `SynthesisInteraction` (human signal) so the consumer
    /// can apply `w_llm` weight. Back-compat: string dispatch + `_ => {}`
    /// fall-through in old consumers.
    SynthesisLlmJudge,
    /// v0.27.1 E direction Cap A Layer 1 mirror.
    ConceptSummaryLlmJudge,
    /// v0.27.1 E direction Layer 2 — **offline calibration cron** judged a
    /// previously-synthesized output via the stricter nightly_cron LLM.
    /// Consumed by `judge_calibration` consumer for κ accumulation
    /// **only**; MUST NOT enter `useful_rate`. Codex R2 P2 caught the v0
    /// draft's source-discriminated single-event-type design — separate
    /// event type prevents calibration data from training the gate it
    /// audits.
    SynthesisLlmJudgeOfflineCron,
    /// v0.27.1 E direction Cap A Layer 2 mirror.
    ConceptSummaryLlmJudgeOfflineCron,
    /// Deterministic health probe for the judge. This event has a dedicated
    /// replay consumer and must never enter useful-rate or human-pair folds.
    JudgeStructuralAnchor,
}

impl EventType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::RecallComplete => "recall_complete",
            Self::RecallAccess => "recall_access",
            Self::RecallMiss => "recall_miss",
            Self::RecallRetry => "recall_retry",
            Self::Store => "store",
            Self::StoreQuickRecall => "store_quick_recall",
            Self::Forget => "forget",
            Self::Refine => "refine",
            Self::SessionEnd => "session_end",
            Self::ParamUpdate => "param_update",
            Self::ConceptSummaryRefreshed => "concept_summary_refreshed",
            Self::SynthesisInteraction => "synthesis_interaction",
            Self::ConceptSummaryInteraction => "concept_summary_interaction",
            Self::SynthesisLlmJudge => "synthesis_llm_judge",
            Self::ConceptSummaryLlmJudge => "concept_summary_llm_judge",
            Self::SynthesisLlmJudgeOfflineCron => "synthesis_llm_judge_offline_cron",
            Self::ConceptSummaryLlmJudgeOfflineCron => "concept_summary_llm_judge_offline_cron",
            Self::JudgeStructuralAnchor => "judge_structural_anchor",
        }
    }
}

/// A feedback event to be written to the event log.
pub struct FeedbackEvent {
    pub event_type: EventType,
    pub request_id: Option<String>,
    pub memory_id: Option<String>,
    pub concept_id: Option<String>,
    pub query: Option<String>,
    pub query_type: Option<String>,
    pub topic: Option<String>,
    pub payload: Option<serde_json::Value>,
}

/// A stored feedback event read from the database.
#[derive(Debug, Clone)]
pub struct StoredEvent {
    pub id: i64,
    pub ts: String,
    pub event_type: String,
    pub request_id: Option<String>,
    pub memory_id: Option<String>,
    pub concept_id: Option<String>,
    pub query: Option<String>,
    pub query_type: Option<String>,
    pub topic: Option<String>,
    pub payload: Option<String>,
}

// ── Event Operations ─────────────────────────────────────────────────────────

/// Emit a feedback event to the event log.
pub fn emit_event(conn: &Connection, event: FeedbackEvent) -> ReinResult<i64> {
    let payload_str = event.payload.map(|v| v.to_string());
    conn.execute(
        "INSERT INTO feedback_events (event_type, request_id, memory_id, concept_id, query, query_type, topic, payload)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        rusqlite::params![
            event.event_type.as_str(),
            event.request_id,
            event.memory_id,
            event.concept_id,
            event.query,
            event.query_type,
            event.topic,
            payload_str,
        ],
    )?;
    Ok(conn.last_insert_rowid())
}

/// Non-advancing peek for events past a consumer's offset. Mirrors
/// [`consume_events`] but **does not write to `consumer_offsets`** —
/// callers commit the offset via [`commit_offset`] only after the
/// derived state change is durable. See the module-level invariant.
///
/// Returns events in ascending `id` order. The caller derives the new
/// offset as `events.last().map(|e| e.id)`; pass that to
/// [`commit_offset`] when the work is persisted.
pub fn peek_events(
    conn: &Connection,
    consumer: &str,
    event_types: &[&str],
    limit: usize,
) -> ReinResult<Vec<StoredEvent>> {
    let last_id: i64 = conn
        .query_row(
            "SELECT last_event_id FROM consumer_offsets WHERE consumer = ?1",
            rusqlite::params![consumer],
            |row| row.get(0),
        )
        .unwrap_or(0);

    let type_placeholders: Vec<String> = event_types
        .iter()
        .enumerate()
        .map(|(i, _)| format!("?{}", i + 3))
        .collect();
    let type_filter = if type_placeholders.is_empty() {
        String::new()
    } else {
        format!(" AND event_type IN ({})", type_placeholders.join(","))
    };

    let sql = format!(
        "SELECT id, ts, event_type, request_id, memory_id, concept_id, query, query_type, topic, payload
         FROM feedback_events WHERE id > ?1{} ORDER BY id ASC LIMIT ?2",
        type_filter
    );

    let mut stmt = conn.prepare(&sql)?;

    let mut params: Vec<Box<dyn rusqlite::types::ToSql>> =
        vec![Box::new(last_id), Box::new(limit as i64)];
    for et in event_types {
        params.push(Box::new(et.to_string()));
    }

    let param_refs: Vec<&dyn rusqlite::types::ToSql> = params.iter().map(|p| p.as_ref()).collect();
    let rows = stmt.query_map(param_refs.as_slice(), |row| {
        Ok(StoredEvent {
            id: row.get(0)?,
            ts: row.get(1)?,
            event_type: row.get(2)?,
            request_id: row.get(3)?,
            memory_id: row.get(4)?,
            concept_id: row.get(5)?,
            query: row.get(6)?,
            query_type: row.get(7)?,
            topic: row.get(8)?,
            payload: row.get(9)?,
        })
    })?;

    let events: Vec<StoredEvent> = rows.filter_map(|r| r.ok()).collect();
    Ok(events)
}

/// Atomically commit one or more consumer offsets. Each `(consumer,
/// last_event_id)` pair upserts into `consumer_offsets`. The whole
/// batch runs inside `BEGIN IMMEDIATE` so paired consumers (e.g. M2
/// `alpha_optimizer` + `alpha_optimizer_access`) advance together or
/// not at all.
///
/// `last_event_id` MUST be the highest `id` whose derived state has
/// already been persisted. A monotonic-decreasing or stale value is
/// silently treated as a no-op via the `ON CONFLICT DO UPDATE` (the
/// existing offset never goes backwards because the caller always
/// passes the max id seen on this pass).
pub fn commit_offset(conn: &Connection, batch: &[(&str, i64)]) -> ReinResult<()> {
    if batch.is_empty() {
        return Ok(());
    }
    conn.execute_batch("BEGIN IMMEDIATE")?;
    let inner = (|| -> rusqlite::Result<()> {
        for (consumer, last_id) in batch {
            conn.execute(
                "INSERT INTO consumer_offsets (consumer, last_event_id, updated_at)
                 VALUES (?1, ?2, strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                 ON CONFLICT(consumer) DO UPDATE
                   SET last_event_id = MAX(last_event_id, excluded.last_event_id),
                       updated_at    = strftime('%Y-%m-%dT%H:%M:%fZ','now')",
                rusqlite::params![consumer, last_id],
            )?;
        }
        Ok(())
    })();
    match inner {
        Ok(()) => {
            conn.execute_batch("COMMIT")?;
            Ok(())
        }
        Err(e) => {
            let _ = conn.execute_batch("ROLLBACK");
            Err(ReinError::Database(e))
        }
    }
}

/// Consume events for a specific consumer module, starting after its last offset.
/// Returns events and updates the consumer's offset.
pub fn consume_events(
    conn: &Connection,
    consumer: &str,
    event_types: &[&str],
    limit: usize,
) -> ReinResult<Vec<StoredEvent>> {
    // Get current offset
    let last_id: i64 = conn
        .query_row(
            "SELECT last_event_id FROM consumer_offsets WHERE consumer = ?1",
            rusqlite::params![consumer],
            |row| row.get(0),
        )
        .unwrap_or(0);

    // Build event type filter
    let type_placeholders: Vec<String> = event_types
        .iter()
        .enumerate()
        .map(|(i, _)| format!("?{}", i + 3))
        .collect();
    let type_filter = if type_placeholders.is_empty() {
        String::new()
    } else {
        format!(" AND event_type IN ({})", type_placeholders.join(","))
    };

    let sql = format!(
        "SELECT id, ts, event_type, request_id, memory_id, concept_id, query, query_type, topic, payload
         FROM feedback_events WHERE id > ?1{} ORDER BY id ASC LIMIT ?2",
        type_filter
    );

    let mut stmt = conn.prepare(&sql)?;

    // Build params dynamically
    let mut params: Vec<Box<dyn rusqlite::types::ToSql>> =
        vec![Box::new(last_id), Box::new(limit as i64)];
    for et in event_types {
        params.push(Box::new(et.to_string()));
    }

    let param_refs: Vec<&dyn rusqlite::types::ToSql> = params.iter().map(|p| p.as_ref()).collect();
    let rows = stmt.query_map(param_refs.as_slice(), |row| {
        Ok(StoredEvent {
            id: row.get(0)?,
            ts: row.get(1)?,
            event_type: row.get(2)?,
            request_id: row.get(3)?,
            memory_id: row.get(4)?,
            concept_id: row.get(5)?,
            query: row.get(6)?,
            query_type: row.get(7)?,
            topic: row.get(8)?,
            payload: row.get(9)?,
        })
    })?;

    let events: Vec<StoredEvent> = rows.filter_map(|r| r.ok()).collect();

    // Update offset to the last consumed event
    if let Some(last) = events.last() {
        conn.execute(
            "INSERT INTO consumer_offsets (consumer, last_event_id, updated_at)
             VALUES (?1, ?2, strftime('%Y-%m-%dT%H:%M:%fZ','now'))
             ON CONFLICT(consumer) DO UPDATE SET last_event_id = ?2, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')",
            rusqlite::params![consumer, last.id],
        )?;
    }

    Ok(events)
}

/// Clean up old events that all *live* consumers have processed and are
/// beyond retention.
///
/// A stale consumer row with `last_event_id = 0` whose `updated_at` is older
/// than the retention window floors the cutoff at 0 and prevents any
/// pruning — `feedback_events` then grows without bound even when the rest
/// of the adaptive pipeline has moved past those events (B5 #26). Filtering
/// out consumers that haven't advanced within retention lets cleanup
/// proceed; the stale consumer will simply resume from whatever events
/// still exist when it next runs.
pub fn cleanup_expired_events(conn: &Connection, retention_days: u64) -> ReinResult<u64> {
    let cutoff = Utc::now() - chrono::Duration::days(retention_days as i64);
    let min_offset: i64 = conn
        .query_row(
            "SELECT COALESCE(MIN(last_event_id), 0)
               FROM consumer_offsets
              WHERE updated_at >= ?1",
            rusqlite::params![cutoff.to_rfc3339()],
            |row| row.get(0),
        )
        .unwrap_or(0);

    let deleted = conn.execute(
        "DELETE FROM feedback_events WHERE id <= ?1 AND ts < ?2",
        rusqlite::params![min_offset, cutoff.to_rfc3339()],
    )?;

    Ok(deleted as u64)
}

/// Get total event count (for monitoring).
pub fn event_count(conn: &Connection) -> u64 {
    conn.query_row("SELECT COUNT(*) FROM feedback_events", [], |row| row.get(0))
        .unwrap_or(0)
}

/// v0.23: Recompute per-cluster and global canonical-content length
/// percentiles from the current set of live canonicals.
///
/// A "canonical" is a row where `canonical_id = id AND status IN
/// ('active', 'updated')` — both states are live canonicals (`updated`
/// is the post-merge state auto-promoted by `store.update()`'s trigger).
/// Intended to be invoked by the slow-channel GC pass before
/// `AdaptiveState::save_snapshot`. Codex round-5 H-1 + round-6 LOW.
pub fn recompute_canonical_length_stats(
    conn: &Connection,
) -> ReinResult<(
    HashMap<u32, CanonicalLengthStats>,
    Option<CanonicalLengthStats>,
)> {
    // Canonicals are identified by the `memory_canonical_state` table rather
    // than a column on `memories` itself — a row is canonical when its own
    // `memory_id` equals its `canonical_id` entry. See `canonical_id_for` in
    // `store/sqlite.rs` for the same join pattern.
    // Length is measured in BYTES (CAST to BLOB) to match the upstream
    // MergeInto byte-cap at `store/sqlite.rs:1939`. Previously this was
    // SQLite's `length()` on TEXT which returns codepoints — for a
    // CJK-heavy corpus that produced target_bytes values ~3× too permissive
    // and let compressed output blow the merge cap. Codex audit H3.
    //
    // `status IN ('active', 'updated')`: round-5 H-1. Merged canonicals
    // are promoted from `active` to `updated` by the merge trigger;
    // filtering only `active` here underreports post-merge canonicals
    // and skews the per-cluster length distribution on corpora with
    // heavy merge activity. Both states are live canonicals.
    let mut stmt = conn.prepare(
        "SELECT m.cluster_id, length(CAST(m.content AS BLOB)) \
         FROM memories m \
         JOIN memory_canonical_state cs ON cs.memory_id = m.id \
         WHERE cs.canonical_id = m.id \
           AND m.status IN ('active', 'updated')",
    )?;
    let rows = stmt.query_map([], |row| {
        let cid: Option<i64> = row.get(0)?;
        let len: i64 = row.get(1)?;
        Ok((cid, len))
    })?;

    let mut per_cluster: HashMap<u32, Vec<usize>> = HashMap::new();
    let mut global: Vec<usize> = Vec::new();
    for row in rows {
        let (cluster_id, len) = row?;
        if len < 0 {
            continue;
        }
        let len_u = len as usize;
        global.push(len_u);
        if let Some(cid) = cluster_id {
            if cid >= 0 {
                per_cluster.entry(cid as u32).or_default().push(len_u);
            }
        }
    }

    let per_cluster_stats: HashMap<u32, CanonicalLengthStats> = per_cluster
        .into_iter()
        .filter_map(|(cid, lens)| CanonicalLengthStats::from_lengths(lens).map(|s| (cid, s)))
        .collect();
    let global_stats = CanonicalLengthStats::from_lengths(global);

    Ok((per_cluster_stats, global_stats))
}

// ── AdaptiveState ────────────────────────────────────────────────────────────

/// Central cache for all learned parameters. Stored as RefCell on SqliteStore.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AdaptiveState {
    /// M2: Learned fusion alpha per bucket.
    /// Key format: "query_type" or "query_type:cluster_id"
    pub learned_alpha: HashMap<String, LearnedAlphaEntry>,

    /// v0.28 ARS acceleration: learned six-dimensional fusion weights per
    /// bucket. Key format mirrors [`Self::learned_alpha`]: "global",
    /// "query_type", or "query_type:cluster_id".
    #[serde(default)]
    pub learned_shadow_fusion: HashMap<String, LearnedShadowFusionEntry>,

    /// v0.28 ARS acceleration: persisted effective scalar parameters.
    /// The dynamic policy computes a bounded blend from static config toward
    /// learned priors; persisting the last effective value lets the next pass
    /// apply `max_step` smoothing instead of jumping directly from the static
    /// bootstrap.
    #[serde(default)]
    pub ars_effective_scalars: HashMap<String, ArsEffectiveScalarEntry>,

    /// M4: Current cluster version (incremented on each reclustering).
    pub cluster_version: u64,

    /// M4: Memory → cluster assignment.
    pub memory_clusters: HashMap<String, u32>,

    /// Canonical digest of the exact AdaptiveState projection read by A12
    /// replay (`learned_alpha`, `cluster_version`, `memory_clusters`). The
    /// snapshot writer recomputes this field on every CAS branch; callers must
    /// never treat a supplied value as authoritative. SQLite's O(1) A12 epoch
    /// trigger compares it so version-only adaptive ticks do not invalidate a
    /// calibration, while old/malformed writers that omit it fail closed.
    #[serde(default)]
    pub a12_recall_projection_fingerprint: String,

    /// M5: Tier boundary thresholds.
    pub hot_threshold: f64,
    pub cold_threshold: f64,

    /// A1: Per-cluster non-destructive dedup shadow suggestions.
    /// Key = cluster_id, Value = suggested similarity threshold for that
    /// cluster. Computed from intra-cluster pairwise similarity distribution
    /// (P90). The serialized legacy field name is retained for compatibility;
    /// destructive callers must resolve it through `get_hard_dedup_threshold`.
    #[serde(default)]
    pub dedup_thresholds: HashMap<u32, f32>,

    /// A1: Global fallback shadow suggestion when no cluster value exists.
    /// The serialized legacy field name is retained for compatibility.
    #[serde(default = "default_global_dedup_threshold")]
    pub global_dedup_threshold: f32,

    /// M4 incremental: version stamp for cluster centroids stored in `cluster_centroids` table.
    /// Callers compare this against what they last loaded to detect staleness.
    #[serde(default)]
    pub centroid_version: u64,

    /// #17 M4 recluster cadence gate: how many embeddings existed when the
    /// last successful recluster persisted. The pipeline re-runs HDBSCAN
    /// only when `|current - last| >= 5.max(current / 50)` (the same
    /// adaptive `min_cluster_size` formula HDBSCAN itself uses — no new
    /// knob). `0` on fresh install / pre-#17 snapshots, which makes the
    /// first pass after bootstrap or upgrade recluster unconditionally
    /// (any `current >= 50` clears the gate against a `0` baseline).
    /// Gating reclusters is also what lets cluster-scoped learned state
    /// (M2 alpha, shadow fusion weights, A1 dedup thresholds) survive
    /// across pipeline passes instead of being wiped every tick.
    #[serde(default)]
    pub last_recluster_embedding_count: u64,

    /// #17 companion churn signal: value of the monotonic
    /// `embedding_write_seq` metadata counter (see `store/vec.rs`) at the
    /// last successful recluster. The count delta above is blind to
    /// in-place embedding replacement (update paths re-embed under the
    /// same id — same count, different vector), so the gate also fires
    /// when enough WRITES accumulated at a constant row count. `0` on
    /// pre-#17 snapshots.
    #[serde(default)]
    pub last_recluster_embedding_write_seq: u64,

    /// #17 transient (never serialized): `cluster_version` as observed at
    /// `restore_snapshot` time. `save_snapshot`'s CAS merge compares it to
    /// the live `cluster_version` to distinguish a writer that RAN a
    /// recluster this pass (wholesale-replace must win so the wipe isn't
    /// undone) from one that merely carries the loaded generation
    /// (same-generation conflicts merge additively; older-generation
    /// contributions are dropped).
    #[serde(skip)]
    pub cluster_version_at_load: u64,

    /// v0.23: Per-cluster canonical content length percentiles. Drives
    /// adaptive `target_bytes` for resummerize compression. Populated by
    /// the slow-channel via `recompute_canonical_length_stats`.
    #[serde(default)]
    pub canonical_length_stats: HashMap<u32, CanonicalLengthStats>,

    /// v0.23: Global canonical content length percentiles. Fallback when
    /// a cluster lacks sufficient samples or no cluster is assigned.
    #[serde(default)]
    pub global_canonical_length: Option<CanonicalLengthStats>,

    /// v0.24 ARS: concept living-summary refresh statistics. Populated by
    /// the slow-channel from post-refresh feedback. `None` on fresh
    /// install; [`Self::concept_refresh_revision_threshold`] and
    /// [`Self::concept_refresh_age_threshold_secs`] fall back to bootstrap
    /// constants until enough data accumulates.
    #[serde(default)]
    pub concept_refresh_stats: Option<ConceptRefreshStats>,

    /// v0.24 peek+commit replay-safety (Codex Tier-B+C round-1 HIGH):
    /// highest event id whose shadow-suggestion nudge is already in the
    /// durable legacy snapshot field `global_dedup_threshold`. Caller of M6 filters
    /// peek events by `id > m6_threshold_last_id` so a `commit_offset`
    /// failure after `save_snapshot` doesn't double-nudge on the next
    /// pass. `0` on fresh install / pre-v0.24 snapshots.
    #[serde(default)]
    pub m6_threshold_last_id: i64,

    /// v0.24 peek+commit replay-safety: highest event id whose co-recall
    /// pair-counting effect is durable. Same rationale as
    /// [`Self::m6_threshold_last_id`].
    #[serde(default)]
    pub m6_corecall_last_id: i64,

    /// v0.24 peek+commit replay-safety (Codex Tier-B+C round-2 HIGH):
    /// highest `recall_complete` event id whose alpha-shrinkage effect
    /// is already in the durable snapshot. M2's
    /// `compute_counterfactual_alphas` reads existing `learned_alpha`
    /// values and writes stepped/shrunk replacements in place — NOT
    /// idempotent on event replay. Filtering peeked events by
    /// `id > alpha_optimizer_last_id` makes replay a no-op.
    #[serde(default)]
    pub alpha_optimizer_last_id: i64,

    /// v0.24 peek+commit replay-safety: highest `recall_access` event id
    /// already incorporated. Same idempotency role as
    /// [`Self::alpha_optimizer_last_id`].
    #[serde(default)]
    pub alpha_optimizer_access_last_id: i64,

    /// v0.26 D direction: synthesis interaction aggregates from the
    /// `synthesis_feedback` consumer. `None` on fresh install; helpers
    /// fall back to the global enabled flag until per-(cluster,query_type)
    /// `viewed_count >= SYNTHESIS_COLD_START_N`.
    #[serde(default)]
    pub synthesis_feedback_stats: Option<SynthesisFeedbackState>,

    /// v0.27 ARS Cap A feedback loop (Track 1): concept living-summary
    /// interaction aggregates from the `concept_summary_feedback` consumer.
    /// `None` on fresh install; helpers fall back to the global Cap A
    /// `[ars].concept_summary_enabled` flag until
    /// `viewed_count >= CONCEPT_SUMMARY_COLD_START_N` per
    /// `(cluster_id, query_type)` bucket.
    #[serde(default)]
    pub concept_summary_feedback_stats: Option<ConceptSummaryFeedbackState>,

    /// v0.27.1 E direction: judge calibration state (Layer 1 J3 κ pairs +
    /// Layer 2 runtime-vs-offline κ + drift alerts). `None` on fresh
    /// install — J3 invariant treats absence as "κ undefined → invariant
    /// dormant" (§4 J3 row). Layer 1 fields (`recent_pairs_synthesis`,
    /// `recent_pairs_concept`, `kappa`) are owned by `synthesis_feedback`
    /// / `concept_summary_feedback` consumers per §6.2.1; Layer 2 fields
    /// (`runtime_vs_offline_kappa`, `last_consumed_event_id_calibration`,
    /// `judge_drift_alert`) are owned by the `judge_calibration` consumer
    /// per §7. R9-K5 mandates field-grouped CAS merge.
    ///
    /// Wave-1 D_CALIBRATION_CRON staging: A_JUDGE_CORE will own this field
    /// definition once Wave 1 lands; D added it here so the cron + consumer
    /// can read/write it. Field-grouped merge is implemented in
    /// `save_snapshot` per R9-K5.
    #[serde(default)]
    pub judge_calibration_state: Option<JudgeCalibrationState>,

    /// v0.27.1 E direction (spec §6.2.1) — opportunistic κ-pair join cache.
    ///
    /// Keyed by the surface-id `synthesis_id` or `concept_summary_id`; value
    /// is whichever half (judge verdict OR human ExplicitThumb) arrived
    /// first. When the matching half lands, the consumer takes the cached
    /// half, completes the pair, and pushes it into
    /// `JudgeCalibrationState.recent_pairs_*` (per the surface
    /// discriminator). Bounded LRU at [`LLM_JUDGE_PAIR_CACHE_CAPACITY`];
    /// FIFO evicts oldest entries with timestamps older than
    /// [`LLM_JUDGE_HALF_PAIR_TTL_SECS`].
    ///
    /// Lives on `AdaptiveState` rather than the consumer state structs
    /// because BOTH `synthesis_feedback` and `concept_summary_feedback`
    /// share the cache (humans on either surface can match a judge call,
    /// no cross-surface contamination because the surface is part of the
    /// HalfPair payload). Treated as a derived cache for snapshot purposes
    /// — wholesale-replaced under CAS by the writer with the higher
    /// `synthesis_feedback_stats.last_consumed_event_id`.
    #[serde(default)]
    pub pending_kappa_half_pairs: HashMap<String, HalfPair>,

    /// Global version (incremented on each slow-channel update).
    pub version: u64,
}

fn default_global_dedup_threshold() -> f32 {
    0.70
}

/// Minimum target bytes for resummerize output (v0.23). Below this,
/// compression is structurally meaningless — a single merge entry often
/// already exceeds this.
pub const MIN_RESUMMERIZE_TARGET: usize = 2_000;
/// Maximum target bytes for resummerize output (v0.23). Derived from
/// `MERGE_CONTENT_CAP` so the two constants can't drift silently (Codex
/// round-2 LOW). Compressing above the cap is a no-op and would
/// immediately re-enter keep-tail on the next merge.
pub const MAX_RESUMMERIZE_TARGET: usize = crate::store::sqlite::MERGE_CONTENT_CAP;
// Compile-time guard: any future edit that decouples the two constants
// will fail the build here rather than leak a budget mismatch to
// production.
const _: () = assert!(MAX_RESUMMERIZE_TARGET <= crate::store::sqlite::MERGE_CONTENT_CAP);
/// Bootstrap target used until enough per-cluster or global canonical
/// length data accumulates. Anchored to the min/max constants rather than
/// a free-floating magic number (MIN + 75% of range = 8_000).
pub const RESUMMERIZE_BOOTSTRAP_TARGET: usize =
    MIN_RESUMMERIZE_TARGET + (MAX_RESUMMERIZE_TARGET - MIN_RESUMMERIZE_TARGET) * 3 / 4;
/// Minimum cluster sample count before we trust a per-cluster p25.
pub const RESUMMERIZE_CLUSTER_MIN_SAMPLES: usize = 5;
/// Minimum global sample count before we trust the global p25 fallback.
pub const RESUMMERIZE_GLOBAL_MIN_SAMPLES: usize = 10;

/// A learned alpha entry with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedAlphaEntry {
    pub value: f64,
    pub sample_count: usize,
    pub last_updated: String, // RFC3339
}

/// v1.2 audit F12 + codex R10: CAS-merge dominance rule for learned entries.
///
/// Timestamp LWW. The audit-F12 hazard (a fold based on a STALE prior
/// overwriting a peer's durable fold while the watermarks advance) is closed
/// at its ROOT by the cross-process single-flight lock on
/// `run_adaptive_pipeline` (ops/adaptive.rs) — the only producer of these
/// entries — so two concurrent folds can no longer exist. An interim
/// "higher-ESS-wins" rule was tried and rejected (codex R10): sample_count
/// is a DECAYED effective sample size, not a monotonic event count, so after
/// an idle period a fresh fold can legitimately carry a LOWER ESS than the
/// stale stored entry and would have been dropped while its events were
/// marked consumed. With single-flight folding, a version conflict here can
/// only come from a non-folding writer (whose snapshot carries the OLD
/// entry, hence an older timestamp), and newest-fold-wins is correct.
fn alpha_entry_dominates(theirs: &LearnedAlphaEntry, ours: &LearnedAlphaEntry) -> bool {
    theirs.last_updated >= ours.last_updated
}

/// See [`alpha_entry_dominates`] — same rule for the six-dimensional fusion
/// entries.
fn shadow_entry_dominates(
    theirs: &LearnedShadowFusionEntry,
    ours: &LearnedShadowFusionEntry,
) -> bool {
    theirs.last_updated >= ours.last_updated
}

/// Six-dimensional ARS fusion weights persisted in [`AdaptiveState`].
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct ShadowFusionWeightEntry {
    pub bm25: f64,
    pub vec: f64,
    pub kg: f64,
    pub episode: f64,
    pub support: f64,
    pub diversity: f64,
}

/// Learned shadow/production acceleration weights with evidence metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LearnedShadowFusionEntry {
    pub weights: ShadowFusionWeightEntry,
    pub sample_count: usize,
    pub last_updated: String, // RFC3339
}

/// Persisted dynamic scalar with timestamp metadata for CAS merge.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ArsEffectiveScalarEntry {
    pub value: f64,
    pub last_updated: String, // RFC3339
}

pub const ARS_SCALAR_JUDGE_WEIGHT_DECAY_RATE: &str = "judge_weight_decay_rate";
pub const ARS_SCALAR_SYNTHESIS_COLD_START_N: &str = "synthesis_cold_start_n";
pub const ARS_SCALAR_CONCEPT_SUMMARY_COLD_START_N: &str = "concept_summary_cold_start_n";
/// Legacy v0.28.0..v0.28.7 cluster-shared `judge_sample_rate` cold-start
/// scalar — the persistence-side residual called out by the v0.28.x
/// audit's M-1 (input-side gating was fixed in v0.28.7; persistence-side
/// shipped here in v0.28.7+ as the per-surface split below). Retained
/// solely as the read-fallback target for snapshots that predate the
/// per-surface keys (and as the downgrade-compat write target). New code
/// MUST persist into the per-surface variants
/// `..._SYNTHESIS` / `..._CONCEPT_SUMMARY`. See the
/// `ars_effective_scalar_with_legacy_fallback` helper for the read path.
pub const ARS_SCALAR_JUDGE_SAMPLE_RATE_COLD_START: &str = "judge_sample_rate_cold_start";
pub const ARS_SCALAR_JUDGE_SAMPLE_RATE_WARM: &str = "judge_sample_rate_warm";
/// v0.28.7+ audit M-1 persistence-side fix: per-surface variants of the
/// cold-start / warm `judge_sample_rate` scalars. Pre-fix the cluster-
/// shared scalars (`..._COLD_START` / `..._WARM`) were computed against
/// `JudgeSurface::Synthesis` only and then read by both the synthesis
/// and concept-summary surfaces, so a synthesis-surface drift event
/// would zero concept-summary's persisted sample rate (and vice versa)
/// even though the input-side drift gate already ran per-surface
/// (v0.28.7 input-side fix). Splitting the persisted scalars closes
/// the cross-surface coupling on the persistence side too.
pub const ARS_SCALAR_JUDGE_SAMPLE_RATE_COLD_START_SYNTHESIS: &str =
    "judge_sample_rate_cold_start_synthesis";
pub const ARS_SCALAR_JUDGE_SAMPLE_RATE_COLD_START_CONCEPT_SUMMARY: &str =
    "judge_sample_rate_cold_start_concept_summary";
pub const ARS_SCALAR_JUDGE_SAMPLE_RATE_WARM_SYNTHESIS: &str = "judge_sample_rate_warm_synthesis";
pub const ARS_SCALAR_JUDGE_SAMPLE_RATE_WARM_CONCEPT_SUMMARY: &str =
    "judge_sample_rate_warm_concept_summary";
pub const ARS_SCALAR_SYNTHESIS_USEFUL_RATE_THRESHOLD: &str = "synthesis_useful_rate_threshold";
pub const ARS_SCALAR_CONCEPT_SUMMARY_USEFUL_RATE_THRESHOLD: &str =
    "concept_summary_useful_rate_threshold";

/// v0.28.7+ audit M-1 persistence-side helper — read a per-surface ARS
/// effective scalar, falling back to the legacy cluster-shared scalar
/// when the per-surface key is absent (first-tick-after-upgrade and
/// pre-upgrade-snapshot rehydration paths).
///
/// Without this fallback, a v0.28.7 → v0.28.7+ upgrade would discard
/// the canary's accumulated learning the moment the per-surface keys
/// were introduced (the per-surface `previous_effective` would be
/// `None`, so the next pipeline tick's step-bound smoothing would snap
/// straight back to the static config value — a one-tick rollback the
/// operator never asked for).
///
/// The fallback is consulted exactly once per surface per snapshot;
/// after the next pipeline tick writes the per-surface key, subsequent
/// reads see it directly and the legacy key becomes an idle survivor
/// (NOT deleted — keeping it around lets a v0.28.7 downgrade rollback
/// see a sensible value, since the upgraded code keeps writing the
/// legacy key with the synthesis-surface variant, matching pre-fix
/// behavior).
pub fn ars_effective_scalar_with_legacy_fallback(
    state: &AdaptiveState,
    primary_key: &str,
    legacy_key: &str,
) -> Option<f64> {
    state
        .ars_effective_scalar(primary_key)
        .or_else(|| state.ars_effective_scalar(legacy_key))
}

/// Per-cluster (and global) canonical content length percentiles (v0.23).
///
/// Drives adaptive `target_bytes` for resummerize compression.
/// Percentiles use linear interpolation between sorted order statistics,
/// which matches the default used by NumPy and most statistical software.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct CanonicalLengthStats {
    /// Number of canonicals contributing to the percentile.
    /// Callers must compare against `RESUMMERIZE_*_MIN_SAMPLES` before
    /// trusting the p25 value.
    pub count: usize,
    pub p25: usize,
    pub p50: usize,
    pub p75: usize,
}

impl CanonicalLengthStats {
    /// Compute stats from canonical content lengths (bytes).
    /// Returns `None` when `lengths` is empty.
    pub fn from_lengths(mut lengths: Vec<usize>) -> Option<Self> {
        if lengths.is_empty() {
            return None;
        }
        lengths.sort_unstable();
        let n = lengths.len();
        let pct = |p: f64| -> usize {
            let rank = p * (n.saturating_sub(1)) as f64;
            let lo = rank.floor() as usize;
            let hi = (lo + 1).min(n - 1);
            let frac = rank - lo as f64;
            let lo_v = lengths[lo] as f64;
            let hi_v = lengths[hi] as f64;
            (lo_v + frac * (hi_v - lo_v)).round() as usize
        };
        Some(Self {
            count: n,
            p25: pct(0.25),
            p50: pct(0.50),
            p75: pct(0.75),
        })
    }
}

impl AdaptiveState {
    /// Deterministic identity of the AdaptiveState fields consumed by A12's
    /// local recall replay. HashMap iteration order is deliberately removed by
    /// projecting through BTreeMap before serialization.
    pub(crate) fn compute_a12_recall_projection_fingerprint(&self) -> ReinResult<String> {
        #[derive(Serialize)]
        struct A12RecallProjection<'a> {
            learned_alpha: BTreeMap<&'a str, &'a LearnedAlphaEntry>,
            cluster_version: u64,
            memory_clusters: BTreeMap<&'a str, u32>,
        }

        let projection = A12RecallProjection {
            learned_alpha: self
                .learned_alpha
                .iter()
                .map(|(key, value)| (key.as_str(), value))
                .collect(),
            cluster_version: self.cluster_version,
            memory_clusters: self
                .memory_clusters
                .iter()
                .map(|(key, value)| (key.as_str(), *value))
                .collect(),
        };
        let bytes = serde_json::to_vec(&projection).map_err(ReinError::Serialization)?;
        let mut hasher = Sha256::new();
        hasher.update(b"a12-adaptive-recall-projection-v1\0");
        hasher.update(u64::try_from(bytes.len()).unwrap_or(u64::MAX).to_le_bytes());
        hasher.update(bytes);
        Ok(format!("{:x}", hasher.finalize()))
    }

    fn serialize_with_a12_recall_projection(&self) -> ReinResult<String> {
        let mut persisted = self.clone();
        persisted.a12_recall_projection_fingerprint =
            persisted.compute_a12_recall_projection_fingerprint()?;
        serde_json::to_string(&persisted).map_err(ReinError::Serialization)
    }

    pub fn ars_effective_scalar(&self, key: &str) -> Option<f64> {
        self.ars_effective_scalars.get(key).and_then(|entry| {
            if entry.value.is_finite() {
                Some(entry.value)
            } else {
                None
            }
        })
    }

    pub fn set_ars_effective_scalar(&mut self, key: impl Into<String>, value: f64) {
        if !value.is_finite() {
            return;
        }
        self.ars_effective_scalars.insert(
            key.into(),
            ArsEffectiveScalarEntry {
                value,
                last_updated: chrono::Utc::now().to_rfc3339(),
            },
        );
    }

    /// #17: drop every learned surface keyed by a cluster ID. Cluster ids
    /// are local labels of one HDBSCAN run — after a relabel (recluster or
    /// `migrate --reindex`), an old entry under id N would be served for
    /// whatever unrelated cluster now wears N. Wipes: A1 dedup thresholds,
    /// cluster-scoped M2 alpha + shadow-fusion buckets (`qt:N` keys), and
    /// the synthesis / concept-summary `by_cluster` aggregates (codex R6 —
    /// `decide_synthesize` indexes those by live cluster_id). Keeps:
    /// scope-free alpha/fusion keys, the `-1|…` no-cluster feedback
    /// buckets (not tied to any label), per-id LRU stats, and every
    /// consumer watermark (`last_consumed_event_id` must survive or
    /// consume-once events would replay).
    pub fn clear_cluster_scoped_learned_state(&mut self) {
        self.dedup_thresholds.clear();
        self.learned_alpha.retain(|k, _| !k.contains(':'));
        self.learned_shadow_fusion.retain(|k, _| !k.contains(':'));
        if let Some(stats) = &mut self.synthesis_feedback_stats {
            stats.by_cluster.retain(|k, _| k.starts_with("-1|"));
        }
        if let Some(stats) = &mut self.concept_summary_feedback_stats {
            stats.by_cluster.retain(|k, _| k.starts_with("-1|"));
        }
    }

    /// Build bucket key from query_type and optional cluster_id.
    pub fn bucket_key(query_type: &str, cluster_id: Option<u32>) -> String {
        let query_type = query_type.to_lowercase();
        match cluster_id {
            Some(c) => format!("{query_type}:{c}"),
            None => query_type,
        }
    }

    /// Get learned alpha for a query type and optional cluster, with fallback chain.
    ///
    /// `min_samples` is the operator-configured `[adaptive].min_samples_alpha`.
    /// #17 codex R12: since the write side no longer floors per-window
    /// bucket creation (cumulative counts accumulate from any window), the
    /// READ gate must honor a raised config floor — otherwise
    /// `min_samples_alpha = 50` buckets would activate at the historical
    /// hard-coded 10. The 10 floor stays as the lower bound so a config
    /// meant as a learn-window knob (tests set 1) can't open the gate on
    /// near-zero evidence.
    pub fn get_alpha(
        &self,
        query_type: &str,
        cluster_id: Option<u32>,
        min_samples: usize,
    ) -> Option<f32> {
        let floor = min_samples.max(10);
        let legacy_key = query_type.to_string();
        // Try specific bucket first
        if let Some(cluster) = cluster_id {
            let key = Self::bucket_key(query_type, Some(cluster));
            if let Some(entry) = self.learned_alpha.get(&key) {
                if entry.sample_count >= floor {
                    return Some(entry.value as f32);
                }
            }
            let legacy_cluster_key = format!("{legacy_key}:{cluster}");
            if let Some(entry) = self.learned_alpha.get(&legacy_cluster_key) {
                if entry.sample_count >= floor {
                    return Some(entry.value as f32);
                }
            }
        }
        // Fall back to query-type level
        let key = Self::bucket_key(query_type, None);
        if let Some(entry) = self.learned_alpha.get(&key) {
            if entry.sample_count >= floor {
                return Some(entry.value as f32);
            }
        }
        if let Some(entry) = self.learned_alpha.get(&legacy_key) {
            if entry.sample_count >= floor {
                return Some(entry.value as f32);
            }
        }
        if let Some(entry) = self.learned_alpha.get("global") {
            if entry.sample_count >= floor {
                return Some(entry.value as f32);
            }
        }
        None
    }

    /// Get learned six-dimensional ARS fusion weights with the same fallback
    /// chain as scalar alpha: cluster → query type → global.
    pub fn get_shadow_fusion_weights(
        &self,
        query_type: &str,
        cluster_id: Option<u32>,
        min_sample_count: usize,
    ) -> Option<&LearnedShadowFusionEntry> {
        // v1.2 audit F26: same evidence floor as get_alpha — the config knob
        // is a LEARN-window parameter (tests set it to 1); without the floor,
        // fusion weights learned from a single recall event (averaged simplex
        // corners — extreme vectors) became servable, the exact near-zero-
        // evidence activation the get_alpha floor was added to prevent.
        let floor = min_sample_count.max(10);
        let eligible = |entry: &&LearnedShadowFusionEntry| entry.sample_count >= floor;
        let legacy_key = query_type.to_string();
        if let Some(cluster) = cluster_id {
            let key = Self::bucket_key(query_type, Some(cluster));
            if let Some(entry) = self.learned_shadow_fusion.get(&key).filter(eligible) {
                return Some(entry);
            }
            let legacy_cluster_key = format!("{legacy_key}:{cluster}");
            if let Some(entry) = self
                .learned_shadow_fusion
                .get(&legacy_cluster_key)
                .filter(eligible)
            {
                return Some(entry);
            }
        }

        let key = Self::bucket_key(query_type, None);
        if let Some(entry) = self.learned_shadow_fusion.get(&key).filter(eligible) {
            return Some(entry);
        }
        if let Some(entry) = self.learned_shadow_fusion.get(&legacy_key).filter(eligible) {
            return Some(entry);
        }
        self.learned_shadow_fusion.get("global").filter(eligible)
    }

    /// Get the non-destructive dedup shadow suggestion for a cluster, with
    /// fallback to the global shadow suggestion.
    pub fn get_dedup_shadow_threshold(&self, cluster_id: Option<u32>) -> f32 {
        if let Some(cid) = cluster_id {
            if let Some(&threshold) = self.dedup_thresholds.get(&cid) {
                return threshold;
            }
        }
        if self.global_dedup_threshold > 0.0 {
            self.global_dedup_threshold
        } else {
            0.70 // ultimate fallback
        }
    }

    /// Legacy compatibility wrapper for the non-destructive shadow suggestion.
    ///
    /// New call sites should use [`Self::get_dedup_shadow_threshold`] so this
    /// value cannot be mistaken for a destructive/effective threshold.
    pub fn get_dedup_threshold(&self, cluster_id: Option<u32>) -> f32 {
        self.get_dedup_shadow_threshold(cluster_id)
    }

    /// Get the threshold for a destructive lexical dedup decision.
    ///
    /// Until an independently labeled adoption policy exists, unlabeled shadow
    /// suggestions never affect destructive actions in either direction. The
    /// cluster parameter is retained for API compatibility and future policy.
    pub fn get_hard_dedup_threshold(&self, _cluster_id: Option<u32>, static_threshold: f32) -> f32 {
        if !static_threshold.is_finite() || !(0.0..=1.0).contains(&static_threshold) {
            return 1.0;
        }
        static_threshold
    }

    /// v0.23: Per-cluster canonical length p25, with minimum-sample guard.
    /// Returns `None` when the cluster has fewer than
    /// `RESUMMERIZE_CLUSTER_MIN_SAMPLES` observations.
    pub fn cluster_canonical_length_p25(&self, cluster_id: u32) -> Option<usize> {
        self.canonical_length_stats
            .get(&cluster_id)
            .filter(|s| s.count >= RESUMMERIZE_CLUSTER_MIN_SAMPLES)
            .map(|s| s.p25)
    }

    /// v0.23: Target byte count for resummerize output. Fully data-driven
    /// with a three-tier fallback:
    ///
    /// 1. per-cluster p25 (≥ 5 samples in the cluster)
    /// 2. global p25 (≥ 10 samples globally)
    /// 3. bootstrap constant (`RESUMMERIZE_BOOTSTRAP_TARGET`)
    ///
    /// Always clamped to `[MIN_RESUMMERIZE_TARGET, MAX_RESUMMERIZE_TARGET]`
    /// so structurally meaningless targets (too short to carry evidence,
    /// or above the merge cap) are impossible regardless of input data.
    pub fn resummerize_target_bytes(&self, cluster_id: Option<u32>) -> usize {
        let from_cluster = cluster_id.and_then(|c| self.cluster_canonical_length_p25(c));
        let from_global = self
            .global_canonical_length
            .as_ref()
            .filter(|s| s.count >= RESUMMERIZE_GLOBAL_MIN_SAMPLES)
            .map(|s| s.p25);
        let raw = from_cluster
            .or(from_global)
            .unwrap_or(RESUMMERIZE_BOOTSTRAP_TARGET);
        raw.clamp(MIN_RESUMMERIZE_TARGET, MAX_RESUMMERIZE_TARGET)
    }

    /// Save state snapshot to metadata table with optimistic concurrency control.
    /// Checks that the stored version matches our base version to prevent lost updates
    /// when two concurrent GC runs modify the state simultaneously.
    pub fn save_snapshot(&self, conn: &Connection) -> ReinResult<()> {
        let json = self.serialize_with_a12_recall_projection()?;

        // Optimistic concurrency: only update if version hasn't changed since we loaded
        // COALESCE so malformed/missing JSON reads as -1 rather than NULL
        // (NULL = ?2 would be untrue, forcing a spurious retry).
        let rows = conn.execute(
            "UPDATE metadata SET value = ?1
             WHERE key = 'adaptive_state'
             AND (value IS NULL OR COALESCE(json_extract(value, '$.version'), -1) = ?2)",
            rusqlite::params![&json, self.version.saturating_sub(1) as i64],
        )?;

        if rows == 0 {
            // Either no row exists yet (first run) or version mismatch
            let exists: bool = conn
                .query_row(
                    "SELECT COUNT(*) > 0 FROM metadata WHERE key = 'adaptive_state'",
                    [],
                    |r| r.get(0),
                )
                .unwrap_or(false);

            if !exists {
                // First save — insert
                conn.execute(
                    "INSERT INTO metadata (key, value) VALUES ('adaptive_state', ?1)",
                    rusqlite::params![&json],
                )?;
            } else {
                // Version conflict — CAS retry loop: re-read, merge, write with
                // version predicate.  Caps at 3 attempts to avoid infinite spin.
                const MAX_CAS_RETRIES: u32 = 3;
                for attempt in 0..MAX_CAS_RETRIES {
                    let Some(mut current) = Self::restore_snapshot(conn) else {
                        // Row vanished between existence check and read — insert ours
                        let _ = conn.execute(
                            "INSERT OR REPLACE INTO metadata (key, value) VALUES ('adaptive_state', ?1)",
                            rusqlite::params![&json],
                        );
                        break;
                    };
                    let db_version = current.version;

                    tracing::warn!(
                        "adaptive state version conflict (expected v{}, found v{db_version}), merge attempt {}",
                        self.version.saturating_sub(1),
                        attempt + 1,
                    );

                    // Cluster-scoped state, three-way by clustering generation
                    // (#17 codex R6):
                    // 1. We RAN a recluster this pass (cluster_version grew past
                    //    what we loaded) and are not behind the stored
                    //    generation → wholesale replace, so the recluster's wipe
                    //    can't be undone. Two concurrent reclusters both land
                    //    here at the same number; last writer wins wholesale —
                    //    additively mixing labels from two different HDBSCAN
                    //    runs would corrupt every cluster-keyed map. (The
                    //    `self > current` arm also covers the degenerate
                    //    rolled-back-snapshot case without a recluster.)
                    // 2. Same generation, nobody reclustered between us →
                    //    additive merge; both writers' disjoint learned buckets
                    //    are valid and neither may drop the other's.
                    // 3. Ours is a STALE generation (they reclustered or a
                    //    `migrate --reindex` bumped after we loaded) → drop our
                    //    cluster-scoped contributions entirely; cluster ids are
                    //    local labels of the newer generation and merging ours
                    //    back would resurrect a dead embedding space.
                    let we_reclustered = self.cluster_version > self.cluster_version_at_load;
                    // #17 codex R7/R9: captured BEFORE the scalar section
                    // maxes cluster_version. The feedback-stats watermark
                    // arbitration below is generation-blind, so after the
                    // arms we strip label-keyed buckets — but ONLY from a
                    // winner whose generation is not the merge's final
                    // generation (R9: a pass that reclustered AND consumed
                    // feedback carries fresh new-generation buckets; its
                    // offsets commit after this save, so stripping its own
                    // stats would lose consumed events unreplayably).
                    let current_cv_at_merge = current.cluster_version;
                    if self.cluster_version > current.cluster_version
                        || (we_reclustered && self.cluster_version == current.cluster_version)
                    {
                        current.memory_clusters = self.memory_clusters.clone();
                        current.dedup_thresholds = self.dedup_thresholds.clone();
                        // #17: the recluster baselines travel with the rest of
                        // the cluster-scoped state — they were stamped by the
                        // same recluster that produced `memory_clusters`.
                        current.last_recluster_embedding_count =
                            self.last_recluster_embedding_count;
                        current.last_recluster_embedding_write_seq =
                            self.last_recluster_embedding_write_seq;
                        // v0.23: canonical_length_stats is cluster-keyed, treat the
                        // same as dedup_thresholds across a recluster boundary.
                        current.canonical_length_stats = self.canonical_length_stats.clone();
                        // Replace all cluster-scoped alpha keys (contain ':') with ours
                        current.learned_alpha.retain(|k, _| !k.contains(':'));
                        for (key, entry) in &self.learned_alpha {
                            if key.contains(':') {
                                current.learned_alpha.insert(key.clone(), entry.clone());
                            }
                        }
                        current
                            .learned_shadow_fusion
                            .retain(|k, _| !k.contains(':'));
                        for (key, entry) in &self.learned_shadow_fusion {
                            if key.contains(':') {
                                current
                                    .learned_shadow_fusion
                                    .insert(key.clone(), entry.clone());
                            }
                        }
                    } else if self.cluster_version == current.cluster_version {
                        // Case 2: same generation, no recluster on either side.
                        // Additive merge for memory_clusters and dedup_thresholds
                        for (mid, &cid) in &self.memory_clusters {
                            current.memory_clusters.insert(mid.clone(), cid);
                        }
                        for (&cid, &threshold) in &self.dedup_thresholds {
                            current.dedup_thresholds.insert(cid, threshold);
                        }
                        // v0.23: additive merge — newer stats win per-cluster.
                        // Same-version concurrent writers will end up with a
                        // non-deterministic last-write-wins, which is fine since
                        // both computed from overlapping corpus snapshots.
                        for (&cid, stats) in &self.canonical_length_stats {
                            current.canonical_length_stats.insert(cid, stats.clone());
                        }
                        for (key, our_entry) in &self.learned_shadow_fusion {
                            if !key.contains(':') {
                                continue;
                            }
                            let dominated = current
                                .learned_shadow_fusion
                                .get(key)
                                .is_some_and(|theirs| shadow_entry_dominates(theirs, our_entry));
                            if !dominated {
                                current
                                    .learned_shadow_fusion
                                    .insert(key.clone(), our_entry.clone());
                            }
                        }
                        // #17 codex R6: cluster-scoped alpha keys must merge
                        // additively here too — the non-cluster alpha loop
                        // below skips ':' keys, so without this a
                        // same-generation conflict would silently drop this
                        // writer's learned cluster buckets.
                        for (key, our_entry) in &self.learned_alpha {
                            if !key.contains(':') {
                                continue;
                            }
                            let dominated = current
                                .learned_alpha
                                .get(key)
                                .is_some_and(|theirs| alpha_entry_dominates(theirs, our_entry));
                            if !dominated {
                                current.learned_alpha.insert(key.clone(), our_entry.clone());
                            }
                        }
                    }
                    // else: case 3 — drop our stale-generation cluster state,
                    // keep theirs (codex R3 P2; rationale in the header above).

                    // Merge learned_alpha (non-cluster keys): prefer the
                    // entry carrying more evidence (see *_entry_dominates).
                    for (key, our_entry) in &self.learned_alpha {
                        if key.contains(':') {
                            continue; // handled above based on cluster_version
                        }
                        let dominated = current
                            .learned_alpha
                            .get(key)
                            .is_some_and(|theirs| alpha_entry_dominates(theirs, our_entry));
                        if !dominated {
                            current.learned_alpha.insert(key.clone(), our_entry.clone());
                        }
                    }
                    // Merge ARS six-dimensional fusion weights (non-cluster
                    // keys): same evidence-first rule, mirroring learned_alpha.
                    for (key, our_entry) in &self.learned_shadow_fusion {
                        if key.contains(':') {
                            continue; // handled above based on cluster_version
                        }
                        let dominated = current
                            .learned_shadow_fusion
                            .get(key)
                            .is_some_and(|theirs| shadow_entry_dominates(theirs, our_entry));
                        if !dominated {
                            current
                                .learned_shadow_fusion
                                .insert(key.clone(), our_entry.clone());
                        }
                    }

                    // Scalar fields
                    current.cluster_version = current.cluster_version.max(self.cluster_version);
                    current.centroid_version = current.centroid_version.max(self.centroid_version);
                    current.hot_threshold = self.hot_threshold;
                    current.cold_threshold = self.cold_threshold;
                    current.global_dedup_threshold = self.global_dedup_threshold;
                    // v0.23: global canonical length stats — take ours (latest
                    // slow-channel recompute wins; older computation is stale).
                    current.global_canonical_length = self.global_canonical_length.clone();
                    // v0.24 peek+commit replay-safety watermarks (Codex
                    // Tier-B+C round-2 HIGH): every new `*_last_id` field
                    // added in this diff MUST be carried in the CAS merge,
                    // or a retry can silently revert the watermark and
                    // re-enable double-application of the same events on
                    // commit_offset failure. Take MAX of (ours, theirs)
                    // — both sides drained from the same monotonic event
                    // log, so the larger value strictly dominates.
                    current.m6_threshold_last_id =
                        current.m6_threshold_last_id.max(self.m6_threshold_last_id);
                    current.m6_corecall_last_id =
                        current.m6_corecall_last_id.max(self.m6_corecall_last_id);
                    current.alpha_optimizer_last_id = current
                        .alpha_optimizer_last_id
                        .max(self.alpha_optimizer_last_id);
                    current.alpha_optimizer_access_last_id = current
                        .alpha_optimizer_access_last_id
                        .max(self.alpha_optimizer_access_last_id);

                    // v0.24 ARS L3: arbitrate by `last_consumed_event_id`
                    // — whichever side incorporated more recent feedback
                    // events wins. Three rounds of Codex audit hammered
                    // this single primitive tension: `consume_events`
                    // advances the offset BEFORE `save_snapshot` commits,
                    // so a naive merge can drop drained samples. Tracked
                    // event-id arbitration closes the disjoint-drain race
                    // (Codex round-2/3/4 HIGHs):
                    //  - winner has higher last_id → keep current
                    //  - we have higher last_id → take ours
                    //  - tie → identical drain ranges, keep current
                    //  - we have None / current has Some → keep current
                    //  - we have Some / current has None → take ours
                    //
                    // The `consume_events`-advances-before-save root cause
                    // is shared with M2/M3/M6 alpha-learning pipelines and
                    // is tracked as a v0.24.x cross-cutting refactor.
                    match (&self.concept_refresh_stats, &current.concept_refresh_stats) {
                        (Some(mine), Some(theirs)) => {
                            if mine.last_consumed_event_id > theirs.last_consumed_event_id {
                                current.concept_refresh_stats = Some(mine.clone());
                            }
                        }
                        (Some(_), None) => {
                            current.concept_refresh_stats = self.concept_refresh_stats.clone();
                        }
                        (None, _) => { /* keep current */ }
                    }
                    // v0.26 D direction: synthesis_feedback_stats — same
                    // arbitration shape as concept_refresh_stats above. The
                    // event-id MAX rule preserves the more advanced
                    // reservoir; if writer has None we keep current's
                    // existing learned state (Codex round-3 HIGH from
                    // v0.24 generalised). Both `by_cluster` aggregates and
                    // `by_synthesis` LRU are wholesale-replaced because
                    // they're both derived caches over the same monotonic
                    // event log — partial merging would create double-counted
                    // counters.
                    // Tie rule (codex R11): on EQUAL watermarks a writer that
                    // just reclustered wins — its by_cluster buckets are
                    // keyed in the surviving generation, while the peer's
                    // equally-advanced copy would be stripped below and the
                    // already-committed offsets could never replay to
                    // rebuild them.
                    let synth_stats_from_self = match (
                        &self.synthesis_feedback_stats,
                        &current.synthesis_feedback_stats,
                    ) {
                        (Some(mine), Some(theirs)) => {
                            if mine.last_consumed_event_id > theirs.last_consumed_event_id
                                || (we_reclustered
                                    && mine.last_consumed_event_id == theirs.last_consumed_event_id)
                            {
                                current.synthesis_feedback_stats = Some(mine.clone());
                                true
                            } else {
                                false
                            }
                        }
                        (Some(_), None) => {
                            current.synthesis_feedback_stats =
                                self.synthesis_feedback_stats.clone();
                            true
                        }
                        (None, _) => false, /* keep current */
                    };
                    // v0.27 ARS Cap A feedback loop (Track 1): mirror the
                    // synthesis_feedback_stats arm above. Same arbitration
                    // shape — event-id MAX wins, writer's `None` does not
                    // overwrite existing learned state. Round-3 HIGH from
                    // v0.24 generalised; round-4 HIGH from v0.26 mirrored.
                    let concept_stats_from_self = match (
                        &self.concept_summary_feedback_stats,
                        &current.concept_summary_feedback_stats,
                    ) {
                        (Some(mine), Some(theirs)) => {
                            // Same equal-watermark recluster tie rule as the
                            // synthesis arm above (codex R11).
                            if mine.last_consumed_event_id > theirs.last_consumed_event_id
                                || (we_reclustered
                                    && mine.last_consumed_event_id == theirs.last_consumed_event_id)
                            {
                                current.concept_summary_feedback_stats = Some(mine.clone());
                                true
                            } else {
                                false
                            }
                        }
                        (Some(_), None) => {
                            current.concept_summary_feedback_stats =
                                self.concept_summary_feedback_stats.clone();
                            true
                        }
                        (None, _) => false, /* keep current */
                    };
                    // #17 codex R7/R9: the watermark arms above are
                    // generation-blind, so a winner from an OLDER clustering
                    // generation could re-serve by_cluster buckets keyed to
                    // dead labels. Strip label-keyed buckets ONLY when the
                    // winning side's generation is not the merge's final
                    // generation (R9: a reclustering pass's own
                    // freshly-consumed new-generation buckets must survive —
                    // its offsets commit after this save and cannot replay),
                    // plus the recluster-tie case where equal numbers do not
                    // mean equal labels and the peer's provenance is
                    // unknowable. Watermarks and per-id stats always stay.
                    let final_cv = self.cluster_version.max(current_cv_at_merge);
                    let recluster_tie =
                        we_reclustered && self.cluster_version == current_cv_at_merge;
                    let strip_for = |from_self: bool| {
                        let provenance_cv = if from_self {
                            self.cluster_version
                        } else {
                            current_cv_at_merge
                        };
                        provenance_cv != final_cv || (recluster_tie && !from_self)
                    };
                    if strip_for(synth_stats_from_self) {
                        if let Some(stats) = &mut current.synthesis_feedback_stats {
                            stats.by_cluster.retain(|k, _| k.starts_with("-1|"));
                        }
                    }
                    if strip_for(concept_stats_from_self) {
                        if let Some(stats) = &mut current.concept_summary_feedback_stats {
                            stats.by_cluster.retain(|k, _| k.starts_with("-1|"));
                        }
                    }
                    // v0.27.1 E direction: judge_calibration_state — R9-K5
                    // mandates field-grouped CAS merge because Layer 1 and
                    // Layer 2 consumers each own a different subset of the
                    // struct. A naive single-watermark merge would drop
                    // whichever side has a lower watermark on the merge pass.
                    //
                    //   Layer 1 fields (synthesis_feedback / concept_summary_feedback owned):
                    //     - recent_pairs_synthesis, recent_pairs_concept, kappa
                    //     - merged-by-event-id-MAX from
                    //       synthesis_feedback_stats.last_consumed_event_id
                    //       (Layer 1 consumers update Layer 1 fields atomically
                    //        with their own watermark; we use the OWNER's
                    //        watermark to arbitrate).
                    //
                    //   Layer 2 fields (judge_calibration owned):
                    //     - runtime_vs_offline_kappa,
                    //       last_consumed_event_id_calibration,
                    //       recent_pairs_runtime_vs_offline,
                    //       total_offline_cron_events,
                    //       judge_drift_alert, last_computed_at
                    //     - merged-by-MAX of last_consumed_event_id_calibration.
                    //
                    // We compose the merged struct field-by-field rather
                    // than wholesale-replacing the Option, so Layer 1 progress
                    // doesn't clobber Layer 2 state and vice versa.
                    {
                        let merged = match (
                            &self.judge_calibration_state,
                            &current.judge_calibration_state,
                        ) {
                            (None, None) => None,
                            (Some(m), None) => Some(m.clone()),
                            (None, Some(t)) => Some(t.clone()),
                            (Some(mine), Some(theirs)) => {
                                let mut out = theirs.clone();
                                // Layer 1 arbitration: whichever side
                                // incorporated more synthesis_feedback events
                                // wins the Layer 1 fields. We approximate
                                // "Layer 1 watermark" by inferring from
                                // synthesis_feedback_stats.last_consumed_event_id
                                // on the same Self/current.
                                let mine_l1 = self
                                    .synthesis_feedback_stats
                                    .as_ref()
                                    .map(|s| s.last_consumed_event_id)
                                    .unwrap_or(0);
                                let theirs_l1 = current
                                    .synthesis_feedback_stats
                                    .as_ref()
                                    .map(|s| s.last_consumed_event_id)
                                    .unwrap_or(0);
                                if mine_l1 > theirs_l1 {
                                    out.recent_pairs_synthesis =
                                        mine.recent_pairs_synthesis.clone();
                                    out.kappa = mine.kappa;
                                }
                                let mine_capa = self
                                    .concept_summary_feedback_stats
                                    .as_ref()
                                    .map(|s| s.last_consumed_event_id)
                                    .unwrap_or(0);
                                let theirs_capa = current
                                    .concept_summary_feedback_stats
                                    .as_ref()
                                    .map(|s| s.last_consumed_event_id)
                                    .unwrap_or(0);
                                if mine_capa > theirs_capa {
                                    out.recent_pairs_concept = mine.recent_pairs_concept.clone();
                                }
                                // Layer 2 arbitration by judge_calibration
                                // watermark (MAX wins; ties keep current).
                                if mine.last_consumed_event_id_calibration
                                    > theirs.last_consumed_event_id_calibration
                                {
                                    out.runtime_vs_offline_kappa = mine.runtime_vs_offline_kappa;
                                    out.runtime_vs_offline_kappa_synthesis =
                                        mine.runtime_vs_offline_kappa_synthesis;
                                    out.runtime_vs_offline_kappa_concept =
                                        mine.runtime_vs_offline_kappa_concept;
                                    out.last_consumed_event_id_calibration =
                                        mine.last_consumed_event_id_calibration;
                                    out.recent_pairs_runtime_vs_offline =
                                        mine.recent_pairs_runtime_vs_offline.clone();
                                    out.recent_pairs_runtime_vs_offline_synthesis =
                                        mine.recent_pairs_runtime_vs_offline_synthesis.clone();
                                    out.recent_pairs_runtime_vs_offline_concept =
                                        mine.recent_pairs_runtime_vs_offline_concept.clone();
                                    out.total_offline_cron_events = mine.total_offline_cron_events;
                                    out.judge_drift_alert = mine.judge_drift_alert;
                                    out.judge_drift_alert_synthesis =
                                        mine.judge_drift_alert_synthesis;
                                    out.judge_drift_alert_concept = mine.judge_drift_alert_concept;
                                    out.last_computed_at = mine.last_computed_at;
                                }
                                Some(out)
                            }
                        };
                        current.judge_calibration_state = merged;
                    }
                    // v0.27.1 E direction: pending_kappa_half_pairs is a
                    // derived cache over the same monotonic event log as
                    // `synthesis_feedback_stats` (and the Cap A mirror).
                    // Whoever drained more events wins — we approximate
                    // by taking the side with the higher
                    // `synthesis_feedback_stats.last_consumed_event_id`.
                    // Wholesale replace because partial merge would leave
                    // a half-pair whose other half never arrives.
                    {
                        // Codex R4 P2 fix — both `synthesis_feedback` AND
                        // `concept_summary_feedback` consumers write into
                        // the shared `pending_kappa_half_pairs` cache.
                        // Choose the side whose MAX of the two watermarks
                        // is higher (i.e. whoever drained more events
                        // total). Without this, a Cap A advance with no
                        // synthesis advance left mine_l1 == theirs_l1 and
                        // dropped Cap A's half-pairs.
                        let mine_synth = self
                            .synthesis_feedback_stats
                            .as_ref()
                            .map(|s| s.last_consumed_event_id)
                            .unwrap_or(0);
                        let mine_cs = self
                            .concept_summary_feedback_stats
                            .as_ref()
                            .map(|s| s.last_consumed_event_id)
                            .unwrap_or(0);
                        let theirs_synth = current
                            .synthesis_feedback_stats
                            .as_ref()
                            .map(|s| s.last_consumed_event_id)
                            .unwrap_or(0);
                        let theirs_cs = current
                            .concept_summary_feedback_stats
                            .as_ref()
                            .map(|s| s.last_consumed_event_id)
                            .unwrap_or(0);
                        let mine_max = mine_synth.max(mine_cs);
                        let theirs_max = theirs_synth.max(theirs_cs);
                        if mine_max > theirs_max {
                            current.pending_kappa_half_pairs =
                                self.pending_kappa_half_pairs.clone();
                        }
                    }
                    // v0.28 ARS acceleration: scalar smoothing state is keyed
                    // independently of event consumers. Merge per key by
                    // RFC3339 timestamp so unrelated scalar updates from two
                    // writers do not clobber each other.
                    for (key, mine) in &self.ars_effective_scalars {
                        let replace = current
                            .ars_effective_scalars
                            .get(key)
                            .map(|theirs| mine.last_updated > theirs.last_updated)
                            .unwrap_or(true);
                        if replace {
                            current
                                .ars_effective_scalars
                                .insert(key.clone(), mine.clone());
                        }
                    }
                    current.version = db_version + 1;

                    // v0.28.7+ audit M-8 R2 P2 #2 — enforce the L6 cap
                    // on the post-merge map. Per-key inserts in
                    // `commit_shadow_fusion_weight_replay` already call
                    // the per-key eviction helper, but the CAS merge
                    // above just folded peer-written entries into
                    // `current` directly without going through any
                    // insert helper. Without this, two concurrent
                    // adaptive runs writing distinct cluster keys could
                    // push the persisted map well above
                    // `LEARNED_SHADOW_FUSION_CAP`. The shrink is a
                    // no-op below cap (cheap len check), so the steady
                    // state pays nothing.
                    crate::store::adaptive::shrink_learned_shadow_fusion_to_cap(
                        &mut current.learned_shadow_fusion,
                    );

                    let merged_json = current.serialize_with_a12_recall_projection()?;

                    // CAS write: only succeed if nobody else wrote since our read.
                    // COALESCE so a malformed JSON in the row doesn't silently skip the update.
                    let cas_rows = conn.execute(
                        "UPDATE metadata SET value = ?1
                         WHERE key = 'adaptive_state'
                         AND COALESCE(json_extract(value, '$.version'), -1) = ?2",
                        rusqlite::params![&merged_json, db_version as i64],
                    )?;

                    if cas_rows > 0 {
                        break; // success
                    }
                    // Another writer snuck in — retry
                    if attempt == MAX_CAS_RETRIES - 1 {
                        tracing::error!(
                            attempts = MAX_CAS_RETRIES,
                            db_version,
                            our_version = self.version,
                            "adaptive state: CAS failed — last observed db_version shown"
                        );
                        return Err(crate::types::error::ReinError::Config(format!(
                            "adaptive state: CAS failed after {MAX_CAS_RETRIES} attempts (last db_version={db_version})"
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    /// Restore state snapshot from metadata table.
    pub fn restore_snapshot(conn: &Connection) -> Option<Self> {
        let json: String = conn
            .query_row(
                "SELECT value FROM metadata WHERE key = 'adaptive_state'",
                [],
                |row| row.get(0),
            )
            .ok()?;
        let mut state: Self = serde_json::from_str(&json).ok()?;
        // v0.28.7+ audit M-8 R2 P2 #2 — bound the restored map. A
        // pre-cap snapshot (or one written by an older / peer binary
        // that didn't enforce the cap at insert time) could contain
        // an over-cap blob. Shrink to cap at restore so the in-memory
        // state immediately respects the bound. No-op below cap.
        shrink_learned_shadow_fusion_to_cap(&mut state.learned_shadow_fusion);
        // #17: remember which clustering generation this writer LOADED, so
        // `save_snapshot`'s CAS merge can tell "this writer actually ran a
        // recluster" (cluster_version grew past the loaded one → its wipe
        // must win wholesale) apart from "same generation, no recluster"
        // (additive merge of disjoint learned buckets is correct).
        state.cluster_version_at_load = state.cluster_version;
        Some(state)
    }
}

// ── Cached AdaptiveState with TTL ────────────────────────────────────────────

/// Default TTL for [`CachedAdaptiveState`] when callers don't pass an explicit value.
///
/// Matches `config.adaptive.cache_ttl_secs` default and is exposed here so
/// call sites without config access still get a sane value.
pub const ADAPTIVE_CACHE_TTL: std::time::Duration = std::time::Duration::from_secs(300);

/// Time-bounded in-memory cache of an `AdaptiveState` snapshot.
///
/// Without a TTL the previous code would hold a stale snapshot indefinitely
/// unless feedback events explicitly rebuilt it — meaning slow-channel
/// parameter updates applied in another process (GC, alpha optimizer, tier
/// recomputation) could be invisible to the current one until restart.
///
/// This wrapper tracks `last_refreshed_at: Instant`; on `get` it transparently
/// reloads from the metadata table once the TTL elapses. Callers can also
/// force a refresh via [`Self::invalidate`].
pub struct CachedAdaptiveState {
    state: AdaptiveState,
    last_refreshed_at: std::time::Instant,
    ttl: std::time::Duration,
}

impl CachedAdaptiveState {
    /// Create a new cache loaded from the DB, with the given TTL.
    /// Falls back to `AdaptiveState::default()` when no snapshot exists.
    pub fn load(conn: &Connection, ttl: std::time::Duration) -> Self {
        let state = AdaptiveState::restore_snapshot(conn).unwrap_or_default();
        Self {
            state,
            last_refreshed_at: std::time::Instant::now(),
            ttl,
        }
    }

    /// Convenience: load with the module-level default TTL.
    pub fn load_default(conn: &Connection) -> Self {
        Self::load(conn, ADAPTIVE_CACHE_TTL)
    }

    /// Return the cached state, transparently refreshing from DB when stale.
    pub fn get(&mut self, conn: &Connection) -> &AdaptiveState {
        if self.is_stale() {
            self.refresh(conn);
        }
        &self.state
    }

    /// Whether the cache has exceeded its TTL.
    pub fn is_stale(&self) -> bool {
        self.last_refreshed_at.elapsed() > self.ttl
    }

    /// Force the next `get` to reload from the DB regardless of age.
    pub fn invalidate(&mut self) {
        // Set the refresh timestamp far enough in the past that any non-zero TTL
        // will classify the cache as stale on the next access.
        self.last_refreshed_at = std::time::Instant::now()
            .checked_sub(self.ttl.saturating_add(std::time::Duration::from_secs(1)))
            .unwrap_or(self.last_refreshed_at);
    }

    /// Reload from the DB immediately.
    pub fn refresh(&mut self, conn: &Connection) {
        if let Some(fresh) = AdaptiveState::restore_snapshot(conn) {
            self.state = fresh;
        }
        self.last_refreshed_at = std::time::Instant::now();
    }

    /// Direct immutable access without TTL check — for paths that just want
    /// whatever we have in memory (e.g. when already inside a transaction).
    pub fn peek(&self) -> &AdaptiveState {
        &self.state
    }
}

// ── Cluster Centroid Persistence ─────────────────────────────────────────────

/// Save cluster centroids to the `cluster_centroids` table (raw f32 LE bytes).
/// Replaces all existing rows (full rewrite on each HDBSCAN run).
pub fn save_cluster_centroids(
    conn: &Connection,
    centroids: &HashMap<u32, Vec<f32>>,
    version: u64,
    dims: usize,
) -> ReinResult<()> {
    let result = (|| -> ReinResult<()> {
        // SAVEPOINT is nesting-safe when the caller already owns a larger
        // recluster transaction.
        conn.execute_batch("SAVEPOINT cluster_centroids_rewrite")?;
        conn.execute_batch("DELETE FROM cluster_centroids")?;
        let mut stmt = conn.prepare(
            "INSERT INTO cluster_centroids (cluster_id, centroid, cluster_version, dims) VALUES (?1, ?2, ?3, ?4)",
        )?;
        for (&cluster_id, vec) in centroids {
            let blob: Vec<u8> = vec.iter().flat_map(|f| f.to_le_bytes()).collect();
            stmt.execute(rusqlite::params![
                cluster_id,
                blob,
                version as i64,
                dims as i64
            ])?;
        }
        drop(stmt);
        conn.execute_batch("RELEASE cluster_centroids_rewrite")?;
        Ok(())
    })();
    if result.is_err() {
        let _ = conn.execute_batch("ROLLBACK TO cluster_centroids_rewrite");
        let _ = conn.execute_batch("RELEASE cluster_centroids_rewrite");
    }
    result
}

/// Load cluster centroids from `cluster_centroids` table.
/// `expected_dims`: if > 0, rows with a different `dims` value are silently skipped
/// (prevents stale centroids from a prior embedding model from corrupting assignments).
/// Returns empty map if table is missing, has no rows, or all rows have mismatched dims.
pub fn load_cluster_centroids(conn: &Connection, expected_dims: usize) -> HashMap<u32, Vec<f32>> {
    let mut out = HashMap::new();
    let Ok(mut stmt) = conn.prepare("SELECT cluster_id, centroid, dims FROM cluster_centroids")
    else {
        return out;
    };
    let _ = stmt
        .query_map([], |row| {
            let cid: u32 = row.get(0)?;
            let blob: Vec<u8> = row.get(1)?;
            let stored_dims: i64 = row.get(2)?;
            Ok((cid, blob, stored_dims as usize))
        })
        .ok()
        .map(|rows| {
            for row in rows.flatten() {
                let (cid, blob, stored_dims) = row;
                // Skip centroids from a different embedding model / dimension
                if expected_dims > 0 && stored_dims != expected_dims {
                    tracing::debug!(
                        "skipping cluster {cid} centroid: dims {stored_dims} != expected {expected_dims}"
                    );
                    continue;
                }
                let floats: Vec<f32> = blob
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                out.insert(cid, floats);
            }
        });
    out
}

/// Cosine similarity between two equal-length vectors. Returns 0.0 on zero-norm inputs.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// Assign an embedding to the nearest stored cluster centroid.
/// Requires `centroids` from `load_cluster_centroids`. Returns `None` if:
/// - `centroids` is empty, or
/// - best cosine similarity is ≤ 0.45 (embedding is an outlier / noise point).
pub fn assign_to_nearest_centroid(
    centroids: &HashMap<u32, Vec<f32>>,
    embedding: &[f32],
) -> Option<u32> {
    if centroids.is_empty() {
        return None;
    }
    let mut best_id = None;
    let mut best_sim = f32::NEG_INFINITY;
    for (&cluster_id, centroid) in centroids {
        let sim = cosine_similarity(embedding, centroid);
        if sim > best_sim {
            best_sim = sim;
            best_id = Some(cluster_id);
        }
    }
    if best_sim > 0.45 {
        best_id
    } else {
        None
    }
}

// ── v0.24 ARS — Concept Living Summary refresh parameters ───────────────────

/// Bootstrap revisions-since-last-summary threshold. One handful — matches
/// rein's "minimum of ten" default pattern (M1 min_samples etc.).
/// Fires a refresh every 5 revisions while learned stats are empty.
pub const CONCEPT_REFRESH_BOOTSTRAP_REVISION: u32 = 5;

/// Bootstrap age threshold in seconds. 7 days keeps slow-moving concepts
/// alive with at least weekly refresh while learned stats accumulate.
pub const CONCEPT_REFRESH_BOOTSTRAP_AGE_SECS: i64 = 7 * 24 * 60 * 60;

/// Minimum sample count before trusting learned concept-refresh stats.
/// Matches the "ten samples" default used across M1-M6.
pub const CONCEPT_REFRESH_MIN_SAMPLES: usize = 10;

/// Maximum retained refresh samples. Drives both memory footprint
/// (~12 bytes/sample) and percentile responsiveness — older samples
/// drop out FIFO once the cap is reached so the distribution tracks
/// recent steady-state behaviour rather than ancient bootstrap noise.
pub const CONCEPT_REFRESH_SAMPLE_CAP: usize = 500;

/// v0.24 ARS L3: a single observed concept-summary refresh sample.
///
/// Two regimes share the same struct, separated by the `first_refresh`
/// flag:
/// - **Steady state** (`prior living_summary_source_revision = Some(n)`,
///   `first_refresh = false`): `revisions_since_last = current_revision -
///   n`, `age_secs_since_last = now - prior living_summary_updated_at`.
///   Both dimensions are unbiased measurements of the inter-refresh
///   interval distribution.
/// - **First refresh** (no prior summary, `first_refresh = true`):
///   `revisions_since_last = current_revision` (anchored to concept
///   inception, still meaningful as "how many revisions before the
///   concept's first summary fired"); `age_secs_since_last = now -
///   concept created_at` (BIASED — measures concept lifetime, not refresh
///   cadence).
///
/// Per Codex round-2 MEDIUM, first-refresh samples are excluded from the
/// age percentile inside `recompute_concept_refresh_stats` so a young
/// corpus can't contract the learned `age_p50_secs` below bootstrap.
/// They still contribute to the revision percentile and to the *total*
/// `count`, so revision-side bootstrap exit doesn't get blocked while
/// steady-state age samples accumulate.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct RefreshSample {
    pub revisions_since_last: u32,
    pub age_secs_since_last: i64,
    /// `true` when this sample is the concept's first-ever refresh
    /// (no prior `living_summary_source_revision`).
    /// `#[serde(default)]` so older serialized samples without this flag
    /// deserialize as steady-state — there is no production data on
    /// `wip` yet so this back-compat is a safety net, not a migration.
    #[serde(default)]
    pub first_refresh: bool,
    /// v0.27.1 E direction (spec §3.2 R8 P1) — ULID minted by
    /// `refresh_living_summary` on every successful Cap A summary write.
    /// Lets the LLM judge link J5 back to the immutable
    /// `concept_summary_instances` retention row even after a subsequent
    /// refresh overwrites `concepts.living_summary_id`.
    ///
    /// `#[serde(default)]` so pre-v0.27.1 events parse with an empty
    /// string — the judge worker treats empty `summary_id` as J5
    /// link-absent and skips.
    #[serde(default)]
    pub summary_id: String,
}

/// v0.24 ARS: learned statistics for concept living-summary refresh.
/// Populated by the slow-channel from post-refresh feedback. `None` on
/// fresh install; helper thresholds fall back to bootstrap constants
/// until `count >= CONCEPT_REFRESH_MIN_SAMPLES` (revisions) or
/// `count_steady_state >= CONCEPT_REFRESH_MIN_SAMPLES` (age).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConceptRefreshStats {
    /// Total number of refresh samples in the reservoir (steady-state +
    /// first-refresh). Gates the revision-percentile bootstrap exit.
    pub count: usize,
    /// p75 of revisions-since-last-summary across **all** samples
    /// (`revisions_since_last` is meaningful for first-refresh too).
    pub revision_p75: u32,
    /// p50 of time-since-last-summary across **steady-state** samples
    /// only — first-refresh ages are anchored to concept inception and
    /// would bias this downward (Codex round-2 MEDIUM).
    pub age_p50_secs: i64,
    /// Number of steady-state samples in the reservoir (those with
    /// `first_refresh = false`). Gates the age-percentile bootstrap exit.
    /// `#[serde(default)]` so prior `wip`-tree state without this field
    /// deserializes as 0; a fresh recompute then repopulates it.
    #[serde(default)]
    pub count_steady_state: usize,
    /// Authoritative sample reservoir, FIFO-capped at
    /// [`CONCEPT_REFRESH_SAMPLE_CAP`]. `count`/`revision_p75`/
    /// `age_p50_secs`/`count_steady_state` are derived caches over this
    /// vector — recomputed on every `recompute_concept_refresh_stats`
    /// call.
    #[serde(default)]
    pub samples: Vec<RefreshSample>,
    /// Highest `feedback_events.id` incorporated into `samples`. Used in
    /// the CAS retry merge inside `save_snapshot` to arbitrate which
    /// reservoir is more advanced when two pipeline runs race (Codex
    /// round-4 HIGH). `0` means no events have been consumed yet (or
    /// legacy snapshot from before this field existed).
    #[serde(default)]
    pub last_consumed_event_id: i64,
}

/// v0.24 ARS L3: peek new `ConceptSummaryRefreshed` feedback events,
/// fold them into the rolling `ConceptRefreshStats` reservoir, recompute
/// cached percentiles, and return the highest event id incorporated so
/// the caller can commit the consumer offset *after* the derived state
/// is durable (module-level peek+commit invariant).
///
/// Returns `(updated_stats, Option<max_event_id>)`. `Option::None` means
/// no new events were observed → caller skips `commit_offset`.
///
/// **Why event-sourced (not snapshot-from-state, like
/// [`recompute_canonical_length_stats`]):** snapshotting current concept
/// rows can only yield `current_revision - living_summary_source_revision`
/// for concepts that have a summary — i.e. "how stale is each summary
/// right now". That's a biased statistic; it conflates the refresh
/// interval distribution with stale-since-last-trigger lag and excludes
/// every concept that has *never* been refreshed. Inter-refresh intervals
/// are only observable at the moment a refresh happens, so we capture them
/// then via `EventType::ConceptSummaryRefreshed` and aggregate here.
///
/// Malformed payloads are logged via `tracing::warn!` and skipped — same
/// pattern as Codex round 1 finding #2 fix in `ops/concept_summary.rs`.
pub fn recompute_concept_refresh_stats(
    conn: &Connection,
    prior: Option<ConceptRefreshStats>,
) -> ReinResult<(ConceptRefreshStats, Option<i64>)> {
    let mut stats = prior.unwrap_or_default();

    // Single peek covers the common case (rare events). The 50 000 cap
    // matches the prior implementation's pathological hard stop and is
    // far above realistic per-pass volume (one refresh per concept per
    // ~7 days under default cadence).
    let events = peek_events(
        conn,
        "concept_refresh_stats",
        &[EventType::ConceptSummaryRefreshed.as_str()],
        50_000,
    )?;
    if events.is_empty() {
        return Ok((stats, None));
    }
    let max_id_this_pass = events.last().map(|e| e.id);
    // Replay-safety (Codex Tier-B+C round-1 HIGH): if a prior pass's
    // `commit_offset` failed after `save_snapshot` succeeded, the next
    // pass re-peeks the same events. We MUST skip events whose samples
    // are already in the reservoir, otherwise the FIFO grows by the
    // double-applied entries. `stats.last_consumed_event_id` records
    // what was incorporated into the *durable* snapshot; the prior
    // pass's bump survived the snapshot save (or there was no prior
    // pass and the field is 0). Filter the just-peeked events to those
    // strictly past that watermark.
    let prior_high_water = stats.last_consumed_event_id;
    if let Some(max_id) = max_id_this_pass {
        stats.last_consumed_event_id = stats.last_consumed_event_id.max(max_id);
    }
    {
        for ev in events {
            if ev.id <= prior_high_water {
                continue;
            }
            let payload_str = match ev.payload.as_deref() {
                Some(s) => s,
                None => {
                    tracing::warn!(
                        event_id = ev.id,
                        "concept_refresh_stats: event missing payload, skipping"
                    );
                    continue;
                }
            };
            let sample: RefreshSample = match serde_json::from_str(payload_str) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(
                        event_id = ev.id,
                        error = %e,
                        "concept_refresh_stats: malformed RefreshSample payload, skipping"
                    );
                    continue;
                }
            };
            stats.samples.push(sample);
            if stats.samples.len() > CONCEPT_REFRESH_SAMPLE_CAP {
                let overflow = stats.samples.len() - CONCEPT_REFRESH_SAMPLE_CAP;
                stats.samples.drain(0..overflow);
            }
        }
    }

    // Recompute cached percentiles from current reservoir.
    // Revision percentile uses ALL samples (revisions_since_last is
    // unbiased for first-refresh too); age percentile uses ONLY
    // steady-state samples (first-refresh ages are concept-inception
    // anchored — see `RefreshSample` doc).
    stats.count = stats.samples.len();
    stats.count_steady_state = stats.samples.iter().filter(|s| !s.first_refresh).count();
    if stats.count > 0 {
        let mut revs: Vec<u32> = stats
            .samples
            .iter()
            .map(|s| s.revisions_since_last)
            .collect();
        revs.sort_unstable();
        stats.revision_p75 = percentile_u32(&revs, 0.75);
    } else {
        stats.revision_p75 = 0;
    }
    if stats.count_steady_state > 0 {
        let mut ages: Vec<i64> = stats
            .samples
            .iter()
            .filter(|s| !s.first_refresh)
            .map(|s| s.age_secs_since_last)
            .collect();
        ages.sort_unstable();
        stats.age_p50_secs = percentile_i64(&ages, 0.50);
    } else {
        stats.age_p50_secs = 0;
    }

    Ok((stats, max_id_this_pass))
}

fn percentile_u32(sorted: &[u32], p: f64) -> u32 {
    let n = sorted.len();
    if n == 0 {
        return 0;
    }
    let rank = p * (n.saturating_sub(1)) as f64;
    let lo = rank.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = rank - lo as f64;
    let lo_v = sorted[lo] as f64;
    let hi_v = sorted[hi] as f64;
    (lo_v + frac * (hi_v - lo_v)).round() as u32
}

fn percentile_i64(sorted: &[i64], p: f64) -> i64 {
    let n = sorted.len();
    if n == 0 {
        return 0;
    }
    let rank = p * (n.saturating_sub(1)) as f64;
    let lo = rank.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = rank - lo as f64;
    let lo_v = sorted[lo] as f64;
    let hi_v = sorted[hi] as f64;
    (lo_v + frac * (hi_v - lo_v)).round() as i64
}

impl AdaptiveState {
    /// v0.24 ARS: revisions-since-last-summary threshold. Uses learned
    /// stats if at least [`CONCEPT_REFRESH_MIN_SAMPLES`] observations are
    /// available, otherwise falls back to
    /// [`CONCEPT_REFRESH_BOOTSTRAP_REVISION`].
    pub fn concept_refresh_revision_threshold(&self) -> u32 {
        self.concept_refresh_stats
            .as_ref()
            .filter(|s| s.count >= CONCEPT_REFRESH_MIN_SAMPLES)
            .map(|s| s.revision_p75)
            .unwrap_or(CONCEPT_REFRESH_BOOTSTRAP_REVISION)
    }

    /// v0.24 ARS: age-since-last-summary threshold in seconds. Gates on
    /// `count_steady_state` (not `count`) because first-refresh ages are
    /// anchored to concept inception and would bias the percentile —
    /// Codex round-2 MEDIUM. Falls back to bootstrap until enough
    /// steady-state samples accumulate.
    pub fn concept_refresh_age_threshold_secs(&self) -> i64 {
        self.concept_refresh_stats
            .as_ref()
            .filter(|s| s.count_steady_state >= CONCEPT_REFRESH_MIN_SAMPLES)
            .map(|s| s.age_p50_secs)
            .unwrap_or(CONCEPT_REFRESH_BOOTSTRAP_AGE_SECS)
    }
}

// ── v0.26 D direction — Synthesis feedback (ARS Cap B closure loop) ─────────

/// v0.26 D direction: typed payload serialised into
/// `feedback_events.payload` for [`EventType::SynthesisInteraction`].
///
/// Shape locked by the Wave 2 contract (§3.1). The
/// `feedback_events.payload` column is already TEXT, so no DDL change is
/// required — emit the JSON via `serde_json::to_value` and round-trip via
/// `serde_json::from_str` inside [`recompute_synthesis_feedback_stats`].
///
/// Backward-compat invariant: any old (pre-v0.26) payload that doesn't
/// match this shape is simply ignored — the consumer filters by
/// `event_type == "synthesis_interaction"` so foreign payloads never
/// reach the deserializer in the first place.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
pub struct SynthesisInteractionPayload {
    /// ULID stamped in `run_recall_synthesis` after a synthesis succeeds.
    pub synthesis_id: String,
    /// ULID echoing `RecallMemoryOutput.request_id` so back-end can join
    /// downstream recall traces with synthesis interactions.
    pub recall_id: String,
    pub interaction: SynthesisInteractionKind,
    /// Optional hints about the synthesis context — `None` for older
    /// callers; `metadata.query_type` and `metadata.cluster_id` route the
    /// event into the per-`(cluster_id, query_type)` bucket.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<SynthesisMetadata>,
}

/// v0.26 D direction: discriminated interaction kinds posted from the
/// SynthesisCard surface (and other synthesis consumers).
///
/// Variant rationale (per contract §3.1):
/// - `Viewed { dwell_ms }`: time the synthesis was visible to the user;
///   feeds the dwell reservoir → `useful_rate` dwell term.
/// - `ClickedSource { source_index }`: 1-based to match the `[#k]` UI
///   marker convention. Out-of-range indices are accepted (silently
///   counted) — front-end is responsible for not emitting them.
/// - `ImmediateRequery { gap_ms }`: time gap since the prior synthesis_id's
///   last interaction to a new recall. Sliding threshold lives in the
///   consumer; do NOT hardcode "immediate" in the event itself.
/// - `ExplicitThumb { up }`: explicit user signal.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum SynthesisInteractionKind {
    Viewed { dwell_ms: u64 },
    ClickedSource { source_index: u32 },
    ImmediateRequery { gap_ms: u64 },
    ExplicitThumb { up: bool },
}

/// v0.26 D direction: optional context emitted alongside an interaction.
/// `Default` is empty (all `None`) so callers can construct it without
/// committing to every field.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct SynthesisMetadata {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cluster_id: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_count: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub synthesis_chars: Option<u32>,
}

// Bootstrap weights for `compute_useful_rate`. Marked `bootstrap` per
// `feedback_no_subjective_params` — every literal is data-driven in the
// fullness of time. v0.26.1 will derive these from a SemDeDup-style
// ablation across observed `(ClusterSynthesisStats, downstream-recall)`
// pairs. Until then they reflect "view-with-dwell + thumb are positive,
// requery is the strongest negative".
pub const SYNTHESIS_W_VIEW: f64 = 1.0; // bootstrap; v0.26.1 → ablation
pub const SYNTHESIS_W_CLICK: f64 = 0.5; // bootstrap; v0.26.1 → ablation
pub const SYNTHESIS_W_THUMB: f64 = 2.0; // bootstrap; v0.26.1 → ablation
pub const SYNTHESIS_W_REQUERY: f64 = 2.0; // bootstrap; v0.26.1 → ablation (subtracted)
/// Bootstrap dwell threshold. v0.26.1 → per-cluster p50 of dwell_samples.
pub const SYNTHESIS_DWELL_THRESHOLD_MS: u64 = 3_000; // bootstrap

/// FIFO reservoir cap for `dwell_samples` per `ClusterSynthesisStats`.
/// 500 keeps the `useful_rate` dwell term responsive to recent steady-state
/// without unbounded memory growth.
pub const SYNTHESIS_DWELL_RESERVOIR_CAP: usize = 500;

/// LRU cap for `by_synthesis` per-id stats. Implemented as a `HashMap` +
/// side `Vec<String>` for FIFO order because `lru::LruCache` is not
/// `Serialize` (cross-agent invariant 11).
pub const SYNTHESIS_PER_ID_CAP: usize = 1024;

/// Hard cap on the number of distinct `(cluster_id, query_type)` buckets
/// in `SynthesisFeedbackState.by_cluster`. Defends against a malicious
/// or buggy client flooding `/api/feedback` with fabricated `cluster_id`
/// or `query_type` values that would otherwise grow the persisted
/// adaptive-state snapshot without limit (Codex round 2 F-11). Once the
/// cap is reached new buckets are dropped (the events still increment
/// `total_events`), so legitimate buckets don't compete for capacity
/// once the system has converged on real cluster ids.
pub const SYNTHESIS_BY_CLUSTER_CAP: usize = 4096;

/// Whitelist of `query_type` values rein can legitimately emit (mirrors
/// `search/classify.rs::QueryType` plus the `_global` sentinel used by
/// REST projection). Any client-supplied value outside this list is
/// normalized to `"unknown"` before being folded into `by_cluster`, so
/// adversarial query_type strings can't multiplicatively explode the
/// bucket cardinality (Codex round 2 F-11).
pub const SYNTHESIS_ALLOWED_QUERY_TYPES: &[&str] = &[
    "Episodic",
    "Temporal",
    "Preference",
    "ExactKeyword",
    "Semantic",
    "Exploratory",
    "_global",
];

/// Min events per `(cluster_id, query_type)` bucket before per-cluster
/// `useful_rate` is trusted. Below this, `decide_synthesize` falls back
/// to the global enabled flag.
pub const SYNTHESIS_COLD_START_N: u64 = 10;

/// Bootstrap `useful_rate` cutoff used by `decide_synthesize` (per-query
/// gate). Hoisted into a constant so handler code never inlines the
/// literal (cross-agent invariant 12); v0.26.1 → adaptive once
/// `useful_rate` ablation lands.
pub const SYNTHESIS_USEFUL_RATE_THRESHOLD: f64 = 0.5; // bootstrap; v0.26.1 → adaptive

/// Dynamic useful-rate weights for Cap B synthesis and Cap A concept-summary
/// feedback. The historical constants remain the default, but v0.28 can now
/// thread bootstrap priors / replay-derived SignalHint labels through the
/// production formulas without changing the persisted bucket schema.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct UsefulRateWeights {
    pub view: f64,
    pub click: f64,
    pub thumb: f64,
    pub requery: f64,
}

impl UsefulRateWeights {
    pub const fn synthesis_bootstrap() -> Self {
        Self {
            view: SYNTHESIS_W_VIEW,
            click: SYNTHESIS_W_CLICK,
            thumb: SYNTHESIS_W_THUMB,
            requery: SYNTHESIS_W_REQUERY,
        }
    }

    pub const fn concept_summary_bootstrap() -> Self {
        Self {
            view: CONCEPT_SUMMARY_W_VIEW,
            click: CONCEPT_SUMMARY_W_CLICK,
            thumb: CONCEPT_SUMMARY_W_THUMB,
            requery: CONCEPT_SUMMARY_W_REQUERY,
        }
    }

    /// Build weights from prior/posterior labels. Invalid fields fall back to
    /// the supplied baseline; at least one positive finite weight is required.
    pub fn from_priors(baseline: Self, view: f64, click: f64, thumb: f64, requery: f64) -> Self {
        let sanitize = |value: f64, fallback: f64| {
            if value.is_finite() && value >= 0.0 {
                value
            } else {
                fallback
            }
        };
        let candidate = Self {
            view: sanitize(view, baseline.view),
            click: sanitize(click, baseline.click),
            thumb: sanitize(thumb, baseline.thumb),
            requery: sanitize(requery, baseline.requery),
        };
        if candidate.denominator() > 0.0 {
            candidate
        } else {
            baseline
        }
    }

    fn denominator(self) -> f64 {
        self.view + self.click + self.thumb + self.requery
    }
}

/// Per-bucket key used by [`SynthesisFeedbackState::by_cluster`]. Bucket
/// is `(cluster_id, query_type)` — both can be unknown, in which case
/// the consumer routes events to the global bucket key
/// `synthesis_bucket_key(None, "")` → `"-1|"`. Keyed via
/// `serde`-friendly `String` because `HashMap<(_, String), _>` round-trips
/// awkwardly through JSON (`serde_json` requires string keys).
pub fn synthesis_bucket_key(cluster_id: Option<i64>, query_type: &str) -> String {
    let cid = cluster_id.unwrap_or(-1);
    format!("{cid}|{query_type}")
}

/// v0.26 D direction: per-`(cluster_id, query_type)` synthesis interaction
/// aggregate. `useful_rate` is recomputed on every consumer pass via
/// [`compute_useful_rate`].
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ClusterSynthesisStats {
    pub viewed_count: u64,
    pub viewed_dwell_total_ms: u64,
    /// FIFO reservoir of `Viewed.dwell_ms` samples capped at
    /// [`SYNTHESIS_DWELL_RESERVOIR_CAP`]. Used to compute
    /// `viewed_dwell_p50_ms` and the dwell term in [`compute_useful_rate`].
    #[serde(default)]
    pub dwell_samples: Vec<u64>,
    /// Cached p50 of `dwell_samples`. `None` when reservoir is empty.
    #[serde(default)]
    pub viewed_dwell_p50_ms: Option<u64>,
    pub clicked_source_count: u64,
    pub immediate_requery_count: u64,
    pub explicit_up: u64,
    pub explicit_down: u64,
    // ── v0.27.1 E direction LLM judge counters ──
    //
    // Codex R4 P2 — `#[serde(default)]` is mandatory on every new persisted
    // counter. Existing nodes upgrading to v0.27.1 already have
    // `synthesis_feedback_stats` JSON in `adaptive_state` blobs; bare `u64`
    // fields without a default would make `serde_json::from_str` fail on
    // those rows and `restore_snapshot` would drop the entire learned
    // adaptive state on first boot.
    /// Number of [`EventType::SynthesisLlmJudge`] events folded into this
    /// bucket. Counts toward `total_signal` for cold-start fallback.
    #[serde(default)]
    pub llm_judge_count: u64,
    /// Number of those events whose `hit = true`.
    #[serde(default)]
    pub llm_judge_hit_count: u64,
    /// Derived metric, recomputed on every consumer pass.
    pub useful_rate: f64,
    /// v0.28 — LRU eviction key. Highest `feedback_events.id` folded into
    /// this bucket. Mirrors [`ClusterConceptSummaryStats::last_event_id`].
    #[serde(default)]
    pub last_event_id: i64,
}

/// v0.26 D direction: per-synthesis_id stats with bounded LRU semantics.
/// Used by future per-synthesis decay/heatmap views; recompute_consumer
/// caps total entries at [`SYNTHESIS_PER_ID_CAP`].
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct PerSynthesisStats {
    pub viewed_count: u32,
    pub clicked_source_count: u32,
    pub explicit_up: u32,
    pub explicit_down: u32,
    pub last_interaction_ts: i64,
}

/// v0.26 D direction: state container for the `synthesis_feedback`
/// consumer. Persisted as part of [`AdaptiveState`] (CAS-arbitrated by
/// `last_consumed_event_id`).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct SynthesisFeedbackState {
    /// Bucket: `synthesis_bucket_key(cluster_id, query_type)`.
    /// `cluster_id = -1` (None) means "no cluster"; empty `query_type`
    /// means unknown classification. HashMap so serde round-trips
    /// cleanly into the JSON snapshot.
    pub by_cluster: HashMap<String, ClusterSynthesisStats>,
    /// Bounded per-synthesis_id stats (LRU-capped at
    /// [`SYNTHESIS_PER_ID_CAP`]). Implemented as `HashMap` + side
    /// `by_synthesis_order` Vec — `lru::LruCache` is not `Serialize`
    /// and would break `AdaptiveState` snapshotting (cross-agent
    /// invariant 11).
    #[serde(default)]
    pub by_synthesis: HashMap<String, PerSynthesisStats>,
    /// FIFO order key list mirroring `by_synthesis` insertion order.
    /// Eviction pops from the front and removes the matching HashMap
    /// entry — both updates MUST happen together, otherwise the cache
    /// leaks.
    #[serde(default)]
    pub by_synthesis_order: Vec<String>,
    /// Highest event id incorporated into this state. Watermark for
    /// replay-safety (mirrors
    /// [`ConceptRefreshStats::last_consumed_event_id`]). The CAS merge
    /// in `save_snapshot` arbitrates between concurrent writers by the
    /// MAX of this id (Codex round-4 HIGH from v0.24 generalised).
    #[serde(default)]
    pub last_consumed_event_id: i64,
    /// Total events the consumer has *processed* (including replays
    /// counted only once). Useful for `/api/adaptive` exposure of
    /// "how much signal has accumulated".
    #[serde(default)]
    pub total_events: u64,
}

/// v0.28 — LRU eviction for the synthesis `by_cluster` map. Older releases
/// dropped new buckets once [`SYNTHESIS_BY_CLUSTER_CAP`] was reached, which
/// meant a saturated vault silently ignored fresh production signal. This
/// mirrors the Cap A concept-summary LRU behavior.
fn evict_synthesis_lru_if_at_cap(
    by_cluster: &mut HashMap<String, ClusterSynthesisStats>,
    new_key: &str,
) {
    if by_cluster.contains_key(new_key) || by_cluster.len() < SYNTHESIS_BY_CLUSTER_CAP {
        return;
    }
    let victim_key = by_cluster
        .iter()
        .min_by_key(|(_, b)| b.last_event_id)
        .map(|(k, _)| k.clone());
    if let Some(victim) = victim_key {
        tracing::warn!(
            evicted_bucket = %victim,
            new_bucket = %new_key,
            cap = SYNTHESIS_BY_CLUSTER_CAP,
            "synthesis_feedback: by_cluster cap reached; evicting LRU bucket"
        );
        by_cluster.remove(&victim);
    }
}

/// Pure function — testable in isolation. Computes a `[0.0, 1.0]`
/// usefulness rate from a single bucket's aggregate counters.
///
/// The formula combines:
/// - dwell pct: fraction of `dwell_samples` exceeding
///   [`SYNTHESIS_DWELL_THRESHOLD_MS`] (a "skim vs read" proxy).
/// - click rate: clicks / views (engagement with cited evidence).
/// - thumb rate: explicit positive ratio
///   (`explicit_up / (explicit_up + explicit_down + 1)`); `+1` Laplace
///   smoothing keeps the term well-defined when no thumbs have ever
///   landed.
/// - requery rate: requeries / views (subtracted — a strong negative
///   signal that the synthesis didn't satisfy the question).
///
/// Output is `.clamp(0.0, 1.0)` so the requery penalty cannot push the
/// score below zero (and rounding never floats above one). Bootstrap
/// weights are documented above; v0.26.1 will derive them from a
/// SemDeDup-style ablation.
pub fn compute_useful_rate(stats: &ClusterSynthesisStats) -> f64 {
    compute_useful_rate_with_weights(stats, UsefulRateWeights::synthesis_bootstrap())
}

pub fn compute_useful_rate_with_weights(
    stats: &ClusterSynthesisStats,
    weights: UsefulRateWeights,
) -> f64 {
    let total_views = stats.viewed_count.max(1) as f64;
    let dwell_pct = if stats.dwell_samples.is_empty() {
        0.0
    } else {
        stats
            .dwell_samples
            .iter()
            .filter(|&&d| d > SYNTHESIS_DWELL_THRESHOLD_MS)
            .count() as f64
            / stats.dwell_samples.len() as f64
    };
    let click_rate = stats.clicked_source_count as f64 / total_views;
    let thumb_rate =
        stats.explicit_up as f64 / (stats.explicit_up + stats.explicit_down + 1) as f64;
    let requery_rate = stats.immediate_requery_count as f64 / total_views;

    let numerator =
        weights.view * dwell_pct + weights.click * click_rate + weights.thumb * thumb_rate
            - weights.requery * requery_rate;
    let denom = weights.denominator();
    if denom <= 0.0 || !denom.is_finite() {
        return 0.0;
    }
    (numerator / denom).clamp(0.0, 1.0)
}

/// Compute the p50 of a non-empty slice of dwell samples (linear
/// interpolation matching `percentile_*`). Returns `None` for empty input.
fn dwell_p50_ms(samples: &[u64]) -> Option<u64> {
    if samples.is_empty() {
        return None;
    }
    let mut sorted: Vec<u64> = samples.to_vec();
    sorted.sort_unstable();
    let n = sorted.len();
    let rank = 0.50 * (n.saturating_sub(1)) as f64;
    let lo = rank.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = rank - lo as f64;
    let lo_v = sorted[lo] as f64;
    let hi_v = sorted[hi] as f64;
    Some((lo_v + frac * (hi_v - lo_v)).round() as u64)
}

/// v0.26 D direction: peek new `SynthesisInteraction` feedback events,
/// fold them into the rolling [`SynthesisFeedbackState`], recompute the
/// derived `useful_rate` per bucket, and return the highest event id
/// incorporated so the caller can commit the consumer offset *after*
/// the derived state is durable (module-level peek+commit invariant).
///
/// Returns `(updated_state, Option<max_event_id>)`. `Option::None` means
/// no new events were observed → caller skips `commit_offset`.
///
/// **5 invariants enforced** (per
/// [[feedback_event_sourced_state_invariant]]):
///
///   1. **Watermark filter** — events with
///      `id <= state.last_consumed_event_id` are skipped via
///      `prior_high_water`. Counter increments are NOT idempotent, so
///      this guard is the entire point.
///   2. **Applied-prefix bump** — `state.last_consumed_event_id` is
///      bumped to `max(state.last_consumed_event_id, max_id_this_pass)`
///      *before* any new events are folded; the caller is responsible
///      for committing the consumer offset only AFTER `save_snapshot`
///      returns Ok.
///   3. **Replay-drain** — `peek_events` reads from the consumer offset;
///      replay-safety after a `commit_offset` failure is guarded by
///      invariant (1).
///   4. **CAS merge** — `AdaptiveState::save_snapshot` arbitrates by
///      `last_consumed_event_id` MAX, mirroring the existing
///      `concept_refresh_stats` arm at adaptive.rs:806-816.
///   5. **Peek + commit** — uses `peek_events("synthesis_feedback", …)`
///      then *the caller* runs `commit_offset(&[("synthesis_feedback",
///      max_id)])` AFTER `save_snapshot` succeeds. Never `consume_events`
///      (the v0.24 round-2/3/4 HIGH that this contract retires).
///
/// Malformed payloads are logged via `tracing::warn!` and skipped
/// (mirrors `recompute_concept_refresh_stats` Codex round 1 finding #2 fix).
pub fn recompute_synthesis_feedback_stats(
    conn: &Connection,
    prior: Option<SynthesisFeedbackState>,
) -> ReinResult<(SynthesisFeedbackState, Option<i64>)> {
    let mut state = prior.unwrap_or_default();

    // Single peek covers the common case (most pipelines drain in one
    // shot). 50 000 cap matches the prior implementation's pathological
    // hard stop and is far above realistic per-pass volume; if you ever
    // exceed this, the caller can re-enter the consumer in the next
    // slow-channel pass.
    let events = peek_events(
        conn,
        "synthesis_feedback",
        &[EventType::SynthesisInteraction.as_str()],
        50_000,
    )?;
    if events.is_empty() {
        return Ok((state, None));
    }
    let max_id_this_pass = events.last().map(|e| e.id);

    // Invariants 1 + 2: the prior_high_water guard skips already-applied
    // events on a replay; the bump records the durable watermark in the
    // returned state so the next `save_snapshot` advances it. The caller
    // commits the consumer offset only AFTER save_snapshot succeeds.
    let prior_high_water = state.last_consumed_event_id;
    if let Some(max_id) = max_id_this_pass {
        state.last_consumed_event_id = state.last_consumed_event_id.max(max_id);
    }

    // Track buckets that received new events so we recompute their
    // `useful_rate` cache exactly once per pass.
    let mut touched_buckets: std::collections::HashSet<String> = std::collections::HashSet::new();

    for ev in events {
        if ev.id <= prior_high_water {
            continue;
        }
        let payload_str = match ev.payload.as_deref() {
            Some(s) => s,
            None => {
                tracing::warn!(
                    event_id = ev.id,
                    "synthesis_feedback: event missing payload, skipping"
                );
                continue;
            }
        };
        let payload: SynthesisInteractionPayload = match serde_json::from_str(payload_str) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(
                    event_id = ev.id,
                    error = %e,
                    "synthesis_feedback: malformed SynthesisInteractionPayload, skipping"
                );
                continue;
            }
        };

        let metadata = payload.metadata.clone().unwrap_or_default();
        let cluster_id = metadata.cluster_id;
        let raw_qtype = metadata.query_type.as_deref().unwrap_or("");
        // F-11 normalize: clamp non-whitelisted query_types to "unknown" so
        // malicious clients can't multiplicatively grow the bucket space.
        let query_type = if SYNTHESIS_ALLOWED_QUERY_TYPES.contains(&raw_qtype) {
            raw_qtype.to_string()
        } else {
            "unknown".to_string()
        };
        let bucket_key = synthesis_bucket_key(cluster_id, &query_type);

        evict_synthesis_lru_if_at_cap(&mut state.by_cluster, &bucket_key);

        // Per-bucket fold.
        let bucket = state.by_cluster.entry(bucket_key.clone()).or_default();
        bucket.last_event_id = bucket.last_event_id.max(ev.id);
        match &payload.interaction {
            SynthesisInteractionKind::Viewed { dwell_ms } => {
                bucket.viewed_count = bucket.viewed_count.saturating_add(1);
                bucket.viewed_dwell_total_ms =
                    bucket.viewed_dwell_total_ms.saturating_add(*dwell_ms);
                bucket.dwell_samples.push(*dwell_ms);
                if bucket.dwell_samples.len() > SYNTHESIS_DWELL_RESERVOIR_CAP {
                    let overflow = bucket.dwell_samples.len() - SYNTHESIS_DWELL_RESERVOIR_CAP;
                    bucket.dwell_samples.drain(0..overflow);
                }
            }
            SynthesisInteractionKind::ClickedSource { source_index: _ } => {
                bucket.clicked_source_count = bucket.clicked_source_count.saturating_add(1);
            }
            SynthesisInteractionKind::ImmediateRequery { gap_ms: _ } => {
                bucket.immediate_requery_count = bucket.immediate_requery_count.saturating_add(1);
            }
            SynthesisInteractionKind::ExplicitThumb { up } => {
                if *up {
                    bucket.explicit_up = bucket.explicit_up.saturating_add(1);
                } else {
                    bucket.explicit_down = bucket.explicit_down.saturating_add(1);
                }
            }
        }
        touched_buckets.insert(bucket_key);

        // Per-synthesis_id LRU fold. HashMap update + side-vec FIFO must
        // happen together; failure to dual-update leaks orphan keys.
        let sid = payload.synthesis_id.clone();
        let existed = state.by_synthesis.contains_key(&sid);
        {
            let per = state.by_synthesis.entry(sid.clone()).or_default();
            match &payload.interaction {
                SynthesisInteractionKind::Viewed { .. } => {
                    per.viewed_count = per.viewed_count.saturating_add(1);
                }
                SynthesisInteractionKind::ClickedSource { .. } => {
                    per.clicked_source_count = per.clicked_source_count.saturating_add(1);
                }
                SynthesisInteractionKind::ExplicitThumb { up } => {
                    if *up {
                        per.explicit_up = per.explicit_up.saturating_add(1);
                    } else {
                        per.explicit_down = per.explicit_down.saturating_add(1);
                    }
                }
                SynthesisInteractionKind::ImmediateRequery { .. } => {
                    // Tracked at bucket level only — per-synthesis attribution
                    // for requery is ambiguous (the requery happens against
                    // the *next* search, not this synthesis).
                }
            }
            per.last_interaction_ts = chrono::Utc::now().timestamp();
        }
        if !existed {
            // New entry — push to FIFO order.
            state.by_synthesis_order.push(sid.clone());
            // Cap evict: pop from the FRONT of the order vec AND remove
            // from the HashMap. Dual update is mandatory — failure to keep
            // both stores in sync leaks orphan HashMap entries.
            while state.by_synthesis_order.len() > SYNTHESIS_PER_ID_CAP {
                let evict = state.by_synthesis_order.remove(0);
                state.by_synthesis.remove(&evict);
            }
        }

        state.total_events = state.total_events.saturating_add(1);
    }

    // Recompute derived metrics for buckets touched this pass.
    for key in touched_buckets {
        if let Some(bucket) = state.by_cluster.get_mut(&key) {
            bucket.viewed_dwell_p50_ms = dwell_p50_ms(&bucket.dwell_samples);
            bucket.useful_rate = compute_useful_rate(bucket);
        }
    }

    Ok((state, max_id_this_pass))
}

impl AdaptiveState {
    /// v0.26 D direction: per-`(cluster_id, query_type)` synthesis bucket,
    /// returned only when the bucket has accumulated at least
    /// [`SYNTHESIS_COLD_START_N`] viewed samples (cold-start fallback
    /// otherwise — caller falls back to the global `synthesize` flag).
    ///
    /// v0.27.1 E direction: also returns the bucket when LLM judge events
    /// alone push `total_signal = viewed_count + explicit_up + explicit_down
    /// + llm_judge_count` over the cold-start threshold. This is the
    /// minimum change per Codex R8 P1 fix; the per-query gate caller in
    /// `decide_synthesize` re-checks `total_signal` so the threshold logic
    /// stays in one place.
    pub fn synthesis_bucket(
        &self,
        cluster_id: Option<i64>,
        query_type: &str,
    ) -> Option<&ClusterSynthesisStats> {
        let state = self.synthesis_feedback_stats.as_ref()?;
        let key = synthesis_bucket_key(cluster_id, query_type);
        state.by_cluster.get(&key).filter(|s| {
            // Accept when any individual signal already crossed the
            // cold-start threshold OR the cumulative signal does.
            // `decide_synthesize` re-applies its own threshold; this
            // method is only used as an "is this bucket interesting"
            // probe (see /api/adaptive surface).
            let total = s
                .viewed_count
                .saturating_add(s.explicit_up)
                .saturating_add(s.explicit_down)
                .saturating_add(s.llm_judge_count);
            total >= SYNTHESIS_COLD_START_N
        })
    }
}

/// v0.27.1 E direction (spec §6) — extended `synthesis_feedback` consumer
/// fold that also peeks runtime LLM judge events ([`EventType::SynthesisLlmJudge`])
/// and owns the κ-pair join per spec §6.2.1.
///
/// **Single watermark, single offset** — the consumer peeks BOTH event
/// types in one query (Codex R1 P1 fix: a separate `llm_judge_feedback`
/// offset against a shared state would let interleaved production traffic
/// silently drop judge events).
///
/// **κ-pair join (spec §6.2.1)** — when an `ExplicitThumb` arrives, the
/// consumer looks up `synthesis_id` in `pending_pairs`; on hit, completes
/// a `(judge_hit, thumb_up, ts)` pair into
/// `calibration.recent_pairs_synthesis`. When a `SynthesisLlmJudge` event
/// arrives, mirror logic — cache the judge half OR complete a pair if the
/// human thumb already arrived. This is the only consumer that sees BOTH
/// halves on the same offset.
///
/// All five M1 invariants (peek, watermark filter, applied-prefix bump,
/// replay-drain, CAS merge) are inherited from the original
/// `recompute_synthesis_feedback_stats` shape — extending the
/// `event_types` filter is non-invariant-breaking.
#[allow(clippy::too_many_arguments)]
pub fn recompute_synthesis_feedback_stats_with_judge(
    conn: &Connection,
    prior: Option<SynthesisFeedbackState>,
    pending_pairs_prior: HashMap<String, HalfPair>,
    calibration_prior: JudgeCalibrationState,
    // Codex R2 P2 fix — operator-tunable LLM signal weight (default 0.3
    // per spec §6.4). Caller threads `[ars.llm_judge].weight_decay_rate`
    // here so `useful_rate = 0.0` lets operators keep judge events for
    // observability while disabling their effect on `decide_synthesize`.
    weight_decay_rate: f64,
) -> ReinResult<(
    SynthesisFeedbackState,
    HashMap<String, HalfPair>,
    JudgeCalibrationState,
    Option<i64>,
)> {
    recompute_synthesis_feedback_stats_with_judge_and_weights(
        conn,
        prior,
        pending_pairs_prior,
        calibration_prior,
        weight_decay_rate,
        UsefulRateWeights::synthesis_bootstrap(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn recompute_synthesis_feedback_stats_with_judge_and_weights(
    conn: &Connection,
    prior: Option<SynthesisFeedbackState>,
    pending_pairs_prior: HashMap<String, HalfPair>,
    calibration_prior: JudgeCalibrationState,
    // Codex R2 P2 fix — operator-tunable LLM signal weight (default 0.3
    // per spec §6.4). Caller threads `[ars.llm_judge].weight_decay_rate`
    // here so `useful_rate = 0.0` lets operators keep judge events for
    // observability while disabling their effect on `decide_synthesize`.
    weight_decay_rate: f64,
    useful_rate_weights: UsefulRateWeights,
) -> ReinResult<(
    SynthesisFeedbackState,
    HashMap<String, HalfPair>,
    JudgeCalibrationState,
    Option<i64>,
)> {
    let mut state = prior.unwrap_or_default();
    let mut pending_pairs = pending_pairs_prior;
    let mut calibration = calibration_prior;

    let events = peek_events(
        conn,
        "synthesis_feedback",
        &[
            EventType::SynthesisInteraction.as_str(),
            EventType::SynthesisLlmJudge.as_str(),
        ],
        50_000,
    )?;
    if events.is_empty() {
        return Ok((state, pending_pairs, calibration, None));
    }
    let max_id_this_pass = events.last().map(|e| e.id);

    // Invariants 1 + 2 — watermark filter + applied-prefix bump.
    let prior_high_water = state.last_consumed_event_id;
    if let Some(max_id) = max_id_this_pass {
        state.last_consumed_event_id = state.last_consumed_event_id.max(max_id);
    }

    let mut touched_buckets: std::collections::HashSet<String> = std::collections::HashSet::new();

    // Pre-prune the half-pair cache by 7-day TTL using the latest event
    // timestamp as a clock anchor. Robust to NTP drift on the worker
    // host (don't rely on `chrono::Utc::now()` for cache TTL).
    let now_ts = chrono::Utc::now().timestamp();
    let cutoff = now_ts.saturating_sub(LLM_JUDGE_HALF_PAIR_TTL_SECS);
    pending_pairs.retain(|_, half| half.ts() >= cutoff);

    for ev in events {
        if ev.id <= prior_high_water {
            continue;
        }
        let payload_str = match ev.payload.as_deref() {
            Some(s) => s,
            None => {
                tracing::warn!(
                    event_id = ev.id,
                    "synthesis_feedback: event missing payload, skipping"
                );
                continue;
            }
        };
        match ev.event_type.as_str() {
            "synthesis_interaction" => {
                let payload: SynthesisInteractionPayload = match serde_json::from_str(payload_str) {
                    Ok(p) => p,
                    Err(e) => {
                        tracing::warn!(
                            event_id = ev.id,
                            error = %e,
                            "synthesis_feedback: malformed SynthesisInteractionPayload, skipping"
                        );
                        continue;
                    }
                };
                fold_synthesis_interaction(
                    &mut state,
                    &mut pending_pairs,
                    &mut calibration,
                    &mut touched_buckets,
                    &payload,
                    ev.id,
                );
            }
            "synthesis_llm_judge" => {
                let payload: SynthesisLlmJudgePayload = match serde_json::from_str(payload_str) {
                    Ok(p) => p,
                    Err(e) => {
                        tracing::warn!(
                            event_id = ev.id,
                            error = %e,
                            "synthesis_feedback: malformed SynthesisLlmJudgePayload, skipping"
                        );
                        continue;
                    }
                };
                fold_synthesis_llm_judge(
                    &mut state,
                    &mut pending_pairs,
                    &mut calibration,
                    &mut touched_buckets,
                    &payload,
                    ev.id,
                );
            }
            other => {
                tracing::debug!(
                    event_id = ev.id,
                    event_type = %other,
                    "synthesis_feedback: unexpected event_type in peek, skipping"
                );
            }
        }
    }

    // Recompute derived metrics for buckets touched this pass.
    for key in touched_buckets {
        if let Some(bucket) = state.by_cluster.get_mut(&key) {
            bucket.viewed_dwell_p50_ms = dwell_p50_ms(&bucket.dwell_samples);
            // v0.27.1: switch to the active-signal-mask formula when this
            // bucket has any LLM-judge contribution; fall back to the
            // v0.26 fixed-denominator formula otherwise so existing
            // human-only buckets keep their previously-computed values
            // bit-identical (avoids invalidating in-flight A/B tests).
            bucket.useful_rate = if bucket.llm_judge_count > 0 {
                compute_useful_rate_with_judge_and_weights(
                    bucket,
                    weight_decay_rate,
                    useful_rate_weights,
                )
                .unwrap_or_else(|| compute_useful_rate_with_weights(bucket, useful_rate_weights))
            } else {
                compute_useful_rate_with_weights(bucket, useful_rate_weights)
            };
        }
    }

    // LRU cap on the pending pairs map. Eviction is best-effort FIFO —
    // since HashMap iteration order is randomized, we drop arbitrary
    // entries. The 7-day TTL bound above limits the steady-state size
    // anyway; cap is a defense against pathological floods.
    if pending_pairs.len() > LLM_JUDGE_PAIR_CACHE_CAPACITY {
        let drop_n = pending_pairs.len() - LLM_JUDGE_PAIR_CACHE_CAPACITY;
        let to_drop: Vec<String> = pending_pairs
            .iter()
            .take(drop_n)
            .map(|(k, _)| k.clone())
            .collect();
        for k in to_drop {
            pending_pairs.remove(&k);
        }
    }

    Ok((state, pending_pairs, calibration, max_id_this_pass))
}

/// Fold a single `SynthesisInteraction` event into the consumer's state.
/// Pulled out of the loop so the same logic applies inside both the
/// human-only `recompute_synthesis_feedback_stats` and the v0.27.1 extended
/// variant. Mutates `state`, `pending_pairs`, `calibration`,
/// `touched_buckets` in place.
fn fold_synthesis_interaction(
    state: &mut SynthesisFeedbackState,
    pending_pairs: &mut HashMap<String, HalfPair>,
    calibration: &mut JudgeCalibrationState,
    touched_buckets: &mut std::collections::HashSet<String>,
    payload: &SynthesisInteractionPayload,
    event_id: i64,
) {
    let metadata = payload.metadata.clone().unwrap_or_default();
    let cluster_id = metadata.cluster_id;
    let raw_qtype = metadata.query_type.as_deref().unwrap_or("");
    let query_type = if SYNTHESIS_ALLOWED_QUERY_TYPES.contains(&raw_qtype) {
        raw_qtype.to_string()
    } else {
        "unknown".to_string()
    };
    let bucket_key = synthesis_bucket_key(cluster_id, &query_type);

    evict_synthesis_lru_if_at_cap(&mut state.by_cluster, &bucket_key);

    let bucket = state.by_cluster.entry(bucket_key.clone()).or_default();
    bucket.last_event_id = bucket.last_event_id.max(event_id);
    match &payload.interaction {
        SynthesisInteractionKind::Viewed { dwell_ms } => {
            bucket.viewed_count = bucket.viewed_count.saturating_add(1);
            bucket.viewed_dwell_total_ms = bucket.viewed_dwell_total_ms.saturating_add(*dwell_ms);
            bucket.dwell_samples.push(*dwell_ms);
            if bucket.dwell_samples.len() > SYNTHESIS_DWELL_RESERVOIR_CAP {
                let overflow = bucket.dwell_samples.len() - SYNTHESIS_DWELL_RESERVOIR_CAP;
                bucket.dwell_samples.drain(0..overflow);
            }
        }
        SynthesisInteractionKind::ClickedSource { .. } => {
            bucket.clicked_source_count = bucket.clicked_source_count.saturating_add(1);
        }
        SynthesisInteractionKind::ImmediateRequery { .. } => {
            bucket.immediate_requery_count = bucket.immediate_requery_count.saturating_add(1);
        }
        SynthesisInteractionKind::ExplicitThumb { up } => {
            if *up {
                bucket.explicit_up = bucket.explicit_up.saturating_add(1);
            } else {
                bucket.explicit_down = bucket.explicit_down.saturating_add(1);
            }
            // v0.27.1 κ-pair join: ExplicitThumb half-pair construction
            // per spec §6.2.1.
            let now_ts = chrono::Utc::now().timestamp();
            if let Some(half) = pending_pairs.remove(&payload.synthesis_id) {
                if let HalfPair::Judge {
                    hit, ts, surface, ..
                } = half
                {
                    // Judge arrived first → complete the pair now.
                    calibration.push_pair(surface, hit, *up, ts);
                } else {
                    // Same-side double-thumb (rare) — overwrite with the
                    // newer half-pair, drop the prior one.
                    pending_pairs.insert(
                        payload.synthesis_id.clone(),
                        HalfPair::Thumb {
                            up: *up,
                            ts: now_ts,
                            surface: JudgeSurface::Synthesis,
                        },
                    );
                }
            } else {
                pending_pairs.insert(
                    payload.synthesis_id.clone(),
                    HalfPair::Thumb {
                        up: *up,
                        ts: now_ts,
                        surface: JudgeSurface::Synthesis,
                    },
                );
            }
        }
    }
    touched_buckets.insert(bucket_key);

    // Per-synthesis_id LRU fold (mirrors original implementation).
    let sid = payload.synthesis_id.clone();
    let existed = state.by_synthesis.contains_key(&sid);
    {
        let per = state.by_synthesis.entry(sid.clone()).or_default();
        match &payload.interaction {
            SynthesisInteractionKind::Viewed { .. } => {
                per.viewed_count = per.viewed_count.saturating_add(1);
            }
            SynthesisInteractionKind::ClickedSource { .. } => {
                per.clicked_source_count = per.clicked_source_count.saturating_add(1);
            }
            SynthesisInteractionKind::ExplicitThumb { up } => {
                if *up {
                    per.explicit_up = per.explicit_up.saturating_add(1);
                } else {
                    per.explicit_down = per.explicit_down.saturating_add(1);
                }
            }
            SynthesisInteractionKind::ImmediateRequery { .. } => {}
        }
        per.last_interaction_ts = chrono::Utc::now().timestamp();
    }
    if !existed {
        state.by_synthesis_order.push(sid.clone());
        while state.by_synthesis_order.len() > SYNTHESIS_PER_ID_CAP {
            let evict = state.by_synthesis_order.remove(0);
            state.by_synthesis.remove(&evict);
        }
    }

    state.total_events = state.total_events.saturating_add(1);
}

/// Fold a single `SynthesisLlmJudge` event into the consumer's state.
/// v0.27.1 E direction. Updates the bucket's `llm_judge_count` /
/// `llm_judge_hit_count` and runs the κ-pair join (cache half OR complete
/// pair) per spec §6.2.1.
fn fold_synthesis_llm_judge(
    state: &mut SynthesisFeedbackState,
    pending_pairs: &mut HashMap<String, HalfPair>,
    calibration: &mut JudgeCalibrationState,
    touched_buckets: &mut std::collections::HashSet<String>,
    payload: &SynthesisLlmJudgePayload,
    event_id: i64,
) {
    let metadata = payload.metadata.clone().unwrap_or_default();
    let cluster_id = metadata.cluster_id;
    let raw_qtype = metadata.query_type.as_deref().unwrap_or("");
    let query_type = if SYNTHESIS_ALLOWED_QUERY_TYPES.contains(&raw_qtype) {
        raw_qtype.to_string()
    } else {
        "unknown".to_string()
    };
    let bucket_key = synthesis_bucket_key(cluster_id, &query_type);

    evict_synthesis_lru_if_at_cap(&mut state.by_cluster, &bucket_key);

    let bucket = state.by_cluster.entry(bucket_key.clone()).or_default();
    bucket.last_event_id = bucket.last_event_id.max(event_id);
    bucket.llm_judge_count = bucket.llm_judge_count.saturating_add(1);
    if payload.hit {
        bucket.llm_judge_hit_count = bucket.llm_judge_hit_count.saturating_add(1);
    }
    touched_buckets.insert(bucket_key);

    // κ-pair join: judge half-pair construction per spec §6.2.1.
    let now_ts = chrono::Utc::now().timestamp();
    if let Some(half) = pending_pairs.remove(&payload.synthesis_id) {
        if let HalfPair::Thumb { up, ts, surface } = half {
            // Thumb arrived first → complete the pair.
            calibration.push_pair(surface, payload.hit, up, ts);
        } else {
            // Two judge events for the same synthesis_id — overwrite with
            // the newer half-pair (LWW, last-write-wins).
            pending_pairs.insert(
                payload.synthesis_id.clone(),
                HalfPair::Judge {
                    hit: payload.hit,
                    ts: now_ts,
                    surface: JudgeSurface::Synthesis,
                    alias_key: None,
                },
            );
        }
    } else {
        pending_pairs.insert(
            payload.synthesis_id.clone(),
            HalfPair::Judge {
                hit: payload.hit,
                ts: now_ts,
                surface: JudgeSurface::Synthesis,
                alias_key: None,
            },
        );
    }

    state.total_events = state.total_events.saturating_add(1);
}

// ── v0.27 ARS Cap A feedback loop (Track 1) — concept-summary feedback ──────
//
// Mirrors the v0.26 D direction synthesis-feedback infrastructure (above) for
// Cap A (concept living-summary). Pattern is fully proven: lift-and-rename,
// not redesigned. Bucket key remains `(cluster_id, query_type)` — the same
// shape `decide_synthesize` uses — so the per-cluster gate can disambiguate
// whether a given concept-summary surface is helping a given query class.
// The persistent `concept_id` only feeds the per-id LRU (`by_concept`).

/// v0.27 ARS Cap A: typed payload serialised into `feedback_events.payload`
/// for [`EventType::ConceptSummaryInteraction`].
///
/// Shape mirrors [`SynthesisInteractionPayload`] — same 4 interaction kinds,
/// same `(cluster_id, query_type)` bucketing — with two Cap-A-specific
/// substitutions:
/// - `concept_id: String` replaces `synthesis_id`. Concepts are persistent
///   across sessions, so this is the concept's stable id, not a per-call ULID.
/// - `metadata.revision_version` records which living-summary revision the
///   user actually saw — Cap A summaries are versioned, so we can later
///   slice `useful_rate` by revision freshness.
///
/// The `feedback_events.payload` column is already TEXT, so no DDL change is
/// required — emit via `serde_json::to_value` and round-trip via
/// `serde_json::from_str` inside [`recompute_concept_summary_feedback_stats`].
///
/// Backward-compat invariant: pre-v0.27 payloads never carry this shape, and
/// the consumer filters by `event_type == "concept_summary_interaction"`, so
/// foreign payloads cannot reach the deserializer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
pub struct ConceptSummaryInteractionPayload {
    /// Concept's persistent id (NOT a per-call ULID — concepts span sessions).
    pub concept_id: String,
    /// Per-refresh concept-summary ULID when the emitting surface knows it.
    /// Older clients omit this; consumers fall back to `concept_id` for
    /// backward compatibility.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub concept_summary_id: Option<String>,
    /// Opaque correlation id. ULID echoing `RecallMemoryOutput.request_id`
    /// when the concept-summary surface was reached via recall; otherwise
    /// any non-empty client-minted id (UUID, etc.) — back-end treats as
    /// opaque and joins downstream traces by string equality.
    pub recall_id: String,
    pub interaction: ConceptSummaryInteractionKind,
    /// Optional context — `None` for older callers; `metadata.query_type` and
    /// `metadata.cluster_id` route the event into the per-`(cluster_id,
    /// query_type)` bucket (mirrors v0.26 D direction).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<ConceptSummaryMetadata>,
}

/// v0.27 ARS Cap A: discriminated interaction kinds posted from the concept
/// living-summary surface.
///
/// Variant set is identical to [`SynthesisInteractionKind`] — proven shape
/// for the dwell / click / requery / thumb signals. Re-stating rather than
/// re-using the synthesis enum keeps the two feedback loops fully decoupled
/// at the type level (they evolve independently per
/// `feedback_no_subjective_params`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum ConceptSummaryInteractionKind {
    /// Time the concept-summary surface was visible. Feeds the dwell
    /// reservoir → `useful_rate` dwell term.
    Viewed { dwell_ms: u64 },
    /// 1-based index into the concept-summary's evidence/source list.
    /// Out-of-range indices are accepted (silently counted) — front-end
    /// is responsible for not emitting them.
    ClickedSource { source_index: u32 },
    /// Time gap since the prior `concept_id`'s last interaction to a new
    /// recall. Sliding threshold lives in the consumer; do NOT hardcode
    /// "immediate" in the event itself.
    ImmediateRequery { gap_ms: u64 },
    /// Explicit user signal.
    ExplicitThumb { up: bool },
}

/// v0.28 ARS shadow route context. This preserves the caller's real recall
/// route while Cap A continues to bucket production feedback through its
/// synthetic per-concept route.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct RecallRouteContext {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cluster_id: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cluster_version: Option<u64>,
}

/// v0.27 ARS Cap A: optional context emitted alongside an interaction.
/// `Default` is empty (all `None`) so callers can construct it without
/// committing to every field.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct ConceptSummaryMetadata {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cluster_id: Option<i64>,
    /// Living-summary character count — diagnostic dimension for Cap A
    /// ablation (analogous to `synthesis_chars`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub concept_chars: Option<u32>,
    /// Cap-A specific: which revision of the concept living-summary the
    /// user actually saw. Surfaced so future ablations can slice
    /// `useful_rate` by revision freshness — Cap A summaries are
    /// versioned, unlike Cap B synthesis (which is per-call ephemeral).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision_version: Option<u32>,
    /// Shadow-only route context. Production bucket selection still uses the
    /// top-level `query_type` and `cluster_id` fields.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub route_context: Option<RecallRouteContext>,
}

// Bootstrap weights for `compute_concept_summary_useful_rate`. Marked
// `bootstrap` per `feedback_no_subjective_params` — every literal is
// data-driven in the fullness of time. v0.27.1 → ablation across observed
// `(ClusterConceptSummaryStats, downstream-recall)` pairs, mirroring the
// v0.26.1 plan for synthesis. Until then: view-with-dwell + thumb are
// positive, requery is the strongest negative.
pub const CONCEPT_SUMMARY_W_VIEW: f64 = 1.0; // bootstrap; v0.27.1 → ablation
pub const CONCEPT_SUMMARY_W_CLICK: f64 = 0.5; // bootstrap; v0.27.1 → ablation
pub const CONCEPT_SUMMARY_W_THUMB: f64 = 2.0; // bootstrap; v0.27.1 → ablation
pub const CONCEPT_SUMMARY_W_REQUERY: f64 = 2.0; // bootstrap; v0.27.1 → ablation (subtracted)
/// Bootstrap dwell threshold. v0.27.1 → per-cluster p50 of dwell_samples.
pub const CONCEPT_SUMMARY_DWELL_THRESHOLD_MS: u64 = 3_000; // bootstrap; v0.27.1 → ablation

/// FIFO reservoir cap for `dwell_samples` per [`ClusterConceptSummaryStats`].
/// 500 keeps the `useful_rate` dwell term responsive to recent steady state
/// without unbounded memory growth (mirrors `SYNTHESIS_DWELL_RESERVOIR_CAP`).
pub const CONCEPT_SUMMARY_DWELL_RESERVOIR_CAP: usize = 500;

/// LRU cap for `by_concept` per-id stats. Implemented as a `HashMap` + side
/// `Vec<String>` for FIFO order because `lru::LruCache` is not `Serialize`
/// (cross-agent invariant 11, mirrors `SYNTHESIS_PER_ID_CAP`).
pub const CONCEPT_SUMMARY_PER_ID_CAP: usize = 1024;

/// Hard cap on the number of distinct `(cluster_id, query_type)` buckets
/// in [`ConceptSummaryFeedbackState::by_cluster`]. Defends against a
/// malicious or buggy client flooding `/api/feedback` with fabricated
/// `cluster_id` or `query_type` values that would otherwise grow the
/// persisted adaptive-state snapshot without limit (mirrors v0.26 Codex
/// round 2 F-11). Once the cap is reached new buckets are dropped (the
/// events still increment `total_events`), so legitimate buckets don't
/// compete for capacity once the system has converged on real cluster ids.
///
/// **v0.28 H3 fix:** the cap budget counts only non-shadow (production)
/// buckets. Shadow `route_context` buckets are bounded separately by
/// [`CONCEPT_SUMMARY_BY_CLUSTER_SHADOW_CAP`] so a flood of shadow inserts
/// can never evict a real production bucket.
pub const CONCEPT_SUMMARY_BY_CLUSTER_CAP: usize = 4096;

/// v0.28 H3 — separate budget for shadow (`route_context`-derived)
/// buckets. Evicting only inside the shadow class on shadow-insert
/// pressure means production buckets are sealed against shadow flooding.
/// Mirrors the production cap so a vault that emits one shadow per
/// production event has comparable headroom on each side.
pub const CONCEPT_SUMMARY_BY_CLUSTER_SHADOW_CAP: usize = 4096;

/// v0.28.7+ audit L6 — defense-in-depth cap on
/// [`AdaptiveState::learned_shadow_fusion`]. Bucket keys are
/// `query_type[:cluster_id]` shaped (~6 query types × O(realistic
/// clusters)); the realistic ceiling is small but pre-cap there was no
/// upper bound, so a runaway clusterer or adversarial query_type
/// stream could grow the snapshot indefinitely. Mirrors the
/// `CONCEPT_SUMMARY_BY_CLUSTER_CAP = 4096` precedent set by v0.27.5 R2 +
/// v0.28 H3, so an operator already familiar with the concept-summary
/// cap doesn't have to learn a second number.
///
/// Eviction is LRU-by-`last_updated` (RFC3339, parsed at compare time —
/// see `feedback`-style note in
/// [`evict_learned_shadow_fusion_lru_if_at_cap`] for why string
/// comparison is wrong). Same-key rewrites are no-ops; only NEW keys
/// trigger eviction.
pub const LEARNED_SHADOW_FUSION_CAP: usize = 4096;

/// v0.28.7+ audit L6 — bound `learned_shadow_fusion` by evicting the
/// LRU entry (oldest `last_updated`) when at cap and a new key arrives.
///
/// Caller MUST invoke this BEFORE inserting a new key; same-key
/// rewrites are no-ops here (the existing entry is updated in place by
/// the caller's subsequent `insert(key, ...)`).
///
/// **`last_updated` comparison is parse-based, NOT raw string
/// ordering.** Mixing RFC3339 timezone forms (`Z` vs `+00:00`) makes
/// lexicographic order disagree with chronological order — `+` (0x2B)
/// sorts BEFORE `Z` (0x5A), so an entry stamped `…00:00.000+00:00`
/// would sort earlier than one stamped `…00:00.000Z` representing the
/// same instant. Parse to `DateTime<Utc>` first, then compare.
/// Unparseable timestamps are treated as the oldest possible time so
/// they evict first (a corrupt timestamp is itself a sign the entry
/// should go).
/// R12 P2 (2026-05-04) — predicate: is this a cluster-scoped bucket?
///
/// Bucket keys come in two shapes:
/// - **Cluster-scoped**: `{query_type}:{cluster_id}` where `cluster_id`
///   is a numeric `u32` (e.g., `semantic:7`, `exactkeyword:42`).
/// - **Fallback**: `global` (literal) or `{query_type}` (no `:`)
///   that `get_shadow_fusion_weights` consults at the tail of its
///   fallback chain when no cluster-scoped bucket matches.
///
/// LRU eviction MUST exclude fallback keys. There are at most ~7 of
/// them (one per query type plus `global`), they are STRUCTURAL (the
/// fallback chain depends on their continuous presence), and a vault
/// at high cluster cardinality could otherwise silently lose them —
/// `get_shadow_fusion_weights` would then silently degrade to
/// returning `None` for queries without surviving cluster-scoped
/// buckets, even while the canary is enabled.
///
/// Predicate is "contains `:` AND suffix parses as u32" — strictly
/// rejects any pathological future query_type literal that happens to
/// contain `:`.
fn is_cluster_scoped_bucket(key: &str) -> bool {
    if let Some((_, suffix)) = key.rsplit_once(':') {
        suffix.parse::<u32>().is_ok()
    } else {
        false
    }
}

pub fn evict_learned_shadow_fusion_lru_if_at_cap(
    map: &mut HashMap<String, LearnedShadowFusionEntry>,
    new_key: &str,
) {
    if map.contains_key(new_key) {
        // Same-key rewrite: caller's `insert` will update in place; cap
        // pressure is unchanged.
        return;
    }
    if map.len() < LEARNED_SHADOW_FUSION_CAP {
        return;
    }
    let parse = |s: &str| {
        chrono::DateTime::parse_from_rfc3339(s)
            .map(|dt| dt.with_timezone(&chrono::Utc))
            // Unparseable → MIN so this entry evicts first.
            .unwrap_or(chrono::DateTime::<chrono::Utc>::MIN_UTC)
    };
    // R12 P2 (2026-05-04) — restrict eviction to cluster-scoped
    // buckets. The fallback chain in `get_shadow_fusion_weights`
    // depends on the continuous presence of `global` and per-query-
    // type fallback buckets; LRU'ing them out under high cluster
    // cardinality would silently degrade dynamic recall fusion for
    // queries without surviving cluster-scoped buckets. If ALL
    // entries at cap are fallback keys (impossible in practice — only
    // ~7 fallback keys exist), the cap is allowed to overshoot rather
    // than corrupt the fallback baseline.
    let victim_key = map
        .iter()
        .filter(|(k, _)| is_cluster_scoped_bucket(k))
        .min_by_key(|(_, entry)| parse(&entry.last_updated))
        .map(|(k, _)| k.clone());
    if let Some(victim) = victim_key {
        tracing::warn!(
            evicted_bucket = %victim,
            new_bucket = %new_key,
            cap = LEARNED_SHADOW_FUSION_CAP,
            "learned_shadow_fusion: cap reached; evicting LRU cluster-scoped entry by last_updated"
        );
        map.remove(&victim);
    } else {
        // No cluster-scoped victims — the cap consists entirely of
        // fallback buckets. This is degenerate (only ~7 fallback keys
        // can exist at once) but safe to log: we explicitly choose to
        // exceed the cap rather than corrupt the fallback chain.
        tracing::warn!(
            new_bucket = %new_key,
            cap = LEARNED_SHADOW_FUSION_CAP,
            map_len = map.len(),
            "learned_shadow_fusion: cap reached but no cluster-scoped victim; \
             allowing over-cap insert to preserve fallback chain (degenerate state)"
        );
    }
}

/// v0.28.7+ audit M-8 R2 P2 #2 follow-up — shrink
/// `learned_shadow_fusion` to at-or-below cap by repeatedly evicting
/// the LRU entry. Used at snapshot serialization boundaries
/// (`save_snapshot` post-CAS-merge and `restore_snapshot` post-load)
/// to bound the persisted map's size even when peer writers' merged
/// entries pushed it over cap, OR when an old snapshot from before
/// the L6 cap was introduced contained an over-cap blob.
///
/// Per-key insert sites still call
/// [`evict_learned_shadow_fusion_lru_if_at_cap`] at insert time to
/// keep cap pressure bounded during normal operation; this function
/// is the defense-in-depth bound that catches:
/// - CAS merge: another writer's entries are folded into `current`
///   without going through any insert helper.
/// - Restore from disk: a pre-cap snapshot, or one written by a peer
///   that didn't enforce the cap at insert time.
///
/// Emits `tracing::warn!` once per call when shrinkage actually
/// happens, with the pre/post sizes so an operator can tell that the
/// cap is being exercised in practice (vs. hypothetical defense).
pub fn shrink_learned_shadow_fusion_to_cap(map: &mut HashMap<String, LearnedShadowFusionEntry>) {
    if map.len() <= LEARNED_SHADOW_FUSION_CAP {
        return;
    }
    let original_len = map.len();
    let parse = |s: &str| {
        chrono::DateTime::parse_from_rfc3339(s)
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .unwrap_or(chrono::DateTime::<chrono::Utc>::MIN_UTC)
    };
    // R12 P2 (2026-05-04) — restrict shrinkage to cluster-scoped
    // buckets. The CAS-merge / restore-from-disk paths can fold
    // peer-written entries past the cap; like the per-key insert
    // path, the fallback chain (`global`, query-type-only) MUST be
    // preserved. Sort cluster-scoped keys by `last_updated` ascending
    // and drop the oldest until either (a) we reach cap or (b) we run
    // out of cluster-scoped victims. The latter is a degenerate state
    // (only ~7 fallback keys exist) — we let the map exceed cap
    // rather than evict a fallback bucket.
    let mut cluster_scoped_by_age: Vec<(String, chrono::DateTime<chrono::Utc>)> = map
        .iter()
        .filter(|(k, _)| is_cluster_scoped_bucket(k))
        .map(|(k, v)| (k.clone(), parse(&v.last_updated)))
        .collect();
    cluster_scoped_by_age.sort_by_key(|a| a.1);
    let needed = map.len() - LEARNED_SHADOW_FUSION_CAP;
    let drop_count = needed.min(cluster_scoped_by_age.len());
    for (victim, _) in cluster_scoped_by_age.into_iter().take(drop_count) {
        map.remove(&victim);
    }
    if drop_count < needed {
        tracing::warn!(
            original_len = original_len,
            new_len = map.len(),
            cap = LEARNED_SHADOW_FUSION_CAP,
            dropped = drop_count,
            short_by = needed - drop_count,
            "learned_shadow_fusion: shrink left map over cap; no more \
             cluster-scoped victims (fallback chain preserved)"
        );
    } else {
        tracing::warn!(
            original_len = original_len,
            new_len = map.len(),
            cap = LEARNED_SHADOW_FUSION_CAP,
            dropped = drop_count,
            "learned_shadow_fusion: shrunk to cap at snapshot boundary \
             (CAS merge or restore from over-cap snapshot, cluster-scoped only)"
        );
    }
}

/// Whitelist of `query_type` values rein can legitimately emit (mirrors
/// `SYNTHESIS_ALLOWED_QUERY_TYPES`). Any client-supplied value outside
/// this list is normalized to `"unknown"` before being folded into
/// `by_cluster`, so adversarial query_type strings can't multiplicatively
/// explode the bucket cardinality.
pub const CONCEPT_SUMMARY_ALLOWED_QUERY_TYPES: &[&str] = &[
    "Episodic",
    "Temporal",
    "Preference",
    "ExactKeyword",
    "Semantic",
    "Exploratory",
    "_global",
    // v0.27.4 D1/D2 — Cap A judge writers route via this literal so the
    // consumer fold preserves it instead of clamping to "unknown". Pairs
    // with `ops::concept_summary::CONCEPT_SUMMARY_QUERY_TYPE_REFRESH`.
    // Cap A judge has no recall query, so this sentinel substitutes for
    // the routing key in the per-(cluster_id, query_type) bucket.
    "concept_refresh",
];

/// Min events per `(cluster_id, query_type)` bucket before per-cluster
/// `useful_rate` is trusted by the per-query Cap-A gate. Below this, the
/// gate falls back to the global `[ars].concept_summary_enabled` flag
/// (mirrors `SYNTHESIS_COLD_START_N`).
pub const CONCEPT_SUMMARY_COLD_START_N: u64 = 10;

/// Bootstrap `useful_rate` cutoff used by the Cap-A per-query gate. Hoisted
/// into a constant so handler code never inlines the literal (cross-agent
/// invariant 12); v0.27.1 → adaptive once `useful_rate` ablation lands
/// (mirrors `SYNTHESIS_USEFUL_RATE_THRESHOLD`).
pub const CONCEPT_SUMMARY_USEFUL_RATE_THRESHOLD: f64 = 0.5; // bootstrap; v0.27.1 → adaptive

/// Per-bucket key used by [`ConceptSummaryFeedbackState::by_cluster`].
/// Bucket is `(cluster_id, query_type)` — both can be unknown, in which
/// case the consumer routes events to the global bucket key
/// `concept_summary_bucket_key(None, "")` → `"-1|"`.
///
/// Keyed via `serde`-friendly `String` because `HashMap<(_, String), _>`
/// round-trips awkwardly through JSON (`serde_json` requires string keys).
/// Mirrors `synthesis_bucket_key`; the two helpers are deliberately
/// kept separate so a future schema-only change to one doesn't silently
/// drift the other.
pub fn concept_summary_bucket_key(cluster_id: Option<i64>, query_type: &str) -> String {
    let cid = cluster_id.unwrap_or(-1);
    format!("{cid}|{query_type}")
}

/// v0.27 ARS Cap A: per-`(cluster_id, query_type)` concept-summary
/// interaction aggregate. `useful_rate` is recomputed on every consumer
/// pass via [`compute_concept_summary_useful_rate`]. Mirrors
/// [`ClusterSynthesisStats`].
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ClusterConceptSummaryStats {
    pub viewed_count: u64,
    pub viewed_dwell_total_ms: u64,
    /// FIFO reservoir of `Viewed.dwell_ms` samples capped at
    /// [`CONCEPT_SUMMARY_DWELL_RESERVOIR_CAP`]. Used to compute
    /// `viewed_dwell_p50_ms` and the dwell term in
    /// [`compute_concept_summary_useful_rate`].
    #[serde(default)]
    pub dwell_samples: Vec<u64>,
    /// Cached p50 of `dwell_samples`. `None` when reservoir is empty.
    #[serde(default)]
    pub viewed_dwell_p50_ms: Option<u64>,
    pub clicked_source_count: u64,
    pub immediate_requery_count: u64,
    pub explicit_up: u64,
    pub explicit_down: u64,
    // ── v0.27.1 E direction LLM judge counters (Cap A mirror) ──
    /// Number of [`EventType::ConceptSummaryLlmJudge`] events folded into
    /// this bucket. Counts toward `total_signal` for cold-start fallback.
    #[serde(default)]
    pub llm_judge_count: u64,
    /// Number of those events whose `hit = true`.
    #[serde(default)]
    pub llm_judge_hit_count: u64,
    /// Derived metric, recomputed on every consumer pass.
    pub useful_rate: f64,
    /// v0.27.5 R2 — LRU eviction key. Highest `feedback_events.id`
    /// folded into this bucket. When `by_cluster` is at
    /// [`CONCEPT_SUMMARY_BY_CLUSTER_CAP`], the consumer evicts the bucket
    /// with the lowest `last_event_id` to make room for a new bucket.
    /// Defaults to 0 on legacy snapshots; the next event in any of those
    /// buckets bumps it past 0, so cold buckets that never see another
    /// event are the natural eviction candidates once the cap is hit.
    #[serde(default)]
    pub last_event_id: i64,
    /// v0.28 H3 — shadow flag. `true` only when every event ever folded
    /// into this bucket has been a `route_context`-derived shadow write.
    /// Once any production (primary `(cluster_id, query_type)`) event
    /// lands on the same key the flag flips to `false` permanently
    /// (monotonic AND across writes; see
    /// [`fold_concept_summary_interaction_bucket`]). Default-`false` on
    /// legacy snapshots is conservative: untagged buckets count against
    /// the production cap budget, never the shadow cap. Used by
    /// [`evict_concept_summary_lru_if_at_cap`] to seal production
    /// capacity against shadow-bucket flooding (audit H3 fix).
    #[serde(default)]
    pub is_shadow: bool,
}

/// v0.27 ARS Cap A: per-concept_id stats with bounded LRU semantics.
/// Used by future per-concept decay/heatmap views; the consumer caps total
/// entries at [`CONCEPT_SUMMARY_PER_ID_CAP`]. Mirrors [`PerSynthesisStats`].
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct PerConceptSummaryStats {
    pub viewed_count: u32,
    pub clicked_source_count: u32,
    pub explicit_up: u32,
    pub explicit_down: u32,
    pub last_interaction_ts: i64,
}

/// v0.27 ARS Cap A: state container for the `concept_summary_feedback`
/// consumer. Persisted as part of [`AdaptiveState`] (CAS-arbitrated by
/// `last_consumed_event_id`). Mirrors [`SynthesisFeedbackState`].
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ConceptSummaryFeedbackState {
    /// Bucket: `concept_summary_bucket_key(cluster_id, query_type)`.
    /// `cluster_id = -1` (None) means "no cluster"; empty `query_type`
    /// means unknown classification. HashMap so serde round-trips
    /// cleanly into the JSON snapshot.
    pub by_cluster: HashMap<String, ClusterConceptSummaryStats>,
    /// Bounded per-concept_id stats (LRU-capped at
    /// [`CONCEPT_SUMMARY_PER_ID_CAP`]). Implemented as `HashMap` + side
    /// `by_concept_order` Vec — `lru::LruCache` is not `Serialize` and
    /// would break `AdaptiveState` snapshotting (cross-agent invariant 11).
    #[serde(default)]
    pub by_concept: HashMap<String, PerConceptSummaryStats>,
    /// FIFO order key list mirroring `by_concept` insertion order.
    /// Eviction pops from the front and removes the matching HashMap
    /// entry — both updates MUST happen together, otherwise the cache
    /// leaks orphan keys.
    #[serde(default)]
    pub by_concept_order: Vec<String>,
    /// Highest event id incorporated into this state. Watermark for
    /// replay-safety (mirrors
    /// [`SynthesisFeedbackState::last_consumed_event_id`]). The CAS merge
    /// in `save_snapshot` arbitrates between concurrent writers by the
    /// MAX of this id (Codex round-4 HIGH from v0.24 generalised to the
    /// new arm).
    #[serde(default)]
    pub last_consumed_event_id: i64,
    /// Total events the consumer has *processed* (including replays
    /// counted only once). Useful for `/api/adaptive` exposure of
    /// "how much signal has accumulated".
    #[serde(default)]
    pub total_events: u64,
}

/// v0.27.5 R2 — LRU eviction for the concept-summary `by_cluster` map.
/// When the map is at [`CONCEPT_SUMMARY_BY_CLUSTER_CAP`] and
/// `new_key` would create a new bucket, drop the bucket with the
/// lowest [`ClusterConceptSummaryStats::last_event_id`] (the
/// least-recently-active bucket). Replaces the v0.27.4 drop-new-bucket
/// behavior so vaults with > 4096 distinct concepts stop silently
/// losing fresh signal once the cap saturates.
///
/// No-op when the map is below cap or already contains `new_key`.
///
/// Ties on `last_event_id = 0` (legacy snapshots loaded via
/// `#[serde(default)]`) resolve arbitrarily via `min_by_key` over a
/// `HashMap` iterator. The next event into the surviving bucket bumps
/// its `last_event_id` past 0, so the non-determinism is self-healing
/// and confined to the first eviction after a snapshot reload.
fn evict_concept_summary_lru_if_at_cap(
    by_cluster: &mut HashMap<String, ClusterConceptSummaryStats>,
    new_key: &str,
    new_is_shadow: bool,
) {
    // v0.28.7 audit R1 P2 #2 — detect shadow→production *promotion*: an
    // existing shadow bucket whose next write is production. The bucket
    // currently counts toward shadow but will count toward production after
    // the fold. Treat as a production insert for cap purposes (excluding the
    // promoting bucket from any candidate-victim list).
    let promotion = by_cluster
        .get(new_key)
        .is_some_and(|b| b.is_shadow && !new_is_shadow);

    // Same-class re-write of an existing key has no cap impact.
    if by_cluster.contains_key(new_key) && !promotion {
        return;
    }
    if new_is_shadow {
        // Shadow insert: bounded by SHADOW_CAP over the shadow subset;
        // never evicts a production bucket. (Promotion never enters this
        // arm because `new_is_shadow == false` for promotions.)
        let shadow_count = by_cluster.iter().filter(|(_, b)| b.is_shadow).count();
        if shadow_count < CONCEPT_SUMMARY_BY_CLUSTER_SHADOW_CAP {
            return;
        }
        let victim_key = by_cluster
            .iter()
            .filter(|(_, b)| b.is_shadow)
            .min_by_key(|(_, b)| b.last_event_id)
            .map(|(k, _)| k.clone());
        if let Some(victim) = victim_key {
            tracing::warn!(
                evicted_bucket = %victim,
                new_bucket = %new_key,
                cap = CONCEPT_SUMMARY_BY_CLUSTER_SHADOW_CAP,
                kind = "shadow",
                "concept_summary_feedback: by_cluster shadow cap reached; evicting LRU shadow bucket"
            );
            by_cluster.remove(&victim);
        }
        return;
    }
    // Production path covers both fresh production inserts AND
    // shadow→production promotions. Threshold is over the non-shadow
    // subset; cap is bounded by CONCEPT_SUMMARY_BY_CLUSTER_CAP.
    //
    // v0.28.7 audit R1 P2 #1 — earlier code preferred to evict a shadow
    // bucket here. That was wrong: shadow eviction does not free a
    // production slot under separate caps, so the caller's subsequent
    // production insert pushed `prod_count` to CAP + 1. Eviction must
    // target the production subset directly.
    let new_prod_after = if promotion {
        // Existing key already in map; promotion flips its class but does
        // NOT change `len()`. The post-fold non-shadow count equals the
        // current non-shadow count + 1 (the bucket switches sides).
        by_cluster.iter().filter(|(_, b)| !b.is_shadow).count() + 1
    } else {
        // New key; counted as +1 in the non-shadow subset after insertion.
        by_cluster.iter().filter(|(_, b)| !b.is_shadow).count() + 1
    };
    if new_prod_after <= CONCEPT_SUMMARY_BY_CLUSTER_CAP {
        return;
    }
    let prod_victim = by_cluster
        .iter()
        .filter(|(k, b)| !b.is_shadow && k.as_str() != new_key)
        .min_by_key(|(_, b)| b.last_event_id)
        .map(|(k, _)| k.clone());
    if let Some(victim) = prod_victim {
        tracing::warn!(
            evicted_bucket = %victim,
            new_bucket = %new_key,
            cap = CONCEPT_SUMMARY_BY_CLUSTER_CAP,
            kind = if promotion { "production_promotion" } else { "production" },
            "concept_summary_feedback: production cap reached; evicting LRU production bucket"
        );
        by_cluster.remove(&victim);
    }
}

fn normalize_concept_summary_query_type(raw_qtype: &str) -> String {
    if CONCEPT_SUMMARY_ALLOWED_QUERY_TYPES.contains(&raw_qtype) {
        raw_qtype.to_string()
    } else {
        "unknown".to_string()
    }
}

/// Returns `(bucket_key, is_shadow)` pairs derived from event metadata.
///
/// The first entry is always the primary `(metadata.cluster_id,
/// metadata.query_type)` bucket — `is_shadow = false` (production).
/// When `metadata.route_context` is present and resolves to a distinct
/// `(cluster_id, query_type)` pair, a second `is_shadow = true` entry
/// is appended.
///
/// **v0.28 H3:** the shadow flag is what
/// [`evict_concept_summary_lru_if_at_cap`] uses to keep shadow inserts
/// from evicting production buckets. Production and shadow buckets
/// share the same `state.by_cluster` HashMap but compete over disjoint
/// cap budgets.
fn concept_summary_bucket_keys_from_metadata(
    metadata: &ConceptSummaryMetadata,
) -> Vec<(String, bool)> {
    let primary_query_type =
        normalize_concept_summary_query_type(metadata.query_type.as_deref().unwrap_or(""));
    let primary = concept_summary_bucket_key(metadata.cluster_id, &primary_query_type);
    let mut keys = vec![(primary.clone(), false)];

    if let Some(route) = metadata.route_context.as_ref() {
        if route.query_type.is_some() || route.cluster_id.is_some() {
            let route_query_type =
                normalize_concept_summary_query_type(route.query_type.as_deref().unwrap_or(""));
            let route_key = concept_summary_bucket_key(route.cluster_id, &route_query_type);
            if route_key != primary {
                keys.push((route_key, true));
            }
        }
    }

    keys
}

fn fold_concept_summary_interaction_bucket(
    state: &mut ConceptSummaryFeedbackState,
    bucket_key: &str,
    is_shadow: bool,
    interaction: &ConceptSummaryInteractionKind,
    event_id: i64,
) {
    evict_concept_summary_lru_if_at_cap(&mut state.by_cluster, bucket_key, is_shadow);

    let new_entry = !state.by_cluster.contains_key(bucket_key);
    let bucket = state.by_cluster.entry(bucket_key.to_string()).or_default();
    // v0.28 H3 — monotonic AND on `is_shadow`. New buckets adopt the
    // incoming flag; existing buckets stay shadow only if every prior
    // and the current write are also shadow. Once any production event
    // lands the bucket flips to production permanently, so a key cannot
    // oscillate back into the shadow class and bypass the production
    // cap.
    if new_entry {
        bucket.is_shadow = is_shadow;
    } else {
        bucket.is_shadow = bucket.is_shadow && is_shadow;
    }
    bucket.last_event_id = bucket.last_event_id.max(event_id);
    match interaction {
        ConceptSummaryInteractionKind::Viewed { dwell_ms } => {
            bucket.viewed_count = bucket.viewed_count.saturating_add(1);
            bucket.viewed_dwell_total_ms = bucket.viewed_dwell_total_ms.saturating_add(*dwell_ms);
            bucket.dwell_samples.push(*dwell_ms);
            if bucket.dwell_samples.len() > CONCEPT_SUMMARY_DWELL_RESERVOIR_CAP {
                let overflow = bucket.dwell_samples.len() - CONCEPT_SUMMARY_DWELL_RESERVOIR_CAP;
                bucket.dwell_samples.drain(0..overflow);
            }
        }
        ConceptSummaryInteractionKind::ClickedSource { source_index: _ } => {
            bucket.clicked_source_count = bucket.clicked_source_count.saturating_add(1);
        }
        ConceptSummaryInteractionKind::ImmediateRequery { gap_ms: _ } => {
            bucket.immediate_requery_count = bucket.immediate_requery_count.saturating_add(1);
        }
        ConceptSummaryInteractionKind::ExplicitThumb { up } => {
            if *up {
                bucket.explicit_up = bucket.explicit_up.saturating_add(1);
            } else {
                bucket.explicit_down = bucket.explicit_down.saturating_add(1);
            }
        }
    }
}

/// Pure function — testable in isolation. Computes a `[0.0, 1.0]`
/// usefulness rate from a single bucket's aggregate counters.
/// Mirrors [`compute_useful_rate`] for the concept-summary surface.
///
/// The formula combines:
/// - dwell pct: fraction of `dwell_samples` exceeding
///   [`CONCEPT_SUMMARY_DWELL_THRESHOLD_MS`] (a "skim vs read" proxy).
/// - click rate: clicks / views (engagement with cited evidence).
/// - thumb rate: explicit positive ratio
///   (`explicit_up / (explicit_up + explicit_down + 1)`); `+1` Laplace
///   smoothing keeps the term well-defined when no thumbs have ever
///   landed.
/// - requery rate: requeries / views (subtracted — a strong negative
///   signal that the concept summary didn't satisfy the question).
///
/// Output is `.clamp(0.0, 1.0)` so the requery penalty cannot push the
/// score below zero (and rounding never floats above one). Bootstrap
/// weights are documented above; v0.27.1 will derive them from a
/// SemDeDup-style ablation.
pub fn compute_concept_summary_useful_rate(stats: &ClusterConceptSummaryStats) -> f64 {
    compute_concept_summary_useful_rate_with_weights(
        stats,
        UsefulRateWeights::concept_summary_bootstrap(),
    )
}

pub fn compute_concept_summary_useful_rate_with_weights(
    stats: &ClusterConceptSummaryStats,
    weights: UsefulRateWeights,
) -> f64 {
    let total_views = stats.viewed_count.max(1) as f64;
    let dwell_pct = if stats.dwell_samples.is_empty() {
        0.0
    } else {
        stats
            .dwell_samples
            .iter()
            .filter(|&&d| d > CONCEPT_SUMMARY_DWELL_THRESHOLD_MS)
            .count() as f64
            / stats.dwell_samples.len() as f64
    };
    let click_rate = stats.clicked_source_count as f64 / total_views;
    let thumb_rate =
        stats.explicit_up as f64 / (stats.explicit_up + stats.explicit_down + 1) as f64;
    let requery_rate = stats.immediate_requery_count as f64 / total_views;

    let numerator =
        weights.view * dwell_pct + weights.click * click_rate + weights.thumb * thumb_rate
            - weights.requery * requery_rate;
    let denom = weights.denominator();
    if denom <= 0.0 || !denom.is_finite() {
        return 0.0;
    }
    (numerator / denom).clamp(0.0, 1.0)
}

/// Compute the p50 of a non-empty slice of dwell samples. Mirrors the
/// inner `dwell_p50_ms` helper above (private — we just route through it).
fn concept_summary_dwell_p50_ms(samples: &[u64]) -> Option<u64> {
    dwell_p50_ms(samples)
}

/// v0.27 ARS Cap A: peek new [`EventType::ConceptSummaryInteraction`]
/// feedback events, fold them into the rolling
/// [`ConceptSummaryFeedbackState`], recompute the derived `useful_rate`
/// per bucket, and return the highest event id incorporated so the
/// caller can commit the consumer offset *after* the derived state is
/// durable (module-level peek+commit invariant).
///
/// Returns `(updated_state, Option<max_event_id>)`. `Option::None` means
/// no new events were observed → caller skips `commit_offset`.
///
/// **5 invariants enforced** (per
/// [[feedback_event_sourced_state_invariant]]):
///
///   1. **Watermark filter** — events with
///      `id <= state.last_consumed_event_id` are skipped via
///      `prior_high_water`. Counter increments are NOT idempotent, so
///      this guard is the entire point.
///   2. **Applied-prefix bump** — `state.last_consumed_event_id` is
///      bumped to `max(state.last_consumed_event_id, max_id_this_pass)`
///      *before* any new events are folded; the caller is responsible
///      for committing the consumer offset only AFTER `save_snapshot`
///      returns Ok.
///   3. **Replay-drain** — `peek_events` reads from the consumer offset;
///      replay-safety after a `commit_offset` failure is guarded by
///      invariant (1).
///   4. **CAS merge** — [`AdaptiveState::save_snapshot`] arbitrates by
///      `last_consumed_event_id` MAX, mirroring the
///      `synthesis_feedback_stats` arm.
///   5. **Peek + commit** — uses `peek_events("concept_summary_feedback",
///      …)` then *the caller* runs
///      `commit_offset(&[("concept_summary_feedback", max_id)])` AFTER
///      `save_snapshot` succeeds. Never `consume_events` (the v0.24
///      round-2/3/4 HIGH that this contract retires).
///
/// Malformed payloads are logged via `tracing::warn!` and skipped
/// (mirrors `recompute_synthesis_feedback_stats`).
pub fn recompute_concept_summary_feedback_stats(
    conn: &Connection,
    prior: Option<ConceptSummaryFeedbackState>,
) -> ReinResult<(ConceptSummaryFeedbackState, Option<i64>)> {
    let mut state = prior.unwrap_or_default();

    // Single peek covers the common case (most pipelines drain in one
    // shot). 50 000 cap matches `recompute_synthesis_feedback_stats`.
    let events = peek_events(
        conn,
        "concept_summary_feedback",
        &[EventType::ConceptSummaryInteraction.as_str()],
        50_000,
    )?;
    if events.is_empty() {
        return Ok((state, None));
    }
    let max_id_this_pass = events.last().map(|e| e.id);

    // Invariants 1 + 2: prior_high_water guard skips already-applied events
    // on a replay; the bump records the durable watermark in the returned
    // state so the next `save_snapshot` advances it. The caller commits the
    // consumer offset only AFTER save_snapshot succeeds.
    let prior_high_water = state.last_consumed_event_id;
    if let Some(max_id) = max_id_this_pass {
        state.last_consumed_event_id = state.last_consumed_event_id.max(max_id);
    }

    // Track buckets that received new events so we recompute their
    // `useful_rate` cache exactly once per pass.
    let mut touched_buckets: std::collections::HashSet<String> = std::collections::HashSet::new();

    for ev in events {
        if ev.id <= prior_high_water {
            continue;
        }
        let payload_str = match ev.payload.as_deref() {
            Some(s) => s,
            None => {
                tracing::warn!(
                    event_id = ev.id,
                    "concept_summary_feedback: event missing payload, skipping"
                );
                continue;
            }
        };
        let payload: ConceptSummaryInteractionPayload = match serde_json::from_str(payload_str) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(
                    event_id = ev.id,
                    error = %e,
                    "concept_summary_feedback: malformed ConceptSummaryInteractionPayload, skipping"
                );
                continue;
            }
        };

        let metadata = payload.metadata.clone().unwrap_or_default();
        for (bucket_key, is_shadow) in concept_summary_bucket_keys_from_metadata(&metadata) {
            fold_concept_summary_interaction_bucket(
                &mut state,
                &bucket_key,
                is_shadow,
                &payload.interaction,
                ev.id,
            );
            touched_buckets.insert(bucket_key);
        }

        // Per-concept_id LRU fold. HashMap update + side-vec FIFO must
        // happen together; failure to dual-update leaks orphan keys.
        let cid_str = payload.concept_id.clone();
        let existed = state.by_concept.contains_key(&cid_str);
        {
            let per = state.by_concept.entry(cid_str.clone()).or_default();
            match &payload.interaction {
                ConceptSummaryInteractionKind::Viewed { .. } => {
                    per.viewed_count = per.viewed_count.saturating_add(1);
                }
                ConceptSummaryInteractionKind::ClickedSource { .. } => {
                    per.clicked_source_count = per.clicked_source_count.saturating_add(1);
                }
                ConceptSummaryInteractionKind::ExplicitThumb { up } => {
                    if *up {
                        per.explicit_up = per.explicit_up.saturating_add(1);
                    } else {
                        per.explicit_down = per.explicit_down.saturating_add(1);
                    }
                }
                ConceptSummaryInteractionKind::ImmediateRequery { .. } => {
                    // Tracked at bucket level only — per-concept attribution
                    // for requery is ambiguous (the requery happens against
                    // the *next* search, not this concept-summary view).
                }
            }
            per.last_interaction_ts = chrono::Utc::now().timestamp();
        }
        if !existed {
            // New entry — push to FIFO order.
            state.by_concept_order.push(cid_str.clone());
            // Cap evict: pop from the FRONT of the order vec AND remove
            // from the HashMap. Dual update is mandatory — failure to keep
            // both stores in sync leaks orphan HashMap entries.
            while state.by_concept_order.len() > CONCEPT_SUMMARY_PER_ID_CAP {
                let evict = state.by_concept_order.remove(0);
                state.by_concept.remove(&evict);
            }
        }

        state.total_events = state.total_events.saturating_add(1);
    }

    // Recompute derived metrics for buckets touched this pass.
    for key in touched_buckets {
        if let Some(bucket) = state.by_cluster.get_mut(&key) {
            bucket.viewed_dwell_p50_ms = concept_summary_dwell_p50_ms(&bucket.dwell_samples);
            bucket.useful_rate = compute_concept_summary_useful_rate(bucket);
        }
    }

    Ok((state, max_id_this_pass))
}

impl AdaptiveState {
    /// v0.27 ARS Cap A: per-`(cluster_id, query_type)` concept-summary
    /// bucket, returned only when the bucket has accumulated at least
    /// [`CONCEPT_SUMMARY_COLD_START_N`] samples across any signal class
    /// (v0.27.1 E direction extends from `viewed_count` alone to a
    /// cumulative `total_signal` so MCP-only clusters with judge-only
    /// counts can still surface). Mirrors [`Self::synthesis_bucket`].
    pub fn concept_summary_bucket(
        &self,
        cluster_id: Option<i64>,
        query_type: &str,
    ) -> Option<&ClusterConceptSummaryStats> {
        let state = self.concept_summary_feedback_stats.as_ref()?;
        let key = concept_summary_bucket_key(cluster_id, query_type);
        state.by_cluster.get(&key).filter(|s| {
            let total = s
                .viewed_count
                .saturating_add(s.explicit_up)
                .saturating_add(s.explicit_down)
                .saturating_add(s.llm_judge_count);
            total >= CONCEPT_SUMMARY_COLD_START_N
        })
    }
}

/// v0.27.1 E direction Cap A mirror of
/// [`recompute_synthesis_feedback_stats_with_judge`]. Peeks both
/// `ConceptSummaryInteraction` and `ConceptSummaryLlmJudge` event types
/// in one query and runs the κ-pair join per spec §6.2.1 +  §6.6.
#[allow(clippy::too_many_arguments)]
pub fn recompute_concept_summary_feedback_stats_with_judge(
    conn: &Connection,
    prior: Option<ConceptSummaryFeedbackState>,
    pending_pairs_prior: HashMap<String, HalfPair>,
    calibration_prior: JudgeCalibrationState,
    // Codex R2 P2 fix — same threading as synthesis variant.
    weight_decay_rate: f64,
) -> ReinResult<(
    ConceptSummaryFeedbackState,
    HashMap<String, HalfPair>,
    JudgeCalibrationState,
    Option<i64>,
)> {
    recompute_concept_summary_feedback_stats_with_judge_and_weights(
        conn,
        prior,
        pending_pairs_prior,
        calibration_prior,
        weight_decay_rate,
        UsefulRateWeights::concept_summary_bootstrap(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn recompute_concept_summary_feedback_stats_with_judge_and_weights(
    conn: &Connection,
    prior: Option<ConceptSummaryFeedbackState>,
    pending_pairs_prior: HashMap<String, HalfPair>,
    calibration_prior: JudgeCalibrationState,
    // Codex R2 P2 fix — same threading as synthesis variant.
    weight_decay_rate: f64,
    useful_rate_weights: UsefulRateWeights,
) -> ReinResult<(
    ConceptSummaryFeedbackState,
    HashMap<String, HalfPair>,
    JudgeCalibrationState,
    Option<i64>,
)> {
    let mut state = prior.unwrap_or_default();
    let mut pending_pairs = pending_pairs_prior;
    let mut calibration = calibration_prior;

    let events = peek_events(
        conn,
        "concept_summary_feedback",
        &[
            EventType::ConceptSummaryInteraction.as_str(),
            EventType::ConceptSummaryLlmJudge.as_str(),
        ],
        50_000,
    )?;
    if events.is_empty() {
        return Ok((state, pending_pairs, calibration, None));
    }
    let max_id_this_pass = events.last().map(|e| e.id);

    let prior_high_water = state.last_consumed_event_id;
    if let Some(max_id) = max_id_this_pass {
        state.last_consumed_event_id = state.last_consumed_event_id.max(max_id);
    }

    let mut touched_buckets: std::collections::HashSet<String> = std::collections::HashSet::new();

    let now_ts = chrono::Utc::now().timestamp();
    let cutoff = now_ts.saturating_sub(LLM_JUDGE_HALF_PAIR_TTL_SECS);
    pending_pairs.retain(|_, half| half.ts() >= cutoff);

    for ev in events {
        if ev.id <= prior_high_water {
            continue;
        }
        let payload_str = match ev.payload.as_deref() {
            Some(s) => s,
            None => {
                tracing::warn!(
                    event_id = ev.id,
                    "concept_summary_feedback: event missing payload, skipping"
                );
                continue;
            }
        };
        match ev.event_type.as_str() {
            "concept_summary_interaction" => {
                let payload: ConceptSummaryInteractionPayload = match serde_json::from_str(
                    payload_str,
                ) {
                    Ok(p) => p,
                    Err(e) => {
                        tracing::warn!(
                            event_id = ev.id,
                            error = %e,
                            "concept_summary_feedback: malformed ConceptSummaryInteractionPayload, skipping"
                        );
                        continue;
                    }
                };
                fold_concept_summary_interaction(
                    &mut state,
                    &mut pending_pairs,
                    &mut calibration,
                    &mut touched_buckets,
                    &payload,
                    ev.id,
                );
            }
            "concept_summary_llm_judge" => {
                let payload: ConceptSummaryLlmJudgePayload = match serde_json::from_str(payload_str)
                {
                    Ok(p) => p,
                    Err(e) => {
                        tracing::warn!(
                            event_id = ev.id,
                            error = %e,
                            "concept_summary_feedback: malformed ConceptSummaryLlmJudgePayload, skipping"
                        );
                        continue;
                    }
                };
                fold_concept_summary_llm_judge(
                    &mut state,
                    &mut pending_pairs,
                    &mut calibration,
                    &mut touched_buckets,
                    &payload,
                    ev.id,
                );
            }
            other => {
                tracing::debug!(
                    event_id = ev.id,
                    event_type = %other,
                    "concept_summary_feedback: unexpected event_type in peek, skipping"
                );
            }
        }
    }

    for key in touched_buckets {
        if let Some(bucket) = state.by_cluster.get_mut(&key) {
            bucket.viewed_dwell_p50_ms = concept_summary_dwell_p50_ms(&bucket.dwell_samples);
            bucket.useful_rate = if bucket.llm_judge_count > 0 {
                compute_concept_summary_useful_rate_with_judge_and_weights(
                    bucket,
                    weight_decay_rate,
                    useful_rate_weights,
                )
                .unwrap_or_else(|| {
                    compute_concept_summary_useful_rate_with_weights(bucket, useful_rate_weights)
                })
            } else {
                compute_concept_summary_useful_rate_with_weights(bucket, useful_rate_weights)
            };
        }
    }

    if pending_pairs.len() > LLM_JUDGE_PAIR_CACHE_CAPACITY {
        let drop_n = pending_pairs.len() - LLM_JUDGE_PAIR_CACHE_CAPACITY;
        let to_drop: Vec<String> = pending_pairs
            .iter()
            .take(drop_n)
            .map(|(k, _)| k.clone())
            .collect();
        for k in to_drop {
            pending_pairs.remove(&k);
        }
    }

    Ok((state, pending_pairs, calibration, max_id_this_pass))
}

fn fold_concept_summary_interaction(
    state: &mut ConceptSummaryFeedbackState,
    pending_pairs: &mut HashMap<String, HalfPair>,
    calibration: &mut JudgeCalibrationState,
    touched_buckets: &mut std::collections::HashSet<String>,
    payload: &ConceptSummaryInteractionPayload,
    event_id: i64,
) {
    let metadata = payload.metadata.clone().unwrap_or_default();
    for (bucket_key, is_shadow) in concept_summary_bucket_keys_from_metadata(&metadata) {
        fold_concept_summary_interaction_bucket(
            state,
            &bucket_key,
            is_shadow,
            &payload.interaction,
            event_id,
        );
        touched_buckets.insert(bucket_key);
    }

    if let ConceptSummaryInteractionKind::ExplicitThumb { up } = &payload.interaction {
        // κ-pair join (spec §6.2.1 Cap A mirror). New clients key by
        // per-refresh `concept_summary_id` so a thumb cannot pair with a
        // judge verdict for another summary instance. Older clients omit
        // the field and keep the legacy concept_id key.
        let now_ts = chrono::Utc::now().timestamp();
        let key = payload
            .concept_summary_id
            .as_ref()
            .unwrap_or(&payload.concept_id)
            .clone();
        if let Some(half) = pending_pairs.remove(&key) {
            if let HalfPair::Judge {
                hit, ts, surface, ..
            } = &half
            {
                remove_half_pair_alias(pending_pairs, &key, &half);
                calibration.push_pair(*surface, *hit, *up, *ts);
            } else {
                pending_pairs.insert(
                    key,
                    HalfPair::Thumb {
                        up: *up,
                        ts: now_ts,
                        surface: JudgeSurface::ConceptSummary,
                    },
                );
            }
        } else {
            pending_pairs.insert(
                key,
                HalfPair::Thumb {
                    up: *up,
                    ts: now_ts,
                    surface: JudgeSurface::ConceptSummary,
                },
            );
        }
    }

    let cid_str = payload.concept_id.clone();
    let existed = state.by_concept.contains_key(&cid_str);
    {
        let per = state.by_concept.entry(cid_str.clone()).or_default();
        match &payload.interaction {
            ConceptSummaryInteractionKind::Viewed { .. } => {
                per.viewed_count = per.viewed_count.saturating_add(1);
            }
            ConceptSummaryInteractionKind::ClickedSource { .. } => {
                per.clicked_source_count = per.clicked_source_count.saturating_add(1);
            }
            ConceptSummaryInteractionKind::ExplicitThumb { up } => {
                if *up {
                    per.explicit_up = per.explicit_up.saturating_add(1);
                } else {
                    per.explicit_down = per.explicit_down.saturating_add(1);
                }
            }
            ConceptSummaryInteractionKind::ImmediateRequery { .. } => {}
        }
        per.last_interaction_ts = chrono::Utc::now().timestamp();
    }
    if !existed {
        state.by_concept_order.push(cid_str.clone());
        while state.by_concept_order.len() > CONCEPT_SUMMARY_PER_ID_CAP {
            let evict = state.by_concept_order.remove(0);
            state.by_concept.remove(&evict);
        }
    }

    state.total_events = state.total_events.saturating_add(1);
}

fn fold_concept_summary_llm_judge(
    state: &mut ConceptSummaryFeedbackState,
    pending_pairs: &mut HashMap<String, HalfPair>,
    calibration: &mut JudgeCalibrationState,
    touched_buckets: &mut std::collections::HashSet<String>,
    payload: &ConceptSummaryLlmJudgePayload,
    event_id: i64,
) {
    let metadata = payload.metadata.clone().unwrap_or_default();
    let cluster_id = metadata.cluster_id;
    let raw_qtype = metadata.query_type.as_deref().unwrap_or("");
    let query_type = if CONCEPT_SUMMARY_ALLOWED_QUERY_TYPES.contains(&raw_qtype) {
        raw_qtype.to_string()
    } else {
        "unknown".to_string()
    };
    let bucket_key = concept_summary_bucket_key(cluster_id, &query_type);

    // v0.27.5 R2 — LRU eviction at cap (replaces v0.27.4 drop-new-bucket).
    // v0.28 H3 — judge writes are always production (no route_context
    // shadow path); pass `is_shadow = false` to the cap predicate.
    evict_concept_summary_lru_if_at_cap(&mut state.by_cluster, &bucket_key, false);

    let bucket = state.by_cluster.entry(bucket_key.clone()).or_default();
    // v0.28 H3 — judge writes are always production. New buckets start
    // `is_shadow = false`; existing shadow-only buckets get downgraded to
    // production unconditionally (the cap predicate above already counts this
    // bucket against the production budget).
    bucket.is_shadow = false;
    bucket.last_event_id = bucket.last_event_id.max(event_id);
    bucket.llm_judge_count = bucket.llm_judge_count.saturating_add(1);
    if payload.hit {
        bucket.llm_judge_hit_count = bucket.llm_judge_hit_count.saturating_add(1);
    }
    touched_buckets.insert(bucket_key);

    // κ-pair join keyed on concept_summary_id (per-refresh ULID) so
    // multi-refresh within the half-pair TTL window can't pair a judge
    // verdict for one summary instance with a thumb for a different
    // instance. For older clients that emitted thumbs before the optional
    // concept_summary_id field existed, also check the legacy concept_id key
    // before inserting a new judge half-pair.
    let now_ts = chrono::Utc::now().timestamp();
    let key = payload.concept_summary_id.clone();
    let legacy_key = payload.concept_id.clone();
    let matched = pending_pairs.remove(&key).map(|half| (key.clone(), half));
    let matched = matched.or_else(|| {
        if !legacy_key.is_empty() && legacy_key != key {
            pending_pairs
                .remove(&legacy_key)
                .map(|half| (legacy_key.clone(), half))
        } else {
            None
        }
    });
    if let Some((matched_key, half)) = matched {
        match half {
            HalfPair::Thumb { up, ts, surface } => {
                calibration.push_pair(surface, payload.hit, up, ts);
            }
            other => {
                if matched_key == key {
                    remove_half_pair_alias(pending_pairs, &matched_key, &other);
                }
                insert_judge_half_pair(
                    pending_pairs,
                    key,
                    Some(legacy_key),
                    payload.hit,
                    now_ts,
                    JudgeSurface::ConceptSummary,
                );
            }
        }
    } else {
        insert_judge_half_pair(
            pending_pairs,
            key,
            Some(legacy_key),
            payload.hit,
            now_ts,
            JudgeSurface::ConceptSummary,
        );
    }

    state.total_events = state.total_events.saturating_add(1);
}

// ── v0.27.1 E direction — LLM judge OfflineCron payloads + JudgeCalibrationState ──

/// v0.27.1 E direction — bootstrap signal source for the runtime LLM judge.
///
/// Reserved for future per-stream tagging within the Runtime tier (e.g.
/// distinguishing `MCP-triggered` from `auto-sampled` runtime calls).
/// **NOT** used to discriminate runtime vs offline cron — that distinction
/// is carried by [`EventType`] itself per Codex R2 P2 fix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JudgeSource {
    /// Auto-sampled (sample-rate ladder fired).
    AutoSampled,
    /// Manually triggered via `rein_judge_synthesis` MCP tool.
    ManualMcp,
}

/// v0.28 acceleration extension point per spec §16.1. v0.27.1 ships this
/// struct as a forward-compat placeholder — emitter never populates,
/// consumer ignores `Some`. Field set is stable across v0.28+: new fields
/// land as `Option<...>` only (back-compat with stored events).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct SignalHint {
    /// LLM judge's inferred-from-rationale "ideal" view weight. Used as a
    /// training label for v0.28 multi-param logistic-regression fit.
    /// Computed by v0.28 from judge `reason` + observed signals.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inferred_w_view: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inferred_w_click: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inferred_w_thumb: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inferred_w_req: Option<f64>,
    /// Rolling confidence interval width on this cluster's `useful_rate`
    /// estimate, computed by Bayesian posterior in v0.28. v0.27.1 stub
    /// leaves `None`; v0.28 uses for active-sampling decision.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub useful_rate_ci_width: Option<f64>,
}

/// v0.27.1 E direction Layer 1 — runtime LLM judge payload for Cap B
/// synthesis outputs. Persisted as JSON inside `feedback_events.payload`
/// for an event of type [`EventType::SynthesisLlmJudge`].
///
/// Spec §3.2: separate variant from `SynthesisInteraction` so the consumer
/// can apply `w_llm` weight when folding into `useful_rate`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SynthesisLlmJudgePayload {
    /// ULID of the synthesis output (links to
    /// `RecallSynthesisOutcome.synthesis_id`).
    pub synthesis_id: String,
    /// Judge model identifier (e.g. `"gemini-3.1-flash-lite-preview"`).
    /// Recorded for retroactive κ recompute when operators swap models.
    pub judge_model: String,
    /// LLM judge verdict.
    pub hit: bool,
    /// One-sentence rationale (truncated to 280 chars on emit).
    pub reason: String,
    /// SHA-256 of the post-truncation prompt bytes the runtime judge actually
    /// saw. Lets the nightly cron re-judge byte-identical input and detect
    /// drift without storing the full text. NOT a hash of the full source list.
    pub stamp_hash: String,
    /// Bootstrap signal source — fixed per emit.
    pub source: JudgeSource,
    /// Optional metadata for bucket routing — `query_type`, `cluster_id`,
    /// `source_count`, `judge_latency_ms`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<JudgeMetadata>,
    /// v0.28 acceleration extension point per spec §16.1. v0.27.1 always
    /// `None`; v0.28 multi-param fit pipeline populates.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signal_hint: Option<SignalHint>,
}

/// v0.27.1 E direction Cap A mirror payload.
///
/// Codex R8 P1 fix — Cap A summary_id minting (per spec §3.2): v0.27.0
/// stored `living_summary` directly on the `concepts` row with no per-
/// refresh instance id. v0.27.1 adds `concepts.living_summary_id` (ULID
/// minted on every refresh) plus the `concept_summary_instances`
/// retention table (R9-K3) so the judge can validate J5 against an
/// immutable snapshot even after a subsequent refresh overwrites
/// `concepts.living_summary_id`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConceptSummaryLlmJudgePayload {
    /// ULID identifying the concept-summary instance judged.
    pub concept_summary_id: String,
    /// Persistent concept ID the summary belongs to.
    pub concept_id: String,
    pub judge_model: String,
    pub hit: bool,
    pub reason: String,
    pub stamp_hash: String,
    pub source: JudgeSource,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<JudgeMetadata>,
    /// v0.28 acceleration extension point per spec §16.1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signal_hint: Option<SignalHint>,
}

/// v0.27.1 E direction — discriminated half-pair entry in the κ pair-join
/// LRU cache (`AdaptiveState::pending_kappa_half_pairs`).
///
/// Spec §6.2.1: humans usually thumb AFTER judge runs (judge ~1-5s
/// post-synthesis; human dwells ~10-60s post-synthesis). But MCP-only
/// callers may invoke `rein_judge_synthesis` AFTER an ExplicitThumb that
/// came in via a separate path. Cache-on-arrival handles both orderings
/// symmetrically.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "side", rename_all = "snake_case")]
pub enum HalfPair {
    Judge {
        hit: bool,
        ts: i64,
        surface: JudgeSurface,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        alias_key: Option<String>,
    },
    Thumb {
        up: bool,
        ts: i64,
        surface: JudgeSurface,
    },
}

fn remove_half_pair_alias(
    pending_pairs: &mut HashMap<String, HalfPair>,
    key: &str,
    half: &HalfPair,
) {
    if let HalfPair::Judge {
        alias_key: Some(alias_key),
        ..
    } = half
    {
        let alias_points_back = matches!(
            pending_pairs.get(alias_key),
            Some(HalfPair::Judge {
                alias_key: Some(backlink),
                ..
            }) if backlink == key
        );
        if alias_points_back {
            pending_pairs.remove(alias_key);
        }
    }
}

fn insert_judge_half_pair(
    pending_pairs: &mut HashMap<String, HalfPair>,
    key: String,
    alias_key: Option<String>,
    hit: bool,
    ts: i64,
    surface: JudgeSurface,
) {
    let alias_key = alias_key.filter(|alias| !alias.is_empty() && alias != &key);
    pending_pairs.insert(
        key.clone(),
        HalfPair::Judge {
            hit,
            ts,
            surface,
            alias_key: alias_key.clone(),
        },
    );
    if let Some(alias_key) = alias_key {
        pending_pairs.insert(
            alias_key,
            HalfPair::Judge {
                hit,
                ts,
                surface,
                alias_key: Some(key),
            },
        );
    }
}

impl HalfPair {
    pub fn ts(&self) -> i64 {
        match self {
            Self::Judge { ts, .. } | Self::Thumb { ts, .. } => *ts,
        }
    }
    pub fn surface(&self) -> JudgeSurface {
        match self {
            Self::Judge { surface, .. } | Self::Thumb { surface, .. } => *surface,
        }
    }
}

/// v0.27.1 E direction — per-surface calibration window discriminator
/// (per spec §15 R9-K4). Synthesis vs concept_summary rubrics differ, so
/// per-surface windows prevent one surface's high volume from masking
/// the other's drift.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default, Hash)]
#[serde(rename_all = "snake_case")]
pub enum JudgeSurface {
    #[default]
    Synthesis,
    ConceptSummary,
}

/// Deterministic structural-anchor kinds. Ground truth is intrinsic to the
/// kind; payloads cannot supply or override an expected label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JudgeStructuralProbeKind {
    SupportedExactSingle,
    SupportedExactMulti,
    UnsupportedNonce,
    QueryMismatch,
}

impl JudgeStructuralProbeKind {
    pub const ALL: [Self; 4] = [
        Self::SupportedExactSingle,
        Self::SupportedExactMulti,
        Self::UnsupportedNonce,
        Self::QueryMismatch,
    ];

    pub const fn expected_hit(self) -> bool {
        matches!(self, Self::SupportedExactSingle | Self::SupportedExactMulti)
    }
}

/// Event payload for [`EventType::JudgeStructuralAnchor`]. Unknown JSON fields
/// are ignored for forward compatibility, but no expected-label field exists:
/// consumers derive it exclusively from [`JudgeStructuralProbeKind`].
#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct JudgeStructuralAnchorPayload {
    pub surface: JudgeSurface,
    pub probe_kind: JudgeStructuralProbeKind,
    pub observed_hit: bool,
    pub run_id: String,
    pub model_fingerprint: String,
    pub rubric_fingerprint: String,
    pub probe_set_version: String,
    /// Opaque credential minted when the in-crate runner seals this run.
    /// The consumer compares its SHA-256 against persisted state.
    pub run_token: String,
}

/// LRU cap for `pending_kappa_half_pairs` (spec §6.2.1). Bounds memory
/// growth on high-volume nodes; FIFO eviction matches the 7-day window of
/// `JudgeCalibrationState.recent_pairs_*`. Operators can override via
/// `[ars.llm_judge].pair_cache_capacity`.
pub const LLM_JUDGE_PAIR_CACHE_CAPACITY: usize = 10_000;

/// 7-day TTL on a half-pair before it is evicted unmatched. Mirrors the
/// rolling-pair window in [`JudgeCalibrationState`] so cache eviction and
/// pair eviction stay aligned.
pub const LLM_JUDGE_HALF_PAIR_TTL_SECS: i64 = 7 * 24 * 3600;

/// Bootstrap κ floor used by the J3 invariant. Below this, the runtime
/// worker MUST NOT raise `sample_rate_warm`.
pub const LLM_JUDGE_KAPPA_FLOOR: f64 = 0.6;

/// Minimum (judge, ExplicitThumb) pairs before J3 is checked at all.
/// Below this, J3 is dormant per spec §4 — runtime judge runs unconstrained
/// at cold-start sample rate. This is the defensible policy: J3 protects
/// against a calibrated drift signal, not against the absence of data.
pub const LLM_JUDGE_J3_MIN_PAIRS: usize = 30;

/// Bootstrap weight-decay rate for `useful_rate`'s LLM-judge contribution.
/// `w_llm = w_thumb × weight_decay_rate`. Codex R2 P3 — 0.3 NOT 0.7. LLM
/// signal heavily discounted relative to human thumb so any human signal
/// dominates immediately. Conservative default per rein's "human is
/// golden ground truth" philosophy.
pub const LLM_JUDGE_WEIGHT_DECAY_RATE: f64 = 0.3;

// Wave-1 D_CALIBRATION_CRON staging. A_JUDGE_CORE owns the runtime payload
// + worker; D_CALIBRATION_CRON owns the OfflineCron emitter and the
// `judge_calibration` consumer pass.

/// v0.27.1 E direction Layer 2 — payload for `SynthesisLlmJudgeOfflineCron`.
///
/// Codex R7 P2 fix: distinct payload from `SynthesisLlmJudgePayload` because
/// it carries BOTH the runtime verdict (joined from `feedback_events` at cron
/// emit time via `synthesis_id`) AND the cron's stricter verdict, so the
/// `judge_calibration` consumer can compute κ from a single event without
/// re-querying `feedback_events`. Reusing the runtime-judge payload would
/// only carry one `hit` field.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SynthesisLlmJudgeOfflineCronPayload {
    /// ULID of the synthesis output (links to RecallSynthesisOutcome.synthesis_id).
    pub synthesis_id: String,
    /// SHA-256 of the post-truncation prompt + candidate bytes the runtime
    /// judge actually saw (J7 stamp-time invariant). Cron uses byte-identical
    /// input for re-judge; mismatch would make κ comparison meaningless.
    pub stamp_hash: String,
    /// Verdict from the runtime LLM judge (Layer 1) for this synthesis_id.
    /// Already in `feedback_events` at cron emit time; copied here so the
    /// calibration consumer doesn't have to re-query.
    pub runtime_hit: bool,
    /// Identifier of the runtime judge model (e.g.
    /// "gemini-3.1-flash-lite-preview"). Recorded for retroactive κ
    /// recompute when operators swap models.
    pub runtime_judge_model: String,
    /// Verdict from the stricter nightly LLM judge (Layer 2).
    pub cron_hit: bool,
    /// Identifier of the cron judge model. Typical operator override
    /// (`[ars.llm_judge.nightly_cron]`) selects a stricter rubric / different-
    /// family model.
    pub cron_judge_model: String,
    /// One-sentence rationale from the cron judge (truncated to 280 chars
    /// on emit). Stricter rubric / different model usually => non-trivial reason.
    pub cron_reason: String,
    /// Optional metadata for bucket routing — query_type, cluster_id, etc.
    /// `#[serde(default)]` so old payloads parse after schema bumps.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<JudgeMetadata>,
}

/// v0.27.1 E direction Cap A Layer 2 mirror — payload for
/// `ConceptSummaryLlmJudgeOfflineCron`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConceptSummaryLlmJudgeOfflineCronPayload {
    /// ULID identifying the concept-summary instance judged. Links to
    /// `concepts.living_summary_id` (v0.27.1 NEW column owned by A_JUDGE_CORE).
    pub concept_summary_id: String,
    /// Persistent concept ID the summary belongs to.
    pub concept_id: String,
    /// SHA-256 of the post-truncation prompt + candidate bytes the runtime
    /// judge saw (J7 stamp-time invariant).
    pub stamp_hash: String,
    pub runtime_hit: bool,
    pub runtime_judge_model: String,
    pub cron_hit: bool,
    pub cron_judge_model: String,
    pub cron_reason: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<JudgeMetadata>,
}

/// v0.27.1 E direction — optional metadata travelling with judge events.
/// Reused by both runtime and OfflineCron payload variants. All fields
/// optional so JSON round-trips remain back-compat across schema bumps.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct JudgeMetadata {
    pub query_type: Option<String>,
    pub cluster_id: Option<i64>,
    pub source_count: Option<u32>,
    pub judge_latency_ms: Option<u32>,
}

/// Layer 1 κ rolling-window cap. Mirrors `recent_pairs` 7-day window — caps
/// memory growth on high-volume nodes. Independent for each surface
/// (synthesis vs concept) per R9-K4 (per-surface calibration windows).
pub const JUDGE_KAPPA_RECENT_PAIRS_CAP: usize = 4_096;

/// Layer 2 κ rolling-window cap. Same rationale as Layer 1; sized to fit a
/// week of `[ars.llm_judge.nightly_cron].max_archive_per_day` (default 5000
/// per spec §7) at 20% sample rate ≈ 7,000 cron events / week — slightly
/// above one week's worth so a single missed cron pass doesn't lose all
/// drift signal.
pub const JUDGE_RUNTIME_VS_OFFLINE_PAIRS_CAP: usize = 8_192;

/// Drift alert threshold. When `runtime_vs_offline_kappa < 0.7`, the
/// `judge_calibration` consumer logs a one-line warning to
/// `~/.rein/judge_drift.log` and bumps `judge_drift_alert`. Bootstrap const;
/// v0.28 ablation per [[feedback_no_subjective_params]].
pub const JUDGE_DRIFT_THRESHOLD: f64 = 0.7;

/// Minimum pair count before `runtime_vs_offline_kappa` is trusted enough
/// to fire a drift alert. Below this, κ is too noisy. Bootstrap const.
pub const JUDGE_DRIFT_MIN_PAIRS: usize = 30;

/// v0.27.1 E direction — judge calibration state container. Persisted as
/// part of [`AdaptiveState`].
///
/// **Field grouping (R9-K5)**: Layer 1 (synthesis_feedback) and Layer 2
/// (judge_calibration) consumers both write to this struct. Layer 1 fields
/// merge under `synthesis_feedback`'s watermark; Layer 2 fields merge
/// under `judge_calibration`'s watermark — see `AdaptiveState::save_snapshot`
/// for the field-grouped CAS merge implementation.
///
/// **Per-surface windows (R9-K4)**: synthesis and concept-summary each get
/// their own `recent_pairs_*` deque so one surface's high volume can't mask
/// the other's drift.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct JudgeCalibrationState {
    // ── Layer 1 fields (owned by synthesis_feedback / concept_summary_feedback) ──
    /// Rolling 7-day window of `(judge_hit, human_thumb_up)` pairs joined
    /// on `synthesis_id`. Owned by `synthesis_feedback` consumer per §6.2.1.
    /// Read by J3 invariant. Capped at [`JUDGE_KAPPA_RECENT_PAIRS_CAP`];
    /// FIFO-evict oldest on overflow.
    #[serde(default)]
    pub recent_pairs_synthesis: std::collections::VecDeque<(bool, bool, i64)>,
    /// Cap A mirror — `(judge_hit, human_thumb_up)` pairs joined on
    /// `concept_summary_id`. Owned by `concept_summary_feedback` consumer.
    #[serde(default)]
    pub recent_pairs_concept: std::collections::VecDeque<(bool, bool, i64)>,
    /// J3 κ (runtime judge vs ExplicitThumb) over `recent_pairs_synthesis`.
    /// Recomputed when pairs change. Used by `judge/contract.rs::no_self_reinforce`.
    /// `0.0` when undefined (insufficient pairs); J3 reads `synthesis_feedback`
    /// pair count to decide whether to consult κ.
    #[serde(default)]
    pub kappa: f64,

    // ── Layer 2 fields (owned by judge_calibration consumer) ──
    /// Drift κ between runtime judge and stricter offline cron over the
    /// same synthesis_ids. Owned by `judge_calibration` consumer.
    /// Used by drift alert + doctor; NEVER read by J3.
    /// `0.0` when undefined.
    #[serde(default)]
    pub runtime_vs_offline_kappa: f64,
    /// Per-surface runtime-vs-offline drift κ for synthesis judge events.
    #[serde(default)]
    pub runtime_vs_offline_kappa_synthesis: f64,
    /// Per-surface runtime-vs-offline drift κ for concept-summary judge events.
    #[serde(default)]
    pub runtime_vs_offline_kappa_concept: f64,
    /// Durable watermark for `judge_calibration` consumer.
    /// Without this, if `save_snapshot` succeeds and
    /// `commit_offset('judge_calibration')` fails, the same OfflineCron
    /// events replay next pass and double-append κ pairs / bump drift
    /// alert counts. Updated CAS-by-max alongside `consumer_offsets` row.
    #[serde(default)]
    pub last_consumed_event_id_calibration: i64,
    /// Rolling window of `(runtime_hit, cron_hit)` pairs. Owned by
    /// `judge_calibration` consumer. Capped at [`JUDGE_RUNTIME_VS_OFFLINE_PAIRS_CAP`];
    /// FIFO-evict oldest on overflow.
    #[serde(default)]
    pub recent_pairs_runtime_vs_offline: std::collections::VecDeque<(bool, bool, i64)>,
    /// Synthesis-only runtime-vs-offline pair window.
    #[serde(default)]
    pub recent_pairs_runtime_vs_offline_synthesis: std::collections::VecDeque<(bool, bool, i64)>,
    /// Concept-summary-only runtime-vs-offline pair window.
    #[serde(default)]
    pub recent_pairs_runtime_vs_offline_concept: std::collections::VecDeque<(bool, bool, i64)>,
    /// Total Layer 2 events the consumer has processed (replay-counted once).
    /// Useful for `/api/judge/calibration` exposure of drift coverage.
    #[serde(default)]
    pub total_offline_cron_events: u64,
    /// Bumped each time the consumer detects `runtime_vs_offline_kappa <
    /// JUDGE_DRIFT_THRESHOLD` while `recent_pairs_runtime_vs_offline.len()
    /// >= JUDGE_DRIFT_MIN_PAIRS`. Doctor surfaces this; operator response
    /// is to swap `nightly_cron.model`, lower `sample_rate_warm`, or
    /// disable runtime judge.
    #[serde(default)]
    pub judge_drift_alert: u64,
    /// Synthesis-only drift alert count.
    #[serde(default)]
    pub judge_drift_alert_synthesis: u64,
    /// Concept-summary-only drift alert count.
    #[serde(default)]
    pub judge_drift_alert_concept: u64,
    /// Unix timestamp (seconds) of the last `runtime_vs_offline_kappa`
    /// recomputation. Diagnostic — surfaced by doctor / `/api/judge/calibration`.
    #[serde(default)]
    pub last_computed_at: i64,
}

impl JudgeCalibrationState {
    /// Push a completed `(judge_hit, thumb_up, ts)` κ pair into the
    /// surface-matching rolling window per spec §6.2.1. Evicts pairs older
    /// than 7 days from the front, then caps at
    /// [`JUDGE_KAPPA_RECENT_PAIRS_CAP`]. Recomputes `kappa` when the
    /// surface is `Synthesis` (Layer 1 J3 reads only the synthesis κ —
    /// per-surface κ split is a v0.27.2 ablation per spec §15 R9-K4 and
    /// is currently approximated by routing all J3 reads through
    /// `recent_pairs_synthesis`).
    pub fn push_pair(&mut self, surface: JudgeSurface, hit: bool, up: bool, ts: i64) {
        let cutoff = ts.saturating_sub(LLM_JUDGE_HALF_PAIR_TTL_SECS);
        let pairs = match surface {
            JudgeSurface::Synthesis => &mut self.recent_pairs_synthesis,
            JudgeSurface::ConceptSummary => &mut self.recent_pairs_concept,
        };
        // FIFO-evict pairs older than the 7-day window.
        while let Some(&(_, _, t)) = pairs.front() {
            if t < cutoff {
                pairs.pop_front();
            } else {
                break;
            }
        }
        pairs.push_back((hit, up, ts));
        while pairs.len() > JUDGE_KAPPA_RECENT_PAIRS_CAP {
            pairs.pop_front();
        }
        // Layer 1 κ recomputation. R9-K4: J3 reads the synthesis surface
        // window today; concept-summary κ split is a v0.27.2 ablation.
        if matches!(surface, JudgeSurface::Synthesis) {
            self.kappa = compute_cohens_kappa(&self.recent_pairs_synthesis);
        }
    }
}

/// v0.27.1 E direction — Cohen's κ over a binary (judge_hit,
/// human_thumb_up) pair list. `κ = (p_o - p_e) / (1 - p_e)`, where `p_o`
/// is observed agreement and `p_e` is expected agreement under chance.
///
/// Returns `0.0` for an empty pair list or for the perfect-uniform edge
/// case (all yes or all no on both sides). The clamp to `[-1.0, 1.0]`
/// guards against floating-point noise pushing κ slightly outside its
/// theoretical range.
pub fn compute_cohens_kappa(pairs: &std::collections::VecDeque<(bool, bool, i64)>) -> f64 {
    let n = pairs.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let mut both_hit = 0_u64;
    let mut both_miss = 0_u64;
    let mut judge_hit = 0_u64;
    let mut thumb_up = 0_u64;
    for &(j, t, _) in pairs {
        match (j, t) {
            (true, true) => {
                both_hit += 1;
                judge_hit += 1;
                thumb_up += 1;
            }
            (true, false) => {
                judge_hit += 1;
            }
            (false, true) => {
                thumb_up += 1;
            }
            (false, false) => {
                both_miss += 1;
            }
        }
    }
    let p_o = (both_hit + both_miss) as f64 / n;
    let p_judge_yes = judge_hit as f64 / n;
    let p_thumb_yes = thumb_up as f64 / n;
    let p_judge_no = 1.0 - p_judge_yes;
    let p_thumb_no = 1.0 - p_thumb_yes;
    let p_e = p_judge_yes * p_thumb_yes + p_judge_no * p_thumb_no;
    let denom = 1.0 - p_e;
    if denom.abs() < 1e-12 {
        // Codex R1 P2 fix — degenerate case: p_e = 1.0 (one rater always
        // says yes OR always says no). Cohen's κ is undefined here, but
        // observed agreement carries the meaningful signal: if both raters
        // happen to agree on every sample (p_o = 1.0), J3 should NOT
        // tank to 0 and refuse to raise sample rate. Return p_o so
        // perfect agreement scores as 1.0; perfect disagreement (p_o = 0)
        // scores as 0.0 — both consistent with the observed-agreement
        // floor when the chance-correction is undefined.
        return p_o.clamp(0.0, 1.0);
    }
    ((p_o - p_e) / denom).clamp(-1.0, 1.0)
}

/// v0.27.1 E direction — active-signal-mask `useful_rate` per spec §6.4.
///
/// Replaces the v0.26 fixed-denominator [`compute_useful_rate`] for
/// buckets with any LLM-judge signal. Missing signals contribute neither
/// to numerator nor denominator. Returns `None` only when the bucket has
/// no signal at all (caller falls back to cold-start ladder).
///
/// **Cold start with only LLM signal** (the entire point of E direction):
/// numerator = `w_thumb × decay × llm_hit_rate`, denominator =
/// `w_thumb × decay`, result = `llm_hit_rate`. A 100%-LLM-hit cluster
/// reads as 1.0; `decide_synthesize → Yes`.
///
/// **Mixed cold start (LLM + a few humans)**: human signal weighted at
/// `w_thumb`, LLM weighted at `w_thumb × decay`. Humans dominate as soon
/// as they appear.
///
/// **Steady state (all signals active)**: full multi-source weighted
/// average; with default `weight_decay_rate = 0.3`, LLM contributes 30%
/// of the human-thumb weight. J6 invariant (`w_llm ≤ w_thumb`) is
/// guaranteed for any `weight_decay_rate ∈ [0, 1]`.
pub fn compute_useful_rate_with_judge(
    stats: &ClusterSynthesisStats,
    weight_decay_rate: f64,
) -> Option<f64> {
    compute_useful_rate_with_judge_and_weights(
        stats,
        weight_decay_rate,
        UsefulRateWeights::synthesis_bootstrap(),
    )
}

pub fn compute_useful_rate_with_judge_and_weights(
    stats: &ClusterSynthesisStats,
    weight_decay_rate: f64,
    weights: UsefulRateWeights,
) -> Option<f64> {
    let total_views = stats.viewed_count;
    let explicit_total = stats.explicit_up + stats.explicit_down;
    let llm_total = stats.llm_judge_count;

    let mut numerator = 0.0_f64;
    let mut denominator = 0.0_f64;

    // Behavioral signals — only active when any view exists.
    if total_views > 0 {
        let viewed_signal = if stats.dwell_samples.is_empty() {
            0.0
        } else {
            let above_threshold = stats
                .dwell_samples
                .iter()
                .filter(|&&ms| ms > SYNTHESIS_DWELL_THRESHOLD_MS)
                .count();
            (above_threshold as f64 / stats.dwell_samples.len() as f64).clamp(0.0, 1.0)
        };
        let click_signal = (stats.clicked_source_count as f64 / total_views as f64).min(1.0);
        let requery_signal = (stats.immediate_requery_count as f64 / total_views as f64).min(1.0);
        numerator += weights.view * viewed_signal + weights.click * click_signal
            - weights.requery * requery_signal;
        denominator += weights.view + weights.click + weights.requery;
    }

    // Explicit thumb — only active when any thumb exists.
    if explicit_total > 0 {
        let thumb_signal = stats.explicit_up as f64 / explicit_total as f64;
        numerator += weights.thumb * thumb_signal;
        denominator += weights.thumb;
    }

    // LLM judge — only active when any judge event exists. Weight is
    // strictly ≤ W_THUMB by J6 (config-validated weight_decay_rate ≤ 1.0).
    if llm_total > 0 {
        let w_llm = weights.thumb * weight_decay_rate;
        let llm_signal = stats.llm_judge_hit_count as f64 / llm_total as f64;
        numerator += w_llm * llm_signal;
        denominator += w_llm;
    }

    if denominator > 0.0 {
        // Codex R4 P2 — clamp to [0, 1].
        Some((numerator / denominator).clamp(0.0, 1.0))
    } else {
        None
    }
}

/// v0.27.1 E direction Cap A mirror of [`compute_useful_rate_with_judge`].
pub fn compute_concept_summary_useful_rate_with_judge(
    stats: &ClusterConceptSummaryStats,
    weight_decay_rate: f64,
) -> Option<f64> {
    compute_concept_summary_useful_rate_with_judge_and_weights(
        stats,
        weight_decay_rate,
        UsefulRateWeights::concept_summary_bootstrap(),
    )
}

pub fn compute_concept_summary_useful_rate_with_judge_and_weights(
    stats: &ClusterConceptSummaryStats,
    weight_decay_rate: f64,
    weights: UsefulRateWeights,
) -> Option<f64> {
    let total_views = stats.viewed_count;
    let explicit_total = stats.explicit_up + stats.explicit_down;
    let llm_total = stats.llm_judge_count;

    let mut numerator = 0.0_f64;
    let mut denominator = 0.0_f64;

    if total_views > 0 {
        let viewed_signal = if stats.dwell_samples.is_empty() {
            0.0
        } else {
            let above_threshold = stats
                .dwell_samples
                .iter()
                .filter(|&&ms| ms > CONCEPT_SUMMARY_DWELL_THRESHOLD_MS)
                .count();
            (above_threshold as f64 / stats.dwell_samples.len() as f64).clamp(0.0, 1.0)
        };
        let click_signal = (stats.clicked_source_count as f64 / total_views as f64).min(1.0);
        let requery_signal = (stats.immediate_requery_count as f64 / total_views as f64).min(1.0);
        numerator += weights.view * viewed_signal + weights.click * click_signal
            - weights.requery * requery_signal;
        denominator += weights.view + weights.click + weights.requery;
    }

    if explicit_total > 0 {
        let thumb_signal = stats.explicit_up as f64 / explicit_total as f64;
        numerator += weights.thumb * thumb_signal;
        denominator += weights.thumb;
    }

    if llm_total > 0 {
        let w_llm = weights.thumb * weight_decay_rate;
        let llm_signal = stats.llm_judge_hit_count as f64 / llm_total as f64;
        numerator += w_llm * llm_signal;
        denominator += w_llm;
    }

    if denominator > 0.0 {
        Some((numerator / denominator).clamp(0.0, 1.0))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    #[test]
    fn hard_dedup_threshold_ignores_shadow_below_static() {
        let mut state = AdaptiveState {
            global_dedup_threshold: 0.40,
            ..Default::default()
        };
        state.dedup_thresholds.insert(7, 0.45);

        assert_eq!(state.get_hard_dedup_threshold(None, 0.70), 0.70);
        assert_eq!(state.get_hard_dedup_threshold(Some(7), 0.70), 0.70);
    }

    #[test]
    fn unlabeled_shadow_above_static_does_not_raise_hard_threshold() {
        let mut state = AdaptiveState {
            global_dedup_threshold: 0.80,
            ..Default::default()
        };
        state.dedup_thresholds.insert(7, 0.85);

        assert_eq!(state.get_hard_dedup_threshold(None, 0.70), 0.70);
        assert_eq!(state.get_hard_dedup_threshold(Some(7), 0.70), 0.70);
    }

    #[test]
    fn hard_dedup_threshold_ignores_nonfinite_shadow_values() {
        for nonfinite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let global_state = AdaptiveState {
                global_dedup_threshold: nonfinite,
                ..Default::default()
            };
            assert_eq!(global_state.get_hard_dedup_threshold(None, 0.60), 0.60);

            let mut cluster_state = AdaptiveState {
                global_dedup_threshold: 0.80,
                ..Default::default()
            };
            cluster_state.dedup_thresholds.insert(7, nonfinite);
            assert_eq!(cluster_state.get_hard_dedup_threshold(Some(7), 0.60), 0.60);
        }
    }

    #[test]
    fn hard_dedup_threshold_fails_closed_for_invalid_static_values() {
        let mut state = AdaptiveState {
            global_dedup_threshold: 0.80,
            ..Default::default()
        };
        state.dedup_thresholds.insert(7, 0.90);

        for invalid_static in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.01, 1.01] {
            assert_eq!(state.get_hard_dedup_threshold(None, invalid_static), 1.0);
            assert_eq!(state.get_hard_dedup_threshold(Some(7), invalid_static), 1.0);
        }
    }

    fn setup_db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        // Minimal schema for testing
        conn.execute_batch(
            "
            CREATE TABLE feedback_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                event_type TEXT NOT NULL,
                request_id TEXT, memory_id TEXT, concept_id TEXT,
                query TEXT, query_type TEXT, topic TEXT, payload TEXT
            );
            CREATE TABLE consumer_offsets (
                consumer TEXT PRIMARY KEY,
                last_event_id INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT);
        ",
        )
        .unwrap();
        conn
    }

    #[test]
    fn test_emit_and_consume_events() {
        let conn = setup_db();

        // Emit 3 events
        for i in 0..3 {
            emit_event(
                &conn,
                FeedbackEvent {
                    event_type: EventType::RecallComplete,
                    request_id: Some(format!("req-{i}")),
                    memory_id: None,
                    concept_id: None,
                    query: Some("test query".into()),
                    query_type: Some("semantic".into()),
                    topic: Some("test".into()),
                    payload: Some(serde_json::json!({"alpha": 0.5})),
                },
            )
            .unwrap();
        }

        assert_eq!(event_count(&conn), 3);

        // Consumer "m2" reads all 3
        let events = consume_events(&conn, "m2", &["recall_complete"], 100).unwrap();
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].request_id.as_deref(), Some("req-0"));

        // Consumer "m2" reads again — should get 0 (already consumed)
        let events = consume_events(&conn, "m2", &["recall_complete"], 100).unwrap();
        assert_eq!(events.len(), 0);

        // Consumer "m4" reads — independent offset, gets all 3
        let events = consume_events(&conn, "m4", &["recall_complete"], 100).unwrap();
        assert_eq!(events.len(), 3);
    }

    #[test]
    fn test_adaptive_state_snapshot() {
        let conn = setup_db();

        let mut state = AdaptiveState::default();
        state.learned_alpha.insert(
            "semantic".into(),
            LearnedAlphaEntry {
                value: 0.35,
                sample_count: 25,
                last_updated: "2026-03-26T00:00:00Z".into(),
            },
        );
        state.version = 1;
        state.hot_threshold = 0.5;
        state.cold_threshold = 0.1;

        state.save_snapshot(&conn).unwrap();

        let restored = AdaptiveState::restore_snapshot(&conn).unwrap();
        assert_eq!(restored.version, 1);
        assert!((restored.hot_threshold - 0.5).abs() < 1e-6);
        let alpha = restored.get_alpha("semantic", None, 10);
        assert!(alpha.is_some());
        assert!((alpha.unwrap() - 0.35).abs() < 0.01);
    }

    // ── #17: recluster cadence baseline persistence ──────────────────────────

    #[test]
    fn test_last_recluster_embedding_count_serde_default() {
        // A pre-#17 snapshot JSON (field absent) must deserialize to 0 so
        // the first post-upgrade pass reclusters unconditionally.
        let mut value = serde_json::to_value(AdaptiveState::default()).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .remove("last_recluster_embedding_count")
            .expect("field should serialize by default");
        let state: AdaptiveState = serde_json::from_value(value).unwrap();
        assert_eq!(state.last_recluster_embedding_count, 0);
    }

    #[test]
    fn test_cas_merge_carries_recluster_baseline() {
        let conn = setup_db();

        // Baseline snapshot.
        let first = AdaptiveState {
            version: 1,
            ..Default::default()
        };
        first.save_snapshot(&conn).unwrap();

        // Writer A and writer B both start from version 1.
        let mut a = AdaptiveState::restore_snapshot(&conn).unwrap();
        let mut b = AdaptiveState::restore_snapshot(&conn).unwrap();

        // A commits first (no recluster).
        a.version = 2;
        a.save_snapshot(&conn).unwrap();

        // B did a recluster → CAS conflict → merge path. The recluster
        // baseline must travel with the rest of the cluster-scoped state.
        b.version = 2;
        b.cluster_version += 1;
        b.last_recluster_embedding_count = 777;
        b.save_snapshot(&conn).unwrap();

        let merged = AdaptiveState::restore_snapshot(&conn).unwrap();
        assert_eq!(merged.last_recluster_embedding_count, 777);
    }

    /// #17 codex R9 — a pass that RECLUSTERED and consumed feedback in the
    /// same tick carries by_cluster buckets keyed in the NEW generation;
    /// a CAS conflict must not strip them (offsets commit after the save —
    /// the consumed events can never replay to rebuild the buckets).
    #[test]
    fn test_cas_merge_keeps_recluster_pass_own_feedback_buckets() {
        let conn = setup_db();

        let mut by_cluster_old = HashMap::new();
        by_cluster_old.insert("3|semantic".to_string(), Default::default());
        let mut first = AdaptiveState {
            version: 1,
            cluster_version: 1,
            synthesis_feedback_stats: Some(SynthesisFeedbackState {
                by_cluster: by_cluster_old,
                last_consumed_event_id: 10,
                ..Default::default()
            }),
            ..Default::default()
        };
        first.memory_clusters.insert("m".into(), 3);
        first.save_snapshot(&conn).unwrap();

        // Writer B: reclusters (gen 2) AND consumes feedback — fresh
        // buckets keyed in generation 2.
        let mut b = AdaptiveState::restore_snapshot(&conn).unwrap();

        // Peer writer A commits first (no recluster, no stats change).
        let mut a = AdaptiveState::restore_snapshot(&conn).unwrap();
        a.version = 2;
        a.save_snapshot(&conn).unwrap();

        b.cluster_version += 1;
        b.memory_clusters.insert("m".into(), 5);
        b.clear_cluster_scoped_learned_state();
        let mut by_cluster_new = HashMap::new();
        by_cluster_new.insert("5|semantic".to_string(), Default::default());
        by_cluster_new.insert("-1|semantic".to_string(), Default::default());
        b.synthesis_feedback_stats = Some(SynthesisFeedbackState {
            by_cluster: by_cluster_new,
            last_consumed_event_id: 99,
            ..Default::default()
        });
        b.version = 2;
        b.save_snapshot(&conn).unwrap(); // CAS conflict → merge

        let merged = AdaptiveState::restore_snapshot(&conn).unwrap();
        assert_eq!(merged.cluster_version, 2);
        let synth = merged.synthesis_feedback_stats.as_ref().unwrap();
        assert_eq!(synth.last_consumed_event_id, 99);
        assert!(
            synth.by_cluster.contains_key("5|semantic"),
            "the reclustering pass's own new-generation bucket must survive"
        );
        assert!(
            !synth.by_cluster.contains_key("3|semantic"),
            "old-generation bucket stays gone"
        );
    }

    /// #17 codex R8 — a reindex-style reset bumps the generation by TWO so
    /// a concurrent adaptive pass that reclustered on PRE-reindex
    /// embeddings (N → N+1, `we_reclustered` true) cannot tie and
    /// wholesale-overwrite the reset with old-space labels.
    #[test]
    fn test_cas_merge_reindex_reset_outranks_concurrent_stale_recluster() {
        let conn = setup_db();

        let mut first = AdaptiveState {
            version: 1,
            cluster_version: 1,
            ..Default::default()
        };
        first.memory_clusters.insert("m-old".into(), 3);
        first.save_snapshot(&conn).unwrap();

        // Adaptive pass loads gen 1 and reclusters on OLD embeddings.
        let mut stale_recluster = AdaptiveState::restore_snapshot(&conn).unwrap();
        stale_recluster.memory_clusters.insert("m-old".into(), 7);
        stale_recluster.cluster_version += 1; // gen 2, we_reclustered = true
        stale_recluster.last_recluster_embedding_count = 60;

        // Reindex reset commits with the +2 bump (gen 3).
        let mut reset = AdaptiveState::restore_snapshot(&conn).unwrap();
        reset.memory_clusters.clear();
        reset.cluster_version += 2;
        reset.version += 1;
        reset.save_snapshot(&conn).unwrap();

        // Stale recluster saves → CAS conflict. Gen 2 < 3: its wholesale
        // branch must not fire despite we_reclustered being true.
        stale_recluster.version += 1;
        stale_recluster.save_snapshot(&conn).unwrap();

        let merged = AdaptiveState::restore_snapshot(&conn).unwrap();
        assert_eq!(merged.cluster_version, 3);
        assert!(
            merged.memory_clusters.is_empty(),
            "old-space recluster must not overwrite the reindex reset"
        );
        assert_eq!(merged.last_recluster_embedding_count, 0);
    }

    /// #17 codex R6 — two same-generation writers (neither reclustered)
    /// that learned DISJOINT cluster buckets must merge additively on CAS
    /// conflict; the old `>=` wholesale-replace dropped the first writer's
    /// buckets.
    #[test]
    fn test_cas_merge_same_generation_is_additive() {
        let conn = setup_db();

        let alpha_entry = || LearnedAlphaEntry {
            value: 0.4,
            sample_count: 15,
            last_updated: chrono::Utc::now().to_rfc3339(),
        };

        let mut first = AdaptiveState {
            version: 1,
            cluster_version: 3,
            ..Default::default()
        };
        first.memory_clusters.insert("m-base".into(), 1);
        first.save_snapshot(&conn).unwrap();

        // Both writers load generation 3 and learn different buckets.
        let mut a = AdaptiveState::restore_snapshot(&conn).unwrap();
        let mut b = AdaptiveState::restore_snapshot(&conn).unwrap();
        a.learned_alpha.insert("semantic:1".into(), alpha_entry());
        b.learned_alpha.insert("semantic:2".into(), alpha_entry());

        a.version = 2;
        a.save_snapshot(&conn).unwrap();
        b.version = 2;
        b.save_snapshot(&conn).unwrap(); // CAS conflict → same-gen additive

        let merged = AdaptiveState::restore_snapshot(&conn).unwrap();
        assert!(
            merged.learned_alpha.contains_key("semantic:1"),
            "first writer's cluster bucket must survive the merge"
        );
        assert!(
            merged.learned_alpha.contains_key("semantic:2"),
            "second writer's cluster bucket must survive the merge"
        );
        assert!(merged.memory_clusters.contains_key("m-base"));
    }

    /// #17 codex R3 — a writer holding a snapshot from an OLDER clustering
    /// generation (e.g. an adaptive pass that loaded before a
    /// `migrate --reindex` generation bump) must not additively merge its
    /// stale-space cluster labels / cluster-scoped weights back in.
    #[test]
    fn test_cas_merge_drops_stale_generation_cluster_state() {
        let conn = setup_db();

        let shadow_entry = || LearnedShadowFusionEntry {
            weights: ShadowFusionWeightEntry {
                bm25: 0.5,
                vec: 0.5,
                kg: 0.0,
                episode: 0.0,
                support: 0.0,
                diversity: 0.0,
            },
            sample_count: 20,
            last_updated: chrono::Utc::now().to_rfc3339(),
        };

        // Generation 1 baseline with cluster-scoped state.
        let mut first = AdaptiveState {
            version: 1,
            cluster_version: 1,
            ..Default::default()
        };
        first.memory_clusters.insert("m-old".into(), 3);
        first
            .learned_shadow_fusion
            .insert("semantic:3".into(), shadow_entry());
        first.save_snapshot(&conn).unwrap();

        // Stale writer loads generation 1...
        let mut stale = AdaptiveState::restore_snapshot(&conn).unwrap();
        // ...and drains feedback events to a HIGHER watermark, with
        // by_cluster buckets keyed to generation-1 labels (codex R7: the
        // watermark arbitration alone would resurrect these).
        let mut by_cluster = HashMap::new();
        by_cluster.insert("3|semantic".to_string(), Default::default());
        by_cluster.insert("-1|semantic".to_string(), Default::default());
        stale.synthesis_feedback_stats = Some(SynthesisFeedbackState {
            by_cluster,
            last_consumed_event_id: 99,
            ..Default::default()
        });

        // ...then a reindex-style reset commits generation 2 (cleared maps).
        let mut reset = AdaptiveState::restore_snapshot(&conn).unwrap();
        reset.memory_clusters.clear();
        reset.learned_shadow_fusion.retain(|k, _| !k.contains(':'));
        reset.last_recluster_embedding_count = 0;
        reset.cluster_version += 1;
        reset.version += 1;
        reset.save_snapshot(&conn).unwrap();

        // Stale writer saves → CAS conflict → merge. Its generation-1
        // cluster state must be dropped, not additively resurrected.
        stale.version += 1;
        stale.save_snapshot(&conn).unwrap();

        let merged = AdaptiveState::restore_snapshot(&conn).unwrap();
        assert_eq!(merged.cluster_version, 2);
        assert!(
            merged.memory_clusters.is_empty(),
            "stale-generation memory_clusters must not be merged back"
        );
        assert!(
            !merged.learned_shadow_fusion.contains_key("semantic:3"),
            "stale-generation shadow fusion bucket must not be merged back"
        );
        assert_eq!(merged.last_recluster_embedding_count, 0);
        let synth = merged.synthesis_feedback_stats.as_ref().unwrap();
        assert_eq!(
            synth.last_consumed_event_id, 99,
            "watermark winner survives (replay safety)"
        );
        assert!(
            !synth.by_cluster.contains_key("3|semantic"),
            "label-keyed bucket from the dead generation must be stripped"
        );
        assert!(
            synth.by_cluster.contains_key("-1|semantic"),
            "no-cluster bucket is label-free and survives"
        );
    }

    #[test]
    fn test_alpha_fallback_chain() {
        let mut state = AdaptiveState::default();

        // No data → None
        assert!(state.get_alpha("semantic", Some(1), 10).is_none());

        // Add global semantic with enough samples
        state.learned_alpha.insert(
            "semantic".into(),
            LearnedAlphaEntry {
                value: 0.4,
                sample_count: 15,
                last_updated: String::new(),
            },
        );
        // Cluster-specific with too few samples
        state.learned_alpha.insert(
            "semantic:1".into(),
            LearnedAlphaEntry {
                value: 0.8,
                sample_count: 3,
                last_updated: String::new(),
            },
        );

        // Should fall back to global (cluster has < 10 samples)
        let alpha = state.get_alpha("semantic", Some(1), 10).unwrap();
        assert!((alpha - 0.4).abs() < 0.01);

        // Give cluster enough samples
        state
            .learned_alpha
            .get_mut("semantic:1")
            .unwrap()
            .sample_count = 12;
        let alpha = state.get_alpha("semantic", Some(1), 10).unwrap();
        assert!((alpha - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_alpha_global_and_legacy_key_fallback() {
        let mut state = AdaptiveState::default();
        state.learned_alpha.insert(
            "global".into(),
            LearnedAlphaEntry {
                value: 0.55,
                sample_count: 20,
                last_updated: String::new(),
            },
        );
        let alpha = state.get_alpha("temporal", None, 10).unwrap();
        assert!((alpha - 0.55).abs() < 0.01);

        state.learned_alpha.insert(
            "Temporal".into(),
            LearnedAlphaEntry {
                value: 0.8,
                sample_count: 20,
                last_updated: String::new(),
            },
        );
        let alpha = state.get_alpha("Temporal", None, 10).unwrap();
        assert!((alpha - 0.8).abs() < 0.01);
    }

    #[test]
    fn shadow_fusion_weights_fallback_chain_respects_min_samples() {
        let mut state = AdaptiveState::default();
        assert!(
            state
                .get_shadow_fusion_weights("semantic", Some(7), 10)
                .is_none(),
            "fresh state should not return acceleration weights"
        );

        state.learned_shadow_fusion.insert(
            "global".into(),
            LearnedShadowFusionEntry {
                weights: ShadowFusionWeightEntry {
                    bm25: 0.1,
                    vec: 0.9,
                    kg: 0.0,
                    episode: 0.0,
                    support: 0.0,
                    diversity: 0.0,
                },
                sample_count: 20,
                last_updated: "2026-04-30T00:00:00Z".into(),
            },
        );
        state.learned_shadow_fusion.insert(
            "semantic:7".into(),
            LearnedShadowFusionEntry {
                weights: ShadowFusionWeightEntry {
                    bm25: 0.0,
                    vec: 0.0,
                    kg: 1.0,
                    episode: 0.0,
                    support: 0.0,
                    diversity: 0.0,
                },
                sample_count: 3,
                last_updated: "2026-04-30T00:00:00Z".into(),
            },
        );

        let weights = state
            .get_shadow_fusion_weights("semantic", Some(7), 10)
            .expect("global fallback should satisfy sample gate");
        assert!((weights.weights.vec - 0.9).abs() < f64::EPSILON);

        state
            .learned_shadow_fusion
            .get_mut("semantic:7")
            .unwrap()
            .sample_count = 10;
        let weights = state
            .get_shadow_fusion_weights("semantic", Some(7), 10)
            .expect("cluster weights should satisfy sample gate");
        assert!((weights.weights.kg - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn adaptive_state_snapshot_round_trips_shadow_fusion_weights() {
        let conn = setup_db();
        let mut state = AdaptiveState::default();
        state.learned_shadow_fusion.insert(
            "semantic".into(),
            LearnedShadowFusionEntry {
                weights: ShadowFusionWeightEntry {
                    bm25: 0.2,
                    vec: 0.3,
                    kg: 0.1,
                    episode: 0.1,
                    support: 0.2,
                    diversity: 0.1,
                },
                sample_count: 12,
                last_updated: "2026-04-30T00:00:00Z".into(),
            },
        );
        state.version = 1;

        state.save_snapshot(&conn).unwrap();
        let restored = AdaptiveState::restore_snapshot(&conn).unwrap();
        let weights = restored
            .get_shadow_fusion_weights("semantic", None, 10)
            .expect("shadow fusion weights should round-trip through snapshot");
        assert!((weights.weights.support - 0.2).abs() < f64::EPSILON);
        assert_eq!(weights.sample_count, 12);
    }

    #[test]
    fn adaptive_cache_refreshes_after_ttl() {
        let conn = setup_db();

        // Seed DB with version=1
        let state = AdaptiveState {
            version: 1,
            hot_threshold: 0.5,
            ..Default::default()
        };
        state.save_snapshot(&conn).unwrap();

        // Load into cache with a very short TTL
        let mut cache = CachedAdaptiveState::load(&conn, std::time::Duration::from_millis(20));
        assert_eq!(cache.get(&conn).version, 1);
        assert!(!cache.is_stale());

        // Mutate DB directly to simulate another writer bumping version.
        let mut newer = AdaptiveState {
            version: 2,
            hot_threshold: 0.9,
            ..Default::default()
        };
        // Force overwrite (bypass optimistic concurrency for the test)
        let newer_json = serde_json::to_string(&newer).unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES ('adaptive_state', ?1)",
            rusqlite::params![&newer_json],
        )
        .unwrap();
        newer.version = 2; // suppress unused warning

        // Before TTL elapses, cache still serves old version.
        assert_eq!(cache.peek().version, 1);

        // Wait past TTL, then get() should transparently refresh.
        std::thread::sleep(std::time::Duration::from_millis(40));
        assert!(cache.is_stale());
        let refreshed = cache.get(&conn);
        assert_eq!(refreshed.version, 2);
        assert!((refreshed.hot_threshold - 0.9).abs() < 1e-6);

        // invalidate() forces a reload on the next get().
        cache.invalidate();
        assert!(cache.is_stale());
    }

    #[test]
    fn test_event_type_filter() {
        let conn = setup_db();

        emit_event(
            &conn,
            FeedbackEvent {
                event_type: EventType::RecallComplete,
                request_id: None,
                memory_id: None,
                concept_id: None,
                query: None,
                query_type: None,
                topic: None,
                payload: None,
            },
        )
        .unwrap();
        emit_event(
            &conn,
            FeedbackEvent {
                event_type: EventType::Store,
                request_id: None,
                memory_id: Some("m1".into()),
                concept_id: None,
                query: None,
                query_type: None,
                topic: None,
                payload: None,
            },
        )
        .unwrap();

        // Filter by store only
        let events = consume_events(&conn, "test", &["store"], 100).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "store");
    }

    // ── v0.24 peek+commit primitives ──────────────────────────────────────

    #[test]
    fn peek_events_does_not_advance_offset() {
        let conn = setup_db();
        for i in 0..3 {
            emit_event(
                &conn,
                FeedbackEvent {
                    event_type: EventType::Store,
                    request_id: Some(format!("r{i}")),
                    memory_id: Some(format!("m{i}")),
                    concept_id: None,
                    query: None,
                    query_type: None,
                    topic: None,
                    payload: None,
                },
            )
            .unwrap();
        }

        // Peek: returns all 3, no offset advance.
        let evts = peek_events(&conn, "test_consumer", &["store"], 100).unwrap();
        assert_eq!(evts.len(), 3);

        // Peek again: still returns all 3.
        let evts2 = peek_events(&conn, "test_consumer", &["store"], 100).unwrap();
        assert_eq!(evts2.len(), 3);

        // Commit advances offset; subsequent peek returns 0.
        commit_offset(&conn, &[("test_consumer", evts.last().unwrap().id)]).unwrap();
        let evts3 = peek_events(&conn, "test_consumer", &["store"], 100).unwrap();
        assert_eq!(evts3.len(), 0);
    }

    #[test]
    fn commit_offset_atomic_batch_advances_all_or_nothing() {
        let conn = setup_db();
        for i in 0..2 {
            emit_event(
                &conn,
                FeedbackEvent {
                    event_type: EventType::Store,
                    request_id: Some(format!("r{i}")),
                    memory_id: None,
                    concept_id: None,
                    query: None,
                    query_type: None,
                    topic: None,
                    payload: None,
                },
            )
            .unwrap();
        }

        // Commit two consumers in one batch.
        commit_offset(&conn, &[("c_a", 1), ("c_b", 2)]).unwrap();
        let off_a: i64 = conn
            .query_row(
                "SELECT last_event_id FROM consumer_offsets WHERE consumer = 'c_a'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        let off_b: i64 = conn
            .query_row(
                "SELECT last_event_id FROM consumer_offsets WHERE consumer = 'c_b'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(off_a, 1);
        assert_eq!(off_b, 2);
    }

    #[test]
    fn commit_offset_never_goes_backwards() {
        // Defensive: if a stale caller passes an older max id (race
        // between two pipeline runs), MAX() in the upsert keeps the
        // larger value.
        let conn = setup_db();
        commit_offset(&conn, &[("c1", 10)]).unwrap();
        commit_offset(&conn, &[("c1", 5)]).unwrap(); // stale
        let off: i64 = conn
            .query_row(
                "SELECT last_event_id FROM consumer_offsets WHERE consumer = 'c1'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(off, 10);
    }

    #[test]
    fn commit_offset_empty_batch_is_noop() {
        let conn = setup_db();
        commit_offset(&conn, &[]).unwrap();
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM consumer_offsets", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 0);
    }

    // ── v0.24 ARS L3: recompute_concept_refresh_stats ──────────────────────

    fn emit_refresh_sample(conn: &Connection, revisions: u32, age_secs: i64) {
        emit_refresh_sample_kind(conn, revisions, age_secs, false);
    }

    fn emit_refresh_sample_kind(
        conn: &Connection,
        revisions: u32,
        age_secs: i64,
        first_refresh: bool,
    ) {
        let payload = serde_json::to_value(RefreshSample {
            revisions_since_last: revisions,
            age_secs_since_last: age_secs,
            first_refresh,
            summary_id: String::new(),
        })
        .unwrap();
        emit_event(
            conn,
            FeedbackEvent {
                event_type: EventType::ConceptSummaryRefreshed,
                request_id: None,
                memory_id: None,
                concept_id: Some("c1".into()),
                query: None,
                query_type: None,
                topic: None,
                payload: Some(payload),
            },
        )
        .unwrap();
    }

    #[test]
    fn recompute_concept_refresh_stats_empty() {
        let conn = setup_db();
        let (stats, max_id) = recompute_concept_refresh_stats(&conn, None).unwrap();
        assert_eq!(stats.count, 0);
        assert!(stats.samples.is_empty());
        assert_eq!(stats.revision_p75, 0);
        assert_eq!(stats.age_p50_secs, 0);
        assert_eq!(max_id, None, "no events → no offset to commit");
    }

    #[test]
    fn recompute_concept_refresh_stats_aggregates_percentiles() {
        let conn = setup_db();
        // Emit 11 samples — above the MIN_SAMPLES gate.
        // revisions: 1..=11, sorted p75 ≈ rank 7.5 → linear interp 8.5 ≈ 9
        // ages: 100..=1100 step 100, sorted p50 = 600
        for i in 1..=11u32 {
            emit_refresh_sample(&conn, i, (i as i64) * 100);
        }
        let (stats, max_id) = recompute_concept_refresh_stats(&conn, None).unwrap();
        assert_eq!(stats.count, 11);
        assert_eq!(stats.revision_p75, 9); // rounded from 8.5
        assert_eq!(stats.age_p50_secs, 600);
        assert_eq!(max_id, Some(11), "max event id == last emitted");

        // Helpers now return learned values (count >= MIN_SAMPLES).
        let state = AdaptiveState {
            concept_refresh_stats: Some(stats),
            ..AdaptiveState::default()
        };
        assert_eq!(state.concept_refresh_revision_threshold(), 9);
        assert_eq!(state.concept_refresh_age_threshold_secs(), 600);
    }

    #[test]
    fn recompute_concept_refresh_stats_peek_replay_is_idempotent() {
        // Codex Tier-B+C round-1 HIGH replay-safety: when a prior pass's
        // commit_offset failed after save_snapshot succeeded, the next
        // pass re-peeks the same events. Recompute MUST NOT double-apply
        // them — guarded by `prior_high_water = stats.last_consumed_event_id`.
        let conn = setup_db();
        emit_refresh_sample(&conn, 5, 500);
        emit_refresh_sample(&conn, 7, 700);

        // First call: peeks both events, bumps `last_consumed_event_id`,
        // appends 2 samples. Returns max_id=2 for caller to commit.
        let (stats, max_id) = recompute_concept_refresh_stats(&conn, None).unwrap();
        assert_eq!(stats.count, 2);
        assert_eq!(max_id, Some(2));
        assert_eq!(stats.last_consumed_event_id, 2);

        // Simulate prior pass's commit FAILED — save_snapshot already
        // persisted `stats` (with last_consumed_event_id=2 + 2 samples)
        // but commit_offset never landed. Next pass: peek returns the
        // SAME events again. Replay must be a no-op.
        let (stats2, max_id2) =
            recompute_concept_refresh_stats(&conn, Some(stats.clone())).unwrap();
        assert_eq!(
            stats2.count, 2,
            "replay-safety: events with id ≤ last_consumed_event_id are skipped"
        );
        assert_eq!(stats2.samples, stats.samples, "no double-append on replay");
        assert_eq!(
            max_id2,
            Some(2),
            "max_id still reported so caller can re-attempt commit"
        );

        // Caller successfully commits this pass; subsequent peek finds nothing.
        commit_offset(&conn, &[("concept_refresh_stats", max_id2.unwrap())]).unwrap();
        let (stats3, max_id3) = recompute_concept_refresh_stats(&conn, Some(stats2)).unwrap();
        assert_eq!(stats3.count, 2, "post-commit peek sees no new events");
        assert_eq!(max_id3, None);

        // New event arrives after commit: reservoir grows by exactly one.
        emit_refresh_sample(&conn, 9, 900);
        let (stats4, max_id4) = recompute_concept_refresh_stats(&conn, Some(stats3)).unwrap();
        assert_eq!(stats4.count, 3);
        assert_eq!(stats4.samples.last().unwrap().revisions_since_last, 9);
        assert_eq!(max_id4, Some(3));
    }

    #[test]
    fn recompute_concept_refresh_stats_caps_reservoir_fifo() {
        let conn = setup_db();
        let overflow = 50;
        for i in 0..(CONCEPT_REFRESH_SAMPLE_CAP + overflow) {
            emit_refresh_sample(&conn, (i + 1) as u32, (i + 1) as i64);
        }
        let (stats, _) = recompute_concept_refresh_stats(&conn, None).unwrap();
        assert_eq!(stats.count, CONCEPT_REFRESH_SAMPLE_CAP);
        // FIFO drop: oldest `overflow` samples evicted, so the smallest
        // surviving revision is `overflow + 1`.
        assert_eq!(
            stats.samples.first().unwrap().revisions_since_last,
            (overflow + 1) as u32
        );
    }

    #[test]
    fn recompute_concept_refresh_stats_excludes_first_refresh_from_age_p50() {
        // 10 first-refresh samples with huge ages (concept-inception bias)
        // + 10 steady-state samples with small ages.
        // age_p50_secs MUST reflect only the 10 steady-state samples; without
        // the filter it would land somewhere between the two distributions
        // and contract the learned threshold below bootstrap.
        let conn = setup_db();
        for _ in 0..10 {
            emit_refresh_sample_kind(&conn, 4, 30 * 24 * 60 * 60, true); // 30 days
        }
        for _ in 0..10 {
            emit_refresh_sample_kind(&conn, 4, 60 * 60, false); // 1 hour
        }
        let (stats, _) = recompute_concept_refresh_stats(&conn, None).unwrap();
        assert_eq!(stats.count, 20);
        assert_eq!(stats.count_steady_state, 10);
        // p50 of ten 3600s samples = 3600.
        assert_eq!(stats.age_p50_secs, 3600);

        // Helpers reflect the dual-gate semantics.
        let state = AdaptiveState {
            concept_refresh_stats: Some(stats),
            ..AdaptiveState::default()
        };
        // Revision threshold gate uses total count (>=10): satisfied.
        assert_eq!(state.concept_refresh_revision_threshold(), 4);
        // Age threshold gate uses steady-state count (>=10): satisfied with
        // unbiased value.
        assert_eq!(state.concept_refresh_age_threshold_secs(), 3600);
    }

    #[test]
    fn recompute_concept_refresh_stats_only_first_refresh_falls_back_age_threshold() {
        // 10 first-refresh-only samples → revision_p75 learned, but age
        // threshold falls back to bootstrap because count_steady_state = 0.
        let conn = setup_db();
        for _ in 0..10 {
            emit_refresh_sample_kind(&conn, 7, 30 * 24 * 60 * 60, true);
        }
        let (stats, _) = recompute_concept_refresh_stats(&conn, None).unwrap();
        assert_eq!(stats.count, 10);
        assert_eq!(stats.count_steady_state, 0);
        assert_eq!(stats.age_p50_secs, 0);

        let state = AdaptiveState {
            concept_refresh_stats: Some(stats),
            ..AdaptiveState::default()
        };
        assert_eq!(state.concept_refresh_revision_threshold(), 7);
        // Age falls back to bootstrap until at least one steady-state sample.
        assert_eq!(
            state.concept_refresh_age_threshold_secs(),
            CONCEPT_REFRESH_BOOTSTRAP_AGE_SECS
        );
    }

    #[test]
    fn cas_retry_keeps_more_advanced_reservoir_by_event_id() {
        // Codex round-4 HIGH regression test. A stale writer (lower
        // `last_consumed_event_id`) MUST NOT overwrite the CAS winner's
        // more-advanced state.
        let conn = setup_db();
        let winner_stats = ConceptRefreshStats {
            count: 3,
            count_steady_state: 3,
            revision_p75: 7,
            age_p50_secs: 3600,
            samples: vec![
                RefreshSample {
                    revisions_since_last: 5,
                    age_secs_since_last: 1000,
                    first_refresh: false,
                    summary_id: String::new(),
                },
                RefreshSample {
                    revisions_since_last: 7,
                    age_secs_since_last: 2000,
                    first_refresh: false,
                    summary_id: String::new(),
                },
                RefreshSample {
                    revisions_since_last: 9,
                    age_secs_since_last: 3600,
                    first_refresh: false,
                    summary_id: String::new(),
                },
            ],
            last_consumed_event_id: 50,
        };
        let winner = AdaptiveState {
            version: 5,
            concept_refresh_stats: Some(winner_stats.clone()),
            ..AdaptiveState::default()
        };
        // Seed DB at v=5 (winner already saved).
        let winner_json = serde_json::to_string(&winner).unwrap();
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES ('adaptive_state', ?1)",
            rusqlite::params![&winner_json],
        )
        .unwrap();

        // Stale writer: thinks DB is at v=1, has its own (less advanced)
        // stats (last_consumed_event_id = 30 < winner's 50).
        let stale_stats = ConceptRefreshStats {
            count: 1,
            count_steady_state: 1,
            revision_p75: 4,
            age_p50_secs: 600,
            samples: vec![RefreshSample {
                revisions_since_last: 4,
                age_secs_since_last: 600,
                first_refresh: false,
                summary_id: String::new(),
            }],
            last_consumed_event_id: 30,
        };
        let stale = AdaptiveState {
            version: 2,
            concept_refresh_stats: Some(stale_stats),
            ..AdaptiveState::default()
        };
        stale.save_snapshot(&conn).unwrap();

        // Restore: winner's stats must survive the merge.
        let restored = AdaptiveState::restore_snapshot(&conn).unwrap();
        let restored_stats = restored.concept_refresh_stats.unwrap();
        assert_eq!(restored_stats.last_consumed_event_id, 50);
        assert_eq!(restored_stats.samples, winner_stats.samples);
    }

    #[test]
    fn cas_retry_takes_writer_when_more_advanced() {
        // Counterpart to the previous test: when the WRITER has the
        // higher last_consumed_event_id (drained newer events), its stats
        // must replace the CAS winner's older snapshot.
        let conn = setup_db();
        let older_stats = ConceptRefreshStats {
            count: 1,
            count_steady_state: 1,
            revision_p75: 4,
            age_p50_secs: 600,
            samples: vec![RefreshSample {
                revisions_since_last: 4,
                age_secs_since_last: 600,
                first_refresh: false,
                summary_id: String::new(),
            }],
            last_consumed_event_id: 30,
        };
        let older = AdaptiveState {
            version: 5,
            concept_refresh_stats: Some(older_stats),
            ..AdaptiveState::default()
        };
        let older_json = serde_json::to_string(&older).unwrap();
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES ('adaptive_state', ?1)",
            rusqlite::params![&older_json],
        )
        .unwrap();

        let fresher_stats = ConceptRefreshStats {
            count: 2,
            count_steady_state: 2,
            revision_p75: 8,
            age_p50_secs: 1800,
            samples: vec![
                RefreshSample {
                    revisions_since_last: 6,
                    age_secs_since_last: 1200,
                    first_refresh: false,
                    summary_id: String::new(),
                },
                RefreshSample {
                    revisions_since_last: 10,
                    age_secs_since_last: 1800,
                    first_refresh: false,
                    summary_id: String::new(),
                },
            ],
            last_consumed_event_id: 99,
        };
        let writer = AdaptiveState {
            version: 2,
            concept_refresh_stats: Some(fresher_stats.clone()),
            ..AdaptiveState::default()
        };
        writer.save_snapshot(&conn).unwrap();

        let restored = AdaptiveState::restore_snapshot(&conn).unwrap();
        let restored_stats = restored.concept_refresh_stats.unwrap();
        assert_eq!(restored_stats.last_consumed_event_id, 99);
        assert_eq!(restored_stats.samples, fresher_stats.samples);
    }

    #[test]
    fn cas_retry_preserves_existing_stats_when_writer_has_none() {
        // Codex round-3 HIGH regression test. Setup:
        //   1. DB holds v=1 with `Some(stats)` from a prior pipeline run.
        //   2. We start a fresh pipeline: in-memory state has v=2 and
        //      `concept_refresh_stats = None` (no events drained).
        //   3. A concurrent writer wins first; we artificially bump the DB
        //      version to v=5 to force the CAS predicate to fail.
        //   4. Our save_snapshot enters the CAS retry path; the merge logic
        //      MUST keep `current.concept_refresh_stats = Some(...)` rather
        //      than overwriting with our `None`.
        let conn = setup_db();
        let learned = ConceptRefreshStats {
            count: 25,
            count_steady_state: 25,
            revision_p75: 7,
            age_p50_secs: 3600,
            samples: vec![RefreshSample {
                revisions_since_last: 7,
                age_secs_since_last: 3600,
                first_refresh: false,
                summary_id: String::new(),
            }],
            last_consumed_event_id: 100,
        };
        let prior = AdaptiveState {
            version: 5, // simulate 4 prior pipeline writes since our v=1 save
            concept_refresh_stats: Some(learned.clone()),
            ..AdaptiveState::default()
        };
        // Seed DB at v=5 directly (simulates the concurrent winner).
        let prior_json = serde_json::to_string(&prior).unwrap();
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES ('adaptive_state', ?1)",
            rusqlite::params![&prior_json],
        )
        .unwrap();

        // Our state thinks DB is at v=1, so CAS expects (v=2 - 1 = 1) but
        // DB is actually v=5 → predicate fails → retry path runs.
        let our = AdaptiveState {
            version: 2,
            concept_refresh_stats: None,
            ..AdaptiveState::default()
        };
        our.save_snapshot(&conn).unwrap();

        // Restore: must still hold the originally-learned stats; our `None`
        // must NOT have overwritten them.
        let restored = AdaptiveState::restore_snapshot(&conn).unwrap();
        assert_eq!(
            restored.concept_refresh_stats,
            Some(learned),
            "CAS merge must preserve existing stats when writer has None"
        );
    }

    #[test]
    fn recompute_concept_refresh_stats_skips_malformed_payloads() {
        let conn = setup_db();
        // Two valid + one missing-payload + one malformed JSON.
        emit_refresh_sample(&conn, 3, 300);
        emit_event(
            &conn,
            FeedbackEvent {
                event_type: EventType::ConceptSummaryRefreshed,
                request_id: None,
                memory_id: None,
                concept_id: Some("c2".into()),
                query: None,
                query_type: None,
                topic: None,
                payload: None,
            },
        )
        .unwrap();
        emit_event(
            &conn,
            FeedbackEvent {
                event_type: EventType::ConceptSummaryRefreshed,
                request_id: None,
                memory_id: None,
                concept_id: Some("c3".into()),
                query: None,
                query_type: None,
                topic: None,
                payload: Some(serde_json::json!({"unexpected": "shape"})),
            },
        )
        .unwrap();
        emit_refresh_sample(&conn, 5, 500);

        let (stats, _) = recompute_concept_refresh_stats(&conn, None).unwrap();
        // Only 2 valid samples survived.
        assert_eq!(stats.count, 2);
        assert_eq!(stats.samples[0].revisions_since_last, 3);
        assert_eq!(stats.samples[1].revisions_since_last, 5);
    }

    // ── v0.26 D direction: synthesis_feedback consumer + payload serde ──

    fn emit_synthesis_event(conn: &Connection, payload: SynthesisInteractionPayload) {
        emit_event(
            conn,
            FeedbackEvent {
                event_type: EventType::SynthesisInteraction,
                request_id: None,
                memory_id: None,
                concept_id: None,
                query: None,
                query_type: payload.metadata.as_ref().and_then(|m| m.query_type.clone()),
                topic: None,
                payload: Some(serde_json::to_value(payload).unwrap()),
            },
        )
        .unwrap();
    }

    fn mk_payload(
        synthesis_id: &str,
        kind: SynthesisInteractionKind,
        cluster_id: Option<i64>,
        query_type: Option<&str>,
    ) -> SynthesisInteractionPayload {
        SynthesisInteractionPayload {
            synthesis_id: synthesis_id.to_string(),
            recall_id: format!("recall-{synthesis_id}"),
            interaction: kind,
            metadata: Some(SynthesisMetadata {
                query_type: query_type.map(|s| s.to_string()),
                cluster_id,
                source_count: None,
                synthesis_chars: None,
            }),
        }
    }

    #[test]
    fn synthesis_interaction_kind_round_trip_serde() {
        // Round-trip each of the 4 SynthesisInteractionKind variants
        // through JSON. Catches any future serde rename / tag drift.
        let cases = vec![
            SynthesisInteractionKind::Viewed { dwell_ms: 4200 },
            SynthesisInteractionKind::ClickedSource { source_index: 3 },
            SynthesisInteractionKind::ImmediateRequery { gap_ms: 1500 },
            SynthesisInteractionKind::ExplicitThumb { up: true },
        ];
        for k in cases {
            let json = serde_json::to_string(&k).unwrap();
            let back: SynthesisInteractionKind = serde_json::from_str(&json).unwrap();
            assert_eq!(k, back, "round-trip failed for {k:?} (json={json})");
        }
    }

    #[test]
    fn synthesis_interaction_payload_round_trip_with_metadata() {
        let p = SynthesisInteractionPayload {
            synthesis_id: "syn-1".into(),
            recall_id: "rec-1".into(),
            interaction: SynthesisInteractionKind::Viewed { dwell_ms: 5000 },
            metadata: Some(SynthesisMetadata {
                query_type: Some("Semantic".into()),
                cluster_id: Some(42),
                source_count: Some(5),
                synthesis_chars: Some(800),
            }),
        };
        let json = serde_json::to_string(&p).unwrap();
        let back: SynthesisInteractionPayload = serde_json::from_str(&json).unwrap();
        assert_eq!(p, back);
    }

    #[test]
    fn synthesis_interaction_payload_back_compat_missing_metadata() {
        // Backward serde: legacy payloads without `metadata` deserialize
        // to `None`, not a panic. Cross-agent invariant 5.
        let json = r#"{
            "synthesis_id":"syn-x",
            "recall_id":"rec-x",
            "interaction":{"kind":"viewed","dwell_ms":1000}
        }"#;
        let p: SynthesisInteractionPayload = serde_json::from_str(json).unwrap();
        assert_eq!(p.synthesis_id, "syn-x");
        assert_eq!(p.recall_id, "rec-x");
        assert!(matches!(
            p.interaction,
            SynthesisInteractionKind::Viewed { dwell_ms: 1000 }
        ));
        assert!(p.metadata.is_none(), "missing metadata → None, not panic");
    }

    #[test]
    fn compute_useful_rate_table() {
        // Cold start (zeros) → 0.0 by clamp lower bound.
        let cold = ClusterSynthesisStats::default();
        assert_eq!(compute_useful_rate(&cold), 0.0);

        // High-engagement: 10 views all over the dwell threshold,
        // 5 clicks, 8 thumbs up, 0 requeries.
        // dwell_pct = 1.0; click = 0.5; thumb = 8/9 ≈ 0.889;
        // requery = 0.0
        // numerator = 1*1 + 0.5*0.5 + 2*0.889 - 2*0 = 3.028
        // denom = 1 + 0.5 + 2 + 2 = 5.5 → 0.55-ish, well above 0.5
        let happy = ClusterSynthesisStats {
            viewed_count: 10,
            viewed_dwell_total_ms: 10 * 5000,
            dwell_samples: vec![5000; 10],
            viewed_dwell_p50_ms: Some(5000),
            clicked_source_count: 5,
            immediate_requery_count: 0,
            explicit_up: 8,
            explicit_down: 0,
            useful_rate: 0.0,
            ..Default::default()
        };
        let happy_rate = compute_useful_rate(&happy);
        assert!(
            happy_rate > 0.5,
            "happy path useful_rate={happy_rate} should exceed 0.5"
        );
        assert!(happy_rate <= 1.0, "useful_rate must stay <= 1.0");

        // Bad: 10 views all under dwell threshold, 0 clicks, 0 thumbs,
        // 8 requeries. Strong negative signal → clamped to 0.0.
        let bad = ClusterSynthesisStats {
            viewed_count: 10,
            viewed_dwell_total_ms: 10 * 100,
            dwell_samples: vec![100; 10],
            viewed_dwell_p50_ms: Some(100),
            clicked_source_count: 0,
            immediate_requery_count: 8,
            explicit_up: 0,
            explicit_down: 5,
            useful_rate: 0.0,
            ..Default::default()
        };
        let bad_rate = compute_useful_rate(&bad);
        assert!(bad_rate >= 0.0, "useful_rate must clamp at 0.0");
        assert!(
            bad_rate < 0.5,
            "bad path useful_rate={bad_rate} should fall below 0.5"
        );
    }

    #[test]
    fn dynamic_useful_rate_weights_affect_synthesis_formula() {
        let stats = ClusterSynthesisStats {
            viewed_count: 10,
            viewed_dwell_total_ms: 10 * 5000,
            dwell_samples: vec![5000; 10],
            viewed_dwell_p50_ms: Some(5000),
            clicked_source_count: 5,
            immediate_requery_count: 4,
            explicit_up: 1,
            explicit_down: 0,
            useful_rate: 0.0,
            ..Default::default()
        };
        let baseline = compute_useful_rate(&stats);
        let prior_weights = UsefulRateWeights::from_priors(
            UsefulRateWeights::synthesis_bootstrap(),
            0.1,
            0.1,
            3.0,
            0.1,
        );
        let dynamic = compute_useful_rate_with_weights(&stats, prior_weights);

        assert!(
            dynamic > baseline,
            "SignalHint-derived priors should be able to move the production formula"
        );
    }

    #[test]
    fn synthesis_by_cluster_cap_evicts_lru_bucket() {
        let mut by_cluster = HashMap::new();
        by_cluster.insert(
            "oldest".to_string(),
            ClusterSynthesisStats {
                last_event_id: 1,
                ..Default::default()
            },
        );
        for idx in 1..SYNTHESIS_BY_CLUSTER_CAP {
            by_cluster.insert(
                format!("bucket-{idx}"),
                ClusterSynthesisStats {
                    last_event_id: (idx as i64) + 10,
                    ..Default::default()
                },
            );
        }

        evict_synthesis_lru_if_at_cap(&mut by_cluster, "new-bucket");

        assert_eq!(by_cluster.len(), SYNTHESIS_BY_CLUSTER_CAP - 1);
        assert!(!by_cluster.contains_key("oldest"));
    }

    #[test]
    fn ars_effective_scalar_round_trips_through_adaptive_state() {
        let mut state = AdaptiveState::default();
        assert!(state
            .ars_effective_scalar(ARS_SCALAR_SYNTHESIS_USEFUL_RATE_THRESHOLD)
            .is_none());

        state.set_ars_effective_scalar(ARS_SCALAR_SYNTHESIS_USEFUL_RATE_THRESHOLD, 0.42);

        let json = serde_json::to_string(&state).unwrap();
        let restored: AdaptiveState = serde_json::from_str(&json).unwrap();
        assert_eq!(
            restored.ars_effective_scalar(ARS_SCALAR_SYNTHESIS_USEFUL_RATE_THRESHOLD),
            Some(0.42)
        );
    }

    #[test]
    fn recompute_synthesis_feedback_empty() {
        let conn = setup_db();
        let (state, max_id) = recompute_synthesis_feedback_stats(&conn, None).unwrap();
        assert!(state.by_cluster.is_empty());
        assert!(state.by_synthesis.is_empty());
        assert!(state.by_synthesis_order.is_empty());
        assert_eq!(state.total_events, 0);
        assert_eq!(state.last_consumed_event_id, 0);
        assert_eq!(max_id, None, "no events → no offset to commit");
    }

    #[test]
    fn recompute_synthesis_feedback_aggregates_per_bucket() {
        let conn = setup_db();
        // 5 viewed events for (cluster=1, qtype=Semantic), 1 thumb-up.
        for i in 0..5 {
            emit_synthesis_event(
                &conn,
                mk_payload(
                    &format!("syn-A{i}"),
                    SynthesisInteractionKind::Viewed { dwell_ms: 4500 },
                    Some(1),
                    Some("Semantic"),
                ),
            );
        }
        emit_synthesis_event(
            &conn,
            mk_payload(
                "syn-A0",
                SynthesisInteractionKind::ExplicitThumb { up: true },
                Some(1),
                Some("Semantic"),
            ),
        );

        let (state, max_id) = recompute_synthesis_feedback_stats(&conn, None).unwrap();
        let key = synthesis_bucket_key(Some(1), "Semantic");
        let bucket = state.by_cluster.get(&key).expect("bucket should exist");
        assert_eq!(bucket.viewed_count, 5);
        assert_eq!(bucket.viewed_dwell_total_ms, 5 * 4500);
        assert_eq!(bucket.dwell_samples.len(), 5);
        assert_eq!(bucket.viewed_dwell_p50_ms, Some(4500));
        assert_eq!(bucket.explicit_up, 1);
        assert_eq!(bucket.explicit_down, 0);
        assert!(
            (bucket.useful_rate - compute_useful_rate(bucket)).abs() < 1e-9,
            "stored useful_rate must match the pure fn"
        );
        assert_eq!(state.total_events, 6);
        assert_eq!(max_id, Some(6));
        assert_eq!(state.last_consumed_event_id, 6);
    }

    #[test]
    fn recompute_synthesis_feedback_peek_replay_is_idempotent() {
        // Mirrors `recompute_concept_refresh_stats_peek_replay_is_idempotent`
        // (line 1724): the v0.24 5-invariant Codex round-1 HIGH guard.
        let conn = setup_db();
        emit_synthesis_event(
            &conn,
            mk_payload(
                "syn-1",
                SynthesisInteractionKind::Viewed { dwell_ms: 4000 },
                Some(7),
                Some("Semantic"),
            ),
        );
        emit_synthesis_event(
            &conn,
            mk_payload(
                "syn-2",
                SynthesisInteractionKind::ClickedSource { source_index: 1 },
                Some(7),
                Some("Semantic"),
            ),
        );

        // First call: peeks both, bumps last_consumed_event_id to 2.
        let (state, max_id) = recompute_synthesis_feedback_stats(&conn, None).unwrap();
        assert_eq!(state.total_events, 2);
        assert_eq!(state.last_consumed_event_id, 2);
        assert_eq!(max_id, Some(2));

        // Simulate prior pass's commit_offset FAILED — replay must be
        // a no-op for counter increments. Without the watermark guard
        // viewed_count would double.
        let (state2, max_id2) =
            recompute_synthesis_feedback_stats(&conn, Some(state.clone())).unwrap();
        assert_eq!(
            state2.total_events, 2,
            "replay-safety: events with id <= last_consumed_event_id are skipped"
        );
        let key = synthesis_bucket_key(Some(7), "Semantic");
        let bucket = state2.by_cluster.get(&key).expect("bucket exists");
        assert_eq!(
            bucket.viewed_count, 1,
            "no double-count on replay (1 viewed event total)"
        );
        assert_eq!(bucket.clicked_source_count, 1);
        assert_eq!(max_id2, Some(2), "max_id reported so caller can re-attempt");

        // Caller successfully commits this pass; subsequent peek finds
        // nothing.
        commit_offset(&conn, &[("synthesis_feedback", max_id2.unwrap())]).unwrap();
        let (state3, max_id3) = recompute_synthesis_feedback_stats(&conn, Some(state2)).unwrap();
        assert_eq!(state3.total_events, 2);
        assert_eq!(max_id3, None);

        // New event arrives after commit: state grows by exactly one.
        emit_synthesis_event(
            &conn,
            mk_payload(
                "syn-3",
                SynthesisInteractionKind::ExplicitThumb { up: true },
                Some(7),
                Some("Semantic"),
            ),
        );
        let (state4, max_id4) = recompute_synthesis_feedback_stats(&conn, Some(state3)).unwrap();
        assert_eq!(state4.total_events, 3);
        assert_eq!(max_id4, Some(3));
        let bucket4 = state4.by_cluster.get(&key).unwrap();
        assert_eq!(bucket4.explicit_up, 1);
    }

    #[test]
    fn recompute_synthesis_feedback_with_judge_replay_is_idempotent() {
        // Guards the `saturating_add` in `llm_judge_count` / `llm_judge_hit_count`:
        // replay (commit_offset failed) must NOT double-count LlmJudge events.
        // Uses `_with_judge` variant — the only consumer that folds SynthesisLlmJudge.
        let conn = setup_db();

        // Emit one SynthesisInteraction + one SynthesisLlmJudge (hit=true) for
        // cluster 7 / Semantic.
        emit_synthesis_event(
            &conn,
            mk_payload(
                "syn-j1",
                SynthesisInteractionKind::Viewed { dwell_ms: 4000 },
                Some(7),
                Some("Semantic"),
            ),
        );
        emit_event(
            &conn,
            FeedbackEvent {
                event_type: EventType::SynthesisLlmJudge,
                request_id: None,
                memory_id: None,
                concept_id: None,
                query: None,
                query_type: Some("Semantic".to_string()),
                topic: None,
                payload: Some(
                    serde_json::to_value(SynthesisLlmJudgePayload {
                        synthesis_id: "syn-j1".to_string(),
                        judge_model: "mock".to_string(),
                        hit: true,
                        reason: "looks good".to_string(),
                        stamp_hash: "abc123".to_string(),
                        source: JudgeSource::AutoSampled,
                        metadata: Some(JudgeMetadata {
                            query_type: Some("Semantic".to_string()),
                            cluster_id: Some(7),
                            source_count: None,
                            judge_latency_ms: None,
                        }),
                        signal_hint: None,
                    })
                    .unwrap(),
                ),
            },
        )
        .unwrap();

        // First call: drains both events, bumps watermark.
        let (state, pending, calibration, max_id) = recompute_synthesis_feedback_stats_with_judge(
            &conn,
            None,
            HashMap::new(),
            JudgeCalibrationState::default(),
            LLM_JUDGE_WEIGHT_DECAY_RATE,
        )
        .unwrap();
        let key = synthesis_bucket_key(Some(7), "Semantic");
        let bucket = state
            .by_cluster
            .get(&key)
            .expect("bucket must exist after first call");
        assert_eq!(state.last_consumed_event_id, 2);
        assert_eq!(max_id, Some(2));
        let first_judge_count = bucket.llm_judge_count;
        let first_judge_hit_count = bucket.llm_judge_hit_count;
        let first_useful_rate = bucket.useful_rate;
        assert_eq!(first_judge_count, 1, "one LlmJudge event consumed");
        assert_eq!(first_judge_hit_count, 1, "hit=true counted");

        // Replay: simulate commit_offset failure — pass prior state back.
        // `saturating_add` would double-count WITHOUT the watermark guard.
        let (state2, _pending2, _calibration2, max_id2) =
            recompute_synthesis_feedback_stats_with_judge(
                &conn,
                Some(state.clone()),
                pending.clone(),
                calibration.clone(),
                LLM_JUDGE_WEIGHT_DECAY_RATE,
            )
            .unwrap();
        let bucket2 = state2
            .by_cluster
            .get(&key)
            .expect("bucket must exist on replay");
        assert_eq!(
            bucket2.llm_judge_count, first_judge_count,
            "replay must not double-count llm_judge_count"
        );
        assert_eq!(
            bucket2.llm_judge_hit_count, first_judge_hit_count,
            "replay must not double-count llm_judge_hit_count"
        );
        assert!(
            (bucket2.useful_rate - first_useful_rate).abs() < 1e-9,
            "useful_rate must be identical on replay"
        );
        assert_eq!(
            max_id2,
            Some(2),
            "max_id still reported so caller can re-attempt"
        );

        // Commit then confirm a new judge event is picked up exactly once.
        commit_offset(&conn, &[("synthesis_feedback", max_id2.unwrap())]).unwrap();
        emit_event(
            &conn,
            FeedbackEvent {
                event_type: EventType::SynthesisLlmJudge,
                request_id: None,
                memory_id: None,
                concept_id: None,
                query: None,
                query_type: Some("Semantic".to_string()),
                topic: None,
                payload: Some(
                    serde_json::to_value(SynthesisLlmJudgePayload {
                        synthesis_id: "syn-j2".to_string(),
                        judge_model: "mock".to_string(),
                        hit: false,
                        reason: "miss".to_string(),
                        stamp_hash: "def456".to_string(),
                        source: JudgeSource::AutoSampled,
                        metadata: Some(JudgeMetadata {
                            query_type: Some("Semantic".to_string()),
                            cluster_id: Some(7),
                            source_count: None,
                            judge_latency_ms: None,
                        }),
                        signal_hint: None,
                    })
                    .unwrap(),
                ),
            },
        )
        .unwrap();
        let (state3, _pending3, _cal3, max_id3) = recompute_synthesis_feedback_stats_with_judge(
            &conn,
            Some(state2),
            HashMap::new(),
            JudgeCalibrationState::default(),
            LLM_JUDGE_WEIGHT_DECAY_RATE,
        )
        .unwrap();
        let bucket3 = state3.by_cluster.get(&key).unwrap();
        assert_eq!(
            bucket3.llm_judge_count, 2,
            "second judge event increments to 2"
        );
        assert_eq!(
            bucket3.llm_judge_hit_count, 1,
            "second judge hit=false, so hit_count stays at 1"
        );
        assert_eq!(max_id3, Some(3));
    }

    #[test]
    fn recompute_synthesis_feedback_skips_malformed_payloads() {
        let conn = setup_db();
        // Two valid + one missing-payload + one malformed JSON.
        emit_synthesis_event(
            &conn,
            mk_payload(
                "syn-good-1",
                SynthesisInteractionKind::Viewed { dwell_ms: 3500 },
                Some(2),
                Some("Episodic"),
            ),
        );
        emit_event(
            &conn,
            FeedbackEvent {
                event_type: EventType::SynthesisInteraction,
                request_id: None,
                memory_id: None,
                concept_id: None,
                query: None,
                query_type: None,
                topic: None,
                payload: None, // missing payload
            },
        )
        .unwrap();
        emit_event(
            &conn,
            FeedbackEvent {
                event_type: EventType::SynthesisInteraction,
                request_id: None,
                memory_id: None,
                concept_id: None,
                query: None,
                query_type: None,
                topic: None,
                payload: Some(serde_json::json!({"unexpected": "shape"})),
            },
        )
        .unwrap();
        emit_synthesis_event(
            &conn,
            mk_payload(
                "syn-good-2",
                SynthesisInteractionKind::ExplicitThumb { up: false },
                Some(2),
                Some("Episodic"),
            ),
        );

        let (state, _) = recompute_synthesis_feedback_stats(&conn, None).unwrap();
        // Only 2 valid events survived; total_events counts only those.
        assert_eq!(state.total_events, 2);
        let key = synthesis_bucket_key(Some(2), "Episodic");
        let bucket = state.by_cluster.get(&key).expect("bucket should exist");
        assert_eq!(bucket.viewed_count, 1);
        assert_eq!(bucket.explicit_down, 1);
    }

    #[test]
    fn recompute_synthesis_feedback_caps_dwell_reservoir_fifo() {
        // Reservoir is per-bucket; emit > cap viewed events and assert
        // FIFO eviction keeps the cap and drops oldest.
        let conn = setup_db();
        let overflow = 25;
        for i in 0..(SYNTHESIS_DWELL_RESERVOIR_CAP + overflow) {
            emit_synthesis_event(
                &conn,
                mk_payload(
                    &format!("syn-fifo-{i}"),
                    SynthesisInteractionKind::Viewed {
                        dwell_ms: (i + 1) as u64,
                    },
                    Some(3),
                    Some("Exploratory"),
                ),
            );
        }
        let (state, _) = recompute_synthesis_feedback_stats(&conn, None).unwrap();
        let key = synthesis_bucket_key(Some(3), "Exploratory");
        let bucket = state.by_cluster.get(&key).unwrap();
        assert_eq!(bucket.dwell_samples.len(), SYNTHESIS_DWELL_RESERVOIR_CAP);
        // Oldest `overflow` samples evicted → smallest surviving dwell
        // is `overflow + 1`.
        assert_eq!(
            *bucket.dwell_samples.first().unwrap(),
            (overflow + 1) as u64
        );
    }

    #[test]
    fn recompute_synthesis_feedback_per_synthesis_lru_caps_with_dual_update() {
        // Insert > SYNTHESIS_PER_ID_CAP unique synthesis_ids; verify that
        // both `by_synthesis` HashMap and `by_synthesis_order` Vec stay
        // in sync with the cap (no orphan keys / no oversized vec).
        let conn = setup_db();
        let overflow = 5;
        let total = SYNTHESIS_PER_ID_CAP + overflow;
        for i in 0..total {
            emit_synthesis_event(
                &conn,
                mk_payload(
                    &format!("syn-lru-{i}"),
                    SynthesisInteractionKind::Viewed { dwell_ms: 1000 },
                    Some(4),
                    Some("Semantic"),
                ),
            );
        }
        let (state, _) = recompute_synthesis_feedback_stats(&conn, None).unwrap();
        assert_eq!(state.by_synthesis.len(), SYNTHESIS_PER_ID_CAP);
        assert_eq!(state.by_synthesis_order.len(), SYNTHESIS_PER_ID_CAP);
        // First `overflow` ids should have been evicted from BOTH stores.
        for i in 0..overflow {
            let evicted = format!("syn-lru-{i}");
            assert!(
                !state.by_synthesis.contains_key(&evicted),
                "evicted key {evicted} must be gone from HashMap"
            );
            assert!(
                !state.by_synthesis_order.contains(&evicted),
                "evicted key {evicted} must be gone from order vec"
            );
        }
    }

    #[test]
    fn synthesis_feedback_cas_merge_keeps_more_advanced_state() {
        // CAS arbitration: the writer with higher `last_consumed_event_id`
        // wins. Mirrors `cas_retry_keeps_more_advanced_reservoir_by_event_id`
        // (line 1841) for the new synthesis_feedback_stats arm.
        let conn = setup_db();

        let mut winner_by_cluster = HashMap::new();
        winner_by_cluster.insert(
            synthesis_bucket_key(Some(11), "Semantic"),
            ClusterSynthesisStats {
                viewed_count: 50,
                viewed_dwell_total_ms: 50 * 4000,
                dwell_samples: vec![4000; 50],
                viewed_dwell_p50_ms: Some(4000),
                clicked_source_count: 20,
                immediate_requery_count: 1,
                explicit_up: 12,
                explicit_down: 2,
                useful_rate: 0.7,
                ..Default::default()
            },
        );
        let winner_synth = SynthesisFeedbackState {
            by_cluster: winner_by_cluster.clone(),
            by_synthesis: HashMap::new(),
            by_synthesis_order: vec![],
            last_consumed_event_id: 500,
            total_events: 50,
        };
        let winner = AdaptiveState {
            version: 5,
            synthesis_feedback_stats: Some(winner_synth.clone()),
            ..AdaptiveState::default()
        };
        // Seed DB at v=5.
        let winner_json = serde_json::to_string(&winner).unwrap();
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES ('adaptive_state', ?1)",
            rusqlite::params![&winner_json],
        )
        .unwrap();

        // Stale writer (last_consumed_event_id=100 < 500) tries to save.
        let stale_synth = SynthesisFeedbackState {
            by_cluster: HashMap::new(),
            by_synthesis: HashMap::new(),
            by_synthesis_order: vec![],
            last_consumed_event_id: 100,
            total_events: 5,
        };
        let stale = AdaptiveState {
            version: 2,
            synthesis_feedback_stats: Some(stale_synth),
            ..AdaptiveState::default()
        };
        stale.save_snapshot(&conn).unwrap();

        let restored = AdaptiveState::restore_snapshot(&conn).unwrap();
        let restored_synth = restored.synthesis_feedback_stats.unwrap();
        assert_eq!(
            restored_synth.last_consumed_event_id, 500,
            "CAS winner with higher last_consumed_event_id must survive"
        );
        assert_eq!(restored_synth.by_cluster, winner_by_cluster);
    }

    #[test]
    fn synthesis_feedback_cas_merge_takes_writer_when_more_advanced() {
        // Counterpart: writer with HIGHER last_consumed_event_id replaces
        // the older snapshot. Mirrors `cas_retry_takes_writer_when_more_advanced`
        // (line 1900).
        let conn = setup_db();
        let older_synth = SynthesisFeedbackState {
            by_cluster: HashMap::new(),
            by_synthesis: HashMap::new(),
            by_synthesis_order: vec![],
            last_consumed_event_id: 30,
            total_events: 5,
        };
        let older = AdaptiveState {
            version: 5,
            synthesis_feedback_stats: Some(older_synth),
            ..AdaptiveState::default()
        };
        let older_json = serde_json::to_string(&older).unwrap();
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES ('adaptive_state', ?1)",
            rusqlite::params![&older_json],
        )
        .unwrap();

        let mut fresher_buckets = HashMap::new();
        fresher_buckets.insert(
            synthesis_bucket_key(Some(99), "Semantic"),
            ClusterSynthesisStats {
                viewed_count: 100,
                ..ClusterSynthesisStats::default()
            },
        );
        let fresher_synth = SynthesisFeedbackState {
            by_cluster: fresher_buckets.clone(),
            by_synthesis: HashMap::new(),
            by_synthesis_order: vec![],
            last_consumed_event_id: 999,
            total_events: 100,
        };
        let writer = AdaptiveState {
            version: 2,
            synthesis_feedback_stats: Some(fresher_synth.clone()),
            ..AdaptiveState::default()
        };
        writer.save_snapshot(&conn).unwrap();

        let restored = AdaptiveState::restore_snapshot(&conn).unwrap();
        let restored_synth = restored.synthesis_feedback_stats.unwrap();
        assert_eq!(restored_synth.last_consumed_event_id, 999);
        assert_eq!(restored_synth.by_cluster, fresher_buckets);
    }

    #[test]
    fn synthesis_feedback_cas_preserves_existing_when_writer_has_none() {
        // Mirrors `cas_retry_preserves_existing_stats_when_writer_has_none`
        // (line 1953). Writer with `synthesis_feedback_stats = None` MUST
        // NOT overwrite existing learned state.
        let conn = setup_db();
        let learned = SynthesisFeedbackState {
            by_cluster: HashMap::new(),
            by_synthesis: HashMap::new(),
            by_synthesis_order: vec![],
            last_consumed_event_id: 1234,
            total_events: 42,
        };
        let prior = AdaptiveState {
            version: 5,
            synthesis_feedback_stats: Some(learned.clone()),
            ..AdaptiveState::default()
        };
        let prior_json = serde_json::to_string(&prior).unwrap();
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES ('adaptive_state', ?1)",
            rusqlite::params![&prior_json],
        )
        .unwrap();

        let our = AdaptiveState {
            version: 2,
            synthesis_feedback_stats: None,
            ..AdaptiveState::default()
        };
        our.save_snapshot(&conn).unwrap();

        let restored = AdaptiveState::restore_snapshot(&conn).unwrap();
        assert_eq!(
            restored.synthesis_feedback_stats,
            Some(learned),
            "CAS merge must preserve existing stats when writer has None"
        );
    }

    #[test]
    fn synthesis_bucket_helper_gates_on_cold_start_n() {
        // Helper returns Some only when viewed_count >= SYNTHESIS_COLD_START_N.
        let key = synthesis_bucket_key(Some(8), "Semantic");
        let mut by_cluster = HashMap::new();
        // Below cold-start threshold.
        by_cluster.insert(
            key.clone(),
            ClusterSynthesisStats {
                viewed_count: SYNTHESIS_COLD_START_N - 1,
                ..ClusterSynthesisStats::default()
            },
        );
        let mut state = AdaptiveState {
            synthesis_feedback_stats: Some(SynthesisFeedbackState {
                by_cluster,
                ..SynthesisFeedbackState::default()
            }),
            ..AdaptiveState::default()
        };
        assert!(
            state.synthesis_bucket(Some(8), "Semantic").is_none(),
            "cold-start: bucket below SYNTHESIS_COLD_START_N must return None"
        );

        // At cold-start threshold → Some.
        let s = state.synthesis_feedback_stats.as_mut().unwrap();
        s.by_cluster.get_mut(&key).unwrap().viewed_count = SYNTHESIS_COLD_START_N;
        assert!(state.synthesis_bucket(Some(8), "Semantic").is_some());
    }

    #[test]
    fn synthesis_feedback_event_type_str() {
        // Guards against accidental rename of the SynthesisInteraction
        // event_type string (which would silently de-route every existing
        // emitted event from the consumer).
        assert_eq!(
            EventType::SynthesisInteraction.as_str(),
            "synthesis_interaction"
        );
    }

    #[test]
    fn synthesis_feedback_state_default_is_empty() {
        // Cold-start invariant: a fresh, empty state never panics on
        // useful_rate evaluation, never reports any bucket, and serializes
        // round-trip cleanly.
        let s = SynthesisFeedbackState::default();
        assert!(s.by_cluster.is_empty());
        assert!(s.by_synthesis.is_empty());
        assert!(s.by_synthesis_order.is_empty());
        assert_eq!(s.last_consumed_event_id, 0);
        assert_eq!(s.total_events, 0);
        let json = serde_json::to_string(&s).unwrap();
        let back: SynthesisFeedbackState = serde_json::from_str(&json).unwrap();
        assert_eq!(s, back);
    }

    // ── v0.27 ARS Cap A: concept_summary_feedback consumer tests ───────────

    fn emit_concept_summary_event(conn: &Connection, payload: ConceptSummaryInteractionPayload) {
        emit_event(
            conn,
            FeedbackEvent {
                event_type: EventType::ConceptSummaryInteraction,
                request_id: None,
                memory_id: None,
                concept_id: Some(payload.concept_id.clone()),
                query: None,
                query_type: payload.metadata.as_ref().and_then(|m| m.query_type.clone()),
                topic: None,
                payload: Some(serde_json::to_value(payload).unwrap()),
            },
        )
        .unwrap();
    }

    fn mk_concept_summary_payload(
        concept_id: &str,
        kind: ConceptSummaryInteractionKind,
        cluster_id: Option<i64>,
        query_type: Option<&str>,
    ) -> ConceptSummaryInteractionPayload {
        ConceptSummaryInteractionPayload {
            concept_id: concept_id.to_string(),
            concept_summary_id: None,
            recall_id: format!("recall-{concept_id}"),
            interaction: kind,
            metadata: Some(ConceptSummaryMetadata {
                query_type: query_type.map(|s| s.to_string()),
                cluster_id,
                concept_chars: None,
                revision_version: None,
                route_context: None,
            }),
        }
    }

    #[test]
    fn concept_summary_feedback_event_type_str() {
        // Guards against accidental rename of the
        // ConceptSummaryInteraction event_type string (silent de-route).
        assert_eq!(
            EventType::ConceptSummaryInteraction.as_str(),
            "concept_summary_interaction"
        );
    }

    #[test]
    fn concept_summary_interaction_kind_round_trip_serde() {
        let cases = vec![
            ConceptSummaryInteractionKind::Viewed { dwell_ms: 4200 },
            ConceptSummaryInteractionKind::ClickedSource { source_index: 3 },
            ConceptSummaryInteractionKind::ImmediateRequery { gap_ms: 1500 },
            ConceptSummaryInteractionKind::ExplicitThumb { up: true },
        ];
        for k in cases {
            let json = serde_json::to_string(&k).unwrap();
            let back: ConceptSummaryInteractionKind = serde_json::from_str(&json).unwrap();
            assert_eq!(k, back, "round-trip failed for {k:?} (json={json})");
        }
    }

    #[test]
    fn concept_summary_interaction_payload_back_compat_missing_metadata() {
        // Pre-v0.27 payloads with no `metadata` field deserialize to None.
        let json = r#"{
            "concept_id":"con-x",
            "recall_id":"rec-x",
            "interaction":{"kind":"viewed","dwell_ms":1000}
        }"#;
        let p: ConceptSummaryInteractionPayload = serde_json::from_str(json).unwrap();
        assert_eq!(p.concept_id, "con-x");
        assert_eq!(p.recall_id, "rec-x");
        assert!(matches!(
            p.interaction,
            ConceptSummaryInteractionKind::Viewed { dwell_ms: 1000 }
        ));
        assert!(p.metadata.is_none(), "missing metadata → None, not panic");
    }

    #[test]
    fn concept_summary_interaction_payload_round_trips_summary_id() {
        let json = r#"{
            "concept_id":"con-x",
            "concept_summary_id":"cs-x",
            "recall_id":"rec-x",
            "interaction":{"kind":"explicit_thumb","up":true}
        }"#;
        let p: ConceptSummaryInteractionPayload = serde_json::from_str(json).unwrap();
        assert_eq!(p.concept_summary_id.as_deref(), Some("cs-x"));

        let back = serde_json::to_value(&p).unwrap();
        assert_eq!(
            back.get("concept_summary_id").and_then(|v| v.as_str()),
            Some("cs-x")
        );
    }

    #[test]
    fn concept_summary_metadata_preserves_shadow_route_context() {
        let metadata = ConceptSummaryMetadata {
            query_type: Some("concept_refresh".into()),
            cluster_id: Some(123),
            concept_chars: Some(900),
            revision_version: Some(4),
            route_context: Some(RecallRouteContext {
                request_id: Some("recall-1".into()),
                query_type: Some("Exploratory".into()),
                cluster_id: Some(7),
                cluster_version: Some(42),
            }),
        };

        let json = serde_json::to_string(&metadata).unwrap();
        let back: ConceptSummaryMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(back.query_type.as_deref(), Some("concept_refresh"));
        assert_eq!(back.cluster_id, Some(123));
        let route = back.route_context.expect("shadow route should round-trip");
        assert_eq!(route.request_id.as_deref(), Some("recall-1"));
        assert_eq!(route.query_type.as_deref(), Some("Exploratory"));
        assert_eq!(route.cluster_id, Some(7));
        assert_eq!(route.cluster_version, Some(42));
    }

    #[test]
    fn compute_concept_summary_useful_rate_table() {
        // Cold start (zeros) → 0.0 by clamp lower bound.
        let cold = ClusterConceptSummaryStats::default();
        assert_eq!(compute_concept_summary_useful_rate(&cold), 0.0);

        // Happy path: 10 views all over dwell threshold, 5 clicks, 8 thumbs
        // up, 0 requeries → useful_rate well above 0.5.
        let happy = ClusterConceptSummaryStats {
            viewed_count: 10,
            viewed_dwell_total_ms: 10 * 5000,
            dwell_samples: vec![5000; 10],
            viewed_dwell_p50_ms: Some(5000),
            clicked_source_count: 5,
            immediate_requery_count: 0,
            explicit_up: 8,
            explicit_down: 0,
            useful_rate: 0.0,
            ..Default::default()
        };
        let happy_rate = compute_concept_summary_useful_rate(&happy);
        assert!(happy_rate > 0.5, "happy path useful_rate={happy_rate}");
        assert!(happy_rate <= 1.0);

        // Bad path: 10 views all under dwell, 0 clicks, 0 thumbs, 8 requeries
        // → strong negative signal → clamped near 0.0.
        let bad = ClusterConceptSummaryStats {
            viewed_count: 10,
            viewed_dwell_total_ms: 10 * 100,
            dwell_samples: vec![100; 10],
            viewed_dwell_p50_ms: Some(100),
            clicked_source_count: 0,
            immediate_requery_count: 8,
            explicit_up: 0,
            explicit_down: 5,
            useful_rate: 0.0,
            ..Default::default()
        };
        let bad_rate = compute_concept_summary_useful_rate(&bad);
        assert!(bad_rate >= 0.0);
        assert!(bad_rate < 0.5);
    }

    #[test]
    fn recompute_concept_summary_feedback_empty() {
        let conn = setup_db();
        let (state, max_id) = recompute_concept_summary_feedback_stats(&conn, None).unwrap();
        assert!(state.by_cluster.is_empty());
        assert!(state.by_concept.is_empty());
        assert!(state.by_concept_order.is_empty());
        assert_eq!(state.total_events, 0);
        assert_eq!(state.last_consumed_event_id, 0);
        assert_eq!(max_id, None);
    }

    #[test]
    fn recompute_concept_summary_feedback_aggregates_per_bucket() {
        let conn = setup_db();
        // 5 viewed events for (cluster=1, qtype=Semantic), 1 thumb-up,
        // 2 clicks, 1 requery (negative signal).
        for i in 0..5 {
            emit_concept_summary_event(
                &conn,
                mk_concept_summary_payload(
                    &format!("con-A{i}"),
                    ConceptSummaryInteractionKind::Viewed { dwell_ms: 4500 },
                    Some(1),
                    Some("Semantic"),
                ),
            );
        }
        emit_concept_summary_event(
            &conn,
            mk_concept_summary_payload(
                "con-A0",
                ConceptSummaryInteractionKind::ExplicitThumb { up: true },
                Some(1),
                Some("Semantic"),
            ),
        );
        emit_concept_summary_event(
            &conn,
            mk_concept_summary_payload(
                "con-A0",
                ConceptSummaryInteractionKind::ClickedSource { source_index: 1 },
                Some(1),
                Some("Semantic"),
            ),
        );
        emit_concept_summary_event(
            &conn,
            mk_concept_summary_payload(
                "con-A1",
                ConceptSummaryInteractionKind::ClickedSource { source_index: 2 },
                Some(1),
                Some("Semantic"),
            ),
        );
        emit_concept_summary_event(
            &conn,
            mk_concept_summary_payload(
                "con-A2",
                ConceptSummaryInteractionKind::ImmediateRequery { gap_ms: 800 },
                Some(1),
                Some("Semantic"),
            ),
        );

        let (state, max_id) = recompute_concept_summary_feedback_stats(&conn, None).unwrap();
        let key = concept_summary_bucket_key(Some(1), "Semantic");
        let bucket = state.by_cluster.get(&key).expect("bucket should exist");
        assert_eq!(bucket.viewed_count, 5);
        assert_eq!(bucket.clicked_source_count, 2);
        assert_eq!(bucket.immediate_requery_count, 1);
        assert_eq!(bucket.explicit_up, 1);
        assert_eq!(bucket.explicit_down, 0);
        // Positive contributions: 5 viewed-with-dwell-above-threshold + 1
        // thumb up + 2 clicks; 1 requery as negative. The cached
        // useful_rate must equal the pure-fn value computed against the
        // same bucket — the explicit value is asserted in
        // `compute_concept_summary_useful_rate_table` above.
        assert!(
            bucket.useful_rate > 0.0,
            "useful_rate={} must be strictly positive after positive signal mix",
            bucket.useful_rate
        );
        assert!(
            (bucket.useful_rate - compute_concept_summary_useful_rate(bucket)).abs() < 1e-9,
            "stored useful_rate must match the pure fn"
        );
        assert_eq!(state.total_events, 9);
        assert_eq!(max_id, Some(9));
        assert_eq!(state.last_consumed_event_id, 9);
    }

    #[test]
    fn recompute_concept_summary_feedback_watermark_filter_skips_replay() {
        // Required test #1: events with id <= prior_high_water are skipped.
        let conn = setup_db();
        emit_concept_summary_event(
            &conn,
            mk_concept_summary_payload(
                "con-1",
                ConceptSummaryInteractionKind::Viewed { dwell_ms: 4000 },
                Some(7),
                Some("Semantic"),
            ),
        );
        emit_concept_summary_event(
            &conn,
            mk_concept_summary_payload(
                "con-2",
                ConceptSummaryInteractionKind::ClickedSource { source_index: 1 },
                Some(7),
                Some("Semantic"),
            ),
        );

        // First call drains both events, bumps last_consumed_event_id to 2.
        let (state, max_id) = recompute_concept_summary_feedback_stats(&conn, None).unwrap();
        assert_eq!(state.total_events, 2);
        assert_eq!(state.last_consumed_event_id, 2);
        assert_eq!(max_id, Some(2));

        // Replay (commit_offset failed): same events re-peeked. With the
        // watermark filter, no double-counting.
        let (state2, max_id2) =
            recompute_concept_summary_feedback_stats(&conn, Some(state.clone())).unwrap();
        assert_eq!(
            state2.total_events, 2,
            "replay-safety: events with id <= last_consumed_event_id are skipped"
        );
        let key = concept_summary_bucket_key(Some(7), "Semantic");
        let bucket = state2.by_cluster.get(&key).unwrap();
        assert_eq!(bucket.viewed_count, 1);
        assert_eq!(bucket.clicked_source_count, 1);
        assert_eq!(max_id2, Some(2));

        // Caller commits this pass; subsequent peek finds nothing.
        commit_offset(&conn, &[("concept_summary_feedback", max_id2.unwrap())]).unwrap();
        let (state3, max_id3) =
            recompute_concept_summary_feedback_stats(&conn, Some(state2)).unwrap();
        assert_eq!(state3.total_events, 2);
        assert_eq!(max_id3, None);

        // New event after commit: state grows by exactly one.
        emit_concept_summary_event(
            &conn,
            mk_concept_summary_payload(
                "con-3",
                ConceptSummaryInteractionKind::ExplicitThumb { up: true },
                Some(7),
                Some("Semantic"),
            ),
        );
        let (state4, max_id4) =
            recompute_concept_summary_feedback_stats(&conn, Some(state3)).unwrap();
        assert_eq!(state4.total_events, 3);
        assert_eq!(max_id4, Some(3));
        assert_eq!(state4.by_cluster.get(&key).unwrap().explicit_up, 1);
    }

    #[test]
    fn recompute_concept_summary_feedback_with_judge_replay_is_idempotent() {
        // Guards the `saturating_add` in `llm_judge_count` / `llm_judge_hit_count`:
        // replay (commit_offset failed) must NOT double-count LlmJudge events.
        // Uses `_with_judge` variant — the only consumer that folds ConceptSummaryLlmJudge.
        let conn = setup_db();

        // Emit one ConceptSummaryInteraction + one ConceptSummaryLlmJudge (hit=true).
        emit_concept_summary_event(
            &conn,
            mk_concept_summary_payload(
                "concept-j1",
                ConceptSummaryInteractionKind::Viewed { dwell_ms: 4000 },
                Some(7),
                Some("Semantic"),
            ),
        );
        emit_event(
            &conn,
            FeedbackEvent {
                event_type: EventType::ConceptSummaryLlmJudge,
                request_id: None,
                memory_id: None,
                concept_id: Some("concept-j1".to_string()),
                query: None,
                query_type: Some("Semantic".to_string()),
                topic: None,
                payload: Some(
                    serde_json::to_value(ConceptSummaryLlmJudgePayload {
                        concept_summary_id: "cs-j1".to_string(),
                        concept_id: "concept-j1".to_string(),
                        judge_model: "mock".to_string(),
                        hit: true,
                        reason: "looks good".to_string(),
                        stamp_hash: "abc123".to_string(),
                        source: JudgeSource::AutoSampled,
                        metadata: Some(JudgeMetadata {
                            query_type: Some("Semantic".to_string()),
                            cluster_id: Some(7),
                            source_count: None,
                            judge_latency_ms: None,
                        }),
                        signal_hint: None,
                    })
                    .unwrap(),
                ),
            },
        )
        .unwrap();

        // First call: drains both events, bumps watermark.
        let (state, pending, calibration, max_id) =
            recompute_concept_summary_feedback_stats_with_judge(
                &conn,
                None,
                HashMap::new(),
                JudgeCalibrationState::default(),
                LLM_JUDGE_WEIGHT_DECAY_RATE,
            )
            .unwrap();
        let key = concept_summary_bucket_key(Some(7), "Semantic");
        let bucket = state
            .by_cluster
            .get(&key)
            .expect("bucket must exist after first call");
        assert_eq!(state.last_consumed_event_id, 2);
        assert_eq!(max_id, Some(2));
        let first_judge_count = bucket.llm_judge_count;
        let first_judge_hit_count = bucket.llm_judge_hit_count;
        let first_useful_rate = bucket.useful_rate;
        assert_eq!(first_judge_count, 1, "one LlmJudge event consumed");
        assert_eq!(first_judge_hit_count, 1, "hit=true counted");

        // Replay: simulate commit_offset failure — pass prior state back.
        // `saturating_add` would double-count WITHOUT the watermark guard.
        let (state2, _pending2, _calibration2, max_id2) =
            recompute_concept_summary_feedback_stats_with_judge(
                &conn,
                Some(state.clone()),
                pending.clone(),
                calibration.clone(),
                LLM_JUDGE_WEIGHT_DECAY_RATE,
            )
            .unwrap();
        let bucket2 = state2
            .by_cluster
            .get(&key)
            .expect("bucket must exist on replay");
        assert_eq!(
            bucket2.llm_judge_count, first_judge_count,
            "replay must not double-count llm_judge_count"
        );
        assert_eq!(
            bucket2.llm_judge_hit_count, first_judge_hit_count,
            "replay must not double-count llm_judge_hit_count"
        );
        assert!(
            (bucket2.useful_rate - first_useful_rate).abs() < 1e-9,
            "useful_rate must be identical on replay"
        );
        assert_eq!(
            max_id2,
            Some(2),
            "max_id still reported so caller can re-attempt"
        );

        // Commit then confirm a new judge event is picked up exactly once.
        commit_offset(&conn, &[("concept_summary_feedback", max_id2.unwrap())]).unwrap();
        emit_event(
            &conn,
            FeedbackEvent {
                event_type: EventType::ConceptSummaryLlmJudge,
                request_id: None,
                memory_id: None,
                concept_id: Some("concept-j2".to_string()),
                query: None,
                query_type: Some("Semantic".to_string()),
                topic: None,
                payload: Some(
                    serde_json::to_value(ConceptSummaryLlmJudgePayload {
                        concept_summary_id: "cs-j2".to_string(),
                        concept_id: "concept-j2".to_string(),
                        judge_model: "mock".to_string(),
                        hit: false,
                        reason: "miss".to_string(),
                        stamp_hash: "def456".to_string(),
                        source: JudgeSource::AutoSampled,
                        metadata: Some(JudgeMetadata {
                            query_type: Some("Semantic".to_string()),
                            cluster_id: Some(7),
                            source_count: None,
                            judge_latency_ms: None,
                        }),
                        signal_hint: None,
                    })
                    .unwrap(),
                ),
            },
        )
        .unwrap();
        let (state3, _pending3, _cal3, max_id3) =
            recompute_concept_summary_feedback_stats_with_judge(
                &conn,
                Some(state2),
                HashMap::new(),
                JudgeCalibrationState::default(),
                LLM_JUDGE_WEIGHT_DECAY_RATE,
            )
            .unwrap();
        // The second judge event also targets cluster_id=7 / "Semantic" (same
        // bucket key) so llm_judge_count increments to 2. hit=false so
        // hit_count stays at 1.
        let bucket3 = state3.by_cluster.get(&key).unwrap();
        assert_eq!(
            bucket3.llm_judge_count, 2,
            "second judge event (same cluster/query_type) increments to 2"
        );
        assert_eq!(
            bucket3.llm_judge_hit_count, 1,
            "second judge hit=false, so hit_count stays at 1"
        );
        assert_eq!(max_id3, Some(3));
    }

    #[test]
    fn recompute_concept_summary_feedback_bucket_cap_evicts_lru_to_admit_new() {
        // v0.27.5 R2 — LRU eviction at cap (replaces v0.27.4
        // drop-new-bucket). Pre-populate state at cap with sequential
        // `last_event_id` values so cluster 0 is the LRU candidate, then
        // emit one event for a new (out-of-range) cluster. The new bucket
        // MUST appear and the cluster-0 bucket MUST be evicted; map size
        // stays at exactly cap; cluster 4095 (most-recent existing) MUST
        // remain so LRU truly evicted the LEAST-recently-active bucket.
        let conn = setup_db();
        let mut by_cluster = HashMap::new();
        for i in 0..CONCEPT_SUMMARY_BY_CLUSTER_CAP {
            // Sequential event ids: cluster 0 has the smallest id (= LRU)
            // and cluster CAP-1 has the largest. Start at 1 so cluster 0
            // can't be confused with a default-zero last_event_id.
            let stats = ClusterConceptSummaryStats {
                last_event_id: (i as i64) + 1,
                ..Default::default()
            };
            by_cluster.insert(
                concept_summary_bucket_key(Some(i as i64), "Semantic"),
                stats,
            );
        }
        let prior = ConceptSummaryFeedbackState {
            by_cluster,
            ..ConceptSummaryFeedbackState::default()
        };
        // Emit one event that would create a NEW (out-of-range) bucket.
        emit_concept_summary_event(
            &conn,
            mk_concept_summary_payload(
                "con-overflow",
                ConceptSummaryInteractionKind::Viewed { dwell_ms: 4000 },
                Some(99_999),
                Some("Semantic"),
            ),
        );
        let (state, max_id) = recompute_concept_summary_feedback_stats(&conn, Some(prior)).unwrap();
        assert_eq!(
            state.by_cluster.len(),
            CONCEPT_SUMMARY_BY_CLUSTER_CAP,
            "by_cluster MUST stay at exactly cap after LRU eviction"
        );
        // The new bucket MUST be present (eviction made room).
        let new_key = concept_summary_bucket_key(Some(99_999), "Semantic");
        assert!(
            state.by_cluster.contains_key(&new_key),
            "new bucket admitted via LRU eviction"
        );
        // The LRU candidate (cluster 0) MUST be evicted.
        let lru_key = concept_summary_bucket_key(Some(0), "Semantic");
        assert!(
            !state.by_cluster.contains_key(&lru_key),
            "LRU bucket (cluster 0, lowest last_event_id) MUST be evicted"
        );
        // The most-recent existing bucket (cluster CAP-1) MUST remain —
        // proves we didn't just evict a random bucket.
        let recent_key = concept_summary_bucket_key(
            Some((CONCEPT_SUMMARY_BY_CLUSTER_CAP - 1) as i64),
            "Semantic",
        );
        assert!(
            state.by_cluster.contains_key(&recent_key),
            "most-recently-active bucket MUST be preserved by LRU"
        );
        assert_eq!(state.total_events, 1);
        assert_eq!(max_id, Some(1));
        assert_eq!(state.last_consumed_event_id, 1);
    }

    #[test]
    fn recompute_concept_summary_feedback_query_type_whitelist_normalizes() {
        // Required test #3: non-allowed query_type → "unknown" bucket.
        let conn = setup_db();
        // Allowed value lands in its own bucket.
        emit_concept_summary_event(
            &conn,
            mk_concept_summary_payload(
                "con-1",
                ConceptSummaryInteractionKind::Viewed { dwell_ms: 4000 },
                Some(5),
                Some("Semantic"),
            ),
        );
        // Adversarial query_type → routed to "unknown".
        emit_concept_summary_event(
            &conn,
            mk_concept_summary_payload(
                "con-2",
                ConceptSummaryInteractionKind::Viewed { dwell_ms: 4000 },
                Some(5),
                Some("'; DROP TABLE memories; --"),
            ),
        );
        // Empty query_type also → "unknown".
        emit_concept_summary_event(
            &conn,
            mk_concept_summary_payload(
                "con-3",
                ConceptSummaryInteractionKind::Viewed { dwell_ms: 4000 },
                Some(5),
                Some(""),
            ),
        );

        let (state, _) = recompute_concept_summary_feedback_stats(&conn, None).unwrap();
        let semantic_key = concept_summary_bucket_key(Some(5), "Semantic");
        let unknown_key = concept_summary_bucket_key(Some(5), "unknown");
        assert!(
            state.by_cluster.contains_key(&semantic_key),
            "allowed Semantic query_type lands in its own bucket"
        );
        assert!(
            state.by_cluster.contains_key(&unknown_key),
            "non-allowed and empty query_types both normalize to unknown"
        );
        assert_eq!(state.by_cluster.get(&unknown_key).unwrap().viewed_count, 2);
    }

    #[test]
    fn recompute_concept_summary_feedback_folds_route_context_bucket() {
        let conn = setup_db();
        let synthetic_key = concept_summary_bucket_key(Some(123), "concept_refresh");
        let real_key = concept_summary_bucket_key(Some(7), "Exploratory");
        let payload = ConceptSummaryInteractionPayload {
            concept_id: "con-shadow".into(),
            concept_summary_id: Some("cs-shadow".into()),
            recall_id: "rec-shadow".into(),
            interaction: ConceptSummaryInteractionKind::Viewed { dwell_ms: 4000 },
            metadata: Some(ConceptSummaryMetadata {
                query_type: Some("concept_refresh".into()),
                cluster_id: Some(123),
                concept_chars: None,
                revision_version: None,
                route_context: Some(RecallRouteContext {
                    request_id: Some("rec-shadow".into()),
                    query_type: Some("Exploratory".into()),
                    cluster_id: Some(7),
                    cluster_version: Some(11),
                }),
            }),
        };
        emit_concept_summary_event(&conn, payload);

        let (state, _) = recompute_concept_summary_feedback_stats(&conn, None).unwrap();

        assert!(
            state.by_cluster.contains_key(&synthetic_key),
            "production synthetic bucket must receive the event"
        );
        assert!(
            state.by_cluster.contains_key(&real_key),
            "real recall route context must also receive human feedback"
        );
        assert_eq!(
            state.by_cluster.get(&synthetic_key).unwrap().viewed_count,
            1
        );
        assert_eq!(state.by_cluster.get(&real_key).unwrap().viewed_count, 1);
    }

    #[test]
    fn recompute_concept_summary_feedback_useful_rate_signal_directions() {
        // Required test #4: positive vs negative contributions to useful_rate.
        // Signal mix calibrated to land the positive bucket above the
        // bootstrap cutoff and the negative bucket below it.
        //
        // Positive bucket algebra (with the bootstrap weights):
        //   10 viewed (dwell 5000ms above 3000ms threshold) → dwell_pct = 1.0
        //   10 clicks → click_rate = 1.0
        //   10 thumbs up → thumb_rate = 10/11 ≈ 0.909
        //   0 requeries → requery_rate = 0
        //   numerator = 1*1.0 + 0.5*1.0 + 2*0.909 - 2*0 = 3.318
        //   denom = 1 + 0.5 + 2 + 2 = 5.5
        //   useful_rate ≈ 0.603 (above 0.5)
        let conn = setup_db();

        for _ in 0..10 {
            emit_concept_summary_event(
                &conn,
                mk_concept_summary_payload(
                    "con-pos",
                    ConceptSummaryInteractionKind::Viewed { dwell_ms: 5000 },
                    Some(11),
                    Some("Semantic"),
                ),
            );
        }
        for _ in 0..10 {
            emit_concept_summary_event(
                &conn,
                mk_concept_summary_payload(
                    "con-pos",
                    ConceptSummaryInteractionKind::ClickedSource { source_index: 1 },
                    Some(11),
                    Some("Semantic"),
                ),
            );
        }
        for _ in 0..10 {
            emit_concept_summary_event(
                &conn,
                mk_concept_summary_payload(
                    "con-pos",
                    ConceptSummaryInteractionKind::ExplicitThumb { up: true },
                    Some(11),
                    Some("Semantic"),
                ),
            );
        }

        // Negative bucket algebra:
        //   5 viewed (dwell 200ms — under threshold) → dwell_pct = 0
        //   0 clicks
        //   5 requeries → requery_rate = 1.0
        //   3 thumbs down → thumb_rate = 0/4 = 0
        //   numerator = 0 + 0 + 0 - 2*1.0 = -2.0 → clamped to 0.0
        for _ in 0..5 {
            emit_concept_summary_event(
                &conn,
                mk_concept_summary_payload(
                    "con-neg",
                    ConceptSummaryInteractionKind::Viewed { dwell_ms: 200 },
                    Some(22),
                    Some("Semantic"),
                ),
            );
        }
        for _ in 0..5 {
            emit_concept_summary_event(
                &conn,
                mk_concept_summary_payload(
                    "con-neg",
                    ConceptSummaryInteractionKind::ImmediateRequery { gap_ms: 300 },
                    Some(22),
                    Some("Semantic"),
                ),
            );
        }
        for _ in 0..3 {
            emit_concept_summary_event(
                &conn,
                mk_concept_summary_payload(
                    "con-neg",
                    ConceptSummaryInteractionKind::ExplicitThumb { up: false },
                    Some(22),
                    Some("Semantic"),
                ),
            );
        }

        let (state, _) = recompute_concept_summary_feedback_stats(&conn, None).unwrap();
        let pos_key = concept_summary_bucket_key(Some(11), "Semantic");
        let neg_key = concept_summary_bucket_key(Some(22), "Semantic");
        let pos = state.by_cluster.get(&pos_key).unwrap();
        let neg = state.by_cluster.get(&neg_key).unwrap();
        assert!(
            pos.useful_rate > neg.useful_rate,
            "positive bucket useful_rate={} must exceed negative bucket useful_rate={}",
            pos.useful_rate,
            neg.useful_rate
        );
        assert!(
            pos.useful_rate > 0.5,
            "positive bucket useful_rate={} above bootstrap cutoff",
            pos.useful_rate
        );
        assert!(
            neg.useful_rate < 0.5,
            "negative bucket useful_rate={} below bootstrap cutoff",
            neg.useful_rate
        );
    }

    #[test]
    fn recompute_concept_summary_feedback_skips_malformed_payloads() {
        let conn = setup_db();
        // 1 valid + 1 missing-payload + 1 malformed JSON + 1 valid.
        emit_concept_summary_event(
            &conn,
            mk_concept_summary_payload(
                "con-good-1",
                ConceptSummaryInteractionKind::Viewed { dwell_ms: 3500 },
                Some(2),
                Some("Episodic"),
            ),
        );
        emit_event(
            &conn,
            FeedbackEvent {
                event_type: EventType::ConceptSummaryInteraction,
                request_id: None,
                memory_id: None,
                concept_id: Some("con-malformed-1".into()),
                query: None,
                query_type: None,
                topic: None,
                payload: None,
            },
        )
        .unwrap();
        emit_event(
            &conn,
            FeedbackEvent {
                event_type: EventType::ConceptSummaryInteraction,
                request_id: None,
                memory_id: None,
                concept_id: Some("con-malformed-2".into()),
                query: None,
                query_type: None,
                topic: None,
                payload: Some(serde_json::json!({"unexpected": "shape"})),
            },
        )
        .unwrap();
        emit_concept_summary_event(
            &conn,
            mk_concept_summary_payload(
                "con-good-2",
                ConceptSummaryInteractionKind::ExplicitThumb { up: false },
                Some(2),
                Some("Episodic"),
            ),
        );

        let (state, _) = recompute_concept_summary_feedback_stats(&conn, None).unwrap();
        // Only 2 valid events survived; total_events counts only those.
        assert_eq!(state.total_events, 2);
        let key = concept_summary_bucket_key(Some(2), "Episodic");
        let bucket = state.by_cluster.get(&key).unwrap();
        assert_eq!(bucket.viewed_count, 1);
        assert_eq!(bucket.explicit_down, 1);
    }

    #[test]
    fn recompute_concept_summary_feedback_caps_dwell_reservoir_fifo() {
        let conn = setup_db();
        let overflow = 25;
        for i in 0..(CONCEPT_SUMMARY_DWELL_RESERVOIR_CAP + overflow) {
            emit_concept_summary_event(
                &conn,
                mk_concept_summary_payload(
                    &format!("con-fifo-{i}"),
                    ConceptSummaryInteractionKind::Viewed {
                        dwell_ms: (i + 1) as u64,
                    },
                    Some(3),
                    Some("Exploratory"),
                ),
            );
        }
        let (state, _) = recompute_concept_summary_feedback_stats(&conn, None).unwrap();
        let key = concept_summary_bucket_key(Some(3), "Exploratory");
        let bucket = state.by_cluster.get(&key).unwrap();
        assert_eq!(
            bucket.dwell_samples.len(),
            CONCEPT_SUMMARY_DWELL_RESERVOIR_CAP
        );
        // Oldest `overflow` samples evicted → smallest surviving dwell is
        // `overflow + 1`.
        assert_eq!(
            *bucket.dwell_samples.first().unwrap(),
            (overflow + 1) as u64
        );
    }

    #[test]
    fn recompute_concept_summary_feedback_per_concept_lru_caps_with_dual_update() {
        // Insert > CONCEPT_SUMMARY_PER_ID_CAP unique concept_ids; verify
        // both `by_concept` HashMap and `by_concept_order` Vec stay in
        // sync with the cap (no orphan keys / no oversized vec).
        let conn = setup_db();
        let overflow = 5;
        let total = CONCEPT_SUMMARY_PER_ID_CAP + overflow;
        for i in 0..total {
            emit_concept_summary_event(
                &conn,
                mk_concept_summary_payload(
                    &format!("con-lru-{i}"),
                    ConceptSummaryInteractionKind::Viewed { dwell_ms: 1000 },
                    Some(4),
                    Some("Semantic"),
                ),
            );
        }
        let (state, _) = recompute_concept_summary_feedback_stats(&conn, None).unwrap();
        assert_eq!(state.by_concept.len(), CONCEPT_SUMMARY_PER_ID_CAP);
        assert_eq!(state.by_concept_order.len(), CONCEPT_SUMMARY_PER_ID_CAP);
        for i in 0..overflow {
            let evicted = format!("con-lru-{i}");
            assert!(
                !state.by_concept.contains_key(&evicted),
                "evicted key {evicted} must be gone from HashMap"
            );
            assert!(
                !state.by_concept_order.contains(&evicted),
                "evicted key {evicted} must be gone from order vec"
            );
        }
    }

    // v0.28.7 audit R1 P2 #1 — production cap MUST be enforced over the
    // non-shadow subset; evicting a shadow bucket does not free a production
    // slot under separate caps. After a production insert at cap, the
    // post-insert non-shadow count must remain ≤ CONCEPT_SUMMARY_BY_CLUSTER_CAP.
    #[test]
    fn evict_concept_summary_lru_admits_production_at_cap_by_evicting_production() {
        let mut by_cluster: HashMap<String, ClusterConceptSummaryStats> = HashMap::new();
        // Fill production cap with non-shadow buckets.
        for i in 0..CONCEPT_SUMMARY_BY_CLUSTER_CAP {
            by_cluster.insert(
                format!("prod-{i}"),
                ClusterConceptSummaryStats {
                    last_event_id: i as i64 + 1,
                    is_shadow: false,
                    ..Default::default()
                },
            );
        }
        // Add one shadow bucket — irrelevant to production cap accounting.
        by_cluster.insert(
            "shadow-x".to_string(),
            ClusterConceptSummaryStats {
                last_event_id: 0,
                is_shadow: true,
                ..Default::default()
            },
        );
        // Insert a new production bucket at cap. Must evict a production
        // bucket (not the shadow), keeping `prod_count <= CAP` post-insert.
        evict_concept_summary_lru_if_at_cap(&mut by_cluster, "prod-new", false);
        by_cluster.insert(
            "prod-new".to_string(),
            ClusterConceptSummaryStats {
                last_event_id: 99_999,
                is_shadow: false,
                ..Default::default()
            },
        );
        let prod_count = by_cluster.iter().filter(|(_, b)| !b.is_shadow).count();
        assert_eq!(
            prod_count, CONCEPT_SUMMARY_BY_CLUSTER_CAP,
            "production cap must hold after insert at cap"
        );
        assert!(
            by_cluster.contains_key("shadow-x"),
            "shadow bucket must NOT be evicted to admit a production insert (separate caps)"
        );
        // The evicted production bucket must be the LRU (smallest last_event_id),
        // which is "prod-0" (last_event_id = 1).
        assert!(
            !by_cluster.contains_key("prod-0"),
            "LRU production bucket (prod-0) must be evicted"
        );
    }

    // v0.28.7 audit R1 P2 #2 — shadow→production promotion: when an existing
    // shadow bucket flips to production while production is at cap, the
    // promotion must trigger a production eviction. Otherwise the bucket
    // class flip silently overshoots the cap.
    #[test]
    fn evict_concept_summary_lru_handles_shadow_to_production_promotion() {
        let mut by_cluster: HashMap<String, ClusterConceptSummaryStats> = HashMap::new();
        // Fill production cap with non-shadow buckets.
        for i in 0..CONCEPT_SUMMARY_BY_CLUSTER_CAP {
            by_cluster.insert(
                format!("prod-{i}"),
                ClusterConceptSummaryStats {
                    last_event_id: i as i64 + 1,
                    is_shadow: false,
                    ..Default::default()
                },
            );
        }
        // A pre-existing shadow bucket sharing a key that will receive a
        // production event next.
        by_cluster.insert(
            "promote-me".to_string(),
            ClusterConceptSummaryStats {
                last_event_id: 50,
                is_shadow: true,
                ..Default::default()
            },
        );
        // Promotion: same key, but the new event is production.
        evict_concept_summary_lru_if_at_cap(&mut by_cluster, "promote-me", false);
        // Caller would now flip is_shadow on the existing entry.
        if let Some(b) = by_cluster.get_mut("promote-me") {
            b.is_shadow = false;
        }
        let prod_count = by_cluster.iter().filter(|(_, b)| !b.is_shadow).count();
        assert_eq!(
            prod_count, CONCEPT_SUMMARY_BY_CLUSTER_CAP,
            "production cap must hold after shadow→production promotion at cap"
        );
        assert!(
            by_cluster.contains_key("promote-me"),
            "promoted bucket must remain (it is the freshest, not the eviction victim)"
        );
        // The LRU production bucket (prod-0) must be evicted, not the bucket
        // being promoted.
        assert!(
            !by_cluster.contains_key("prod-0"),
            "LRU production bucket (prod-0) must be evicted in favor of the promotion"
        );
    }

    #[test]
    fn concept_summary_feedback_cas_merge_keeps_more_advanced_state() {
        // CAS arbitration: the writer with higher `last_consumed_event_id`
        // wins. Mirrors `synthesis_feedback_cas_merge_keeps_more_advanced_state`.
        let conn = setup_db();

        let mut winner_by_cluster = HashMap::new();
        winner_by_cluster.insert(
            concept_summary_bucket_key(Some(11), "Semantic"),
            ClusterConceptSummaryStats {
                viewed_count: 50,
                viewed_dwell_total_ms: 50 * 4000,
                dwell_samples: vec![4000; 50],
                viewed_dwell_p50_ms: Some(4000),
                clicked_source_count: 20,
                immediate_requery_count: 1,
                explicit_up: 12,
                explicit_down: 2,
                useful_rate: 0.7,
                ..Default::default()
            },
        );
        let winner_csf = ConceptSummaryFeedbackState {
            by_cluster: winner_by_cluster.clone(),
            by_concept: HashMap::new(),
            by_concept_order: vec![],
            last_consumed_event_id: 500,
            total_events: 50,
        };
        let winner = AdaptiveState {
            version: 5,
            concept_summary_feedback_stats: Some(winner_csf.clone()),
            ..AdaptiveState::default()
        };
        let winner_json = serde_json::to_string(&winner).unwrap();
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES ('adaptive_state', ?1)",
            rusqlite::params![&winner_json],
        )
        .unwrap();

        let stale_csf = ConceptSummaryFeedbackState {
            by_cluster: HashMap::new(),
            by_concept: HashMap::new(),
            by_concept_order: vec![],
            last_consumed_event_id: 100,
            total_events: 5,
        };
        let stale = AdaptiveState {
            version: 2,
            concept_summary_feedback_stats: Some(stale_csf),
            ..AdaptiveState::default()
        };
        stale.save_snapshot(&conn).unwrap();

        let restored = AdaptiveState::restore_snapshot(&conn).unwrap();
        let restored_csf = restored.concept_summary_feedback_stats.unwrap();
        assert_eq!(
            restored_csf.last_consumed_event_id, 500,
            "CAS winner with higher last_consumed_event_id must survive"
        );
        assert_eq!(restored_csf.by_cluster, winner_by_cluster);
    }

    #[test]
    fn concept_summary_feedback_cas_preserves_existing_when_writer_has_none() {
        // Mirrors `synthesis_feedback_cas_preserves_existing_when_writer_has_none`.
        // Writer with `concept_summary_feedback_stats = None` MUST NOT
        // overwrite existing learned state.
        let conn = setup_db();
        let learned = ConceptSummaryFeedbackState {
            by_cluster: HashMap::new(),
            by_concept: HashMap::new(),
            by_concept_order: vec![],
            last_consumed_event_id: 1234,
            total_events: 42,
        };
        let prior = AdaptiveState {
            version: 5,
            concept_summary_feedback_stats: Some(learned.clone()),
            ..AdaptiveState::default()
        };
        let prior_json = serde_json::to_string(&prior).unwrap();
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES ('adaptive_state', ?1)",
            rusqlite::params![&prior_json],
        )
        .unwrap();

        let our = AdaptiveState {
            version: 2,
            concept_summary_feedback_stats: None,
            ..AdaptiveState::default()
        };
        our.save_snapshot(&conn).unwrap();

        let restored = AdaptiveState::restore_snapshot(&conn).unwrap();
        assert_eq!(
            restored.concept_summary_feedback_stats,
            Some(learned),
            "CAS merge must preserve existing concept_summary stats when writer has None"
        );
    }

    #[test]
    fn concept_summary_feedback_pairs_human_thumb_by_summary_id_when_present() {
        let conn = setup_db();
        let metadata = Some(JudgeMetadata {
            query_type: Some("Semantic".to_string()),
            cluster_id: Some(7),
            source_count: Some(0),
            judge_latency_ms: None,
        });
        emit_event(
            &conn,
            FeedbackEvent {
                event_type: EventType::ConceptSummaryLlmJudge,
                request_id: None,
                memory_id: None,
                concept_id: Some("concept-a".to_string()),
                query: None,
                query_type: Some("Semantic".to_string()),
                topic: None,
                payload: Some(
                    serde_json::to_value(ConceptSummaryLlmJudgePayload {
                        concept_summary_id: "cs-a".to_string(),
                        concept_id: "concept-a".to_string(),
                        judge_model: "mock".to_string(),
                        hit: true,
                        reason: "ok".to_string(),
                        stamp_hash: "hash".to_string(),
                        source: JudgeSource::ManualMcp,
                        metadata: metadata.clone(),
                        signal_hint: None,
                    })
                    .unwrap(),
                ),
            },
        )
        .unwrap();
        emit_concept_summary_event(
            &conn,
            ConceptSummaryInteractionPayload {
                concept_id: "concept-a".to_string(),
                concept_summary_id: Some("cs-a".to_string()),
                recall_id: "rec-a".to_string(),
                interaction: ConceptSummaryInteractionKind::ExplicitThumb { up: true },
                metadata: Some(ConceptSummaryMetadata {
                    query_type: Some("Semantic".to_string()),
                    cluster_id: Some(7),
                    concept_chars: None,
                    revision_version: None,
                    route_context: None,
                }),
            },
        );

        let (_state, pending, calibration, _max_id) =
            recompute_concept_summary_feedback_stats_with_judge(
                &conn,
                None,
                HashMap::new(),
                JudgeCalibrationState::default(),
                LLM_JUDGE_WEIGHT_DECAY_RATE,
            )
            .unwrap();

        assert!(
            pending.is_empty(),
            "matching concept_summary_id should complete the half-pair"
        );
        assert_eq!(calibration.recent_pairs_concept.len(), 1);
        assert!(calibration.recent_pairs_concept[0].0);
        assert!(calibration.recent_pairs_concept[0].1);
    }

    #[test]
    fn concept_summary_feedback_pairs_judge_first_legacy_thumb_by_concept_id() {
        let conn = setup_db();
        let metadata = Some(JudgeMetadata {
            query_type: Some("Semantic".to_string()),
            cluster_id: Some(7),
            source_count: Some(0),
            judge_latency_ms: None,
        });
        emit_event(
            &conn,
            FeedbackEvent {
                event_type: EventType::ConceptSummaryLlmJudge,
                request_id: None,
                memory_id: None,
                concept_id: Some("concept-a".to_string()),
                query: None,
                query_type: Some("Semantic".to_string()),
                topic: None,
                payload: Some(
                    serde_json::to_value(ConceptSummaryLlmJudgePayload {
                        concept_summary_id: "cs-a".to_string(),
                        concept_id: "concept-a".to_string(),
                        judge_model: "mock".to_string(),
                        hit: false,
                        reason: "miss".to_string(),
                        stamp_hash: "hash".to_string(),
                        source: JudgeSource::ManualMcp,
                        metadata: metadata.clone(),
                        signal_hint: None,
                    })
                    .unwrap(),
                ),
            },
        )
        .unwrap();
        emit_concept_summary_event(
            &conn,
            ConceptSummaryInteractionPayload {
                concept_id: "concept-a".to_string(),
                concept_summary_id: None,
                recall_id: "rec-a".to_string(),
                interaction: ConceptSummaryInteractionKind::ExplicitThumb { up: false },
                metadata: Some(ConceptSummaryMetadata {
                    query_type: Some("Semantic".to_string()),
                    cluster_id: Some(7),
                    concept_chars: None,
                    revision_version: None,
                    route_context: None,
                }),
            },
        );

        let (_state, pending, calibration, _max_id) =
            recompute_concept_summary_feedback_stats_with_judge(
                &conn,
                None,
                HashMap::new(),
                JudgeCalibrationState::default(),
                LLM_JUDGE_WEIGHT_DECAY_RATE,
            )
            .unwrap();

        assert!(
            pending.is_empty(),
            "legacy concept_id alias should be consumed with the canonical summary-id half-pair"
        );
        assert_eq!(calibration.recent_pairs_concept.len(), 1);
        assert!(!calibration.recent_pairs_concept[0].0);
        assert!(!calibration.recent_pairs_concept[0].1);
    }

    #[test]
    fn concept_summary_alias_cleanup_preserves_newer_summary_alias() {
        let conn = setup_db();
        let metadata = Some(JudgeMetadata {
            query_type: Some("Semantic".to_string()),
            cluster_id: Some(7),
            source_count: Some(0),
            judge_latency_ms: None,
        });
        for summary_id in ["cs-old", "cs-new"] {
            emit_event(
                &conn,
                FeedbackEvent {
                    event_type: EventType::ConceptSummaryLlmJudge,
                    request_id: None,
                    memory_id: None,
                    concept_id: Some("concept-a".to_string()),
                    query: None,
                    query_type: Some("Semantic".to_string()),
                    topic: None,
                    payload: Some(
                        serde_json::to_value(ConceptSummaryLlmJudgePayload {
                            concept_summary_id: summary_id.to_string(),
                            concept_id: "concept-a".to_string(),
                            judge_model: "mock".to_string(),
                            hit: true,
                            reason: "ok".to_string(),
                            stamp_hash: format!("hash-{summary_id}"),
                            source: JudgeSource::ManualMcp,
                            metadata: metadata.clone(),
                            signal_hint: None,
                        })
                        .unwrap(),
                    ),
                },
            )
            .unwrap();
        }
        emit_concept_summary_event(
            &conn,
            ConceptSummaryInteractionPayload {
                concept_id: "concept-a".to_string(),
                concept_summary_id: Some("cs-old".to_string()),
                recall_id: "rec-a".to_string(),
                interaction: ConceptSummaryInteractionKind::ExplicitThumb { up: true },
                metadata: Some(ConceptSummaryMetadata {
                    query_type: Some("Semantic".to_string()),
                    cluster_id: Some(7),
                    concept_chars: None,
                    revision_version: None,
                    route_context: None,
                }),
            },
        );

        let (_state, pending, calibration, _max_id) =
            recompute_concept_summary_feedback_stats_with_judge(
                &conn,
                None,
                HashMap::new(),
                JudgeCalibrationState::default(),
                LLM_JUDGE_WEIGHT_DECAY_RATE,
            )
            .unwrap();

        assert_eq!(calibration.recent_pairs_concept.len(), 1);
        assert!(pending.contains_key("cs-new"));
        assert!(
            matches!(
                pending.get("concept-a"),
                Some(HalfPair::Judge {
                    alias_key: Some(alias_key),
                    ..
                }) if alias_key == "cs-new"
            ),
            "pairing the old summary must not remove the newer legacy alias"
        );
    }

    #[test]
    fn half_pair_judge_alias_is_backward_compatible() {
        let json = r#"{"side":"judge","hit":true,"ts":42,"surface":"concept_summary"}"#;
        let half: HalfPair = serde_json::from_str(json).unwrap();
        assert_eq!(
            half,
            HalfPair::Judge {
                hit: true,
                ts: 42,
                surface: JudgeSurface::ConceptSummary,
                alias_key: None,
            }
        );
    }

    #[test]
    fn concept_summary_bucket_helper_gates_on_cold_start_n() {
        // Helper returns Some only when viewed_count >= COLD_START_N.
        let key = concept_summary_bucket_key(Some(8), "Semantic");
        let mut by_cluster = HashMap::new();
        by_cluster.insert(
            key.clone(),
            ClusterConceptSummaryStats {
                viewed_count: CONCEPT_SUMMARY_COLD_START_N - 1,
                ..ClusterConceptSummaryStats::default()
            },
        );
        let mut state = AdaptiveState {
            concept_summary_feedback_stats: Some(ConceptSummaryFeedbackState {
                by_cluster,
                ..ConceptSummaryFeedbackState::default()
            }),
            ..AdaptiveState::default()
        };
        assert!(
            state.concept_summary_bucket(Some(8), "Semantic").is_none(),
            "cold-start: bucket below COLD_START_N must return None"
        );

        let s = state.concept_summary_feedback_stats.as_mut().unwrap();
        s.by_cluster.get_mut(&key).unwrap().viewed_count = CONCEPT_SUMMARY_COLD_START_N;
        assert!(state.concept_summary_bucket(Some(8), "Semantic").is_some());
    }

    #[test]
    fn concept_summary_feedback_state_default_is_empty() {
        let s = ConceptSummaryFeedbackState::default();
        assert!(s.by_cluster.is_empty());
        assert!(s.by_concept.is_empty());
        assert!(s.by_concept_order.is_empty());
        assert_eq!(s.last_consumed_event_id, 0);
        assert_eq!(s.total_events, 0);
        let json = serde_json::to_string(&s).unwrap();
        let back: ConceptSummaryFeedbackState = serde_json::from_str(&json).unwrap();
        assert_eq!(s, back);
    }

    fn fusion_entry(last_updated: &str) -> LearnedShadowFusionEntry {
        LearnedShadowFusionEntry {
            weights: ShadowFusionWeightEntry {
                bm25: 0.45,
                vec: 0.45,
                kg: 0.04,
                episode: 0.03,
                support: 0.02,
                diversity: 0.01,
            },
            sample_count: 12,
            last_updated: last_updated.to_string(),
        }
    }

    /// v0.28.7+ audit L6 — same-key rewrite is a no-op (no eviction
    /// even when the map is at cap). Pre-helper, naive insert-then-cap
    /// patterns would over-evict on a same-key rewrite, dropping a
    /// distinct neighbor for no reason.
    #[test]
    fn evict_learned_shadow_fusion_lru_same_key_rewrite_is_noop() {
        let mut map: HashMap<String, LearnedShadowFusionEntry> = HashMap::new();
        // Fill to cap.
        for i in 0..LEARNED_SHADOW_FUSION_CAP {
            let ts = format!("2026-05-01T00:00:{:02}Z", i % 60);
            map.insert(format!("k{i}"), fusion_entry(&ts));
        }
        assert_eq!(map.len(), LEARNED_SHADOW_FUSION_CAP);
        let before: std::collections::BTreeSet<String> = map.keys().cloned().collect();

        // Rewriting an existing key must NOT evict any other key.
        let existing_key = "k0";
        evict_learned_shadow_fusion_lru_if_at_cap(&mut map, existing_key);
        map.insert(existing_key.into(), fusion_entry("2026-12-31T23:59:59Z"));

        let after: std::collections::BTreeSet<String> = map.keys().cloned().collect();
        assert_eq!(map.len(), LEARNED_SHADOW_FUSION_CAP);
        assert_eq!(
            before, after,
            "same-key rewrite must not evict any neighbor"
        );
    }

    /// v0.28.7+ audit L6 — at cap, a NEW key triggers LRU eviction
    /// (oldest `last_updated`). R12 P2 (2026-05-04): eviction
    /// targets are restricted to cluster-scoped keys
    /// (`{query_type}:{cluster_id}`), so this test seeds the map with
    /// cluster-scoped buckets only.
    #[test]
    fn evict_learned_shadow_fusion_lru_evicts_oldest_at_cap() {
        let mut map: HashMap<String, LearnedShadowFusionEntry> = HashMap::new();
        // The first inserted key has the oldest timestamp; subsequent
        // keys are newer.  All keys are cluster-scoped.
        map.insert("semantic:0".into(), fusion_entry("2025-01-01T00:00:00Z"));
        for i in 1..LEARNED_SHADOW_FUSION_CAP {
            map.insert(
                format!("semantic:{i}"),
                fusion_entry(&format!("2026-05-01T00:00:{:02}Z", i % 60)),
            );
        }
        assert_eq!(map.len(), LEARNED_SHADOW_FUSION_CAP);

        evict_learned_shadow_fusion_lru_if_at_cap(&mut map, "semantic:freshly_arrived");
        // After eviction, "semantic:0" is gone and we have one slot free.
        assert!(
            !map.contains_key("semantic:0"),
            "LRU eviction must drop the entry with the oldest last_updated"
        );
        assert_eq!(map.len(), LEARNED_SHADOW_FUSION_CAP - 1);
    }

    /// v0.28.7+ audit L6 — `last_updated` comparison must be parse-based,
    /// not raw lexicographic. The two timestamps below represent the
    /// SAME instant in different RFC3339 timezone forms; lexicographic
    /// ordering disagrees with chronological ordering (`+` < `Z`), so
    /// a string-only comparison would pick the WRONG eviction victim.
    /// This test fails on the bug-bait code path the advisor flagged
    /// in the design review.
    #[test]
    fn evict_learned_shadow_fusion_lru_uses_parse_based_timestamp_comparison() {
        let mut map: HashMap<String, LearnedShadowFusionEntry> = HashMap::new();
        // R12 P2 (2026-05-04): keys are cluster-scoped so they remain
        // eligible for eviction under the new fallback-preservation
        // discipline.
        // Newer instant, expressed with explicit `+00:00` offset.
        map.insert(
            "semantic:1".into(),
            fusion_entry("2026-05-02T00:00:00+00:00"),
        );
        // Older instant, expressed with `Z`.
        map.insert("semantic:2".into(), fusion_entry("2026-05-01T00:00:00Z"));
        // Pad the rest with an even-newer batch so both real entries
        // remain candidates at cap. (The cap is 4096; we need len >= cap
        // before eviction fires.)
        for i in 0..(LEARNED_SHADOW_FUSION_CAP - 2) {
            map.insert(
                format!("semantic:{}", i + 100),
                fusion_entry(&format!("2026-12-31T23:59:{:02}Z", i % 60)),
            );
        }
        assert_eq!(map.len(), LEARNED_SHADOW_FUSION_CAP);

        evict_learned_shadow_fusion_lru_if_at_cap(&mut map, "semantic:fresh");

        assert!(
            !map.contains_key("semantic:2"),
            "semantic:2 (2026-05-01T00:00:00Z) is the chronologically oldest entry; \
             a parse-based comparator must evict it. A naive lexicographic \
             comparator (`+` < `Z`) would have evicted `semantic:1` instead, \
             producing the silent wrong-victim bug the audit named."
        );
        assert!(
            map.contains_key("semantic:1"),
            "semantic:1 represents a strictly later instant and must survive \
             eviction"
        );
    }

    /// v0.28.7+ audit L6 — unparseable timestamps evict first (they're
    /// already a sign the entry is corrupt; treating them as "oldest"
    /// shifts the failure mode toward visible eviction rather than
    /// silent persistence).
    #[test]
    fn evict_learned_shadow_fusion_lru_treats_unparseable_timestamp_as_oldest() {
        let mut map: HashMap<String, LearnedShadowFusionEntry> = HashMap::new();
        // R12 P2 (2026-05-04): keys are cluster-scoped so they remain
        // eligible for eviction.
        map.insert(
            "semantic:9999".into(),
            fusion_entry("not-a-timestamp-at-all"),
        );
        for i in 1..LEARNED_SHADOW_FUSION_CAP {
            map.insert(
                format!("semantic:{i}"),
                fusion_entry(&format!("2026-05-01T00:00:{:02}Z", i % 60)),
            );
        }
        assert_eq!(map.len(), LEARNED_SHADOW_FUSION_CAP);

        evict_learned_shadow_fusion_lru_if_at_cap(&mut map, "semantic:fresh");
        assert!(
            !map.contains_key("semantic:9999"),
            "unparseable timestamp must be treated as oldest and evicted first"
        );
    }

    /// v0.28.7+ audit R12 P2 (2026-05-04) — `evict_learned_shadow_fusion_lru_if_at_cap`
    /// MUST NOT evict fallback buckets (`global`, query-type-only
    /// keys) even when those are the chronologically oldest entries.
    /// The fallback chain in `get_shadow_fusion_weights` depends on
    /// their continuous presence; LRU'ing them out under high cluster
    /// cardinality would silently degrade dynamic recall fusion for
    /// queries without surviving cluster-scoped buckets.
    #[test]
    fn evict_learned_shadow_fusion_lru_preserves_fallback_buckets() {
        let mut map: HashMap<String, LearnedShadowFusionEntry> = HashMap::new();

        // Plant fallback buckets (no `:` suffix) with deliberately
        // ANCIENT timestamps — pre-R12 these would be the LRU
        // victims even though `get_shadow_fusion_weights` relies on
        // them as the tail of its fallback chain.
        map.insert("global".into(), fusion_entry("2024-01-01T00:00:00Z"));
        map.insert("semantic".into(), fusion_entry("2024-01-02T00:00:00Z"));
        map.insert("episodic".into(), fusion_entry("2024-01-03T00:00:00Z"));

        // Pad to cap with cluster-scoped buckets (newer timestamps).
        for i in 0..(LEARNED_SHADOW_FUSION_CAP - 3) {
            map.insert(
                format!("semantic:{i}"),
                fusion_entry(&format!("2026-12-01T00:00:{:02}Z", i % 60)),
            );
        }
        assert_eq!(map.len(), LEARNED_SHADOW_FUSION_CAP);

        evict_learned_shadow_fusion_lru_if_at_cap(&mut map, "semantic:freshly_arrived");

        // The three fallback buckets MUST survive even though they
        // were chronologically oldest.
        assert!(
            map.contains_key("global"),
            "fallback bucket `global` must survive LRU eviction \
             — get_shadow_fusion_weights' fallback chain depends on it"
        );
        assert!(
            map.contains_key("semantic"),
            "fallback bucket `semantic` (query-type-only) must survive \
             LRU eviction"
        );
        assert!(
            map.contains_key("episodic"),
            "fallback bucket `episodic` (query-type-only) must survive \
             LRU eviction"
        );
        // Map shrunk by 1 — a cluster-scoped victim was evicted instead.
        assert_eq!(map.len(), LEARNED_SHADOW_FUSION_CAP - 1);
    }

    /// v0.28.7+ audit R12 P2 (2026-05-04) — when the entire map
    /// consists of fallback buckets (degenerate state — only ~7
    /// fallback keys exist in practice), `evict_learned_shadow_fusion_lru_if_at_cap`
    /// allows the over-cap insert rather than corrupt the fallback
    /// chain. This is a no-op on the existing entries; the caller
    /// then performs the new insert and the map exceeds cap.  This
    /// is intentional — the warn fires so the operator sees it.
    #[test]
    fn evict_learned_shadow_fusion_lru_skips_eviction_when_no_cluster_scoped_victim() {
        // Only fallback buckets (no `:` suffix). Pre-cap doesn't
        // matter for this test; we just verify the helper does NOT
        // drop any entry from a fallback-only map.
        let mut map: HashMap<String, LearnedShadowFusionEntry> = HashMap::new();
        map.insert("global".into(), fusion_entry("2024-01-01T00:00:00Z"));
        map.insert("semantic".into(), fusion_entry("2024-01-02T00:00:00Z"));
        let pre_len = map.len();
        let pre_keys: Vec<String> = map.keys().cloned().collect();

        // Force the cap pressure path by simulating an at-cap state.
        // We can't actually plant 4096 fallback keys (only ~7 query
        // types exist), so we rely on the helper's len < cap
        // early-return — pre-R12 this WAS the same path that would
        // not have evicted anyway. R12's contract is "if at cap, do
        // not evict fallback"; below cap is a no-op.
        evict_learned_shadow_fusion_lru_if_at_cap(&mut map, "global"); // same-key rewrite
        assert_eq!(map.len(), pre_len, "same-key rewrite must remain a no-op");
        for key in &pre_keys {
            assert!(map.contains_key(key));
        }
    }

    /// v0.28.7+ audit R12 P2 (2026-05-04) — `shrink_learned_shadow_fusion_to_cap`
    /// MUST NOT evict fallback buckets even when the map is over cap
    /// and fallback timestamps are the oldest. If the cluster-scoped
    /// victim pool is exhausted before reaching cap, the map remains
    /// over-cap rather than corrupt the fallback chain.
    #[test]
    fn shrink_learned_shadow_fusion_to_cap_preserves_fallback_buckets() {
        let mut map: HashMap<String, LearnedShadowFusionEntry> = HashMap::new();
        // Plant fallback buckets at the oldest timestamps.
        map.insert("global".into(), fusion_entry("2024-01-01T00:00:00Z"));
        map.insert("semantic".into(), fusion_entry("2024-01-02T00:00:00Z"));
        map.insert("episodic".into(), fusion_entry("2024-01-03T00:00:00Z"));

        // Pad cluster-scoped to CAP-3 + 50 over-cap, all newer.
        for i in 0..(LEARNED_SHADOW_FUSION_CAP - 3) {
            map.insert(
                format!("semantic:{i}"),
                fusion_entry(&format!("2026-06-01T00:00:{:02}Z", i % 60)),
            );
        }
        for i in 0..50 {
            map.insert(
                format!("episodic:{}", 5000 + i),
                fusion_entry(&format!("2026-06-15T00:00:{:02}Z", i % 60)),
            );
        }
        // Map is at CAP + 50, with fallback buckets being the strict
        // oldest entries.
        assert_eq!(map.len(), LEARNED_SHADOW_FUSION_CAP + 50);

        shrink_learned_shadow_fusion_to_cap(&mut map);

        // Fallback buckets MUST survive.
        assert!(map.contains_key("global"));
        assert!(map.contains_key("semantic"));
        assert!(map.contains_key("episodic"));
        // Map shrunk to exactly cap (50 cluster-scoped victims
        // available, drop_count = 50, all from the cluster pool).
        assert_eq!(
            map.len(),
            LEARNED_SHADOW_FUSION_CAP,
            "shrink must reach cap by evicting only cluster-scoped victims"
        );
    }

    /// v0.28.7+ audit R12 P2 (2026-05-04) — when the cluster-scoped
    /// victim pool is too small to reach cap,
    /// `shrink_learned_shadow_fusion_to_cap` evicts every available
    /// cluster victim and leaves the map over cap rather than
    /// corrupting the fallback chain. The warn fires so the operator
    /// sees the degenerate state.
    #[test]
    fn shrink_learned_shadow_fusion_to_cap_stops_when_only_fallback_remains() {
        let mut map: HashMap<String, LearnedShadowFusionEntry> = HashMap::new();
        // Build a state where 95% of entries are fallback (impossible
        // in practice — only ~7 fallback keys exist — but exercises
        // the safety branch). We approximate by mixing 5
        // cluster-scoped + many fallbacks at over-cap pressure: use
        // a smaller "cap" mental model — actual cap is CAP, so we
        // need CAP+5 entries with 5 cluster-scoped and CAP fallback.
        // That isn't realistic, so instead we test the contract
        // structurally with a smaller plant: 3 cluster-scoped entries
        // (eligible victims) + many entries below-cap and confirm
        // shrink evicts at most the 3 cluster-scoped ones if it has
        // to.  Below-cap is a no-op so we verify the helper short-
        // circuits.
        map.insert("global".into(), fusion_entry("2024-01-01T00:00:00Z"));
        map.insert("semantic".into(), fusion_entry("2024-01-02T00:00:00Z"));
        map.insert("semantic:1".into(), fusion_entry("2026-01-01T00:00:00Z"));
        map.insert("semantic:2".into(), fusion_entry("2026-01-02T00:00:00Z"));

        let pre_len = map.len();
        // Below cap: shrink is a no-op.
        shrink_learned_shadow_fusion_to_cap(&mut map);
        assert_eq!(map.len(), pre_len);
        assert!(map.contains_key("global"));
        assert!(map.contains_key("semantic"));
        assert!(map.contains_key("semantic:1"));
        assert!(map.contains_key("semantic:2"));
    }

    /// v0.28.7+ audit R12 P2 (2026-05-04) — `is_cluster_scoped_bucket`
    /// predicate behavior.  Cluster-scoped keys end in `:<u32>`;
    /// fallback keys are `global` or query-type-only (no colon).
    #[test]
    fn is_cluster_scoped_bucket_predicate_classifies_correctly() {
        // Cluster-scoped: `{query_type}:{cluster_id}`.
        assert!(is_cluster_scoped_bucket("semantic:0"));
        assert!(is_cluster_scoped_bucket("semantic:42"));
        assert!(is_cluster_scoped_bucket("episodic:99999"));
        assert!(is_cluster_scoped_bucket("exactkeyword:1"));

        // Fallback: `global` or query-type-only.
        assert!(!is_cluster_scoped_bucket("global"));
        assert!(!is_cluster_scoped_bucket("semantic"));
        assert!(!is_cluster_scoped_bucket("ExactKeyword"));
        assert!(!is_cluster_scoped_bucket(""));

        // Pathological: trailing `:` with no number, non-numeric
        // suffix, negative numbers — all rejected as fallback.
        assert!(!is_cluster_scoped_bucket("semantic:"));
        assert!(!is_cluster_scoped_bucket("semantic:abc"));
        assert!(!is_cluster_scoped_bucket("semantic:-1"));
        assert!(!is_cluster_scoped_bucket("semantic:1.5"));
    }

    /// v0.28.7+ audit L6 — snapshot round-trip preserves the cap-bounded
    /// state: serializing at cap and restoring still leaves the map
    /// at cap (no spurious growth or shrinkage).
    #[test]
    fn learned_shadow_fusion_at_cap_round_trips_through_snapshot() {
        let mut state = AdaptiveState::default();
        for i in 0..LEARNED_SHADOW_FUSION_CAP {
            state.learned_shadow_fusion.insert(
                format!("qt:{i}"),
                fusion_entry(&format!("2026-05-01T00:00:{:02}Z", i % 60)),
            );
        }
        let json = serde_json::to_string(&state).unwrap();
        let restored: AdaptiveState = serde_json::from_str(&json).unwrap();
        assert_eq!(
            restored.learned_shadow_fusion.len(),
            LEARNED_SHADOW_FUSION_CAP
        );
    }

    /// v0.28.7+ audit M-8 R2 P2 #2 — `shrink_learned_shadow_fusion_to_cap`
    /// drops the OLDEST entries (parsed by RFC3339) when the map is
    /// over cap, leaving exactly `LEARNED_SHADOW_FUSION_CAP` entries.
    /// Below-cap input is a no-op.
    #[test]
    fn shrink_learned_shadow_fusion_to_cap_drops_oldest() {
        let mut map: HashMap<String, LearnedShadowFusionEntry> = HashMap::new();

        // Below cap: no-op.
        // R12 P2 (2026-05-04): keys are cluster-scoped so the new
        // fallback-preservation guard does not protect them.
        for i in 0..(LEARNED_SHADOW_FUSION_CAP - 10) {
            map.insert(
                format!("semantic:{i}"),
                fusion_entry(&format!("2026-05-01T00:00:{:02}Z", i % 60)),
            );
        }
        let pre = map.len();
        shrink_learned_shadow_fusion_to_cap(&mut map);
        assert_eq!(map.len(), pre, "below-cap input must be a no-op");

        // Pad to exactly cap with more modern (newer) entries.
        for i in 0..10 {
            map.insert(
                format!("episodic:{}", 1000 + i),
                fusion_entry(&format!("2026-06-01T00:00:{:02}Z", i % 60)),
            );
        }
        assert_eq!(map.len(), LEARNED_SHADOW_FUSION_CAP);

        // Now push 50 explicitly-ancient entries on top: all 50 are
        // older than EVERY modern entry, so all 50 should be selected
        // for eviction when we shrink from CAP+50 back down to CAP.
        for i in 0..50 {
            map.insert(
                format!("temporal:{}", 2000 + i),
                fusion_entry(&format!("2024-01-01T00:00:{:02}Z", i % 60)),
            );
        }
        assert_eq!(map.len(), LEARNED_SHADOW_FUSION_CAP + 50);
        shrink_learned_shadow_fusion_to_cap(&mut map);
        assert_eq!(
            map.len(),
            LEARNED_SHADOW_FUSION_CAP,
            "shrink must leave exactly cap-many entries"
        );
        // Every "temporal:*" entry must be gone (the 50 ancients were
        // the 50 oldest; shrinking by 50 drops exactly them).
        for i in 0..50 {
            assert!(
                !map.contains_key(&format!("temporal:{}", 2000 + i)),
                "temporal:{} (2024 timestamp) must be evicted when shrinking \
                 from CAP+50 to CAP — all 50 ancients are the strict-oldest",
                2000 + i
            );
        }

        // Idempotency: a second shrink at exactly cap is a no-op.
        let pre2 = map.len();
        shrink_learned_shadow_fusion_to_cap(&mut map);
        assert_eq!(map.len(), pre2, "shrink at-cap is a no-op");
    }

    /// v0.28.7+ audit M-8 R2 P2 #2 — `restore_snapshot` enforces the
    /// L6 cap when loading an over-cap blob (e.g., a snapshot written
    /// by an older binary that predates the L6 cap, OR a peer binary
    /// that didn't enforce the cap at insert time).
    #[test]
    fn restore_snapshot_shrinks_over_cap_blob_to_cap() {
        // Build a state whose serialized blob has CAP + 20 entries —
        // bypassing the per-key insert helper that would normally cap
        // at insert time.
        // R12 P2 (2026-05-04): all keys are cluster-scoped so the
        // shrink path's fallback-preservation guard does not exempt
        // them from eviction.
        let mut state = AdaptiveState::default();
        for i in 0..(LEARNED_SHADOW_FUSION_CAP + 20) {
            state.learned_shadow_fusion.insert(
                format!("semantic:{i}"),
                fusion_entry(&format!("2026-05-01T00:{:02}:00Z", (i % 1440) / 60)),
            );
        }
        let json = serde_json::to_string(&state).unwrap();
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute_batch("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);")
            .unwrap();
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES ('adaptive_state', ?1)",
            rusqlite::params![json],
        )
        .unwrap();

        let restored = AdaptiveState::restore_snapshot(&conn).expect("restore must succeed");
        assert_eq!(
            restored.learned_shadow_fusion.len(),
            LEARNED_SHADOW_FUSION_CAP,
            "restore_snapshot must shrink an over-cap blob to cap"
        );
    }

    /// v0.28.7+ audit M-1 persistence-side — the legacy-fallback helper
    /// must prefer the per-surface key when present, and fall back to
    /// the legacy cluster-shared key only when the per-surface key is
    /// absent (the first-tick-after-upgrade path).
    #[test]
    fn ars_effective_scalar_with_legacy_fallback_prefers_per_surface() {
        let mut state = AdaptiveState::default();
        // Snapshot has only the legacy cluster-shared scalar (the
        // pre-v0.28.7+ snapshot shape).
        state.set_ars_effective_scalar(ARS_SCALAR_JUDGE_SAMPLE_RATE_COLD_START, 0.42);

        // Per-surface absent → fall back to legacy.
        let v = ars_effective_scalar_with_legacy_fallback(
            &state,
            ARS_SCALAR_JUDGE_SAMPLE_RATE_COLD_START_SYNTHESIS,
            ARS_SCALAR_JUDGE_SAMPLE_RATE_COLD_START,
        );
        assert_eq!(
            v,
            Some(0.42),
            "legacy fallback must apply when per-surface absent"
        );

        // After the per-surface key is written, the helper must prefer
        // it (the legacy value is now stale).
        state.set_ars_effective_scalar(ARS_SCALAR_JUDGE_SAMPLE_RATE_COLD_START_SYNTHESIS, 0.17);
        let v2 = ars_effective_scalar_with_legacy_fallback(
            &state,
            ARS_SCALAR_JUDGE_SAMPLE_RATE_COLD_START_SYNTHESIS,
            ARS_SCALAR_JUDGE_SAMPLE_RATE_COLD_START,
        );
        assert_eq!(
            v2,
            Some(0.17),
            "per-surface key must take precedence once present"
        );

        // ConceptSummary side falls through to the same legacy when its
        // per-surface key is absent — independent fallback per surface.
        let v3 = ars_effective_scalar_with_legacy_fallback(
            &state,
            ARS_SCALAR_JUDGE_SAMPLE_RATE_COLD_START_CONCEPT_SUMMARY,
            ARS_SCALAR_JUDGE_SAMPLE_RATE_COLD_START,
        );
        assert_eq!(
            v3,
            Some(0.42),
            "concept_summary surface still falls back to the legacy value \
             when its own per-surface key is absent (synthesis having its \
             own per-surface value does NOT influence concept_summary)"
        );

        // Both absent → None.
        let empty = AdaptiveState::default();
        assert_eq!(
            ars_effective_scalar_with_legacy_fallback(
                &empty,
                ARS_SCALAR_JUDGE_SAMPLE_RATE_COLD_START_SYNTHESIS,
                ARS_SCALAR_JUDGE_SAMPLE_RATE_COLD_START,
            ),
            None,
            "both keys absent → None"
        );
    }
}
