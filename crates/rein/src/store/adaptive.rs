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
use std::collections::HashMap;

// ── Event Types ──────────────────────────────────────────────────────────────

/// All feedback event types emitted by the system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventType {
    RecallComplete,   // recall returned results (includes full candidate set)
    RecallAccess,     // agent used a recalled memory
    RecallMiss,       // recall returned but not accessed (record-only)
    RecallRetry,      // same query recalled again in session
    Store,            // new memory stored
    StoreQuickRecall, // memory recalled shortly after being stored
    Forget,           // agent explicitly forgot/deprecated
    Refine,           // concept refined/superseded
    SessionEnd,       // hook_stop fired
    ParamUpdate,      // slow-channel parameter update (audit trail)
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

    /// M4: Current cluster version (incremented on each reclustering).
    pub cluster_version: u64,

    /// M4: Memory → cluster assignment.
    pub memory_clusters: HashMap<String, u32>,

    /// M5: Tier boundary thresholds.
    pub hot_threshold: f64,
    pub cold_threshold: f64,

    /// A1: Per-cluster dedup thresholds (replaces fixed 0.70).
    /// Key = cluster_id, Value = similarity threshold for that cluster.
    /// Computed from intra-cluster pairwise similarity distribution (P90).
    #[serde(default)]
    pub dedup_thresholds: HashMap<u32, f32>,

    /// A1: Global (fallback) dedup threshold when no cluster-specific value exists.
    #[serde(default = "default_global_dedup_threshold")]
    pub global_dedup_threshold: f32,

    /// M4 incremental: version stamp for cluster centroids stored in `cluster_centroids` table.
    /// Callers compare this against what they last loaded to detect staleness.
    #[serde(default)]
    pub centroid_version: u64,

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
    /// highest event id whose threshold-nudge effect is already in the
    /// durable snapshot of `global_dedup_threshold`. Caller of M6 filters
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
    /// Build bucket key from query_type and optional cluster_id.
    pub fn bucket_key(query_type: &str, cluster_id: Option<u32>) -> String {
        let query_type = query_type.to_lowercase();
        match cluster_id {
            Some(c) => format!("{query_type}:{c}"),
            None => query_type,
        }
    }

    /// Get learned alpha for a query type and optional cluster, with fallback chain.
    pub fn get_alpha(&self, query_type: &str, cluster_id: Option<u32>) -> Option<f32> {
        let legacy_key = query_type.to_string();
        // Try specific bucket first
        if let Some(cluster) = cluster_id {
            let key = Self::bucket_key(query_type, Some(cluster));
            if let Some(entry) = self.learned_alpha.get(&key) {
                if entry.sample_count >= 10 {
                    return Some(entry.value as f32);
                }
            }
            let legacy_cluster_key = format!("{legacy_key}:{cluster}");
            if let Some(entry) = self.learned_alpha.get(&legacy_cluster_key) {
                if entry.sample_count >= 10 {
                    return Some(entry.value as f32);
                }
            }
        }
        // Fall back to query-type level
        let key = Self::bucket_key(query_type, None);
        if let Some(entry) = self.learned_alpha.get(&key) {
            if entry.sample_count >= 10 {
                return Some(entry.value as f32);
            }
        }
        if let Some(entry) = self.learned_alpha.get(&legacy_key) {
            if entry.sample_count >= 10 {
                return Some(entry.value as f32);
            }
        }
        if let Some(entry) = self.learned_alpha.get("global") {
            if entry.sample_count >= 10 {
                return Some(entry.value as f32);
            }
        }
        None
    }

    /// Get dedup threshold for a cluster, with fallback to global threshold.
    pub fn get_dedup_threshold(&self, cluster_id: Option<u32>) -> f32 {
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
        let json = serde_json::to_string(self).map_err(ReinError::Serialization)?;

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

                    // Cluster-scoped state: if we ran a recluster at the same or newer
                    // version, replace wholesale to avoid resurrecting entries deleted
                    // during recluster.  Two concurrent reclusters at the same version
                    // would each increment once, so >= catches both cases.
                    if self.cluster_version >= current.cluster_version {
                        current.memory_clusters = self.memory_clusters.clone();
                        current.dedup_thresholds = self.dedup_thresholds.clone();
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
                    } else {
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
                    }

                    // Merge learned_alpha (non-cluster keys): prefer newer timestamp
                    for (key, our_entry) in &self.learned_alpha {
                        if key.contains(':') {
                            continue; // handled above based on cluster_version
                        }
                        let dominated = current
                            .learned_alpha
                            .get(key)
                            .is_some_and(|theirs| theirs.last_updated >= our_entry.last_updated);
                        if !dominated {
                            current.learned_alpha.insert(key.clone(), our_entry.clone());
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
                    current.alpha_optimizer_last_id =
                        current.alpha_optimizer_last_id.max(self.alpha_optimizer_last_id);
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
                    match (
                        &self.synthesis_feedback_stats,
                        &current.synthesis_feedback_stats,
                    ) {
                        (Some(mine), Some(theirs)) => {
                            if mine.last_consumed_event_id > theirs.last_consumed_event_id {
                                current.synthesis_feedback_stats = Some(mine.clone());
                            }
                        }
                        (Some(_), None) => {
                            current.synthesis_feedback_stats =
                                self.synthesis_feedback_stats.clone();
                        }
                        (None, _) => { /* keep current */ }
                    }
                    // v0.27 ARS Cap A feedback loop (Track 1): mirror the
                    // synthesis_feedback_stats arm above. Same arbitration
                    // shape — event-id MAX wins, writer's `None` does not
                    // overwrite existing learned state. Round-3 HIGH from
                    // v0.24 generalised; round-4 HIGH from v0.26 mirrored.
                    match (
                        &self.concept_summary_feedback_stats,
                        &current.concept_summary_feedback_stats,
                    ) {
                        (Some(mine), Some(theirs)) => {
                            if mine.last_consumed_event_id > theirs.last_consumed_event_id {
                                current.concept_summary_feedback_stats = Some(mine.clone());
                            }
                        }
                        (Some(_), None) => {
                            current.concept_summary_feedback_stats =
                                self.concept_summary_feedback_stats.clone();
                        }
                        (None, _) => { /* keep current */ }
                    }
                    current.version = db_version + 1;

                    let merged_json =
                        serde_json::to_string(&current).map_err(ReinError::Serialization)?;

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
        serde_json::from_str(&json).ok()
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
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
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
        let mut revs: Vec<u32> = stats.samples.iter().map(|s| s.revisions_since_last).collect();
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
    /// Derived metric, recomputed on every consumer pass.
    pub useful_rate: f64,
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

    let numerator = SYNTHESIS_W_VIEW * dwell_pct
        + SYNTHESIS_W_CLICK * click_rate
        + SYNTHESIS_W_THUMB * thumb_rate
        - SYNTHESIS_W_REQUERY * requery_rate;
    let denom = SYNTHESIS_W_VIEW + SYNTHESIS_W_CLICK + SYNTHESIS_W_THUMB + SYNTHESIS_W_REQUERY;
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

        // F-11 hard cap: drop event if creating a new bucket would push
        // by_cluster past the cap. Existing buckets continue to receive
        // updates so legitimate signal isn't lost.
        if !state.by_cluster.contains_key(&bucket_key)
            && state.by_cluster.len() >= SYNTHESIS_BY_CLUSTER_CAP
        {
            tracing::warn!(
                cluster_id = ?cluster_id,
                query_type = %query_type,
                cap = SYNTHESIS_BY_CLUSTER_CAP,
                "synthesis_feedback: by_cluster cap reached; dropping new bucket event"
            );
            // Still bump total_events so the consumer offset advances and
            // we don't replay this event forever.
            state.total_events = state.total_events.saturating_add(1);
            continue;
        }

        // Per-bucket fold.
        let bucket = state.by_cluster.entry(bucket_key.clone()).or_default();
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
    pub fn synthesis_bucket(
        &self,
        cluster_id: Option<i64>,
        query_type: &str,
    ) -> Option<&ClusterSynthesisStats> {
        let state = self.synthesis_feedback_stats.as_ref()?;
        let key = synthesis_bucket_key(cluster_id, query_type);
        state
            .by_cluster
            .get(&key)
            .filter(|s| s.viewed_count >= SYNTHESIS_COLD_START_N)
    }
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
pub const CONCEPT_SUMMARY_BY_CLUSTER_CAP: usize = 4096;

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
    /// Derived metric, recomputed on every consumer pass.
    pub useful_rate: f64,
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

    let numerator = CONCEPT_SUMMARY_W_VIEW * dwell_pct
        + CONCEPT_SUMMARY_W_CLICK * click_rate
        + CONCEPT_SUMMARY_W_THUMB * thumb_rate
        - CONCEPT_SUMMARY_W_REQUERY * requery_rate;
    let denom = CONCEPT_SUMMARY_W_VIEW
        + CONCEPT_SUMMARY_W_CLICK
        + CONCEPT_SUMMARY_W_THUMB
        + CONCEPT_SUMMARY_W_REQUERY;
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
        let cluster_id = metadata.cluster_id;
        let raw_qtype = metadata.query_type.as_deref().unwrap_or("");
        // Whitelist normalize: clamp non-allowed query_types to "unknown" so
        // malicious clients can't multiplicatively grow the bucket space.
        let query_type = if CONCEPT_SUMMARY_ALLOWED_QUERY_TYPES.contains(&raw_qtype) {
            raw_qtype.to_string()
        } else {
            "unknown".to_string()
        };
        let bucket_key = concept_summary_bucket_key(cluster_id, &query_type);

        // Hard cap: drop event if creating a new bucket would push
        // by_cluster past the cap. Existing buckets continue to receive
        // updates so legitimate signal isn't lost.
        if !state.by_cluster.contains_key(&bucket_key)
            && state.by_cluster.len() >= CONCEPT_SUMMARY_BY_CLUSTER_CAP
        {
            tracing::warn!(
                cluster_id = ?cluster_id,
                query_type = %query_type,
                cap = CONCEPT_SUMMARY_BY_CLUSTER_CAP,
                "concept_summary_feedback: by_cluster cap reached; dropping new bucket event"
            );
            // Still bump total_events so the consumer offset advances and
            // we don't replay this event forever.
            state.total_events = state.total_events.saturating_add(1);
            continue;
        }

        // Per-bucket fold.
        let bucket = state.by_cluster.entry(bucket_key.clone()).or_default();
        match &payload.interaction {
            ConceptSummaryInteractionKind::Viewed { dwell_ms } => {
                bucket.viewed_count = bucket.viewed_count.saturating_add(1);
                bucket.viewed_dwell_total_ms =
                    bucket.viewed_dwell_total_ms.saturating_add(*dwell_ms);
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
        touched_buckets.insert(bucket_key);

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
    /// [`CONCEPT_SUMMARY_COLD_START_N`] viewed samples (cold-start
    /// fallback otherwise — caller falls back to the global Cap A
    /// `[ars].concept_summary_enabled` flag). Mirrors [`Self::synthesis_bucket`].
    pub fn concept_summary_bucket(
        &self,
        cluster_id: Option<i64>,
        query_type: &str,
    ) -> Option<&ClusterConceptSummaryStats> {
        let state = self.concept_summary_feedback_stats.as_ref()?;
        let key = concept_summary_bucket_key(cluster_id, query_type);
        state
            .by_cluster
            .get(&key)
            .filter(|s| s.viewed_count >= CONCEPT_SUMMARY_COLD_START_N)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

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
        let alpha = restored.get_alpha("semantic", None);
        assert!(alpha.is_some());
        assert!((alpha.unwrap() - 0.35).abs() < 0.01);
    }

    #[test]
    fn test_alpha_fallback_chain() {
        let mut state = AdaptiveState::default();

        // No data → None
        assert!(state.get_alpha("semantic", Some(1)).is_none());

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
        let alpha = state.get_alpha("semantic", Some(1)).unwrap();
        assert!((alpha - 0.4).abs() < 0.01);

        // Give cluster enough samples
        state
            .learned_alpha
            .get_mut("semantic:1")
            .unwrap()
            .sample_count = 12;
        let alpha = state.get_alpha("semantic", Some(1)).unwrap();
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
        let alpha = state.get_alpha("temporal", None).unwrap();
        assert!((alpha - 0.55).abs() < 0.01);

        state.learned_alpha.insert(
            "Temporal".into(),
            LearnedAlphaEntry {
                value: 0.8,
                sample_count: 20,
                last_updated: String::new(),
            },
        );
        let alpha = state.get_alpha("Temporal", None).unwrap();
        assert!((alpha - 0.8).abs() < 0.01);
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
            emit_event(&conn, FeedbackEvent {
                event_type: EventType::Store,
                request_id: Some(format!("r{i}")),
                memory_id: Some(format!("m{i}")),
                concept_id: None,
                query: None, query_type: None, topic: None,
                payload: None,
            }).unwrap();
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
            emit_event(&conn, FeedbackEvent {
                event_type: EventType::Store,
                request_id: Some(format!("r{i}")),
                memory_id: None, concept_id: None,
                query: None, query_type: None, topic: None, payload: None,
            }).unwrap();
        }

        // Commit two consumers in one batch.
        commit_offset(&conn, &[("c_a", 1), ("c_b", 2)]).unwrap();
        let off_a: i64 = conn.query_row(
            "SELECT last_event_id FROM consumer_offsets WHERE consumer = 'c_a'",
            [], |r| r.get(0),
        ).unwrap();
        let off_b: i64 = conn.query_row(
            "SELECT last_event_id FROM consumer_offsets WHERE consumer = 'c_b'",
            [], |r| r.get(0),
        ).unwrap();
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
        let off: i64 = conn.query_row(
            "SELECT last_event_id FROM consumer_offsets WHERE consumer = 'c1'",
            [], |r| r.get(0),
        ).unwrap();
        assert_eq!(off, 10);
    }

    #[test]
    fn commit_offset_empty_batch_is_noop() {
        let conn = setup_db();
        commit_offset(&conn, &[]).unwrap();
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM consumer_offsets",
            [], |r| r.get(0),
        ).unwrap();
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
        let (stats2, max_id2) = recompute_concept_refresh_stats(&conn, Some(stats.clone())).unwrap();
        assert_eq!(
            stats2.count, 2,
            "replay-safety: events with id ≤ last_consumed_event_id are skipped"
        );
        assert_eq!(stats2.samples, stats.samples, "no double-append on replay");
        assert_eq!(max_id2, Some(2), "max_id still reported so caller can re-attempt commit");

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
                RefreshSample { revisions_since_last: 5, age_secs_since_last: 1000, first_refresh: false },
                RefreshSample { revisions_since_last: 7, age_secs_since_last: 2000, first_refresh: false },
                RefreshSample { revisions_since_last: 9, age_secs_since_last: 3600, first_refresh: false },
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
                RefreshSample { revisions_since_last: 6, age_secs_since_last: 1200, first_refresh: false },
                RefreshSample { revisions_since_last: 10, age_secs_since_last: 1800, first_refresh: false },
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
                query_type: payload
                    .metadata
                    .as_ref()
                    .and_then(|m| m.query_type.clone()),
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
        };
        let bad_rate = compute_useful_rate(&bad);
        assert!(bad_rate >= 0.0, "useful_rate must clamp at 0.0");
        assert!(
            bad_rate < 0.5,
            "bad path useful_rate={bad_rate} should fall below 0.5"
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
                    SynthesisInteractionKind::Viewed { dwell_ms: (i + 1) as u64 },
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
        assert_eq!(*bucket.dwell_samples.first().unwrap(), (overflow + 1) as u64);
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
        assert!(state.synthesis_bucket(Some(8), "Semantic").is_none(),
            "cold-start: bucket below SYNTHESIS_COLD_START_N must return None");

        // At cold-start threshold → Some.
        let s = state.synthesis_feedback_stats.as_mut().unwrap();
        s.by_cluster
            .get_mut(&key)
            .unwrap()
            .viewed_count = SYNTHESIS_COLD_START_N;
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

    fn emit_concept_summary_event(
        conn: &Connection,
        payload: ConceptSummaryInteractionPayload,
    ) {
        emit_event(
            conn,
            FeedbackEvent {
                event_type: EventType::ConceptSummaryInteraction,
                request_id: None,
                memory_id: None,
                concept_id: Some(payload.concept_id.clone()),
                query: None,
                query_type: payload
                    .metadata
                    .as_ref()
                    .and_then(|m| m.query_type.clone()),
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
            recall_id: format!("recall-{concept_id}"),
            interaction: kind,
            metadata: Some(ConceptSummaryMetadata {
                query_type: query_type.map(|s| s.to_string()),
                cluster_id,
                concept_chars: None,
                revision_version: None,
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
        assert_eq!(state2.total_events, 2, "replay-safety: events with id <= last_consumed_event_id are skipped");
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
    fn recompute_concept_summary_feedback_bucket_cap_drops_new_buckets() {
        // Required test #2: bucket cap behavior.
        // We can't realistically emit 4096 events in a unit test, so we
        // pre-populate state past the cap and verify that a new bucket
        // event drops with `total_events` still incrementing.
        let conn = setup_db();
        let mut by_cluster = HashMap::new();
        for i in 0..CONCEPT_SUMMARY_BY_CLUSTER_CAP {
            by_cluster.insert(
                concept_summary_bucket_key(Some(i as i64), "Semantic"),
                ClusterConceptSummaryStats::default(),
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
        let (state, max_id) =
            recompute_concept_summary_feedback_stats(&conn, Some(prior)).unwrap();
        assert_eq!(state.by_cluster.len(), CONCEPT_SUMMARY_BY_CLUSTER_CAP);
        // New bucket NOT inserted, but total_events bumps so the offset
        // advances and we don't replay this event forever.
        let cap_key = concept_summary_bucket_key(Some(99_999), "Semantic");
        assert!(!state.by_cluster.contains_key(&cap_key));
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
                    ConceptSummaryInteractionKind::Viewed { dwell_ms: (i + 1) as u64 },
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
        assert_eq!(*bucket.dwell_samples.first().unwrap(), (overflow + 1) as u64);
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
}
