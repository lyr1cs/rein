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
}
