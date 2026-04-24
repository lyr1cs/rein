//! Adaptive Engine: event sourcing, per-consumer offsets, and AdaptiveState cache.
//! Foundation module (M1) for the unified self-adaptive engine.

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

/// v0.24 ARS: learned statistics for concept living-summary refresh.
/// Populated by the slow-channel from post-refresh feedback. `None` on
/// fresh install; helper thresholds fall back to bootstrap constants
/// until `count >= CONCEPT_REFRESH_MIN_SAMPLES`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConceptRefreshStats {
    /// Number of concepts contributing to the percentile. Callers must
    /// compare against [`CONCEPT_REFRESH_MIN_SAMPLES`] before trusting.
    pub count: usize,
    /// p75 of revisions-since-last-summary among refreshed concepts.
    pub revision_p75: u32,
    /// p50 of time-since-last-summary (seconds) among refreshed concepts.
    pub age_p50_secs: i64,
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

    /// v0.24 ARS: age-since-last-summary threshold in seconds. Same
    /// fallback pattern as [`Self::concept_refresh_revision_threshold`].
    pub fn concept_refresh_age_threshold_secs(&self) -> i64 {
        self.concept_refresh_stats
            .as_ref()
            .filter(|s| s.count >= CONCEPT_REFRESH_MIN_SAMPLES)
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
}
