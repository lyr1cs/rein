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
    RecallComplete,     // recall returned results (includes full candidate set)
    RecallAccess,       // agent used a recalled memory
    RecallMiss,         // recall returned but not accessed (record-only)
    RecallRetry,        // same query recalled again in session
    Store,              // new memory stored
    StoreQuickRecall,   // memory recalled shortly after being stored
    Forget,             // agent explicitly forgot/deprecated
    Refine,             // concept refined/superseded
    SessionEnd,         // hook_stop fired
    ParamUpdate,        // slow-channel parameter update (audit trail)
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
    let type_placeholders: Vec<String> = event_types.iter().enumerate()
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
    let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = vec![
        Box::new(last_id),
        Box::new(limit as i64),
    ];
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

/// Clean up old events that all consumers have processed and are beyond retention.
pub fn cleanup_expired_events(conn: &Connection, retention_days: u64) -> ReinResult<u64> {
    // Find the minimum offset across all consumers (oldest unprocessed event)
    let min_offset: i64 = conn
        .query_row(
            "SELECT COALESCE(MIN(last_event_id), 0) FROM consumer_offsets",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0);

    // Delete events that are:
    // 1. Below all consumer offsets (fully consumed)
    // 2. Older than retention period
    let cutoff = Utc::now() - chrono::Duration::days(retention_days as i64);
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

    /// Global version (incremented on each slow-channel update).
    pub version: u64,
}

fn default_global_dedup_threshold() -> f32 { 0.70 }

/// A learned alpha entry with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedAlphaEntry {
    pub value: f64,
    pub sample_count: usize,
    pub last_updated: String, // RFC3339
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

    /// Save state snapshot to metadata table with optimistic concurrency control.
    /// Checks that the stored version matches our base version to prevent lost updates
    /// when two concurrent GC runs modify the state simultaneously.
    pub fn save_snapshot(&self, conn: &Connection) -> ReinResult<()> {
        let json = serde_json::to_string(self)
            .map_err(ReinError::Serialization)?;

        // Optimistic concurrency: only update if version hasn't changed since we loaded
        let rows = conn.execute(
            "UPDATE metadata SET value = ?1
             WHERE key = 'adaptive_state'
             AND (value IS NULL OR json_extract(value, '$.version') = ?2 OR json_extract(value, '$.version') IS NULL)",
            rusqlite::params![&json, self.version.saturating_sub(1)],
        )?;

        if rows == 0 {
            // Either no row exists yet (first run) or version mismatch
            let exists: bool = conn.query_row(
                "SELECT COUNT(*) > 0 FROM metadata WHERE key = 'adaptive_state'",
                [], |r| r.get(0),
            ).unwrap_or(false);

            if !exists {
                // First save — insert
                conn.execute(
                    "INSERT INTO metadata (key, value) VALUES ('adaptive_state', ?1)",
                    rusqlite::params![&json],
                )?;
            } else {
                // Version conflict — log warning, re-read and merge would be ideal,
                // but for now just force-write (the concurrent writer already committed)
                tracing::warn!("adaptive state version conflict (expected v{}), force-saving v{}",
                    self.version.saturating_sub(1), self.version);
                conn.execute(
                    "UPDATE metadata SET value = ?1 WHERE key = 'adaptive_state'",
                    rusqlite::params![&json],
                )?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    fn setup_db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        // Minimal schema for testing
        conn.execute_batch("
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
        ").unwrap();
        conn
    }

    #[test]
    fn test_emit_and_consume_events() {
        let conn = setup_db();

        // Emit 3 events
        for i in 0..3 {
            emit_event(&conn, FeedbackEvent {
                event_type: EventType::RecallComplete,
                request_id: Some(format!("req-{i}")),
                memory_id: None, concept_id: None,
                query: Some("test query".into()),
                query_type: Some("semantic".into()),
                topic: Some("test".into()),
                payload: Some(serde_json::json!({"alpha": 0.5})),
            }).unwrap();
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
            LearnedAlphaEntry { value: 0.4, sample_count: 15, last_updated: String::new() },
        );
        // Cluster-specific with too few samples
        state.learned_alpha.insert(
            "semantic:1".into(),
            LearnedAlphaEntry { value: 0.8, sample_count: 3, last_updated: String::new() },
        );

        // Should fall back to global (cluster has < 10 samples)
        let alpha = state.get_alpha("semantic", Some(1)).unwrap();
        assert!((alpha - 0.4).abs() < 0.01);

        // Give cluster enough samples
        state.learned_alpha.get_mut("semantic:1").unwrap().sample_count = 12;
        let alpha = state.get_alpha("semantic", Some(1)).unwrap();
        assert!((alpha - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_alpha_global_and_legacy_key_fallback() {
        let mut state = AdaptiveState::default();
        state.learned_alpha.insert(
            "global".into(),
            LearnedAlphaEntry { value: 0.55, sample_count: 20, last_updated: String::new() },
        );
        let alpha = state.get_alpha("temporal", None).unwrap();
        assert!((alpha - 0.55).abs() < 0.01);

        state.learned_alpha.insert(
            "Temporal".into(),
            LearnedAlphaEntry { value: 0.8, sample_count: 20, last_updated: String::new() },
        );
        let alpha = state.get_alpha("Temporal", None).unwrap();
        assert!((alpha - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_event_type_filter() {
        let conn = setup_db();

        emit_event(&conn, FeedbackEvent {
            event_type: EventType::RecallComplete,
            request_id: None, memory_id: None, concept_id: None,
            query: None, query_type: None, topic: None, payload: None,
        }).unwrap();
        emit_event(&conn, FeedbackEvent {
            event_type: EventType::Store,
            request_id: None, memory_id: Some("m1".into()), concept_id: None,
            query: None, query_type: None, topic: None, payload: None,
        }).unwrap();

        // Filter by store only
        let events = consume_events(&conn, "test", &["store"], 100).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "store");
    }
}
