//! System health snapshot — cheap polling signals for `/api/health`.
//!
//! Exposes side-index state (HNSW, Tantivy), async queue depth, and the
//! pending-grayzone backlog so external watchers can detect lag without
//! running the full `rein doctor` pipeline.
//!
//! Design note: all probes here must be cheap enough to poll every few
//! seconds. We only stat files and run bounded `SELECT COUNT(*)` queries;
//! we never open the HNSW or Tantivy index (that is `doctor`'s job).

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::config::ReinConfig;
use crate::extract::hooks::queue::{collect_queue_diagnostics, QueueGroupDiagnostics};
use crate::search::warmup;
use crate::store::hnsw::HnswIndex;
use crate::store::SqliteStore;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IndexStatus {
    /// `.dirty` marker present — the next rebuild will pick up missed writes.
    pub dirty: bool,
    /// `.rebuilding` marker present — a warmup/rebuild is in progress.
    #[serde(default)]
    pub rebuilding: bool,
    /// A rebuild marker exists without a live owner lock.
    #[serde(default)]
    pub stale_rebuild_marker: bool,
    /// Index file (HNSW) or directory (Tantivy) exists on disk.
    pub index_exists: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueueSummary {
    pub pending: usize,
    pub inflight: usize,
    pub dead_letters: usize,
}

impl From<&crate::extract::hooks::queue::QueueDiagnostics> for QueueSummary {
    fn from(d: &crate::extract::hooks::queue::QueueDiagnostics) -> Self {
        Self {
            pending: d.pending,
            inflight: d.inflight,
            dead_letters: d.dead_letters,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueuesSnapshot {
    pub memory: QueueSummary,
    pub cleanup: QueueSummary,
    pub dedup: QueueSummary,
    pub merge_refinement: QueueSummary,
}

impl From<&QueueGroupDiagnostics> for QueuesSnapshot {
    fn from(g: &QueueGroupDiagnostics) -> Self {
        Self {
            memory: (&g.memory).into(),
            cleanup: (&g.cleanup).into(),
            dedup: (&g.dedup).into(),
            merge_refinement: (&g.merge_refinement).into(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IndexesSnapshot {
    pub hnsw: IndexStatus,
    pub tantivy: IndexStatus,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GrayzoneSnapshot {
    pub pending: usize,
}

/// Top-level derived liveness: `ok` is false when any operator-actionable
/// issue is detected. `issues` is a short human-readable list — callers can
/// surface these directly in a status line.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SystemStatus {
    pub ok: bool,
    pub issues: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SystemHealthSnapshot {
    pub indexes: IndexesSnapshot,
    pub queues: QueuesSnapshot,
    pub grayzone: GrayzoneSnapshot,
    pub status: SystemStatus,
}

/// Collect all cheap health signals. Safe to call on every poll.
pub fn collect(store: &SqliteStore, config: &ReinConfig) -> SystemHealthSnapshot {
    let db_path = store.db_path();

    let indexes = IndexesSnapshot {
        hnsw: probe_hnsw(db_path),
        tantivy: probe_tantivy(db_path),
    };

    let queues_raw = collect_queue_diagnostics(config);
    let queues = QueuesSnapshot::from(&queues_raw);

    let grayzone = GrayzoneSnapshot {
        pending: count_pending_grayzone(store),
    };

    let status = derive_status(&indexes, &queues, &grayzone);

    SystemHealthSnapshot {
        indexes,
        queues,
        grayzone,
        status,
    }
}

fn probe_hnsw(db_path: &Path) -> IndexStatus {
    // HNSW paths are derived from db_path with no extension (e.g.
    // `~/.rein/memories.db` → `~/.rein/memories.usearch`).
    let base = db_path.with_extension("");
    let index_file = base.with_extension("usearch");
    let dirty = HnswIndex::dirty_marker_path(&base).exists();
    let rebuilding = HnswIndex::rebuilding_marker_path(&base).exists();
    IndexStatus {
        dirty,
        rebuilding,
        stale_rebuild_marker: false,
        index_exists: index_file.exists(),
    }
}

fn probe_tantivy(db_path: &Path) -> IndexStatus {
    let index_dir = db_path.with_extension("tantivy");
    let dirty = warmup::tantivy_dirty_path(db_path).exists();
    let rebuild_state = warmup::tantivy_rebuild_state(db_path);
    IndexStatus {
        dirty,
        rebuilding: matches!(rebuild_state, warmup::TantivyRebuildState::Running),
        stale_rebuild_marker: matches!(rebuild_state, warmup::TantivyRebuildState::StaleMarker),
        index_exists: index_dir.exists(),
    }
}

fn count_pending_grayzone(store: &SqliteStore) -> usize {
    store
        .conn()
        .query_row("SELECT COUNT(*) FROM pending_grayzone_jobs", [], |row| {
            row.get::<_, i64>(0)
        })
        .map(|n| n.max(0) as usize)
        .unwrap_or(0)
}

fn derive_status(
    indexes: &IndexesSnapshot,
    queues: &QueuesSnapshot,
    grayzone: &GrayzoneSnapshot,
) -> SystemStatus {
    let mut issues = Vec::new();

    if indexes.hnsw.dirty {
        issues.push("hnsw index dirty".to_string());
    }
    if indexes.hnsw.rebuilding {
        issues.push("hnsw index rebuilding".to_string());
    }
    if indexes.tantivy.dirty {
        issues.push("tantivy index dirty".to_string());
    }
    if indexes.tantivy.rebuilding {
        issues.push("tantivy index rebuilding".to_string());
    }
    if indexes.tantivy.stale_rebuild_marker {
        issues.push("tantivy rebuild marker stale".to_string());
    }

    let dead_letters = queues.memory.dead_letters
        + queues.cleanup.dead_letters
        + queues.dedup.dead_letters
        + queues.merge_refinement.dead_letters;
    if dead_letters > 0 {
        issues.push(format!("{dead_letters} queue dead-letter(s)"));
    }

    // Pending grayzone is expected to be small; flag only when it grows.
    if grayzone.pending > 50 {
        issues.push(format!("{} grayzone jobs pending", grayzone.pending));
    }

    SystemStatus {
        ok: issues.is_empty(),
        issues,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ReinConfig;
    use crate::store::SqliteStore;
    use tempfile::tempdir;

    #[cfg(unix)]
    fn hold_file_lock(path: &Path) -> std::fs::File {
        use std::os::unix::io::AsRawFd;

        let file = std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(path)
            .unwrap();
        let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        assert_eq!(rc, 0, "test failed to acquire advisory lock");
        file
    }

    fn test_store() -> (tempfile::TempDir, SqliteStore, ReinConfig) {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("memories.db");
        let mut config = ReinConfig::default();
        config.database.path = db_path.to_string_lossy().into_owned();
        let store = SqliteStore::new(&db_path, "text-embedding-3-small", 3072).unwrap();
        (dir, store, config)
    }

    #[test]
    fn empty_store_reports_clean_status() {
        let (_dir, store, config) = test_store();
        let snap = collect(&store, &config);

        assert!(!snap.indexes.hnsw.dirty);
        assert!(!snap.indexes.hnsw.index_exists);
        assert!(!snap.indexes.tantivy.dirty);
        assert_eq!(snap.grayzone.pending, 0);
        assert!(
            snap.status.ok,
            "expected ok=true, issues={:?}",
            snap.status.issues
        );
        assert!(snap.status.issues.is_empty());
    }

    #[test]
    fn hnsw_dirty_marker_surfaces_issue() {
        let (_dir, store, config) = test_store();
        let base = store.db_path().with_extension("");
        std::fs::write(HnswIndex::dirty_marker_path(&base), b"dirty").unwrap();

        let snap = collect(&store, &config);
        assert!(snap.indexes.hnsw.dirty);
        assert!(!snap.status.ok);
        assert!(snap
            .status
            .issues
            .iter()
            .any(|i| i.contains("hnsw index dirty")));
    }

    #[test]
    fn tantivy_dirty_marker_surfaces_issue() {
        let (_dir, store, config) = test_store();
        // Ensure the tantivy dir exists before writing the marker inside it.
        let dir_path = store.db_path().with_extension("tantivy");
        std::fs::create_dir_all(&dir_path).unwrap();
        std::fs::write(warmup::tantivy_dirty_path(store.db_path()), b"dirty").unwrap();

        let snap = collect(&store, &config);
        assert!(snap.indexes.tantivy.dirty);
        assert!(!snap.status.ok);
        assert!(snap
            .status
            .issues
            .iter()
            .any(|i| i.contains("tantivy index dirty")));
    }

    #[test]
    #[cfg(unix)]
    fn tantivy_running_rebuild_surfaces_issue() {
        let (_dir, store, config) = test_store();
        let lock_path = warmup::tantivy_rebuild_lock_path(store.db_path());
        let _lock = hold_file_lock(&lock_path);
        std::fs::write(
            warmup::tantivy_rebuilding_path(store.db_path()),
            b"rebuilding",
        )
        .unwrap();

        let snap = collect(&store, &config);

        assert!(snap.indexes.tantivy.rebuilding);
        assert!(!snap.status.ok);
        assert!(snap
            .status
            .issues
            .iter()
            .any(|i| i.contains("tantivy index rebuilding")));
    }

    #[test]
    fn tantivy_stale_rebuild_marker_surfaces_distinct_issue() {
        let (_dir, store, config) = test_store();
        std::fs::write(
            warmup::tantivy_rebuilding_path(store.db_path()),
            b"rebuilding",
        )
        .unwrap();

        let snap = collect(&store, &config);

        assert!(!snap.indexes.tantivy.rebuilding);
        assert!(snap.indexes.tantivy.stale_rebuild_marker);
        assert!(!snap.status.ok);
        assert!(snap
            .status
            .issues
            .iter()
            .any(|i| i.contains("tantivy rebuild marker stale")));
        assert!(!snap
            .status
            .issues
            .iter()
            .any(|i| i.contains("tantivy index rebuilding")));
    }
}
