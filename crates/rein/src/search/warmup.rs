//! Background embedding cache warmup.
//! Pre-computes embeddings for all memories that don't have cached vectors,
//! eliminating the 255ms Google API delay during recall.

use crate::config::ReinConfig;
use crate::embed::{create_embedder, prepend_metadata, EmbedCache};
use crate::store::SqliteStore;
use crate::types::Embedder as _;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TantivyRebuildOutcome {
    SkippedInMemory,
    Rebuilt { indexed: usize, errors: usize },
    AlreadyRunning { lock_path: PathBuf },
    Failed { reason: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TantivyRebuildState {
    Idle,
    Running,
    StaleMarker,
}

/// Warm up the embedding cache by pre-computing embeddings for uncached memories.
/// Returns (cached_count, error_count).
pub async fn warmup(store: &SqliteStore, config: &ReinConfig) -> (usize, usize) {
    // Always rebuild side indexes from existing data, even if all embeddings are cached
    populate_tantivy(store);
    populate_hnsw(store, config);

    let embedder = match create_embedder(config) {
        Some(e) => e,
        None => {
            tracing::info!("no embedding provider configured, skipping warmup");
            return (0, 0);
        }
    };

    let model = config.embedding_model();

    // Get all memory IDs and their content
    let memories = match store.get_all_for_warmup() {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!("failed to list memories for warmup: {e}");
            return (0, 0);
        }
    };

    let total = memories.len();
    if total == 0 {
        return (0, 0);
    }

    // Filter out already-cached ones
    let uncached: Vec<(String, String, String, String, String)> = memories
        .into_iter()
        .filter(|(_, topic, summary, content, _)| {
            let text = prepend_metadata(topic, summary, content);
            EmbedCache::get(store.conn(), &text, &model)
                .ok()
                .flatten()
                .is_none()
        })
        .collect();

    if uncached.is_empty() {
        tracing::info!("warmup: all {total} memories already cached");
        return (0, 0);
    }

    tracing::info!("warmup: {}/{total} memories need embedding", uncached.len());

    let mut cached = 0usize;
    let mut errors = 0usize;

    // Process in batches of 100
    for chunk in uncached.chunks(100) {
        let texts: Vec<String> = chunk
            .iter()
            .map(|(_, topic, summary, content, _)| prepend_metadata(topic, summary, content))
            .collect();
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

        match embedder.embed_batch(&text_refs).await {
            Ok(embeddings) => {
                for (i, emb) in embeddings.iter().enumerate() {
                    if i < texts.len() {
                        if EmbedCache::put(store.conn(), &texts[i], &model, emb).is_ok() {
                            cached += 1;
                        } else {
                            errors += 1;
                        }
                    }
                }
            }
            Err(e) => {
                tracing::warn!("warmup batch failed: {e}");
                errors += chunk.len();
            }
        }
    }

    tracing::info!("warmup complete: {cached} cached, {errors} errors");

    // Rebuild side indexes again to include newly-cached embeddings
    populate_hnsw(store, config);

    (cached, errors)
}

/// Populate (or rebuild) the HNSW index from all cached embeddings in SQLite.
/// Clears the existing index first to remove stale entries.
/// Returns `true` if the index is now in a clean, usable state (success or intentionally empty).
/// Returns `false` if the rebuild was skipped or failed (caller should restore the dirty marker).
pub fn populate_hnsw(store: &SqliteStore, config: &ReinConfig) -> bool {
    let db_path = store.db_path();
    if db_path.to_str() == Some(":memory:") {
        return true; // in-memory test databases need no index
    }
    let hnsw_path = db_path.with_extension("");
    let lock_path = hnsw_path.with_extension("usearch.lock");
    let dims = config.embedding.dimensions;
    let model = config.embedding_model();

    let lock_file = match std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(&lock_path)
    {
        Ok(file) => file,
        Err(e) => {
            tracing::warn!("hnsw: failed to open rebuild lock: {e}");
            return false;
        }
    };
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        let rc = unsafe { libc::flock(lock_file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if rc != 0 {
            tracing::debug!(
                "hnsw: rebuild lock held by another process, skipping: {}",
                std::io::Error::last_os_error()
            );
            return false;
        }
    }

    // Clear stale index before rebuilding
    let _ = std::fs::remove_file(hnsw_path.with_extension("usearch"));
    let _ = std::fs::remove_file(hnsw_path.with_extension("usearch.meta"));

    let mut index = match crate::store::hnsw::HnswIndex::open(&hnsw_path, dims) {
        Ok(idx) => idx,
        Err(e) => {
            tracing::warn!("hnsw: failed to open index: {e}");
            #[cfg(unix)]
            {
                use std::os::unix::io::AsRawFd;
                let _ = unsafe { libc::flock(lock_file.as_raw_fd(), libc::LOCK_UN) };
            }
            return false;
        }
    };

    // Get all memories and their cached embeddings
    let memories = match store.get_all_for_warmup() {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!("hnsw: failed to list memories: {e}");
            #[cfg(unix)]
            {
                use std::os::unix::io::AsRawFd;
                let _ = unsafe { libc::flock(lock_file.as_raw_fd(), libc::LOCK_UN) };
            }
            return false;
        }
    };

    let mut inserted = 0usize;
    for (id, topic, summary, content, _keywords) in &memories {
        let text = prepend_metadata(topic, summary, content);
        if let Ok(Some(emb)) = EmbedCache::get(store.conn(), &text, &model) {
            if emb.len() == dims && index.insert(id, &emb).is_ok() {
                inserted += 1;
            }
        }
    }

    let mut rebuild_ok = false;
    if inserted > 0 {
        match index.save() {
            Ok(()) => {
                tracing::info!("hnsw: indexed {inserted} vectors");
                rebuild_ok = true;
            }
            Err(e) => tracing::warn!("hnsw: failed to save index: {e}"),
        }
    } else if memories.is_empty() {
        rebuild_ok = true; // no memories at all, empty index is intentionally correct
    } else {
        tracing::debug!(
            "hnsw: {} memories but 0 cached embeddings, keeping dirty marker",
            memories.len()
        );
    }
    // Clear the legacy `.dirty` marker on success (no-op when called from async path
    // since `.dirty` was already renamed to `.rebuilding` before this function was called).
    if rebuild_ok {
        let _ = std::fs::remove_file(crate::store::hnsw::HnswIndex::dirty_marker_path(&hnsw_path));
    }

    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        let _ = unsafe { libc::flock(lock_file.as_raw_fd(), libc::LOCK_UN) };
    }
    rebuild_ok
}

/// Populate the Tantivy FTS index from all memories in SQLite.
/// Clears the existing index first to remove stale entries.
/// Uses a file lock to prevent concurrent rebuilds across processes.
pub fn populate_tantivy(store: &SqliteStore) {
    let _ = try_populate_tantivy(store);
}

/// Populate the Tantivy FTS index and report whether this process owned the rebuild.
pub fn try_populate_tantivy(store: &SqliteStore) -> TantivyRebuildOutcome {
    let db_path = store.db_path();
    if db_path.to_str() == Some(":memory:") {
        return TantivyRebuildOutcome::SkippedInMemory;
    }
    let tantivy_path = db_path.with_extension("tantivy");
    let lock_path = tantivy_rebuild_lock_path(db_path);
    let rebuilding_path = tantivy_rebuilding_path(db_path);

    // Acquire exclusive file lock — skip if another process is rebuilding.
    let lock_file = match std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(&lock_path)
    {
        Ok(f) => f,
        Err(e) => {
            return TantivyRebuildOutcome::Failed {
                reason: format!("failed to open rebuild lock {}: {e}", lock_path.display()),
            }
        }
    };
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        let fd = lock_file.as_raw_fd();
        let ret = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
        if ret != 0 {
            tracing::debug!("tantivy: another process is rebuilding, skipping");
            return TantivyRebuildOutcome::AlreadyRunning { lock_path };
        }
    }

    if let Err(e) = std::fs::write(&rebuilding_path, b"rebuilding") {
        unlock_tantivy_rebuild_lock(&lock_file);
        return TantivyRebuildOutcome::Failed {
            reason: format!(
                "failed to write rebuild marker {}: {e}",
                rebuilding_path.display()
            ),
        };
    }

    // Clear stale index before rebuilding
    if let Err(e) = std::fs::remove_dir_all(&tantivy_path) {
        if e.kind() != std::io::ErrorKind::NotFound {
            mark_tantivy_dirty(db_path);
            let _ = std::fs::remove_file(&rebuilding_path);
            unlock_tantivy_rebuild_lock(&lock_file);
            return TantivyRebuildOutcome::Failed {
                reason: format!(
                    "failed to clear stale index {}: {e}",
                    tantivy_path.display()
                ),
            };
        }
    }

    let tantivy = match crate::store::tantivy_fts::TantivyFts::open(&tantivy_path) {
        Ok(t) => t,
        Err(e) => {
            tracing::warn!("tantivy: failed to open index: {e}");
            mark_tantivy_dirty(db_path);
            let _ = std::fs::remove_file(&rebuilding_path);
            unlock_tantivy_rebuild_lock(&lock_file);
            return TantivyRebuildOutcome::Failed {
                reason: format!("failed to open index {}: {e}", tantivy_path.display()),
            };
        }
    };

    let memories = match store.get_all_for_warmup() {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!("tantivy: failed to list memories: {e}");
            mark_tantivy_dirty(db_path);
            let _ = std::fs::remove_file(&rebuilding_path);
            unlock_tantivy_rebuild_lock(&lock_file);
            return TantivyRebuildOutcome::Failed {
                reason: format!("failed to list memories: {e}"),
            };
        }
    };

    let mut indexed = 0usize;
    let mut errors = 0usize;
    for (id, topic, summary, content, keywords) in &memories {
        if tantivy
            .insert_strict(id, topic, summary, content, keywords)
            .is_ok()
        {
            indexed += 1;
        } else {
            errors += 1;
        }
    }

    if indexed > 0 {
        tracing::info!("tantivy: indexed {indexed} documents ({errors} errors)");
    }

    finish_tantivy_rebuild_markers(db_path, indexed, errors, memories.is_empty());

    // Lock released when lock_file is dropped.
    let _ = std::fs::remove_file(&rebuilding_path);
    unlock_tantivy_rebuild_lock(&lock_file);
    drop(lock_file);
    TantivyRebuildOutcome::Rebuilt { indexed, errors }
}

pub fn tantivy_dirty_path(db_path: &Path) -> PathBuf {
    db_path.with_extension("tantivy").join(".dirty")
}

pub fn tantivy_rebuild_lock_path(db_path: &Path) -> PathBuf {
    db_path.with_extension("tantivy.rebuild.lock")
}

pub fn tantivy_rebuilding_path(db_path: &Path) -> PathBuf {
    db_path.with_extension("tantivy.rebuilding")
}

pub fn tantivy_rebuild_state(db_path: &Path) -> TantivyRebuildState {
    let marker_exists = tantivy_rebuilding_path(db_path).exists();
    let lock_path = tantivy_rebuild_lock_path(db_path);
    if lock_path.exists() && tantivy_rebuild_lock_is_held(&lock_path) {
        TantivyRebuildState::Running
    } else if marker_exists {
        TantivyRebuildState::StaleMarker
    } else {
        TantivyRebuildState::Idle
    }
}

fn mark_tantivy_dirty(db_path: &Path) {
    let dirty_path = tantivy_dirty_path(db_path);
    if let Some(parent) = dirty_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(dirty_path, b"dirty");
}

fn finish_tantivy_rebuild_markers(
    db_path: &Path,
    indexed: usize,
    errors: usize,
    memories_empty: bool,
) {
    // Only clear dirty marker if rebuild succeeded with actual data, or there
    // are truly no memories. Any partial error must keep the marker so a later
    // repair can pick up missing documents.
    if errors == 0 && (indexed > 0 || memories_empty) {
        let _ = std::fs::remove_file(tantivy_dirty_path(db_path));
    } else {
        mark_tantivy_dirty(db_path);
        if indexed == 0 && !memories_empty {
            tracing::debug!("tantivy: non-empty store but 0 indexed, keeping dirty marker");
        }
    }
}

fn tantivy_rebuild_lock_is_held(lock_path: &Path) -> bool {
    let file = match std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(lock_path)
    {
        Ok(file) => file,
        Err(_) => return false,
    };

    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        let ret = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if ret != 0 {
            return true;
        }
        let _ = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_UN) };
    }

    false
}

fn unlock_tantivy_rebuild_lock(lock_file: &std::fs::File) {
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        let _ = unsafe { libc::flock(lock_file.as_raw_fd(), libc::LOCK_UN) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::SqliteStore;

    fn test_store() -> (tempfile::TempDir, SqliteStore) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("memories.db");
        let store = SqliteStore::new(&db_path, "text-embedding-3-small", 3072).unwrap();
        (dir, store)
    }

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

    #[test]
    #[cfg(unix)]
    fn try_populate_tantivy_reports_already_running_when_rebuild_lock_held() {
        let (_dir, store) = test_store();
        let dirty = tantivy_dirty_path(store.db_path());
        std::fs::create_dir_all(dirty.parent().unwrap()).unwrap();
        std::fs::write(&dirty, b"dirty").unwrap();
        let lock_path = tantivy_rebuild_lock_path(store.db_path());
        let _lock = hold_file_lock(&lock_path);

        let outcome = try_populate_tantivy(&store);

        assert_eq!(
            outcome,
            TantivyRebuildOutcome::AlreadyRunning {
                lock_path: lock_path.clone()
            }
        );
        assert!(
            dirty.exists(),
            "dirty marker must remain for the active owner"
        );
    }

    #[test]
    fn try_populate_tantivy_clears_dirty_and_rebuilding_marker_on_success() {
        let (_dir, store) = test_store();
        let dirty = tantivy_dirty_path(store.db_path());
        std::fs::create_dir_all(dirty.parent().unwrap()).unwrap();
        std::fs::write(&dirty, b"dirty").unwrap();
        let rebuilding = tantivy_rebuilding_path(store.db_path());
        std::fs::write(&rebuilding, b"rebuilding").unwrap();

        let outcome = try_populate_tantivy(&store);

        assert_eq!(
            outcome,
            TantivyRebuildOutcome::Rebuilt {
                indexed: 0,
                errors: 0
            }
        );
        assert!(!dirty.exists(), "clean rebuild should clear dirty marker");
        assert!(
            !rebuilding.exists(),
            "successful rebuild should clear external rebuilding marker"
        );
    }

    #[test]
    fn finish_tantivy_rebuild_marks_dirty_after_partial_errors() {
        let (_dir, store) = test_store();
        let dirty = tantivy_dirty_path(store.db_path());
        std::fs::create_dir_all(dirty.parent().unwrap()).unwrap();
        assert!(!dirty.exists());

        finish_tantivy_rebuild_markers(store.db_path(), 1, 1, false);

        assert!(
            dirty.exists(),
            "partial rebuild errors must keep Tantivy dirty for repair"
        );
    }
}
