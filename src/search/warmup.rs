//! Background embedding cache warmup.
//! Pre-computes embeddings for all memories that don't have cached vectors,
//! eliminating the 255ms Google API delay during recall.

use crate::config::ReinConfig;
use crate::embed::{create_embedder, prepend_metadata, EmbedCache};
use crate::store::SqliteStore;
use crate::types::Embedder as _;
use std::path::{Path, PathBuf};

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
    let db_path = store.db_path();
    if db_path.to_str() == Some(":memory:") {
        return;
    }
    let tantivy_path = db_path.with_extension("tantivy");
    let lock_path = db_path.with_extension("tantivy.rebuild.lock");

    // Acquire exclusive file lock — skip if another process is rebuilding.
    let lock_file = match std::fs::File::create(&lock_path) {
        Ok(f) => f,
        Err(_) => return,
    };
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        let fd = lock_file.as_raw_fd();
        let ret = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
        if ret != 0 {
            tracing::debug!("tantivy: another process is rebuilding, skipping");
            return;
        }
    }

    // Clear stale index before rebuilding
    let _ = std::fs::remove_dir_all(&tantivy_path);

    let tantivy = match crate::store::tantivy_fts::TantivyFts::open(&tantivy_path) {
        Ok(t) => t,
        Err(e) => {
            tracing::warn!("tantivy: failed to open index: {e}");
            return;
        }
    };

    let memories = match store.get_all_for_warmup() {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!("tantivy: failed to list memories: {e}");
            return;
        }
    };

    let mut indexed = 0usize;
    let mut errors = 0usize;
    for (id, topic, summary, content, keywords) in &memories {
        if tantivy
            .insert(id, topic, summary, content, keywords)
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

    // Only clear dirty marker if rebuild succeeded with actual data, or there are truly no memories
    if errors == 0 && (indexed > 0 || memories.is_empty()) {
        let _ = std::fs::remove_file(tantivy_dirty_path(db_path));
    } else if indexed == 0 && !memories.is_empty() {
        tracing::debug!(
            "tantivy: {} memories but 0 indexed, keeping dirty marker",
            memories.len()
        );
    }

    // Lock released when lock_file is dropped.
    drop(lock_file);
    let _ = std::fs::remove_file(&lock_path);
}

pub fn tantivy_dirty_path(db_path: &Path) -> PathBuf {
    db_path.with_extension("tantivy").join(".dirty")
}
