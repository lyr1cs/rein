use std::collections::HashMap;
use std::path::{Path, PathBuf};

use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

use crate::types::error::ReinError;
use crate::types::ReinResult;

/// HNSW-based vector index using usearch.
/// Provides O(log n) approximate nearest neighbor search.
pub struct HnswIndex {
    index: Index,
    path: PathBuf,
    id_to_key: HashMap<String, u64>,
    key_to_id: HashMap<u64, String>,
    next_key: u64,
    dims: usize,
}

impl HnswIndex {
    /// Open or create an HNSW index at the given path.
    pub fn open(path: &Path, dims: usize) -> ReinResult<Self> {
        let opts = IndexOptions {
            dimensions: dims,
            metric: MetricKind::Cos,
            quantization: ScalarKind::F32,
            ..Default::default()
        };

        let index = Index::new(&opts).map_err(|e| ReinError::Config(format!("hnsw init: {e}")))?;

        let index_path = path.with_extension("usearch");
        let meta_path = path.with_extension("usearch.meta");

        let (id_to_key, key_to_id, next_key) = if index_path.exists() {
            index
                .load(index_path.to_str().unwrap_or(""))
                .map_err(|e| ReinError::Config(format!("hnsw load: {e}")))?;
            if !meta_path.exists() {
                Self::mark_dirty(path);
                return Err(ReinError::Config(
                    "hnsw metadata missing; rebuild required".to_string(),
                ));
            }
            let meta = std::fs::read_to_string(&meta_path)
                .map_err(|e| ReinError::Config(format!("hnsw meta read: {e}")))?;
            let (id_to_key, key_to_id, next_key) = deserialize_meta(&meta);
            let index_size = index.size();
            if (index_size > 0 && key_to_id.is_empty())
                || key_to_id.len() != id_to_key.len()
                || key_to_id.len() != index_size
            {
                Self::mark_dirty(path);
                return Err(ReinError::Config(
                    "hnsw metadata corrupt or out of sync; rebuild required".to_string(),
                ));
            }
            (id_to_key, key_to_id, next_key)
        } else {
            // Reserve capacity for initial use
            index
                .reserve(1000)
                .map_err(|e| ReinError::Config(format!("hnsw reserve: {e}")))?;
            (HashMap::new(), HashMap::new(), 0)
        };

        Ok(Self {
            index,
            path: path.to_path_buf(),
            id_to_key,
            key_to_id,
            next_key,
            dims,
        })
    }

    /// Number of vectors in the index.
    pub fn len(&self) -> usize {
        self.index.size()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.index.size() == 0
    }

    /// Whether `id` currently has an entry in the index.
    pub fn contains(&self, id: &str) -> bool {
        self.id_to_key.contains_key(id)
    }

    /// Insert a vector for the given memory ID.
    pub fn insert(&mut self, id: &str, vector: &[f32]) -> ReinResult<()> {
        let key = self.next_key;
        let old_key = self.id_to_key.get(id).copied();

        // Ensure capacity
        if self.index.size() >= self.index.capacity() {
            let new_cap = self.index.capacity() * 2 + 100;
            self.index
                .reserve(new_cap)
                .map_err(|e| ReinError::Config(format!("hnsw reserve: {e}")))?;
        }

        self.index
            .add(key, vector)
            .map_err(|e| ReinError::Config(format!("hnsw add: {e}")))?;
        self.next_key += 1;
        if let Some(old_key) = old_key {
            let _ = self.index.remove(old_key);
            self.key_to_id.remove(&old_key);
        }
        self.id_to_key.insert(id.to_string(), key);
        self.key_to_id.insert(key, id.to_string());

        Ok(())
    }

    /// Search for the top-k nearest neighbors.
    /// Returns pairs of (memory_id, distance).
    pub fn search(&self, query: &[f32], limit: usize) -> ReinResult<Vec<(String, f32)>> {
        if self.index.size() == 0 {
            return Ok(vec![]);
        }

        let results = self
            .index
            .search(query, limit)
            .map_err(|e| ReinError::Config(format!("hnsw search: {e}")))?;

        let mut out = Vec::new();
        for i in 0..results.keys.len() {
            if let Some(id) = self.key_to_id.get(&results.keys[i]) {
                out.push((id.clone(), results.distances[i]));
            }
        }

        Ok(out)
    }

    /// Delete a vector by memory ID.
    pub fn delete(&mut self, id: &str) -> ReinResult<()> {
        if let Some(&key) = self.id_to_key.get(id) {
            let _ = self.index.remove(key);
            self.id_to_key.remove(id);
            self.key_to_id.remove(&key);
        }
        Ok(())
    }

    /// Save the index and metadata to disk.
    ///
    /// NOTE: save() does NOT clear the `.dirty` marker (Codex round-5 M-2).
    /// An incremental save after a missed write only proves THIS write
    /// landed — earlier writes that set `.dirty` (open failures, lock
    /// contention, crash recovery gaps) may still need a full rebuild
    /// to reconcile. The rebuild path is the only legitimate clearer of
    /// `.dirty`: `take_dirty_for_rebuild` renames `.dirty` → `.rebuilding`
    /// as it starts, and a successful rebuild completes by dropping the
    /// `.rebuilding` marker via `clear_rebuilding`. Incremental writes
    /// that succeed while `.dirty` exists should leave the marker alone
    /// so warmup still triggers the recovery rebuild.
    pub fn save(&self) -> ReinResult<()> {
        let index_path = self.path.with_extension("usearch");
        self.index
            .save(index_path.to_str().unwrap_or(""))
            .map_err(|e| ReinError::Config(format!("hnsw save: {e}")))?;

        // Save metadata
        let meta_path = self.path.with_extension("usearch.meta");
        let meta = serialize_meta(&self.id_to_key, self.next_key);
        std::fs::write(&meta_path, meta)
            .map_err(|e| ReinError::Config(format!("hnsw meta save: {e}")))?;

        Ok(())
    }

    /// Dimensions this index was created with.
    pub fn dims(&self) -> usize {
        self.dims
    }

    pub fn dirty_marker_path(path: &Path) -> PathBuf {
        path.with_extension("usearch.dirty")
    }

    pub fn rebuilding_marker_path(path: &Path) -> PathBuf {
        path.with_extension("usearch.rebuilding")
    }

    pub fn mark_dirty(path: &Path) {
        let _ = std::fs::write(Self::dirty_marker_path(path), b"dirty");
    }

    /// Returns true if either the `.dirty` or `.rebuilding` marker exists.
    pub fn is_dirty(path: &Path) -> bool {
        Self::dirty_marker_path(path).exists() || Self::rebuilding_marker_path(path).exists()
    }

    /// Atomically claim the dirty marker for rebuild: renames `.dirty` → `.rebuilding`.
    /// Returns `true` if this caller won the race and should proceed with the rebuild.
    /// Returns `false` if another thread already claimed it (`.dirty` was gone).
    pub fn take_dirty_for_rebuild(path: &Path) -> bool {
        let dirty = Self::dirty_marker_path(path);
        let rebuilding = Self::rebuilding_marker_path(path);
        if dirty.exists() {
            std::fs::rename(&dirty, &rebuilding).is_ok()
        } else {
            false
        }
    }

    /// Remove the `.rebuilding` marker after a background rebuild completes or fails.
    pub fn clear_rebuilding(path: &Path) {
        let _ = std::fs::remove_file(Self::rebuilding_marker_path(path));
    }
}

fn serialize_meta(i2k: &HashMap<String, u64>, next_key: u64) -> String {
    let mut lines = vec![format!("next_key:{next_key}")];
    for (id, key) in i2k {
        lines.push(format!("{key}:{id}"));
    }
    lines.join("\n")
}

fn deserialize_meta(meta: &str) -> (HashMap<String, u64>, HashMap<u64, String>, u64) {
    let mut i2k = HashMap::new();
    let mut k2i = HashMap::new();
    let mut next_key = 0u64;

    for line in meta.lines() {
        if let Some(nk) = line.strip_prefix("next_key:") {
            next_key = nk.parse().unwrap_or(0);
        } else if let Some((key_str, id)) = line.split_once(':') {
            if let Ok(key) = key_str.parse::<u64>() {
                i2k.insert(id.to_string(), key);
                k2i.insert(key, id.to_string());
            }
        }
    }

    (i2k, k2i, next_key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw_basic_operations() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_index");

        let mut index = HnswIndex::open(&path, 3).unwrap();
        assert!(index.is_empty());

        // Insert vectors
        index.insert("mem1", &[1.0, 0.0, 0.0]).unwrap();
        index.insert("mem2", &[0.0, 1.0, 0.0]).unwrap();
        index.insert("mem3", &[1.0, 0.1, 0.0]).unwrap();
        assert_eq!(index.len(), 3);

        // Search: mem3 should be closest to mem1
        let results = index.search(&[1.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "mem1"); // exact match first

        // Delete
        index.delete("mem1").unwrap();
        assert_eq!(index.len(), 2);

        // Save and reload
        index.save().unwrap();
        let reloaded = HnswIndex::open(&path, 3).unwrap();
        assert_eq!(reloaded.len(), 2);
        let results2 = reloaded.search(&[1.0, 0.0, 0.0], 2).unwrap();
        assert!(!results2.is_empty());
    }

    #[test]
    fn test_hnsw_update_existing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_update");

        let mut index = HnswIndex::open(&path, 3).unwrap();
        index.insert("mem1", &[1.0, 0.0, 0.0]).unwrap();
        index.insert("mem1", &[0.0, 1.0, 0.0]).unwrap(); // replace
        assert_eq!(index.len(), 1);

        let results = index.search(&[0.0, 1.0, 0.0], 1).unwrap();
        assert_eq!(results[0].0, "mem1");
    }

    #[test]
    fn test_hnsw_empty_search() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_empty");

        let index = HnswIndex::open(&path, 3).unwrap();
        let results = index.search(&[1.0, 0.0, 0.0], 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn incremental_save_does_not_clear_pre_existing_dirty_marker() {
        // Codex round-5 M-2: an unrelated write-failure somewhere upstream
        // marks `.dirty`. A later successful incremental save on a
        // DIFFERENT entry must NOT clear the marker — the dirty signal
        // records "something somewhere in this index may be out of sync
        // with the source of truth and only a full rebuild can
        // reconcile." If `save()` cleared `.dirty`, the recovery rebuild
        // would never fire.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_dirty_preservation");

        // Simulate an earlier missed write setting the marker (e.g., lock
        // contention during `update_hnsw`). The index itself is fine.
        let mut index = HnswIndex::open(&path, 3).unwrap();
        index.insert("mem1", &[1.0, 0.0, 0.0]).unwrap();
        HnswIndex::mark_dirty(&path);
        assert!(HnswIndex::is_dirty(&path));

        // A subsequent successful incremental save (maybe from a
        // different op path, e.g. resummerize's delete) proceeds
        // normally but must NOT clear the dirty marker.
        index.save().unwrap();
        assert!(
            HnswIndex::is_dirty(&path),
            "incremental save must preserve pre-existing dirty marker; \
             only take_dirty_for_rebuild / clear_rebuilding may clear it"
        );

        // The legitimate clearer of `.dirty` is the rebuild claim.
        assert!(HnswIndex::take_dirty_for_rebuild(&path));
        assert!(
            !HnswIndex::dirty_marker_path(&path).exists(),
            "take_dirty_for_rebuild must rename .dirty → .rebuilding"
        );
        HnswIndex::clear_rebuilding(&path);
        assert!(!HnswIndex::is_dirty(&path));
    }

    #[test]
    fn test_hnsw_open_requires_metadata_for_populated_index() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_meta_missing");

        let mut index = HnswIndex::open(&path, 3).unwrap();
        index.insert("mem1", &[1.0, 0.0, 0.0]).unwrap();
        index.save().unwrap();

        std::fs::remove_file(path.with_extension("usearch.meta")).unwrap();

        let err = match HnswIndex::open(&path, 3) {
            Ok(_) => panic!("expected metadata-missing open to fail"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("metadata"));
        assert!(HnswIndex::is_dirty(&path));
    }

    #[test]
    fn test_meta_serialization() {
        let mut map = HashMap::new();
        map.insert("id1".to_string(), 0u64);
        map.insert("id2".to_string(), 1u64);

        let serialized = serialize_meta(&map, 2);
        let (i2k, k2i, next_key) = deserialize_meta(&serialized);

        assert_eq!(next_key, 2);
        assert_eq!(i2k.len(), 2);
        assert_eq!(k2i.len(), 2);
        assert_eq!(*i2k.get("id1").unwrap(), 0);
        assert_eq!(*i2k.get("id2").unwrap(), 1);
    }
}
