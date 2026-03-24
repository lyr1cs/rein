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

        let index = Index::new(&opts)
            .map_err(|e| ReinError::Config(format!("hnsw init: {e}")))?;

        let index_path = path.with_extension("usearch");
        let meta_path = path.with_extension("usearch.meta");

        let (id_to_key, key_to_id, next_key) = if index_path.exists() {
            index
                .load(index_path.to_str().unwrap_or(""))
                .map_err(|e| ReinError::Config(format!("hnsw load: {e}")))?;
            // Load metadata
            if meta_path.exists() {
                let meta = std::fs::read_to_string(&meta_path)
                    .map_err(|e| ReinError::Config(format!("hnsw meta read: {e}")))?;
                deserialize_meta(&meta)
            } else {
                (HashMap::new(), HashMap::new(), 0)
            }
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

    /// Insert a vector for the given memory ID.
    pub fn insert(&mut self, id: &str, vector: &[f32]) -> ReinResult<()> {
        let key = self.next_key;
        self.next_key += 1;

        // Remove old entry if exists
        if let Some(&old_key) = self.id_to_key.get(id) {
            let _ = self.index.remove(old_key);
            self.key_to_id.remove(&old_key);
        }

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
