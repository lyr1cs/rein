use crate::types::error::ReinResult;
use rusqlite::Connection;
use sha2::{Digest, Sha256};

pub struct EmbedCache;

impl EmbedCache {
    /// Look up a cached embedding by query text.
    /// `model` is included in the hash so different models don't share cache entries.
    pub fn get(conn: &Connection, query: &str, model: &str) -> ReinResult<Option<Vec<f32>>> {
        let hash = Self::hash_query(model, query);
        let mut stmt = conn.prepare("SELECT embedding FROM embed_cache WHERE query_hash = ?1")?;
        let result = stmt.query_row([&hash], |row| {
            let blob: Vec<u8> = row.get(0)?;
            Ok(blob)
        });

        match result {
            Ok(blob) => Ok(Some(Self::bytes_to_f32(&blob))),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Store an embedding in the cache, keyed by model + query text.
    /// Evicts oldest entries when cache exceeds 10,000 entries.
    /// When count exceeds 5,000, also runs TTL cleanup (entries older than 30 days).
    pub fn put(conn: &Connection, query: &str, model: &str, embedding: &[f32]) -> ReinResult<()> {
        let hash = Self::hash_query(model, query);
        let blob = Self::f32_to_bytes(embedding);
        let now = chrono::Utc::now().to_rfc3339();
        conn.execute(
            "INSERT OR REPLACE INTO embed_cache (query_hash, embedding, created_at) VALUES (?1, ?2, ?3)",
            rusqlite::params![hash, blob, now],
        )?;
        let count: usize = conn
            .query_row("SELECT COUNT(*) FROM embed_cache", [], |row| row.get(0))
            .unwrap_or(0);
        // TTL cleanup: when cache is moderately full, remove stale entries
        if count > 5_000 {
            let _ = Self::cleanup(conn, 30);
        }
        // Evict oldest entries if cache still exceeds hard limit
        const MAX_CACHE_ENTRIES: usize = 10_000;
        if count > MAX_CACHE_ENTRIES {
            conn.execute(
                "DELETE FROM embed_cache WHERE query_hash IN (
                    SELECT query_hash FROM embed_cache ORDER BY created_at ASC LIMIT ?1
                )",
                rusqlite::params![count - MAX_CACHE_ENTRIES],
            )?;
        }
        Ok(())
    }

    /// Delete cache entries older than `max_age_days`.
    /// Returns the number of entries removed.
    pub fn cleanup(conn: &Connection, max_age_days: u64) -> ReinResult<usize> {
        let cutoff = chrono::Utc::now() - chrono::Duration::days(max_age_days as i64);
        let cutoff_str = cutoff.to_rfc3339();
        let deleted = conn.execute(
            "DELETE FROM embed_cache WHERE created_at < ?1",
            rusqlite::params![cutoff_str],
        )?;
        Ok(deleted)
    }

    fn hash_query(model: &str, query: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(model.as_bytes());
        hasher.update(b":");
        hasher.update(query.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    fn f32_to_bytes(embedding: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(embedding.len() * 4);
        for &val in embedding {
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        bytes
    }

    fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::schema::{init_schema, init_sqlite_vec};

    fn setup_db() -> Connection {
        init_sqlite_vec();
        let conn = Connection::open_in_memory().unwrap();
        init_schema(&conn, 3072).unwrap();
        conn
    }

    #[test]
    fn test_cache_roundtrip() {
        let conn = setup_db();
        let mut vec = Vec::with_capacity(3072);
        for i in 0..3072 {
            vec.push(i as f32 * 0.001);
        }

        EmbedCache::put(&conn, "test query", "test-model", &vec).unwrap();
        let result = EmbedCache::get(&conn, "test query", "test-model")
            .unwrap()
            .unwrap();

        assert_eq!(result.len(), 3072);
        for (a, b) in vec.iter().zip(result.iter()) {
            assert!((a - b).abs() < f32::EPSILON, "mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_cache_miss() {
        let conn = setup_db();
        let result = EmbedCache::get(&conn, "nonexistent query", "test-model").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_cache_model_isolation() {
        let conn = setup_db();
        let vec: Vec<f32> = (0..3072).map(|i| i as f32 * 0.001).collect();
        EmbedCache::put(&conn, "same query", "model-a", &vec).unwrap();
        // Different model should not find the cache entry
        let result = EmbedCache::get(&conn, "same query", "model-b").unwrap();
        assert!(result.is_none(), "different model should not share cache");
        // Same model finds it
        let result = EmbedCache::get(&conn, "same query", "model-a").unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn test_metadata_prefix() {
        use crate::embed::prepend_metadata;
        let result = prepend_metadata("debug", "OOM fix", "connection pool leak");
        assert_eq!(result, "topic:debug | OOM fix | connection pool leak");
    }
}
