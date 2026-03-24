use crate::types::error::ReinResult;
use rusqlite::Connection;
use sha2::{Digest, Sha256};

pub struct EmbedCache;

impl EmbedCache {
    /// Look up a cached embedding by query text.
    pub fn get(conn: &Connection, query: &str) -> ReinResult<Option<Vec<f32>>> {
        let hash = Self::hash_query(query);
        let mut stmt =
            conn.prepare("SELECT embedding FROM embed_cache WHERE query_hash = ?1")?;
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

    /// Store an embedding in the cache, keyed by query text.
    pub fn put(conn: &Connection, query: &str, embedding: &[f32]) -> ReinResult<()> {
        let hash = Self::hash_query(query);
        let blob = Self::f32_to_bytes(embedding);
        let now = chrono::Utc::now().to_rfc3339();
        conn.execute(
            "INSERT OR REPLACE INTO embed_cache (query_hash, embedding, created_at) VALUES (?1, ?2, ?3)",
            rusqlite::params![hash, blob, now],
        )?;
        Ok(())
    }

    fn hash_query(query: &str) -> String {
        let mut hasher = Sha256::new();
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
        init_schema(&conn).unwrap();
        conn
    }

    #[test]
    fn test_cache_roundtrip() {
        let conn = setup_db();
        let mut vec = Vec::with_capacity(3072);
        for i in 0..3072 {
            vec.push(i as f32 * 0.001);
        }

        EmbedCache::put(&conn, "test query", &vec).unwrap();
        let result = EmbedCache::get(&conn, "test query").unwrap().unwrap();

        assert_eq!(result.len(), 3072);
        for (a, b) in vec.iter().zip(result.iter()) {
            assert!((a - b).abs() < f32::EPSILON, "mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_cache_miss() {
        let conn = setup_db();
        let result = EmbedCache::get(&conn, "nonexistent query").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_metadata_prefix() {
        use crate::embed::google::prepend_metadata;
        let result = prepend_metadata("debug", "OOM fix", "connection pool leak");
        assert_eq!(result, "topic:debug | OOM fix | connection pool leak");
    }
}
