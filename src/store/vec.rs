use crate::types::ReinResult;
use rusqlite::Connection;

/// Insert an embedding vector for a memory.
pub fn insert_embedding(conn: &Connection, id: &str, embedding: &[f32]) -> ReinResult<()> {
    let bytes = embedding_to_bytes(embedding);
    conn.execute(
        "INSERT OR REPLACE INTO vec_memories(id, embedding) VALUES (?1, ?2)",
        rusqlite::params![id, bytes],
    )?;
    Ok(())
}

/// Delete an embedding vector for a memory.
pub fn delete_embedding(conn: &Connection, id: &str) -> ReinResult<()> {
    conn.execute("DELETE FROM vec_memories WHERE id = ?1", rusqlite::params![id])?;
    Ok(())
}

/// Search for nearest neighbors by embedding vector.
pub fn search_vec(
    conn: &Connection,
    embedding: &[f32],
    limit: usize,
) -> ReinResult<Vec<(String, f32)>> {
    let bytes = embedding_to_bytes(embedding);
    let mut stmt = conn.prepare(
        "SELECT id, distance
         FROM vec_memories
         WHERE embedding MATCH ?1
         ORDER BY distance
         LIMIT ?2",
    )?;
    let rows = stmt.query_map(rusqlite::params![bytes, limit as i64], |row| {
        let id: String = row.get(0)?;
        let distance: f64 = row.get(1)?;
        Ok((id, distance as f32))
    })?;
    Ok(rows.collect::<Result<Vec<_>, _>>()?)
}

fn embedding_to_bytes(embedding: &[f32]) -> Vec<u8> {
    embedding
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect()
}
