//! Durable record of memories the embedding provider refuses.
//!
//! A text the provider rejects when it is sent alone (too long for the
//! model, unacceptable content) is rejected again every time it is sent
//! unchanged. Without a record, every warmup, reindex and vector-dedup pass
//! re-sends it, and a job that treats any failure as fatal never completes.
//!
//! One `metadata` row per memory, keyed `unembeddable:<memory id>`, holding
//! the vector provenance (`model:dimensions`) and the hash of the exact text
//! that was refused. A record only applies while both still match: editing
//! the memory or changing the embedding model makes it stale, and the
//! memory is tried again.

use crate::types::error::ReinResult;
use rusqlite::{Connection, OptionalExtension};

const KEY_PREFIX: &str = "unembeddable:";

/// What the record says about embedding `text` for a memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// No record for this memory.
    Unknown,
    /// This exact text was refused under this provenance: do not send it.
    Refused,
    /// A record exists for another text or model; try again.
    Stale,
}

fn key(id: &str) -> String {
    format!("{KEY_PREFIX}{id}")
}

/// The stored value for a refusal of `text` under `provenance`. Callers that
/// must decide later whether to record (see
/// [`crate::embed::BulkEmbedSession::any_success`]) keep this instead of
/// the text.
pub fn fingerprint(provenance: &str, text: &str) -> String {
    value(provenance, text)
}

/// Record a refusal from a [`fingerprint`] taken earlier.
pub fn record_fingerprint(conn: &Connection, id: &str, fingerprint: &str) -> ReinResult<()> {
    conn.execute(
        "INSERT INTO metadata (key, value) VALUES (?1, ?2)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        rusqlite::params![key(id), fingerprint],
    )?;
    Ok(())
}

fn value(provenance: &str, text: &str) -> String {
    use sha2::{Digest, Sha256};
    format!("{provenance}\n{:x}", Sha256::digest(text.as_bytes()))
}

/// Look up the record for `id` against the text and provenance about to be
/// embedded. A read failure is `Unknown`: the memory is simply tried.
pub fn verdict(conn: &Connection, id: &str, provenance: &str, text: &str) -> Verdict {
    let stored: Option<String> = conn
        .query_row(
            "SELECT value FROM metadata WHERE key = ?1",
            [key(id)],
            |row| row.get::<_, Option<String>>(0),
        )
        .optional()
        .ok()
        .flatten()
        .flatten();
    match stored {
        None => Verdict::Unknown,
        Some(stored) if stored == value(provenance, text) => Verdict::Refused,
        Some(_) => Verdict::Stale,
    }
}

/// Record that the provider refused `text` for memory `id`.
pub fn record(conn: &Connection, id: &str, provenance: &str, text: &str) -> ReinResult<()> {
    conn.execute(
        "INSERT INTO metadata (key, value) VALUES (?1, ?2)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        rusqlite::params![key(id), value(provenance, text)],
    )?;
    Ok(())
}

/// Drop every record. A full re-embedding judges every memory again.
pub fn clear_all(conn: &Connection) -> ReinResult<usize> {
    Ok(conn.execute(
        "DELETE FROM metadata WHERE substr(key, 1, ?1) = ?2",
        rusqlite::params![KEY_PREFIX.len() as i64, KEY_PREFIX],
    )?)
}

/// Drop the record for `id` (its text was embedded after all).
pub fn clear(conn: &Connection, id: &str) -> ReinResult<()> {
    conn.execute("DELETE FROM metadata WHERE key = ?1", [key(id)])?;
    Ok(())
}

/// Number of records written under `provenance`.
pub fn count(conn: &Connection, provenance: &str) -> usize {
    conn.query_row(
        "SELECT COUNT(*) FROM metadata
         WHERE substr(key, 1, ?1) = ?2 AND substr(value, 1, ?3) = ?4",
        rusqlite::params![
            KEY_PREFIX.len() as i64,
            KEY_PREFIX,
            provenance.len() as i64 + 1,
            format!("{provenance}\n")
        ],
        |row| row.get::<_, i64>(0),
    )
    .map(|n| n.max(0) as usize)
    .unwrap_or(0)
}

/// Remove records whose memory no longer exists.
pub fn prune_orphans(conn: &Connection) -> ReinResult<usize> {
    Ok(conn.execute(
        "DELETE FROM metadata
         WHERE substr(key, 1, ?1) = ?2
           AND substr(key, ?1 + 1) NOT IN (SELECT id FROM memories)",
        rusqlite::params![KEY_PREFIX.len() as i64, KEY_PREFIX],
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conn() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT);
             CREATE TABLE memories (id TEXT PRIMARY KEY);
             INSERT INTO memories VALUES ('m1');",
        )
        .unwrap();
        conn
    }

    #[test]
    fn a_record_applies_only_to_the_same_text_and_model() {
        let conn = conn();
        assert_eq!(verdict(&conn, "m1", "model-a:8", "text"), Verdict::Unknown);

        record(&conn, "m1", "model-a:8", "text").unwrap();
        assert_eq!(verdict(&conn, "m1", "model-a:8", "text"), Verdict::Refused);
        assert_eq!(verdict(&conn, "m1", "model-a:8", "edited"), Verdict::Stale);
        assert_eq!(verdict(&conn, "m1", "model-b:8", "text"), Verdict::Stale);
        assert_eq!(count(&conn, "model-a:8"), 1);
        assert_eq!(count(&conn, "model-a:80"), 0);
        assert_eq!(count(&conn, "model-b:8"), 0);

        // Recording again replaces the row.
        record(&conn, "m1", "model-b:8", "text").unwrap();
        assert_eq!(verdict(&conn, "m1", "model-b:8", "text"), Verdict::Refused);
        assert_eq!(count(&conn, "model-a:8"), 0);

        clear(&conn, "m1").unwrap();
        assert_eq!(verdict(&conn, "m1", "model-b:8", "text"), Verdict::Unknown);
    }

    #[test]
    fn clear_all_removes_only_the_records() {
        let conn = conn();
        record(&conn, "m1", "model-a:8", "text").unwrap();
        conn.execute("INSERT INTO metadata VALUES ('other', 'kept')", [])
            .unwrap();
        assert_eq!(clear_all(&conn).unwrap(), 1);
        assert_eq!(verdict(&conn, "m1", "model-a:8", "text"), Verdict::Unknown);
        let kept: i64 = conn
            .query_row("SELECT COUNT(*) FROM metadata", [], |r| r.get(0))
            .unwrap();
        assert_eq!(kept, 1);
    }

    #[test]
    fn records_of_deleted_memories_are_pruned() {
        let conn = conn();
        record(&conn, "m1", "model-a:8", "text").unwrap();
        record(&conn, "gone", "model-a:8", "text").unwrap();
        conn.execute("INSERT INTO metadata VALUES ('other', 'kept')", [])
            .unwrap();

        assert_eq!(prune_orphans(&conn).unwrap(), 1);
        assert_eq!(verdict(&conn, "m1", "model-a:8", "text"), Verdict::Refused);
        assert_eq!(
            verdict(&conn, "gone", "model-a:8", "text"),
            Verdict::Unknown
        );
        let other: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM metadata WHERE key = 'other'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(other, 1);
    }
}
