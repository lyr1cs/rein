use crate::types::ReinResult;
use rusqlite::Connection;

/// Metadata key holding the monotonic embedding-write counter. Every
/// insert/replace/delete through the two helpers below bumps it. The #17
/// recluster cadence gate reads it because the row COUNT alone is blind to
/// in-place embedding replacement (`SqliteStore::update` re-embeds changed
/// content under the same id — same count, different vector): a vault that
/// only ever updates would otherwise never recluster. `vec_memories` is a
/// vec0 virtual table, so neither triggers nor rowid heuristics can track
/// this — the shared write chokepoints are the reliable place.
pub const EMBEDDING_WRITE_SEQ_KEY: &str = "embedding_write_seq";

fn bump_embedding_write_seq(conn: &Connection) -> ReinResult<()> {
    conn.execute(
        "INSERT INTO metadata(key, value) VALUES (?1, '1')
         ON CONFLICT(key) DO UPDATE SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT)",
        rusqlite::params![EMBEDDING_WRITE_SEQ_KEY],
    )?;
    Ok(())
}

/// Read the monotonic embedding-write counter (0 when never bumped).
pub fn embedding_write_seq(conn: &Connection) -> u64 {
    conn.query_row(
        "SELECT CAST(value AS INTEGER) FROM metadata WHERE key = ?1",
        rusqlite::params![EMBEDDING_WRITE_SEQ_KEY],
        |r| r.get::<_, i64>(0),
    )
    .map(|v| v.max(0) as u64)
    .unwrap_or(0)
}

/// Insert an embedding vector for a memory.
///
/// Counter atomicity (codex R9 + R11): the seq bump and the vec0 write
/// commit under ONE savepoint. Two separate autocommits would open both
/// failure directions — write-then-crash leaves a CHANGED vector the
/// cadence gate never hears about, and bump-then-write lets an adaptive
/// pass on another connection read the bumped counter between the two
/// commits, recluster over the OLD vector, and stamp the new seq as
/// covered. A savepoint nests harmlessly inside caller transactions and
/// forms its own transaction otherwise.
/// [`insert_embedding`] guarded by vector provenance: when the store is
/// stamped for a model (`vec_rows_provenance`, written by warmup / reindex)
/// and the writer's `"<model>:<dims>"` differs, the write is refused so a
/// process still running an older configuration cannot slip an old-model
/// vector into a table attested as the new model.
pub fn insert_embedding_checked(
    conn: &Connection,
    id: &str,
    embedding: &[f32],
    writer_provenance: &str,
) -> ReinResult<()> {
    insert_embedding_inner(conn, id, embedding, Some(writer_provenance))
}

pub fn insert_embedding(conn: &Connection, id: &str, embedding: &[f32]) -> ReinResult<()> {
    insert_embedding_inner(conn, id, embedding, None)
}

/// Compare the store's `vec_rows_provenance` stamp with the writer's
/// `model:dims`. Must run inside the write savepoint AFTER the write lock
/// has been taken (see `insert_embedding_inner`): read outside the lock, a
/// concurrent reindex swap could commit a new stamp between the read and
/// the insert and the old-model vector would land in the new-model table,
/// pass every later provenance check, and never be repaired by warmup.
///
/// An unstamped table is claimed by its first writer when it is empty (a
/// fresh database) and refused while it holds rows of unknown provenance
/// (codex round-13 P1): letting a checked writer add rows to such a table
/// would make it permanently indistinguishable from a legacy one, so a later
/// model change could never be detected and writers of different models
/// could share the table. `rein warmup --trust-existing-vectors` attests the
/// rows, `rein migrate --reindex` re-embeds them.
fn check_vector_provenance(conn: &Connection, writer_provenance: &str) -> ReinResult<()> {
    use rusqlite::OptionalExtension;
    let stamped: Option<String> = conn
        .query_row(
            "SELECT value FROM metadata WHERE key = ?1",
            rusqlite::params![crate::store::schema::VEC_ROWS_PROVENANCE_KEY],
            |row| row.get(0),
        )
        .optional()?;
    match stamped {
        Some(stamped) if stamped == writer_provenance => Ok(()),
        Some(stamped) => Err(crate::types::ReinError::VectorProvenance(format!(
            "vector store is stamped for embedding model {stamped} but this writer uses \
             {writer_provenance}; restart with the current configuration or run \
             `rein migrate --reindex`"
        ))),
        None => {
            let has_rows: bool = conn.query_row(
                "SELECT EXISTS(SELECT 1 FROM vec_memories LIMIT 1)",
                [],
                |row| row.get(0),
            )?;
            if has_rows {
                return Err(crate::types::ReinError::VectorProvenance(format!(
                    "vector store holds rows without model provenance; run \
                     `rein warmup --trust-existing-vectors` if they were produced by \
                     {writer_provenance}, or `rein migrate --reindex` to re-embed them"
                )));
            }
            // First vector of a fresh database: claim the table for this
            // model inside the same savepoint as the row.
            conn.execute(
                "INSERT INTO metadata(key, value) VALUES (?1, ?2)",
                rusqlite::params![
                    crate::store::schema::VEC_ROWS_PROVENANCE_KEY,
                    writer_provenance
                ],
            )?;
            Ok(())
        }
    }
}

static REFUSED_WRITE_WARNED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Log a refused checked write once per process at `warn` (the condition is
/// per database, not per row) and at `debug` afterwards.
pub fn note_refused_vector_write(id: &str, reason: &str) {
    if !REFUSED_WRITE_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        tracing::warn!(
            id,
            "vector write refused; memory kept without a vector and flagged needs_vec_dedup = 1 \
             for retry: {reason}"
        );
    } else {
        tracing::debug!(id, "vector write refused: {reason}");
    }
}

fn insert_embedding_inner(
    conn: &Connection,
    id: &str,
    embedding: &[f32],
    writer_provenance: Option<&str>,
) -> ReinResult<()> {
    let bytes = embedding_to_bytes(embedding);
    conn.execute_batch("SAVEPOINT vec_embed_write")?;
    let result = (|| -> ReinResult<()> {
        // The bump is a write, so it takes the connection's write lock
        // before anything below runs: a concurrent reindex swap has either
        // already committed (and its stamp is the one compared next) or
        // waits for this savepoint to finish. In WAL mode a write attempted
        // on a stale read snapshot fails with SQLITE_BUSY_SNAPSHOT instead
        // of landing, which keeps the check and the write atomic even when
        // the caller opened its read transaction before the swap.
        bump_embedding_write_seq(conn)?;
        if let Some(writer) = writer_provenance {
            check_vector_provenance(conn, writer)?;
        }
        conn.execute(
            "INSERT OR REPLACE INTO vec_memories(id, embedding) VALUES (?1, ?2)",
            rusqlite::params![id, bytes],
        )?;
        Ok(())
    })();
    match result {
        Ok(()) => {
            // Propagate RELEASE failures (codex R12): as the outermost
            // savepoint this IS the commit — SQLITE_FULL / I/O errors here
            // mean nothing was made durable, and swallowing them would
            // report a phantom embedding write. On failure, best-effort
            // unwind the savepoint too (codex R13) — returning with it
            // still open would leave the long-lived connection stuck in a
            // failed transaction holding locks.
            if let Err(e) = conn.execute_batch("RELEASE vec_embed_write") {
                let _ = conn.execute_batch("ROLLBACK TO vec_embed_write");
                let _ = conn.execute_batch("RELEASE vec_embed_write");
                return Err(e.into());
            }
            Ok(())
        }
        Err(e) => {
            let _ = conn.execute_batch("ROLLBACK TO vec_embed_write");
            let _ = conn.execute_batch("RELEASE vec_embed_write");
            Err(e)
        }
    }
}

/// Delete an embedding vector for a memory.
///
/// Bump-after here (unlike `insert_embedding`): the bump is conditioned on
/// a row actually being removed, and a crash between delete and bump is
/// covered anyway — deletions move the ROW COUNT, which is the cadence
/// gate's other churn signal. Only count-invisible in-place replacement
/// needs the bump-first guarantee.
pub fn delete_embedding(conn: &Connection, id: &str) -> ReinResult<()> {
    let removed = conn.execute(
        "DELETE FROM vec_memories WHERE id = ?1",
        rusqlite::params![id],
    )?;
    if removed > 0 {
        bump_embedding_write_seq(conn)?;
    }
    Ok(())
}

/// Over-fetch multiplier when filtering vec results by topic / live status.
///
/// `vec_memories` (sqlite-vec virtual table) requires `LIMIT` inside the
/// `MATCH` query — there is no way to push a join predicate into the ANN
/// scan itself. So we over-fetch the top `limit * VEC_OVERFETCH_MULTIPLIER`
/// candidates and then filter by `status IN (...)` / `superseded_by IS NULL`
/// (Bug #2) and optional `topic` (Bug #O2) in an outer join. The multiplier
/// is bounded so a single recall can never scan the entire table even on
/// stores where most rows are deprecated. 8 is large enough that even a
/// pathological topic with 12% prevalence still yields enough live hits to
/// fill the requested `limit` in expectation.
const VEC_OVERFETCH_MULTIPLIER: usize = 8;
/// Live-status SQL predicate. Excludes only `Deprecated` (terminal dead
/// rows from `apply_evolution`). Superseded rows (`superseded_by IS NOT NULL`
/// with `status='active'`) are NOT excluded — `collapse_results_to_canonicals`
/// in `recall.rs` maps them to their live canonical successor under the
/// canonical-first read model. Filtering them here would silently lose
/// queries that match only the old/evidence text. v0.26.2 R2 Codex F3.
const VEC_LIVE_STATUS_FILTER: &str = "m.status IN ('active', 'updated')";

/// Search for nearest neighbors by embedding vector, filtered to live memories.
///
/// "Live" = `status IN ('active', 'updated')` (see `VEC_LIVE_STATUS_FILTER`).
/// Deprecated rows (set by `apply_evolution`) are excluded. Superseded rows
/// (set by `mark_superseded`, which flips `superseded_by` but leaves
/// `status = 'active'`) are NOT excluded — they are mapped to their live
/// canonical successor in `recall.rs::collapse_results_to_canonicals` under the
/// canonical-first read model. Filtering `superseded_by IS NULL` here would
/// silently drop queries that match only the old/evidence text (v0.26.2 R2 F3).
///
/// `_topic` is accepted for caller-side documentation: callers express
/// "I am topic-restricted" so future readers see the intent at the call
/// site, but the SQL never filters on topic — the actual topic comparison
/// happens in `recall.rs::rank_and_filter` via `crate::ops::normalize_topic_name`
/// two-sided match. Pushing `m.topic = ?` into SQL would silently drop
/// normalized equivalents (e.g. stored `rust-lang` vs caller `Rust Lang`).
/// Bug #O2 + v0.26.2 R2 Codex F2 + R3 F1 (renamed `topic` → `_topic` to
/// silence unused-variable lint without dropping the documented signature).
pub fn search_vec(
    conn: &Connection,
    embedding: &[f32],
    _topic: Option<&str>,
    limit: usize,
) -> ReinResult<Vec<(String, f32)>> {
    let bytes = embedding_to_bytes(embedding);
    // v0.26.2 R2 Codex F1: always over-fetch (regardless of topic) so
    // live-status attrition (deprecated rows surfacing in the top-k ANN
    // hits) doesn't drop us below `limit`. Without this, when the nearest
    // `limit` embeddings happen to all be deprecated, the live filter
    // discards them and live candidates just below the cutoff are never
    // considered. Cost is bounded — vec0 ANN scan is cheap and the JOIN +
    // filter happens on at most `overfetch` rows.
    let overfetch = limit.saturating_mul(VEC_OVERFETCH_MULTIPLIER).max(limit);

    // Inner LIMIT = overfetch caps the ANN scan; the outer JOIN filters
    // dead rows but does NOT re-cap. The Rust-side take(limit) below
    // restores the caller's contract for topic=None callers (dedup/GC
    // path: `embedding_candidate_lookup`, `run_vec_dedup_inner`) which
    // expect `limit` candidates exactly. Topic=Some callers (recall) skip
    // the take so `rank_and_filter` has enough material to apply
    // normalized topic comparison without false negatives. v0.26.2 R4 F1.
    let sql = format!(
        "SELECT v.id, v.distance \
         FROM (SELECT id, distance FROM vec_memories \
               WHERE embedding MATCH ?1 \
               ORDER BY distance \
               LIMIT ?2) v \
         JOIN memories m ON m.id = v.id \
         WHERE {VEC_LIVE_STATUS_FILTER} \
         ORDER BY v.distance"
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(rusqlite::params![bytes, overfetch as i64], |row| {
        let id: String = row.get(0)?;
        let distance: f64 = row.get(1)?;
        Ok((id, distance as f32))
    })?;
    let all = rows.collect::<Result<Vec<_>, _>>()?;
    Ok(if _topic.is_some() {
        all
    } else {
        all.into_iter().take(limit).collect()
    })
}

/// Fetch an embedding vector by memory id, if present.
/// Ids that currently have a stored vector row.
pub fn list_embedding_ids(conn: &Connection) -> ReinResult<std::collections::HashSet<String>> {
    let mut stmt = conn.prepare("SELECT id FROM vec_memories")?;
    let ids = stmt
        .query_map([], |row| row.get::<_, String>(0))?
        .collect::<Result<std::collections::HashSet<String>, _>>()?;
    Ok(ids)
}

pub fn get_embedding(conn: &Connection, id: &str) -> ReinResult<Option<Vec<f32>>> {
    let result: Result<Vec<u8>, _> = conn.query_row(
        "SELECT embedding FROM vec_memories WHERE id = ?1",
        rusqlite::params![id],
        |row| row.get(0),
    );
    match result {
        Ok(bytes) => Ok(Some(bytes_to_embedding(&bytes))),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

fn embedding_to_bytes(embedding: &[f32]) -> Vec<u8> {
    embedding.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn bytes_to_embedding(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::SqliteStore;
    use crate::types::{Importance, Memory, MemoryLayer, MemoryStatus, MemoryStore, Source};
    use chrono::Utc;

    /// Same fixture shape as the FTS tests — explicit status / superseded_by
    /// so the live filter is the only signal driving the test.
    fn fixture(topic: &str) -> Memory {
        Memory {
            id: ulid::Ulid::new().to_string(),
            layer: MemoryLayer::LTM,
            topic: topic.to_string(),
            summary: format!("vec test {topic}"),
            content: format!("vec test {topic} content"),
            keywords: vec![],
            importance: Importance::Medium,
            source: Source::Manual,
            strength: 0.8,
            decay_lambda: 0.02,
            access_count: 0,
            superseded_by: None,
            canonical_id: None,
            support_count: 1,
            merge_count: 0,
            dedup_confidence: 1.0,
            source_diversity: 1.0,
            contradiction_score: 0.0,
            related_ids: vec![],
            concept_ids: vec![],
            status: MemoryStatus::Active,
            embedding: None,
            tier: crate::store::tiering::MemoryTier::Warm,
            cluster_id: None,
            archival_summary: None,
            archival_summary_at: None,
            archival_summary_version: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_accessed: Utc::now(),
        }
    }

    /// Make a deterministic 3072d embedding with a single non-zero coordinate.
    /// Cosine similarity between two such vectors is 0 unless the same axis
    /// is set, so we can place known "near" / "far" candidates by axis index.
    fn one_hot(axis: usize, magnitude: f32) -> Vec<f32> {
        let mut v = vec![0.0f32; 3072];
        v[axis] = magnitude;
        v
    }

    /// Bug #2 (HIGH) + R2 Codex F3: `search_vec` must skip `Deprecated`
    /// (terminal dead rows from `apply_evolution`) but MUST surface
    /// superseded rows (`superseded_by IS NOT NULL` with `status='active'`)
    /// — `collapse_results_to_canonicals` later maps them to the live
    /// canonical successor. Pre-R2 the SQL filter dropped both, which
    /// silently lost queries matching only the old/evidence text.
    #[test]
    fn search_vec_excludes_deprecated_keeps_superseded_for_canonical_collapse() {
        let store = SqliteStore::in_memory().unwrap();

        let live = store.store(fixture("topic-a")).unwrap();
        let updated = store.store(fixture("topic-a")).unwrap();
        let deprecated = store.store(fixture("topic-a")).unwrap();
        let superseded = store.store(fixture("topic-a")).unwrap();

        // Drive each row's status / superseded_by into the configuration the
        // test wants to verify (raw SQL — sidesteps `update()` so we can place
        // exact tombstone shapes including the mark_superseded variant where
        // status stays Active).
        store
            .conn()
            .execute(
                "UPDATE memories SET status = 'updated' WHERE id = ?1",
                rusqlite::params![updated],
            )
            .unwrap();
        store
            .conn()
            .execute(
                "UPDATE memories SET status = 'deprecated' WHERE id = ?1",
                rusqlite::params![deprecated],
            )
            .unwrap();
        store
            .conn()
            .execute(
                "UPDATE memories SET superseded_by = ?2 WHERE id = ?1",
                rusqlite::params![superseded, live],
            )
            .unwrap();

        // Place the dead rows at the closest cosine distance and the live ones
        // farther out, so a faulty filter would let the dead ones win.
        insert_embedding(store.conn(), &deprecated, &one_hot(0, 1.0)).unwrap();
        // Provenance guard: a foreign writer is refused once the table is stamped.
        store
            .conn()
            .execute(
                "INSERT INTO metadata(key, value) VALUES (?1, 'model-a:1')",
                rusqlite::params![crate::store::schema::VEC_ROWS_PROVENANCE_KEY],
            )
            .unwrap();
        assert!(
            insert_embedding_checked(store.conn(), "guarded", &one_hot(0, 0.2), "model-b:1")
                .is_err()
        );
        insert_embedding_checked(store.conn(), "guarded", &one_hot(0, 0.2), "model-a:1").unwrap();
        store
            .conn()
            .execute(
                "DELETE FROM metadata WHERE key = ?1",
                rusqlite::params![crate::store::schema::VEC_ROWS_PROVENANCE_KEY],
            )
            .unwrap();
        delete_embedding(store.conn(), "guarded").unwrap();
        insert_embedding(store.conn(), &superseded, &one_hot(0, 0.99)).unwrap();
        insert_embedding(store.conn(), &updated, &one_hot(0, 0.5)).unwrap();
        insert_embedding(store.conn(), &live, &one_hot(0, 0.4)).unwrap();

        let results = search_vec(store.conn(), &one_hot(0, 1.0), None, 4).unwrap();
        let ids: Vec<&str> = results.iter().map(|(id, _)| id.as_str()).collect();
        assert!(ids.contains(&live.as_str()), "live Active row must surface");
        assert!(ids.contains(&updated.as_str()), "Updated row must surface");
        assert!(
            !ids.contains(&deprecated.as_str()),
            "Deprecated row must be hidden by the live-status filter"
        );
        // R2 Codex F3: superseded rows MUST surface so the canonical-first
        // collapse step downstream can map them to the live successor.
        assert!(
            ids.contains(&superseded.as_str()),
            "superseded row must surface for canonical-collapse mapping"
        );
    }

    /// Bug #O2 + v0.26.2 R2 Codex finding F2: when `topic` is `Some`,
    /// `search_vec` must over-fetch enough candidates to give the
    /// post-fetch normalized topic comparison in `recall.rs::rank_and_filter`
    /// a fighting chance. The actual topic comparison stays in Rust because
    /// pushing `m.topic = ?` into SQL silently drops normalized equivalents
    /// (e.g. stored `rust-lang` vs caller `Rust Lang`).
    #[test]
    fn search_vec_with_topic_overfetches_for_post_filter() {
        let store = SqliteStore::in_memory().unwrap();

        // Two memories on topic A (placed FAR from the query) and several on
        // topic B (placed NEAR the query). With over-fetch, the topic-A
        // rows must appear in `search_vec`'s output even though they are
        // cosine-orthogonal to the query.
        let a1 = store.store(fixture("topic-a")).unwrap();
        let a2 = store.store(fixture("topic-a")).unwrap();
        let b_ids: Vec<String> = (0..6)
            .map(|_| store.store(fixture("topic-b")).unwrap())
            .collect();

        // topic-A vectors live on axis 0; topic-B vectors on axis 1. The
        // query is axis 1 → topic-B is "near" (cosine 1), topic-A is
        // orthogonal (cosine 0). Without over-fetch, `limit=3` would only
        // return topic-B.
        insert_embedding(store.conn(), &a1, &one_hot(0, 1.0)).unwrap();
        insert_embedding(store.conn(), &a2, &one_hot(0, 0.5)).unwrap();
        for (i, id) in b_ids.iter().enumerate() {
            insert_embedding(store.conn(), id, &one_hot(1, 1.0 - i as f32 * 0.05)).unwrap();
        }

        let query = one_hot(1, 1.0);

        // No topic filter → returned set is ordered by distance ascending.
        // R2 F1: search_vec ALWAYS over-fetches now (regardless of topic),
        // so the cap is `limit * over-fetch-multiplier`, not `limit`. The
        // top-K cap is the caller's responsibility (rank_and_filter does
        // it). What we pin here is the ORDERING: the first 3 entries must
        // all be topic-B (closest-by-distance).
        let unfiltered = search_vec(store.conn(), &query, None, 3).unwrap();
        assert!(
            unfiltered.len() >= 3,
            "over-fetched set must contain at least `limit` rows; got {}",
            unfiltered.len()
        );
        assert!(
            unfiltered
                .iter()
                .take(3)
                .all(|(id, _)| b_ids.contains(&id.to_string())),
            "the top-3 closest must all come from topic-B \
             (otherwise the test fixture is broken)"
        );

        // With topic="topic-a" → over-fetched superset MUST contain both
        // topic-A rows so the post-filter in rank_and_filter can pick them
        // out. The result is NOT topic-filtered here — that happens in
        // `recall.rs::rank_and_filter` with normalize_topic_name.
        let overfetched = search_vec(store.conn(), &query, Some("topic-a"), 3).unwrap();
        let returned_ids: std::collections::HashSet<&str> =
            overfetched.iter().map(|(id, _)| id.as_str()).collect();
        assert!(
            returned_ids.contains(a1.as_str()),
            "over-fetch must include topic-A row a1; got {returned_ids:?}"
        );
        assert!(
            returned_ids.contains(a2.as_str()),
            "over-fetch must include topic-A row a2; got {returned_ids:?}"
        );
        assert!(
            overfetched.len() > 3,
            "topic-restricted call must over-fetch beyond `limit` so the \
             post-filter has material; got {} rows",
            overfetched.len()
        );
    }
}

#[cfg(test)]
mod checked_write_tests {
    use super::*;
    use crate::store::schema::VEC_ROWS_PROVENANCE_KEY;

    fn stamp(conn: &Connection, value: &str) {
        conn.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?1, ?2)",
            rusqlite::params![VEC_ROWS_PROVENANCE_KEY, value],
        )
        .unwrap();
    }

    fn vec_rows(conn: &Connection) -> i64 {
        conn.query_row("SELECT COUNT(*) FROM vec_memories", [], |r| r.get(0))
            .unwrap()
    }

    /// Codex round-11 P1: the stamp comparison and the vector write must be
    /// one atomic unit. A writer that read the OLD stamp before a reindex
    /// swap committed must not be able to land its old-model vector in the
    /// new-model table afterwards.
    #[test]
    fn checked_insert_cannot_land_after_stamp_moved_under_a_stale_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("memories.db");
        let dims = 4usize;
        let store = crate::store::SqliteStore::new(&path, "model-a", dims).unwrap();
        let writer = store.conn();
        writer.execute_batch("PRAGMA journal_mode=WAL").unwrap();
        stamp(writer, "model-a:4");

        // The old-model writer opens a read transaction and observes the
        // old stamp (what the pre-fix code did outside any transaction).
        writer.execute_batch("BEGIN").unwrap();
        let seen: String = writer
            .query_row(
                "SELECT value FROM metadata WHERE key = ?1",
                rusqlite::params![VEC_ROWS_PROVENANCE_KEY],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(seen, "model-a:4");

        // A reindex on another connection swaps the stamp and commits; WAL
        // readers do not block writers so this succeeds immediately.
        let swapper = rusqlite::Connection::open(&path).unwrap();
        swapper
            .busy_timeout(std::time::Duration::from_secs(5))
            .unwrap();
        stamp(&swapper, "model-b:4");

        // The stale writer's checked insert must fail rather than land a
        // model-a vector in the model-b table.
        let err = insert_embedding_checked(writer, "m1", &[0.1; 4], "model-a:4")
            .expect_err("write on a stale snapshot must not land");
        assert!(!err.to_string().is_empty(), "error must carry a reason");
        writer.execute_batch("ROLLBACK").unwrap();
        assert_eq!(vec_rows(writer), 0, "no mixed-model row may exist");

        // Outside the stale transaction the old-model writer is refused by
        // the stamp itself, and the new-model writer is accepted.
        let err = insert_embedding_checked(writer, "m1", &[0.1; 4], "model-a:4").unwrap_err();
        assert!(
            err.to_string()
                .contains("stamped for embedding model model-b:4"),
            "{err}"
        );
        assert_eq!(vec_rows(writer), 0);
        insert_embedding_checked(writer, "m1", &[0.1; 4], "model-b:4").unwrap();
        assert_eq!(vec_rows(writer), 1);
    }

    fn stamp_value(conn: &Connection) -> Option<String> {
        use rusqlite::OptionalExtension;
        conn.query_row(
            "SELECT value FROM metadata WHERE key = ?1",
            rusqlite::params![VEC_ROWS_PROVENANCE_KEY],
            |r| r.get(0),
        )
        .optional()
        .unwrap()
    }

    /// Codex round-13 P1: the first checked write on an empty, unstamped
    /// table claims it for the writer's model.
    #[test]
    fn first_checked_write_claims_an_empty_unstamped_table() {
        let store = crate::store::SqliteStore::in_memory().unwrap();
        let conn = store.conn();
        let dims = store.dims;
        assert!(stamp_value(conn).is_none());
        insert_embedding_checked(conn, "m1", &vec![0.5; dims], "mine:3072").unwrap();
        assert_eq!(stamp_value(conn).as_deref(), Some("mine:3072"));
        let err = insert_embedding_checked(conn, "m2", &vec![0.5; dims], "other:3072").unwrap_err();
        assert!(
            matches!(err, crate::types::ReinError::VectorProvenance(_)),
            "{err}"
        );
        assert_eq!(vec_rows(conn), 1);
    }

    /// Codex round-13 P1: rows of unknown provenance refuse every checked
    /// write until the operator attests or re-embeds them.
    #[test]
    fn checked_write_refuses_an_unstamped_nonempty_table() {
        let store = crate::store::SqliteStore::in_memory().unwrap();
        let conn = store.conn();
        let dims = store.dims;
        // Legacy row written by an older binary: no stamp.
        insert_embedding(conn, "legacy", &vec![0.5; dims]).unwrap();
        assert!(stamp_value(conn).is_none());
        let err = insert_embedding_checked(conn, "m1", &vec![0.5; dims], "mine:3072").unwrap_err();
        assert!(
            matches!(err, crate::types::ReinError::VectorProvenance(_)),
            "{err}"
        );
        assert!(
            err.to_string().contains("without model provenance"),
            "{err}"
        );
        assert!(
            stamp_value(conn).is_none(),
            "a refused write must not stamp"
        );
        assert_eq!(vec_rows(conn), 1);
    }

    /// A refused checked write leaves the write sequence untouched: the
    /// bump runs inside the same savepoint and is rolled back with it.
    #[test]
    fn refused_checked_insert_rolls_back_the_write_seq_bump() {
        let store = crate::store::SqliteStore::in_memory().unwrap();
        let conn = store.conn();
        stamp(conn, "other:3072");
        let before = embedding_write_seq(conn);
        let dims = store.dims;
        let err = insert_embedding_checked(conn, "m1", &vec![0.5; dims], "mine:3072").unwrap_err();
        assert!(err
            .to_string()
            .contains("stamped for embedding model other:3072"));
        assert_eq!(embedding_write_seq(conn), before);
        assert_eq!(vec_rows(conn), 0);
    }
}
