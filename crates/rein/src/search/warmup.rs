//! Background embedding cache warmup.
//! Pre-computes embeddings for all memories that don't have cached vectors,
//! eliminating the 255ms Google API delay during recall.

use crate::config::ReinConfig;
use crate::embed::{create_embedder, prepend_metadata, EmbedCache};
use crate::store::SqliteStore;
use crate::types::traits::MemoryStore as _;
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
/// Outcome of one `warmup` pass.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct WarmupReport {
    /// Live memories that had neither a vector row nor a cache entry and were
    /// embedded through the provider.
    pub embedded: usize,
    /// Live memories whose embedding was still in `embed_cache` but had no
    /// `vec_memories` row; the row was written without a provider call.
    pub backfilled_from_cache: usize,
    /// Provider or storage failures.
    pub errors: usize,
    /// Cache misses left unembedded because no embedding provider is configured.
    pub skipped_no_provider: usize,
    /// The embedding model differed from the last complete pass, so every live
    /// row was scheduled for re-embedding.
    pub model_changed: bool,
}

impl WarmupReport {
    /// Rows added to `vec_memories` by this pass.
    pub fn rows_added(&self) -> usize {
        self.embedded + self.backfilled_from_cache
    }
}

/// Make every live memory (`status IN ('active', 'updated')`) durable in the
/// vector store, then rebuild HNSW.
///
/// `embed_cache` is a bounded, time-evicting cache (30-day cleanup once it
/// exceeds 5 000 rows), so an embedding that only lives there disappears; the
/// `vec_memories` row is the durable copy that recall, sqlite-vec replay and
/// the HNSW rebuild read. Earlier versions only refilled the cache, which is
/// why coverage never converged on long-lived databases.
pub async fn warmup(store: &SqliteStore, config: &ReinConfig) -> WarmupReport {
    let embedder = create_embedder(config);
    warmup_with_embedder(store, config, embedder.as_ref()).await
}

/// [`warmup`] with an explicit provider (tests inject a mock).
pub async fn warmup_with_embedder(
    store: &SqliteStore,
    config: &ReinConfig,
    embedder: Option<&crate::embed::EmbedderKind>,
) -> WarmupReport {
    // v0.30.2 B3 / B6: orphan-stage cleanup BEFORE any rebuild decision so a
    // crash mid-swap from a previous run can't leave the search subsystem
    // staring at a half-built `.new` dir.
    let db_path = store.db_path();
    if db_path.to_str() != Some(":memory:") {
        // v0.30.3 codex R20 P2: migrate legacy `.tantivy/.dirty` markers
        // to the new sibling `.tantivy.dirty` location. Pre-v0.30.3
        // markers were inside the swapped dir; this rename moves them
        // out so they survive swap. Run BEFORE cleanup so we don't
        // delete a marker we should have migrated.
        let legacy = tantivy_dirty_path_legacy(db_path);
        if legacy.exists() {
            let canonical = tantivy_dirty_path(db_path);
            if !canonical.exists() {
                let _ = std::fs::rename(&legacy, &canonical);
            } else {
                let _ = std::fs::remove_file(&legacy);
            }
        }
        cleanup_tantivy_staging(db_path);
        cleanup_hnsw_staging(db_path);
    }

    // v0.30.2 B3 / B6: cold-start rebuild now gated on
    // `[warmup].always_rebuild_side_indexes` (default true preserves prior
    // behavior) AND on the missing-or-dirty signal. With the flag flipped to
    // false in an operator config, we only rebuild when there's a concrete
    // reason (no index on disk or a dirty marker).
    let should_rebuild_tantivy = side_index_rebuild_needed_tantivy(store, db_path, config);
    let should_rebuild_hnsw = side_index_rebuild_needed_hnsw(db_path, config);

    if should_rebuild_tantivy {
        populate_tantivy(store);
    } else {
        tracing::debug!("warmup: skipping cold-start tantivy rebuild (gate satisfied)");
    }
    if should_rebuild_hnsw {
        populate_hnsw(store, config);
    } else {
        tracing::debug!("warmup: skipping cold-start hnsw rebuild (gate satisfied)");
    }

    if embedder.is_none() {
        tracing::info!(
            "no embedding provider configured; warmup restores vector rows from cache only"
        );
    }
    let report = backfill_missing_vec_rows(store, config, embedder).await;
    tracing::info!(
        embedded = report.embedded,
        backfilled_from_cache = report.backfilled_from_cache,
        errors = report.errors,
        skipped_no_provider = report.skipped_no_provider,
        "warmup complete"
    );

    if report.rows_added() > 0 {
        // Rebuild side indexes so the new rows are searchable. A model
        // migration only counts rows once it was published atomically, so
        // this can never promote a mixed old/new-model table.
        populate_hnsw(store, config);
    }

    report
}

/// Write `embedding` as the vector row for `id`. When a row already exists
/// the delete + insert pair runs inside a savepoint so a failed insert (for
/// example a dimension mismatch) leaves the previous vector in place instead
/// of silently lowering coverage.
pub(crate) fn replace_embedding_row(
    conn: &rusqlite::Connection,
    id: &str,
    embedding: &[f32],
    existed: bool,
) -> crate::types::ReinResult<()> {
    if !existed {
        return crate::store::vec::insert_embedding(conn, id, embedding);
    }
    conn.execute_batch("SAVEPOINT vec_replace")?;
    let result = crate::store::vec::delete_embedding(conn, id)
        .and_then(|_| crate::store::vec::insert_embedding(conn, id, embedding));
    match result {
        Ok(()) => {
            if let Err(e) = conn.execute_batch("RELEASE vec_replace") {
                let _ = conn.execute_batch("ROLLBACK TO vec_replace");
                let _ = conn.execute_batch("RELEASE vec_replace");
                return Err(e.into());
            }
            Ok(())
        }
        Err(e) => {
            let _ = conn.execute_batch("ROLLBACK TO vec_replace");
            let _ = conn.execute_batch("RELEASE vec_replace");
            Err(e)
        }
    }
}

/// Insert a `vec_memories` row for every live memory that lacks one, using
/// the cache when it still holds the embedding and the provider otherwise.
/// Rows are written inside SQLite; the HNSW side index is left to the caller
/// (`populate_hnsw`), matching the side-index discipline used everywhere else.
pub async fn backfill_missing_vec_rows(
    store: &SqliteStore,
    config: &ReinConfig,
    embedder: Option<&crate::embed::EmbedderKind>,
) -> WarmupReport {
    let model = config.embedding_model();
    let dims = config.embedding.dimensions;
    let mut report = WarmupReport::default();

    // Vector rows carry no per-row model tag. Remember which model the last
    // complete warmup wrote rows with; a different model (even with the same
    // dimensions) means every existing row is stale and must be re-embedded,
    // otherwise new-model query vectors would be compared with old-model
    // document vectors.
    let provenance = format!("{model}:{dims}");
    let stored_provenance = read_metadata(store, VEC_ROWS_PROVENANCE_KEY);
    let reembed_all = stored_provenance
        .as_deref()
        .is_some_and(|stored| stored != provenance);

    let existing = match crate::store::vec::list_embedding_ids(store.conn()) {
        Ok(ids) => ids,
        Err(e) => {
            tracing::warn!("warmup: failed to list vector rows: {e}");
            report.errors += 1;
            return report;
        }
    };

    // Collect the rows to work on so memory stays bounded by the gap (or,
    // for a migration, by the live set) and no write happens while the row
    // cursor is open.
    let mut pending: Vec<(String, String)> = Vec::new();
    let mut live_total = 0_usize;
    if let Err(e) = store.for_each_for_warmup(|row| {
        if row.status != "active" && row.status != "updated" {
            return Ok(());
        }
        live_total += 1;
        if !reembed_all && existing.contains(&row.id) {
            return Ok(());
        }
        pending.push((
            row.id,
            prepend_metadata(&row.topic, &row.summary, &row.content),
        ));
        Ok(())
    }) {
        tracing::warn!("warmup: failed to list memories: {e}");
        report.errors += 1;
        return report;
    }

    if reembed_all {
        report.model_changed = true;
        tracing::warn!(
            stored = stored_provenance.as_deref().unwrap_or(""),
            current = %provenance,
            live_total,
            "warmup: embedding model changed since the last warmup; re-embedding every live memory"
        );
        migrate_live_rows(
            store,
            embedder,
            &model,
            dims,
            &provenance,
            &existing,
            pending,
            &mut report,
        )
        .await;
        return report;
    }

    if pending.is_empty() {
        tracing::info!(
            live_total,
            "warmup: every live memory already has a vector row"
        );
        write_metadata(store, VEC_ROWS_PROVENANCE_KEY, &provenance);
        return report;
    }
    tracing::info!(
        live_total,
        missing = pending.len(),
        "warmup: live memories without a vector row"
    );

    let mut to_embed: Vec<(String, String)> = Vec::new();
    for (id, text) in pending {
        match EmbedCache::get(store.conn(), &text, &model) {
            Ok(Some(embedding)) if embedding.len() == dims => {
                match replace_embedding_row(store.conn(), &id, &embedding, existing.contains(&id)) {
                    Ok(()) => report.backfilled_from_cache += 1,
                    Err(e) => {
                        tracing::warn!(id, "warmup: failed to restore vector row from cache: {e}");
                        report.errors += 1;
                    }
                }
            }
            _ => to_embed.push((id, text)),
        }
    }

    if to_embed.is_empty() {
        if report.errors == 0 {
            write_metadata(store, VEC_ROWS_PROVENANCE_KEY, &provenance);
        }
        return report;
    }
    let Some(embedder) = embedder else {
        report.skipped_no_provider = to_embed.len();
        return report;
    };

    for chunk in to_embed.chunks(100) {
        let texts: Vec<&str> = chunk.iter().map(|(_, text)| text.as_str()).collect();
        match embedder.embed_batch(&texts).await {
            Ok(embeddings) => {
                for ((id, text), embedding) in chunk.iter().zip(embeddings.iter()) {
                    if embedding.len() != dims {
                        tracing::warn!(
                            id,
                            got = embedding.len(),
                            expected = dims,
                            "warmup: provider returned an embedding of the wrong size"
                        );
                        report.errors += 1;
                        continue;
                    }
                    if let Err(e) = EmbedCache::put(store.conn(), text, &model, embedding) {
                        tracing::warn!(id, "warmup: failed to cache embedding: {e}");
                    }
                    match replace_embedding_row(store.conn(), id, embedding, existing.contains(id))
                    {
                        Ok(()) => report.embedded += 1,
                        Err(e) => {
                            tracing::warn!(id, "warmup: failed to insert vector row: {e}");
                            report.errors += 1;
                        }
                    }
                }
                if embeddings.len() < chunk.len() {
                    report.errors += chunk.len() - embeddings.len();
                }
            }
            Err(e) => {
                tracing::warn!("warmup batch failed: {e}");
                report.errors += chunk.len();
            }
        }
    }

    if report.errors == 0 {
        write_metadata(store, VEC_ROWS_PROVENANCE_KEY, &provenance);
    }
    report
}

/// Embedding-model migration: every live row is re-embedded under the new
/// model and the vector table is replaced in ONE transaction only after every
/// embedding succeeded. A partial result is never written, so old- and
/// new-model vectors can never coexist in `vec_memories` (which recall reads
/// directly and the HNSW rebuild promotes). Successful embeddings are cached
/// before the publish step so a retry after a provider failure is cheap.
#[allow(clippy::too_many_arguments)]
async fn migrate_live_rows(
    store: &SqliteStore,
    embedder: Option<&crate::embed::EmbedderKind>,
    model: &str,
    dims: usize,
    provenance: &str,
    existing: &std::collections::HashSet<String>,
    pending: Vec<(String, String)>,
    report: &mut WarmupReport,
) {
    let mut staged: Vec<(String, Vec<f32>)> = Vec::with_capacity(pending.len());
    let mut to_embed: Vec<(String, String)> = Vec::new();
    for (id, text) in pending {
        match EmbedCache::get(store.conn(), &text, model) {
            Ok(Some(embedding)) if embedding.len() == dims => {
                report.backfilled_from_cache += 1;
                staged.push((id, embedding));
            }
            _ => to_embed.push((id, text)),
        }
    }
    if !to_embed.is_empty() {
        let Some(embedder) = embedder else {
            report.skipped_no_provider = to_embed.len();
            report.backfilled_from_cache = 0;
            tracing::warn!(
                "warmup: model migration needs an embedding provider; no rows were changed"
            );
            return;
        };
        for chunk in to_embed.chunks(100) {
            let texts: Vec<&str> = chunk.iter().map(|(_, text)| text.as_str()).collect();
            match embedder.embed_batch(&texts).await {
                Ok(embeddings) if embeddings.len() == chunk.len() => {
                    for ((id, text), embedding) in chunk.iter().zip(embeddings) {
                        if embedding.len() != dims {
                            report.errors += 1;
                            continue;
                        }
                        if let Err(e) = EmbedCache::put(store.conn(), text, model, &embedding) {
                            tracing::warn!(id, "warmup: failed to cache embedding: {e}");
                        }
                        staged.push((id.clone(), embedding));
                    }
                }
                Ok(embeddings) => {
                    report.errors += chunk.len() - embeddings.len().min(chunk.len());
                }
                Err(e) => {
                    tracing::warn!("warmup migration batch failed: {e}");
                    report.errors += chunk.len();
                }
            }
            if report.errors > 0 {
                break;
            }
        }
    }
    if report.errors > 0 {
        tracing::warn!(
            errors = report.errors,
            staged = staged.len(),
            "warmup: model migration incomplete; vector rows left untouched (re-run warmup)"
        );
        report.embedded = 0;
        report.backfilled_from_cache = 0;
        return;
    }

    // Publish atomically: replace live rows, drop rows for memories that are
    // no longer live, record provenance.
    let conn = store.conn();
    let publish = || -> crate::types::ReinResult<()> {
        conn.execute_batch("BEGIN IMMEDIATE")?;
        let result = (|| -> crate::types::ReinResult<()> {
            let staged_ids: std::collections::HashSet<&str> =
                staged.iter().map(|(id, _)| id.as_str()).collect();
            for stale in existing
                .iter()
                .filter(|id| !staged_ids.contains(id.as_str()))
            {
                crate::store::vec::delete_embedding(conn, stale)?;
            }
            for (id, embedding) in &staged {
                replace_embedding_row(conn, id, embedding, existing.contains(id))?;
            }
            conn.execute(
                "INSERT INTO metadata(key, value) VALUES (?1, ?2) \
                 ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                rusqlite::params![VEC_ROWS_PROVENANCE_KEY, provenance],
            )?;
            Ok(())
        })();
        match result {
            Ok(()) => conn.execute_batch("COMMIT").map_err(Into::into),
            Err(e) => {
                let _ = conn.execute_batch("ROLLBACK");
                Err(e)
            }
        }
    };
    match publish() {
        Ok(()) => {
            report.embedded = staged.len() - report.backfilled_from_cache;
            tracing::info!(
                rows = staged.len(),
                "warmup: embedding model migration published"
            );
        }
        Err(e) => {
            tracing::warn!("warmup: model migration publish failed, rolled back: {e}");
            report.errors += 1;
            report.embedded = 0;
            report.backfilled_from_cache = 0;
        }
    }
}

/// Metadata key recording `"<model>:<dims>"` of the last complete warmup.
pub const VEC_ROWS_PROVENANCE_KEY: &str = "vec_rows_provenance";

fn read_metadata(store: &SqliteStore, key: &str) -> Option<String> {
    use rusqlite::OptionalExtension;
    store
        .conn()
        .query_row(
            "SELECT value FROM metadata WHERE key = ?1",
            rusqlite::params![key],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .ok()
        .flatten()
}

fn write_metadata(store: &SqliteStore, key: &str, value: &str) {
    if let Err(e) = store.conn().execute(
        "INSERT INTO metadata(key, value) VALUES (?1, ?2) \
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        rusqlite::params![key, value],
    ) {
        tracing::warn!(key, "warmup: failed to persist metadata: {e}");
    }
}

/// Populate (or rebuild) the HNSW index from all cached embeddings in SQLite.
/// v0.30.2 B4: builds at a `<base>_new.usearch` staging path and atomically
/// swaps once the new index is fully written. The previous index stays
/// readable for the entire rebuild duration; crash-recovery (handled by
/// `cleanup_hnsw_staging` at warmup entry) wipes orphan staging files.
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
            // v0.30.3 codex R17 P2: preserve the rebuild signal when
            // we couldn't acquire LOCK_EX. The recall-side LOCK_SH
            // reader (added R11 P2 fix) can block our LOCK_EX | NB,
            // and the caller (HTTP background warmup / doctor --fix)
            // treats a `false` return as "no-op" without re-marking
            // dirty. Without this mark, one concurrent read can
            // silently cancel a needed rebuild, leaving HNSW stale.
            crate::store::hnsw::HnswIndex::mark_dirty(&hnsw_path);
            return false;
        }
    }

    // v0.30.2 B4: stage to `<base>_new.usearch` + `<base>_new.usearch.meta`
    // so the existing production index keeps serving until the new one is
    // fully written. The previous (destructive) sequence was
    // `remove .usearch / remove .usearch.meta / open / save` which left the
    // search subsystem with no usable HNSW index for the duration of the
    // rebuild. Crash mid-write here used to nuke the old index forever; now
    // it just leaves orphan `_new.*` files that `cleanup_hnsw_staging` mops
    // up at the next warmup entry.
    let staging_index = hnsw_staging_index_path(&hnsw_path);
    let staging_meta = hnsw_staging_meta_path(&hnsw_path);
    // v0.30.3 codex R16 P2: HnswIndex::open accepts a base path and
    // internally uses `Path::with_extension("usearch")` to derive the
    // index/meta filenames. For dotted DB names (`memories.v1`)
    // `with_extension` would strip the `.v1` segment, aliasing prod.
    // Give it a path with a placeholder extension that with_extension
    // can strip without losing our `_new` discriminator.
    let staging_open_base = hnsw_path.with_file_name(format!(
        "{}_new.placeholder",
        hnsw_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("hnsw")
    ));
    let _ = std::fs::remove_file(&staging_index);
    let _ = std::fs::remove_file(&staging_meta);

    let mut index = match crate::store::hnsw::HnswIndex::open(&staging_open_base, dims) {
        Ok(idx) => idx,
        Err(e) => {
            tracing::warn!("hnsw: failed to open staging index: {e}");
            // Clean partial staging artifacts so the next pass starts clean.
            let _ = std::fs::remove_file(&staging_index);
            let _ = std::fs::remove_file(&staging_meta);
            #[cfg(unix)]
            {
                use std::os::unix::io::AsRawFd;
                let _ = unsafe { libc::flock(lock_file.as_raw_fd(), libc::LOCK_UN) };
            }
            return false;
        }
    };

    // Stream memories and their embeddings directly into the HNSW index.
    // F6 D-M3: peak heap is now one `WarmupRow` instead of a Vec of the
    // entire `memories` table.
    //
    // v1.2 audit F11/F14: EmbedCache is a bounded LRU (10k entries) — on
    // stores past the cap, and after `rein migrate --reindex` (which wipes
    // the cache and fills only vec_memories), a cache-only rebuild silently
    // truncated the index (or hit the 0-insert dead end in a permanent
    // .dirty loop) while being promoted as healthy. vec_memories is the
    // AUTHORITATIVE durable vector store — fall back to it per row; the
    // cache remains the fast path (no per-row text hashing when warm).
    let mut inserted = 0usize;
    let mut total_rows = 0usize;
    let mut skipped_no_vector = 0usize;
    let stream_result = store.for_each_for_warmup(|row| {
        total_rows += 1;
        let text = prepend_metadata(&row.topic, &row.summary, &row.content);
        let emb = match EmbedCache::get(store.conn(), &text, &model) {
            Ok(Some(emb)) => Some(emb),
            _ => crate::store::vec::get_embedding(store.conn(), &row.id)
                .ok()
                .flatten(),
        };
        match emb {
            Some(emb) if emb.len() == dims => {
                if index.insert(&row.id, &emb).is_ok() {
                    inserted += 1;
                }
            }
            _ => {
                skipped_no_vector += 1;
            }
        }
        Ok(())
    });
    if skipped_no_vector > 0 {
        // No silent caps: rows without any stored vector (never embedded)
        // are expected on partially-warmed stores, but the count must be
        // visible — a large number here means the vector channel is serving
        // a materially incomplete index.
        tracing::warn!(
            inserted,
            skipped_no_vector,
            total_rows,
            "hnsw rebuild: some memories have no stored embedding (cache miss + no vec_memories row)"
        );
    }
    if let Err(e) = stream_result {
        tracing::warn!("hnsw: failed to list memories: {e}");
        // v0.30.3 codex R11 P2: mirror the tantivy path — when streaming
        // fails partway, the staging is incomplete (or the DB changed
        // during the rebuild). The previous prod files might still be
        // serving stale results without a dirty marker — gated warmup
        // would never re-trigger. Mark dirty + remove staging so the
        // next request observes the recovery signal and retries.
        crate::store::hnsw::HnswIndex::mark_dirty(&hnsw_path);
        let _ = std::fs::remove_file(&staging_index);
        let _ = std::fs::remove_file(&staging_meta);
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            let _ = unsafe { libc::flock(lock_file.as_raw_fd(), libc::LOCK_UN) };
        }
        return false;
    }

    let memories_empty = total_rows == 0;
    let mut rebuild_ok = false;
    if inserted > 0 {
        match index.save() {
            Ok(()) => {
                // v0.30.2 B4: atomically swap staging files into production
                // names. We drop the index struct first so the underlying
                // usearch handle can't hold an open mapping on the staging
                // file while we rename it.
                drop(index);
                let prod_index = hnsw_path.with_extension("usearch");
                let prod_meta = hnsw_path.with_extension("usearch.meta");
                // Mark dirty BEFORE the swap so a crash between the two
                // renames (which would leave `.usearch` and `.usearch.meta`
                // mismatched) is recoverable on next startup. The marker is
                // cleared only after BOTH renames succeed.
                crate::store::hnsw::HnswIndex::mark_dirty(&hnsw_path);
                // v0.30.3 codex R15 P2: Windows fallback for the
                // rename-over-existing case. Unix rename overwrites
                // atomically; Windows fails if the target exists or is
                // held open. Best-effort remove first.
                #[cfg(windows)]
                {
                    let _ = std::fs::remove_file(&prod_index);
                    let _ = std::fs::remove_file(&prod_meta);
                }
                match std::fs::rename(&staging_index, &prod_index)
                    .and_then(|_| std::fs::rename(&staging_meta, &prod_meta))
                {
                    Ok(()) => {
                        tracing::info!("hnsw: indexed {inserted} vectors (atomic swap ok)");
                        rebuild_ok = true;
                    }
                    Err(e) => {
                        tracing::warn!("hnsw: staging swap failed: {e}");
                        // Best-effort cleanup of any partial state.
                        let _ = std::fs::remove_file(&staging_index);
                        let _ = std::fs::remove_file(&staging_meta);
                    }
                }
            }
            Err(e) => {
                tracing::warn!("hnsw: failed to save index: {e}");
                // v0.30.3 codex R14 P2-#2: explicitly mark dirty when
                // save() fails. The old `.usearch` pair still exists
                // (we haven't yet renamed staging into place), but it
                // may not include vectors that were inserted into the
                // staging index this run. Without a dirty marker,
                // gated warmup (`always_rebuild_side_indexes = false`)
                // would see "no dirty + clean prod files" and skip the
                // next rebuild, leaving new memories out of HNSW
                // indefinitely.
                crate::store::hnsw::HnswIndex::mark_dirty(&hnsw_path);
                let _ = std::fs::remove_file(&staging_index);
                let _ = std::fs::remove_file(&staging_meta);
            }
        }
    } else if memories_empty {
        // No memories — clean staging AND any stale production index.
        // v0.30.3 codex R2 P2: previously we only removed staging here,
        // which left a pre-deletion `.usearch` / `.usearch.meta` on disk
        // even though the DB is now empty. The dirty-marker clear below
        // would then mark a stale index "clean" and vector recall would
        // keep returning deleted IDs. Now we remove the production pair
        // too so the empty-DB state is correctly reflected on disk.
        let _ = std::fs::remove_file(&staging_index);
        let _ = std::fs::remove_file(&staging_meta);
        let prod_index = hnsw_path.with_extension("usearch");
        let prod_meta = hnsw_path.with_extension("usearch.meta");
        let _ = std::fs::remove_file(&prod_index);
        let _ = std::fs::remove_file(&prod_meta);
        rebuild_ok = true; // no memories at all, empty index is intentionally correct
    } else {
        // v0.30.3 codex R7 P2: when called from startup warmup (not from
        // recall's `take_dirty_for_rebuild` path), there may be NO
        // pre-existing dirty marker to "keep". A non-empty DB with 0
        // cached embeddings (e.g. fresh deploy with no cache warmed yet,
        // or transient cache read failures) returning false from this
        // branch would leave any stale production `.usearch` files in
        // place AND CLEAN — `vec_search_direct` would keep serving stale
        // results. Explicitly mark dirty so vec_search falls through to
        // sqlite-vec and the NEXT warmup retries.
        tracing::debug!(
            "hnsw: {total_rows} memories but 0 cached embeddings, marking dirty for retry"
        );
        crate::store::hnsw::HnswIndex::mark_dirty(&hnsw_path);
        let _ = std::fs::remove_file(&staging_index);
        let _ = std::fs::remove_file(&staging_meta);
    }
    // Clear the legacy `.dirty` marker on success (no-op when called from async path
    // since `.dirty` was already renamed to `.rebuilding` before this function was called).
    // v0.30.3 codex R12 P2: also clear `.rebuilding` on success. If a
    // stranded `.rebuilding` marker triggered this rebuild via
    // `is_dirty()` (which includes rebuilding markers), success here
    // must remove BOTH markers — otherwise `.rebuilding` persists,
    // `is_dirty()` keeps returning true, `take_dirty_for_rebuild` fails
    // (no `.dirty` to take), and recall permanently bypasses HNSW.
    if rebuild_ok {
        let _ = std::fs::remove_file(crate::store::hnsw::HnswIndex::dirty_marker_path(&hnsw_path));
        let _ = std::fs::remove_file(crate::store::hnsw::HnswIndex::rebuilding_marker_path(
            &hnsw_path,
        ));
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
    // v0.30.3 codex R22 P2: capture dirty marker mtime BEFORE scan so
    // `finish_tantivy_rebuild_markers` can distinguish "this rebuild's
    // own claim" from "a concurrent mutation set it later".
    let scan_dirty_mtime = std::fs::metadata(tantivy_dirty_path(db_path))
        .and_then(|m| m.modified())
        .ok();

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

    // v0.30.2 B2: build the new index at `<db>.tantivy.new` first so the
    // production `<db>.tantivy` directory keeps serving readers for the
    // entire rebuild. The previous (destructive) sequence
    // (`remove_dir_all(.tantivy)` BEFORE `open(.tantivy)`) left the index
    // window-empty for the rebuild's full duration; a crash inside that
    // window destroyed the index permanently. Crash mid-build now leaves
    // only orphan `.tantivy.new` (and possibly `.tantivy.old`) which
    // `cleanup_tantivy_staging` clears at the next warmup entry.
    let staging_path = tantivy_staging_path(db_path);
    let backup_path = tantivy_backup_path(db_path);
    if let Err(e) = std::fs::remove_dir_all(&staging_path) {
        if e.kind() != std::io::ErrorKind::NotFound {
            mark_tantivy_dirty(db_path);
            let _ = std::fs::remove_file(&rebuilding_path);
            unlock_tantivy_rebuild_lock(&lock_file);
            return TantivyRebuildOutcome::Failed {
                reason: format!(
                    "failed to clear stale staging dir {}: {e}",
                    staging_path.display()
                ),
            };
        }
    }

    let tantivy = match crate::store::tantivy_fts::TantivyFts::open(&staging_path) {
        Ok(t) => t,
        Err(e) => {
            tracing::warn!("tantivy: failed to open staging index: {e}");
            mark_tantivy_dirty(db_path);
            let _ = std::fs::remove_dir_all(&staging_path);
            let _ = std::fs::remove_file(&rebuilding_path);
            unlock_tantivy_rebuild_lock(&lock_file);
            return TantivyRebuildOutcome::Failed {
                reason: format!(
                    "failed to open staging index {}: {e}",
                    staging_path.display()
                ),
            };
        }
    };

    // Stream memories into the Tantivy index. F6 D-M3: peak heap stays
    // O(row_size) — the entire `memories` table is never materialized.
    let mut indexed = 0usize;
    let mut errors = 0usize;
    let mut total_rows = 0usize;
    let stream_result = store.for_each_for_warmup(|row| {
        total_rows += 1;
        if tantivy
            .insert_strict(
                &row.id,
                &row.topic,
                &row.summary,
                &row.content,
                &row.keywords,
            )
            .is_ok()
        {
            indexed += 1;
        } else {
            errors += 1;
        }
        Ok(())
    });
    if let Err(e) = stream_result {
        tracing::warn!("tantivy: failed to list memories: {e}");
        mark_tantivy_dirty(db_path);
        drop(tantivy);
        let _ = std::fs::remove_dir_all(&staging_path);
        let _ = std::fs::remove_file(&rebuilding_path);
        unlock_tantivy_rebuild_lock(&lock_file);
        return TantivyRebuildOutcome::Failed {
            reason: format!("failed to list memories: {e}"),
        };
    }

    if indexed > 0 {
        tracing::info!("tantivy: indexed {indexed} documents ({errors} errors)");
    }

    // v0.30.3 codex R5 P2: any `insert_strict` failure means the staging
    // index is incomplete. Promoting it would replace a complete prior
    // index with a partial one — the dirty marker triggers a later retry,
    // but the last good Tantivy data is already gone. Abort the swap
    // here, leave the previous prod dir alone, set dirty so the next
    // rebuild attempt fires.
    if errors > 0 {
        tracing::warn!(
            "tantivy: rebuild had {errors} insert errors — not promoting partial index, keeping prior prod"
        );
        mark_tantivy_dirty(db_path);
        drop(tantivy);
        let _ = std::fs::remove_dir_all(&staging_path);
        let _ = std::fs::remove_file(&rebuilding_path);
        unlock_tantivy_rebuild_lock(&lock_file);
        return TantivyRebuildOutcome::Failed {
            reason: format!("{errors} memories failed to index during rebuild"),
        };
    }

    // v0.30.2 B2: drop the writer before renaming so any held tantivy
    // resources release first (matters on platforms where rename of an
    // in-use directory fails, and is harmless on Unix).
    drop(tantivy);

    // Atomic-ish swap: move the current production index aside, slide the
    // staging dir into place, then delete the backup. Crash mid-sequence
    // leaves a `<db>.tantivy.old` orphan that the next warmup entry
    // (`cleanup_tantivy_staging`) removes. We mark the index dirty if any
    // step fails so the next request triggers a fresh rebuild.
    //
    // v0.30.3 codex R9 P2: a previous interrupted swap may have left
    // `.tantivy.old` as the only valid index. The recall-spawn path
    // enters `try_populate_tantivy` directly (not via `warmup()` which
    // would have run `cleanup_tantivy_staging` first), so we MUST
    // restore any pre-existing backup-with-real-data before starting a
    // new swap dance — otherwise the unconditional pre-swap
    // `remove_dir_all(&backup_path)` would discard the last good index
    // before the new staging is promoted, and a failure here would lose
    // it permanently. If backup has real segments and prod is unusable,
    // restore it to prod first; then proceed.
    let prod_currently_has_segments = tantivy_has_segments(&tantivy_path);
    if backup_path.exists() && tantivy_has_segments(&backup_path) && !prod_currently_has_segments {
        if tantivy_path.exists() {
            let _ = std::fs::remove_dir_all(&tantivy_path);
        }
        if let Err(e) = std::fs::rename(&backup_path, &tantivy_path) {
            tracing::warn!(
                "tantivy: failed to pre-restore backup {} -> {}: {e}",
                backup_path.display(),
                tantivy_path.display()
            );
        } else {
            tracing::info!(
                "tantivy: pre-restored backup-with-data {} -> {} before new rebuild",
                backup_path.display(),
                tantivy_path.display()
            );
        }
    }
    // v0.30.3 codex R11 P2 + R12 P2: remove any pre-existing backup so
    // the swap's `rename(prod → backup)` doesn't fail with EEXIST.
    // BUT only delete when it's safe — if backup still has segments AND
    // current prod has NO segments, the pre-restore must have failed
    // (couldn't `remove_dir_all` an unusable prod, or rename failed).
    // In that case backup is still the only valid index. Don't delete;
    // skip this swap attempt and mark dirty so the next pass via
    // `cleanup_tantivy_staging` can retry the restore.
    let cur_backup_has_segments = tantivy_has_segments(&backup_path);
    let cur_prod_has_segments = tantivy_has_segments(&tantivy_path);
    if backup_path.exists() {
        if cur_backup_has_segments && !cur_prod_has_segments {
            // Pre-restore failed earlier; backup is the last good copy.
            // Abort this rebuild rather than overwrite the only valid
            // index — `cleanup_tantivy_staging` will retry the restore.
            tracing::warn!(
                "tantivy: pre-restore failed and backup is the only valid copy — aborting swap, marking dirty so cleanup retries"
            );
            mark_tantivy_dirty(db_path);
            // `tantivy` writer was already dropped a few lines above via
            // the unconditional `drop(tantivy)` between the index-log
            // and this swap block.
            let _ = std::fs::remove_dir_all(&staging_path);
            let _ = std::fs::remove_file(&rebuilding_path);
            unlock_tantivy_rebuild_lock(&lock_file);
            return TantivyRebuildOutcome::Failed {
                reason: "backup-with-segments + empty prod; cleanup must restore first".to_string(),
            };
        }
        let _ = std::fs::remove_dir_all(&backup_path);
    }
    let prod_exists = tantivy_path.exists();
    let mut swap_ok = true;

    if prod_exists {
        if let Err(e) = std::fs::rename(&tantivy_path, &backup_path) {
            tracing::warn!(
                "tantivy: failed to stash old index {} -> {}: {e}",
                tantivy_path.display(),
                backup_path.display()
            );
            swap_ok = false;
        }
    }
    if swap_ok {
        if let Err(e) = std::fs::rename(&staging_path, &tantivy_path) {
            tracing::warn!(
                "tantivy: failed to promote staging {} -> {}: {e}",
                staging_path.display(),
                tantivy_path.display()
            );
            // Try to restore the previous index from backup so the search
            // path stays usable.
            if prod_exists {
                let _ = std::fs::rename(&backup_path, &tantivy_path);
            }
            swap_ok = false;
        }
    }
    // v0.30.3 codex R3 P2: only delete the backup when the swap fully
    // succeeded. On failure paths the backup may be the only remaining
    // valid index (e.g. promote-failed AND restore-failed). Leave it on
    // disk so `cleanup_tantivy_staging` can restore it via the
    // backup-restore branch on the next warmup entry.
    if swap_ok {
        let _ = std::fs::remove_dir_all(&backup_path);
    }
    let _ = std::fs::remove_dir_all(&staging_path);

    if !swap_ok {
        mark_tantivy_dirty(db_path);
        let _ = std::fs::remove_file(&rebuilding_path);
        unlock_tantivy_rebuild_lock(&lock_file);
        return TantivyRebuildOutcome::Failed {
            reason: format!(
                "failed to swap staging into production at {}",
                tantivy_path.display()
            ),
        };
    }

    let memories_empty = total_rows == 0;
    finish_tantivy_rebuild_markers(db_path, indexed, errors, memories_empty, scan_dirty_mtime);

    // Lock released when lock_file is dropped.
    let _ = std::fs::remove_file(&rebuilding_path);
    unlock_tantivy_rebuild_lock(&lock_file);
    drop(lock_file);
    TantivyRebuildOutcome::Rebuilt { indexed, errors }
}

/// v0.30.3 codex R20 P2: the dirty marker MUST live OUTSIDE the
/// `<db>.tantivy/` directory — that directory is renamed to `.old` and
/// deleted on a successful staging swap, so a marker nested inside it
/// either disappears with the backup (signal lost) or causes the
/// `rename(staging → prod)` to fail with EEXIST (promotion lost). The
/// sibling-file location survives the swap untouched.
pub fn tantivy_dirty_path(db_path: &Path) -> PathBuf {
    // For `memories.db` → `memories.tantivy.dirty` (sibling file).
    // For dotted `memories.v1.db` → `memories.v1.tantivy.dirty`.
    db_path.with_extension("tantivy.dirty")
}

/// Legacy path inside the directory; used at warmup entry to migrate
/// pre-v0.30.3 markers to the new sibling location.
pub fn tantivy_dirty_path_legacy(db_path: &Path) -> PathBuf {
    db_path.with_extension("tantivy").join(".dirty")
}

pub fn tantivy_rebuild_lock_path(db_path: &Path) -> PathBuf {
    db_path.with_extension("tantivy.rebuild.lock")
}

pub fn tantivy_rebuilding_path(db_path: &Path) -> PathBuf {
    db_path.with_extension("tantivy.rebuilding")
}

/// v0.30.3 codex R7/R8 P2: "this directory contains real indexed data".
/// Used by the cleanup-staging restore path and the gated-warmup rebuild
/// predicate to distinguish a freshly-opened-but-empty index (only
/// `meta.json` + `.tokenizer_v2`, no segments) from an actually populated
/// index. Tantivy segment files use the `.idx` extension; presence of any
/// `.idx` file is the reliable signal.
fn tantivy_has_segments(tantivy_path: &Path) -> bool {
    tantivy_path
        .read_dir()
        .map(|it| {
            it.filter_map(|e| e.ok())
                .any(|e| e.file_name().to_string_lossy().ends_with(".idx"))
        })
        .unwrap_or(false)
}

/// v0.30.2 B2: staging directory used while a Tantivy rebuild is in flight.
/// Production `.tantivy` keeps serving readers until the swap completes.
pub fn tantivy_staging_path(db_path: &Path) -> PathBuf {
    db_path.with_extension("tantivy.new")
}

/// v0.30.2 B2: transient backup of the previous Tantivy dir during the swap
/// window. A crash here leaves an orphan that `cleanup_tantivy_staging`
/// removes at the next warmup entry.
pub fn tantivy_backup_path(db_path: &Path) -> PathBuf {
    db_path.with_extension("tantivy.old")
}

/// v0.30.2 B4: staging base for the HNSW rebuild. Returns
/// `<hnsw_path>_new` as a PathBuf (no extension). Callers must NOT use
/// `Path::with_extension` on this base — see [`hnsw_staging_index_path`]
/// and [`hnsw_staging_meta_path`] for the safe extension-appending
/// helpers. Kept for back-compat with tests that observe the base.
pub fn hnsw_staging_base(hnsw_path: &Path) -> PathBuf {
    let file_name = hnsw_path
        .file_name()
        .and_then(|s| s.to_str())
        .map(|s| format!("{s}_new"))
        .unwrap_or_else(|| "hnsw_new".to_string());
    hnsw_path.with_file_name(file_name)
}

/// v0.30.3 codex R16 P2: Path::with_extension strips the last
/// dot-section, so `staging_base.with_extension("usearch")` for a
/// dotted DB name like `memories.v1.db` (yielding hnsw_path
/// `memories.v1`, staging_base `memories.v1_new`) WIPES `.v1_new` and
/// produces `memories.usearch` — the same path as production. That
/// makes cleanup_hnsw_staging delete the live index. Use direct
/// string append instead.
pub fn hnsw_staging_index_path(hnsw_path: &Path) -> PathBuf {
    let parent = hnsw_path.parent().unwrap_or_else(|| Path::new(""));
    let file = hnsw_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("hnsw");
    parent.join(format!("{file}_new.usearch"))
}

/// Same defense for the meta sibling.
pub fn hnsw_staging_meta_path(hnsw_path: &Path) -> PathBuf {
    let parent = hnsw_path.parent().unwrap_or_else(|| Path::new(""));
    let file = hnsw_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("hnsw");
    parent.join(format!("{file}_new.usearch.meta"))
}

/// v0.30.2 B2: crash-recovery cleanup of orphan Tantivy staging dirs.
/// Called at warmup entry. Removes both `.tantivy.new` and `.tantivy.old`
/// best-effort; failures are logged but never fatal.
///
/// When the production `.tantivy` dir is missing AND we observe an orphan
/// `.tantivy.old`, that's evidence of a process kill mid-swap; mark the
/// index dirty so the next request triggers a fresh rebuild even when
/// `[warmup].always_rebuild_side_indexes = false`.
pub fn cleanup_tantivy_staging(db_path: &Path) {
    // v0.30.3 codex R22 P2: migrate the legacy `.tantivy/.dirty` marker
    // to the new sibling `.tantivy.dirty` location. The warmup() entry
    // also does this, but `warmup()` may not run on stdio surfaces
    // (where `background_warmup = false` by default) — so legacy
    // markers would survive forever for stdio users without this. The
    // recall-spawn rebuild path and `doctor --fix` both go through
    // this cleanup so all paths now migrate.
    let legacy = tantivy_dirty_path_legacy(db_path);
    if legacy.exists() {
        let canonical = tantivy_dirty_path(db_path);
        if !canonical.exists() {
            let _ = std::fs::rename(&legacy, &canonical);
        } else {
            let _ = std::fs::remove_file(&legacy);
        }
    }
    // v0.30.3 codex R2 P2: HOLD the rebuild lock through the entire cleanup
    // (not just probe-and-release) so another process can't acquire the
    // lock AFTER our probe and start writing `.tantivy.new` that we then
    // delete out from under them. Probe-only (R1) had a TOCTOU window;
    // hold-through-cleanup closes it.
    let lock_path = tantivy_rebuild_lock_path(db_path);
    let lock_file = match std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(&lock_path)
    {
        Ok(f) => f,
        Err(e) => {
            tracing::debug!(
                "warmup: failed to open tantivy rebuild lock {} for cleanup: {e}",
                lock_path.display()
            );
            return;
        }
    };
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        let fd = lock_file.as_raw_fd();
        let rc = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
        if rc != 0 {
            tracing::debug!(
                "warmup: skipping tantivy staging cleanup — rebuild lock held by another process"
            );
            return;
        }
    }
    // Lock is held for the rest of this function; drop(lock_file) at scope
    // exit releases it (kernel auto-releases flock on close).

    let prod = db_path.with_extension("tantivy");
    let staging = tantivy_staging_path(db_path);
    let backup = tantivy_backup_path(db_path);

    // v0.30.3 codex R1 P2 + R5 P2 + R6 P2 + R8 P2: production may be
    // missing, empty, marker-only, OR metadata-only. The last case
    // happens when `TantivyFts::open` creates an empty index after the
    // swap was interrupted — the dir then contains `meta.json` +
    // `.tokenizer_v2` but no actual indexed data. My earlier check
    // counted ANY non-hidden file as "real data" (R6 fix), but
    // `meta.json` is non-hidden and present in empty indexes too. The
    // only reliable signal for "this dir contains indexed data" is the
    // presence of `.idx` segment files. Apply that as the unified
    // check throughout the staging/restore path.
    let prod_unusable = !prod.exists() || !tantivy_has_segments(&prod);
    let restored_from_backup = if prod_unusable && backup.exists() {
        // Remove the empty prod dir (if any) so rename can succeed.
        if prod.exists() {
            let _ = std::fs::remove_dir_all(&prod);
        }
        match std::fs::rename(&backup, &prod) {
            Ok(()) => {
                tracing::warn!(
                    "warmup: tantivy swap was interrupted; restored backup {} -> {}",
                    backup.display(),
                    prod.display()
                );
                true
            }
            Err(e) => {
                tracing::warn!(
                    "warmup: failed to restore tantivy backup {} -> {}: {e}",
                    backup.display(),
                    prod.display()
                );
                false
            }
        }
    } else {
        false
    };

    // v0.30.3 codex R5 P2: `prod_unusable` captures the pre-restore
    // observation. `!prod.exists()` alone would miss the
    // "empty-dir-created-by-TantivyFts::open" case the restore path
    // above already handles.
    let swap_interrupted =
        restored_from_backup || (prod_unusable && (staging.exists() || backup.exists()));

    // Always remove staging (it's an orphan from a failed/interrupted
    // rebuild and never the canonical state). Only remove backup if it
    // STILL exists after the restore attempt above — that means either
    // restore failed, OR the rebuild had completed cleanly previously and
    // the backup is stale post-success.
    if staging.exists() {
        if let Err(e) = std::fs::remove_dir_all(&staging) {
            tracing::warn!(
                "warmup: failed to clean orphan tantivy staging {}: {e}",
                staging.display()
            );
        } else {
            tracing::info!(
                "warmup: removed orphan tantivy staging dir {}",
                staging.display()
            );
        }
    }
    // v0.30.3 codex R6 P2 + R8 P2: only delete the backup if prod now has
    // REAL indexed data (segment files), not just metadata or markers.
    // `prod.exists()` alone, or "any non-hidden file present", would
    // mis-classify a freshly-opened empty index (which has `meta.json`
    // + `.tokenizer_v2` but no segments) as a valid prod and discard
    // the still-valid backup.
    if backup.exists() && tantivy_has_segments(&prod) {
        if let Err(e) = std::fs::remove_dir_all(&backup) {
            tracing::warn!(
                "warmup: failed to clean stale tantivy backup {}: {e}",
                backup.display()
            );
        }
    }

    if swap_interrupted {
        tracing::warn!(
            "warmup: tantivy swap was interrupted — marking dirty so next request rebuilds"
        );
        mark_tantivy_dirty(db_path);
    }
}

/// v0.30.2 B4: crash-recovery cleanup of orphan HNSW staging files.
/// Mirrors `cleanup_tantivy_staging` for the `_new.usearch` /
/// `_new.usearch.meta` pair.
pub fn cleanup_hnsw_staging(db_path: &Path) {
    let hnsw_path = db_path.with_extension("");

    // v0.30.3 codex R2 P2: HOLD the HNSW rebuild lock through the entire
    // cleanup (not just probe-and-release) so another process can't
    // acquire the lock AFTER our probe and start writing
    // `<base>_new.usearch` that we then delete out from under them.
    // populate_hnsw uses the same `<base>.usearch.lock` path with
    // `LOCK_EX | LOCK_NB`. Always create-and-acquire (avoid existence
    // race: if we checked-then-opened, another process could create+lock
    // between the two ops).
    let lock_path = hnsw_path.with_extension("usearch.lock");
    let lock_file = match std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(&lock_path)
    {
        Ok(f) => f,
        Err(e) => {
            tracing::debug!(
                "warmup: failed to open hnsw rebuild lock {} for cleanup: {e}",
                lock_path.display()
            );
            return;
        }
    };
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        let fd = lock_file.as_raw_fd();
        let rc = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
        if rc != 0 {
            tracing::debug!(
                "warmup: skipping hnsw staging cleanup — rebuild lock held by another process"
            );
            return;
        }
    }
    // Lock held until function returns (kernel releases on close).

    // v0.30.3 codex R16 P2: use the direct-concat helpers so dotted DB
    // names (e.g. `memories.v1.db`) don't make staging alias prod.
    for p in [
        hnsw_staging_index_path(&hnsw_path),
        hnsw_staging_meta_path(&hnsw_path),
    ] {
        if p.exists() {
            if let Err(e) = std::fs::remove_file(&p) {
                tracing::warn!(
                    "warmup: failed to clean orphan hnsw staging {}: {e}",
                    p.display()
                );
            } else {
                tracing::info!("warmup: removed orphan hnsw staging file {}", p.display());
            }
        }
    }
    drop(lock_file);
}

/// v0.30.2 B3: gate the cold-start tantivy rebuild on either
/// `[warmup].always_rebuild_side_indexes` (default true preserves prior
/// behavior) OR a concrete recovery signal (missing index dir, dirty
/// marker, or stale rebuild marker).
fn side_index_rebuild_needed_tantivy(
    store: &SqliteStore,
    db_path: &Path,
    config: &ReinConfig,
) -> bool {
    if config.warmup.always_rebuild_side_indexes {
        return true;
    }
    let tantivy_path = db_path.with_extension("tantivy");
    // v0.30.3 codex R5 P2: pre-tokenizer-v2 indexes have the dir but no
    // `.tokenizer_v2` marker file. The next `TantivyFts::open` would
    // wipe-and-recreate for migration WITHOUT setting `.dirty`, leaving
    // gated warmups treating the empty migrated index as clean forever.
    let tokenizer_marker = tantivy_path.join(".tokenizer_v2");
    // v0.30.3 codex R7 P2: `TantivyFts::open` on a read/update path can
    // create a freshly-marked index that has the marker but no segment
    // data. Gate that to "and SQLite actually has memories" — an empty
    // index on an empty DB is not a defect and rebuilding it every cold
    // start would be wasteful. Only fire the rebuild signal when there's
    // an observable mismatch (memories exist but index is empty).
    let store_has_memories = store.stats().map(|s| s.total_memories > 0).unwrap_or(false);
    let empty_but_marked_with_data = tantivy_path.exists()
        && tokenizer_marker.exists()
        && !tantivy_has_segments(&tantivy_path)
        && store_has_memories;
    !tantivy_path.exists()
        || !tokenizer_marker.exists()
        || empty_but_marked_with_data
        || tantivy_dirty_path(db_path).exists()
        || matches!(
            tantivy_rebuild_state(db_path),
            TantivyRebuildState::StaleMarker
        )
}

/// v0.30.2 B6: gate the cold-start HNSW rebuild on either
/// `[warmup].always_rebuild_side_indexes` (default true preserves prior
/// behavior) OR a concrete recovery signal (missing files, dirty marker).
fn side_index_rebuild_needed_hnsw(db_path: &Path, config: &ReinConfig) -> bool {
    if config.warmup.always_rebuild_side_indexes {
        return true;
    }
    let hnsw_path = db_path.with_extension("");
    let index_file = hnsw_path.with_extension("usearch");
    let meta_file = hnsw_path.with_extension("usearch.meta");
    !index_file.exists()
        || !meta_file.exists()
        || crate::store::hnsw::HnswIndex::is_dirty(&hnsw_path)
}

/// v0.30.2 B5 TTL helper used by `rein doctor --fix`: an HNSW
/// `.rebuilding` marker older than `ttl` is treated as a stranded
/// rebuild (worker panicked without restoring `.dirty`). Returns the
/// marker path that was reset to `.dirty`, or `None` if nothing needed
/// to be done.
pub fn reset_stale_hnsw_rebuilding(db_path: &Path, ttl: std::time::Duration) -> Option<PathBuf> {
    let hnsw_path = db_path.with_extension("");
    let rebuilding = crate::store::hnsw::HnswIndex::rebuilding_marker_path(&hnsw_path);
    let metadata = std::fs::metadata(&rebuilding).ok()?;
    let modified = metadata.modified().ok()?;
    let age = modified.elapsed().unwrap_or_default();
    if age >= ttl {
        // v0.30.3 codex R6 P2: probe the HNSW rebuild lock before assuming
        // the marker is stranded. `take_dirty_for_rebuild` renames
        // `.dirty → .rebuilding` WITHOUT refreshing mtime, so an active
        // rebuild that started from an old `.dirty` marker will have an
        // old `.rebuilding` mtime even though it's live. Resetting the
        // marker out from under the worker leaves HNSW dirty after the
        // worker finishes (it clears `.rebuilding`, which is now a
        // no-op). If the lock is held: don't touch the marker.
        let lock_path = hnsw_path.with_extension("usearch.lock");
        if lock_path.exists() {
            if let Ok(lock_file) = std::fs::OpenOptions::new()
                .create(true)
                .read(true)
                .write(true)
                .truncate(false)
                .open(&lock_path)
            {
                #[cfg(unix)]
                {
                    use std::os::unix::io::AsRawFd;
                    let fd = lock_file.as_raw_fd();
                    let rc = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
                    if rc != 0 {
                        tracing::debug!(
                            "warmup: skipping stale-hnsw-rebuilding reset — rebuild lock held by another process"
                        );
                        return None;
                    }
                    // Release immediately so the real rebuild path can
                    // race behind us if it wants the lock.
                    let _ = unsafe { libc::flock(fd, libc::LOCK_UN) };
                }
                drop(lock_file);
            }
        }
        let dirty = crate::store::hnsw::HnswIndex::dirty_marker_path(&hnsw_path);
        let _ = std::fs::rename(&rebuilding, &dirty);
        return Some(rebuilding);
    }
    None
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

/// v0.30.4 D4 (deferred from v0.30.3 audit cycle): symmetric helper to
/// `reset_stale_hnsw_rebuilding` for the Tantivy side.
///
/// Background: HNSW already has a TTL-based stale-marker recovery
/// helper (see `reset_stale_hnsw_rebuilding`) wired into
/// `rein doctor --fix`.  Tantivy did not, so an interrupted rebuild
/// that left a stale `.rebuilding` marker without `.dirty` could
/// strand the index in `StaleMarker` state — the recall path's
/// R23-era partial fix re-triggers spawn on `StaleMarker` but the
/// spawn lock-acquire can keep failing if a zombie process is
/// holding the flock.  This helper closes the gap by mirroring the
/// HNSW approach: if the `.rebuilding` marker is older than `ttl`
/// AND the rebuild lock is NOT held by an active worker, rename
/// the marker to `.dirty` so the next recall re-triggers a fresh
/// rebuild from a clean slate.
///
/// Returns the path of the marker that was renamed (for caller
/// logging), or `None` if no action was taken.
pub fn reset_stale_tantivy_rebuilding(db_path: &Path, ttl: std::time::Duration) -> Option<PathBuf> {
    let rebuilding = tantivy_rebuilding_path(db_path);
    let metadata = std::fs::metadata(&rebuilding).ok()?;
    let modified = metadata.modified().ok()?;
    let age = modified.elapsed().unwrap_or_default();
    if age < ttl {
        return None;
    }
    // Probe the rebuild lock — don't touch the marker if a live
    // worker holds it.  An active rebuild that started from an old
    // `.dirty` (and thus has an old `.rebuilding` mtime) is still
    // legitimate; resetting the marker out from under it would leave
    // tantivy dirty after the worker finishes (it would clear
    // `.rebuilding`, which would then be a no-op).
    let lock_path = tantivy_rebuild_lock_path(db_path);
    if lock_path.exists() && tantivy_rebuild_lock_is_held(&lock_path) {
        tracing::debug!(
            "warmup: skipping stale-tantivy reset — rebuild lock held by another worker"
        );
        return None;
    }
    let dirty = tantivy_dirty_path(db_path);
    if let Some(parent) = dirty.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    // Use `rename` (atomic on Unix) so the next recall sees either the
    // old `.rebuilding` or the new `.dirty`, never neither.
    if std::fs::rename(&rebuilding, &dirty).is_err() {
        return None;
    }
    Some(rebuilding)
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
    scan_dirty_mtime: Option<std::time::SystemTime>,
) {
    // Only clear dirty marker if rebuild succeeded with actual data, or there
    // are truly no memories. Any partial error must keep the marker so a later
    // repair can pick up missing documents.
    if errors == 0 && (indexed > 0 || memories_empty) {
        // v0.30.3 codex R22 P2: only remove the dirty marker if it has
        // not been touched AFTER our rebuild scan started. A concurrent
        // `update_tantivy`/`remove_from_tantivy` running mid-rebuild
        // sets the dirty marker (R19 fix) to signal "my mutation is
        // not in your staging snapshot". Unconditional removal here
        // would wipe that signal and leave the next rebuild blind to
        // the mutation. Compare mtimes — if the on-disk marker is
        // newer than what we observed at scan start, leave it.
        let dirty = tantivy_dirty_path(db_path);
        let safe_to_remove = match (
            scan_dirty_mtime,
            std::fs::metadata(&dirty).and_then(|m| m.modified()).ok(),
        ) {
            (Some(scan_ts), Some(cur_ts)) => cur_ts <= scan_ts,
            (None, Some(_)) => false, // marker appeared after scan started
            _ => true,                // no current marker, nothing to preserve
        };
        if safe_to_remove {
            let _ = std::fs::remove_file(&dirty);
        } else {
            tracing::debug!(
                "tantivy: keeping dirty marker — concurrent mutation observed after scan started"
            );
        }
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

        finish_tantivy_rebuild_markers(store.db_path(), 1, 1, false, None);

        assert!(
            dirty.exists(),
            "partial rebuild errors must keep Tantivy dirty for repair"
        );
    }

    // -------- v0.30.2 B2 — staging swap regression --------

    /// B2: a successful rebuild produces a `.tantivy` directory and leaves
    /// no `.tantivy.new` / `.tantivy.old` orphans behind.
    #[test]
    fn try_populate_tantivy_b2_leaves_no_staging_orphans_on_success() {
        let (_dir, store) = test_store();
        let db = store.db_path();

        let outcome = try_populate_tantivy(&store);
        assert!(matches!(outcome, TantivyRebuildOutcome::Rebuilt { .. }));

        let prod = db.with_extension("tantivy");
        let staging = tantivy_staging_path(db);
        let backup = tantivy_backup_path(db);
        assert!(
            prod.exists(),
            "production tantivy dir must exist after success"
        );
        assert!(
            !staging.exists(),
            "staging dir must not survive a successful swap"
        );
        assert!(
            !backup.exists(),
            "backup dir must not survive a successful swap"
        );
    }

    /// B2: when the previous index exists, the rebuild keeps it accessible
    /// during the build (we cannot deterministically check the in-flight
    /// state from a single-threaded test, but we can assert the production
    /// path was not unlinked at any observable point — it must exist at
    /// the start of the rebuild and at the end).
    #[test]
    fn try_populate_tantivy_b2_preserves_prior_index_directory() {
        let (_dir, store) = test_store();
        let db = store.db_path();

        // First rebuild creates the production dir.
        let _ = try_populate_tantivy(&store);
        let prod = db.with_extension("tantivy");
        assert!(prod.exists(), "first rebuild should produce a tantivy dir");

        // Second rebuild must end with the production dir still present
        // (staging swap is atomic-ish — the previous index is renamed to
        // `.tantivy.old` only briefly, then deleted after promotion).
        let _ = try_populate_tantivy(&store);
        assert!(
            prod.exists(),
            "second rebuild must leave production dir in place"
        );
        assert!(!tantivy_staging_path(db).exists());
        assert!(!tantivy_backup_path(db).exists());
    }

    /// B2 simulated mid-build interruption: create the production index,
    /// then create an orphan `.tantivy.new` staging dir (as if a prior
    /// rebuild crashed before swap). Assert the production index is STILL
    /// loadable by `TantivyFts::open` (the previous-index invariant from
    /// the brief), and that `cleanup_tantivy_staging` clears the orphan
    /// without disturbing production.
    #[test]
    fn b2_previous_index_still_loads_after_simulated_interruption() {
        let (_dir, store) = test_store();
        let db = store.db_path();

        // Build a valid production tantivy.
        let _ = try_populate_tantivy(&store);
        let prod = db.with_extension("tantivy");
        assert!(prod.exists());
        // Confirm baseline: open returns Ok.
        crate::store::tantivy_fts::TantivyFts::open(&prod)
            .expect("baseline tantivy must be openable");

        // Stage an orphan staging dir (simulates "we crashed inside the
        // staging build, before the swap step").
        let staging = tantivy_staging_path(db);
        std::fs::create_dir_all(&staging).unwrap();
        std::fs::write(staging.join("partial.txt"), b"half-written").unwrap();

        // Cleanup (runs at next warmup entry) removes the orphan without
        // touching production.
        cleanup_tantivy_staging(db);
        assert!(!staging.exists(), "orphan staging removed");
        assert!(prod.exists(), "production dir still present");

        // PREVIOUS-INDEX INVARIANT: the production tantivy is still
        // loadable. A regression that did `remove_dir_all(&prod)` BEFORE
        // the staging build (the pre-B2 sequence) would have left this
        // dir empty and `TantivyFts::open` could still create a fresh
        // (empty) one — but our orphan-cleanup path explicitly preserves
        // it.
        crate::store::tantivy_fts::TantivyFts::open(&prod)
            .expect("previous tantivy must still load after interruption recovery");
    }

    /// B4 simulated interruption: create the production HNSW pair, stamp
    /// an orphan `_new.usearch` (as if save failed mid-write), assert the
    /// production pair is unaffected and `HnswIndex::open` still works.
    #[test]
    fn b4_previous_hnsw_still_loads_after_simulated_interruption() {
        let (_dir, store) = test_store();
        let db = store.db_path();
        let hnsw_base = db.with_extension("");
        let mut cfg = ReinConfig::default();
        cfg.embedding.dimensions = 3072;

        // Build an empty production state (memories.is_empty branch
        // returns true). Synthesize the files because the empty-branch
        // doesn't write them.
        populate_hnsw(&store, &cfg);
        let prod_index = hnsw_base.with_extension("usearch");
        let prod_meta = hnsw_base.with_extension("usearch.meta");
        if !prod_index.exists() {
            std::fs::write(&prod_index, b"prod-index").unwrap();
        }
        if !prod_meta.exists() {
            std::fs::write(&prod_meta, b"prod-meta").unwrap();
        }
        let pre_index_bytes = std::fs::read(&prod_index).unwrap();
        let pre_meta_bytes = std::fs::read(&prod_meta).unwrap();

        // Drop a half-written staging pair (simulates "save panicked mid
        // write").
        let staging_index = hnsw_staging_index_path(&hnsw_base);
        let staging_meta = hnsw_staging_meta_path(&hnsw_base);
        std::fs::write(&staging_index, b"partial").unwrap();
        std::fs::write(&staging_meta, b"partial").unwrap();

        cleanup_hnsw_staging(db);

        // Orphans gone, production untouched.
        assert!(!staging_index.exists());
        assert!(!staging_meta.exists());
        assert_eq!(std::fs::read(&prod_index).unwrap(), pre_index_bytes);
        assert_eq!(std::fs::read(&prod_meta).unwrap(), pre_meta_bytes);
    }

    /// B2 crash recovery: orphan `.tantivy.new` left over from a prior
    /// interrupted rebuild must be cleaned by `cleanup_tantivy_staging`.
    #[test]
    fn cleanup_tantivy_staging_removes_orphan_staging_and_backup() {
        let (_dir, store) = test_store();
        let db = store.db_path();
        let staging = tantivy_staging_path(db);
        let backup = tantivy_backup_path(db);
        std::fs::create_dir_all(&staging).unwrap();
        std::fs::write(staging.join("junk"), b"x").unwrap();
        std::fs::create_dir_all(&backup).unwrap();
        std::fs::write(backup.join("junk"), b"x").unwrap();

        cleanup_tantivy_staging(db);

        assert!(!staging.exists(), "orphan staging must be cleaned");
        assert!(!backup.exists(), "orphan backup must be cleaned");
    }

    // -------- v0.30.2 B4 — HNSW staging --------

    /// B4: orphan HNSW staging files (`*_new.usearch[.meta]`) are cleaned
    /// at warmup entry by `cleanup_hnsw_staging`.
    #[test]
    fn cleanup_hnsw_staging_removes_orphan_files() {
        let (_dir, store) = test_store();
        let db = store.db_path();
        let hnsw_path = db.with_extension("");
        let staging_index = hnsw_staging_index_path(&hnsw_path);
        let staging_meta = hnsw_staging_meta_path(&hnsw_path);
        std::fs::write(&staging_index, b"orphan").unwrap();
        std::fs::write(&staging_meta, b"orphan").unwrap();

        cleanup_hnsw_staging(db);

        assert!(
            !staging_index.exists(),
            "orphan _new.usearch must be cleaned"
        );
        assert!(
            !staging_meta.exists(),
            "orphan _new.usearch.meta must be cleaned"
        );
    }

    /// v0.30.3 codex R16 P2 regression: for a DB filename with a dot in
    /// the stem (e.g. `memories.v1.db`), the staging path computation
    /// MUST NOT alias the production HNSW path. Previously the
    /// `.with_extension("usearch")` step on `staging_base = memories.v1_new`
    /// would strip `.v1_new` and produce `memories.usearch` — the same
    /// path as the bugged production HNSW. The new helpers append the
    /// extension via string concat so the staging path retains its
    /// `_new.usearch` suffix.
    #[test]
    fn hnsw_staging_does_not_alias_prod_for_dotted_db_names() {
        let hnsw_path = std::path::PathBuf::from("/tmp/memories.v1");
        let staging_index = hnsw_staging_index_path(&hnsw_path);
        assert_eq!(
            staging_index,
            std::path::PathBuf::from("/tmp/memories.v1_new.usearch")
        );
        let staging_meta = hnsw_staging_meta_path(&hnsw_path);
        assert_eq!(
            staging_meta,
            std::path::PathBuf::from("/tmp/memories.v1_new.usearch.meta")
        );
        // Critically: neither aliases what the codebase computes as
        // production for `memories.v1` (which the latent
        // `with_extension` bug renders as `memories.usearch` — staging
        // must NOT collide with that path).
        let buggy_prod_path = hnsw_path.with_extension("usearch");
        assert_ne!(staging_index, buggy_prod_path);
    }

    /// B4 helper sanity: the staging base for HNSW produces names that do
    /// NOT collide with the production `.usearch` / `.usearch.meta` pair.
    #[test]
    fn hnsw_staging_base_does_not_collide_with_production() {
        let p = std::path::PathBuf::from("/tmp/memories");
        let staging = hnsw_staging_base(&p);
        assert_eq!(staging, std::path::PathBuf::from("/tmp/memories_new"));
        assert_eq!(
            staging.with_extension("usearch"),
            std::path::PathBuf::from("/tmp/memories_new.usearch")
        );
        assert_eq!(
            staging.with_extension("usearch.meta"),
            std::path::PathBuf::from("/tmp/memories_new.usearch.meta")
        );
    }

    // -------- v0.30.2 B3 / B6 — cold-start gating --------

    /// B3: with `always_rebuild_side_indexes = false` and a clean state
    /// (no dirty marker, no stale rebuilding marker, index dir present),
    /// the gating helper must return `false`.
    #[test]
    fn side_index_rebuild_needed_tantivy_false_when_clean_and_gate_off() {
        let (_dir, store) = test_store();
        let db = store.db_path();
        // First rebuild leaves a clean tantivy dir
        let _ = try_populate_tantivy(&store);

        let mut cfg = ReinConfig::default();
        cfg.warmup.always_rebuild_side_indexes = false;
        assert!(
            !side_index_rebuild_needed_tantivy(&store, db, &cfg),
            "with gate off and clean state, rebuild must be skipped"
        );
    }

    /// B3: with `always_rebuild_side_indexes = true` (default), the gate
    /// always returns `true` to preserve prior behavior.
    #[test]
    fn side_index_rebuild_needed_tantivy_true_when_gate_on() {
        let (_dir, store) = test_store();
        let db = store.db_path();
        let _ = try_populate_tantivy(&store);

        let cfg = ReinConfig::default();
        assert!(cfg.warmup.always_rebuild_side_indexes);
        assert!(side_index_rebuild_needed_tantivy(&store, db, &cfg));
    }

    /// B3: a dirty marker forces the rebuild regardless of the flag.
    #[test]
    fn side_index_rebuild_needed_tantivy_dirty_overrides_gate_off() {
        let (_dir, store) = test_store();
        let db = store.db_path();
        let _ = try_populate_tantivy(&store);
        let dirty = tantivy_dirty_path(db);
        std::fs::create_dir_all(dirty.parent().unwrap()).unwrap();
        std::fs::write(&dirty, b"dirty").unwrap();

        let mut cfg = ReinConfig::default();
        cfg.warmup.always_rebuild_side_indexes = false;
        assert!(side_index_rebuild_needed_tantivy(&store, db, &cfg));
    }

    /// B6: same gating, HNSW side.
    #[test]
    fn side_index_rebuild_needed_hnsw_true_when_files_missing_and_gate_off() {
        let (_dir, store) = test_store();
        let db = store.db_path();
        let mut cfg = ReinConfig::default();
        cfg.warmup.always_rebuild_side_indexes = false;
        // No `.usearch` files exist yet — must rebuild.
        assert!(side_index_rebuild_needed_hnsw(db, &cfg));
    }

    // -------- B3 / B6 end-to-end: gate prevents `populate_*` from running --------

    /// B3 end-to-end: with the flag flipped to `false` AND the Tantivy
    /// production dir already present and clean, the next `warmup()` call
    /// must NOT re-run `populate_tantivy`. We assert this by stamping a
    /// known-fingerprint file into the production dir and verifying it
    /// still exists (and the swap-staging orphans are absent) after
    /// `warmup()` returns. A regression that ignores the gate would
    /// `remove_dir_all`-clobber the fingerprint via the staging swap.
    #[test]
    fn b3_warmup_skips_populate_tantivy_when_gate_off_and_clean() {
        let (_dir, store) = test_store();
        let db = store.db_path();
        // First rebuild creates the production dir.
        let _ = try_populate_tantivy(&store);
        let prod = db.with_extension("tantivy");
        assert!(prod.exists());

        // Stamp a sentinel file that would survive only if no rebuild ran.
        // (`TantivyFts::open` doesn't write this name, and `populate_*`
        // staging swap would wipe it via `remove_dir_all` on the old dir.)
        let sentinel = prod.join(".b3-test-sentinel");
        std::fs::write(&sentinel, b"do not clobber").unwrap();
        let pre_mtime = std::fs::metadata(&sentinel).unwrap().modified().unwrap();

        let mut cfg = ReinConfig::default();
        cfg.warmup.always_rebuild_side_indexes = false;

        // Run only the side-index pre-amble — equivalent to the first
        // block of `warmup()` since `create_embedder` would early-return
        // anyway without an API key.
        cleanup_tantivy_staging(db);
        cleanup_hnsw_staging(db);
        if side_index_rebuild_needed_tantivy(&store, db, &cfg) {
            populate_tantivy(&store);
        }
        if side_index_rebuild_needed_hnsw(db, &cfg) {
            populate_hnsw(&store, &cfg);
        }

        // Sentinel must survive — proves populate_tantivy was NOT called.
        assert!(
            sentinel.exists(),
            "with gate off and clean state, populate_tantivy must NOT run (sentinel would be clobbered by swap)"
        );
        let post_mtime = std::fs::metadata(&sentinel).unwrap().modified().unwrap();
        assert_eq!(
            pre_mtime, post_mtime,
            "sentinel mtime must not change (no rebuild fired)"
        );
        assert!(!tantivy_staging_path(db).exists());
        assert!(!tantivy_backup_path(db).exists());
    }

    /// B6 end-to-end: same idea, HNSW side. Builds an empty HNSW index,
    /// stamps a sentinel mtime on the `.usearch` file, then with the gate
    /// off re-runs the gating + populate; sentinel mtime must not change.
    #[test]
    fn b6_warmup_skips_populate_hnsw_when_gate_off_and_clean() {
        let (_dir, store) = test_store();
        let db = store.db_path();
        let cfg_on = ReinConfig::default();
        // First populate_hnsw with no embeddings produces a valid empty
        // state (returns true from the `memories.is_empty()` branch); we
        // make sure the .usearch / .usearch.meta files are present.
        populate_hnsw(&store, &cfg_on);
        let hnsw_base = db.with_extension("");
        let prod_index = hnsw_base.with_extension("usearch");
        let prod_meta = hnsw_base.with_extension("usearch.meta");

        // Production files may not exist (empty store skips file save);
        // synthesize them so the gating helper sees a "clean" state.
        if !prod_index.exists() {
            std::fs::write(&prod_index, b"sentinel-index").unwrap();
        }
        if !prod_meta.exists() {
            std::fs::write(&prod_meta, b"sentinel-meta").unwrap();
        }
        let pre_index = std::fs::read(&prod_index).unwrap();
        let pre_meta = std::fs::read(&prod_meta).unwrap();

        let mut cfg = ReinConfig::default();
        cfg.warmup.always_rebuild_side_indexes = false;

        // Equivalent of warmup()'s pre-amble.
        cleanup_tantivy_staging(db);
        cleanup_hnsw_staging(db);
        if side_index_rebuild_needed_hnsw(db, &cfg) {
            populate_hnsw(&store, &cfg);
        }

        // Sentinels survive: gate prevented populate_hnsw from running.
        assert_eq!(std::fs::read(&prod_index).unwrap(), pre_index);
        assert_eq!(std::fs::read(&prod_meta).unwrap(), pre_meta);
        // No staging orphans left around.
        let staging_base = hnsw_staging_base(&hnsw_base);
        assert!(!staging_base.with_extension("usearch").exists());
        assert!(!staging_base.with_extension("usearch.meta").exists());
    }

    // -------- v0.30.2 B5 — stale `.rebuilding` TTL reset --------

    /// B5: an HNSW `.rebuilding` marker older than the TTL is renamed to
    /// `.dirty` so the next recall request retries.
    #[test]
    fn reset_stale_hnsw_rebuilding_renames_old_marker_to_dirty() {
        let (_dir, store) = test_store();
        let db = store.db_path();
        let hnsw_path = db.with_extension("");
        let rebuilding = crate::store::hnsw::HnswIndex::rebuilding_marker_path(&hnsw_path);
        let dirty = crate::store::hnsw::HnswIndex::dirty_marker_path(&hnsw_path);
        std::fs::write(&rebuilding, b"rebuilding").unwrap();

        // TTL of zero forces an immediate rename.
        let result = reset_stale_hnsw_rebuilding(db, std::time::Duration::from_secs(0));
        assert_eq!(result.as_deref(), Some(rebuilding.as_path()));
        assert!(!rebuilding.exists(), "marker must be renamed away");
        assert!(dirty.exists(), "marker must be promoted to .dirty");
    }

    /// B5: a fresh `.rebuilding` marker (well under the TTL) is left alone.
    #[test]
    fn reset_stale_hnsw_rebuilding_keeps_fresh_marker() {
        let (_dir, store) = test_store();
        let db = store.db_path();
        let hnsw_path = db.with_extension("");
        let rebuilding = crate::store::hnsw::HnswIndex::rebuilding_marker_path(&hnsw_path);
        std::fs::write(&rebuilding, b"rebuilding").unwrap();

        let result = reset_stale_hnsw_rebuilding(db, std::time::Duration::from_secs(3600));
        assert_eq!(result, None);
        assert!(rebuilding.exists());
    }

    // -------- v0.30.4 D4 — symmetric stale tantivy `.rebuilding` TTL reset --

    /// D4: a Tantivy `.rebuilding` marker older than the TTL is renamed to
    /// `.dirty` so the next recall re-triggers a fresh rebuild from a clean
    /// slate.  Mirror of `reset_stale_hnsw_rebuilding_renames_old_marker_to_dirty`.
    #[test]
    fn reset_stale_tantivy_rebuilding_renames_old_marker_to_dirty() {
        let (_dir, store) = test_store();
        let db = store.db_path();
        let rebuilding = tantivy_rebuilding_path(db);
        let dirty = tantivy_dirty_path(db);
        std::fs::write(&rebuilding, b"rebuilding").unwrap();

        // TTL of zero forces an immediate rename.
        let result = reset_stale_tantivy_rebuilding(db, std::time::Duration::from_secs(0));
        assert_eq!(result.as_deref(), Some(rebuilding.as_path()));
        assert!(!rebuilding.exists(), "marker must be renamed away");
        assert!(dirty.exists(), "marker must be promoted to .dirty");
    }

    /// D4: a fresh Tantivy `.rebuilding` marker (well under the TTL) is left
    /// alone.  Mirror of `reset_stale_hnsw_rebuilding_keeps_fresh_marker`.
    #[test]
    fn reset_stale_tantivy_rebuilding_keeps_fresh_marker() {
        let (_dir, store) = test_store();
        let db = store.db_path();
        let rebuilding = tantivy_rebuilding_path(db);
        std::fs::write(&rebuilding, b"rebuilding").unwrap();

        let result = reset_stale_tantivy_rebuilding(db, std::time::Duration::from_secs(3600));
        assert_eq!(result, None);
        assert!(rebuilding.exists());
    }

    /// D4: when an active rebuild worker is holding the flock on the
    /// `tantivy.rebuild.lock` file, the helper MUST NOT rename the marker
    /// out from under it — doing so would leave the index in `Idle` state
    /// after the live worker eventually clears its `.rebuilding` (which
    /// would then be a no-op), without the `.dirty` marker, so the next
    /// recall would NOT trigger a rebuild.
    #[cfg(unix)]
    #[test]
    fn reset_stale_tantivy_rebuilding_skips_when_lock_held() {
        let (_dir, store) = test_store();
        let db = store.db_path();
        let rebuilding = tantivy_rebuilding_path(db);
        let lock_path = tantivy_rebuild_lock_path(db);
        std::fs::write(&rebuilding, b"rebuilding").unwrap();

        // Hold the rebuild lock exclusively for the duration of the test.
        // Use the same flock pattern as the real rebuild path.
        let lock_file = std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&lock_path)
            .unwrap();
        use std::os::unix::io::AsRawFd;
        let fd = lock_file.as_raw_fd();
        let rc = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
        assert_eq!(rc, 0, "test setup: failed to acquire lock");

        // TTL=0 would normally force a rename, but the held lock must
        // suppress the action.
        let result = reset_stale_tantivy_rebuilding(db, std::time::Duration::from_secs(0));
        assert_eq!(
            result, None,
            "D4: must not reset the marker while a live rebuild holds the lock"
        );
        assert!(rebuilding.exists(), "marker must be preserved");

        // Release lock for cleanup.
        let _ = unsafe { libc::flock(fd, libc::LOCK_UN) };
        drop(lock_file);
    }

    fn live_memory(topic: &str, content: &str) -> crate::types::Memory {
        use crate::types::{Importance, MemoryLayer, MemoryStatus, MemoryTier, Source};
        let now = chrono::Utc::now();
        crate::types::Memory {
            id: ulid::Ulid::new().to_string(),
            layer: MemoryLayer::LTM,
            topic: topic.to_string(),
            summary: content.to_string(),
            content: content.to_string(),
            keywords: vec![],
            importance: Importance::Medium,
            source: Source::Manual,
            strength: 1.0,
            decay_lambda: 0.06,
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
            tier: MemoryTier::Warm,
            cluster_id: None,
            archival_summary: None,
            archival_summary_at: None,
            archival_summary_version: None,
            created_at: now,
            updated_at: now,
            last_accessed: now,
        }
    }

    fn enriched(memory: &crate::types::Memory) -> String {
        prepend_metadata(&memory.topic, &memory.summary, &memory.content)
    }

    #[tokio::test]
    async fn warmup_backfills_live_memory_without_vec_row_from_cache() {
        use crate::types::MemoryStore;
        let store = SqliteStore::in_memory().unwrap();
        let config = ReinConfig::default();
        let dims = config.embedding.dimensions;
        let memory = live_memory("backfill", "cached but never written as a vector row");
        let id = store.store(memory.clone()).unwrap();
        assert!(crate::store::vec::get_embedding(store.conn(), &id)
            .unwrap()
            .is_none());
        EmbedCache::put(
            store.conn(),
            &enriched(&memory),
            &config.embedding_model(),
            &vec![0.5; dims],
        )
        .unwrap();

        let report = backfill_missing_vec_rows(&store, &config, None).await;

        assert_eq!(report.backfilled_from_cache, 1);
        assert_eq!(report.embedded, 0);
        assert_eq!(report.errors, 0);
        assert_eq!(report.skipped_no_provider, 0);
        let stored = crate::store::vec::get_embedding(store.conn(), &id)
            .unwrap()
            .expect("vector row restored from cache");
        assert_eq!(stored.len(), dims);

        // Idempotent: a second pass finds nothing to do.
        let again = backfill_missing_vec_rows(&store, &config, None).await;
        assert_eq!(again, WarmupReport::default());
    }

    #[tokio::test]
    async fn warmup_backfill_skips_deprecated_and_already_indexed_rows() {
        use crate::types::{MemoryStatus, MemoryStore};
        let store = SqliteStore::in_memory().unwrap();
        let config = ReinConfig::default();
        let dims = config.embedding.dimensions;

        let mut deprecated = live_memory("deprecated", "a deprecated row without a vector");
        deprecated.status = MemoryStatus::Deprecated;
        let deprecated_id = store.store(deprecated.clone()).unwrap();
        EmbedCache::put(
            store.conn(),
            &enriched(&deprecated),
            &config.embedding_model(),
            &vec![0.1; dims],
        )
        .unwrap();

        let mut indexed = live_memory("indexed", "already has a vector row");
        indexed.embedding = Some(vec![0.2; dims]);
        let indexed_id = store.store(indexed).unwrap();

        let report = backfill_missing_vec_rows(&store, &config, None).await;

        assert_eq!(report, WarmupReport::default());
        assert!(
            crate::store::vec::get_embedding(store.conn(), &deprecated_id)
                .unwrap()
                .is_none()
        );
        assert!(crate::store::vec::get_embedding(store.conn(), &indexed_id)
            .unwrap()
            .is_some());
    }

    #[tokio::test]
    async fn warmup_backfill_counts_cache_misses_without_provider() {
        use crate::types::MemoryStore;
        let store = SqliteStore::in_memory().unwrap();
        let config = ReinConfig::default();
        let memory = live_memory("miss", "no cache entry and no provider");
        let id = store.store(memory).unwrap();

        let report = backfill_missing_vec_rows(&store, &config, None).await;

        assert_eq!(report.skipped_no_provider, 1);
        assert_eq!(report.rows_added(), 0);
        assert!(crate::store::vec::get_embedding(store.conn(), &id)
            .unwrap()
            .is_none());
    }

    #[cfg(feature = "test-support")]
    #[tokio::test]
    async fn warmup_aborts_model_migration_without_writing_rows() {
        use crate::types::MemoryStore;
        let (_dir, store) = test_store();
        let config = ReinConfig::default();
        let dims = config.embedding.dimensions;
        let mut memory = live_memory("partial", "row from the previous model");
        memory.embedding = Some(vec![0.9; dims]);
        let id = store.store(memory).unwrap();
        store.conn().execute("DELETE FROM embed_cache", []).unwrap();
        write_metadata(
            &store,
            VEC_ROWS_PROVENANCE_KEY,
            &format!("old-model:{dims}"),
        );
        let failing = crate::embed::EmbedderKind::Mock(
            crate::embed::MockEmbedder::with_persistent_error(dims, "provider down"),
        );

        let report = warmup_with_embedder(&store, &config, Some(&failing)).await;

        assert!(report.model_changed);
        assert!(report.errors > 0);
        assert_eq!(report.rows_added(), 0);
        let kept = crate::store::vec::get_embedding(store.conn(), &id)
            .unwrap()
            .expect("old-model row is left untouched by a failed migration");
        assert!((kept[0] - 0.9).abs() < 1e-6);
        assert_eq!(
            read_metadata(&store, VEC_ROWS_PROVENANCE_KEY).as_deref(),
            Some(format!("old-model:{dims}").as_str()),
            "provenance is not advanced by an incomplete migration"
        );
    }

    #[test]
    fn replace_embedding_row_keeps_old_vector_when_insert_fails() {
        use crate::types::MemoryStore;
        let store = SqliteStore::in_memory().unwrap();
        let dims = ReinConfig::default().embedding.dimensions;
        let mut memory = live_memory("atomic", "row with a vector");
        memory.embedding = Some(vec![0.9; dims]);
        let id = store.store(memory).unwrap();

        // Wrong dimension: sqlite-vec rejects the insert after the delete ran.
        let wrong = vec![0.1; dims + 1];
        assert!(replace_embedding_row(store.conn(), &id, &wrong, true).is_err());
        let kept = crate::store::vec::get_embedding(store.conn(), &id)
            .unwrap()
            .expect("previous vector survives a failed replacement");
        assert!((kept[0] - 0.9).abs() < 1e-6);

        replace_embedding_row(store.conn(), &id, &vec![0.3; dims], true).unwrap();
        let replaced = crate::store::vec::get_embedding(store.conn(), &id)
            .unwrap()
            .unwrap();
        assert!((replaced[0] - 0.3).abs() < 1e-6);
    }

    #[cfg(feature = "test-support")]
    #[tokio::test]
    async fn warmup_reembeds_every_row_after_model_change() {
        use crate::types::MemoryStore;
        let store = SqliteStore::in_memory().unwrap();
        let config = ReinConfig::default();
        let dims = config.embedding.dimensions;
        let mut memory = live_memory("model-change", "row written by the previous model");
        memory.embedding = Some(vec![0.9; dims]);
        let id = store.store(memory).unwrap();
        // store() also caches the supplied embedding under the current model
        // key; drop it so the only way to a current-model vector is the
        // provider (a real model change would miss the cache the same way).
        store.conn().execute("DELETE FROM embed_cache", []).unwrap();
        write_metadata(
            &store,
            VEC_ROWS_PROVENANCE_KEY,
            &format!("old-model:{dims}"),
        );
        let embedder = crate::embed::EmbedderKind::Mock(
            crate::embed::MockEmbedder::with_fixed_vector(dims, vec![0.25; dims]),
        );

        let report = backfill_missing_vec_rows(&store, &config, Some(&embedder)).await;

        assert_eq!(
            report.embedded, 1,
            "existing row re-embedded under the new model"
        );
        let stored = crate::store::vec::get_embedding(store.conn(), &id)
            .unwrap()
            .unwrap();
        assert!((stored[0] - 0.25).abs() < 1e-6);
        assert_eq!(
            read_metadata(&store, VEC_ROWS_PROVENANCE_KEY).as_deref(),
            Some(format!("{}:{dims}", config.embedding_model()).as_str())
        );
        // Same model again: nothing to do.
        let again = backfill_missing_vec_rows(&store, &config, Some(&embedder)).await;
        assert_eq!(again, WarmupReport::default());
    }

    #[cfg(feature = "test-support")]
    #[tokio::test]
    async fn warmup_backfill_embeds_cache_misses_with_provider() {
        use crate::types::MemoryStore;
        let store = SqliteStore::in_memory().unwrap();
        let config = ReinConfig::default();
        let dims = config.embedding.dimensions;
        let memory = live_memory("embed", "no cache entry, provider available");
        let id = store.store(memory.clone()).unwrap();
        let embedder = crate::embed::EmbedderKind::Mock(
            crate::embed::MockEmbedder::with_fixed_vector(dims, vec![0.25; dims]),
        );

        let report = backfill_missing_vec_rows(&store, &config, Some(&embedder)).await;

        assert_eq!(report.embedded, 1);
        assert_eq!(report.errors, 0);
        let stored = crate::store::vec::get_embedding(store.conn(), &id)
            .unwrap()
            .expect("vector row written after embedding");
        assert_eq!(stored.len(), dims);
        assert!(
            EmbedCache::get(store.conn(), &enriched(&memory), &config.embedding_model())
                .unwrap()
                .is_some(),
            "the embedding is also cached for store-time reuse"
        );
    }
}
