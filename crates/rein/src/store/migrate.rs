use crate::config::ReinConfig;
use crate::search::chunker::semantic_chunk;
use crate::store::SqliteStore;
use crate::types::error::ReinResult;
use crate::types::{
    Importance, Memory, MemoryLayer, MemoryStatus, MemoryStore, MemoryTier, Source,
};

/// Report summarizing a QMD migration run.
pub struct MigrationReport {
    pub documents_read: usize,
    pub chunks_created: usize,
    pub embeddings_generated: usize,
    pub errors: usize,
}

impl std::fmt::Display for MigrationReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Migration complete: {} documents -> {} chunks, {} embeddings, {} errors",
            self.documents_read, self.chunks_created, self.embeddings_generated, self.errors
        )
    }
}

/// A single document read from the QMD database.
struct QmdDocument {
    collection: String,
    title: String,
    content: String,
}

/// Migrate data from a QMD SQLite database into the rein memory store.
///
/// QMD schema:
///   documents: id, collection, path, title, hash, active
///   content: hash PK, doc TEXT
pub async fn migrate_from_qmd<E: crate::types::Embedder>(
    qmd_path: &std::path::Path,
    store: &SqliteStore,
    config: &ReinConfig,
    embedder: Option<&E>,
) -> anyhow::Result<MigrationReport> {
    // 1. Open QMD SQLite (read-only)
    let qmd_conn = rusqlite::Connection::open_with_flags(
        qmd_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    )?;

    // 2. Read all active documents with content
    let mut stmt = qmd_conn.prepare(
        "SELECT d.collection, d.title, c.doc FROM documents d JOIN content c ON d.hash = c.hash WHERE d.active = 1",
    )?;

    let documents: Vec<QmdDocument> = stmt
        .query_map([], |row| {
            Ok(QmdDocument {
                collection: row.get(0)?,
                title: row.get(1)?,
                content: row.get(2)?,
            })
        })?
        .filter_map(|r| match r {
            Ok(m) => Some(m),
            Err(e) => {
                tracing::warn!("failed to deserialize memory row: {e}");
                None
            }
        })
        .collect();

    let total = documents.len();
    let mut report = MigrationReport {
        documents_read: total,
        chunks_created: 0,
        embeddings_generated: 0,
        errors: 0,
    };

    // 3. For each document, chunk and store
    for (idx, doc) in documents.iter().enumerate() {
        let chunks = semantic_chunk(
            &doc.content,
            config.chunking.max_tokens,
            config.chunking.overlap_percent,
        );

        // Optionally batch-embed the chunks
        let embeddings: Option<Vec<Vec<f32>>> = if let Some(emb) = embedder {
            let chunk_refs: Vec<&str> = chunks.iter().map(|s| s.as_str()).collect();
            let mut all_embeddings: Vec<Vec<f32>> = Vec::with_capacity(chunks.len());
            for batch in chunk_refs.chunks(100) {
                match emb.embed_batch(batch).await {
                    Ok(batch_embs) => {
                        report.embeddings_generated += batch_embs.len();
                        all_embeddings.extend(batch_embs);
                    }
                    Err(e) => {
                        tracing::warn!("Embedding batch failed: {e}");
                        report.errors += 1;
                        all_embeddings.extend(std::iter::repeat_with(Vec::new).take(batch.len()));
                    }
                }
            }
            Some(all_embeddings)
        } else {
            None
        };

        for (chunk_idx, chunk) in chunks.iter().enumerate() {
            let now = chrono::Utc::now();
            let importance = Importance::Medium;
            let embedding = embeddings
                .as_ref()
                .and_then(|embs| embs.get(chunk_idx))
                .filter(|e| !e.is_empty())
                .cloned();

            let memory = Memory {
                id: ulid::Ulid::new().to_string(),
                layer: MemoryLayer::STM,
                topic: doc.collection.clone(),
                summary: doc.title.clone(),
                content: chunk.clone(),
                keywords: vec![],
                importance,
                source: Source::Migration,
                strength: 1.0,
                decay_lambda: config.decay.base_lambda * importance.decay_factor(),
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
                status: MemoryStatus::default(),
                embedding,
                tier: MemoryTier::Warm,
                cluster_id: None,
                archival_summary: None,
                archival_summary_at: None,
                archival_summary_version: None,
                created_at: now,
                updated_at: now,
                last_accessed: now,
            };

            match store.store(memory) {
                Ok(_) => report.chunks_created += 1,
                Err(e) => {
                    tracing::warn!("Failed to store chunk: {e}");
                    report.errors += 1;
                }
            }
        }

        println!("[{}/{}] Migrated: {}", idx + 1, total, doc.title);
    }

    Ok(report)
}

/// Report summarizing a reindex run.
pub struct ReindexReport {
    pub total: usize,
    pub embedded: usize,
    pub errors: usize,
}

impl std::fmt::Display for ReindexReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Reindex complete: {}/{} memories embedded, {} errors",
            self.embedded, self.total, self.errors
        )
    }
}

/// Re-embed all existing memories with the current embedding model.
///
/// 1. Rebuilds the vector index (drops old vec_memories, creates new with current dims)
/// 2. Fetches all memories and batch-embeds them
/// 3. Clears the embed_cache (old model's cache)
pub async fn reindex(store: &SqliteStore, config: &ReinConfig) -> anyhow::Result<ReindexReport> {
    use crate::store::schema;
    use crate::types::Embedder as _;

    // 1. Validate embedder is available BEFORE touching the vector index
    let embedder = crate::embed::create_embedder(config).ok_or_else(|| {
        anyhow::anyhow!("no embedding provider configured (set provider and API key)")
    })?;

    // 2. Health check: test embed one string to verify the API works
    let test_result = embedder.embed("health check").await;
    if let Err(e) = test_result {
        return Err(anyhow::anyhow!(
            "embedding health check failed: {e}. Vector index NOT modified."
        ));
    }

    // 3. Get all memories.  Do not touch the existing vector index until every
    // replacement embedding has been computed successfully.
    let mut stmt = store
        .conn()
        .prepare("SELECT id, topic, summary, content FROM memories")?;
    let rows: Vec<(String, String, String, String)> = stmt
        .query_map([], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
        })?
        .filter_map(|r| r.ok())
        .collect();

    let total = rows.len();
    if total == 0 {
        return Ok(ReindexReport {
            total: 0,
            embedded: 0,
            errors: 0,
        });
    }

    // 4. Batch embed and STREAM each batch to a staging table so memory usage
    // stays O(batch) rather than O(total). At 3072-dim floats, 100k memories
    // would otherwise buffer >1GB before the swap — the staging table keeps
    // it on disk and lets us drop batches as they're persisted.
    schema::create_embed_staging(store.conn())?;
    let mut embedded = 0usize;
    let mut errors = 0usize;

    for chunk in rows.chunks(50) {
        let texts: Vec<String> = chunk
            .iter()
            .map(|(_, topic, summary, content)| {
                crate::embed::prepend_metadata(topic, summary, content)
            })
            .collect();
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

        match embedder.embed_batch(&text_refs).await {
            Ok(embs) => {
                if embs.len() != chunk.len() {
                    tracing::warn!(
                        "batch returned {} embeddings for {} inputs, skipping batch",
                        embs.len(),
                        chunk.len()
                    );
                    errors += chunk.len();
                    continue;
                }
                // Persist this batch into the staging table inside its own short
                // transaction — fast sequential appends, no buffered state across batches.
                let tx_result: ReinResult<()> = (|| {
                    store.conn().execute_batch("BEGIN")?;
                    {
                        let mut stmt = store.conn().prepare(
                            "INSERT OR REPLACE INTO embed_staging(id, embedding) VALUES (?1, ?2)",
                        )?;
                        for (i, emb) in embs.iter().enumerate() {
                            let id = &chunk[i].0;
                            if emb.len() != config.embedding.dimensions {
                                tracing::warn!(
                                    "embedding for {id} has {} dims, expected {}",
                                    emb.len(),
                                    config.embedding.dimensions
                                );
                                errors += 1;
                                continue;
                            }
                            let bytes: Vec<u8> = emb.iter().flat_map(|f| f.to_le_bytes()).collect();
                            stmt.execute(rusqlite::params![id, bytes])?;
                            embedded += 1;
                        }
                    }
                    store.conn().execute_batch("COMMIT")?;
                    Ok(())
                })();
                if let Err(e) = tx_result {
                    let _ = store.conn().execute_batch("ROLLBACK");
                    tracing::warn!("staging batch failed: {e}");
                    errors += chunk.len();
                }
            }
            Err(e) => {
                tracing::warn!("batch embed failed: {e}");
                errors += chunk.len();
            }
        }

        eprintln!("[{}/{}] Reindexing...", embedded + errors, total);
    }

    if errors > 0 {
        // Clean up staging so a subsequent run starts fresh.
        let _ = store
            .conn()
            .execute_batch("DROP TABLE IF EXISTS embed_staging;");
        anyhow::bail!("embedding failed for {errors}/{total} memories; vector index NOT modified");
    }

    // 5. Atomically replace vec_memories + embed_cache from the staging table.
    // A failure here rolls back to the old index (staging is preserved for retry).
    schema::replace_vector_index_from_staging(store.conn(), config.embedding.dimensions)?;

    // 6. The embedding space just changed wholesale while the row COUNT
    // stayed identical, so the #17 count-delta recluster cadence gate
    // would otherwise skip HDBSCAN indefinitely — and every
    // cluster-derived surface (memories.cluster_id fallback in recall,
    // cluster_centroids in dedup assignment, per-cluster survival
    // curves, cluster-scoped alpha / shadow-fusion / dedup-threshold
    // maps) would keep serving labels learned in the OLD space until
    // then. Clear them all: the next adaptive pass reclusters
    // unconditionally (`should_recluster` on an empty map) and rebuilds
    // in the new space; until then lookups degrade to the global
    // fallbacks, which is correct — old-space cluster geometry is
    // meaningless against new-space embeddings.
    //
    // Ordering (codex R11): this reset runs IMMEDIATELY after the swap,
    // BEFORE the slow HNSW/Tantivy rebuild below — a live server (or an
    // interruption during that rebuild) must not keep serving old-space
    // cluster labels for the minutes the rebuild can take. The churn
    // credit itself rides inside `replace_vector_index_from_staging`'s
    // savepoint (codex R10), so even dying right here leaves the cadence
    // gate re-armed and the next adaptive pass self-heals (codex R8).
    //
    // Within the reset, the SNAPSHOT generation bump commits FIRST and the
    // SQL row clears second (codex R14): a concurrent adaptive HDBSCAN run
    // on the old vectors re-checks the persisted `cluster_version` inside
    // its write savepoint right before committing its labels. SQLite's
    // single-writer serialization then leaves it only two outcomes — it
    // serializes after the bump (guard sees the new generation → aborts),
    // or it commits entirely before this reset (the SQL clears below wipe
    // its rows). Clears-first would open a third interleaving where the
    // stale run passes its guard pre-bump and repopulates the rows
    // post-clear.
    if let Some(mut state) = crate::store::adaptive::AdaptiveState::restore_snapshot(store.conn()) {
        state.memory_clusters.clear();
        state.clear_cluster_scoped_learned_state();
        state.canonical_length_stats.clear();
        state.last_recluster_embedding_count = 0;
        state.last_recluster_embedding_write_seq = 0;
        // The cleared map IS a new clustering generation: bumping
        // cluster_version makes this reset dominate `save_snapshot`'s CAS
        // merge, so an adaptive pipeline writer that loaded its snapshot
        // BEFORE the reindex cannot merge old-space `memory_clusters` /
        // cluster-scoped weights back in. Bump by TWO (codex R8): a
        // concurrent adaptive pass that loaded generation N and reclusters
        // on pre-reindex embeddings saves N+1 — with a +1 reset it would
        // TIE and its `we_reclustered` wholesale-replace could overwrite
        // this reset with old-space labels. A single-step writer can never
        // tie N+2. centroid_version tracks it as in
        // `run_hdbscan_clustering`.
        state.cluster_version += 2;
        state.centroid_version = state.cluster_version;
        state.version += 1;
        if let Err(e) = state.save_snapshot(store.conn()) {
            // Fail loud (codex R5): with the old snapshot intact
            // (non-empty `memory_clusters`, matching baselines, and a
            // churn counter the bulk swap never bumped), the cadence gate
            // would skip HDBSCAN indefinitely on old-space labels.
            // `save_snapshot` already retried its CAS merge internally,
            // so a persistent failure here is a real storage problem.
            anyhow::bail!(
                "reindex embedded {embedded}/{total} memories and swapped the vector index, \
                 but persisting the cluster-state reset failed: {e}. Re-run \
                 `rein migrate --reindex` so the adaptive engine reclusters in the new \
                 embedding space."
            );
        }
    }
    let sql_reset: ReinResult<()> = (|| {
        store.conn().execute_batch("BEGIN")?;
        store
            .conn()
            .execute("UPDATE memories SET cluster_id = NULL", [])?;
        store.conn().execute("DELETE FROM cluster_centroids", [])?;
        store
            .conn()
            .execute("DELETE FROM metadata WHERE key LIKE 'survival_curve:%'", [])?;
        store.conn().execute_batch("COMMIT")?;
        Ok(())
    })();
    if let Err(e) = sql_reset {
        let _ = store.conn().execute_batch("ROLLBACK");
        // Fail loud (codex R5): the new embeddings are committed, but old-
        // space cluster rows survived. The bulk swap bypassed
        // `insert_embedding`, so the churn counter did not move either —
        // nothing would ever force the recluster.
        anyhow::bail!(
            "reindex embedded {embedded}/{total} memories and swapped the vector index, \
             but clearing stale cluster rows failed: {e}. Re-run `rein migrate --reindex` \
             so recall/dedup stop using cluster labels from the old embedding space."
        );
    }

    // 7. Rebuild HNSW and Tantivy side indexes from fresh data (after the
    // cluster reset above — codex R11 ordering).
    let hnsw_path = store.db_path().with_extension("");
    let _ = std::fs::remove_file(hnsw_path.with_extension("usearch"));
    let _ = std::fs::remove_file(hnsw_path.with_extension("usearch.meta"));
    let tantivy_path = store.db_path().with_extension("tantivy");
    let _ = std::fs::remove_dir_all(&tantivy_path);
    crate::search::warmup::populate_hnsw(store, config);
    crate::search::warmup::populate_tantivy(store);

    Ok(ReindexReport {
        total,
        embedded,
        errors,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_migrate_qmd_mock() {
        // Create a mock QMD database
        let qmd_file = NamedTempFile::new().unwrap();
        let qmd_path = qmd_file.path();

        {
            let conn = rusqlite::Connection::open(qmd_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE documents (
                    id INTEGER PRIMARY KEY,
                    collection TEXT NOT NULL,
                    path TEXT NOT NULL,
                    title TEXT NOT NULL,
                    hash TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1
                );
                CREATE TABLE content (
                    hash TEXT PRIMARY KEY,
                    doc TEXT NOT NULL
                );",
            )
            .unwrap();

            conn.execute(
                "INSERT INTO content (hash, doc) VALUES (?1, ?2)",
                rusqlite::params!["hash1", "This is the first document content. It has multiple sentences. Should be stored."],
            ).unwrap();
            conn.execute(
                "INSERT INTO content (hash, doc) VALUES (?1, ?2)",
                rusqlite::params![
                    "hash2",
                    "Second document about Rust programming. Memory management is important."
                ],
            )
            .unwrap();

            conn.execute(
                "INSERT INTO documents (id, collection, path, title, hash, active) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                rusqlite::params![1, "notes", "/notes/doc1.md", "First Doc", "hash1", 1],
            ).unwrap();
            conn.execute(
                "INSERT INTO documents (id, collection, path, title, hash, active) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                rusqlite::params![2, "rust", "/rust/doc2.md", "Second Doc", "hash2", 1],
            ).unwrap();

            // Inactive document — should NOT be migrated
            conn.execute(
                "INSERT INTO content (hash, doc) VALUES (?1, ?2)",
                rusqlite::params!["hash3", "Inactive document content."],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO documents (id, collection, path, title, hash, active) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                rusqlite::params![3, "archive", "/archive/doc3.md", "Inactive Doc", "hash3", 0],
            ).unwrap();
        }

        // Create rein store
        let store = SqliteStore::in_memory().unwrap();
        let config = ReinConfig::default();

        let report = migrate_from_qmd(
            qmd_path,
            &store,
            &config,
            None::<&crate::embed::EmbedderKind>,
        )
        .await
        .unwrap();

        assert_eq!(report.documents_read, 2);
        assert!(
            report.chunks_created >= 2,
            "Expected at least 2 chunks, got {}",
            report.chunks_created
        );
        assert_eq!(report.embeddings_generated, 0);
        assert_eq!(report.errors, 0);

        // Verify memories exist in the store
        let topics = store.list_topics().unwrap();
        assert!(
            topics.contains(&"notes".to_string()),
            "Expected 'notes' topic"
        );
        assert!(
            topics.contains(&"rust".to_string()),
            "Expected 'rust' topic"
        );
    }
}
