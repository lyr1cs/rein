use crate::config::ReinConfig;
use crate::search::chunker::semantic_chunk;
use crate::store::SqliteStore;
use crate::types::{Importance, Memory, MemoryLayer, MemoryStatus, MemoryStore, Source};

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
                        all_embeddings.extend(
                            std::iter::repeat_with(Vec::new).take(batch.len()),
                        );
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
                related_ids: vec![],
                concept_ids: vec![],
                status: MemoryStatus::default(),
                embedding,
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
pub async fn reindex(
    store: &SqliteStore,
    config: &ReinConfig,
) -> anyhow::Result<ReindexReport> {
    use crate::store::schema;
    use crate::types::Embedder as _;

    // 1. Validate embedder is available BEFORE touching the vector index
    let embedder = crate::embed::create_embedder(config)
        .ok_or_else(|| anyhow::anyhow!("no embedding provider configured (set provider and API key)"))?;

    // 2. Health check: test embed one string to verify the API works
    let test_result = embedder.embed("health check").await;
    if let Err(e) = test_result {
        return Err(anyhow::anyhow!("embedding health check failed: {e}. Vector index NOT modified."));
    }

    // 3. Now safe to rebuild vector index
    schema::rebuild_vector_index(store.conn(), config.embedding.dimensions)?;

    // Delete stale HNSW and Tantivy side indexes (will be rebuilt after re-embedding)
    let hnsw_path = store.db_path().with_extension("");
    let _ = std::fs::remove_file(hnsw_path.with_extension("usearch"));
    let _ = std::fs::remove_file(hnsw_path.with_extension("usearch.meta"));
    let tantivy_path = store.db_path().with_extension("tantivy");
    let _ = std::fs::remove_dir_all(&tantivy_path);

    // 4. Get all memories
    let mut stmt = store
        .conn()
        .prepare("SELECT id, topic, summary, content FROM memories")?;
    let rows: Vec<(String, String, String, String)> = stmt
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)))?
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

    // 5. Batch embed and insert vectors
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
                    tracing::warn!("batch returned {} embeddings for {} inputs, skipping batch", embs.len(), chunk.len());
                    errors += chunk.len();
                    continue;
                }
                for (i, emb) in embs.iter().enumerate() {
                    let id = &chunk[i].0;
                    if let Err(e) = crate::store::vec::insert_embedding(store.conn(), id, emb) {
                        tracing::warn!("failed to insert embedding for {id}: {e}");
                        errors += 1;
                    } else {
                        embedded += 1;
                    }
                }
            }
            Err(e) => {
                tracing::warn!("batch embed failed: {e}");
                errors += chunk.len();
            }
        }

        eprintln!("[{}/{}] Reindexing...", embedded + errors, total);
    }

    // 5. Clear embed_cache (old model's cache)
    store.conn().execute("DELETE FROM embed_cache", [])?;

    // 6. Rebuild HNSW and Tantivy side indexes from fresh data
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
                rusqlite::params!["hash2", "Second document about Rust programming. Memory management is important."],
            ).unwrap();

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

        let report = migrate_from_qmd(qmd_path, &store, &config, None::<&crate::embed::EmbedderKind>)
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
