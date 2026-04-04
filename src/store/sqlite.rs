use chrono::{DateTime, Utc};
use rusqlite::Connection;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use crate::extract::DedupAction;
use crate::types::*;

use super::{fts, schema, vec};

/// Report from store_knowledge_units().
#[derive(Debug, Default)]
pub struct KnowledgeStoreReport {
    pub memoirs_created: usize,
    pub concepts_added: usize,
    pub concepts_refined: usize,
    pub links_added: usize,
}

/// SQLite-backed memory store with FTS5 and vector search.
///
/// Wraps `rusqlite::Connection` which is `!Send`. All database access should
/// happen on the thread that created the connection. The MCP server uses
/// per-request connections with `SQLITE_OPEN_FULL_MUTEX` (serialized mode).
pub struct SqliteStore {
    pub(crate) conn: Connection,
    db_path: PathBuf,
    pub(crate) dims: usize,
    /// Cached Tantivy index — avoids reopening + allocating 15MB IndexWriter per operation.
    tantivy_cache: std::cell::RefCell<Option<super::tantivy_fts::TantivyFts>>,
}

impl SqliteStore {
    /// Open or create a database at the given path.
    /// Uses SQLITE_OPEN_FULL_MUTEX for thread-safe access via serialized mode.
    /// `model` and `dims` track the embedding model; if changed, vector index is rebuilt.
    pub fn new(path: &Path, model: &str, dims: usize) -> ReinResult<Self> {
        schema::init_sqlite_vec();
        let conn = Connection::open_with_flags(
            path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_WRITE
                | rusqlite::OpenFlags::SQLITE_OPEN_CREATE
                | rusqlite::OpenFlags::SQLITE_OPEN_FULL_MUTEX,
        )?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;
        schema::init_schema(&conn, dims)?;

        // Check if embedding model changed since last run (warn only, don't auto-rebuild)
        if schema::check_embedding_model(&conn, model, dims)? {
            eprintln!("rein: WARNING — embedding model changed to '{model}' ({dims}d).");
            eprintln!("rein: Existing vectors are incompatible. Run 'rein migrate --reindex' to rebuild.");
            eprintln!("rein: FTS search still works. Vector search may return incorrect results.");
        }

        Ok(Self { conn, db_path: path.to_path_buf(), dims, tantivy_cache: std::cell::RefCell::new(None) })
    }

    /// Create an in-memory database for testing (default 3072 dims).
    pub fn in_memory() -> ReinResult<Self> {
        schema::init_sqlite_vec();
        let conn = Connection::open_in_memory()?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;
        schema::init_schema(&conn, 3072)?;
        Ok(Self { conn, db_path: PathBuf::from(":memory:"), dims: 3072, tantivy_cache: std::cell::RefCell::new(None) })
    }
    /// Access the underlying SQLite connection (for direct queries).
    pub fn conn(&self) -> &Connection {
        &self.conn
    }

    /// The path to the database file (or ":memory:" for in-memory databases).
    pub fn db_path(&self) -> &Path {
        &self.db_path
    }

    /// Get all memories in a topic (for dedup scanning).
    pub fn get_by_topic(&self, topic: &str) -> ReinResult<Vec<Memory>> {
        let mut stmt = self.conn.prepare("SELECT * FROM memories WHERE topic = ?1")?;
        let rows = stmt.query_map(rusqlite::params![topic], |row| {
            row_to_memory(row).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e))
            })
        })?;
        Ok(rows.filter_map(|r| match r {
            Ok(m) => Some(m),
            Err(e) => {
                tracing::warn!("failed to deserialize memory row: {e}");
                None
            }
        }).collect())
    }

    /// Batch-get memories by IDs. Returns found memories (skips missing ones).
    /// More efficient than calling get() in a loop — single query with WHERE id IN (...).
    pub fn get_batch(&self, ids: &[String]) -> Vec<Memory> {
        if ids.is_empty() { return vec![]; }
        // SQLite doesn't support array parameters, so build a parameterized IN clause
        let placeholders: Vec<String> = (1..=ids.len()).map(|i| format!("?{i}")).collect();
        let sql = format!("SELECT * FROM memories WHERE id IN ({})", placeholders.join(","));
        let params: Vec<&dyn rusqlite::types::ToSql> = ids.iter().map(|id| id as &dyn rusqlite::types::ToSql).collect();
        let mut stmt = match self.conn.prepare(&sql) {
            Ok(s) => s,
            Err(_) => return vec![],
        };
        let rows = match stmt.query_map(params.as_slice(), |row| {
            row_to_memory(row).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e))
            })
        }) {
            Ok(r) => r,
            Err(_) => return vec![],
        };
        rows.filter_map(|r| r.ok()).collect()
    }

    /// Get the most recently created memories.
    pub fn recent(&self, limit: usize) -> ReinResult<Vec<Memory>> {
        let mut stmt = self.conn.prepare(
            "SELECT * FROM memories ORDER BY created_at DESC LIMIT ?1"
        )?;
        let rows = stmt.query_map(rusqlite::params![limit], |row| {
            row_to_memory(row).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e))
            })
        })?;
        Ok(rows.filter_map(|r| match r {
            Ok(m) => Some(m),
            Err(e) => {
                tracing::warn!("failed to deserialize memory row: {e}");
                None
            }
        }).collect())
    }

    /// Persist a raw session artifact separately from derived memories.
    pub fn store_session_artifact(&self, mut artifact: SessionArtifact) -> ReinResult<String> {
        let id = if artifact.id.is_empty() {
            ulid::Ulid::new().to_string()
        } else {
            artifact.id.clone()
        };
        artifact.id = id.clone();
        let created_at = if artifact.created_at.timestamp() == 0 {
            Utc::now()
        } else {
            artifact.created_at
        };

        self.conn.execute(
            "INSERT INTO session_artifacts (
                id, schema_version, artifact_kind, session_id, title, summary, source_agent,
                source_label, is_subagent, started_at, ended_at, turn_count, transcript_text,
                transcript_json, episode_id, created_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)",
            rusqlite::params![
                &artifact.id,
                artifact.schema_version,
                artifact.artifact_kind,
                artifact.session_id,
                artifact.title,
                artifact.summary,
                artifact.source_agent,
                artifact.source_label,
                if artifact.is_subagent { 1 } else { 0 },
                artifact.started_at.map(|dt| dt.to_rfc3339()),
                artifact.ended_at.map(|dt| dt.to_rfc3339()),
                artifact.turn_count,
                artifact.transcript_text,
                artifact.transcript_json,
                artifact.episode_id,
                created_at.to_rfc3339(),
            ],
        )?;

        Ok(id)
    }

    /// Link a raw session artifact to the episode created from it.
    pub fn link_session_artifact_episode(
        &self,
        artifact_id: &str,
        episode_id: &str,
    ) -> ReinResult<()> {
        let rows = self.conn.execute(
            "UPDATE session_artifacts SET episode_id = ?1 WHERE id = ?2",
            rusqlite::params![episode_id, artifact_id],
        )?;
        if rows == 0 {
            return Err(ReinError::NotFound(format!(
                "session artifact '{}' not found",
                artifact_id
            )));
        }
        Ok(())
    }

}

/// Map a rusqlite Row to a Memory struct.
///
/// Expected column order matches `SELECT * FROM memories`:
/// id, layer, topic, summary, content, keywords, importance, source,
/// strength, decay_lambda, access_count, superseded_by, related_ids,
/// created_at, updated_at, last_accessed
pub fn row_to_memory(row: &rusqlite::Row) -> ReinResult<Memory> {
    let id: String = row.get("id").map_err(ReinError::Database)?;
    let layer_str: String = row.get("layer").map_err(ReinError::Database)?;
    let topic: String = row.get("topic").map_err(ReinError::Database)?;
    let summary: String = row.get("summary").map_err(ReinError::Database)?;
    let content: String = row.get("content").map_err(ReinError::Database)?;
    let keywords_json: String = row.get("keywords").map_err(ReinError::Database)?;
    let importance_str: String = row.get("importance").map_err(ReinError::Database)?;
    let source_str: String = row.get("source").map_err(ReinError::Database)?;
    let strength: f64 = row.get("strength").map_err(ReinError::Database)?;
    let decay_lambda: f64 = row.get("decay_lambda").map_err(ReinError::Database)?;
    let access_count: u32 = row.get("access_count").map_err(ReinError::Database)?;
    let superseded_by: Option<String> = row.get("superseded_by").map_err(ReinError::Database)?;
    let related_ids_json: String = row.get("related_ids").map_err(ReinError::Database)?;
    let concept_ids_json: String = row.get("concept_ids").unwrap_or_else(|_| "[]".to_string());
    let status_str: String = row.get::<_, String>("status").unwrap_or_else(|_| "active".to_string());
    let created_at_str: String = row.get("created_at").map_err(ReinError::Database)?;
    let updated_at_str: String = row.get("updated_at").map_err(ReinError::Database)?;
    let last_accessed_str: String = row.get("last_accessed").map_err(ReinError::Database)?;

    let layer = MemoryLayer::from_str(&layer_str)
        .map_err(ReinError::Config)?;
    let importance = Importance::from_str(&importance_str)
        .map_err(ReinError::Config)?;
    let source = Source::from_str(&source_str)
        .map_err(ReinError::Config)?;
    let status = MemoryStatus::from_str(&status_str)
        .unwrap_or_default();

    let keywords: Vec<String> = serde_json::from_str(&keywords_json)?;
    let related_ids: Vec<String> = serde_json::from_str(&related_ids_json)?;
    let concept_ids: Vec<String> = serde_json::from_str(&concept_ids_json).unwrap_or_default();

    let created_at = DateTime::parse_from_rfc3339(&created_at_str)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| ReinError::Config(format!("invalid created_at: {e}")))?;
    let updated_at = DateTime::parse_from_rfc3339(&updated_at_str)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| ReinError::Config(format!("invalid updated_at: {e}")))?;
    let last_accessed = DateTime::parse_from_rfc3339(&last_accessed_str)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| ReinError::Config(format!("invalid last_accessed: {e}")))?;

    Ok(Memory {
        id,
        layer,
        topic,
        summary,
        content,
        keywords,
        importance,
        source,
        strength,
        decay_lambda,
        access_count,
        superseded_by,
        related_ids,
        concept_ids,
        status,
        embedding: None,
        tier: row.get::<_, String>("tier").unwrap_or_else(|_| "warm".to_string()),
        cluster_id: row.get::<_, Option<u32>>("cluster_id").unwrap_or(None),
        created_at,
        updated_at,
        last_accessed,
    })
}

impl MemoryStore for SqliteStore {
    fn store(&self, mut memory: Memory) -> ReinResult<String> {
        let id = if memory.id.is_empty() {
            ulid::Ulid::new().to_string()
        } else {
            memory.id.clone()
        };
        memory.id = id.clone();
        let now = Utc::now();
        memory.created_at = now;
        memory.updated_at = now;
        memory.last_accessed = now;

        let keywords_json = serde_json::to_string(&memory.keywords)?;
        let related_ids_json = serde_json::to_string(&memory.related_ids)?;
        let concept_ids_json = serde_json::to_string(&memory.concept_ids)?;

        // Store layer as uppercase for SQL CHECK constraint
        let layer_db = match memory.layer {
            MemoryLayer::LTM => "LTM",
            MemoryLayer::STM => "STM",
        };

        self.conn.execute(
            "INSERT INTO memories (id, layer, topic, summary, content, keywords, importance, source,
             strength, decay_lambda, access_count, superseded_by, related_ids, concept_ids, status,
             created_at, updated_at, last_accessed)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18)",
            rusqlite::params![
                id,
                layer_db,
                memory.topic,
                memory.summary,
                memory.content,
                keywords_json,
                memory.importance.as_str(),
                memory.source.as_str(),
                memory.strength,
                memory.decay_lambda,
                memory.access_count,
                memory.superseded_by,
                related_ids_json,
                concept_ids_json,
                memory.status.to_string(),
                memory.created_at.to_rfc3339(),
                memory.updated_at.to_rfc3339(),
                memory.last_accessed.to_rfc3339(),
            ],
        )?;

        if let Some(ref emb) = memory.embedding {
            vec::insert_embedding(&self.conn, &id, emb)?;
        }

        // Update side indexes (reuse keywords_json from above)
        self.update_tantivy(&id, &memory.topic, &memory.summary, &memory.content, &keywords_json);
        self.update_hnsw(&id, memory.embedding.as_deref());

        Ok(id)
    }

    fn get(&self, id: &str) -> ReinResult<Memory> {
        let mut stmt = self.conn.prepare("SELECT * FROM memories WHERE id = ?1")?;
        let memory = stmt
            .query_row(rusqlite::params![id], |row| {
                row_to_memory(row).map_err(|e| {
                    rusqlite::Error::FromSqlConversionFailure(
                        0,
                        rusqlite::types::Type::Text,
                        Box::new(e),
                    )
                })
            })
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => {
                    ReinError::NotFound(format!("memory {id} not found"))
                }
                other => ReinError::Database(other),
            })?;

        Ok(memory)
    }

    fn update(&self, memory: &Memory) -> ReinResult<()> {
        let keywords_json = serde_json::to_string(&memory.keywords)?;
        let related_ids_json = serde_json::to_string(&memory.related_ids)?;
        let concept_ids_json = serde_json::to_string(&memory.concept_ids)?;
        let now = Utc::now();

        let layer_db = match memory.layer {
            MemoryLayer::LTM => "LTM",
            MemoryLayer::STM => "STM",
        };

        // Auto-set status to "updated" when content changes
        let status = if memory.status == MemoryStatus::Active {
            MemoryStatus::Updated
        } else {
            memory.status
        };

        let rows = self.conn.execute(
            "UPDATE memories SET layer=?1, topic=?2, summary=?3, content=?4, keywords=?5,
             importance=?6, source=?7, strength=?8, decay_lambda=?9, access_count=?10,
             superseded_by=?11, related_ids=?12, concept_ids=?13, status=?14, updated_at=?15
             WHERE id=?16",
            rusqlite::params![
                layer_db,
                memory.topic,
                memory.summary,
                memory.content,
                keywords_json,
                memory.importance.as_str(),
                memory.source.as_str(),
                memory.strength,
                memory.decay_lambda,
                memory.access_count,
                memory.superseded_by,
                related_ids_json,
                concept_ids_json,
                status.to_string(),
                now.to_rfc3339(),
                memory.id,
            ],
        )?;

        if rows == 0 {
            return Err(ReinError::NotFound(format!(
                "memory {} not found",
                memory.id
            )));
        }

        if let Some(ref emb) = memory.embedding {
            vec::insert_embedding(&self.conn, &memory.id, emb)?;
        }

        // Update side indexes (reuse keywords_json from above)
        self.update_tantivy(&memory.id, &memory.topic, &memory.summary, &memory.content, &keywords_json);
        self.update_hnsw(&memory.id, memory.embedding.as_deref());

        Ok(())
    }

    fn delete(&self, id: &str) -> ReinResult<()> {
        let rows = self
            .conn
            .execute("DELETE FROM memories WHERE id = ?1", rusqlite::params![id])?;
        if rows == 0 {
            return Err(ReinError::NotFound(format!("memory {id} not found")));
        }
        // FTS cleanup handled by trigger; clean up vector table
        if let Err(e) = vec::delete_embedding(&self.conn, id) {
            tracing::warn!("failed to delete embedding for {id}: {e}");
        }

        // Remove from side indexes (fire-and-forget)
        self.remove_from_tantivy(id);
        self.remove_from_hnsw(id);

        Ok(())
    }

    fn search_fts(
        &self,
        query: &str,
        topic: Option<&str>,
        limit: usize,
    ) -> ReinResult<Vec<Memory>> {
        let results = fts::search_fts(&self.conn, query, topic, limit)?;
        Ok(results.into_iter().map(|(m, _)| m).collect())
    }

    fn search_vec(
        &self,
        embedding: &[f32],
        topic: Option<&str>,
        limit: usize,
    ) -> ReinResult<Vec<Memory>> {
        // Fetch more than needed to compensate for topic filtering
        let fetch_limit = if topic.is_some() { limit * 3 } else { limit };
        let results = vec::search_vec(&self.conn, embedding, fetch_limit)?;
        let mut memories = Vec::new();
        for (id, _distance) in results {
            match self.get(&id) {
                Ok(m) => {
                    if let Some(t) = topic {
                        if m.topic == t {
                            memories.push(m);
                        }
                    } else {
                        memories.push(m);
                    }
                }
                Err(ReinError::NotFound(_)) => continue,
                Err(e) => return Err(e),
            }
        }
        memories.truncate(limit);
        Ok(memories)
    }

    fn apply_decay(&self) -> ReinResult<u64> {
        // Check if decay was run recently
        let last_decay: Option<String> = self
            .conn
            .query_row(
                "SELECT value FROM metadata WHERE key = 'last_decay_at'",
                [],
                |row| row.get(0),
            )
            .ok();

        if let Some(ref last) = last_decay {
            if let Ok(dt) = DateTime::parse_from_rfc3339(last) {
                let hours = (Utc::now() - dt.with_timezone(&Utc)).num_hours();
                if hours < 24 {
                    return Ok(0);
                }
            }
        }

        let now = Utc::now();

        // Fetch all non-critical memories
        let mut stmt = self.conn.prepare(
            "SELECT id, layer, decay_lambda, access_count, last_accessed, strength
             FROM memories WHERE importance != 'critical'",
        )?;

        struct DecayRow {
            id: String,
            new_strength: f64,
        }

        let updates: Vec<DecayRow> = stmt
            .query_map([], |row| {
                let id: String = row.get(0)?;
                let layer_str: String = row.get(1)?;
                let decay_lambda: f64 = row.get(2)?;
                let access_count: u32 = row.get(3)?;
                let last_accessed_str: String = row.get(4)?;
                let _strength: f64 = row.get(5)?;

                Ok((id, layer_str, decay_lambda, access_count, last_accessed_str))
            })?
            .filter_map(|r| match r {
                Ok(v) => Some(v),
                Err(e) => {
                    tracing::warn!("failed to deserialize memory row: {e}");
                    None
                }
            })
            .filter_map(|(id, layer_str, decay_lambda, access_count, last_accessed_str)| {
                let last_accessed = DateTime::parse_from_rfc3339(&last_accessed_str).ok()?;
                let days = (now - last_accessed.with_timezone(&Utc)).num_seconds() as f64 / 86400.0;
                if days <= 0.0 {
                    return None;
                }

                let lambda_eff = decay_lambda / (1.0 + access_count as f64 * 0.2);
                let layer = MemoryLayer::from_str(&layer_str).ok()?;
                let beta = layer.beta();
                let new_strength = (-lambda_eff * days.powf(beta)).exp();

                Some(DecayRow {
                    id,
                    new_strength,
                })
            })
            .collect();

        let count = updates.len() as u64;
        // Use SAVEPOINT instead of BEGIN TRANSACTION so this can nest inside
        // an outer SAVEPOINT (e.g., GC dry-run preview).
        self.conn.execute_batch("SAVEPOINT decay_batch")?;
        for u in &updates {
            self.conn.execute(
                "UPDATE memories SET strength = ?1 WHERE id = ?2",
                rusqlite::params![u.new_strength, u.id],
            )?;
        }
        // Record last decay time inside the savepoint (not after)
        self.conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES ('last_decay_at', ?1)",
            rusqlite::params![now.to_rfc3339()],
        )?;
        self.conn.execute_batch("RELEASE decay_batch")?;

        Ok(count)
    }

    fn prune(&self, threshold: f64) -> ReinResult<u64> {
        let mem_pruned = self.prune_memories_only(threshold, false)?;
        let concept_pruned = self.prune_low_quality_concepts().unwrap_or(0);
        if concept_pruned > 0 {
            tracing::info!("pruned {concept_pruned} low-quality concepts");
        }
        Ok(mem_pruned + concept_pruned)
    }

    fn list_topics(&self) -> ReinResult<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT topic FROM memories GROUP BY topic ORDER BY COUNT(*) DESC",
        )?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    fn consolidate(&self, topic: &str) -> ReinResult<Vec<Memory>> {
        // Use SAVEPOINT for nesting safety (may be called within an existing transaction)
        self.conn.execute_batch("SAVEPOINT consolidate")?;

        // Collect all memories for the topic
        let memories: Vec<Memory> = {
            let mut stmt = self
                .conn
                .prepare("SELECT * FROM memories WHERE topic = ?1")?;
            let rows = stmt.query_map(rusqlite::params![topic], |row| {
                row_to_memory(row).map_err(|e| {
                    rusqlite::Error::FromSqlConversionFailure(
                        0,
                        rusqlite::types::Type::Text,
                        Box::new(e),
                    )
                })
            })?;
            rows.collect::<Result<Vec<_>, _>>()?
        };

        // Delete all memories for the topic
        if let Err(e) = self.conn.execute(
            "DELETE FROM memories WHERE topic = ?1",
            rusqlite::params![topic],
        ) {
            let _ = self.conn.execute_batch("ROLLBACK TO consolidate; RELEASE consolidate");
            return Err(e.into());
        }

        // Clean side indexes for deleted memories
        for m in &memories {
            self.remove_from_tantivy(&m.id);
            self.remove_from_hnsw(&m.id);
        }

        self.conn.execute_batch("RELEASE consolidate")?;
        Ok(memories)
    }

    fn stats(&self) -> ReinResult<StoreStats> {
        let total_memories: usize = self
            .conn
            .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))?;
        let ltm_count: usize = self.conn.query_row(
            "SELECT COUNT(*) FROM memories WHERE layer = 'LTM'",
            [],
            |row| row.get(0),
        )?;
        let stm_count: usize = self.conn.query_row(
            "SELECT COUNT(*) FROM memories WHERE layer = 'STM'",
            [],
            |row| row.get(0),
        )?;
        let topic_count: usize = self.conn.query_row(
            "SELECT COUNT(DISTINCT topic) FROM memories",
            [],
            |row| row.get(0),
        )?;
        let avg_strength: f64 = self
            .conn
            .query_row(
                "SELECT COALESCE(AVG(strength), 0.0) FROM memories",
                [],
                |row| row.get(0),
            )?;

        let memoir_count: usize = self.conn
            .query_row("SELECT COUNT(*) FROM memoirs", [], |row| row.get(0))
            .unwrap_or(0);
        let concept_count: usize = self.conn
            .query_row("SELECT COUNT(*) FROM concepts", [], |row| row.get(0))
            .unwrap_or(0);
        let link_count: usize = self.conn
            .query_row("SELECT COUNT(*) FROM concept_links", [], |row| row.get(0))
            .unwrap_or(0);

        Ok(StoreStats {
            total_memories,
            ltm_count,
            stm_count,
            topic_count,
            avg_strength,
            memoir_count,
            concept_count,
            link_count,
        })
    }

    fn health(&self, topic: Option<&str>) -> ReinResult<Vec<HealthReport>> {
        fn parse_health_row(row: &rusqlite::Row) -> rusqlite::Result<HealthReport> {
            let topic: String = row.get(0)?;
            let count: usize = row.get(1)?;
            let avg_strength: f64 = row.get(2)?;
            let stale_count: usize = row.get(3)?;
            Ok(HealthReport {
                topic,
                count,
                avg_strength,
                stale_count,
                needs_consolidation: count > 10,
            })
        }

        let reports = if let Some(t) = topic {
            let mut stmt = self.conn.prepare(
                "SELECT topic, COUNT(*) as cnt, AVG(strength) as avg_s,
                        SUM(CASE WHEN strength < 0.3 THEN 1 ELSE 0 END) as stale
                 FROM memories WHERE topic = ?1
                 GROUP BY topic",
            )?;
            let rows = stmt.query_map(rusqlite::params![t], parse_health_row)?;
            rows.collect::<Result<Vec<_>, _>>()?
        } else {
            let mut stmt = self.conn.prepare(
                "SELECT topic, COUNT(*) as cnt, AVG(strength) as avg_s,
                        SUM(CASE WHEN strength < 0.3 THEN 1 ELSE 0 END) as stale
                 FROM memories
                 GROUP BY topic
                 ORDER BY cnt DESC",
            )?;
            let rows = stmt.query_map([], parse_health_row)?;
            rows.collect::<Result<Vec<_>, _>>()?
        };

        Ok(reports)
    }
}

impl SqliteStore {
    /// SQL-only DELETE for dry-run preview: removes from SQLite but does NOT
    /// touch Tantivy/HNSW side indexes. Intended to run inside a SAVEPOINT
    /// that will be rolled back, so concept evaluation sees the correct state.
    pub(crate) fn prune_memories_sql_only(&self, threshold: f64) -> ReinResult<u64> {
        let rows = self.conn.execute(
            "DELETE FROM memories WHERE layer = 'STM' AND strength < ?1
             AND importance NOT IN ('critical', 'high')",
            rusqlite::params![threshold],
        )?;
        Ok(rows as u64)
    }

    /// Prune weak STM memories only (without concept pruning).
    /// Used by ops::run_gc() to separate memory and concept pruning counts.
    /// When `preview` is true, only counts candidates without deleting or touching side indexes.
    pub(crate) fn prune_memories_only(&self, threshold: f64, preview: bool) -> ReinResult<u64> {
        if preview {
            let count: u64 = self.conn.query_row(
                "SELECT COUNT(*) FROM memories WHERE layer = 'STM' AND strength < ?1
                 AND importance NOT IN ('critical', 'high')",
                rusqlite::params![threshold],
                |row| row.get(0),
            )?;
            return Ok(count);
        }

        let mut stmt = self.conn.prepare(
            "SELECT id FROM memories WHERE layer = 'STM' AND strength < ?1
             AND importance NOT IN ('critical', 'high')",
        )?;
        let ids: Vec<String> = stmt.query_map(rusqlite::params![threshold], |row| row.get(0))?
            .filter_map(|r| r.ok())
            .collect();
        drop(stmt);

        let rows = self.conn.execute(
            "DELETE FROM memories WHERE layer = 'STM' AND strength < ?1
             AND importance NOT IN ('critical', 'high')",
            rusqlite::params![threshold],
        )?;

        for id in &ids {
            self.remove_from_tantivy(id);
            self.remove_from_hnsw(id);
        }

        Ok(rows as u64)
    }

    /// Store a memory with deduplication logic.
    ///
    /// Checks for existing similar memories using FTS and Jaccard similarity.
    /// Uses `BEGIN IMMEDIATE` to acquire a write lock upfront, preventing concurrent
    /// requests from both seeing stale state and double-inserting.
    /// - If a similar memory exists within the time window, merges content into it.
    /// - If a similar memory exists but is older, supersedes it with the new memory.
    /// - Otherwise, creates a new memory.
    pub fn store_with_dedup(
        &self,
        memory: Memory,
        similarity_threshold: f32,
        time_window_days: i64,
    ) -> ReinResult<String> {
        // First pass: run check_dedup with caller's threshold to find best candidate
        let dedup_action = crate::extract::check_dedup(
            self, &memory.topic, &memory.content, similarity_threshold, time_window_days,
        )?;

        // Derive cluster from the matched candidate (if any) for adaptive threshold.
        // This is more accurate than picking a random topic neighbor.
        let candidate_cluster = match &dedup_action {
            DedupAction::MergeInto(id) | DedupAction::Supersede(id) | DedupAction::GrayZone(id, _) => {
                self.get(id).ok().and_then(|m| m.cluster_id)
            }
            DedupAction::CreateNew => memory.cluster_id,
        };

        // Re-check with adaptive threshold if it differs from caller's threshold
        let dedup_action = if let Some(state) = crate::store::adaptive::AdaptiveState::restore_snapshot(&self.conn) {
            let effective = state.get_dedup_threshold(candidate_cluster);
            if (effective - similarity_threshold).abs() > 0.01 {
                // Threshold changed — re-run dedup with adaptive value
                crate::extract::check_dedup(
                    self, &memory.topic, &memory.content, effective, time_window_days,
                )?
            } else {
                dedup_action
            }
        } else {
            dedup_action
        };
        let resolved_action = match dedup_action {
            DedupAction::GrayZone(ref candidate_id, sim) => {
                // LLM semantic check outside transaction
                let is_dup = if let Ok(existing) = self.get(candidate_id) {
                    let new_content = memory.content.clone();
                    let old_content = existing.content.clone();
                    std::panic::catch_unwind(|| {
                        tokio::task::block_in_place(|| {
                            tokio::runtime::Handle::current().block_on(async {
                                let config = crate::config::ReinConfig::load().unwrap_or_default();
                                crate::extract::llm::llm_is_duplicate(&config, &new_content, &old_content).await
                            })
                        })
                    }).unwrap_or(false)
                } else {
                    false
                };
                if is_dup {
                    tracing::info!("LLM confirmed duplicate (sim={sim:.2}), merging into {candidate_id}");
                    DedupAction::MergeInto(candidate_id.clone())
                } else {
                    tracing::info!("LLM says not duplicate (sim={sim:.2}), creating new");
                    DedupAction::CreateNew
                }
            }
            other => other,
        };

        self.conn.execute_batch("BEGIN IMMEDIATE")?;
        let result = self.store_with_dedup_resolved(memory, resolved_action);
        match &result {
            Ok(_) => { self.conn.execute_batch("COMMIT")?; }
            Err(_) => { let _ = self.conn.execute_batch("ROLLBACK"); }
        }
        result
    }

    /// Execute a pre-resolved dedup action within a BEGIN IMMEDIATE transaction.
    fn store_with_dedup_resolved(
        &self,
        memory: Memory,
        action: DedupAction,
    ) -> ReinResult<String> {
        match action {
            DedupAction::CreateNew => {
                let id = self.store(memory)?;
                // Mark for deferred embedding-based dedup in GC slow channel
                let _ = self.conn.execute(
                    "UPDATE memories SET needs_vec_dedup = 1 WHERE id = ?1",
                    rusqlite::params![&id],
                );
                Ok(id)
            }
            DedupAction::MergeInto(existing_id) => {
                // Provenance-preserving merge: extract unique lines from new content
                // and append with temporal marker, rather than blind concatenation
                if let Ok(mut existing) = self.get(&existing_id) {
                    let unique = crate::ops::extract_unique_lines(&memory.content, &existing.content);
                    if !unique.is_empty() {
                        existing.content.push_str(&format!(
                            "\n\n[merged on {}]\n{}",
                            chrono::Utc::now().format("%Y-%m-%d"), unique
                        ));
                    }
                    // Cap content length to prevent unbounded growth from repeated merges.
                    // Trim older content (front) to preserve newly merged tail.
                    if existing.content.len() > 10_000 {
                        let excess = existing.content.len() - 10_000;
                        // Find safe UTF-8 boundary at or after the trim point
                        let mut trim_at = excess;
                        while trim_at < existing.content.len() && !existing.content.is_char_boundary(trim_at) {
                            trim_at += 1;
                        }
                        existing.content = existing.content[trim_at..].to_string();
                    }
                    existing.summary = existing.content.chars().take(100).collect();
                    // Merge keywords (deduplicated)
                    for kw in &memory.keywords {
                        if !existing.keywords.contains(kw) {
                            existing.keywords.push(kw.clone());
                        }
                    }
                    // Upgrade importance if new memory is more important
                    if memory.importance > existing.importance {
                        existing.importance = memory.importance;
                        existing.layer = existing.importance.auto_layer();
                        existing.decay_lambda = 0.06 * existing.importance.decay_factor();
                    }
                    existing.strength = (existing.strength + 0.2).min(1.0);
                    existing.updated_at = chrono::Utc::now();
                    self.update(&existing)?;
                    Ok(existing_id)
                } else {
                    self.store(memory)
                }
            }
            DedupAction::Supersede(old_id) => {
                let new_id = self.store(memory)?;
                // Mark old memory as superseded
                self.mark_superseded(&old_id, &new_id)?;
                // Mark new memory for deferred embedding-based dedup
                let _ = self.conn.execute(
                    "UPDATE memories SET needs_vec_dedup = 1 WHERE id = ?1",
                    rusqlite::params![&new_id],
                );
                Ok(new_id)
            }
            DedupAction::GrayZone(_, _) => {
                // GrayZone is always resolved before calling this function
                // (see store_with_dedup which pre-resolves via LLM outside the transaction).
                // Treat as CreateNew as a safe fallback.
                tracing::warn!("unexpected GrayZone in store_with_dedup_resolved, creating new");
                self.store(memory)
            }
        }
    }

    /// Path for the Tantivy FTS index directory (scoped to DB file).
    /// e.g., `~/.rein/memories.db` → `~/.rein/memories.tantivy`
    fn tantivy_path(&self) -> PathBuf {
        self.db_path.with_extension("tantivy")
    }

    /// Path for the HNSW index directory.
    fn hnsw_path(&self) -> PathBuf {
        self.db_path.with_extension("")
    }

    /// Get or lazily initialize cached Tantivy instance (avoids 15MB IndexWriter alloc per op).
    fn with_tantivy<F>(&self, f: F) where F: FnOnce(&super::tantivy_fts::TantivyFts) {
        if self.db_path.to_str() == Some(":memory:") { return; }
        let mut cache = self.tantivy_cache.borrow_mut();
        if cache.is_none() {
            *cache = super::tantivy_fts::TantivyFts::open(&self.tantivy_path()).ok();
        }
        if let Some(ref tantivy) = *cache {
            f(tantivy);
        }
    }

    /// Fire-and-forget: update Tantivy index after a write.
    fn update_tantivy(&self, id: &str, topic: &str, summary: &str, content: &str, keywords: &str) {
        self.with_tantivy(|t| { let _ = t.insert(id, topic, summary, content, keywords); });
    }

    /// Acquire an exclusive file lock on the HNSW index, open it, run `f`, then save.
    /// Prevents concurrent HNSW writes from overwriting each other.
    fn with_hnsw_lock<F, R>(&self, dims: usize, f: F) -> Option<R>
    where
        F: FnOnce(&mut crate::store::hnsw::HnswIndex) -> R,
    {
        let hnsw_path = self.hnsw_path();
        if self.db_path.to_str() == Some(":memory:") {
            return None;
        }

        let lock_path = hnsw_path.with_extension("usearch.lock");
        let lock_file = std::fs::File::create(&lock_path).ok()?;
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            let rc = unsafe { libc::flock(lock_file.as_raw_fd(), libc::LOCK_EX) };
            if rc != 0 {
                tracing::warn!("HNSW flock(LOCK_EX) failed: {}", std::io::Error::last_os_error());
                return None;
            }
        }

        let mut index = crate::store::hnsw::HnswIndex::open(&hnsw_path, dims).ok()?;
        let result = f(&mut index);
        if let Err(e) = index.save() {
            tracing::warn!("HNSW index save failed: {e}");
        }

        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            let _ = unsafe { libc::flock(lock_file.as_raw_fd(), libc::LOCK_UN) };
        }
        drop(lock_file);

        Some(result)
    }

    /// Fire-and-forget: update HNSW index after a write (if embedding available).
    fn update_hnsw(&self, id: &str, embedding: Option<&[f32]>) {
        if let Some(emb) = embedding {
            self.with_hnsw_lock(emb.len(), |index| {
                let _ = index.insert(id, emb);
            });
        }
    }

    /// Fire-and-forget: remove from Tantivy index after a delete.
    pub(crate) fn remove_from_tantivy(&self, id: &str) {
        self.with_tantivy(|t| { let _ = t.delete(id); });
    }

    /// Fire-and-forget: remove from HNSW index after a delete.
    pub(crate) fn remove_from_hnsw(&self, id: &str) {
        self.with_hnsw_lock(self.dims, |index| {
            let _ = index.delete(id);
        });
    }

    /// Mark an old memory as superseded by a new one.
    pub fn mark_superseded(&self, old_id: &str, new_id: &str) -> ReinResult<()> {
        let rows = self.conn.execute(
            "UPDATE memories SET superseded_by = ?1 WHERE id = ?2",
            rusqlite::params![new_id, old_id],
        )?;
        if rows == 0 {
            return Err(ReinError::NotFound(format!("memory {old_id} not found")));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Importance, Source};

    fn test_memory(topic: &str, summary: &str, importance: Importance) -> Memory {
        Memory {
            id: ulid::Ulid::new().to_string(),
            layer: importance.auto_layer(),
            topic: topic.to_string(),
            summary: summary.to_string(),
            content: format!("Content for {summary}"),
            keywords: vec![],
            importance,
            source: Source::Manual,
            strength: 1.0,
            decay_lambda: 0.06 * importance.decay_factor(),
            access_count: 0,
            superseded_by: None,
            related_ids: vec![],
            concept_ids: vec![],
            status: MemoryStatus::default(),
            embedding: None,
            tier: "warm".to_string(),
            cluster_id: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_accessed: Utc::now(),
        }
    }

    #[test]
    fn test_store_and_get() {
        let store = SqliteStore::in_memory().unwrap();
        let mem = test_memory("rust", "ownership rules", Importance::High);
        let original_summary = mem.summary.clone();
        let original_topic = mem.topic.clone();

        let id = store.store(mem).unwrap();
        let fetched = store.get(&id).unwrap();

        assert_eq!(fetched.id, id);
        assert_eq!(fetched.summary, original_summary);
        assert_eq!(fetched.topic, original_topic);
        assert_eq!(fetched.layer, MemoryLayer::LTM);
        assert_eq!(fetched.importance, Importance::High);
        assert_eq!(fetched.access_count, 0); // get() is now read-only, no side effects
    }

    #[test]
    fn test_delete() {
        let store = SqliteStore::in_memory().unwrap();
        let mem = test_memory("rust", "borrow checker", Importance::Medium);
        let id = store.store(mem).unwrap();

        store.delete(&id).unwrap();
        let result = store.get(&id);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ReinError::NotFound(_)));
    }

    #[test]
    fn test_update() {
        let store = SqliteStore::in_memory().unwrap();
        let mem = test_memory("rust", "lifetimes", Importance::Medium);
        let id = store.store(mem).unwrap();

        let mut fetched = store.get(&id).unwrap();
        fetched.content = "Updated content about lifetimes".to_string();
        store.update(&fetched).unwrap();

        let updated = store.get(&id).unwrap();
        assert_eq!(updated.content, "Updated content about lifetimes");
    }

    #[test]
    fn test_list_topics() {
        let store = SqliteStore::in_memory().unwrap();
        store.store(test_memory("rust", "ownership", Importance::High)).unwrap();
        store.store(test_memory("rust", "borrowing", Importance::Medium)).unwrap();
        store.store(test_memory("python", "decorators", Importance::Low)).unwrap();

        let topics = store.list_topics().unwrap();
        assert_eq!(topics.len(), 2);
        // rust has 2 entries, should come first
        assert_eq!(topics[0], "rust");
        assert_eq!(topics[1], "python");
    }

    #[test]
    fn test_fts_search() {
        let store = SqliteStore::in_memory().unwrap();
        store.store(test_memory("rust", "ownership rules", Importance::High)).unwrap();
        store.store(test_memory("rust", "borrow checker", Importance::Medium)).unwrap();
        store.store(test_memory("python", "decorators", Importance::Low)).unwrap();

        let results = store.search_fts("ownership", None, 10).unwrap();
        assert!(!results.is_empty());
        assert!(results.iter().any(|m| m.summary.contains("ownership")));
    }

    #[test]
    fn test_fts_sanitize() {
        use crate::store::fts::sanitize_fts_query;
        let result = sanitize_fts_query("hello* -world (test)");
        assert!(!result.contains('*'));
        assert!(!result.contains('-'));
        assert!(!result.contains('('));
        assert!(!result.contains(')'));
        // Each token should be quoted
        assert!(result.contains("\"hello\""));
        assert!(result.contains("\"world\""));
        assert!(result.contains("\"test\""));
    }

    #[test]
    fn test_fts_injection() {
        let store = SqliteStore::in_memory().unwrap();
        store.store(test_memory("test", "normal memory", Importance::Low)).unwrap();

        // Should not crash on malicious input
        let result = store.search_fts("\" OR 1=1; DROP TABLE memories; --", None, 10);
        assert!(result.is_ok());

        let result = store.search_fts("***^^^\"\"\"", None, 10);
        assert!(result.is_ok());
    }

    #[test]
    fn test_apply_decay() {
        let store = SqliteStore::in_memory().unwrap();

        // Store memories with different importance
        let mut critical = test_memory("test", "critical info", Importance::Critical);
        critical.created_at = Utc::now() - chrono::Duration::days(30);
        store.store(critical).unwrap();

        let mut medium = test_memory("test", "medium info", Importance::Medium);
        medium.created_at = Utc::now() - chrono::Duration::days(30);
        let med_id = store.store(medium).unwrap();

        let mut low = test_memory("test", "low info", Importance::Low);
        low.created_at = Utc::now() - chrono::Duration::days(30);
        let low_id = store.store(low).unwrap();

        // Manually set created_at and last_accessed in the past so decay has effect
        let past = (Utc::now() - chrono::Duration::days(30)).to_rfc3339();
        store.conn.execute(
            "UPDATE memories SET created_at = ?1, last_accessed = ?1 WHERE id IN (?2, ?3)",
            rusqlite::params![past, med_id, low_id],
        ).unwrap();

        let count = store.apply_decay().unwrap();
        assert!(count > 0);

        // Verify strength was reduced for non-critical
        let med = store.get(&med_id).unwrap();
        assert!(med.strength < 1.0, "Medium memory strength should decay");

        let low_mem = store.get(&low_id).unwrap();
        assert!(low_mem.strength < 1.0, "Low memory strength should decay");
    }

    #[test]
    fn test_prune() {
        let store = SqliteStore::in_memory().unwrap();

        // STM + Low importance + low strength -> should be pruned
        let id_low = store.store(test_memory("test", "forgettable", Importance::Low)).unwrap();
        store.conn.execute(
            "UPDATE memories SET strength = 0.05 WHERE id = ?1",
            rusqlite::params![id_low],
        ).unwrap();

        // STM + Medium importance + low strength -> should be pruned
        let id_med = store.store(test_memory("test", "somewhat forgettable", Importance::Medium)).unwrap();
        store.conn.execute(
            "UPDATE memories SET strength = 0.05 WHERE id = ?1",
            rusqlite::params![id_med],
        ).unwrap();

        // LTM + Critical importance + low strength -> should NOT be pruned
        let id_crit = store.store(test_memory("test", "critical never forget", Importance::Critical)).unwrap();
        store.conn.execute(
            "UPDATE memories SET strength = 0.05 WHERE id = ?1",
            rusqlite::params![id_crit],
        ).unwrap();

        // LTM + High importance + low strength -> should NOT be pruned (importance=high)
        let id_high = store.store(test_memory("test", "important stuff", Importance::High)).unwrap();
        store.conn.execute(
            "UPDATE memories SET strength = 0.05 WHERE id = ?1",
            rusqlite::params![id_high],
        ).unwrap();

        let pruned = store.prune(0.1).unwrap();
        assert_eq!(pruned, 2); // low and medium STM

        // Critical and High should still exist
        assert!(store.get(&id_crit).is_ok());
        assert!(store.get(&id_high).is_ok());

        // Low and Medium should be gone
        assert!(store.get(&id_low).is_err());
        assert!(store.get(&id_med).is_err());
    }

    fn test_memory_with_content(topic: &str, summary: &str, content: &str, importance: Importance) -> Memory {
        Memory {
            id: ulid::Ulid::new().to_string(),
            layer: importance.auto_layer(),
            topic: topic.to_string(),
            summary: summary.to_string(),
            content: content.to_string(),
            keywords: vec![],
            importance,
            source: Source::Manual,
            strength: 1.0,
            decay_lambda: 0.06 * importance.decay_factor(),
            access_count: 0,
            superseded_by: None,
            related_ids: vec![],
            concept_ids: vec![],
            status: MemoryStatus::default(),
            embedding: None,
            tier: "warm".to_string(),
            cluster_id: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_accessed: Utc::now(),
        }
    }

    #[test]
    fn test_store_with_dedup_create() {
        let store = SqliteStore::in_memory().unwrap();
        let mem1 = test_memory_with_content(
            "rust",
            "ownership rules",
            "Rust ownership rules are fundamental to memory safety",
            Importance::High,
        );
        let id1 = store.store_with_dedup(mem1, 0.85, 7).unwrap();

        let mem2 = test_memory_with_content(
            "rust",
            "async programming",
            "Async programming in Rust uses futures and tokio runtime",
            Importance::Medium,
        );
        let id2 = store.store_with_dedup(mem2, 0.85, 7).unwrap();

        // Both should exist as separate memories (different content)
        assert_ne!(id1, id2);
        let stats = store.stats().unwrap();
        assert_eq!(stats.total_memories, 2);
    }

    #[test]
    fn test_store_with_dedup_merge() {
        let store = SqliteStore::in_memory().unwrap();

        let mem1 = test_memory_with_content(
            "rust",
            "ownership rules",
            "Rust ownership rules are fundamental to memory safety in systems programming",
            Importance::High,
        );
        let id1 = store.store_with_dedup(mem1, 0.85, 7).unwrap();

        // Store very similar content (same words, minor addition)
        let mem2 = test_memory_with_content(
            "rust",
            "ownership rules updated",
            "Rust ownership rules are fundamental to memory safety in Rust systems programming",
            Importance::High,
        );
        let id2 = store.store_with_dedup(mem2, 0.85, 7).unwrap();

        // Should merge into existing (same id returned)
        assert_eq!(id1, id2);

        // Only one memory should exist
        let stats = store.stats().unwrap();
        assert_eq!(stats.total_memories, 1);

        // Content should be merged
        let merged = store.get(&id1).unwrap();
        assert!(merged.content.contains("\n\n"));
    }

    #[test]
    fn test_store_with_dedup_supersede() {
        let store = SqliteStore::in_memory().unwrap();

        let mem1 = test_memory_with_content(
            "rust",
            "ownership rules",
            "Rust ownership rules are fundamental to memory safety in systems programming",
            Importance::High,
        );
        let id1 = store.store_with_dedup(mem1, 0.85, 7).unwrap();

        // Manually set created_at to 10 days ago
        let old_date = (Utc::now() - chrono::Duration::days(10)).to_rfc3339();
        store.conn.execute(
            "UPDATE memories SET created_at = ?1 WHERE id = ?2",
            rusqlite::params![old_date, id1],
        ).unwrap();

        // Store very similar content
        let mem2 = test_memory_with_content(
            "rust",
            "ownership rules updated",
            "Rust ownership rules are fundamental to memory safety in Rust systems programming",
            Importance::High,
        );
        let id2 = store.store_with_dedup(mem2, 0.85, 7).unwrap();

        // Should supersede (different id)
        assert_ne!(id1, id2);

        // Two memories should exist
        let stats = store.stats().unwrap();
        assert_eq!(stats.total_memories, 2);

        // Old memory should have superseded_by set
        let old = store.get(&id1).unwrap();
        assert_eq!(old.superseded_by, Some(id2.clone()));
    }

    #[test]
    fn test_store_with_dedup_sets_needs_vec_dedup() {
        let store = SqliteStore::in_memory().unwrap();
        let mem = test_memory_with_content(
            "test",
            "unique content",
            "This is completely unique content that has no duplicates anywhere",
            Importance::Medium,
        );
        let id = store.store_with_dedup(mem, 0.85, 7).unwrap();

        // New memories created via dedup should be flagged for vec dedup
        let flag: i32 = store.conn.query_row(
            "SELECT needs_vec_dedup FROM memories WHERE id = ?1",
            rusqlite::params![&id],
            |row| row.get(0),
        ).unwrap();
        assert_eq!(flag, 1, "new memory should have needs_vec_dedup = 1");
    }

    #[test]
    fn test_needs_vec_dedup_schema() {
        // Verify the needs_vec_dedup column exists and defaults to 0 for direct store
        let store = SqliteStore::in_memory().unwrap();
        let mem = test_memory("test", "direct store", Importance::Medium);
        let id = store.store(mem).unwrap();

        let flag: i32 = store.conn.query_row(
            "SELECT needs_vec_dedup FROM memories WHERE id = ?1",
            rusqlite::params![&id],
            |row| row.get(0),
        ).unwrap();
        // Direct store (not via dedup) defaults to 0
        assert_eq!(flag, 0, "direct store should have needs_vec_dedup = 0");
    }

    #[test]
    fn test_stm_to_ltm_promotion() {
        let store = SqliteStore::in_memory().unwrap();
        let mem = test_memory("test", "frequently accessed", Importance::Medium);
        // Medium → STM
        assert_eq!(mem.layer, MemoryLayer::STM);
        let id = store.store(mem).unwrap();

        // Access 6 times
        for _ in 0..6 {
            store.record_access(&id).unwrap();
        }

        // Should be promoted to LTM
        let fetched = store.get(&id).unwrap();
        assert_eq!(fetched.layer, MemoryLayer::LTM);
        assert_eq!(fetched.access_count, 6);
    }
}
