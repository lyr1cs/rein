use chrono::{DateTime, Utc};
use rusqlite::Connection;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use crate::extract::intelligent_merge::{InsertionVerdict, VerdictResult};
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

pub(crate) const MEMORY_SELECT_COLUMNS: &str = "m.id, m.layer, m.topic, m.summary, m.content, \
    m.keywords, m.importance, m.source, m.strength, m.decay_lambda, m.access_count, \
    m.superseded_by, COALESCE(cs.canonical_id, m.id) AS canonical_id, \
    COALESCE(cs.support_count, 1) AS support_count, COALESCE(cs.merge_count, 0) AS merge_count, \
    COALESCE(cs.dedup_confidence, 1.0) AS dedup_confidence, \
    COALESCE(cs.source_diversity, 1.0) AS source_diversity, \
    COALESCE(cs.contradiction_score, 0.0) AS contradiction_score, \
    m.related_ids, m.concept_ids, m.status, m.tier, m.cluster_id, m.created_at, m.updated_at, m.last_accessed";

pub(crate) fn memory_select_base() -> String {
    format!(
        "SELECT {MEMORY_SELECT_COLUMNS} FROM memories m \
         LEFT JOIN memory_canonical_state cs ON cs.memory_id = m.id"
    )
}

impl SqliteStore {
    fn infer_cluster_from_cache(&self, memory: &Memory) -> Option<u32> {
        let config = crate::config::ReinConfig::load().ok()?;
        let model = config.embedding_model();
        let enriched =
            crate::embed::prepend_metadata(&memory.topic, &memory.summary, &memory.content);
        let embedding = crate::embed::EmbedCache::get(self.conn(), &enriched, &model)
            .ok()
            .flatten()?;

        // Prefer centroid-based assignment (accurate, no N+1 queries).
        // Pass embedding dims so stale centroids from a prior model are automatically rejected.
        let centroids =
            crate::store::adaptive::load_cluster_centroids(self.conn(), embedding.len());
        if !centroids.is_empty() {
            return crate::store::adaptive::assign_to_nearest_centroid(&centroids, &embedding);
        }

        // Fallback: nearest-neighbor heuristic (used before first full HDBSCAN run).
        crate::store::vec::search_vec(self.conn(), &embedding, 8)
            .ok()?
            .into_iter()
            .find_map(|(id, _)| self.get(&id).ok().and_then(|m| m.cluster_id))
    }

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
            eprintln!(
                "rein: Existing vectors are incompatible. Run 'rein migrate --reindex' to rebuild."
            );
            eprintln!("rein: FTS search still works. Vector search may return incorrect results.");
        }

        Ok(Self {
            conn,
            db_path: path.to_path_buf(),
            dims,
            tantivy_cache: std::cell::RefCell::new(None),
        })
    }

    /// Create an in-memory database for testing (default 3072 dims).
    pub fn in_memory() -> ReinResult<Self> {
        schema::init_sqlite_vec();
        let conn = Connection::open_in_memory()?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;
        schema::init_schema(&conn, 3072)?;
        Ok(Self {
            conn,
            db_path: PathBuf::from(":memory:"),
            dims: 3072,
            tantivy_cache: std::cell::RefCell::new(None),
        })
    }
    /// Access the underlying SQLite connection (for direct queries).
    pub fn conn(&self) -> &Connection {
        &self.conn
    }

    /// The path to the database file (or ":memory:" for in-memory databases).
    pub fn db_path(&self) -> &Path {
        &self.db_path
    }

    pub fn collapse_to_canonicals(
        &self,
        memories: Vec<Memory>,
        limit: usize,
    ) -> ReinResult<Vec<Memory>> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for memory in memories {
            if memory.superseded_by.is_some()
                || !matches!(memory.status, MemoryStatus::Active | MemoryStatus::Updated)
            {
                continue;
            }
            let canonical_id = self
                .canonical_id_for(&memory.id)
                .unwrap_or_else(|_| memory.id.clone());
            if !seen.insert(canonical_id.clone()) {
                continue;
            }
            let canonical = if canonical_id == memory.id {
                memory
            } else {
                self.get(&canonical_id).unwrap_or(memory)
            };
            if canonical.superseded_by.is_none()
                && matches!(
                    canonical.status,
                    MemoryStatus::Active | MemoryStatus::Updated
                )
            {
                out.push(canonical);
            }
            if out.len() >= limit {
                break;
            }
        }
        Ok(out)
    }

    pub fn get_canonical(&self, id: &str) -> ReinResult<Memory> {
        let canonical_id = self.canonical_id_for(id)?;
        self.get(&canonical_id)
    }

    /// Get all memories in a topic (for dedup scanning).
    pub fn get_by_topic(&self, topic: &str) -> ReinResult<Vec<Memory>> {
        let sql = format!("{} WHERE m.topic = ?1", memory_select_base());
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(rusqlite::params![topic], |row| {
            row_to_memory(row).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    0,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })
        })?;
        Ok(rows
            .filter_map(|r| match r {
                Ok(m) => Some(m),
                Err(e) => {
                    tracing::warn!("failed to deserialize memory row: {e}");
                    None
                }
            })
            .collect())
    }

    /// Batch-get memories by IDs. Returns found memories (skips missing ones).
    /// More efficient than calling get() in a loop — single query with WHERE id IN (...).
    pub fn get_batch(&self, ids: &[String]) -> Vec<Memory> {
        if ids.is_empty() {
            return vec![];
        }
        // SQLite doesn't support array parameters, so build a parameterized IN clause
        let placeholders: Vec<String> = (1..=ids.len()).map(|i| format!("?{i}")).collect();
        let sql = format!(
            "{} WHERE m.id IN ({})",
            memory_select_base(),
            placeholders.join(",")
        );
        let params: Vec<&dyn rusqlite::types::ToSql> = ids
            .iter()
            .map(|id| id as &dyn rusqlite::types::ToSql)
            .collect();
        let mut stmt = match self.conn.prepare(&sql) {
            Ok(s) => s,
            Err(_) => return vec![],
        };
        let rows = match stmt.query_map(params.as_slice(), |row| {
            row_to_memory(row).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    0,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })
        }) {
            Ok(r) => r,
            Err(_) => return vec![],
        };
        rows.filter_map(|r| r.ok()).collect()
    }

    /// Get the most recently created memories.
    pub fn recent(&self, limit: usize) -> ReinResult<Vec<Memory>> {
        let sql = format!(
            "{} ORDER BY m.created_at DESC LIMIT ?1",
            memory_select_base()
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(rusqlite::params![limit], |row| {
            row_to_memory(row).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    0,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })
        })?;
        let memories: Vec<Memory> = rows
            .filter_map(|r| match r {
                Ok(m) => Some(m),
                Err(e) => {
                    tracing::warn!("failed to deserialize memory row: {e}");
                    None
                }
            })
            .collect();
        self.collapse_to_canonicals(memories, limit)
    }

    pub fn get_by_cluster(&self, cluster_id: u32, limit: usize) -> ReinResult<Vec<Memory>> {
        let sql = format!(
            "{} WHERE m.cluster_id = ?1 AND m.superseded_by IS NULL \
             AND m.status IN ('active', 'updated') \
             ORDER BY m.updated_at DESC LIMIT ?2",
            memory_select_base()
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(rusqlite::params![cluster_id, limit as i64], |row| {
            row_to_memory(row).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    0,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })
        })?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    pub fn add_memory_evidence(&self, evidence: MemoryEvidence) -> ReinResult<String> {
        let id = if evidence.id.is_empty() {
            ulid::Ulid::new().to_string()
        } else {
            evidence.id.clone()
        };
        self.conn.execute(
            "INSERT INTO memory_evidence (
                id, canonical_id, memory_id, source_topic, summary, content, keywords, source, created_at, imported_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            rusqlite::params![
                &id,
                evidence.canonical_id,
                evidence.memory_id,
                evidence.source_topic,
                evidence.summary,
                evidence.content,
                serde_json::to_string(&evidence.keywords)?,
                evidence.source.as_str(),
                evidence.created_at.to_rfc3339(),
                evidence.imported_at.to_rfc3339(),
            ],
        )?;
        Ok(id)
    }

    pub fn snapshot_memory_as_evidence(
        &self,
        canonical_id: &str,
        memory: &Memory,
    ) -> ReinResult<String> {
        let id = self.add_memory_evidence(MemoryEvidence {
            id: String::new(),
            canonical_id: canonical_id.to_string(),
            memory_id: Some(memory.id.clone()),
            source_topic: memory.topic.clone(),
            summary: memory.summary.clone(),
            content: memory.content.clone(),
            keywords: memory.keywords.clone(),
            source: memory.source,
            created_at: memory.created_at,
            imported_at: Utc::now(),
        })?;
        let _ = self.refresh_canonical_state(canonical_id);
        Ok(id)
    }

    pub fn list_memory_evidence(
        &self,
        canonical_id: &str,
        limit: usize,
    ) -> ReinResult<Vec<MemoryEvidence>> {
        let mut stmt = self.conn.prepare(
            "SELECT * FROM memory_evidence WHERE canonical_id = ?1 ORDER BY imported_at DESC LIMIT ?2",
        )?;
        let rows = stmt.query_map(rusqlite::params![canonical_id, limit as i64], |row| {
            row_to_memory_evidence(row).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    0,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })
        })?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    pub fn record_dedup_decision(&self, decision: DedupDecision) -> ReinResult<String> {
        let id = if decision.id.is_empty() {
            ulid::Ulid::new().to_string()
        } else {
            decision.id.clone()
        };
        self.conn.execute(
            "INSERT INTO dedup_decisions (
                id, winner_id, loser_id, canonical_id, lexical_score, embedding_score,
                relation, confidence, reason, operator, reversible, merged_summary,
                novel_facts, conflict_detected, payload, created_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)",
            rusqlite::params![
                &id,
                decision.winner_id,
                decision.loser_id,
                decision.canonical_id,
                decision.lexical_score,
                decision.embedding_score,
                decision.relation.as_str(),
                decision.confidence,
                decision.reason,
                decision.operator,
                if decision.reversible { 1 } else { 0 },
                decision.merged_summary,
                serde_json::to_string(&decision.novel_facts)?,
                if decision.conflict_detected { 1 } else { 0 },
                decision.payload.map(|value| value.to_string()),
                decision.created_at.to_rfc3339(),
            ],
        )?;
        Ok(id)
    }

    pub fn list_dedup_decisions(
        &self,
        canonical_id: Option<&str>,
        limit: usize,
    ) -> ReinResult<Vec<DedupDecision>> {
        self.list_dedup_decisions_filtered(canonical_id, None, limit)
    }

    /// Like [`list_dedup_decisions`] but supports an additional optional
    /// `operator` filter (e.g. `"llm_verdict"` or `"auto"`).
    pub fn list_dedup_decisions_filtered(
        &self,
        canonical_id: Option<&str>,
        operator: Option<&str>,
        limit: usize,
    ) -> ReinResult<Vec<DedupDecision>> {
        let map_err = |e: ReinError| {
            rusqlite::Error::FromSqlConversionFailure(
                0,
                rusqlite::types::Type::Text,
                Box::new(e),
            )
        };
        let decisions = match (canonical_id, operator) {
            (Some(cid), Some(op)) => {
                let mut stmt = self.conn.prepare(
                    "SELECT * FROM dedup_decisions \
                     WHERE canonical_id = ?1 AND operator = ?2 \
                     ORDER BY created_at DESC LIMIT ?3",
                )?;
                let rows = stmt.query_map(
                    rusqlite::params![cid, op, limit as i64],
                    |row| row_to_dedup_decision(row).map_err(map_err),
                )?;
                rows.filter_map(|r| r.ok()).collect()
            }
            (Some(cid), None) => {
                let mut stmt = self.conn.prepare(
                    "SELECT * FROM dedup_decisions WHERE canonical_id = ?1 \
                     ORDER BY created_at DESC LIMIT ?2",
                )?;
                let rows = stmt.query_map(
                    rusqlite::params![cid, limit as i64],
                    |row| row_to_dedup_decision(row).map_err(map_err),
                )?;
                rows.filter_map(|r| r.ok()).collect()
            }
            (None, Some(op)) => {
                let mut stmt = self.conn.prepare(
                    "SELECT * FROM dedup_decisions WHERE operator = ?1 \
                     ORDER BY created_at DESC LIMIT ?2",
                )?;
                let rows = stmt.query_map(
                    rusqlite::params![op, limit as i64],
                    |row| row_to_dedup_decision(row).map_err(map_err),
                )?;
                rows.filter_map(|r| r.ok()).collect()
            }
            (None, None) => {
                let mut stmt = self.conn.prepare(
                    "SELECT * FROM dedup_decisions ORDER BY created_at DESC LIMIT ?1",
                )?;
                let rows = stmt.query_map(
                    rusqlite::params![limit as i64],
                    |row| row_to_dedup_decision(row).map_err(map_err),
                )?;
                rows.filter_map(|r| r.ok()).collect()
            }
        };
        Ok(decisions)
    }

    pub fn list_canonical_memories(&self, limit: usize) -> ReinResult<Vec<Memory>> {
        let sql = format!(
            "{} WHERE m.superseded_by IS NULL AND COALESCE(cs.canonical_id, m.id) = m.id \
             AND m.status IN ('active', 'updated') \
             ORDER BY m.updated_at DESC LIMIT ?1",
            memory_select_base()
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(rusqlite::params![limit as i64], |row| {
            row_to_memory(row).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    0,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })
        })?;
        Ok(rows.filter_map(|r| r.ok()).collect())
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
    let canonical_id: Option<String> = row.get("canonical_id").unwrap_or(None);
    let support_count: u32 = row.get("support_count").unwrap_or(1);
    let merge_count: u32 = row.get("merge_count").unwrap_or(0);
    let dedup_confidence: f32 = row.get("dedup_confidence").unwrap_or(1.0);
    let source_diversity: f32 = row.get("source_diversity").unwrap_or(1.0);
    let contradiction_score: f32 = row.get("contradiction_score").unwrap_or(0.0);
    let related_ids_json: String = row.get("related_ids").map_err(ReinError::Database)?;
    let concept_ids_json: String = row.get("concept_ids").unwrap_or_else(|_| "[]".to_string());
    let status_str: String = row
        .get::<_, String>("status")
        .unwrap_or_else(|_| "active".to_string());
    let created_at_str: String = row.get("created_at").map_err(ReinError::Database)?;
    let updated_at_str: String = row.get("updated_at").map_err(ReinError::Database)?;
    let last_accessed_str: String = row.get("last_accessed").map_err(ReinError::Database)?;

    let layer = MemoryLayer::from_str(&layer_str).map_err(ReinError::Config)?;
    let importance = Importance::from_str(&importance_str).map_err(ReinError::Config)?;
    let source = Source::from_str(&source_str).map_err(ReinError::Config)?;
    let status = MemoryStatus::from_str(&status_str).unwrap_or_default();

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
        canonical_id,
        support_count,
        merge_count,
        dedup_confidence,
        source_diversity,
        contradiction_score,
        related_ids,
        concept_ids,
        status,
        embedding: None,
        tier: row
            .get::<_, String>("tier")
            .unwrap_or_else(|_| "warm".to_string())
            .parse()
            .unwrap_or_default(),
        cluster_id: row.get::<_, Option<u32>>("cluster_id").unwrap_or(None),
        created_at,
        updated_at,
        last_accessed,
    })
}

fn row_to_memory_evidence(row: &rusqlite::Row) -> ReinResult<MemoryEvidence> {
    let created_at: String = row.get("created_at").map_err(ReinError::Database)?;
    let imported_at: String = row.get("imported_at").map_err(ReinError::Database)?;
    let keywords_json: String = row.get("keywords").map_err(ReinError::Database)?;
    Ok(MemoryEvidence {
        id: row.get("id").map_err(ReinError::Database)?,
        canonical_id: row.get("canonical_id").map_err(ReinError::Database)?,
        memory_id: row.get("memory_id").unwrap_or(None),
        source_topic: row.get("source_topic").map_err(ReinError::Database)?,
        summary: row.get("summary").map_err(ReinError::Database)?,
        content: row.get("content").map_err(ReinError::Database)?,
        keywords: serde_json::from_str(&keywords_json).unwrap_or_default(),
        source: Source::from_str(
            &row.get::<_, String>("source")
                .map_err(ReinError::Database)?,
        )
        .map_err(ReinError::Config)?,
        created_at: DateTime::parse_from_rfc3339(&created_at)
            .map(|dt| dt.with_timezone(&Utc))
            .map_err(|e| ReinError::Config(format!("invalid evidence created_at: {e}")))?,
        imported_at: DateTime::parse_from_rfc3339(&imported_at)
            .map(|dt| dt.with_timezone(&Utc))
            .map_err(|e| ReinError::Config(format!("invalid evidence imported_at: {e}")))?,
    })
}

fn row_to_dedup_decision(row: &rusqlite::Row) -> ReinResult<DedupDecision> {
    let novel_facts_json: String = row
        .get::<_, String>("novel_facts")
        .unwrap_or_else(|_| "[]".to_string());
    let created_at: String = row.get("created_at").map_err(ReinError::Database)?;
    let payload_str: Option<String> = row.get("payload").unwrap_or(None);
    Ok(DedupDecision {
        id: row.get("id").map_err(ReinError::Database)?,
        winner_id: row.get("winner_id").unwrap_or(None),
        loser_id: row.get("loser_id").unwrap_or(None),
        canonical_id: row.get("canonical_id").unwrap_or(None),
        lexical_score: row.get("lexical_score").unwrap_or(None),
        embedding_score: row.get("embedding_score").unwrap_or(None),
        relation: row
            .get::<_, String>("relation")
            .map_err(ReinError::Database)?
            .parse()
            .map_err(ReinError::Config)?,
        confidence: row.get::<_, f32>("confidence").unwrap_or(0.0),
        reason: row.get("reason").map_err(ReinError::Database)?,
        operator: row
            .get::<_, String>("operator")
            .unwrap_or_else(|_| "auto".to_string()),
        reversible: row.get::<_, i64>("reversible").unwrap_or(1) != 0,
        merged_summary: row.get("merged_summary").unwrap_or(None),
        novel_facts: serde_json::from_str(&novel_facts_json).unwrap_or_default(),
        conflict_detected: row.get::<_, i64>("conflict_detected").unwrap_or(0) != 0,
        payload: payload_str.and_then(|value| serde_json::from_str(&value).ok()),
        created_at: DateTime::parse_from_rfc3339(&created_at)
            .map(|dt| dt.with_timezone(&Utc))
            .map_err(|e| ReinError::Config(format!("invalid dedup_decision created_at: {e}")))?,
    })
}

impl MemoryStore for SqliteStore {
    fn store(&self, mut memory: Memory) -> ReinResult<String> {
        // Normalize topic to prevent fragmentation (case, hyphen, underscore variants)
        memory.topic = crate::ops::normalize_topic_name(&memory.topic);

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
             tier, cluster_id, created_at, updated_at, last_accessed)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20)",
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
                memory.tier.to_string(),
                memory.cluster_id,
                memory.created_at.to_rfc3339(),
                memory.updated_at.to_rfc3339(),
                memory.last_accessed.to_rfc3339(),
            ],
        )?;

        if let Some(ref emb) = memory.embedding {
            vec::insert_embedding(&self.conn, &id, emb)?;
        }

        // Update side indexes (reuse keywords_json from above)
        self.update_tantivy(
            &id,
            &memory.topic,
            &memory.summary,
            &memory.content,
            &keywords_json,
        );
        self.update_hnsw(&id, memory.embedding.as_deref());
        let _ = self.snapshot_memory_as_evidence(&id, &memory);

        Ok(id)
    }

    fn get(&self, id: &str) -> ReinResult<Memory> {
        let sql = format!("{} WHERE m.id = ?1", memory_select_base());
        let mut stmt = self.conn.prepare(&sql)?;
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
             superseded_by=?11, related_ids=?12, concept_ids=?13, status=?14, tier=?15,
             cluster_id=?16, updated_at=?17, last_accessed=?18 WHERE id=?19",
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
                memory.tier.to_string(),
                memory.cluster_id,
                now.to_rfc3339(),
                memory.last_accessed.to_rfc3339(),
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
        self.update_tantivy(
            &memory.id,
            &memory.topic,
            &memory.summary,
            &memory.content,
            &keywords_json,
        );
        self.update_hnsw(&memory.id, memory.embedding.as_deref());

        Ok(())
    }

    fn delete(&self, id: &str) -> ReinResult<()> {
        // Wrap the delete + cross-reference cleanup in a single transaction so a
        // partial failure cannot leave concept.source_memory_ids / memory.related_ids /
        // episodes.memory_ids pointing at a non-existent memory.
        self.conn.execute("BEGIN IMMEDIATE", [])?;
        let result: ReinResult<()> = (|| {
            clean_memory_refs(&self.conn, id)?;
            let rows = self
                .conn
                .execute("DELETE FROM memories WHERE id = ?1", rusqlite::params![id])?;
            if rows == 0 {
                return Err(ReinError::NotFound(format!("memory {id} not found")));
            }
            // FTS cleanup handled by trigger; memory_canonical_state CASCADE.
            // vec_memories is a vec0 virtual table with NO FK cascade — the comment
            // that claimed otherwise was wrong. Failure here must abort the tx or
            // we leave a ghost embedding that the vector channel will keep returning.
            vec::delete_embedding(&self.conn, id)?;
            Ok(())
        })();

        match result {
            Ok(()) => {
                self.conn.execute("COMMIT", [])?;
                // Side indexes are fire-and-forget AFTER commit so their failure
                // can't roll back successful DB deletion.
                self.remove_from_tantivy(id);
                self.remove_from_hnsw(id);
                Ok(())
            }
            Err(e) => {
                let _ = self.conn.execute("ROLLBACK", []);
                Err(e)
            }
        }
    }

    fn search_fts(
        &self,
        query: &str,
        topic: Option<&str>,
        limit: usize,
    ) -> ReinResult<Vec<Memory>> {
        let results = fts::search_fts(&self.conn, query, topic, limit)?;
        let memories = results.into_iter().map(|(m, _)| m).collect();
        self.collapse_to_canonicals(memories, limit)
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
            match self.get_canonical(&id) {
                Ok(m) => {
                    if let Some(t) = topic {
                        if m.topic == t || crate::extract::topics_are_variants(&m.topic, t) {
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
        self.collapse_to_canonicals(memories, limit)
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
            // Snapshot of last_accessed so the UPDATE is a conditional CAS: if another
            // writer (e.g. intelligent_merge, record_access) bumps this row's
            // last_accessed between the SELECT above and the UPDATE below, the WHERE
            // clause misses and the stale decay value is discarded rather than
            // clobbering the fresher strength.
            last_accessed_snapshot: String,
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
            .filter_map(
                |(id, layer_str, decay_lambda, access_count, last_accessed_str)| {
                    let last_accessed = DateTime::parse_from_rfc3339(&last_accessed_str).ok()?;
                    let days =
                        (now - last_accessed.with_timezone(&Utc)).num_seconds() as f64 / 86400.0;
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
                        last_accessed_snapshot: last_accessed_str,
                    })
                },
            )
            .collect();

        let count = updates.len() as u64;
        // Use SAVEPOINT instead of BEGIN TRANSACTION so this can nest inside
        // an outer SAVEPOINT (e.g., GC dry-run preview).
        self.conn.execute_batch("SAVEPOINT decay_batch")?;
        for u in &updates {
            self.conn.execute(
                "UPDATE memories SET strength = ?1 WHERE id = ?2 AND last_accessed = ?3",
                rusqlite::params![u.new_strength, u.id, u.last_accessed_snapshot],
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
        let mut stmt = self
            .conn
            .prepare("SELECT topic FROM memories GROUP BY topic ORDER BY COUNT(*) DESC")?;
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
                .prepare("SELECT * FROM memories WHERE topic = ?1 AND status = 'active' AND superseded_by IS NULL")?;
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
            let _ = self
                .conn
                .execute_batch("ROLLBACK TO consolidate; RELEASE consolidate");
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
        let total_memories: usize =
            self.conn
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
        let topic_count: usize =
            self.conn
                .query_row("SELECT COUNT(DISTINCT topic) FROM memories", [], |row| {
                    row.get(0)
                })?;
        let avg_strength: f64 = self.conn.query_row(
            "SELECT COALESCE(AVG(strength), 0.0) FROM memories",
            [],
            |row| row.get(0),
        )?;

        let memoir_count: usize = self
            .conn
            .query_row("SELECT COUNT(*) FROM memoirs", [], |row| row.get(0))
            .unwrap_or(0);
        let concept_count: usize = self
            .conn
            .query_row("SELECT COUNT(*) FROM concepts", [], |row| row.get(0))
            .unwrap_or(0);
        let link_count: usize = self
            .conn
            .query_row("SELECT COUNT(*) FROM concept_links", [], |row| row.get(0))
            .unwrap_or(0);
        let hot_count: usize = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE tier = 'hot'",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);
        let warm_count: usize = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE tier = 'warm'",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);
        let cold_count: usize = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE tier = 'cold'",
                [],
                |row| row.get(0),
            )
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
            hot_count,
            warm_count,
            cold_count,
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

        // Acquire a write lock upfront so the SELECT and DELETE operate on the same
        // candidate set.  Without this, another writer can insert or decay a memory
        // between the two statements, making side-index cleanup incorrect.
        self.conn.execute_batch("BEGIN IMMEDIATE")?;

        let ids: Vec<String> = {
            let mut stmt = self.conn.prepare(
                "SELECT id FROM memories WHERE layer = 'STM' AND strength < ?1
                 AND importance NOT IN ('critical', 'high')",
            )?;
            let result: Vec<String> = stmt
                .query_map(rusqlite::params![threshold], |row| row.get(0))?
                .filter_map(|r| r.ok())
                .collect();
            result
        };

        let rows = self.conn.execute(
            "DELETE FROM memories WHERE layer = 'STM' AND strength < ?1
             AND importance NOT IN ('critical', 'high')",
            rusqlite::params![threshold],
        )?;

        // Remove pruned memory IDs from concept.source_memory_ids in a single pass.
        // We load all concepts once, filter to those with any pruned ID, and update them.
        // This avoids one LIKE full-table scan per pruned memory (O(n*m) → O(m)).
        if !ids.is_empty() {
            let pruned_set: std::collections::HashSet<&str> =
                ids.iter().map(|s| s.as_str()).collect();
            let all_concepts: Vec<(String, String)> = self
                .conn
                .prepare(
                    "SELECT id, source_memory_ids FROM concepts WHERE source_memory_ids != '[]'",
                )
                .and_then(|mut stmt| {
                    stmt.query_map([], |row| {
                        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                    })
                    .map(|mapped| mapped.filter_map(|r| r.ok()).collect())
                })
                .unwrap_or_default();
            for (concept_id, raw_ids) in all_concepts {
                if let Ok(mut mids) = serde_json::from_str::<Vec<String>>(&raw_ids) {
                    let before = mids.len();
                    mids.retain(|mid| !pruned_set.contains(mid.as_str()));
                    if mids.len() != before {
                        let updated = serde_json::to_string(&mids).unwrap_or_default();
                        self.conn.execute(
                            "UPDATE concepts SET source_memory_ids = ?1 WHERE id = ?2",
                            rusqlite::params![updated, concept_id],
                        )?;
                    }
                }
            }
        }

        self.conn.execute_batch("COMMIT")?;

        // Side-index cleanup runs outside the transaction (non-SQL, fire-and-forget)
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
        mut memory: Memory,
        similarity_threshold: f32,
        time_window_days: i64,
    ) -> ReinResult<String> {
        // Normalize topic to prevent fragmentation (case, hyphen, underscore variants)
        memory.topic = crate::ops::normalize_topic_name(&memory.topic);

        // Pre-flight intelligent classification: run the LLM call BEFORE grabbing
        // the write lock to avoid holding BEGIN IMMEDIATE for 1-2s. If the pre-flight
        // candidate still matches the in-transaction dedup decision, we apply the
        // pre-computed verdict; otherwise we fall back to mechanical behavior.
        let preflight =
            self.preflight_intelligent_classify(&memory, similarity_threshold, time_window_days);

        self.conn.execute_batch("BEGIN IMMEDIATE")?;
        // Third tuple element is the `pending_grayzone_jobs` row id written inside
        // the transaction; the post-COMMIT enqueue deletes this row on success.
        // On crash, startup drains surviving rows back into the file queue.
        let mut pending_grayzone: Option<(String, String, f32)> = None;

        let decision = (|| -> ReinResult<String> {
            let memory = memory;
            let inferred_cluster = memory
                .cluster_id
                .or_else(|| self.infer_cluster_from_cache(&memory));
            let dedup_action = crate::extract::check_dedup(
                self,
                &memory.topic,
                &memory.summary,
                &memory.content,
                similarity_threshold,
                time_window_days,
                inferred_cluster,
            )?;

            let candidate_cluster = match &dedup_action {
                DedupAction::MergeInto(id)
                | DedupAction::Supersede(id)
                | DedupAction::GrayZone(id, _) => self.get(id).ok().and_then(|m| m.cluster_id),
                DedupAction::CreateNew => memory.cluster_id,
            };

            // Per-cluster adaptive threshold recheck — honor [adaptive] enabled=false
            // so disabling adaptive actually disables it end-to-end (not just in the
            // outer effective_dedup_threshold helper).
            let dedup_action = {
                let loaded_config = crate::config::ReinConfig::load().unwrap_or_default();
                if loaded_config.adaptive.enabled {
                    if let Some(state) =
                        crate::store::adaptive::AdaptiveState::restore_snapshot(&self.conn)
                    {
                        let effective = state.get_dedup_threshold(candidate_cluster);
                        if (effective - similarity_threshold).abs() > 0.01 {
                            crate::extract::check_dedup(
                                self,
                                &memory.topic,
                                &memory.summary,
                                &memory.content,
                                effective,
                                time_window_days,
                                inferred_cluster,
                            )?
                        } else {
                            dedup_action
                        }
                    } else {
                        dedup_action
                    }
                } else {
                    dedup_action
                }
            };

            // Gray-zone resolution. Apply pre-flight intelligent_merge verdict
            // when its candidate matches the in-transaction candidate; otherwise
            // fall back to the mechanical canonical-anchor → pending-queue chain.
            // Intelligent-merge direct path: when the pre-flight verdict applies,
            // handle Ignore/Update/Merge WITHOUT routing through store_with_dedup_resolved's
            // MergeInto (which mechanically appends unique lines). Ignore means "drop
            // incoming"; Update/Merge means "replace existing content with synthesis".
            // Only CreateNew falls through to the mechanical path.
            let mut im_verdict_for_provenance: Option<(
                String,
                f32,
                InsertionVerdict,
                Option<String>,
            )> = None;
            let mut evidence_snapshot: Option<Memory> = None;
            let mut intelligent_result: Option<String> = None;

            let resolved_action = match dedup_action {
                DedupAction::GrayZone(candidate_id, sim) => {
                    // Race guards: candidate id + content hash match.
                    let applied = preflight.as_ref().and_then(|(pf_id, pf_hash, v)| {
                        if pf_id != &candidate_id {
                            crate::extract::intelligent_merge::note_stale_race();
                            return None;
                        }
                        let current = self.get(&candidate_id).ok()?;
                        if content_hash(&current.content) != *pf_hash {
                            crate::extract::intelligent_merge::note_stale_race();
                            tracing::debug!(
                                candidate_id,
                                "intelligent_merge: stale verdict — content changed since pre-flight"
                            );
                            return None;
                        }
                        Some(v.clone())
                    });

                    if let Some(v) = applied {
                        let existing = self.get(&candidate_id).ok();
                        match v.verdict {
                            InsertionVerdict::Ignore => {
                                // Drop incoming. Bump access_count on existing so the signal
                                // that someone tried to re-state the same fact isn't lost.
                                let _ = self.conn.execute(
                                    "UPDATE memories SET access_count = access_count + 1,
                                     last_accessed = ?1 WHERE id = ?2",
                                    rusqlite::params![
                                        chrono::Utc::now().to_rfc3339(),
                                        &candidate_id
                                    ],
                                );
                                im_verdict_for_provenance = Some((
                                    candidate_id.clone(),
                                    sim,
                                    v.verdict,
                                    v.reasoning.clone(),
                                ));
                                intelligent_result = Some(candidate_id);
                                DedupAction::CreateNew // placeholder; short-circuited below
                            }
                            InsertionVerdict::Update | InsertionVerdict::Merge => {
                                if let Some(synth) = v.synthesized.clone() {
                                    // Go through the full update() path so Tantivy/HNSW
                                    // side indexes, status=Updated, and updated_at are all
                                    // refreshed — otherwise list views and recall ranking
                                    // keep reflecting stale metadata after synthesis.
                                    // summary stays unchanged unless the incoming summary
                                    // is materially different; keeping summary steady avoids
                                    // thrashing summary-based UI labels.
                                    if let Some(existing_mem) = existing.clone() {
                                        let mut updated = existing_mem.clone();
                                        updated.content = synth;
                                        // If incoming has a different/richer summary, adopt it.
                                        if !memory.summary.is_empty()
                                            && memory.summary != existing_mem.summary
                                        {
                                            updated.summary = memory.summary.clone();
                                        }
                                        // Merge keywords (dedup + cap).
                                        for kw in &memory.keywords {
                                            if !updated.keywords.contains(kw)
                                                && updated.keywords.len() < 16
                                            {
                                                updated.keywords.push(kw.clone());
                                            }
                                        }
                                        let _ = self.update(&updated);
                                    } else {
                                        // Existing memory vanished between snapshot and write;
                                        // fall through to mechanical merge to stay safe.
                                        tracing::warn!(
                                            candidate_id,
                                            "intelligent_merge: existing vanished pre-update, falling back"
                                        );
                                        return self.store_with_dedup_resolved(
                                            memory,
                                            DedupAction::MergeInto(candidate_id.clone()),
                                        );
                                    }
                                    evidence_snapshot = existing;
                                    im_verdict_for_provenance = Some((
                                        candidate_id.clone(),
                                        sim,
                                        v.verdict,
                                        v.reasoning.clone(),
                                    ));
                                    intelligent_result = Some(candidate_id);
                                    DedupAction::CreateNew // placeholder; short-circuited below
                                } else {
                                    // LLM returned no synthesis — fall back to mechanical merge.
                                    tracing::warn!(
                                        candidate_id,
                                        verdict = ?v.verdict,
                                        "intelligent_merge: verdict lacks synthesized content, falling back"
                                    );
                                    DedupAction::MergeInto(candidate_id)
                                }
                            }
                            InsertionVerdict::CreateNew => {
                                im_verdict_for_provenance = Some((
                                    candidate_id.clone(),
                                    sim,
                                    v.verdict,
                                    v.reasoning.clone(),
                                ));
                                DedupAction::CreateNew
                            }
                        }
                    } else if let Some(canonical_id) =
                        self.grayzone_canonical_anchor(&candidate_id)?
                    {
                        tracing::debug!(
                            "gray-zone dedup: reusing canonical {canonical_id} for candidate {candidate_id} (sim={sim:.2})"
                        );
                        DedupAction::MergeInto(canonical_id)
                    } else {
                        if self.get(&candidate_id).is_ok() {
                            // Persist inside the transaction so a crash between
                            // COMMIT and the file-queue enqueue cannot lose this
                            // pair silently. The post-COMMIT path deletes the row
                            // on successful enqueue; startup drains any survivors.
                            let job_id = ulid::Ulid::new().to_string();
                            self.conn.execute(
                                "INSERT INTO pending_grayzone_jobs
                                 (id, candidate_id, result_id, sim, created_at)
                                 VALUES (?1, ?2, ?3, ?4, ?5)",
                                rusqlite::params![
                                    &job_id,
                                    &candidate_id,
                                    &memory.id,
                                    sim as f64,
                                    chrono::Utc::now().to_rfc3339(),
                                ],
                            )?;
                            pending_grayzone = Some((job_id, candidate_id, sim));
                        }
                        DedupAction::CreateNew
                    }
                }
                other => other,
            };

            // If the intelligent-merge direct path applied, use its result id;
            // otherwise defer to the mechanical store_with_dedup_resolved.
            let result_id = if let Some(id) = intelligent_result {
                id
            } else {
                self.store_with_dedup_resolved(memory, resolved_action)?
            };

            // (#3) Preserve the pre-merge existing memory as evidence so Update/Merge
            // doesn't discard the prior version — it can be surfaced on demand.
            if let Some(existing) = evidence_snapshot {
                let canonical_id = self
                    .canonical_id_for(&result_id)
                    .unwrap_or(result_id.clone());
                let _ = self.conn.execute(
                    "INSERT OR IGNORE INTO memory_evidence
                     (id, canonical_id, memory_id, source_topic, summary, content,
                      keywords, source, created_at, imported_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                    rusqlite::params![
                        ulid::Ulid::new().to_string(),
                        canonical_id,
                        existing.id,
                        existing.topic,
                        existing.summary,
                        existing.content,
                        serde_json::to_string(&existing.keywords).unwrap_or_else(|_| "[]".into()),
                        match existing.source {
                            crate::types::Source::Manual => "manual",
                            crate::types::Source::Hook => "hook",
                            crate::types::Source::Migration => "migration",
                            crate::types::Source::Supermemory => "supermemory",
                            crate::types::Source::Proxy => "proxy",
                        },
                        existing.created_at.to_rfc3339(),
                        chrono::Utc::now().to_rfc3339(),
                    ],
                );
            }

            // (#2) Provenance: record the LLM verdict in dedup_decisions so future
            // introspection can explain why this canonical exists in its current form.
            if let Some((cand_id, sim, verdict, reasoning)) = im_verdict_for_provenance {
                let (relation, merged_summary, conflict) = match verdict {
                    InsertionVerdict::Ignore => ("duplicate", None, false),
                    InsertionVerdict::Update => {
                        ("update", memory_content_snapshot(self, &result_id), false)
                    }
                    InsertionVerdict::Merge => {
                        ("update", memory_content_snapshot(self, &result_id), false)
                    }
                    InsertionVerdict::CreateNew => ("distinct", None, false),
                };
                let _ = self.conn.execute(
                    "INSERT INTO dedup_decisions
                     (id, winner_id, loser_id, canonical_id, lexical_score, embedding_score,
                      relation, confidence, reason, operator, reversible,
                      merged_summary, novel_facts, conflict_detected, payload, created_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 'llm_verdict', 0,
                             ?10, '[]', ?11, NULL, ?12)",
                    rusqlite::params![
                        ulid::Ulid::new().to_string(),
                        &result_id,
                        &cand_id,
                        self.canonical_id_for(&result_id)
                            .unwrap_or_else(|_| result_id.clone()),
                        Option::<f64>::None,
                        sim as f64,
                        relation,
                        0.8f64,
                        reasoning.as_deref().unwrap_or(""),
                        merged_summary,
                        conflict as i64,
                        chrono::Utc::now().to_rfc3339(),
                    ],
                );
            }

            Ok(result_id)
        })();

        match decision {
            Ok(result) => {
                self.conn.execute_batch("COMMIT")?;
                if let Some((pending_row_id, candidate_id, sim)) = pending_grayzone {
                    let config = crate::config::ReinConfig::load().unwrap_or_default();
                    match crate::extract::hooks::queue::queue_dedup_job(
                        &config,
                        candidate_id.clone(),
                        result.clone(),
                        Some(sim),
                        "store_gray_zone",
                    ) {
                        Ok(job_id) => {
                            crate::extract::hooks::queue::spawn_dedup_worker(&config);
                            // File queue took ownership — drop the durable row.
                            let _ = self.conn.execute(
                                "DELETE FROM pending_grayzone_jobs WHERE id = ?1",
                                rusqlite::params![&pending_row_id],
                            );
                            tracing::debug!(
                                "queued dedup job {job_id} for gray-zone pair {} <-> {} (sim={sim:.2})",
                                candidate_id,
                                result
                            );
                        }
                        Err(error) => {
                            // Leave the durable row in place; startup drain will
                            // retry. This is the whole point of the SQLite backing.
                            tracing::warn!(
                                "failed to queue gray-zone dedup job for {} <-> {}: {} (will retry on next drain)",
                                candidate_id,
                                result,
                                error
                            );
                        }
                    }
                }
                Ok(result)
            }
            Err(e) => {
                let _ = self.conn.execute_batch("ROLLBACK");
                Err(e)
            }
        }
    }

    /// Pre-flight intelligent-merge classification.
    ///
    /// Runs a read-only dedup check outside the write transaction. If the
    /// result is GrayZone AND intelligent_merge is enabled AND an LLM is
    /// reachable, classifies the pair and returns `(candidate_id, verdict)`.
    ///
    /// Keeping this call OUTSIDE `BEGIN IMMEDIATE` prevents the LLM latency
    /// (1-2s) from holding the write lock. The in-transaction path re-checks
    /// dedup and only applies the pre-flight verdict when the candidate id
    /// matches — otherwise it falls back to the mechanical route.
    ///
    /// Returns None when:
    /// - feature disabled
    /// - no gray-zone candidate detected in read-only check
    /// - LLM unreachable, failed, or parse error
    fn preflight_intelligent_classify(
        &self,
        memory: &Memory,
        similarity_threshold: f32,
        time_window_days: i64,
    ) -> Option<(String, u64, VerdictResult)> {
        let config = crate::config::ReinConfig::load().unwrap_or_default();
        if !config.intelligent_merge.enabled {
            return None;
        }

        // Read-only dedup preview using the SAME effective threshold the
        // in-transaction recheck will use — otherwise preflight may classify
        // pairs that later get ruled out (wasted LLM call), or skip pairs
        // that the recheck will reclassify as GrayZone (never gets verdict).
        let inferred_cluster = memory
            .cluster_id
            .or_else(|| self.infer_cluster_from_cache(memory));
        // Honor [adaptive] enabled=false throughout — otherwise preflight picks
        // the learned threshold while in-transaction recheck picks the static one,
        // creating a mismatch that defeats the content-hash race guard.
        let effective_threshold = if config.adaptive.enabled {
            crate::store::adaptive::AdaptiveState::restore_snapshot(&self.conn)
                .map(|s| s.get_dedup_threshold(inferred_cluster))
                .unwrap_or(similarity_threshold)
        } else {
            similarity_threshold
        };

        let preview = crate::extract::check_dedup(
            self,
            &memory.topic,
            &memory.summary,
            &memory.content,
            effective_threshold,
            time_window_days,
            inferred_cluster,
        )
        .ok()?;

        let DedupAction::GrayZone(candidate_id, _sim) = preview else {
            return None;
        };

        let existing = self.get(&candidate_id).ok()?;
        let existing_hash = content_hash(&existing.content);
        let existing_snippet = crate::extract::intelligent_merge::MemorySnippet::from(&existing);
        let incoming_snippet = crate::extract::intelligent_merge::MemorySnippet {
            topic: memory.topic.clone(),
            summary: memory.summary.clone(),
            content: memory.content.clone(),
        };

        let verdict = crate::extract::intelligent_merge::classify_insertion(
            &config,
            &existing_snippet,
            &incoming_snippet,
        )?;

        tracing::info!(
            candidate_id,
            verdict = ?verdict.verdict,
            reasoning = verdict.reasoning.as_deref().unwrap_or(""),
            "intelligent_merge preflight verdict"
        );
        Some((candidate_id, existing_hash, verdict))
    }

    fn grayzone_canonical_anchor(&self, candidate_id: &str) -> ReinResult<Option<String>> {
        let canonical_id = self.canonical_id_for(candidate_id)?;
        let Some(canonical) = self.get(&canonical_id).ok() else {
            return Ok(None);
        };

        if canonical.superseded_by.is_some()
            || !matches!(
                canonical.status,
                MemoryStatus::Active | MemoryStatus::Updated
            )
        {
            return Ok(None);
        }

        let has_evidence = self
            .conn
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM memory_evidence WHERE canonical_id = ?1)",
                rusqlite::params![&canonical_id],
                |row| row.get::<_, i64>(0),
            )
            .unwrap_or(0)
            > 0;

        if has_evidence || canonical_id != candidate_id {
            Ok(Some(canonical_id))
        } else {
            Ok(None)
        }
    }

    /// Execute a pre-resolved dedup action within a BEGIN IMMEDIATE transaction.
    fn store_with_dedup_resolved(&self, memory: Memory, action: DedupAction) -> ReinResult<String> {
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
                    let unique =
                        crate::ops::extract_unique_lines(&memory.content, &existing.content);
                    let canonical_id = self.canonical_id_for(&existing_id)?;
                    if !unique.is_empty() {
                        existing.content.push_str(&format!(
                            "\n\n[merged on {}]\n{}",
                            chrono::Utc::now().format("%Y-%m-%d"),
                            unique
                        ));
                    }
                    // Cap content length to prevent unbounded growth from repeated merges.
                    // Trim from the end (newest additions) to preserve original core content.
                    if existing.content.len() > 10_000 {
                        let mut trim_at = 10_000;
                        while trim_at > 0 && !existing.content.is_char_boundary(trim_at) {
                            trim_at -= 1;
                        }
                        existing.content.truncate(trim_at);
                    }
                    existing.summary = existing
                        .content
                        .chars()
                        .take(crate::types::SUMMARY_MAX_CHARS)
                        .collect();
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
                    let _ = self.snapshot_memory_as_evidence(&canonical_id, &memory);
                    let _ = self.record_dedup_decision(DedupDecision {
                        id: String::new(),
                        winner_id: Some(existing_id.clone()),
                        loser_id: Some(memory.id.clone()),
                        canonical_id: Some(canonical_id.clone()),
                        lexical_score: None,
                        embedding_score: None,
                        relation: DedupRelation::Duplicate,
                        confidence: 0.9,
                        reason: "store_merge".to_string(),
                        operator: "auto".to_string(),
                        reversible: true,
                        merged_summary: Some(existing.summary.clone()),
                        novel_facts: unique
                            .lines()
                            .map(|line| line.trim().to_string())
                            .filter(|line| !line.is_empty())
                            .collect(),
                        conflict_detected: false,
                        payload: None,
                        created_at: Utc::now(),
                    });
                    Ok(existing_id)
                } else {
                    self.store(memory)
                }
            }
            DedupAction::Supersede(old_id) => {
                let new_id = self.store(memory)?;
                // Mark old memory as superseded
                self.mark_superseded(&old_id, &new_id)?;
                let _ = self.record_dedup_decision(DedupDecision {
                    id: String::new(),
                    winner_id: Some(new_id.clone()),
                    loser_id: Some(old_id.clone()),
                    canonical_id: Some(new_id.clone()),
                    lexical_score: None,
                    embedding_score: None,
                    relation: DedupRelation::Update,
                    confidence: 0.9,
                    reason: "store_supersede".to_string(),
                    operator: "auto".to_string(),
                    reversible: true,
                    merged_summary: None,
                    novel_facts: vec![],
                    conflict_detected: false,
                    payload: None,
                    created_at: Utc::now(),
                });
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
    fn with_tantivy<F>(&self, f: F)
    where
        F: FnOnce(&super::tantivy_fts::TantivyFts),
    {
        if self.db_path.to_str() == Some(":memory:") {
            return;
        }
        let mut cache = self.tantivy_cache.borrow_mut();
        if cache.is_none() {
            *cache = super::tantivy_fts::TantivyFts::open(&self.tantivy_path()).ok();
        }
        if let Some(ref tantivy) = *cache {
            f(tantivy);
        }
    }

    /// Backfill `session_artifacts.episode_id` for rows orphaned by a crash
    /// between `create_episode` and `link_session_artifact_episode` in the
    /// stop-hook pipeline. Matches on `session_id == episode.source_session_id`
    /// so the relationship can be reconstructed idempotently — safe to call on
    /// every startup even when there are no orphans.
    pub fn repair_orphan_artifact_episode_links(&self) -> ReinResult<usize> {
        let affected = self.conn.execute(
            "UPDATE session_artifacts
                SET episode_id = (
                    SELECT e.id FROM episodes AS e
                     WHERE e.source_session_id = session_artifacts.session_id
                     ORDER BY e.created_at DESC
                     LIMIT 1
                )
              WHERE session_artifacts.episode_id IS NULL
                AND session_artifacts.session_id IS NOT NULL
                AND EXISTS (
                    SELECT 1 FROM episodes AS e
                     WHERE e.source_session_id = session_artifacts.session_id
                )",
            [],
        )?;
        Ok(affected)
    }

    /// Drain the durable `pending_grayzone_jobs` table back into the file queue.
    ///
    /// Covers the crash-after-COMMIT / enqueue-failure window in
    /// `store_with_dedup_resolved`. Safe to call on every startup; if nothing
    /// is pending, it's an O(1) query. On successful enqueue each row is
    /// deleted; on persistent queue failure the row is left for the next call.
    pub fn drain_pending_grayzone_jobs(
        &self,
        config: &crate::config::ReinConfig,
    ) -> ReinResult<usize> {
        let mut stmt = self.conn.prepare(
            "SELECT id, candidate_id, result_id, sim FROM pending_grayzone_jobs
             ORDER BY created_at",
        )?;
        let rows: Vec<(String, String, String, Option<f64>)> = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, Option<f64>>(3)?,
                ))
            })?
            .filter_map(|r| r.ok())
            .collect();
        drop(stmt);

        let mut drained = 0usize;
        for (row_id, candidate_id, result_id, sim) in rows {
            match crate::extract::hooks::queue::queue_dedup_job(
                config,
                candidate_id.clone(),
                result_id.clone(),
                sim.map(|v| v as f32),
                "startup_drain",
            ) {
                Ok(_) => {
                    let _ = self.conn.execute(
                        "DELETE FROM pending_grayzone_jobs WHERE id = ?1",
                        rusqlite::params![&row_id],
                    );
                    drained += 1;
                }
                Err(error) => {
                    tracing::warn!(
                        "failed to drain pending grayzone job {row_id} for {} <-> {}: {error}",
                        candidate_id,
                        result_id
                    );
                }
            }
        }
        if drained > 0 {
            crate::extract::hooks::queue::spawn_dedup_worker(config);
        }
        Ok(drained)
    }

    /// Fire-and-forget: update Tantivy index after a write.
    fn update_tantivy(&self, id: &str, topic: &str, summary: &str, content: &str, keywords: &str) {
        self.with_tantivy(|t| {
            let _ = t.insert(id, topic, summary, content, keywords);
        });
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
                tracing::warn!(
                    "HNSW flock(LOCK_EX) failed: {}",
                    std::io::Error::last_os_error()
                );
                return None;
            }
        }

        let mut index = match crate::store::hnsw::HnswIndex::open(&hnsw_path, dims) {
            Ok(index) => index,
            Err(e) => {
                tracing::warn!("HNSW index open failed: {e}");
                crate::store::hnsw::HnswIndex::mark_dirty(&hnsw_path);
                return None;
            }
        };
        let result = f(&mut index);
        if let Err(e) = index.save() {
            tracing::warn!("HNSW index save failed: {e}");
            crate::store::hnsw::HnswIndex::mark_dirty(&hnsw_path);
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
    ///
    /// If the lock can't be acquired (contention, flock failure) the index is
    /// marked dirty so the next rebuild will pick up the missed write, rather
    /// than silently serving a stale index.
    fn update_hnsw(&self, id: &str, embedding: Option<&[f32]>) {
        if let Some(emb) = embedding {
            match self.with_hnsw_lock(emb.len(), |index| index.insert(id, emb)) {
                Some(Ok(())) => {}
                Some(Err(error)) => {
                    tracing::warn!("HNSW insert failed for {id}: {error}");
                    crate::store::hnsw::HnswIndex::mark_dirty(&self.hnsw_path());
                }
                None => {
                    // Could not acquire lock / :memory: / open failed.
                    // :memory: path legitimately has no side-index; skip silently.
                    if self.db_path.to_str() != Some(":memory:") {
                        tracing::warn!(
                            "HNSW insert skipped for {id} (lock unavailable); marking dirty"
                        );
                        crate::store::hnsw::HnswIndex::mark_dirty(&self.hnsw_path());
                    }
                }
            }
        }
    }

    /// Fire-and-forget: remove from Tantivy index after a delete.
    pub(crate) fn remove_from_tantivy(&self, id: &str) {
        self.with_tantivy(|t| {
            let _ = t.delete(id);
        });
    }

    /// Fire-and-forget: remove from HNSW index after a delete.
    ///
    /// Mirrors [`Self::update_hnsw`]: on lock-unavailable or open/save failure
    /// we mark the index dirty so the next warmup rebuild picks up the missed
    /// removal. Without this, a failed delete leaves a stale embedding that the
    /// vector channel keeps returning for the lifetime of the index file.
    pub(crate) fn remove_from_hnsw(&self, id: &str) {
        match self.with_hnsw_lock(self.dims, |index| index.delete(id)) {
            Some(Ok(())) => {}
            Some(Err(error)) => {
                tracing::warn!("HNSW delete failed for {id}: {error}");
                crate::store::hnsw::HnswIndex::mark_dirty(&self.hnsw_path());
            }
            None => {
                if self.db_path.to_str() != Some(":memory:") {
                    tracing::warn!(
                        "HNSW delete skipped for {id} (lock unavailable); marking dirty"
                    );
                    crate::store::hnsw::HnswIndex::mark_dirty(&self.hnsw_path());
                }
            }
        }
    }

    pub fn canonical_id_for(&self, memory_id: &str) -> ReinResult<String> {
        Ok(self
            .conn
            .query_row(
                "SELECT COALESCE(canonical_id, memory_id) FROM memory_canonical_state WHERE memory_id = ?1",
                rusqlite::params![memory_id],
                |row| row.get(0),
            )
            .unwrap_or_else(|_| memory_id.to_string()))
    }

    pub fn refresh_canonical_state(&self, canonical_id: &str) -> ReinResult<()> {
        self.conn.execute(
            "INSERT OR IGNORE INTO memory_canonical_state(memory_id, canonical_id) VALUES (?1, ?1)",
            rusqlite::params![canonical_id],
        )?;
        self.conn.execute(
            "UPDATE memory_canonical_state
             SET support_count = (
                    SELECT COUNT(*) FROM memory_evidence WHERE canonical_id = ?1
                 ),
                 merge_count = CASE
                    WHEN (SELECT COUNT(*) FROM memory_evidence WHERE canonical_id = ?1) > 0
                    THEN (SELECT COUNT(*) FROM memory_evidence WHERE canonical_id = ?1) - 1
                    ELSE 0
                 END,
                 source_diversity = (
                    SELECT CAST(COUNT(DISTINCT source) AS REAL) FROM memory_evidence WHERE canonical_id = ?1
                 ),
                 last_merged_at = CASE
                    WHEN (SELECT COUNT(*) FROM memory_evidence WHERE canonical_id = ?1) > 1
                    THEN strftime('%Y-%m-%dT%H:%M:%fZ','now')
                    ELSE last_merged_at
                 END
             WHERE memory_id = ?1",
            rusqlite::params![canonical_id],
        )?;
        Ok(())
    }

    /// Mark an old memory as superseded by a new one.
    ///
    /// Wrapped in a SAVEPOINT so that if the canonical_state update fails after the
    /// memories row has been updated, both changes roll back together — preventing
    /// a superseded memory from pointing at a dangling canonical reference.
    pub fn mark_superseded(&self, old_id: &str, new_id: &str) -> ReinResult<()> {
        let canonical_id = self.canonical_id_for(new_id)?;
        self.conn.execute("SAVEPOINT mark_superseded", [])?;
        let result: ReinResult<()> = (|| {
            let rows = self.conn.execute(
                "UPDATE memories SET superseded_by = ?1 WHERE id = ?2",
                rusqlite::params![new_id, old_id],
            )?;
            if rows == 0 {
                return Err(ReinError::NotFound(format!("memory {old_id} not found")));
            }
            self.conn.execute(
                "INSERT OR IGNORE INTO memory_canonical_state(memory_id, canonical_id) VALUES (?1, ?2)",
                rusqlite::params![old_id, canonical_id],
            )?;
            self.conn.execute(
                "UPDATE memory_canonical_state SET canonical_id = ?1 WHERE memory_id = ?2",
                rusqlite::params![canonical_id, old_id],
            )?;
            Ok(())
        })();
        match result {
            Ok(()) => {
                self.conn.execute("RELEASE mark_superseded", [])?;
                Ok(())
            }
            Err(e) => {
                let _ = self.conn.execute("ROLLBACK TO mark_superseded", []);
                let _ = self.conn.execute("RELEASE mark_superseded", []);
                Err(e)
            }
        }
    }
}

/// Stable 64-bit hash of a memory content string, used as a race guard so a
/// pre-flight LLM verdict can be discarded if the existing memory's content
/// was mutated between pre-flight and the write transaction.
fn content_hash(content: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    content.hash(&mut hasher);
    hasher.finish()
}

/// Fetch a short summary/content snapshot for provenance logging.
/// Returns just the content string if the id exists, None otherwise.
fn memory_content_snapshot(store: &SqliteStore, id: &str) -> Option<String> {
    store.get(id).ok().map(|m| m.content)
}

/// Remove a memory ID from every JSON array across the schema that might reference it.
///
/// Must be called inside an open transaction. Targets:
/// - `concepts.source_memory_ids`
/// - `memories.related_ids` (other memories pointing at this one)
/// - `episodes.memory_ids`
/// - `knowledge_units.source_memory_ids` (if present — knowledge graph layer)
///
/// `memory_canonical_state` is handled via ON DELETE CASCADE on the schema FK.
/// This matches the cleanup already performed by batch prune paths so single-memory
/// deletion (rein_forget / CLI / REST) no longer orphans JSON refs.
pub(super) fn clean_memory_refs(conn: &Connection, memory_id: &str) -> ReinResult<()> {
    // Quoted JSON string literal form, e.g. "01KP..."
    let quoted = format!("\"{memory_id}\"");
    let like = format!("%{quoted}%");

    // concepts.source_memory_ids
    conn.execute(
        "UPDATE concepts
         SET source_memory_ids = COALESCE(
            (SELECT json_group_array(value)
             FROM json_each(source_memory_ids)
             WHERE value != ?1),
            '[]')
         WHERE source_memory_ids LIKE ?2",
        rusqlite::params![memory_id, like],
    )?;

    // memories.related_ids — other memories that list this id
    conn.execute(
        "UPDATE memories
         SET related_ids = COALESCE(
            (SELECT json_group_array(value)
             FROM json_each(related_ids)
             WHERE value != ?1),
            '[]')
         WHERE related_ids LIKE ?2 AND id != ?1",
        rusqlite::params![memory_id, like],
    )?;

    // episodes.memory_ids
    // Guarded with sqlite_master check so older schemas without episodes don't fail.
    let has_episodes: bool = conn
        .query_row(
            "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='episodes'",
            [],
            |r| r.get(0),
        )
        .unwrap_or(false);
    if has_episodes {
        conn.execute(
            "UPDATE episodes
             SET memory_ids = COALESCE(
                (SELECT json_group_array(value)
                 FROM json_each(memory_ids)
                 WHERE value != ?1),
                '[]')
             WHERE memory_ids LIKE ?2",
            rusqlite::params![memory_id, like],
        )?;
    }

    Ok(())
}

/// Remove a concept ID from every JSON array that might reference it.
///
/// Caller must open a transaction.  Targets:
/// - `memories.concept_ids`
/// - `episodes.concept_ids`
///
/// `concept_links` and `concept_revisions` are handled via ON DELETE CASCADE.
#[allow(dead_code)]
fn clean_concept_refs(conn: &Connection, concept_id: &str) -> ReinResult<()> {
    let quoted = format!("\"{concept_id}\"");
    let like = format!("%{quoted}%");

    conn.execute(
        "UPDATE memories
         SET concept_ids = COALESCE(
            (SELECT json_group_array(value)
             FROM json_each(concept_ids)
             WHERE value != ?1),
            '[]')
         WHERE concept_ids LIKE ?2",
        rusqlite::params![concept_id, like],
    )?;

    let has_episodes: bool = conn
        .query_row(
            "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='episodes'",
            [],
            |r| r.get(0),
        )
        .unwrap_or(false);
    if has_episodes {
        conn.execute(
            "UPDATE episodes
             SET concept_ids = COALESCE(
                (SELECT json_group_array(value)
                 FROM json_each(concept_ids)
                 WHERE value != ?1),
                '[]')
             WHERE concept_ids LIKE ?2",
            rusqlite::params![concept_id, like],
        )?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Importance, Source};
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    struct EnvRestore {
        key: &'static str,
        value: Option<String>,
    }

    impl Drop for EnvRestore {
        fn drop(&mut self) {
            match &self.value {
                Some(value) => std::env::set_var(self.key, value),
                None => std::env::remove_var(self.key),
            }
        }
    }

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
            canonical_id: None,
            support_count: 1,
            merge_count: 0,
            dedup_confidence: 1.0,
            source_diversity: 1.0,
            contradiction_score: 0.0,
            related_ids: vec![],
            concept_ids: vec![],
            status: MemoryStatus::default(),
            embedding: None,
            tier: MemoryTier::Warm,
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
    fn test_preflight_classify_returns_none_without_grayzone() {
        // With no stored memories there's no candidate for dedup, so preflight
        // must return None regardless of the feature flag — no LLM call attempted.
        let store = SqliteStore::in_memory().unwrap();
        let mem = test_memory("rust", "unique content", Importance::Medium);
        let result = store.preflight_intelligent_classify(&mem, 0.7, 7);
        assert!(
            result.is_none(),
            "empty store → no gray zone → no classification"
        );
    }

    #[test]
    fn test_delete_cleans_related_ids_in_other_memories() {
        // A memory's id should be scrubbed from other memories' related_ids
        // JSON arrays when it's deleted — otherwise those arrays point at
        // a non-existent id.
        let store = SqliteStore::in_memory().unwrap();
        let target = test_memory("rust", "target", Importance::Medium);
        let target_id = store.store(target).unwrap();
        let mut other = test_memory("rust", "other", Importance::Medium);
        other.related_ids = vec![target_id.clone(), "other-id".to_string()];
        let other_id = store.store(other).unwrap();

        store.delete(&target_id).unwrap();

        let refreshed = store.get(&other_id).unwrap();
        assert_eq!(refreshed.related_ids, vec!["other-id".to_string()]);
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
        assert_eq!(updated.status, MemoryStatus::Updated);
        assert!(store
            .recent(10)
            .unwrap()
            .iter()
            .any(|memory| memory.id == id));
        assert!(store
            .search_fts("Updated content", None, 10)
            .unwrap()
            .iter()
            .any(|memory| memory.id == id));
    }

    #[test]
    fn test_list_topics() {
        let store = SqliteStore::in_memory().unwrap();
        store
            .store(test_memory("rust", "ownership", Importance::High))
            .unwrap();
        store
            .store(test_memory("rust", "borrowing", Importance::Medium))
            .unwrap();
        store
            .store(test_memory("python", "decorators", Importance::Low))
            .unwrap();

        let topics = store.list_topics().unwrap();
        assert_eq!(topics.len(), 2);
        // rust has 2 entries, should come first
        assert_eq!(topics[0], "rust");
        assert_eq!(topics[1], "python");
    }

    #[test]
    fn test_fts_search() {
        let store = SqliteStore::in_memory().unwrap();
        store
            .store(test_memory("rust", "ownership rules", Importance::High))
            .unwrap();
        store
            .store(test_memory("rust", "borrow checker", Importance::Medium))
            .unwrap();
        store
            .store(test_memory("python", "decorators", Importance::Low))
            .unwrap();

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
        store
            .store(test_memory("test", "normal memory", Importance::Low))
            .unwrap();

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
        store
            .conn
            .execute(
                "UPDATE memories SET created_at = ?1, last_accessed = ?1 WHERE id IN (?2, ?3)",
                rusqlite::params![past, med_id, low_id],
            )
            .unwrap();

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
        let id_low = store
            .store(test_memory("test", "forgettable", Importance::Low))
            .unwrap();
        store
            .conn
            .execute(
                "UPDATE memories SET strength = 0.05 WHERE id = ?1",
                rusqlite::params![id_low],
            )
            .unwrap();

        // STM + Medium importance + low strength -> should be pruned
        let id_med = store
            .store(test_memory(
                "test",
                "somewhat forgettable",
                Importance::Medium,
            ))
            .unwrap();
        store
            .conn
            .execute(
                "UPDATE memories SET strength = 0.05 WHERE id = ?1",
                rusqlite::params![id_med],
            )
            .unwrap();

        // LTM + Critical importance + low strength -> should NOT be pruned
        let id_crit = store
            .store(test_memory(
                "test",
                "critical never forget",
                Importance::Critical,
            ))
            .unwrap();
        store
            .conn
            .execute(
                "UPDATE memories SET strength = 0.05 WHERE id = ?1",
                rusqlite::params![id_crit],
            )
            .unwrap();

        // LTM + High importance + low strength -> should NOT be pruned (importance=high)
        let id_high = store
            .store(test_memory("test", "important stuff", Importance::High))
            .unwrap();
        store
            .conn
            .execute(
                "UPDATE memories SET strength = 0.05 WHERE id = ?1",
                rusqlite::params![id_high],
            )
            .unwrap();

        let pruned = store.prune(0.1).unwrap();
        assert_eq!(pruned, 2); // low and medium STM

        // Critical and High should still exist
        assert!(store.get(&id_crit).is_ok());
        assert!(store.get(&id_high).is_ok());

        // Low and Medium should be gone
        assert!(store.get(&id_low).is_err());
        assert!(store.get(&id_med).is_err());
    }

    fn test_memory_with_content(
        topic: &str,
        summary: &str,
        content: &str,
        importance: Importance,
    ) -> Memory {
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
            canonical_id: None,
            support_count: 1,
            merge_count: 0,
            dedup_confidence: 1.0,
            source_diversity: 1.0,
            contradiction_score: 0.0,
            related_ids: vec![],
            concept_ids: vec![],
            status: MemoryStatus::default(),
            embedding: None,
            tier: MemoryTier::Warm,
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
        store
            .conn
            .execute(
                "UPDATE memories SET created_at = ?1 WHERE id = ?2",
                rusqlite::params![old_date, id1],
            )
            .unwrap();

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
    fn test_store_with_dedup_grayzone_reuses_established_canonical() {
        let store = SqliteStore::in_memory().unwrap();

        let canonical = test_memory_with_content(
            "docker",
            "docker setup",
            "docker compose local development stack keeps api database cache queue search metrics logging stable safe deterministic reusable portable observable maintainable candidate old",
            Importance::High,
        );
        let canonical_id = store.store_with_dedup(canonical, 0.95, 7).unwrap();

        let grayzone = test_memory_with_content(
            "docker",
            "docker setup variant",
            "docker compose local development stack keeps api database cache queue search metrics logging stable safe deterministic reusable portable observable maintainable candidate new",
            Importance::High,
        );
        let result_id = store.store_with_dedup(grayzone, 0.95, 7).unwrap();

        assert_eq!(
            result_id, canonical_id,
            "gray-zone dedup should reuse canonical"
        );

        let stats = store.stats().unwrap();
        assert_eq!(
            stats.total_memories, 1,
            "gray-zone reuse should not create a raw memory"
        );

        let evidence = store.list_memory_evidence(&canonical_id, 10).unwrap();
        assert_eq!(
            evidence.len(),
            2,
            "canonical evidence should record both memories"
        );
        assert!(
            store
                .get(&canonical_id)
                .unwrap()
                .content
                .contains("[merged on"),
            "canonical content should record provenance from gray-zone merge"
        );
    }

    #[test]
    #[serial_test::serial(global_state)]
    fn test_store_with_dedup_intelligent_merge_e2e_records_provenance() {
        let _guard = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        let _restore_config = EnvRestore {
            key: "REIN_CONFIG",
            value: std::env::var("REIN_CONFIG").ok(),
        };
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("rein.toml");
        std::fs::write(
            &config_path,
            r#"
[intelligent_merge]
enabled = true
"#,
        )
        .unwrap();
        std::env::set_var("REIN_CONFIG", &config_path);

        let _mock = crate::extract::intelligent_merge::test_hook::install(Box::new(
            |_existing, _incoming| {
                Some(crate::extract::intelligent_merge::VerdictResult {
                    verdict: crate::extract::intelligent_merge::InsertionVerdict::Merge,
                    synthesized: Some(
                        "docker compose stack uses api database cache queue search metrics with synthesized provenance"
                            .to_string(),
                    ),
                    reasoning: Some("complementary docker stack notes".to_string()),
                })
            },
        ));

        let store = SqliteStore::in_memory().unwrap();
        let existing = test_memory_with_content(
            "docker",
            "docker setup",
            "docker compose local development stack keeps api database cache queue search metrics logging stable safe deterministic reusable portable observable maintainable candidate old",
            Importance::High,
        );
        let existing_id = store.store_with_dedup(existing, 1.10, 7).unwrap();

        let incoming = test_memory_with_content(
            "docker",
            "docker setup extra",
            "docker compose local development stack keeps api database cache queue search metrics logging stable safe deterministic reusable portable observable maintainable candidate new",
            Importance::High,
        );
        let result_id = store.store_with_dedup(incoming, 1.10, 7).unwrap();

        assert_eq!(result_id, existing_id);
        let merged = store.get(&existing_id).unwrap();
        assert!(merged.content.contains("synthesized provenance"));
        assert!(
            !merged.content.contains("[merged on"),
            "intelligent merge should use synthesized content instead of mechanical append"
        );

        let evidence = store.list_memory_evidence(&existing_id, 10).unwrap();
        assert!(
            evidence
                .iter()
                .any(|item| item.content.contains("candidate old")),
            "pre-merge existing memory should be preserved as evidence"
        );
        let decisions = store.list_dedup_decisions(Some(&existing_id), 10).unwrap();
        let verdict = decisions
            .iter()
            .find(|decision| decision.operator == "llm_verdict")
            .expect("llm verdict should be recorded");
        assert_eq!(verdict.relation, DedupRelation::Update);
        assert!(verdict.reason.contains("complementary docker stack notes"));
        assert!(verdict
            .merged_summary
            .as_deref()
            .unwrap_or("")
            .contains("synthesized provenance"));
    }

    #[test]
    fn test_store_with_dedup_uses_cached_cluster_hint() {
        let store = SqliteStore::in_memory().unwrap();

        let mut existing = test_memory_with_content(
            "docker",
            "docker stack",
            "docker compose local development stack service api database cache queue search metrics logging stable deterministic reusable maintainable portable observable local safe stable reusable maintainable",
            Importance::High,
        );
        existing.cluster_id = Some(7);
        let existing_id = store.store(existing).unwrap();

        let mut embedding = vec![0.0f32; 3072];
        embedding[0] = 1.0;
        let _ = crate::store::vec::insert_embedding(store.conn(), &existing_id, &embedding);

        let candidate = test_memory_with_content(
            "docker",
            "docker stack variant",
            "docker compose local development stack service api database cache queue search metrics logging stable deterministic reusable maintainable portable observable testing staged portable observable efficient",
            Importance::High,
        );
        let enriched = crate::embed::prepend_metadata(
            &candidate.topic,
            &candidate.summary,
            &candidate.content,
        );
        let model = crate::config::ReinConfig::load()
            .unwrap_or_default()
            .embedding_model();
        let _ = crate::embed::EmbedCache::put(store.conn(), &enriched, &model, &embedding);

        let inferred_cluster = store.infer_cluster_from_cache(&candidate);
        assert_eq!(inferred_cluster, Some(7));

        assert_eq!(
            inferred_cluster,
            Some(7),
            "cached local embedding should infer the nearest cluster without a remote call"
        );
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
        let flag: i32 = store
            .conn
            .query_row(
                "SELECT needs_vec_dedup FROM memories WHERE id = ?1",
                rusqlite::params![&id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(flag, 1, "new memory should have needs_vec_dedup = 1");
    }

    #[test]
    fn test_needs_vec_dedup_schema() {
        // Verify the needs_vec_dedup column exists and defaults to 0 for direct store
        let store = SqliteStore::in_memory().unwrap();
        let mem = test_memory("test", "direct store", Importance::Medium);
        let id = store.store(mem).unwrap();

        let flag: i32 = store
            .conn
            .query_row(
                "SELECT needs_vec_dedup FROM memories WHERE id = ?1",
                rusqlite::params![&id],
                |row| row.get(0),
            )
            .unwrap();
        // Direct store (not via dedup) defaults to 0
        assert_eq!(flag, 0, "direct store should have needs_vec_dedup = 0");
    }

    #[test]
    fn test_consolidate_topics_atomic() {
        let store = SqliteStore::in_memory().unwrap();
        store
            .store(test_memory(
                "Docker Deployment",
                "compose",
                Importance::High,
            ))
            .unwrap();
        store
            .store(test_memory(
                "docker-deployment",
                "compose duplicate",
                Importance::High,
            ))
            .unwrap();

        let replacement = Memory {
            id: ulid::Ulid::new().to_string(),
            layer: MemoryLayer::LTM,
            topic: "Docker Deployment".to_string(),
            summary: "Docker deployment consolidated".to_string(),
            content: "Use docker compose and pin image tags.".to_string(),
            keywords: vec!["docker".to_string(), "deployment".to_string()],
            importance: Importance::High,
            source: Source::Manual,
            strength: 1.0,
            decay_lambda: 0.02,
            access_count: 2,
            superseded_by: None,
            canonical_id: None,
            support_count: 2,
            merge_count: 1,
            dedup_confidence: 1.0,
            source_diversity: 2.0,
            contradiction_score: 0.0,
            related_ids: vec![],
            concept_ids: vec![],
            status: MemoryStatus::default(),
            embedding: None,
            tier: MemoryTier::Hot,
            cluster_id: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_accessed: Utc::now(),
        };

        let old = store
            .consolidate_topics_atomic(&["docker-deployment".to_string()], replacement)
            .unwrap();

        // Both original memories were stored with normalized topic "docker-deployment"
        assert_eq!(old.len(), 2);
        // Replacement topic "Docker Deployment" is normalized to "docker-deployment" at store time
        let consolidated = store.get_by_topic("docker-deployment").unwrap();
        assert_eq!(consolidated.len(), 1);
    }

    #[test]
    fn consolidate_by_ids_atomic_strips_related_ids_from_survivors() {
        // Verifies that when consolidate_by_ids_atomic deletes M1 + M2, any OTHER
        // memory with those IDs in its related_ids gets cleanly scrubbed — preventing
        // the dangling-ref corruption that was the top CRITICAL in the v0.20 audit.
        let store = SqliteStore::in_memory().unwrap();
        let m1_id = store
            .store(test_memory("topic-a", "first", Importance::Medium))
            .unwrap();
        let m2_id = store
            .store(test_memory("topic-a", "second", Importance::Medium))
            .unwrap();

        let mut survivor = test_memory("topic-b", "survivor", Importance::Medium);
        survivor.related_ids = vec![m1_id.clone(), m2_id.clone(), "untouched".to_string()];
        let survivor_id = store.store(survivor).unwrap();

        let replacement = test_memory("topic-a", "merged", Importance::High);

        store
            .consolidate_by_ids_atomic(&[m1_id.clone(), m2_id.clone()], replacement)
            .unwrap();

        let after = store.get(&survivor_id).unwrap();
        assert!(
            !after.related_ids.contains(&m1_id),
            "survivor.related_ids must not retain deleted {m1_id}"
        );
        assert!(
            !after.related_ids.contains(&m2_id),
            "survivor.related_ids must not retain deleted {m2_id}"
        );
        assert!(
            after.related_ids.iter().any(|id| id == "untouched"),
            "unrelated related_ids entries must be preserved"
        );
    }

    #[test]
    fn test_memory_evidence_roundtrip() {
        let store = SqliteStore::in_memory().unwrap();
        let canonical_id = store
            .store(test_memory("docker", "compose notes", Importance::High))
            .unwrap();
        let fetched = store.get(&canonical_id).unwrap();

        let evidence_id = store
            .snapshot_memory_as_evidence(&canonical_id, &fetched)
            .unwrap();
        let evidence = store.list_memory_evidence(&canonical_id, 10).unwrap();

        assert!(!evidence_id.is_empty());
        assert!(!evidence.is_empty());
        assert!(evidence
            .iter()
            .any(|item| item.canonical_id == canonical_id));
    }

    #[test]
    fn test_get_reflects_canonical_state_stats() {
        let store = SqliteStore::in_memory().unwrap();
        let canonical_id = store
            .store(test_memory("docker", "compose notes", Importance::High))
            .unwrap();
        let fetched = store.get(&canonical_id).unwrap();
        store
            .snapshot_memory_as_evidence(&canonical_id, &fetched)
            .unwrap();

        let refreshed = store.get(&canonical_id).unwrap();
        assert_eq!(
            refreshed.canonical_id.as_deref(),
            Some(canonical_id.as_str())
        );
        assert_eq!(refreshed.support_count, 2);
        assert_eq!(refreshed.merge_count, 1);
        assert!(refreshed.source_diversity >= 1.0);
    }

    #[test]
    fn test_dedup_decision_roundtrip() {
        let store = SqliteStore::in_memory().unwrap();
        let winner = store
            .store(test_memory("docker", "winner", Importance::High))
            .unwrap();
        let loser = store
            .store(test_memory("docker", "loser", Importance::High))
            .unwrap();

        let decision_id = store
            .record_dedup_decision(DedupDecision {
                id: String::new(),
                winner_id: Some(winner.clone()),
                loser_id: Some(loser.clone()),
                canonical_id: Some(winner.clone()),
                lexical_score: Some(0.91),
                embedding_score: None,
                relation: DedupRelation::Duplicate,
                confidence: 0.91,
                reason: "test".to_string(),
                operator: "manual".to_string(),
                reversible: true,
                merged_summary: Some("winner".to_string()),
                novel_facts: vec!["fact".to_string()],
                conflict_detected: false,
                payload: None,
                created_at: Utc::now(),
            })
            .unwrap();

        let decisions = store.list_dedup_decisions(Some(&winner), 10).unwrap();
        assert!(!decision_id.is_empty());
        assert_eq!(decisions.len(), 1);
        assert_eq!(decisions[0].relation, DedupRelation::Duplicate);
    }

    #[test]
    fn test_list_canonical_memories_filters_superseded() {
        let store = SqliteStore::in_memory().unwrap();
        let winner = store
            .store(test_memory("docker", "winner", Importance::High))
            .unwrap();
        let loser = store
            .store(test_memory("docker", "loser", Importance::High))
            .unwrap();

        store.mark_superseded(&loser, &winner).unwrap();

        let canonicals = store.list_canonical_memories(10).unwrap();
        assert_eq!(canonicals.len(), 1);
        assert_eq!(canonicals[0].id, winner);
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

    #[test]
    fn test_stm_to_ltm_promotion_uses_survival_curve_threshold() {
        let store = SqliteStore::in_memory().unwrap();
        let mut mem = test_memory("test", "cluster-guided promotion", Importance::Medium);
        mem.cluster_id = Some(7);
        let id = store.store(mem).unwrap();

        let curve = crate::search::survival::SurvivalCurve {
            steps: vec![(0.0, 1.0), (14.0, 0.4)],
            event_count: 20,
            total_count: 25,
            median_survival: Some(14.0),
        };
        store
            .conn()
            .execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?1, ?2)",
                rusqlite::params!["survival_curve:7", serde_json::to_string(&curve).unwrap()],
            )
            .unwrap();

        for _ in 0..4 {
            store.record_access(&id).unwrap();
        }

        let fetched = store.get(&id).unwrap();
        assert_eq!(fetched.layer, MemoryLayer::LTM);
        assert_eq!(fetched.access_count, 4);
    }
}
