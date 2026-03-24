use chrono::{DateTime, Utc};
use rusqlite::Connection;
use std::path::Path;
use std::str::FromStr;

use crate::extract::{check_dedup, DedupAction};
use crate::types::*;

use super::{fts, schema, vec};

/// SQLite-backed memory store with FTS5 and vector search.
///
/// Wraps `rusqlite::Connection` which is `!Send`. All database access should
/// happen on the thread that created the connection. The MCP server uses
/// `Mutex<SqliteStore>` with `SQLITE_OPEN_FULL_MUTEX` to allow safe cross-thread access.
pub struct SqliteStore {
    conn: Connection,
}

impl SqliteStore {
    /// Open or create a database at the given path.
    /// Uses SQLITE_OPEN_FULL_MUTEX for thread-safe access via serialized mode.
    pub fn new(path: &Path) -> ReinResult<Self> {
        schema::init_sqlite_vec();
        let conn = Connection::open_with_flags(
            path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_WRITE
                | rusqlite::OpenFlags::SQLITE_OPEN_CREATE
                | rusqlite::OpenFlags::SQLITE_OPEN_FULL_MUTEX,
        )?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;
        schema::init_schema(&conn)?;
        Ok(Self { conn })
    }

    /// Create an in-memory database for testing.
    pub fn in_memory() -> ReinResult<Self> {
        schema::init_sqlite_vec();
        let conn = Connection::open_in_memory()?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;
        schema::init_schema(&conn)?;
        Ok(Self { conn })
    }
    /// Access the underlying SQLite connection (for direct queries).
    pub fn conn(&self) -> &Connection {
        &self.conn
    }

    /// Get all memories in a topic (for dedup scanning).
    pub fn get_by_topic(&self, topic: &str) -> ReinResult<Vec<Memory>> {
        let mut stmt = self.conn.prepare("SELECT * FROM memories WHERE topic = ?1")?;
        let rows = stmt.query_map(rusqlite::params![topic], |row| {
            row_to_memory(row).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e))
            })
        })?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    /// Record an access to a memory (bumps access_count and last_accessed).
    /// Call this only when memories are returned to the user via recall, NOT on internal lookups.
    pub fn record_access(&self, id: &str) -> ReinResult<()> {
        let now = Utc::now();
        self.conn.execute(
            "UPDATE memories SET last_accessed = ?1, access_count = access_count + 1 WHERE id = ?2",
            rusqlite::params![now.to_rfc3339(), id],
        )?;
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
    let created_at_str: String = row.get("created_at").map_err(ReinError::Database)?;
    let updated_at_str: String = row.get("updated_at").map_err(ReinError::Database)?;
    let last_accessed_str: String = row.get("last_accessed").map_err(ReinError::Database)?;

    let layer = MemoryLayer::from_str(&layer_str)
        .map_err(|e| ReinError::Config(e))?;
    let importance = Importance::from_str(&importance_str)
        .map_err(|e| ReinError::Config(e))?;
    let source = Source::from_str(&source_str)
        .map_err(|e| ReinError::Config(e))?;

    let keywords: Vec<String> = serde_json::from_str(&keywords_json)?;
    let related_ids: Vec<String> = serde_json::from_str(&related_ids_json)?;

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
        embedding: None,
        created_at,
        updated_at,
        last_accessed,
    })
}

impl MemoryStore for SqliteStore {
    async fn store(&self, mut memory: Memory) -> ReinResult<String> {
        let id = ulid::Ulid::new().to_string();
        memory.id = id.clone();
        let now = Utc::now();
        memory.created_at = now;
        memory.updated_at = now;
        memory.last_accessed = now;

        let keywords_json = serde_json::to_string(&memory.keywords)?;
        let related_ids_json = serde_json::to_string(&memory.related_ids)?;

        // Store layer as uppercase for SQL CHECK constraint
        let layer_db = match memory.layer {
            MemoryLayer::LTM => "LTM",
            MemoryLayer::STM => "STM",
        };

        self.conn.execute(
            "INSERT INTO memories (id, layer, topic, summary, content, keywords, importance, source,
             strength, decay_lambda, access_count, superseded_by, related_ids,
             created_at, updated_at, last_accessed)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)",
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
                memory.created_at.to_rfc3339(),
                memory.updated_at.to_rfc3339(),
                memory.last_accessed.to_rfc3339(),
            ],
        )?;

        if let Some(ref emb) = memory.embedding {
            vec::insert_embedding(&self.conn, &id, emb)?;
        }

        Ok(id)
    }

    async fn get(&self, id: &str) -> ReinResult<Memory> {
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

    async fn update(&self, memory: &Memory) -> ReinResult<()> {
        let keywords_json = serde_json::to_string(&memory.keywords)?;
        let related_ids_json = serde_json::to_string(&memory.related_ids)?;
        let now = Utc::now();

        let layer_db = match memory.layer {
            MemoryLayer::LTM => "LTM",
            MemoryLayer::STM => "STM",
        };

        let rows = self.conn.execute(
            "UPDATE memories SET layer=?1, topic=?2, summary=?3, content=?4, keywords=?5,
             importance=?6, source=?7, strength=?8, decay_lambda=?9, access_count=?10,
             superseded_by=?11, related_ids=?12, updated_at=?13
             WHERE id=?14",
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

        Ok(())
    }

    async fn delete(&self, id: &str) -> ReinResult<()> {
        let rows = self
            .conn
            .execute("DELETE FROM memories WHERE id = ?1", rusqlite::params![id])?;
        if rows == 0 {
            return Err(ReinError::NotFound(format!("memory {id} not found")));
        }
        // FTS cleanup handled by trigger; clean up vector table
        let _ = vec::delete_embedding(&self.conn, id);
        Ok(())
    }

    async fn search_fts(
        &self,
        query: &str,
        topic: Option<&str>,
        limit: usize,
    ) -> ReinResult<Vec<Memory>> {
        let results = fts::search_fts(&self.conn, query, topic, limit)?;
        Ok(results.into_iter().map(|(m, _)| m).collect())
    }

    async fn search_vec(
        &self,
        embedding: &[f32],
        topic: Option<&str>,
        limit: usize,
    ) -> ReinResult<Vec<Memory>> {
        let results = vec::search_vec(&self.conn, embedding, limit)?;
        let mut memories = Vec::new();
        for (id, _distance) in results {
            match self.get(&id).await {
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
        Ok(memories)
    }

    async fn apply_decay(&self) -> ReinResult<u64> {
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
            "SELECT id, layer, decay_lambda, access_count, created_at, strength
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
                let created_at_str: String = row.get(4)?;
                let _strength: f64 = row.get(5)?;

                Ok((id, layer_str, decay_lambda, access_count, created_at_str))
            })?
            .filter_map(|r| r.ok())
            .filter_map(|(id, layer_str, decay_lambda, access_count, created_at_str)| {
                let created_at = DateTime::parse_from_rfc3339(&created_at_str).ok()?;
                let days = (now - created_at.with_timezone(&Utc)).num_seconds() as f64 / 86400.0;
                if days <= 0.0 {
                    return None;
                }

                let lambda_eff = decay_lambda / (1.0 + access_count as f64 * 0.2);
                let beta = if layer_str == "LTM" { 0.8 } else { 1.2 };
                let new_strength = (-lambda_eff * days.powf(beta)).exp();

                Some(DecayRow {
                    id,
                    new_strength,
                })
            })
            .collect();

        let count = updates.len() as u64;
        for u in &updates {
            self.conn.execute(
                "UPDATE memories SET strength = ?1 WHERE id = ?2",
                rusqlite::params![u.new_strength, u.id],
            )?;
        }

        // Record last decay time
        self.conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES ('last_decay_at', ?1)",
            rusqlite::params![now.to_rfc3339()],
        )?;

        Ok(count)
    }

    async fn prune(&self, threshold: f64) -> ReinResult<u64> {
        let rows = self.conn.execute(
            "DELETE FROM memories WHERE layer = 'STM' AND strength < ?1
             AND importance NOT IN ('critical', 'high')",
            rusqlite::params![threshold],
        )?;
        Ok(rows as u64)
    }

    async fn list_topics(&self) -> ReinResult<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT topic FROM memories GROUP BY topic ORDER BY COUNT(*) DESC",
        )?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    async fn consolidate(&self, topic: &str) -> ReinResult<Vec<Memory>> {
        // Use a transaction to ensure atomicity: either all delete or nothing
        self.conn.execute_batch("BEGIN TRANSACTION")?;

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
            let _ = self.conn.execute_batch("ROLLBACK");
            return Err(e.into());
        }

        self.conn.execute_batch("COMMIT")?;
        Ok(memories)
    }

    async fn stats(&self) -> ReinResult<StoreStats> {
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

        Ok(StoreStats {
            total_memories,
            ltm_count,
            stm_count,
            topic_count,
            avg_strength,
        })
    }

    async fn health(&self, topic: Option<&str>) -> ReinResult<Vec<HealthReport>> {
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
    /// Store a memory with deduplication logic.
    ///
    /// Checks for existing similar memories using FTS and Jaccard similarity.
    /// - If a similar memory exists within the time window, merges content into it.
    /// - If a similar memory exists but is older, supersedes it with the new memory.
    /// - Otherwise, creates a new memory.
    pub async fn store_with_dedup(
        &self,
        memory: Memory,
        similarity_threshold: f32,
        time_window_days: i64,
    ) -> ReinResult<String> {
        match check_dedup(self, &memory.topic, &memory.content, similarity_threshold, time_window_days).await? {
            DedupAction::CreateNew => {
                self.store(memory).await
            }
            DedupAction::MergeInto(existing_id) => {
                // Update existing memory: append content, refresh summary/keywords, boost strength
                if let Ok(mut existing) = self.get(&existing_id).await {
                    existing.content = format!("{}\n\n{}", existing.content, memory.content);
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
                    }
                    existing.strength = (existing.strength + 0.2).min(1.0);
                    existing.updated_at = chrono::Utc::now();
                    self.update(&existing).await?;
                    Ok(existing_id)
                } else {
                    self.store(memory).await
                }
            }
            DedupAction::Supersede(old_id) => {
                let new_id = self.store(memory).await?;
                // Mark old memory as superseded
                self.mark_superseded(&old_id, &new_id).await?;
                Ok(new_id)
            }
        }
    }

    /// Mark an old memory as superseded by a new one.
    pub async fn mark_superseded(&self, old_id: &str, new_id: &str) -> ReinResult<()> {
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
            embedding: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_accessed: Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_store_and_get() {
        let store = SqliteStore::in_memory().unwrap();
        let mem = test_memory("rust", "ownership rules", Importance::High);
        let original_summary = mem.summary.clone();
        let original_topic = mem.topic.clone();

        let id = store.store(mem).await.unwrap();
        let fetched = store.get(&id).await.unwrap();

        assert_eq!(fetched.id, id);
        assert_eq!(fetched.summary, original_summary);
        assert_eq!(fetched.topic, original_topic);
        assert_eq!(fetched.layer, MemoryLayer::LTM);
        assert_eq!(fetched.importance, Importance::High);
        assert_eq!(fetched.access_count, 0); // get() is now read-only, no side effects
    }

    #[tokio::test]
    async fn test_delete() {
        let store = SqliteStore::in_memory().unwrap();
        let mem = test_memory("rust", "borrow checker", Importance::Medium);
        let id = store.store(mem).await.unwrap();

        store.delete(&id).await.unwrap();
        let result = store.get(&id).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ReinError::NotFound(_)));
    }

    #[tokio::test]
    async fn test_update() {
        let store = SqliteStore::in_memory().unwrap();
        let mem = test_memory("rust", "lifetimes", Importance::Medium);
        let id = store.store(mem).await.unwrap();

        let mut fetched = store.get(&id).await.unwrap();
        fetched.content = "Updated content about lifetimes".to_string();
        store.update(&fetched).await.unwrap();

        let updated = store.get(&id).await.unwrap();
        assert_eq!(updated.content, "Updated content about lifetimes");
    }

    #[tokio::test]
    async fn test_list_topics() {
        let store = SqliteStore::in_memory().unwrap();
        store
            .store(test_memory("rust", "ownership", Importance::High))
            .await
            .unwrap();
        store
            .store(test_memory("rust", "borrowing", Importance::Medium))
            .await
            .unwrap();
        store
            .store(test_memory("python", "decorators", Importance::Low))
            .await
            .unwrap();

        let topics = store.list_topics().await.unwrap();
        assert_eq!(topics.len(), 2);
        // rust has 2 entries, should come first
        assert_eq!(topics[0], "rust");
        assert_eq!(topics[1], "python");
    }

    #[tokio::test]
    async fn test_fts_search() {
        let store = SqliteStore::in_memory().unwrap();
        store
            .store(test_memory("rust", "ownership rules", Importance::High))
            .await
            .unwrap();
        store
            .store(test_memory("rust", "borrow checker", Importance::Medium))
            .await
            .unwrap();
        store
            .store(test_memory("python", "decorators", Importance::Low))
            .await
            .unwrap();

        let results = store.search_fts("ownership", None, 10).await.unwrap();
        assert!(!results.is_empty());
        assert!(results.iter().any(|m| m.summary.contains("ownership")));
    }

    #[tokio::test]
    async fn test_fts_sanitize() {
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

    #[tokio::test]
    async fn test_fts_injection() {
        let store = SqliteStore::in_memory().unwrap();
        store
            .store(test_memory("test", "normal memory", Importance::Low))
            .await
            .unwrap();

        // Should not crash on malicious input
        let result = store
            .search_fts("\" OR 1=1; DROP TABLE memories; --", None, 10)
            .await;
        assert!(result.is_ok());

        let result = store
            .search_fts("***^^^\"\"\"", None, 10)
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_apply_decay() {
        let store = SqliteStore::in_memory().unwrap();

        // Store memories with different importance
        let mut critical = test_memory("test", "critical info", Importance::Critical);
        critical.created_at = Utc::now() - chrono::Duration::days(30);
        store.store(critical).await.unwrap();

        let mut medium = test_memory("test", "medium info", Importance::Medium);
        medium.created_at = Utc::now() - chrono::Duration::days(30);
        let med_id = store.store(medium).await.unwrap();

        let mut low = test_memory("test", "low info", Importance::Low);
        low.created_at = Utc::now() - chrono::Duration::days(30);
        let low_id = store.store(low).await.unwrap();

        // Manually set created_at in the past so decay has effect
        store.conn.execute(
            "UPDATE memories SET created_at = ?1 WHERE id IN (?2, ?3)",
            rusqlite::params![
                (Utc::now() - chrono::Duration::days(30)).to_rfc3339(),
                med_id,
                low_id
            ],
        ).unwrap();

        let count = store.apply_decay().await.unwrap();
        assert!(count > 0);

        // Verify strength was reduced for non-critical
        let med = store.get(&med_id).await.unwrap();
        assert!(med.strength < 1.0, "Medium memory strength should decay");

        let low_mem = store.get(&low_id).await.unwrap();
        assert!(low_mem.strength < 1.0, "Low memory strength should decay");
    }

    #[tokio::test]
    async fn test_prune() {
        let store = SqliteStore::in_memory().unwrap();

        // STM + Low importance + low strength -> should be pruned
        let id_low = store
            .store(test_memory("test", "forgettable", Importance::Low))
            .await
            .unwrap();
        store.conn.execute(
            "UPDATE memories SET strength = 0.05 WHERE id = ?1",
            rusqlite::params![id_low],
        ).unwrap();

        // STM + Medium importance + low strength -> should be pruned
        let id_med = store
            .store(test_memory("test", "somewhat forgettable", Importance::Medium))
            .await
            .unwrap();
        store.conn.execute(
            "UPDATE memories SET strength = 0.05 WHERE id = ?1",
            rusqlite::params![id_med],
        ).unwrap();

        // LTM + Critical importance + low strength -> should NOT be pruned
        let id_crit = store
            .store(test_memory("test", "critical never forget", Importance::Critical))
            .await
            .unwrap();
        store.conn.execute(
            "UPDATE memories SET strength = 0.05 WHERE id = ?1",
            rusqlite::params![id_crit],
        ).unwrap();

        // LTM + High importance + low strength -> should NOT be pruned (importance=high)
        let id_high = store
            .store(test_memory("test", "important stuff", Importance::High))
            .await
            .unwrap();
        store.conn.execute(
            "UPDATE memories SET strength = 0.05 WHERE id = ?1",
            rusqlite::params![id_high],
        ).unwrap();

        let pruned = store.prune(0.1).await.unwrap();
        assert_eq!(pruned, 2); // low and medium STM

        // Critical and High should still exist
        assert!(store.get(&id_crit).await.is_ok());
        assert!(store.get(&id_high).await.is_ok());

        // Low and Medium should be gone
        assert!(store.get(&id_low).await.is_err());
        assert!(store.get(&id_med).await.is_err());
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
            embedding: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_accessed: Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_store_with_dedup_create() {
        let store = SqliteStore::in_memory().unwrap();
        let mem1 = test_memory_with_content(
            "rust",
            "ownership rules",
            "Rust ownership rules are fundamental to memory safety",
            Importance::High,
        );
        let id1 = store.store_with_dedup(mem1, 0.85, 7).await.unwrap();

        let mem2 = test_memory_with_content(
            "rust",
            "async programming",
            "Async programming in Rust uses futures and tokio runtime",
            Importance::Medium,
        );
        let id2 = store.store_with_dedup(mem2, 0.85, 7).await.unwrap();

        // Both should exist as separate memories (different content)
        assert_ne!(id1, id2);
        let stats = store.stats().await.unwrap();
        assert_eq!(stats.total_memories, 2);
    }

    #[tokio::test]
    async fn test_store_with_dedup_merge() {
        let store = SqliteStore::in_memory().unwrap();

        let mem1 = test_memory_with_content(
            "rust",
            "ownership rules",
            "Rust ownership rules are fundamental to memory safety in systems programming",
            Importance::High,
        );
        let id1 = store.store_with_dedup(mem1, 0.85, 7).await.unwrap();

        // Store very similar content (same words, minor addition)
        let mem2 = test_memory_with_content(
            "rust",
            "ownership rules updated",
            "Rust ownership rules are fundamental to memory safety in systems programming today",
            Importance::High,
        );
        let id2 = store.store_with_dedup(mem2, 0.85, 7).await.unwrap();

        // Should merge into existing (same id returned)
        assert_eq!(id1, id2);

        // Only one memory should exist
        let stats = store.stats().await.unwrap();
        assert_eq!(stats.total_memories, 1);

        // Content should be merged
        let merged = store.get(&id1).await.unwrap();
        assert!(merged.content.contains("\n\n"));
    }

    #[tokio::test]
    async fn test_store_with_dedup_supersede() {
        let store = SqliteStore::in_memory().unwrap();

        let mem1 = test_memory_with_content(
            "rust",
            "ownership rules",
            "Rust ownership rules are fundamental to memory safety in systems programming",
            Importance::High,
        );
        let id1 = store.store_with_dedup(mem1, 0.85, 7).await.unwrap();

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
            "Rust ownership rules are fundamental to memory safety in systems programming today",
            Importance::High,
        );
        let id2 = store.store_with_dedup(mem2, 0.85, 7).await.unwrap();

        // Should supersede (different id)
        assert_ne!(id1, id2);

        // Two memories should exist
        let stats = store.stats().await.unwrap();
        assert_eq!(stats.total_memories, 2);

        // Old memory should have superseded_by set
        let old = store.get(&id1).await.unwrap();
        assert_eq!(old.superseded_by, Some(id2.clone()));
    }
}
