use crate::types::ReinResult;
use rusqlite::Connection;
use std::sync::Once;

static SQLITE_VEC_INIT: Once = Once::new();

/// Initialize sqlite-vec extension. Must be called before creating any connection.
///
/// # Safety
/// `sqlite3_vec_init` has the signature expected by `sqlite3_auto_extension`:
///   `fn(*mut sqlite3, *mut *mut c_char, *const sqlite3_api_routines) -> c_int`
/// The transmute converts the typed function pointer to the `Option<unsafe extern "C" fn()>`
/// that `sqlite3_auto_extension` accepts. This is the standard pattern for loading
/// SQLite extensions and is used by the sqlite-vec and ICM projects.
pub fn init_sqlite_vec() {
    SQLITE_VEC_INIT.call_once(|| unsafe {
        rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
            sqlite_vec::sqlite3_vec_init as *const (),
        )));
    });
}

/// Create all tables, indexes, triggers, and virtual tables.
/// `dims` is the embedding dimension (e.g., 3072 for gemini-embedding-001).
pub fn init_schema(conn: &Connection, dims: usize) -> ReinResult<()> {
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            layer TEXT NOT NULL CHECK(layer IN ('LTM', 'STM')),
            topic TEXT NOT NULL,
            summary TEXT NOT NULL,
            content TEXT NOT NULL,
            keywords TEXT NOT NULL DEFAULT '[]',
            importance TEXT NOT NULL CHECK(importance IN ('critical', 'high', 'medium', 'low')),
            source TEXT NOT NULL CHECK(source IN ('manual', 'hook', 'migration', 'supermemory')),
            strength REAL NOT NULL DEFAULT 1.0,
            decay_lambda REAL NOT NULL,
            access_count INTEGER NOT NULL DEFAULT 0,
            superseded_by TEXT,
            related_ids TEXT NOT NULL DEFAULT '[]',
            concept_ids TEXT NOT NULL DEFAULT '[]',
            status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'updated', 'deprecated')),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_accessed TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_memories_topic ON memories(topic);
        CREATE INDEX IF NOT EXISTS idx_memories_layer ON memories(layer);
        CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance);
        CREATE INDEX IF NOT EXISTS idx_memories_strength ON memories(strength);
        CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at);
        CREATE INDEX IF NOT EXISTS idx_memories_last_accessed ON memories(last_accessed);

        -- NOTE: `id` is included as a searchable FTS column for join convenience
        -- (JOIN memories m ON m.id = f.id). Ideally id would not be indexed for
        -- search, but removing it requires switching to rowid-based joins and
        -- updating all triggers. Low impact; leaving as-is.
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            id, topic, summary, content, keywords,
            tokenize='unicode61'
        );

        -- Triggers to keep FTS in sync
        CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(id, topic, summary, content, keywords)
            VALUES (new.id, new.topic, new.summary, new.content, new.keywords);
        END;

        CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories
        WHEN old.content != new.content OR old.topic != new.topic
          OR old.summary != new.summary OR old.keywords != new.keywords
        BEGIN
            DELETE FROM memories_fts WHERE id = old.id;
            INSERT INTO memories_fts(id, topic, summary, content, keywords)
            VALUES (new.id, new.topic, new.summary, new.content, new.keywords);
        END;

        CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
            DELETE FROM memories_fts WHERE id = old.id;
        END;

        CREATE TABLE IF NOT EXISTS embed_cache (
            query_hash TEXT PRIMARY KEY,
            embedding BLOB,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        );

        CREATE TABLE IF NOT EXISTS memoirs (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            description TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS concepts (
            id TEXT PRIMARY KEY,
            memoir_id TEXT NOT NULL REFERENCES memoirs(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            definition TEXT NOT NULL,
            labels TEXT NOT NULL DEFAULT '[]',
            source_memory_ids TEXT NOT NULL DEFAULT '[]',
            confidence REAL NOT NULL DEFAULT 0.5,
            revision INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(memoir_id, name)
        );

        CREATE INDEX IF NOT EXISTS idx_concepts_memoir ON concepts(memoir_id);

        CREATE VIRTUAL TABLE IF NOT EXISTS concepts_fts USING fts5(
            id, name, definition, labels,
            tokenize='unicode61'
        );

        CREATE TRIGGER IF NOT EXISTS concepts_ai AFTER INSERT ON concepts BEGIN
            INSERT INTO concepts_fts(id, name, definition, labels)
            VALUES (new.id, new.name, new.definition, new.labels);
        END;

        CREATE TRIGGER IF NOT EXISTS concepts_au AFTER UPDATE ON concepts BEGIN
            DELETE FROM concepts_fts WHERE id = old.id;
            INSERT INTO concepts_fts(id, name, definition, labels)
            VALUES (new.id, new.name, new.definition, new.labels);
        END;

        CREATE TRIGGER IF NOT EXISTS concepts_ad AFTER DELETE ON concepts BEGIN
            DELETE FROM concepts_fts WHERE id = old.id;
        END;

        CREATE TABLE IF NOT EXISTS concept_links (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
            target_id TEXT NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
            relation TEXT NOT NULL,
            weight REAL NOT NULL DEFAULT 1.0,
            created_at TEXT NOT NULL,
            CHECK(source_id != target_id)
        );

        CREATE INDEX IF NOT EXISTS idx_links_source ON concept_links(source_id);
        CREATE INDEX IF NOT EXISTS idx_links_target ON concept_links(target_id);
        ",
    )?;

    // Vector table needs sqlite-vec loaded; dimension is dynamic
    let vec_sql = format!(
        "CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(
            id TEXT PRIMARY KEY,
            embedding float[{dims}] distance_metric=cosine
        );"
    );
    conn.execute_batch(&vec_sql)?;

    // Migrate: add status column if missing (for existing databases)
    let has_status: bool = conn.query_row(
        "SELECT COUNT(*) FROM pragma_table_info('memories') WHERE name='status'",
        [],
        |row| row.get::<_, i64>(0),
    ).unwrap_or(0) > 0;
    if !has_status {
        conn.execute_batch("ALTER TABLE memories ADD COLUMN status TEXT NOT NULL DEFAULT 'active'")?;
    }

    // Migrate: add concept_ids to memories if missing
    let has_concept_ids: bool = conn.query_row(
        "SELECT COUNT(*) FROM pragma_table_info('memories') WHERE name='concept_ids'",
        [],
        |row| row.get::<_, i64>(0),
    ).unwrap_or(0) > 0;
    if !has_concept_ids {
        conn.execute_batch("ALTER TABLE memories ADD COLUMN concept_ids TEXT NOT NULL DEFAULT '[]'").ok();
    }

    // Migrate: add source_memory_ids to concepts if missing
    let has_source_memory_ids: bool = conn.query_row(
        "SELECT COUNT(*) FROM pragma_table_info('concepts') WHERE name='source_memory_ids'",
        [],
        |row| row.get::<_, i64>(0),
    ).unwrap_or(0) > 0;
    if !has_source_memory_ids {
        conn.execute_batch("ALTER TABLE concepts ADD COLUMN source_memory_ids TEXT NOT NULL DEFAULT '[]'").ok();
    }

    // Migrate: add temporal fields to concept_links
    let has_valid_from: bool = conn.query_row(
        "SELECT COUNT(*) FROM pragma_table_info('concept_links') WHERE name='valid_from'",
        [], |row| row.get::<_, i64>(0),
    ).unwrap_or(0) > 0;
    if !has_valid_from {
        conn.execute_batch("ALTER TABLE concept_links ADD COLUMN valid_from TEXT")?;
        conn.execute_batch("ALTER TABLE concept_links ADD COLUMN valid_until TEXT")?;
        // Backfill: existing links get valid_from = created_at
        conn.execute_batch("UPDATE concept_links SET valid_from = created_at WHERE valid_from IS NULL").ok();
    }

    // Migrate: add last_episode_id to concepts
    let has_episode_id: bool = conn.query_row(
        "SELECT COUNT(*) FROM pragma_table_info('concepts') WHERE name='last_episode_id'",
        [], |row| row.get::<_, i64>(0),
    ).unwrap_or(0) > 0;
    if !has_episode_id {
        conn.execute_batch("ALTER TABLE concepts ADD COLUMN last_episode_id TEXT").ok();
    }

    // Create concept_revisions table (revision history)
    conn.execute_batch("
        CREATE TABLE IF NOT EXISTS concept_revisions (
            id TEXT PRIMARY KEY,
            concept_id TEXT NOT NULL,
            revision INTEGER NOT NULL,
            definition TEXT NOT NULL,
            confidence REAL NOT NULL,
            labels TEXT NOT NULL DEFAULT '[]',
            source_memory_ids TEXT NOT NULL DEFAULT '[]',
            episode_id TEXT,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_revisions_concept ON concept_revisions(concept_id);
        CREATE INDEX IF NOT EXISTS idx_revisions_created ON concept_revisions(created_at);
    ")?;

    // Create episodes table (session nodes in temporal graph)
    conn.execute_batch("
        CREATE TABLE IF NOT EXISTS episodes (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            outcome TEXT NOT NULL DEFAULT '',
            decisions TEXT NOT NULL DEFAULT '[]',
            concept_ids TEXT NOT NULL DEFAULT '[]',
            memory_ids TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_episodes_created ON episodes(created_at);
    ")?;

    // Index for temporal concept queries
    conn.execute_batch("CREATE INDEX IF NOT EXISTS idx_concepts_updated ON concepts(updated_at)").ok();

    // === Adaptive Engine tables (v0.5.0) ===

    // M1: Feedback event log (append-only event sourcing)
    conn.execute_batch("
        CREATE TABLE IF NOT EXISTS feedback_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
            event_type TEXT NOT NULL,
            request_id TEXT,
            memory_id TEXT,
            concept_id TEXT,
            query TEXT,
            query_type TEXT,
            topic TEXT,
            payload TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_fe_ts ON feedback_events(ts);
        CREATE INDEX IF NOT EXISTS idx_fe_type ON feedback_events(event_type);
        CREATE INDEX IF NOT EXISTS idx_fe_request ON feedback_events(request_id);
    ")?;

    // M1: Per-consumer offset tracking (each module tracks its own read position)
    conn.execute_batch("
        CREATE TABLE IF NOT EXISTS consumer_offsets (
            consumer TEXT PRIMARY KEY,
            last_event_id INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
        );
    ")?;

    // M5: Cold archive for compressed cold-tier memories
    conn.execute_batch("
        CREATE TABLE IF NOT EXISTS cold_archive (
            memory_id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            archived_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
        );
    ")?;

    // M5: Migrate: add tier column to memories
    let has_tier: bool = conn.query_row(
        "SELECT COUNT(*) FROM pragma_table_info('memories') WHERE name='tier'",
        [], |row| row.get::<_, i64>(0),
    ).unwrap_or(0) > 0;
    if !has_tier {
        conn.execute_batch("ALTER TABLE memories ADD COLUMN tier TEXT NOT NULL DEFAULT 'warm'").ok();
    }

    // M4: Migrate: add cluster_id column to memories
    let has_cluster_id: bool = conn.query_row(
        "SELECT COUNT(*) FROM pragma_table_info('memories') WHERE name='cluster_id'",
        [], |row| row.get::<_, i64>(0),
    ).unwrap_or(0) > 0;
    if !has_cluster_id {
        conn.execute_batch("ALTER TABLE memories ADD COLUMN cluster_id INTEGER").ok();
    }

    // Migrate FTS tokenizer from porter to unicode61 (for CJK support)
    migrate_fts_tokenizer(conn)?;

    Ok(())
}

/// Check if FTS tables use the old porter tokenizer and rebuild with unicode61.
fn migrate_fts_tokenizer(conn: &Connection) -> ReinResult<()> {
    // Check current tokenizer by querying FTS config
    let needs_rebuild: bool = conn
        .query_row(
            "SELECT sql FROM sqlite_master WHERE name = 'memories_fts'",
            [],
            |row| {
                let sql: String = row.get(0)?;
                Ok(sql.contains("porter"))
            },
        )
        .unwrap_or(false);

    if needs_rebuild {
        tracing::info!("rebuilding FTS index with unicode61 tokenizer (CJK support)");

        // Wrap in transaction — DROP+CREATE+INSERT should be atomic
        conn.execute_batch("BEGIN")?;
        let result = (|| -> ReinResult<()> {
            conn.execute_batch("DROP TABLE IF EXISTS memories_fts;")?;
            conn.execute_batch(
                "CREATE VIRTUAL TABLE memories_fts USING fts5(
                    id, topic, summary, content, keywords,
                    tokenize='unicode61'
                );"
            )?;
            conn.execute_batch(
                "INSERT INTO memories_fts(id, topic, summary, content, keywords)
                 SELECT id, topic, summary, content, keywords FROM memories;"
            )?;

            let concepts_needs_rebuild: bool = conn
                .query_row(
                    "SELECT sql FROM sqlite_master WHERE name = 'concepts_fts'",
                    [], |row| { let sql: String = row.get(0)?; Ok(sql.contains("porter")) },
                ).unwrap_or(false);

            if concepts_needs_rebuild {
                conn.execute_batch("DROP TABLE IF EXISTS concepts_fts;")?;
                conn.execute_batch(
                    "CREATE VIRTUAL TABLE concepts_fts USING fts5(
                        id, name, definition, labels,
                        tokenize='unicode61'
                    );"
                )?;
                conn.execute_batch(
                    "INSERT INTO concepts_fts(id, name, definition, labels)
                     SELECT id, name, definition, labels FROM concepts;"
                )?;
            }
            Ok(())
        })();
        match result {
            Ok(()) => { conn.execute_batch("COMMIT")?; }
            Err(e) => { let _ = conn.execute_batch("ROLLBACK"); return Err(e); }
        }

        tracing::info!("FTS index rebuilt with unicode61");
    }

    Ok(())
}

/// Check if the embedding model has changed since last run.
/// Returns true if model changed (requiring re-embedding).
/// Stores current model info in metadata for next startup.
pub fn check_embedding_model(
    conn: &Connection,
    model: &str,
    dims: usize,
) -> ReinResult<bool> {
    let stored_model: Option<String> = conn
        .query_row(
            "SELECT value FROM metadata WHERE key = 'embedding_model'",
            [],
            |row| row.get(0),
        )
        .ok();

    let stored_dims: Option<usize> = conn
        .query_row(
            "SELECT value FROM metadata WHERE key = 'embedding_dims'",
            [],
            |row| {
                let s: String = row.get(0)?;
                Ok(s.parse::<usize>().unwrap_or(0))
            },
        )
        .ok();

    let changed = match (&stored_model, stored_dims) {
        (Some(m), Some(d)) => m != model || d != dims,
        (None, None) => false, // First run, no change
        _ => true,             // Partial info, treat as changed
    };

    // Only write if first run or values changed — avoids unnecessary WAL writes on every startup
    let needs_write = stored_model.is_none() || changed;
    if needs_write {
        conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES ('embedding_model', ?1)",
            rusqlite::params![model],
        )?;
        conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES ('embedding_dims', ?1)",
            rusqlite::params![dims.to_string()],
        )?;
    }

    Ok(changed)
}

/// Rebuild vector index when embedding model changes.
/// Drops vec_memories and embed_cache, recreates with new dimensions.
pub fn rebuild_vector_index(conn: &Connection, dims: usize) -> ReinResult<()> {
    tracing::warn!("embedding model changed — rebuilding vector index (dims={dims})");
    conn.execute_batch("DROP TABLE IF EXISTS vec_memories;")?;
    conn.execute_batch("DELETE FROM embed_cache;")?;
    let vec_sql = format!(
        "CREATE VIRTUAL TABLE vec_memories USING vec0(
            id TEXT PRIMARY KEY,
            embedding float[{dims}] distance_metric=cosine
        );"
    );
    conn.execute_batch(&vec_sql)?;
    Ok(())
}
