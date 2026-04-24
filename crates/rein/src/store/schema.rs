use crate::types::ReinResult;
use rusqlite::Connection;
use std::sync::Once;

static SQLITE_VEC_INIT: Once = Once::new();

/// DDL for every trigger that hangs off the `memories` table.
///
/// Any migration that renames/drops `memories` MUST re-run this block —
/// SQLite tears down all triggers when the owning table is dropped, and
/// if any trigger is missing after the migration, its side-effects
/// (FTS population, `memory_canonical_state` backfill) silently stop
/// firing for every row inserted afterward.
const MEMORIES_TRIGGERS_SQL: &str = "\
    CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN \
        INSERT INTO memories_fts(id, topic, summary, content, keywords) \
        VALUES (new.id, new.topic, new.summary, new.content, new.keywords); \
    END; \
    CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories \
        WHEN old.content != new.content OR old.topic != new.topic \
          OR old.summary != new.summary OR old.keywords != new.keywords \
    BEGIN \
        DELETE FROM memories_fts WHERE id = old.id; \
        INSERT INTO memories_fts(id, topic, summary, content, keywords) \
        VALUES (new.id, new.topic, new.summary, new.content, new.keywords); \
    END; \
    CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN \
        DELETE FROM memories_fts WHERE id = old.id; \
    END; \
    CREATE TRIGGER IF NOT EXISTS memories_canonical_ai AFTER INSERT ON memories BEGIN \
        INSERT OR IGNORE INTO memory_canonical_state(memory_id, canonical_id) \
        VALUES (new.id, new.id); \
    END;";

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
            source TEXT NOT NULL CHECK(source IN ('manual', 'hook', 'migration', 'supermemory', 'proxy')),
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

        -- Triggers to keep FTS + canonical_state in sync.
        -- DDL duplicated as MEMORIES_TRIGGERS_SQL below; any `DROP TABLE memories`
        -- migration MUST re-run that const or canonical bookkeeping silently stops.
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

        CREATE TABLE IF NOT EXISTS memory_canonical_state (
            memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
            canonical_id TEXT REFERENCES memories(id) ON DELETE SET NULL,
            support_count INTEGER NOT NULL DEFAULT 1,
            merge_count INTEGER NOT NULL DEFAULT 0,
            dedup_confidence REAL NOT NULL DEFAULT 1.0,
            source_diversity REAL NOT NULL DEFAULT 1.0,
            contradiction_score REAL NOT NULL DEFAULT 0.0,
            last_merged_at TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_memory_canonical ON memory_canonical_state(canonical_id);

        CREATE TRIGGER IF NOT EXISTS memories_canonical_ai AFTER INSERT ON memories BEGIN
            INSERT OR IGNORE INTO memory_canonical_state(memory_id, canonical_id)
            VALUES (new.id, new.id);
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

        CREATE TABLE IF NOT EXISTS memory_evidence (
            id TEXT PRIMARY KEY,
            canonical_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
            memory_id TEXT,
            source_topic TEXT NOT NULL,
            summary TEXT NOT NULL,
            content TEXT NOT NULL,
            keywords TEXT NOT NULL DEFAULT '[]',
            source TEXT NOT NULL CHECK(source IN ('manual', 'hook', 'migration', 'supermemory', 'proxy')),
            created_at TEXT NOT NULL,
            imported_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_memory_evidence_canonical ON memory_evidence(canonical_id, imported_at DESC);
        CREATE INDEX IF NOT EXISTS idx_memory_evidence_memory_id ON memory_evidence(memory_id);

        CREATE TABLE IF NOT EXISTS dedup_decisions (
            id TEXT PRIMARY KEY,
            winner_id TEXT,
            loser_id TEXT,
            canonical_id TEXT,
            lexical_score REAL,
            embedding_score REAL,
            relation TEXT NOT NULL CHECK(relation IN ('duplicate', 'update', 'related', 'distinct')),
            confidence REAL NOT NULL DEFAULT 0.0,
            reason TEXT NOT NULL,
            operator TEXT NOT NULL DEFAULT 'auto',
            reversible INTEGER NOT NULL DEFAULT 1,
            merged_summary TEXT,
            novel_facts TEXT NOT NULL DEFAULT '[]',
            conflict_detected INTEGER NOT NULL DEFAULT 0,
            payload TEXT,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_dedup_decisions_canonical ON dedup_decisions(canonical_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_dedup_decisions_winner ON dedup_decisions(winner_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_dedup_decisions_loser ON dedup_decisions(loser_id, created_at DESC);

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
            CHECK(source_id != target_id),
            UNIQUE(source_id, target_id, relation)
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
    let has_status: bool = conn
        .query_row(
            "SELECT COUNT(*) FROM pragma_table_info('memories') WHERE name='status'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .unwrap_or(0)
        > 0;
    if !has_status {
        conn.execute_batch(
            "ALTER TABLE memories ADD COLUMN status TEXT NOT NULL DEFAULT 'active'",
        )?;
    }

    // Migrate: add concept_ids to memories if missing
    let has_concept_ids: bool = conn
        .query_row(
            "SELECT COUNT(*) FROM pragma_table_info('memories') WHERE name='concept_ids'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .unwrap_or(0)
        > 0;
    if !has_concept_ids {
        conn.execute_batch(
            "ALTER TABLE memories ADD COLUMN concept_ids TEXT NOT NULL DEFAULT '[]'",
        )
        .ok();
    }

    // Migrate: add source_memory_ids to concepts if missing
    let has_source_memory_ids: bool = conn
        .query_row(
            "SELECT COUNT(*) FROM pragma_table_info('concepts') WHERE name='source_memory_ids'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .unwrap_or(0)
        > 0;
    if !has_source_memory_ids {
        conn.execute_batch(
            "ALTER TABLE concepts ADD COLUMN source_memory_ids TEXT NOT NULL DEFAULT '[]'",
        )
        .ok();
    }

    // Migrate: add temporal fields to concept_links
    let has_valid_from: bool = conn
        .query_row(
            "SELECT COUNT(*) FROM pragma_table_info('concept_links') WHERE name='valid_from'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .unwrap_or(0)
        > 0;
    if !has_valid_from {
        conn.execute_batch("ALTER TABLE concept_links ADD COLUMN valid_from TEXT")?;
        conn.execute_batch("ALTER TABLE concept_links ADD COLUMN valid_until TEXT")?;
        // Backfill: existing links get valid_from = created_at
        conn.execute_batch(
            "UPDATE concept_links SET valid_from = created_at WHERE valid_from IS NULL",
        )
        .ok();
    }

    // Migrate: add last_episode_id to concepts
    let has_episode_id: bool = conn
        .query_row(
            "SELECT COUNT(*) FROM pragma_table_info('concepts') WHERE name='last_episode_id'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .unwrap_or(0)
        > 0;
    if !has_episode_id {
        conn.execute_batch("ALTER TABLE concepts ADD COLUMN last_episode_id TEXT")
            .ok();
    }

    // Migrate: add v0.24 ARS Concept Living Summary fields (three nullable
    // columns). Backfill is lazy — rows without a `living_summary` just
    // hold NULLs until their revision count crosses the refresh trigger
    // in `should_refresh_living_summary`.
    let has_living_summary: bool = conn
        .query_row(
            "SELECT COUNT(*) FROM pragma_table_info('concepts') WHERE name='living_summary'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .unwrap_or(0)
        > 0;
    if !has_living_summary {
        conn.execute_batch(
            "ALTER TABLE concepts ADD COLUMN living_summary TEXT;
             ALTER TABLE concepts ADD COLUMN living_summary_updated_at TEXT;
             ALTER TABLE concepts ADD COLUMN living_summary_source_revision INTEGER;",
        )
        .ok();
    }

    // Create concept_revisions table (revision history)
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS concept_revisions (
            id TEXT PRIMARY KEY,
            concept_id TEXT NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
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
    ",
    )?;

    // Create episodes table (session nodes in temporal graph)
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS episodes (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            outcome TEXT NOT NULL DEFAULT '',
            decisions TEXT NOT NULL DEFAULT '[]',
            primary_topics TEXT NOT NULL DEFAULT '[]',
            tags TEXT NOT NULL DEFAULT '[]',
            involved_agents TEXT NOT NULL DEFAULT '[]',
            important_paths TEXT NOT NULL DEFAULT '[]',
            temporal_keywords TEXT NOT NULL DEFAULT '[]',
            source_session_id TEXT,
            concept_ids TEXT NOT NULL DEFAULT '[]',
            memory_ids TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_episodes_created ON episodes(created_at);
        CREATE INDEX IF NOT EXISTS idx_episodes_source_session ON episodes(source_session_id);
    ",
    )?;

    conn.execute_batch("
        CREATE TABLE IF NOT EXISTS session_artifacts (
            id TEXT PRIMARY KEY,
            schema_version INTEGER NOT NULL DEFAULT 1,
            artifact_kind TEXT NOT NULL DEFAULT 'session',
            session_id TEXT,
            title TEXT,
            summary TEXT,
            source_agent TEXT,
            source_label TEXT,
            is_subagent INTEGER NOT NULL DEFAULT 0,
            started_at TEXT,
            ended_at TEXT,
            turn_count INTEGER NOT NULL DEFAULT 0,
            transcript_text TEXT NOT NULL,
            transcript_json TEXT,
            episode_id TEXT REFERENCES episodes(id) ON DELETE SET NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_session_artifacts_created ON session_artifacts(created_at);
        CREATE INDEX IF NOT EXISTS idx_session_artifacts_session_id ON session_artifacts(session_id);
        CREATE INDEX IF NOT EXISTS idx_session_artifacts_episode_id ON session_artifacts(episode_id);
    ")?;

    // Index for temporal concept queries
    conn.execute_batch("CREATE INDEX IF NOT EXISTS idx_concepts_updated ON concepts(updated_at)")
        .ok();

    // === Adaptive Engine tables (v0.5.0) ===

    // M1: Feedback event log (append-only event sourcing)
    conn.execute_batch(
        "
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
    ",
    )?;

    // M1: Per-consumer offset tracking (each module tracks its own read position)
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS consumer_offsets (
            consumer TEXT PRIMARY KEY,
            last_event_id INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
        );
    ",
    )?;

    // M5: Cold archive for compressed cold-tier memories
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS cold_archive (
            memory_id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            archived_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
        );
    ",
    )?;

    // M5: Migrate: add tier column to memories
    let has_tier: bool = conn
        .query_row(
            "SELECT COUNT(*) FROM pragma_table_info('memories') WHERE name='tier'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .unwrap_or(0)
        > 0;
    if !has_tier {
        conn.execute_batch("ALTER TABLE memories ADD COLUMN tier TEXT NOT NULL DEFAULT 'warm'")
            .ok();
    }

    // M4: Migrate: add cluster_id column to memories
    let has_cluster_id: bool = conn
        .query_row(
            "SELECT COUNT(*) FROM pragma_table_info('memories') WHERE name='cluster_id'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .unwrap_or(0)
        > 0;
    if !has_cluster_id {
        conn.execute_batch("ALTER TABLE memories ADD COLUMN cluster_id INTEGER")
            .ok();
    }

    // Migrate: add needs_vec_dedup flag for deferred embedding-based dedup
    let has_nvd: bool = conn
        .query_row(
            "SELECT COUNT(*) FROM pragma_table_info('memories') WHERE name='needs_vec_dedup'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .unwrap_or(0)
        > 0;
    if !has_nvd {
        conn.execute_batch(
            "ALTER TABLE memories ADD COLUMN needs_vec_dedup INTEGER NOT NULL DEFAULT 0",
        )
        .ok();
        // Backfill: mark all existing active memories for vec dedup sweep
        conn.execute_batch("UPDATE memories SET needs_vec_dedup = 1 WHERE status = 'active'")
            .ok();
    }

    migrate_episode_columns(conn)?;

    // Migrate: widen source CHECK to include 'proxy' (v0.9.4)
    // SQLite doesn't support ALTER CHECK, so we recreate the table if needed.
    // Skip for in-memory databases (fresh schema already includes 'proxy').
    let is_memory_db = conn
        .path()
        .map(|p| p == ":memory:" || p.is_empty())
        .unwrap_or(true);
    if !is_memory_db {
        migrate_source_check(conn)?;
    }

    // Migrate FTS tokenizer from porter to unicode61 (for CJK support)
    migrate_fts_tokenizer(conn)?;

    // Backfill canonical state for databases created before the trigger existed.
    conn.execute_batch(
        "INSERT OR IGNORE INTO memory_canonical_state(memory_id, canonical_id)
         SELECT id, id FROM memories;",
    )
    .ok();

    // M4 incremental: cluster centroids table (separate from AdaptiveState JSON to avoid bloat).
    // `dims` is stored so load_cluster_centroids can reject stale centroids after a model change.
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS cluster_centroids (
            cluster_id INTEGER PRIMARY KEY,
            centroid BLOB NOT NULL,
            cluster_version INTEGER NOT NULL DEFAULT 0,
            dims INTEGER NOT NULL DEFAULT 0
        );
    ",
    )?;

    // Gray-zone dedup pairs that still need a follow-up LLM verdict. Inserted
    // INSIDE the store transaction so a crash between COMMIT and the file-queue
    // enqueue can't lose the pair silently. Drained on startup and whenever the
    // live enqueue succeeds post-COMMIT.
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS pending_grayzone_jobs (
            id TEXT PRIMARY KEY,
            candidate_id TEXT NOT NULL,
            result_id TEXT NOT NULL,
            sim REAL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_pending_grayzone_created
            ON pending_grayzone_jobs(created_at);
    ",
    )?;

    // v0.23: canonical resummerize flag + audit log. Must run AFTER
    // migrate_source_check — that path recreates `memories` with a hardcoded
    // column list and would silently drop the new columns on the first pass
    // if the ALTER ran first. The no-backfill rule means even that race is
    // harmless (rows default to 0), but keeping the order correct avoids a
    // first-vs-second-boot drift in the schema.
    migrate_resummerize(conn)?;

    Ok(())
}

/// v0.23 resummerize migration: adds `memories.needs_resummerize` /
/// `memories.last_resummarized_at` and creates the `resummerize_runs` audit
/// table. Idempotent: safe to call on fresh DBs and on already-migrated DBs.
///
/// No backfill for `needs_resummerize` — rows default to 0. The flag flips
/// only on new `MergeInto` cap hits and is cleared by the resummerize worker;
/// we don't retroactively mark pre-migration canonicals as needing a sweep.
fn migrate_resummerize(conn: &Connection) -> ReinResult<()> {
    // ── memories.needs_resummerize ────────────────────────────────────────
    let has_needs: bool = conn
        .query_row(
            "SELECT COUNT(*) FROM pragma_table_info('memories') WHERE name='needs_resummerize'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .unwrap_or(0)
        > 0;
    if !has_needs {
        conn.execute_batch(
            "ALTER TABLE memories ADD COLUMN needs_resummerize INTEGER NOT NULL DEFAULT 0",
        )
        .ok();
    }

    // ── memories.last_resummarized_at ─────────────────────────────────────
    let has_last: bool = conn
        .query_row(
            "SELECT COUNT(*) FROM pragma_table_info('memories') WHERE name='last_resummarized_at'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .unwrap_or(0)
        > 0;
    if !has_last {
        conn.execute_batch("ALTER TABLE memories ADD COLUMN last_resummarized_at TEXT")
            .ok();
    }

    // ── memories.in_progress_resummerize_at ───────────────────────────────
    // Lease column for atomic claim-then-process (Codex audit H6). A
    // worker claims a row by setting this to the current timestamp; no
    // other worker may claim the same row until either the current holder
    // clears it (success/failure path) or the lease goes stale (> 5 min
    // old — see `STALE_CLAIM_TIMEOUT_SECS` in `ops/resummerize.rs`).
    let has_ipr: bool = conn
        .query_row(
            "SELECT COUNT(*) FROM pragma_table_info('memories') WHERE name='in_progress_resummerize_at'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .unwrap_or(0)
        > 0;
    if !has_ipr {
        conn.execute_batch("ALTER TABLE memories ADD COLUMN in_progress_resummerize_at TEXT")
            .ok();
    }

    // ── resummerize_runs audit table ──────────────────────────────────────
    // * `output_chars` / `output_hash` are nullable (missing on LLM failure).
    // * `violations` is a JSON array persisted as TEXT (see
    //   `store/resummerize_audit.rs::finish_resummerize_run`).
    // * `status` is free-form TEXT rather than a CHECK-constrained enum so
    //   adding a new terminal status in a future release doesn't require a
    //   schema migration. `ResummerizeRunStatus::from_str` is the authoritative
    //   enum boundary.
    // * ON DELETE CASCADE: when the canonical is purged (gc/forget), its
    //   audit rows go with it — we never surface orphaned audit entries.
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS resummerize_runs (
            id TEXT PRIMARY KEY,
            canonical_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
            input_evidence_count INTEGER NOT NULL DEFAULT 0,
            input_canonical_chars INTEGER NOT NULL DEFAULT 0,
            output_chars INTEGER,
            output_hash TEXT,
            target_bytes INTEGER NOT NULL,
            status TEXT NOT NULL,
            violations TEXT,
            error TEXT,
            llm_backend TEXT,
            created_at TEXT NOT NULL,
            finished_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_resummerize_runs_canonical
            ON resummerize_runs(canonical_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_resummerize_runs_status
            ON resummerize_runs(status, created_at DESC);
    ",
    )?;

    Ok(())
}

fn migrate_episode_columns(conn: &Connection) -> ReinResult<()> {
    let ensure_column = |name: &str, ddl: &str| {
        let has_column: bool = conn
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('episodes') WHERE name=?1",
                rusqlite::params![name],
                |row| row.get::<_, i64>(0),
            )
            .unwrap_or(0)
            > 0;
        if !has_column {
            conn.execute_batch(ddl).ok();
        }
    };

    ensure_column(
        "primary_topics",
        "ALTER TABLE episodes ADD COLUMN primary_topics TEXT NOT NULL DEFAULT '[]'",
    );
    ensure_column(
        "tags",
        "ALTER TABLE episodes ADD COLUMN tags TEXT NOT NULL DEFAULT '[]'",
    );
    ensure_column(
        "involved_agents",
        "ALTER TABLE episodes ADD COLUMN involved_agents TEXT NOT NULL DEFAULT '[]'",
    );
    ensure_column(
        "important_paths",
        "ALTER TABLE episodes ADD COLUMN important_paths TEXT NOT NULL DEFAULT '[]'",
    );
    ensure_column(
        "temporal_keywords",
        "ALTER TABLE episodes ADD COLUMN temporal_keywords TEXT NOT NULL DEFAULT '[]'",
    );
    ensure_column(
        "source_session_id",
        "ALTER TABLE episodes ADD COLUMN source_session_id TEXT",
    );
    conn.execute_batch(
        "CREATE INDEX IF NOT EXISTS idx_episodes_source_session ON episodes(source_session_id)",
    )
    .ok();

    Ok(())
}

/// Widen the source CHECK constraint to include 'proxy'.
/// Only runs if the current schema rejects 'proxy' inserts.
fn migrate_source_check(conn: &Connection) -> ReinResult<()> {
    // Side-effect free schema inspection is safer than probing with an INSERT.
    let create_sql: String = conn.query_row(
        "SELECT COALESCE(sql, '') FROM sqlite_master WHERE type = 'table' AND name = 'memories'",
        [],
        |row| row.get(0),
    )?;
    let needs_migration = !create_sql.to_lowercase().contains("'proxy'");

    if !needs_migration {
        return Ok(());
    }

    // SQLite's default `ALTER TABLE … RENAME` (since 3.25) rewrites every
    // reference to the renamed table in other objects — including the
    // `REFERENCES memories(id)` clauses on `memory_canonical_state`,
    // `memory_evidence`, and `dedup_decisions`. Those get rewritten to
    // `REFERENCES memories_old`, and when we subsequently DROP
    // `memories_old` the dependent FKs become dangling pointers that
    // cascade into "no such table: memories_old" errors on the next
    // INSERT into the new `memories` (via trigger-driven
    // INSERT INTO memory_canonical_state).
    //
    // Setting `legacy_alter_table = 1` restores pre-3.25 behavior: the
    // rename updates ONLY the renamed table, leaving every `REFERENCES
    // memories(id)` in dependent schemas correctly pointing at whatever
    // table we create next under the name `memories`. That is exactly the
    // semantics this migration wants.
    //
    // We also toggle `foreign_keys` off so the transaction body can
    // execute DDL without intermediate FK checks; both pragmas are
    // restored to their caller-observed state before returning.
    let fk_was_on: bool = conn
        .query_row("PRAGMA foreign_keys", [], |r| r.get::<_, i64>(0))
        .map(|v| v != 0)
        .unwrap_or(false);
    let legacy_was_on: bool = conn
        .query_row("PRAGMA legacy_alter_table", [], |r| r.get::<_, i64>(0))
        .map(|v| v != 0)
        .unwrap_or(false);
    conn.execute_batch("PRAGMA foreign_keys = OFF; PRAGMA legacy_alter_table = 1;")
        .map_err(crate::types::ReinError::Database)?;

    // Wrap entire migration in an EXCLUSIVE transaction to prevent corruption on crash
    conn.execute_batch("BEGIN EXCLUSIVE")
        .map_err(crate::types::ReinError::Database)?;

    let migration_result = (|| -> ReinResult<()> {
        // Step 1: Rename live table so we can recreate with new CHECK constraint
        conn.execute_batch("ALTER TABLE memories RENAME TO memories_old")
            .map_err(crate::types::ReinError::Database)?;

        // Check if old table has the embedding column (pre-proxy DBs may not)
        let has_embedding: bool = conn
            .prepare("SELECT embedding FROM memories_old LIMIT 0")
            .is_ok();

        // Step 2: Conditionally copy embedding column (missing in pre-proxy databases)
        let insert_sql = if has_embedding {
            "INSERT INTO memories (id, layer, topic, summary, content, keywords, importance, source, \
                 strength, decay_lambda, access_count, superseded_by, related_ids, created_at, \
                 updated_at, last_accessed, embedding, status, concept_ids, tier, cluster_id, needs_vec_dedup) \
             SELECT id, layer, topic, summary, content, keywords, importance, source, \
                 strength, decay_lambda, access_count, superseded_by, related_ids, created_at, \
                 updated_at, last_accessed, embedding, status, concept_ids, tier, cluster_id, needs_vec_dedup \
             FROM memories_old;"
        } else {
            "INSERT INTO memories (id, layer, topic, summary, content, keywords, importance, source, \
                 strength, decay_lambda, access_count, superseded_by, related_ids, created_at, \
                 updated_at, last_accessed, status, concept_ids, tier, cluster_id, needs_vec_dedup) \
             SELECT id, layer, topic, summary, content, keywords, importance, source, \
                 strength, decay_lambda, access_count, superseded_by, related_ids, created_at, \
                 updated_at, last_accessed, status, concept_ids, tier, cluster_id, needs_vec_dedup \
             FROM memories_old;"
        };

        // Step 3: Create new table with widened CHECK
        conn.execute_batch(
            "CREATE TABLE memories ( \
                 id TEXT PRIMARY KEY NOT NULL, \
                 layer TEXT NOT NULL CHECK(layer IN ('LTM', 'STM')), \
                 topic TEXT NOT NULL, \
                 summary TEXT NOT NULL, \
                 content TEXT NOT NULL, \
                 keywords TEXT NOT NULL DEFAULT '[]', \
                 importance TEXT NOT NULL CHECK(importance IN ('critical', 'high', 'medium', 'low')), \
                 source TEXT NOT NULL CHECK(source IN ('manual', 'hook', 'migration', 'supermemory', 'proxy')), \
                 strength REAL NOT NULL, \
                 decay_lambda REAL NOT NULL, \
                 access_count INTEGER NOT NULL DEFAULT 0, \
                 superseded_by TEXT, \
                 related_ids TEXT NOT NULL DEFAULT '[]', \
                 created_at TEXT NOT NULL, \
                 updated_at TEXT NOT NULL, \
                 last_accessed TEXT NOT NULL, \
                 embedding BLOB, \
                 status TEXT NOT NULL DEFAULT 'active', \
                 concept_ids TEXT NOT NULL DEFAULT '[]', \
                 tier TEXT NOT NULL DEFAULT 'warm', \
                 cluster_id INTEGER, \
                 needs_vec_dedup INTEGER NOT NULL DEFAULT 0 \
             );"
        ).map_err(crate::types::ReinError::Database)?;
        conn.execute_batch(insert_sql)
            .map_err(crate::types::ReinError::Database)?;
        conn.execute_batch("DROP TABLE memories_old;")
            .map_err(crate::types::ReinError::Database)?;

        // Recreate indexes destroyed by table rename/drop
        conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_memories_topic ON memories(topic); \
             CREATE INDEX IF NOT EXISTS idx_memories_layer ON memories(layer); \
             CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance); \
             CREATE INDEX IF NOT EXISTS idx_memories_strength ON memories(strength); \
             CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at); \
             CREATE INDEX IF NOT EXISTS idx_memories_last_accessed ON memories(last_accessed);",
        )
        .map_err(crate::types::ReinError::Database)?;

        // Recreate every trigger destroyed by the table rename. Before v0.21.0
        // this only covered the three FTS triggers, silently leaving
        // `memories_canonical_ai` gone on migrated databases — new inserts then
        // failed to populate `memory_canonical_state`, breaking canonical_id_for,
        // support/diversity signals, and dedup bookkeeping. Now sourced from the
        // single `MEMORIES_TRIGGERS_SQL` const so future migrations can't drift.
        // Explicitly drop any triggers SQLite auto-renamed to `memories_old`
        // during step 1. Without this, `CREATE TRIGGER IF NOT EXISTS` below
        // is a no-op (name already in sqlite_master), the trigger stays
        // bound to `memories_old`, and gets silently destroyed by the
        // subsequent `DROP TABLE memories_old` — leaving `memories` with no
        // triggers at all after migration.
        conn.execute_batch(
            "DROP TRIGGER IF EXISTS memories_ai; \
             DROP TRIGGER IF EXISTS memories_au; \
             DROP TRIGGER IF EXISTS memories_ad; \
             DROP TRIGGER IF EXISTS memories_canonical_ai;",
        )
        .map_err(crate::types::ReinError::Database)?;

        conn.execute_batch(MEMORIES_TRIGGERS_SQL)
            .map_err(crate::types::ReinError::Database)?;

        // Backfill canonical_state for rows copied in above so they match what
        // the trigger would have inserted on a fresh write.
        conn.execute_batch(
            "INSERT OR IGNORE INTO memory_canonical_state(memory_id, canonical_id) \
             SELECT id, id FROM memories;",
        )
        .map_err(crate::types::ReinError::Database)?;

        Ok(())
    })();

    let commit_result = match migration_result {
        Ok(()) => conn
            .execute_batch("COMMIT")
            .map_err(crate::types::ReinError::Database),
        Err(e) => {
            let _ = conn.execute_batch("ROLLBACK");
            Err(e)
        }
    };

    // Restore the pragmas to caller-observed state regardless of outcome.
    if fk_was_on {
        let _ = conn.execute_batch("PRAGMA foreign_keys = ON");
    }
    if !legacy_was_on {
        let _ = conn.execute_batch("PRAGMA legacy_alter_table = 0");
    }

    commit_result?;

    tracing::info!("migrated source CHECK to include 'proxy'");
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
                );",
            )?;
            conn.execute_batch(
                "INSERT INTO memories_fts(id, topic, summary, content, keywords)
                 SELECT id, topic, summary, content, keywords FROM memories;",
            )?;

            let concepts_needs_rebuild: bool = conn
                .query_row(
                    "SELECT sql FROM sqlite_master WHERE name = 'concepts_fts'",
                    [],
                    |row| {
                        let sql: String = row.get(0)?;
                        Ok(sql.contains("porter"))
                    },
                )
                .unwrap_or(false);

            if concepts_needs_rebuild {
                conn.execute_batch("DROP TABLE IF EXISTS concepts_fts;")?;
                conn.execute_batch(
                    "CREATE VIRTUAL TABLE concepts_fts USING fts5(
                        id, name, definition, labels,
                        tokenize='unicode61'
                    );",
                )?;
                conn.execute_batch(
                    "INSERT INTO concepts_fts(id, name, definition, labels)
                     SELECT id, name, definition, labels FROM concepts;",
                )?;
            }
            Ok(())
        })();
        match result {
            Ok(()) => {
                conn.execute_batch("COMMIT")?;
            }
            Err(e) => {
                let _ = conn.execute_batch("ROLLBACK");
                return Err(e);
            }
        }

        tracing::info!("FTS index rebuilt with unicode61");
    }

    Ok(())
}

/// Check if the embedding model has changed since last run.
/// Returns true if model changed (requiring re-embedding).
/// Stores current model info in metadata for next startup.
pub fn check_embedding_model(conn: &Connection, model: &str, dims: usize) -> ReinResult<bool> {
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
    let vec_sql = format!(
        "CREATE VIRTUAL TABLE vec_memories USING vec0(
            id TEXT PRIMARY KEY,
            embedding float[{dims}] distance_metric=cosine
        );"
    );
    let result = (|| -> ReinResult<()> {
        conn.execute_batch("SAVEPOINT rebuild_vector_index")?;
        conn.execute_batch("DROP TABLE IF EXISTS vec_memories;")?;
        conn.execute_batch("DELETE FROM embed_cache;")?;
        conn.execute_batch(&vec_sql)?;
        conn.execute_batch("RELEASE rebuild_vector_index")?;
        Ok(())
    })();
    if result.is_err() {
        let _ = conn.execute_batch("ROLLBACK TO rebuild_vector_index");
        let _ = conn.execute_batch("RELEASE rebuild_vector_index");
    }
    result
}

/// Atomically replace vec_memories and embed_cache after new embeddings were
/// successfully computed.  The old index remains visible if table creation or
/// any insert fails before the savepoint is released.
pub fn replace_vector_index(
    conn: &Connection,
    dims: usize,
    embeddings: &[(String, Vec<f32>)],
) -> ReinResult<()> {
    tracing::warn!(
        "replacing vector index atomically (dims={dims}, rows={})",
        embeddings.len()
    );
    let vec_sql = format!(
        "CREATE VIRTUAL TABLE vec_memories USING vec0(
            id TEXT PRIMARY KEY,
            embedding float[{dims}] distance_metric=cosine
        );"
    );
    let result = (|| -> ReinResult<()> {
        conn.execute_batch("SAVEPOINT replace_vector_index")?;
        conn.execute_batch("DROP TABLE IF EXISTS vec_memories;")?;
        conn.execute_batch("DELETE FROM embed_cache;")?;
        conn.execute_batch(&vec_sql)?;
        {
            let mut stmt =
                conn.prepare("INSERT OR REPLACE INTO vec_memories(id, embedding) VALUES (?1, ?2)")?;
            for (id, embedding) in embeddings {
                let bytes: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
                stmt.execute(rusqlite::params![id, bytes])?;
            }
        }
        conn.execute_batch("RELEASE replace_vector_index")?;
        Ok(())
    })();
    if result.is_err() {
        let _ = conn.execute_batch("ROLLBACK TO replace_vector_index");
        let _ = conn.execute_batch("RELEASE replace_vector_index");
    }
    result
}

/// Create a staging table for streaming embeddings during reindex.
/// Stream inserts into `embed_staging(id, embedding BLOB)` chunk by chunk to
/// keep memory usage O(chunk) instead of O(total) — 100k × 3072-dim floats
/// is >1GB in the caller's Vec, but the BLOB is persisted to disk here.
pub fn create_embed_staging(conn: &Connection) -> ReinResult<()> {
    conn.execute_batch(
        "DROP TABLE IF EXISTS embed_staging;
         CREATE TABLE embed_staging (
            id TEXT PRIMARY KEY,
            embedding BLOB NOT NULL
         );",
    )?;
    Ok(())
}

/// Drain `embed_staging` into a freshly-built `vec_memories` under one savepoint.
/// Mirrors `replace_vector_index` but reads from the staging table so the caller
/// doesn't have to hold the full embedding set in RAM. On success also drops the
/// staging table; on failure the old index is preserved by rollback and the
/// staging table remains (caller can retry or clean up).
pub fn replace_vector_index_from_staging(conn: &Connection, dims: usize) -> ReinResult<()> {
    let row_count: i64 = conn
        .query_row("SELECT COUNT(*) FROM embed_staging", [], |r| r.get(0))
        .unwrap_or(0);
    tracing::warn!(
        "replacing vector index atomically from staging (dims={dims}, rows={row_count})"
    );
    let vec_sql = format!(
        "CREATE VIRTUAL TABLE vec_memories USING vec0(
            id TEXT PRIMARY KEY,
            embedding float[{dims}] distance_metric=cosine
        );"
    );
    let result = (|| -> ReinResult<()> {
        conn.execute_batch("SAVEPOINT replace_vector_index_staged")?;
        conn.execute_batch("DROP TABLE IF EXISTS vec_memories;")?;
        conn.execute_batch("DELETE FROM embed_cache;")?;
        conn.execute_batch(&vec_sql)?;
        {
            let mut read = conn.prepare("SELECT id, embedding FROM embed_staging")?;
            let mut write =
                conn.prepare("INSERT OR REPLACE INTO vec_memories(id, embedding) VALUES (?1, ?2)")?;
            let mut rows = read.query([])?;
            while let Some(row) = rows.next()? {
                let id: String = row.get(0)?;
                let bytes: Vec<u8> = row.get(1)?;
                write.execute(rusqlite::params![id, bytes])?;
            }
        }
        conn.execute_batch("DROP TABLE embed_staging;")?;
        conn.execute_batch("RELEASE replace_vector_index_staged")?;
        Ok(())
    })();
    if result.is_err() {
        let _ = conn.execute_batch("ROLLBACK TO replace_vector_index_staged");
        let _ = conn.execute_batch("RELEASE replace_vector_index_staged");
    }
    result
}

#[cfg(test)]
mod migration_tests {
    use super::*;
    use tempfile::NamedTempFile;

    /// Builds a file-backed DB that looks like a pre-proxy v0.9.3 database:
    /// `source` CHECK without 'proxy', so the next `init_schema` triggers
    /// `migrate_source_check` — the exact path that used to drop the
    /// `memories_canonical_ai` trigger without recreating it.
    fn seed_pre_proxy_db() -> NamedTempFile {
        let file = NamedTempFile::new().expect("tempfile");
        let conn = Connection::open(file.path()).expect("open");
        init_sqlite_vec();
        conn.execute_batch(
            "CREATE TABLE memories (
                 id TEXT PRIMARY KEY NOT NULL,
                 layer TEXT NOT NULL,
                 topic TEXT NOT NULL,
                 summary TEXT NOT NULL,
                 content TEXT NOT NULL,
                 keywords TEXT NOT NULL DEFAULT '[]',
                 importance TEXT NOT NULL,
                 source TEXT NOT NULL CHECK(source IN ('manual','hook','migration','supermemory')),
                 strength REAL NOT NULL,
                 decay_lambda REAL NOT NULL,
                 access_count INTEGER NOT NULL DEFAULT 0,
                 superseded_by TEXT,
                 related_ids TEXT NOT NULL DEFAULT '[]',
                 created_at TEXT NOT NULL,
                 updated_at TEXT NOT NULL,
                 last_accessed TEXT NOT NULL,
                 status TEXT NOT NULL DEFAULT 'active',
                 concept_ids TEXT NOT NULL DEFAULT '[]',
                 tier TEXT NOT NULL DEFAULT 'warm',
                 cluster_id INTEGER,
                 needs_vec_dedup INTEGER NOT NULL DEFAULT 0
             );
             INSERT INTO memories (id, layer, topic, summary, content, importance, source,
                 strength, decay_lambda, created_at, updated_at, last_accessed)
             VALUES ('legacy-1', 'LTM', 't', 's', 'c', 'medium', 'manual',
                 1.0, 0.0, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z');",
        )
        .expect("seed");
        file
    }

    /// Regression test for the v0.21.0 migration-trigger bug:
    /// after `migrate_source_check` drops+recreates `memories`, every trigger
    /// on the table must still fire. Previously only FTS triggers were
    /// recreated, so `memories_canonical_ai` was silently lost and
    /// `memory_canonical_state` rows for new inserts were never created.
    #[test]
    fn migrate_source_check_recreates_canonical_trigger() {
        let file = seed_pre_proxy_db();
        let conn = Connection::open(file.path()).unwrap();
        // Running init_schema triggers migrate_source_check on a file-backed
        // DB whose `source` CHECK doesn't allow 'proxy'.
        init_schema(&conn, 4).expect("init_schema after migration");

        // Dump post-migration schema — any object whose SQL still references
        // `memories_old` is a dangling reference that will explode on the
        // next INSERT into `memories`.
        let dangling: Vec<(String, String, String)> = {
            let mut stmt = conn
                .prepare(
                    "SELECT type, name, sql FROM sqlite_master \
                      WHERE sql LIKE '%memories_old%'",
                )
                .unwrap();
            stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
                .unwrap()
                .filter_map(|r| r.ok())
                .collect()
        };
        assert!(
            dangling.is_empty(),
            "schema objects still reference memories_old after migration: {:?}",
            dangling
        );

        // A fresh insert post-migration must populate memory_canonical_state
        // via the canonical_ai trigger — this is what used to silently fail.
        conn.execute_batch(
            "INSERT INTO memories (id, layer, topic, summary, content, importance, source,
                 strength, decay_lambda, created_at, updated_at, last_accessed)
             VALUES ('post-mig-1', 'LTM', 't', 's', 'c', 'medium', 'proxy',
                 1.0, 0.0, '2026-04-20T00:00:00Z', '2026-04-20T00:00:00Z', '2026-04-20T00:00:00Z');",
        )
        .unwrap();

        let canonical: Option<String> = conn
            .query_row(
                "SELECT canonical_id FROM memory_canonical_state WHERE memory_id = 'post-mig-1'",
                [],
                |r| r.get(0),
            )
            .ok();
        assert_eq!(
            canonical.as_deref(),
            Some("post-mig-1"),
            "memories_canonical_ai trigger must fire on post-migration inserts"
        );

        // Pre-existing rows carried through the migration are backfilled.
        let backfilled: Option<String> = conn
            .query_row(
                "SELECT canonical_id FROM memory_canonical_state WHERE memory_id = 'legacy-1'",
                [],
                |r| r.get(0),
            )
            .ok();
        assert_eq!(backfilled.as_deref(), Some("legacy-1"));

        // FTS triggers must still be intact too — sanity check against future regressions.
        let fts_hit: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM memories_fts WHERE id = 'post-mig-1'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(
            fts_hit, 1,
            "memories_ai trigger must also survive migration"
        );
    }
}
