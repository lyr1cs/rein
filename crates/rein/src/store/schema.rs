// ============================================================================
// v0.31 candidate — pending Codex audit before commit (Agent F4 / A-H2).
// Audit finding: `migrate_oauth_tables` ran a serial CREATE / probe / ALTER /
// UPDATE / COUNT on every `open_store()` call, scaling per-open work linearly
// with grant table size.  Compounded with A-H3 (`verify_bearer` opening a
// fresh store per request) this produced a perf cliff under claude.ai polling
// load.  This patch routes the OAuth migration through `oauth_schema_version`
// (stored in the `schema_migrations` table) so steady-state opens skip the
// migration after the first run.  See `reviews/fix-20260511-F4-oauth.md`.
// ============================================================================
use crate::types::{ReinError, ReinResult};
use rusqlite::{Connection, OptionalExtension};
use std::sync::Once;

/// v0.31 candidate (Agent F4 / A-H2): current OAuth migration schema version.
/// Bump when adding a new ALTER / column / index for the OAuth tables.
/// Stored in `schema_migrations(name='oauth', version=N)` so we don't collide
/// with SQLite's single global `PRAGMA user_version` slot.
const OAUTH_SCHEMA_VERSION: i64 = 1;

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

/// Single global schema version, stored in SQLite's `PRAGMA user_version`
/// (an otherwise-unused header slot — this is the first thing to claim it).
///
/// `BASELINE_SCHEMA_VERSION` marks the de-facto schema produced by the
/// idempotent legacy bring-up (`bring_up_to_baseline`) as of v0.38.0. Forward
/// migrations in `MIGRATIONS` use versions strictly greater than this.
///
/// INVARIANT (resurrection-safety): the legacy bring-up — every probe-then-ALTER
/// and every `migrate_*` fn — runs ONLY when `user_version < BASELINE`. Once a DB
/// is stamped at baseline, only the forward migration runner mutates DDL, so a
/// forward `DROP COLUMN` is never silently re-added by a re-probe on the next
/// open. This is the central correctness property of the framework; it is
/// pinned by `migration_drop_column_is_not_resurrected` in `migration_tests`.
///
/// NOTE: this is distinct from the per-row `schema_version` columns on
/// `session_artifacts` / `ars_parameter_policy` — those version individual row
/// *payloads* for forward-compatible deserialization and are intentionally left
/// untouched here (the ars R8 peek-before-deserialize path depends on them).
const BASELINE_SCHEMA_VERSION: i64 = 1;

/// A single forward, version-gated schema migration.
///
/// `up` runs inside a transaction that also bumps `PRAGMA user_version` to
/// `version`; the DDL and the version bump commit (or roll back) together, so a
/// failed migration leaves both the schema and the recorded version unchanged
/// (fail-loud + atomic). Use `up` for rename / type-change / drop via SQLite's
/// 12-step table-rebuild — reuse `MEMORIES_TRIGGERS_SQL` when rebuilding
/// `memories`, since dropping the table tears its triggers down.
///
/// CONTRACT: `up` already runs inside `apply_migrations`' `BEGIN IMMEDIATE`, so
/// it must NOT open its own transaction or toggle connection-level pragmas
/// (`PRAGMA foreign_keys` is a no-op inside a transaction); use a `SAVEPOINT`
/// if nested rollback scope is needed. It is therefore NOT a drop-in copy of
/// `migrate_source_check`, which runs in autocommit during legacy bring-up and
/// opens its own `BEGIN EXCLUSIVE`.
struct Migration {
    version: i64,
    #[allow(dead_code)] // retained for logging / future migration ledger.
    name: &'static str,
    up: fn(&Connection) -> ReinResult<()>,
    /// True when `up` is additive + idempotent — `CREATE ... IF NOT EXISTS`,
    /// `ADD COLUMN`, backfills that re-run cleanly. Such a migration carries no
    /// resurrection hazard (a concurrent/unserialized second run is a no-op via
    /// `IF NOT EXISTS` + the in-lock version recheck + the atomic version
    /// stamp), so it may run on a lock-degraded open path (a non-unix build, or
    /// an flock-create failure on NFS / a read-only dir). A `false` migration
    /// (drop / rename / 12-step rebuild) MUST be serialized and fails closed if
    /// the cross-process lock can't be taken.
    idempotent: bool,
}

/// Forward migrations applied after baseline, in ascending `version` order.
///
/// v2 (`create_memory_triples`) is the first real forward migration: the
/// durable triple/fact table for the #A5 persistence unlock. Additive
/// `CREATE TABLE` / `CREATE INDEX` only, so it is resurrection-safe and runs
/// unconditionally on every open; triple POPULATION and READS are separately
/// gated by `[dedup].persist_triples` (default off → empty table → no behavior
/// change). The synthetic-migration tests in `migration_tests` still cover the
/// drop / rollback / re-apply slices via their own `&[Migration]` lists.
/// Schema version that installs the `a12_input_epoch` row + triggers. The
/// every-open self-heal below keys on this same constant.
const A12_INPUT_EPOCH_SCHEMA_VERSION: i64 = 4;

const MIGRATIONS: &[Migration] = &[
    Migration {
        version: 2,
        name: "create_memory_triples",
        up: create_memory_triples,
        // Additive `CREATE TABLE/INDEX IF NOT EXISTS` — resurrection-safe, so it
        // may run on a lock-degraded open (non-unix build / flock-create failure)
        // without bricking DB open. See the fail-closed gate below.
        idempotent: true,
    },
    Migration {
        version: 3,
        name: "create_memory_triple_meta",
        up: create_memory_triple_meta,
        // Additive `CREATE TABLE IF NOT EXISTS` — same resurrection-safety
        // argument as v2.
        idempotent: true,
    },
    Migration {
        version: A12_INPUT_EPOCH_SCHEMA_VERSION,
        name: "create_a12_input_epoch",
        up: create_a12_input_epoch,
        // Metadata initialization plus CREATE TRIGGER IF NOT EXISTS only.
        // Replaying the migration cannot duplicate trigger programs or reset
        // an already-advanced epoch.
        idempotent: true,
    },
];

/// v2 forward migration: durable `(subject, predicate, object)` fact table for
/// #A5 triple persistence.
///
/// Additive only (resurrection-safe). Runs inside `apply_migrations`'
/// `BEGIN IMMEDIATE`, so per the `Migration` CONTRACT it MUST NOT open its own
/// transaction or toggle connection-level pragmas. `memory_id` carries
/// `ON DELETE CASCADE`, so every hard-delete path (forget / gc / by-topic /
/// STM prune) cleans triples automatically under `PRAGMA foreign_keys=ON`.
/// `UNIQUE(memory_id, subject, predicate, object)` makes population idempotent;
/// its left-prefix autoindex already serves per-memory lookups, so only the
/// `subject` index (for future fact-space queries) is created explicitly.
fn create_memory_triples(conn: &Connection) -> ReinResult<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS memory_triples (
            memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
            subject TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(memory_id, subject, predicate, object)
        );
        CREATE INDEX IF NOT EXISTS idx_memory_triples_subject ON memory_triples(subject);",
    )?;
    Ok(())
}

/// v3 forward migration: per-memory extraction stamp enabling the #A5
/// dedup-reuse READER (the v2 table alone cannot express the reader's three
/// correctness prerequisites).
///
/// One row per memory whose triples have been extracted and persisted:
///   * row PRESENCE = cold-start backfill marker — distinguishes "genuinely
///     zero facts" (row, zero `memory_triples` rows) from "never extracted"
///     (no row; pre-v1.2 DBs and OFF→ON config flips). This cannot live as a
///     per-row column on `memory_triples`: both states produce zero rows there.
///   * `content_hash` = SHA-256 of the exact content the triples were
///     extracted from; reader recomputes when it doesn't match the candidate's
///     current content (self-healing — eager-writer gaps become misses, never
///     wrong verdicts).
///   * `extractor_version` = `TRIPLE_EXTRACTOR_VERSION` at write time; reader
///     treats any other value as stale (the dedup new-content side always
///     recomputes under CURRENT semantics — mixed-semantics overlap scoring
///     could flip a verdict).
///
/// Additive only (resurrection-safe). Runs inside `apply_migrations`'
/// `BEGIN IMMEDIATE` per the `Migration` CONTRACT (no own transaction, no
/// pragma toggles). `ON DELETE CASCADE` mirrors `memory_triples` so
/// hard-delete paths clean the stamp automatically.
fn create_memory_triple_meta(conn: &Connection) -> ReinResult<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS memory_triple_meta (
            memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
            content_hash TEXT NOT NULL,
            extractor_version INTEGER NOT NULL,
            extracted_at TEXT NOT NULL
        );",
    )?;
    Ok(())
}

/// v4 forward migration: monotonic O(1) invalidation token for online A12.
///
/// Offline calibration still hashes the complete SQLite recall snapshot. Live
/// recall only needs to know whether any input to that snapshot changed since
/// the generation was sealed, so every base-table write increments one durable
/// counter. FTS shadow tables are derived synchronously from `memories` and
/// `concepts`; vec0 writes flow through `embedding_write_seq`, which the
/// metadata trigger below observes. This avoids triggers on virtual/shadow
/// tables and keeps the epoch independent of SQLite extension internals.
fn create_a12_input_epoch(conn: &Connection) -> ReinResult<()> {
    conn.execute(
        "INSERT OR IGNORE INTO metadata(key, value) VALUES ('a12_input_epoch', '0')",
        [],
    )?;

    // Existing adaptive snapshots predate the projection digest. Backfill only
    // when this binary can parse the row, and preserve unknown top-level fields
    // so a mixed-version deployment never loses forward-compatible data. A
    // malformed/future row deliberately stays without a digest; the trigger
    // below will then conservatively advance the epoch on every rewrite.
    let adaptive_raw = conn
        .query_row(
            "SELECT value FROM metadata WHERE key = 'adaptive_state'",
            [],
            |row| row.get::<_, String>(0),
        )
        .optional()?;
    if let Some(raw) = adaptive_raw {
        if let Ok(state) = serde_json::from_str::<crate::store::adaptive::AdaptiveState>(&raw) {
            if let Ok(fingerprint) = state.compute_a12_recall_projection_fingerprint() {
                conn.execute(
                    "UPDATE metadata
                     SET value = json_set(value, '$.a12_recall_projection_fingerprint', ?1)
                     WHERE key = 'adaptive_state'",
                    rusqlite::params![fingerprint],
                )?;
            }
        }
    }

    // One row can legitimately bump more than once (for example a memory
    // INSERT also creates canonical state). Exact deltas are irrelevant; only
    // monotonic change matters. The nested metadata UPDATE cannot recurse:
    // all metadata triggers below exclude `a12_input_epoch` in their WHEN
    // predicates, including when PRAGMA recursive_triggers is enabled.
    const BUMP: &str = "
        UPDATE metadata SET value = CASE
            WHEN typeof(value) = 'text'
             AND value <> ''
             AND value NOT GLOB '*[^0-9]*'
             AND length(value) <= 19
             AND (length(value) < 19 OR value < '9223372036854775807')
            THEN CAST(CAST(value AS INTEGER) + 1 AS TEXT)
            ELSE RAISE(ABORT, 'a12_input_epoch is malformed, negative, or exhausted')
        END WHERE key = 'a12_input_epoch';
        SELECT CASE WHEN changes() = 1 THEN 1
            ELSE RAISE(ABORT, 'a12_input_epoch is missing') END;";
    for table in [
        "memories",
        "memory_canonical_state",
        "memory_evidence",
        "concepts",
        "concept_links",
        "episodes",
    ] {
        for (suffix, operation) in [("ai", "INSERT"), ("au", "UPDATE"), ("ad", "DELETE")] {
            conn.execute_batch(&format!(
                "CREATE TRIGGER IF NOT EXISTS a12_input_epoch_{table}_{suffix} \
                 AFTER {operation} ON {table} BEGIN {BUMP} END;"
            ))?;
        }
    }

    let relevant_new = "NEW.key IN ('adaptive_state', 'rerank_weights', 'embedding_write_seq') \
                        OR NEW.key GLOB 'survival_curve:*'";
    let relevant_old = "OLD.key IN ('adaptive_state', 'rerank_weights', 'embedding_write_seq') \
                        OR OLD.key GLOB 'survival_curve:*'";
    // AdaptiveState is large and its version advances on every cadence. Only
    // three fields affect A12 replay, so current writers persist their shared
    // projection digest. Equality is trusted only for two valid 64-hex text
    // digests. Missing, malformed, or legacy digests always bump (fail closed).
    let same_valid_adaptive_projection = "
        OLD.key = 'adaptive_state'
        AND NEW.key = 'adaptive_state'
        AND CASE
            WHEN json_valid(OLD.value) = 0 OR json_valid(NEW.value) = 0 THEN 0
            WHEN json_type(OLD.value, '$.a12_recall_projection_fingerprint') != 'text'
              OR json_type(NEW.value, '$.a12_recall_projection_fingerprint') != 'text'
            THEN 0
            WHEN length(json_extract(OLD.value, '$.a12_recall_projection_fingerprint')) != 64
              OR length(json_extract(NEW.value, '$.a12_recall_projection_fingerprint')) != 64
            THEN 0
            WHEN json_extract(OLD.value, '$.a12_recall_projection_fingerprint')
                   GLOB '*[^0-9A-Fa-f]*'
              OR json_extract(NEW.value, '$.a12_recall_projection_fingerprint')
                   GLOB '*[^0-9A-Fa-f]*'
            THEN 0
            WHEN json_extract(OLD.value, '$.a12_recall_projection_fingerprint') =
                 json_extract(NEW.value, '$.a12_recall_projection_fingerprint')
            THEN 1
            ELSE 0
        END = 1";
    conn.execute_batch(&format!(
        "CREATE TRIGGER IF NOT EXISTS a12_input_epoch_metadata_ai
             AFTER INSERT ON metadata WHEN {relevant_new} BEGIN {BUMP} END;
         CREATE TRIGGER IF NOT EXISTS a12_input_epoch_metadata_au
             AFTER UPDATE ON metadata
             WHEN ({relevant_old} OR {relevant_new})
              AND (OLD.key IS NOT NEW.key OR OLD.value IS NOT NEW.value)
              AND NOT ({same_valid_adaptive_projection})
             BEGIN {BUMP} END;
         CREATE TRIGGER IF NOT EXISTS a12_input_epoch_metadata_ad
             AFTER DELETE ON metadata WHEN {relevant_old} BEGIN {BUMP} END;"
    ))?;
    Ok(())
}

/// Read the global schema version from `PRAGMA user_version` (0 on a DB this
/// framework has never stamped).
fn schema_user_version(conn: &Connection) -> ReinResult<i64> {
    Ok(conn.query_row("PRAGMA user_version", [], |row| row.get(0))?)
}

/// Stamp `PRAGMA user_version`. When called inside an open transaction the
/// write participates in it (the value lives in the database header page).
fn set_schema_user_version(conn: &Connection, version: i64) -> ReinResult<()> {
    conn.pragma_update(None, "user_version", version)?;
    Ok(())
}

/// Highest version this build knows how to reach: the max migration version, or
/// the baseline when no forward migrations exist yet. A DB stamped above this
/// was written by a newer rein and must not be opened (see downgrade guard).
fn max_known_schema_version(migrations: &[Migration]) -> i64 {
    migrations
        .iter()
        .map(|m| m.version)
        .max()
        .unwrap_or(BASELINE_SCHEMA_VERSION)
        .max(BASELINE_SCHEMA_VERSION)
}

/// Idempotent `ALTER TABLE ... ADD COLUMN` that fails loud on every error
/// EXCEPT the benign "duplicate column name" race — i.e. a peer process added
/// the column between our `pragma_table_info` probe and this ALTER. Previously
/// these sites used `.ok()`, which also swallowed disk-full / corruption /
/// lock failures and left a load-bearing column silently missing on old DBs.
fn add_column_idempotent(conn: &Connection, sql: &str) -> ReinResult<()> {
    match conn.execute_batch(sql) {
        Ok(()) => Ok(()),
        // SQLite emits SQLITE_ERROR with the stable message
        // "duplicate column name: <col>" when the column already exists.
        Err(e) if e.to_string().contains("duplicate column name") => Ok(()),
        Err(e) => Err(e.into()),
    }
}

/// Cross-process serialization for the open-time bring-up + forward migrations.
///
/// `bring_up_to_baseline` can't be wrapped in a SQL transaction
/// (`migrate_source_check` toggles `PRAGMA foreign_keys` — a no-op inside a txn
/// — and opens its own `BEGIN EXCLUSIVE`; vec0 virtual tables are created there
/// too), so a SQLite lock can't span the sequence. Instead take an OS advisory
/// lock (`flock`) on a sidecar `<db>.migrate.lock`, mirroring `with_hnsw_lock`.
/// The winner runs detect → bring-up → migrate while peers block on the lock,
/// then re-read the stamped `user_version` and skip — so a forward `DROP`/rename
/// is never resurrected by a peer's stale bring-up running over an
/// already-migrated schema.
///
/// `:memory:` and pathless connections are single-process (no shared inode) and
/// need no lock (`NotNeeded`). On a file-backed DB a create/`flock` failure
/// yields `Failed`, and the caller fails closed before any mutating schema work
/// rather than running it unserialized (a degraded lock there could let two
/// concurrent first-opens resurrect a dropped column). Released on drop (and
/// again implicitly when the fd closes).
enum MigrationLock {
    /// Exclusive advisory lock held on `<db>.migrate.lock`.
    Held(std::fs::File),
    /// Single-process connection (`:memory:` / pathless) — migration work may
    /// proceed without a cross-process lock.
    NotNeeded,
    /// File-backed DB whose lock could not be created/acquired — the caller
    /// MUST fail closed before running any mutating schema work.
    Failed,
}

impl MigrationLock {
    fn acquire(conn: &Connection) -> Self {
        let path = match conn.path() {
            Some(p) if p != ":memory:" && !p.is_empty() => p.to_string(),
            _ => return Self::NotNeeded,
        };
        let lock_path = format!("{path}.migrate.lock");
        let file = match std::fs::File::create(&lock_path) {
            Ok(f) => f,
            Err(e) => {
                tracing::warn!("migration lock: create {lock_path} failed: {e}");
                return Self::Failed;
            }
        };
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) };
            if rc != 0 {
                tracing::warn!(
                    "migration flock(LOCK_EX) failed: {}",
                    std::io::Error::last_os_error()
                );
                return Self::Failed;
            }
            Self::Held(file)
        }
        #[cfg(not(unix))]
        {
            // No portable advisory file lock without an extra dependency, so a
            // non-unix file-backed DB is reported `Failed`: a pending FORWARD
            // migration then fails closed (it can't be serialized here), while
            // idempotent baseline bring-up still proceeds. rein's production
            // target is macOS (unix), where the `flock` above is the real lock.
            let _ = file;
            Self::Failed
        }
    }

    /// True when migration work may safely proceed: an exclusive lock is held,
    /// or the connection is single-process. `Failed` returns false so the caller
    /// fails closed instead of running migrations unserialized.
    fn serialized(&self) -> bool {
        !matches!(self, Self::Failed)
    }
}

impl Drop for MigrationLock {
    fn drop(&mut self) {
        #[cfg(unix)]
        if let Self::Held(file) = self {
            use std::os::unix::io::AsRawFd;
            let _ = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_UN) };
        }
    }
}

/// Bring the schema up to the current version and run any pending forward
/// migrations. Safe to call on every open: it is cheap once a DB is at the
/// latest version (one `PRAGMA user_version` read, then nothing).
///
/// `dims` is the embedding dimension (e.g., 3072 for gemini-embedding-001).
pub fn init_schema(conn: &Connection, dims: usize) -> ReinResult<()> {
    init_schema_with_migrations(conn, dims, MIGRATIONS)
}

/// Core open-time schema flow, parameterized over the migration list so tests
/// can drive synthetic forward migrations (drop / rollback / re-apply) through
/// the exact production code path.
fn init_schema_with_migrations(
    conn: &Connection,
    dims: usize,
    migrations: &[Migration],
) -> ReinResult<()> {
    // Serialize the whole detect → bring-up → migrate sequence across processes
    // via an OS advisory lock on a sidecar file (see `MigrationLock`). Held until
    // this function returns; a peer that opens the same pre-baseline DB blocks
    // here, then re-reads the stamped version below and skips bring-up + applied
    // migrations — closing the first-open resurrection/double-apply race that a
    // SQL transaction can't cover (bring-up runs in autocommit).
    let migration_lock = MigrationLock::acquire(conn);

    let current = schema_user_version(conn)?;
    let max_known = max_known_schema_version(migrations);

    // Downgrade guard. A DB stamped by a newer rein may have had columns
    // dropped / renamed that THIS build's bring-up would resurrect (and whose
    // forward migrations this build doesn't have). Refuse rather than corrupt.
    if current > max_known {
        return Err(ReinError::Config(format!(
            "database schema version {current} is newer than this rein build \
             supports (max {max_known}); upgrade rein to open this database"
        )));
    }

    // Fail closed ONLY when a NON-IDEMPOTENT forward migration is pending but
    // the cross-process lock couldn't be taken (a file-backed DB with a
    // create/`flock` failure on NFS / a read-only dir, or a non-unix build). A
    // `DROP`/rename/rebuild run unserialized is the resurrection/double-apply
    // hazard, so it must refuse rather than corrupt.
    //
    // An ADDITIVE + idempotent forward migration (`CREATE ... IF NOT EXISTS`,
    // `ADD COLUMN`) is resurrection-safe: a concurrent/unserialized second run
    // is a no-op via `IF NOT EXISTS` + the in-lock version recheck in
    // `apply_migrations` + the atomic version stamp. Blocking it would brick a
    // normal Windows/source install or any DB on a filesystem where flock can't
    // be created — so those proceed on the lock-degraded path. Idempotent
    // baseline bring-up (`current < BASELINE`) is likewise never gated.
    // Unix file DBs hold the real lock above; `:memory:` is single-process.
    let unsafe_migration_pending = migrations
        .iter()
        .any(|m| m.version > current && !m.idempotent);
    if unsafe_migration_pending && !migration_lock.serialized() {
        return Err(ReinError::Config(
            "could not acquire the schema migration lock (<db>.migrate.lock); \
             refusing to run a non-idempotent forward schema migration unserialized"
                .to_string(),
        ));
    }

    // One-time legacy bring-up. `bring_up_to_baseline` is idempotent, so it
    // brings ANY prior state — fresh DB or a pre-baseline one from an older
    // rein — up to the baseline schema; it needs no fragile legacy-vs-fresh
    // detection. Gated `< BASELINE` so a forward DROP/rename is never undone by
    // a re-probe on the next open.
    if current < BASELINE_SCHEMA_VERSION {
        bring_up_to_baseline(conn, dims)?;
        // Stamp the baseline under a write lock with an in-lock re-read so a
        // concurrent first-open can't stamp the version *backwards*. The lock
        // spans only the stamp, not the bring-up: `bring_up_to_baseline` MUST
        // run in autocommit (it can't be wrapped in a transaction —
        // `migrate_source_check` toggles `PRAGMA foreign_keys`, a no-op inside
        // a txn, and opens its own `BEGIN EXCLUSIVE`; vec0 virtual tables are
        // also created there).
        //
        // First-open serialization across processes is now provided by the
        // `MigrationLock` flock held for this whole function (acquired at the
        // top), so a peer can't run bring-up over an already-migrated schema and
        // resurrect a dropped column. This in-lock monotonic stamp guard + the
        // `apply_migrations` recheck remain as defense-in-depth, and cover the
        // lock-degraded paths (`:memory:`, create/flock failure, non-unix).
        conn.execute_batch("BEGIN IMMEDIATE")?;
        let stamped = (|| -> ReinResult<()> {
            if schema_user_version(conn)? < BASELINE_SCHEMA_VERSION {
                set_schema_user_version(conn, BASELINE_SCHEMA_VERSION)?;
            }
            Ok(())
        })();
        match stamped {
            Ok(()) => conn.execute_batch("COMMIT")?,
            Err(e) => {
                let _ = conn.execute_batch("ROLLBACK");
                return Err(e);
            }
        }
    }

    apply_migrations(conn, migrations)?;
    heal_a12_input_epoch_row(conn)?;
    Ok(())
}

/// Every-open self-heal for the v4 `a12_input_epoch` row. The v4 triggers
/// RAISE on every base-table write when the row is missing, and
/// `apply_migrations` never re-runs a stamped migration — a deleted row would
/// otherwise wedge all writes permanently. `INSERT OR IGNORE` cannot reset a
/// live counter; a re-created row restarts at zero, which any sealed A12
/// run's `source_input_epoch` then mismatches, forcing fail-closed
/// recalibration (the safe direction; writes were trigger-rejected while the
/// row was missing, so no data change can hide behind the reset). A
/// present-but-malformed value is deliberately NOT rewritten here: silently
/// re-baselining could serve stale A12 evidence, so doctor surfaces it with
/// an explicit repair path that also invalidates the active calibration.
fn heal_a12_input_epoch_row(conn: &Connection) -> ReinResult<()> {
    if schema_user_version(conn)? < A12_INPUT_EPOCH_SCHEMA_VERSION {
        return Ok(());
    }
    conn.execute(
        "INSERT OR IGNORE INTO metadata(key, value) VALUES ('a12_input_epoch', '0')",
        [],
    )?;
    Ok(())
}

/// Apply every migration whose `version` exceeds the DB's current
/// `user_version`, in order, each in its own transaction. The version bump is
/// part of the same transaction as the DDL, so a mid-migration failure rolls
/// back both — the DB is never left at a half-applied version.
fn apply_migrations(conn: &Connection, migrations: &[Migration]) -> ReinResult<()> {
    let mut prev = BASELINE_SCHEMA_VERSION;
    for m in migrations {
        debug_assert!(
            m.version > BASELINE_SCHEMA_VERSION,
            "forward migration {} must be > baseline {BASELINE_SCHEMA_VERSION}",
            m.version
        );
        debug_assert!(
            m.version > prev,
            "migrations must be authored in strictly ascending version order"
        );
        prev = m.version;

        if m.version <= schema_user_version(conn)? {
            continue; // already applied on a prior open
        }

        conn.execute_batch("BEGIN IMMEDIATE")?;
        let applied = (|| -> ReinResult<()> {
            // Double-checked locking: a peer process may have applied this
            // migration between our pre-lock `schema_user_version` read above
            // and our acquiring the write lock here. Re-read inside the
            // transaction and skip if already applied — running a
            // non-idempotent `up` (drop / rename / table-rebuild) twice could
            // fail or damage the just-migrated schema.
            if m.version <= schema_user_version(conn)? {
                return Ok(());
            }
            (m.up)(conn)?;
            set_schema_user_version(conn, m.version)?;
            Ok(())
        })();
        match applied {
            Ok(()) => conn.execute_batch("COMMIT")?,
            Err(e) => {
                let _ = conn.execute_batch("ROLLBACK");
                return Err(e);
            }
        }
    }
    Ok(())
}

/// Idempotent legacy bring-up: creates every base table / index / trigger /
/// virtual table and runs the historical probe-then-ALTER + `migrate_*` column
/// additions. Invoked only when `user_version < BASELINE` (see
/// `init_schema_with_migrations`); conceptually this whole function *is* "the
/// migration to baseline". Do not add new schema changes here — author a
/// `Migration` in `MIGRATIONS` instead, or the resurrection-safety invariant
/// breaks.
fn bring_up_to_baseline(conn: &Connection, dims: usize) -> ReinResult<()> {
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
        add_column_idempotent(
            conn,
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
        add_column_idempotent(
            conn,
            "ALTER TABLE memories ADD COLUMN concept_ids TEXT NOT NULL DEFAULT '[]'",
        )?;
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
        add_column_idempotent(
            conn,
            "ALTER TABLE concepts ADD COLUMN source_memory_ids TEXT NOT NULL DEFAULT '[]'",
        )?;
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
        add_column_idempotent(conn, "ALTER TABLE concept_links ADD COLUMN valid_from TEXT")?;
        add_column_idempotent(
            conn,
            "ALTER TABLE concept_links ADD COLUMN valid_until TEXT",
        )?;
        // Backfill: existing links get valid_from = created_at
        conn.execute_batch(
            "UPDATE concept_links SET valid_from = created_at WHERE valid_from IS NULL",
        )?;
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
        add_column_idempotent(conn, "ALTER TABLE concepts ADD COLUMN last_episode_id TEXT")?;
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
        // Split into per-column fail-loud ALTERs: a peer-race duplicate on the
        // first column must not abort the batch and silently skip the rest.
        add_column_idempotent(conn, "ALTER TABLE concepts ADD COLUMN living_summary TEXT")?;
        add_column_idempotent(
            conn,
            "ALTER TABLE concepts ADD COLUMN living_summary_updated_at TEXT",
        )?;
        add_column_idempotent(
            conn,
            "ALTER TABLE concepts ADD COLUMN living_summary_source_revision INTEGER",
        )?;
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
    conn.execute_batch("CREATE INDEX IF NOT EXISTS idx_concepts_updated ON concepts(updated_at)")?;

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
        add_column_idempotent(
            conn,
            "ALTER TABLE memories ADD COLUMN tier TEXT NOT NULL DEFAULT 'warm'",
        )?;
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
        add_column_idempotent(conn, "ALTER TABLE memories ADD COLUMN cluster_id INTEGER")?;
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
        add_column_idempotent(
            conn,
            "ALTER TABLE memories ADD COLUMN needs_vec_dedup INTEGER NOT NULL DEFAULT 0",
        )?;
        // Backfill: mark all existing active memories for vec dedup sweep
        conn.execute_batch("UPDATE memories SET needs_vec_dedup = 1 WHERE status = 'active'")?;
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
    )?;

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

    // v0.26 Cap C: cold-tier archival summary fields. Same ordering rationale
    // as `migrate_resummerize` — must run AFTER `migrate_source_check` so the
    // table-recreate path doesn't silently drop the new columns.
    migrate_cold_archive_summary(conn)?;

    // v0.27.1 E direction: judge_call_ledger table for J2 atomic call-cap
    // reservation. Idempotent — safe to call on already-migrated DBs.
    migrate_judge_call_ledger(conn)?;

    // v0.27.1 E direction: concepts.living_summary_id column (Cap A summary
    // instance ULID, R8 P1 fix) + concept_summary_instances retention table
    // (R9-K3) so the judge can validate J5 against an immutable snapshot
    // even after a subsequent refresh overwrites the live row.
    migrate_concepts_living_summary_id(conn)?;
    migrate_concept_summary_instances(conn)?;

    // v0.27.3 F4 A3 fix — partial UNIQUE index on feedback_events for
    // OfflineCron event dedup. Closes the concurrent-cron emit race that
    // the prior LIKE check + reserve_call + emit_event check-then-act
    // pattern could not (two concurrent cron passes can both pass the
    // LIKE check, both make the HTTP call, both emit). Constraint
    // violations are absorbed in the cron emit path as no-ops.
    migrate_offlinecron_dedup_index(conn)?;

    // v0.27.5 R3: pre-LLM cron claim row. Closes the v0.27.4 R9 P2
    // residual race — two concurrent cron passes can both clear the
    // `cron_event_already_emitted` LIKE check, both `reserve_call`
    // burns of `daily_call_cap`, both pay for an LLM call, with the
    // second eventually losing on the F4 A3 UNIQUE index. The
    // `cron_claims` row inserted via INSERT OR IGNORE BEFORE
    // `reserve_call` makes the dedup atomic and pre-LLM, so only the
    // claim winner pays.
    migrate_cron_claims(conn)?;
    migrate_oauth_tables(conn)?;

    Ok(())
}

/// v0.29 Phase A — OAuth provider persistence.
///
/// The tables are created during normal schema init so a future switch to
/// `[server].auth = "oauth"` does not require a separate operator migration.
/// A single HS256 signing key is generated once per database and kept in
/// `oauth_signing_keys`; later rotations add rows rather than replacing it.
///
/// v0.31 candidate (Agent F4 / A-H2): gated by `schema_migrations` row so
/// every steady-state `open_store()` call short-circuits after one cheap
/// `SELECT version FROM schema_migrations WHERE name='oauth'` lookup.  The
/// fingerprint-backfill UPDATE only runs when the column was just added.
pub fn migrate_oauth_tables(conn: &Connection) -> ReinResult<()> {
    // 1. Cheap version probe: `schema_migrations` is a small generic table we
    //    create on demand.  Avoids claiming SQLite's single global
    //    `PRAGMA user_version` slot (which other migrators may want).
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS schema_migrations (
            name TEXT PRIMARY KEY,
            version INTEGER NOT NULL,
            applied_at INTEGER NOT NULL
        );",
    )?;
    let current_version: i64 = conn
        .query_row(
            "SELECT version FROM schema_migrations WHERE name = 'oauth'",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0);
    if current_version >= OAUTH_SCHEMA_VERSION {
        // Already at (or beyond) the target version — steady-state fast path.
        return Ok(());
    }

    // 2. Create base tables.  Idempotent CREATEs let upgrades from older
    //    DBs converge to the current shape without separate migration code.
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS oauth_clients (
            client_id TEXT PRIMARY KEY,
            client_secret_hash TEXT,
            client_name TEXT NOT NULL,
            redirect_uris TEXT NOT NULL,
            grant_types TEXT NOT NULL,
            token_endpoint_auth_method TEXT NOT NULL CHECK(token_endpoint_auth_method IN ('none', 'client_secret_basic')),
            registered_at INTEGER NOT NULL,
            last_used_at INTEGER,
            revoked_at INTEGER
        );

        CREATE TABLE IF NOT EXISTS oauth_auth_codes (
            code TEXT PRIMARY KEY,
            client_id TEXT NOT NULL REFERENCES oauth_clients(client_id),
            redirect_uri TEXT NOT NULL,
            code_challenge TEXT NOT NULL,
            code_challenge_method TEXT NOT NULL CHECK(code_challenge_method = 'S256'),
            issued_at INTEGER NOT NULL,
            expires_at INTEGER NOT NULL,
            consumed_at INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_oauth_auth_codes_expires
            ON oauth_auth_codes(expires_at);

        CREATE TABLE IF NOT EXISTS oauth_grants (
            grant_id TEXT PRIMARY KEY,
            client_id TEXT NOT NULL REFERENCES oauth_clients(client_id),
            access_token_jti TEXT NOT NULL UNIQUE,
            access_expires_at INTEGER NOT NULL,
            refresh_token_hash TEXT NOT NULL,
            refresh_token_fingerprint TEXT NOT NULL,
            refresh_expires_at INTEGER NOT NULL,
            issued_at INTEGER NOT NULL,
            revoked_at INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_oauth_grants_jti
            ON oauth_grants(access_token_jti);
        CREATE INDEX IF NOT EXISTS idx_oauth_grants_client
            ON oauth_grants(client_id);

        CREATE TABLE IF NOT EXISTS oauth_signing_keys (
            kid TEXT PRIMARY KEY,
            secret_hex TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            rotated_at INTEGER
        );
        ",
    )?;

    // 3. v0.30.0 refresh_token_fingerprint column ALTER (release-day
    //    migration documented in v0.30.0 devlog).  Two cases:
    //
    //    a) Column missing — fresh v0.29.x→v0.31 upgrade. ALTER first,
    //       then the NULL cleanup below will revoke every pre-existing
    //       grant exactly once.
    //    b) Column present — either a fresh install (CREATE TABLE above
    //       declared the column from the start, no NULL rows possible)
    //       OR a v0.30.x boot that crashed AFTER the ALTER but BEFORE
    //       the cleanup UPDATE, leaving stale NULL-fingerprint rows.
    //
    //    R1 P2-#2: case (b)-crash must still revoke those NULL rows
    //    before this function returns, otherwise the schema_migrations
    //    gate marks OAuth v1 complete with revocable grants left
    //    unrevoked — `active_grant_by_jti` still accepts their access
    //    tokens until normal JWT `exp` expiry.  The earlier gate-on-ALTER
    //    branch skipped cleanup whenever the column already existed,
    //    which silently swallowed this case.
    //
    //    The COUNT(*) probe is cheap on the steady-state path (post-v1
    //    schema migration version pin still short-circuits the whole
    //    function before we even reach here on subsequent reopens; this
    //    path runs at most once per crash-recovery).
    let has_refresh_fingerprint: bool = conn
        .query_row(
            "SELECT COUNT(*) FROM pragma_table_info('oauth_grants') WHERE name='refresh_token_fingerprint'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .unwrap_or(0)
        > 0;
    if !has_refresh_fingerprint {
        add_column_idempotent(
            conn,
            "ALTER TABLE oauth_grants ADD COLUMN refresh_token_fingerprint TEXT",
        )?;
    }
    // Always check for NULL-fingerprint rows.  After the ALTER above (case
    // a) every existing row will satisfy this; after a crash-mid-migration
    // boot (case b) the leftover rows will.  Steady-state on a healthy
    // schema returns 0 and skips the UPDATE — equivalent perf to the
    // previous gate-on-ALTER branch but closes the crash-recovery hole.
    let null_fingerprint_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM oauth_grants WHERE refresh_token_fingerprint IS NULL",
            [],
            |row| row.get::<_, i64>(0),
        )
        .unwrap_or(0);
    if null_fingerprint_count > 0 {
        conn.execute(
            "UPDATE oauth_grants
             SET revoked_at = COALESCE(revoked_at, ?1)
             WHERE refresh_token_fingerprint IS NULL",
            [chrono::Utc::now().timestamp()],
        )?;
    }
    conn.execute_batch(
        "CREATE INDEX IF NOT EXISTS idx_oauth_grants_refresh_fingerprint
            ON oauth_grants(client_id, refresh_token_fingerprint);",
    )?;

    let key_count: i64 = conn.query_row("SELECT COUNT(*) FROM oauth_signing_keys", [], |row| {
        row.get(0)
    })?;
    if key_count == 0 {
        let mut secret = [0u8; 32];
        let rng = ring::rand::SystemRandom::new();
        ring::rand::SecureRandom::fill(&rng, &mut secret)
            .map_err(|_| ReinError::Config("failed to generate OAuth signing key".to_string()))?;
        let secret_hex = secret
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        let now = chrono::Utc::now().timestamp();
        let kid = ulid::Ulid::new().to_string();
        conn.execute(
            "INSERT INTO oauth_signing_keys (kid, secret_hex, created_at, rotated_at)
             VALUES (?1, ?2, ?3, NULL)",
            rusqlite::params![kid, secret_hex, now],
        )?;
    }

    // 4. Record completion at the new schema version.  Use INSERT OR REPLACE
    //    so re-running a partial migration (e.g. process crashed before this
    //    line) bumps the version correctly.
    conn.execute(
        "INSERT OR REPLACE INTO schema_migrations (name, version, applied_at)
         VALUES ('oauth', ?1, ?2)",
        rusqlite::params![OAUTH_SCHEMA_VERSION, chrono::Utc::now().timestamp()],
    )?;

    Ok(())
}

/// v0.27.5 R3 — pre-LLM cron claim table. PRIMARY KEY on
/// `(event_type, surface_id, stamp_hash)` lets `INSERT OR IGNORE`
/// atomically arbitrate between concurrent cron workers: only the
/// first writer to claim a tuple proceeds to `reserve_call`/LLM/emit;
/// the loser sees `INSERT OR IGNORE` affect 0 rows and short-circuits
/// to `SkippedDuplicate` without burning any `daily_call_cap` quota.
///
/// `claim_token` (Codex R3 P2 fix) makes `release_cron_claim` safe
/// after stale-claim takeover: every successful claim mints a fresh
/// ULID, takeover overwrites the token, and DELETE predicates on the
/// caller's token so a slow original cron whose claim was stolen by
/// a fresh peer can't accidentally clear the new owner's row.
///
/// Idempotent. Safe to call on fresh DBs and DBs that already have the
/// table (the `claim_token` ALTER is gated on a `pragma_table_info`
/// check so re-runs are no-ops).
pub fn migrate_cron_claims(conn: &Connection) -> ReinResult<()> {
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS cron_claims (
            event_type TEXT NOT NULL,
            surface_id TEXT NOT NULL,
            stamp_hash TEXT NOT NULL,
            claim_token TEXT NOT NULL DEFAULT '',
            claimed_at INTEGER NOT NULL,
            PRIMARY KEY (event_type, surface_id, stamp_hash)
        );
        CREATE INDEX IF NOT EXISTS idx_cron_claims_claimed_at
            ON cron_claims(claimed_at);
        ",
    )?;
    // Idempotent ALTER for any DB that landed on the v0.27.5 pre-codex-R3
    // schema (no `claim_token` column). `pragma_table_info` guards prevent
    // re-running the ALTER on already-migrated DBs.
    let has_claim_token: bool = conn
        .query_row(
            "SELECT COUNT(*) FROM pragma_table_info('cron_claims') WHERE name='claim_token'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .unwrap_or(0)
        > 0;
    if !has_claim_token {
        add_column_idempotent(
            conn,
            "ALTER TABLE cron_claims ADD COLUMN claim_token TEXT NOT NULL DEFAULT ''",
        )?;
    }
    Ok(())
}

/// v0.27.3 F4 A3 — partial UNIQUE index on feedback_events for OfflineCron
/// event dedup. SQLite supports indexes on JSON-extract expressions and
/// partial indexes via `WHERE`. The index covers
/// `(event_type, synthesis_id|concept_summary_id, stamp_hash)` and only
/// applies to the two OfflineCron event types so the rest of
/// feedback_events is untouched.
///
/// Idempotent.
pub fn migrate_offlinecron_dedup_index(conn: &Connection) -> ReinResult<()> {
    // v0.27.4 codex R1 P1: backfill duplicate rows BEFORE creating the
    // UNIQUE index. Existing DBs that hit the v0.27.x concurrent
    // OfflineCron emit race already have duplicates by
    // `(event_type, surface_id, stamp_hash)`; CREATE UNIQUE INDEX would
    // fail and lock the user out of `init_schema`/`open_store`. Coalesce
    // by keeping the lowest `id` per duplicate group (oldest row wins,
    // preserving the offset that the consumer already advanced past).
    conn.execute_batch(
        "DELETE FROM feedback_events \
         WHERE event_type IN ( \
                 'synthesis_llm_judge_offline_cron', \
                 'concept_summary_llm_judge_offline_cron' \
             ) \
           AND json_valid(payload) \
           AND id NOT IN ( \
                 SELECT MIN(id) FROM feedback_events \
                 WHERE event_type IN ( \
                         'synthesis_llm_judge_offline_cron', \
                         'concept_summary_llm_judge_offline_cron' \
                     ) \
                   AND json_valid(payload) \
                 GROUP BY \
                     event_type, \
                     COALESCE( \
                         json_extract(payload, '$.synthesis_id'), \
                         json_extract(payload, '$.concept_summary_id') \
                     ), \
                     json_extract(payload, '$.stamp_hash') \
             );",
    )?;
    // codex R10 P2: guard `json_extract` against malformed `payload`
    // text. SQLite raises `malformed JSON` when `json_extract` runs on
    // a non-JSON string, which would make `init_schema` (and therefore
    // `open_store`) fail at startup on any DB that has even a single
    // OfflineCron event with a corrupted payload. The runtime
    // consumers already tolerate malformed payloads by skipping them;
    // the migration/index must be at least as forgiving. We add
    // `json_valid(payload)` to both the partial index `WHERE` clause
    // (so the index simply omits malformed rows) and the dedup-DELETE
    // above (so the GROUP BY never invokes `json_extract` on bad rows).
    conn.execute_batch(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_feedback_events_offlinecron_dedup \
         ON feedback_events( \
             event_type, \
             COALESCE( \
                 json_extract(payload, '$.synthesis_id'), \
                 json_extract(payload, '$.concept_summary_id') \
             ), \
             json_extract(payload, '$.stamp_hash') \
         ) \
         WHERE event_type IN ( \
             'synthesis_llm_judge_offline_cron', \
             'concept_summary_llm_judge_offline_cron' \
         ) \
           AND json_valid(payload);",
    )?;
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
        add_column_idempotent(
            conn,
            "ALTER TABLE memories ADD COLUMN needs_resummerize INTEGER NOT NULL DEFAULT 0",
        )?;
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
        add_column_idempotent(
            conn,
            "ALTER TABLE memories ADD COLUMN last_resummarized_at TEXT",
        )?;
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
        add_column_idempotent(
            conn,
            "ALTER TABLE memories ADD COLUMN in_progress_resummerize_at TEXT",
        )?;
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

/// v0.26 Cap C migration: adds cold-tier archival-summary fields to
/// `memories`. Idempotent — re-running on a DB that already has the columns
/// is a no-op. No backfill: existing cold rows stay NULL until the next
/// `run_tiering` pass re-flags them via `needs_archival_summary = 1`.
///
/// Columns:
/// * `archival_summary` — LLM-generated condensed summary (NULL until worker
///   runs)
/// * `archival_summary_at` — Unix epoch seconds when the summary was last
///   generated (NULL iff `archival_summary` is NULL)
/// * `archival_summary_version` — `ARCHIVAL_SUMMARY_VERSION` snapshot at
///   generation time; bumps invalidate prior summaries (NULL iff
///   `archival_summary` is NULL)
/// * `needs_archival_summary` — worker-internal flag set by `run_tiering`
///   on cold transition, cleared by `run_cold_archive_summary` after
///   successful contract-gated persist
fn migrate_cold_archive_summary(conn: &Connection) -> ReinResult<()> {
    let has_archival_summary: bool = conn
        .query_row(
            "SELECT COUNT(*) FROM pragma_table_info('memories') WHERE name='archival_summary'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .unwrap_or(0)
        > 0;
    if !has_archival_summary {
        // Per-column fail-loud ALTERs (a peer-race duplicate on the first must
        // not abort the rest).
        add_column_idempotent(
            conn,
            "ALTER TABLE memories ADD COLUMN archival_summary TEXT",
        )?;
        add_column_idempotent(
            conn,
            "ALTER TABLE memories ADD COLUMN archival_summary_at INTEGER",
        )?;
        add_column_idempotent(
            conn,
            "ALTER TABLE memories ADD COLUMN archival_summary_version INTEGER",
        )?;
        add_column_idempotent(
            conn,
            "ALTER TABLE memories ADD COLUMN needs_archival_summary INTEGER NOT NULL DEFAULT 0",
        )?;
    }
    let has_in_progress: bool = conn
        .query_row(
            "SELECT COUNT(*) FROM pragma_table_info('memories') WHERE name='in_progress_archival_summary_at'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .unwrap_or(0)
        > 0;
    if !has_in_progress {
        add_column_idempotent(
            conn,
            "ALTER TABLE memories ADD COLUMN in_progress_archival_summary_at TEXT",
        )?;
        add_column_idempotent(
            conn,
            "ALTER TABLE memories ADD COLUMN archival_claim_token TEXT",
        )?;
    }
    // v0.27.5 R1: too-large backoff — `claim_batch` deprioritizes rows whose
    // last attempt returned `AttemptOutcome::TooLarge` so a permanently
    // oversized row doesn't starve newer eligible rows. Unix epoch seconds;
    // NULL means "never marked too-large" and sorts first in the claim
    // ORDER BY (highest priority).
    let has_last_too_large: bool = conn
        .query_row(
            "SELECT COUNT(*) FROM pragma_table_info('memories') WHERE name='last_too_large_at'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .unwrap_or(0)
        > 0;
    if !has_last_too_large {
        add_column_idempotent(
            conn,
            "ALTER TABLE memories ADD COLUMN last_too_large_at INTEGER",
        )?;
    }
    // No backfill: existing cold rows stay NULL until run_tiering re-flags them.
    Ok(())
}

/// v0.27.1 E direction (spec §4 J2 + R8 P1 fix): atomic call-cap reservation
/// ledger for the runtime LLM judge worker.
///
/// `judge_call_ledger.id` is a ULID minted by the worker before the LLM HTTP
/// call; `status` transitions `reserved → done | failed | stale`.
/// `done` = LLM call succeeded. `failed` = LLM call attempted but errored
/// or returned unparseable output (still counts toward daily_call_cap as
/// it incurred an HTTP cost). `stale` = reservation never made it to the
/// LLM call (worker crash, reaped by next sweep) and is excluded from
/// the cap count — Codex R4 P2 fix.
/// The rolling 24h reservation count is computed from this table inside
/// `BEGIN IMMEDIATE` so two dispatchers can't both observe the same
/// below-cap count and burst `N × cap` calls. Stale `reserved` rows
/// older than 5 minutes are reaped by the worker on each pull
/// (worker-crash recovery; mirrors v0.23 resummerize claim-token pattern).
///
/// Idempotent: safe to call on already-migrated DBs. Pre-v0.27.1 R4
/// rows with status='failed' that were actually stale-reaped (rather
/// than LLM-attempted) will incorrectly count toward the cap until they
/// age out of the 24h window — acceptable transient cost for the
/// schema migration.
pub fn migrate_judge_call_ledger(conn: &Connection) -> ReinResult<()> {
    // CHECK constraints can't be altered; we re-create the table only if
    // the old constraint is in place. Detect via PRAGMA — if the table
    // exists with the old shape, drop+recreate. Since this is brand-new
    // in v0.27.1, drop is safe (no production data yet).
    let needs_migrate: bool = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master \
             WHERE type='table' AND name='judge_call_ledger' \
             AND sql NOT LIKE '%''stale''%'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .unwrap_or(0)
        > 0;
    if needs_migrate {
        // Fail loud on real errors (lock/disk), but tolerate a peer that
        // already dropped the old-shape table in a concurrent first open.
        match conn.execute("DROP TABLE judge_call_ledger", []) {
            Ok(_) => {}
            Err(e) if e.to_string().contains("no such table") => {}
            Err(e) => return Err(e.into()),
        }
    }
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS judge_call_ledger (
            id TEXT PRIMARY KEY,
            ts INTEGER NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('reserved','done','failed','stale'))
        );
        CREATE INDEX IF NOT EXISTS idx_judge_call_ledger_ts_status
            ON judge_call_ledger(ts, status);
        ",
    )?;
    Ok(())
}

/// v0.27.1 E direction (spec §3.2 R8 P1 + §15 R9-K3): add
/// `concepts.living_summary_id TEXT` column. Minted as a ULID on every
/// `refresh_living_summary` call so judge events can link back to a stable
/// per-instance id.
///
/// Idempotent.
pub fn migrate_concepts_living_summary_id(conn: &Connection) -> ReinResult<()> {
    let has_col: bool = conn
        .query_row(
            "SELECT COUNT(*) FROM pragma_table_info('concepts') WHERE name='living_summary_id'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .unwrap_or(0)
        > 0;
    if !has_col {
        add_column_idempotent(
            conn,
            "ALTER TABLE concepts ADD COLUMN living_summary_id TEXT",
        )?;
    }
    Ok(())
}

/// v0.27.1 E direction (spec §15 R9-K3): retention table for concept-summary
/// instances. When a concept is refreshed before a queued/manual judge
/// processes the previous summary, `concepts.living_summary_id` is
/// overwritten — without this table, J5 link-present can't validate the
/// old summary. Rows are retained for 7 days; pruned by the worker reaper.
///
/// Idempotent.
pub fn migrate_concept_summary_instances(conn: &Connection) -> ReinResult<()> {
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS concept_summary_instances (
            summary_id TEXT PRIMARY KEY,
            concept_id TEXT NOT NULL,
            summary_text TEXT NOT NULL,
            refreshed_at INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_concept_summary_instances_concept
            ON concept_summary_instances(concept_id, refreshed_at DESC);
        CREATE INDEX IF NOT EXISTS idx_concept_summary_instances_refreshed
            ON concept_summary_instances(refreshed_at);
        ",
    )?;
    Ok(())
}

fn migrate_episode_columns(conn: &Connection) -> ReinResult<()> {
    let ensure_column = |name: &str, ddl: &str| -> ReinResult<()> {
        let has_column: bool = conn
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('episodes') WHERE name=?1",
                rusqlite::params![name],
                |row| row.get::<_, i64>(0),
            )
            .unwrap_or(0)
            > 0;
        if !has_column {
            add_column_idempotent(conn, ddl)?;
        }
        Ok(())
    };

    ensure_column(
        "primary_topics",
        "ALTER TABLE episodes ADD COLUMN primary_topics TEXT NOT NULL DEFAULT '[]'",
    )?;
    ensure_column(
        "tags",
        "ALTER TABLE episodes ADD COLUMN tags TEXT NOT NULL DEFAULT '[]'",
    )?;
    ensure_column(
        "involved_agents",
        "ALTER TABLE episodes ADD COLUMN involved_agents TEXT NOT NULL DEFAULT '[]'",
    )?;
    ensure_column(
        "important_paths",
        "ALTER TABLE episodes ADD COLUMN important_paths TEXT NOT NULL DEFAULT '[]'",
    )?;
    ensure_column(
        "temporal_keywords",
        "ALTER TABLE episodes ADD COLUMN temporal_keywords TEXT NOT NULL DEFAULT '[]'",
    )?;
    ensure_column(
        "source_session_id",
        "ALTER TABLE episodes ADD COLUMN source_session_id TEXT",
    )?;
    conn.execute_batch(
        "CREATE INDEX IF NOT EXISTS idx_episodes_source_session ON episodes(source_session_id)",
    )?;

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
        // #17 codex R10: credit the embedding churn counter INSIDE the swap
        // savepoint. A wholesale index replacement rewrites every vector at
        // an unchanged row count — committed separately, an interruption
        // right after the swap would leave the recluster cadence gate
        // closed on old-space cluster labels indefinitely. Riding the same
        // commit makes the gate's reopening exactly as durable as the new
        // vectors themselves (the next adaptive pass then reclusters and
        // rewipes any stale cluster-scoped state — self-healing even if
        // the caller's follow-up resets never ran).
        conn.execute(
            "INSERT INTO metadata(key, value) VALUES (?1, CAST(?2 AS TEXT))
             ON CONFLICT(key) DO UPDATE SET value = CAST(CAST(value AS INTEGER) + ?2 AS TEXT)",
            rusqlite::params![
                crate::store::vec::EMBEDDING_WRITE_SEQ_KEY,
                embeddings.len() as i64
            ],
        )?;
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
            embedding BLOB NOT NULL,
            text_hash TEXT NOT NULL DEFAULT ''
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
    replace_vector_index_from_staging_validated(conn, dims, None).map(|_| ())
}

/// Metadata key recording `"<model>:<dims>"` of the vector rows (see
/// `search::warmup`). Written inside the swap savepoint so the table and its
/// provenance can never disagree.
pub const VEC_ROWS_PROVENANCE_KEY: &str = "vec_rows_provenance";

/// [`replace_vector_index_from_staging`] that also revalidates every staged
/// row against the current `memories` text: a row whose `text_hash` no longer
/// matches (edited, rewritten, or deleted while embeddings were computed) is
/// skipped instead of published under a stale vector. Rows staged without a
/// hash (legacy callers) are published as before. Returns the skipped count.
pub fn replace_vector_index_from_staging_validated(
    conn: &Connection,
    dims: usize,
    provenance: Option<&str>,
) -> ReinResult<usize> {
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
    let mut skipped = 0_usize;
    let result = (|| -> ReinResult<()> {
        conn.execute_batch("SAVEPOINT replace_vector_index_staged")?;
        conn.execute_batch("DROP TABLE IF EXISTS vec_memories;")?;
        conn.execute_batch("DELETE FROM embed_cache;")?;
        conn.execute_batch(&vec_sql)?;
        {
            let mut read = conn.prepare("SELECT id, embedding, text_hash FROM embed_staging")?;
            let mut write =
                conn.prepare("INSERT OR REPLACE INTO vec_memories(id, embedding) VALUES (?1, ?2)")?;
            let mut current =
                conn.prepare("SELECT topic, summary, content FROM memories WHERE id = ?1")?;
            let mut rows = read.query([])?;
            while let Some(row) = rows.next()? {
                let id: String = row.get(0)?;
                let bytes: Vec<u8> = row.get(1)?;
                let staged_hash: String = row.get(2)?;
                if !staged_hash.is_empty() {
                    use rusqlite::OptionalExtension;
                    let live_hash = current
                        .query_row(rusqlite::params![id], |r| {
                            Ok(crate::store::migrate::enriched_text_hash(
                                &crate::embed::prepend_metadata(
                                    &r.get::<_, String>(0)?,
                                    &r.get::<_, String>(1)?,
                                    &r.get::<_, String>(2)?,
                                ),
                            ))
                        })
                        .optional()?;
                    if live_hash.as_deref() != Some(staged_hash.as_str()) {
                        skipped += 1;
                        continue;
                    }
                }
                write.execute(rusqlite::params![id, bytes])?;
            }
        }
        // #17 codex R10: churn credit inside the swap savepoint — see the
        // twin comment in `replace_vector_index` for the interrupted-
        // reindex rationale.
        conn.execute(
            "INSERT INTO metadata(key, value) VALUES (?1, CAST(?2 AS TEXT))
             ON CONFLICT(key) DO UPDATE SET value = CAST(CAST(value AS INTEGER) + ?2 AS TEXT)",
            rusqlite::params![crate::store::vec::EMBEDDING_WRITE_SEQ_KEY, row_count],
        )?;
        if let Some(provenance) = provenance {
            conn.execute(
                "INSERT INTO metadata(key, value) VALUES (?1, ?2) \
                 ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                rusqlite::params![VEC_ROWS_PROVENANCE_KEY, provenance],
            )?;
        }
        conn.execute_batch("DROP TABLE embed_staging;")?;
        conn.execute_batch("RELEASE replace_vector_index_staged")?;
        Ok(())
    })();
    if result.is_err() {
        let _ = conn.execute_batch("ROLLBACK TO replace_vector_index_staged");
        let _ = conn.execute_batch("RELEASE replace_vector_index_staged");
    }
    if skipped > 0 {
        tracing::warn!(
            skipped,
            "vector index swap skipped rows whose memory changed after staging; the next warmup fills them"
        );
    }
    result.map(|_| skipped)
}

#[cfg(test)]
mod migration_tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn a12_input_epoch(conn: &Connection) -> Option<u64> {
        conn.query_row(
            "SELECT value FROM metadata WHERE key = 'a12_input_epoch'",
            [],
            |row| row.get::<_, String>(0),
        )
        .ok()
        .and_then(|value| value.parse().ok())
    }

    #[test]
    fn a12_input_epoch_migration_is_initialized_and_idempotent() {
        init_sqlite_vec();
        let conn = Connection::open_in_memory().expect("open memory db");

        init_schema(&conn, 4).expect("first schema init");
        assert_eq!(schema_user_version(&conn).unwrap(), 4);
        assert_eq!(a12_input_epoch(&conn), Some(0));

        init_schema(&conn, 4).expect("second schema init");
        assert_eq!(a12_input_epoch(&conn), Some(0));
        let trigger_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master \
                 WHERE type = 'trigger' AND name GLOB 'a12_input_epoch_*'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(trigger_count, 21, "v4 installs one trigger per write path");
    }

    #[test]
    fn a12_input_epoch_row_self_heals_on_reopen_without_resetting_live_counter() {
        init_sqlite_vec();
        let conn = Connection::open_in_memory().expect("open memory db");
        init_schema(&conn, 4).expect("first schema init");

        // An advanced counter survives bring-up untouched.
        conn.execute(
            "UPDATE metadata SET value = '57' WHERE key = 'a12_input_epoch'",
            [],
        )
        .unwrap();
        init_schema(&conn, 4).expect("re-open with advanced counter");
        assert_eq!(a12_input_epoch(&conn), Some(57));

        // Deleting the row wedges every epoch-guarded write path...
        conn.execute("DELETE FROM metadata WHERE key = 'a12_input_epoch'", [])
            .unwrap();
        let wedged = conn.execute(
            "INSERT INTO metadata(key, value) VALUES ('adaptive_state', '{}')",
            [],
        );
        assert!(
            wedged
                .unwrap_err()
                .to_string()
                .contains("a12_input_epoch is missing"),
            "missing epoch row must abort epoch-guarded writes"
        );

        // ...and the next open self-heals it at zero instead of leaving the
        // database permanently write-wedged.
        init_schema(&conn, 4).expect("re-open after epoch row deletion");
        assert_eq!(a12_input_epoch(&conn), Some(0));
        conn.execute(
            "INSERT INTO metadata(key, value) VALUES ('adaptive_state', '{}')",
            [],
        )
        .expect("writes must work again after the self-heal");
        assert_eq!(a12_input_epoch(&conn), Some(1));
    }

    #[test]
    fn a12_epoch_ignores_only_valid_equal_adaptive_projection_digests() {
        init_sqlite_vec();
        let conn = Connection::open_in_memory().expect("open memory db");
        init_schema(&conn, 4).expect("schema init");

        let state = crate::store::adaptive::AdaptiveState::default();
        state.save_snapshot(&conn).expect("first adaptive snapshot");
        let after_insert = a12_input_epoch(&conn).unwrap();

        let mut version_only =
            crate::store::adaptive::AdaptiveState::restore_snapshot(&conn).unwrap();
        version_only.version += 1;
        version_only
            .save_snapshot(&conn)
            .expect("version-only adaptive update");
        assert_eq!(
            a12_input_epoch(&conn),
            Some(after_insert),
            "bookkeeping-only adaptive writes must not invalidate A12"
        );

        let mut replay_relevant =
            crate::store::adaptive::AdaptiveState::restore_snapshot(&conn).unwrap();
        replay_relevant.version += 1;
        replay_relevant.cluster_version += 1;
        replay_relevant
            .save_snapshot(&conn)
            .expect("projection-changing adaptive update");
        assert!(
            a12_input_epoch(&conn).unwrap() > after_insert,
            "a replay-relevant adaptive change must invalidate A12"
        );
    }

    #[test]
    fn a12_epoch_mixed_version_adaptive_writer_fails_closed_repeatedly() {
        init_sqlite_vec();
        let conn = Connection::open_in_memory().expect("open memory db");
        init_schema(&conn, 4).expect("schema init");
        crate::store::adaptive::AdaptiveState::default()
            .save_snapshot(&conn)
            .expect("current writer snapshot");

        let before_legacy = a12_input_epoch(&conn).unwrap();
        conn.execute(
            "UPDATE metadata
             SET value = json_remove(value, '$.a12_recall_projection_fingerprint')
             WHERE key = 'adaptive_state'",
            [],
        )
        .unwrap();
        let after_digest_removal = a12_input_epoch(&conn).unwrap();
        assert!(after_digest_removal > before_legacy);

        conn.execute(
            "UPDATE metadata
             SET value = json_set(value, '$.version', 99)
             WHERE key = 'adaptive_state'",
            [],
        )
        .unwrap();
        assert!(
            a12_input_epoch(&conn).unwrap() > after_digest_removal,
            "a row without a valid digest must invalidate on every old-writer update"
        );
    }

    #[test]
    fn a12_epoch_migration_backfills_digest_without_losing_unknown_fields() {
        init_sqlite_vec();
        let conn = Connection::open_in_memory().expect("open memory db");
        init_schema_with_migrations(&conn, 4, &MIGRATIONS[..2]).expect("initialize through v3");

        let mut value = serde_json::to_value(crate::store::adaptive::AdaptiveState::default())
            .expect("serialize legacy adaptive state");
        value["future_writer_field"] = serde_json::json!({"preserve": true});
        value
            .as_object_mut()
            .unwrap()
            .remove("a12_recall_projection_fingerprint");
        conn.execute(
            "INSERT INTO metadata(key, value) VALUES ('adaptive_state', ?1)",
            [serde_json::to_string(&value).unwrap()],
        )
        .unwrap();

        init_schema(&conn, 4).expect("apply v4");
        let raw: String = conn
            .query_row(
                "SELECT value FROM metadata WHERE key = 'adaptive_state'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        let migrated: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(migrated["future_writer_field"]["preserve"], true);
        let digest = migrated["a12_recall_projection_fingerprint"]
            .as_str()
            .expect("v4 must add projection digest");
        assert_eq!(digest.len(), 64);
        assert!(digest.bytes().all(|byte| byte.is_ascii_hexdigit()));
    }

    #[test]
    fn a12_epoch_tracks_projection_change_through_adaptive_cas_merge() {
        init_sqlite_vec();
        let conn = Connection::open_in_memory().expect("open memory db");
        init_schema(&conn, 4).expect("schema init");
        crate::store::adaptive::AdaptiveState::default()
            .save_snapshot(&conn)
            .expect("initial snapshot");

        let base = crate::store::adaptive::AdaptiveState::restore_snapshot(&conn).unwrap();
        let mut peer = base.clone();
        peer.version += 1;
        peer.memory_clusters.insert("peer".to_string(), 1);
        peer.save_snapshot(&conn).expect("peer snapshot");
        let after_peer = a12_input_epoch(&conn).unwrap();

        let mut stale = base;
        stale.version += 1;
        stale.memory_clusters.insert("stale".to_string(), 2);
        stale.save_snapshot(&conn).expect("CAS-merged snapshot");
        assert!(a12_input_epoch(&conn).unwrap() > after_peer);
        let merged = crate::store::adaptive::AdaptiveState::restore_snapshot(&conn).unwrap();
        assert_eq!(merged.memory_clusters.get("peer"), Some(&1));
        assert_eq!(merged.memory_clusters.get("stale"), Some(&2));
        assert_eq!(
            merged.a12_recall_projection_fingerprint,
            merged.compute_a12_recall_projection_fingerprint().unwrap()
        );
    }

    #[test]
    fn a12_input_epoch_tracks_every_sqlite_recall_input_without_recursing() {
        init_sqlite_vec();
        let conn = Connection::open_in_memory().expect("open memory db");
        init_schema(&conn, 4).expect("schema init");
        conn.pragma_update(None, "recursive_triggers", true)
            .unwrap();

        let mut prior = a12_input_epoch(&conn).unwrap();
        let mut assert_advanced = |label: &str| {
            let next = a12_input_epoch(&conn).unwrap();
            assert!(next > prior, "{label} must advance A12 input epoch");
            prior = next;
        };

        conn.execute_batch(
            "INSERT INTO memories (id, layer, topic, summary, content, importance, source,
                 strength, decay_lambda, created_at, updated_at, last_accessed)
             VALUES ('epoch-memory', 'LTM', 't', 's', 'c', 'medium', 'manual',
                 1.0, 0.0, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z',
                 '2026-01-01T00:00:00Z');",
        )
        .unwrap();
        assert_advanced("memory insert");

        conn.execute(
            "INSERT INTO memory_evidence (
                 id, memory_id, canonical_id, source, source_topic, summary, content,
                 keywords, created_at, imported_at
             ) VALUES ('epoch-evidence', 'epoch-memory', 'epoch-memory', 'manual',
                 't', 's', 'evidence', '[]', '2026-01-01T00:00:00Z',
                 '2026-01-01T00:00:00Z')",
            [],
        )
        .unwrap();
        assert_advanced("evidence insert");

        conn.execute_batch(
            "INSERT INTO memoirs(id, name, description, created_at, updated_at)
             VALUES ('epoch-memoir', 'epoch memoir', '', '2026-01-01T00:00:00Z',
                     '2026-01-01T00:00:00Z');
             INSERT INTO concepts(id, memoir_id, name, definition, created_at, updated_at)
             VALUES ('epoch-concept-a', 'epoch-memoir', 'a', 'definition a',
                     '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z');",
        )
        .unwrap();
        assert_advanced("concept insert");

        conn.execute(
            "INSERT INTO concepts(id, memoir_id, name, definition, created_at, updated_at)
             VALUES ('epoch-concept-b', 'epoch-memoir', 'b', 'definition b',
                     '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
            [],
        )
        .unwrap();
        assert_advanced("second concept insert");

        conn.execute(
            "INSERT INTO concept_links(id, source_id, target_id, relation, created_at)
             VALUES ('epoch-link', 'epoch-concept-a', 'epoch-concept-b', 'related',
                     '2026-01-01T00:00:00Z')",
            [],
        )
        .unwrap();
        assert_advanced("concept-link insert");

        conn.execute(
            "INSERT INTO episodes(id, title, created_at)
             VALUES ('epoch-episode', 'episode', '2026-01-01T00:00:00Z')",
            [],
        )
        .unwrap();
        assert_advanced("episode insert");

        crate::store::vec::insert_embedding(&conn, "epoch-memory", &[0.0; 4]).unwrap();
        assert_advanced("embedding write sequence");

        conn.execute(
            "INSERT INTO metadata(key, value) VALUES ('adaptive_state', '{}')",
            [],
        )
        .unwrap();
        assert_advanced("adaptive metadata");

        conn.execute(
            "INSERT INTO metadata(key, value) VALUES ('rerank_weights', '{}')",
            [],
        )
        .unwrap();
        assert_advanced("rerank metadata");

        conn.execute(
            "INSERT INTO metadata(key, value) VALUES ('survival_curve:global', '{}')",
            [],
        )
        .unwrap();
        assert_advanced("survival metadata");

        conn.execute(
            "INSERT INTO metadata(key, value) VALUES ('unrelated_epoch_probe', '1')",
            [],
        )
        .unwrap();
        assert_eq!(
            a12_input_epoch(&conn),
            Some(prior),
            "unrelated metadata must not invalidate A12"
        );
    }

    #[test]
    fn corrupt_a12_input_epoch_aborts_source_writes_without_resetting_counter() {
        init_sqlite_vec();
        for corrupt in [
            Some("corrupt"),
            Some("-1"),
            Some("9223372036854775808"),
            None,
        ] {
            let conn = Connection::open_in_memory().expect("open memory db");
            init_schema(&conn, 4).expect("schema init");
            match corrupt {
                Some(raw) => {
                    conn.execute(
                        "UPDATE metadata SET value = ?1 WHERE key = 'a12_input_epoch'",
                        [raw],
                    )
                    .unwrap();
                }
                None => {
                    conn.execute("DELETE FROM metadata WHERE key = 'a12_input_epoch'", [])
                        .unwrap();
                }
            }

            let result = conn.execute_batch(
                "INSERT INTO memories (id, layer, topic, summary, content, importance, source,
                     strength, decay_lambda, created_at, updated_at, last_accessed)
                 VALUES ('must-rollback', 'LTM', 't', 's', 'c', 'medium', 'manual',
                     1.0, 0.0, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z',
                     '2026-01-01T00:00:00Z');",
            );

            assert!(
                result.is_err(),
                "bad epoch {corrupt:?} must fail the source write"
            );
            let after = conn
                .query_row(
                    "SELECT value FROM metadata WHERE key = 'a12_input_epoch'",
                    [],
                    |row| row.get::<_, String>(0),
                )
                .ok();
            assert_eq!(
                after.as_deref(),
                corrupt,
                "the trigger must preserve corruption"
            );
            let leaked: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM memories WHERE id = 'must-rollback'",
                    [],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(leaked, 0, "the source mutation must roll back atomically");
        }
    }

    /// #17 codex R10 — a wholesale vector-index swap must credit the
    /// embedding churn counter inside the SAME commit, so an interrupted
    /// reindex can never leave new vectors behind a closed recluster gate.
    #[test]
    fn replace_vector_index_credits_churn_counter() {
        let store = crate::store::sqlite::SqliteStore::in_memory().unwrap();
        let before = crate::store::vec::embedding_write_seq(store.conn());
        let embeddings: Vec<(String, Vec<f32>)> = (0..3)
            .map(|i| (format!("m{i}"), vec![0.0f32; 3072]))
            .collect();
        replace_vector_index(store.conn(), 3072, &embeddings).unwrap();
        let after = crate::store::vec::embedding_write_seq(store.conn());
        assert_eq!(after - before, 3, "swap credits one write per row");
    }

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

    #[test]
    fn init_schema_creates_oauth_tables_and_initial_signing_key() {
        init_sqlite_vec();
        let conn = Connection::open_in_memory().expect("open memory db");
        init_schema(&conn, 4).expect("init schema");

        for table in [
            "oauth_clients",
            "oauth_auth_codes",
            "oauth_grants",
            "oauth_signing_keys",
        ] {
            let count: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?1",
                    [table],
                    |row| row.get(0),
                )
                .expect("query sqlite_master");
            assert_eq!(count, 1, "{table} should exist");
        }

        let key_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM oauth_signing_keys", [], |row| {
                row.get(0)
            })
            .expect("query signing keys");
        assert_eq!(key_count, 1, "schema init should create one signing key");

        let secret_hex: String = conn
            .query_row(
                "SELECT secret_hex FROM oauth_signing_keys LIMIT 1",
                [],
                |row| row.get(0),
            )
            .expect("query signing key secret");
        assert_eq!(secret_hex.len(), 64, "secret should be 32 random bytes hex");
    }

    #[test]
    fn oauth_migration_revokes_legacy_grants_without_refresh_fingerprint() {
        init_sqlite_vec();
        let conn = Connection::open_in_memory().expect("open memory db");
        conn.execute_batch(
            "
            CREATE TABLE oauth_clients (
                client_id TEXT PRIMARY KEY,
                client_secret_hash TEXT,
                client_name TEXT NOT NULL,
                redirect_uris TEXT NOT NULL,
                grant_types TEXT NOT NULL,
                token_endpoint_auth_method TEXT NOT NULL,
                registered_at INTEGER NOT NULL,
                last_used_at INTEGER,
                revoked_at INTEGER
            );
            CREATE TABLE oauth_grants (
                grant_id TEXT PRIMARY KEY,
                client_id TEXT NOT NULL REFERENCES oauth_clients(client_id),
                access_token_jti TEXT NOT NULL UNIQUE,
                access_expires_at INTEGER NOT NULL,
                refresh_token_hash TEXT NOT NULL,
                refresh_expires_at INTEGER NOT NULL,
                issued_at INTEGER NOT NULL,
                revoked_at INTEGER
            );
            INSERT INTO oauth_clients (
                client_id, client_secret_hash, client_name, redirect_uris, grant_types,
                token_endpoint_auth_method, registered_at, last_used_at, revoked_at
            ) VALUES (
                'legacy-client', NULL, 'legacy', '[]', '[]', 'none', 1, NULL, NULL
            );
            INSERT INTO oauth_grants (
                grant_id, client_id, access_token_jti, access_expires_at,
                refresh_token_hash, refresh_expires_at, issued_at, revoked_at
            ) VALUES (
                'legacy-grant', 'legacy-client', 'legacy-jti', 999999,
                'legacy-hash', 999999, 1, NULL
            );
            ",
        )
        .expect("create legacy oauth schema");

        migrate_oauth_tables(&conn).expect("migrate oauth tables");

        let has_fingerprint: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('oauth_grants') WHERE name='refresh_token_fingerprint'",
                [],
                |row| row.get(0),
            )
            .expect("query migrated column");
        assert_eq!(has_fingerprint, 1);

        let index_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name='idx_oauth_grants_refresh_fingerprint'",
                [],
                |row| row.get(0),
            )
            .expect("query migrated index");
        assert_eq!(index_count, 1);

        let revoked_at: Option<i64> = conn
            .query_row(
                "SELECT revoked_at FROM oauth_grants WHERE grant_id = 'legacy-grant'",
                [],
                |row| row.get(0),
            )
            .expect("query legacy grant");
        assert!(
            revoked_at.is_some(),
            "legacy unindexed refresh grants must be revoked instead of kept on the slow path"
        );
    }

    // v0.31 candidate (Agent F4 / A-H2) — regression: re-running
    // `migrate_oauth_tables` after the initial run must be a fast no-op via
    // the `schema_migrations` version gate.  The strict assertion is that
    // re-runs do not perform an `UPDATE oauth_grants ... WHERE
    // refresh_token_fingerprint IS NULL` (which scaled linearly with grant
    // count under the v0.30.x path).
    #[test]
    fn oauth_migration_is_idempotent_via_version_gate() {
        init_sqlite_vec();
        let conn = Connection::open_in_memory().expect("open memory db");

        // First migration: bootstraps everything.
        migrate_oauth_tables(&conn).expect("first migrate");
        let version_after_first: i64 = conn
            .query_row(
                "SELECT version FROM schema_migrations WHERE name = 'oauth'",
                [],
                |row| row.get(0),
            )
            .expect("query schema version");
        assert_eq!(version_after_first, OAUTH_SCHEMA_VERSION);

        // Insert a valid grant (with fingerprint) so that re-runs would, in
        // the absence of the version gate, still execute the COUNT/UPDATE.
        conn.execute(
            "INSERT INTO oauth_clients (
                client_id, client_secret_hash, client_name, redirect_uris, grant_types,
                token_endpoint_auth_method, registered_at, last_used_at, revoked_at
            ) VALUES ('c', NULL, 'c', '[]', '[]', 'none', 1, NULL, NULL)",
            [],
        )
        .expect("insert client");
        conn.execute(
            "INSERT INTO oauth_grants (
                grant_id, client_id, access_token_jti, access_expires_at,
                refresh_token_hash, refresh_token_fingerprint,
                refresh_expires_at, issued_at, revoked_at
            ) VALUES ('g', 'c', 'jti', 999999, 'hash', 'fp', 999999, 1, NULL)",
            [],
        )
        .expect("insert grant");

        // Re-run a few times.  Each call must short-circuit via the version
        // gate.  We assert the grant's revoked_at stays NULL (no spurious
        // UPDATE) and the signing-key table has exactly one row.
        for _ in 0..5 {
            migrate_oauth_tables(&conn).expect("re-migrate");
        }
        let revoked_at: Option<i64> = conn
            .query_row(
                "SELECT revoked_at FROM oauth_grants WHERE grant_id = 'g'",
                [],
                |row| row.get(0),
            )
            .expect("query grant");
        assert!(
            revoked_at.is_none(),
            "fingerprint-backfill UPDATE must not run on re-migration"
        );
        let key_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM oauth_signing_keys", [], |row| {
                row.get(0)
            })
            .expect("query signing keys");
        assert_eq!(key_count, 1, "signing key must not be regenerated");
    }

    // v0.31 candidate (Agent F4 / A-H2) — perf stub.  Real coverage would
    // assert "1000 re-runs complete in < 100 ms" against a grants table of
    // ~10k rows, demonstrating the version-gate short-circuit.  Marked
    // `#[ignore]` because per-CI perf gates require coordinated infra
    // (warm-up, statistical bounds) — leave to Codex audit reviewer to
    // decide whether to wire into a `cargo bench` harness instead.
    #[test]
    #[ignore]
    fn oauth_migration_idempotent_replay_under_100ms() {
        init_sqlite_vec();
        let conn = Connection::open_in_memory().expect("open memory db");
        migrate_oauth_tables(&conn).expect("initial migrate");
        let start = std::time::Instant::now();
        for _ in 0..1000 {
            migrate_oauth_tables(&conn).expect("replay migrate");
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed < std::time::Duration::from_millis(100),
            "1000 idempotent re-migrations should complete in < 100ms; took {elapsed:?}"
        );
    }

    // ====================================================================
    // v0.38 schema-versioning framework (user_version + forward migrations)
    // ====================================================================

    fn column_exists(conn: &Connection, table: &str, col: &str) -> bool {
        conn.query_row(
            &format!("SELECT COUNT(*) FROM pragma_table_info('{table}') WHERE name = ?1"),
            [col],
            |r| r.get::<_, i64>(0),
        )
        .unwrap_or(0)
            > 0
    }

    #[test]
    fn fresh_db_stamps_baseline_version() {
        init_sqlite_vec();
        let conn = Connection::open_in_memory().expect("open memory db");
        assert_eq!(schema_user_version(&conn).unwrap(), 0);
        init_schema(&conn, 4).expect("init schema");
        assert_eq!(
            schema_user_version(&conn).unwrap(),
            max_known_schema_version(MIGRATIONS),
            "a fresh DB must be stamped at the highest known version \
             (baseline bring-up + every forward migration applied)"
        );
    }

    #[test]
    fn legacy_db_brings_up_and_stamps_baseline() {
        let file = seed_pre_proxy_db();
        let conn = Connection::open(file.path()).expect("open");
        assert_eq!(
            schema_user_version(&conn).unwrap(),
            0,
            "a pre-framework DB starts at user_version 0"
        );
        init_schema(&conn, 4).expect("init schema on legacy db");
        assert_eq!(
            schema_user_version(&conn).unwrap(),
            max_known_schema_version(MIGRATIONS),
            "legacy bring-up must stamp baseline then apply every forward migration"
        );
        // The bring-up actually ran (migrate_source_check widened the source
        // CHECK), so a 'proxy'-sourced insert is now accepted.
        conn.execute_batch(
            "INSERT INTO memories (id, layer, topic, summary, content, importance, source,
                 strength, decay_lambda, created_at, updated_at, last_accessed)
             VALUES ('proxy-1', 'LTM', 't', 's', 'c', 'medium', 'proxy',
                 1.0, 0.0, '2026-04-20T00:00:00Z', '2026-04-20T00:00:00Z', '2026-04-20T00:00:00Z');",
        )
        .expect("proxy source must be allowed after bring-up");
    }

    /// THE discriminating test for the whole framework: a column dropped by a
    /// forward migration must NOT be silently re-added by `init_schema`'s
    /// legacy probe-then-ALTER bring-up on the next open. If this fails, the
    /// resurrection-safety invariant is broken and rename/type-change/drop
    /// migrations are unsafe.
    #[test]
    fn migration_drop_column_is_not_resurrected() {
        init_sqlite_vec();
        let conn = Connection::open_in_memory().expect("open memory db");

        fn drop_cluster_id(c: &Connection) -> ReinResult<()> {
            c.execute_batch("ALTER TABLE memories DROP COLUMN cluster_id")?;
            Ok(())
        }
        let migrations = &[Migration {
            version: BASELINE_SCHEMA_VERSION + 1,
            name: "drop_cluster_id",
            up: drop_cluster_id,
            idempotent: false, // DROP COLUMN — must be serialized.
        }];

        // First open: bring up to baseline (which creates `cluster_id`), then
        // the forward migration drops it.
        init_schema_with_migrations(&conn, 4, migrations).expect("first open");
        assert_eq!(
            schema_user_version(&conn).unwrap(),
            BASELINE_SCHEMA_VERSION + 1
        );
        assert!(
            !column_exists(&conn, "memories", "cluster_id"),
            "forward migration should have dropped cluster_id"
        );

        // Second open on the same DB: bring-up is gated out (>= baseline). The
        // dropped column must stay gone.
        init_schema_with_migrations(&conn, 4, migrations).expect("second open");
        assert!(
            !column_exists(&conn, "memories", "cluster_id"),
            "init_schema must NOT resurrect a column dropped by a forward migration"
        );
    }

    /// Snapshot every core (non-shadow) table and its column set for a fresh
    /// baseline DB. FTS5/vec0 auto-shadow tables are excluded — their internal
    /// DDL is sqlite-version dependent.
    #[cfg(test)]
    fn baseline_schema_snapshot(conn: &Connection) -> String {
        let mut names: Vec<String> = conn
            .prepare("SELECT name FROM sqlite_master WHERE type IN ('table') AND name NOT LIKE 'sqlite_%'")
            .unwrap()
            .query_map([], |r| r.get::<_, String>(0))
            .unwrap()
            .filter_map(|r| r.ok())
            .filter(|n| !n.contains("_fts") && !n.starts_with("vec_"))
            .collect();
        names.sort();
        let mut out = String::new();
        for t in &names {
            let mut cols: Vec<String> = conn
                .prepare(&format!("SELECT name FROM pragma_table_info('{t}')"))
                .unwrap()
                .query_map([], |r| r.get::<_, String>(0))
                .unwrap()
                .filter_map(|r| r.ok())
                .collect();
            cols.sort();
            out.push_str(&format!("{t}: {}\n", cols.join(", ")));
        }
        out
    }

    /// v1.0 SCHEMA FREEZE: the set of core tables and their columns is pinned.
    /// Any column/table add/drop/rename outside a forward `Migration` breaks
    /// this test — enforcing the 1.0 contract that all schema evolution goes
    /// through `MIGRATIONS`. If a change is intentional, author a forward
    /// migration and update `GOLDEN` to match.
    #[test]
    fn baseline_schema_is_frozen() {
        init_sqlite_vec();
        let conn = Connection::open_in_memory().expect("open memory db");
        init_schema(&conn, 4).expect("init schema");
        let snapshot = baseline_schema_snapshot(&conn);
        // Frozen at v1.0 (2026-05-30). Update ONLY alongside a forward Migration.
        const GOLDEN: &str = "\
cluster_centroids: centroid, cluster_id, cluster_version, dims
cold_archive: archived_at, content, memory_id
concept_links: created_at, id, relation, source_id, target_id, valid_from, valid_until, weight
concept_revisions: concept_id, confidence, created_at, definition, episode_id, id, labels, revision, source_memory_ids
concept_summary_instances: concept_id, refreshed_at, summary_id, summary_text
concepts: confidence, created_at, definition, id, labels, last_episode_id, living_summary, living_summary_id, living_summary_source_revision, living_summary_updated_at, memoir_id, name, revision, source_memory_ids, updated_at
consumer_offsets: consumer, last_event_id, updated_at
cron_claims: claim_token, claimed_at, event_type, stamp_hash, surface_id
dedup_decisions: canonical_id, confidence, conflict_detected, created_at, embedding_score, id, lexical_score, loser_id, merged_summary, novel_facts, operator, payload, reason, relation, reversible, winner_id
embed_cache: created_at, embedding, query_hash
episodes: concept_ids, created_at, decisions, id, important_paths, involved_agents, memory_ids, outcome, primary_topics, source_session_id, tags, temporal_keywords, title
feedback_events: concept_id, event_type, id, memory_id, payload, query, query_type, request_id, topic, ts
judge_call_ledger: id, status, ts
memoirs: created_at, description, id, name, updated_at
memories: access_count, archival_claim_token, archival_summary, archival_summary_at, archival_summary_version, cluster_id, concept_ids, content, created_at, decay_lambda, id, importance, in_progress_archival_summary_at, in_progress_resummerize_at, keywords, last_accessed, last_resummarized_at, last_too_large_at, layer, needs_archival_summary, needs_resummerize, needs_vec_dedup, related_ids, source, status, strength, summary, superseded_by, tier, topic, updated_at
memory_canonical_state: canonical_id, contradiction_score, dedup_confidence, last_merged_at, memory_id, merge_count, source_diversity, support_count
memory_evidence: canonical_id, content, created_at, id, imported_at, keywords, memory_id, source, source_topic, summary
memory_triple_meta: content_hash, extracted_at, extractor_version, memory_id
memory_triples: confidence, created_at, memory_id, object, predicate, subject
metadata: key, value
oauth_auth_codes: client_id, code, code_challenge, code_challenge_method, consumed_at, expires_at, issued_at, redirect_uri
oauth_clients: client_id, client_name, client_secret_hash, grant_types, last_used_at, redirect_uris, registered_at, revoked_at, token_endpoint_auth_method
oauth_grants: access_expires_at, access_token_jti, client_id, grant_id, issued_at, refresh_expires_at, refresh_token_fingerprint, refresh_token_hash, revoked_at
oauth_signing_keys: created_at, kid, rotated_at, secret_hex
pending_grayzone_jobs: candidate_id, created_at, id, result_id, sim
resummerize_runs: canonical_id, created_at, error, finished_at, id, input_canonical_chars, input_evidence_count, llm_backend, output_chars, output_hash, status, target_bytes, violations
schema_migrations: applied_at, name, version
session_artifacts: artifact_kind, created_at, ended_at, episode_id, id, is_subagent, schema_version, session_id, source_agent, source_label, started_at, summary, title, transcript_json, transcript_text, turn_count";
        assert_eq!(
            snapshot.trim(),
            GOLDEN.trim(),
            "v1.0 schema changed outside a Migration. If intentional, author a \
             forward Migration and update GOLDEN.\nACTUAL:\n{snapshot}"
        );
    }

    /// A migration that fails mid-way must roll back BOTH its DDL and the
    /// `user_version` bump — they share one transaction.
    #[test]
    fn failed_migration_rolls_back_ddl_and_version_together() {
        init_sqlite_vec();
        let conn = Connection::open_in_memory().expect("open memory db");

        fn half_then_fail(c: &Connection) -> ReinResult<()> {
            c.execute_batch("CREATE TABLE migration_partial (x INTEGER)")?;
            Err(ReinError::Config("simulated migration failure".into()))
        }
        let migrations = &[Migration {
            version: BASELINE_SCHEMA_VERSION + 1,
            name: "half_then_fail",
            up: half_then_fail,
            idempotent: false,
        }];

        let result = init_schema_with_migrations(&conn, 4, migrations);
        assert!(
            result.is_err(),
            "a failing migration must surface the error"
        );
        assert_eq!(
            schema_user_version(&conn).unwrap(),
            BASELINE_SCHEMA_VERSION,
            "user_version must NOT advance when the migration fails"
        );
        let leaked: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='migration_partial'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(
            leaked, 0,
            "DDL from a failed migration must roll back with the version bump"
        );
    }

    #[test]
    fn downgrade_guard_refuses_newer_database() {
        init_sqlite_vec();
        let conn = Connection::open_in_memory().expect("open memory db");
        init_schema(&conn, 4).expect("init schema");
        // Simulate a DB stamped by a future rein build.
        set_schema_user_version(&conn, max_known_schema_version(MIGRATIONS) + 5).unwrap();
        match init_schema(&conn, 4) {
            Err(ReinError::Config(msg)) => assert!(
                msg.contains("newer than this rein build"),
                "unexpected downgrade error message: {msg}"
            ),
            other => panic!("expected a Config downgrade error, got {other:?}"),
        }
    }

    #[test]
    fn forward_migration_applies_exactly_once() {
        init_sqlite_vec();
        let conn = Connection::open_in_memory().expect("open memory db");

        // No IF NOT EXISTS: a second application would error "table already
        // exists", so a clean second open proves the runner ran it once.
        fn create_marker(c: &Connection) -> ReinResult<()> {
            c.execute_batch("CREATE TABLE migration_marker (x INTEGER)")?;
            Ok(())
        }
        let migrations = &[Migration {
            version: BASELINE_SCHEMA_VERSION + 1,
            name: "create_marker",
            up: create_marker,
            idempotent: false,
        }];

        init_schema_with_migrations(&conn, 4, migrations).expect("first open");
        init_schema_with_migrations(&conn, 4, migrations)
            .expect("second open must skip the already-applied migration");
        assert_eq!(
            schema_user_version(&conn).unwrap(),
            BASELINE_SCHEMA_VERSION + 1
        );
        let marker: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='migration_marker'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(marker, 1, "migration table must exist exactly once");
    }

    #[test]
    fn add_column_idempotent_swallows_duplicate_but_propagates_real_errors() {
        let conn = Connection::open_in_memory().expect("open memory db");
        conn.execute_batch("CREATE TABLE t (a INTEGER)").unwrap();

        // New column → succeeds.
        add_column_idempotent(&conn, "ALTER TABLE t ADD COLUMN b INTEGER").unwrap();
        // Same column again → "duplicate column name" → benign, swallowed.
        add_column_idempotent(&conn, "ALTER TABLE t ADD COLUMN b INTEGER")
            .expect("duplicate-column race must be swallowed");
        // A genuine error (missing table) must fail loud, NOT be swallowed.
        assert!(
            add_column_idempotent(&conn, "ALTER TABLE no_such_table ADD COLUMN z INTEGER").is_err(),
            "non-duplicate ALTER errors must propagate"
        );
    }

    /// A migration at or below the DB's current `user_version` must never have
    /// its `up` re-run — covers the idempotent-skip path and guards the
    /// double-checked-lock recheck added for concurrent first-open (a peer that
    /// applied the migration between our pre-lock read and the write lock).
    #[test]
    fn already_applied_migration_is_not_reapplied() {
        init_sqlite_vec();
        let conn = Connection::open_in_memory().expect("open memory db");
        init_schema(&conn, 4).expect("init schema");

        // Stamp the DB as if version BASELINE+1 was already applied.
        set_schema_user_version(&conn, BASELINE_SCHEMA_VERSION + 1).unwrap();

        // `up` always errors; a correct skip never calls it.
        fn poison(_c: &Connection) -> ReinResult<()> {
            Err(ReinError::Config(
                "migration up must not run when already applied".into(),
            ))
        }
        let migrations = &[Migration {
            version: BASELINE_SCHEMA_VERSION + 1,
            name: "poison",
            up: poison,
            idempotent: false,
        }];

        apply_migrations(&conn, migrations)
            .expect("an already-applied migration must be skipped, not re-run");
        assert_eq!(
            schema_user_version(&conn).unwrap(),
            BASELINE_SCHEMA_VERSION + 1
        );
    }

    /// File-backed DB exercises the `MigrationLock` flock path (the in-memory
    /// tests skip it: `:memory:` takes no lock). First open stamps baseline and
    /// drops the lock cleanly; the sidecar exists; a second open is a clean
    /// no-op (re-acquire + release, version unchanged).
    #[test]
    fn file_db_init_schema_takes_and_releases_migration_lock() {
        init_sqlite_vec();
        let dir = tempfile::TempDir::new().expect("tempdir");
        let db = dir.path().join("rein-miglock-test.db");
        {
            let conn = Connection::open(&db).expect("open file db");
            init_schema(&conn, 4).expect("first init on a file db");
            assert_eq!(
                schema_user_version(&conn).unwrap(),
                max_known_schema_version(MIGRATIONS)
            );
        }
        let lock = std::path::PathBuf::from(format!("{}.migrate.lock", db.display()));
        assert!(
            lock.exists(),
            "migration lock sidecar should exist next to a file-backed DB"
        );
        // Second open: the lock is re-acquired and released, bring-up is skipped
        // (version already at baseline), and no error surfaces.
        let conn2 = Connection::open(&db).expect("reopen file db");
        init_schema(&conn2, 4).expect("second init must be a clean no-op");
        assert_eq!(
            schema_user_version(&conn2).unwrap(),
            max_known_schema_version(MIGRATIONS)
        );
    }

    /// Drives the REAL `MIGRATIONS` const (not a synthetic slice): the first
    /// forward migration (v2) must create `memory_triples` and stamp
    /// user_version to the max known version. Fails loudly if v2 is unwired.
    #[test]
    fn migration_creates_memory_triples_and_stamps_v2() {
        init_sqlite_vec();
        let conn = Connection::open_in_memory().expect("open memory db");
        init_schema(&conn, 4).expect("init schema with real MIGRATIONS");
        assert!(
            max_known_schema_version(MIGRATIONS) >= 2,
            "the v2 memory_triples migration must be wired into MIGRATIONS"
        );
        assert_eq!(
            schema_user_version(&conn).unwrap(),
            max_known_schema_version(MIGRATIONS),
            "init must stamp the max known version after applying forward migrations"
        );
        let table: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='memory_triples'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(table, 1, "memory_triples table must exist after migration");
        let idx: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name='idx_memory_triples_subject'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(idx, 1, "the subject index must exist after migration");
    }

    /// Drives the REAL `MIGRATIONS` const: the v3 forward migration must
    /// create `memory_triple_meta` (the #A5 reuse-reader stamp table) and the
    /// stamped version must reach at least 3. Also exercises the v2→v3
    /// upgrade path: a DB stamped at v2 (one release behind) must come out of
    /// `init_schema` with the meta table and the max known version.
    #[test]
    fn migration_v3_creates_triple_meta_and_upgrades_v2_db() {
        init_sqlite_vec();
        assert!(
            max_known_schema_version(MIGRATIONS) >= 3,
            "the v3 memory_triple_meta migration must be wired into MIGRATIONS"
        );

        // Fresh DB: both forward migrations apply in order.
        let conn = Connection::open_in_memory().expect("open memory db");
        init_schema(&conn, 4).expect("init schema with real MIGRATIONS");
        let table: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='memory_triple_meta'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(table, 1, "memory_triple_meta must exist after migration");

        // v2-era DB (v1.0/v1.1 release): wind the stamp back to 2 and drop the
        // meta table to simulate, then re-open — only v3 should apply.
        conn.execute_batch("DROP TABLE memory_triple_meta").unwrap();
        set_schema_user_version(&conn, 2).unwrap();
        init_schema(&conn, 4).expect("re-init on a v2-stamped DB");
        assert_eq!(
            schema_user_version(&conn).unwrap(),
            max_known_schema_version(MIGRATIONS),
            "v2→v3 upgrade must stamp the max known version"
        );
        let table: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='memory_triple_meta'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(table, 1, "v2→v3 upgrade must create memory_triple_meta");
    }
}
