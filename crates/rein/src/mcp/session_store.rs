//! Persistent `SessionStore` for the Streamable HTTP transport
//! (GitHub issue #4).
//!
//! rmcp expires an idle session after `SessionConfig::DEFAULT_KEEP_ALIVE`
//! (300 s) and answers later requests that carry the old `Mcp-Session-Id`
//! with `404 Session not found`. Several MCP clients never re-initialize
//! on that 404 (typescript-sdk #1708), so every tool call fails until the
//! client process restarts. Before answering 404, rmcp consults the
//! configured `SessionStore`: a hit re-creates the session under its
//! original id by replaying the stored `initialize`, and the request is
//! served normally.
//!
//! This store keeps that state in memory and mirrors it to a sidecar JSON
//! file next to the database, so restores also work across `rein serve`
//! restarts.
//!
//! Invariants:
//! - `load` never touches I/O and never returns `Err`. rmcp turns an `Err`
//!   from `load` into a 500 for the client.
//! - `order.len() == entries.len()`; `order` is oldest-first and drives
//!   eviction.
//! - Retained bytes are bounded: an entry larger than `MAX_SESSION_BYTES`
//!   is not retained at all (the session still works, it just cannot be
//!   restored), and the oldest entries are evicted until the total stays
//!   under `MAX_TOTAL_BYTES` and `MAX_SESSIONS`. Each entry is serialized
//!   once, at insert time, so a snapshot is a bounded concatenation under
//!   the lock, never a re-serialization of every session.
//! - Persistence failures are logged and never change in-memory state.
//! - `store` / `delete` return only after their own file write finished
//!   (or failed), so "returned" means "on disk".
//!
//! Security: restore only fires for ids already in this store, and ids are
//! server-generated UUIDs written after a successful `initialize`. rein's
//! auth policy runs on every request before rmcp sees it, so restore never
//! bypasses auth. The sidecar holds session ids plus the client-reported
//! name / version / capabilities — no tokens, no memory content — and is
//! written with mode 0600.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

use rmcp::transport::streamable_http_server::session::{
    SessionState, SessionStore, SessionStoreError,
};

/// Upper bound on remembered sessions. Engineering safety valve in the
/// same spirit as the 1 MiB `/mcp` body cap: a client that never sends
/// `DELETE` leaks one entry per restart, and 1024 entries is years of
/// daily restarts for a handful of clients. The oldest entry is dropped
/// first; a dropped session gets one 404 and re-initializes per spec.
pub const MAX_SESSIONS: usize = 1024;

/// Largest serialized entry that is retained. A real `initialize` carries
/// a client name, version and a capabilities object — a few hundred
/// bytes. Anything near this limit is a client stuffing `_meta`, and such
/// a session is simply not restorable (it keeps working until it expires).
pub const MAX_SESSION_BYTES: usize = 16 * 1024;

/// Total serialized bytes kept in memory and on disk. Bounds the snapshot
/// that is concatenated under the lock and fsynced on every `initialize`
/// and `DELETE`, independent of `MAX_SESSIONS`.
pub const MAX_TOTAL_BYTES: usize = 4 * 1024 * 1024;

/// Sidecar file format version. Bump when `StoredSession` changes shape.
pub const FILE_SCHEMA: u32 = 1;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct StoredSession {
    id: String,
    created_at_unix: u64,
    state: SessionState,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct SidecarFile {
    schema: u32,
    sessions: Vec<StoredSession>,
}

/// One retained session: the state rmcp asks for on restore, plus its
/// JSON serialized exactly once so snapshots never re-encode it.
#[derive(Debug)]
struct Entry {
    state: SessionState,
    json: String,
}

impl Entry {
    /// `None` when the serialized form exceeds `MAX_SESSION_BYTES`.
    fn new(session: &StoredSession) -> Option<Self> {
        let json = serde_json::to_string(session).ok()?;
        if json.len() > MAX_SESSION_BYTES {
            return None;
        }
        Some(Self {
            state: session.state.clone(),
            json,
        })
    }
}

#[derive(Debug, Default)]
struct Inner {
    /// Oldest first. Always the same length as `entries`.
    order: VecDeque<String>,
    entries: HashMap<String, Entry>,
    /// Sum of `entries[*].json.len()`.
    total_bytes: usize,
    /// Incremented on every mutation. Snapshots carry it so an older
    /// snapshot can never overwrite a newer one on disk.
    generation: u64,
}

impl Inner {
    /// Insert or overwrite (overwrite keeps the position in `order`), then
    /// evict oldest-first until both caps hold. Returns `false` when the
    /// entry itself is over budget and was not retained.
    fn insert(&mut self, id: &str, session: &StoredSession) -> bool {
        let Some(entry) = Entry::new(session) else {
            // An oversized entry must not linger from an earlier,
            // smaller `initialize` under the same id.
            self.remove(id);
            return false;
        };
        if let Some(existing) = self.entries.get_mut(id) {
            self.total_bytes = self.total_bytes - existing.json.len() + entry.json.len();
            *existing = entry;
        } else {
            self.order.push_back(id.to_owned());
            self.total_bytes += entry.json.len();
            self.entries.insert(id.to_owned(), entry);
        }
        self.evict_to_budget();
        true
    }

    fn remove(&mut self, id: &str) -> bool {
        let Some(entry) = self.entries.remove(id) else {
            return false;
        };
        self.total_bytes -= entry.json.len();
        self.order.retain(|other| other != id);
        true
    }

    fn evict_to_budget(&mut self) {
        while self.order.len() > MAX_SESSIONS || self.total_bytes > MAX_TOTAL_BYTES {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            if let Some(entry) = self.entries.remove(&oldest) {
                self.total_bytes -= entry.json.len();
            }
        }
    }
}

/// rmcp `SessionStore` backed by memory plus an optional sidecar file.
pub struct PersistentSessionStore {
    path: Option<PathBuf>,
    inner: Mutex<Inner>,
    /// Highest generation written to disk. Held across the whole file
    /// write so concurrent writers serialize and ordering is enforced.
    written: Arc<Mutex<u64>>,
}

impl PersistentSessionStore {
    /// Memory-only store (tests, and the control group that proves the
    /// expiry path without a store).
    pub fn in_memory() -> Self {
        Self {
            path: None,
            inner: Mutex::new(Inner::default()),
            written: Arc::new(Mutex::new(0)),
        }
    }

    /// Load the sidecar at `path` (missing → empty; unreadable → warn +
    /// empty, never a startup failure) and persist every later change
    /// back to it. The same byte and count budgets apply to what is
    /// loaded, so a hand-edited or pre-budget file cannot bypass them.
    pub fn open(path: PathBuf) -> Self {
        let mut inner = Inner::default();
        match read_sidecar(&path) {
            Ok(Some(file)) => {
                for session in &file.sessions {
                    inner.insert(&session.id, session);
                }
            }
            Ok(None) => {}
            Err(err) => tracing::warn!(
                path = %path.display(),
                error = %err,
                "mcp session sidecar unreadable; starting with no restorable sessions (rewritten on the next initialize)"
            ),
        }
        Self {
            path: Some(path),
            inner: Mutex::new(inner),
            written: Arc::new(Mutex::new(0)),
        }
    }

    pub fn len(&self) -> usize {
        self.lock().entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn contains(&self, session_id: &str) -> bool {
        self.lock().entries.contains_key(session_id)
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Write `snapshot` to the sidecar if one is configured. Never fails
    /// the caller: in-memory state is authoritative for this process.
    async fn persist(&self, snapshot: Snapshot) {
        let Some(path) = self.path.clone() else {
            return;
        };
        let written = Arc::clone(&self.written);
        let path_display = path.display().to_string();
        match tokio::task::spawn_blocking(move || persist_blocking(&path, &written, snapshot)).await
        {
            Ok(Ok(_)) => {}
            Ok(Err(err)) => tracing::warn!(
                path = %path_display,
                error = %err,
                "mcp session sidecar write failed; sessions stay restorable in this process only"
            ),
            Err(join) => tracing::warn!(
                path = %path_display,
                error = %join,
                "mcp session sidecar writer task failed"
            ),
        }
    }
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Serialized view of the store at one generation.
struct Snapshot {
    generation: u64,
    json: String,
}

impl Snapshot {
    /// Concatenates the pre-serialized entries in `order`; bounded by
    /// `MAX_TOTAL_BYTES` plus framing, so this is cheap under the lock.
    fn of(inner: &Inner) -> Self {
        let mut json = String::with_capacity(inner.total_bytes + 64);
        json.push_str("{\"schema\":");
        json.push_str(&FILE_SCHEMA.to_string());
        json.push_str(",\"sessions\":[");
        let mut first = true;
        for id in &inner.order {
            let Some(entry) = inner.entries.get(id) else {
                continue;
            };
            if !first {
                json.push(',');
            }
            first = false;
            json.push_str(&entry.json);
        }
        json.push_str("]}");
        Self {
            generation: inner.generation,
            json,
        }
    }
}

/// Write `snapshot` unless a newer generation is already on disk.
/// Returns `Ok(true)` when written, `Ok(false)` when skipped as stale.
/// The `written` lock is held for the whole write so concurrent writers
/// serialize.
fn persist_blocking(
    path: &Path,
    written: &Mutex<u64>,
    snapshot: Snapshot,
) -> std::io::Result<bool> {
    let mut written = written
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if snapshot.generation <= *written {
        return Ok(false);
    }
    write_atomic(path, snapshot.json.as_bytes())?;
    *written = snapshot.generation;
    Ok(true)
}

/// tmp + fsync + rename, mode 0600 on unix. A stale `.tmp` from a crashed
/// run is removed first so `create_new` always applies the mode.
fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("json.tmp");
    let _ = std::fs::remove_file(&tmp);
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(&tmp)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);
    std::fs::rename(&tmp, path)
}

/// `Ok(None)` when the file does not exist; `Err` for any other read or
/// parse failure (the message is content-free: it names the serde or I/O
/// error, never file contents).
fn read_sidecar(path: &Path) -> Result<Option<SidecarFile>, String> {
    let raw = match std::fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err.to_string()),
    };
    let file: SidecarFile = serde_json::from_str(&raw).map_err(|err| err.to_string())?;
    if file.schema != FILE_SCHEMA {
        return Err(format!(
            "unsupported sidecar schema {} (expected {FILE_SCHEMA})",
            file.schema
        ));
    }
    Ok(Some(file))
}

#[async_trait::async_trait]
impl SessionStore for PersistentSessionStore {
    async fn load(&self, session_id: &str) -> Result<Option<SessionState>, SessionStoreError> {
        let found = self
            .lock()
            .entries
            .get(session_id)
            .map(|entry| entry.state.clone());
        if let Some(state) = &found {
            // rmcp does not call back into rein when it restores; this is
            // the only point where "a restore is about to happen" is
            // observable. Session ids are not sensitive.
            tracing::info!(
                session_id,
                client = %state.initialize_params.client_info.name,
                client_version = %state.initialize_params.client_info.version,
                "mcp session restore: expired or pre-restart session found in store"
            );
        }
        Ok(found)
    }

    async fn store(&self, session_id: &str, state: &SessionState) -> Result<(), SessionStoreError> {
        // Serialize outside the lock; `Inner::insert` only copies.
        let session = StoredSession {
            id: session_id.to_owned(),
            created_at_unix: now_unix(),
            state: state.clone(),
        };
        let snapshot = {
            let mut inner = self.lock();
            let retained = inner.insert(session_id, &session);
            if !retained {
                tracing::info!(
                    session_id,
                    limit = MAX_SESSION_BYTES,
                    "mcp session not retained for restore: initialize params exceed the per-session byte budget"
                );
            }
            inner.generation += 1;
            Snapshot::of(&inner)
        };
        self.persist(snapshot).await;
        Ok(())
    }

    async fn delete(&self, session_id: &str) -> Result<(), SessionStoreError> {
        let snapshot = {
            let mut inner = self.lock();
            if !inner.remove(session_id) {
                return Ok(());
            }
            inner.generation += 1;
            Snapshot::of(&inner)
        };
        self.persist(snapshot).await;
        Ok(())
    }
}

/// What `rein doctor` reports about the sidecar without loading it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SidecarStatus {
    Missing,
    Readable { sessions: usize },
    Corrupt(String),
}

pub fn inspect_sidecar(path: &Path) -> SidecarStatus {
    match read_sidecar(path) {
        Ok(None) => SidecarStatus::Missing,
        Ok(Some(file)) => SidecarStatus::Readable {
            sessions: file.sessions.len(),
        },
        Err(err) => SidecarStatus::Corrupt(err),
    }
}

/// `<db dir>/<db file name>.mcp-sessions-<bind>-<port>-<hash>.json`.
/// Follows `REIN_DB` / `[database].path` like every other rein state file.
///
/// Identity is carried by `<hash>`: the first 16 hex chars of
/// SHA-256 over the exact database file name and bind host, so two
/// databases that differ only in characters the readable prefix cannot
/// show (`work notes.db` vs `work_notes.db`, or `memories.db` vs
/// `memories.sqlite`), two binds on one port, or `--sse` and `--gui` on
/// one database never share a sidecar (nor a `.tmp`). The sanitized
/// prefix exists only so an operator can tell the files apart by eye.
pub fn mcp_session_store_path(config: &crate::config::ReinConfig) -> PathBuf {
    use sha2::{Digest, Sha256};

    let db = config.resolve_db_path();
    let dir = db
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let db_name = db
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("memories.db");
    let bind = config.server.sse_bind.trim();
    let port = config.server.sse_port;

    let mut hasher = Sha256::new();
    hasher.update(db_name.as_bytes());
    hasher.update([0u8]);
    hasher.update(bind.as_bytes());
    let digest = hasher.finalize();
    let hash: String = digest.iter().take(8).map(|b| format!("{b:02x}")).collect();

    dir.join(format!(
        "{}.mcp-sessions-{}-{port}-{hash}.json",
        filename_safe(db_name),
        filename_safe(bind)
    ))
}

/// Keep `[A-Za-z0-9.-]`, replace everything else (IPv6 colons, spaces,
/// path separators) with `_` so the readable part of the sidecar name is
/// a safe single path segment on every platform. Lossy by design; the
/// hash next to it carries identity.
fn filename_safe(raw: &str) -> String {
    raw.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `SessionState` through serde: rmcp's request/implementation
    /// structs are `#[non_exhaustive]`, JSON is the stable way to construct
    /// them from outside the crate.
    fn state(client_name: &str) -> SessionState {
        serde_json::from_value(serde_json::json!({
            "initialize_params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": client_name, "version": "0"}
            }
        }))
        .expect("valid SessionState JSON")
    }

    /// A state whose serialized entry is roughly `bytes` long.
    fn padded_state(bytes: usize) -> SessionState {
        state(&"x".repeat(bytes))
    }

    fn client_name(state: &SessionState) -> String {
        state.initialize_params.client_info.name.clone()
    }

    fn stored(id: &str, state: SessionState) -> StoredSession {
        StoredSession {
            id: id.to_owned(),
            created_at_unix: 1,
            state,
        }
    }

    #[tokio::test]
    async fn load_store_delete_roundtrip() {
        let store = PersistentSessionStore::in_memory();
        assert!(store.load("s1").await.unwrap().is_none());
        assert!(store.is_empty());

        store.store("s1", &state("alpha")).await.unwrap();
        let loaded = store
            .load("s1")
            .await
            .unwrap()
            .expect("stored session loads");
        assert_eq!(client_name(&loaded), "alpha");
        assert_eq!(store.len(), 1);
        assert!(store.contains("s1"));

        store.delete("s1").await.unwrap();
        assert!(store.load("s1").await.unwrap().is_none());
        assert!(!store.contains("s1"));
        assert!(store.is_empty());
        assert_eq!(store.lock().total_bytes, 0);
    }

    #[tokio::test]
    async fn delete_of_unknown_id_is_a_noop() {
        let store = PersistentSessionStore::in_memory();
        store.delete("never-stored").await.unwrap();
        assert!(store.is_empty());
    }

    #[tokio::test]
    async fn storing_same_id_overwrites_without_duplicating() {
        let store = PersistentSessionStore::in_memory();
        store.store("s1", &state("first")).await.unwrap();
        store.store("s2", &state("second")).await.unwrap();
        store.store("s1", &state("first-v2")).await.unwrap();

        assert_eq!(store.len(), 2);
        assert_eq!(
            client_name(&store.load("s1").await.unwrap().unwrap()),
            "first-v2"
        );
        let inner = store.lock();
        assert_eq!(inner.order.len(), inner.entries.len());
        assert_eq!(
            inner.order.front().map(String::as_str),
            Some("s1"),
            "overwrite keeps position"
        );
        let expected: usize = inner.entries.values().map(|e| e.json.len()).sum();
        assert_eq!(
            inner.total_bytes, expected,
            "byte accounting after overwrite"
        );
    }

    #[tokio::test]
    async fn evicts_oldest_beyond_cap() {
        let store = PersistentSessionStore::in_memory();
        for i in 0..=MAX_SESSIONS {
            store.store(&format!("s{i}"), &state("c")).await.unwrap();
        }
        assert_eq!(store.len(), MAX_SESSIONS);
        assert!(!store.contains("s0"), "oldest entry evicted");
        assert!(store.contains("s1"));
        assert!(store.contains(&format!("s{MAX_SESSIONS}")));
        let inner = store.lock();
        assert_eq!(inner.order.len(), inner.entries.len());
    }

    #[tokio::test]
    async fn oversized_session_is_not_retained() {
        let store = PersistentSessionStore::in_memory();
        store.store("small", &state("ok")).await.unwrap();
        store
            .store("huge", &padded_state(MAX_SESSION_BYTES + 1))
            .await
            .unwrap();
        assert!(!store.contains("huge"), "over-budget entry is dropped");
        assert!(store.load("huge").await.unwrap().is_none());
        assert!(store.contains("small"));
        assert_eq!(store.len(), 1);

        // A later oversized initialize under an id that was small before
        // must evict the stale small entry rather than keep restoring it.
        store
            .store("small", &padded_state(MAX_SESSION_BYTES + 1))
            .await
            .unwrap();
        assert!(!store.contains("small"));
        assert_eq!(store.lock().total_bytes, 0);
    }

    #[tokio::test]
    async fn total_byte_budget_evicts_oldest() {
        let store = PersistentSessionStore::in_memory();
        let per_entry = MAX_SESSION_BYTES / 2;
        let mut n = 0usize;
        loop {
            store
                .store(&format!("s{n}"), &padded_state(per_entry))
                .await
                .unwrap();
            n += 1;
            if n * per_entry > MAX_TOTAL_BYTES + 4 * per_entry {
                break;
            }
        }
        let inner = store.lock();
        assert!(
            inner.total_bytes <= MAX_TOTAL_BYTES,
            "{}",
            inner.total_bytes
        );
        assert!(inner.total_bytes > MAX_TOTAL_BYTES / 2, "budget is used");
        assert!(!inner.entries.contains_key("s0"), "oldest evicted first");
        assert!(inner.entries.contains_key(&format!("s{}", n - 1)));
        assert_eq!(inner.order.len(), inner.entries.len());
        let expected: usize = inner.entries.values().map(|e| e.json.len()).sum();
        assert_eq!(inner.total_bytes, expected);
    }

    #[tokio::test]
    async fn persists_and_reopens_from_sidecar() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mcp-sessions-8680.json");
        {
            let store = PersistentSessionStore::open(path.clone());
            assert!(store.is_empty(), "fresh path starts empty");
            store.store("s1", &state("alpha")).await.unwrap();
            store.store("s2", &state("beta")).await.unwrap();
            assert!(path.exists(), "store returns only after the write landed");
        }
        let reopened = PersistentSessionStore::open(path.clone());
        assert_eq!(reopened.len(), 2);
        assert_eq!(
            client_name(&reopened.load("s2").await.unwrap().unwrap()),
            "beta"
        );
        assert_eq!(reopened.path(), Some(path.as_path()));

        // The hand-assembled snapshot is valid JSON of the declared schema.
        let parsed: SidecarFile = serde_json::from_str(&std::fs::read_to_string(&path).unwrap())
            .expect("snapshot parses as SidecarFile");
        assert_eq!(parsed.schema, FILE_SCHEMA);
        assert_eq!(parsed.sessions.len(), 2);
        assert_eq!(parsed.sessions[0].id, "s1");

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600, "sidecar is private to the user");
        }
        assert!(
            !path.with_extension("json.tmp").exists(),
            "tmp file renamed away"
        );
    }

    #[tokio::test]
    async fn empty_store_persists_as_empty_list() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mcp-sessions-8680.json");
        let store = PersistentSessionStore::open(path.clone());
        store.store("s1", &state("alpha")).await.unwrap();
        store.delete("s1").await.unwrap();
        let parsed: SidecarFile = serde_json::from_str(&std::fs::read_to_string(&path).unwrap())
            .expect("empty snapshot parses");
        assert!(parsed.sessions.is_empty());
    }

    #[tokio::test]
    async fn delete_is_persisted() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mcp-sessions-8680.json");
        let store = PersistentSessionStore::open(path.clone());
        store.store("s1", &state("alpha")).await.unwrap();
        store.delete("s1").await.unwrap();
        let reopened = PersistentSessionStore::open(path);
        assert!(reopened.is_empty());
    }

    #[tokio::test]
    async fn corrupt_sidecar_opens_empty_and_is_rewritten() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mcp-sessions-8680.json");
        std::fs::write(&path, "{not json").unwrap();
        let store = PersistentSessionStore::open(path.clone());
        assert!(store.is_empty());
        store.store("s1", &state("alpha")).await.unwrap();
        let reopened = PersistentSessionStore::open(path);
        assert_eq!(reopened.len(), 1);
    }

    #[tokio::test]
    async fn unknown_schema_opens_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mcp-sessions-8680.json");
        std::fs::write(&path, r#"{"schema":99,"sessions":[]}"#).unwrap();
        assert!(PersistentSessionStore::open(path).is_empty());
    }

    #[tokio::test]
    async fn reopen_applies_count_and_byte_budgets() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mcp-sessions-8680.json");
        let mut sessions: Vec<StoredSession> = (0..MAX_SESSIONS + 5)
            .map(|i| stored(&format!("s{i}"), state("c")))
            .collect();
        sessions.push(stored("huge", padded_state(MAX_SESSION_BYTES + 1)));
        let file = SidecarFile {
            schema: FILE_SCHEMA,
            sessions,
        };
        std::fs::write(&path, serde_json::to_string(&file).unwrap()).unwrap();
        let store = PersistentSessionStore::open(path);
        assert_eq!(store.len(), MAX_SESSIONS);
        assert!(!store.contains("s0"));
        assert!(!store.contains("s4"));
        assert!(store.contains("s5"));
        assert!(store.contains(&format!("s{}", MAX_SESSIONS + 4)));
        assert!(!store.contains("huge"), "oversized file entry is skipped");
    }

    #[test]
    fn older_snapshot_never_overwrites_newer() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mcp-sessions-8680.json");
        let written = Mutex::new(0u64);
        let newer = Snapshot {
            generation: 2,
            json: "NEWER".to_string(),
        };
        let older = Snapshot {
            generation: 1,
            json: "OLDER".to_string(),
        };
        assert!(persist_blocking(&path, &written, newer).unwrap());
        assert!(!persist_blocking(&path, &written, older).unwrap());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "NEWER");
        assert_eq!(*written.lock().unwrap(), 2);
    }

    #[test]
    fn inspect_reports_missing_readable_and_corrupt() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mcp-sessions-8680.json");
        assert_eq!(inspect_sidecar(&path), SidecarStatus::Missing);

        let file = SidecarFile {
            schema: FILE_SCHEMA,
            sessions: vec![stored("s1", state("c"))],
        };
        std::fs::write(&path, serde_json::to_string(&file).unwrap()).unwrap();
        assert_eq!(
            inspect_sidecar(&path),
            SidecarStatus::Readable { sessions: 1 }
        );

        std::fs::write(&path, "{not json").unwrap();
        assert!(matches!(inspect_sidecar(&path), SidecarStatus::Corrupt(_)));
    }

    #[test]
    fn store_path_is_scoped_by_database_bind_and_port() {
        fn path_for(db: &str, bind: &str, port: u16) -> PathBuf {
            let mut config = crate::config::ReinConfig::default();
            config.database.path = format!("/tmp/rein-plan-test/{db}");
            config.server.sse_bind = bind.to_string();
            config.server.sse_port = port;
            mcp_session_store_path(&config)
        }

        let base = path_for("memories.db", "127.0.0.1", 9999);
        let name = base.file_name().unwrap().to_str().unwrap().to_owned();
        assert_eq!(base.parent().unwrap(), Path::new("/tmp/rein-plan-test"));
        assert!(
            name.starts_with("memories.db.mcp-sessions-127.0.0.1-9999-"),
            "{name}"
        );
        assert!(name.ends_with(".json"), "{name}");
        let hash = &name["memories.db.mcp-sessions-127.0.0.1-9999-".len()..name.len() - 5];
        assert_eq!(hash.len(), 16, "{name}");
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()), "{name}");
        assert_eq!(base, path_for("memories.db", "127.0.0.1", 9999), "stable");

        // Every component of identity changes the path.
        assert_ne!(base, path_for("memories.sqlite", "127.0.0.1", 9999));
        assert_ne!(base, path_for("memories.db", "::1", 9999));
        assert_ne!(base, path_for("memories.db", "127.0.0.1", 9998));

        // Names that sanitize to the same readable prefix still differ.
        let spaced = path_for("work notes.db", "127.0.0.1", 9999);
        let underscored = path_for("work_notes.db", "127.0.0.1", 9999);
        assert!(spaced
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .starts_with("work_notes.db.mcp-sessions-"));
        assert_ne!(spaced, underscored);

        // IPv6 binds are sanitized in the readable part.
        let v6 = path_for("memories.db", "::1", 9999);
        assert!(v6
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .starts_with("memories.db.mcp-sessions-__1-9999-"));
    }
}
