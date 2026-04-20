//! Hand-rolled async SQLite connection pool.
//!
//! Why hand-rolled and not `deadpool-sqlite`: every version ≥ 0.10 pins
//! rusqlite ≥ 0.31 → 0.38, each linking `libsqlite3-sys` at a version that
//! conflicts with rein's current `rusqlite 0.34 → libsqlite3-sys 0.32`.
//! Upgrading rusqlite is a cross-cutting dep change orthogonal to the
//! v0.22 async-hardening scope; once rein graduates to rusqlite 0.38 we
//! can swap this module out for deadpool-sqlite without touching callers.
//! See `docs/superpowers/specs/2026-04-21-v0.22-async-hardening-design.md`.
//!
//! ## Semantics
//!
//! - Fixed-size pool: N connections opened at construction; no dynamic
//!   growth. Running out of permits means `get()` awaits a returning
//!   guard.
//! - Each connection is opened with `SQLITE_OPEN_FULL_MUTEX` (per-conn
//!   serialized, not global). WAL mode + NORMAL sync + 5s busy_timeout +
//!   FK ON are set once at open time.
//! - `sqlite-vec` loads automatically via the global `sqlite3_auto_extension`
//!   hook registered by `store::schema::init_sqlite_vec` — no per-conn
//!   setup needed.
//! - `PoolGuard::interact(|c| ...)` mirrors deadpool's ergonomic: the
//!   closure runs on a blocking thread via `tokio::task::spawn_blocking`,
//!   so sync rusqlite calls never stall the reactor.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use rusqlite::{Connection, OpenFlags};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::store::schema;
use crate::types::{ReinError, ReinResult};

/// Default main-pool size. Balances p50 recall latency (wants ≥ 3 conns
/// for the 3-channel fanout) against memory/fd footprint.
pub fn default_pool_size() -> usize {
    let par = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    // min(8, max(2, par / 2)). The upper cap reflects a pragmatic
    // per-process fd budget; real high-concurrency tuning goes via
    // REIN_POOL_SIZE.
    (par / 2).clamp(2, 8)
}

/// Default feedback-queue pool size. The queue DB is small and has only
/// one writer at a time (drain worker); 4 is plenty.
pub fn default_feedback_queue_pool_size() -> usize {
    4
}

struct PoolInner {
    db_path: PathBuf,
    size: usize,
    free: Mutex<Vec<Connection>>,
    permits: Arc<Semaphore>,
}

#[derive(Clone)]
pub struct ConnPool {
    inner: Arc<PoolInner>,
}

impl ConnPool {
    /// Pre-open `size` connections to `db_path`, apply pragmas, return the
    /// pool. Fails if any single `open_conn` fails (no partial pool).
    pub fn new(db_path: impl AsRef<Path>, size: usize) -> ReinResult<Self> {
        assert!(size >= 1, "pool size must be ≥ 1");
        let db_path = db_path.as_ref().to_owned();
        schema::init_sqlite_vec();
        let mut conns = Vec::with_capacity(size);
        for _ in 0..size {
            conns.push(open_conn(&db_path)?);
        }
        Ok(Self {
            inner: Arc::new(PoolInner {
                db_path,
                size,
                free: Mutex::new(conns),
                permits: Arc::new(Semaphore::new(size)),
            }),
        })
    }

    /// Acquire a connection from the pool. Awaits if all are checked out.
    ///
    /// Requires an active Tokio runtime (both for the semaphore wait and
    /// for `interact()` on the returned guard). Returns an error instead
    /// of panicking if none is entered.
    pub async fn get(&self) -> ReinResult<PoolGuard> {
        if tokio::runtime::Handle::try_current().is_err() {
            return Err(ReinError::Config(
                "ConnPool::get requires a Tokio runtime to be entered".into(),
            ));
        }
        let permit = self
            .inner
            .permits
            .clone()
            .acquire_owned()
            .await
            .map_err(|e| ReinError::Config(format!("pool semaphore closed: {e}")))?;
        let conn = self
            .inner
            .free
            .lock()
            .expect("pool free-list mutex poisoned")
            .pop()
            .expect("semaphore invariant: permit acquired implies free conn available");
        Ok(PoolGuard {
            pool: self.inner.clone(),
            conn: Some(conn),
            _permit: Some(permit),
        })
    }

    pub fn size(&self) -> usize {
        self.inner.size
    }

    pub fn db_path(&self) -> &Path {
        &self.inner.db_path
    }

    /// Snapshot observable metrics for `/api/health` and doctor output.
    /// Values are point-in-time, not synchronized.
    pub fn metrics(&self) -> PoolMetrics {
        let idle = self
            .inner
            .free
            .lock()
            .expect("pool free-list mutex poisoned")
            .len();
        let available_permits = self.inner.permits.available_permits();
        PoolMetrics {
            size: self.inner.size,
            idle,
            in_use: self.inner.size - idle,
            available_permits,
        }
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct PoolMetrics {
    pub size: usize,
    pub idle: usize,
    pub in_use: usize,
    pub available_permits: usize,
}

/// A borrowed connection. Drops back into the pool on `Drop` (fallback if
/// `interact()` was not used). Prefer `interact()` for actual work — it
/// handles the spawn_blocking hop and gives a clean async surface.
pub struct PoolGuard {
    pool: Arc<PoolInner>,
    conn: Option<Connection>,
    _permit: Option<OwnedSemaphorePermit>,
}

impl PoolGuard {
    /// Run a synchronous SQLite operation on a blocking thread. The
    /// connection is automatically returned to the pool when the task
    /// finishes (success, error, or panic).
    pub async fn interact<F, R>(mut self, f: F) -> ReinResult<R>
    where
        F: FnOnce(&mut Connection) -> ReinResult<R> + Send + 'static,
        R: Send + 'static,
    {
        let mut conn = self
            .conn
            .take()
            .expect("conn present (interact consumes guard)");
        let pool = self.pool.clone();
        let permit = self
            ._permit
            .take()
            .expect("permit present (interact consumes guard)");

        let join = tokio::task::spawn_blocking(move || {
            let result = f(&mut conn);
            (result, conn)
        })
        .await;

        match join {
            Ok((result, conn)) => {
                return_conn(&pool, conn);
                drop(permit); // release slot for next waiter
                result
            }
            Err(err) => {
                // Panic in the closure: spawn_blocking propagated a
                // JoinError, and `conn` is gone (dropped inside the
                // blocking thread's cleanup). Without replacement, the
                // next `get()` would acquire this permit's successor and
                // hit `expect("semaphore invariant")` — there's no conn
                // in the free list. Self-heal by opening a fresh conn to
                // restore pool size. If that itself fails (disk error,
                // permission, etc.), permanently shrink the pool by
                // forgetting the permit; the health endpoint surfaces
                // the shrink via `PoolMetrics::size` vs observed idle.
                match open_conn(&pool.db_path) {
                    Ok(fresh) => {
                        return_conn(&pool, fresh);
                        drop(permit); // release slot; a conn is waiting
                    }
                    Err(open_err) => {
                        tracing::warn!(
                            original_panic = %err,
                            open_err = %open_err,
                            "pool interact panicked and replacement conn open failed; \
                             shrinking pool by one"
                        );
                        // `permit` is Option<OwnedSemaphorePermit> from
                        // self._permit.take(); the take() upstream
                        // already unwrapped it at binding time via
                        // `.expect("permit present")`, so here permit
                        // is the owned permit — forget() shrinks the
                        // semaphore by one.
                        permit.forget();
                    }
                }
                Err(ReinError::Config(format!(
                    "pool interact task panicked: {err}"
                )))
            }
        }
    }

    /// Escape hatch for callers that need to hold the conn across multiple
    /// async awaits. Prefer `interact` for simple cases.
    pub fn conn(&self) -> &Connection {
        self.conn.as_ref().expect("conn present")
    }

    pub fn conn_mut(&mut self) -> &mut Connection {
        self.conn.as_mut().expect("conn present")
    }

    /// Detach the connection from the guard while **keeping the permit
    /// alive** (handed over to the returned `DetachedGuard`). This lets
    /// callers wrap the raw `Connection` in a `SqliteStore` for sync use
    /// without breaking the semaphore invariant
    /// `permit acquired ↔ free conn exists`.
    ///
    /// Typical use at an adapter layer:
    ///
    /// ```ignore
    /// let guard = pool.get().await?;
    /// let (conn, detached) = guard.detach();
    /// let store = SqliteStore::from_conn(conn, path, dims);
    /// // ... synchronous SqliteStore work (no await) ...
    /// detached.put_back(store.into_conn());
    /// // conn returned to pool + permit released
    /// ```
    pub fn detach(mut self) -> (Connection, DetachedGuard) {
        let conn = self
            .conn
            .take()
            .expect("conn present (detach consumes guard)");
        let permit = self
            ._permit
            .take()
            .expect("permit present (detach consumes guard)");
        let pool = self.pool.clone();
        // Dropping `self` now is a no-op: both `conn` and `_permit` are None.
        (
            conn,
            DetachedGuard {
                inner: pool,
                permit: Some(permit),
            },
        )
    }
}

/// Companion token returned by `PoolGuard::detach`. Holds the semaphore
/// permit so the pool's invariants stay sound while the caller has
/// manual ownership of the raw `Connection`. Call `put_back()` exactly
/// once with the `Connection` when done — or `Drop` fires and the slot
/// is **permanently forgotten** (pool shrinks by one), because the conn
/// was lost and releasing the permit normally would let the next
/// `get()` find the free list empty.
pub struct DetachedGuard {
    inner: Arc<PoolInner>,
    permit: Option<OwnedSemaphorePermit>,
}

impl DetachedGuard {
    /// Return the connection to the pool and release the permit, waking
    /// any pending `pool.get()` awaiter.
    pub fn put_back(mut self, conn: Connection) {
        return_conn(&self.inner, conn);
        // Drop the permit explicitly so it releases the slot normally
        // (new waiters can grab it because a conn is back in the free
        // list). We take it out of `self` so the subsequent `self` drop
        // in the `Drop` impl sees `None` and does not go down the
        // "forget to shrink" branch.
        drop(self.permit.take());
    }
}

impl Drop for DetachedGuard {
    fn drop(&mut self) {
        // Only fires if `put_back` was NOT called — either the caller
        // leaked the conn (bug) or the surrounding code panicked between
        // `detach()` and `put_back()` (e.g. `try_vector_search` expect()
        // failure in the pooled recall path).
        //
        // Self-heal symmetric with `PoolGuard::interact`'s panic arm: try
        // to open a fresh conn against the same DB and return it to the
        // pool, keeping `size` invariant. Only fall through to
        // `permit.forget()` (permanent shrink) if the replacement open
        // also fails — otherwise repeated detach-path panics would
        // monotonically shrink the pool until `get()` blocks forever
        // with no user-visible error (Codex F3 HIGH).
        if let Some(permit) = self.permit.take() {
            match open_conn(&self.inner.db_path) {
                Ok(fresh) => {
                    tracing::warn!(
                        pool_db = %self.inner.db_path.display(),
                        "DetachedGuard dropped without put_back; opened replacement conn"
                    );
                    return_conn(&self.inner, fresh);
                    drop(permit); // release slot normally — conn available
                }
                Err(open_err) => {
                    tracing::error!(
                        pool_db = %self.inner.db_path.display(),
                        err = %open_err,
                        "DetachedGuard dropped without put_back AND replacement conn \
                         open failed; pool size shrinks by one"
                    );
                    permit.forget();
                }
            }
        }
    }
}

impl Drop for PoolGuard {
    fn drop(&mut self) {
        if let Some(conn) = self.conn.take() {
            return_conn(&self.pool, conn);
        }
        // _permit drops here, releasing the slot.
    }
}

fn return_conn(pool: &Arc<PoolInner>, conn: Connection) {
    pool.free
        .lock()
        .expect("pool free-list mutex poisoned")
        .push(conn);
}

/// Open a single connection with rein's standard pragmas. Exposed for
/// callers that need a throwaway conn outside the pool (e.g. migrations).
pub fn open_conn(db_path: &Path) -> ReinResult<Connection> {
    let conn = Connection::open_with_flags(
        db_path,
        OpenFlags::SQLITE_OPEN_READ_WRITE
            | OpenFlags::SQLITE_OPEN_CREATE
            | OpenFlags::SQLITE_OPEN_FULL_MUTEX,
    )?;
    conn.execute_batch(
        "PRAGMA journal_mode=WAL;
         PRAGMA synchronous=NORMAL;
         PRAGMA busy_timeout=5000;
         PRAGMA foreign_keys=ON;",
    )?;
    Ok(conn)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn tmp_db() -> (TempDir, PathBuf) {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("rein-pool-test.db");
        (dir, p)
    }

    #[tokio::test]
    async fn pool_opens_n_conns_and_serves_them() {
        let (_dir, path) = tmp_db();
        let pool = ConnPool::new(&path, 3).unwrap();
        assert_eq!(pool.size(), 3);
        let m = pool.metrics();
        assert_eq!(m.size, 3);
        assert_eq!(m.idle, 3);
        assert_eq!(m.in_use, 0);

        let g1 = pool.get().await.unwrap();
        let g2 = pool.get().await.unwrap();
        let m = pool.metrics();
        assert_eq!(m.idle, 1);
        assert_eq!(m.in_use, 2);
        drop(g1);
        drop(g2);
        let m = pool.metrics();
        assert_eq!(m.idle, 3);
    }

    #[tokio::test]
    async fn pragma_wal_and_fk_applied() {
        let (_dir, path) = tmp_db();
        let pool = ConnPool::new(&path, 1).unwrap();
        let guard = pool.get().await.unwrap();
        let mode: String = guard
            .interact(|c| {
                Ok(c.query_row("PRAGMA journal_mode", [], |r| r.get::<_, String>(0))?)
            })
            .await
            .unwrap();
        assert_eq!(mode, "wal");

        let guard = pool.get().await.unwrap();
        let fk: i64 = guard
            .interact(|c| Ok(c.query_row("PRAGMA foreign_keys", [], |r| r.get::<_, i64>(0))?))
            .await
            .unwrap();
        assert_eq!(fk, 1);
    }

    #[tokio::test]
    async fn interact_returns_conn_on_success() {
        let (_dir, path) = tmp_db();
        let pool = ConnPool::new(&path, 1).unwrap();
        for _ in 0..5 {
            let guard = pool.get().await.unwrap();
            let one: i64 = guard
                .interact(|c| Ok(c.query_row("SELECT 1", [], |r| r.get::<_, i64>(0))?))
                .await
                .unwrap();
            assert_eq!(one, 1);
            assert_eq!(pool.metrics().idle, 1, "conn returned to pool");
        }
    }

    #[tokio::test]
    async fn concurrent_gets_serialize_when_pool_exhausted() {
        let (_dir, path) = tmp_db();
        let pool = ConnPool::new(&path, 2).unwrap();
        let p1 = pool.clone();
        let p2 = pool.clone();
        let p3 = pool.clone();
        let h1 = tokio::spawn(async move {
            let g = p1.get().await.unwrap();
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            drop(g);
        });
        let h2 = tokio::spawn(async move {
            let g = p2.get().await.unwrap();
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            drop(g);
        });
        // h3 must wait for h1 or h2 to release
        let h3 = tokio::spawn(async move {
            let t0 = std::time::Instant::now();
            let _g = p3.get().await.unwrap();
            assert!(
                t0.elapsed() >= std::time::Duration::from_millis(40),
                "pool exhaustion did not back-pressure: elapsed={:?}",
                t0.elapsed()
            );
        });
        h1.await.unwrap();
        h2.await.unwrap();
        h3.await.unwrap();
    }

    #[tokio::test]
    async fn sqlite_vec_extension_loads_via_pool() {
        let (_dir, path) = tmp_db();
        let pool = ConnPool::new(&path, 1).unwrap();
        let guard = pool.get().await.unwrap();
        let v: String = guard
            .interact(|c| Ok(c.query_row("SELECT vec_version()", [], |r| r.get::<_, String>(0))?))
            .await
            .unwrap();
        assert!(!v.is_empty(), "vec_version() returned empty: {v:?}");
    }

    #[tokio::test]
    async fn default_size_respects_cpu_count() {
        let size = default_pool_size();
        assert!((2..=8).contains(&size), "default_pool_size out of range: {size}");
    }

    #[tokio::test]
    async fn interact_panic_self_heals_via_replacement_conn() {
        // Codex Finding #1/#2: without replacement, a panicked closure
        // leaves the permit releasable but the free-list one short,
        // so the next get() hits expect("semaphore invariant"). With
        // the self-heal, the pool remains serviceable.
        let (_dir, path) = tmp_db();
        let pool = ConnPool::new(&path, 2).unwrap();

        let guard = pool.get().await.unwrap();
        let err = guard
            .interact(|_conn| -> ReinResult<()> { panic!("synthetic panic in closure") })
            .await
            .expect_err("panic should bubble up as error");
        assert!(
            format!("{err}").contains("panicked"),
            "error message should mention panic: {err}"
        );

        // Invariant: pool is back to full size via replacement conn.
        let m = pool.metrics();
        assert_eq!(m.size, 2);
        assert_eq!(m.idle, 2, "panic should not shrink pool if replacement succeeds");

        // And the next get() on the repopulated slot must not panic.
        let g1 = pool.get().await.unwrap();
        let g2 = pool.get().await.unwrap();
        drop(g1);
        drop(g2);
    }

    #[tokio::test]
    async fn detached_guard_dropped_without_put_back_self_heals() {
        // Codex round-0 Finding #5 + round-1 Finding #3: when put_back
        // is never called (caller leaked the conn or panicked between
        // detach and put_back), Drop must NOT simply forget the permit
        // — that monotonically shrinks the pool under repeated panics.
        // Drop instead opens a fresh replacement conn and returns it,
        // keeping pool `size` invariant.
        let (_dir, path) = tmp_db();
        let pool = ConnPool::new(&path, 3).unwrap();

        // Simulate 5 detach-then-drop cycles (mimics repeated panics in
        // a hot recall path). If Drop shrinks by one each time, after
        // 3 the pool is exhausted. Self-heal should keep it at 3.
        for _ in 0..5 {
            let guard = pool.get().await.unwrap();
            let (_conn, detached) = guard.detach();
            drop(detached); // _conn drops too, but the DetachedGuard
                            // opens a fresh one for the pool
        }

        // Pool is still at full size + fully idle.
        let m = pool.metrics();
        assert_eq!(m.size, 3, "self-heal should preserve pool size");
        assert_eq!(m.idle, 3, "fresh conns returned to pool");
        assert_eq!(m.available_permits, 3, "permits restored normally");

        // And subsequent get() calls still succeed.
        let _g1 = pool.get().await.unwrap();
        let _g2 = pool.get().await.unwrap();
        let _g3 = pool.get().await.unwrap();
    }

    #[test]
    fn get_without_tokio_runtime_errors_instead_of_panicking() {
        // Codex Finding #6: acquire_owned().await would panic without
        // a runtime. We now return a Config error. Exercise from a
        // plain std thread (no Tokio runtime entered) and poll the
        // future manually; the error should surface before it awaits.
        use std::future::Future;
        use std::pin::Pin;
        use std::task::{Context, Poll, Waker};
        let (_dir, path) = tmp_db();
        let pool = ConnPool::new(&path, 1).unwrap();

        let handle = std::thread::spawn(move || {
            let mut fut = Box::pin(pool.get());
            let waker = Waker::noop();
            let mut cx = Context::from_waker(waker);
            match Pin::new(&mut fut).poll(&mut cx) {
                Poll::Ready(Err(e)) => {
                    assert!(
                        format!("{e}").to_lowercase().contains("tokio"),
                        "error should mention tokio: {e}"
                    );
                }
                Poll::Ready(Ok(_)) => panic!("unexpected success without runtime"),
                Poll::Pending => panic!("should complete immediately with runtime error"),
            }
        });
        handle.join().expect("std-thread test succeeded");
    }

    #[tokio::test]
    async fn detach_roundtrip_via_sqlite_store_preserves_pool_invariants() {
        use crate::store::SqliteStore;
        use std::path::PathBuf;

        // Use an actual rein DB fixture so schema exists.
        let (_dir, path) = tmp_db();
        // Bootstrap schema via the usual path (opens + closes its own conn).
        {
            let store = SqliteStore::new(&path, "text-embedding-3-large", 3072).unwrap();
            drop(store);
        }
        let pool = ConnPool::new(&path, 2).unwrap();

        // Round-trip a conn: pool.get → detach → SqliteStore → into_conn → put_back
        for _ in 0..5 {
            let guard = pool.get().await.unwrap();
            let (conn, detached) = guard.detach();
            let store =
                SqliteStore::from_conn(conn, PathBuf::from(&path), 3072);
            // A trivial schema-level query — confirms we're really on the
            // migrated DB, not an in-memory throwaway.
            let count: i64 = store
                .conn()
                .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
                .unwrap();
            assert!(count >= 0);
            let conn = store.into_conn();
            detached.put_back(conn);
        }

        // All conns returned to pool; size invariant holds.
        let m = pool.metrics();
        assert_eq!(m.size, 2);
        assert_eq!(m.idle, 2);
        assert_eq!(m.available_permits, 2);
    }
}
