//! D4 — process-level pool cache semantics on `ReinConfig::open_store`.
//!
//! Validates the v0.22 service-wiring invariants introduced in
//! `docs/superpowers/specs/2026-04-21-v0.22-async-hardening-design.md`:
//!
//! 1. Inside a tokio runtime, `open_store` attaches a pool.
//! 2. Two successive `open_store` calls for the same db_path share the
//!    **same** `ConnPool` instance (Arc identity) — i.e. the cache hits.
//! 3. Two `open_store` calls for **different** db_paths get different
//!    pools (keyed-by-path, not singleton).
//! 4. After all stores holding a pool drop, the `Weak` in the cache goes
//!    stale and the next `open_store` rebuilds a fresh pool.
//! 5. `:memory:` DBs bypass the pool (store.pool() is None).
//! 6. `REIN_ASYNC_P1=0` opt-out bypasses the pool.
//! 7. Non-tokio (`#[test]`) callers bypass the pool.

use std::sync::Arc;

use rein::config::ReinConfig;
use rein::store::pool::ConnPool;
use tempfile::TempDir;

fn config_for(db_path: &std::path::Path) -> ReinConfig {
    let mut c = ReinConfig::default();
    c.database.path = db_path.to_string_lossy().into_owned();
    c
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn same_path_hits_cache_returns_same_pool() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("same.db");
    let cfg = config_for(&db_path);

    let s1 = cfg.open_store().expect("open_store #1");
    let s2 = cfg.open_store().expect("open_store #2");

    let p1 = s1.pool().expect("store #1 has pool in tokio runtime");
    let p2 = s2.pool().expect("store #2 has pool in tokio runtime");
    assert!(
        Arc::ptr_eq(p1, p2),
        "cache must return the same ConnPool Arc for the same db_path"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn different_paths_get_different_pools() {
    let tmp = TempDir::new().unwrap();
    let db_a = tmp.path().join("a.db");
    let db_b = tmp.path().join("b.db");
    let store_a = config_for(&db_a).open_store().expect("open a");
    let store_b = config_for(&db_b).open_store().expect("open b");
    let pa = store_a.pool().expect("a has pool");
    let pb = store_b.pool().expect("b has pool");
    assert!(
        !Arc::ptr_eq(pa, pb),
        "distinct db_paths must resolve to distinct pools (keyed-by-path)"
    );
    assert_eq!(pa.db_path(), db_a);
    assert_eq!(pb.db_path(), db_b);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn weak_cache_rebuilds_after_last_store_drops() {
    use std::sync::Weak;
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("weak.db");
    let cfg = config_for(&db_path);

    // First store → pool #1.
    let s1 = cfg.open_store().expect("open_store #1");
    let p1_strong: Arc<ConnPool> = s1.pool().expect("pool attached").clone();
    // Snapshot a Weak alongside the raw pointer so we can (a) verify the
    // cached entry actually went stale after drop, and (b) prove identity
    // differs across rebuild. We must drop `p1_strong` alongside `s1` —
    // a lingering strong ref would keep the pool alive and defeat the test.
    let weak: Weak<ConnPool> = Arc::downgrade(&p1_strong);
    let p1_ptr: *const ConnPool = Arc::as_ptr(&p1_strong);
    drop(p1_strong);
    drop(s1);

    assert!(
        weak.upgrade().is_none(),
        "after all Arc<ConnPool> strong refs drop, the cached Weak must be stale"
    );

    // Next open must rebuild.
    let s2 = cfg.open_store().expect("open_store #2 after drop");
    let p2: &Arc<ConnPool> = s2.pool().expect("pool re-attached");
    assert_eq!(p2.db_path(), db_path);
    assert!(p2.metrics().idle > 0, "rebuilt pool has idle conns");
    // Identity check: the allocator MAY reuse the same address, so a
    // strict `ne` would be flaky. Instead, the stale-weak assertion above
    // plus the rebuild succeeding is sufficient evidence that cache went
    // through the rebuild branch rather than returning a dangling Weak.
    let _ = p1_ptr;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn memory_db_bypasses_pool() {
    let mut c = ReinConfig::default();
    c.database.path = ":memory:".into();
    let store = c.open_store().expect("open :memory: store");
    assert!(
        store.pool().is_none(),
        ":memory: DB must bypass the pool — each conn is a distinct empty DB"
    );
}

#[test]
fn non_tokio_caller_bypasses_pool() {
    // Plain `#[test]` has no tokio runtime; `Handle::try_current()` returns
    // Err and open_store must skip the pool construction entirely.
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("notokio.db");
    let cfg = config_for(&db_path);
    let store = cfg.open_store().expect("open_store in non-tokio thread");
    assert!(
        store.pool().is_none(),
        "non-tokio caller must bypass the pool (CLI direct path)"
    );
}

// The REIN_ASYNC_P1=0 opt-out is intentionally not tested here. `set_var` is
// unsafe in Rust 2024 (data race with libc getenv) and would pollute
// parallel tests in the same process, producing flaky failures in unrelated
// tests that call `open_store`. The opt-out is a single `if
// std::env::var(...).ok().as_deref() == Some("0")` early-return; its
// correctness is reviewable by inspection.
