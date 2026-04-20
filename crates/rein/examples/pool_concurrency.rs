//! D0 bench — v0.22 async-hardening go/no-go.
//!
//! Answers: does rein's workload actually benefit from multi-connection
//! parallel reads under SQLite WAL + FULL_MUTEX? The v2 spec claims yes
//! (per-conn WAL readers don't block each other); v1 implicitly claimed no
//! (single-conn FULL_MUTEX serializes join!). This bench measures both.
//!
//! Four scenarios:
//!   (a) 1 conn × 3 serial reads
//!   (b) 1 conn shared via Mutex × 3 tokio::join! (should ≈ a; confirms v1 α fails)
//!   (c) 3 thread-local conns × 3 tokio::join! (validates γ)
//!   (d) 8 thread-local conns × 10 concurrent "recall" (each = 3 reads; pool scale)
//!
//! Run:  cargo run --example pool_concurrency --release
//!
//! Go/no-go: (c) vs (a) wall-clock improvement ≥ 30% → proceed to D1.

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use rusqlite::{Connection, OpenFlags};
use tempfile::TempDir;
use tokio::sync::Mutex;

const FIXTURE_ROWS: usize = 2000;
const BENCH_ITERS: usize = 200;
const RECALL_FANOUT: usize = 3;
const SCENARIO_D_CONCURRENCY: usize = 10;

thread_local! {
    static TL_CONN: RefCell<Option<Connection>> = const { RefCell::new(None) };
}

fn open_conn(db_path: &Path) -> Connection {
    let conn = Connection::open_with_flags(
        db_path,
        OpenFlags::SQLITE_OPEN_READ_WRITE
            | OpenFlags::SQLITE_OPEN_CREATE
            | OpenFlags::SQLITE_OPEN_FULL_MUTEX,
    )
    .expect("open conn");
    conn.execute_batch(
        "PRAGMA journal_mode=WAL;
         PRAGMA synchronous=NORMAL;
         PRAGMA busy_timeout=5000;
         PRAGMA foreign_keys=ON;",
    )
    .expect("pragma setup");
    conn
}

fn with_tl_conn<R>(db_path: &Path, f: impl FnOnce(&Connection) -> R) -> R {
    TL_CONN.with(|cell| {
        let mut c = cell.borrow_mut();
        if c.is_none() {
            *c = Some(open_conn(db_path));
        }
        f(c.as_ref().unwrap())
    })
}

fn setup_fixture() -> (TempDir, PathBuf) {
    let tmp = TempDir::new().expect("tempdir");
    let db_path = tmp.path().join("bench.db");

    // Seed writer conn
    let conn = open_conn(&db_path);
    conn.execute_batch(
        "CREATE TABLE memories (
             id       INTEGER PRIMARY KEY,
             topic    TEXT NOT NULL,
             content  TEXT NOT NULL,
             created  INTEGER NOT NULL
         );
         CREATE VIRTUAL TABLE memories_fts USING fts5(
             topic, content,
             content='memories', content_rowid='id',
             tokenize='porter unicode61'
         );
         CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
             INSERT INTO memories_fts(rowid, topic, content)
             VALUES (new.id, new.topic, new.content);
         END;",
    )
    .expect("schema");

    // Realistic-shaped content: ~200 chars each, word-diverse enough for FTS5
    let tx = conn.unchecked_transaction().unwrap();
    let words = [
        "memory", "recall", "vector", "embedding", "search", "fusion", "pool",
        "connection", "parallel", "concurrent", "async", "tokio", "rusqlite",
        "tantivy", "hnsw", "sqlite", "adaptive", "feedback", "dedup", "concept",
        "episode", "temporal", "cluster", "survival", "decay", "canonical",
        "evidence", "knowledge", "graph", "traversal", "ranking", "rerank",
    ];
    for i in 0..FIXTURE_ROWS {
        let topic = format!("topic-{}", i % 50);
        // Pseudorandom word bag
        let mut content = String::new();
        for j in 0..20 {
            content.push_str(words[(i * 31 + j * 17) % words.len()]);
            content.push(' ');
        }
        content.push_str(&format!("item-{i}"));
        tx.execute(
            "INSERT INTO memories (topic, content, created) VALUES (?, ?, ?)",
            rusqlite::params![topic, content, i as i64],
        )
        .unwrap();
    }
    tx.commit().unwrap();
    conn.execute_batch("PRAGMA wal_checkpoint(PASSIVE);").unwrap();
    eprintln!(
        "[fixture] seeded {FIXTURE_ROWS} rows at {}",
        db_path.display()
    );
    (tmp, db_path)
}

/// A representative query shape: FTS5 MATCH + fallback scan.
/// We vary the query string per iteration so FTS5 cache doesn't short-circuit.
fn do_one_query(conn: &Connection, query_word: &str) -> usize {
    let mut stmt = conn
        .prepare_cached(
            "SELECT m.id, m.topic, m.content
             FROM memories_fts f
             JOIN memories m ON m.id = f.rowid
             WHERE memories_fts MATCH ?
             ORDER BY rank
             LIMIT 20",
        )
        .unwrap();
    let rows: Vec<(i64, String, String)> = stmt
        .query_map([query_word], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?))
        })
        .unwrap()
        .filter_map(|r| r.ok())
        .collect();
    rows.len()
}

fn query_words() -> Vec<&'static str> {
    vec![
        "memory", "recall", "vector", "embedding", "pool", "tokio", "tantivy",
        "hnsw", "adaptive", "feedback", "concept", "episode", "ranking",
        "parallel", "concurrent", "async", "cluster", "survival", "graph",
        "canonical",
    ]
}

fn report(label: &str, times: &mut [Duration], started: Instant, total_iters: usize) {
    times.sort();
    let total = started.elapsed();
    let p50 = times[times.len() / 2];
    let p95 = times[(times.len() as f64 * 0.95) as usize];
    let p99 = times[(times.len() as f64 * 0.99) as usize];
    let mean = times.iter().sum::<Duration>() / times.len() as u32;
    eprintln!(
        "  {label:<40} wall={total:>8.2?} iters={total_iters:>4} mean={mean:>8.2?} p50={p50:>8.2?} p95={p95:>8.2?} p99={p99:>8.2?}"
    );
}

async fn scenario_a_serial_one_conn(db_path: &Path) {
    let conn = open_conn(db_path);
    let words = query_words();
    let started = Instant::now();
    let mut times = Vec::with_capacity(BENCH_ITERS);
    for i in 0..BENCH_ITERS {
        let t0 = Instant::now();
        let w0 = words[i % words.len()];
        let w1 = words[(i * 2) % words.len()];
        let w2 = words[(i * 3) % words.len()];
        do_one_query(&conn, w0);
        do_one_query(&conn, w1);
        do_one_query(&conn, w2);
        times.push(t0.elapsed());
    }
    report("(a) 1 conn × 3 serial reads", &mut times, started, BENCH_ITERS);
}

async fn scenario_b_shared_conn_join(db_path: &Path) {
    let conn = Arc::new(Mutex::new(open_conn(db_path)));
    let words = query_words();
    let started = Instant::now();
    let mut times = Vec::with_capacity(BENCH_ITERS);
    for i in 0..BENCH_ITERS {
        let t0 = Instant::now();
        let w0 = words[i % words.len()].to_string();
        let w1 = words[(i * 2) % words.len()].to_string();
        let w2 = words[(i * 3) % words.len()].to_string();
        let c0 = conn.clone();
        let c1 = conn.clone();
        let c2 = conn.clone();
        let (r0, r1, r2) = tokio::join!(
            tokio::task::spawn_blocking(move || {
                let h = tokio::runtime::Handle::current();
                let g = h.block_on(c0.lock());
                do_one_query(&g, &w0)
            }),
            tokio::task::spawn_blocking(move || {
                let h = tokio::runtime::Handle::current();
                let g = h.block_on(c1.lock());
                do_one_query(&g, &w1)
            }),
            tokio::task::spawn_blocking(move || {
                let h = tokio::runtime::Handle::current();
                let g = h.block_on(c2.lock());
                do_one_query(&g, &w2)
            }),
        );
        r0.unwrap();
        r1.unwrap();
        r2.unwrap();
        times.push(t0.elapsed());
    }
    report(
        "(b) 1 conn × 3 join! (shared Mutex)",
        &mut times,
        started,
        BENCH_ITERS,
    );
}

async fn scenario_c_per_channel_conn(db_path: &Path) {
    let words = query_words();
    // Warm thread-local conns
    for _ in 0..RECALL_FANOUT {
        let p = db_path.to_owned();
        tokio::task::spawn_blocking(move || {
            with_tl_conn(&p, |_| {});
        })
        .await
        .unwrap();
    }
    let started = Instant::now();
    let mut times = Vec::with_capacity(BENCH_ITERS);
    for i in 0..BENCH_ITERS {
        let t0 = Instant::now();
        let p0 = db_path.to_owned();
        let p1 = db_path.to_owned();
        let p2 = db_path.to_owned();
        let w0 = words[i % words.len()].to_string();
        let w1 = words[(i * 2) % words.len()].to_string();
        let w2 = words[(i * 3) % words.len()].to_string();
        let (r0, r1, r2) = tokio::join!(
            tokio::task::spawn_blocking(move || with_tl_conn(&p0, |c| do_one_query(c, &w0))),
            tokio::task::spawn_blocking(move || with_tl_conn(&p1, |c| do_one_query(c, &w1))),
            tokio::task::spawn_blocking(move || with_tl_conn(&p2, |c| do_one_query(c, &w2))),
        );
        r0.unwrap();
        r1.unwrap();
        r2.unwrap();
        times.push(t0.elapsed());
    }
    report(
        "(c) 3 thread-local conns × 3 join!",
        &mut times,
        started,
        BENCH_ITERS,
    );
}

async fn scenario_d_pool_scale(db_path: &Path) {
    let words = query_words();
    // Warm ~8 thread-local conns by spawning 8 blocking tasks
    let mut warmup = Vec::new();
    for _ in 0..8 {
        let p = db_path.to_owned();
        warmup.push(tokio::task::spawn_blocking(move || {
            with_tl_conn(&p, |_| {});
            std::thread::sleep(Duration::from_millis(50));
        }));
    }
    for w in warmup {
        w.await.unwrap();
    }
    let started = Instant::now();
    let mut times = Vec::with_capacity(BENCH_ITERS / SCENARIO_D_CONCURRENCY);
    for round in 0..(BENCH_ITERS / SCENARIO_D_CONCURRENCY) {
        let t0 = Instant::now();
        let mut tasks = Vec::with_capacity(SCENARIO_D_CONCURRENCY);
        for slot in 0..SCENARIO_D_CONCURRENCY {
            let p0 = db_path.to_owned();
            let p1 = db_path.to_owned();
            let p2 = db_path.to_owned();
            let seed = round * SCENARIO_D_CONCURRENCY + slot;
            let w0 = words[seed % words.len()].to_string();
            let w1 = words[(seed * 2) % words.len()].to_string();
            let w2 = words[(seed * 3) % words.len()].to_string();
            tasks.push(tokio::spawn(async move {
                let (r0, r1, r2) = tokio::join!(
                    tokio::task::spawn_blocking(move || with_tl_conn(&p0, |c| do_one_query(c, &w0))),
                    tokio::task::spawn_blocking(move || with_tl_conn(&p1, |c| do_one_query(c, &w1))),
                    tokio::task::spawn_blocking(move || with_tl_conn(&p2, |c| do_one_query(c, &w2))),
                );
                (r0.unwrap(), r1.unwrap(), r2.unwrap())
            }));
        }
        for t in tasks {
            t.await.unwrap();
        }
        times.push(t0.elapsed());
    }
    report(
        "(d) 8-conn pool × 10 concurrent recall",
        &mut times,
        started,
        BENCH_ITERS,
    );
}

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() {
    let (_tmp, db_path) = setup_fixture();
    eprintln!();
    eprintln!("=== rein v0.22 D0 — pool concurrency bench ===");
    eprintln!(
        "  fixture: {FIXTURE_ROWS} rows; iters: {BENCH_ITERS}; per-iter fanout: {RECALL_FANOUT}"
    );
    eprintln!();

    // Warm-up pass (prime page cache, fts aux tables)
    eprintln!("[warmup]");
    scenario_a_serial_one_conn(&db_path).await;
    eprintln!();

    eprintln!("[measure]");
    scenario_a_serial_one_conn(&db_path).await;
    scenario_b_shared_conn_join(&db_path).await;
    scenario_c_per_channel_conn(&db_path).await;
    scenario_d_pool_scale(&db_path).await;
    eprintln!();

    eprintln!("[go/no-go rule] (c).mean vs (a).mean improvement ≥ 30% → proceed to D1.");
    eprintln!("[go/no-go rule] (c).mean vs (a).mean improvement < 30% → revisit γ, consider v0.23 δ.");
}
