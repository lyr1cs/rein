//! D5 bench — F1 go/no-go.
//!
//! Measures feedback INSERT latency under concurrent `store_with_dedup`-shaped
//! writers. Determines which F1 solution to pick: D / E / C / A.
//!
//! Premise to test: spec §F1 claims `store_with_dedup`'s `BEGIN IMMEDIATE` holds
//! the write lock "long enough" to block feedback INSERT.  But the actual code
//! (store/sqlite.rs:1396-1412) has been carefully optimized — the LLM call
//! (1-2s) was explicitly moved OUTSIDE `BEGIN IMMEDIATE`.  Only ~5-20ms of
//! mechanical SQL remains inside.  If this bench shows tail latency of emit_event
//! under contention is small, the whole F1 project may be unnecessary.
//!
//! Scenarios:
//!   (a) baseline: solo emit_event, no contenders
//!   (b) contention: emit_event while 3 concurrent writers run
//!       `store_with_dedup`-shaped transactions at ~500 tx/sec each
//!
//! Decision:
//!   (b).p99 < 50ms     → D (do nothing; spec §F1 premise is false)
//!   50ms < p99 < 500ms → E (spawn_blocking fire-and-forget, ~30 LoC)
//!   p99 > 500ms        → C (shrink BEGIN IMMEDIATE) or A (separate DB file)
//!
//! Run:  cargo run --example feedback_contention --release

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use rusqlite::{Connection, OpenFlags};
use tempfile::TempDir;

const FIXTURE_ROWS: usize = 1000;
const BENCH_ITERS: usize = 2000; // larger n for robust p99/p99.5
const CONTENDER_COUNT: usize = 3; // concurrent store-like writers
const CONTENDER_PACE_MS: u64 = 2; // ~500 tx/sec per writer; matches real rein peak

fn open_conn(db_path: &Path) -> Connection {
    let conn = Connection::open_with_flags(
        db_path,
        OpenFlags::SQLITE_OPEN_READ_WRITE
            | OpenFlags::SQLITE_OPEN_CREATE
            | OpenFlags::SQLITE_OPEN_FULL_MUTEX,
    )
    .expect("open conn");
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA synchronous = NORMAL;
         PRAGMA busy_timeout = 5000;
         PRAGMA foreign_keys = ON;",
    )
    .expect("pragmas");
    conn
}

fn setup_fixture() -> (TempDir, PathBuf) {
    let tmp = TempDir::new().expect("tempdir");
    let db_path = tmp.path().join("bench.db");
    let conn = open_conn(&db_path);

    // Schema modeled on rein's actual tables — enough shape to mimic the
    // SQL work that store_with_dedup performs inside BEGIN IMMEDIATE.
    conn.execute_batch(
        "CREATE TABLE memories (
             id         INTEGER PRIMARY KEY,
             topic      TEXT NOT NULL,
             content    TEXT NOT NULL,
             cluster_id INTEGER,
             strength   REAL DEFAULT 1.0,
             created    INTEGER NOT NULL
         );
         CREATE INDEX idx_m_topic ON memories(topic);
         CREATE INDEX idx_m_cluster ON memories(cluster_id);

         CREATE VIRTUAL TABLE memories_fts USING fts5(
             topic, content,
             content='memories', content_rowid='id',
             tokenize='porter unicode61'
         );
         CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
             INSERT INTO memories_fts(rowid, topic, content)
             VALUES (new.id, new.topic, new.content);
         END;

         CREATE TABLE feedback_events (
             id         INTEGER PRIMARY KEY AUTOINCREMENT,
             event_type TEXT NOT NULL,
             request_id TEXT,
             memory_id  TEXT,
             ts         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
         );
         CREATE INDEX idx_fe_ts ON feedback_events(ts);

         CREATE TABLE concept_links (
             id         INTEGER PRIMARY KEY,
             memory_id  INTEGER,
             concept_id INTEGER,
             ts         INTEGER
         );",
    )
    .expect("schema");

    // Seed 1000 memories with word-diverse content (so FTS inside dedup check has real work)
    let words = [
        "memory", "recall", "vector", "embedding", "search", "fusion", "pool",
        "connection", "parallel", "concurrent", "async", "tokio", "rusqlite",
        "tantivy", "hnsw", "sqlite", "adaptive", "feedback", "dedup", "concept",
    ];
    let tx = conn.unchecked_transaction().unwrap();
    for i in 0..FIXTURE_ROWS {
        let mut content = String::new();
        for j in 0..10 {
            content.push_str(words[(i * 31 + j * 17) % words.len()]);
            content.push(' ');
        }
        content.push_str(&format!("item-{i}"));
        tx.execute(
            "INSERT INTO memories (topic, content, cluster_id, created) VALUES (?, ?, ?, ?)",
            rusqlite::params![
                format!("topic-{}", i % 50),
                content,
                (i % 10) as i64,
                i as i64,
            ],
        )
        .unwrap();
    }
    tx.commit().unwrap();
    conn.execute_batch("PRAGMA wal_checkpoint(PASSIVE);").unwrap();
    eprintln!(
        "[fixture] {FIXTURE_ROWS} memories seeded at {}",
        db_path.display()
    );
    (tmp, db_path)
}

/// Simulate rein's `emit_event`: a single INSERT into feedback_events.
/// This is what recall.rs:1529 does — ~0.5ms uncontended.
fn do_emit_event(conn: &Connection, i: usize) {
    conn.execute(
        "INSERT INTO feedback_events (event_type, request_id, memory_id) VALUES (?, ?, ?)",
        rusqlite::params![
            "recall_complete",
            format!("req-{i}"),
            format!("mem-{}", i % FIXTURE_ROWS),
        ],
    )
    .expect("emit_event");
}

/// Simulate `store_with_dedup`'s in-transaction body — what runs inside
/// BEGIN IMMEDIATE after LLM preflight is already done.
///
/// Mirrors real rein work:
///   - check_dedup (FTS MATCH query against memories_fts)
///   - INSERT memories (triggers FTS5 index update)
///   - INSERT concept_links (3 rows)
///   - UPDATE memories.strength for a cluster (simulates decay touch)
fn do_store_with_dedup(conn: &Connection, i: usize) {
    conn.execute_batch("BEGIN IMMEDIATE").expect("begin");

    // check_dedup: FTS MATCH (same shape rein uses)
    let word = ["memory", "recall", "vector", "pool", "tokio"][i % 5];
    let _: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH ?",
            [word],
            |r| r.get(0),
        )
        .unwrap();

    // INSERT memories (FTS5 trigger fires). INTEGER PRIMARY KEY auto-assigns rowid.
    conn.execute(
        "INSERT INTO memories (topic, content, cluster_id, created)
         VALUES (?, ?, ?, ?)",
        rusqlite::params![
            format!("new-topic-{i}"),
            format!("new content {word} {i} parallel concurrent"),
            (i % 10) as i64,
            i as i64,
        ],
    )
    .unwrap();
    let new_id = conn.last_insert_rowid();

    // INSERT 3 concept_links
    for j in 0..3 {
        conn.execute(
            "INSERT INTO concept_links (memory_id, concept_id, ts) VALUES (?, ?, ?)",
            rusqlite::params![new_id, j as i64, i as i64],
        )
        .unwrap();
    }

    // UPDATE a cluster's strength (decay touch)
    conn.execute(
        "UPDATE memories SET strength = strength * 0.99 WHERE cluster_id = ?",
        rusqlite::params![(i % 10) as i64],
    )
    .unwrap();

    conn.execute_batch("COMMIT").expect("commit");
}

fn report(label: &str, times: &mut [Duration]) {
    times.sort();
    let n = times.len();
    let p50 = times[n / 2];
    let p95 = times[(n as f64 * 0.95) as usize];
    let p99 = times[(n as f64 * 0.99) as usize];
    let mean = times.iter().sum::<Duration>() / n as u32;
    let max = times.last().copied().unwrap_or_default();
    eprintln!(
        "  {label:<42} n={n:<4} mean={mean:>8.2?}  p50={p50:>8.2?}  p95={p95:>8.2?}  p99={p99:>8.2?}  max={max:>8.2?}"
    );
}

fn scenario_a_baseline(db_path: &Path) -> Duration {
    eprintln!("\n[A] baseline — solo emit_event, no contenders");
    let conn = open_conn(db_path);
    let mut times = Vec::with_capacity(BENCH_ITERS);
    for i in 0..BENCH_ITERS {
        let t0 = Instant::now();
        do_emit_event(&conn, i);
        times.push(t0.elapsed());
    }
    let mut sorted = times.clone();
    sorted.sort();
    let p99 = sorted[(sorted.len() as f64 * 0.99) as usize];
    report("(a) solo emit_event", &mut times);
    p99
}

fn scenario_b_contention(db_path: &Path) -> Duration {
    eprintln!(
        "\n[B] contention — emit_event during {CONTENDER_COUNT} concurrent store_with_dedup writers (~500 tx/sec each)"
    );

    let running = Arc::new(AtomicBool::new(true));
    let tx_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    // Disambiguate contender id ranges to avoid PK collisions.
    // Each contender gets a 10M-slot block so their new_ids never overlap.
    let id_block = 10_000_000usize;
    let mut handles = vec![];
    for contender_idx in 0..CONTENDER_COUNT {
        let db_path = db_path.to_owned();
        let running = running.clone();
        let tx_count = tx_count.clone();
        let base = contender_idx * id_block;
        let h = thread::spawn(move || {
            let conn = open_conn(&db_path);
            let mut i = 0usize;
            while running.load(Ordering::Relaxed) {
                do_store_with_dedup(&conn, base + i);
                i += 1;
                tx_count.fetch_add(1, Ordering::Relaxed);
                if CONTENDER_PACE_MS > 0 {
                    thread::sleep(Duration::from_millis(CONTENDER_PACE_MS));
                }
            }
        });
        handles.push(h);
    }

    // Warm up — let contenders reach steady state
    thread::sleep(Duration::from_millis(200));

    let conn = open_conn(db_path);
    let started = Instant::now();
    let mut times = Vec::with_capacity(BENCH_ITERS);
    for i in 0..BENCH_ITERS {
        let t0 = Instant::now();
        do_emit_event(&conn, 1_000_000 + i);
        times.push(t0.elapsed());
    }
    let bench_wall = started.elapsed();

    running.store(false, Ordering::Relaxed);
    for h in handles {
        h.join().unwrap();
    }

    let total_contender_tx = tx_count.load(Ordering::Relaxed);
    eprintln!(
        "  (contenders ran {} store_with_dedup tx during {:.2?} → {:.0} tx/sec total)",
        total_contender_tx,
        bench_wall,
        total_contender_tx as f64 / bench_wall.as_secs_f64()
    );

    let mut sorted = times.clone();
    sorted.sort();
    let p99 = sorted[(sorted.len() as f64 * 0.99) as usize];
    report("(b) emit_event under contention", &mut times);
    p99
}

fn main() {
    eprintln!("D5 F1 bench — feedback INSERT latency under concurrent store_with_dedup writers");
    eprintln!(
        "Config: {FIXTURE_ROWS} memories, {BENCH_ITERS} samples, {CONTENDER_COUNT} contenders @ {CONTENDER_PACE_MS}ms pace"
    );

    let (_tmp, db_path) = setup_fixture();

    let p99_a = scenario_a_baseline(&db_path);
    let p99_b = scenario_b_contention(&db_path);

    let ratio = p99_b.as_secs_f64() / p99_a.as_secs_f64().max(1e-9);
    eprintln!("\n=== Decision ===");
    eprintln!("  p99 baseline (a):  {p99_a:?}");
    eprintln!("  p99 contention (b): {p99_b:?}  ({ratio:.1}x baseline)");
    eprintln!();
    if p99_b < Duration::from_millis(50) {
        eprintln!("  → D (do nothing)           p99 < 50ms, spec §F1 premise does NOT hold");
    } else if p99_b < Duration::from_millis(500) {
        eprintln!("  → E (spawn_blocking)       50 < p99 < 500ms, fire-and-forget ~30 LoC sufficient");
    } else {
        eprintln!("  → C (shrink BEGIN IMMEDIATE) or A (separate DB)  p99 > 500ms, real bottleneck");
    }
}
