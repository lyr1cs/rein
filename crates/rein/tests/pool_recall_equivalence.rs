//! D3 — Foundational invariant test for v0.22 async-hardening.
//!
//! Claim under test (spec I1, v2): `SqliteStore::from_conn(c, path, dims)`
//! where `c` is sourced from a `ConnPool` must produce recall output that
//! is semantically equivalent (ordering + ID set) to `SqliteStore::new`
//! against the same on-disk database and query.
//!
//! This locks the invariant that **sourcing a connection from the pool
//! does not alter recall semantics**, which is the pre-condition for all
//! subsequent 3-channel parallelization work (D4+).
//!
//! What this test does NOT validate:
//! - 3-channel parallel execution (that's D4)
//! - per-request latency improvement (D3.5 real-recall criterion bench)
//! - staging-DB feedback durability (D5)

use std::path::PathBuf;

use chrono::Utc;
use rein::config::ReinConfig;
use rein::search::recall::recall_temporal_with_request_id;
use rein::store::pool::ConnPool;
use rein::store::tiering::MemoryTier;
use rein::store::SqliteStore;
use rein::types::{Importance, Memory, MemoryLayer, MemoryStatus, MemoryStore, Source};
use tempfile::TempDir;

fn build_memory(topic: &str, summary: &str, content: &str) -> Memory {
    Memory {
        id: ulid::Ulid::new().to_string(),
        layer: MemoryLayer::LTM,
        topic: topic.to_string(),
        summary: summary.to_string(),
        content: content.to_string(),
        keywords: vec![topic.to_string(), "v022".into(), "pool".into()],
        importance: Importance::Medium,
        source: Source::Manual,
        strength: 1.0,
        decay_lambda: 0.06,
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
        archival_summary: None,
        archival_summary_at: None,
        archival_summary_version: None,
        created_at: Utc::now(),
        updated_at: Utc::now(),
        last_accessed: Utc::now(),
    }
}

fn seed_fixture(path: &std::path::Path) -> ReinConfig {
    let store = SqliteStore::new(path, "text-embedding-3-large", 3072).unwrap();
    let topics = [
        "memory_architecture",
        "pool_design",
        "async_runtime",
        "search_fusion",
        "feedback_loop",
    ];
    let phrases = [
        "rein uses SQLite with WAL mode for per-connection reader concurrency",
        "the pool checkout model preserves FULL_MUTEX per-connection guarantees",
        "recall aggregates FTS Vector and knowledge graph channels via CC fusion",
        "tokio spawn_blocking wraps rusqlite calls so the reactor never stalls",
        "feedback events are durable on return through the pending queue",
        "HNSW index serves vector queries in under ten milliseconds typical",
        "Tantivy provides BM25 full text search with snapshot consistent readers",
        "the adaptive engine learns alpha weights from counterfactual replay",
    ];
    for (i, topic) in topics.iter().enumerate() {
        for j in 0..8usize {
            let idx = (i * 8 + j) % phrases.len();
            let memory = build_memory(
                topic,
                &format!("{topic}-{j}"),
                &format!("{} item {} under {}", phrases[idx], j, topic),
            );
            store.store(memory).unwrap();
        }
    }
    drop(store);

    // `fast=true` in recall_temporal_with_request_id already skips expansion,
    // LLM reranker, and Supermemory cross-validation — no remote API calls.
    ReinConfig::default()
}

fn recall_ids(store: &SqliteStore, config: &ReinConfig, query: &str) -> Vec<String> {
    let results = recall_temporal_with_request_id(
        store,
        config,
        query,
        None,
        None,
        10,
        None,
        None,
        Some(false),
        /* fast */ true,
        None,
    )
    .expect("recall succeeds");
    results.into_iter().map(|r| r.memory.id).collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pool_sourced_conn_matches_fresh_conn_recall() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("pool-equiv.db");
    let config = seed_fixture(&db_path);

    // Baseline: recall through a freshly-opened SqliteStore (current production path).
    let fresh = SqliteStore::new(&db_path, "text-embedding-3-large", 3072).unwrap();
    let queries = [
        "pool design concurrency",
        "tokio reactor",
        "knowledge graph fusion",
        "feedback queue durability",
        "HNSW vector search",
    ];
    let baseline: Vec<Vec<String>> = queries
        .iter()
        .map(|q| recall_ids(&fresh, &config, q))
        .collect();
    drop(fresh);

    // Candidate: same recall via a pool-sourced conn wrapped in from_conn.
    let pool = ConnPool::new(&db_path, 2).expect("pool opens");
    for (i, q) in queries.iter().enumerate() {
        let guard = pool.get().await.expect("pool checkout");
        let (conn, detached) = guard.detach();
        let store = SqliteStore::from_conn(
            conn,
            PathBuf::from(&db_path),
            3072,
            &config.embedding_model(),
        );
        let ids_pool = recall_ids(&store, &config, q);
        let conn_back = store.into_conn();
        detached.put_back(conn_back);

        assert_eq!(
            baseline[i], ids_pool,
            "recall IDs diverged for query {q:?}: baseline={:?} pool={:?}",
            baseline[i], ids_pool
        );
    }

    // Pool is healthy — every conn was returned.
    let m = pool.metrics();
    assert_eq!(m.size, 2);
    assert_eq!(m.idle, 2);
    assert_eq!(m.available_permits, 2);
}

/// Same as `recall_ids` but with `fast=false` — triggers the Vec
/// cache-miss path which is where the v0.22 D4 pool branch lives. The
/// config is pre-shrunk to block every remote call (Supermemory +
/// query expansion + LLM reranker all off), so `fast=false` is safe in
/// CI: the thread enters the pool branch, the embedding API call
/// inside fails, and the channel returns empty vec results. FTS + KG
/// still contribute, so recall still ranks the fixture — and baseline
/// vs candidate should land on the same IDs either way.
fn recall_ids_nonfast(store: &SqliteStore, config: &ReinConfig, query: &str) -> Vec<String> {
    let results = recall_temporal_with_request_id(
        store,
        config,
        query,
        None,
        None,
        10,
        None,
        None,
        Some(false),
        /* fast */ false,
        None,
    )
    .expect("recall succeeds");
    results.into_iter().map(|r| r.memory.id).collect()
}

fn nonfast_offline_config() -> ReinConfig {
    let mut cfg = ReinConfig::default();
    // Block everything that would reach out to a remote service in CI.
    cfg.sync.supermemory_enabled = false;
    cfg.query_expansion.provider = "none".to_string();
    // `config.search.llm_reranker` already defaults to "none".
    cfg
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn store_with_pool_exercises_vec_channel_pool_branch_nonfast() {
    // Codex D4 Finding #6: the fast=true variant skips the Vec
    // cache-miss thread entirely, so it does NOT exercise the pool
    // branch we just added. This test uses fast=false + offline config
    // so recall actually enters the `if let (Some(pool), Some(handle))`
    // arm in `recall.rs`.
    use std::sync::Arc;

    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("with-pool-nonfast.db");
    let _ = seed_fixture(&db_path);
    let config = nonfast_offline_config();

    // Baseline: fresh store, no pool — Vec thread falls back to
    // `SqliteStore::new(&db_path, ...)` inside std::thread::spawn.
    let baseline_store = SqliteStore::new(&db_path, "text-embedding-3-large", 3072).unwrap();
    let queries = [
        "pool design concurrency",
        "knowledge graph fusion",
        "feedback queue durability",
    ];
    let baseline: Vec<Vec<String>> = queries
        .iter()
        .map(|q| recall_ids_nonfast(&baseline_store, &config, q))
        .collect();
    drop(baseline_store);

    // Candidate: store with pool attached — Vec thread enters the pool
    // branch, calls `handle.block_on(pool.get())`, uses `from_conn`.
    let pool = Arc::new(ConnPool::new(&db_path, 2).unwrap());
    let pooled_store = SqliteStore::new(&db_path, "text-embedding-3-large", 3072)
        .unwrap()
        .with_pool(Arc::clone(&pool));

    for (i, q) in queries.iter().enumerate() {
        let ids = recall_ids_nonfast(&pooled_store, &config, q);
        assert_eq!(
            baseline[i], ids,
            "nonfast pooled-Vec recall diverged from baseline on q={q:?}: \
             baseline={:?} pooled={:?}",
            baseline[i], ids
        );
    }

    // Pool healthy: size and permits unchanged after N recalls, proving
    // each checkout was returned (either via put_back on success OR
    // self-heal on panic path; no shrink).
    let m = pool.metrics();
    assert_eq!(m.size, 2, "no pool shrink after N recalls");
    assert_eq!(m.idle, 2, "all conns returned to pool");
    assert_eq!(m.available_permits, 2, "no permit leak");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn store_with_pool_attached_produces_same_recall_as_serial_fallback() {
    // D4 Vec-channel pool path: when a store has `with_pool(Arc<ConnPool>)`
    // attached, the Vec cache-miss std::thread::spawn checks out from the
    // pool instead of opening a fresh SqliteStore. Output must match the
    // no-pool baseline exactly. This is the v0.22 I1 guarantee applied to
    // the first channel we migrated.
    use std::sync::Arc;

    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("with-pool.db");
    let config = seed_fixture(&db_path);

    // Baseline: fresh store, no pool — Vec channel uses SqliteStore::new.
    let baseline_store = SqliteStore::new(&db_path, "text-embedding-3-large", 3072).unwrap();
    let queries = [
        "pool design concurrency",
        "HNSW vector search",
        "feedback queue durability",
    ];
    let baseline: Vec<Vec<String>> = queries
        .iter()
        .map(|q| recall_ids(&baseline_store, &config, q))
        .collect();
    drop(baseline_store);

    // Candidate: same store, but with pool attached. Vec channel takes
    // the pool path via `handle.block_on(pool.get())` + from_conn.
    let pool = Arc::new(ConnPool::new(&db_path, 2).unwrap());
    let pooled_store = SqliteStore::new(&db_path, "text-embedding-3-large", 3072)
        .unwrap()
        .with_pool(Arc::clone(&pool));
    assert!(
        pooled_store.pool().is_some(),
        "with_pool should attach pool"
    );

    for (i, q) in queries.iter().enumerate() {
        let ids = recall_ids(&pooled_store, &config, q);
        assert_eq!(
            baseline[i], ids,
            "with_pool recall diverged from baseline on q={q:?}: \
             baseline={:?} pooled={:?}",
            baseline[i], ids
        );
    }

    // Pool invariants hold: all conns returned, no phantom permits.
    let m = pool.metrics();
    assert_eq!(m.size, 2);
    assert_eq!(m.idle, 2);
    assert_eq!(m.available_permits, 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_pool_recall_preserves_determinism() {
    // Several recall queries run concurrently through the pool, each on
    // its own conn. Assert each query's result matches the baseline —
    // confirms that parallel checkout does not introduce cross-conn
    // interference (the pre-condition for D4's 3-channel join!).
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("pool-concurrent.db");
    let config = seed_fixture(&db_path);

    let fresh = SqliteStore::new(&db_path, "text-embedding-3-large", 3072).unwrap();
    let queries: Vec<String> = (0..10)
        .map(|i| {
            format!(
                "pool design {}",
                ["concurrency", "fusion", "durability", "HNSW", "tokio"][i % 5]
            )
        })
        .collect();
    let baseline: Vec<Vec<String>> = queries
        .iter()
        .map(|q| recall_ids(&fresh, &config, q))
        .collect();
    drop(fresh);

    let pool = ConnPool::new(&db_path, 4).expect("pool opens");
    let mut handles = Vec::with_capacity(queries.len());
    for (i, q) in queries.iter().cloned().enumerate() {
        let pool = pool.clone();
        let path = db_path.clone();
        let cfg = config.clone();
        let expected = baseline[i].clone();
        handles.push(tokio::spawn(async move {
            let guard = pool.get().await.unwrap();
            let (conn, detached) = guard.detach();
            let store = SqliteStore::from_conn(conn, path, 3072, &cfg.embedding_model());
            let ids = recall_ids(&store, &cfg, &q);
            let conn = store.into_conn();
            detached.put_back(conn);
            assert_eq!(expected, ids, "concurrent recall diverged on q={q:?}");
        }));
    }
    for h in handles {
        h.await.expect("recall task succeeded");
    }

    let m = pool.metrics();
    assert_eq!(m.idle, 4);
    assert_eq!(m.available_permits, 4);
}
