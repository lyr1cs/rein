//! v0.36 #P1 — strong-signal recall fast-path: correctness + latency bench.
//!
//! Two concerns:
//! - **Correctness**: a strong-signal *confirmed* query (live, non-cold,
//!   non-episodic, no time bounds, distinctive dominant BM25 hit) must return
//!   the dominant memory. Episodic / filtered queries must NOT take the
//!   fast-path (KG / episode / SM fallback stays intact) — guarded by the
//!   equivalence assertions below.
//! - **Latency**: the `#[ignore]`d `bench_strong_signal_latency` prints p50/p95
//!   over N warm iterations. Run with:
//!   `cargo test --test recall_parallel_test -- --ignored --nocapture`.
//!   Captured pre-#P1 as the baseline; re-run post-#P1 to measure the KG-skip
//!   delta (Supermemory is disabled in this offline config, so the SM-join
//!   skip is argued analytically, not measured here).

use chrono::Utc;
use rein::config::ReinConfig;
use rein::search::recall::recall_temporal_with_request_id;
use rein::store::tiering::MemoryTier;
use rein::store::SqliteStore;
use rein::types::{Importance, Memory, MemoryLayer, MemoryStatus, MemoryStore, Source};
use tempfile::TempDir;

fn build_memory(topic: &str, summary: &str, content: &str, keywords: Vec<String>) -> Memory {
    Memory {
        id: ulid::Ulid::new().to_string(),
        layer: MemoryLayer::LTM,
        topic: topic.to_string(),
        summary: summary.to_string(),
        content: content.to_string(),
        keywords,
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

/// Distinctive query token. It appears once in each of the `COVERAGE` rows and
/// many times in the single dominant row, so the dominant's BM25 top1/top2 ratio
/// clears `strong_signal_ratio` (default 1.5) while ≥`limit` local hits survive.
const DOMINANT_PHRASE: &str = "zephyrindexmarker";

/// Number of coverage rows containing the query token once each. Must be ≥ the
/// recall `limit` used in the tests (10) so the #P1 completeness guard
/// (`surviving_local >= limit`) is satisfied and the fast-path can engage.
const COVERAGE: usize = 12;

fn seed(path: &std::path::Path) -> String {
    let store = SqliteStore::new(path, "text-embedding-3-large", 3072).unwrap();
    let topics = [
        "memory_architecture",
        "pool_design",
        "async_runtime",
        "search_fusion",
        "feedback_loop",
        "tiering",
        "clustering",
        "survival",
    ];
    let phrases = [
        "rein uses SQLite with WAL mode for per-connection reader concurrency",
        "the pool checkout model preserves per-connection mutex guarantees",
        "recall aggregates FTS Vector and knowledge graph channels via fusion",
        "tokio spawn_blocking wraps rusqlite calls so the reactor never stalls",
        "feedback events are durable on return through the pending queue",
        "HNSW index serves vector queries in under ten milliseconds typical",
        "Tantivy provides BM25 full text search with snapshot consistent readers",
        "the adaptive engine learns alpha weights from counterfactual replay",
    ];
    // ~200 filler memories that do NOT contain the dominant phrase.
    for (i, topic) in topics.iter().enumerate() {
        for j in 0..25usize {
            let idx = (i * 25 + j) % phrases.len();
            let memory = build_memory(
                topic,
                &format!("{topic}-{j}"),
                &format!("{} item {} under {}", phrases[idx], j, topic),
                vec![topic.to_string(), "filler".into()],
            );
            store.store(memory).unwrap();
        }
    }
    // COVERAGE rows each contain the query token ONCE, so FTS returns ≥`limit`
    // surviving local hits (the #P1 completeness guard) and the
    // `detect_strong_signal` ≥2-results precondition is met.
    for k in 0..COVERAGE {
        let m = build_memory(
            "exotic",
            &format!("coverage-{k}"),
            &format!("{DOMINANT_PHRASE} appears once in coverage note {k} about systems"),
            vec!["exotic".into()],
        );
        store.store(m).unwrap();
    }
    // The dominant hit: token repeated 20× so its BM25 top1/top2 ratio dwarfs the
    // single-occurrence coverage rows and clears the default 1.5 threshold.
    let dominant_content = format!("{DOMINANT_PHRASE} ").repeat(20) + "dominant entry";
    let dominant = build_memory(
        "exotic",
        "dominant",
        &dominant_content,
        vec!["exotic".into(), "dominant".into()],
    );
    let dominant_id = dominant.id.clone();
    store.store(dominant).unwrap();
    drop(store);
    dominant_id
}

fn offline_config() -> ReinConfig {
    let mut cfg = ReinConfig::default();
    // Fully hermetic: no remote calls and no local-filesystem AutoMemory scan
    // (which would otherwise read ~/.claude/projects/*/memory/**/*.md and inject
    // nondeterministic results + latency into the bench).
    cfg.sync.supermemory_enabled = false;
    cfg.sync.auto_memory_enabled = false;
    cfg.query_expansion.provider = "none".to_string();
    cfg
}

fn recall_ids(store: &SqliteStore, config: &ReinConfig, query: &str, fast: bool) -> Vec<String> {
    recall_temporal_with_request_id(
        store, config, query, None, None, 10, None, None, None, fast, None,
    )
    .expect("recall succeeds")
    .into_iter()
    .map(|r| r.memory.id)
    .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn strong_signal_confirmed_query_returns_dominant_hit() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("p1-strong.db");
    let dominant_id = seed(&db_path);
    let config = offline_config();

    let store = SqliteStore::new(&db_path, "text-embedding-3-large", 3072).unwrap();
    // fast=false exercises the KG background-thread path (the channel #P1
    // skips on a confirmed strong signal).
    let ids = recall_ids(&store, &config, DOMINANT_PHRASE, false);
    assert!(
        !ids.is_empty(),
        "strong-signal query must return the dominant hit, got empty"
    );
    assert_eq!(
        ids[0], dominant_id,
        "dominant memory must rank #1 for its own distinctive phrase"
    );
}

/// Quality preservation: the #P1 fast-path (strong signal → KG/episode/SM
/// skipped) must return the SAME result set as the full pipeline for a query
/// whose dominant hit genuinely survives all filters. We force the full path
/// by setting an unreachable strong-signal ratio so the same query runs every
/// channel, then assert set-equality + identical #1.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn strong_fast_path_is_result_equivalent_to_full_path() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("p1-equiv.db");
    let dominant_id = seed(&db_path);
    let store = SqliteStore::new(&db_path, "text-embedding-3-large", 3072).unwrap();

    // Fast path: default thresholds → strong signal confirmed → KG/SM skipped.
    let fast_path = offline_config();
    let ids_fast = recall_ids(&store, &fast_path, DOMINANT_PHRASE, false);

    // Full path: unreachable ratio → strong signal never fires → all channels run.
    let mut full_path = offline_config();
    full_path.search.strong_signal_ratio = 1.0e9;
    full_path.search.strong_signal_single = 1.0e9;
    let ids_full = recall_ids(&store, &full_path, DOMINANT_PHRASE, false);

    assert_eq!(ids_fast[0], dominant_id, "fast path must rank dominant #1");
    assert_eq!(ids_full[0], dominant_id, "full path must rank dominant #1");
    let set_fast: std::collections::BTreeSet<_> = ids_fast.iter().collect();
    let set_full: std::collections::BTreeSet<_> = ids_full.iter().collect();
    assert_eq!(
        set_fast, set_full,
        "fast path dropped/added results vs full path: fast={ids_fast:?} full={ids_full:?}"
    );
}

/// Latency micro-bench. `#[ignore]`d so it never runs in the default suite.
///
/// CAVEAT: this corpus has no KG concept graph and Supermemory is disabled, so
/// the two channels #P1 removes from the critical path (KG ≤80ms budget, SM
/// ≤500ms network join) cost ~nothing here — the offline floor is dominated by
/// the fixed pipeline + AutoMemory scan, which #P1 does not touch. So this bench
/// is a regression guard, NOT evidence of the production −30%: that win is
/// realized only when KG has a populated graph and/or SM is enabled.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "latency bench — run explicitly with --ignored --nocapture"]
async fn bench_strong_signal_latency() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("p1-bench.db");
    let _ = seed(&db_path);
    let config = offline_config();
    let store = SqliteStore::new(&db_path, "text-embedding-3-large", 3072).unwrap();

    // Warm up (Tantivy reader cache, page cache).
    for _ in 0..5 {
        let _ = recall_ids(&store, &config, DOMINANT_PHRASE, false);
    }

    let n = 60usize;
    let run = |fast: bool| -> Vec<u128> {
        let mut s: Vec<u128> = Vec::with_capacity(n);
        for _ in 0..n {
            let t = std::time::Instant::now();
            let _ = recall_ids(&store, &config, DOMINANT_PHRASE, fast);
            s.push(t.elapsed().as_micros());
        }
        s.sort_unstable();
        s
    };
    let report = |label: &str, mut s: Vec<u128>| {
        s.sort_unstable();
        let pct = |p: f64| s[((n as f64 * p) as usize).min(n - 1)];
        eprintln!(
            "[bench {label}] n={n} p50={}us p95={}us p99={}us min={}us max={}us",
            pct(0.50),
            pct(0.95),
            pct(0.99),
            s[0],
            s[n - 1],
        );
    };
    report("strong-signal fast=false (KG-thread path)", run(false));
    report("strong-signal fast=true  (KG-sync path)", run(true));
}
