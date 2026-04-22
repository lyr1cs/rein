//! D3.5 — real-recall no-regression gate (I6 invariant).
//!
//! Validates that v0.22 D4's pool + internal channel parallelism does not
//! regress single-user recall p50 latency by more than 2% compared to the
//! pre-v0.22 serial fallback path.
//!
//! From spec `docs/superpowers/specs/2026-04-21-v0.22-async-hardening-design.md` §I6:
//!   "pool + tokio::join! 下，single-user p50 latency 回归 ≤ 2%。
//!    若违反：回退 P1 内部并行，仅保留 pool 用于 MCP request-level 吞吐。"
//!
//! Method:
//!   - Seed a 500-memory fixture (FTS5 channel does real work; vec/KG channels
//!     are cold since no embeddings/concepts are pre-populated — this is
//!     a conservative test since parallel wins are smaller on FTS-only load).
//!   - Scenario (a): REIN_ASYNC_P1=0 — forces pre-v0.22 serial path in open_store
//!   - Scenario (b): default — D4 pool + try_get + internal parallelism
//!   - Report p50/p95/p99; decide PASS/FAIL per I6.
//!
//! Run:  cargo run --example recall_regression --release

use std::path::Path;
use std::time::{Duration, Instant};

use chrono::Utc;
use rein::config::ReinConfig;
use rein::search::recall::recall_temporal_with_request_id;
use rein::store::tiering::MemoryTier;
use rein::store::SqliteStore;
use rein::types::{Importance, Memory, MemoryLayer, MemoryStatus, MemoryStore, Source};
use tempfile::TempDir;

const FIXTURE_MEMS: usize = 500;
const BENCH_WARMUP: usize = 20;
const BENCH_ITERS: usize = 200;

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
        created_at: Utc::now(),
        updated_at: Utc::now(),
        last_accessed: Utc::now(),
    }
}

fn seed_fixture(path: &Path) {
    let store = SqliteStore::new(path, "text-embedding-3-large", 3072).unwrap();
    let topics = [
        "memory_architecture",
        "pool_design",
        "async_runtime",
        "search_fusion",
        "feedback_loop",
        "knowledge_graph",
        "temporal_model",
        "adaptive_engine",
        "cluster_survival",
        "ranking_rerank",
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
        "Kaplan-Meier per-cluster survival curves replace fixed Ebbinghaus decay",
        "HDBSCAN with condensed tree and EOMBST extracts semantic clusters",
        "hot warm cold three-tier memory tiering uses t-digest quantile estimator",
        "maximal marginal relevance reranking balances diversity against relevance",
        "canonical-first read model collapses evidence behind a single anchor",
        "auto_link detects related memories on store using FTS and jaccard overlap",
        "HTTP REST endpoints bridge Neural Wiki GUI to the MCP inventory adapters",
    ];
    let mem_per_topic = FIXTURE_MEMS / topics.len();
    for (i, topic) in topics.iter().enumerate() {
        for j in 0..mem_per_topic {
            let idx = (i * mem_per_topic + j) % phrases.len();
            let memory = build_memory(
                topic,
                &format!("{topic}-{j}"),
                &format!(
                    "{} item {j} under {topic} extended-id {} category-token {} \
                     phrase-ref {}",
                    phrases[idx],
                    i * mem_per_topic + j,
                    phrases[(idx + 1) % phrases.len()],
                    phrases[(idx + 2) % phrases.len()],
                ),
            );
            store.store(memory).unwrap();
        }
    }
}

fn queries() -> Vec<&'static str> {
    vec![
        "memory architecture with SQLite WAL",
        "pool checkout FULL_MUTEX connection",
        "recall FTS vector knowledge graph fusion",
        "tokio spawn_blocking reactor rusqlite",
        "feedback queue durable events",
        "HNSW vector index milliseconds",
        "Tantivy BM25 snapshot readers",
        "adaptive alpha counterfactual replay",
        "Kaplan-Meier survival Ebbinghaus decay",
        "HDBSCAN cluster condensed tree EOMBST",
        "hot warm cold tiering t-digest",
        "maximal marginal relevance diversity",
        "canonical read model evidence",
        "auto_link jaccard FTS overlap",
        "REST endpoints Neural Wiki MCP",
        "temporal model clustering ranking",
        "adaptive engine learning rerank",
        "knowledge graph BFS episodes links",
        "feedback event sourcing queue",
        "search fusion RRF CC channels",
    ]
}

fn run_recall_loop(config: &ReinConfig, queries: &[&str]) -> Vec<Duration> {
    let store = config.open_store().expect("open_store");

    // Warmup: prime FTS readers, any lazy indexes, and Gemini auth paths if present.
    for i in 0..BENCH_WARMUP {
        let _ = recall_temporal_with_request_id(
            &store,
            config,
            queries[i % queries.len()],
            None,
            None,
            10,
            None,
            None,
            Some(false),
            /* fast */ true,
            None,
        );
    }

    let mut times = Vec::with_capacity(BENCH_ITERS);
    for i in 0..BENCH_ITERS {
        let q = queries[i % queries.len()];
        let t0 = Instant::now();
        let _ = recall_temporal_with_request_id(
            &store,
            config,
            q,
            None,
            None,
            10,
            None,
            None,
            Some(false),
            /* fast */ true,
            None,
        )
        .expect("recall");
        times.push(t0.elapsed());
    }
    times
}

fn report(label: &str, times: &mut [Duration]) -> (Duration, Duration) {
    times.sort();
    let n = times.len();
    let p50 = times[n / 2];
    let p95 = times[(n as f64 * 0.95) as usize];
    let p99 = times[(n as f64 * 0.99) as usize];
    let mean = times.iter().sum::<Duration>() / n as u32;
    let max = times.last().copied().unwrap_or_default();
    eprintln!(
        "  {label:<32} n={n:<4} mean={mean:>8.2?}  p50={p50:>8.2?}  p95={p95:>8.2?}  p99={p99:>8.2?}  max={max:>8.2?}"
    );
    (p50, p95)
}

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() {
    eprintln!("D3.5 — real-recall no-regression gate (I6 invariant)");
    eprintln!(
        "Config: {FIXTURE_MEMS} memories, {BENCH_ITERS} samples, fast=true (local 3-channel, no remote)"
    );

    let tmp = TempDir::new().expect("tempdir");
    let db_path = tmp.path().join("d3_5.db");
    // Pin the rein DB to our fixture.
    std::env::set_var("REIN_DB", &db_path);

    eprintln!("[fixture] seeding {FIXTURE_MEMS} memories ...");
    let t_seed = Instant::now();
    seed_fixture(&db_path);
    eprintln!("[fixture] seeded in {:.2?}", t_seed.elapsed());

    let config = ReinConfig::default();

    // Scenario A: REIN_ASYNC_P1=0 → open_store returns a pool-less SqliteStore
    // → recall.rs falls back to pre-v0.22 path (fresh SqliteStore::new per channel).
    eprintln!("\n[A] serial (REIN_ASYNC_P1=0, pre-v0.22 fallback path)");
    std::env::set_var("REIN_ASYNC_P1", "0");
    let mut times_a = run_recall_loop(&config, &queries());
    let (p50_a, p95_a) = report("(a) serial", &mut times_a);

    // Scenario B: default → open_store attaches a pool, recall.rs uses
    // `try_get` + internal per-channel parallelism (D4 production path).
    eprintln!("\n[B] parallel (pool + try_get + internal parallelism, D4 prod)");
    std::env::remove_var("REIN_ASYNC_P1");
    let mut times_b = run_recall_loop(&config, &queries());
    let (p50_b, p95_b) = report("(b) parallel", &mut times_b);

    // I6 gate: regression ≤ 2%.
    let ratio = p50_b.as_secs_f64() / p50_a.as_secs_f64().max(1e-9);
    let regression_pct = (ratio - 1.0) * 100.0;
    let p95_ratio = p95_b.as_secs_f64() / p95_a.as_secs_f64().max(1e-9);
    let p95_regression_pct = (p95_ratio - 1.0) * 100.0;

    eprintln!("\n=== I6 Gate ===");
    eprintln!("  p50 serial (a):   {p50_a:?}");
    eprintln!("  p50 parallel (b): {p50_b:?}  ({ratio:.3}x, regression {regression_pct:+.2}%)");
    eprintln!("  p95 serial (a):   {p95_a:?}");
    eprintln!(
        "  p95 parallel (b): {p95_b:?}  ({p95_ratio:.3}x, regression {p95_regression_pct:+.2}%)"
    );
    eprintln!();
    if regression_pct <= 2.0 {
        eprintln!("  → PASS — D4 pool + internal parallelism preserves single-user latency");
    } else if regression_pct <= 10.0 {
        eprintln!("  → MARGINAL ({regression_pct:+.2}%) — consider raising gate to 10% if workload mostly FTS-only");
    } else {
        eprintln!("  → FAIL — disable internal tokio::join! in recall.rs, keep pool for MCP-level throughput only");
    }
}
