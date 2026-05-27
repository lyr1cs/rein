//! v0.33 latency gate — recall_fast wall-time against a per-fixture budget.
//! Regression tripwire, not an absolute SLA (machine-dependent).
//!
//! Hermetic by construction: seeds an isolated `SqliteStore::in_memory()` per
//! fixture from the on-disk JSON corpus, then times `recall_fast` against it.
//! With a `:memory:` store there is no HNSW side-index and `recall_fast` skips
//! Supermemory + query expansion + LLM reranker, so the path is pure
//! FTS5 BM25 + KG signals — no live embedding-API call, no network.  The
//! measurement is therefore reproducible in *shape* (which path runs) even if
//! the wall-time itself varies by machine; the gate exists to catch a
//! regression where recall_fast gets dramatically slower, not to assert an
//! absolute SLA.
//!
//! Timing protocol per fixture: 1 warmup call (discarded — primes the
//! connection / FTS5 caches) then 3 timed calls; we keep the MINIMUM elapsed.
//! Latency is one-sided (a scheduler preemption / allocator hiccup can only
//! make a sample slower, never faster), so the minimum is the cleanest
//! estimate of the true cost.  `hit = min_ms < budget_ms`.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use chrono::Utc;
use serde::Deserialize;

use crate::config::ReinConfig;
use crate::eval::gates::{
    fixture_corpus_fingerprint, FixtureResult, Gate, GateScorecard, ScorecardKind,
    SCORECARD_SCHEMA_VERSION,
};
use crate::store::SqliteStore;
use crate::types::{
    Importance, Memory, MemoryLayer, MemoryStatus, MemoryStore, MemoryTier, Source,
};

/// Per-fixture recall limit passed to `recall_fast`. Matches the recall gate's
/// "10" — wide enough that the timed path exercises real fusion work.
const FIXTURE_RECALL_LIMIT: usize = 10;

/// Number of timed measurements per fixture (after one discarded warmup).
const TIMED_RUNS: usize = 3;

pub struct LatencyGate;

#[derive(Debug, Deserialize)]
struct LatencyFixture {
    id: String,
    query: String,
    #[serde(default)]
    topic: Option<String>,
    #[serde(default)]
    keyword: Option<String>,
    budget_ms: u64,
    seed_memories: Vec<SeedMemory>,
}

#[derive(Debug, Deserialize)]
struct SeedMemory {
    id: String,
    content: String,
    #[serde(default)]
    topic: Option<String>,
    #[serde(default)]
    keywords: Vec<String>,
    #[serde(default)]
    summary: Option<String>,
}

impl Gate for LatencyGate {
    fn name(&self) -> &'static str {
        "latency"
    }

    fn run(&self, _store: &SqliteStore, _config: &ReinConfig) -> Result<GateScorecard> {
        // The latency gate seeds its OWN ephemeral :memory: store from each
        // fixture's `seed_memories` (rather than running against the caller's
        // `store`). This makes the measured path reproducible across machines &
        // DB states. The signature takes &SqliteStore + &ReinConfig for
        // Gate-trait uniformity; we ignore them here.
        let (fixtures, fixture_fingerprint) = load_latency_fixtures()?;
        let mut per_fixture = Vec::with_capacity(fixtures.len());

        for fx in &fixtures {
            let hit = run_one_fixture(fx)?;
            per_fixture.push(FixtureResult {
                fixture_id: fx.id.clone(),
                hit,
            });
        }

        let hits = per_fixture.iter().filter(|f| f.hit).count() as f64;
        let total = per_fixture.len() as f64;
        let score = if total > 0.0 { hits / total } else { 0.0 };

        Ok(GateScorecard {
            schema_version: SCORECARD_SCHEMA_VERSION,
            gate_name: "latency".to_string(),
            kind: ScorecardKind::Run,
            created_at: Utc::now().timestamp(),
            rein_version: env!("CARGO_PKG_VERSION").to_string(),
            build_fingerprint: env!("REIN_BUILD_FINGERPRINT").to_string(),
            // Runtime fingerprint of the corpus this run actually read — not a
            // build-time env value (codex v0.32.1 R3 P2).
            fixture_fingerprint,
            fixture_count: per_fixture.len(),
            score,
            per_fixture,
        })
    }
}

/// Path to the latency fixture corpus.
///
/// Resolved at RUNTIME rather than via `env!("CARGO_MANIFEST_DIR")` (v0.32
/// post-ship privacy fix): the compile-time `env!()` form bakes the absolute
/// build-time path into the released binary (Rust's `--remap-path-prefix`
/// does not apply to `env!()` expansions), so `strings rein | grep '/Users/'`
/// would expose the builder's `$HOME`.
///
/// Resolution order:
/// 1. `CARGO_MANIFEST_DIR` env var (set by cargo for `cargo test` and
///    `cargo run --bin …`).
/// 2. CWD-relative `crates/rein/tests/fixtures/eval_gates/latency`
///    (operator running rein-eval from the source-repo root).
fn fixture_dir() -> PathBuf {
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        return PathBuf::from(manifest)
            .join("tests")
            .join("fixtures")
            .join("eval_gates")
            .join("latency");
    }
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join("crates")
        .join("rein")
        .join("tests")
        .join("fixtures")
        .join("eval_gates")
        .join("latency")
}

/// Load all `case_*.json` latency fixtures; returns `(fixtures, fingerprint)`
/// where the fingerprint is computed over the bytes actually read.
fn load_latency_fixtures() -> Result<(Vec<LatencyFixture>, String)> {
    let dir = fixture_dir();
    let entries =
        std::fs::read_dir(&dir).with_context(|| format!("read fixture dir {}", dir.display()))?;
    let mut paths: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            let Some(name) = p.file_name().and_then(|n| n.to_str()) else {
                return false;
            };
            name.starts_with("case_") && name.ends_with(".json") && p.is_file()
        })
        .collect();
    paths.sort();

    if paths.is_empty() {
        bail!(
            "latency gate found no fixtures matching `case_*.json` in {}. \
             A 0-fixture scorecard silently disables the gate; fixtures live at \
             `crates/rein/tests/fixtures/eval_gates/latency/case_*.json`. \
             Run rein-eval from the source repo (`cargo run -p rein --bin rein-eval`).",
            dir.display(),
        );
    }

    let mut fixtures = Vec::with_capacity(paths.len());
    let mut corpus: Vec<(String, Vec<u8>)> = Vec::with_capacity(paths.len());
    for path in paths {
        let bytes = std::fs::read(&path)
            .with_context(|| format!("read latency fixture {}", path.display()))?;
        let fx: LatencyFixture = serde_json::from_slice(&bytes)
            .with_context(|| format!("parse latency fixture {}", path.display()))?;
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_string();
        corpus.push((name, bytes));
        fixtures.push(fx);
    }
    fixtures.sort_by(|a, b| a.id.cmp(&b.id));
    let fingerprint = fixture_corpus_fingerprint(corpus);
    Ok((fixtures, fingerprint))
}

/// Build a `Memory` row from a `SeedMemory`. Mirrors the recall gate's
/// `seed_to_memory` exactly so the two gates seed identical store rows.
fn seed_to_memory(seed: &SeedMemory) -> Memory {
    let now = Utc::now();
    let importance = Importance::Medium;
    let summary = seed
        .summary
        .clone()
        .unwrap_or_else(|| first_sentence(&seed.content));
    Memory {
        id: seed.id.clone(),
        layer: MemoryLayer::STM,
        topic: seed.topic.clone().unwrap_or_else(|| "default".to_string()),
        summary,
        content: seed.content.clone(),
        keywords: seed.keywords.clone(),
        importance,
        source: Source::Manual,
        strength: 1.0,
        decay_lambda: 0.06 * importance.decay_factor(),
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
        created_at: now,
        updated_at: now,
        last_accessed: now,
    }
}

fn first_sentence(content: &str) -> String {
    content
        .split_terminator(['.', '!', '?', '\n'])
        .next()
        .unwrap_or(content)
        .trim()
        .to_string()
}

/// Run one latency fixture against an isolated `:memory:` store.
///
/// Seeds the store, runs one discarded warmup `recall_fast`, then `TIMED_RUNS`
/// timed calls, keeping the minimum elapsed. Returns `Ok(true)` if the minimum
/// is under `fx.budget_ms`. Hard errors (store init / recall failure)
/// propagate as `Err` so the harness can distinguish a gate-infrastructure
/// breakage from a latency regression.
fn run_one_fixture(fx: &LatencyFixture) -> Result<bool> {
    let store = SqliteStore::in_memory()
        .with_context(|| format!("init in-memory store for fixture {}", fx.id))?;

    for seed in &fx.seed_memories {
        let memory = seed_to_memory(seed);
        store
            .store(memory)
            .with_context(|| format!("seed memory {} for fixture {}", seed.id, fx.id))?;
    }

    let config = ReinConfig::default();

    // Warmup — result discarded (primes connection / FTS5 caches so the
    // first cold call doesn't dominate the measurement).
    let _ = crate::search::recall::recall_fast(
        &store,
        &config,
        &fx.query,
        fx.topic.as_deref(),
        fx.keyword.as_deref(),
        FIXTURE_RECALL_LIMIT,
    )
    .with_context(|| format!("warmup recall_fast failed for fixture {}", fx.id))?;

    // Timed runs — keep the minimum (latency is one-sided noise).
    let mut min_elapsed: Option<std::time::Duration> = None;
    for _ in 0..TIMED_RUNS {
        let start = Instant::now();
        let _ = crate::search::recall::recall_fast(
            &store,
            &config,
            &fx.query,
            fx.topic.as_deref(),
            fx.keyword.as_deref(),
            FIXTURE_RECALL_LIMIT,
        )
        .with_context(|| format!("timed recall_fast failed for fixture {}", fx.id))?;
        let elapsed = start.elapsed();
        min_elapsed = Some(match min_elapsed {
            Some(prev) if prev <= elapsed => prev,
            _ => elapsed,
        });
    }

    // `min_elapsed` is always `Some` because TIMED_RUNS >= 1.
    let min_ms = min_elapsed
        .expect("at least one timed run")
        .as_secs_f64()
        * 1000.0;
    Ok(min_ms < (fx.budget_ms as f64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ReinConfig;
    use crate::store::SqliteStore;

    #[test]
    fn latency_gate_is_not_stub() {
        assert!(!LatencyGate.is_stub());
        assert_eq!(LatencyGate.name(), "latency");
    }

    #[test]
    fn latency_gate_loads_fixtures() {
        let (fixtures, fingerprint) =
            load_latency_fixtures().expect("latency fixture dir must be readable");
        assert!(!fixtures.is_empty(), "latency corpus is empty");
        assert_eq!(fingerprint.len(), 32, "fingerprint must be 32 hex chars");
        // Sorted by id => first <= last.
        let first = &fixtures.first().unwrap().id;
        let last = &fixtures.last().unwrap().id;
        assert!(first.as_str() <= last.as_str(), "fixtures not sorted by id");
        for fx in &fixtures {
            assert!(
                !fx.seed_memories.is_empty(),
                "fixture {} has no seed memories",
                fx.id
            );
            assert!(fx.budget_ms > 0, "fixture {} has a non-positive budget", fx.id);
        }
    }

    #[test]
    fn latency_gate_run_returns_scorecard() {
        let store = SqliteStore::in_memory().unwrap();
        let config = ReinConfig::default();
        let sc = LatencyGate.run(&store, &config).unwrap();
        let (fixtures, fingerprint) = load_latency_fixtures().unwrap();
        assert_eq!(sc.gate_name, "latency");
        assert_eq!(sc.schema_version, SCORECARD_SCHEMA_VERSION);
        assert_eq!(sc.kind, ScorecardKind::Run);
        assert_eq!(sc.fixture_count, fixtures.len());
        assert_eq!(sc.per_fixture.len(), fixtures.len());
        assert_eq!(sc.fixture_fingerprint, fingerprint);
        assert!(sc.score >= 0.0 && sc.score <= 1.0);
        // per_fixture emitted in sorted id order (stable McNemar pairing). We do
        // NOT assert all-hit — wall-time is machine-dependent.
        let ids: Vec<String> = sc.per_fixture.iter().map(|f| f.fixture_id.clone()).collect();
        let mut sorted = ids.clone();
        sorted.sort();
        assert_eq!(ids, sorted);
    }
}
