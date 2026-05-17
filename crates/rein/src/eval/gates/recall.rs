//! v0.32 recall gate — runs recall against a fixture corpus and scores hit-rate.
//!
//! Hermetic by construction: seeds an isolated `SqliteStore::in_memory()` per
//! fixture from the on-disk JSON corpus, then runs `recall_fast` against it.
//! `recall_fast` skips Supermemory + query expansion + LLM reranker, and with
//! a `:memory:` store there is no HNSW side-index, so we never need to make a
//! live embedding-API call. The cache-miss branch in `recall_temporal` lands
//! on `VecSearchState::Skip` (recall.rs:680-685) and scoring falls back to
//! pure FTS5 BM25 + KG signals — exactly the deterministic mix we want for a
//! reproducible gate.
//!
//! `MockEmbedder` therefore is not used here; the `test-support` feature is
//! not required to run the gate. (The spec text mentions MockEmbedder as a
//! belt-and-braces option; with the `:memory:` + `fast=true` combination the
//! belt alone is enough.)

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use chrono::Utc;
use serde::Deserialize;

use crate::config::ReinConfig;
use crate::eval::gates::{
    FixtureResult, Gate, GateScorecard, ScorecardKind, SCORECARD_SCHEMA_VERSION,
};
use crate::store::SqliteStore;
use crate::types::{
    Importance, Memory, MemoryLayer, MemoryStatus, MemoryStore, MemoryTier, Source,
};

/// How many top results to inspect when computing hit. Keeps the gate
/// sensitive to ranking without being strict-by-1 brittle (spec §"Recall
/// gate" step 4).
const TOP_K_HIT_WINDOW: usize = 3;

/// Per-fixture recall limit passed to `recall_fast`. Picked to match the
/// spec's "10" — wide enough that ties don't push the target past top-3.
const FIXTURE_RECALL_LIMIT: usize = 10;

pub struct RecallGate;

#[derive(Debug, Deserialize)]
struct RecallFixture {
    id: String,
    query: String,
    #[serde(default)]
    topic: Option<String>,
    #[serde(default)]
    keyword: Option<String>,
    seed_memories: Vec<SeedMemory>,
    expected_memory_ids: Vec<String>,
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
    // optional embedding seed; if absent, gate uses MockEmbedder via test-support feature
    // (currently unused — see module-level doc).
}

impl Gate for RecallGate {
    fn name(&self) -> &'static str {
        "recall"
    }

    fn run(&self, _store: &SqliteStore, _config: &ReinConfig) -> Result<GateScorecard> {
        // The recall gate seeds its OWN ephemeral :memory: store from the fixture's
        // `seed_memories` (rather than running against the caller's `store`). This
        // makes results reproducible across machines & DB states. The signature
        // takes &SqliteStore + &ReinConfig for Gate-trait uniformity; we ignore
        // them here.
        let fixtures = load_recall_fixtures()?;
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
            gate_name: "recall".to_string(),
            kind: ScorecardKind::Run,
            created_at: Utc::now().timestamp(),
            rein_version: env!("CARGO_PKG_VERSION").to_string(),
            fixture_count: per_fixture.len(),
            score,
            per_fixture,
        })
    }
}

/// Path to the recall fixture corpus.
///
/// v0.32 post-ship privacy fix: resolved at RUNTIME rather than via
/// `env!("CARGO_MANIFEST_DIR")`.  The compile-time `env!()` form bakes
/// the absolute build-time path into the released binary (Rust's
/// `--remap-path-prefix` does not apply to `env!()` expansions, only
/// to `file!()` and debuginfo), so `strings rein-darwin-arm64 | grep
/// '/Users/'` exposes the builder's `$HOME`.
///
/// Runtime resolution order:
///
/// 1. `CARGO_MANIFEST_DIR` env var (set by cargo for `cargo test` and
///    `cargo run --bin …` — works without any embedded literal).
/// 2. CWD-relative `crates/rein/tests/fixtures/eval_gates/recall`
///    (operator running rein-eval from the source-repo root).
///
/// If neither yields a directory containing fixtures, the
/// load-empty guard in `load_recall_fixtures` produces the operator-
/// safe "no fixtures available" error.
fn fixture_dir() -> PathBuf {
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        return PathBuf::from(manifest)
            .join("tests")
            .join("fixtures")
            .join("eval_gates")
            .join("recall");
    }
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join("crates")
        .join("rein")
        .join("tests")
        .join("fixtures")
        .join("eval_gates")
        .join("recall")
}

/// Load all `case_*.json` fixtures from `tests/fixtures/eval_gates/recall/`.
/// Path is resolved relative to `CARGO_MANIFEST_DIR` so this works in both
/// `cargo test` (running from the crate root) and `cargo run --bin rein-eval`.
///
/// v0.32 R1 P2-#1: empty corpus is an ERROR, not an empty-success.  Without
/// this guard, an installed binary running far from the source tree (where
/// `CARGO_MANIFEST_DIR/tests/fixtures/...` doesn't exist) silently produces
/// a 0-fixture scorecard that gets classified as `NoData` / stub-like and
/// the recall gate effectively disables itself.  Failing here forces the
/// caller (rein-eval / trust_measurement / doctor) to surface "no fixtures
/// available" explicitly.
fn load_recall_fixtures() -> Result<Vec<RecallFixture>> {
    let dir = fixture_dir();
    // v0.32 R4 P3: use `read_dir` + filename filter instead of
    // `glob::glob(format!("{dir}/case_*.json"))`.  Interpolating
    // `dir.display()` into a glob pattern makes any metacharacters in
    // the checkout path (`[`, `?`, `*`) part of the glob syntax, so a
    // path like `/Users/me/test[1]/...` would not match its own
    // contents and the gate would mis-report "no fixtures".  `read_dir`
    // + a literal extension/prefix check is metacharacter-safe.
    let entries =
        std::fs::read_dir(&dir).with_context(|| format!("read fixture dir {}", dir.display()))?;
    let mut paths: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            let Some(name) = p.file_name().and_then(|n| n.to_str()) else {
                return false;
            };
            // Match the original `case_*.json` shape literally.
            name.starts_with("case_") && name.ends_with(".json") && p.is_file()
        })
        .collect();
    paths.sort();

    if paths.is_empty() {
        bail!(
            "recall gate found no fixtures matching `case_*.json` in {}. \
             This is fatal — a 0-fixture scorecard silently disables the gate. \
             Fixtures live in the crate tree at \
             `crates/rein/tests/fixtures/eval_gates/recall/case_*.json`; \
             an installed release binary running outside the source tree \
             will hit this path.  Build & run rein-eval from the source \
             repo (e.g. `cargo run -p rein --bin rein-eval`).",
            dir.display(),
        );
    }

    let mut fixtures = Vec::with_capacity(paths.len());
    for path in paths {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("read recall fixture {}", path.display()))?;
        let fx: RecallFixture = serde_json::from_str(&text)
            .with_context(|| format!("parse recall fixture {}", path.display()))?;
        fixtures.push(fx);
    }

    // Deterministic ordering for stable scorecard `per_fixture` serialization
    // and for the harness's McNemar pairing on `fixture_id` intersection.
    fixtures.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(fixtures)
}

/// Build a `Memory` row from a `SeedMemory`. Mirrors `make_memory` in
/// `tests/test_phase15_2.rs` for shape, but keeps the caller's `id` so
/// fixtures can use stable identifiers like "m1" / "m2" (the `MemoryStore::
/// store` impl preserves non-empty IDs — see `store/sqlite.rs:885-890`).
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
        // `MemoryStore::store` unconditionally overwrites these three (see
        // sqlite.rs:891-894), so the values here are effectively placeholders;
        // we still set them to `now` so the struct is self-consistent if a
        // caller ever bypasses `store()`.
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

/// Run one recall fixture against an isolated `:memory:` store.
///
/// Returns `Ok(true)` if any of `fx.expected_memory_ids` appears in the top
/// `TOP_K_HIT_WINDOW` results, else `Ok(false)`. Hard errors (store init,
/// recall failure) propagate as `Err` so the harness can distinguish a
/// gate-infrastructure breakage from a quality regression.
fn run_one_fixture(fx: &RecallFixture) -> Result<bool> {
    // Step 1: Isolated in-memory store. `SqliteStore::in_memory()` initializes
    // with 3072 dims (matches the in-tree default for `text-embedding-3-small`).
    // No tempdir is needed — :memory: gives us a fresh schema per fixture and
    // skips all HNSW side-index work (the only path that would require a real
    // embedder). See module-level doc for the no-MockEmbedder rationale.
    let store = SqliteStore::in_memory()
        .with_context(|| format!("init in-memory store for fixture {}", fx.id))?;

    // Step 2: Seed the store. `embedding: None` means `update_hnsw` is a no-op
    // for :memory:, and `update_tantivy` runs against the FTS5 mirror without
    // needing a real Tantivy directory. Per-fixture IDs (m1/m2/…) are preserved
    // verbatim by `MemoryStore::store` because they're non-empty.
    for seed in &fx.seed_memories {
        let memory = seed_to_memory(seed);
        store
            .store(memory)
            .with_context(|| format!("seed memory {} for fixture {}", seed.id, fx.id))?;
    }

    // Step 3: Recall with default config. `recall_fast` keeps the gate hermetic
    // (no Supermemory / expansion / LLM rerank) and with a :memory: store the
    // vector channel lands in `VecSearchState::Skip` — FTS5 + KG only.
    let config = ReinConfig::default();
    let results = crate::search::recall::recall_fast(
        &store,
        &config,
        &fx.query,
        fx.topic.as_deref(),
        fx.keyword.as_deref(),
        FIXTURE_RECALL_LIMIT,
    )
    .with_context(|| format!("recall_fast failed for fixture {}", fx.id))?;

    // Step 4: Hit = any expected id appears in the top-K window.
    let hit = results
        .iter()
        .take(TOP_K_HIT_WINDOW)
        .any(|r| fx.expected_memory_ids.iter().any(|id| id == &r.memory.id));
    Ok(hit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ReinConfig;
    use crate::store::SqliteStore;

    #[test]
    fn recall_gate_loads_fixtures() {
        let fixtures = load_recall_fixtures().expect("fixture dir must be readable");
        assert!(
            !fixtures.is_empty(),
            "recall fixture corpus is empty — gate would produce score=0"
        );
        // Sorted by id => first fixture id should sort <= last.
        let first = &fixtures.first().unwrap().id;
        let last = &fixtures.last().unwrap().id;
        assert!(first.as_str() <= last.as_str(), "fixtures not sorted by id");

        // Every fixture must have at least one expected id present in its
        // seed memories (otherwise it's impossible to ever hit).
        for fx in &fixtures {
            assert!(
                !fx.expected_memory_ids.is_empty(),
                "fixture {} has no expected_memory_ids",
                fx.id
            );
            let seed_ids: std::collections::HashSet<_> =
                fx.seed_memories.iter().map(|s| s.id.as_str()).collect();
            for expected in &fx.expected_memory_ids {
                assert!(
                    seed_ids.contains(expected.as_str()),
                    "fixture {} expected id {} not in seed memories",
                    fx.id,
                    expected
                );
            }
            assert!(
                !fx.seed_memories.is_empty(),
                "fixture {} has no seed memories",
                fx.id
            );
            assert!(
                fx.seed_memories.len() <= 5,
                "fixture {} has {} seed memories (spec caps at 5)",
                fx.id,
                fx.seed_memories.len()
            );
        }
    }

    #[test]
    fn recall_gate_run_returns_scorecard_with_all_fixtures() {
        let store = SqliteStore::in_memory().unwrap();
        let config = ReinConfig::default();
        let scorecard = RecallGate.run(&store, &config).unwrap();
        let fixtures = load_recall_fixtures().unwrap();

        assert_eq!(scorecard.gate_name, "recall");
        assert_eq!(scorecard.schema_version, SCORECARD_SCHEMA_VERSION);
        assert_eq!(scorecard.kind, ScorecardKind::Run);
        assert_eq!(scorecard.fixture_count, fixtures.len());
        assert_eq!(scorecard.per_fixture.len(), fixtures.len());
        assert!(scorecard.score >= 0.0 && scorecard.score <= 1.0);
        // Output ordering must mirror sorted fixture order so paired McNemar
        // pairing in the harness is stable across runs.
        let mut sorted = scorecard
            .per_fixture
            .iter()
            .map(|f| f.fixture_id.clone())
            .collect::<Vec<_>>();
        let original = sorted.clone();
        sorted.sort();
        assert_eq!(
            sorted, original,
            "per_fixture must be emitted in sorted fixture_id order"
        );
    }

    #[test]
    fn recall_gate_keyword_query_finds_exact_match() {
        // Direct invocation of `run_one_fixture` with a synthetic fixture —
        // avoids dependence on the on-disk corpus for this assertion.
        let fx = RecallFixture {
            id: "synthetic_keyword".to_string(),
            query: "SqliteStore".to_string(),
            topic: None,
            keyword: Some("SqliteStore".to_string()),
            seed_memories: vec![
                SeedMemory {
                    id: "m1".to_string(),
                    content: "SqliteStore implements the storage layer for rein with per-request connections and a Tantivy singleton cache.".to_string(),
                    topic: Some("rein-internals".to_string()),
                    keywords: vec!["SqliteStore".to_string(), "storage".to_string(), "tantivy".to_string()],
                    summary: None,
                },
                SeedMemory {
                    id: "m2".to_string(),
                    content: "Personal note about adopting a cat from the shelter last weekend.".to_string(),
                    topic: Some("personal".to_string()),
                    keywords: vec!["cat".to_string(), "shelter".to_string()],
                    summary: None,
                },
            ],
            expected_memory_ids: vec!["m1".to_string()],
        };
        let hit = run_one_fixture(&fx).unwrap();
        assert!(
            hit,
            "exact-keyword query 'SqliteStore' should recall m1 over m2"
        );
    }

    #[test]
    fn recall_gate_handles_empty_seed_memories_gracefully() {
        // A zero-seed fixture should never hit (there's nothing to recall),
        // but `run_one_fixture` MUST NOT error — the store should accept the
        // empty seed list and recall should return an empty result vec.
        let fx = RecallFixture {
            id: "empty".to_string(),
            query: "anything".to_string(),
            topic: None,
            keyword: None,
            seed_memories: vec![],
            expected_memory_ids: vec!["does-not-exist".to_string()],
        };
        let hit = run_one_fixture(&fx).unwrap();
        assert!(!hit, "empty seed should never produce a hit");
    }

    #[test]
    fn recall_gate_fixture_distribution_matches_spec() {
        // Spec §"Fixture coverage required": episodic ×4, temporal ×3,
        // preference ×3, exact_keyword ×4, semantic ×3, exploratory ×3.
        // The fixture id naming convention encodes the query type in the
        // prefix, so we can count by parsing.
        let fixtures = load_recall_fixtures().unwrap();
        let mut counts = std::collections::HashMap::<&str, usize>::new();
        for fx in &fixtures {
            // Identify the prefix family. Order matters — exact_keyword has a
            // two-token prefix; check longest first.
            let family = if fx.id.starts_with("exact_keyword_") {
                "exact_keyword"
            } else if fx.id.starts_with("episodic_") {
                "episodic"
            } else if fx.id.starts_with("temporal_") {
                "temporal"
            } else if fx.id.starts_with("preference_") {
                "preference"
            } else if fx.id.starts_with("semantic_") {
                "semantic"
            } else if fx.id.starts_with("exploratory_") {
                "exploratory"
            } else {
                panic!(
                    "fixture {} does not match any spec query-type prefix",
                    fx.id
                );
            };
            *counts.entry(family).or_insert(0) += 1;
        }
        assert_eq!(counts.get("episodic").copied().unwrap_or(0), 4);
        assert_eq!(counts.get("temporal").copied().unwrap_or(0), 3);
        assert_eq!(counts.get("preference").copied().unwrap_or(0), 3);
        assert_eq!(counts.get("exact_keyword").copied().unwrap_or(0), 4);
        assert_eq!(counts.get("semantic").copied().unwrap_or(0), 3);
        assert_eq!(counts.get("exploratory").copied().unwrap_or(0), 3);
        assert_eq!(fixtures.len(), 20);
    }
}
