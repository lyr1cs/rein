//! v0.33 admission gate — gray-zone admit/merge/duplicate decision quality.
//! Hermetic + pure (score_candidate, no store, no config, no LLM).
//!
//! Each fixture carries a candidate (`topic` + `content`) and a small set of
//! `existing` memories.  We build a `Memory` from each existing entry, score
//! the candidate against it with `extract::dedup::score_candidate` (the same
//! lexical max(Jaccard, containment) the live `check_dedup` path uses), and
//! take the best (max) `final_score`.  The decision band is then derived from
//! that best similarity using the live `gray_zone_lower_bound` helper:
//!
//! * `best > MERGE_THRESHOLD` (0.70) → `duplicate` (live strict auto-merge)
//! * `best < gray_zone_lower(..)`     → `admit_new`   (≈ `best < 0.35`)
//! * otherwise (∈ [lower, 0.70])      → `gray_zone`   (LLM review band)
//!
//! `hit = decision == fx.expected_decision`.  Same inputs always produce the
//! same decision (no store / config / LLM), which is what a reproducible gate
//! needs.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use chrono::Utc;
use serde::Deserialize;

use crate::config::ReinConfig;
use crate::eval::gates::{
    fixture_corpus_fingerprint, FixtureResult, Gate, GateScorecard, ScorecardKind,
    SCORECARD_SCHEMA_VERSION,
};
use crate::extract::dedup::{gray_zone_lower_bound, score_candidate};
use crate::store::SqliteStore;
use crate::types::{Importance, Memory, MemoryLayer, MemoryStatus, MemoryTier, Source};

pub struct AdmissionGate;

#[derive(Debug, Deserialize)]
struct AdmissionFixture {
    id: String,
    topic: String,
    content: String,
    existing: Vec<ExistingMemory>,
    expected_decision: String,
}

#[derive(Debug, Deserialize)]
struct ExistingMemory {
    id: String,
    topic: String,
    content: String,
}

impl Gate for AdmissionGate {
    fn name(&self) -> &'static str {
        "admission"
    }

    fn run(&self, _store: &SqliteStore, _config: &ReinConfig) -> Result<GateScorecard> {
        let (fixtures, fixture_fingerprint) = load_admission_fixtures()?;
        let mut per_fixture = Vec::with_capacity(fixtures.len());
        for fx in &fixtures {
            per_fixture.push(FixtureResult {
                fixture_id: fx.id.clone(),
                hit: classify_one(fx),
            });
        }
        let hits = per_fixture.iter().filter(|f| f.hit).count() as f64;
        let total = per_fixture.len() as f64;
        let score = if total > 0.0 { hits / total } else { 0.0 };

        Ok(GateScorecard {
            schema_version: SCORECARD_SCHEMA_VERSION,
            gate_name: "admission".to_string(),
            kind: ScorecardKind::Run,
            created_at: Utc::now().timestamp(),
            rein_version: env!("CARGO_PKG_VERSION").to_string(),
            build_fingerprint: env!("REIN_BUILD_FINGERPRINT").to_string(),
            fixture_fingerprint,
            fixture_count: per_fixture.len(),
            score,
            per_fixture,
        })
    }
}

/// `hit` = the admission classifier agrees with the ground-truth label.
fn classify_one(fx: &AdmissionFixture) -> bool {
    let mut best = 0.0f32;
    for existing in &fx.existing {
        let mem = existing_to_memory(existing);
        // cluster_id = None here (matches `mem.cluster_id = None`), so
        // `cluster_match` is always false and best == lexical similarity
        // plus an optional +0.05 topic bump.  Fixtures deliberately use
        // non-overlapping topics so the bump never fires; best == lexical.
        let score = score_candidate(&fx.topic, &fx.content, &mem, None).final_score;
        if score > best {
            best = score;
        }
    }
    decide(best) == fx.expected_decision
}

/// Merge threshold mirroring the live `check_dedup` auto-merge bound
/// (`config.search.dedup_similarity`, default 0.70). At/above this the live
/// path auto-merges (→ duplicate); the `[gray_zone_lower_bound, MERGE_THRESHOLD)`
/// band routes to LLM review (→ gray_zone), NOT an automatic duplicate.
/// codex v0.33 R1 P2: an earlier 0.50 boundary mislabeled the 0.50–0.70 band
/// as `duplicate`, so a direct-merge→gray-zone regression in that band would
/// be scored against the wrong expected decision. Documented default; runtime
/// config calibration deferred (`docs/backlog/v0.33-eval-gate-calibration.md`).
const MERGE_THRESHOLD: f32 = 0.70;

/// Pure decision band — extracted so it is unit-testable without fixtures.
///
/// Three bands derived from the best candidate similarity, matching the live
/// `check_dedup` semantics:
/// * `best > MERGE_THRESHOLD` (0.70)           → "duplicate" (auto-merge)
/// * `best < gray_zone_lower_bound(best, true)` → "admit_new" (≈ `best < 0.35`)
/// * otherwise (∈ [lower, MERGE_THRESHOLD])    → "gray_zone" (LLM review)
///
/// The merge boundary is STRICT (`>`), mirroring `check_dedup`, which
/// auto-merges only when `best_sim > effective_threshold`; an exactly-threshold
/// pair falls through to GrayZone (codex v0.33 R2 P3).
fn decide(best: f32) -> &'static str {
    let lower = gray_zone_lower_bound(best, true);
    if best > MERGE_THRESHOLD {
        "duplicate"
    } else if best < lower {
        "admit_new"
    } else {
        "gray_zone"
    }
}

/// Build a `Memory` row from an `ExistingMemory`.  Mirrors `recall::
/// seed_to_memory` field-for-field; the input has no keywords / summary so
/// keywords default to empty and summary to the first sentence of content.
/// Only `id` / `topic` / `content` matter for `score_candidate`; everything
/// else is a benign self-consistent default.  `cluster_id` is `None` so
/// `cluster_match` never fires.
fn existing_to_memory(existing: &ExistingMemory) -> Memory {
    let now = Utc::now();
    let importance = Importance::Medium;
    Memory {
        id: existing.id.clone(),
        layer: MemoryLayer::STM,
        topic: existing.topic.clone(),
        summary: first_sentence(&existing.content),
        content: existing.content.clone(),
        keywords: vec![],
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

/// Fixture dir, resolved at RUNTIME (mirrors `recall::fixture_dir` /
/// `dedup::fixture_dir` — no `env!("CARGO_MANIFEST_DIR")` literal baked into
/// the binary).
fn fixture_dir() -> PathBuf {
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        return PathBuf::from(manifest)
            .join("tests")
            .join("fixtures")
            .join("eval_gates")
            .join("admission");
    }
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join("crates")
        .join("rein")
        .join("tests")
        .join("fixtures")
        .join("eval_gates")
        .join("admission")
}

/// Load all `case_*.json` admission fixtures; returns `(fixtures, fingerprint)`
/// where the fingerprint is computed over the bytes actually read.
fn load_admission_fixtures() -> Result<(Vec<AdmissionFixture>, String)> {
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
            "admission gate found no fixtures matching `case_*.json` in {}. \
             A 0-fixture scorecard silently disables the gate; fixtures live at \
             `crates/rein/tests/fixtures/eval_gates/admission/case_*.json`. \
             Run rein-eval from the source repo (`cargo run -p rein --bin rein-eval`).",
            dir.display(),
        );
    }

    let mut fixtures = Vec::with_capacity(paths.len());
    let mut corpus: Vec<(String, Vec<u8>)> = Vec::with_capacity(paths.len());
    for path in paths {
        let bytes = std::fs::read(&path)
            .with_context(|| format!("read admission fixture {}", path.display()))?;
        let fx: AdmissionFixture = serde_json::from_slice(&bytes)
            .with_context(|| format!("parse admission fixture {}", path.display()))?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ReinConfig;
    use crate::store::SqliteStore;

    #[test]
    fn admission_gate_is_not_stub() {
        assert!(!AdmissionGate.is_stub());
        assert_eq!(AdmissionGate.name(), "admission");
    }

    #[test]
    fn admission_gate_loads_fixtures() {
        let (fixtures, fingerprint) =
            load_admission_fixtures().expect("admission fixture dir must be readable");
        assert!(!fixtures.is_empty(), "admission corpus is empty");
        assert_eq!(fingerprint.len(), 32, "fingerprint must be 32 hex chars");
        // All three decision classes present so the gate exercises every band.
        for class in ["admit_new", "gray_zone", "duplicate"] {
            assert!(
                fixtures.iter().any(|f| f.expected_decision == class),
                "corpus has no `{class}`-labeled fixtures"
            );
        }
    }

    #[test]
    fn admission_gate_run_returns_scorecard() {
        let store = SqliteStore::in_memory().unwrap();
        let config = ReinConfig::default();
        let sc = AdmissionGate.run(&store, &config).unwrap();
        let (fixtures, fingerprint) = load_admission_fixtures().unwrap();
        assert_eq!(sc.gate_name, "admission");
        assert_eq!(sc.schema_version, SCORECARD_SCHEMA_VERSION);
        assert_eq!(sc.kind, ScorecardKind::Run);
        assert_eq!(sc.fixture_count, fixtures.len());
        assert_eq!(sc.fixture_fingerprint, fingerprint);
        assert!(sc.score >= 0.0 && sc.score <= 1.0);
        // per_fixture emitted in sorted id order (stable McNemar pairing).
        let ids: Vec<String> = sc
            .per_fixture
            .iter()
            .map(|f| f.fixture_id.clone())
            .collect();
        let mut sorted = ids.clone();
        sorted.sort();
        assert_eq!(ids, sorted);
    }

    #[test]
    fn admission_decision_bands() {
        // Synthetic similarities exercising each band directly, matching the
        // live check_dedup bands (auto-merge at MERGE_THRESHOLD = 0.70).
        assert_eq!(decide(0.9), "duplicate", "> 0.70 → duplicate");
        // codex v0.33 R2 P3: live check_dedup auto-merges on `> threshold`
        // (strict); an exactly-0.70 pair falls through to GrayZone.
        assert_eq!(
            decide(0.7),
            "gray_zone",
            "exactly 0.70 (boundary) → gray_zone"
        );
        // codex v0.33 R1 P2: the 0.50–0.70 band is GrayZone (LLM review) in
        // the live path, NOT an automatic duplicate.
        assert_eq!(decide(0.6), "gray_zone", "0.50–0.70 band → gray_zone");
        assert_eq!(decide(0.5), "gray_zone", "exactly 0.50 → gray_zone");
        assert_eq!(decide(0.4), "gray_zone", "mid overlap → gray_zone");
        assert_eq!(decide(0.35), "gray_zone", "lower boundary → gray_zone");
        assert_eq!(decide(0.1), "admit_new", "low overlap → admit_new");
        assert_eq!(decide(0.0), "admit_new", "no existing → admit_new");
    }
}
