//! v0.33 dedup gate — pairwise duplicate-detection quality over a fixture
//! corpus of labeled text pairs.
//!
//! Hermetic + pure: scores each pair with `extract::dedup::similarity` (max of
//! Jaccard / containment over normalized tokens) and classifies it a duplicate
//! when `similarity >= DEDUP_THRESHOLD`.  No store, no config, no LLM — the
//! same input always produces the same hit, which is what a reproducible gate
//! needs.  `hit = (similarity(a, b) >= threshold) == is_duplicate`.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use chrono::Utc;
use serde::Deserialize;

use crate::config::ReinConfig;
use crate::eval::gates::{
    fixture_corpus_fingerprint, FixtureResult, Gate, GateScorecard, ScorecardKind,
    SCORECARD_SCHEMA_VERSION,
};
use crate::extract::dedup::similarity;
use crate::store::SqliteStore;

/// Classification threshold: the merge bound used by `check_dedup`
/// (`gray_zone_lower_bound` upper).  Documented default for v0.33; production
/// calibration deferred to `docs/backlog/v0.33-eval-gate-calibration.md`.
const DEDUP_THRESHOLD: f32 = 0.50;

pub struct DedupGate;

#[derive(Debug, Deserialize)]
struct DedupFixture {
    id: String,
    text_a: String,
    text_b: String,
    is_duplicate: bool,
}

impl Gate for DedupGate {
    fn name(&self) -> &'static str {
        "dedup"
    }

    fn run(&self, _store: &SqliteStore, _config: &ReinConfig) -> Result<GateScorecard> {
        let (fixtures, fixture_fingerprint) = load_dedup_fixtures()?;
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
            gate_name: "dedup".to_string(),
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

/// `hit` = the similarity classifier agrees with the ground-truth label.
fn classify_one(fx: &DedupFixture) -> bool {
    let predicted_duplicate = similarity(&fx.text_a, &fx.text_b) >= DEDUP_THRESHOLD;
    predicted_duplicate == fx.is_duplicate
}

/// Fixture dir, resolved at RUNTIME (mirrors `recall::fixture_dir` — no
/// `env!("CARGO_MANIFEST_DIR")` literal baked into the binary).
fn fixture_dir() -> PathBuf {
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        return PathBuf::from(manifest)
            .join("tests")
            .join("fixtures")
            .join("eval_gates")
            .join("dedup");
    }
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join("crates")
        .join("rein")
        .join("tests")
        .join("fixtures")
        .join("eval_gates")
        .join("dedup")
}

/// Load all `case_*.json` dedup fixtures; returns `(fixtures, fingerprint)`
/// where the fingerprint is computed over the bytes actually read.
fn load_dedup_fixtures() -> Result<(Vec<DedupFixture>, String)> {
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
            "dedup gate found no fixtures matching `case_*.json` in {}. \
             A 0-fixture scorecard silently disables the gate; fixtures live at \
             `crates/rein/tests/fixtures/eval_gates/dedup/case_*.json`. \
             Run rein-eval from the source repo (`cargo run -p rein --bin rein-eval`).",
            dir.display(),
        );
    }

    let mut fixtures = Vec::with_capacity(paths.len());
    let mut corpus: Vec<(String, Vec<u8>)> = Vec::with_capacity(paths.len());
    for path in paths {
        let bytes = std::fs::read(&path)
            .with_context(|| format!("read dedup fixture {}", path.display()))?;
        let fx: DedupFixture = serde_json::from_slice(&bytes)
            .with_context(|| format!("parse dedup fixture {}", path.display()))?;
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
    fn dedup_gate_is_not_stub() {
        assert!(!DedupGate.is_stub());
        assert_eq!(DedupGate.name(), "dedup");
    }

    #[test]
    fn dedup_gate_loads_fixtures() {
        let (fixtures, fingerprint) =
            load_dedup_fixtures().expect("dedup fixture dir must be readable");
        assert!(!fixtures.is_empty(), "dedup corpus is empty");
        assert_eq!(fingerprint.len(), 32, "fingerprint must be 32 hex chars");
        // Both classes present so the gate exercises true/false branches.
        assert!(
            fixtures.iter().any(|f| f.is_duplicate),
            "corpus has no duplicate-labeled pairs"
        );
        assert!(
            fixtures.iter().any(|f| !f.is_duplicate),
            "corpus has no distinct-labeled pairs"
        );
    }

    #[test]
    fn dedup_gate_run_returns_scorecard() {
        let store = SqliteStore::in_memory().unwrap();
        let config = ReinConfig::default();
        let sc = DedupGate.run(&store, &config).unwrap();
        let (fixtures, fingerprint) = load_dedup_fixtures().unwrap();
        assert_eq!(sc.gate_name, "dedup");
        assert_eq!(sc.schema_version, SCORECARD_SCHEMA_VERSION);
        assert_eq!(sc.kind, ScorecardKind::Run);
        assert_eq!(sc.fixture_count, fixtures.len());
        assert_eq!(sc.fixture_fingerprint, fingerprint);
        assert!(sc.score >= 0.0 && sc.score <= 1.0);
        // per_fixture emitted in sorted id order (stable McNemar pairing).
        let ids: Vec<String> = sc.per_fixture.iter().map(|f| f.fixture_id.clone()).collect();
        let mut sorted = ids.clone();
        sorted.sort();
        assert_eq!(ids, sorted);
    }

    #[test]
    fn dedup_classify_exact_duplicate_and_unrelated() {
        let dup = DedupFixture {
            id: "t_dup".to_string(),
            text_a: "Connection pooling reuses open database connections".to_string(),
            text_b: "Connection pooling reuses open database connections".to_string(),
            is_duplicate: true,
        };
        assert!(classify_one(&dup), "identical text must classify as duplicate");

        let distinct = DedupFixture {
            id: "t_distinct".to_string(),
            text_a: "I adopted a cat from the shelter last weekend".to_string(),
            text_b: "The Kubernetes cluster runs three worker nodes".to_string(),
            is_duplicate: false,
        };
        assert!(
            classify_one(&distinct),
            "unrelated text must classify as distinct (hit = correct negative)"
        );
    }
}
