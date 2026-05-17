//! v0.32 dedup gate — STUB.  Real implementation deferred to v0.33+.

use anyhow::Result;

use crate::config::ReinConfig;
use crate::eval::gates::{Gate, GateScorecard, ScorecardKind, SCORECARD_SCHEMA_VERSION};
use crate::store::SqliteStore;

pub struct DedupGate;

impl Gate for DedupGate {
    fn name(&self) -> &'static str {
        "dedup"
    }

    fn is_stub(&self) -> bool {
        true
    }

    fn run(&self, _store: &SqliteStore, _config: &ReinConfig) -> Result<GateScorecard> {
        Ok(GateScorecard {
            schema_version: SCORECARD_SCHEMA_VERSION,
            gate_name: "dedup".to_string(),
            kind: ScorecardKind::Run,
            created_at: chrono::Utc::now().timestamp(),
            rein_version: env!("CARGO_PKG_VERSION").to_string(),
            fixture_count: 0,
            score: 0.0,
            per_fixture: vec![],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ReinConfig;
    use crate::store::SqliteStore;
    use tempfile::tempdir;

    #[test]
    fn dedup_stub_is_marked_stub() {
        assert!(DedupGate.is_stub());
        assert_eq!(DedupGate.name(), "dedup");
    }

    #[test]
    fn dedup_stub_run_returns_empty_scorecard() {
        let dir = tempdir().unwrap();
        let mut config = ReinConfig::default();
        config.database.path = dir
            .path()
            .join("memories.db")
            .to_string_lossy()
            .into_owned();
        let store = SqliteStore::new(
            &dir.path().join("memories.db"),
            "text-embedding-3-small",
            3072,
        )
        .unwrap();
        let sc = DedupGate.run(&store, &config).unwrap();
        assert_eq!(sc.fixture_count, 0);
        assert_eq!(sc.score, 0.0);
        assert!(sc.per_fixture.is_empty());
    }
}
