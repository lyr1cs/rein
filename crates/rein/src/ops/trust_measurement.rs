//! Unified Trust & Measurement snapshot for ARS rollout operations.

use serde::Serialize;

use crate::config::ReinConfig;
use crate::ops::ars_release_gate::ArsAccelerationReleaseGateReport;
use crate::ops::system_health::SystemHealthSnapshot;
use crate::store::SqliteStore;

pub const TRUST_MEASUREMENT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct MeasurementGate {
    pub name: String,
    pub status: String,
    pub signal: String,
    pub command: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct IndexConsistencyReport {
    pub status: String,
    pub active_memory_count: u64,
    pub vector_row_count: u64,
    pub missing_embeddings: u64,
    pub orphan_embeddings: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct BackgroundObservabilityReport {
    pub status: String,
    pub adaptive_version: u64,
    pub learned_shadow_fusion_buckets: usize,
    pub feedback_event_count: u64,
    pub consumer_offset_count: u64,
    pub system: SystemHealthSnapshot,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ActiveLearningReport {
    pub status: String,
    pub llm_judge_enabled: bool,
    pub synthesis_enabled: bool,
    pub concept_summary_enabled: bool,
    pub nightly_cron_enabled: bool,
    pub sample_rate_cold_start: f64,
    pub sample_rate_warm: f64,
    pub nightly_cron_sample_rate: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct TrustMeasurementReport {
    pub schema_version: u32,
    pub purpose: String,
    pub release_gate: ArsAccelerationReleaseGateReport,
    pub eval_gates: Vec<MeasurementGate>,
    pub index_consistency: IndexConsistencyReport,
    pub background_observability: BackgroundObservabilityReport,
    pub active_learning: ActiveLearningReport,
}

pub fn collect(store: &SqliteStore, config: &ReinConfig) -> TrustMeasurementReport {
    let state =
        crate::store::adaptive::AdaptiveState::restore_snapshot(store.conn()).unwrap_or_default();
    let system = crate::ops::system_health::collect(store, config);
    let index_consistency = collect_index_consistency(store);
    let background_status = if system.status.ok && index_consistency.status == "ok" {
        "ok"
    } else {
        "attention"
    };

    TrustMeasurementReport {
        schema_version: TRUST_MEASUREMENT_SCHEMA_VERSION,
        purpose: "trust_measurement_platform_for_ars_rollout".to_string(),
        release_gate: crate::ops::ars_release_gate::ars_acceleration_release_gate_report(
            store, config,
        ),
        eval_gates: eval_gates(),
        index_consistency,
        background_observability: BackgroundObservabilityReport {
            status: background_status.to_string(),
            adaptive_version: state.version,
            learned_shadow_fusion_buckets: state.learned_shadow_fusion.len(),
            feedback_event_count: count_table(store, "feedback_events"),
            consumer_offset_count: count_table(store, "consumer_offsets"),
            system,
        },
        active_learning: active_learning_report(config),
    }
}

/// v0.32 (T&M Phase 2): each `MeasurementGate` is now populated from a real
/// eval-gate harness round-trip — `docs/eval-baselines/{name}.json` (baseline
/// scorecard committed to the repo) compared against
/// `target/eval-gates/{name}-run.json` (last `rein-eval gate run` artifact)
/// via paired McNemar.  When either side is missing the status falls back to
/// `no_baseline` / `no_run` / `stub` so the report still serializes cleanly
/// for the doctor + GUI consumers.
///
/// The CWD used for scorecard lookup is `env::current_dir()` — the typical
/// invocation pattern is `cd source/rein && rein trust-measurement`.  In a
/// deployed binary running far from the source repo the scorecards won't be
/// found and every gate degrades to `no_baseline`; that's the honest answer
/// because nothing has been measured there.
fn eval_gates() -> Vec<MeasurementGate> {
    use crate::eval::gates::{self, ScorecardLoad, ScorecardStatus};
    let repo_root = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    let target_dir = repo_root.join("target");

    let mut gates_out = Vec::new();
    for gate in gates::all_gates() {
        let name = gate.name().to_string();
        let baseline_path = gates::baseline_path(&repo_root, &name);
        let run_path = gates::run_path(&target_dir, &name);

        // v0.32 R4 P2-#2: corrupt scorecard short-circuits to a
        // dedicated "corrupt" status — caller knows to repair the
        // file rather than treating it as merely absent.
        //
        // v0.32 R9 P2-#1: `signal` is serialized over the PUBLIC
        // `/api/trust-measurement` route, so it must NOT include
        // absolute filesystem paths, the process CWD, or any other
        // host-local detail.  Operators who need the full path can
        // get it via `rein doctor` (local-only).  We deliberately
        // drop the underlying `read_scorecard` error message from the
        // corrupt branch for the same reason — it embeds the
        // absolute scorecard path via `with_context(...)`.
        let baseline_load = gates::load_scorecard(&baseline_path);
        let current_load = gates::load_scorecard(&run_path);
        if matches!(baseline_load, ScorecardLoad::Corrupt(_)) {
            gates_out.push(MeasurementGate {
                name: name.clone(),
                status: "corrupt".to_string(),
                signal: "baseline scorecard exists but failed to parse — run \
                         `rein doctor` for the full diagnostic"
                    .to_string(),
                command: format!("rein-eval gate baseline --gate {name}"),
            });
            continue;
        }
        if matches!(current_load, ScorecardLoad::Corrupt(_)) {
            gates_out.push(MeasurementGate {
                name: name.clone(),
                status: "corrupt".to_string(),
                signal: "run scorecard exists but failed to parse — run \
                         `rein doctor` for the full diagnostic"
                    .to_string(),
                command: format!("rein-eval gate run --gate {name}"),
            });
            continue;
        }
        let baseline = match baseline_load {
            ScorecardLoad::Loaded(sc) => Some(sc),
            _ => None,
        };
        let current = match current_load {
            ScorecardLoad::Loaded(sc) => Some(sc),
            _ => None,
        };

        let (status, signal, command) = if gate.is_stub() {
            (
                "stub".to_string(),
                "placeholder for v0.32 — full gate impl deferred to v0.33+".to_string(),
                format!("rein-eval gate run --gate {name}"),
            )
        } else if baseline.is_none() {
            (
                "no_baseline".to_string(),
                // v0.32 R9 P2-#1: generic message — see comment block above.
                "no baseline scorecard committed for this gate".to_string(),
                format!("rein-eval gate baseline --gate {name}"),
            )
        } else if current.is_none() {
            (
                "no_run".to_string(),
                // v0.32 R9 P2-#1: generic message — see comment block above.
                "baseline exists; run scorecard not yet generated for this build".to_string(),
                format!("rein-eval gate run --gate {name}"),
            )
        } else {
            let cmp = gates::compare_scorecards(
                &name,
                baseline.as_ref(),
                current.as_ref(),
                gates::DEFAULT_NOISE_FLOOR,
            );
            let status = match cmp.status {
                ScorecardStatus::Ship => "ship".to_string(),
                ScorecardStatus::Bail => "bail".to_string(),
                ScorecardStatus::NoData => "no_data".to_string(),
            };
            (
                status,
                cmp.reason,
                format!("rein-eval gate compare --gate {name}"),
            )
        };

        gates_out.push(MeasurementGate {
            name,
            status,
            signal,
            command,
        });
    }
    gates_out
}

fn collect_index_consistency(store: &SqliteStore) -> IndexConsistencyReport {
    let active_memory_count = query_count(
        store,
        "SELECT COUNT(*) FROM memories WHERE status IN ('active','updated')",
    );
    let vector_row_count = query_count(store, "SELECT COUNT(*) FROM vec_memories");
    let missing_embeddings = query_count(
        store,
        "SELECT COUNT(*)
           FROM memories m
           LEFT JOIN vec_memories v ON v.id = m.id
          WHERE m.status IN ('active','updated')
            AND v.id IS NULL",
    );
    let orphan_embeddings = query_count(
        store,
        "SELECT COUNT(*)
           FROM vec_memories v
           LEFT JOIN memories m ON m.id = v.id
          WHERE m.id IS NULL",
    );
    let status = if missing_embeddings == 0 && orphan_embeddings == 0 {
        "ok"
    } else {
        "attention"
    };
    IndexConsistencyReport {
        status: status.to_string(),
        active_memory_count,
        vector_row_count,
        missing_embeddings,
        orphan_embeddings,
    }
}

fn active_learning_report(config: &ReinConfig) -> ActiveLearningReport {
    let active = config.ars.llm_judge.enabled
        && (config.ars.llm_judge.synthesis_enabled || config.ars.llm_judge.concept_summary_enabled)
        && config.ars.llm_judge.nightly_cron.enabled;
    ActiveLearningReport {
        status: if active { "active" } else { "disabled" }.to_string(),
        llm_judge_enabled: config.ars.llm_judge.enabled,
        synthesis_enabled: config.ars.llm_judge.synthesis_enabled,
        concept_summary_enabled: config.ars.llm_judge.concept_summary_enabled,
        nightly_cron_enabled: config.ars.llm_judge.nightly_cron.enabled,
        sample_rate_cold_start: config.ars.llm_judge.sample_rate_cold_start,
        sample_rate_warm: config.ars.llm_judge.sample_rate_warm,
        nightly_cron_sample_rate: config.ars.llm_judge.nightly_cron.sample_rate,
    }
}

fn count_table(store: &SqliteStore, table: &str) -> u64 {
    let sql = format!("SELECT COUNT(*) FROM {table}");
    query_count(store, &sql)
}

fn query_count(store: &SqliteStore, sql: &str) -> u64 {
    store
        .conn()
        .query_row(sql, [], |row| row.get::<_, i64>(0))
        .map(|n| n.max(0) as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ReinConfig;
    use crate::store::SqliteStore;
    use tempfile::tempdir;

    #[test]
    fn trust_measurement_report_surfaces_eval_gates_and_observability() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("memories.db");
        let mut config = ReinConfig::default();
        config.database.path = db_path.to_string_lossy().into_owned();
        // v0.28.7 H0 — defaults reverted to `false`. Explicit opt-in here so the
        // test still exercises the "report mirrors enabled-state" path.
        config.ars.llm_judge.enabled = true;
        config.ars.llm_judge.nightly_cron.enabled = true;
        let store = SqliteStore::new(&db_path, "text-embedding-3-small", 3072).unwrap();

        let report = collect(&store, &config);

        assert_eq!(report.schema_version, TRUST_MEASUREMENT_SCHEMA_VERSION);
        for gate in ["recall", "dedup", "admission", "latency"] {
            assert!(
                report.eval_gates.iter().any(|item| item.name == gate),
                "missing eval gate {gate}"
            );
        }
        assert_eq!(report.index_consistency.missing_embeddings, 0);
        assert!(report.background_observability.system.status.ok);
        assert!(report.active_learning.llm_judge_enabled);
        assert!(report.active_learning.nightly_cron_enabled);
    }
}
