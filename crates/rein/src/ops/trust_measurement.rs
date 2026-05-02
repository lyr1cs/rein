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

fn eval_gates() -> Vec<MeasurementGate> {
    vec![
        MeasurementGate {
            name: "recall".to_string(),
            status: "available".to_string(),
            signal: "paired recall/synthesis hit scorecard".to_string(),
            command: "rein-eval synthesis compare".to_string(),
        },
        MeasurementGate {
            name: "dedup".to_string(),
            status: "available".to_string(),
            signal: "cluster-aware dedup thresholds and doctor/index checks".to_string(),
            command: "rein adaptive-status --json".to_string(),
        },
        MeasurementGate {
            name: "admission".to_string(),
            status: "available".to_string(),
            signal: "cluster admission and promotion decisions".to_string(),
            command: "rein adaptive-status --json".to_string(),
        },
        MeasurementGate {
            name: "latency".to_string(),
            status: "available".to_string(),
            signal: "health queue lag plus release-gate policy readiness".to_string(),
            command: "rein health --json".to_string(),
        },
    ]
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
