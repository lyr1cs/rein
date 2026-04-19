//! Diagnostics-category op handlers (Phase 1 PoC: stats, health).

use clap::Args;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use rein_macros::op;

use crate::ops::system_health::{
    self, GrayzoneSnapshot, IndexesSnapshot, QueuesSnapshot, SystemStatus,
};
use crate::ops::{IntoCliText, IntoJson, IntoMarkdown, OpsRuntime};
use crate::types::{HealthReport, MemoryStore, ReinResult, StoreStats};

#[derive(Serialize, Clone, Debug)]
pub struct StatsOutput {
    pub total_memories: usize,
    pub ltm_count: usize,
    pub stm_count: usize,
    pub topic_count: usize,
    pub avg_strength: f64,
    pub memoir_count: usize,
    pub concept_count: usize,
    pub link_count: usize,
    pub hot_count: usize,
    pub warm_count: usize,
    pub cold_count: usize,
}

impl From<StoreStats> for StatsOutput {
    fn from(s: StoreStats) -> Self {
        Self {
            total_memories: s.total_memories,
            ltm_count: s.ltm_count,
            stm_count: s.stm_count,
            topic_count: s.topic_count,
            avg_strength: s.avg_strength,
            memoir_count: s.memoir_count,
            concept_count: s.concept_count,
            link_count: s.link_count,
            hot_count: s.hot_count,
            warm_count: s.warm_count,
            cold_count: s.cold_count,
        }
    }
}

impl IntoJson for StatsOutput {
    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

impl IntoMarkdown for StatsOutput {
    fn to_markdown(&self) -> String {
        format!(
            "**Memory stats**\n\
             - total: {}\n\
             - LTM: {} / STM: {}\n\
             - topics: {}\n\
             - avg strength: {:.3}\n\
             - tiers: hot={} warm={} cold={}\n\
             - memoirs: {} concepts: {} links: {}",
            self.total_memories,
            self.ltm_count,
            self.stm_count,
            self.topic_count,
            self.avg_strength,
            self.hot_count,
            self.warm_count,
            self.cold_count,
            self.memoir_count,
            self.concept_count,
            self.link_count,
        )
    }
}

impl IntoCliText for StatsOutput {
    fn to_cli_text(&self) -> String {
        self.to_markdown()
    }
}

impl OpsRuntime {
    #[op(
        name = "stats",
        category = "memory",
        description = "Show store statistics — counts, layers, tiers",
        cli(name = "stats"),
        mcp(name = "rein_stats"),
        rest(method = "GET", path = "/api/stats"),
    )]
    pub fn stats(&self) -> ReinResult<StatsOutput> {
        let stats = self.with_store(|s| s.stats())?;
        Ok(stats.into())
    }
}

#[derive(Args, Deserialize, JsonSchema, Debug, Clone, Default)]
pub struct HealthParams {
    /// Filter reports to a single topic (positional; pre-A1 CLI compat).
    #[serde(default)]
    pub topic: Option<String>,
}

#[derive(Serialize, Clone, Debug)]
pub struct HealthReportItem {
    pub topic: String,
    pub count: usize,
    pub avg_strength: f64,
    pub stale_count: usize,
    pub needs_consolidation: bool,
}

impl From<HealthReport> for HealthReportItem {
    fn from(r: HealthReport) -> Self {
        Self {
            topic: r.topic,
            count: r.count,
            avg_strength: r.avg_strength,
            stale_count: r.stale_count,
            needs_consolidation: r.needs_consolidation,
        }
    }
}

#[derive(Serialize, Clone, Debug)]
pub struct HealthOutput {
    pub reports: Vec<HealthReportItem>,
    pub indexes: IndexesSnapshot,
    pub queues: QueuesSnapshot,
    pub grayzone: GrayzoneSnapshot,
    pub status: SystemStatus,
}

impl IntoJson for HealthOutput {
    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

impl IntoMarkdown for HealthOutput {
    fn to_markdown(&self) -> String {
        let mut out = String::new();
        if self.reports.is_empty() {
            out.push_str("(no topic reports)\n");
        } else {
            out.push_str("**Topic health**\n");
            for r in &self.reports {
                out.push_str(&format!(
                    "- {}: count={} avg_strength={:.3} stale={} consolidate={}\n",
                    r.topic, r.count, r.avg_strength, r.stale_count, r.needs_consolidation
                ));
            }
        }
        out.push_str(&format!(
            "\n**System**: {}",
            if self.status.ok { "OK" } else { "DEGRADED" }
        ));
        if !self.status.issues.is_empty() {
            out.push_str("\nIssues:");
            for issue in &self.status.issues {
                out.push_str(&format!("\n- {}", issue));
            }
        }
        out.push_str(&format!(
            "\n**Queues**: memory p={} i={} d={} | cleanup p={} i={} d={} | dedup p={} i={} d={} | merge_refine p={} i={} d={}",
            self.queues.memory.pending, self.queues.memory.inflight, self.queues.memory.dead_letters,
            self.queues.cleanup.pending, self.queues.cleanup.inflight, self.queues.cleanup.dead_letters,
            self.queues.dedup.pending, self.queues.dedup.inflight, self.queues.dedup.dead_letters,
            self.queues.merge_refinement.pending, self.queues.merge_refinement.inflight, self.queues.merge_refinement.dead_letters,
        ));
        out.push_str(&format!("\n**Grayzone pending**: {}", self.grayzone.pending));
        out
    }
}

impl IntoCliText for HealthOutput {
    fn to_cli_text(&self) -> String {
        self.to_markdown()
    }
}

impl OpsRuntime {
    #[op(
        name = "health",
        category = "health",
        description = "Per-topic memory health + system index/queue lag",
        cli(name = "health"),
        mcp(name = "rein_health"),
        rest(method = "GET", path = "/api/health"),
    )]
    pub fn health(&self, params: HealthParams) -> ReinResult<HealthOutput> {
        let topic = params.topic.as_deref();
        let store = self.config.open_store()?;
        let reports = store.health(topic)?;
        let system = system_health::collect(&store, &self.config);

        let reports = reports.into_iter().map(HealthReportItem::from).collect();

        Ok(HealthOutput {
            reports,
            indexes: system.indexes,
            queues: system.queues,
            grayzone: system.grayzone,
            status: system.status,
        })
    }
}
