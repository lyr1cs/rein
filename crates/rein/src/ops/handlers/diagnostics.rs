//! Diagnostics-category op handlers (Phase 1 PoC: stats, health).

use serde::Serialize;

use rein_macros::op;

use crate::ops::{IntoCliText, IntoJson, IntoMarkdown, OpsRuntime};
use crate::types::{MemoryStore, ReinResult, StoreStats};

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
