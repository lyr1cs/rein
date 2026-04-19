//! Maintenance / memory-maintenance ops — Phase 2.3 A1 migration.
//!
//! Each op replaces a legacy MCP #[tool] handler + CLI clap arm pair.
//! Business logic stays in `crate::store` / `crate::ops::*`; handlers
//! parse params, route dry_run through OpsRuntime (Task 0), call the
//! underlying function, and shape the response.

use clap::Args;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use rein_macros::op;

use crate::ops::{IntoCliText, IntoJson, IntoMarkdown, OpsRuntime};
use crate::types::{Memory, MemoryEvidence, ReinResult};

fn default_canonicals_limit() -> usize {
    20
}

#[derive(Args, Deserialize, JsonSchema, Debug, Clone)]
pub struct CanonicalsParams {
    /// Maximum number of canonical memories to return.
    #[serde(default = "default_canonicals_limit")]
    #[arg(short, long, default_value = "20")]
    pub limit: usize,
}

impl Default for CanonicalsParams {
    fn default() -> Self {
        Self {
            limit: default_canonicals_limit(),
        }
    }
}

#[derive(Serialize, Clone, Debug)]
pub struct CanonicalsOutput {
    pub canonicals: Vec<Memory>,
}

impl IntoJson for CanonicalsOutput {
    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

impl IntoMarkdown for CanonicalsOutput {
    fn to_markdown(&self) -> String {
        if self.canonicals.is_empty() {
            return "No canonical memories found".to_string();
        }
        let mut out = String::new();
        for memory in &self.canonicals {
            out.push_str(&format!(
                "- {} [{}] support={} merges={} diversity={:.2} dedup_conf={:.2}\n",
                memory.id,
                memory.summary,
                memory.support_count,
                memory.merge_count,
                memory.source_diversity,
                memory.dedup_confidence,
            ));
        }
        out
    }
}

impl IntoCliText for CanonicalsOutput {
    fn to_cli_text(&self) -> String {
        // Mirror the pre-A1 `handle_canonicals` output format verbatim so
        // scripts that parse it continue to work.
        if self.canonicals.is_empty() {
            return "No canonical memories found".to_string();
        }
        let mut out = String::new();
        for memory in &self.canonicals {
            out.push_str(&format!(
                "- {} [{}] support={} merges={} diversity={:.2} dedup_conf={:.2}\n",
                memory.id,
                memory.summary,
                memory.support_count,
                memory.merge_count,
                memory.source_diversity,
                memory.dedup_confidence,
            ));
        }
        out
    }
}

impl OpsRuntime {
    #[op(
        name = "canonicals",
        category = "memory",
        description = "List canonical memories — one row per canonical, ordered by recency. Includes support count, merge count, source diversity, and dedup confidence.",
        cli(name = "canonicals"),
        mcp(name = "rein_canonicals"),
        rest(method = "GET", path = "/api/canonicals"),
    )]
    pub fn canonicals(&self, params: CanonicalsParams) -> ReinResult<CanonicalsOutput> {
        self.with_store(|store| {
            let canonicals = store.list_canonical_memories(params.limit)?;
            Ok(CanonicalsOutput { canonicals })
        })
    }
}

fn default_evidence_limit() -> usize {
    20
}

#[derive(Args, Deserialize, JsonSchema, Debug, Clone)]
pub struct EvidenceParams {
    /// Canonical memory ID whose evidence snapshots to list.
    pub canonical_id: String,
    /// Maximum number of evidence rows to return.
    #[serde(default = "default_evidence_limit")]
    #[arg(short, long, default_value = "20")]
    pub limit: usize,
}

impl Default for EvidenceParams {
    fn default() -> Self {
        Self {
            canonical_id: String::new(),
            limit: default_evidence_limit(),
        }
    }
}

#[derive(Serialize, Clone, Debug)]
pub struct EvidenceOutput {
    pub canonical_id: String,
    pub evidence: Vec<MemoryEvidence>,
}

impl IntoJson for EvidenceOutput {
    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

impl IntoMarkdown for EvidenceOutput {
    fn to_markdown(&self) -> String {
        if self.evidence.is_empty() {
            return format!("No evidence found for canonical '{}'", self.canonical_id);
        }
        let mut out = String::new();
        for item in &self.evidence {
            out.push_str(&format!(
                "- {} [{}] {}\n{}\n",
                item.id, item.source_topic, item.summary, item.content
            ));
        }
        out
    }
}

impl IntoCliText for EvidenceOutput {
    fn to_cli_text(&self) -> String {
        // Mirror the pre-A1 `handle_evidence` output format verbatim so
        // scripts that parse it continue to work.
        if self.evidence.is_empty() {
            return format!("No evidence found for canonical '{}'", self.canonical_id);
        }
        let mut out = String::new();
        for item in &self.evidence {
            out.push_str(&format!(
                "- {} [{}] {}\n{}\n",
                item.id, item.source_topic, item.summary, item.content
            ));
        }
        out
    }
}

impl OpsRuntime {
    #[op(
        name = "evidence",
        category = "memory",
        description = "List evidence snapshots for a canonical memory, ordered by import time descending.",
        cli(name = "evidence"),
        mcp(name = "rein_evidence"),
        rest(method = "GET", path = "/api/evidence"),
    )]
    pub fn evidence(&self, params: EvidenceParams) -> ReinResult<EvidenceOutput> {
        let canonical_id = params.canonical_id.clone();
        self.with_store(|store| {
            let evidence = store.list_memory_evidence(&canonical_id, params.limit)?;
            Ok(EvidenceOutput {
                canonical_id: canonical_id.clone(),
                evidence,
            })
        })
    }
}
