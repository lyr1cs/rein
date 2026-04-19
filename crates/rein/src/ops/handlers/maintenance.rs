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
use crate::types::{Memory, MemoryEvidence, MemoryStore, ReinResult};

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

// ── gc ──────────────────────────────────────────────────────────────────────

#[derive(clap::Args, serde::Deserialize, schemars::JsonSchema, Debug, Clone, Default)]
pub struct GcParams {
    /// Preview without applying changes: report how many memories/concepts
    /// would be decayed or pruned without modifying the database.
    #[serde(default)]
    #[arg(long)]
    pub dry_run: bool,
    /// Strength threshold below which weak STM memories are pruned. Defaults
    /// to the configured `decay.prune_threshold`.
    #[serde(default)]
    #[arg(long)]
    pub threshold: Option<f64>,
}

#[derive(Serialize, Clone, Debug)]
pub struct GcOutput {
    pub decayed: u64,
    pub pruned: u64,
    pub concepts: u64,
    pub dry_run: bool,
    pub threshold: f64,
}

impl IntoJson for GcOutput {
    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

impl IntoMarkdown for GcOutput {
    fn to_markdown(&self) -> String {
        // Mirrors the pre-A1 `rein_gc` MCP non-compact output so MCP callers
        // that parse the string keep working.
        if self.dry_run {
            let mut s = format!(
                "GC dry run: {} weak STM memories would be pruned (threshold: {})",
                self.pruned, self.threshold
            );
            if self.concepts > 0 {
                s.push_str(&format!(", {} low-quality concepts", self.concepts));
            }
            s
        } else {
            let mut s = format!(
                "GC complete: decayed {} memories, pruned {} weak STM memories (threshold: {})",
                self.decayed, self.pruned, self.threshold
            );
            if self.concepts > 0 {
                s.push_str(&format!(", {} low-quality concepts", self.concepts));
            }
            s
        }
    }
}

impl IntoCliText for GcOutput {
    fn to_cli_text(&self) -> String {
        // Mirrors the pre-A1 `handle_gc` CLI output verbatim so shell scripts
        // that grep or parse this text continue to work.
        if self.dry_run {
            let mut s = format!(
                "Would decay {} and prune {} weak STM memories (threshold: {})",
                self.decayed, self.pruned, self.threshold
            );
            if self.concepts > 0 {
                s.push_str(&format!(", {} low-quality concepts", self.concepts));
            }
            s
        } else {
            let mut s = format!(
                "Decayed {} memories, pruned {} weak STM memories (threshold: {})",
                self.decayed, self.pruned, self.threshold
            );
            if self.concepts > 0 {
                s.push_str(&format!(", {} low-quality concepts", self.concepts));
            }
            s
        }
    }
}

impl OpsRuntime {
    #[op(
        name = "gc",
        category = "maintenance",
        description = "Run garbage collection: apply decay to all memories, then prune weak STM memories below the configured strength threshold. Use dry_run=true to preview.",
        mutating = true,
        cli(name = "gc"),
        mcp(name = "rein_gc"),
        rest(method = "POST", path = "/api/gc"),
        auth = "mutation_marker",
    )]
    pub fn gc(&self, params: GcParams) -> ReinResult<GcOutput> {
        self.set_dry_run(params.dry_run);
        let dry_run = self.dry_run();
        let threshold = params
            .threshold
            .unwrap_or(self.config.decay.prune_threshold);
        let config = self.config.clone();

        self.with_store(|store| {
            let (decayed, pruned, concepts) =
                crate::ops::run_gc_adaptive(store, &config, threshold, dry_run)?;
            Ok(GcOutput {
                decayed,
                pruned,
                concepts,
                dry_run,
                threshold,
            })
        })
    }
}

// ── intelligent_merge_try ────────────────────────────────────────────────────

/// Params for the intelligent-merge dry-run classifier.
///
/// Both IDs must exist in the store. The classifier asks an LLM to label the
/// relationship (Ignore / Update / Merge / CreateNew) and returns reasoning and
/// optional synthesized content. No data is written.
#[derive(clap::Args, serde::Deserialize, schemars::JsonSchema, Debug, Clone, Default)]
pub struct IntelligentMergeTryParams {
    /// Existing memory ID (the "baseline" in the merge comparison).
    #[serde(default)]
    pub existing: String,
    /// Incoming memory ID (the candidate being evaluated against `existing`).
    #[serde(default)]
    pub incoming: String,
}

/// Output of the intelligent-merge dry-run classifier.
#[derive(Serialize, Clone, Debug)]
pub struct IntelligentMergeTryOutput {
    pub existing_summary: String,
    pub incoming_summary: String,
    /// `None` when no LLM is configured or on API error (check logs with REIN_LOG=debug).
    pub verdict: Option<String>,
    pub reasoning: Option<String>,
    pub synthesis: Option<String>,
}

impl IntoJson for IntelligentMergeTryOutput {
    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

impl IntoMarkdown for IntelligentMergeTryOutput {
    fn to_markdown(&self) -> String {
        let mut out = format!(
            "**existing**: {}\n**incoming**: {}\n",
            self.existing_summary, self.incoming_summary
        );
        match &self.verdict {
            Some(v) => {
                out.push_str(&format!("\n**verdict**: {v}\n"));
                if let Some(r) = &self.reasoning {
                    out.push_str(&format!("**reason**: {r}\n"));
                }
                if let Some(s) = &self.synthesis {
                    out.push_str(&format!("**synthesis**: {s}\n"));
                }
            }
            None => {
                out.push_str(
                    "\nclassifier returned None (no LLM configured, or API error — check logs with REIN_LOG=debug)\n",
                );
            }
        }
        out
    }
}

impl IntoCliText for IntelligentMergeTryOutput {
    fn to_cli_text(&self) -> String {
        // Preserves the pre-A1 `handle_intelligent_merge_try` output verbatim,
        // including the blank line separator and conditional fields.
        let mut out = format!(
            "→ existing: {}\n→ incoming: {}\n\n",
            self.existing_summary, self.incoming_summary
        );
        match &self.verdict {
            Some(v) => {
                out.push_str(&format!("verdict  : {v}\n"));
                if let Some(r) = &self.reasoning {
                    out.push_str(&format!("reason   : {r}\n"));
                }
                if let Some(s) = &self.synthesis {
                    out.push_str(&format!("synthesis: {s}\n"));
                }
            }
            None => {
                out.push_str(
                    "classifier returned None (no LLM configured, or API error — check logs with REIN_LOG=debug)\n",
                );
            }
        }
        out
    }
}

impl OpsRuntime {
    #[op(
        name = "intelligent_merge_try",
        category = "maintenance",
        description = "Dry-run intelligent merge: classify the relationship between two candidate memories (Ignore / Update / Merge / CreateNew), report the decision path, without committing changes.",
        mutating = false,
        cli(name = "intelligent-merge-try"),
    )]
    pub fn intelligent_merge_try(
        &self,
        params: IntelligentMergeTryParams,
    ) -> ReinResult<IntelligentMergeTryOutput> {
        use crate::extract::intelligent_merge::{classify_insertion, MemorySnippet};

        let existing_id = params.existing.clone();
        let incoming_id = params.incoming.clone();
        let config = self.config.clone();

        self.with_store(|store| {
            let existing = store.get(&existing_id)?;
            let incoming = store.get(&incoming_id)?;

            let a = MemorySnippet::from(&existing);
            let b = MemorySnippet::from(&incoming);

            let existing_summary = existing.summary.clone();
            let incoming_summary = incoming.summary.clone();

            let result = classify_insertion(&config, &a, &b);

            Ok(IntelligentMergeTryOutput {
                existing_summary,
                incoming_summary,
                verdict: result.as_ref().map(|v| format!("{:?}", v.verdict)),
                reasoning: result.as_ref().and_then(|v| v.reasoning.clone()),
                synthesis: result.as_ref().and_then(|v| v.synthesized.clone()),
            })
        })
    }
}
