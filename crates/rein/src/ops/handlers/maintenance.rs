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
use crate::types::{Memory, MemoryEvidence, MemoryStore, ReinError, ReinResult};

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

// ── dedup ────────────────────────────────────────────────────────────────────

#[derive(clap::Args, serde::Deserialize, schemars::JsonSchema, Debug, Clone, Default)]
pub struct DedupParams {
    /// Preview without applying changes: report how many duplicates would be
    /// removed without modifying the database.
    #[serde(default)]
    #[arg(long)]
    pub dry_run: bool,
    /// Deduplicate across normalized topic variants instead of exact-topic only.
    #[serde(default)]
    #[arg(long)]
    pub merge_variants: bool,
}

#[derive(Serialize, Clone, Debug)]
pub struct DedupOutput {
    pub found: u32,
    pub removed: u32,
    pub dry_run: bool,
    pub merge_variants: bool,
    pub threshold: f32,
}

impl IntoJson for DedupOutput {
    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

impl IntoMarkdown for DedupOutput {
    fn to_markdown(&self) -> String {
        // Mirrors the pre-A1 `rein_dedup` MCP non-compact output so MCP callers
        // that parse the string continue to work.
        if self.dry_run {
            format!(
                "Dedup scan: found {} potential duplicates (dry run, none removed)",
                self.found
            )
        } else {
            format!(
                "Dedup scan: found {} duplicates, removed {}",
                self.found, self.removed
            )
        }
    }
}

impl IntoCliText for DedupOutput {
    fn to_cli_text(&self) -> String {
        // Mirrors the pre-A1 `handle_dedup` CLI output verbatim so shell scripts
        // that parse this text continue to work.
        if self.dry_run {
            format!("Found {} duplicates (dry-run, nothing removed)", self.found)
        } else {
            format!("Removed {} of {} duplicates", self.removed, self.found)
        }
    }
}

impl OpsRuntime {
    #[op(
        name = "dedup",
        category = "maintenance",
        description = "Scan for duplicate memories using content similarity. Use dry_run=true to preview without deleting. Optional merge_variants collapses semantic variants.",
        mutating = true,
        cli(name = "dedup"),
        mcp(name = "rein_dedup"),
        rest(method = "POST", path = "/api/dedup"),
        auth = "mutation_marker",
    )]
    pub fn dedup(&self, params: DedupParams) -> ReinResult<DedupOutput> {
        self.set_dry_run(params.dry_run);
        let dry_run = self.dry_run();
        let merge_variants = params.merge_variants;
        let config = self.config.clone();

        self.with_store(|store| {
            let threshold = crate::ops::effective_dedup_threshold(store, &config);
            let (found, removed) =
                crate::ops::run_dedup(store, &config, threshold, dry_run, merge_variants)?;
            Ok(DedupOutput {
                found,
                removed,
                dry_run,
                merge_variants,
                threshold,
            })
        })
    }
}

// ── dedup_concepts ───────────────────────────────────────────────────────────

/// Params for the dedup-concepts op.
///
/// No fields — the underlying `SqliteStore::dedup_concepts` performs the full
/// scan and merge atomically. A `dry_run` mode is intentionally omitted: the
/// underlying function does not support preview and changing it would be a
/// logic change outside the scope of this migration.
#[derive(clap::Args, serde::Deserialize, schemars::JsonSchema, Debug, Clone, Default)]
pub struct DedupConceptsParams {}

/// Output of the dedup-concepts op.
#[derive(Serialize, Clone, Debug)]
pub struct DedupConceptsOutput {
    /// Number of duplicate groups merged (each group = 2+ concepts with same
    /// normalized name within a memoir).
    pub groups_merged: usize,
    /// Total number of duplicate concept records removed.
    pub concepts_removed: usize,
}

impl IntoJson for DedupConceptsOutput {
    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

impl IntoMarkdown for DedupConceptsOutput {
    fn to_markdown(&self) -> String {
        // Mirrors the MCP-visible summary format.
        format!(
            "Concept dedup: merged {} groups, removed {} duplicate concepts",
            self.groups_merged, self.concepts_removed
        )
    }
}

impl IntoCliText for DedupConceptsOutput {
    fn to_cli_text(&self) -> String {
        // Mirror the pre-A1 `handle_dedup_concepts` CLI output verbatim so
        // shell scripts that parse it continue to work.
        format!(
            "Concept dedup: merged {} groups, removed {} duplicate concepts",
            self.groups_merged, self.concepts_removed
        )
    }
}

impl OpsRuntime {
    #[op(
        name = "dedup_concepts",
        category = "knowledge",
        description = "Deduplicate concepts in the knowledge graph. Merges concepts with the same normalized name within each memoir, keeping the oldest as canonical.",
        mutating = true,
        cli(name = "dedup-concepts"),
        mcp(name = "rein_dedup_concepts"),
        rest(method = "POST", path = "/api/dedup_concepts"),
        auth = "mutation_marker",
    )]
    pub fn dedup_concepts(&self, _params: DedupConceptsParams) -> ReinResult<DedupConceptsOutput> {
        self.with_store(|store| {
            let (groups_merged, concepts_removed) = store.dedup_concepts()?;
            Ok(DedupConceptsOutput {
                groups_merged,
                concepts_removed,
            })
        })
    }
}

// ── dedup_log ────────────────────────────────────────────────────────────────

fn default_dedup_log_limit() -> usize {
    50
}

/// Query parameters for the dedup-log op.
///
/// `canonical` and `operator` are optional filters.  `limit` is clamped to
/// `[1, 500]` so the REST surface cannot request arbitrarily large payloads.
#[derive(clap::Args, serde::Deserialize, schemars::JsonSchema, Debug, Clone)]
pub struct DedupLogParams {
    /// Return only decisions whose `canonical_id` matches this value.
    #[serde(default)]
    #[arg(long)]
    pub canonical: Option<String>,
    /// Return only decisions whose `operator` matches this value (e.g. `llm_verdict`, `auto`).
    #[serde(default)]
    #[arg(long)]
    pub operator: Option<String>,
    /// Maximum number of decisions to return.
    #[serde(default = "default_dedup_log_limit")]
    #[arg(short, long, default_value = "50")]
    pub limit: usize,
}

impl Default for DedupLogParams {
    fn default() -> Self {
        Self {
            canonical: None,
            operator: None,
            limit: default_dedup_log_limit(),
        }
    }
}

/// A single dedup decision row — all 15 fields stored in the DB.
///
/// `novel_facts` is kept as a raw JSON string (e.g. `"[]"` or `"[\"fact\"]"`)
/// to preserve exact wire-format parity with the legacy derived REST handler
/// consumed by the Neural Wiki GUI (Provenance page).
#[derive(Serialize, Clone, Debug)]
pub struct DedupDecisionRow {
    pub id: String,
    pub winner_id: Option<String>,
    pub loser_id: Option<String>,
    pub canonical_id: Option<String>,
    pub lexical_score: Option<f32>,
    pub embedding_score: Option<f32>,
    pub relation: String,
    pub confidence: f32,
    pub reason: String,
    pub operator: String,
    pub reversible: bool,
    pub merged_summary: Option<String>,
    /// Raw JSON string — kept as-is from the DB column to match existing GUI contract.
    pub novel_facts: String,
    pub conflict_detected: bool,
    pub created_at: String,
}

impl DedupDecisionRow {
    fn from_decision(d: crate::types::DedupDecision) -> Self {
        let novel_facts = serde_json::to_string(&d.novel_facts).unwrap_or_else(|_| "[]".to_string());
        let created_at = d.created_at.to_rfc3339();
        Self {
            id: d.id,
            winner_id: d.winner_id,
            loser_id: d.loser_id,
            canonical_id: d.canonical_id,
            lexical_score: d.lexical_score,
            embedding_score: d.embedding_score,
            relation: d.relation.to_string(),
            confidence: d.confidence,
            reason: d.reason,
            operator: d.operator,
            reversible: d.reversible,
            merged_summary: d.merged_summary,
            novel_facts,
            conflict_detected: d.conflict_detected,
            created_at,
        }
    }
}

#[derive(Serialize, Clone, Debug)]
pub struct DedupLogOutput {
    pub decisions: Vec<DedupDecisionRow>,
}

impl IntoJson for DedupLogOutput {
    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

impl IntoMarkdown for DedupLogOutput {
    fn to_markdown(&self) -> String {
        if self.decisions.is_empty() {
            return "No dedup decisions found".to_string();
        }
        let mut out = String::new();
        for d in &self.decisions {
            out.push_str(&format!(
                "- {} relation={} confidence={:.2} winner={:?} loser={:?} reason={}\n",
                d.id, d.relation, d.confidence, d.winner_id, d.loser_id, d.reason
            ));
        }
        out
    }
}

impl IntoCliText for DedupLogOutput {
    fn to_cli_text(&self) -> String {
        // Mirror the pre-A1 `handle_dedup_log` output format verbatim.
        if self.decisions.is_empty() {
            return "No dedup decisions found".to_string();
        }
        let mut out = String::new();
        for d in &self.decisions {
            out.push_str(&format!(
                "- {} relation={} confidence={:.2} winner={:?} loser={:?} reason={}\n",
                d.id, d.relation, d.confidence, d.winner_id, d.loser_id, d.reason
            ));
        }
        out
    }
}

impl OpsRuntime {
    #[op(
        name = "dedup_log",
        category = "maintenance",
        description = "Show recent deduplication decisions (kept/merged/skipped with reasons). Read-only trace for debugging dedup behavior.",
        cli(name = "dedup-log"),
        rest(method = "GET", path = "/api/dedup_decisions"),
    )]
    pub fn dedup_log(&self, params: DedupLogParams) -> ReinResult<DedupLogOutput> {
        let canonical = params.canonical.clone();
        let operator = params.operator.clone();
        let limit = params.limit.clamp(1, 500);
        self.with_store(|store| {
            let decisions = store.list_dedup_decisions_filtered(
                canonical.as_deref(),
                operator.as_deref(),
                limit,
            )?;
            Ok(DedupLogOutput {
                decisions: decisions.into_iter().map(DedupDecisionRow::from_decision).collect(),
            })
        })
    }
}

// ── migrate ──────────────────────────────────────────────────────────────────

/// Params for the migrate command.
///
/// Without `--reindex`, imports data from a QMD SQLite database. With
/// `--reindex`, re-embeds all existing memories with the current embedding
/// model (rebuilds the vector index).
#[derive(clap::Args, serde::Deserialize, schemars::JsonSchema, Debug, Clone, Default)]
pub struct MigrateParams {
    /// Path to the QMD SQLite database. Defaults to `~/.cache/qmd/index.sqlite`.
    /// Ignored when `--reindex` is set.
    #[serde(default)]
    #[arg(long)]
    pub from_qmd: Option<String>,
    /// Re-embed all memories with the current embedding model and rebuild the
    /// vector index. Mutually exclusive with `--from-qmd` (reindex takes
    /// priority when both are set).
    #[serde(default)]
    #[arg(long)]
    pub reindex: bool,
}

/// Summary of a migrate run.
#[derive(Serialize, Clone, Debug)]
pub struct MigrateOutput {
    /// Human-readable summary line (mirrors the pre-A1 println output).
    pub summary: String,
    /// True when `--reindex` mode was used.
    pub reindex: bool,
    /// Number of QMD documents read (only populated in from-qmd mode).
    pub documents_read: Option<usize>,
    /// Number of chunks / memories created (from-qmd) or re-embedded (reindex).
    pub items_processed: usize,
    /// Number of errors encountered.
    pub errors: usize,
}

impl IntoJson for MigrateOutput {
    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

impl IntoMarkdown for MigrateOutput {
    fn to_markdown(&self) -> String {
        self.summary.clone()
    }
}

impl IntoCliText for MigrateOutput {
    fn to_cli_text(&self) -> String {
        // Mirror the pre-A1 `handle_migrate` output format verbatim.
        self.summary.clone()
    }
}

impl OpsRuntime {
    #[op(
        name = "migrate",
        category = "maintenance",
        description = "Import data from a QMD SQLite database into rein memories, or reindex all memories with the current embedding model. Admin-only: CLI surface, no MCP/REST exposure.",
        mutating = true,
        cli(name = "migrate"),
    )]
    pub fn migrate(&self, params: MigrateParams) -> ReinResult<MigrateOutput> {
        let config = self.config.clone();
        if params.reindex {
            let store = config.open_store()?;
            let run = async move {
                crate::store::migrate::reindex(&store, &config)
                    .await
                    .map_err(|e| ReinError::Config(e.to_string()))
                    .map(|report| MigrateOutput {
                        summary: report.to_string(),
                        reindex: true,
                        documents_read: None,
                        items_processed: report.embedded,
                        errors: report.errors,
                    })
            };
            if let Ok(handle) = tokio::runtime::Handle::try_current() {
                tokio::task::block_in_place(|| handle.block_on(run))
            } else {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|e| ReinError::Config(format!("failed to build tokio runtime: {e}")))?;
                rt.block_on(run)
            }
        } else {
            let qmd_path = params.from_qmd.map(std::path::PathBuf::from).unwrap_or_else(|| {
                let home = std::env::var("HOME").unwrap_or_default();
                std::path::PathBuf::from(home).join(".cache/qmd/index.sqlite")
            });
            let store = config.open_store()?;
            let embedder = crate::embed::create_embedder(&config);
            let run = async move {
                crate::store::migrate::migrate_from_qmd(
                    &qmd_path,
                    &store,
                    &config,
                    embedder.as_ref(),
                )
                .await
                .map_err(|e| ReinError::Config(e.to_string()))
                .map(|report| MigrateOutput {
                    summary: report.to_string(),
                    reindex: false,
                    documents_read: Some(report.documents_read),
                    items_processed: report.chunks_created,
                    errors: report.errors,
                })
            };
            if let Ok(handle) = tokio::runtime::Handle::try_current() {
                tokio::task::block_in_place(|| handle.block_on(run))
            } else {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|e| ReinError::Config(format!("failed to build tokio runtime: {e}")))?;
                rt.block_on(run)
            }
        }
    }
}
