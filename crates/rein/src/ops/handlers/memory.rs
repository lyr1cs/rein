//! Memory-category op handlers (Phase 2.5).
//!
//! Each op in this module replaces one legacy MCP `#[tool]`, one derived REST
//! branch, and one CLI arm with a single `#[op]` registered in inventory.
//! The store's public API (`SqliteStore::delete`, etc.) is called directly so
//! no REST shim layer is needed.

use rein_macros::op;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::ops::{IntoCliText, IntoJson, IntoMarkdown, OpsRuntime};
use crate::types::{MemoryStore, ReinResult};

// ── forget ───────────────────────────────────────────────────────────────────

/// Parameters for the forget op (`DELETE /api/memories/{id}`).
///
/// The `id` field is bound from the `{id}` path segment on the REST surface
/// (spec §4, path wins over query/body). On the MCP surface, clients pass
/// `{"id": "..."}` as a normal JSON parameter. On the CLI surface, `id` is a
/// required positional argument.
#[derive(clap::Args, Deserialize, JsonSchema, Debug, Clone, Default)]
pub struct ForgetParams {
    /// The memory ID to permanently delete.
    #[arg()]
    pub id: String,
}

/// Output shape for the forget op.
#[derive(Serialize, Clone, Debug)]
pub struct ForgetOutput {
    pub id: String,
    pub deleted: bool,
}

impl IntoJson for ForgetOutput {
    fn to_json(&self) -> serde_json::Value {
        json!({ "id": self.id, "deleted": self.deleted })
    }
}

impl IntoMarkdown for ForgetOutput {
    fn to_markdown(&self) -> String {
        // Compact contract: matches the pre-A1 legacy MCP compact branch
        // verbatim (`format!("ok:{id}")`) so MCP callers that parse this
        // string continue to work.
        format!("ok:{}", self.id)
    }
}

impl IntoCliText for ForgetOutput {
    fn to_cli_text(&self) -> String {
        // Verbatim match of the pre-A1 `handle_forget` CLI output so shell
        // scripts that parse this text continue to work.
        format!("Deleted memory: {}", self.id)
    }
}

// ── list_topics ─────────────────────────────────────────────────────────────

#[derive(clap::Args, Deserialize, JsonSchema, Debug, Clone, Default)]
pub struct ListTopicsParams {}

#[derive(Serialize, Clone, Debug)]
pub struct ListTopicsOutput {
    pub topics: Vec<String>,
    #[serde(skip)]
    pub compact: bool,
}

impl IntoJson for ListTopicsOutput {
    fn to_json(&self) -> serde_json::Value {
        json!({ "topics": self.topics })
    }
}

impl IntoMarkdown for ListTopicsOutput {
    fn to_markdown(&self) -> String {
        crate::mcp::compact::format_topics(&self.topics, self.compact)
    }
}

impl IntoCliText for ListTopicsOutput {
    fn to_cli_text(&self) -> String {
        crate::mcp::compact::format_topics(&self.topics, self.compact)
    }
}

// ── recent ──────────────────────────────────────────────────────────────────

#[derive(clap::Args, Deserialize, JsonSchema, Debug, Clone, Default)]
pub struct RecentParams {
    /// Maximum number of recent memories to return (default 20, max 100).
    #[arg(short, long, default_value = "20")]
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Serialize, Clone, Debug)]
pub struct RecentOutput {
    pub memories: Vec<crate::types::Memory>,
    #[serde(skip)]
    pub compact: bool,
}

impl IntoJson for RecentOutput {
    fn to_json(&self) -> serde_json::Value {
        let items: Vec<serde_json::Value> = self
            .memories
            .iter()
            .map(crate::mcp::rest::memory_to_json)
            .collect();
        json!({ "memories": items })
    }
}

fn format_recent_line(m: &crate::types::Memory, compact: bool) -> String {
    if compact {
        format!("[{}] {}", m.topic, m.summary)
    } else {
        let age = chrono::Utc::now().signed_duration_since(m.created_at);
        let age_str = if age.num_days() > 0 {
            format!("{}d ago", age.num_days())
        } else if age.num_hours() > 0 {
            format!("{}h ago", age.num_hours())
        } else {
            format!("{}m ago", age.num_minutes())
        };
        format!(
            "[{}] {} ({}, {}, str:{:.2})\n  id: {}",
            m.topic, m.summary, m.importance, age_str, m.strength, m.id
        )
    }
}

impl IntoMarkdown for RecentOutput {
    fn to_markdown(&self) -> String {
        if self.memories.is_empty() {
            return "No memories found.".to_string();
        }
        self.memories
            .iter()
            .map(|m| format_recent_line(m, self.compact))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl IntoCliText for RecentOutput {
    fn to_cli_text(&self) -> String {
        if self.memories.is_empty() {
            return "No memories found.".to_string();
        }
        self.memories
            .iter()
            .map(|m| {
                if self.compact {
                    format!("[{}] {}", m.topic, m.summary)
                } else {
                    let age = chrono::Utc::now().signed_duration_since(m.created_at);
                    let age_str = if age.num_days() > 0 {
                        format!("{}d ago", age.num_days())
                    } else if age.num_hours() > 0 {
                        format!("{}h ago", age.num_hours())
                    } else {
                        format!("{}m ago", age.num_minutes())
                    };
                    format!(
                        "[{}] {} ({}, {})",
                        m.topic, m.summary, m.importance, age_str
                    )
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

// ── recall ──────────────────────────────────────────────────────────────────

#[derive(clap::Args, Deserialize, JsonSchema, Debug, Clone, Default)]
pub struct RecallMemoryParams {
    /// The search query string.
    #[arg()]
    pub query: String,
    /// Optional topic filter.
    #[arg(short, long)]
    #[serde(default)]
    pub topic: Option<String>,
    /// Optional keyword filter.
    #[arg(short, long)]
    #[serde(default)]
    pub keyword: Option<String>,
    /// Maximum number of results to return (default 10, max 200).
    #[arg(short, long)]
    #[serde(default)]
    pub limit: Option<usize>,
    /// Filter memories created after this date (YYYY-MM-DD or RFC3339).
    #[arg(long)]
    #[serde(default)]
    pub from: Option<String>,
    /// Filter memories created before this date.
    #[arg(long)]
    #[serde(default)]
    pub to: Option<String>,
    /// Override query expansion: true=force, false=disable, null=use config default.
    #[arg(long)]
    #[serde(default)]
    pub expand: Option<bool>,
    /// Cap B (v0.25): when true, the LLM synthesizes a concise narrative over the
    /// top results and returns it alongside the normal results list. Opt-in via the
    /// `[ars].recall_synthesis_enabled` config flag — without that flag this param
    /// has no effect (the outcome will report `skipped_disabled`).
    #[arg(long)]
    #[serde(default)]
    pub synthesize: Option<bool>,
}

#[derive(Serialize, Clone, Debug)]
pub struct RecallMemoryOutput {
    pub results: Vec<crate::search::recall::RecallResult>,
    pub route: String,
    pub request_id: String,
    /// Cap B (v0.25): present only when `synthesize=true` was requested.
    /// `None` means caller did not request synthesis (default behavior).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub synthesis: Option<crate::ops::recall_synthesis::RecallSynthesisOutcome>,
    #[serde(skip)]
    pub compact: bool,
}

impl IntoJson for RecallMemoryOutput {
    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

impl IntoMarkdown for RecallMemoryOutput {
    fn to_markdown(&self) -> String {
        let mut text = crate::mcp::compact::format_recall_results_mcp(&self.results, self.compact);
        if text.is_empty() {
            text = if self.compact {
                "none".to_string()
            } else {
                "No memories found.".to_string()
            };
        }
        if !self.compact && !self.results.is_empty() {
            text = format!(
                "[route: {} | request_id: {}] {}",
                self.route, self.request_id, text
            );
        }
        // Cap B: prepend synthesis narrative when present so MCP clients see
        // the answer first, then the supporting memory list. Skip in compact
        // mode (the compact contract is a single short line).
        if !self.compact {
            if let Some(synth) = self.synthesis.as_ref() {
                if let Some(prose) = synth.synthesis.as_deref() {
                    if !prose.is_empty() {
                        text = format!("[synthesis] {prose}\n\n{text}");
                    }
                }
            }
        }
        text
    }
}

impl IntoCliText for RecallMemoryOutput {
    fn to_cli_text(&self) -> String {
        let body = crate::mcp::compact::format_recall_results(&self.results, self.compact);
        // Cap B: prepend the synthesized narrative when present so the CLI
        // user sees the LLM-produced answer first, then the supporting list.
        // Without this prepend, `rein recall --synthesize` would pay the LLM
        // latency/API cost without any visible benefit on the CLI surface
        // (Codex audit Round 1 finding P2-2).
        if !self.compact {
            if let Some(synth) = self.synthesis.as_ref() {
                if let Some(prose) = synth.synthesis.as_deref() {
                    if !prose.is_empty() {
                        return format!("[synthesis]\n{prose}\n\n{body}");
                    }
                }
            }
        }
        body
    }
}

// ── store ───────────────────────────────────────────────────────────────────

fn deserialize_keywords<'de, D>(deserializer: D) -> Result<Option<Vec<String>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StringOrVec {
        Vec(Vec<String>),
        String(String),
    }
    match Option::<StringOrVec>::deserialize(deserializer)? {
        None => Ok(None),
        Some(StringOrVec::Vec(v)) => Ok(Some(v)),
        Some(StringOrVec::String(s)) => Ok(Some(
            s.split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect(),
        )),
    }
}

#[derive(clap::Args, Deserialize, JsonSchema, Debug, Clone, Default)]
pub struct StoreMemoryParams {
    /// Topic/category for the memory.
    #[arg(short, long)]
    pub topic: String,
    /// The content to store.
    #[arg(short, long)]
    pub content: String,
    /// Importance level: low, medium, high, or critical (default medium).
    #[arg(short = 'I', long, default_value = "medium")]
    #[serde(default)]
    pub importance: Option<String>,
    /// Keywords (comma-separated on CLI, array or comma-string on MCP).
    #[arg(short, long, value_delimiter = ',')]
    #[serde(default, deserialize_with = "deserialize_keywords")]
    pub keywords: Option<Vec<String>>,
}

#[derive(Serialize, Clone, Debug)]
pub struct StoreMemoryOutput {
    pub id: String,
    #[serde(skip)]
    pub compact: bool,
}

impl IntoJson for StoreMemoryOutput {
    fn to_json(&self) -> serde_json::Value {
        json!({ "id": self.id })
    }
}

impl IntoMarkdown for StoreMemoryOutput {
    fn to_markdown(&self) -> String {
        crate::mcp::compact::format_store_result(&self.id, self.compact)
    }
}

impl IntoCliText for StoreMemoryOutput {
    fn to_cli_text(&self) -> String {
        crate::mcp::compact::format_store_result(&self.id, self.compact)
    }
}

// ── update ──────────────────────────────────────────────────────────────────

#[derive(clap::Args, Deserialize, JsonSchema, Debug, Clone, Default)]
pub struct UpdateParams {
    /// The memory ID to update.
    #[arg()]
    pub id: String,
    /// New content for the memory.
    #[arg(short, long)]
    pub content: String,
    /// New importance level (low, medium, high, critical).
    #[arg(short = 'I', long)]
    #[serde(default)]
    pub importance: Option<String>,
}

#[derive(Serialize, Clone, Debug)]
pub struct UpdateOutput {
    pub id: String,
    pub updated: bool,
}

impl IntoJson for UpdateOutput {
    fn to_json(&self) -> serde_json::Value {
        json!({ "id": self.id, "updated": self.updated })
    }
}

impl IntoMarkdown for UpdateOutput {
    fn to_markdown(&self) -> String {
        format!("ok:{}", self.id)
    }
}

impl IntoCliText for UpdateOutput {
    fn to_cli_text(&self) -> String {
        format!("Updated memory: {}", self.id)
    }
}

// ── timeline ────────────────────────────────────────────────────────────────

#[derive(clap::Args, Deserialize, JsonSchema, Debug, Clone, Default)]
pub struct TimelineParams {
    /// Start date (YYYY-MM-DD or RFC3339).
    #[arg(long)]
    #[serde(default)]
    pub from: Option<String>,
    /// End date (YYYY-MM-DD or RFC3339).
    #[arg(long)]
    #[serde(default)]
    pub to: Option<String>,
    /// Maximum entries (default 20, max 200).
    #[arg(short, long)]
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Serialize, Clone, Debug)]
pub struct TimelineOutput {
    pub events: Vec<(chrono::DateTime<chrono::Utc>, String)>,
}

impl IntoJson for TimelineOutput {
    fn to_json(&self) -> serde_json::Value {
        json!({
            "events": self.events.iter().map(|(dt, desc)| json!({
                "timestamp": dt.to_rfc3339(),
                "description": desc,
            })).collect::<Vec<_>>()
        })
    }
}

impl IntoMarkdown for TimelineOutput {
    fn to_markdown(&self) -> String {
        if self.events.is_empty() {
            return "No events found in the specified range.".to_string();
        }
        self.events
            .iter()
            .map(|(dt, desc)| format!("{} {}", dt.format("%Y-%m-%d %H:%M"), desc))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl IntoCliText for TimelineOutput {
    fn to_cli_text(&self) -> String {
        IntoMarkdown::to_markdown(self)
    }
}

// ── concept_history ─────────────────────────────────────────────────────────

#[derive(clap::Args, Deserialize, JsonSchema, Debug, Clone, Default)]
pub struct ConceptHistoryParams {
    /// Memoir containing the concept.
    #[arg(long)]
    pub memoir: String,
    /// Name of the concept.
    #[arg(long)]
    pub name: String,
    /// Maximum revisions to return (default 10, max 100).
    #[arg(short, long)]
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Serialize, Clone, Debug)]
pub struct ConceptHistoryOutput {
    pub current_name: String,
    pub current_revision: u32,
    pub current_confidence: f32,
    pub current_definition: String,
    /// v0.24 ARS: rolling LLM-generated summary of the concept's current state.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub living_summary: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub living_summary_updated_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub living_summary_source_revision: Option<u32>,
    pub history: Vec<ConceptRevisionSummary>,
}

#[derive(Serialize, Clone, Debug)]
pub struct ConceptRevisionSummary {
    pub revision: u32,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub episode_id: Option<String>,
    pub definition: String,
}

impl IntoJson for ConceptHistoryOutput {
    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

impl IntoMarkdown for ConceptHistoryOutput {
    fn to_markdown(&self) -> String {
        let mut lines = vec![format!(
            "## {} (current: r{}, confidence: {:.2})\n{}\n",
            self.current_name,
            self.current_revision,
            self.current_confidence,
            self.current_definition
        )];
        if let Some(summary) = &self.living_summary {
            lines.push(format!("### Living summary\n{summary}\n"));
        }
        if self.history.is_empty() {
            lines.push("No revision history (concept has not been refined yet).".to_string());
        } else {
            lines.push(format!(
                "### Revision History ({} entries)\n",
                self.history.len()
            ));
            for rev in &self.history {
                let ep = rev.episode_id.as_deref().unwrap_or("none");
                lines.push(format!(
                    "- **r{}** ({}) [episode: {}]\n  {}\n",
                    rev.revision,
                    rev.created_at.format("%Y-%m-%d %H:%M"),
                    ep,
                    rev.definition.chars().take(200).collect::<String>()
                ));
            }
        }
        lines.join("\n")
    }
}

impl IntoCliText for ConceptHistoryOutput {
    fn to_cli_text(&self) -> String {
        IntoMarkdown::to_markdown(self)
    }
}

fn parse_dt_start(s: &str) -> Option<chrono::DateTime<chrono::Utc>> {
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(s) {
        return Some(dt.with_timezone(&chrono::Utc));
    }
    if let Ok(date) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        let dt = date.and_hms_opt(0, 0, 0)?;
        return Some(chrono::DateTime::<chrono::Utc>::from_naive_utc_and_offset(
            dt,
            chrono::Utc,
        ));
    }
    None
}

fn parse_dt_end(s: &str) -> Option<chrono::DateTime<chrono::Utc>> {
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(s) {
        return Some(dt.with_timezone(&chrono::Utc));
    }
    if let Ok(date) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        let dt = date.and_hms_opt(23, 59, 59)?;
        return Some(chrono::DateTime::<chrono::Utc>::from_naive_utc_and_offset(
            dt,
            chrono::Utc,
        ));
    }
    None
}

// ── get_memory ──────────────────────────────────────────────────────────────

#[derive(clap::Args, Deserialize, JsonSchema, Debug, Clone, Default)]
pub struct GetMemoryParams {
    /// The memory ID to fetch.
    #[arg()]
    pub id: String,
}

/// Wraps the rich JSON shape returned by the pre-A1 derived handler:
/// flattened memory fields + a nested `memory` key + `evidence` array.
#[derive(Serialize, Clone, Debug)]
#[serde(transparent)]
pub struct GetMemoryOutput(pub serde_json::Value);

impl IntoJson for GetMemoryOutput {
    fn to_json(&self) -> serde_json::Value {
        self.0.clone()
    }
}

impl IntoMarkdown for GetMemoryOutput {
    fn to_markdown(&self) -> String {
        let id = self.0.get("id").and_then(|v| v.as_str()).unwrap_or("?");
        let summary = self.0.get("summary").and_then(|v| v.as_str()).unwrap_or("");
        format!("memory {id}: {summary}")
    }
}

impl IntoCliText for GetMemoryOutput {
    fn to_cli_text(&self) -> String {
        serde_json::to_string_pretty(&self.0).unwrap_or_default()
    }
}

impl OpsRuntime {
    #[op(
        name = "get_memory",
        category = "memory",
        description = "Fetch a single memory by ID with linked evidence.",
        rest(method = "GET", path = "/api/memories/{id}"),
        auth = "read_token"
    )]
    pub fn get_memory(&self, params: GetMemoryParams) -> ReinResult<GetMemoryOutput> {
        let id = params.id.clone();
        self.with_store(|store| {
            let m = store.get(&id)?;
            let canonical_id = m.canonical_id.clone().unwrap_or_else(|| m.id.clone());
            let mut body = crate::mcp::rest::memory_to_json(&m);
            // `evidence` is preview-capped at 200 rows; `evidence_total` is
            // the un-truncated row count so the UI can label honestly
            // (e.g. "Showing 200 of 543") instead of claiming the preview
            // size IS the total. Cap chosen well above the typical evidence
            // count while still bounded for absurd canonicals.
            const EVIDENCE_PREVIEW_CAP: usize = 200;
            let evidence_total = store.count_memory_evidence(&canonical_id).unwrap_or(0);
            let evidence = store
                .list_memory_evidence(&canonical_id, EVIDENCE_PREVIEW_CAP)
                .unwrap_or_default()
                .into_iter()
                .filter(|item| item.memory_id.as_deref() != Some(canonical_id.as_str()))
                .map(|item| {
                    json!({
                        "id": item.id,
                        "canonical_id": item.canonical_id,
                        "memory_id": item.memory_id,
                        "source_topic": item.source_topic,
                        "summary": item.summary,
                        "content": item.content,
                        "keywords": item.keywords,
                        "source": format!("{}", item.source),
                        "created_at": item.created_at.to_rfc3339(),
                        "imported_at": item.imported_at.to_rfc3339(),
                    })
                })
                .collect::<Vec<_>>();
            if let Some(obj) = body.as_object_mut() {
                obj.insert("memory".to_string(), crate::mcp::rest::memory_to_json(&m));
                obj.insert("evidence".to_string(), json!(evidence));
                obj.insert("evidence_total".to_string(), json!(evidence_total));
            }
            Ok(GetMemoryOutput(body))
        })
    }

    #[op(
        name = "recall",
        category = "memory",
        description = "Search and recall memories by semantic query. 3-channel waterfall: FTS5 (Tantivy BM25) → HNSW vectors → Gemini API, fused via RRF/CC with query expansion, LLM reranking, and Ebbinghaus decay weighting.",
        cli(name = "recall"),
        mcp(name = "rein_recall")
    )]
    pub fn recall(&self, params: RecallMemoryParams) -> ReinResult<RecallMemoryOutput> {
        let limit = params.limit.unwrap_or(10).min(200);
        let time_from = params.from.as_deref().and_then(parse_dt_start);
        let time_to = params.to.as_deref().and_then(parse_dt_end);
        let request_id = ulid::Ulid::new().to_string();
        let query = params.query.clone();
        let topic = params.topic.clone();
        let keyword = params.keyword.clone();
        let expand = params.expand;
        let synthesize = params.synthesize;
        let compact = self.compact();

        let route =
            crate::search::classify::classify(&query, time_from.is_some(), time_to.is_some());
        let route_name = route.query_type.to_string();
        let req_id_clone = request_id.clone();

        // Phase 1: pull results AND a snapshot of the adaptive state out of
        // `with_store`. The adaptive snapshot is what `run_recall_synthesis`
        // needs for the v0.26 D-direction per-query gate (see
        // `decide_synthesize`); loading it inside the store closure keeps
        // every recall call observing a single consistent state across the
        // search and the synthesis decision.
        //
        // Synthesis itself is called OUTSIDE the store closure because it
        // (a) needs no DB access and (b) wraps a `block_in_place` call to
        // drive the LLM that should not be nested inside any store
        // transaction guard.
        let (results, adaptive_state, ars_parameter_policy_canary) = self.with_store(|store| {
            let results = crate::search::recall::recall_temporal_with_request_id(
                store,
                &self.config,
                &query,
                topic.as_deref(),
                keyword.as_deref(),
                limit,
                time_from,
                time_to,
                expand,
                false,
                Some(req_id_clone.clone()),
            )
            .map_err(|e| crate::types::ReinError::Config(format!("{e}")))?;
            // v0.26 D direction: load the adaptive snapshot once per
            // recall (matches the established pattern in
            // `ops/resummerize.rs:150`, `ops/concept_summary.rs:95`,
            // `ops/dedup.rs:992`). `unwrap_or_default` mirrors fresh-install
            // behavior; the gate's cold-start branches handle the empty
            // case gracefully.
            let adaptive_state =
                crate::store::adaptive::AdaptiveState::restore_snapshot(store.conn())
                    .unwrap_or_default();
            let ars_parameter_policy_canary =
                crate::ops::ars_tuning::parameter_policy_allows_runtime(
                    store.conn(),
                    &self.config,
                    &adaptive_state,
                );
            Ok((results, adaptive_state, ars_parameter_policy_canary))
        })?;

        // Phase 2: optional synthesis (Cap B / v0.26 D direction). Returns
        // `None` when caller did not request synthesis; returns
        // `Some(outcome)` otherwise — the outcome's `skipped_*` flags
        // explain what happened. The fresh `adaptive_state` snapshot drives
        // `decide_synthesize`'s per-cluster gate; passing `None` would
        // force every recall through the global flag and erase the
        // per-cluster signal entirely.
        //
        // v0.26.1: pass the classified `query_type.synthesis_bucket_label()`
        // (capitalised "Semantic"|"Episodic"|...) so the gate reads from
        // the same per-cluster bucket the M1 consumer writes into. v0.26.0
        // hardcoded "Semantic" inside `run_recall_synthesis`, which silently
        // misrouted every non-Semantic query.
        let synthesis = crate::ops::recall_synthesis::run_recall_synthesis_with_policy(
            &results,
            &query,
            &self.config,
            synthesize,
            route.query_type.synthesis_bucket_label(),
            Some(&adaptive_state),
            None,
            ars_parameter_policy_canary,
        );

        Ok(RecallMemoryOutput {
            results,
            route: route_name,
            request_id,
            synthesis,
            compact,
        })
    }

    #[op(
        name = "store",
        category = "memory",
        description = "Store a new memory with topic, content, importance, and keywords. Automatically deduplicates against existing memories.",
        mutating = true,
        cli(name = "store"),
        mcp(name = "rein_store")
    )]
    pub fn store_memory(&self, params: StoreMemoryParams) -> ReinResult<StoreMemoryOutput> {
        if params.content.len() > 100_000 {
            return Err(crate::types::ReinError::Config(
                "content too large (max 100KB)".to_string(),
            )
            .with_kind(crate::types::OpsErrorKind::BadRequest));
        }
        let importance: crate::types::Importance =
            match params.importance.as_deref().unwrap_or("medium").parse() {
                Ok(imp) => imp,
                Err(_) => {
                    return Err(crate::types::ReinError::Config(format!(
                        "invalid importance {:?}: must be one of low, medium, high, critical",
                        params.importance.as_deref().unwrap_or("")
                    ))
                    .with_kind(crate::types::OpsErrorKind::BadRequest))
                }
            };
        let keywords = params.keywords.unwrap_or_default();
        let memory = crate::ops::build_memory(
            &self.config,
            params.topic,
            params.content.clone(),
            importance,
            keywords,
            crate::types::Source::Manual,
        );
        let config = self.config.clone();
        let compact = self.compact();
        self.with_store(|store| {
            let id = crate::ops::store_memory(store, &config, memory)?;
            Ok(StoreMemoryOutput { id, compact })
        })
    }

    #[op(
        name = "update",
        category = "memory",
        description = "Update the content of an existing memory by ID. Optionally reassign importance, which adjusts the decay layer.",
        mutating = true,
        cli(name = "update"),
        mcp(name = "rein_update")
    )]
    pub fn update(&self, params: UpdateParams) -> ReinResult<UpdateOutput> {
        let base_lambda = self.config.decay.base_lambda;
        let id = params.id.clone();
        // M2: validate importance early before opening a transaction.
        let new_importance: Option<crate::types::Importance> = match params.importance.as_deref() {
            None => None,
            Some(s) => match s.parse::<crate::types::Importance>() {
                Ok(imp) => Some(imp),
                Err(_) => {
                    return Err(crate::types::ReinError::Config(format!(
                        "invalid importance {:?}: must be one of low, medium, high, critical",
                        s
                    ))
                    .with_kind(crate::types::OpsErrorKind::BadRequest))
                }
            },
        };
        self.with_store(|store| {
            // H2: wrap read-modify-write in BEGIN IMMEDIATE to prevent lost updates.
            let conn = store.conn();
            conn.execute_batch("BEGIN IMMEDIATE")?;
            let result = (|| -> ReinResult<UpdateOutput> {
                let mut memory = store.get(&id)?;
                memory.content = params.content.clone();
                memory.summary = params
                    .content
                    .chars()
                    .take(crate::types::SUMMARY_MAX_CHARS)
                    .collect();
                if let Some(imp) = new_importance {
                    memory.importance = imp;
                    memory.layer = imp.auto_layer();
                    memory.decay_lambda = base_lambda * imp.decay_factor();
                }
                memory.updated_at = chrono::Utc::now();
                store.update(&memory)?;
                Ok(UpdateOutput {
                    id: id.clone(),
                    updated: true,
                })
            })();
            match result {
                Ok(out) => {
                    conn.execute_batch("COMMIT")?;
                    Ok(out)
                }
                Err(e) => {
                    let _ = conn.execute_batch("ROLLBACK");
                    Err(e)
                }
            }
        })
    }

    #[op(
        name = "timeline",
        category = "knowledge",
        description = "Chronological timeline of knowledge events: episodes, concept revisions, memory creation. Supports date-range filtering.",
        cli(name = "timeline"),
        mcp(name = "rein_timeline")
    )]
    pub fn timeline(&self, params: TimelineParams) -> ReinResult<TimelineOutput> {
        let limit = params.limit.unwrap_or(20).min(200);
        let from = params.from.as_deref().and_then(parse_dt_start);
        let to = params.to.as_deref().and_then(parse_dt_end);
        if params.from.is_some() && from.is_none() {
            return Err(crate::types::ReinError::Config(format!(
                "invalid 'from' date format: {:?}",
                params.from
            ))
            .with_kind(crate::types::OpsErrorKind::BadRequest));
        }
        if params.to.is_some() && to.is_none() {
            return Err(crate::types::ReinError::Config(format!(
                "invalid 'to' date format: {:?}",
                params.to
            ))
            .with_kind(crate::types::OpsErrorKind::BadRequest));
        }

        self.with_store(|store| {
            let mut events: Vec<(chrono::DateTime<chrono::Utc>, String)> = Vec::new();

            let episodes = match (from, to) {
                (Some(f), Some(t)) => store.get_episodes_in_range(f, t)?,
                (Some(f), None) => store.get_episodes_in_range(
                    f,
                    chrono::Utc::now() + chrono::Duration::days(1),
                )?,
                (None, Some(t)) => store.get_episodes_in_range(
                    chrono::DateTime::parse_from_rfc3339("2000-01-01T00:00:00Z")
                        .unwrap()
                        .with_timezone(&chrono::Utc),
                    t,
                )?,
                (None, None) => store.list_episodes(limit)?,
            };
            for ep in &episodes {
                let decisions = if ep.decisions.is_empty() {
                    String::new()
                } else {
                    format!(" | decisions: {}", ep.decisions.join(", "))
                };
                events.push((
                    ep.created_at,
                    format!(
                        "[episode] {} — {} concepts, {} memories{}",
                        ep.title,
                        ep.concept_ids.len(),
                        ep.memory_ids.len(),
                        decisions
                    ),
                ));
            }

            {
                let mut where_clauses = Vec::new();
                let mut param_values: Vec<String> = Vec::new();
                if let Some(f) = from {
                    param_values.push(f.to_rfc3339());
                    where_clauses.push(format!("r.created_at >= ?{}", param_values.len()));
                }
                if let Some(t) = to {
                    param_values.push(t.to_rfc3339());
                    where_clauses.push(format!("r.created_at <= ?{}", param_values.len()));
                }
                let where_str = if where_clauses.is_empty() {
                    String::new()
                } else {
                    format!(" WHERE {}", where_clauses.join(" AND "))
                };
                let rev_sql = format!(
                    "SELECT r.revision, r.definition, r.created_at, c.name, c.memoir_id \
                     FROM concept_revisions r JOIN concepts c ON r.concept_id = c.id{} \
                     ORDER BY r.created_at DESC LIMIT {}",
                    where_str, limit
                );
                if let Ok(mut stmt) = store.conn().prepare(&rev_sql) {
                    let extract =
                        |row: &rusqlite::Row| -> rusqlite::Result<(u32, String, String, String)> {
                            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
                        };
                    let collected: Vec<_> = match param_values.len() {
                        0 => stmt
                            .query_map([], extract)
                            .map(|rows| rows.filter_map(|r| r.ok()).collect())
                            .unwrap_or_default(),
                        1 => stmt
                            .query_map(rusqlite::params![param_values[0]], extract)
                            .map(|rows| rows.filter_map(|r| r.ok()).collect())
                            .unwrap_or_default(),
                        _ => stmt
                            .query_map(
                                rusqlite::params![param_values[0], param_values[1]],
                                extract,
                            )
                            .map(|rows| rows.filter_map(|r| r.ok()).collect())
                            .unwrap_or_default(),
                    };
                    for (rev, def, created_str, name) in collected {
                        if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(&created_str) {
                            let short_def: String = def.chars().take(80).collect();
                            events.push((
                                dt.with_timezone(&chrono::Utc),
                                format!("[revision] {} r{}: {}", name, rev, short_def),
                            ));
                        }
                    }
                }
            }

            let mem_events: Vec<(chrono::DateTime<chrono::Utc>, String, String, String)> =
                if from.is_some() || to.is_some() {
                    let mut where_parts = Vec::new();
                    let mut param_values: Vec<String> = Vec::new();
                    if let Some(f) = from {
                        where_parts.push(format!("created_at >= ?{}", param_values.len() + 1));
                        param_values.push(f.to_rfc3339());
                    }
                    if let Some(t) = to {
                        where_parts.push(format!("created_at <= ?{}", param_values.len() + 1));
                        param_values.push(t.to_rfc3339());
                    }
                    let sql = format!(
                        "SELECT topic, summary, created_at FROM memories WHERE {} ORDER BY created_at DESC LIMIT {}",
                        where_parts.join(" AND "),
                        limit
                    );
                    let mut stmt = store
                        .conn()
                        .prepare(&sql)
                        .map_err(crate::types::ReinError::Database)?;
                    let rows: Vec<_> = match param_values.len() {
                        1 => stmt
                            .query_map(rusqlite::params![param_values[0]], |row| {
                                Ok((
                                    row.get::<_, String>(0)?,
                                    row.get::<_, String>(1)?,
                                    row.get::<_, String>(2)?,
                                ))
                            })
                            .map_err(crate::types::ReinError::Database)?
                            .filter_map(|r| r.ok())
                            .collect(),
                        2 => stmt
                            .query_map(
                                rusqlite::params![param_values[0], param_values[1]],
                                |row| {
                                    Ok((
                                        row.get::<_, String>(0)?,
                                        row.get::<_, String>(1)?,
                                        row.get::<_, String>(2)?,
                                    ))
                                },
                            )
                            .map_err(crate::types::ReinError::Database)?
                            .filter_map(|r| r.ok())
                            .collect(),
                        _ => Vec::new(),
                    };
                    rows.into_iter()
                        .filter_map(|(topic, summary, created_str)| {
                            chrono::DateTime::parse_from_rfc3339(&created_str).ok().map(
                                |dt| {
                                    (
                                        dt.with_timezone(&chrono::Utc),
                                        topic.clone(),
                                        summary.clone(),
                                        created_str.clone(),
                                    )
                                },
                            )
                        })
                        .collect()
                } else {
                    store
                        .recent(limit)?
                        .iter()
                        .map(|m| {
                            (
                                m.created_at,
                                m.topic.clone(),
                                m.summary.clone(),
                                m.created_at.to_rfc3339(),
                            )
                        })
                        .collect()
                };
            for (dt, topic, summary, _) in &mem_events {
                events.push((*dt, format!("[memory] [{}] {}", topic, summary)));
            }

            events.sort_by_key(|e| std::cmp::Reverse(e.0));
            events.truncate(limit);
            Ok(TimelineOutput { events })
        })
    }

    #[op(
        name = "concept_history",
        category = "knowledge",
        description = "Show revision history of a concept — when and how its definition changed over time.",
        cli(name = "concept-history"),
        mcp(name = "rein_concept_history")
    )]
    pub fn concept_history(
        &self,
        params: ConceptHistoryParams,
    ) -> ReinResult<ConceptHistoryOutput> {
        let limit = params.limit.unwrap_or(10).min(100);
        self.with_store(|store| {
            let current = store
                .get_concept(&params.memoir, &params.name)?
                .ok_or_else(|| {
                    crate::types::ReinError::NotFound(format!(
                        "concept '{}' not found",
                        params.name
                    ))
                })?;
            let history = store.get_concept_history(&params.memoir, &params.name, limit)?;
            Ok(ConceptHistoryOutput {
                current_name: current.name,
                current_revision: current.revision,
                current_confidence: current.confidence,
                current_definition: current.definition,
                living_summary: current.living_summary,
                living_summary_updated_at: current
                    .living_summary_updated_at
                    .map(|dt| dt.to_rfc3339()),
                living_summary_source_revision: current.living_summary_source_revision,
                history: history
                    .into_iter()
                    .map(|rev| ConceptRevisionSummary {
                        revision: rev.revision,
                        created_at: rev.created_at,
                        episode_id: rev.episode_id,
                        definition: rev.definition,
                    })
                    .collect(),
            })
        })
    }

    #[op(
        name = "list_topics",
        category = "memory",
        description = "List all unique topics across stored memories.",
        cli(name = "topics"),
        mcp(name = "rein_list_topics"),
        rest(method = "GET", path = "/api/topics")
    )]
    pub fn list_topics(&self, _params: ListTopicsParams) -> ReinResult<ListTopicsOutput> {
        let compact = self.compact();
        self.with_store(|store| {
            let topics = store.list_topics()?;
            Ok(ListTopicsOutput { topics, compact })
        })
    }

    #[op(
        name = "recent",
        category = "memory",
        description = "Show the most recently created memories ordered by creation time.",
        cli(name = "recent"),
        mcp(name = "rein_recent"),
        rest(method = "GET", path = "/api/recent")
    )]
    pub fn recent(&self, params: RecentParams) -> ReinResult<RecentOutput> {
        let limit = params.limit.unwrap_or(20).clamp(1, 100);
        let compact = self.compact();
        self.with_store(|store| {
            let memories = store.recent(limit)?;
            Ok(RecentOutput { memories, compact })
        })
    }

    #[op(
        name = "forget",
        category = "memory",
        description = "Permanently delete a memory by ID. Removes rows from memories + fts_memories + vec_memories + any related KG links. Irreversible.",
        mutating = true,
        cli(name = "forget"),
        mcp(name = "rein_forget"),
        rest(method = "DELETE", path = "/api/memories/{id}"),
        auth = "mutation_marker"
    )]
    pub fn forget(&self, params: ForgetParams) -> ReinResult<ForgetOutput> {
        let id = params.id.clone();
        self.with_store(|store| {
            store.delete(&id)?;
            Ok(ForgetOutput {
                id: id.clone(),
                deleted: true,
            })
        })
    }
}
