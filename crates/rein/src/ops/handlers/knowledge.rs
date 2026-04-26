//! Knowledge-category op handlers (Phase 2.6).
//!
//! Covers the `memoir_*` family (list / show / export / inspect / search /
//! create / add_concept / refine / link). Each op replaces one legacy MCP
//! `#[tool]` in `mcp/server.rs` and any matching derived REST branch in
//! `mcp/rest.rs`. Store access goes directly through `SqliteStore`'s public
//! memoir API (`list_memoirs`, `export_memoir`, etc.) — no shim layer.
//!
//! REST coverage parity with pre-A1 is intentional. Ops that had no legacy
//! REST surface (search / search_all / create / add_concept / refine / link)
//! stay MCP-only under A1. The `memoir_inspect` REST surface stays derived in
//! `mcp/rest.rs::handle_memoir_path` because `/api/memoirs/{name}/inspect/
//! {concept}` needs two path parameters, which exceeds the single-seg
//! contract of the current path-template framework (spec §Q2, to be revisited
//! post-v0.21).

use rein_macros::op;
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::json;

use crate::ops::{IntoCliText, IntoJson, IntoMarkdown, OpsErrorKind, OpsRuntime};
use crate::types::{ReinError, ReinResult};

// ── memoir_list ──────────────────────────────────────────────────────────────

#[derive(clap::Args, Deserialize, JsonSchema, Debug, Clone, Default)]
pub struct MemoirListParams {}

#[derive(Serialize, Clone, Debug)]
pub struct MemoirListOutput {
    pub memoirs: Vec<crate::types::Memoir>,
    #[serde(skip)]
    pub compact: bool,
}

impl IntoJson for MemoirListOutput {
    fn to_json(&self) -> serde_json::Value {
        json!({ "memoirs": self.memoirs })
    }
}

impl IntoMarkdown for MemoirListOutput {
    fn to_markdown(&self) -> String {
        if self.memoirs.is_empty() {
            return if self.compact {
                "none".to_string()
            } else {
                "No memoirs found.".to_string()
            };
        }
        let mut text = String::new();
        for m in &self.memoirs {
            if self.compact {
                text.push_str(&format!("{}:{}\n", m.name, m.description));
            } else {
                text.push_str(&format!("- {} — {}\n", m.name, m.description));
            }
        }
        text
    }
}

impl IntoCliText for MemoirListOutput {
    fn to_cli_text(&self) -> String {
        self.to_markdown()
    }
}

// ── memoir_show ──────────────────────────────────────────────────────────────

/// Parameters for the memoir_show op (`GET /api/memoirs/{name}`).
///
/// The `name` field is bound from the `{name}` path segment on the REST
/// surface (spec §4, path wins over query/body). On MCP the client passes
/// `{"name": "..."}` as a normal JSON parameter.
#[derive(clap::Args, Deserialize, JsonSchema, Debug, Clone, Default)]
pub struct MemoirShowParams {
    /// Name of the memoir to show.
    pub name: String,
}

/// Output shape for `memoir_show`. Carries both the ASCII render (MCP
/// surface parity with pre-A1) and the parsed JSON export (REST surface
/// parity). Two store calls — one per format — keep output bit-exact with
/// legacy; the op is read-only so the extra call is acceptable.
#[derive(Serialize, Clone, Debug)]
pub struct MemoirShowOutput {
    pub memoir: crate::types::Memoir,
    pub ascii: String,
    #[serde(skip)]
    pub json_value: serde_json::Value,
    #[serde(skip)]
    pub compact: bool,
}

impl IntoJson for MemoirShowOutput {
    fn to_json(&self) -> serde_json::Value {
        self.json_value.clone()
    }
}

impl IntoMarkdown for MemoirShowOutput {
    fn to_markdown(&self) -> String {
        if self.compact {
            self.ascii.clone()
        } else {
            format!(
                "Memoir: {} — {}\n\n{}",
                self.memoir.name, self.memoir.description, self.ascii
            )
        }
    }
}

impl IntoCliText for MemoirShowOutput {
    fn to_cli_text(&self) -> String {
        self.to_markdown()
    }
}

// ── memoir_export ────────────────────────────────────────────────────────────

/// Parameters for the memoir_export op (`GET /api/memoirs/{name}/export`).
///
/// The `name` field is bound from the `{name}` path segment on REST;
/// `format` comes from `?format=` query string (REST) or the MCP payload.
///
/// Phase 2.6 F2: manual deserialization preserves the pre-A1 MCP wire format
/// where clients sent `{"memoir": "..."}` while still preferring canonical
/// `name` when both keys are present. That keeps REST path binding dominant
/// even if a legacy `memoir=` query alias is supplied.
#[derive(clap::Args, JsonSchema, Debug, Clone, Default)]
pub struct MemoirExportParams {
    /// Name of the memoir to export.
    pub name: String,
    /// Export format: json, ascii, or dot (default json).
    pub format: Option<String>,
}

impl<'de> Deserialize<'de> for MemoirExportParams {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct MemoirExportParamsWire {
            name: Option<String>,
            memoir: Option<String>,
            #[serde(default)]
            format: Option<String>,
        }

        let wire = MemoirExportParamsWire::deserialize(deserializer)?;
        let name = wire
            .name
            .or(wire.memoir)
            .ok_or_else(|| serde::de::Error::missing_field("name"))?;

        Ok(Self {
            name,
            format: wire.format,
        })
    }
}

/// Output shape for `memoir_export`. REST surface returns the raw body as
/// JSON (structured) or text (ascii / dot) — `IntoJson` picks the right
/// representation by inspecting `format`. MCP returns the raw string, which
/// mirrors the pre-A1 MCP tool's behaviour (the legacy tool returned the
/// raw output unchanged).
#[derive(Serialize, Clone, Debug)]
pub struct MemoirExportOutput {
    pub format: String,
    pub output: String,
}

impl IntoJson for MemoirExportOutput {
    fn to_json(&self) -> serde_json::Value {
        // Only reached for format=json — REST dispatcher uses to_raw_response
        // first for ascii/dot, and MCP uses IntoMarkdown (raw string). Fall
        // back to a wrapped string if the stored export isn't parseable JSON
        // (mirrors legacy `api_memoir_show` tolerance).
        serde_json::from_str::<serde_json::Value>(&self.output)
            .unwrap_or_else(|_| json!({ "raw": self.output }))
    }

    fn to_raw_response(&self) -> Option<(&'static str, Vec<u8>)> {
        // Pre-A1 contract: text/plain for ascii/dot, application/json for
        // json. Phase 3 eliminated the F1 dispatcher hack (op_name guard)
        // by pushing the content-type decision into the op itself.
        match self.format.as_str() {
            "ascii" | "dot" => Some(("text/plain", self.output.as_bytes().to_vec())),
            _ => None,
        }
    }
}

impl IntoMarkdown for MemoirExportOutput {
    fn to_markdown(&self) -> String {
        // Pre-A1 MCP tool returned the raw export body verbatim; preserve.
        self.output.clone()
    }
}

impl IntoCliText for MemoirExportOutput {
    fn to_cli_text(&self) -> String {
        self.output.clone()
    }
}

// ── memoir_inspect ───────────────────────────────────────────────────────────

/// Parameters for the memoir_inspect op (MCP-only under A1).
///
/// No REST surface: `/api/memoirs/{name}/inspect/{concept}` needs two path
/// parameters and the path-template framework currently caps at one (spec
/// §Q2). That endpoint continues to be served by the derived handler at
/// `mcp/rest.rs::handle_memoir_path`; Phase 3 drift-checks treat it as a
/// derived REST arm directly counted by `count_rest_operations_in_source`.
#[derive(clap::Args, Deserialize, JsonSchema, Debug, Clone, Default)]
pub struct MemoirInspectParams {
    /// Name of the memoir.
    pub memoir: String,
    /// Name of the concept to inspect.
    pub name: String,
    /// BFS depth (default 1, max 5).
    ///
    /// Phase 2.6 F2: accepts both `{"depth": 2}` and `{"depth": "2"}` on
    /// the MCP wire for pre-A1 client compatibility.
    #[serde(
        default,
        deserialize_with = "crate::mcp::tools::deserialize_option_usize_from_string"
    )]
    pub depth: Option<usize>,
}

#[derive(Serialize, Clone, Debug)]
pub struct MemoirInspectOutput {
    pub center: crate::types::Concept,
    pub neighbors: Vec<crate::types::Concept>,
    pub links: Vec<crate::types::ConceptLink>,
    #[serde(skip)]
    pub compact: bool,
}

impl IntoJson for MemoirInspectOutput {
    fn to_json(&self) -> serde_json::Value {
        json!({
            "center": self.center,
            "neighbors": self.neighbors,
            "links": self.links,
        })
    }
}

impl IntoMarkdown for MemoirInspectOutput {
    fn to_markdown(&self) -> String {
        let mut text = String::new();
        if self.compact {
            text.push_str(&format!(
                "center:{}:c{:.1}:r{}\n",
                self.center.name, self.center.confidence, self.center.revision
            ));
            for n in &self.neighbors {
                text.push_str(&format!(
                    "neighbor:{}:c{:.1}:r{}\n",
                    n.name, n.confidence, n.revision
                ));
            }
            for l in &self.links {
                text.push_str(&format!(
                    "link:{}->{}:{}\n",
                    l.source_id, l.target_id, l.relation
                ));
            }
        } else {
            text.push_str(&format!(
                "Center: {} (conf:{:.1}, rev:{})\n  {}\n\n",
                self.center.name,
                self.center.confidence,
                self.center.revision,
                self.center.definition
            ));
            if !self.neighbors.is_empty() {
                text.push_str("Neighbors:\n");
                for n in &self.neighbors {
                    text.push_str(&format!(
                        "  - {} (conf:{:.1}, rev:{}) — {}\n",
                        n.name, n.confidence, n.revision, n.definition
                    ));
                }
            }
            if !self.links.is_empty() {
                text.push_str("\nLinks:\n");
                for l in &self.links {
                    text.push_str(&format!(
                        "  {} --{}-> {}\n",
                        l.source_id, l.relation, l.target_id
                    ));
                }
            }
        }
        text
    }
}

impl IntoCliText for MemoirInspectOutput {
    fn to_cli_text(&self) -> String {
        self.to_markdown()
    }
}

// ── memoir_search / memoir_search_all ────────────────────────────────────────
//
// Both ops are MCP-only under A1 — pre-A1 had no REST surface and Phase 2.6
// follows the "don't add what wasn't there" rule. Concept output rendering
// is shared between the two ops below via `format_concepts_*`.

#[derive(clap::Args, Deserialize, JsonSchema, Debug, Clone, Default)]
pub struct MemoirSearchParams {
    /// Name of the memoir to search in.
    pub memoir: String,
    /// Full-text search query.
    pub query: String,
    /// Maximum number of results (default 10, max 100).
    ///
    /// Phase 2.6 F2: accepts both `{"limit": 10}` and `{"limit": "10"}` on
    /// the MCP wire for pre-A1 client compatibility.
    #[serde(
        default,
        deserialize_with = "crate::mcp::tools::deserialize_option_usize_from_string"
    )]
    pub limit: Option<usize>,
}

#[derive(clap::Args, Deserialize, JsonSchema, Debug, Clone, Default)]
pub struct MemoirSearchAllParams {
    /// Full-text search query.
    pub query: String,
    /// Maximum number of results (default 10, max 100).
    #[serde(
        default,
        deserialize_with = "crate::mcp::tools::deserialize_option_usize_from_string"
    )]
    pub limit: Option<usize>,
}

#[derive(Serialize, Clone, Debug)]
pub struct ConceptSearchOutput {
    pub concepts: Vec<crate::types::Concept>,
    #[serde(skip)]
    pub compact: bool,
}

impl IntoJson for ConceptSearchOutput {
    fn to_json(&self) -> serde_json::Value {
        json!({ "concepts": self.concepts })
    }
}

impl IntoMarkdown for ConceptSearchOutput {
    fn to_markdown(&self) -> String {
        if self.concepts.is_empty() {
            return if self.compact {
                "none".to_string()
            } else {
                "No concepts found.".to_string()
            };
        }
        let mut text = String::new();
        for c in &self.concepts {
            if self.compact {
                text.push_str(&format!(
                    "{}:{}:r{}:c{:.1}\n",
                    c.name, c.definition, c.revision, c.confidence
                ));
            } else {
                text.push_str(&format!(
                    "- {} (rev:{}, conf:{:.1}) — {}\n",
                    c.name, c.revision, c.confidence, c.definition
                ));
            }
        }
        text
    }
}

impl IntoCliText for ConceptSearchOutput {
    fn to_cli_text(&self) -> String {
        self.to_markdown()
    }
}

// ── memoir_create ────────────────────────────────────────────────────────────

#[derive(clap::Args, Deserialize, JsonSchema, Debug, Clone, Default)]
pub struct MemoirCreateParams {
    /// Name for the memoir (must be unique).
    pub name: String,
    /// Optional description.
    #[serde(default)]
    pub description: Option<String>,
}

#[derive(Serialize, Clone, Debug)]
pub struct MemoirCreateOutput {
    pub id: String,
    pub name: String,
    #[serde(skip)]
    pub compact: bool,
}

impl IntoJson for MemoirCreateOutput {
    fn to_json(&self) -> serde_json::Value {
        json!({ "id": self.id, "name": self.name })
    }
}

impl IntoMarkdown for MemoirCreateOutput {
    fn to_markdown(&self) -> String {
        if self.compact {
            format!("ok:{}", self.id)
        } else {
            format!("Created memoir '{}': {}", self.name, self.id)
        }
    }
}

impl IntoCliText for MemoirCreateOutput {
    fn to_cli_text(&self) -> String {
        self.to_markdown()
    }
}

// ── memoir_add_concept ───────────────────────────────────────────────────────

#[derive(clap::Args, Deserialize, JsonSchema, Debug, Clone, Default)]
pub struct ConceptAddParams {
    /// Name of the memoir to add the concept to.
    pub memoir: String,
    /// Name of the concept.
    pub name: String,
    /// Definition of the concept.
    pub definition: String,
    /// Optional comma-separated labels.
    #[serde(default)]
    pub labels: Option<String>,
}

#[derive(Serialize, Clone, Debug)]
pub struct ConceptAddOutput {
    pub id: String,
    pub name: String,
    pub memoir: String,
    #[serde(skip)]
    pub compact: bool,
}

impl IntoJson for ConceptAddOutput {
    fn to_json(&self) -> serde_json::Value {
        json!({ "id": self.id, "name": self.name, "memoir": self.memoir })
    }
}

impl IntoMarkdown for ConceptAddOutput {
    fn to_markdown(&self) -> String {
        if self.compact {
            format!("ok:{}", self.id)
        } else {
            format!(
                "Added concept '{}' to memoir '{}': {}",
                self.name, self.memoir, self.id
            )
        }
    }
}

impl IntoCliText for ConceptAddOutput {
    fn to_cli_text(&self) -> String {
        self.to_markdown()
    }
}

// ── memoir_refine ────────────────────────────────────────────────────────────

#[derive(clap::Args, Deserialize, JsonSchema, Debug, Clone, Default)]
pub struct ConceptRefineParams {
    /// Name of the memoir containing the concept.
    pub memoir: String,
    /// Name of the concept to refine.
    pub name: String,
    /// New definition.
    pub definition: String,
}

#[derive(Serialize, Clone, Debug)]
pub struct ConceptRefineOutput {
    pub name: String,
    pub memoir: String,
    #[serde(skip)]
    pub compact: bool,
}

impl IntoJson for ConceptRefineOutput {
    fn to_json(&self) -> serde_json::Value {
        json!({ "name": self.name, "memoir": self.memoir, "refined": true })
    }
}

impl IntoMarkdown for ConceptRefineOutput {
    fn to_markdown(&self) -> String {
        if self.compact {
            format!("ok:{}", self.name)
        } else {
            format!(
                "Refined concept '{}' in memoir '{}'",
                self.name, self.memoir
            )
        }
    }
}

impl IntoCliText for ConceptRefineOutput {
    fn to_cli_text(&self) -> String {
        self.to_markdown()
    }
}

// ── memoir_link ──────────────────────────────────────────────────────────────

#[derive(clap::Args, Deserialize, JsonSchema, Debug, Clone, Default)]
pub struct LinkParams {
    /// Name of the memoir containing both concepts.
    pub memoir: String,
    /// Name of the source concept.
    pub from: String,
    /// Name of the target concept.
    pub to: String,
    /// Relation type: part_of, depends_on, related_to, contradicts, refines,
    /// alternative_to, caused_by, instance_of, superseded_by.
    pub relation: String,
}

#[derive(Serialize, Clone, Debug)]
pub struct LinkOutput {
    pub id: String,
    pub from: String,
    pub to: String,
    pub relation: String,
    #[serde(skip)]
    pub compact: bool,
}

impl IntoJson for LinkOutput {
    fn to_json(&self) -> serde_json::Value {
        json!({
            "id": self.id,
            "from": self.from,
            "to": self.to,
            "relation": self.relation,
        })
    }
}

impl IntoMarkdown for LinkOutput {
    fn to_markdown(&self) -> String {
        if self.compact {
            format!("ok:{}", self.id)
        } else {
            format!(
                "Linked '{}' --{}-> '{}': {}",
                self.from, self.relation, self.to, self.id
            )
        }
    }
}

impl IntoCliText for LinkOutput {
    fn to_cli_text(&self) -> String {
        self.to_markdown()
    }
}

// ── concept_state ────────────────────────────────────────────────────────────

#[derive(clap::Args, Deserialize, JsonSchema, Debug, Clone, Default)]
pub struct ConceptStateParams {
    /// Concept ID to fetch.
    pub concept_id: String,
    /// v0.27 R2 P2: optional query context for the Cap A adaptive gate.
    /// When BOTH `query_type` and `cluster_id` are supplied, the response
    /// consults `decide_concept_summary_quality`; if the gate returns
    /// `Skip`, `living_summary` is null and `living_summary_suppressed` is
    /// true. When either is None, the gate is bypassed (caller receives
    /// the full summary — back-compat with v0.24/v0.25/v0.26 callers).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[clap(skip)]
    pub query_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[clap(skip)]
    pub cluster_id: Option<i64>,
}

#[derive(Serialize, Clone, Debug)]
pub struct ConceptStateOutput {
    pub id: String,
    pub memoir_id: String,
    pub name: String,
    pub definition: String,
    pub revision: u32,
    pub last_episode_id: Option<String>,
    pub living_summary: Option<String>,
    pub living_summary_updated_at: Option<String>,
    pub living_summary_source_revision: Option<u32>,
    pub created_at: String,
    pub updated_at: String,
    /// v0.27 R2 P2: true when the Cap A adaptive gate suppressed
    /// `living_summary`. Default false (gate bypassed when context absent).
    #[serde(default, skip_serializing_if = "is_false")]
    pub living_summary_suppressed: bool,
    /// v0.27 R5 P2: representative cluster_id for this concept (the mode of
    /// its source memories' cluster_ids). Surfaced so GUI feedback events
    /// can route into the correct `(cluster_id, query_type)` bucket of the
    /// Cap A adaptive gate. None when the concept has no clustered source
    /// memories (the gate's cold-start fallback applies).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cluster_id: Option<i64>,
}

fn is_false(b: &bool) -> bool { !*b }

impl IntoJson for ConceptStateOutput {
    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

impl IntoMarkdown for ConceptStateOutput {
    fn to_markdown(&self) -> String {
        let mut text = format!(
            "## {} (r{})\n{}\n",
            self.name, self.revision, self.definition
        );
        if let Some(summary) = &self.living_summary {
            text.push_str(&format!("\n### Living summary\n{summary}\n"));
            if let Some(ts) = &self.living_summary_updated_at {
                text.push_str(&format!("_updated at {ts}"));
                if let Some(rev) = self.living_summary_source_revision {
                    text.push_str(&format!(", source revision r{rev}"));
                }
                text.push_str("_\n");
            }
        }
        text
    }
}

impl IntoCliText for ConceptStateOutput {
    fn to_cli_text(&self) -> String {
        self.to_markdown()
    }
}

// ── concept_summary_refresh ──────────────────────────────────────────────────

#[derive(clap::Args, Deserialize, JsonSchema, Debug, Clone, Default)]
pub struct ConceptSummaryRefreshParams {
    /// Concept ID to refresh. None = batch mode over all eligible concepts.
    #[serde(default)]
    pub concept_id: Option<String>,
    /// If true, run eligibility selection only and do not call the LLM or write.
    #[serde(default)]
    pub dry_run: Option<bool>,
}

/// Wrapper around `ops::concept_summary::ConceptSummaryOutcome` so this crate
/// can attach the `IntoJson` / `IntoMarkdown` / `IntoCliText` impls that the
/// `#[op]` adapter requires, without colliding with impls Agent 1 may add on
/// the bare outcome type.
#[derive(Serialize, Clone, Debug)]
pub struct ConceptSummaryRefreshOutput(pub crate::ops::concept_summary::ConceptSummaryOutcome);

impl IntoJson for ConceptSummaryRefreshOutput {
    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(&self.0).unwrap_or(serde_json::Value::Null)
    }
}

impl IntoMarkdown for ConceptSummaryRefreshOutput {
    fn to_markdown(&self) -> String {
        let o = &self.0;
        if o.skipped_disabled {
            return "Concept summary refresh disabled in config".to_string();
        }
        if o.skipped_no_llm {
            return "Concept summary refresh skipped: no LLM provider configured".to_string();
        }
        format!(
            "Concept summary refresh{}: attempted {}, succeeded {}, ineligible {}, llm_failed {}",
            if o.dry_run { " (dry run)" } else { "" },
            o.attempted,
            o.succeeded,
            o.skipped_not_eligible,
            o.llm_failed,
        )
    }
}

impl IntoCliText for ConceptSummaryRefreshOutput {
    fn to_cli_text(&self) -> String {
        self.to_markdown()
    }
}

impl OpsRuntime {
    #[op(
        name = "memoir_list",
        category = "knowledge",
        description = "List all memoirs (named knowledge graphs).",
        mcp(name = "rein_memoir_list"),
        rest(method = "GET", path = "/api/memoirs")
    )]
    pub fn memoir_list(&self, _params: MemoirListParams) -> ReinResult<MemoirListOutput> {
        let compact = self.compact();
        self.with_store(|store| {
            let memoirs = store.list_memoirs()?;
            Ok(MemoirListOutput { memoirs, compact })
        })
    }

    #[op(
        name = "memoir_show",
        category = "knowledge",
        description = "Show a memoir's details and list all its concepts + links.",
        mcp(name = "rein_memoir_show"),
        rest(method = "GET", path = "/api/memoirs/{name}")
    )]
    pub fn memoir_show(&self, params: MemoirShowParams) -> ReinResult<MemoirShowOutput> {
        let compact = self.compact();
        let name = params.name.clone();
        self.with_store(|store| {
            let memoir = store
                .get_memoir(&name)?
                .ok_or_else(|| ReinError::NotFound(format!("memoir '{name}' not found")))?;
            let ascii = store.export_memoir(&name, "ascii")?;
            let json_str = store.export_memoir(&name, "json")?;
            // Fall back to a wrapped raw string if the json export is not
            // valid JSON — mirrors legacy `api_memoir_show` tolerance.
            let json_value = serde_json::from_str::<serde_json::Value>(&json_str)
                .unwrap_or_else(|_| json!({ "raw": json_str }));
            Ok(MemoirShowOutput {
                memoir,
                ascii,
                json_value,
                compact,
            })
        })
    }

    #[op(
        name = "memoir_create",
        category = "knowledge",
        description = "Create a new memoir (named knowledge graph container).",
        mutating = true,
        mcp(name = "rein_memoir_create")
    )]
    pub fn memoir_create(&self, params: MemoirCreateParams) -> ReinResult<MemoirCreateOutput> {
        let compact = self.compact();
        let name = params.name.clone();
        self.with_store(|store| {
            let memoir = crate::types::Memoir {
                id: String::new(),
                name: name.clone(),
                description: params.description.unwrap_or_default(),
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
            };
            let id = store.create_memoir(memoir)?;
            Ok(MemoirCreateOutput { id, name, compact })
        })
    }

    #[op(
        name = "memoir_add_concept",
        category = "knowledge",
        description = "Add a concept (knowledge node) to a memoir with name, definition, and optional comma-separated labels.",
        mutating = true,
        mcp(name = "rein_memoir_add_concept")
    )]
    pub fn memoir_add_concept(&self, params: ConceptAddParams) -> ReinResult<ConceptAddOutput> {
        let compact = self.compact();
        let name = params.name.clone();
        let memoir = params.memoir.clone();
        let labels: Vec<String> = params
            .labels
            .as_deref()
            .map(|l| {
                l.split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            })
            .unwrap_or_default();
        self.with_store(|store| {
            let concept = crate::types::Concept {
                id: String::new(),
                memoir_id: memoir.clone(),
                name: name.clone(),
                definition: params.definition.clone(),
                labels,
                source_memory_ids: vec![],
                confidence: 0.5,
                revision: 1,
                last_episode_id: None,
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                living_summary: None,
                living_summary_updated_at: None,
                living_summary_source_revision: None,
            };
            let id = store.add_concept(concept)?;
            Ok(ConceptAddOutput {
                id,
                name,
                memoir,
                compact,
            })
        })
    }

    #[op(
        name = "memoir_refine",
        category = "knowledge",
        description = "Refine a concept: update definition, increment revision, snapshot prior revision, boost confidence.",
        mutating = true,
        mcp(name = "rein_memoir_refine")
    )]
    pub fn memoir_refine(&self, params: ConceptRefineParams) -> ReinResult<ConceptRefineOutput> {
        let compact = self.compact();
        let name = params.name.clone();
        let memoir = params.memoir.clone();
        self.with_store(|store| {
            store.refine_concept(&memoir, &name, &params.definition)?;
            Ok(ConceptRefineOutput {
                name,
                memoir,
                compact,
            })
        })
    }

    #[op(
        name = "memoir_link",
        category = "knowledge",
        description = "Create a typed relation (edge) between two concepts in the same memoir. Cross-memoir links are forbidden. Relations: part_of, depends_on, related_to, contradicts, refines, alternative_to, caused_by, instance_of, superseded_by.",
        mutating = true,
        mcp(name = "rein_memoir_link")
    )]
    pub fn memoir_link(&self, params: LinkParams) -> ReinResult<LinkOutput> {
        let compact = self.compact();
        let memoir = params.memoir.clone();
        let from_name = params.from.clone();
        let to_name = params.to.clone();
        let relation_str = params.relation.clone();
        let relation = relation_str
            .parse::<crate::types::Relation>()
            .map_err(|e| ReinError::Config(e.to_string()).with_kind(OpsErrorKind::BadRequest))?;
        self.with_store(|store| {
            // Wrap the name→id lookups + add_link in BEGIN IMMEDIATE so a
            // concurrent delete cannot make the link reference a stale ID
            // (Phase 2.4 F2 / Phase 2.5 H2 nomenclature). add_link itself
            // re-validates the memoir invariant inside the transaction.
            let conn = store.conn();
            conn.execute_batch("BEGIN IMMEDIATE")?;
            let result = (|| -> ReinResult<LinkOutput> {
                let from = store.get_concept(&memoir, &from_name)?.ok_or_else(|| {
                    ReinError::NotFound(format!(
                        "concept '{from_name}' not found in memoir '{memoir}'"
                    ))
                })?;
                let to = store.get_concept(&memoir, &to_name)?.ok_or_else(|| {
                    ReinError::NotFound(format!(
                        "concept '{to_name}' not found in memoir '{memoir}'"
                    ))
                })?;
                let link = crate::types::ConceptLink {
                    id: String::new(),
                    source_id: from.id,
                    target_id: to.id,
                    relation,
                    weight: 1.0,
                    created_at: chrono::Utc::now(),
                    valid_from: None,
                    valid_until: None,
                };
                let id = store.add_link(link)?;
                Ok(LinkOutput {
                    id,
                    from: from_name.clone(),
                    to: to_name.clone(),
                    relation: relation_str.clone(),
                    compact,
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
        name = "memoir_search",
        category = "knowledge",
        description = "Full-text search for concepts within a single memoir.",
        mcp(name = "rein_memoir_search")
    )]
    pub fn memoir_search(&self, params: MemoirSearchParams) -> ReinResult<ConceptSearchOutput> {
        let compact = self.compact();
        let limit = params.limit.unwrap_or(10).min(100);
        let memoir = params.memoir.clone();
        let query = params.query.clone();
        self.with_store(|store| {
            let concepts = store.search_concepts(&memoir, &query, limit)?;
            Ok(ConceptSearchOutput { concepts, compact })
        })
    }

    #[op(
        name = "memoir_search_all",
        category = "knowledge",
        description = "Full-text search for concepts across all memoirs.",
        mcp(name = "rein_memoir_search_all")
    )]
    pub fn memoir_search_all(
        &self,
        params: MemoirSearchAllParams,
    ) -> ReinResult<ConceptSearchOutput> {
        let compact = self.compact();
        let limit = params.limit.unwrap_or(10).min(100);
        let query = params.query.clone();
        self.with_store(|store| {
            let concepts = store.search_all_concepts(&query, limit)?;
            Ok(ConceptSearchOutput { concepts, compact })
        })
    }

    #[op(
        name = "memoir_inspect",
        category = "knowledge",
        description = "Inspect a concept's neighborhood via BFS traversal. Returns the concept, its neighbors, and connecting links up to the specified depth.",
        mcp(name = "rein_memoir_inspect")
    )]
    pub fn memoir_inspect(&self, params: MemoirInspectParams) -> ReinResult<MemoirInspectOutput> {
        let compact = self.compact();
        let depth = params.depth.unwrap_or(1).min(5);
        let memoir = params.memoir.clone();
        let name = params.name.clone();
        self.with_store(|store| {
            let (center, neighbors, links) = store.inspect_concept(&memoir, &name, depth)?;
            Ok(MemoirInspectOutput {
                center,
                neighbors,
                links,
                compact,
            })
        })
    }

    #[op(
        name = "memoir_export",
        category = "knowledge",
        description = "Export a memoir's knowledge graph. Formats: json (structured), ascii (human-readable), dot (Graphviz).",
        mcp(name = "rein_memoir_export"),
        rest(method = "GET", path = "/api/memoirs/{name}/export")
    )]
    pub fn memoir_export(&self, params: MemoirExportParams) -> ReinResult<MemoirExportOutput> {
        let format = params
            .format
            .as_deref()
            .map(|s| s.to_ascii_lowercase())
            .unwrap_or_else(|| "json".to_string());
        if !matches!(format.as_str(), "json" | "ascii" | "dot") {
            return Err(ReinError::Config(format!(
                "invalid export format '{format}' (expected one of: json, ascii, dot)"
            ))
            .with_kind(OpsErrorKind::BadRequest));
        }
        let name = params.name.clone();
        self.with_store(|store| {
            let output = store.export_memoir(&name, &format)?;
            Ok(MemoirExportOutput { format, output })
        })
    }

    #[op(
        name = "concept_state",
        category = "knowledge",
        description = "Fetch a concept's full state by ID, including its living summary (v0.24 ARS).",
        mcp(name = "rein_concept_state"),
        rest(method = "GET", path = "/api/concepts/{concept_id}/state")
    )]
    pub fn concept_state(&self, params: ConceptStateParams) -> ReinResult<ConceptStateOutput> {
        let concept_id = params.concept_id.clone();
        let query_type = params.query_type.clone();
        let cluster_id = params.cluster_id;
        let global_enabled = self.config.ars.concept_summary_enabled;
        let cold_start_n = self.config.ars.concept_summary_cold_start_n;
        self.with_store(|store| {
            let concept = store
                .get_concept_by_id(&concept_id)?
                .ok_or_else(|| ReinError::NotFound(format!("concept '{concept_id}' not found")))?;

            // v0.27 R5 P2 fix: compute the concept's representative
            // cluster_id (mode of source memories' cluster_ids) so GUI
            // surfaces can echo it through feedback events into the right
            // adaptive bucket. When source memories have no clusters or
            // the concept has none, returns None (gate cold-start path).
            let representative_cluster_id: Option<i64> = if concept.source_memory_ids.is_empty() {
                None
            } else {
                let placeholders = (0..concept.source_memory_ids.len())
                    .map(|_| "?")
                    .collect::<Vec<_>>()
                    .join(",");
                let sql = format!(
                    "SELECT cluster_id FROM memories \
                     WHERE id IN ({placeholders}) AND cluster_id IS NOT NULL"
                );
                let mut counts: std::collections::HashMap<i64, usize> =
                    std::collections::HashMap::new();
                if let Ok(mut stmt) = store.conn().prepare(&sql) {
                    let rows = stmt.query_map(
                        rusqlite::params_from_iter(concept.source_memory_ids.iter()),
                        |row| row.get::<_, Option<i64>>(0),
                    );
                    if let Ok(rows) = rows {
                        for cid in rows.flatten().flatten() {
                            *counts.entry(cid).or_insert(0) += 1;
                        }
                    }
                }
                counts.into_iter().max_by_key(|(_, n)| *n).map(|(cid, _)| cid)
            };

            // v0.27 R2 P2 fix: consult the Cap A adaptive gate when caller
            // supplies query context. v0.27 R6 P2 fix: also use the
            // computed `representative_cluster_id` as a fallback when the
            // caller supplies query_type but not cluster_id (GUI callers
            // can't know the representative cluster_id until this endpoint
            // returns it — without the fallback the gate would never fire
            // on first-fetch). Caller-supplied cluster_id still wins over
            // the computed fallback.
            //
            // v0.27 R10 P2 fix: call the gate whenever `query_type` is
            // present, even when `effective_cluster_id` is None. The
            // gate's `OperatorDisabled` branch (global flag = false) MUST
            // bind regardless of cluster context, otherwise unclustered
            // concepts could render summaries that the operator turned
            // off globally. None cluster_id falls into the gate's
            // cold-start `Yes` branch when global is true.
            let effective_cluster_id = cluster_id.or(representative_cluster_id);
            let (living_summary, suppressed) = match &query_type {
                Some(qtype) => {
                    let adaptive_state =
                        crate::store::adaptive::AdaptiveState::restore_snapshot(store.conn())
                            .unwrap_or_default();
                    match crate::ops::concept_summary::decide_concept_summary_quality(
                        global_enabled,
                        effective_cluster_id,
                        qtype,
                        Some(&adaptive_state),
                        cold_start_n,
                    ) {
                        crate::ops::concept_summary::ConceptSummaryDecision::Yes => {
                            (concept.living_summary.clone(), false)
                        }
                        crate::ops::concept_summary::ConceptSummaryDecision::Skip(_) => {
                            (None, concept.living_summary.is_some())
                        }
                    }
                }
                None => (concept.living_summary.clone(), false),
            };

            Ok(ConceptStateOutput {
                id: concept.id,
                memoir_id: concept.memoir_id,
                name: concept.name,
                definition: concept.definition,
                revision: concept.revision,
                last_episode_id: concept.last_episode_id,
                living_summary,
                living_summary_updated_at: concept
                    .living_summary_updated_at
                    .map(|dt| dt.to_rfc3339()),
                living_summary_source_revision: concept.living_summary_source_revision,
                created_at: concept.created_at.to_rfc3339(),
                updated_at: concept.updated_at.to_rfc3339(),
                living_summary_suppressed: suppressed,
                cluster_id: representative_cluster_id,
            })
        })
    }

    #[op(
        name = "concept_summary_refresh",
        category = "knowledge",
        description = "Regenerate concept living_summary via LLM. Single-concept when concept_id is set, otherwise batch over all eligible concepts. Use dry_run=true to preview.",
        mutating = true,
        mcp(name = "rein_concept_summary_refresh"),
        rest(method = "POST", path = "/api/concepts/summary_refresh"),
        auth = "mutation_marker"
    )]
    pub fn concept_summary_refresh(
        &self,
        params: ConceptSummaryRefreshParams,
    ) -> ReinResult<ConceptSummaryRefreshOutput> {
        let dry_run = params.dry_run.unwrap_or(false);
        self.set_dry_run(dry_run);
        let concept_id = params.concept_id.clone();
        let config = self.config.clone();
        self.with_store(|store| {
            let outcome = crate::ops::concept_summary::run_concept_summary(
                store,
                &config,
                concept_id.as_deref(),
                dry_run,
            )?;
            Ok(ConceptSummaryRefreshOutput(outcome))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::Arc;

    // ── F2 wire-compat regression tests ─────────────────────────────────────

    fn runtime_with_seeded_memoir(
        memoir_name: &str,
    ) -> (Arc<crate::ops::OpsRuntime>, tempfile::TempDir) {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let mut config = crate::config::ReinConfig::default();
        config.database.path = tmp
            .path()
            .join("memories.db")
            .to_string_lossy()
            .into_owned();
        let config = Arc::new(config);
        let store = config.open_store().expect("open seeded store");
        store
            .create_memoir(crate::types::Memoir {
                id: String::new(),
                name: memoir_name.to_string(),
                description: "seeded memoir".to_string(),
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
            })
            .expect("create memoir");
        store
            .add_concept(crate::types::Concept {
                id: String::new(),
                memoir_id: memoir_name.to_string(),
                name: "concept-alpha".to_string(),
                definition: "seeded concept".to_string(),
                labels: vec!["seed".to_string()],
                source_memory_ids: vec![],
                confidence: 0.9,
                revision: 1,
                last_episode_id: None,
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                living_summary: None,
                living_summary_updated_at: None,
                living_summary_source_revision: None,
            })
            .expect("add concept");
        (Arc::new(crate::ops::OpsRuntime::for_rest(config)), tmp)
    }

    #[test]
    fn memoir_export_params_accepts_legacy_memoir_field_alias() {
        let params: MemoirExportParams =
            serde_json::from_value(json!({"memoir": "foo"})).expect("alias should deserialize");
        assert_eq!(params.name, "foo");
        assert_eq!(params.format, None);
    }

    #[test]
    fn memoir_export_params_accepts_canonical_name_field() {
        let params: MemoirExportParams =
            serde_json::from_value(json!({"name": "foo", "format": "ascii"}))
                .expect("canonical field should deserialize");
        assert_eq!(params.name, "foo");
        assert_eq!(params.format.as_deref(), Some("ascii"));
    }

    #[test]
    fn memoir_export_params_prefer_canonical_name_over_legacy_alias() {
        let params: MemoirExportParams =
            serde_json::from_value(json!({"memoir": "legacy", "name": "canonical"}))
                .expect("canonical field should take precedence");
        assert_eq!(params.name, "canonical");
    }

    #[test]
    fn memoir_search_params_accepts_string_limit() {
        let params: MemoirSearchParams =
            serde_json::from_value(json!({"memoir": "m", "query": "q", "limit": "10"}))
                .expect("string limit should deserialize");
        assert_eq!(params.limit, Some(10));
    }

    #[test]
    fn memoir_search_params_accepts_number_limit() {
        let params: MemoirSearchParams =
            serde_json::from_value(json!({"memoir": "m", "query": "q", "limit": 10}))
                .expect("number limit should deserialize");
        assert_eq!(params.limit, Some(10));
    }

    #[test]
    fn memoir_search_all_params_accepts_string_limit() {
        let params: MemoirSearchAllParams =
            serde_json::from_value(json!({"query": "q", "limit": "25"}))
                .expect("string limit should deserialize");
        assert_eq!(params.limit, Some(25));
    }

    #[test]
    fn memoir_inspect_params_accepts_string_depth() {
        let params: MemoirInspectParams =
            serde_json::from_value(json!({"memoir": "m", "name": "c", "depth": "2"}))
                .expect("string depth should deserialize");
        assert_eq!(params.depth, Some(2));
    }

    #[test]
    fn memoir_inspect_params_accepts_number_depth() {
        let params: MemoirInspectParams =
            serde_json::from_value(json!({"memoir": "m", "name": "c", "depth": 2}))
                .expect("number depth should deserialize");
        assert_eq!(params.depth, Some(2));
    }

    // ── F1 IntoJson envelope shape (guards the fallback rendering) ──────────

    #[test]
    fn memoir_export_output_json_format_returns_parsed_json() {
        let out = MemoirExportOutput {
            format: "json".to_string(),
            output: r#"{"memoir": {"name": "x"}}"#.to_string(),
        };
        let value = out.to_json();
        assert_eq!(
            value
                .get("memoir")
                .and_then(|m| m.get("name"))
                .and_then(|n| n.as_str()),
            Some("x")
        );
    }

    #[test]
    fn memoir_export_output_ascii_uses_raw_response_text_plain() {
        let out = MemoirExportOutput {
            format: "ascii".to_string(),
            output: "=== Memoir: x ===\n".to_string(),
        };
        let (content_type, body) = out
            .to_raw_response()
            .expect("ascii should opt into raw response path");
        assert_eq!(content_type, "text/plain");
        assert_eq!(std::str::from_utf8(&body).unwrap(), "=== Memoir: x ===\n");
    }

    #[test]
    fn memoir_export_output_dot_uses_raw_response_text_plain() {
        let out = MemoirExportOutput {
            format: "dot".to_string(),
            output: "digraph x { }\n".to_string(),
        };
        let (content_type, body) = out
            .to_raw_response()
            .expect("dot should opt into raw response path");
        assert_eq!(content_type, "text/plain");
        assert_eq!(std::str::from_utf8(&body).unwrap(), "digraph x { }\n");
    }

    #[test]
    fn memoir_export_output_json_stays_on_jsonpath() {
        let out = MemoirExportOutput {
            format: "json".to_string(),
            output: r#"{"memoir": {"name": "x"}}"#.to_string(),
        };
        assert!(
            out.to_raw_response().is_none(),
            "json must go through IntoJson::to_json"
        );
    }

    #[tokio::test]
    async fn memoir_export_rest_invoke_uses_into_json_contract() {
        let (runtime, _tmp) = runtime_with_seeded_memoir("wire-export");
        let entry = inventory::iter::<crate::ops::OpsRestEntry>()
            .find(|e| e.op_name == "memoir_export")
            .expect("memoir_export REST entry");
        let (_status, body, content_type) = (entry.invoke)(
            runtime,
            std::collections::HashMap::from([("name", "wire-export".to_string())]),
            "format=json".to_string(),
            None,
        )
        .await
        .expect("memoir_export invoke");
        assert_eq!(content_type, "application/json");
        let value: serde_json::Value =
            serde_json::from_slice(&body).expect("memoir_export body is valid JSON");
        assert!(
            value.get("concepts").is_some(),
            "REST adapter must serialize IntoJson::to_json() output, got {value}"
        );
        assert!(
            value.get("format").is_none() && value.get("output").is_none(),
            "REST adapter must not leak the raw Serialize shape, got {value}"
        );
    }

    #[tokio::test]
    async fn memoir_export_rest_path_value_overrides_legacy_query_alias() {
        let (runtime, _tmp) = runtime_with_seeded_memoir("wire-export");
        let entry = inventory::iter::<crate::ops::OpsRestEntry>()
            .find(|e| e.op_name == "memoir_export")
            .expect("memoir_export REST entry");
        let (_status, body, content_type) = (entry.invoke)(
            runtime,
            std::collections::HashMap::from([("name", "wire-export".to_string())]),
            "memoir=wrong-memoir&format=json".to_string(),
            None,
        )
        .await
        .expect("memoir_export invoke");
        assert_eq!(content_type, "application/json");
        let value: serde_json::Value =
            serde_json::from_slice(&body).expect("memoir_export body is valid JSON");
        assert_eq!(
            value
                .get("memoir")
                .and_then(|memoir| memoir.get("name"))
                .and_then(|name| name.as_str()),
            Some("wire-export"),
            "REST path binding must beat the legacy memoir query alias"
        );
    }

    #[tokio::test]
    async fn memoir_show_mcp_invoke_uses_into_json_contract() {
        let (runtime, _tmp) = runtime_with_seeded_memoir("wire-show");
        let entry = inventory::iter::<crate::ops::OpsMcpEntry>()
            .find(|e| e.op_name == "memoir_show")
            .expect("memoir_show MCP entry");
        let out = (entry.invoke)(runtime, json!({ "name": "wire-show" }))
            .await
            .expect("memoir_show invoke");
        let value: serde_json::Value =
            serde_json::from_str(&out).expect("memoir_show MCP output is valid JSON");
        assert!(
            value.get("concepts").is_some(),
            "MCP adapter must serialize IntoJson::to_json() output, got {value}"
        );
        assert!(
            value.get("ascii").is_none(),
            "MCP adapter must not leak the raw Serialize shape, got {value}"
        );
    }

    // ── v0.24 ARS handler tests ──────────────────────────────────────────────

    /// Helper: look up the seeded concept's id from `runtime_with_seeded_memoir`.
    fn seeded_concept_id(runtime: &crate::ops::OpsRuntime, memoir_name: &str) -> String {
        let store = runtime.config().open_store().expect("open store");
        let concept = store
            .get_concept(memoir_name, "concept-alpha")
            .expect("lookup")
            .expect("concept-alpha exists");
        concept.id
    }

    #[test]
    fn concept_state_surfaces_living_summary_fields_when_present() {
        let (runtime, _tmp) = runtime_with_seeded_memoir("ars-state");
        let concept_id = seeded_concept_id(&runtime, "ars-state");
        // Seed living_summary directly — simulates a prior concept_summary_refresh.
        runtime
            .config()
            .open_store()
            .unwrap()
            .conn()
            .execute(
                "UPDATE concepts SET living_summary = ?1, \
                 living_summary_updated_at = ?2, living_summary_source_revision = ?3 \
                 WHERE id = ?4",
                rusqlite::params![
                    "rolling summary prose",
                    "2026-04-24T12:00:00Z",
                    1i64,
                    &concept_id,
                ],
            )
            .unwrap();

        let out = runtime
            .concept_state(ConceptStateParams {
                concept_id: concept_id.clone(),
                ..Default::default()
            })
            .expect("concept_state");
        assert_eq!(out.id, concept_id);
        assert_eq!(out.name, "concept-alpha");
        assert_eq!(out.living_summary.as_deref(), Some("rolling summary prose"));
        // `DateTime<Utc>::to_rfc3339()` normalizes 'Z' → '+00:00'; just check
        // the date component survives the round-trip.
        let ls_at = out.living_summary_updated_at.expect("timestamp present");
        assert!(
            ls_at.starts_with("2026-04-24T12:00:00"),
            "living_summary_updated_at did not round-trip: {ls_at}"
        );
        assert_eq!(out.living_summary_source_revision, Some(1));
    }

    #[test]
    fn concept_state_returns_none_fields_when_summary_absent() {
        let (runtime, _tmp) = runtime_with_seeded_memoir("ars-none");
        let concept_id = seeded_concept_id(&runtime, "ars-none");

        let out = runtime
            .concept_state(ConceptStateParams {
                concept_id: concept_id.clone(),
                ..Default::default()
            })
            .expect("concept_state");
        assert_eq!(out.id, concept_id);
        assert!(out.living_summary.is_none());
        assert!(out.living_summary_updated_at.is_none());
        assert!(out.living_summary_source_revision.is_none());
    }

    #[test]
    fn concept_state_returns_not_found_for_missing_id() {
        let (runtime, _tmp) = runtime_with_seeded_memoir("ars-404");
        let err = runtime
            .concept_state(ConceptStateParams {
                concept_id: "nonexistent".to_string(),
                ..Default::default()
            })
            .expect_err("missing concept must error");
        // Assert the error surface carries the not-found signal (string match
        // on the enum variant's Display is sufficient here — the REST adapter
        // maps this to 404 via OpsErrorKind).
        let err_str = format!("{err}");
        assert!(
            err_str.contains("not found"),
            "expected NotFound-flavored error, got: {err_str}"
        );
    }

    #[test]
    fn concept_summary_refresh_returns_skipped_disabled_when_ars_off() {
        let (runtime, _tmp) = runtime_with_seeded_memoir("ars-disabled");
        // Default ReinConfig has `[ars].concept_summary_enabled = false`.
        let out = runtime
            .concept_summary_refresh(ConceptSummaryRefreshParams {
                concept_id: None,
                dry_run: Some(false),
            })
            .expect("refresh call");
        assert!(
            out.0.skipped_disabled,
            "disabled ARS config must short-circuit before touching any LLM"
        );
        assert_eq!(out.0.attempted, 0);
        assert_eq!(out.0.succeeded, 0);
        assert_eq!(out.0.llm_failed, 0);
    }

    #[test]
    fn concept_summary_refresh_dry_run_on_disabled_still_short_circuits() {
        // Even dry_run must respect the enabled gate — dry_run preview is not
        // meant to bypass configuration.
        let (runtime, _tmp) = runtime_with_seeded_memoir("ars-disabled-dry");
        let out = runtime
            .concept_summary_refresh(ConceptSummaryRefreshParams {
                concept_id: None,
                dry_run: Some(true),
            })
            .expect("refresh dry_run");
        assert!(out.0.skipped_disabled);
        assert_eq!(out.0.attempted, 0);
    }
}
