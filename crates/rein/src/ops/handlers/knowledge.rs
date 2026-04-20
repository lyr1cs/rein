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
use serde::{Deserialize, Serialize};
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
#[derive(clap::Args, Deserialize, JsonSchema, Debug, Clone, Default)]
pub struct MemoirExportParams {
    /// Name of the memoir to export.
    pub name: String,
    /// Export format: json, ascii, or dot (default json).
    #[serde(default)]
    pub format: Option<String>,
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
        if self.format == "json" {
            // Legacy REST parsed the string as JSON and returned that directly;
            // fall back to a wrapped string on parse failure.
            serde_json::from_str::<serde_json::Value>(&self.output)
                .unwrap_or_else(|_| json!({ "raw": self.output }))
        } else {
            json!({ "format": self.format, "output": self.output })
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
/// `mcp/rest.rs::handle_memoir_path` and is registered in
/// `registry::REST_OPERATIONS` to keep drift checks accurate.
#[derive(clap::Args, Deserialize, JsonSchema, Debug, Clone, Default)]
pub struct MemoirInspectParams {
    /// Name of the memoir.
    pub memoir: String,
    /// Name of the concept to inspect.
    pub name: String,
    /// BFS depth (default 1, max 5).
    #[serde(default)]
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
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(clap::Args, Deserialize, JsonSchema, Debug, Clone, Default)]
pub struct MemoirSearchAllParams {
    /// Full-text search query.
    pub query: String,
    /// Maximum number of results (default 10, max 100).
    #[serde(default)]
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

impl OpsRuntime {
    #[op(
        name = "memoir_list",
        category = "knowledge",
        description = "List all memoirs (named knowledge graphs).",
        mcp(name = "rein_memoir_list"),
        rest(method = "GET", path = "/api/memoirs"),
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
        rest(method = "GET", path = "/api/memoirs/{name}"),
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
        name = "memoir_search",
        category = "knowledge",
        description = "Full-text search for concepts within a single memoir.",
        mcp(name = "rein_memoir_search"),
    )]
    pub fn memoir_search(
        &self,
        params: MemoirSearchParams,
    ) -> ReinResult<ConceptSearchOutput> {
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
        mcp(name = "rein_memoir_search_all"),
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
        mcp(name = "rein_memoir_inspect"),
    )]
    pub fn memoir_inspect(
        &self,
        params: MemoirInspectParams,
    ) -> ReinResult<MemoirInspectOutput> {
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
        rest(method = "GET", path = "/api/memoirs/{name}/export"),
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
}
