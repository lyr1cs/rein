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
