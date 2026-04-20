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

use crate::ops::{IntoCliText, IntoJson, IntoMarkdown, OpsRuntime};
use crate::types::ReinResult;

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
}
