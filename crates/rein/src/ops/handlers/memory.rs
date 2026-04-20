//! Memory-category op handlers (Phase 2.5).
//!
//! Each op in this module replaces one legacy MCP `#[tool]`, one derived REST
//! branch, and one CLI arm with a single `#[op]` registered in inventory.
//! The store's public API (`SqliteStore::delete`, etc.) is called directly so
//! no REST shim layer is needed.

use rein_macros::op;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::ops::{IntoCliText, IntoMarkdown, OpsRuntime};
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

impl OpsRuntime {
    #[op(
        name = "forget",
        category = "memory",
        description = "Permanently delete a memory by ID. Removes rows from memories + fts_memories + vec_memories + any related KG links. Irreversible.",
        mutating = true,
        cli(name = "forget"),
        mcp(name = "rein_forget"),
        rest(method = "DELETE", path = "/api/memories/{id}"),
        auth = "mutation_marker",
    )]
    pub fn forget(&self, params: ForgetParams) -> ReinResult<ForgetOutput> {
        let id = params.id.clone();
        self.with_store(|store| {
            store.delete(&id)?;
            Ok(ForgetOutput { id: id.clone(), deleted: true })
        })
    }
}
