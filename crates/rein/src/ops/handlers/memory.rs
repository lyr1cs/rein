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

fn memory_to_json_internal(m: &crate::types::Memory) -> serde_json::Value {
    let summary_short: String = m.summary.chars().take(110).collect();
    json!({
        "id": m.id,
        "layer": format!("{}", m.layer),
        "topic": m.topic,
        "summary": m.summary,
        "summary_short": summary_short,
        "content": m.content,
        "keywords": m.keywords,
        "importance": format!("{}", m.importance),
        "source": format!("{}", m.source),
        "strength": m.strength,
        "decay_lambda": m.decay_lambda,
        "access_count": m.access_count,
        "canonical_id": m.canonical_id,
        "support_count": m.support_count,
        "status": format!("{}", m.status),
        "created_at": m.created_at.to_rfc3339(),
        "updated_at": m.updated_at.to_rfc3339(),
    })
}

impl OpsRuntime {
    #[op(
        name = "get_memory",
        category = "memory",
        description = "Fetch a single memory by ID with linked evidence.",
        rest(method = "GET", path = "/api/memories/{id}"),
        auth = "read_token",
    )]
    pub fn get_memory(&self, params: GetMemoryParams) -> ReinResult<GetMemoryOutput> {
        let id = params.id.clone();
        self.with_store(|store| {
            let m = store.get(&id)?;
            let canonical_id = m.canonical_id.clone().unwrap_or_else(|| m.id.clone());
            let mut body = memory_to_json_internal(&m);
            let evidence = store
                .list_memory_evidence(&canonical_id, 12)
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
                obj.insert("memory".to_string(), memory_to_json_internal(&m));
                obj.insert("evidence".to_string(), json!(evidence));
            }
            Ok(GetMemoryOutput(body))
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
