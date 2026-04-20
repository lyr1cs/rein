//! Inventory entry definitions collected by `inventory::collect!`.
//!
//! `#[op]` emits one `OpsMetadata` per op, plus up to one `OpsCliEntry`,
//! `OpsMcpEntry`, `OpsRestEntry` based on declared surfaces. Adapters iterate
//! `inventory::iter::<Entry>` at startup to build surface registries.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::OnceLock;

use bytes::Bytes;

use crate::ops::runtime::OpsRuntime;
use crate::types::ReinResult;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpKind {
    Unary,
}

/// Per-op authorization requirement honored by the REST adapter. Pre-H3
/// audit (2026-04-19) the inventory dispatch ran **before** route-local
/// `require_mutation_marker` / `require_read_token` gates, so any migrated
/// protected route would silently bypass auth. Declaring the policy as
/// metadata pushes enforcement into the dispatcher so Phase 2.2 onwards
/// can migrate POST/DELETE ops safely.
///
/// MCP surface ignores the policy today (stdio channel is all-or-nothing
/// at the transport layer). Metadata is still recorded so the future B1
/// MCP proxy aggregator can filter tools by auth policy when federating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthPolicy {
    /// No gate. Default for read-only endpoints that expose non-sensitive
    /// data (e.g. `/api/stats`, `/api/health`, `/api/adaptive`).
    Public,
    /// Requires `x-rein-action: 1` header. Used for POST/DELETE routes
    /// that mutate server state. Mirrors `require_mutation_marker`.
    MutationMarker,
    /// Requires `x-rein-token` matching `$REIN_HTTP_TOKEN`. Used for GET
    /// routes that expose raw upstream transcripts or similar sensitive
    /// reads. Permissive when `$REIN_HTTP_TOKEN` is unset (dev mode).
    /// Mirrors `require_read_token`.
    ReadToken,
}

impl AuthPolicy {
    pub fn as_str(&self) -> &'static str {
        match self {
            AuthPolicy::Public => "public",
            AuthPolicy::MutationMarker => "mutation_marker",
            AuthPolicy::ReadToken => "read_token",
        }
    }
}

pub struct OpsCliEntry {
    pub name: &'static str,
    pub op_name: &'static str,
    pub parent: Option<&'static str>,
    pub aliases: &'static [&'static str],
    pub hidden: bool,
    pub build_clap: fn() -> clap::Command,
    pub invoke: fn(
        runtime: Arc<OpsRuntime>,
        matches: &clap::ArgMatches,
    ) -> Pin<Box<dyn Future<Output = ReinResult<String>> + Send>>,
}

pub struct OpsMcpEntry {
    pub op_name: &'static str,
    pub mcp_name: &'static str,
    pub description: &'static str,
    /// True when the op writes/mutates state (e.g. gc, dedup, consolidate,
    /// cleanup). Phase 3 removed the MCP nudge counter so this flag has no
    /// runtime reader today — it is retained as metadata for future adapters:
    /// B1 MCP proxy aggregator will filter by mutating policy when
    /// federating multiple rein instances, and external MCP clients may
    /// inspect it through `list_tools` extensions.
    pub mutating: bool,
    pub input_schema: fn() -> schemars::Schema,
    pub invoke: fn(
        runtime: Arc<OpsRuntime>,
        params_json: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = ReinResult<String>> + Send>>,
}

/// A single parsed segment of an op's REST path template.
///
/// For a path like `/api/memories/{id}` the segment list is:
/// `[Literal("api"), Literal("memories"), Param("id")]`.
///
/// T1 scope: the enum is defined here so downstream code can reference it, but
/// the `#[op]` macro still emits `&[]` for every entry. T2 will parse `{seg}`
/// occupants at macro expansion time; T3 will use the populated list in the
/// REST dispatcher's template-match pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathSegment {
    /// A fixed path component, e.g. `"api"` or `"memories"`.
    Literal(&'static str),
    /// A named path parameter placeholder, e.g. `"id"` for `{id}`.
    Param(&'static str),
}

pub struct OpsRestEntry {
    pub method: hyper::Method,
    pub path_template: &'static str,
    pub path_segments: &'static [PathSegment],
    pub op_name: &'static str,
    /// Duplicated from `OpsMetadata::auth_policy` so the REST dispatcher can
    /// pick the gate in O(1) without a metadata scan per request.
    pub auth_policy: AuthPolicy,
    pub invoke: fn(
        runtime: Arc<OpsRuntime>,
        path_values: HashMap<&'static str, String>,
        query: String,
        body: Option<Bytes>,
    ) -> Pin<
        Box<dyn Future<Output = ReinResult<(hyper::StatusCode, Bytes, &'static str)>> + Send>,
    >,
}

pub struct OpsMetadata {
    pub name: &'static str,
    pub category: &'static str,
    pub description: &'static str,
    pub kind: OpKind,
    pub mutating: bool,
    pub cli_visible: bool,
    pub mcp_visible: bool,
    pub rest_visible: bool,
    pub mcp_name: Option<&'static str>,
    pub rest_method: Option<hyper::Method>,
    pub rest_path: Option<&'static str>,
    pub auth_policy: AuthPolicy,
    pub params_schema: fn() -> schemars::Schema,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InventoryDuplicateReport {
    pub cli_names: Vec<String>,
    pub mcp_names: Vec<String>,
    pub rest_routes: Vec<String>,
}

impl InventoryDuplicateReport {
    pub fn is_empty(&self) -> bool {
        self.cli_names.is_empty() && self.mcp_names.is_empty() && self.rest_routes.is_empty()
    }

    pub fn messages(&self) -> Vec<String> {
        let mut messages = Vec::new();
        if !self.cli_names.is_empty() {
            messages.push(format!(
                "duplicate CLI command names: {}",
                self.cli_names.join(", ")
            ));
        }
        if !self.mcp_names.is_empty() {
            messages.push(format!(
                "duplicate MCP tool names: {}",
                self.mcp_names.join(", ")
            ));
        }
        if !self.rest_routes.is_empty() {
            messages.push(format!(
                "duplicate REST method/path registrations: {}",
                self.rest_routes.join(", ")
            ));
        }
        messages
    }
}

pub fn duplicate_report() -> InventoryDuplicateReport {
    let mut report = InventoryDuplicateReport::default();

    let mut cli_seen: HashMap<&'static str, &'static str> = HashMap::new();
    for entry in inventory::iter::<OpsCliEntry>() {
        if let Some(prev) = cli_seen.insert(entry.name, entry.op_name) {
            report
                .cli_names
                .push(format!("{} ({} vs {})", entry.name, prev, entry.op_name));
        }
    }

    let mut mcp_seen: HashMap<&'static str, &'static str> = HashMap::new();
    for entry in inventory::iter::<OpsMcpEntry>() {
        if let Some(prev) = mcp_seen.insert(entry.mcp_name, entry.op_name) {
            report.mcp_names.push(format!(
                "{} ({} vs {})",
                entry.mcp_name, prev, entry.op_name
            ));
        }
    }

    let mut rest_seen: HashMap<(String, &'static str), &'static str> = HashMap::new();
    for entry in inventory::iter::<OpsRestEntry>() {
        let key = (entry.method.as_str().to_string(), entry.path_template);
        if let Some(prev) = rest_seen.insert(key.clone(), entry.op_name) {
            report.rest_routes.push(format!(
                "{} {} ({} vs {})",
                key.0, key.1, prev, entry.op_name
            ));
        }
    }

    report.cli_names.sort();
    report.cli_names.dedup();
    report.mcp_names.sort();
    report.mcp_names.dedup();
    report.rest_routes.sort();
    report.rest_routes.dedup();
    report
}

pub fn ensure_unique_registrations() {
    static RESULT: OnceLock<Result<(), String>> = OnceLock::new();
    let result = RESULT.get_or_init(|| {
        let report = duplicate_report();
        if report.is_empty() {
            Ok(())
        } else {
            Err(report.messages().join("; "))
        }
    });
    if let Err(message) = result {
        panic!("duplicate inventory registration detected: {message}");
    }
}

inventory::collect!(OpsCliEntry);
inventory::collect!(OpsMcpEntry);
inventory::collect!(OpsRestEntry);
inventory::collect!(OpsMetadata);
