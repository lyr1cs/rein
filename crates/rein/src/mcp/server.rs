use std::sync::atomic::{AtomicU32, Ordering};

use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{ServerCapabilities, ServerInfo};
use rmcp::{tool, tool_router, ServerHandler, ServiceExt};

use crate::config::ReinConfig;
use crate::mcp::tools::*;
use crate::store::SqliteStore;
use crate::types::*;

/// MCP server for rein memory system.
///
/// Creates a new SqliteStore (SQLite connection) per request instead of sharing
/// a single Mutex<SqliteStore>. This eliminates the Mutex bottleneck and enables
/// concurrent read operations. SQLite with WAL mode + FULL_MUTEX handles write
/// serialization internally.
pub struct ReinServer {
    config: ReinConfig,
    non_store_count: AtomicU32,
    tool_router: ToolRouter<Self>,
}

impl std::fmt::Debug for ReinServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReinServer")
            .field("compact", &self.config.server.compact)
            .finish_non_exhaustive()
    }
}

impl ReinServer {
    /// Create a new ReinServer.
    ///
    /// Stores the config so each request can open its own connection via
    /// `config.open_store()`, eliminating the Mutex bottleneck.
    pub fn new(config: ReinConfig) -> Self {
        Self {
            config,
            non_store_count: AtomicU32::new(0),
            tool_router: Self::tool_router(),
        }
    }

    fn compact(&self) -> bool {
        self.config.server.compact
    }

    /// Open a per-request SqliteStore, run a synchronous closure, return the result.
    /// Each call gets its own SQLite connection — no Mutex needed.
    /// SQLite WAL mode handles concurrent readers and serializes writers.
    fn with_store<F, R>(&self, f: F) -> Result<R, String>
    where
        F: FnOnce(&SqliteStore) -> Result<R, ReinError>,
    {
        let store = self.config.open_store().map_err(|e| format!("{e}"))?;
        f(&store).map_err(|e| format!("{e}"))
    }

    /// Dispatch an inventory-registered op by MCP tool name without requiring
    /// a `RequestContext`. Applies the same counter policy and nudge-skip logic
    /// as `call_tool`. Used by regression tests that cannot construct a full
    /// rmcp `RequestContext`.
    #[cfg(test)]
    pub async fn dispatch_inventory(
        &self,
        tool_name: &str,
        args: serde_json::Value,
    ) -> Option<Result<String, String>> {
        let entry = inventory::iter::<crate::ops::OpsMcpEntry>()
            .find(|e| e.mcp_name == tool_name)?;

        if entry.mutating {
            self.non_store_count.store(0, Ordering::Relaxed);
        } else {
            self.non_store_count.fetch_add(1, Ordering::Relaxed);
        }

        let runtime = std::sync::Arc::new(crate::ops::OpsRuntime::for_mcp(
            std::sync::Arc::new(self.config.clone()),
        ));
        // M1: mirror compact flag so test-path dispatch honours compact rendering.
        runtime.set_compact(self.compact());
        Some((entry.invoke)(runtime, args).await.map_err(|e| e.to_string()))
    }

    /// Read the current non-store counter value. Used by regression tests.
    #[cfg(test)]
    pub fn non_store_count(&self) -> u32 {
        self.non_store_count.load(Ordering::Relaxed)
    }
}

const HTTP_SESSION_COOKIE: &str = "rein_http_token";

fn constant_time_eq(left: &str, right: &str) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.bytes()
        .zip(right.bytes())
        .fold(0u8, |acc, (x, y)| acc | (x ^ y))
        == 0
}

fn cookie_value(headers: &hyper::HeaderMap, name: &str) -> Option<String> {
    let cookies = headers.get("cookie")?.to_str().ok()?;
    cookies.split(';').find_map(|part| {
        let (key, value) = part.trim().split_once('=')?;
        if key.trim() == name {
            Some(value.trim().to_string())
        } else {
            None
        }
    })
}

fn request_has_valid_http_auth(headers: &hyper::HeaderMap, expected: &str) -> bool {
    let auth_header = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let expected_str = format!("Bearer {expected}");
    if constant_time_eq(auth_header, &expected_str) {
        return true;
    }
    cookie_value(headers, HTTP_SESSION_COOKIE)
        .map(|value| constant_time_eq(&value, expected))
        .unwrap_or(false)
}

#[tool_router]
impl ReinServer {

    // rein_ingest_session migrated to #[op] (see ops/handlers/session.rs).
    // Dispatch flows through the OpsMcpEntry inventory; legacy tool_router
    // delegation below is still used for tools that haven't migrated yet.


    // rein_stats + rein_health migrated to #[op] (see ops/handlers/diagnostics.rs).
    // Dispatch is handled by the custom impl ServerHandler below, which checks
    // the OpsMcpEntry inventory before delegating to tool_router for legacy tools.

    // rein_consolidate migrated to #[op] inventory (see ops/handlers/maintenance.rs).
    // Dispatch is handled by the custom impl ServerHandler below, which checks
    // the OpsMcpEntry inventory before delegating to tool_router for legacy tools.

    // ===== Knowledge Graph (Memoir/Concept/Link) tools =====

    /// Create a new memoir (knowledge graph container).
    #[tool(
        name = "rein_memoir_create",
        description = "Create a new memoir (named knowledge graph). Use to organize concepts and their relationships."
    )]
    fn rein_memoir_create(&self, Parameters(params): Parameters<MemoirCreateParams>) -> String {
        self.non_store_count.store(0, Ordering::Relaxed);
        let compact = self.compact();

        let memoir = crate::types::Memoir {
            id: String::new(),
            name: params.name.clone(),
            description: params.description.unwrap_or_default(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };

        let result = self.with_store(|store| store.create_memoir(memoir));

        match result {
            Ok(id) => {
                if compact {
                    format!("ok:{id}")
                } else {
                    format!("Created memoir '{}': {id}", params.name)
                }
            }
            Err(e) => format!("Error: {e}"),
        }
    }

    // rein_memoir_list migrated to #[op] inventory (see ops/handlers/knowledge.rs).

    /// Show a memoir and all its concepts.
    #[tool(
        name = "rein_memoir_show",
        description = "Show memoir details and list all concepts within it."
    )]
    fn rein_memoir_show(&self, Parameters(params): Parameters<MemoirShowParams>) -> String {
        self.non_store_count.fetch_add(1, Ordering::Relaxed);
        let compact = self.compact();

        let result = self.with_store(|store| {
            let memoir = store.get_memoir(&params.name)?.ok_or_else(|| {
                ReinError::NotFound(format!("memoir '{}' not found", params.name))
            })?;
            let export = store.export_memoir(&params.name, "ascii")?;
            Ok((memoir, export))
        });

        match result {
            Ok((memoir, export)) => {
                if compact {
                    export
                } else {
                    format!(
                        "Memoir: {} — {}\n\n{}",
                        memoir.name, memoir.description, export
                    )
                }
            }
            Err(e) => format!("Error: {e}"),
        }
    }

    /// Add a concept to a memoir.
    #[tool(
        name = "rein_memoir_add_concept",
        description = "Add a concept (knowledge node) to a memoir with name, definition, and optional labels."
    )]
    fn rein_memoir_add_concept(&self, Parameters(params): Parameters<ConceptAddParams>) -> String {
        self.non_store_count.store(0, Ordering::Relaxed);
        let compact = self.compact();

        let labels: Vec<String> = params
            .labels
            .map(|l| {
                l.split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            })
            .unwrap_or_default();

        let concept = crate::types::Concept {
            id: String::new(),
            memoir_id: params.memoir.clone(),
            name: params.name.clone(),
            definition: params.definition,
            labels,
            source_memory_ids: vec![],
            confidence: 0.5,
            revision: 1,
            last_episode_id: None,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };

        let result = self.with_store(|store| store.add_concept(concept));

        match result {
            Ok(id) => {
                if compact {
                    format!("ok:{id}")
                } else {
                    format!(
                        "Added concept '{}' to memoir '{}': {id}",
                        params.name, params.memoir
                    )
                }
            }
            Err(e) => format!("Error: {e}"),
        }
    }

    /// Refine a concept's definition.
    #[tool(
        name = "rein_memoir_refine",
        description = "Refine a concept: update definition, increment revision, boost confidence."
    )]
    fn rein_memoir_refine(&self, Parameters(params): Parameters<ConceptRefineParams>) -> String {
        self.non_store_count.store(0, Ordering::Relaxed);
        let compact = self.compact();

        let result = self.with_store(|store| {
            store.refine_concept(&params.memoir, &params.name, &params.definition)
        });

        match result {
            Ok(()) => {
                if compact {
                    format!("ok:{}", params.name)
                } else {
                    format!(
                        "Refined concept '{}' in memoir '{}'",
                        params.name, params.memoir
                    )
                }
            }
            Err(e) => format!("Error: {e}"),
        }
    }

    /// Search concepts within a memoir.
    #[tool(
        name = "rein_memoir_search",
        description = "Full-text search for concepts within a memoir."
    )]
    fn rein_memoir_search(&self, Parameters(params): Parameters<ConceptSearchParams>) -> String {
        self.non_store_count.fetch_add(1, Ordering::Relaxed);
        let compact = self.compact();
        let limit = params.limit.unwrap_or(10).min(100);

        let result =
            self.with_store(|store| store.search_concepts(&params.memoir, &params.query, limit));

        match result {
            Ok(concepts) => {
                if concepts.is_empty() {
                    return if compact {
                        "none".to_string()
                    } else {
                        "No concepts found.".to_string()
                    };
                }
                let mut text = String::new();
                for c in &concepts {
                    if compact {
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
            Err(e) => format!("Error: {e}"),
        }
    }

    /// Search concepts across all memoirs.
    #[tool(
        name = "rein_memoir_search_all",
        description = "Full-text search for concepts across all memoirs."
    )]
    fn rein_memoir_search_all(
        &self,
        Parameters(params): Parameters<ConceptSearchAllParams>,
    ) -> String {
        self.non_store_count.fetch_add(1, Ordering::Relaxed);
        let compact = self.compact();
        let limit = params.limit.unwrap_or(10).min(100);

        let result = self.with_store(|store| store.search_all_concepts(&params.query, limit));

        match result {
            Ok(concepts) => {
                if concepts.is_empty() {
                    return if compact {
                        "none".to_string()
                    } else {
                        "No concepts found.".to_string()
                    };
                }
                let mut text = String::new();
                for c in &concepts {
                    if compact {
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
            Err(e) => format!("Error: {e}"),
        }
    }

    /// Create a typed relation between two concepts.
    #[tool(
        name = "rein_memoir_link",
        description = "Create a typed relation (edge) between two concepts in a memoir. Relations: part_of, depends_on, related_to, contradicts, refines, alternative_to, caused_by, instance_of, superseded_by."
    )]
    fn rein_memoir_link(&self, Parameters(params): Parameters<LinkParams>) -> String {
        self.non_store_count.store(0, Ordering::Relaxed);
        let compact = self.compact();

        let relation = match params.relation.parse::<crate::types::Relation>() {
            Ok(r) => r,
            Err(e) => return format!("Error: {e}"),
        };

        let result = self.with_store(|store| {
            let from = store
                .get_concept(&params.memoir, &params.from)?
                .ok_or_else(|| {
                    ReinError::NotFound(format!(
                        "concept '{}' not found in memoir '{}'",
                        params.from, params.memoir
                    ))
                })?;
            let to = store
                .get_concept(&params.memoir, &params.to)?
                .ok_or_else(|| {
                    ReinError::NotFound(format!(
                        "concept '{}' not found in memoir '{}'",
                        params.to, params.memoir
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
            store.add_link(link)
        });

        match result {
            Ok(id) => {
                if compact {
                    format!("ok:{id}")
                } else {
                    format!(
                        "Linked '{}' --{}-> '{}': {id}",
                        params.from, params.relation, params.to
                    )
                }
            }
            Err(e) => format!("Error: {e}"),
        }
    }

    /// Inspect a concept's neighborhood via BFS.
    #[tool(
        name = "rein_memoir_inspect",
        description = "Inspect a concept's neighborhood via BFS traversal. Returns the concept, its neighbors, and connecting links up to the specified depth."
    )]
    fn rein_memoir_inspect(&self, Parameters(params): Parameters<InspectParams>) -> String {
        self.non_store_count.fetch_add(1, Ordering::Relaxed);
        let compact = self.compact();
        let depth = params.depth.unwrap_or(1).min(5);

        let result =
            self.with_store(|store| store.inspect_concept(&params.memoir, &params.name, depth));

        match result {
            Ok((center, neighbors, links)) => {
                let mut text = String::new();
                if compact {
                    text.push_str(&format!(
                        "center:{}:c{:.1}:r{}\n",
                        center.name, center.confidence, center.revision
                    ));
                    for n in &neighbors {
                        text.push_str(&format!(
                            "neighbor:{}:c{:.1}:r{}\n",
                            n.name, n.confidence, n.revision
                        ));
                    }
                    for l in &links {
                        text.push_str(&format!(
                            "link:{}->{}:{}\n",
                            l.source_id, l.target_id, l.relation
                        ));
                    }
                } else {
                    text.push_str(&format!(
                        "Center: {} (conf:{:.1}, rev:{})\n  {}\n\n",
                        center.name, center.confidence, center.revision, center.definition
                    ));
                    if !neighbors.is_empty() {
                        text.push_str("Neighbors:\n");
                        for n in &neighbors {
                            text.push_str(&format!(
                                "  - {} (conf:{:.1}, rev:{}) — {}\n",
                                n.name, n.confidence, n.revision, n.definition
                            ));
                        }
                    }
                    if !links.is_empty() {
                        text.push_str("\nLinks:\n");
                        for l in &links {
                            text.push_str(&format!(
                                "  {} --{}-> {}\n",
                                l.source_id, l.relation, l.target_id
                            ));
                        }
                    }
                }
                text
            }
            Err(e) => format!("Error: {e}"),
        }
    }

    /// Export a memoir graph.
    #[tool(
        name = "rein_memoir_export",
        description = "Export a memoir's knowledge graph. Formats: json (structured), ascii (human-readable), dot (Graphviz)."
    )]
    fn rein_memoir_export(&self, Parameters(params): Parameters<ExportParams>) -> String {
        self.non_store_count.fetch_add(1, Ordering::Relaxed);
        let format = params.format.as_deref().unwrap_or("json");

        let result = self.with_store(|store| store.export_memoir(&params.memoir, format));

        match result {
            Ok(output) => output,
            Err(e) => format!("Error: {e}"),
        }
    }

    // rein_dedup migrated to #[op] inventory (see ops/handlers/maintenance.rs).
    // POST /api/dedup REST surface added. auth = "mutation_marker".

    // rein_cleanup migrated to #[op] inventory (see ops/handlers/maintenance.rs).
    // POST /api/cleanup REST surface added. auth = "mutation_marker".

    // rein_recent migrated to #[op] inventory (see ops/handlers/memory.rs).

    // rein_organize migrated to #[op] inventory (see ops/handlers/maintenance.rs).

    // rein_timeline migrated to #[op] inventory (see ops/handlers/memory.rs).
    // Note: derived REST /api/timeline remains — JSON shape differs; Phase 3 cleanup.

    // rein_concept_history migrated to #[op] inventory (see ops/handlers/memory.rs).
}

impl ServerHandler for ReinServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions("rein: Multi-source cross-validated memory for AI agents. Use rein_store to save important context and rein_recall to search memories.")
    }

    async fn list_tools(
        &self,
        _request: Option<rmcp::model::PaginatedRequestParams>,
        _context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<rmcp::model::ListToolsResult, rmcp::ErrorData> {
        let mut tools = self.tool_router.list_all();
        for entry in inventory::iter::<crate::ops::OpsMcpEntry>() {
            tools.push(inventory_entry_to_tool(entry));
        }
        tools.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(rmcp::model::ListToolsResult {
            tools,
            meta: None,
            next_cursor: None,
        })
    }

    async fn call_tool(
        &self,
        request: rmcp::model::CallToolRequestParams,
        context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<rmcp::model::CallToolResult, rmcp::ErrorData> {
        let tool_name = request.name.as_ref();

        // Inventory-dispatched (migrated #[op]) tools: apply mutating-aware
        // counter policy and return structured JSON without the nudge banner.
        // Legacy #[tool] handlers manage non_store_count themselves; they also
        // append the nudge via maybe_nudge at the end of their own return string.
        // Concatenating the banner to a serialized JSON payload (inventory path)
        // would produce invalid JSON — so the nudge is intentionally skipped here.
        if let Some(entry) = inventory::iter::<crate::ops::OpsMcpEntry>()
            .find(|e| e.mcp_name == tool_name)
        {
            // Mutating ops reset the counter (write); read-ish ops increment it.
            if entry.mutating {
                self.non_store_count.store(0, Ordering::Relaxed);
            } else {
                self.non_store_count.fetch_add(1, Ordering::Relaxed);
            }

            let args_value = request
                .arguments
                .clone()
                .map(serde_json::Value::Object)
                .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
            let runtime = std::sync::Arc::new(crate::ops::OpsRuntime::for_mcp(
                std::sync::Arc::new(self.config.clone()),
            ));
            // M1: propagate the server-level compact flag so the macro-emitted
            // MCP output branch renders IntoMarkdown when compact is set.
            runtime.set_compact(self.compact());
            return match (entry.invoke)(runtime, args_value).await {
                Ok(body) => Ok(rmcp::model::CallToolResult::success(vec![
                    rmcp::model::Content::text(body),
                ])),
                Err(e) => Ok(rmcp::model::CallToolResult::error(vec![
                    rmcp::model::Content::text(e.to_string()),
                ])),
            };
        }

        let tcc = rmcp::handler::server::tool::ToolCallContext::new(self, request, context);
        self.tool_router.call(tcc).await
    }

    fn get_tool(&self, name: &str) -> Option<rmcp::model::Tool> {
        if let Some(entry) = inventory::iter::<crate::ops::OpsMcpEntry>()
            .find(|e| e.mcp_name == name)
        {
            return Some(inventory_entry_to_tool(entry));
        }
        self.tool_router.get(name).cloned()
    }
}

/// Convert an `OpsMcpEntry` into the `rmcp::model::Tool` that `list_tools`
/// returns. Schemars emits a JSON Value; rmcp's Tool wants a JsonObject
/// (map). Fallback to an empty object schema if the emitted schema isn't
/// a JSON object (shouldn't happen given our macro emission, but guard).
fn inventory_entry_to_tool(entry: &crate::ops::OpsMcpEntry) -> rmcp::model::Tool {
    let schema = (entry.input_schema)();
    let value: serde_json::Value = schema.into();
    let obj = match value {
        serde_json::Value::Object(m) => m,
        _ => {
            let mut m = serde_json::Map::new();
            m.insert(
                "type".to_string(),
                serde_json::Value::String("object".to_string()),
            );
            m
        }
    };
    rmcp::model::Tool::new(entry.mcp_name, entry.description, obj)
}

/// Spawn background warmup task for embedding cache pre-computation.
fn spawn_background_warmup(config: &ReinConfig) {
    let warmup_config = config.clone();
    // SqliteStore is not Send, so we must run the entire warmup (including async embed)
    // inside spawn_blocking. Use a dedicated current-thread runtime for the async parts.
    tokio::task::spawn_blocking(move || {
        if let Ok(store) = warmup_config.open_store() {
            // Drain any `pending_grayzone_jobs` rows left over from a crash
            // between the store COMMIT and the file-queue enqueue in the
            // previous session — before doing index warmup.
            match store.drain_pending_grayzone_jobs(&warmup_config) {
                Ok(count) if count > 0 => tracing::info!(
                    "drained {count} pending grayzone jobs recovered from prior session"
                ),
                Ok(_) => {}
                Err(e) => tracing::warn!("pending grayzone drain failed: {e}"),
            }
            // Repair any session_artifacts orphaned by a crash between
            // create_episode and link_session_artifact_episode in the
            // previous session (B3 #18).
            match store.repair_orphan_artifact_episode_links() {
                Ok(count) if count > 0 => tracing::info!(
                    "repaired {count} orphan session_artifact → episode links"
                ),
                Ok(_) => {}
                Err(e) => tracing::warn!("artifact-episode repair failed: {e}"),
            }
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build();
            if let Ok(rt) = rt {
                rt.block_on(crate::search::warmup::warmup(&store, &warmup_config));
            }
        }
    });
}

/// Start the MCP server over stdio.
pub async fn run_stdio(config: ReinConfig) -> anyhow::Result<()> {
    spawn_background_warmup(&config);

    let server = ReinServer::new(config);

    let transport = rmcp::transport::io::stdio();
    let service = server
        .serve(transport)
        .await
        .map_err(|e| anyhow::anyhow!("MCP server init error: {e}"))?;
    service
        .waiting()
        .await
        .map_err(|e| anyhow::anyhow!("MCP server error: {e}"))?;
    Ok(())
}

/// Start the MCP server over HTTP (Streamable HTTP / SSE).
/// Accessible via Tailscale or LAN for remote memory queries.
pub async fn run_http(config: ReinConfig) -> anyhow::Result<()> {
    spawn_background_warmup(&config);

    use http_body_util::BodyExt;
    use rmcp::transport::streamable_http_server::{
        session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
    };
    use std::sync::Arc;
    use tokio_util::sync::CancellationToken;

    let bind = format!("{}:{}", config.server.sse_bind, config.server.sse_port);
    let config_clone = config.clone();

    // Bearer token authentication
    let auth_token = std::env::var("REIN_HTTP_TOKEN")
        .ok()
        .filter(|token| !token.trim().is_empty());
    let allow_loopback_unauth = config.server.allow_unauthenticated_loopback
        && (config.server.sse_bind == "127.0.0.1"
            || config.server.sse_bind == "::1"
            || config.server.sse_bind == "localhost");
    if auth_token.is_none() && !allow_loopback_unauth {
        return Err(anyhow::anyhow!(
            "REIN_HTTP_TOKEN must be set for HTTP/SSE access on '{}'. \
             Set REIN_HTTP_TOKEN=<secret> or explicitly opt into unauthenticated loopback with [server].allow_unauthenticated_loopback=true",
            config.server.sse_bind
        ));
    }

    let cancel = CancellationToken::new();

    let session_manager = Arc::new(LocalSessionManager::default());
    let http_config = StreamableHttpServerConfig::default()
        .with_stateful_mode(true)
        .with_cancellation_token(cancel.clone());

    // NOTE: Each HTTP session creates its own ReinServer with a separate SqliteStore.
    // The Mutex only serializes within a single session (MCP handles one request at a time).
    // Cross-session concurrency is handled by having independent connections.
    let service = StreamableHttpService::new(
        move || {
            let server = ReinServer::new(config_clone.clone());
            Ok(server)
        },
        session_manager,
        http_config,
    );

    let listener = match tokio::net::TcpListener::bind(&bind).await {
        Ok(l) => l,
        Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => {
            eprintln!("Error: port {} is already in use.", config.server.sse_port);
            eprintln!("Another rein instance may be running. Check with:");
            eprintln!("  lsof -i :{}", config.server.sse_port);
            eprintln!("Kill it first, or use a different port via REIN_SSE_PORT.");
            return Err(anyhow::anyhow!(
                "port {} already in use",
                config.server.sse_port
            ));
        }
        Err(e) => return Err(e.into()),
    };
    tracing::info!("rein HTTP server listening on {bind}");
    eprintln!("rein HTTP server listening on http://{bind}/mcp");
    if config.server.gui_enabled {
        eprintln!("Neural Wiki GUI available at http://{bind}/");
    }
    // Write PID file for service management (rein gui on/off, rein dashboard).
    let _ = crate::service::write_pid("gui");

    let rest_config = config.clone();
    let service = hyper::service::service_fn(move |req: hyper::Request<_>| {
        let svc = service.clone();
        let token = auth_token.clone();
        let cfg = rest_config.clone();
        async move {
            // Check bearer token for API/MCP paths only (GUI static assets are public
            // so the SPA can bootstrap and show a token input dialog).
            //
            // DELETE /api/session is exempted so a browser holding a stale/invalid
            // cookie can clear it without being 401-blocked — otherwise the SPA
            // would be stuck resending the bad cookie until manual browser-data wipe.
            let path = req.uri().path();
            let method = req.method();
            let is_clear_session = method == hyper::Method::DELETE && path == "/api/session";
            let needs_auth =
                !is_clear_session && (path.starts_with("/api/") || path.starts_with("/mcp"));
            if needs_auth {
                if let Some(ref expected) = token {
                    if !request_has_valid_http_auth(req.headers(), expected) {
                        return Ok::<_, std::convert::Infallible>(
                            hyper::Response::builder()
                                .status(401)
                                .body(
                                    http_body_util::Full::new(bytes::Bytes::from("Unauthorized"))
                                        .map_err(|never: std::convert::Infallible| match never {})
                                        .boxed(),
                                )
                                .unwrap(),
                        );
                    }
                }
            } // needs_auth

            // H5 (Phase 2.2): /api/* always handled by REST; body bytes are
            // collected for POST/PUT/PATCH/DELETE before dispatch. GUI asset
            // paths (non-/api/, non-/mcp when gui_enabled) reuse the body-less
            // entry point since they never consult the body. /mcp and unknown
            // paths fall through to MCP dispatch with the original body.
            if path.starts_with("/api/") {
                let response =
                    crate::mcp::rest::handle_api_request(req, &cfg).await;
                return Ok::<_, std::convert::Infallible>(response);
            }

            if config.server.gui_enabled && !path.starts_with("/mcp") {
                if let Some(response) =
                    crate::mcp::rest::handle_rest_request(&req, &cfg).await
                {
                    return Ok::<_, std::convert::Infallible>(response);
                }
            }

            // /mcp or unmatched path: pass through with original body.
            Ok::<_, std::convert::Infallible>(svc.handle(req).await)
        }
    });

    // Graceful shutdown: accept loop stops on Ctrl-C (and SIGTERM on Unix) so
    // in-flight connections get a chance to finish instead of being torn down.
    loop {
        let accept = tokio::select! {
            res = listener.accept() => res,
            _ = crate::service::shutdown_signal() => {
                tracing::info!("rein HTTP server: shutdown signal received, stopping accept loop");
                eprintln!("rein HTTP server: shutting down gracefully");
                cancel.cancel();
                crate::service::remove_pid("gui");
                break;
            }
        };
        let (stream, addr) = match accept {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("http accept error: {e}");
                continue;
            }
        };
        tracing::debug!("connection from {addr}");
        let svc = service.clone();
        tokio::spawn(async move {
            if let Err(e) =
                hyper_util::server::conn::auto::Builder::new(hyper_util::rt::TokioExecutor::new())
                    .serve_connection(hyper_util::rt::TokioIo::new(stream), svc)
                    .await
            {
                tracing::warn!("connection error: {e}");
            }
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_auth_accepts_matching_cookie() {
        let mut headers = hyper::HeaderMap::new();
        headers.insert(
            "cookie",
            hyper::header::HeaderValue::from_static("rein_http_token=secret-token"),
        );
        assert!(request_has_valid_http_auth(&headers, "secret-token"));
    }

    #[test]
    fn request_auth_rejects_empty_cookie_value() {
        let mut headers = hyper::HeaderMap::new();
        headers.insert(
            "cookie",
            hyper::header::HeaderValue::from_static("rein_http_token="),
        );
        assert!(!request_has_valid_http_auth(&headers, "secret-token"));
    }
}
