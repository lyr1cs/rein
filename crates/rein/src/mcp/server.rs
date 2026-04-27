use rmcp::model::{ServerCapabilities, ServerInfo};
use rmcp::{ServerHandler, ServiceExt};

use crate::config::ReinConfig;

/// MCP server for rein memory system.
///
/// Creates a new SqliteStore (SQLite connection) per request instead of sharing
/// a single Mutex<SqliteStore>. This eliminates the Mutex bottleneck and enables
/// concurrent read operations. SQLite with WAL mode + FULL_MUTEX handles write
/// serialization internally.
///
/// Post-Phase-3: every MCP tool is served through `OpsMcpEntry` inventory. The
/// old `#[tool_router]` scaffolding, the `non_store_count` / nudge-banner
/// counter, and the `tool_router` fallback in `call_tool` have been removed.
pub struct ReinServer {
    config: ReinConfig,
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
        crate::ops::inventory::ensure_unique_registrations();
        Self { config }
    }

    fn compact(&self) -> bool {
        self.config.server.compact
    }

    /// Dispatch an inventory-registered op by MCP tool name without requiring
    /// a `RequestContext`. Used by regression tests that cannot construct a
    /// full rmcp `RequestContext`.
    #[cfg(test)]
    pub async fn dispatch_inventory(
        &self,
        tool_name: &str,
        args: serde_json::Value,
    ) -> Option<Result<String, String>> {
        let entry =
            inventory::iter::<crate::ops::OpsMcpEntry>().find(|e| e.mcp_name == tool_name)?;
        let runtime = std::sync::Arc::new(crate::ops::OpsRuntime::for_mcp(std::sync::Arc::new(
            self.config.clone(),
        )));
        runtime.set_compact(self.compact());
        Some(
            (entry.invoke)(runtime, args)
                .await
                .map_err(|e| e.to_string()),
        )
    }
}

const HTTP_SESSION_COOKIE: &str = "rein_http_token";

fn constant_time_eq(left: &str, right: &str) -> bool {
    use sha2::{Digest, Sha256};

    let left_hash = Sha256::digest(left.as_bytes());
    let right_hash = Sha256::digest(right.as_bytes());
    left_hash
        .iter()
        .zip(right_hash.iter())
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
    let token_header = headers
        .get("x-rein-token")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let expected_str = format!("Bearer {expected}");
    if constant_time_eq(auth_header, &expected_str) || constant_time_eq(token_header, expected) {
        return true;
    }
    cookie_value(headers, HTTP_SESSION_COOKIE)
        .map(|value| constant_time_eq(&value, expected))
        .unwrap_or(false)
}

impl ServerHandler for ReinServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions(
                "rein: Multi-source cross-validated memory MCP server. \
                 Persistent memory across sessions, shared across all rein clients via ~/.rein/memories.db.\n\
                 \n\
                 TRIGGER (English / 中文): memory / 记忆, recall / 召回 / 回忆, remember / 记住, \
                 save / 存储 / 保存, search past / 搜索历史, knowledge graph / 知识图谱 / 概念, \
                 memoir / 知识库, timeline / 时间线 / 历史, episode / session / 会话, \
                 past sessions / 之前的工作 / 上次说过. When the user (any language) asks to save, \
                 store, recall, or search information across sessions, USE THESE TOOLS — do NOT \
                 say you don't know what rein is.\n\
                 \n\
                 Core: rein_store (new facts/decisions/preferences), rein_recall (retrieval, \
                 supports from/to temporal range and synthesize=true for narrative). \
                 Knowledge graph: rein_memoir_* (10 tools — create/list/show/add_concept/refine/\
                 search/search_all/link/inspect/export). Temporal: rein_timeline, \
                 rein_concept_history. Adaptive/feedback: rein_adaptive_status, rein_feedback, \
                 rein_feedback_concept_summary, rein_concept_state, rein_archive_summary_refresh. \
                 Maintenance: rein_consolidate, rein_dedup, rein_cleanup, rein_gc, rein_organize. \
                 Listing: rein_list_topics, rein_recent, rein_stats, rein_health. \
                 Total: 36 tools as of v0.27.0.\n\
                 \n\
                 Defaults: call rein_recall at the start of a session when the user references \
                 past work; call rein_store after solving bugs, making architecture decisions, \
                 or learning user preferences. After acting on a recalled memory, call \
                 rein_feedback to drive adaptive learning.",
            )
    }

    async fn list_tools(
        &self,
        _request: Option<rmcp::model::PaginatedRequestParams>,
        _context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<rmcp::model::ListToolsResult, rmcp::ErrorData> {
        let mut tools: Vec<rmcp::model::Tool> = inventory::iter::<crate::ops::OpsMcpEntry>()
            .map(inventory_entry_to_tool)
            .collect();
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
        _context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<rmcp::model::CallToolResult, rmcp::ErrorData> {
        let tool_name = request.name.as_ref();
        let entry = inventory::iter::<crate::ops::OpsMcpEntry>().find(|e| e.mcp_name == tool_name);
        let Some(entry) = entry else {
            return Ok(rmcp::model::CallToolResult::error(vec![
                rmcp::model::Content::text(format!("unknown tool: {tool_name}")),
            ]));
        };
        let args_value = request
            .arguments
            .clone()
            .map(serde_json::Value::Object)
            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
        let runtime = std::sync::Arc::new(crate::ops::OpsRuntime::for_mcp(std::sync::Arc::new(
            self.config.clone(),
        )));
        // Propagate the server-level compact flag so the macro-emitted MCP
        // output branch renders IntoMarkdown when compact is set.
        runtime.set_compact(self.compact());
        match (entry.invoke)(runtime, args_value).await {
            Ok(body) => Ok(rmcp::model::CallToolResult::success(vec![
                rmcp::model::Content::text(body),
            ])),
            Err(e) => Ok(rmcp::model::CallToolResult::error(vec![
                rmcp::model::Content::text(e.to_string()),
            ])),
        }
    }

    fn get_tool(&self, name: &str) -> Option<rmcp::model::Tool> {
        inventory::iter::<crate::ops::OpsMcpEntry>()
            .find(|e| e.mcp_name == name)
            .map(inventory_entry_to_tool)
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
                Ok(count) if count > 0 => {
                    tracing::info!("repaired {count} orphan session_artifact → episode links")
                }
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
    let gui_enabled = config.server.gui_enabled;
    let service = hyper::service::service_fn(move |req: hyper::Request<_>| {
        let svc = service.clone();
        let token = auth_token.clone();
        let cfg = rest_config.clone();
        async move {
            // v0.26.2 default-deny: any request requires bearer auth UNLESS
            // it is the stale-cookie clear path OR a GUI surface served when
            // the SPA is enabled. Earlier allowlist (`/api/` || `/mcp`) let
            // unmatched paths fall through to the MCP service, so a request
            // like `POST /not-mcp` ran MCP `initialize` with no token. The
            // pure helper `http_request_needs_auth` is unit-tested below.
            let path = req.uri().path();
            let method = req.method();
            let needs_auth = http_request_needs_auth(method, path, gui_enabled);
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
                let response = crate::mcp::rest::handle_api_request(req, &cfg).await;
                return Ok::<_, std::convert::Infallible>(response);
            }

            if config.server.gui_enabled && !path.starts_with("/mcp") {
                if let Some(response) = crate::mcp::rest::handle_rest_request(&req, &cfg).await {
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

/// v0.26.2 default-deny auth gate. Any HTTP request requires bearer-token
/// auth unless one of two narrow exemptions applies:
///
/// 1. `DELETE /api/session` — lets the SPA clear a stale/invalid cookie
///    even when the cookie no longer matches the live token. Without this
///    exemption a browser holding a bad cookie would loop on 401.
/// 2. Any non-`/api/`, non-`/mcp` path when `gui_enabled` is `true` — the
///    SPA must reach `/`, `/index.html`, `/assets/...`, and SPA-fallback
///    routes (e.g. `/synthesis-lab`) to bootstrap and show the
///    token-input dialog.
///
/// All other paths require auth; the previous allowlist (`/api/` || `/mcp`)
/// let `POST /not-mcp` fall through to the MCP service with no token.
pub(crate) fn http_request_needs_auth(method: &hyper::Method, path: &str, gui_enabled: bool) -> bool {
    if method == hyper::Method::DELETE && path == "/api/session" {
        return false;
    }
    if path.starts_with("/api/") || path.starts_with("/mcp") {
        return true;
    }
    // Unknown path: open to the GUI when it's serving (for SPA bootstrap +
    // static assets); otherwise default-deny.
    !gui_enabled
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_time_eq_matches_proxy_digest_style() {
        assert!(constant_time_eq("secret-token", "secret-token"));
        assert!(!constant_time_eq("secret-token", "secret"));
        assert!(!constant_time_eq("secret-token", "secret-token-longer"));
    }

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
    fn request_auth_accepts_matching_x_rein_token() {
        let mut headers = hyper::HeaderMap::new();
        headers.insert(
            "x-rein-token",
            hyper::header::HeaderValue::from_static("secret-token"),
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

    #[test]
    fn request_auth_rejects_wrong_length_bearer_token() {
        let mut headers = hyper::HeaderMap::new();
        headers.insert(
            "authorization",
            hyper::header::HeaderValue::from_static("Bearer secret-token-longer"),
        );
        assert!(!request_has_valid_http_auth(&headers, "secret-token"));
    }

    // v0.26.2: auth-gate path predicate regression tests for the bypass
    // bug where unmatched paths fell through to the MCP service.
    use hyper::Method;

    #[test]
    fn auth_required_for_api_path() {
        assert!(http_request_needs_auth(&Method::GET, "/api/foo", false));
        assert!(http_request_needs_auth(&Method::GET, "/api/foo", true));
    }

    #[test]
    fn auth_required_for_mcp_path() {
        assert!(http_request_needs_auth(&Method::POST, "/mcp", false));
        assert!(http_request_needs_auth(&Method::POST, "/mcp", true));
        assert!(http_request_needs_auth(&Method::POST, "/mcp/init", false));
    }

    #[test]
    fn auth_required_for_unknown_path_when_gui_disabled() {
        // The bypass: previously `POST /not-mcp` skipped auth and dispatched
        // to MCP svc.handle. Default-deny closes that hole.
        assert!(http_request_needs_auth(&Method::POST, "/not-mcp", false));
        assert!(http_request_needs_auth(&Method::GET, "/", false));
        assert!(http_request_needs_auth(&Method::GET, "/index.html", false));
    }

    #[test]
    fn auth_skipped_for_unknown_path_when_gui_enabled() {
        // GUI mode: SPA boots from `/`, asset paths under `/assets/...`,
        // SPA fallback to index.html for any client-side route.
        assert!(!http_request_needs_auth(&Method::GET, "/", true));
        assert!(!http_request_needs_auth(&Method::GET, "/index.html", true));
        assert!(!http_request_needs_auth(&Method::GET, "/assets/app.js", true));
        assert!(!http_request_needs_auth(&Method::GET, "/synthesis-lab", true));
    }

    #[test]
    fn auth_skipped_for_delete_api_session() {
        assert!(!http_request_needs_auth(
            &Method::DELETE,
            "/api/session",
            false
        ));
        assert!(!http_request_needs_auth(
            &Method::DELETE,
            "/api/session",
            true
        ));
    }

    #[test]
    fn auth_required_for_other_session_methods() {
        assert!(http_request_needs_auth(
            &Method::POST,
            "/api/session",
            false
        ));
        assert!(http_request_needs_auth(&Method::GET, "/api/session", true));
    }
}
