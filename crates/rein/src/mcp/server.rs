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
        let entry = inventory::iter::<crate::ops::OpsMcpEntry>()
            .find(|e| e.mcp_name == tool_name)?;
        let runtime = std::sync::Arc::new(crate::ops::OpsRuntime::for_mcp(
            std::sync::Arc::new(self.config.clone()),
        ));
        runtime.set_compact(self.compact());
        Some((entry.invoke)(runtime, args).await.map_err(|e| e.to_string()))
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
        let entry = inventory::iter::<crate::ops::OpsMcpEntry>()
            .find(|e| e.mcp_name == tool_name);
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
        let runtime = std::sync::Arc::new(crate::ops::OpsRuntime::for_mcp(
            std::sync::Arc::new(self.config.clone()),
        ));
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
