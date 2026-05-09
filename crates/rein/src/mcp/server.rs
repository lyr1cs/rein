use rmcp::model::{ServerCapabilities, ServerInfo};
use rmcp::{ServerHandler, ServiceExt};

use crate::config::ReinConfig;
use http_body_util::BodyExt;
use serde_json::Value;

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
const SEC_FETCH_SITE: &str = "sec-fetch-site";

type HttpBoxBody = http_body_util::combinators::BoxBody<bytes::Bytes, std::convert::Infallible>;

#[derive(Debug, Clone, PartialEq, Eq)]
struct HttpRequestHost {
    host: String,
    port: Option<u16>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum HttpGuardRejection {
    BadRequest(&'static str),
    Forbidden(&'static str),
}

impl HttpGuardRejection {
    fn status(&self) -> hyper::StatusCode {
        match self {
            Self::BadRequest(_) => hyper::StatusCode::BAD_REQUEST,
            Self::Forbidden(_) => hyper::StatusCode::FORBIDDEN,
        }
    }

    fn message(&self) -> &'static str {
        match self {
            Self::BadRequest(message) | Self::Forbidden(message) => message,
        }
    }
}

fn plain_http_response(
    status: hyper::StatusCode,
    body: &'static str,
) -> hyper::Response<HttpBoxBody> {
    hyper::Response::builder()
        .status(status)
        .header(hyper::header::CONTENT_TYPE, "text/plain; charset=utf-8")
        .body(http_body_util::Full::new(bytes::Bytes::from_static(body.as_bytes())).boxed())
        .unwrap_or_else(|_| {
            hyper::Response::new(
                http_body_util::Full::new(bytes::Bytes::from_static(b"internal error")).boxed(),
            )
        })
}

fn http_guard_response(rejection: HttpGuardRejection) -> hyper::Response<HttpBoxBody> {
    plain_http_response(rejection.status(), rejection.message())
}

fn normalize_host_for_guard(host: &str) -> String {
    let trimmed = host.trim();
    let without_brackets = trimmed
        .strip_prefix('[')
        .and_then(|rest| rest.strip_suffix(']'))
        .unwrap_or(trimmed);
    without_brackets.trim_end_matches('.').to_ascii_lowercase()
}

fn parse_http_authority(value: &str) -> Option<HttpRequestHost> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    let authority = hyper::http::uri::Authority::try_from(value).ok()?;
    Some(HttpRequestHost {
        host: normalize_host_for_guard(authority.host()),
        port: authority.port_u16(),
    })
}

fn parse_allowed_http_authority(value: &str) -> Option<HttpRequestHost> {
    parse_http_authority(value).or_else(|| {
        let host = normalize_host_for_guard(value);
        if host.is_empty() {
            None
        } else {
            Some(HttpRequestHost { host, port: None })
        }
    })
}

fn parse_http_request_host(
    headers: &hyper::HeaderMap,
) -> Result<HttpRequestHost, HttpGuardRejection> {
    let host = headers
        .get(hyper::header::HOST)
        .ok_or(HttpGuardRejection::BadRequest("missing Host header"))?;
    let host = host
        .to_str()
        .map_err(|_| HttpGuardRejection::BadRequest("invalid Host header"))?;
    parse_http_authority(host).ok_or(HttpGuardRejection::BadRequest("invalid Host header"))
}

fn is_specific_bind_host(host: &str) -> bool {
    !matches!(host, "" | "*" | "0.0.0.0" | "::")
}

/// v0.27.3 F5/C3 startup predicate: returns `true` when the bind host is a
/// wildcard (`0.0.0.0`, `::`, `*`, empty) AND no explicit allowlist is
/// configured. Centralized so the `run_http` startup-refusal and unit
/// tests share a single source of truth.
pub(crate) fn wildcard_bind_requires_allowlist(
    bind_host: &str,
    allowed_hosts: Option<&[String]>,
) -> bool {
    !is_specific_bind_host(bind_host) && allowed_hosts.is_none_or(|hosts| hosts.is_empty())
}

/// True when the in-process Host-header guard should run for this bind.
/// Specific binds (loopback or concrete LAN IPs) always enable the guard;
/// wildcard binds enable it only when an explicit `[server].allowed_hosts`
/// is set (v0.27.3 F5/C3 — startup refuses to come up without one for
/// wildcard listeners, so the guard is never silently disabled there).
fn http_host_guard_enabled(bind_host: &str, allowed_hosts: Option<&[String]>) -> bool {
    if allowed_hosts.is_some_and(|hosts| !hosts.is_empty()) {
        return true;
    }
    is_specific_bind_host(&normalize_host_for_guard(bind_host))
}

fn http_allowed_hosts(bind_host: &str, allowed_hosts: Option<&[String]>) -> Vec<String> {
    // v0.27.3 F5/C3: when the operator supplies an explicit allowlist,
    // honor it as-is (still keep loopback so curl/health-checks work).
    if let Some(extra) = allowed_hosts {
        if !extra.is_empty() {
            let mut allowed = vec![
                "localhost".to_string(),
                "127.0.0.1".to_string(),
                "::1".to_string(),
            ];
            for host in extra {
                let trimmed = host.trim();
                if trimmed.is_empty() {
                    continue;
                }
                allowed.push(trimmed.to_string());
            }
            return allowed;
        }
    }

    let mut allowed = vec![
        "localhost".to_string(),
        "127.0.0.1".to_string(),
        "::1".to_string(),
    ];
    let normalized_bind = normalize_host_for_guard(bind_host);
    if is_specific_bind_host(&normalized_bind)
        && !allowed
            .iter()
            .any(|host| normalize_host_for_guard(host) == normalized_bind)
    {
        allowed.push(bind_host.trim().to_string());
    }
    allowed
}

fn http_host_is_allowed(
    host: &HttpRequestHost,
    bind_host: &str,
    allowed_hosts: Option<&[String]>,
) -> bool {
    http_allowed_hosts(bind_host, allowed_hosts)
        .iter()
        .filter_map(|allowed| parse_allowed_http_authority(allowed))
        .any(|allowed| {
            allowed.host == host.host
                && match allowed.port {
                    Some(port) => host.port == Some(port),
                    None => true,
                }
        })
}

fn validate_http_request_host(
    headers: &hyper::HeaderMap,
    bind_host: &str,
    allowed_hosts: Option<&[String]>,
) -> Result<HttpRequestHost, HttpGuardRejection> {
    let host = parse_http_request_host(headers)?;
    if !http_host_is_allowed(&host, bind_host, allowed_hosts) {
        return Err(HttpGuardRejection::Forbidden("Host header is not allowed"));
    }
    Ok(host)
}

fn is_mutating_http_surface_request(method: &hyper::Method, path: &str) -> bool {
    matches!(
        *method,
        hyper::Method::POST | hyper::Method::PUT | hyper::Method::PATCH | hyper::Method::DELETE
    ) && (path.starts_with("/api/") || path.starts_with("/mcp"))
}

fn request_has_browser_mutation_headers(headers: &hyper::HeaderMap) -> bool {
    headers.contains_key(hyper::header::ORIGIN) || headers.contains_key(SEC_FETCH_SITE)
}

fn origin_authority(origin: &str) -> Option<(String, HttpRequestHost)> {
    let uri = origin.trim().parse::<hyper::Uri>().ok()?;
    let scheme = uri.scheme_str()?.to_ascii_lowercase();
    let authority = uri.authority()?;
    Some((
        scheme,
        HttpRequestHost {
            host: normalize_host_for_guard(authority.host()),
            port: authority.port_u16(),
        },
    ))
}

fn default_port_for_scheme(scheme: &str) -> Option<u16> {
    match scheme {
        "http" => Some(80),
        "https" => Some(443),
        _ => None,
    }
}

fn same_origin_as_request(origin: &str, request_host: &HttpRequestHost) -> bool {
    let Some((scheme, origin_host)) = origin_authority(origin) else {
        return false;
    };
    if !matches!(scheme.as_str(), "http" | "https") {
        return false;
    }
    let origin_port = origin_host
        .port
        .or_else(|| default_port_for_scheme(&scheme));
    let request_port = request_host
        .port
        .or_else(|| default_port_for_scheme(&scheme));
    origin_host.host == request_host.host && origin_port == request_port
}

fn is_loopback_request_host(host: &HttpRequestHost) -> bool {
    if host.host == "localhost" {
        return true;
    }
    host.host
        .parse::<std::net::IpAddr>()
        .is_ok_and(|ip| ip.is_loopback())
}

fn mcp_body_mutating_tool_name(body: &[u8]) -> Option<String> {
    let value = serde_json::from_slice::<Value>(body).ok()?;
    json_rpc_mutating_tool_name(&value).map(str::to_string)
}

fn json_rpc_mutating_tool_name(value: &Value) -> Option<&str> {
    if let Some(batch) = value.as_array() {
        return batch.iter().find_map(json_rpc_mutating_tool_name);
    }

    let obj = value.as_object()?;
    if obj.get("method").and_then(Value::as_str) != Some("tools/call") {
        return None;
    }
    let name = obj
        .get("params")
        .and_then(|params| params.get("name"))
        .and_then(Value::as_str)?;
    inventory::iter::<crate::ops::OpsMcpEntry>()
        .any(|entry| entry.mcp_name == name && entry.mutating)
        .then_some(name)
}

fn validate_browser_mutation_guard(
    method: &hyper::Method,
    path: &str,
    headers: &hyper::HeaderMap,
    request_host: &HttpRequestHost,
) -> Result<(), HttpGuardRejection> {
    if !is_mutating_http_surface_request(method, path) {
        return Ok(());
    }

    if let Some(fetch_site) = headers.get(SEC_FETCH_SITE).and_then(|v| v.to_str().ok()) {
        match fetch_site.to_ascii_lowercase().as_str() {
            "same-origin" | "none" => {}
            _ => {
                return Err(HttpGuardRejection::Forbidden(
                    "cross-site browser request blocked",
                ));
            }
        }
    }

    if let Some(origin) = headers
        .get(hyper::header::ORIGIN)
        .and_then(|v| v.to_str().ok())
    {
        if !same_origin_as_request(origin, request_host) {
            return Err(HttpGuardRejection::Forbidden(
                "cross-origin browser request blocked",
            ));
        }
    }

    Ok(())
}

async fn collect_mcp_request_body_capped<B>(
    body: B,
) -> Result<bytes::Bytes, hyper::Response<HttpBoxBody>>
where
    B: hyper::body::Body<Data = bytes::Bytes> + Unpin,
    B::Error: std::fmt::Display,
{
    crate::mcp::rest::collect_body_capped(body).await
}

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
        // AGPL §13: callers see source location + license straight from the
        // server's self-description, so a network MCP user always has a
        // path to the modified Corresponding Source. `format!` rather than
        // a raw `&str` so `crate::SOURCE_URL` / `crate::LICENSE_SPDX`
        // stay the single source of truth.
        let instructions = format!(
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
             rein_feedback_concept_summary, rein_concept_state, rein_archive_summary_refresh, \
             rein_judge_synthesis, rein_judge_concept_summary. \
             Maintenance: rein_consolidate, rein_dedup, rein_cleanup, rein_gc, rein_organize. \
             Listing: rein_list_topics, rein_recent, rein_stats, rein_health. \
             Total: 38 tools as of v0.27.5.\n\
             \n\
             Defaults: call rein_recall at the start of a session when the user references \
             past work; call rein_store after solving bugs, making architecture decisions, \
             or learning user preferences. After acting on a recalled memory, call \
             rein_feedback to drive adaptive learning.\n\
             \n\
             License: {license} — source: {source}",
            license = crate::LICENSE_SPDX,
            source = crate::SOURCE_URL,
        );
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions(instructions)
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StartupSurface {
    Stdio,
    Http,
}

fn should_spawn_background_warmup(config: &ReinConfig, surface: StartupSurface) -> bool {
    if !config.server.background_warmup {
        return false;
    }
    match surface {
        StartupSurface::Stdio => config.server.stdio_background_warmup,
        StartupSurface::Http => true,
    }
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
    if should_spawn_background_warmup(&config, StartupSurface::Stdio) {
        spawn_background_warmup(&config);
    }

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
    if should_spawn_background_warmup(&config, StartupSurface::Http) {
        spawn_background_warmup(&config);
    }

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
    let allow_loopback_unauth = config.server.loopback_unauth_requested();
    if auth_token.is_some() && allow_loopback_unauth {
        tracing::warn!(
            "REIN_HTTP_TOKEN is set while [server].allow_unauthenticated_loopback=true; \
             token auth wins and the loopback flag is a no-op at runtime"
        );
    } else if auth_token.is_some() && config.server.allow_unauthenticated_loopback {
        tracing::warn!(
            "REIN_HTTP_TOKEN is set while [server].allow_unauthenticated_loopback=true, \
             but the flag cannot take effect because server.sse_bind is not loopback; \
             bearer auth remains required"
        );
    }
    if auth_token.is_none() && !allow_loopback_unauth {
        return Err(anyhow::anyhow!(
            "REIN_HTTP_TOKEN must be set for HTTP/SSE access on '{}'. \
             Set REIN_HTTP_TOKEN=<secret> or explicitly opt into unauthenticated loopback with [server].allow_unauthenticated_loopback=true",
            config.server.sse_bind
        ));
    }

    // v0.27.3 F5/C3 (codex R1 P1 amendment): refuse to start on wildcard
    // binds (`0.0.0.0`, `::`, `*`, empty) unless EITHER
    // `[server].allowed_hosts` is explicitly set OR a bearer token is
    // configured. With a bearer token, every request must carry valid
    // auth, so DNS-rebinding requests from third-party origins fail at
    // the auth check; the Host guard is defense-in-depth, not the only
    // defense. This keeps the shipped Docker image (`REIN_SSE_BIND=0.0.0.0`
    // + `REIN_HTTP_TOKEN`) bootable while still requiring an explicit
    // allowlist for unauthenticated wildcard listeners.
    if wildcard_bind_requires_allowlist(
        &config.server.sse_bind,
        config.server.allowed_hosts.as_deref(),
    ) && auth_token.is_none()
    {
        anyhow::bail!(
            "Refusing to start: bind is wildcard ({}) without REIN_HTTP_TOKEN and without \
             [server].allowed_hosts. Either set REIN_HTTP_TOKEN=<secret> (bearer-protected), \
             or set [server].allowed_hosts = [\"hostname1\", \"hostname2\"] in ~/.rein/config.toml. \
             Wildcard binds require at least one of these to defend against DNS rebinding.",
            config.server.sse_bind
        );
    }

    let cancel = CancellationToken::new();

    let session_manager = Arc::new(LocalSessionManager::default());
    // v0.28.13 hotfix: propagate `[server].allowed_hosts` to rmcp's own
    // streamable-HTTP host guard. rmcp 1.6 added its own DNS-rebinding
    // host check (default `["localhost", "127.0.0.1", "::1"]`) which runs
    // ahead of rein's `validate_http_request_host`. Without this bridge,
    // rein's own allowlist is ignored for any non-loopback Host header
    // (e.g. requests proxied through a Tailscale Funnel hostname).
    //
    // Three cases:
    //   1. Operator set `[server].allowed_hosts` explicitly → mirror the
    //      rein-derived allowlist (loopback + extras) into rmcp.
    //   2. Specific bind (loopback or LAN IP) without an explicit
    //      allowlist → mirror the bind-derived allowlist into rmcp.
    //   3. Wildcard bind (`0.0.0.0`/`::`) with no allowlist → bearer auth
    //      is the only sentinel (the startup guard above enforced that
    //      `REIN_HTTP_TOKEN` is set), so disable rmcp's host check rather
    //      than letting it reject every non-loopback request and break
    //      documented Docker / bearer-protected deployment modes.
    let bind_host = &config.server.sse_bind;
    let cfg_allowed = config.server.allowed_hosts.as_deref();
    let rmcp_allowed_hosts = if cfg_allowed.is_some_and(|hosts| !hosts.is_empty())
        || is_specific_bind_host(&normalize_host_for_guard(bind_host))
    {
        Some(http_allowed_hosts(bind_host, cfg_allowed))
    } else {
        None
    };
    let mut http_config = StreamableHttpServerConfig::default()
        .with_stateful_mode(true)
        .with_cancellation_token(cancel.clone());
    http_config = match rmcp_allowed_hosts {
        Some(hosts) => http_config.with_allowed_hosts(hosts),
        None => http_config.disable_allowed_hosts(),
    };

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
    // Plain SSE/MCP HTTP must not masquerade as GUI, or `rein gui off` can
    // stop a non-GUI server.
    let pid_name = if config.server.gui_enabled {
        "gui"
    } else {
        "http"
    };
    let _ = crate::service::write_pid(pid_name);

    let rest_config = config.clone();
    let gui_enabled = config.server.gui_enabled;
    let service = hyper::service::service_fn(move |req: hyper::Request<_>| {
        let svc = service.clone();
        let token = auth_token.clone();
        let cfg = rest_config.clone();
        async move {
            let path = req.uri().path().to_string();
            let method = req.method().clone();
            let allowed_hosts = cfg.server.allowed_hosts.as_deref();
            let host_for_origin = if http_host_guard_enabled(&cfg.server.sse_bind, allowed_hosts) {
                match validate_http_request_host(req.headers(), &cfg.server.sse_bind, allowed_hosts)
                {
                    Ok(host) => Some(host),
                    Err(rejection) => {
                        return Ok::<_, std::convert::Infallible>(http_guard_response(rejection));
                    }
                }
            } else {
                parse_http_request_host(req.headers()).ok()
            };

            if let Some(host) = host_for_origin.as_ref() {
                if let Err(rejection) =
                    validate_browser_mutation_guard(&method, &path, req.headers(), host)
                {
                    return Ok::<_, std::convert::Infallible>(http_guard_response(rejection));
                }
            } else if is_mutating_http_surface_request(&method, &path)
                && request_has_browser_mutation_headers(req.headers())
            {
                return Ok::<_, std::convert::Infallible>(http_guard_response(
                    HttpGuardRejection::BadRequest("missing Host header"),
                ));
            }

            // v0.26.2 default-deny: any request requires bearer auth UNLESS
            // it is the stale-cookie clear path OR a GUI surface served when
            // the SPA is enabled. Earlier allowlist (`/api/` || `/mcp`) let
            // unmatched paths fall through to the MCP service, so a request
            // like `POST /not-mcp` ran MCP `initialize` with no token. The
            // pure helper `http_request_needs_auth` is unit-tested below.
            let needs_auth = http_request_needs_auth(&method, &path, gui_enabled);
            if needs_auth {
                if let Some(ref expected) = token {
                    if !request_has_valid_http_auth(req.headers(), expected) {
                        return Ok::<_, std::convert::Infallible>(plain_http_response(
                            hyper::StatusCode::UNAUTHORIZED,
                            "Unauthorized",
                        ));
                    }
                }
            } // needs_auth

            // H5 (Phase 2.2): /api/* always handled by REST; body bytes are
            // collected for POST/PUT/PATCH/DELETE before dispatch. GUI asset
            // paths (non-/api/, non-/mcp when gui_enabled) reuse the body-less
            // entry point since they never consult the body. /mcp and unknown
            // paths fall through to MCP dispatch with the original body.
            if path.starts_with("/api/") {
                if token.is_none()
                    && is_mutating_http_surface_request(&method, &path)
                    && !host_for_origin
                        .as_ref()
                        .is_some_and(is_loopback_request_host)
                {
                    return Ok::<_, std::convert::Infallible>(plain_http_response(
                        hyper::StatusCode::FORBIDDEN,
                        "unauthenticated public HTTP REST cannot mutate state",
                    ));
                }
                let response = crate::mcp::rest::handle_api_request(req, &cfg).await;
                return Ok::<_, std::convert::Infallible>(response);
            }

            if gui_enabled && !path.starts_with("/mcp") {
                if let Some(response) = crate::mcp::rest::handle_rest_request(&req, &cfg).await {
                    return Ok::<_, std::convert::Infallible>(response);
                }
            }

            // /mcp or unmatched path: pass through with original body.
            let (parts, body) = req.into_parts();
            let body = match collect_mcp_request_body_capped(body).await {
                Ok(body) => body,
                Err(response) => return Ok::<_, std::convert::Infallible>(response),
            };
            if token.is_none()
                && !host_for_origin
                    .as_ref()
                    .is_some_and(is_loopback_request_host)
            {
                if let Some(tool_name) = mcp_body_mutating_tool_name(&body) {
                    tracing::warn!(
                        tool = tool_name,
                        "blocked unauthenticated public HTTP MCP mutating tool call"
                    );
                    return Ok::<_, std::convert::Infallible>(plain_http_response(
                        hyper::StatusCode::FORBIDDEN,
                        "unauthenticated public HTTP MCP cannot call mutating tools",
                    ));
                }
            }
            let req = hyper::Request::from_parts(parts, http_body_util::Full::new(body));
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
                crate::service::remove_pid(pid_name);
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
pub(crate) fn http_request_needs_auth(
    method: &hyper::Method,
    path: &str,
    gui_enabled: bool,
) -> bool {
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
    fn stdio_startup_warmup_is_disabled_by_default() {
        let config = ReinConfig::default();

        assert!(!should_spawn_background_warmup(
            &config,
            StartupSurface::Stdio
        ));
        assert!(should_spawn_background_warmup(
            &config,
            StartupSurface::Http
        ));
    }

    #[test]
    fn stdio_startup_warmup_is_explicit_opt_in() {
        let mut config = ReinConfig::default();
        config.server.stdio_background_warmup = true;

        assert!(should_spawn_background_warmup(
            &config,
            StartupSurface::Stdio
        ));

        config.server.background_warmup = false;
        assert!(!should_spawn_background_warmup(
            &config,
            StartupSurface::Stdio
        ));
        assert!(!should_spawn_background_warmup(
            &config,
            StartupSurface::Http
        ));
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
        assert!(!http_request_needs_auth(
            &Method::GET,
            "/assets/app.js",
            true
        ));
        assert!(!http_request_needs_auth(
            &Method::GET,
            "/synthesis-lab",
            true
        ));
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

    #[test]
    fn loopback_host_guard_rejects_dns_rebinding_host() {
        let mut headers = hyper::HeaderMap::new();
        headers.insert(
            hyper::header::HOST,
            hyper::header::HeaderValue::from_static("attacker.example:8691"),
        );

        assert!(matches!(
            validate_http_request_host(&headers, "127.0.0.1", None),
            Err(HttpGuardRejection::Forbidden(_))
        ));
    }

    #[test]
    fn host_guard_allows_loopback_and_specific_bind_hosts() {
        for host in ["localhost:8691", "127.0.0.1:8691", "[::1]:8691"] {
            let mut headers = hyper::HeaderMap::new();
            headers.insert(
                hyper::header::HOST,
                hyper::header::HeaderValue::from_str(host).unwrap(),
            );
            assert!(
                validate_http_request_host(&headers, "127.0.0.1", None).is_ok(),
                "{host} should be accepted"
            );
        }

        let mut headers = hyper::HeaderMap::new();
        headers.insert(
            hyper::header::HOST,
            hyper::header::HeaderValue::from_static("192.0.2.10:8691"),
        );
        assert!(validate_http_request_host(&headers, "192.0.2.10", None).is_ok());
    }

    #[test]
    fn host_guard_honors_explicit_allowed_hosts_on_wildcard_bind() {
        let allowed: Vec<String> = vec!["rein.internal".to_string()];
        let mut headers = hyper::HeaderMap::new();
        headers.insert(
            hyper::header::HOST,
            hyper::header::HeaderValue::from_static("rein.internal:8691"),
        );
        assert!(
            validate_http_request_host(&headers, "0.0.0.0", Some(&allowed)).is_ok(),
            "explicit allowed_hosts entry must be accepted on wildcard bind"
        );

        let mut headers = hyper::HeaderMap::new();
        headers.insert(
            hyper::header::HOST,
            hyper::header::HeaderValue::from_static("attacker.example:8691"),
        );
        assert!(
            matches!(
                validate_http_request_host(&headers, "0.0.0.0", Some(&allowed)),
                Err(HttpGuardRejection::Forbidden(_))
            ),
            "host outside allowed_hosts must still be rejected on wildcard bind"
        );
    }

    #[test]
    fn host_guard_enabled_when_allowed_hosts_set_even_for_wildcard_bind() {
        let allowed: Vec<String> = vec!["rein.internal".to_string()];
        // Wildcard bind alone disables the guard (legacy v0.26.x behavior),
        // but an explicit allow-list re-enables it (v0.27.3 F5/C3).
        assert!(!http_host_guard_enabled("0.0.0.0", None));
        assert!(http_host_guard_enabled("0.0.0.0", Some(&allowed)));
        // Empty allow-list is treated the same as None.
        let empty: Vec<String> = Vec::new();
        assert!(!http_host_guard_enabled("0.0.0.0", Some(&empty)));
    }

    #[test]
    fn c3_wildcard_bind_without_allowlist_is_refused() {
        // Wildcard binds with no allowlist trip the startup refusal.
        for wildcard in ["0.0.0.0", "::", "*", ""] {
            assert!(
                wildcard_bind_requires_allowlist(wildcard, None),
                "{wildcard} should require an explicit allowlist"
            );
            let empty: Vec<String> = Vec::new();
            assert!(
                wildcard_bind_requires_allowlist(wildcard, Some(&empty)),
                "{wildcard} with empty allowlist should still refuse"
            );
        }

        // Wildcard with an explicit allowlist is permitted.
        let allowed: Vec<String> = vec!["rein.internal".to_string()];
        for wildcard in ["0.0.0.0", "::"] {
            assert!(
                !wildcard_bind_requires_allowlist(wildcard, Some(&allowed)),
                "{wildcard} with allowlist should be allowed"
            );
        }

        // Specific binds never trigger the refusal.
        for specific in ["127.0.0.1", "::1", "localhost", "192.0.2.10"] {
            assert!(
                !wildcard_bind_requires_allowlist(specific, None),
                "{specific} is specific; refusal should not apply"
            );
        }
    }

    #[test]
    fn c1_default_loopback_is_authenticated() {
        // v0.27.3 F5/C1: fresh ServerConfig and ProxyConfig must default
        // to authenticated mode. Operators must explicitly opt into
        // unauthenticated loopback by writing the flag.
        let server = crate::config::ServerConfig::default();
        assert!(
            !server.allow_unauthenticated_loopback,
            "ServerConfig::default().allow_unauthenticated_loopback must be false"
        );
        let proxy = crate::config::ProxyConfig::default();
        assert!(
            !proxy.allow_unauthenticated_loopback,
            "ProxyConfig::default().allow_unauthenticated_loopback must be false"
        );
        // C3 sibling: allowed_hosts defaults to None so wildcard binds
        // hit the startup refusal until operators opt in.
        assert!(
            server.allowed_hosts.is_none(),
            "ServerConfig::default().allowed_hosts must be None"
        );
    }

    #[test]
    fn browser_mutating_surface_guard_rejects_cross_site_origin() {
        let mut headers = hyper::HeaderMap::new();
        headers.insert(
            hyper::header::HOST,
            hyper::header::HeaderValue::from_static("localhost:8691"),
        );
        headers.insert(
            hyper::header::ORIGIN,
            hyper::header::HeaderValue::from_static("http://attacker.example"),
        );
        headers.insert(
            "sec-fetch-site",
            hyper::header::HeaderValue::from_static("cross-site"),
        );
        let host = validate_http_request_host(&headers, "127.0.0.1", None).unwrap();

        assert!(matches!(
            validate_browser_mutation_guard(&Method::POST, "/api/memories", &headers, &host),
            Err(HttpGuardRejection::Forbidden(_))
        ));
        assert!(matches!(
            validate_browser_mutation_guard(&Method::POST, "/mcp", &headers, &host),
            Err(HttpGuardRejection::Forbidden(_))
        ));
    }

    #[test]
    fn browser_mutating_surface_guard_allows_same_origin_and_native_clients() {
        let mut same_origin = hyper::HeaderMap::new();
        same_origin.insert(
            hyper::header::HOST,
            hyper::header::HeaderValue::from_static("localhost:8691"),
        );
        same_origin.insert(
            hyper::header::ORIGIN,
            hyper::header::HeaderValue::from_static("http://localhost:8691"),
        );
        same_origin.insert(
            "sec-fetch-site",
            hyper::header::HeaderValue::from_static("same-origin"),
        );
        let host = validate_http_request_host(&same_origin, "127.0.0.1", None).unwrap();
        assert!(validate_browser_mutation_guard(
            &Method::POST,
            "/api/memories",
            &same_origin,
            &host
        )
        .is_ok());
        assert!(
            validate_browser_mutation_guard(&Method::POST, "/mcp", &same_origin, &host).is_ok()
        );

        let mut native = hyper::HeaderMap::new();
        native.insert(
            hyper::header::HOST,
            hyper::header::HeaderValue::from_static("localhost:8691"),
        );
        let host = validate_http_request_host(&native, "127.0.0.1", None).unwrap();
        assert!(
            validate_browser_mutation_guard(&Method::POST, "/api/memories", &native, &host).is_ok()
        );
        assert!(validate_browser_mutation_guard(&Method::POST, "/mcp", &native, &host).is_ok());
    }

    #[test]
    fn browser_mutating_surface_guard_allows_https_same_origin() {
        let mut headers = hyper::HeaderMap::new();
        headers.insert(
            hyper::header::HOST,
            hyper::header::HeaderValue::from_static("rein.example.com"),
        );
        headers.insert(
            hyper::header::ORIGIN,
            hyper::header::HeaderValue::from_static("https://rein.example.com"),
        );
        headers.insert(
            "sec-fetch-site",
            hyper::header::HeaderValue::from_static("same-origin"),
        );
        let allowed = vec!["rein.example.com".to_string()];
        let host = validate_http_request_host(&headers, "127.0.0.1", Some(&allowed)).unwrap();

        assert!(validate_browser_mutation_guard(&Method::POST, "/mcp", &headers, &host).is_ok());

        headers.insert(
            hyper::header::ORIGIN,
            hyper::header::HeaderValue::from_static("https://attacker.example"),
        );
        assert!(matches!(
            validate_browser_mutation_guard(&Method::POST, "/mcp", &headers, &host),
            Err(HttpGuardRejection::Forbidden(_))
        ));
    }

    #[test]
    fn public_mcp_mutation_detector_flags_mutating_tools_only() {
        let store_call = br#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"rein_store","arguments":{"content":"x"}}}"#;
        assert_eq!(
            mcp_body_mutating_tool_name(store_call).as_deref(),
            Some("rein_store")
        );

        let recall_call = br#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"rein_recall","arguments":{"query":"x"}}}"#;
        assert_eq!(mcp_body_mutating_tool_name(recall_call), None);

        let batch = br#"[{"jsonrpc":"2.0","id":1,"method":"tools/list"},{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"rein_forget","arguments":{"id":"1"}}}]"#;
        assert_eq!(
            mcp_body_mutating_tool_name(batch).as_deref(),
            Some("rein_forget")
        );
    }

    #[tokio::test]
    async fn mcp_body_cap_rejects_chunked_body_without_content_length() {
        use http_body_util::StreamBody;
        use hyper::body::Frame;

        let first_chunk = bytes::Bytes::from(vec![b'a'; 1024 * 1024]);
        let chunks = tokio_stream::iter(vec![
            Ok::<_, std::convert::Infallible>(Frame::data(first_chunk)),
            Ok(Frame::data(bytes::Bytes::from_static(b"x"))),
        ]);
        let body = StreamBody::new(chunks);
        let response = collect_mcp_request_body_capped(body).await.unwrap_err();

        assert_eq!(response.status(), hyper::StatusCode::PAYLOAD_TOO_LARGE);
    }
}
