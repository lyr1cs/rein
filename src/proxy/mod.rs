//! Transparent LLM API proxy with conservative recording and extraction.
//!
//! Intercepts requests to Anthropic (`/v1/messages`), OpenAI
//! (`/v1/chat/completions` / `/v1/responses`), and Codex first-party backend
//! routes (`/backend-api/codex/*`), forwards them to the upstream provider,
//! streams responses back, and asynchronously records memory candidates from
//! supported response shapes.

mod anthropic;
mod extract;
mod jwt;
mod openai;
mod policy;
mod provider;
mod responses;
mod ws_mirror;

use jwt::{bearer_jwt_info, decode_jwt_payload_for_logging, redact_jwt_payload};
#[cfg(test)]
use jwt::current_unix_timestamp;
use ws_mirror::WebSocketMirrorState;

use crate::config::ReinConfig;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
#[cfg(test)]
use base64::engine::general_purpose::URL_SAFE_NO_PAD as BASE64_URL_SAFE_NO_PAD;
use base64::Engine as _;
use bytes::Bytes;
use futures_util::{SinkExt, StreamExt};
use http_body_util::{BodyExt, Full, StreamBody};
use hyper::body::Frame;
use provider::{ProviderKind, RecordingMode};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpStream;
use tokio::sync::mpsc;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::error::ProtocolError as WsProtocolError;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::protocol::Role;
use tokio_tungstenite::tungstenite::Error as WsError;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Basic proxy metrics exposed via `GET /rein/metrics`.
struct ProxyMetrics {
    request_count: AtomicU64,
    error_count: AtomicU64,
    extraction_count: AtomicU64,
}

impl ProxyMetrics {
    fn new() -> Self {
        Self {
            request_count: AtomicU64::new(0),
            error_count: AtomicU64::new(0),
            extraction_count: AtomicU64::new(0),
        }
    }

    fn to_json(&self) -> String {
        format!(
            r#"{{"request_count":{},"error_count":{},"extraction_count":{}}}"#,
            self.request_count.load(Ordering::Relaxed),
            self.error_count.load(Ordering::Relaxed),
            self.extraction_count.load(Ordering::Relaxed),
        )
    }
}

// ---------------------------------------------------------------------------
// Shared state passed into every request handler
// ---------------------------------------------------------------------------

struct ProxyState {
    metrics: ProxyMetrics,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ResolvedRoute {
    provider: ProviderKind,
    recording_mode: RecordingMode,
}

#[derive(Debug, Clone)]
struct ProxyArtifactInput {
    route: ResolvedRoute,
    method: String,
    path: String,
    session_id: Option<String>,
    request_headers: Vec<(String, String)>,
    request_body: Bytes,
}

// WebSocketMirrorState (struct + ~270-line impl) moved to
// `src/proxy/ws_mirror.rs` in v0.19.0. Re-imported at the top of this file.

/// Start the transparent proxy server.
pub async fn run_proxy(config: ReinConfig) -> anyhow::Result<()> {
    // REIN_PROXY_ACTIVE is set by the caller (main.rs) before entering async
    // to avoid unsound set_var in multi-threaded context.

    let bind = format!("{}:{}", config.proxy.bind, config.proxy.port);

    // Security: require auth token by default, even on loopback.
    let is_loopback = config.proxy.bind == "127.0.0.1"
        || config.proxy.bind == "localhost"
        || config.proxy.bind == "::1";
    let auth_token = std::env::var("REIN_PROXY_TOKEN")
        .ok()
        .filter(|token| !token.trim().is_empty())
        .or_else(|| {
            std::env::var("REIN_HTTP_TOKEN")
                .ok()
                .filter(|token| !token.trim().is_empty())
        });
    let allow_loopback_unauth = config.proxy.allow_unauthenticated_loopback && is_loopback;
    if auth_token.is_none() && !allow_loopback_unauth {
        anyhow::bail!(
            "rein proxy: refusing to start on '{}' without REIN_PROXY_TOKEN set. \
             Set REIN_PROXY_TOKEN=<secret> or explicitly opt into unauthenticated loopback with [proxy].allow_unauthenticated_loopback=true.",
            config.proxy.bind
        );
    }

    let listener = tokio::net::TcpListener::bind(&bind).await?;
    eprintln!("rein proxy: running in record-only mode (automatic injection removed)");

    eprintln!("rein proxy listening on http://{bind}");
    eprintln!("  Anthropic: set ANTHROPIC_BASE_URL=http://{bind}");
    eprintln!("  OpenAI:    set OPENAI_BASE_URL=http://{bind}");
    // Write PID file for service management (rein proxy on/off, rein dashboard).
    let _ = crate::service::write_pid("proxy");

    // Shared reqwest client for all upstream requests (connection pooling).
    let upstream_client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()?;

    let state = Arc::new(ProxyState {
        metrics: ProxyMetrics::new(),
    });

    // Graceful shutdown: stop accept loop on Ctrl-C OR SIGTERM (Unix).
    loop {
        let accept = tokio::select! {
            res = listener.accept() => res,
            _ = crate::service::shutdown_signal() => {
                tracing::info!("rein proxy: received shutdown signal, stopping accept loop");
                eprintln!("rein proxy: shutting down gracefully");
                crate::service::remove_pid("proxy");
                break;
            }
        };

        let (stream, addr) = match accept {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("proxy accept error: {e}");
                continue;
            }
        };
        tracing::debug!("proxy connection from {addr}");
        let config = config.clone();
        let client = upstream_client.clone();
        let auth = auth_token.clone();
        let state = Arc::clone(&state);
        tokio::spawn(async move {
            let service = hyper::service::service_fn(move |req| {
                let config = config.clone();
                let client = client.clone();
                let auth = auth.clone();
                let state = Arc::clone(&state);
                async move { handle_request(req, config, client, auth.as_deref(), state).await }
            });

            if let Err(e) =
                hyper_util::server::conn::auto::Builder::new(hyper_util::rt::TokioExecutor::new())
                    .serve_connection_with_upgrades(hyper_util::rt::TokioIo::new(stream), service)
                    .await
            {
                tracing::warn!("proxy connection error: {e}");
            }
        });
    }

    Ok(())
}

type BoxBody = http_body_util::combinators::BoxBody<Bytes, hyper::Error>;
const WS_GUID: &str = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
type UpstreamWsStream = WebSocketStream<MaybeTlsStream<TcpStream>>;

fn box_body<B>(body: B) -> BoxBody
where
    B: hyper::body::Body<Data = Bytes, Error = hyper::Error> + Send + Sync + 'static,
{
    BoxBody::new(body)
}

fn full_body(data: Bytes) -> BoxBody {
    box_body(Full::new(data).map_err(|never| match never {}))
}

/// Constant-time equality for proxy token comparison.
///
/// Hashes BOTH sides to a fixed-size digest before the XOR-accumulate
/// compare. The earlier form short-circuited on length mismatch, which
/// leaked the expected token's length to a wall-clock-attacker — for
/// short-by-design tokens that narrows brute-force search meaningfully
/// (B7 LOW). With SHA-256 both sides always end up at 32 bytes, so
/// mismatches are indistinguishable whether they differ in length or
/// content.
fn proxy_token_eq(left: &str, right: &str) -> bool {
    use sha2::{Digest, Sha256};
    let lh = Sha256::digest(left.as_bytes());
    let rh = Sha256::digest(right.as_bytes());
    lh.iter()
        .zip(rh.iter())
        .fold(0u8, |acc, (x, y)| acc | (x ^ y))
        == 0
}

fn error_response(status: u16, msg: &str) -> hyper::Response<BoxBody> {
    hyper::Response::builder()
        .status(status)
        .header("content-type", "text/plain")
        .body(full_body(Bytes::from(msg.to_string())))
        .unwrap_or_else(|_| hyper::Response::new(full_body(Bytes::from("internal error"))))
}

/// True when the request path contains RFC 3986 dot-segment wildcards
/// (`/../`, `/./`, or trailing `/..`). `reqwest`/`url` normalize these on
/// the wire, which decouples the path used for local policy decisions from
/// the path the upstream actually sees. Reject such requests at the edge
/// so the two views are always identical.
fn has_traversal_segments(path: &str) -> bool {
    // Strip query string — path traversal only matters for the path portion.
    let path_only = path.split('?').next().unwrap_or(path);
    for segment in path_only.split('/') {
        if segment == ".." || segment == "." {
            return true;
        }
    }
    // Also catch `//` (empty segment) which some servers collapse asymmetrically.
    // We allow exactly one leading `/` but reject any `//` inside.
    if let Some(rest) = path_only.strip_prefix('/') {
        if rest.contains("//") {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod traversal_guard_tests {
    use super::has_traversal_segments;

    #[test]
    fn rejects_parent_dot_dot() {
        assert!(has_traversal_segments("/backend-api/codex/../v1/chat/completions"));
        assert!(has_traversal_segments("/foo/.."));
        assert!(has_traversal_segments("/../etc/passwd"));
    }

    #[test]
    fn rejects_single_dot_segments() {
        assert!(has_traversal_segments("/foo/./bar"));
        assert!(has_traversal_segments("/./"));
    }

    #[test]
    fn rejects_double_slash() {
        assert!(has_traversal_segments("/foo//bar"));
    }

    #[test]
    fn accepts_clean_paths() {
        assert!(!has_traversal_segments("/v1/messages"));
        assert!(!has_traversal_segments("/backend-api/codex/responses"));
        assert!(!has_traversal_segments("/responses?stream=true"));
        assert!(!has_traversal_segments("/"));
    }

    #[test]
    fn ignores_dots_inside_segments() {
        // "..foo" is not the ".." segment — must not trip.
        assert!(!has_traversal_segments("/a.b/c..d/e"));
        assert!(!has_traversal_segments("/config.yml"));
    }
}

/// Uniform response for WebSocket upgrade failure: always 426 Upgrade Required.
///
/// Previously this returned 426 for `/responses`-family providers and 502 for
/// everything else. The 502 variant was confusing because WS upgrade is a policy
/// decision, not an upstream error. 426 is semantically correct for every kind
/// of "upgrade refused" case; the response body carries the specific reason.
fn websocket_upstream_failure_response(provider: ProviderKind) -> hyper::Response<BoxBody> {
    let msg = match provider {
        ProviderKind::OpenAiResponses | ProviderKind::CodexFirstParty => {
            "responses websocket unavailable upstream; retry over HTTP"
        }
        _ => "upstream does not support websocket upgrade on this route",
    };
    error_response(426, msg)
}

fn recording_mode_label(mode: RecordingMode) -> &'static str {
    if mode.captures_structured_text() {
        "structured-text"
    } else if mode.captures_artifact_mirror_only() {
        "artifact-mirror-only"
    } else {
        unreachable!("unknown recording mode")
    }
}

fn is_benign_websocket_read_error(error: &WsError) -> bool {
    matches!(
        error,
        WsError::ConnectionClosed
            | WsError::AlreadyClosed
            | WsError::Protocol(WsProtocolError::ResetWithoutClosingHandshake)
    ) || error
        .to_string()
        .contains("Connection reset without closing handshake")
}

/// Build a response, falling back to a plain 200 with error text on builder failure.
fn build_response(
    builder: hyper::http::response::Builder,
    body: BoxBody,
) -> hyper::Response<BoxBody> {
    builder
        .body(body)
        .unwrap_or_else(|_| hyper::Response::new(full_body(Bytes::from("internal error"))))
}

fn capture_request_headers(headers: &hyper::HeaderMap) -> Vec<(String, String)> {
    headers
        .iter()
        .filter_map(|(name, value)| {
            value.to_str().ok().map(|value| {
                (
                    name.as_str().to_ascii_lowercase(),
                    redact_header_value(name.as_str(), value),
                )
            })
        })
        .collect()
}

fn capture_response_headers(headers: &reqwest::header::HeaderMap) -> Vec<(String, String)> {
    headers
        .iter()
        .filter_map(|(name, value)| {
            value.to_str().ok().map(|value| {
                (
                    name.as_str().to_ascii_lowercase(),
                    redact_header_value(name.as_str(), value),
                )
            })
        })
        .collect()
}

fn redact_header_value(name: &str, value: &str) -> String {
    match name.to_ascii_lowercase().as_str() {
        "authorization"
        | "x-rein-token"
        | "cookie"
        | "set-cookie"
        | "proxy-authorization"
        | "proxy-authenticate"
        | "chatgpt-account-id" => "<redacted>".to_string(),
        _ => crate::extract::hooks::parsing::redact_secrets(value),
    }
}

fn default_codex_originator() -> &'static str {
    "codex_cli_rs"
}

fn default_codex_user_agent() -> String {
    format!(
        "{}/{} ({}; {}) rein-proxy",
        default_codex_originator(),
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS,
        std::env::consts::ARCH
    )
}

fn websocket_upstream_origin(ws_url: &str) -> Option<String> {
    ws_url
        .strip_prefix("wss://")
        .map(|rest| format!("https://{}", rest.split('/').next().unwrap_or_default()))
        .or_else(|| {
            ws_url
                .strip_prefix("ws://")
                .map(|rest| format!("http://{}", rest.split('/').next().unwrap_or_default()))
        })
}

fn extract_session_id(headers: &hyper::HeaderMap) -> Option<String> {
    headers
        .get("x-client-request-id")
        .or_else(|| headers.get("x-codex-parent-thread-id"))
        .and_then(|value| value.to_str().ok())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn format_artifact_body(bytes: &[u8], truncated: bool) -> String {
    const MAX_BODY_CHARS: usize = 20_000;
    let preview: String = String::from_utf8_lossy(bytes)
        .chars()
        .take(MAX_BODY_CHARS)
        .collect();
    let preview = crate::extract::hooks::parsing::redact_secrets(&preview);
    if truncated {
        format!("{preview}\n[truncated]")
    } else {
        preview
    }
}

fn maybe_store_first_party_artifact(
    config: &ReinConfig,
    artifact: ProxyArtifactInput,
    status: u16,
    response_headers: Vec<(String, String)>,
    response_body: Vec<u8>,
    response_truncated: bool,
    streaming: bool,
) {
    if !matches!(
        artifact.route.provider,
        ProviderKind::CodexFirstParty | ProviderKind::ChatGptBackend
    ) {
        return;
    }

    let config = config.clone();
    tokio::task::spawn_blocking(move || {
        let request_body = format_artifact_body(&artifact.request_body, false);
        let response_body = format_artifact_body(&response_body, response_truncated);
        let provider_label = match artifact.route.provider {
            ProviderKind::CodexFirstParty => "codex-first-party",
            ProviderKind::ChatGptBackend => "chatgpt-backend",
            ProviderKind::Anthropic => "anthropic",
            ProviderKind::OpenAi => "openai",
            ProviderKind::OpenAiResponses => "openai-responses",
        };
        let capture_mode = recording_mode_label(artifact.route.recording_mode);
        let format_headers = |headers: &[(String, String)]| -> String {
            headers
                .iter()
                .map(|(name, value)| format!("{name}: {value}"))
                .collect::<Vec<_>>()
                .join("\n")
        };
        let transcript_text = format!(
            "provider: {provider_label}\ncapture_mode: {capture_mode}\nmethod: {}\npath: {}\nstatus: {status}\nstreaming: {streaming}\n\nrequest_headers:\n{}\n\nrequest_body:\n{}\n\nresponse_headers:\n{}\n\nresponse_body:\n{}",
            artifact.method,
            artifact.path,
            format_headers(&artifact.request_headers),
            request_body,
            format_headers(&response_headers),
            response_body,
        );
        let transcript_json = serde_json::json!({
            "provider": provider_label,
            "capture_mode": capture_mode,
            "method": artifact.method,
            "path": artifact.path,
            "status": status,
            "streaming": streaming,
            "request_headers": artifact.request_headers,
            "request_body": request_body,
            "response_headers": response_headers,
            "response_body": response_body,
        });
        let session_artifact = crate::types::SessionArtifact {
            id: String::new(),
            schema_version: 1,
            artifact_kind: "proxy_first_party".to_string(),
            session_id: artifact.session_id,
            title: Some(format!("{} {}", artifact.method, artifact.path)),
            summary: Some(format!("{provider_label} {} -> {status}", artifact.method)),
            source_agent: Some("proxy".to_string()),
            source_label: Some(format!("proxy:{provider_label}")),
            is_subagent: false,
            started_at: None,
            ended_at: None,
            turn_count: 2,
            transcript_text,
            transcript_json: Some(transcript_json.to_string()),
            episode_id: None,
            created_at: chrono::Utc::now(),
        };
        match config.open_store() {
            Ok(store) => {
                if let Err(e) = store.store_session_artifact(session_artifact) {
                    tracing::warn!("proxy artifact: failed to store session artifact: {e}");
                }
            }
            Err(e) => tracing::warn!("proxy artifact: failed to open store: {e}"),
        }
    });
}

fn maybe_store_first_party_ws_artifact(
    config: &ReinConfig,
    artifact: ProxyArtifactInput,
    response_headers: Vec<(String, String)>,
    request_event_messages: Vec<String>,
    response_event_messages: Vec<String>,
    truncated: bool,
) {
    if !matches!(artifact.route.provider, ProviderKind::CodexFirstParty) {
        return;
    }

    let config = config.clone();
    tokio::task::spawn_blocking(move || {
        let capture_mode = recording_mode_label(artifact.route.recording_mode);
        let format_headers = |headers: &[(String, String)]| -> String {
            headers
                .iter()
                .map(|(name, value)| format!("{name}: {value}"))
                .collect::<Vec<_>>()
                .join("\n")
        };
        let mut request_events = request_event_messages.join("\n");
        if truncated && !request_events.is_empty() {
            request_events.push_str("\n[truncated]");
        } else if truncated && request_events.is_empty() {
            request_events = "[truncated]".to_string();
        }
        let mut response_events = response_event_messages.join("\n");
        if truncated && !response_events.is_empty() {
            response_events.push_str("\n[truncated]");
        } else if truncated && response_events.is_empty() {
            response_events = "[truncated]".to_string();
        }
        let transcript_text = format!(
            "provider: codex-first-party-ws\ncapture_mode: {capture_mode}\nmethod: {}\npath: {}\n\nrequest_headers:\n{}\n\nhandshake_response_headers:\n{}\n\nclient_ws_events:\n{}\n\nupstream_ws_events:\n{}",
            artifact.method,
            artifact.path,
            format_headers(&artifact.request_headers),
            format_headers(&response_headers),
            request_events,
            response_events,
        );
        let transcript_json = serde_json::json!({
            "provider": "codex-first-party-ws",
            "capture_mode": capture_mode,
            "method": artifact.method,
            "path": artifact.path,
            "request_headers": artifact.request_headers,
            "handshake_response_headers": response_headers,
            "client_ws_events": request_event_messages,
            "upstream_ws_events": response_event_messages,
            "truncated": truncated,
        });
        let session_artifact = crate::types::SessionArtifact {
            id: String::new(),
            schema_version: 1,
            artifact_kind: "proxy_first_party_ws".to_string(),
            session_id: artifact.session_id,
            title: Some(format!("WS {} {}", artifact.method, artifact.path)),
            summary: Some("codex-first-party websocket mirror".to_string()),
            source_agent: Some("proxy".to_string()),
            source_label: Some("proxy:codex-first-party-ws".to_string()),
            is_subagent: false,
            started_at: None,
            ended_at: None,
            turn_count: (request_event_messages.len() + response_event_messages.len()) as u32,
            transcript_text,
            transcript_json: Some(transcript_json.to_string()),
            episode_id: None,
            created_at: chrono::Utc::now(),
        };
        match config.open_store() {
            Ok(store) => {
                if let Err(e) = store.store_session_artifact(session_artifact) {
                    tracing::warn!("proxy artifact: failed to store websocket artifact: {e}");
                }
            }
            Err(e) => tracing::warn!("proxy artifact: failed to open store: {e}"),
        }
    });
}

// `impl WebSocketMirrorState { ... }` moved to `src/proxy/ws_mirror.rs`
// in v0.19.0 (audit `proxy/mod.rs` split). ~270 lines extracted.

pub(super) fn truncate_utf8_to_byte_limit(text: &str, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text.to_string();
    }
    let mut end = max_bytes.min(text.len());
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    text[..end].to_string()
}

// ---------------------------------------------------------------------------
// Retry with exponential backoff
// ---------------------------------------------------------------------------

async fn send_with_retry(
    client: &reqwest::Client,
    method: reqwest::Method,
    url: &str,
    headers: reqwest::header::HeaderMap,
    body: Bytes,
    max_retries: u32,
    retry_base_ms: u64,
) -> Result<reqwest::Response, reqwest::Error> {
    let mut attempt = 0u32;
    loop {
        let mut req = client.request(method.clone(), url);
        req = req.headers(headers.clone());
        if !body.is_empty() {
            req = req.body(body.to_vec());
        }

        match req.send().await {
            Ok(resp) => {
                // Only retry 5xx for idempotent methods (GET, HEAD, OPTIONS).
                // POST/PUT/PATCH are not idempotent — LLM providers may have
                // already billed tokens or triggered side effects.
                let is_idempotent = matches!(
                    method,
                    reqwest::Method::GET | reqwest::Method::HEAD | reqwest::Method::OPTIONS
                );
                if resp.status().is_server_error() && is_idempotent && attempt < max_retries {
                    tracing::warn!(
                        "upstream returned {} (attempt {}/{}), retrying",
                        resp.status(),
                        attempt + 1,
                        max_retries
                    );
                } else {
                    return Ok(resp);
                }
            }
            Err(e) => {
                if attempt < max_retries {
                    tracing::warn!(
                        "upstream network error (attempt {}/{}): {e}",
                        attempt + 1,
                        max_retries
                    );
                } else {
                    return Err(e);
                }
            }
        }

        // Exponential backoff with jitter (10-25% of base delay).
        let base = retry_base_ms.saturating_mul(1u64 << attempt);
        // Simple deterministic jitter derived from current time nanos.
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos() as u64;
        let jitter_range = base / 4; // 25% max
        let jitter = if jitter_range > 0 {
            (nanos % jitter_range).max(base / 10) // at least 10%
        } else {
            0
        };
        let delay = base + jitter;
        tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
        attempt += 1;
    }
}

/// Handle a single proxied request.
async fn handle_request(
    req: hyper::Request<hyper::body::Incoming>,
    config: ReinConfig,
    client: reqwest::Client,
    expected_token: Option<&str>,
    state: Arc<ProxyState>,
) -> Result<hyper::Response<BoxBody>, hyper::Error> {
    state.metrics.request_count.fetch_add(1, Ordering::Relaxed);

    // Auth check for non-localhost binds.
    // Constant-time compare to prevent timing side channels that would leak the token byte by byte.
    if let Some(expected) = expected_token {
        let auth_header = req
            .headers()
            .get("x-rein-token")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        if !proxy_token_eq(auth_header, expected) {
            state.metrics.error_count.fetch_add(1, Ordering::Relaxed);
            return Ok(error_response(401, "unauthorized"));
        }
    }

    let method = req.method().clone();
    let uri = req.uri().clone();
    // Preserve query string for passthrough (e.g., /v1/models?foo=bar).
    let path_and_query = uri
        .path_and_query()
        .map(|pq| pq.as_str().to_string())
        .unwrap_or_else(|| uri.path().to_string());
    let path = uri.path().to_string();
    let headers = req.headers().clone();

    // Reject path-traversal segments before ANY routing decision.
    // `reqwest`/`url` normalize `/foo/../bar` → `/bar` on the wire, which means
    // the path we use for provider detection, recording-mode selection, and
    // ArtifactMirrorOnly gating can disagree with what upstream actually sees.
    // That divergence breaks the "the policy decision matches the wire path"
    // invariant; the safest behavior is to refuse such paths at the edge.
    if has_traversal_segments(&path_and_query) {
        state.metrics.error_count.fetch_add(1, Ordering::Relaxed);
        return Ok(error_response(400, "path contains traversal segments"));
    }

    // Metrics endpoint.
    if method == hyper::Method::GET && path == "/rein/metrics" {
        let json = state.metrics.to_json();
        return Ok(build_response(
            hyper::Response::builder()
                .status(200)
                .header("content-type", "application/json"),
            full_body(Bytes::from(json)),
        ));
    }

    // Detect provider from request path plus auth semantics. Codex ChatGPT-login
    // traffic can hit public-looking paths like `/responses` and `/models`, but
    // still belongs to the first-party backend route family.
    let route = resolve_route(&path, req.headers());

    if let Some(route) = route {
        let provider_kind = route.provider;
        if let Some(msg) = responses_scope_error(provider_kind, &method, req.headers()) {
            state.metrics.error_count.fetch_add(1, Ordering::Relaxed);
            return Ok(error_response(401, &msg));
        }
    }

    if method == hyper::Method::GET && is_websocket_upgrade(req.headers()) {
        if let Some(route) = route.filter(|route| {
            route.recording_mode.captures_structured_text()
                && route.provider.supports_websocket_passthrough()
        }) {
            return handle_websocket_proxy(
                req,
                &config,
                &path_and_query,
                &headers,
                route.provider,
            )
            .await;
        }
    }

    if let Some(content_length) = headers
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<usize>().ok())
    {
        if content_length > config.proxy.max_request_body {
            state.metrics.error_count.fetch_add(1, Ordering::Relaxed);
            return Ok(error_response(413, "request body too large"));
        }
    }

    // Read request body incrementally with size cap (handles chunked TE too).
    let max_body = config.proxy.max_request_body;
    let mut body_buf = Vec::new();
    let mut body_stream = req.into_body();
    loop {
        let frame = match body_stream.frame().await {
            Some(Ok(f)) => f,
            Some(Err(e)) => {
                tracing::warn!("failed to read request body: {e}");
                state.metrics.error_count.fetch_add(1, Ordering::Relaxed);
                return Ok(error_response(400, "failed to read request body"));
            }
            None => break,
        };
        if let Some(data) = frame.data_ref() {
            body_buf.extend_from_slice(data);
            if body_buf.len() > max_body {
                state.metrics.error_count.fetch_add(1, Ordering::Relaxed);
                return Ok(error_response(413, "request body too large"));
            }
        }
    }
    let body_bytes = Bytes::from(body_buf);

    let body_size = body_bytes.len();
    tracing::debug!(%method, path = %path_and_query, body_size, "proxy request");

    // If not a known sampling endpoint, passthrough unmodified.
    let route = match route {
        Some(route) => route,
        None => {
            return forward_raw(
                &client,
                &config,
                &method,
                &path_and_query,
                &headers,
                body_bytes,
            )
            .await;
        }
    };

    let provider = route.provider;
    let query = if route.recording_mode.captures_structured_text() {
        extract_query_for_recording(&provider, &body_bytes)
    } else {
        None
    };
    tracing::debug!(
        query_len = query.as_deref().map(str::len).unwrap_or(0),
        orig_bytes = body_size,
        "proxy record-only request"
    );
    let artifact_input = ProxyArtifactInput {
        route,
        method: method.to_string(),
        path: path_and_query.clone(),
        session_id: extract_session_id(&headers),
        request_headers: capture_request_headers(&headers),
        request_body: body_bytes.clone(),
    };

    // Build upstream URL (rewrite path if needed, e.g. /responses → /v1/responses).
    let upstream_base = provider.upstream_url(&config);
    let rewritten_path = provider.rewrite_path(&path_and_query);
    let upstream_url = format!("{upstream_base}{rewritten_path}");

    // Build upstream headers (skip hop-by-hop, recalculate content-length).
    let mut upstream_headers = reqwest::header::HeaderMap::new();
    for (name, value) in headers.iter() {
        let name_str = name.as_str();
        if should_strip_request_header(name_str) {
            continue;
        }
        if let Ok(rname) = reqwest::header::HeaderName::from_bytes(name.as_str().as_bytes()) {
            if let Ok(rval) = reqwest::header::HeaderValue::from_bytes(value.as_ref()) {
                upstream_headers.insert(rname, rval);
            }
        }
    }
    if let Ok(cl) = reqwest::header::HeaderValue::from_str(&body_bytes.len().to_string()) {
        upstream_headers.insert(reqwest::header::CONTENT_LENGTH, cl);
    }

    let req_method =
        reqwest::Method::from_bytes(method.as_str().as_bytes()).unwrap_or(reqwest::Method::POST);

    // Send upstream request with retry.
    let upstream_resp = match send_with_retry(
        &client,
        req_method,
        &upstream_url,
        upstream_headers,
        body_bytes,
        config.proxy.max_retries,
        config.proxy.retry_base_ms,
    )
    .await
    {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("upstream request failed: {e}");
            state.metrics.error_count.fetch_add(1, Ordering::Relaxed);
            return Ok(error_response(502, "upstream request failed"));
        }
    };

    let status = upstream_resp.status();
    eprintln!("rein proxy: upstream responded {status}");
    let resp_headers = upstream_resp.headers().clone();
    let is_streaming = resp_headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .map(|ct| ct.contains("text/event-stream"))
        .unwrap_or(false);

    if is_streaming {
        stream_response(
            upstream_resp,
            status,
            &resp_headers,
            &config,
            artifact_input,
            query,
            &state,
        )
        .await
    } else {
        // Large response streaming: if Content-Length exceeds max_response_buffer,
        // stream directly without buffering/extraction.
        let content_length: usize = resp_headers
            .get("content-length")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        if content_length > config.proxy.max_response_buffer {
            tracing::info!(
                "response too large ({content_length} bytes), streaming without extraction"
            );
            return stream_response(
                upstream_resp,
                status,
                &resp_headers,
                &config,
                artifact_input,
                query,
                &state,
            )
            .await;
        }

        // Non-streaming: read full response, extract, forward.
        let resp_body = match upstream_resp.bytes().await {
            Ok(body) => body,
            Err(error) => {
                tracing::warn!("failed to read upstream response body: {error}");
                state.metrics.error_count.fetch_add(1, Ordering::Relaxed);
                return Ok(error_response(502, "bad gateway"));
            }
        };

        // Async extract from non-streaming response (with backpressure).
        if config.proxy.extract_enabled && route.recording_mode.captures_structured_text() {
            if let Some(text) = provider.extract_assistant_text_full(&resp_body) {
                if policy::should_extract_response(&config, query.as_deref(), &text) {
                    maybe_spawn_extraction(&config, &state, query.clone(), text);
                }
            }
        }

        maybe_store_first_party_artifact(
            &config,
            artifact_input,
            status.as_u16(),
            capture_response_headers(&resp_headers),
            resp_body.to_vec(),
            false,
            false,
        );

        let mut builder = hyper::Response::builder().status(status.as_u16());
        for (name, value) in resp_headers.iter() {
            if name.as_str() != "transfer-encoding" {
                builder = builder.header(name.as_str(), value);
            }
        }
        Ok(build_response(builder, full_body(resp_body)))
    }
}

/// Spawn an extraction task. extract_and_store only appends to a durable
/// file-based queue (cheap I/O), so no concurrency limiting is needed here.
/// Actual LLM extraction happens in the background worker with its own rate limiting.
fn maybe_spawn_extraction(
    config: &ReinConfig,
    state: &Arc<ProxyState>,
    query: Option<String>,
    text: String,
) {
    state
        .metrics
        .extraction_count
        .fetch_add(1, Ordering::Relaxed);
    let cfg = config.clone();
    tokio::spawn(async move {
        extract::extract_and_store(&cfg, query, text).await;
    });
}

/// Stream SSE response back to client while buffering assistant text.
async fn stream_response(
    upstream_resp: reqwest::Response,
    status: reqwest::StatusCode,
    resp_headers: &reqwest::header::HeaderMap,
    config: &ReinConfig,
    artifact: ProxyArtifactInput,
    query: Option<String>,
    state: &Arc<ProxyState>,
) -> Result<hyper::Response<BoxBody>, hyper::Error> {
    let (tx, rx) = mpsc::channel::<Result<Frame<Bytes>, hyper::Error>>(64);

    let extract_enabled =
        config.proxy.extract_enabled && artifact.route.recording_mode.captures_structured_text();
    let max_sse_buffer = config.proxy.max_sse_buffer;
    let provider_clone = artifact.route.provider;
    let config_clone = config.clone();
    let query_clone = query.clone();
    let state_clone = Arc::clone(state);
    let response_headers = capture_response_headers(resp_headers);

    // Spawn task to read upstream stream, forward chunks, buffer text.
    tokio::spawn(async move {
        let mut stream = upstream_resp.bytes_stream();
        let mut assistant_buf = String::new();
        let mut raw_response_buf = Vec::new();
        // SSE line buffer: transport chunks may split across SSE event boundaries.
        let mut sse_line_buf = String::new();
        // Whether SSE parsing has been abandoned due to buffer overflow.
        let mut sse_parsing_active = true;
        const MAX_EXTRACT_BUF: usize = 200_000; // ~50K tokens, prevent OOM
        const MAX_ARTIFACT_BUF: usize = 256_000;
        let mut artifact_truncated = false;

        use futures_util::StreamExt;
        let mut clean_completion = false;
        loop {
            let chunk_result = match stream.next().await {
                Some(chunk_result) => chunk_result,
                None => {
                    clean_completion = true;
                    break;
                }
            };
            match chunk_result {
                Ok(chunk) => {
                    if raw_response_buf.len() < MAX_ARTIFACT_BUF {
                        let remaining = MAX_ARTIFACT_BUF.saturating_sub(raw_response_buf.len());
                        let to_copy = remaining.min(chunk.len());
                        raw_response_buf.extend_from_slice(&chunk[..to_copy]);
                        if to_copy < chunk.len() {
                            artifact_truncated = true;
                        }
                    } else {
                        artifact_truncated = true;
                    }
                    // Parse SSE chunks for assistant text extraction.
                    // Buffer incomplete lines across chunk boundaries.
                    if extract_enabled
                        && sse_parsing_active
                        && assistant_buf.len() < MAX_EXTRACT_BUF
                    {
                        if let Ok(text) = std::str::from_utf8(&chunk) {
                            sse_line_buf.push_str(text);

                            // Cap SSE line buffer to prevent unbounded growth.
                            if sse_line_buf.len() > max_sse_buffer {
                                tracing::warn!(
                                    "SSE line buffer exceeded {} bytes, forwarding raw (skipping extraction)",
                                    max_sse_buffer
                                );
                                sse_parsing_active = false;
                                sse_line_buf.clear();
                                assistant_buf.clear(); // discard partial content — don't extract truncated responses
                            } else {
                                // Process only complete lines (ending with \n).
                                while let Some(newline_pos) = sse_line_buf.find('\n') {
                                    let line = sse_line_buf[..newline_pos].to_string();
                                    sse_line_buf = sse_line_buf[newline_pos + 1..].to_string();
                                    if let Some(extracted) =
                                        provider_clone.extract_assistant_text_sse(line.as_bytes())
                                    {
                                        assistant_buf.push_str(&extracted);
                                    }
                                }
                            }
                        }
                    }
                    // Forward chunk to client (unmodified, byte-perfect).
                    if tx.send(Ok(Frame::data(chunk))).await.is_err() {
                        break; // Client disconnected.
                    }
                }
                Err(e) => {
                    tracing::warn!("upstream stream error: {e}");
                    break;
                }
            }
        }
        drop(tx); // Signal end of stream.

        // After stream completes, extract memories (with backpressure).
        if clean_completion
            && extract_enabled
            && policy::should_extract_response(
                &config_clone,
                query_clone.as_deref(),
                &assistant_buf,
            )
        {
            maybe_spawn_extraction(&config_clone, &state_clone, query_clone, assistant_buf);
        }

        maybe_store_first_party_artifact(
            &config_clone,
            artifact,
            status.as_u16(),
            response_headers,
            raw_response_buf,
            artifact_truncated,
            true,
        );
    });

    // Build response with streaming body.
    let body_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    let stream_body = StreamBody::new(body_stream);

    let mut builder = hyper::Response::builder().status(status.as_u16());
    for (name, value) in resp_headers.iter() {
        if name.as_str() != "transfer-encoding" {
            builder = builder.header(name.as_str(), value);
        }
    }

    Ok(build_response(builder, box_body(stream_body)))
}

/// Forward a request unmodified (for non-LLM endpoints like /v1/models).
async fn forward_raw(
    client: &reqwest::Client,
    config: &ReinConfig,
    method: &hyper::Method,
    path: &str,
    headers: &hyper::HeaderMap,
    body: Bytes,
) -> Result<hyper::Response<BoxBody>, hyper::Error> {
    // Try to guess upstream from path pattern.
    let upstream_base =
        if path.starts_with("/v1/messages") || headers.get("anthropic-version").is_some() {
            &config.proxy.anthropic_upstream
        } else {
            &config.proxy.openai_upstream
        };
    let upstream_url = format!("{upstream_base}{path}");

    let mut req = client.request(
        reqwest::Method::from_bytes(method.as_str().as_bytes()).unwrap_or(reqwest::Method::GET),
        &upstream_url,
    );
    for (name, value) in headers.iter() {
        if !should_strip_request_header(name.as_str()) {
            if let Ok(v) = value.to_str() {
                req = req.header(name.as_str(), v);
            }
        }
    }
    if !body.is_empty() {
        req = req.body(body.to_vec());
    }

    match req.send().await {
        Ok(resp) => {
            let status = resp.status();
            let resp_headers = resp.headers().clone();
            // Cap response body to prevent unbounded memory allocation.
            let max_raw = config.proxy.max_response_buffer;
            let mut resp_bytes = Vec::new();
            let mut stream = resp.bytes_stream();
            use futures_util::StreamExt;
            let mut exceeded = false;
            while let Some(chunk) = stream.next().await {
                match chunk {
                    Ok(c) => {
                        resp_bytes.extend_from_slice(&c);
                        if resp_bytes.len() > max_raw {
                            exceeded = true;
                            break;
                        }
                    }
                    Err(e) => {
                        tracing::warn!("forward_raw upstream read error: {e}");
                        return Ok(error_response(502, "upstream read failed"));
                    }
                }
            }
            if exceeded {
                tracing::warn!("forward_raw: response exceeded {max_raw} bytes, returning 502");
                return Ok(error_response(502, "upstream response too large"));
            }
            let resp_body = Bytes::from(resp_bytes);

            let mut builder = hyper::Response::builder().status(status.as_u16());
            for (name, value) in resp_headers.iter() {
                if name.as_str() != "transfer-encoding" {
                    builder = builder.header(name.as_str(), value);
                }
            }
            Ok(build_response(builder, full_body(resp_body)))
        }
        Err(e) => {
            tracing::warn!("forward_raw upstream error: {e}");
            Ok(error_response(502, "upstream request failed"))
        }
    }
}

fn extract_query_for_recording(provider: &ProviderKind, body_bytes: &[u8]) -> Option<String> {
    let body: serde_json::Value = serde_json::from_slice(body_bytes).ok()?;
    let query = provider.extract_query(&body);
    let query = query.trim();
    if query.is_empty() {
        None
    } else {
        Some(query.to_string())
    }
}

fn resolve_route(path: &str, headers: &hyper::HeaderMap) -> Option<ResolvedRoute> {
    let provider = if prefers_codex_first_party(path, headers) {
        ProviderKind::CodexFirstParty
    } else if prefers_chatgpt_backend(path, headers) {
        ProviderKind::ChatGptBackend
    } else {
        ProviderKind::detect(path)?
    };
    Some(ResolvedRoute {
        provider,
        recording_mode: provider.recording_mode_for_path(path),
    })
}

fn prefers_codex_first_party(path: &str, headers: &hyper::HeaderMap) -> bool {
    if ProviderKind::detect(path) == Some(ProviderKind::CodexFirstParty) {
        return true;
    }
    if !ProviderKind::is_ambiguous_codex_first_party_path(path) {
        return false;
    }
    // Ambiguous path resolution: when the client sent a bare/first-party path like
    // `/responses`, `/models`, or `/authenticate_app_v2`, a ChatGPT-login JWT is
    // sufficient evidence to route to the Codex first-party upstream.
    bearer_jwt_info(headers).is_some_and(|info| info.is_chatgpt_login)
}

fn prefers_chatgpt_backend(path: &str, _headers: &hyper::HeaderMap) -> bool {
    if ProviderKind::detect(path) == Some(ProviderKind::ChatGptBackend) {
        return true;
    }
    if !ProviderKind::is_ambiguous_chatgpt_backend_path(path) {
        return false;
    }
    true
}

fn should_strip_request_header(name: &str) -> bool {
    matches!(
        name,
        "host"
            | "content-length"
            | "transfer-encoding"
            | "connection"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "x-rein-token"
    )
}

async fn handle_websocket_proxy(
    req: hyper::Request<hyper::body::Incoming>,
    config: &ReinConfig,
    path_and_query: &str,
    headers: &hyper::HeaderMap,
    provider: ProviderKind,
) -> Result<hyper::Response<BoxBody>, hyper::Error> {
    let recording_mode = provider.recording_mode_for_path(path_and_query);
    // H5: Only /responses-family routes (StructuredText) are eligible for WS upgrade.
    // First-party ChatGPT backend helper routes (/wham/*, /connectors/*, /models, etc.)
    // are recorded as ArtifactMirrorOnly over HTTP and must NEVER upgrade to WS —
    // Codex does not use WS for those endpoints, so an upgrade request there is
    // either a protocol confusion or an attacker probing for an attack surface.
    if !recording_mode.captures_structured_text() {
        tracing::warn!(
            path = %path_and_query,
            "rejecting websocket upgrade on ArtifactMirrorOnly route"
        );
        return Ok(websocket_upstream_failure_response(provider));
    }

    let artifact = ProxyArtifactInput {
        route: ResolvedRoute {
            provider,
            recording_mode,
        },
        method: "GET".to_string(),
        path: path_and_query.to_string(),
        session_id: extract_session_id(headers),
        request_headers: capture_request_headers(headers),
        request_body: Bytes::new(),
    };
    let client_key = match headers
        .get("sec-websocket-key")
        .and_then(|v| v.to_str().ok())
    {
        Some(key) if !key.trim().is_empty() => key.to_string(),
        _ => return Ok(error_response(400, "missing sec-websocket-key")),
    };

    let upstream = match connect_upstream_websocket(config, provider, path_and_query, headers).await
    {
        Ok(upstream) => upstream,
        Err(UpstreamWsError::Unauthorized) => {
            // H8: upstream 401 is semantically meaningful — Codex clients do a single
            // refresh-retry on 401. Rewriting to 426/502 breaks their retry loop. Pass
            // the 401 through unchanged so the client sees exactly what upstream said.
            tracing::info!(
                "websocket upstream returned 401; passing through for client refresh"
            );
            return Ok(error_response(
                401,
                "websocket upstream rejected credentials",
            ));
        }
        Err(UpstreamWsError::Other(e)) => {
            tracing::warn!("websocket upstream connect failed: {e}");
            return Ok(websocket_upstream_failure_response(provider));
        }
    };
    let response_protocol = upstream.protocol.clone();
    let accept = websocket_accept(&client_key);
    let on_upgrade = hyper::upgrade::on(req);
    let config = config.clone();
    tokio::spawn(async move {
        match on_upgrade.await {
            Ok(upgraded) => {
                relay_websocket_with_mirror(
                    &config,
                    artifact,
                    hyper_util::rt::TokioIo::new(upgraded),
                    upstream,
                )
                .await;
            }
            Err(e) => tracing::warn!("client websocket upgrade failed: {e}"),
        }
    });

    let mut builder = hyper::Response::builder()
        .status(101)
        .header("connection", "Upgrade")
        .header("upgrade", "websocket")
        .header("sec-websocket-accept", accept);
    if let Some(protocol) = response_protocol {
        builder = builder.header("sec-websocket-protocol", protocol);
    }
    Ok(build_response(builder, full_body(Bytes::new())))
}

struct UpstreamWebsocket {
    stream: UpstreamWsStream,
    protocol: Option<String>,
    headers: Vec<(String, String)>,
}

/// Typed error for upstream WebSocket connection attempts. `Unauthorized` is
/// distinguished so the proxy can pass 401 through to the client unchanged
/// (Codex clients do a single refresh-retry on 401; rewriting breaks their retry).
enum UpstreamWsError {
    Unauthorized,
    Other(anyhow::Error),
}

impl<E: Into<anyhow::Error>> From<E> for UpstreamWsError {
    fn from(e: E) -> Self {
        UpstreamWsError::Other(e.into())
    }
}

async fn connect_upstream_websocket(
    config: &ReinConfig,
    provider: ProviderKind,
    path_and_query: &str,
    headers: &hyper::HeaderMap,
) -> Result<UpstreamWebsocket, UpstreamWsError> {
    let upstream_base = provider.upstream_url(config);
    let rewritten_path = provider.rewrite_path(path_and_query).into_owned();
    let http_url = format!("{upstream_base}{rewritten_path}");
    let ws_url = if let Some(rest) = http_url.strip_prefix("https://") {
        format!("wss://{rest}")
    } else if let Some(rest) = http_url.strip_prefix("http://") {
        format!("ws://{rest}")
    } else if http_url.starts_with("wss://") || http_url.starts_with("ws://") {
        http_url.clone()
    } else {
        return Err(UpstreamWsError::Other(anyhow::anyhow!(
            "unsupported websocket scheme in upstream url: {http_url}"
        )));
    };

    let mut request = ws_url
        .clone()
        .into_client_request()
        .map_err(|e| UpstreamWsError::Other(anyhow::anyhow!("failed to build websocket request: {e}")))?;
    for (name, value) in headers.iter() {
        let name_str = name.as_str();
        if should_strip_ws_handshake_header(name_str) {
            continue;
        }
        request.headers_mut().insert(name, value.clone());
    }
    if request.headers().contains_key("origin") {
        if let Some(origin) = websocket_upstream_origin(&ws_url) {
            if let Ok(value) = hyper::header::HeaderValue::from_str(&origin) {
                request.headers_mut().insert("origin", value);
            }
        }
    }
    if !request.headers().contains_key("originator") {
        request.headers_mut().insert(
            "originator",
            hyper::header::HeaderValue::from_static(default_codex_originator()),
        );
    }
    if !request.headers().contains_key("user-agent") {
        if let Ok(value) = hyper::header::HeaderValue::from_str(&default_codex_user_agent()) {
            request.headers_mut().insert("user-agent", value);
        }
    }

    let (stream, response) = match connect_async(request).await {
        Ok(pair) => pair,
        Err(WsError::Http(resp)) if resp.status().as_u16() == 401 => {
            return Err(UpstreamWsError::Unauthorized);
        }
        Err(e) => {
            return Err(UpstreamWsError::Other(anyhow::anyhow!(
                "upstream websocket handshake failed: {e}"
            )));
        }
    };
    let response_headers = response
        .headers()
        .iter()
        .filter_map(|(name, value)| {
            value.to_str().ok().map(|value| {
                (
                    name.as_str().to_ascii_lowercase(),
                    redact_header_value(name.as_str(), value),
                )
            })
        })
        .collect::<Vec<_>>();

    Ok(UpstreamWebsocket {
        stream,
        protocol: response
            .headers()
            .get("sec-websocket-protocol")
            .and_then(|value| value.to_str().ok())
            .map(ToString::to_string),
        headers: response_headers,
    })
}

async fn relay_websocket_with_mirror(
    config: &ReinConfig,
    artifact: ProxyArtifactInput,
    upgraded: hyper_util::rt::TokioIo<hyper::upgrade::Upgraded>,
    upstream: UpstreamWebsocket,
) {
    let response_headers = upstream.headers.clone();
    let should_collect = artifact.route.recording_mode.captures_structured_text();
    let client_ws = WebSocketStream::from_raw_socket(upgraded, Role::Server, None).await;
    let (mut client_sink, mut client_stream) = client_ws.split();
    let (mut upstream_sink, mut upstream_stream) = upstream.stream.split();
    let mut request_mirror = WebSocketMirrorState::default();
    let mut response_mirror = WebSocketMirrorState::default();

    loop {
        tokio::select! {
            upstream_msg = upstream_stream.next() => {
                match upstream_msg {
                    Some(Ok(message)) => {
                        if should_collect {
                            response_mirror.record_message(&message, true);
                        }
                        let is_close = matches!(message, Message::Close(_));
                        if client_sink.send(message).await.is_err() {
                            break;
                        }
                        if is_close {
                            break;
                        }
                    }
                    Some(Err(e)) => {
                        if is_benign_websocket_read_error(&e) {
                            tracing::debug!("websocket upstream relay ended benignly: {e}");
                        } else {
                            tracing::warn!("websocket upstream relay read failed: {e}");
                        }
                        break;
                    }
                    None => break,
                }
            }
            client_msg = client_stream.next() => {
                match client_msg {
                    Some(Ok(message)) => {
                        if should_collect {
                            request_mirror.record_message(&message, false);
                        }
                        let is_close = matches!(message, Message::Close(_));
                        if upstream_sink.send(message).await.is_err() {
                            break;
                        }
                        if is_close {
                            break;
                        }
                    }
                    Some(Err(e)) => {
                        if is_benign_websocket_read_error(&e) {
                            tracing::debug!("websocket client relay ended benignly: {e}");
                        } else {
                            tracing::warn!("websocket client relay read failed: {e}");
                        }
                        break;
                    }
                    None => break,
                }
            }
        }
    }

    let _ = tokio::time::timeout(Duration::from_millis(250), async {
        let _ = client_sink.close().await;
        let _ = upstream_sink.close().await;
    })
    .await;

    let request_query = request_mirror.request_query.clone();
    if should_collect
        && config.proxy.extract_enabled
        && policy::should_extract_response(config, request_query.as_deref(), &response_mirror.assistant_text)
    {
        let state = Arc::new(ProxyState {
            metrics: ProxyMetrics::new(),
        });
        maybe_spawn_extraction(
            config,
            &state,
            request_query,
            response_mirror.assistant_text.clone(),
        );
    }

    maybe_store_first_party_ws_artifact(
        config,
        artifact,
        response_headers,
        request_mirror.event_messages,
        response_mirror.event_messages,
        request_mirror.truncated || response_mirror.truncated,
    );
}

fn websocket_accept(key: &str) -> String {
    let mut ctx = ring::digest::Context::new(&ring::digest::SHA1_FOR_LEGACY_USE_ONLY);
    ctx.update(key.as_bytes());
    ctx.update(WS_GUID.as_bytes());
    BASE64_STANDARD.encode(ctx.finish())
}

fn is_websocket_upgrade(headers: &hyper::HeaderMap) -> bool {
    let upgrade = headers
        .get("upgrade")
        .and_then(|v| v.to_str().ok())
        .map(|v| v.to_ascii_lowercase())
        .unwrap_or_default();
    let connection = headers
        .get("connection")
        .and_then(|v| v.to_str().ok())
        .map(|v| v.to_ascii_lowercase())
        .unwrap_or_default();
    upgrade.contains("websocket") && connection.contains("upgrade")
}

fn should_strip_ws_handshake_header(name: &str) -> bool {
    matches!(
        name,
        "host"
            | "content-length"
            | "transfer-encoding"
            | "connection"
            | "upgrade"
            | "sec-websocket-key"
            | "sec-websocket-version"
            | "sec-websocket-extensions"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "x-rein-token"
    )
}

fn responses_scope_error(
    provider: ProviderKind,
    method: &hyper::Method,
    headers: &hyper::HeaderMap,
) -> Option<String> {
    if provider != ProviderKind::OpenAiResponses {
        return None;
    }

    let required_scope = if method == hyper::Method::GET {
        "api.responses.read"
    } else if method == hyper::Method::POST {
        "api.responses.write"
    } else {
        return None;
    };

    let info = bearer_jwt_info(headers)?;
    if info.scopes.iter().any(|scope| scope == required_scope) {
        return None;
    }

    // Emit a redacted diagnostic so ops can debug scope mismatches without
    // leaking the full JWT payload. `redact_jwt_payload` keeps only iss/aud/
    // exp/iat/nbf/scp + has_chatgpt_login. The `account_fingerprint` field
    // is a non-invertible short hash of chatgpt_account_id (when present),
    // so two calls from the same subscription group together in the log
    // without exposing the raw account id.
    if let Some(decoded) = decode_jwt_payload_for_logging(headers) {
        let account_fp = decoded
            .get("https://api.openai.com/auth")
            .and_then(|v| v.get("chatgpt_account_id"))
            .or_else(|| decoded.get("chatgpt_account_id"))
            .and_then(|v| v.as_str())
            .map(jwt::hashed_account_fingerprint);
        tracing::info!(
            required_scope = %required_scope,
            redacted_claims = %redact_jwt_payload(&decoded),
            account_fingerprint = account_fp.as_deref().unwrap_or("<none>"),
            "responses scope check failed; bearer does not carry the required scope"
        );
    }

    Some(format!(
        "Codex proxy request blocked locally: bearer token is missing required scope '{required_scope}'. \
This usually means ChatGPT login tokens are not compatible with OpenAI Responses API proxying."
    ))
}

// JWT decode helpers moved to `src/proxy/jwt.rs` in v0.18.1 (audit 'ws_mirror + jwt + routing' split).
// Re-imported below via the module declaration at the top of this file.

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;
    use std::collections::HashMap;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc::{self, Receiver};
    use std::thread;
    use std::time::Duration;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    struct CapturedRequest {
        method: String,
        path: String,
        headers: HashMap<String, String>,
        body: Vec<u8>,
    }

    fn find_header_end_for_test(buf: &[u8]) -> Option<usize> {
        buf.windows(4).position(|window| window == b"\r\n\r\n")
    }

    fn spawn_capture_http_server(
        response_status: &str,
        response_body: &str,
    ) -> (String, Receiver<CapturedRequest>) {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let addr = listener.local_addr().unwrap();
        let response_status = response_status.to_string();
        let response_body = response_body.to_string();
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = Vec::new();
            let mut tmp = [0u8; 4096];
            let header_end = loop {
                let n = stream.read(&mut tmp).unwrap();
                if n == 0 {
                    return;
                }
                buf.extend_from_slice(&tmp[..n]);
                if let Some(end) = find_header_end_for_test(&buf) {
                    break end;
                }
            };

            let header_text = String::from_utf8_lossy(&buf[..header_end]);
            let mut lines = header_text.split("\r\n");
            let request_line = lines.next().unwrap_or_default();
            let mut parts = request_line.split_whitespace();
            let method = parts.next().unwrap_or_default().to_string();
            let path = parts.next().unwrap_or_default().to_string();
            let mut headers = HashMap::new();
            for line in lines {
                if let Some((name, value)) = line.split_once(':') {
                    headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_string());
                }
            }
            let content_length = headers
                .get("content-length")
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(0);
            let mut body = buf[header_end + 4..].to_vec();
            while body.len() < content_length {
                let n = stream.read(&mut tmp).unwrap();
                if n == 0 {
                    break;
                }
                body.extend_from_slice(&tmp[..n]);
            }
            tx.send(CapturedRequest {
                method,
                path,
                headers,
                body,
            })
            .unwrap();

            let response = format!(
                "HTTP/1.1 {response_status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response_body}",
                response_body.len()
            );
            let _ = stream.write_all(response.as_bytes());
        });
        (format!("http://{}", addr), rx)
    }

    fn spawn_capture_websocket_server() -> (String, Receiver<CapturedRequest>) {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let addr = listener.local_addr().unwrap();
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = Vec::new();
            let mut tmp = [0u8; 4096];
            let header_end = loop {
                let n = stream.read(&mut tmp).unwrap();
                if n == 0 {
                    return;
                }
                buf.extend_from_slice(&tmp[..n]);
                if let Some(end) = find_header_end_for_test(&buf) {
                    break end;
                }
            };

            let header_text = String::from_utf8_lossy(&buf[..header_end]);
            let mut lines = header_text.split("\r\n");
            let request_line = lines.next().unwrap_or_default();
            let mut parts = request_line.split_whitespace();
            let method = parts.next().unwrap_or_default().to_string();
            let path = parts.next().unwrap_or_default().to_string();
            let mut headers = HashMap::new();
            for line in lines {
                if let Some((name, value)) = line.split_once(':') {
                    headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_string());
                }
            }
            let accept = headers
                .get("sec-websocket-key")
                .map(|value| websocket_accept(value))
                .unwrap_or_else(|| "invalid".to_string());
            tx.send(CapturedRequest {
                method,
                path,
                headers,
                body: Vec::new(),
            })
            .unwrap();

            let response = format!(
                "HTTP/1.1 101 Switching Protocols\r\n\
Upgrade: websocket\r\n\
Connection: Upgrade\r\n\
Sec-WebSocket-Accept: {accept}\r\n\
\r\n"
            );
            let _ = stream.write_all(response.as_bytes());
        });
        (format!("http://{}", addr), rx)
    }

    fn encode_ws_frame(opcode: u8, payload: &[u8], fin: bool, rsv1: bool) -> Vec<u8> {
        let mut frame = Vec::with_capacity(payload.len() + 10);
        let mut first = opcode & 0x0f;
        if fin {
            first |= 0x80;
        }
        if rsv1 {
            first |= 0x40;
        }
        frame.push(first);
        match payload.len() {
            len @ 0..=125 => frame.push(len as u8),
            len @ 126..=65535 => {
                frame.push(126);
                frame.extend_from_slice(&(len as u16).to_be_bytes());
            }
            len => {
                frame.push(127);
                frame.extend_from_slice(&(len as u64).to_be_bytes());
            }
        }
        frame.extend_from_slice(payload);
        frame
    }

    fn encode_masked_ws_frame(opcode: u8, payload: &[u8], fin: bool) -> Vec<u8> {
        let mut frame = Vec::with_capacity(payload.len() + 14);
        let mut first = opcode & 0x0f;
        if fin {
            first |= 0x80;
        }
        frame.push(first);
        let mask = [0x11, 0x22, 0x33, 0x44];
        match payload.len() {
            len @ 0..=125 => frame.push(0x80 | len as u8),
            len @ 126..=65535 => {
                frame.push(0x80 | 126);
                frame.extend_from_slice(&(len as u16).to_be_bytes());
            }
            len => {
                frame.push(0x80 | 127);
                frame.extend_from_slice(&(len as u64).to_be_bytes());
            }
        }
        frame.extend_from_slice(&mask);
        for (index, byte) in payload.iter().enumerate() {
            frame.push(byte ^ mask[index % 4]);
        }
        frame
    }

    fn encode_ws_text_frame(text: &str) -> Vec<u8> {
        encode_ws_frame(0x1, text.as_bytes(), true, false)
    }

    fn encode_ws_text_frame_with_flags(text: &str, fin: bool, rsv1: bool) -> Vec<u8> {
        encode_ws_frame(0x1, text.as_bytes(), fin, rsv1)
    }

    fn encode_ws_continuation_frame(text: &str, fin: bool) -> Vec<u8> {
        encode_ws_frame(0x0, text.as_bytes(), fin, false)
    }

    fn encode_ws_compressed_text_frame(text: &str) -> Vec<u8> {
        use std::io::Write as _;
        let mut encoder =
            flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(text.as_bytes()).unwrap();
        let mut payload = encoder.finish().unwrap();
        if payload.ends_with(&[0x00, 0x00, 0xff, 0xff]) {
            payload.truncate(payload.len() - 4);
        }
        encode_ws_frame(0x1, &payload, true, true)
    }

    fn encode_ws_close_frame() -> Vec<u8> {
        vec![0x88, 0x00]
    }

    fn encode_masked_ws_text_frame(text: &str) -> Vec<u8> {
        encode_masked_ws_frame(0x1, text.as_bytes(), true)
    }

    fn spawn_capture_websocket_server_with_frames(
        frames: Vec<Vec<u8>>,
    ) -> (String, Receiver<CapturedRequest>) {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let addr = listener.local_addr().unwrap();
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = Vec::new();
            let mut tmp = [0u8; 4096];
            let header_end = loop {
                let n = stream.read(&mut tmp).unwrap();
                if n == 0 {
                    return;
                }
                buf.extend_from_slice(&tmp[..n]);
                if let Some(end) = find_header_end_for_test(&buf) {
                    break end;
                }
            };

            let header_text = String::from_utf8_lossy(&buf[..header_end]);
            let mut lines = header_text.split("\r\n");
            let request_line = lines.next().unwrap_or_default();
            let mut parts = request_line.split_whitespace();
            let method = parts.next().unwrap_or_default().to_string();
            let path = parts.next().unwrap_or_default().to_string();
            let mut headers = HashMap::new();
            for line in lines {
                if let Some((name, value)) = line.split_once(':') {
                    headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_string());
                }
            }
            let accept = headers
                .get("sec-websocket-key")
                .map(|value| websocket_accept(value))
                .unwrap_or_else(|| "invalid".to_string());
            tx.send(CapturedRequest {
                method,
                path,
                headers,
                body: Vec::new(),
            })
            .unwrap();

            let response = format!(
                "HTTP/1.1 101 Switching Protocols\r\n\
Upgrade: websocket\r\n\
Connection: Upgrade\r\n\
Sec-WebSocket-Accept: {accept}\r\n\
Sec-WebSocket-Extensions: permessage-deflate\r\n\
\r\n"
            );
            let _ = stream.write_all(response.as_bytes());
            thread::sleep(Duration::from_millis(20));
            for frame in frames {
                let _ = stream.write_all(&frame);
            }
        });
        (format!("http://{}", addr), rx)
    }

    fn unused_local_base_url(scheme: &str) -> String {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        format!("{scheme}://{}", addr)
    }

    fn wait_for_artifact_count(db_path: &std::path::Path, artifact_kind: &str) {
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        loop {
            if let Ok(conn) = Connection::open(db_path) {
                let mut stmt = conn
                    .prepare("SELECT COUNT(*) FROM session_artifacts WHERE artifact_kind = ?1")
                    .unwrap();
                let count: i64 = stmt.query_row([artifact_kind], |row| row.get(0)).unwrap();
                if count > 0 {
                    return;
                }
            }
            if std::time::Instant::now() >= deadline {
                panic!("timed out waiting for artifact_kind={artifact_kind}");
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    async fn spawn_one_shot_proxy(
        config: ReinConfig,
        expected_token: Option<String>,
    ) -> (String, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .unwrap();
        let addr = listener.local_addr().unwrap();
        let state = Arc::new(ProxyState {
            metrics: ProxyMetrics::new(),
        });
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .unwrap();
        let handle = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let io = hyper_util::rt::TokioIo::new(stream);
            let service = hyper::service::service_fn(move |req| {
                let config = config.clone();
                let client = client.clone();
                let state = Arc::clone(&state);
                let expected_token = expected_token.clone();
                async move {
                    handle_request(req, config, client, expected_token.as_deref(), state).await
                }
            });
            let _ =
                hyper_util::server::conn::auto::Builder::new(hyper_util::rt::TokioExecutor::new())
                    .serve_connection_with_upgrades(io, service)
                    .await;
        });
        (format!("http://{}", addr), handle)
    }

    fn fake_jwt_payload(payload: serde_json::Value) -> String {
        let header = BASE64_URL_SAFE_NO_PAD.encode(br#"{"alg":"none"}"#);
        let payload = BASE64_URL_SAFE_NO_PAD.encode(payload.to_string().as_bytes());
        format!("{header}.{payload}.sig")
    }

    fn fake_jwt_with_scopes(scopes: &[&str]) -> String {
        fake_jwt_payload(serde_json::json!({ "scp": scopes }))
    }

    fn fake_chatgpt_login_jwt(scopes: &[&str]) -> String {
        fake_jwt_payload(serde_json::json!({
            "scp": scopes,
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "acct_test"
            }
        }))
    }

    #[derive(Clone, Copy)]
    enum TokenFixture {
        None,
        ApiResponses,
        JwtWithoutResponsesScope,
        ChatGptLogin,
        ChatGptLoginWithResponsesScope,
    }

    #[derive(Clone, Copy)]
    struct RouteSupportCase {
        name: &'static str,
        method: &'static str,
        path: &'static str,
        token: TokenFixture,
        scope_provider: Option<ProviderKind>,
        expected_scope_error_contains: Option<&'static str>,
        expected_route: Option<ResolvedRoute>,
    }

    #[derive(Clone)]
    struct SupportMatrixRow {
        path_family: String,
        trigger: String,
        upstream: String,
        recording_mode: String,
        coverage: String,
    }

    fn route_support_cases() -> Vec<RouteSupportCase> {
        vec![
            RouteSupportCase {
                name: "anthropic_messages",
                method: "POST",
                path: "/v1/messages",
                token: TokenFixture::None,
                scope_provider: None,
                expected_scope_error_contains: None,
                expected_route: Some(ResolvedRoute {
                    provider: ProviderKind::Anthropic,
                    recording_mode: RecordingMode::StructuredText,
                }),
            },
            RouteSupportCase {
                name: "openai_chat_completions",
                method: "POST",
                path: "/v1/chat/completions",
                token: TokenFixture::None,
                scope_provider: None,
                expected_scope_error_contains: None,
                expected_route: Some(ResolvedRoute {
                    provider: ProviderKind::OpenAi,
                    recording_mode: RecordingMode::StructuredText,
                }),
            },
            RouteSupportCase {
                name: "responses_missing_public_scope",
                method: "POST",
                path: "/responses",
                token: TokenFixture::JwtWithoutResponsesScope,
                scope_provider: Some(ProviderKind::OpenAiResponses),
                expected_scope_error_contains: Some("api.responses.write"),
                expected_route: Some(ResolvedRoute {
                    provider: ProviderKind::OpenAiResponses,
                    recording_mode: RecordingMode::StructuredText,
                }),
            },
            RouteSupportCase {
                name: "responses_api_scope_routes_public",
                method: "POST",
                path: "/responses",
                token: TokenFixture::ApiResponses,
                scope_provider: Some(ProviderKind::OpenAiResponses),
                expected_scope_error_contains: None,
                expected_route: Some(ResolvedRoute {
                    provider: ProviderKind::OpenAiResponses,
                    recording_mode: RecordingMode::StructuredText,
                }),
            },
            RouteSupportCase {
                name: "chatgpt_login_responses_routes_first_party",
                method: "POST",
                path: "/responses",
                token: TokenFixture::ChatGptLogin,
                scope_provider: Some(ProviderKind::CodexFirstParty),
                expected_scope_error_contains: None,
                expected_route: Some(ResolvedRoute {
                    provider: ProviderKind::CodexFirstParty,
                    recording_mode: RecordingMode::StructuredText,
                }),
            },
            RouteSupportCase {
                name: "chatgpt_login_models_stays_artifact_mirror_only",
                method: "GET",
                path: "/models",
                token: TokenFixture::ChatGptLoginWithResponsesScope,
                scope_provider: Some(ProviderKind::CodexFirstParty),
                expected_scope_error_contains: None,
                expected_route: Some(ResolvedRoute {
                    provider: ProviderKind::CodexFirstParty,
                    recording_mode: RecordingMode::ArtifactMirrorOnly,
                }),
            },
            RouteSupportCase {
                name: "chatgpt_login_compact_stays_artifact_mirror_only",
                method: "POST",
                path: "/responses/compact",
                token: TokenFixture::ChatGptLoginWithResponsesScope,
                scope_provider: Some(ProviderKind::CodexFirstParty),
                expected_scope_error_contains: None,
                expected_route: Some(ResolvedRoute {
                    provider: ProviderKind::CodexFirstParty,
                    recording_mode: RecordingMode::ArtifactMirrorOnly,
                }),
            },
            RouteSupportCase {
                name: "chatgpt_login_memories_stays_artifact_mirror_only",
                method: "POST",
                path: "/memories/trace_summarize",
                token: TokenFixture::ChatGptLogin,
                scope_provider: Some(ProviderKind::CodexFirstParty),
                expected_scope_error_contains: None,
                expected_route: Some(ResolvedRoute {
                    provider: ProviderKind::CodexFirstParty,
                    recording_mode: RecordingMode::ArtifactMirrorOnly,
                }),
            },
            RouteSupportCase {
                name: "backend_api_responses_structured_text",
                method: "POST",
                path: "/backend-api/codex/responses",
                token: TokenFixture::None,
                scope_provider: Some(ProviderKind::CodexFirstParty),
                expected_scope_error_contains: None,
                expected_route: Some(ResolvedRoute {
                    provider: ProviderKind::CodexFirstParty,
                    recording_mode: RecordingMode::StructuredText,
                }),
            },
            RouteSupportCase {
                name: "backend_api_compact_artifact_mirror_only",
                method: "POST",
                path: "/backend-api/codex/responses/compact",
                token: TokenFixture::None,
                scope_provider: Some(ProviderKind::CodexFirstParty),
                expected_scope_error_contains: None,
                expected_route: Some(ResolvedRoute {
                    provider: ProviderKind::CodexFirstParty,
                    recording_mode: RecordingMode::ArtifactMirrorOnly,
                }),
            },
            RouteSupportCase {
                name: "backend_api_memories_artifact_mirror_only",
                method: "POST",
                path: "/backend-api/codex/memories/trace_summarize",
                token: TokenFixture::None,
                scope_provider: Some(ProviderKind::CodexFirstParty),
                expected_scope_error_contains: None,
                expected_route: Some(ResolvedRoute {
                    provider: ProviderKind::CodexFirstParty,
                    recording_mode: RecordingMode::ArtifactMirrorOnly,
                }),
            },
            RouteSupportCase {
                name: "chatgpt_backend_wham_usage",
                method: "GET",
                path: "/wham/usage",
                token: TokenFixture::ChatGptLogin,
                scope_provider: None,
                expected_scope_error_contains: None,
                expected_route: Some(ResolvedRoute {
                    provider: ProviderKind::ChatGptBackend,
                    recording_mode: RecordingMode::ArtifactMirrorOnly,
                }),
            },
            RouteSupportCase {
                name: "chatgpt_backend_connectors_directory",
                method: "POST",
                path: "/connectors/directory/list",
                token: TokenFixture::ChatGptLogin,
                scope_provider: None,
                expected_scope_error_contains: None,
                expected_route: Some(ResolvedRoute {
                    provider: ProviderKind::ChatGptBackend,
                    recording_mode: RecordingMode::ArtifactMirrorOnly,
                }),
            },
            RouteSupportCase {
                name: "backend_api_wham_usage_routes_chatgpt_backend",
                method: "GET",
                path: "/backend-api/wham/usage",
                token: TokenFixture::ChatGptLogin,
                scope_provider: None,
                expected_scope_error_contains: None,
                expected_route: Some(ResolvedRoute {
                    provider: ProviderKind::ChatGptBackend,
                    recording_mode: RecordingMode::ArtifactMirrorOnly,
                }),
            },
            RouteSupportCase {
                name: "unknown_v1_models_route_rejected",
                method: "GET",
                path: "/v1/models",
                token: TokenFixture::None,
                scope_provider: None,
                expected_scope_error_contains: None,
                expected_route: None,
            },
            // --- v0.18.x cross-product expansion (audit gap fill) ---
            // /api/codex/* family: Codex CLI emits this when chatgpt_base_url
            // does NOT contain /backend-api. rein v0.18 MUST accept it.
            RouteSupportCase {
                name: "api_codex_tasks_list_routes_first_party",
                method: "GET",
                path: "/api/codex/tasks/list",
                token: TokenFixture::None,
                scope_provider: Some(ProviderKind::CodexFirstParty),
                expected_scope_error_contains: None,
                expected_route: Some(ResolvedRoute {
                    provider: ProviderKind::CodexFirstParty,
                    recording_mode: RecordingMode::ArtifactMirrorOnly,
                }),
            },
            RouteSupportCase {
                name: "api_codex_responses_routes_first_party_structured",
                method: "POST",
                path: "/api/codex/responses",
                token: TokenFixture::ChatGptLogin,
                scope_provider: Some(ProviderKind::CodexFirstParty),
                expected_scope_error_contains: None,
                expected_route: Some(ResolvedRoute {
                    provider: ProviderKind::CodexFirstParty,
                    recording_mode: RecordingMode::StructuredText,
                }),
            },
            // /backend-api/codex/authenticate_app_v2 — Codex agent identity bootstrap.
            RouteSupportCase {
                name: "backend_api_authenticate_app_v2_mirror_only",
                method: "POST",
                path: "/backend-api/codex/authenticate_app_v2",
                token: TokenFixture::ChatGptLogin,
                scope_provider: Some(ProviderKind::CodexFirstParty),
                expected_scope_error_contains: None,
                expected_route: Some(ResolvedRoute {
                    provider: ProviderKind::CodexFirstParty,
                    recording_mode: RecordingMode::ArtifactMirrorOnly,
                }),
            },
            // /backend-api/codex/safety/arc — Codex safety/moderation endpoint.
            RouteSupportCase {
                name: "backend_api_safety_arc_mirror_only",
                method: "POST",
                path: "/backend-api/codex/safety/arc",
                token: TokenFixture::ChatGptLogin,
                scope_provider: Some(ProviderKind::CodexFirstParty),
                expected_scope_error_contains: None,
                expected_route: Some(ResolvedRoute {
                    provider: ProviderKind::CodexFirstParty,
                    recording_mode: RecordingMode::ArtifactMirrorOnly,
                }),
            },
            // Negative test: a plain JWT (no chatgpt_account_id, no responses scope)
            // hitting POST /responses must produce a scope error rather than routing
            // to the first-party backend. (Earlier this case was mis-named — the
            // fixture is `JwtWithoutResponsesScope`, NOT a ChatGPT-login token.)
            RouteSupportCase {
                name: "jwt_without_responses_scope_on_openai_responses_errs",
                method: "POST",
                path: "/responses",
                token: TokenFixture::JwtWithoutResponsesScope,
                scope_provider: Some(ProviderKind::OpenAiResponses),
                expected_scope_error_contains: Some("api.responses.write"),
                expected_route: Some(ResolvedRoute {
                    provider: ProviderKind::OpenAiResponses,
                    recording_mode: RecordingMode::StructuredText,
                }),
            },
            // GET /responses with the `api.responses.read` scope passes the
            // scope gate (v0.18.2 audit: GET + read-scope branch previously
            // unexercised).
            RouteSupportCase {
                name: "responses_api_read_scope_get_passes",
                method: "GET",
                path: "/responses",
                token: TokenFixture::ApiResponses,
                scope_provider: Some(ProviderKind::OpenAiResponses),
                expected_scope_error_contains: None,
                expected_route: Some(ResolvedRoute {
                    provider: ProviderKind::OpenAiResponses,
                    recording_mode: RecordingMode::StructuredText,
                }),
            },
            // GET /responses with a bare JWT (no responses scope) must error
            // mentioning `api.responses.read`, not `.write`.
            RouteSupportCase {
                name: "responses_missing_read_scope_on_get_errs",
                method: "GET",
                path: "/responses",
                token: TokenFixture::JwtWithoutResponsesScope,
                scope_provider: Some(ProviderKind::OpenAiResponses),
                expected_scope_error_contains: Some("api.responses.read"),
                expected_route: Some(ResolvedRoute {
                    provider: ProviderKind::OpenAiResponses,
                    recording_mode: RecordingMode::StructuredText,
                }),
            },
            // Negative test: /models with no token → Codex doesn't send this directly,
            // but rein should still make a routing decision (falls to ChatGptBackend
            // via ambiguous_chatgpt_backend_path or stays unrouted).
            RouteSupportCase {
                name: "models_no_token_unrouted",
                method: "GET",
                path: "/models",
                token: TokenFixture::None,
                scope_provider: None,
                expected_scope_error_contains: None,
                expected_route: None,
            },
        ]
    }

    fn headers_for_token_fixture(token: TokenFixture) -> hyper::HeaderMap {
        let mut headers = hyper::HeaderMap::new();
        let token = match token {
            TokenFixture::None => None,
            TokenFixture::ApiResponses => {
                Some(fake_jwt_with_scopes(&["api.responses.read", "api.responses.write"]))
            }
            TokenFixture::JwtWithoutResponsesScope => {
                Some(fake_jwt_with_scopes(&["openid", "profile", "offline_access"]))
            }
            TokenFixture::ChatGptLogin => {
                Some(fake_chatgpt_login_jwt(&["openid", "profile", "offline_access"]))
            }
            TokenFixture::ChatGptLoginWithResponsesScope => Some(fake_chatgpt_login_jwt(&[
                "openid",
                "profile",
                "offline_access",
                "api.responses.read",
                "api.responses.write",
            ])),
        };
        if let Some(token) = token {
            headers.insert(
                "authorization",
                hyper::header::HeaderValue::from_str(&format!("Bearer {token}")).unwrap(),
            );
        }
        headers
    }

    fn support_matrix_rows() -> Vec<SupportMatrixRow> {
        let route_rows = route_support_cases()
            .into_iter()
            .filter_map(|case| match case.name {
                "responses_api_scope_routes_public" => Some(SupportMatrixRow {
                    path_family: "/responses".to_string(),
                    trigger: "API-key scope: api.responses.read + api.responses.write".to_string(),
                    upstream: "openai_upstream (/v1/responses)".to_string(),
                    recording_mode: "StructuredText".to_string(),
                    coverage:
                        "route_resolution_support_matrix, proxy_forwards_api_responses_route_to_openai_upstream_with_v1_prefix"
                            .to_string(),
                }),
                "chatgpt_login_responses_routes_first_party" => Some(SupportMatrixRow {
                    path_family: "/responses".to_string(),
                    trigger: "ChatGPT login token".to_string(),
                    upstream: "codex_upstream (/responses)".to_string(),
                    recording_mode: "StructuredText".to_string(),
                    coverage:
                        "route_resolution_support_matrix, proxy_forwards_chatgpt_login_responses_route_to_codex_upstream"
                            .to_string(),
                }),
                "chatgpt_login_models_stays_artifact_mirror_only" => Some(SupportMatrixRow {
                    path_family: "/models".to_string(),
                    trigger: "ChatGPT login token".to_string(),
                    upstream: "codex_upstream (/models)".to_string(),
                    recording_mode: "ArtifactMirrorOnly".to_string(),
                    coverage:
                        "route_resolution_support_matrix, proxy_forwards_chatgpt_login_models_route_to_codex_upstream"
                            .to_string(),
                }),
                "chatgpt_login_compact_stays_artifact_mirror_only" => Some(SupportMatrixRow {
                    path_family: "/responses/compact".to_string(),
                    trigger: "ChatGPT login token".to_string(),
                    upstream: "codex_upstream (/responses/compact)".to_string(),
                    recording_mode: "ArtifactMirrorOnly".to_string(),
                    coverage:
                        "route_resolution_support_matrix, proxy_forwards_chatgpt_login_compact_route_to_codex_upstream"
                            .to_string(),
                }),
                "chatgpt_login_memories_stays_artifact_mirror_only" => Some(SupportMatrixRow {
                    path_family: "/memories/trace_summarize".to_string(),
                    trigger: "ChatGPT login token".to_string(),
                    upstream: "codex_upstream (/memories/trace_summarize)".to_string(),
                    recording_mode: "ArtifactMirrorOnly".to_string(),
                    coverage:
                        "route_resolution_support_matrix, proxy_forwards_chatgpt_login_memories_route_to_codex_upstream"
                            .to_string(),
                }),
                "chatgpt_backend_wham_usage" => Some(SupportMatrixRow {
                    path_family: "/wham/*".to_string(),
                    trigger: "ChatGPT login token".to_string(),
                    upstream: "chatgpt_upstream".to_string(),
                    recording_mode: "ArtifactMirrorOnly".to_string(),
                    coverage:
                        "route_resolution_support_matrix, proxy_forwards_chatgpt_helper_paths_to_chatgpt_upstream, proxy_forwards_chatgpt_tasks_list_route_to_chatgpt_upstream, proxy_forwards_chatgpt_task_details_route_to_chatgpt_upstream, proxy_forwards_chatgpt_sibling_turns_route_to_chatgpt_upstream, proxy_forwards_chatgpt_requirements_route_to_chatgpt_upstream"
                            .to_string(),
                }),
                "chatgpt_backend_connectors_directory" => Some(SupportMatrixRow {
                    path_family: "/connectors/*".to_string(),
                    trigger: "ChatGPT login token".to_string(),
                    upstream: "chatgpt_upstream".to_string(),
                    recording_mode: "ArtifactMirrorOnly".to_string(),
                    coverage:
                        "route_resolution_support_matrix, proxy_forwards_chatgpt_connector_directory_route_to_chatgpt_upstream, proxy_forwards_chatgpt_workspace_connector_route_to_chatgpt_upstream"
                            .to_string(),
                }),
                _ => None,
            })
            .collect::<Vec<_>>();

        let mut rows = route_rows;
        rows.extend([
            SupportMatrixRow {
                path_family: "/v1/agent/register".to_string(),
                trigger: "ChatGPT login token".to_string(),
                upstream: "chatgpt_upstream".to_string(),
                recording_mode: "ArtifactMirrorOnly".to_string(),
                coverage:
                    "route_resolution_support_matrix, proxy_forwards_chatgpt_agent_register_route_to_chatgpt_upstream"
                        .to_string(),
            },
            SupportMatrixRow {
                path_family: "/authenticate_app_v2".to_string(),
                trigger: "ChatGPT login token".to_string(),
                upstream: "chatgpt_upstream".to_string(),
                recording_mode: "ArtifactMirrorOnly".to_string(),
                coverage:
                    "route_resolution_support_matrix, proxy_forwards_chatgpt_authenticate_app_route_to_chatgpt_upstream"
                        .to_string(),
            },
            SupportMatrixRow {
                path_family: "/codex/safety/arc".to_string(),
                trigger: "ChatGPT login token".to_string(),
                upstream: "chatgpt_upstream".to_string(),
                recording_mode: "ArtifactMirrorOnly".to_string(),
                coverage:
                    "route_resolution_support_matrix, proxy_forwards_chatgpt_arc_monitor_route_to_chatgpt_upstream"
                        .to_string(),
            },
            SupportMatrixRow {
                path_family: "GET /responses WebSocket upgrade".to_string(),
                trigger: "ChatGPT login token".to_string(),
                upstream: "codex_upstream (/responses)".to_string(),
                recording_mode: "StructuredText + proxy_first_party_ws artifact".to_string(),
                coverage:
                    "proxy_forwards_chatgpt_login_websocket_upgrade_to_codex_upstream, proxy_returns_426_when_codex_websocket_upstream_is_unavailable, proxy_stores_redacted_first_party_websocket_artifact_for_chatgpt_login_responses, websocket_request_mirror_extracts_response_create_query, websocket_mirror_reassembles_fragmented_text_frames, websocket_mirror_decodes_compressed_text_frames"
                        .to_string(),
            },
        ]);
        rows
    }

    fn normalize_doc_cell(cell: &str) -> String {
        cell.replace('`', "")
            .replace("已覆盖：", "")
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn render_support_matrix_rows(rows: &[SupportMatrixRow]) -> Vec<Vec<String>> {
        rows.iter()
            .map(|row| {
                vec![
                    normalize_doc_cell(&row.path_family),
                    normalize_doc_cell(&row.trigger),
                    normalize_doc_cell(&row.upstream),
                    normalize_doc_cell(&row.recording_mode),
                    normalize_doc_cell(&row.coverage),
                ]
            })
            .collect()
    }

    fn extract_support_matrix_rows(doc: &str) -> Vec<Vec<String>> {
        let section = doc
            .split("## 2. 当前支持矩阵")
            .nth(1)
            .and_then(|rest| rest.split("\n## ").next())
            .expect("support matrix section should exist");
        section
            .lines()
            .filter(|line| line.trim_start().starts_with('|'))
            .filter(|line| !line.contains("---"))
            .skip(1)
            .map(|line| {
                line.trim()
                    .trim_matches('|')
                    .split('|')
                    .map(normalize_doc_cell)
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    #[test]
    fn responses_scope_support_matrix() {
        for case in route_support_cases() {
            let Some(provider) = case.scope_provider else {
                continue;
            };
            let headers = headers_for_token_fixture(case.token);
            let method = hyper::Method::from_bytes(case.method.as_bytes()).unwrap();
            let actual = responses_scope_error(provider, &method, &headers);
            match case.expected_scope_error_contains {
                Some(fragment) => {
                    let msg = actual.unwrap_or_else(|| {
                        panic!("{} should fail responses scope checks", case.name)
                    });
                    assert!(
                        msg.contains(fragment),
                        "{} returned unexpected scope message: {msg}",
                        case.name
                    );
                }
                None => assert!(
                    actual.is_none(),
                    "{} unexpectedly failed responses scope checks: {:?}",
                    case.name,
                    actual
                ),
            }
        }
    }

    #[test]
    fn route_resolution_support_matrix() {
        for case in route_support_cases() {
            let headers = headers_for_token_fixture(case.token);
            assert_eq!(
                resolve_route(case.path, &headers),
                case.expected_route,
                "{} resolved unexpectedly for path {}",
                case.name,
                case.path
            );
        }
    }

    #[test]
    fn support_matrix_doc_row_parity() {
        let doc =
            include_str!("../../../../docs/reference/codex-subscription-proxy-support-matrix.md");
        assert_eq!(
            render_support_matrix_rows(&support_matrix_rows()),
            extract_support_matrix_rows(doc),
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn proxy_forwards_api_responses_route_to_openai_upstream_with_v1_prefix() {
        let (openai_upstream, openai_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");
        let (codex_upstream, _codex_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");
        let (chatgpt_upstream, _chatgpt_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");

        let mut config = ReinConfig::default();
        config.proxy.openai_upstream = openai_upstream;
        config.proxy.codex_upstream = codex_upstream;
        config.proxy.chatgpt_upstream = chatgpt_upstream;

        let (proxy_base, proxy_task) =
            spawn_one_shot_proxy(config, Some("secret".to_string())).await;
        let token = fake_jwt_with_scopes(&["api.responses.read", "api.responses.write"]);
        let response = reqwest::Client::new()
            .post(format!("{proxy_base}/responses"))
            .header("x-rein-token", "secret")
            .header("authorization", format!("Bearer {token}"))
            .header("content-type", "application/json")
            .body(r#"{"input":"hello"}"#)
            .send()
            .await
            .unwrap();
        assert!(response.status().is_success());

        let captured = openai_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        let expected_auth = format!("Bearer {token}");
        assert_eq!(captured.method, "POST");
        assert_eq!(captured.path, "/v1/responses");
        assert_eq!(
            captured.headers.get("authorization").map(String::as_str),
            Some(expected_auth.as_str())
        );
        assert!(!captured.headers.contains_key("x-rein-token"));
        assert_eq!(captured.body, br#"{"input":"hello"}"#);
        proxy_task.abort();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn proxy_forwards_claude_messages_body_unmodified() {
        let (anthropic_upstream, anthropic_rx) =
            spawn_capture_http_server("200 OK", r#"{"content":[{"type":"text","text":"ok"}]}"#);
        let (openai_upstream, _openai_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");
        let (codex_upstream, _codex_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");
        let (chatgpt_upstream, _chatgpt_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");

        let mut config = ReinConfig::default();
        config.proxy.extract_enabled = false;
        config.proxy.anthropic_upstream = anthropic_upstream;
        config.proxy.openai_upstream = openai_upstream;
        config.proxy.codex_upstream = codex_upstream;
        config.proxy.chatgpt_upstream = chatgpt_upstream;

        let (proxy_base, proxy_task) =
            spawn_one_shot_proxy(config, Some("secret".to_string())).await;
        let body = r#"{"model":"claude-sonnet","system":"system seed","messages":[{"role":"user","content":"hello"}]}"#;
        let response = reqwest::Client::new()
            .post(format!("{proxy_base}/v1/messages"))
            .header("x-rein-token", "secret")
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .body(body)
            .send()
            .await
            .unwrap();
        assert!(response.status().is_success());

        let captured = anthropic_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        assert_eq!(captured.method, "POST");
        assert_eq!(captured.path, "/v1/messages");
        assert_eq!(captured.body, body.as_bytes());
        assert!(!captured.headers.contains_key("x-rein-token"));
        proxy_task.abort();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn proxy_forwards_chatgpt_login_responses_route_to_codex_upstream() {
        let (openai_upstream, _openai_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");
        let (codex_upstream, codex_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");
        let (chatgpt_upstream, _chatgpt_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");

        let mut config = ReinConfig::default();
        config.proxy.openai_upstream = openai_upstream;
        config.proxy.codex_upstream = codex_upstream;
        config.proxy.chatgpt_upstream = chatgpt_upstream;

        let (proxy_base, proxy_task) =
            spawn_one_shot_proxy(config, Some("secret".to_string())).await;
        let token = fake_chatgpt_login_jwt(&["openid", "profile", "offline_access"]);
        let response = reqwest::Client::new()
            .post(format!("{proxy_base}/responses"))
            .header("x-rein-token", "secret")
            .header("authorization", format!("Bearer {token}"))
            .header("chatgpt-account-id", "acct_test")
            .header("content-type", "application/json")
            .body(r#"{"input":"hello"}"#)
            .send()
            .await
            .unwrap();
        assert!(response.status().is_success());

        let captured = codex_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        let expected_auth = format!("Bearer {token}");
        assert_eq!(captured.method, "POST");
        assert_eq!(captured.path, "/responses");
        assert_eq!(
            captured.headers.get("authorization").map(String::as_str),
            Some(expected_auth.as_str())
        );
        assert_eq!(
            captured
                .headers
                .get("chatgpt-account-id")
                .map(String::as_str),
            Some("acct_test")
        );
        proxy_task.abort();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn proxy_forwards_chatgpt_helper_paths_to_chatgpt_upstream() {
        let (openai_upstream, _openai_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");
        let (codex_upstream, _codex_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");
        let (chatgpt_upstream, chatgpt_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");

        let mut config = ReinConfig::default();
        config.proxy.openai_upstream = openai_upstream;
        config.proxy.codex_upstream = codex_upstream;
        config.proxy.chatgpt_upstream = chatgpt_upstream;

        let (proxy_base, proxy_task) =
            spawn_one_shot_proxy(config, Some("secret".to_string())).await;
        let token = fake_chatgpt_login_jwt(&["openid", "profile", "offline_access"]);
        let response = reqwest::Client::new()
            .get(format!("{proxy_base}/wham/usage"))
            .header("x-rein-token", "secret")
            .header("authorization", format!("Bearer {token}"))
            .header("chatgpt-account-id", "acct_test")
            .send()
            .await
            .unwrap();
        assert!(response.status().is_success());

        let captured = chatgpt_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        assert_eq!(captured.method, "GET");
        assert_eq!(captured.path, "/wham/usage");
        assert_eq!(
            captured
                .headers
                .get("chatgpt-account-id")
                .map(String::as_str),
            Some("acct_test")
        );
        proxy_task.abort();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn proxy_forwards_chatgpt_login_models_route_to_codex_upstream() {
        let (openai_upstream, _openai_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");
        let (codex_upstream, codex_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");
        let (chatgpt_upstream, _chatgpt_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");

        let mut config = ReinConfig::default();
        config.proxy.openai_upstream = openai_upstream;
        config.proxy.codex_upstream = codex_upstream;
        config.proxy.chatgpt_upstream = chatgpt_upstream;

        let (proxy_base, proxy_task) =
            spawn_one_shot_proxy(config, Some("secret".to_string())).await;
        let token = fake_chatgpt_login_jwt(&["openid", "profile", "offline_access"]);
        let response = reqwest::Client::new()
            .get(format!("{proxy_base}/models?client_version=0.120.0"))
            .header("x-rein-token", "secret")
            .header("authorization", format!("Bearer {token}"))
            .header("chatgpt-account-id", "acct_test")
            .send()
            .await
            .unwrap();
        assert!(response.status().is_success());

        let captured = codex_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        assert_eq!(captured.method, "GET");
        assert_eq!(captured.path, "/models?client_version=0.120.0");
        assert_eq!(
            captured
                .headers
                .get("chatgpt-account-id")
                .map(String::as_str),
            Some("acct_test")
        );
        proxy_task.abort();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn proxy_forwards_chatgpt_login_compact_route_to_codex_upstream() {
        let (openai_upstream, _openai_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");
        let (codex_upstream, codex_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");
        let (chatgpt_upstream, _chatgpt_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");

        let mut config = ReinConfig::default();
        config.proxy.openai_upstream = openai_upstream;
        config.proxy.codex_upstream = codex_upstream;
        config.proxy.chatgpt_upstream = chatgpt_upstream;

        let (proxy_base, proxy_task) =
            spawn_one_shot_proxy(config, Some("secret".to_string())).await;
        let token = fake_chatgpt_login_jwt(&["openid", "profile", "offline_access"]);
        let response = reqwest::Client::new()
            .post(format!("{proxy_base}/responses/compact"))
            .header("x-rein-token", "secret")
            .header("authorization", format!("Bearer {token}"))
            .header("chatgpt-account-id", "acct_test")
            .header("content-type", "application/json")
            .body(r#"{"items":[]}"#)
            .send()
            .await
            .unwrap();
        assert!(response.status().is_success());

        let captured = codex_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        assert_eq!(captured.method, "POST");
        assert_eq!(captured.path, "/responses/compact");
        assert_eq!(captured.body, br#"{"items":[]}"#);
        proxy_task.abort();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn proxy_forwards_chatgpt_login_memories_route_to_codex_upstream() {
        let (openai_upstream, _openai_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");
        let (codex_upstream, codex_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");
        let (chatgpt_upstream, _chatgpt_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");

        let mut config = ReinConfig::default();
        config.proxy.openai_upstream = openai_upstream;
        config.proxy.codex_upstream = codex_upstream;
        config.proxy.chatgpt_upstream = chatgpt_upstream;

        let (proxy_base, proxy_task) =
            spawn_one_shot_proxy(config, Some("secret".to_string())).await;
        let token = fake_chatgpt_login_jwt(&["openid", "profile", "offline_access"]);
        let response = reqwest::Client::new()
            .post(format!("{proxy_base}/memories/trace_summarize"))
            .header("x-rein-token", "secret")
            .header("authorization", format!("Bearer {token}"))
            .header("chatgpt-account-id", "acct_test")
            .header("content-type", "application/json")
            .body(r#"{"raw_memories":[]}"#)
            .send()
            .await
            .unwrap();
        assert!(response.status().is_success());

        let captured = codex_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        assert_eq!(captured.method, "POST");
        assert_eq!(captured.path, "/memories/trace_summarize");
        assert_eq!(captured.body, br#"{"raw_memories":[]}"#);
        proxy_task.abort();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn proxy_forwards_chatgpt_connector_directory_route_to_chatgpt_upstream() {
        let (openai_upstream, _openai_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");
        let (codex_upstream, _codex_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");
        let (chatgpt_upstream, chatgpt_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");

        let mut config = ReinConfig::default();
        config.proxy.openai_upstream = openai_upstream;
        config.proxy.codex_upstream = codex_upstream;
        config.proxy.chatgpt_upstream = chatgpt_upstream;

        let (proxy_base, proxy_task) =
            spawn_one_shot_proxy(config, Some("secret".to_string())).await;
        let token = fake_chatgpt_login_jwt(&["openid", "profile", "offline_access"]);
        let response = reqwest::Client::new()
            .get(format!(
                "{proxy_base}/connectors/directory/list?external_logos=true"
            ))
            .header("x-rein-token", "secret")
            .header("authorization", format!("Bearer {token}"))
            .header("chatgpt-account-id", "acct_test")
            .send()
            .await
            .unwrap();
        assert!(response.status().is_success());

        let captured = chatgpt_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        assert_eq!(captured.method, "GET");
        assert_eq!(
            captured.path,
            "/connectors/directory/list?external_logos=true"
        );
        proxy_task.abort();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn proxy_forwards_chatgpt_workspace_connector_route_to_chatgpt_upstream() {
        let (openai_upstream, _openai_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");
        let (codex_upstream, _codex_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");
        let (chatgpt_upstream, chatgpt_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");

        let mut config = ReinConfig::default();
        config.proxy.openai_upstream = openai_upstream;
        config.proxy.codex_upstream = codex_upstream;
        config.proxy.chatgpt_upstream = chatgpt_upstream;

        let (proxy_base, proxy_task) =
            spawn_one_shot_proxy(config, Some("secret".to_string())).await;
        let token = fake_chatgpt_login_jwt(&["openid", "profile", "offline_access"]);
        let response = reqwest::Client::new()
            .get(format!(
                "{proxy_base}/connectors/directory/list_workspace?external_logos=true"
            ))
            .header("x-rein-token", "secret")
            .header("authorization", format!("Bearer {token}"))
            .header("chatgpt-account-id", "acct_test")
            .send()
            .await
            .unwrap();
        assert!(response.status().is_success());

        let captured = chatgpt_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        assert_eq!(captured.method, "GET");
        assert_eq!(
            captured.path,
            "/connectors/directory/list_workspace?external_logos=true"
        );
        proxy_task.abort();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn proxy_forwards_chatgpt_tasks_list_route_to_chatgpt_upstream() {
        let (openai_upstream, _openai_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");
        let (codex_upstream, _codex_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");
        let (chatgpt_upstream, chatgpt_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");

        let mut config = ReinConfig::default();
        config.proxy.openai_upstream = openai_upstream;
        config.proxy.codex_upstream = codex_upstream;
        config.proxy.chatgpt_upstream = chatgpt_upstream;

        let (proxy_base, proxy_task) =
            spawn_one_shot_proxy(config, Some("secret".to_string())).await;
        let token = fake_chatgpt_login_jwt(&["openid", "profile", "offline_access"]);
        let response = reqwest::Client::new()
            .get(format!("{proxy_base}/wham/tasks/list?limit=10"))
            .header("x-rein-token", "secret")
            .header("authorization", format!("Bearer {token}"))
            .header("chatgpt-account-id", "acct_test")
            .send()
            .await
            .unwrap();
        assert!(response.status().is_success());

        let captured = chatgpt_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        assert_eq!(captured.method, "GET");
        assert_eq!(captured.path, "/wham/tasks/list?limit=10");
        proxy_task.abort();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn proxy_forwards_chatgpt_authenticate_app_route_to_chatgpt_upstream() {
        let (openai_upstream, _openai_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");
        let (codex_upstream, _codex_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");
        let (chatgpt_upstream, chatgpt_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");

        let mut config = ReinConfig::default();
        config.proxy.openai_upstream = openai_upstream;
        config.proxy.codex_upstream = codex_upstream;
        config.proxy.chatgpt_upstream = chatgpt_upstream;

        let (proxy_base, proxy_task) =
            spawn_one_shot_proxy(config, Some("secret".to_string())).await;
        let token = fake_chatgpt_login_jwt(&["openid", "profile", "offline_access"]);
        let response = reqwest::Client::new()
            .post(format!("{proxy_base}/authenticate_app_v2"))
            .header("x-rein-token", "secret")
            .header("authorization", format!("Bearer {token}"))
            .header("chatgpt-account-id", "acct_test")
            .header("content-type", "application/json")
            .body(r#"{"challenge":"abc"}"#)
            .send()
            .await
            .unwrap();
        assert!(response.status().is_success());

        let captured = chatgpt_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        assert_eq!(captured.method, "POST");
        assert_eq!(captured.path, "/authenticate_app_v2");
        assert_eq!(captured.body, br#"{"challenge":"abc"}"#);
        proxy_task.abort();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn proxy_forwards_chatgpt_agent_register_route_to_chatgpt_upstream() {
        let (openai_upstream, _openai_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");
        let (codex_upstream, _codex_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");
        let (chatgpt_upstream, chatgpt_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");

        let mut config = ReinConfig::default();
        config.proxy.openai_upstream = openai_upstream;
        config.proxy.codex_upstream = codex_upstream;
        config.proxy.chatgpt_upstream = chatgpt_upstream;

        let (proxy_base, proxy_task) =
            spawn_one_shot_proxy(config, Some("secret".to_string())).await;
        let token = fake_chatgpt_login_jwt(&["openid", "profile", "offline_access"]);
        let response = reqwest::Client::new()
            .post(format!("{proxy_base}/v1/agent/register"))
            .header("x-rein-token", "secret")
            .header("authorization", format!("Bearer {token}"))
            .header("chatgpt-account-id", "acct_test")
            .header("content-type", "application/json")
            .body(r#"{"public_key":"ssh-ed25519 AAA"}"#)
            .send()
            .await
            .unwrap();
        assert!(response.status().is_success());

        let captured = chatgpt_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        assert_eq!(captured.method, "POST");
        assert_eq!(captured.path, "/v1/agent/register");
        assert_eq!(captured.body, br#"{"public_key":"ssh-ed25519 AAA"}"#);
        proxy_task.abort();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn proxy_forwards_chatgpt_arc_monitor_route_to_chatgpt_upstream() {
        let (openai_upstream, _openai_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");
        let (codex_upstream, _codex_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");
        let (chatgpt_upstream, chatgpt_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");

        let mut config = ReinConfig::default();
        config.proxy.openai_upstream = openai_upstream;
        config.proxy.codex_upstream = codex_upstream;
        config.proxy.chatgpt_upstream = chatgpt_upstream;

        let (proxy_base, proxy_task) =
            spawn_one_shot_proxy(config, Some("secret".to_string())).await;
        let token = fake_chatgpt_login_jwt(&["openid", "profile", "offline_access"]);
        let response = reqwest::Client::new()
            .post(format!("{proxy_base}/codex/safety/arc"))
            .header("x-rein-token", "secret")
            .header("authorization", format!("Bearer {token}"))
            .header("chatgpt-account-id", "acct_test")
            .header("content-type", "application/json")
            .body(r#"{"action":"review"}"#)
            .send()
            .await
            .unwrap();
        assert!(response.status().is_success());

        let captured = chatgpt_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        assert_eq!(captured.method, "POST");
        assert_eq!(captured.path, "/codex/safety/arc");
        assert_eq!(captured.body, br#"{"action":"review"}"#);
        proxy_task.abort();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn proxy_forwards_chatgpt_task_details_route_to_chatgpt_upstream() {
        let (openai_upstream, _openai_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");
        let (codex_upstream, _codex_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");
        let (chatgpt_upstream, chatgpt_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");

        let mut config = ReinConfig::default();
        config.proxy.openai_upstream = openai_upstream;
        config.proxy.codex_upstream = codex_upstream;
        config.proxy.chatgpt_upstream = chatgpt_upstream;

        let (proxy_base, proxy_task) =
            spawn_one_shot_proxy(config, Some("secret".to_string())).await;
        let token = fake_chatgpt_login_jwt(&["openid", "profile", "offline_access"]);
        let response = reqwest::Client::new()
            .get(format!("{proxy_base}/wham/tasks/task_123"))
            .header("x-rein-token", "secret")
            .header("authorization", format!("Bearer {token}"))
            .header("chatgpt-account-id", "acct_test")
            .send()
            .await
            .unwrap();
        assert!(response.status().is_success());

        let captured = chatgpt_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        assert_eq!(captured.method, "GET");
        assert_eq!(captured.path, "/wham/tasks/task_123");
        proxy_task.abort();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn proxy_forwards_chatgpt_sibling_turns_route_to_chatgpt_upstream() {
        let (openai_upstream, _openai_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");
        let (codex_upstream, _codex_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");
        let (chatgpt_upstream, chatgpt_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");

        let mut config = ReinConfig::default();
        config.proxy.openai_upstream = openai_upstream;
        config.proxy.codex_upstream = codex_upstream;
        config.proxy.chatgpt_upstream = chatgpt_upstream;

        let (proxy_base, proxy_task) =
            spawn_one_shot_proxy(config, Some("secret".to_string())).await;
        let token = fake_chatgpt_login_jwt(&["openid", "profile", "offline_access"]);
        let response = reqwest::Client::new()
            .get(format!(
                "{proxy_base}/wham/tasks/task_123/turns/turn_456/sibling_turns"
            ))
            .header("x-rein-token", "secret")
            .header("authorization", format!("Bearer {token}"))
            .header("chatgpt-account-id", "acct_test")
            .send()
            .await
            .unwrap();
        assert!(response.status().is_success());

        let captured = chatgpt_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        assert_eq!(captured.method, "GET");
        assert_eq!(
            captured.path,
            "/wham/tasks/task_123/turns/turn_456/sibling_turns"
        );
        proxy_task.abort();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn proxy_forwards_chatgpt_requirements_route_to_chatgpt_upstream() {
        let (openai_upstream, _openai_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");
        let (codex_upstream, _codex_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");
        let (chatgpt_upstream, chatgpt_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");

        let mut config = ReinConfig::default();
        config.proxy.openai_upstream = openai_upstream;
        config.proxy.codex_upstream = codex_upstream;
        config.proxy.chatgpt_upstream = chatgpt_upstream;

        let (proxy_base, proxy_task) =
            spawn_one_shot_proxy(config, Some("secret".to_string())).await;
        let token = fake_chatgpt_login_jwt(&["openid", "profile", "offline_access"]);
        let response = reqwest::Client::new()
            .get(format!("{proxy_base}/wham/config/requirements"))
            .header("x-rein-token", "secret")
            .header("authorization", format!("Bearer {token}"))
            .header("chatgpt-account-id", "acct_test")
            .send()
            .await
            .unwrap();
        assert!(response.status().is_success());

        let captured = chatgpt_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        assert_eq!(captured.method, "GET");
        assert_eq!(captured.path, "/wham/config/requirements");
        proxy_task.abort();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn proxy_forwards_chatgpt_login_websocket_upgrade_to_codex_upstream() {
        let (openai_upstream, _openai_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");
        let (codex_upstream, codex_rx) = spawn_capture_websocket_server();
        let (chatgpt_upstream, _chatgpt_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");
        let expected_origin = codex_upstream.clone();

        let mut config = ReinConfig::default();
        config.proxy.openai_upstream = openai_upstream;
        config.proxy.codex_upstream = codex_upstream;
        config.proxy.chatgpt_upstream = chatgpt_upstream;

        let (proxy_base, proxy_task) =
            spawn_one_shot_proxy(config, Some("secret".to_string())).await;
        let addr = proxy_base.trim_start_matches("http://");
        let token = fake_chatgpt_login_jwt(&["openid", "profile", "offline_access"]);

        let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
        let request = format!(
            "GET /responses HTTP/1.1\r\n\
Host: {addr}\r\n\
Connection: Upgrade\r\n\
Upgrade: websocket\r\n\
Sec-WebSocket-Version: 13\r\n\
Sec-WebSocket-Key: dGVzdC1rZXk=\r\n\
Sec-WebSocket-Extensions: permessage-deflate; client_max_window_bits\r\n\
x-rein-token: secret\r\n\
Authorization: Bearer {token}\r\n\
ChatGPT-Account-ID: acct_test\r\n\
Origin: http://127.0.0.1:8788\r\n\
OpenAI-Beta: responses_websockets=2026-02-06\r\n\
X-Codex-Turn-State: turn_state_123\r\n\
X-Codex-Turn-Metadata: turn_meta_456\r\n\
x-client-request-id: ws_upgrade_123\r\n\
\r\n"
        );
        stream.write_all(request.as_bytes()).await.unwrap();

        let mut response = Vec::new();
        let mut tmp = [0u8; 1024];
        loop {
            let n = stream.read(&mut tmp).await.unwrap();
            if n == 0 {
                break;
            }
            response.extend_from_slice(&tmp[..n]);
            if find_header_end_for_test(&response).is_some() {
                break;
            }
        }
        let response_text = String::from_utf8_lossy(&response);
        assert!(response_text.starts_with("HTTP/1.1 101"));
        assert!(!response_text.contains("Sec-WebSocket-Extensions"));

        let captured = codex_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        let expected_auth = format!("Bearer {token}");
        assert_eq!(captured.method, "GET");
        assert_eq!(captured.path, "/responses");
        assert_eq!(
            captured.headers.get("authorization").map(String::as_str),
            Some(expected_auth.as_str())
        );
        assert_eq!(
            captured
                .headers
                .get("chatgpt-account-id")
                .map(String::as_str),
            Some("acct_test")
        );
        assert_eq!(
            captured.headers.get("openai-beta").map(String::as_str),
            Some("responses_websockets=2026-02-06")
        );
        assert_eq!(
            captured
                .headers
                .get("x-codex-turn-state")
                .map(String::as_str),
            Some("turn_state_123")
        );
        assert_eq!(
            captured
                .headers
                .get("x-codex-turn-metadata")
                .map(String::as_str),
            Some("turn_meta_456")
        );
        assert_eq!(
            captured
                .headers
                .get("x-client-request-id")
                .map(String::as_str),
            Some("ws_upgrade_123")
        );
        assert_eq!(
            captured.headers.get("origin").map(String::as_str),
            Some(expected_origin.as_str())
        );
        assert_eq!(
            captured.headers.get("originator").map(String::as_str),
            Some(default_codex_originator())
        );
        assert!(
            captured
                .headers
                .get("user-agent")
                .is_some_and(|value| value.contains("rein-proxy"))
        );
        assert!(!captured.headers.contains_key("sec-websocket-extensions"));
        proxy_task.abort();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn proxy_does_not_upgrade_artifact_only_first_party_paths() {
        let (openai_upstream, _openai_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");
        let (codex_upstream, codex_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");
        let (chatgpt_upstream, _chatgpt_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");

        let mut config = ReinConfig::default();
        config.proxy.openai_upstream = openai_upstream;
        config.proxy.codex_upstream = codex_upstream;
        config.proxy.chatgpt_upstream = chatgpt_upstream;

        let (proxy_base, proxy_task) =
            spawn_one_shot_proxy(config, Some("secret".to_string())).await;
        let addr = proxy_base.trim_start_matches("http://");
        let token = fake_chatgpt_login_jwt(&["openid", "profile", "offline_access"]);

        let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
        let request = format!(
            "GET /models HTTP/1.1\r\n\
Host: {addr}\r\n\
Connection: Upgrade\r\n\
Upgrade: websocket\r\n\
Sec-WebSocket-Version: 13\r\n\
Sec-WebSocket-Key: dGVzdC1rZXk=\r\n\
x-rein-token: secret\r\n\
Authorization: Bearer {token}\r\n\
ChatGPT-Account-ID: acct_test\r\n\
\r\n"
        );
        stream.write_all(request.as_bytes()).await.unwrap();

        let mut response = Vec::new();
        let mut tmp = [0u8; 1024];
        loop {
            let n = stream.read(&mut tmp).await.unwrap();
            if n == 0 {
                break;
            }
            response.extend_from_slice(&tmp[..n]);
            if find_header_end_for_test(&response).is_some() {
                break;
            }
        }
        let response_text = String::from_utf8_lossy(&response);
        assert!(response_text.starts_with("HTTP/1.1 200"));
        assert!(!response_text.starts_with("HTTP/1.1 101"));

        let captured = codex_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        assert_eq!(captured.method, "GET");
        assert_eq!(captured.path, "/models");
        proxy_task.abort();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn proxy_returns_426_when_codex_websocket_upstream_is_unavailable() {
        let (openai_upstream, _openai_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");
        let (chatgpt_upstream, _chatgpt_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");

        let mut config = ReinConfig::default();
        config.proxy.openai_upstream = openai_upstream;
        config.proxy.codex_upstream = unused_local_base_url("http");
        config.proxy.chatgpt_upstream = chatgpt_upstream;

        let (proxy_base, proxy_task) =
            spawn_one_shot_proxy(config, Some("secret".to_string())).await;
        let addr = proxy_base.trim_start_matches("http://");
        let token = fake_chatgpt_login_jwt(&["openid", "profile", "offline_access"]);

        let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
        let request = format!(
            "GET /responses HTTP/1.1\r\n\
Host: {addr}\r\n\
Connection: Upgrade\r\n\
Upgrade: websocket\r\n\
Sec-WebSocket-Version: 13\r\n\
Sec-WebSocket-Key: dGVzdC1rZXk=\r\n\
x-rein-token: secret\r\n\
Authorization: Bearer {token}\r\n\
ChatGPT-Account-ID: acct_test\r\n\
\r\n"
        );
        stream.write_all(request.as_bytes()).await.unwrap();

        let mut response = Vec::new();
        let mut tmp = [0u8; 1024];
        loop {
            let n = stream.read(&mut tmp).await.unwrap();
            if n == 0 {
                break;
            }
            response.extend_from_slice(&tmp[..n]);
            if find_header_end_for_test(&response).is_some() {
                break;
            }
        }
        let response_text = String::from_utf8_lossy(&response);
        assert!(response_text.starts_with("HTTP/1.1 426"));
        assert!(response_text.contains("retry over HTTP"));

        proxy_task.abort();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn proxy_stores_redacted_first_party_artifact_for_chatgpt_login_responses() {
        let tempdir = tempfile::tempdir().unwrap();
        let db_path = tempdir.path().join("proxy.db");

        let (openai_upstream, _openai_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");
        let (codex_upstream, _codex_rx) = spawn_capture_http_server(
            "200 OK",
            "{\"output\":[{\"type\":\"output_text\",\"text\":\"hello back\"}]}",
        );
        let (chatgpt_upstream, _chatgpt_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");

        let mut config = ReinConfig::default();
        config.database.path = db_path.display().to_string();
        config.proxy.extract_enabled = false;
        config.proxy.openai_upstream = openai_upstream;
        config.proxy.codex_upstream = codex_upstream;
        config.proxy.chatgpt_upstream = chatgpt_upstream;
        let _ = config.open_store().unwrap();

        let (proxy_base, proxy_task) =
            spawn_one_shot_proxy(config, Some("secret".to_string())).await;
        let token = fake_chatgpt_login_jwt(&["openid", "profile", "offline_access"]);
        let response = reqwest::Client::new()
            .post(format!("{proxy_base}/responses"))
            .header("x-rein-token", "secret")
            .header("authorization", format!("Bearer {token}"))
            .header("chatgpt-account-id", "acct_test")
            .header("x-client-request-id", "thread_123")
            .header("content-type", "application/json")
            .body(r#"{"input":"hello"}"#)
            .send()
            .await
            .unwrap();
        assert!(response.status().is_success());

        tokio::time::sleep(Duration::from_millis(250)).await;
        wait_for_artifact_count(&db_path, "proxy_first_party");

        let conn = Connection::open(&db_path).unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT artifact_kind, session_id, source_agent, source_label, transcript_text \
                 FROM session_artifacts ORDER BY created_at DESC LIMIT 1",
            )
            .unwrap();
        let row = stmt
            .query_row([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, Option<String>>(1)?,
                    row.get::<_, Option<String>>(2)?,
                    row.get::<_, Option<String>>(3)?,
                    row.get::<_, String>(4)?,
                ))
            })
            .unwrap();

        assert_eq!(row.0, "proxy_first_party");
        assert_eq!(row.1.as_deref(), Some("thread_123"));
        assert_eq!(row.2.as_deref(), Some("proxy"));
        assert_eq!(row.3.as_deref(), Some("proxy:codex-first-party"));
        assert!(row.4.contains("path: /responses"));
        assert!(row.4.contains("capture_mode: structured-text"));
        assert!(row.4.contains("request_body:"));
        assert!(row.4.contains("response_body:"));
        assert!(row.4.contains("authorization: <redacted>"));
        assert!(row.4.contains("chatgpt-account-id: <redacted>"));
        assert!(!row.4.contains("Bearer ey"));
        assert!(!row.4.contains("acct_test"));

        proxy_task.abort();
    }

    #[tokio::test(flavor = "current_thread")]
    #[serial_test::serial(global_state)]
    async fn proxy_http_artifact_is_visible_via_rest_api() {
        let tempdir = tempfile::tempdir().unwrap();
        let db_path = tempdir.path().join("proxy-rest.db");

        let (openai_upstream, _openai_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");
        let (codex_upstream, _codex_rx) = spawn_capture_http_server(
            "200 OK",
            "{\"output\":[{\"type\":\"output_text\",\"text\":\"hello back\"}]}",
        );
        let (chatgpt_upstream, _chatgpt_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");

        let mut config = ReinConfig::default();
        config.database.path = db_path.display().to_string();
        config.proxy.extract_enabled = false;
        config.proxy.openai_upstream = openai_upstream;
        config.proxy.codex_upstream = codex_upstream;
        config.proxy.chatgpt_upstream = chatgpt_upstream;
        let _ = config.open_store().unwrap();

        let (proxy_base, proxy_task) =
            spawn_one_shot_proxy(config.clone(), Some("secret".to_string())).await;
        let token = fake_chatgpt_login_jwt(&["openid", "profile", "offline_access"]);
        let response = reqwest::Client::new()
            .post(format!("{proxy_base}/responses"))
            .header("x-rein-token", "secret")
            .header("authorization", format!("Bearer {token}"))
            .header("chatgpt-account-id", "acct_test")
            .header("x-client-request-id", "thread_rest_123")
            .header("content-type", "application/json")
            .body(r#"{"input":"hello"}"#)
            .send()
            .await
            .unwrap();
        assert!(response.status().is_success());

        tokio::time::sleep(Duration::from_millis(250)).await;

        let list_req = hyper::Request::builder()
            .method("GET")
            .uri("/api/artifacts?limit=10&offset=0")
            .body(())
            .unwrap();
        let list_resp = crate::mcp::rest::handle_rest_request(&list_req, &config)
            .await
            .unwrap();
        let list_body = list_resp.into_body().collect().await.unwrap().to_bytes();
        let list_json: serde_json::Value = serde_json::from_slice(&list_body).unwrap();
        let artifacts = list_json["artifacts"].as_array().unwrap();
        let artifact = artifacts
            .iter()
            .find(|artifact| artifact["artifact_kind"] == "proxy_first_party")
            .expect("proxy_first_party artifact should be listed");
        let artifact_id = artifact["id"].as_str().unwrap();
        assert_eq!(
            artifact["source_label"].as_str(),
            Some("proxy:codex-first-party")
        );

        let detail_req = hyper::Request::builder()
            .method("GET")
            .uri(format!(
                "/api/artifacts/{}?include_transcript=true",
                artifact_id
            ))
            .body(())
            .unwrap();
        let detail_resp = crate::mcp::rest::handle_rest_request(&detail_req, &config)
            .await
            .unwrap();
        let detail_body = detail_resp.into_body().collect().await.unwrap().to_bytes();
        let detail_json: serde_json::Value = serde_json::from_slice(&detail_body).unwrap();
        assert_eq!(detail_json["id"].as_str(), Some(artifact_id));
        assert_eq!(detail_json["session_id"].as_str(), Some("thread_rest_123"));
        assert_eq!(
            detail_json["artifact_kind"].as_str(),
            Some("proxy_first_party")
        );
        assert_eq!(detail_json["transcript_available"].as_bool(), Some(true));
        let transcript = detail_json["transcript_text"].as_str().unwrap();
        assert!(transcript.contains("path: /responses"));
        assert!(transcript.contains("authorization: <redacted>"));
        assert!(transcript.contains("chatgpt-account-id: <redacted>"));
        assert!(!transcript.contains("Bearer ey"));
        assert!(!transcript.contains("acct_test"));

        proxy_task.abort();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn proxy_stores_redacted_first_party_websocket_artifact_for_chatgpt_login_responses() {
        let tempdir = tempfile::tempdir().unwrap();
        let db_path = tempdir.path().join("proxy-ws.db");

        let (openai_upstream, _openai_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");
        let (codex_upstream, _codex_rx) = spawn_capture_websocket_server_with_frames(vec![
            encode_ws_text_frame(r#"{"type":"response.created","response":{"id":"resp1"}}"#),
            encode_ws_text_frame(r#"{"type":"response.output_text.delta","delta":"hello ws"}"#),
            encode_ws_text_frame(r#"{"type":"response.completed","response":{"id":"resp1"}}"#),
            encode_ws_close_frame(),
        ]);
        let (chatgpt_upstream, _chatgpt_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");

        let mut config = ReinConfig::default();
        config.database.path = db_path.display().to_string();
        config.proxy.extract_enabled = false;
        config.proxy.openai_upstream = openai_upstream;
        config.proxy.codex_upstream = codex_upstream;
        config.proxy.chatgpt_upstream = chatgpt_upstream;
        let _ = config.open_store().unwrap();

        let (proxy_base, proxy_task) =
            spawn_one_shot_proxy(config, Some("secret".to_string())).await;
        let addr = proxy_base.trim_start_matches("http://");
        let token = fake_chatgpt_login_jwt(&["openid", "profile", "offline_access"]);

        let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
        let request = format!(
            "GET /responses HTTP/1.1\r\n\
Host: {addr}\r\n\
Connection: Upgrade\r\n\
Upgrade: websocket\r\n\
Sec-WebSocket-Version: 13\r\n\
Sec-WebSocket-Key: dGVzdC1rZXk=\r\n\
x-rein-token: secret\r\n\
Authorization: Bearer {token}\r\n\
ChatGPT-Account-ID: acct_test\r\n\
x-client-request-id: ws_thread_123\r\n\
\r\n"
        );
        stream.write_all(request.as_bytes()).await.unwrap();

        let mut response = Vec::new();
        let mut tmp = [0u8; 1024];
        loop {
            let n = stream.read(&mut tmp).await.unwrap();
            if n == 0 {
                break;
            }
            response.extend_from_slice(&tmp[..n]);
            if response.windows(2).any(|window| window == b"\x88\x00") {
                break;
            }
        }
        assert!(!response.is_empty());
        tokio::time::sleep(Duration::from_millis(250)).await;
        wait_for_artifact_count(&db_path, "proxy_first_party_ws");

        let conn = Connection::open(&db_path).unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT artifact_kind, session_id, source_label, transcript_text \
                 FROM session_artifacts ORDER BY created_at DESC LIMIT 1",
            )
            .unwrap();
        let row = stmt
            .query_row([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, Option<String>>(1)?,
                    row.get::<_, Option<String>>(2)?,
                    row.get::<_, String>(3)?,
                ))
            })
            .unwrap();

        assert_eq!(row.0, "proxy_first_party_ws");
        assert_eq!(row.1.as_deref(), Some("ws_thread_123"));
        assert_eq!(row.2.as_deref(), Some("proxy:codex-first-party-ws"));
        assert!(row.3.contains("path: /responses"));
        assert!(row.3.contains("capture_mode: structured-text"));
        assert!(row.3.contains("response.output_text.delta"));
        assert!(row.3.contains("hello ws"));
        assert!(row.3.contains("authorization: <redacted>"));
        assert!(row.3.contains("chatgpt-account-id: <redacted>"));
        assert!(!row.3.contains("Bearer ey"));
        assert!(!row.3.contains("acct_test"));

        proxy_task.abort();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn proxy_websocket_artifact_uses_parent_thread_id_and_records_client_events() {
        let tempdir = tempfile::tempdir().unwrap();
        let db_path = tempdir.path().join("proxy-ws-parent.db");

        let (openai_upstream, _openai_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");
        let (codex_upstream, _codex_rx) = spawn_capture_websocket_server_with_frames(vec![
            encode_ws_text_frame(r#"{"type":"response.created","response":{"id":"resp2"}}"#),
            encode_ws_text_frame(r#"{"type":"response.output_text.delta","delta":"hello parent"}"#),
            encode_ws_text_frame(r#"{"type":"response.completed","response":{"id":"resp2"}}"#),
            encode_ws_close_frame(),
        ]);
        let (chatgpt_upstream, _chatgpt_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");

        let mut config = ReinConfig::default();
        config.database.path = db_path.display().to_string();
        config.proxy.extract_enabled = false;
        config.proxy.openai_upstream = openai_upstream;
        config.proxy.codex_upstream = codex_upstream;
        config.proxy.chatgpt_upstream = chatgpt_upstream;
        let _ = config.open_store().unwrap();

        let (proxy_base, proxy_task) =
            spawn_one_shot_proxy(config, Some("secret".to_string())).await;
        let addr = proxy_base.trim_start_matches("http://");
        let token = fake_chatgpt_login_jwt(&["openid", "profile", "offline_access"]);

        let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
        let request = format!(
            "GET /responses HTTP/1.1\r\n\
Host: {addr}\r\n\
Connection: Upgrade\r\n\
Upgrade: websocket\r\n\
Sec-WebSocket-Version: 13\r\n\
Sec-WebSocket-Key: dGVzdC1rZXk=\r\n\
x-rein-token: secret\r\n\
Authorization: Bearer {token}\r\n\
ChatGPT-Account-ID: acct_test\r\n\
x-codex-parent-thread-id: parent_thread_123\r\n\
x-codex-window-id: win_123\r\n\
x-openai-subagent: worker_abc\r\n\
\r\n"
        );
        stream.write_all(request.as_bytes()).await.unwrap();

        let mut response = Vec::new();
        let mut tmp = [0u8; 1024];
        loop {
            let n = stream.read(&mut tmp).await.unwrap();
            if n == 0 {
                break;
            }
            response.extend_from_slice(&tmp[..n]);
            if find_header_end_for_test(&response).is_some() {
                break;
            }
        }
        let response_text = String::from_utf8_lossy(&response);
        assert!(response_text.starts_with("HTTP/1.1 101"));

        let client_event = r#"{"type":"response.create","input":[{"role":"user","content":"hello from parent"}]}"#;
        stream
            .write_all(&encode_masked_ws_text_frame(client_event))
            .await
            .unwrap();

        let mut ws_buf = Vec::new();
        loop {
            let n = stream.read(&mut tmp).await.unwrap();
            if n == 0 {
                break;
            }
            ws_buf.extend_from_slice(&tmp[..n]);
            if ws_buf.windows(2).any(|window| window == b"\x88\x00") {
                break;
            }
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
        wait_for_artifact_count(&db_path, "proxy_first_party_ws");

        let conn = Connection::open(&db_path).unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT artifact_kind, session_id, source_label, transcript_text \
                 FROM session_artifacts ORDER BY created_at DESC LIMIT 1",
            )
            .unwrap();
        let row = stmt
            .query_row([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, Option<String>>(1)?,
                    row.get::<_, Option<String>>(2)?,
                    row.get::<_, String>(3)?,
                ))
            })
            .unwrap();

        assert_eq!(row.0, "proxy_first_party_ws");
        assert_eq!(row.1.as_deref(), Some("parent_thread_123"));
        assert_eq!(row.2.as_deref(), Some("proxy:codex-first-party-ws"));
        assert!(row.3.contains("x-codex-parent-thread-id: parent_thread_123"));
        assert!(row.3.contains("x-codex-window-id: win_123"));
        assert!(row.3.contains("x-openai-subagent: worker_abc"));
        assert!(row.3.contains("client_ws_events:"));
        assert!(row.3.contains("response.create"));
        assert!(row.3.contains("hello from parent"));
        assert!(row.3.contains("upstream_ws_events:"));
        assert!(row.3.contains("response.created"));
        assert!(!row.3.contains("Bearer ey"));
        assert!(!row.3.contains("acct_test"));

        proxy_task.abort();
    }

    #[test]
    fn websocket_mirror_reassembles_fragmented_text_frames() {
        let mut mirror = WebSocketMirrorState::default();
        mirror.feed(
            &encode_ws_text_frame_with_flags(
                r#"{"type":"response.output_text.delta","delta":"hello "#,
                false,
                false,
            ),
            true,
        );
        mirror.feed(&encode_ws_continuation_frame(r#"ws"}"#, true), true);

        assert_eq!(mirror.assistant_text, "hello ws");
        assert_eq!(mirror.event_messages.len(), 1);
        assert!(mirror.event_messages[0].contains("response.output_text.delta"));
        assert!(mirror.event_messages[0].contains("hello ws"));
    }

    #[test]
    fn websocket_mirror_decodes_compressed_text_frames() {
        let mut mirror = WebSocketMirrorState::default();
        mirror.feed(
            &encode_ws_compressed_text_frame(
                r#"{"type":"response.output_text.delta","delta":"hello compressed"}"#,
            ),
            true,
        );

        assert_eq!(mirror.assistant_text, "hello compressed");
        assert_eq!(mirror.event_messages.len(), 1);
        assert!(mirror.event_messages[0].contains("response.output_text.delta"));
        assert!(mirror.event_messages[0].contains("hello compressed"));
    }

    #[test]
    fn websocket_request_mirror_extracts_response_create_query() {
        let mut mirror = WebSocketMirrorState::default();
        mirror.record_message(
            &Message::Text(
                r#"{"type":"response.create","input":[{"role":"user","content":"hello ws query"}]}"#
                    .to_string()
                    .into(),
            ),
            false,
        );

        assert_eq!(mirror.request_query.as_deref(), Some("hello ws query"));
    }

    #[test]
    fn websocket_mirror_truncates_events_by_utf8_bytes() {
        let mut mirror = WebSocketMirrorState::default();
        mirror.push_event_limited("你好abc".to_string(), 5);

        assert_eq!(mirror.event_messages, vec!["你".to_string()]);
        assert!(mirror.truncated);
        assert_eq!(mirror.event_bytes, "你".len());
    }

    #[test]
    fn benign_websocket_read_errors_are_detected() {
        assert!(is_benign_websocket_read_error(&WsError::ConnectionClosed));
        assert!(is_benign_websocket_read_error(&WsError::AlreadyClosed));
        assert!(is_benign_websocket_read_error(&WsError::Protocol(
            WsProtocolError::ResetWithoutClosingHandshake
        )));
        assert!(!is_benign_websocket_read_error(&WsError::Protocol(
            WsProtocolError::SendAfterClosing
        )));
    }

    // --- Stream A fixes: security + drift regression tests ---

    fn build_jwt_with_payload(payload: serde_json::Value) -> String {
        fake_jwt_payload(payload)
    }

    fn headers_with_bearer(token: &str) -> hyper::HeaderMap {
        let mut h = hyper::HeaderMap::new();
        h.insert(
            "authorization",
            hyper::header::HeaderValue::from_str(&format!("Bearer {token}")).unwrap(),
        );
        h
    }

    fn empty_mirror_state() -> WebSocketMirrorState {
        WebSocketMirrorState {
            pending: Vec::new(),
            fragmented_payload: None,
            fragmented_compressed: false,
            event_messages: Vec::new(),
            assistant_text: String::new(),
            request_query: None,
            event_bytes: 0,
            truncated: false,
            close_seen: false,
        }
    }

    #[test]
    fn bearer_jwt_info_ignores_expired_token() {
        // exp is 1 hour in the past ⇒ function must return None.
        let exp = current_unix_timestamp() - 3600;
        let token = build_jwt_with_payload(serde_json::json!({
            "exp": exp,
            "scp": ["api.responses.write"],
        }));
        let headers = headers_with_bearer(&token);
        assert!(
            bearer_jwt_info(&headers).is_none(),
            "expired JWT must not yield claims"
        );
    }

    #[test]
    fn bearer_jwt_info_accepts_future_exp() {
        let exp = current_unix_timestamp() + 3600;
        let token = build_jwt_with_payload(serde_json::json!({
            "exp": exp,
            "scp": ["api.responses.write"],
        }));
        let headers = headers_with_bearer(&token);
        let info = bearer_jwt_info(&headers).expect("valid JWT should parse");
        assert!(info.has_public_responses_scope);
    }

    #[test]
    fn redact_jwt_payload_keeps_only_safe_fields() {
        let payload = serde_json::json!({
            "iss": "issuer.example",
            "aud": "rein",
            "exp": 9999999999i64,
            "iat": 1_700_000_000i64,
            "sub": "user_123",
            "scp": ["api.responses.write"],
            "chatgpt_account_id": "acct_abc",
            "private_note": "SHOULD NOT LEAK",
            "email": "user@example.com",
            "access_token": "rot-secret",
        });
        let r = redact_jwt_payload(&payload);
        // Safe claims retained.
        assert_eq!(r.get("iss").and_then(|v| v.as_str()), Some("issuer.example"));
        assert_eq!(r.get("aud").and_then(|v| v.as_str()), Some("rein"));
        assert_eq!(r.get("scp"), payload.get("scp"));
        // Identifying claims dropped (v0.18.2: tightened allowlist).
        assert!(r.get("sub").is_none(), "sub must not leak");
        assert!(
            r.get("chatgpt_account_id").is_none(),
            "chatgpt_account_id must not leak"
        );
        // Presence-only signal of ChatGPT-login shape.
        assert_eq!(r.get("has_chatgpt_login").and_then(|v| v.as_bool()), Some(true));
        // Everything else dropped.
        assert!(r.get("private_note").is_none());
        assert!(r.get("email").is_none());
        assert!(r.get("access_token").is_none());
    }

    #[test]
    fn redact_jwt_payload_has_chatgpt_login_false_for_api_key() {
        let payload = serde_json::json!({
            "iss": "issuer",
            "scp": ["api.responses.write"],
        });
        let r = redact_jwt_payload(&payload);
        assert_eq!(r.get("has_chatgpt_login").and_then(|v| v.as_bool()), Some(false));
    }

    #[test]
    fn redact_jwt_payload_has_chatgpt_login_true_for_nested_auth() {
        let payload = serde_json::json!({
            "iss": "issuer",
            "https://api.openai.com/auth": {"chatgpt_account_id": "acct_42"},
        });
        let r = redact_jwt_payload(&payload);
        assert_eq!(r.get("has_chatgpt_login").and_then(|v| v.as_bool()), Some(true));
        // The raw account id must NOT appear anywhere in the redacted value.
        let serialized = r.to_string();
        assert!(
            !serialized.contains("acct_42"),
            "account id must not appear in redacted payload, got: {serialized}"
        );
    }

    #[test]
    fn hashed_account_fingerprint_in_process_properties() {
        // Narrow contract: stable within a single process, different inputs
        // get different tags, raw input does not appear in the tag.
        // Cross-process / cross-build stability is NOT promised — see jwt.rs docs.
        let fp1 = jwt::hashed_account_fingerprint("acct_abc123");
        let fp2 = jwt::hashed_account_fingerprint("acct_abc123");
        let fp3 = jwt::hashed_account_fingerprint("acct_different");

        assert_eq!(fp1, fp2, "same input within a process must produce same tag");
        assert_ne!(fp1, fp3, "different inputs should produce different tags");
        assert_eq!(fp1.len(), 8, "8 hex chars");
        assert!(fp1.chars().all(|c| c.is_ascii_hexdigit()));
        assert!(
            !fp1.contains("acct"),
            "raw input substring must not appear in the tag"
        );
    }

    #[test]
    fn websocket_mirror_rejects_inflate_bomb() {
        use flate2::write::DeflateEncoder;
        use flate2::Compression;
        use std::io::Write as _;
        // Build a compressed payload that decompresses past the 1 MiB cap.
        let raw = vec![b'a'; (WebSocketMirrorState::MAX_INFLATED_BYTES as usize) + 4096];
        let mut enc = DeflateEncoder::new(Vec::new(), Compression::best());
        enc.write_all(&raw).unwrap();
        let compressed = enc.finish().unwrap();
        // The highly compressible 'a' repetition yields a tiny compressed blob that
        // decompresses past the cap.
        assert!(compressed.len() < raw.len() / 10);
        let decoded = WebSocketMirrorState::decode_text_payload(&compressed, true);
        assert!(
            decoded.is_none(),
            "inflate cap must reject oversized decompressed payload"
        );
    }

    #[test]
    fn websocket_mirror_accepts_small_compressed_payload() {
        use flate2::write::DeflateEncoder;
        use flate2::Compression;
        use std::io::Write as _;
        let raw = b"response.output_text.delta hello";
        let mut enc = DeflateEncoder::new(Vec::new(), Compression::best());
        enc.write_all(raw).unwrap();
        let compressed = enc.finish().unwrap();
        let decoded = WebSocketMirrorState::decode_text_payload(&compressed, true);
        assert_eq!(decoded.as_deref(), Some("response.output_text.delta hello"));
    }

    #[test]
    fn provider_kind_detects_api_codex_family() {
        assert_eq!(
            ProviderKind::detect("/api/codex/tasks/list"),
            Some(ProviderKind::CodexFirstParty)
        );
        assert_eq!(
            ProviderKind::detect("/api/codex/responses"),
            Some(ProviderKind::CodexFirstParty)
        );
    }

    #[test]
    fn provider_kind_rewrites_api_codex_prefix() {
        let pk = ProviderKind::CodexFirstParty;
        assert_eq!(&*pk.rewrite_path("/api/codex/tasks/list"), "/tasks/list");
        assert_eq!(&*pk.rewrite_path("/api/codex/responses"), "/responses");
    }

    #[test]
    fn prefers_codex_first_party_no_longer_has_dead_branch() {
        // After M1: any ChatGPT-login JWT on an ambiguous path routes to CodexFirstParty.
        let token = fake_chatgpt_login_jwt(&["openid", "profile"]);
        let headers = headers_with_bearer(&token);
        // Ambiguous /responses → CodexFirstParty when chatgpt_account_id is present.
        assert!(prefers_codex_first_party("/responses", &headers));
        // Ambiguous /models → same outcome.
        assert!(prefers_codex_first_party("/models", &headers));
        // Without JWT → false.
        let empty = hyper::HeaderMap::new();
        assert!(!prefers_codex_first_party("/responses", &empty));
    }

    #[test]
    fn websocket_mirror_new_text_frame_clears_fragmentation_state() {
        let mut state = empty_mirror_state();
        // Inject a stale fragmented payload to simulate mid-fragmentation.
        state.fragmented_payload = Some(vec![b'p', b'a', b'r', b't']);
        state.fragmented_compressed = true;
        // Build a complete (fin=1) text frame with no mask, short payload "x".
        // RFC 6455 server→client: fin=1, rsv1=0, opcode=1, masked=0, len=1, payload=x.
        let frame = [0x81, 0x01, b'x'];
        state.feed(&frame, true);
        assert!(
            state.fragmented_payload.is_none(),
            "new text frame must clear stale fragmentation state"
        );
        assert!(!state.fragmented_compressed);
    }

    #[test]
    fn websocket_mirror_close_clears_fragmentation_state() {
        let mut state = empty_mirror_state();
        state.fragmented_payload = Some(vec![b'x'; 10]);
        state.fragmented_compressed = true;
        // Close frame: fin=1, opcode=8, unmasked, zero payload → [0x88, 0x00].
        let frame = [0x88, 0x00];
        state.feed(&frame, true);
        assert!(state.close_seen);
        assert!(state.fragmented_payload.is_none());
        assert!(!state.fragmented_compressed);
    }

    #[test]
    fn websocket_upstream_failure_response_is_uniformly_426() {
        // L1: unify 426 for all provider kinds on WS upgrade refusal.
        for kind in [
            ProviderKind::OpenAiResponses,
            ProviderKind::CodexFirstParty,
            ProviderKind::ChatGptBackend,
            ProviderKind::Anthropic,
            ProviderKind::OpenAi,
        ] {
            let resp = websocket_upstream_failure_response(kind);
            assert_eq!(
                resp.status(),
                hyper::StatusCode::UPGRADE_REQUIRED,
                "provider {:?} must yield 426",
                kind
            );
        }
    }

    #[test]
    fn should_strip_request_header_preserves_codex_family() {
        // H7 regression: the header denylist must NOT include Codex-specific
        // identification headers — they pass through verbatim.
        for name in [
            "x-codex-turn-state",
            "x-codex-installation-id",
            "x-codex-window-id",
            "x-codex-parent-thread-id",
            "openai-beta",
            "originator",
            "x-openai-fedramp",
            "x-request-id",
            "x-client-request-id",
            "user-agent",
        ] {
            assert!(
                !should_strip_request_header(name),
                "Codex-family header {name} must pass through"
            );
        }
        // And the strip list still catches hop-by-hop + rein's own token.
        assert!(should_strip_request_header("host"));
        assert!(should_strip_request_header("x-rein-token"));
        assert!(should_strip_request_header("content-length"));
    }

    // --- v0.18.1 WS boundary-case hardening tests ---

    #[test]
    fn websocket_mirror_rejects_oversize_len127_frame() {
        // Craft a frame header with len == 127 (8-byte extended length) whose
        // declared length exceeds the cap. Must NOT panic / allocate — state
        // should clear `pending` and reset fragmentation.
        let mut state = empty_mirror_state();
        // Pre-seed a fragmentation context to make sure the guard clears it.
        state.fragmented_payload = Some(b"stale".to_vec());
        state.fragmented_compressed = true;
        // Frame: fin=1, opcode=1 (text), masked=0, len byte=127.
        // Extended length = 1 GiB (exceeds 16 MiB cap).
        let mut frame = vec![0x81, 127];
        frame.extend_from_slice(&(1_073_741_824u64).to_be_bytes());
        state.feed(&frame, true);
        assert!(
            state.fragmented_payload.is_none(),
            "oversize frame must clear fragmentation state"
        );
        assert!(
            state.pending.is_empty(),
            "oversize frame must drain pending buffer"
        );
    }

    #[test]
    fn websocket_mirror_defers_on_partial_len127_header() {
        // If only 5 of the 8 extended-length bytes have arrived, the mirror
        // must NOT consume anything and must wait for more data.
        let mut state = empty_mirror_state();
        // 2-byte header + 5 (not 8) bytes of extended length = partial frame.
        let partial = [0x81u8, 127, 0, 0, 0, 0, 1];
        state.feed(&partial, true);
        assert_eq!(
            state.pending.len(),
            partial.len(),
            "partial len==127 header must be held in pending for more data"
        );
    }

    #[test]
    fn websocket_mirror_decodes_malformed_zlib_gracefully() {
        // A compressed-text frame whose payload is not a valid deflate stream must
        // return None from decode_text_payload, not panic.
        let garbage = [0xff, 0xfe, 0xfd, 0xfc, 0xfb];
        let out = WebSocketMirrorState::decode_text_payload(&garbage, true);
        assert!(out.is_none(), "malformed deflate must not panic and must return None");
    }

    #[test]
    fn websocket_mirror_ping_frame_does_not_corrupt_state() {
        let mut state = empty_mirror_state();
        // Inject an active fragmentation context; a ping frame must not disturb it.
        state.fragmented_payload = Some(b"partial".to_vec());
        state.fragmented_compressed = false;
        // Ping frame: fin=1, opcode=9, zero payload → [0x89, 0x00].
        let ping = [0x89, 0x00];
        state.feed(&ping, true);
        // Ping is ignored (opcode not in {0x0, 0x1, 0x8}); fragmentation state preserved.
        assert_eq!(state.fragmented_payload.as_deref(), Some(b"partial".as_slice()));
        assert!(!state.close_seen);
    }

    #[test]
    fn websocket_mirror_pong_frame_does_not_corrupt_state() {
        let mut state = empty_mirror_state();
        state.fragmented_payload = Some(b"partial".to_vec());
        // Pong frame: fin=1, opcode=10 (0xA), zero payload → [0x8A, 0x00].
        let pong = [0x8a, 0x00];
        state.feed(&pong, true);
        assert_eq!(state.fragmented_payload.as_deref(), Some(b"partial".as_slice()));
        assert!(!state.close_seen);
    }

    #[test]
    fn websocket_mirror_decodes_compressed_fragmented_message() {
        // Build a compressed message, then split it into start+continuation fragments
        // at a byte boundary. Both fragments carry rsv1=1 conceptually — but per RFC
        // 7692, only the first (opcode 0x1) frame carries rsv1; continuation frames
        // (opcode 0x0) inherit the compressed context from the start frame.
        use flate2::write::DeflateEncoder;
        use flate2::Compression;
        use std::io::Write as _;
        let raw = b"response.output_text.delta hello world";
        let mut enc = DeflateEncoder::new(Vec::new(), Compression::best());
        enc.write_all(raw).unwrap();
        let compressed = enc.finish().unwrap();
        let mid = compressed.len() / 2;
        let (a, b) = compressed.split_at(mid);

        let mut state = empty_mirror_state();
        // Start frame: fin=0, rsv1=1, opcode=1 → 0x41 (first byte), len = a.len().
        // Byte 0: 0b0_1_00_0001 (fin=0, rsv1=1, rsv2=0, rsv3=0, opcode=1) = 0x41
        // We assume a.len() < 126 for simplicity in this synthetic test.
        assert!(a.len() < 126, "test requires small first half");
        let mut frame1 = vec![0x41, a.len() as u8];
        frame1.extend_from_slice(a);
        state.feed(&frame1, true);
        assert!(
            state.fragmented_payload.is_some(),
            "start frame must open fragmentation context"
        );
        assert!(state.fragmented_compressed);

        // Continuation + fin=1: fin=1, rsv1=0, opcode=0 → 0x80, len = b.len().
        assert!(b.len() < 126, "test requires small second half");
        let mut frame2 = vec![0x80, b.len() as u8];
        frame2.extend_from_slice(b);
        state.feed(&frame2, true);
        // The fragmented payload is consumed after the final frame.
        assert!(state.fragmented_payload.is_none());
        // One event got recorded with the decoded text.
        let combined = state.event_messages.join("\n");
        assert!(
            combined.contains("response.output_text.delta"),
            "decoded output must surface in event log, got: {combined}"
        );
    }

    #[test]
    fn decode_jwt_payload_for_logging_returns_none_for_missing_header() {
        let headers = hyper::HeaderMap::new();
        assert!(decode_jwt_payload_for_logging(&headers).is_none());
    }

    #[test]
    fn decode_jwt_payload_for_logging_returns_none_for_non_bearer() {
        let mut headers = hyper::HeaderMap::new();
        headers.insert(
            "authorization",
            hyper::header::HeaderValue::from_static("Basic dXNlcjpwYXNz"),
        );
        assert!(decode_jwt_payload_for_logging(&headers).is_none());
    }

    #[test]
    fn decode_jwt_payload_for_logging_extracts_claims_for_redaction() {
        let token = build_jwt_with_payload(serde_json::json!({
            "exp": current_unix_timestamp() + 3600,
            "scp": ["openid"],
            "private_note": "SHOULD NOT LEAK AFTER REDACT",
        }));
        let headers = headers_with_bearer(&token);
        let decoded = decode_jwt_payload_for_logging(&headers).expect("valid JWT should decode");
        // The raw decode includes private_note (pre-redaction).
        assert!(decoded.get("private_note").is_some());
        // Redaction drops it.
        let red = redact_jwt_payload(&decoded);
        assert!(red.get("private_note").is_none());
        assert!(red.get("exp").is_some());
    }

    // --- v0.19.0 upstream-error passthrough tests (audit LOW) ---
    //
    // These verify that status codes meaningful to clients — 5xx server errors,
    // 429 rate-limit, and mid-stream connection drops — are passed through
    // unchanged so clients can implement their own retry/backoff. rein does NOT
    // rewrite them to 4xx or internal status codes.

    #[tokio::test(flavor = "current_thread")]
    async fn proxy_passes_through_upstream_500_status_unchanged() {
        let (openai_upstream, _openai_rx) =
            spawn_capture_http_server("500 Internal Server Error", r#"{"error":"boom"}"#);
        let (codex_upstream, _codex_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");
        let (chatgpt_upstream, _chatgpt_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");

        let mut config = ReinConfig::default();
        config.proxy.openai_upstream = openai_upstream;
        config.proxy.codex_upstream = codex_upstream;
        config.proxy.chatgpt_upstream = chatgpt_upstream;

        let (proxy_base, proxy_task) =
            spawn_one_shot_proxy(config, Some("secret".to_string())).await;
        let token = fake_jwt_with_scopes(&["api.responses.read", "api.responses.write"]);
        let response = reqwest::Client::new()
            .post(format!("{proxy_base}/responses"))
            .header("x-rein-token", "secret")
            .header("authorization", format!("Bearer {token}"))
            .header("content-type", "application/json")
            .body(r#"{"input":"hello"}"#)
            .send()
            .await
            .unwrap();
        assert_eq!(
            response.status().as_u16(),
            500,
            "upstream 500 must pass through to client unchanged"
        );
        let body = response.text().await.unwrap_or_default();
        assert!(
            body.contains("boom"),
            "upstream body must pass through unchanged, got: {body}"
        );
        proxy_task.abort();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn proxy_passes_through_upstream_429_with_retry_after_header() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let addr = listener.local_addr().unwrap();
        // Hand-crafted server so we can add a Retry-After header.
        thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 4096];
            let _ = stream.read(&mut buf);
            let _ = stream.write_all(
                b"HTTP/1.1 429 Too Many Requests\r\n\
Retry-After: 42\r\n\
Content-Type: application/json\r\n\
Content-Length: 18\r\n\
Connection: close\r\n\r\n\
{\"error\":\"slow\"}\r\n",
            );
        });

        let openai_upstream = format!("http://{}", addr);
        let (codex_upstream, _codex_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");
        let (chatgpt_upstream, _chatgpt_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");

        let mut config = ReinConfig::default();
        config.proxy.openai_upstream = openai_upstream;
        config.proxy.codex_upstream = codex_upstream;
        config.proxy.chatgpt_upstream = chatgpt_upstream;

        let (proxy_base, proxy_task) =
            spawn_one_shot_proxy(config, Some("secret".to_string())).await;
        let token = fake_jwt_with_scopes(&["api.responses.read", "api.responses.write"]);
        let response = reqwest::Client::new()
            .post(format!("{proxy_base}/responses"))
            .header("x-rein-token", "secret")
            .header("authorization", format!("Bearer {token}"))
            .header("content-type", "application/json")
            .body(r#"{"input":"hello"}"#)
            .send()
            .await
            .unwrap();
        assert_eq!(
            response.status().as_u16(),
            429,
            "upstream 429 must pass through"
        );
        let retry_after = response
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        assert_eq!(
            retry_after.as_deref(),
            Some("42"),
            "Retry-After header must pass through"
        );
        proxy_task.abort();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn proxy_surfaces_upstream_connection_drop_as_5xx() {
        // Bind a listener that accepts then immediately drops the connection
        // before writing any response bytes.  reqwest/hyper should return an
        // error; the proxy should map this to a 5xx status.
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let addr = listener.local_addr().unwrap();
        thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            drop(stream);
        });

        let openai_upstream = format!("http://{}", addr);
        let (codex_upstream, _codex_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");
        let (chatgpt_upstream, _chatgpt_rx) = spawn_capture_http_server("200 OK", "{\"ok\":true}");

        let mut config = ReinConfig::default();
        config.proxy.openai_upstream = openai_upstream;
        config.proxy.codex_upstream = codex_upstream;
        config.proxy.chatgpt_upstream = chatgpt_upstream;

        let (proxy_base, proxy_task) =
            spawn_one_shot_proxy(config, Some("secret".to_string())).await;
        let token = fake_jwt_with_scopes(&["api.responses.read", "api.responses.write"]);
        let response = reqwest::Client::new()
            .post(format!("{proxy_base}/responses"))
            .header("x-rein-token", "secret")
            .header("authorization", format!("Bearer {token}"))
            .header("content-type", "application/json")
            .body(r#"{"input":"hello"}"#)
            .send()
            .await
            .unwrap();
        // Tightened in v0.19.1 (Codex review LOW): the current implementation
        // in `handle_request` maps upstream-connection errors deterministically
        // to `error_response(502, "upstream request failed")`. Assert the
        // exact code + body so a future accidental regression to 503/504 or
        // a different body message is caught immediately.
        assert_eq!(
            response.status().as_u16(),
            502,
            "upstream drop must surface as exactly 502"
        );
        let body = response.text().await.unwrap_or_default();
        assert!(
            body.contains("upstream request failed"),
            "502 body must carry the stable 'upstream request failed' marker, got: {body}"
        );
        proxy_task.abort();
    }
}
