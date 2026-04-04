//! Transparent LLM API proxy with conservative recording and extraction.
//!
//! Intercepts requests to Anthropic (`/v1/messages`) and OpenAI (`/v1/chat/completions`)
//! APIs, forwards them to the upstream provider, streams responses back, and
//! asynchronously records memory candidates from responses.

mod anthropic;
mod extract;
mod openai;
mod policy;
mod provider;
mod responses;

use crate::config::ReinConfig;
use bytes::Bytes;
use http_body_util::{BodyExt, Full, StreamBody};
use hyper::body::Frame;
use provider::ProviderKind;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;

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

/// Start the transparent proxy server.
pub async fn run_proxy(config: ReinConfig) -> anyhow::Result<()> {
    // REIN_PROXY_ACTIVE is set by the caller (main.rs) before entering async
    // to avoid unsound set_var in multi-threaded context.

    let bind = format!("{}:{}", config.proxy.bind, config.proxy.port);

    // Security: require auth token for non-localhost binds (same as run_http).
    let is_loopback = config.proxy.bind == "127.0.0.1"
        || config.proxy.bind == "localhost"
        || config.proxy.bind == "::1";
    let auth_token = std::env::var("REIN_PROXY_TOKEN")
        .ok()
        .or_else(|| std::env::var("REIN_HTTP_TOKEN").ok());
    if !is_loopback && auth_token.is_none() {
        anyhow::bail!(
            "rein proxy: refusing to bind to non-loopback address '{}' without REIN_PROXY_TOKEN set. \
             Set REIN_PROXY_TOKEN=<secret> or bind to 127.0.0.1.",
            config.proxy.bind
        );
    }

    let listener = tokio::net::TcpListener::bind(&bind).await?;
    eprintln!("rein proxy: running in record-only mode (automatic injection removed)");

    eprintln!("rein proxy listening on http://{bind}");
    eprintln!("  Anthropic: set ANTHROPIC_BASE_URL=http://{bind}");
    eprintln!("  OpenAI:    set OPENAI_BASE_URL=http://{bind}");

    // Shared reqwest client for all upstream requests (connection pooling).
    let upstream_client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()?;

    let state = Arc::new(ProxyState {
        metrics: ProxyMetrics::new(),
    });

    // Graceful shutdown: stop accept loop on ctrl-c.
    loop {
        let accept = tokio::select! {
            res = listener.accept() => res,
            _ = tokio::signal::ctrl_c() => {
                tracing::info!("rein proxy: received shutdown signal, stopping accept loop");
                eprintln!("rein proxy: shutting down gracefully");
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
                async move {
                    handle_request(req, config, client, auth.as_deref(), state).await
                }
            });

            if let Err(e) = hyper_util::server::conn::auto::Builder::new(
                hyper_util::rt::TokioExecutor::new(),
            )
            .serve_connection(hyper_util::rt::TokioIo::new(stream), service)
            .await
            {
                tracing::warn!("proxy connection error: {e}");
            }
        });
    }

    Ok(())
}

type BoxBody = http_body_util::combinators::BoxBody<Bytes, hyper::Error>;

fn box_body<B>(body: B) -> BoxBody
where
    B: hyper::body::Body<Data = Bytes, Error = hyper::Error> + Send + Sync + 'static,
{
    BoxBody::new(body)
}

fn full_body(data: Bytes) -> BoxBody {
    box_body(Full::new(data).map_err(|never| match never {}))
}

fn error_response(status: u16, msg: &str) -> hyper::Response<BoxBody> {
    hyper::Response::builder()
        .status(status)
        .header("content-type", "text/plain")
        .body(full_body(Bytes::from(msg.to_string())))
        .unwrap_or_else(|_| hyper::Response::new(full_body(Bytes::from("internal error"))))
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
    if let Some(expected) = expected_token {
        let auth_header = req
            .headers()
            .get("x-rein-token")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        if auth_header != expected {
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

    // Detect provider from request path. Sampling routes are tracked only to
    // capture source query metadata for recording; requests are not mutated.
    let provider = ProviderKind::detect(&path);

    // Read full request body.
    let body_bytes = match req.collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => {
            tracing::warn!("failed to read request body: {e}");
            state.metrics.error_count.fetch_add(1, Ordering::Relaxed);
            return Ok(error_response(400, "failed to read request body"));
        }
    };

    // Log request details.
    let body_size = body_bytes.len();
    eprintln!("rein proxy: {method} {path_and_query} ({body_size} bytes)");

    // If not a known sampling endpoint, passthrough unmodified.
    let provider = match provider {
        Some(p) => p,
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

    let query = extract_query_for_recording(&provider, &body_bytes);
    eprintln!(
        "rein proxy: injected=false query={:?} orig={body_size} modified={}",
        query.as_deref().unwrap_or(""),
        body_size
    );

    // Build upstream URL (rewrite path if needed, e.g. /responses → /v1/responses).
    let upstream_base = provider.upstream_url(&config);
    let rewritten_path = provider.rewrite_path(&path_and_query);
    let upstream_url = format!("{upstream_base}{rewritten_path}");

    // Build upstream headers (skip hop-by-hop, recalculate content-length).
    let mut upstream_headers = reqwest::header::HeaderMap::new();
    for (name, value) in headers.iter() {
        let name_str = name.as_str();
        if matches!(
            name_str,
            "host" | "content-length" | "transfer-encoding" | "connection"
        ) {
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
        stream_response(upstream_resp, status, &resp_headers, &config, &provider, query, &state)
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
                upstream_resp, status, &resp_headers, &config, &provider, query, &state,
            )
            .await;
        }

        // Non-streaming: read full response, extract, forward.
        let resp_body = upstream_resp.bytes().await.unwrap_or_default();

        // Async extract from non-streaming response (with backpressure).
        if config.proxy.extract_enabled {
            if let Some(text) = provider.extract_assistant_text_full(&resp_body) {
                if policy::should_extract_response(&config, query.as_deref(), &text) {
                    maybe_spawn_extraction(&config, &state, query.clone(), text);
                }
            }
        }

        let mut builder = hyper::Response::builder().status(status.as_u16());
        for (name, value) in resp_headers.iter() {
            if name.as_str() != "transfer-encoding" {
                builder = builder.header(name.as_str(), value);
            }
        }
        Ok(build_response(builder, full_body(resp_body)))
    }
}

/// Attempt to spawn an extraction task, respecting the concurrency semaphore.
fn maybe_spawn_extraction(
    config: &ReinConfig,
    state: &Arc<ProxyState>,
    query: Option<String>,
    text: String,
) {
    // extract_and_store only does a queue file append (cheap I/O),
    // so no concurrency limit needed here. Actual LLM extraction
    // happens in the background worker which has its own rate limiting.
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
    provider: &ProviderKind,
    query: Option<String>,
    state: &Arc<ProxyState>,
) -> Result<hyper::Response<BoxBody>, hyper::Error> {
    let (tx, rx) = mpsc::channel::<Result<Frame<Bytes>, hyper::Error>>(64);

    let extract_enabled = config.proxy.extract_enabled;
    let max_sse_buffer = config.proxy.max_sse_buffer;
    let provider_clone = *provider;
    let config_clone = config.clone();
    let query_clone = query.clone();
    let state_clone = Arc::clone(state);

    // Spawn task to read upstream stream, forward chunks, buffer text.
    tokio::spawn(async move {
        let mut stream = upstream_resp.bytes_stream();
        let mut assistant_buf = String::new();
        // SSE line buffer: transport chunks may split across SSE event boundaries.
        let mut sse_line_buf = String::new();
        // Whether SSE parsing has been abandoned due to buffer overflow.
        let mut sse_parsing_active = true;
        const MAX_EXTRACT_BUF: usize = 200_000; // ~50K tokens, prevent OOM

        use futures_util::StreamExt;
        while let Some(chunk_result) = stream.next().await {
            match chunk_result {
                Ok(chunk) => {
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
                                    sse_line_buf =
                                        sse_line_buf[newline_pos + 1..].to_string();
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
        if extract_enabled
            && policy::should_extract_response(
                &config_clone,
                query_clone.as_deref(),
                &assistant_buf,
            )
        {
            maybe_spawn_extraction(&config_clone, &state_clone, query_clone, assistant_buf);
        }
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
        if !matches!(
            name.as_str(),
            "host" | "content-length" | "transfer-encoding" | "connection"
        ) {
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
            let resp_body = resp.bytes().await.unwrap_or_default();

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

fn extract_query_for_recording(
    provider: &ProviderKind,
    body_bytes: &[u8],
) -> Option<String> {
    let body: serde_json::Value = serde_json::from_slice(body_bytes).ok()?;
    let query = provider.extract_query(&body);
    let query = query.trim();
    if query.is_empty() {
        None
    } else {
        Some(query.to_string())
    }
}
