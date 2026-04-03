//! Transparent LLM API proxy with memory injection and extraction.
//!
//! Intercepts requests to Anthropic (`/v1/messages`) and OpenAI (`/v1/chat/completions`)
//! APIs, injects recalled memories into the system prompt, forwards to the upstream
//! provider, streams responses back, and asynchronously extracts memories from responses.
//!
//! Uses a dedicated blocking thread with a resident SqliteStore to avoid per-request
//! connection overhead and to prevent blocking the tokio async executor.

mod anthropic;
mod extract;
mod inject;
mod openai;
mod policy;
mod provider;

use crate::config::ReinConfig;
use bytes::Bytes;
use http_body_util::{BodyExt, Full, StreamBody};
use hyper::body::Frame;
use provider::ProviderKind;
use tokio::sync::{mpsc, oneshot};

/// A recall request sent to the dedicated store thread.
struct RecallRequest {
    query: String,
    budget_tokens: usize,
    reply: oneshot::Sender<Option<String>>,
}

/// Start the transparent proxy server.
pub async fn run_proxy(config: ReinConfig) -> anyhow::Result<()> {
    // Signal hooks to skip injection (proxy handles it).
    std::env::set_var("REIN_PROXY_ACTIVE", "1");

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

    // Injection is optional. In record-only mode we skip the resident recall
    // thread entirely so the proxy stays as close to a transparent forwarder as
    // possible.
    let recall_tx = if config.proxy.inject_enabled {
        let (recall_tx, mut recall_rx) = mpsc::channel::<RecallRequest>(32);
        let config_for_store = config.clone();
        std::thread::spawn(move || {
            let store = match config_for_store.open_store() {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("rein proxy: failed to open store: {e}");
                    return;
                }
            };

            // Pre-warm indexes on the store thread (Layer 1: resident indexes).
            crate::search::warmup::populate_tantivy(&store);
            crate::search::warmup::populate_hnsw(&store, &config_for_store);
            eprintln!("rein proxy: indexes warmed up");

            while let Some(req) = recall_rx.blocking_recv() {
                let result =
                    inject::recall_and_format(&store, &config_for_store, &req.query, req.budget_tokens);
                let _ = req.reply.send(result);
            }
        });
        Some(recall_tx)
    } else {
        eprintln!("rein proxy: running in record-only mode (injection disabled)");
        None
    };

    eprintln!("rein proxy listening on http://{bind}");
    eprintln!("  Anthropic: set ANTHROPIC_BASE_URL=http://{bind}");
    eprintln!("  OpenAI:    set OPENAI_BASE_URL=http://{bind}");

    // Shared reqwest client for all upstream requests (connection pooling).
    let upstream_client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()?;

    loop {
        let (stream, addr) = listener.accept().await?;
        tracing::debug!("proxy connection from {addr}");
        let config = config.clone();
        let client = upstream_client.clone();
        let recall_tx = recall_tx.clone();

        let auth = auth_token.clone();
        tokio::spawn(async move {
            let service = hyper::service::service_fn(move |req| {
                let config = config.clone();
                let client = client.clone();
                let auth = auth.clone();
                let recall_tx = recall_tx.clone();
                async move {
                    handle_request(req, config, client, auth.as_deref(), recall_tx.clone()).await
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
        .unwrap()
}

/// Handle a single proxied request.
async fn handle_request(
    req: hyper::Request<hyper::body::Incoming>,
    config: ReinConfig,
    client: reqwest::Client,
    expected_token: Option<&str>,
    recall_tx: Option<mpsc::Sender<RecallRequest>>,
) -> Result<hyper::Response<BoxBody>, hyper::Error> {
    // Auth check for non-localhost binds.
    if let Some(expected) = expected_token {
        let auth_header = req
            .headers()
            .get("x-rein-token")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        if auth_header != expected {
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

    // Detect provider from request path.
    let provider = ProviderKind::detect(&path);

    // Read full request body.
    let body_bytes = match req.collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => {
            tracing::warn!("failed to read request body: {e}");
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
    let (modified_body, query) = if config.proxy.inject_enabled {
        match inject_memories(&config, &provider, &body_bytes, recall_tx.as_ref()).await {
            Ok(result) => result,
            Err(e) => {
                tracing::warn!("injection failed, forwarding unmodified: {e}");
                (body_bytes.to_vec(), query)
            }
        }
    } else {
        (body_bytes.to_vec(), query)
    };

    let injected = modified_body.len() != body_size;
    eprintln!(
        "rein proxy: injected={injected} query={:?} orig={body_size} modified={}",
        query.as_deref().unwrap_or(""),
        modified_body.len()
    );

    // Build upstream URL.
    let upstream_base = provider.upstream_url(&config);
    let upstream_url = format!("{upstream_base}{path_and_query}");

    // Build upstream request with original headers.
    let mut upstream_req = client.request(
        reqwest::Method::from_bytes(method.as_str().as_bytes()).unwrap_or(reqwest::Method::POST),
        &upstream_url,
    );

    for (name, value) in headers.iter() {
        let name_str = name.as_str();
        // Skip hop-by-hop headers and content-length (recalculated).
        if matches!(
            name_str,
            "host" | "content-length" | "transfer-encoding" | "connection"
        ) {
            continue;
        }
        if let Ok(v) = value.to_str() {
            upstream_req = upstream_req.header(name_str, v);
        }
    }

    upstream_req = upstream_req
        .header("content-length", modified_body.len().to_string())
        .body(modified_body);

    // Send upstream request.
    let upstream_resp = match upstream_req.send().await {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("upstream request failed: {e}");
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
        stream_response(upstream_resp, status, &resp_headers, &config, &provider, query).await
    } else {
        // Non-streaming: read full response, extract, forward.
        let resp_body = upstream_resp.bytes().await.unwrap_or_default();

        // Async extract from non-streaming response.
        if config.proxy.extract_enabled {
            if let Some(text) = provider.extract_assistant_text_full(&resp_body) {
                if policy::should_extract_response(&config, query.as_deref(), &text) {
                    let cfg = config.clone();
                    let query = query.clone();
                    tokio::spawn(async move {
                        extract::extract_and_store(&cfg, query, text).await;
                    });
                }
            }
        }

        let mut builder = hyper::Response::builder().status(status.as_u16());
        for (name, value) in resp_headers.iter() {
            if name.as_str() != "transfer-encoding" {
                builder = builder.header(name.as_str(), value);
            }
        }
        Ok(builder.body(full_body(resp_body)).unwrap())
    }
}

/// Stream SSE response back to client while buffering assistant text.
async fn stream_response(
    upstream_resp: reqwest::Response,
    status: reqwest::StatusCode,
    resp_headers: &reqwest::header::HeaderMap,
    config: &ReinConfig,
    provider: &ProviderKind,
    query: Option<String>,
) -> Result<hyper::Response<BoxBody>, hyper::Error> {
    let (tx, rx) = mpsc::channel::<Result<Frame<Bytes>, hyper::Error>>(64);

    let extract_enabled = config.proxy.extract_enabled;
    let provider_clone = *provider;
    let config_clone = config.clone();
    let query_clone = query.clone();

    // Spawn task to read upstream stream, forward chunks, buffer text.
    tokio::spawn(async move {
        let mut stream = upstream_resp.bytes_stream();
        let mut assistant_buf = String::new();
        // SSE line buffer: transport chunks may split across SSE event boundaries.
        let mut sse_line_buf = String::new();
        const MAX_EXTRACT_BUF: usize = 200_000; // ~50K tokens, prevent OOM

        use futures_util::StreamExt;
        while let Some(chunk_result) = stream.next().await {
            match chunk_result {
                Ok(chunk) => {
                    // Parse SSE chunks for assistant text extraction.
                    // Buffer incomplete lines across chunk boundaries.
                    if extract_enabled && assistant_buf.len() < MAX_EXTRACT_BUF {
                        if let Ok(text) = std::str::from_utf8(&chunk) {
                            sse_line_buf.push_str(text);
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

        // After stream completes, extract memories.
        if extract_enabled
            && policy::should_extract_response(&config_clone, query_clone.as_deref(), &assistant_buf)
        {
            extract::extract_and_store(&config_clone, query_clone, assistant_buf).await;
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

    Ok(builder.body(box_body(stream_body)).unwrap())
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
            Ok(builder.body(full_body(resp_body)).unwrap())
        }
        Err(e) => {
            tracing::warn!("forward_raw upstream error: {e}");
            Ok(error_response(502, "upstream request failed"))
        }
    }
}

/// Parse request body, inject memories, return modified body and extracted query.
/// Recall runs asynchronously on the dedicated store thread via channel.
async fn inject_memories(
    config: &ReinConfig,
    provider: &ProviderKind,
    body_bytes: &[u8],
    recall_tx: Option<&mpsc::Sender<RecallRequest>>,
) -> anyhow::Result<(Vec<u8>, Option<String>)> {
    let Some(recall_tx) = recall_tx else {
        return Ok((body_bytes.to_vec(), None));
    };
    let mut body: serde_json::Value = serde_json::from_slice(body_bytes)?;

    // Extract query from messages.
    let query = provider.extract_query(&body);
    if !policy::should_attempt_injection(config, &query) {
        return Ok((body_bytes.to_vec(), None));
    }

    // Skip if already injected (prevent re-injection on retries).
    if provider.has_injected_context(&body) {
        return Ok((body_bytes.to_vec(), Some(query)));
    }

    // Estimate current token usage and compute injection budget.
    let body_str = serde_json::to_string(&body)?;
    let used_tokens = body_str.len() / 4;
    let model = body
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("unknown");
    let model_max = inject::model_max_tokens(model);
    // Subtract requested output tokens from available window.
    let output_reserved = body
        .get("max_tokens")
        .or_else(|| body.get("max_completion_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(4096) as usize;
    let remaining = model_max
        .saturating_sub(used_tokens)
        .saturating_sub(output_reserved);
    let budget_tokens = (remaining / 20).min(config.proxy.inject_limit); // remaining * 0.05

    if budget_tokens < 50 {
        return Ok((body_bytes.to_vec(), Some(query)));
    }

    // Send recall request to dedicated store thread (non-blocking).
    let (reply_tx, reply_rx) = oneshot::channel();
    if recall_tx
        .send(RecallRequest {
            query: query.clone(),
            budget_tokens,
            reply: reply_tx,
        })
        .await
        .is_err()
    {
        // Store thread died — forward unmodified.
        return Ok((body_bytes.to_vec(), Some(query)));
    }

    // Keep proxy invisible: if recall doesn't finish quickly, forward as-is.
    let context = match tokio::time::timeout(
        std::time::Duration::from_millis(config.proxy.inject_timeout_ms),
        reply_rx,
    )
    .await
    {
        Ok(Ok(Some(ctx))) => ctx,
        _ => return Ok((body_bytes.to_vec(), Some(query))),
    };

    // Inject into the appropriate location.
    provider.inject_context(&mut body, &context);

    let modified = serde_json::to_vec(&body)?;
    Ok((modified, Some(query)))
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
