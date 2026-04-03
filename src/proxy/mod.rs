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

    // Spawn a dedicated blocking thread that owns a resident SqliteStore.
    // All recall operations go through this thread via channel, avoiding:
    // 1. Per-request connection overhead (open_store ~10-30ms)
    // 2. Tantivy index reload per request (~15MB allocation)
    // 3. Blocking the tokio async executor with sync I/O
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

        // Process recall requests sequentially on this thread.
        // SqliteStore is !Send, so it must stay on this thread.
        while let Some(req) = recall_rx.blocking_recv() {
            let result =
                inject::recall_and_format(&store, &config_for_store, &req.query, req.budget_tokens);
            let _ = req.reply.send(result);
        }
    });

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
                    handle_request(req, config, client, auth.as_deref(), recall_tx).await
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
    recall_tx: mpsc::Sender<RecallRequest>,
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

    // If not a known LLM endpoint, passthrough unmodified.
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

    // Parse and inject memories (async — recall runs on dedicated thread).
    let (modified_body, query) =
        match inject_memories(&config, &provider, &body_bytes, &recall_tx).await {
            Ok(result) => result,
            Err(e) => {
                tracing::warn!("injection failed, forwarding unmodified: {e}");
                (body_bytes.to_vec(), None)
            }
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
                if !text.is_empty() {
                    let cfg = config.clone();
                    tokio::spawn(async move {
                        extract::extract_and_store(&cfg, text).await;
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
    _query: Option<String>,
) -> Result<hyper::Response<BoxBody>, hyper::Error> {
    let (tx, rx) = mpsc::channel::<Result<Frame<Bytes>, hyper::Error>>(64);

    let extract_enabled = config.proxy.extract_enabled;
    let provider_clone = *provider;
    let config_clone = config.clone();

    // Spawn task to read upstream stream, forward chunks, buffer text.
    tokio::spawn(async move {
        let mut stream = upstream_resp.bytes_stream();
        let mut assistant_buf = String::new();
        const MAX_EXTRACT_BUF: usize = 200_000; // ~50K tokens, prevent OOM

        use futures_util::StreamExt;
        while let Some(chunk_result) = stream.next().await {
            match chunk_result {
                Ok(chunk) => {
                    // Parse SSE chunks for assistant text extraction.
                    if extract_enabled && assistant_buf.len() < MAX_EXTRACT_BUF {
                        if let Some(text) = provider_clone.extract_assistant_text_sse(&chunk) {
                            assistant_buf.push_str(&text);
                        }
                    }
                    // Forward chunk to client.
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
        if extract_enabled && !assistant_buf.is_empty() {
            extract::extract_and_store(&config_clone, assistant_buf).await;
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

/// Check if the body already contains rein-context injection.
fn body_str_contains_rein_context(body: &serde_json::Value) -> bool {
    let s = body.to_string();
    s.contains("<rein-context>")
}

/// Parse request body, inject memories, return modified body and extracted query.
/// Recall runs asynchronously on the dedicated store thread via channel.
async fn inject_memories(
    config: &ReinConfig,
    provider: &ProviderKind,
    body_bytes: &[u8],
    recall_tx: &mpsc::Sender<RecallRequest>,
) -> anyhow::Result<(Vec<u8>, Option<String>)> {
    let mut body: serde_json::Value = serde_json::from_slice(body_bytes)?;

    // Extract query from messages.
    let query = provider.extract_query(&body);
    if query.is_empty() || query.chars().count() < 5 {
        return Ok((body_bytes.to_vec(), None));
    }

    // Skip if already injected (prevent re-injection on retries).
    if body_str_contains_rein_context(&body) {
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

    // Wait for recall result (typically ~2-5ms with warm indexes).
    let context = match reply_rx.await {
        Ok(Some(ctx)) => ctx,
        _ => return Ok((body_bytes.to_vec(), Some(query))),
    };

    // Inject into the appropriate location.
    provider.inject_context(&mut body, &context);

    let modified = serde_json::to_vec(&body)?;
    Ok((modified, Some(query)))
}
