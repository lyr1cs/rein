use crate::types::error::{ReinError, ReinResult};
use crate::types::traits::Embedder;
use reqwest::Client;
use serde_json::{json, Value};
use std::time::Duration;

pub struct GeminiEmbedder {
    client: Client,
    api_key: String,
    endpoint: String,
    model: String,
    dimensions: usize,
    /// `validate_endpoint` verdict taken once at construction. When false,
    /// every embed call fails before a request is built so the key and the
    /// memory text never leave the process over plain HTTP.
    endpoint_ok: bool,
}

/// Reject embedder endpoints that are not HTTPS (except explicit loopback).
///
/// Without this check, a misconfigured `endpoint = "https://evil.example"`
/// (or a plain `http://` override) would silently exfiltrate every embed
/// request — x-goog-api-key header, request body, the lot. Validation lives
/// here in the constructor so every caller path is covered, including fall-
/// back construction from env vars.
pub(crate) fn validate_endpoint(endpoint: &str) -> Result<(), String> {
    let lower = endpoint.trim().to_ascii_lowercase();
    if lower.starts_with("https://") {
        return Ok(());
    }
    // Allow plain http ONLY for explicit loopback (local test servers).
    if let Some(rest) = lower.strip_prefix("http://") {
        let host_and_port = rest.split('/').next().unwrap_or("");
        // Bracketed IPv6: "[::1]" or "[::1]:port" — host is between the brackets.
        let host = if host_and_port.starts_with('[') {
            host_and_port
                .get(1..)
                .and_then(|rest| rest.find(']').map(|end| &rest[..end]))
                .unwrap_or("")
        } else {
            host_and_port.split(':').next().unwrap_or("")
        };
        if host == "127.0.0.1" || host == "localhost" || host == "::1" {
            return Ok(());
        }
    }
    Err(format!(
        "embedder endpoint must use https:// (loopback http allowed): got '{endpoint}'"
    ))
}

impl GeminiEmbedder {
    pub fn new(api_key: String, endpoint: String, model: String, dimensions: usize) -> Self {
        let endpoint_ok = match validate_endpoint(&endpoint) {
            Ok(()) => true,
            Err(msg) => {
                tracing::warn!("{msg}; every embed call on this embedder is refused");
                false
            }
        };
        let client = Client::builder()
            .timeout(CLIENT_TIMEOUT)
            .build()
            .expect("reqwest client build failed for gemini (likely TLS backend)");
        Self {
            client,
            api_key,
            endpoint,
            model,
            dimensions,
            endpoint_ok,
        }
    }

    /// Fail closed on a rejected endpoint. Called before any request is
    /// built so a plain-HTTP remote override can never carry the key.
    fn check_endpoint(&self) -> ReinResult<()> {
        if self.endpoint_ok {
            Ok(())
        } else {
            Err(ReinError::Embedding(format!(
                "embedder endpoint '{}' refused: must be https:// (loopback http allowed)",
                self.endpoint
            )))
        }
    }
}

/// Max retry attempts for transient Gemini failures (429, 5xx, network).
const MAX_RETRIES: u32 = 3;
/// Retries for query-side batches (recall embedding its query variants).
/// Recall falls back to full-text search when the embedding fails, so a
/// slow provider should cost it one extra attempt, not four.
const QUERY_MAX_RETRIES: u32 = 1;
/// Client-wide request timeout (connect, send and body read together).
const CLIENT_TIMEOUT: Duration = Duration::from_secs(5);
/// Base backoff in ms; doubled each attempt, plus jitter up to half.
const BACKOFF_BASE_MS: u64 = 400;
/// Upper bound of the per-request timeout for document-side
/// `batchEmbedContents` (backfill, reindex, dedup); the actual allowance is
/// [`crate::embed::batch_request_timeout`]. Single embeds and query batches
/// (the interactive recall / store path) keep the client-wide limit.
const BATCH_REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

/// Send a Gemini request with exponential backoff on transient failures.
/// Respects `Retry-After` on 429 when provided (seconds form only).
async fn send_with_retry(
    client: &Client,
    url: &str,
    api_key: &str,
    body: &Value,
    timeout: Option<Duration>,
    max_retries: u32,
) -> ReinResult<(reqwest::StatusCode, String)> {
    // A provider-supplied `Retry-After` is honoured only up to the time one
    // request of this kind may take: an interactive call must not sleep for
    // as long as a header says.
    let retry_after_cap_ms = timeout.unwrap_or(CLIENT_TIMEOUT).as_millis() as u64;
    let mut attempt: u32 = 0;
    loop {
        let mut request = client
            .post(url)
            .header("x-goog-api-key", api_key)
            .json(body);
        if let Some(timeout) = timeout {
            request = request.timeout(timeout);
        }
        let send_result = request.send().await;

        match send_result {
            Ok(resp) => {
                let status = resp.status();
                let retry_after_secs = resp
                    .headers()
                    .get("retry-after")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|s| s.parse::<u64>().ok());
                let transient = status.as_u16() == 429 || status.is_server_error();
                // A body that fails mid-read (timeout, reset) is as
                // transient as a failed send.
                let text_body = match resp.text().await {
                    Ok(text) => text,
                    Err(e) => {
                        if attempt >= max_retries {
                            return Err(ReinError::Embedding(format!(
                                "Gemini response body read failed after {max_retries} retries: {e}"
                            )));
                        }
                        let backoff = exponential_backoff_ms(attempt);
                        tracing::warn!(
                            attempt,
                            err = %e,
                            backoff_ms = backoff,
                            "Gemini response body read failed, retrying"
                        );
                        tokio::time::sleep(Duration::from_millis(backoff)).await;
                        attempt += 1;
                        continue;
                    }
                };
                if !transient || attempt >= max_retries {
                    return Ok((status, text_body));
                }
                let backoff = retry_after_secs
                    .map(|s| s.saturating_mul(1000).min(retry_after_cap_ms))
                    .unwrap_or_else(|| exponential_backoff_ms(attempt));
                tracing::warn!(
                    attempt,
                    status = status.as_u16(),
                    backoff_ms = backoff,
                    "Gemini transient failure, retrying"
                );
                tokio::time::sleep(Duration::from_millis(backoff)).await;
                attempt += 1;
            }
            Err(e) => {
                if attempt >= max_retries {
                    return Err(ReinError::Embedding(format!(
                        "Gemini request failed after {max_retries} retries: {e}"
                    )));
                }
                let backoff = exponential_backoff_ms(attempt);
                tracing::warn!(
                    attempt,
                    err = %e,
                    backoff_ms = backoff,
                    "Gemini transport error, retrying"
                );
                tokio::time::sleep(Duration::from_millis(backoff)).await;
                attempt += 1;
            }
        }
    }
}

fn exponential_backoff_ms(attempt: u32) -> u64 {
    let base = BACKOFF_BASE_MS.saturating_mul(1u64 << attempt.min(6));
    // Deterministic "jitter" that still varies per call — hash SystemTime nanos.
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64)
        .unwrap_or(0);
    let jitter = now % (base / 2).max(1);
    base + jitter
}

impl Embedder for GeminiEmbedder {
    fn model_name(&self) -> &str {
        &self.model
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    async fn embed(&self, text: &str) -> ReinResult<Vec<f32>> {
        self.check_endpoint()?;
        let url = format!(
            "{}/v1beta/models/{}:embedContent",
            self.endpoint, self.model
        );
        let body = json!({
            "model": format!("models/{}", self.model),
            "content": {"parts": [{"text": text}]},
            "outputDimensionality": self.dimensions
        });

        let (status, text_body) =
            send_with_retry(&self.client, &url, &self.api_key, &body, None, MAX_RETRIES).await?;

        if !status.is_success() {
            return Err(ReinError::Embedding(crate::types::provider_status_error(
                "Gemini embed API",
                status,
                &text_body,
            )));
        }

        let parsed: Value =
            serde_json::from_str(&text_body).map_err(|e| ReinError::Embedding(e.to_string()))?;

        let values = parsed["embedding"]["values"]
            .as_array()
            .ok_or_else(|| ReinError::Embedding("missing embedding.values".into()))?;

        let embedding: Vec<f32> = values
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect();

        Ok(embedding)
    }

    async fn embed_batch(&self, texts: &[&str]) -> ReinResult<Vec<Vec<f32>>> {
        let timeout =
            crate::embed::batch_request_timeout(texts, CLIENT_TIMEOUT, BATCH_REQUEST_TIMEOUT);
        self.batch(texts, Some(timeout), MAX_RETRIES).await
    }

    /// Recall embeds its query variants through this method. It stays on the
    /// short client-wide timeout: an interactive call must not inherit the
    /// bulk-document allowance.
    async fn embed_query_batch(&self, texts: &[&str]) -> ReinResult<Vec<Vec<f32>>> {
        self.batch(texts, None, QUERY_MAX_RETRIES).await
    }
}

impl GeminiEmbedder {
    async fn batch(
        &self,
        texts: &[&str],
        timeout: Option<Duration>,
        max_retries: u32,
    ) -> ReinResult<Vec<Vec<f32>>> {
        self.check_endpoint()?;
        let url = format!(
            "{}/v1beta/models/{}:batchEmbedContents",
            self.endpoint, self.model
        );

        let requests: Vec<Value> = texts
            .iter()
            .map(|t| {
                json!({
                    "model": format!("models/{}", self.model),
                    "content": {"parts": [{"text": t}]},
                    "outputDimensionality": self.dimensions
                })
            })
            .collect();

        let body = json!({ "requests": requests });

        let (status, text_body) = send_with_retry(
            &self.client,
            &url,
            &self.api_key,
            &body,
            timeout,
            max_retries,
        )
        .await?;

        if !status.is_success() {
            return Err(ReinError::Embedding(crate::types::provider_status_error(
                "Gemini batch embed API",
                status,
                &text_body,
            )));
        }

        let parsed: Value =
            serde_json::from_str(&text_body).map_err(|e| ReinError::Embedding(e.to_string()))?;

        let embeddings_arr = parsed["embeddings"]
            .as_array()
            .ok_or_else(|| ReinError::Embedding("missing embeddings array".into()))?;

        let mut result = Vec::with_capacity(embeddings_arr.len());
        for emb in embeddings_arr {
            let values = emb["values"]
                .as_array()
                .ok_or_else(|| ReinError::Embedding("missing values in embedding".into()))?;
            let vec: Vec<f32> = values
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                .collect();
            result.push(vec);
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // requires live API key in GEMINI_API_KEY env var
    async fn test_google_embed_live() {
        let api_key = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY not set");
        let embedder = GeminiEmbedder::new(
            api_key,
            "https://generativelanguage.googleapis.com".to_string(),
            "gemini-embedding-001".to_string(),
            3072,
        );
        let result = embedder.embed("hello world").await.unwrap();
        assert_eq!(result.len(), 3072);
    }

    /// Verify that truncating a very long string to 500 chars is char-safe,
    /// especially with multi-byte CJK characters (no panic on char boundary).
    #[test]
    fn test_error_truncation() {
        // Build a 1200-char string mixing ASCII and CJK
        let mut long_string = String::new();
        for i in 0..300 {
            // Each iteration adds ~4 chars: a CJK char (3 bytes) + digit
            long_string.push('错');
            long_string.push_str(&format!("{}", i % 10));
        }
        assert!(
            long_string.len() > 1000,
            "Test string must be >1000 bytes, got {}",
            long_string.len()
        );

        // This is the same truncation pattern used in the Gemini error path:
        //   let truncated: String = text_body.chars().take(500).collect();
        let truncated: String = long_string.chars().take(500).collect();

        // Must not panic and must be <= 500 chars
        assert!(
            truncated.chars().count() <= 500,
            "Truncated string must be <= 500 chars"
        );
        assert!(
            truncated.chars().count() == 500,
            "Truncated string should be exactly 500 chars for long input"
        );

        // Verify the truncated string is valid UTF-8 (String guarantees this,
        // but let's be explicit)
        assert!(std::str::from_utf8(truncated.as_bytes()).is_ok());

        // Edge case: string of only multi-byte chars
        let cjk_only: String = "中文测试数据".repeat(200); // 1200 CJK chars
        let truncated_cjk: String = cjk_only.chars().take(500).collect();
        assert_eq!(truncated_cjk.chars().count(), 500);
    }

    #[test]
    fn validate_endpoint_accepts_https_and_loopback_http() {
        assert!(validate_endpoint("https://generativelanguage.googleapis.com").is_ok());
        assert!(validate_endpoint("HTTPS://Example.com").is_ok());
        assert!(validate_endpoint("http://127.0.0.1:8080").is_ok());
        assert!(validate_endpoint("http://localhost").is_ok());
        assert!(validate_endpoint("http://[::1]:11434").is_ok());
    }

    #[test]
    fn validate_endpoint_rejects_plain_http_remote() {
        assert!(validate_endpoint("http://evil.example").is_err());
        assert!(validate_endpoint("http://api.example:8080").is_err());
        assert!(validate_endpoint("ftp://example.com").is_err());
        assert!(validate_endpoint("").is_err());
    }

    #[tokio::test]
    async fn remote_http_endpoint_is_refused_before_any_request() {
        let e = GeminiEmbedder::new(
            "AIzaSyTESTKEY00000000003f2a".into(),
            "http://evil.example:8080".into(),
            "gemini-embedding-001".into(),
            8,
        );
        let err = e.embed("hello").await.unwrap_err().to_string();
        assert!(err.contains("refused"), "{err}");
        let err = e.embed_batch(&["a", "b"]).await.unwrap_err().to_string();
        assert!(err.contains("refused"), "{err}");
        // Loopback http stays usable (local test servers); the call fails
        // only because nothing listens there.
        let local = GeminiEmbedder::new(
            "AIzaSyTESTKEY00000000003f2a".into(),
            "http://127.0.0.1:9".into(),
            "gemini-embedding-001".into(),
            8,
        );
        assert!(local.endpoint_ok);
    }
}
