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
}

impl GeminiEmbedder {
    pub fn new(api_key: String, endpoint: String, model: String, dimensions: usize) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .unwrap_or_else(|e| {
                tracing::warn!("failed to build reqwest client with timeout: {e}, using default");
                Client::default()
            });
        Self {
            client,
            api_key,
            endpoint,
            model,
            dimensions,
        }
    }
}

impl Embedder for GeminiEmbedder {
    fn model_name(&self) -> &str {
        &self.model
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    async fn embed(&self, text: &str) -> ReinResult<Vec<f32>> {
        let url = format!(
            "{}/v1beta/models/{}:embedContent",
            self.endpoint, self.model
        );
        let body = json!({
            "model": format!("models/{}", self.model),
            "content": {"parts": [{"text": text}]},
            "outputDimensionality": self.dimensions
        });

        let resp = self
            .client
            .post(&url)
            .header("x-goog-api-key", &self.api_key)
            .json(&body)
            .send()
            .await?;
        let status = resp.status();
        let text_body = resp.text().await?;

        if !status.is_success() {
            let truncated: String = text_body.chars().take(500).collect();
            return Err(ReinError::Embedding(format!(
                "Gemini API returned {}: {truncated}",
                status
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

        let resp = self
            .client
            .post(&url)
            .header("x-goog-api-key", &self.api_key)
            .json(&body)
            .send()
            .await?;
        let status = resp.status();
        let text_body = resp.text().await?;

        if !status.is_success() {
            let truncated: String = text_body.chars().take(500).collect();
            return Err(ReinError::Embedding(format!(
                "Gemini batch API returned {}: {truncated}",
                status
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
}
