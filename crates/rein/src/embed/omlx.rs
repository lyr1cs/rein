use crate::types::error::{ReinError, ReinResult};
use crate::types::traits::Embedder;
use reqwest::Client;
use serde_json::json;
use std::time::Duration;

/// OMLX local embedding backend (OpenAI-compatible API).
pub struct OmlxEmbedder {
    client: Client,
    endpoint: String,
    model: String,
    dimensions: usize,
}

impl OmlxEmbedder {
    pub fn new(endpoint: String, model: String, dimensions: usize) -> Self {
        // Builder failure returning `Client::default()` silently drops the
        // 10-second timeout — a hung OMLX endpoint would then stall recall.
        // `.expect` is correct: builder() only fails on a broken TLS backend
        // and that's a boot-time problem we want surfaced, not swallowed.
        let client = Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .expect("reqwest client build failed for omlx (likely TLS backend)");
        Self {
            client,
            endpoint,
            model,
            dimensions,
        }
    }
}

impl Embedder for OmlxEmbedder {
    fn model_name(&self) -> &str {
        &self.model
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    async fn embed(&self, text: &str) -> ReinResult<Vec<f32>> {
        let url = format!("{}/embeddings", self.endpoint);
        let body = json!({
            "model": &self.model,
            "input": text,
        });

        let resp = self.client.post(&url).json(&body).send().await?;
        let status = resp.status();
        let text_body = resp.text().await?;

        if !status.is_success() {
            let truncated: String = text_body.chars().take(500).collect();
            return Err(ReinError::Embedding(format!(
                "OMLX API returned {}: {truncated}",
                status
            )));
        }

        let parsed: serde_json::Value =
            serde_json::from_str(&text_body).map_err(|e| ReinError::Embedding(e.to_string()))?;

        let values = parsed["data"][0]["embedding"]
            .as_array()
            .ok_or_else(|| ReinError::Embedding("missing data[0].embedding".into()))?;

        let embedding: Vec<f32> = values
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect();

        if embedding.len() != self.dimensions {
            return Err(ReinError::Embedding(format!(
                "dimension mismatch: expected {}, got {}",
                self.dimensions,
                embedding.len()
            )));
        }

        Ok(embedding)
    }

    async fn embed_batch(&self, texts: &[&str]) -> ReinResult<Vec<Vec<f32>>> {
        let url = format!("{}/embeddings", self.endpoint);
        let body = json!({
            "model": &self.model,
            "input": texts,
        });

        let resp = self.client.post(&url).json(&body).send().await?;
        let status = resp.status();
        let text_body = resp.text().await?;

        if !status.is_success() {
            let truncated: String = text_body.chars().take(500).collect();
            return Err(ReinError::Embedding(format!(
                "OMLX batch API returned {}: {truncated}",
                status
            )));
        }

        let parsed: serde_json::Value =
            serde_json::from_str(&text_body).map_err(|e| ReinError::Embedding(e.to_string()))?;

        let data = parsed["data"]
            .as_array()
            .ok_or_else(|| ReinError::Embedding("missing data array".into()))?;

        let mut result = Vec::with_capacity(data.len());
        for item in data {
            let values = item["embedding"]
                .as_array()
                .ok_or_else(|| ReinError::Embedding("missing embedding in data item".into()))?;
            let embedding: Vec<f32> = values
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                .collect();

            if embedding.len() != self.dimensions {
                return Err(ReinError::Embedding(format!(
                    "dimension mismatch: expected {}, got {}",
                    self.dimensions,
                    embedding.len()
                )));
            }

            result.push(embedding);
        }

        Ok(result)
    }
}
