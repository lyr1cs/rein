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
        let client = Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap_or_default();
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
            return Err(ReinError::Embedding(format!(
                "OMLX API returned {}: {}",
                status, text_body
            )));
        }

        let parsed: serde_json::Value = serde_json::from_str(&text_body)
            .map_err(|e| ReinError::Embedding(e.to_string()))?;

        let values = parsed["data"][0]["embedding"]
            .as_array()
            .ok_or_else(|| ReinError::Embedding("missing data[0].embedding".into()))?;

        Ok(values
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect())
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
            return Err(ReinError::Embedding(format!(
                "OMLX batch API returned {}: {}",
                status, text_body
            )));
        }

        let parsed: serde_json::Value = serde_json::from_str(&text_body)
            .map_err(|e| ReinError::Embedding(e.to_string()))?;

        let data = parsed["data"]
            .as_array()
            .ok_or_else(|| ReinError::Embedding("missing data array".into()))?;

        let mut result = Vec::with_capacity(data.len());
        for item in data {
            let values = item["embedding"]
                .as_array()
                .ok_or_else(|| ReinError::Embedding("missing embedding in data item".into()))?;
            result.push(
                values
                    .iter()
                    .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                    .collect(),
            );
        }

        Ok(result)
    }
}
