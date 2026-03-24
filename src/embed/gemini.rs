use crate::types::error::{ReinError, ReinResult};
use crate::types::traits::Embedder;
use reqwest::Client;
use serde_json::{json, Value};
use std::time::Duration;

pub struct GeminiEmbedder {
    client: Client,
    api_key: String,
    model: String,
    dimensions: usize,
}

impl GeminiEmbedder {
    pub fn new(api_key: String, model: String, dimensions: usize) -> Self {
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
            "https://generativelanguage.googleapis.com/v1beta/models/{}:embedContent",
            self.model
        );
        let body = json!({
            "model": format!("models/{}", self.model),
            "content": {"parts": [{"text": text}]},
            "outputDimensionality": self.dimensions
        });

        let resp = self.client.post(&url)
            .header("x-goog-api-key", &self.api_key)
            .json(&body).send().await?;
        let status = resp.status();
        let text_body = resp.text().await?;

        if !status.is_success() {
            return Err(ReinError::Embedding(format!(
                "Gemini API returned {}: {}",
                status, text_body
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
            "https://generativelanguage.googleapis.com/v1beta/models/{}:batchEmbedContents",
            self.model
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

        let resp = self.client.post(&url)
            .header("x-goog-api-key", &self.api_key)
            .json(&body).send().await?;
        let status = resp.status();
        let text_body = resp.text().await?;

        if !status.is_success() {
            return Err(ReinError::Embedding(format!(
                "Gemini batch API returned {}: {}",
                status, text_body
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
        let embedder = GeminiEmbedder::new(api_key, "gemini-embedding-001".to_string(), 3072);
        let result = embedder.embed("hello world").await.unwrap();
        assert_eq!(result.len(), 3072);
    }
}
