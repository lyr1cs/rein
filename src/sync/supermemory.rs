use crate::types::{Importance, Memory, MemoryLayer, Source};

pub struct SupermemoryClient {
    client: reqwest::Client,
    api_key: String,
}

impl SupermemoryClient {
    pub fn new(api_key: String) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(5))
                .build()
                .unwrap_or_default(),
            api_key,
        }
    }

    /// Search Supermemory for relevant memories. Returns empty vec on failure.
    pub async fn search(&self, query: &str, limit: usize) -> Vec<Memory> {
        // Graceful degradation: never return an error
        match self.search_inner(query, limit).await {
            Ok(results) => results,
            Err(e) => {
                tracing::warn!("supermemory search failed: {e}");
                vec![]
            }
        }
    }

    async fn search_inner(&self, query: &str, limit: usize) -> anyhow::Result<Vec<Memory>> {
        // POST https://api.supermemory.ai/v3/search
        // Headers: Authorization: Bearer {api_key}
        // Body: { "query": "...", "limit": N }
        let resp = self
            .client
            .post("https://api.supermemory.ai/v3/search")
            .bearer_auth(&self.api_key)
            .json(&serde_json::json!({
                "q": query,
                "limit": limit,
            }))
            .send()
            .await?
            .error_for_status()?;

        let body: serde_json::Value = resp.json().await?;

        // Parse response: { "results": [ { "documentId", "title", "chunks": [{ "content", "score" }] } ] }
        let results = body
            .get("results")
            .and_then(|r| r.as_array())
            .cloned()
            .unwrap_or_default();

        let memories = results
            .into_iter()
            .filter_map(|item| {
                // Concatenate all chunk contents for this document
                let content = item.get("chunks")
                    .and_then(|c| c.as_array())
                    .map(|chunks| {
                        chunks.iter()
                            .filter_map(|c| c.get("content").and_then(|v| v.as_str()))
                            .collect::<Vec<_>>()
                            .join("\n\n")
                    })
                    .unwrap_or_default();

                if content.is_empty() {
                    return None;
                }

                let id = item
                    .get("documentId")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();

                let title = item
                    .get("title")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                Some(Memory {
                    id: format!("sm:{id}"),
                    layer: MemoryLayer::LTM,
                    topic: "supermemory".to_string(),
                    summary: if title.is_empty() { content.chars().take(100).collect() } else { title },
                    content,
                    keywords: vec![],
                    importance: Importance::Medium,
                    source: Source::Supermemory,
                    strength: 1.0,
                    decay_lambda: 0.0,
                    access_count: 0,
                    superseded_by: None,
                    related_ids: vec![],
                    embedding: None,
                    created_at: chrono::Utc::now(),
                    updated_at: chrono::Utc::now(),
                    last_accessed: chrono::Utc::now(),
                })
            })
            .collect();

        Ok(memories)
    }
}
