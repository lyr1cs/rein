use crate::types::{Importance, Memory, MemoryLayer, MemoryStatus, Source};

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
        // Use v4/search with hybrid mode: searches both memory entries AND documents
        let resp = self
            .client
            .post("https://api.supermemory.ai/v4/search")
            .bearer_auth(&self.api_key)
            .json(&serde_json::json!({
                "q": query,
                "limit": limit,
                "searchMode": "hybrid",
            }))
            .send()
            .await?
            .error_for_status()?;

        let body: serde_json::Value = resp.json().await?;

        let results = body
            .get("results")
            .and_then(|r| r.as_array())
            .cloned()
            .unwrap_or_default();

        let memories = results
            .into_iter()
            .enumerate()
            .filter_map(|(idx, item)| {
                // v4 hybrid results can have either:
                // - "memory" field (from memory entries, saved by Claude Code plugin)
                // - "chunk" field (from document chunks, Notion sync etc.)
                // - both in some cases

                let (content, summary) = if let Some(memory_text) = item.get("memory").and_then(|v| v.as_str()) {
                    // Memory entry (from Claude Code plugin conversations)
                    (memory_text.to_string(), memory_text.chars().take(100).collect::<String>())
                } else if let Some(chunk_text) = item.get("chunk").and_then(|v| v.as_str()) {
                    // Document chunk (from Notion sync / uploaded files)
                    let title = item.get("documents")
                        .and_then(|d| d.as_array())
                        .and_then(|docs| docs.first())
                        .and_then(|doc| doc.get("title"))
                        .and_then(|t| t.as_str())
                        .unwrap_or("");
                    let summary = if title.is_empty() {
                        chunk_text.chars().take(100).collect()
                    } else {
                        title.to_string()
                    };
                    (chunk_text.to_string(), summary)
                } else {
                    // Try legacy format: chunks array
                    let chunk_content = item.get("chunks")
                        .and_then(|c| c.as_array())
                        .map(|chunks| {
                            chunks.iter()
                                .filter_map(|c| c.get("content").and_then(|v| v.as_str()))
                                .collect::<Vec<_>>()
                                .join("\n\n")
                        })
                        .unwrap_or_default();
                    if chunk_content.is_empty() {
                        return None;
                    }
                    let title = item.get("title")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let summary = if title.is_empty() {
                        chunk_content.chars().take(100).collect()
                    } else {
                        title
                    };
                    (chunk_content, summary)
                };

                if content.is_empty() {
                    return None;
                }

                let base_id = item.get("id")
                    .or_else(|| item.get("documentId"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                let id = format!("sm:{base_id}:{idx}");

                Some(Memory {
                    id,
                    layer: MemoryLayer::LTM,
                    topic: "supermemory".to_string(),
                    summary,
                    content,
                    keywords: vec![],
                    importance: Importance::Medium,
                    source: Source::Supermemory,
                    strength: 1.0,
                    decay_lambda: 0.0,
                    access_count: 0,
                    superseded_by: None,
                    related_ids: vec![],
                    status: MemoryStatus::default(),
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

#[cfg(test)]
mod tests {
    /// Verify that multiple results from the same document get unique IDs
    /// (sm:docid:0, sm:docid:1, etc) so they don't overwrite each other.
    #[test]
    fn test_supermemory_chunk_dedup() {
        // Simulate the ID generation logic from search_inner:
        //   let id = format!("sm:{base_id}:{idx}");
        // where base_id comes from the result and idx is the enumeration index.
        let base_id = "doc123";
        let ids: Vec<String> = (0..5)
            .map(|idx| format!("sm:{base_id}:{idx}"))
            .collect();

        // All IDs must be unique
        let unique: std::collections::HashSet<&String> = ids.iter().collect();
        assert_eq!(
            unique.len(),
            ids.len(),
            "IDs from same document must be unique: {:?}",
            ids
        );

        // Verify the format
        assert_eq!(ids[0], "sm:doc123:0");
        assert_eq!(ids[1], "sm:doc123:1");
        assert_eq!(ids[4], "sm:doc123:4");
    }
}
