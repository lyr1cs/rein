//! Async memory extraction from assistant responses.

use crate::config::ReinConfig;
use crate::extract::hooks::parsing::looks_like_secret;
use crate::extract::llm::ExtractedMemory;
use crate::types::Importance;

/// Extract memories from assistant text and store them.
///
/// Runs asynchronously (spawned via `tokio::spawn`) — never blocks the response stream.
pub async fn extract_and_store(
    config: &ReinConfig,
    source_query: Option<String>,
    assistant_text: String,
) {
    // Skip very short responses (unlikely to contain useful memories).
    if assistant_text.len() < 100 {
        return;
    }

    if !super::policy::should_extract_response(
        config,
        source_query.as_deref(),
        &assistant_text,
    ) {
        return;
    }

    let extracted =
        crate::extract::llm::extract_with_fallback(config, &assistant_text, 3).await;

    let extracted = dedup_extracted_items(extracted);

    if extracted.is_empty() {
        return;
    }

    let store = match config.open_store() {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!("proxy extract: failed to open store: {e}");
            return;
        }
    };

    let mut stored = 0;
    for item in extracted {
        if !item.should_store {
            continue;
        }
        if looks_like_secret(&item.content) {
            continue;
        }

        let importance: Importance = item
            .importance
            .parse()
            .unwrap_or(Importance::Medium);

        let memory = crate::ops::build_memory(
            config,
            item.topic,
            item.content,
            importance,
            item.keywords,
            crate::types::Source::Proxy,
        );

        match crate::ops::store_memory(&store, config, memory) {
            Ok(_id) => stored += 1,
            Err(e) => tracing::warn!("proxy extract: store failed: {e}"),
        }
    }

    if stored > 0 {
        tracing::info!(stored, "proxy: extracted and stored memories from response");
    }
}

fn dedup_extracted_items(items: Vec<ExtractedMemory>) -> Vec<ExtractedMemory> {
    let mut unique: Vec<ExtractedMemory> = Vec::new();
    'outer: for item in items {
        for existing in &unique {
            let summary_sim = crate::extract::similarity(&item.summary, &existing.summary);
            let content_sim = crate::extract::similarity(&item.content, &existing.content);
            if item.topic == existing.topic && (summary_sim > 0.82 || content_sim > 0.82) {
                continue 'outer;
            }
        }
        unique.push(item);
    }
    unique
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedup_proxy_extracted_items() {
        let items = vec![
            ExtractedMemory {
                topic: "debug".into(),
                summary: "Fixed sqlite lock".into(),
                content: "Fixed sqlite lock by serializing writes.".into(),
                keywords: vec![],
                importance: "high".into(),
                should_store: true,
                quality_confidence: 0.8,
            },
            ExtractedMemory {
                topic: "debug".into(),
                summary: "Fixed sqlite locking".into(),
                content: "Fixed sqlite locking by serializing writes.".into(),
                keywords: vec![],
                importance: "high".into(),
                should_store: true,
                quality_confidence: 0.7,
            },
        ];
        assert_eq!(dedup_extracted_items(items).len(), 1);
    }
}
