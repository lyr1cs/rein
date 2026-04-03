//! Async memory extraction from assistant responses.

use crate::config::ReinConfig;
use crate::extract::hooks::parsing::looks_like_secret;
use crate::types::Importance;

/// Extract memories from assistant text and store them.
///
/// Runs asynchronously (spawned via `tokio::spawn`) — never blocks the response stream.
pub async fn extract_and_store(config: &ReinConfig, assistant_text: String) {
    // Skip very short responses (unlikely to contain useful memories).
    if assistant_text.len() < 100 {
        return;
    }

    let extracted =
        crate::extract::llm::extract_with_fallback(config, &assistant_text, 3).await;

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
