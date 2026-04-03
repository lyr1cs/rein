//! Async memory extraction from assistant responses.

use crate::config::ReinConfig;
use crate::extract::hooks::parsing::runtime_agent_label;
use crate::extract::hooks::queue::{queue_memory_job, spawn_memory_worker, MemoryJobMode};

/// Queue assistant text for background extraction and storage.
///
/// Runs asynchronously (spawned via `tokio::spawn`) but only appends a durable
/// queue record on the hot path. Actual extraction/storage happens in the
/// background worker.
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

    if let Err(e) = queue_memory_job(
        config,
        MemoryJobMode::Quick,
        "proxy_response",
        "source:main-agent",
        runtime_agent_label(),
        false,
        20,
        source_query,
        assistant_text,
    ) {
        tracing::warn!("proxy extract: failed to queue memory job: {e}");
        return;
    }
    spawn_memory_worker(config);
}
