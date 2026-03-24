//! Background embedding cache warmup.
//! Pre-computes embeddings for all memories that don't have cached vectors,
//! eliminating the 255ms Google API delay during recall.

use crate::config::ReinConfig;
use crate::embed::{EmbedCache, create_embedder, prepend_metadata};
use crate::store::SqliteStore;
use crate::types::Embedder as _;

/// Warm up the embedding cache by pre-computing embeddings for uncached memories.
/// Returns (cached_count, error_count).
pub async fn warmup(store: &SqliteStore, config: &ReinConfig) -> (usize, usize) {
    let embedder = match create_embedder(config) {
        Some(e) => e,
        None => {
            tracing::info!("no embedding provider configured, skipping warmup");
            return (0, 0);
        }
    };

    let model = config.embedding_model();

    // Get all memory IDs and their content
    let memories = match store.get_all_for_warmup() {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!("failed to list memories for warmup: {e}");
            return (0, 0);
        }
    };

    let total = memories.len();
    if total == 0 {
        return (0, 0);
    }

    // Filter out already-cached ones
    let uncached: Vec<(String, String, String, String)> = memories
        .into_iter()
        .filter(|(_, topic, summary, content)| {
            let text = prepend_metadata(topic, summary, content);
            EmbedCache::get(store.conn(), &text, &model)
                .ok()
                .flatten()
                .is_none()
        })
        .collect();

    if uncached.is_empty() {
        tracing::info!("warmup: all {total} memories already cached");
        return (0, 0);
    }

    tracing::info!("warmup: {}/{total} memories need embedding", uncached.len());

    let mut cached = 0usize;
    let mut errors = 0usize;

    // Process in batches of 100
    for chunk in uncached.chunks(100) {
        let texts: Vec<String> = chunk
            .iter()
            .map(|(_, topic, summary, content)| prepend_metadata(topic, summary, content))
            .collect();
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

        match embedder.embed_batch(&text_refs).await {
            Ok(embeddings) => {
                for (i, emb) in embeddings.iter().enumerate() {
                    if i < texts.len() {
                        if EmbedCache::put(store.conn(), &texts[i], &model, emb).is_ok() {
                            cached += 1;
                        } else {
                            errors += 1;
                        }
                    }
                }
            }
            Err(e) => {
                tracing::warn!("warmup batch failed: {e}");
                errors += chunk.len();
            }
        }
    }

    tracing::info!("warmup complete: {cached} cached, {errors} errors");
    (cached, errors)
}
