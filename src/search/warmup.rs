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

    // Populate HNSW index from all cached embeddings
    populate_hnsw(store, config);

    // Populate Tantivy FTS index
    populate_tantivy(store);

    (cached, errors)
}

/// Populate (or rebuild) the HNSW index from all cached embeddings in SQLite.
fn populate_hnsw(store: &SqliteStore, config: &ReinConfig) {
    let db_path = store.db_path();
    if db_path.to_str() == Some(":memory:") {
        return; // skip for in-memory test databases
    }
    let hnsw_path = db_path.with_extension("");
    let dims = config.embedding.dimensions;
    let model = config.embedding_model();

    let mut index = match crate::store::hnsw::HnswIndex::open(&hnsw_path, dims) {
        Ok(idx) => idx,
        Err(e) => {
            tracing::warn!("hnsw: failed to open index: {e}");
            return;
        }
    };

    // Get all memories and their cached embeddings
    let memories = match store.get_all_for_warmup() {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!("hnsw: failed to list memories: {e}");
            return;
        }
    };

    let mut inserted = 0usize;
    for (id, topic, summary, content) in &memories {
        let text = prepend_metadata(topic, summary, content);
        if let Ok(Some(emb)) = EmbedCache::get(store.conn(), &text, &model) {
            if emb.len() == dims {
                if index.insert(id, &emb).is_ok() {
                    inserted += 1;
                }
            }
        }
    }

    if inserted > 0 {
        if let Err(e) = index.save() {
            tracing::warn!("hnsw: failed to save index: {e}");
        } else {
            tracing::info!("hnsw: indexed {inserted} vectors");
        }
    }
}

/// Populate the Tantivy FTS index from all memories in SQLite.
fn populate_tantivy(store: &SqliteStore) {
    let db_path = store.db_path();
    if db_path.to_str() == Some(":memory:") {
        return;
    }
    let parent = db_path.parent().unwrap_or(std::path::Path::new("."));

    let tantivy = match crate::store::tantivy_fts::TantivyFts::open(parent) {
        Ok(t) => t,
        Err(e) => {
            tracing::warn!("tantivy: failed to open index: {e}");
            return;
        }
    };

    let memories = match store.get_all_for_warmup() {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!("tantivy: failed to list memories: {e}");
            return;
        }
    };

    let mut indexed = 0usize;
    for (id, topic, summary, content) in &memories {
        // Use summary as keywords placeholder (keywords aren't returned by get_all_for_warmup)
        if tantivy.insert(id, topic, summary, content, "").is_ok() {
            indexed += 1;
        }
    }

    if indexed > 0 {
        tracing::info!("tantivy: indexed {indexed} documents");
    }
}
