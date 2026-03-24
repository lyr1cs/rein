pub mod cache;
pub mod gemini;

pub use cache::EmbedCache;
pub use gemini::GeminiEmbedder;

/// Prepend topic and summary metadata to text before embedding.
/// This improves retrieval by encoding context into the vector.
pub fn prepend_metadata(topic: &str, summary: &str, text: &str) -> String {
    format!("topic:{} | {} | {}", topic, summary, text)
}

/// Create an embedder from config. Returns None if provider is "none" or API key is missing.
///
/// Currently returns GeminiEmbedder. When adding new providers, either:
/// - Add new variants here returning the concrete type, or
/// - Use an enum wrapper that implements Embedder for all variants.
pub fn create_embedder(config: &crate::config::ReinConfig) -> Option<GeminiEmbedder> {
    match config.embedding.provider.as_str() {
        "google" => {
            let api_key = config.embedding.google.api_key.as_ref()?;
            Some(GeminiEmbedder::new(
                api_key.clone(),
                config.embedding.google.model.clone(),
                config.embedding.dimensions,
            ))
        }
        // Future providers: add new match arms returning concrete types,
        // or refactor to an EmbedderKind enum wrapping all providers.
        "none" => None,
        other => {
            tracing::warn!("unknown embedding provider: {other}, falling back to none");
            None
        }
    }
}
