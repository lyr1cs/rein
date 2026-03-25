pub mod cache;
pub mod gemini;
pub mod omlx;

pub use cache::EmbedCache;
pub use gemini::GeminiEmbedder;
pub use omlx::OmlxEmbedder;

use crate::types::error::ReinResult;
use crate::types::traits::Embedder;

/// Prepend topic and summary metadata to text before embedding.
/// This improves retrieval by encoding context into the vector.
pub fn prepend_metadata(topic: &str, summary: &str, text: &str) -> String {
    format!("topic:{} | {} | {}", topic, summary, text)
}

/// Enum dispatch wrapper for all embedding backends.
/// Needed because async traits are not dyn-safe.
pub enum EmbedderKind {
    Gemini(GeminiEmbedder),
    Omlx(OmlxEmbedder),
}

impl Embedder for EmbedderKind {
    fn model_name(&self) -> &str {
        match self {
            Self::Gemini(e) => e.model_name(),
            Self::Omlx(e) => e.model_name(),
        }
    }

    fn dimensions(&self) -> usize {
        match self {
            Self::Gemini(e) => e.dimensions(),
            Self::Omlx(e) => e.dimensions(),
        }
    }

    async fn embed(&self, text: &str) -> ReinResult<Vec<f32>> {
        match self {
            Self::Gemini(e) => e.embed(text).await,
            Self::Omlx(e) => e.embed(text).await,
        }
    }

    async fn embed_batch(&self, texts: &[&str]) -> ReinResult<Vec<Vec<f32>>> {
        match self {
            Self::Gemini(e) => e.embed_batch(texts).await,
            Self::Omlx(e) => e.embed_batch(texts).await,
        }
    }
}

/// Create an embedder from config. Returns None if provider is "none" or API key is missing.
pub fn create_embedder(config: &crate::config::ReinConfig) -> Option<EmbedderKind> {
    use crate::config::Provider;
    match config.embedding_provider() {
        Provider::Google => {
            let api_key = config.embedding.google.api_key.as_ref()?;
            Some(EmbedderKind::Gemini(GeminiEmbedder::new(
                api_key.clone(),
                config.embedding.google.endpoint.clone(),
                config.embedding.google.model.clone(),
                config.embedding.dimensions,
            )))
        }
        Provider::Omlx => Some(EmbedderKind::Omlx(OmlxEmbedder::new(
            config.embedding.omlx.endpoint.clone(),
            config.embedding.omlx.model.clone(),
            config.embedding.dimensions,
        ))),
        Provider::None => None,
    }
}
