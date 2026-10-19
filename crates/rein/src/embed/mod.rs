pub mod cache;
pub mod gemini;
pub mod openai_compat;

pub use cache::EmbedCache;
pub use gemini::GeminiEmbedder;
pub use openai_compat::{OpenAiCompatibleEmbedder, OpenAiCompatibleOptions};

#[cfg(feature = "test-support")]
pub use mock::MockEmbedder;

#[cfg(feature = "test-support")]
pub mod mock;

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
    OpenAiCompatible(OpenAiCompatibleEmbedder),
    /// Test-only scripted embedder. Lets integration tests drive the
    /// real `run_vec_dedup` / HNSW / sqlite-vec pipelines end-to-end
    /// without a live API key. Gated behind `test-support` so it never
    /// ships in release binaries. See `embed/mock.rs` for the
    /// scripted-response contract.
    #[cfg(feature = "test-support")]
    Mock(MockEmbedder),
}

impl Embedder for EmbedderKind {
    fn model_name(&self) -> &str {
        match self {
            Self::Gemini(e) => e.model_name(),
            Self::OpenAiCompatible(e) => e.model_name(),
            #[cfg(feature = "test-support")]
            Self::Mock(e) => e.model_name(),
        }
    }

    fn dimensions(&self) -> usize {
        match self {
            Self::Gemini(e) => e.dimensions(),
            Self::OpenAiCompatible(e) => e.dimensions(),
            #[cfg(feature = "test-support")]
            Self::Mock(e) => e.dimensions(),
        }
    }

    async fn embed(&self, text: &str) -> ReinResult<Vec<f32>> {
        match self {
            Self::Gemini(e) => e.embed(text).await,
            Self::OpenAiCompatible(e) => e.embed(text).await,
            #[cfg(feature = "test-support")]
            Self::Mock(e) => e.embed(text).await,
        }
    }

    async fn embed_batch(&self, texts: &[&str]) -> ReinResult<Vec<Vec<f32>>> {
        match self {
            Self::Gemini(e) => e.embed_batch(texts).await,
            Self::OpenAiCompatible(e) => e.embed_batch(texts).await,
            #[cfg(feature = "test-support")]
            Self::Mock(e) => e.embed_batch(texts).await,
        }
    }

    fn asymmetric(&self) -> bool {
        match self {
            Self::Gemini(e) => e.asymmetric(),
            Self::OpenAiCompatible(e) => e.asymmetric(),
            #[cfg(feature = "test-support")]
            Self::Mock(e) => e.asymmetric(),
        }
    }

    async fn embed_query(&self, text: &str) -> ReinResult<Vec<f32>> {
        match self {
            Self::Gemini(e) => e.embed_query(text).await,
            Self::OpenAiCompatible(e) => e.embed_query(text).await,
            #[cfg(feature = "test-support")]
            Self::Mock(e) => e.embed_query(text).await,
        }
    }

    async fn embed_query_batch(&self, texts: &[&str]) -> ReinResult<Vec<Vec<f32>>> {
        match self {
            Self::Gemini(e) => e.embed_query_batch(texts).await,
            Self::OpenAiCompatible(e) => e.embed_query_batch(texts).await,
            #[cfg(feature = "test-support")]
            Self::Mock(e) => e.embed_query_batch(texts).await,
        }
    }
}

/// Create an embedder from config. Returns None if provider is "none" or API key is missing.
pub fn create_embedder(config: &crate::config::ReinConfig) -> Option<EmbedderKind> {
    use crate::config::Provider;
    match config.embedding_provider() {
        Provider::Google => {
            let api_key = config.embedding.google.api_key.as_ref()?;
            if let Err(msg) = gemini::validate_endpoint(&config.embedding.google.endpoint) {
                // Fail closed: no embedder at all (recall falls back to FTS)
                // rather than one that would post the key over plain HTTP.
                tracing::warn!("{msg}; embedding disabled");
                return None;
            }
            Some(EmbedderKind::Gemini(GeminiEmbedder::new(
                api_key.clone(),
                config.embedding.google.endpoint.clone(),
                config.embedding.google.model.clone(),
                config.embedding.dimensions,
            )))
        }
        Provider::Omlx => {
            let name = config.embedding_provider_name();
            let preset = crate::vendors::vendor_preset(&name);
            let table = &config.embedding.omlx;
            let endpoint = resolve_embedding_endpoint(&name, &table.endpoint, preset)?;
            let api_key_env = table
                .api_key_env
                .clone()
                .filter(|s| !s.trim().is_empty())
                .or_else(|| preset.and_then(|p| p.api_key_env).map(|s| s.to_string()));
            let api_key = api_key_env.as_deref().and_then(|n| config.secret(n));
            if api_key_env.is_some() && api_key.is_none() {
                // Same contract as Google: no key → embedding disabled (doctor /
                // providers report it; recall falls back to FTS).
                return None;
            }
            let auth_header = table
                .auth_header
                .as_deref()
                .and_then(crate::vendors::AuthHeader::parse)
                .or_else(|| preset.map(|p| p.auth_header))
                .unwrap_or(crate::vendors::AuthHeader::Bearer);
            let encoding = table
                .encoding
                .as_deref()
                .and_then(crate::vendors::EmbeddingEncoding::parse)
                .or_else(|| preset.map(|p| p.encoding))
                .unwrap_or(crate::vendors::EmbeddingEncoding::Float);
            Some(EmbedderKind::OpenAiCompatible(
                OpenAiCompatibleEmbedder::new(
                    endpoint,
                    table.model.clone(),
                    config.embedding.dimensions,
                    OpenAiCompatibleOptions {
                        api_key,
                        auth_header,
                        send_dimensions: table.send_dimensions,
                        query_input_type: table.query_input_type.clone(),
                        document_input_type: table.document_input_type.clone(),
                        input_type_field: preset
                            .map(|p| p.input_type_field.to_string())
                            .unwrap_or_else(|| "input_type".to_string()),
                        dimensions_field: preset
                            .map(|p| p.dimensions_field.to_string())
                            .unwrap_or_else(|| "dimensions".to_string()),
                        encoding,
                        max_batch: table
                            .max_batch
                            .or_else(|| preset.map(|p| p.max_batch))
                            .unwrap_or(256),
                    },
                ),
            ))
        }
        Provider::None => None,
    }
}

/// The compiled default (`http://localhost:8000/v1`) counts as "unset" for a
/// named vendor so `provider = "openai"` never posts to localhost by accident.
/// Returns `None` (embedding disabled) when a vendor needs an explicit
/// endpoint and none was given.
pub fn resolve_embedding_endpoint(
    name: &str,
    configured: &str,
    preset: Option<&crate::vendors::VendorPreset>,
) -> Option<String> {
    let is_compiled_default = configured == "http://localhost:8000/v1";
    match preset {
        Some(p) if p.name != "omlx" => {
            if !is_compiled_default && !configured.trim().is_empty() {
                Some(configured.trim_end_matches('/').to_string())
            } else if let Some(ep) = p.embeddings_endpoint {
                Some(ep.to_string())
            } else {
                tracing::warn!(
                    provider = name,
                    "embedding provider needs an explicit [embedding.openai].endpoint; embedding disabled"
                );
                None
            }
        }
        _ => Some(configured.trim_end_matches('/').to_string()),
    }
}

#[cfg(all(test, feature = "test-support"))]
mod tests {
    use super::*;

    #[tokio::test]
    async fn embed_query_defaults_to_embed() {
        let e = EmbedderKind::Mock(MockEmbedder::with_fixed_vector(3, vec![1.0, 2.0, 3.0]));
        assert!(!e.asymmetric());
        assert_eq!(e.embed_query("q").await.unwrap(), vec![1.0, 2.0, 3.0]);
        assert_eq!(e.embed_query_batch(&["a", "b"]).await.unwrap().len(), 2);
    }

    #[test]
    fn named_vendor_replaces_compiled_localhost_default() {
        let p = crate::vendors::vendor_preset("perplexity");
        assert_eq!(
            resolve_embedding_endpoint("perplexity", "http://localhost:8000/v1", p).as_deref(),
            Some("https://api.perplexity.ai/v1")
        );
        assert_eq!(
            resolve_embedding_endpoint("perplexity", "https://proxy.example/v1/", p).as_deref(),
            Some("https://proxy.example/v1")
        );
        let omlx = crate::vendors::vendor_preset("omlx");
        assert_eq!(
            resolve_embedding_endpoint("omlx", "http://localhost:8000/v1", omlx).as_deref(),
            Some("http://localhost:8000/v1")
        );
        assert!(resolve_embedding_endpoint(
            "azure",
            "http://localhost:8000/v1",
            crate::vendors::vendor_preset("azure")
        )
        .is_none());
    }

    #[test]
    fn create_embedder_refuses_remote_http_google_endpoint() {
        let base = "[embedding]\nprovider = \"google\"\ndimensions = 8\n[embedding.google]\nmodel = \"gemini-embedding-001\"\napi_key = \"AIzaSyTESTKEY00000000003f2a\"\n";
        let cfg = crate::config::ReinConfig::load_from_str(&format!(
            "{base}endpoint = \"http://evil.example:8080\"\n"
        ))
        .unwrap();
        assert!(create_embedder(&cfg).is_none());
        let cfg = crate::config::ReinConfig::load_from_str(&format!(
            "{base}endpoint = \"https://generativelanguage.googleapis.com\"\n"
        ))
        .unwrap();
        assert!(create_embedder(&cfg).is_some());
        let cfg = crate::config::ReinConfig::load_from_str(&format!(
            "{base}endpoint = \"http://127.0.0.1:8080\"\n"
        ))
        .unwrap();
        assert!(create_embedder(&cfg).is_some());
    }

    #[test]
    fn create_embedder_disabled_when_vendor_key_missing() {
        std::env::remove_var("OPENAI_API_KEY");
        let mut cfg = crate::config::ReinConfig::load_from_str(
            "[embedding]\nprovider = \"openai\"\ndimensions = 8\n[embedding.openai]\nmodel = \"text-embedding-3-small\"\n",
        )
        .unwrap();
        assert!(create_embedder(&cfg).is_none());
        cfg.secrets
            .insert("OPENAI_API_KEY".into(), "sk-live".into());
        let e = create_embedder(&cfg).expect("embedder with key from secrets file");
        assert_eq!(e.model_name(), "text-embedding-3-small");
        assert_eq!(cfg.embedding_model(), "openai:text-embedding-3-small");
    }
}
