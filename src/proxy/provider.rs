//! Provider detection and dispatch.

use crate::config::ReinConfig;

/// Supported LLM API providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderKind {
    Anthropic,
    OpenAi,
    // Future: Gemini
}

impl ProviderKind {
    /// Detect provider from the request path.
    pub fn detect(path: &str) -> Option<Self> {
        let normalized = path.trim_end_matches('/');
        if normalized == "/v1/messages" {
            Some(Self::Anthropic)
        } else if normalized == "/v1/chat/completions" {
            Some(Self::OpenAi)
        } else {
            None
        }
    }

    /// Get the upstream base URL for this provider.
    pub fn upstream_url<'a>(&self, config: &'a ReinConfig) -> &'a str {
        match self {
            Self::Anthropic => &config.proxy.anthropic_upstream,
            Self::OpenAi => &config.proxy.openai_upstream,
        }
    }

    /// Extract the user query from the request body.
    pub fn extract_query(&self, body: &serde_json::Value) -> String {
        match self {
            Self::Anthropic => super::anthropic::extract_query(body),
            Self::OpenAi => super::openai::extract_query(body),
        }
    }

    /// Extract assistant text from a non-streaming response body.
    pub fn extract_assistant_text_full(&self, body: &[u8]) -> Option<String> {
        match self {
            Self::Anthropic => super::anthropic::extract_assistant_text_full(body),
            Self::OpenAi => super::openai::extract_assistant_text_full(body),
        }
    }

    /// Extract assistant text fragment from an SSE chunk.
    pub fn extract_assistant_text_sse(&self, chunk: &[u8]) -> Option<String> {
        let text = std::str::from_utf8(chunk).ok()?;
        match self {
            Self::Anthropic => super::anthropic::extract_assistant_text_sse(text),
            Self::OpenAi => super::openai::extract_assistant_text_sse(text),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ProviderKind;

    #[test]
    fn detect_only_exact_sampling_routes() {
        assert_eq!(ProviderKind::detect("/v1/messages"), Some(ProviderKind::Anthropic));
        assert_eq!(ProviderKind::detect("/v1/messages/count_tokens"), None);
        assert_eq!(ProviderKind::detect("/v1/chat/completions"), Some(ProviderKind::OpenAi));
        assert_eq!(ProviderKind::detect("/v1/models"), None);
    }
}
