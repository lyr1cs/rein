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
        if path.starts_with("/v1/messages") {
            Some(Self::Anthropic)
        } else if path.starts_with("/v1/chat/completions") {
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

    /// Inject context into the system prompt.
    pub fn inject_context(&self, body: &mut serde_json::Value, context: &str) {
        match self {
            Self::Anthropic => super::anthropic::inject_context(body, context),
            Self::OpenAi => super::openai::inject_context(body, context),
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
