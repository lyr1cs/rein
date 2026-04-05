//! Provider detection and dispatch.

use crate::config::ReinConfig;

/// Supported LLM API providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderKind {
    Anthropic,
    OpenAi,
    /// OpenAI Responses API (`/responses` → upstream `/v1/responses`).
    OpenAiResponses,
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
        } else if normalized == "/responses"
            || normalized.starts_with("/responses/")
            || normalized == "/v1/responses"
            || normalized.starts_with("/v1/responses/")
        {
            Some(Self::OpenAiResponses)
        } else {
            None
        }
    }

    /// Get the upstream base URL for this provider.
    pub fn upstream_url<'a>(&self, config: &'a ReinConfig) -> &'a str {
        match self {
            Self::Anthropic => &config.proxy.anthropic_upstream,
            Self::OpenAi | Self::OpenAiResponses => &config.proxy.openai_upstream,
        }
    }

    /// Rewrite the request path for upstream.
    ///
    /// The Codex SDK sends `/responses` but OpenAI expects `/v1/responses`.
    /// Other providers pass through unchanged.
    pub fn rewrite_path<'a>(&self, path: &'a str) -> std::borrow::Cow<'a, str> {
        match self {
            Self::OpenAiResponses => {
                if path.starts_with("/responses") {
                    std::borrow::Cow::Owned(format!("/v1{path}"))
                } else {
                    std::borrow::Cow::Borrowed(path)
                }
            }
            _ => std::borrow::Cow::Borrowed(path),
        }
    }

    /// Extract the user query from the request body.
    pub fn extract_query(&self, body: &serde_json::Value) -> String {
        match self {
            Self::Anthropic => super::anthropic::extract_query(body),
            Self::OpenAi => super::openai::extract_query(body),
            Self::OpenAiResponses => super::responses::extract_query(body),
        }
    }

    /// Extract assistant text from a non-streaming response body.
    pub fn extract_assistant_text_full(&self, body: &[u8]) -> Option<String> {
        match self {
            Self::Anthropic => super::anthropic::extract_assistant_text_full(body),
            Self::OpenAi => super::openai::extract_assistant_text_full(body),
            Self::OpenAiResponses => super::responses::extract_assistant_text_full(body),
        }
    }

    /// Extract assistant text fragment from an SSE chunk.
    pub fn extract_assistant_text_sse(&self, chunk: &[u8]) -> Option<String> {
        let text = std::str::from_utf8(chunk).ok()?;
        match self {
            Self::Anthropic => super::anthropic::extract_assistant_text_sse(text),
            Self::OpenAi => super::openai::extract_assistant_text_sse(text),
            Self::OpenAiResponses => super::responses::extract_assistant_text_sse(text),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ProviderKind;

    #[test]
    fn detect_only_exact_sampling_routes() {
        assert_eq!(
            ProviderKind::detect("/v1/messages"),
            Some(ProviderKind::Anthropic)
        );
        assert_eq!(ProviderKind::detect("/v1/messages/count_tokens"), None);
        assert_eq!(
            ProviderKind::detect("/v1/chat/completions"),
            Some(ProviderKind::OpenAi)
        );
        assert_eq!(ProviderKind::detect("/v1/models"), None);
        assert_eq!(
            ProviderKind::detect("/responses"),
            Some(ProviderKind::OpenAiResponses)
        );
        assert_eq!(
            ProviderKind::detect("/v1/responses"),
            Some(ProviderKind::OpenAiResponses)
        );
    }

    #[test]
    fn rewrite_path_responses_api() {
        let provider = ProviderKind::OpenAiResponses;
        assert_eq!(
            provider.rewrite_path("/responses").as_ref(),
            "/v1/responses"
        );
        assert_eq!(
            provider.rewrite_path("/responses?stream=true").as_ref(),
            "/v1/responses?stream=true"
        );
        // Already has /v1 prefix — no double rewrite
        assert_eq!(
            provider.rewrite_path("/v1/responses").as_ref(),
            "/v1/responses"
        );
    }

    #[test]
    fn rewrite_path_other_providers_noop() {
        assert_eq!(
            ProviderKind::Anthropic
                .rewrite_path("/v1/messages")
                .as_ref(),
            "/v1/messages"
        );
        assert_eq!(
            ProviderKind::OpenAi
                .rewrite_path("/v1/chat/completions")
                .as_ref(),
            "/v1/chat/completions"
        );
    }
}
