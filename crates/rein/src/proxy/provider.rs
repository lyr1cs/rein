//! Provider detection and dispatch.

use crate::config::ReinConfig;

/// Supported LLM API providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderKind {
    Anthropic,
    OpenAi,
    /// OpenAI Responses API (`/responses` → upstream `/v1/responses`).
    OpenAiResponses,
    /// ChatGPT backend root (`/backend-api/*` except `/backend-api/codex/*`).
    ChatGptBackend,
    /// Codex subscription first-party backend (`/backend-api/codex/*`).
    CodexFirstParty,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordingMode {
    StructuredText,
    ArtifactMirrorOnly,
}

impl RecordingMode {
    pub fn captures_structured_text(self) -> bool {
        matches!(self, Self::StructuredText)
    }

    pub fn captures_artifact_mirror_only(self) -> bool {
        matches!(self, Self::ArtifactMirrorOnly)
    }
}

const CHATGPT_BACKEND_PREFIX: &str = "/backend-api";
const CODEX_FIRST_PARTY_PREFIX: &str = "/backend-api/codex";
/// Alternate path family Codex CLI uses when `chatgpt_base_url` does NOT contain
/// `/backend-api`. `PathStyle::from_base_url` in upstream Codex picks `/api/codex/`
/// in that case. rein accepts both to avoid routing breakage when users point
/// `chatgpt_base_url` at a bare `http://127.0.0.1:PORT` loopback.
const API_CODEX_PREFIX: &str = "/api/codex";

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
        } else if normalized == CODEX_FIRST_PARTY_PREFIX
            || normalized.starts_with("/backend-api/codex/")
            || normalized == API_CODEX_PREFIX
            || normalized.starts_with("/api/codex/")
        {
            Some(Self::CodexFirstParty)
        } else if normalized == CHATGPT_BACKEND_PREFIX || normalized.starts_with("/backend-api/") {
            Some(Self::ChatGptBackend)
        } else {
            None
        }
    }

    /// Get the upstream base URL for this provider.
    pub fn upstream_url<'a>(&self, config: &'a ReinConfig) -> &'a str {
        match self {
            Self::Anthropic => &config.proxy.anthropic_upstream,
            Self::OpenAi | Self::OpenAiResponses => &config.proxy.openai_upstream,
            Self::ChatGptBackend => &config.proxy.chatgpt_upstream,
            Self::CodexFirstParty => &config.proxy.codex_upstream,
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
            Self::CodexFirstParty => {
                if let Some(stripped) = path.strip_prefix(CODEX_FIRST_PARTY_PREFIX) {
                    std::borrow::Cow::Owned(stripped.to_string())
                } else if let Some(stripped) = path.strip_prefix(API_CODEX_PREFIX) {
                    // Alternate Codex path family when upstream base URL lacks /backend-api.
                    std::borrow::Cow::Owned(stripped.to_string())
                } else {
                    std::borrow::Cow::Borrowed(path)
                }
            }
            Self::ChatGptBackend => {
                if let Some(stripped) = path.strip_prefix(CHATGPT_BACKEND_PREFIX) {
                    std::borrow::Cow::Owned(stripped.to_string())
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
            Self::ChatGptBackend => String::new(),
            Self::OpenAiResponses | Self::CodexFirstParty => super::responses::extract_query(body),
        }
    }

    /// Extract assistant text from a non-streaming response body.
    pub fn extract_assistant_text_full(&self, body: &[u8]) -> Option<String> {
        match self {
            Self::Anthropic => super::anthropic::extract_assistant_text_full(body),
            Self::OpenAi => super::openai::extract_assistant_text_full(body),
            Self::ChatGptBackend => None,
            Self::OpenAiResponses | Self::CodexFirstParty => {
                super::responses::extract_assistant_text_full(body)
            }
        }
    }

    /// Extract assistant text fragment from an SSE chunk.
    pub fn extract_assistant_text_sse(&self, chunk: &[u8]) -> Option<String> {
        let text = std::str::from_utf8(chunk).ok()?;
        match self {
            Self::Anthropic => super::anthropic::extract_assistant_text_sse(text),
            Self::OpenAi => super::openai::extract_assistant_text_sse(text),
            Self::ChatGptBackend => None,
            Self::OpenAiResponses | Self::CodexFirstParty => {
                super::responses::extract_assistant_text_sse(text)
            }
        }
    }

    pub fn supports_websocket_passthrough(&self) -> bool {
        matches!(self, Self::OpenAiResponses | Self::CodexFirstParty)
    }

    pub fn recording_mode_for_path(&self, path: &str) -> RecordingMode {
        match self {
            Self::Anthropic | Self::OpenAi | Self::OpenAiResponses => RecordingMode::StructuredText,
            Self::ChatGptBackend => RecordingMode::ArtifactMirrorOnly,
            Self::CodexFirstParty => {
                let rewritten = self.rewrite_path(path);
                let normalized = rewritten.split('?').next().unwrap_or_default();
                if normalized == "/responses" {
                    RecordingMode::StructuredText
                } else {
                    RecordingMode::ArtifactMirrorOnly
                }
            }
        }
    }

    pub fn is_ambiguous_codex_first_party_path(path: &str) -> bool {
        let normalized = path.trim_end_matches('/');
        normalized == "/responses"
            || normalized.starts_with("/responses/")
            || normalized == "/models"
            || normalized.starts_with("/models/")
            || normalized == "/memories/trace_summarize"
    }

    pub fn is_ambiguous_chatgpt_backend_path(path: &str) -> bool {
        let normalized = path.trim_end_matches('/');
        normalized.starts_with("/wham/")
            || normalized.starts_with("/connectors/")
            || normalized == "/authenticate_app_v2"
            || normalized == "/v1/agent/register"
            || normalized == "/codex/safety/arc"
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
        assert_eq!(
            ProviderKind::detect("/backend-api/codex"),
            Some(ProviderKind::CodexFirstParty)
        );
        assert_eq!(
            ProviderKind::detect("/backend-api/codex/responses"),
            Some(ProviderKind::CodexFirstParty)
        );
        assert_eq!(
            ProviderKind::detect("/backend-api/codex/models"),
            Some(ProviderKind::CodexFirstParty)
        );
        assert_eq!(
            ProviderKind::detect("/backend-api/wham/usage"),
            Some(ProviderKind::ChatGptBackend)
        );
        assert_eq!(
            ProviderKind::detect("/backend-api/connectors/directory/list"),
            Some(ProviderKind::ChatGptBackend)
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

    #[test]
    fn rewrite_path_codex_first_party_strips_prefix() {
        let provider = ProviderKind::CodexFirstParty;
        assert_eq!(
            provider
                .rewrite_path("/backend-api/codex/responses")
                .as_ref(),
            "/responses"
        );
        assert_eq!(
            provider
                .rewrite_path("/backend-api/codex/models?client_version=1")
                .as_ref(),
            "/models?client_version=1"
        );
        assert_eq!(provider.rewrite_path("/backend-api/codex").as_ref(), "");
    }

    #[test]
    fn rewrite_path_chatgpt_backend_strips_prefix() {
        let provider = ProviderKind::ChatGptBackend;
        assert_eq!(
            provider.rewrite_path("/backend-api/wham/usage").as_ref(),
            "/wham/usage"
        );
        assert_eq!(
            provider
                .rewrite_path("/backend-api/connectors/directory/list?x=1")
                .as_ref(),
            "/connectors/directory/list?x=1"
        );
    }

    #[test]
    fn ambiguous_codex_first_party_paths_are_flagged() {
        assert!(ProviderKind::is_ambiguous_codex_first_party_path(
            "/responses"
        ));
        assert!(ProviderKind::is_ambiguous_codex_first_party_path(
            "/responses/compact"
        ));
        assert!(ProviderKind::is_ambiguous_codex_first_party_path("/models"));
        assert!(ProviderKind::is_ambiguous_codex_first_party_path(
            "/memories/trace_summarize"
        ));
        assert!(!ProviderKind::is_ambiguous_codex_first_party_path(
            "/v1/responses"
        ));
    }

    #[test]
    fn ambiguous_chatgpt_backend_paths_are_flagged() {
        assert!(ProviderKind::is_ambiguous_chatgpt_backend_path(
            "/wham/usage"
        ));
        assert!(ProviderKind::is_ambiguous_chatgpt_backend_path(
            "/connectors/directory/list"
        ));
        assert!(ProviderKind::is_ambiguous_chatgpt_backend_path(
            "/authenticate_app_v2"
        ));
        assert!(ProviderKind::is_ambiguous_chatgpt_backend_path(
            "/v1/agent/register"
        ));
        assert!(ProviderKind::is_ambiguous_chatgpt_backend_path(
            "/codex/safety/arc"
        ));
        assert!(!ProviderKind::is_ambiguous_chatgpt_backend_path(
            "/backend-api/codex/responses"
        ));
    }
}
