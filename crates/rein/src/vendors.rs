//! Vendor presets for OpenAI-compatible chat / embeddings endpoints.
//!
//! Every name in [`VENDOR_PRESETS`] resolves to `config::Provider::Omlx`
//! (the OpenAI-compatible code path). A preset only supplies defaults —
//! endpoint, API-key env var, auth header, request quirks, capability
//! bits — and every field can be overridden in the matching
//! `[section.openai]` (alias `[section.omlx]`) table. Nothing here does
//! I/O.
//!
//! Endpoint / env / capability values were checked against each vendor's
//! public API documentation on 2026-10-19.

/// Which request header carries the API key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthHeader {
    /// `Authorization: Bearer <key>` — OpenAI and nearly every compatible vendor.
    Bearer,
    /// `x-api-key: <key>` — Anthropic-style.
    XApiKey,
    /// `api-key: <key>` — Azure OpenAI.
    ApiKey,
}

impl AuthHeader {
    /// Parse the config spelling (`"bearer"` | `"x-api-key"` | `"api-key"`), case-insensitive.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "bearer" => Some(Self::Bearer),
            "x-api-key" => Some(Self::XApiKey),
            "api-key" => Some(Self::ApiKey),
            _ => None,
        }
    }

    pub fn config_name(self) -> &'static str {
        match self {
            Self::Bearer => "bearer",
            Self::XApiKey => "x-api-key",
            Self::ApiKey => "api-key",
        }
    }

    pub fn header_name(self) -> &'static str {
        match self {
            Self::Bearer => "authorization",
            Self::XApiKey => "x-api-key",
            Self::ApiKey => "api-key",
        }
    }

    pub fn header_value(self, key: &str) -> String {
        match self {
            Self::Bearer => format!("Bearer {key}"),
            Self::XApiKey | Self::ApiKey => key.to_string(),
        }
    }
}

/// How an embeddings endpoint encodes `data[i].embedding`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingEncoding {
    /// JSON array of numbers (OpenAI default).
    Float,
    /// Base64 string of signed int8 values (Perplexity `pplx-embed-*`).
    Base64Int8,
}

impl EmbeddingEncoding {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "float" => Some(Self::Float),
            "base64_int8" => Some(Self::Base64Int8),
            _ => None,
        }
    }

    pub fn config_name(self) -> &'static str {
        match self {
            Self::Float => "float",
            Self::Base64Int8 => "base64_int8",
        }
    }
}

/// Static defaults for one vendor name. Everything is overridable from config.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VendorPreset {
    pub name: &'static str,
    /// Base URL for `/chat/completions`; `None` = operator must set it (Azure) or
    /// the section's legacy localhost default applies (`omlx`).
    pub chat_endpoint: Option<&'static str>,
    /// Base URL for `/embeddings`; `None` = operator must set it.
    pub embeddings_endpoint: Option<&'static str>,
    /// Env var read when the table has no `api_key_env`; `None` = no auth header.
    pub api_key_env: Option<&'static str>,
    pub auth_header: AuthHeader,
    /// JSON field name used for query/document role on embeddings requests.
    pub input_type_field: &'static str,
    /// JSON field name for the requested vector size (`dimensions`; Voyage and
    /// Mistral use `output_dimension`).
    pub dimensions_field: &'static str,
    pub encoding: EmbeddingEncoding,
    /// Max texts per embeddings request; `embed_batch` chunks to this.
    pub max_batch: usize,
    /// Prepend `/no_think` to system prompts by default (Qwen3 local servers).
    pub disable_thinking: bool,
    pub chat: bool,
    pub embeddings: bool,
}

const fn preset(
    name: &'static str,
    chat_endpoint: Option<&'static str>,
    embeddings_endpoint: Option<&'static str>,
    api_key_env: Option<&'static str>,
    chat: bool,
    embeddings: bool,
) -> VendorPreset {
    VendorPreset {
        name,
        chat_endpoint,
        embeddings_endpoint,
        api_key_env,
        auth_header: AuthHeader::Bearer,
        input_type_field: "input_type",
        dimensions_field: "dimensions",
        encoding: EmbeddingEncoding::Float,
        max_batch: 256,
        disable_thinking: false,
        chat,
        embeddings,
    }
}

/// Keep this table sorted: hosted vendors first, local servers last.
pub const VENDOR_PRESETS: &[VendorPreset] = &[
    VendorPreset {
        max_batch: 2048,
        ..preset(
            "openai",
            Some("https://api.openai.com/v1"),
            Some("https://api.openai.com/v1"),
            Some("OPENAI_API_KEY"),
            true,
            true,
        )
    },
    // Embeddings live under /v1, chat (Orchestrator) under /router/v1; the
    // legacy Sonar chat path is no longer documented. Responses are base64 int8.
    VendorPreset {
        encoding: EmbeddingEncoding::Base64Int8,
        max_batch: 512,
        ..preset(
            "perplexity",
            Some("https://api.perplexity.ai/router/v1"),
            Some("https://api.perplexity.ai/v1"),
            Some("PERPLEXITY_API_KEY"),
            true,
            true,
        )
    },
    // DeepSeek documents the base URL without `/v1`; the client appends
    // `/chat/completions`, which is exactly the documented path.
    preset(
        "deepseek",
        Some("https://api.deepseek.com"),
        None,
        Some("DEEPSEEK_API_KEY"),
        true,
        false,
    ),
    preset(
        "openrouter",
        Some("https://openrouter.ai/api/v1"),
        Some("https://openrouter.ai/api/v1"),
        Some("OPENROUTER_API_KEY"),
        true,
        true,
    ),
    preset(
        "groq",
        Some("https://api.groq.com/openai/v1"),
        None,
        Some("GROQ_API_KEY"),
        true,
        false,
    ),
    VendorPreset {
        dimensions_field: "output_dimension",
        ..preset(
            "mistral",
            Some("https://api.mistral.ai/v1"),
            Some("https://api.mistral.ai/v1"),
            Some("MISTRAL_API_KEY"),
            true,
            true,
        )
    },
    preset(
        "xai",
        Some("https://api.x.ai/v1"),
        None,
        Some("XAI_API_KEY"),
        true,
        false,
    ),
    // Together's embeddings endpoint is no longer documented (2026-10-19);
    // chat only. Operators can still point `openai` + endpoint at it.
    preset(
        "together",
        Some("https://api.together.ai/v1"),
        None,
        Some("TOGETHER_API_KEY"),
        true,
        false,
    ),
    VendorPreset {
        max_batch: 2048,
        ..preset(
            "fireworks",
            Some("https://api.fireworks.ai/inference/v1"),
            Some("https://api.fireworks.ai/inference/v1"),
            Some("FIREWORKS_API_KEY"),
            true,
            true,
        )
    },
    VendorPreset {
        max_batch: 1000,
        dimensions_field: "output_dimension",
        ..preset(
            "voyage",
            None,
            Some("https://api.voyageai.com/v1"),
            Some("VOYAGE_API_KEY"),
            false,
            true,
        )
    },
    VendorPreset {
        input_type_field: "task",
        max_batch: 2048,
        ..preset(
            "jina",
            None,
            Some("https://api.jina.ai/v1"),
            Some("JINA_API_KEY"),
            false,
            true,
        )
    },
    // Cohere's compatibility layer accepts input/model/encoding_format only:
    // no input_type, no dimensions.
    preset(
        "cohere",
        Some("https://api.cohere.ai/compatibility/v1"),
        Some("https://api.cohere.ai/compatibility/v1"),
        Some("COHERE_API_KEY"),
        true,
        true,
    ),
    // OpenAI-compatible endpoint; `response_format` is ignored there.
    preset(
        "anthropic",
        Some("https://api.anthropic.com/v1"),
        None,
        Some("ANTHROPIC_API_KEY"),
        true,
        false,
    ),
    // Azure v1 API: endpoint = https://<res>.openai.azure.com/openai/v1,
    // `model` = deployment name, `api-version` optional. Key in `api-key`
    // header; Entra tokens work with `auth_header = "bearer"`.
    VendorPreset {
        auth_header: AuthHeader::ApiKey,
        ..preset(
            "azure",
            None,
            None,
            Some("AZURE_OPENAI_API_KEY"),
            true,
            true,
        )
    },
    // nv-embedqa models require input_type = query | passage; the hosted
    // endpoint has no `dimensions` parameter.
    preset(
        "nvidia",
        Some("https://integrate.api.nvidia.com/v1"),
        Some("https://integrate.api.nvidia.com/v1"),
        Some("NVIDIA_API_KEY"),
        true,
        true,
    ),
    preset(
        "cerebras",
        Some("https://api.cerebras.ai/v1"),
        None,
        Some("CEREBRAS_API_KEY"),
        true,
        false,
    ),
    // The HF router's OpenAI-compatible surface is chat-only; embeddings go
    // through the feature-extraction task (not OpenAI-shaped).
    preset(
        "huggingface",
        Some("https://router.huggingface.co/v1"),
        None,
        Some("HF_TOKEN"),
        true,
        false,
    ),
    // DashScope compatible mode: ≤ 10 texts per embeddings request (v3/v4).
    VendorPreset {
        max_batch: 10,
        ..preset(
            "dashscope",
            Some("https://dashscope.aliyuncs.com/compatible-mode/v1"),
            Some("https://dashscope.aliyuncs.com/compatible-mode/v1"),
            Some("DASHSCOPE_API_KEY"),
            true,
            true,
        )
    },
    VendorPreset {
        max_batch: 64,
        ..preset(
            "zhipu",
            Some("https://open.bigmodel.cn/api/paas/v4"),
            Some("https://open.bigmodel.cn/api/paas/v4"),
            Some("ZAI_API_KEY"),
            true,
            true,
        )
    },
    VendorPreset {
        max_batch: 32,
        ..preset(
            "siliconflow",
            Some("https://api.siliconflow.cn/v1"),
            Some("https://api.siliconflow.cn/v1"),
            Some("SILICONFLOW_API_KEY"),
            true,
            true,
        )
    },
    preset(
        "moonshot",
        Some("https://api.moonshot.cn/v1"),
        None,
        Some("MOONSHOT_API_KEY"),
        true,
        false,
    ),
    // Ark's OpenAI-compat guide excludes embedding models and the text
    // embedding models are being retired; chat only.
    preset(
        "volcengine",
        Some("https://ark.cn-beijing.volces.com/api/v3"),
        None,
        Some("ARK_API_KEY"),
        true,
        false,
    ),
    // ── local servers (no key) ───────────────────────────────────────────
    preset(
        "ollama",
        Some("http://localhost:11434/v1"),
        Some("http://localhost:11434/v1"),
        None,
        true,
        true,
    ),
    preset(
        "lmstudio",
        Some("http://localhost:1234/v1"),
        Some("http://localhost:1234/v1"),
        None,
        true,
        true,
    ),
    preset(
        "vllm",
        Some("http://localhost:8000/v1"),
        Some("http://localhost:8000/v1"),
        None,
        true,
        true,
    ),
    // `omlx` keeps the pre-v1.5 per-section localhost defaults (resolver
    // decides 11434 for extract, 8000 for the rest), so `chat_endpoint` is
    // None here on purpose.
    VendorPreset {
        disable_thinking: true,
        ..preset(
            "omlx",
            None,
            Some("http://localhost:8000/v1"),
            None,
            true,
            true,
        )
    },
];

/// Look up a preset by config name. Case-insensitive; `"local"` is the
/// legacy alias of `"omlx"`. Google / Gemini / none are *not* presets.
pub fn vendor_preset(name: &str) -> Option<&'static VendorPreset> {
    let lower = name.trim().to_ascii_lowercase();
    let lower = if lower == "local" {
        "omlx".to_string()
    } else {
        lower
    };
    VENDOR_PRESETS.iter().find(|p| p.name == lower)
}

/// Canonical lowercase name for a config string that selected the
/// OpenAI-compatible path (`"local"` → `"omlx"`).
pub fn canonical_vendor_name(name: &str) -> String {
    vendor_preset(name)
        .map(|p| p.name.to_string())
        .unwrap_or_else(|| name.trim().to_ascii_lowercase())
}

/// All preset names, for error messages and docs.
pub fn vendor_names() -> Vec<&'static str> {
    VENDOR_PRESETS.iter().map(|p| p.name).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_preset_name_is_lowercase_and_unique() {
        let mut seen = std::collections::HashSet::new();
        for p in VENDOR_PRESETS {
            assert_eq!(
                p.name,
                p.name.to_ascii_lowercase(),
                "{} not lowercase",
                p.name
            );
            assert!(seen.insert(p.name), "duplicate preset {}", p.name);
            assert!(p.chat || p.embeddings, "{} has no capability", p.name);
        }
    }

    #[test]
    fn presets_with_a_default_endpoint_use_https_or_loopback() {
        for p in VENDOR_PRESETS {
            for ep in [p.chat_endpoint, p.embeddings_endpoint]
                .into_iter()
                .flatten()
            {
                let ok = ep.starts_with("https://")
                    || ep.starts_with("http://localhost")
                    || ep.starts_with("http://127.0.0.1");
                assert!(ok, "{}: endpoint {ep} must be https or loopback", p.name);
                assert!(
                    !ep.ends_with('/'),
                    "{}: endpoint {ep} must not end with /",
                    p.name
                );
            }
        }
    }

    #[test]
    fn lookup_is_case_insensitive_and_maps_local_alias() {
        assert_eq!(vendor_preset("OpenAI").unwrap().name, "openai");
        assert_eq!(vendor_preset("local").unwrap().name, "omlx");
        assert!(vendor_preset("google").is_none());
        assert!(vendor_preset("gemini").is_none());
        assert!(vendor_preset("none").is_none());
        assert!(vendor_preset("").is_none());
        assert!(
            vendor_preset("nomic").is_none(),
            "Nomic's API is not OpenAI-shaped"
        );
    }

    #[test]
    fn capability_bits_match_the_spec_table() {
        let p = |n: &str| vendor_preset(n).unwrap();
        assert!(!p("deepseek").embeddings);
        assert!(!p("groq").embeddings);
        assert!(!p("xai").embeddings);
        assert!(!p("anthropic").embeddings);
        assert!(!p("cerebras").embeddings);
        assert!(!p("moonshot").embeddings);
        assert!(!p("together").embeddings);
        assert!(!p("huggingface").embeddings);
        assert!(!p("volcengine").embeddings);
        assert!(!p("voyage").chat);
        assert!(!p("jina").chat);
        assert!(p("perplexity").chat && p("perplexity").embeddings);
        assert!(p("openrouter").chat && p("openrouter").embeddings);
        assert!(p("omlx").chat && p("omlx").embeddings);
        assert_eq!(p("voyage").dimensions_field, "output_dimension");
        assert_eq!(p("mistral").dimensions_field, "output_dimension");
        assert_eq!(p("openai").dimensions_field, "dimensions");
        assert_eq!(p("jina").input_type_field, "task");
        assert_eq!(p("dashscope").max_batch, 10);
        assert_eq!(p("zhipu").api_key_env, Some("ZAI_API_KEY"));
        assert_eq!(
            p("deepseek").chat_endpoint,
            Some("https://api.deepseek.com")
        );
    }

    #[test]
    fn perplexity_has_split_endpoints_and_int8_encoding() {
        let p = vendor_preset("perplexity").unwrap();
        assert_eq!(p.chat_endpoint, Some("https://api.perplexity.ai/router/v1"));
        assert_eq!(p.embeddings_endpoint, Some("https://api.perplexity.ai/v1"));
        assert_eq!(p.encoding, EmbeddingEncoding::Base64Int8);
        assert_eq!(p.max_batch, 512);
        assert_eq!(p.api_key_env, Some("PERPLEXITY_API_KEY"));
    }

    #[test]
    fn only_omlx_disables_thinking_by_default() {
        for p in VENDOR_PRESETS {
            assert_eq!(p.disable_thinking, p.name == "omlx", "{}", p.name);
        }
    }

    #[test]
    fn azure_requires_explicit_endpoint_and_uses_api_key_header() {
        let p = vendor_preset("azure").unwrap();
        assert!(p.chat_endpoint.is_none() && p.embeddings_endpoint.is_none());
        assert_eq!(p.auth_header, AuthHeader::ApiKey);
    }

    #[test]
    fn auth_header_parse_and_render() {
        assert_eq!(AuthHeader::parse("Bearer"), Some(AuthHeader::Bearer));
        assert_eq!(AuthHeader::parse("x-api-key"), Some(AuthHeader::XApiKey));
        assert_eq!(AuthHeader::parse("api-key"), Some(AuthHeader::ApiKey));
        assert_eq!(AuthHeader::parse("basic"), None);
        assert_eq!(AuthHeader::Bearer.header_name(), "authorization");
        assert_eq!(AuthHeader::Bearer.header_value("k1"), "Bearer k1");
        assert_eq!(AuthHeader::XApiKey.header_name(), "x-api-key");
        assert_eq!(AuthHeader::XApiKey.header_value("k1"), "k1");
        assert_eq!(AuthHeader::ApiKey.header_name(), "api-key");
    }

    #[test]
    fn encoding_parse() {
        assert_eq!(
            EmbeddingEncoding::parse("float"),
            Some(EmbeddingEncoding::Float)
        );
        assert_eq!(
            EmbeddingEncoding::parse("BASE64_INT8"),
            Some(EmbeddingEncoding::Base64Int8)
        );
        assert_eq!(EmbeddingEncoding::parse("base64"), None);
    }
}
