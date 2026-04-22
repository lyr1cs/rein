use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, Weak};

/// Typed provider enum for compile-time safety.
/// Parsed from String in config, prevents typo-driven misconfiguration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    Google,
    Omlx,
    None,
}

impl Provider {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "google" => Provider::Google,
            "omlx" => Provider::Omlx,
            "none" => Provider::None,
            other => {
                tracing::warn!("unknown provider '{other}', falling back to None");
                Provider::None
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Config structs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
#[derive(Default)]
pub struct ReinConfig {
    pub database: DatabaseConfig,
    pub embedding: EmbeddingConfig,
    pub search: SearchConfig,
    pub chunking: ChunkingConfig,
    pub sync: SyncConfig,
    pub decay: DecayConfig,
    pub server: ServerConfig,
    #[serde(default)]
    pub hooks: HooksConfig,
    #[serde(default)]
    pub extract: ExtractConfig,
    #[serde(default)]
    pub adaptive: AdaptiveConfig,
    #[serde(default)]
    pub query_expansion: QueryExpansionConfig,
    #[serde(default)]
    pub proxy: ProxyConfig,
    #[serde(default)]
    pub async_memory: AsyncMemoryConfig,
    #[serde(default)]
    pub cleanup: CleanupConfig,
    #[serde(default)]
    pub intelligent_merge: IntelligentMergeConfig,
}

/// Config for the LLM-driven intelligent-merge classifier (opt-in).
///
/// When `enabled = true`, store_with_dedup consults an LLM on gray-zone
/// similarity cases and chooses among ignore / update / merge / create_new
/// instead of the mechanical jaccard/containment threshold.
///
/// `provider` is optional override — when "none" (default) the classifier
/// falls back to the provider configured under `[query_expansion]`. Set
/// to "google" or "omlx" to use an independent provider configured in the
/// nested blocks below.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IntelligentMergeConfig {
    #[serde(default)]
    pub enabled: bool,
    /// "google" | "omlx" | "none" — default "none" means reuse query_expansion.
    #[serde(default = "default_im_provider")]
    pub provider: String,
    #[serde(default)]
    pub google: GoogleExpandConfig,
    #[serde(default)]
    pub omlx: OmlxExpandConfig,
}

fn default_im_provider() -> String {
    "none".to_string()
}

impl Default for IntelligentMergeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            provider: default_im_provider(),
            google: GoogleExpandConfig::default(),
            omlx: OmlxExpandConfig::default(),
        }
    }
}

impl IntelligentMergeConfig {
    /// Returns the resolved provider for the classifier. "none" means
    /// "fall back to query_expansion" — callers should consult that when
    /// this returns Provider::None.
    pub fn resolved_provider(&self) -> Provider {
        match self.provider.to_lowercase().as_str() {
            "google" | "gemini" => Provider::Google,
            "omlx" | "local" => Provider::Omlx,
            _ => Provider::None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatabaseConfig {
    pub path: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingConfig {
    pub provider: String,
    pub dimensions: usize,
    pub google: GoogleEmbeddingConfig,
    pub omlx: OmlxEmbeddingConfig,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoogleEmbeddingConfig {
    pub model: String,
    #[serde(default)]
    pub api_key: Option<String>,
    /// API endpoint override (for proxies in China, etc.)
    /// Default: "https://generativelanguage.googleapis.com"
    #[serde(default = "default_google_endpoint")]
    pub endpoint: String,
}

fn default_google_endpoint() -> String {
    "https://generativelanguage.googleapis.com".to_string()
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OmlxEmbeddingConfig {
    pub endpoint: String,
    #[serde(default = "default_omlx_model")]
    pub model: String,
}

fn default_omlx_model() -> String {
    "default".to_string()
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchConfig {
    pub rrf_k: f64,
    pub rrf_fts_weight: f64,
    pub rrf_vec_weight: f64,
    /// Fusion method: "rrf" (Reciprocal Rank Fusion) or "cc" (Convex Combination).
    /// CC normalizes scores to [0,1] and blends with alpha; often more accurate (Bruch 2023).
    #[serde(default = "default_fusion_method")]
    pub fusion_method: String,
    /// Alpha for CC fusion: score = alpha * sparse + (1-alpha) * dense. Default 0.5.
    #[serde(default = "default_cc_alpha")]
    pub cc_alpha: f64,
    pub dedup_similarity: f64,
    pub dedup_time_window_days: i64,
    /// LLM reranker provider: "google", "omlx", or "none". Default: "none".
    #[serde(default = "default_llm_reranker")]
    pub llm_reranker: String,
    /// Number of top candidates to send to LLM reranker. Default: 15.
    #[serde(default = "default_llm_reranker_top_n")]
    pub llm_reranker_top_n: usize,
    /// Max ms to wait for background LLM reranker before returning linear scores.
    /// The reranker runs concurrently with cross-validation; this budget starts from the
    /// beginning of the recall pipeline. Default: 1500ms (0 = synchronous legacy mode).
    #[serde(default = "default_llm_reranker_timeout_ms")]
    pub llm_reranker_timeout_ms: u64,
    /// MMR lambda: relevance-diversity tradeoff. 1.0 = off (pure relevance), 0.3 = strong diversity.
    #[serde(default = "default_mmr_lambda")]
    pub mmr_lambda: f64,
    /// Strong-BM25-signal detection: top1 / top2 >= this ratio bypasses LLM rerank + expansion.
    #[serde(default = "default_strong_signal_ratio")]
    pub strong_signal_ratio: f32,
    /// Single-positive-result strong signal: only result with BM25 score >= this value.
    #[serde(default = "default_strong_signal_single")]
    pub strong_signal_single: f32,
}

fn default_fusion_method() -> String {
    "rrf".to_string()
}
fn default_cc_alpha() -> f64 {
    0.5
}
fn default_llm_reranker() -> String {
    "none".to_string()
}
fn default_llm_reranker_top_n() -> usize {
    15
}
fn default_llm_reranker_timeout_ms() -> u64 {
    1500
}
fn default_mmr_lambda() -> f64 {
    1.0
}
fn default_strong_signal_ratio() -> f32 {
    1.5
}
fn default_strong_signal_single() -> f32 {
    3.0
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChunkingConfig {
    pub max_tokens: usize,
    pub overlap_percent: usize,
    pub metadata_prefix: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SyncConfig {
    pub supermemory_enabled: bool,
    pub auto_memory_enabled: bool,
    pub auto_memory_glob: String,
    #[serde(default)]
    pub api_key: Option<String>,
    /// Supermemory API endpoint override
    #[serde(default = "default_supermemory_endpoint")]
    pub endpoint: String,
}

fn default_supermemory_endpoint() -> String {
    "https://api.supermemory.ai".to_string()
}

/// Configuration for the adaptive engine (M1-M5).
/// All parameters here are operational settings, not model parameters —
/// model parameters are learned from data.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdaptiveConfig {
    /// Enable the adaptive engine. Default true.
    #[serde(default = "default_adaptive_enabled")]
    pub enabled: bool,
    /// Minimum accessed events per bucket before learning alpha. Default 10.
    #[serde(default = "default_min_samples_alpha")]
    pub min_samples_alpha: usize,
    /// Minimum cluster sample count for survival curve. Default 20.
    #[serde(default = "default_survival_cold_start")]
    pub survival_cold_start: usize,
    /// Minimum memories before enabling tiering. Default 100.
    #[serde(default = "default_tier_cold_start")]
    pub tier_cold_start: usize,
    /// Event retention in days. Default 90.
    #[serde(default = "default_event_retention_days")]
    pub event_retention_days: u64,
    /// Max alpha change per GC cycle. Default 0.15.
    #[serde(default = "default_alpha_max_step")]
    pub alpha_max_step: f64,
    /// Bayesian shrinkage prior strength. Default 5.0.
    #[serde(default = "default_shrinkage_prior")]
    pub shrinkage_prior: f64,
    /// Time-to-live for in-memory AdaptiveState caches, in seconds. Default 300 (5 min).
    /// A cached snapshot older than this is considered stale and triggers a refresh
    /// from the metadata table on the next read.
    #[serde(default = "default_cache_ttl_secs")]
    pub cache_ttl_secs: u64,
}

fn default_adaptive_enabled() -> bool {
    true
}
fn default_min_samples_alpha() -> usize {
    10
}
fn default_survival_cold_start() -> usize {
    20
}
fn default_tier_cold_start() -> usize {
    100
}
fn default_event_retention_days() -> u64 {
    90
}
fn default_alpha_max_step() -> f64 {
    0.15
}
fn default_shrinkage_prior() -> f64 {
    5.0
}
fn default_cache_ttl_secs() -> u64 {
    300
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            enabled: default_adaptive_enabled(),
            min_samples_alpha: default_min_samples_alpha(),
            survival_cold_start: default_survival_cold_start(),
            tier_cold_start: default_tier_cold_start(),
            event_retention_days: default_event_retention_days(),
            alpha_max_step: default_alpha_max_step(),
            shrinkage_prior: default_shrinkage_prior(),
            cache_ttl_secs: default_cache_ttl_secs(),
        }
    }
}

// ---------------------------------------------------------------------------
// Cleanup / consolidation / dedup config
// ---------------------------------------------------------------------------

/// Configuration for cleanup, consolidation, and dedup operations.
/// Controls LLM usage and thresholds during `rein cleanup` and `rein dedup`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(default)]
pub struct CleanupConfig {
    /// Skip LLM consolidation for single-memory topic groups. Default: true.
    #[serde(default = "default_true")]
    pub skip_single_memory: bool,
    /// Skip LLM consolidation for groups with total content under this char count. Default: 500.
    #[serde(default = "default_skip_short_content_chars")]
    pub skip_short_content_chars: usize,
    /// Batch size for LLM calls during consolidation. Default: 8.
    #[serde(default = "default_cleanup_llm_batch_size")]
    pub llm_batch_size: usize,
    /// Max LLM calls per dedup run (gray zone verdicts). Default: 8.
    #[serde(default = "default_cleanup_llm_budget")]
    pub llm_budget: usize,
    /// Embedding-based dedup: above this cosine similarity, merge directly without LLM. Default: 0.80.
    #[serde(default = "default_vec_dedup_strong_threshold")]
    pub vec_dedup_strong_threshold: f64,
    /// Embedding-based dedup: below this cosine similarity, ignore. Default: 0.70.
    #[serde(default = "default_vec_dedup_weak_threshold")]
    pub vec_dedup_weak_threshold: f64,
    /// Delay (ms) between LLM consolidation batches. Default: 200.
    #[serde(default = "default_inter_batch_delay_ms")]
    pub inter_batch_delay_ms: u64,
}

fn default_skip_short_content_chars() -> usize {
    500
}
fn default_cleanup_llm_batch_size() -> usize {
    8
}
fn default_cleanup_llm_budget() -> usize {
    8
}
fn default_vec_dedup_strong_threshold() -> f64 {
    0.80
}
fn default_vec_dedup_weak_threshold() -> f64 {
    0.70
}
fn default_inter_batch_delay_ms() -> u64 {
    200
}

impl Default for CleanupConfig {
    fn default() -> Self {
        Self {
            skip_single_memory: true,
            skip_short_content_chars: default_skip_short_content_chars(),
            llm_batch_size: default_cleanup_llm_batch_size(),
            llm_budget: default_cleanup_llm_budget(),
            vec_dedup_strong_threshold: default_vec_dedup_strong_threshold(),
            vec_dedup_weak_threshold: default_vec_dedup_weak_threshold(),
            inter_batch_delay_ms: default_inter_batch_delay_ms(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecayConfig {
    pub base_lambda: f64,
    pub ltm_beta: f64,
    pub stm_beta: f64,
    pub interval_hours: u64,
    pub prune_threshold: f64,
    pub stm_to_ltm_access_count: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    pub compact: bool,
    pub sse_enabled: bool,
    pub sse_port: u16,
    pub sse_bind: String,
    #[serde(default)]
    pub gui_enabled: bool,
    #[serde(default)]
    pub allow_unauthenticated_loopback: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HooksConfig {
    pub min_turns: usize,
    pub context_before: usize,
    pub context_after: usize,
    pub max_items_per_session: usize,
    #[serde(default = "default_signal_keywords")]
    pub signal_keywords: Vec<String>,
    #[serde(default = "default_buffer_dir")]
    pub buffer_dir: String,
    /// Buffer size (in characters) that triggers a mid-session LLM extraction.
    /// When accumulated buffer content exceeds this threshold, hook_post triggers
    /// an incremental extraction and clears the buffer.
    /// Higher = less frequent extraction (good for 1M+ context models).
    /// Lower = more frequent extraction (good for smaller context models).
    /// 0 = never trigger mid-session extraction (only at session end).
    #[serde(default = "default_buffer_flush_threshold")]
    pub buffer_flush_threshold: usize,
}

fn default_buffer_dir() -> String {
    "auto".to_string()
}

fn default_buffer_flush_threshold() -> usize {
    50000 // ~12K-25K tokens, triggers ~2-4 times in a long session
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExtractConfig {
    pub provider: String,
    pub google: GoogleExtractConfig,
    pub omlx: OmlxExtractConfig,
    /// Inject existing memory summaries into extraction prompts to reduce duplicates.
    /// Default: false (opt-in to avoid sending memory content to remote providers).
    #[serde(default)]
    pub inject_existing_context: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(default)]
pub struct AsyncMemoryConfig {
    /// Which LLM provider the async memory worker uses.
    /// "inherit" follows [extract].provider.
    pub provider: String,
    /// Maximum retry attempts for failed async memory jobs before dead-lettering.
    pub max_retries: u32,
    /// Base retry backoff in milliseconds. Exponential backoff uses this as step 0.
    pub base_backoff_ms: u64,
    /// Maximum jobs a single worker run will process before exiting.
    pub max_jobs_per_run: usize,
    /// Batch size used when selecting ready jobs from the queue.
    pub batch_size: usize,
    /// Minimum time between worker spawns for the same project queue.
    pub spawn_cooldown_ms: u64,
    /// Maximum items kept in the project working set.
    pub max_working_set_items: usize,
    /// Maximum items kept in the project always-on index.
    pub max_always_on_items: usize,
    /// Maximum number of injected memory-surface items selected per query.
    pub selection_limit: usize,
    /// Time window for suppressing near-duplicate queued events.
    pub fingerprint_window_ms: u64,
    /// Number of recent event fingerprints kept per project.
    pub recent_event_cache_size: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoogleExtractConfig {
    pub model: String,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default = "default_google_endpoint")]
    pub endpoint: String,
    /// Max input characters. 0 = no truncation (default for gemini-3.1-flash-lite-preview which supports 1M tokens).
    #[serde(default)]
    pub max_input_chars: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OmlxExtractConfig {
    pub endpoint: String,
    #[serde(default = "default_omlx_extract_model")]
    pub model: String,
    /// Max input characters for local models (default 16000, suitable for 7B-13B models).
    #[serde(default = "default_omlx_max_input_chars")]
    pub max_input_chars: usize,
    /// Prepend /no_think to system prompts (for Qwen3 thinking mode). Default: true.
    #[serde(default = "default_true")]
    pub disable_thinking: bool,
}

fn default_omlx_max_input_chars() -> usize {
    16000
}

fn default_omlx_extract_model() -> String {
    "default".to_string()
}

// ---------------------------------------------------------------------------
// Query Expansion config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueryExpansionConfig {
    /// Provider: "google", "omlx", or "none". Default: "google".
    #[serde(default = "default_expand_provider")]
    pub provider: String,
    /// Maximum number of expanded query variants. Default: 3.
    #[serde(default = "default_max_expansions")]
    pub max_expansions: usize,
    #[serde(default)]
    pub google: GoogleExpandConfig,
    #[serde(default)]
    pub omlx: OmlxExpandConfig,
}

fn default_expand_provider() -> String {
    "google".to_string()
}
fn default_max_expansions() -> usize {
    3
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoogleExpandConfig {
    #[serde(default = "default_expand_google_model")]
    pub model: String,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default = "default_google_endpoint")]
    pub endpoint: String,
}

fn default_expand_google_model() -> String {
    "gemini-3.1-flash-lite-preview".to_string()
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OmlxExpandConfig {
    #[serde(default = "default_omlx_expand_endpoint")]
    pub endpoint: String,
    #[serde(default = "default_omlx_model")]
    pub model: String,
    /// Prepend /no_think to system prompts (for Qwen3 thinking mode). Default: true.
    #[serde(default = "default_true")]
    pub disable_thinking: bool,
}

fn default_omlx_expand_endpoint() -> String {
    "http://localhost:8000/v1".to_string()
}

fn default_true() -> bool {
    true
}

// ---------------------------------------------------------------------------
// Default implementations
// ---------------------------------------------------------------------------

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            path: "auto".to_string(),
        }
    }
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            provider: "google".to_string(),
            dimensions: 3072,
            google: GoogleEmbeddingConfig::default(),
            omlx: OmlxEmbeddingConfig::default(),
        }
    }
}

impl Default for GoogleEmbeddingConfig {
    fn default() -> Self {
        Self {
            model: "gemini-embedding-001".to_string(),
            api_key: None,
            endpoint: default_google_endpoint(),
        }
    }
}

impl Default for OmlxEmbeddingConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://localhost:8000/v1".to_string(),
            model: "default".to_string(),
        }
    }
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            rrf_k: 60.0,
            rrf_fts_weight: 0.3,
            rrf_vec_weight: 0.7,
            fusion_method: "rrf".to_string(),
            cc_alpha: 0.5,
            dedup_similarity: 0.70,
            dedup_time_window_days: 7,
            llm_reranker: default_llm_reranker(),
            llm_reranker_top_n: default_llm_reranker_top_n(),
            llm_reranker_timeout_ms: default_llm_reranker_timeout_ms(),
            mmr_lambda: default_mmr_lambda(),
            strong_signal_ratio: default_strong_signal_ratio(),
            strong_signal_single: default_strong_signal_single(),
        }
    }
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            overlap_percent: 10,
            metadata_prefix: true,
        }
    }
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            supermemory_enabled: true,
            auto_memory_enabled: true,
            auto_memory_glob: "~/.claude/projects/*/memory/**/*.md".to_string(),
            api_key: None,
            endpoint: default_supermemory_endpoint(),
        }
    }
}

impl Default for DecayConfig {
    fn default() -> Self {
        Self {
            base_lambda: 0.06,
            ltm_beta: 0.8,
            stm_beta: 1.2,
            interval_hours: 24,
            prune_threshold: 0.05,
            stm_to_ltm_access_count: 5,
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            compact: false,
            sse_enabled: false,
            sse_port: 8680,
            sse_bind: "127.0.0.1".to_string(),
            gui_enabled: false,
            allow_unauthenticated_loopback: true,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(default)]
pub struct ProxyConfig {
    pub port: u16,
    pub bind: String,
    pub anthropic_upstream: String,
    pub openai_upstream: String,
    pub chatgpt_upstream: String,
    pub codex_upstream: String,
    pub extract_enabled: bool,
    pub store_min_chars: usize,
    pub store_min_score: u32,
    pub max_retries: u32,
    pub retry_base_ms: u64,
    pub max_request_body: usize,
    pub max_response_buffer: usize,
    pub max_sse_buffer: usize,
    pub max_concurrent_extractions: usize,
    pub allow_unauthenticated_loopback: bool,
}

impl Default for ProxyConfig {
    fn default() -> Self {
        Self {
            port: 8690,
            bind: "127.0.0.1".to_string(),
            anthropic_upstream: "https://api.anthropic.com".to_string(),
            openai_upstream: "https://api.openai.com".to_string(),
            chatgpt_upstream: "https://chatgpt.com/backend-api".to_string(),
            codex_upstream: "https://chatgpt.com/backend-api/codex".to_string(),
            extract_enabled: true,
            store_min_chars: 220,
            store_min_score: 3,
            max_retries: 2,
            retry_base_ms: 500,
            max_request_body: 1_048_576,
            max_response_buffer: 1_048_576,
            max_sse_buffer: 1_048_576,
            max_concurrent_extractions: 4,
            allow_unauthenticated_loopback: true,
        }
    }
}

impl Default for HooksConfig {
    fn default() -> Self {
        Self {
            min_turns: 20,
            context_before: 3,
            context_after: 1,
            max_items_per_session: 10,
            signal_keywords: default_signal_keywords(),
            buffer_dir: default_buffer_dir(),
            buffer_flush_threshold: default_buffer_flush_threshold(),
        }
    }
}

fn default_signal_keywords() -> Vec<String> {
    vec![
        // English
        "decided",
        "chose",
        "architecture",
        "design",
        "pattern",
        "bug",
        "fix",
        "fixed",
        "resolved",
        "error",
        "crash",
        "installed",
        "deployed",
        "migrated",
        "important",
        "remember",
        "solution",
        "tradeoff",
        "upgrade",
        "deprecated",
        "workflow",
        "released",
        "because",
        "reason",
        "switched",
        "selected",
        "prefer",
        "root cause",
        "workaround",
        "conclusion",
        // Chinese (for matching Chinese conversation content)
        "决策",
        "选型",
        "架构",
        "设计",
        "模式",
        "修复",
        "解决",
        "安装",
        "部署",
        "迁移",
        "重要",
        "记住",
        "记录",
        "方案",
        "权衡",
        "升级",
        "废弃",
        "流程",
        "发布",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

impl Default for ExtractConfig {
    fn default() -> Self {
        Self {
            provider: "google".to_string(),
            google: GoogleExtractConfig::default(),
            omlx: OmlxExtractConfig::default(),
            inject_existing_context: false,
        }
    }
}

impl Default for AsyncMemoryConfig {
    fn default() -> Self {
        Self {
            provider: "inherit".to_string(),
            max_retries: 3,
            base_backoff_ms: 2_000,
            max_jobs_per_run: 32,
            batch_size: 8,
            spawn_cooldown_ms: 1_500,
            max_working_set_items: 40,
            max_always_on_items: 24,
            selection_limit: 5,
            fingerprint_window_ms: 120_000,
            recent_event_cache_size: 256,
        }
    }
}

impl Default for GoogleExtractConfig {
    fn default() -> Self {
        Self {
            model: "gemini-3.1-flash-lite-preview".to_string(),
            api_key: None,
            endpoint: default_google_endpoint(),
            max_input_chars: 0, // 0 = no truncation (1M token model)
        }
    }
}

impl Default for OmlxExtractConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://localhost:11434/v1".to_string(),
            model: "default".to_string(),
            max_input_chars: default_omlx_max_input_chars(),
            disable_thinking: true,
        }
    }
}

impl Default for QueryExpansionConfig {
    fn default() -> Self {
        Self {
            provider: default_expand_provider(),
            max_expansions: default_max_expansions(),
            google: GoogleExpandConfig::default(),
            omlx: OmlxExpandConfig::default(),
        }
    }
}

impl Default for GoogleExpandConfig {
    fn default() -> Self {
        Self {
            model: default_expand_google_model(),
            api_key: None,
            endpoint: default_google_endpoint(),
        }
    }
}

impl Default for OmlxExpandConfig {
    fn default() -> Self {
        Self {
            endpoint: default_omlx_expand_endpoint(),
            model: "default".to_string(),
            disable_thinking: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

fn nonempty_env(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

impl ReinConfig {
    /// Load configuration with the following priority (highest wins):
    /// 1. Environment variable overrides
    /// 2. TOML config file (`$REIN_CONFIG` or `~/.config/rein/config.toml`)
    /// 3. Compiled-in defaults
    pub fn load() -> anyhow::Result<Self> {
        let mut config = Self::default();

        // Determine config file path
        let config_path = std::env::var("REIN_CONFIG")
            .map(PathBuf::from)
            .unwrap_or_else(|_| dirs_config_path());

        // Merge TOML file if it exists
        if config_path.exists() {
            let contents = std::fs::read_to_string(&config_path)?;
            config = merge_toml(config, &contents)?;
        }

        // Environment variable overrides
        if let Some(db) = nonempty_env("REIN_DB") {
            config.database.path = db;
        }
        if let Some(key) = nonempty_env("GEMINI_API_KEY") {
            config.embedding.google.api_key = Some(key);
        }
        if let Some(key) = nonempty_env("SUPERMEMORY_CC_API_KEY") {
            config.sync.api_key = Some(key);
        }
        // Reuse GEMINI_API_KEY for extract if not set separately
        if config.extract.google.api_key.is_none() {
            if let Some(key) = nonempty_env("GEMINI_API_KEY") {
                config.extract.google.api_key = Some(key);
            }
        }
        // Reuse GEMINI_API_KEY for query expansion if not set separately
        if config.query_expansion.google.api_key.is_none() {
            if let Some(key) = nonempty_env("GEMINI_API_KEY") {
                config.query_expansion.google.api_key = Some(key);
            }
        }
        // Server overrides (useful for Docker: REIN_SSE_BIND=0.0.0.0)
        if let Some(bind) = nonempty_env("REIN_SSE_BIND") {
            config.server.sse_bind = bind;
        }
        if let Some(port) = nonempty_env("REIN_SSE_PORT") {
            match port.parse::<u16>() {
                Ok(p) => config.server.sse_port = p,
                Err(_) => eprintln!("rein: WARNING — REIN_SSE_PORT='{port}' is not a valid port number, using default {}", config.server.sse_port),
            }
        }
        // Proxy overrides
        if let Some(bind) = nonempty_env("REIN_PROXY_BIND") {
            config.proxy.bind = bind;
        }
        if let Some(port) = nonempty_env("REIN_PROXY_PORT") {
            match port.parse::<u16>() {
                Ok(p) => config.proxy.port = p,
                Err(_) => eprintln!("rein: WARNING — REIN_PROXY_PORT='{port}' is not a valid port number, using default {}", config.proxy.port),
            }
        }
        if let Some(provider) = nonempty_env("REIN_ASYNC_MEMORY_PROVIDER") {
            config.async_memory.provider = provider;
        }

        config.validate()?;
        Ok(config)
    }

    /// Load configuration from a specific TOML string (for testing).
    pub fn load_from_str(toml_str: &str) -> anyhow::Result<Self> {
        let config = Self::default();
        let merged = merge_toml(config, toml_str)?;
        merged.validate()?;
        Ok(merged)
    }

    /// Get typed embedding provider (prevents typo misconfiguration).
    pub fn embedding_provider(&self) -> Provider {
        Provider::from_str(&self.embedding.provider)
    }

    /// Get typed extract provider.
    pub fn extract_provider(&self) -> Provider {
        Provider::from_str(&self.extract.provider)
    }

    /// Get typed query expansion provider.
    pub fn expand_provider(&self) -> Provider {
        Provider::from_str(&self.query_expansion.provider)
    }

    /// Get typed LLM reranker provider.
    pub fn reranker_provider(&self) -> Provider {
        Provider::from_str(&self.search.llm_reranker)
    }

    /// The embedding model name (for cache keying and model-change detection).
    pub fn embedding_model(&self) -> String {
        match self.embedding_provider() {
            Provider::Omlx => format!("omlx:{}", self.embedding.omlx.model),
            Provider::Google => format!("google:{}", self.embedding.google.model),
            Provider::None => "none".to_string(),
        }
    }

    /// Validate configuration and return an error for invalid values.
    pub fn validate(&self) -> anyhow::Result<()> {
        validate_provider_name("embedding.provider", &self.embedding.provider)?;
        validate_provider_name("extract.provider", &self.extract.provider)?;
        validate_provider_name("query_expansion.provider", &self.query_expansion.provider)?;
        validate_provider_name("search.llm_reranker", &self.search.llm_reranker)?;
        validate_provider_name_or_inherit("async_memory.provider", &self.async_memory.provider)?;

        if self.database.path.trim().is_empty() {
            anyhow::bail!("database.path must not be empty");
        }
        if self.server.sse_bind.trim().is_empty() {
            anyhow::bail!("server.sse_bind must not be empty");
        }
        if self.server.sse_port == 0 {
            anyhow::bail!("server.sse_port must be in 1..=65535");
        }
        if self.proxy.bind.trim().is_empty() {
            anyhow::bail!("proxy.bind must not be empty");
        }
        if self.proxy.port == 0 {
            anyhow::bail!("proxy.port must be in 1..=65535");
        }

        match self.embedding_provider() {
            Provider::Google => {
                if self.embedding.google.api_key.is_none() {
                    eprintln!("rein: WARNING — embedding provider is 'google' but GEMINI_API_KEY is not set");
                    eprintln!("rein: Vector search and embedding will be disabled. FTS search still works.");
                }
            }
            Provider::Omlx => {
                eprintln!(
                    "rein: using OMLX embedding backend at {}",
                    self.embedding.omlx.endpoint
                );
            }
            Provider::None => {}
        }
        if self.sync.supermemory_enabled && self.sync.api_key.is_none() {
            eprintln!("rein: NOTE — supermemory is enabled but SUPERMEMORY_CC_API_KEY is not set");
        }
        match self.extract_provider() {
            Provider::Google => {
                if self.extract.google.api_key.is_none() {
                    eprintln!("rein: NOTE — extract provider is 'google' but GEMINI_API_KEY is not set, LLM extraction disabled");
                }
            }
            Provider::Omlx => {
                eprintln!(
                    "rein: using OMLX extract backend at {}",
                    self.extract.omlx.endpoint
                );
            }
            Provider::None => {}
        }
        match self.expand_provider() {
            Provider::Google => {
                if self.query_expansion.google.api_key.is_none() {
                    eprintln!("rein: NOTE — query expansion provider is 'google' but GEMINI_API_KEY is not set, expansion disabled");
                }
            }
            Provider::Omlx => {
                eprintln!(
                    "rein: using OMLX query expansion backend at {}",
                    self.query_expansion.omlx.endpoint
                );
            }
            Provider::None => {}
        }

        // Validate search config
        if self.search.cc_alpha < 0.0 || self.search.cc_alpha > 1.0 {
            anyhow::bail!(
                "search.cc_alpha must be in [0.0, 1.0], got {}",
                self.search.cc_alpha
            );
        }
        if self.search.dedup_similarity < 0.0 || self.search.dedup_similarity > 1.0 {
            anyhow::bail!(
                "search.dedup_similarity must be in [0.0, 1.0], got {}",
                self.search.dedup_similarity
            );
        }
        // Validate decay config
        if self.decay.base_lambda <= 0.0 {
            anyhow::bail!(
                "decay.base_lambda must be positive, got {}",
                self.decay.base_lambda
            );
        }
        if self.decay.prune_threshold < 0.0 {
            anyhow::bail!(
                "decay.prune_threshold must be non-negative, got {}",
                self.decay.prune_threshold
            );
        }
        // Validate embedding dimensions
        if self.embedding.dimensions == 0 {
            anyhow::bail!("embedding.dimensions must be > 0");
        }

        // Validate cleanup config
        let c = &self.cleanup;
        if c.vec_dedup_strong_threshold < 0.0 || c.vec_dedup_strong_threshold > 1.0 {
            anyhow::bail!(
                "cleanup.vec_dedup_strong_threshold must be in [0.0, 1.0], got {}",
                c.vec_dedup_strong_threshold
            );
        }
        if c.vec_dedup_weak_threshold < 0.0 || c.vec_dedup_weak_threshold > 1.0 {
            anyhow::bail!(
                "cleanup.vec_dedup_weak_threshold must be in [0.0, 1.0], got {}",
                c.vec_dedup_weak_threshold
            );
        }
        if c.vec_dedup_weak_threshold > c.vec_dedup_strong_threshold {
            anyhow::bail!(
                "cleanup.vec_dedup_weak_threshold ({}) must be <= vec_dedup_strong_threshold ({})",
                c.vec_dedup_weak_threshold,
                c.vec_dedup_strong_threshold
            );
        }
        if c.llm_batch_size == 0 {
            anyhow::bail!("cleanup.llm_batch_size must be >= 1");
        }

        Ok(())
    }

    /// Open a SqliteStore with the current config's model and dimensions.
    ///
    /// v0.22 P1: when the caller is inside a Tokio runtime AND the DB is
    /// file-backed AND `REIN_ASYNC_P1` is not `0`, attach a process-level
    /// connection pool keyed by path. `recall`'s 3-channel fanout uses the
    /// pool to elide per-channel schema-init + embedding-model checks.
    /// Non-tokio callers (CLI direct invocations) and `:memory:` DBs
    /// bypass the pool entirely and use the pre-v0.22 path.
    pub fn open_store(&self) -> crate::types::ReinResult<crate::store::SqliteStore> {
        let db_path = self.resolve_db_path();
        let store = crate::store::SqliteStore::new(
            &db_path,
            &self.embedding_model(),
            self.embedding.dimensions,
        )?;

        // Opt-out: REIN_ASYNC_P1=0 disables the pool path entirely.
        // Read per-call (not cached): tests and `doctor --fix` can toggle it
        // mid-process. The cost is a ~100ns env lookup against a ~ms-scale
        // store open — negligible, and preserves test-time controllability.
        if std::env::var("REIN_ASYNC_P1").ok().as_deref() == Some("0") {
            return Ok(store);
        }
        // In-memory DBs cannot share connections — each new conn opens a
        // distinct empty DB. Bypass the pool at init time, not at recall
        // use site, so the `is_memory_db` guards in recall.rs stay
        // consistent with the store carrying no pool.
        if db_path.to_str() == Some(":memory:") {
            return Ok(store);
        }
        // Non-tokio callers (e.g. the CLI direct path) would have the pool
        // ignored by recall.rs via `Handle::try_current()` anyway; skipping
        // cache+build here saves the setup cost and keeps those paths
        // bit-identical to the pre-v0.22 behavior.
        if tokio::runtime::Handle::try_current().is_err() {
            return Ok(store);
        }
        let pool = pool_for_path(&db_path)?;
        Ok(store.with_pool(pool))
    }

    /// Resolve the database path. `"auto"` → `~/.rein/memories.db`
    pub fn resolve_db_path(&self) -> PathBuf {
        if self.database.path == "auto" {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
            let new_dir = PathBuf::from(&home).join(".rein");
            let new_path = new_dir.join("memories.db");

            // Check for old location and auto-migrate
            if !new_path.exists() {
                if let Some(dirs) = directories::ProjectDirs::from("", "", "rein") {
                    let old_path = dirs.data_dir().join("memories.db");
                    if old_path.exists() {
                        std::fs::create_dir_all(&new_dir).ok();
                        if std::fs::rename(&old_path, &new_path).is_ok() {
                            eprintln!(
                                "rein: migrated database from {} to {}",
                                old_path.display(),
                                new_path.display()
                            );
                        }
                    }
                }
            }

            std::fs::create_dir_all(&new_dir).ok();
            new_path
        } else {
            PathBuf::from(&self.database.path)
        }
    }
}

// ---------------------------------------------------------------------------
// v0.22 P1 process-level pool cache (keyed by DB path).
//
// Keyed by PathBuf, not singleton: cargo integration tests run in parallel
// inside one process, each with its own `TempDir::new()` → unique db_path.
// A singleton pool would pin to whichever path wins the init race and silently
// serve every subsequent test from the wrong DB.
//
// Weak<ConnPool>, not Arc<ConnPool>: when the last `SqliteStore` holding the
// strong ref drops (e.g. a test's TempDir destructor tears down with it), the
// pool and its connections drop with it → file descriptors are reclaimed
// promptly. Without `Weak`, 58 tempdir tests would leak ~464 fds across a run.
// Opportunistic `retain` in `pool_for_path` prunes dead weak entries so the
// map stays bounded by live pools, not ever-seen pools.
// ---------------------------------------------------------------------------

static OPEN_STORE_POOL_CACHE: OnceLock<
    Mutex<HashMap<PathBuf, Weak<crate::store::pool::ConnPool>>>,
> = OnceLock::new();

fn pool_cache() -> &'static Mutex<HashMap<PathBuf, Weak<crate::store::pool::ConnPool>>> {
    OPEN_STORE_POOL_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn pool_for_path(
    db_path: &Path,
) -> crate::types::ReinResult<Arc<crate::store::pool::ConnPool>> {
    // Normalize the path spelling for cache keying so that `./rein.db`,
    // `rein.db`, and `/abs/rein.db` (all pointing at the same file)
    // resolve to the same pool (Codex v0.22 round-2 LOW #3).
    //
    // `std::path::absolute` handles relative→absolute without requiring
    // the file to exist — unlike `canonicalize`, which errors on
    // first-run when the DB file is about to be created by
    // `ConnPool::new` below.  Symlinks and `.`/`..` components are NOT
    // resolved; callers needing full canonicalization should
    // `canonicalize` upstream.
    //
    // Key resolution runs INSIDE the Mutex guard so concurrent callers
    // can never race to create two pools for the same file (Codex
    // round-2 clarification).  `absolute()` is fast (no syscall on the
    // happy path) so holding the lock across it is cheap.
    let mut map = pool_cache()
        .lock()
        .expect("open_store pool cache mutex poisoned");
    let key = std::path::absolute(db_path).unwrap_or_else(|_| db_path.to_path_buf());
    // Opportunistic prune: drop entries whose `SqliteStore` holders are gone.
    map.retain(|_, weak| weak.strong_count() > 0);
    if let Some(weak) = map.get(&key) {
        if let Some(pool) = weak.upgrade() {
            return Ok(pool);
        }
    }
    let size = crate::store::pool::default_pool_size();
    // Pass the original `db_path` to `ConnPool::new`; cache keying is the
    // only place we need the normalized form.
    let pool = Arc::new(crate::store::pool::ConnPool::new(db_path, size)?);
    map.insert(key, Arc::downgrade(&pool));
    Ok(pool)
}

fn validate_provider_name(field: &str, value: &str) -> anyhow::Result<()> {
    match value.to_lowercase().as_str() {
        "google" | "omlx" | "none" => Ok(()),
        _ => anyhow::bail!("invalid {field}='{value}'. Expected one of: google, omlx, none"),
    }
}

fn validate_provider_name_or_inherit(field: &str, value: &str) -> anyhow::Result<()> {
    match value.to_lowercase().as_str() {
        "inherit" | "google" | "omlx" | "none" => Ok(()),
        _ => {
            anyhow::bail!("invalid {field}='{value}'. Expected one of: inherit, google, omlx, none")
        }
    }
}

/// Default config file location: `~/.config/rein/config.toml`
/// Locate the config file. First existing path wins:
///   1. ~/.rein/config.toml          (alongside the database)
///   2. ~/.config/rein/config.toml   (XDG standard, works cross-platform)
///   3. ~/Library/Application Support/rein/config.toml  (macOS native via `directories`)
/// If none exist, return the XDG path so that new users get the conventional location.
fn dirs_config_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());

    let candidates = [
        // ~/.rein/config.toml — single home alongside the database
        PathBuf::from(&home).join(".rein/config.toml"),
        // XDG standard (Linux default, also common on macOS)
        PathBuf::from(&home).join(".config/rein/config.toml"),
    ];

    // Check dotfile paths first
    for path in &candidates {
        if path.exists() {
            return path.clone();
        }
    }

    // macOS native path via `directories` crate
    if let Some(dirs) = directories::ProjectDirs::from("", "", "rein") {
        let native = dirs.config_dir().join("config.toml");
        if native.exists() {
            return native;
        }
    }

    // Nothing found — default to XDG so new users get the conventional location
    candidates[1].clone()
}

/// Merge a TOML string over an existing config by deserializing into a
/// `toml::Value` table and overlaying it field-by-field so that missing keys
/// in the file keep their default values.
fn merge_toml(base: ReinConfig, toml_str: &str) -> anyhow::Result<ReinConfig> {
    // Serialize defaults to a toml::Value table
    let default_toml = toml::to_string(&serde_to_value(&base)?)?;
    let mut base_val: toml::Value = toml::from_str(&default_toml)?;

    // Parse the user file
    let user_val: toml::Value = toml::from_str(toml_str)?;

    // Deep-merge user over base
    deep_merge(&mut base_val, &user_val);

    // Deserialize the merged table back into ReinConfig
    let merged: ReinConfig = base_val.try_into()?;
    Ok(merged)
}

/// Convert a ReinConfig to a toml::Value via serde_json round-trip
/// (since ReinConfig doesn't derive Serialize, we use the default TOML).
fn serde_to_value(config: &ReinConfig) -> anyhow::Result<toml::Value> {
    // Build manually from defaults by serializing the embedded default.toml
    // and then patching the runtime values. Simpler: just use the embedded
    // default TOML as the base and patch non-serializable fields.
    let default_str = include_str!("../config/default.toml");
    let mut val: toml::Value = toml::from_str(default_str)?;

    // Patch fields that may differ from the embedded default (e.g., api_key)
    if let Some(tbl) = val
        .get_mut("embedding")
        .and_then(|v| v.get_mut("google"))
        .and_then(|v| v.as_table_mut())
    {
        if let Some(ref key) = config.embedding.google.api_key {
            tbl.insert("api_key".to_string(), toml::Value::String(key.clone()));
        }
    }
    if let Some(tbl) = val.get_mut("sync").and_then(|v| v.as_table_mut()) {
        if let Some(ref key) = config.sync.api_key {
            tbl.insert("api_key".to_string(), toml::Value::String(key.clone()));
        }
    }
    if let Some(tbl) = val
        .get_mut("extract")
        .and_then(|v| v.get_mut("google"))
        .and_then(|v| v.as_table_mut())
    {
        if let Some(ref key) = config.extract.google.api_key {
            tbl.insert("api_key".to_string(), toml::Value::String(key.clone()));
        }
    }
    if let Some(tbl) = val
        .get_mut("query_expansion")
        .and_then(|v| v.get_mut("google"))
        .and_then(|v| v.as_table_mut())
    {
        if let Some(ref key) = config.query_expansion.google.api_key {
            tbl.insert("api_key".to_string(), toml::Value::String(key.clone()));
        }
    }

    // Patch database path
    if let Some(tbl) = val.get_mut("database").and_then(|v| v.as_table_mut()) {
        tbl.insert(
            "path".to_string(),
            toml::Value::String(config.database.path.clone()),
        );
    }

    Ok(val)
}

fn deep_merge(base: &mut toml::Value, overlay: &toml::Value) {
    if let (Some(base_tbl), Some(overlay_tbl)) = (base.as_table_mut(), overlay.as_table()) {
        for (key, val) in overlay_tbl {
            if let Some(existing) = base_tbl.get_mut(key) {
                deep_merge(existing, val);
            } else {
                base_tbl.insert(key.clone(), val.clone());
            }
        }
    } else {
        *base = overlay.clone();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = ReinConfig::default();
        assert!((cfg.search.rrf_k - 60.0).abs() < f64::EPSILON);
        assert_eq!(cfg.embedding.dimensions, 3072);
        assert!(!cfg.server.compact);
        assert_eq!(cfg.database.path, "auto");
        assert_eq!(cfg.embedding.provider, "google");
        assert_eq!(cfg.chunking.max_tokens, 512);
        assert!((cfg.decay.base_lambda - 0.06).abs() < f64::EPSILON);
    }

    #[test]
    fn test_load_from_toml() {
        let toml_str = r#"
[search]
rrf_k = 30.0
"#;
        let cfg = ReinConfig::load_from_str(toml_str).unwrap();
        // Override applied
        assert!((cfg.search.rrf_k - 30.0).abs() < f64::EPSILON);
        // Other defaults preserved
        assert_eq!(cfg.embedding.dimensions, 3072);
        assert!(!cfg.server.compact);
        assert_eq!(cfg.database.path, "auto");
    }

    /// RAII guard: remember the current env var value and restore it on drop,
    /// so a panic inside the test does not leak state to subsequent tests.
    struct EnvGuard {
        key: &'static str,
        prior: Option<String>,
    }

    impl EnvGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let prior = std::env::var(key).ok();
            std::env::set_var(key, value);
            Self { key, prior }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.prior {
                Some(v) => std::env::set_var(self.key, v),
                None => std::env::remove_var(self.key),
            }
        }
    }

    #[test]
    #[serial_test::serial(global_state)]
    fn test_env_override_db() {
        // `global_state` serializes with every other test in the crate that
        // mutates process-global env vars (doctor + mcp::rest::tests suites).
        // RAII guards ensure env is restored even if the assertion panics.
        let _db = EnvGuard::set("REIN_DB", "/tmp/test.db");
        let _cfg_path = EnvGuard::set("REIN_CONFIG", "/nonexistent/path/config.toml");
        let cfg = ReinConfig::load().unwrap();
        assert_eq!(cfg.database.path, "/tmp/test.db");
    }

    #[test]
    fn test_resolve_db_path_auto() {
        let cfg = ReinConfig::default();
        let path = cfg.resolve_db_path();
        let path_str = path.to_string_lossy();
        assert!(
            path_str.contains("rein") && path_str.ends_with("memories.db"),
            "Expected path containing 'rein/memories.db', got: {path_str}"
        );
    }

    #[test]
    fn test_db_path_auto() {
        // Just verify the logic: "auto" should map to ~/.rein/memories.db
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
        let expected = std::path::PathBuf::from(&home).join(".rein/memories.db");
        // Don't call resolve_db_path() as it has filesystem side effects
        assert!(expected.to_string_lossy().ends_with(".rein/memories.db"));
    }

    #[test]
    fn test_db_path_custom() {
        let mut config = ReinConfig::default();
        // Custom path should be returned as-is
        config.database.path = "/custom/path/test.db".to_string();
        assert_eq!(
            config.resolve_db_path(),
            std::path::PathBuf::from("/custom/path/test.db")
        );
    }

    #[test]
    fn test_unknown_config_field_rejected() {
        let toml_str = r#"
[search]
rrf_k = 30.0
unknown_knob = true
"#;
        assert!(ReinConfig::load_from_str(toml_str).is_err());
    }

    #[test]
    fn test_invalid_provider_rejected_by_validate() {
        let mut cfg = ReinConfig::default();
        cfg.embedding.provider = "bogus".to_string();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_path_prefers_xdg_over_native() {
        // dirs_config_path() should return a path ending in config.toml
        let path = super::dirs_config_path();
        assert!(
            path.to_string_lossy().ends_with("config.toml"),
            "Expected path ending in config.toml, got: {}",
            path.display()
        );
        // On a clean machine with no config file, should default to XDG
        // (~/.config/rein/config.toml), not macOS native.
        // Skip the assertion if any config file already exists (XDG, dot-rein,
        // or macOS native via ProjectDirs) — the test is only meaningful on a
        // machine with no pre-existing config.
        let home = std::env::var("HOME").unwrap_or_default();
        let xdg = std::path::PathBuf::from(&home).join(".config/rein/config.toml");
        let dot_rein = std::path::PathBuf::from(&home).join(".rein/config.toml");
        let native_exists = directories::ProjectDirs::from("", "", "rein")
            .map(|d| d.config_dir().join("config.toml").exists())
            .unwrap_or(false);
        if !xdg.exists() && !dot_rein.exists() && !native_exists {
            assert_eq!(
                path, xdg,
                "Default should be XDG path when no config exists"
            );
        }
    }

    #[test]
    fn test_merge_all_four_api_keys() {
        let toml_str = r#"
[embedding.google]
api_key = "test-embed-key"

[extract.google]
api_key = "test-extract-key"

[query_expansion.google]
api_key = "test-expand-key"

[sync]
api_key = "test-sync-key"
"#;
        let cfg = ReinConfig::load_from_str(toml_str).expect("merge should succeed");
        assert_eq!(
            cfg.embedding.google.api_key.as_deref(),
            Some("test-embed-key"),
            "embedding api_key lost"
        );
        assert_eq!(
            cfg.extract.google.api_key.as_deref(),
            Some("test-extract-key"),
            "extract api_key lost"
        );
        assert_eq!(
            cfg.query_expansion.google.api_key.as_deref(),
            Some("test-expand-key"),
            "expand api_key lost"
        );
        assert_eq!(
            cfg.sync.api_key.as_deref(),
            Some("test-sync-key"),
            "sync api_key lost"
        );
    }
}
