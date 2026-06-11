use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::sync::atomic::{AtomicBool, Ordering};
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
// Config schema version (v1.0)
// ---------------------------------------------------------------------------

/// Current config schema version understood by this rein build.
///
/// Monotonic integer (NOT a tunable parameter — pure metadata, like the
/// SQLite `PRAGMA user_version` schema counter). Bump it whenever the config
/// surface changes in a way that needs a load-time transform (a field rename,
/// removal, or semantic change) and add the corresponding migration step.
///
/// Forward-compat contract: a config stamped with a version GREATER than this
/// was written by a newer rein and is refused at load (`validate`), so an
/// older binary never silently misreads future semantics. A version EQUAL to
/// this, or the unversioned sentinel `0` (any pre-1.0 config, or the derived
/// `Default`), is the current baseline and loads as-is.
pub const CURRENT_CONFIG_VERSION: u32 = 1;

fn default_config_version() -> u32 {
    CURRENT_CONFIG_VERSION
}

// ---------------------------------------------------------------------------
// Config structs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
#[derive(Default)]
pub struct ReinConfig {
    /// Config schema version (see [`CURRENT_CONFIG_VERSION`]). Absent in any
    /// pre-1.0 config → deserializes to the current version; the derived
    /// `Default` leaves it `0` (unversioned baseline). A value newer than this
    /// build understands is rejected at load. Never a behavior knob.
    #[serde(default = "default_config_version")]
    pub config_version: u32,
    /// `[database]` — SQLite memory store path.
    pub database: DatabaseConfig,
    /// `[embedding]` — embedding provider, vector dimensions, and per-provider (Google / OMLX) endpoints.
    pub embedding: EmbeddingConfig,
    /// `[search]` — fusion (RRF / CC), LLM reranker, MMR diversity, and strong-signal bypass tuning.
    pub search: SearchConfig,
    /// `[chunking]` — chunk max-tokens, overlap percent, and metadata-prefix toggle.
    pub chunking: ChunkingConfig,
    /// `[sync]` — Supermemory cross-validation + auto-memory file sync (enable flags, glob, API key, endpoint).
    pub sync: SyncConfig,
    /// `[decay]` — Ebbinghaus decay lambdas/betas, sweep interval, prune threshold, and STM→LTM promotion count.
    pub decay: DecayConfig,
    /// `[server]` — MCP/SSE/GUI surface: SSE bind/port, auth policy, allowed hosts, and background warmup.
    pub server: ServerConfig,
    /// `[hooks]` — session-extraction hook tuning: min turns, context window, per-session cap, signal keywords, buffer flush threshold.
    #[serde(default)]
    pub hooks: HooksConfig,
    /// `[extract]` — LLM extraction provider (Google / OMLX) and existing-context injection toggle.
    #[serde(default)]
    pub extract: ExtractConfig,
    /// `[adaptive]` — operational settings for the M1-M5 adaptive engine (cold-start thresholds, retention, cache TTL).
    #[serde(default)]
    pub adaptive: AdaptiveConfig,
    /// `[query_expansion]` — query-rewrite provider (default "google"), max variants, and per-provider endpoints.
    #[serde(default)]
    pub query_expansion: QueryExpansionConfig,
    /// `[proxy]` — LLM-capture reverse proxy: bind/port, upstreams, body/buffer limits, and extraction gating.
    #[serde(default)]
    pub proxy: ProxyConfig,
    /// `[async_memory]` — async memory worker queue: provider, retry/backoff, batch size, and working-set caps.
    #[serde(default)]
    pub async_memory: AsyncMemoryConfig,
    /// `[cleanup]` — consolidation/dedup thresholds, LLM batch size + budget, and vector-dedup similarity bands.
    #[serde(default)]
    pub cleanup: CleanupConfig,
    /// `[intelligent_merge]` — opt-in LLM gray-zone merge classifier (enable flag, provider override, per-provider config).
    #[serde(default)]
    pub intelligent_merge: IntelligentMergeConfig,
    /// `[resummerize]` — v0.23 slow-channel LLM canonical recompression (enable flag, LLM backend, batch size).
    #[serde(default)]
    pub resummerize: ResummerizeConfig,
    /// `[ars]` — Adaptive Retention / Synthesis (Cap A/B/C): summary, recall-synthesis, and cold-archive knobs.
    #[serde(default)]
    pub ars: ArsConfig,
    /// `[dedup]` — v0.27 dedup paths: triple-overlap threshold, N-merge fan-out cap, temporal-supersede flag.
    #[serde(default)]
    pub dedup: DedupConfig,
    /// v0.27.1 Track 2 — `[llm]` parent section providing 4-level
    /// inheritance precedence for every LLM consumer (extract /
    /// query_expansion / search.llm_reranker / ars.recall_synthesis /
    /// ars.concept_summary / ars.cold_archive / ars.llm_judge / etc.).
    ///
    /// Back-compat: when absent, `resolve_llm_for` skips levels 2 and 3
    /// of the precedence chain and falls through to per-section explicit
    /// (level 1) and the hardcoded baseline (level 4) — so every v0.26.x
    /// config continues to load identically.
    #[serde(default)]
    pub llm: LlmDefaultsConfig,
    /// v0.30.2 B3/B6 — `[warmup]` gating for cold-start side-index
    /// rebuilds. Default preserves prior behavior (always rebuild).
    #[serde(default)]
    pub warmup: WarmupConfig,
}

/// v0.30.2 B3/B6: cold-start side-index rebuild policy. Operators can
/// opt out of the unconditional Tantivy + HNSW rebuilds that ran on
/// every cold start; recovery signals (missing index, dirty marker,
/// stale rebuilding marker) still trigger a rebuild regardless.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WarmupConfig {
    /// When true (default), the cold-start warmup unconditionally
    /// rebuilds the Tantivy and HNSW side indexes from SQLite. When
    /// false, the rebuild is gated on missing-or-dirty signals only.
    #[serde(default = "default_always_rebuild_side_indexes")]
    pub always_rebuild_side_indexes: bool,
}

fn default_always_rebuild_side_indexes() -> bool {
    true
}

impl Default for WarmupConfig {
    fn default() -> Self {
        Self {
            always_rebuild_side_indexes: default_always_rebuild_side_indexes(),
        }
    }
}

/// v0.27 Track 2 config — feature flags + thresholds for the new
/// triple-overlap, N-merge, and temporal-supersede dedup paths.
///
/// Mirrors the `[ars]` shape: each knob is a bootstrap value to be
/// replaced by feedback-driven adaptation in v0.27.1+.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DedupConfig {
    /// v0.27 Track 2 #5: triple-overlap threshold for upgrading text
    /// gray-zone matches to `MergeInto`. When triple-set Jaccard
    /// (Agent C `triple_overlap_score`) ≥ this value AND text similarity
    /// is in the gray zone, the candidate is merged. Default `0.7`.
    /// bootstrap; v0.27.1+ → ablation
    #[serde(default = "default_triple_overlap_threshold")]
    pub triple_overlap_threshold: f64,
    /// v0.27 Track 2 #6: cap on N-merge fan-out. When ≥2 candidates exceed
    /// the merge threshold, the highest-similarity candidate becomes the
    /// canonical winner; up to `n_merge_max_candidates - 1` losers are
    /// folded into evidence rows in a single savepoint. Default `5`.
    /// bootstrap; v0.27.1+ → ablation
    #[serde(default = "default_n_merge_max_candidates")]
    pub n_merge_max_candidates: usize,
    /// v0.27 Track 2 #8: feature flag for temporal supersede chains.
    /// When `false` (default), text-similarity merges proceed even when
    /// temporal anchors disagree — preserving v0.26.2 behavior.
    /// When `true`, temporal conflict downgrades `MergeInto` to a
    /// `TemporalSupersede` decision (which currently degrades to
    /// `Supersede` for memories pending v0.28+ schema work).
    /// bootstrap; v0.27.1+ → ablation
    #[serde(default)]
    pub temporal_supersede_enabled: bool,
    /// v0.39 #A5: when `true`, store-time + content-mutation paths persist
    /// rule-based `(subject, predicate, object)` triples into the durable
    /// `memory_triples` table (DELETE-then-INSERT, no LLM in the write lock).
    /// v1.2 wires the read side: the dedup gray-zone path reuses the
    /// persisted set when the `memory_triple_meta` stamp validates
    /// (content hash + extractor version), recomputing + self-healing on any
    /// miss — verdicts are bit-identical either way, reuse only skips the
    /// re-extraction. Default `false` → no rows written, reader dark →
    /// behavior bit-identical to prior releases (the dedup gray-zone path
    /// recomputes triples in-flight).
    /// Not a tuning parameter — a pure on/off for the experimental fact-layer
    /// foundation; the merge gate stays `triple_overlap_threshold`.
    #[serde(default)]
    pub persist_triples: bool,
}

fn default_triple_overlap_threshold() -> f64 {
    0.7 // bootstrap; v0.27.1+ → ablation
}

fn default_n_merge_max_candidates() -> usize {
    5 // bootstrap; v0.27.1+ → ablation
}

impl Default for DedupConfig {
    fn default() -> Self {
        Self {
            triple_overlap_threshold: default_triple_overlap_threshold(),
            n_merge_max_candidates: default_n_merge_max_candidates(),
            temporal_supersede_enabled: false,
            persist_triples: false,
        }
    }
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
    /// Master switch for the LLM intelligent-merge classifier. Default `false` (mechanical jaccard/containment only).
    #[serde(default)]
    pub enabled: bool,
    /// "google" | "omlx" | "none" — default "none" means reuse query_expansion.
    #[serde(default = "default_im_provider")]
    pub provider: String,
    /// Independent Google provider config (model / api_key / endpoint), used only when `provider = "google"`.
    #[serde(default)]
    pub google: GoogleExpandConfig,
    /// Independent OMLX provider config (endpoint / model / disable_thinking), used only when `provider = "omlx"`.
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

/// Config for the v0.23 resummerize slow-channel op.
///
/// When triggered by a MergeInto cap hit, the op reads `memory_evidence`,
/// calls the configured LLM, validates the output against the Lossless
/// Compression Contract, and rewrites the canonical only on contract pass.
/// On any failure the keep-tail fallback already committed at merge time
/// remains in place, so the op is safe to retry.
///
/// `llm_backend = "inherit"` (the default) reuses `[extract].provider`.
/// Explicit values `"google"` / `"omlx"` override for this op only.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResummerizeConfig {
    /// Feature flag. Default `false` until adversarial evaluation in v0.23
    /// confirms the LLM path beats keep-tail on the scorecard.
    #[serde(default)]
    pub enabled: bool,
    /// LLM provider: `"inherit"` | `"google"` | `"omlx"`.
    #[serde(default = "default_resummerize_backend")]
    pub llm_backend: String,
    /// Max canonicals processed per slow-channel invocation.
    #[serde(default = "default_resummerize_batch_size")]
    pub batch_size: usize,
}

fn default_resummerize_backend() -> String {
    "inherit".to_string()
}

fn default_resummerize_batch_size() -> usize {
    16
}

impl Default for ResummerizeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            llm_backend: default_resummerize_backend(),
            batch_size: default_resummerize_batch_size(),
        }
    }
}

impl ResummerizeConfig {
    /// Resolve the backend: `"inherit"` defers to `[extract].provider`,
    /// everything else matches the same provider names used elsewhere.
    pub fn resolved_provider(&self, fallback_extract_provider: Provider) -> Provider {
        match self.llm_backend.to_lowercase().as_str() {
            "google" | "gemini" => Provider::Google,
            "omlx" | "local" => Provider::Omlx,
            "none" => Provider::None,
            _ => fallback_extract_provider, // "inherit" or unknown → fall back
        }
    }
}

/// Config for the v0.24+ ARS (Adaptive Retention / Synthesis).
///
/// **Cap A** (v0.24) — Concept Living Summary: background refresh of
/// `living_summary` on Concept nodes via `should_refresh_living_summary`.
///
/// **Cap B** (v0.25) — Recall-time Synthesis: opt-in `synthesize=true` param
/// on `rein_recall` / `/api/memories`. When enabled, the LLM produces a
/// short narrative over the top-N recall results and returns it as
/// `RecallSynthesisOutcome` alongside the normal results list.
///
/// `llm_backend = "inherit"` (the default) reuses `[extract].provider`.
/// Explicit values `"google"` / `"omlx"` override for this section only.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArsConfig {
    // ── Cap A ────────────────────────────────────────────────────────────────
    /// Enable Cap A concept living-summary refresh. Default `false` (opt-in).
    /// Also the global fallback the per-cluster Cap-A gate falls back to.
    #[serde(default)]
    pub concept_summary_enabled: bool,
    /// LLM provider for ARS: `"inherit"` | `"google"` | `"omlx"` | `"none"`.
    /// Default `"inherit"` defers to `[extract].provider` (see `resolved_provider`).
    #[serde(default = "default_ars_backend")]
    pub llm_backend: String,
    /// Max concepts refreshed per slow-channel Cap-A pass. Default 16; must be >= 1.
    #[serde(default = "default_ars_batch_size")]
    pub batch_size: usize,
    // ── Cap B ────────────────────────────────────────────────────────────────
    /// Enable recall-time synthesis (Cap B). Default `false` (opt-in).
    #[serde(default)]
    pub recall_synthesis_enabled: bool,
    /// Minimum number of results required before synthesis is attempted.
    /// Synthesis over 0-2 results is not useful and increases latency.
    /// Default: 3.
    #[serde(default = "default_recall_synthesis_min_results")]
    pub recall_synthesis_min_results: usize,
    /// v0.26.1: Min events per `(cluster_id, query_type)` bucket before
    /// per-cluster `useful_rate` is trusted by the per-query synthesis gate.
    /// Below this, `decide_synthesize` falls back to the global
    /// `recall_synthesis_enabled` flag. Default 10 (matches
    /// `store::adaptive::SYNTHESIS_COLD_START_N`); operators on a fresh
    /// canary may lower this to 3-5 to let the per-cluster gate fire
    /// sooner against the partial event stream a soak collects.
    #[serde(default = "default_synthesis_cold_start_n")]
    pub synthesis_cold_start_n: u64,
    /// v0.27 ARS Cap A feedback loop: Min events per `(cluster_id,
    /// query_type)` bucket before per-cluster `useful_rate` is trusted by
    /// the per-query Cap-A gate. Below this, the gate falls back to the
    /// global `concept_summary_enabled` flag. Default 10 (matches
    /// `store::adaptive::CONCEPT_SUMMARY_COLD_START_N`); operators on a
    /// fresh canary may lower this to 3-5 to let the per-cluster gate
    /// fire sooner against a partial event stream. Mirrors the v0.26.1
    /// `synthesis_cold_start_n` knob.
    #[serde(default = "default_concept_summary_cold_start_n")]
    pub concept_summary_cold_start_n: u64,
    // ── Cap C v0.26 ──────────────────────────────────────────────────────────
    /// Enable cold-tier archival summary (Cap C). Default `false` (opt-in,
    /// per spec §8 invariant 3 — flipping to true is a separate v0.26.x
    /// canary, NOT v0.26.0). When enabled, a slow-channel worker generates
    /// archival summaries for cold-tier memories and exposes them via
    /// recall when present. See `ops/cold_archive_summary.rs`.
    #[serde(default)]
    pub cold_archive_enabled: bool,
    /// Target chars for archival summary. Default 600. Bootstrap; v0.27+
    /// may make this adaptive on cold-tier length distribution.
    /// TODO: ablation.
    #[serde(default = "default_cold_archive_target_chars")]
    pub cold_archive_target_chars: usize,
    /// Batch size per slow-channel pass. Default 16. Bootstrap; v0.27+ may
    /// make this adaptive on cold-tier backlog depth.
    #[serde(default = "default_cold_archive_batch_size")]
    pub cold_archive_batch_size: usize,
    // ── v0.27.1 Track 1 — runtime LLM judge ──────────────────────────────────
    /// `[ars.llm_judge]` sub-table — default-on runtime LLM judge worker for
    /// auto-feedback on synthesis / concept-summary outputs, bounded by
    /// provider resolution, sample rates, and daily caps.
    /// J6 invariant `weight_decay_rate ∈ [0.0, 1.0]` is validated at boot
    /// via `validate_ars_llm_judge`.
    #[serde(default)]
    pub llm_judge: ArsLlmJudgeConfig,
    // ── v0.28 ARS acceleration controller ──────────────────────────────────
    /// `[ars.acceleration]` sub-table — default-on controller for ARS
    /// acceleration work. Runtime decisions still fail closed through the
    /// parameter-policy row, evidence gates, and drift checks.
    #[serde(default)]
    pub acceleration: ArsAccelerationConfig,
}

fn default_ars_backend() -> String {
    "inherit".to_string()
}

fn default_ars_batch_size() -> usize {
    16
}

fn default_recall_synthesis_min_results() -> usize {
    3
}

fn default_synthesis_cold_start_n() -> u64 {
    10
}

fn default_concept_summary_cold_start_n() -> u64 {
    10 // bootstrap; v0.27.1 → ablation (mirrors default_synthesis_cold_start_n)
}

// Cap C v0.26 defaults — see `ops/cold_archive_summary.rs::ARCHIVAL_SUMMARY_*`
// for the constants these mirror. We duplicate the literal here rather than
// import from the ops module so `config.rs` stays free of `ops` imports
// (config is a foundation crate; ops depends on it, not vice-versa).
fn default_cold_archive_target_chars() -> usize {
    600 // bootstrap; v0.27+ → adaptive
}

fn default_cold_archive_batch_size() -> usize {
    16 // bootstrap; v0.27+ → adaptive on backlog depth
}

impl Default for ArsConfig {
    fn default() -> Self {
        Self {
            concept_summary_enabled: false,
            llm_backend: default_ars_backend(),
            batch_size: default_ars_batch_size(),
            recall_synthesis_enabled: false,
            recall_synthesis_min_results: default_recall_synthesis_min_results(),
            synthesis_cold_start_n: default_synthesis_cold_start_n(),
            concept_summary_cold_start_n: default_concept_summary_cold_start_n(),
            // Cap C v0.26 — opt-in (spec §8 invariant 3)
            cold_archive_enabled: false,
            cold_archive_target_chars: default_cold_archive_target_chars(),
            cold_archive_batch_size: default_cold_archive_batch_size(),
            // v0.27.1 Track 1 / v0.28.6 default-on runtime LLM judge
            llm_judge: ArsLlmJudgeConfig::default(),
            // v0.28.6 — default-on, policy-gated acceleration
            acceleration: ArsAccelerationConfig::default(),
        }
    }
}

impl ArsConfig {
    pub fn resolved_provider(&self, fallback_extract_provider: Provider) -> Provider {
        match self.llm_backend.to_lowercase().as_str() {
            "google" | "gemini" => Provider::Google,
            "omlx" | "local" => Provider::Omlx,
            "none" => Provider::None,
            _ => fallback_extract_provider,
        }
    }
}

// ---------------------------------------------------------------------------
// v0.28 — `[ars.acceleration]` shadow acceleration config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArsAccelerationConfig {
    /// Master feature flag. Default `true`: acceleration signals are collected
    /// and the policy gate can promote eligible runtime parameters.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Keep acceleration in shadow mode. Default `false`; runtime adoption is
    /// still blocked until `ars_parameter_policy` reaches Canary mode with a
    /// positive adoption weight.
    #[serde(default)]
    pub shadow_only: bool,
}

impl Default for ArsAccelerationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            shadow_only: false,
        }
    }
}

// ---------------------------------------------------------------------------
// v0.27.1 Track 1 — `[ars.llm_judge]` runtime LLM judge config
// ---------------------------------------------------------------------------

/// `[ars.llm_judge]` config block — default-on runtime LLM judge worker for
/// auto-feedback on synthesis (Cap B) and concept-summary (Cap A) outputs.
///
/// Provider/model fields are NOT stored here; they resolve via
/// `ReinConfig::resolve_llm_for("ars.llm_judge")` per Track 2 §8 (the
/// 4-level precedence chain). This config block carries the policy /
/// rate / cost knobs that are unique to the judge worker.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArsLlmJudgeConfig {
    /// Master feature flag. Default `false` per the v0.28 charter
    /// non-goal "Do not make LLM judge default-on" — operators must
    /// explicitly opt in (incurs LLM API spend). Provider resolution,
    /// sample rates, daily caps, and policy gates kick in only after
    /// this flag is set.
    #[serde(default)]
    pub enabled: bool,
    /// Judge Cap B (synthesis) outputs. Default `true` once master is on.
    #[serde(default = "default_true")]
    pub synthesis_enabled: bool,
    /// Judge Cap A (concept-summary) outputs. Default `true` once master
    /// is on.
    #[serde(default = "default_true")]
    pub concept_summary_enabled: bool,
    /// Recall-ranking judge — deferred to v0.27.2+.
    #[serde(default)]
    pub recall_ranking_enabled: bool,
    /// Optional section-level judge input cap. When set, this overrides
    /// the resolved `[llm.{provider}].max_input_chars` for the runtime
    /// judge enqueue path.
    #[serde(default)]
    pub max_input_chars: Option<usize>,
    /// Sample-rate when cluster human-signal count is below
    /// `human_signal_threshold` (cold start). Default 1.0 (100%).
    #[serde(default = "default_judge_sample_rate_cold_start")]
    pub sample_rate_cold_start: f64,
    /// Sample-rate when cluster has ≥ `human_signal_threshold` human
    /// events (warm). Default 0.2 (20%).
    #[serde(default = "default_judge_sample_rate_warm")]
    pub sample_rate_warm: f64,
    /// Cold→warm trigger per cluster. Default 50.
    #[serde(default = "default_judge_human_signal_threshold")]
    pub human_signal_threshold: u64,
    /// `useful_rate` weight decay: `w_llm = w_thumb × weight_decay_rate`.
    /// Codex R2 P3 — default 0.3 (LLM at 30% of human signal). J6
    /// invariant requires `weight_decay_rate ∈ [0.0, 1.0]` AND finite;
    /// validated at boot via `validate_ars_llm_judge`.
    #[serde(default = "default_judge_weight_decay_rate")]
    pub weight_decay_rate: f64,
    /// Hard cap on LLM judge HTTP calls per rolling 24h. Default 10000.
    /// Worker drops events when hit (J2 invariant).
    #[serde(default = "default_judge_daily_call_cap")]
    pub daily_call_cap: u64,
    /// TTL for the synthesis-cache jsonl entry used by the manual MCP
    /// rehydration path (`rein_judge_synthesis`). Default 600s.
    #[serde(default = "default_judge_cache_ttl_secs")]
    pub cache_ttl_secs: u64,
    /// `[ars.llm_judge.nightly_cron]` sub-table — Layer 2 calibration
    /// cron policy.
    #[serde(default)]
    pub nightly_cron: ArsLlmJudgeNightlyCronConfig,
}

/// `[ars.llm_judge.nightly_cron]` sub-table — Layer 2 of the calibration
/// triangle. Re-judges a sampled subset of the last 24h synthesis events
/// with a (potentially stricter) LLM regime, and accumulates κ vs the
/// runtime judge.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArsLlmJudgeNightlyCronConfig {
    /// Master cron flag. Default `false` per the v0.28 charter
    /// non-goal — gated separately from `[ars.llm_judge].enabled`
    /// but defaults off for the same reason (shared daily-call ledger,
    /// LLM API spend).
    #[serde(default)]
    pub enabled: bool,
    /// Fraction of the last 24h synthesis events to re-judge. Default
    /// 0.2 (20%).
    #[serde(default = "default_judge_cron_sample_rate")]
    pub sample_rate: f64,
}

impl Default for ArsLlmJudgeNightlyCronConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            sample_rate: default_judge_cron_sample_rate(),
        }
    }
}

fn default_judge_sample_rate_cold_start() -> f64 {
    1.0 // bootstrap; v0.27.2+ → adaptive
}

fn default_judge_sample_rate_warm() -> f64 {
    0.2 // bootstrap; v0.27.2+ → adaptive
}

fn default_judge_human_signal_threshold() -> u64 {
    50 // bootstrap; v0.27.2+ → adaptive
}

fn default_judge_weight_decay_rate() -> f64 {
    0.3 // Codex R2 P3 — LLM at 30% of human signal
}

fn default_judge_daily_call_cap() -> u64 {
    10_000
}

fn default_judge_cache_ttl_secs() -> u64 {
    600
}

fn default_judge_cron_sample_rate() -> f64 {
    0.2
}

impl Default for ArsLlmJudgeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            synthesis_enabled: true,
            concept_summary_enabled: true,
            recall_ranking_enabled: false,
            max_input_chars: None,
            sample_rate_cold_start: default_judge_sample_rate_cold_start(),
            sample_rate_warm: default_judge_sample_rate_warm(),
            human_signal_threshold: default_judge_human_signal_threshold(),
            weight_decay_rate: default_judge_weight_decay_rate(),
            daily_call_cap: default_judge_daily_call_cap(),
            cache_ttl_secs: default_judge_cache_ttl_secs(),
            nightly_cron: ArsLlmJudgeNightlyCronConfig::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// v0.27.1 Track 2 — `[llm]` parent config + 4-level inheritance
// ---------------------------------------------------------------------------

/// `[llm]` parent config block — single source of truth for provider /
/// model defaults shared across every LLM consumer. Each consumer's
/// section can still override at level 1 (section-explicit) or level 2
/// (section-provider); the parent block is level 3. Level 4 is the
/// hardcoded baseline in code.
///
/// Back-compat: when `[llm]` is absent (every v0.26.x config), `provider`
/// defaults to "none", which causes `resolve_llm_for` to skip levels 2-3
/// of the precedence chain — the resolver falls through to level 1
/// (per-section) and level 4 (hardcoded), preserving v0.26.x semantics
/// exactly.
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct LlmDefaultsConfig {
    /// Global default provider name: `"google"` | `"omlx"` | `"none"` |
    /// absent (`""`). Empty string + missing field both mean "no `[llm]`
    /// provider was set" — back-compat path.
    #[serde(default)]
    pub provider: String,
    /// Provider-agnostic temperature override. Optional; consumers fall
    /// back to their own internal default when `None`.
    #[serde(default)]
    pub temperature: Option<f64>,
    /// Provider-agnostic request timeout override (ms). Optional.
    #[serde(default)]
    pub request_timeout_ms: Option<u64>,
    /// Provider-agnostic max-retries override. Optional.
    #[serde(default)]
    pub max_retries: Option<u32>,
    /// `[llm.google]` provider sub-table.
    #[serde(default)]
    pub google: LlmProviderTable,
    /// `[llm.omlx]` provider sub-table.
    #[serde(default)]
    pub omlx: LlmProviderTable,
}

/// Per-provider sub-table inside `[llm.{provider}]` (or
/// `[{section}.{provider}]` overrides). Fields are optional so the
/// resolver can detect "not set at this precedence level" and walk
/// further.
///
/// **Provider-scoped fields walk as a unit** (Codex R5 P2): once
/// `provider` is selected at some precedence level, the resolver reads
/// `model` / `api_key_env` / `endpoint` / `max_input_chars` from the
/// SELECTED provider's sub-table at that level (or walks back through
/// level 2/3 with the same provider). They never independently fall
/// over to the OTHER provider's sub-table.
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct LlmProviderTable {
    /// Model name (e.g. `"gemini-3.1-flash-lite-preview"`,
    /// `"gemini-3.1-pro"`, `"default"`).
    #[serde(default)]
    pub model: Option<String>,
    /// Environment variable that holds the provider's API key. Resolver
    /// passes this through unchanged; the consumer reads `std::env::var`.
    /// Convention mirrors the pre-Track-2 pattern of `GEMINI_API_KEY`.
    #[serde(default)]
    pub api_key_env: Option<String>,
    /// Provider endpoint override (proxy / mirror / local server).
    #[serde(default)]
    pub endpoint: Option<String>,
    /// Max prompt characters. `0` is reserved for known 1M-token Gemini
    /// models; the consumer rejects `0` for any other model and falls
    /// back to a 16K safety limit.
    #[serde(default)]
    pub max_input_chars: Option<usize>,
}

/// Source level a `ResolvedLlmConfig` came from — surfaced for
/// `rein doctor` diagnostics + Layer-3 telemetry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecedenceSource {
    /// Level 1: `[{section}.{provider}].{field}` — section-explicit.
    SectionExplicit,
    /// Level 2: `[{section}].provider` selects → `[llm.{provider}]`
    /// scoped fields read at level 3 with the section-chosen provider.
    SectionProvider,
    /// Level 3: `[llm].provider` + `[llm.{provider}]`.
    GlobalDefault,
    /// Level 4: hardcoded baseline in code.
    HardcodedFallback,
}

/// Resolved LLM config for a given consumer section. Returned by
/// `ReinConfig::resolve_llm_for(section)`.
///
/// Fields mirror the existing per-section LLM blob shape so call-site
/// migration is mostly mechanical. `source` is reported for telemetry.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedLlmConfig {
    /// Resolved provider — `Provider::None` means "no LLM configured for
    /// this section" (consumer should disable that path).
    pub provider: Provider,
    /// Resolved model name. Empty when `provider == None`.
    pub model: String,
    /// Resolved env-var name for the API key (e.g. `"GEMINI_API_KEY"`).
    /// `None` when not applicable (OMLX / None).
    pub api_key_env: Option<String>,
    /// Resolved endpoint (proxy / mirror / local server).
    pub endpoint: String,
    /// Max prompt characters. `0` means "no truncation" — only valid for
    /// known 1M-token Gemini models; consumer enforces.
    pub max_input_chars: usize,
    /// Optional temperature override (consumer falls back to its own
    /// default when `None`).
    pub temperature: Option<f64>,
    /// Optional request-timeout override in ms.
    pub request_timeout_ms: Option<u64>,
    /// Optional max-retries override.
    pub max_retries: Option<u32>,
    /// Which precedence level supplied the dominant `provider` choice.
    pub source: PrecedenceSource,
    /// Consumer section name (echoed back for logging).
    pub section: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatabaseConfig {
    /// SQLite database path. `"auto"` (default) resolves to `~/.rein/memories.db`;
    /// `:memory:` for an in-memory DB. Overridden by the `REIN_DB` env var.
    pub path: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingConfig {
    /// Embedding backend: `"google"` (default), `"omlx"`, or `"none"` (disables embedding).
    pub provider: String,
    /// Embedding vector length (default 3072). Must match the vector index dimensionality.
    pub dimensions: usize,
    /// `[embedding.google]` — Gemini embedding settings (used when `provider = "google"`).
    pub google: GoogleEmbeddingConfig,
    /// `[embedding.omlx]` — local OpenAI-compatible embedding settings (used when `provider = "omlx"`).
    pub omlx: OmlxEmbeddingConfig,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoogleEmbeddingConfig {
    /// Gemini embedding model name. Default: "gemini-embedding-001".
    pub model: String,
    /// Gemini API key; falls back to the `GEMINI_API_KEY` env var. When unset the
    /// Google embedder is disabled (no embeddings produced).
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
    /// OpenAI-compatible embedding endpoint. Default: "http://localhost:8000/v1".
    pub endpoint: String,
    /// OMLX embedding model name sent to the endpoint. Default: "default".
    #[serde(default = "default_omlx_model")]
    pub model: String,
}

fn default_omlx_model() -> String {
    "default".to_string()
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchConfig {
    /// Reciprocal Rank Fusion rank constant `k` (default 60.0). Larger values flatten
    /// the contribution of rank position.
    pub rrf_k: f64,
    /// RRF weight applied to the BM25/FTS channel. Default: 0.3.
    pub rrf_fts_weight: f64,
    /// RRF weight applied to the vector (HNSW) channel. Default: 0.7.
    pub rrf_vec_weight: f64,
    /// Fusion method: "rrf" (Reciprocal Rank Fusion) or "cc" (Convex Combination).
    /// CC normalizes scores to [0,1] and blends with alpha; often more accurate (Bruch 2023).
    #[serde(default = "default_fusion_method")]
    pub fusion_method: String,
    /// Alpha for CC fusion: score = alpha * sparse + (1-alpha) * dense. Default 0.5.
    #[serde(default = "default_cc_alpha")]
    pub cc_alpha: f64,
    /// Cosine-similarity threshold in [0.0, 1.0] for store-time dedup and auto-linking.
    /// Default: 0.70.
    pub dedup_similarity: f64,
    /// Lookback window (days) for store-time dedup candidate matching. Default: 7.
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
    /// Max tokens per chunk (estimated at ~4 chars/token). Default: 512.
    pub max_tokens: usize,
    /// Percent of `max_tokens` carried over from the end of the previous chunk. Default: 10.
    pub overlap_percent: usize,
    /// Whether to prepend topic/summary metadata to chunk text before embedding. Default: true.
    pub metadata_prefix: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SyncConfig {
    /// Enable Supermemory cross-source recall (skipped in fast-path recall). Default: true.
    pub supermemory_enabled: bool,
    /// Enable auto-memory file recall over `auto_memory_glob` (skipped in fast-path recall). Default: true.
    pub auto_memory_enabled: bool,
    /// Glob for auto-memory markdown files scanned during recall.
    /// Default: "~/.claude/projects/*/memory/**/*.md".
    pub auto_memory_glob: String,
    /// Supermemory API key; falls back to the `SUPERMEMORY_CC_API_KEY` env var.
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
    /// Base per-memory exponential decay rate. Default 0.06. The effective
    /// per-row rate is `base_lambda * importance.decay_factor()`.
    pub base_lambda: f64,
    /// Long-term-memory decay shaping factor. Default 0.8.
    pub ltm_beta: f64,
    /// Short-term-memory decay shaping factor. Default 1.2.
    pub stm_beta: f64,
    /// Nominal decay/GC cadence in hours. Default 24.
    pub interval_hours: u64,
    /// Strength floor below which weak STM memories are pruned during `gc`.
    /// Default 0.05; used unless overridden by the `--threshold` flag.
    pub prune_threshold: f64,
    /// Access count at which an STM memory is eligible for STM->LTM promotion. Default 5.
    pub stm_to_ltm_access_count: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    /// Render MCP tool output in compact form (sets the runtime compact flag).
    /// Default false.
    pub compact: bool,
    /// Enable the HTTP/SSE server surface. Default false.
    pub sse_enabled: bool,
    /// TCP port for the HTTP/SSE (and GUI) server. Default 8680.
    pub sse_port: u16,
    /// Bind host for the HTTP/SSE server. Default "127.0.0.1". A wildcard bind
    /// (0.0.0.0, ::, *) requires `allowed_hosts` to be set.
    pub sse_bind: String,
    /// Explicit HTTP auth policy (loopback_only / bearer_required / oauth /
    /// public). Default None; resolved by `resolve_auth_policy`.
    #[serde(default)]
    pub auth: Option<AuthPolicyConfig>,
    /// Externally reachable base URL, used to build OAuth/metadata endpoints.
    /// Default None (falls back to the local bind address).
    #[serde(default)]
    pub public_url: Option<String>,
    /// Run background warmup/side-index repair when service surfaces start.
    #[serde(default = "default_server_background_warmup")]
    pub background_warmup: bool,
    /// Opt stdio MCP startup into background warmup.
    ///
    /// Default false because MCP clients such as Codex may start one stdio
    /// process per subagent, and each background warmup can contend on local
    /// side-index writer locks.
    #[serde(default)]
    pub stdio_background_warmup: bool,
    /// Serve the embedded Neural Wiki GUI on non-`/mcp` paths. Default false.
    #[serde(default)]
    pub gui_enabled: bool,
    /// v0.27.3 F5/C3: optional explicit Host-header allowlist used by the
    /// HTTP guard. Required when binding to a wildcard address
    /// (`0.0.0.0`, `::`, `*`) to defend against DNS-rebinding. When `None`
    /// and the bind is specific (loopback or a concrete LAN IP), the
    /// allowlist is derived from the bind host.
    #[serde(default)]
    pub allowed_hosts: Option<Vec<String>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthPolicyConfig {
    LoopbackOnly,
    BearerRequired,
    #[serde(rename = "oauth")]
    OAuth,
    Public,
}

impl From<AuthPolicyConfig> for crate::auth::AuthPolicy {
    fn from(value: AuthPolicyConfig) -> Self {
        match value {
            AuthPolicyConfig::LoopbackOnly => Self::LoopbackOnly,
            AuthPolicyConfig::BearerRequired => Self::BearerRequired,
            AuthPolicyConfig::OAuth => Self::OAuth,
            AuthPolicyConfig::Public => Self::Public,
        }
    }
}

/// True when an HTTP bind address is unambiguously local loopback.
pub(crate) fn is_loopback_bind_host(bind: &str) -> bool {
    matches!(bind, "127.0.0.1" | "::1" | "localhost")
}

// v0.35 Phase 3: `ServerConfig::loopback_unauth_requested` and the
// `allow_unauthenticated_loopback` field it gated were removed. The legacy
// bool no longer participates in runtime auth resolution — callers must use
// `[server].auth` explicitly. Configs that still carry the legacy bool are
// transformed by `migrate_legacy_server_auth` at TOML load time (see
// `merge_toml`). Lookup of "what is the runtime policy" is now solely
// `ReinConfig::resolve_auth_policy`.

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HooksConfig {
    /// Minimum transcript turns (or lines, when turn count is unknown) before
    /// a session is extracted. Default 20.
    pub min_turns: usize,
    /// Number of lines kept before each signal-keyword hit when extracting
    /// signal windows. Default 3.
    pub context_before: usize,
    /// Number of lines kept after each signal-keyword hit when extracting
    /// signal windows. Default 1.
    pub context_after: usize,
    /// Cap on memories persisted per hook session. Default 10.
    pub max_items_per_session: usize,
    /// `[hooks.codex]` — Codex-specific hook behavior (context injection and
    /// guardrails).
    #[serde(default)]
    pub codex: CodexHooksConfig,
    /// Keywords that mark transcript lines as memory-worthy signals. Defaults
    /// to a built-in EN/CJK list (see `default_signal_keywords`).
    #[serde(default = "default_signal_keywords")]
    pub signal_keywords: Vec<String>,
    /// Directory for hook session buffers. Default "auto" (resolves to the
    /// database parent dir, e.g. ~/.rein/).
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

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CodexHooksConfig {
    /// Add bounded Rein memory context to Codex UserPromptSubmit.
    /// Default false because it changes model context.
    #[serde(default)]
    pub inject_prompt_context: bool,
    /// Add bounded project context to Codex SessionStart.
    /// Default false because it changes model context.
    #[serde(default)]
    pub inject_session_context: bool,
    /// Enable conservative PreToolUse / PermissionRequest deny rules.
    #[serde(default = "default_codex_guardrails_enabled")]
    pub guardrails_enabled: bool,
    /// Hard cap for injected additionalContext text.
    #[serde(default = "default_codex_max_additional_context_chars")]
    pub max_additional_context_chars: usize,
}

fn default_buffer_dir() -> String {
    "auto".to_string()
}

fn default_buffer_flush_threshold() -> usize {
    50000 // ~12K-25K tokens, triggers ~2-4 times in a long session
}

fn default_server_background_warmup() -> bool {
    true
}

fn default_codex_guardrails_enabled() -> bool {
    true
}

fn default_codex_max_additional_context_chars() -> usize {
    1200
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExtractConfig {
    /// LLM provider for memory extraction: "google", "omlx", or "none".
    /// Default: "google". "none" disables LLM labeling (rule-based fallback only).
    pub provider: String,
    /// `[extract.google]` — Gemini extraction backend (used when provider = "google").
    pub google: GoogleExtractConfig,
    /// `[extract.omlx]` — local OpenAI-compatible extraction backend (used when provider = "omlx").
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
    /// Gemini model id. Default: "gemini-3.1-flash-lite-preview".
    pub model: String,
    /// Gemini API key. Default: None (populated from GEMINI_API_KEY at load time).
    #[serde(default)]
    pub api_key: Option<String>,
    /// API endpoint override (for proxies in China, etc.)
    /// Default: "https://generativelanguage.googleapis.com"
    #[serde(default = "default_google_endpoint")]
    pub endpoint: String,
    /// Max input characters. 0 = no truncation (default for gemini-3.1-flash-lite-preview which supports 1M tokens).
    #[serde(default)]
    pub max_input_chars: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OmlxExtractConfig {
    /// OpenAI-compatible base URL for the local model. Default: "http://localhost:11434/v1".
    pub endpoint: String,
    /// Local model name passed to the OpenAI-compatible endpoint. Default: "default".
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
    /// `[query_expansion.google]` — Gemini expansion backend (used when provider = "google").
    #[serde(default)]
    pub google: GoogleExpandConfig,
    /// `[query_expansion.omlx]` — local OpenAI-compatible expansion backend (used when provider = "omlx").
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
    /// Gemini model id for query expansion. Default: "gemini-3.1-flash-lite-preview".
    #[serde(default = "default_expand_google_model")]
    pub model: String,
    /// Gemini API key. Default: None (falls back to GEMINI_API_KEY).
    #[serde(default)]
    pub api_key: Option<String>,
    /// API endpoint override (for proxies in China, etc.)
    /// Default: "https://generativelanguage.googleapis.com"
    #[serde(default = "default_google_endpoint")]
    pub endpoint: String,
}

fn default_expand_google_model() -> String {
    "gemini-3.1-flash-lite-preview".to_string()
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OmlxExpandConfig {
    /// OpenAI-compatible base URL for the local expansion model. Default: "http://localhost:8000/v1".
    #[serde(default = "default_omlx_expand_endpoint")]
    pub endpoint: String,
    /// Local model name passed to the OpenAI-compatible endpoint. Default: "default".
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
            auth: None,
            public_url: None,
            background_warmup: default_server_background_warmup(),
            stdio_background_warmup: false,
            gui_enabled: false,
            // v0.27.3 F5/C3: when [server].sse_bind is a wildcard
            // (0.0.0.0, ::, *, empty), startup refuses unless allowed_hosts
            // is set. None means "derive allowlist from bind", which is
            // the safe default for specific (loopback or LAN-IP) binds.
            allowed_hosts: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(default)]
pub struct ProxyConfig {
    /// TCP port the recording proxy listens on. Default: 8690.
    pub port: u16,
    /// Address the proxy binds to. Default: "127.0.0.1" (loopback only).
    pub bind: String,
    /// Upstream forwarded to for Anthropic API traffic. Default: "https://api.anthropic.com".
    pub anthropic_upstream: String,
    /// Upstream forwarded to for OpenAI API traffic. Default: "https://api.openai.com".
    pub openai_upstream: String,
    /// Upstream forwarded to for ChatGPT backend traffic. Default: "https://chatgpt.com/backend-api".
    pub chatgpt_upstream: String,
    /// Upstream forwarded to for Codex (first-party) traffic. Default: "https://chatgpt.com/backend-api/codex".
    pub codex_upstream: String,
    /// Gate for record-only memory extraction from proxied responses. Default: true.
    pub extract_enabled: bool,
    /// Minimum response length (chars) before a response is considered for extraction. Default: 220.
    pub store_min_chars: usize,
    /// Minimum sentence-significance score required to extract when no strong signal is present. Default: 3.
    pub store_min_score: u32,
    /// Max upstream retry attempts for idempotent/transient failures. Default: 2.
    pub max_retries: u32,
    /// Base backoff in milliseconds; exponential per retry attempt. Default: 500.
    pub retry_base_ms: u64,
    /// Max buffered request body in bytes. Default: 1_048_576 (1 MiB).
    pub max_request_body: usize,
    /// Max buffered non-streaming response in bytes. Default: 1_048_576 (1 MiB).
    pub max_response_buffer: usize,
    /// Max buffered SSE stream content in bytes. Default: 1_048_576 (1 MiB).
    pub max_sse_buffer: usize,
    /// Max concurrent proxy-side extraction tasks. Default: 4.
    pub max_concurrent_extractions: usize,
    /// Opt-in to serving unauthenticated loopback requests without REIN_PROXY_TOKEN.
    /// Default: false (default-deny; operators must opt in or set the token).
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
            // v0.27.3 F5/C1: default-deny. Operators must opt in to
            // unauthenticated loopback or set REIN_PROXY_TOKEN. The doctor
            // template already writes `false` (see doctor.rs:1767).
            allow_unauthenticated_loopback: false,
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
            codex: CodexHooksConfig::default(),
            signal_keywords: default_signal_keywords(),
            buffer_dir: default_buffer_dir(),
            buffer_flush_threshold: default_buffer_flush_threshold(),
        }
    }
}

impl Default for CodexHooksConfig {
    fn default() -> Self {
        Self {
            inject_prompt_context: false,
            inject_session_context: false,
            guardrails_enabled: default_codex_guardrails_enabled(),
            max_additional_context_chars: default_codex_max_additional_context_chars(),
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
            selection_limit: 2,
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
    pub fn resolve_auth_policy(&self) -> anyhow::Result<crate::auth::AuthPolicy> {
        if let Some(policy) = self.server.auth {
            let policy = crate::auth::AuthPolicy::from(policy);
            if matches!(
                policy,
                crate::auth::AuthPolicy::BearerRequired | crate::auth::AuthPolicy::OAuth
            ) && nonempty_env("REIN_HTTP_TOKEN").is_none()
            {
                let purpose = match policy {
                    crate::auth::AuthPolicy::BearerRequired => "bearer auth",
                    crate::auth::AuthPolicy::OAuth => "OAuth owner approval",
                    _ => unreachable!(),
                };
                anyhow::bail!(
                    "[server].auth = \"{}\" requires REIN_HTTP_TOKEN for {purpose}",
                    policy.as_str()
                );
            }
            return Ok(policy);
        }

        let has_http_token = nonempty_env("REIN_HTTP_TOKEN").is_some();
        if has_http_token {
            return Ok(crate::auth::AuthPolicy::BearerRequired);
        }

        // v0.35 Phase 3: the implicit `allow_unauthenticated_loopback` bool
        // was removed. If neither `[server].auth` nor `REIN_HTTP_TOKEN` is
        // set, the only valid posture is to refuse to start — operators
        // must choose an explicit policy. Legacy configs that still carry
        // the bool are transformed by `migrate_legacy_server_auth` at TOML
        // load time, so this branch is unreachable for them.
        Err(anyhow::anyhow!(
            "REIN_HTTP_TOKEN must be set for HTTP/SSE access on '{}'. \
             Set REIN_HTTP_TOKEN=<secret>, or set [server].auth explicitly: \
             \"public\" for non-discoverable read-only tunnels, \
             \"loopback_only\" for strict local-only access, \
             \"bearer_required\" for token auth (paired with REIN_HTTP_TOKEN), \
             or \"oauth\" for multi-user remote auth.",
            self.server.sse_bind
        ))
    }

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
        // v1.0 forward-compat guard: refuse a config written by a newer rein.
        // (Unversioned `0` and the current version both pass — see
        // `CURRENT_CONFIG_VERSION`.) Mirrors the SQLite schema downgrade guard.
        if self.config_version > CURRENT_CONFIG_VERSION {
            anyhow::bail!(
                "config_version {} is newer than this rein build understands (max {}); upgrade rein or remove the config_version key",
                self.config_version,
                CURRENT_CONFIG_VERSION
            );
        }
        validate_provider_name("embedding.provider", &self.embedding.provider)?;
        // Codex R4 P2 fix — extract / query_expansion / search.llm_reranker
        // accept "inherit" sentinel (in addition to google/omlx/none) so
        // operators can opt INTO `[llm]` global override. Without this
        // there's no way to override default-pinned sections without
        // writing each one explicitly.
        validate_provider_name_or_inherit("extract.provider", &self.extract.provider)?;
        validate_provider_name_or_inherit(
            "query_expansion.provider",
            &self.query_expansion.provider,
        )?;
        validate_provider_name_or_inherit("search.llm_reranker", &self.search.llm_reranker)?;
        validate_provider_name_or_inherit("async_memory.provider", &self.async_memory.provider)?;
        validate_provider_name_or_inherit(
            "resummerize.llm_backend",
            &self.resummerize.llm_backend,
        )?;
        validate_provider_name_or_inherit("ars.llm_backend", &self.ars.llm_backend)?;

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
        if self.resummerize.batch_size == 0 {
            anyhow::bail!("resummerize.batch_size must be >= 1");
        }
        if self.ars.batch_size == 0 {
            anyhow::bail!("ars.batch_size must be >= 1");
        }
        // v0.27.1 J6 invariant — `weight_decay_rate` must be finite + in
        // [0.0, 1.0] so `w_llm = w_thumb × weight_decay_rate` enforces both
        // ordering (`w_llm ≤ w_thumb`) and non-negativity (`w_llm ≥ 0`).
        // Codex R8 P2 fix — v0 only checked `> 1.0`, accepting negative
        // values that would make `w_llm` subtract judge hits from the
        // human signal. NaN / non-finite also rejected.
        validate_ars_llm_judge(&self.ars.llm_judge)?;

        // v0.27.1 Track 2 — validate the optional `[llm]` provider name
        // when present (empty string = absent, treated as back-compat).
        if !self.llm.provider.is_empty() {
            validate_provider_name("llm.provider", &self.llm.provider)?;
        }

        Ok(())
    }

    /// v0.27.1 Track 2 — resolve the effective LLM config for a given
    /// consumer section.
    ///
    /// `section` is the dotted path identifying the consumer; valid
    /// values per spec §8.5:
    ///
    /// - `"extract"`, `"extract.async_memory"`, `"extract.intelligent_merge"`,
    ///   `"extract.dedup"`
    /// - `"query_expansion"`
    /// - `"search.llm_reranker"`
    /// - `"ars.recall_synthesis"`, `"ars.concept_summary"`,
    ///   `"ars.cold_archive"`
    /// - `"resummerize"`
    /// - `"ars.llm_judge"`, `"ars.llm_judge.nightly_cron"`
    ///
    /// Walks the 4-level precedence chain per spec §8.1. Returns
    /// `Err(Config(...))` only when:
    /// - the section name is unknown, or
    /// - the resolved provider is `Google`/`Omlx` but no provider
    ///   sub-table at any walked level supplied a `model` (Codex R5 P2
    ///   fail-fast — silent cross-provider corruption beats nothing).
    ///
    /// **Provider-scoped fields walk as a unit**: once `provider` is
    /// chosen at a precedence level, the resolver reads
    /// `model` / `api_key_env` / `endpoint` / `max_input_chars` from the
    /// SELECTED provider's sub-table at that level (or walks back through
    /// level 2 → level 3 with the same provider). They never
    /// independently fall over to the OTHER provider's sub-table.
    pub fn resolve_llm_for(&self, section: &str) -> crate::types::ReinResult<ResolvedLlmConfig> {
        resolve_llm_for_impl(self, section)
    }

    /// Open a SqliteStore with the current config's model and dimensions.
    ///
    /// v0.22 P1: when the caller is inside a Tokio runtime AND the DB is
    /// file-backed AND `REIN_ASYNC_P1` is not `0`, attach a process-level
    /// connection pool keyed by path. `recall`'s 3-channel fanout uses the
    /// pool to elide per-channel schema-init + embedding-model checks.
    /// Plain non-tokio callers (for example `#[test]` or std-thread code)
    /// and `:memory:` DBs bypass the pool entirely and use the pre-v0.22
    /// path. `#[tokio::main]` CLI entrypoints do run inside a runtime, so
    /// they follow the same best-effort attach path as async services.
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
        // Non-tokio callers would have the pool ignored by recall.rs via
        // `Handle::try_current()` anyway; skipping cache+build here saves
        // the setup cost and keeps those paths bit-identical to the
        // pre-v0.22 behavior.
        if tokio::runtime::Handle::try_current().is_err() {
            return Ok(store);
        }
        match pool_for_path(&db_path) {
            Ok(pool) => Ok(store.with_pool(pool)),
            Err(err) => {
                tracing::warn!(
                    db_path = %db_path.display(),
                    err = %err,
                    "open_store pool initialization failed; falling back to single connection"
                );
                Ok(store)
            }
        }
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

    /// v0.31.1 candidate (D2 from `v0.31.1-oauth-deeper`): stable identity
    /// string for scoping the process-global OAuth bearer cache.
    ///
    /// The bearer cache (`auth::oauth::bearer_cache`) lives in a `static`
    /// and is therefore shared by every `ReinConfig` opened in the same
    /// process.  Keying entries solely on `sha256(token)` would let a
    /// token cached against DB A return `true` against DB B for the
    /// cache TTL — a theoretical cross-vault confusion surface that the
    /// R5 codex audit flagged.  This identity threads through into the
    /// cache key (`auth::oauth::bearer_cache_key`) so the two configs
    /// see disjoint cache namespaces.
    ///
    /// Prefer the canonicalised path (resolves symlinks) so the same
    /// physical DB reached through different access paths still hits
    /// the same cache namespace.  If canonicalisation fails — typically
    /// because the file doesn't exist yet during init — fall back to
    /// the resolved (but non-canonical) path string.  Either form is
    /// stable across the process lifetime, which is all the cache key
    /// requires.
    pub fn stable_db_identity(&self) -> String {
        let resolved = self.resolve_db_path();
        match resolved.canonicalize() {
            Ok(canonical) => canonical.to_string_lossy().into_owned(),
            Err(_) => resolved.to_string_lossy().into_owned(),
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

#[cfg(test)]
static FAIL_NEXT_POOL_FOR_PATH: AtomicBool = AtomicBool::new(false);

fn pool_cache() -> &'static Mutex<HashMap<PathBuf, Weak<crate::store::pool::ConnPool>>> {
    OPEN_STORE_POOL_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn pool_for_path(db_path: &Path) -> crate::types::ReinResult<Arc<crate::store::pool::ConnPool>> {
    #[cfg(test)]
    if FAIL_NEXT_POOL_FOR_PATH.swap(false, Ordering::SeqCst) {
        return Err(crate::types::ReinError::Config(
            "synthetic pool init failure for test".into(),
        ));
    }

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

/// Read-only access to a path's pool metrics, IF a live pool exists for it.
/// Used by `doctor::check_pool_saturation` to surface saturation counts
/// without forcing a fresh pool open (which would race the recall fanout's
/// own first-touch). Returns `None` when no live pool is registered for the
/// path — typically means no recall traffic has hit this DB this process.
pub fn pool_metrics_for_path(
    db_path: &Path,
) -> crate::types::ReinResult<crate::store::pool::PoolMetrics> {
    let map = pool_cache()
        .lock()
        .expect("open_store pool cache mutex poisoned");
    let key = std::path::absolute(db_path).unwrap_or_else(|_| db_path.to_path_buf());
    let weak = map
        .get(&key)
        .ok_or_else(|| crate::types::ReinError::Config("no pool registered for path".into()))?;
    let pool = weak
        .upgrade()
        .ok_or_else(|| crate::types::ReinError::Config("pool dropped".into()))?;
    Ok(pool.metrics())
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

// ---------------------------------------------------------------------------
// v0.27.1 Track 1 — `[ars.llm_judge]` validation (J6 invariant)
// ---------------------------------------------------------------------------

/// Validate `[ars.llm_judge]`. Enforces J6: `weight_decay_rate` finite
/// and in `[0.0, 1.0]`. Codex R8 P2 fix — v0 only checked `> 1.0`,
/// accepting negative values that would make `w_llm` subtract judge
/// hits from the human signal. NaN / non-finite also rejected.
pub fn validate_ars_llm_judge(cfg: &ArsLlmJudgeConfig) -> anyhow::Result<()> {
    if !cfg.weight_decay_rate.is_finite() || !(0.0..=1.0).contains(&cfg.weight_decay_rate) {
        anyhow::bail!(
            "ars.llm_judge.weight_decay_rate must be finite and in [0.0, 1.0], got {}",
            cfg.weight_decay_rate
        );
    }
    if !(0.0..=1.0).contains(&cfg.sample_rate_cold_start) || !cfg.sample_rate_cold_start.is_finite()
    {
        anyhow::bail!(
            "ars.llm_judge.sample_rate_cold_start must be finite and in [0.0, 1.0], got {}",
            cfg.sample_rate_cold_start
        );
    }
    if !(0.0..=1.0).contains(&cfg.sample_rate_warm) || !cfg.sample_rate_warm.is_finite() {
        anyhow::bail!(
            "ars.llm_judge.sample_rate_warm must be finite and in [0.0, 1.0], got {}",
            cfg.sample_rate_warm
        );
    }
    if cfg.nightly_cron.enabled
        && (!(0.0..=1.0).contains(&cfg.nightly_cron.sample_rate)
            || !cfg.nightly_cron.sample_rate.is_finite())
    {
        anyhow::bail!(
            "ars.llm_judge.nightly_cron.sample_rate must be finite and in [0.0, 1.0], got {}",
            cfg.nightly_cron.sample_rate
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// v0.27.1 Track 2 — 4-level LLM config inheritance resolver
// ---------------------------------------------------------------------------

/// Section identity — the resolver knows how to read each consumer's
/// pre-Track-2 explicit fields (level 1) and the hardcoded baseline
/// (level 4) for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LlmSection {
    Extract,
    ExtractAsyncMemory,
    ExtractIntelligentMerge,
    ExtractDedup,
    QueryExpansion,
    SearchLlmReranker,
    ArsRecallSynthesis,
    ArsConceptSummary,
    ArsColdArchive,
    Resummerize,
    ArsLlmJudge,
    ArsLlmJudgeNightlyCron,
}

impl LlmSection {
    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "extract" => Self::Extract,
            "extract.async_memory" => Self::ExtractAsyncMemory,
            "extract.intelligent_merge" => Self::ExtractIntelligentMerge,
            "extract.dedup" => Self::ExtractDedup,
            "query_expansion" => Self::QueryExpansion,
            "search.llm_reranker" => Self::SearchLlmReranker,
            "ars.recall_synthesis" => Self::ArsRecallSynthesis,
            "ars.concept_summary" => Self::ArsConceptSummary,
            "ars.cold_archive" => Self::ArsColdArchive,
            "resummerize" => Self::Resummerize,
            "ars.llm_judge" => Self::ArsLlmJudge,
            "ars.llm_judge.nightly_cron" => Self::ArsLlmJudgeNightlyCron,
            _ => return None,
        })
    }
}

/// Level-1 (section-explicit) snapshot. `provider` may be `"inherit"`
/// for the slow-channel sections (`resummerize`, `ars.*`,
/// `async_memory`) or absent (`""`) for all v0.27.1-new sections.
struct SectionExplicit<'a> {
    /// `Some(name)` when the section ITSELF carries a `provider = "..."`
    /// field. Lowercased. Empty string treated as `None`.
    provider: Option<&'a str>,
    /// Whether `provider == "inherit"` — slow-channel "fall through" hint.
    inherit: bool,
    /// Section-level, provider-agnostic max input override.
    max_input_chars: Option<usize>,
    /// Section's own provider sub-tables (legacy v0.26.x shape).
    google_model: Option<&'a str>,
    google_api_key_env: Option<&'static str>,
    google_endpoint: Option<&'a str>,
    google_max_input_chars: Option<usize>,
    omlx_model: Option<&'a str>,
    omlx_endpoint: Option<&'a str>,
    omlx_max_input_chars: Option<usize>,
}

impl<'a> SectionExplicit<'a> {
    fn empty() -> Self {
        Self {
            provider: None,
            inherit: false,
            max_input_chars: None,
            google_model: None,
            google_api_key_env: None,
            google_endpoint: None,
            google_max_input_chars: None,
            omlx_model: None,
            omlx_endpoint: None,
            omlx_max_input_chars: None,
        }
    }
}

/// Snapshot the section-explicit (level-1) fields for a given consumer.
/// Centralizes the legacy field-path knowledge so the resolver core stays
/// generic.
///
/// For sections that didn't exist before v0.27.1 (`ars.llm_judge`,
/// `ars.llm_judge.nightly_cron`, dotted `extract.*` virtual sections),
/// returns `SectionExplicit::empty()` — they have no level-1 explicit
/// shape; resolution falls through to level 2/3/4.
fn section_explicit<'a>(cfg: &'a ReinConfig, sec: LlmSection) -> SectionExplicit<'a> {
    use LlmSection::*;
    // Codex R4 P2 fix — revert the R2 "default-suppress" heuristic. We
    // can't reliably distinguish "operator wrote `provider = \"google\"`
    // to opt out of a global OMLX" from "merge_toml filled the compiled
    // default". The opt-out semantics requires the explicit value to
    // win.
    //
    // To still let operators globally override `[llm]` for v0.26.x
    // sections, accept the explicit sentinel `"inherit"` on extract /
    // query_expansion / search.llm_reranker — same pattern already used
    // by `async_memory.provider` and `resummerize.llm_backend`. Inherit
    // means "fall through to `[llm]`"; any other value (including
    // compiled-default `"google"`) is treated as section-explicit.
    let extract_inherit = cfg.extract.provider.eq_ignore_ascii_case("inherit");
    let query_expansion_inherit = cfg.query_expansion.provider.eq_ignore_ascii_case("inherit");
    let reranker_inherit = cfg.search.llm_reranker.eq_ignore_ascii_case("inherit");
    match sec {
        Extract => SectionExplicit {
            provider: if extract_inherit {
                None
            } else {
                Some(cfg.extract.provider.as_str())
            },
            inherit: extract_inherit,
            max_input_chars: None,
            google_model: nonempty(&cfg.extract.google.model),
            google_api_key_env: Some("GEMINI_API_KEY"),
            google_endpoint: nonempty(&cfg.extract.google.endpoint),
            google_max_input_chars: Some(cfg.extract.google.max_input_chars),
            omlx_model: nonempty(&cfg.extract.omlx.model),
            omlx_endpoint: nonempty(&cfg.extract.omlx.endpoint),
            omlx_max_input_chars: Some(cfg.extract.omlx.max_input_chars),
        },
        ExtractAsyncMemory => {
            // `[async_memory].provider` may be "inherit" → walk to
            // `[extract]` semantics; the resolver core treats that as a
            // level-1 skip (move to level 2/3).
            let prov = cfg.async_memory.provider.to_lowercase();
            let inherit = prov == "inherit";
            SectionExplicit {
                provider: if inherit {
                    None
                } else {
                    Some(cfg.async_memory.provider.as_str())
                },
                inherit,
                max_input_chars: None,
                // No async-memory-specific provider sub-table; reuse
                // `[extract.{provider}]` (matches the v0.26.x semantic
                // implemented in `extract/llm.rs::create_memory_worker_extractor`).
                google_model: nonempty(&cfg.extract.google.model),
                google_api_key_env: Some("GEMINI_API_KEY"),
                google_endpoint: nonempty(&cfg.extract.google.endpoint),
                google_max_input_chars: Some(cfg.extract.google.max_input_chars),
                omlx_model: nonempty(&cfg.extract.omlx.model),
                omlx_endpoint: nonempty(&cfg.extract.omlx.endpoint),
                omlx_max_input_chars: Some(cfg.extract.omlx.max_input_chars),
            }
        }
        ExtractIntelligentMerge => {
            // `[intelligent_merge].provider == "none"` is the back-compat
            // sentinel meaning "fall back to query_expansion" — v0 used
            // a separate `resolved_provider` accessor. Map it to inherit
            // (level-1 skip) so the resolver walks levels 2/3.
            let prov = cfg.intelligent_merge.provider.to_lowercase();
            let inherit = matches!(prov.as_str(), "none" | "inherit");
            SectionExplicit {
                provider: if inherit {
                    None
                } else {
                    Some(cfg.intelligent_merge.provider.as_str())
                },
                inherit,
                max_input_chars: None,
                google_model: nonempty(&cfg.intelligent_merge.google.model),
                google_api_key_env: Some("GEMINI_API_KEY"),
                google_endpoint: nonempty(&cfg.intelligent_merge.google.endpoint),
                google_max_input_chars: None,
                omlx_model: nonempty(&cfg.intelligent_merge.omlx.model),
                omlx_endpoint: nonempty(&cfg.intelligent_merge.omlx.endpoint),
                omlx_max_input_chars: None,
            }
        }
        ExtractDedup => {
            // No dedicated `[extract.dedup]` block in v0.26.x — the
            // dedup verdict path reuses `[extract]`. Mirror it as
            // level-1 inherit so the resolver walks to level 2/3.
            SectionExplicit::empty()
        }
        QueryExpansion => SectionExplicit {
            provider: if query_expansion_inherit {
                None
            } else {
                Some(cfg.query_expansion.provider.as_str())
            },
            inherit: query_expansion_inherit,
            max_input_chars: None,
            google_model: nonempty(&cfg.query_expansion.google.model),
            google_api_key_env: Some("GEMINI_API_KEY"),
            google_endpoint: nonempty(&cfg.query_expansion.google.endpoint),
            google_max_input_chars: None,
            omlx_model: nonempty(&cfg.query_expansion.omlx.model),
            omlx_endpoint: nonempty(&cfg.query_expansion.omlx.endpoint),
            omlx_max_input_chars: None,
        },
        SearchLlmReranker => {
            // `[search].llm_reranker` is the section-provider field for
            // the reranker; the actual provider sub-tables it reads from
            // are `[query_expansion.{provider}]` (see
            // `search/rerank_llm.rs`). Mirror that v0.26.x behavior.
            SectionExplicit {
                provider: if reranker_inherit {
                    None
                } else {
                    Some(cfg.search.llm_reranker.as_str())
                },
                inherit: reranker_inherit,
                max_input_chars: None,
                google_model: nonempty(&cfg.query_expansion.google.model),
                google_api_key_env: Some("GEMINI_API_KEY"),
                google_endpoint: nonempty(&cfg.query_expansion.google.endpoint),
                google_max_input_chars: None,
                omlx_model: nonempty(&cfg.query_expansion.omlx.model),
                omlx_endpoint: nonempty(&cfg.query_expansion.omlx.endpoint),
                omlx_max_input_chars: None,
            }
        }
        ArsRecallSynthesis | ArsConceptSummary | ArsColdArchive => {
            // All three Ars Cap A/B/C sections share `[ars].llm_backend`
            // ("inherit" / "google" / "omlx" / "none") and reuse
            // `[extract.{provider}]` blocks for provider-scoped fields
            // (matches v0.26.x semantics in `ops/concept_summary.rs` etc.)
            let backend = cfg.ars.llm_backend.to_lowercase();
            let inherit = backend == "inherit";
            SectionExplicit {
                provider: if inherit {
                    None
                } else {
                    Some(cfg.ars.llm_backend.as_str())
                },
                inherit,
                max_input_chars: None,
                google_model: nonempty(&cfg.extract.google.model),
                google_api_key_env: Some("GEMINI_API_KEY"),
                google_endpoint: nonempty(&cfg.extract.google.endpoint),
                google_max_input_chars: Some(cfg.extract.google.max_input_chars),
                omlx_model: nonempty(&cfg.extract.omlx.model),
                omlx_endpoint: nonempty(&cfg.extract.omlx.endpoint),
                omlx_max_input_chars: Some(cfg.extract.omlx.max_input_chars),
            }
        }
        Resummerize => {
            let backend = cfg.resummerize.llm_backend.to_lowercase();
            let inherit = backend == "inherit";
            SectionExplicit {
                provider: if inherit {
                    None
                } else {
                    Some(cfg.resummerize.llm_backend.as_str())
                },
                inherit,
                max_input_chars: None,
                google_model: nonempty(&cfg.extract.google.model),
                google_api_key_env: Some("GEMINI_API_KEY"),
                google_endpoint: nonempty(&cfg.extract.google.endpoint),
                google_max_input_chars: Some(cfg.extract.google.max_input_chars),
                omlx_model: nonempty(&cfg.extract.omlx.model),
                omlx_endpoint: nonempty(&cfg.extract.omlx.endpoint),
                omlx_max_input_chars: Some(cfg.extract.omlx.max_input_chars),
            }
        }
        ArsLlmJudge | ArsLlmJudgeNightlyCron => SectionExplicit {
            // New-in-v0.27.1 sections — no level-1 provider shape.
            // `max_input_chars` is intentionally section-local, so it
            // can override the global `[llm.{provider}]` cap once a
            // provider is resolved at level 3.
            max_input_chars: cfg.ars.llm_judge.max_input_chars,
            ..SectionExplicit::empty()
        },
    }
}

/// Hardcoded baseline (level 4) per consumer.
///
/// Pre-Track-2 invariant: "no provider configured = no LLM call" —
/// every consumer's level-4 fallback is `Provider::None`, so the
/// consumer disables its LLM path gracefully. The endpoint string
/// stored here is purely informational (no HTTP traffic when provider
/// is None) but gets carried through anyway for telemetry symmetry.
fn hardcoded_fallback(sec: LlmSection) -> ResolvedLlmConfig {
    ResolvedLlmConfig {
        provider: Provider::None,
        model: String::new(),
        api_key_env: None,
        endpoint: default_google_endpoint(),
        max_input_chars: 0,
        temperature: None,
        request_timeout_ms: None,
        max_retries: None,
        source: PrecedenceSource::HardcodedFallback,
        section: section_name(sec).to_string(),
    }
}

fn section_name(sec: LlmSection) -> &'static str {
    use LlmSection::*;
    match sec {
        Extract => "extract",
        ExtractAsyncMemory => "extract.async_memory",
        ExtractIntelligentMerge => "extract.intelligent_merge",
        ExtractDedup => "extract.dedup",
        QueryExpansion => "query_expansion",
        SearchLlmReranker => "search.llm_reranker",
        ArsRecallSynthesis => "ars.recall_synthesis",
        ArsConceptSummary => "ars.concept_summary",
        ArsColdArchive => "ars.cold_archive",
        Resummerize => "resummerize",
        ArsLlmJudge => "ars.llm_judge",
        ArsLlmJudgeNightlyCron => "ars.llm_judge.nightly_cron",
    }
}

fn nonempty(s: &str) -> Option<&str> {
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

/// Pick a provider name string and return `Some(Provider)` if it parses
/// to a concrete provider (Google / Omlx). `"none"` and unknown values
/// produce `None`. Empty strings also produce `None` (back-compat: a
/// missing field).
fn pick_provider(name: &str) -> Option<Provider> {
    match name.to_lowercase().as_str() {
        "google" | "gemini" => Some(Provider::Google),
        "omlx" | "local" => Some(Provider::Omlx),
        _ => None,
    }
}

/// Read the four provider-scoped fields from a level-1 section snapshot
/// for the chosen provider. Returns `None` for any missing field.
fn level1_scoped_fields<'a>(
    explicit: &SectionExplicit<'a>,
    chosen: Provider,
) -> (
    Option<&'a str>,      // model
    Option<&'static str>, // api_key_env
    Option<&'a str>,      // endpoint
    Option<usize>,        // max_input_chars
) {
    match chosen {
        Provider::Google => (
            explicit.google_model,
            explicit.google_api_key_env,
            explicit.google_endpoint,
            explicit.google_max_input_chars,
        ),
        Provider::Omlx => (
            explicit.omlx_model,
            None,
            explicit.omlx_endpoint,
            explicit.omlx_max_input_chars,
        ),
        Provider::None => (None, None, None, None),
    }
}

/// Read the four provider-scoped fields from `[llm.{provider}]`
/// (level-3 sub-table). Returns `None` for any field not set in the
/// config file. Borrows from `llm` — the caller (`build_resolved_for_provider`)
/// already converts everything to `String` for the owned
/// `ResolvedLlmConfig`, so no `'static` lifetime is needed.
fn level3_scoped_fields<'a>(
    llm: &'a LlmDefaultsConfig,
    chosen: Provider,
) -> (
    Option<&'a str>,
    Option<&'a str>,
    Option<&'a str>,
    Option<usize>,
) {
    let table = match chosen {
        Provider::Google => &llm.google,
        Provider::Omlx => &llm.omlx,
        Provider::None => return (None, None, None, None),
    };
    // Operator may set `[llm.google].api_key_env = "MY_KEY"` to point at
    // a non-default env var. When unset, fall through to the pre-Track-2
    // baseline `"GEMINI_API_KEY"` for Google. OMLX has no API key.
    let api_key_env: Option<&'a str> = match chosen {
        Provider::Google => table
            .api_key_env
            .as_deref()
            .filter(|s| !s.is_empty())
            .or(Some("GEMINI_API_KEY")),
        _ => None,
    };
    (
        table.model.as_deref().filter(|s| !s.is_empty()),
        api_key_env,
        table.endpoint.as_deref().filter(|s| !s.is_empty()),
        table.max_input_chars,
    )
}

/// Build the `ResolvedLlmConfig` for a given chosen provider, walking
/// the provider-scoped fields as a unit per spec §8.1.
///
/// Precedence for scoped fields (with `provider` already chosen at level
/// `provider_level`):
/// 1. If `provider_level == SectionExplicit` or `SectionProvider`, read
///    scoped fields from the `explicit` snapshot's `[{provider}]` block.
/// 2. Else (or if level-1 missing a field) read `[llm.{provider}].{field}`.
/// 3. Final fallback to baseline values inside this fn.
///
/// Note: `SectionProvider` (level 2) is the "inherit" path — the
/// `explicit` snapshot the caller passes in MUST already be re-pointed
/// at the inherited section's data (e.g. `[extract]` for an
/// `[ars].llm_backend = "inherit"` walk). The caller — `resolve_llm_for_impl`
/// — sets that up before calling here.
fn build_resolved_for_provider(
    sec: LlmSection,
    section: &str,
    chosen: Provider,
    provider_level: PrecedenceSource,
    explicit: &SectionExplicit<'_>,
    llm: &LlmDefaultsConfig,
) -> crate::types::ReinResult<ResolvedLlmConfig> {
    // Level 1 / 2 — section-explicit (or inherited-section's) scoped
    // fields. Both levels read from the same `explicit` snapshot since
    // the resolver pre-rewires `explicit` to point at the inherited
    // section's blocks before calling here in the inherit path.
    // Codex R5 P2 fix — when the section explicitly opted into inherit
    // (provider = "inherit" sentinel), level-1 scoped fields MUST NOT
    // win over level-3 `[llm.{provider}]`. The inherit signal means the
    // operator wants the global block to apply uniformly; reading
    // `[extract.google].model` before `[llm.google].model` defeats the
    // whole point of the sentinel. Only `SectionExplicit` (provider was
    // genuinely written by user, not inherit) preserves level-1 priority.
    let (l1_model, l1_api_key_env, l1_endpoint, l1_max_input_chars) = match provider_level {
        PrecedenceSource::SectionExplicit => level1_scoped_fields(explicit, chosen),
        PrecedenceSource::SectionProvider if !explicit.inherit => {
            level1_scoped_fields(explicit, chosen)
        }
        _ => (None, None, None, None),
    };

    // Level 3 — `[llm.{provider}]` scoped fields.
    let (l3_model, l3_api_key_env, l3_endpoint, l3_max_input_chars) =
        level3_scoped_fields(llm, chosen);

    // Codex R3 P2 fix — when the provider was selected via GlobalDefault
    // (`[llm].provider = "omlx"`) but the operator didn't write
    // `[llm.omlx].model`, fall back to the legacy section's same-provider
    // sub-table (e.g. `[extract.omlx].model`) before erroring. Without
    // this, an operator who only writes `[llm].provider = "omlx"` and
    // leaves the section default-populated sees `resolve_llm_for("extract")`
    // fail with no model configured, even though `[extract.omlx]` has the
    // same default `"default"` model the section path would have used.
    let (l_legacy_model, l_legacy_api_key_env, l_legacy_endpoint, l_legacy_max_input_chars) =
        match provider_level {
            PrecedenceSource::GlobalDefault => level1_scoped_fields(explicit, chosen),
            _ => (None, None, None, None),
        };

    let model = l1_model
        .or(l3_model)
        .or(l_legacy_model)
        .map(|s| s.to_string())
        .ok_or_else(|| {
            crate::types::ReinError::Config(format!(
                "resolve_llm_for(\"{section}\"): provider = {provider:?} but no \
                 model is configured at any precedence level — set \
                 `[{section}.{provider_lc}].model` or \
                 `[llm.{provider_lc}].model`",
                provider = chosen,
                provider_lc = match chosen {
                    Provider::Google => "google",
                    Provider::Omlx => "omlx",
                    Provider::None => unreachable!(),
                }
            ))
        })?;

    // Coerce `&'static str` from level 1 into `&str` matching level 3
    // before `.or()` — the borrow ends in `to_string()` either way.
    // Codex R3 P2: include legacy-section fallback at the end of the
    // chain for the GlobalDefault path (mirror of model fallback above).
    let api_key_env = l1_api_key_env
        .map(|s| s.to_string())
        .or_else(|| l3_api_key_env.map(|s| s.to_string()))
        .or_else(|| l_legacy_api_key_env.map(|s| s.to_string()));

    let endpoint = l1_endpoint
        .map(|s| s.to_string())
        .or_else(|| l3_endpoint.map(|s| s.to_string()))
        .or_else(|| l_legacy_endpoint.map(|s| s.to_string()))
        .unwrap_or_else(|| match (sec, chosen) {
            (LlmSection::QueryExpansion | LlmSection::SearchLlmReranker, Provider::Omlx) => {
                "http://localhost:8000/v1".to_string()
            }
            (_, Provider::Omlx) => "http://localhost:11434/v1".to_string(),
            (_, _) => default_google_endpoint(),
        });

    let max_input_chars = explicit
        .max_input_chars
        .or(l1_max_input_chars)
        .or(l3_max_input_chars)
        .or(l_legacy_max_input_chars)
        .unwrap_or(match chosen {
            Provider::Google => 0, // 1M-token model default
            Provider::Omlx => default_omlx_max_input_chars(),
            Provider::None => 0,
        });

    Ok(ResolvedLlmConfig {
        provider: chosen,
        model,
        api_key_env,
        endpoint,
        max_input_chars,
        temperature: llm.temperature,
        request_timeout_ms: llm.request_timeout_ms,
        max_retries: llm.max_retries,
        source: provider_level,
        section: section.to_string(),
    })
}

/// Core resolver — walks the 4-level precedence chain.
fn resolve_llm_for_impl(
    cfg: &ReinConfig,
    section: &str,
) -> crate::types::ReinResult<ResolvedLlmConfig> {
    // Special-case the dotted nightly_cron section — it reads its
    // section-provider field from `[ars.llm_judge]` rather than from a
    // `[ars.llm_judge.nightly_cron].provider` field (which doesn't exist
    // in v0.27.1; operator overrides via `[ars.llm_judge.nightly_cron.{p}]`
    // are deferred to v0.27.2+ when the cron actually runs).
    let sec = LlmSection::parse(section).ok_or_else(|| {
        crate::types::ReinError::Config(format!(
            "resolve_llm_for: unknown section \"{section}\". Valid sections: \
             extract, extract.async_memory, extract.intelligent_merge, \
             extract.dedup, query_expansion, search.llm_reranker, \
             ars.recall_synthesis, ars.concept_summary, ars.cold_archive, \
             resummerize, ars.llm_judge, ars.llm_judge.nightly_cron"
        ))
    })?;

    let explicit = section_explicit(cfg, sec);

    // Level 1 — section-explicit `provider` (skipped when `inherit` or
    // when the section has no level-1 explicit shape).
    if let Some(name) = explicit.provider {
        if let Some(chosen) = pick_provider(name) {
            return build_resolved_for_provider(
                sec,
                section,
                chosen,
                PrecedenceSource::SectionExplicit,
                &explicit,
                &cfg.llm,
            );
        }
        // Section provider was set but resolves to None — back-compat
        // path: a `provider = "none"` extract block disables LLM paths.
        // `expand_provider() == None` is also a real production setting.
        // Skip levels 2/3 — falling through to `[llm]` would silently
        // re-enable an LLM the operator turned off.
        if name.eq_ignore_ascii_case("none") {
            return Ok(ResolvedLlmConfig {
                provider: Provider::None,
                model: String::new(),
                api_key_env: None,
                endpoint: default_google_endpoint(),
                max_input_chars: 0,
                temperature: cfg.llm.temperature,
                request_timeout_ms: cfg.llm.request_timeout_ms,
                max_retries: cfg.llm.max_retries,
                source: PrecedenceSource::SectionExplicit,
                section: section.to_string(),
            });
        }
        // Unknown provider name — validate() should have caught this
        // earlier, but defensively fall through to the [llm] chain.
    }

    // Level 2 — section-explicit `inherit` AND `[llm].provider` exists →
    // read scoped fields from `[llm.{provider}]` but the provider choice
    // came from `[llm]`. (Conceptually identical to level 3 in this
    // implementation since "inherit" is the only thing that distinguishes
    // them; spec §8.1 lists them separately for documentation symmetry.)
    if explicit.inherit {
        if let Some(chosen) = pick_provider(&cfg.llm.provider) {
            return build_resolved_for_provider(
                sec,
                section,
                chosen,
                PrecedenceSource::SectionProvider,
                &explicit,
                &cfg.llm,
            );
        }
        // No `[llm]` provider AND section says "inherit" — for the
        // existing v0.26.x slow-channel sections (`ars.*`,
        // `resummerize`, `async_memory`), "inherit" historically meant
        // "fall back to `[extract].provider`". Preserve that semantic
        // for back-compat by walking `[extract]`.
        if matches!(
            sec,
            LlmSection::ExtractAsyncMemory
                | LlmSection::ArsRecallSynthesis
                | LlmSection::ArsConceptSummary
                | LlmSection::ArsColdArchive
                | LlmSection::Resummerize
                | LlmSection::ExtractIntelligentMerge
        ) {
            // Synthesize a level-1 result by reading `[extract]` directly.
            let extract_explicit = section_explicit(cfg, LlmSection::Extract);
            if let Some(name) = extract_explicit.provider {
                if let Some(chosen) = pick_provider(name) {
                    return build_resolved_for_provider(
                        sec,
                        section,
                        chosen,
                        PrecedenceSource::SectionProvider,
                        &extract_explicit,
                        &cfg.llm,
                    );
                }
                if name.eq_ignore_ascii_case("none") {
                    return Ok(ResolvedLlmConfig {
                        provider: Provider::None,
                        model: String::new(),
                        api_key_env: None,
                        endpoint: default_google_endpoint(),
                        max_input_chars: 0,
                        temperature: cfg.llm.temperature,
                        request_timeout_ms: cfg.llm.request_timeout_ms,
                        max_retries: cfg.llm.max_retries,
                        source: PrecedenceSource::SectionProvider,
                        section: section.to_string(),
                    });
                }
            }
        }
    }

    // Level 3 — global default `[llm].provider` + `[llm.{provider}]`.
    // Note: for the new-in-v0.27.1 sections (`ars.llm_judge`,
    // `ars.llm_judge.nightly_cron`, `extract.dedup`) we entered this
    // branch directly because `explicit.provider` was `None` and
    // `explicit.inherit` was `false` (their level-1 was empty()).
    if let Some(chosen) = pick_provider(&cfg.llm.provider) {
        return build_resolved_for_provider(
            sec,
            section,
            chosen,
            PrecedenceSource::GlobalDefault,
            &explicit,
            &cfg.llm,
        );
    }

    // For `ars.llm_judge.nightly_cron`, level 2 also walks back to
    // `[ars.llm_judge]` resolution. Only relevant when no `[llm]` was
    // set; matches spec §3.3 commentary about cron inheriting from
    // `[ars.llm_judge]` by default.
    if sec == LlmSection::ArsLlmJudgeNightlyCron {
        return resolve_llm_for_impl(cfg, "ars.llm_judge");
    }

    // For `extract.intelligent_merge`, "fall back to query_expansion"
    // is the v0.26.x semantic (`extract/intelligent_merge.rs::
    // build_classifier`). Reproduce that walk before hitting level 4.
    if sec == LlmSection::ExtractIntelligentMerge {
        let qe_explicit = section_explicit(cfg, LlmSection::QueryExpansion);
        if let Some(name) = qe_explicit.provider {
            if let Some(chosen) = pick_provider(name) {
                return build_resolved_for_provider(
                    sec,
                    section,
                    chosen,
                    PrecedenceSource::SectionProvider,
                    &qe_explicit,
                    &cfg.llm,
                );
            }
            if name.eq_ignore_ascii_case("none") {
                return Ok(none_resolved(sec, section, &cfg.llm));
            }
        }
    }

    // For `extract.dedup`, "inherit" semantic is "fall back to
    // [extract]" (matches v0.26.x `extract/dedup.rs::
    // build_dedup_extractor`).
    if sec == LlmSection::ExtractDedup {
        let ex_explicit = section_explicit(cfg, LlmSection::Extract);
        if let Some(name) = ex_explicit.provider {
            if let Some(chosen) = pick_provider(name) {
                return build_resolved_for_provider(
                    sec,
                    section,
                    chosen,
                    PrecedenceSource::SectionProvider,
                    &ex_explicit,
                    &cfg.llm,
                );
            }
            if name.eq_ignore_ascii_case("none") {
                return Ok(none_resolved(sec, section, &cfg.llm));
            }
        }
    }

    // Level 4 — hardcoded baseline (Provider::None for every
    // pre-Track-2-untouched consumer when no provider was configured at
    // any level).
    Ok(hardcoded_fallback(sec))
}

/// Helper: build a `Provider::None` resolved config — used by every
/// "section explicitly turned LLM off" branch.
fn none_resolved(sec: LlmSection, section: &str, llm: &LlmDefaultsConfig) -> ResolvedLlmConfig {
    ResolvedLlmConfig {
        provider: Provider::None,
        model: String::new(),
        api_key_env: None,
        endpoint: default_google_endpoint(),
        max_input_chars: 0,
        temperature: llm.temperature,
        request_timeout_ms: llm.request_timeout_ms,
        max_retries: llm.max_retries,
        source: PrecedenceSource::SectionExplicit,
        section: if section.is_empty() {
            section_name(sec).to_string()
        } else {
            section.to_string()
        },
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

    // v0.35 Phase 3: transform any surviving legacy
    // `[server].allow_unauthenticated_loopback` into an explicit
    // `[server].auth` policy before serde decode (the field was removed
    // from ServerConfig + `deny_unknown_fields` would otherwise reject the
    // load). See `migrate_legacy_server_auth` for the mapping rules.
    migrate_legacy_server_auth(&mut base_val)?;

    // Deserialize the merged table back into ReinConfig
    let merged: ReinConfig = base_val.try_into()?;
    Ok(merged)
}

/// v0.35 Phase 3 — TOML load-time migration of the removed
/// `[server].allow_unauthenticated_loopback` bool into an explicit
/// `[server].auth` policy.
///
/// Mapping rules (all derived from the v0.34 runtime semantics of
/// `ReinConfig::resolve_auth_policy` so behavior is preserved across the
/// upgrade for every config shape that previously resolved):
///
/// - bool absent or false → no-op (the key is stripped; defaults take over).
/// - bool true + explicit `[server].auth` set → the legacy bool was always
///   silently overridden by explicit policy; strip it and emit a one-time
///   WARN so the operator removes the dead key from their config.
/// - bool true + `REIN_HTTP_TOKEN` set → bearer auth always won over the
///   bool at runtime; map to `auth = "bearer_required"` and emit WARN.
/// - bool true + loopback bind + no token → previous runtime resolved to
///   `Public`; map to `auth = "public"` and emit WARN.
/// - bool true + non-loopback bind + no token → the v0.34 doctor already
///   reported FAIL for this (the bool could not take effect anyway); fail
///   the load with a clear migration message instead of letting an
///   ambiguous startup proceed.
fn migrate_legacy_server_auth(val: &mut toml::Value) -> anyhow::Result<()> {
    let Some(server_table) = val.get_mut("server").and_then(|v| v.as_table_mut()) else {
        return Ok(());
    };
    let Some(legacy) = server_table.remove("allow_unauthenticated_loopback") else {
        return Ok(());
    };
    let opted_in = legacy.as_bool().unwrap_or(false);
    if !opted_in {
        // bool=false was the default; the key is removed silently so the
        // strict `deny_unknown_fields` decoder accepts the merged table.
        return Ok(());
    }
    if server_table.contains_key("auth") {
        tracing::warn!(
            target: "rein::config::migration",
            "[server].allow_unauthenticated_loopback = true was ignored because [server].auth is set explicitly; remove the legacy bool from your config (it is no longer recognized as of v0.35.0)"
        );
        return Ok(());
    }
    let has_token = nonempty_env("REIN_HTTP_TOKEN").is_some();
    // The effective bind at runtime is `REIN_SSE_BIND` env (applied later in
    // `load()`) when set, falling back to the TOML `[server].sse_bind`,
    // falling back to the loopback default. Reading the env var here means
    // a legacy bool combined with `REIN_SSE_BIND=0.0.0.0` cannot silently
    // migrate to `auth = "public"` based on a stale loopback default. See
    // codex review feedback on the v0.35 Phase 3 patch.
    let bind = nonempty_env("REIN_SSE_BIND").unwrap_or_else(|| {
        server_table
            .get("sse_bind")
            .and_then(|v| v.as_str())
            .unwrap_or("127.0.0.1")
            .to_string()
    });
    let bind = bind.as_str();
    if has_token {
        server_table.insert(
            "auth".to_string(),
            toml::Value::String("bearer_required".to_string()),
        );
        tracing::warn!(
            target: "rein::config::migration",
            migrated_to = "bearer_required",
            "[server].allow_unauthenticated_loopback = true was migrated to [server].auth = \"bearer_required\" (REIN_HTTP_TOKEN is set, so token auth wins); remove the legacy bool from your config (it is no longer recognized as of v0.35.0)"
        );
    } else if is_loopback_bind_host(bind) {
        server_table.insert(
            "auth".to_string(),
            toml::Value::String("public".to_string()),
        );
        tracing::warn!(
            target: "rein::config::migration",
            migrated_to = "public",
            "[server].allow_unauthenticated_loopback = true was migrated to [server].auth = \"public\" (loopback bind, no token); remove the legacy bool from your config (it is no longer recognized as of v0.35.0)"
        );
    } else {
        // Codex P1 v2: do NOT bail at config load. The legacy bool could
        // never take effect on a non-loopback bind without a token even
        // under v0.34 (`loopback_unauth_requested()` returned false), and
        // failing the load now would brick stdio/CLI commands that don't
        // touch the HTTP surface. Instead leave `[server].auth` unset
        // and let `resolve_auth_policy` produce the actionable error at
        // the point the HTTP surface is actually requested.
        tracing::warn!(
            target: "rein::config::migration",
            "[server].allow_unauthenticated_loopback = true was stripped but NOT migrated: bind '{bind}' is not loopback and REIN_HTTP_TOKEN is not set, so no safe policy can be inferred automatically. HTTP/SSE will refuse to start until you set [server].auth explicitly (\"public\", \"loopback_only\", \"bearer_required\", or \"oauth\")."
        );
    }
    Ok(())
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
    use tempfile::TempDir;

    #[test]
    fn test_default_config() {
        let cfg = ReinConfig::default();
        assert!((cfg.search.rrf_k - 60.0).abs() < f64::EPSILON);
        assert_eq!(cfg.embedding.dimensions, 3072);
        assert!(!cfg.server.compact);
        assert!(cfg.server.background_warmup);
        assert!(!cfg.server.stdio_background_warmup);
        assert_eq!(cfg.database.path, "auto");
        assert_eq!(cfg.embedding.provider, "google");
        assert_eq!(cfg.chunking.max_tokens, 512);
        assert!((cfg.decay.base_lambda - 0.06).abs() < f64::EPSILON);
        assert!(!cfg.hooks.codex.inject_prompt_context);
        assert!(!cfg.hooks.codex.inject_session_context);
        assert!(cfg.hooks.codex.guardrails_enabled);
        assert_eq!(cfg.hooks.codex.max_additional_context_chars, 1200);
        assert_eq!(cfg.async_memory.selection_limit, 2);
        assert!(cfg.ars.acceleration.enabled);
        assert!(!cfg.ars.acceleration.shadow_only);
        // H0 (v0.28 audit): LLM judge defaults reverted to false in v0.28.7;
        // operators must explicitly opt in. See
        // `test_default_ars_llm_judge_disabled` for the dedicated regression
        // guard.
        assert!(!cfg.ars.llm_judge.enabled);
        assert!(!cfg.ars.llm_judge.nightly_cron.enabled);
    }

    #[test]
    fn test_load_from_toml() {
        let toml_str = r#"
[search]
rrf_k = 30.0

[hooks.codex]
inject_prompt_context = true
inject_session_context = true
guardrails_enabled = false
max_additional_context_chars = 1024
"#;
        let cfg = ReinConfig::load_from_str(toml_str).unwrap();
        // Override applied
        assert!((cfg.search.rrf_k - 30.0).abs() < f64::EPSILON);
        assert!(cfg.hooks.codex.inject_prompt_context);
        assert!(cfg.hooks.codex.inject_session_context);
        assert!(!cfg.hooks.codex.guardrails_enabled);
        assert_eq!(cfg.hooks.codex.max_additional_context_chars, 1024);
        // Other defaults preserved
        assert_eq!(cfg.embedding.dimensions, 3072);
        assert!(!cfg.server.compact);
        assert_eq!(cfg.database.path, "auto");
        // H0 (v0.28 audit): nightly_cron defaults to false; the user TOML
        // here does not opt into it.
        assert!(!cfg.ars.llm_judge.nightly_cron.enabled);
    }

    #[test]
    fn config_version_defaults_and_downgrade_guard() {
        // A config that omits config_version loads at the current version
        // (it is merged over the embedded default.toml which stamps it).
        let cfg = ReinConfig::load_from_str("[search]\nrrf_k = 30.0\n").unwrap();
        assert_eq!(cfg.config_version, CURRENT_CONFIG_VERSION);

        // The bare derived Default carries the unversioned sentinel 0, which
        // the guard treats as the current baseline (validate must accept it).
        let def = ReinConfig::default();
        assert_eq!(def.config_version, 0);
        def.validate()
            .expect("unversioned (0) config must validate as current baseline");

        // An explicit current version round-trips and validates.
        let cur =
            ReinConfig::load_from_str(&format!("config_version = {CURRENT_CONFIG_VERSION}\n"))
                .unwrap();
        assert_eq!(cur.config_version, CURRENT_CONFIG_VERSION);

        // A newer-than-supported version is refused at load (forward-compat
        // guard), so an older binary never misreads future config semantics.
        let err = ReinConfig::load_from_str(&format!(
            "config_version = {}\n",
            CURRENT_CONFIG_VERSION + 1
        ))
        .unwrap_err();
        assert!(
            err.to_string().contains("newer than this rein build"),
            "expected forward-compat rejection, got: {err}"
        );
    }

    #[test]
    fn test_load_from_toml_preserves_compact_hook_defaults_when_omitted() {
        let toml_str = r#"
[search]
rrf_k = 30.0
"#;
        let cfg = ReinConfig::load_from_str(toml_str).unwrap();

        assert_eq!(cfg.hooks.codex.max_additional_context_chars, 1200);
        assert_eq!(cfg.async_memory.selection_limit, 2);
        assert!(cfg.server.background_warmup);
        assert!(!cfg.server.stdio_background_warmup);
    }

    #[test]
    fn test_load_from_toml_parses_server_background_warmup_policy() {
        let toml_str = r#"
[server]
background_warmup = false
stdio_background_warmup = true
"#;
        let cfg = ReinConfig::load_from_str(toml_str).unwrap();

        assert!(!cfg.server.background_warmup);
        assert!(cfg.server.stdio_background_warmup);
    }

    #[test]
    fn test_load_from_toml_parses_ars_acceleration() {
        let toml_str = r#"
[ars.acceleration]
enabled = true
"#;
        let cfg = ReinConfig::load_from_str(toml_str).unwrap();

        assert!(cfg.ars.acceleration.enabled);
        assert!(!cfg.ars.acceleration.shadow_only);
    }

    #[test]
    fn test_load_from_toml_preserves_explicit_shadow_acceleration() {
        let toml_str = r#"
[ars.acceleration]
enabled = true
shadow_only = true
"#;
        let cfg = ReinConfig::load_from_str(toml_str).unwrap();

        assert!(cfg.ars.acceleration.enabled);
        assert!(cfg.ars.acceleration.shadow_only);
    }

    #[test]
    fn test_load_from_toml_parses_non_shadow_acceleration() {
        let toml_str = r#"
[ars.acceleration]
enabled = true
shadow_only = false
"#;
        let cfg = ReinConfig::load_from_str(toml_str).unwrap();

        assert!(cfg.ars.acceleration.enabled);
        assert!(!cfg.ars.acceleration.shadow_only);
    }

    /// H0 regression guard (v0.28 audit): the master `[ars.llm_judge]`
    /// feature flag must default to `false` per the v0.28 charter
    /// non-goal "Do not make LLM judge default-on". A v0.28.6 regression
    /// flipped both the Rust `Default` impl and the embedded `default.toml`
    /// to `true`; this test pins the Rust-default side.
    #[test]
    fn test_default_ars_llm_judge_disabled() {
        let cfg = ReinConfig::default();
        assert!(
            !cfg.ars.llm_judge.enabled,
            "[ars.llm_judge].enabled must default to false (v0.28 charter non-goal)"
        );
    }

    /// H0 regression guard: the Layer 2 nightly cron must also default
    /// to `false` since it shares the same daily-call ledger and is
    /// part of the same opt-in surface.
    #[test]
    fn test_default_ars_llm_judge_nightly_cron_disabled() {
        let cfg = ReinConfig::default();
        assert!(
            !cfg.ars.llm_judge.nightly_cron.enabled,
            "[ars.llm_judge.nightly_cron].enabled must default to false"
        );
    }

    /// H0 regression guard: the embedded `config/default.toml` is the
    /// foundation of `merge_toml`. Even if the Rust `Default` impl is
    /// `false`, an `enabled = true` line in the embedded TOML would
    /// override it for every operator who does not explicitly set the
    /// field in their own `~/.rein/config.toml`. This test loads the
    /// embedded TOML through the same path the binary uses (empty user
    /// override) and asserts both gates remain off.
    #[test]
    fn test_embedded_default_toml_does_not_force_llm_judge_on() {
        let cfg = ReinConfig::load_from_str("")
            .expect("empty user TOML must load against embedded default.toml");
        assert!(
            !cfg.ars.llm_judge.enabled,
            "embedded default.toml must not force [ars.llm_judge].enabled = true"
        );
        assert!(
            !cfg.ars.llm_judge.nightly_cron.enabled,
            "embedded default.toml must not force [ars.llm_judge.nightly_cron].enabled = true"
        );
    }

    #[test]
    fn test_ars_llm_judge_max_input_chars_overrides_resolved_llm_default() {
        let toml_str = r#"
[llm]
provider = "google"

[llm.google]
model = "gemini-test"
max_input_chars = 4096

[ars.llm_judge]
max_input_chars = 512
"#;
        let cfg = ReinConfig::load_from_str(toml_str)
            .expect("ars.llm_judge.max_input_chars should be accepted");
        let resolved = cfg
            .resolve_llm_for("ars.llm_judge")
            .expect("ars.llm_judge should resolve via [llm]");

        assert_eq!(resolved.max_input_chars, 512);
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

        fn remove(key: &'static str) -> Self {
            let prior = std::env::var(key).ok();
            std::env::remove_var(key);
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

    // v0.35 Phase 3: `server_loopback_unauth_requested_requires_loopback_bind`
    // was deleted. Both the bool and its `loopback_unauth_requested()` helper
    // are gone; the equivalent semantics live in the TOML-load-time
    // `migrate_legacy_server_auth` transformation, exercised by the
    // `legacy_server_auth_*` migration tests below.

    #[test]
    fn auth_policy_parses_explicit_server_auth_values() {
        let cfg = ReinConfig::load_from_str(
            r#"
[server]
auth = "oauth"
public_url = "https://rein.example.com"
"#,
        )
        .expect("server.auth should parse");

        assert_eq!(cfg.server.auth, Some(AuthPolicyConfig::OAuth));
        assert_eq!(
            cfg.server.public_url.as_deref(),
            Some("https://rein.example.com")
        );
    }

    #[test]
    #[serial_test::serial(global_state)]
    fn explicit_auth_policy_is_not_changed_by_rein_http_token() {
        let _token = EnvGuard::set("REIN_HTTP_TOKEN", "secret-token");
        let cfg = ReinConfig::load_from_str(
            r#"
[server]
auth = "loopback_only"
"#,
        )
        .expect("server.auth should parse");

        assert_eq!(
            cfg.resolve_auth_policy().expect("policy should resolve"),
            crate::auth::AuthPolicy::LoopbackOnly
        );
    }

    #[test]
    #[serial_test::serial(global_state)]
    fn auth_policy_derives_bearer_required_from_token_and_old_false_flag() {
        let _token = EnvGuard::set("REIN_HTTP_TOKEN", "secret-token");
        let cfg = ReinConfig::load_from_str(
            r#"
[server]
allow_unauthenticated_loopback = false
"#,
        )
        .expect("legacy config should parse");

        assert_eq!(
            cfg.resolve_auth_policy().expect("policy should resolve"),
            crate::auth::AuthPolicy::BearerRequired
        );
    }

    #[test]
    #[serial_test::serial(global_state)]
    fn auth_policy_preserves_legacy_token_precedence_over_loopback_flag() {
        let _token = EnvGuard::set("REIN_HTTP_TOKEN", "secret-token");
        let cfg = ReinConfig::load_from_str(
            r#"
[server]
sse_bind = "127.0.0.1"
allow_unauthenticated_loopback = true
"#,
        )
        .expect("legacy config should parse");

        assert_eq!(
            cfg.resolve_auth_policy().expect("policy should resolve"),
            crate::auth::AuthPolicy::BearerRequired
        );
    }

    #[test]
    #[serial_test::serial(global_state)]
    fn auth_policy_preserves_legacy_public_read_from_loopback_flag() {
        let _token = EnvGuard::remove("REIN_HTTP_TOKEN");
        // v0.35 Phase 3: migration reads REIN_SSE_BIND env to decide
        // loopback vs non-loopback; isolate the test from an externally
        // set value.
        let _bind = EnvGuard::remove("REIN_SSE_BIND");
        let cfg = ReinConfig::load_from_str(
            r#"
[server]
sse_bind = "127.0.0.1"
allow_unauthenticated_loopback = true
allowed_hosts = ["rein.example.com"]
"#,
        )
        .expect("legacy loopback config should parse");

        assert_eq!(
            cfg.resolve_auth_policy().expect("policy should resolve"),
            crate::auth::AuthPolicy::Public
        );
    }

    #[test]
    #[serial_test::serial(global_state)]
    fn auth_policy_rejects_implicit_public_http_without_token_or_loopback_opt_in() {
        let _token = EnvGuard::remove("REIN_HTTP_TOKEN");
        let cfg = ReinConfig::load_from_str(
            r#"
[server]
sse_bind = "0.0.0.0"
allow_unauthenticated_loopback = false
"#,
        )
        .expect("config should parse");

        let err = cfg
            .resolve_auth_policy()
            .expect_err("implicit unauthenticated public HTTP must not resolve");
        assert!(
            err.to_string().contains("REIN_HTTP_TOKEN"),
            "unexpected error: {err}"
        );
    }

    #[test]
    #[serial_test::serial(global_state)]
    fn explicit_bearer_required_requires_rein_http_token() {
        let _token = EnvGuard::remove("REIN_HTTP_TOKEN");
        let cfg = ReinConfig::load_from_str(
            r#"
[server]
auth = "bearer_required"
"#,
        )
        .expect("config should parse");

        let err = cfg
            .resolve_auth_policy()
            .expect_err("bearer_required without token must fail");
        assert!(err.to_string().contains("requires REIN_HTTP_TOKEN"));
    }

    #[test]
    #[serial_test::serial(global_state)]
    fn explicit_oauth_requires_owner_approval_token() {
        let _token = EnvGuard::remove("REIN_HTTP_TOKEN");
        let cfg = ReinConfig::load_from_str(
            r#"
[server]
auth = "oauth"
"#,
        )
        .expect("config should parse");

        let err = cfg
            .resolve_auth_policy()
            .expect_err("oauth without owner token must fail");
        assert!(err.to_string().contains("OAuth owner approval"));
    }

    // v0.35 Phase 3 — legacy `[server].allow_unauthenticated_loopback`
    // migration. The bool was removed from `ServerConfig`; configs that
    // still carry it are transformed by `migrate_legacy_server_auth` at
    // TOML load time. The migration target is chosen to preserve the
    // v0.34 runtime semantics of `resolve_auth_policy` (see the function
    // doc comment for the full mapping table).

    #[test]
    #[serial_test::serial(global_state)]
    fn legacy_server_auth_migrates_loopback_unauth_to_public_when_no_token() {
        let _token = EnvGuard::remove("REIN_HTTP_TOKEN");
        let _bind = EnvGuard::remove("REIN_SSE_BIND");
        let cfg = ReinConfig::load_from_str(
            r#"
[server]
sse_bind = "127.0.0.1"
allow_unauthenticated_loopback = true
"#,
        )
        .expect("legacy bool on loopback bind should migrate to public");

        assert_eq!(cfg.server.auth, Some(AuthPolicyConfig::Public));
        assert_eq!(
            cfg.resolve_auth_policy().expect("policy resolves"),
            crate::auth::AuthPolicy::Public
        );
    }

    #[test]
    #[serial_test::serial(global_state)]
    fn legacy_server_auth_migrates_loopback_unauth_to_bearer_when_token_set() {
        let _token = EnvGuard::set("REIN_HTTP_TOKEN", "migration-token");
        let _bind = EnvGuard::remove("REIN_SSE_BIND");
        let cfg = ReinConfig::load_from_str(
            r#"
[server]
sse_bind = "127.0.0.1"
allow_unauthenticated_loopback = true
"#,
        )
        .expect("legacy bool + token should migrate to bearer_required");

        assert_eq!(cfg.server.auth, Some(AuthPolicyConfig::BearerRequired));
        assert_eq!(
            cfg.resolve_auth_policy().expect("policy resolves"),
            crate::auth::AuthPolicy::BearerRequired
        );
    }

    #[test]
    #[serial_test::serial(global_state)]
    fn legacy_server_auth_migration_strips_bool_without_safe_inference_on_non_loopback() {
        // Codex P1 v2 (v0.35 Phase 3): the migration must NOT bail at load
        // time for the non-loopback + no-token case, because that would
        // brick stdio/CLI commands (`rein doctor`, `rein recall`, …) that
        // don't touch the HTTP surface. Strip the legacy bool; leave
        // `[server].auth` unset; the actionable failure surfaces only when
        // `resolve_auth_policy` is actually called by `rein serve --sse`.
        let _token = EnvGuard::remove("REIN_HTTP_TOKEN");
        let _bind = EnvGuard::remove("REIN_SSE_BIND");
        let cfg = ReinConfig::load_from_str(
            r#"
[server]
sse_bind = "0.0.0.0"
allow_unauthenticated_loopback = true
"#,
        )
        .expect("legacy bool on non-loopback bind without token must still load");

        assert!(
            cfg.server.auth.is_none(),
            "no safe policy can be inferred; auth must stay None"
        );
        let err = cfg
            .resolve_auth_policy()
            .expect_err("resolve_auth_policy must reject the no-policy + no-token shape");
        let err_str = err.to_string();
        assert!(err_str.contains("REIN_HTTP_TOKEN must be set"));
        assert!(err_str.contains("[server].auth explicitly"));
    }

    #[test]
    #[serial_test::serial(global_state)]
    fn legacy_server_auth_migrates_silently_when_bool_is_false() {
        let _token = EnvGuard::set("REIN_HTTP_TOKEN", "false-flag-token");
        let _bind = EnvGuard::remove("REIN_SSE_BIND");
        let cfg = ReinConfig::load_from_str(
            r#"
[server]
sse_bind = "127.0.0.1"
allow_unauthenticated_loopback = false
"#,
        )
        .expect("legacy bool = false should be stripped silently");

        // No explicit policy set (bool was the only [server] auth-shaping
        // directive); token presence still routes to bearer_required at
        // resolve time.
        assert!(cfg.server.auth.is_none());
        assert_eq!(
            cfg.resolve_auth_policy().expect("policy resolves"),
            crate::auth::AuthPolicy::BearerRequired
        );
    }

    #[test]
    #[serial_test::serial(global_state)]
    fn legacy_server_auth_migration_honors_rein_sse_bind_env_override() {
        // Codex P1 (v0.35 Phase 3): when REIN_SSE_BIND points at a
        // non-loopback address but the TOML still has `sse_bind = "127.0.0.1"`
        // alongside the legacy bool, the migration must use the env bind to
        // decide loopback vs non-loopback — otherwise it would silently map
        // to `auth = "public"` and start an unauthenticated listener on the
        // env-overridden non-loopback bind.
        //
        // Codex P1 v2: load must still succeed (so stdio/CLI commands keep
        // working with an old config). The migration strips the bool;
        // resolve_auth_policy catches the bad posture at HTTP startup.
        let _token = EnvGuard::remove("REIN_HTTP_TOKEN");
        let _bind = EnvGuard::set("REIN_SSE_BIND", "0.0.0.0");
        let cfg = ReinConfig::load_from_str(
            r#"
[server]
sse_bind = "127.0.0.1"
allow_unauthenticated_loopback = true
"#,
        )
        .expect("load must succeed even when REIN_SSE_BIND overrides to non-loopback");
        assert!(
            cfg.server.auth.is_none(),
            "REIN_SSE_BIND override must defeat the legacy-bool->public migration; auth stays None"
        );
        let err = cfg
            .resolve_auth_policy()
            .expect_err("HTTP/SSE startup must refuse the unauthenticated non-loopback posture");
        assert!(err.to_string().contains("REIN_HTTP_TOKEN must be set"));
    }

    #[test]
    #[serial_test::serial(global_state)]
    fn legacy_server_auth_migration_yields_to_explicit_auth_policy() {
        let _token = EnvGuard::remove("REIN_HTTP_TOKEN");
        let _bind = EnvGuard::remove("REIN_SSE_BIND");
        let cfg = ReinConfig::load_from_str(
            r#"
[server]
sse_bind = "127.0.0.1"
auth = "loopback_only"
allow_unauthenticated_loopback = true
"#,
        )
        .expect("explicit auth should win, legacy bool stripped silently");

        assert_eq!(cfg.server.auth, Some(AuthPolicyConfig::LoopbackOnly));
        assert_eq!(
            cfg.resolve_auth_policy().expect("policy resolves"),
            crate::auth::AuthPolicy::LoopbackOnly
        );
    }

    #[test]
    fn auth_policy_rejects_unknown_server_auth_value() {
        let err = ReinConfig::load_from_str(
            r#"
[server]
auth = "tokenish"
"#,
        )
        .expect_err("unknown auth policy must fail config parsing");

        assert!(
            err.to_string().contains("unknown variant") || err.to_string().contains("tokenish"),
            "unexpected error: {err}"
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
    fn test_invalid_slow_channel_backend_rejected_by_validate() {
        let mut cfg = ReinConfig::default();
        cfg.resummerize.llm_backend = "bogus".to_string();
        assert!(cfg.validate().is_err());

        let mut cfg = ReinConfig::default();
        cfg.ars.llm_backend = "bogus".to_string();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_slow_channel_batch_size_must_be_nonzero() {
        let mut cfg = ReinConfig::default();
        cfg.resummerize.batch_size = 0;
        assert!(cfg.validate().is_err());

        let mut cfg = ReinConfig::default();
        cfg.ars.batch_size = 0;
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

    #[test]
    #[serial_test::serial(global_state)]
    fn open_store_falls_back_to_single_store_when_pool_init_fails() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("fallback.db");
        let mut cfg = ReinConfig::default();
        cfg.database.path = db_path.to_string_lossy().into_owned();

        FAIL_NEXT_POOL_FOR_PATH.store(true, Ordering::SeqCst);

        let runtime = tokio::runtime::Runtime::new().unwrap();
        let store = runtime
            .block_on(async { cfg.open_store() })
            .expect("open_store should succeed even if pool init fails");

        assert_eq!(store.db_path(), db_path);
        assert!(
            store.pool().is_none(),
            "pool init failure should fall back to the already-open store"
        );
        assert!(
            store
                .conn()
                .query_row("SELECT 1", [], |row| row.get::<_, i64>(0))
                .is_ok(),
            "fallback store must remain usable"
        );
    }
}
