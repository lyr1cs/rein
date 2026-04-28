//! v0.27.1 E direction (spec §3 + §4 + §10) — runtime LLM judge worker.
//!
//! Pulls jobs from the per-shard JSONL queue, validates the J* invariants
//! against the carried payload (J7 stamp-time isolation: source bytes
//! travel inline with the job, never re-queried), reserves a J2 call-cap
//! slot, calls the configured judge LLM, and emits a
//! [`SynthesisLlmJudge`] / [`ConceptSummaryLlmJudge`] feedback event. All
//! failures are best-effort (J4): worker logs + drops; never propagates
//! to the recall critical path.
//!
//! [`SynthesisLlmJudge`]: crate::store::adaptive::EventType::SynthesisLlmJudge
//! [`ConceptSummaryLlmJudge`]: crate::store::adaptive::EventType::ConceptSummaryLlmJudge
//!
//! # Pipeline-interaction discipline (spec §5)
//!
//! Worker writes `feedback_events` + `judge_call_ledger` only. NEVER
//! touches `update()` / `apply_evolution` / `cold_archive` / `M5 strip` /
//! `memories` / `concepts` — see `judge::contract::J1_ALLOWED_WRITE_TABLES`.
//! This is the entire reason the judge worker stays out of the v0.26.x
//! 4-way pipeline-interaction matrix.

use crate::extract::llm::ExtractorKind;
use crate::judge::contract::{
    self, JudgeContext, JudgePayload, ReservationToken, LLM_JUDGE_DAILY_CALL_CAP_DEFAULT,
};
use crate::store::adaptive::{
    emit_event, ConceptSummaryLlmJudgePayload, EventType, FeedbackEvent, JudgeMetadata,
    JudgeSource, SynthesisLlmJudgePayload,
};
use crate::store::SqliteStore;
use crate::types::{ReinError, ReinResult};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Hard cap on the rationale string carried in the emitted event. Spec §3.2
/// says "truncated to 280 chars on emit" — keep it as a const so the
/// truncation point is visible in one place.
pub const JUDGE_REASON_MAX_CHARS: usize = 280;

/// Maximum input bytes the worker hands to the judge LLM. Mirrors the
/// 16K safety fallback used by `extract::llm::resolve_max_input_chars`
/// for non-1M-token Gemini families. Operators can override via
/// `[ars.llm_judge].max_input_chars`. For now this is a bootstrap const.
pub const JUDGE_MAX_INPUT_CHARS: usize = 16_384;

/// v0.27.1 E direction — surface kind discriminator carried by every
/// queue payload (spec §3.1). `Synthesis` jobs map to
/// `EventType::SynthesisLlmJudge`; `ConceptSummary` jobs map to
/// `EventType::ConceptSummaryLlmJudge`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JudgeJobKind {
    Synthesis,
    ConceptSummary,
}

/// v0.27.1 E direction — one row of the judge job queue
/// (`<resolve_buffer_dir>/queue/<db_hash>/judge_<shard>.jsonl`).
///
/// J7 invariant — the **post-truncation** prompt + candidate bytes the
/// runtime judge will actually score travel inline in `prompt` /
/// `candidate`, never re-queried from `memories` / `concepts` between
/// enqueue and judge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgeJob {
    pub kind: JudgeJobKind,
    /// Surface-id ULID. For `Synthesis`, this is the synthesis_id; for
    /// `ConceptSummary`, the concept_summary_id minted in
    /// `refresh_living_summary`.
    pub surface_id: String,
    /// Concept ULID (only set for `ConceptSummary`; `None` for
    /// `Synthesis`). Routed into the Cap A payload so the consumer can
    /// look up `concept_summary_instances` retention rows.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub concept_id: Option<String>,
    pub query: String,
    /// Post-truncation prompt the judge sees. Carried inline (J7).
    pub prompt: String,
    /// Synthesis prose / concept summary text being judged.
    pub candidate: String,
    /// `[query, prompt, candidate]` triple — concatenated and SHA-256'd
    /// at emit time to populate `payload.stamp_hash`. Computed once
    /// here so the cron's stamp-hash join is byte-deterministic.
    pub stamp_hash: String,
    pub source: JudgeSource,
    pub query_type: Option<String>,
    pub cluster_id: Option<i64>,
    pub source_count: Option<u32>,
    /// Codex R1 P3 fix — manual MCP `rein_judge_synthesis` /
    /// `rein_judge_concept_summary` callers may pass a per-call model
    /// override (e.g. for A/B comparison vs the configured judge model).
    /// `None` = use the resolver-resolved `[ars.llm_judge]` model. The
    /// worker reads this and falls back to its default extractor when
    /// `None`. Default-omit on serialize keeps back-compat with
    /// auto-sampled jsonl rows that pre-date this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge_model_override: Option<String>,
}

impl JudgeJob {
    /// Compute the canonical `stamp_hash` over `(query || prompt ||
    /// candidate)` bytes. Caller fills `self.stamp_hash` with this; the
    /// worker re-derives it from the same inputs and J7 enforces equality.
    pub fn compute_stamp_hash(query: &str, prompt: &str, candidate: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(query.as_bytes());
        hasher.update(b"\x00");
        hasher.update(prompt.as_bytes());
        hasher.update(b"\x00");
        hasher.update(candidate.as_bytes());
        let digest = hasher.finalize();
        let mut out = String::with_capacity(digest.len() * 2);
        for byte in digest.iter() {
            use std::fmt::Write as _;
            let _ = write!(&mut out, "{byte:02x}");
        }
        out
    }
}

/// v0.27.1 E direction — judge prompt template (spec §3.1 sketch).
/// The judge's job is to score a single synthesis / concept-summary
/// faithfulness against the carried prompt+candidate. Output format is
/// `HIT: yes|no\nWHY: <one sentence>` — prose-mode (`raw_text_with_prompt`),
/// NOT JSON-mode, per [[feedback_provider_shaped_tests]] guidance: mixing
/// prose-prompts with JSON-mode silently no-ops the pipeline (v0.23.0
/// resummerize bug).
const JUDGE_SYSTEM_PROMPT: &str = "You are an LLM-as-judge for a memory \
synthesis system. Given a query, a synthesis prose output, and the \
underlying source memories, decide whether the synthesis FAITHFULLY \
answers the query using ONLY the source memories. \
\n\nOutput format (strict): \
\nLine 1: `HIT: yes` if the synthesis is faithful to the sources and \
answers the query, OR `HIT: no` if it hallucinates / contradicts / \
fails to answer. \
\nLine 2: `WHY: <one short sentence rationale>` (under 280 characters). \
\nDo NOT add preamble, code fences, or extra lines.";

/// Result of a single dispatch — emitted (with the new event id) or
/// dropped (with a reason). Used by the integration test harness to
/// verify the worker honored J* invariants without spinning up a
/// long-running task.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DispatchResult {
    Emitted(i64),
    Dropped(DropReason),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DropReason {
    /// J2 cap reached — caller should retry next sweep.
    DailyCapReached,
    /// J5 / J7 invariant violation — payload structurally broken.
    ContractViolation(String),
    /// LLM call errored or returned an unparseable verdict (J4 — never
    /// blocks recall path).
    LlmError(String),
}

/// Synchronous dispatch entry point. The async equivalent
/// (`dispatch_async`) is a `tokio::task::spawn_blocking` wrapper around
/// this; for v0.27.1 we keep the worker synchronous to mirror the
/// resummerize / cold_archive_summary / concept_summary patterns and
/// avoid introducing new tokio-runtime invariants.
///
/// Per spec §13 OQ #1, this fn runs on a dedicated OS thread inside the
/// `rein` server process; revisit if memory pressure becomes a problem.
pub fn dispatch_one(
    store: &SqliteStore,
    extractor: &ExtractorKind,
    job: JudgeJob,
    daily_cap: u64,
) -> ReinResult<DispatchResult> {
    // J5 + J7 (link-present + stamp-hash) — defense-in-depth: the queue
    // payload is supposed to be well-formed, but a corrupt jsonl line
    // shouldn't bypass invariants.
    let computed_stamp = JudgeJob::compute_stamp_hash(&job.query, &job.prompt, &job.candidate);
    let ctx = JudgeContext {
        stamp_time_source: job.candidate.as_str(),
        computed_stamp_hash: computed_stamp.as_str(),
        // J3 is read by the worker only when raising the warm sample
        // rate (operator action). Per-job dispatch never raises sample
        // rate, so we pass dormant defaults. The κ floor enforcement
        // happens in `ops/judge_calibration.rs` (D agent territory).
        surface_kappa_pair_count: 0,
        surface_kappa: 0.0,
        raising_sample_rate: false,
    };

    // For invariant-validation only — re-construct a stub payload of the
    // right discriminant. The actual payload landing in feedback_events
    // is built later, after the LLM call.
    let pre_synth;
    let pre_concept;
    let pre_payload = match job.kind {
        JudgeJobKind::Synthesis => {
            pre_synth = SynthesisLlmJudgePayload {
                synthesis_id: job.surface_id.clone(),
                judge_model: String::new(),
                hit: false,
                reason: String::new(),
                stamp_hash: computed_stamp.clone(),
                source: job.source,
                metadata: None,
                signal_hint: None,
            };
            JudgePayload::Synthesis(&pre_synth)
        }
        JudgeJobKind::ConceptSummary => {
            pre_concept = ConceptSummaryLlmJudgePayload {
                concept_summary_id: job.surface_id.clone(),
                concept_id: job.concept_id.clone().unwrap_or_default(),
                judge_model: String::new(),
                hit: false,
                reason: String::new(),
                stamp_hash: computed_stamp.clone(),
                source: job.source,
                metadata: None,
                signal_hint: None,
            };
            JudgePayload::ConceptSummary(&pre_concept)
        }
    };
    if let Err(v) = contract::validate_pre_emit(&ctx, &pre_payload) {
        tracing::warn!(violation = %v, "judge worker: pre-emit invariant violated, dropping");
        return Ok(DispatchResult::Dropped(DropReason::ContractViolation(
            v.to_string(),
        )));
    }
    // J7 also guards against the queue-side hash being lazy / wrong.
    if computed_stamp != job.stamp_hash {
        tracing::warn!(
            payload_hash = %job.stamp_hash,
            computed_hash = %computed_stamp,
            "judge worker: J7 stamp_hash mismatch on queue payload, dropping"
        );
        return Ok(DispatchResult::Dropped(DropReason::ContractViolation(
            "stamp_hash queue mismatch".to_string(),
        )));
    }

    // J2 atomic reservation — runs `BEGIN IMMEDIATE` so concurrent
    // workers can't burst N×cap.
    let token = match contract::reserve_call(store.conn(), daily_cap)? {
        Some(t) => t,
        None => return Ok(DispatchResult::Dropped(DropReason::DailyCapReached)),
    };

    // v0.27.2 R8-P3 fix — real per-call judge_model_override. When the
    // manual MCP caller passed `judge_model_override = Some(model)`,
    // build an alternate extractor with that model name + same
    // provider/endpoint/api_key as the resolved `[ars.llm_judge]`.
    // The override extractor lives only for this dispatch_one call.
    // For Mock extractor (test-support), override is ignored — tests
    // script the response queue directly.
    let override_extractor = build_override_extractor(extractor, job.judge_model_override.as_deref());
    let active_extractor: &ExtractorKind = override_extractor.as_ref().unwrap_or(extractor);

    // LLM call (J4 — failures never propagate).
    let raw = match call_judge_sync(active_extractor, &job.prompt, &job.candidate) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(error = %e, "judge worker: LLM call failed, ledger row → failed");
            let _ = token.fail(store.conn());
            return Ok(DispatchResult::Dropped(DropReason::LlmError(e.to_string())));
        }
    };

    let (hit, reason) = match parse_judge_output(&raw) {
        Some(parsed) => parsed,
        None => {
            tracing::warn!(
                raw = %raw,
                "judge worker: LLM output unparseable (expected `HIT: yes|no\\nWHY: ...`), dropping"
            );
            let _ = token.fail(store.conn());
            return Ok(DispatchResult::Dropped(DropReason::LlmError(
                "unparseable verdict".to_string(),
            )));
        }
    };

    // Codex R2 P2 fix — judge_model_override telemetry. Per-call extractor
    // swap (build a new Gemini/Omlx with the override model) requires
    // resolver plumbing not present in v0.27.1; record the override as
    // the requested model name when set so audit trail reflects operator
    // intent, otherwise use the coarse extractor family id. v0.27.2 will
    // honor the override by constructing an alternate extractor; tracked
    // in spec §15 as known issue.
    // v0.27.2 R8-P3 — record the ACTUAL model used. When override
    // applied, that's the override-built extractor; otherwise the
    // configured judge extractor. `active_extractor` already points
    // at whichever was used for the LLM call above.
    let model_id = judge_model_id(active_extractor);
    let metadata = JudgeMetadata {
        query_type: job.query_type.clone(),
        cluster_id: job.cluster_id,
        source_count: job.source_count,
        judge_latency_ms: None,
    };

    // Build the final payload + emit event.
    let event = match job.kind {
        JudgeJobKind::Synthesis => {
            let payload = SynthesisLlmJudgePayload {
                synthesis_id: job.surface_id.clone(),
                judge_model: model_id,
                hit,
                reason: truncate_chars(&reason, JUDGE_REASON_MAX_CHARS),
                stamp_hash: computed_stamp,
                source: job.source,
                metadata: Some(metadata),
                signal_hint: None,
            };
            FeedbackEvent {
                event_type: EventType::SynthesisLlmJudge,
                request_id: None,
                memory_id: None,
                concept_id: None,
                query: Some(job.query.clone()),
                query_type: job.query_type.clone(),
                topic: None,
                payload: Some(serde_json::to_value(&payload).map_err(ReinError::Serialization)?),
            }
        }
        JudgeJobKind::ConceptSummary => {
            let payload = ConceptSummaryLlmJudgePayload {
                concept_summary_id: job.surface_id.clone(),
                concept_id: job.concept_id.clone().unwrap_or_default(),
                judge_model: model_id,
                hit,
                reason: truncate_chars(&reason, JUDGE_REASON_MAX_CHARS),
                stamp_hash: computed_stamp,
                source: job.source,
                metadata: Some(metadata),
                signal_hint: None,
            };
            FeedbackEvent {
                event_type: EventType::ConceptSummaryLlmJudge,
                request_id: None,
                memory_id: None,
                concept_id: job.concept_id.clone(),
                query: Some(job.query.clone()),
                query_type: job.query_type.clone(),
                topic: None,
                payload: Some(serde_json::to_value(&payload).map_err(ReinError::Serialization)?),
            }
        }
    };

    let event_id = match emit_event(store.conn(), event) {
        Ok(id) => id,
        Err(e) => {
            tracing::warn!(error = %e, "judge worker: emit_event failed; ledger row → failed");
            let _ = token.fail(store.conn());
            return Ok(DispatchResult::Dropped(DropReason::LlmError(e.to_string())));
        }
    };
    let _ = ReservationToken::commit(&token, store.conn());
    Ok(DispatchResult::Emitted(event_id))
}

/// Truncate a string to `max_chars` Unicode-scalar values (NOT bytes —
/// CJK-safe per [[feedback_rust_cjk_alphanumeric_trap]]).
fn truncate_chars(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    s.chars().take(max_chars).collect()
}

/// v0.27.2 R8-P3 — construct an alternate extractor with the same
/// provider + endpoint + api_key as the configured judge but with a
/// different model name. Returns `None` when override is absent /
/// empty / equals the current extractor's model (no-op).
///
/// For Gemini and OMLX, the model is a per-instance String, so
/// cloning the existing extractor's connection-shaped fields and
/// swapping just the model gives a working alternate. For Mock
/// (test-support), override is ignored — tests use scripted responses.
fn build_override_extractor(
    base: &ExtractorKind,
    override_model: Option<&str>,
) -> Option<ExtractorKind> {
    let model = override_model.filter(|s| !s.is_empty())?;
    match base {
        ExtractorKind::Gemini(g) => {
            if g.model == model {
                return None;
            }
            Some(ExtractorKind::Gemini(
                crate::extract::llm::GeminiExtractor::new(
                    g.api_key.clone(),
                    g.endpoint.clone(),
                    model.to_string(),
                ),
            ))
        }
        ExtractorKind::Omlx(o) => {
            if o.model == model {
                return None;
            }
            // Codex v0.27.2 R1 P2 fix — preserve base's
            // `disable_thinking`. v0 hardcoded `false` here, so an
            // operator with `extract.omlx.disable_thinking = true` who
            // used judge_model_override would see the override call
            // emit a different protocol than the configured judge,
            // changing output format / latency. Read base's value.
            Some(ExtractorKind::Omlx(
                crate::extract::llm::OmlxExtractor::new(
                    o.endpoint.clone(),
                    model.to_string(),
                    o.disable_thinking,
                ),
            ))
        }
        #[cfg(feature = "test-support")]
        ExtractorKind::Mock(_) => None,
    }
}

/// Identify the judge model on the emitted event using the SAME
/// `provider:model` format that `call_cron_judge` uses, so calibration
/// joins can match runtime + offline cron events that share a model.
/// Codex R9 P2 fix — v0 returned only "gemini"/"omlx" so two
/// operators on the same provider with different models couldn't tell
/// their events apart in calibration / audit queries.
fn judge_model_id(extractor: &ExtractorKind) -> String {
    match extractor {
        ExtractorKind::Gemini(g) => format!("gemini:{}", g.model),
        ExtractorKind::Omlx(o) => format!("omlx:{}", o.model),
        #[cfg(feature = "test-support")]
        ExtractorKind::Mock(_) => "mock".to_string(),
    }
}

/// Run the judge prompt synchronously over the configured extractor.
/// Mirrors `ops::concept_summary::call_llm_sync` so the runtime
/// invariants ("don't use `reqwest::blocking` inside tokio") stay
/// consistent across ARS modules.
pub fn call_judge_sync(extractor: &ExtractorKind, prompt: &str, candidate: &str) -> ReinResult<String> {
    // Codex R6 P2 fix — truncate the combined prompt+candidate to
    // JUDGE_MAX_INPUT_CHARS BEFORE the LLM call. v0 sent
    // `prompt.len() + candidate.len()` unbounded; large-context Cap B
    // can produce 100KB+ prompts, but `[ars.llm_judge]` may point at
    // a smaller / local model that overflows context or bills surprise
    // tokens. CJK-safe via `.chars()` truncation per pitfall doc.
    let combined = format!("{prompt}\n\nCandidate:\n{candidate}");
    let user: String = combined.chars().take(JUDGE_MAX_INPUT_CHARS).collect();
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        tokio::task::block_in_place(|| {
            handle
                .block_on(async { extractor.raw_text_with_prompt(JUDGE_SYSTEM_PROMPT, &user).await })
        })
    } else {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| ReinError::Config(format!("failed to build tokio runtime: {e}")))?;
        rt.block_on(async { extractor.raw_text_with_prompt(JUDGE_SYSTEM_PROMPT, &user).await })
    }
}

/// Parse the strict-format judge output `HIT: yes|no\nWHY: <reason>`.
/// Returns `None` on any deviation — caller logs + drops the event so
/// noise doesn't pollute the calibration κ. Trailing whitespace,
/// blank-line preamble, and case mismatches on `yes`/`no` are accepted.
pub fn parse_judge_output(raw: &str) -> Option<(bool, String)> {
    let mut hit: Option<bool> = None;
    let mut reason: Option<String> = None;
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = strip_prefix_ci(line, "HIT:") {
            let v = rest.trim().to_ascii_lowercase();
            hit = match v.as_str() {
                "yes" | "true" | "y" | "1" => Some(true),
                "no" | "false" | "n" | "0" => Some(false),
                _ => None,
            };
        } else if let Some(rest) = strip_prefix_ci(line, "WHY:") {
            reason = Some(rest.trim().to_string());
        }
    }
    match (hit, reason) {
        (Some(h), Some(r)) => Some((h, r)),
        _ => None,
    }
}

fn strip_prefix_ci<'a>(s: &'a str, prefix: &str) -> Option<&'a str> {
    let lower = s.to_ascii_lowercase();
    if lower.starts_with(&prefix.to_ascii_lowercase()) {
        Some(&s[prefix.len()..])
    } else {
        None
    }
}

/// Bootstrap default cap reader. v0.27.1 reads from a const; v0.27.2 will
/// read from `[ars.llm_judge].daily_call_cap` once B2 wires the resolver.
pub fn default_daily_call_cap() -> u64 {
    LLM_JUDGE_DAILY_CALL_CAP_DEFAULT
}

/// Stats from one drain pass. Logged at the end of `run_adaptive_pipeline`
/// for operator visibility into the worker tick.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DrainStats {
    pub emitted: u64,
    pub dropped: u64,
    pub errors: u64,
    pub malformed: u64,
}

/// Codex R1 P1 fix — drain the judge worker queue.
///
/// Reads the per-shard JSONL queue at
/// `<resolve_buffer_dir>/queue/<db_hash>/judge_queue.jsonl`, parses each
/// line as a [`JudgeJob`], and dispatches one-by-one through
/// [`dispatch_one`]. Default-off when `[ars.llm_judge].enabled = false`
/// (no queue file is ever written under that flag, so no-op fast).
///
/// Atomic file rotation: `judge_queue.jsonl` → `judge_queue.jsonl.processing`
/// before reading, so concurrent enqueues land in a fresh file. After the
/// drain finishes, the `.processing` file is removed. On crash mid-drain,
/// the `.processing` file remains and is NOT auto-replayed (avoids
/// double-charging the daily cap on partial work) — operators reap it
/// manually if needed; spec leaves a future "stale .processing reaper"
/// to v0.27.2.
///
/// Called on each `run_adaptive_pipeline` tick (slow channel) — same
/// cadence as M2/M3/M4/M5/M6 + synthesis_feedback / concept_summary_feedback
/// / judge_calibration consumers.
/// v0.27.2 R5-K2 fix — reap expired entries from the synthesis +
/// concept-summary judge rehydration caches. The append-only jsonl
/// caches grow unbounded on long-running nodes; manual MCP lookup
/// already enforces TTL via `cache_lookup_value_with_ttl`, but the
/// disk file kept all old rows. This reaper rewrites each cache file
/// in place, keeping only rows whose `stamped_at` is within the
/// configured `[ars.llm_judge].cache_ttl_secs` window.
///
/// Default-off (`enabled = false` short-circuits before the reaper
/// touches disk). Best-effort: any IO error is logged and ignored
/// — the manual lookup TTL guard remains the correctness boundary.
///
/// Called from `drain_queue` so the reaper runs at the same slow-channel
/// cadence as judge dispatch. No need for a dedicated thread.
pub fn reap_expired_caches(config: &crate::config::ReinConfig) {
    if !config.ars.llm_judge.enabled {
        return;
    }
    let ttl_secs = config.ars.llm_judge.cache_ttl_secs;
    if ttl_secs == 0 {
        return; // 0 means "never expire" — skip reaper.
    }
    let synth_path =
        crate::ops::handlers::judge::synthesis_cache_path_for_config(config);
    let concept_path =
        crate::ops::handlers::judge::concept_summary_cache_path_for_config(config);
    for path in [&synth_path, &concept_path] {
        if let Err(e) = reap_one_cache_file(path, ttl_secs) {
            tracing::warn!(
                path = %path.display(),
                error = %e,
                "judge cache reaper: failed to reap (non-fatal)"
            );
        }
    }
}

fn reap_one_cache_file(
    path: &std::path::Path,
    ttl_secs: u64,
) -> std::io::Result<()> {
    if !path.exists() {
        return Ok(());
    }
    // v0.27.2 R1 P2 fix — take the rotation lock FIRST, then read.
    // The v0 reaper read the snapshot OUTSIDE the lock, then renamed
    // a stale snapshot back over the live path; a concurrent append
    // landing between read and rename was silently lost.
    use std::os::fd::AsRawFd as _;
    let rotation_lock_path = path.with_extension("jsonl.rotation_lock");
    let rotation_lock_file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(&rotation_lock_path)
        .ok();
    if let Some(ref f) = rotation_lock_file {
        let _ = unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX) };
    }

    // Inside lock: read → filter → rewrite atomically. Release lock
    // only after rename. Concurrent appenders block on the lockfile
    // and unblock seeing the post-reap fresh inode.
    let result: std::io::Result<(usize, usize)> = (|| {
        let text = std::fs::read_to_string(path)?;
        let now = chrono::Utc::now();
        let mut keep: Vec<&str> = Vec::new();
        let mut total = 0usize;
        let mut expired = 0usize;
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            total += 1;
            let v: serde_json::Value = match serde_json::from_str(trimmed) {
                Ok(v) => v,
                Err(_) => {
                    expired += 1;
                    continue;
                }
            };
            let stamped_at = v.get("stamped_at").and_then(|x| x.as_str());
            let live = stamped_at
                .and_then(|ts| chrono::DateTime::parse_from_rfc3339(ts).ok())
                .map(|parsed| {
                    let age = now.signed_duration_since(parsed.with_timezone(&chrono::Utc));
                    age.num_seconds() <= ttl_secs as i64
                })
                .unwrap_or(false);
            if live {
                keep.push(trimmed);
            } else {
                expired += 1;
            }
        }
        if expired == 0 {
            return Ok((total, 0));
        }
        let tmp_path = path.with_extension(format!(
            "jsonl.reap-tmp-{}-{}",
            chrono::Utc::now().timestamp_millis(),
            std::process::id()
        ));
        let body = if keep.is_empty() {
            String::new()
        } else {
            let mut out = keep.join("\n");
            out.push('\n');
            out
        };
        std::fs::write(&tmp_path, &body)?;
        std::fs::rename(&tmp_path, path)?;
        Ok((total, expired))
    })();

    // Always release the lock, even on error.
    if let Some(ref f) = rotation_lock_file {
        let _ = unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_UN) };
    }
    drop(rotation_lock_file);

    let (total, expired) = result?;
    if expired > 0 {
        tracing::debug!(
            path = %path.display(),
            total,
            expired,
            kept = total - expired,
            "judge cache reaper: pruned"
        );
    }
    Ok(())
}

/// v0.27.x C4 — reap stale `judge_queue.jsonl.processing-{ts}-{pid}`
/// files left by crashed prior drains. After 24h orphans are freed.
/// Use the timestamp EMBEDDED IN THE FILENAME (millis since unix epoch)
/// — Codex C234 P2 fix — NOT mtime, because rename inherits the old
/// queue file's mtime and could falsely-mark in-progress batches as
/// stale. Also: 24h floor (not 1h) so the no-extractor restore path's
/// 1h-aged files are preserved for manual recovery.
/// Best-effort: errors logged and ignored.
fn reap_stale_processing_files(config: &crate::config::ReinConfig) {
    let queue_path =
        crate::ops::handlers::judge::judge_queue_path_for_config(config);
    let queue_dir = match queue_path.parent() {
        Some(d) => d,
        None => return,
    };
    let entries = match std::fs::read_dir(queue_dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    let now_ms = chrono::Utc::now().timestamp_millis();
    const STALE_AGE_MS: i64 = 24 * 3600 * 1000; // 24 hours
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        // Filename pattern: judge_queue.jsonl.processing-{ts_ms}-{pid}
        // Parse the timestamp; skip if it doesn't match the pattern.
        let Some(suffix) = name.strip_prefix("judge_queue.jsonl.processing-")
        else {
            continue;
        };
        let Some(ts_str) = suffix.split('-').next() else {
            continue;
        };
        let Ok(ts_ms) = ts_str.parse::<i64>() else {
            continue;
        };
        let age_ms = now_ms - ts_ms;
        if age_ms > STALE_AGE_MS {
            if let Err(e) = std::fs::remove_file(&path) {
                tracing::warn!(
                    path = %path.display(),
                    error = %e,
                    "judge drain: failed to reap stale .processing-* file"
                );
            } else {
                tracing::debug!(
                    path = %path.display(),
                    age_secs = age_ms / 1000,
                    "judge drain: reaped stale .processing-* file"
                );
            }
        }
    }
}

pub fn drain_queue(store: &SqliteStore, config: &crate::config::ReinConfig) -> DrainStats {
    use std::io::BufRead;

    if !config.ars.llm_judge.enabled {
        // Codex C234 P2 fix — fast no-op when judge is disabled. Don't
        // touch the queue dir at all; default-off must mean zero new
        // disk writes.
        return DrainStats::default();
    }

    // v0.27.2 R5-K2 — opportunistic reap on each drain tick. Cheap
    // when files are small/missing; bounded by daily-rotated cache
    // size (~80MB worst case at full daily_call_cap). Cache reap is
    // independent of extractor configuration (caches just get
    // truncated by TTL).
    reap_expired_caches(config);

    // Codex C234-R3 P2 fix — resolve extractor FIRST (before queue
    // existence check) so we can:
    //   (a) skip stale-.processing reaping when no extractor is
    //       configured (preserves manual-recovery files), AND
    //   (b) still reap stale .processing files in the idle-but-
    //       configured case where queue is empty but old .processing-*
    //       crash orphans exist on disk.
    let extractor =
        match crate::ops::concept_summary::create_ars_extractor(config, "ars.llm_judge") {
            Some(e) => e,
            None => {
                // No extractor — preserve any existing .processing-*
                // files for manual recovery; don't reap them. Cache
                // reaper above is harmless in this case.
                tracing::debug!(
                    "judge drain: no extractor configured for [ars.llm_judge]; \
                     skipping drain + .processing reap (preserving any \
                     existing files for manual recovery)"
                );
                return DrainStats::default();
            }
        };

    // v0.27.x C4 — reap stale .processing-{ts}-{pid} files from
    // crashed prior drains. Now safe: extractor resolved means any
    // .processing-* files older than 24h are real crash orphans,
    // not preserved-for-recovery batches from misconfig.
    reap_stale_processing_files(config);

    let queue_path = crate::ops::handlers::judge::judge_queue_path_for_config(config);
    if !queue_path.exists() {
        return DrainStats::default();
    }

    // Codex R9 P2 fix — use a unique-per-drain processing path so a
    // crashed prior drain's `.processing` file isn't clobbered by the
    // current rename. Suffix with the current timestamp + pid for
    // stat-able operator recovery.
    let processing_path = queue_path.with_extension(format!(
        "jsonl.processing-{}-{}",
        chrono::Utc::now().timestamp_millis(),
        std::process::id()
    ));

    // Codex R7 P2 fix — use a SEPARATE lockfile (`.lock` sibling) for
    // rotation+drain coordination, NOT the queue file inode itself.
    // R6's first attempt locked the queue file fd then renamed the file
    // out from under the lock — appenders waiting on flock would
    // unblock immediately after rename and write to the now-`.processing`
    // inode, which the drain would then `remove_file` and lose those
    // appended bytes. The lockfile sits at `judge_queue.jsonl.lock` and
    // is held for the ENTIRE rotation+drain duration so concurrent
    // appenders block on flock until drain finishes; meanwhile new
    // appenders simply find no `judge_queue.jsonl` (drain renamed it)
    // and create a fresh one. The append_jsonl_line helper takes the
    // SAME lockfile flock before any append.
    use std::os::fd::AsRawFd as _;
    let rotation_lock_path = queue_path.with_extension("jsonl.rotation_lock");
    let rotation_lock_file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(&rotation_lock_path)
        .ok();
    if let Some(ref f) = rotation_lock_file {
        // Block until we own the rotation lock — a concurrent appender
        // that already grabbed it must finish first; then we rotate.
        let _ = unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX) };
    }
    let rename_result = std::fs::rename(&queue_path, &processing_path);

    // Release the rotation lock IMMEDIATELY after rename completes —
    // Codex R8 P1 fix. R7 held the lock through the entire dispatch
    // loop (which can be thousands of slow HTTP calls) and blocked
    // concurrent enqueue paths waiting on the same lock, stalling
    // recall/MCP requests. The lock only needs to protect rotation
    // atomicity; once `.processing` exists, the drain reads it
    // independently of the (now-fresh) `judge_queue.jsonl`. Concurrent
    // appenders re-flock, see the queue file is fresh-or-missing,
    // create+append cleanly.
    if let Some(ref f) = rotation_lock_file {
        let _ = unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_UN) };
    }
    drop(rotation_lock_file);

    if let Err(e) = rename_result {
        if e.kind() != std::io::ErrorKind::NotFound {
            tracing::warn!("judge drain: rename queue→processing failed: {e}");
        }
        return DrainStats::default();
    }

    // Codex R2 P1 fix — read configured cap, not hardcoded default.
    // When an operator sets `[ars.llm_judge].daily_call_cap` lower
    // than 10000 (e.g. for cost control), the drain MUST honor it.
    let daily_cap = config.ars.llm_judge.daily_call_cap;

    let mut stats = DrainStats::default();

    let file = match std::fs::File::open(&processing_path) {
        Ok(f) => f,
        Err(e) => {
            tracing::warn!("judge drain: open processing failed: {e}");
            return stats;
        }
    };
    let reader = std::io::BufReader::new(file);
    for line in reader.lines() {
        let line = match line {
            Ok(l) if l.trim().is_empty() => continue,
            Ok(l) => l,
            Err(e) => {
                tracing::warn!("judge drain: read line: {e}");
                stats.errors += 1;
                continue;
            }
        };
        let job: JudgeJob = match serde_json::from_str(&line) {
            Ok(j) => j,
            Err(e) => {
                tracing::warn!(error = %e, "judge drain: malformed JudgeJob jsonl line, skipping");
                stats.malformed += 1;
                continue;
            }
        };
        match dispatch_one(store, &extractor, job, daily_cap) {
            Ok(DispatchResult::Emitted(_)) => stats.emitted += 1,
            Ok(DispatchResult::Dropped(reason)) => {
                tracing::debug!(?reason, "judge drain: job dropped");
                stats.dropped += 1;
            }
            Err(e) => {
                tracing::warn!("judge drain: dispatch error: {e}");
                stats.errors += 1;
            }
        }
    }

    // Best-effort cleanup. If removal fails, the next drain skips this
    // file (we only rename queue → processing, never the other way).
    // Codex R8 P1 fix — rotation lock was already released after the
    // rename, so this cleanup runs without blocking enqueue paths.
    if let Err(e) = std::fs::remove_file(&processing_path) {
        tracing::warn!("judge drain: remove processing file failed: {e}");
    }

    stats
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_judge_output_happy_path() {
        let raw = "HIT: yes\nWHY: The synthesis cites every source.";
        let (hit, reason) = parse_judge_output(raw).unwrap();
        assert!(hit);
        assert!(reason.contains("synthesis"));
    }

    #[test]
    fn parse_judge_output_lowercase_no() {
        let raw = "HIT: no\nWHY: Hallucinated `tantivy` version.";
        let (hit, _) = parse_judge_output(raw).unwrap();
        assert!(!hit);
    }

    #[test]
    fn parse_judge_output_swallows_blank_preamble() {
        let raw = "\n\nHIT: yes\nWHY: ok\n";
        let (hit, reason) = parse_judge_output(raw).unwrap();
        assert!(hit);
        assert_eq!(reason, "ok");
    }

    #[test]
    fn parse_judge_output_rejects_missing_why() {
        let raw = "HIT: yes\n";
        assert!(parse_judge_output(raw).is_none());
    }

    #[test]
    fn parse_judge_output_rejects_unparseable_verdict() {
        let raw = "HIT: maybe\nWHY: hedging\n";
        assert!(parse_judge_output(raw).is_none());
    }

    #[test]
    fn truncate_chars_preserves_short_strings() {
        assert_eq!(truncate_chars("hello", 10), "hello");
    }

    #[test]
    fn truncate_chars_caps_long_strings() {
        let s = "abcdefghij";
        assert_eq!(truncate_chars(s, 5), "abcde");
    }

    #[test]
    fn truncate_chars_cjk_safe() {
        // 5 Chinese characters; ASCII byte count would be 15 (3 each in UTF-8).
        let s = "一二三四五六七八九十";
        assert_eq!(truncate_chars(s, 3), "一二三");
    }

    #[test]
    fn stamp_hash_deterministic() {
        let h1 = JudgeJob::compute_stamp_hash("q", "p", "c");
        let h2 = JudgeJob::compute_stamp_hash("q", "p", "c");
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64); // SHA-256 → 64 hex chars
    }

    #[test]
    fn stamp_hash_changes_with_inputs() {
        let h1 = JudgeJob::compute_stamp_hash("q", "p", "c");
        let h2 = JudgeJob::compute_stamp_hash("q!", "p", "c");
        let h3 = JudgeJob::compute_stamp_hash("q", "p", "c!");
        assert_ne!(h1, h2);
        assert_ne!(h1, h3);
        assert_ne!(h2, h3);
    }
}
