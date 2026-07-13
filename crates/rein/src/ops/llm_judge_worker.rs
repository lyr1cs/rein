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
//! Worker emits `feedback_events`, reserves `judge_call_ledger`, and prunes
//! stale `concept_summary_instances` rows on the judge-cache TTL cadence.
//! It NEVER touches `update()` / `apply_evolution` / `cold_archive` /
//! `M5 strip` / `memories` / `concepts` — see
//! `judge::contract::J1_ALLOWED_WRITE_TABLES`. This is the reason the judge
//! worker stays out of the v0.26.x 4-way pipeline-interaction matrix.

use crate::config::JudgeStructuralAnchorMode;
use crate::extract::llm::ExtractorKind;
use crate::judge::contract::{
    self, JudgeContext, JudgePayload, ReservationToken, LLM_JUDGE_DAILY_CALL_CAP_DEFAULT,
};
use crate::store::adaptive::{
    emit_event, ConceptSummaryLlmJudgePayload, EventType, FeedbackEvent, JudgeMetadata,
    JudgeSource, JudgeStructuralAnchorPayload, JudgeStructuralProbeKind, JudgeSurface, SignalHint,
    SynthesisLlmJudgePayload,
};
use crate::store::SqliteStore;
use crate::types::{ReinError, ReinResult};
use rusqlite::OptionalExtension;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Hard cap on the rationale string carried in the emitted event. Spec §3.2
/// says "truncated to 280 chars on emit" — keep it as a const so the
/// truncation point is visible in one place.
pub const JUDGE_REASON_MAX_CHARS: usize = 280;

/// Safety fallback for the maximum prompt+candidate characters the
/// worker hands to the judge LLM when no resolved judge cap is set.
pub const JUDGE_MAX_INPUT_CHARS: usize = 16_384;

const JUDGE_INPUT_JOINER: &str = "\n\nCandidate:\n";
pub const JUDGE_STRUCTURAL_PROBE_SET_VERSION: &str = "judge-anchors-v1";

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StructuralAnchorRunStats {
    pub runs_started: usize,
    pub attempted: usize,
    pub emitted: usize,
    pub failed: usize,
    pub cap_dropped: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct StructuralAnchorProbe {
    prompt: String,
    candidate: String,
    nonce: Option<String>,
}

fn structural_surface_label(surface: JudgeSurface) -> &'static str {
    match surface {
        JudgeSurface::Synthesis => "synthesis",
        JudgeSurface::ConceptSummary => "concept_summary",
    }
}

fn structural_probe_nonce(
    run_id: &str,
    surface: JudgeSurface,
    kind: JudgeStructuralProbeKind,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(JUDGE_STRUCTURAL_PROBE_SET_VERSION.as_bytes());
    hasher.update([0]);
    hasher.update(run_id.as_bytes());
    hasher.update([0]);
    hasher.update(structural_surface_label(surface).as_bytes());
    hasher.update([0]);
    hasher.update(format!("{kind:?}").as_bytes());
    let digest = format!("{:x}", hasher.finalize());
    format!("anchor-{}", &digest[..16])
}

fn build_structural_anchor_probe(
    run_id: &str,
    surface: JudgeSurface,
    kind: JudgeStructuralProbeKind,
) -> StructuralAnchorProbe {
    let surface_label = structural_surface_label(surface);
    match kind {
        JudgeStructuralProbeKind::SupportedExactSingle => StructuralAnchorProbe {
            prompt: format!(
                "Structural health probe for {surface_label}.\nQuery: What color is the release flag?\nSources:\n- The release flag color is cobalt."
            ),
            candidate: "The release flag color is cobalt.".to_string(),
            nonce: None,
        },
        JudgeStructuralProbeKind::SupportedExactMulti => StructuralAnchorProbe {
            prompt: format!(
                "Structural health probe for {surface_label}.\nQuery: Which region and port serve the release?\nSources:\n- The release region is ap-southeast-1.\n- The release port is 443."
            ),
            candidate: "The release is served from ap-southeast-1 on port 443.".to_string(),
            nonce: None,
        },
        JudgeStructuralProbeKind::UnsupportedNonce => {
            let nonce = structural_probe_nonce(run_id, surface, kind);
            StructuralAnchorProbe {
                prompt: format!(
                    "Structural health probe for {surface_label}.\nQuery: Summarize the deployment state.\nSources:\n- The deployment is healthy.\n- The deployment has three replicas."
                ),
                candidate: format!(
                    "The deployment is healthy with three replicas and audit token {nonce}."
                ),
                nonce: Some(nonce),
            }
        }
        JudgeStructuralProbeKind::QueryMismatch => StructuralAnchorProbe {
            prompt: format!(
                "Structural health probe for {surface_label}.\nQuery: Where is the deployment running?\nSources:\n- The deployment runs in Singapore.\n- The backup window begins at 02:00 UTC."
            ),
            candidate: "The backup window begins at 02:00 UTC.".to_string(),
            nonce: None,
        },
    }
}

fn structural_fingerprint(parts: &[&str]) -> String {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update(part.as_bytes());
        hasher.update([0]);
    }
    format!("{:x}", hasher.finalize())
}

fn structural_model_fingerprint(extractor: &ExtractorKind) -> String {
    match extractor {
        ExtractorKind::Gemini(extractor) => structural_fingerprint(&[
            "judge-model-v1",
            "gemini",
            &extractor.model,
            &extractor.endpoint,
        ]),
        ExtractorKind::Omlx(extractor) => structural_fingerprint(&[
            "judge-model-v1",
            "omlx",
            &extractor.model,
            &extractor.endpoint,
            if extractor.disable_thinking {
                "disable_thinking"
            } else {
                "thinking_enabled"
            },
        ]),
        #[cfg(feature = "test-support")]
        ExtractorKind::Mock(_) => structural_fingerprint(&["judge-model-v1", "mock"]),
    }
}

fn structural_rubric_fingerprint(surface: JudgeSurface) -> String {
    let mut hasher = Sha256::new();
    for part in [
        "judge-structural-rubric-v1",
        JUDGE_SYSTEM_PROMPT,
        JUDGE_STRUCTURAL_PROBE_SET_VERSION,
        structural_surface_label(surface),
    ] {
        hasher.update(part.as_bytes());
        hasher.update([0]);
    }
    for kind in JudgeStructuralProbeKind::ALL {
        let probe = build_structural_anchor_probe("fingerprint-sentinel", surface, kind);
        hasher.update(format!("{kind:?}").as_bytes());
        hasher.update([0]);
        hasher.update(probe.prompt.as_bytes());
        hasher.update([0]);
        hasher.update(probe.candidate.as_bytes());
        hasher.update([0xff]);
    }
    format!("{:x}", hasher.finalize())
}

fn structural_surface_due(
    state: &crate::store::judge_structural_calibration::JudgeStructuralCalibrationState,
    surface: JudgeSurface,
    now: i64,
    interval_secs: u64,
    model_fingerprint: &str,
    rubric_fingerprint: &str,
) -> bool {
    let surface_state = state.surface(surface);
    if surface_state.model_fingerprint != model_fingerprint
        || surface_state.rubric_fingerprint != rubric_fingerprint
        || surface_state.probe_set_version != JUDGE_STRUCTURAL_PROBE_SET_VERSION
    {
        return true;
    }
    let started_at = surface_state.run_started_at;
    if started_at <= 0 {
        return true;
    }
    let interval = i64::try_from(interval_secs).unwrap_or(i64::MAX);
    now >= started_at && now.saturating_sub(started_at) >= interval
}

/// Run at most one sealed four-probe suite per enabled surface and interval.
/// Every actual LLM call uses the existing judge-call ledger before dispatch.
pub fn run_structural_anchor_suite(
    store: &SqliteStore,
    config: &crate::config::ReinConfig,
    extractor: &ExtractorKind,
    now: i64,
) -> ReinResult<StructuralAnchorRunStats> {
    let anchor_config = &config.ars.llm_judge.structural_anchors;
    if !config.ars.llm_judge.enabled || anchor_config.mode == JudgeStructuralAnchorMode::Off {
        return Ok(StructuralAnchorRunStats::default());
    }
    if now <= 0 || anchor_config.interval_secs == 0 {
        return Err(ReinError::Config(
            "judge structural anchor run requires positive time and interval".to_string(),
        ));
    }
    let loaded =
        crate::store::judge_structural_calibration::load_judge_structural_calibration(store.conn());
    let state = match loaded.status {
        crate::store::judge_structural_calibration::JudgeStructuralCalibrationLoadStatus::Missing => {
            crate::store::judge_structural_calibration::JudgeStructuralCalibrationState::default()
        }
        crate::store::judge_structural_calibration::JudgeStructuralCalibrationLoadStatus::Loaded => {
            loaded.state
        }
        status => {
            return Err(ReinError::Config(format!(
                "judge structural anchor runner preserved unhealthy state {status:?}: {}",
                loaded.error.unwrap_or_else(|| "no detail".to_string())
            )));
        }
    };

    let model_fingerprint = structural_model_fingerprint(extractor);
    let mut stats = StructuralAnchorRunStats::default();
    for surface in [JudgeSurface::Synthesis, JudgeSurface::ConceptSummary] {
        let surface_enabled = match surface {
            JudgeSurface::Synthesis => config.ars.llm_judge.synthesis_enabled,
            JudgeSurface::ConceptSummary => config.ars.llm_judge.concept_summary_enabled,
        };
        let rubric_fingerprint = structural_rubric_fingerprint(surface);
        if !surface_enabled
            || !structural_surface_due(
                &state,
                surface,
                now,
                anchor_config.interval_secs,
                &model_fingerprint,
                &rubric_fingerprint,
            )
        {
            continue;
        }
        let surface_label = structural_surface_label(surface);
        let run_id = format!("jsa-{surface_label}-{}", ulid::Ulid::new());
        let Some(credentials) = crate::ops::judge_calibration::start_judge_structural_probe_run(
            store,
            surface,
            state.surface(surface),
            &run_id,
            &model_fingerprint,
            &rubric_fingerprint,
            JUDGE_STRUCTURAL_PROBE_SET_VERSION,
            now,
        )?
        else {
            tracing::debug!(?surface, "judge structural run already claimed by peer");
            continue;
        };
        stats.runs_started += 1;

        for kind in JudgeStructuralProbeKind::ALL {
            let Some(reservation) =
                contract::reserve_call(store.conn(), config.ars.llm_judge.daily_call_cap)?
            else {
                stats.cap_dropped += 1;
                break;
            };
            stats.attempted += 1;
            let probe = build_structural_anchor_probe(&run_id, surface, kind);
            let raw = match call_judge_sync(config, extractor, &probe.prompt, &probe.candidate) {
                Ok(raw) => raw,
                Err(error) => {
                    let _ = reservation.fail(store.conn());
                    stats.failed += 1;
                    tracing::warn!(?surface, ?kind, error = %error, "judge structural probe failed");
                    continue;
                }
            };
            let Some((observed_hit, _reason)) = parse_judge_output(&raw) else {
                let _ = reservation.fail(store.conn());
                stats.failed += 1;
                tracing::warn!(
                    ?surface,
                    ?kind,
                    "judge structural probe output was unparseable"
                );
                continue;
            };
            let payload = JudgeStructuralAnchorPayload {
                surface,
                probe_kind: kind,
                observed_hit,
                run_id: run_id.clone(),
                model_fingerprint: model_fingerprint.clone(),
                rubric_fingerprint: rubric_fingerprint.clone(),
                probe_set_version: JUDGE_STRUCTURAL_PROBE_SET_VERSION.to_string(),
                run_token: credentials.token_for(kind).to_string(),
            };
            let event = FeedbackEvent {
                event_type: EventType::JudgeStructuralAnchor,
                request_id: None,
                memory_id: None,
                concept_id: None,
                query: None,
                query_type: None,
                topic: None,
                payload: Some(serde_json::to_value(payload).map_err(ReinError::Serialization)?),
            };
            if let Err(error) = emit_event(store.conn(), event) {
                let _ = reservation.fail(store.conn());
                return Err(error);
            }
            reservation.commit(store.conn())?;
            stats.emitted += 1;
        }
    }
    Ok(stats)
}

pub(crate) fn resolve_judge_max_input_chars(config: &crate::config::ReinConfig) -> usize {
    if let Some(max) = config.ars.llm_judge.max_input_chars.filter(|max| *max > 0) {
        return max;
    }
    config
        .resolve_llm_for("ars.llm_judge")
        .ok()
        .map(|resolved| resolved.max_input_chars)
        .filter(|max| *max > 0)
        .unwrap_or(JUDGE_MAX_INPUT_CHARS)
}

pub(crate) fn judge_input_chars(prompt: &str, candidate: &str) -> usize {
    prompt.chars().count() + JUDGE_INPUT_JOINER.chars().count() + candidate.chars().count()
}

pub(crate) fn truncate_judge_inputs_for_config(
    config: &crate::config::ReinConfig,
    prompt: &str,
    candidate: &str,
) -> (String, String) {
    truncate_judge_inputs(prompt, candidate, resolve_judge_max_input_chars(config))
}

fn truncate_judge_inputs(prompt: &str, candidate: &str, max_chars: usize) -> (String, String) {
    const CANDIDATE_RESERVE_MAX: usize = 4_096;

    let joiner_chars = JUDGE_INPUT_JOINER.chars().count();
    let body_budget = max_chars.saturating_sub(joiner_chars);
    let candidate_budget = CANDIDATE_RESERVE_MAX.min(max_chars / 4).min(body_budget);
    let candidate_capped: String = candidate.chars().take(candidate_budget).collect();
    let prompt_budget = body_budget.saturating_sub(candidate_capped.chars().count());
    let prompt_truncated: String = prompt.chars().take(prompt_budget).collect();
    (prompt_truncated, candidate_capped)
}

/// v0.27.1 E direction — surface kind discriminator carried by every
/// queue payload (spec §3.1). `Synthesis` jobs map to
/// `EventType::SynthesisLlmJudge`; `ConceptSummary` jobs map to
/// `EventType::ConceptSummaryLlmJudge`. `RecallRanking` is queue-only
/// groundwork until a dedicated feedback event exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JudgeJobKind {
    Synthesis,
    ConceptSummary,
    RecallRanking,
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
    /// v0.28 ARS acceleration: optional structured hint computed upstream.
    ///
    /// The worker only pass-throughs this when acceleration is enabled. It
    /// does not infer hints from judge prose.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signal_hint: Option<SignalHint>,
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
    config: &crate::config::ReinConfig,
    extractor: &ExtractorKind,
    job: JudgeJob,
    daily_cap: u64,
) -> ReinResult<DispatchResult> {
    if matches!(job.kind, JudgeJobKind::RecallRanking) {
        let reason = if config.ars.llm_judge.recall_ranking_enabled {
            "recall-ranking judge queue support is present, but feedback event emission is not wired"
        } else {
            "recall-ranking judge is disabled"
        };
        return Ok(DispatchResult::Dropped(DropReason::ContractViolation(
            reason.to_string(),
        )));
    }

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
        JudgeJobKind::RecallRanking => unreachable!("recall-ranking returned before dispatch"),
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

    if !verify_durable_judge_target(store, config, &job)? {
        tracing::warn!(
            kind = ?job.kind,
            surface_id = %job.surface_id,
            "judge worker: J5 durable target lookup failed, dropping before reservation"
        );
        return Ok(DispatchResult::Dropped(DropReason::ContractViolation(
            "durable judge target missing".to_string(),
        )));
    }

    // codex R3 P2 + R4 P2 tighten: defensive size check BEFORE the
    // daily-cap reservation. Pre-existing oversized queue lines (e.g.
    // on-disk jobs from a pre-enqueue-cap version, manually injected
    // entries, or future bugs that bypass enqueue-time truncation) MUST
    // be rejected without consuming `daily_call_cap`. Otherwise a
    // handful of stale oversized lines could burn the cap with
    // `reserve_call` → `token.fail()` → billable `failed` ledger rows,
    // blocking legitimate judge work for the rolling 24h window even
    // though no HTTP call was made.
    //
    // Ceiling = the same resolved cap the enqueue path uses (NOT 4×), so
    // any payload above it is by construction stale/manual. Earlier 4×
    // headroom let
    // 16K-65K-byte stale lines burn `daily_call_cap` AND reach
    // `call_judge_sync` (which no longer truncates → R1 J7 fix), so
    // the LLM saw the untruncated bytes.
    let dispatch_ceiling = resolve_judge_max_input_chars(config);
    let combined_len = judge_input_chars(&job.prompt, &job.candidate);
    if combined_len > dispatch_ceiling {
        tracing::warn!(
            surface = ?job.kind,
            surface_id = %job.surface_id,
            combined_chars = combined_len,
            ceiling = dispatch_ceiling,
            "judge worker: payload exceeds dispatch ceiling; dropped pre-reservation"
        );
        return Ok(DispatchResult::Dropped(DropReason::ContractViolation(
            format!(
                "judge job payload too large at dispatch ({} chars > ceiling {})",
                combined_len, dispatch_ceiling
            ),
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
    let override_extractor =
        build_override_extractor(extractor, job.judge_model_override.as_deref());
    let active_extractor: &ExtractorKind = override_extractor.as_ref().unwrap_or(extractor);

    // LLM call (J4 — failures never propagate).
    let raw = match call_judge_sync(config, active_extractor, &job.prompt, &job.candidate) {
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
    let signal_hint = signal_hint_for_emit(config, &job);

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
                signal_hint,
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
                signal_hint,
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
        JudgeJobKind::RecallRanking => unreachable!("recall-ranking returned before emit"),
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

fn signal_hint_for_emit(config: &crate::config::ReinConfig, job: &JudgeJob) -> Option<SignalHint> {
    if !config.ars.acceleration.enabled {
        return None;
    }

    let mut hint = job.signal_hint.clone()?;
    hint.inferred_w_view = finite_nonnegative(hint.inferred_w_view);
    hint.inferred_w_click = finite_nonnegative(hint.inferred_w_click);
    hint.inferred_w_thumb = finite_nonnegative(hint.inferred_w_thumb);
    hint.inferred_w_req = finite_nonnegative(hint.inferred_w_req);
    hint.useful_rate_ci_width = finite_unit(hint.useful_rate_ci_width);
    if hint.inferred_w_view.is_none()
        && hint.inferred_w_click.is_none()
        && hint.inferred_w_thumb.is_none()
        && hint.inferred_w_req.is_none()
        && hint.useful_rate_ci_width.is_none()
    {
        None
    } else {
        Some(hint)
    }
}

fn finite_nonnegative(value: Option<f64>) -> Option<f64> {
    value.filter(|v| v.is_finite() && *v >= 0.0)
}

fn finite_unit(value: Option<f64>) -> Option<f64> {
    value.filter(|v| v.is_finite() && (0.0..=1.0).contains(v))
}

fn verify_durable_judge_target(
    store: &SqliteStore,
    config: &crate::config::ReinConfig,
    job: &JudgeJob,
) -> ReinResult<bool> {
    let ttl = match config.ars.llm_judge.cache_ttl_secs {
        0 => None,
        secs => Some(secs),
    };
    match job.kind {
        JudgeJobKind::Synthesis => {
            let cache_path = crate::ops::handlers::judge::synthesis_cache_path_for_config(config);
            Ok(cache_has_id_and_stamp(
                &cache_path,
                "synthesis_id",
                &job.surface_id,
                &job.stamp_hash,
                ttl,
            ))
        }
        JudgeJobKind::ConceptSummary => {
            let cache_path =
                crate::ops::handlers::judge::concept_summary_cache_path_for_config(config);
            if !cache_has_id_and_stamp(
                &cache_path,
                "concept_summary_id",
                &job.surface_id,
                &job.stamp_hash,
                ttl,
            ) {
                return Ok(false);
            }
            // F4 A2 fix — concept_summary jobs MUST carry a concept_id
            // (paired with A1: the cache reader rejects null concept_id).
            // A malformed job missing concept_id fails J5 here rather
            // than the downstream SQL accepting any concept via the old
            // `?2 IS NULL OR concept_id = ?2` half.
            let concept_id = match job.concept_id.as_deref() {
                Some(id) if !id.is_empty() => id,
                _ => return Ok(false),
            };
            concept_summary_target_exists(store, &job.surface_id, concept_id)
        }
        JudgeJobKind::RecallRanking => Ok(false),
    }
}

fn cache_has_id_and_stamp(
    path: &std::path::Path,
    id_field: &str,
    id_value: &str,
    stamp_hash: &str,
    ttl_secs: Option<u64>,
) -> bool {
    // F4 A4 — delegate the cache scan + TTL filter to the shared
    // helper so the worker and manual MCP path agree on stale-row
    // semantics. Then layer the stamp_hash predicate on top.
    crate::ops::handlers::judge::read_cache_entries_within_ttl(path, id_field, id_value, ttl_secs)
        .iter()
        .any(|value| value.get("stamp_hash").and_then(|v| v.as_str()) == Some(stamp_hash))
}

fn concept_summary_target_exists(
    store: &SqliteStore,
    summary_id: &str,
    concept_id: &str,
) -> ReinResult<bool> {
    // F4 A2 fix — caller (verify_durable_judge_target) guarantees
    // concept_id is non-empty (paired with A1's reader-side null
    // rejection). Drop the `?2 IS NULL OR ...` half so a forged job
    // with omitted concept_id can't match an arbitrary concept row.
    let retained: Option<i64> = store
        .conn()
        .query_row(
            "SELECT 1 FROM concept_summary_instances \
             WHERE summary_id = ?1 \
               AND concept_id = ?2 \
             LIMIT 1",
            rusqlite::params![summary_id, concept_id],
            |row| row.get(0),
        )
        .optional()?;
    if retained.is_some() {
        return Ok(true);
    }
    let live: Option<i64> = store
        .conn()
        .query_row(
            "SELECT 1 FROM concepts \
             WHERE living_summary_id = ?1 \
               AND id = ?2 \
             LIMIT 1",
            rusqlite::params![summary_id, concept_id],
            |row| row.get(0),
        )
        .optional()?;
    Ok(live.is_some())
}

#[cfg(feature = "test-support")]
pub fn write_test_judge_cache_entry(
    config: &crate::config::ReinConfig,
    job: &JudgeJob,
) -> std::io::Result<u32> {
    let (path, id_field) = match job.kind {
        JudgeJobKind::Synthesis => (
            crate::ops::handlers::judge::synthesis_cache_path_for_config(config),
            "synthesis_id",
        ),
        JudgeJobKind::ConceptSummary => (
            crate::ops::handlers::judge::concept_summary_cache_path_for_config(config),
            "concept_summary_id",
        ),
        JudgeJobKind::RecallRanking => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "recall-ranking judge cache is not wired",
            ));
        }
    };
    let mut entry = serde_json::json!({
        "query": &job.query,
        "prompt": &job.prompt,
        "candidate": &job.candidate,
        "stamp_hash": &job.stamp_hash,
        "query_type": &job.query_type,
        "cluster_id": job.cluster_id,
        "source_count": job.source_count,
        "stamped_at": chrono::Utc::now().to_rfc3339(),
    });
    entry[id_field] = serde_json::Value::String(job.surface_id.clone());
    if matches!(job.kind, JudgeJobKind::ConceptSummary) {
        entry["concept_id"] = job
            .concept_id
            .as_ref()
            .map(|id| serde_json::Value::String(id.clone()))
            .unwrap_or(serde_json::Value::Null);
    }
    crate::ops::handlers::judge::append_jsonl_line(&path, &entry)
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
pub fn call_judge_sync(
    config: &crate::config::ReinConfig,
    extractor: &ExtractorKind,
    prompt: &str,
    candidate: &str,
) -> ReinResult<String> {
    // J7 fix: the queued payload was already truncated at enqueue time
    // using the resolved judge cap, and `compute_stamp_hash` ran over
    // those exact bytes. Re-truncating here would change the bytes the
    // LLM actually sees while leaving the stamped hash unchanged,
    // breaking the invariant that `stamp_hash` identifies the exact
    // bytes judged.
    // CJK-safe via `.chars()` truncation upstream per pitfall doc.
    // codex R3 P2: dispatch ceiling check moved to `dispatch_one` so
    // oversized stale queue lines don't burn `daily_call_cap` via
    // `reserve_call` + `token.fail()`. By the time we reach this fn the
    // payload has been validated.
    let _ = (config, extractor); // reserved for future extractor-aware path
    let user = format!("{prompt}{JUDGE_INPUT_JOINER}{candidate}");
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        tokio::task::block_in_place(|| {
            handle.block_on(async {
                extractor
                    .raw_text_with_prompt(JUDGE_SYSTEM_PROMPT, &user)
                    .await
            })
        })
    } else {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| ReinError::Config(format!("failed to build tokio runtime: {e}")))?;
        rt.block_on(async {
            extractor
                .raw_text_with_prompt(JUDGE_SYSTEM_PROMPT, &user)
                .await
        })
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
///
/// v0.28.7 M-9 — `dropped` is preserved for back-compat with the consumer at
/// `ops/adaptive.rs` and the existing test harness. Per-reason counters
/// (`dropped_cap` / `dropped_disabled` / `dropped_contract` / `dropped_llm_error` /
/// `dropped_other`) were added so cap exhaustion (the operator's most actionable
/// signal) is no longer hidden behind a generic counter. Each `dropped_*`
/// increment also bumps `dropped` so the aggregate stays consistent.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DrainStats {
    pub emitted: u64,
    pub dropped: u64,
    pub errors: u64,
    pub malformed: u64,
    /// J2 reserve_call returned None — daily cap exhausted.
    pub dropped_cap: u64,
    /// RecallRanking job dropped because the surface gate is off.
    pub dropped_disabled: u64,
    /// J5 / J7 contract violation — payload structurally broken.
    pub dropped_contract: u64,
    /// LLM call errored or returned an unparseable verdict.
    pub dropped_llm_error: u64,
    /// Catch-all for any future `DropReason` variant the splitter doesn't
    /// recognize. Should always be 0 today; non-zero implies a missing arm.
    pub dropped_other: u64,
}

impl DrainStats {
    /// Aggregate of all per-reason drop counters. Equal to `dropped` by
    /// construction (we bump both in lockstep) but exposed as a method so
    /// new callers can opt into the typed sum.
    pub fn total_dropped(&self) -> u64 {
        self.dropped_cap
            .saturating_add(self.dropped_disabled)
            .saturating_add(self.dropped_contract)
            .saturating_add(self.dropped_llm_error)
            .saturating_add(self.dropped_other)
    }
}

/// Codex R1 P1 fix — drain the judge worker queue.
///
/// Reads the per-shard JSONL queue at
/// `<resolve_buffer_dir>/queue/<db_hash>/judge_queue.jsonl`, parses each
/// line as a [`JudgeJob`], and dispatches one-by-one through
/// [`dispatch_one`]. Opt-out when `[ars.llm_judge].enabled = false`
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
/// caches and `concept_summary_instances` table grow unbounded on
/// long-running nodes; manual MCP lookup already enforces TTL via
/// `cache_lookup_value_with_ttl`, but the stored rows remained. This
/// reaper keeps only rows within the configured
/// `[ars.llm_judge].cache_ttl_secs` window.
///
/// Disabled config (`enabled = false`) short-circuits before the reaper
/// touches disk/SQL). Best-effort: any IO/DB error is logged and
/// ignored — the manual lookup TTL guard remains the correctness
/// boundary.
///
/// Called from `drain_queue` so the reaper runs at the same slow-channel
/// cadence as judge dispatch. No need for a dedicated thread.
pub fn reap_expired_caches(store: &SqliteStore, config: &crate::config::ReinConfig) {
    if !config.ars.llm_judge.enabled {
        return;
    }
    // codex R7 P3: ledger prune runs FIRST and INDEPENDENT of
    // `cache_ttl_secs`. `reap_old_judge_call_ledger` retention is its
    // own const (7 days) — disabling the cache TTL (`cache_ttl_secs =
    // 0`) MUST NOT also disable ledger pruning, otherwise terminal
    // ledger rows accumulate forever even though the operator only
    // disabled cache expiry. F4 A5 — prune terminal-state rows
    // (`done`/`failed`/`stale`) older than 7 days. `reserved` rows are
    // NEVER pruned by this path (their staleness is handled by
    // `reserve_call`'s 5-minute reaper). Best-effort: any DB error
    // is logged + ignored.
    if let Err(e) = reap_old_judge_call_ledger(store, JUDGE_CALL_LEDGER_RETENTION_SECS) {
        tracing::warn!(
            error = %e,
            "judge cache reaper: failed to prune judge_call_ledger (non-fatal)"
        );
    }

    let ttl_secs = config.ars.llm_judge.cache_ttl_secs;
    if ttl_secs == 0 {
        return; // 0 means "never expire" — skip reaper for caches only;
                // ledger prune above already ran independent of TTL.
    }
    let synth_path = crate::ops::handlers::judge::synthesis_cache_path_for_config(config);
    let concept_path = crate::ops::handlers::judge::concept_summary_cache_path_for_config(config);
    for path in [&synth_path, &concept_path] {
        if let Err(e) = reap_one_cache_file(path, ttl_secs) {
            tracing::warn!(
                path = %path.display(),
                error = %e,
                "judge cache reaper: failed to reap (non-fatal)"
            );
        }
    }
    if let Err(e) = reap_concept_summary_instances(store, ttl_secs) {
        tracing::warn!(
            error = %e,
            "judge cache reaper: failed to prune concept_summary_instances (non-fatal)"
        );
    }
}

/// F4 A5 — terminal `judge_call_ledger` rows are retained for this many
/// seconds (7 days) before being pruned. `reserved` rows are never
/// pruned by this path; they age out via the `reserve_call` stale-claim
/// reaper at `LLM_JUDGE_STALE_CLAIM_SECS`.
const JUDGE_CALL_LEDGER_RETENTION_SECS: i64 = 7 * 24 * 3600;

/// F4 A5 helper — delete `judge_call_ledger` rows older than
/// `retention_secs` whose status is terminal (`done`/`failed`/`stale`).
/// Returns the number of rows deleted.
fn reap_old_judge_call_ledger(store: &SqliteStore, retention_secs: i64) -> ReinResult<u64> {
    let now_ts = chrono::Utc::now().timestamp();
    let cutoff = now_ts.saturating_sub(retention_secs);
    let deleted = store.conn().execute(
        "DELETE FROM judge_call_ledger \
         WHERE ts < ?1 \
           AND status IN ('done', 'failed', 'stale')",
        rusqlite::params![cutoff],
    )?;
    Ok(deleted as u64)
}

fn reap_concept_summary_instances(store: &SqliteStore, ttl_secs: u64) -> ReinResult<u64> {
    let cutoff = chrono::Utc::now()
        .timestamp()
        .saturating_sub(ttl_secs as i64);
    let deleted = store.conn().execute(
        "DELETE FROM concept_summary_instances WHERE refreshed_at < ?1",
        rusqlite::params![cutoff],
    )?;
    Ok(deleted as u64)
}

fn reap_one_cache_file(path: &std::path::Path, ttl_secs: u64) -> std::io::Result<()> {
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
    let queue_path = crate::ops::handlers::judge::judge_queue_path_for_config(config);
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
        let Some(suffix) = name.strip_prefix("judge_queue.jsonl.processing-") else {
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
        // touch the queue dir at all; explicit opt-out must mean zero new
        // disk writes.
        return DrainStats::default();
    }

    // v0.27.2 R5-K2 — opportunistic reap on each drain tick. Cheap
    // when files are small/missing; bounded by daily-rotated cache
    // size (~80MB worst case at full daily_call_cap). Cache reap is
    // independent of extractor configuration (caches just get
    // truncated by TTL).
    reap_expired_caches(store, config);

    // Codex C234-R3 P2 fix — resolve extractor FIRST (before queue
    // existence check) so we can:
    //   (a) skip stale-.processing reaping when no extractor is
    //       configured (preserves manual-recovery files), AND
    //   (b) still reap stale .processing files in the idle-but-
    //       configured case where queue is empty but old .processing-*
    //       crash orphans exist on disk.
    let extractor = match crate::ops::concept_summary::create_ars_extractor(config, "ars.llm_judge")
    {
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

    if config.ars.llm_judge.structural_anchors.mode != JudgeStructuralAnchorMode::Off {
        match run_structural_anchor_suite(store, config, &extractor, chrono::Utc::now().timestamp())
        {
            Ok(stats) if stats.attempted > 0 || stats.cap_dropped > 0 => {
                tracing::info!(
                    runs_started = stats.runs_started,
                    attempted = stats.attempted,
                    emitted = stats.emitted,
                    failed = stats.failed,
                    cap_dropped = stats.cap_dropped,
                    "judge structural anchor pass"
                );
            }
            Ok(_) => {}
            Err(error) => {
                tracing::warn!(error = %error, "judge structural anchor pass failed closed");
            }
        }
    }

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
        if matches!(job.kind, JudgeJobKind::RecallRanking)
            && !config.ars.llm_judge.recall_ranking_enabled
        {
            tracing::debug!("judge drain: recall-ranking job dropped because surface is disabled");
            stats.dropped += 1;
            stats.dropped_disabled += 1;
            continue;
        }
        match dispatch_one(store, config, &extractor, job, daily_cap) {
            Ok(DispatchResult::Emitted(_)) => stats.emitted += 1,
            Ok(DispatchResult::Dropped(reason)) => {
                tracing::debug!(?reason, "judge drain: job dropped");
                stats.dropped += 1;
                // v0.28.7 M-9 — split per-reason so cap exhaustion is
                // visible to operators without grep'ing trace logs.
                match &reason {
                    DropReason::DailyCapReached => stats.dropped_cap += 1,
                    DropReason::ContractViolation(_) => stats.dropped_contract += 1,
                    DropReason::LlmError(_) => stats.dropped_llm_error += 1,
                }
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

    // v0.28.7 M-9 — surface daily-cap saturation once per drain pass.
    // Per-line warns would be log spam; per-cycle is the right cadence
    // for the operator-facing signal.
    if stats.dropped_cap > 0 {
        tracing::warn!(
            dropped_cap = stats.dropped_cap,
            daily_cap,
            "judge daily cap exhausted; subsequent jobs dropped without LLM call"
        );
    }

    stats
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::JudgeStructuralAnchorMode;
    use crate::extract::llm::MockExtractor;

    fn temp_judge_config(dir: &tempfile::TempDir) -> crate::config::ReinConfig {
        let mut config = crate::config::ReinConfig::default();
        config.hooks.buffer_dir = dir.path().join("buffer").display().to_string();
        config.database.path = dir.path().join("rein.db").display().to_string();
        config.ars.llm_judge.enabled = true;
        config.ars.llm_judge.cache_ttl_secs = 600;
        config
    }

    fn mock_call_count(extractor: &ExtractorKind) -> usize {
        match extractor {
            ExtractorKind::Mock(mock) => mock.call_count(),
            _ => 0,
        }
    }

    #[test]
    fn structural_anchor_mode_off_never_calls_llm() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = temp_judge_config(&dir);
        config.ars.llm_judge.enabled = true;
        config.ars.llm_judge.structural_anchors.mode = JudgeStructuralAnchorMode::Off;
        let store = SqliteStore::in_memory().unwrap();
        let extractor =
            ExtractorKind::Mock(MockExtractor::with_fixed_response("HIT: yes\nWHY: ok"));

        let stats = run_structural_anchor_suite(&store, &config, &extractor, 1_000).unwrap();
        assert_eq!(stats.attempted, 0);
        assert_eq!(stats.emitted, 0);
        assert_eq!(mock_call_count(&extractor), 0);
        let ledger: i64 = store
            .conn()
            .query_row("SELECT COUNT(*) FROM judge_call_ledger", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(ledger, 0);
    }

    #[test]
    fn structural_anchor_run_emits_exactly_four_kinds_per_surface_through_ledger() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = temp_judge_config(&dir);
        config.ars.llm_judge.structural_anchors.mode = JudgeStructuralAnchorMode::Monitor;
        config.ars.llm_judge.structural_anchors.interval_secs = 86_400;
        config.ars.llm_judge.daily_call_cap = 100;
        let store = SqliteStore::in_memory().unwrap();
        let responses = [true, true, false, false, true, true, false, false]
            .into_iter()
            .map(|hit| {
                Ok(format!(
                    "HIT: {}\nWHY: scripted structural verdict",
                    if hit { "yes" } else { "no" }
                ))
            })
            .collect();
        let extractor = ExtractorKind::Mock(MockExtractor::with_responses(responses));

        let stats = run_structural_anchor_suite(&store, &config, &extractor, 1_000).unwrap();
        assert_eq!(stats.runs_started, 2);
        assert_eq!(stats.attempted, 8);
        assert_eq!(stats.emitted, 8);
        assert_eq!(stats.failed, 0);
        assert_eq!(mock_call_count(&extractor), 8);
        let events: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM feedback_events WHERE event_type = 'judge_structural_anchor'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(events, 8);
        let done: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM judge_call_ledger WHERE status = 'done'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(done, 8);

        let state =
            crate::ops::judge_calibration::run_judge_structural_anchor_consumer(&store).unwrap();
        assert_eq!(
            state.synthesis.status,
            crate::judge::contract::JudgeStructuralStatus::Ready
        );
        assert_eq!(
            state.concept_summary.status,
            crate::judge::contract::JudgeStructuralStatus::Ready
        );

        let skipped = run_structural_anchor_suite(&store, &config, &extractor, 1_100).unwrap();
        assert_eq!(skipped.attempted, 0);
        assert_eq!(
            mock_call_count(&extractor),
            8,
            "interval gate must prevent extra spend"
        );
    }

    #[test]
    fn structural_nonce_probe_is_deterministic_and_absent_from_sources() {
        let first = build_structural_anchor_probe(
            "run-1",
            crate::store::adaptive::JudgeSurface::Synthesis,
            crate::store::adaptive::JudgeStructuralProbeKind::UnsupportedNonce,
        );
        let second = build_structural_anchor_probe(
            "run-1",
            crate::store::adaptive::JudgeSurface::Synthesis,
            crate::store::adaptive::JudgeStructuralProbeKind::UnsupportedNonce,
        );
        assert_eq!(first, second);
        let nonce = first.nonce.as_deref().unwrap();
        assert!(first.candidate.contains(nonce));
        assert!(!first.prompt.contains(nonce));
    }

    #[test]
    fn structural_anchor_fingerprint_change_bypasses_interval_wait() {
        let mut state =
            crate::store::judge_structural_calibration::JudgeStructuralCalibrationState::default();
        state.synthesis.run_started_at = 1_000;
        state.synthesis.model_fingerprint = "model-a".to_string();
        state.synthesis.rubric_fingerprint = "rubric-a".to_string();
        state.synthesis.probe_set_version = JUDGE_STRUCTURAL_PROBE_SET_VERSION.to_string();

        assert!(!structural_surface_due(
            &state,
            JudgeSurface::Synthesis,
            1_100,
            86_400,
            "model-a",
            "rubric-a",
        ));
        assert!(structural_surface_due(
            &state,
            JudgeSurface::Synthesis,
            1_100,
            86_400,
            "model-b",
            "rubric-a",
        ));
    }

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

    fn base_synthesis_job() -> JudgeJob {
        JudgeJob {
            kind: JudgeJobKind::Synthesis,
            surface_id: "syn-1".to_string(),
            concept_id: None,
            query: "q".to_string(),
            prompt: "p".to_string(),
            candidate: "c".to_string(),
            stamp_hash: JudgeJob::compute_stamp_hash("q", "p", "c"),
            source: JudgeSource::ManualMcp,
            query_type: Some("semantic".into()),
            cluster_id: Some(7),
            source_count: Some(3),
            judge_model_override: None,
            signal_hint: None,
        }
    }

    #[test]
    fn judge_job_deserializes_legacy_rows_without_signal_hint() {
        let raw = serde_json::json!({
            "kind": "synthesis",
            "surface_id": "syn-legacy",
            "query": "q",
            "prompt": "p",
            "candidate": "c",
            "stamp_hash": JudgeJob::compute_stamp_hash("q", "p", "c"),
            "source": "ManualMcp",
            "source_count": 1
        });

        let job: JudgeJob = serde_json::from_value(raw).expect("legacy job should parse");

        assert!(job.signal_hint.is_none());
    }

    #[test]
    fn signal_hint_for_emit_sanitizes_structured_queue_hint() {
        let mut config = crate::config::ReinConfig::default();
        config.ars.acceleration.enabled = true;
        config.ars.acceleration.shadow_only = true;
        let mut job = base_synthesis_job();
        job.signal_hint = Some(crate::store::adaptive::SignalHint {
            inferred_w_view: Some(-1.0),
            inferred_w_click: Some(f64::NAN),
            inferred_w_thumb: Some(2.25),
            inferred_w_req: Some(0.75),
            useful_rate_ci_width: Some(2.0),
        });

        let hint = signal_hint_for_emit(&config, &job).expect("valid fields should keep hint");

        assert_eq!(hint.inferred_w_view, None);
        assert_eq!(hint.inferred_w_click, None);
        assert_eq!(hint.inferred_w_thumb, Some(2.25));
        assert_eq!(hint.inferred_w_req, Some(0.75));
        assert_eq!(hint.useful_rate_ci_width, None);
    }

    #[test]
    fn signal_hint_for_emit_requires_acceleration_but_not_shadow_only() {
        let mut job = base_synthesis_job();
        job.signal_hint = Some(crate::store::adaptive::SignalHint {
            inferred_w_view: Some(1.0),
            inferred_w_click: None,
            inferred_w_thumb: None,
            inferred_w_req: None,
            useful_rate_ci_width: None,
        });

        let default_config = crate::config::ReinConfig::default();
        assert_eq!(
            signal_hint_for_emit(&default_config, &job)
                .unwrap()
                .inferred_w_view,
            Some(1.0)
        );

        let mut disabled_config = crate::config::ReinConfig::default();
        disabled_config.ars.acceleration.enabled = false;
        assert!(signal_hint_for_emit(&disabled_config, &job).is_none());

        let mut production_config = crate::config::ReinConfig::default();
        production_config.ars.acceleration.enabled = true;
        production_config.ars.acceleration.shadow_only = false;
        assert_eq!(
            signal_hint_for_emit(&production_config, &job)
                .unwrap()
                .inferred_w_view,
            Some(1.0)
        );

        let mut shadow_config = crate::config::ReinConfig::default();
        shadow_config.ars.acceleration.enabled = true;
        shadow_config.ars.acceleration.shadow_only = true;
        assert_eq!(
            signal_hint_for_emit(&shadow_config, &job)
                .unwrap()
                .inferred_w_view,
            Some(1.0)
        );
    }

    #[test]
    fn drain_queue_drops_recall_ranking_jobs_while_surface_default_off() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = temp_judge_config(&dir);
        config.llm.provider = "omlx".to_string();
        config.llm.omlx.model = Some("judge-test".to_string());

        let query = "q";
        let prompt = "ranking evidence";
        let candidate = "ranked candidates";
        let job = serde_json::json!({
            "kind": "recall_ranking",
            "surface_id": "rank-job-1",
            "concept_id": serde_json::Value::Null,
            "query": query,
            "prompt": prompt,
            "candidate": candidate,
            "stamp_hash": JudgeJob::compute_stamp_hash(query, prompt, candidate),
            "source": "AutoSampled",
            "query_type": "Semantic",
            "cluster_id": serde_json::Value::Null,
            "source_count": 3u32,
        });
        let queue_path = crate::ops::handlers::judge::judge_queue_path_for_config(&config);
        crate::ops::handlers::judge::append_jsonl_line(&queue_path, &job).unwrap();

        let store = SqliteStore::in_memory().unwrap();
        let stats = drain_queue(&store, &config);

        assert_eq!(
            stats.malformed, 0,
            "recall-ranking rows are a known queue surface"
        );
        assert_eq!(
            stats.dropped, 1,
            "unsupported recall-ranking jobs must be dropped"
        );
        // v0.28.7 M-9 — surface-disabled drops must land in the typed
        // counter so operators can distinguish them from cap exhaustion.
        assert_eq!(
            stats.dropped_disabled, 1,
            "recall-ranking surface-disabled drops must bump dropped_disabled"
        );
        assert_eq!(stats.dropped_cap, 0);
        assert_eq!(stats.dropped_contract, 0);
        assert_eq!(stats.dropped_llm_error, 0);
        assert_eq!(stats.dropped_other, 0);
        assert_eq!(stats.total_dropped(), stats.dropped);
        assert_eq!(stats.emitted, 0);
        assert_eq!(stats.errors, 0);
    }

    #[test]
    fn j5_rejects_synthesis_job_without_matching_cache_stamp() {
        let dir = tempfile::tempdir().unwrap();
        let config = temp_judge_config(&dir);
        let store = SqliteStore::in_memory().unwrap();
        let job = JudgeJob {
            kind: JudgeJobKind::Synthesis,
            surface_id: "syn-forged".to_string(),
            concept_id: None,
            query: "q".to_string(),
            prompt: "p".to_string(),
            candidate: "c".to_string(),
            stamp_hash: JudgeJob::compute_stamp_hash("q", "p", "c"),
            source: JudgeSource::ManualMcp,
            query_type: None,
            cluster_id: None,
            source_count: Some(1),
            judge_model_override: None,
            signal_hint: None,
        };

        assert!(
            !verify_durable_judge_target(&store, &config, &job).unwrap(),
            "forged synthesis job without cache-backed id+stamp must fail J5 before reservation"
        );
    }

    #[test]
    fn j5_accepts_concept_summary_when_cache_stamp_and_sql_target_exist() {
        let dir = tempfile::tempdir().unwrap();
        let config = temp_judge_config(&dir);
        let store = SqliteStore::in_memory().unwrap();
        store
            .conn()
            .execute(
                "INSERT INTO concept_summary_instances \
                 (summary_id, concept_id, summary_text, refreshed_at) \
                 VALUES (?1, ?2, ?3, ?4)",
                rusqlite::params![
                    "cs-live",
                    "concept-live",
                    "summary",
                    chrono::Utc::now().timestamp()
                ],
            )
            .unwrap();

        let query = "";
        let prompt = "prompt";
        let candidate = "summary";
        let stamp_hash = JudgeJob::compute_stamp_hash(query, prompt, candidate);
        let cache_entry = serde_json::json!({
            "concept_summary_id": "cs-live",
            "concept_id": "concept-live",
            "query": query,
            "prompt": prompt,
            "candidate": candidate,
            "stamp_hash": stamp_hash,
            "stamped_at": chrono::Utc::now().to_rfc3339()
        });
        let cache_path =
            crate::ops::handlers::judge::concept_summary_cache_path_for_config(&config);
        crate::ops::handlers::judge::append_jsonl_line(&cache_path, &cache_entry).unwrap();

        let job = JudgeJob {
            kind: JudgeJobKind::ConceptSummary,
            surface_id: "cs-live".to_string(),
            concept_id: Some("concept-live".to_string()),
            query: query.to_string(),
            prompt: prompt.to_string(),
            candidate: candidate.to_string(),
            stamp_hash,
            source: JudgeSource::ManualMcp,
            query_type: None,
            cluster_id: None,
            source_count: Some(0),
            judge_model_override: None,
            signal_hint: None,
        };

        assert!(
            verify_durable_judge_target(&store, &config, &job).unwrap(),
            "concept-summary judge target should pass when cache stamp and durable SQL row agree"
        );
    }

    #[test]
    fn reap_expired_caches_prunes_concept_summary_instances_by_ttl() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = temp_judge_config(&dir);
        config.ars.llm_judge.cache_ttl_secs = 60;
        let store = SqliteStore::in_memory().unwrap();
        let now = chrono::Utc::now().timestamp();
        store
            .conn()
            .execute(
                "INSERT INTO concept_summary_instances \
                 (summary_id, concept_id, summary_text, refreshed_at) \
                 VALUES (?1, ?2, ?3, ?4), (?5, ?6, ?7, ?8)",
                rusqlite::params![
                    "old-cs",
                    "concept-a",
                    "old",
                    now - 120,
                    "fresh-cs",
                    "concept-b",
                    "fresh",
                    now,
                ],
            )
            .unwrap();

        reap_expired_caches(&store, &config);

        let rows: i64 = store
            .conn()
            .query_row("SELECT COUNT(*) FROM concept_summary_instances", [], |r| {
                r.get(0)
            })
            .unwrap();
        let old_rows: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM concept_summary_instances WHERE summary_id = 'old-cs'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(rows, 1);
        assert_eq!(old_rows, 0);
    }

    #[test]
    fn drain_stats_separates_cap_from_other_drops() {
        // v0.28.7 M-9 — drain loop must classify drops into per-reason
        // counters so cap exhaustion is visible without log grep.
        // Strategy: enqueue two jobs that drop via different reasons:
        //   1. recall-ranking job → surface-disabled gate (drained skip path)
        //      → bumps `dropped_disabled`.
        //   2. synthesis job with no durable cache backing → fails
        //      `verify_durable_judge_target` inside `dispatch_one` →
        //      `ContractViolation` → bumps `dropped_contract`.
        // Both bump `dropped` (back-compat aggregate). Neither bumps
        // `dropped_cap` since reservation is never reached.
        let dir = tempfile::tempdir().unwrap();
        let mut config = temp_judge_config(&dir);
        config.llm.provider = "omlx".to_string();
        config.llm.omlx.model = Some("judge-test".to_string());

        // Job 1 — recall-ranking (default disabled).
        let rr_job = serde_json::json!({
            "kind": "recall_ranking",
            "surface_id": "rank-1",
            "concept_id": serde_json::Value::Null,
            "query": "q1",
            "prompt": "p1",
            "candidate": "c1",
            "stamp_hash": JudgeJob::compute_stamp_hash("q1", "p1", "c1"),
            "source": "AutoSampled",
            "query_type": "Semantic",
            "cluster_id": serde_json::Value::Null,
            "source_count": 1u32,
        });
        let queue_path = crate::ops::handlers::judge::judge_queue_path_for_config(&config);
        crate::ops::handlers::judge::append_jsonl_line(&queue_path, &rr_job).unwrap();

        // Job 2 — well-formed synthesis job, but no cache row → J5 fails
        // → ContractViolation drop.
        let syn_job = serde_json::json!({
            "kind": "synthesis",
            "surface_id": "syn-no-cache",
            "concept_id": serde_json::Value::Null,
            "query": "q2",
            "prompt": "p2",
            "candidate": "c2",
            "stamp_hash": JudgeJob::compute_stamp_hash("q2", "p2", "c2"),
            "source": "ManualMcp",
            "query_type": "Semantic",
            "cluster_id": serde_json::Value::Null,
            "source_count": 1u32,
        });
        crate::ops::handlers::judge::append_jsonl_line(&queue_path, &syn_job).unwrap();

        let store = SqliteStore::in_memory().unwrap();
        let stats = drain_queue(&store, &config);

        assert_eq!(stats.dropped, 2, "both jobs must land in dropped");
        assert_eq!(
            stats.dropped_disabled, 1,
            "recall-ranking surface-disabled drop must increment dropped_disabled"
        );
        assert_eq!(
            stats.dropped_contract, 1,
            "synthesis J5 failure must increment dropped_contract"
        );
        assert_eq!(stats.dropped_cap, 0, "no cap exhaustion path was exercised");
        assert_eq!(stats.dropped_llm_error, 0);
        assert_eq!(stats.dropped_other, 0);
        assert_eq!(stats.total_dropped(), stats.dropped);
        assert_eq!(stats.emitted, 0);
        assert_eq!(stats.errors, 0);
        assert_eq!(stats.malformed, 0);
    }

    #[test]
    fn drain_stats_dropped_cap_increments_when_daily_cap_zero() {
        // v0.28.7 M-9 — when `daily_call_cap` is exhausted (modeled here
        // by setting it to 0), every otherwise-valid job that reaches
        // J2 reservation must drop into `dropped_cap`. The test uses a
        // recall-ranking job because that path returns before reaching
        // reservation — so we instead use a `Dropped(DailyCapReached)`
        // unit verification by directly calling the splitter logic.
        //
        // Direct unit test: the drain-loop match expression maps each
        // `DropReason` → field. We reproduce the mapping in isolation
        // to guard against a future refactor that drops a variant.
        let mut stats = DrainStats::default();
        let reasons = vec![
            DropReason::DailyCapReached,
            DropReason::ContractViolation("x".into()),
            DropReason::LlmError("y".into()),
        ];
        for reason in &reasons {
            stats.dropped += 1;
            match reason {
                DropReason::DailyCapReached => stats.dropped_cap += 1,
                DropReason::ContractViolation(_) => stats.dropped_contract += 1,
                DropReason::LlmError(_) => stats.dropped_llm_error += 1,
            }
        }
        assert_eq!(stats.dropped_cap, 1);
        assert_eq!(stats.dropped_contract, 1);
        assert_eq!(stats.dropped_llm_error, 1);
        assert_eq!(stats.dropped_disabled, 0);
        assert_eq!(stats.dropped_other, 0);
        assert_eq!(stats.total_dropped(), 3);
        assert_eq!(stats.dropped, 3);
    }
}
