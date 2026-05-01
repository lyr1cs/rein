//! v0.27.1 E direction Layer 2 — nightly stricter offline calibration cron
//! + `judge_calibration` M1 consumer + κ accumulator + drift alert
//! + `bootstrap_priors_from_corpus` v0.28 fixture bootstrap.
//!
//! Module owns three responsibilities:
//!
//! 1. **Cron job** ([`run_judge_calibration_cron`]): scans the cron-sample
//!    archive jsonl in the DB-scoped queue dir, joins each entry against the
//!    runtime judge's verdict in `feedback_events` by `synthesis_id`, re-judges
//!    with the stricter `[ars.llm_judge.nightly_cron]`-resolved LLM, and
//!    EMITS-ONLY (writes `SynthesisLlmJudgeOfflineCron` /
//!    `ConceptSummaryLlmJudgeOfflineCron` events to `feedback_events`). Cron
//!    NEVER writes `runtime_vs_offline_kappa` directly — the `judge_calibration`
//!    consumer does (Codex R6 P2 fix).
//!
//! 2. **`judge_calibration` M1 consumer**
//!    ([`recompute_judge_calibration_state`]): peeks the two OfflineCron event
//!    types past its own offset, joins (runtime_hit, cron_hit) pairs from the
//!    payload, recomputes `runtime_vs_offline_kappa`, and writes to
//!    `AdaptiveState.judge_calibration_state`. Bumps `judge_drift_alert` when
//!    κ < `JUDGE_DRIFT_THRESHOLD` (and pair count ≥ `JUDGE_DRIFT_MIN_PAIRS`).
//!    Strict 5-invariant compliance per
//!    [[feedback_event_sourced_state_invariant]] — peek + commit, watermark
//!    filter, applied-prefix bump, replay-drain, CAS merge.
//!
//! 3. **`bootstrap_priors_from_corpus`** ([`bootstrap_priors_from_corpus`]):
//!    v0.28 fixture bootstrap (§16.2). Default-off returns
//!    [`BootstrapPriors::const_defaults`] — pure const, no I/O, no LLM. When
//!    `[ars.acceleration].enabled=true`, a valid DB-scoped prior snapshot wins;
//!    otherwise an embedded S1 fixture corpus supplies explicit `signal_hint`
//!    labels.
//!
//! ## Cron is emit-only (§7 step 5; Codex R6 P2)
//!
//! All durable state writes (`runtime_vs_offline_kappa`, `judge_drift_alert`,
//! `recent_pairs_runtime_vs_offline`) happen exclusively in the consumer. The
//! cron writes ONLY `feedback_events` rows. This means a cron crash mid-run
//! leaves κ and the alert counter unchanged — the consumer recomputes
//! deterministically on its next pass over whatever events landed durably.
//!
//! ## Sample-rate is config-driven (§7; Codex R8 P2)
//!
//! Cron-archive eligibility is a deterministic SHA-256 hash of `synthesis_id`
//! against `[ars.llm_judge.nightly_cron].sample_rate`. v0 spec hardcoded
//! `mod 5 == 0` for 20% which would ignore operator-configured rates. We
//! provide [`should_archive_for_cron`] as a pure helper used both at
//! synthesis-mint time (E_INTEGRATE) and within tests.
//!
//! ## Queue paths are DB-scoped (§7; Codex R8 P2 / R9-K2)
//!
//! [`cron_archive_path`] resolves to
//! `<resolve_buffer_dir(config)>/queue/<db_hash>/synthesis_cron_archive_YYYYMMDD_<shard>.jsonl`
//! — separate Rein databases get isolated archives. Mirrors the
//! `extract::hooks::queue::project_scoped_path` layout used by
//! `memory_*.jsonl`.

use crate::config::ReinConfig;
use crate::eval::llm_judge::kappa_runtime_vs_offline;
use crate::store::adaptive::{
    commit_offset, emit_event, peek_events, AdaptiveState,
    ConceptSummaryLlmJudgeOfflineCronPayload, ConceptSummaryLlmJudgePayload, EventType,
    FeedbackEvent, JudgeCalibrationState, JudgeMetadata, JudgeSurface, SignalHint,
    SynthesisLlmJudgeOfflineCronPayload, SynthesisLlmJudgePayload, JUDGE_DRIFT_MIN_PAIRS,
    JUDGE_DRIFT_THRESHOLD, JUDGE_RUNTIME_VS_OFFLINE_PAIRS_CAP,
};
use crate::store::SqliteStore;
use crate::types::{ReinError, ReinResult};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

/// Fixed consumer name for `consumer_offsets` row.
pub const JUDGE_CALIBRATION_CONSUMER: &str = "judge_calibration";

/// Maximum events the consumer drains in one pass. Above this it re-enters
/// on the next slow-channel cycle. Mirrors `recompute_synthesis_feedback_stats`'
/// pathological upper-bound (50_000) — far above realistic per-pass volume.
const PEEK_BATCH_LIMIT: usize = 50_000;

// ── §16.2 v0.28 bootstrap_priors_from_corpus ────────────────────────────────

/// v0.28 entrypoint for offline Bayesian prior derivation from the fixture
/// corpus + production replay events. S1 derives from a dedicated checked-in,
/// compile-time embedded fixture corpus with explicit `signal_hint` labels;
/// replay remains handled by [`bootstrap_priors_from_replay`].
///
/// Current v0.28.0 ships the safe shadow foundation:
/// 1. Snapshot precedence through a DB-scoped `bootstrap_priors.json`
/// 2. Fixture bootstrap for deterministic cold-start priors
/// 3. Separate bounded production replay via [`bootstrap_priors_from_replay`]
///
/// Later production activation can:
/// 1. Run multi-param logistic regression / Bayesian posterior inference on
///    `signal_hint`-labeled events to estimate cluster-pooled priors for
///    W_VIEW / W_CLICK / W_THUMB / W_REQ / useful_rate threshold
/// 2. Apply hierarchical shrinkage (S2 brainstorm: topic → cluster →
///    memory) so cold clusters borrow same-topic prior
/// 3. Write to DB-scoped
///    `<resolve_buffer_dir>/queue/<db_hash>/bootstrap_priors.json` snapshot —
///    adaptive engine reads on boot and uses as Bayesian prior; production
///    feedback updates the posterior
///
/// **v0.28.0 S1**: default config still performs no file I/O and no LLM calls.
/// `[ars.acceleration].enabled=true` opts into reading a DB-scoped
/// `bootstrap_priors.json` snapshot, then the embedded fixture corpus when no
/// valid snapshot exists.
pub fn bootstrap_priors_from_corpus(config: &ReinConfig) -> ReinResult<BootstrapPriors> {
    if !config.ars.acceleration.enabled {
        return Ok(BootstrapPriors::const_defaults());
    }
    if let Some(priors) = load_bootstrap_priors_snapshot(config) {
        return Ok(priors);
    }
    let hints = load_signal_hints_from_fixture_corpus();
    Ok(derive_priors_from_signal_hints(&hints))
}

/// Opt-in v0.28 replay bootstrap from already-durable runtime judge events.
///
/// Default-off is deliberately pure with respect to the database: callers can
/// pass an unopened/missing schema test connection and still receive constants.
/// When enabled, a valid DB-scoped snapshot wins; otherwise this scans a bounded
/// recent window for explicit `signal_hint` labels.
pub fn bootstrap_priors_from_replay(
    config: &ReinConfig,
    conn: &Connection,
) -> ReinResult<BootstrapPriors> {
    if !config.ars.acceleration.enabled {
        return Ok(BootstrapPriors::const_defaults());
    }
    if let Some(priors) = load_bootstrap_priors_snapshot(config) {
        return Ok(priors);
    }
    let cutoff = chrono::Utc::now().timestamp() - 30 * 24 * 60 * 60;
    let hints = load_signal_hints_from_feedback_events(conn, cutoff, 50_000)?;
    Ok(derive_priors_from_signal_hints(&hints))
}

/// DB-scoped prior snapshot path for v0.28 acceleration bootstrap.
///
/// Kept under the same queue namespace as cron archives so multiple Rein
/// databases using the same `buffer_dir` do not share learned priors.
pub fn bootstrap_priors_snapshot_path(config: &ReinConfig) -> PathBuf {
    db_scoped_queue_dir(config, true).join("bootstrap_priors.json")
}

/// Bootstrap priors snapshot. Default-off = const defaults; v0.28+ = posterior-
/// derived from fixture corpus + production replay (see
/// [`bootstrap_priors_from_corpus`]).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct BootstrapPriors {
    pub w_view: f64,
    pub w_click: f64,
    pub w_thumb: f64,
    pub w_req: f64,
    pub useful_rate_threshold: f64,
    pub weight_decay_rate: f64,
    /// Confidence in this prior (Bayesian: pseudo-observation count).
    /// Default-off returns 0.0 (no production-derived prior).
    pub prior_confidence: f64,
}

impl BootstrapPriors {
    /// Hardcoded defaults used when acceleration is off or no usable bootstrap
    /// labels exist. Caller never branches on which path produced these.
    pub fn const_defaults() -> Self {
        Self {
            w_view: 1.0,
            w_click: 1.5,
            w_thumb: 2.0,
            w_req: 1.5,
            useful_rate_threshold: 0.5,
            weight_decay_rate: 0.3,
            prior_confidence: 0.0,
        }
    }
}

fn derive_priors_from_signal_hints(hints: &[SignalHint]) -> BootstrapPriors {
    let defaults = BootstrapPriors::const_defaults();
    let mut view = PriorAccumulator::default();
    let mut click = PriorAccumulator::default();
    let mut thumb = PriorAccumulator::default();
    let mut req = PriorAccumulator::default();
    let mut ci = PriorAccumulator::default();

    for hint in hints {
        view.push_nonnegative(hint.inferred_w_view);
        click.push_nonnegative(hint.inferred_w_click);
        thumb.push_nonnegative(hint.inferred_w_thumb);
        req.push_nonnegative(hint.inferred_w_req);
        ci.push_unit(hint.useful_rate_ci_width);
    }

    let confidence = view
        .count
        .max(click.count)
        .max(thumb.count)
        .max(req.count)
        .max(ci.count) as f64;
    if confidence == 0.0 {
        return defaults;
    }

    let useful_rate_threshold = ci
        .mean()
        .map(|width| (defaults.useful_rate_threshold + (width - 0.5) * 0.2).clamp(0.1, 0.9))
        .unwrap_or(defaults.useful_rate_threshold);

    BootstrapPriors {
        w_view: view.mean().unwrap_or(defaults.w_view),
        w_click: click.mean().unwrap_or(defaults.w_click),
        w_thumb: thumb.mean().unwrap_or(defaults.w_thumb),
        w_req: req.mean().unwrap_or(defaults.w_req),
        useful_rate_threshold,
        weight_decay_rate: defaults.weight_decay_rate,
        prior_confidence: confidence,
    }
}

fn extract_signal_hint_from_judge_event(event_type: &str, payload: &str) -> Option<SignalHint> {
    match event_type {
        "synthesis_llm_judge" => serde_json::from_str::<SynthesisLlmJudgePayload>(payload)
            .ok()
            .and_then(|p| p.signal_hint),
        "concept_summary_llm_judge" => {
            serde_json::from_str::<ConceptSummaryLlmJudgePayload>(payload)
                .ok()
                .and_then(|p| p.signal_hint)
        }
        _ => None,
    }
}

fn load_signal_hints_from_feedback_events(
    conn: &Connection,
    cutoff_unix_ts: i64,
    limit: usize,
) -> ReinResult<Vec<SignalHint>> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let capped_limit = limit.min(50_000) as i64;
    let mut stmt = conn.prepare(
        "SELECT event_type, payload FROM feedback_events
         WHERE event_type IN ('synthesis_llm_judge', 'concept_summary_llm_judge')
           AND payload IS NOT NULL
           AND ts > datetime(?1, 'unixepoch')
         ORDER BY id DESC
         LIMIT ?2",
    )?;
    let rows = stmt.query_map(rusqlite::params![cutoff_unix_ts, capped_limit], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?))
    })?;
    let mut hints = Vec::new();
    for row in rows {
        let (event_type, payload) = row?;
        let Some(payload) = payload else { continue };
        if let Some(hint) = extract_signal_hint_from_judge_event(&event_type, &payload) {
            hints.push(hint);
        }
    }
    Ok(hints)
}

fn load_signal_hints_from_fixture_corpus() -> Vec<SignalHint> {
    const ARS_BOOTSTRAP_SIGNAL_HINTS_JSON: &str =
        include_str!("../../tests/fixtures/ars_bootstrap/signal_hints.json");
    let Ok(rows) = serde_json::from_str::<Vec<serde_json::Value>>(ARS_BOOTSTRAP_SIGNAL_HINTS_JSON)
    else {
        return Vec::new();
    };
    rows.into_iter()
        .filter_map(|row| signal_hint_from_fixture_row(row.get("signal_hint")?))
        .collect()
}

fn signal_hint_from_fixture_row(value: &serde_json::Value) -> Option<SignalHint> {
    let object = value.as_object()?;
    Some(SignalHint {
        inferred_w_view: json_field_as_f64(object.get("inferred_w_view")),
        inferred_w_click: json_field_as_f64(object.get("inferred_w_click")),
        inferred_w_thumb: json_field_as_f64(object.get("inferred_w_thumb")),
        inferred_w_req: json_field_as_f64(object.get("inferred_w_req")),
        useful_rate_ci_width: json_field_as_f64(object.get("useful_rate_ci_width")),
    })
}

fn json_field_as_f64(value: Option<&serde_json::Value>) -> Option<f64> {
    value.and_then(serde_json::Value::as_f64)
}

fn load_bootstrap_priors_snapshot(config: &ReinConfig) -> Option<BootstrapPriors> {
    let raw = std::fs::read_to_string(bootstrap_priors_snapshot_read_path(config)).ok()?;
    let priors = serde_json::from_str::<BootstrapPriors>(&raw).ok()?;
    sanitize_bootstrap_priors(priors)
}

fn bootstrap_priors_snapshot_read_path(config: &ReinConfig) -> PathBuf {
    db_scoped_queue_dir(config, false).join("bootstrap_priors.json")
}

fn sanitize_bootstrap_priors(priors: BootstrapPriors) -> Option<BootstrapPriors> {
    let BootstrapPriors {
        w_view,
        w_click,
        w_thumb,
        w_req,
        useful_rate_threshold,
        weight_decay_rate,
        prior_confidence,
    } = priors;
    if [w_view, w_click, w_thumb, w_req, prior_confidence]
        .into_iter()
        .any(|value| !value.is_finite() || value < 0.0)
    {
        return None;
    }
    if !useful_rate_threshold.is_finite() || !(0.0..=1.0).contains(&useful_rate_threshold) {
        return None;
    }
    if !weight_decay_rate.is_finite() || !(0.0..=1.0).contains(&weight_decay_rate) {
        return None;
    }
    Some(BootstrapPriors {
        w_view,
        w_click,
        w_thumb,
        w_req,
        useful_rate_threshold,
        weight_decay_rate,
        prior_confidence,
    })
}

#[derive(Default)]
struct PriorAccumulator {
    sum: f64,
    count: u64,
}

impl PriorAccumulator {
    fn push_nonnegative(&mut self, value: Option<f64>) {
        let Some(value) = value else {
            return;
        };
        if value.is_finite() && value >= 0.0 {
            self.sum += value;
            self.count = self.count.saturating_add(1);
        }
    }

    fn push_unit(&mut self, value: Option<f64>) {
        let Some(value) = value else {
            return;
        };
        if value.is_finite() {
            self.sum += value.clamp(0.0, 1.0);
            self.count = self.count.saturating_add(1);
        }
    }

    fn mean(&self) -> Option<f64> {
        (self.count > 0).then_some(self.sum / self.count as f64)
    }
}

// ── §7 deterministic sample helper ──────────────────────────────────────────

/// Decide whether `synthesis_id` is in the nightly cron sample.
///
/// Deterministic SHA-256 hash of the id bytes mapped to a `[0.0, 1.0)`
/// fraction; sampled iff fraction `< rate`. Mint-time decision lets the cron
/// know exactly which synthesis_ids to expect in the archive without scanning
/// the runtime cache (improves cron determinism + reproducibility across
/// reruns).
///
/// `rate` is clamped to `[0.0, 1.0]`. `rate == 0.0` → never sample;
/// `rate == 1.0` → always sample. NaN / negative / >1 are treated as 0.0
/// (defensive — config validation should have caught these but the helper
/// must never panic).
///
/// The hash is taken over the first 8 bytes of the SHA-256 digest, big-endian
/// `u64`. SHA-256 has well-distributed output bytes — a 64-bit prefix is
/// statistically equivalent to a uniform sample for this use case.
pub fn should_archive_for_cron(synthesis_id: &str, rate: f64) -> bool {
    // NaN → 0.0 (defensive default: never sample on garbage). +inf clamps to
    // 1.0 (always sample), -inf clamps to 0.0. clamp on NaN propagates NaN
    // so we MUST gate is_nan first.
    let rate = if rate.is_nan() {
        0.0
    } else {
        rate.clamp(0.0, 1.0)
    };
    if rate == 0.0 {
        return false;
    }
    if rate >= 1.0 {
        return true;
    }
    let mut hasher = Sha256::new();
    hasher.update(synthesis_id.as_bytes());
    let digest = hasher.finalize();
    let mut prefix = [0u8; 8];
    prefix.copy_from_slice(&digest[..8]);
    let n = u64::from_be_bytes(prefix);
    let frac = (n as f64) / (u64::MAX as f64);
    frac < rate
}

// ── Queue path resolution (§7 / R9-K2 DB-scoped) ────────────────────────────

/// Resolve the cron archive jsonl path for a given UTC date + shard.
///
/// `<resolve_buffer_dir(config)>/queue/<db_hash>/synthesis_cron_archive_<YYYYMMDD>_<shard>.jsonl`
///
/// Mirrors `extract::hooks::queue::project_scoped_path` (private) — we
/// reproduce the exact `db_hash` derivation here to keep one durable layout
/// across both queue families. If `extract::hooks::queue` ever exposes the
/// helper publicly, this should switch to it (TODO Wave-1.5).
pub fn cron_archive_path(config: &ReinConfig, date: chrono::NaiveDate, shard: u32) -> PathBuf {
    let queue_dir = db_scoped_queue_dir(config, true);
    let date_str = date.format("%Y%m%d").to_string();
    queue_dir.join(format!("synthesis_cron_archive_{date_str}_{shard}.jsonl"))
}

fn db_scoped_queue_dir(config: &ReinConfig, create: bool) -> PathBuf {
    let base = crate::extract::hooks::buffer::resolve_buffer_dir(config);
    let db_tag = {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        config.resolve_db_path().hash(&mut h);
        format!("{:016x}", h.finish())
    };
    let queue_dir = base.join("queue").join(&db_tag);
    if create {
        let _ = std::fs::create_dir_all(&queue_dir);
    }
    queue_dir
}

// ── M1 consumer — `judge_calibration` ───────────────────────────────────────

/// Peek new `SynthesisLlmJudgeOfflineCron` + `ConceptSummaryLlmJudgeOfflineCron`
/// events past the consumer's offset, fold them into the rolling
/// `JudgeCalibrationState`, recompute `runtime_vs_offline_kappa`, and return
/// the highest event id incorporated so the caller can commit the consumer
/// offset *after* the derived state is durable.
///
/// Returns `(updated_state, Option<max_event_id>)`. `Option::None` means no
/// new events were observed → caller skips `commit_offset`.
///
/// **5 invariants enforced** (per [[feedback_event_sourced_state_invariant]]):
///
///   1. **Watermark filter** — events with `id <=
///      state.last_consumed_event_id_calibration` are skipped via
///      `prior_high_water`. κ-pair appends are NOT idempotent (they grow the
///      VecDeque), so this guard is the entire point.
///   2. **Applied-prefix bump** — `state.last_consumed_event_id_calibration`
///      is bumped to `max(state.last_consumed_event_id_calibration,
///      max_id_this_pass)` *before* any new events are folded; the caller
///      commits the consumer offset only AFTER `save_snapshot` returns Ok.
///   3. **Replay-drain** — `peek_events` reads from the consumer offset;
///      replay-safety after a `commit_offset` failure is guarded by (1).
///   4. **CAS merge** — `AdaptiveState::save_snapshot` arbitrates Layer 2
///      fields (incl. this state) by `last_consumed_event_id_calibration`
///      MAX. R9-K5 field-grouped merge protects Layer 1 fields owned by
///      other consumers.
///   5. **Peek + commit** — uses `peek_events("judge_calibration", …)`,
///      then *the caller* runs `commit_offset(&[("judge_calibration",
///      max_id)])` AFTER `save_snapshot` succeeds. Never `consume_events`.
///
/// Malformed payloads are logged via `tracing::warn!` and skipped (mirrors
/// `recompute_synthesis_feedback_stats`).
///
/// **Drift alert side-effect**: when `recent_pairs_runtime_vs_offline.len()
/// >= JUDGE_DRIFT_MIN_PAIRS` AND `runtime_vs_offline_kappa <
/// JUDGE_DRIFT_THRESHOLD`, the consumer bumps `judge_drift_alert` and writes a
/// one-line warning to `~/.rein/judge_drift.log` (best-effort — logging
/// failure does NOT abort the consumer). Doctor surfaces the alert.
pub fn recompute_judge_calibration_state(
    conn: &Connection,
    prior: Option<JudgeCalibrationState>,
    drift_log_path: Option<&std::path::Path>,
) -> ReinResult<(JudgeCalibrationState, Option<i64>)> {
    let mut state = prior.unwrap_or_default();

    let events = peek_events(
        conn,
        JUDGE_CALIBRATION_CONSUMER,
        &[
            EventType::SynthesisLlmJudgeOfflineCron.as_str(),
            EventType::ConceptSummaryLlmJudgeOfflineCron.as_str(),
        ],
        PEEK_BATCH_LIMIT,
    )?;

    if events.is_empty() {
        return Ok((state, None));
    }

    let max_id_this_pass = events.last().map(|e| e.id);

    // Invariants 1 + 2: prior_high_water guards against double-application
    // on replay; bump applied-prefix BEFORE folding so a later
    // `save_snapshot` carries the new watermark even if intermediate folding
    // partially fails.
    let prior_high_water = state.last_consumed_event_id_calibration;
    if let Some(max_id) = max_id_this_pass {
        state.last_consumed_event_id_calibration =
            state.last_consumed_event_id_calibration.max(max_id);
    }

    let now = chrono::Utc::now().timestamp();
    let mut any_new_pair = false;
    let mut any_new_synthesis_pair = false;
    let mut any_new_concept_pair = false;

    for ev in events {
        if ev.id <= prior_high_water {
            continue;
        }

        let payload_str = match ev.payload.as_deref() {
            Some(s) => s,
            None => {
                tracing::warn!(
                    event_id = ev.id,
                    "judge_calibration: event missing payload, skipping"
                );
                continue;
            }
        };

        let pair: Option<(JudgeSurface, bool, bool)> = match ev.event_type.as_str() {
            "synthesis_llm_judge_offline_cron" => {
                match serde_json::from_str::<SynthesisLlmJudgeOfflineCronPayload>(payload_str) {
                    Ok(p) => Some((JudgeSurface::Synthesis, p.runtime_hit, p.cron_hit)),
                    Err(e) => {
                        tracing::warn!(
                            event_id = ev.id,
                            error = %e,
                            "judge_calibration: malformed SynthesisLlmJudgeOfflineCronPayload, skipping"
                        );
                        None
                    }
                }
            }
            "concept_summary_llm_judge_offline_cron" => {
                match serde_json::from_str::<ConceptSummaryLlmJudgeOfflineCronPayload>(payload_str)
                {
                    Ok(p) => Some((JudgeSurface::ConceptSummary, p.runtime_hit, p.cron_hit)),
                    Err(e) => {
                        tracing::warn!(
                            event_id = ev.id,
                            error = %e,
                            "judge_calibration: malformed ConceptSummaryLlmJudgeOfflineCronPayload, skipping"
                        );
                        None
                    }
                }
            }
            other => {
                tracing::warn!(
                    event_id = ev.id,
                    event_type = %other,
                    "judge_calibration: unexpected event type, skipping"
                );
                None
            }
        };

        if let Some((surface, runtime, cron)) = pair {
            let ts = ev.ts_to_unix().unwrap_or(now);
            state
                .recent_pairs_runtime_vs_offline
                .push_back((runtime, cron, ts));
            // FIFO-evict oldest pair if over cap.
            while state.recent_pairs_runtime_vs_offline.len() > JUDGE_RUNTIME_VS_OFFLINE_PAIRS_CAP {
                state.recent_pairs_runtime_vs_offline.pop_front();
            }
            match surface {
                JudgeSurface::Synthesis => {
                    state
                        .recent_pairs_runtime_vs_offline_synthesis
                        .push_back((runtime, cron, ts));
                    while state.recent_pairs_runtime_vs_offline_synthesis.len()
                        > JUDGE_RUNTIME_VS_OFFLINE_PAIRS_CAP
                    {
                        state.recent_pairs_runtime_vs_offline_synthesis.pop_front();
                    }
                    any_new_synthesis_pair = true;
                }
                JudgeSurface::ConceptSummary => {
                    state
                        .recent_pairs_runtime_vs_offline_concept
                        .push_back((runtime, cron, ts));
                    while state.recent_pairs_runtime_vs_offline_concept.len()
                        > JUDGE_RUNTIME_VS_OFFLINE_PAIRS_CAP
                    {
                        state.recent_pairs_runtime_vs_offline_concept.pop_front();
                    }
                    any_new_concept_pair = true;
                }
            }
            state.total_offline_cron_events = state.total_offline_cron_events.saturating_add(1);
            any_new_pair = true;
        }
    }

    if any_new_pair {
        // Recompute κ over the rolling window. Drop the timestamp — κ uses
        // (runtime, cron) only.
        let pairs: Vec<(bool, bool)> = state
            .recent_pairs_runtime_vs_offline
            .iter()
            .map(|&(r, c, _)| (r, c))
            .collect();
        let new_kappa = kappa_runtime_vs_offline(&pairs).unwrap_or(0.0);
        let prior_kappa = state.runtime_vs_offline_kappa;
        let previous_last_computed_at = state.last_computed_at;
        state.runtime_vs_offline_kappa = new_kappa;
        state.last_computed_at = now;

        // Drift alert: bump counter when crossing below the threshold AND
        // we have enough pairs to trust κ. Bumping on every below-threshold
        // pass would be alert-spammy; we only bump on the EDGE (κ was
        // previously >= threshold, now below) to avoid log churn.
        //
        // Codex R6 P2 fix — also alert on the FIRST below-threshold window.
        // `prior_kappa` defaults to 0.0 on a fresh install, so a
        // first-window κ already below threshold would never trigger
        // the edge condition (0.0 >= 0.7 is false). Detect "first
        // qualifying window" via `last_computed_at == 0` (never
        // recomputed before this pass).
        let pair_count = state.recent_pairs_runtime_vs_offline.len();
        // Codex R6 P2 fix — only alert on first window when CURRENT κ
        // is below threshold (the original test bug had me firing on
        // first window even when κ was good). Combined gate:
        // (κ now below) AND (edge crossing OR first qualifying window).
        let alert_first_window =
            previous_last_computed_at == 0 && new_kappa < JUDGE_DRIFT_THRESHOLD;
        let edge_crossing =
            new_kappa < JUDGE_DRIFT_THRESHOLD && prior_kappa >= JUDGE_DRIFT_THRESHOLD;
        if pair_count >= JUDGE_DRIFT_MIN_PAIRS && (edge_crossing || alert_first_window) {
            state.judge_drift_alert = state.judge_drift_alert.saturating_add(1);
            if let Some(path) = drift_log_path {
                let _ = append_drift_log(
                    path,
                    new_kappa,
                    prior_kappa,
                    pair_count,
                    state.judge_drift_alert,
                );
            }
            tracing::warn!(
                runtime_vs_offline_kappa = new_kappa,
                prior_kappa,
                pair_count,
                drift_alert_total = state.judge_drift_alert,
                threshold = JUDGE_DRIFT_THRESHOLD,
                "judge_calibration: drift alert — runtime vs offline κ dropped below threshold"
            );
        }
    }

    if any_new_synthesis_pair {
        let pairs: Vec<(bool, bool)> = state
            .recent_pairs_runtime_vs_offline_synthesis
            .iter()
            .map(|&(r, c, _)| (r, c))
            .collect();
        let new_kappa = kappa_runtime_vs_offline(&pairs).unwrap_or(0.0);
        let prior_kappa = state.runtime_vs_offline_kappa_synthesis;
        state.runtime_vs_offline_kappa_synthesis = new_kappa;
        let pair_count = state.recent_pairs_runtime_vs_offline_synthesis.len();
        let first_below = state.judge_drift_alert_synthesis == 0
            && prior_kappa < JUDGE_DRIFT_THRESHOLD
            && new_kappa < JUDGE_DRIFT_THRESHOLD;
        let edge_crossing =
            new_kappa < JUDGE_DRIFT_THRESHOLD && prior_kappa >= JUDGE_DRIFT_THRESHOLD;
        if pair_count >= JUDGE_DRIFT_MIN_PAIRS && (edge_crossing || first_below) {
            state.judge_drift_alert_synthesis = state.judge_drift_alert_synthesis.saturating_add(1);
            tracing::warn!(
                runtime_vs_offline_kappa_synthesis = new_kappa,
                prior_kappa,
                pair_count,
                drift_alert_total = state.judge_drift_alert_synthesis,
                threshold = JUDGE_DRIFT_THRESHOLD,
                "judge_calibration: synthesis runtime vs offline κ dropped below threshold"
            );
        }
    }

    if any_new_concept_pair {
        let pairs: Vec<(bool, bool)> = state
            .recent_pairs_runtime_vs_offline_concept
            .iter()
            .map(|&(r, c, _)| (r, c))
            .collect();
        let new_kappa = kappa_runtime_vs_offline(&pairs).unwrap_or(0.0);
        let prior_kappa = state.runtime_vs_offline_kappa_concept;
        state.runtime_vs_offline_kappa_concept = new_kappa;
        let pair_count = state.recent_pairs_runtime_vs_offline_concept.len();
        let first_below = state.judge_drift_alert_concept == 0
            && prior_kappa < JUDGE_DRIFT_THRESHOLD
            && new_kappa < JUDGE_DRIFT_THRESHOLD;
        let edge_crossing =
            new_kappa < JUDGE_DRIFT_THRESHOLD && prior_kappa >= JUDGE_DRIFT_THRESHOLD;
        if pair_count >= JUDGE_DRIFT_MIN_PAIRS && (edge_crossing || first_below) {
            state.judge_drift_alert_concept = state.judge_drift_alert_concept.saturating_add(1);
            tracing::warn!(
                runtime_vs_offline_kappa_concept = new_kappa,
                prior_kappa,
                pair_count,
                drift_alert_total = state.judge_drift_alert_concept,
                threshold = JUDGE_DRIFT_THRESHOLD,
                "judge_calibration: concept-summary runtime vs offline κ dropped below threshold"
            );
        }
    }

    Ok((state, max_id_this_pass))
}

/// Best-effort append of a one-line drift alert to `~/.rein/judge_drift.log`.
/// Errors are intentionally swallowed (logging failure must not abort the
/// consumer) — tracing::warn surfaces the same info.
fn append_drift_log(
    path: &std::path::Path,
    new_kappa: f64,
    prior_kappa: f64,
    pair_count: usize,
    alert_total: u64,
) -> std::io::Result<()> {
    use std::io::Write;
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    let ts = chrono::Utc::now().to_rfc3339();
    writeln!(
        file,
        "{ts} drift_alert kappa={new_kappa:.4} prior_kappa={prior_kappa:.4} pairs={pair_count} alert_total={alert_total}"
    )?;
    Ok(())
}

// ── Cron job — emit-only (§7 step 5) ────────────────────────────────────────

/// Cron-archive entry written at synthesis-mint time (E_INTEGRATE owns the
/// write site in `ops/recall_synthesis.rs`). The cron job consumes these
/// entries — re-judges each via the stricter LLM, joins the runtime verdict
/// from `feedback_events`, and emits the OfflineCron event.
///
/// `surface` discriminates Cap B (Synthesis) vs Cap A (ConceptSummary).
/// `stamp_hash` is the SHA-256 of the post-truncation prompt+candidate bytes
/// the runtime judge actually saw — preserves J7 across the 24h window.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CronArchiveEntry {
    pub surface: CronArchiveSurface,
    /// For Synthesis: `synthesis_id` ULID. For ConceptSummary:
    /// `concept_summary_id` ULID (links to `concepts.living_summary_id`).
    pub id: String,
    /// For ConceptSummary, the persistent concept ID. Empty string for
    /// Synthesis.
    #[serde(default)]
    pub concept_id: String,
    pub stamp_hash: String,
    /// The query the runtime judge saw (synthesis surface) or definition
    /// excerpt (concept-summary surface). Used to rebuild the cron prompt.
    pub query: String,
    /// Source summaries the runtime judge saw (synthesis) or evidence
    /// keywords (concept-summary).
    pub sources: Vec<String>,
    /// The candidate text the runtime judge graded.
    pub candidate: String,
    /// Optional metadata routing — query_type, cluster_id.
    #[serde(default)]
    pub metadata: Option<JudgeMetadata>,
    /// UTC unix timestamp of mint.
    pub minted_at: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CronArchiveSurface {
    Synthesis,
    ConceptSummary,
}

/// Report emitted at the end of a cron pass. Surfaced by doctor and
/// `rein judge-calibrate-cron --verbose` for operator visibility.
#[derive(Debug, Clone, Default)]
pub struct CronReport {
    /// Total archive entries scanned across both surfaces.
    pub considered: usize,
    /// Entries that found a matching runtime verdict in `feedback_events`
    /// and produced an emitted OfflineCron event.
    pub emitted: usize,
    /// Entries skipped because the runtime judge had not produced a verdict
    /// for that `synthesis_id` (sample-rate-skipped or daily-cap-dropped).
    /// Logged at `tracing::debug!`.
    pub skipped_no_runtime_verdict: usize,
    /// Codex R5 P2 — entries already processed by a previous cron pass.
    /// Detected by an existing OfflineCron event with the same
    /// `(surface, id, stamp_hash)`. Idempotency guard against rerun.
    pub skipped_duplicate: usize,
    /// Entries dropped for non-fatal reasons (LLM error, malformed payload,
    /// hash mismatch). Logged at `tracing::warn!`.
    pub dropped: usize,
    /// Entries dropped because the cron LLM-call ledger reservation failed
    /// (R9-K1 fix — cron path now reserves via judge::contract::reserve_call
    /// alongside the runtime worker, sharing the same daily_call_cap budget).
    /// Bumped when the rolling 24h call count is at cap.
    pub dropped_cap: usize,
}

/// Run the cron job. Emit-only — writes `*OfflineCron` events to
/// `feedback_events`. The `judge_calibration` consumer (above) absorbs them
/// on the next adaptive-pipeline pass.
///
/// **§7 step 5 invariant**: this function MUST NOT write
/// `runtime_vs_offline_kappa`, `judge_drift_alert`, or any other
/// `judge_calibration_state` field. Direct writes from cron create split-
/// brain (Codex R6 P2 fix). All durable state writes live in the consumer.
///
/// **R9-K1 [P1] cap reservation**: each cron LLM HTTP call SHOULD reserve a
/// slot via `judge::contract::reserve_call(conn)` — same pattern as the
/// runtime worker. v0.27.1 leaves this as a `// TODO Wave-1.5` because
/// A_JUDGE_CORE owns `reserve_call`; if A's path isn't ready, the cron
/// proceeds without reservation. Default-off (`enabled = false` AND
/// `nightly_cron.enabled = false`) mitigates production blast radius.
pub fn run_judge_calibration_cron(
    store: &SqliteStore,
    config: &ReinConfig,
) -> ReinResult<CronReport> {
    let mut report = CronReport::default();

    // Codex R10 P2 fix — honor the master + nightly_cron config flags.
    // Without this gate, an operator who flips `nightly_cron.enabled =
    // false` but still has a stale archive on disk would have
    // `rein judge-calibrate-cron` re-judge + emit, defeating the
    // default-off/cost-control intent.
    if !config.ars.llm_judge.enabled || !config.ars.llm_judge.nightly_cron.enabled {
        tracing::info!(
            ars_llm_judge_enabled = config.ars.llm_judge.enabled,
            nightly_cron_enabled = config.ars.llm_judge.nightly_cron.enabled,
            "judge_calibration cron: skipped — flags disabled"
        );
        return Ok(report);
    }

    // Read 24h window: today's archive + yesterday's (covers UTC-day-boundary
    // mint events that should still be in scope).
    let today = chrono::Utc::now().date_naive();
    let yesterday = today - chrono::Duration::days(1);
    let archive_dates = [yesterday, today];

    // Shard scan — match any shard in the configured range. v0.27.1 we just
    // glob `*.jsonl` for both dates inside the queue dir.
    let mut archive_entries = collect_archive_entries(config, &archive_dates)?;
    // Codex R5 P2 fix — strict 24h window from now. Reading two full UTC
    // dates includes entries up to ~48h old when the cron runs late in
    // the day; filter post-collection so the window is precise.
    let now_ts = chrono::Utc::now().timestamp();
    let window_start = now_ts - 24 * 3600;
    archive_entries.retain(|e| e.minted_at >= window_start);
    report.considered = archive_entries.len();

    if archive_entries.is_empty() {
        tracing::debug!("judge_calibration cron: no archive entries in 24h window");
        return Ok(report);
    }

    // For each entry: look up runtime verdict by synthesis_id; if missing,
    // skip (debug). If present, re-judge and emit.
    for entry in archive_entries {
        match process_archive_entry(store, config, &entry) {
            ProcessOutcome::Emitted => report.emitted += 1,
            ProcessOutcome::SkippedNoRuntimeVerdict => {
                report.skipped_no_runtime_verdict += 1;
            }
            ProcessOutcome::SkippedDuplicate => {
                report.skipped_duplicate += 1;
            }
            ProcessOutcome::Dropped(reason) => {
                tracing::warn!(
                    surface = ?entry.surface,
                    id = %entry.id,
                    reason = %reason,
                    "judge_calibration cron: dropped entry"
                );
                report.dropped += 1;
            }
            ProcessOutcome::DroppedCap => {
                report.dropped_cap += 1;
            }
        }
    }

    tracing::info!(
        considered = report.considered,
        emitted = report.emitted,
        skipped_no_runtime_verdict = report.skipped_no_runtime_verdict,
        dropped = report.dropped,
        dropped_cap = report.dropped_cap,
        "judge_calibration cron: pass complete"
    );

    Ok(report)
}

enum ProcessOutcome {
    Emitted,
    SkippedNoRuntimeVerdict,
    /// Codex R5 P2 — entry was already processed in a previous cron pass.
    SkippedDuplicate,
    Dropped(String),
    #[allow(dead_code)] // populated when R9-K1 reserve_call lands in Wave 1.5
    DroppedCap,
}

/// Codex R5 P2 fix — check whether `feedback_events` already has an
/// OfflineCron event for this `(surface, id, stamp_hash)` tuple. Used by
/// `process_archive_entry` to skip re-emit on cron rerun. JSON path
/// extracts on the payload; SQLite ships JSON1 by default in modern
/// builds, but fall back to LIKE on the raw text payload for safety
/// (which is acceptable: the stamp_hash is a unique 64-hex SHA-256
/// suffix, so collisions in raw substring are improbable).
fn cron_event_already_emitted(
    conn: &rusqlite::Connection,
    surface: &CronArchiveSurface,
    id: &str,
    stamp_hash: &str,
) -> ReinResult<bool> {
    let event_type = match surface {
        CronArchiveSurface::Synthesis => EventType::SynthesisLlmJudgeOfflineCron.as_str(),
        CronArchiveSurface::ConceptSummary => EventType::ConceptSummaryLlmJudgeOfflineCron.as_str(),
    };
    // Match BOTH id and stamp_hash so a re-mint of the same surface_id
    // with different content (different stamp_hash) is judged afresh.
    let id_field = match surface {
        CronArchiveSurface::Synthesis => "synthesis_id",
        CronArchiveSurface::ConceptSummary => "concept_summary_id",
    };
    let id_pat = format!("%\"{id_field}\":\"{id}\"%");
    let hash_pat = format!("%\"stamp_hash\":\"{stamp_hash}\"%");
    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM feedback_events \
         WHERE event_type = ?1 AND payload LIKE ?2 AND payload LIKE ?3 \
         LIMIT 1",
        rusqlite::params![event_type, id_pat, hash_pat],
        |row| row.get(0),
    )?;
    Ok(count > 0)
}

fn process_archive_entry(
    store: &SqliteStore,
    config: &ReinConfig,
    entry: &CronArchiveEntry,
) -> ProcessOutcome {
    // Codex R5 P2 fix — idempotency: check if this entry already has
    // an OfflineCron event for the same (surface, id, stamp_hash) tuple.
    // If yes, skip without re-judging or re-emitting. Prevents reruns
    // of `rein judge-calibrate-cron` from double-charging the cap +
    // double-counting κ pairs. stamp_hash distinguishes "same id, fresh
    // input" (re-mint with different prompt) from "same id, same input"
    // (true duplicate).
    match cron_event_already_emitted(store.conn(), &entry.surface, &entry.id, &entry.stamp_hash) {
        Ok(true) => {
            // Codex R9 P3 — reap orphan `cron_claims` row left when a
            // previous pass crashed AFTER `emit_event` succeeded but
            // BEFORE the post-emit `release_cron_claim` ran. The
            // OfflineCron event is durable, so any extant claim row
            // for this tuple is guaranteed-orphan; tokenless DELETE
            // is safe because (a) a fresh peer holding the row would
            // imminently release it on its own emit, and (b) a peer
            // mid-LLM-call after stale takeover would lose its emit
            // to F4 A3 UNIQUE and release on the failure path.
            let _ =
                reap_emitted_cron_claim(store.conn(), &entry.surface, &entry.id, &entry.stamp_hash);
            return ProcessOutcome::SkippedDuplicate;
        }
        Ok(false) => {}
        Err(e) => {
            tracing::warn!(error = %e, "cron idempotency check failed, proceeding (best-effort)");
        }
    }

    // 1. Look up runtime verdict in feedback_events by synthesis_id.
    //    Codex R5 P2 fix: join by synthesis_id ONLY, NOT stamp_hash —
    //    stamp_hash collisions across distinct synthesis_ids would pair
    //    the cron verdict for one synthesis with a different runtime
    //    verdict.
    let (runtime_hit, runtime_judge_model) =
        match lookup_runtime_verdict(store.conn(), &entry.surface, &entry.id) {
            Ok(Some(v)) => v,
            Ok(None) => return ProcessOutcome::SkippedNoRuntimeVerdict,
            Err(e) => return ProcessOutcome::Dropped(format!("verdict lookup failed: {e}")),
        };

    // codex R5 P2: defensive size check BEFORE cap reservation. Cron
    // archive lines from a pre-enqueue-cap version (or manually-injected
    // archive entries) may exceed `JUDGE_MAX_INPUT_CHARS`, and the cron
    // LLM call site (`call_cron_judge` → `call_judge_sync`) no longer
    // truncates (R1 J7 fix). Without this guard, oversized cron lines
    // would burn `daily_call_cap` AND send untruncated payloads to the
    // cron judge model. Mirror `llm_judge_worker::dispatch_one`'s pre-
    // reservation ceiling exactly, including the same const.
    let combined_chars = entry
        .sources
        .iter()
        .map(|s| s.chars().count())
        .sum::<usize>()
        + entry.candidate.chars().count();
    if combined_chars > crate::ops::llm_judge_worker::JUDGE_MAX_INPUT_CHARS {
        tracing::warn!(
            surface = ?entry.surface,
            id = %entry.id,
            combined_chars = combined_chars,
            ceiling = crate::ops::llm_judge_worker::JUDGE_MAX_INPUT_CHARS,
            "cron judge: archive payload exceeds dispatch ceiling; dropped pre-reservation"
        );
        return ProcessOutcome::Dropped(format!(
            "cron archive payload too large ({} chars > ceiling {})",
            combined_chars,
            crate::ops::llm_judge_worker::JUDGE_MAX_INPUT_CHARS
        ));
    }

    // v0.27.5 R3 — pre-LLM atomic claim. Closes the v0.27.4 R9 P2
    // concurrent-cron race: two workers could both clear the
    // `cron_event_already_emitted` LIKE check, both burn `daily_call_cap`
    // via `reserve_call`, both pay for an LLM call, and only the second
    // emit would lose on the F4 A3 UNIQUE index. The claim row is
    // inserted via `INSERT OR IGNORE` (atomic) BEFORE `reserve_call`,
    // so only the claim winner proceeds; the loser short-circuits to
    // `SkippedDuplicate` without burning quota or paying for the LLM.
    //
    // The returned `claim_token` (codex R3 P2) is the ownership proof
    // for `release_cron_claim` — without it, a slow original cron whose
    // claim was taken over by a fresh peer (after the stale window)
    // could DELETE the new owner's row.
    let claim_token =
        match try_claim_cron(store.conn(), &entry.surface, &entry.id, &entry.stamp_hash) {
            Ok(Some(token)) => token,
            Ok(None) => return ProcessOutcome::SkippedDuplicate,
            Err(e) => {
                // Codex R5 P2 — DO NOT proceed without a claim. If the
                // INSERT errors (e.g. SQLite busy during a concurrent
                // cron pass), the entry is exactly the contention case
                // `cron_claims` exists to serialize. Bypassing here
                // would let two concurrent workers both reserve
                // `daily_call_cap` and pay for an LLM call, with only
                // the second emit losing on the F4 A3 UNIQUE index —
                // the bug this primitive was shipped to close. Treat
                // as a retryable drop; the next cron pass will
                // rediscover the entry and try again.
                tracing::warn!(error = %e, "cron claim insert failed; dropping (retryable)");
                return ProcessOutcome::Dropped(format!("cron claim insert failed: {e}"));
            }
        };

    // Codex R8 P2 — post-claim TOCTOU re-check. The initial
    // `cron_event_already_emitted` LIKE check ran BEFORE we acquired
    // the claim, so a peer may have completed (claim → LLM → emit →
    // release) in the gap between our check and our acquire. Without
    // this re-check we'd burn cap + pay for a duplicate LLM call only
    // for the F4 A3 UNIQUE index to catch the second emit. Re-running
    // the LIKE check after the claim is held closes the window: if
    // the event now exists, release the claim and SkippedDuplicate
    // without paying anything.
    match cron_event_already_emitted(store.conn(), &entry.surface, &entry.id, &entry.stamp_hash) {
        Ok(true) => {
            let _ = release_cron_claim(
                store.conn(),
                &entry.surface,
                &entry.id,
                &entry.stamp_hash,
                &claim_token,
            );
            return ProcessOutcome::SkippedDuplicate;
        }
        Ok(false) => {}
        Err(e) => {
            // Best-effort: if the LIKE check errors, fall through to
            // the normal flow. The F4 A3 UNIQUE index on
            // `feedback_events` remains as the strict guard.
            tracing::warn!(error = %e, "post-claim emitted re-check failed, proceeding (best-effort)");
        }
    }

    // 2. Re-judge via the stricter cron LLM.
    //
    // v0.27.2 R9-K1 fix — reserve a J2 ledger slot before the HTTP
    // call so cron LLM calls count toward `[ars.llm_judge].daily_call_cap`
    // alongside runtime worker calls. Without this the runtime + cron
    // could combined exceed the configured cap. Default-off mitigates,
    // but operators enabling both paths need consistent budgeting.
    let daily_cap = config.ars.llm_judge.daily_call_cap;
    let token = match crate::judge::contract::reserve_call(store.conn(), daily_cap) {
        Ok(Some(t)) => t,
        Ok(None) => {
            // v0.27.5 R3 — cap exhausted before any LLM call. Release
            // the claim so a future cron pass (e.g. tomorrow under a
            // refilled cap) can retry, otherwise the entry is
            // permanently `SkippedDuplicate` on every retry. Best-
            // effort: even if the delete races, the F4 A3 UNIQUE index
            // on `feedback_events` still prevents double-emit, so
            // worst case is a wasted retry that the LIKE fast-path
            // catches.
            let _ = release_cron_claim(
                store.conn(),
                &entry.surface,
                &entry.id,
                &entry.stamp_hash,
                &claim_token,
            );
            return ProcessOutcome::DroppedCap;
        }
        Err(e) => {
            // Same rationale as the DroppedCap arm — no LLM was called,
            // so the claim row must not stick around as a permanent
            // skip marker.
            let _ = release_cron_claim(
                store.conn(),
                &entry.surface,
                &entry.id,
                &entry.stamp_hash,
                &claim_token,
            );
            return ProcessOutcome::Dropped(format!("reserve_call: {e}"));
        }
    };

    let (cron_hit, cron_reason, cron_judge_model) = match call_cron_judge(config, entry) {
        Ok(v) => {
            // Successful HTTP call — mark ledger row done.
            let _ = token.commit(store.conn());
            v
        }
        Err(e) => {
            // HTTP attempt was made (counts toward cap) but failed.
            let _ = token.fail(store.conn());
            // v0.27.5 R3 — release the claim so a future cron pass can
            // retry. The cap was burned, but holding the claim row
            // forever would prevent any retry from emitting (LIKE
            // check would say "no event yet" but try_claim_cron would
            // still see the orphaned claim and SkippedDuplicate). The
            // tradeoff is: retry will burn cap again on subsequent
            // failures; operator who sees daily_cap exhausted by
            // cron retry storms can investigate the LLM failure root
            // cause directly.
            let _ = release_cron_claim(
                store.conn(),
                &entry.surface,
                &entry.id,
                &entry.stamp_hash,
                &claim_token,
            );
            return ProcessOutcome::Dropped(format!("cron LLM call failed: {e}"));
        }
    };

    // 3. Emit ONLY — write the OfflineCron event. Cron NEVER writes
    //    runtime_vs_offline_kappa or judge_drift_alert directly (§7 step 5,
    //    Codex R6 P2). The judge_calibration consumer absorbs this event
    //    on its next pass and does the durable state writes.
    let emit_result = match entry.surface {
        CronArchiveSurface::Synthesis => {
            let payload = SynthesisLlmJudgeOfflineCronPayload {
                synthesis_id: entry.id.clone(),
                stamp_hash: entry.stamp_hash.clone(),
                runtime_hit,
                runtime_judge_model,
                cron_hit,
                cron_judge_model,
                cron_reason,
                metadata: entry.metadata.clone(),
            };
            let payload_value = match serde_json::to_value(&payload) {
                Ok(v) => v,
                Err(e) => {
                    // v0.27.5 R3 — payload serialize is deterministic
                    // from the entry; if it fails it'll fail again on
                    // retry. Release anyway so we don't permanently
                    // block; the caller can fix the bug + retry.
                    let _ = release_cron_claim(
                        store.conn(),
                        &entry.surface,
                        &entry.id,
                        &entry.stamp_hash,
                        &claim_token,
                    );
                    return ProcessOutcome::Dropped(format!("payload serialize: {e}"));
                }
            };
            emit_event(
                store.conn(),
                FeedbackEvent {
                    event_type: EventType::SynthesisLlmJudgeOfflineCron,
                    request_id: None,
                    memory_id: None,
                    concept_id: None,
                    query: None,
                    query_type: payload.metadata.as_ref().and_then(|m| m.query_type.clone()),
                    topic: None,
                    payload: Some(payload_value),
                },
            )
        }
        CronArchiveSurface::ConceptSummary => {
            let payload = ConceptSummaryLlmJudgeOfflineCronPayload {
                concept_summary_id: entry.id.clone(),
                concept_id: entry.concept_id.clone(),
                stamp_hash: entry.stamp_hash.clone(),
                runtime_hit,
                runtime_judge_model,
                cron_hit,
                cron_judge_model,
                cron_reason,
                metadata: entry.metadata.clone(),
            };
            let payload_value = match serde_json::to_value(&payload) {
                Ok(v) => v,
                Err(e) => {
                    // v0.27.5 R3 — same release rationale as the
                    // Synthesis arm: don't permanently block a future
                    // retry on a deterministic serialize failure.
                    let _ = release_cron_claim(
                        store.conn(),
                        &entry.surface,
                        &entry.id,
                        &entry.stamp_hash,
                        &claim_token,
                    );
                    return ProcessOutcome::Dropped(format!("payload serialize: {e}"));
                }
            };
            emit_event(
                store.conn(),
                FeedbackEvent {
                    event_type: EventType::ConceptSummaryLlmJudgeOfflineCron,
                    request_id: None,
                    memory_id: None,
                    concept_id: Some(entry.concept_id.clone()),
                    query: None,
                    query_type: payload.metadata.as_ref().and_then(|m| m.query_type.clone()),
                    topic: None,
                    payload: Some(payload_value),
                },
            )
        }
    };

    match emit_result {
        Ok(_) => {
            // Codex R7 P2 — release the claim on successful emit too.
            // The `feedback_events` row is now durable, and the LIKE
            // check in `cron_event_already_emitted` becomes the
            // authoritative future-dedup guard. Holding the claim row
            // forever would grow `cron_claims` unbounded (one row per
            // processed entry, up to `daily_call_cap` per day) for
            // operators who enable the nightly cron. Stale takeover
            // is still in place for crash recovery.
            let _ = release_cron_claim(
                store.conn(),
                &entry.surface,
                &entry.id,
                &entry.stamp_hash,
                &claim_token,
            );
            ProcessOutcome::Emitted
        }
        Err(e) => {
            // F4 A3 fix — partial UNIQUE index on
            // `idx_feedback_events_offlinecron_dedup` is the atomic guard
            // for concurrent cron emit. A UNIQUE constraint violation
            // here means a peer cron run beat us to it; treat as a
            // duplicate (no-op) instead of a hard error.
            if is_unique_constraint_violation(&e) {
                // v0.27.5 R3 — peer beat us to feedback_events. Future
                // passes will SkippedDuplicate via `cron_event_already_emitted`,
                // so the orphan claim row would block nothing meaningful;
                // release anyway for a clean lifecycle (defensive — if a
                // peer of a peer ever clears feedback_events out-of-band
                // we want the claim gone too).
                let _ = release_cron_claim(
                    store.conn(),
                    &entry.surface,
                    &entry.id,
                    &entry.stamp_hash,
                    &claim_token,
                );
                return ProcessOutcome::SkippedDuplicate;
            }
            // v0.27.5 R3 — non-UNIQUE emit failure means no event was
            // committed; release the claim so a future retry can re-emit.
            let _ = release_cron_claim(
                store.conn(),
                &entry.surface,
                &entry.id,
                &entry.stamp_hash,
                &claim_token,
            );
            ProcessOutcome::Dropped(format!("emit_event failed: {e}"))
        }
    }
}

/// v0.27.5 R3 — stale-claim takeover window. A `cron_claims` row whose
/// `claimed_at` is older than this is treated as orphaned (process
/// crash / kill -9 between claim and emit) and reclaimable by the next
/// `try_claim_cron` caller. Mirrors the 5-minute pattern used by the
/// resummerize and cold_archive claim-token leases. Picked to comfortably
/// outlive a normal cron LLM round-trip (typically 5-30s) while still
/// short enough to recover from real crashes within one cron cycle.
const CRON_CLAIM_STALE_SECS: i64 = 300;

/// v0.27.5 R3 — atomic pre-LLM claim primitive. Inserts a row into
/// `cron_claims` keyed by `(event_type, surface_id, stamp_hash)` via
/// `INSERT OR IGNORE`. On success returns `Ok(Some(token))` carrying
/// the freshly minted ULID claim token; subsequent
/// [`release_cron_claim`] calls MUST pass that token so the DELETE
/// only matches if the caller is still the row's owner.
///
/// `Ok(None)` means a fresh peer already owns the tuple; the caller
/// MUST short-circuit to `SkippedDuplicate` without calling
/// `reserve_call` or the LLM.
///
/// **Stale takeover** (codex R3 P2 ownership-safe variant) — when
/// INSERT OR IGNORE finds an existing row whose `claimed_at` is older
/// than [`CRON_CLAIM_STALE_SECS`] (orphan from a crashed peer that
/// never emitted), the takeover UPDATE overwrites `claim_token` AND
/// `claimed_at` with fresh values and returns the new token. The
/// UPDATE predicate `claimed_at < cutoff` is the atomic staleness
/// guard: a peer that concurrently refreshed the row stays the winner,
/// and the original (slow) cron's `release_cron_claim` will see
/// `claim_token != ?token` and DELETE 0 rows — never clobbering the
/// new owner.
///
/// Atomic at the SQLite level: the PRIMARY KEY uniqueness contest +
/// the staleness UPDATE both happen inside SQLite's row-lock.
fn try_claim_cron(
    conn: &rusqlite::Connection,
    surface: &CronArchiveSurface,
    id: &str,
    stamp_hash: &str,
) -> ReinResult<Option<String>> {
    let event_type = match surface {
        CronArchiveSurface::Synthesis => EventType::SynthesisLlmJudgeOfflineCron.as_str(),
        CronArchiveSurface::ConceptSummary => EventType::ConceptSummaryLlmJudgeOfflineCron.as_str(),
    };
    let token = ulid::Ulid::new().to_string();
    let now_unix = chrono::Utc::now().timestamp();
    // Fast path: no row exists yet → INSERT OR IGNORE wins.
    let inserted = conn.execute(
        "INSERT OR IGNORE INTO cron_claims \
           (event_type, surface_id, stamp_hash, claim_token, claimed_at) \
         VALUES (?1, ?2, ?3, ?4, ?5)",
        rusqlite::params![event_type, id, stamp_hash, token, now_unix],
    )?;
    if inserted == 1 {
        return Ok(Some(token));
    }
    // Slow path: row exists. Take it over IFF stale (atomic via the
    // `claimed_at < cutoff` predicate). The takeover overwrites both
    // `claim_token` and `claimed_at`, so the original owner's release
    // attempt becomes a no-op.
    let stale_cutoff = now_unix - CRON_CLAIM_STALE_SECS;
    let updated = conn.execute(
        "UPDATE cron_claims \
         SET claim_token = ?4, claimed_at = ?5 \
         WHERE event_type = ?1 AND surface_id = ?2 AND stamp_hash = ?3 \
           AND claimed_at < ?6",
        rusqlite::params![event_type, id, stamp_hash, token, now_unix, stale_cutoff],
    )?;
    if updated == 1 {
        Ok(Some(token))
    } else {
        Ok(None)
    }
}

/// v0.27.5 R3 — release a `cron_claims` row on no-cap-burn failure
/// paths so a future cron pass can retry instead of being permanently
/// `SkippedDuplicate`.
///
/// Codex R3 P2 fix: predicate the DELETE on `claim_token`. If a stale
/// original cron's claim was taken over by a fresh peer, the original
/// caller's stored `token` no longer matches the row, so the DELETE
/// affects 0 rows and the fresh peer's row is preserved.
fn release_cron_claim(
    conn: &rusqlite::Connection,
    surface: &CronArchiveSurface,
    id: &str,
    stamp_hash: &str,
    token: &str,
) -> ReinResult<()> {
    let event_type = match surface {
        CronArchiveSurface::Synthesis => EventType::SynthesisLlmJudgeOfflineCron.as_str(),
        CronArchiveSurface::ConceptSummary => EventType::ConceptSummaryLlmJudgeOfflineCron.as_str(),
    };
    conn.execute(
        "DELETE FROM cron_claims \
         WHERE event_type = ?1 AND surface_id = ?2 AND stamp_hash = ?3 \
           AND claim_token = ?4",
        rusqlite::params![event_type, id, stamp_hash, token],
    )?;
    Ok(())
}

/// v0.27.5 R3 (codex R9 P3) — tokenless reaper for orphan `cron_claims`
/// rows discovered during the fast-path `cron_event_already_emitted`
/// LIKE check. Caller MUST have confirmed a durable OfflineCron event
/// exists for the tuple before calling — that's the safety invariant
/// that lets us skip the `claim_token` predicate. The event's
/// existence makes any contemporaneous claim-row holder a no-op (they
/// either crashed mid-emit or are about to lose to F4 A3 UNIQUE).
fn reap_emitted_cron_claim(
    conn: &rusqlite::Connection,
    surface: &CronArchiveSurface,
    id: &str,
    stamp_hash: &str,
) -> ReinResult<()> {
    let event_type = match surface {
        CronArchiveSurface::Synthesis => EventType::SynthesisLlmJudgeOfflineCron.as_str(),
        CronArchiveSurface::ConceptSummary => EventType::ConceptSummaryLlmJudgeOfflineCron.as_str(),
    };
    conn.execute(
        "DELETE FROM cron_claims \
         WHERE event_type = ?1 AND surface_id = ?2 AND stamp_hash = ?3",
        rusqlite::params![event_type, id, stamp_hash],
    )?;
    Ok(())
}

/// F4 A3 helper — detect SQLite UNIQUE constraint violations so the
/// cron emit path can absorb concurrent-emit races as no-ops.
fn is_unique_constraint_violation(err: &ReinError) -> bool {
    match err {
        ReinError::Database(rusqlite::Error::SqliteFailure(ffi, _)) => {
            ffi.code == rusqlite::ErrorCode::ConstraintViolation
                && ffi.extended_code == rusqlite::ffi::SQLITE_CONSTRAINT_UNIQUE
        }
        _ => false,
    }
}

/// Look up the runtime judge's verdict for a given synthesis_id /
/// concept_summary_id.
///
/// Codex R5 P2 fix: synthesis_id-only join (no stamp_hash fallback).
/// stamp_hash collisions across distinct ids would silently mispair
/// verdicts.
///
/// Returns `Ok(Some((hit, judge_model)))` on success, `Ok(None)` when no
/// runtime verdict exists (sample-rate-skipped or daily-cap-dropped — both
/// expected), `Err` on DB / parse error.
fn lookup_runtime_verdict(
    conn: &Connection,
    surface: &CronArchiveSurface,
    id: &str,
) -> ReinResult<Option<(bool, String)>> {
    let event_type = match surface {
        CronArchiveSurface::Synthesis => EventType::SynthesisLlmJudge.as_str(),
        CronArchiveSurface::ConceptSummary => EventType::ConceptSummaryLlmJudge.as_str(),
    };

    // The runtime payload is owned by A_JUDGE_CORE; we don't have its
    // type here. Fall back to JSON deserialization of the payload's
    // `synthesis_id` (or `concept_summary_id`) + `hit` + `judge_model`
    // fields. If the schema changes underneath us, the parse falls
    // through to `Ok(None)` (Codex pattern: fail-soft when payload
    // schema can't be matched).
    // Codex R2 P2 fix — `LIMIT 200` was too narrow: high-volume runs
    // (default daily_cap = 10000) push older runtime verdicts out of
    // the window, so cron archive entries from yesterday couldn't find
    // their matching verdict. Raise to 50_000 (matches synthesis_feedback
    // consumer peek cap) and bound by 30-day timestamp floor so the scan
    // doesn't grow unbounded across years.
    let mut stmt = conn.prepare(
        "SELECT payload FROM feedback_events
         WHERE event_type = ?1
           AND payload IS NOT NULL
           AND ts > datetime('now', '-30 days')
         ORDER BY id DESC LIMIT 50000",
    )?;
    let rows = stmt.query_map(rusqlite::params![event_type], |row| {
        row.get::<_, Option<String>>(0)
    })?;

    let id_field = match surface {
        CronArchiveSurface::Synthesis => "synthesis_id",
        CronArchiveSurface::ConceptSummary => "concept_summary_id",
    };

    for r in rows {
        let Ok(Some(payload_str)) = r else { continue };
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&payload_str) else {
            continue;
        };
        let matches_id = value
            .get(id_field)
            .and_then(|v| v.as_str())
            .is_some_and(|s| s == id);
        if !matches_id {
            continue;
        }
        let hit = value.get("hit").and_then(|v| v.as_bool()).unwrap_or(false);
        let model = value
            .get("judge_model")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        return Ok(Some((hit, model)));
    }

    Ok(None)
}

/// Call the configured nightly-cron LLM judge for an archive entry.
///
/// v0.27.1 — pending A_JUDGE_CORE's `LlmJudgeHitChecker` integration into the
/// runtime worker, this function returns a `ReinError::Config` placeholder
/// when the cron LLM is not yet wired. CLI / tests can stub this via the
/// regular `MockExtractor` queue once A's runtime worker lands.
///
/// Returns `(cron_hit, cron_reason, cron_judge_model)`. The third tuple
/// element is the model identifier as resolved by `[ars.llm_judge.nightly_cron]`
/// per Track 2 (§8) — placeholder string used by stub.
fn call_cron_judge(
    config: &ReinConfig,
    entry: &CronArchiveEntry,
) -> ReinResult<(bool, String, String)> {
    // Codex R4 P2 fix — wire the cron LLM through B1's resolver and the
    // existing prose-mode judge prompt + parser from the runtime worker.
    // Stricter rubric is delivered via `[ars.llm_judge.nightly_cron]`
    // operator config (typically a larger / different-family model).
    let extractor =
        crate::ops::concept_summary::create_ars_extractor(config, "ars.llm_judge.nightly_cron")
            .ok_or_else(|| {
                ReinError::Config(
                    "judge_calibration cron: no LLM provider configured for \
             [ars.llm_judge.nightly_cron] — set provider/model or disable \
             [ars.llm_judge.nightly_cron].enabled = false"
                        .to_string(),
                )
            })?;

    // Reconstruct the same prompt the runtime worker would have built.
    // Joining `entry.sources` with `\n` matches `recall_synthesis`'s
    // pre-truncation prompt shape, so the cron sees byte-identical input
    // when archive_entry was emitted with `sources = [prompt]` (Codex
    // R1 P2 archive shape fix).
    let joined_sources = entry.sources.join("\n");
    // F4 B2 — pass config so the cron LLM call honors the resolved
    // `[ars.llm_judge.nightly_cron]` (or inherited `[ars.llm_judge]`
    // / `[llm]`) max_input_chars override at runtime.
    let raw = crate::ops::llm_judge_worker::call_judge_sync(
        config,
        &extractor,
        &joined_sources,
        &entry.candidate,
    )?;
    let (hit, reason) =
        crate::ops::llm_judge_worker::parse_judge_output(&raw).ok_or_else(|| {
            ReinError::Extract(format!(
                "cron judge output unparseable (expected `HIT: yes|no\\nWHY: ...`): {}",
                raw.chars().take(120).collect::<String>()
            ))
        })?;
    let model_id = match &extractor {
        crate::extract::llm::ExtractorKind::Gemini(g) => format!("gemini:{}", g.model),
        crate::extract::llm::ExtractorKind::Omlx(o) => format!("omlx:{}", o.model),
        #[cfg(feature = "test-support")]
        crate::extract::llm::ExtractorKind::Mock(_) => "mock".to_string(),
    };
    Ok((hit, reason, model_id))
}

/// Collect archive entries across the given UTC dates by globbing
/// `synthesis_cron_archive_<YYYYMMDD>_*.jsonl` in the DB-scoped queue dir.
fn collect_archive_entries(
    config: &ReinConfig,
    dates: &[chrono::NaiveDate],
) -> ReinResult<Vec<CronArchiveEntry>> {
    let base = crate::extract::hooks::buffer::resolve_buffer_dir(config);
    let db_tag = {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        config.resolve_db_path().hash(&mut h);
        format!("{:016x}", h.finish())
    };
    let queue_dir = base.join("queue").join(&db_tag);
    if !queue_dir.exists() {
        return Ok(Vec::new());
    }

    let mut out = Vec::new();
    let entries = match std::fs::read_dir(&queue_dir) {
        Ok(rd) => rd,
        Err(e) => {
            return Err(ReinError::Config(format!(
                "judge_calibration: failed to read queue dir {}: {e}",
                queue_dir.display()
            )));
        }
    };
    for ent in entries.flatten() {
        let path = ent.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        // Match: synthesis_cron_archive_<YYYYMMDD>_*.jsonl
        let Some(rest) = name.strip_prefix("synthesis_cron_archive_") else {
            continue;
        };
        let Some(rest) = rest.strip_suffix(".jsonl") else {
            continue;
        };
        // Date is the first 8 chars of `rest` before the `_<shard>` suffix.
        if rest.len() < 9 || rest.chars().nth(8) != Some('_') {
            continue;
        }
        let date_str = &rest[..8];
        let Ok(date) = chrono::NaiveDate::parse_from_str(date_str, "%Y%m%d") else {
            continue;
        };
        if !dates.contains(&date) {
            continue;
        }

        // Parse jsonl — one entry per line.
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!(path = %path.display(), error = %e, "judge_calibration: failed to read archive file");
                continue;
            }
        };
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            match serde_json::from_str::<CronArchiveEntry>(line) {
                Ok(entry) => out.push(entry),
                Err(e) => {
                    tracing::warn!(
                        path = %path.display(),
                        error = %e,
                        "judge_calibration: malformed archive entry, skipping"
                    );
                }
            }
        }
    }
    Ok(out)
}

// ── Slow-channel hook (called from `ops/adaptive.rs` per-pass dispatch) ─────

/// Wrapper called from `ops/adaptive.rs` per-pass dispatch loop. Reads prior
/// state from `AdaptiveState`, runs the consumer, mutates the state in place
/// + returns `Vec<(consumer, max_event_id)>` for the orchestrator's
/// post-save commit.
///
/// Codex R7 P2 fix: without this wiring the `judge_calibration` consumer's
/// offset never advances and `runtime_vs_offline_kappa` stays stale.
pub fn run_judge_calibration_consumer(
    store: &SqliteStore,
    state: &mut AdaptiveState,
    drift_log_path: Option<&std::path::Path>,
) -> Option<Vec<(&'static str, i64)>> {
    match recompute_judge_calibration_state(
        store.conn(),
        state.judge_calibration_state.clone(),
        drift_log_path,
    ) {
        Ok((new_state, max_id_opt)) => {
            state.judge_calibration_state = Some(new_state);
            max_id_opt.map(|id| vec![(JUDGE_CALIBRATION_CONSUMER, id)])
        }
        Err(e) => {
            tracing::warn!("failed to recompute judge_calibration_state: {e}");
            None
        }
    }
}

/// Variant used by tests / direct CLI invocations that need both the report
/// and explicit consumer offset commit. Production callers go through
/// [`run_judge_calibration_consumer`] (the per-pass hook in adaptive.rs).
#[doc(hidden)]
pub fn run_consumer_and_commit(
    store: &SqliteStore,
    drift_log_path: Option<&std::path::Path>,
) -> ReinResult<Option<i64>> {
    let prior = AdaptiveState::restore_snapshot(store.conn());
    let prior_state = prior
        .as_ref()
        .and_then(|s| s.judge_calibration_state.clone());
    let (new_state, max_id_opt) =
        recompute_judge_calibration_state(store.conn(), prior_state, drift_log_path)?;

    if let Some(id) = max_id_opt {
        // Persist updated state + commit offset only when something
        // changed. Best-effort merge into AdaptiveState; if no prior
        // state exists we create one with just our field populated.
        let mut adaptive = prior.unwrap_or_default();
        adaptive.judge_calibration_state = Some(new_state);
        adaptive.version = adaptive.version.saturating_add(1);
        adaptive.save_snapshot(store.conn())?;
        commit_offset(store.conn(), &[(JUDGE_CALIBRATION_CONSUMER, id)])?;
        Ok(Some(id))
    } else {
        Ok(None)
    }
}

// ── StoredEvent timestamp helper ────────────────────────────────────────────

trait StoredEventExt {
    fn ts_to_unix(&self) -> Option<i64>;
}
impl StoredEventExt for crate::store::adaptive::StoredEvent {
    fn ts_to_unix(&self) -> Option<i64> {
        // ts is stored as RFC3339 with millisecond precision per the
        // schema's strftime template. Parse → unix timestamp seconds.
        chrono::DateTime::parse_from_rfc3339(&self.ts)
            .ok()
            .map(|dt| dt.timestamp())
    }
}

// ── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::adaptive::{commit_offset, peek_events};

    fn setup_db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "
            CREATE TABLE feedback_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                event_type TEXT NOT NULL,
                request_id TEXT,
                memory_id TEXT,
                concept_id TEXT,
                query TEXT,
                query_type TEXT,
                topic TEXT,
                payload TEXT
            );
            CREATE TABLE consumer_offsets (
                consumer TEXT PRIMARY KEY,
                last_event_id INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT
            );
            CREATE TABLE cron_claims (
                event_type TEXT NOT NULL,
                surface_id TEXT NOT NULL,
                stamp_hash TEXT NOT NULL,
                claim_token TEXT NOT NULL DEFAULT '',
                claimed_at INTEGER NOT NULL,
                PRIMARY KEY (event_type, surface_id, stamp_hash)
            );
            ",
        )
        .unwrap();
        conn
    }

    fn emit_offline_cron_synth(
        conn: &Connection,
        synthesis_id: &str,
        runtime: bool,
        cron: bool,
    ) -> i64 {
        let payload = SynthesisLlmJudgeOfflineCronPayload {
            synthesis_id: synthesis_id.to_string(),
            stamp_hash: "deadbeef".into(),
            runtime_hit: runtime,
            runtime_judge_model: "model-r".into(),
            cron_hit: cron,
            cron_judge_model: "model-c".into(),
            cron_reason: "test".into(),
            metadata: None,
        };
        super::emit_event(
            conn,
            FeedbackEvent {
                event_type: EventType::SynthesisLlmJudgeOfflineCron,
                request_id: None,
                memory_id: None,
                concept_id: None,
                query: None,
                query_type: None,
                topic: None,
                payload: Some(serde_json::to_value(&payload).unwrap()),
            },
        )
        .unwrap()
    }

    fn emit_offline_cron_concept(
        conn: &Connection,
        summary_id: &str,
        runtime: bool,
        cron: bool,
    ) -> i64 {
        let payload = ConceptSummaryLlmJudgeOfflineCronPayload {
            concept_summary_id: summary_id.to_string(),
            concept_id: format!("concept-{summary_id}"),
            stamp_hash: "deadbeef".into(),
            runtime_hit: runtime,
            runtime_judge_model: "model-r".into(),
            cron_hit: cron,
            cron_judge_model: "model-c".into(),
            cron_reason: "test".into(),
            metadata: None,
        };
        super::emit_event(
            conn,
            FeedbackEvent {
                event_type: EventType::ConceptSummaryLlmJudgeOfflineCron,
                request_id: None,
                memory_id: None,
                concept_id: None,
                query: None,
                query_type: None,
                topic: None,
                payload: Some(serde_json::to_value(&payload).unwrap()),
            },
        )
        .unwrap()
    }

    fn emit_runtime_synth_with_hint(
        conn: &Connection,
        synthesis_id: &str,
        hint: SignalHint,
    ) -> i64 {
        let payload = SynthesisLlmJudgePayload {
            synthesis_id: synthesis_id.to_string(),
            judge_model: "model-r".into(),
            hit: true,
            reason: "useful".into(),
            stamp_hash: "deadbeef".into(),
            source: crate::store::adaptive::JudgeSource::AutoSampled,
            metadata: None,
            signal_hint: Some(hint),
        };
        super::emit_event(
            conn,
            FeedbackEvent {
                event_type: EventType::SynthesisLlmJudge,
                request_id: None,
                memory_id: None,
                concept_id: None,
                query: None,
                query_type: None,
                topic: None,
                payload: Some(serde_json::to_value(&payload).unwrap()),
            },
        )
        .unwrap()
    }

    fn emit_runtime_synth_without_hint(conn: &Connection, synthesis_id: &str) -> i64 {
        let payload = SynthesisLlmJudgePayload {
            synthesis_id: synthesis_id.to_string(),
            judge_model: "model-r".into(),
            hit: true,
            reason: "useful".into(),
            stamp_hash: "deadbeef".into(),
            source: crate::store::adaptive::JudgeSource::AutoSampled,
            metadata: None,
            signal_hint: None,
        };
        super::emit_event(
            conn,
            FeedbackEvent {
                event_type: EventType::SynthesisLlmJudge,
                request_id: None,
                memory_id: None,
                concept_id: None,
                query: None,
                query_type: None,
                topic: None,
                payload: Some(serde_json::to_value(&payload).unwrap()),
            },
        )
        .unwrap()
    }

    fn emit_runtime_concept_with_hint(
        conn: &Connection,
        concept_summary_id: &str,
        hint: SignalHint,
    ) -> i64 {
        let payload = ConceptSummaryLlmJudgePayload {
            concept_summary_id: concept_summary_id.to_string(),
            concept_id: "concept-1".into(),
            judge_model: "model-r".into(),
            hit: true,
            reason: "good summary".into(),
            stamp_hash: "cafebabe".into(),
            source: crate::store::adaptive::JudgeSource::AutoSampled,
            metadata: None,
            signal_hint: Some(hint),
        };
        super::emit_event(
            conn,
            FeedbackEvent {
                event_type: EventType::ConceptSummaryLlmJudge,
                request_id: None,
                memory_id: None,
                concept_id: Some("concept-1".into()),
                query: None,
                query_type: None,
                topic: None,
                payload: Some(serde_json::to_value(&payload).unwrap()),
            },
        )
        .unwrap()
    }

    #[test]
    fn bootstrap_priors_default_off_returns_const_defaults() {
        // §16.2 contract — default-off MUST return BootstrapPriors::const_defaults
        // bit-for-bit without reading snapshots or fixture corpus.
        let config = crate::config::ReinConfig::default();
        let priors = bootstrap_priors_from_corpus(&config).expect("default-off never errors");
        let defaults = BootstrapPriors::const_defaults();
        assert_eq!(priors, defaults);
        assert_eq!(priors.prior_confidence, 0.0);
    }

    #[test]
    fn bootstrap_priors_enabled_derives_from_fixture_signal_hints() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let mut config = crate::config::ReinConfig::default();
        config.ars.acceleration.enabled = true;
        config.hooks.buffer_dir = tmp.path().join("buffers").display().to_string();
        config.database.path = tmp.path().join("memories.db").display().to_string();

        let priors = bootstrap_priors_from_corpus(&config).expect("fixture corpus path");

        assert!((priors.w_view - 1.7).abs() < 1e-12, "got {priors:?}");
        assert!((priors.w_click - 2.1).abs() < 1e-12, "got {priors:?}");
        assert!((priors.w_thumb - 2.8).abs() < 1e-12, "got {priors:?}");
        assert!((priors.w_req - 1.1).abs() < 1e-12, "got {priors:?}");
        assert!(
            (priors.useful_rate_threshold - 0.46).abs() < 1e-12,
            "got {priors:?}"
        );
        assert_eq!(priors.prior_confidence, 2.0);
        assert!(
            !tmp.path().join("buffers").exists(),
            "fixture bootstrap must not create queue/snapshot directories"
        );
    }

    #[test]
    fn bootstrap_priors_default_off_does_not_create_snapshot_dir() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let mut config = crate::config::ReinConfig::default();
        config.hooks.buffer_dir = tmp.path().join("buffers").display().to_string();
        config.database.path = tmp.path().join("memories.db").display().to_string();

        let priors = bootstrap_priors_from_corpus(&config).expect("default path");

        assert_eq!(priors, BootstrapPriors::const_defaults());
        assert!(
            !tmp.path().join("buffers").exists(),
            "default-off prior bootstrap must not create queue/snapshot directories"
        );
    }

    #[test]
    fn bootstrap_priors_enabled_missing_snapshot_reads_fixture_without_creating_snapshot_dir() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let mut config = crate::config::ReinConfig::default();
        config.ars.acceleration.enabled = true;
        config.hooks.buffer_dir = tmp.path().join("buffers").display().to_string();
        config.database.path = tmp.path().join("memories.db").display().to_string();

        let priors = bootstrap_priors_from_corpus(&config).expect("missing snapshot falls back");

        assert_ne!(priors, BootstrapPriors::const_defaults());
        assert!(
            !tmp.path().join("buffers").exists(),
            "missing snapshot read path must not create queue/snapshot directories"
        );
    }

    #[test]
    fn bootstrap_priors_loads_valid_snapshot_when_acceleration_enabled() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let mut config = crate::config::ReinConfig::default();
        config.ars.acceleration.enabled = true;
        config.hooks.buffer_dir = tmp.path().join("buffers").display().to_string();
        config.database.path = tmp.path().join("memories.db").display().to_string();
        let snapshot = BootstrapPriors {
            w_view: 1.2,
            w_click: 1.4,
            w_thumb: 2.2,
            w_req: 0.8,
            useful_rate_threshold: 0.61,
            weight_decay_rate: 0.25,
            prior_confidence: 11.0,
        };
        std::fs::write(
            bootstrap_priors_snapshot_path(&config),
            serde_json::to_string(&snapshot).unwrap(),
        )
        .unwrap();

        let priors = bootstrap_priors_from_corpus(&config).expect("snapshot should load");

        assert_eq!(priors, snapshot);
    }

    #[test]
    fn bootstrap_priors_ignores_corrupt_snapshot_and_reads_fixture() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let mut config = crate::config::ReinConfig::default();
        config.ars.acceleration.enabled = true;
        config.hooks.buffer_dir = tmp.path().join("buffers").display().to_string();
        config.database.path = tmp.path().join("memories.db").display().to_string();
        std::fs::write(bootstrap_priors_snapshot_path(&config), "{not json").unwrap();

        let priors = bootstrap_priors_from_corpus(&config).expect("corrupt snapshot falls back");

        assert_ne!(priors, BootstrapPriors::const_defaults());
    }

    #[test]
    fn bootstrap_priors_from_replay_default_off_does_not_read_db() {
        let config = crate::config::ReinConfig::default();
        let conn = Connection::open_in_memory().unwrap();

        let priors = bootstrap_priors_from_replay(&config, &conn).expect("default-off path");

        assert_eq!(priors, BootstrapPriors::const_defaults());
    }

    #[test]
    fn bootstrap_priors_from_replay_prefers_valid_snapshot() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let mut config = crate::config::ReinConfig::default();
        config.ars.acceleration.enabled = true;
        config.hooks.buffer_dir = tmp.path().join("buffers").display().to_string();
        config.database.path = tmp.path().join("memories.db").display().to_string();
        let snapshot = BootstrapPriors {
            w_view: 1.25,
            w_click: 1.75,
            w_thumb: 2.25,
            w_req: 1.1,
            useful_rate_threshold: 0.55,
            weight_decay_rate: 0.2,
            prior_confidence: 9.0,
        };
        std::fs::write(
            bootstrap_priors_snapshot_path(&config),
            serde_json::to_string(&snapshot).unwrap(),
        )
        .unwrap();
        let conn = Connection::open_in_memory().unwrap();

        let priors = bootstrap_priors_from_replay(&config, &conn).expect("snapshot path");

        assert_eq!(priors, snapshot);
    }

    #[test]
    fn bootstrap_priors_from_replay_enabled_ignores_hintless_runtime_events() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let mut config = crate::config::ReinConfig::default();
        config.ars.acceleration.enabled = true;
        config.hooks.buffer_dir = tmp.path().join("buffers").display().to_string();
        config.database.path = tmp.path().join("memories.db").display().to_string();
        let conn = setup_db();
        emit_runtime_synth_without_hint(&conn, "synth-hintless");

        let priors = bootstrap_priors_from_replay(&config, &conn).expect("replay path");

        assert_eq!(priors, BootstrapPriors::const_defaults());
    }

    #[test]
    fn bootstrap_priors_from_replay_derives_from_runtime_signal_hints() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let mut config = crate::config::ReinConfig::default();
        config.ars.acceleration.enabled = true;
        config.hooks.buffer_dir = tmp.path().join("buffers").display().to_string();
        config.database.path = tmp.path().join("memories.db").display().to_string();
        let conn = setup_db();
        emit_runtime_synth_with_hint(
            &conn,
            "synth-1",
            SignalHint {
                inferred_w_view: Some(1.4),
                inferred_w_click: Some(2.0),
                inferred_w_thumb: Some(2.6),
                inferred_w_req: Some(0.8),
                useful_rate_ci_width: Some(0.25),
            },
        );
        emit_runtime_concept_with_hint(
            &conn,
            "summary-1",
            SignalHint {
                inferred_w_view: Some(0.6),
                inferred_w_click: Some(1.0),
                inferred_w_thumb: Some(1.4),
                inferred_w_req: Some(2.2),
                useful_rate_ci_width: Some(0.75),
            },
        );

        let priors = bootstrap_priors_from_replay(&config, &conn).expect("replay path");

        assert!((priors.w_view - 1.0).abs() < f64::EPSILON);
        assert!((priors.w_click - 1.5).abs() < f64::EPSILON);
        assert!((priors.w_thumb - 2.0).abs() < f64::EPSILON);
        assert!((priors.w_req - 1.5).abs() < f64::EPSILON);
        assert_eq!(priors.prior_confidence, 2.0);
    }

    #[test]
    fn extract_signal_hint_accepts_only_runtime_judge_events() {
        let hint = SignalHint {
            inferred_w_view: Some(1.4),
            inferred_w_click: Some(1.6),
            inferred_w_thumb: Some(2.2),
            inferred_w_req: Some(0.9),
            useful_rate_ci_width: Some(0.4),
        };
        let payload = SynthesisLlmJudgePayload {
            synthesis_id: "synth-1".into(),
            judge_model: "model-r".into(),
            hit: true,
            reason: "useful".into(),
            stamp_hash: "deadbeef".into(),
            source: crate::store::adaptive::JudgeSource::AutoSampled,
            metadata: None,
            signal_hint: Some(hint.clone()),
        };
        let raw = serde_json::to_string(&payload).unwrap();

        assert_eq!(
            extract_signal_hint_from_judge_event("synthesis_llm_judge", &raw),
            Some(hint)
        );
        assert_eq!(
            extract_signal_hint_from_judge_event("synthesis_llm_judge_offline_cron", &raw),
            None
        );
        assert_eq!(
            extract_signal_hint_from_judge_event("synthesis_llm_judge", "{not json"),
            None
        );
    }

    #[test]
    fn load_signal_hints_from_feedback_events_honors_limit() {
        let conn = setup_db();
        emit_runtime_synth_with_hint(
            &conn,
            "synth-1",
            SignalHint {
                inferred_w_view: Some(1.0),
                inferred_w_click: None,
                inferred_w_thumb: None,
                inferred_w_req: None,
                useful_rate_ci_width: None,
            },
        );
        emit_runtime_synth_with_hint(
            &conn,
            "synth-2",
            SignalHint {
                inferred_w_view: Some(2.0),
                inferred_w_click: None,
                inferred_w_thumb: None,
                inferred_w_req: None,
                useful_rate_ci_width: None,
            },
        );

        let hints = load_signal_hints_from_feedback_events(&conn, 0, 1).expect("load hints");

        assert_eq!(hints.len(), 1);
        assert_eq!(hints[0].inferred_w_view, Some(2.0));
    }

    #[test]
    fn derive_priors_from_signal_hints_uses_finite_hints() {
        let hints = vec![
            SignalHint {
                inferred_w_view: Some(1.2),
                inferred_w_click: Some(1.8),
                inferred_w_thumb: Some(2.4),
                inferred_w_req: Some(1.0),
                useful_rate_ci_width: Some(0.25),
            },
            SignalHint {
                inferred_w_view: Some(0.8),
                inferred_w_click: Some(1.2),
                inferred_w_thumb: Some(1.6),
                inferred_w_req: Some(2.0),
                useful_rate_ci_width: Some(0.75),
            },
        ];

        let priors = derive_priors_from_signal_hints(&hints);

        assert!((priors.w_view - 1.0).abs() < f64::EPSILON);
        assert!((priors.w_click - 1.5).abs() < f64::EPSILON);
        assert!((priors.w_thumb - 2.0).abs() < f64::EPSILON);
        assert!((priors.w_req - 1.5).abs() < f64::EPSILON);
        assert_eq!(priors.useful_rate_threshold, 0.5);
        assert_eq!(
            priors.weight_decay_rate,
            BootstrapPriors::const_defaults().weight_decay_rate
        );
        assert_eq!(priors.prior_confidence, 2.0);
    }

    #[test]
    fn derive_priors_from_signal_hints_ignores_invalid_values_per_field() {
        let defaults = BootstrapPriors::const_defaults();
        let hints = vec![
            SignalHint {
                inferred_w_view: Some(f64::NAN),
                inferred_w_click: Some(-1.0),
                inferred_w_thumb: Some(f64::INFINITY),
                inferred_w_req: Some(3.0),
                useful_rate_ci_width: Some(2.0),
            },
            SignalHint {
                inferred_w_view: None,
                inferred_w_click: None,
                inferred_w_thumb: None,
                inferred_w_req: None,
                useful_rate_ci_width: None,
            },
        ];

        let priors = derive_priors_from_signal_hints(&hints);

        assert_eq!(priors.w_view, defaults.w_view);
        assert_eq!(priors.w_click, defaults.w_click);
        assert_eq!(priors.w_thumb, defaults.w_thumb);
        assert_eq!(priors.w_req, 3.0);
        assert!((0.1..=0.9).contains(&priors.useful_rate_threshold));
        assert_eq!(priors.prior_confidence, 1.0);
    }

    #[test]
    fn should_archive_for_cron_zero_rate_never_samples() {
        for i in 0..100 {
            let id = format!("synth_{i}");
            assert!(!should_archive_for_cron(&id, 0.0));
        }
    }

    #[test]
    fn should_archive_for_cron_one_rate_always_samples() {
        for i in 0..100 {
            let id = format!("synth_{i}");
            assert!(should_archive_for_cron(&id, 1.0));
        }
    }

    #[test]
    fn should_archive_for_cron_deterministic_for_same_id() {
        // Same id + same rate must always produce the same outcome —
        // cron must know which synthesis_ids to expect without scanning
        // the runtime cache.
        let id = "synth_abc123";
        let first = should_archive_for_cron(id, 0.5);
        for _ in 0..10 {
            assert_eq!(should_archive_for_cron(id, 0.5), first);
        }
    }

    #[test]
    fn should_archive_for_cron_rejects_non_finite() {
        assert!(!should_archive_for_cron("x", f64::NAN));
        assert!(!should_archive_for_cron("x", f64::NEG_INFINITY));
        // Positive infinity clamps to 1.0 → always sample.
        assert!(should_archive_for_cron("x", f64::INFINITY));
    }

    #[test]
    fn should_archive_for_cron_approximates_target_rate() {
        // Distribution check: across many ids at rate=0.2, sampled count
        // should land within ±5% of 200/1000.
        let mut sampled = 0;
        for i in 0..1000 {
            let id = format!("synth_{i:08}");
            if should_archive_for_cron(&id, 0.2) {
                sampled += 1;
            }
        }
        // Allow 150-250 (±5% absolute, ±25% relative) — SHA-256 prefix is
        // uniform so this is a generous tolerance.
        assert!(
            (150..=250).contains(&sampled),
            "expected ~200 samples at rate 0.2, got {sampled}"
        );
    }

    #[test]
    fn consumer_no_events_returns_none_max_id() {
        let conn = setup_db();
        let (state, max_id) = recompute_judge_calibration_state(&conn, None, None).unwrap();
        assert!(max_id.is_none());
        assert_eq!(state.total_offline_cron_events, 0);
        assert_eq!(state.runtime_vs_offline_kappa, 0.0);
    }

    #[test]
    fn consumer_aggregates_runtime_vs_cron_pairs() {
        let conn = setup_db();
        // 4 synthetic events: 3 agree (true,true) + 1 disagree (true,false).
        emit_offline_cron_synth(&conn, "s1", true, true);
        emit_offline_cron_synth(&conn, "s2", true, true);
        emit_offline_cron_synth(&conn, "s3", true, true);
        emit_offline_cron_synth(&conn, "s4", true, false);

        let (state, max_id) = recompute_judge_calibration_state(&conn, None, None).unwrap();
        assert_eq!(max_id, Some(4));
        assert_eq!(state.total_offline_cron_events, 4);
        assert_eq!(state.recent_pairs_runtime_vs_offline.len(), 4);
        // Must compute SOME κ value (specifically degenerate — see kappa
        // tests; runtime always true → marginal degenerate, p_e=cron_true_rate
        // so kappa is well-defined). Whatever the value, advancing the
        // applied-prefix is the contract.
        assert!(state.last_consumed_event_id_calibration >= 4);
    }

    #[test]
    fn consumer_tracks_runtime_vs_offline_per_surface() {
        let conn = setup_db();
        for i in 0..30 {
            emit_offline_cron_synth(&conn, &format!("s{i}"), true, true);
            emit_offline_cron_concept(&conn, &format!("c{i}"), true, false);
        }

        let (state, max_id) = recompute_judge_calibration_state(&conn, None, None).unwrap();

        assert_eq!(max_id, Some(60));
        assert_eq!(state.recent_pairs_runtime_vs_offline.len(), 60);
        assert_eq!(state.recent_pairs_runtime_vs_offline_synthesis.len(), 30);
        assert_eq!(state.recent_pairs_runtime_vs_offline_concept.len(), 30);
        assert!(state.runtime_vs_offline_kappa_synthesis >= 0.99);
        assert!(state.runtime_vs_offline_kappa_concept < JUDGE_DRIFT_THRESHOLD);
        assert_eq!(state.judge_drift_alert_synthesis, 0);
        assert_eq!(state.judge_drift_alert_concept, 1);
    }

    #[test]
    fn consumer_replay_safety_skips_already_applied_events() {
        let conn = setup_db();
        emit_offline_cron_synth(&conn, "s1", true, true);
        emit_offline_cron_synth(&conn, "s2", false, false);

        // First pass — drains both events.
        let (state, max_id) = recompute_judge_calibration_state(&conn, None, None).unwrap();
        assert_eq!(max_id, Some(2));
        assert_eq!(state.total_offline_cron_events, 2);
        // The caller is responsible for committing the offset; the
        // consumer's applied-prefix bump is enough on its own to skip
        // already-applied events on replay (invariant 1). Confirm replay
        // with the RETURNED state as prior produces no new pairs.
        let (state2, max_id2) =
            recompute_judge_calibration_state(&conn, Some(state.clone()), None).unwrap();
        // peek still returns events (offset not committed), but the
        // applied-prefix in `state` filters them all out.
        assert_eq!(
            state2.total_offline_cron_events, 2,
            "no replay double-count"
        );
        // max_id is None when no NEW events past prior_high_water are
        // seen. Wait — peek returns events past consumer_offset (not state
        // watermark). On a fresh DB without commit_offset call,
        // consumer_offset stays 0 and peek returns the same events. The
        // state's applied-prefix filters them. max_id WILL be Some(2)
        // because peek returns max event id — but the loop doesn't fold
        // any of them. That's expected per the contract.
        assert_eq!(max_id2, Some(2));
    }

    #[test]
    fn consumer_after_commit_offset_skips_replay_entirely() {
        let conn = setup_db();
        emit_offline_cron_synth(&conn, "s1", true, true);
        emit_offline_cron_synth(&conn, "s2", false, false);

        // Drain + commit offset.
        let (_state, max_id) = recompute_judge_calibration_state(&conn, None, None).unwrap();
        commit_offset(&conn, &[(JUDGE_CALIBRATION_CONSUMER, max_id.unwrap())]).unwrap();

        // Verify peek returns nothing now.
        let events = peek_events(
            &conn,
            JUDGE_CALIBRATION_CONSUMER,
            &[
                EventType::SynthesisLlmJudgeOfflineCron.as_str(),
                EventType::ConceptSummaryLlmJudgeOfflineCron.as_str(),
            ],
            10,
        )
        .unwrap();
        assert!(events.is_empty(), "post-commit peek must be empty");
    }

    #[test]
    fn consumer_drift_alert_bumps_on_threshold_cross() {
        let conn = setup_db();
        // Seed 30 perfect-agreement pairs first → κ = 1.0, no alert.
        for i in 0..30 {
            let id = format!("s{i}");
            emit_offline_cron_synth(&conn, &id, true, true);
        }
        let drift_log = std::env::temp_dir().join(format!(
            "rein_drift_test_{}.log",
            chrono::Utc::now().timestamp_millis()
        ));
        let _ = std::fs::remove_file(&drift_log);

        let (state1, max_id1) =
            recompute_judge_calibration_state(&conn, None, Some(&drift_log)).unwrap();
        assert!(max_id1.is_some());
        assert_eq!(state1.judge_drift_alert, 0);
        assert!(state1.runtime_vs_offline_kappa >= 0.99); // perfect agreement
        commit_offset(&conn, &[(JUDGE_CALIBRATION_CONSUMER, max_id1.unwrap())]).unwrap();

        // Now flood with 30 perfect-disagreement pairs → κ ≈ -1.0, crossing
        // below threshold → drift alert fires once.
        for i in 30..60 {
            let id = format!("s{i}");
            emit_offline_cron_synth(&conn, &id, true, false);
        }
        let (state2, max_id2) =
            recompute_judge_calibration_state(&conn, Some(state1), Some(&drift_log)).unwrap();
        assert!(max_id2.is_some());
        assert_eq!(state2.judge_drift_alert, 1, "alert must fire on crossing");
        assert!(state2.runtime_vs_offline_kappa < JUDGE_DRIFT_THRESHOLD);

        // Drift log should have at least one line.
        let log_text = std::fs::read_to_string(&drift_log).unwrap_or_default();
        assert!(
            log_text.contains("drift_alert"),
            "drift log should contain alert line, got: {log_text}"
        );
        let _ = std::fs::remove_file(&drift_log);
    }

    #[test]
    fn consumer_drift_alert_does_not_fire_below_min_pairs() {
        let conn = setup_db();
        // Only 5 disagreement pairs (below JUDGE_DRIFT_MIN_PAIRS=30).
        for i in 0..5 {
            let id = format!("s{i}");
            emit_offline_cron_synth(&conn, &id, true, false);
        }
        let (state, _) = recompute_judge_calibration_state(&conn, None, None).unwrap();
        assert_eq!(state.judge_drift_alert, 0, "below min_pairs => no alert");
    }

    #[test]
    fn cron_archive_path_is_db_scoped() {
        // Smoke test: path layout matches the `<base>/queue/<db_hash>/<file>`
        // contract regardless of base value. Avoid asserting the exact
        // hash since DefaultHasher is platform-defined; assert structure.
        let config = crate::config::ReinConfig::default();
        let date = chrono::NaiveDate::from_ymd_opt(2026, 5, 1).unwrap();
        let p = cron_archive_path(&config, date, 0);
        let s = p.to_string_lossy().into_owned();
        assert!(s.contains("queue"));
        assert!(s.contains("synthesis_cron_archive_20260501_0.jsonl"));
    }

    #[test]
    fn try_claim_cron_first_winner_wins_loser_skips() {
        // v0.27.5 R3 — pre-LLM atomic claim primitive. The first
        // INSERT OR IGNORE for a (event_type, surface_id, stamp_hash)
        // tuple MUST return Ok(Some(token)); a second for the same
        // tuple MUST return Ok(None). This proves two concurrent
        // crons can't both burn `daily_call_cap` on the same entry.
        let conn = setup_db();
        let surface = CronArchiveSurface::Synthesis;

        // First writer wins.
        let first = try_claim_cron(&conn, &surface, "synth-1", "stamp-A").unwrap();
        assert!(first.is_some(), "first writer must win the claim");

        // Second writer (concurrent peer) loses.
        let second = try_claim_cron(&conn, &surface, "synth-1", "stamp-A").unwrap();
        assert!(
            second.is_none(),
            "second writer of identical tuple MUST observe the conflict"
        );

        // Different stamp_hash on same id is a fresh tuple — proves the
        // claim is keyed correctly (re-mint with different prompt judges
        // afresh).
        let fresh_stamp = try_claim_cron(&conn, &surface, "synth-1", "stamp-B").unwrap();
        assert!(
            fresh_stamp.is_some(),
            "different stamp_hash on same id MUST be a fresh tuple"
        );

        // Different surface_id under the same event_type also fresh.
        let fresh_id = try_claim_cron(&conn, &surface, "synth-2", "stamp-A").unwrap();
        assert!(
            fresh_id.is_some(),
            "different surface_id MUST be a fresh tuple"
        );

        // ConceptSummary surface uses a different event_type → also fresh,
        // even with identical id + stamp.
        let fresh_surface = try_claim_cron(
            &conn,
            &CronArchiveSurface::ConceptSummary,
            "synth-1",
            "stamp-A",
        )
        .unwrap();
        assert!(
            fresh_surface.is_some(),
            "different event_type MUST be a fresh tuple"
        );

        // Verify exactly 4 winning rows landed.
        let row_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM cron_claims", [], |row| row.get(0))
            .unwrap();
        assert_eq!(
            row_count, 4,
            "only winning claims persist; losers are no-ops"
        );

        // Each winning claim MUST hold a non-empty token (codex R3 P2
        // ownership-safety requires it for `release_cron_claim`).
        assert!(!first.unwrap().is_empty());
        assert!(!fresh_stamp.unwrap().is_empty());
        assert!(!fresh_id.unwrap().is_empty());
        assert!(!fresh_surface.unwrap().is_empty());
    }

    #[test]
    fn try_claim_cron_takes_over_stale_claim_after_crash_window() {
        // v0.27.5 R3 — stale-claim takeover. If a previous cron crashed
        // after `try_claim_cron` inserted a row but before it emitted,
        // the row's `claimed_at` is older than `CRON_CLAIM_STALE_SECS`.
        // The next caller MUST treat the row as orphaned and take it
        // over (return Ok(Some(new_token)) and bump claimed_at to now).
        // Without this, the entry would be permanently `SkippedDuplicate`.
        let conn = setup_db();
        let surface = CronArchiveSurface::Synthesis;
        let event_type = EventType::SynthesisLlmJudgeOfflineCron.as_str();

        // Pre-insert a stale claim row (claimed_at far in the past) with
        // a known token belonging to the "original" (now dead) cron.
        conn.execute(
            "INSERT INTO cron_claims (event_type, surface_id, stamp_hash, claim_token, claimed_at) \
             VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![event_type, "synth-1", "stamp-A", "ORIGINAL_TOKEN", 1_000_000i64],
        )
        .unwrap();

        // Stale takeover: try_claim_cron returns Ok(Some(new_token)),
        // claim_token is overwritten, claimed_at bumped to ~now.
        let new_token = try_claim_cron(&conn, &surface, "synth-1", "stamp-A").unwrap();
        let new_token = new_token.expect("stale claim row MUST be reclaimable");
        assert_ne!(new_token, "ORIGINAL_TOKEN", "takeover mints a fresh token");

        let (stored_token, claimed_at): (String, i64) = conn
            .query_row(
                "SELECT claim_token, claimed_at FROM cron_claims \
                 WHERE event_type = ?1 AND surface_id = ?2 AND stamp_hash = ?3",
                rusqlite::params![event_type, "synth-1", "stamp-A"],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(stored_token, new_token, "stored token MUST match returned");
        let now_unix = chrono::Utc::now().timestamp();
        assert!(
            (now_unix - claimed_at).abs() < 10,
            "stale takeover MUST refresh claimed_at to ~now (got {claimed_at} vs now {now_unix})"
        );

        // Concurrent peer immediately after takeover: claim is fresh,
        // so peer loses.
        let peer = try_claim_cron(&conn, &surface, "synth-1", "stamp-A").unwrap();
        assert!(
            peer.is_none(),
            "after takeover the new owner is fresh, peer MUST lose"
        );

        // Codex R3 P2 ownership-safety: the original (slow) cron's
        // `release_cron_claim` with the OLD token MUST be a no-op,
        // never clobbering the fresh peer's row.
        release_cron_claim(&conn, &surface, "synth-1", "stamp-A", "ORIGINAL_TOKEN").unwrap();
        let still_owned: String = conn
            .query_row(
                "SELECT claim_token FROM cron_claims \
                 WHERE event_type = ?1 AND surface_id = ?2 AND stamp_hash = ?3",
                rusqlite::params![event_type, "synth-1", "stamp-A"],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(
            still_owned, new_token,
            "original-token release MUST NOT delete the fresh peer's row"
        );
    }

    #[test]
    fn release_cron_claim_lets_a_future_cron_pass_retry() {
        // v0.27.5 R3 — `release_cron_claim` is called on no-cap-burn
        // failure paths (`reserve_call` → Ok(None) / Err) so a future
        // cron pass can retry. After release, `try_claim_cron` for the
        // same tuple MUST succeed again.
        let conn = setup_db();
        let surface = CronArchiveSurface::Synthesis;

        // First claim wins.
        let token1 = try_claim_cron(&conn, &surface, "synth-1", "stamp-A").unwrap();
        let token1 = token1.expect("first claim wins");
        // Concurrent peer loses.
        assert!(try_claim_cron(&conn, &surface, "synth-1", "stamp-A")
            .unwrap()
            .is_none());

        // Simulate a no-cap-burn failure: owner releases the claim.
        release_cron_claim(&conn, &surface, "synth-1", "stamp-A", &token1).unwrap();

        // Future cron pass for the same tuple MUST succeed again.
        let retry = try_claim_cron(&conn, &surface, "synth-1", "stamp-A").unwrap();
        assert!(retry.is_some(), "after release, future claim MUST succeed");

        // Releasing a non-existent claim is a no-op (best-effort
        // semantics — used inside `let _ = ...` patterns).
        release_cron_claim(&conn, &surface, "synth-never", "stamp-Z", "wrong-token").unwrap();
    }

    #[test]
    fn cron_run_with_empty_archive_returns_zero_report() {
        let config = crate::config::ReinConfig::default();
        // Open a fresh in-memory store (no archive files in scope).
        let store = SqliteStore::new(
            &config.resolve_db_path(),
            &config.embedding_model(),
            crate::config::ReinConfig::default().embedding.dimensions,
        )
        .or_else(|_| {
            // fallback: if the resolved DB doesn't exist on disk, create
            // a tempfile path
            let tmp = std::env::temp_dir().join(format!(
                "rein_test_{}.db",
                chrono::Utc::now().timestamp_millis()
            ));
            SqliteStore::new(
                &tmp,
                &config.embedding_model(),
                crate::config::ReinConfig::default().embedding.dimensions,
            )
        });
        if let Ok(store) = store {
            let report =
                run_judge_calibration_cron(&store, &config).expect("empty archive must succeed");
            assert_eq!(report.considered, 0);
            assert_eq!(report.emitted, 0);
        }
    }
}
