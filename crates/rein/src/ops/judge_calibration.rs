//! v0.27.1 E direction Layer 2 — nightly stricter offline calibration cron
//! + `judge_calibration` M1 consumer + κ accumulator + drift alert
//! + `bootstrap_priors_from_corpus` v0.28 stub.
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
//!    v0.28 forward-compat hook (§16.2). v0.27.1 returns
//!    [`BootstrapPriors::const_defaults`] — pure const, no I/O, no LLM. Locks
//!    the function signature so v0.28 implementation slots in without
//!    changing caller signatures.
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
    ConceptSummaryLlmJudgeOfflineCronPayload, EventType, FeedbackEvent, JudgeCalibrationState,
    JudgeMetadata, SynthesisLlmJudgeOfflineCronPayload, JUDGE_DRIFT_MIN_PAIRS,
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

// ── §16.2 v0.28 bootstrap_priors_from_corpus stub ────────────────────────────

/// v0.28 entrypoint for offline Bayesian prior derivation from the fixture
/// corpus + production replay events. v0.27.1 ships as a no-op stub returning
/// hardcoded defaults — the call site exists so v0.28 implementation slots in
/// without changing caller signatures.
///
/// v0.28 will:
/// 1. Load `crates/rein/tests/fixtures/recall_synthesis/*` + production
///    replay events from `feedback_events` last 30d
/// 2. Run multi-param logistic regression / Bayesian posterior inference on
///    `signal_hint`-labeled events to estimate cluster-pooled priors for
///    W_VIEW / W_CLICK / W_THUMB / W_REQ / useful_rate threshold
/// 3. Apply hierarchical shrinkage (S2 brainstorm: topic → cluster →
///    memory) so cold clusters borrow same-topic prior
/// 4. Write to `~/.rein/judge_priors.json` snapshot — adaptive engine reads
///    on boot and uses as Bayesian prior; production feedback updates the
///    posterior
///
/// **v0.27.1**: returns `BootstrapPriors::const_defaults()`. NO file I/O.
/// NO LLM calls. The `_config` parameter is intentionally unused and kept
/// only to lock the v0.28 signature.
pub fn bootstrap_priors_from_corpus(_config: &ReinConfig) -> ReinResult<BootstrapPriors> {
    // v0.27.1 stub: return const defaults. NO file I/O, NO LLM calls.
    Ok(BootstrapPriors::const_defaults())
}

/// Bootstrap priors snapshot. v0.27.1 = const defaults; v0.28+ = posterior-
/// derived from corpus + production replay (see [`bootstrap_priors_from_corpus`]).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct BootstrapPriors {
    pub w_view: f64,
    pub w_click: f64,
    pub w_thumb: f64,
    pub w_req: f64,
    pub useful_rate_threshold: f64,
    pub weight_decay_rate: f64,
    /// Confidence in this prior (Bayesian: pseudo-observation count).
    /// v0.27.1 stub returns 0.0 (no production-derived prior).
    pub prior_confidence: f64,
}

impl BootstrapPriors {
    /// v0.27.1 hardcoded defaults. v0.28 replaces with corpus-derived
    /// posterior. Caller never branches on which path produced these.
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
    let base = crate::extract::hooks::buffer::resolve_buffer_dir(config);
    let db_tag = {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        config.resolve_db_path().hash(&mut h);
        format!("{:016x}", h.finish())
    };
    let queue_dir = base.join("queue").join(&db_tag);
    let _ = std::fs::create_dir_all(&queue_dir);
    let date_str = date.format("%Y%m%d").to_string();
    queue_dir.join(format!("synthesis_cron_archive_{date_str}_{shard}.jsonl"))
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

        // Same `(runtime_hit, cron_hit)` extraction shape for both surfaces —
        // the consumer doesn't differentiate at the κ level. Per-surface drift
        // would require splitting `recent_pairs_runtime_vs_offline` into
        // synthesis vs concept arms, deferred to v0.28+ (matches R9-K4 spirit
        // for the J3 layer; v0.27.1 keeps Layer 2 single-window since only
        // one drift alert exists).
        let pair: Option<(bool, bool)> = match ev.event_type.as_str() {
            "synthesis_llm_judge_offline_cron" => {
                match serde_json::from_str::<SynthesisLlmJudgeOfflineCronPayload>(payload_str) {
                    Ok(p) => Some((p.runtime_hit, p.cron_hit)),
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
                    Ok(p) => Some((p.runtime_hit, p.cron_hit)),
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

        if let Some((runtime, cron)) = pair {
            state.recent_pairs_runtime_vs_offline.push_back((
                runtime,
                cron,
                ev.ts_to_unix().unwrap_or(now),
            ));
            // FIFO-evict oldest pair if over cap.
            while state.recent_pairs_runtime_vs_offline.len() > JUDGE_RUNTIME_VS_OFFLINE_PAIRS_CAP {
                state.recent_pairs_runtime_vs_offline.pop_front();
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
        Ok(true) => return ProcessOutcome::SkippedDuplicate,
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
        Ok(None) => return ProcessOutcome::DroppedCap,
        Err(e) => return ProcessOutcome::Dropped(format!("reserve_call: {e}")),
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
        Ok(_) => ProcessOutcome::Emitted,
        Err(e) => ProcessOutcome::Dropped(format!("emit_event failed: {e}")),
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
    let raw = crate::ops::llm_judge_worker::call_judge_sync(
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

    #[test]
    fn bootstrap_priors_stub_returns_const_defaults() {
        // §16.2 contract — v0.27.1 stub MUST return BootstrapPriors::const_defaults
        // bit-for-bit. Caller cannot branch on which path produced these.
        let config = crate::config::ReinConfig::default();
        let priors = bootstrap_priors_from_corpus(&config).expect("stub never errors");
        let defaults = BootstrapPriors::const_defaults();
        assert_eq!(priors, defaults);
        assert_eq!(priors.prior_confidence, 0.0);
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
