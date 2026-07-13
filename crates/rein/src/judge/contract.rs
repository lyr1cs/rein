//! v0.27.1 E direction — Judge Contract (J1-J7, spec §4).
//!
//! Each invariant is a pure function `fn(&JudgeContext, &JudgePayload) ->
//! Result<(), JudgeViolation>`. The worker validates each invariant before
//! emitting an event; violation → log + drop (no panic, no recall-path
//! impact).
//!
//! Modeled on `crate::compression::contract` (v0.23 Lossless Compression
//! Contract) and the v0.26.2 invariant-discipline pattern.
//!
//! # Pipeline-interaction discipline (spec §5)
//!
//! The judge worker is **deliberately designed to NOT enter the v0.26.x
//! 4-way pipeline-interaction matrix** (`update()` × `apply_evolution`
//! × `cold_archive` × `M5 strip`). J1 enforces this by allow-listing the
//! worker's write set to `{feedback_events, judge_call_ledger,
//! consumer_offsets, adaptive_state, concept_summary_instances}` only. See
//! `feedback_pipeline_interaction_matrix.md` for the broader v0.26.x
//! audit lesson.

use crate::store::adaptive::{
    ConceptSummaryLlmJudgePayload, SynthesisLlmJudgePayload, LLM_JUDGE_J3_MIN_PAIRS,
    LLM_JUDGE_KAPPA_FLOOR,
};
use crate::types::{ReinError, ReinResult};
use rusqlite::Connection;
use std::time::{SystemTime, UNIX_EPOCH};

/// Daily call cap (J2) — bootstrap default. Operator overrides via
/// `[ars.llm_judge].daily_call_cap`. Worker drops jobs when the rolling
/// 24h reservation count meets or exceeds this value.
pub const LLM_JUDGE_DAILY_CALL_CAP_DEFAULT: u64 = 10_000;

/// Stale-claim timeout (J2) — bootstrap default. Reservations older than
/// this are reaped by the worker on each pull, mirroring v0.23
/// resummerize claim-token discipline.
pub const LLM_JUDGE_STALE_CLAIM_SECS: i64 = 5 * 60;

/// Health of the deterministic structural-anchor suite for one judge surface.
/// Structural anchors are deliberately separate from human-pair κ evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JudgeStructuralStatus {
    Disabled,
    Collecting,
    Ready,
    Failed,
    Stale,
    FingerprintMismatch,
    Corrupt,
    Unknown,
}

/// Evidence consumed by [`judge_trust_gate`].
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct JudgeCalibrationEvidence {
    pub human_pair_count: usize,
    pub human_kappa: Option<f64>,
    pub structural_status: JudgeStructuralStatus,
    pub enforce_structural_anchors: bool,
}

/// A requested use of judge trust. Structural anchors may preserve the
/// configured baseline, but never authorize any of the automatic increases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JudgeTrustAction {
    KeepConfiguredBaseline,
    IncreaseJudgeWeight,
    IncreaseSampleRate,
    PromoteJudgeScope,
    PromoteRecallFusion,
}

/// Evidence source that actually authorized the returned decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JudgeCalibrationBasis {
    HumanKappa,
    StructuralAnchors,
    ConfiguredBaseline,
    Untrusted,
}

/// Stable reason for a trust-gate outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JudgeTrustReason {
    HumanKappaHealthy,
    HumanKappaBelowFloor,
    HumanKappaMissingOrInvalid,
    StructuralAnchorsReady,
    ConfiguredBaselineFallback,
    StructuralHealthEnforced,
}

/// Fully resolved trust policy. Callers must use `configured_baseline_scale`
/// for both judge contribution and configured sample rate rather than
/// re-deriving whether either surface stays configured or is forced to zero.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct JudgeTrustDecision {
    pub basis: JudgeCalibrationBasis,
    pub action_allowed: bool,
    /// Only `0.0` (fail closed) or `1.0` (retain configured baseline).
    pub configured_baseline_scale: f64,
    pub blocks_release: bool,
    pub reason: JudgeTrustReason,
}

/// Resolve whether `requested_action` is authorized and whether the configured
/// judge baseline remains live.
pub fn judge_trust_gate(
    evidence: &JudgeCalibrationEvidence,
    requested_action: JudgeTrustAction,
) -> JudgeTrustDecision {
    let human_kappa_is_authoritative = evidence.human_pair_count >= LLM_JUDGE_J3_MIN_PAIRS;

    if human_kappa_is_authoritative {
        match evidence.human_kappa {
            Some(kappa) if !kappa.is_finite() || !(-1.0..=1.0).contains(&kappa) => {
                return JudgeTrustDecision {
                    basis: JudgeCalibrationBasis::Untrusted,
                    action_allowed: false,
                    configured_baseline_scale: 0.0,
                    blocks_release: true,
                    reason: JudgeTrustReason::HumanKappaMissingOrInvalid,
                };
            }
            Some(kappa) if kappa < LLM_JUDGE_KAPPA_FLOOR => {
                return JudgeTrustDecision {
                    basis: JudgeCalibrationBasis::Untrusted,
                    action_allowed: false,
                    configured_baseline_scale: 0.0,
                    blocks_release: true,
                    reason: JudgeTrustReason::HumanKappaBelowFloor,
                };
            }
            Some(_) => {}
            _ => {
                return JudgeTrustDecision {
                    basis: JudgeCalibrationBasis::Untrusted,
                    action_allowed: false,
                    configured_baseline_scale: 0.0,
                    blocks_release: true,
                    reason: JudgeTrustReason::HumanKappaMissingOrInvalid,
                };
            }
        }
    }

    let structural_failure = matches!(
        evidence.structural_status,
        JudgeStructuralStatus::Failed
            | JudgeStructuralStatus::Stale
            | JudgeStructuralStatus::FingerprintMismatch
            | JudgeStructuralStatus::Corrupt
            | JudgeStructuralStatus::Unknown
    );
    if evidence.enforce_structural_anchors && structural_failure {
        return JudgeTrustDecision {
            basis: JudgeCalibrationBasis::Untrusted,
            action_allowed: false,
            configured_baseline_scale: 0.0,
            blocks_release: true,
            reason: JudgeTrustReason::StructuralHealthEnforced,
        };
    }

    if human_kappa_is_authoritative {
        return JudgeTrustDecision {
            basis: JudgeCalibrationBasis::HumanKappa,
            action_allowed: true,
            configured_baseline_scale: 1.0,
            blocks_release: false,
            reason: JudgeTrustReason::HumanKappaHealthy,
        };
    }

    let (basis, reason) = if evidence.structural_status == JudgeStructuralStatus::Ready {
        (
            JudgeCalibrationBasis::StructuralAnchors,
            JudgeTrustReason::StructuralAnchorsReady,
        )
    } else {
        (
            JudgeCalibrationBasis::ConfiguredBaseline,
            JudgeTrustReason::ConfiguredBaselineFallback,
        )
    };

    JudgeTrustDecision {
        basis,
        action_allowed: requested_action == JudgeTrustAction::KeepConfiguredBaseline,
        configured_baseline_scale: 1.0,
        blocks_release: false,
        reason,
    }
}

/// v0.27.1 — discriminated payload variants the contract validates against.
/// Lets J5 (link-present) / J7 (stamp-time) operate uniformly across both
/// runtime payloads without monomorphizing every invariant.
#[derive(Debug, Clone)]
pub enum JudgePayload<'a> {
    Synthesis(&'a SynthesisLlmJudgePayload),
    ConceptSummary(&'a ConceptSummaryLlmJudgePayload),
}

impl<'a> JudgePayload<'a> {
    /// Surface-id ULID — `synthesis_id` for Cap B, `concept_summary_id`
    /// for Cap A.
    pub fn surface_id(&self) -> &str {
        match self {
            Self::Synthesis(p) => p.synthesis_id.as_str(),
            Self::ConceptSummary(p) => p.concept_summary_id.as_str(),
        }
    }

    /// Stamp-hash for J7 (post-truncation prompt bytes the runtime judge
    /// actually saw).
    pub fn stamp_hash(&self) -> &str {
        match self {
            Self::Synthesis(p) => p.stamp_hash.as_str(),
            Self::ConceptSummary(p) => p.stamp_hash.as_str(),
        }
    }
}

/// v0.27.1 — context the worker passes to each invariant. Carries the
/// minimal set of fields the J* fns inspect; deliberately small so each
/// invariant stays cheap and side-effect-free.
#[derive(Debug, Clone)]
pub struct JudgeContext<'a> {
    /// Synthesis or concept-summary surface text the worker reads from
    /// the queue payload (J7 stamp-time invariant — never re-queried
    /// from `memories` / `concepts` between enqueue and judge).
    pub stamp_time_source: &'a str,
    /// J7 — sha256 of the post-truncation prompt bytes the worker
    /// actually fed to the judge model. MUST match `payload.stamp_hash`.
    pub computed_stamp_hash: &'a str,
    /// J3 — pair count for the surface's κ accumulator.
    /// `recent_pairs_synthesis.len()` for Cap B; `recent_pairs_concept`
    /// for Cap A. Used to decide whether κ is "defined" (i.e. ≥
    /// [`LLM_JUDGE_J3_MIN_PAIRS`]).
    pub surface_kappa_pair_count: usize,
    /// J3 — current κ value for the surface.
    pub surface_kappa: f64,
    /// Whether the worker is about to raise the warm sample rate. J3
    /// trips ONLY when this is true — judge runs at cold-start sample
    /// rate even with κ < floor (the runtime gate is open by design).
    pub raising_sample_rate: bool,
}

/// v0.27.1 E direction — discriminated invariant-violation kind. Each
/// variant maps 1:1 to a J* invariant in spec §4. Worker increments
/// `AdaptiveState.judge_contract_violations` per kind and skips emit;
/// doctor surfaces the count.
#[derive(Debug, Clone, PartialEq)]
pub enum JudgeViolation {
    /// J1 — worker attempted a write outside the allow-list
    /// `{feedback_events, judge_call_ledger, consumer_offsets,
    /// adaptive_state, concept_summary_instances}`.
    NoMemoryWrites { table: String },
    /// J2 — daily call cap reached. Surfaced when reservation fails;
    /// worker drops the job.
    DailyCapReached { current: u64, cap: u64 },
    /// J3 — κ vs ExplicitThumb is defined (pair count ≥
    /// [`LLM_JUDGE_J3_MIN_PAIRS`]) AND κ < floor; worker MUST NOT raise
    /// `sample_rate_warm`.
    SelfReinforce {
        kappa: f64,
        floor: f64,
        pairs: usize,
    },
    /// J4 — worker errored in a way that would propagate to the recall
    /// critical path. Worker downgrades to a tracing::warn! and drops
    /// the event. (Reserved for future structured failure modes.)
    BlocksRecall { reason: String },
    /// J5 — judge event payload's `surface_id` doesn't link to a known
    /// synthesis / concept-summary instance.
    LinkAbsent { surface_id: String },
    /// J6 — `weight_decay_rate` outside `[0.0, 1.0]` (or non-finite).
    /// Caught at config validation time; this variant exists for
    /// runtime defense-in-depth if the value is ever computed
    /// dynamically.
    WeightDecayInvalid { value: f64 },
    /// J7 — payload `stamp_hash` doesn't match the worker's computed
    /// hash over the post-truncation prompt bytes.
    StampHashMismatch {
        payload_hash: String,
        computed_hash: String,
    },
}

impl std::fmt::Display for JudgeViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoMemoryWrites { table } => write!(
                f,
                "J1 violation: worker attempted write to non-allowlisted table '{table}'"
            ),
            Self::DailyCapReached { current, cap } => write!(
                f,
                "J2 violation: daily call cap reached (current={current}, cap={cap})"
            ),
            Self::SelfReinforce { kappa, floor, pairs } => write!(
                f,
                "J3 violation: κ {kappa:.3} < floor {floor:.3} (over {pairs} pairs); refusing sample-rate raise"
            ),
            Self::BlocksRecall { reason } => write!(f, "J4 violation: {reason}"),
            Self::LinkAbsent { surface_id } => write!(
                f,
                "J5 violation: surface_id '{surface_id}' does not link to a known instance"
            ),
            Self::WeightDecayInvalid { value } => write!(
                f,
                "J6 violation: weight_decay_rate {value} outside [0, 1] or non-finite"
            ),
            Self::StampHashMismatch {
                payload_hash,
                computed_hash,
            } => write!(
                f,
                "J7 violation: stamp_hash mismatch (payload={payload_hash}, computed={computed_hash})"
            ),
        }
    }
}

/// J1 — the worker's allow-listed write set. Any write the worker performs
/// must target one of these tables; everything else is a violation. This
/// is the load-bearing invariant that keeps the judge worker out of the
/// v0.26.x 4-way pipeline-interaction matrix.
pub const J1_ALLOWED_WRITE_TABLES: &[&str] = &[
    "feedback_events",
    "judge_call_ledger",
    "consumer_offsets",
    "metadata",                  // adaptive_state lives inside metadata table
    "concept_summary_instances", // judge cache TTL reaper prunes Cap A retention rows
];

/// J1 — verify a candidate SQL target table is in the allow-list. Pure
/// function; the worker calls this before every INSERT/UPDATE.
pub fn no_memory_writes(table: &str) -> Result<(), JudgeViolation> {
    if J1_ALLOWED_WRITE_TABLES.contains(&table) {
        Ok(())
    } else {
        Err(JudgeViolation::NoMemoryWrites {
            table: table.to_string(),
        })
    }
}

/// J3 — when κ vs `ExplicitThumb` is **defined** (`pair_count ≥
/// [`LLM_JUDGE_J3_MIN_PAIRS`]`) AND κ < `[`LLM_JUDGE_KAPPA_FLOOR`]`,
/// worker MUST NOT raise `sample_rate_warm`.
///
/// When κ is undefined (insufficient pairs — the entire reason E direction
/// exists), J3 is dormant: the runtime judge runs unconstrained at
/// cold-start sample rate. This is the defensible policy: J3 protects
/// against a calibrated drift signal, not against the absence of data.
pub fn no_self_reinforce(ctx: &JudgeContext) -> Result<(), JudgeViolation> {
    if !ctx.raising_sample_rate {
        return Ok(());
    }
    if ctx.surface_kappa_pair_count < LLM_JUDGE_J3_MIN_PAIRS {
        // J3 dormant — caller will see "κ undefined, J3 dormant" surface
        // via doctor (a one-line operator info, not warn).
        return Ok(());
    }
    if ctx.surface_kappa < LLM_JUDGE_KAPPA_FLOOR {
        return Err(JudgeViolation::SelfReinforce {
            kappa: ctx.surface_kappa,
            floor: LLM_JUDGE_KAPPA_FLOOR,
            pairs: ctx.surface_kappa_pair_count,
        });
    }
    Ok(())
}

/// J5 — judge event payload MUST link back to the source via
/// `synthesis_id` / `concept_summary_id`. Orphan events (link target
/// missing) MUST NOT be emitted. The worker resolves the link by checking
/// either the runtime synthesis cache (Cap B) or
/// `concept_summary_instances` (Cap A); this fn validates the surface_id
/// is at least non-empty before the worker checks the durable table.
pub fn link_present(payload: &JudgePayload<'_>) -> Result<(), JudgeViolation> {
    let id = payload.surface_id();
    if id.is_empty() {
        return Err(JudgeViolation::LinkAbsent {
            surface_id: id.to_string(),
        });
    }
    Ok(())
}

/// J6 — `weight_decay_rate ∈ [0.0, 1.0]` AND finite. Caught at config
/// validation time; this fn exists for runtime defense-in-depth.
pub fn weight_decay_valid(value: f64) -> Result<(), JudgeViolation> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(JudgeViolation::WeightDecayInvalid { value });
    }
    Ok(())
}

/// J7 — payload `stamp_hash` MUST match the worker's computed hash over
/// the post-truncation prompt bytes the judge actually saw. Lets the
/// nightly cron re-judge byte-identical input and detect drift; mismatch
/// means the worker accidentally re-truncated or re-queried source bytes.
pub fn stamp_hash_matches(
    ctx: &JudgeContext<'_>,
    payload: &JudgePayload<'_>,
) -> Result<(), JudgeViolation> {
    if ctx.computed_stamp_hash == payload.stamp_hash() {
        Ok(())
    } else {
        Err(JudgeViolation::StampHashMismatch {
            payload_hash: payload.stamp_hash().to_string(),
            computed_hash: ctx.computed_stamp_hash.to_string(),
        })
    }
}

/// Validate ALL invariants applicable BEFORE the LLM HTTP call. Worker
/// runs this on each job pull; violation → drop with `tracing::warn!`,
/// no event emitted. J2 (call cap) is enforced by [`reserve_call`]
/// directly — it's the one invariant that requires DB state and is
/// inherently atomic.
pub fn validate_pre_emit(
    ctx: &JudgeContext<'_>,
    payload: &JudgePayload<'_>,
) -> Result<(), JudgeViolation> {
    link_present(payload)?;
    stamp_hash_matches(ctx, payload)?;
    Ok(())
}

/// v0.27.1 E direction (spec §4 J2 + R8 P1) — atomic call-cap reservation
/// token returned by [`reserve_call`].
///
/// Drop the token (or call [`ReservationToken::commit`] /
/// [`ReservationToken::fail`]) to transition the ledger row to a terminal
/// state. Stale `reserved` rows older than [`LLM_JUDGE_STALE_CLAIM_SECS`]
/// are reaped by the worker on each pull.
#[derive(Debug)]
pub struct ReservationToken {
    id: String,
}

impl ReservationToken {
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Mark the reservation as `done` after a successful HTTP call.
    pub fn commit(&self, conn: &Connection) -> ReinResult<()> {
        conn.execute(
            "UPDATE judge_call_ledger SET status = 'done' WHERE id = ?1 AND status = 'reserved'",
            rusqlite::params![&self.id],
        )?;
        Ok(())
    }

    /// Mark the reservation as `failed` after the HTTP call errored.
    /// The reservation still counts toward the rolling 24h cap (we
    /// already paid the upstream-rate-limit cost); the row is preserved
    /// so doctor can surface judge-error rates.
    pub fn fail(&self, conn: &Connection) -> ReinResult<()> {
        conn.execute(
            "UPDATE judge_call_ledger SET status = 'failed' WHERE id = ?1 AND status = 'reserved'",
            rusqlite::params![&self.id],
        )?;
        Ok(())
    }
}

/// J2 — atomic call-cap reservation. Runs `INSERT INTO judge_call_ledger
/// (id, ts, status='reserved') WHERE rolling_count < cap` in a single
/// `BEGIN IMMEDIATE` so two dispatchers can't both observe the same
/// below-cap count and burst N×cap calls (Codex R6 P2 fix).
///
/// **Reaps stale `reserved` rows** older than [`LLM_JUDGE_STALE_CLAIM_SECS`]
/// inside the same transaction so worker-crash recovery doesn't leave the
/// cap permanently saturated.
///
/// Returns:
/// - `Ok(Some(token))` — reservation acquired; caller MUST call `.commit()`
///   or `.fail()` on the token after the LLM HTTP call lands
/// - `Ok(None)` — cap reached, worker drops the job
/// - `Err(...)` — DB error (worker logs + drops)
pub fn reserve_call(conn: &Connection, daily_cap: u64) -> ReinResult<Option<ReservationToken>> {
    let now_ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .map_err(|e| ReinError::Config(format!("system time before unix epoch: {e}")))?;
    let window_start = now_ts.saturating_sub(24 * 3600);
    let stale_cutoff = now_ts.saturating_sub(LLM_JUDGE_STALE_CLAIM_SECS);

    // Generate ULID-shaped id outside the txn (cheap; pure compute).
    let id = format!("jl_{}", ulid::Ulid::new());

    let token = (|| -> ReinResult<Option<ReservationToken>> {
        conn.execute("BEGIN IMMEDIATE", [])?;
        // Reap stale `reserved` rows first so they don't pin the cap.
        // Codex R4 P2 fix — distinguish stale-reaped (no HTTP call ever
        // made) from genuine LLM/contract failures (HTTP call made,
        // billed). Reaped rows get status='stale' so they're excluded
        // from the cap count; `failed` is reserved for ledger rows
        // that DID make an HTTP attempt.
        conn.execute(
            "UPDATE judge_call_ledger SET status = 'stale' \
             WHERE status = 'reserved' AND ts < ?1",
            rusqlite::params![stale_cutoff],
        )?;
        // Rolling count: count rows representing actual or in-flight
        // billable calls. `done`/`failed` both represent HTTP attempts
        // (succeeded or LLM error after the call was made). `reserved`
        // is in-flight. `stale` is reaped never-attempted reservation —
        // excluded.
        let count: u64 = conn.query_row(
            "SELECT COUNT(*) FROM judge_call_ledger \
             WHERE ts >= ?1 AND status IN ('reserved', 'done', 'failed')",
            rusqlite::params![window_start],
            |row| row.get::<_, i64>(0),
        )? as u64;
        if count >= daily_cap {
            tracing::warn!(
                current = count,
                cap = daily_cap,
                "judge_call_ledger: J2 cap reached, dropping job"
            );
            conn.execute("COMMIT", [])?;
            return Ok(None);
        }
        conn.execute(
            "INSERT INTO judge_call_ledger (id, ts, status) VALUES (?1, ?2, 'reserved')",
            rusqlite::params![&id, now_ts],
        )?;
        conn.execute("COMMIT", [])?;
        Ok(Some(ReservationToken { id: id.clone() }))
    })();

    if token.is_err() {
        let _ = conn.execute("ROLLBACK", []);
    }
    token
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::adaptive::JudgeSource;

    fn make_synth(synthesis_id: &str, stamp: &str) -> SynthesisLlmJudgePayload {
        SynthesisLlmJudgePayload {
            synthesis_id: synthesis_id.to_string(),
            judge_model: "gemini-3.1-flash-lite-preview".to_string(),
            hit: true,
            reason: "test".to_string(),
            stamp_hash: stamp.to_string(),
            source: JudgeSource::AutoSampled,
            metadata: None,
            signal_hint: None,
        }
    }

    fn ctx_for(stamp: &str, kappa_pairs: usize, kappa: f64, raising: bool) -> JudgeContext<'_> {
        JudgeContext {
            stamp_time_source: "stub source",
            computed_stamp_hash: stamp,
            surface_kappa_pair_count: kappa_pairs,
            surface_kappa: kappa,
            raising_sample_rate: raising,
        }
    }

    #[test]
    fn j1_allows_listed_tables() {
        for &t in J1_ALLOWED_WRITE_TABLES {
            assert!(no_memory_writes(t).is_ok());
        }
    }

    #[test]
    fn j1_blocks_unlisted_table() {
        let v = no_memory_writes("memories").unwrap_err();
        assert!(matches!(v, JudgeViolation::NoMemoryWrites { .. }));
    }

    #[test]
    fn j3_dormant_below_min_pairs() {
        let ctx = ctx_for("h", LLM_JUDGE_J3_MIN_PAIRS - 1, 0.1, true);
        assert!(no_self_reinforce(&ctx).is_ok());
    }

    #[test]
    fn j3_blocks_below_floor_when_defined() {
        let ctx = ctx_for(
            "h",
            LLM_JUDGE_J3_MIN_PAIRS + 5,
            LLM_JUDGE_KAPPA_FLOOR - 0.1,
            true,
        );
        let err = no_self_reinforce(&ctx).unwrap_err();
        assert!(matches!(err, JudgeViolation::SelfReinforce { .. }));
    }

    #[test]
    fn j3_passes_above_floor() {
        let ctx = ctx_for(
            "h",
            LLM_JUDGE_J3_MIN_PAIRS + 5,
            LLM_JUDGE_KAPPA_FLOOR + 0.1,
            true,
        );
        assert!(no_self_reinforce(&ctx).is_ok());
    }

    #[test]
    fn j3_skips_when_not_raising() {
        // Bad κ but worker isn't raising → invariant inactive.
        let ctx = ctx_for("h", LLM_JUDGE_J3_MIN_PAIRS + 5, 0.0, false);
        assert!(no_self_reinforce(&ctx).is_ok());
    }

    #[test]
    fn j5_rejects_empty_surface_id() {
        let p = make_synth("", "h");
        let payload = JudgePayload::Synthesis(&p);
        let v = link_present(&payload).unwrap_err();
        assert!(matches!(v, JudgeViolation::LinkAbsent { .. }));
    }

    #[test]
    fn j5_accepts_present_surface_id() {
        let p = make_synth("syn123", "h");
        let payload = JudgePayload::Synthesis(&p);
        assert!(link_present(&payload).is_ok());
    }

    #[test]
    fn j6_rejects_out_of_range() {
        assert!(weight_decay_valid(-0.1).is_err());
        assert!(weight_decay_valid(1.1).is_err());
        assert!(weight_decay_valid(f64::NAN).is_err());
        assert!(weight_decay_valid(f64::INFINITY).is_err());
    }

    #[test]
    fn j6_accepts_in_range() {
        assert!(weight_decay_valid(0.0).is_ok());
        assert!(weight_decay_valid(0.3).is_ok());
        assert!(weight_decay_valid(1.0).is_ok());
    }

    #[test]
    fn j7_rejects_hash_mismatch() {
        let p = make_synth("syn123", "expected_hash");
        let payload = JudgePayload::Synthesis(&p);
        let ctx = ctx_for("different_hash", 0, 0.0, false);
        let v = stamp_hash_matches(&ctx, &payload).unwrap_err();
        assert!(matches!(v, JudgeViolation::StampHashMismatch { .. }));
    }

    #[test]
    fn j7_accepts_hash_match() {
        let p = make_synth("syn123", "matching_hash");
        let payload = JudgePayload::Synthesis(&p);
        let ctx = ctx_for("matching_hash", 0, 0.0, false);
        assert!(stamp_hash_matches(&ctx, &payload).is_ok());
    }

    #[test]
    fn validate_pre_emit_chains_invariants() {
        let p = make_synth("syn123", "h");
        let payload = JudgePayload::Synthesis(&p);
        let ctx = ctx_for("h", 0, 0.0, false);
        assert!(validate_pre_emit(&ctx, &payload).is_ok());
    }

    #[test]
    fn zero_human_pairs_failed_structural_anchor_blocks_trust_increase() {
        let evidence = JudgeCalibrationEvidence {
            human_pair_count: 0,
            human_kappa: None,
            structural_status: JudgeStructuralStatus::Failed,
            enforce_structural_anchors: true,
        };

        let decision = judge_trust_gate(&evidence, JudgeTrustAction::IncreaseJudgeWeight);

        assert!(!decision.action_allowed);
        assert_eq!(decision.configured_baseline_scale, 0.0);
        assert!(decision.blocks_release);
    }

    const ALL_TRUST_ACTIONS: [JudgeTrustAction; 5] = [
        JudgeTrustAction::KeepConfiguredBaseline,
        JudgeTrustAction::IncreaseJudgeWeight,
        JudgeTrustAction::IncreaseSampleRate,
        JudgeTrustAction::PromoteJudgeScope,
        JudgeTrustAction::PromoteRecallFusion,
    ];

    #[test]
    fn insufficient_human_structural_status_action_matrix() {
        use JudgeCalibrationBasis::{ConfiguredBaseline, StructuralAnchors, Untrusted};
        use JudgeStructuralStatus::{
            Collecting, Corrupt, Disabled, Failed, FingerprintMismatch, Ready, Stale, Unknown,
        };
        use JudgeTrustReason::{
            ConfiguredBaselineFallback, StructuralAnchorsReady, StructuralHealthEnforced,
        };

        let cases = [
            (
                Disabled,
                false,
                ConfiguredBaseline,
                1.0,
                false,
                ConfiguredBaselineFallback,
            ),
            (
                Disabled,
                true,
                ConfiguredBaseline,
                1.0,
                false,
                ConfiguredBaselineFallback,
            ),
            (
                Collecting,
                false,
                ConfiguredBaseline,
                1.0,
                false,
                ConfiguredBaselineFallback,
            ),
            (
                Collecting,
                true,
                ConfiguredBaseline,
                1.0,
                false,
                ConfiguredBaselineFallback,
            ),
            (
                Ready,
                false,
                StructuralAnchors,
                1.0,
                false,
                StructuralAnchorsReady,
            ),
            (
                Ready,
                true,
                StructuralAnchors,
                1.0,
                false,
                StructuralAnchorsReady,
            ),
            (
                Failed,
                false,
                ConfiguredBaseline,
                1.0,
                false,
                ConfiguredBaselineFallback,
            ),
            (Failed, true, Untrusted, 0.0, true, StructuralHealthEnforced),
            (
                Stale,
                false,
                ConfiguredBaseline,
                1.0,
                false,
                ConfiguredBaselineFallback,
            ),
            (Stale, true, Untrusted, 0.0, true, StructuralHealthEnforced),
            (
                FingerprintMismatch,
                false,
                ConfiguredBaseline,
                1.0,
                false,
                ConfiguredBaselineFallback,
            ),
            (
                FingerprintMismatch,
                true,
                Untrusted,
                0.0,
                true,
                StructuralHealthEnforced,
            ),
            (
                Corrupt,
                false,
                ConfiguredBaseline,
                1.0,
                false,
                ConfiguredBaselineFallback,
            ),
            (
                Corrupt,
                true,
                Untrusted,
                0.0,
                true,
                StructuralHealthEnforced,
            ),
            (
                Unknown,
                false,
                ConfiguredBaseline,
                1.0,
                false,
                ConfiguredBaselineFallback,
            ),
            (
                Unknown,
                true,
                Untrusted,
                0.0,
                true,
                StructuralHealthEnforced,
            ),
        ];

        for (status, enforce, basis, baseline_scale, blocks_release, reason) in cases {
            let evidence = JudgeCalibrationEvidence {
                human_pair_count: 0,
                human_kappa: None,
                structural_status: status,
                enforce_structural_anchors: enforce,
            };

            for action in ALL_TRUST_ACTIONS {
                let decision = judge_trust_gate(&evidence, action);
                assert_eq!(
                    decision.basis, basis,
                    "status={status:?}, action={action:?}"
                );
                assert_eq!(
                    decision.action_allowed,
                    baseline_scale == 1.0 && action == JudgeTrustAction::KeepConfiguredBaseline,
                    "status={status:?}, action={action:?}"
                );
                assert_eq!(
                    decision.configured_baseline_scale, baseline_scale,
                    "status={status:?}, action={action:?}"
                );
                assert_eq!(
                    decision.blocks_release, blocks_release,
                    "status={status:?}, action={action:?}"
                );
                assert_eq!(
                    decision.reason, reason,
                    "status={status:?}, action={action:?}"
                );
            }
        }
    }

    #[test]
    fn healthy_human_kappa_authorizes_all_actions_unless_enforced_structure_fails() {
        let healthy = JudgeCalibrationEvidence {
            human_pair_count: LLM_JUDGE_J3_MIN_PAIRS,
            human_kappa: Some(LLM_JUDGE_KAPPA_FLOOR),
            structural_status: JudgeStructuralStatus::Collecting,
            enforce_structural_anchors: true,
        };

        for action in ALL_TRUST_ACTIONS {
            let decision = judge_trust_gate(&healthy, action);
            assert_eq!(decision.basis, JudgeCalibrationBasis::HumanKappa);
            assert!(decision.action_allowed, "action={action:?}");
            assert_eq!(decision.configured_baseline_scale, 1.0);
            assert!(!decision.blocks_release);
            assert_eq!(decision.reason, JudgeTrustReason::HumanKappaHealthy);
        }

        let failed = JudgeCalibrationEvidence {
            structural_status: JudgeStructuralStatus::Failed,
            ..healthy
        };
        for action in ALL_TRUST_ACTIONS {
            let decision = judge_trust_gate(&failed, action);
            assert_eq!(decision.basis, JudgeCalibrationBasis::Untrusted);
            assert!(!decision.action_allowed, "action={action:?}");
            assert_eq!(decision.configured_baseline_scale, 0.0);
            assert!(decision.blocks_release);
            assert_eq!(decision.reason, JudgeTrustReason::StructuralHealthEnforced);
        }
    }

    #[test]
    fn human_negative_evidence_wins_over_ready_structural_anchors() {
        for kappa in [Some(LLM_JUDGE_KAPPA_FLOOR - 0.01), Some(f64::NAN), None] {
            let evidence = JudgeCalibrationEvidence {
                human_pair_count: LLM_JUDGE_J3_MIN_PAIRS,
                human_kappa: kappa,
                structural_status: JudgeStructuralStatus::Ready,
                enforce_structural_anchors: true,
            };

            for action in ALL_TRUST_ACTIONS {
                let decision = judge_trust_gate(&evidence, action);
                assert_eq!(decision.basis, JudgeCalibrationBasis::Untrusted);
                assert!(
                    !decision.action_allowed,
                    "kappa={kappa:?}, action={action:?}"
                );
                assert_eq!(decision.configured_baseline_scale, 0.0);
                assert!(decision.blocks_release);
                let expected_reason = if kappa.is_some_and(f64::is_finite) {
                    JudgeTrustReason::HumanKappaBelowFloor
                } else {
                    JudgeTrustReason::HumanKappaMissingOrInvalid
                };
                assert_eq!(decision.reason, expected_reason);
            }
        }
    }

    #[test]
    fn authoritative_human_kappa_outside_theoretical_range_fails_closed() {
        for kappa in [1.000_1, -1.000_1] {
            let evidence = JudgeCalibrationEvidence {
                human_pair_count: LLM_JUDGE_J3_MIN_PAIRS,
                human_kappa: Some(kappa),
                structural_status: JudgeStructuralStatus::Ready,
                enforce_structural_anchors: true,
            };

            for action in ALL_TRUST_ACTIONS {
                let decision = judge_trust_gate(&evidence, action);
                assert_eq!(decision.basis, JudgeCalibrationBasis::Untrusted);
                assert!(!decision.action_allowed);
                assert_eq!(decision.configured_baseline_scale, 0.0);
                assert!(decision.blocks_release);
                assert_eq!(
                    decision.reason,
                    JudgeTrustReason::HumanKappaMissingOrInvalid
                );
            }
        }
    }

    #[test]
    fn human_kappa_below_minimum_pair_count_is_not_authoritative() {
        let evidence = JudgeCalibrationEvidence {
            human_pair_count: LLM_JUDGE_J3_MIN_PAIRS - 1,
            human_kappa: Some(0.0),
            structural_status: JudgeStructuralStatus::Ready,
            enforce_structural_anchors: true,
        };

        let baseline = judge_trust_gate(&evidence, JudgeTrustAction::KeepConfiguredBaseline);
        assert_eq!(baseline.basis, JudgeCalibrationBasis::StructuralAnchors);
        assert!(baseline.action_allowed);
        assert_eq!(baseline.configured_baseline_scale, 1.0);

        let increase = judge_trust_gate(&evidence, JudgeTrustAction::IncreaseSampleRate);
        assert_eq!(increase.basis, JudgeCalibrationBasis::StructuralAnchors);
        assert!(!increase.action_allowed);
        assert_eq!(increase.configured_baseline_scale, 1.0);
    }

    fn setup_ledger_db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        crate::store::schema::migrate_judge_call_ledger(&conn).unwrap();
        conn
    }

    #[test]
    fn reserve_call_returns_token_when_below_cap() {
        let conn = setup_ledger_db();
        let token = reserve_call(&conn, 100).unwrap();
        assert!(token.is_some());
    }

    #[test]
    fn reserve_call_returns_none_when_cap_reached() {
        let conn = setup_ledger_db();
        // Fill to cap.
        for _ in 0..3 {
            let token = reserve_call(&conn, 3).unwrap().expect("reserved");
            token.commit(&conn).unwrap();
        }
        let next = reserve_call(&conn, 3).unwrap();
        assert!(next.is_none());
    }

    #[test]
    fn reserve_call_token_commit_and_fail_terminal() {
        let conn = setup_ledger_db();
        let token = reserve_call(&conn, 100).unwrap().expect("reserved");
        token.commit(&conn).unwrap();
        let status: String = conn
            .query_row(
                "SELECT status FROM judge_call_ledger WHERE id = ?1",
                rusqlite::params![token.id()],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(status, "done");

        let token2 = reserve_call(&conn, 100).unwrap().expect("reserved");
        token2.fail(&conn).unwrap();
        let status2: String = conn
            .query_row(
                "SELECT status FROM judge_call_ledger WHERE id = ?1",
                rusqlite::params![token2.id()],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(status2, "failed");
    }
}
