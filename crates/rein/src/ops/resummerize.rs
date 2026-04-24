//! v0.23 resummerize slow-channel op.
//!
//! Replaces the keep-tail truncation fallback with LLM-driven canonical
//! recompression, gated by the Lossless Compression Contract
//! (`crate::compression::contract`). Safe-by-default: any failure in the
//! pipeline leaves the canonical unchanged, so keep-tail remains the
//! effective state. Triggered rows carry `memories.needs_resummerize = 1`
//! which the `MergeInto` cap branch sets on every cap hit.
//!
//! Threading: the outer `run_resummerize` is sync so it composes with the
//! existing sync ops surface. The LLM call is async and uses the
//! `block_in_place + Handle::current().block_on` pattern established by
//! `ops/dedup.rs`; see `call_llm_sync` below.
//!
//! Contract gate: output is rewritten only when `contract::check_all`
//! returns `Ok(())`. Three consecutive failures on the same canonical
//! clear the flag (status = `Exhausted`) so a persistently broken case
//! doesn't loop forever against the LLM.

use crate::compression::contract::{self, ContractInput, EvidenceEntry};
use crate::config::{Provider, ReinConfig};
use crate::extract::llm::{strip_code_fences, ExtractorKind};
use crate::store::adaptive::AdaptiveState;
use crate::store::resummerize_audit::{self, ResummerizeRunRow, ResummerizeRunStatus};
use crate::store::SqliteStore;
use crate::types::traits::MemoryStore;
use crate::types::{ReinError, ReinResult, SUMMARY_MAX_CHARS};
use chrono::Utc;
use rusqlite::OptionalExtension;
use sha2::{Digest, Sha256};

/// Clear the `needs_resummerize` flag after this many consecutive
/// non-success runs for the same canonical. Prevents infinite LLM spam on
/// cases the model structurally can't satisfy.
const MAX_CONSECUTIVE_FAILURES: usize = 3;

/// A claim older than this is considered stale and may be taken by
/// another worker. This is a **failure-recovery floor**, not a tunable —
/// an LLM call completing in 5 minutes is already an anomaly (typical is
/// 1–3 s), and anything slower is a worker that probably died. Codex H6.
const STALE_CLAIM_TIMEOUT_SECS: i64 = 300;

/// System prompt for the LLM compression call.
///
/// Kept deliberately short: the Lossless Compression Contract is the source
/// of truth for "what the output must preserve". The prompt only needs to
/// steer the model toward the same goals. Contract violations are caught
/// post-hoc; the prompt is a hint, not a guarantee.
/// Stays private: the eval harness uses `call_llm_sync` as its entry point,
/// and `call_llm_sync` embeds `SYSTEM_PROMPT` internally, so production and
/// eval naturally share this string without a public export. McNemar
/// comparability is guaranteed as long as `call_llm_sync` is the sole
/// caller.
const SYSTEM_PROMPT: &str = "You are a canonical memory compression engine. \
Given a current canonical text plus its merge evidence, emit a shorter \
canonical replacement that preserves every distinct fact, every date and \
version anchor, every CJK character, and every fenced code block. Never \
silently resolve contradictions — keep both sides. Fit within the target \
byte budget provided. Output only the new canonical text, nothing else: \
no preamble, no explanation, no code fences wrapping the whole answer.";

// ── Public surface ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct ResummerizeOutcome {
    pub attempted: u32,
    pub succeeded: u32,
    pub contract_failed: u32,
    pub llm_failed: u32,
    pub exhausted: u32,
    pub length_exceeded: u32,
    /// Count of canonicals where our claim was lost to another worker
    /// between the LLM call and commit (stale-claim recovery reassigned
    /// the row). Codex round-2 HIGH: these outcomes are silent w.r.t.
    /// audit — the successful worker's record is the truth — but the
    /// count lets operators see claim-loss frequency in the returned
    /// stats.
    pub claim_lost: u32,
    pub skipped_no_llm: bool,
    pub skipped_disabled: bool,
    pub dry_run: bool,
}

/// Run resummerize over a batch of canonicals flagged with
/// `needs_resummerize = 1`. If `canonical_id` is provided, only that one
/// is processed (subject to the same gates as a batch member).
pub fn run_resummerize(
    store: &SqliteStore,
    config: &ReinConfig,
    canonical_id: Option<&str>,
    dry_run: bool,
) -> ReinResult<ResummerizeOutcome> {
    run_resummerize_inner(store, config, canonical_id, dry_run, None)
}

/// Test-only entry point that bypasses `create_resummerize_extractor` so
/// integration tests can inject a `MockExtractor` and exercise the real
/// `apply_resummerize` / contract-gate / claim / audit paths end-to-end.
///
/// Available only under the `test-support` feature (which `dev-dependencies`
/// activates automatically for integration tests).
#[cfg(feature = "test-support")]
pub fn run_resummerize_with_extractor(
    store: &SqliteStore,
    config: &ReinConfig,
    canonical_id: Option<&str>,
    dry_run: bool,
    extractor: ExtractorKind,
) -> ReinResult<ResummerizeOutcome> {
    run_resummerize_inner(store, config, canonical_id, dry_run, Some(extractor))
}

fn run_resummerize_inner(
    store: &SqliteStore,
    config: &ReinConfig,
    canonical_id: Option<&str>,
    dry_run: bool,
    extractor_override: Option<ExtractorKind>,
) -> ReinResult<ResummerizeOutcome> {
    let mut outcome = ResummerizeOutcome {
        dry_run,
        ..Default::default()
    };

    if !config.resummerize.enabled {
        outcome.skipped_disabled = true;
        return Ok(outcome);
    }

    // Dry-run: report backlog depth without claiming anything or calling
    // the LLM. A preview must be safe to run even without API keys
    // configured (Codex H1).
    if dry_run {
        let eligible = preview_eligible(store, canonical_id)?;
        outcome.attempted = eligible.len() as u32;
        return Ok(outcome);
    }

    let extractor = match extractor_override {
        Some(e) => e,
        None => match create_resummerize_extractor(config) {
            Some(e) => e,
            None => {
                outcome.skipped_no_llm = true;
                return Ok(outcome);
            }
        },
    };

    let state = AdaptiveState::restore_snapshot(store.conn()).unwrap_or_default();

    // Atomically claim a batch so concurrent workers don't pick the same
    // rows (Codex H6). `claim_batch` returns only rows whose
    // `in_progress_resummerize_at` was NULL or stale at claim time; those
    // rows are now marked as in-progress until `release_claim` or the
    // success-path UPDATE in `apply_resummerize` clears the marker.
    let claims = claim_batch(store, canonical_id, config.resummerize.batch_size)?;

    for claim in claims {
        outcome.attempted += 1;
        match resummerize_one(store, config, &extractor, &state, &claim) {
            Ok(Verdict::Success) => outcome.succeeded += 1,
            Ok(Verdict::ContractViolation) => outcome.contract_failed += 1,
            Ok(Verdict::LlmError) => outcome.llm_failed += 1,
            Ok(Verdict::Exhausted) => outcome.exhausted += 1,
            Ok(Verdict::LengthExceeded) => outcome.length_exceeded += 1,
            Ok(Verdict::ClaimLost) => outcome.claim_lost += 1,
            Err(e) => {
                tracing::warn!(
                    canonical_id = %claim.canonical_id,
                    error = %e,
                    "resummerize one failed"
                );
            }
        }
    }

    Ok(outcome)
}

// ── Implementation ───────────────────────────────────────────────────────────

enum Verdict {
    Success,
    ContractViolation,
    LlmError,
    LengthExceeded,
    Exhausted,
    /// Claim was taken by another worker (stale timeout elapsed, a fresh
    /// worker reclaimed, and our ownership predicate no longer matches at
    /// commit time). Rewrite was rolled back, no audit row written, no
    /// canonical mutation — the other worker's output stands. Codex
    /// round-2 HIGH.
    ClaimLost,
}

/// Canonical id + the RFC3339 timestamp we stamped into
/// `in_progress_resummerize_at` when claiming. The token is carried end
/// to end and re-checked at commit time (Codex round-2 HIGH) so a stale
/// worker whose claim has been reassigned can't overwrite the newer
/// owner's rewrite. Lexicographic comparison of RFC3339 strings is
/// time-ordered because we always emit `YYYY-MM-DDTHH:MM:SS.fffZ` via
/// `chrono::Utc::now().to_rfc3339()`.
#[derive(Debug, Clone)]
struct Claim {
    canonical_id: String,
    token: String,
}

/// Build the LLM extractor honoring `[resummerize].llm_backend`.
///
/// Previously this delegated to `create_extractor(config)`, which only
/// ever reads `[extract].provider`. The resulting behavior was that
/// `llm_backend = "omlx"` with `extract.provider = "google"` silently
/// used Gemini — the override was ignored (Codex audit M7). Now we build
/// the extractor directly from the resolved resummerize provider so the
/// override actually takes effect.
/// `pub` so `bin/rein_eval.rs::cmd_run` can reuse the same backend-selection
/// path as production. Pre-fix the eval used `create_extractor` (which
/// follows `[extract].provider`) while production used
/// `create_resummerize_extractor` (which follows
/// `[resummerize].llm_backend`). If an operator configured
/// `extract.provider = "google"` with
/// `resummerize.llm_backend = "omlx"`, the eval scorecard would have
/// tested Gemini while production ran OMLX — the `compare` verdict is
/// meaningless against a different backend. Post-fix audit M-1.
/// **Not a stable public API.**
pub fn create_resummerize_extractor(config: &ReinConfig) -> Option<ExtractorKind> {
    let extract_provider = config.extract_provider();
    match config.resummerize.resolved_provider(extract_provider) {
        Provider::None => None,
        Provider::Google => {
            let api_key = config.extract.google.api_key.as_ref()?.clone();
            Some(ExtractorKind::Gemini(
                crate::extract::llm::GeminiExtractor::new(
                    api_key,
                    config.extract.google.endpoint.clone(),
                    config.extract.google.model.clone(),
                ),
            ))
        }
        Provider::Omlx => Some(ExtractorKind::Omlx(
            crate::extract::llm::OmlxExtractor::new(
                config.extract.omlx.endpoint.clone(),
                config.extract.omlx.model.clone(),
                config.extract.omlx.disable_thinking,
            ),
        )),
    }
}

/// SQL predicate shared by every eligibility check (dry-run preview, claim,
/// backlog).
///
/// A row is eligible for resummerize iff it is:
/// 1. flagged (`needs_resummerize = 1`)
/// 2. a live canonical — `status IN ('active', 'updated')`. Both states
///    represent a reachable live canonical: `active` = the row was just
///    created / self-canonical, `updated` = the row has been through the
///    merge `update()` path (which auto-promotes the status from active
///    to updated via trigger). Before the round-5 H-1 fix this was
///    `status = 'active'`, which silently excluded every merge-capped
///    canonical — the exact rows the resummerize op exists to handle.
/// 3. not superseded (`superseded_by IS NULL`)
/// 4. its own canonical (`memory_canonical_state.canonical_id = id`)
/// 5. not currently claimed by another live worker — meaning
///    `in_progress_resummerize_at` is either NULL or older than
///    `STALE_CLAIM_TIMEOUT_SECS` (Codex H6 stale-claim recovery).
///
/// The stale threshold is compared against the `?stale_cutoff` bind
/// parameter supplied by the caller (must be present for any query using
/// this predicate).
const ELIGIBILITY_PREDICATE: &str = "\
    m.needs_resummerize = 1 \
    AND m.status IN ('active', 'updated') \
    AND m.superseded_by IS NULL \
    AND EXISTS ( \
        SELECT 1 FROM memory_canonical_state cs \
        WHERE cs.memory_id = m.id AND cs.canonical_id = m.id \
    ) \
    AND ( \
        m.in_progress_resummerize_at IS NULL \
        OR m.in_progress_resummerize_at < ?stale \
    )\
";

fn stale_cutoff_rfc3339() -> String {
    (Utc::now() - chrono::Duration::seconds(STALE_CLAIM_TIMEOUT_SECS)).to_rfc3339()
}

/// Dry-run variant of batch selection: report which rows *would* be
/// processed without mutating `in_progress_resummerize_at`. Used only by
/// the preview path.
fn preview_eligible(store: &SqliteStore, only: Option<&str>) -> ReinResult<Vec<String>> {
    let stale = stale_cutoff_rfc3339();
    // Swap the named `?stale` marker for a positional `?1` to match what
    // rusqlite's numbered-param binding expects. (rusqlite does not
    // support named parameters on the public prepare path.)
    let predicate = ELIGIBILITY_PREDICATE.replace("?stale", "?1");

    if let Some(id) = only {
        let sql = format!("SELECT m.id FROM memories m WHERE {predicate} AND m.id = ?2",);
        let mut stmt = store.conn().prepare(&sql)?;
        let hit: Option<String> = stmt
            .query_row(rusqlite::params![&stale, id], |row| row.get(0))
            .optional()?;
        return Ok(hit.into_iter().collect());
    }
    let sql = format!(
        "SELECT m.id FROM memories m WHERE {predicate} \
         ORDER BY COALESCE(m.updated_at, m.created_at) ASC",
    );
    let mut stmt = store.conn().prepare(&sql)?;
    let rows = stmt.query_map(rusqlite::params![&stale], |row| row.get::<_, String>(0))?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(|e| e.into())
}

/// Atomically claim up to `batch_size` eligible rows (or the single `only`
/// id) by setting `in_progress_resummerize_at = now`. Returns the ids that
/// were successfully claimed — never more than `batch_size`, and never
/// rows that another live worker holds a fresh claim on.
///
/// The UPDATE-IN-subquery form is the SQLite idiom for "ORDER BY + LIMIT +
/// atomic visibility": the subquery picks candidates under the current
/// snapshot, the outer UPDATE claims them with RETURNING so the caller
/// knows exactly which ids are live. Two concurrent workers each get a
/// disjoint slice; the loser for any specific row gets an empty
/// RETURNING row (SQLite serializes writes).
fn claim_batch(
    store: &SqliteStore,
    only: Option<&str>,
    batch_size: usize,
) -> ReinResult<Vec<Claim>> {
    let now = Utc::now().to_rfc3339();
    let stale = stale_cutoff_rfc3339();
    let predicate_subq = ELIGIBILITY_PREDICATE.replace("?stale", "?2");

    if let Some(id) = only {
        // Single-id claim via subquery — matches the batch path shape and
        // avoids SQLite's rejection of `UPDATE memories AS m` as the outer
        // statement form (aliases on the target table are only accepted
        // inside UPDATE...FROM since 3.33, not on the bare UPDATE...SET).
        //
        // Params: ?1 = now, ?2 = stale_cutoff, ?3 = id.
        let sql = format!(
            "UPDATE memories \
             SET in_progress_resummerize_at = ?1 \
             WHERE id IN ( \
                 SELECT m.id FROM memories m \
                 WHERE m.id = ?3 AND {predicate_subq} \
             ) \
             RETURNING id",
        );
        let mut stmt = store.conn().prepare(&sql)?;
        let hit: Option<String> = stmt
            .query_row(rusqlite::params![&now, &stale, id], |row| row.get(0))
            .optional()?;
        return match hit {
            Some(canonical_id) => Ok(vec![Claim {
                canonical_id,
                token: now.clone(),
            }]),
            None => {
                tracing::info!(
                    canonical_id = %id,
                    "resummerize: targeted canonical_id not eligible \
                     (not flagged, superseded, non-canonical, deactivated, \
                     or currently claimed by another worker); skipping"
                );
                Ok(Vec::new())
            }
        };
    }

    let limit = batch_size.max(1) as i64;
    // Batch claim: ?1 = now, ?2 = stale_cutoff, ?3 = limit.
    let sql = format!(
        "UPDATE memories \
         SET in_progress_resummerize_at = ?1 \
         WHERE id IN ( \
             SELECT m.id FROM memories m \
             WHERE {predicate_subq} \
             ORDER BY COALESCE(m.updated_at, m.created_at) ASC \
             LIMIT ?3 \
         ) \
         RETURNING id",
    );
    let mut stmt = store.conn().prepare(&sql)?;
    let rows = stmt.query_map(rusqlite::params![&now, &stale, limit], |row| {
        row.get::<_, String>(0)
    })?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map(|ids| {
            ids.into_iter()
                .map(|canonical_id| Claim {
                    canonical_id,
                    token: now.clone(),
                })
                .collect()
        })
        .map_err(|e| e.into())
}

/// Release a claim on a canonical, **only if we still own it** (the
/// timestamp we stamped at claim time is still on the row). If a stale
/// timeout elapsed and another worker reclaimed in the meantime, this
/// becomes a no-op instead of clobbering the new owner's marker.
/// Codex round-2 HIGH.
fn release_claim(store: &SqliteStore, canonical_id: &str, token: &str) -> ReinResult<()> {
    store.conn().execute(
        "UPDATE memories \
         SET in_progress_resummerize_at = NULL \
         WHERE id = ?1 AND in_progress_resummerize_at = ?2",
        rusqlite::params![canonical_id, token],
    )?;
    Ok(())
}

/// Execute one canonical resummerize end-to-end.
///
/// Invariant: only called from the non-dry-run path of `run_resummerize`
/// **after** `claim_batch` has already set `in_progress_resummerize_at`.
/// This wrapper always releases the claim on return (idempotent — the
/// success-path UPDATE in `apply_resummerize` may have already nulled it).
fn resummerize_one(
    store: &SqliteStore,
    config: &ReinConfig,
    extractor: &ExtractorKind,
    state: &AdaptiveState,
    claim: &Claim,
) -> ReinResult<Verdict> {
    let result = resummerize_one_inner(store, config, extractor, state, claim);
    // Release the claim on exit **only if we still own it** (Codex round-2
    // HIGH): if the stale timeout elapsed and another worker reclaimed
    // the row, we must not null their marker. `release_claim` is
    // idempotent — on the Success path `apply_resummerize` already nulled
    // the marker with the same token predicate, so this is a 0-row UPDATE.
    if let Err(e) = release_claim(store, &claim.canonical_id, &claim.token) {
        tracing::warn!(
            canonical_id = %claim.canonical_id,
            error = %e,
            "failed to release resummerize claim; will be reclaimed after stale timeout"
        );
    }
    result
}

fn resummerize_one_inner(
    store: &SqliteStore,
    _config: &ReinConfig,
    extractor: &ExtractorKind,
    state: &AdaptiveState,
    claim: &Claim,
) -> ReinResult<Verdict> {
    let canonical_id = claim.canonical_id.as_str();

    // Codex round-6 MEDIUM: capture the canonical row, its raw
    // `updated_at` text, and its merge-evidence history inside a single
    // `BEGIN DEFERRED` snapshot so a concurrent writer can't leak
    // between the three reads. Without this, the pre-round-6 code had a
    // race where `store.get()` returned OLD content and a subsequent
    // `SELECT updated_at` returned a NEWER value written in between —
    // the LLM would compress the old content while the CAS CAS'd
    // against the new updated_at, and the post-LLM apply would
    // overwrite the newer state with stale-compressed output.
    store.conn().execute_batch("BEGIN DEFERRED")?;
    let canonical_result = store.get(canonical_id);
    let raw_result: rusqlite::Result<String> = store.conn().query_row(
        "SELECT updated_at FROM memories WHERE id = ?1",
        rusqlite::params![canonical_id],
        |row| row.get(0),
    );
    // Full evidence history, oldest-first. Prior versions called
    // `list_memory_evidence(id, 1_000)` which silently truncated long
    // histories AND returned newest-first despite the prompt's claim of
    // "oldest first — newer merges appear later" — a long-lived canonical
    // could pass the contract while the LLM never saw the older facts.
    // Agent A adversarial finding A-2 (post-v0.23.0).
    let evidence_result = store.list_all_memory_evidence_oldest_first(canonical_id);
    // COMMIT closes the read lock; deferred txns in WAL mode don't strictly
    // need explicit commit but doing it cleanly releases the snapshot
    // immediately rather than at statement-scope drop. Codex round-7 LOW:
    // log on COMMIT failure and best-effort ROLLBACK so the connection
    // returns to a known state instead of carrying a zombie open
    // transaction into the next write.
    if let Err(commit_err) = store.conn().execute_batch("COMMIT") {
        tracing::warn!(
            canonical_id = %canonical_id,
            error = %commit_err,
            "resummerize snapshot COMMIT failed; attempting ROLLBACK \
             to restore connection state"
        );
        let _ = store.conn().execute_batch("ROLLBACK");
    }

    let canonical = canonical_result?;
    let canonical_updated_at_raw = raw_result.map_err(|e| {
        ReinError::Config(format!(
            "resummerize: failed to read updated_at for CAS: {e}"
        ))
    })?;
    let evidence_raw = evidence_result?;

    // If the canonical is no longer over the cap (keep-tail plus a later
    // delete could leave it below), clear the flag and skip.
    if canonical.content.is_empty() {
        clear_flag(store, canonical_id)?;
        return Ok(Verdict::Success);
    }

    // Consecutive-failure guard. This is a read on `resummarize_runs`
    // (a different table than `memories`), so the race semantics don't
    // require it to share the snapshot above — but it must run AFTER
    // the snapshot so we don't miss an audit row written by a concurrent
    // worker between our claim and our read.
    let prior_failures = resummerize_audit::count_recent_consecutive_failures(
        store.conn(),
        canonical_id,
        MAX_CONSECUTIVE_FAILURES,
    )?;
    if prior_failures >= MAX_CONSECUTIVE_FAILURES {
        record_exhaustion(store, canonical_id, prior_failures)?;
        return Ok(Verdict::Exhausted);
    }
    let evidence: Vec<EvidenceEntry> = evidence_raw
        .iter()
        .map(|e| EvidenceEntry {
            content: e.content.clone(),
            merged_at: e.imported_at,
        })
        .collect();

    let target_bytes = state.resummerize_target_bytes(canonical.cluster_id);
    let input = ContractInput {
        evidence: &evidence,
        current_canonical: &canonical.content,
        target_bytes,
    };

    let prompt = build_prompt(&input);
    let input_canonical_chars = canonical.content.chars().count() as u32;
    let input_evidence_count = evidence.len() as u32;
    let llm_backend = extractor_backend_tag(extractor);

    let run_id = ulid::Ulid::new().to_string();
    resummerize_audit::insert_resummerize_run(
        store.conn(),
        &ResummerizeRunRow::starting(
            run_id.clone(),
            canonical_id.to_string(),
            input_evidence_count,
            input_canonical_chars,
            target_bytes as u32,
            llm_backend.clone(),
            Utc::now(),
        ),
    )?;

    let llm_output = match call_llm_sync(extractor, &prompt) {
        Ok(text) => strip_code_fences(&text),
        Err(e) => {
            tracing::warn!(canonical_id = %canonical_id, error = %e, "resummerize LLM call failed");
            resummerize_audit::finish_resummerize_run(
                store.conn(),
                &run_id,
                None,
                None,
                ResummerizeRunStatus::LlmError,
                &[],
                Some(e.to_string()),
                Utc::now(),
            )?;
            return Ok(Verdict::LlmError);
        }
    };

    // Explicit length gate before the full contract run — gives a
    // distinguishable status rather than burying it in `ContractViolation`.
    // Unit is bytes (matches `MERGE_CONTENT_CAP` + adaptive target_bytes);
    // see Codex H3 reconciliation.
    //
    // Clamp the tolerance at `MERGE_CONTENT_CAP` so an output at
    // `MAX_RESUMMERIZE_TARGET + 10%` (Codex round-2 MEDIUM) can't slip
    // past this gate and then immediately blow the upstream merge cap
    // on the next real merge.
    let tolerance = (target_bytes + target_bytes / 10).min(crate::store::sqlite::MERGE_CONTENT_CAP);
    let output_bytes = llm_output.len();
    if output_bytes > tolerance {
        // Post-audit round-2 MED-1: atomically recheck ownership AND write
        // the terminal status under one `BEGIN IMMEDIATE`. Pre-fix had a
        // window between the recheck SELECT and the `finish_resummerize_run`
        // UPDATE where a peer `MergeInto` could commit and our write would
        // still land as `LengthExceeded` (countable → fuse). IMMEDIATE
        // grabs a reserved lock so no concurrent writer slips in until we
        // commit.
        let verdict = finish_with_ownership_check(
            store,
            &run_id,
            &claim.canonical_id,
            &claim.token,
            &canonical_updated_at_raw,
            output_bytes as u32,
            &sha256_hex(&llm_output),
            ResummerizeRunStatus::LengthExceeded,
            &["length_bounded".to_string()],
            Some(format!(
                "output {} bytes exceeded tolerance {}",
                output_bytes, tolerance,
            )),
            "length check raced with concurrent writer",
        )?;
        return Ok(verdict);
    }

    match contract::check_all(&input, &llm_output) {
        Ok(()) => {
            match apply_resummerize(
                store,
                canonical,
                &llm_output,
                &claim.token,
                &canonical_updated_at_raw,
            )? {
                ApplyResult::Applied => {
                    resummerize_audit::finish_resummerize_run(
                        store.conn(),
                        &run_id,
                        Some(output_bytes as u32),
                        Some(sha256_hex(&llm_output)),
                        ResummerizeRunStatus::Success,
                        &[],
                        None,
                        Utc::now(),
                    )?;
                    // Agent D Q6 — KG concept revisions can drift when
                    // resummerize materially changes canonical wording.
                    // `concepts.source_memory_ids` still points at this
                    // canonical ID but the underlying content has
                    // shifted. The full fix (automatic re-extraction of
                    // concepts from the new canonical) is v0.24 — it
                    // requires an LLM call, a refresh queue, and
                    // backpressure coordination. For now, flag affected
                    // concepts so operators have a handle + doctor can
                    // surface the drift backlog.
                    if let Ok(count) =
                        mark_concepts_needing_refresh_for_canonical(store, &claim.canonical_id)
                    {
                        if count > 0 {
                            tracing::warn!(
                                canonical_id = %claim.canonical_id,
                                concepts_affected = count,
                                "resummerize: KG concepts reference this canonical; \
                                 their definitions may now be semantically stale. \
                                 Re-run concept extraction via `rein memoir refine` \
                                 or wait for v0.24 automatic refresh."
                            );
                        }
                    }
                    // Agent D Q7 — episode replay fidelity. Episodes
                    // reference memory IDs; after resummerize, replaying
                    // an episode returns the NEW canonical content even
                    // if the episode was captured against the old
                    // content. The full fix (content-hash snapshot at
                    // episode creation) is v0.24 — it needs a schema
                    // change and a replay-path refactor to check hashes.
                    // For now, flag episodes that may have drifted so
                    // operators can audit before acting on replayed
                    // episode content.
                    if let Ok(count) =
                        count_episodes_referencing_canonical(store, &claim.canonical_id)
                    {
                        if count > 0 {
                            tracing::warn!(
                                canonical_id = %claim.canonical_id,
                                episodes_affected = count,
                                "resummerize: episodes reference this canonical \
                                 (episode replay will now return the rewritten \
                                 content, not the content captured at session \
                                 time). Full replay fidelity requires content \
                                 snapshots — see v0.24 plan."
                            );
                        }
                    }
                    Ok(Verdict::Success)
                }
                ApplyResult::ClaimLost => {
                    // Another worker reclaimed the row, or a different
                    // writer changed the canonical after we built the LLM
                    // input snapshot. Discard this output — newer state
                    // is authoritative and we must not overwrite it.
                    tracing::info!(
                        canonical_id = %claim.canonical_id,
                        our_token = %claim.token,
                        "resummerize: claim lost or canonical changed \
                         concurrently; discarding output"
                    );
                    // Terminal-update the open audit row to `claim_lost`
                    // rather than deleting it. Prior versions best-effort
                    // DELETEd and commented that "if the delete fails,
                    // the row lingers but its status is `llm_error`
                    // (the starting placeholder), which over time may
                    // trip the 3-strike fuse unnecessarily" — Agent D
                    // Q2/Q15 picked that up in the post-v0.23.0 review.
                    // With an explicit terminal status, the fuse counter
                    // (see `count_recent_consecutive_failures` treatment
                    // of `ClaimLost` below) correctly classifies this as
                    // a non-failure, AND the audit row stays durable so
                    // the sunk LLM cost is visible to `rein doctor`.
                    resummerize_audit::finish_resummerize_run(
                        store.conn(),
                        &run_id,
                        Some(output_bytes as u32),
                        Some(sha256_hex(&llm_output)),
                        ResummerizeRunStatus::ClaimLost,
                        &[],
                        None,
                        Utc::now(),
                    )?;
                    Ok(Verdict::ClaimLost)
                }
            }
        }
        Err(violations) => {
            let violation_names: Vec<String> =
                violations.iter().map(|v| v.invariant.to_string()).collect();
            let detail = violations
                .iter()
                .map(|v| format!("{}: {}", v.invariant, v.detail))
                .collect::<Vec<_>>()
                .join("; ");
            // Post-audit round-2 MED-1: same atomic recheck-and-finish as
            // the length path above. A concurrent MergeInto that updated
            // the canonical after our snapshot can cause the LLM — which
            // saw stale evidence + an older canonical — to produce
            // output that doesn't satisfy the NEW canonical's contract
            // input. `finish_with_ownership_check` ensures the audit row
            // says `ClaimLost` (non-counting) when that happens rather
            // than `ContractViolation` (countable → fuse).
            let race_detail = format!(
                "contract check raced with concurrent writer; \
                 original violations would have been: {}",
                detail
            );
            let verdict = finish_with_ownership_check(
                store,
                &run_id,
                &claim.canonical_id,
                &claim.token,
                &canonical_updated_at_raw,
                output_bytes as u32,
                &sha256_hex(&llm_output),
                ResummerizeRunStatus::ContractViolation,
                &violation_names,
                Some(detail),
                &race_detail,
            )?;
            Ok(verdict)
        }
    }
}

/// Apply a successful resummerize atomically.
///
/// Wraps three destructive steps in one `BEGIN IMMEDIATE` transaction
/// (Codex H2 — prior flow had a crash window between each step):
///
/// 1. Snapshot the pre-resummerize canonical into `memory_evidence`.
///    This is the single missing frame for post-hoc rollback — the
///    keep-tail canonical as it stood immediately before LLM rewrite.
///    Per `feedback_cleanup_caution.md`, destructive rewrites must
///    preserve prior state rather than trust downstream reconstruction.
///
/// 2. Rewrite canonical content/summary/updated_at + clear
///    `needs_resummerize` + clear `in_progress_resummerize_at` + stamp
///    `last_resummarized_at`, **all in one direct SQL UPDATE** that
///    bypasses `store.update()` (Codex M8). `store.update()` routes
///    through triggers that increment `support_count`/`merge_count` as
///    if this were a merge — a successful resummerize would otherwise
///    inflate both counters per run and flip `status` from `active` to
///    `updated`, making `recompute_canonical_length_stats` skip
///    resummerized rows on its next pass.
///
/// 3. If either step fails, ROLLBACK leaves the row in exactly the state
///    it was in at entry — the caller can retry (subject to the 3-strike
///    fuse) or the stale-claim timer will free the row for another
///    worker. No phantom snapshot, no half-rewritten canonical.
/// Outcome of `apply_resummerize`. `Applied` means the canonical was
/// rewritten and the transaction committed; `ClaimLost` means another
/// worker had reclaimed the row (stale-timeout path) so we ROLLBACKed
/// rather than overwriting their work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ApplyResult {
    Applied,
    ClaimLost,
}

fn apply_resummerize(
    store: &SqliteStore,
    canonical: crate::types::Memory,
    new_content: &str,
    claim_token: &str,
    canonical_updated_at_raw: &str,
) -> ReinResult<ApplyResult> {
    let canonical_id = canonical.id.clone();
    // Cache the bits we need AFTER the transaction commits for the
    // side-index refresh (Codex round-3 HIGH). Do this before the
    // closure so we don't have to borrow `canonical` across the match.
    let new_summary: String = new_content.chars().take(SUMMARY_MAX_CHARS).collect();
    let canonical_topic = canonical.topic.clone();
    let canonical_keywords = canonical.keywords.clone();
    let conn = store.conn();

    conn.execute_batch("BEGIN IMMEDIATE")?;
    // Track whether the ownership UPDATE actually matched so we know
    // whether to COMMIT (Applied) or ROLLBACK (ClaimLost).
    let txn_result = (|| -> ReinResult<ApplyResult> {
        // Step 1: snapshot pre-rewrite canonical. Uses `add_memory_evidence`
        // directly rather than `snapshot_memory_as_evidence` because the
        // latter calls `refresh_canonical_state`, which re-derives
        // `support_count` / `merge_count` as `COUNT(memory_evidence) - 1`
        // — so every successful resummerize would inflate both counters
        // (Codex M8). Resummerize is a content rewrite, not a merge: it
        // adds no new supporting memory, only a recovery frame of the
        // prior canonical. Skipping the refresh here keeps the counters
        // bound to actual merge history.
        //
        // Snapshot failure aborts the rewrite (transaction ROLLBACKs) —
        // we never mutate the canonical without a recovery frame landing.
        store.add_memory_evidence(crate::types::MemoryEvidence {
            id: String::new(),
            canonical_id: canonical_id.clone(),
            memory_id: Some(canonical.id.clone()),
            source_topic: canonical.topic.clone(),
            summary: canonical.summary.clone(),
            content: canonical.content.clone(),
            keywords: canonical.keywords.clone(),
            source: canonical.source,
            created_at: canonical.created_at,
            imported_at: Utc::now(),
        })?;

        // Step 2: canonical + flags, one UPDATE, no triggers. The
        // `in_progress_resummerize_at = ?claim_token` clause turns this
        // into a compare-and-swap on claim ownership (Codex round-2
        // HIGH), and `updated_at = ?snapshot_updated_at` extends that
        // CAS to the canonical snapshot we compressed. If another writer
        // mutated the canonical after our LLM input snapshot (merge,
        // keyword edit, supersede, deactivation), this UPDATE affects 0
        // rows and we roll back the evidence snapshot rather than
        // overwriting newer state with stale compressed output.
        let now = Utc::now().to_rfc3339();
        let affected = conn.execute(
            "UPDATE memories \
             SET content = ?1, \
                 summary = ?2, \
                 updated_at = ?3, \
                 last_resummarized_at = ?3, \
                 needs_resummerize = 0, \
                 needs_vec_dedup = 1, \
                 in_progress_resummerize_at = NULL \
             WHERE id = ?4 \
               AND in_progress_resummerize_at = ?5 \
               AND updated_at = ?6 \
               AND status IN ('active', 'updated') \
               AND superseded_by IS NULL",
            rusqlite::params![
                new_content,
                &new_summary,
                &now,
                &canonical_id,
                claim_token,
                canonical_updated_at_raw
            ],
        )?;

        if affected == 0 {
            Ok(ApplyResult::ClaimLost)
        } else {
            // The canonical text changed, so any sqlite-vec row derived
            // from the old content is stale. Delete it inside the same
            // transaction and let the existing deferred
            // `needs_vec_dedup` pipeline regenerate the vector later.
            crate::store::vec::delete_embedding(conn, &canonical_id)?;
            Ok(ApplyResult::Applied)
        }
    })();

    match txn_result {
        Ok(ApplyResult::Applied) => {
            match conn.execute_batch("COMMIT") {
                Ok(()) => {
                    // Side-index sync after a successful rewrite (Codex
                    // round-3 HIGH). Direct SQL bypassed the `update()`
                    // path that would normally fire `update_tantivy` and
                    // `update_hnsw`. We already deleted the stale
                    // sqlite-vec row inside the transaction, so the
                    // post-COMMIT work here only needs to refresh Tantivy
                    // and evict the stale HNSW entry. SQLite's
                    // `memories_fts` trigger already kept FTS5 in sync
                    // with the content change.
                    //
                    // STALE-INDEX WINDOW (Agent D Q3):
                    // Between COMMIT and `refresh_indexes_after_canonical_rewrite`
                    // returning (~10-50ms on warm caches), a concurrent recall on
                    // a separate connection can:
                    //   * read the NEW canonical content from SQLite (WAL is
                    //     read-consistent — content correctness is safe), AND
                    //   * receive HNSW / Tantivy SCORES that reflect the OLD
                    //     embedding / posting. Functionally the row is still
                    //     returned with the right ID; only the relevance
                    //     ranking is briefly off.
                    // Self-heals on the next `run_vec_dedup` sweep (re-embed +
                    // HNSW re-insert via `update_hnsw_for_vec_dedup`) and on
                    // the next Tantivy commit (`update_tantivy` issues one
                    // synchronously below). The fully invariant fix —
                    // version-check `updated_at` in the recall fusion path —
                    // is recall.rs work, deferred to v0.24.
                    let refresh_started = std::time::Instant::now();
                    store.refresh_indexes_after_canonical_rewrite(
                        &canonical_id,
                        &canonical_topic,
                        &new_summary,
                        new_content,
                        &canonical_keywords,
                    );
                    let refresh_micros = refresh_started.elapsed().as_micros();
                    tracing::debug!(
                        canonical_id = %canonical_id,
                        refresh_micros,
                        "resummerize: side-index refresh complete \
                         (recall observed stale scores for at most this window)"
                    );
                    Ok(ApplyResult::Applied)
                }
                Err(e) => {
                    // Codex round-2 MEDIUM: a failed COMMIT may leave the
                    // transaction open and silently poison every
                    // subsequent write on this connection. Force an
                    // explicit ROLLBACK before surfacing the error so the
                    // connection returns to a defined state. If the
                    // ROLLBACK itself fails (Codex round-3 residual), log
                    // both errors so the operator isn't blind to the
                    // second failure.
                    if let Err(rollback_err) = conn.execute_batch("ROLLBACK") {
                        tracing::warn!(
                            commit_error = %e,
                            rollback_error = %rollback_err,
                            "resummerize COMMIT failed AND subsequent ROLLBACK \
                             also failed; connection state is indeterminate, \
                             expect the next write on this connection to surface \
                             a secondary error"
                        );
                    }
                    Err(e.into())
                }
            }
        }
        Ok(ApplyResult::ClaimLost) => {
            // Claim mismatch / stale snapshot is NOT an error —
            // discard the snapshot we wrote inside the transaction
            // (ROLLBACK) and report back.
            let _ = conn.execute_batch("ROLLBACK");
            Ok(ApplyResult::ClaimLost)
        }
        Err(e) => {
            let _ = conn.execute_batch("ROLLBACK");
            Err(e)
        }
    }
}

/// Test-only surface for driving internal resummerize helpers from
/// integration tests. Gated behind `test-support` so it's absent from
/// production binaries. Exists purely so the Codex round-2 HIGH fix
/// (claim-lost path) has direct unit-test coverage without needing to
/// stage a cross-thread race in a single-threaded test runner.
#[cfg(feature = "test-support")]
pub mod test_hooks {
    use super::*;

    /// Outcome mirror of the private `ApplyResult`, re-exported so test
    /// code doesn't need visibility on the internal enum.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ApplyOutcome {
        Applied,
        ClaimLost,
    }

    /// Claim a single canonical and return the stamped claim token.
    /// `None` when the row is ineligible (not flagged / superseded /
    /// already claimed by a live worker).
    pub fn claim_for_test(store: &SqliteStore, canonical_id: &str) -> ReinResult<Option<String>> {
        let claims = claim_batch(store, Some(canonical_id), 1)?;
        Ok(claims.into_iter().next().map(|c| c.token))
    }

    /// Pure ownership check — same semantics as the production helper
    /// used before writing a countable terminal status. Exposed so the
    /// Codex H-1 regression test can directly exercise the drift-detection
    /// logic without staging a real cross-thread race.
    pub fn check_claim_still_held_for_test(
        store: &SqliteStore,
        canonical_id: &str,
        claim_token: &str,
        snapshot_updated_at_raw: &str,
    ) -> ReinResult<bool> {
        super::check_claim_still_held(store, canonical_id, claim_token, snapshot_updated_at_raw)
    }

    /// Run `apply_resummerize` directly with a caller-provided token.
    /// Integration tests pass a token that has since been reassigned in
    /// the DB to exercise the claim-lost ROLLBACK path. Reads the raw
    /// `updated_at` text from the row at call time so the CAS uses the
    /// byte-identical stored value (matches round-5 M-1 semantics).
    pub fn apply_for_test(
        store: &SqliteStore,
        canonical_id: &str,
        new_content: &str,
        claim_token: &str,
    ) -> ReinResult<ApplyOutcome> {
        let canonical = store.get(canonical_id)?;
        let updated_at_raw: String = store
            .conn()
            .query_row(
                "SELECT updated_at FROM memories WHERE id = ?1",
                rusqlite::params![canonical_id],
                |row| row.get(0),
            )
            .map_err(|e| {
                crate::types::ReinError::Config(format!(
                    "apply_for_test: failed to read updated_at: {e}"
                ))
            })?;
        match apply_resummerize(store, canonical, new_content, claim_token, &updated_at_raw)? {
            ApplyResult::Applied => Ok(ApplyOutcome::Applied),
            ApplyResult::ClaimLost => Ok(ApplyOutcome::ClaimLost),
        }
    }
}

/// Post-audit round-2 MED-1: atomically (under `BEGIN IMMEDIATE`) verify
/// ownership and write the appropriate terminal audit row. If ownership
/// is still held, writes the `intended_status` (`LengthExceeded` or
/// `ContractViolation`); otherwise writes `ClaimLost` with `race_detail`
/// as the error text. Returns the matching `Verdict`.
///
/// Why atomic: before the audit's round-2 MED-1 fix, the call sequence
/// was `check_claim_still_held()` → `finish_resummerize_run()` as TWO
/// separate statements. A peer `MergeInto` could commit in the window
/// between them, so `check` returned `true` but the row we were about to
/// write the `ContractViolation` audit entry AGAINST had already moved
/// on. IMMEDIATE grabs a reserved lock; concurrent writers wait until
/// our COMMIT.
///
/// Returns `Ok(Verdict::ClaimLost | Verdict::LengthExceeded |
/// Verdict::ContractViolation)`. The `Success` variant is not a valid
/// `intended_status` for this helper — it's handled on the Ok-contract
/// branch of `apply_resummerize`, which uses its own 5-way CAS rather
/// than this wrapper.
#[allow(clippy::too_many_arguments)]
fn finish_with_ownership_check(
    store: &SqliteStore,
    run_id: &str,
    canonical_id: &str,
    claim_token: &str,
    snapshot_updated_at_raw: &str,
    output_bytes: u32,
    output_hash: &str,
    intended_status: ResummerizeRunStatus,
    intended_violations: &[String],
    intended_error: Option<String>,
    race_detail: &str,
) -> ReinResult<Verdict> {
    let conn = store.conn();
    conn.execute_batch("BEGIN IMMEDIATE")?;
    let now = Utc::now();
    let result: ReinResult<Verdict> = (|| {
        let still_ours =
            check_claim_still_held(store, canonical_id, claim_token, snapshot_updated_at_raw)?;
        if still_ours {
            resummerize_audit::finish_resummerize_run(
                conn,
                run_id,
                Some(output_bytes),
                Some(output_hash.to_string()),
                intended_status,
                intended_violations,
                intended_error,
                now,
            )?;
            let verdict = match intended_status {
                ResummerizeRunStatus::LengthExceeded => Verdict::LengthExceeded,
                ResummerizeRunStatus::ContractViolation => Verdict::ContractViolation,
                _ => {
                    return Err(ReinError::Config(format!(
                        "finish_with_ownership_check called with unsupported \
                         intended_status={intended_status:?}"
                    )));
                }
            };
            Ok(verdict)
        } else {
            resummerize_audit::finish_resummerize_run(
                conn,
                run_id,
                Some(output_bytes),
                Some(output_hash.to_string()),
                ResummerizeRunStatus::ClaimLost,
                &[],
                Some(race_detail.to_string()),
                now,
            )?;
            Ok(Verdict::ClaimLost)
        }
    })();
    match result {
        Ok(verdict) => {
            if let Err(commit_err) = conn.execute_batch("COMMIT") {
                tracing::warn!(
                    run_id = %run_id,
                    canonical_id = %canonical_id,
                    error = %commit_err,
                    "finish_with_ownership_check: COMMIT failed; attempting ROLLBACK"
                );
                let _ = conn.execute_batch("ROLLBACK");
                return Err(commit_err.into());
            }
            Ok(verdict)
        }
        Err(e) => {
            let _ = conn.execute_batch("ROLLBACK");
            // Post-audit round-2 LOW #2: if the recheck SQL errors (I/O,
            // schema corruption, whatever), the starter row that
            // `insert_resummerize_run` opened would be left with
            // `finished_at = NULL`. That's silently invisible to doctor
            // / fuse / failure_rate, so operators can't even SEE that
            // this run attempted. Best-effort write an `llm_error`
            // terminal status (non-counting per M-3) so the run is at
            // least durable in the audit table. If the best-effort write
            // ALSO fails, we propagate the original error — doctor may
            // miss this one case but the process isn't wedged.
            //
            // Round-3 audit Finding 6: log the secondary failure too, so
            // an operator digging into the original error has a trail of
            // the orphan-row cause. Prior `let _ = ...` silently dropped
            // the secondary error; if both the recheck and the finish
            // failed (e.g. the connection itself died), there was no
            // evidence of it in logs.
            let best_effort_err =
                format!("ownership recheck SQL failed during terminal status write: {e}");
            if let Err(finish_err) = resummerize_audit::finish_resummerize_run(
                conn,
                run_id,
                Some(output_bytes),
                Some(output_hash.to_string()),
                ResummerizeRunStatus::LlmError,
                &[],
                Some(best_effort_err),
                now,
            ) {
                tracing::warn!(
                    run_id = %run_id,
                    canonical_id = %canonical_id,
                    recheck_err = %e,
                    finish_err = %finish_err,
                    "finish_with_ownership_check: best-effort llm_error \
                     finish ALSO failed; starter row will remain with \
                     finished_at=NULL until a janitor picks it up"
                );
            }
            Err(e)
        }
    }
}

/// Codex post-fix audit H-1: before writing a countable terminal status
/// (`LengthExceeded` / `ContractViolation`), verify that this worker still
/// owns the claim AND the canonical's `updated_at` still matches the
/// snapshot we based the LLM call on. If either has drifted, a concurrent
/// `MergeInto` or stale-claim takeover has already invalidated our work —
/// and the `apply_resummerize` 5-way CAS would have rejected our rewrite
/// anyway. Writing `ContractViolation` in that case is a false positive
/// that counts toward the 3-strike fuse and can permanently strand a row.
///
/// Returns `Ok(true)` when ownership + snapshot are both still valid,
/// `Ok(false)` on any mismatch. The recheck is a read-only SELECT on the
/// same connection; it may race again with a subsequent writer, but the
/// window between this check and the audit-row write is microseconds and
/// a race at that granularity is indistinguishable from "ownership was
/// still held when we decided." The `apply_resummerize` path's
/// `BEGIN IMMEDIATE` + 5-way CAS remains the authoritative safety net.
fn check_claim_still_held(
    store: &SqliteStore,
    canonical_id: &str,
    claim_token: &str,
    snapshot_updated_at_raw: &str,
) -> ReinResult<bool> {
    let row: Option<(Option<String>, String, String)> = store
        .conn()
        .query_row(
            "SELECT in_progress_resummerize_at, updated_at, status \
               FROM memories WHERE id = ?1",
            rusqlite::params![canonical_id],
            |r| {
                Ok((
                    r.get::<_, Option<String>>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                ))
            },
        )
        .optional()?;
    let Some((in_progress_at, updated_at, status)) = row else {
        return Ok(false);
    };
    let still_claimed = in_progress_at.as_deref() == Some(claim_token);
    let snapshot_matches = updated_at == snapshot_updated_at_raw;
    let live_status = status == "active" || status == "updated";
    Ok(still_claimed && snapshot_matches && live_status)
}

/// Agent D Q7: count episodes whose `memory_ids` contain this canonical.
/// Episodes persist only memory IDs + concept IDs; there's no content
/// snapshot. After resummerize, replaying an episode dereferences the ID
/// to the CURRENT canonical content, not the content captured at session
/// time. Full replay fidelity requires a content-hash schema addition to
/// episodes (v0.24). For now this counter is emitted via `tracing::warn`
/// from the resummerize success path so operators have a drift signal.
fn count_episodes_referencing_canonical(
    store: &SqliteStore,
    canonical_id: &str,
) -> ReinResult<u64> {
    // `episodes.memory_ids` is a JSON array column. Using `json_each`
    // with `value = ?1` avoids the LIKE escape pitfalls of
    // `%"<id>"%`-style probes (double-quote wasn't escaped in the pre-fix
    // implementation; a canonical_id containing `"` would have been
    // stored as `\"` in the array and missed by the probe). Post-fix
    // audit L-1.
    let count: i64 = store.conn().query_row(
        "SELECT COUNT(DISTINCT e.id) \
           FROM episodes e, json_each(e.memory_ids) \
           WHERE json_each.value = ?1",
        rusqlite::params![canonical_id],
        |row| row.get(0),
    )?;
    Ok(count as u64)
}

/// Agent D Q6: count KG concepts whose `source_memory_ids` references
/// this canonical. The column stores a JSON array of memory IDs, so we
/// use the same LIKE-with-quoted-id pattern established by
/// `store/knowledge.rs` when it needs to find concept back-references.
///
/// Currently the count is emitted via a `tracing::warn` so operators
/// have a signal that concept definitions may now be stale relative to
/// the rewritten canonical. Full automatic re-extraction is v0.24 work
/// — it needs an LLM call, a refresh queue, and shared quota
/// coordination with the other LLM features.
fn mark_concepts_needing_refresh_for_canonical(
    store: &SqliteStore,
    canonical_id: &str,
) -> ReinResult<u64> {
    // `concepts.source_memory_ids` is a JSON array column. Using
    // `json_each` with `value = ?1` avoids the LIKE escape pitfalls of
    // `%"<id>"%`-style probes (double-quote wasn't escaped in the
    // pre-fix implementation; an id containing `"` would have been
    // stored as `\"` in the array and missed by the probe). Post-fix
    // audit L-1.
    let count: i64 = store.conn().query_row(
        "SELECT COUNT(DISTINCT c.id) \
           FROM concepts c, json_each(c.source_memory_ids) \
           WHERE json_each.value = ?1",
        rusqlite::params![canonical_id],
        |row| row.get(0),
    )?;
    Ok(count as u64)
}

/// Clear the `needs_resummerize` flag and release any claim on the row.
/// Used on early-exit paths where no audit row is recorded (e.g. the row
/// became empty between flagging and processing).
fn clear_flag(store: &SqliteStore, canonical_id: &str) -> ReinResult<()> {
    store.conn().execute(
        "UPDATE memories \
         SET needs_resummerize = 0, in_progress_resummerize_at = NULL \
         WHERE id = ?1",
        rusqlite::params![canonical_id],
    )?;
    Ok(())
}

fn record_exhaustion(
    store: &SqliteStore,
    canonical_id: &str,
    prior_failures: usize,
) -> ReinResult<()> {
    let exhaustion_row = ResummerizeRunRow {
        id: ulid::Ulid::new().to_string(),
        canonical_id: canonical_id.to_string(),
        input_evidence_count: 0,
        input_canonical_chars: 0,
        output_chars: None,
        output_hash: None,
        target_bytes: 0,
        status: ResummerizeRunStatus::Exhausted,
        violations: Vec::new(),
        error: Some(format!(
            "cleared after {prior_failures} consecutive failures"
        )),
        llm_backend: None,
        created_at: Utc::now(),
        finished_at: Some(Utc::now()),
    };
    resummerize_audit::insert_resummerize_run(store.conn(), &exhaustion_row)?;
    // clear_flag also clears in_progress_resummerize_at so the row is
    // fully demoted from the work queue.
    clear_flag(store, canonical_id)?;
    Ok(())
}

// ── LLM wiring ───────────────────────────────────────────────────────────────

/// `pub` so the `rein-eval` binary (a separate crate target) can drive the
/// same async-bridging pattern used in production. Drift here = eval running
/// under different runtime semantics than prod (e.g. blocking vs.
/// block_in_place), which would invalidate latency / failure-mode
/// comparisons. **Not a stable public API.**
pub fn call_llm_sync(extractor: &ExtractorKind, prompt: &str) -> ReinResult<String> {
    // Mirrors `ops/dedup.rs:452` / `consolidation.rs:643` pattern so this op
    // runs correctly whether invoked from inside an existing tokio runtime
    // (MCP/REST) or a fresh sync context (CLI).
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        tokio::task::block_in_place(|| {
            handle.block_on(async { extractor.raw_text_with_prompt(SYSTEM_PROMPT, prompt).await })
        })
    } else {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| ReinError::Config(format!("failed to build tokio runtime: {e}")))?;
        rt.block_on(async { extractor.raw_text_with_prompt(SYSTEM_PROMPT, prompt).await })
    }
}

fn extractor_backend_tag(extractor: &ExtractorKind) -> Option<String> {
    Some(
        match extractor {
            ExtractorKind::Gemini(_) => "gemini",
            ExtractorKind::Omlx(_) => "omlx",
            #[cfg(feature = "test-support")]
            ExtractorKind::Mock(_) => "mock",
        }
        .to_string(),
    )
}

/// `pub` so the `rein-eval` binary (a separate crate target) can build the
/// prompt the same way production does. Sharing this function (rather than
/// duplicating it in the eval bin) is the load-bearing piece that makes
/// baseline and treatment scorecards comparable. **Not a stable public
/// API.**
pub fn build_prompt(input: &ContractInput) -> String {
    let mut buf = String::with_capacity(
        input.current_canonical.len()
            + input
                .evidence
                .iter()
                .map(|e| e.content.len())
                .sum::<usize>()
            + 512,
    );
    buf.push_str(&format!(
        "Target bytes: {} (must fit within target + 10%)\n\n",
        input.target_bytes
    ));
    buf.push_str("Current canonical (possibly keep-tail truncated):\n<<<CANONICAL>>>\n");
    buf.push_str(input.current_canonical);
    buf.push_str("\n<<<END CANONICAL>>>\n\n");
    buf.push_str("Merge evidence (oldest first — newer merges appear later):\n");
    for (i, e) in input.evidence.iter().enumerate() {
        buf.push_str(&format!(
            "--- Evidence #{} merged at {} ---\n",
            i + 1,
            e.merged_at.format("%Y-%m-%d")
        ));
        buf.push_str(&e.content);
        if !e.content.ends_with('\n') {
            buf.push('\n');
        }
    }
    buf.push_str("\nNow produce the new canonical text.");
    buf
}

fn sha256_hex(s: &str) -> String {
    let mut h = Sha256::new();
    h.update(s.as_bytes());
    let bytes = h.finalize();
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

// ── Backlog / health helpers (used by doctor) ────────────────────────────────

/// Count of canonicals outstanding for resummerize (flagged + active +
/// non-superseded + own canonical). **Includes** rows currently claimed
/// by a live worker so the doctor reports true work remaining, not
/// "ready-to-pick" which can momentarily drop to zero during active
/// processing. Codex audit H5.
pub fn backlog_count(store: &SqliteStore) -> ReinResult<u64> {
    // Core eligibility minus the claim-status clause. Claimed rows are
    // still "work outstanding" — they'll return to the ready pool if the
    // claim expires, and if they succeed they turn into a -1 to this
    // count anyway.
    // `status IN ('active', 'updated')` mirrors the ELIGIBILITY_PREDICATE
    // widening from round-5 H-1 so backlog depth reflects actual work
    // the op will pick up (merged canonicals get promoted to `updated`).
    const BACKLOG_PREDICATE: &str = "\
        m.needs_resummerize = 1 \
        AND m.status IN ('active', 'updated') \
        AND m.superseded_by IS NULL \
        AND EXISTS ( \
            SELECT 1 FROM memory_canonical_state cs \
            WHERE cs.memory_id = m.id AND cs.canonical_id = m.id \
        )\
    ";
    let sql = format!("SELECT COUNT(*) FROM memories m WHERE {BACKLOG_PREDICATE}");
    Ok(store
        .conn()
        .query_row(&sql, [], |r| r.get(0))
        .optional()?
        .unwrap_or(0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_memory(content: &str) -> crate::types::Memory {
        crate::types::Memory {
            id: ulid::Ulid::new().to_string(),
            layer: crate::types::Importance::Medium.auto_layer(),
            topic: "resummerize-test".to_string(),
            summary: content
                .chars()
                .take(crate::types::SUMMARY_MAX_CHARS)
                .collect(),
            content: content.to_string(),
            keywords: vec!["seed".to_string()],
            importance: crate::types::Importance::Medium,
            source: crate::types::Source::Manual,
            strength: 1.0,
            decay_lambda: 0.06 * crate::types::Importance::Medium.decay_factor(),
            access_count: 0,
            superseded_by: None,
            canonical_id: None,
            support_count: 1,
            merge_count: 0,
            dedup_confidence: 1.0,
            source_diversity: 1.0,
            contradiction_score: 0.0,
            related_ids: vec![],
            concept_ids: vec![],
            status: crate::types::MemoryStatus::default(),
            embedding: None,
            tier: crate::types::MemoryTier::Warm,
            cluster_id: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_accessed: Utc::now(),
        }
    }

    #[test]
    fn backlog_count_is_zero_on_empty_db() {
        let store = SqliteStore::in_memory().unwrap();
        assert_eq!(backlog_count(&store).unwrap(), 0);
    }

    #[test]
    fn claim_batch_returns_only_flagged_rows() {
        // This test relies on Agent A's migration landing `needs_resummerize`
        // and the H6 migration landing `in_progress_resummerize_at`. Cleaner
        // integration coverage lives in `tests/resummerize_integration.rs`.
        let store = SqliteStore::in_memory().unwrap();
        let batch = claim_batch(&store, None, 10).unwrap();
        assert!(batch.is_empty());
    }

    #[test]
    fn preview_eligible_is_empty_on_clean_db() {
        let store = SqliteStore::in_memory().unwrap();
        let previewed = preview_eligible(&store, None).unwrap();
        assert!(previewed.is_empty());
    }

    #[test]
    fn build_prompt_includes_target_and_evidence() {
        let evidence = vec![EvidenceEntry {
            content: "fact A".to_string(),
            merged_at: Utc::now(),
        }];
        let input = ContractInput {
            evidence: &evidence,
            current_canonical: "base canonical text",
            target_bytes: 1_234,
        };
        let prompt = build_prompt(&input);
        assert!(prompt.contains("Target bytes: 1234"));
        assert!(prompt.contains("base canonical text"));
        assert!(prompt.contains("fact A"));
    }

    #[test]
    fn sha256_hex_stable_and_hex() {
        let h = sha256_hex("hello");
        assert_eq!(h.len(), 64);
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(h, sha256_hex("hello"));
        assert_ne!(h, sha256_hex("HELLO"));
    }

    #[test]
    fn apply_resummerize_rolls_back_if_canonical_changed_after_snapshot() {
        let store = SqliteStore::in_memory().unwrap();
        let canonical_id = store.store(test_memory("original canonical body")).unwrap();
        store
            .conn()
            .execute(
                "UPDATE memories SET needs_resummerize = 1 WHERE id = ?1",
                rusqlite::params![&canonical_id],
            )
            .unwrap();

        let claim = claim_batch(&store, Some(&canonical_id), 1)
            .unwrap()
            .into_iter()
            .next()
            .expect("flagged canonical should be claimable");
        let snapshot = store.get(&canonical_id).unwrap();
        // Capture raw updated_at BEFORE the concurrent mutation — this is
        // the snapshot the LLM-output would need to prove consistency against.
        let snapshot_updated_at_raw: String = store
            .conn()
            .query_row(
                "SELECT updated_at FROM memories WHERE id = ?1",
                rusqlite::params![&canonical_id],
                |row| row.get(0),
            )
            .unwrap();
        let evidence_before = store
            .list_memory_evidence(&canonical_id, 100)
            .unwrap()
            .len();

        let concurrent_updated_at = Utc::now().to_rfc3339();
        store
            .conn()
            .execute(
                "UPDATE memories SET keywords = ?1, updated_at = ?2 WHERE id = ?3",
                rusqlite::params![
                    "[\"concurrent-change\"]",
                    &concurrent_updated_at,
                    &canonical_id
                ],
            )
            .unwrap();

        let outcome = apply_resummerize(
            &store,
            snapshot,
            "compressed canonical body",
            &claim.token,
            &snapshot_updated_at_raw,
        )
        .unwrap();

        assert_eq!(
            outcome,
            ApplyResult::ClaimLost,
            "stale snapshot must not overwrite a canonical that changed after the LLM snapshot"
        );
        assert_eq!(
            store.get(&canonical_id).unwrap().keywords,
            vec!["concurrent-change".to_string()],
            "concurrent mutation should survive the aborted resummerize apply"
        );
        assert_eq!(
            store
                .list_memory_evidence(&canonical_id, 100)
                .unwrap()
                .len(),
            evidence_before,
            "aborted apply must roll back the pre-rewrite evidence snapshot"
        );
    }

    #[test]
    fn apply_resummerize_clears_stale_vec_row_and_flags_reembed() {
        let store = SqliteStore::in_memory().unwrap();
        let mut memory = test_memory("canonical body with embedding");
        let mut embedding = vec![0.0; 3072];
        embedding[0] = 1.0;
        memory.embedding = Some(embedding);
        let canonical_id = store.store(memory).unwrap();
        store
            .conn()
            .execute(
                "UPDATE memories SET needs_resummerize = 1 WHERE id = ?1",
                rusqlite::params![&canonical_id],
            )
            .unwrap();

        let claim = claim_batch(&store, Some(&canonical_id), 1)
            .unwrap()
            .into_iter()
            .next()
            .expect("flagged canonical should be claimable");
        let snapshot = store.get(&canonical_id).unwrap();
        let snapshot_updated_at_raw: String = store
            .conn()
            .query_row(
                "SELECT updated_at FROM memories WHERE id = ?1",
                rusqlite::params![&canonical_id],
                |row| row.get(0),
            )
            .unwrap();
        assert!(
            crate::store::vec::get_embedding(store.conn(), &canonical_id)
                .unwrap()
                .is_some(),
            "test precondition: vec row should exist before resummerize"
        );

        let outcome = apply_resummerize(
            &store,
            snapshot,
            "compressed canonical body with new semantics",
            &claim.token,
            &snapshot_updated_at_raw,
        )
        .unwrap();

        assert_eq!(outcome, ApplyResult::Applied);
        assert!(
            crate::store::vec::get_embedding(store.conn(), &canonical_id)
                .unwrap()
                .is_none(),
            "successful resummerize must clear the stale sqlite-vec row"
        );
        let needs_vec_dedup: i64 = store
            .conn()
            .query_row(
                "SELECT needs_vec_dedup FROM memories WHERE id = ?1",
                rusqlite::params![&canonical_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(
            needs_vec_dedup, 1,
            "successful resummerize must queue the canonical for deferred vector regeneration"
        );
    }
}
