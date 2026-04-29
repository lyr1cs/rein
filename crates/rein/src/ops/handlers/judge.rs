//! v0.27.1 Wave 1 — runtime LLM judge MCP handlers.
//!
//! Two `#[op]`-registered tools that let MCP-only callers (Claude Code /
//! Codex / agent-browser) manually trigger an LLM judge call against a
//! recently-produced synthesis (Cap B) or concept living-summary (Cap A).
//! Without this surface the judge queue only sees auto-enqueued events
//! sampled from GUI dwell/click traffic — so a pure-CLI/MCP install
//! never primes ARS adaptive feedback.
//!
//! Design — see
//! `docs/superpowers/specs/2026-04-27-v0.27.1-runtime-llm-judge-design.md`
//! §9.1, §9.2, §15 R9-K7. Highlights:
//!
//! * Both handlers are `mutating = true` (queue write); auth =
//!   `mutation_marker` like `rein_feedback`. Category = `"adaptive"`
//!   (mirrors `rein_feedback_concept_summary`; `"judge"` / `"feedback"`
//!   not in [`ALLOWED_CATEGORIES`]).
//! * **No synchronous `daily_cap_reached` reporting (R9-K7 option a).**
//!   `daily_call_cap` is reserved atomically inside the worker
//!   (`judge/contract.rs::reserve_call`, owned by A_JUDGE_CORE) at
//!   HTTP-call time; this handler can't observe live cap state without
//!   re-introducing the very TOCTOU race J2 is designed to avoid. The
//!   skipped-reason string is documented but never emitted from this
//!   surface.
//! * Cache file is read-only / best-effort: cache miss
//!   (`synthesis_id_expired_or_unknown`) is the loud failure mode for
//!   "synthesis already aged out of the 10-min TTL window" — there is
//!   no silent fallback to live-row rehydration (J7).
//! * The handler rehydrates the J7 stamp-time payload from the cache,
//!   re-stamps a `ManualMcp`-sourced [`JudgeJob`], and appends the JSON
//!   line to the worker queue file (Wave 1.5).
//!
//! ## Wave 1.5 wiring
//!
//! * `[ars.llm_judge]` exists (A_JUDGE_CORE + B1_CONFIG_RESOLVER landed
//!   the struct). The disabled gate now reads `enabled` AND the per-
//!   surface `synthesis_enabled` / `concept_summary_enabled` flags
//!   directly off `OpsRuntime::config()`.
//! * Does not touch `mcp/server.rs` — the `#[op]` macro with a
//!   nested `mcp(...)` clause auto-registers via `inventory::submit!`
//!   at expansion time; per-tool glue in `mcp/server.rs` would conflict
//!   with the single-source A1 design (Codex R3 P2 fix).
//! * Path + cache helpers are `pub(crate)` so the auto-enqueue side
//!   in `ops/recall_synthesis.rs` and `ops/concept_summary.rs` can
//!   write the same on-disk shape the manual-MCP path consumes.

use std::hash::{Hash, Hasher};
use std::path::PathBuf;

use rein_macros::op;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::ops::{IntoCliText, IntoJson, IntoMarkdown, OpsRuntime};
use crate::types::{OpsErrorKind, ReinError, ReinResult};

// ---------------------------------------------------------------------------
// Shared output shape
// ---------------------------------------------------------------------------

/// Outcome of a manual judge enqueue request.
///
/// Hand-rolled `IntoJson` + `IntoMarkdown` + `IntoCliText` impls below
/// (no `OpsRender` derive exists — see `crates/rein/src/ops/render.rs`
/// module doc; the spec's `#[derive(OpsRender)]` snippet is aspirational).
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct JudgeEnqueueResult {
    /// `true` when the job was appended to the judge queue. `false` when
    /// pre-flight gating dropped it (see [`Self::skipped_reason`]).
    pub enqueued: bool,
    /// Position the job took in the on-disk queue (1-indexed line number
    /// within `judge_<shard>.jsonl` AFTER the append). Best-effort —
    /// `None` when the file system errored or when not enqueued.
    pub queue_position: Option<u32>,
    /// Why the enqueue was skipped, when [`Self::enqueued`] is `false`.
    ///
    /// Defined values (string literals — handler emits at most one of these,
    /// and the v0.27.1 surface restricts to the 2 actually emit-able by
    /// this handler; see module doc for the cap-skip carve-out):
    ///
    /// * `"judge_disabled"` — `[ars.llm_judge].enabled = false` OR the
    ///   per-surface flag (`synthesis_enabled` / `concept_summary_enabled`)
    ///   is false.
    /// * `"synthesis_id_expired_or_unknown"` (or
    ///   `"concept_summary_id_expired_or_unknown"` for the Cap A op) —
    ///   no matching cache entry in the 10-min TTL window. J7
    ///   (stamp-time snapshot) forbids silent live-row rehydration.
    ///
    /// **Reserved-but-not-emitted** values (per spec §9.1 enum, may be
    /// surfaced by `worker_drop_log` / `judge_call_ledger` but never by
    /// this MCP handler — Codex R9-K7 carve-out): `"daily_cap_reached"`,
    /// `"weight_decay_invalid"`. Cap is enforced atomically at
    /// HTTP-call time inside the worker (`reserve_call`); a pre-enqueue
    /// read-only check would re-introduce the J2 TOCTOU race.
    pub skipped_reason: Option<String>,
}

impl IntoJson for JudgeEnqueueResult {
    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

impl IntoMarkdown for JudgeEnqueueResult {
    fn to_markdown(&self) -> String {
        match (self.enqueued, &self.skipped_reason) {
            (true, _) => {
                let pos = self
                    .queue_position
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| "?".into());
                format!("judge enqueued (queue position: {pos})")
            }
            (false, Some(reason)) => format!("judge skipped: {reason}"),
            (false, None) => "judge skipped (unknown reason)".to_string(),
        }
    }
}

impl IntoCliText for JudgeEnqueueResult {
    fn to_cli_text(&self) -> String {
        // CLI users get the same one-liner as MCP compact mode.
        IntoMarkdown::to_markdown(self)
    }
}

// ---------------------------------------------------------------------------
// Cap B — synthesis judge
// ---------------------------------------------------------------------------

/// Parameters for [`OpsRuntime::judge_synthesis`] /
/// `rein_judge_synthesis`.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
pub struct JudgeSynthesisParams {
    /// ULID minted by `run_recall_synthesis` and echoed back through
    /// `RecallSynthesisOutcome.synthesis_id`. Must reference a synthesis
    /// produced within the last `[ars.llm_judge].cache_ttl_secs` seconds
    /// (default 600).
    pub synthesis_id: String,
    /// Optional per-call override of the judge model (e.g. for A/B
    /// experiments). When `None`, falls back to the resolved
    /// `[ars.llm_judge]` provider chain (Track 2 §8). Bare model name —
    /// the resolved provider determines whether this is a Gemini SKU
    /// or an OMLX model id.
    #[serde(default)]
    pub judge_model_override: Option<String>,
}

// ---------------------------------------------------------------------------
// Cap A — concept-summary judge
// ---------------------------------------------------------------------------

/// Parameters for [`OpsRuntime::judge_concept_summary`] /
/// `rein_judge_concept_summary`.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
pub struct JudgeConceptSummaryParams {
    /// ULID minted by `concept/summary.rs` on a successful refresh and
    /// surfaced via `ConceptSummaryRefreshOutput.concept_summary_id` /
    /// the GUI Brain page. Must reference a refresh produced within the
    /// last `[ars.llm_judge].cache_ttl_secs` seconds (default 600).
    pub concept_summary_id: String,
    /// Optional per-call override of the judge model. See
    /// [`JudgeSynthesisParams::judge_model_override`].
    #[serde(default)]
    pub judge_model_override: Option<String>,
}

// ---------------------------------------------------------------------------
// `#[op]` registrations
// ---------------------------------------------------------------------------

impl OpsRuntime {
    #[op(
        name = "judge_synthesis",
        category = "adaptive",
        description = "Manually trigger an LLM judge call for a previous \
                       synthesis output. Useful for MCP-only callers \
                       (Claude Code / Codex) that won't naturally produce \
                       GUI dwell/click events. Fire-and-forget — returns \
                       immediately, judge runs async in the worker. The \
                       referenced synthesis must have been produced \
                       within the last `cache_ttl_secs` seconds (default \
                       600). Daily call cap is enforced atomically at \
                       HTTP-call time inside the worker, NOT at enqueue \
                       — manual triggers bypass sample rate but still \
                       count against the cap.",
        mcp(name = "rein_judge_synthesis"),
        auth = "mutation_marker",
        mutating = true
    )]
    pub fn judge_synthesis(&self, params: JudgeSynthesisParams) -> ReinResult<JudgeEnqueueResult> {
        if params.synthesis_id.trim().is_empty() {
            return Err(ReinError::Config("synthesis_id cannot be empty".into())
                .with_kind(OpsErrorKind::BadRequest));
        }

        // Default-off gate via `[ars.llm_judge].enabled` AND
        // `synthesis_enabled`. Returns `judge_disabled` rather than 4xx
        // so the MCP client gets a structured `enqueued = false` response
        // (matches the `synthesis_id_expired_or_unknown` shape).
        if !judge_enabled_for_synthesis(self) {
            return Ok(JudgeEnqueueResult {
                enqueued: false,
                queue_position: None,
                skipped_reason: Some("judge_disabled".into()),
            });
        }

        // Source rehydration — J7 stamp-time snapshot. Cache miss is the
        // loud failure mode; we never silently rehydrate from the live
        // row (concurrent `update()` / `apply_evolution` could poison
        // the judgment).
        let cache_path = synthesis_cache_path(self);
        // Codex R3 P2 fix — enforce `[ars.llm_judge].cache_ttl_secs` here.
        // Without TTL the cache file (append-only) keeps serving stale
        // pre-truncation prompts forever; J7 stamp-time guarantees only
        // hold within the window the spec advertises.
        let ttl = self.config().ars.llm_judge.cache_ttl_secs;
        let cache_entry = match cache_lookup_value_with_ttl(
            &cache_path,
            "synthesis_id",
            &params.synthesis_id,
            Some(ttl),
        ) {
            Some(v) => v,
            None => {
                return Ok(JudgeEnqueueResult {
                    enqueued: false,
                    queue_position: None,
                    skipped_reason: Some("synthesis_id_expired_or_unknown".into()),
                });
            }
        };

        // Reconstruct the full `JudgeJob` shape that `dispatch_one`
        // deserializes (J7: post-truncation prompt + candidate + stamp_hash
        // travel inline). A malformed cache row degrades to "expired" so
        // the worker never sees a partial payload.
        let Some(job) = build_synthesis_judge_job(
            &cache_entry,
            "ManualMcp",
            params.judge_model_override.as_deref(),
        ) else {
            tracing::warn!(
                target: "rein.judge",
                synthesis_id = %params.synthesis_id,
                "rein_judge_synthesis: cache entry malformed (missing prompt/candidate/stamp_hash)",
            );
            return Ok(JudgeEnqueueResult {
                enqueued: false,
                queue_position: None,
                skipped_reason: Some("synthesis_id_expired_or_unknown".into()),
            });
        };

        // Manual triggers bypass sample-rate (§6.5) but still respect
        // `daily_call_cap` — that check happens in the worker
        // (`reserve_call`), not here (R9-K7).
        // Codex R1 P3 fix — `judge_model_override` is now serialized into
        // the queued JudgeJob via `build_*_judge_job`. Worker reads it
        // and uses an alternate extractor when set.

        let queue_path = judge_queue_path(self);
        let queue_position = match append_jsonl_line(&queue_path, &job) {
            Ok(pos) => Some(pos),
            Err(e) => {
                tracing::warn!(
                    target: "rein.judge",
                    "rein_judge_synthesis: failed to append to judge queue: {e}",
                );
                None
            }
        };

        Ok(JudgeEnqueueResult {
            enqueued: queue_position.is_some(),
            queue_position,
            skipped_reason: None,
        })
    }

    #[op(
        name = "judge_concept_summary",
        category = "adaptive",
        description = "Manually trigger an LLM judge call for a previous \
                       concept living-summary refresh (ARS Cap A). \
                       Mirror of `rein_judge_synthesis` for the Cap A \
                       surface — same TTL, same worker queue, same cap \
                       semantics. Useful for MCP-only callers that want \
                       to bootstrap Cap A `useful_rate` without GUI \
                       traffic.",
        mcp(name = "rein_judge_concept_summary"),
        auth = "mutation_marker",
        mutating = true
    )]
    pub fn judge_concept_summary(
        &self,
        params: JudgeConceptSummaryParams,
    ) -> ReinResult<JudgeEnqueueResult> {
        if params.concept_summary_id.trim().is_empty() {
            return Err(
                ReinError::Config("concept_summary_id cannot be empty".into())
                    .with_kind(OpsErrorKind::BadRequest),
            );
        }

        if !judge_enabled_for_concept_summary(self) {
            return Ok(JudgeEnqueueResult {
                enqueued: false,
                queue_position: None,
                skipped_reason: Some("judge_disabled".into()),
            });
        }

        let cache_path = concept_summary_cache_path(self);
        // Codex R3 P2 fix — enforce TTL (mirror of synthesis path above).
        let ttl = self.config().ars.llm_judge.cache_ttl_secs;
        let cache_entry = match cache_lookup_value_with_ttl(
            &cache_path,
            "concept_summary_id",
            &params.concept_summary_id,
            Some(ttl),
        ) {
            Some(v) => v,
            None => {
                return Ok(JudgeEnqueueResult {
                    enqueued: false,
                    queue_position: None,
                    skipped_reason: Some("concept_summary_id_expired_or_unknown".into()),
                });
            }
        };

        let Some(job) = build_concept_summary_judge_job(
            &cache_entry,
            "ManualMcp",
            params.judge_model_override.as_deref(),
        ) else {
            tracing::warn!(
                target: "rein.judge",
                concept_summary_id = %params.concept_summary_id,
                "rein_judge_concept_summary: cache entry malformed (missing prompt/candidate/stamp_hash)",
            );
            return Ok(JudgeEnqueueResult {
                enqueued: false,
                queue_position: None,
                skipped_reason: Some("concept_summary_id_expired_or_unknown".into()),
            });
        };

        // Codex R1 P3 fix — `judge_model_override` is now serialized into
        // the queued JudgeJob via `build_*_judge_job`. Worker reads it
        // and uses an alternate extractor when set.
        let queue_path = judge_queue_path(self);
        let queue_position = match append_jsonl_line(&queue_path, &job) {
            Ok(pos) => Some(pos),
            Err(e) => {
                tracing::warn!(
                    target: "rein.judge",
                    "rein_judge_concept_summary: failed to append to judge queue: {e}",
                );
                None
            }
        };

        Ok(JudgeEnqueueResult {
            enqueued: queue_position.is_some(),
            queue_position,
            skipped_reason: None,
        })
    }
}

// ---------------------------------------------------------------------------
// Wave 1.5 helpers — shared with `ops/recall_synthesis.rs` +
// `ops/concept_summary.rs` auto-enqueue paths.
//
// All of these are `pub(crate)` so the same on-disk shape is produced by
// the auto-enqueue side and consumed by the manual-MCP path here. A_JUDGE_CORE
// may move them into `ops::llm_judge_worker` in a future sprint.
// ---------------------------------------------------------------------------

/// Master + per-surface enabled gate for the synthesis runtime LLM judge.
/// Reads `[ars.llm_judge]` (B1_CONFIG_RESOLVER landed the struct).
fn judge_enabled_for_synthesis(runtime: &OpsRuntime) -> bool {
    let cfg = &runtime.config().ars.llm_judge;
    cfg.enabled && cfg.synthesis_enabled
}

/// Mirror gate for the concept-summary surface.
fn judge_enabled_for_concept_summary(runtime: &OpsRuntime) -> bool {
    let cfg = &runtime.config().ars.llm_judge;
    cfg.enabled && cfg.concept_summary_enabled
}

/// Replicate the db-hash-shard scheme used by `extract/hooks/queue.rs`
/// (`project_scoped_path`) without depending on its private helper.
/// Same hash function (`DefaultHasher`) so the same db_tag is produced
/// — Wave 1.5 can swap to a shared helper without renaming files on
/// disk.
pub(crate) fn queue_scoped_path_for_config(
    config: &crate::config::ReinConfig,
    prefix: &str,
) -> PathBuf {
    let base = crate::extract::hooks::buffer::resolve_buffer_dir(config);
    let db_tag = {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        config.resolve_db_path().hash(&mut h);
        format!("{:016x}", h.finish())
    };
    let queue_dir = base.join("queue").join(&db_tag);
    let _ = std::fs::create_dir_all(&queue_dir);
    queue_dir.join(format!("{prefix}.jsonl"))
}

/// Path to the synthesis-source rehydration cache. The auto-enqueue side
/// in `ops/recall_synthesis.rs` writes to this path on a successful
/// synthesis when `[ars.llm_judge].enabled = true`.
pub(crate) fn synthesis_cache_path_for_config(config: &crate::config::ReinConfig) -> PathBuf {
    queue_scoped_path_for_config(config, "synthesis_cache")
}

fn synthesis_cache_path(runtime: &OpsRuntime) -> PathBuf {
    synthesis_cache_path_for_config(runtime.config())
}

/// Path to the concept-summary rehydration cache.
pub(crate) fn concept_summary_cache_path_for_config(config: &crate::config::ReinConfig) -> PathBuf {
    queue_scoped_path_for_config(config, "concept_summary_cache")
}

fn concept_summary_cache_path(runtime: &OpsRuntime) -> PathBuf {
    concept_summary_cache_path_for_config(runtime.config())
}

/// Path to the judge worker queue. Same path used by the auto-enqueue
/// side so the worker drains both manual + auto jobs from one queue.
pub(crate) fn judge_queue_path_for_config(config: &crate::config::ReinConfig) -> PathBuf {
    queue_scoped_path_for_config(config, "judge_queue")
}

fn judge_queue_path(runtime: &OpsRuntime) -> PathBuf {
    judge_queue_path_for_config(runtime.config())
}

/// Linear-scan the cache jsonl for an entry with the given id field.
/// Best-effort: file-missing or io-error returns `false` (cache miss is
/// indistinguishable from a missing file at this surface, by design).
/// Test-only — the production paths use [`cache_lookup_value`] so they
/// can rehydrate the full J7 stamp-time payload, not just check existence.
#[cfg(test)]
fn cache_contains_id(path: &std::path::Path, id_field: &str, id_value: &str) -> bool {
    cache_lookup_value(path, id_field, id_value).is_some()
}

/// Linear-scan the cache jsonl for an entry with the given id field and
/// return the parsed JSON line. Used by manual-MCP rehydration to
/// reconstruct the J7 stamp-time payload (prompt + candidate +
/// stamp_hash).
///
/// Test-only since R3 P2 — production paths now go through
/// [`cache_lookup_value_with_ttl`] with the configured `cache_ttl_secs`
/// so stale rows return `*_expired_or_unknown` instead of being
/// rehydrated.
#[cfg(test)]
fn cache_lookup_value(
    path: &std::path::Path,
    id_field: &str,
    id_value: &str,
) -> Option<serde_json::Value> {
    cache_lookup_value_with_ttl(path, id_field, id_value, None)
}

/// Codex R3 P2 fix — TTL-aware lookup for production manual judge paths.
///
/// `ttl_secs = None` skips the freshness check (used by tests + the
/// public `cache_lookup_value` shim). When `Some(secs)`, rows whose
/// `stamped_at` (RFC3339) is older than `now - secs` are treated as
/// expired and the lookup returns `None` — manual MCP handlers then
/// surface `*_expired_or_unknown` instead of enqueuing a stale snapshot.
///
/// F4 A4 — delegates to [`read_cache_entries_within_ttl`] for shared
/// TTL semantics with the worker-side `cache_has_id_and_stamp`. Returns
/// the LAST matching live entry so re-mints supersede older rows in
/// the append-only jsonl.
fn cache_lookup_value_with_ttl(
    path: &std::path::Path,
    id_field: &str,
    id_value: &str,
    ttl_secs: Option<u64>,
) -> Option<serde_json::Value> {
    read_cache_entries_within_ttl(path, id_field, id_value, ttl_secs)
        .into_iter()
        .last()
}

/// F4 A4 shared helper — scan an append-only judge cache jsonl and
/// return all live entries matching `(id_field == id_value)`. Used by
/// both [`cache_lookup_value_with_ttl`] (handlers/judge.rs manual MCP)
/// and [`crate::ops::llm_judge_worker::cache_has_id_and_stamp`]
/// (worker-side J5 verification) so the two call sites agree on
/// stale-row semantics.
///
/// **Stale-row contract**: when `ttl_secs = Some(secs)`, a row is
/// considered live iff:
///   1. its `stamped_at` field is present AND parseable as RFC3339, AND
///   2. `(now - stamped_at).as_secs() <= secs`
///
/// Rows with missing / unparseable `stamped_at` while TTL is active
/// are treated as MISSING (skipped) — a strict "stale = absent" rule.
/// This is the F4 A4 alignment vs the v0.27.2 split where the manual
/// MCP path treated such rows as live.
///
/// `ttl_secs = None` skips the freshness check entirely (used by tests
/// + the test-only `cache_lookup_value` shim).
///
/// Returns entries in file order; the cache is append-only so callers
/// wanting the latest re-mint should take `.last()`.
pub(crate) fn read_cache_entries_within_ttl(
    path: &std::path::Path,
    id_field: &str,
    id_value: &str,
    ttl_secs: Option<u64>,
) -> Vec<serde_json::Value> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let now = chrono::Utc::now();
    let mut matches = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) else {
            continue;
        };
        if value.get(id_field).and_then(|v| v.as_str()) != Some(id_value) {
            continue;
        }
        if let Some(ttl) = ttl_secs {
            let Some(stamped_at) = value.get("stamped_at").and_then(|v| v.as_str()) else {
                // F4 A4 — strict "stale = absent": missing stamped_at
                // while TTL active is treated as expired, NOT live.
                continue;
            };
            let Ok(parsed) = chrono::DateTime::parse_from_rfc3339(stamped_at) else {
                // Same rule for unparseable timestamps.
                continue;
            };
            let age = now.signed_duration_since(parsed.with_timezone(&chrono::Utc));
            if age.num_seconds() > ttl as i64 {
                continue;
            }
        }
        matches.push(value);
    }
    matches
}

/// Append a single JSON line to the queue file. Returns the 1-indexed
/// line number AFTER the append (so the first job in an empty file
/// reports `queue_position = 1`).
pub(crate) fn append_jsonl_line(
    path: &std::path::Path,
    value: &serde_json::Value,
) -> std::io::Result<u32> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let line = serde_json::to_string(value)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    // Codex R7 P2 fix — coordinate with the judge drain rotation
    // lockfile (same `<path>.rotation_lock` sibling that
    // `llm_judge_worker::drain_queue` flocks during rename+drain). This
    // ensures appends never land in a `.processing` file the drain is
    // about to remove. R6's per-fd flock was insufficient because it
    // locked the queue inode that drain renames out from under the lock.
    // The drain holds the rotation lock for the ENTIRE rotation+drain
    // window, so this append blocks until the drain finishes.
    use std::io::Write as _;
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
    // Re-open the queue file AFTER acquiring the lock — the drain may
    // have just renamed the previous inode to .processing, in which
    // case we want a fresh queue file.
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(path)?;
    let mut buf = Vec::with_capacity(line.len() + 1);
    buf.extend_from_slice(line.as_bytes());
    buf.push(b'\n');
    let write_result = file.write_all(&buf);
    drop(file);
    // Release rotation lock.
    if let Some(ref f) = rotation_lock_file {
        let _ = unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_UN) };
    }
    drop(rotation_lock_file);
    write_result?;
    let count = match std::fs::read_to_string(path) {
        Ok(text) => text.lines().filter(|l| !l.trim().is_empty()).count() as u32,
        Err(_) => 0,
    };
    Ok(count.max(1))
}

/// Build a `JudgeJob` JSON value from a synthesis-cache entry. The
/// cache row must carry the post-truncation `prompt`, `candidate`, and
/// `stamp_hash` fields that `recall_synthesis.rs` writes; all other
/// fields are best-effort and route through `metadata` on the emitted
/// event. Returns `None` if the cache row is malformed (missing
/// required fields).
fn build_synthesis_judge_job(
    cache_entry: &serde_json::Value,
    source: &str,                       // "AutoSampled" | "ManualMcp"
    judge_model_override: Option<&str>, // Codex R1 P3 fix
) -> Option<serde_json::Value> {
    let synthesis_id = cache_entry.get("synthesis_id")?.as_str()?;
    let query = cache_entry.get("query")?.as_str()?;
    let prompt = cache_entry.get("prompt")?.as_str()?;
    let candidate = cache_entry.get("candidate")?.as_str()?;
    let stamp_hash = cache_entry.get("stamp_hash")?.as_str()?;
    let query_type = cache_entry
        .get("query_type")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let cluster_id = cache_entry.get("cluster_id").and_then(|v| v.as_i64());
    let source_count = cache_entry
        .get("source_count")
        .and_then(|v| v.as_u64())
        .map(|n| n as u32);
    Some(serde_json::json!({
        "kind": "synthesis",
        "surface_id": synthesis_id,
        "concept_id": serde_json::Value::Null,
        "query": query,
        "prompt": prompt,
        "candidate": candidate,
        "stamp_hash": stamp_hash,
        "source": source,
        "query_type": query_type,
        "cluster_id": cluster_id,
        "source_count": source_count,
        // None at serialize time → field omitted (back-compat with rows
        // emitted before v0.27.1 added the override).
        "judge_model_override": judge_model_override,
    }))
}

/// Cap A mirror of [`build_synthesis_judge_job`].
fn build_concept_summary_judge_job(
    cache_entry: &serde_json::Value,
    source: &str,
    judge_model_override: Option<&str>, // Codex R1 P3 fix
) -> Option<serde_json::Value> {
    let concept_summary_id = cache_entry.get("concept_summary_id")?.as_str()?;
    // F4 A1 fix — require non-null/non-missing concept_id so the
    // downstream reader doesn't propagate `None` into the SQL target
    // existence check (which then matches any concept via the
    // `?2 IS NULL OR concept_id = ?2` half). Production writer always
    // populates this; defense-in-depth at the reader closes the gap.
    let concept_id = cache_entry.get("concept_id")?.as_str()?;
    let query = cache_entry.get("query")?.as_str()?;
    let prompt = cache_entry.get("prompt")?.as_str()?;
    let candidate = cache_entry.get("candidate")?.as_str()?;
    let stamp_hash = cache_entry.get("stamp_hash")?.as_str()?;
    let query_type = cache_entry
        .get("query_type")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let cluster_id = cache_entry.get("cluster_id").and_then(|v| v.as_i64());
    let source_count = cache_entry
        .get("source_count")
        .and_then(|v| v.as_u64())
        .map(|n| n as u32);
    Some(serde_json::json!({
        "kind": "concept_summary",
        "surface_id": concept_summary_id,
        "concept_id": concept_id,
        "query": query,
        "prompt": prompt,
        "candidate": candidate,
        "stamp_hash": stamp_hash,
        "source": source,
        "query_type": query_type,
        "cluster_id": cluster_id,
        "source_count": source_count,
        "judge_model_override": judge_model_override,
    }))
}

// ---------------------------------------------------------------------------
// Tests — mirror `concept_summary_feedback.rs` house style.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip the synthesis params shape through JSON. Catches any
    /// future serde rename / tag drift on the params struct itself.
    #[test]
    fn judge_synthesis_params_round_trip_serde() {
        let p = JudgeSynthesisParams {
            synthesis_id: "01HZ-test".into(),
            judge_model_override: Some("gemini-3.1-pro".into()),
        };
        let json = serde_json::to_string(&p).unwrap();
        let back: JudgeSynthesisParams = serde_json::from_str(&json).unwrap();
        assert_eq!(back.synthesis_id, p.synthesis_id);
        assert_eq!(back.judge_model_override, p.judge_model_override);
    }

    /// `judge_model_override` is optional and defaults to `None`.
    #[test]
    fn judge_synthesis_params_override_optional() {
        let json = serde_json::json!({ "synthesis_id": "abc" });
        let parsed: JudgeSynthesisParams =
            serde_json::from_value(json).expect("missing override must parse to None");
        assert_eq!(parsed.synthesis_id, "abc");
        assert!(parsed.judge_model_override.is_none());
    }

    /// JsonSchema derive sanity check — schema must list `synthesis_id`
    /// as required.
    #[test]
    fn judge_synthesis_params_jsonschema_renders() {
        let schema = schemars::schema_for!(JudgeSynthesisParams);
        let value = serde_json::to_value(&schema).expect("schema serializes to JSON");
        let required = value
            .pointer("/required")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let names: std::collections::HashSet<&str> =
            required.iter().filter_map(|v| v.as_str()).collect();
        assert!(
            names.contains("synthesis_id"),
            "expected required field synthesis_id in schema, got {names:?}"
        );
    }

    /// Concept-summary mirror of the round-trip test.
    #[test]
    fn judge_concept_summary_params_round_trip_serde() {
        let p = JudgeConceptSummaryParams {
            concept_summary_id: "01HZ-cs".into(),
            judge_model_override: None,
        };
        let json = serde_json::to_string(&p).unwrap();
        let back: JudgeConceptSummaryParams = serde_json::from_str(&json).unwrap();
        assert_eq!(back.concept_summary_id, p.concept_summary_id);
        assert!(back.judge_model_override.is_none());
    }

    /// Concept-summary schema sanity check.
    #[test]
    fn judge_concept_summary_params_jsonschema_renders() {
        let schema = schemars::schema_for!(JudgeConceptSummaryParams);
        let value = serde_json::to_value(&schema).expect("schema serializes to JSON");
        let required = value
            .pointer("/required")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let names: std::collections::HashSet<&str> =
            required.iter().filter_map(|v| v.as_str()).collect();
        assert!(
            names.contains("concept_summary_id"),
            "expected required field concept_summary_id in schema, got {names:?}"
        );
    }

    /// Output JSON shape — `enqueued = false` + `skipped_reason` round-trips.
    #[test]
    fn judge_enqueue_result_skipped_to_json() {
        let r = JudgeEnqueueResult {
            enqueued: false,
            queue_position: None,
            skipped_reason: Some("judge_disabled".into()),
        };
        let v = r.to_json();
        assert_eq!(v.get("enqueued").and_then(|x| x.as_bool()), Some(false));
        assert_eq!(
            v.get("skipped_reason").and_then(|x| x.as_str()),
            Some("judge_disabled")
        );
    }

    /// Markdown-mode (MCP compact / CLI) renders the skipped reason.
    #[test]
    fn judge_enqueue_result_skipped_to_markdown() {
        let r = JudgeEnqueueResult {
            enqueued: false,
            queue_position: None,
            skipped_reason: Some("synthesis_id_expired_or_unknown".into()),
        };
        let md = IntoMarkdown::to_markdown(&r);
        assert!(md.contains("skipped"), "got: {md}");
        assert!(md.contains("synthesis_id_expired_or_unknown"), "got: {md}");
    }

    /// Markdown-mode for a successful enqueue includes the queue position.
    #[test]
    fn judge_enqueue_result_enqueued_to_markdown() {
        let r = JudgeEnqueueResult {
            enqueued: true,
            queue_position: Some(7),
            skipped_reason: None,
        };
        let md = IntoMarkdown::to_markdown(&r);
        assert!(md.contains("enqueued"), "got: {md}");
        assert!(md.contains("7"), "got: {md}");
    }

    /// Cache lookup is a linear scan; matching id returns true,
    /// non-matching returns false.
    #[test]
    fn cache_contains_id_basic() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("cache.jsonl");
        std::fs::write(
            &path,
            "{\"synthesis_id\":\"abc\",\"x\":1}\n\
             {\"synthesis_id\":\"def\"}\n\
             not-json-line\n\
             \n",
        )
        .unwrap();
        assert!(cache_contains_id(&path, "synthesis_id", "abc"));
        assert!(cache_contains_id(&path, "synthesis_id", "def"));
        assert!(!cache_contains_id(&path, "synthesis_id", "ghi"));
        assert!(!cache_contains_id(&path, "concept_summary_id", "abc"));
    }

    /// Missing cache file → cache-miss (false), not an error.
    #[test]
    fn cache_contains_id_missing_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("nonexistent.jsonl");
        assert!(!cache_contains_id(&path, "synthesis_id", "abc"));
    }

    /// `append_jsonl_line` creates the file + parent dirs and reports
    /// the 1-indexed line position.
    #[test]
    fn append_jsonl_line_increments_position() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("nested").join("queue.jsonl");
        let p1 = append_jsonl_line(&path, &serde_json::json!({"x": 1})).unwrap();
        let p2 = append_jsonl_line(&path, &serde_json::json!({"x": 2})).unwrap();
        assert_eq!(p1, 1);
        assert_eq!(p2, 2);
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(text.lines().count(), 2);
    }
}
