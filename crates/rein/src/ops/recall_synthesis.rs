//! Cap B: recall-time multi-memory synthesis.
//!
//! When `synthesize=true` is passed to `rein_recall`, the top-N results are
//! fed to an LLM that synthesizes a concise narrative directly answering the
//! query. The result is returned as `RecallSynthesisOutcome` alongside the
//! normal results list.
//!
//! The LLM call uses `raw_text_with_prompt` (prose mode, NOT JSON mode) and
//! carries an explicit hallucination guardrail: "synthesize from the provided
//! memories only; do not invent facts."

use crate::config::ReinConfig;
use crate::extract::llm::{strip_code_fences, ExtractorKind};
use crate::ops::concept_summary::create_concept_summary_extractor;
use crate::search::recall::RecallResult;
use crate::store::adaptive::{
    synthesis_bucket_key, AdaptiveState, SYNTHESIS_COLD_START_N, SYNTHESIS_USEFUL_RATE_THRESHOLD,
};
use crate::types::ReinResult;
use serde::Serialize;

const SYNTHESIS_SYSTEM_PROMPT: &str = "\
You are a memory synthesizer for a personal knowledge system. \
Given a search query and a set of retrieved memories, produce a concise \
3-to-6-sentence narrative that directly answers the query using ONLY the \
provided memories. Do not invent facts, do not draw on knowledge outside \
the provided memories. If the memories are contradictory, note the \
contradiction explicitly. Output plain prose only — no preamble, no bullet \
points, no code fences, no headings.\n\n\
CRITICAL — synthesize from the provided memories only; do not invent facts \
not present in the memory list below.\n\n\
After each sentence or clause that draws from a specific memory, insert the \
source marker [#k] where k is the 1-based rank of the source memory in the \
input list. If a sentence draws from multiple memories, list all markers, \
e.g. [#1][#3]. Place markers at the end of the relevant sentence or clause, \
before the period or comma. If a sentence is purely connective and doesn't \
make a sourced claim (e.g. \"However,\" or \"Overall,\"), omit the marker.";

fn is_false(b: &bool) -> bool {
    !*b
}

/// A single inline citation extracted from the synthesized prose.
///
/// Citations point a UI badge at the **char offset** in the cleaned prose
/// (after `[#k]` markers were stripped) where the cited claim ends. The
/// offset is in `chars()`, NOT bytes — JS strings are UTF-16, Rust strings
/// are UTF-8, and the only common ground is character count. CJK content
/// (where 1 char = 3 bytes UTF-8 = 1 UTF-16 code unit) is the canonical
/// case where byte offsets would silently desync the two stacks.
///
/// Multiple citations sharing the same `span_end` (e.g. the LLM emitted
/// `[#1][#3]` together) keep their distinct ranks — the UI groups them
/// visually but tracks them as separate badges so each is clickable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Citation {
    /// 1-based rank of the source memory (e.g. 3 means the 3rd result in
    /// the input list, matching `RecallCard rank={idx + 1}` in the GUI).
    pub rank: usize,
    /// Char offset in the **clean** prose (after marker removal) where the
    /// cited claim ends. The UI inserts the badge at this offset using
    /// char-aware string slicing — never byte indexing.
    pub span_end: usize,
}

/// Outcome of a recall-time synthesis attempt.
///
/// Serializes to match the committed TypeScript `RecallSynthesisOutcome`
/// interface in `gui/src/api/types.ts`:
/// ```ts
/// export interface RecallSynthesisOutcome {
///   synthesis?: string;
///   query: string;
///   source_count: number;
///   model_used?: string;
///   skipped_disabled?: boolean;
///   skipped_adaptive_decision?: boolean;
///   skipped_no_llm?: boolean;
///   skipped_too_few_results?: boolean;
///   citations?: Citation[];
///   synthesis_id?: string;
/// }
/// export interface Citation {
///   rank: number;     // 1-based
///   span_end: number; // char offset in clean prose
/// }
/// ```
#[derive(Debug, Clone, Serialize)]
pub struct RecallSynthesisOutcome {
    /// The synthesized narrative. `None` when synthesis was skipped or failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub synthesis: Option<String>,
    /// The original query string (echoed for UI correlation).
    pub query: String,
    /// Number of results fed to the LLM (0 when skipped before LLM call).
    pub source_count: usize,
    /// Model identifier, if determinable at call time. Reserved for future use.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_used: Option<String>,
    /// `true` when `[ars].recall_synthesis_enabled = false` (operator opted
    /// out globally). Mutually exclusive with `skipped_adaptive_decision`.
    #[serde(skip_serializing_if = "is_false")]
    pub skipped_disabled: bool,
    /// v0.26 D direction: `true` when the per-query adaptive decision
    /// (`decide_synthesize`) returned `Skip(AdaptiveDecision)` — i.e. the
    /// global flag was on, but the cluster's `useful_rate` is below
    /// `SYNTHESIS_USEFUL_RATE_THRESHOLD`. Distinct from `skipped_disabled`
    /// (operator-off) so the GUI can surface "adaptive declined" as a
    /// separate state from "operator off". Mutually exclusive with
    /// `skipped_disabled` in practice.
    #[serde(default, skip_serializing_if = "is_false")]
    pub skipped_adaptive_decision: bool,
    /// `true` when no LLM provider is configured or the API key is absent.
    #[serde(skip_serializing_if = "is_false")]
    pub skipped_no_llm: bool,
    /// `true` when `results.len() < [ars].recall_synthesis_min_results`.
    #[serde(skip_serializing_if = "is_false")]
    pub skipped_too_few_results: bool,
    /// Inline citations parsed out of the LLM's `[#k]` markers. Empty when
    /// the LLM emitted no markers (older models / non-compliance) or when
    /// synthesis was skipped. Char offsets are aligned with the cleaned
    /// `synthesis` field — markers are removed before this is computed.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub citations: Vec<Citation>,
    /// v0.26 D direction: ULID identifying this synthesis output.
    /// Populated **only** when synthesis succeeded (i.e. `synthesis.is_some()`
    /// after a successful LLM call); `None` on every skipped path
    /// (`skipped_disabled`, `skipped_adaptive_decision`, `skipped_no_llm`,
    /// `skipped_too_few_results`) AND when the LLM call returned an empty
    /// or error response. Clients pass this back via `rein_feedback`
    /// `SynthesisInteraction` events to close the M1 feedback loop.
    /// Clients that receive a synthesis with `synthesis_id = None` MUST
    /// NOT post interaction events.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub synthesis_id: Option<String>,
}

/// Reason a per-query synthesize gate skipped. Used inside
/// [`SynthesizeDecision::Skip`].
///
/// The two reasons are reported through different `RecallSynthesisOutcome`
/// flags so the GUI / observability surfaces can distinguish "operator off"
/// from "the adaptive engine has learned this cluster doesn't benefit from
/// synthesis". Without the split they collapse to "skipped" and the
/// drift-detection signal is lost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipReason {
    /// `[ars].recall_synthesis_enabled = false` — operator opted out
    /// globally. Maps to `outcome.skipped_disabled = true`.
    OperatorDisabled,
    /// Per-query adaptive decision: cluster's `useful_rate` below
    /// `SYNTHESIS_USEFUL_RATE_THRESHOLD`. Maps to
    /// `outcome.skipped_adaptive_decision = true`.
    AdaptiveDecision,
}

/// Per-query adaptive synthesize decision returned by [`decide_synthesize`].
///
/// `Yes` flows into the existing synthesis path; `Skip(reason)` short-circuits
/// with the matching `outcome.skipped_*` flag set per [`SkipReason`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SynthesizeDecision {
    Yes,
    Skip(SkipReason),
}

/// Per-query synthesize decision used by `run_recall_synthesis`.
///
/// Cold-start fallback policy: when `global_enabled` is `true` but the
/// adaptive state cannot disambiguate (no `cluster_id`, no adaptive state
/// snapshot, no per-cluster bucket yet, or per-cluster events <
/// `cold_start_n`), the function returns [`SynthesizeDecision::Yes`] — i.e.
/// "behave like v0.25.x and let synthesis run". This is mandatory per
/// contract §8 invariant 4: the per-query gate must NEVER silently skip
/// synthesis just because adaptive data is missing.
///
/// When `global_enabled` is `false`, the function ALWAYS returns
/// `Skip(OperatorDisabled)` — operator override wins over any adaptive
/// signal.
///
/// Bucket key for `by_cluster.get(...)` is built via the canonical
/// `synthesis_bucket_key` helper from `store::adaptive`, matching the
/// `"{cid}|{qtype}"` format documented on
/// `SynthesisFeedbackState.by_cluster`. Both sides reuse the same helper
/// so they cannot drift; mismatch would produce a silent dead-code path.
///
/// Pure function — no IO, no allocation beyond the cluster-key string.
pub fn decide_synthesize(
    global_enabled: bool,
    cluster_id: Option<i64>,
    query_type: &str,
    adaptive_state: Option<&AdaptiveState>,
    cold_start_n: u64,
) -> SynthesizeDecision {
    // Operator override wins. Even with rich adaptive data, if the operator
    // disabled the global flag, the synthesis path is off.
    if !global_enabled {
        return SynthesizeDecision::Skip(SkipReason::OperatorDisabled);
    }

    // Cold-start ladder: each missing piece falls back to "Yes" (matches
    // pre-adaptive v0.25.x behavior). Per contract §8 invariant 4 the gate
    // must NEVER skip silently just because the per-query data is missing.
    //
    // `cluster_id = None` short-circuits to Yes — the synthesis bucket
    // helper supports a `-1` "no cluster" key, but we deliberately do NOT
    // route the gate through it: the global `-1` bucket aggregates events
    // across ALL queries that lacked a cluster (different queries, different
    // characteristics), so its `useful_rate` is too noisy to gate
    // individual recalls on. The global bucket is preserved for the
    // consumer-side `/api/adaptive` rollup, not for runtime gating.
    let Some(cid) = cluster_id else {
        return SynthesizeDecision::Yes;
    };
    let Some(state) = adaptive_state else {
        return SynthesizeDecision::Yes;
    };
    let Some(synth_state) = state.synthesis_feedback_stats.as_ref() else {
        return SynthesizeDecision::Yes;
    };
    let key = synthesis_bucket_key(Some(cid), query_type);
    let Some(cluster) = synth_state.by_cluster.get(&key) else {
        return SynthesizeDecision::Yes;
    };
    if cluster.viewed_count < cold_start_n {
        return SynthesizeDecision::Yes;
    }

    // Per-cluster gate: skip if learned useful_rate is below the bootstrap
    // threshold (cluster has acquired enough events to disagree with the
    // global default).
    if cluster.useful_rate >= SYNTHESIS_USEFUL_RATE_THRESHOLD {
        SynthesizeDecision::Yes
    } else {
        SynthesizeDecision::Skip(SkipReason::AdaptiveDecision)
    }
}

/// Run recall-time synthesis over `results` for `query`.
///
/// Returns `None` when synthesis was not requested (`synthesize` is `None` or
/// `Some(false)`). Returns `Some(RecallSynthesisOutcome)` when requested — the
/// outcome's `skipped_*` flags explain why synthesis was not produced (if any).
///
/// v0.26 D direction: `adaptive_state` carries the per-cluster
/// `SynthesisFeedbackState` consumed by [`decide_synthesize`]. Pass `None`
/// for cold-start callers (e.g. CLI, tests) — the gate degrades gracefully
/// to v0.25.x behavior. Production should pass
/// `AdaptiveState::restore_snapshot(store.conn()).unwrap_or_default()`
/// from inside the recall handler so synthesis sees the same adaptive
/// snapshot used elsewhere in the request.
///
/// `extractor_override` is only used by tests (feature `test-support`); pass
/// `None` in production.
pub fn run_recall_synthesis(
    results: &[RecallResult],
    query: &str,
    config: &ReinConfig,
    synthesize: Option<bool>,
    adaptive_state: Option<&AdaptiveState>,
    extractor_override: Option<ExtractorKind>,
) -> Option<RecallSynthesisOutcome> {
    if synthesize != Some(true) {
        return None;
    }

    let source_count = results.len();
    let mut outcome = RecallSynthesisOutcome {
        synthesis: None,
        query: query.to_string(),
        source_count,
        model_used: None,
        skipped_disabled: false,
        skipped_adaptive_decision: false,
        skipped_no_llm: false,
        skipped_too_few_results: false,
        citations: Vec::new(),
        synthesis_id: None,
    };

    // v0.26 D direction: per-query adaptive gate. Replaces the standalone
    // `if !config.ars.recall_synthesis_enabled { … }` check so operator-off
    // and per-cluster-decline route through the same decision surface.
    //
    // Cluster id source: first result's `cluster_id`. Top-1 is a reasonable
    // proxy because results are already ranked by score, and per-cluster
    // synthesis-quality signals are most relevant where the strongest
    // candidate sits.
    //
    // query_type: hardcoded to `"Semantic"` for v0.26.0. This matches the
    // capitalisation B_FEEDBACK_EVENT writes into the
    // `SynthesisInteractionPayload.metadata.query_type` field (verified in
    // `store/adaptive.rs` test fixtures; see e.g.
    // `recompute_synthesis_feedback_*` tests using `synthesis_bucket_key(_,
    // "Semantic")`). The bucket key is built by both sides via
    // `synthesis_bucket_key`, so the strings MUST match exactly — using
    // lowercase here would silently miss every per-cluster bucket and turn
    // the gate into dead code. Note this is DIFFERENT from
    // `QueryType::Semantic.to_string()` (search/classify.rs:412), which
    // produces lowercase `"semantic"` for the route name — that surface
    // is unrelated to the synthesis bucket key.
    //
    // TODO v0.26.1: wire query classifier output through RecallResult /
    // recall handler so the bucket key reflects the real query type. The
    // canonical mapping is `QueryType -> "Semantic"|"Episodic"|...` (each
    // variant capitalised to match the v0.26.0 payload convention).
    let cluster_id = results
        .first()
        .and_then(|r| r.memory.cluster_id)
        .map(|c| c as i64);
    let query_type = "Semantic"; // TODO v0.26.1: wire query classifier output
    match decide_synthesize(
        config.ars.recall_synthesis_enabled,
        cluster_id,
        query_type,
        adaptive_state,
        SYNTHESIS_COLD_START_N,
    ) {
        SynthesizeDecision::Yes => { /* fall through to synthesis path */ }
        SynthesizeDecision::Skip(SkipReason::OperatorDisabled) => {
            outcome.skipped_disabled = true;
            return Some(outcome);
        }
        SynthesizeDecision::Skip(SkipReason::AdaptiveDecision) => {
            outcome.skipped_adaptive_decision = true;
            return Some(outcome);
        }
    }

    let min_results = config.ars.recall_synthesis_min_results;
    if source_count < min_results {
        outcome.skipped_too_few_results = true;
        return Some(outcome);
    }

    let extractor = match extractor_override {
        Some(e) => e,
        None => match create_concept_summary_extractor(config) {
            Some(e) => e,
            None => {
                outcome.skipped_no_llm = true;
                return Some(outcome);
            }
        },
    };

    // Cap B safety: bound the prompt size by the same `max_input_chars`
    // safeguard the extractor would apply on `extract`/`raw_with_prompt`
    // through `prepare_with_context_for_kind`. Without this, a caller with
    // `synthesize=true` + `limit=200` + 100KB memories could send a
    // multi-megabyte payload to the LLM provider — costly, slow, and
    // possibly over the model's context window. Codex audit Round 2 P2.
    let max_chars = crate::extract::llm::resolve_max_input_for_kind(config, &extractor);
    // Codex R2 G4: use `included_count` (the actual number of memory
    // blocks the LLM sees in the prompt after truncation) — not the
    // pre-truncation `source_count` — as the citation max-rank. Without
    // this, a marker like `[#10]` is accepted even when truncation only
    // included the first 5 memories, so the UI would render an inline
    // reference to a source the LLM never saw.
    let (prompt, included_count) =
        build_synthesis_prompt_with_count(results, query, max_chars);
    match call_synthesis_llm_sync(&extractor, &prompt) {
        Ok(raw) => {
            let text = strip_code_fences(&raw).trim().to_string();
            if !text.is_empty() {
                // Strip [#k] markers and extract citations. Any marker
                // pointing past `included_count` is dropped silently
                // (defensive — the LLM should never emit out-of-range
                // markers under the system prompt, but compliance is not
                // guaranteed).
                let (clean, citations) = extract_citations(&text, included_count);
                outcome.synthesis = Some(clean);
                outcome.citations = citations;
                // v0.26 D direction: stamp a fresh ULID **only** on a
                // successful synthesis. Empty LLM output (text was empty
                // post-strip) leaves `synthesis_id = None` so clients
                // know not to emit interaction feedback. Per contract
                // §8 invariant 9.
                outcome.synthesis_id = Some(ulid::Ulid::new().to_string());
            }
        }
        Err(e) => {
            tracing::warn!(
                query = %query,
                error = %e,
                "recall_synthesis: LLM call failed (non-fatal, returning results without synthesis)"
            );
        }
    }

    Some(outcome)
}

const TRUNCATION_NOTICE: &str =
    "\n[…remaining memories truncated to fit the LLM input budget]\n";
const FOOTER: &str =
    "\nNow produce the concise narrative synthesis based solely on the memories above.";

/// Strip `[#k]` source markers from synthesized prose and return the
/// citation list keyed by char offset into the cleaned output.
///
/// Marker grammar (deliberately strict — any deviation drops the marker
/// silently rather than corrupting the clean prose):
///   `[` `#` <one or more ASCII digits> `]`
///
/// The offset returned for each citation is the **char** count of the
/// cleaned prose at the position the marker appeared. Consecutive markers
/// like `[#1][#3]` collapse to two citations both at the same `span_end`.
/// Invalid markers — non-numeric body, rank `0`, rank > `max_rank`, or a
/// missing `]` — are passed through unchanged into the cleaned output so
/// the LLM's prose isn't silently mutilated by edge cases. This is a
/// pure function (no allocations beyond the output buffers, no IO).
///
/// Example: `"Foo[#1]."` → `("Foo.", [Citation { rank: 1, span_end: 3 }])`
/// CJK example: `"中文[#1]。"` → `("中文。", [Citation { rank: 1, span_end: 2 }])`
/// — note `span_end` is **char** count, not bytes.
/// `pub` so the v0.25.2 A3 `rein-eval synthesis` binary can mirror
/// production by stripping markers from the raw LLM output before
/// scoring (Codex R2 G5 — without this, `treatment_summary` /
/// `treatment_length` carry literal `[#k]` text that the production UI
/// would never render, inflating length and risking spurious keyword
/// hits on numeric tokens inside markers).
pub fn extract_citations(prose: &str, max_rank: usize) -> (String, Vec<Citation>) {
    let mut clean = String::with_capacity(prose.len());
    let mut citations: Vec<Citation> = Vec::new();
    // Char count of `clean` so far. Tracked separately from `clean.len()`
    // because the latter is a byte length and we need a **char** offset
    // for the JS frontend to slice without UTF-16 conversion gymnastics.
    let mut clean_chars: usize = 0;

    let chars: Vec<char> = prose.chars().collect();
    let n = chars.len();
    let mut i = 0;

    while i < n {
        let c = chars[i];
        // Try to parse a marker starting at i: `[` `#` <digits> `]`.
        if c == '[' && i + 3 < n && chars[i + 1] == '#' {
            // Walk digits from i+2.
            let digit_start = i + 2;
            let mut j = digit_start;
            while j < n && chars[j].is_ascii_digit() {
                j += 1;
            }
            // Need at least one digit + a closing ']'.
            if j > digit_start && j < n && chars[j] == ']' {
                let digits: String = chars[digit_start..j].iter().collect();
                // `digits` is non-empty ASCII digits → parse cannot fail
                // for reasonable lengths. Use saturating fallback to
                // 0 (which is dropped by the rank filter) for paranoid
                // multi-MB digit strings rather than panicking.
                let rank = digits.parse::<usize>().unwrap_or(0);
                if rank >= 1 && rank <= max_rank {
                    citations.push(Citation {
                        rank,
                        span_end: clean_chars,
                    });
                }
                // Whether the rank was valid or not, swallow the marker
                // so the user-visible prose stays clean. Out-of-range
                // markers are quality issues the user should not see.
                i = j + 1;
                continue;
            }
            // Fall through: malformed marker, treat `[` as literal.
        }
        clean.push(c);
        clean_chars += 1;
        i += 1;
    }

    (clean, citations)
}

/// Build the synthesis prompt with priority-aware truncation.
///
/// `max_chars = 0` means "no cap" (used by Mock in tests). Otherwise the
/// total prompt length stays within `max_chars` chars: top-ranked memories
/// are included whole, and the first memory that would overflow is
/// truncated mid-content + a `TRUNCATION_NOTICE` appended; remaining
/// memories are dropped. The footer always appears at the end.
///
/// Query is itself capped to `max(max_chars / 4, QUERY_BUDGET_FLOOR)` so a
/// runaway long query (e.g. multi-KB accidental paste) cannot starve the
/// memory body and bypass the overall cap (Codex audit Round 3 P2). Final
/// defensive `take(max_chars)` is applied as a safety net guaranteeing
/// the total prompt never exceeds the budget regardless of edge cases in
/// the reservation arithmetic.
///
/// `pub` so the v0.25.1 A3 `rein-eval synthesis` binary can construct the
/// exact same prompt that production uses — eval-vs-production drift here
/// would invalidate the McNemar comparison.
///
/// Backward-compat shim: returns just the prompt string. New callers that
/// need to validate citations against the actually-included memory blocks
/// (Codex R2 G4 — `[#k]` markers past the truncation point can be silently
/// dropped) should call [`build_synthesis_prompt_with_count`] directly.
pub fn build_synthesis_prompt(results: &[RecallResult], query: &str, max_chars: usize) -> String {
    build_synthesis_prompt_with_count(results, query, max_chars).0
}

/// Same as [`build_synthesis_prompt`] but also returns `included_count` —
/// the number of memory blocks (1-based ranks 1..=N) that the LLM
/// actually sees in the prompt. When prompt truncation drops trailing
/// memories, `included_count < results.len()`. Citation parsing should
/// pass `included_count` (not `results.len()`) as `max_rank` so the LLM
/// can't legitimately cite a source it never saw.
pub fn build_synthesis_prompt_with_count(
    results: &[RecallResult],
    query: &str,
    max_chars: usize,
) -> (String, usize) {
    // Query budget: cap query so it cannot consume the whole prompt
    // budget. Floor of QUERY_BUDGET_FLOOR comfortably fits typical
    // natural-language queries (~50-200 chars) without truncation.
    const QUERY_BUDGET_DIVISOR: usize = 4;
    const QUERY_BUDGET_FLOOR: usize = 256;
    const QUERY_TRUNC_NOTICE: &str = " […query truncated for prompt budget]";

    let query_chars = query.chars().count();
    let (query_owned, query_truncated): (String, bool) = if max_chars == 0
        || query_chars <= QUERY_BUDGET_FLOOR
    {
        (query.to_string(), false)
    } else {
        let budget = (max_chars / QUERY_BUDGET_DIVISOR).max(QUERY_BUDGET_FLOOR);
        if query_chars > budget {
            (query.chars().take(budget).collect(), true)
        } else {
            (query.to_string(), false)
        }
    };

    let header = if query_truncated {
        format!(
            "Query: {query_owned}{QUERY_TRUNC_NOTICE}\n\nMemories (ordered by relevance, most relevant first):\n"
        )
    } else {
        format!(
            "Query: {query_owned}\n\nMemories (ordered by relevance, most relevant first):\n"
        )
    };

    if max_chars == 0 {
        let mut buf = String::with_capacity(
            header.len()
                + results
                    .iter()
                    .map(|r| r.memory.content.len() + r.memory.topic.len() + 32)
                    .sum::<usize>()
                + FOOTER.len(),
        );
        buf.push_str(&header);
        for (i, r) in results.iter().enumerate() {
            push_memory_block(&mut buf, i + 1, r);
        }
        buf.push_str(FOOTER);
        return (buf, results.len());
    }

    // Reserve headroom for header + footer + the truncation notice (only
    // appended if we actually truncate, but reserving unconditionally
    // keeps the budget arithmetic simple and never overshoots `max_chars`).
    let reserved = header.chars().count()
        + FOOTER.chars().count()
        + TRUNCATION_NOTICE.chars().count();
    let body_budget = max_chars.saturating_sub(reserved);

    let mut buf = String::with_capacity(max_chars + 32);
    buf.push_str(&header);

    let mut used: usize = 0;
    let mut truncated = false;
    // Codex R2 G4: `included_count` tracks how many memory blocks the
    // LLM actually sees in the prompt. Updated AFTER the header is
    // pushed (because that's the marker the LLM keys citations on); a
    // memory whose header didn't fit is NOT included even though its
    // index existed in `results`.
    let mut included_count: usize = 0;

    for (i, r) in results.iter().enumerate() {
        let block_header = format!("\n[{}] Topic: {}\n", i + 1, r.memory.topic);
        let header_chars = block_header.chars().count();
        if used + header_chars >= body_budget {
            // No room even for this memory's header line — stop.
            truncated = true;
            break;
        }
        buf.push_str(&block_header);
        used += header_chars;
        included_count = i + 1;

        let content_chars = r.memory.content.chars().count();
        let trailing_newline = if r.memory.content.ends_with('\n') { 0 } else { 1 };
        let needed = content_chars + trailing_newline;
        let remaining = body_budget.saturating_sub(used);

        if needed <= remaining {
            buf.push_str(&r.memory.content);
            if trailing_newline == 1 {
                buf.push('\n');
            }
            used += needed;
        } else {
            // Truncate this memory's content and stop adding more memories.
            // `remaining` may be 0 here, in which case we still want to mark
            // truncation so the LLM knows facts were dropped. The block
            // header was already pushed, so this memory IS counted in
            // `included_count` — the LLM sees its rank and partial content.
            let take = remaining.saturating_sub(trailing_newline);
            if take > 0 {
                let partial: String = r.memory.content.chars().take(take).collect();
                buf.push_str(&partial);
            }
            buf.push('\n');
            truncated = true;
            break;
        }
    }

    if truncated {
        buf.push_str(TRUNCATION_NOTICE);
    }
    buf.push_str(FOOTER);

    // Final defensive cap — guarantees the prompt never exceeds
    // `max_chars` even if a future change to the budget arithmetic above
    // miscalculates a corner case (e.g. floor > max_chars / 4 when
    // max_chars is itself smaller than QUERY_BUDGET_FLOOR + reserved).
    // Truncating from the end may drop the footer; that is acceptable as
    // a last-resort safety net since the LLM still receives a valid
    // header + memory body and will produce *some* answer rather than the
    // call being rejected for over-length.
    if buf.chars().count() > max_chars {
        buf = buf.chars().take(max_chars).collect();
    }
    (buf, included_count)
}

fn push_memory_block(buf: &mut String, index: usize, r: &RecallResult) {
    buf.push_str(&format!("\n[{index}] Topic: {}\n", r.memory.topic));
    buf.push_str(&r.memory.content);
    if !r.memory.content.ends_with('\n') {
        buf.push('\n');
    }
}

/// Call the configured LLM extractor to produce the synthesis narrative.
///
/// `pub` so the v0.25.1 A3 `rein-eval synthesis` binary can drive the same
/// LLM bridge production uses (system prompt + prose-mode `raw_text_with_prompt`
/// path), keeping eval and production exercising identical request shapes.
pub fn call_synthesis_llm_sync(extractor: &ExtractorKind, prompt: &str) -> ReinResult<String> {
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        tokio::task::block_in_place(|| {
            handle.block_on(async {
                extractor
                    .raw_text_with_prompt(SYNTHESIS_SYSTEM_PROMPT, prompt)
                    .await
            })
        })
    } else {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| {
                crate::types::ReinError::Config(format!("failed to build tokio runtime: {e}"))
            })?;
        rt.block_on(async {
            extractor
                .raw_text_with_prompt(SYNTHESIS_SYSTEM_PROMPT, prompt)
                .await
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ReinConfig;

    fn make_memory(i: usize) -> crate::types::Memory {
        use crate::types::{Importance, MemoryLayer, MemoryStatus, Source};
        crate::types::Memory {
            id: format!("mem-{i}"),
            layer: MemoryLayer::LTM,
            topic: format!("topic-{i}"),
            summary: format!("summary {i}"),
            content: format!("content of memory {i}: important fact about the subject"),
            keywords: vec![],
            importance: Importance::Medium,
            source: Source::Manual,
            strength: 1.0,
            decay_lambda: 0.06,
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
            status: MemoryStatus::default(),
            embedding: None,
            tier: crate::types::MemoryTier::Warm,
            cluster_id: None,
            archival_summary: None,
            archival_summary_at: None,
            archival_summary_version: None,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            last_accessed: chrono::Utc::now(),
        }
    }

    fn make_results(n: usize) -> Vec<RecallResult> {
        (0..n)
            .map(|i| RecallResult {
                memory: make_memory(i),
                score: 0.9 - (i as f32 * 0.05),
                confidence: 0.8,
                sources_hit: 2,
                evidence_count: 0,
                evidence_preview: vec![],
                archival_summary: None,
            })
            .collect()
    }

    #[test]
    fn not_requested_returns_none() {
        let config = ReinConfig::default();
        let results = make_results(5);
        assert!(
            run_recall_synthesis(&results, "test query", &config, None, None, None).is_none(),
            "None synthesize param → None outcome"
        );
        assert!(
            run_recall_synthesis(&results, "test query", &config, Some(false), None, None).is_none(),
            "Some(false) synthesize param → None outcome"
        );
    }

    #[test]
    fn skipped_disabled_when_feature_off() {
        let config = ReinConfig::default(); // recall_synthesis_enabled = false
        let results = make_results(5);
        let outcome =
            run_recall_synthesis(&results, "test", &config, Some(true), None, None).unwrap();
        assert!(outcome.skipped_disabled, "feature off → skipped_disabled");
        assert!(!outcome.skipped_adaptive_decision);
        assert!(!outcome.skipped_no_llm);
        assert!(!outcome.skipped_too_few_results);
        assert!(outcome.synthesis.is_none());
        assert!(
            outcome.synthesis_id.is_none(),
            "skipped paths leave synthesis_id = None"
        );
        assert_eq!(outcome.query, "test");
        assert_eq!(outcome.source_count, 5);
    }

    #[test]
    fn skipped_too_few_results() {
        let mut config = ReinConfig::default();
        config.ars.recall_synthesis_enabled = true;
        config.ars.recall_synthesis_min_results = 3;
        let results = make_results(2); // < 3
        let outcome =
            run_recall_synthesis(&results, "test", &config, Some(true), None, None).unwrap();
        assert!(
            outcome.skipped_too_few_results,
            "2 results < min 3 → skipped_too_few_results"
        );
        assert!(!outcome.skipped_disabled);
        assert!(!outcome.skipped_adaptive_decision);
        assert!(!outcome.skipped_no_llm);
        assert!(outcome.synthesis_id.is_none());
    }

    #[test]
    fn skipped_no_llm_when_provider_is_none() {
        let mut config = ReinConfig::default();
        config.ars.recall_synthesis_enabled = true;
        config.ars.recall_synthesis_min_results = 3;
        // extract.provider defaults to "google" but api_key is None →
        // create_concept_summary_extractor returns None → skipped_no_llm
        config.extract.provider = "none".to_string();
        let results = make_results(5); // >= 3
        let outcome =
            run_recall_synthesis(&results, "test", &config, Some(true), None, None).unwrap();
        assert!(outcome.skipped_no_llm, "no provider → skipped_no_llm");
        assert!(!outcome.skipped_disabled);
        assert!(!outcome.skipped_adaptive_decision);
        assert!(!outcome.skipped_too_few_results);
        assert!(outcome.synthesis_id.is_none());
    }

    #[cfg(feature = "test-support")]
    #[test]
    fn success_with_mock_extractor() {
        use crate::extract::llm::MockExtractor;

        let mut config = ReinConfig::default();
        config.ars.recall_synthesis_enabled = true;
        config.ars.recall_synthesis_min_results = 3;
        let results = make_results(5);
        let mock =
            ExtractorKind::Mock(MockExtractor::with_fixed_response("Synthesized narrative."));
        let outcome =
            run_recall_synthesis(&results, "test query", &config, Some(true), None, Some(mock))
                .unwrap();
        assert!(!outcome.skipped_disabled);
        assert!(!outcome.skipped_adaptive_decision);
        assert!(!outcome.skipped_no_llm);
        assert!(!outcome.skipped_too_few_results);
        assert_eq!(
            outcome.synthesis.as_deref(),
            Some("Synthesized narrative."),
            "synthesis text matches mock response"
        );
        assert_eq!(outcome.source_count, 5);
        assert_eq!(outcome.query, "test query");
        // v0.26 D direction: successful synthesis stamps a ULID.
        let synth_id = outcome
            .synthesis_id
            .as_deref()
            .expect("successful synthesis must populate synthesis_id");
        assert_eq!(
            synth_id.len(),
            26,
            "synthesis_id is a ULID (Crockford base32, 26 chars); got {synth_id:?}"
        );
    }

    // ── prompt cap (Round 2 P2 regression coverage) ──────────────────────────

    /// `max_chars = 0` keeps the legacy "no cap" behavior — needed so test
    /// callers using `MockExtractor` (which has no real input limit) still
    /// see the full prompt, and so callers with a 1M-context Gemini and an
    /// explicit `max_input_chars = 0` opt-out are not silently truncated
    /// behind their back.
    #[test]
    fn build_synthesis_prompt_no_cap_includes_all_content() {
        let results = make_results(3);
        let prompt = build_synthesis_prompt(&results, "q", 0);
        assert!(prompt.contains("content of memory 0"));
        assert!(prompt.contains("content of memory 1"));
        assert!(prompt.contains("content of memory 2"));
        assert!(!prompt.contains("truncated"), "no cap → no truncation notice");
    }

    /// When the budget comfortably fits everything, no truncation notice
    /// should appear.
    #[test]
    fn build_synthesis_prompt_under_cap_no_truncation_notice() {
        let results = make_results(2);
        let prompt = build_synthesis_prompt(&results, "q", 10_000);
        assert!(prompt.contains("content of memory 0"));
        assert!(prompt.contains("content of memory 1"));
        assert!(
            !prompt.contains("truncated"),
            "under budget → no truncation notice; got prompt = {prompt:?}"
        );
    }

    /// The core regression: a long-content batch with a tight budget gets
    /// truncated, and the total prompt length stays within the cap.
    #[test]
    fn build_synthesis_prompt_caps_long_content() {
        // 10 results × 5_000-char content each = ~50KB raw body
        let results: Vec<RecallResult> = (0..10)
            .map(|i| {
                let mut m = make_memory(i);
                m.content = "x".repeat(5_000);
                RecallResult {
                    memory: m,
                    score: 0.9 - (i as f32 * 0.05),
                    confidence: 0.8,
                    sources_hit: 2,
                    evidence_count: 0,
                    evidence_preview: vec![],
                    archival_summary: None,
                }
            })
            .collect();

        let cap = 8_000;
        let prompt = build_synthesis_prompt(&results, "q", cap);
        let prompt_chars = prompt.chars().count();
        assert!(
            prompt_chars <= cap,
            "prompt ({prompt_chars} chars) must stay within cap ({cap})"
        );
        assert!(
            prompt.contains("truncated"),
            "long content + tight cap → truncation notice expected"
        );
        assert!(
            prompt.contains("[1] Topic: topic-0"),
            "highest-priority memory must always be included"
        );
        assert!(
            !prompt.contains("[10] Topic: topic-9"),
            "lowest-priority memory must be dropped under tight cap"
        );
    }

    /// Edge case: budget so tight even the first memory's header line
    /// doesn't fit — the function must not panic and must still emit the
    /// truncation notice + footer.
    #[test]
    fn build_synthesis_prompt_extreme_tight_cap_does_not_panic() {
        let results = make_results(3);
        // Just enough room for header + footer + truncation notice; zero
        // body budget. saturating_sub keeps body_budget at 0; the loop
        // bails on the first memory.
        let prompt = build_synthesis_prompt(&results, "q", 200);
        assert!(prompt.contains("Now produce the concise narrative"));
        assert!(prompt.contains("truncated"));
    }

    /// Round 3 P2 regression: a multi-KB query string must not bypass the
    /// prompt cap — query is itself budgeted to a fraction of `max_chars`.
    #[test]
    fn build_synthesis_prompt_caps_long_query() {
        let results = make_results(3);
        let long_query = "a".repeat(10_000);
        let cap = 8_000;
        let prompt = build_synthesis_prompt(&results, &long_query, cap);
        let prompt_chars = prompt.chars().count();
        assert!(
            prompt_chars <= cap,
            "long-query prompt ({prompt_chars} chars) must stay within cap ({cap})"
        );
        assert!(
            prompt.contains("query truncated"),
            "long query → query-truncation notice expected; got first 200 chars: {:?}",
            prompt.chars().take(200).collect::<String>()
        );
    }

    /// Defensive: even a query LARGER than the entire `max_chars` budget
    /// must not panic and must produce a prompt within the cap. The final
    /// `take(max_chars)` safety net is what saves us here.
    #[test]
    fn build_synthesis_prompt_query_larger_than_cap_is_capped() {
        let results = make_results(3);
        let huge_query = "z".repeat(50_000);
        let cap = 1_000;
        let prompt = build_synthesis_prompt(&results, &huge_query, cap);
        let prompt_chars = prompt.chars().count();
        assert!(
            prompt_chars <= cap,
            "huge-query prompt ({prompt_chars} chars) must stay within cap ({cap}), \
             defensive take(max_chars) safety net failed"
        );
    }

    /// Floor check: a query under `QUERY_BUDGET_FLOOR` (256 chars) must
    /// pass through untruncated even at small caps, so legitimate
    /// natural-language queries never lose words to over-aggressive
    /// truncation.
    #[test]
    fn build_synthesis_prompt_short_query_not_truncated() {
        let results = make_results(2);
        let normal_query = "what did I decide about caching last week?";
        let prompt = build_synthesis_prompt(&results, normal_query, 8_000);
        assert!(
            prompt.contains(normal_query),
            "short natural-language query must appear verbatim"
        );
        assert!(
            !prompt.contains("query truncated"),
            "short query → no query-truncation notice"
        );
    }

    // ── citation parser (v0.25.2 ARS Cap B inline citations) ──────────────

    /// Basic sanity: a single `[#1]` marker is stripped from the prose
    /// and surfaced as a citation pointing at the char position the
    /// marker occupied (which is also the char count of the prose
    /// preceding the marker).
    #[test]
    fn extract_citations_strips_markers() {
        let (clean, cites) = extract_citations("Foo[#1].", 5);
        assert_eq!(clean, "Foo.");
        assert_eq!(
            cites,
            vec![Citation {
                rank: 1,
                span_end: 3,
            }]
        );
    }

    /// Consecutive markers `[#1][#3]` collapse to two distinct citations
    /// at the same `span_end`. The frontend will group them visually but
    /// each rank stays clickable independently.
    #[test]
    fn extract_citations_handles_consecutive() {
        let (clean, cites) = extract_citations("Foo[#1][#3].", 5);
        assert_eq!(clean, "Foo.");
        assert_eq!(
            cites,
            vec![
                Citation { rank: 1, span_end: 3 },
                Citation { rank: 3, span_end: 3 },
            ]
        );
    }

    /// Invalid ranks (`[#0]`, `[#99]` when only 5 results, `[#abc]`)
    /// must be dropped silently. Well-formed but out-of-range markers
    /// (`[#0]`, `[#99]`) get their marker text removed from the clean
    /// prose; truly malformed markers (`[#abc]`) pass through as literal
    /// text since the LLM may legitimately have meant `[#abc]` in prose
    /// (e.g. a code snippet).
    #[test]
    fn extract_citations_drops_invalid_rank() {
        // rank=0 → swallow marker, no citation
        let (clean, cites) = extract_citations("Foo [#0].", 5);
        assert_eq!(clean, "Foo .");
        assert!(cites.is_empty());

        // rank > max_rank → swallow marker, no citation
        let (clean, cites) = extract_citations("Foo [#99].", 5);
        assert_eq!(clean, "Foo .");
        assert!(cites.is_empty());

        // malformed body → pass through as literal text
        let (clean, cites) = extract_citations("Foo [#abc].", 5);
        assert_eq!(clean, "Foo [#abc].");
        assert!(cites.is_empty());

        // unterminated marker → pass through
        let (clean, cites) = extract_citations("Foo [#1.", 5);
        assert_eq!(clean, "Foo [#1.");
        assert!(cites.is_empty());
    }

    /// Empty / no-marker input → empty citation vec, prose returned unchanged.
    #[test]
    fn extract_citations_empty_input() {
        let (clean, cites) = extract_citations("", 5);
        assert_eq!(clean, "");
        assert!(cites.is_empty());

        let (clean, cites) = extract_citations("Plain prose with no markers.", 5);
        assert_eq!(clean, "Plain prose with no markers.");
        assert!(cites.is_empty());
    }

    /// CJK-safe: `span_end` must be a CHAR offset, not a byte offset.
    /// "中文" is 6 bytes UTF-8 but 2 chars; a marker after it must
    /// produce `span_end: 2`. This is the canonical case where a
    /// byte-offset bug would silently desync Rust + JS.
    #[test]
    fn extract_citations_unicode_safe() {
        let (clean, cites) = extract_citations("中文[#1]。", 5);
        assert_eq!(clean, "中文。");
        assert_eq!(
            cites,
            vec![Citation { rank: 1, span_end: 2 }],
            "span_end must be 2 (char count of 中文), not 6 (byte length)"
        );

        // Marker between two CJK runs.
        let (clean, cites) = extract_citations("缓存策略[#2]需要复审[#3]。", 5);
        assert_eq!(clean, "缓存策略需要复审。");
        assert_eq!(
            cites,
            vec![
                Citation { rank: 2, span_end: 4 },
                Citation { rank: 3, span_end: 8 },
            ]
        );
    }

    /// Citation at the very start of the prose lands at `span_end: 0`.
    /// (Spec example: `"[#1]Foo." -> ("Foo.", [{1,0}])`)
    #[test]
    fn extract_citations_at_start() {
        let (clean, cites) = extract_citations("[#1]Foo.", 5);
        assert_eq!(clean, "Foo.");
        assert_eq!(cites, vec![Citation { rank: 1, span_end: 0 }]);
    }

    /// Multi-claim spec example: `"Foo[#1][#2]bar[#3]." → ("Foobar.", …)`.
    #[test]
    fn extract_citations_multi_claim_inline() {
        let (clean, cites) = extract_citations("Foo[#1][#2]bar[#3].", 5);
        assert_eq!(clean, "Foobar.");
        assert_eq!(
            cites,
            vec![
                Citation { rank: 1, span_end: 3 },
                Citation { rank: 2, span_end: 3 },
                Citation { rank: 3, span_end: 6 },
            ]
        );
    }

    #[cfg(feature = "test-support")]
    #[test]
    fn run_recall_synthesis_extracts_citations_from_mock() {
        use crate::extract::llm::MockExtractor;

        let mut config = ReinConfig::default();
        config.ars.recall_synthesis_enabled = true;
        config.ars.recall_synthesis_min_results = 3;
        let results = make_results(5);
        let mock = ExtractorKind::Mock(MockExtractor::with_fixed_response(
            "The auth middleware was rewritten[#1][#3]. The new design uses session storage[#2].",
        ));
        let outcome =
            run_recall_synthesis(&results, "auth", &config, Some(true), None, Some(mock)).unwrap();
        assert_eq!(
            outcome.synthesis.as_deref(),
            Some("The auth middleware was rewritten. The new design uses session storage."),
            "markers must be stripped from the synthesis text"
        );
        assert_eq!(
            outcome.citations,
            vec![
                Citation {
                    rank: 1,
                    span_end: "The auth middleware was rewritten".chars().count(),
                },
                Citation {
                    rank: 3,
                    span_end: "The auth middleware was rewritten".chars().count(),
                },
                Citation {
                    rank: 2,
                    span_end: "The auth middleware was rewritten. The new design uses session storage"
                        .chars()
                        .count(),
                },
            ]
        );
    }

    #[cfg(feature = "test-support")]
    #[test]
    fn run_recall_synthesis_drops_out_of_range_citations() {
        use crate::extract::llm::MockExtractor;

        let mut config = ReinConfig::default();
        config.ars.recall_synthesis_enabled = true;
        config.ars.recall_synthesis_min_results = 3;
        // 3 results → max_rank=3. Marker [#9] points past the last
        // result and must be dropped without affecting the [#1] citation.
        let results = make_results(3);
        let mock = ExtractorKind::Mock(MockExtractor::with_fixed_response(
            "Sourced claim[#1] and a hallucinated claim[#9].",
        ));
        let outcome =
            run_recall_synthesis(&results, "test", &config, Some(true), None, Some(mock)).unwrap();
        assert_eq!(
            outcome.synthesis.as_deref(),
            Some("Sourced claim and a hallucinated claim."),
            "out-of-range marker swallowed alongside the in-range one"
        );
        assert_eq!(
            outcome.citations,
            vec![Citation {
                rank: 1,
                span_end: "Sourced claim".chars().count(),
            }],
            "out-of-range [#9] dropped, [#1] preserved"
        );
    }

    // ── v0.26 D direction: decide_synthesize gate (cold-start, threshold,
    // operator-disabled, mutually-exclusive flags) ──────────────────────────

    use crate::store::adaptive::{
        ClusterSynthesisStats, SynthesisFeedbackState, SYNTHESIS_COLD_START_N,
        SYNTHESIS_USEFUL_RATE_THRESHOLD,
    };

    /// Build a `ClusterSynthesisStats` with a target `useful_rate` and
    /// `viewed_count` — the only fields `decide_synthesize` reads.
    fn cluster_stats(viewed: u64, rate: f64) -> ClusterSynthesisStats {
        ClusterSynthesisStats {
            viewed_count: viewed,
            useful_rate: rate,
            ..Default::default()
        }
    }

    /// Helper: build an `AdaptiveState` with one bucket pre-populated.
    fn state_with_bucket(key: &str, stats: ClusterSynthesisStats) -> AdaptiveState {
        let mut sfs = SynthesisFeedbackState::default();
        sfs.by_cluster.insert(key.to_string(), stats);
        AdaptiveState {
            synthesis_feedback_stats: Some(sfs),
            ..Default::default()
        }
    }

    /// Cold-start path 1: empty `Option<&AdaptiveState>` → return Yes
    /// (matches v0.25.x behavior). Per contract §8 invariant 4.
    #[test]
    fn decide_synthesize_cold_start_no_state_returns_yes() {
        let decision = decide_synthesize(true, Some(42), "Semantic", None, SYNTHESIS_COLD_START_N);
        assert_eq!(decision, SynthesizeDecision::Yes);
    }

    /// Cold-start path 2: state present but `synthesis_feedback_stats =
    /// None` (fresh install) → return Yes.
    #[test]
    fn decide_synthesize_cold_start_no_synthesis_state_returns_yes() {
        let state = AdaptiveState::default(); // synthesis_feedback_stats: None
        let decision = decide_synthesize(
            true,
            Some(42),
            "Semantic",
            Some(&state),
            SYNTHESIS_COLD_START_N,
        );
        assert_eq!(decision, SynthesizeDecision::Yes);
    }

    /// Cold-start path 3: cluster_id is None — runtime gate must NOT route
    /// through the global `-1` bucket (too noisy for individual recalls).
    #[test]
    fn decide_synthesize_cold_start_no_cluster_id_returns_yes() {
        // Even with a populated state, no cluster_id → cold-start fallback.
        let state = state_with_bucket(
            &synthesis_bucket_key(Some(42), "Semantic"),
            cluster_stats(100, 0.1), // would skip if it were looked up
        );
        let decision =
            decide_synthesize(true, None, "Semantic", Some(&state), SYNTHESIS_COLD_START_N);
        assert_eq!(decision, SynthesizeDecision::Yes);
    }

    /// Cold-start path 4: cluster has data but `viewed_count <
    /// COLD_START_N` → return Yes (insufficient samples to trust).
    #[test]
    fn decide_synthesize_cold_start_insufficient_samples_returns_yes() {
        let state = state_with_bucket(
            &synthesis_bucket_key(Some(42), "Semantic"),
            cluster_stats(SYNTHESIS_COLD_START_N - 1, 0.0), // 1 short of threshold, useful_rate awful
        );
        let decision = decide_synthesize(
            true,
            Some(42),
            "Semantic",
            Some(&state),
            SYNTHESIS_COLD_START_N,
        );
        assert_eq!(decision, SynthesizeDecision::Yes);
    }

    /// Warm cluster, useful_rate ABOVE threshold → return Yes (adaptive
    /// signal agrees with global default).
    #[test]
    fn decide_synthesize_warm_cluster_above_threshold_returns_yes() {
        let state = state_with_bucket(
            &synthesis_bucket_key(Some(42), "Semantic"),
            cluster_stats(SYNTHESIS_COLD_START_N + 10, SYNTHESIS_USEFUL_RATE_THRESHOLD + 0.1),
        );
        let decision = decide_synthesize(
            true,
            Some(42),
            "Semantic",
            Some(&state),
            SYNTHESIS_COLD_START_N,
        );
        assert_eq!(decision, SynthesizeDecision::Yes);
    }

    /// Warm cluster, useful_rate BELOW threshold → return
    /// `Skip(AdaptiveDecision)`. The adaptive signal disagrees with the
    /// global default, so we trust it.
    #[test]
    fn decide_synthesize_warm_cluster_below_threshold_returns_skip_adaptive() {
        let state = state_with_bucket(
            &synthesis_bucket_key(Some(42), "Semantic"),
            cluster_stats(SYNTHESIS_COLD_START_N + 10, SYNTHESIS_USEFUL_RATE_THRESHOLD - 0.1),
        );
        let decision = decide_synthesize(
            true,
            Some(42),
            "Semantic",
            Some(&state),
            SYNTHESIS_COLD_START_N,
        );
        assert_eq!(decision, SynthesizeDecision::Skip(SkipReason::AdaptiveDecision));
    }

    /// Operator-off ALWAYS short-circuits — even with rich adaptive data
    /// suggesting synthesis would help, the operator override wins.
    #[test]
    fn decide_synthesize_operator_disabled_overrides_adaptive() {
        let state = state_with_bucket(
            &synthesis_bucket_key(Some(42), "Semantic"),
            cluster_stats(SYNTHESIS_COLD_START_N + 10, 1.0), // perfect useful_rate
        );
        let decision = decide_synthesize(
            false, // operator off
            Some(42),
            "Semantic",
            Some(&state),
            SYNTHESIS_COLD_START_N,
        );
        assert_eq!(
            decision,
            SynthesizeDecision::Skip(SkipReason::OperatorDisabled)
        );
    }

    /// Operator-off + cold-start (no adaptive state at all) — still
    /// `OperatorDisabled`, NOT `AdaptiveDecision`. The two reasons are
    /// distinct surfaces for the GUI.
    #[test]
    fn decide_synthesize_operator_disabled_with_no_state() {
        let decision =
            decide_synthesize(false, Some(42), "Semantic", None, SYNTHESIS_COLD_START_N);
        assert_eq!(
            decision,
            SynthesizeDecision::Skip(SkipReason::OperatorDisabled)
        );
    }

    /// Bucket-key mismatch (different query_type, same cluster) → cold-start
    /// fallback — gate looks at the wrong bucket and finds nothing.
    /// Proves the function honors the query_type partition; v0.26.1 will
    /// pull the real query_type from the classifier so different intents
    /// don't bleed into one another.
    #[test]
    fn decide_synthesize_query_type_partition_isolates_buckets() {
        // Cluster 42 has plenty of "Episodic" data saying "skip" — but
        // we're asking about "Semantic", which has no bucket → Yes.
        let state = state_with_bucket(
            &synthesis_bucket_key(Some(42), "Episodic"),
            cluster_stats(SYNTHESIS_COLD_START_N + 50, 0.0), // would Skip if it matched
        );
        let decision = decide_synthesize(
            true,
            Some(42),
            "Semantic",
            Some(&state),
            SYNTHESIS_COLD_START_N,
        );
        assert_eq!(
            decision,
            SynthesizeDecision::Yes,
            "different query_type → different bucket → cold-start fallback"
        );
    }

    /// Custom `cold_start_n` plumbing: pass a smaller value to flip from
    /// cold-start to per-cluster behavior.
    #[test]
    fn decide_synthesize_custom_cold_start_n() {
        let state = state_with_bucket(
            &synthesis_bucket_key(Some(42), "Semantic"),
            cluster_stats(3, SYNTHESIS_USEFUL_RATE_THRESHOLD - 0.1),
        );
        // viewed=3 < default cold_start_n=10 → Yes
        assert_eq!(
            decide_synthesize(true, Some(42), "Semantic", Some(&state), 10),
            SynthesizeDecision::Yes
        );
        // viewed=3 >= custom cold_start_n=2 → consults useful_rate (below
        // threshold) → Skip
        assert_eq!(
            decide_synthesize(true, Some(42), "Semantic", Some(&state), 2),
            SynthesizeDecision::Skip(SkipReason::AdaptiveDecision)
        );
    }

    // ── RecallSynthesisOutcome serde — synthesis_id + skipped_adaptive_decision ──

    /// Round-trip: a populated `synthesis_id` survives JSON serialization
    /// and is visible to clients.
    #[test]
    fn outcome_serde_roundtrip_synthesis_id() {
        let outcome = RecallSynthesisOutcome {
            synthesis: Some("test prose".to_string()),
            query: "q".to_string(),
            source_count: 3,
            model_used: None,
            skipped_disabled: false,
            skipped_adaptive_decision: false,
            skipped_no_llm: false,
            skipped_too_few_results: false,
            citations: Vec::new(),
            synthesis_id: Some("01ARZ3NDEKTSV4RRFFQ69G5FAV".to_string()),
        };
        let json = serde_json::to_string(&outcome).unwrap();
        // Field MUST be present in the wire format when populated.
        assert!(
            json.contains("synthesis_id"),
            "synthesis_id absent from JSON: {json}"
        );
        assert!(json.contains("01ARZ3NDEKTSV4RRFFQ69G5FAV"));
    }

    /// `synthesis_id = None` MUST be omitted from the wire format
    /// (`skip_serializing_if = "Option::is_none"`) — old clients that
    /// don't know the field stay bit-identical to their previous experience.
    #[test]
    fn outcome_serde_synthesis_id_omitted_when_none() {
        let outcome = RecallSynthesisOutcome {
            synthesis: None,
            query: "q".to_string(),
            source_count: 0,
            model_used: None,
            skipped_disabled: true,
            skipped_adaptive_decision: false,
            skipped_no_llm: false,
            skipped_too_few_results: false,
            citations: Vec::new(),
            synthesis_id: None,
        };
        let json = serde_json::to_string(&outcome).unwrap();
        assert!(
            !json.contains("synthesis_id"),
            "synthesis_id SHOULD be elided when None; got JSON: {json}"
        );
    }

    /// `skipped_adaptive_decision = true` is visible in the wire format
    /// (skip_serializing_if = false collapses the field, but true stays).
    #[test]
    fn outcome_serde_skipped_adaptive_decision_visible_when_true() {
        let outcome = RecallSynthesisOutcome {
            synthesis: None,
            query: "q".to_string(),
            source_count: 5,
            model_used: None,
            skipped_disabled: false,
            skipped_adaptive_decision: true,
            skipped_no_llm: false,
            skipped_too_few_results: false,
            citations: Vec::new(),
            synthesis_id: None,
        };
        let json = serde_json::to_string(&outcome).unwrap();
        assert!(
            json.contains("skipped_adaptive_decision"),
            "true value MUST appear in wire format: {json}"
        );
    }

    /// Mutual-exclusivity sanity: no `decide_synthesize` outcome can set
    /// both `skipped_disabled` AND `skipped_adaptive_decision`. Verified at
    /// the gate-result level — `OperatorDisabled` short-circuits before
    /// the per-cluster check can fire.
    #[test]
    fn skipped_flags_are_mutually_exclusive() {
        // Build a config that passes the per-cluster check (operator on,
        // useful_rate above threshold) AND a config that fails (operator
        // off). Run both through the gate and assert the resulting
        // outcome has at most ONE skipped flag set.
        let warm_state = state_with_bucket(
            &synthesis_bucket_key(Some(42), "Semantic"),
            cluster_stats(SYNTHESIS_COLD_START_N + 10, SYNTHESIS_USEFUL_RATE_THRESHOLD - 0.1),
        );
        let scenarios = [
            // (global, expected_decision)
            (true, SynthesizeDecision::Skip(SkipReason::AdaptiveDecision)),
            (false, SynthesizeDecision::Skip(SkipReason::OperatorDisabled)),
        ];
        for (global, expected) in scenarios {
            let decision = decide_synthesize(
                global,
                Some(42),
                "Semantic",
                Some(&warm_state),
                SYNTHESIS_COLD_START_N,
            );
            assert_eq!(decision, expected);
            // Reduce to the (skipped_disabled, skipped_adaptive_decision)
            // pair the run_recall_synthesis gate would set:
            let (sd, sa) = match decision {
                SynthesizeDecision::Yes => (false, false),
                SynthesizeDecision::Skip(SkipReason::OperatorDisabled) => (true, false),
                SynthesizeDecision::Skip(SkipReason::AdaptiveDecision) => (false, true),
            };
            // Only one (or zero) flag may be true at a time.
            assert!(
                !(sd && sa),
                "flags must be mutually exclusive (got skipped_disabled={sd}, skipped_adaptive_decision={sa})"
            );
        }
    }

    /// Backward serde: deserialize a v0.25.x payload (no `synthesis_id`,
    /// no `skipped_adaptive_decision` keys) — both new fields default
    /// gracefully.
    ///
    /// `RecallSynthesisOutcome` is `Serialize` only in production code,
    /// so this test uses a structurally-equivalent local `Deserialize`
    /// shadow type to verify the wire shape parses cleanly. If a future
    /// edit accidentally drops `#[serde(default)]` from the new fields,
    /// this test will fail with a "missing field" error.
    #[test]
    fn outcome_serde_backward_compat_old_payload_parses() {
        use serde::Deserialize;

        #[derive(Deserialize)]
        #[allow(dead_code)]
        struct OutcomeReadback {
            #[serde(default)]
            synthesis: Option<String>,
            query: String,
            source_count: usize,
            #[serde(default)]
            skipped_disabled: bool,
            // The new v0.26.0 fields MUST default cleanly when missing
            // from the wire payload.
            #[serde(default)]
            skipped_adaptive_decision: bool,
            #[serde(default)]
            synthesis_id: Option<String>,
        }

        // v0.25.x-era payload: no synthesis_id / skipped_adaptive_decision.
        let old_json = r#"{
            "synthesis": "old prose",
            "query": "q",
            "source_count": 3,
            "skipped_disabled": false
        }"#;
        let parsed: OutcomeReadback = serde_json::from_str(old_json)
            .expect("v0.25.x payload must still parse on v0.26.0 readback");
        assert_eq!(parsed.synthesis.as_deref(), Some("old prose"));
        assert!(!parsed.skipped_adaptive_decision);
        assert!(parsed.synthesis_id.is_none());
    }

    /// End-to-end: when the per-query gate returns
    /// `Skip(AdaptiveDecision)`, `run_recall_synthesis` MUST set
    /// `outcome.skipped_adaptive_decision = true` AND leave
    /// `synthesis_id = None` AND skip the LLM call entirely (so the test
    /// can run without a real provider). Wires together
    /// `decide_synthesize` + `run_recall_synthesis` + the new outcome
    /// flag so a future regression that breaks the gate→outcome plumbing
    /// surfaces here, not 6 layers deep in production.
    #[test]
    fn run_recall_synthesis_routes_adaptive_skip_into_outcome_flag() {
        // Build results with cluster_id=0 so the gate's cluster_id branch
        // engages instead of cold-start fallback. (`make_memory` defaults
        // cluster_id=None which would short-circuit to Yes.)
        let mut results = make_results(5);
        for r in &mut results {
            r.memory.cluster_id = Some(0);
        }

        // Adaptive state: bucket exists, has plenty of samples, but
        // `useful_rate` below threshold → Skip(AdaptiveDecision).
        let state = state_with_bucket(
            &synthesis_bucket_key(Some(0), "Semantic"),
            cluster_stats(SYNTHESIS_COLD_START_N + 10, SYNTHESIS_USEFUL_RATE_THRESHOLD - 0.1),
        );

        let mut config = ReinConfig::default();
        config.ars.recall_synthesis_enabled = true;
        config.ars.recall_synthesis_min_results = 3;

        let outcome =
            run_recall_synthesis(&results, "test", &config, Some(true), Some(&state), None)
                .expect("synthesis was requested → Some(outcome)");

        assert!(
            outcome.skipped_adaptive_decision,
            "per-query gate skip MUST set skipped_adaptive_decision; got {outcome:?}"
        );
        assert!(
            !outcome.skipped_disabled,
            "operator was on; skipped_disabled MUST stay false to keep the two flags distinguishable"
        );
        assert!(
            outcome.synthesis.is_none(),
            "skipped paths MUST NOT carry a synthesis prose"
        );
        assert!(
            outcome.synthesis_id.is_none(),
            "skipped paths MUST leave synthesis_id = None (contract §8 invariant 9)"
        );
    }
}
