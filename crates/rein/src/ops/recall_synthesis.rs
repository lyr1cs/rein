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
not present in the memory list below.";

fn is_false(b: &bool) -> bool {
    !*b
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
///   skipped_no_llm?: boolean;
///   skipped_too_few_results?: boolean;
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
    /// `true` when `[ars].recall_synthesis_enabled = false` (operator opted out).
    #[serde(skip_serializing_if = "is_false")]
    pub skipped_disabled: bool,
    /// `true` when no LLM provider is configured or the API key is absent.
    #[serde(skip_serializing_if = "is_false")]
    pub skipped_no_llm: bool,
    /// `true` when `results.len() < [ars].recall_synthesis_min_results`.
    #[serde(skip_serializing_if = "is_false")]
    pub skipped_too_few_results: bool,
}

/// Run recall-time synthesis over `results` for `query`.
///
/// Returns `None` when synthesis was not requested (`synthesize` is `None` or
/// `Some(false)`). Returns `Some(RecallSynthesisOutcome)` when requested — the
/// outcome's `skipped_*` flags explain why synthesis was not produced (if any).
///
/// `extractor_override` is only used by tests (feature `test-support`); pass
/// `None` in production.
pub fn run_recall_synthesis(
    results: &[RecallResult],
    query: &str,
    config: &ReinConfig,
    synthesize: Option<bool>,
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
        skipped_no_llm: false,
        skipped_too_few_results: false,
    };

    if !config.ars.recall_synthesis_enabled {
        outcome.skipped_disabled = true;
        return Some(outcome);
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
    let prompt = build_synthesis_prompt(results, query, max_chars);
    match call_synthesis_llm_sync(&extractor, &prompt) {
        Ok(raw) => {
            let text = strip_code_fences(&raw).trim().to_string();
            if !text.is_empty() {
                outcome.synthesis = Some(text);
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
pub fn build_synthesis_prompt(results: &[RecallResult], query: &str, max_chars: usize) -> String {
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
        return buf;
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
            // truncation so the LLM knows facts were dropped.
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
    buf
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
            })
            .collect()
    }

    #[test]
    fn not_requested_returns_none() {
        let config = ReinConfig::default();
        let results = make_results(5);
        assert!(
            run_recall_synthesis(&results, "test query", &config, None, None).is_none(),
            "None synthesize param → None outcome"
        );
        assert!(
            run_recall_synthesis(&results, "test query", &config, Some(false), None).is_none(),
            "Some(false) synthesize param → None outcome"
        );
    }

    #[test]
    fn skipped_disabled_when_feature_off() {
        let config = ReinConfig::default(); // recall_synthesis_enabled = false
        let results = make_results(5);
        let outcome =
            run_recall_synthesis(&results, "test", &config, Some(true), None).unwrap();
        assert!(outcome.skipped_disabled, "feature off → skipped_disabled");
        assert!(!outcome.skipped_no_llm);
        assert!(!outcome.skipped_too_few_results);
        assert!(outcome.synthesis.is_none());
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
            run_recall_synthesis(&results, "test", &config, Some(true), None).unwrap();
        assert!(
            outcome.skipped_too_few_results,
            "2 results < min 3 → skipped_too_few_results"
        );
        assert!(!outcome.skipped_disabled);
        assert!(!outcome.skipped_no_llm);
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
            run_recall_synthesis(&results, "test", &config, Some(true), None).unwrap();
        assert!(outcome.skipped_no_llm, "no provider → skipped_no_llm");
        assert!(!outcome.skipped_disabled);
        assert!(!outcome.skipped_too_few_results);
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
            run_recall_synthesis(&results, "test query", &config, Some(true), Some(mock))
                .unwrap();
        assert!(!outcome.skipped_disabled);
        assert!(!outcome.skipped_no_llm);
        assert!(!outcome.skipped_too_few_results);
        assert_eq!(
            outcome.synthesis.as_deref(),
            Some("Synthesized narrative."),
            "synthesis text matches mock response"
        );
        assert_eq!(outcome.source_count, 5);
        assert_eq!(outcome.query, "test query");
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
}
