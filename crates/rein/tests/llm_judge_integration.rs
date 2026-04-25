//! v0.25.3 LLM-judged hit-checker integration tests (Agent C).
//!
//! Drives the `LlmJudgeHitChecker` defined in `eval/llm_judge.rs` (Agent A)
//! through `MockExtractor` so the same surface the `rein-eval` binary
//! (Agent B) wires up is exercised without a live LLM provider.
//!
//! ## Interface drift from the spec'd locked contract
//!
//! Agent A's actual signatures differ from the v0.25.3 task contract:
//!
//! - `judge_synthesis(query: &str, source_summaries: &[&str], candidate: &str)`
//!   uses `&[&str]` (not `&[String]`). Tests pass `&["..."]` literals.
//! - `judge_concept_summary(definition: &str, living_summary: Option<&str>,
//!   evidence_keywords: &[String])` mirrors `score_concept_case` (not the
//!   query/sources/synth shape from the locked contract).
//! - `JudgeMode::ConceptSummaryFactCoverage` (not `…EvidenceCoverage`).
//!
//! The tests below conform to Agent A's actual public API as shipped.
//!
//! Gated on `feature = "test-support"` because `MockExtractor` lives behind
//! that flag and is absent from the release binary.

#![cfg(feature = "test-support")]

use std::sync::Arc;

use rein::eval::llm_judge::{
    JudgeMode, JudgeOutcome, LlmJudgeHitChecker, LLM_JUDGE_VERSION,
};
use rein::eval::HIT_CHECKER_VERSION;
use rein::extract::{ExtractorKind, MockExtractor};

/// Build a checker in `SynthesisSourceCoverage` mode wrapping the supplied
/// mock extractor.
fn make_synthesis_checker(mock: MockExtractor) -> LlmJudgeHitChecker {
    let extractor = Arc::new(ExtractorKind::Mock(mock));
    LlmJudgeHitChecker::new(extractor, JudgeMode::SynthesisSourceCoverage)
}

#[test]
fn judge_synthesis_hit_path() {
    let mock = MockExtractor::with_responses(vec![Ok(
        r#"{"hit": true, "reason": "covers source 1 + 2"}"#.to_string(),
    )]);
    let judge = make_synthesis_checker(mock);
    let outcome: JudgeOutcome = judge
        .judge_synthesis(
            "what is foo?",
            &["source A", "source B"],
            "synthesized text covering both sources",
        )
        .expect("hit-path mock returned valid JSON; judge must succeed");
    assert!(outcome.hit, "mock returned hit=true; outcome must mirror it");
    assert!(
        !outcome.reason.is_empty(),
        "reason field must propagate from the LLM JSON to the outcome"
    );
}

#[test]
fn judge_synthesis_miss_path() {
    let mock = MockExtractor::with_responses(vec![Ok(
        r#"{"hit": false, "reason": "synthesis ignores source 2"}"#.to_string(),
    )]);
    let judge = make_synthesis_checker(mock);
    let outcome = judge
        .judge_synthesis(
            "what is bar?",
            &["source A", "source B"],
            "partial synthesis missing source 2",
        )
        .expect("miss-path mock still returns valid JSON; judge must succeed");
    assert!(
        !outcome.hit,
        "mock returned hit=false; outcome must mirror it (false-negative not silently flipped)"
    );
}

#[test]
fn judge_synthesis_malformed_returns_err() {
    // Mock returns plain prose with no JSON object — the judge must surface
    // this as `Err` rather than treat it as a silent miss. A silent miss
    // would let provider misconfiguration (wrong response_format, prose
    // leak) skew the eval scorecard with no diagnostic.
    let mock = MockExtractor::with_responses(vec![Ok("not json at all".to_string())]);
    let judge = make_synthesis_checker(mock);
    let result = judge.judge_synthesis("what is baz?", &["src"], "synth output");
    assert!(
        result.is_err(),
        "malformed (non-JSON) LLM response must propagate as Err, not silently miss; \
         got {result:?}"
    );
}

#[test]
fn judge_synthesis_strips_code_fences() {
    // Agent A's contract: the judge must strip ```json ... ``` fences before
    // parsing. Many LLM providers wrap structured output in code fences even
    // under JSON-mode hints; without fence stripping the JSON parser would
    // fail on the literal backticks. The fence string uses an `r##"..."##`
    // raw delimiter so the embedded triple backticks are unambiguous.
    let fenced = r##"```json
{"hit": true, "reason": "fenced response parses cleanly"}
```"##
        .to_string();
    let mock = MockExtractor::with_responses(vec![Ok(fenced)]);
    let judge = make_synthesis_checker(mock);
    let outcome = judge
        .judge_synthesis("fence test", &["src"], "synthesized with fences")
        .expect("fenced JSON must round-trip after Agent A strips the code fence");
    assert!(outcome.hit, "fence-stripped payload says hit=true");
}

#[test]
fn judge_concept_summary_hit_path() {
    // Agent A's `judge_concept_summary` signature mirrors `score_concept_case`:
    //   judge_concept_summary(definition, living_summary, evidence_keywords)
    // (not the query/sources/synth shape suggested by the locked contract).
    let mock = MockExtractor::with_responses(vec![Ok(
        r#"{"hit": true, "reason": "definition covers all evidence themes"}"#.to_string(),
    )]);
    let extractor = Arc::new(ExtractorKind::Mock(mock));
    let judge = LlmJudgeHitChecker::new(extractor, JudgeMode::ConceptSummaryFactCoverage);
    let evidence = vec!["evidence theme 1".to_string(), "evidence theme 2".to_string()];
    let outcome = judge
        .judge_concept_summary(
            "concept definition covering both evidence themes",
            Some("living summary reinforcing evidence theme 2"),
            &evidence,
        )
        .expect("hit-path mock for concept-summary must succeed");
    assert!(outcome.hit, "mock returned hit=true; outcome must mirror it");
    assert!(
        !outcome.reason.is_empty(),
        "reason field must propagate for concept-summary mode too"
    );
}

#[test]
fn arc_extractor_shared_between_modes() {
    // One Arc<ExtractorKind> shared between two LlmJudgeHitChecker instances
    // with different JudgeMode — proves the sharing pattern Agent B uses to
    // wire one extractor into both code paths of the eval loop. The mock
    // queue is shared, so we enqueue TWO responses (one drained per call).
    let mock = MockExtractor::with_responses(vec![
        Ok(r#"{"hit": true, "reason": "synthesis call"}"#.to_string()),
        Ok(r#"{"hit": true, "reason": "concept-summary call"}"#.to_string()),
    ]);
    let extractor = Arc::new(ExtractorKind::Mock(mock));

    let synthesis_judge =
        LlmJudgeHitChecker::new(Arc::clone(&extractor), JudgeMode::SynthesisSourceCoverage);
    let concept_judge = LlmJudgeHitChecker::new(
        Arc::clone(&extractor),
        JudgeMode::ConceptSummaryFactCoverage,
    );

    let syn_outcome = synthesis_judge
        .judge_synthesis("query", &["source"], "synthesized response")
        .expect("first call drains response 1");
    assert!(syn_outcome.hit);

    let cs_outcome = concept_judge
        .judge_concept_summary("definition text", None, &["evidence".to_string()])
        .expect("second call drains response 2 from the shared queue");
    assert!(cs_outcome.hit);
}

#[test]
fn llm_judge_version_distinct_from_keyword() {
    // Guards against accidentally making versions collide. `compare` in the
    // rein-eval scorecard regime relies on the version constants to detect
    // methodology drift between baseline (keyword overlap) and treatment
    // (LLM-judged) runs — if the constants ever match, the regime-mismatch
    // bail-out can't fire.
    assert_ne!(
        LLM_JUDGE_VERSION, HIT_CHECKER_VERSION,
        "LLM_JUDGE_VERSION must be distinct from HIT_CHECKER_VERSION so \
         compare can detect regime mismatch between scorecards"
    );
}

#[test]
fn mock_response_consumed_per_call() {
    // One mock with 2 responses; call judge_synthesis twice. Both calls
    // succeed → drains the queue → proves single-shot per case (no caching,
    // no replay). A regression that double-consumes or fails to consume
    // would surface here as a queue-exhaustion error on the second call or
    // a leftover response for a phantom third.
    let mock = MockExtractor::with_responses(vec![
        Ok(r#"{"hit": true, "reason": "first call"}"#.to_string()),
        Ok(r#"{"hit": false, "reason": "second call"}"#.to_string()),
    ]);
    let judge = make_synthesis_checker(mock);

    let first = judge
        .judge_synthesis("q1", &["s1"], "r1")
        .expect("first call drains response 1");
    assert!(first.hit, "response 1 says hit=true");

    let second = judge
        .judge_synthesis("q2", &["s2"], "r2")
        .expect("second call drains response 2 (proves per-call consumption)");
    assert!(!second.hit, "response 2 says hit=false");
}
