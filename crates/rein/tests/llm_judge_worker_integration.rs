//! v0.27.1 E direction integration tests for the runtime LLM judge worker.
//!
//! Drives `crate::ops::llm_judge_worker::dispatch_one` end-to-end with a
//! [`MockExtractor`] queue so the same surface a production worker would
//! use is exercised without a live LLM provider. The mock seeds `HIT:
//! yes\nWHY: ...` lines so `parse_judge_output` is also covered through
//! the dispatcher.
//!
//! Gated on `feature = "test-support"` because `MockExtractor` lives behind
//! that flag and is absent from the release binary.

#![cfg(feature = "test-support")]

use rein::extract::llm::{ExtractorKind, MockExtractor};
use rein::judge::contract::LLM_JUDGE_DAILY_CALL_CAP_DEFAULT;
use rein::ops::llm_judge_worker::{
    dispatch_one, parse_judge_output, DispatchResult, DropReason, JudgeJob, JudgeJobKind,
};
use rein::store::adaptive::{
    peek_events, EventType, JudgeSource, SynthesisLlmJudgePayload, LLM_JUDGE_J3_MIN_PAIRS,
    LLM_JUDGE_KAPPA_FLOOR,
};
use rein::store::SqliteStore;

fn temp_store() -> (SqliteStore, tempfile::TempDir) {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let mut config = rein::config::ReinConfig::default();
    config.database.path = tmp
        .path()
        .join("memories.db")
        .to_string_lossy()
        .into_owned();
    let store = config.open_store().expect("open store");
    (store, tmp)
}

fn make_job(
    kind: JudgeJobKind,
    surface_id: &str,
    concept_id: Option<&str>,
    query: &str,
    prompt: &str,
    candidate: &str,
) -> JudgeJob {
    let stamp_hash = JudgeJob::compute_stamp_hash(query, prompt, candidate);
    JudgeJob {
        kind,
        surface_id: surface_id.to_string(),
        concept_id: concept_id.map(String::from),
        query: query.to_string(),
        prompt: prompt.to_string(),
        candidate: candidate.to_string(),
        stamp_hash,
        source: JudgeSource::AutoSampled,
        query_type: Some("Semantic".to_string()),
        cluster_id: Some(7),
        source_count: Some(3),
        judge_model_override: None,
    }
}

#[test]
fn dispatch_emits_synthesis_judge_event_on_hit() {
    let (store, _tmp) = temp_store();
    let extractor = ExtractorKind::Mock(MockExtractor::with_fixed_response(
        "HIT: yes\nWHY: synthesis cites every evidence id.",
    ));
    let job = make_job(
        JudgeJobKind::Synthesis,
        "syn_test_1",
        None,
        "what changed in v0.27?",
        "Sources: [#1] note about migration. [#2] decision to add index.",
        "v0.27 added a migration and a new index.",
    );

    let res = dispatch_one(&store, &extractor, job, LLM_JUDGE_DAILY_CALL_CAP_DEFAULT)
        .expect("dispatch ok");
    let event_id = match res {
        DispatchResult::Emitted(id) => id,
        other => panic!("expected Emitted, got {other:?}"),
    };

    // Verify a SynthesisLlmJudge event landed in feedback_events with the
    // expected payload.
    let events = peek_events(
        store.conn(),
        "_test_synth_consumer",
        &[EventType::SynthesisLlmJudge.as_str()],
        100,
    )
    .expect("peek events");
    assert!(
        events.iter().any(|e| e.id == event_id),
        "emitted event must be peekable"
    );
    let target = events.iter().find(|e| e.id == event_id).unwrap();
    let payload: SynthesisLlmJudgePayload = serde_json::from_str(target.payload.as_deref().unwrap())
        .expect("payload deserializes");
    assert_eq!(payload.synthesis_id, "syn_test_1");
    assert!(payload.hit, "verdict should be HIT=yes");
    assert!(
        payload.reason.contains("evidence"),
        "reason preserved verbatim from mock"
    );
}

#[test]
fn dispatch_emits_concept_summary_judge_event_on_miss() {
    let (store, _tmp) = temp_store();
    let extractor = ExtractorKind::Mock(MockExtractor::with_fixed_response(
        "HIT: no\nWHY: paraphrased the version string into a generic term.",
    ));
    let job = make_job(
        JudgeJobKind::ConceptSummary,
        "cs_test_1",
        Some("concept_alpha"),
        "what is HNSW used for?",
        "Definition: usearch HNSW wrapper.",
        "A vector library wrapper.",
    );

    let res = dispatch_one(&store, &extractor, job, LLM_JUDGE_DAILY_CALL_CAP_DEFAULT)
        .expect("dispatch ok");
    let event_id = match res {
        DispatchResult::Emitted(id) => id,
        other => panic!("expected Emitted, got {other:?}"),
    };

    let events = peek_events(
        store.conn(),
        "_test_concept_consumer",
        &[EventType::ConceptSummaryLlmJudge.as_str()],
        100,
    )
    .expect("peek events");
    let target = events.iter().find(|e| e.id == event_id).unwrap();
    assert_eq!(target.event_type, "concept_summary_llm_judge");
}

#[test]
fn dispatch_drops_when_daily_cap_reached() {
    let (store, _tmp) = temp_store();
    let extractor = ExtractorKind::Mock(MockExtractor::with_responses(vec![
        Ok("HIT: yes\nWHY: ok".to_string()),
        Ok("HIT: yes\nWHY: ok".to_string()),
        Ok("HIT: yes\nWHY: ok".to_string()),
    ]));
    // Cap = 2. Third dispatch should drop with DailyCapReached.
    for _ in 0..2 {
        let job = make_job(
            JudgeJobKind::Synthesis,
            "syn_cap",
            None,
            "q",
            "p",
            "c",
        );
        let res = dispatch_one(&store, &extractor, job, 2).expect("dispatch ok");
        assert!(matches!(res, DispatchResult::Emitted(_)));
    }
    let job = make_job(JudgeJobKind::Synthesis, "syn_cap2", None, "q", "p", "c");
    let res = dispatch_one(&store, &extractor, job, 2).expect("dispatch ok");
    match res {
        DispatchResult::Dropped(DropReason::DailyCapReached) => {}
        other => panic!("expected DailyCapReached, got {other:?}"),
    }
}

#[test]
fn dispatch_drops_on_unparseable_llm_output() {
    let (store, _tmp) = temp_store();
    let extractor = ExtractorKind::Mock(MockExtractor::with_fixed_response(
        "this is not the strict format",
    ));
    let job = make_job(
        JudgeJobKind::Synthesis,
        "syn_bad",
        None,
        "q",
        "p",
        "c",
    );
    let res = dispatch_one(&store, &extractor, job, LLM_JUDGE_DAILY_CALL_CAP_DEFAULT)
        .expect("dispatch ok");
    match res {
        DispatchResult::Dropped(DropReason::LlmError(msg)) => {
            assert!(msg.contains("unparseable") || msg.contains("verdict"));
        }
        other => panic!("expected LlmError(unparseable verdict), got {other:?}"),
    }
}

#[test]
fn dispatch_drops_on_empty_surface_id() {
    let (store, _tmp) = temp_store();
    let extractor = ExtractorKind::Mock(MockExtractor::with_fixed_response(
        "HIT: yes\nWHY: ok",
    ));
    // Empty surface_id violates J5 link-present.
    let job = make_job(JudgeJobKind::Synthesis, "", None, "q", "p", "c");
    let res = dispatch_one(&store, &extractor, job, LLM_JUDGE_DAILY_CALL_CAP_DEFAULT)
        .expect("dispatch ok");
    match res {
        DispatchResult::Dropped(DropReason::ContractViolation(msg)) => {
            assert!(msg.contains("LinkAbsent") || msg.contains("J5"));
        }
        other => panic!("expected ContractViolation (J5), got {other:?}"),
    }
}

#[test]
fn dispatch_drops_on_stamp_hash_mismatch() {
    let (store, _tmp) = temp_store();
    let extractor = ExtractorKind::Mock(MockExtractor::with_fixed_response(
        "HIT: yes\nWHY: ok",
    ));
    let mut job = make_job(JudgeJobKind::Synthesis, "syn_hash", None, "q", "p", "c");
    // Forge an invalid stamp_hash so J7 must catch the mismatch.
    job.stamp_hash = "deadbeef".to_string();
    let res = dispatch_one(&store, &extractor, job, LLM_JUDGE_DAILY_CALL_CAP_DEFAULT)
        .expect("dispatch ok");
    match res {
        DispatchResult::Dropped(DropReason::ContractViolation(msg)) => {
            assert!(msg.contains("stamp_hash"));
        }
        other => panic!("expected ContractViolation (J7), got {other:?}"),
    }
}

#[test]
fn dispatch_lets_llm_error_failures_drop_gracefully() {
    let (store, _tmp) = temp_store();
    let extractor = ExtractorKind::Mock(MockExtractor::with_persistent_error("upstream 503"));
    let job = make_job(JudgeJobKind::Synthesis, "syn_err", None, "q", "p", "c");
    let res = dispatch_one(&store, &extractor, job, LLM_JUDGE_DAILY_CALL_CAP_DEFAULT)
        .expect("dispatch never propagates LLM errors per J4");
    match res {
        DispatchResult::Dropped(DropReason::LlmError(_)) => {}
        other => panic!("expected LlmError, got {other:?}"),
    }
}

#[test]
fn parse_judge_output_invariants() {
    assert_eq!(
        parse_judge_output("HIT: yes\nWHY: faithful").unwrap(),
        (true, "faithful".to_string())
    );
    assert_eq!(
        parse_judge_output("HIT: NO\nWHY: hallucinated").unwrap(),
        (false, "hallucinated".to_string())
    );
    assert!(parse_judge_output("HIT: yes").is_none());
    assert!(parse_judge_output("WHY: only why").is_none());
}

#[test]
fn invariant_constants_are_stable() {
    // Anchor the bootstrap constants against accidental drift.
    assert_eq!(LLM_JUDGE_J3_MIN_PAIRS, 30);
    assert!((LLM_JUDGE_KAPPA_FLOOR - 0.6).abs() < 1e-9);
}
