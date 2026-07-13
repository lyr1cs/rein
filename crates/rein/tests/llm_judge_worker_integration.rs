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

use rein::config::{JudgeStructuralAnchorMode, ReinConfig};
use rein::extract::llm::{ExtractorKind, MockExtractor};
use rein::judge::contract::LLM_JUDGE_DAILY_CALL_CAP_DEFAULT;
use rein::ops::llm_judge_worker::{
    dispatch_one, parse_judge_output, run_structural_anchor_suite, write_test_judge_cache_entry,
    DispatchResult, DropReason, JudgeJob, JudgeJobKind,
};
use rein::store::adaptive::{
    commit_offset, peek_events, AdaptiveState, ClusterConceptSummaryStats, ClusterSynthesisStats,
    ConceptSummaryFeedbackState, EventType, JudgeCalibrationState, JudgeSource, SignalHint,
    SynthesisFeedbackState, SynthesisLlmJudgePayload, LLM_JUDGE_J3_MIN_PAIRS,
    LLM_JUDGE_KAPPA_FLOOR,
};
use rein::store::SqliteStore;

fn temp_store() -> (SqliteStore, ReinConfig, tempfile::TempDir) {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let mut config = rein::config::ReinConfig::default();
    config.database.path = tmp
        .path()
        .join("memories.db")
        .to_string_lossy()
        .into_owned();
    config.hooks.buffer_dir = tmp.path().join("buffer").to_string_lossy().into_owned();
    config.ars.llm_judge.enabled = true;
    config.ars.llm_judge.cache_ttl_secs = 600;
    let store = config.open_store().expect("open store");
    (store, config, tmp)
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
        signal_hint: None,
    }
}

fn prepare_j5_target(store: &SqliteStore, config: &ReinConfig, job: &JudgeJob) {
    write_test_judge_cache_entry(config, job).expect("write judge cache entry");
    if matches!(job.kind, JudgeJobKind::ConceptSummary) {
        let concept_id = job.concept_id.as_deref().unwrap_or("concept-test");
        store
            .conn()
            .execute(
                "INSERT OR IGNORE INTO concept_summary_instances \
                 (summary_id, concept_id, summary_text, refreshed_at) \
                 VALUES (?1, ?2, ?3, ?4)",
                rusqlite::params![
                    &job.surface_id,
                    concept_id,
                    &job.candidate,
                    chrono::Utc::now().timestamp()
                ],
            )
            .expect("insert concept summary target");
    }
}

#[test]
fn dispatch_emits_synthesis_judge_event_on_hit() {
    let (store, config, _tmp) = temp_store();
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
    prepare_j5_target(&store, &config, &job);

    let res = dispatch_one(
        &store,
        &config,
        &extractor,
        job,
        LLM_JUDGE_DAILY_CALL_CAP_DEFAULT,
    )
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
    let payload: SynthesisLlmJudgePayload =
        serde_json::from_str(target.payload.as_deref().unwrap()).expect("payload deserializes");
    assert_eq!(payload.synthesis_id, "syn_test_1");
    assert!(payload.hit, "verdict should be HIT=yes");
    assert!(
        payload.reason.contains("evidence"),
        "reason preserved verbatim from mock"
    );
}

#[test]
fn dispatch_strips_signal_hint_when_acceleration_is_disabled() {
    let (store, mut config, _tmp) = temp_store();
    config.ars.acceleration.enabled = false;
    let extractor = ExtractorKind::Mock(MockExtractor::with_fixed_response(
        "HIT: yes\nWHY: synthesis cites every evidence id.",
    ));
    let mut job = make_job(
        JudgeJobKind::Synthesis,
        "syn_hint_disabled",
        None,
        "q",
        "p",
        "c",
    );
    job.signal_hint = Some(SignalHint {
        inferred_w_view: Some(1.0),
        inferred_w_click: Some(1.5),
        inferred_w_thumb: Some(2.0),
        inferred_w_req: Some(0.75),
        useful_rate_ci_width: Some(0.25),
    });
    prepare_j5_target(&store, &config, &job);

    let result = dispatch_one(
        &store,
        &config,
        &extractor,
        job,
        LLM_JUDGE_DAILY_CALL_CAP_DEFAULT,
    );

    assert!(matches!(result, Ok(DispatchResult::Emitted(_))));
    let events = peek_events(
        store.conn(),
        "test-signal-hint-disabled",
        &[EventType::SynthesisLlmJudge.as_str()],
        10,
    )
    .expect("peek");
    let payload: SynthesisLlmJudgePayload =
        serde_json::from_str(events[0].payload.as_ref().unwrap()).expect("parse payload");
    assert!(
        payload.signal_hint.is_none(),
        "disabled acceleration must strip queued signal hints"
    );
}

#[test]
fn dispatch_emits_sanitized_signal_hint_when_acceleration_is_enabled() {
    let (store, mut config, _tmp) = temp_store();
    config.ars.acceleration.enabled = true;
    config.ars.acceleration.shadow_only = false;
    let extractor = ExtractorKind::Mock(MockExtractor::with_fixed_response(
        "HIT: yes\nWHY: synthesis cites every evidence id.",
    ));
    let mut job = make_job(
        JudgeJobKind::Synthesis,
        "syn_hint_shadow",
        None,
        "q",
        "p",
        "c",
    );
    job.signal_hint = Some(SignalHint {
        inferred_w_view: Some(f64::NEG_INFINITY),
        inferred_w_click: Some(-0.1),
        inferred_w_thumb: Some(2.25),
        inferred_w_req: Some(0.75),
        useful_rate_ci_width: Some(1.25),
    });
    prepare_j5_target(&store, &config, &job);

    let result = dispatch_one(
        &store,
        &config,
        &extractor,
        job,
        LLM_JUDGE_DAILY_CALL_CAP_DEFAULT,
    );

    assert!(matches!(result, Ok(DispatchResult::Emitted(_))));
    let events = peek_events(
        store.conn(),
        "test-signal-hint-shadow",
        &[EventType::SynthesisLlmJudge.as_str()],
        10,
    )
    .expect("peek");
    let payload: SynthesisLlmJudgePayload =
        serde_json::from_str(events[0].payload.as_ref().unwrap()).expect("parse payload");
    let hint = payload
        .signal_hint
        .expect("enabled acceleration keeps valid hint fields");
    assert_eq!(hint.inferred_w_view, None);
    assert_eq!(hint.inferred_w_click, None);
    assert_eq!(hint.inferred_w_thumb, Some(2.25));
    assert_eq!(hint.inferred_w_req, Some(0.75));
    assert_eq!(hint.useful_rate_ci_width, None);
}

#[test]
fn dispatch_emits_concept_summary_judge_event_on_miss() {
    let (store, config, _tmp) = temp_store();
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
    prepare_j5_target(&store, &config, &job);

    let res = dispatch_one(
        &store,
        &config,
        &extractor,
        job,
        LLM_JUDGE_DAILY_CALL_CAP_DEFAULT,
    )
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
    let (store, config, _tmp) = temp_store();
    let extractor = ExtractorKind::Mock(MockExtractor::with_responses(vec![
        Ok("HIT: yes\nWHY: ok".to_string()),
        Ok("HIT: yes\nWHY: ok".to_string()),
        Ok("HIT: yes\nWHY: ok".to_string()),
    ]));
    // Cap = 2. Third dispatch should drop with DailyCapReached.
    for _ in 0..2 {
        let job = make_job(JudgeJobKind::Synthesis, "syn_cap", None, "q", "p", "c");
        prepare_j5_target(&store, &config, &job);
        let res = dispatch_one(&store, &config, &extractor, job, 2).expect("dispatch ok");
        assert!(matches!(res, DispatchResult::Emitted(_)));
    }
    let job = make_job(JudgeJobKind::Synthesis, "syn_cap2", None, "q", "p", "c");
    prepare_j5_target(&store, &config, &job);
    let res = dispatch_one(&store, &config, &extractor, job, 2).expect("dispatch ok");
    match res {
        DispatchResult::Dropped(DropReason::DailyCapReached) => {}
        other => panic!("expected DailyCapReached, got {other:?}"),
    }
}

#[test]
fn dispatch_drops_on_unparseable_llm_output() {
    let (store, config, _tmp) = temp_store();
    let extractor = ExtractorKind::Mock(MockExtractor::with_fixed_response(
        "this is not the strict format",
    ));
    let job = make_job(JudgeJobKind::Synthesis, "syn_bad", None, "q", "p", "c");
    prepare_j5_target(&store, &config, &job);
    let res = dispatch_one(
        &store,
        &config,
        &extractor,
        job,
        LLM_JUDGE_DAILY_CALL_CAP_DEFAULT,
    )
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
    let (store, config, _tmp) = temp_store();
    let extractor = ExtractorKind::Mock(MockExtractor::with_fixed_response("HIT: yes\nWHY: ok"));
    // Empty surface_id violates J5 link-present.
    let job = make_job(JudgeJobKind::Synthesis, "", None, "q", "p", "c");
    let res = dispatch_one(
        &store,
        &config,
        &extractor,
        job,
        LLM_JUDGE_DAILY_CALL_CAP_DEFAULT,
    )
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
    let (store, config, _tmp) = temp_store();
    let extractor = ExtractorKind::Mock(MockExtractor::with_fixed_response("HIT: yes\nWHY: ok"));
    let mut job = make_job(JudgeJobKind::Synthesis, "syn_hash", None, "q", "p", "c");
    // Forge an invalid stamp_hash so J7 must catch the mismatch.
    job.stamp_hash = "deadbeef".to_string();
    let res = dispatch_one(
        &store,
        &config,
        &extractor,
        job,
        LLM_JUDGE_DAILY_CALL_CAP_DEFAULT,
    )
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
    let (store, config, _tmp) = temp_store();
    let extractor = ExtractorKind::Mock(MockExtractor::with_persistent_error("upstream 503"));
    let job = make_job(JudgeJobKind::Synthesis, "syn_err", None, "q", "p", "c");
    prepare_j5_target(&store, &config, &job);
    let res = dispatch_one(
        &store,
        &config,
        &extractor,
        job,
        LLM_JUDGE_DAILY_CALL_CAP_DEFAULT,
    )
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

// ── v0.27.x C2 end-to-end integration tests ──────────────────────────────

/// Full pipeline: dispatch_one emits SynthesisLlmJudge → consumer
/// recompute_synthesis_feedback_stats_with_judge folds it into
/// per-cluster stats with the configured weight_decay_rate.
#[test]
fn dispatch_then_consumer_fold_updates_useful_rate() {
    use rein::store::adaptive::recompute_synthesis_feedback_stats_with_judge;
    use std::collections::HashMap;

    let (store, config, _tmp) = temp_store();
    let extractor = ExtractorKind::Mock(MockExtractor::with_fixed_response(
        "HIT: yes\nWHY: covers all source facts.",
    ));
    let job = make_job(
        JudgeJobKind::Synthesis,
        "synth_e2e_001",
        None,
        "what is rein?",
        "rein is a memory system",
        "rein is a multi-source cross-validated memory MCP server.",
    );
    prepare_j5_target(&store, &config, &job);
    let result = dispatch_one(
        &store,
        &config,
        &extractor,
        job,
        LLM_JUDGE_DAILY_CALL_CAP_DEFAULT,
    );
    assert!(matches!(result, Ok(DispatchResult::Emitted(_))));

    // Drain the emitted event into adaptive state via the consumer.
    let (state, _pairs, _calibration, max_id) = recompute_synthesis_feedback_stats_with_judge(
        store.conn(),
        None,
        HashMap::new(),
        Default::default(),
        0.3,
    )
    .expect("consumer fold succeeds");
    assert!(max_id.is_some(), "consumer should have advanced offset");
    // make_job sets cluster_id=Some(7), query_type=Some("Semantic").
    let key = rein::store::adaptive::synthesis_bucket_key(Some(7), "Semantic");
    let bucket = state
        .by_cluster
        .get(&key)
        .expect("bucket should exist after fold");
    assert_eq!(bucket.llm_judge_count, 1);
    assert_eq!(bucket.llm_judge_hit_count, 1);
    // useful_rate = w_thumb × 0.3 × 1.0 / (w_thumb × 0.3) = 1.0
    assert!(
        bucket.useful_rate > 0.99,
        "judge-only hit bucket should yield useful_rate ≈ 1.0, got {}",
        bucket.useful_rate
    );
}

/// Daily cap honored: when the rolling 24h count is at cap,
/// dispatch_one returns Dropped(DailyCapReached).
#[test]
fn dispatch_drops_when_cap_reached() {
    let (store, config, _tmp) = temp_store();
    let extractor = ExtractorKind::Mock(MockExtractor::with_fixed_response("HIT: yes\nWHY: ok."));
    // Fill the ledger up to a low custom cap.
    let cap: u64 = 2;
    for i in 0..cap as usize {
        let job = make_job(
            JudgeJobKind::Synthesis,
            &format!("synth_cap_{i}"),
            None,
            "q",
            "p",
            "c",
        );
        // Re-create the mock for each call (consumes one scripted response per dispatch).
        let extr = ExtractorKind::Mock(MockExtractor::with_fixed_response("HIT: yes\nWHY: ok."));
        prepare_j5_target(&store, &config, &job);
        let _ = dispatch_one(&store, &config, &extr, job, cap);
    }
    // Third job at cap=2 must be dropped without an LLM call.
    let job = make_job(
        JudgeJobKind::Synthesis,
        "synth_cap_overflow",
        None,
        "q",
        "p",
        "c",
    );
    prepare_j5_target(&store, &config, &job);
    let result = dispatch_one(&store, &config, &extractor, job, cap);
    assert!(
        matches!(
            result,
            Ok(DispatchResult::Dropped(DropReason::DailyCapReached))
        ),
        "expected DailyCapReached, got {result:?}"
    );
    // Verify the third event was NOT emitted to feedback_events.
    let synth_events = peek_events(
        store.conn(),
        "test-cap-counter",
        &[EventType::SynthesisLlmJudge.as_str()],
        100,
    )
    .expect("peek");
    assert_eq!(
        synth_events.len(),
        cap as usize,
        "exactly `cap` events should have been emitted"
    );
}

/// judge_model_override builds an alternate Gemini extractor with the
/// override model, and the emitted event records the override model.
#[test]
fn judge_model_override_records_override_model() {
    let (store, config, _tmp) = temp_store();
    let extractor = ExtractorKind::Mock(MockExtractor::with_fixed_response("HIT: yes\nWHY: ok."));
    let mut job = make_job(
        JudgeJobKind::Synthesis,
        "synth_override_001",
        None,
        "q",
        "p",
        "c",
    );
    job.judge_model_override = Some("custom-judge-model".to_string());
    // Mock extractor doesn't honor override (per spec); but the
    // emitted event SHOULD record the actual model used (which is
    // "mock" for MockExtractor). This anchors the v0.27.2 R8-P3 fix
    // that we record actual, never hallucinate the override.
    prepare_j5_target(&store, &config, &job);
    let result = dispatch_one(
        &store,
        &config,
        &extractor,
        job,
        LLM_JUDGE_DAILY_CALL_CAP_DEFAULT,
    );
    assert!(matches!(result, Ok(DispatchResult::Emitted(_))));
    let events = peek_events(
        store.conn(),
        "test-override",
        &[EventType::SynthesisLlmJudge.as_str()],
        10,
    )
    .expect("peek");
    let payload: SynthesisLlmJudgePayload =
        serde_json::from_str(events[0].payload.as_ref().unwrap()).expect("parse");
    // Mock path → override is ignored, recorded model is "mock".
    assert_eq!(payload.judge_model, "mock");
    assert_eq!(payload.synthesis_id, "synth_override_001");
    assert!(matches!(payload.source, JudgeSource::AutoSampled));
}

#[test]
fn structural_anchor_events_only_advance_the_independent_structural_state() {
    let (store, mut config, _tmp) = temp_store();
    config.ars.llm_judge.synthesis_enabled = true;
    config.ars.llm_judge.concept_summary_enabled = true;
    config.ars.llm_judge.structural_anchors.mode = JudgeStructuralAnchorMode::Monitor;
    config.ars.llm_judge.structural_anchors.interval_secs = 86_400;
    config.ars.llm_judge.daily_call_cap = 100;

    let mut synthesis = SynthesisFeedbackState {
        last_consumed_event_id: 41,
        total_events: 9,
        ..SynthesisFeedbackState::default()
    };
    synthesis.by_cluster.insert(
        "7|Semantic".to_string(),
        ClusterSynthesisStats {
            useful_rate: 0.73,
            viewed_count: 11,
            ..ClusterSynthesisStats::default()
        },
    );
    let mut concept = ConceptSummaryFeedbackState {
        last_consumed_event_id: 42,
        total_events: 10,
        ..ConceptSummaryFeedbackState::default()
    };
    concept.by_cluster.insert(
        "7|Semantic".to_string(),
        ClusterConceptSummaryStats {
            useful_rate: 0.61,
            viewed_count: 12,
            ..ClusterConceptSummaryStats::default()
        },
    );
    let mut calibration = JudgeCalibrationState {
        last_consumed_event_id_calibration: 43,
        runtime_vs_offline_kappa: 0.81,
        runtime_vs_offline_kappa_synthesis: 0.82,
        runtime_vs_offline_kappa_concept: 0.80,
        ..JudgeCalibrationState::default()
    };
    calibration.push_pair(
        rein::store::adaptive::JudgeSurface::Synthesis,
        true,
        true,
        100,
    );
    calibration.push_pair(
        rein::store::adaptive::JudgeSurface::ConceptSummary,
        false,
        false,
        100,
    );
    calibration
        .recent_pairs_runtime_vs_offline
        .push_back((true, true, 100));
    calibration
        .recent_pairs_runtime_vs_offline_synthesis
        .push_back((true, true, 100));
    calibration
        .recent_pairs_runtime_vs_offline_concept
        .push_back((false, false, 100));
    let before = AdaptiveState {
        version: 7,
        synthesis_feedback_stats: Some(synthesis),
        concept_summary_feedback_stats: Some(concept),
        judge_calibration_state: Some(calibration),
        ..AdaptiveState::default()
    };
    before
        .save_snapshot(store.conn())
        .expect("seed adaptive state");
    commit_offset(
        store.conn(),
        &[
            ("synthesis_feedback", 41),
            ("concept_summary_feedback", 42),
            (rein::ops::judge_calibration::JUDGE_CALIBRATION_CONSUMER, 43),
        ],
    )
    .expect("seed unrelated consumer watermarks");
    for index in 0..50 {
        rein::store::adaptive::emit_event(
            store.conn(),
            rein::store::adaptive::FeedbackEvent {
                event_type: EventType::Store,
                request_id: Some(format!("structural-isolation-filler-{index}")),
                memory_id: None,
                concept_id: None,
                query: None,
                query_type: None,
                topic: None,
                payload: None,
            },
        )
        .expect("seed event ids beyond legacy consumer watermarks");
    }

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
    let stats = run_structural_anchor_suite(&store, &config, &extractor, 1_000)
        .expect("run structural suite");
    assert_eq!(stats.emitted, 8);
    let structural = rein::ops::judge_calibration::run_judge_structural_anchor_consumer(&store)
        .expect("fold structural anchors");
    assert!(
        structural.last_event_id > 43,
        "anchor events must sit beyond every seeded legacy watermark"
    );
    assert_eq!(
        structural.synthesis.status,
        rein::judge::contract::JudgeStructuralStatus::Ready
    );
    assert_eq!(
        structural.concept_summary.status,
        rein::judge::contract::JudgeStructuralStatus::Ready
    );

    let consumer_input =
        AdaptiveState::restore_snapshot(store.conn()).expect("seeded adaptive state");
    let (synthesis_after_consume, synthesis_pairs, synthesis_calibration, synthesis_max_id) =
        rein::store::adaptive::recompute_synthesis_feedback_stats_with_judge(
            store.conn(),
            consumer_input.synthesis_feedback_stats.clone(),
            consumer_input.pending_kappa_half_pairs.clone(),
            consumer_input
                .judge_calibration_state
                .clone()
                .unwrap_or_default(),
            config.ars.llm_judge.weight_decay_rate,
        )
        .expect("run synthesis legacy consumer");
    assert_eq!(synthesis_max_id, None);
    assert_eq!(
        synthesis_after_consume,
        before.synthesis_feedback_stats.clone().unwrap()
    );
    assert_eq!(synthesis_pairs, before.pending_kappa_half_pairs);
    assert_eq!(
        synthesis_calibration,
        before.judge_calibration_state.clone().unwrap()
    );
    assert_eq!(
        synthesis_after_consume.by_cluster["7|Semantic"].useful_rate,
        0.73
    );

    let (concept_after_consume, concept_pairs, concept_calibration, concept_max_id) =
        rein::store::adaptive::recompute_concept_summary_feedback_stats_with_judge(
            store.conn(),
            consumer_input.concept_summary_feedback_stats.clone(),
            consumer_input.pending_kappa_half_pairs.clone(),
            consumer_input
                .judge_calibration_state
                .clone()
                .unwrap_or_default(),
            config.ars.llm_judge.weight_decay_rate,
        )
        .expect("run concept-summary legacy consumer");
    assert_eq!(concept_max_id, None);
    assert_eq!(
        concept_after_consume,
        before.concept_summary_feedback_stats.clone().unwrap()
    );
    assert_eq!(concept_pairs, before.pending_kappa_half_pairs);
    assert_eq!(
        concept_calibration,
        before.judge_calibration_state.clone().unwrap()
    );
    assert_eq!(
        concept_after_consume.by_cluster["7|Semantic"].useful_rate,
        0.61
    );

    let mut judge_consumer_state = consumer_input.clone();
    let judge_offset_batch = rein::ops::judge_calibration::run_judge_calibration_consumer(
        &store,
        &mut judge_consumer_state,
        None,
    );
    assert_eq!(judge_offset_batch, None);
    assert_eq!(
        judge_consumer_state.judge_calibration_state,
        before.judge_calibration_state
    );

    let after = AdaptiveState::restore_snapshot(store.conn()).expect("adaptive state remains");
    assert_eq!(
        serde_json::to_value(&after).unwrap(),
        serde_json::to_value(&before).unwrap(),
        "anchor events must not mutate AdaptiveState"
    );
    for (consumer, expected) in [
        ("synthesis_feedback", 41_i64),
        ("concept_summary_feedback", 42_i64),
        (
            rein::ops::judge_calibration::JUDGE_CALIBRATION_CONSUMER,
            43_i64,
        ),
    ] {
        let actual: i64 = store
            .conn()
            .query_row(
                "SELECT last_event_id FROM consumer_offsets WHERE consumer = ?1",
                [consumer],
                |row| row.get(0),
            )
            .expect("seeded consumer offset remains");
        assert_eq!(actual, expected, "anchor events advanced {consumer}");
    }
    let structural_offset: i64 = store
        .conn()
        .query_row(
            "SELECT last_event_id FROM consumer_offsets WHERE consumer = ?1",
            [rein::ops::judge_calibration::JUDGE_STRUCTURAL_ANCHOR_CONSUMER],
            |row| row.get(0),
        )
        .expect("independent structural offset");
    assert_eq!(structural_offset, structural.last_event_id);

    let observability = rein::ops::trust_measurement::collect_judge_calibration_observability(
        &store, &config, 1_000,
    );
    assert_eq!(observability.structural_watermark, structural.last_event_id);
    assert_eq!(observability.human_runtime_watermark, 43);
    assert_eq!(observability.synthesis.human.pair_count, 1);
    assert_eq!(observability.synthesis.runtime_vs_nightly.pair_count, 1);
    assert_eq!(observability.synthesis.structural.load_status, "loaded");
    assert_eq!(
        observability.synthesis.structural.status, "unknown",
        "the test-only Mock extractor has no config-resolvable production fingerprint"
    );
}
