//! v0.24 ARS Capability A — concept living-summary integration tests.
//!
//! Covers the end-to-end `run_concept_summary_with_extractor` path via
//! `MockExtractor`: gating, dry-run, eligibility selection (single + batch),
//! LLM error handling, and the atomic UPDATE to the three `living_summary*`
//! columns. Exercises real `should_refresh_living_summary` + real SQL
//! without a live LLM provider.

use chrono::Utc;
use rein::config::ReinConfig;
use rein::extract::{ExtractorKind, MockExtractor};
use rein::ops::concept_summary::run_concept_summary_with_extractor;
use rein::store::SqliteStore;
use rein::types::{Concept, Memoir};

fn make_memoir(name: &str) -> Memoir {
    Memoir {
        id: String::new(),
        name: name.to_string(),
        description: "integ memoir".to_string(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
    }
}

fn make_concept(memoir_name: &str, name: &str, revision: u32) -> Concept {
    Concept {
        id: String::new(),
        memoir_id: memoir_name.to_string(),
        name: name.to_string(),
        definition: format!("definition of {name}"),
        labels: vec!["integ".to_string()],
        source_memory_ids: vec![],
        confidence: 0.8,
        revision,
        last_episode_id: None,
        created_at: Utc::now(),
        updated_at: Utc::now(),
        living_summary: None,
        living_summary_updated_at: None,
        living_summary_source_revision: None,
    }
}

/// Create an in-memory store with one concept and `[ars].concept_summary_enabled = true`.
fn setup(revision: u32) -> (SqliteStore, ReinConfig, String) {
    let store = SqliteStore::in_memory().unwrap();
    store.create_memoir(make_memoir("integ-memoir")).unwrap();
    let concept_id = store
        .add_concept(make_concept("integ-memoir", "test-concept", revision))
        .unwrap();
    let mut config = ReinConfig::default();
    config.ars.concept_summary_enabled = true;
    (store, config, concept_id)
}

fn read_living_summary(
    store: &SqliteStore,
    id: &str,
) -> (Option<String>, Option<String>, Option<i64>) {
    store
        .conn()
        .query_row(
            "SELECT living_summary, living_summary_updated_at, living_summary_source_revision \
             FROM concepts WHERE id = ?1",
            rusqlite::params![id],
            |r| {
                Ok((
                    r.get::<_, Option<String>>(0)?,
                    r.get::<_, Option<String>>(1)?,
                    r.get::<_, Option<i64>>(2)?,
                ))
            },
        )
        .unwrap()
}

#[test]
fn skipped_when_ars_disabled_bypasses_llm() {
    let (store, mut config, id) = setup(10);
    config.ars.concept_summary_enabled = false;
    let mock = ExtractorKind::Mock(MockExtractor::with_fixed_response("unused"));
    let outcome =
        run_concept_summary_with_extractor(&store, &config, Some(&id), false, mock).unwrap();
    assert!(outcome.skipped_disabled);
    assert_eq!(outcome.attempted, 0);
    assert_eq!(outcome.succeeded, 0);
    let (ls, _, _) = read_living_summary(&store, &id);
    assert!(ls.is_none(), "disabled path must not touch living_summary");
}

#[test]
fn dry_run_reports_eligibility_without_writing() {
    let (store, config, id) = setup(10);
    let mock = ExtractorKind::Mock(MockExtractor::with_fixed_response("unused"));
    let outcome =
        run_concept_summary_with_extractor(&store, &config, Some(&id), true, mock).unwrap();
    assert!(outcome.dry_run);
    assert_eq!(
        outcome.attempted, 1,
        "dry_run counts eligible rows as attempted"
    );
    assert_eq!(outcome.succeeded, 0, "dry_run must not count succeeded");
    let (ls, _, _) = read_living_summary(&store, &id);
    assert!(ls.is_none(), "dry_run must not write living_summary");
}

#[test]
fn single_target_eligible_concept_receives_summary() {
    let (store, config, id) = setup(10);
    let summary = "A three-sentence synthesis of the concept's current state. \
         It references identifiers like A1 registry and inventory. \
         No prior living_summary existed.";
    let mock = ExtractorKind::Mock(MockExtractor::with_fixed_response(summary));
    let outcome =
        run_concept_summary_with_extractor(&store, &config, Some(&id), false, mock).unwrap();
    assert_eq!(outcome.attempted, 1);
    assert_eq!(outcome.succeeded, 1);
    assert_eq!(outcome.llm_failed, 0);
    let (ls, ls_at, ls_rev) = read_living_summary(&store, &id);
    assert_eq!(ls.as_deref(), Some(summary));
    assert!(
        ls_at.is_some(),
        "living_summary_updated_at must be populated"
    );
    // Source revision must match concept.revision at time of call (freeze semantics).
    assert_eq!(ls_rev, Some(10));
}

#[test]
fn single_target_ineligible_concept_is_skipped_but_counted() {
    // revision=2 is below bootstrap threshold 5 → trigger returns false.
    let (store, config, id) = setup(2);
    let mock = ExtractorKind::Mock(MockExtractor::with_fixed_response("unused"));
    let outcome =
        run_concept_summary_with_extractor(&store, &config, Some(&id), false, mock).unwrap();
    assert_eq!(outcome.attempted, 0);
    assert_eq!(outcome.succeeded, 0);
    assert_eq!(outcome.skipped_not_eligible, 1);
    let (ls, _, _) = read_living_summary(&store, &id);
    assert!(ls.is_none());
}

#[test]
fn single_target_missing_concept_is_counted() {
    let (store, config, _id) = setup(10);
    let mock = ExtractorKind::Mock(MockExtractor::with_fixed_response("unused"));
    let outcome =
        run_concept_summary_with_extractor(&store, &config, Some("nonexistent-id"), false, mock)
            .unwrap();
    assert_eq!(outcome.attempted, 0);
    assert_eq!(outcome.skipped_not_eligible, 1);
}

#[test]
fn llm_error_counted_and_concept_unchanged() {
    let (store, config, id) = setup(10);
    // `with_persistent_error` queues 4 errors, covering retries if the caller
    // implements them (current impl doesn't — one call, one error → one fail).
    let mock = ExtractorKind::Mock(MockExtractor::with_persistent_error("simulated outage"));
    let outcome =
        run_concept_summary_with_extractor(&store, &config, Some(&id), false, mock).unwrap();
    assert_eq!(outcome.attempted, 1);
    assert_eq!(outcome.succeeded, 0);
    assert_eq!(outcome.llm_failed, 1);
    let (ls, _, _) = read_living_summary(&store, &id);
    assert!(
        ls.is_none(),
        "failed LLM call must not write living_summary"
    );
}

#[test]
fn empty_llm_output_counts_as_llm_failed() {
    // Agent 1's empty-output guard keeps degenerate summaries out of the DB.
    let (store, config, id) = setup(10);
    let mock = ExtractorKind::Mock(MockExtractor::with_fixed_response("   "));
    let outcome =
        run_concept_summary_with_extractor(&store, &config, Some(&id), false, mock).unwrap();
    assert_eq!(outcome.attempted, 1);
    assert_eq!(outcome.succeeded, 0);
    assert_eq!(outcome.llm_failed, 1);
    let (ls, _, _) = read_living_summary(&store, &id);
    assert!(ls.is_none());
}

#[test]
fn batch_mode_processes_eligible_and_silently_filters_ineligible() {
    let (store, config, eligible_a) = setup(10);
    let eligible_b = store
        .add_concept(make_concept("integ-memoir", "concept-b", 7))
        .unwrap();
    let _ineligible = store
        .add_concept(make_concept("integ-memoir", "concept-c", 1))
        .unwrap();

    let mock = ExtractorKind::Mock(MockExtractor::with_responses(vec![
        Ok("summary alpha".to_string()),
        Ok("summary beta".to_string()),
    ]));
    let outcome = run_concept_summary_with_extractor(&store, &config, None, false, mock).unwrap();
    assert_eq!(
        outcome.attempted, 2,
        "only 2 concepts cross the revision gate"
    );
    assert_eq!(outcome.succeeded, 2);
    assert_eq!(outcome.llm_failed, 0);
    // Batch silently filters: skipped_not_eligible stays 0 (counter semantics
    // per agent 1's design — see ops/concept_summary.rs::select_eligible).
    assert_eq!(outcome.skipped_not_eligible, 0);

    let (ls_a, _, _) = read_living_summary(&store, &eligible_a);
    let (ls_b, _, _) = read_living_summary(&store, &eligible_b);
    assert!(ls_a.is_some());
    assert!(ls_b.is_some());
}

#[test]
fn batch_mode_respects_batch_size_cap() {
    let (store, mut config, _first_id) = setup(10);
    // Add 3 more eligible concepts → total 4 eligible.
    for i in 0..3 {
        store
            .add_concept(make_concept(
                "integ-memoir",
                &format!("concept-extra-{i}"),
                8,
            ))
            .unwrap();
    }
    config.ars.batch_size = 2; // cap to 2 per invocation

    let mock = ExtractorKind::Mock(MockExtractor::with_responses(vec![
        Ok("s1".to_string()),
        Ok("s2".to_string()),
    ]));
    let outcome = run_concept_summary_with_extractor(&store, &config, None, false, mock).unwrap();
    assert_eq!(outcome.attempted, 2, "batch_size=2 caps attempted at 2");
    assert_eq!(outcome.succeeded, 2);
}

#[test]
fn successful_refresh_emits_concept_summary_refreshed_event_and_populates_stats() {
    // L3 wiring end-to-end: a successful refresh should
    //   1. emit a ConceptSummaryRefreshed feedback event with sample payload
    //   2. recompute_concept_refresh_stats consumes that event into the
    //      AdaptiveState reservoir.
    let (store, config, id) = setup(10);
    let mock = ExtractorKind::Mock(MockExtractor::with_fixed_response("current state synthesis"));
    let outcome =
        run_concept_summary_with_extractor(&store, &config, Some(&id), false, mock).unwrap();
    assert_eq!(outcome.succeeded, 1);

    // (1) feedback event was written.
    let event_count: i64 = store
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM feedback_events WHERE event_type = 'concept_summary_refreshed'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(event_count, 1, "exactly one refresh event expected");

    let payload: String = store
        .conn()
        .query_row(
            "SELECT payload FROM feedback_events WHERE event_type = 'concept_summary_refreshed'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&payload).unwrap();
    // First-refresh semantics: revisions_since_last anchored to 0
    // (no prior `living_summary_source_revision`), so it equals
    // the concept's current revision (10). first_refresh = true so
    // recompute can exclude this sample's age from the percentile
    // (Codex round-2 MEDIUM).
    assert_eq!(parsed["revisions_since_last"], 10);
    assert!(parsed["age_secs_since_last"].as_i64().unwrap() >= 0);
    assert_eq!(parsed["first_refresh"], true);

    // (2) recompute drains the event into the reservoir; the sample
    // counts toward `count` (gates revision threshold) but NOT toward
    // `count_steady_state` (gates age threshold).
    let (stats, max_id) =
        rein::store::adaptive::recompute_concept_refresh_stats(store.conn(), None).unwrap();
    assert_eq!(stats.count, 1);
    assert_eq!(stats.count_steady_state, 0);
    assert_eq!(stats.samples[0].revisions_since_last, 10);
    assert!(stats.samples[0].first_refresh);
    assert!(
        max_id.is_some(),
        "recompute should report event id for caller to commit"
    );
}

#[test]
fn failed_refresh_does_not_emit_event() {
    // LLM error path must not emit a ConceptSummaryRefreshed event — the
    // event represents a *successful* refresh observation, not an attempt.
    let (store, config, id) = setup(10);
    let mock = ExtractorKind::Mock(MockExtractor::with_persistent_error("simulated outage"));
    let outcome =
        run_concept_summary_with_extractor(&store, &config, Some(&id), false, mock).unwrap();
    assert_eq!(outcome.llm_failed, 1);

    let event_count: i64 = store
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM feedback_events WHERE event_type = 'concept_summary_refreshed'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(event_count, 0, "failed refresh must not emit refresh event");
}

#[test]
fn summary_prompt_uses_most_recent_revisions_when_history_exceeds_limit() {
    let store = SqliteStore::in_memory().unwrap();
    store.create_memoir(make_memoir("integ-memoir")).unwrap();
    let concept_id = store
        .add_concept(make_concept("integ-memoir", "test-concept", 1))
        .unwrap();
    for revision in 2..=26 {
        store
            .refine_concept(
                "integ-memoir",
                "test-concept",
                &format!("definition revision {revision}"),
            )
            .unwrap();
    }

    let mut config = ReinConfig::default();
    config.ars.concept_summary_enabled = true;
    let (mock, probe) = MockExtractor::with_fixed_response_and_probe("current summary");

    let outcome = run_concept_summary_with_extractor(
        &store,
        &config,
        Some(&concept_id),
        false,
        ExtractorKind::Mock(mock),
    )
    .unwrap();

    assert_eq!(outcome.attempted, 1);
    assert_eq!(outcome.succeeded, 1);
    let prompt = probe
        .last_text_prompt()
        .expect("mock should record the concept-summary prompt");
    assert!(
        !prompt.contains("--- Revision #1 ("),
        "prompt should drop the oldest revision once history exceeds the limit:\n{prompt}"
    );
    assert!(
        !prompt.contains("--- Revision #5 ("),
        "prompt should keep only the newest 20 revisions:\n{prompt}"
    );
    let first_kept = prompt
        .find("--- Revision #6 (")
        .expect("prompt should keep revision 6 as the oldest item in the latest-20 window");
    let last_kept = prompt
        .find("--- Revision #25 (")
        .expect("prompt should keep the latest stored revision");
    assert!(
        first_kept < last_kept,
        "latest-20 revision window should still be chronological:\n{prompt}"
    );
}
