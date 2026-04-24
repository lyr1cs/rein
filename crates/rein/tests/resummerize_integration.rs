//! v0.23 resummerize integration tests.
//!
//! These cover:
//! - the MergeInto cap → `needs_resummerize` flag wire-up
//! - gating behavior (`enabled = false`, no LLM provider)
//! - adaptive target_bytes end-to-end (cluster stats → `resummerize_target_bytes`)
//! - percentile recompute over a synthetic corpus
//! - **End-to-end resummerize via `MockExtractor`** — drives the real
//!   `run_resummerize_with_extractor` path, exercises the contract gate,
//!   atomic apply_resummerize transaction, pre-snapshot, counter-drift
//!   fix (M8), and claim cleanup without a live LLM.
//! - concurrent-claim lifecycle (H6)
//!
//! The `rein-eval resummerize run` harness runs this same surface against
//! real fixtures + real LLM once credentials are configured.

use rein::config::ReinConfig;
use rein::extract::MockExtractor;
use rein::ops::resummerize::run_resummerize_with_extractor;
use rein::store::adaptive::{
    recompute_canonical_length_stats, AdaptiveState, CanonicalLengthStats, MAX_RESUMMERIZE_TARGET,
    MIN_RESUMMERIZE_TARGET, RESUMMERIZE_BOOTSTRAP_TARGET,
};
use rein::store::SqliteStore;
use rein::types::*;

fn make_memory(topic: &str, content: &str) -> Memory {
    Memory {
        id: ulid::Ulid::new().to_string(),
        layer: Importance::Medium.auto_layer(),
        topic: topic.to_string(),
        summary: content.chars().take(SUMMARY_MAX_CHARS).collect(),
        content: content.to_string(),
        keywords: vec![],
        importance: Importance::Medium,
        source: Source::Manual,
        strength: 1.0,
        decay_lambda: 0.06 * Importance::Medium.decay_factor(),
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
        tier: MemoryTier::Warm,
        cluster_id: None,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        last_accessed: chrono::Utc::now(),
    }
}

/// Direct wire-up test for the MergeInto cap branch in
/// `store_with_dedup_resolved`. A merge that pushes content past the 10KB
/// cap must set `needs_resummerize = 1` alongside applying the keep-tail
/// stopgap. Both conditions are strict asserts — the test fails loudly if
/// the merge classifier picks a non-MergeInto path, since in that case
/// this test is not actually exercising the wire-up and would silently
/// green a future regression.
#[test]
fn merge_into_cap_sets_needs_resummerize_flag() {
    let store = SqliteStore::in_memory().unwrap();

    // Seed a canonical + snapshot evidence so dedup's "has evidence"
    // path treats it as a proper canonical and MergeInto is the natural
    // decision for a near-duplicate incoming memory. Start the content
    // just under the 10KB cap so the first merge append crosses it.
    let base_lines: Vec<String> = (0..950)
        .map(|i| format!("canonical fact line {i} with recurring tokens"))
        .collect();
    let base_content = base_lines.join("\n");
    let canonical = make_memory("resummerize-cap-test", &base_content);
    let canonical_id = store.store(canonical).unwrap();

    // Force-grow the canonical past the cap via direct UPDATE. This
    // guarantees the next merge entering `store_with_dedup_resolved`
    // will see len > MERGE_CONTENT_CAP regardless of what
    // `extract_unique_lines` produces. The point is to validate the
    // wire-up between the cap branch and the needs_resummerize column,
    // not the classifier's merge-decision heuristics.
    let over_cap = format!("{base_content}\n{}", "x".repeat(1_500));
    store
        .conn()
        .execute(
            "UPDATE memories SET content = ?1 WHERE id = ?2",
            rusqlite::params![&over_cap, &canonical_id],
        )
        .unwrap();

    // Incoming memory with ≥ 90% token overlap with the canonical, plus
    // one distinctive new line. Same topic + heavy overlap makes the
    // jaccard / containment dedup decision MergeInto for any realistic
    // similarity_threshold < 0.9.
    let mut incoming_lines = base_lines[..900].to_vec();
    incoming_lines.push("new unique fact line added by incoming".to_string());
    let incoming_content = incoming_lines.join("\n");
    let incoming = make_memory("resummerize-cap-test", &incoming_content);

    let returned_id = store.store_with_dedup(incoming, 0.3, 90).unwrap();
    assert_eq!(
        returned_id, canonical_id,
        "store_with_dedup should return the canonical's id when MergeInto fires"
    );

    let after = store.get(&canonical_id).unwrap();
    assert!(
        after.content.chars().count() <= 10_050,
        "keep-tail cap was not enforced: content is {} chars (expected ≤ 10_050). \
         The merge classifier likely took a non-MergeInto path, in which case \
         this test is not exercising the wire-up. Investigate store_with_dedup \
         dedup resolution for the test fixture.",
        after.content.chars().count()
    );

    let flag: i64 = store
        .conn()
        .query_row(
            "SELECT needs_resummerize FROM memories WHERE id = ?1",
            rusqlite::params![&canonical_id],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        flag, 1,
        "needs_resummerize must be set by the cap branch alongside keep-tail enforcement"
    );
}

// ── MockExtractor-driven end-to-end helpers ──────────────────────────────

/// Build a store + enabled config + a flagged canonical ready for
/// resummerize. Returns the canonical id.
fn setup_flagged_canonical(pre_content: &str) -> (SqliteStore, ReinConfig, String) {
    let store = SqliteStore::in_memory().unwrap();
    let canonical = make_memory("resummerize-e2e", pre_content);
    let canonical_id = store.store(canonical).unwrap();
    store
        .conn()
        .execute(
            "UPDATE memories SET needs_resummerize = 1 WHERE id = ?1",
            rusqlite::params![&canonical_id],
        )
        .unwrap();
    let mut config = ReinConfig::default();
    config.resummerize.enabled = true;
    (store, config, canonical_id)
}

fn canonical_merge_count(store: &SqliteStore, id: &str) -> i64 {
    store
        .conn()
        .query_row(
            "SELECT merge_count FROM memory_canonical_state WHERE memory_id = ?1",
            rusqlite::params![id],
            |r| r.get(0),
        )
        .unwrap()
}

fn canonical_support_count(store: &SqliteStore, id: &str) -> i64 {
    store
        .conn()
        .query_row(
            "SELECT support_count FROM memory_canonical_state WHERE memory_id = ?1",
            rusqlite::params![id],
            |r| r.get(0),
        )
        .unwrap()
}

fn canonical_status(store: &SqliteStore, id: &str) -> String {
    store
        .conn()
        .query_row(
            "SELECT status FROM memories WHERE id = ?1",
            rusqlite::params![id],
            |r| r.get(0),
        )
        .unwrap()
}

fn canonical_in_progress(store: &SqliteStore, id: &str) -> Option<String> {
    store
        .conn()
        .query_row(
            "SELECT in_progress_resummerize_at FROM memories WHERE id = ?1",
            rusqlite::params![id],
            |r| r.get::<_, Option<String>>(0),
        )
        .unwrap()
}

fn needs_resummerize_flag(store: &SqliteStore, id: &str) -> i64 {
    store
        .conn()
        .query_row(
            "SELECT needs_resummerize FROM memories WHERE id = ?1",
            rusqlite::params![id],
            |r| r.get(0),
        )
        .unwrap()
}

fn audit_statuses(store: &SqliteStore, id: &str) -> Vec<String> {
    let mut stmt = store
        .conn()
        .prepare(
            "SELECT status FROM resummerize_runs WHERE canonical_id = ?1 ORDER BY created_at ASC",
        )
        .unwrap();
    let rows = stmt
        .query_map(rusqlite::params![id], |r| r.get::<_, String>(0))
        .unwrap();
    rows.collect::<rusqlite::Result<Vec<_>>>().unwrap()
}

/// End-to-end success path through `MockExtractor`. Verifies:
/// - contract gate passes and canonical is rewritten
/// - pre-rewrite canonical is snapshotted into `memory_evidence` (H2+cleanup_caution)
/// - `merge_count` and `support_count` are NOT inflated (Codex M8)
/// - `status` stays `active` so later `recompute_canonical_length_stats`
///   keeps observing this row (M8)
/// - `needs_resummerize` flag is cleared, `in_progress_resummerize_at` is cleared
/// - one audit row is written with status=`success`
/// - the mock was called exactly once
#[test]
fn mock_success_rewrites_canonical_and_clears_flag() {
    let pre = "Pre-resummerize canonical with distinctive token XYZZY and supporting prose.";
    let (store, config, canonical_id) = setup_flagged_canonical(pre);
    let pre_merge_count = canonical_merge_count(&store, &canonical_id);
    let pre_support_count = canonical_support_count(&store, &canonical_id);
    // `store.store()` in `setup_flagged_canonical` created an initial
    // evidence snapshot; resummerize must add one MORE (the pre-rewrite
    // canonical). We record the before-count to verify the delta.
    let evidence_before = store
        .list_memory_evidence(&canonical_id, 100)
        .unwrap()
        .len();

    // Compressed output is a literal substring of the pre content, so all
    // its trigrams are present in the reference set and `no_new_facts`
    // trivially passes (the test is about plumbing, not about stressing
    // the contract's heuristics — those have their own unit coverage).
    let compressed = "canonical with distinctive token XYZZY".to_string();
    let mock =
        rein::extract::ExtractorKind::Mock(MockExtractor::with_fixed_response(compressed.clone()));
    let outcome =
        run_resummerize_with_extractor(&store, &config, Some(&canonical_id), false, mock).unwrap();

    assert_eq!(outcome.attempted, 1);
    assert_eq!(
        outcome.succeeded, 1,
        "mock output is a pre substring; contract should pass. \
         outcome = {outcome:?}"
    );
    assert_eq!(outcome.contract_failed, 0);
    assert_eq!(outcome.llm_failed, 0);
    assert_eq!(outcome.length_exceeded, 0);

    let current = store.get(&canonical_id).unwrap();
    assert_eq!(current.content, compressed);
    assert_eq!(needs_resummerize_flag(&store, &canonical_id), 0);
    assert_eq!(canonical_in_progress(&store, &canonical_id), None);

    // Counters must not inflate across resummerize (Codex M8).
    assert_eq!(
        canonical_merge_count(&store, &canonical_id),
        pre_merge_count,
        "M8: merge_count inflated by resummerize"
    );
    assert_eq!(
        canonical_support_count(&store, &canonical_id),
        pre_support_count,
        "M8: support_count inflated by resummerize"
    );
    assert_eq!(
        canonical_status(&store, &canonical_id),
        "active",
        "M8: status flipped to 'updated'; recompute_canonical_length_stats \
         would silently skip this row on next pass"
    );

    // Pre-rewrite canonical landed in evidence as an ADDITIONAL row on top
    // of the initial store() snapshot (H2 + feedback_cleanup_caution).
    let evidence_after = store.list_memory_evidence(&canonical_id, 100).unwrap();
    assert_eq!(
        evidence_after.len(),
        evidence_before + 1,
        "resummerize must add exactly one pre-rewrite evidence row"
    );
    assert!(
        evidence_after
            .iter()
            .any(|e| e.content.contains("XYZZY") && e.content.contains("distinctive")),
        "pre-resummerize canonical content must be present in memory_evidence"
    );

    // Exactly one audit row, status=success.
    assert_eq!(audit_statuses(&store, &canonical_id), vec!["success"]);
}

/// Contract-violating LLM output MUST NOT mutate the canonical. The mock
/// returns a response that drops a recurring temporal anchor — the
/// `temporal_anchors_preserved` invariant catches it and aborts the
/// rewrite. Keep-tail canonical remains in place, flag stays set, audit
/// row records the violation. Demonstrates H2's "atomic or nothing"
/// guarantee end-to-end.
#[test]
fn mock_contract_violation_leaves_canonical_untouched() {
    // Seed content with two occurrences of a temporal anchor so the
    // invariant enforces preservation.
    let pre = "Canonical body mentioning 2026-04-22 twice: 2026-04-22 is the release date.";
    let (store, config, canonical_id) = setup_flagged_canonical(pre);
    let pre_content = store.get(&canonical_id).unwrap().content.clone();
    // Baseline BEFORE the run: the initial store() snapshot is already
    // present. Contract-violation path must NOT add a phantom pre-snapshot
    // (H2 atomicity + Codex round-2 L2 ordering fix).
    let evidence_before = store
        .list_memory_evidence(&canonical_id, 100)
        .unwrap()
        .len();

    // Mock response drops the anchor entirely → `temporal_anchors_preserved`
    // fails → contract gate rejects.
    let bad = "Vague compressed summary with no date at all.".to_string();
    let mock = rein::extract::ExtractorKind::Mock(MockExtractor::with_fixed_response(bad));
    let outcome =
        run_resummerize_with_extractor(&store, &config, Some(&canonical_id), false, mock).unwrap();

    assert_eq!(outcome.attempted, 1);
    assert_eq!(outcome.succeeded, 0);
    assert_eq!(outcome.contract_failed, 1);

    // Canonical content unchanged. Flag stays set (row re-enters the
    // backlog for a future worker, subject to the 3-strike fuse).
    assert_eq!(store.get(&canonical_id).unwrap().content, pre_content);
    assert_eq!(needs_resummerize_flag(&store, &canonical_id), 1);
    // Claim released so another worker can retry.
    assert_eq!(canonical_in_progress(&store, &canonical_id), None);
    // Audit captured the violation with status=contract_violation.
    assert_eq!(
        audit_statuses(&store, &canonical_id),
        vec!["contract_violation"]
    );
    // No phantom pre-snapshot from the failed run.
    assert_eq!(
        store
            .list_memory_evidence(&canonical_id, 100)
            .unwrap()
            .len(),
        evidence_before,
        "contract violation must not write a pre-snapshot evidence row"
    );
}

/// LLM error path: mock raises an error, op records it, canonical
/// untouched, flag remains set, claim released. No atomicity-violation
/// side effects (no phantom snapshot, no partial write).
#[test]
fn mock_llm_error_leaves_canonical_untouched() {
    let (store, config, canonical_id) =
        setup_flagged_canonical("canonical body unchanged after LLM outage");
    let pre_content = store.get(&canonical_id).unwrap().content.clone();
    // Capture baseline BEFORE the run (Codex round-2 L2): if we capture
    // after, a regression that writes a phantom snapshot would still
    // pass because both the "before" and "after" reads include it.
    let evidence_before = store
        .list_memory_evidence(&canonical_id, 100)
        .unwrap()
        .len();

    let mock = rein::extract::ExtractorKind::Mock(MockExtractor::with_persistent_error(
        "simulated API outage",
    ));
    let outcome =
        run_resummerize_with_extractor(&store, &config, Some(&canonical_id), false, mock).unwrap();

    assert_eq!(outcome.attempted, 1);
    assert_eq!(outcome.llm_failed, 1);
    assert_eq!(store.get(&canonical_id).unwrap().content, pre_content);
    assert_eq!(needs_resummerize_flag(&store, &canonical_id), 1);
    assert_eq!(canonical_in_progress(&store, &canonical_id), None);
    assert_eq!(
        store
            .list_memory_evidence(&canonical_id, 100)
            .unwrap()
            .len(),
        evidence_before,
        "LLM error path must not write a pre-snapshot evidence row"
    );
    assert_eq!(audit_statuses(&store, &canonical_id), vec!["llm_error"]);
}

/// 3-strike exhaustion fuse. After three consecutive **contract** failures
/// against the same canonical, the flag is cleared (so the LLM stops
/// being hit on a structurally broken case) and one audit row is written
/// with status=`exhausted`.
///
/// Post-fix semantics (Agent A HIGH + Agent D Q14): `llm_error` rows are
/// treated as transient and do NOT count toward the fuse; only the
/// deterministic LLM-quality failures (`contract_violation`,
/// `length_exceeded`) count. See the companion test
/// `mock_llm_errors_do_not_trip_three_strike_fuse` for that non-counting
/// path, and the unit test
/// `store::resummerize_audit::tests::consecutive_failures_fuse_logic`
/// for the full matrix.
#[test]
fn mock_three_strike_fuse_exhausts_after_consecutive_contract_failures() {
    let (store, config, canonical_id) =
        setup_flagged_canonical("canonical body subject to repeated contract failure");

    // Drive three separate run invocations, each mocking an LLM response
    // that will fail the Lossless Compression Contract (empty string fails
    // `no_new_facts` trivially among other invariants). Each recorded
    // audit row is `contract_violation`, which DOES count toward the
    // fuse.
    for attempt in 0..3 {
        let mock = rein::extract::ExtractorKind::Mock(MockExtractor::with_fixed_response(format!(
            "bogus-{attempt}"
        )));
        let outcome =
            run_resummerize_with_extractor(&store, &config, Some(&canonical_id), false, mock)
                .unwrap();
        assert_eq!(
            outcome.contract_failed, 1,
            "attempt {attempt}: should record contract failure"
        );
    }

    // Fourth run: count_recent_consecutive_failures sees 3 contract
    // violations and the fuse records exhaustion, clears the flag. The
    // mock is never actually called on this attempt because the fuse runs
    // before the LLM call.
    let mock = rein::extract::ExtractorKind::Mock(MockExtractor::with_responses(vec![]));
    let outcome =
        run_resummerize_with_extractor(&store, &config, Some(&canonical_id), false, mock).unwrap();
    assert_eq!(outcome.exhausted, 1);

    assert_eq!(
        needs_resummerize_flag(&store, &canonical_id),
        0,
        "3-strike fuse must clear the flag so this row stops being picked"
    );
    let statuses = audit_statuses(&store, &canonical_id);
    assert_eq!(statuses.len(), 4);
    assert_eq!(
        statuses[..3],
        [
            "contract_violation".to_string(),
            "contract_violation".to_string(),
            "contract_violation".to_string()
        ]
    );
    assert_eq!(statuses[3], "exhausted");
}

/// Agent A HIGH finding: transient LLM errors (network blip / 429 / 5xx)
/// must NOT count toward the 3-strike exhaustion fuse — otherwise a
/// provider outage permanently strands every flagged canonical. Only the
/// deterministic LLM-quality failure classes (`contract_violation`,
/// `length_exceeded`) count. Persistent API issues stay operator-visible
/// via `recent_failure_rate`, which still counts `llm_error`.
#[test]
fn mock_llm_errors_do_not_trip_three_strike_fuse() {
    let (store, config, canonical_id) =
        setup_flagged_canonical("canonical body subject to LLM outage");

    // Drive five separate run invocations, each raising an LLM error.
    // Pre-fix behavior: fuse trips after 3 and the flag clears. Post-fix
    // behavior: none count, flag stays set, canonical stays eligible.
    for attempt in 0..5 {
        let mock = rein::extract::ExtractorKind::Mock(MockExtractor::with_persistent_error(
            format!("outage #{}", attempt + 1),
        ));
        let outcome =
            run_resummerize_with_extractor(&store, &config, Some(&canonical_id), false, mock)
                .unwrap();
        assert_eq!(
            outcome.llm_failed, 1,
            "attempt {attempt}: should record llm failure"
        );
        assert_eq!(
            outcome.exhausted, 0,
            "attempt {attempt}: transient errors must not exhaust"
        );
    }

    assert_eq!(
        needs_resummerize_flag(&store, &canonical_id),
        1,
        "flag must stay set — transient LLM errors should not strand the row"
    );
    let statuses = audit_statuses(&store, &canonical_id);
    assert_eq!(statuses.len(), 5);
    for s in &statuses {
        assert_eq!(s, "llm_error");
    }
}

/// When a single `canonical_id` is targeted but the row is ineligible
/// (e.g. flag off), the op returns `attempted = 0` and does not touch
/// the canonical or write an audit row. Observability-via-side-effects:
/// we assert content unchanged + no `resummerize_runs` entry + no
/// `memory_evidence` snapshot.
#[test]
fn mock_targeted_canonical_ineligible_is_skipped() {
    let store = SqliteStore::in_memory().unwrap();
    // Store a canonical but DO NOT flip needs_resummerize.
    let canonical = make_memory("t", "body");
    let canonical_id = store.store(canonical).unwrap();
    let mut config = ReinConfig::default();
    config.resummerize.enabled = true;
    // Baseline BEFORE the run (Codex round-2 L2): capture evidence count
    // now so a regression that writes a phantom snapshot during an
    // ineligible run would be detectable. Capturing after the run
    // defeats the check (both samples would include any phantom row).
    let evidence_before = store
        .list_memory_evidence(&canonical_id, 100)
        .unwrap()
        .len();

    let mock =
        rein::extract::ExtractorKind::Mock(MockExtractor::with_fixed_response("would-rewrite"));
    let outcome =
        run_resummerize_with_extractor(&store, &config, Some(&canonical_id), false, mock).unwrap();

    assert_eq!(outcome.attempted, 0);
    assert_eq!(outcome.succeeded, 0);
    // Canonical content unchanged.
    assert_eq!(store.get(&canonical_id).unwrap().content, "body");
    // No audit row was written — the eligibility gate short-circuits
    // before `insert_resummerize_run` fires.
    assert!(audit_statuses(&store, &canonical_id).is_empty());
    // No NEW evidence snapshot was taken (ineligible row never reaches
    // apply_resummerize's pre-snapshot).
    assert_eq!(
        store
            .list_memory_evidence(&canonical_id, 100)
            .unwrap()
            .len(),
        evidence_before
    );
}

/// When config disables resummerize, `run_resummerize` returns early
/// without touching the LLM or mutating any rows.
#[test]
fn resummerize_skipped_when_disabled() {
    let store = SqliteStore::in_memory().unwrap();
    let config = ReinConfig::default();
    assert!(!config.resummerize.enabled, "default must be disabled");

    let outcome = rein::ops::resummerize::run_resummerize(&store, &config, None, false).unwrap();

    assert!(outcome.skipped_disabled);
    assert!(!outcome.skipped_no_llm);
    assert_eq!(outcome.attempted, 0);
    assert_eq!(outcome.succeeded, 0);
}

/// When resummerize is enabled but no LLM provider is configured
/// (default test config has no API key), the op returns `skipped_no_llm`
/// rather than failing or looping.
#[test]
fn resummerize_skipped_when_no_llm_configured() {
    let store = SqliteStore::in_memory().unwrap();
    let mut config = ReinConfig::default();
    config.resummerize.enabled = true;
    // extract.google.api_key defaults to None → create_extractor() → None.

    let outcome = rein::ops::resummerize::run_resummerize(&store, &config, None, false).unwrap();

    assert!(outcome.skipped_no_llm);
    assert!(!outcome.skipped_disabled);
    assert_eq!(outcome.attempted, 0);
}

/// Backlog count reports only active rows flagged for resummerize.
#[test]
fn backlog_count_tracks_flagged_canonicals() {
    let store = SqliteStore::in_memory().unwrap();
    let m = make_memory("t", "content");
    let id = store.store(m).unwrap();

    assert_eq!(rein::ops::resummerize::backlog_count(&store).unwrap(), 0);

    store
        .conn()
        .execute(
            "UPDATE memories SET needs_resummerize = 1 WHERE id = ?1",
            rusqlite::params![&id],
        )
        .unwrap();
    assert_eq!(rein::ops::resummerize::backlog_count(&store).unwrap(), 1);

    // Soft-delete style status flip should drop the row from backlog.
    store
        .conn()
        .execute(
            "UPDATE memories SET status = 'deprecated' WHERE id = ?1",
            rusqlite::params![&id],
        )
        .unwrap();
    assert_eq!(rein::ops::resummerize::backlog_count(&store).unwrap(), 0);
}

/// AdaptiveState target_bytes fallback chain is honored end-to-end.
#[test]
fn target_bytes_fallback_chain() {
    let mut state = AdaptiveState::default();

    // Empty state → bootstrap.
    assert_eq!(
        state.resummerize_target_bytes(None),
        RESUMMERIZE_BOOTSTRAP_TARGET
    );
    assert_eq!(
        state.resummerize_target_bytes(Some(7)),
        RESUMMERIZE_BOOTSTRAP_TARGET
    );

    // Global stats with enough samples → global p25.
    state.global_canonical_length = Some(CanonicalLengthStats {
        count: 50,
        p25: 4_500,
        p50: 6_000,
        p75: 7_500,
    });
    assert_eq!(state.resummerize_target_bytes(None), 4_500);
    assert_eq!(state.resummerize_target_bytes(Some(7)), 4_500);

    // Cluster stats win over global when sample count is sufficient.
    state.canonical_length_stats.insert(
        7,
        CanonicalLengthStats {
            count: 10,
            p25: 3_200,
            p50: 4_500,
            p75: 6_000,
        },
    );
    assert_eq!(state.resummerize_target_bytes(Some(7)), 3_200);
    // Unknown cluster falls back to global p25.
    assert_eq!(state.resummerize_target_bytes(Some(99)), 4_500);

    // Under-sampled cluster falls back to global.
    state.canonical_length_stats.insert(
        8,
        CanonicalLengthStats {
            count: 2, // below RESUMMERIZE_CLUSTER_MIN_SAMPLES
            p25: 1_000,
            p50: 1_500,
            p75: 2_000,
        },
    );
    assert_eq!(state.resummerize_target_bytes(Some(8)), 4_500);
}

/// target_bytes is always clamped to [MIN, MAX] regardless of raw
/// percentile magnitude. Guarantees structurally-valid targets even on
/// pathological input distributions.
#[test]
fn target_bytes_is_always_clamped() {
    // p25 well below MIN → clamp up.
    let state_low = AdaptiveState {
        global_canonical_length: Some(CanonicalLengthStats {
            count: 100,
            p25: 500,
            p50: 800,
            p75: 1_200,
        }),
        ..AdaptiveState::default()
    };
    assert_eq!(
        state_low.resummerize_target_bytes(None),
        MIN_RESUMMERIZE_TARGET
    );

    // p25 above MAX → clamp down.
    let state_high = AdaptiveState {
        global_canonical_length: Some(CanonicalLengthStats {
            count: 100,
            p25: 50_000,
            p50: 80_000,
            p75: 120_000,
        }),
        ..AdaptiveState::default()
    };
    assert_eq!(
        state_high.resummerize_target_bytes(None),
        MAX_RESUMMERIZE_TARGET
    );
}

/// `recompute_canonical_length_stats` pulls per-cluster distributions
/// directly from SQLite, producing the stats AdaptiveState later persists.
#[test]
fn recompute_canonical_length_stats_reads_active_canonicals() {
    let store = SqliteStore::in_memory().unwrap();

    // Insert canonicals with mixed cluster assignments.
    let ids: Vec<String> = (0..20)
        .map(|i| {
            let content: String = "x".repeat(100 + i * 50);
            let m = make_memory(&format!("topic-{i}"), &content);
            store.store(m).unwrap()
        })
        .collect();

    // Assign first 10 to cluster 1, next 10 to cluster 2.
    for (i, id) in ids.iter().enumerate() {
        let cid = if i < 10 { 1 } else { 2 };
        store
            .conn()
            .execute(
                "UPDATE memories SET cluster_id = ?1 WHERE id = ?2",
                rusqlite::params![cid, id],
            )
            .unwrap();
    }

    let (per_cluster, global) = recompute_canonical_length_stats(store.conn()).unwrap();

    let global = global.expect("global stats populated");
    assert_eq!(global.count, 20);
    // Smallest 25% among 20 = first 5 canonicals: lengths 100..300.
    // p25 ≈ lengths[(0.25 * 19)] = lengths[4.75] ≈ interpolated between
    // the 5th and 6th smallest.
    assert!(global.p25 >= 100 && global.p25 <= 400, "p25={}", global.p25);

    let c1 = per_cluster.get(&1).expect("cluster 1 stats populated");
    assert_eq!(c1.count, 10);
    let c2 = per_cluster.get(&2).expect("cluster 2 stats populated");
    assert_eq!(c2.count, 10);
    // Cluster 2 memories have the larger content tail, so its p25 > c1.p25.
    assert!(
        c2.p25 > c1.p25,
        "cluster 2 should have larger p25 than cluster 1 ({} vs {})",
        c2.p25,
        c1.p25
    );
}

/// Tempfile-backed end-to-end test that validates the side-index
/// refresh path actually fires (the :memory: store path skips both
/// Tantivy and HNSW — see `store/sqlite.rs` `with_tantivy` and
/// `with_hnsw_lock`, both return early for `:memory:`). The existing
/// MockExtractor tests in this file exercise the flag/claim/audit
/// plumbing but not the real-filesystem side-index work introduced by
/// Codex rounds 3 and 4.
///
/// What this test specifically covers:
/// - `SqliteStore::new(path, ..)` path (vs `in_memory()`)
/// - successful resummerize rewrites canonical content
/// - sqlite-vec row (`vec_memories` SQL table) is deleted inside the
///   transaction — Codex round-4 HIGH-2 guard, repeated here against a
///   real on-disk DB to catch any `:memory:` vs filesystem divergence
/// - `needs_vec_dedup = 1` flag is set so the adaptive pipeline
///   regenerates the embedding
/// - FTS search returns the new content (proves the FTS5 trigger path
///   stayed in sync across the direct SQL UPDATE in `apply_resummerize`)
/// - FTS search does NOT return the pre-rewrite content (proves a stale
///   token from before resummerize no longer matches)
/// - The Tantivy index directory exists on disk after resummerize (proxy
///   for "side-index refresh actually ran"; observing Tantivy content
///   directly from integration tests requires the recall pipeline and
///   is out of scope here).
#[test]
fn tempfile_resummerize_syncs_side_indexes_end_to_end() {
    let tmp = tempfile::tempdir().unwrap();
    let db_path = tmp.path().join("resummerize_side_index.db");
    let store = SqliteStore::new(&db_path, "test-model", 3072).unwrap();

    // Seed canonical with a PRE-only marker (in content) and a SHARED
    // token (in both pre and post) so we can tell which content the FTS
    // is actually indexing.
    let pre_content =
        "Pre-rewrite canonical body: SHAREDTOKEN, plus PREONLYMARKER for distinction.";
    let memory = make_memory("resummerize-tempfile-test", pre_content);
    let canonical_id = store.store(memory).unwrap();

    // Sanity: both tokens visible in FTS before resummerize runs.
    let hits_pre = store.search_fts("PREONLYMARKER", None, 10).unwrap();
    assert!(
        hits_pre.iter().any(|m| m.id == canonical_id),
        "fixture precondition: PREONLYMARKER should be findable before resummerize"
    );

    // Flag for resummerize.
    store
        .conn()
        .execute(
            "UPDATE memories SET needs_resummerize = 1 WHERE id = ?1",
            rusqlite::params![&canonical_id],
        )
        .unwrap();

    // Run resummerize with a mock that returns a literal substring of
    // the pre content so every trigram is present in the input superset
    // and `no_new_facts` passes. The new content drops PREONLYMARKER.
    let post_content = "canonical body: SHAREDTOKEN";
    let mut config = ReinConfig::default();
    config.resummerize.enabled = true;
    let mock = rein::extract::ExtractorKind::Mock(MockExtractor::with_fixed_response(
        post_content.to_string(),
    ));
    let outcome =
        run_resummerize_with_extractor(&store, &config, Some(&canonical_id), false, mock).unwrap();

    assert_eq!(
        outcome.succeeded, 1,
        "fixture expected contract-clean rewrite; outcome = {outcome:?}"
    );

    // Canonical content rewritten on disk.
    let current = store.get(&canonical_id).unwrap();
    assert_eq!(current.content, post_content);

    // sqlite-vec row deleted — Codex round-4 HIGH-2. The row had no
    // embedding to begin with (we didn't seed one), so the precondition
    // is "no vec row" and the assertion is "still no vec row". The
    // meaningful guard is the :memory: Codex test; this tempfile variant
    // mainly proves the delete call doesn't panic on a real DB.
    assert!(
        rein::store::vec::get_embedding(store.conn(), &canonical_id)
            .unwrap()
            .is_none(),
        "vec_memories row must not exist after resummerize"
    );

    // needs_vec_dedup = 1 so the adaptive pipeline regenerates.
    let needs_vec_dedup: i64 = store
        .conn()
        .query_row(
            "SELECT needs_vec_dedup FROM memories WHERE id = ?1",
            rusqlite::params![&canonical_id],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(needs_vec_dedup, 1);

    // FTS sees the new content (SHAREDTOKEN).
    let hits_shared = store.search_fts("SHAREDTOKEN", None, 10).unwrap();
    assert!(
        hits_shared.iter().any(|m| m.id == canonical_id),
        "post-rewrite content must be FTS-indexed; hits = {}",
        hits_shared.len()
    );

    // FTS does NOT see the pre-only marker (proves the SQL UPDATE path
    // propagated to the FTS5 virtual table via the memories_au trigger).
    let hits_pre_after = store.search_fts("PREONLYMARKER", None, 10).unwrap();
    assert!(
        !hits_pre_after.iter().any(|m| m.id == canonical_id),
        "PREONLYMARKER must NOT be in the canonical after resummerize"
    );

    // Tantivy index directory exists on disk — proxy for "the side-index
    // refresh path ran on a non-:memory: store". The directory is
    // created lazily on first write, so its presence after resummerize
    // confirms `with_tantivy` actually fired (on :memory: it would
    // early-return and never create the dir).
    let tantivy_dir = db_path.with_extension("tantivy");
    assert!(
        tantivy_dir.exists(),
        "Tantivy index dir should exist at {} after resummerize",
        tantivy_dir.display()
    );
}

/// Codex round-5 H-2 regression guard: `run_vec_dedup` used to clear
/// `needs_vec_dedup = 0` on embed / search failure branches, so one
/// transient error permanently stripped a canonical's vector recall.
/// Post-fix the flag must be preserved on failure so the next sweep
/// retries.
///
/// Uses `#[tokio::test(flavor = "multi_thread")]` because
/// `embed_vec_dedup_batch` calls `tokio::task::block_in_place +
/// Handle::current().block_on(...)` which requires an active multi-thread
/// runtime. Without this annotation the embed path panics and the test
/// "passes" for the wrong reason (caught panic → None → flag preserved
/// regardless of the real fix).
#[tokio::test(flavor = "multi_thread")]
async fn vec_dedup_preserves_flag_on_embed_failure() {
    use rein::embed::{EmbedderKind, MockEmbedder};
    use rein::ops::dedup::run_vec_dedup_with_embedder;

    let store = SqliteStore::in_memory().unwrap();
    let memory = make_memory("h2-flag-retention-test", "body content");
    let canonical_id = store.store(memory).unwrap();

    // Flag the row for vec-dedup sweep. No pre-existing embedding, so
    // the embed step is the critical path.
    store
        .conn()
        .execute(
            "UPDATE memories SET needs_vec_dedup = 1 WHERE id = ?1",
            rusqlite::params![&canonical_id],
        )
        .unwrap();

    // Configure resummerize (which in turn pulls the adaptive provider
    // so create_embedder would normally return a real client) is
    // irrelevant here — we override the embedder directly.
    let config = rein::config::ReinConfig::default();

    // Mock embedder that fails every call, simulating a persistent API
    // outage. Dims don't matter because the embed call fails before any
    // vector is handled.
    let mock = EmbedderKind::Mock(MockEmbedder::with_persistent_error(
        3072,
        "simulated API outage",
    ));
    run_vec_dedup_with_embedder(&store, &config, mock);

    let flag: i64 = store
        .conn()
        .query_row(
            "SELECT needs_vec_dedup FROM memories WHERE id = ?1",
            rusqlite::params![&canonical_id],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        flag, 1,
        "needs_vec_dedup must remain 1 after embed failure so the next sweep can retry; \
         round-5 H-2 regression"
    );
}

/// Codex round-5 H-3 regression guard: after `apply_resummerize` evicts
/// the stale HNSW entry, a subsequent `run_vec_dedup` sweep with fresh
/// embedding input must re-insert into HNSW (not only into sqlite-vec).
/// Uses a tempfile store because `:memory:` skips HNSW entirely, and
/// `#[tokio::test(flavor = "multi_thread")]` because the embed path
/// requires an active runtime (see H-2 test above).
#[tokio::test(flavor = "multi_thread")]
async fn vec_dedup_reinserts_into_hnsw_after_successful_embed() {
    use rein::embed::{EmbedderKind, MockEmbedder};
    use rein::ops::dedup::run_vec_dedup_with_embedder;
    use rein::store::hnsw::HnswIndex;

    let tmp = tempfile::tempdir().unwrap();
    let db_path = tmp.path().join("vec_dedup_hnsw.db");
    let dims = 8usize; // small for test speed
    let store = SqliteStore::new(&db_path, "mock-embedder", dims).unwrap();

    let memory = make_memory("h3-hnsw-reinsertion-test", "canonical body");
    let canonical_id = store.store(memory).unwrap();
    store
        .conn()
        .execute(
            "UPDATE memories SET needs_vec_dedup = 1 WHERE id = ?1",
            rusqlite::params![&canonical_id],
        )
        .unwrap();

    // Verify HNSW empty before the sweep.
    let hnsw_base = db_path.with_extension("");
    let index_before = HnswIndex::open(&hnsw_base, dims).unwrap();
    assert_eq!(
        index_before.len(),
        0,
        "precondition: HNSW should be empty before the sweep"
    );
    // Release the lock held by the open by dropping.
    drop(index_before);

    // Run vec-dedup with a mock that returns a fixed vector. We have to
    // configure resummerize-enabled config — actually no, run_vec_dedup
    // only reads the embedding config. Defaults are fine since the
    // override bypasses the provider factory.
    let mut config = rein::config::ReinConfig::default();
    config.embedding.dimensions = dims;
    let fixed: Vec<f32> = (0..dims).map(|i| i as f32 / dims as f32).collect();
    let mock = EmbedderKind::Mock(MockEmbedder::with_fixed_vector(dims, fixed));
    run_vec_dedup_with_embedder(&store, &config, mock);

    // Successful sweep clears the flag.
    let flag: i64 = store
        .conn()
        .query_row(
            "SELECT needs_vec_dedup FROM memories WHERE id = ?1",
            rusqlite::params![&canonical_id],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(flag, 0, "successful vec_dedup must clear the flag");

    // HNSW now contains the entry. Round-5 H-3 was precisely: sweep
    // inserted into sqlite-vec but not HNSW, so the canonical was
    // silently missing from vector recall via the HNSW channel.
    let index_after = HnswIndex::open(&hnsw_base, dims).unwrap();
    assert_eq!(
        index_after.len(),
        1,
        "HNSW must contain the re-embedded canonical after a successful \
         vec_dedup sweep; round-5 H-3 regression"
    );
}

/// Codex round-6 H-1 coverage: the vec-dedup candidate gate must treat an
/// `updated` canonical as a live strong-match target, not only as a pending
/// source row. Pre-fix this exact state was skipped by the inner candidate
/// filter.
#[tokio::test(flavor = "multi_thread")]
async fn vec_dedup_accepts_updated_canonical_as_strong_match_candidate() {
    use rein::embed::{EmbedderKind, MockEmbedder};
    use rein::ops::dedup::run_vec_dedup_with_embedder;

    let store = SqliteStore::in_memory().unwrap();
    let dims = 3072usize;
    let mut config = rein::config::ReinConfig::default();
    config.embedding.dimensions = dims;

    let mut candidate = make_memory("h1-updated-candidate", "existing canonical body");
    candidate.access_count = 1;
    let candidate_id = store.store(candidate).unwrap();
    store
        .conn()
        .execute(
            "UPDATE memories SET status = 'updated' WHERE id = ?1",
            rusqlite::params![&candidate_id],
        )
        .unwrap();

    let pending_id = store
        .store(make_memory(
            "h1-updated-candidate",
            "incoming near-duplicate body",
        ))
        .unwrap();
    store
        .conn()
        .execute(
            "UPDATE memories SET needs_vec_dedup = 1 WHERE id = ?1",
            rusqlite::params![&pending_id],
        )
        .unwrap();

    let fixed = vec![0.25f32; dims];
    rein::store::vec::insert_embedding(store.conn(), &candidate_id, &fixed).unwrap();
    let mock = EmbedderKind::Mock(MockEmbedder::with_fixed_vector(dims, fixed));
    run_vec_dedup_with_embedder(&store, &config, mock);

    let pending_after = store.get(&pending_id).unwrap();
    assert_eq!(
        pending_after.superseded_by.as_deref(),
        Some(candidate_id.as_str()),
        "vec_dedup must accept an 'updated' canonical as the winner; round-6 H-1 regression"
    );
}

/// Codex round-7 MEDIUM regression guard: a failed strong-match merge must
/// preserve `needs_vec_dedup = 1` so the row can retry on the next sweep.
/// Pre-fix the savepoint rolled back, but the end-of-loop clear still fired.
#[tokio::test(flavor = "multi_thread")]
async fn vec_dedup_preserves_flag_when_strong_merge_mark_superseded_fails() {
    use rein::embed::{EmbedderKind, MockEmbedder};
    use rein::ops::dedup::run_vec_dedup_with_embedder;

    let store = SqliteStore::in_memory().unwrap();
    let dims = 3072usize;
    let mut config = rein::config::ReinConfig::default();
    config.embedding.dimensions = dims;

    let mut candidate = make_memory("m2-strong-merge-failure", "existing canonical body");
    candidate.access_count = 1;
    let candidate_id = store.store(candidate).unwrap();

    let pending_id = store
        .store(make_memory(
            "m2-strong-merge-failure",
            "incoming near-duplicate body",
        ))
        .unwrap();
    store
        .conn()
        .execute(
            "UPDATE memories SET needs_vec_dedup = 1 WHERE id = ?1",
            rusqlite::params![&pending_id],
        )
        .unwrap();

    let trigger_sql = format!(
        "CREATE TEMP TRIGGER fail_vec_dedup_supersede
         BEFORE UPDATE OF superseded_by ON memories
         WHEN OLD.id = '{pending_id}'
         BEGIN
             SELECT RAISE(FAIL, 'simulated mark_superseded failure');
         END;"
    );
    store.conn().execute_batch(&trigger_sql).unwrap();

    let fixed = vec![0.5f32; dims];
    rein::store::vec::insert_embedding(store.conn(), &candidate_id, &fixed).unwrap();
    let mock = EmbedderKind::Mock(MockEmbedder::with_fixed_vector(dims, fixed));
    run_vec_dedup_with_embedder(&store, &config, mock);

    let pending_after = store.get(&pending_id).unwrap();
    assert!(
        pending_after.superseded_by.is_none(),
        "failed strong merge must roll back the supersede write"
    );
    let flag: i64 = store
        .conn()
        .query_row(
            "SELECT needs_vec_dedup FROM memories WHERE id = ?1",
            rusqlite::params![&pending_id],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        flag, 1,
        "failed strong merge must preserve needs_vec_dedup for retry; round-7 MEDIUM regression"
    );
}

/// Codex round-5 H-1 regression guard: `claim_batch` /
/// `count_needs_resummerize` / `backlog_count` / `run_vec_dedup` all
/// previously filtered `status = 'active'` — but the merge path
/// auto-promotes `status` to `'updated'` via `store.update()`'s trigger.
/// Result pre-fix: a merge that also set `needs_resummerize = 1` produced
/// a row with `status = 'updated'` + `needs_resummerize = 1` that slow
/// channels silently never picked up. This test seeds that exact state
/// and verifies the widened `status IN ('active', 'updated')` predicate
/// now admits it.
#[test]
fn merged_status_updated_canonical_is_eligible_for_resummerize() {
    use rein::ops::resummerize::test_hooks::claim_for_test;

    let store = SqliteStore::in_memory().unwrap();
    let memory = make_memory("h1-merge-promotion-test", "canonical body");
    let canonical_id = store.store(memory).unwrap();

    // Promote status to 'updated' (simulating the merge path's trigger
    // side-effect) AND flag for resummerize.
    store
        .conn()
        .execute(
            "UPDATE memories SET status = 'updated', needs_resummerize = 1 WHERE id = ?1",
            rusqlite::params![&canonical_id],
        )
        .unwrap();

    // claim_for_test must succeed — pre-H-1 it would return None.
    let token = claim_for_test(&store, &canonical_id)
        .unwrap()
        .expect("updated-status merged canonical must be claimable; round-5 H-1 regression");
    assert!(!token.is_empty());

    // The audit-layer count helper must also include this row.
    let count = rein::store::resummerize_audit::count_needs_resummerize(store.conn()).unwrap();
    assert_eq!(
        count, 1,
        "count_needs_resummerize must include 'updated'-status rows; round-5 H-1"
    );

    // backlog_count (the doctor-facing helper) must too.
    let backlog = rein::ops::resummerize::backlog_count(&store).unwrap();
    assert_eq!(
        backlog, 1,
        "backlog_count must include 'updated'-status rows; round-5 H-1"
    );
}

/// Codex round-5 M-1 regression guard: the 5-way CAS in `apply_resummerize`
/// must compare against the exact byte sequence stored in
/// `memories.updated_at`, not a reserialized-via-chrono form. If the
/// stored value is a valid RFC3339 variant (e.g. `+00:00` offset instead
/// of chrono's `Z`), the pre-fix code parsed + reserialized and then
/// lexicographically mismatched the DB text, forever looping `ClaimLost`.
/// Post-fix threads raw DB text through.
#[test]
fn rfc3339_offset_variant_updated_at_does_not_spuriously_claim_lost() {
    use rein::ops::resummerize::test_hooks::{apply_for_test, claim_for_test, ApplyOutcome};

    let store = SqliteStore::in_memory().unwrap();
    let memory = make_memory("m1-rfc3339-variant-test", "original canonical body");
    let canonical_id = store.store(memory).unwrap();

    // Overwrite `updated_at` with a valid RFC3339 that uses `+00:00`
    // offset instead of the `Z`-suffix chrono's to_rfc3339() produces.
    // Same instant, different bytes.
    store
        .conn()
        .execute(
            "UPDATE memories
                SET updated_at = ?1,
                    needs_resummerize = 1
              WHERE id = ?2",
            rusqlite::params!["2026-01-15T10:30:00+00:00", &canonical_id],
        )
        .unwrap();

    let token = claim_for_test(&store, &canonical_id)
        .unwrap()
        .expect("eligible row should be claimable");

    // apply_for_test reads updated_at verbatim and uses it in the CAS.
    // Pre-M-1 it would have reserialized to `2026-01-15T10:30:00Z` and
    // CAS'd to ClaimLost. Post-M-1 it compares raw bytes → Applied.
    let outcome =
        apply_for_test(&store, &canonical_id, "compressed canonical body", &token).unwrap();
    assert_eq!(
        outcome,
        ApplyOutcome::Applied,
        "RFC3339 format variance must not spuriously ClaimLost; round-5 M-1 regression"
    );

    // Canonical actually rewritten.
    assert_eq!(
        store.get(&canonical_id).unwrap().content,
        "compressed canonical body"
    );
}

/// Codex post-fix audit H-1: `check_claim_still_held` must return false
/// when (a) the claim token no longer matches, (b) the snapshot
/// `updated_at` has drifted, or (c) the row has been demoted to a
/// non-live status. Ensures the length-check and contract-check paths in
/// `run_resummerize_inner` don't classify a concurrent-MergeInto race as
/// a countable `LengthExceeded` / `ContractViolation`.
#[test]
fn check_claim_still_held_detects_all_three_drift_cases() {
    use rein::ops::resummerize::test_hooks::{check_claim_still_held_for_test, claim_for_test};

    let (store, _config, canonical_id) = setup_flagged_canonical("body");
    let token = claim_for_test(&store, &canonical_id)
        .unwrap()
        .expect("fresh flagged row must be claimable");
    let snapshot_updated_at: String = store
        .conn()
        .query_row(
            "SELECT updated_at FROM memories WHERE id = ?1",
            rusqlite::params![&canonical_id],
            |r| r.get(0),
        )
        .unwrap();

    // Happy path — ownership held, snapshot matches, status live.
    assert!(
        check_claim_still_held_for_test(&store, &canonical_id, &token, &snapshot_updated_at)
            .unwrap(),
        "unchanged row must report ownership still held"
    );

    // (a) Claim token mismatch — simulate a peer worker reclaiming.
    assert!(
        !check_claim_still_held_for_test(
            &store,
            &canonical_id,
            "some-other-token",
            &snapshot_updated_at
        )
        .unwrap(),
        "token mismatch must report ownership lost"
    );

    // (b) Snapshot updated_at mismatch — simulate a concurrent MergeInto
    // bumping the row's updated_at while our worker was mid-LLM-call.
    store
        .conn()
        .execute(
            "UPDATE memories SET updated_at = ?1 WHERE id = ?2",
            rusqlite::params!["2027-01-01T00:00:00Z", &canonical_id],
        )
        .unwrap();
    assert!(
        !check_claim_still_held_for_test(&store, &canonical_id, &token, &snapshot_updated_at)
            .unwrap(),
        "updated_at drift must report ownership lost"
    );

    // (c) Status demoted to non-live — simulate `cleanup` marking the row
    // deprecated while we held the claim. Reset updated_at first so (b)
    // isn't also tripping the check.
    store
        .conn()
        .execute(
            "UPDATE memories SET updated_at = ?1, status = 'deprecated' WHERE id = ?2",
            rusqlite::params![&snapshot_updated_at, &canonical_id],
        )
        .unwrap();
    assert!(
        !check_claim_still_held_for_test(&store, &canonical_id, &token, &snapshot_updated_at)
            .unwrap(),
        "non-live status must report ownership lost"
    );
}

/// Codex round-2 HIGH regression guard: if a stale worker returns from
/// its LLM call AFTER another worker has already reclaimed and rewritten
/// the row, the late worker's `apply_resummerize` MUST NOT overwrite
/// the newer output. The claim token carried end-to-end + CAS clause in
/// the UPDATE's WHERE turns this into a hard ROLLBACK path.
#[test]
fn apply_with_stale_claim_token_rolls_back_to_claim_lost() {
    use rein::ops::resummerize::test_hooks::{apply_for_test, claim_for_test, ApplyOutcome};

    let (store, _config, canonical_id) =
        setup_flagged_canonical("original canonical body that must survive a stolen claim");

    // Worker A claims the row (simulates the first part of
    // `run_resummerize_inner` executing).
    let token_a = claim_for_test(&store, &canonical_id)
        .unwrap()
        .expect("freshly-flagged row should be eligible for claim");

    // Between A's LLM call and A's commit, Worker B's stale-timeout
    // reclaim path reassigns the row. We simulate this with a direct
    // UPDATE that stamps a different token onto `in_progress_resummerize_at`.
    store
        .conn()
        .execute(
            "UPDATE memories SET in_progress_resummerize_at = ?1 WHERE id = ?2",
            rusqlite::params!["simulated-worker-b-token", &canonical_id],
        )
        .unwrap();

    // Baseline captured BEFORE A's doomed commit attempt.
    let content_before = store.get(&canonical_id).unwrap().content.clone();
    let evidence_before = store
        .list_memory_evidence(&canonical_id, 100)
        .unwrap()
        .len();

    // A returns from its LLM call and tries to commit with its (now
    // stale) token. The CAS clause in the UPDATE fails to match any
    // row, apply_resummerize rolls back the snapshot, and reports
    // ClaimLost — no canonical mutation, no phantom evidence row.
    let verdict = apply_for_test(
        &store,
        &canonical_id,
        "compressed output that would be perfectly safe",
        &token_a,
    )
    .unwrap();

    assert_eq!(verdict, ApplyOutcome::ClaimLost);
    assert_eq!(
        store.get(&canonical_id).unwrap().content,
        content_before,
        "stale worker overwrote canonical despite claim mismatch"
    );
    assert_eq!(
        store
            .list_memory_evidence(&canonical_id, 100)
            .unwrap()
            .len(),
        evidence_before,
        "stale worker left a phantom pre-snapshot despite ROLLBACK"
    );
    // And B's claim marker is still in place — A's release_claim also
    // narrows to its own token, so it shouldn't have clobbered B.
    assert_eq!(
        canonical_in_progress(&store, &canonical_id).as_deref(),
        Some("simulated-worker-b-token"),
        "stale worker nulled the new claim marker"
    );
}

/// H6 concurrent-claim test. Verifies that two sequential invocations of
/// the run path can't both process the same flagged canonical, and that
/// a stale claim (>5 min old) can be reclaimed.
///
/// Exercised indirectly: we drive the state transitions via the public
/// surface (`run_resummerize` dry-run preview + direct flag flips) and
/// read the `in_progress_resummerize_at` column to verify claim lifecycle.
#[test]
fn in_progress_claim_blocks_second_worker_and_expires_after_stale_timeout() {
    let store = SqliteStore::in_memory().unwrap();

    let canonical = make_memory("h6-claim-test", "seed content that will be resummerized");
    let canonical_id = store.store(canonical).unwrap();
    store
        .conn()
        .execute(
            "UPDATE memories SET needs_resummerize = 1 WHERE id = ?1",
            rusqlite::params![&canonical_id],
        )
        .unwrap();

    // Simulate a worker claim by directly setting in_progress_resummerize_at
    // to a fresh timestamp. This is the exact SQL the claim_batch UPDATE
    // issues atomically; we write it manually here because the in-process
    // run path either fully succeeds or fully releases.
    let now = chrono::Utc::now().to_rfc3339();
    store
        .conn()
        .execute(
            "UPDATE memories SET in_progress_resummerize_at = ?1 WHERE id = ?2",
            rusqlite::params![&now, &canonical_id],
        )
        .unwrap();

    // dry_run preview must NOT include this row — it's claimed.
    // preview_eligible is module-private; the observable proxy is
    // run_resummerize(dry_run=true) which reports `attempted = 0` because
    // the sole flagged row is claimed.
    let mut config = ReinConfig::default();
    config.resummerize.enabled = true;
    let outcome = rein::ops::resummerize::run_resummerize(&store, &config, None, true).unwrap();
    assert_eq!(
        outcome.attempted, 0,
        "fresh claim must hide the row from the dry-run preview"
    );

    // Force the claim to look stale by rewinding >5 min into the past.
    let stale_ts = (chrono::Utc::now() - chrono::Duration::seconds(301)).to_rfc3339();
    store
        .conn()
        .execute(
            "UPDATE memories SET in_progress_resummerize_at = ?1 WHERE id = ?2",
            rusqlite::params![&stale_ts, &canonical_id],
        )
        .unwrap();

    let outcome = rein::ops::resummerize::run_resummerize(&store, &config, None, true).unwrap();
    assert_eq!(
        outcome.attempted, 1,
        "stale claim (>5 min old) must allow reclaim by a subsequent worker"
    );

    // And backlog_count includes claimed rows regardless of stale status —
    // doctor reports "work outstanding", not "ready right now".
    let backlog = rein::ops::resummerize::backlog_count(&store).unwrap();
    assert_eq!(backlog, 1);
}
