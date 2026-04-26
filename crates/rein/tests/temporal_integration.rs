//! Integration tests for `extract::temporal` (v0.27.0 Track 2 #8 — Agent D).
//!
//! These exercise the public dispatcher [`extract_temporal`] end-to-end with
//! `MockExtractor::Sequence` so the same surface that the dedup pipeline
//! (Agent E) will call is tested without a live LLM provider.
//!
//! Gated on `feature = "test-support"` because `MockExtractor` lives behind
//! that flag and is absent from the release binary.

#![cfg(feature = "test-support")]

use chrono::{DateTime, NaiveDate, TimeZone, Utc};
use rein::extract::temporal::{
    extract_temporal, extract_temporal_llm, extract_temporal_rule_based, AnchorKind,
};
use rein::extract::{ExtractorKind, MockExtractor};

fn dt(y: i32, mo: u32, d: u32) -> DateTime<Utc> {
    let nd = NaiveDate::from_ymd_opt(y, mo, d).expect("valid date");
    Utc.from_utc_datetime(&nd.and_hms_opt(0, 0, 0).expect("midnight"))
}

fn fixed_now() -> DateTime<Utc> {
    // 2026-04-26 12:00:00 UTC — Sunday, ISO week 17.
    Utc.with_ymd_and_hms(2026, 4, 26, 12, 0, 0).unwrap()
}

#[tokio::test]
async fn dispatcher_uses_llm_when_extractor_returns_anchors() {
    // Mock returns one anchor — dispatcher should prefer LLM output and skip
    // the rule-based path entirely.
    let mock = MockExtractor::with_fixed_response(
        r#"[{"kind":"absolute","start_iso":"2024-01-01T00:00:00Z","end_iso":"2025-01-01T00:00:00Z","raw_phrase":"2024","confidence":0.95}]"#,
    );
    let extractor = ExtractorKind::Mock(mock);
    let result = extract_temporal(Some(&extractor), "in 2024 we shipped", fixed_now())
        .await
        .expect("dispatcher returns Ok");
    assert_eq!(result.len(), 1, "LLM path should produce one anchor");
    assert_eq!(result[0].kind, AnchorKind::Absolute);
    assert_eq!(result[0].start, Some(dt(2024, 1, 1)));
}

#[tokio::test]
async fn dispatcher_falls_back_to_rules_when_llm_returns_empty_array() {
    let mock = MockExtractor::with_fixed_response("[]");
    let extractor = ExtractorKind::Mock(mock);
    // Content the rule-based path can match.
    let result = extract_temporal(Some(&extractor), "yesterday I shipped", fixed_now())
        .await
        .expect("dispatcher returns Ok");
    assert_eq!(
        result.len(),
        1,
        "rule-based fallback should produce one anchor for 'yesterday'"
    );
    assert_eq!(result[0].kind, AnchorKind::Relative);
    assert_eq!(result[0].start, Some(dt(2026, 4, 25)));
}

#[tokio::test]
async fn dispatcher_falls_back_to_rules_when_llm_errors() {
    // Persistent error — the LLM call surfaces a Config error which
    // extract_temporal_llm catches and returns Ok(empty), so the
    // dispatcher should still try the rule-based path.
    let mock = MockExtractor::with_persistent_error("provider outage");
    let extractor = ExtractorKind::Mock(mock);
    let result = extract_temporal(Some(&extractor), "since 2026", fixed_now())
        .await
        .expect("dispatcher must always return Ok");
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].kind, AnchorKind::OpenEnd);
    assert_eq!(result[0].start, Some(dt(2026, 1, 1)));
}

#[tokio::test]
async fn dispatcher_no_extractor_uses_rules_only() {
    let result = extract_temporal(None, "2026-04-26 we cut v0.27", fixed_now())
        .await
        .expect("dispatcher must return Ok");
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].kind, AnchorKind::Absolute);
    assert_eq!(result[0].start, Some(dt(2026, 4, 26)));
}

#[tokio::test]
async fn bilingual_content_extracts_both_languages_via_rules() {
    // The dispatcher with no extractor goes straight to rule-based, which
    // must handle EN and CJK markers in the same content.
    let content = "yesterday I tested 上周 we deployed";
    let result = extract_temporal_rule_based(content, fixed_now());
    let kinds: Vec<_> = result.iter().map(|a| a.kind).collect();
    assert_eq!(kinds.iter().filter(|k| **k == AnchorKind::Relative).count(), 2);
    // One should be the prior day; one should be the prior ISO week.
    let starts: Vec<_> = result.iter().filter_map(|a| a.start).collect();
    assert!(
        starts.contains(&dt(2026, 4, 25)),
        "yesterday should produce 2026-04-25 start"
    );
    assert!(
        starts.contains(&dt(2026, 4, 13)),
        "上周 should produce 2026-04-13 start"
    );
}

#[tokio::test]
async fn llm_path_resolves_relative_phrase_via_now_in_prompt() {
    // The mock returns whatever we script, but verifies that
    // extract_temporal_llm is async-callable and parses correctly.
    let mock = MockExtractor::with_fixed_response(
        r#"[{"kind":"relative","start_iso":"2026-04-25T00:00:00Z","end_iso":"2026-04-26T00:00:00Z","raw_phrase":"yesterday","confidence":0.85}]"#,
    );
    let extractor = ExtractorKind::Mock(mock);
    let result = extract_temporal_llm(&extractor, "yesterday", fixed_now())
        .await
        .expect("happy path returns Ok");
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].kind, AnchorKind::Relative);
    assert_eq!(result[0].start, Some(dt(2026, 4, 25)));
    assert_eq!(result[0].end, Some(dt(2026, 4, 26)));
}

#[tokio::test]
async fn llm_path_malformed_json_degrades_gracefully() {
    let mock = MockExtractor::with_fixed_response("not even close to JSON");
    let extractor = ExtractorKind::Mock(mock);
    let result = extract_temporal_llm(&extractor, "yesterday", fixed_now())
        .await
        .expect("malformed JSON must NOT propagate as Err");
    assert!(result.is_empty(), "malformed JSON yields empty Vec");
}
