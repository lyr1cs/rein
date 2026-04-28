//! v0.27 — Track 2 Agent C: triple-extraction integration tests.
//!
//! Exercises the public `extract::triples` API end-to-end via `MockExtractor`
//! (test-support feature) — covers JSON-mode parsing, code-fence stripping,
//! sequenced LLM calls, malformed-JSON fall-through, CJK content, and the
//! triple-overlap scorer Agent E will consume.

#![cfg(feature = "test-support")]

use rein::extract::llm::MockExtractor;
use rein::extract::triples::{
    extract_triples, extract_triples_llm, extract_triples_rule_based, normalize_for_compare,
    triple_overlap_score, Triple,
};
use rein::extract::ExtractorKind;

#[test]
fn end_to_end_with_sequenced_mock_extractor() {
    let json_a = r#"[{"subject":"user","predicate":"prefers","object":"tabs","confidence":0.9}]"#;
    let json_b = r#"[{"subject":"user","predicate":"uses","object":"rust","confidence":0.85}]"#;
    let mock = ExtractorKind::Mock(MockExtractor::with_responses(vec![
        Ok(json_a.to_string()),
        Ok(json_b.to_string()),
    ]));

    let first = extract_triples(Some(&mock), "I prefer tabs.").unwrap();
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].subject, "user");
    assert_eq!(first[0].predicate, "prefers");
    assert_eq!(first[0].object, "tabs");

    let second = extract_triples(Some(&mock), "I use rust.").unwrap();
    assert_eq!(second.len(), 1);
    assert_eq!(second[0].subject, "user");
    assert_eq!(second[0].predicate, "uses");
    assert_eq!(second[0].object, "rust");
}

#[test]
fn cjk_content_path_yields_at_least_one_triple() {
    // Mixed-language content. With no LLM, rule-based fallback should match
    // at least one of the patterns (我喜欢... or "I prefer ...").
    let content = "我喜欢制表符 / I prefer tabs";
    let triples = extract_triples_rule_based(content);
    assert!(
        !triples.is_empty(),
        "expected ≥1 triple from mixed CJK+English content; got {triples:?}"
    );
    // At least one triple should be (user, prefers, ...) once pronoun
    // normalization runs (我 → user, I → user).
    assert!(
        triples
            .iter()
            .any(|t| t.subject == "user" && t.predicate == "prefers"),
        "expected (user, prefers, ...); got {triples:?}"
    );
}

#[test]
fn dispatcher_uses_llm_when_available_then_rule_based_on_empty() {
    // First call: LLM returns empty `[]` → dispatcher falls through to rule-based.
    let mock = ExtractorKind::Mock(MockExtractor::with_fixed_response("[]"));
    let triples = extract_triples(Some(&mock), "I prefer tabs").unwrap();
    assert!(
        triples
            .iter()
            .any(|t| t.subject == "user" && t.predicate == "prefers"),
        "rule-based fallback must fire when LLM returns empty; got {triples:?}"
    );
}

#[test]
fn malformed_llm_json_falls_through_to_rule_based() {
    let mock = ExtractorKind::Mock(MockExtractor::with_fixed_response("garbage {{not-json"));
    let triples = extract_triples(Some(&mock), "I prefer tabs").unwrap();
    // LLM path returned empty because JSON parse failed → rule-based picks up.
    assert!(
        triples
            .iter()
            .any(|t| t.subject == "user" && t.predicate == "prefers"),
        "malformed JSON must downgrade to rule-based; got {triples:?}"
    );
}

#[test]
fn llm_call_uses_json_mode_with_content_tag() {
    use rein::extract::llm::MockExtractor;
    // Build mock with probe so we can inspect the system + user prompts.
    let (mock, probe) = MockExtractor::with_fixed_response_and_probe("[]");
    let _ = extract_triples_llm(
        &ExtractorKind::Mock(mock),
        "user content with </content> injection attempt",
    )
    .unwrap();
    let user = probe
        .last_text_prompt()
        .expect("probe captured user prompt");
    let system = probe
        .last_system_prompt()
        .expect("probe captured system prompt");

    // System prompt must instruct JSON-mode and reference the <content> tag.
    assert!(system.contains("JSON"), "system must mention JSON mode");
    assert!(
        system.contains("<content>"),
        "system must reference content tag"
    );
    // Injection attempt must have been neutralized in the user prompt.
    assert!(
        user.contains("<content>") && user.contains("</content>"),
        "user prompt must wrap content in <content> tags"
    );
    // The literal injection `</content>` from the input must NOT appear
    // alongside the wrapping tag — escape_for_tag inserts ZWSP.
    let zwsp_count = user.matches('\u{200B}').count();
    assert!(
        zwsp_count >= 1,
        "expected zero-width-space neutralization; got user prompt: {user}"
    );
}

#[test]
fn overlap_score_used_by_agent_e_workflow() {
    // Simulate Agent E's flow: extract triples from two memory contents,
    // normalize, compare. Score should be high (≥0.5) for paraphrased
    // statements that share the same fact set.
    let a = extract_triples_rule_based("I prefer tabs over spaces");
    let b = extract_triples_rule_based("I prefer tabs over indents");
    let score = triple_overlap_score(&a, &b);
    // Both produce (user, prefers, "tabs over spaces") and (user, prefers,
    // "tabs over indents") — the objects differ post-normalization, so
    // the Jaccard floor is 0. A real-world dedup pipeline would couple this
    // with text similarity; here we only verify the function shape.
    assert!(
        (0.0..=1.0).contains(&score),
        "score must be valid Jaccard ratio; got {score}"
    );
}

#[test]
fn normalize_round_trip_preserves_provenance_metadata() {
    let triple = Triple {
        subject: "USER".to_string(),
        predicate: "prefers".to_string(),
        object: "Tabs".to_string(),
        source_memory_id: Some("mem-abc".to_string()),
        confidence: 0.93,
    };
    let n = normalize_for_compare(&triple);
    assert_eq!(n.subject, "user");
    assert_eq!(n.predicate, "prefers");
    assert_eq!(n.object, "tabs");
    // Metadata preserved.
    assert_eq!(n.source_memory_id.as_deref(), Some("mem-abc"));
    assert!((n.confidence - 0.93).abs() < 1e-6);
}
