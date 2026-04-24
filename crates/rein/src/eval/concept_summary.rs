//! Scoring helper for v0.24 ARS Capability A (concept living summary) eval.
//!
//! Parallel to the v0.23 resummerize scoring path: evidence-keyword overlap
//! against `definition` (baseline) vs `definition + " " + living_summary`
//! (treatment), with CJK-aware tokenization inherited from
//! [`crate::eval::tokenize`].
//!
//! The asymmetry in the scored text — baseline sees only `definition`, treatment
//! sees `definition + " " + living_summary` — is deliberate: it mirrors how a
//! downstream consumer reads a concept card (definition first, living summary
//! appended when present). The `rein-eval concept-summary` binary records
//! `baseline_length = definition.len()` and
//! `treatment_length = definition.len() + 1 + living_summary.len()` to match
//! exactly what was scored.

use std::collections::HashSet;

use super::{tokenize, KeywordOverlapHitChecker};

/// Return `true` if a majority of `evidence_keywords` appear (tokenized,
/// lowercase, CJK-aware) in `definition + " " + living_summary.unwrap_or("")`.
///
/// Majority semantics: `hits * 2 > evidence_keywords.len()` — strict majority,
/// so 3 keywords need 2 hits, 4 keywords need 3, 5 keywords need 3. An empty
/// `evidence_keywords` list trivially scores as a miss (no keywords → no
/// evidence that the summary is informative).
///
/// The `checker` parameter is accepted so future callers can swap in a
/// different `HitChecker` without changing the signature; the Phase 1 body
/// only needs the shared `tokenize` routine, so the argument is currently
/// unused beyond signaling "this is the scorer that produced the outcome".
/// Marked `#[allow(unused_variables)]` until a richer checker gets a
/// per-keyword semantic pass.
#[allow(unused_variables)]
pub fn score_concept_case(
    definition: &str,
    living_summary: Option<&str>,
    evidence_keywords: &[String],
    checker: &KeywordOverlapHitChecker,
) -> bool {
    if evidence_keywords.is_empty() {
        return false;
    }
    let mut combined =
        String::with_capacity(definition.len() + 1 + living_summary.map(|s| s.len()).unwrap_or(0));
    combined.push_str(definition);
    if let Some(ls) = living_summary {
        combined.push(' ');
        combined.push_str(ls);
    }

    // Build the candidate-text token set once and probe each keyword.
    // `tokenize` lowercases Latin input and routes CJK through jieba, so
    // the keyword check naturally handles mixed-script keywords (e.g. the
    // v0.22 kickoff fixture's "D5 bench" alongside "压倒性推翻").
    let tokens: HashSet<String> = tokenize(&combined).into_iter().collect();

    let mut hits = 0usize;
    for kw in evidence_keywords {
        // A keyword may itself be multi-token (e.g. "operation registry") —
        // tokenize it the same way and require every sub-token to appear.
        let sub_tokens = tokenize(kw);
        if sub_tokens.is_empty() {
            continue;
        }
        if sub_tokens.iter().all(|t| tokens.contains(t)) {
            hits += 1;
        }
    }

    hits * 2 > evidence_keywords.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn majority_over_three_keywords_hits() {
        // 2 of 3 keywords present → strict majority, hit.
        let def = "A1 operation registry centralizes op dispatch.";
        let ls = Some("Single-seg path templates and inventory-based routing.");
        let kws = vec![
            "operation registry".to_string(),
            "inventory".to_string(),
            "tensor cores".to_string(),
        ];
        assert!(score_concept_case(def, ls, &kws, &KeywordOverlapHitChecker));
    }

    #[test]
    fn minority_keywords_miss() {
        let def = "A1 operation registry centralizes op dispatch.";
        let ls: Option<&str> = None;
        let kws = vec![
            "nonexistent".to_string(),
            "tensor".to_string(),
            "tensor cores".to_string(),
        ];
        assert!(!score_concept_case(
            def,
            ls,
            &kws,
            &KeywordOverlapHitChecker
        ));
    }

    #[test]
    fn living_summary_rescues_missing_evidence() {
        // The definition alone misses the "jieba" keyword; the living
        // summary recovers it — simulating ARS Capability A's expected win.
        let def = "Hybrid lexical dedup using token shingles.";
        let kws = vec![
            "token".to_string(),
            "jieba".to_string(),
            "ngram".to_string(),
        ];
        assert!(!score_concept_case(
            def,
            None,
            &kws,
            &KeywordOverlapHitChecker,
        ));
        let ls = Some("CJK content routes through jieba with bigram fallback.");
        // With living_summary, 2/3 keywords hit ("token", "jieba") → majority.
        assert!(score_concept_case(def, ls, &kws, &KeywordOverlapHitChecker));
    }

    #[test]
    fn empty_keywords_miss() {
        assert!(!score_concept_case(
            "anything",
            Some("something"),
            &[],
            &KeywordOverlapHitChecker
        ));
    }

    #[test]
    fn cjk_keywords_tokenize_via_jieba() {
        // Chinese keywords must be matched after both candidate text and
        // keyword pass through the same jieba-segmenting tokenizer. Pick
        // keywords that jieba emits as a single token (avoiding words that
        // jieba splits differently from their in-context occurrence).
        let def = "A1 操作 注册表 集中 调度 所有";
        let ls = Some("inventory 路由");
        let kws = vec![
            "操作".to_string(),
            "inventory".to_string(),
            "不相关".to_string(),
        ];
        // Hits: 操作 (def), inventory (ls). Miss: 不相关 (neither).
        // 2 hits of 3 → strict majority → pass.
        assert!(score_concept_case(def, ls, &kws, &KeywordOverlapHitChecker));
    }
}
