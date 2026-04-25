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

use crate::types::traits::Embedder;

use super::{block_on_future, cosine_similarity, tokenize, KeywordOverlapHitChecker};

/// Return `true` if a majority of `evidence_keywords` appear (tokenized,
/// lowercase, CJK-aware, **stemmed** under v3) in
/// `definition + " " + living_summary.unwrap_or("")`.
///
/// Majority semantics: `hits * 2 > evidence_keywords.len()` — strict majority,
/// so 3 keywords need 2 hits, 4 keywords need 3, 5 keywords need 3. An empty
/// `evidence_keywords` list trivially scores as a miss (no keywords → no
/// evidence that the summary is informative).
///
/// ## v3 hybrid scoring (HIT_CHECKER_VERSION = 3)
///
/// 1. **Stem fast path**: every keyword's tokenized + stemmed sub-tokens
///    are required to appear in the candidate token set. This collapses
///    morphological variants (`extract` ≈ `extracting` ≈ `extraction`) that
///    v2 missed.
/// 2. **Semantic fallback**: for keywords that fail the stem pass, if
///    `checker.semantic` is `Some`, the keyword's embedding is compared
///    against the candidate text's embedding. If cosine ≥ threshold, the
///    keyword is credited as a semantic hit. This catches synonymy
///    (`Ebbinghaus` ≈ `decay`, `STM` ≈ `short-term memory`) that no
///    purely lexical check can bridge.
///
/// Stem hits take precedence — the embedder is only invoked for the
/// remainder set, so a corpus where every keyword stems out incurs zero
/// API calls. Embedder failures (network, quota) are logged via
/// `tracing::warn!` and treated as "no semantic hit"; production scoring
/// degrades gracefully to the stem-only floor in that case rather than
/// poisoning the scorecard with hard errors.
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
    // `tokenize` lowercases Latin input, routes CJK through jieba, and
    // (under v3) stems every ASCII token via Porter2 — so the keyword
    // check naturally handles mixed-script keywords plus morphological
    // variants in one pass.
    let tokens: HashSet<String> = tokenize(&combined).into_iter().collect();

    let mut hits = 0usize;
    let mut unmatched: Vec<&String> = Vec::new();
    for kw in evidence_keywords {
        // A keyword may itself be multi-token (e.g. "operation registry") —
        // tokenize it the same way and require every sub-token to appear.
        let sub_tokens = tokenize(kw);
        if sub_tokens.is_empty() {
            continue;
        }
        if sub_tokens.iter().all(|t| tokens.contains(t)) {
            hits += 1;
        } else {
            unmatched.push(kw);
        }
    }

    if !unmatched.is_empty() {
        if let Some(sem) = checker.semantic.as_ref() {
            hits += semantic_hits(&combined, &unmatched, sem);
        }
    }

    hits * 2 > evidence_keywords.len()
}

/// Count how many of `unmatched` keywords pass the semantic threshold.
///
/// Strategy: one batched embedding call covering `[combined, kw_1, kw_2, ...]`
/// so the network round-trip cost is amortized across the per-case batch.
/// On any embedder failure the function returns 0 (graceful degradation —
/// stem-only floor stands).
fn semantic_hits(combined: &str, unmatched: &[&String], sem: &super::SemanticFallback) -> usize {
    if unmatched.is_empty() {
        return 0;
    }
    let mut texts: Vec<&str> = Vec::with_capacity(unmatched.len() + 1);
    texts.push(combined);
    for kw in unmatched {
        texts.push(kw.as_str());
    }
    let embed_result = block_on_future(async move { sem.embedder.embed_batch(&texts).await });
    let embeddings = match embed_result {
        Ok(v) if v.len() == unmatched.len() + 1 => v,
        Ok(v) => {
            tracing::warn!(
                expected = unmatched.len() + 1,
                got = v.len(),
                "semantic fallback: embed_batch returned wrong arity, treating all keywords as misses"
            );
            return 0;
        }
        Err(e) => {
            tracing::warn!(error = %e, "semantic fallback: embedder failed, falling back to stem-only floor");
            return 0;
        }
    };
    let text_emb = &embeddings[0];
    let mut hits = 0usize;
    for kw_emb in embeddings.iter().skip(1) {
        let sim = cosine_similarity(text_emb, kw_emb);
        if sim >= sem.similarity_threshold {
            hits += 1;
        }
    }
    hits
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
        assert!(score_concept_case(def, ls, &kws, &KeywordOverlapHitChecker::stem_only()));
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
            &KeywordOverlapHitChecker::stem_only()
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
            &KeywordOverlapHitChecker::stem_only(),
        ));
        let ls = Some("CJK content routes through jieba with bigram fallback.");
        // With living_summary, 2/3 keywords hit ("token", "jieba") → majority.
        assert!(score_concept_case(def, ls, &kws, &KeywordOverlapHitChecker::stem_only()));
    }

    #[test]
    fn empty_keywords_miss() {
        assert!(!score_concept_case(
            "anything",
            Some("something"),
            &[],
            &KeywordOverlapHitChecker::stem_only()
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
        assert!(score_concept_case(def, ls, &kws, &KeywordOverlapHitChecker::stem_only()));
    }
}
