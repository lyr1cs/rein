//! Evaluation harness for the v0.23 resummerize feature.
//!
//! This module provides the paired-comparison machinery used by the standalone
//! `rein-eval` binary to decide whether the resummerize treatment can ship:
//!
//! - [`mcnemar`] — paired non-inferiority test (McNemar) with chi-squared and
//!   exact-binomial paths and a Wald CI.
//! - [`scorecard`] — `Scorecard` I/O shape, per-category aggregates, and the
//!   [`scorecard::decide_ship`] policy that consumes a McNemar result.
//! - [`HitChecker`] trait — pluggable "did the evidence answer the question?"
//!   oracle. The default [`KeywordOverlapHitChecker`] is a deliberately simple
//!   Phase 1 stub (see trait docs).
//!
//! Real execution (calling into `rein` to produce keep-tail baseline contexts
//! and resummerize treatment contexts, then feeding them through the
//! `HitChecker`) is the main thread's responsibility. This module stops at
//! defining the shapes and the decision math.

pub mod ablation;
pub mod concept_summary;
pub mod gates;
pub mod llm_judge;
pub mod mcnemar;
pub mod scorecard;

pub use llm_judge::{JudgeMode, JudgeOutcome, LlmJudgeHitChecker, LLM_JUDGE_VERSION};
pub use mcnemar::{mcnemar, McNemarResult, PairedOutcome};
pub use scorecard::{
    decide_ship, CategoryStats, DecideShipKind, Scorecard, ShipDecision, ShipReason,
};

/// Oracle that decides whether a surfaced evidence/context answers the
/// canonical expected content for a case.
///
/// ## Phase 1
///
/// The default implementation ([`KeywordOverlapHitChecker`]) uses simple
/// frequency-based keyword extraction from the evidence entry and checks that
/// at least 3 of the top-5 keywords also appear in the canonical. This is a
/// coarse heuristic useful for bootstrapping the harness — future work should
/// replace it with a semantic, QA-style check (e.g. LLM-judged answer
/// containment). Main thread may swap in a better impl without touching this
/// crate's eval module.
pub trait HitChecker: Send + Sync {
    /// Return `true` if `evidence_entry_content` is a hit for the case whose
    /// canonical expected text is `canonical`.
    fn check_hit(&self, evidence_entry_content: &str, canonical: &str) -> bool;
}

/// Minimal stop-word list for the default keyword-overlap hit checker.
const STOP_WORDS: &[&str] = &[
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "of", "to", "for", "in", "on",
    "and", "or", "but", "not",
];

/// CJK function words / filler that jieba's `cut` emits as distinct tokens
/// but which carry no discriminating signal for keyword overlap. Without
/// this filter, top-5 frequency selection on Chinese input gets dominated
/// by 这个 / 可以 / 那么 / 就是 and similar — the Chinese equivalents of
/// STOP_WORDS above. Post-fix audit L-3. Covers common Mandarin + a few
/// cross-CJK picks; extensible via corpus-frequency filtering later.
const CJK_STOP_WORDS: &[&str] = &[
    // Simplified Chinese multi-char function words (the Latin `chars()
    // .count() > 1` filter already drops single-char tokens, so
    // single-char entries like 的/是/在 listed in the pre-fix version
    // were redundant — audit round-2 LOW #10 removed them).
    "这个",
    "那个",
    "可以",
    "那么",
    "就是",
    "一个",
    "一些",
    "没有",
    "但是",
    "而且",
    "所以",
    "因为",
    "如果",
    "虽然",
    "然后",
    "之后",
    "之前",
    "可能",
    "我们",
    "你们",
    "他们",
    "她们",
    "它们",
    "这样",
    "那样",
    "这里",
    "那里",
    "什么",
    "怎么",
    "为什么",
    "哪里",
    "哪个",
    // Traditional Chinese variants (post-fix audit round-2 LOW #10 gap —
    // missing these let 這個 / 那個 / 沒有 dominate top-5 on Traditional
    // fixtures). Round-3 audit Finding 8 added `什麼` and `為什麼` —
    // the Taiwan-standard orthography for "what" / "why" that the
    // round-2 pass missed (it only had the 甚 variants which are more
    // common in Hong Kong).
    "這個",
    "那個",
    "可以",
    "沒有",
    "但是",
    "而且",
    "所以",
    "因為",
    "如果",
    "雖然",
    "然後",
    "之後",
    "之前",
    "我們",
    "你們",
    "他們",
    "她們",
    "它們",
    "這樣",
    "那樣",
    "這裡",
    "那裡",
    "甚麼",
    "怎麼",
    "為甚麼",
    "什麼",
    "為什麼",
    "哪裡",
    "哪個",
    "一個",
    "一些",
    // Japanese particles + common filler (multi-char; single-char
    // particles are already filtered by the length > 1 predicate)
    "です",
    "ます",
    "この",
    "その",
    "あの",
    "これ",
    "それ",
    "あれ",
    "する",
    "した",
    "している",
    "ように",
    "ための",
    "ことが",
    "ことは",
    // Korean particles + common filler (same story — single-char ones
    // drop out via the length filter)
    "그것",
    "이것",
    "저것",
    "합니다",
    "있습니다",
    "없습니다",
    "그리고",
    "하지만",
    "때문에",
    "그래서",
];

/// Hit-checker version. Bumped whenever `tokenize` or the overlap predicate
/// changes meaning. v1 was Latin-only `is_alphanumeric` split (every CJK
/// character was treated as alphanumeric, so a whole sentence tokenized as
/// a single token — making the eval's `cjk` category numbers noise).
/// v2 routes CJK content through `extract::dedup::tokenize_for_search`
/// (jieba + bigrams), making the scorer fair to CJK fixtures. v3 (v0.25.2)
/// stems Latin tokens via Snowball Porter2 so morphological variants
/// (extract / extracting / extraction) collapse to the same form, AND adds
/// an optional semantic fallback ([`SemanticFallback`]) where keywords
/// that miss the stemmed token-set are checked against an embedding-cosine
/// similarity threshold. Scorecards produced under different versions
/// must NOT be compared via `compare` without acknowledging the
/// methodology change.
pub const HIT_CHECKER_VERSION: u32 = 3;

/// Stem an ASCII alphabetic token via Snowball Porter2.
///
/// Returns the input unchanged for any token that contains a non-ASCII
/// character (CJK), a digit, or punctuation — preserving the existing
/// CJK pathway via jieba and avoiding accidental morphological collapse
/// on numeric or mixed-script tokens like "v0.25" or "ARS".
///
/// `feedback_rust_cjk_alphanumeric_trap`: the all-ASCII-alpha guard is the
/// reason CJK tokens never reach the stemmer — Rust's `char::is_alphabetic`
/// returns `true` for CJK, so the right gate is `is_ascii_alphabetic`.
pub(crate) fn stem_if_latin(token: &str) -> String {
    if !token.is_empty() && token.chars().all(|c| c.is_ascii_alphabetic()) {
        // Stemmer construction is cheap (it only borrows static rule
        // tables); the per-call allocation cost is dominated by the
        // returned String.
        rust_stemmers::Stemmer::create(rust_stemmers::Algorithm::English)
            .stem(token)
            .into_owned()
    } else {
        token.to_string()
    }
}

/// Tokenize text for keyword overlap. Routes CJK-containing input through
/// `jieba` segmentation so the eval scorer doesn't collapse CJK sentences
/// into single mega-tokens. Returned Vec preserves duplicates so the
/// frequency-counting in `KeywordOverlapHitChecker::check_hit` works the
/// same on both paths. Latin/ASCII input keeps the simple `is_alphanumeric`
/// split for backward-comparable v1 behavior on the existing fixture set's
/// non-CJK cases.
pub(crate) fn tokenize(s: &str) -> Vec<String> {
    if crate::extract::dedup::contains_cjk(s) {
        // CJK path — call jieba directly so duplicates are preserved (the
        // production `tokenize_for_search` returns a sorted dedup'd Vec via
        // an internal HashSet, which would collapse "用户" appearing 5×
        // into a single token and break the top-5 frequency selection).
        // Filter pure-punctuation, single-char, and stop-word tokens for
        // parity with the Latin path's `len > 2 && !stop_word` filter.
        // Also filter CJK function words via CJK_STOP_WORDS (post-fix
        // audit L-3) so the top-5 aren't dominated by 这个/可以/那么.
        return crate::extract::dedup::jieba()
            .cut(s, true)
            .into_iter()
            .map(|t| t.to_lowercase().trim().to_string())
            .filter(|t| {
                !t.is_empty()
                    && t.chars().count() > 1
                    && !STOP_WORDS.contains(&t.as_str())
                    && !CJK_STOP_WORDS.contains(&t.as_str())
                    && t.chars().any(|c| c.is_alphanumeric())
            })
            // v3: stem ASCII subtokens emitted by jieba (e.g. "extract"
            // alongside "用户"). CJK tokens go through unchanged because
            // `stem_if_latin` only fires when every char is ASCII alpha.
            .map(|t| stem_if_latin(&t))
            .collect();
    }
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty() && w.len() > 2 && !STOP_WORDS.contains(w))
        // v3: stem to collapse morphological variants. Tokens with digits
        // (e.g. "v0", "0252") survive untouched because `stem_if_latin`
        // gates on ascii-alphabetic-only.
        .map(stem_if_latin)
        .collect()
}

/// Default cosine threshold for the v3 semantic fallback. Calibrated for
/// the Google `text-embedding-*` model family — empirically, semantically
/// related sentence-level pairs cluster ≥0.55 in this embedding space
/// while unrelated pairs sit below it. Operators can override via the
/// `REIN_EVAL_SEMANTIC_THRESHOLD` env var read in `bin/rein_eval.rs`.
pub const DEFAULT_SEMANTIC_THRESHOLD: f32 = 0.55;

/// Optional semantic fallback for [`KeywordOverlapHitChecker`].
///
/// When attached, keywords that miss the stemmed token-set get a second
/// chance via embedding cosine similarity: if `embedder.embed(keyword)`
/// has cosine ≥ `similarity_threshold` against `embedder.embed(text)`,
/// the keyword counts as a hit. This catches synonymy that stemming
/// can't bridge (e.g. "Ebbinghaus" ≈ "decay").
///
/// The embedder is `Arc`'d so the same handle can be cheaply cloned into
/// every checker instance the eval pipeline constructs (one per command
/// invocation; the embedder's HTTP client is reused under the hood).
#[derive(Clone)]
pub struct SemanticFallback {
    pub embedder: std::sync::Arc<crate::embed::EmbedderKind>,
    pub similarity_threshold: f32,
}

impl std::fmt::Debug for SemanticFallback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use crate::types::traits::Embedder;
        f.debug_struct("SemanticFallback")
            .field("model", &self.embedder.model_name())
            .field("similarity_threshold", &self.similarity_threshold)
            .finish()
    }
}

/// Hybrid [`HitChecker`]: stem-tokenized fast path + optional semantic
/// fallback (see [`SemanticFallback`]).
///
/// `Default` returns the stem-only configuration so existing call sites
/// (and unit tests) compile unchanged. Production rein-eval commands wrap
/// the configured embedder via [`Self::with_semantic`] when one is
/// available, so the same checker covers both regimes.
///
/// The legacy `check_hit(evidence, canonical)` predicate (top-5 frequency
/// from evidence, ≥3 overlap with canonical) is preserved for the v0.23
/// resummerize baseline path. The richer per-keyword scoring used by
/// concept-summary + recall-synthesis lives in [`crate::eval::concept_summary::score_concept_case`].
#[derive(Debug, Default, Clone)]
pub struct KeywordOverlapHitChecker {
    pub semantic: Option<SemanticFallback>,
}

impl KeywordOverlapHitChecker {
    /// Stem-only checker — no embedder calls, fully sync, deterministic.
    pub fn stem_only() -> Self {
        Self { semantic: None }
    }

    /// Stem fast path + semantic fallback. Use the configured embedder
    /// for any keyword that misses the stem-tokenized text token-set.
    pub fn with_semantic(
        embedder: std::sync::Arc<crate::embed::EmbedderKind>,
        similarity_threshold: f32,
    ) -> Self {
        Self {
            semantic: Some(SemanticFallback {
                embedder,
                similarity_threshold,
            }),
        }
    }
}

impl HitChecker for KeywordOverlapHitChecker {
    fn check_hit(&self, evidence_entry_content: &str, canonical: &str) -> bool {
        let tokens = tokenize(evidence_entry_content);
        if tokens.is_empty() {
            return false;
        }

        // Count frequencies.
        let mut counts: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
        for t in &tokens {
            *counts.entry(t.clone()).or_insert(0) += 1;
        }

        // Pick top-5 by frequency, breaking ties by lexicographic order so
        // this is deterministic.
        let mut scored: Vec<(String, u32)> = counts.into_iter().collect();
        scored.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let top_k: Vec<String> = scored.into_iter().take(5).map(|(w, _)| w).collect();
        if top_k.is_empty() {
            return false;
        }

        let canonical_lower = canonical.to_lowercase();
        let canonical_tokens: std::collections::HashSet<String> =
            tokenize(&canonical_lower).into_iter().collect();

        let overlap = top_k
            .iter()
            .filter(|w| canonical_tokens.contains(*w))
            .count();
        overlap >= 3
    }
}

/// Cosine similarity between two equally-sized embeddings.
/// Returns 0.0 if either vector is empty or zero-norm — degenerate inputs
/// can never match, so 0 is the safe sentinel under the threshold check.
pub(crate) fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Run a `Future` to completion from a sync caller, mirroring
/// `ops::recall_synthesis::call_synthesis_llm_sync`'s pattern: prefer the
/// surrounding tokio runtime via `Handle::try_current`, otherwise spin up
/// a current-thread runtime. Centralized here so semantic-fallback
/// embedding calls and any future async eval helpers share one
/// implementation rather than re-deriving the bridge.
pub(crate) fn block_on_future<F, T>(fut: F) -> Result<T, crate::types::ReinError>
where
    F: std::future::Future<Output = Result<T, crate::types::ReinError>>,
{
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        return tokio::task::block_in_place(|| handle.block_on(fut));
    }
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| {
            crate::types::ReinError::Config(format!("failed to build tokio runtime: {e}"))
        })?;
    rt.block_on(fut)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keyword_checker_hit_case() {
        // Shared high-frequency tokens between evidence and canonical:
        // "concepts", "wiki", "links" each appear in both. The Phase 1
        // checker is literal (no stemming), so we pick words that repeat
        // consistently across both strings.
        let evidence = "Neural wiki renders concepts with concepts wiki links and concepts links.";
        let canonical = "Neural wiki renders concepts with links between concepts.";
        let checker = KeywordOverlapHitChecker::stem_only();
        assert!(checker.check_hit(evidence, canonical));
    }

    #[test]
    fn keyword_checker_miss_case() {
        let evidence = "Rust borrow checker ownership lifetimes compile errors";
        let canonical = "Python garbage collector reference counting";
        let checker = KeywordOverlapHitChecker::stem_only();
        assert!(!checker.check_hit(evidence, canonical));
    }

    #[test]
    fn keyword_checker_empty_evidence_misses() {
        let checker = KeywordOverlapHitChecker::stem_only();
        assert!(!checker.check_hit("", "anything here"));
    }

    #[test]
    fn keyword_checker_stem_collapses_morphology() {
        // v3 (HIT_CHECKER_VERSION = 3): Snowball Porter2 stemming collapses
        // morphological variants so "extracting"/"extraction"/"extracts"
        // share one stem. Three distinct stems in the evidence
        // (extract*, recall*, summar*) all appear in the canonical →
        // ≥3 overlap → hit. Under v2's bare tokenizer those variants would
        // have been three different surface forms apiece (frequency 1 each),
        // letting unrelated singletons displace them from the top-5.
        let evidence = "extract extraction extracts extracting recall recalled recalling \
                        summary summarize summarized";
        let canonical = "extract recall summary pipeline inventory";
        let checker = KeywordOverlapHitChecker::stem_only();
        assert!(
            checker.check_hit(evidence, canonical),
            "v3 stemming must collapse morphological families so the top-5 \
             slots are dominated by extract*, recall*, summar* — all of \
             which the canonical also stems to."
        );
    }

    #[test]
    fn stem_if_latin_preserves_cjk_and_digits() {
        // CJK input → unchanged (Rust's `is_alphabetic` returns true for
        // CJK; the gate is `is_ascii_alphabetic` to keep them out of the
        // English stemmer per `feedback_rust_cjk_alphanumeric_trap`).
        assert_eq!(stem_if_latin("用户"), "用户");
        // Digits → unchanged (would otherwise corrupt version tokens).
        assert_eq!(stem_if_latin("v0252"), "v0252");
        // Pure ASCII alpha → stemmed.
        assert_eq!(stem_if_latin("extracting"), "extract");
        assert_eq!(stem_if_latin("recalled"), "recal");
    }

    #[test]
    fn cosine_similarity_basics() {
        // Identical vectors → 1.0
        let a = vec![1.0_f32, 0.0, 0.0];
        let b = vec![1.0_f32, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
        // Orthogonal → 0.0
        let c = vec![0.0_f32, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 1e-6);
        // Opposite → -1.0
        let d = vec![-1.0_f32, 0.0, 0.0];
        assert!((cosine_similarity(&a, &d) + 1.0).abs() < 1e-6);
        // Empty / mismatched length → 0.0 (safe sentinel under threshold).
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
        assert_eq!(cosine_similarity(&[1.0_f32], &[1.0, 1.0]), 0.0);
        // Zero vector → 0.0 (no division-by-zero).
        let z = vec![0.0_f32, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &z), 0.0);
    }

    #[test]
    fn keyword_checker_cjk_word_level_hit() {
        // v2 (HIT_CHECKER_VERSION = 2) routes CJK input through jieba.
        // v1 used `is_alphanumeric` which collapsed every Chinese sentence
        // into a single mega-token (CJK chars are alphanumeric in
        // Unicode), making the eval's `cjk` category numbers noise.
        // After the fix, repeated CJK words like "用户偏好" are
        // distinguishable tokens that the top-5/≥3-overlap predicate can
        // actually score.
        let evidence = "用户偏好简洁输出 用户偏好中文回复 用户偏好周末减少通知 用户偏好简洁输出";
        let canonical = "用户偏好简洁输出和中文回复，周末减少通知";
        let checker = KeywordOverlapHitChecker::stem_only();
        assert!(
            checker.check_hit(evidence, canonical),
            "CJK content with shared `用户`/`偏好`/`简洁`/`输出` tokens between \
             evidence and canonical must score as a hit; v1 always returned \
             false here because tokenize() emitted a single mega-token"
        );
    }

    #[test]
    fn keyword_checker_cjk_paraphrase_miss() {
        // Same shape as keyword_checker_miss_case but in CJK — Korean
        // evidence vs unrelated Chinese canonical. v1 would return true
        // (single-token degenerate equality on long strings); v2 correctly
        // separates words and finds no overlap.
        let evidence = "리뷰어가 커밋 메시지 형식에 대해 피드백을 남겼습니다";
        let canonical = "用户偏好简洁输出和中文回复";
        let checker = KeywordOverlapHitChecker::stem_only();
        assert!(
            !checker.check_hit(evidence, canonical),
            "evidence and canonical share no meaningful tokens; checker \
             must return false"
        );
    }
}
