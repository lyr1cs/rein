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

pub mod mcnemar;
pub mod scorecard;

pub use mcnemar::{mcnemar, McNemarResult, PairedOutcome};
pub use scorecard::{decide_ship, CategoryStats, Scorecard, ShipDecision, ShipReason};

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

/// Tokenize into lowercase alphanumeric words, filtering stop words and
/// very-short tokens. Deliberately simple — see [`HitChecker`] docs for the
/// Phase 1 caveat.
fn tokenize(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty() && w.len() > 2 && !STOP_WORDS.contains(w))
        .map(|w| w.to_string())
        .collect()
}

/// Phase 1 [`HitChecker`] stub: extract top-5 keywords from the evidence
/// entry by raw frequency, then return `true` iff at least 3 of those keywords
/// appear in the canonical text.
///
/// This is intentionally simple and expected to be replaced by a better
/// (LLM-backed / semantic) oracle in later phases.
#[derive(Debug, Default, Clone, Copy)]
pub struct KeywordOverlapHitChecker;

impl HitChecker for KeywordOverlapHitChecker {
    fn check_hit(&self, evidence_entry_content: &str, canonical: &str) -> bool {
        let tokens = tokenize(evidence_entry_content);
        if tokens.is_empty() {
            return false;
        }

        // Count frequencies.
        let mut counts: std::collections::HashMap<String, u32> =
            std::collections::HashMap::new();
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

        let overlap = top_k.iter().filter(|w| canonical_tokens.contains(*w)).count();
        overlap >= 3
    }
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
        let checker = KeywordOverlapHitChecker;
        assert!(checker.check_hit(evidence, canonical));
    }

    #[test]
    fn keyword_checker_miss_case() {
        let evidence = "Rust borrow checker ownership lifetimes compile errors";
        let canonical = "Python garbage collector reference counting";
        let checker = KeywordOverlapHitChecker;
        assert!(!checker.check_hit(evidence, canonical));
    }

    #[test]
    fn keyword_checker_empty_evidence_misses() {
        let checker = KeywordOverlapHitChecker;
        assert!(!checker.check_hit("", "anything here"));
    }
}
