//! Exact-semantics index for A12 leave-one-out exclusion.
//!
//! `loo_exclusion` used to compare every held-out evidence view against every
//! raw memory and every evidence row, hashing and tokenising both sides per
//! pair: O(views × (memories + evidence)) full tokenisations, which was the
//! two-hour stage of the 2026-09-02 live run on 9.7k memories.
//!
//! This index hashes and tokenises each row once and answers a query in
//! O(Σ posting lengths of the query's tokens). It reproduces the original
//! `crate::extract::dedup::similarity` decision exactly:
//!
//! * `similarity = max(jaccard, containment)` over `normalize_tokens` sets;
//! * both set functions are `None`/`0` without a shared token, so for a
//!   bound `> 0` only rows sharing at least one token can qualify, and the
//!   posting-list candidate set is complete;
//! * for a bound `<= 0` every row qualifies (`0.0 >= 0.0`), so every row is
//!   returned;
//! * the arithmetic is the same `f32` division over the same integer counts
//!   (`union = a + b - c`, `smaller = min(a, b)`).
//!
//! Equal-content rows come from a SHA-256 map, matching the original
//! `sha256_hex(a) == sha256_hex(b)` test.

use std::collections::{HashMap, HashSet};

use sha2::{Digest, Sha256};

use crate::extract::dedup::normalize_tokens;

/// A raw memory row as seen by the exclusion step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct A12IndexedMemory<'a> {
    pub id: &'a str,
    pub content: &'a str,
    pub stable_family_id: &'a str,
    pub live_tip_id: Option<&'a str>,
}

/// An evidence row as seen by the exclusion step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct A12IndexedEvidence<'a> {
    pub id: &'a str,
    pub content: &'a str,
}

struct IndexedText {
    hash: String,
    /// Interned token ids, sorted and unique.
    tokens: Vec<u32>,
}

/// Result of one exclusion query. Index positions refer to the order rows
/// were passed to [`A12ExclusionIndex::build`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct A12ExclusionMatches {
    pub content_hash: String,
    pub equal_memory_indices: Vec<usize>,
    pub near_duplicate_memory_indices: Vec<usize>,
    /// Evidence ids whose content is byte-equal or near-duplicate to the query.
    pub held_out_evidence_ids: Vec<String>,
}

pub(crate) struct A12ExclusionIndex<'a> {
    memories: Vec<A12IndexedMemory<'a>>,
    memory_texts: Vec<IndexedText>,
    evidence_ids: Vec<&'a str>,
    evidence_texts: Vec<IndexedText>,
    interner: HashMap<String, u32>,
    memory_by_hash: HashMap<String, Vec<usize>>,
    evidence_by_hash: HashMap<String, Vec<usize>>,
    memory_postings: HashMap<u32, Vec<usize>>,
    evidence_postings: HashMap<u32, Vec<usize>>,
}

pub(crate) fn sha256_hex(content: &str) -> String {
    format!("{:x}", Sha256::digest(content.as_bytes()))
}

impl<'a> A12ExclusionIndex<'a> {
    pub(crate) fn build(
        memories: Vec<A12IndexedMemory<'a>>,
        evidence: Vec<A12IndexedEvidence<'a>>,
    ) -> Self {
        let mut interner: HashMap<String, u32> = HashMap::new();
        let mut memory_texts = Vec::with_capacity(memories.len());
        let mut memory_by_hash: HashMap<String, Vec<usize>> = HashMap::new();
        let mut memory_postings: HashMap<u32, Vec<usize>> = HashMap::new();
        for (index, memory) in memories.iter().enumerate() {
            let text = index_text(memory.content, &mut interner);
            memory_by_hash
                .entry(text.hash.clone())
                .or_default()
                .push(index);
            for token in &text.tokens {
                memory_postings.entry(*token).or_default().push(index);
            }
            memory_texts.push(text);
        }

        let mut evidence_ids = Vec::with_capacity(evidence.len());
        let mut evidence_texts = Vec::with_capacity(evidence.len());
        let mut evidence_by_hash: HashMap<String, Vec<usize>> = HashMap::new();
        let mut evidence_postings: HashMap<u32, Vec<usize>> = HashMap::new();
        for (index, row) in evidence.iter().enumerate() {
            let text = index_text(row.content, &mut interner);
            evidence_by_hash
                .entry(text.hash.clone())
                .or_default()
                .push(index);
            for token in &text.tokens {
                evidence_postings.entry(*token).or_default().push(index);
            }
            evidence_ids.push(row.id);
            evidence_texts.push(text);
        }

        Self {
            memories,
            memory_texts,
            evidence_ids,
            evidence_texts,
            interner,
            memory_by_hash,
            evidence_by_hash,
            memory_postings,
            evidence_postings,
        }
    }

    pub(crate) fn memory(&self, index: usize) -> &A12IndexedMemory<'a> {
        &self.memories[index]
    }

    /// Rows byte-equal to `content` and rows whose lexical similarity reaches
    /// `bound`, plus every evidence id in either category.
    pub(crate) fn query(&self, content: &str, bound: f32) -> A12ExclusionMatches {
        let content_hash = sha256_hex(content);
        let query_tokens: HashSet<String> = normalize_tokens(content);
        let query_size = query_tokens.len();
        let query_ids: Vec<u32> = query_tokens
            .iter()
            .filter_map(|token| self.interner.get(token).copied())
            .collect();

        let mut equal_memory_indices = self
            .memory_by_hash
            .get(&content_hash)
            .cloned()
            .unwrap_or_default();
        equal_memory_indices.sort_unstable();

        let near_duplicate_memory_indices = if bound <= 0.0 {
            (0..self.memories.len()).collect()
        } else {
            near_duplicates(
                &query_ids,
                query_size,
                bound,
                self.memory_texts.len(),
                &self.memory_postings,
                &self.memory_texts,
            )
        };

        let mut evidence_indices: Vec<usize> = self
            .evidence_by_hash
            .get(&content_hash)
            .cloned()
            .unwrap_or_default();
        if bound <= 0.0 {
            evidence_indices.extend(0..self.evidence_texts.len());
        } else {
            evidence_indices.extend(near_duplicates(
                &query_ids,
                query_size,
                bound,
                self.evidence_texts.len(),
                &self.evidence_postings,
                &self.evidence_texts,
            ));
        }
        evidence_indices.sort_unstable();
        evidence_indices.dedup();
        let mut held_out_evidence_ids: Vec<String> = evidence_indices
            .into_iter()
            .map(|index| self.evidence_ids[index].to_string())
            .collect();
        held_out_evidence_ids.sort();
        held_out_evidence_ids.dedup();

        A12ExclusionMatches {
            content_hash,
            equal_memory_indices,
            near_duplicate_memory_indices,
            held_out_evidence_ids,
        }
    }
}

fn index_text(content: &str, interner: &mut HashMap<String, u32>) -> IndexedText {
    let mut tokens: Vec<u32> = normalize_tokens(content)
        .into_iter()
        .map(|token| {
            let next = interner.len() as u32;
            *interner.entry(token).or_insert(next)
        })
        .collect();
    tokens.sort_unstable();
    tokens.dedup();
    IndexedText {
        hash: sha256_hex(content),
        tokens,
    }
}

/// Rows whose `max(jaccard, containment)` with the query reaches `bound`
/// (`bound > 0`). Sorted ascending, unique.
fn near_duplicates(
    query_ids: &[u32],
    query_size: usize,
    bound: f32,
    row_count: usize,
    postings: &HashMap<u32, Vec<usize>>,
    texts: &[IndexedText],
) -> Vec<usize> {
    let mut counts = vec![0_usize; row_count];
    let mut touched = Vec::new();
    for token in query_ids {
        if let Some(rows) = postings.get(token) {
            for &row in rows {
                if counts[row] == 0 {
                    touched.push(row);
                }
                counts[row] += 1;
            }
        }
    }
    let mut matches: Vec<usize> = touched
        .into_iter()
        .filter(|&row| {
            let intersection = counts[row];
            let row_size = texts[row].tokens.len();
            // Mirrors `jaccard_from_sets` / `containment_from_sets`: both are
            // defined here because the row shares at least one token.
            let union = query_size + row_size - intersection;
            let jaccard = intersection as f32 / union as f32;
            let smaller = query_size.min(row_size);
            let containment = intersection as f32 / smaller as f32;
            jaccard.max(containment) >= bound
        })
        .collect();
    matches.sort_unstable();
    matches
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extract::dedup::similarity;

    fn memory<'a>(id: &'a str, content: &'a str) -> A12IndexedMemory<'a> {
        A12IndexedMemory {
            id,
            content,
            stable_family_id: id,
            live_tip_id: Some(id),
        }
    }

    fn evidence<'a>(id: &'a str, content: &'a str) -> A12IndexedEvidence<'a> {
        A12IndexedEvidence { id, content }
    }

    fn reference_near(rows: &[&str], query: &str, bound: f32) -> Vec<usize> {
        rows.iter()
            .enumerate()
            .filter(|(_, content)| similarity(query, content) >= bound)
            .map(|(index, _)| index)
            .collect()
    }

    #[test]
    fn index_bound_zero_marks_every_row() {
        let index = A12ExclusionIndex::build(
            vec![
                memory("a", "alpha beta"),
                memory("b", ""),
                memory("c", "!!!"),
            ],
            vec![evidence("e1", "gamma"), evidence("e2", "")],
        );
        let matches = index.query("unrelated words", 0.0);
        assert_eq!(matches.near_duplicate_memory_indices, vec![0, 1, 2]);
        assert_eq!(matches.held_out_evidence_ids, vec!["e1", "e2"]);
        assert!(matches.equal_memory_indices.is_empty());
    }

    #[test]
    fn index_empty_view_has_no_near_duplicates_above_zero() {
        let index = A12ExclusionIndex::build(
            vec![memory("a", "alpha beta"), memory("b", "")],
            vec![evidence("e1", ""), evidence("e2", "alpha")],
        );
        let matches = index.query("", 0.5);
        assert!(matches.near_duplicate_memory_indices.is_empty());
        // Byte-equal empty rows are still equal-content / held-out evidence.
        assert_eq!(matches.equal_memory_indices, vec![1]);
        assert_eq!(matches.held_out_evidence_ids, vec!["e1"]);
        assert_eq!(matches.content_hash, sha256_hex(""));
    }

    #[test]
    fn index_outputs_sorted_and_deduped() {
        let rows = [
            "alpha beta gamma",
            "alpha beta",
            "beta gamma delta",
            "alpha beta gamma",
        ];
        let index = A12ExclusionIndex::build(
            rows.iter()
                .enumerate()
                .map(|(i, content)| A12IndexedMemory {
                    id: Box::leak(format!("m{i}").into_boxed_str()),
                    content,
                    stable_family_id: "f",
                    live_tip_id: None,
                })
                .collect(),
            rows.iter()
                .enumerate()
                .map(|(i, content)| A12IndexedEvidence {
                    id: Box::leak(format!("e{i}").into_boxed_str()),
                    content,
                })
                .collect(),
        );
        let matches = index.query("alpha beta gamma", 0.6);
        assert_eq!(matches.equal_memory_indices, vec![0, 3]);
        let mut sorted = matches.near_duplicate_memory_indices.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(matches.near_duplicate_memory_indices, sorted);
        assert_eq!(
            matches.near_duplicate_memory_indices,
            reference_near(&rows, "alpha beta gamma", 0.6)
        );
        let mut ids = matches.held_out_evidence_ids.clone();
        ids.sort();
        ids.dedup();
        assert_eq!(matches.held_out_evidence_ids, ids);
    }

    #[test]
    fn index_matches_similarity_reference_on_mixed_corpus() {
        let rows = [
            "the quick brown fox jumps over the lazy dog",
            "quick brown fox",
            "an entirely different sentence about databases",
            "数据库 索引 性能 优化",
            "数据库 性能",
            "",
            "!!! ???",
            "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG",
            "fox dog quick brown lazy jumps over the",
            "index performance database tuning 数据库",
        ];
        let index = A12ExclusionIndex::build(
            rows.iter()
                .enumerate()
                .map(|(i, content)| A12IndexedMemory {
                    id: Box::leak(format!("m{i}").into_boxed_str()),
                    content,
                    stable_family_id: "f",
                    live_tip_id: None,
                })
                .collect(),
            Vec::new(),
        );
        for query in rows.iter().chain(["quick fox", "数据库", "lazy"].iter()) {
            for bound in [0.0_f32, 0.2, 0.35, 0.5, 0.7, 0.9, 1.0] {
                let matches = index.query(query, bound);
                assert_eq!(
                    matches.near_duplicate_memory_indices,
                    reference_near(&rows, query, bound),
                    "query={query:?} bound={bound}"
                );
            }
        }
    }
}
