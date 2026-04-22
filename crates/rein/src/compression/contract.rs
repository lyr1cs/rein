//! Lossless compression contract — seven invariants that any proposed
//! resummerize output must satisfy.
//!
//! # Design
//!
//! Each invariant is a pure function `fn(&ContractInput, &str) -> Result<(),
//! Violation>`. All checks are *lexical/structural* — they act as a coarse
//! guardrail and deliberately trade recall for zero false negatives on the
//! narrow classes of regression they target. Semantic equivalence (NLI-style)
//! and true contradiction resolution are explicitly out of scope; they are
//! future work for a later version of the contract.
//!
//! # The seven invariants
//!
//! | id | name                       | catches                                           |
//! |----|----------------------------|---------------------------------------------------|
//! | 1  | `no_new_facts`             | output introducing content absent from the input  |
//! | 2  | `temporal_anchors_preserved` | dropped dates / version tags that recurred       |
//! | 3  | `conflict_not_silenced`    | silently resolving contested preferences           |
//! | 4  | `length_bounded`           | exceeding the caller-supplied length budget        |
//! | 5  | `evidence_immutable`       | (DB-layer stub; see module docs)                   |
//! | 6  | `cjk_integrity`            | dropping recurring CJK / Hiragana / Hangul chars   |
//! | 7  | `code_block_preserved`     | stripping fenced code block opening fences         |
//!
//! # Parameters (hardcoded by design)
//!
//! - `no_new_facts`: requires ≥ 90% of output trigrams to also appear in input
//! - `temporal_anchors_preserved`: anchor must recur ≥ 2 times in evidence
//! - `conflict_not_silenced`: ≥ 50% of distinct objects per verb must persist
//! - `length_bounded`: output char count ≤ target_bytes × 1.1
//! - `cjk_integrity`: char must appear ≥ 2 times in evidence
//!
//! These are intentionally not tunable — the philosophy is zero subjective
//! params. If a number needs to change, it's a contract revision, not a knob.
//!
//! # Adversarial fixtures
//!
//! Seed fixtures live in `crates/rein/tests/fixtures/resummerize/` as JSON
//! arrays grouped by category:
//!
//! - `cjk.json` — CJK / kana / hangul merges
//! - `code_blocks.json` — fenced Rust/SQL/Python/TOML blocks
//! - `temporal_anchors.json` — dates, versions, bare years
//! - `contradictions.json` — contested preferences / policies
//! - `mixed_encoding.json` — URLs, file paths, unicode punctuation
//!
//! Each case has the shape:
//!
//! ```text
//! {
//!   "case_id": "cjk_001",
//!   "category": "cjk",
//!   "description": "...",
//!   "evidence": [{"content": "...", "merged_at": "2026-04-01T10:00:00Z"}, ...],
//!   "current_canonical": "...",
//!   "target_bytes": 8000,
//!   "expected_must_pass": ["cjk_integrity", "length_bounded"]
//! }
//! ```
//!
//! `merged_at` is always RFC3339 UTC. `expected_must_pass` is the subset of
//! invariant names that any acceptable resummerize of this case MUST satisfy
//! — the eval harness will use this list to score LLM-generated candidates.

use chrono::{DateTime, Utc};
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

/// One piece of evidence feeding into a resummerize call.
///
/// `content` is the raw text that was merged into the canonical. `merged_at`
/// is when it was appended. The contract only uses `content`; `merged_at` is
/// retained so callers and fixtures can reason about ordering.
#[derive(Debug, Clone)]
pub struct EvidenceEntry {
    pub content: String,
    pub merged_at: DateTime<Utc>,
}

/// Snapshot of the input to a resummerize call.
#[derive(Debug, Clone)]
pub struct ContractInput<'a> {
    pub evidence: &'a [EvidenceEntry],
    pub current_canonical: &'a str,
    pub target_bytes: usize,
}

/// A single invariant failure. `invariant` is the stable name (matches the
/// table above); `detail` is a human-readable explanation that may change
/// between versions — tests SHOULD assert on `.invariant`, not on `detail`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Violation {
    pub invariant: &'static str,
    pub detail: String,
}

impl Violation {
    fn new(invariant: &'static str, detail: impl Into<String>) -> Self {
        Self {
            invariant,
            detail: detail.into(),
        }
    }
}

/// Top-level gate. Returns `Ok(())` only if all seven invariants pass;
/// otherwise returns the complete list of violations (so callers can log the
/// full failure set rather than one-at-a-time).
pub fn check_all(input: &ContractInput, output: &str) -> Result<(), Vec<Violation>> {
    let mut violations = Vec::new();
    for (_, res) in check_each(input, output) {
        if let Err(v) = res {
            violations.push(v);
        }
    }
    if violations.is_empty() {
        Ok(())
    } else {
        Err(violations)
    }
}

/// Per-invariant inspection, used by the eval harness for scoring.
pub fn check_each(
    input: &ContractInput,
    output: &str,
) -> Vec<(&'static str, Result<(), Violation>)> {
    vec![
        ("no_new_facts", no_new_facts(input, output)),
        (
            "temporal_anchors_preserved",
            temporal_anchors_preserved(input, output),
        ),
        ("conflict_not_silenced", conflict_not_silenced(input, output)),
        ("length_bounded", length_bounded(input, output)),
        ("evidence_immutable", evidence_immutable(input, output)),
        ("cjk_integrity", cjk_integrity(input, output)),
        ("code_block_preserved", code_block_preserved(input, output)),
    ]
}

// ---------------------------------------------------------------------------
// 1. no_new_facts
// ---------------------------------------------------------------------------

/// Checks that at least 90% of output trigrams also appear somewhere in the
/// concatenation of `evidence` + `current_canonical`. Lowercases first; CJK
/// characters are unaffected by lowercasing. Whitespace-only trigrams are
/// skipped.
///
/// This is a lexical proxy for "no new facts" — it catches gross invention
/// (e.g. an LLM making up names) but not paraphrase-level contradictions or
/// semantic insertion of minority views. True NLI-style entailment checking
/// is future work.
pub fn no_new_facts(input: &ContractInput, output: &str) -> Result<(), Violation> {
    let mut input_bag: HashSet<[char; 3]> = HashSet::new();
    for e in input.evidence {
        extend_trigrams(&e.content, &mut input_bag);
    }
    extend_trigrams(input.current_canonical, &mut input_bag);

    let output_trigrams: Vec<[char; 3]> = collect_trigrams(output);
    if output_trigrams.is_empty() {
        return Ok(());
    }
    let mut matched = 0usize;
    for tg in &output_trigrams {
        if input_bag.contains(tg) {
            matched += 1;
        }
    }
    // ratio ≥ 0.90  ⇔  10 * matched ≥ 9 * total (integer-safe).
    if 10 * matched < 9 * output_trigrams.len() {
        let ratio = matched as f64 / output_trigrams.len() as f64;
        return Err(Violation::new(
            "no_new_facts",
            format!(
                "only {:.1}% of output trigrams found in input (need ≥ 90%)",
                ratio * 100.0
            ),
        ));
    }
    Ok(())
}

fn collect_trigrams(text: &str) -> Vec<[char; 3]> {
    let lowered: Vec<char> = text.to_lowercase().chars().collect();
    let mut out = Vec::new();
    if lowered.len() < 3 {
        return out;
    }
    for w in lowered.windows(3) {
        if w[0].is_whitespace() && w[1].is_whitespace() && w[2].is_whitespace() {
            continue;
        }
        out.push([w[0], w[1], w[2]]);
    }
    out
}

fn extend_trigrams(text: &str, bag: &mut HashSet<[char; 3]>) {
    for tg in collect_trigrams(text) {
        bag.insert(tg);
    }
}

// ---------------------------------------------------------------------------
// 2. temporal_anchors_preserved
// ---------------------------------------------------------------------------

fn date_regex() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"\b\d{4}-\d{1,2}-\d{1,2}\b").unwrap())
}

fn version_regex() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"\bv\d+\.\d+(?:\.\d+)?\b").unwrap())
}

fn year_regex() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"\b20\d{2}\b").unwrap())
}

fn extract_anchors(text: &str, bag: &mut HashMap<String, u32>) {
    for re in [date_regex(), version_regex(), year_regex()] {
        for m in re.find_iter(text) {
            *bag.entry(m.as_str().to_owned()).or_insert(0) += 1;
        }
    }
}

/// Every date / version / bare-year token that appears ≥ 2 times across all
/// evidence must appear ≥ 1 time in the output. Note that a full date
/// (`2026-04-22`) and its bare-year substring (`2026`) are tracked as
/// separate anchors — overlap is fine and both counts are kept independently.
pub fn temporal_anchors_preserved(
    input: &ContractInput,
    output: &str,
) -> Result<(), Violation> {
    let mut evidence_anchors: HashMap<String, u32> = HashMap::new();
    for e in input.evidence {
        extract_anchors(&e.content, &mut evidence_anchors);
    }
    let mut missing = Vec::new();
    for (anchor, count) in &evidence_anchors {
        if *count >= 2 && !output.contains(anchor.as_str()) {
            missing.push(anchor.clone());
        }
    }
    if !missing.is_empty() {
        missing.sort();
        return Err(Violation::new(
            "temporal_anchors_preserved",
            format!("dropped recurring anchors: {}", missing.join(", ")),
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 3. conflict_not_silenced
// ---------------------------------------------------------------------------

fn positive_pref_regex() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(r"(?i)(prefer|prefers|like|likes|use|uses|chose|chooses|decided)\s+(\w+)")
            .unwrap()
    })
}

fn negative_pref_regex() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"(?i)(don't|do\s*not|avoid|reject)\s+(\w+)").unwrap())
}

/// Detects contested preferences in evidence and requires **every** distinct
/// object for a given verb to persist in the output. The earlier ≥50%
/// threshold was tightened per Codex audit finding H4 — with two contested
/// objects the 50% rule permits one-side silencing, which is precisely the
/// invariant's whole reason to exist.
///
/// # Limitations (by design)
///
/// - We match the *literal captured verb* (e.g. `prefer` vs `prefers` are
///   distinct). This is a coarse filter — the spec calls this out as
///   "future work".
/// - We do not attempt to detect the subject. We assume contested objects
///   under the same verb refer to the same subject.
/// - A phrase like "prefer Rust" vs "don't prefer Rust" is currently
///   considered one object under `prefer` and one object under
///   `don't`/`do not`, not a contradiction between them. That gap is
///   documented and left for a future invariant that reasons about polarity.
pub fn conflict_not_silenced(
    input: &ContractInput,
    output: &str,
) -> Result<(), Violation> {
    let mut per_verb: HashMap<String, HashSet<String>> = HashMap::new();
    for e in input.evidence {
        collect_prefs(&e.content, &mut per_verb);
    }
    let output_lower = output.to_lowercase();
    for objects in per_verb.values() {
        if objects.len() < 2 {
            continue;
        }
        let kept = objects
            .iter()
            .filter(|o| output_lower.contains(o.as_str()))
            .count();
        // All contested objects must survive. Dropping even one collapses
        // the contradiction the invariant is supposed to preserve.
        if kept < objects.len() {
            let mut listed: Vec<&String> = objects.iter().collect();
            listed.sort();
            let listed_s: Vec<String> = listed.iter().map(|s| (*s).clone()).collect();
            return Err(Violation::new(
                "conflict_not_silenced",
                format!(
                    "contested objects dropped: {} (kept {}/{})",
                    listed_s.join(", "),
                    kept,
                    objects.len()
                ),
            ));
        }
    }
    Ok(())
}

fn collect_prefs(text: &str, out: &mut HashMap<String, HashSet<String>>) {
    for cap in positive_pref_regex().captures_iter(text) {
        let verb = cap.get(1).map(|m| m.as_str().to_lowercase()).unwrap_or_default();
        let obj = cap.get(2).map(|m| m.as_str().to_lowercase()).unwrap_or_default();
        if !verb.is_empty() && !obj.is_empty() {
            out.entry(verb).or_default().insert(obj);
        }
    }
    for cap in negative_pref_regex().captures_iter(text) {
        let verb_raw = cap.get(1).map(|m| m.as_str().to_lowercase()).unwrap_or_default();
        let verb = verb_raw.split_whitespace().collect::<Vec<_>>().join(" ");
        let obj = cap.get(2).map(|m| m.as_str().to_lowercase()).unwrap_or_default();
        if !verb.is_empty() && !obj.is_empty() {
            out.entry(verb).or_default().insert(obj);
        }
    }
}

// ---------------------------------------------------------------------------
// 4. length_bounded
// ---------------------------------------------------------------------------

/// UTF-8 byte length of the output must be ≤ min(target_bytes + 10%,
/// `MERGE_CONTENT_CAP`).
///
/// Unit is **bytes**, not codepoints, which matches upstream semantics:
/// - `MERGE_CONTENT_CAP` compares `String::len()` (bytes) against
///   10_000, so the cap is byte-based.
/// - `recompute_canonical_length_stats` uses `length(CAST(content AS BLOB))`
///   (Codex H3), so the adaptive target the LLM sees is also a byte count.
///
/// The 10% tolerance used to be free — so a `target_bytes = 10_000`
/// output of 10_500 bytes passed the contract but immediately blew the
/// upstream cap on the next merge (Codex round-2 MEDIUM). We now clamp
/// the budget at `MERGE_CONTENT_CAP` regardless of how high the
/// target+tolerance computes to.
pub fn length_bounded(input: &ContractInput, output: &str) -> Result<(), Violation> {
    let actual = output.len();
    let tolerant = input.target_bytes.saturating_add(input.target_bytes / 10);
    let budget = tolerant.min(crate::store::sqlite::MERGE_CONTENT_CAP);
    if actual > budget {
        return Err(Violation::new(
            "length_bounded",
            format!(
                "output {} bytes > budget {} bytes (target {} + 10%, clamped at cap {})",
                actual,
                budget,
                input.target_bytes,
                crate::store::sqlite::MERGE_CONTENT_CAP,
            ),
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 5. evidence_immutable (stub by design)
// ---------------------------------------------------------------------------

/// Stub — evidence immutability is enforced at the DB layer (Agent A's
/// store). The caller MUST verify that `memory_evidence` rows are unchanged
/// before and after resummerize. This function always returns `Ok(())` so
/// that `check_each` still reports the invariant name in its output.
pub fn evidence_immutable(_input: &ContractInput, _output: &str) -> Result<(), Violation> {
    Ok(())
}

// ---------------------------------------------------------------------------
// 6. cjk_integrity
// ---------------------------------------------------------------------------

fn is_cjk(ch: char) -> bool {
    let c = ch as u32;
    (0x4E00..=0x9FFF).contains(&c)            // CJK Unified
        || (0x3400..=0x4DBF).contains(&c)     // CJK Extension A
        || (0x20000..=0x2A6DF).contains(&c)   // CJK Extension B
        || (0x3040..=0x309F).contains(&c)     // Hiragana
        || (0x30A0..=0x30FF).contains(&c)     // Katakana
        || (0xAC00..=0xD7AF).contains(&c)     // Hangul syllables
}

/// Every CJK / kana / hangul codepoint appearing ≥ 2 times across evidence
/// must appear ≥ 1 time in the output. This is a fairly aggressive check —
/// it rejects outputs that drop any recurring ideograph, which is the right
/// default for a corpus that's partially Chinese.
pub fn cjk_integrity(input: &ContractInput, output: &str) -> Result<(), Violation> {
    let mut counts: HashMap<char, u32> = HashMap::new();
    for e in input.evidence {
        for ch in e.content.chars() {
            if is_cjk(ch) {
                *counts.entry(ch).or_insert(0) += 1;
            }
        }
    }
    let output_chars: HashSet<char> = output.chars().collect();
    let mut missing: Vec<char> = counts
        .iter()
        .filter_map(|(ch, n)| {
            if *n >= 2 && !output_chars.contains(ch) {
                Some(*ch)
            } else {
                None
            }
        })
        .collect();
    if !missing.is_empty() {
        missing.sort();
        let listed: String = missing.iter().collect();
        return Err(Violation::new(
            "cjk_integrity",
            format!("dropped recurring CJK codepoints: {}", listed),
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 7. code_block_preserved
// ---------------------------------------------------------------------------

/// Line-based fenced-code-block parser. Returns `(info_string, body)`
/// pairs where `body` is the content between the opening and matching
/// closing fence, with leading/trailing whitespace trimmed.
///
/// The first component is CommonMark's **info string** — the entire
/// suffix after the three backticks (trimmed). In typical markdown this
/// is just a language identifier (e.g. `rust`, `sql`), but it may also
/// include metadata tokens. We preserve the whole suffix rather than
/// parsing off a "language" subtoken; evidence and output are parsed
/// symmetrically so equality matching still works regardless of what
/// operators put there. Codex round-3 residual doc clarification.
///
/// Uses a simple state machine rather than a regex because the previous
/// `(?sm).*?` form misbehaves on nested tutorial fences (the non-greedy
/// match closes on the FIRST inner fence, truncating the outer body)
/// and silently accepted empty-body fences. Codex round-2 MEDIUM.
///
/// Semantics:
/// - A line is a fence iff its trimmed form starts with three backticks.
///   The characters after the backticks (also trimmed) are the info
///   string (empty if none).
/// - The first fence opens a block; the next fence closes it. This is
///   how CommonMark / GitHub-flavored markdown actually renders — an
///   inner ``` is the closing delimiter, not a nested opening — so any
///   tutorial "show how to write a fenced block" content that embeds
///   ``` inside an outer block will produce two separate captured
///   blocks here (a quirk, but consistent with how the evidence and
///   output will BOTH be parsed, so the equality check still works).
/// - An unclosed opening fence at EOF is treated as if EOF closed it.
/// - Empty-body blocks (whitespace only between fences) are discarded
///   so a `\n```\n` output can't "preserve" a deleted body.
fn parse_code_blocks(text: &str) -> Vec<(String, String)> {
    let mut blocks = Vec::new();
    // (lang, body lines) while inside an open block.
    let mut current: Option<(String, Vec<&str>)> = None;
    for line in text.lines() {
        let trimmed = line.trim();
        let is_fence = trimmed.starts_with("```");
        if is_fence {
            match current.take() {
                Some((info_string, body_lines)) => {
                    // Close the current block.
                    let body = body_lines.join("\n").trim().to_string();
                    if !body.is_empty() {
                        blocks.push((info_string, body));
                    }
                }
                None => {
                    // Open a new block. Info string = everything after the
                    // backticks on the opening fence line, trimmed.
                    let info_string = trimmed
                        .trim_start_matches('`')
                        .trim()
                        .to_string();
                    current = Some((info_string, Vec::new()));
                }
            }
        } else if let Some((_, body_lines)) = current.as_mut() {
            body_lines.push(line);
        }
    }
    if let Some((lang, body_lines)) = current {
        let body = body_lines.join("\n").trim().to_string();
        if !body.is_empty() {
            blocks.push((lang, body));
        }
    }
    blocks
}

/// Every distinct fenced code block body in evidence must appear in the
/// output, matched by `(info_string, body)` as a pair.
///
/// # Why the body, not just the fence
///
/// The pre-M10 version only matched opening-fence lines, so an output like
/// ` ```rust\n``` ` would pass even if the entire body was deleted. The
/// M10 version matched full bodies via regex but tripped on nested/empty
/// fences (Codex round-2). We now parse fences line-by-line and compare
/// the resulting `(lang, body)` sets — exact match, whitespace-trimmed.
///
/// # Limitations
///
/// - Exact-match on body. A re-indented or reformatted block counts as
///   missing. This is the right default for a preservation invariant.
/// - Nested triple-backtick content produces two separate blocks; both
///   evidence and output get the same treatment, so equality still
///   works for typical cases.
pub fn code_block_preserved(
    input: &ContractInput,
    output: &str,
) -> Result<(), Violation> {
    let mut needed: HashSet<(String, String)> = HashSet::new();
    for e in input.evidence {
        for block in parse_code_blocks(&e.content) {
            needed.insert(block);
        }
    }
    if needed.is_empty() {
        return Ok(());
    }

    let mut output_blocks: HashSet<(String, String)> = HashSet::new();
    for block in parse_code_blocks(output) {
        output_blocks.insert(block);
    }

    let missing: Vec<&(String, String)> = needed
        .iter()
        .filter(|k| !output_blocks.contains(*k))
        .collect();
    if !missing.is_empty() {
        let mut sorted: Vec<&(String, String)> = missing.into_iter().collect();
        sorted.sort();
        let listed = sorted
            .iter()
            .map(|(lang, body)| {
                let preview: String = body.chars().take(40).collect();
                format!("```{lang} [{preview}…]")
            })
            .collect::<Vec<_>>()
            .join(" | ");
        return Err(Violation::new(
            "code_block_preserved",
            format!("missing code blocks (matched by lang+body): {listed}"),
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn ts(day: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 4, day, 10, 0, 0).unwrap()
    }

    fn ev(content: &str, day: u32) -> EvidenceEntry {
        EvidenceEntry {
            content: content.to_owned(),
            merged_at: ts(day),
        }
    }

    fn input_from<'a>(
        ev: &'a [EvidenceEntry],
        canonical: &'a str,
        target: usize,
    ) -> ContractInput<'a> {
        ContractInput {
            evidence: ev,
            current_canonical: canonical,
            target_bytes: target,
        }
    }

    // ---- 1. no_new_facts --------------------------------------------------

    #[test]
    fn no_new_facts_accepts_subset_of_input() {
        let evidence = vec![ev("rein uses sqlite and tantivy for search", 1)];
        let canonical = "rein uses sqlite and tantivy";
        let input = input_from(&evidence, canonical, 8000);
        let output = "rein uses sqlite and tantivy";
        assert!(no_new_facts(&input, output).is_ok());
    }

    #[test]
    fn no_new_facts_rejects_invented_content() {
        let evidence = vec![ev("rein uses sqlite for storage", 1)];
        let input = input_from(&evidence, "", 8000);
        // Long injected text guarantees the ratio falls below 90%.
        let output = "rein uses postgresql and supabase for cloud-native elastic scaling";
        let err = no_new_facts(&input, output).unwrap_err();
        assert_eq!(err.invariant, "no_new_facts");
    }

    // ---- 2. temporal_anchors_preserved ------------------------------------

    #[test]
    fn anchors_preserved_when_output_mentions_recurring_date() {
        let evidence = vec![
            ev("shipped v0.22.0 on 2026-04-22", 1),
            ev("the 2026-04-22 release included A1", 2),
        ];
        let input = input_from(&evidence, "", 8000);
        let output = "v0.22.0 shipped on 2026-04-22";
        assert!(temporal_anchors_preserved(&input, output).is_ok());
    }

    #[test]
    fn anchors_preserved_fails_when_recurring_date_dropped() {
        let evidence = vec![
            ev("shipped v0.22.0 on 2026-04-22", 1),
            ev("the 2026-04-22 release", 2),
        ];
        let input = input_from(&evidence, "", 8000);
        let output = "shipped v0.22.0 recently";
        let err = temporal_anchors_preserved(&input, output).unwrap_err();
        assert_eq!(err.invariant, "temporal_anchors_preserved");
        assert!(err.detail.contains("2026-04-22"));
    }

    #[test]
    fn anchors_ignore_singletons() {
        // 2026-01-01 appears only once — dropping it is fine.
        let evidence = vec![ev("an event occurred on 2026-01-01", 1)];
        let input = input_from(&evidence, "", 8000);
        assert!(temporal_anchors_preserved(&input, "no dates here").is_ok());
    }

    // ---- 3. conflict_not_silenced -----------------------------------------

    #[test]
    fn conflict_silenced_when_one_side_dropped() {
        // Codex audit H4: the prior ≥50% threshold allowed "user prefers
        // rust / user prefers python" to silently resolve to just "rust"
        // and pass. A single contested object dropped must now fail.
        let evidence = vec![
            ev("user prefers rust", 1),
            ev("user prefers python", 2),
        ];
        let input = input_from(&evidence, "", 8000);
        let output = "user prefers rust in some contexts";
        let err = conflict_not_silenced(&input, output).unwrap_err();
        assert_eq!(err.invariant, "conflict_not_silenced");
    }

    #[test]
    fn conflict_passes_when_all_objects_remain() {
        let evidence = vec![
            ev("user prefers rust", 1),
            ev("user prefers python", 2),
        ];
        let input = input_from(&evidence, "", 8000);
        let output = "user prefers rust for systems, python for scripts";
        assert!(conflict_not_silenced(&input, output).is_ok());
    }

    #[test]
    fn conflict_silenced_when_all_objects_dropped() {
        let evidence = vec![
            ev("team chose rust", 1),
            ev("team chose python", 2),
            ev("team chose go", 3),
        ];
        let input = input_from(&evidence, "", 8000);
        let output = "team chose a language";
        let err = conflict_not_silenced(&input, output).unwrap_err();
        assert_eq!(err.invariant, "conflict_not_silenced");
    }

    // ---- 4. length_bounded ------------------------------------------------

    #[test]
    fn length_under_budget_passes() {
        let input = input_from(&[], "", 10);
        assert!(length_bounded(&input, "short").is_ok());
    }

    #[test]
    fn length_over_budget_fails() {
        let input = input_from(&[], "", 10);
        // budget = 10 + 1 = 11 chars
        let err = length_bounded(&input, "0123456789abcdef").unwrap_err();
        assert_eq!(err.invariant, "length_bounded");
    }

    #[test]
    fn length_uses_bytes_not_chars_for_cjk() {
        // Codex audit H3: the contract is byte-based, matching upstream
        // MergeInto cap (`MERGE_CONTENT_CAP = 10_000` on `String::len()`)
        // and the adaptive target_bytes (now `length(CAST(content AS BLOB))`).
        //
        // 10 CJK chars = 30 bytes. target_bytes = 10, budget = 11 bytes →
        // 30 > 11 means the check must FAIL. A char-based check would
        // have silently let a 30 KB CJK output pass an 8 KB target and
        // then blow the upstream 10 KB byte cap.
        let input = input_from(&[], "", 10);
        let err = length_bounded(&input, "中文文字测试样本数据").unwrap_err();
        assert_eq!(err.invariant, "length_bounded");

        // Same 30 bytes passes under an 80-byte budget (target 80 + 10%).
        let generous = input_from(&[], "", 80);
        assert!(length_bounded(&generous, "中文文字测试样本数据").is_ok());
    }

    #[test]
    fn length_bounded_clamps_tolerance_at_merge_cap() {
        // Codex round-2 MEDIUM: with `target_bytes = MAX_RESUMMERIZE_TARGET`
        // (= `MERGE_CONTENT_CAP` = 10_000), the naive `target + 10%` = 11_000
        // used to admit an 11_000-byte output that would immediately blow
        // the upstream byte cap on the next merge. Budget must clamp at
        // `MERGE_CONTENT_CAP`.
        use crate::store::sqlite::MERGE_CONTENT_CAP;
        let input = input_from(&[], "", MERGE_CONTENT_CAP);
        // Exactly at cap is fine.
        let at_cap = "a".repeat(MERGE_CONTENT_CAP);
        assert!(length_bounded(&input, &at_cap).is_ok());
        // One byte over the cap must fail, even though target + 10% =
        // 11_000 would nominally allow it.
        let over_cap = "a".repeat(MERGE_CONTENT_CAP + 1);
        let err = length_bounded(&input, &over_cap).unwrap_err();
        assert_eq!(err.invariant, "length_bounded");
    }

    // ---- 5. evidence_immutable --------------------------------------------

    #[test]
    fn evidence_immutable_is_stub_always_ok() {
        let input = input_from(&[], "", 8000);
        assert!(evidence_immutable(&input, "anything").is_ok());
        assert!(evidence_immutable(&input, "").is_ok());
    }

    // ---- 6. cjk_integrity -------------------------------------------------

    #[test]
    fn cjk_kept_when_output_retains_recurring_chars() {
        let evidence = vec![
            ev("用户偏好中文输出", 1),
            ev("用户偏好简洁", 2),
        ];
        let input = input_from(&evidence, "", 8000);
        // 用户偏好 each appear ≥ 2 times in evidence
        let output = "用户偏好中文输出且简洁";
        assert!(cjk_integrity(&input, output).is_ok());
    }

    #[test]
    fn cjk_fails_when_recurring_char_dropped() {
        let evidence = vec![
            ev("用户偏好中文", 1),
            ev("用户偏好简洁", 2),
        ];
        let input = input_from(&evidence, "", 8000);
        let output = "user prefers cjk";
        let err = cjk_integrity(&input, output).unwrap_err();
        assert_eq!(err.invariant, "cjk_integrity");
    }

    #[test]
    fn cjk_ignores_singletons() {
        let evidence = vec![ev("一 only once", 1)];
        let input = input_from(&evidence, "", 8000);
        assert!(cjk_integrity(&input, "english only").is_ok());
    }

    // ---- 7. code_block_preserved ------------------------------------------

    #[test]
    fn code_fences_preserved_when_output_has_them() {
        let evidence = vec![ev(
            "example:\n```rust\nfn main() {}\n```",
            1,
        )];
        let input = input_from(&evidence, "", 8000);
        let output = "example:\n```rust\nfn main() {}\n```";
        assert!(code_block_preserved(&input, output).is_ok());
    }

    #[test]
    fn code_fences_fail_when_output_drops_them() {
        let evidence = vec![ev(
            "example:\n```rust\nfn main() {}\n```",
            1,
        )];
        let input = input_from(&evidence, "", 8000);
        let output = "example: fn main() {}";
        let err = code_block_preserved(&input, output).unwrap_err();
        assert_eq!(err.invariant, "code_block_preserved");
        assert!(err.detail.contains("rust"));
    }

    #[test]
    fn code_block_empty_body_output_is_rejected() {
        // Codex round-2 MEDIUM: the previous regex-based implementation
        // matched fence lines only, so ` ```rust\n``` ` passed even when
        // the body was entirely deleted. The line-based parser now drops
        // empty-body fences from the parsed set, so an output that
        // preserves the fences but nothing inside can no longer pass
        // when the evidence carries a real body.
        let evidence = vec![ev(
            "example:\n```rust\nfn main() { println!(\"preserved\"); }\n```",
            1,
        )];
        let input = input_from(&evidence, "", 8000);
        // Output keeps the opening/closing fence lines but with a blank
        // body in between — functionally the block is gone.
        let empty_body_output = "example:\n```rust\n\n```";
        let err = code_block_preserved(&input, empty_body_output).unwrap_err();
        assert_eq!(err.invariant, "code_block_preserved");
    }

    #[test]
    fn code_block_parser_handles_unclosed_eof_fence() {
        // Unclosed opening fence at EOF is treated as if EOF closed it,
        // so the body still counts as a block. Prevents a subtle
        // truncation-by-EOF exploit where dropping the closing fence
        // would (pre-parser) erase the block.
        let evidence = vec![ev(
            "```rust\nfn retained() {}\n",
            1,
        )];
        let input = input_from(&evidence, "", 8000);
        // Output also unclosed but same body → passes.
        let output = "```rust\nfn retained() {}\n";
        assert!(code_block_preserved(&input, output).is_ok());
    }

    // ---- check_all / check_each integration -------------------------------

    #[test]
    fn check_all_collects_multiple_violations() {
        let evidence = vec![
            ev("shipped v0.22.0 on 2026-04-22", 1),
            ev("release on 2026-04-22 included 用户偏好", 2),
            ev("release on 2026-04-22 用户偏好", 3),
        ];
        let input = input_from(&evidence, "", 10);
        // Bad output: too long, drops date, drops CJK
        let output = "a very long english output with none of the right tokens whatsoever";
        let err = check_all(&input, output).unwrap_err();
        let names: HashSet<&str> = err.iter().map(|v| v.invariant).collect();
        assert!(names.contains("length_bounded"));
        assert!(names.contains("temporal_anchors_preserved"));
        assert!(names.contains("cjk_integrity"));
    }

    #[test]
    fn check_each_returns_all_seven_invariants() {
        // Use a canonical that already contains the output verbatim so
        // no_new_facts trivially passes — this test only verifies the
        // surface of check_each, not per-invariant behavior.
        let input = input_from(&[], "trivial output", 8000);
        let results = check_each(&input, "trivial output");
        let names: Vec<&'static str> = results.iter().map(|(n, _)| *n).collect();
        assert_eq!(
            names,
            vec![
                "no_new_facts",
                "temporal_anchors_preserved",
                "conflict_not_silenced",
                "length_bounded",
                "evidence_immutable",
                "cjk_integrity",
                "code_block_preserved",
            ]
        );
        for (_, r) in &results {
            assert!(r.is_ok(), "unexpected violation: {:?}", r);
        }
    }

    #[test]
    fn no_new_facts_passes_when_output_under_three_chars() {
        let input = input_from(&[], "", 8000);
        // Fewer than 3 chars = no trigrams = trivially ok.
        assert!(no_new_facts(&input, "ab").is_ok());
        assert!(no_new_facts(&input, "").is_ok());
    }

    #[test]
    fn check_all_passes_on_well_formed_resummerize() {
        let evidence = vec![
            ev(
                "rein v0.22.0 shipped on 2026-04-22 with A1 operation registry",
                1,
            ),
            ev(
                "the 2026-04-22 release moved 38 ops behind the #[op] macro",
                2,
            ),
        ];
        let input = input_from(
            &evidence,
            "rein v0.22.0 shipped 2026-04-22 with A1 registry",
            200,
        );
        let output = "rein v0.22.0 shipped on 2026-04-22 with the A1 operation registry (38 ops behind #[op] macro)";
        let result = check_all(&input, output);
        assert!(result.is_ok(), "expected ok, got {:?}", result);
    }
}
