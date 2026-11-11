use crate::extract::dedup::similarity;

/// Signal categories with associated keywords and scores.
struct Signal {
    keywords: &'static [&'static str],
    score: u32,
}

const SIGNALS: &[Signal] = &[
    Signal {
        keywords: &[
            "uses",
            "architecture",
            "pattern",
            "algorithm",
            "design",
            "framework",
        ],
        score: 3,
    },
    Signal {
        keywords: &[
            "error",
            "fixed",
            "bug",
            "workaround",
            "resolved",
            "crash",
            "fail",
        ],
        score: 3,
    },
    Signal {
        keywords: &[
            "decided",
            "chose",
            "prefer",
            "switched to",
            "selected",
            "picked",
        ],
        score: 4,
    },
    Signal {
        keywords: &[
            "configured",
            "setup",
            "installed",
            "enabled",
            "env",
            "config",
        ],
        score: 2,
    },
    Signal {
        keywords: &[
            "commit", "deploy", "migrate", "refactor", "release", "merge",
        ],
        score: 2,
    },
];

/// Score a single sentence based on keyword matches.
pub fn score_sentence(sentence: &str) -> u32 {
    let lower = sentence.to_lowercase();
    let mut score = 0u32;
    for signal in SIGNALS {
        for keyword in signal.keywords {
            if lower.contains(keyword) {
                score += signal.score;
                break; // Only count each signal category once per sentence
            }
        }
    }
    score
}

/// Largest sentence, in bytes, that pattern extraction accepts as a fact.
/// A single statement never needs more room than a whole merged canonical
/// gets, so anything longer is a blob (minified JSON, base64, a log or code
/// dump without sentence structure) and is skipped rather than stored.
const MAX_FACT_BYTES: usize = crate::store::sqlite::MERGE_CONTENT_CAP;

/// Sentence terminators that end a sentence without trailing whitespace:
/// the ideographic full stop, its halfwidth form, and the fullwidth forms
/// of `.`, `!` and `?`. CJK prose puts no space between sentences.
fn is_cjk_terminator(ch: char) -> bool {
    matches!(ch, '。' | '｡' | '．' | '！' | '？')
}

/// Closing quotes and brackets that stay attached to the sentence they close
/// when they directly follow a CJK terminator (`他说：“好。”`).
fn is_closing_mark(ch: char) -> bool {
    matches!(
        ch,
        '”' | '’' | '」' | '』' | '）' | '】' | '》' | '〉' | ')' | ']'
    )
}

/// Split text into sentences. A sentence ends at:
/// - `.` `!` `?` followed by whitespace or end-of-string,
/// - a CJK terminator (plus any closing quotes/brackets right after it),
/// - a blank line (a line break followed by optional spaces/tabs and another
///   line break). A single line break stays inside the sentence, so
///   hard-wrapped prose is not cut at the wrap.
///
/// Uses pre-collected char_indices for O(n) performance instead of repeated nth() calls.
fn split_sentences(text: &str) -> Vec<&str> {
    let mut sentences = Vec::new();
    let indices: Vec<(usize, char)> = text.char_indices().collect();
    let len = indices.len();
    let mut start = 0;
    // True while walking the run of terminators/closing marks that follows a
    // CJK terminator; the sentence ends at the last char of that run.
    let mut in_cjk_ending = false;

    for i in 0..len {
        let ch = indices[i].1;
        let next = indices.get(i + 1).map(|&(_, c)| c);
        let ends_here = if ch == '\n' {
            indices[i + 1..]
                .iter()
                .map(|&(_, c)| c)
                .find(|c| !matches!(*c, ' ' | '\t' | '\r'))
                == Some('\n')
        } else if is_cjk_terminator(ch) || (in_cjk_ending && is_closing_mark(ch)) {
            in_cjk_ending = true;
            next.is_none_or(|c| !is_cjk_terminator(c) && !is_closing_mark(c))
        } else if ch == '.' || ch == '!' || ch == '?' {
            next.is_none_or(char::is_whitespace)
        } else {
            false
        };

        if ends_here {
            in_cjk_ending = false;
            let byte_start = indices[start].0;
            let byte_end = if i + 1 < len {
                indices[i + 1].0
            } else {
                text.len()
            };
            let sentence = text[byte_start..byte_end].trim();
            if !sentence.is_empty() {
                sentences.push(sentence);
            }
            start = i + 1;
        }
    }

    // Handle trailing text without sentence-ending punctuation
    if start < len {
        let byte_start = indices[start].0;
        let trailing = text[byte_start..].trim();
        if !trailing.is_empty() {
            sentences.push(trailing);
        }
    }

    sentences
}

/// Extract important facts from text. Returns sentences scoring above threshold.
/// Deduplicates using Jaccard similarity > 0.6.
///
/// Sentences longer than `MAX_FACT_BYTES` are skipped before scoring, so a
/// blob neither becomes a fact nor costs a lowercase + tokenize pass.
pub fn extract_facts(text: &str, threshold: u32) -> Vec<String> {
    let mut results: Vec<String> = Vec::new();
    let mut skipped_blobs = 0usize;
    let mut skipped_bytes = 0usize;

    for sentence in split_sentences(text) {
        if sentence.len() > MAX_FACT_BYTES {
            skipped_blobs += 1;
            skipped_bytes += sentence.len();
            continue;
        }
        let score = score_sentence(sentence);
        if score >= threshold {
            // Dedup: skip sentences with Jaccard > 0.6 vs already-selected sentences
            let is_dup = results
                .iter()
                .any(|existing| similarity(existing, sentence) > 0.6);
            if !is_dup {
                results.push(sentence.to_string());
            }
        }
    }

    if skipped_blobs > 0 {
        tracing::debug!(
            skipped_blobs,
            skipped_bytes,
            max_fact_bytes = MAX_FACT_BYTES,
            "pattern extraction skipped oversized sentences"
        );
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_architecture() {
        let score = score_sentence("The system uses a microservices architecture");
        assert!(score >= 3, "Expected score >= 3, got {score}");
    }

    #[test]
    fn test_score_decision() {
        let score = score_sentence("We decided to use PostgreSQL");
        assert!(score >= 4, "Expected score >= 4, got {score}");
    }

    #[test]
    fn test_score_error_fix() {
        let score = score_sentence("Fixed the OOM bug by closing connections");
        assert!(score >= 3, "Expected score >= 3, got {score}");
    }

    #[test]
    fn test_score_low() {
        let score = score_sentence("Hello world");
        assert_eq!(score, 0);
    }

    #[test]
    fn test_extract_filters_low() {
        let text = "Hello world. The system uses a microservices architecture. Today is sunny. We decided to use PostgreSQL for the database.";
        let facts = extract_facts(text, 3);
        assert!(
            facts.len() >= 2,
            "Expected at least 2 facts, got {}",
            facts.len()
        );
        assert!(facts.iter().any(|f| f.contains("microservices")));
        assert!(facts.iter().any(|f| f.contains("PostgreSQL")));
        // "Hello world" and "Today is sunny" should not be included
        assert!(!facts.iter().any(|f| f.contains("Hello world")));
        assert!(!facts.iter().any(|f| f.contains("sunny")));
    }

    #[test]
    fn split_sentences_handles_cjk_terminators_and_paragraph_breaks() {
        // CJK prose has no whitespace after its terminators.
        let cjk = "我们决定使用PostgreSQL。系统采用微服务架构！这样可以吗？他说：“好。”然后继续";
        assert_eq!(
            split_sentences(cjk),
            vec![
                "我们决定使用PostgreSQL。",
                "系统采用微服务架构！",
                "这样可以吗？",
                "他说：“好。”",
                "然后继续",
            ]
        );

        // A blank line separates paragraphs that carry no punctuation.
        assert_eq!(
            split_sentences("fixed the crash\r\n\r\nenabled the config\n \t\nthird item"),
            vec!["fixed the crash", "enabled the config", "third item"]
        );

        // A single line break is a wrap inside the sentence, not a boundary.
        let wrapped = "We decided to use PostgreSQL because the\nexisting tooling supports it.";
        assert_eq!(split_sentences(wrapped), vec![wrapped]);

        // Latin rules are unchanged: a dot inside a token is not a boundary.
        assert_eq!(
            split_sentences("Upgrade to v1.5.1 now. Read config.toml first!"),
            vec!["Upgrade to v1.5.1 now.", "Read config.toml first!"]
        );
    }

    #[test]
    fn extract_facts_skips_blobs_and_keeps_real_sentences() {
        // One "sentence" with a keyword but no boundary at all (base64 /
        // minified JSON shape), longer than a whole canonical may be.
        let blob = format!("error{}", "A".repeat(MAX_FACT_BYTES));
        let text = format!(
            "团队 chose PostgreSQL，因为数据库需要事务。{blob}\n\nWe decided to use SQLite."
        );
        let facts = extract_facts(&text, 3);
        assert_eq!(
            facts,
            vec![
                "团队 chose PostgreSQL，因为数据库需要事务。".to_string(),
                "We decided to use SQLite.".to_string(),
            ]
        );
        assert!(facts.iter().all(|f| f.len() <= MAX_FACT_BYTES));
    }

    #[test]
    fn test_extract_dedup() {
        let text = "The system uses a microservices architecture. The system uses a microservices architecture pattern.";
        let facts = extract_facts(text, 3);
        assert_eq!(
            facts.len(),
            1,
            "Expected 1 fact after dedup, got {}",
            facts.len()
        );
    }
}
