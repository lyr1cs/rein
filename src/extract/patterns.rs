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

/// Split text into sentences by `.` `!` `?` followed by whitespace or end-of-string.
/// Uses pre-collected char_indices for O(n) performance instead of repeated nth() calls.
fn split_sentences(text: &str) -> Vec<&str> {
    let mut sentences = Vec::new();
    let indices: Vec<(usize, char)> = text.char_indices().collect();
    let len = indices.len();
    let mut start = 0;

    for i in 0..len {
        let ch = indices[i].1;
        if ch == '.' || ch == '!' || ch == '?' {
            // Check if followed by whitespace or end-of-string
            let at_end = i + 1 >= len;
            let followed_by_space = !at_end && indices[i + 1].1.is_whitespace();
            if at_end || followed_by_space {
                let byte_start = indices[start].0;
                let byte_end = if i + 1 < len { indices[i + 1].0 } else { text.len() };
                let sentence = text[byte_start..byte_end].trim();
                if !sentence.is_empty() {
                    sentences.push(sentence);
                }
                start = i + 1;
            }
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
pub fn extract_facts(text: &str, threshold: u32) -> Vec<String> {
    let sentences = split_sentences(text);
    let mut results: Vec<String> = Vec::new();

    for sentence in sentences {
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
        assert!(facts.len() >= 2, "Expected at least 2 facts, got {}", facts.len());
        assert!(facts.iter().any(|f| f.contains("microservices")));
        assert!(facts.iter().any(|f| f.contains("PostgreSQL")));
        // "Hello world" and "Today is sunny" should not be included
        assert!(!facts.iter().any(|f| f.contains("Hello world")));
        assert!(!facts.iter().any(|f| f.contains("sunny")));
    }

    #[test]
    fn test_extract_dedup() {
        let text = "The system uses a microservices architecture. The system uses a microservices architecture pattern.";
        let facts = extract_facts(text, 3);
        assert_eq!(facts.len(), 1, "Expected 1 fact after dedup, got {}", facts.len());
    }
}
