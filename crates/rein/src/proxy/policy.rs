//! Fast path safety policy for proxy recording and extraction.

use crate::config::ReinConfig;

const SMALLTALK_PHRASES: &[&str] = &[
    "hi",
    "hello",
    "hey",
    "hi there",
    "hello there",
    "good morning",
    "good afternoon",
    "good evening",
    "thanks",
    "thank you",
    "thx",
    "ok",
    "okay",
    "cool",
    "nice",
    "你好",
    "您好",
    "哈喽",
    "嗨",
    "早上好",
    "下午好",
    "晚上好",
    "谢谢",
    "多谢",
    "好的",
    "行",
    "收到",
    "在吗",
];

const QUESTION_HINTS: &[&str] = &[
    "what",
    "why",
    "how",
    "when",
    "where",
    "which",
    "帮我",
    "怎么",
    "如何",
    "为什么",
    "什么",
    "哪个",
];

pub fn should_extract_response(
    config: &ReinConfig,
    source_query: Option<&str>,
    assistant_text: &str,
) -> bool {
    let text = assistant_text.trim();
    if text.chars().count() < config.proxy.store_min_chars {
        return false;
    }
    if source_query.is_some_and(looks_like_smalltalk) {
        return false;
    }
    if looks_like_error_response(text) {
        return false;
    }
    if crate::extract::hooks::scoring::worth_extracting(text) {
        return true;
    }
    crate::extract::patterns::score_sentence(text) >= config.proxy.store_min_score
}

pub fn looks_like_smalltalk(text: &str) -> bool {
    let normalized = normalize_phrase(text);
    if normalized.is_empty() {
        return true;
    }
    if SMALLTALK_PHRASES.contains(&normalized.as_str()) {
        return true;
    }

    let words: Vec<&str> = normalized.split_whitespace().collect();
    let has_question_hint = QUESTION_HINTS.iter().any(|kw| normalized.contains(kw));
    !has_question_hint && words.len() <= 2 && normalized.chars().count() <= 12
}

pub fn looks_like_error_response(text: &str) -> bool {
    let lower = text.to_lowercase();
    [
        "rate limit",
        "request rejected",
        "authentication error",
        "unable to connect to api",
        "api error",
        "connection refused",
        "too many requests",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn normalize_phrase(text: &str) -> String {
    text.to_lowercase()
        .chars()
        .map(|ch| {
            if ch.is_alphanumeric() || ('\u{4e00}'..='\u{9fff}').contains(&ch) || ch.is_whitespace()
            {
                ch
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smalltalk_detection() {
        assert!(looks_like_smalltalk("hello"));
        assert!(looks_like_smalltalk("你好"));
        assert!(looks_like_smalltalk("thanks!"));
        assert!(!looks_like_smalltalk("how do I fix sqlite locking"));
        assert!(!looks_like_smalltalk("为什么索引会卡住"));
    }

    #[test]
    fn test_error_response_detection() {
        assert!(looks_like_error_response(
            "API Error: Request rejected (429)"
        ));
        assert!(!looks_like_error_response("We decided to use SQLite"));
    }
}
