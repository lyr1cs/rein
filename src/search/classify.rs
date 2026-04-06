//! Lightweight rule-based query classifier for autonomous retrieval routing.
//! Classifies queries into Episodic/Temporal/Preference/ExactKeyword/Semantic/Exploratory and
//! produces per-query search strategy overrides. No LLM calls — pure string matching.

/// Query type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryType {
    /// "what happened in X session/meeting", "recap of last session"
    Episodic,
    /// "when did X change?", "what happened last week?"
    Temporal,
    /// "what do I prefer/like/want"
    Preference,
    /// Short precise terms, function names, code identifiers
    ExactKeyword,
    /// "things related to memory management"
    Semantic,
    /// "what do I know about...", broad open questions
    Exploratory,
}

/// Per-query search strategy produced by the classifier.
pub struct QueryStrategy {
    pub query_type: QueryType,
    /// CC fusion alpha override (None = use config default).
    pub cc_alpha: Option<f32>,
    /// Multiply the requested limit (1.0 = default, 2.0 = double).
    pub limit_multiplier: f32,
    /// Skip vector search path entirely.
    pub skip_vec: bool,
    /// Skip FTS search path entirely.
    pub skip_fts: bool,
    /// Auto-inject time bounds if not already set by caller.
    pub force_temporal: bool,
    /// For relative temporal queries ("last week"), how many days back.
    pub temporal_days_back: Option<i64>,
}

impl Default for QueryStrategy {
    fn default() -> Self {
        Self {
            query_type: QueryType::Semantic,
            cc_alpha: None,
            limit_multiplier: 1.0,
            skip_vec: false,
            skip_fts: false,
            force_temporal: false,
            temporal_days_back: None,
        }
    }
}

/// Classify a query and produce a search strategy.
/// `has_time_from`/`has_time_to` indicate if the caller already supplied temporal bounds.
pub fn classify(query: &str, has_time_from: bool, has_time_to: bool) -> QueryStrategy {
    let lower = query.to_lowercase();
    let word_count = query.split_whitespace().count();

    // --- Priority 1: Episodic (must check BEFORE Temporal — "what happened" overlaps) ---
    if is_episodic(&lower) {
        return QueryStrategy {
            query_type: QueryType::Episodic,
            cc_alpha: Some(0.5),
            limit_multiplier: 1.5,
            skip_vec: false,
            skip_fts: false,
            force_temporal: false,
            temporal_days_back: None,
        };
    }

    // --- Priority 2: Temporal ---
    if is_temporal(&lower) {
        // Only auto-inject time bounds for relative temporal queries ("last week", "recently").
        // Absolute references ("since 2025-01-01", "history of X") should not be narrowed.
        let is_relative = is_relative_temporal(&lower);
        let days_back = if is_relative {
            Some(extract_temporal_days(&lower))
        } else {
            None
        };
        return QueryStrategy {
            query_type: QueryType::Temporal,
            cc_alpha: Some(0.7),
            limit_multiplier: 1.5,
            skip_vec: false,
            skip_fts: false,
            force_temporal: is_relative && !has_time_from && !has_time_to,
            temporal_days_back: days_back,
        };
    }

    // --- Priority 3: Preference ---
    if is_preference(&lower) {
        return QueryStrategy {
            query_type: QueryType::Preference,
            cc_alpha: Some(0.4),
            limit_multiplier: 2.0,
            skip_vec: false,
            skip_fts: false,
            force_temporal: false,
            temporal_days_back: None,
        };
    }

    // --- Priority 4: ExactKeyword ---
    if is_exact_keyword(query, &lower, word_count) {
        return QueryStrategy {
            query_type: QueryType::ExactKeyword,
            cc_alpha: Some(0.85), // heavy BM25 bias
            limit_multiplier: 1.0,
            skip_vec: false,
            skip_fts: false,
            force_temporal: false,
            temporal_days_back: None,
        };
    }

    // --- Priority 5: Exploratory ---
    if is_exploratory(&lower) {
        return QueryStrategy {
            query_type: QueryType::Exploratory,
            cc_alpha: Some(0.5),   // balanced
            limit_multiplier: 2.0, // cast wide net
            skip_vec: false,
            skip_fts: false,
            force_temporal: false,
            temporal_days_back: None,
        };
    }

    // --- Default: Semantic ---
    QueryStrategy {
        query_type: QueryType::Semantic,
        cc_alpha: Some(0.3), // vector-dominant
        limit_multiplier: 1.0,
        skip_vec: false,
        skip_fts: false,
        force_temporal: false,
        temporal_days_back: None,
    }
}

fn is_episodic(lower: &str) -> bool {
    const EPISODIC_EN: &[&str] = &[
        "what did we decide",
        "recap of",
        "session summary",
        "last conversation",
        "meeting notes",
    ];
    const EPISODIC_ZH: &[&str] = &["会议记录", "上次会话", "总结一下上次"];

    for pat in EPISODIC_EN {
        if lower.contains(pat) {
            return true;
        }
    }
    for pat in EPISODIC_ZH {
        if lower.contains(pat) {
            return true;
        }
    }

    // Compound pattern: "what happened in ... meeting/session"
    if lower.contains("what happened") && (lower.contains("meeting") || lower.contains("session")) {
        return true;
    }

    false
}

fn is_preference(lower: &str) -> bool {
    const PREFERENCE_EN: &[&str] = &[
        "prefer",
        "favorite",
        "favourite",
        "do i like",
        "what do i like",
        "my preference",
        "i tend to",
        "usually choose",
        "go-to",
    ];
    const PREFERENCE_ZH: &[&str] = &["偏好", "喜欢", "最爱", "倾向"];

    for pat in PREFERENCE_EN {
        if lower.contains(pat) {
            return true;
        }
    }
    for pat in PREFERENCE_ZH {
        if lower.contains(pat) {
            return true;
        }
    }

    false
}

fn is_temporal(lower: &str) -> bool {
    const TEMPORAL_EN: &[&str] = &[
        "when did",
        "when was",
        "what changed",
        "what happened",
        "last week",
        "last month",
        "yesterday",
        "recently",
        "recent",
        "this week",
        "this month",
        "changed since",
        "history of",
        "timeline",
        "before march",
        "after january",
        "since last",
    ];
    const TEMPORAL_ZH: &[&str] = &[
        "什么时候",
        "上周",
        "上个月",
        "昨天",
        "最近",
        "变化",
        "历史",
        "改了什么",
        "发生了什么",
    ];

    for pat in TEMPORAL_EN {
        if lower.contains(pat) {
            return true;
        }
    }
    for pat in TEMPORAL_ZH {
        if lower.contains(pat) {
            return true;
        }
    }

    // Date pattern: any YYYY-MM-DD like string (not just 202x)
    let has_date = lower
        .chars()
        .zip(lower.chars().skip(4))
        .any(|(a, b)| a.is_ascii_digit() && b == '-');
    if has_date {
        return true;
    }

    false
}

fn is_relative_temporal(lower: &str) -> bool {
    const RELATIVE: &[&str] = &[
        "last week",
        "last month",
        "this week",
        "this month",
        "yesterday",
        "recently",
        "recent",
        "上周",
        "上个月",
        "昨天",
        "最近",
    ];
    RELATIVE.iter().any(|pat| lower.contains(pat))
}

fn extract_temporal_days(lower: &str) -> i64 {
    if lower.contains("yesterday") || lower.contains("昨天") {
        return 1;
    }
    if lower.contains("last week") || lower.contains("this week") || lower.contains("上周") {
        return 7;
    }
    if lower.contains("last month") || lower.contains("this month") || lower.contains("上个月") {
        return 30;
    }
    if lower.contains("recently") || lower.contains("最近") {
        return 14;
    }
    30 // default: look back 30 days
}

fn is_exact_keyword(query: &str, lower: &str, word_count: usize) -> bool {
    // Short queries (1-3 ASCII words) without question/exploratory words.
    let is_mostly_ascii = query.len() < word_count * 15; // CJK chars are 3 bytes each

    // For CJK queries: use jieba word count instead of whitespace word count.
    // A short CJK phrase like "数据库" (1 whitespace token but 1 jieba word) is a keyword lookup,
    // while "数据库连接池怎么修复" (1 whitespace token but 5+ jieba words) is a question.
    if crate::extract::dedup::contains_cjk(query) {
        // Use jieba word count only (not bigrams) for routing decisions.
        // tokenize_for_search includes bigrams which inflate the count.
        let cjk_word_count = crate::extract::extract_keywords_from_text(query, 50).len();
        // CJK question markers
        let has_question = lower.contains('？')
            || lower.contains("什么")
            || lower.contains("怎么")
            || lower.contains("如何")
            || lower.contains("为什么")
            || lower.contains("哪里")
            || lower.contains("哪个");
        if cjk_word_count <= 3 && !has_question {
            return true;
        }
        // Don't fall through to ASCII logic for CJK queries
        return false;
    }

    if word_count <= 3
        && is_mostly_ascii
        && !lower.starts_with("what")
        && !lower.starts_with("how")
        && !lower.starts_with("why")
        && !lower.starts_with("where")
        && !lower.starts_with("summarize")
        && !lower.starts_with("tell")
        && !lower.starts_with("everything")
    {
        return true;
    }

    // Code patterns: snake_case, CamelCase, ::, .rs, fn, struct
    if query.contains("::") || query.contains(".rs") {
        return true;
    }
    if lower.starts_with("fn ") || lower.starts_with("struct ") {
        return true;
    }

    // Predominantly snake_case or CamelCase (check if most characters follow code pattern)
    if word_count <= 4 {
        let code_words = query
            .split_whitespace()
            .filter(|w| {
                w.contains('_')
                    || (w.len() > 2
                        && w.chars().any(|c| c.is_uppercase())
                        && w.chars().any(|c| c.is_lowercase()))
            })
            .count();
        if code_words > 0 && code_words >= word_count / 2 {
            return true;
        }
    }

    // Quoted terms
    if query.contains('"') || query.contains('`') {
        return true;
    }

    false
}

fn is_exploratory(lower: &str) -> bool {
    const EXPLORATORY_EN: &[&str] = &[
        "what do i know",
        "what have i",
        "everything about",
        "tell me about",
        "summarize",
        "overview",
        "all about",
        "what do we know",
    ];
    const EXPLORATORY_ZH: &[&str] = &["我知道什么", "关于", "总结", "概述", "所有"];

    for pat in EXPLORATORY_EN {
        if lower.contains(pat) {
            return true;
        }
    }
    // Chinese exploratory: only match longer patterns to avoid false positives
    for pat in EXPLORATORY_ZH {
        if lower.contains(pat) && lower.len() > pat.len() + 6 {
            return true;
        }
    }

    false
}

impl std::fmt::Display for QueryType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Episodic => write!(f, "episodic"),
            Self::Temporal => write!(f, "temporal"),
            Self::Preference => write!(f, "preference"),
            Self::ExactKeyword => write!(f, "exact"),
            Self::Semantic => write!(f, "semantic"),
            Self::Exploratory => write!(f, "exploratory"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_queries() {
        assert_eq!(
            classify("when did the API change?", false, false).query_type,
            QueryType::Temporal
        );
        assert_eq!(
            classify("what happened last week", false, false).query_type,
            QueryType::Temporal
        );
        assert_eq!(
            classify("yesterday's decisions", false, false).query_type,
            QueryType::Temporal
        );
        assert_eq!(
            classify("什么时候改了", false, false).query_type,
            QueryType::Temporal
        );
        assert_eq!(
            classify("changes since 2026-03-20", false, false).query_type,
            QueryType::Temporal
        );
    }

    #[test]
    fn test_temporal_with_existing_bounds() {
        let s = classify("recent changes", true, true);
        assert_eq!(s.query_type, QueryType::Temporal);
        assert!(!s.force_temporal);
    }

    #[test]
    fn test_temporal_days_back() {
        assert_eq!(
            classify("yesterday's work", false, false).temporal_days_back,
            Some(1)
        );
        assert_eq!(
            classify("what happened last week", false, false).temporal_days_back,
            Some(7)
        );
        assert_eq!(
            classify("last month summary", false, false).temporal_days_back,
            Some(30)
        );
    }

    #[test]
    fn test_exact_keyword() {
        assert_eq!(
            classify("ownership rules", false, false).query_type,
            QueryType::ExactKeyword
        );
        assert_eq!(
            classify("SqliteStore", false, false).query_type,
            QueryType::ExactKeyword
        );
        assert_eq!(
            classify("fn recall_temporal", false, false).query_type,
            QueryType::ExactKeyword
        );
        assert_eq!(
            classify("store::sqlite.rs", false, false).query_type,
            QueryType::ExactKeyword
        );
    }

    #[test]
    fn test_exploratory() {
        assert_eq!(
            classify("what do I know about Rust", false, false).query_type,
            QueryType::Exploratory
        );
        assert_eq!(
            classify("everything about the search pipeline", false, false).query_type,
            QueryType::Exploratory
        );
        assert_eq!(
            classify("tell me about memory management", false, false).query_type,
            QueryType::Exploratory
        );
    }

    #[test]
    fn test_semantic_default() {
        assert_eq!(
            classify(
                "how does memory management relate to performance",
                false,
                false
            )
            .query_type,
            QueryType::Semantic
        );
        assert_eq!(
            classify(
                "connection pooling strategies for high throughput",
                false,
                false
            )
            .query_type,
            QueryType::Semantic
        );
    }

    #[test]
    fn test_alpha_ordering() {
        let exact = classify("SqliteStore", false, false);
        let temporal = classify("when did X change", false, false);
        let semantic = classify(
            "things related to caching strategies and optimization",
            false,
            false,
        );
        // ExactKeyword should have highest alpha (most BM25), Semantic lowest
        assert!(exact.cc_alpha.unwrap() > temporal.cc_alpha.unwrap());
        assert!(temporal.cc_alpha.unwrap() > semantic.cc_alpha.unwrap());
    }

    #[test]
    fn test_exploratory_limit_multiplier() {
        let s = classify("what do I know about rein architecture", false, false);
        assert_eq!(s.query_type, QueryType::Exploratory);
        assert!(s.limit_multiplier > 1.5);
    }

    #[test]
    fn test_preference_queries() {
        assert_eq!(
            classify("what food do I prefer", false, false).query_type,
            QueryType::Preference
        );
        assert_eq!(
            classify("my favorite programming language", false, false).query_type,
            QueryType::Preference
        );
    }

    #[test]
    fn test_episodic_queries() {
        assert_eq!(
            classify("what did we decide about the API", false, false).query_type,
            QueryType::Episodic
        );
        assert_eq!(
            classify("recap of the last session", false, false).query_type,
            QueryType::Episodic
        );
    }

    #[test]
    fn test_episodic_vs_temporal_disambiguation() {
        // "what happened in our meeting" should be Episodic, not Temporal
        assert_eq!(
            classify("what happened in our last meeting", false, false).query_type,
            QueryType::Episodic
        );
        // "what happened last week" should still be Temporal (no session/meeting reference)
        assert_eq!(
            classify("what happened last week", false, false).query_type,
            QueryType::Temporal
        );
    }
}
