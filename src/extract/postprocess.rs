//! Rule-based post-processing for extracted memories.
//!
//! Applied after LLM extraction (or pattern fallback) but before admission control.
//! Enriches memories with date keywords, preference markers, and knowledge-update signals.

use super::llm::ExtractedMemory;

/// Extract ISO dates (YYYY-MM-DD) and informal English dates from text.
/// Returns keywords in the form `date:YYYY-MM-DD`.
pub fn extract_date_keywords(content: &str) -> Vec<String> {
    let mut dates = Vec::new();

    // ISO format: YYYY-MM-DD
    let bytes = content.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    while i + 9 < len {
        // Look for pattern: 4 digits, dash, 2 digits, dash, 2 digits
        if bytes[i].is_ascii_digit()
            && bytes[i + 1].is_ascii_digit()
            && bytes[i + 2].is_ascii_digit()
            && bytes[i + 3].is_ascii_digit()
            && bytes[i + 4] == b'-'
            && bytes[i + 5].is_ascii_digit()
            && bytes[i + 6].is_ascii_digit()
            && bytes[i + 7] == b'-'
            && bytes[i + 8].is_ascii_digit()
            && bytes[i + 9].is_ascii_digit()
        {
            let s = &content[i..i + 10];
            // Basic validation: month 01-12, day 01-31
            let month: u32 = s[5..7].parse().unwrap_or(0);
            let day: u32 = s[8..10].parse().unwrap_or(0);
            if (1..=12).contains(&month) && (1..=31).contains(&day) {
                let kw = format!("date:{}", s);
                if !dates.contains(&kw) {
                    dates.push(kw);
                }
            }
            i += 10;
        } else {
            i += 1;
        }
    }

    // Informal English dates: "March 15th", "Jan 3rd", "December 1st", "Feb 22nd", etc.
    let month_map: &[(&[&str], &str)] = &[
        (&["january", "jan"], "01"),
        (&["february", "feb"], "02"),
        (&["march", "mar"], "03"),
        (&["april", "apr"], "04"),
        (&["may"], "05"),
        (&["june", "jun"], "06"),
        (&["july", "jul"], "07"),
        (&["august", "aug"], "08"),
        (&["september", "sep"], "09"),
        (&["october", "oct"], "10"),
        (&["november", "nov"], "11"),
        (&["december", "dec"], "12"),
    ];

    let lower = content.to_lowercase();
    for (names, mm) in month_map {
        for name in *names {
            let mut search_from = 0;
            while let Some(pos) = lower[search_from..].find(name) {
                let abs_pos = search_from + pos;
                search_from = abs_pos + name.len();

                // After month name, expect optional space then 1-2 digit day, optional suffix
                let rest = &lower[search_from..];
                let rest = rest.trim_start();

                // Parse day number
                let mut day_digits = String::new();
                for ch in rest.chars() {
                    if ch.is_ascii_digit() {
                        day_digits.push(ch);
                        if day_digits.len() == 2 {
                            break;
                        }
                    } else {
                        break;
                    }
                }

                if day_digits.is_empty() {
                    continue;
                }

                let day: u32 = day_digits.parse().unwrap_or(0);
                if !(1..=31).contains(&day) {
                    continue;
                }

                // Try to find a year nearby (within ~20 chars after day)
                let after_day = &rest[day_digits.len()..];
                // Skip optional suffix (st, nd, rd, th) and punctuation
                let after_suffix = after_day
                    .trim_start_matches("st")
                    .trim_start_matches("nd")
                    .trim_start_matches("rd")
                    .trim_start_matches("th")
                    .trim_start_matches(',')
                    .trim_start();

                let year = extract_nearby_year(after_suffix).unwrap_or(current_year());

                let kw = format!("date:{}-{}-{:02}", year, mm, day);
                if !dates.contains(&kw) {
                    dates.push(kw);
                }
            }
        }
    }

    dates
}

/// Try to extract a 4-digit year from the start of a string.
fn extract_nearby_year(s: &str) -> Option<u32> {
    let mut digits = String::new();
    for ch in s.chars() {
        if ch.is_ascii_digit() {
            digits.push(ch);
            if digits.len() == 4 {
                break;
            }
        } else if !digits.is_empty() {
            break;
        } else {
            // skip leading non-digit chars but only a few
            continue;
        }
    }
    if digits.len() == 4 {
        let y: u32 = digits.parse().ok()?;
        if (1900..=2100).contains(&y) {
            return Some(y);
        }
    }
    None
}

/// Get current year as fallback for informal dates without year.
fn current_year() -> u32 {
    chrono::Utc::now()
        .format("%Y")
        .to_string()
        .parse()
        .unwrap_or(2026)
}

/// Detect if content expresses a user preference.
/// Checks for common preference keywords in English and Chinese.
pub fn is_preference_content(content: &str) -> bool {
    let lower = content.to_lowercase();
    let en_keywords = [
        "i prefer",
        "i like",
        "i love",
        "my favorite",
        "my favourite",
        "i usually",
        "i always",
        "i tend to",
        "i often",
        "prefer to",
        "rather than",
        "instead of using",
        "i enjoy",
        "i favor",
        "i favour",
    ];
    let zh_keywords = [
        "我喜欢",
        "我偏好",
        "我习惯",
        "我倾向",
        "我更喜欢",
        "我通常",
        "我总是",
        "比较喜欢",
        "更倾向于",
        "偏爱",
    ];

    for kw in &en_keywords {
        if lower.contains(kw) {
            return true;
        }
    }
    for kw in &zh_keywords {
        if content.contains(kw) {
            return true;
        }
    }
    false
}

/// Detect if content describes a knowledge update (something changed/replaced).
pub fn is_knowledge_update(content: &str) -> bool {
    let lower = content.to_lowercase();
    let patterns = [
        "changed to",
        "switched to",
        "moved to",
        "migrated to",
        "replaced with",
        "upgraded to",
        "downgraded to",
        "updated to",
        "no longer",
        "instead of",
        "now using",
        "now uses",
        "deprecated",
        "replaced by",
        "superseded by",
        "was renamed to",
        "renamed to",
    ];
    let zh_patterns = [
        "改为",
        "换成",
        "迁移到",
        "升级到",
        "不再使用",
        "替换为",
        "现在使用",
        "已弃用",
        "已废弃",
    ];

    for p in &patterns {
        if lower.contains(p) {
            return true;
        }
    }
    for p in &zh_patterns {
        if content.contains(p) {
            return true;
        }
    }
    false
}

/// Apply all post-processing rules to an extracted memory.
///
/// 1. Inject date keywords extracted from content
/// 2. Mark preferences with topic + keyword
/// 3. Mark knowledge updates with keyword + importance boost
pub fn postprocess(memory: &mut ExtractedMemory) {
    // 1. Date keywords
    let date_kws = extract_date_keywords(&memory.content);
    for kw in date_kws {
        if !memory.keywords.contains(&kw) {
            memory.keywords.push(kw);
        }
    }

    // 2. Preference detection
    if is_preference_content(&memory.content) {
        memory.topic = "user_preference".to_string();
        if !memory.keywords.contains(&"preference".to_string()) {
            memory.keywords.push("preference".to_string());
        }
    }

    // 3. Knowledge update detection
    if is_knowledge_update(&memory.content) {
        if !memory.keywords.contains(&"knowledge_update".to_string()) {
            memory.keywords.push("knowledge_update".to_string());
        }
        // Boost importance: low→medium, medium→high
        match memory.importance.as_str() {
            "low" => memory.importance = "medium".to_string(),
            "medium" => memory.importance = "high".to_string(),
            // high and critical stay as-is
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_memory(content: &str) -> ExtractedMemory {
        ExtractedMemory {
            topic: "general".to_string(),
            summary: content.chars().take(50).collect(),
            content: content.to_string(),
            keywords: vec![],
            importance: "medium".to_string(),
            should_store: true,
            quality_confidence: 0.7,
        }
    }

    // --- Date extraction tests ---

    #[test]
    fn test_extract_iso_date() {
        let dates = extract_date_keywords("Deployed on 2026-03-15 successfully");
        assert_eq!(dates, vec!["date:2026-03-15"]);
    }

    #[test]
    fn test_extract_multiple_iso_dates() {
        let dates = extract_date_keywords("From 2026-01-10 to 2026-02-20");
        assert_eq!(dates.len(), 2);
        assert!(dates.contains(&"date:2026-01-10".to_string()));
        assert!(dates.contains(&"date:2026-02-20".to_string()));
    }

    #[test]
    fn test_extract_informal_date_with_year() {
        let dates = extract_date_keywords("Meeting on March 15th, 2026 was productive");
        assert_eq!(dates, vec!["date:2026-03-15"]);
    }

    #[test]
    fn test_extract_informal_date_short_month() {
        let dates = extract_date_keywords("Jan 3rd release");
        assert_eq!(dates.len(), 1);
        assert!(dates[0].starts_with("date:"));
        assert!(dates[0].ends_with("-01-03"));
    }

    #[test]
    fn test_extract_no_dates() {
        let dates = extract_date_keywords("No dates here at all");
        assert!(dates.is_empty());
    }

    #[test]
    fn test_invalid_iso_date_rejected() {
        let dates = extract_date_keywords("Bad date: 2026-13-40");
        assert!(dates.is_empty());
    }

    // --- Preference detection tests ---

    #[test]
    fn test_preference_english() {
        assert!(is_preference_content("I prefer using Vim over Emacs"));
        assert!(is_preference_content("My favorite editor is VSCode"));
        assert!(is_preference_content("I usually run tests first"));
    }

    #[test]
    fn test_preference_chinese() {
        assert!(is_preference_content("我喜欢用 Rust 写代码"));
        assert!(is_preference_content("我偏好暗色主题"));
    }

    #[test]
    fn test_not_preference() {
        assert!(!is_preference_content("The system uses SQLite for storage"));
        assert!(!is_preference_content("Deployed to production"));
    }

    // --- Knowledge update detection tests ---

    #[test]
    fn test_knowledge_update_english() {
        assert!(is_knowledge_update("We switched to PostgreSQL"));
        assert!(is_knowledge_update("The API was deprecated in v2"));
        assert!(is_knowledge_update("Now using async runtime"));
        assert!(is_knowledge_update("Migrated to the new cluster"));
    }

    #[test]
    fn test_knowledge_update_chinese() {
        assert!(is_knowledge_update("数据库已迁移到新集群"));
        assert!(is_knowledge_update("改为使用新的 API"));
    }

    #[test]
    fn test_not_knowledge_update() {
        assert!(!is_knowledge_update("The system uses SQLite"));
        assert!(!is_knowledge_update("Built with Rust"));
    }

    // --- Postprocess integration tests ---

    #[test]
    fn test_postprocess_injects_date_keywords() {
        let mut mem = make_memory("Released version 2.0 on 2026-03-15");
        postprocess(&mut mem);
        assert!(mem.keywords.contains(&"date:2026-03-15".to_string()));
    }

    #[test]
    fn test_postprocess_sets_preference() {
        let mut mem = make_memory("I prefer dark mode for all editors");
        postprocess(&mut mem);
        assert_eq!(mem.topic, "user_preference");
        assert!(mem.keywords.contains(&"preference".to_string()));
    }

    #[test]
    fn test_postprocess_sets_knowledge_update() {
        let mut mem = make_memory("We switched to PostgreSQL for the new service");
        postprocess(&mut mem);
        assert!(mem.keywords.contains(&"knowledge_update".to_string()));
        assert_eq!(mem.importance, "high"); // boosted from medium
    }

    #[test]
    fn test_postprocess_importance_boost_low_to_medium() {
        let mut mem = make_memory("Changed to a new deployment pipeline");
        mem.importance = "low".to_string();
        postprocess(&mut mem);
        assert_eq!(mem.importance, "medium");
    }

    #[test]
    fn test_postprocess_importance_no_double_boost() {
        let mut mem = make_memory("Upgraded to the latest framework");
        mem.importance = "critical".to_string();
        postprocess(&mut mem);
        assert_eq!(mem.importance, "critical"); // stays critical
    }

    #[test]
    fn test_postprocess_combined() {
        let mut mem =
            make_memory("On 2026-01-15 I switched to Neovim because I prefer modal editing");
        postprocess(&mut mem);
        assert!(mem.keywords.contains(&"date:2026-01-15".to_string()));
        assert!(mem.keywords.contains(&"knowledge_update".to_string()));
        // Preference wins for topic
        assert_eq!(mem.topic, "user_preference");
        assert!(mem.keywords.contains(&"preference".to_string()));
    }

    #[test]
    fn test_postprocess_no_duplicate_keywords() {
        let mut mem = make_memory("Switched to new system");
        mem.keywords.push("knowledge_update".to_string());
        postprocess(&mut mem);
        let count = mem
            .keywords
            .iter()
            .filter(|k| k.as_str() == "knowledge_update")
            .count();
        assert_eq!(count, 1);
    }
}
