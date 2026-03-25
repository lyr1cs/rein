//! Signal scoring and filtering for hook extraction.

use crate::config::ReinConfig;

/// Compute quality_confidence from pattern score when LLM is unavailable.
pub fn pattern_quality_confidence(text: &str) -> f64 {
    let score = crate::extract::patterns::score_sentence(text);
    match score {
        0 => 0.1,
        1..=2 => 0.3,
        3..=4 => 0.5,
        _ => 0.8,
    }
}

/// Quick local check: does this text likely contain anything worth storing?
pub fn worth_extracting(text: &str) -> bool {
    if text.len() < 80 { return false; }

    let dominated_by_code = text.matches("```").count() >= 2
        || text.matches("---").count() >= 3
        || text.contains("Assistant:")
        || text.starts_with("let ")
        || text.starts_with("fn ")
        || text.starts_with("pub ")
        || text.starts_with("use ")
        || text.starts_with("impl ");
    if dominated_by_code { return false; }

    let score = crate::extract::patterns::score_sentence(text);
    if score >= 3 { return true; }

    let lower = text.to_lowercase();
    let value_signals = [
        "because", "reason", "instead of", "switched to",
        "root cause", "workaround", "decided",
        "chose", "selected", "prefer",
        "因为", "原因", "切换到", "决定",
    ];
    value_signals.iter().any(|s| lower.contains(s))
}

/// Extract context windows around signal keywords from transcript text.
pub fn extract_signal_windows(text: &str, config: &ReinConfig) -> Vec<String> {
    let context_before = config.hooks.context_before;
    let context_after = config.hooks.context_after;
    let signal_keywords = &config.hooks.signal_keywords;

    let lines: Vec<&str> = text.lines().collect();
    let mut hit_ranges: Vec<(usize, usize)> = Vec::new();

    for (i, line) in lines.iter().enumerate() {
        let lower = line.to_lowercase();
        if signal_keywords.iter().any(|kw| lower.contains(kw.as_str())) && line.len() > 15 {
            let start = i.saturating_sub(context_before);
            let end = (i + context_after + 1).min(lines.len());
            hit_ranges.push((start, end));
        }
    }

    let merged = merge_ranges(&hit_ranges);
    merged.iter()
        .map(|(start, end)| lines[*start..*end].join("\n"))
        .collect()
}

fn merge_ranges(ranges: &[(usize, usize)]) -> Vec<(usize, usize)> {
    if ranges.is_empty() { return vec![]; }
    let mut sorted = ranges.to_vec();
    sorted.sort_by_key(|r| r.0);
    let mut merged = vec![sorted[0]];
    for &(start, end) in &sorted[1..] {
        let last = merged.last_mut().unwrap();
        if start <= last.1 {
            last.1 = last.1.max(end);
        } else {
            merged.push((start, end));
        }
    }
    merged
}
