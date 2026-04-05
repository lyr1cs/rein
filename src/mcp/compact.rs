use crate::types::{HealthReport, Memory, StoreStats};

pub fn format_store_result(id: &str, compact: bool) -> String {
    if compact {
        format!("ok:{id}")
    } else {
        format!("Stored memory with ID: {id}")
    }
}

pub fn format_recall_results(results: &[(Memory, f32)], compact: bool) -> String {
    if compact {
        results
            .iter()
            .map(|(m, _)| format!("[{}] {}", m.topic, m.summary))
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        results
            .iter()
            .map(|(m, score)| {
                format!(
                    "--- [{topic}] {summary} ---\n{content}\n  score: {score:.3}\n  importance: {imp}\n  strength: {str:.3}\n  keywords: {kw}\n  id: {id}",
                    topic = m.topic,
                    summary = m.summary,
                    content = m.content,
                    imp = m.importance,
                    str = m.strength,
                    kw = if m.keywords.is_empty() { "(none)".to_string() } else { m.keywords.join(", ") },
                    id = m.id,
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

pub fn format_topics(topics: &[String], compact: bool) -> String {
    if compact {
        topics.join(",")
    } else {
        if topics.is_empty() {
            return "No topics found.".to_string();
        }
        topics
            .iter()
            .enumerate()
            .map(|(i, t)| format!("  {}. {}", i + 1, t))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

pub fn format_stats(stats: &StoreStats, compact: bool) -> String {
    if compact {
        format!(
            "total:{} ltm:{} stm:{} topics:{} str:{:.3} memoirs:{} concepts:{} links:{}",
            stats.total_memories,
            stats.ltm_count,
            stats.stm_count,
            stats.topic_count,
            stats.avg_strength,
            stats.memoir_count,
            stats.concept_count,
            stats.link_count
        )
    } else {
        format!(
            "Memory Store Statistics:\n  Total memories: {}\n  LTM: {}\n  STM: {}\n  Topics: {}\n  Avg strength: {:.3}\n  Memoirs: {}\n  Concepts: {}\n  Links: {}",
            stats.total_memories, stats.ltm_count, stats.stm_count, stats.topic_count,
            stats.avg_strength, stats.memoir_count, stats.concept_count, stats.link_count
        )
    }
}

pub fn format_health(reports: &[HealthReport], compact: bool) -> String {
    if compact {
        reports
            .iter()
            .map(|r| {
                format!(
                    "{}:cnt={},avg={:.2},stale={},consolidate={}",
                    r.topic, r.count, r.avg_strength, r.stale_count, r.needs_consolidation
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        if reports.is_empty() {
            return "No topics to report on.".to_string();
        }
        reports
            .iter()
            .map(|r| {
                format!(
                    "Topic: {}\n  Memories: {}\n  Avg strength: {:.3}\n  Stale: {}\n  Needs consolidation: {}",
                    r.topic, r.count, r.avg_strength, r.stale_count,
                    if r.needs_consolidation { "yes" } else { "no" }
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}
