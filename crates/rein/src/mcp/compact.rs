use crate::search::recall::RecallResult;
use crate::types::{HealthReport, StoreStats};

const MAX_MCP_RECALL_OUTPUT_CHARS: usize = 12_000;
const MAX_RECALL_SUMMARY_CHARS: usize = 120;
const MAX_RECALL_CONTENT_CHARS: usize = 360;
const MAX_RECALL_EVIDENCE_CHARS: usize = 120;
const MAX_RECALL_EVIDENCE_PREVIEW_ITEMS: usize = 2;

pub fn format_store_result(id: &str, compact: bool) -> String {
    if compact {
        format!("ok:{id}")
    } else {
        format!("Stored memory with ID: {id}")
    }
}

pub fn format_recall_results(results: &[RecallResult], compact: bool) -> String {
    format_recall_results_with_budget(results, compact, None)
}

pub fn format_recall_results_mcp(results: &[RecallResult], compact: bool) -> String {
    format_recall_results_with_budget(results, compact, Some(MAX_MCP_RECALL_OUTPUT_CHARS))
}

fn format_recall_results_with_budget(
    results: &[RecallResult],
    compact: bool,
    max_chars: Option<usize>,
) -> String {
    if compact {
        let entries = results
            .iter()
            .map(|r| {
                format!(
                    "[{}] {}",
                    r.memory.topic,
                    truncate_chars(&r.memory.summary, MAX_RECALL_SUMMARY_CHARS)
                )
            })
            .collect::<Vec<_>>();
        join_entries_with_budget(&entries, "\n", max_chars)
    } else {
        let entries = results
            .iter()
            .map(|r| {
                let m = &r.memory;
                let evidence_block = if r.evidence_preview.is_empty() {
                    format!("  evidence_count: {}", r.evidence_count)
                } else {
                    format!(
                        "  evidence_count: {}\n  evidence_preview: {}",
                        r.evidence_count,
                        r.evidence_preview
                            .iter()
                            .take(MAX_RECALL_EVIDENCE_PREVIEW_ITEMS)
                            .map(|preview| truncate_chars(preview, MAX_RECALL_EVIDENCE_CHARS))
                            .collect::<Vec<_>>()
                            .join(" | ")
                    )
                };
                format!(
                    "--- [{topic}] {summary} ---\n{content}\n  score: {score:.3}\n  confidence: {confidence:.3}\n  importance: {imp}\n  strength: {str:.3}\n  support: {support}\n  diversity: {diversity:.2}\n{evidence_block}\n  keywords: {kw}\n  id: {id}",
                    topic = m.topic,
                    summary = truncate_chars(&m.summary, MAX_RECALL_SUMMARY_CHARS),
                    content = truncate_chars(&m.content, MAX_RECALL_CONTENT_CHARS),
                    score = r.score,
                    confidence = r.confidence,
                    imp = m.importance,
                    str = m.strength,
                    support = m.support_count,
                    diversity = m.source_diversity,
                    evidence_block = evidence_block,
                    kw = if m.keywords.is_empty() { "(none)".to_string() } else { m.keywords.join(", ") },
                    id = m.id,
                )
            })
            .collect::<Vec<_>>();
        join_entries_with_budget(&entries, "\n\n", max_chars)
    }
}

fn join_entries_with_budget(
    entries: &[String],
    separator: &str,
    max_chars: Option<usize>,
) -> String {
    let Some(max_chars) = max_chars else {
        return entries.join(separator);
    };

    let mut rendered = Vec::new();
    let mut used_chars = 0usize;
    let separator_chars = separator.chars().count();

    for (idx, entry) in entries.iter().enumerate() {
        let entry_chars = entry.chars().count();
        let sep_chars = if rendered.is_empty() {
            0
        } else {
            separator_chars
        };
        let remaining = entries.len().saturating_sub(idx + 1);
        let omitted_note = if remaining > 0 {
            format!(
                "\n\n... {} more results omitted to stay within the MCP output limit. Narrow the query or lower `limit` for the full set.",
                remaining
            )
        } else {
            String::new()
        };
        let omitted_note_chars = omitted_note.chars().count();

        if used_chars + sep_chars + entry_chars + omitted_note_chars > max_chars {
            if rendered.is_empty() {
                let available = max_chars.saturating_sub(omitted_note_chars);
                if available > 0 {
                    rendered.push(truncate_chars(entry, available));
                }
            }
            if !omitted_note.is_empty() {
                rendered.push(omitted_note.trim_start_matches('\n').to_string());
            }
            break;
        }

        if sep_chars > 0 {
            used_chars += sep_chars;
        }
        used_chars += entry_chars;
        rendered.push(entry.clone());
    }

    rendered.join(separator)
}

fn truncate_chars(input: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    if input.chars().count() <= max_chars {
        return input.to_string();
    }

    let keep = max_chars.saturating_sub(3).max(1);
    let mut truncated = input.chars().take(keep).collect::<String>();
    if max_chars > 3 {
        truncated.push_str("...");
    }
    truncated
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Importance, Memory, MemoryLayer, MemoryStatus, MemoryTier, Source};
    use chrono::Utc;

    fn recall_result(id: &str, content: &str) -> RecallResult {
        RecallResult {
            memory: Memory {
                id: id.to_string(),
                layer: MemoryLayer::LTM,
                topic: "rein".to_string(),
                summary: "A very long recall summary for testing formatter truncation behavior"
                    .to_string(),
                content: content.to_string(),
                keywords: vec!["memory".to_string(), "recall".to_string()],
                importance: Importance::High,
                source: Source::Manual,
                strength: 0.8,
                decay_lambda: 0.02,
                access_count: 0,
                superseded_by: None,
                canonical_id: None,
                support_count: 3,
                merge_count: 0,
                dedup_confidence: 1.0,
                source_diversity: 2.0,
                contradiction_score: 0.0,
                related_ids: vec![],
                concept_ids: vec![],
                status: MemoryStatus::Active,
                embedding: None,
                tier: MemoryTier::Warm,
                cluster_id: Some(7),
                created_at: Utc::now(),
                updated_at: Utc::now(),
                last_accessed: Utc::now(),
            },
            score: 0.91,
            confidence: 0.88,
            sources_hit: 3,
            evidence_count: 4,
            evidence_preview: vec![
                "evidence preview one that should also be clipped for MCP output".to_string(),
                "evidence preview two that is deliberately verbose to exercise the cap".to_string(),
                "extra preview omitted".to_string(),
            ],
        }
    }

    #[test]
    fn mcp_recall_output_truncates_large_payloads() {
        let results = (0..40)
            .map(|idx| recall_result(&format!("id-{idx}"), &"x".repeat(2_000)))
            .collect::<Vec<_>>();

        let output = format_recall_results_mcp(&results, false);

        assert!(output.contains("more results omitted"));
        assert!(output.chars().count() <= MAX_MCP_RECALL_OUTPUT_CHARS + 200);
        assert!(!output.contains(&"x".repeat(500)));
    }
}
