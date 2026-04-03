//! Project-scoped working set derived from recent compact/stop extractions.

use crate::config::ReinConfig;
use crate::extract::llm::{EpisodeSummary, ExtractedConcept, ExtractedMemory};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const MAX_ITEMS: usize = 40;
const DEFAULT_SELECT_LIMIT: usize = 5;
const MIN_SELECT_SCORE: f32 = 0.32;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkingSetItem {
    pub kind: String,
    pub topic: String,
    pub summary: String,
    pub detail: String,
    pub score: f32,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct WorkingSetState {
    #[serde(default)]
    items: Vec<WorkingSetItem>,
}

pub fn project_working_set_path(config: &ReinConfig) -> std::path::PathBuf {
    let cwd = std::env::current_dir()
        .ok()
        .and_then(|p| p.canonicalize().ok())
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    let mut hasher = Sha256::new();
    hasher.update(cwd.to_string_lossy().as_bytes());
    let digest = format!("{:x}", hasher.finalize());
    super::buffer::resolve_buffer_dir(config).join(format!("working_set_{}.json", &digest[..12]))
}

pub fn load_working_set(config: &ReinConfig) -> Vec<WorkingSetItem> {
    let path = project_working_set_path(config);
    let Ok(text) = std::fs::read_to_string(path) else {
        return vec![];
    };
    serde_json::from_str::<WorkingSetState>(&text)
        .map(|state| state.items)
        .unwrap_or_default()
}

pub fn update_working_set(
    config: &ReinConfig,
    memories: &[ExtractedMemory],
    concepts: &[ExtractedConcept],
    episode: Option<&EpisodeSummary>,
) -> anyhow::Result<()> {
    let path = project_working_set_path(config);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut current = load_working_set(config);
    let mut incoming = build_items(memories, concepts, episode);

    current.append(&mut incoming);
    let merged = merge_items(current);
    let state = WorkingSetState { items: merged };
    std::fs::write(path, serde_json::to_string_pretty(&state)?)?;
    Ok(())
}

pub fn select_relevant_items(config: &ReinConfig, query: &str) -> Vec<WorkingSetItem> {
    let query = query.trim();
    if query.is_empty() || looks_like_smalltalk(query) {
        return vec![];
    }

    let mut scored: Vec<(f32, WorkingSetItem)> = load_working_set(config)
        .into_iter()
        .map(|item| {
            let sim = crate::extract::similarity(query, &item.detail)
                .max(crate::extract::similarity(query, &item.summary))
                .max(crate::extract::similarity(query, &item.topic));
            ((sim * 0.7) + (item.score * 0.3), item)
        })
        .filter(|(score, _)| *score >= MIN_SELECT_SCORE)
        .collect();

    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.1.updated_at.cmp(&a.1.updated_at))
    });
    scored
        .into_iter()
        .take(DEFAULT_SELECT_LIMIT)
        .map(|(_, item)| item)
        .collect()
}

fn build_items(
    memories: &[ExtractedMemory],
    concepts: &[ExtractedConcept],
    episode: Option<&EpisodeSummary>,
) -> Vec<WorkingSetItem> {
    let now = Utc::now();
    let mut items = Vec::new();

    for memory in memories {
        let summary = compact(&memory.summary, 120);
        let detail = compact(if memory.content.trim().is_empty() {
            &memory.summary
        } else {
            &memory.content
        }, 220);
        if detail.is_empty() {
            continue;
        }
        items.push(WorkingSetItem {
            kind: "memory".to_string(),
            topic: memory.topic.clone(),
            summary,
            detail,
            score: memory_score(memory),
            updated_at: now,
        });
    }

    for concept in concepts {
        let detail = compact(&concept.definition, 220);
        if detail.is_empty() {
            continue;
        }
        items.push(WorkingSetItem {
            kind: "concept".to_string(),
            topic: concept.memoir.clone(),
            summary: compact(&concept.name, 120),
            detail,
            score: concept.quality_confidence as f32 * 0.8 + 0.1,
            updated_at: now,
        });
    }

    if let Some(ep) = episode {
        let decisions = if ep.decisions.is_empty() {
            String::new()
        } else {
            format!(" Decisions: {}", ep.decisions.join("; "))
        };
        let detail = compact(&format!("{} Outcome: {}{}", ep.title, ep.outcome, decisions), 220);
        if !detail.is_empty() {
            items.push(WorkingSetItem {
                kind: "episode".to_string(),
                topic: "session".to_string(),
                summary: compact(&ep.title, 120),
                detail,
                score: 0.85,
                updated_at: now,
            });
        }
    }

    items
}

fn merge_items(mut items: Vec<WorkingSetItem>) -> Vec<WorkingSetItem> {
    items.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.updated_at.cmp(&a.updated_at))
    });

    let mut merged: Vec<WorkingSetItem> = Vec::new();
    'outer: for item in items {
        for existing in &mut merged {
            if existing.kind == item.kind
                && crate::extract::similarity(&existing.detail, &item.detail) > 0.85
            {
                if item.score > existing.score {
                    *existing = item;
                }
                continue 'outer;
            }
        }
        merged.push(item);
        if merged.len() >= MAX_ITEMS {
            break;
        }
    }
    merged
}

fn memory_score(memory: &ExtractedMemory) -> f32 {
    let importance = match memory.importance.to_lowercase().as_str() {
        "critical" => 1.0,
        "high" => 0.9,
        "medium" => 0.7,
        "low" => 0.5,
        _ => 0.6,
    };
    (importance * 0.6) + (memory.quality_confidence as f32 * 0.4)
}

fn compact(text: &str, max_chars: usize) -> String {
    let trimmed = text.trim();
    let mut out: String = trimmed.chars().take(max_chars).collect();
    if trimmed.chars().count() > max_chars {
        out.push_str("...");
    }
    out
}

fn looks_like_smalltalk(text: &str) -> bool {
    let lower = text.to_lowercase();
    let phrases = [
        "hi", "hello", "hey", "thanks", "thank you", "你好", "您好", "哈喽", "嗨", "谢谢",
    ];
    let normalized = lower
        .chars()
        .map(|ch| if ch.is_alphanumeric() || ch.is_whitespace() || ('\u{4e00}'..='\u{9fff}').contains(&ch) { ch } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    phrases.contains(&normalized.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_dedups_similar_items() {
        let now = Utc::now();
        let items = vec![
            WorkingSetItem { kind: "memory".into(), topic: "debug".into(), summary: "a".into(), detail: "fixed sqlite locking".into(), score: 0.7, updated_at: now },
            WorkingSetItem { kind: "memory".into(), topic: "debug".into(), summary: "b".into(), detail: "fixed sqlite locking issue".into(), score: 0.9, updated_at: now },
        ];
        let merged = merge_items(items);
        assert_eq!(merged.len(), 1);
        assert!((merged[0].score - 0.9).abs() < f32::EPSILON);
    }
}
