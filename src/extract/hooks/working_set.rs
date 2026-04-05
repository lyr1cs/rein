//! Project-scoped memory surfaces:
//! - working set: recent, session-biased
//! - always-on index: smaller, stabler, project-level summaries

use crate::config::ReinConfig;
use crate::extract::llm::{EpisodeSummary, ExtractedConcept, ExtractedMemory};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};


const MIN_SELECT_SCORE: f32 = 0.32;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkingSetItem {
    pub kind: String,
    pub topic: String,
    pub summary: String,
    pub detail: String,
    pub agent_label: String,
    pub is_subagent: bool,
    pub score: f32,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct WorkingSetState {
    #[serde(default)]
    items: Vec<WorkingSetItem>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct AlwaysOnState {
    #[serde(default)]
    items: Vec<WorkingSetItem>,
}

pub fn project_working_set_path(config: &ReinConfig) -> std::path::PathBuf {
    project_scoped_path(config, "working_set")
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
    agent_label: &str,
    is_subagent: bool,
) -> anyhow::Result<()> {
    let path = project_working_set_path(config);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // File lock prevents concurrent updates from overwriting each other.
    let lock_path = path.with_extension("json.lock");
    let lock_file = std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(&lock_path)?;
    let fd = std::os::fd::AsRawFd::as_raw_fd(&lock_file);
    // Blocking lock — wait for other updaters to finish. Retry on EINTR.
    loop {
        let rc = unsafe { libc::flock(fd, libc::LOCK_EX) };
        if rc == 0 {
            break;
        }
        let err = std::io::Error::last_os_error();
        if err.raw_os_error() == Some(libc::EINTR) {
            continue; // interrupted by signal, retry
        }
        tracing::warn!(
            "working_set: flock failed: {}, proceeding without lock",
            err
        );
        break;
    }

    let mut current = load_working_set(config);
    let mut incoming = build_items(memories, concepts, episode, agent_label, is_subagent);

    current.append(&mut incoming);
    let merged = merge_items(current, config.async_memory.max_working_set_items);
    let state = WorkingSetState { items: merged };
    std::fs::write(&path, serde_json::to_string_pretty(&state)?)?;

    let _ = unsafe { libc::flock(fd, libc::LOCK_UN) };
    drop(lock_file);
    Ok(())
}

pub fn project_always_on_index_path(config: &ReinConfig) -> std::path::PathBuf {
    project_scoped_path(config, "always_on_index")
}

pub fn load_always_on_index(config: &ReinConfig) -> Vec<WorkingSetItem> {
    let path = project_always_on_index_path(config);
    let Ok(text) = std::fs::read_to_string(path) else {
        return vec![];
    };
    serde_json::from_str::<AlwaysOnState>(&text)
        .map(|state| state.items)
        .unwrap_or_default()
}

pub fn update_always_on_index(
    config: &ReinConfig,
    memories: &[ExtractedMemory],
    concepts: &[ExtractedConcept],
    episode: Option<&EpisodeSummary>,
    agent_label: &str,
    is_subagent: bool,
) -> anyhow::Result<()> {
    let path = project_always_on_index_path(config);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // File lock prevents concurrent updates.
    let lock_path = path.with_extension("json.lock");
    let lock_file = std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(&lock_path)?;
    let fd = std::os::fd::AsRawFd::as_raw_fd(&lock_file);
    // Retry on EINTR
    loop {
        let rc = unsafe { libc::flock(fd, libc::LOCK_EX) };
        if rc == 0 {
            break;
        }
        let err = std::io::Error::last_os_error();
        if err.raw_os_error() == Some(libc::EINTR) {
            continue;
        }
        tracing::warn!(
            "always_on_index: flock failed: {}, proceeding without lock",
            err
        );
        break;
    }

    let mut current = load_always_on_index(config);
    let mut incoming = build_always_on_items(memories, concepts, episode, agent_label, is_subagent);
    current.append(&mut incoming);
    let merged = merge_items(current, config.async_memory.max_always_on_items);
    let state = AlwaysOnState { items: merged };
    std::fs::write(&path, serde_json::to_string_pretty(&state)?)?;

    let _ = unsafe { libc::flock(fd, libc::LOCK_UN) };
    drop(lock_file);
    Ok(())
}

pub fn select_relevant_items(config: &ReinConfig, query: &str) -> Vec<WorkingSetItem> {
    let query = query.trim();
    if query.is_empty() || looks_like_smalltalk(query) {
        return vec![];
    }

    let working = load_working_set(config).into_iter().map(|mut item| {
        item.score = item.score.max(0.1);
        item
    });
    let always_on = load_always_on_index(config).into_iter().map(|mut item| {
        item.score = item.score.max(0.2);
        item
    });

    let mut scored: Vec<(f32, WorkingSetItem)> = working
        .chain(always_on)
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
    merge_items(
        scored.into_iter().map(|(_, item)| item).collect(),
        config.async_memory.selection_limit,
    )
}

fn build_items(
    memories: &[ExtractedMemory],
    concepts: &[ExtractedConcept],
    episode: Option<&EpisodeSummary>,
    agent_label: &str,
    is_subagent: bool,
) -> Vec<WorkingSetItem> {
    let now = Utc::now();
    let mut items = Vec::new();

    for memory in memories {
        let summary = compact(&memory.summary, 120);
        let detail = compact(
            if memory.content.trim().is_empty() {
                &memory.summary
            } else {
                &memory.content
            },
            220,
        );
        if detail.is_empty() {
            continue;
        }
        items.push(WorkingSetItem {
            kind: "memory".to_string(),
            topic: memory.topic.clone(),
            summary,
            detail,
            agent_label: agent_label.to_string(),
            is_subagent,
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
            agent_label: agent_label.to_string(),
            is_subagent,
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
        let detail = compact(
            &format!("{} Outcome: {}{}", ep.title, ep.outcome, decisions),
            220,
        );
        if !detail.is_empty() {
            items.push(WorkingSetItem {
                kind: "episode".to_string(),
                topic: "session".to_string(),
                summary: compact(&ep.title, 120),
                detail,
                agent_label: agent_label.to_string(),
                is_subagent,
                score: 0.85,
                updated_at: now,
            });
        }
    }

    items
}

fn build_always_on_items(
    memories: &[ExtractedMemory],
    concepts: &[ExtractedConcept],
    episode: Option<&EpisodeSummary>,
    agent_label: &str,
    is_subagent: bool,
) -> Vec<WorkingSetItem> {
    let now = Utc::now();
    let mut items = Vec::new();

    for memory in memories {
        let topic = memory.topic.to_lowercase();
        let importance_high = matches!(
            memory.importance.to_lowercase().as_str(),
            "high" | "critical"
        );
        let stable_topic = [
            "architecture",
            "decision",
            "design",
            "workflow",
            "config",
            "user_preference",
        ]
        .iter()
        .any(|k| topic.contains(k));
        if !importance_high && !stable_topic && memory.quality_confidence < 0.7 {
            continue;
        }
        let summary = compact(&memory.summary, 110);
        let detail = compact(&memory.content, 180);
        if summary.is_empty() || detail.is_empty() {
            continue;
        }
        items.push(WorkingSetItem {
            kind: "always_on_memory".to_string(),
            topic: memory.topic.clone(),
            summary,
            detail,
            agent_label: agent_label.to_string(),
            is_subagent,
            score: memory_score(memory).max(0.7),
            updated_at: now,
        });
    }

    for concept in concepts {
        if concept.quality_confidence < 0.7 {
            continue;
        }
        let summary = compact(&concept.name, 110);
        let detail = compact(&concept.definition, 180);
        if summary.is_empty() || detail.is_empty() {
            continue;
        }
        items.push(WorkingSetItem {
            kind: "always_on_concept".to_string(),
            topic: concept.memoir.clone(),
            summary,
            detail,
            agent_label: agent_label.to_string(),
            is_subagent,
            score: (concept.quality_confidence as f32).max(0.72),
            updated_at: now,
        });
    }

    if let Some(ep) = episode {
        if !ep.decisions.is_empty() {
            items.push(WorkingSetItem {
                kind: "always_on_episode".to_string(),
                topic: "session".to_string(),
                summary: compact(&ep.title, 110),
                detail: compact(
                    &format!("{} Decisions: {}", ep.outcome, ep.decisions.join("; ")),
                    180,
                ),
                agent_label: agent_label.to_string(),
                is_subagent,
                score: 0.7,
                updated_at: now,
            });
        }
    }

    items
}

fn merge_items(mut items: Vec<WorkingSetItem>, max_items: usize) -> Vec<WorkingSetItem> {
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
                && existing.agent_label == item.agent_label
                && existing.is_subagent == item.is_subagent
                && (crate::extract::similarity(&existing.detail, &item.detail) > 0.85
                    || crate::extract::similarity(&existing.summary, &item.summary) > 0.88)
            {
                if item.score > existing.score {
                    *existing = item;
                }
                continue 'outer;
            }
        }
        merged.push(item);
        if merged.len() >= max_items {
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
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
        "你好",
        "您好",
        "哈喽",
        "嗨",
        "谢谢",
    ];
    let normalized = lower
        .chars()
        .map(|ch| {
            if ch.is_alphanumeric() || ch.is_whitespace() || ('\u{4e00}'..='\u{9fff}').contains(&ch)
            {
                ch
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    phrases.contains(&normalized.as_str())
}

fn project_scoped_path(config: &ReinConfig, prefix: &str) -> std::path::PathBuf {
    let base = super::buffer::resolve_buffer_dir(config);
    let db_tag = {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        config.resolve_db_path().hash(&mut h);
        format!("{:016x}", h.finish())
    };
    let queue_dir = base.join("queue").join(&db_tag);
    let _ = std::fs::create_dir_all(&queue_dir);
    queue_dir.join(format!("{prefix}.json"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_dedups_similar_items() {
        let now = Utc::now();
        let items = vec![
            WorkingSetItem {
                kind: "memory".into(),
                topic: "debug".into(),
                summary: "a".into(),
                detail: "fixed sqlite locking".into(),
                agent_label: "claude-code".into(),
                is_subagent: false,
                score: 0.7,
                updated_at: now,
            },
            WorkingSetItem {
                kind: "memory".into(),
                topic: "debug".into(),
                summary: "b".into(),
                detail: "fixed sqlite locking issue".into(),
                agent_label: "claude-code".into(),
                is_subagent: false,
                score: 0.9,
                updated_at: now,
            },
        ];
        let merged = merge_items(items, 40);
        assert_eq!(merged.len(), 1);
        assert!((merged[0].score - 0.9).abs() < f32::EPSILON);
    }
}
