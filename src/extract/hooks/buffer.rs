//! Session buffer I/O for hook_post → hook_stop pipeline.

use crate::config::ReinConfig;

/// Resolve the buffer directory (auto = ~/.rein/).
pub fn resolve_buffer_dir(config: &ReinConfig) -> std::path::PathBuf {
    if config.hooks.buffer_dir == "auto" {
        config.resolve_db_path().parent()
            .unwrap_or(std::path::Path::new("/tmp"))
            .to_path_buf()
    } else {
        std::path::PathBuf::from(&config.hooks.buffer_dir)
    }
}

/// Derive a session-scoped buffer file path from the hook input.
pub fn session_buffer_path(config: &ReinConfig, input: &str) -> std::path::PathBuf {
    let session_id = if let Ok(json) = serde_json::from_str::<serde_json::Value>(input) {
        if let Some(path) = json.get("transcript_path").and_then(|v| v.as_str()) {
            use sha2::{Sha256, Digest};
            let hash = Sha256::digest(path.as_bytes());
            format!("{:x}", hash).chars().take(12).collect()
        } else {
            format!("pid{}", std::process::id())
        }
    } else {
        format!("pid{}", std::process::id())
    };
    resolve_buffer_dir(config).join(format!("buffer_{session_id}.jsonl"))
}

/// Append a text entry to the session buffer file.
pub fn append_to_buffer(path: &std::path::Path, text: &str, source: &str) -> anyhow::Result<()> {
    use std::io::Write;
    let entry = serde_json::json!({
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "text": text,
        "source": source,
    });
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(path)?;
    writeln!(file, "{}", entry)?;
    Ok(())
}

/// Read all text entries from a buffer file and delete it.
pub fn read_and_clear_buffer(path: &std::path::Path) -> Vec<String> {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return vec![],
    };
    let _ = std::fs::remove_file(path);
    content.lines()
        .filter_map(|line| {
            serde_json::from_str::<serde_json::Value>(line).ok()
                .and_then(|v| v.get("text").and_then(|t| t.as_str()).map(|s| s.to_string()))
        })
        .collect()
}

/// Adaptive flush threshold: adjusts based on signal density in the buffer.
pub fn adaptive_flush_threshold(base: usize, buf_path: &std::path::Path) -> usize {
    let content = match std::fs::read_to_string(buf_path) {
        Ok(c) => c,
        Err(_) => return base,
    };
    let total_lines = content.lines().count();
    if total_lines < 5 { return base; }

    let high_signal = content.lines()
        .filter_map(|line| {
            serde_json::from_str::<serde_json::Value>(line).ok()
                .and_then(|v| v.get("text").and_then(|t| t.as_str()).map(|s| s.to_string()))
        })
        .filter(|text| super::scoring::worth_extracting(text))
        .count();

    let density = high_signal as f64 / total_lines as f64;
    if density > 0.5 { base / 2 }
    else if density < 0.1 { base * 2 }
    else { base }
}

/// Clean up stale buffer files older than 24 hours.
pub fn cleanup_stale_buffers(config: &ReinConfig) {
    let buf_dir = resolve_buffer_dir(config);
    let pattern = buf_dir.join("buffer_*.jsonl");
    if let Ok(entries) = glob::glob(&pattern.to_string_lossy()) {
        let cutoff = chrono::Utc::now() - chrono::Duration::hours(24);
        for entry in entries.flatten() {
            if let Ok(meta) = std::fs::metadata(&entry) {
                if let Ok(modified) = meta.modified() {
                    let modified_utc: chrono::DateTime<chrono::Utc> = modified.into();
                    if modified_utc < cutoff {
                        tracing::info!("cleaning stale buffer: {}", entry.display());
                        let _ = std::fs::remove_file(&entry);
                    }
                }
            }
        }
    }
}

/// Store an episode summary as a concept in the "sessions" memoir.
pub fn store_episode_concept(
    store: &crate::store::SqliteStore,
    episode: &crate::extract::llm::EpisodeSummary,
) -> crate::types::ReinResult<()> {
    if store.get_memoir("sessions")?.is_none() {
        let memoir = crate::types::Memoir {
            id: String::new(),
            name: "sessions".to_string(),
            description: "Auto-created session episode summaries".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };
        store.create_memoir(memoir)?;
    }
    let date = format!("{}-{}", chrono::Utc::now().format("%Y-%m-%d-%H%M"),
        ulid::Ulid::new().to_string().chars().take(6).collect::<String>());
    let definition = if episode.decisions.is_empty() {
        format!("{}\nOutcome: {}", episode.title, episode.outcome)
    } else {
        format!("{}\nOutcome: {}\nDecisions: {}", episode.title, episode.outcome, episode.decisions.join("; "))
    };
    let concept = crate::types::Concept {
        id: String::new(),
        memoir_id: "sessions".to_string(),
        name: format!("session-{}", date),
        definition,
        labels: vec!["episode".to_string()],
        source_memory_ids: vec![],
        confidence: 0.8,
        revision: 1,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
    };
    store.add_concept(concept)?;
    Ok(())
}
