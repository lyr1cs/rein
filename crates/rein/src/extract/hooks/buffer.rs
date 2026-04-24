//! Session buffer I/O for hook_post → hook_stop pipeline.

use crate::config::ReinConfig;
use crate::extract::hooks::parsing::redact_secrets;

/// Hard cap on per-session buffer file size. Above this, appends drop (and
/// the event is marked in tracing) instead of growing without bound. A
/// misbehaving client tool can easily dump hundreds of MB of output per
/// PostToolUse; without the cap, a single long session could OOM the hook
/// process when `read_and_clear_buffer` pulls the whole file into memory.
///
/// 16 MiB is ~20× the largest realistic session buffer we've seen, so
/// legitimate use never trips the cap.
const MAX_BUFFER_BYTES: u64 = 16 * 1024 * 1024;

/// Resolve the buffer directory (auto = ~/.rein/).
pub fn resolve_buffer_dir(config: &ReinConfig) -> std::path::PathBuf {
    if config.hooks.buffer_dir == "auto" {
        config
            .resolve_db_path()
            .parent()
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
            use sha2::{Digest, Sha256};
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
    with_buffer_lock(path, || {
        use std::io::Write;
        if let Ok(meta) = std::fs::metadata(path) {
            if meta.len() >= MAX_BUFFER_BYTES {
                tracing::warn!(
                    path = ?path,
                    size = meta.len(),
                    cap = MAX_BUFFER_BYTES,
                    "session buffer hit size cap; dropping append"
                );
                return Ok(());
            }
        }
        let entry = serde_json::json!({
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "text": redact_secrets(text),
            "source": source,
        });
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(path)?;
        writeln!(file, "{}", entry)?;
        Ok(())
    })
}

/// Safely read a buffer file, bounded by `MAX_BUFFER_BYTES`. Returns `None`
/// when the file is absent or unreadable; oversize files are dropped and
/// cleared on disk to guarantee progress.
fn read_buffer_bounded(path: &std::path::Path) -> Option<String> {
    let meta = std::fs::metadata(path).ok()?;
    if meta.len() > MAX_BUFFER_BYTES {
        tracing::warn!(
            path = ?path,
            size = meta.len(),
            cap = MAX_BUFFER_BYTES,
            "session buffer exceeded size cap; discarding"
        );
        let _ = std::fs::remove_file(path);
        return None;
    }
    std::fs::read_to_string(path).ok()
}

/// Read all text entries from a buffer file and delete it.
pub fn read_and_clear_buffer(path: &std::path::Path) -> Vec<String> {
    with_buffer_lock(path, || {
        let content = match read_buffer_bounded(path) {
            Some(c) => c,
            None => return Ok(vec![]),
        };
        let _ = std::fs::remove_file(path);
        Ok(content
            .lines()
            .filter_map(|line| {
                serde_json::from_str::<serde_json::Value>(line)
                    .ok()
                    .and_then(|v| {
                        v.get("text")
                            .and_then(|t| t.as_str())
                            .map(|s| s.to_string())
                    })
            })
            .collect())
    })
    .unwrap_or_default()
}

/// Adaptive flush threshold: adjusts based on signal density in the buffer.
pub fn adaptive_flush_threshold(base: usize, buf_path: &std::path::Path) -> usize {
    let content = match read_buffer_bounded(buf_path) {
        Some(c) => c,
        None => return base,
    };
    let total_lines = content.lines().count();
    if total_lines < 5 {
        return base;
    }

    let high_signal = content
        .lines()
        .filter_map(|line| {
            serde_json::from_str::<serde_json::Value>(line)
                .ok()
                .and_then(|v| {
                    v.get("text")
                        .and_then(|t| t.as_str())
                        .map(|s| s.to_string())
                })
        })
        .filter(|text| super::scoring::worth_extracting(text))
        .count();

    let density = high_signal as f64 / total_lines as f64;
    if density > 0.5 {
        base / 2
    } else if density < 0.1 {
        base * 2
    } else {
        base
    }
}

/// Path for the flush-count marker file for a given session buffer.
pub fn flush_marker_path(buf_path: &std::path::Path) -> std::path::PathBuf {
    buf_path.with_extension("flushed")
}

/// Record that a mid-session flush occurred for this session.
/// Atomically increments the flush count stored in the marker file.
pub fn mark_flushed(buf_path: &std::path::Path) {
    let marker = flush_marker_path(buf_path);
    let count = std::fs::read_to_string(&marker)
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
        .unwrap_or(0);
    let _ = std::fs::write(&marker, (count + 1).to_string());
}

/// Read the number of mid-session flushes recorded for this session.
/// Returns 0 if the marker file does not exist.
pub fn flush_count(buf_path: &std::path::Path) -> u32 {
    let marker = flush_marker_path(buf_path);
    std::fs::read_to_string(&marker)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0)
}

/// Delete the flush marker for this session (call at hook_stop end).
pub fn clear_flush_marker(buf_path: &std::path::Path) {
    let _ = std::fs::remove_file(flush_marker_path(buf_path));
}

/// Clean up stale buffer files older than 24 hours.
pub fn cleanup_stale_buffers(config: &ReinConfig) {
    let buf_dir = resolve_buffer_dir(config);
    // Clean stale buffer files
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
                        // Also remove associated lock and flush marker files
                        let lock = buffer_lock_path(&entry);
                        let _ = std::fs::remove_file(&lock);
                        let marker = flush_marker_path(&entry);
                        let _ = std::fs::remove_file(&marker);
                    }
                }
            }
        }
    }
    // Clean orphaned lock files (no matching buffer) — only if not currently held
    #[cfg(unix)]
    {
        let lock_pattern = buf_dir.join("buffer_*.jsonl.lock");
        if let Ok(entries) = glob::glob(&lock_pattern.to_string_lossy()) {
            for entry in entries.flatten() {
                let buf_path = entry.with_extension(""); // strip .lock
                if !buf_path.exists() {
                    // Try non-blocking lock to verify nobody holds this lock file
                    if let Ok(f) = std::fs::OpenOptions::new().read(true).open(&entry) {
                        use std::os::unix::io::AsRawFd;
                        let rc =
                            unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
                        if rc == 0 {
                            // We got the lock — nobody else holds it, safe to remove
                            let _ = unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_UN) };
                            drop(f);
                            tracing::debug!("cleaning orphaned lock: {}", entry.display());
                            let _ = std::fs::remove_file(&entry);
                        }
                        // rc != 0 means someone holds it — skip
                    }
                }
            }
        }
    }
}

fn buffer_lock_path(path: &std::path::Path) -> std::path::PathBuf {
    std::path::PathBuf::from(format!("{}.lock", path.display()))
}

fn with_buffer_lock<T, F>(path: &std::path::Path, f: F) -> anyhow::Result<T>
where
    F: FnOnce() -> anyhow::Result<T>,
{
    let lock_path = buffer_lock_path(path);
    if let Some(parent) = lock_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let lock_file = std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(&lock_path)?;
    #[cfg(unix)]
    {
        use std::os::fd::AsRawFd;
        let rc = unsafe { libc::flock(lock_file.as_raw_fd(), libc::LOCK_EX) };
        if rc != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
    }
    let result = f();
    #[cfg(unix)]
    {
        use std::os::fd::AsRawFd;
        let _ = unsafe { libc::flock(lock_file.as_raw_fd(), libc::LOCK_UN) };
    }
    drop(lock_file);
    result
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
    let date = format!(
        "{}-{}",
        chrono::Utc::now().format("%Y-%m-%d-%H%M"),
        ulid::Ulid::new()
            .to_string()
            .chars()
            .take(6)
            .collect::<String>()
    );
    let definition = if episode.decisions.is_empty() {
        format!("{}\nOutcome: {}", episode.title, episode.outcome)
    } else {
        format!(
            "{}\nOutcome: {}\nDecisions: {}",
            episode.title,
            episode.outcome,
            episode.decisions.join("; ")
        )
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
        last_episode_id: None,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        living_summary: None,
        living_summary_updated_at: None,
        living_summary_source_revision: None,
    };
    store.add_concept(concept)?;
    Ok(())
}
