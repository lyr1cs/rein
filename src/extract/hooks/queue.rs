//! Async memory queue for record-only proxy and hooks.

use crate::config::ReinConfig;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryJobMode {
    Quick,
    Full,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryJob {
    pub id: String,
    pub mode: MemoryJobMode,
    pub source: String,
    pub source_label: String,
    pub agent_label: String,
    pub is_subagent: bool,
    pub priority: u8,
    pub source_query: Option<String>,
    pub text: String,
    #[serde(default)]
    pub attempts: u32,
    #[serde(default)]
    pub next_attempt_at: Option<DateTime<Utc>>,
    pub created_at: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorkerStats {
    pub processed: u64,
    pub requeued: u64,
    pub dead_lettered: u64,
    pub suppressed_duplicates: u64,
    pub last_run_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RecentEventEntry {
    fingerprint: String,
    preview: String,
    agent_label: String,
    created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct RecentEventState {
    #[serde(default)]
    items: Vec<RecentEventEntry>,
}

pub fn queue_memory_job(
    config: &ReinConfig,
    mode: MemoryJobMode,
    source: &str,
    source_label: &str,
    agent_label: String,
    is_subagent: bool,
    priority: u8,
    source_query: Option<String>,
    text: String,
) -> anyhow::Result<()> {
    if text.trim().is_empty() {
        return Ok(());
    }
    append_raw_archive(
        config,
        mode.clone(),
        source,
        source_label,
        &agent_label,
        is_subagent,
        priority,
        source_query.as_deref(),
        &text,
    )?;

    if suppress_duplicate_event(
        config,
        source,
        &agent_label,
        is_subagent,
        source_query.as_deref(),
        &text,
    )? {
        let mut stats = load_worker_stats(config);
        stats.suppressed_duplicates += 1;
        let _ = save_worker_stats(config, &stats);
        return Ok(());
    }

    // Also check pending queue for similar jobs (prevents cross-session duplicates
    // that fall outside the fingerprint_window_ms).
    let path = queue_path(config);
    if let Ok(queue_content) = std::fs::read_to_string(&path) {
        let preview: String = text.chars().take(500).collect();
        for line in queue_content.lines().rev().take(50) {
            if let Ok(existing) = serde_json::from_str::<MemoryJob>(line) {
                if crate::extract::similarity(&preview, &existing.text.chars().take(500).collect::<String>()) > 0.85 {
                    return Ok(()); // Already queued
                }
            }
        }
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let job = MemoryJob {
        id: ulid::Ulid::new().to_string(),
        mode,
        source: source.to_string(),
        source_label: source_label.to_string(),
        agent_label,
        is_subagent,
        priority,
        source_query,
        text,
        attempts: 0,
        next_attempt_at: None,
        created_at: Utc::now().to_rfc3339(),
    };
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(path)?;
    writeln!(file, "{}", serde_json::to_string(&job)?)?;
    Ok(())
}

pub fn spawn_memory_worker(config: &ReinConfig) {
    if std::env::var("REIN_MEMORY_WORKER").as_deref() == Ok("1") {
        return;
    }
    if !should_spawn_worker(config) {
        return;
    }
    let Ok(exe) = std::env::current_exe() else {
        return;
    };
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("worker")
        .arg("memory")
        .env("REIN_MEMORY_WORKER", "1")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    let _ = cmd.spawn();
    let _ = touch_spawn_marker(config);
}

pub async fn drain_memory_queue(config: &ReinConfig) -> anyhow::Result<u32> {
    let path = queue_path(config);
    let inflight = inflight_path(config);
    let lock = lock_path(config);
    if let Some(parent) = lock.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let lock_file = std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(&lock)?;
    let fd = std::os::fd::AsRawFd::as_raw_fd(&lock_file);
    let rc = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
    if rc != 0 {
        return Ok(0);
    }

    // lock_file is held for the entire operation (including async phases).
    // The flock is advisory and process-scoped — it survives across awaits
    // because the file descriptor stays open in lock_file.
    let result = drain_memory_queue_locked(config, &path, &inflight).await;

    // Explicitly unlock + drop (lock released when fd closes).
    let _ = unsafe { libc::flock(fd, libc::LOCK_UN) };
    drop(lock_file);
    result
}

async fn drain_memory_queue_locked(
    config: &ReinConfig,
    path: &std::path::Path,
    inflight: &std::path::Path,
) -> anyhow::Result<u32> {
    recover_inflight(path, inflight)?;

    if !path.exists() {
        return Ok(0);
    }
    let meta = std::fs::metadata(path)?;
    if meta.len() == 0 {
        return Ok(0);
    }

    std::fs::rename(path, inflight)?;
    let content = std::fs::read_to_string(inflight).unwrap_or_default();
    let jobs = content.lines()
        .filter_map(|line| serde_json::from_str::<MemoryJob>(line).ok())
        .collect::<Vec<_>>();

    let now = Utc::now();
    let mut deferred = Vec::new();
    let mut ready = Vec::new();
    for job in jobs {
        if job.next_attempt_at.is_some_and(|ts| ts > now) {
            deferred.push(job);
        } else {
            ready.push(job);
        }
    }
    ready.sort_by(|a, b| {
        b.priority.cmp(&a.priority)
            .then_with(|| a.created_at.cmp(&b.created_at))
    });

    let mut processed = 0u32;
    let mut remaining = Vec::new();
    let mut stats = load_worker_stats(config);
    let total_ready = ready.len();
    for job in ready.into_iter().take(config.async_memory.max_jobs_per_run) {
        match process_job(config, job.clone()).await {
            Ok(done) => {
                processed += done;
                stats.processed += done as u64;
            }
            Err(e) => {
                tracing::warn!("memory worker job failed: {e}");
                if job.attempts + 1 >= config.async_memory.max_retries {
                    let _ = append_dead_letter(config, &job, &e.to_string());
                    stats.dead_lettered += 1;
                } else {
                    remaining.push(reschedule_job(config, job));
                    stats.requeued += 1;
                }
            }
        }
    }

    // Preserve unprocessed ready jobs and future-scheduled jobs.
    if total_ready > config.async_memory.max_jobs_per_run {
        let mut ready_tail = content.lines()
            .filter_map(|line| serde_json::from_str::<MemoryJob>(line).ok())
            .filter(|job| !job.next_attempt_at.is_some_and(|ts| ts > now))
            .collect::<Vec<_>>();
        ready_tail.sort_by(|a, b| {
            b.priority.cmp(&a.priority)
                .then_with(|| a.created_at.cmp(&b.created_at))
        });
        remaining.extend(ready_tail.into_iter().skip(config.async_memory.max_jobs_per_run));
    }
    remaining.extend(deferred);

    // Write remaining jobs back to queue BEFORE deleting inflight.
    // This ensures no data loss if we crash between these two operations:
    // - If we crash after writing queue but before deleting inflight,
    //   recover_inflight() will append inflight back (duplicates are harmless
    //   because jobs have unique IDs and dedup catches them).
    if !remaining.is_empty() {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(path)?;
        for job in &remaining {
            writeln!(file, "{}", serde_json::to_string(job)?)?;
        }
    }

    // Safe to delete inflight now — remaining jobs are persisted in queue.
    let _ = std::fs::remove_file(inflight);
    stats.last_run_at = Some(Utc::now().to_rfc3339());
    let _ = save_worker_stats(config, &stats);
    Ok(processed)
}

async fn process_job(config: &ReinConfig, job: MemoryJob) -> anyhow::Result<u32> {
    match job.mode {
        MemoryJobMode::Quick => {
            let extracted = crate::extract::llm::extract_with_worker_preference(config, &job.text, 2).await;
            let extracted = dedup_quick(extracted, &job);
            let stored = super::persist::process_quick_extraction(
                config,
                extracted,
                &job.agent_label,
                job.is_subagent,
            )?;
            Ok(stored)
        }
        MemoryJobMode::Full => {
            let result = crate::extract::llm::extract_full_with_worker_preference(config, &job.text).await;
            let (memories, _concepts, _links) = super::persist::process_full_extraction(
                config,
                result,
                &job.agent_label,
                job.is_subagent,
            )?;
            Ok(memories)
        }
    }
}

fn queue_path(config: &ReinConfig) -> std::path::PathBuf {
    project_scoped_path(config, "memory_queue")
}

fn inflight_path(config: &ReinConfig) -> std::path::PathBuf {
    project_scoped_path(config, "memory_queue_inflight")
}

fn lock_path(config: &ReinConfig) -> std::path::PathBuf {
    project_scoped_path(config, "memory_queue_lock")
}

fn dead_letter_path(config: &ReinConfig) -> std::path::PathBuf {
    project_scoped_path(config, "memory_queue_dead")
}

fn stats_path(config: &ReinConfig) -> std::path::PathBuf {
    project_scoped_path(config, "memory_worker_stats")
}

fn archive_path(config: &ReinConfig) -> std::path::PathBuf {
    let date = Utc::now().format("%Y%m%d").to_string();
    project_scoped_path(config, &format!("memory_raw_{date}"))
}

fn spawn_marker_path(config: &ReinConfig) -> std::path::PathBuf {
    project_scoped_path(config, "memory_worker_spawn")
}

fn recent_events_path(config: &ReinConfig) -> std::path::PathBuf {
    project_scoped_path(config, "memory_recent_events")
}

fn project_scoped_path(config: &ReinConfig, prefix: &str) -> std::path::PathBuf {
    let cwd = std::env::current_dir()
        .ok()
        .and_then(|p| p.canonicalize().ok())
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    let mut hasher = Sha256::new();
    hasher.update(cwd.to_string_lossy().as_bytes());
    let digest = format!("{:x}", hasher.finalize());
    super::buffer::resolve_buffer_dir(config).join(format!("{prefix}_{}.jsonl", &digest[..12]))
}

fn should_spawn_worker(config: &ReinConfig) -> bool {
    let path = spawn_marker_path(config);
    let Ok(meta) = std::fs::metadata(path) else {
        return true;
    };
    let Ok(modified) = meta.modified() else {
        return true;
    };
    let modified_utc: DateTime<Utc> = modified.into();
    let elapsed = Utc::now() - modified_utc;
    elapsed.num_milliseconds() >= config.async_memory.spawn_cooldown_ms as i64
}

fn touch_spawn_marker(config: &ReinConfig) -> anyhow::Result<()> {
    let path = spawn_marker_path(config);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, Utc::now().to_rfc3339())?;
    Ok(())
}

fn recover_inflight(path: &std::path::Path, inflight: &std::path::Path) -> anyhow::Result<()> {
    // Read inflight file directly — if it doesn't exist, read_to_string returns Err
    // and we return Ok. No TOCTOU race between exists() and read().
    let content = match std::fs::read_to_string(inflight) {
        Ok(c) => c,
        Err(_) => return Ok(()), // File doesn't exist or unreadable — nothing to recover.
    };
    if !content.trim().is_empty() {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(path)?;
        file.write_all(content.as_bytes())?;
    }
    let _ = std::fs::remove_file(inflight);
    Ok(())
}

fn dedup_quick(
    items: Vec<crate::extract::llm::ExtractedMemory>,
    job: &MemoryJob,
) -> Vec<crate::extract::llm::ExtractedMemory> {
    let mut unique: Vec<crate::extract::llm::ExtractedMemory> = Vec::new();
    'outer: for item in items {
        if job.is_subagent {
            let score = crate::extract::patterns::score_sentence(&item.content)
                .max(crate::extract::patterns::score_sentence(&item.summary));
            let strong_topic = ["architecture", "decision", "debug", "config", "workflow"]
                .iter()
                .any(|k| item.topic.to_lowercase().contains(k));
            if item.quality_confidence < 0.7 && score < 4 && !strong_topic {
                continue;
            }
        }
        for existing in &unique {
            if item.topic == existing.topic
                && (crate::extract::similarity(&item.summary, &existing.summary) > 0.82
                    || crate::extract::similarity(&item.content, &existing.content) > 0.82)
            {
                continue 'outer;
            }
        }
        unique.push(item);
    }
    unique
}

fn reschedule_job(config: &ReinConfig, mut job: MemoryJob) -> MemoryJob {
    let attempts = job.attempts + 1;
    // Cap exponent at 10 to prevent overflow (max backoff = base * 1024 ≈ 34 minutes at 2000ms base).
    let exp = job.attempts.min(10);
    let backoff = config.async_memory.base_backoff_ms.saturating_mul(2u64.saturating_pow(exp));
    job.attempts = attempts;
    job.next_attempt_at = Some(Utc::now() + chrono::Duration::milliseconds(backoff as i64));
    job
}

fn append_dead_letter(config: &ReinConfig, job: &MemoryJob, error: &str) -> anyhow::Result<()> {
    let path = dead_letter_path(config);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(path)?;
    let payload = serde_json::json!({
        "job": job,
        "error": error,
        "failed_at": Utc::now().to_rfc3339(),
    });
    writeln!(file, "{}", serde_json::to_string(&payload)?)?;
    Ok(())
}

fn load_worker_stats(config: &ReinConfig) -> WorkerStats {
    let path = stats_path(config);
    let Ok(text) = std::fs::read_to_string(path) else {
        return WorkerStats::default();
    };
    serde_json::from_str(&text).unwrap_or_default()
}

fn save_worker_stats(config: &ReinConfig, stats: &WorkerStats) -> anyhow::Result<()> {
    let path = stats_path(config);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(stats)?)?;
    Ok(())
}

fn append_raw_archive(
    config: &ReinConfig,
    mode: MemoryJobMode,
    source: &str,
    source_label: &str,
    agent_label: &str,
    is_subagent: bool,
    priority: u8,
    source_query: Option<&str>,
    text: &str,
) -> anyhow::Result<()> {
    let path = archive_path(config);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let payload = serde_json::json!({
        "id": ulid::Ulid::new().to_string(),
        "mode": mode,
        "source": source,
        "source_label": source_label,
        "agent_label": agent_label,
        "is_subagent": is_subagent,
        "priority": priority,
        "source_query": source_query,
        "text": text,
        "created_at": Utc::now().to_rfc3339(),
    });
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(path)?;
    writeln!(file, "{}", serde_json::to_string(&payload)?)?;
    Ok(())
}

fn suppress_duplicate_event(
    config: &ReinConfig,
    source: &str,
    agent_label: &str,
    is_subagent: bool,
    source_query: Option<&str>,
    text: &str,
) -> anyhow::Result<bool> {
    let path = recent_events_path(config);
    let now = Utc::now();
    let mut state = load_recent_events(config);
    let window_ms = config.async_memory.fingerprint_window_ms as i64;

    state.items.retain(|item| (now - item.created_at).num_milliseconds() <= window_ms);

    let normalized = normalized_event_text(source, source_query, text);
    let preview: String = normalized.chars().take(1000).collect();
    let fingerprint = sha256_hex(&preview);

    let duplicate = state.items.iter().any(|item| {
        if item.agent_label != agent_label {
            return false;
        }
        if is_subagent && !item.agent_label.contains(":") {
            return false;
        }
        if item.fingerprint == fingerprint {
            return true;
        }
        crate::extract::similarity(&item.preview, &preview) > 0.94
    });

    state.items.push(RecentEventEntry {
        fingerprint,
        preview,
        agent_label: agent_label.to_string(),
        created_at: now,
    });
    if state.items.len() > config.async_memory.recent_event_cache_size {
        let start = state.items.len() - config.async_memory.recent_event_cache_size;
        state.items = state.items.split_off(start);
    }
    save_recent_events(config, &path, &state)?;
    Ok(duplicate)
}

fn load_recent_events(config: &ReinConfig) -> RecentEventState {
    let path = recent_events_path(config);
    let Ok(text) = std::fs::read_to_string(path) else {
        return RecentEventState::default();
    };
    serde_json::from_str(&text).unwrap_or_default()
}

fn save_recent_events(
    _config: &ReinConfig,
    path: &std::path::Path,
    state: &RecentEventState,
) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(state)?)?;
    Ok(())
}

fn normalized_event_text(source: &str, source_query: Option<&str>, text: &str) -> String {
    let joined = match source_query {
        Some(query) => format!("{source}\n{query}\n{text}"),
        None => format!("{source}\n{text}"),
    };
    joined
        .to_lowercase()
        .chars()
        .map(|ch| {
            if ch.is_alphanumeric()
                || ('\u{4e00}'..='\u{9fff}').contains(&ch)
                || ch.is_whitespace()
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

fn sha256_hex(s: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(s.as_bytes());
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_event_text_collapses_punctuation() {
        let a = normalized_event_text("hook_stop", Some("hello"), "Fixed sqlite-locking!");
        let b = normalized_event_text("hook_stop", Some("hello"), "Fixed sqlite locking.");
        assert!(crate::extract::similarity(&a, &b) > 0.94);
    }
}
