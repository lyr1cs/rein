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
    pub last_run_at: Option<String>,
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
    let path = queue_path(config);
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
    let rc = unsafe { libc::flock(std::os::fd::AsRawFd::as_raw_fd(&lock_file), libc::LOCK_EX | libc::LOCK_NB) };
    if rc != 0 {
        return Ok(0);
    }

    let result = drain_memory_queue_locked(config, &path, &inflight).await;
    let _ = unsafe { libc::flock(std::os::fd::AsRawFd::as_raw_fd(&lock_file), libc::LOCK_UN) };
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

    if !remaining.is_empty() {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(path)?;
        for job in remaining {
            writeln!(file, "{}", serde_json::to_string(&job)?)?;
        }
    }

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

fn spawn_marker_path(config: &ReinConfig) -> std::path::PathBuf {
    project_scoped_path(config, "memory_worker_spawn")
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
    if !inflight.exists() {
        return Ok(());
    }
    let content = std::fs::read_to_string(inflight).unwrap_or_default();
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
    let backoff = config.async_memory.base_backoff_ms.saturating_mul(2u64.saturating_pow(job.attempts));
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
