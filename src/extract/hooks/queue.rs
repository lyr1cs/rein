//! Async memory queue for record-only proxy and hooks.

use crate::config::ReinConfig;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryJobMode {
    Quick,
    Full,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryJob {
    pub mode: MemoryJobMode,
    pub source: String,
    pub source_query: Option<String>,
    pub text: String,
    pub created_at: String,
}

pub fn queue_memory_job(
    config: &ReinConfig,
    mode: MemoryJobMode,
    source: &str,
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
        mode,
        source: source.to_string(),
        source_query,
        text,
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

pub fn spawn_memory_worker() {
    if std::env::var("REIN_MEMORY_WORKER").as_deref() == Ok("1") {
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
    let mut jobs = content.lines()
        .filter_map(|line| serde_json::from_str::<MemoryJob>(line).ok())
        .collect::<Vec<_>>();

    let mut processed = 0u32;
    let mut remaining = Vec::new();
    while let Some(job) = jobs.first().cloned() {
        jobs.remove(0);
        match process_job(config, job.clone()).await {
            Ok(done) => {
                processed += done;
            }
            Err(e) => {
                tracing::warn!("memory worker job failed: {e}");
                remaining.push(job);
                remaining.extend(jobs.into_iter());
                break;
            }
        }
    }

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
    Ok(processed)
}

async fn process_job(config: &ReinConfig, job: MemoryJob) -> anyhow::Result<u32> {
    match job.mode {
        MemoryJobMode::Quick => {
            let extracted = crate::extract::llm::extract_with_fallback(config, &job.text, 2).await;
            let extracted = dedup_quick(extracted);
            let stored = super::persist::process_quick_extraction(config, extracted)?;
            Ok(stored)
        }
        MemoryJobMode::Full => {
            let result = crate::extract::llm::extract_full_with_fallback(config, &job.text).await;
            let (memories, _concepts, _links) = super::persist::process_full_extraction(config, result)?;
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

fn dedup_quick(items: Vec<crate::extract::llm::ExtractedMemory>) -> Vec<crate::extract::llm::ExtractedMemory> {
    let mut unique: Vec<crate::extract::llm::ExtractedMemory> = Vec::new();
    'outer: for item in items {
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
