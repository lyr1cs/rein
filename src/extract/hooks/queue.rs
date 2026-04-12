//! Async memory queue for record-only proxy and hooks.

use crate::config::ReinConfig;
use crate::extract::hooks::parsing::redact_secrets;
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
    /// Pre-stored artifact ID — worker links it to the derived episode after extraction.
    #[serde(default)]
    pub artifact_id: Option<String>,
    /// Serialized SessionIngest JSON — lets the worker use the full report path.
    #[serde(default)]
    pub session_json: Option<String>,
    #[serde(default)]
    pub attempts: u32,
    #[serde(default)]
    pub next_attempt_at: Option<DateTime<Utc>>,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupJob {
    pub id: String,
    #[serde(default)]
    pub topic: Option<String>,
    #[serde(default)]
    pub topics: Vec<String>,
    #[serde(default)]
    pub pattern: Option<String>,
    #[serde(default)]
    pub all: bool,
    #[serde(default)]
    pub exact_topics: bool,
    #[serde(default)]
    pub dry_run: bool,
    #[serde(default)]
    pub attempts: u32,
    #[serde(default)]
    pub next_attempt_at: Option<DateTime<Utc>>,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DedupJob {
    pub id: String,
    pub existing_id: String,
    pub new_id: String,
    #[serde(default)]
    pub lexical_score: Option<f32>,
    #[serde(default)]
    pub reason: String,
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

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueueDiagnostics {
    pub pending: usize,
    pub inflight: usize,
    pub dead_letters: usize,
    pub stats: WorkerStats,
    pub issues: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueueGroupDiagnostics {
    pub memory: QueueDiagnostics,
    pub cleanup: QueueDiagnostics,
    pub dedup: QueueDiagnostics,
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

/// Queue a memory job with optional session metadata for artifact-episode linking.
pub fn queue_memory_job_with_session(
    config: &ReinConfig,
    mode: MemoryJobMode,
    source: &str,
    source_label: &str,
    agent_label: String,
    is_subagent: bool,
    priority: u8,
    source_query: Option<String>,
    text: String,
    artifact_id: Option<String>,
    session_json: Option<String>,
) -> anyhow::Result<()> {
    _queue_memory_job(
        config,
        mode,
        source,
        source_label,
        agent_label,
        is_subagent,
        priority,
        source_query,
        text,
        artifact_id,
        session_json,
    )
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
    _queue_memory_job(
        config,
        mode,
        source,
        source_label,
        agent_label,
        is_subagent,
        priority,
        source_query,
        text,
        None,
        None,
    )
}

fn _queue_memory_job(
    config: &ReinConfig,
    mode: MemoryJobMode,
    source: &str,
    source_label: &str,
    agent_label: String,
    is_subagent: bool,
    priority: u8,
    source_query: Option<String>,
    text: String,
    artifact_id: Option<String>,
    session_json: Option<String>,
) -> anyhow::Result<()> {
    let text = redact_secrets(&text);
    let source_query = source_query.map(|q| redact_secrets(&q));
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
    let path = queue_path(config);
    with_advisory_lock(&lock_path(config), true, || {
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
        let preview: String = text.chars().take(500).collect();
        let is_full = matches!(mode, MemoryJobMode::Full);
        let queue_content = std::fs::read_to_string(&path).unwrap_or_default();
        for line in queue_content.lines().rev().take(50) {
            if let Ok(existing) = serde_json::from_str::<MemoryJob>(line) {
                // Don't suppress a Full job if existing is Quick.
                if is_full && matches!(existing.mode, MemoryJobMode::Quick) {
                    continue;
                }
                if crate::extract::similarity(
                    &preview,
                    &existing.text.chars().take(500).collect::<String>(),
                ) > 0.85
                {
                    return Ok(()); // Already queued with same or higher fidelity
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
            artifact_id,
            session_json,
            attempts: 0,
            next_attempt_at: None,
            created_at: Utc::now().to_rfc3339(),
        };
        append_jsonl(&path, &serde_json::to_string(&job)?)
    })
}

pub fn spawn_memory_worker(config: &ReinConfig) {
    if std::env::var("REIN_MEMORY_WORKER").as_deref() == Ok("1") {
        return;
    }
    if !should_spawn_worker(
        &spawn_marker_path(config),
        config.async_memory.spawn_cooldown_ms,
    ) {
        return;
    }
    // Touch spawn marker BEFORE spawning to close TOCTOU race window where two
    // concurrent hook invocations both see the cooldown as expired.
    let _ = touch_spawn_marker(config);
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

pub(crate) fn collect_queue_diagnostics(config: &ReinConfig) -> QueueGroupDiagnostics {
    let (memory_pending, memory_pending_issue) =
        diagnostic_count(&queue_path(config), "memory_queue");
    let (memory_inflight, memory_inflight_issue) =
        diagnostic_count(&inflight_path(config), "memory_queue_inflight");
    let (memory_dead, memory_dead_issue) =
        diagnostic_count(&dead_letter_path(config), "memory_queue_dead");
    let (memory_stats, memory_stats_issue) =
        diagnostic_stats(&stats_path(config), "memory_worker_stats");

    let (cleanup_pending, cleanup_pending_issue) =
        diagnostic_count(&cleanup_queue_path(config), "cleanup_queue");
    let (cleanup_inflight, cleanup_inflight_issue) =
        diagnostic_count(&cleanup_inflight_path(config), "cleanup_queue_inflight");
    let (cleanup_dead, cleanup_dead_issue) =
        diagnostic_count(&cleanup_dead_letter_path(config), "cleanup_queue_dead");
    let (cleanup_stats, cleanup_stats_issue) =
        diagnostic_stats(&cleanup_stats_path(config), "cleanup_worker_stats");

    let (dedup_pending, dedup_pending_issue) =
        diagnostic_count(&dedup_queue_path(config), "dedup_queue");
    let (dedup_inflight, dedup_inflight_issue) =
        diagnostic_count(&dedup_inflight_path(config), "dedup_queue_inflight");
    let (dedup_dead, dedup_dead_issue) =
        diagnostic_count(&dedup_dead_letter_path(config), "dedup_queue_dead");
    let (dedup_stats, dedup_stats_issue) =
        diagnostic_stats(&dedup_stats_path(config), "dedup_worker_stats");

    QueueGroupDiagnostics {
        memory: QueueDiagnostics {
            pending: memory_pending,
            inflight: memory_inflight,
            dead_letters: memory_dead,
            stats: memory_stats,
            issues: collect_issues([
                memory_pending_issue,
                memory_inflight_issue,
                memory_dead_issue,
                memory_stats_issue,
            ]),
        },
        cleanup: QueueDiagnostics {
            pending: cleanup_pending,
            inflight: cleanup_inflight,
            dead_letters: cleanup_dead,
            stats: cleanup_stats,
            issues: collect_issues([
                cleanup_pending_issue,
                cleanup_inflight_issue,
                cleanup_dead_issue,
                cleanup_stats_issue,
            ]),
        },
        dedup: QueueDiagnostics {
            pending: dedup_pending,
            inflight: dedup_inflight,
            dead_letters: dedup_dead,
            stats: dedup_stats,
            issues: collect_issues([
                dedup_pending_issue,
                dedup_inflight_issue,
                dedup_dead_issue,
                dedup_stats_issue,
            ]),
        },
    }
}

pub fn queue_cleanup_job(
    config: &ReinConfig,
    topic: Option<String>,
    topics: Vec<String>,
    pattern: Option<String>,
    all: bool,
    exact_topics: bool,
    dry_run: bool,
) -> anyhow::Result<String> {
    let job = CleanupJob {
        id: ulid::Ulid::new().to_string(),
        topic,
        topics,
        pattern,
        all,
        exact_topics,
        dry_run,
        attempts: 0,
        next_attempt_at: None,
        created_at: Utc::now().to_rfc3339(),
    };

    let fingerprint = cleanup_job_fingerprint(&job);
    let path = cleanup_queue_path(config);
    let inflight = cleanup_inflight_path(config);
    with_advisory_lock(&cleanup_lock_path(config), true, || {
        for queued in read_cleanup_jobs(&path)
            .into_iter()
            .chain(read_cleanup_jobs(&inflight).into_iter())
        {
            if cleanup_job_fingerprint(&queued) == fingerprint {
                return Ok(queued.id);
            }
        }
        append_jsonl(&path, &serde_json::to_string(&job)?)?;
        Ok(job.id.clone())
    })
}

pub fn spawn_cleanup_worker(config: &ReinConfig) {
    if std::env::var("REIN_CLEANUP_WORKER").as_deref() == Ok("1") {
        return;
    }
    if !should_spawn_worker(
        &cleanup_spawn_marker_path(config),
        config.async_memory.spawn_cooldown_ms,
    ) {
        return;
    }
    // Touch marker BEFORE spawning to close TOCTOU race window
    let _ = touch_worker_marker(&cleanup_spawn_marker_path(config));
    let Ok(exe) = std::env::current_exe() else {
        return;
    };
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("worker")
        .arg("cleanup-queue")
        .env("REIN_CLEANUP_WORKER", "1")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    let _ = cmd.spawn();
}

pub fn queue_dedup_job(
    config: &ReinConfig,
    existing_id: String,
    new_id: String,
    lexical_score: Option<f32>,
    reason: impl Into<String>,
) -> anyhow::Result<String> {
    let job = DedupJob {
        id: ulid::Ulid::new().to_string(),
        existing_id,
        new_id,
        lexical_score,
        reason: reason.into(),
        attempts: 0,
        next_attempt_at: None,
        created_at: Utc::now().to_rfc3339(),
    };

    let fingerprint = dedup_job_fingerprint(&job);
    let path = dedup_queue_path(config);
    let inflight = dedup_inflight_path(config);
    with_advisory_lock(&dedup_lock_path(config), true, || {
        for queued in read_dedup_jobs(&path)
            .into_iter()
            .chain(read_dedup_jobs(&inflight).into_iter())
        {
            if dedup_job_fingerprint(&queued) == fingerprint {
                return Ok(queued.id);
            }
        }
        append_jsonl(&path, &serde_json::to_string(&job)?)?;
        Ok(job.id.clone())
    })
}

pub fn spawn_dedup_worker(config: &ReinConfig) {
    if std::env::var("REIN_DEDUP_WORKER").as_deref() == Ok("1") {
        return;
    }
    if !should_spawn_worker(
        &dedup_spawn_marker_path(config),
        config.async_memory.spawn_cooldown_ms,
    ) {
        return;
    }
    // Touch marker BEFORE spawning to close TOCTOU race window
    let _ = touch_worker_marker(&dedup_spawn_marker_path(config));
    let Ok(exe) = std::env::current_exe() else {
        return;
    };
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("worker")
        .arg("dedup-queue")
        .env("REIN_DEDUP_WORKER", "1")
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
    let fd = std::os::fd::AsRawFd::as_raw_fd(&lock_file);
    let rc = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
    if rc != 0 {
        let err = std::io::Error::last_os_error();
        if err.raw_os_error() != Some(libc::EWOULDBLOCK) && err.raw_os_error() != Some(libc::EAGAIN)
        {
            tracing::warn!("flock failed: {}", err);
        }
        return Ok(0); // another worker is running or lock unavailable
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

pub async fn drain_cleanup_queue(config: &ReinConfig) -> anyhow::Result<u32> {
    let path = cleanup_queue_path(config);
    let inflight = cleanup_inflight_path(config);
    let lock = cleanup_lock_path(config);
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
        let err = std::io::Error::last_os_error();
        if err.raw_os_error() != Some(libc::EWOULDBLOCK) && err.raw_os_error() != Some(libc::EAGAIN)
        {
            tracing::warn!("cleanup flock failed: {}", err);
        }
        return Ok(0);
    }

    let result = drain_cleanup_queue_locked(config, &path, &inflight).await;

    let _ = unsafe { libc::flock(fd, libc::LOCK_UN) };
    drop(lock_file);
    result
}

pub async fn drain_dedup_queue(config: &ReinConfig) -> anyhow::Result<u32> {
    let path = dedup_queue_path(config);
    let inflight = dedup_inflight_path(config);
    let lock = dedup_lock_path(config);
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
        let err = std::io::Error::last_os_error();
        if err.raw_os_error() != Some(libc::EWOULDBLOCK) && err.raw_os_error() != Some(libc::EAGAIN)
        {
            tracing::warn!("dedup flock failed: {}", err);
        }
        return Ok(0);
    }

    let result = drain_dedup_queue_locked(config, &path, &inflight).await;

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
    let jobs = content
        .lines()
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
        b.priority
            .cmp(&a.priority)
            .then_with(|| a.created_at.cmp(&b.created_at))
    });

    let mut processed = 0u32;
    let mut remaining = Vec::new();
    let mut stats = load_worker_stats(config);

    // Split ready jobs: process first batch, keep the rest
    let split_at = config.async_memory.max_jobs_per_run.min(ready.len());
    let ready_tail = ready.split_off(split_at);
    let to_process = ready; // first `split_at` jobs

    for job in to_process {
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

    // Preserve unprocessed ready jobs (already split above) and future-scheduled jobs.
    remaining.extend(ready_tail);
    remaining.extend(deferred);

    // Write remaining jobs atomically: write to temp file first, then rename.
    // This prevents partial writes from panics or crashes from corrupting the queue.
    if !remaining.is_empty() {
        use std::io::Write;
        let tmp_path = path.with_extension("jsonl.tmp");
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp_path)?;
        for job in &remaining {
            writeln!(file, "{}", serde_json::to_string(job)?)?;
        }
        file.sync_all()?;
        std::fs::rename(&tmp_path, path)?;
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
            let extracted =
                crate::extract::llm::extract_with_worker_preference(config, &job.text, 2).await;
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
            // If we have a serialized SessionIngest, use the full report path
            // so artifact-episode linking and rich metadata are preserved.
            if let Some(ref session_json) = job.session_json {
                if let Ok(session) =
                    serde_json::from_str::<crate::types::SessionIngest>(session_json)
                {
                    let result =
                        crate::extract::llm::extract_full_with_worker_preference(config, &job.text)
                            .await;
                    let mut report = crate::ops::ingest_extraction_report(
                        config,
                        &session,
                        result,
                        Some(&job.agent_label),
                        job.is_subagent,
                    )?;
                    // Link pre-stored artifact to the derived episode
                    if let (Some(ref artifact_id), Some(ref episode_id)) =
                        (&job.artifact_id, &report.episode_id)
                    {
                        if let Ok(store) = config.open_store() {
                            let _ = store.link_session_artifact_episode(artifact_id, episode_id);
                        }
                    }
                    report.artifact_id = job.artifact_id.clone();
                    return Ok(report.memory_count);
                }
            }
            // Fallback: legacy text extraction path
            let result =
                crate::extract::llm::extract_full_with_worker_preference(config, &job.text).await;
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

fn cleanup_queue_path(config: &ReinConfig) -> std::path::PathBuf {
    project_scoped_path(config, "cleanup_queue")
}

fn cleanup_inflight_path(config: &ReinConfig) -> std::path::PathBuf {
    project_scoped_path(config, "cleanup_queue_inflight")
}

fn cleanup_lock_path(config: &ReinConfig) -> std::path::PathBuf {
    project_scoped_path(config, "cleanup_queue_lock")
}

fn cleanup_dead_letter_path(config: &ReinConfig) -> std::path::PathBuf {
    project_scoped_path(config, "cleanup_queue_dead")
}

fn cleanup_stats_path(config: &ReinConfig) -> std::path::PathBuf {
    project_scoped_path(config, "cleanup_worker_stats")
}

fn cleanup_spawn_marker_path(config: &ReinConfig) -> std::path::PathBuf {
    project_scoped_path(config, "cleanup_worker_spawn")
}

fn dedup_queue_path(config: &ReinConfig) -> std::path::PathBuf {
    project_scoped_path(config, "dedup_queue")
}

fn dedup_inflight_path(config: &ReinConfig) -> std::path::PathBuf {
    project_scoped_path(config, "dedup_queue_inflight")
}

fn dedup_lock_path(config: &ReinConfig) -> std::path::PathBuf {
    project_scoped_path(config, "dedup_queue_lock")
}

fn dedup_dead_letter_path(config: &ReinConfig) -> std::path::PathBuf {
    project_scoped_path(config, "dedup_queue_dead")
}

fn dedup_stats_path(config: &ReinConfig) -> std::path::PathBuf {
    project_scoped_path(config, "dedup_worker_stats")
}

fn dedup_spawn_marker_path(config: &ReinConfig) -> std::path::PathBuf {
    project_scoped_path(config, "dedup_worker_spawn")
}

fn project_scoped_path(config: &ReinConfig, prefix: &str) -> std::path::PathBuf {
    let base = super::buffer::resolve_buffer_dir(config);
    // Derive a short discriminator from the DB path so separate rein instances
    // (different databases) get isolated queues, while instances sharing the same
    // DB correctly share one queue.
    let db_tag = {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        config.resolve_db_path().hash(&mut h);
        format!("{:016x}", h.finish())
    };
    let queue_dir = base.join("queue").join(&db_tag);
    let _ = std::fs::create_dir_all(&queue_dir);
    queue_dir.join(format!("{prefix}.jsonl"))
}

fn count_jsonl_lines_checked(path: &std::path::Path) -> anyhow::Result<usize> {
    match std::fs::read_to_string(path) {
        Ok(text) => Ok(text.lines().filter(|line| !line.trim().is_empty()).count()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(0),
        Err(e) => Err(e.into()),
    }
}

fn load_worker_stats_checked(path: &std::path::Path) -> anyhow::Result<WorkerStats> {
    match std::fs::read_to_string(path) {
        Ok(text) => Ok(serde_json::from_str(&text)?),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(WorkerStats::default()),
        Err(e) => Err(e.into()),
    }
}

fn diagnostic_count(path: &std::path::Path, label: &str) -> (usize, Option<String>) {
    match count_jsonl_lines_checked(path) {
        Ok(count) => (count, None),
        Err(e) => (0, Some(format!("{label}: {e}"))),
    }
}

fn diagnostic_stats(path: &std::path::Path, label: &str) -> (WorkerStats, Option<String>) {
    match load_worker_stats_checked(path) {
        Ok(stats) => (stats, None),
        Err(e) => (WorkerStats::default(), Some(format!("{label}: {e}"))),
    }
}

fn collect_issues(entries: [Option<String>; 4]) -> Vec<String> {
    entries.into_iter().flatten().collect()
}

fn should_spawn_worker(path: &std::path::Path, cooldown_ms: u64) -> bool {
    let Ok(meta) = std::fs::metadata(path) else {
        return true;
    };
    let Ok(modified) = meta.modified() else {
        return true;
    };
    let modified_utc: DateTime<Utc> = modified.into();
    let elapsed = Utc::now() - modified_utc;
    elapsed.num_milliseconds() >= cooldown_ms as i64
}

fn touch_spawn_marker(config: &ReinConfig) -> anyhow::Result<()> {
    touch_worker_marker(&spawn_marker_path(config))
}

fn touch_worker_marker(path: &std::path::Path) -> anyhow::Result<()> {
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
            let topic_lower = item.topic.to_lowercase();
            // High-value topics from subagents always pass (architectural decisions, etc.)
            let high_value = [
                "architecture",
                "decision",
                "design",
                "security",
                "migration",
            ]
            .iter()
            .any(|k| topic_lower.contains(k));
            // Medium-value topics get a lower bar
            let medium_value = ["debug", "config", "workflow", "deployment", "fix"]
                .iter()
                .any(|k| topic_lower.contains(k));
            if high_value {
                // Always admit high-value subagent items
            } else if medium_value {
                // Relaxed threshold for medium-value topics
                if item.quality_confidence < 0.4 && score < 3 {
                    continue;
                }
            } else {
                // Strict threshold for low-value subagent noise
                if item.quality_confidence < 0.7 && score < 4 {
                    continue;
                }
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
    let backoff = config
        .async_memory
        .base_backoff_ms
        .saturating_mul(2u64.saturating_pow(exp));
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

fn append_cleanup_dead_letter(
    config: &ReinConfig,
    job: &CleanupJob,
    error: &str,
) -> anyhow::Result<()> {
    let path = cleanup_dead_letter_path(config);
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

fn append_dedup_dead_letter(
    config: &ReinConfig,
    job: &DedupJob,
    error: &str,
) -> anyhow::Result<()> {
    let path = dedup_dead_letter_path(config);
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

fn load_cleanup_worker_stats(config: &ReinConfig) -> WorkerStats {
    let path = cleanup_stats_path(config);
    let Ok(text) = std::fs::read_to_string(path) else {
        return WorkerStats::default();
    };
    serde_json::from_str(&text).unwrap_or_default()
}

fn load_dedup_worker_stats(config: &ReinConfig) -> WorkerStats {
    let path = dedup_stats_path(config);
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
    // Atomic write: write to temp file then rename (prevents corruption on crash)
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(stats)?)?;
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

fn save_cleanup_worker_stats(config: &ReinConfig, stats: &WorkerStats) -> anyhow::Result<()> {
    let path = cleanup_stats_path(config);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(stats)?)?;
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

fn save_dedup_worker_stats(config: &ReinConfig, stats: &WorkerStats) -> anyhow::Result<()> {
    let path = dedup_stats_path(config);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(stats)?)?;
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

async fn drain_cleanup_queue_locked(
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
    let jobs = content
        .lines()
        .filter_map(|line| serde_json::from_str::<CleanupJob>(line).ok())
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
    ready.sort_by(|a, b| a.created_at.cmp(&b.created_at));

    let mut processed = 0u32;
    let mut remaining = Vec::new();
    let mut stats = load_cleanup_worker_stats(config);

    let split_at = config.async_memory.batch_size.min(ready.len());
    let ready_tail = ready.split_off(split_at);
    let to_process = ready;

    for job in to_process {
        match process_cleanup_job(config, job.clone()).await {
            Ok(done) => {
                processed += done;
                stats.processed += done as u64;
            }
            Err(e) => {
                tracing::warn!("cleanup worker job failed: {e}");
                if job.attempts + 1 >= config.async_memory.max_retries {
                    let _ = append_cleanup_dead_letter(config, &job, &e.to_string());
                    stats.dead_lettered += 1;
                } else {
                    remaining.push(reschedule_cleanup_job(config, job));
                    stats.requeued += 1;
                }
            }
        }
    }

    remaining.extend(ready_tail);
    remaining.extend(deferred);

    if !remaining.is_empty() {
        use std::io::Write;
        let tmp_path = path.with_extension("jsonl.tmp");
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp_path)?;
        for job in &remaining {
            writeln!(file, "{}", serde_json::to_string(job)?)?;
        }
        file.sync_all()?;
        std::fs::rename(&tmp_path, path)?;
    }

    let _ = std::fs::remove_file(inflight);
    stats.last_run_at = Some(Utc::now().to_rfc3339());
    let _ = save_cleanup_worker_stats(config, &stats);
    Ok(processed)
}

async fn process_cleanup_job(config: &ReinConfig, job: CleanupJob) -> anyhow::Result<u32> {
    let store = config.open_store()?;
    let scope_all =
        job.all || (job.topic.is_none() && job.topics.is_empty() && job.pattern.is_none());
    let merge_variants = !job.exact_topics;
    let groups = crate::ops::resolve_topic_groups(
        &store,
        job.topic.as_deref(),
        &job.topics,
        job.pattern.as_deref(),
        scope_all,
        merge_variants,
    )?;
    if groups.is_empty() {
        tracing::info!(
            "cleanup worker: no topics matched for queued job {}",
            job.id
        );
        return Ok(1);
    }

    let report =
        crate::ops::run_cleanup_async(&store, config, &groups, merge_variants, job.dry_run).await?;
    tracing::info!(
        "cleanup worker: job {} finished; groups={}, memories={}, dedup_removed={}/{}",
        job.id,
        report.consolidation.groups_processed,
        report.consolidation.memories_replaced,
        report.duplicates_merged,
        report.duplicates_found
    );
    Ok(1)
}

async fn drain_dedup_queue_locked(
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
    let jobs = content
        .lines()
        .filter_map(|line| serde_json::from_str::<DedupJob>(line).ok())
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
    ready.sort_by(|a, b| a.created_at.cmp(&b.created_at));

    let mut processed = 0u32;
    let mut remaining = Vec::new();
    let mut stats = load_dedup_worker_stats(config);

    let split_at = config.async_memory.batch_size.min(ready.len());
    let ready_tail = ready.split_off(split_at);
    let to_process = ready;

    for job in to_process {
        match process_dedup_job(config, job.clone()).await {
            Ok(done) => {
                processed += done;
                stats.processed += done as u64;
            }
            Err(e) => {
                tracing::warn!("dedup worker job failed: {e}");
                if job.attempts + 1 >= config.async_memory.max_retries {
                    let _ = append_dedup_dead_letter(config, &job, &e.to_string());
                    stats.dead_lettered += 1;
                } else {
                    remaining.push(reschedule_dedup_job(config, job));
                    stats.requeued += 1;
                }
            }
        }
    }

    remaining.extend(ready_tail);
    remaining.extend(deferred);

    if !remaining.is_empty() {
        use std::io::Write;
        let tmp_path = path.with_extension("jsonl.tmp");
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp_path)?;
        for job in &remaining {
            writeln!(file, "{}", serde_json::to_string(job)?)?;
        }
        file.sync_all()?;
        std::fs::rename(&tmp_path, path)?;
    }

    let _ = std::fs::remove_file(inflight);
    stats.last_run_at = Some(Utc::now().to_rfc3339());
    let _ = save_dedup_worker_stats(config, &stats);
    Ok(processed)
}

async fn process_dedup_job(config: &ReinConfig, job: DedupJob) -> anyhow::Result<u32> {
    let store = config.open_store()?;
    let relation = crate::ops::resolve_dedup_job_async(
        &store,
        config,
        &job.existing_id,
        &job.new_id,
        job.lexical_score,
        &job.reason,
    )
    .await?;
    tracing::info!(
        "dedup worker: job {} resolved {} vs {} => {}",
        job.id,
        job.existing_id,
        job.new_id,
        relation
    );
    Ok(1)
}

fn cleanup_job_fingerprint(job: &CleanupJob) -> String {
    let mut topics = job.topics.clone();
    topics.sort();
    topics.dedup();
    format!(
        "topic={:?}|topics={:?}|pattern={:?}|all={}|exact={}|dry={}",
        job.topic, topics, job.pattern, job.all, job.exact_topics, job.dry_run
    )
}

fn dedup_job_fingerprint(job: &DedupJob) -> String {
    let (left, right) = if job.existing_id <= job.new_id {
        (&job.existing_id, &job.new_id)
    } else {
        (&job.new_id, &job.existing_id)
    };
    format!("{left}|{right}|{}", job.reason)
}

fn read_cleanup_jobs(path: &std::path::Path) -> Vec<CleanupJob> {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .filter_map(|line| serde_json::from_str::<CleanupJob>(line).ok())
        .collect()
}

fn read_dedup_jobs(path: &std::path::Path) -> Vec<DedupJob> {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .filter_map(|line| serde_json::from_str::<DedupJob>(line).ok())
        .collect()
}

fn reschedule_cleanup_job(config: &ReinConfig, mut job: CleanupJob) -> CleanupJob {
    let attempts = job.attempts + 1;
    let exp = job.attempts.min(10);
    let backoff = config
        .async_memory
        .base_backoff_ms
        .saturating_mul(2u64.saturating_pow(exp));
    job.attempts = attempts;
    job.next_attempt_at = Some(Utc::now() + chrono::Duration::milliseconds(backoff as i64));
    job
}

fn reschedule_dedup_job(config: &ReinConfig, mut job: DedupJob) -> DedupJob {
    let attempts = job.attempts + 1;
    let exp = job.attempts.min(10);
    let backoff = config
        .async_memory
        .base_backoff_ms
        .saturating_mul(2u64.saturating_pow(exp));
    job.attempts = attempts;
    job.next_attempt_at = Some(Utc::now() + chrono::Duration::milliseconds(backoff as i64));
    job
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
    append_jsonl(&path, &serde_json::to_string(&payload)?)
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

    state
        .items
        .retain(|item| (now - item.created_at).num_milliseconds() <= window_ms);

    let normalized = normalized_event_text(source, source_query, text);
    let preview: String = normalized.chars().take(2000).collect();
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
    // Atomic write: write to temp file then rename (prevents corruption on crash)
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(state)?)?;
    std::fs::rename(&tmp, path)?;
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
            if ch.is_alphanumeric() || ('\u{4e00}'..='\u{9fff}').contains(&ch) || ch.is_whitespace()
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

fn append_jsonl(path: &std::path::Path, line: &str) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let lock_path = std::path::PathBuf::from(format!("{}.lock", path.display()));
    with_advisory_lock(&lock_path, true, || {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(path)?;
        writeln!(file, "{line}")?;
        Ok(())
    })
}

fn with_advisory_lock<T, F>(lock_path: &std::path::Path, blocking: bool, f: F) -> anyhow::Result<T>
where
    F: FnOnce() -> anyhow::Result<T>,
{
    if let Some(parent) = lock_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let lock_file = std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(lock_path)?;
    #[cfg(unix)]
    {
        use std::os::fd::AsRawFd;
        let mode = if blocking {
            libc::LOCK_EX
        } else {
            libc::LOCK_EX | libc::LOCK_NB
        };
        let rc = unsafe { libc::flock(lock_file.as_raw_fd(), mode) };
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_event_text_collapses_punctuation() {
        let a = normalized_event_text("hook_stop", Some("hello"), "Fixed sqlite-locking!");
        let b = normalized_event_text("hook_stop", Some("hello"), "Fixed sqlite locking.");
        assert!(crate::extract::similarity(&a, &b) > 0.94);
    }

    #[test]
    fn cleanup_job_fingerprint_sorts_topics() {
        let a = CleanupJob {
            id: "a".into(),
            topic: None,
            topics: vec!["rmcp".into(), "docker".into()],
            pattern: None,
            all: false,
            exact_topics: false,
            dry_run: false,
            attempts: 0,
            next_attempt_at: None,
            created_at: Utc::now().to_rfc3339(),
        };
        let b = CleanupJob {
            id: "b".into(),
            topic: None,
            topics: vec!["docker".into(), "rmcp".into(), "docker".into()],
            pattern: None,
            all: false,
            exact_topics: false,
            dry_run: false,
            attempts: 0,
            next_attempt_at: None,
            created_at: Utc::now().to_rfc3339(),
        };

        assert_eq!(cleanup_job_fingerprint(&a), cleanup_job_fingerprint(&b));
    }

    #[test]
    fn dedup_job_fingerprint_sorts_pair_ids() {
        let a = DedupJob {
            id: "a".into(),
            existing_id: "old".into(),
            new_id: "new".into(),
            lexical_score: Some(0.62),
            reason: "store_gray_zone".into(),
            attempts: 0,
            next_attempt_at: None,
            created_at: Utc::now().to_rfc3339(),
        };
        let b = DedupJob {
            id: "b".into(),
            existing_id: "new".into(),
            new_id: "old".into(),
            lexical_score: Some(0.62),
            reason: "store_gray_zone".into(),
            attempts: 0,
            next_attempt_at: None,
            created_at: Utc::now().to_rfc3339(),
        };

        assert_eq!(dedup_job_fingerprint(&a), dedup_job_fingerprint(&b));
    }
}
