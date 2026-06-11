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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeRefinementJob {
    pub id: String,
    /// ID of the winner memory whose merged content should be refined.
    pub winner_id: String,
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
    pub merge_refinement: QueueDiagnostics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RecentEventEntry {
    fingerprint: String,
    /// v0.37 hook-dup fix: sha256 of the FULL normalized event text (not the
    /// 2000-char preview `fingerprint`). Used for the cross-agent-label exact
    /// duplicate check so a shared preamble with a differing tail cannot
    /// false-suppress a distinct event. `#[serde(default)]` → legacy cache
    /// rows deserialize to "" and never cross-match.
    #[serde(default)]
    full_fingerprint: String,
    /// v0.37 hook-dup fix: whether the event was queued as a Full extraction.
    /// The cross-agent exact-duplicate path must NOT let an incoming Full job
    /// be suppressed by a stored Quick one (Full is higher fidelity), mirroring
    /// the pending-queue Full-over-Quick exception. `#[serde(default)]` → legacy
    /// rows default to `false` (Quick), which only ever loosens suppression.
    #[serde(default)]
    is_full: bool,
    /// v1.2 audit F2: char count of the FULL normalized event text. Lets the
    /// suppressor distinguish "same event re-captured" (suppress) from "the
    /// same growing document, now longer" (prefix EXTENSION — new content the
    /// earlier capture never saw). `#[serde(default)]` → legacy rows are 0 =
    /// unknown, which disables extension detection (falls back to suppress —
    /// the pre-v1.2 behavior).
    #[serde(default)]
    normalized_len: usize,
    preview: String,
    agent_label: String,
    created_at: DateTime<Utc>,
}

/// v1.2 audit F2: tri-state verdict from the recent-events suppressor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SuppressVerdict {
    /// No recent match — proceed to normal enqueue.
    New,
    /// Re-capture of already-seen content — drop.
    Duplicate,
    /// Prefix-extension of recently-seen content (e.g. the per-turn Stop
    /// firing re-rendering a GROWING session transcript): the tail is new
    /// information the earlier capture never had, so it must not be dropped —
    /// but per-turn re-extraction of the whole transcript would be O(N²)
    /// LLM cost. The enqueue path resolves this by REPLACING a still-pending
    /// matching job with the longer snapshot (one job per drain cycle, always
    /// the latest), enqueueing fresh only when nothing is pending.
    Extension,
}

/// v1.2 audit F2/F3: what actually happened to an enqueue request. `Err` from
/// the queue fns still means I/O failure; this is the Ok-payload so callers
/// (hook_post's flushed-content ledger in particular) can distinguish
/// "content WILL be extracted" from "content was dropped as a duplicate" —
/// recording dropped content in the ledger would let hook_stop strip turns
/// that were never extracted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnqueueOutcome {
    /// A new job was appended to the queue.
    Enqueued,
    /// A still-pending job for the same growing document was replaced with
    /// this longer snapshot.
    Replaced,
    /// codex remediation R8 P2: a SIMILAR (≥0.85 on the 500-char preview)
    /// job is already pending — the content is on a path to extraction via
    /// that job, so this is not a drop. Distinct from `Enqueued`/`Replaced`
    /// because the pending job's text is NOT byte-identical to ours:
    /// hook_post must record the flush marker (a mid-session pass covers
    /// this content) but must NOT ledger-record our exact lines (the
    /// pending job may never extract them verbatim — same false-ledger
    /// hazard as audit F3).
    CoveredByPending,
    /// Dropped as a duplicate (recent-events match).
    SuppressedDuplicate,
}

impl EnqueueOutcome {
    /// True when the submitted content is on a path to extraction.
    pub fn accepted(self) -> bool {
        !matches!(self, EnqueueOutcome::SuppressedDuplicate)
    }

    /// True when the QUEUED text is byte-identical to (or a strict extension
    /// of) the submitted content — the only cases where ledger-recording the
    /// submitted lines as "flushed" is sound.
    pub fn carries_exact_content(self) -> bool {
        matches!(self, EnqueueOutcome::Enqueued | EnqueueOutcome::Replaced)
    }
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
) -> anyhow::Result<EnqueueOutcome> {
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
) -> anyhow::Result<EnqueueOutcome> {
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
) -> anyhow::Result<EnqueueOutcome> {
    let text = redact_secrets(&text);
    let source_query = source_query.map(|q| redact_secrets(&q));
    if text.trim().is_empty() {
        // Nothing extractable was submitted; treat as suppressed so callers
        // never ledger-record empty content.
        return Ok(EnqueueOutcome::SuppressedDuplicate);
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
        let verdict = suppress_duplicate_event(
            config,
            source,
            &agent_label,
            is_subagent,
            matches!(mode, MemoryJobMode::Full),
            source_query.as_deref(),
            &text,
        )?;
        if verdict == SuppressVerdict::Duplicate {
            let mut stats = load_worker_stats(config);
            stats.suppressed_duplicates += 1;
            let _ = save_worker_stats(config, &stats);
            return Ok(EnqueueOutcome::SuppressedDuplicate);
        }

        // v1.2 audit F2: a prefix EXTENSION of recently-seen content (the
        // per-turn Stop firing re-rendering a growing transcript) carries new
        // tail information and must not be dropped — but re-extracting the
        // full document once per turn would be O(N²) LLM cost. Resolve by
        // REPLACING a still-pending job for the same document with the longer
        // snapshot: one job per drain cycle, always the latest content at
        // drain time. When nothing matching is pending (already drained),
        // fall through and enqueue fresh — it then absorbs further
        // extensions itself until the next drain.
        if verdict == SuppressVerdict::Extension {
            if let Some(replaced) = try_replace_pending_extension(
                &path,
                &agent_label,
                source,
                &text,
                &mode,
                &artifact_id,
                &session_json,
            )? {
                return Ok(replaced);
            }
        } else {
            // Pending-queue similarity check (prevents cross-session
            // duplicates that fall outside fingerprint_window_ms). Skipped
            // for extensions: a growing document is ~identical to its own
            // pending predecessor in the first 500 chars by construction,
            // and the precise prefix scan above already ran.
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
                        // Already queued with same or higher fidelity — the
                        // content reaches extraction via the pending job
                        // (codex R8 P2: NOT a drop, but also not exact).
                        return Ok(EnqueueOutcome::CoveredByPending);
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
            artifact_id,
            session_json,
            attempts: 0,
            next_attempt_at: None,
            created_at: Utc::now().to_rfc3339(),
        };
        append_jsonl(&path, &serde_json::to_string(&job)?)?;
        Ok(EnqueueOutcome::Enqueued)
    })
}

/// v1.2 audit F2: replace a still-pending job that the incoming text strictly
/// extends (same agent + source, pending text is a byte prefix of the
/// incoming text — the growing-transcript shape). Keeps the pending job's
/// queue position and id but swaps in the longer text, the latest artifact id
/// and session payload, and resets the retry counters (the content changed).
/// Returns Ok(None) when nothing pending matches. Caller holds the queue
/// advisory lock.
#[allow(clippy::too_many_arguments)]
fn try_replace_pending_extension(
    path: &std::path::Path,
    agent_label: &str,
    source: &str,
    text: &str,
    mode: &MemoryJobMode,
    artifact_id: &Option<String>,
    session_json: &Option<String>,
) -> anyhow::Result<Option<EnqueueOutcome>> {
    let queue_content = std::fs::read_to_string(path).unwrap_or_default();
    if queue_content.is_empty() {
        return Ok(None);
    }
    let mut lines: Vec<String> = queue_content.lines().map(|l| l.to_string()).collect();
    let mut replaced_at: Option<usize> = None;
    // Scan newest-first: the most recent pending snapshot is the one this
    // extension grew from.
    for idx in (0..lines.len()).rev() {
        let Ok(existing) = serde_json::from_str::<MemoryJob>(&lines[idx]) else {
            continue;
        };
        if existing.agent_label != agent_label || existing.source != source {
            continue;
        }
        if existing.text.len() < text.len() && text.starts_with(&existing.text) {
            let mut updated = existing;
            updated.mode = mode.clone();
            updated.text = text.to_string();
            updated.artifact_id = artifact_id.clone();
            updated.session_json = session_json.clone();
            updated.attempts = 0;
            updated.next_attempt_at = None;
            lines[idx] = serde_json::to_string(&updated)?;
            replaced_at = Some(idx);
            break;
        }
    }
    let Some(_) = replaced_at else {
        return Ok(None);
    };
    // Atomic rewrite (tmp + rename), mirroring save_recent_events.
    let tmp = path.with_extension("tmp-replace");
    std::fs::write(&tmp, lines.join("\n") + "\n")?;
    // codex R13 P2: the rename replaces the destination INODE — carry the
    // existing queue file's permissions onto the tmp so a restrictive mode
    // (e.g. 0600 from a prior umask) is never silently loosened to the
    // current process umask on queued memory text.
    #[cfg(unix)]
    if let Ok(meta) = std::fs::metadata(path) {
        let _ = std::fs::set_permissions(&tmp, meta.permissions());
    }
    // codex R5 P2: Windows rename fails when the destination exists — and in
    // this path it exists by definition. Best-effort remove first (same
    // pattern as the warmup staging swap); Unix rename overwrites atomically
    // and skips this.
    #[cfg(windows)]
    {
        let _ = std::fs::remove_file(path);
    }
    std::fs::rename(&tmp, path)?;
    Ok(Some(EnqueueOutcome::Replaced))
}

/// Unconditional worker-process spawn — no env guard, no cooldown. Used by
/// `drain_memory_queue` to chain a SUCCESSOR worker when it exits on the
/// per-run cap with work still pending (codex remediation R3 P2: the hooks
/// that enqueued during the drain already consumed their spawn attempts
/// against the held drain lock / cooldown, so without a successor the tail
/// sits until an unrelated future enqueue). Touches the spawn marker so the
/// normal cooldown accounting still sees the spawn.
fn spawn_worker_process(config: &ReinConfig) {
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
    // (Marker touch happens inside spawn_worker_process.)
    spawn_worker_process(config);
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
        merge_refinement: {
            let (mr_pending, mr_pending_issue) = diagnostic_count(
                &merge_refinement_queue_path(config),
                "merge_refinement_queue",
            );
            let (mr_inflight, mr_inflight_issue) = diagnostic_count(
                &merge_refinement_inflight_path(config),
                "merge_refinement_queue_inflight",
            );
            let (mr_dead, mr_dead_issue) = diagnostic_count(
                &merge_refinement_dead_letter_path(config),
                "merge_refinement_queue_dead",
            );
            let (mr_stats, mr_stats_issue) = diagnostic_stats(
                &merge_refinement_stats_path(config),
                "merge_refinement_worker_stats",
            );
            QueueDiagnostics {
                pending: mr_pending,
                inflight: mr_inflight,
                dead_letters: mr_dead,
                stats: mr_stats,
                issues: collect_issues([
                    mr_pending_issue,
                    mr_inflight_issue,
                    mr_dead_issue,
                    mr_stats_issue,
                ]),
            }
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
            .chain(read_cleanup_jobs(&inflight))
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
            .chain(read_dedup_jobs(&inflight))
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

// ---------------------------------------------------------------------------
// Merge-refinement queue
// ---------------------------------------------------------------------------

/// Queue an async LLM synthesis pass on a winner memory after a merge.
/// Fire-and-forget: errors are logged and suppressed so the hot store path is unaffected.
pub fn queue_merge_refinement_job(config: &ReinConfig, winner_id: String) {
    let job = MergeRefinementJob {
        id: ulid::Ulid::new().to_string(),
        winner_id: winner_id.clone(),
        attempts: 0,
        next_attempt_at: None,
        created_at: Utc::now().to_rfc3339(),
    };
    let path = merge_refinement_queue_path(config);
    let inflight = merge_refinement_inflight_path(config);
    let lock = merge_refinement_lock_path(config);
    let fingerprint = winner_id.clone();
    let result = with_advisory_lock(&lock, true, || {
        // Deduplicate: skip if a job for this winner_id is already pending or in-flight.
        for queued in read_merge_refinement_jobs(&path)
            .into_iter()
            .chain(read_merge_refinement_jobs(&inflight))
        {
            if queued.winner_id == fingerprint {
                return Ok(queued.id);
            }
        }
        append_jsonl(&path, &serde_json::to_string(&job)?)?;
        Ok(job.id.clone())
    });
    if let Err(e) = result {
        tracing::debug!("queue_merge_refinement_job failed for {winner_id}: {e}");
    }
    spawn_merge_refinement_worker(config);
}

pub fn spawn_merge_refinement_worker(config: &ReinConfig) {
    if std::env::var("REIN_MERGE_REFINEMENT_WORKER").as_deref() == Ok("1") {
        return;
    }
    if !should_spawn_worker(
        &merge_refinement_spawn_marker_path(config),
        config.async_memory.spawn_cooldown_ms,
    ) {
        return;
    }
    let _ = touch_worker_marker(&merge_refinement_spawn_marker_path(config));
    let Ok(exe) = std::env::current_exe() else {
        return;
    };
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("worker")
        .arg("merge-refinement-queue")
        .env("REIN_MERGE_REFINEMENT_WORKER", "1")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    let _ = cmd.spawn();
}

pub async fn drain_merge_refinement_queue(config: &ReinConfig) -> anyhow::Result<u32> {
    let path = merge_refinement_queue_path(config);
    let inflight = merge_refinement_inflight_path(config);
    let lock = merge_refinement_lock_path(config);
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
            tracing::warn!("merge_refinement flock failed: {}", err);
        }
        return Ok(0);
    }
    let result = drain_merge_refinement_queue_locked(config, &path, &inflight).await;
    let _ = unsafe { libc::flock(fd, libc::LOCK_UN) };
    drop(lock_file);
    result
}

async fn drain_merge_refinement_queue_locked(
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
        .filter_map(|line| serde_json::from_str::<MergeRefinementJob>(line).ok())
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
    let mut stats = load_merge_refinement_worker_stats(config);

    let split_at = config.async_memory.batch_size.min(ready.len());
    let ready_tail = ready.split_off(split_at);
    let to_process = ready;

    for job in to_process {
        match process_merge_refinement_job(config, job.clone()).await {
            Ok(done) => {
                processed += done;
                stats.processed += done as u64;
            }
            Err(e) => {
                tracing::warn!("merge_refinement worker job failed: {e}");
                if job.attempts + 1 >= config.async_memory.max_retries {
                    let _ = append_merge_refinement_dead_letter(config, &job, &e.to_string());
                    stats.dead_lettered += 1;
                } else {
                    remaining.push(reschedule_merge_refinement_job(config, job));
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
    let _ = save_merge_refinement_worker_stats(config, &stats);
    Ok(processed)
}

async fn process_merge_refinement_job(
    config: &ReinConfig,
    job: MergeRefinementJob,
) -> anyhow::Result<u32> {
    use crate::types::MemoryStore as _;
    let store = config.open_store()?;
    let memory = match store.get(&job.winner_id) {
        Ok(m) => m,
        Err(_) => {
            // Memory may have been deleted — treat as success to avoid dead-letter spam.
            tracing::debug!(
                "merge_refinement: winner {} not found, skipping",
                job.winner_id
            );
            return Ok(1);
        }
    };

    // Only refine if the content actually contains merge markers.
    if !memory.content.contains("[merged") {
        return Ok(1);
    }

    let refined = crate::extract::llm::llm_refine_merged_content(config, &memory.content).await?;
    if let Some(new_content) = refined {
        if new_content != memory.content {
            let mut updated = memory;
            updated.content = new_content;
            updated.summary = updated
                .content
                .chars()
                .take(crate::types::SUMMARY_MAX_CHARS)
                .collect();
            updated.updated_at = chrono::Utc::now();
            store.update(&updated)?;
            tracing::info!("merge_refinement: synthesized memory {}", job.winner_id);
        }
    }
    Ok(1)
}

fn read_merge_refinement_jobs(path: &std::path::Path) -> Vec<MergeRefinementJob> {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect()
}

fn reschedule_merge_refinement_job(
    config: &ReinConfig,
    mut job: MergeRefinementJob,
) -> MergeRefinementJob {
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

fn append_merge_refinement_dead_letter(
    config: &ReinConfig,
    job: &MergeRefinementJob,
    error: &str,
) -> anyhow::Result<()> {
    let entry = serde_json::json!({
        "job": job,
        "error": error,
        "failed_at": Utc::now().to_rfc3339(),
    });
    append_jsonl(
        &merge_refinement_dead_letter_path(config),
        &entry.to_string(),
    )
}

fn load_merge_refinement_worker_stats(config: &ReinConfig) -> WorkerStats {
    let path = merge_refinement_stats_path(config);
    let Ok(text) = std::fs::read_to_string(path) else {
        return WorkerStats::default();
    };
    serde_json::from_str(&text).unwrap_or_default()
}

fn save_merge_refinement_worker_stats(
    config: &ReinConfig,
    stats: &WorkerStats,
) -> anyhow::Result<()> {
    let path = merge_refinement_stats_path(config);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(stats)?)?;
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

fn merge_refinement_queue_path(config: &ReinConfig) -> std::path::PathBuf {
    project_scoped_path(config, "merge_refinement_queue")
}

fn merge_refinement_inflight_path(config: &ReinConfig) -> std::path::PathBuf {
    project_scoped_path(config, "merge_refinement_queue_inflight")
}

fn merge_refinement_lock_path(config: &ReinConfig) -> std::path::PathBuf {
    project_scoped_path(config, "merge_refinement_queue_lock")
}

fn merge_refinement_dead_letter_path(config: &ReinConfig) -> std::path::PathBuf {
    project_scoped_path(config, "merge_refinement_queue_dead")
}

fn merge_refinement_stats_path(config: &ReinConfig) -> std::path::PathBuf {
    project_scoped_path(config, "merge_refinement_worker_stats")
}

fn merge_refinement_spawn_marker_path(config: &ReinConfig) -> std::path::PathBuf {
    project_scoped_path(config, "merge_refinement_worker_spawn")
}

pub async fn drain_memory_queue(config: &ReinConfig) -> anyhow::Result<u32> {
    let path = queue_path(config);
    let inflight = inflight_path(config);
    // v1.2 audit F5: the drain previously held the ENQUEUE lock
    // (`lock_path`) for the entire async drain — up to 32 jobs of network
    // LLM extraction — while `_queue_memory_job` takes a BLOCKING flock on
    // the same file, so every hook enqueue stalled behind the worker
    // (easily past Claude Code's 60s hook timeout). Single-drainer mutual
    // exclusion now uses a DEDICATED drain lock held across the drain; the
    // shared enqueue lock is taken only around the brief claim phase
    // (recover + rename snapshot) inside drain_memory_queue_locked.
    let lock = drain_lock_path(config);
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
    //
    // codex v1.2 remediation R1 P1: with enqueues no longer blocked behind
    // the drain, a hook can append a job WHILE this worker is processing —
    // and that hook's spawn attempt is consumed immediately (suppressed by
    // the spawn cooldown or by this very drain lock), so without a recheck
    // the job would sit unprocessed until some future enqueue. Loop: when a
    // pass fully drained the ready set (processed < max_jobs_per_run) and
    // the queue file is non-empty again, run another pass. A pass that hit
    // the per-run cap exits as before (pre-existing batch semantics — the
    // write-back preserved the tail for the next worker), and a pass that
    // processed 0 (only deferred/future jobs remain) also exits.
    // codex remediation R3 P1: cap/liveness decisions key off JOBS CLAIMED
    // (and an explicit hit-cap signal) from each pass — `stored` counts
    // memories produced by process_job, which undercounts whenever jobs are
    // filtered/empty, and comparing it against max_jobs_per_run both
    // bypassed the LLM-job cap and mis-detected "fully drained".
    let mut total_stored: u32 = 0;
    // codex remediation R4 P2: max_jobs_per_run is a WORKER-LIFETIME budget
    // ("worker drains up to this many jobs before exiting"), enforced
    // cumulatively across every pass — including the F5/R2 re-check passes.
    // Without this, steady hook traffic kept one worker claiming fresh
    // arrivals forever instead of handing off after the run budget.
    let mut budget: usize = config.async_memory.max_jobs_per_run;
    let result = 'outer: loop {
        // Inner passes with the drain lock held; yields the last pass.
        let last_pass: DrainPass = loop {
            match drain_memory_queue_locked(config, &path, &inflight, budget).await {
                Ok(pass) => {
                    total_stored = total_stored.saturating_add(pass.stored);
                    budget = budget.saturating_sub(pass.jobs_claimed as usize);
                    if !pass.hit_cap
                        && pass.jobs_claimed > 0
                        && budget > 0
                        && queue_has_ready_job(&path)
                    {
                        continue;
                    }
                    break pass;
                }
                Err(e) => break 'outer Err(e),
            }
        };
        // Budget exhausted (or a pass left a ready tail behind): exit with
        // the tail preserved — the cap bounds THIS worker's LLM spend.
        // codex R3 P2: jobs appended during the drain consumed their spawn
        // attempts against our held lock / the cooldown, so chain an
        // unconditional successor worker (after releasing the lock) instead
        // of relying on a future enqueue.
        if last_pass.hit_cap || budget == 0 {
            let _ = unsafe { libc::flock(fd, libc::LOCK_UN) };
            // codex R9 P2: gate the successor on READY work (locked read) —
            // a budget exhausted by rescheduling failures alone leaves only
            // future-dated retries, and a successor spawned for those exits
            // immediately while its marker touch arms the cooldown against
            // the next REAL enqueue.
            // codex R5 P2: never chain a successor when the configured
            // budget is zero — max_jobs_per_run = 0 means the worker is
            // effectively disabled, and an unconditional (cooldown-free)
            // successor would spawn-loop forever on a non-empty queue.
            if config.async_memory.max_jobs_per_run > 0 && queue_has_ready_job_locked(config, &path)
            {
                spawn_worker_process(config);
            }
            break Ok(total_stored);
        }
        // codex remediation R2 P2 (+ R7 P2): the pre-release check races
        // hooks — a job appended after the inner loop's check but before our
        // LOCK_UN sees a held drain lock, its spawned worker exits, and the
        // job strands. Close the window by RELEASING first, then rechecking
        // for READY work specifically (R7: a metadata-length check spun a
        // false reclaim cycle on deferred-only queues AND missed the
        // deferred-only early exit stranding a fresh ready job): anything
        // ready appended before the recheck is visible to us (reclaim the
        // lock and run another cycle — the claim takes the entire ready
        // set, so cycles only repeat when NEW ready work keeps arriving);
        // anything appended after it finds the lock free, so its own
        // spawned worker proceeds.
        let _ = unsafe { libc::flock(fd, libc::LOCK_UN) };
        // codex R9 P2: locked read — serializes against a hook mid-append.
        if !queue_has_ready_job_locked(config, &path) {
            break Ok(total_stored);
        }
        let rc = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
        if rc != 0 {
            // Another worker holds the drain lock now — the pending job is
            // its to process.
            break Ok(total_stored);
        }
        // Reclaimed with the lock held — loop back for another full cycle.
    };

    // Explicitly unlock + drop (lock released when fd closes; harmless if
    // already unlocked on the no-reclaim exit paths).
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
    max_jobs: usize,
) -> anyhow::Result<DrainPass> {
    if max_jobs == 0 {
        return Ok(DrainPass::default());
    }
    // v1.2 audit F5: claim phase under the SHARED enqueue lock (brief —
    // recover + rename snapshot + read). Enqueues only ever wait for this,
    // not for the LLM processing below.
    let content = with_advisory_lock(&lock_path(config), true, || {
        recover_inflight(path, inflight)?;
        if !path.exists() {
            return Ok(None);
        }
        let meta = std::fs::metadata(path)?;
        if meta.len() == 0 {
            return Ok(None);
        }
        std::fs::rename(path, inflight)?;
        Ok(Some(std::fs::read_to_string(inflight).unwrap_or_default()))
    })?;
    let Some(content) = content else {
        return Ok(DrainPass::default());
    };
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

    // Split ready jobs: process first batch (bounded by the caller's
    // remaining worker budget — codex R4 P2), keep the rest.
    let split_at = max_jobs.min(ready.len());
    let ready_tail = ready.split_off(split_at);
    // codex remediation R3 P1: the caller's cap/liveness logic needs JOBS
    // claimed and whether the per-run cap left a ready tail behind —
    // `processed` (memories stored) is the wrong unit for both.
    let jobs_claimed = split_at as u32;
    let hit_cap = !ready_tail.is_empty();
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
    //
    // v1.2 audit F5 follow-through: the enqueue lock is NOT held during
    // processing anymore, so jobs may have been appended to `path` while we
    // worked. The write-back must MERGE them (remaining first, then the new
    // arrivals) under the enqueue lock — a plain rename would clobber every
    // job enqueued during the drain.
    if !remaining.is_empty() {
        with_advisory_lock(&lock_path(config), true, || {
            use std::io::Write;
            let newly_enqueued = std::fs::read_to_string(path).unwrap_or_default();
            let tmp_path = path.with_extension("jsonl.tmp");
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&tmp_path)?;
            for job in &remaining {
                writeln!(file, "{}", serde_json::to_string(job)?)?;
            }
            for line in newly_enqueued.lines().filter(|l| !l.trim().is_empty()) {
                writeln!(file, "{line}")?;
            }
            file.sync_all()?;
            // codex R5 P2: hooks may have recreated `path` during processing
            // (we hold the enqueue lock now, but the file can exist) and
            // Windows rename fails onto an existing destination.
            #[cfg(windows)]
            {
                let _ = std::fs::remove_file(path);
            }
            std::fs::rename(&tmp_path, path)?;
            Ok(())
        })?;
    }

    // Safe to delete inflight now — remaining jobs are persisted in queue.
    let _ = std::fs::remove_file(inflight);
    stats.last_run_at = Some(Utc::now().to_rfc3339());
    let _ = save_worker_stats(config, &stats);
    Ok(DrainPass {
        stored: processed,
        jobs_claimed,
        hit_cap,
    })
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

/// v1.2 audit F5: dedicated single-drainer lock, distinct from the enqueue
/// lock so hooks never block behind a worker's LLM processing.
fn drain_lock_path(config: &ReinConfig) -> std::path::PathBuf {
    project_scoped_path(config, "memory_queue_drain_lock")
}

/// Result of one locked drain pass (codex remediation R3 P1): the caller's
/// cap/liveness logic needs the number of queue JOBS claimed and an explicit
/// hit-cap signal — `stored` (memories produced) undercounts whenever jobs
/// extract to nothing and must only be used for reporting.
#[derive(Debug, Default, Clone, Copy)]
struct DrainPass {
    stored: u32,
    jobs_claimed: u32,
    hit_cap: bool,
}

/// True when the queue file holds at least one job whose retry timer has
/// elapsed (codex R7 P2: liveness rechecks must distinguish READY work from
/// deferred-only backlogs — a length check both spun false reclaim cycles on
/// deferred-only queues and let a fresh ready job strand behind the
/// deferred-only early exit). Unparseable lines count as ready so corrupt
/// entries still get a pass to dead-letter them.
fn queue_has_ready_job(path: &std::path::Path) -> bool {
    let Ok(content) = std::fs::read_to_string(path) else {
        return false;
    };
    let now = Utc::now();
    content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .any(|line| match serde_json::from_str::<MemoryJob>(line) {
            Ok(job) => job.next_attempt_at.is_none_or(|ts| ts <= now),
            Err(_) => true,
        })
}

/// `queue_has_ready_job` under the ENQUEUE lock (codex R9 P2): a hook holds
/// `lock_path` while creating + writing the queue file, so an unlocked read
/// can observe the file created but the JSONL line not yet written and
/// wrongly conclude "no ready work" — exactly the strand this recheck is
/// meant to close. Taking the lock serializes against in-flight appends.
fn queue_has_ready_job_locked(config: &ReinConfig, path: &std::path::Path) -> bool {
    with_advisory_lock(&lock_path(config), true, || Ok(queue_has_ready_job(path))).unwrap_or(false)
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
        // fsync before deleting the inflight source: the append may only live in
        // the page cache, so a crash between write and remove would orphan the jobs.
        file.sync_all()?;
        // Also sync the directory entry so the append is durable across fs crashes.
        if let Some(parent) = path.parent() {
            if let Ok(dir) = std::fs::File::open(parent) {
                let _ = dir.sync_all();
            }
        }
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
    is_full: bool,
    source_query: Option<&str>,
    text: &str,
) -> anyhow::Result<SuppressVerdict> {
    let path = recent_events_path(config);
    let now = Utc::now();
    let mut state = load_recent_events(config);
    let window_ms = config.async_memory.fingerprint_window_ms as i64;

    state
        .items
        .retain(|item| (now - item.created_at).num_milliseconds() <= window_ms);

    let normalized = normalized_event_text(source, source_query, text);
    let normalized_len = normalized.chars().count();
    let preview: String = normalized.chars().take(2000).collect();
    let fingerprint = sha256_hex(&preview);
    // Cross-agent exact-duplicate fingerprint: hash the RAW (already
    // secret-redacted) event text verbatim — NOT the lossy
    // `normalized_event_text`, which lowercases and strips punctuation /
    // operators (so `if x == y` and `if x != y` would collide). Raw bytes give
    // a true content identity that is:
    //   - exact   → distinct code-bearing events never collide;
    //   - full    → no truncation-prefix collision;
    //   - source-agnostic → the same buffered content re-queued by different
    //     hooks (mid-session `hook_post` flush vs `hook_stop` /
    //     `hook_stop_incremental`) hashes identically and IS caught.
    // Byte-identical text across agent labels within the window can only be a
    // double-capture, never two independent observations.
    let full_fingerprint = sha256_hex(text);

    let mut saw_duplicate = false;
    let mut saw_extension = false;
    for item in state.items.iter() {
        // v0.37 hook-dup fix — Full-over-Quick, applied UNIFORMLY first: an
        // incoming Full job is never suppressed by a stored Quick entry on ANY
        // match path (mirrors the pending-queue exception). A Full extraction
        // is higher fidelity than the Quick one it may share content with, so
        // it must always be allowed through.
        if is_full && !item.is_full {
            continue;
        }
        // FULL-content, cross-agent exact match: catches the post-flush + stop
        // re-queue of identical buffered content even when the two firings
        // carry different hook sources / agent labels. Hashing the full text
        // (not the truncated preview) means a shared preamble with a differing
        // tail cannot false-suppress a distinct event. Guarded on non-empty so
        // legacy cache rows (no `full_fingerprint`) never cross-match.
        if !item.full_fingerprint.is_empty() && item.full_fingerprint == full_fingerprint {
            saw_duplicate = true;
            break;
        }
        // Within-agent path: truncated-preview fingerprint + fuzzy similarity,
        // scoped to the same agent so genuinely distinct agents' similar
        // content is not cross-suppressed.
        if item.agent_label != agent_label {
            continue;
        }
        if is_subagent && !item.agent_label.contains(":") {
            continue;
        }
        // v1.2 audit F2 (High): a strict prefix EXTENSION is the same growing
        // document with NEW tail content (the per-turn Stop firing shape:
        // each firing re-renders the full transcript, so successive texts are
        // prefix-extending). Suppressing it as a duplicate permanently lost
        // every turn after the first capture in a continuously-active session
        // — each suppressed firing re-pushed a fresh-timestamp entry with the
        // SAME prefix fingerprint, so the window self-renewed forever.
        // Extension test: stored entry knows its full normalized length
        // (0 = legacy row → unknown → keep old suppress behavior), incoming
        // is strictly longer, and incoming's prefix at the stored PREVIEW
        // length is char-identical to the stored preview.
        let stored_preview_len = item.preview.chars().count();
        let is_extension = item.normalized_len > 0
            && normalized_len > item.normalized_len
            && stored_preview_len > 0
            && normalized
                .chars()
                .take(stored_preview_len)
                .eq(item.preview.chars());
        if item.fingerprint == fingerprint
            || crate::extract::similarity(&item.preview, &preview) > 0.94
        {
            if is_extension {
                saw_extension = true;
                continue;
            }
            saw_duplicate = true;
            break;
        }
    }
    let verdict = if saw_duplicate {
        SuppressVerdict::Duplicate
    } else if saw_extension {
        SuppressVerdict::Extension
    } else {
        SuppressVerdict::New
    };

    state.items.push(RecentEventEntry {
        fingerprint,
        full_fingerprint,
        is_full,
        normalized_len,
        preview,
        agent_label: agent_label.to_string(),
        created_at: now,
    });
    if state.items.len() > config.async_memory.recent_event_cache_size {
        let start = state.items.len() - config.async_memory.recent_event_cache_size;
        state.items = state.items.split_off(start);
    }
    save_recent_events(config, &path, &state)?;
    Ok(verdict)
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
    fn raw_full_fingerprint_distinguishes_shared_preview_different_tail() {
        // v0.37 hook-dup safety: two events sharing the first 2000 normalized
        // chars but differing in the tail collide on the truncated preview
        // fingerprint yet must produce DIFFERENT RAW full fingerprints — so the
        // cross-agent exact-dup check (which keys off the raw full fingerprint)
        // can never false-suppress a distinct long event with a shared preamble.
        let shared = "x ".repeat(1500); // ~3000 chars >> the 2000-char preview cap
        let a = format!("{shared} alpha tail");
        let b = format!("{shared} beta tail");
        let a_preview: String = normalized_event_text("hook_post", None, &a)
            .chars()
            .take(2000)
            .collect();
        let b_preview: String = normalized_event_text("hook_post", None, &b)
            .chars()
            .take(2000)
            .collect();
        assert_eq!(
            sha256_hex(&a_preview),
            sha256_hex(&b_preview),
            "previews must collide on the shared 2000-char prefix"
        );
        assert_ne!(
            sha256_hex(&a),
            sha256_hex(&b),
            "raw full fingerprints must differ on the distinct tails"
        );
    }

    #[test]
    fn raw_full_fingerprint_preserves_code_operators() {
        // The raw (unnormalized) hash must distinguish code-bearing events that
        // differ only in operators — `normalized_event_text` strips `==`/`!=`
        // to the same text, so the cross-agent exact path MUST hash raw bytes.
        let a = "if x == y { merge() }";
        let b = "if x != y { merge() }";
        assert_eq!(
            normalized_event_text("", None, a),
            normalized_event_text("", None, b),
            "normalization collapses the operators (why raw hashing is required)"
        );
        assert_ne!(
            sha256_hex(a),
            sha256_hex(b),
            "raw full fingerprints must NOT collide on distinct operators"
        );
    }

    fn isolated_config() -> (ReinConfig, tempfile::TempDir) {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let mut config = ReinConfig::default();
        config.database.path = tmp
            .path()
            .join("queue-test.db")
            .to_string_lossy()
            .into_owned();
        config.hooks.buffer_dir = tmp.path().join("buffer").to_string_lossy().into_owned();
        (config, tmp)
    }

    /// v1.2 audit F2 (High): the per-turn Stop firing shape — successive
    /// prefix-extending transcripts — must classify as Extension (new tail
    /// content), while a re-send of identical content stays Duplicate.
    /// Pre-fix, the extension was suppressed AND re-pushed a fresh-timestamp
    /// entry with the same prefix fingerprint, so a continuously-active
    /// session never ingested anything past the first capture.
    #[test]
    fn suppressor_classifies_prefix_extension_vs_duplicate() {
        let (config, _tmp) = isolated_config();
        let base =
            "User: we decided to use postgres for billing because sqlite locks.\n".repeat(40); // well past the 2000-char preview cap
        let extended = format!("{base}Assistant: noted, migration plan saved.\n");
        let extended_more = format!("{extended}User: also pin MSRV at 1.86.\n");

        let v1 = suppress_duplicate_event(&config, "hook_stop", "main", false, true, None, &base)
            .unwrap();
        assert_eq!(v1, SuppressVerdict::New);

        let v2 = suppress_duplicate_event(&config, "hook_stop", "main", false, true, None, &base)
            .unwrap();
        assert_eq!(
            v2,
            SuppressVerdict::Duplicate,
            "identical re-send stays duplicate"
        );

        let v3 =
            suppress_duplicate_event(&config, "hook_stop", "main", false, true, None, &extended)
                .unwrap();
        assert_eq!(
            v3,
            SuppressVerdict::Extension,
            "longer prefix-extension must pass"
        );

        let v4 = suppress_duplicate_event(
            &config,
            "hook_stop",
            "main",
            false,
            true,
            None,
            &extended_more,
        )
        .unwrap();
        assert_eq!(
            v4,
            SuppressVerdict::Extension,
            "every further growth step is again an extension"
        );

        let v5 = suppress_duplicate_event(
            &config,
            "hook_stop",
            "main",
            false,
            true,
            None,
            &extended_more,
        )
        .unwrap();
        assert_eq!(
            v5,
            SuppressVerdict::Duplicate,
            "re-send of the latest snapshot is a duplicate again"
        );
    }

    /// v1.2 audit F2: pending-queue replacement — an extension swaps the
    /// still-pending shorter snapshot in place (same id, reset attempts,
    /// latest text/session payload), leaving unrelated jobs untouched.
    #[test]
    fn pending_extension_replaces_matching_job_in_place() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let path = tmp.path().join("memory_queue");
        let job = |id: &str, agent: &str, text: &str| MemoryJob {
            id: id.into(),
            mode: MemoryJobMode::Full,
            source: "ingest_session".into(),
            source_label: "source:main-agent".into(),
            agent_label: agent.into(),
            is_subagent: false,
            priority: 95,
            source_query: None,
            text: text.into(),
            artifact_id: Some("artifact-old".into()),
            session_json: Some("{\"old\":true}".into()),
            attempts: 2,
            next_attempt_at: None,
            created_at: Utc::now().to_rfc3339(),
        };
        let lines = [
            serde_json::to_string(&job("other", "subagent:x", "unrelated text")).unwrap(),
            serde_json::to_string(&job("target", "main", "session transcript v1")).unwrap(),
        ];
        std::fs::write(&path, lines.join("\n") + "\n").unwrap();

        let outcome = try_replace_pending_extension(
            &path,
            "main",
            "ingest_session",
            "session transcript v1 plus a new turn",
            &MemoryJobMode::Full,
            &Some("artifact-new".into()),
            &Some("{\"new\":true}".into()),
        )
        .unwrap();
        assert_eq!(outcome, Some(EnqueueOutcome::Replaced));

        let content = std::fs::read_to_string(&path).unwrap();
        let jobs: Vec<MemoryJob> = content
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(jobs.len(), 2, "no jobs added or lost");
        assert_eq!(jobs[0].text, "unrelated text", "unrelated job untouched");
        let replaced = &jobs[1];
        assert_eq!(replaced.id, "target", "queue position + id preserved");
        assert_eq!(replaced.text, "session transcript v1 plus a new turn");
        assert_eq!(replaced.artifact_id.as_deref(), Some("artifact-new"));
        assert_eq!(replaced.session_json.as_deref(), Some("{\"new\":true}"));
        assert_eq!(replaced.attempts, 0, "retry counters reset for new content");

        // Non-extension (different document) must not match anything.
        let none = try_replace_pending_extension(
            &path,
            "main",
            "ingest_session",
            "a completely different document",
            &MemoryJobMode::Full,
            &None,
            &None,
        )
        .unwrap();
        assert_eq!(none, None);
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
