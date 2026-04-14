//! Hook system for Claude Code integration.
//!
//! Submodules:
//! - `parsing` — JSON payload extraction and transcript processing
//! - `buffer` — Session buffer I/O and lifecycle
//! - `persist` — durable store + working-set persistence
//! - `queue` — async memory queue and worker
//! - `scoring` — Signal scoring and filtering
//! - `working_set` — project-scoped session working surface for prompt injection

pub mod buffer;
pub mod parsing;
pub mod persist;
pub mod queue;
pub mod scoring;
pub mod working_set;

use crate::config::ReinConfig;

use self::buffer::{
    adaptive_flush_threshold, append_to_buffer, cleanup_stale_buffers, clear_flush_marker,
    flush_count, mark_flushed, read_and_clear_buffer, session_buffer_path,
};
use self::parsing::*;
use self::queue::*;
use self::scoring::{extract_signal_windows, worth_extracting};

// ---------------------------------------------------------------------------
// Hook implementations
// ---------------------------------------------------------------------------

/// Layer 0: PostToolUse — buffer + content-triggered mid-session extraction.
pub async fn hook_post(config: &ReinConfig) -> anyhow::Result<()> {
    let input = std::io::read_to_string(std::io::stdin())?;
    let text = extract_hook_text(&input);
    if text.is_empty() {
        return Ok(());
    }
    let (agent_label, is_subagent) = classify_hook_agent(&input);

    let buf_path = session_buffer_path(config, &input);
    let source_name = if is_subagent { "post:subagent" } else { "post" };
    let _ = append_to_buffer(&buf_path, &text, source_name);

    if !worth_extracting(&text) {
        return Ok(());
    }

    let base_threshold = config.hooks.buffer_flush_threshold;
    let threshold = if base_threshold > 0 {
        let adaptive = adaptive_flush_threshold(base_threshold, &buf_path);
        if is_subagent {
            adaptive.saturating_mul(2)
        } else {
            adaptive
        }
    } else {
        0
    };

    if threshold > 0 {
        let buf_size = std::fs::metadata(&buf_path)
            .map(|m| m.len() as usize)
            .unwrap_or(0);
        if buf_size >= threshold {
            tracing::info!(
                "buffer reached {}B (threshold {}B), triggering mid-session extraction",
                buf_size,
                threshold
            );
            let buffered = read_and_clear_buffer(&buf_path);
            if !buffered.is_empty() {
                let combined = buffered
                    .iter()
                    .filter(|t| !looks_like_secret(t))
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("\n---\n");
                if !combined.is_empty() {
                    let mode = if is_subagent {
                        MemoryJobMode::Quick
                    } else {
                        MemoryJobMode::Full
                    };
                    let priority = if is_subagent { 10 } else { 40 };
                    if let Err(e) = queue_memory_job(
                        config,
                        mode,
                        "hook_post",
                        if is_subagent {
                            "source:subagent"
                        } else {
                            "source:main-agent"
                        },
                        agent_label.clone(),
                        is_subagent,
                        priority,
                        None,
                        combined,
                    ) {
                        eprintln!("rein: failed to queue memory job: {e}");
                    } else {
                        mark_flushed(&buf_path);
                    }
                    spawn_memory_worker(config);
                }
            }
        }
    }
    Ok(())
}

/// Layer 1: PreCompact — LLM extraction + buffer.
pub async fn hook_compact(config: &ReinConfig) -> anyhow::Result<()> {
    let input = std::io::read_to_string(std::io::stdin())?;
    let text = extract_hook_text(&input);
    if text.is_empty() {
        return Ok(());
    }
    let (agent_label, is_subagent) = classify_hook_agent(&input);

    if let Some(mut session) = extract_hook_session_ingest(&input) {
        session.artifact_kind = "compact".to_string();
        session.source_label = Some("hook_compact".to_string());
        session.compact_summary = Some(text.clone());
        let _ = crate::ops::queue_ingest_session(config, &session, Some(&agent_label), is_subagent);
    } else {
        let priority = if is_subagent { 25 } else { 90 };
        let _ = queue_memory_job(
            config,
            MemoryJobMode::Quick,
            "hook_compact",
            if is_subagent {
                "source:subagent"
            } else {
                "source:main-agent"
            },
            agent_label,
            is_subagent,
            priority,
            None,
            text.clone(),
        );
        spawn_memory_worker(config);
    }
    let buf_path = session_buffer_path(config, &input);
    let _ = append_to_buffer(&buf_path, &text, "compact");
    Ok(())
}

/// Layer 2: UserPromptSubmit — currently a no-op.
/// Injection is handled by the rein MCP prompt hook, not by this code path.
pub async fn hook_prompt(config: &ReinConfig) -> anyhow::Result<()> {
    let _ = config;
    Ok(())
}

/// Layer 3: Stop — full knowledge extraction on session end.
pub async fn hook_stop(config: &ReinConfig) -> anyhow::Result<()> {
    cleanup_stale_buffers(config);

    let input = std::io::read_to_string(std::io::stdin())?;
    if input.trim().is_empty() {
        return Ok(());
    }
    let (agent_label, is_subagent) = classify_hook_agent(&input);

    let turn_count = count_transcript_turns(&input);
    let min_turns = config.hooks.min_turns;
    if turn_count > 0 && turn_count < min_turns {
        return Ok(());
    }

    let has_llm = crate::extract::llm::create_extractor(config).is_some();
    let text = if has_llm {
        extract_hook_text_for_llm(&input)
    } else {
        extract_hook_text(&input)
    };

    if turn_count == 0 && text.lines().count() < min_turns {
        return Ok(());
    }

    let buf_path = session_buffer_path(config, &input);
    let prior_flushes = flush_count(&buf_path);
    let buffered = read_and_clear_buffer(&buf_path);
    clear_flush_marker(&buf_path);

    // When mid-session flushes already extracted the bulk of this session's content,
    // skip re-extracting the full transcript to avoid duplication.
    // Instead: only process the remaining buffered content (since last flush) +
    // episode synthesis (which reads session metadata, not raw transcript text).
    let incremental_mode = prior_flushes > 0;

    if has_llm {
        if let Some(mut session) = extract_hook_session_ingest(&input) {
            if !buffered.is_empty() {
                session.tool_outputs = buffered.clone();
            }
            if incremental_mode {
                // Bulk of this session's content was already extracted incrementally.
                // Only clear turns when there is fallback content for episode synthesis
                // (buffered tool outputs OR a compact_summary from Claude's compaction).
                // If neither exists, keeping the turns prevents silently losing the episode
                // record and any stop-time memories on sessions with no summary fields.
                let has_fallback =
                    !buffered.is_empty() || session.compact_summary.is_some();
                if has_fallback {
                    session.turns.clear();
                }
            }
            let _ =
                crate::ops::queue_ingest_session(config, &session, Some(&agent_label), is_subagent);
        } else if incremental_mode {
            // Session was already extracted incrementally; only process remaining buffer.
            if !buffered.is_empty() {
                let combined = buffered
                    .iter()
                    .filter(|t| !looks_like_secret(t))
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("\n---\n");
                if !combined.is_empty() {
                    let priority = if is_subagent { 25 } else { 60 };
                    if let Err(e) = queue_memory_job(
                        config,
                        MemoryJobMode::Quick,
                        "hook_stop_incremental",
                        if is_subagent {
                            "source:subagent"
                        } else {
                            "source:main-agent"
                        },
                        agent_label.clone(),
                        is_subagent,
                        priority,
                        None,
                        combined,
                    ) {
                        eprintln!("rein: failed to queue incremental session tail: {e}");
                    } else {
                        spawn_memory_worker(config);
                    }
                }
            }
        } else {
            let combined = if buffered.is_empty() {
                text.lines()
                    .filter(|l| !looks_like_secret(l))
                    .collect::<Vec<_>>()
                    .join("\n")
            } else {
                let transcript = text
                    .lines()
                    .filter(|l| !looks_like_secret(l))
                    .collect::<Vec<_>>()
                    .join("\n");
                format!(
                    "{}\n\n--- Buffered tool outputs ---\n{}",
                    transcript,
                    buffered.join("\n---\n")
                )
            };
            if combined.is_empty() {
                return Ok(());
            }

            let priority = if is_subagent { 35 } else { 100 };
            if let Err(e) = queue_memory_job(
                config,
                MemoryJobMode::Full,
                "hook_stop",
                if is_subagent {
                    "source:subagent"
                } else {
                    "source:main-agent"
                },
                agent_label.clone(),
                is_subagent,
                priority,
                None,
                combined,
            ) {
                eprintln!("rein: failed to queue session memory: {e}");
                return Ok(());
            }
            spawn_memory_worker(config);
        }
        eprintln!("rein: queued session memory processing");
    } else {
        let windows = extract_signal_windows(&text, config);
        if windows.is_empty() {
            return Ok(());
        }

        let max_items = config.hooks.max_items_per_session;
        let combined: String = windows
            .iter()
            .take(max_items)
            .filter(|w| !looks_like_secret(w))
            .cloned()
            .collect::<Vec<_>>()
            .join("\n---\n");
        if combined.is_empty() {
            return Ok(());
        }

        let priority = if is_subagent { 30 } else { 95 };
        if let Err(e) = queue_memory_job(
            config,
            MemoryJobMode::Quick,
            "hook_stop_fallback",
            if is_subagent {
                "source:subagent"
            } else {
                "source:main-agent"
            },
            agent_label,
            is_subagent,
            priority,
            None,
            combined,
        ) {
            eprintln!("rein: failed to queue session memory: {e}");
            return Ok(());
        }
        spawn_memory_worker(config);
        eprintln!("rein: queued session memory processing");
    }
    Ok(())
}
