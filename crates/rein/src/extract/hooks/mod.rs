//! Hook system for agent integrations such as Claude Code and Codex CLI.
//!
//! Submodules:
//! - `parsing` — JSON payload extraction and transcript processing
//! - `buffer` — Session buffer I/O and lifecycle
//! - `persist` — durable store + working-set persistence
//! - `queue` — async memory queue and worker
//! - `scoring` — Signal scoring and filtering
//! - `working_set` — project-scoped memory surface for explicit recall and extraction context

pub mod buffer;
pub mod parsing;
pub mod persist;
pub mod queue;
pub mod scoring;
pub mod working_set;

use crate::config::ReinConfig;

use self::buffer::{
    adaptive_flush_threshold, append_to_buffer, cleanup_stale_buffers, clear_flush_marker,
    clear_flushed_ledger, filter_turns_against_flushed, flush_count, mark_flushed,
    read_and_clear_buffer, read_flushed_hashes, record_flushed_content, session_buffer_path,
};
use self::parsing::*;
use self::queue::*;
use self::scoring::{extract_signal_windows, worth_extracting};
use self::working_set::WorkingSetItem;

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
                // codex v1.2 R1 P2: split filter from join — the ledger below
                // must record ONLY the items actually queued for extraction.
                // Recording the unfiltered slice would let a secret-looking
                // false positive (never extracted here) strip its verbatim
                // occurrence from the stop-time turns, losing the fact.
                let flushable: Vec<String> = buffered
                    .iter()
                    .filter(|t| !looks_like_secret(t))
                    .cloned()
                    .collect();
                let combined = flushable.join("\n---\n");
                if !combined.is_empty() {
                    let mode = if is_subagent {
                        MemoryJobMode::Quick
                    } else {
                        MemoryJobMode::Full
                    };
                    let priority = if is_subagent { 10 } else { 40 };
                    match queue_memory_job(
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
                        Err(e) => {
                            eprintln!("rein: failed to queue memory job: {e}");
                        }
                        // v1.2 audit F3: gate the flush marker AND the ledger
                        // on the job actually being accepted for extraction.
                        // A suppressed duplicate was NOT queued — recording it
                        // as flushed would let hook_stop strip its verbatim
                        // occurrence from the transcript turns even though it
                        // was never extracted (losing the fact), and the
                        // flush marker would wrongly flip the session into
                        // incremental mode.
                        //
                        // codex R8 P2: ledger-record ONLY when the queued
                        // text carries our exact content (Enqueued/Replaced).
                        // A CoveredByPending match routes the content to
                        // extraction via a merely-SIMILAR pending job, so the
                        // flush marker is recorded but our exact lines are
                        // not — the pending job may never extract them
                        // verbatim (same false-ledger hazard as F3).
                        Ok(outcome) if outcome.accepted() => {
                            mark_flushed(&buf_path);
                            if outcome.carries_exact_content() {
                                // v1.2: remember WHAT was flushed (content
                                // hashes) so hook_stop can filter verbatim
                                // re-occurrences out of the transcript turns
                                // instead of re-extracting them.
                                record_flushed_content(&buf_path, &flushable);
                            }
                        }
                        Ok(_) => {}
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

/// Codex SessionStart — optionally add bounded project memory context.
pub async fn hook_session_start(config: &ReinConfig) -> anyhow::Result<()> {
    let input = std::io::read_to_string(std::io::stdin())?;
    if parse_codex_event_payload(&input, "SessionStart").is_none()
        || !config.hooks.codex.inject_session_context
    {
        return Ok(());
    }

    let mut items = working_set::load_always_on_index(config);
    if items.is_empty() {
        items = working_set::load_working_set(config);
    }
    if let Some(context) = format_codex_context(
        "Rein project memory",
        items,
        config.hooks.codex.max_additional_context_chars,
        config.async_memory.selection_limit,
    ) {
        print_codex_json_output(&codex_additional_context_output("SessionStart", &context))?;
    }
    Ok(())
}

/// Codex PreToolUse — conservative deny-only guardrails.
pub async fn hook_pre_tool_use(config: &ReinConfig) -> anyhow::Result<()> {
    let input = std::io::read_to_string(std::io::stdin())?;
    let Some(json) = parse_codex_event_payload(&input, "PreToolUse") else {
        return Ok(());
    };
    if !config.hooks.codex.guardrails_enabled {
        return Ok(());
    }
    if let Some(reason) = codex_guardrail_reason(&json) {
        print_codex_json_output(&codex_pre_tool_use_deny(&reason))?;
    }
    Ok(())
}

/// Codex PermissionRequest — conservative deny-only guardrails.
pub async fn hook_permission_request(config: &ReinConfig) -> anyhow::Result<()> {
    let input = std::io::read_to_string(std::io::stdin())?;
    let Some(json) = parse_codex_event_payload(&input, "PermissionRequest") else {
        return Ok(());
    };
    if !config.hooks.codex.guardrails_enabled {
        return Ok(());
    }
    if let Some(reason) = codex_guardrail_reason(&json) {
        print_codex_json_output(&codex_permission_request_deny(&reason))?;
    }
    Ok(())
}

/// Codex UserPromptSubmit — optionally add bounded, relevant Rein memory context.
pub async fn hook_prompt(config: &ReinConfig) -> anyhow::Result<()> {
    let input = std::io::read_to_string(std::io::stdin())?;
    let Some(json) = parse_codex_event_payload(&input, "UserPromptSubmit") else {
        return Ok(());
    };
    if !config.hooks.codex.inject_prompt_context {
        return Ok(());
    }

    let prompt = json.get("prompt").and_then(|v| v.as_str()).unwrap_or("");
    let items = working_set::select_relevant_items(config, prompt);
    if let Some(context) = format_codex_context(
        "Rein memory",
        items,
        config.hooks.codex.max_additional_context_chars,
        config.async_memory.selection_limit,
    ) {
        print_codex_json_output(&codex_additional_context_output(
            "UserPromptSubmit",
            &context,
        ))?;
    }
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
    // v1.2: load the flushed-content ledger BEFORE clearing session state —
    // the incremental no-fallback path below filters transcript turns
    // against it (content-level dedup of already-extracted tool output).
    let flushed_hashes = read_flushed_hashes(&buf_path);
    clear_flush_marker(&buf_path);
    clear_flushed_ledger(&buf_path);

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
                // v1.2 content-level dedup (audit F18 unified both branches):
                // the turns historically either re-fed the FULL transcript to
                // the LLM (no-fallback branch — re-extracting content the
                // mid-session flushes already covered) or were CLEARED
                // wholesale (fallback branch — discarding conversation facts
                // that never entered the buffer: only PostToolUse tool output
                // is buffered, so buffered content is not a substitute for
                // conversation turns). Both branches now strip exactly what
                // the ledger proves was flushed — verbatim whole-turn or
                // long-line matches — and keep every conversation fact.
                // (The v0.38 offset approach was rejected: conversation turns
                // never enter the buffer, so truncation would drop
                // never-extracted facts.)
                session.turns = filter_turns_against_flushed(
                    std::mem::take(&mut session.turns),
                    &flushed_hashes,
                );
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

fn parse_codex_event_payload(input: &str, expected_event: &str) -> Option<serde_json::Value> {
    let json = serde_json::from_str::<serde_json::Value>(input).ok()?;
    let event = json.get("hook_event_name").and_then(|v| v.as_str())?;
    (event == expected_event).then_some(json)
}

fn print_codex_json_output(value: &serde_json::Value) -> anyhow::Result<()> {
    println!("{}", serde_json::to_string(value)?);
    Ok(())
}

fn codex_additional_context_output(event: &str, context: &str) -> serde_json::Value {
    serde_json::json!({
        "hookSpecificOutput": {
            "hookEventName": event,
            "additionalContext": context
        }
    })
}

fn codex_pre_tool_use_deny(reason: &str) -> serde_json::Value {
    serde_json::json!({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason
        }
    })
}

fn codex_permission_request_deny(reason: &str) -> serde_json::Value {
    serde_json::json!({
        "hookSpecificOutput": {
            "hookEventName": "PermissionRequest",
            "decision": {
                "behavior": "deny",
                "message": reason
            }
        }
    })
}

fn format_codex_context<I>(
    title: &str,
    items: I,
    max_chars: usize,
    max_items: usize,
) -> Option<String>
where
    I: IntoIterator<Item = WorkingSetItem>,
{
    if max_chars == 0 || max_items == 0 {
        return None;
    }

    let mut lines = vec![format!("{title} (background; user/repo instructions win):")];
    for item in items.into_iter().take(max_items) {
        let topic = cap_chars(&clean_codex_context_field(&item.topic), 48);
        let summary = cap_chars(&clean_codex_context_field(&item.summary), 96);
        let detail = cap_chars(&clean_codex_context_field(&item.detail), 160);
        if summary.is_empty() && detail.is_empty() {
            continue;
        }
        let label = if topic.is_empty() { "memory" } else { &topic };
        let body = if summary.is_empty() || context_detail_is_redundant(&summary, &detail) {
            detail
        } else if detail.is_empty() {
            summary
        } else {
            format!("{summary}; {detail}")
        };
        lines.push(format!("- {label}: {body}"));
    }

    if lines.len() <= 1 {
        return None;
    }
    Some(cap_chars(&lines.join("\n"), max_chars))
}

fn clean_codex_context_field(text: &str) -> String {
    redact_secrets(text)
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(str::trim)
        .collect::<Vec<_>>()
        .join(" ")
}

fn context_detail_is_redundant(summary: &str, detail: &str) -> bool {
    if summary.is_empty() || detail.is_empty() {
        return false;
    }
    if summary.eq_ignore_ascii_case(detail) {
        return true;
    }
    if summary.chars().count() < 24 {
        return false;
    }
    detail.to_lowercase().contains(&summary.to_lowercase())
}

fn cap_chars(text: &str, max_chars: usize) -> String {
    let char_count = text.chars().count();
    if char_count <= max_chars {
        return text.to_string();
    }
    if max_chars <= 3 {
        return text.chars().take(max_chars).collect();
    }
    let mut out: String = text.chars().take(max_chars - 3).collect();
    out.push_str("...");
    out
}

fn codex_guardrail_reason(json: &serde_json::Value) -> Option<String> {
    let command = codex_tool_command(json)?;
    let tool_name = json
        .get("tool_name")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let shell_like_tool = ["bash", "shell", "terminal", "exec", "command"]
        .iter()
        .any(|needle| tool_name.contains(needle));
    if !shell_like_tool && tool_name != "bash" {
        return None;
    }

    let normalized = normalize_shell_command(&command);
    if normalized.contains(":(){") || normalized.contains(":(){ :|:& };:") {
        return Some("blocked fork-bomb shell pattern".to_string());
    }
    if rm_force_targets_dangerous(&normalized) {
        return Some("blocked high-risk rm -rf target".to_string());
    }
    if contains_dangerous_disk_command(&normalized) {
        return Some("blocked high-risk disk mutation command".to_string());
    }
    if exfiltrates_secret_material(&normalized) {
        return Some("blocked command that appears to transmit secret material".to_string());
    }
    None
}

fn codex_tool_command(json: &serde_json::Value) -> Option<String> {
    let input = json.get("tool_input")?;
    match input {
        serde_json::Value::String(text) => non_empty_string(text),
        serde_json::Value::Object(map) => ["command", "cmd", "script", "input"]
            .iter()
            .find_map(|key| map.get(*key).and_then(codex_value_to_string)),
        value => codex_value_to_string(value),
    }
}

fn codex_value_to_string(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(text) => non_empty_string(text),
        serde_json::Value::Array(items) => {
            let joined = items
                .iter()
                .filter_map(codex_value_to_string)
                .collect::<Vec<_>>()
                .join(" ");
            non_empty_string(&joined)
        }
        serde_json::Value::Object(_) => serde_json::to_string(value)
            .ok()
            .and_then(|text| non_empty_string(&text)),
        serde_json::Value::Null => None,
        serde_json::Value::Bool(_) | serde_json::Value::Number(_) => {
            non_empty_string(&value.to_string())
        }
    }
}

fn non_empty_string(text: &str) -> Option<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn normalize_shell_command(command: &str) -> String {
    command
        .replace(['"', '\''], "")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
}

fn rm_force_targets_dangerous(command: &str) -> bool {
    let tokens = command.split_whitespace().collect::<Vec<_>>();
    for (idx, token) in tokens.iter().enumerate() {
        if *token != "rm" && !token.ends_with("/rm") {
            continue;
        }
        let mut recursive = false;
        let mut force = false;
        for target in tokens.iter().skip(idx + 1) {
            if target.starts_with('-') {
                recursive |= target.contains('r') || target.contains('R');
                force |= target.contains('f');
                continue;
            }
            if recursive && force && is_dangerous_rm_target(target) {
                return true;
            }
        }
    }
    false
}

fn is_dangerous_rm_target(target: &str) -> bool {
    let target = target.trim_end_matches(';');
    if matches!(
        target,
        "/" | "/*"
            | "."
            | "./"
            | "./*"
            | ".."
            | "../"
            | "../*"
            | "~"
            | "~/"
            | "~/*"
            | "$home"
            | "$home/"
            | "$home/*"
            | "${home}"
            | "${home}/"
            | "${home}/*"
            | "*"
    ) {
        return true;
    }
    std::env::var("HOME")
        .ok()
        .map(|home| target == home || target == format!("{home}/") || target == format!("{home}/*"))
        .unwrap_or(false)
}

fn contains_dangerous_disk_command(command: &str) -> bool {
    command.contains("mkfs ")
        || command.contains("mkfs.")
        || command.starts_with("mkfs")
        || command.contains("diskutil erase")
        || command.contains("diskutil partition")
        || ((command.starts_with("dd ") || command.contains(" dd "))
            && command.contains("of=/dev/"))
}

fn exfiltrates_secret_material(command: &str) -> bool {
    let transmits = ["curl ", "wget ", "nc ", "netcat ", "scp ", "rsync "]
        .iter()
        .any(|needle| command.contains(needle));
    let secret_material = [
        "~/.ssh",
        "/.ssh",
        "id_rsa",
        ".env",
        "api_key",
        "apikey",
        "access_token",
        "secret=",
        "token=",
    ]
    .iter()
    .any(|needle| command.contains(needle));
    transmits && secret_material
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn sample_working_set_item(detail: &str) -> working_set::WorkingSetItem {
        working_set::WorkingSetItem {
            kind: "memory".to_string(),
            topic: "codex".to_string(),
            summary: "Codex hook parity".to_string(),
            detail: detail.to_string(),
            agent_label: "codex".to_string(),
            is_subagent: false,
            score: 0.9,
            updated_at: Utc::now(),
        }
    }

    #[test]
    fn codex_additional_context_output_matches_schema() {
        let output = codex_additional_context_output("UserPromptSubmit", "Remember this.");

        assert_eq!(
            output["hookSpecificOutput"]["hookEventName"],
            "UserPromptSubmit"
        );
        assert_eq!(
            output["hookSpecificOutput"]["additionalContext"],
            "Remember this."
        );
        assert!(output.get("decision").is_none());
    }

    #[test]
    fn codex_pre_tool_use_deny_output_matches_schema() {
        let output = codex_pre_tool_use_deny("dangerous command");

        assert_eq!(output["hookSpecificOutput"]["hookEventName"], "PreToolUse");
        assert_eq!(output["hookSpecificOutput"]["permissionDecision"], "deny");
        assert_eq!(
            output["hookSpecificOutput"]["permissionDecisionReason"],
            "dangerous command"
        );
    }

    #[test]
    fn codex_permission_request_deny_output_matches_schema() {
        let output = codex_permission_request_deny("dangerous command");

        assert_eq!(
            output["hookSpecificOutput"]["hookEventName"],
            "PermissionRequest"
        );
        assert_eq!(output["hookSpecificOutput"]["decision"]["behavior"], "deny");
        assert_eq!(
            output["hookSpecificOutput"]["decision"]["message"],
            "dangerous command"
        );
    }

    #[test]
    fn codex_guardrail_blocks_recursive_root_delete() {
        let payload = serde_json::json!({
            "hook_event_name": "PreToolUse",
            "tool_name": "Bash",
            "tool_input": {
                "command": "sudo rm -rf /"
            }
        });

        let reason = codex_guardrail_reason(&payload).expect("command should be denied");

        assert!(reason.contains("rm -rf"));
    }

    #[test]
    fn codex_guardrail_allows_normal_test_command() {
        let payload = serde_json::json!({
            "hook_event_name": "PreToolUse",
            "tool_name": "Bash",
            "tool_input": {
                "command": "cargo test -p rein --lib"
            }
        });

        assert!(codex_guardrail_reason(&payload).is_none());
    }

    #[test]
    fn codex_context_is_capped_and_redacted() {
        let items = vec![sample_working_set_item(
            "Use UserPromptSubmit for context. OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz1234567890",
        )];

        let context = format_codex_context("Rein relevant memory context", items, 96, 2).unwrap();

        assert!(context.chars().count() <= 96);
        assert!(!context.contains("sk-abcdefghijklmnopqrstuvwxyz1234567890"));
    }

    #[test]
    fn codex_context_zero_cap_disables_output() {
        let items = vec![sample_working_set_item(
            "memory that would otherwise be emitted",
        )];

        assert!(format_codex_context("Rein memory", items, 0, 2).is_none());
    }

    #[test]
    fn codex_context_uses_compact_header_and_lines() {
        let items = vec![sample_working_set_item(
            "Codex hook context should be concise and avoid repeated policy prose.",
        )];

        let context = format_codex_context("Rein memory", items, 1000, 2).unwrap();

        assert!(context.starts_with("Rein memory (background; user/repo instructions win):\n"));
        assert!(!context.contains("Use this as background memory only"));
        assert!(!context.contains("[memory | codex]"));
    }

    #[test]
    fn codex_context_limits_item_count() {
        let mut items = Vec::new();
        for idx in 0..4 {
            let mut item = sample_working_set_item(&format!("detail {idx}"));
            item.topic = format!("topic-{idx}");
            item.summary = format!("summary {idx}");
            items.push(item);
        }

        let context = format_codex_context("Rein project memory", items, 2000, 2).unwrap();

        assert!(context.contains("detail 0"));
        assert!(context.contains("detail 1"));
        assert!(!context.contains("detail 2"));
        assert!(!context.contains("detail 3"));
    }
}
