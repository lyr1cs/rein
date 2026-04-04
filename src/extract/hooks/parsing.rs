//! Payload parsing for Claude Code hook JSON formats.

/// Check if a line likely contains secrets
pub fn looks_like_secret(line: &str) -> bool {
    let lower = line.to_lowercase();
    let patterns = [
        "api_key=", "api-key=", "apikey=",
        "token=", "secret=", "password=",
        "authorization:", "bearer ",
        "export gemini_api_key", "export supermemory",
        "export rein_http_token", "export openai_api_key",
        "sk-", "gho_", "ghp_", "sm_",
        "-----begin", "-----end",
    ];
    patterns.iter().any(|p| lower.contains(p))
}

/// Returns the subagent identifier if the hook fired inside a subagent.
pub fn hook_agent_id(input: &str) -> Option<String> {
    serde_json::from_str::<serde_json::Value>(input)
        .ok()
        .and_then(|json| json.get("agent_id").and_then(|v| v.as_str()).map(|s| s.to_string()))
}

pub fn hook_agent_type(input: &str) -> Option<String> {
    serde_json::from_str::<serde_json::Value>(input)
        .ok()
        .and_then(|json| json.get("agent_type").and_then(|v| v.as_str()).map(|s| s.to_string()))
}

pub fn is_subagent_hook(input: &str) -> bool {
    hook_agent_id(input).is_some()
}

/// Best-effort runtime client/agent label.
/// Prefer explicit override, then detect common agent environments.
pub fn runtime_agent_label() -> String {
    if let Ok(label) = std::env::var("REIN_AGENT_LABEL") {
        let trimmed = label.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    if std::env::var("CODEX_THREAD_ID").is_ok() || std::env::var("CODEX_CI").is_ok() {
        return "codex".to_string();
    }
    if std::env::var("CLAUDECODE").is_ok() || std::env::var("CLAUDE_CODE_ENTRYPOINT").is_ok() {
        return "claude-code".to_string();
    }
    if std::env::var("CURSOR_TRACE_ID").is_ok() || std::env::var("CURSOR_AGENT").is_ok() {
        return "cursor".to_string();
    }
    if std::env::var("WINDSURF").is_ok() || std::env::var("CODEIUM").is_ok() {
        return "windsurf".to_string();
    }
    if std::env::var("GEMINI_CLI").is_ok() {
        return "gemini".to_string();
    }
    if std::env::var("OPENCODE").is_ok() {
        return "opencode".to_string();
    }
    "unknown-agent".to_string()
}

pub fn classify_hook_agent(input: &str) -> (String, bool) {
    let runtime = runtime_agent_label();
    let agent_id = hook_agent_id(input);
    let agent_type = hook_agent_type(input);
    if let Some(agent_id) = agent_id {
        let short_id: String = agent_id.chars().take(8).collect();
        let label = agent_type
            .filter(|t| !t.trim().is_empty())
            .map(|t| format!("{runtime}:{t}@{short_id}"))
            .unwrap_or_else(|| format!("{runtime}:subagent@{short_id}"));
        tracing::debug!("subagent hook detected: agent_id={agent_id}, label={label}");
        (label, true)
    } else {
        (runtime, false)
    }
}

/// Extract text content from a Claude Code hook JSON payload.
/// Falls back to raw input if not valid JSON.
pub fn extract_hook_text(input: &str) -> String {
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(input) {
        if let Some(output) = json.get("tool_output").and_then(|v| v.as_str()) {
            return output.to_string();
        }
        if let Some(path) = json.get("transcript_path").and_then(|v| v.as_str()) {
            if let Ok(transcript_content) = std::fs::read_to_string(path) {
                return extract_transcript_text(&transcript_content);
            }
        }
        if let Some(transcript) = json.get("transcript").and_then(|v| v.as_str()) {
            return transcript.to_string();
        }
        if let Some(summary) = json.get("summary").and_then(|v| v.as_str()) {
            return summary.to_string();
        }
        return String::new();
    }
    input.to_string()
}

/// Like extract_hook_text but with larger limits for LLM consumption.
pub fn extract_hook_text_for_llm(input: &str) -> String {
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(input) {
        if let Some(output) = json.get("tool_output").and_then(|v| v.as_str()) {
            return output.to_string();
        }
        if let Some(path) = json.get("transcript_path").and_then(|v| v.as_str()) {
            if let Ok(transcript_content) = std::fs::read_to_string(path) {
                return extract_transcript_text_for_llm(&transcript_content);
            }
        }
        if let Some(transcript) = json.get("transcript").and_then(|v| v.as_str()) {
            return transcript.to_string();
        }
        if let Some(summary) = json.get("summary").and_then(|v| v.as_str()) {
            return summary.to_string();
        }
        return String::new();
    }
    input.to_string()
}

fn extract_transcript_text_with_limits(jsonl: &str, max_turns: usize, max_chars_per_turn: usize) -> String {
    let mut turns = Vec::new();
    for line in jsonl.lines() {
        if let Ok(entry) = serde_json::from_str::<serde_json::Value>(line) {
            let msg_type = entry.get("type").and_then(|v| v.as_str()).unwrap_or("");
            if msg_type != "human" && msg_type != "assistant" {
                continue;
            }
            let content = if let Some(msg) = entry.get("message") {
                extract_message_content(msg.get("content"))
            } else {
                extract_message_content(entry.get("content"))
            };
            if !content.is_empty() {
                let prefix = if msg_type == "human" { "User" } else { "Assistant" };
                let truncated: String = content.chars().take(max_chars_per_turn).collect();
                turns.push(format!("{}: {}", prefix, truncated));
            }
        }
    }
    let start = if turns.len() > max_turns { turns.len() - max_turns } else { 0 };
    turns[start..].join("\n\n")
}

fn extract_transcript_text(jsonl: &str) -> String {
    extract_transcript_text_with_limits(jsonl, 20, 500)
}

fn extract_transcript_text_for_llm(jsonl: &str) -> String {
    extract_transcript_text_with_limits(jsonl, 200, 4000)
}

fn extract_message_content(content: Option<&serde_json::Value>) -> String {
    match content {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Array(arr)) => {
            arr.iter()
                .filter_map(|item| {
                    if let Some(s) = item.as_str() {
                        Some(s.to_string())
                    } else if item.get("type").and_then(|t| t.as_str()) == Some("text") {
                        item.get("text").and_then(|t| t.as_str()).map(|s| s.to_string())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join(" ")
        }
        _ => String::new(),
    }
}

/// Count actual conversation turns from a hook payload.
pub fn count_transcript_turns(input: &str) -> usize {
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(input) {
        if let Some(path) = json.get("transcript_path").and_then(|v| v.as_str()) {
            if let Ok(content) = std::fs::read_to_string(path) {
                return content.lines()
                    .filter(|line| {
                        serde_json::from_str::<serde_json::Value>(line)
                            .ok()
                            .and_then(|e| e.get("type").and_then(|t| t.as_str()).map(|t| t == "human" || t == "assistant"))
                            .unwrap_or(false)
                    })
                    .count();
            }
        }
    }
    0
}

/// Parse a hook payload into a structured SessionIngest when possible.
pub fn extract_hook_session_ingest(input: &str) -> Option<crate::types::SessionIngest> {
    let json = serde_json::from_str::<serde_json::Value>(input).ok()?;
    let summary = json
        .get("summary")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    if let Some(path) = json.get("transcript_path").and_then(|v| v.as_str()) {
        let transcript_content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!("session ingest: failed to read transcript {}: {}", path, e);
                return None;
            }
        };
        let mut turns = Vec::new();
        const MAX_HOOK_TURNS: usize = 500;
        const MAX_HOOK_CHARS: usize = 500_000;
        let mut total_chars = 0usize;
        for line in transcript_content.lines() {
            let Ok(entry) = serde_json::from_str::<serde_json::Value>(line) else {
                continue;
            };
            let msg_type = entry.get("type").and_then(|v| v.as_str()).unwrap_or("");
            if msg_type != "human" && msg_type != "assistant" {
                continue;
            }
            let content = if let Some(msg) = entry.get("message") {
                extract_message_content(msg.get("content"))
            } else {
                extract_message_content(entry.get("content"))
            };
            if content.trim().is_empty() {
                continue;
            }
            let role = if msg_type == "human" { "User" } else { "Assistant" };
            total_chars += content.len();
            turns.push(crate::types::SessionTurn {
                role: role.to_string(),
                content,
            });
            if turns.len() >= MAX_HOOK_TURNS || total_chars >= MAX_HOOK_CHARS {
                break;
            }
        }
        return Some(crate::types::SessionIngest {
            schema_version: 1,
            artifact_kind: "session".to_string(),
            session_id: Some(path.to_string()),
            title: None,
            started_at: None,
            ended_at: None,
            summary,
            source_agent: Some(runtime_agent_label()),
            source_label: Some("hook_stop".to_string()),
            compact_summary: None,
            tool_outputs: vec![],
            turns,
        });
    }

    if let Some(transcript) = json.get("transcript").and_then(|v| v.as_str()) {
        return Some(crate::types::SessionIngest {
            schema_version: 1,
            artifact_kind: "session".to_string(),
            session_id: None,
            title: None,
            started_at: None,
            ended_at: None,
            summary,
            source_agent: Some(runtime_agent_label()),
            source_label: Some("hook_stop".to_string()),
            compact_summary: None,
            tool_outputs: vec![],
            turns: vec![crate::types::SessionTurn {
                role: "Transcript".to_string(),
                content: transcript.to_string(),
            }],
        });
    }

    None
}
