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
