//! Payload parsing for Claude Code hook JSON formats.

use regex::Regex;
use std::collections::VecDeque;
use std::io::{BufRead, BufReader};
use std::sync::OnceLock;

/// Hard cap on hook transcript files loaded via `transcript_path`.
///
/// Hook payloads are untrusted input, so never follow an arbitrary path into an
/// unbounded `read_to_string`. 16 MiB is far above real transcript sizes while
/// still preventing accidental or malicious large-file ingestion.
const MAX_HOOK_TRANSCRIPT_BYTES: u64 = 16 * 1024 * 1024;

/// Check if a line likely contains secrets
pub fn looks_like_secret(line: &str) -> bool {
    let lower = line.to_lowercase();
    let patterns = [
        "api_key=",
        "api-key=",
        "apikey=",
        "token=",
        "access_token=",
        "client_secret=",
        "secret=",
        "password=",
        "authorization:",
        "bearer ",
        "export gemini_api_key",
        "export supermemory",
        "export rein_http_token",
        "export openai_api_key",
        "sk-",
        "gho_",
        "ghp_",
        "sm_",
        "-----begin",
        "-----end",
    ];
    patterns.iter().any(|p| lower.contains(p))
        || secret_redactors().iter().any(|pair| pair.0.is_match(line))
}

/// Redact obvious secrets from free-form text before persistence or display.
pub fn redact_secrets(text: &str) -> String {
    let mut redacted = text.to_string();
    for (re, replacement) in secret_redactors().iter() {
        redacted = re.replace_all(&redacted, *replacement).into_owned();
    }
    redacted
}

fn secret_redactors() -> &'static Vec<(Regex, &'static str)> {
    static REDACTORS: OnceLock<Vec<(Regex, &'static str)>> = OnceLock::new();
    REDACTORS.get_or_init(|| {
        vec![
            (
                Regex::new(r"(?is)-----BEGIN [^-]+-----.*?-----END [^-]+-----").unwrap(),
                "[REDACTED_PEM_BLOCK]",
            ),
            (
                Regex::new(r"(?i)\b(authorization\s*:\s*bearer)\s+[^\s]+").unwrap(),
                "$1 [REDACTED]",
            ),
            (
                Regex::new(r#"(?i)(["'](?:api[_-]?key|access[_-]?token|client[_-]?secret|token|secret|password)["']\s*:\s*")([^"]*)(")"#).unwrap(),
                "$1[REDACTED]$3",
            ),
            (
                Regex::new(r#"(?i)(["'](?:api[_-]?key|access[_-]?token|client[_-]?secret|token|secret|password)["']\s*:\s*')([^']*)(')"#).unwrap(),
                "$1[REDACTED]$3",
            ),
            (
                Regex::new(r#"(?i)\b((?:api[_-]?key|access[_-]?token|client[_-]?secret|token|secret|password)\s*[:=]\s*)("[^"]*"|'[^']*'|[^\s]+)"#).unwrap(),
                "$1[REDACTED]",
            ),
            (
                Regex::new(r"(?i)\b(export\s+(?:gemini_api_key|supermemory_cc_api_key|rein_http_token|openai_api_key)\s*=\s*)[^\s]+").unwrap(),
                "$1[REDACTED]",
            ),
            (Regex::new(r"\bsk-[A-Za-z0-9_-]{20,}\b").unwrap(), "[REDACTED_API_KEY]"),
            (
                Regex::new(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b").unwrap(),
                "[REDACTED_GITHUB_TOKEN]",
            ),
            (
                Regex::new(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b").unwrap(),
                "[REDACTED_GITHUB_TOKEN]",
            ),
            (Regex::new(r"\bhf_[A-Za-z]{20,}\b").unwrap(), "[REDACTED_HF_TOKEN]"),
            (Regex::new(r"\bAKIA[0-9A-Z]{16}\b").unwrap(), "[REDACTED_AWS_KEY]"),
            (Regex::new(r"\bAIza[0-9A-Za-z\-_]{20,}\b").unwrap(), "[REDACTED_GCP_KEY]"),
            (
                Regex::new(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b").unwrap(),
                "[REDACTED_SLACK_TOKEN]",
            ),
            (Regex::new(r"\bsm_[A-Za-z0-9_-]{20,}\b").unwrap(), "[REDACTED_SM_TOKEN]"),
        ]
    })
}

/// Returns the subagent identifier if the hook fired inside a subagent.
pub fn hook_agent_id(input: &str) -> Option<String> {
    serde_json::from_str::<serde_json::Value>(input)
        .ok()
        .and_then(|json| {
            json.get("agent_id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
}

pub fn hook_agent_type(input: &str) -> Option<String> {
    serde_json::from_str::<serde_json::Value>(input)
        .ok()
        .and_then(|json| {
            json.get("agent_type")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
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
            if let Some(reader) = open_transcript_reader(path, "extract_hook_text") {
                return extract_transcript_text(reader);
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
            if let Some(reader) = open_transcript_reader(path, "extract_hook_text_for_llm") {
                return extract_transcript_text_for_llm(reader);
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

fn open_transcript_reader(path: &str, context: &str) -> Option<BufReader<std::fs::File>> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return None;
    }

    let transcript_path = std::path::Path::new(trimmed);
    let has_jsonl_ext = transcript_path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| matches!(ext.to_ascii_lowercase().as_str(), "jsonl" | "ndjson"))
        .unwrap_or(false);
    if !has_jsonl_ext {
        tracing::warn!(
            context = context,
            path = %trimmed,
            "hook transcript_path rejected: expected .jsonl/.ndjson file"
        );
        return None;
    }

    let metadata = match std::fs::symlink_metadata(transcript_path) {
        Ok(metadata) => metadata,
        Err(error) => {
            tracing::warn!(
                context = context,
                path = %trimmed,
                error = %error,
                "hook transcript_path unreadable"
            );
            return None;
        }
    };
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        tracing::warn!(
            context = context,
            path = %trimmed,
            "hook transcript_path rejected: not a regular file"
        );
        return None;
    }
    if metadata.len() > MAX_HOOK_TRANSCRIPT_BYTES {
        tracing::warn!(
            context = context,
            path = %trimmed,
            size = metadata.len(),
            cap = MAX_HOOK_TRANSCRIPT_BYTES,
            "hook transcript_path rejected: file exceeds size cap"
        );
        return None;
    }

    match std::fs::File::open(transcript_path) {
        Ok(file) => Some(BufReader::new(file)),
        Err(error) => {
            tracing::warn!(
                context = context,
                path = %trimmed,
                error = %error,
                "hook transcript_path open failed"
            );
            None
        }
    }
}

fn extract_transcript_turn(entry: &serde_json::Value) -> Option<(&'static str, String)> {
    let msg_type = entry.get("type").and_then(|v| v.as_str()).unwrap_or("");
    if msg_type != "human" && msg_type != "assistant" {
        return None;
    }
    let content = if let Some(msg) = entry.get("message") {
        extract_message_content(msg.get("content"))
    } else {
        extract_message_content(entry.get("content"))
    };
    if content.is_empty() {
        return None;
    }
    let role = if msg_type == "human" {
        "User"
    } else {
        "Assistant"
    };
    Some((role, content))
}

fn extract_transcript_text_with_limits<R: BufRead>(
    reader: R,
    max_turns: usize,
    max_chars_per_turn: usize,
) -> String {
    if max_turns == 0 {
        return String::new();
    }

    let mut turns = VecDeque::with_capacity(max_turns.min(64));
    for line in reader.lines() {
        let Ok(line) = line else {
            continue;
        };
        let Ok(entry) = serde_json::from_str::<serde_json::Value>(&line) else {
            continue;
        };
        if let Some((role, content)) = extract_transcript_turn(&entry) {
            let truncated: String = content.chars().take(max_chars_per_turn).collect();
            if turns.len() == max_turns {
                turns.pop_front();
            }
            turns.push_back(format!("{role}: {truncated}"));
        }
    }
    turns.into_iter().collect::<Vec<_>>().join("\n\n")
}

fn extract_transcript_text<R: BufRead>(reader: R) -> String {
    extract_transcript_text_with_limits(reader, 20, 500)
}

fn extract_transcript_text_for_llm<R: BufRead>(reader: R) -> String {
    extract_transcript_text_with_limits(reader, 200, 4000)
}

fn extract_message_content(content: Option<&serde_json::Value>) -> String {
    match content {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Array(arr)) => arr
            .iter()
            .filter_map(|item| {
                if let Some(s) = item.as_str() {
                    Some(s.to_string())
                } else if item.get("type").and_then(|t| t.as_str()) == Some("text") {
                    item.get("text")
                        .and_then(|t| t.as_str())
                        .map(|s| s.to_string())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join(" "),
        _ => String::new(),
    }
}

/// Count actual conversation turns from a hook payload.
pub fn count_transcript_turns(input: &str) -> usize {
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(input) {
        if let Some(path) = json.get("transcript_path").and_then(|v| v.as_str()) {
            if let Some(reader) = open_transcript_reader(path, "count_transcript_turns") {
                return reader
                    .lines()
                    .map_while(Result::ok)
                    .filter_map(|line| serde_json::from_str::<serde_json::Value>(&line).ok())
                    .filter(|entry| extract_transcript_turn(entry).is_some())
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
        let reader = open_transcript_reader(path, "extract_hook_session_ingest")?;
        let mut turns = Vec::new();
        const MAX_HOOK_TURNS: usize = 500;
        const MAX_HOOK_CHARS: usize = 500_000;
        let mut total_chars = 0usize;
        for line in reader.lines() {
            let Ok(line) = line else {
                continue;
            };
            let Ok(entry) = serde_json::from_str::<serde_json::Value>(&line) else {
                continue;
            };
            let Some((role, content)) = extract_transcript_turn(&entry) else {
                continue;
            };
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

#[cfg(test)]
mod tests {
    use super::{
        count_transcript_turns, extract_hook_session_ingest, extract_hook_text,
        extract_hook_text_for_llm, looks_like_secret, redact_secrets, MAX_HOOK_TRANSCRIPT_BYTES,
    };
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn redact_masks_assignment_and_bearer_values() {
        let input = "authorization: Bearer sk-secret-12345678901234567890\nOPENAI_API_KEY=sk-proj-abcdefghijklmnopqrstuvwxyz1234567890";
        let output = redact_secrets(input);
        assert!(!output.contains("sk-secret-12345678901234567890"));
        assert!(!output.contains("sk-proj-abcdefghijklmnopqrstuvwxyz1234567890"));
        assert!(output.contains("[REDACTED]") || output.contains("[REDACTED_API_KEY]"));
    }

    #[test]
    fn secret_detection_catches_known_token_formats() {
        assert!(looks_like_secret(
            "Authorization: Bearer sk-abcdefghijklmnopqrstuvwx1234567890"
        ));
        assert!(looks_like_secret(
            "github_pat_abcdefghijklmnopqrstuvwxyz0123456789"
        ));
        assert!(!looks_like_secret(
            "We chose PostgreSQL for the billing database."
        ));
    }

    #[test]
    fn redact_masks_json_quoted_secret_keys() {
        let input = r#"{"token":"plain-secret","access_token":"rot-secret","client_secret":"shh"}"#;
        let output = redact_secrets(input);
        assert!(!output.contains("plain-secret"));
        assert!(!output.contains("rot-secret"));
        assert!(!output.contains("shh"));
        assert!(output.contains(r#""token":"[REDACTED]""#));
        assert!(output.contains(r#""access_token":"[REDACTED]""#));
        assert!(output.contains(r#""client_secret":"[REDACTED]""#));
    }

    #[test]
    fn transcript_path_rejects_non_jsonl_files_and_falls_back() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("transcript.txt");
        let mut file = std::fs::File::create(&path).unwrap();
        writeln!(
            file,
            r#"{{"type":"assistant","message":{{"content":"secret transcript"}}}}"#
        )
        .unwrap();

        let payload = serde_json::json!({
            "transcript_path": path,
            "summary": "fallback summary",
            "transcript": "fallback transcript"
        })
        .to_string();

        assert_eq!(extract_hook_text(&payload), "fallback transcript");
        assert_eq!(extract_hook_text_for_llm(&payload), "fallback transcript");
        assert_eq!(count_transcript_turns(&payload), 0);
        assert!(extract_hook_session_ingest(&payload).is_none());
    }

    #[test]
    fn transcript_path_rejects_oversize_files_and_falls_back() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("transcript.jsonl");
        let file = std::fs::File::create(&path).unwrap();
        file.set_len(MAX_HOOK_TRANSCRIPT_BYTES + 1).unwrap();

        let payload = serde_json::json!({
            "transcript_path": path,
            "summary": "fallback summary",
        })
        .to_string();

        assert_eq!(extract_hook_text(&payload), "fallback summary");
        assert_eq!(extract_hook_text_for_llm(&payload), "fallback summary");
        assert_eq!(count_transcript_turns(&payload), 0);
        assert!(extract_hook_session_ingest(&payload).is_none());
    }
}
