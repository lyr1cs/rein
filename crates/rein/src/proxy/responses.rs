//! OpenAI Responses API (`/v1/responses`) format handling.
//!
//! The Responses API uses a different request/response format than Chat Completions:
//! - Request: `input` (string or array) instead of `messages`
//! - Response: `output[]` with `type: "output_text"` items
//! - SSE: `response.output_text.delta` events with `delta.text`

use serde_json::Value;

/// Extract the user query from a Responses API request body.
///
/// Handles both string input (`"input": "hello"`) and array input
/// (`"input": [{"role": "user", "content": "hello"}]`).
pub fn extract_query(body: &Value) -> String {
    // String input
    if let Some(s) = body.get("input").and_then(|v| v.as_str()) {
        return s.to_string();
    }

    // Array input — find last user message
    if let Some(items) = body.get("input").and_then(|v| v.as_array()) {
        for item in items.iter().rev() {
            let role = item.get("role").and_then(|r| r.as_str());
            if role == Some("user") {
                // Content can be a string or structured
                if let Some(s) = item.get("content").and_then(|c| c.as_str()) {
                    return s.to_string();
                }
                // Array of content parts
                if let Some(parts) = item.get("content").and_then(|c| c.as_array()) {
                    let mut texts = Vec::new();
                    for part in parts {
                        if part.get("type").and_then(|t| t.as_str()) == Some("input_text") {
                            if let Some(t) = part.get("text").and_then(|t| t.as_str()) {
                                texts.push(t.to_string());
                            }
                        }
                    }
                    if !texts.is_empty() {
                        return texts.join(" ");
                    }
                }
            }
        }
    }

    // Fallback to instructions
    if let Some(s) = body.get("instructions").and_then(|v| v.as_str()) {
        return s.to_string();
    }

    String::new()
}

/// Extract assistant text from a non-streaming Responses API response.
///
/// Response format: `{"output": [{"type": "output_text", "text": "..."}]}`
pub fn extract_assistant_text_full(resp_bytes: &[u8]) -> Option<String> {
    let resp: Value = serde_json::from_slice(resp_bytes).ok()?;
    let output = resp.get("output")?.as_array()?;
    let mut texts = Vec::new();
    for item in output {
        if item.get("type").and_then(|t| t.as_str()) == Some("output_text") {
            if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                texts.push(text.to_string());
            }
        }
        // Also handle message type with nested content
        if item.get("type").and_then(|t| t.as_str()) == Some("message") {
            if let Some(content) = item.get("content").and_then(|c| c.as_array()) {
                for part in content {
                    if part.get("type").and_then(|t| t.as_str()) == Some("output_text") {
                        if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                            texts.push(text.to_string());
                        }
                    }
                }
            }
        }
    }
    if texts.is_empty() {
        None
    } else {
        Some(texts.join("\n"))
    }
}

/// Extract assistant text fragments from Responses API SSE chunks.
///
/// SSE format:
/// ```text
/// data: {"type":"response.output_text.delta","delta":{"text":"Hello"}}
/// ```
pub fn extract_assistant_text_sse(chunk_text: &str) -> Option<String> {
    let mut result = String::new();
    for line in chunk_text.lines() {
        let line = line.trim();
        let data = match line.strip_prefix("data: ") {
            Some(d) => d,
            None => continue,
        };
        if data == "[DONE]" {
            continue;
        }
        if let Ok(json) = serde_json::from_str::<Value>(data) {
            let event_type = json.get("type").and_then(|t| t.as_str()).unwrap_or("");
            if event_type == "response.output_text.delta" {
                if let Some(text) = json
                    .get("delta")
                    .and_then(|d| d.get("text"))
                    .and_then(|t| t.as_str())
                {
                    result.push_str(text);
                }
            }
        }
    }
    if result.is_empty() {
        None
    } else {
        Some(result)
    }
}

/// Extract assistant text fragment from a single Responses websocket JSON event.
///
/// Expected event shape:
/// `{"type":"response.output_text.delta","delta":"Hello"}`
pub fn extract_assistant_text_ws_message(message_text: &str) -> Option<String> {
    let json = serde_json::from_str::<Value>(message_text).ok()?;
    if json.get("type").and_then(|t| t.as_str()) != Some("response.output_text.delta") {
        return None;
    }
    json.get("delta")
        .and_then(|delta| delta.as_str())
        .map(str::to_string)
}

/// Extract the user query from a single Responses websocket request message.
///
/// Expected request shape:
/// `{"type":"response.create","input":[...],"instructions":"..."}`
pub fn extract_query_ws_message(message_text: &str) -> Option<String> {
    let json = serde_json::from_str::<Value>(message_text).ok()?;
    if json.get("type").and_then(|t| t.as_str()) != Some("response.create") {
        return None;
    }
    let query = extract_query(&json);
    let query = query.trim();
    if query.is_empty() {
        None
    } else {
        Some(query.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_extract_query_string_input() {
        let body = json!({"model": "gpt-5", "input": "What is memory?"});
        assert_eq!(extract_query(&body), "What is memory?");
    }

    #[test]
    fn test_extract_query_array_input() {
        let body = json!({
            "model": "gpt-5",
            "input": [
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "Answer"},
                {"role": "user", "content": "Follow-up"}
            ]
        });
        assert_eq!(extract_query(&body), "Follow-up");
    }

    #[test]
    fn test_extract_query_instructions_fallback() {
        let body = json!({"model": "gpt-5", "instructions": "Be helpful"});
        assert_eq!(extract_query(&body), "Be helpful");
    }

    #[test]
    fn test_extract_full_output_text() {
        let resp = json!({
            "id": "resp_123",
            "output": [{"type": "output_text", "text": "Hello world"}]
        });
        let bytes = serde_json::to_vec(&resp).unwrap();
        assert_eq!(
            extract_assistant_text_full(&bytes),
            Some("Hello world".to_string())
        );
    }

    #[test]
    fn test_extract_full_message_type() {
        let resp = json!({
            "id": "resp_123",
            "output": [{
                "type": "message",
                "content": [{"type": "output_text", "text": "Nested text"}]
            }]
        });
        let bytes = serde_json::to_vec(&resp).unwrap();
        assert_eq!(
            extract_assistant_text_full(&bytes),
            Some("Nested text".to_string())
        );
    }

    #[test]
    fn test_extract_sse_delta() {
        let chunk =
            "data: {\"type\":\"response.output_text.delta\",\"delta\":{\"text\":\"Hello\"}}\n\n";
        assert_eq!(extract_assistant_text_sse(chunk), Some("Hello".to_string()));
    }

    #[test]
    fn test_extract_sse_non_text_event() {
        let chunk = "data: {\"type\":\"response.started\",\"response\":{\"id\":\"resp_123\"}}\n\n";
        assert_eq!(extract_assistant_text_sse(chunk), None);
    }

    #[test]
    fn test_extract_sse_done() {
        let chunk = "data: [DONE]\n\n";
        assert_eq!(extract_assistant_text_sse(chunk), None);
    }

    #[test]
    fn test_extract_ws_delta() {
        let msg = r#"{"type":"response.output_text.delta","delta":"Hello"}"#;
        assert_eq!(
            extract_assistant_text_ws_message(msg),
            Some("Hello".to_string())
        );
    }

    #[test]
    fn test_extract_ws_non_text_event() {
        let msg = r#"{"type":"response.created","response":{"id":"resp_123"}}"#;
        assert_eq!(extract_assistant_text_ws_message(msg), None);
    }

    #[test]
    fn test_extract_ws_request_query() {
        let msg = r#"{"type":"response.create","input":[{"role":"user","content":"Hello from ws"}]}"#;
        assert_eq!(
            extract_query_ws_message(msg),
            Some("Hello from ws".to_string())
        );
    }

    #[test]
    fn test_extract_ws_request_non_create_event() {
        let msg = r#"{"type":"response.completed","response":{"id":"resp_123"}}"#;
        assert_eq!(extract_query_ws_message(msg), None);
    }
}
