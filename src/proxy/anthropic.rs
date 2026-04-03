//! Anthropic `/v1/messages` format handling.

use serde_json::Value;

/// Extract the last user message text as the recall query.
pub fn extract_query(body: &Value) -> String {
    let messages = match body.get("messages").and_then(|m| m.as_array()) {
        Some(m) => m,
        None => return String::new(),
    };

    // Find last user message.
    for msg in messages.iter().rev() {
        if msg.get("role").and_then(|r| r.as_str()) != Some("user") {
            continue;
        }
        return extract_text_content(msg.get("content"));
    }
    String::new()
}

/// Extract text from Anthropic content (string or array of content blocks).
fn extract_text_content(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(blocks)) => {
            let mut texts = Vec::new();
            for block in blocks {
                if block.get("type").and_then(|t| t.as_str()) == Some("text") {
                    if let Some(t) = block.get("text").and_then(|t| t.as_str()) {
                        texts.push(t.to_string());
                    }
                }
            }
            texts.join(" ")
        }
        _ => String::new(),
    }
}

/// Inject context into the Anthropic `system` field.
///
/// Anthropic format: `system` is either a string or an array of `{"type":"text","text":"..."}`.
pub fn inject_context(body: &mut Value, context: &str) {
    match body.get("system") {
        Some(Value::String(existing)) => {
            let combined = format!("{context}\n\n{existing}");
            body["system"] = Value::String(combined);
        }
        Some(Value::Array(_)) => {
            let block = serde_json::json!({"type": "text", "text": context});
            if let Some(arr) = body.get_mut("system").and_then(|v| v.as_array_mut()) {
                arr.insert(0, block);
            }
        }
        _ => {
            body["system"] = Value::String(context.to_string());
        }
    }
}

/// Extract assistant text from a non-streaming Anthropic response.
pub fn extract_assistant_text_full(resp_bytes: &[u8]) -> Option<String> {
    let resp: Value = serde_json::from_slice(resp_bytes).ok()?;
    let content = resp.get("content")?.as_array()?;
    let mut texts = Vec::new();
    for block in content {
        if block.get("type").and_then(|t| t.as_str()) == Some("text") {
            if let Some(t) = block.get("text").and_then(|t| t.as_str()) {
                texts.push(t.to_string());
            }
        }
    }
    if texts.is_empty() {
        None
    } else {
        Some(texts.join("\n"))
    }
}

/// Extract assistant text fragments from Anthropic SSE chunks.
///
/// Anthropic SSE format:
/// ```text
/// event: content_block_delta
/// data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}
/// ```
pub fn extract_assistant_text_sse(chunk_text: &str) -> Option<String> {
    let mut result = String::new();
    for line in chunk_text.lines() {
        let line = line.trim();
        if !line.starts_with("data: ") {
            continue;
        }
        let data = &line[6..];
        if let Ok(json) = serde_json::from_str::<Value>(data) {
            // content_block_delta
            if let Some(delta) = json.get("delta") {
                if let Some(text) = delta.get("text").and_then(|t| t.as_str()) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_extract_query_string_content() {
        let body = json!({
            "messages": [
                {"role": "user", "content": "Hello world"},
                {"role": "assistant", "content": "Hi!"},
                {"role": "user", "content": "What is rein?"}
            ]
        });
        assert_eq!(extract_query(&body), "What is rein?");
    }

    #[test]
    fn test_extract_query_content_blocks() {
        let body = json!({
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "Check this"},
                    {"type": "image", "source": {"data": "..."}}
                ]}
            ]
        });
        assert_eq!(extract_query(&body), "Check this");
    }

    #[test]
    fn test_inject_system_string() {
        let mut body = json!({
            "system": "You are helpful.",
            "messages": []
        });
        inject_context(&mut body, "<rein-context>memory</rein-context>");
        let sys = body["system"].as_str().unwrap();
        assert!(sys.starts_with("<rein-context>"));
        assert!(sys.contains("You are helpful."));
    }

    #[test]
    fn test_inject_system_array() {
        let mut body = json!({
            "system": [{"type": "text", "text": "You are helpful."}],
            "messages": []
        });
        inject_context(&mut body, "<rein-context>memory</rein-context>");
        let arr = body["system"].as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["text"].as_str().unwrap(), "<rein-context>memory</rein-context>");
    }

    #[test]
    fn test_inject_system_absent() {
        let mut body = json!({"messages": []});
        inject_context(&mut body, "<rein-context>memory</rein-context>");
        assert_eq!(body["system"].as_str().unwrap(), "<rein-context>memory</rein-context>");
    }

    #[test]
    fn test_parse_sse_content_block_delta() {
        let chunk = "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n";
        assert_eq!(extract_assistant_text_sse(chunk), Some("Hello".to_string()));
    }

    #[test]
    fn test_parse_sse_no_text() {
        let chunk = "event: message_start\ndata: {\"type\":\"message_start\"}\n\n";
        assert_eq!(extract_assistant_text_sse(chunk), None);
    }
}
