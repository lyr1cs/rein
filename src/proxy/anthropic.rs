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
