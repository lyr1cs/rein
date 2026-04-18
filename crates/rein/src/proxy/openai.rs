//! OpenAI `/v1/chat/completions` format handling.

use serde_json::Value;

/// Extract the last user message text as the recall query.
pub fn extract_query(body: &Value) -> String {
    let messages = match body.get("messages").and_then(|m| m.as_array()) {
        Some(m) => m,
        None => return String::new(),
    };

    for msg in messages.iter().rev() {
        if msg.get("role").and_then(|r| r.as_str()) != Some("user") {
            continue;
        }
        return extract_text_content(msg.get("content"));
    }
    String::new()
}

/// Extract text from OpenAI content (string or array of content parts).
fn extract_text_content(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => {
            let mut texts = Vec::new();
            for part in parts {
                if part.get("type").and_then(|t| t.as_str()) == Some("text") {
                    if let Some(t) = part.get("text").and_then(|t| t.as_str()) {
                        texts.push(t.to_string());
                    }
                }
            }
            texts.join(" ")
        }
        _ => String::new(),
    }
}

/// Extract assistant text from a non-streaming OpenAI response.
pub fn extract_assistant_text_full(resp_bytes: &[u8]) -> Option<String> {
    let resp: Value = serde_json::from_slice(resp_bytes).ok()?;
    let choices = resp.get("choices")?.as_array()?;
    let first = choices.first()?;
    let content = first.get("message")?.get("content")?.as_str()?;
    Some(content.to_string())
}

/// Extract assistant text fragments from OpenAI SSE chunks.
///
/// OpenAI SSE format:
/// ```text
/// data: {"choices":[{"delta":{"content":"Hello"}}]}
/// ```
/// Stream ends with `data: [DONE]`.
pub fn extract_assistant_text_sse(chunk_text: &str) -> Option<String> {
    let mut result = String::new();
    for line in chunk_text.lines() {
        let line = line.trim();
        if !line.starts_with("data: ") {
            continue;
        }
        let data = &line[6..];
        if data == "[DONE]" {
            continue;
        }
        if let Ok(json) = serde_json::from_str::<Value>(data) {
            if let Some(choices) = json.get("choices").and_then(|c| c.as_array()) {
                for choice in choices {
                    if let Some(content) = choice
                        .get("delta")
                        .and_then(|d| d.get("content"))
                        .and_then(|c| c.as_str())
                    {
                        result.push_str(content);
                    }
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
    fn test_extract_query_simple() {
        let body = json!({
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is memory?"},
            ]
        });
        assert_eq!(extract_query(&body), "What is memory?");
    }

    #[test]
    fn test_extract_query_multipart() {
        let body = json!({
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "Analyze this"},
                    {"type": "image_url", "image_url": {"url": "..."}}
                ]}
            ]
        });
        assert_eq!(extract_query(&body), "Analyze this");
    }

    #[test]
    fn test_parse_sse_delta() {
        let chunk = "data: {\"choices\":[{\"delta\":{\"content\":\"world\"}}]}\n\n";
        assert_eq!(extract_assistant_text_sse(chunk), Some("world".to_string()));
    }

    #[test]
    fn test_parse_sse_done() {
        let chunk = "data: [DONE]\n\n";
        assert_eq!(extract_assistant_text_sse(chunk), None);
    }
}
