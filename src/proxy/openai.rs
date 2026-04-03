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

/// Inject context into the OpenAI messages array.
///
/// If a system message exists, append context to it.
/// Otherwise insert a new system message at position 0.
pub fn inject_context(body: &mut Value, context: &str) {
    let messages = match body.get_mut("messages").and_then(|m| m.as_array_mut()) {
        Some(m) => m,
        None => return,
    };

    // Find existing system message.
    for msg in messages.iter_mut() {
        if msg.get("role").and_then(|r| r.as_str()) == Some("system") {
            match msg.get("content") {
                Some(Value::String(existing)) => {
                    let combined = format!("{context}\n\n{existing}");
                    msg["content"] = Value::String(combined);
                }
                Some(Value::Array(_)) => {
                    // Structured content parts — prepend a text part.
                    let new_part = serde_json::json!({"type": "text", "text": context});
                    if let Some(arr) = msg.get_mut("content").and_then(|v| v.as_array_mut()) {
                        arr.insert(0, new_part);
                    }
                }
                _ => {
                    msg["content"] = Value::String(context.to_string());
                }
            }
            return;
        }
    }

    // No system message found — insert one.
    let sys_msg = serde_json::json!({"role": "system", "content": context});
    messages.insert(0, sys_msg);
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
    fn test_inject_system_existing() {
        let mut body = json!({
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"}
            ]
        });
        inject_context(&mut body, "<rein-context>memory</rein-context>");
        let sys = body["messages"][0]["content"].as_str().unwrap();
        assert!(sys.starts_with("<rein-context>"));
        assert!(sys.contains("You are helpful."));
    }

    #[test]
    fn test_inject_system_missing() {
        let mut body = json!({
            "messages": [
                {"role": "user", "content": "Hi"}
            ]
        });
        inject_context(&mut body, "<rein-context>memory</rein-context>");
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"].as_str().unwrap(), "system");
        assert!(msgs[0]["content"].as_str().unwrap().contains("<rein-context>"));
    }

    #[test]
    fn test_inject_system_array_content() {
        let mut body = json!({
            "messages": [
                {"role": "system", "content": [
                    {"type": "text", "text": "You are helpful."},
                    {"type": "text", "text": "Be concise."}
                ]},
                {"role": "user", "content": "Hi"}
            ]
        });
        inject_context(&mut body, "<rein-context>memory</rein-context>");
        let content = body["messages"][0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 3);
        assert_eq!(content[0]["text"].as_str().unwrap(), "<rein-context>memory</rein-context>");
        assert_eq!(content[1]["text"].as_str().unwrap(), "You are helpful.");
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
