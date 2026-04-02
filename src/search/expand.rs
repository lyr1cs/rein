use crate::config::{Provider, ReinConfig};
use crate::types::error::{ReinError, ReinResult};
use reqwest::Client;
use serde_json::{json, Value};

const EXPAND_SYSTEM_PROMPT: &str = r#"Given a memory search query, generate alternative phrasings that would help find relevant stored memories.

Focus on:
- Synonym substitution (different words, same meaning)
- Language switching (if query is Chinese, add an English variant and vice versa)
- Specificity adjustment (more general OR more specific phrasing)

Output a JSON array of strings. Example:
["alternative phrasing 1", "alternative phrasing 2"]

Rules:
- Each alternative must be meaningfully different from the original
- Keep alternatives concise (under 100 characters each)
- If the query is already very specific (e.g., a function name or exact ID), return []
- Support both English and Chinese"#;

// ---------------------------------------------------------------------------
// Gemini expander
// ---------------------------------------------------------------------------

struct GeminiExpander {
    client: &'static Client,
    api_key: String,
    endpoint: String,
    model: String,
}

impl GeminiExpander {
    fn new(api_key: String, endpoint: String, model: String) -> Self {
        Self { client: crate::search::cache::http_client_10s(), api_key, endpoint, model }
    }

    async fn expand(&self, query: &str, max: usize) -> ReinResult<Vec<String>> {
        let prompt = format!(
            "{}\n\nGenerate up to {} alternatives.\n\nInput query: \"{}\"",
            EXPAND_SYSTEM_PROMPT, max, query
        );
        let url = format!(
            "{}/v1beta/models/{}:generateContent",
            self.endpoint, self.model
        );
        let body = json!({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.3
            }
        });

        let resp = self.client.post(&url)
            .header("x-goog-api-key", &self.api_key)
            .json(&body)
            .send()
            .await?;
        let status = resp.status();
        let text_body = resp.text().await?;

        if !status.is_success() {
            return Err(ReinError::Extract(format!(
                "Gemini expand API returned {}: {}",
                status,
                crate::types::truncate_for_error(&text_body, 500)
            )));
        }

        let parsed: Value = serde_json::from_str(&text_body)
            .map_err(|e| ReinError::Extract(e.to_string()))?;

        let content = parsed["candidates"][0]["content"]["parts"][0]["text"]
            .as_str()
            .ok_or_else(|| ReinError::Extract("missing candidates[0].content.parts[0].text".into()))?;

        parse_expansion_response(content, max)
    }
}

// ---------------------------------------------------------------------------
// OMLX expander (OpenAI-compatible)
// ---------------------------------------------------------------------------

struct OmlxExpander {
    client: &'static Client,
    endpoint: String,
    model: String,
}

impl OmlxExpander {
    fn new(endpoint: String, model: String) -> Self {
        Self { client: crate::search::cache::http_client_15s(), endpoint, model }
    }

    async fn expand(&self, query: &str, max: usize) -> ReinResult<Vec<String>> {
        let user_msg = format!(
            "Generate up to {} alternatives.\n\nInput query: \"{}\"",
            max, query
        );
        let url = format!("{}/chat/completions", self.endpoint);
        let make_body = |use_json_mode: bool| {
            let mut body = json!({
                "model": &self.model,
                "messages": [
                    {"role": "system", "content": EXPAND_SYSTEM_PROMPT},
                    {"role": "user", "content": &user_msg}
                ],
                "temperature": 0.3
            });
            if use_json_mode {
                body["response_format"] = json!({"type": "json_object"});
            }
            body
        };

        // Try with JSON mode first; retry without if model rejects it
        let text_body = match self.client.post(&url).json(&make_body(true)).send().await {
            Ok(resp) if resp.status().is_success() => resp.text().await?,
            _ => {
                tracing::info!("OMLX expand JSON mode failed, retrying without response_format");
                let resp = self.client.post(&url).json(&make_body(false)).send().await?;
                let status = resp.status();
                let body = resp.text().await?;
                if !status.is_success() {
                    let truncated: String = body.chars().take(500).collect();
                    return Err(ReinError::Extract(format!("OMLX expand API returned {}: {truncated}", status)));
                }
                body
            }
        };

        let parsed: Value = serde_json::from_str(&text_body)
            .map_err(|e| ReinError::Extract(e.to_string()))?;

        let content = parsed["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| ReinError::Extract("missing choices[0].message.content".into()))?;

        parse_expansion_response(content, max)
    }
}

// ---------------------------------------------------------------------------
// Enum dispatch
// ---------------------------------------------------------------------------

enum ExpanderKind {
    Gemini(GeminiExpander),
    Omlx(OmlxExpander),
}

impl ExpanderKind {
    async fn expand(&self, query: &str, max: usize) -> ReinResult<Vec<String>> {
        match self {
            Self::Gemini(e) => e.expand(query, max).await,
            Self::Omlx(e) => e.expand(query, max).await,
        }
    }
}

fn create_expander(config: &ReinConfig) -> Option<ExpanderKind> {
    match config.expand_provider() {
        Provider::Google => {
            let api_key = config.query_expansion.google.api_key.as_ref()?;
            Some(ExpanderKind::Gemini(GeminiExpander::new(
                api_key.clone(),
                config.query_expansion.google.endpoint.clone(),
                config.query_expansion.google.model.clone(),
            )))
        }
        Provider::Omlx => Some(ExpanderKind::Omlx(OmlxExpander::new(
            config.query_expansion.omlx.endpoint.clone(),
            config.query_expansion.omlx.model.clone(),
        ))),
        Provider::None => None,
    }
}

// ---------------------------------------------------------------------------
// Public entry point (sync, blocks on async internally)
// ---------------------------------------------------------------------------

/// Expand a query into alternative phrasings using LLM.
/// Returns expanded queries (NOT including original). Falls back to empty vec on failure.
/// `max_override` allows callers to request fewer expansions than config default.
pub fn expand_query(config: &ReinConfig, query: &str, max_override: Option<usize>) -> Vec<String> {
    let expander = match create_expander(config) {
        Some(e) => e,
        None => return vec![],
    };

    let max = max_override.unwrap_or(config.query_expansion.max_expansions);
    let provider = &config.query_expansion.provider;

    // Check in-memory cache
    let cache_k = crate::search::cache::cache_key(&[query, provider, &max.to_string()]);
    if let Ok(cache) = crate::search::cache::expand_cache().lock() {
        if let Some(cached) = cache.get(&cache_k) {
            tracing::debug!(query_len = query.len(), "expansion cache hit");
            return cached;
        }
    }

    let expand_start = std::time::Instant::now();

    let result = if let Ok(handle) = tokio::runtime::Handle::try_current() {
        tokio::task::block_in_place(|| handle.block_on(expander.expand(query, max)))
    } else {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| crate::types::error::ReinError::Extract(format!("failed to create runtime: {e}")));
        match rt {
            Ok(rt) => rt.block_on(expander.expand(query, max)),
            Err(e) => Err(e),
        }
    };

    match result {
        Ok(expansions) => {
            tracing::info!(
                count = expansions.len(),
                elapsed_ms = expand_start.elapsed().as_millis() as u64,
                "query expanded"
            );
            for (i, q) in expansions.iter().enumerate() {
                tracing::debug!(idx = i, expanded = %q, "expansion variant");
            }
            // Store in cache
            if let Ok(mut cache) = crate::search::cache::expand_cache().lock() {
                cache.put(cache_k, expansions.clone());
            }
            expansions
        }
        Err(e) => {
            tracing::warn!(
                elapsed_ms = expand_start.elapsed().as_millis() as u64,
                "query expansion failed, using original only: {e}"
            );
            vec![]
        }
    }
}

// ---------------------------------------------------------------------------
// Response parsing
// ---------------------------------------------------------------------------

fn parse_expansion_response(content: &str, max: usize) -> ReinResult<Vec<String>> {
    let cleaned = strip_code_fences(content);
    let parsed: Value = serde_json::from_str(&cleaned)
        .map_err(|e| ReinError::Extract(format!("failed to parse expansion JSON: {e}")))?;

    let arr = if let Some(arr) = parsed.as_array() {
        arr.clone()
    } else if let Some(obj) = parsed.as_object() {
        // Some models wrap: {"queries": [...]} or {"alternatives": [...]}
        obj.values()
            .find_map(|v| v.as_array().cloned())
            .unwrap_or_default()
    } else {
        return Ok(vec![]);
    };

    let expansions: Vec<String> = arr.iter()
        .filter_map(|v| v.as_str().map(|s| s.trim().to_string()))
        .filter(|s| !s.is_empty() && s.len() <= 200)
        .take(max)
        .collect();

    Ok(expansions)
}

/// Strip markdown code fences from LLM output.
fn strip_code_fences(s: &str) -> String {
    let trimmed = s.trim();
    if let Some(rest) = trimmed.strip_prefix("```json") {
        rest.strip_suffix("```").unwrap_or(rest).trim().to_string()
    } else if let Some(rest) = trimmed.strip_prefix("```") {
        rest.strip_suffix("```").unwrap_or(rest).trim().to_string()
    } else {
        trimmed.to_string()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_expansion_array() {
        let input = r#"["user UI theme preference", "dark mode light mode"]"#;
        let result = parse_expansion_response(input, 3).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], "user UI theme preference");
    }

    #[test]
    fn test_parse_expansion_with_fences() {
        let input = "```json\n[\"alternative 1\", \"alternative 2\"]\n```";
        let result = parse_expansion_response(input, 3).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_parse_expansion_wrapped_object() {
        let input = r#"{"queries": ["alt1", "alt2"]}"#;
        let result = parse_expansion_response(input, 3).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_parse_expansion_empty() {
        let input = "[]";
        let result = parse_expansion_response(input, 3).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_parse_expansion_truncates_to_max() {
        let input = r#"["a", "b", "c", "d", "e"]"#;
        let result = parse_expansion_response(input, 2).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_strip_code_fences() {
        assert_eq!(strip_code_fences("```json\n[\"a\"]\n```"), "[\"a\"]");
        assert_eq!(strip_code_fences("[\"a\"]"), "[\"a\"]");
        assert_eq!(strip_code_fences("```\n[\"a\"]\n```"), "[\"a\"]");
    }
}
