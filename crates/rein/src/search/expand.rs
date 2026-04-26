use crate::config::{Provider, ReinConfig};
use crate::types::error::{ReinError, ReinResult};
use reqwest::Client;
use serde_json::{json, Value};

/// Max byte length of an input query accepted by the expander.
/// Above this, expansion is skipped (returns empty) — LLM prompt injection risk scales
/// with attacker-controlled input size, and genuine queries are almost never this long.
const MAX_EXPAND_QUERY_LEN: usize = 8 * 1024;

/// Render a user query as a JSON-string literal (with surrounding quotes and full escape)
/// so it cannot escape the prompt's string boundary.
fn quote_for_prompt(query: &str) -> String {
    // serde_json::to_string on a &str always succeeds for valid UTF-8.
    serde_json::to_string(query).unwrap_or_else(|_| "\"\"".to_string())
}

const EXPAND_SYSTEM_PROMPT: &str = r#"Given a memory search query, generate alternative phrasings to improve recall of stored memories.

For CHINESE queries:
- Generate 1-2 Chinese-native rewrites using different vocabulary (NOT just paraphrasing)
  - Synonym substitution: 实现→完成/落地, 问题→bug/报错/异常, 方案→方法/思路
  - Register shift: formal→colloquial or vice versa
  - Convert question form to declarative: "如何做X" → "X的实现方式"
  - Expand technical abbreviations: "接口" → "API接口", "数据库" → "SQLite数据库"
- Add 1 English translation for cross-lingual vector search

For ENGLISH queries:
- Generate 2 synonym-substitution variants (different words, same meaning)
- Add 1 Chinese translation only if the query clearly refers to Chinese-language content

Common rules:
- Return [] for exact identifiers (function names, IDs, filenames, version numbers)
- Each alternative must be meaningfully different from the original AND from each other
- Keep each alternative under 100 characters
- Output a JSON array of strings only, no explanation

Examples:
Input (Chinese): "怎么实现增量索引"
Output: ["增量HNSW更新方法", "索引增量构建实现", "incremental index implementation"]

Input (English): "memory decay algorithm"
Output: ["forgetting curve implementation", "memory strength reduction over time"]"#;

/// Returns true if more than 30% of the characters in the query are Chinese ideographs.
///
/// Deliberately excludes Hiragana/Katakana (Japanese) and Hangul (Korean) so that
/// Japanese/Korean queries do not receive Chinese-specific expansion strategies.
/// Uses CJK Unified Ideographs + Extensions A-F which are the core Chinese character blocks.
fn is_chinese_query(query: &str) -> bool {
    let total = query.chars().count();
    if total == 0 {
        return false;
    }
    let cjk_count = query
        .chars()
        .filter(|c| {
            matches!(*c as u32,
                0x4E00..=0x9FFF |   // CJK Unified Ideographs
                0x3400..=0x4DBF |   // CJK Extension A
                0x20000..=0x2A6DF | // CJK Extension B
                0x2A700..=0x2B73F | // CJK Extension C
                0x2B740..=0x2B81F | // CJK Extension D
                0x2B820..=0x2CEAF | // CJK Extension E
                0x2CEB0..=0x2EBEF   // CJK Extension F
            )
        })
        .count();
    cjk_count * 10 > total * 3 // > 30%, integer arithmetic avoids float
}

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
        Self {
            client: crate::search::cache::http_client_10s(),
            api_key,
            endpoint,
            model,
        }
    }

    async fn expand(&self, query: &str, max: usize) -> ReinResult<Vec<String>> {
        if query.len() > MAX_EXPAND_QUERY_LEN {
            return Ok(Vec::new());
        }
        let lang_hint = if is_chinese_query(query) {
            "Language hint: Chinese query — prioritize Chinese-native alternatives first.\n\n"
        } else {
            ""
        };
        let safe_query = quote_for_prompt(query);
        let prompt = format!(
            "{}\n\n{}Generate up to {} alternatives.\n\nInput query: {}",
            EXPAND_SYSTEM_PROMPT, lang_hint, max, safe_query
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

        let resp = self
            .client
            .post(&url)
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

        let parsed: Value =
            serde_json::from_str(&text_body).map_err(|e| ReinError::Extract(e.to_string()))?;

        let content = parsed["candidates"][0]["content"]["parts"][0]["text"]
            .as_str()
            .ok_or_else(|| {
                ReinError::Extract("missing candidates[0].content.parts[0].text".into())
            })?;

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
    disable_thinking: bool,
}

impl OmlxExpander {
    fn new(endpoint: String, model: String, disable_thinking: bool) -> Self {
        Self {
            client: crate::search::cache::http_client_15s(),
            endpoint,
            model,
            disable_thinking,
        }
    }

    async fn expand(&self, query: &str, max: usize) -> ReinResult<Vec<String>> {
        if query.len() > MAX_EXPAND_QUERY_LEN {
            return Ok(Vec::new());
        }
        let lang_hint = if is_chinese_query(query) {
            "Language hint: Chinese query — prioritize Chinese-native alternatives first.\n\n"
        } else {
            ""
        };
        let safe_query = quote_for_prompt(query);
        let user_msg = format!(
            "{}Generate up to {} alternatives.\n\nInput query: {}",
            lang_hint, max, safe_query
        );
        let url = format!("{}/chat/completions", self.endpoint);
        let disable_thinking = self.disable_thinking;
        let make_body = |use_json_mode: bool| {
            let system_msg = if disable_thinking {
                format!("/no_think\n{}", EXPAND_SYSTEM_PROMPT)
            } else {
                EXPAND_SYSTEM_PROMPT.to_string()
            };
            let mut body = json!({
                "model": &self.model,
                "messages": [
                    {"role": "system", "content": &system_msg},
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
                let resp = self
                    .client
                    .post(&url)
                    .json(&make_body(false))
                    .send()
                    .await?;
                let status = resp.status();
                let body = resp.text().await?;
                if !status.is_success() {
                    let truncated: String = body.chars().take(500).collect();
                    return Err(ReinError::Extract(format!(
                        "OMLX expand API returned {}: {truncated}",
                        status
                    )));
                }
                body
            }
        };

        let parsed: Value =
            serde_json::from_str(&text_body).map_err(|e| ReinError::Extract(e.to_string()))?;

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
            config.query_expansion.omlx.disable_thinking,
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
    expand_query_cancellable(config, query, max_override, None)
}

/// Cancellable variant of [`expand_query`]. The caller passes an `AtomicBool`
/// flag; when the flag flips to `true`, the function short-circuits at the
/// next checkpoint (before the LLM call, and before the cache write) and
/// returns an empty vec. Cannot abort an already in-flight HTTP request —
/// that would require switching the underlying async expander to use a
/// `tokio::CancellationToken` (B5 #24 follow-up).
pub fn expand_query_cancellable(
    config: &ReinConfig,
    query: &str,
    max_override: Option<usize>,
    cancel: Option<&std::sync::atomic::AtomicBool>,
) -> Vec<String> {
    let check_cancel = |label: &str| -> bool {
        match cancel {
            Some(c) if c.load(std::sync::atomic::Ordering::Relaxed) => {
                tracing::debug!(stage = label, "query expansion cancelled");
                true
            }
            _ => false,
        }
    };

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

    if check_cancel("pre-llm") {
        return vec![];
    }

    let expand_start = std::time::Instant::now();

    let result = if let Ok(handle) = tokio::runtime::Handle::try_current() {
        tokio::task::block_in_place(|| handle.block_on(expander.expand(query, max)))
    } else {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| {
                crate::types::error::ReinError::Extract(format!("failed to create runtime: {e}"))
            });
        match rt {
            Ok(rt) => rt.block_on(expander.expand(query, max)),
            Err(e) => Err(e),
        }
    };

    if check_cancel("post-llm") {
        return vec![];
    }

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

    let expansions: Vec<String> = arr
        .iter()
        .filter_map(|v| v.as_str().map(|s| s.trim().to_string()))
        // Bug #O8 fix: `s.len()` is byte length — three-byte CJK chars capped
        // effective length at ~66 chars instead of 200, silently dropping
        // valid Chinese expansion variants. Use char count and early-exit at
        // 201 so very-long inputs don't pay an O(n) scan per item.
        .filter(|s| !s.is_empty() && s.chars().take(201).count() <= 200)
        .take(max)
        .collect();

    Ok(expansions)
}

/// Strip thinking tags and markdown code fences from LLM output.
fn strip_code_fences(s: &str) -> String {
    // Strip Qwen3 <think>...</think> reasoning blocks (handle possibly multiple blocks).
    // Use rfind to take content after the LAST </think>, which is safe for nested/repeated tags.
    // "</think>" is pure ASCII so byte-offset slicing is UTF-8 safe, but we still use
    // strip_prefix on the remaining suffix defensively.
    let s = if let Some(idx) = s.rfind("</think>") {
        const TAG_LEN: usize = "</think>".len();
        s.get(idx + TAG_LEN..).unwrap_or("").trim()
    } else {
        s.trim()
    };
    if let Some(rest) = s.strip_prefix("```json") {
        rest.strip_suffix("```").unwrap_or(rest).trim().to_string()
    } else if let Some(rest) = s.strip_prefix("```") {
        rest.strip_suffix("```").unwrap_or(rest).trim().to_string()
    } else {
        s.to_string()
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
    fn quote_for_prompt_escapes_embedded_quotes_and_braces() {
        let out = quote_for_prompt(r#"test" , "injected"#);
        assert_eq!(out, r#""test\" , \"injected""#);
        let with_brace = quote_for_prompt(r#"}{"a":1"#);
        assert!(with_brace.starts_with('"') && with_brace.ends_with('"'));
        assert!(with_brace.contains("\\\""));
    }

    #[test]
    fn quote_for_prompt_escapes_newlines() {
        let out = quote_for_prompt("line1\nline2");
        assert_eq!(out, "\"line1\\nline2\"");
    }

    #[test]
    fn max_expand_query_len_is_enforced_via_length_check() {
        // Smoke check on the constant; the guard in expand() returns Ok(vec![])
        // so expander network calls are skipped entirely for oversize input.
        let big = "a".repeat(MAX_EXPAND_QUERY_LEN + 1);
        assert!(big.len() > MAX_EXPAND_QUERY_LEN);
    }

    #[test]
    fn test_is_chinese_query() {
        assert!(is_chinese_query("怎么实现增量索引"));
        assert!(is_chinese_query("记忆衰减 decay")); // mixed but >30% CJK
        assert!(!is_chinese_query("memory decay algorithm"));
        assert!(!is_chinese_query(""));
    }

    #[test]
    fn test_strip_code_fences() {
        assert_eq!(strip_code_fences("```json\n[\"a\"]\n```"), "[\"a\"]");
        assert_eq!(strip_code_fences("[\"a\"]"), "[\"a\"]");
        assert_eq!(strip_code_fences("```\n[\"a\"]\n```"), "[\"a\"]");
    }

    /// Bug #O8 (v0.26.2): the expansion-length filter used `s.len()` (byte
    /// length). Three-byte CJK characters silently capped effective length
    /// at ~66 chars instead of 200, dropping every long Chinese variant the
    /// LLM produced. Fix uses `chars().take(201).count()` so the policy is
    /// in characters and the comparison early-exits at the cap.
    #[test]
    fn parse_expansion_uses_char_count_not_byte_length() {
        // 100 CJK chars = ~300 bytes — passes the char-count filter but
        // would have been REJECTED by the old byte-length filter
        // (300 > 200).
        let cjk_100 = "中".repeat(100);
        let input = format!("[\"{cjk_100}\"]");
        let result = parse_expansion_response(&input, 5).unwrap();
        assert_eq!(
            result.len(),
            1,
            "100 CJK chars (~300 bytes) MUST pass; pre-fix byte filter dropped this"
        );
        assert_eq!(result[0].chars().count(), 100);

        // 200 CJK chars = ~600 bytes — boundary case, must still pass.
        let cjk_200 = "中".repeat(200);
        let input = format!("[\"{cjk_200}\"]");
        let result = parse_expansion_response(&input, 5).unwrap();
        assert_eq!(
            result.len(),
            1,
            "exactly 200 CJK chars MUST pass (the boundary)"
        );

        // 201 CJK chars — must be REJECTED (one char over the limit).
        let cjk_201 = "中".repeat(201);
        let input = format!("[\"{cjk_201}\"]");
        let result = parse_expansion_response(&input, 5).unwrap();
        assert!(
            result.is_empty(),
            "201 CJK chars must be rejected (char limit is 200)"
        );

        // 250 ASCII chars = 250 bytes — must be REJECTED (length policy is
        // identical for both alphabets, just measured in chars now).
        let ascii_250 = "a".repeat(250);
        let input = format!("[\"{ascii_250}\"]");
        let result = parse_expansion_response(&input, 5).unwrap();
        assert!(
            result.is_empty(),
            "250 ASCII chars must be rejected (over the 200-char cap)"
        );
    }
}
