use crate::config::{Provider, ReinConfig};
use crate::types::error::{ReinError, ReinResult};
use crate::types::Memory;
use reqwest::Client;
use serde_json::{json, Value};
use std::time::Duration;

const RERANK_SYSTEM_PROMPT: &str = r#"You are a memory relevance scoring system. Given a search query and a list of stored memories, rate each memory's relevance to the query.

Output a JSON object: {"scores": [s1, s2, ...]}
Each score is a float from 0.0 (irrelevant) to 1.0 (highly relevant).
The scores array must have exactly the same length as the number of memories provided.

Scoring guidelines:
- 0.9-1.0: Direct answer to the query
- 0.7-0.8: Closely related, provides useful context
- 0.4-0.6: Somewhat related, tangentially useful
- 0.1-0.3: Barely related
- 0.0: Completely irrelevant
- Consider both semantic similarity and factual relevance
- Support both English and Chinese"#;

// ---------------------------------------------------------------------------
// Gemini reranker
// ---------------------------------------------------------------------------

struct GeminiReranker {
    client: Client,
    api_key: String,
    endpoint: String,
    model: String,
}

impl GeminiReranker {
    fn new(api_key: String, endpoint: String, model: String) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(15))
            .build()
            .unwrap_or_default();
        Self { client, api_key, endpoint, model }
    }

    async fn rerank(&self, query: &str, candidates: &[(Memory, f32)]) -> ReinResult<Vec<f32>> {
        let memories_text = format_candidates(candidates);
        let prompt = format!(
            "{}\n\nQuery: \"{}\"\n\nMemories:\n{}",
            RERANK_SYSTEM_PROMPT, query, memories_text
        );
        let url = format!(
            "{}/v1beta/models/{}:generateContent",
            self.endpoint, self.model
        );
        let body = json!({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.0
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
                "Gemini rerank API returned {}: {}",
                status,
                crate::types::truncate_for_error(&text_body, 500)
            )));
        }

        let parsed: Value = serde_json::from_str(&text_body)
            .map_err(|e| ReinError::Extract(e.to_string()))?;

        let content = parsed["candidates"][0]["content"]["parts"][0]["text"]
            .as_str()
            .ok_or_else(|| ReinError::Extract("missing candidates[0].content.parts[0].text".into()))?;

        parse_rerank_response(content, candidates.len())
    }
}

// ---------------------------------------------------------------------------
// OMLX reranker (OpenAI-compatible)
// ---------------------------------------------------------------------------

struct OmlxReranker {
    client: Client,
    endpoint: String,
    model: String,
}

impl OmlxReranker {
    fn new(endpoint: String, model: String) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(20))
            .build()
            .unwrap_or_default();
        Self { client, endpoint, model }
    }

    async fn rerank(&self, query: &str, candidates: &[(Memory, f32)]) -> ReinResult<Vec<f32>> {
        let memories_text = format_candidates(candidates);
        let user_msg = format!(
            "Query: \"{}\"\n\nMemories:\n{}",
            query, memories_text
        );
        let url = format!("{}/chat/completions", self.endpoint);
        let make_body = |use_json_mode: bool| {
            let mut body = json!({
                "model": &self.model,
                "messages": [
                    {"role": "system", "content": RERANK_SYSTEM_PROMPT},
                    {"role": "user", "content": &user_msg}
                ],
                "temperature": 0.0
            });
            if use_json_mode {
                body["response_format"] = json!({"type": "json_object"});
            }
            body
        };

        let text_body = match self.client.post(&url).json(&make_body(true)).send().await {
            Ok(resp) if resp.status().is_success() => resp.text().await?,
            _ => {
                tracing::info!("OMLX rerank JSON mode failed, retrying without response_format");
                let resp = self.client.post(&url).json(&make_body(false)).send().await?;
                let status = resp.status();
                let body = resp.text().await?;
                if !status.is_success() {
                    let truncated: String = body.chars().take(500).collect();
                    return Err(ReinError::Extract(format!("OMLX rerank API returned {}: {truncated}", status)));
                }
                body
            }
        };

        let parsed: Value = serde_json::from_str(&text_body)
            .map_err(|e| ReinError::Extract(e.to_string()))?;

        let content = parsed["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| ReinError::Extract("missing choices[0].message.content".into()))?;

        parse_rerank_response(content, candidates.len())
    }
}

// ---------------------------------------------------------------------------
// Enum dispatch
// ---------------------------------------------------------------------------

enum RerankerKind {
    Gemini(GeminiReranker),
    Omlx(OmlxReranker),
}

impl RerankerKind {
    async fn rerank(&self, query: &str, candidates: &[(Memory, f32)]) -> ReinResult<Vec<f32>> {
        match self {
            Self::Gemini(r) => r.rerank(query, candidates).await,
            Self::Omlx(r) => r.rerank(query, candidates).await,
        }
    }
}

fn create_reranker(config: &ReinConfig) -> Option<RerankerKind> {
    match config.reranker_provider() {
        Provider::Google => {
            // Reuse query_expansion google config for API key and endpoint
            let api_key = config.query_expansion.google.api_key.as_ref()?;
            Some(RerankerKind::Gemini(GeminiReranker::new(
                api_key.clone(),
                config.query_expansion.google.endpoint.clone(),
                config.query_expansion.google.model.clone(),
            )))
        }
        Provider::Omlx => Some(RerankerKind::Omlx(OmlxReranker::new(
            config.query_expansion.omlx.endpoint.clone(),
            config.query_expansion.omlx.model.clone(),
        ))),
        Provider::None => None,
    }
}

// ---------------------------------------------------------------------------
// Public entry point (sync)
// ---------------------------------------------------------------------------

/// Rerank candidates using LLM. Returns new scores aligned with input order.
/// Falls back to original scores on failure.
pub fn rerank_with_llm(
    config: &ReinConfig,
    query: &str,
    candidates: &[(Memory, f32)],
) -> Vec<f32> {
    let reranker = match create_reranker(config) {
        Some(r) => r,
        None => return candidates.iter().map(|(_, s)| *s).collect(),
    };

    let top_n = config.search.llm_reranker_top_n.min(candidates.len());
    let rerank_start = std::time::Instant::now();

    let result = if let Ok(handle) = tokio::runtime::Handle::try_current() {
        tokio::task::block_in_place(|| handle.block_on(reranker.rerank(query, &candidates[..top_n])))
    } else {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| ReinError::Extract(format!("failed to create runtime: {e}")))
            .and_then(|rt| rt.block_on(reranker.rerank(query, &candidates[..top_n])))
    };

    match result {
        Ok(mut scores) => {
            tracing::info!(
                count = scores.len(),
                elapsed_ms = rerank_start.elapsed().as_millis() as u64,
                "llm reranked"
            );
            // Scale LLM scores to [0, 2.0] to match linear reranker output range
            for s in &mut scores {
                *s *= 2.0;
            }
            // Append original scores for candidates beyond top_n
            scores.extend(candidates[top_n..].iter().map(|(_, s)| *s));
            scores
        }
        Err(e) => {
            tracing::warn!(
                elapsed_ms = rerank_start.elapsed().as_millis() as u64,
                "llm reranking failed, keeping linear scores: {e}"
            );
            candidates.iter().map(|(_, s)| *s).collect()
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn format_candidates(candidates: &[(Memory, f32)]) -> String {
    candidates.iter().enumerate().map(|(i, (mem, _score))| {
        let preview: String = mem.content.chars().take(200).collect();
        let date = mem.created_at.format("%Y-%m-%d");
        format!(
            "{}. [{}] {} (created: {}, accessed: {} times)",
            i + 1, mem.topic, preview, date, mem.access_count
        )
    }).collect::<Vec<_>>().join("\n")
}

fn parse_rerank_response(content: &str, expected_len: usize) -> ReinResult<Vec<f32>> {
    let cleaned = strip_code_fences(content);
    let parsed: Value = serde_json::from_str(&cleaned)
        .map_err(|e| ReinError::Extract(format!("failed to parse rerank JSON: {e}")))?;

    // Try {"scores": [...]} format
    let arr = if let Some(arr) = parsed.get("scores").and_then(|v| v.as_array()) {
        arr.clone()
    } else if let Some(arr) = parsed.as_array() {
        arr.clone()
    } else {
        return Err(ReinError::Extract("rerank response must be {\"scores\": [...]} or [...]".into()));
    };

    let scores: Vec<f32> = arr.iter()
        .map(|v| v.as_f64().unwrap_or(0.0) as f32)
        .map(|s| s.clamp(0.0, 1.0))
        .collect();

    if scores.len() != expected_len {
        tracing::warn!(
            expected = expected_len,
            got = scores.len(),
            "rerank score count mismatch, padding/truncating"
        );
        let mut result = scores;
        result.resize(expected_len, 0.0);
        Ok(result)
    } else {
        Ok(scores)
    }
}

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
// Strong signal detection
// ---------------------------------------------------------------------------

/// Detect if BM25 top result is significantly stronger than runner-up.
/// Returns true if we should skip LLM reranking (and optionally expansion).
pub fn detect_strong_signal(fts_ranked: &[(String, f32)]) -> bool {
    if fts_ranked.len() < 2 {
        return false;
    }

    let mut scores: Vec<f32> = fts_ranked.iter()
        .map(|(_, s)| *s)
        .filter(|s| *s > 0.0) // Only positive Tantivy scores, not FTS5 rank sentinels
        .collect();
    scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    if scores.len() < 2 {
        return false;
    }

    let top1 = scores[0];
    let top2 = scores[1];

    // Strong signal: top1 is at least 1.5x the runner-up
    if top2 > 0.0 && top1 / top2 >= 1.5 {
        tracing::debug!(top1, top2, ratio = top1 / top2, "strong BM25 signal detected");
        return true;
    }

    false
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_rerank_scores() {
        let input = r#"{"scores": [0.9, 0.3, 0.7]}"#;
        let result = parse_rerank_response(input, 3).unwrap();
        assert_eq!(result.len(), 3);
        assert!((result[0] - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_parse_rerank_bare_array() {
        let input = "[0.8, 0.2, 0.5]";
        let result = parse_rerank_response(input, 3).unwrap();
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_parse_rerank_with_fences() {
        let input = "```json\n{\"scores\": [0.5, 0.5]}\n```";
        let result = parse_rerank_response(input, 2).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_parse_rerank_clamps() {
        let input = r#"{"scores": [1.5, -0.3]}"#;
        let result = parse_rerank_response(input, 2).unwrap();
        assert!((result[0] - 1.0).abs() < 0.01);
        assert!((result[1] - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_parse_rerank_length_mismatch() {
        let input = r#"{"scores": [0.5]}"#;
        let result = parse_rerank_response(input, 3).unwrap();
        assert_eq!(result.len(), 3);
        assert!((result[2] - 0.0).abs() < 0.01); // padded with 0
    }

    #[test]
    fn test_strong_signal_detected() {
        let ranked = vec![
            ("a".to_string(), 10.0),
            ("b".to_string(), 3.0),
            ("c".to_string(), 2.0),
        ];
        assert!(detect_strong_signal(&ranked));
    }

    #[test]
    fn test_strong_signal_not_detected() {
        let ranked = vec![
            ("a".to_string(), 5.0),
            ("b".to_string(), 4.5),
            ("c".to_string(), 4.0),
        ];
        assert!(!detect_strong_signal(&ranked));
    }

    #[test]
    fn test_strong_signal_single_result() {
        let ranked = vec![("a".to_string(), 10.0)];
        assert!(!detect_strong_signal(&ranked));
    }

    #[test]
    fn test_strong_signal_ignores_negative_scores() {
        // FTS5 rank sentinels are negative
        let ranked = vec![
            ("a".to_string(), 8.0),
            ("b".to_string(), -1.0),
            ("c".to_string(), -2.0),
        ];
        assert!(!detect_strong_signal(&ranked)); // Only 1 positive score → false
    }
}
