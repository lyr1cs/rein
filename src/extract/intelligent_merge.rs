//! Intelligent memory insertion classifier (POC — shadow mode only).
//!
//! Given two memory candidates (an existing one and an incoming one) whose
//! raw lexical similarity falls into a gray zone, ask an LLM to classify
//! the relationship so future store paths can make semantic decisions
//! (Ignore / Update / Merge / CreateNew) instead of mechanical threshold
//! judgments.
//!
//! This POC only produces the verdict and optional synthesized content. It
//! does NOT yet change store_with_dedup behavior — callers are expected to
//! log the verdict for evaluation and continue with the existing mechanical
//! merge path.

use crate::config::{Provider, ReinConfig};
use crate::types::error::{ReinError, ReinResult};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// The LLM's classification of how the incoming memory relates to the existing one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InsertionVerdict {
    /// Incoming memory adds no new information — discard it, boost existing.
    Ignore,
    /// Incoming is a newer/more accurate version of the same fact — replace existing content.
    Update,
    /// Both contain complementary information — synthesize a unified record.
    Merge,
    /// Related but semantically distinct — create a new record, link them.
    CreateNew,
}

/// The classifier's structured output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerdictResult {
    pub verdict: InsertionVerdict,
    /// Present only for `Update` and `Merge` — LLM-produced synthesized content.
    #[serde(default)]
    pub synthesized: Option<String>,
    /// One-sentence rationale. Useful for shadow-mode evaluation and future provenance.
    #[serde(default)]
    pub reasoning: Option<String>,
}

const CLASSIFY_SYSTEM_PROMPT: &str = r#"You classify how an INCOMING memory relates to an EXISTING memory.

Output one of four verdicts:
- "ignore": incoming adds no new information; discard it
- "update": incoming is a newer/more accurate version of the same fact; replace existing content
- "merge": both hold complementary information; produce a unified synthesis
- "create_new": semantically distinct though topically related; keep both

Rules:
- Prefer "create_new" over "merge" when topics or entities differ meaningfully.
- For "update" and "merge", produce a concise synthesized summary (under 500 chars)
  that preserves temporal anchors (dates) and unique details from both inputs.
- Never fabricate facts not present in either input.
- Keep reasoning to one sentence.

Output a single JSON object with keys: verdict, synthesized (string|null), reasoning (string).
No markdown, no code fences."#;

fn build_user_message(existing: &MemorySnippet, incoming: &MemorySnippet) -> String {
    format!(
        "EXISTING:\ntopic: {}\nsummary: {}\ncontent: {}\n\nINCOMING:\ntopic: {}\nsummary: {}\ncontent: {}",
        existing.topic,
        existing.summary,
        truncate_chars(&existing.content, 1500),
        incoming.topic,
        incoming.summary,
        truncate_chars(&incoming.content, 1500),
    )
}

fn truncate_chars(s: &str, max: usize) -> String {
    let mut out: String = s.chars().take(max).collect();
    if s.chars().count() > max {
        out.push('…');
    }
    out
}

/// Minimal view of a memory needed for classification.
#[derive(Debug, Clone)]
pub struct MemorySnippet {
    pub topic: String,
    pub summary: String,
    pub content: String,
}

impl From<&crate::types::Memory> for MemorySnippet {
    fn from(m: &crate::types::Memory) -> Self {
        Self {
            topic: m.topic.clone(),
            summary: m.summary.clone(),
            content: m.content.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Gray-zone gate — only ask LLM when raw similarity is ambiguous
// ---------------------------------------------------------------------------

/// Gray zone: raw similarity ranges where LLM classification is worth the cost.
/// Below `LOW`: obviously distinct, create new without asking.
/// Above `HIGH`: obviously same, merge without asking.
/// Between: ambiguous, classifier adds value.
pub const GRAYZONE_LOW: f32 = 0.50;
pub const GRAYZONE_HIGH: f32 = 0.85;

/// Returns true if a similarity score sits in the gray zone that benefits
/// from LLM classification.
pub fn is_gray_zone(similarity: f32) -> bool {
    similarity >= GRAYZONE_LOW && similarity < GRAYZONE_HIGH
}

// ---------------------------------------------------------------------------
// Public entry point (sync, blocks on async internally — mirrors expand.rs)
// ---------------------------------------------------------------------------

/// Classify the relationship between two memories. Returns `None` on any
/// failure (no LLM configured, API error, parse error). Never panics.
pub fn classify_insertion(
    config: &ReinConfig,
    existing: &MemorySnippet,
    incoming: &MemorySnippet,
) -> Option<VerdictResult> {
    let classifier = build_classifier(config)?;
    let user_msg = build_user_message(existing, incoming);

    let result = if let Ok(handle) = tokio::runtime::Handle::try_current() {
        tokio::task::block_in_place(|| handle.block_on(classifier.classify(&user_msg)))
    } else {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .ok()?;
        rt.block_on(classifier.classify(&user_msg))
    };

    match result {
        Ok(v) => Some(v),
        Err(e) => {
            tracing::debug!("intelligent_merge classify failed: {e}");
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Backend selection (reuses Gemini / OMLX pattern from expand.rs)
// ---------------------------------------------------------------------------

enum ClassifierKind {
    Gemini(GeminiClassifier),
    Omlx(OmlxClassifier),
}

impl ClassifierKind {
    async fn classify(&self, user_msg: &str) -> ReinResult<VerdictResult> {
        match self {
            Self::Gemini(c) => c.classify(user_msg).await,
            Self::Omlx(c) => c.classify(user_msg).await,
        }
    }
}

fn build_classifier(config: &ReinConfig) -> Option<ClassifierKind> {
    // Prefer the dedicated `[intelligent_merge]` provider block when set;
    // otherwise fall back to `[query_expansion]` so existing setups keep working.
    let (provider, google, omlx) = match config.intelligent_merge.resolved_provider() {
        Provider::None => (
            config.expand_provider(),
            (
                config.query_expansion.google.api_key.clone(),
                config.query_expansion.google.endpoint.clone(),
                config.query_expansion.google.model.clone(),
            ),
            (
                config.query_expansion.omlx.endpoint.clone(),
                config.query_expansion.omlx.model.clone(),
                config.query_expansion.omlx.disable_thinking,
            ),
        ),
        own => (
            own,
            (
                config.intelligent_merge.google.api_key.clone(),
                config.intelligent_merge.google.endpoint.clone(),
                config.intelligent_merge.google.model.clone(),
            ),
            (
                config.intelligent_merge.omlx.endpoint.clone(),
                config.intelligent_merge.omlx.model.clone(),
                config.intelligent_merge.omlx.disable_thinking,
            ),
        ),
    };

    match provider {
        Provider::Google => {
            let api_key = google.0?;
            Some(ClassifierKind::Gemini(GeminiClassifier {
                client: crate::search::cache::http_client_15s(),
                api_key,
                endpoint: google.1,
                model: google.2,
            }))
        }
        Provider::Omlx => Some(ClassifierKind::Omlx(OmlxClassifier {
            client: crate::search::cache::http_client_15s(),
            endpoint: omlx.0,
            model: omlx.1,
            disable_thinking: omlx.2,
        })),
        Provider::None => None,
    }
}

struct GeminiClassifier {
    client: &'static Client,
    api_key: String,
    endpoint: String,
    model: String,
}

impl GeminiClassifier {
    async fn classify(&self, user_msg: &str) -> ReinResult<VerdictResult> {
        let prompt = format!("{CLASSIFY_SYSTEM_PROMPT}\n\n{user_msg}");
        let url = format!(
            "{}/v1beta/models/{}:generateContent",
            self.endpoint, self.model
        );
        let body = json!({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.1
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
                "Gemini classify API returned {}: {}",
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
        parse_verdict(content)
    }
}

struct OmlxClassifier {
    client: &'static Client,
    endpoint: String,
    model: String,
    disable_thinking: bool,
}

impl OmlxClassifier {
    async fn classify(&self, user_msg: &str) -> ReinResult<VerdictResult> {
        let system_msg = if self.disable_thinking {
            format!("/no_think\n{CLASSIFY_SYSTEM_PROMPT}")
        } else {
            CLASSIFY_SYSTEM_PROMPT.to_string()
        };
        let url = format!("{}/chat/completions", self.endpoint);
        let body = json!({
            "model": &self.model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        });

        let resp = self.client.post(&url).json(&body).send().await?;
        let status = resp.status();
        let text_body = resp.text().await?;
        if !status.is_success() {
            return Err(ReinError::Extract(format!(
                "OMLX classify API returned {}: {}",
                status,
                crate::types::truncate_for_error(&text_body, 500)
            )));
        }
        let parsed: Value =
            serde_json::from_str(&text_body).map_err(|e| ReinError::Extract(e.to_string()))?;
        let content = parsed["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| ReinError::Extract("missing choices[0].message.content".into()))?;
        parse_verdict(content)
    }
}

// ---------------------------------------------------------------------------
// Response parsing
// ---------------------------------------------------------------------------

fn parse_verdict(content: &str) -> ReinResult<VerdictResult> {
    let cleaned = strip_fences(content);
    let v: VerdictResult = serde_json::from_str(&cleaned).map_err(|e| {
        ReinError::Extract(format!(
            "verdict JSON parse failed: {e} (preview: {})",
            crate::types::truncate_for_error(&cleaned, 200)
        ))
    })?;
    // Ignore/CreateNew should never carry synthesized content — clear it.
    let cleared = matches!(
        v.verdict,
        InsertionVerdict::Ignore | InsertionVerdict::CreateNew
    );
    Ok(VerdictResult {
        synthesized: if cleared { None } else { v.synthesized },
        ..v
    })
}

fn strip_fences(s: &str) -> String {
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
    fn gray_zone_bounds() {
        assert!(!is_gray_zone(0.0));
        assert!(!is_gray_zone(GRAYZONE_LOW - 0.01));
        assert!(is_gray_zone(GRAYZONE_LOW));
        assert!(is_gray_zone(0.7));
        assert!(!is_gray_zone(GRAYZONE_HIGH));
        assert!(!is_gray_zone(1.0));
    }

    #[test]
    fn parse_all_four_verdicts() {
        let cases = [
            (r#"{"verdict":"ignore","reasoning":"duplicate"}"#, InsertionVerdict::Ignore),
            (r#"{"verdict":"update","synthesized":"new text","reasoning":"newer"}"#, InsertionVerdict::Update),
            (r#"{"verdict":"merge","synthesized":"combined","reasoning":"complementary"}"#, InsertionVerdict::Merge),
            (r#"{"verdict":"create_new","reasoning":"different"}"#, InsertionVerdict::CreateNew),
        ];
        for (json, expected) in cases {
            let v = parse_verdict(json).unwrap();
            assert_eq!(v.verdict, expected);
        }
    }

    #[test]
    fn ignore_verdict_drops_synthesized() {
        // LLM sometimes fills synthesized even when verdict is ignore — we clear it.
        let v = parse_verdict(r#"{"verdict":"ignore","synthesized":"oops","reasoning":"dup"}"#).unwrap();
        assert!(v.synthesized.is_none());
    }

    #[test]
    fn strip_think_block() {
        let s = "<think>reasoning</think>\n{\"verdict\":\"ignore\"}";
        assert_eq!(strip_fences(s), "{\"verdict\":\"ignore\"}");
    }

    #[test]
    fn strip_json_fence() {
        let s = "```json\n{\"verdict\":\"ignore\"}\n```";
        assert_eq!(strip_fences(s), "{\"verdict\":\"ignore\"}");
    }

    #[test]
    fn truncate_chars_safe_on_cjk() {
        let s = "中文测试".repeat(500); // 2000 CJK chars
        let truncated = truncate_chars(&s, 100);
        // 100 chars + ellipsis
        assert_eq!(truncated.chars().count(), 101);
    }

    #[test]
    fn build_user_message_layout() {
        let existing = MemorySnippet {
            topic: "a".into(),
            summary: "sa".into(),
            content: "ca".into(),
        };
        let incoming = MemorySnippet {
            topic: "b".into(),
            summary: "sb".into(),
            content: "cb".into(),
        };
        let msg = build_user_message(&existing, &incoming);
        assert!(msg.contains("EXISTING:"));
        assert!(msg.contains("INCOMING:"));
        assert!(msg.contains("sa"));
        assert!(msg.contains("sb"));
    }
}
