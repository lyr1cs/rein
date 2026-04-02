use crate::config::ReinConfig;
use crate::types::error::{ReinError, ReinResult};
use reqwest::Client;
use serde_json::{json, Value};
use std::time::Duration;


const EXTRACT_SYSTEM_PROMPT: &str = r#"You are a memory extraction system. Analyze the following text from a coding session and extract facts worth remembering long-term.

Output a JSON array of objects. Each object has:
- "topic": short category (e.g., "architecture", "decision", "debug", "workflow", "config", "learning")
- "summary": concise summary under 100 characters
- "content": the full fact to remember (1-3 sentences)
- "keywords": array of 2-5 relevant keywords
- "importance": one of "low", "medium", "high", "critical"
- "should_store": boolean — false if trivial/noisy/not worth remembering
- "quality_confidence": float 0-1, your confidence this is worth remembering long-term (0.9+ critical, 0.7-0.9 useful, 0.4-0.7 maybe, <0.4 probably not)

Rules:
- Skip greetings, acknowledgments, and trivial chatter
- Skip content that looks like secrets (API keys, tokens, passwords)
- Prefer actionable facts: decisions made, bugs fixed, architecture choices, configurations changed
- Support both English and Chinese text
- Return an empty array [] if nothing is worth storing
- Keep summaries concise and content factual"#;

/// A memory extracted by the LLM with structured fields.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct ExtractedMemory {
    pub topic: String,
    pub summary: String,
    pub content: String,
    #[serde(default)]
    pub keywords: Vec<String>,
    #[serde(default = "default_importance")]
    pub importance: String,
    #[serde(default = "default_should_store")]
    pub should_store: bool,
    #[serde(default = "default_quality_confidence")]
    pub quality_confidence: f64,
}

fn default_importance() -> String {
    "medium".to_string()
}

fn default_should_store() -> bool {
    true
}

fn default_quality_confidence() -> f64 {
    0.5
}

// ---------------------------------------------------------------------------
// Full extraction result (knowledge-centric, for hook_stop)
// ---------------------------------------------------------------------------

/// Complete extraction result including memories, concepts, links, and episode.
#[derive(Debug, Clone, Default, serde::Deserialize, serde::Serialize)]
pub struct ExtractionResult {
    #[serde(default)]
    pub memories: Vec<ExtractedMemory>,
    #[serde(default)]
    pub concepts: Vec<ExtractedConcept>,
    #[serde(default)]
    pub links: Vec<ExtractedLink>,
    #[serde(default)]
    pub episode: Option<EpisodeSummary>,
}

/// A knowledge unit (concept) extracted by the LLM.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct ExtractedConcept {
    pub name: String,
    pub definition: String,
    #[serde(default)]
    pub labels: Vec<String>,
    #[serde(default = "default_memoir")]
    pub memoir: String,
    #[serde(default = "default_concept_type")]
    pub concept_type: String,
    #[serde(default = "default_quality_confidence")]
    pub quality_confidence: f64,
}

fn default_memoir() -> String {
    "general".to_string()
}

fn default_concept_type() -> String {
    "fact".to_string()
}

/// A typed relation between two concepts.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct ExtractedLink {
    pub from: String,
    pub to: String,
    pub relation: String,
}

/// Session episode summary.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct EpisodeSummary {
    pub title: String,
    #[serde(default)]
    pub outcome: String,
    #[serde(default)]
    pub decisions: Vec<String>,
}

const EXTRACT_FULL_PROMPT: &str = r#"You are a knowledge extraction system. Analyze the following coding session transcript and extract structured knowledge.

Output a JSON object with these fields:
- "memories": array of facts worth remembering (same format as before)
- "concepts": array of knowledge units to add to the knowledge graph
- "links": array of relationships between concepts
- "episode": session summary object, or null if session is too short

Each memory: {"topic", "summary" (under 100 chars), "content" (1-3 sentences), "keywords" (2-5), "importance" ("low"|"medium"|"high"|"critical"), "should_store" (bool), "quality_confidence" (float 0-1)}

Each concept: {"name" (short identifier), "definition" (1-2 sentences), "labels" (tags), "memoir" (category: "architecture", "debugging", "workflow", "config", "learning", "tooling"), "concept_type" ("fact" or "skill"), "quality_confidence" (float 0-1)}

Each link: {"from" (concept name), "to" (concept name), "relation" (one of: part_of, depends_on, related_to, contradicts, refines, alternative_to, caused_by, instance_of, superseded_by)}
Links must connect concepts within the SAME memoir category.

Episode: {"title" (what the session accomplished), "outcome" (result), "decisions" (array of key decisions made)}

Rules:
- Skip trivial content, greetings, and secrets
- Max 10 concepts, max 10 links per extraction
- concept_type "fact" = declarative knowledge, "skill" = procedural knowledge (how to do X)
- Support both English and Chinese
- When dates are mentioned (explicit or relative), resolve to ISO format and add as keyword: "date:YYYY-MM-DD"
- When user expresses preferences (explicit or implicit), set topic to "user_preference", add "preference" as keyword, preserve exact words
- When information contradicts previous knowledge, set importance to "high", add "knowledge_update" as keyword
- Return {"memories":[],"concepts":[],"links":[],"episode":null} if nothing worth extracting"#;

// ---------------------------------------------------------------------------
// Gemini extractor (Google generateContent API)
// ---------------------------------------------------------------------------

pub struct GeminiExtractor {
    pub(crate) client: Client,
    pub(crate) api_key: String,
    pub(crate) endpoint: String,
    pub(crate) model: String,
}

impl GeminiExtractor {
    pub fn new(api_key: String, endpoint: String, model: String) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(15))
            .build()
            .unwrap_or_default();
        Self { client, api_key, endpoint, model }
    }

    /// Common Gemini API call: send prompt + text, return raw content text from response.
    async fn call_api(&self, system_prompt: &str, text: &str) -> ReinResult<String> {
        let url = format!(
            "{}/v1beta/models/{}:generateContent",
            self.endpoint, self.model
        );
        let body = json!({
            "contents": [{
                "parts": [{"text": format!("{}\n\n---\n\n{}", system_prompt, text)}]
            }],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.1
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
                "Gemini API returned {}: {}", status, crate::types::truncate_for_error(&text_body, 500)
            )));
        }

        let parsed: Value = serde_json::from_str(&text_body)
            .map_err(|e| ReinError::Extract(e.to_string()))?;

        parsed["candidates"][0]["content"]["parts"][0]["text"]
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| ReinError::Extract("missing candidates[0].content.parts[0].text".into()))
    }

    pub async fn extract(&self, text: &str) -> ReinResult<Vec<ExtractedMemory>> {
        let content = self.call_api(EXTRACT_SYSTEM_PROMPT, text).await?;
        parse_llm_json(&content)
    }

    pub async fn extract_full(&self, text: &str) -> ReinResult<ExtractionResult> {
        let content = self.call_api(EXTRACT_FULL_PROMPT, text).await?;
        parse_extraction_result(&content)
    }
}

// ---------------------------------------------------------------------------
// OMLX extractor (OpenAI-compatible chat/completions API)
// Works with: Ollama, LM Studio, vLLM, LocalAI, etc.
// ---------------------------------------------------------------------------

pub struct OmlxExtractor {
    pub(crate) client: Client,
    pub(crate) endpoint: String,
    pub(crate) model: String,
    pub(crate) disable_thinking: bool,
}

impl OmlxExtractor {
    pub fn new(endpoint: String, model: String, disable_thinking: bool) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap_or_default();
        Self { client, endpoint, model, disable_thinking }
    }

    pub async fn extract(&self, text: &str) -> ReinResult<Vec<ExtractedMemory>> {
        let content_text = self.call_and_extract_content(text, EXTRACT_SYSTEM_PROMPT).await?;
        parse_llm_json(&content_text)
    }

    pub async fn extract_full(&self, text: &str) -> ReinResult<ExtractionResult> {
        let content_text = self.call_and_extract_content(text, EXTRACT_FULL_PROMPT).await?;
        parse_extraction_result(&content_text)
    }

    async fn call_and_extract_content(&self, text: &str, system_prompt: &str) -> ReinResult<String> {
        let url = format!("{}/chat/completions", self.endpoint);
        let prefixed_prompt = if self.disable_thinking {
            format!("/no_think\n{}", system_prompt)
        } else {
            system_prompt.to_string()
        };
        let make_body = |use_json_mode: bool| {
            let mut body = json!({
                "model": &self.model,
                "messages": [
                    {"role": "system", "content": &prefixed_prompt},
                    {"role": "user", "content": text}
                ],
                "temperature": 0.1
            });
            if use_json_mode {
                body["response_format"] = json!({"type": "json_object"});
            }
            body
        };

        // Try with JSON mode first; retry without if the model rejects it
        let text_body = match self.client.post(&url).json(&make_body(true)).send().await {
            Ok(resp) if resp.status().is_success() => resp.text().await?,
            _ => {
                tracing::info!("OMLX JSON mode failed, retrying without response_format");
                let resp = self.client.post(&url).json(&make_body(false)).send().await?;
                let status = resp.status();
                let body = resp.text().await?;
                if !status.is_success() {
                    let truncated: String = body.chars().take(500).collect();
                    return Err(ReinError::Extract(format!("OMLX API returned {}: {truncated}", status)));
                }
                body
            }
        };

        let parsed: Value = serde_json::from_str(&text_body)
            .map_err(|e| ReinError::Extract(e.to_string()))?;

        parsed["choices"][0]["message"]["content"]
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| ReinError::Extract("missing choices[0].message.content".into()))
    }
}

// ---------------------------------------------------------------------------
// Enum dispatch
// ---------------------------------------------------------------------------

pub enum ExtractorKind {
    Gemini(GeminiExtractor),
    Omlx(OmlxExtractor),
}

impl ExtractorKind {
    pub async fn extract(&self, text: &str) -> ReinResult<Vec<ExtractedMemory>> {
        match self {
            Self::Gemini(e) => e.extract(text).await,
            Self::Omlx(e) => e.extract(text).await,
        }
    }

    pub async fn extract_full(&self, text: &str) -> ReinResult<ExtractionResult> {
        match self {
            Self::Gemini(e) => e.extract_full(text).await,
            Self::Omlx(e) => e.extract_full(text).await,
        }
    }
}

/// Create an extractor from config. Returns None if provider is "none" or API key is missing.
pub fn create_extractor(config: &ReinConfig) -> Option<ExtractorKind> {
    use crate::config::Provider;
    match config.extract_provider() {
        Provider::Google => {
            let api_key = config.extract.google.api_key.as_ref()?;
            Some(ExtractorKind::Gemini(GeminiExtractor::new(
                api_key.clone(),
                config.extract.google.endpoint.clone(),
                config.extract.google.model.clone(),
            )))
        }
        Provider::Omlx => Some(ExtractorKind::Omlx(OmlxExtractor::new(
            config.extract.omlx.endpoint.clone(),
            config.extract.omlx.model.clone(),
            config.extract.omlx.disable_thinking,
        ))),
        Provider::None => None,
    }
}

// ---------------------------------------------------------------------------
// Fallback: LLM extraction with pattern-based fallback
// ---------------------------------------------------------------------------

/// Truncate input text based on config (safe default for unknown models).
fn prepare_input(config: &ReinConfig, text: &str) -> String {
    let max_chars = resolve_max_input_chars(config);
    if max_chars > 0 { text.chars().take(max_chars).collect() } else { text.to_string() }
}

/// Convert pattern-based facts to ExtractedMemory structs (common fallback path).
fn facts_to_memories(text: &str, threshold: u32) -> Vec<ExtractedMemory> {
    crate::extract::patterns::extract_facts(text, threshold)
        .into_iter()
        .map(|fact| {
            let qc = crate::extract::hooks::scoring::pattern_quality_confidence(&fact);
            ExtractedMemory {
                topic: "auto-extracted".to_string(),
                summary: fact.chars().take(100).collect(),
                content: fact,
                keywords: vec![],
                importance: "medium".to_string(),
                should_store: true,
                quality_confidence: qc,
            }
        })
        .collect()
}

/// Extract memories using LLM if available, falling back to pattern-based extraction.
pub async fn extract_with_fallback(
    config: &ReinConfig,
    text: &str,
    pattern_threshold: u32,
) -> Vec<ExtractedMemory> {
    if let Some(extractor) = create_extractor(config) {
        let input = prepare_input(config, text);
        match extractor.extract(&input).await {
            Ok(memories) if !memories.is_empty() => return memories,
            Ok(_) => {}
            Err(e) => tracing::warn!("LLM extraction failed, falling back to patterns: {e}"),
        }
    }
    facts_to_memories(text, pattern_threshold)
}

/// LLM semantic dedup: ask the model if two texts are about the same thing.
/// Returns true if the LLM judges them as duplicates.
pub async fn llm_is_duplicate(config: &ReinConfig, text_a: &str, text_b: &str) -> bool {
    let Some(extractor) = create_extractor(config) else {
        return false; // no LLM available, can't judge
    };

    let prompt = format!(
        "Are these two texts saying essentially the same thing? Answer ONLY \"yes\" or \"no\".\n\nText A: {}\n\nText B: {}",
        text_a.chars().take(500).collect::<String>(),
        text_b.chars().take(500).collect::<String>(),
    );

    let result = match &extractor {
        ExtractorKind::Gemini(e) => {
            let url = format!("{}/v1beta/models/{}:generateContent", e.endpoint, e.model);
            let body = json!({
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.0, "maxOutputTokens": 5}
            });
            e.client.post(&url)
                .header("x-goog-api-key", &e.api_key)
                .json(&body)
                .send().await
                .ok()
                .and_then(|r| if r.status().is_success() { Some(r) } else { None })
        }
        ExtractorKind::Omlx(e) => {
            let url = format!("{}/chat/completions", e.endpoint);
            let body = json!({
                "model": &e.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0, "max_tokens": 5
            });
            e.client.post(&url)
                .json(&body)
                .send().await
                .ok()
                .and_then(|r| if r.status().is_success() { Some(r) } else { None })
        }
    };

    if let Some(resp) = result {
        if let Ok(body) = resp.text().await {
            let lower = body.to_lowercase();
            // Check for "yes" in various response formats
            // Gemini: candidates[0].content.parts[0].text
            // OMLX: choices[0].message.content
            // Also handle raw "yes"
            return lower.contains("yes");
        }
    }
    false // default: not duplicate (conservative)
}

/// Full extraction (memories + concepts + links + episode) with fallback.
/// Used by hook_stop for session-level knowledge extraction.
pub async fn extract_full_with_fallback(
    config: &ReinConfig,
    text: &str,
) -> ExtractionResult {
    if let Some(extractor) = create_extractor(config) {
        let input = prepare_input(config, text);

        match extractor.extract_full(&input).await {
            Ok(result) => return result,
            Err(e) => {
                tracing::warn!("LLM full extraction failed, falling back to simple: {e}");
                if let Ok(memories) = extractor.extract(&input).await {
                    if !memories.is_empty() {
                        return ExtractionResult { memories, ..Default::default() };
                    }
                }
                tracing::warn!("LLM simple extraction also failed, falling back to patterns");
            }
        }
    }

    ExtractionResult {
        memories: facts_to_memories(text, 2),
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// Input truncation safety
// ---------------------------------------------------------------------------

/// Safe default for models not in the known large-context list.
const SAFE_DEFAULT_MAX_CHARS: usize = 16000;

/// Check if a Gemini model is known to support 1M+ token input.
/// All Gemini 2.0+ models (2.0, 2.5, 3.x) support 1,048,576 input tokens.
fn is_large_context_model(model: &str) -> bool {
    model.starts_with("gemini-2.") ||
    model.starts_with("gemini-3") ||
    model.starts_with("gemini-2-") ||
    model.starts_with("gemini-3.")
}

/// Resolve the effective max_input_chars for the current config.
/// - If the user explicitly set a value (> 0), use it.
/// - If max_input_chars is 0 (no truncation), only allow it for known large-context Gemini models.
/// - For any other model with 0, apply a safe default to prevent API errors and memory loss.
fn resolve_max_input_chars(config: &ReinConfig) -> usize {
    use crate::config::Provider;
    match config.extract_provider() {
        Provider::Omlx => {
            let configured = config.extract.omlx.max_input_chars;
            if configured > 0 { configured } else { SAFE_DEFAULT_MAX_CHARS }
        }
        Provider::Google | Provider::None => {
            let configured = config.extract.google.max_input_chars;
            if configured > 0 {
                return configured;
            }
            // 0 means "no truncation" — only safe for known large-context models
            let model = &config.extract.google.model;
            if is_large_context_model(model) {
                0 // truly no truncation
            } else {
                tracing::info!(
                    "extract model '{}' not recognized as 1M-token model, applying safe limit of {} chars. \
                     Set [extract.google] max_input_chars to override.",
                    model, SAFE_DEFAULT_MAX_CHARS
                );
                SAFE_DEFAULT_MAX_CHARS
            }
        }
    }
}

// ---------------------------------------------------------------------------
// JSON parsing helpers
// ---------------------------------------------------------------------------

/// Parse LLM output as an ExtractionResult (full extraction).
/// Handles partial results: if concepts/links fail to parse, still returns memories.
fn parse_extraction_result(text: &str) -> ReinResult<ExtractionResult> {
    let cleaned = strip_code_fences(text.trim());

    // Try parsing as complete ExtractionResult
    if let Ok(result) = serde_json::from_str::<ExtractionResult>(&cleaned) {
        let mut result = result;
        result.memories.retain(|m| m.should_store);
        return Ok(result);
    }

    // If full parse fails, try extracting just memories from the object
    if let Ok(obj) = serde_json::from_str::<Value>(&cleaned) {
        let mut result = ExtractionResult::default();

        // Try to parse each field independently for partial success
        if let Some(memories_val) = obj.get("memories") {
            if let Ok(memories) = serde_json::from_value::<Vec<ExtractedMemory>>(memories_val.clone()) {
                result.memories = memories.into_iter().filter(|m| m.should_store).collect();
            }
        }
        if let Some(concepts_val) = obj.get("concepts") {
            if let Ok(concepts) = serde_json::from_value::<Vec<ExtractedConcept>>(concepts_val.clone()) {
                result.concepts = concepts;
            }
        }
        if let Some(links_val) = obj.get("links") {
            if let Ok(links) = serde_json::from_value::<Vec<ExtractedLink>>(links_val.clone()) {
                result.links = links;
            }
        }
        if let Some(episode_val) = obj.get("episode") {
            if !episode_val.is_null() {
                result.episode = serde_json::from_value::<EpisodeSummary>(episode_val.clone()).ok();
            }
        }

        if !result.memories.is_empty() || !result.concepts.is_empty() {
            return Ok(result);
        }

        // Last resort: try parsing as a flat memories array
        for (_key, val) in obj.as_object().into_iter().flatten() {
            if let Ok(memories) = serde_json::from_value::<Vec<ExtractedMemory>>(val.clone()) {
                result.memories = memories.into_iter().filter(|m| m.should_store).collect();
                if !result.memories.is_empty() {
                    return Ok(result);
                }
            }
        }
    }

    // Try as plain memories array (backward compat)
    if let Ok(memories) = parse_llm_json(&cleaned) {
        return Ok(ExtractionResult { memories, ..Default::default() });
    }

    Err(ReinError::Extract(format!(
        "failed to parse extraction result: {}",
        cleaned.chars().take(200).collect::<String>()
    )))
}

/// Parse LLM output as a JSON array of ExtractedMemory.
/// Handles common LLM quirks: markdown code fences, top-level object wrapping.
fn parse_llm_json(text: &str) -> ReinResult<Vec<ExtractedMemory>> {
    let cleaned = strip_code_fences(text.trim());

    // Try parsing as a JSON array directly
    if let Ok(memories) = serde_json::from_str::<Vec<ExtractedMemory>>(&cleaned) {
        return Ok(memories.into_iter().filter(|m| m.should_store).collect());
    }

    // Some models wrap in an object like {"memories": [...]} or {"results": [...]}
    if let Ok(obj) = serde_json::from_str::<Value>(&cleaned) {
        if let Some(obj_map) = obj.as_object() {
            for (_key, val) in obj_map {
                if let Ok(memories) = serde_json::from_value::<Vec<ExtractedMemory>>(val.clone()) {
                    return Ok(memories.into_iter().filter(|m| m.should_store).collect());
                }
            }
        }
    }

    Err(ReinError::Extract(format!(
        "failed to parse LLM output as ExtractedMemory array: {}",
        cleaned.chars().take(200).collect::<String>()
    )))
}

/// Strip markdown code fences (```json ... ```) from LLM output.
fn strip_code_fences(text: &str) -> String {
    // Strip Qwen3 <think>...</think> reasoning blocks first
    let trimmed = if let Some(idx) = text.find("</think>") {
        text[idx + 8..].trim()
    } else {
        text.trim()
    };
    if trimmed.starts_with("```") {
        let after_first = if let Some(nl) = trimmed.find('\n') {
            &trimmed[nl + 1..]
        } else {
            trimmed.trim_start_matches("```json").trim_start_matches("```")
        };
        if let Some(end) = after_first.rfind("```") {
            return after_first[..end].trim().to_string();
        }
        return after_first.trim().to_string();
    }
    trimmed.to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_json_array() {
        let json = r#"[
            {"topic": "decision", "summary": "Chose SQLite", "content": "We chose SQLite for storage.", "keywords": ["sqlite", "storage"], "importance": "high", "should_store": true},
            {"topic": "debug", "summary": "Fixed OOM", "content": "Fixed OOM by closing connections.", "keywords": ["oom", "fix"], "importance": "medium", "should_store": true}
        ]"#;
        let result = parse_llm_json(json).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].topic, "decision");
        assert_eq!(result[1].topic, "debug");
    }

    #[test]
    fn test_parse_json_with_code_fences() {
        let json = "```json\n[{\"topic\": \"config\", \"summary\": \"Set up env\", \"content\": \"Configured env vars.\", \"keywords\": [\"env\"], \"importance\": \"low\", \"should_store\": true}]\n```";
        let result = parse_llm_json(json).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].topic, "config");
    }

    #[test]
    fn test_parse_wrapped_object() {
        let json = r#"{"memories": [{"topic": "workflow", "summary": "Deploy step", "content": "Deployed to prod.", "keywords": ["deploy"], "importance": "medium", "should_store": true}]}"#;
        let result = parse_llm_json(json).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_should_store_filtering() {
        let json = r#"[
            {"topic": "chat", "summary": "Greeting", "content": "Hello!", "keywords": [], "importance": "low", "should_store": false},
            {"topic": "decision", "summary": "Chose Rust", "content": "Picked Rust.", "keywords": ["rust"], "importance": "high", "should_store": true}
        ]"#;
        let result = parse_llm_json(json).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].topic, "decision");
    }

    #[test]
    fn test_empty_array() {
        let json = "[]";
        let result = parse_llm_json(json).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_default_fields() {
        let json = r#"[{"topic": "test", "summary": "minimal", "content": "just topic/summary/content"}]"#;
        let result = parse_llm_json(json).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].importance, "medium");
        assert!(result[0].keywords.is_empty());
        assert!(result[0].should_store);
    }

    #[test]
    fn test_strip_code_fences() {
        assert_eq!(strip_code_fences("```json\n[]\n```"), "[]");
        assert_eq!(strip_code_fences("```\n[]\n```"), "[]");
        assert_eq!(strip_code_fences("[]"), "[]");
    }

    #[test]
    fn test_fallback_path() {
        // With provider="none", extract_with_fallback should use patterns
        let mut config = ReinConfig::default();
        config.extract.provider = "none".to_string();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let result = rt.block_on(extract_with_fallback(
            &config,
            "The system uses a microservices architecture. We decided to use PostgreSQL.",
            3,
        ));
        // Should get pattern-based results
        assert!(!result.is_empty());
        assert_eq!(result[0].topic, "auto-extracted");
        assert_eq!(result[0].importance, "medium");
    }

    #[test]
    fn test_parse_extraction_result_full() {
        let json = r#"{
            "memories": [{"topic": "decision", "summary": "Chose SQLite", "content": "We chose SQLite.", "keywords": ["sqlite"], "importance": "high", "should_store": true}],
            "concepts": [{"name": "SQLite", "definition": "Embedded database engine", "labels": ["database"], "memoir": "architecture", "concept_type": "fact"}],
            "links": [{"from": "SQLite", "to": "Embedded DB", "relation": "instance_of"}],
            "episode": {"title": "Database selection", "outcome": "Chose SQLite", "decisions": ["Use SQLite for embedded storage"]}
        }"#;
        let result = parse_extraction_result(json).unwrap();
        assert_eq!(result.memories.len(), 1);
        assert_eq!(result.concepts.len(), 1);
        assert_eq!(result.concepts[0].memoir, "architecture");
        assert_eq!(result.links.len(), 1);
        assert!(result.episode.is_some());
        assert_eq!(result.episode.unwrap().title, "Database selection");
    }

    #[test]
    fn test_parse_extraction_result_partial() {
        // Only memories parse successfully, concepts field is malformed
        let json = r#"{
            "memories": [{"topic": "test", "summary": "works", "content": "partial test"}],
            "concepts": "not an array",
            "links": []
        }"#;
        let result = parse_extraction_result(json).unwrap();
        assert_eq!(result.memories.len(), 1);
        assert!(result.concepts.is_empty()); // failed to parse, defaults to empty
    }

    #[test]
    fn test_parse_extraction_result_missing_fields() {
        // Only memories present, other fields missing entirely
        let json = r#"{"memories": [{"topic": "t", "summary": "s", "content": "c"}]}"#;
        let result = parse_extraction_result(json).unwrap();
        assert_eq!(result.memories.len(), 1);
        assert!(result.concepts.is_empty());
        assert!(result.links.is_empty());
        assert!(result.episode.is_none());
    }

    #[test]
    fn test_parse_extraction_result_defaults() {
        let json = r#"{"memories":[],"concepts":[{"name":"test","definition":"def"}],"links":[],"episode":null}"#;
        let result = parse_extraction_result(json).unwrap();
        assert_eq!(result.concepts[0].memoir, "general"); // default
        assert_eq!(result.concepts[0].concept_type, "fact"); // default
    }

    #[test]
    fn test_full_fallback_path() {
        let mut config = ReinConfig::default();
        config.extract.provider = "none".to_string();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let result = rt.block_on(extract_full_with_fallback(
            &config,
            "We decided to use PostgreSQL. The system uses a microservices architecture.",
        ));
        assert!(!result.memories.is_empty());
        assert!(result.concepts.is_empty()); // no LLM, no concepts
    }

    #[tokio::test]
    #[ignore] // requires live API key in GEMINI_API_KEY env var
    async fn test_gemini_extract_live() {
        let api_key = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY not set");
        let extractor = GeminiExtractor::new(
            api_key,
            "https://generativelanguage.googleapis.com".to_string(),
            "gemini-3.1-flash-lite-preview".to_string(),
        );
        let result = extractor.extract(
            "We decided to use SQLite instead of PostgreSQL because rein needs to be a single binary with no external dependencies. This was a key architecture decision."
        ).await.unwrap();
        assert!(!result.is_empty());
        // Verify structured fields are populated
        for mem in &result {
            assert!(!mem.topic.is_empty());
            assert!(!mem.summary.is_empty());
            assert!(!mem.content.is_empty());
        }
    }
}
