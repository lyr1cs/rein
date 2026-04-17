//! Shared business operations used by both CLI (main.rs) and MCP (server.rs).
//! Extracted to prevent logic drift between the two entrypoints.

use crate::config::ReinConfig;
use crate::extract;
use crate::extract::llm::ExtractedMemory;
use crate::store::SqliteStore;
use crate::types::*;

pub mod adaptive;
pub mod consolidation;
pub mod dedup;
pub mod registry;

pub use adaptive::*;
pub use consolidation::*;
pub use dedup::*;
pub use registry::*;

/// Build a Memory struct from user-provided fields.
/// Used by both `rein store` CLI and `rein_store` MCP tool.
pub fn build_memory(
    config: &ReinConfig,
    topic: String,
    content: String,
    importance: Importance,
    keywords: Vec<String>,
    source: Source,
) -> Memory {
    // Apply rule-based postprocessing for additive keyword enrichment only.
    // User-supplied topic and importance are authoritative — postprocess cannot override them.
    let summary: String = content
        .chars()
        .take(crate::types::SUMMARY_MAX_CHARS)
        .collect();
    let mut extracted = crate::extract::llm::ExtractedMemory {
        topic: topic.clone(),
        summary: summary.clone(),
        content: content.clone(),
        keywords: keywords.clone(),
        importance: format!("{}", importance),
        should_store: true,
        quality_confidence: 0.5,
    };
    crate::extract::postprocess::postprocess(&mut extracted);

    // Only take enriched keywords from postprocess; keep caller's topic and importance
    Memory {
        id: ulid::Ulid::new().to_string(),
        layer: importance.auto_layer(),
        topic,
        summary,
        content,
        keywords: extracted.keywords, // enriched with date/preference/update keywords
        importance,
        source,
        strength: 1.0,
        decay_lambda: config.decay.base_lambda * importance.decay_factor(),
        access_count: 0,
        superseded_by: None,
        canonical_id: None,
        support_count: 1,
        merge_count: 0,
        dedup_confidence: 1.0,
        source_diversity: 1.0,
        contradiction_score: 0.0,
        related_ids: vec![],
        concept_ids: vec![],
        status: MemoryStatus::default(),
        embedding: None,
        tier: MemoryTier::Warm,
        cluster_id: None,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        last_accessed: chrono::Utc::now(),
    }
}

/// Return the dedup similarity to use for a run.
///
/// A1 full rollout: prefer the learned per-cluster / global threshold from
/// AdaptiveState, fall back to the static config value only when no adaptive
/// snapshot exists yet (first run, tests). Callers that know a specific
/// cluster should call `AdaptiveState::get_dedup_threshold(Some(cluster))`
/// directly; this helper is for "global default" call sites.
pub fn effective_dedup_threshold(store: &crate::store::SqliteStore, config: &ReinConfig) -> f32 {
    // Honor [adaptive] enabled=false: when the operator has explicitly disabled
    // the adaptive engine we must not silently keep applying a stale learned
    // threshold from an older snapshot. Fall back to the static config value.
    if !config.adaptive.enabled {
        return config.search.dedup_similarity as f32;
    }
    match crate::store::adaptive::AdaptiveState::restore_snapshot(store.conn()) {
        Some(state) => state.get_dedup_threshold(None),
        None => config.search.dedup_similarity as f32,
    }
}

const AUTO_CONCEPT_MEMOIR: &str = "auto-discovered";
const AUTO_CONCEPT_MAX: usize = 3;
const AUTO_CONCEPT_MIN_CHARS: usize = 80;
const AUTO_CONCEPT_MAX_KEYWORDS: usize = 6;
const AUTO_CONCEPT_MAX_KWS_IN_DEF: usize = 160;

#[derive(Debug, Clone)]
struct AutoConceptSeed {
    name: String,
    source: &'static str,
}

#[derive(Debug, Clone)]
struct ConflictMarker {
    score: f32,
    evidence_keyword: String,
    matched_concept: String,
}

/// Store a memory with full post-processing (shared by CLI and MCP paths).
/// Runs: store_with_dedup → conflict marking → auto concept discovery → auto_link
/// → activate_related_concepts → apply_evolution.
pub fn store_memory(
    store: &crate::store::SqliteStore,
    config: &ReinConfig,
    memory: Memory,
) -> crate::types::ReinResult<String> {
    let original_id = memory.id.clone();
    // A1: use adaptive threshold when available, fall back to static config.
    // store_with_dedup internally resolves per-cluster threshold per candidate;
    // this is the coarse pre-filter default.
    let dedup_sim = effective_dedup_threshold(store, config);
    let id = store.store_with_dedup(memory, dedup_sim, config.search.dedup_time_window_days)?;
    let stored_memory = store.get(&id)?;

    if let Some(marker) = detect_conflict_marker(store, &stored_memory)? {
        if let Err(e) = persist_conflict_marker(store, &stored_memory, &marker) {
            tracing::warn!(
                memory_id = %stored_memory.id,
                conflict_score = marker.score,
                matched_concept = %marker.matched_concept,
                "failed to persist conflict marker: {e}"
            );
        }
    }

    let auto_concepts = derive_auto_concepts(&stored_memory);
    if !auto_concepts.is_empty() {
        let source_memory_ids = vec![id.clone()];
        if let Err(e) =
            store.store_knowledge_units_with_sources(&auto_concepts, &[], &source_memory_ids)
        {
            tracing::warn!(
                memory_id = %stored_memory.id,
                concept_count = auto_concepts.len(),
                "failed to store auto-discovered concepts: {e}"
            );
        }
    }

    // Only run post-processing for newly created memories, not merge-into targets.
    // store_with_dedup returns the existing ID on MergeInto — running evolution
    // against a merged record could corrupt provenance of unrelated memories.
    let is_new = id == original_id;
    if is_new {
        let content = stored_memory.content.clone();
        let _ = store.auto_link(&id, dedup_sim, 5);
        let _ = store.activate_related_concepts(&content);
        let _ = store.apply_evolution(&id, &content, None);
    }
    Ok(id)
}

fn derive_auto_concepts(memory: &Memory) -> Vec<crate::extract::llm::ExtractedConcept> {
    if !should_auto_discover_concepts(memory) {
        return vec![];
    }

    let evidence = concept_evidence(memory);
    let concept_type = infer_auto_concept_type(memory);
    let mut seeds: Vec<AutoConceptSeed> = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for (raw, source) in auto_concept_candidates(memory) {
        let name = normalize_auto_concept_name(&raw);
        if !is_useful_auto_concept_name(&name) || !seen.insert(name.clone()) {
            continue;
        }
        seeds.push(AutoConceptSeed { name, source });
        if seeds.len() >= AUTO_CONCEPT_MAX {
            break;
        }
    }

    seeds
        .into_iter()
        .map(|seed| crate::extract::llm::ExtractedConcept {
            name: seed.name.clone(),
            definition: match seed.source {
                "topic" => format!(
                    "{} summarizes the main subject of this memory. Evidence: {}",
                    seed.name.replace('-', " "),
                    evidence
                ),
                "keyword" => format!(
                    "{} is a locally discovered keyword from this memory. Evidence: {}",
                    seed.name.replace('-', " "),
                    evidence
                ),
                _ => format!(
                    "{} was inferred locally from the stored memory. Evidence: {}",
                    seed.name.replace('-', " "),
                    evidence
                ),
            },
            labels: vec!["auto".to_string(), seed.source.to_string()],
            memoir: AUTO_CONCEPT_MEMOIR.to_string(),
            concept_type: concept_type.to_string(),
            quality_confidence: 0.35,
        })
        .collect()
}

fn should_auto_discover_concepts(memory: &Memory) -> bool {
    let text = format!("{} {} {}", memory.topic, memory.summary, memory.content);
    if text.chars().count() < AUTO_CONCEPT_MIN_CHARS {
        return false;
    }

    !memory.topic.trim().is_empty() || !memory.keywords.is_empty()
}

fn detect_conflict_marker(
    store: &crate::store::SqliteStore,
    memory: &Memory,
) -> crate::types::ReinResult<Option<ConflictMarker>> {
    if !has_conflict_signal(memory) {
        return Ok(None);
    }

    let query = build_conflict_query(memory);
    if query.is_empty() {
        return Ok(None);
    }

    let candidates = store.search_all_concepts(&query, 5)?;
    if candidates.is_empty() {
        return Ok(None);
    }

    let memory_text = format!("{} {} {}", memory.topic, memory.summary, memory.content);
    let mut best: Option<ConflictMarker> = None;

    for concept in candidates {
        let sim = crate::extract::similarity(&memory_text, &concept.definition);
        if sim >= 0.45 {
            continue;
        }

        let score = (0.80 - sim).clamp(0.0, 0.80);
        let evidence_keyword = format!("conflict-{}", normalize_auto_concept_name(&concept.name));
        let candidate = ConflictMarker {
            score,
            evidence_keyword,
            matched_concept: concept.name.clone(),
        };

        if best
            .as_ref()
            .map(|existing| candidate.score > existing.score)
            .unwrap_or(true)
        {
            best = Some(candidate);
        }
    }

    Ok(best)
}

fn persist_conflict_marker(
    store: &crate::store::SqliteStore,
    memory: &Memory,
    marker: &ConflictMarker,
) -> crate::types::ReinResult<()> {
    let mut keywords = memory.keywords.clone();
    if !keywords.iter().any(|kw| kw == "conflict") {
        keywords.push("conflict".to_string());
    }
    if !keywords.iter().any(|kw| kw == &marker.evidence_keyword) {
        keywords.push(marker.evidence_keyword.clone());
    }
    if keywords.len() > 16 {
        keywords.truncate(16);
    }

    let score = memory.contradiction_score.max(marker.score);
    let keywords_json = serde_json::to_string(&keywords)?;

    // Wrap both UPDATEs in a SAVEPOINT so a failure between them cannot leave
    // memory_canonical_state.contradiction_score set while memories.keywords
    // still reflects the pre-conflict state (partial-write corruption).
    store
        .conn()
        .execute_batch("SAVEPOINT persist_conflict_marker")?;
    let result: crate::types::ReinResult<()> = (|| {
        let rows = store.conn().execute(
            "UPDATE memory_canonical_state SET contradiction_score = ?1 WHERE memory_id = ?2",
            rusqlite::params![score, memory.id],
        )?;
        if rows == 0 {
            return Err(crate::types::ReinError::NotFound(format!(
                "memory {} not found (canonical_state)",
                memory.id
            )));
        }
        let now = chrono::Utc::now().to_rfc3339();
        let rows = store.conn().execute(
            "UPDATE memories SET keywords = ?1, updated_at = ?2 WHERE id = ?3",
            rusqlite::params![keywords_json, now, memory.id],
        )?;
        if rows == 0 {
            return Err(crate::types::ReinError::NotFound(format!(
                "memory {} not found (memories)",
                memory.id
            )));
        }
        Ok(())
    })();
    match result {
        Ok(()) => {
            store
                .conn()
                .execute_batch("RELEASE persist_conflict_marker")?;
            Ok(())
        }
        Err(e) => {
            let _ = store
                .conn()
                .execute_batch("ROLLBACK TO persist_conflict_marker");
            let _ = store
                .conn()
                .execute_batch("RELEASE persist_conflict_marker");
            Err(e)
        }
    }
}

fn auto_concept_candidates(memory: &Memory) -> Vec<(String, &'static str)> {
    let mut out: Vec<(String, &'static str)> = Vec::new();
    let mut seen = std::collections::HashSet::new();

    let topic = normalize_auto_concept_name(&memory.topic);
    push_auto_concept_candidate(&mut out, &mut seen, topic, "topic");

    for raw in memory.keywords.iter().cloned().chain(
        crate::extract::extract_keywords_from_text(
            &format!("{} {}", memory.summary, memory.content),
            AUTO_CONCEPT_MAX_KEYWORDS,
        ),
    ) {
        if out.len() >= AUTO_CONCEPT_MAX {
            break;
        }
        push_auto_concept_candidate(
            &mut out,
            &mut seen,
            normalize_auto_concept_name(&raw),
            "keyword",
        );
    }

    out
}

fn push_auto_concept_candidate(
    out: &mut Vec<(String, &'static str)>,
    seen: &mut std::collections::HashSet<String>,
    name: String,
    source: &'static str,
) {
    if !is_useful_auto_concept_name(&name) || !seen.insert(name.clone()) {
        return;
    }
    out.push((name, source));
}

fn normalize_auto_concept_name(raw: &str) -> String {
    let sanitized: String = raw
        .trim()
        .chars()
        .map(|ch| {
            if ch.is_alphanumeric() || ch.is_whitespace() || ch == '-' || ch == '_' {
                ch
            } else {
                ' '
            }
        })
        .collect();
    crate::store::memoir::normalize_concept_name(&sanitized)
}

fn is_useful_auto_concept_name(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }

    let normalized = name.trim().to_lowercase();
    if normalized.chars().all(|ch| ch.is_ascii_digit()) {
        return false;
    }

    const GENERIC: &[&str] = &[
        "auto",
        "content",
        "general",
        "idea",
        "info",
        "information",
        "knowledge",
        "manual",
        "memory",
        "note",
        "notes",
        "summary",
        "thing",
        "things",
        "stuff",
        "update",
        "updated",
        "change",
        "changed",
        "switch",
        "switched",
        "move",
        "moved",
        "use",
        "used",
        "using",
        "fix",
        "fixed",
        "run",
        "running",
        "create",
        "created",
        "store",
        "stored",
        "concept",
        "concepts",
        "knowledge-update",
        "conflict",
        "conflict-update",
    ];

    if GENERIC.iter().any(|bad| normalized == *bad) {
        return false;
    }

    normalized.chars().count() >= 2
}

fn concept_evidence(memory: &Memory) -> String {
    let source = if memory.summary.trim().is_empty() {
        memory.content.trim()
    } else {
        memory.summary.trim()
    };
    let evidence: String = source
        .chars()
        .filter(|c| *c != '\n' && *c != '\r')
        .take(AUTO_CONCEPT_MAX_KWS_IN_DEF)
        .collect();
    if evidence.is_empty() {
        memory
            .content
            .chars()
            .filter(|c| *c != '\n' && *c != '\r')
            .take(AUTO_CONCEPT_MAX_KWS_IN_DEF)
            .collect()
    } else {
        evidence
    }
}

fn infer_auto_concept_type(memory: &Memory) -> &'static str {
    let text = format!("{} {} {}", memory.topic, memory.summary, memory.content).to_lowercase();
    let skill_signals = [
        "how to",
        "steps",
        "procedure",
        "workflow",
        "run ",
        "configure",
        "configuration",
        "install",
        "deploy",
        "debug",
        "fix ",
        "migrate",
        "use ",
        "guide",
        "recipe",
        "操作",
        "步骤",
        "配置",
        "调试",
    ];
    if skill_signals.iter().any(|sig| text.contains(sig)) {
        "skill"
    } else {
        "fact"
    }
}

fn has_conflict_signal(memory: &Memory) -> bool {
    let text = format!("{} {} {}", memory.topic, memory.summary, memory.content);
    if crate::extract::postprocess::is_knowledge_update(&text) {
        return true;
    }
    let lower = text.to_lowercase();
    [
        "conflict",
        "contradict",
        "contradiction",
        "inconsistent",
        "inconsistency",
        "no longer",
        "instead of",
        "replaced",
        "replacing",
        "switched to",
        "moved to",
        "deprecated",
        "superseded",
        "not using",
        "does not use",
        "doesn't use",
        "wrong",
        "false",
        "不再",
        "冲突",
        "矛盾",
        "替换",
        "改为",
        "不使用",
        "不一致",
    ]
    .iter()
    .any(|sig| lower.contains(sig))
}

fn build_conflict_query(memory: &Memory) -> String {
    let source = format!(
        "{} {} {}",
        memory.content.replace('-', " "),
        memory.summary,
        memory.topic
    );
    let mut parts: Vec<String> = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for raw in source.split_whitespace() {
        let term = normalize_conflict_query_token(raw);
        if !is_useful_conflict_query_token(&term) || !seen.insert(term.clone()) {
            continue;
        }
        parts.push(term);
        if parts.len() >= 2 {
            break;
        }
    }

    if parts.is_empty() {
        for raw in memory.keywords.iter().cloned().chain(
            crate::extract::extract_keywords_from_text(
                &format!("{} {}", memory.summary, memory.content.replace('-', " ")),
                AUTO_CONCEPT_MAX_KEYWORDS,
            ),
        ) {
            let term = normalize_conflict_query_token(&raw);
            if !is_useful_conflict_query_token(&term) || !seen.insert(term.clone()) {
                continue;
            }
            parts.push(term);
            if parts.len() >= 2 {
                break;
            }
        }
    }

    parts.join(" ")
}

fn normalize_conflict_query_token(raw: &str) -> String {
    let mut token = normalize_auto_concept_name(raw);
    if token.len() > 3 && token.ends_with('s') && !token.ends_with("ss") {
        token.pop();
    }
    if token.ends_with("ies") && token.len() > 4 {
        token.truncate(token.len() - 3);
        token.push('y');
    }
    token
}

fn is_useful_conflict_query_token(token: &str) -> bool {
    if token.is_empty() {
        return false;
    }
    if token.chars().all(|ch| ch.is_ascii_digit()) {
        return false;
    }

    const STOPWORDS: &[&str] = &[
        "a",
        "an",
        "and",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "for",
        "from",
        "in",
        "into",
        "is",
        "it",
        "no",
        "not",
        "of",
        "on",
        "or",
        "so",
        "the",
        "to",
        "we",
        "with",
        "you",
        "only",
        "just",
        "longer",
        "model",
        "use",
        "used",
        "using",
        "switch",
        "switched",
        "move",
        "moved",
        "replace",
        "replaced",
        "update",
        "updated",
        "change",
        "changed",
        "storage",
        "model",
        "models",
        "system",
        "systems",
        "data",
        "state",
        "note",
        "notes",
        "memory",
        "memories",
        "knowledge",
    ];

    !STOPWORDS.contains(&token)
}

/// Ingest a full session/transcript through the full extraction path.
/// Produces memories, concepts, links, and an optional episode summary.
pub async fn ingest_session_text(
    config: &ReinConfig,
    text: &str,
    agent_label: Option<&str>,
    is_subagent: bool,
) -> crate::types::ReinResult<(u32, u32, u32)> {
    let report = ingest_session_text_report(config, text, agent_label, is_subagent).await?;
    Ok((report.memory_count, report.concept_count, report.link_count))
}

/// Render a structured session payload into extraction-friendly text.
pub fn render_session_ingest(session: &SessionIngest) -> String {
    let mut lines = Vec::new();

    if let Some(title) = session.title.as_deref() {
        lines.push(format!("[Session title: {title}]"));
    }
    if let Some(session_id) = session.session_id.as_deref() {
        lines.push(format!("[Session id: {session_id}]"));
    }
    if let Some(started_at) = session.started_at {
        lines.push(format!("[Session started_at: {}]", started_at.to_rfc3339()));
    }
    if let Some(summary) = session.summary.as_deref() {
        lines.push(format!("[Session summary]\n{summary}"));
    }
    if let Some(compact_summary) = session.compact_summary.as_deref() {
        lines.push(format!("[Compact summary]\n{compact_summary}"));
    }
    for output in &session.tool_outputs {
        if !output.trim().is_empty() {
            lines.push(format!("[Tool output]\n{}", output.trim()));
        }
    }

    for turn in &session.turns {
        if turn.content.trim().is_empty() {
            continue;
        }
        let role = if turn.role.trim().is_empty() {
            "Unknown"
        } else {
            turn.role.trim()
        };
        lines.push(format!("{role}: {}", turn.content.trim()));
    }

    lines.join("\n")
}

/// Ingest a structured session/transcript.
pub async fn ingest_session(
    config: &ReinConfig,
    session: &SessionIngest,
    agent_label: Option<&str>,
    is_subagent: bool,
) -> crate::types::ReinResult<(u32, u32, u32)> {
    let report = ingest_session_report(config, session, agent_label, is_subagent).await?;
    Ok((report.memory_count, report.concept_count, report.link_count))
}

pub async fn ingest_session_text_report(
    config: &ReinConfig,
    text: &str,
    agent_label: Option<&str>,
    is_subagent: bool,
) -> crate::types::ReinResult<IngestReport> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Ok(IngestReport::default());
    }

    let session = SessionIngest {
        schema_version: 1,
        artifact_kind: "session".to_string(),
        session_id: None,
        title: None,
        started_at: None,
        ended_at: None,
        summary: None,
        source_agent: agent_label.map(|s| s.to_string()),
        source_label: Some("explicit-ingest".to_string()),
        compact_summary: None,
        tool_outputs: vec![],
        turns: vec![SessionTurn {
            role: "session".to_string(),
            content: trimmed.to_string(),
        }],
    };
    ingest_session_report(config, &session, agent_label, is_subagent).await
}

pub async fn ingest_session_report(
    config: &ReinConfig,
    session: &SessionIngest,
    agent_label: Option<&str>,
    is_subagent: bool,
) -> crate::types::ReinResult<IngestReport> {
    let text = render_session_ingest(session);
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Ok(IngestReport::default());
    }

    // Filter secrets from transcript before persisting artifact
    let scrubbed = crate::extract::hooks::parsing::redact_secrets(trimmed);

    let store = config.open_store()?;
    let artifact = SessionArtifact {
        id: String::new(),
        schema_version: session.schema_version,
        artifact_kind: session.artifact_kind.clone(),
        session_id: session.session_id.clone(),
        title: session.title.clone(),
        summary: session.summary.clone(),
        source_agent: session
            .source_agent
            .clone()
            .or_else(|| agent_label.map(|s| s.to_string())),
        source_label: session.source_label.clone(),
        is_subagent,
        started_at: session.started_at,
        ended_at: session.ended_at,
        turn_count: session.turns.len() as u32,
        transcript_text: scrubbed.clone(),
        transcript_json: None, // raw JSON may contain secrets; omit from artifact
        episode_id: None,
        created_at: chrono::Utc::now(),
    };
    let artifact_id = store.store_session_artifact(artifact)?;

    let result = extract::llm::extract_full_with_fallback(config, &scrubbed).await;
    let mut report = ingest_extraction_report(config, session, result, agent_label, is_subagent)?;
    report.artifact_id = Some(artifact_id.clone());
    if let Some(ref episode_id) = report.episode_id {
        let _ = store.link_session_artifact_episode(&artifact_id, episode_id);
    }
    Ok(report)
}

pub fn ingest_extraction_report(
    config: &ReinConfig,
    session: &SessionIngest,
    mut result: extract::llm::ExtractionResult,
    agent_label: Option<&str>,
    is_subagent: bool,
) -> crate::types::ReinResult<IngestReport> {
    let max_items = config.hooks.max_items_per_session;
    result.memories.truncate(max_items);

    if result.memories.is_empty() && result.concepts.is_empty() && result.episode.is_none() {
        return Ok(IngestReport {
            session_id: session.session_id.clone(),
            turn_count: session.turns.len() as u32,
            ..Default::default()
        });
    }

    let store = config.open_store()?;
    let episode_for_ws = result.episode.clone();
    let memories_for_ws = result.memories.clone();
    let concepts_for_ws = result.concepts.clone();
    let ingest_agent = agent_label
        .or(session.source_agent.as_deref())
        .unwrap_or("manual-ingest");
    let memory_stats = crate::extract::hooks::persist::store_extracted_report(
        &store,
        config,
        result.memories,
        ingest_agent,
        is_subagent,
    );
    let memory_count = memory_stats.stored_count;
    let stored_memory_ids = memory_stats.stored_ids.clone();
    let _ = crate::extract::hooks::working_set::update_working_set(
        config,
        &memories_for_ws,
        &concepts_for_ws,
        episode_for_ws.as_ref(),
        ingest_agent,
        is_subagent,
    );
    let _ = crate::extract::hooks::working_set::update_always_on_index(
        config,
        &memories_for_ws,
        &concepts_for_ws,
        episode_for_ws.as_ref(),
        ingest_agent,
        is_subagent,
    );
    let kg_report = store
        .store_knowledge_units_with_sources(&result.concepts, &result.links, &stored_memory_ids)
        .unwrap_or_default();

    let session_concept_ids: Vec<String> = result
        .concepts
        .iter()
        .filter_map(|c| {
            store
                .get_concept(&c.memoir, &c.name)
                .ok()
                .flatten()
                .map(|con| con.id)
        })
        .collect();

    let primary_topics = derive_primary_topics(&memories_for_ws, &concepts_for_ws);
    let temporal_keywords = derive_temporal_keywords(&memories_for_ws, session);
    let important_paths = derive_important_paths(&memories_for_ws, session);
    let mut episode_id = None;

    if let Some(ref ep) = result.episode {
        let mut tags = primary_topics.clone();
        tags.push(if is_subagent {
            "subagent".to_string()
        } else {
            "main-agent".to_string()
        });
        if session.compact_summary.is_some() {
            tags.push("compact".to_string());
        }
        tags.sort();
        tags.dedup();

        let episode = Episode {
            id: String::new(),
            title: ep.title.clone(),
            outcome: ep.outcome.clone(),
            decisions: ep.decisions.clone(),
            primary_topics: primary_topics.clone(),
            tags,
            involved_agents: vec![session
                .source_agent
                .clone()
                .unwrap_or_else(|| ingest_agent.to_string())],
            important_paths,
            temporal_keywords,
            source_session_id: session.session_id.clone(),
            concept_ids: session_concept_ids.clone(),
            memory_ids: stored_memory_ids.clone(),
            created_at: session
                .ended_at
                .or(session.started_at)
                .unwrap_or_else(chrono::Utc::now),
        };
        let episode_result = (|| -> crate::types::ReinResult<String> {
            store.conn().execute_batch("SAVEPOINT episode_linking")?;
            let created_episode_id = store.create_episode(episode)?;
            let cutoff = (chrono::Utc::now() - chrono::Duration::hours(24)).to_rfc3339();
            for cid in &session_concept_ids {
                store.conn().execute(
                    "UPDATE concepts SET last_episode_id = ?1 WHERE id = ?2",
                    rusqlite::params![created_episode_id, cid],
                )?;
                store.conn().execute(
                    "UPDATE concept_revisions SET episode_id = ?1 WHERE concept_id = ?2 AND episode_id IS NULL AND created_at >= ?3",
                    rusqlite::params![created_episode_id, cid, cutoff],
                )?;
            }
            store.conn().execute_batch("RELEASE episode_linking")?;
            Ok(created_episode_id)
        })();
        match episode_result {
            Ok(created_episode_id) => {
                episode_id = Some(created_episode_id);
                if let Err(e) = crate::extract::hooks::buffer::store_episode_concept(&store, ep) {
                    tracing::warn!("failed to store episode concept: {e}");
                }
            }
            Err(e) => {
                let _ = store.conn().execute_batch("ROLLBACK TO episode_linking");
                let _ = store.conn().execute_batch("RELEASE episode_linking");
                tracing::warn!("failed to create/link episode: {e}");
            }
        }
    }

    Ok(IngestReport {
        queued: false,
        artifact_id: None,
        session_id: session.session_id.clone(),
        episode_id,
        memory_count,
        concept_count: (kg_report.concepts_added + kg_report.concepts_refined) as u32,
        link_count: kg_report.links_added as u32,
        turn_count: session.turns.len() as u32,
        filtered_count: memory_stats.filtered_count,
        secret_filtered_count: memory_stats.secret_filtered_count,
        created_count: memory_stats.created_count,
        merged_count: memory_stats.merged_count,
        superseded_count: memory_stats.superseded_count,
        stored_memory_ids,
        primary_topics,
    })
}

fn derive_primary_topics(
    memories: &[ExtractedMemory],
    concepts: &[crate::extract::llm::ExtractedConcept],
) -> Vec<String> {
    let mut topics: Vec<String> = memories.iter().map(|m| m.topic.clone()).collect();
    topics.extend(concepts.iter().map(|c| c.memoir.clone()));
    topics.sort();
    topics.dedup();
    topics.truncate(8);
    topics
}

fn derive_temporal_keywords(memories: &[ExtractedMemory], session: &SessionIngest) -> Vec<String> {
    let mut kws: Vec<String> = memories
        .iter()
        .flat_map(|m| m.keywords.iter().cloned())
        .filter(|kw| kw.starts_with("date:"))
        .collect();
    if let Some(started_at) = session.started_at {
        kws.push(format!("date:{}", started_at.format("%Y-%m-%d")));
    }
    if let Some(ended_at) = session.ended_at {
        kws.push(format!("date:{}", ended_at.format("%Y-%m-%d")));
    }
    kws.sort();
    kws.dedup();
    kws.truncate(8);
    kws
}

fn derive_important_paths(memories: &[ExtractedMemory], session: &SessionIngest) -> Vec<String> {
    let mut paths = Vec::new();
    let mut collect_from = |text: &str| {
        if paths.len() >= 200 {
            return;
        } // bound intermediate allocation
        for token in text.split_whitespace() {
            let trimmed = token.trim_matches(|c: char| ",:;()[]{}'\"".contains(c));
            let looks_like_path = trimmed.contains('/')
                || [
                    ".rs", ".ts", ".tsx", ".js", ".jsx", ".py", ".md", ".toml", ".json",
                ]
                .iter()
                .any(|suffix| trimmed.ends_with(suffix));
            if looks_like_path && trimmed.len() > 3 {
                paths.push(trimmed.to_string());
            }
        }
    };

    for memory in memories {
        collect_from(&memory.summary);
        collect_from(&memory.content);
    }
    if let Some(summary) = session.summary.as_deref() {
        collect_from(summary);
    }
    if let Some(compact_summary) = session.compact_summary.as_deref() {
        collect_from(compact_summary);
    }
    for output in &session.tool_outputs {
        collect_from(output);
    }

    paths.sort();
    paths.dedup();
    paths.truncate(10);
    paths
}

/// Sync wrapper for transcript ingestion.
/// Used by CLI and MCP tool handlers that need a blocking entrypoint.
pub fn ingest_session_text_sync(
    config: &ReinConfig,
    text: &str,
    agent_label: Option<&str>,
    is_subagent: bool,
) -> crate::types::ReinResult<(u32, u32, u32)> {
    let report = ingest_session_text_sync_report(config, text, agent_label, is_subagent)?;
    Ok((report.memory_count, report.concept_count, report.link_count))
}

pub fn ingest_session_text_sync_report(
    config: &ReinConfig,
    text: &str,
    agent_label: Option<&str>,
    is_subagent: bool,
) -> crate::types::ReinResult<IngestReport> {
    let cfg = config.clone();
    let text = text.to_string();
    let agent_label = agent_label.map(|s| s.to_string());

    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        tokio::task::block_in_place(|| {
            handle.block_on(ingest_session_text_report(
                &cfg,
                &text,
                agent_label.as_deref(),
                is_subagent,
            ))
        })
    } else {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| ReinError::Config(format!("failed to build tokio runtime: {e}")))?;
        rt.block_on(ingest_session_text_report(
            &cfg,
            &text,
            agent_label.as_deref(),
            is_subagent,
        ))
    }
}

/// Sync wrapper for structured session ingestion.
pub fn ingest_session_sync(
    config: &ReinConfig,
    session: &SessionIngest,
    agent_label: Option<&str>,
    is_subagent: bool,
) -> crate::types::ReinResult<(u32, u32, u32)> {
    let report = ingest_session_sync_report(config, session, agent_label, is_subagent)?;
    Ok((report.memory_count, report.concept_count, report.link_count))
}

pub fn ingest_session_sync_report(
    config: &ReinConfig,
    session: &SessionIngest,
    agent_label: Option<&str>,
    is_subagent: bool,
) -> crate::types::ReinResult<IngestReport> {
    let cfg = config.clone();
    let session = session.clone();
    let agent_label = agent_label.map(|s| s.to_string());

    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        tokio::task::block_in_place(|| {
            handle.block_on(ingest_session_report(
                &cfg,
                &session,
                agent_label.as_deref(),
                is_subagent,
            ))
        })
    } else {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| ReinError::Config(format!("failed to build tokio runtime: {e}")))?;
        rt.block_on(ingest_session_report(
            &cfg,
            &session,
            agent_label.as_deref(),
            is_subagent,
        ))
    }
}

pub fn queue_ingest_session_text(
    config: &ReinConfig,
    text: &str,
    agent_label: Option<&str>,
    is_subagent: bool,
) -> crate::types::ReinResult<IngestReport> {
    let session = SessionIngest {
        schema_version: 1,
        artifact_kind: "session".to_string(),
        session_id: None,
        title: None,
        started_at: None,
        ended_at: None,
        summary: None,
        source_agent: agent_label.map(|s| s.to_string()),
        source_label: Some("explicit-ingest".to_string()),
        compact_summary: None,
        tool_outputs: vec![],
        turns: vec![SessionTurn {
            role: "session".to_string(),
            content: text.to_string(),
        }],
    };
    queue_ingest_session(config, &session, agent_label, is_subagent)
}

pub fn queue_ingest_session(
    config: &ReinConfig,
    session: &SessionIngest,
    agent_label: Option<&str>,
    is_subagent: bool,
) -> crate::types::ReinResult<IngestReport> {
    let text = render_session_ingest(session);
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Ok(IngestReport::default());
    }
    // Filter secrets from transcript before persisting artifact
    let scrubbed = crate::extract::hooks::parsing::redact_secrets(trimmed);
    let store = config.open_store()?;
    let artifact = SessionArtifact {
        id: String::new(),
        schema_version: session.schema_version,
        artifact_kind: session.artifact_kind.clone(),
        session_id: session.session_id.clone(),
        title: session.title.clone(),
        summary: session.summary.clone(),
        source_agent: session
            .source_agent
            .clone()
            .or_else(|| agent_label.map(|s| s.to_string())),
        source_label: session.source_label.clone(),
        is_subagent,
        started_at: session.started_at,
        ended_at: session.ended_at,
        turn_count: session.turns.len() as u32,
        transcript_text: scrubbed.clone(),
        transcript_json: None, // raw JSON may contain secrets; omit from artifact
        episode_id: None,
        created_at: chrono::Utc::now(),
    };
    let artifact_id = store.store_session_artifact(artifact)?;
    let mut sanitized_session = session.clone();
    sanitized_session.summary = sanitized_session
        .summary
        .map(|s| crate::extract::hooks::parsing::redact_secrets(&s));
    sanitized_session.compact_summary = sanitized_session
        .compact_summary
        .map(|s| crate::extract::hooks::parsing::redact_secrets(&s));
    sanitized_session.tool_outputs = sanitized_session
        .tool_outputs
        .into_iter()
        .map(|s| crate::extract::hooks::parsing::redact_secrets(&s))
        .collect();
    sanitized_session.turns = sanitized_session
        .turns
        .into_iter()
        .map(|turn| crate::types::SessionTurn {
            role: turn.role,
            content: crate::extract::hooks::parsing::redact_secrets(&turn.content),
        })
        .collect();
    crate::extract::hooks::queue::queue_memory_job_with_session(
        config,
        crate::extract::hooks::queue::MemoryJobMode::Full,
        "ingest_session",
        session.source_label.as_deref().unwrap_or(if is_subagent {
            "source:subagent"
        } else {
            "source:main-agent"
        }),
        agent_label
            .or(session.source_agent.as_deref())
            .unwrap_or("manual-ingest")
            .to_string(),
        is_subagent,
        if is_subagent { 50 } else { 95 },
        None,
        scrubbed, // use scrubbed text, not raw trimmed
        Some(artifact_id.clone()),
        serde_json::to_string(&sanitized_session).ok(),
    )
    .map_err(|e| ReinError::Config(format!("{e}")))?;
    crate::extract::hooks::queue::spawn_memory_worker(config);

    Ok(IngestReport {
        queued: true,
        artifact_id: Some(artifact_id),
        session_id: session.session_id.clone(),
        turn_count: session.turns.len() as u32,
        ..Default::default()
    })
}

/// Normalize topic names for variant grouping.
/// Lowercases and collapses all non-alphanumeric runs into `-`.
pub fn normalize_topic_name(name: &str) -> String {
    let mut normalized = String::with_capacity(name.len());
    let mut prev_sep = false;

    for ch in name.trim().to_lowercase().chars() {
        if ch.is_alphanumeric() {
            normalized.push(ch);
            prev_sep = false;
        } else if !prev_sep && !normalized.is_empty() {
            normalized.push('-');
            prev_sep = true;
        }
    }

    normalized.trim_matches('-').to_string()
}

fn topic_display_score(topic: &str) -> i32 {
    let spaces = topic.chars().filter(|c| *c == ' ').count() as i32;
    let hyphens = topic.chars().filter(|c| *c == '-').count() as i32;
    let underscores = topic.chars().filter(|c| *c == '_').count() as i32;
    let uppercase = topic.chars().filter(|c| c.is_uppercase()).count() as i32;
    spaces * 3 + uppercase - hyphens - (underscores * 2)
}

fn choose_canonical_topic(store: &SqliteStore, topics: &[String]) -> ReinResult<String> {
    let mut best_topic = topics
        .first()
        .cloned()
        .ok_or_else(|| ReinError::Config("empty topic group".to_string()))?;
    let mut best_count = 0usize;
    let mut best_score = i32::MIN;

    for topic in topics {
        let count = store.get_by_topic(topic)?.len();
        let score = topic_display_score(topic);
        let better = count > best_count
            || (count == best_count && score > best_score)
            || (count == best_count && score == best_score && topic.len() < best_topic.len())
            || (count == best_count
                && score == best_score
                && topic.len() == best_topic.len()
                && topic.to_lowercase() < best_topic.to_lowercase());
        if better {
            best_topic = topic.clone();
            best_count = count;
            best_score = score;
        }
    }

    Ok(best_topic)
}

/// Resolve user-facing topic selectors into concrete consolidation groups.
pub fn resolve_topic_groups(
    store: &SqliteStore,
    topic: Option<&str>,
    topics: &[String],
    pattern: Option<&str>,
    all: bool,
    merge_variants: bool,
) -> ReinResult<Vec<TopicGroup>> {
    let stored_topics = store.list_topics()?;

    let mut selected = if let Some(topic) = topic {
        vec![topic.to_string()]
    } else if !topics.is_empty() {
        topics.to_vec()
    } else if let Some(pattern) = pattern {
        let glob = glob::Pattern::new(pattern)
            .map_err(|e| ReinError::Config(format!("invalid pattern '{pattern}': {e}")))?;
        let opts = glob::MatchOptions {
            case_sensitive: false,
            require_literal_separator: false,
            require_literal_leading_dot: false,
        };
        stored_topics
            .iter()
            .filter(|candidate| glob.matches_with(candidate, opts))
            .cloned()
            .collect::<Vec<_>>()
    } else if all || merge_variants {
        stored_topics.clone()
    } else {
        return Err(ReinError::Config(
            "select a topic, --topics, --pattern, or --all".to_string(),
        ));
    };

    let mut seen = std::collections::HashSet::new();
    selected.retain(|value| seen.insert(value.clone()));

    if selected.is_empty() {
        return Ok(vec![]);
    }

    if !merge_variants {
        return Ok(selected
            .into_iter()
            .map(|selected_topic| TopicGroup {
                canonical_topic: selected_topic.clone(),
                topics: vec![selected_topic],
            })
            .collect());
    }

    let mut grouped_topics: Vec<Vec<String>> = Vec::new();
    let mut by_key: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

    for selected_topic in selected {
        let key = normalize_topic_name(&selected_topic);
        if let Some(index) = by_key.get(&key).copied() {
            grouped_topics[index].push(selected_topic);
        } else {
            by_key.insert(key, grouped_topics.len());
            grouped_topics.push(vec![selected_topic]);
        }
    }

    let mut groups = Vec::with_capacity(grouped_topics.len());
    for topics in grouped_topics {
        let canonical_topic = choose_canonical_topic(store, &topics)?;
        groups.push(TopicGroup {
            canonical_topic,
            topics,
        });
    }
    Ok(groups)
}

pub(crate) fn stronger_tier(left: MemoryTier, right: MemoryTier) -> MemoryTier {
    match (left, right) {
        (MemoryTier::Hot, _) | (_, MemoryTier::Hot) => MemoryTier::Hot,
        (MemoryTier::Warm, _) | (_, MemoryTier::Warm) => MemoryTier::Warm,
        _ => MemoryTier::Cold,
    }
}

/// Run GC: apply decay + prune weak memories + prune low-quality concepts.
/// In dry-run mode, wraps operations in a savepoint to preview without committing.
/// Returns (decayed_count, memory_pruned_count, concept_pruned_count).
pub fn run_gc(store: &SqliteStore, threshold: f64, dry_run: bool) -> ReinResult<(u64, u64, u64)> {
    if dry_run {
        // Preview mode: DELETE within savepoint so concept evaluation sees the
        // same DB state as a real GC. Side indexes (Tantivy/HNSW) are NOT touched.
        // ROLLBACK undoes the SQLite DELETE at the end.
        store
            .conn()
            .execute_batch("SAVEPOINT gc_preview")
            .map_err(crate::types::ReinError::Database)?;

        let decayed = store.apply_decay()?;
        // SQL DELETE only (no Tantivy/HNSW removal) — savepoint will rollback
        let mem_pruned = store.prune_memories_sql_only(threshold)?;
        let concept_pruned = store.prune_low_quality_concepts().unwrap_or(0);

        store
            .conn()
            .execute_batch("ROLLBACK TO gc_preview")
            .map_err(crate::types::ReinError::Database)?;
        store
            .conn()
            .execute_batch("RELEASE gc_preview")
            .map_err(crate::types::ReinError::Database)?;

        Ok((decayed, mem_pruned, concept_pruned))
    } else {
        let decayed = store.apply_decay()?;
        let mem_pruned = store.prune_memories_only(threshold, false)?;
        let concept_pruned = store.prune_low_quality_concepts().unwrap_or(0);
        if concept_pruned > 0 {
            tracing::info!("pruned {concept_pruned} low-quality concepts");
        }
        Ok((decayed, mem_pruned, concept_pruned))
    }
}

/// Run GC with adaptive engine pipeline. Combines standard GC + adaptive learning.
pub fn run_gc_adaptive(
    store: &SqliteStore,
    config: &ReinConfig,
    threshold: f64,
    dry_run: bool,
) -> ReinResult<(u64, u64, u64)> {
    let result = run_gc(store, threshold, dry_run)?;
    if !dry_run {
        run_adaptive_pipeline(store, config);
    }
    Ok(result)
}

/// Return adaptive engine status as a JSON value for inspection.
/// Queries AdaptiveState, reranker weights, event counts, survival curves.
pub fn adaptive_status(store: &SqliteStore) -> serde_json::Value {
    let conn = store.conn();

    // Learned alphas from AdaptiveState
    let state = crate::store::adaptive::AdaptiveState::restore_snapshot(conn).unwrap_or_default();

    let learned_alphas: serde_json::Value = state
        .learned_alpha
        .iter()
        .map(|(k, entry)| {
            (
                k.clone(),
                serde_json::json!({
                    "value": entry.value,
                    "sample_count": entry.sample_count,
                    "last_updated": entry.last_updated,
                }),
            )
        })
        .collect::<serde_json::Map<String, serde_json::Value>>()
        .into();

    // Reranker weights (all 17)
    let weights = crate::search::rerank::load_weights(conn);
    let reranker_weights = serde_json::to_value(&weights).unwrap_or_default();

    // Cluster info
    let unique_clusters: std::collections::HashSet<u32> =
        state.memory_clusters.values().copied().collect();
    let cluster_info = serde_json::json!({
        "cluster_version": state.cluster_version,
        "unique_clusters": unique_clusters.len(),
        "assigned_memories": state.memory_clusters.len(),
    });

    // Tier boundaries
    let tier_boundaries = serde_json::json!({
        "hot_threshold": state.hot_threshold,
        "cold_threshold": state.cold_threshold,
    });

    // Event counts by type
    let event_counts: serde_json::Value = conn
        .prepare("SELECT event_type, COUNT(*) FROM feedback_events GROUP BY event_type")
        .map(|mut stmt| {
            let rows: Vec<(String, i64)> = stmt
                .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
                .ok()
                .map(|r| r.filter_map(|x| x.ok()).collect())
                .unwrap_or_default();
            rows
        })
        .map(|rows| {
            let map: serde_json::Map<String, serde_json::Value> = rows
                .into_iter()
                .map(|(k, v)| (k, serde_json::Value::from(v)))
                .collect();
            serde_json::Value::Object(map)
        })
        .unwrap_or_else(|_| serde_json::json!({}));

    // Survival curves summary
    let survival_curves: Vec<serde_json::Value> = conn
        .prepare("SELECT key, value FROM metadata WHERE key LIKE 'survival_curve:%'")
        .map(|mut stmt| {
            let rows: Vec<(String, String)> = stmt
                .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
                .ok()
                .map(|r| r.filter_map(|x| x.ok()).collect())
                .unwrap_or_default();
            rows
        })
        .map(|rows| {
            rows.into_iter()
                .filter_map(|(key, value)| {
                    let cluster_id = key.strip_prefix("survival_curve:")?;
                    let curve: serde_json::Value = serde_json::from_str(&value).ok()?;
                    let median = curve.get("median_survival").cloned();
                    let steps = curve.get("steps").cloned();
                    Some(serde_json::json!({
                        "cluster_id": cluster_id,
                        "median_survival": median,
                        "steps": steps,
                    }))
                })
                .collect()
        })
        .unwrap_or_default();

    // Dedup thresholds
    let dedup_thresholds = serde_json::json!({
        "per_cluster": state.dedup_thresholds,
        "global": state.global_dedup_threshold,
    });

    let recent_avg: f64 = conn
        .query_row(
            "SELECT COALESCE(AVG(strength), 0.5) FROM (SELECT strength FROM memories ORDER BY created_at DESC LIMIT 100)",
            [],
            |r| r.get(0),
        )
        .unwrap_or(0.5);
    let global_admission = if recent_avg < 0.4 {
        (0.2_f64 * 1.1_f64).min(0.60)
    } else if recent_avg > 0.7 {
        (0.2_f64 * 0.9_f64).max(0.15)
    } else {
        0.2_f64
    };
    let survival_lookup: std::collections::HashMap<u32, crate::search::survival::SurvivalCurve> =
        survival_curves
            .iter()
            .filter_map(|curve| {
                let cluster_id = curve.get("cluster_id")?.as_str()?.parse::<u32>().ok()?;
                let steps = curve.get("steps")?.clone();
                let median_survival = curve.get("median_survival").and_then(|v| v.as_f64());
                Some((
                    cluster_id,
                    crate::search::survival::SurvivalCurve {
                        steps: serde_json::from_value(steps).ok()?,
                        event_count: 0,
                        total_count: 0,
                        median_survival,
                    },
                ))
            })
            .collect();
    let cluster_profiles: Vec<serde_json::Value> = unique_clusters
        .iter()
        .filter_map(|cid| {
            let stats: Option<(u32, f64)> = conn
                .query_row(
                    "SELECT COUNT(*), AVG(strength) FROM memories
                     WHERE cluster_id = ?1 AND superseded_by IS NULL AND status IN ('active', 'updated')",
                    rusqlite::params![cid],
                    |row| Ok((row.get(0)?, row.get::<_, Option<f64>>(1)?.unwrap_or(0.0))),
                )
                .ok();
            let (memory_count, avg_strength) = stats?;
            let admission_threshold = if avg_strength > 0.0 && recent_avg > 0.0 {
                let cluster_threshold = (global_admission * (recent_avg / avg_strength)).clamp(0.15, 0.60);
                let blend = (memory_count as f64 / 8.0).clamp(0.0, 1.0);
                (global_admission * (1.0 - blend) + cluster_threshold * blend).clamp(0.15, 0.60)
            } else {
                global_admission
            };
            let promotion_threshold = survival_lookup
                .get(cid)
                .map(crate::search::survival::promotion_access_threshold)
                .unwrap_or(5);
            let median_survival = survival_lookup.get(cid).and_then(|curve| curve.median_survival);
            Some(serde_json::json!({
                "cluster_id": cid,
                "memory_count": memory_count,
                "avg_strength": avg_strength,
                "dedup_threshold": state.get_dedup_threshold(Some(*cid)),
                "admission_threshold": admission_threshold,
                "promotion_threshold": promotion_threshold,
                "median_survival": median_survival,
            }))
        })
        .collect();

    serde_json::json!({
        "learned_alphas": learned_alphas,
        "reranker_weights": reranker_weights,
        "cluster_info": cluster_info,
        "tier_boundaries": tier_boundaries,
        "event_counts": event_counts,
        "survival_curves": survival_curves,
        "dedup_thresholds": dedup_thresholds,
        "cluster_profiles": cluster_profiles,
    })
}

/// Upgrade report returned by run_upgrade.
#[derive(Debug, Default)]
pub struct UpgradeReport {
    pub topics_processed: usize,
    pub enriched: usize,
    pub deprecated: usize,
    pub concepts: usize,
    pub links: usize,
    pub memoirs: usize,
    /// Per-topic dry-run preview messages
    pub preview_lines: Vec<String>,
}

/// Run memory upgrade: LLM enrichment + knowledge graph extraction, or local rules fallback.
/// Used by both `rein upgrade` CLI and potential MCP tool.
pub async fn run_upgrade(
    store: &SqliteStore,
    config: &ReinConfig,
    topic_filter: Option<&str>,
    dry_run: bool,
) -> ReinResult<UpgradeReport> {
    let has_llm = extract::llm::create_extractor(config).is_some();
    let mut report = UpgradeReport::default();

    let topics = if let Some(t) = topic_filter {
        vec![t.to_string()]
    } else {
        store.list_topics()?
    };

    for topic_name in &topics {
        let memories = store.get_by_topic(topic_name)?;
        if memories.is_empty() {
            continue;
        }
        report.topics_processed += 1;

        let combined: String = memories
            .iter()
            .map(|m| format!("[{}] {}\n{}", m.topic, m.summary, m.content))
            .collect::<Vec<_>>()
            .join("\n---\n");

        let result = extract::llm::extract_full_with_fallback(config, &combined).await;

        if has_llm {
            if dry_run {
                let enrichable = result.memories.len().min(memories.len());
                report.preview_lines.push(format!(
                    "  topic '{}': would enrich {} memories, create {} concepts, {} links",
                    topic_name,
                    enrichable,
                    result.concepts.len(),
                    result.links.len()
                ));
                for c in &result.concepts {
                    report.preview_lines.push(format!(
                        "    concept: [{}] {} ({})",
                        c.memoir, c.name, c.concept_type
                    ));
                }
                for l in &result.links {
                    report
                        .preview_lines
                        .push(format!("    link: {} --{}-> {}", l.from, l.relation, l.to));
                }
                report.concepts += result.concepts.len();
                report.links += result.links.len();
                report.enriched += enrichable;
            } else {
                // LLM quality audit + enrichment
                for new_mem in &result.memories {
                    let best_match = memories.iter().max_by(|a, b| {
                        let sim_a = extract::similarity(&a.content, &new_mem.content);
                        let sim_b = extract::similarity(&b.content, &new_mem.content);
                        sim_a
                            .partial_cmp(&sim_b)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    if let Some(old) = best_match {
                        let sim = extract::similarity(&old.content, &new_mem.content);
                        if sim > 0.3 {
                            if new_mem.quality_confidence < 0.2 {
                                let _ = store.conn().execute(
                                    "UPDATE memories SET status = 'deprecated' WHERE id = ?1",
                                    rusqlite::params![old.id],
                                );
                                report.deprecated += 1;
                                continue;
                            }
                            let mut enriched = old.clone();
                            enriched.topic = new_mem.topic.clone();
                            enriched.summary = new_mem.summary.clone();
                            enriched.keywords = new_mem.keywords.clone();
                            if let Ok(imp) = new_mem.importance.parse::<Importance>() {
                                enriched.importance = imp;
                                enriched.layer = imp.auto_layer();
                                enriched.decay_lambda =
                                    config.decay.base_lambda * imp.decay_factor();
                            }
                            if store.update(&enriched).is_ok() {
                                report.enriched += 1;
                            }
                        }
                    }
                }

                let memory_ids: Vec<String> = memories.iter().map(|m| m.id.clone()).collect();
                if !result.concepts.is_empty() || !result.links.is_empty() {
                    match store.store_knowledge_units_with_sources(
                        &result.concepts,
                        &result.links,
                        &memory_ids,
                    ) {
                        Ok(r) => {
                            report.memoirs += r.memoirs_created;
                            report.concepts += r.concepts_added + r.concepts_refined;
                            report.links += r.links_added;
                        }
                        Err(e) => {
                            tracing::warn!("knowledge_units error for topic '{}': {e}", topic_name)
                        }
                    }
                }

                for mem in &memories {
                    let _ = store.auto_link(&mem.id, config.search.dedup_similarity as f32, 5);
                    let _ = store.activate_related_memories(&mem.content, 3);
                    let _ = store.activate_related_concepts(&mem.content);
                }
            }
        } else {
            // No-LLM path: local rule-based enrichment
            for old in &memories {
                if old.topic != "auto-extracted" {
                    continue;
                }
                let lower = old.content.to_lowercase();
                let new_topic = if ["architecture", "design", "component", "架构", "设计"]
                    .iter()
                    .any(|k| lower.contains(k))
                {
                    "architecture"
                } else if ["decided", "chose", "选型", "决策", "tradeoff"]
                    .iter()
                    .any(|k| lower.contains(k))
                {
                    "decision"
                } else if ["bug", "fix", "error", "crash", "修复", "解决"]
                    .iter()
                    .any(|k| lower.contains(k))
                {
                    "debug"
                } else if [
                    "deploy", "install", "config", "migrate", "部署", "安装", "迁移",
                ]
                .iter()
                .any(|k| lower.contains(k))
                {
                    "workflow"
                } else {
                    "general"
                };

                let keywords: Vec<String> = old
                    .content
                    .split_whitespace()
                    .filter(|w| w.len() > 3)
                    .map(|w| {
                        w.trim_matches(|c: char| !c.is_alphanumeric())
                            .to_lowercase()
                    })
                    .filter(|w| {
                        !w.is_empty()
                            && ![
                                "the", "this", "that", "with", "from", "have", "been", "into",
                                "will",
                            ]
                            .contains(&w.as_str())
                    })
                    .take(5)
                    .collect();

                if dry_run {
                    if new_topic != "auto-extracted" || !keywords.is_empty() {
                        report.preview_lines.push(format!(
                            "  → would reclassify '{}' → topic='{}', keywords={:?}",
                            old.summary.chars().take(40).collect::<String>(),
                            new_topic,
                            keywords
                        ));
                        report.enriched += 1;
                    }
                } else {
                    let mut enriched = old.clone();
                    enriched.topic = new_topic.to_string();
                    if !keywords.is_empty() {
                        enriched.keywords = keywords;
                    }
                    let score = extract::score_sentence(&old.content);
                    if score >= 4 {
                        enriched.importance = Importance::High;
                        enriched.layer = enriched.importance.auto_layer();
                        enriched.decay_lambda =
                            config.decay.base_lambda * enriched.importance.decay_factor();
                    }
                    if store.update(&enriched).is_ok() {
                        report.enriched += 1;
                    }
                }
            }
        }
    }

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ReinConfig;
    use crate::store::SqliteStore;
    use chrono::Utc;

    fn test_memory(topic: &str, summary: &str, content: &str) -> Memory {
        Memory {
            id: ulid::Ulid::new().to_string(),
            layer: MemoryLayer::LTM,
            topic: topic.to_string(),
            summary: summary.to_string(),
            content: content.to_string(),
            keywords: vec![],
            importance: Importance::High,
            source: Source::Manual,
            strength: 1.0,
            decay_lambda: 0.02,
            access_count: 0,
            superseded_by: None,
            canonical_id: None,
            support_count: 1,
            merge_count: 0,
            dedup_confidence: 1.0,
            source_diversity: 1.0,
            contradiction_score: 0.0,
            related_ids: vec![],
            concept_ids: vec![],
            status: MemoryStatus::default(),
            embedding: None,
            tier: MemoryTier::Warm,
            cluster_id: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_accessed: Utc::now(),
        }
    }

    #[test]
    fn test_extract_unique_lines_preserves_dates() {
        let source =
            "2026-03-15 direct login failed\n2026-03-17 used jump host\ngeneral note about SSH";
        let target = "general note about SSH and connection\n2026-03-22 containerTag mismatch";

        let unique = extract_unique_lines(source, target);
        // Should preserve date-anchored lines even if partially overlapping
        assert!(
            unique.contains("2026-03-15"),
            "should keep date 03-15: {unique}"
        );
        assert!(
            unique.contains("2026-03-17"),
            "should keep date 03-17: {unique}"
        );
    }

    #[test]
    fn test_extract_unique_lines_filters_duplicates() {
        let source = "line A\nline B\nline C";
        let target = "line A\nline C\nline D";

        let unique = extract_unique_lines(source, target);
        assert!(unique.contains("line B"), "should keep unique line B");
        assert!(
            !unique.contains("line A"),
            "should not keep duplicate line A"
        );
        assert!(
            !unique.contains("line C"),
            "should not keep duplicate line C"
        );
    }

    #[test]
    fn test_extract_unique_lines_empty_source() {
        assert!(extract_unique_lines("", "anything").is_empty());
    }

    #[test]
    fn test_extract_unique_lines_all_unique() {
        let unique = extract_unique_lines("alpha\nbeta", "gamma\ndelta");
        assert!(unique.contains("alpha"));
        assert!(unique.contains("beta"));
    }

    #[test]
    fn test_normalize_topic_name_groups_variants() {
        assert_eq!(
            normalize_topic_name("Docker Deployment"),
            normalize_topic_name("docker-deployment")
        );
        assert_eq!(
            normalize_topic_name("docker deployment"),
            normalize_topic_name("docker_deployment")
        );
        assert_eq!(
            normalize_topic_name("  RMCP 1.3.0 Compatibility "),
            "rmcp-1-3-0-compatibility"
        );
    }

    #[test]
    fn test_derive_auto_concepts_uses_topic_and_keywords() {
        let memory = test_memory(
            "Vector Index Rebuild",
            "Fix vector index rebuild OOM after reindex",
            "We need to rebuild the vector index after the model change and reindex the cache.",
        );
        let concepts = derive_auto_concepts(&memory);
        assert!(
            (1..=3).contains(&concepts.len()),
            "expected 1-3 concepts, got {}",
            concepts.len()
        );
        assert!(
            concepts.iter().all(|c| c.memoir == AUTO_CONCEPT_MEMOIR),
            "auto concepts should stay in the auto-discovered memoir"
        );
        assert!(
            concepts.iter().any(|c| c.name == "vector-index-rebuild"),
            "topic should yield a concept"
        );
        assert!(
            concepts.iter().any(|c| c.name.contains("reindex")),
            "keywords/content should yield a second concept"
        );
    }

    #[test]
    fn test_detect_conflict_marker_requires_signal_and_low_similarity() {
        let store = SqliteStore::in_memory().unwrap();
        store
            .store_knowledge_units_with_sources(
                &[crate::extract::llm::ExtractedConcept {
                    name: "append-only-log".to_string(),
                    definition: "An append-only log stores immutable events for replication."
                        .to_string(),
                    labels: vec!["storage".to_string()],
                    memoir: "architecture".to_string(),
                    concept_type: "fact".to_string(),
                    quality_confidence: 0.9,
                }],
                &[],
                &[],
            )
            .unwrap();

        let memory = test_memory(
            "Storage model",
            "We no longer use append-only logs and switched to a B-tree index.",
            "We no longer use append-only logs and switched to a B-tree index.",
        );

        let marker = detect_conflict_marker(&store, &memory).unwrap();
        let marker = marker.expect("expected a conflict marker");
        assert!(marker.score > 0.0);
        assert!(marker.evidence_keyword.starts_with("conflict-"));
        assert_eq!(marker.matched_concept, "append-only-log");
    }

    #[test]
    fn test_store_memory_adds_auto_concepts_and_conflict_markers() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = ReinConfig::default();
        config.database.path = dir.path().join("rein.sqlite").display().to_string();
        config.embedding.provider = "none".to_string();

        let store = config.open_store().unwrap();
        store
            .store_knowledge_units_with_sources(
                &[crate::extract::llm::ExtractedConcept {
                    name: "append-only-log".to_string(),
                    definition: "An append-only log stores immutable events for replication."
                        .to_string(),
                    labels: vec!["storage".to_string()],
                    memoir: "architecture".to_string(),
                    concept_type: "fact".to_string(),
                    quality_confidence: 0.9,
                }],
                &[],
                &[],
            )
            .unwrap();

        let memory = test_memory(
            "Storage Model",
            "Storage model update: we no longer use append-only logs and switched to a B-tree index.",
            "Storage model update: we no longer use append-only logs and switched to a B-tree index.",
        );
        let id = store_memory(&store, &config, memory).unwrap();
        let stored = store.get(&id).unwrap();

        assert!(
            stored.contradiction_score > 0.0,
            "conflict marker should bump contradiction_score"
        );
        assert!(
            stored.keywords.iter().any(|kw| kw == "conflict"),
            "conflict keyword should be added"
        );
        assert!(
            stored.keywords.iter().any(|kw| kw.starts_with("conflict-")),
            "conflict evidence keyword should be added"
        );
        assert!(
            !stored.concept_ids.is_empty(),
            "auto-discovered concepts should be linked back to the memory"
        );
        assert!(
            store
                .get_concept(AUTO_CONCEPT_MEMOIR, "storage-model")
                .unwrap()
                .is_some(),
            "topic should create an auto-discovered concept"
        );
    }

    #[test]
    fn test_resolve_topic_groups_merges_variants() {
        let store = SqliteStore::in_memory().unwrap();
        store
            .store(test_memory(
                "Docker Deployment",
                "primary",
                "deploy via compose",
            ))
            .unwrap();
        store
            .store(test_memory(
                "Docker Deployment",
                "primary 2",
                "pin image tag",
            ))
            .unwrap();
        store
            .store(test_memory(
                "docker-deployment",
                "secondary",
                "same topic variant",
            ))
            .unwrap();
        store
            .store(test_memory("CP2K MPI Failure", "cp2k", "exit code 7"))
            .unwrap();

        // Since store() normalizes topics at write time, "Docker Deployment" and
        // "docker-deployment" both become "docker-deployment". Variant merging still
        // works for any topics that slip through without normalization (e.g. direct SQL).
        let groups = resolve_topic_groups(&store, None, &[], None, true, true).unwrap();
        let docker_group = groups
            .iter()
            .find(|group| {
                group
                    .topics
                    .iter()
                    .any(|topic| topic == "docker-deployment")
            })
            .unwrap();

        // All docker topics normalized to "docker-deployment" at store time
        assert_eq!(docker_group.canonical_topic, "docker-deployment");
        assert_eq!(docker_group.topics.len(), 1);
    }

    #[test]
    fn test_run_dedup_merge_variants_scans_across_topic_variants() {
        let store = SqliteStore::in_memory().unwrap();
        let config = ReinConfig::default();
        // Since store() normalizes topics at write time, both topics become "docker-deployment".
        // Duplicates are now found even without merge_variants since they share the same topic.
        store
            .store(test_memory(
                "Docker Deployment",
                "compose setup",
                "Use docker compose for the local stack",
            ))
            .unwrap();
        store
            .store(test_memory(
                "docker-deployment",
                "compose setup duplicate",
                "Use docker compose for the local stack",
            ))
            .unwrap();

        let (without_variants, _) = run_dedup(&store, &config, 0.70, true, false).unwrap();
        let (with_variants, _) = run_dedup(&store, &config, 0.70, true, true).unwrap();

        // Both find the duplicate since topics are pre-normalized at store time
        assert_eq!(without_variants, 1);
        assert_eq!(with_variants, 1);
    }

    #[tokio::test]
    async fn test_run_consolidation_async_dry_run_reports_grouped_variants() {
        let store = SqliteStore::in_memory().unwrap();
        let config = ReinConfig::default();
        store
            .store(test_memory(
                "query expansion strategy",
                "strategy",
                "Use LLM-based synonym expansion",
            ))
            .unwrap();
        store
            .store(test_memory(
                "query-expansion-strategy",
                "strategy duplicate",
                "Prefer query rewrites with synonyms",
            ))
            .unwrap();

        let groups = resolve_topic_groups(&store, None, &[], None, true, true).unwrap();
        let report = run_consolidation_async(&store, &config, &groups, None, true)
            .await
            .unwrap();

        assert_eq!(report.groups_processed, 1);
        assert_eq!(report.memories_replaced, 2);
        assert_eq!(report.groups.len(), 1);
        assert_eq!(report.groups[0].memory_count, 2);
        assert!(report.groups[0].created_id.is_none());
    }
}
