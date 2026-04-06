//! Consolidation operations: topic grouping, memory consolidation (sync/async),
//! cleanup orchestration, and consolidation event emission.

use crate::config::ReinConfig;
use crate::extract::llm::ExtractedMemory;
use crate::store::SqliteStore;
use crate::types::*;

use super::adaptive::run_adaptive_pipeline;
use super::dedup::{emit_cleanup_event, record_deleted_memory_as_evidence, run_dedup_scoped};

const CANONICAL_SUMMARY_MAX_CHARS: usize = 240;

fn normalize_summary(summary: &str) -> String {
    let normalized = summary.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut out = String::new();
    for ch in normalized.chars().take(CANONICAL_SUMMARY_MAX_CHARS) {
        out.push(ch);
    }
    out
}

/// Build a consolidated Memory from a topic.
pub fn build_consolidated(
    config: &ReinConfig,
    topic: String,
    summary: String,
    related_ids: Vec<String>,
) -> Memory {
    let importance = Importance::High;
    Memory {
        id: ulid::Ulid::new().to_string(),
        layer: MemoryLayer::LTM,
        topic,
        summary: normalize_summary(&summary),
        content: summary,
        keywords: vec![],
        importance,
        source: Source::Manual,
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
        related_ids,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopicGroup {
    pub canonical_topic: String,
    pub topics: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct ConsolidationGroupReport {
    pub canonical_topic: String,
    pub source_topics: Vec<String>,
    pub memory_count: usize,
    pub created_id: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct ConsolidateReport {
    pub groups: Vec<ConsolidationGroupReport>,
    pub groups_processed: usize,
    pub memories_replaced: usize,
    pub dry_run: bool,
}

#[derive(Debug, Clone, Default)]
pub struct CleanupReport {
    pub consolidation: ConsolidateReport,
    pub duplicates_found: u32,
    pub duplicates_merged: u32,
    pub dry_run: bool,
}

fn render_summary_template(
    template: &str,
    canonical_topic: &str,
    source_topics: &[String],
    memory_count: usize,
) -> String {
    template
        .replace("{topic}", canonical_topic)
        .replace("{count}", &memory_count.to_string())
        .replace("{topics}", &source_topics.join(", "))
}

fn synthesize_consolidation_summary(
    canonical_topic: &str,
    source_topics: &[String],
    memory_count: usize,
) -> String {
    if source_topics.len() > 1 {
        format!(
            "{canonical_topic}: merged {memory_count} memories from {} topic variants",
            source_topics.len()
        )
    } else {
        format!("{canonical_topic}: consolidated {memory_count} memories")
    }
}

fn collect_unique_detail_lines(memories: &[&Memory], max_lines: usize, max_chars: usize) -> String {
    let mut seen = std::collections::HashSet::new();
    let mut lines = Vec::new();
    let mut total_chars = 0usize;

    for memory in memories {
        for line in memory.content.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            let normalized = trimmed.to_lowercase();
            if !seen.insert(normalized) {
                continue;
            }

            let bullet = format!("- {trimmed}");
            let bullet_chars = bullet.chars().count() + 1;
            if lines.len() >= max_lines || total_chars + bullet_chars > max_chars {
                lines.push(format!(
                    "- ... truncated after {} unique detail lines",
                    lines.len()
                ));
                return lines.join("\n");
            }
            total_chars += bullet_chars;
            lines.push(bullet);
        }
    }

    lines.join("\n")
}

fn synthesize_consolidation_content(
    canonical_topic: &str,
    source_topics: &[String],
    memories: &[&Memory],
) -> String {
    let mut parts = Vec::new();
    parts.push(format!(
        "Consolidated {} memories into topic '{canonical_topic}'.",
        memories.len()
    ));
    if source_topics.len() > 1 {
        parts.push(format!("Source topics: {}", source_topics.join(", ")));
    }

    parts.push(String::new());
    parts.push("Summaries:".to_string());
    for memory in memories.iter().take(20) {
        parts.push(format!("- [{}] {}", memory.topic, memory.summary));
    }
    if memories.len() > 20 {
        parts.push(format!("- ... {} more memories", memories.len() - 20));
    }

    let details = collect_unique_detail_lines(memories, 40, 8000);
    if !details.is_empty() {
        parts.push(String::new());
        parts.push("Details:".to_string());
        parts.push(details);
    }

    parts.join("\n")
}

fn dominant_tier(memories: &[&Memory]) -> MemoryTier {
    if memories.iter().any(|memory| memory.tier == MemoryTier::Hot) {
        MemoryTier::Hot
    } else if memories
        .iter()
        .any(|memory| memory.tier == MemoryTier::Warm)
    {
        MemoryTier::Warm
    } else {
        MemoryTier::Cold
    }
}

fn is_current_consolidation_memory(memory: &Memory) -> bool {
    memory.superseded_by.is_none()
        && matches!(
            memory.status,
            MemoryStatus::Active | MemoryStatus::Updated
        )
}

pub fn build_consolidated_from_memories(
    config: &ReinConfig,
    canonical_topic: String,
    source_topics: &[String],
    memories: &[Memory],
    summary_template: Option<&str>,
) -> Memory {
    let mut ordered: Vec<&Memory> = memories.iter().collect();
    ordered.sort_by(|a, b| b.created_at.cmp(&a.created_at));

    let memory_count = ordered.len();
    let rendered = summary_template.map(|template| {
        render_summary_template(template, &canonical_topic, source_topics, memory_count)
    });
    let summary = rendered.clone().unwrap_or_else(|| {
        synthesize_consolidation_summary(&canonical_topic, source_topics, memory_count)
    });
    let content = rendered.unwrap_or_else(|| {
        synthesize_consolidation_content(&canonical_topic, source_topics, &ordered)
    });

    let mut keyword_seen = std::collections::HashSet::new();
    let mut keywords = Vec::new();
    for memory in &ordered {
        for keyword in &memory.keywords {
            if keyword_seen.insert(keyword.to_lowercase()) {
                keywords.push(keyword.clone());
            }
        }
    }
    keywords.truncate(24);

    let importance = ordered
        .iter()
        .map(|memory| memory.importance)
        .max()
        .unwrap_or(Importance::High)
        .max(Importance::High);

    let related_ids = ordered.iter().map(|memory| memory.id.clone()).collect();
    let total_access_count: u32 = ordered.iter().map(|memory| memory.access_count).sum();
    let avg_strength =
        ordered.iter().map(|memory| memory.strength).sum::<f64>() / memory_count.max(1) as f64;
    let reinforced_strength =
        (avg_strength + 0.05 * (memory_count.saturating_sub(1) as f64)).min(1.0);
    let decay_lambda = ordered.iter().map(|memory| memory.decay_lambda).fold(
        config.decay.base_lambda * importance.decay_factor(),
        f64::min,
    );
    let last_accessed = ordered
        .iter()
        .map(|memory| memory.last_accessed)
        .max()
        .unwrap_or_else(chrono::Utc::now);

    Memory {
        id: ulid::Ulid::new().to_string(),
        layer: importance.auto_layer(),
        topic: canonical_topic,
        summary: normalize_summary(&summary),
        content,
        keywords,
        importance,
        source: Source::Manual,
        strength: reinforced_strength,
        decay_lambda,
        access_count: total_access_count,
        superseded_by: None,
        canonical_id: None,
        support_count: memory_count as u32,
        merge_count: memory_count.saturating_sub(1) as u32,
        dedup_confidence: 0.9,
        source_diversity: source_topics.len().max(1) as f32,
        contradiction_score: 0.0,
        related_ids,
        concept_ids: vec![],
        status: MemoryStatus::default(),
        embedding: None,
        tier: dominant_tier(&ordered),
        cluster_id: None,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        last_accessed,
    }
}

fn build_consolidated_from_extracted(
    config: &ReinConfig,
    canonical_topic: String,
    source_topics: &[String],
    memories: &[Memory],
    extracted: ExtractedMemory,
) -> Memory {
    let mut consolidated = build_consolidated_from_memories(
        config,
        canonical_topic.clone(),
        source_topics,
        memories,
        None,
    );

    let importance = extracted
        .importance
        .parse::<Importance>()
        .unwrap_or(Importance::High)
        .max(Importance::High);

    consolidated.topic = canonical_topic;
    consolidated.summary = normalize_summary(&extracted.summary);
    consolidated.content = extracted.content;
    if !extracted.keywords.is_empty() {
        consolidated.keywords = extracted.keywords;
    }
    consolidated.importance = importance;
    consolidated.layer = importance.auto_layer();
    consolidated.decay_lambda = config.decay.base_lambda * importance.decay_factor();
    consolidated
}

/// Build a consolidated memory, preferring LLM synthesis when no explicit summary was provided.
pub async fn build_consolidated_from_memories_async(
    config: &ReinConfig,
    canonical_topic: String,
    source_topics: &[String],
    memories: &[Memory],
    summary_template: Option<&str>,
) -> Memory {
    if summary_template.is_some() {
        return build_consolidated_from_memories(
            config,
            canonical_topic,
            source_topics,
            memories,
            summary_template,
        );
    }

    match crate::extract::llm::summarize_topic_group(
        config,
        &canonical_topic,
        source_topics,
        memories,
    )
    .await
    {
        Ok(Some(extracted)) => build_consolidated_from_extracted(
            config,
            canonical_topic,
            source_topics,
            memories,
            extracted,
        ),
        Ok(None) => {
            build_consolidated_from_memories(config, canonical_topic, source_topics, memories, None)
        }
        Err(error) => {
            tracing::warn!(
                "llm consolidation failed for topic '{}': {}",
                canonical_topic,
                error
            );
            build_consolidated_from_memories(config, canonical_topic, source_topics, memories, None)
        }
    }
}

/// Execute one or more consolidations selected by topic/pattern, optionally grouping
/// normalized topic variants together.
pub fn run_consolidation(
    store: &SqliteStore,
    config: &ReinConfig,
    groups: &[TopicGroup],
    summary_template: Option<&str>,
    dry_run: bool,
) -> ReinResult<ConsolidateReport> {
    let mut report = ConsolidateReport {
        dry_run,
        ..Default::default()
    };
    let mut changed = false;

    for group in groups {
        let memories = load_group_memories(store, group)?;

        if memories.is_empty() {
            report.groups.push(ConsolidationGroupReport {
                canonical_topic: group.canonical_topic.clone(),
                source_topics: group.topics.clone(),
                memory_count: 0,
                created_id: None,
            });
            continue;
        }

        report.memories_replaced += memories.len();
        report.groups_processed += 1;

        let created_id = if dry_run {
            None
        } else {
            let replacement = build_consolidated_from_memories(
                config,
                group.canonical_topic.clone(),
                &group.topics,
                &memories,
                summary_template,
            );
            let new_id = replacement.id.clone();
            // Use ID-based deletion to prevent TOCTOU data loss (new memories added
            // to the topic between load_group_memories and this commit are preserved).
            let memory_ids: Vec<String> = memories.iter().map(|m| m.id.clone()).collect();
            let old_memories = store.consolidate_by_ids_atomic(&memory_ids, replacement)?;
            emit_consolidation_events(store, group, &new_id, &old_memories);
            changed = true;
            Some(new_id)
        };

        report.groups.push(ConsolidationGroupReport {
            canonical_topic: group.canonical_topic.clone(),
            source_topics: group.topics.clone(),
            memory_count: memories.len(),
            created_id,
        });
    }

    if changed {
        run_adaptive_pipeline(store, config);
    }

    Ok(report)
}

/// Async batch consolidation: LLM synthesis for each group is generated in parallel,
/// then writes are committed sequentially to keep SQLite mutations deterministic.
pub async fn run_consolidation_async(
    store: &SqliteStore,
    config: &ReinConfig,
    groups: &[TopicGroup],
    summary_template: Option<&str>,
    dry_run: bool,
) -> ReinResult<ConsolidateReport> {
    let mut report = ConsolidateReport {
        dry_run,
        ..Default::default()
    };
    let mut non_empty = Vec::new();
    let mut changed = false;

    for group in groups {
        let memories = load_group_memories(store, group)?;
        if memories.is_empty() {
            report.groups.push(ConsolidationGroupReport {
                canonical_topic: group.canonical_topic.clone(),
                source_topics: group.topics.clone(),
                memory_count: 0,
                created_id: None,
            });
        } else {
            non_empty.push((group.clone(), memories));
        }
    }
    non_empty.sort_by(|(left_group, left_memories), (right_group, right_memories)| {
        right_memories
            .len()
            .cmp(&left_memories.len())
            .then_with(|| left_group.canonical_topic.cmp(&right_group.canonical_topic))
    });

    let summary_template_owned = summary_template.map(str::to_string);
    let config_owned = config.clone();
    let llm_batch_size = config.async_memory.batch_size.max(1);
    if non_empty.len() > llm_batch_size {
        tracing::info!(
            "consolidation: processing {} groups in batches of {llm_batch_size}",
            non_empty.len()
        );
    }
    for batch in non_empty.chunks(llm_batch_size) {
        let batch: Vec<_> = batch.to_vec();
        let prepared =
            futures_util::future::join_all(batch.into_iter().map(|(group, memories)| {
                let config = config_owned.clone();
                let summary_template = summary_template_owned.clone();
                async move {
                    let replacement = if dry_run {
                        None
                    } else {
                        Some(
                            build_consolidated_from_memories_async(
                                &config,
                                group.canonical_topic.clone(),
                                &group.topics,
                                &memories,
                                summary_template.as_deref(),
                            )
                            .await,
                        )
                    };
                    (group, memories, replacement)
                }
            }))
            .await;

        // Inter-batch delay: avoid overwhelming LLM API / proxy with rapid sequential batches
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        for (group, memories, replacement) in prepared {
            report.memories_replaced += memories.len();
            report.groups_processed += 1;

            let created_id = if let Some(replacement) = replacement {
                let new_id = replacement.id.clone();
                // Use ID-based deletion to prevent TOCTOU data loss
                let memory_ids: Vec<String> = memories.iter().map(|m| m.id.clone()).collect();
                let old_memories = store.consolidate_by_ids_atomic(&memory_ids, replacement)?;
                emit_consolidation_events(store, &group, &new_id, &old_memories);
                changed = true;
                Some(new_id)
            } else {
                None
            };

            report.groups.push(ConsolidationGroupReport {
                canonical_topic: group.canonical_topic.clone(),
                source_topics: group.topics.clone(),
                memory_count: memories.len(),
                created_id,
            });
        }
    }

    if changed {
        run_adaptive_pipeline(store, config);
    }

    Ok(report)
}

/// Sync wrapper for async consolidation so MCP handlers can reuse the same logic.
pub fn run_consolidation_sync(
    store: &SqliteStore,
    config: &ReinConfig,
    groups: &[TopicGroup],
    summary_template: Option<&str>,
    dry_run: bool,
) -> ReinResult<ConsolidateReport> {
    let cfg = config.clone();
    let groups = groups.to_vec();
    let summary = summary_template.map(|value| value.to_string());

    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        tokio::task::block_in_place(|| {
            handle.block_on(run_consolidation_async(
                store,
                &cfg,
                &groups,
                summary.as_deref(),
                dry_run,
            ))
        })
    } else {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| ReinError::Config(format!("failed to build tokio runtime: {e}")))?;
        rt.block_on(run_consolidation_async(
            store,
            &cfg,
            &groups,
            summary.as_deref(),
            dry_run,
        ))
    }
}

/// One-click cleanup: consolidate fragmented topics first, then run content dedup.
/// Default callers should pass groups resolved from the desired scope; for a full-store
/// cleanup, resolve with `all=true`.
pub async fn run_cleanup_async(
    store: &SqliteStore,
    config: &ReinConfig,
    groups: &[TopicGroup],
    merge_variants: bool,
    dry_run: bool,
) -> ReinResult<CleanupReport> {
    let consolidation = run_consolidation_async(store, config, groups, None, dry_run).await?;
    let threshold = config.search.dedup_similarity as f32;
    let (duplicates_found, duplicates_merged) =
        run_dedup_scoped(store, config, groups, threshold, dry_run, merge_variants)?;

    Ok(CleanupReport {
        consolidation,
        duplicates_found,
        duplicates_merged,
        dry_run,
    })
}

pub fn run_cleanup_sync(
    store: &SqliteStore,
    config: &ReinConfig,
    groups: &[TopicGroup],
    merge_variants: bool,
    dry_run: bool,
) -> ReinResult<CleanupReport> {
    let cfg = config.clone();
    let groups = groups.to_vec();

    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        tokio::task::block_in_place(|| {
            handle.block_on(run_cleanup_async(
                store,
                &cfg,
                &groups,
                merge_variants,
                dry_run,
            ))
        })
    } else {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| ReinError::Config(format!("failed to build tokio runtime: {e}")))?;
        rt.block_on(run_cleanup_async(
            store,
            &cfg,
            &groups,
            merge_variants,
            dry_run,
        ))
    }
}

pub(crate) fn load_group_memories(store: &SqliteStore, group: &TopicGroup) -> ReinResult<Vec<Memory>> {
    let mut memories = Vec::new();
    for topic in &group.topics {
        memories.extend(
            store
                .get_by_topic(topic)?
                .into_iter()
                .filter(is_current_consolidation_memory),
        );
    }
    Ok(memories)
}

fn emit_consolidation_events(
    store: &SqliteStore,
    group: &TopicGroup,
    new_id: &str,
    old_memories: &[Memory],
) {
    emit_cleanup_event(
        store,
        crate::store::adaptive::EventType::Store,
        Some(new_id.to_string()),
        Some(group.canonical_topic.clone()),
        serde_json::json!({
            "source": "consolidate",
            "source_topics": group.topics.clone(),
            "source_count": old_memories.len(),
        }),
    );

    for old_memory in old_memories {
        record_deleted_memory_as_evidence(store, new_id, old_memory);
        let _ = store.record_dedup_decision(DedupDecision {
            id: String::new(),
            winner_id: Some(new_id.to_string()),
            loser_id: None,
            canonical_id: Some(new_id.to_string()),
            lexical_score: None,
            embedding_score: None,
            relation: DedupRelation::Update,
            confidence: 0.85,
            reason: "consolidate".to_string(),
            operator: "manual".to_string(),
            reversible: true,
            merged_summary: Some(group.canonical_topic.clone()),
            novel_facts: vec![],
            conflict_detected: false,
            payload: Some(serde_json::json!({
                "source_memory_id": old_memory.id,
                "source_topic": old_memory.topic,
            })),
            created_at: chrono::Utc::now(),
        });
        emit_cleanup_event(
            store,
            crate::store::adaptive::EventType::Forget,
            Some(old_memory.id.clone()),
            Some(old_memory.topic.clone()),
            serde_json::json!({
                "source": "consolidate",
                "replacement_id": new_id,
                "canonical_topic": group.canonical_topic.clone(),
            }),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ReinConfig;
    use crate::store::SqliteStore;
    use chrono::Utc;

    fn test_memory(topic: &str, summary: &str, content: &str, keywords: Vec<&str>) -> Memory {
        Memory {
            id: ulid::Ulid::new().to_string(),
            layer: MemoryLayer::LTM,
            topic: topic.to_string(),
            summary: summary.to_string(),
            content: content.to_string(),
            keywords: keywords.into_iter().map(String::from).collect(),
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
    fn test_run_consolidation_merges_topic() {
        let store = SqliteStore::in_memory().unwrap();
        let config = ReinConfig::default();

        store
            .store(test_memory(
                "rust-testing",
                "unit tests",
                "Write unit tests with #[test]",
                vec!["rust", "testing"],
            ))
            .unwrap();
        store
            .store(test_memory(
                "rust-testing",
                "integration tests",
                "Integration tests go in tests/ directory",
                vec!["rust", "integration"],
            ))
            .unwrap();
        store
            .store(test_memory(
                "rust-testing",
                "test helpers",
                "Create helper functions for test fixtures",
                vec!["testing", "helpers"],
            ))
            .unwrap();

        let groups = crate::ops::resolve_topic_groups(&store, None, &[], None, true, false)
            .unwrap();
        let report = run_consolidation(&store, &config, &groups, None, false).unwrap();

        assert_eq!(report.groups_processed, 1, "should process 1 group");
        assert!(!report.dry_run);

        // After consolidation the topic should still have at least 1 memory
        let remaining = store.get_by_topic("rust-testing").unwrap();
        assert!(
            !remaining.is_empty(),
            "consolidated topic should have at least 1 memory"
        );
        // The consolidated memory should exist and have a created_id
        assert!(
            report.groups[0].created_id.is_some(),
            "non-dry-run should produce a created_id"
        );
    }

    #[test]
    fn test_run_consolidation_dry_run() {
        let store = SqliteStore::in_memory().unwrap();
        let config = ReinConfig::default();

        store
            .store(test_memory(
                "rust-testing",
                "unit tests",
                "Write unit tests with #[test]",
                vec!["rust"],
            ))
            .unwrap();
        store
            .store(test_memory(
                "rust-testing",
                "integration tests",
                "Integration tests in tests/ directory",
                vec!["rust"],
            ))
            .unwrap();
        store
            .store(test_memory(
                "rust-testing",
                "test helpers",
                "Helper functions for fixtures",
                vec!["testing"],
            ))
            .unwrap();

        let groups = crate::ops::resolve_topic_groups(&store, None, &[], None, true, false)
            .unwrap();
        let report = run_consolidation(&store, &config, &groups, None, true).unwrap();

        assert!(report.dry_run);
        assert_eq!(report.groups_processed, 1);
        assert!(
            report.groups[0].created_id.is_none(),
            "dry_run should not create a new memory"
        );

        // Original 3 memories must still exist unchanged
        let remaining = store.get_by_topic("rust-testing").unwrap();
        assert_eq!(remaining.len(), 3, "dry_run must not alter existing memories");
    }

    #[test]
    fn test_load_group_memories_prefers_current_memories() {
        let store = SqliteStore::in_memory().unwrap();

        let mut active = test_memory(
            "canonical-topic",
            "active summary",
            "Current canonical memory",
            vec!["active"],
        );
        active.status = MemoryStatus::Active;

        let mut updated = test_memory(
            "canonical-topic",
            "updated summary",
            "Current updated memory",
            vec!["updated"],
        );
        updated.status = MemoryStatus::Updated;

        let mut superseded = test_memory(
            "canonical-topic",
            "superseded summary",
            "Historical memory that should be ignored",
            vec!["old"],
        );
        superseded.superseded_by = Some("replacement-id".to_string());

        store.store(active).unwrap();
        store.store(updated).unwrap();
        store.store(superseded).unwrap();

        let group = TopicGroup {
            canonical_topic: "canonical-topic".to_string(),
            topics: vec!["canonical-topic".to_string()],
        };
        let memories = load_group_memories(&store, &group).unwrap();

        assert_eq!(memories.len(), 2, "should keep only current memories");
        assert!(
            memories
                .iter()
                .any(|memory| memory.status == MemoryStatus::Updated),
            "updated memories must remain eligible for consolidation"
        );
        assert!(
            memories.iter().all(|memory| memory.superseded_by.is_none()),
            "superseded history must not be fed into consolidation"
        );
    }

    #[tokio::test]
    async fn test_run_consolidation_async_prioritizes_larger_groups() {
        let store = SqliteStore::in_memory().unwrap();
        let config = ReinConfig::default();

        store
            .store(test_memory("small-topic", "one", "single memory", vec!["small"]))
            .unwrap();

        for i in 0..3 {
            store
                .store(test_memory(
                    "large-topic",
                    &format!("large {i}"),
                    &format!("large memory {i}"),
                    vec!["large"],
                ))
                .unwrap();
        }

        let groups = crate::ops::resolve_topic_groups(&store, None, &[], None, true, false)
            .unwrap();
        let report = run_consolidation_async(&store, &config, &groups, None, true)
            .await
            .unwrap();

        let first_non_empty = report.groups.iter().find(|group| group.memory_count > 0).unwrap();
        assert_eq!(first_non_empty.canonical_topic, "large-topic");
    }

    #[test]
    fn test_build_consolidated_from_memories() {
        let config = ReinConfig::default();

        let m1 = test_memory(
            "rust-testing",
            "unit tests",
            "Write unit tests with #[test] attribute",
            vec!["rust", "unit-test"],
        );
        let m2 = test_memory(
            "rust-testing",
            "integration tests",
            "Integration tests live in tests/ directory",
            vec!["rust", "integration"],
        );
        let m3 = test_memory(
            "rust-testing",
            "test helpers",
            "Helper functions simplify test setup",
            vec!["helpers", "rust"],
        );

        let memories = vec![m1, m2, m3];
        let source_topics = vec!["rust-testing".to_string()];
        let consolidated = build_consolidated_from_memories(
            &config,
            "rust-testing".to_string(),
            &source_topics,
            &memories,
            None,
        );

        // Content should reference all source memories
        assert!(
            consolidated.content.contains("unit tests")
                || consolidated.content.contains("integration tests")
                || consolidated.content.contains("test helpers"),
            "consolidated content should reference source summaries"
        );

        // Keywords should be merged and deduplicated (case-insensitive)
        assert!(
            consolidated.keywords.contains(&"rust".to_string()),
            "merged keywords should contain 'rust'"
        );
        assert!(
            consolidated.keywords.contains(&"unit-test".to_string()),
            "merged keywords should contain 'unit-test'"
        );
        assert!(
            consolidated.keywords.contains(&"integration".to_string()),
            "merged keywords should contain 'integration'"
        );
        assert!(
            consolidated.keywords.contains(&"helpers".to_string()),
            "merged keywords should contain 'helpers'"
        );

        // No duplicate 'rust' (case-insensitive dedup)
        let rust_count = consolidated
            .keywords
            .iter()
            .filter(|k| k.to_lowercase() == "rust")
            .count();
        assert_eq!(rust_count, 1, "keywords should be deduplicated");

        assert_eq!(consolidated.topic, "rust-testing");
        assert_eq!(consolidated.support_count, 3);
        assert_eq!(consolidated.merge_count, 2);
    }

    #[test]
    fn test_run_cleanup_consolidates_and_dedup() {
        let store = SqliteStore::in_memory().unwrap();
        let config = ReinConfig::default();

        // 3 memories under "docker-deployment", 2 of which are near-duplicates
        store
            .store(test_memory(
                "docker-deployment",
                "compose setup",
                "Use docker compose for the local development stack",
                vec!["docker", "compose"],
            ))
            .unwrap();
        store
            .store(test_memory(
                "docker-deployment",
                "compose setup dup",
                "Use docker compose for the local development stack",
                vec!["docker", "compose"],
            ))
            .unwrap();
        store
            .store(test_memory(
                "docker-deployment",
                "image pinning",
                "Always pin Docker image tags to specific versions",
                vec!["docker", "images"],
            ))
            .unwrap();

        // 2 memories under "rust-testing"
        store
            .store(test_memory(
                "rust-testing",
                "unit tests",
                "Write unit tests with #[test]",
                vec!["rust"],
            ))
            .unwrap();
        store
            .store(test_memory(
                "rust-testing",
                "integration tests",
                "Integration tests in tests/ directory",
                vec!["rust"],
            ))
            .unwrap();

        let groups =
            crate::ops::resolve_topic_groups(&store, None, &[], None, true, true).unwrap();

        // Use the sync run_consolidation + run_dedup directly to avoid needing
        // the full async cleanup pipeline (which tries LLM synthesis).
        let consolidation = run_consolidation(&store, &config, &groups, None, false).unwrap();
        let threshold = config.search.dedup_similarity as f32;
        let (duplicates_found, duplicates_merged) =
            crate::ops::dedup::run_dedup(&store, &config, threshold, false, true).unwrap();

        let report = CleanupReport {
            consolidation,
            duplicates_found,
            duplicates_merged,
            dry_run: false,
        };

        assert!(
            report.consolidation.groups_processed >= 2,
            "should consolidate at least 2 topic groups, got {}",
            report.consolidation.groups_processed
        );
        assert!(!report.dry_run);
    }

    #[test]
    fn test_build_consolidated_from_extracted_preserves_llm_summary_length() {
        let config = ReinConfig::default();
        let memories = vec![test_memory(
            "rust-testing",
            "unit tests",
            "Write unit tests with #[test] attribute",
            vec!["rust"],
        )];
        let extracted = ExtractedMemory {
            topic: "rust-testing".to_string(),
            summary: "This is a deliberately long LLM-generated summary that should remain intact instead of being mechanically truncated to one hundred characters after consolidation.".to_string(),
            content: "Detailed consolidated content".to_string(),
            keywords: vec!["rust".to_string()],
            importance: "high".to_string(),
            should_store: true,
            quality_confidence: 0.9,
        };

        let consolidated = build_consolidated_from_extracted(
            &config,
            "rust-testing".to_string(),
            &["rust-testing".to_string()],
            &memories,
            extracted,
        );

        assert!(
            consolidated.summary.len() > 100,
            "LLM summary should no longer be mechanically truncated"
        );
    }

    #[tokio::test]
    async fn test_run_cleanup_respects_selected_groups_for_dedup() {
        let store = SqliteStore::in_memory().unwrap();
        let config = ReinConfig::default();

        store
            .store(test_memory(
                "docker",
                "compose setup",
                "Use docker compose for the local development stack",
                vec!["docker"],
            ))
            .unwrap();
        store
            .store(test_memory(
                "docker",
                "compose setup duplicate",
                "Use docker compose for the local development stack",
                vec!["docker"],
            ))
            .unwrap();
        let rust_1 = store
            .store(test_memory(
                "rust-testing",
                "unit tests",
                "Write unit tests with #[test]",
                vec!["rust"],
            ))
            .unwrap();
        let rust_2 = store
            .store(test_memory(
                "rust-testing",
                "unit tests duplicate",
                "Write unit tests with #[test]",
                vec!["rust"],
            ))
            .unwrap();

        let groups = crate::ops::resolve_topic_groups(
            &store,
            Some("docker".to_string()).as_deref(),
            &[],
            None,
            false,
            false,
        )
        .unwrap();

        let report = run_cleanup_async(&store, &config, &groups, false, false)
            .await
            .unwrap();
        assert_eq!(report.consolidation.groups_processed, 1);

        let rust_superseded = [store.get(&rust_1).unwrap(), store.get(&rust_2).unwrap()]
            .into_iter()
            .filter(|memory| memory.superseded_by.is_some())
            .count();
        let current_docker = load_group_memories(
            &store,
            &TopicGroup {
                canonical_topic: "docker".to_string(),
                topics: vec!["docker".to_string()],
            },
        )
        .unwrap();

        assert_eq!(current_docker.len(), 1, "selected topic should be consolidated");
        assert_eq!(rust_superseded, 0, "non-selected topic must remain untouched");
    }

    #[test]
    fn test_emit_consolidation_events() {
        let store = SqliteStore::in_memory().unwrap();
        let config = ReinConfig::default();

        store
            .store(test_memory(
                "rust-testing",
                "unit tests",
                "Write unit tests with #[test]",
                vec!["rust"],
            ))
            .unwrap();
        store
            .store(test_memory(
                "rust-testing",
                "integration tests",
                "Integration tests in tests/ directory",
                vec!["rust"],
            ))
            .unwrap();

        let before_count = crate::store::adaptive::event_count(store.conn());

        let groups = crate::ops::resolve_topic_groups(&store, None, &[], None, true, false)
            .unwrap();
        // Run actual consolidation (not dry_run) which internally calls emit_consolidation_events
        let report = run_consolidation(&store, &config, &groups, None, false).unwrap();
        assert!(report.groups_processed >= 1);

        let after_count = crate::store::adaptive::event_count(store.conn());
        assert!(
            after_count > before_count,
            "consolidation should emit feedback events: before={before_count}, after={after_count}"
        );
    }
}
