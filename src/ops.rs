//! Shared business operations used by both CLI (main.rs) and MCP (server.rs).
//! Extracted to prevent logic drift between the two entrypoints.

use crate::config::ReinConfig;
use crate::extract;
use crate::store::SqliteStore;
use crate::types::*;

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
    Memory {
        id: ulid::Ulid::new().to_string(),
        layer: importance.auto_layer(),
        topic,
        summary: content.chars().take(100).collect(),
        content,
        keywords,
        importance,
        source,
        strength: 1.0,
        decay_lambda: config.decay.base_lambda * importance.decay_factor(),
        access_count: 0,
        superseded_by: None,
        related_ids: vec![],
        concept_ids: vec![],
        status: MemoryStatus::default(),
        embedding: None,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        last_accessed: chrono::Utc::now(),
    }
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
        summary: summary.chars().take(100).collect(),
        content: summary,
        keywords: vec![],
        importance,
        source: Source::Manual,
        strength: 1.0,
        decay_lambda: config.decay.base_lambda * importance.decay_factor(),
        access_count: 0,
        superseded_by: None,
        related_ids,
        concept_ids: vec![],
        status: MemoryStatus::default(),
        embedding: None,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        last_accessed: chrono::Utc::now(),
    }
}

/// Run GC: apply decay + prune weak memories + prune low-quality concepts.
/// In dry-run mode, wraps operations in a savepoint to preview without committing.
/// Returns (decayed_count, memory_pruned_count, concept_pruned_count).
pub fn run_gc(store: &SqliteStore, threshold: f64, dry_run: bool) -> ReinResult<(u64, u64, u64)> {
    if dry_run {
        store.conn().execute_batch("SAVEPOINT gc_preview")
            .map_err(crate::types::ReinError::Database)?;

        let decayed = store.apply_decay()?;

        // Actually execute prune within savepoint so concept evaluation sees
        // the same DB state as a real GC (memories deleted first, then concepts).
        let mem_pruned = store.prune_memories_only(threshold)?;
        let concept_pruned = store.prune_low_quality_concepts().unwrap_or(0);

        store.conn().execute_batch("ROLLBACK TO gc_preview")
            .map_err(crate::types::ReinError::Database)?;
        store.conn().execute_batch("RELEASE gc_preview")
            .map_err(crate::types::ReinError::Database)?;

        Ok((decayed, mem_pruned, concept_pruned))
    } else {
        let decayed = store.apply_decay()?;
        let mem_pruned = store.prune_memories_only(threshold)?;
        let concept_pruned = store.prune_low_quality_concepts().unwrap_or(0);
        if concept_pruned > 0 {
            tracing::info!("pruned {concept_pruned} low-quality concepts");
        }
        Ok((decayed, mem_pruned, concept_pruned))
    }
}

/// Run dedup scan across all topics.
/// Returns (duplicates_found, duplicates_removed).
pub fn run_dedup(store: &SqliteStore, threshold: f32, dry_run: bool) -> ReinResult<(u32, u32)> {
    let topics = store.list_topics()?;
    let mut dups_found = 0u32;
    let mut dups_removed = 0u32;
    for topic in &topics {
        let mems = store.get_by_topic(topic)?;
        let mut to_delete: std::collections::HashSet<String> = std::collections::HashSet::new();
        for i in 0..mems.len() {
            if to_delete.contains(&mems[i].id) { continue; }
            for j in (i + 1)..mems.len() {
                if to_delete.contains(&mems[j].id) { continue; }
                let sim = crate::extract::similarity(&mems[i].content, &mems[j].content);
                if sim >= threshold {
                    to_delete.insert(mems[i].id.clone());
                    dups_found += 1;
                    if dry_run {
                        tracing::debug!("dup: '{}' ~ '{}'", &mems[i].summary.chars().take(40).collect::<String>(), &mems[j].summary.chars().take(40).collect::<String>());
                    }
                    break;
                }
            }
        }
        if !dry_run {
            for id in &to_delete {
                store.delete(id)?;
                dups_removed += 1;
            }
        }
    }
    Ok((dups_found, dups_removed))
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
        if memories.is_empty() { continue; }
        report.topics_processed += 1;

        let combined: String = memories.iter()
            .map(|m| format!("[{}] {}\n{}", m.topic, m.summary, m.content))
            .collect::<Vec<_>>()
            .join("\n---\n");

        let result = extract::llm::extract_full_with_fallback(config, &combined).await;

        if has_llm {
            if dry_run {
                let enrichable = result.memories.len().min(memories.len());
                report.preview_lines.push(format!(
                    "  topic '{}': would enrich {} memories, create {} concepts, {} links",
                    topic_name, enrichable, result.concepts.len(), result.links.len()
                ));
                for c in &result.concepts {
                    report.preview_lines.push(format!("    concept: [{}] {} ({})", c.memoir, c.name, c.concept_type));
                }
                for l in &result.links {
                    report.preview_lines.push(format!("    link: {} --{}-> {}", l.from, l.relation, l.to));
                }
                report.concepts += result.concepts.len();
                report.links += result.links.len();
                report.enriched += enrichable;
            } else {
                // LLM quality audit + enrichment
                for new_mem in &result.memories {
                    let best_match = memories.iter()
                        .max_by(|a, b| {
                            let sim_a = extract::similarity(&a.content, &new_mem.content);
                            let sim_b = extract::similarity(&b.content, &new_mem.content);
                            sim_a.partial_cmp(&sim_b).unwrap_or(std::cmp::Ordering::Equal)
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
                                enriched.decay_lambda = config.decay.base_lambda * imp.decay_factor();
                            }
                            if store.update(&enriched).is_ok() {
                                report.enriched += 1;
                            }
                        }
                    }
                }

                let memory_ids: Vec<String> = memories.iter().map(|m| m.id.clone()).collect();
                if !result.concepts.is_empty() || !result.links.is_empty() {
                    match store.store_knowledge_units_with_sources(&result.concepts, &result.links, &memory_ids) {
                        Ok(r) => {
                            report.memoirs += r.memoirs_created;
                            report.concepts += r.concepts_added + r.concepts_refined;
                            report.links += r.links_added;
                        }
                        Err(e) => tracing::warn!("knowledge_units error for topic '{}': {e}", topic_name),
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
                if old.topic != "auto-extracted" { continue; }
                let lower = old.content.to_lowercase();
                let new_topic = if ["architecture", "design", "component", "架构", "设计"].iter().any(|k| lower.contains(k)) {
                    "architecture"
                } else if ["decided", "chose", "选型", "决策", "tradeoff"].iter().any(|k| lower.contains(k)) {
                    "decision"
                } else if ["bug", "fix", "error", "crash", "修复", "解决"].iter().any(|k| lower.contains(k)) {
                    "debug"
                } else if ["deploy", "install", "config", "migrate", "部署", "安装", "迁移"].iter().any(|k| lower.contains(k)) {
                    "workflow"
                } else {
                    "general"
                };

                let keywords: Vec<String> = old.content
                    .split_whitespace()
                    .filter(|w| w.len() > 3)
                    .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
                    .filter(|w| !w.is_empty() && !["the", "this", "that", "with", "from", "have", "been", "into", "will"].contains(&w.as_str()))
                    .take(5)
                    .collect();

                if dry_run {
                    if new_topic != "auto-extracted" || !keywords.is_empty() {
                        report.preview_lines.push(format!(
                            "  → would reclassify '{}' → topic='{}', keywords={:?}",
                            old.summary.chars().take(40).collect::<String>(), new_topic, keywords
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
                        enriched.decay_lambda = config.decay.base_lambda * enriched.importance.decay_factor();
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
