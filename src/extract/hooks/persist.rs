//! Durable persistence helpers shared by hooks and async memory worker.

use crate::config::ReinConfig;
use crate::extract::llm::{ExtractedMemory, ExtractionResult};
use crate::types::MemoryTier;

use super::buffer::store_episode_concept;
use super::parsing::looks_like_secret;
use super::working_set::{update_always_on_index, update_working_set};

#[derive(Debug, Default, Clone)]
pub struct StoreExtractedStats {
    pub stored_count: u32,
    pub stored_ids: Vec<String>,
    pub filtered_count: u32,
    pub secret_filtered_count: u32,
    pub created_count: u32,
    pub merged_count: u32,
    pub superseded_count: u32,
}

/// Compute adaptive admission threshold from recent quality data.
/// Base = 0.2. Adjusts up if recent quality is low, down if high.
fn adaptive_admission_threshold(store: &crate::store::SqliteStore) -> f64 {
    let base = 0.2;
    let recent_avg: f64 = store.conn()
        .query_row(
            "SELECT COALESCE(AVG(strength), 0.5) FROM (SELECT strength FROM memories ORDER BY created_at DESC LIMIT 100)",
            [],
            |r| r.get(0),
        ).unwrap_or(0.5);

    if recent_avg < 0.4 {
        (base * 1.1_f64).min(0.60)
    } else if recent_avg > 0.7 {
        (base * 0.9_f64).max(0.15)
    } else {
        base
    }
}

/// Multi-factor admission score (A-MAC 2026 inspired).
fn multi_factor_admission_score(
    store: &crate::store::SqliteStore,
    item: &ExtractedMemory,
) -> f64 {
    let llm_conf = item.quality_confidence;

    let novelty = {
        let existing = store.get_by_topic(&item.topic).unwrap_or_default();
        if existing.is_empty() {
            1.0
        } else {
            let max_sim = existing.iter()
                .map(|m| crate::extract::similarity(&item.content, &m.content))
                .fold(0.0_f32, f32::max);
            (1.0 - max_sim as f64).max(0.0)
        }
    };

    let type_prior = {
        let t = item.topic.to_lowercase();
        if ["architecture", "decision", "design"].iter().any(|k| t.contains(k)) {
            0.9
        } else if ["workflow", "deployment", "config"].iter().any(|k| t.contains(k)) {
            0.7
        } else if ["debug", "error", "fix"].iter().any(|k| t.contains(k)) {
            0.5
        } else {
            0.6
        }
    };

    let recency = 0.7;

    if llm_conf < 0.05 {
        return 0.0;
    }

    0.45 * llm_conf + 0.25 * novelty + 0.15 * type_prior + 0.15 * recency
}

/// Store a list of ExtractedMemory items into the database.
/// Filters secrets and deduplicates. Returns (count, stored_ids).
pub fn store_extracted(
    store: &crate::store::SqliteStore,
    config: &ReinConfig,
    mut items: Vec<ExtractedMemory>,
    agent_label: &str,
    is_subagent: bool,
) -> (u32, Vec<String>) {
    let stats = store_extracted_report(store, config, items, agent_label, is_subagent);
    (stats.stored_count, stats.stored_ids)
}

pub fn store_extracted_report(
    store: &crate::store::SqliteStore,
    config: &ReinConfig,
    mut items: Vec<ExtractedMemory>,
    agent_label: &str,
    is_subagent: bool,
) -> StoreExtractedStats {
    for item in &mut items {
        crate::extract::postprocess::postprocess(item);
        if !item.keywords.iter().any(|k| k == &format!("agent:{agent_label}")) {
            item.keywords.push(format!("agent:{agent_label}"));
        }
        if is_subagent && !item.keywords.iter().any(|k| k == "source:subagent") {
            item.keywords.push("source:subagent".to_string());
        }
    }

    let mut stats = StoreExtractedStats::default();
    for item in items {
        if looks_like_secret(&item.content) {
            stats.secret_filtered_count += 1;
            continue;
        }

        let threshold = adaptive_admission_threshold(store);
        let admission = multi_factor_admission_score(store, &item);
        if admission < threshold {
            tracing::debug!(
                "skipping low-quality memory (admission={:.2} < threshold={:.2}): {}",
                admission,
                threshold,
                item.summary
            );
            stats.filtered_count += 1;
            continue;
        }

        let content_for_activation = item.content.clone();
        let importance = item.importance.parse::<crate::types::Importance>()
            .unwrap_or(crate::types::Importance::Medium);
        let proposed_id = ulid::Ulid::new().to_string();
        let memory = crate::types::Memory {
            id: proposed_id.clone(),
            layer: importance.auto_layer(),
            topic: item.topic,
            summary: item.summary,
            content: item.content,
            keywords: item.keywords,
            importance,
            source: crate::types::Source::Hook,
            strength: item.quality_confidence.max(0.3),
            decay_lambda: config.decay.base_lambda * importance.decay_factor(),
            access_count: 0,
            superseded_by: None,
            related_ids: vec![],
            concept_ids: vec![],
            status: crate::types::MemoryStatus::default(),
            embedding: None,
            tier: MemoryTier::Warm,
            cluster_id: None,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            last_accessed: chrono::Utc::now(),
        };
        if let Ok(id) = store.store_with_dedup(
            memory,
            config.search.dedup_similarity as f32,
            config.search.dedup_time_window_days,
        ) {
            let _ = store.auto_link(&id, config.search.dedup_similarity as f32, 5);
            let _ = store.activate_related_memories(&content_for_activation, 3);
            let _ = store.activate_related_concepts(&content_for_activation);
            let _ = store.apply_evolution(&id, &content_for_activation, None);

            stats.stored_ids.push(id.clone());
            stats.stored_count += 1;
            if id != proposed_id {
                stats.merged_count += 1;
            } else {
                let superseded_rows: u32 = store
                    .conn()
                    .query_row(
                        "SELECT COUNT(*) FROM memories WHERE superseded_by = ?1",
                        rusqlite::params![&id],
                        |r| r.get(0),
                    )
                    .unwrap_or(0);
                if superseded_rows > 0 {
                    stats.superseded_count += 1;
                } else {
                    stats.created_count += 1;
                }
            }
        }
    }
    stats
}

pub fn process_quick_extraction(
    config: &ReinConfig,
    extracted: Vec<ExtractedMemory>,
    agent_label: &str,
    is_subagent: bool,
) -> anyhow::Result<u32> {
    if extracted.is_empty() {
        return Ok(0);
    }
    let store = config.open_store()?;
    let extracted_for_ws = extracted.clone();
    let (stored, _ids) = store_extracted(&store, config, extracted, agent_label, is_subagent);
    let _ = update_working_set(config, &extracted_for_ws, &[], None, agent_label, is_subagent);
    let _ = update_always_on_index(config, &extracted_for_ws, &[], None, agent_label, is_subagent);
    Ok(stored)
}

pub fn process_full_extraction(
    config: &ReinConfig,
    mut result: ExtractionResult,
    agent_label: &str,
    is_subagent: bool,
) -> anyhow::Result<(u32, u32, u32)> {
    let max_items = config.hooks.max_items_per_session;
    result.memories.truncate(max_items);

    if result.memories.is_empty() && result.concepts.is_empty() && result.episode.is_none() {
        return Ok((0, 0, 0));
    }

    let store = config.open_store()?;
    let episode_for_ws = result.episode.clone();
    let memories_for_ws = result.memories.clone();
    let concepts_for_ws = result.concepts.clone();
    let (mem_count, memory_ids) = store_extracted(&store, config, result.memories, agent_label, is_subagent);
    let _ = update_working_set(config, &memories_for_ws, &concepts_for_ws, episode_for_ws.as_ref(), agent_label, is_subagent);
    let _ = update_always_on_index(config, &memories_for_ws, &concepts_for_ws, episode_for_ws.as_ref(), agent_label, is_subagent);
    let kg_report = store.store_knowledge_units_with_sources(&result.concepts, &result.links, &memory_ids)
        .unwrap_or_default();

    let session_concept_ids: Vec<String> = result.concepts.iter()
        .filter_map(|c| {
            store.get_concept(&c.memoir, &c.name).ok().flatten().map(|con| con.id)
        })
        .collect();

    if let Some(ref ep) = result.episode {
        let episode = crate::types::Episode {
            id: String::new(),
            title: ep.title.clone(),
            outcome: ep.outcome.clone(),
            decisions: ep.decisions.clone(),
            primary_topics: vec![],
            tags: vec![],
            involved_agents: vec![agent_label.to_string()],
            important_paths: vec![],
            temporal_keywords: vec![],
            source_session_id: None,
            concept_ids: session_concept_ids.clone(),
            memory_ids: memory_ids.clone(),
            created_at: chrono::Utc::now(),
        };
        match store.create_episode(episode) {
            Ok(episode_id) => {
                for cid in &session_concept_ids {
                    let _ = store.conn().execute(
                        "UPDATE concepts SET last_episode_id = ?1 WHERE id = ?2",
                        rusqlite::params![episode_id, cid],
                    );
                    let cutoff = (chrono::Utc::now() - chrono::Duration::hours(24)).to_rfc3339();
                    let _ = store.conn().execute(
                        "UPDATE concept_revisions SET episode_id = ?1 WHERE concept_id = ?2 AND episode_id IS NULL AND created_at >= ?3",
                        rusqlite::params![episode_id, cid, cutoff],
                    );
                }
            }
            Err(e) => tracing::warn!("failed to create episode: {e}"),
        }

        if let Err(e) = store_episode_concept(&store, ep) {
            tracing::warn!("failed to store episode concept: {e}");
        }
    }

    Ok((
        mem_count,
        (kg_report.concepts_added + kg_report.concepts_refined) as u32,
        kg_report.links_added as u32,
    ))
}
