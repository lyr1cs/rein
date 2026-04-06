//! Durable persistence helpers shared by hooks and async memory worker.

use crate::config::ReinConfig;
use crate::extract::dedup::normalize_topic_key;
use crate::extract::llm::{ExtractedMemory, ExtractionResult};
use crate::types::traits::MemoryStore;
use crate::types::{Memory, MemoryStatus, MemoryTier};

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
fn multi_factor_admission_score(store: &crate::store::SqliteStore, item: &ExtractedMemory) -> f64 {
    let llm_conf = item.quality_confidence;

    let novelty = {
        let existing = current_topic_memories(store, &item.topic);
        novelty_from_memories(&existing, &item.content)
    };

    let type_prior = {
        let t = item.topic.to_lowercase();
        if ["architecture", "decision", "design"]
            .iter()
            .any(|k| t.contains(k))
        {
            0.9
        } else if ["workflow", "deployment", "config"]
            .iter()
            .any(|k| t.contains(k))
        {
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

fn novelty_from_memories(existing: &[Memory], content: &str) -> f64 {
    if existing.is_empty() {
        return 1.0;
    }
    let max_sim = existing
        .iter()
        .map(|m| crate::extract::similarity(content, &m.content))
        .fold(0.0_f32, f32::max);
    (1.0 - f64::from(max_sim)).max(0.0)
}

fn current_topic_memories(store: &crate::store::SqliteStore, topic: &str) -> Vec<Memory> {
    let mut seen = std::collections::HashSet::new();
    let normalized = normalize_topic_key(topic);
    let mut memories = Vec::new();
    for existing_topic in store.list_topics().unwrap_or_default() {
        if normalize_topic_key(&existing_topic) == normalized {
            memories.extend(store.get_by_topic(&existing_topic).unwrap_or_default());
        }
    }

    memories
        .into_iter()
        .filter(|memory| {
            memory.superseded_by.is_none()
                && matches!(memory.status, MemoryStatus::Active | MemoryStatus::Updated)
        })
        .filter(|memory| {
            let canonical_key = store
                .canonical_id_for(&memory.id)
                .unwrap_or_else(|_| memory.id.clone());
            seen.insert(canonical_key)
        })
        .collect()
}

/// Store a list of ExtractedMemory items into the database.
/// Filters secrets and deduplicates. Returns (count, stored_ids).
pub fn store_extracted(
    store: &crate::store::SqliteStore,
    config: &ReinConfig,
    items: Vec<ExtractedMemory>,
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
        if !item
            .keywords
            .iter()
            .any(|k| k == &format!("agent:{agent_label}"))
        {
            item.keywords.push(format!("agent:{agent_label}"));
        }
        if is_subagent && !item.keywords.iter().any(|k| k == "source:subagent") {
            item.keywords.push("source:subagent".to_string());
        }
    }

    let mut stats = StoreExtractedStats::default();
    // Compute threshold once per batch to avoid N redundant DB queries.
    // The threshold (avg of last 100 strengths) shifts negligibly as items are added.
    let threshold = adaptive_admission_threshold(store);
    for item in items {
        if looks_like_secret(&item.content) {
            stats.secret_filtered_count += 1;
            continue;
        }

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
        let importance = item
            .importance
            .parse::<crate::types::Importance>()
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
            canonical_id: None,
            support_count: 1,
            merge_count: 0,
            dedup_confidence: item.quality_confidence as f32,
            source_diversity: 1.0,
            contradiction_score: 0.0,
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
            stats.stored_ids.push(id.clone());
            stats.stored_count += 1;
            if id != proposed_id {
                stats.merged_count += 1;
            } else {
                let _ = store.auto_link(&id, config.search.dedup_similarity as f32, 5);
                let _ = store.activate_related_memories(&content_for_activation, 3);
                let _ = store.activate_related_concepts(&content_for_activation);
                let _ = store.apply_evolution(&id, &content_for_activation, None);
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

fn surface_memories_for_ids(
    store: &crate::store::SqliteStore,
    ids: &[String],
) -> Vec<ExtractedMemory> {
    let mut seen = std::collections::HashSet::new();
    let mut memories = Vec::new();

    for id in ids {
        let Ok(memory) = store.get_canonical(id) else {
            continue;
        };
        if !seen.insert(memory.id.clone()) {
            continue;
        }

        memories.push(ExtractedMemory {
            topic: memory.topic.clone(),
            summary: memory.summary.clone(),
            content: memory.content.clone(),
            keywords: memory.keywords.clone(),
            importance: format!("{}", memory.importance),
            should_store: true,
            quality_confidence: f64::from(memory.dedup_confidence).max(memory.strength),
        });
    }

    memories
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
    let (stored, ids) = store_extracted(&store, config, extracted, agent_label, is_subagent);
    let memories_for_ws = surface_memories_for_ids(&store, &ids);
    let _ = update_working_set(
        config,
        &memories_for_ws,
        &[],
        None,
        agent_label,
        is_subagent,
    );
    let _ = update_always_on_index(
        config,
        &memories_for_ws,
        &[],
        None,
        agent_label,
        is_subagent,
    );
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
    let concepts_for_ws = result.concepts.clone();
    let (mem_count, memory_ids) =
        store_extracted(&store, config, result.memories, agent_label, is_subagent);
    let memories_for_ws = surface_memories_for_ids(&store, &memory_ids);
    let _ = update_working_set(
        config,
        &memories_for_ws,
        &concepts_for_ws,
        episode_for_ws.as_ref(),
        agent_label,
        is_subagent,
    );
    let _ = update_always_on_index(
        config,
        &memories_for_ws,
        &concepts_for_ws,
        episode_for_ws.as_ref(),
        agent_label,
        is_subagent,
    );
    let kg_report = store
        .store_knowledge_units_with_sources(&result.concepts, &result.links, &memory_ids)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extract::hooks::working_set::{
        load_always_on_index, load_working_set,
    };
    use crate::types::traits::MemoryStore;
    use crate::store::SqliteStore;
    use crate::types::{Importance, MemoryLayer, MemoryStatus, MemoryTier, Source};

    fn test_config(name: &str) -> ReinConfig {
        let mut config = ReinConfig::default();
        let stamp = chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default();
        let base = std::env::temp_dir().join(format!("rein-persist-{name}-{stamp}"));
        std::fs::create_dir_all(&base).unwrap();
        config.database.path = base.join("memories.db").display().to_string();
        config.hooks.buffer_dir = base.join("buffers").display().to_string();
        config
    }

    fn extracted_memory(topic: &str, summary: &str, content: &str) -> ExtractedMemory {
        ExtractedMemory {
            topic: topic.to_string(),
            summary: summary.to_string(),
            content: content.to_string(),
            keywords: vec![],
            importance: "high".to_string(),
            should_store: true,
            quality_confidence: 0.9,
        }
    }

    fn stored_memory(topic: &str, summary: &str, content: &str) -> crate::types::Memory {
        crate::types::Memory {
            id: ulid::Ulid::new().to_string(),
            layer: MemoryLayer::LTM,
            topic: topic.to_string(),
            summary: summary.to_string(),
            content: content.to_string(),
            keywords: vec![],
            importance: Importance::High,
            source: Source::Hook,
            strength: 0.9,
            decay_lambda: 0.06,
            access_count: 0,
            superseded_by: None,
            canonical_id: None,
            support_count: 1,
            merge_count: 0,
            dedup_confidence: 0.9,
            source_diversity: 1.0,
            contradiction_score: 0.0,
            related_ids: vec![],
            concept_ids: vec![],
            status: MemoryStatus::Active,
            embedding: None,
            tier: MemoryTier::Warm,
            cluster_id: None,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            last_accessed: chrono::Utc::now(),
        }
    }

    #[test]
    fn current_topic_memories_excludes_superseded_and_keeps_updated() {
        let store = SqliteStore::in_memory().unwrap();

        let active_id = store
            .store(stored_memory(
                "docker",
                "active memory",
                "Current canonical docker note",
            ))
            .unwrap();
        let mut active = store.get(&active_id).unwrap();
        active.summary = "updated summary".to_string();
        store.update(&active).unwrap();

        let loser_id = store
            .store(stored_memory(
                "docker",
                "duplicate memory",
                "Current canonical docker note",
            ))
            .unwrap();
        let winner_id = store
            .store(stored_memory(
                "docker",
                "winner memory",
                "Different docker guidance",
            ))
            .unwrap();
        store.mark_superseded(&loser_id, &winner_id).unwrap();

        let current = current_topic_memories(&store, "docker");
        assert_eq!(current.len(), 2);
        assert!(current
            .iter()
            .any(|memory| memory.status == MemoryStatus::Updated));
        assert!(current.iter().all(|memory| memory.superseded_by.is_none()));
    }

    #[test]
    fn canonical_view_novelty_ignores_superseded_duplicates() {
        let store = SqliteStore::in_memory().unwrap();

        let duplicate_id = store
            .store(stored_memory(
                "docker",
                "duplicate memory",
                "Use docker compose for local development",
            ))
            .unwrap();
        let current_id = store
            .store(stored_memory(
                "docker",
                "current memory",
                "Use docker compose for deployment docs",
            ))
            .unwrap();
        store.mark_superseded(&duplicate_id, &current_id).unwrap();

        let item = extracted_memory(
            "docker",
            "new docker note",
            "Use docker compose for local development",
        );

        let raw_memories = store.get_by_topic(&item.topic).unwrap_or_default();
        let raw_novelty = novelty_from_memories(&raw_memories, &item.content);
        let canonical_novelty =
            novelty_from_memories(&current_topic_memories(&store, &item.topic), &item.content);

        assert!(
            canonical_novelty > raw_novelty,
            "canonical view should be more novel than raw topic view"
        );
        assert!(
            canonical_novelty > 0.0,
            "canonical view should still leave room for novelty"
        );
    }

    #[test]
    fn quick_extraction_updates_surfaces_from_canonical_memories() {
        let config = test_config("quick");
        let extracted = vec![
            extracted_memory(
                "docker",
                "compose setup",
                "Use docker compose for local development and testing",
            ),
            extracted_memory(
                "docker",
                "compose setup variant",
                "Use docker compose for local development and deployment",
            ),
        ];

        let stored = process_quick_extraction(&config, extracted, "tester", false).unwrap();
        assert_eq!(stored, 2, "both extracted memories should be admitted");

        let working = load_working_set(&config);
        let always_on = load_always_on_index(&config);

        assert_eq!(working.len(), 1, "working set should reflect one canonical memory");
        assert_eq!(always_on.len(), 1, "always-on index should reflect one canonical memory");
        assert!(
            working[0].detail.contains("docker compose"),
            "surface item should be built from stored canonical content"
        );
    }

    #[test]
    fn current_topic_memories_normalizes_topic_variants() {
        let config = test_config("variant-topic");
        let _ = process_quick_extraction(
            &config,
            vec![extracted_memory(
                "docker-deployment",
                "compose setup",
                "Use docker compose for local development and testing",
            )],
            "tester",
            false,
        )
        .unwrap();

        let store = config.open_store().unwrap();
        let current = current_topic_memories(&store, "Docker Deployment");
        assert_eq!(current.len(), 1);
        assert_eq!(current[0].topic, "docker-deployment");
    }
}
