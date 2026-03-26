//! Shared business operations used by both CLI (main.rs) and MCP (server.rs).
//! Extracted to prevent logic drift between the two entrypoints.

use crate::config::ReinConfig;
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
