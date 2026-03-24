use crate::config::ReinConfig;
use crate::store::SqliteStore;
use crate::types::MemoryStore;

/// Layer 0: PostToolUse -- extract facts from tool output.
/// Reads JSON from stdin (tool output), extracts important sentences, stores as Source::Hook.
pub async fn hook_post(config: &ReinConfig) -> anyhow::Result<()> {
    let input = std::io::read_to_string(std::io::stdin())?;
    let facts = crate::extract::patterns::extract_facts(&input, 3);

    if facts.is_empty() {
        return Ok(());
    }

    let store = config.open_store()?;

    for fact in facts {
        let importance = crate::types::Importance::Medium;
        let memory = crate::types::Memory {
            id: ulid::Ulid::new().to_string(),
            layer: importance.auto_layer(),
            topic: "auto-extracted".to_string(),
            summary: fact.chars().take(100).collect(),
            content: fact,
            keywords: vec![],
            importance,
            source: crate::types::Source::Hook,
            strength: 1.0,
            decay_lambda: config.decay.base_lambda * importance.decay_factor(),
            access_count: 0,
            superseded_by: None,
            related_ids: vec![],
            embedding: None,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            last_accessed: chrono::Utc::now(),
        };
        let _ = store
            .store_with_dedup(
                memory,
                config.search.dedup_similarity as f32,
                config.search.dedup_time_window_days,
            );
    }
    Ok(())
}

/// Layer 1: PreCompact -- extract memories before context compression.
/// Same as hook_post but reads transcript (potentially longer text) with lower threshold.
pub async fn hook_compact(config: &ReinConfig) -> anyhow::Result<()> {
    let input = std::io::read_to_string(std::io::stdin())?;
    let facts = crate::extract::patterns::extract_facts(&input, 2);

    if facts.is_empty() {
        return Ok(());
    }

    let store = config.open_store()?;

    for fact in facts {
        let importance = crate::types::Importance::Medium;
        let memory = crate::types::Memory {
            id: ulid::Ulid::new().to_string(),
            layer: importance.auto_layer(),
            topic: "auto-extracted".to_string(),
            summary: fact.chars().take(100).collect(),
            content: fact,
            keywords: vec![],
            importance,
            source: crate::types::Source::Hook,
            strength: 1.0,
            decay_lambda: config.decay.base_lambda * importance.decay_factor(),
            access_count: 0,
            superseded_by: None,
            related_ids: vec![],
            embedding: None,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            last_accessed: chrono::Utc::now(),
        };
        let _ = store
            .store_with_dedup(
                memory,
                config.search.dedup_similarity as f32,
                config.search.dedup_time_window_days,
            );
    }
    Ok(())
}

/// Layer 2: UserPromptSubmit -- inject recalled memories into context.
/// Reads user prompt from stdin, recalls relevant memories, outputs context block to stdout.
pub async fn hook_prompt(config: &ReinConfig) -> anyhow::Result<()> {
    let query = std::io::read_to_string(std::io::stdin())?;
    let query = query.trim();
    if query.is_empty() {
        return Ok(());
    }

    let store = config.open_store()?;
    let results = store.search_fts(query, None, 5)?;

    if results.is_empty() {
        return Ok(());
    }

    // Output as supermemory-context block
    println!("<supermemory-context>");
    println!("The following is recalled context from rein memory.");
    println!();
    for memory in &results {
        println!("## {}", memory.summary);
        println!("{}", memory.content);
        println!();
    }
    println!("</supermemory-context>");
    Ok(())
}
