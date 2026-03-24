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
/// Reads user prompt from stdin, uses full recall pipeline, outputs context block to stdout.
pub async fn hook_prompt(config: &ReinConfig) -> anyhow::Result<()> {
    let query = std::io::read_to_string(std::io::stdin())?;
    let query = query.trim();
    if query.is_empty() {
        return Ok(());
    }

    let store = config.open_store()?;
    let results = crate::search::recall::recall(&store, config, query, None, None, 5)?;

    if results.is_empty() {
        return Ok(());
    }

    // Output as rein-context block (compatible with supermemory-context)
    println!("<rein-context>");
    println!("Recalled from rein memory (confidence shown):");
    println!();
    for r in &results {
        let conf = if r.sources_hit >= 3 {
            "HIGH"
        } else if r.sources_hit >= 2 {
            "MED"
        } else {
            "LOW"
        };
        println!("## [{}] {}", conf, r.memory.summary);
        println!("{}", r.memory.content);
        println!();
    }
    println!("</rein-context>");
    Ok(())
}

/// Layer 3: Stop -- extract session summary and save to memory on conversation end.
/// Reads conversation transcript from stdin, extracts key facts and decisions, stores them.
pub async fn hook_stop(config: &ReinConfig) -> anyhow::Result<()> {
    let input = std::io::read_to_string(std::io::stdin())?;
    if input.trim().is_empty() {
        return Ok(());
    }

    // Extract important facts with low threshold to capture more from session end
    let facts = crate::extract::patterns::extract_facts(&input, 2);

    // Also extract any lines that look like decisions or outcomes
    let decision_keywords = ["decided", "chose", "will use", "switched to", "completed",
        "deployed", "installed", "configured", "created", "fixed", "resolved",
        "stored", "migrated", "released", "published", "committed"];
    let mut decisions: Vec<String> = Vec::new();
    for line in input.lines() {
        let lower = line.to_lowercase();
        if decision_keywords.iter().any(|kw| lower.contains(kw)) && line.len() > 20 && line.len() < 500 {
            // Dedup against already-extracted facts
            let is_dup = facts.iter().any(|f| crate::extract::similarity(f, line) > 0.6);
            if !is_dup {
                decisions.push(line.trim().to_string());
            }
        }
    }

    let all_items: Vec<String> = facts.into_iter().chain(decisions.into_iter()).collect();
    if all_items.is_empty() {
        return Ok(());
    }

    let store = config.open_store()?;
    let mut stored = 0;

    for item in all_items.iter().take(10) {
        // Cap at 10 items per session
        let importance = crate::types::Importance::Medium;
        let memory = crate::types::Memory {
            id: String::new(), // will be generated by store()
            layer: importance.auto_layer(),
            topic: "session-summary".to_string(),
            summary: item.chars().take(100).collect(),
            content: item.clone(),
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
        if store.store_with_dedup(memory, config.search.dedup_similarity as f32, config.search.dedup_time_window_days).is_ok() {
            stored += 1;
        }
    }

    if stored > 0 {
        eprintln!("rein: saved {stored} memories from session");
    }
    Ok(())
}
