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
        tier: "warm".to_string(),
        cluster_id: None,
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
        tier: "warm".to_string(),
        cluster_id: None,
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
        // Preview mode: DELETE within savepoint so concept evaluation sees the
        // same DB state as a real GC. Side indexes (Tantivy/HNSW) are NOT touched.
        // ROLLBACK undoes the SQLite DELETE at the end.
        store.conn().execute_batch("SAVEPOINT gc_preview")
            .map_err(crate::types::ReinError::Database)?;

        let decayed = store.apply_decay()?;
        // SQL DELETE only (no Tantivy/HNSW removal) — savepoint will rollback
        let mem_pruned = store.prune_memories_sql_only(threshold)?;
        let concept_pruned = store.prune_low_quality_concepts().unwrap_or(0);

        store.conn().execute_batch("ROLLBACK TO gc_preview")
            .map_err(crate::types::ReinError::Database)?;
        store.conn().execute_batch("RELEASE gc_preview")
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
pub fn run_gc_adaptive(store: &SqliteStore, config: &ReinConfig, threshold: f64, dry_run: bool) -> ReinResult<(u64, u64, u64)> {
    let result = run_gc(store, threshold, dry_run)?;
    if !dry_run {
        run_adaptive_pipeline(store, config);
    }
    Ok(result)
}

/// Run the adaptive engine slow-channel pipeline after GC.
/// Order: M4 (HDBSCAN) → M3 (Survival) → M5 (Tiering) → M2 (Alpha) → persist.
/// Each step is gated by readiness checks; failures skip subsequent steps.
pub fn run_adaptive_pipeline(store: &SqliteStore, config: &ReinConfig) {
    if !config.adaptive.enabled {
        return;
    }

    let _span = tracing::info_span!("adaptive_pipeline").entered();

    // Restore or create AdaptiveState
    let mut state = crate::store::adaptive::AdaptiveState::restore_snapshot(store.conn())
        .unwrap_or_default();

    // Count memories for readiness checks
    let mem_count: u64 = store.conn()
        .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
        .unwrap_or(0);

    // Step 1: M4 — HDBSCAN clustering (skip if < 50 memories with embeddings)
    let embeddings_count: u64 = store.conn()
        .query_row("SELECT COUNT(*) FROM vec_memories", [], |r| r.get(0))
        .unwrap_or(0);

    if embeddings_count >= 50 {
        tracing::debug!("M4: running HDBSCAN on {} embeddings", embeddings_count);
        // Get all embeddings from vec_memories
        // Note: full HDBSCAN on embeddings is computationally expensive for large sets.
        // For now, we just increment cluster_version and log.
        // Full HDBSCAN integration requires reading embeddings from vec table which
        // needs sqlite-vec specific queries — deferred to integration phase.
        state.cluster_version += 1;
        tracing::debug!("M4: cluster_version incremented to {}", state.cluster_version);
    }

    // Step 1b: A1 — Compute per-cluster dedup thresholds from intra-cluster similarity distribution
    if !state.memory_clusters.is_empty() {
        compute_per_cluster_dedup_thresholds(store, &mut state);
    }

    // Step 2: M3 — Survival curves (per-cluster, needs cluster assignments)
    // Skipped until M4 produces actual cluster assignments.
    // The Kaplan-Meier module is ready but needs access_time data per cluster.

    // Step 3: M5 — Tier migration (needs >= tier_cold_start memories)
    if mem_count >= config.adaptive.tier_cold_start as u64 {
        tracing::debug!("M5: computing tier boundaries for {} memories", mem_count);
        let mut boundaries = crate::store::tiering::TierBoundaries::new();

        // Compute access rates for all memories
        if let Ok(mut stmt) = store.conn().prepare(
            "SELECT access_count, created_at FROM memories WHERE status = 'active'"
        ) {
            let rates: Vec<f64> = stmt.query_map([], |row| {
                let ac: u32 = row.get(0)?;
                let created_str: String = row.get(1)?;
                let created = chrono::DateTime::parse_from_rfc3339(&created_str)
                    .map(|dt| dt.with_timezone(&chrono::Utc))
                    .unwrap_or_else(|_| chrono::Utc::now());
                Ok(crate::store::tiering::compute_access_rate(ac, created))
            }).ok().map(|rows| rows.filter_map(|r| r.ok()).collect()).unwrap_or_default();

            if !rates.is_empty() {
                boundaries.update(&rates);
                state.hot_threshold = boundaries.hot_threshold;
                state.cold_threshold = boundaries.cold_threshold;
                tracing::debug!("M5: hot={:.4}, cold={:.4}", boundaries.hot_threshold, boundaries.cold_threshold);
            }
        }
    }

    // Step 4: M2 — Counterfactual alpha optimization
    // NOTE: Do NOT consume_events here — that would advance offsets and lose data
    // before the learner is fully wired. Only peek at event count for diagnostics.
    let recall_event_count: u64 = store.conn()
        .query_row(
            "SELECT COUNT(*) FROM feedback_events WHERE event_type = 'recall_complete'",
            [], |r| r.get(0),
        )
        .unwrap_or(0);
    if recall_event_count > 0 {
        tracing::debug!("M2: {} recall_complete events available (alpha learning pending full integration)", recall_event_count);
    }

    // Step 5: Embedding-based dedup for memories marked needs_vec_dedup
    run_vec_dedup(store, config);

    // Step 6: Persist snapshot + emit param_update event
    state.version += 1;
    if let Err(e) = state.save_snapshot(store.conn()) {
        tracing::warn!("failed to save adaptive state: {e}");
    } else {
        tracing::debug!("adaptive state v{} saved", state.version);
    }

    // Step 7: Cleanup expired events
    let cleaned = crate::store::adaptive::cleanup_expired_events(
        store.conn(),
        config.adaptive.event_retention_days,
    ).unwrap_or(0);
    if cleaned > 0 {
        tracing::debug!("cleaned {cleaned} expired events");
    }
}

/// Compute per-cluster dedup thresholds from intra-cluster content similarity.
/// For each cluster with >= 5 members, compute pairwise Jaccard/Containment similarity
/// and use P90 as that cluster's dedup threshold (SemDeDup-inspired approach).
fn compute_per_cluster_dedup_thresholds(
    store: &SqliteStore,
    state: &mut crate::store::adaptive::AdaptiveState,
) {
    use std::collections::HashMap;

    // Group memory IDs by cluster
    let mut clusters: HashMap<u32, Vec<String>> = HashMap::new();
    for (mem_id, &cluster_id) in &state.memory_clusters {
        clusters.entry(cluster_id).or_default().push(mem_id.clone());
    }

    let mut all_sims: Vec<f32> = Vec::new();

    for (cluster_id, mem_ids) in &clusters {
        if mem_ids.len() < 5 { continue; }

        // Sample up to 20 members to keep computation bounded
        let sample: Vec<&str> = mem_ids.iter().take(20).map(|s| s.as_str()).collect();
        let mut sims: Vec<f32> = Vec::new();

        // Fetch content for sampled members
        let contents: Vec<String> = sample.iter()
            .filter_map(|id| store.get(id).ok().map(|m| m.content))
            .collect();

        // Compute pairwise similarities
        for i in 0..contents.len() {
            for j in (i + 1)..contents.len() {
                let sim = crate::extract::similarity(&contents[i], &contents[j]);
                sims.push(sim);
                all_sims.push(sim);
            }
        }

        if sims.len() >= 3 {
            sims.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            // P90: the threshold where 90% of intra-cluster pairs are below
            let p90_idx = (sims.len() as f64 * 0.90).floor() as usize;
            let p90_idx = p90_idx.min(sims.len() - 1);
            let threshold = sims[p90_idx].max(0.40).min(0.90); // Clamp to sane range
            state.dedup_thresholds.insert(*cluster_id, threshold);
            tracing::debug!("A1: cluster {cluster_id} dedup threshold = {threshold:.3} (from {} pairs)", sims.len());
        }
    }

    // Update global threshold from all-clusters distribution
    if all_sims.len() >= 10 {
        all_sims.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p90_idx = (all_sims.len() as f64 * 0.90).floor() as usize;
        let p90_idx = p90_idx.min(all_sims.len() - 1);
        let global = all_sims[p90_idx].max(0.40).min(0.90);
        state.global_dedup_threshold = global;
        tracing::debug!("A1: global dedup threshold = {global:.3} (from {} total pairs)", all_sims.len());
    }
}

/// Embedding-based dedup sweep for memories marked `needs_vec_dedup`.
/// Computes embeddings (if missing), searches vec_memories for near-duplicates,
/// and merges/supersedes matches. Runs in the GC slow channel (zero hot-path cost).
fn run_vec_dedup(store: &SqliteStore, config: &ReinConfig) {
    let conn = store.conn();

    // Fetch memories needing vec dedup (batch limit to avoid holding resources too long)
    let pending: Vec<(String, String, String, String)> = match conn.prepare(
        "SELECT id, topic, summary, content FROM memories
         WHERE needs_vec_dedup = 1 AND status = 'active' AND superseded_by IS NULL
         LIMIT 50"
    ) {
        Ok(mut stmt) => stmt.query_map([], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
        }).ok().map(|rows| rows.filter_map(|r| r.ok()).collect()).unwrap_or_default(),
        Err(_) => return,
    };

    if pending.is_empty() {
        return;
    }

    tracing::debug!("vec_dedup: processing {} memories", pending.len());

    // Create embedder (needed for computing embeddings of new memories)
    let embedder = match crate::embed::create_embedder(config) {
        Some(e) => e,
        None => {
            tracing::debug!("vec_dedup: no embedder configured, skipping (flags preserved for later)");
            return;
        }
    };

    let model_name = config.embedding_model();
    let mut merged = 0u32;

    for (id, topic, summary, content) in &pending {
        // Step 1: Get or compute embedding for this memory
        let enriched = crate::embed::prepend_metadata(topic, summary, content);
        let embedding = match crate::embed::EmbedCache::get(conn, &enriched, &model_name) {
            Ok(Some(cached)) => {
                // Ensure this memory is also in vec_memories (cache may exist from warmup
                // without a corresponding vec_memories row)
                let _ = crate::store::vec::insert_embedding(conn, id, &cached);
                cached
            }
            _ => {
                // Compute embedding (async → sync bridge)
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(async {
                            use crate::types::traits::Embedder;
                            embedder.embed(&enriched).await
                        })
                    })
                })) {
                    Ok(Ok(emb)) => {
                        // Cache + store in vec_memories
                        let _ = crate::embed::EmbedCache::put(conn, &enriched, &model_name, &emb);
                        let _ = crate::store::vec::insert_embedding(conn, id, &emb);
                        emb
                    }
                    _ => {
                        tracing::debug!("vec_dedup: failed to compute embedding for {id}");
                        let _ = conn.execute(
                            "UPDATE memories SET needs_vec_dedup = 0 WHERE id = ?1",
                            rusqlite::params![id],
                        );
                        continue;
                    }
                }
            }
        };

        // Step 2: Search vec_memories for near-duplicates (excluding self)
        let vec_results = match crate::store::vec::search_vec(conn, &embedding, 10) {
            Ok(r) => r,
            Err(_) => {
                let _ = conn.execute(
                    "UPDATE memories SET needs_vec_dedup = 0 WHERE id = ?1",
                    rusqlite::params![id],
                );
                continue;
            }
        };

        let mut found_dup = false;
        for (candidate_id, distance) in &vec_results {
            if candidate_id == id { continue; }

            // cosine distance → similarity
            let sim = 1.0 - (*distance as f64);
            if sim < 0.70 { break; } // Results are sorted by distance; no point continuing

            let candidate = match store.get(candidate_id) {
                Ok(m) => m,
                Err(_) => continue,
            };
            if candidate.superseded_by.is_some() || candidate.status != crate::types::MemoryStatus::Active {
                continue;
            }

            if sim > 0.85 {
                // Strong semantic match — merge into the older (more established) memory
                let (keep_id, discard_id, merge_content) = if candidate.access_count >= 1 || candidate.created_at < chrono::Utc::now() - chrono::Duration::hours(1) {
                    (&candidate.id, id, content.as_str())
                } else {
                    (id, candidate_id, candidate.content.as_str())
                };

                // Use store.get + store.update to properly trigger FTS/Tantivy/HNSW refresh
                if let Ok(mut kept) = store.get(keep_id) {
                    kept.content = format!("{}\n\n{}", kept.content, merge_content);
                    kept.summary = kept.content.chars().take(100).collect();
                    kept.strength = (kept.strength + 0.2).min(1.0);
                    kept.updated_at = chrono::Utc::now();
                    let _ = store.update(&kept);
                }
                let _ = store.mark_superseded(discard_id, keep_id);

                tracing::info!("vec_dedup: merged {discard_id} into {keep_id} (cosine_sim={sim:.3})");
                merged += 1;
                found_dup = true;
                break;
            } else if sim > 0.70 {
                // Moderate match — use LLM to confirm (if available)
                let is_dup = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(async {
                            crate::extract::llm::llm_is_duplicate(config, content, &candidate.content).await
                        })
                    })
                })).unwrap_or(false);

                if is_dup {
                    let _ = conn.execute(
                        "UPDATE memories SET superseded_by = ?1 WHERE id = ?2",
                        rusqlite::params![&candidate.id, id],
                    );
                    tracing::info!("vec_dedup: LLM confirmed dup, superseded {id} by {} (cosine_sim={sim:.3})", candidate.id);
                    merged += 1;
                    found_dup = true;
                    break;
                }
            }
        }

        // Clear the flag whether or not we found a dup
        let _ = conn.execute(
            "UPDATE memories SET needs_vec_dedup = 0 WHERE id = ?1",
            rusqlite::params![id],
        );
        // If this memory was merged away, also clear the other's flag
        if found_dup {
            // Already handled above via superseded_by
        }
    }

    if merged > 0 {
        tracing::info!("vec_dedup: merged {merged} semantic duplicates");
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
