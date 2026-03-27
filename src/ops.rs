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
        run_hdbscan_clustering(store, &mut state, embeddings_count as usize);
    }

    // Step 1b: A1 — Compute per-cluster dedup thresholds
    if !state.memory_clusters.is_empty() {
        compute_per_cluster_dedup_thresholds(store, &mut state);
    }

    // Step 2: M3 — Build per-cluster survival curves from access data
    if !state.memory_clusters.is_empty() {
        build_survival_curves(store, &state);
    }

    // Step 3: M5 — Tier boundaries + cold_archive migration
    if mem_count >= config.adaptive.tier_cold_start as u64 {
        run_tiering(store, &mut state, config);
    }

    // Step 4: M2 — Counterfactual alpha optimization (consume events, learn alphas)
    run_alpha_learning(store, &mut state, config);

    // Step 4b: M6 — Consume threshold exploration data + co-recall signal → update dedup thresholds
    run_m6_threshold_learning(store, &mut state);

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

// ===========================================================================
// M4: HDBSCAN clustering — read embeddings, cluster, store assignments
// ===========================================================================

fn run_hdbscan_clustering(
    store: &SqliteStore,
    state: &mut crate::store::adaptive::AdaptiveState,
    count: usize,
) {
    tracing::debug!("M4: running HDBSCAN on {count} embeddings");

    // Read all embeddings — hdbscan() internally handles sampling for n > 3000
    // Cap at 10000 to avoid excessive memory use even with sampling
    let load_limit = count.min(10_000);
    let embeddings: Vec<(String, Vec<f32>)> = match store.conn().prepare(
        "SELECT id, embedding FROM vec_memories LIMIT ?1"
    ) {
        Ok(mut stmt) => stmt.query_map(
            rusqlite::params![load_limit as i64],
            |row| {
                let id: String = row.get(0)?;
                let blob: Vec<u8> = row.get(1)?;
                let floats: Vec<f32> = blob.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                Ok((id, floats))
            },
        ).ok().map(|rows| rows.filter_map(|r| r.ok()).collect()).unwrap_or_default(),
        Err(_) => return,
    };

    if embeddings.len() < 50 { return; }

    let min_cluster_size = 5.max(embeddings.len() / 50); // adaptive: ~2% of dataset
    let result = crate::store::hdbscan::hdbscan(&embeddings, min_cluster_size);

    // Clear stale cluster assignments ONLY for memories that were part of this clustering input.
    // Memories outside the input set keep their existing cluster_id (from a previous run).
    let clustered_ids: std::collections::HashSet<&str> = embeddings.iter()
        .map(|(id, _)| id.as_str()).collect();
    for id in &clustered_ids {
        let _ = store.conn().execute(
            "UPDATE memories SET cluster_id = NULL WHERE id = ?1",
            rusqlite::params![id],
        );
    }
    // Clear stale per-cluster survival curves (will be rebuilt by M3)
    let _ = store.conn().execute("DELETE FROM metadata WHERE key LIKE 'survival_curve:%'", []);
    // Clear stale per-cluster dedup thresholds
    state.dedup_thresholds.clear();

    // Store new cluster assignments from this run
    state.memory_clusters.clear();
    for (i, label) in result.labels.iter().enumerate() {
        let mem_id = &embeddings[i].0;
        if let Some(cluster_id) = label {
            state.memory_clusters.insert(mem_id.clone(), *cluster_id);
            let _ = store.conn().execute(
                "UPDATE memories SET cluster_id = ?1 WHERE id = ?2",
                rusqlite::params![*cluster_id, mem_id],
            );
        }
    }

    // Load persisted cluster assignments for memories NOT in this clustering input
    // (keeps DB and in-memory state synchronized for capped/sampled runs)
    if let Ok(mut stmt) = store.conn().prepare(
        "SELECT id, cluster_id FROM memories WHERE cluster_id IS NOT NULL"
    ) {
        let _ = stmt.query_map([], |row| {
            let id: String = row.get(0)?;
            let cid: u32 = row.get(1)?;
            Ok((id, cid))
        }).ok().map(|rows| {
            for row in rows.flatten() {
                state.memory_clusters.entry(row.0).or_insert(row.1);
            }
        });
    }

    state.cluster_version += 1;
    tracing::info!(
        "M4: {} clusters, {} noise points, {} assigned (v{})",
        result.clusters.len(), result.noise_indices.len(),
        state.memory_clusters.len(), state.cluster_version,
    );
}

// ===========================================================================
// M3: Build per-cluster survival curves from access timestamps
// ===========================================================================

fn build_survival_curves(
    store: &SqliteStore,
    state: &crate::store::adaptive::AdaptiveState,
) {
    use std::collections::HashMap;

    // Group memories by cluster, collect access timestamps
    let mut cluster_intervals: HashMap<u32, Vec<crate::search::survival::SurvivalInterval>> = HashMap::new();
    let now = chrono::Utc::now();

    for (mem_id, &cluster_id) in &state.memory_clusters {
        // Get created_at, last_accessed, access_count for this memory
        let row: Option<(String, String, u32)> = store.conn().query_row(
            "SELECT created_at, last_accessed, access_count FROM memories WHERE id = ?1",
            rusqlite::params![mem_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        ).ok();

        let (created_str, last_str, access_count) = match row {
            Some(r) => r,
            None => continue,
        };

        let created = chrono::DateTime::parse_from_rfc3339(&created_str)
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .unwrap_or(now);
        let last_accessed = chrono::DateTime::parse_from_rfc3339(&last_str)
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .unwrap_or(now);

        let intervals = cluster_intervals.entry(cluster_id).or_default();

        if access_count > 0 {
            // Approximate access intervals: spread access_count evenly between created_at and last_accessed
            let total_days = (last_accessed - created).num_seconds() as f64 / 86400.0;
            if total_days > 0.0 && access_count > 1 {
                let interval = total_days / access_count as f64;
                for _ in 0..access_count.min(20) {
                    intervals.push(crate::search::survival::SurvivalInterval {
                        duration_days: interval,
                        is_event: true,
                    });
                }
            }
            // Censored: time since last access
            let censored = (now - last_accessed).num_seconds() as f64 / 86400.0;
            intervals.push(crate::search::survival::SurvivalInterval {
                duration_days: censored.max(0.0),
                is_event: false,
            });
        } else {
            // Never accessed after creation — single censored observation
            let age = (now - created).num_seconds() as f64 / 86400.0;
            intervals.push(crate::search::survival::SurvivalInterval {
                duration_days: age.max(0.0),
                is_event: false,
            });
        }
    }

    // Build curves and store as metadata (one per cluster)
    let mut curves_built = 0u32;
    for (cluster_id, intervals) in &cluster_intervals {
        if intervals.len() < 10 { continue; } // Need minimum data for meaningful curve

        if let Some(curve) = crate::search::survival::kaplan_meier(intervals) {
            // Store curve as JSON in metadata table for scoring to pick up
            let key = format!("survival_curve:{cluster_id}");
            if let Ok(json) = serde_json::to_string(&curve) {
                let _ = store.conn().execute(
                    "INSERT INTO metadata (key, value) VALUES (?1, ?2)
                     ON CONFLICT(key) DO UPDATE SET value = ?2",
                    rusqlite::params![key, json],
                );
                curves_built += 1;
            }
        }
    }

    if curves_built > 0 {
        tracing::info!("M3: built {curves_built} per-cluster survival curves");
    }
}

// ===========================================================================
// M5: Tier boundaries + cold_archive migration
// ===========================================================================

fn run_tiering(
    store: &SqliteStore,
    state: &mut crate::store::adaptive::AdaptiveState,
    _config: &ReinConfig,
) {
    tracing::debug!("M5: computing tier boundaries");
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
        }
    }

    // Update tier labels on memories
    if state.hot_threshold > 0.0 && state.cold_threshold > 0.0 {
        let _ = store.conn().execute(
            "UPDATE memories SET tier = 'hot'
             WHERE status = 'active' AND tier != 'hot'
             AND CAST(access_count AS REAL) / MAX(1, CAST(
               (julianday('now') - julianday(created_at)) AS REAL)) >= ?1",
            rusqlite::params![state.hot_threshold],
        );
        let _ = store.conn().execute(
            "UPDATE memories SET tier = 'cold'
             WHERE status = 'active' AND tier != 'cold'
             AND CAST(access_count AS REAL) / MAX(1, CAST(
               (julianday('now') - julianday(created_at)) AS REAL)) <= ?1",
            rusqlite::params![state.cold_threshold],
        );
        let _ = store.conn().execute(
            "UPDATE memories SET tier = 'warm'
             WHERE status = 'active' AND (
               tier NOT IN ('hot', 'cold')
               OR (tier = 'hot' AND CAST(access_count AS REAL) / MAX(1, CAST(
                 (julianday('now') - julianday(created_at)) AS REAL)) < ?1)
               OR (tier = 'cold' AND CAST(access_count AS REAL) / MAX(1, CAST(
                 (julianday('now') - julianday(created_at)) AS REAL)) > ?2)
             )",
            rusqlite::params![state.hot_threshold, state.cold_threshold],
        );
    }

    // Migrate cold memories to cold_archive (content → summary, original in archive)
    let migrated: u64 = match store.conn().execute(
        "INSERT OR IGNORE INTO cold_archive (memory_id, content, archived_at)
         SELECT id, content, strftime('%Y-%m-%dT%H:%M:%fZ','now')
         FROM memories
         WHERE tier = 'cold' AND strength < 0.3 AND access_count = 0
         AND id NOT IN (SELECT memory_id FROM cold_archive)",
        [],
    ) {
        Ok(n) => n as u64,
        Err(_) => 0,
    };

    // Strip archived memories to summary-only via store.update() to keep Tantivy in sync
    if migrated > 0 {
        // Fetch archived memory IDs and update through the proper API
        let archived_ids: Vec<String> = store.conn().prepare(
            "SELECT memory_id FROM cold_archive WHERE memory_id IN (
                SELECT id FROM memories WHERE tier = 'cold' AND strength < 0.3 AND access_count = 0
            )"
        ).ok().and_then(|mut stmt| {
            stmt.query_map([], |row| row.get(0)).ok()
                .map(|rows| rows.filter_map(|r| r.ok()).collect())
        }).unwrap_or_default();

        for aid in &archived_ids {
            if let Ok(mut mem) = store.get(aid) {
                mem.content = mem.summary.clone();
                let _ = store.update(&mem); // Triggers Tantivy + FTS update
            }
        }
        tracing::info!("M5: migrated {migrated} cold memories to archive ({} stripped), hot={:.4} cold={:.4}",
            archived_ids.len(), state.hot_threshold, state.cold_threshold);
    } else {
        tracing::debug!("M5: hot={:.4}, cold={:.4}, no migrations needed",
            state.hot_threshold, state.cold_threshold);
    }
}

// ===========================================================================
// M2: Counterfactual alpha optimization — consume events, learn alphas
// ===========================================================================

fn run_alpha_learning(
    store: &SqliteStore,
    state: &mut crate::store::adaptive::AdaptiveState,
    config: &ReinConfig,
) {
    let conn = store.conn();

    // Peek at recall_complete events WITHOUT consuming (don't advance offset yet).
    // We only advance the offset for events that successfully match access data.
    let last_offset: i64 = conn.query_row(
        "SELECT last_event_id FROM consumer_offsets WHERE consumer = 'alpha_optimizer'",
        [], |r| r.get(0),
    ).unwrap_or(0);

    let events: Vec<crate::store::adaptive::StoredEvent> = match conn.prepare(
        "SELECT id, ts, event_type, request_id, memory_id, concept_id, query, query_type, topic, payload
         FROM feedback_events WHERE id > ?1 AND event_type = 'recall_complete'
         ORDER BY id ASC LIMIT 100"
    ) {
        Ok(mut stmt) => stmt.query_map(rusqlite::params![last_offset], |row| {
            Ok(crate::store::adaptive::StoredEvent {
                id: row.get(0)?, ts: row.get(1)?, event_type: row.get(2)?,
                request_id: row.get(3)?, memory_id: row.get(4)?, concept_id: row.get(5)?,
                query: row.get(6)?, query_type: row.get(7)?, topic: row.get(8)?,
                payload: row.get(9)?,
            })
        }).ok().map(|rows| rows.filter_map(|r| r.ok()).collect()).unwrap_or_default(),
        Err(_) => return,
    };

    if events.is_empty() { return; }

    // Consume recall_access events (these are fire-and-forget, safe to advance)
    let access_events = crate::store::adaptive::consume_events(
        conn, "alpha_optimizer_access", &["recall_access"], 500,
    ).unwrap_or_default();

    // Build RecallEvent structs from stored events
    let mut recall_events: Vec<crate::search::alpha_optimizer::RecallEvent> = Vec::new();

    for event in &events {
        let request_id = match &event.request_id {
            Some(r) => r.clone(),
            None => continue,
        };

        // Parse payload for candidate logs
        let payload = match &event.payload {
            Some(p) => p,
            None => continue,
        };
        // Payload is {"candidates": [...], "alpha_used": ..., ...}
        let payload_obj: serde_json::Value = serde_json::from_str(payload).unwrap_or_default();
        let candidates_json: Vec<serde_json::Value> = payload_obj.get("candidates")
            .and_then(|c| c.as_array())
            .cloned()
            .unwrap_or_default();
        if candidates_json.is_empty() { continue; }

        let candidates: Vec<crate::search::alpha_optimizer::CandidateLog> = candidates_json.iter()
            .filter_map(|c| {
                Some(crate::search::alpha_optimizer::CandidateLog {
                    memory_id: c.get("id")?.as_str()?.to_string(),
                    bm25_norm: c.get("bm25_norm")?.as_f64()? as f32,
                    vec_norm: c.get("vec_norm")?.as_f64()? as f32,
                })
            })
            .collect();

        if candidates.is_empty() { continue; }

        let ts = chrono::DateTime::parse_from_rfc3339(&event.ts)
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .unwrap_or_else(|_| chrono::Utc::now());

        // Find which candidate memories were actually accessed (injected by hook_prompt).
        // Match by: (a) memory_id appears in this recall's candidate set, AND
        // (b) access event timestamp is within 10 minutes of the recall event.
        // The time window reduces false attribution when the same memory appears
        // in multiple unrelated recalls.
        let candidate_ids: std::collections::HashSet<&str> = candidates.iter()
            .map(|c| c.memory_id.as_str()).collect();
        let accessed_ids: Vec<String> = access_events.iter()
            .filter(|a| {
                // Time-window filter: access must be within 10 min of recall
                let access_ts = chrono::DateTime::parse_from_rfc3339(&a.ts)
                    .map(|dt| dt.with_timezone(&chrono::Utc))
                    .unwrap_or(ts);
                let diff = (access_ts - ts).num_seconds().abs();
                diff < 600 // 10 minutes
            })
            .filter_map(|a| a.memory_id.as_deref())
            .filter(|mid| candidate_ids.contains(mid))
            .map(|s| s.to_string())
            .collect::<std::collections::HashSet<_>>()
            .into_iter().collect();

        recall_events.push(crate::search::alpha_optimizer::RecallEvent {
            request_id,
            candidates,
            accessed_ids,
            timestamp: ts,
        });
    }

    // Only learn from events that have actual access data
    let events_with_access: Vec<_> = recall_events.iter()
        .filter(|e| !e.accessed_ids.is_empty())
        .cloned()
        .collect();

    // Advance offset through contiguous prefix of matched or expired events.
    // Stop at the first live unmatched event (its access signal may arrive later).
    // 24h expiry prevents a single stale event from permanently blocking the pipeline.
    let cutoff = chrono::Utc::now() - chrono::Duration::hours(24);
    let matched_request_ids: std::collections::HashSet<&str> = recall_events.iter()
        .filter(|re| !re.accessed_ids.is_empty())
        .map(|re| re.request_id.as_str())
        .collect();

    let mut advance_to: Option<i64> = None;
    for event in &events {
        let rid = event.request_id.as_deref().unwrap_or("");
        let is_matched = matched_request_ids.contains(rid);
        let is_expired = chrono::DateTime::parse_from_rfc3339(&event.ts)
            .map(|dt| dt.with_timezone(&chrono::Utc) < cutoff)
            .unwrap_or(false);

        if is_matched || is_expired {
            advance_to = Some(event.id);
        } else {
            // Live unmatched event — stop here, retry next cycle
            break;
        }
    }

    if let Some(new_offset) = advance_to {
        let _ = conn.execute(
            "INSERT INTO consumer_offsets (consumer, last_event_id, updated_at)
             VALUES ('alpha_optimizer', ?1, strftime('%Y-%m-%dT%H:%M:%fZ','now'))
             ON CONFLICT(consumer) DO UPDATE SET last_event_id = ?1, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')",
            rusqlite::params![new_offset],
        );
    }

    if events_with_access.is_empty() {
        tracing::debug!("M2: peeked {} events but none had access data yet (will retry)", events.len());
        return;
    }

    // Compute global alpha
    let decay_lambda = 0.06; // ~11 day half-life for event weighting
    if let Some(learned) = crate::search::alpha_optimizer::optimize_alpha(&events_with_access, decay_lambda) {
        let key = "global".to_string();
        let current = state.learned_alpha.get(&key).map(|e| e.value).unwrap_or(0.5);
        let stepped = crate::search::alpha_optimizer::apply_max_step(
            current, learned.value, config.adaptive.alpha_max_step,
        );
        let shrunk = crate::search::alpha_optimizer::bayesian_shrinkage(
            stepped, 0.5, learned.sample_count, config.adaptive.shrinkage_prior,
        );

        state.learned_alpha.insert(key, crate::store::adaptive::LearnedAlphaEntry {
            value: shrunk,
            sample_count: learned.sample_count,
            last_updated: chrono::Utc::now().to_rfc3339(),
        });

        tracing::info!(
            "M2: learned global alpha = {shrunk:.3} (from {} events, raw={:.3})",
            learned.sample_count, learned.value
        );
    }

    // Per-query-type alphas
    for qt in &["Temporal", "ExactKeyword", "Semantic", "Exploratory"] {
        let qt_events: Vec<_> = events_with_access.iter()
            .filter(|e| {
                // Match by looking at event's query_type in the original stored events
                events.iter().any(|se| {
                    se.request_id.as_deref() == Some(&e.request_id)
                    && se.query_type.as_deref() == Some(qt)
                })
            })
            .cloned()
            .collect();

        if qt_events.len() < config.adaptive.min_samples_alpha { continue; }

        if let Some(learned) = crate::search::alpha_optimizer::optimize_alpha(&qt_events, decay_lambda) {
            let global_alpha = state.learned_alpha.get("global")
                .map(|e| e.value).unwrap_or(0.5);
            let shrunk = crate::search::alpha_optimizer::bayesian_shrinkage(
                learned.value, global_alpha, learned.sample_count, config.adaptive.shrinkage_prior,
            );

            state.learned_alpha.insert(qt.to_string(), crate::store::adaptive::LearnedAlphaEntry {
                value: shrunk,
                sample_count: learned.sample_count,
                last_updated: chrono::Utc::now().to_rfc3339(),
            });

            tracing::info!("M2: learned {qt} alpha = {shrunk:.3} ({} events)", learned.sample_count);
        }
    }
}

// ===========================================================================
// M6: Threshold learning — consume exploration data + co-recall signal
// ===========================================================================

fn run_m6_threshold_learning(
    store: &SqliteStore,
    state: &mut crate::store::adaptive::AdaptiveState,
) {
    let conn = store.conn();

    // --- Part 1: Consume threshold_exploration events (from randomized A/B test) ---
    let events = crate::store::adaptive::consume_events(
        conn, "m6_threshold", &["param_update"], 200,
    ).unwrap_or_default();

    // Filter to threshold_exploration events only
    let explore_events: Vec<_> = events.iter()
        .filter(|e| e.query_type.as_deref() == Some("threshold_exploration"))
        .collect();

    if explore_events.len() >= 10 {
        // Causal inference: compare dedup rates at different thresholds
        // Group by whether threshold was raised or lowered
        let mut raised_dedup = 0u32;  // threshold raised (harder to dedup) → was_dedup count
        let mut raised_total = 0u32;
        let mut lowered_dedup = 0u32; // threshold lowered (easier to dedup) → was_dedup count
        let mut lowered_total = 0u32;

        for event in &explore_events {
            let payload: serde_json::Value = event.payload.as_deref()
                .and_then(|p| serde_json::from_str(p).ok())
                .unwrap_or_default();

            let offset = payload.get("offset").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let was_dedup = payload.get("was_dedup").and_then(|v| v.as_bool()).unwrap_or(false);

            if offset > 0.01 {
                raised_total += 1;
                if was_dedup { raised_dedup += 1; }
            } else if offset < -0.01 {
                lowered_total += 1;
                if was_dedup { lowered_dedup += 1; }
            }
        }

        // If lowering the threshold catches significantly more duplicates,
        // the current threshold is too high → nudge global threshold down
        if raised_total >= 5 && lowered_total >= 5 {
            let raised_rate = raised_dedup as f64 / raised_total as f64;
            let lowered_rate = lowered_dedup as f64 / lowered_total as f64;

            if lowered_rate > raised_rate + 0.15 {
                // Lowering threshold catches 15%+ more duplicates → threshold too high
                let adjustment = -0.02;
                state.global_dedup_threshold = (state.global_dedup_threshold + adjustment as f32).clamp(0.40, 0.90);
                tracing::info!(
                    "M6: lowered global threshold to {:.3} (lowered_rate={:.2}, raised_rate={:.2})",
                    state.global_dedup_threshold, lowered_rate, raised_rate
                );
            } else if raised_rate > lowered_rate + 0.15 {
                // Raising threshold still catches duplicates → threshold too low (too aggressive)
                let adjustment = 0.02;
                state.global_dedup_threshold = (state.global_dedup_threshold + adjustment as f32).clamp(0.40, 0.90);
                tracing::info!(
                    "M6: raised global threshold to {:.3} (raised_rate={:.2}, lowered_rate={:.2})",
                    state.global_dedup_threshold, raised_rate, lowered_rate
                );
            } else {
                tracing::debug!("M6: threshold stable (lowered={:.2}, raised={:.2})", lowered_rate, raised_rate);
            }
        }
    }

    // --- Part 2: Co-recall frequency signal ---
    // If two memories always appear together in recall results, they might be duplicates
    // that slipped through dedup (threshold was too high).
    let recall_events = crate::store::adaptive::consume_events(
        conn, "m6_corecall", &["recall_complete"], 100,
    ).unwrap_or_default();

    if recall_events.len() >= 5 {
        // Count pair co-occurrences in recall results
        let mut pair_counts: std::collections::HashMap<(String, String), u32> = std::collections::HashMap::new();
        let mut mem_counts: std::collections::HashMap<String, u32> = std::collections::HashMap::new();

        for event in &recall_events {
            let payload: serde_json::Value = event.payload.as_deref()
                .and_then(|p| serde_json::from_str(p).ok())
                .unwrap_or_default();

            // Payload is {"candidates": [...], ...} — extract candidate IDs
            let ids: Vec<String> = payload.get("candidates")
                .and_then(|c| c.as_array())
                .map(|arr| arr.iter()
                    .filter_map(|c| c.get("id").and_then(|v| v.as_str()).map(|s| s.to_string()))
                    .take(10) // Only top-10 results
                    .collect())
                .unwrap_or_default();

            for id in &ids {
                *mem_counts.entry(id.clone()).or_default() += 1;
            }
            for i in 0..ids.len() {
                for j in (i + 1)..ids.len() {
                    let key = if ids[i] < ids[j] {
                        (ids[i].clone(), ids[j].clone())
                    } else {
                        (ids[j].clone(), ids[i].clone())
                    };
                    *pair_counts.entry(key).or_default() += 1;
                }
            }
        }

        // Find pairs that co-occur in >80% of their individual appearances
        let event_count = recall_events.len() as u32;
        let mut suspicious_pairs = 0u32;
        for ((id_a, id_b), co_count) in &pair_counts {
            let count_a = mem_counts.get(id_a).copied().unwrap_or(0);
            let count_b = mem_counts.get(id_b).copied().unwrap_or(0);
            let min_count = count_a.min(count_b);

            // Co-recall rate: how often do they appear together relative to individually
            if min_count >= 3 && *co_count as f64 / min_count as f64 > 0.80 {
                // These two memories are almost always recalled together → likely duplicates
                // Check their content similarity
                if let (Ok(mem_a), Ok(mem_b)) = (store.get(id_a), store.get(id_b)) {
                    let sim = crate::extract::similarity(&mem_a.content, &mem_b.content);
                    if sim > 0.30 {
                        // Moderate+ similarity AND high co-recall → flag for merge
                        tracing::info!(
                            "M6 co-recall: '{}'↔'{}' co-recall={}/{}, sim={:.2} — likely duplicate",
                            &mem_a.summary.chars().take(30).collect::<String>(),
                            &mem_b.summary.chars().take(30).collect::<String>(),
                            co_count, min_count, sim,
                        );
                        // Mark the newer one for vec dedup (it will be caught in the next sweep)
                        let newer_id = if mem_a.created_at > mem_b.created_at { id_a } else { id_b };
                        let _ = conn.execute(
                            "UPDATE memories SET needs_vec_dedup = 1 WHERE id = ?1",
                            rusqlite::params![newer_id],
                        );
                        suspicious_pairs += 1;
                    }
                }
            }
        }

        // If many co-recall pairs found, threshold is probably too high
        if suspicious_pairs > 0 && event_count >= 10 {
            let pair_ratio = suspicious_pairs as f64 / event_count as f64;
            if pair_ratio > 0.2 {
                state.global_dedup_threshold = (state.global_dedup_threshold - 0.02).clamp(0.40, 0.90);
                tracing::info!(
                    "M6: co-recall signal lowered threshold to {:.3} ({suspicious_pairs} suspicious pairs in {event_count} events)",
                    state.global_dedup_threshold
                );
            }
        }
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
            let threshold = sims[p90_idx].clamp(0.40, 0.90); // Clamp to sane range
            state.dedup_thresholds.insert(*cluster_id, threshold);
            tracing::debug!("A1: cluster {cluster_id} dedup threshold = {threshold:.3} (from {} pairs)", sims.len());
        }
    }

    // Update global threshold from all-clusters distribution
    if all_sims.len() >= 10 {
        all_sims.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p90_idx = (all_sims.len() as f64 * 0.90).floor() as usize;
        let p90_idx = p90_idx.min(all_sims.len() - 1);
        let global = all_sims[p90_idx].clamp(0.40, 0.90);
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
                // Strong semantic match — provenance-preserving merge
                let (keep_id, discard_id, discard_content, discard_created) = if candidate.access_count >= 1 || candidate.created_at < chrono::Utc::now() - chrono::Duration::hours(1) {
                    (&candidate.id, id, content.to_string(), "recent".to_string())
                } else {
                    (id, candidate_id, candidate.content.clone(), candidate.created_at.format("%Y-%m-%d").to_string())
                };

                // Extract unique lines from the loser and append to winner
                if let Ok(mut kept) = store.get(keep_id) {
                    let unique = extract_unique_lines(&discard_content, &kept.content);
                    if !unique.is_empty() {
                        kept.content.push_str(&format!(
                            "\n\n[merged from {discard_id} on {discard_created}]\n{unique}"
                        ));
                    }
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
/// Run dedup scan across all topics with provenance-preserving merge.
///
/// Instead of hard-deleting duplicates (which loses temporal anchors and unique
/// details), this extracts unique lines from the "loser" and appends them to the
/// "winner" with a provenance marker. The loser is then superseded, not deleted.
///
/// Returns (duplicates_found, duplicates_merged).
pub fn run_dedup(store: &SqliteStore, threshold: f32, dry_run: bool) -> ReinResult<(u32, u32)> {
    let topics = store.list_topics()?;
    let mut dups_found = 0u32;
    let mut dups_merged = 0u32;
    for topic in &topics {
        let mems: Vec<_> = store.get_by_topic(topic)?
            .into_iter()
            .filter(|m| m.superseded_by.is_none())
            .collect();
        let mut processed: std::collections::HashSet<String> = std::collections::HashSet::new();
        for i in 0..mems.len() {
            if processed.contains(&mems[i].id) { continue; }
            for j in (i + 1)..mems.len() {
                if processed.contains(&mems[j].id) { continue; }
                let sim = crate::extract::similarity(&mems[i].content, &mems[j].content);
                if sim >= threshold {
                    dups_found += 1;
                    // Determine winner (longer/newer) and loser
                    let (winner_idx, loser_idx) = if mems[j].content.len() >= mems[i].content.len() {
                        (j, i)
                    } else {
                        (i, j)
                    };
                    if dry_run {
                        tracing::debug!("dup: '{}' ~ '{}'",
                            &mems[loser_idx].summary.chars().take(40).collect::<String>(),
                            &mems[winner_idx].summary.chars().take(40).collect::<String>());
                    } else {
                        // Provenance-preserving merge
                        let unique = extract_unique_lines(&mems[loser_idx].content, &mems[winner_idx].content);
                        if !unique.is_empty() {
                            let provenance = format!(
                                "\n\n[merged from {} on {}]\n{}",
                                mems[loser_idx].id,
                                mems[loser_idx].created_at.format("%Y-%m-%d"),
                                unique,
                            );
                            if let Ok(mut winner) = store.get(&mems[winner_idx].id) {
                                winner.content.push_str(&provenance);
                                for kw in &mems[loser_idx].keywords {
                                    if !winner.keywords.contains(kw) {
                                        winner.keywords.push(kw.clone());
                                    }
                                }
                                winner.strength = (winner.strength + 0.1).min(1.0);
                                let _ = store.update(&winner);
                            }
                        }
                        let _ = store.mark_superseded(&mems[loser_idx].id, &mems[winner_idx].id);
                        dups_merged += 1;
                    }
                    // Mark only the loser as processed
                    processed.insert(mems[loser_idx].id.clone());
                    // If mems[i] was the loser, stop scanning (it's been superseded)
                    // If mems[i] was the winner, continue scanning for more duplicates
                    if loser_idx == i { break; }
                }
            }
        }
    }
    Ok((dups_found, dups_merged))
}

/// Extract lines from `source` that are not present in `target`.
/// Used for provenance-preserving merge: keeps unique temporal anchors and details.
pub fn extract_unique_lines(source: &str, target: &str) -> String {
    let target_lower = target.to_lowercase();
    let unique: Vec<&str> = source.lines()
        .filter(|line| {
            let trimmed = line.trim();
            if trimmed.is_empty() { return false; }
            // Skip merge markers from previous merges (prevent marker accumulation)
            if trimmed.starts_with("[merged from ") || trimmed.starts_with("[merged on ") {
                return false;
            }
            // Keep lines that have dates (temporal anchors) or aren't found in target
            let has_date = trimmed.chars().any(|c| c.is_ascii_digit())
                && {
                    // Dynamic date detection: current year ± 10
                    use chrono::Datelike;
                    let year = chrono::Utc::now().year();
                    let has_year = ((year - 10)..=(year + 10)).any(|y: i32| trimmed.contains(&y.to_string()));
                    has_year || trimmed.contains("月")
                };
            let is_unique = !target_lower.contains(&trimmed.to_lowercase());
            has_date || is_unique
        })
        .collect();
    unique.join("\n")
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_unique_lines_preserves_dates() {
        let source = "2026-03-15 direct login failed\n2026-03-17 used jump host\ngeneral note about SSH";
        let target = "general note about SSH and connection\n2026-03-22 containerTag mismatch";

        let unique = extract_unique_lines(source, target);
        // Should preserve date-anchored lines even if partially overlapping
        assert!(unique.contains("2026-03-15"), "should keep date 03-15: {unique}");
        assert!(unique.contains("2026-03-17"), "should keep date 03-17: {unique}");
    }

    #[test]
    fn test_extract_unique_lines_filters_duplicates() {
        let source = "line A\nline B\nline C";
        let target = "line A\nline C\nline D";

        let unique = extract_unique_lines(source, target);
        assert!(unique.contains("line B"), "should keep unique line B");
        assert!(!unique.contains("line A"), "should not keep duplicate line A");
        assert!(!unique.contains("line C"), "should not keep duplicate line C");
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
}
