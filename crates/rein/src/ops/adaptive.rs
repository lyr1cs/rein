//! Adaptive pipeline: HDBSCAN clustering, survival curves, tiering, alpha learning,
//! reranker weight learning, M6 threshold learning, and per-cluster dedup thresholds.

use crate::config::ReinConfig;
use crate::store::SqliteStore;
use crate::types::traits::MemoryStore;

use super::dedup::run_vec_dedup;

/// Run the adaptive engine slow-channel pipeline after GC.
/// Order: M4 (HDBSCAN) → M3 (Survival) → M5 (Tiering) → M2 (Alpha) → persist.
/// Each step is gated by readiness checks; failures skip subsequent steps.
pub fn run_adaptive_pipeline(store: &SqliteStore, config: &ReinConfig) {
    if !config.adaptive.enabled {
        return;
    }

    let _span = tracing::info_span!("adaptive_pipeline").entered();

    // Restore or create AdaptiveState
    let mut state =
        crate::store::adaptive::AdaptiveState::restore_snapshot(store.conn()).unwrap_or_default();

    // Snapshot state before learning for convergence tracking
    let prev_state = state.clone();

    // Count memories for readiness checks
    let mem_count: u64 = store
        .conn()
        .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
        .unwrap_or(0);

    // Step 1: M4 — HDBSCAN clustering (skip if < 50 memories with embeddings)
    let embeddings_count: u64 = store
        .conn()
        .query_row("SELECT COUNT(*) FROM vec_memories", [], |r| r.get(0))
        .unwrap_or(0);

    if embeddings_count >= 50 {
        run_hdbscan_clustering(store, &mut state, embeddings_count as usize);
    }

    // Step 1b: A1 — Compute per-cluster dedup thresholds
    if !state.memory_clusters.is_empty() {
        compute_per_cluster_dedup_thresholds(store, &mut state);
    }

    // Step 1c: v0.23 — per-cluster + global canonical length percentiles.
    // Drives adaptive `target_bytes` for resummerize compression. Cheap DB
    // scan; runs even without clusters so the global fallback accumulates
    // from day one.
    match crate::store::adaptive::recompute_canonical_length_stats(store.conn()) {
        Ok((per_cluster, global)) => {
            state.canonical_length_stats = per_cluster;
            state.global_canonical_length = global;
        }
        Err(e) => tracing::warn!("failed to recompute canonical_length_stats: {e}"),
    }

    // v0.24 peek+commit: collect (consumer, max_event_id) batches from
    // any event-sourced helper that mutates `state`. Each batch is
    // committed only after `state.save_snapshot()` succeeds (Step 6),
    // so a crash mid-pipeline replays cleanly on the next pass — the
    // events are NOT marked consumed unless the derived state change is
    // durable. Module-level invariant in `store/adaptive.rs`.
    //
    // Each helper returns `Vec<(&'static str, i64)>` of pending offsets
    // (size 1 for L3/M6, size 2 for M2 alpha pair). The orchestrator
    // routes each batch to a single `commit_offset` call so paired
    // consumers advance atomically.
    let mut pending_offset_batches: Vec<Vec<(&'static str, i64)>> = Vec::new();

    // Step 1d: v0.24 ARS L3 — concept refresh-interval percentiles. Drains
    // any new `ConceptSummaryRefreshed` events into the rolling reservoir
    // and recomputes the cached p75/p50. Until at least
    // `CONCEPT_REFRESH_MIN_SAMPLES` samples accumulate, the trigger
    // threshold helpers fall back to bootstrap constants — see
    // `AdaptiveState::concept_refresh_revision_threshold`.
    match crate::store::adaptive::recompute_concept_refresh_stats(
        store.conn(),
        state.concept_refresh_stats.clone(),
    ) {
        Ok((stats, max_id)) => {
            state.concept_refresh_stats = Some(stats);
            if let Some(id) = max_id {
                pending_offset_batches.push(vec![("concept_refresh_stats", id)]);
            }
        }
        Err(e) => tracing::warn!("failed to recompute concept_refresh_stats: {e}"),
    }

    // Step 1e: v0.26 D direction — synthesis interaction feedback. Drains any
    // new `SynthesisInteraction` events into per-cluster ClusterSynthesisStats
    // so `decide_synthesize` can route adaptive synthesis decisions. Until
    // `SYNTHESIS_COLD_START_N` events accumulate per (cluster_id, query_type)
    // bucket, the gate falls back to the global flag — see
    // `ops/recall_synthesis::decide_synthesize`. Same peek+commit + CAS-merge
    // pattern as concept_refresh_stats above (5-invariant pattern).
    match crate::store::adaptive::recompute_synthesis_feedback_stats(
        store.conn(),
        state.synthesis_feedback_stats.clone(),
    ) {
        Ok((stats, max_id)) => {
            state.synthesis_feedback_stats = Some(stats);
            if let Some(id) = max_id {
                pending_offset_batches.push(vec![("synthesis_feedback", id)]);
            }
        }
        Err(e) => tracing::warn!("failed to recompute synthesis_feedback_stats: {e}"),
    }

    // Step 2: M3 — Build per-cluster survival curves from access data
    if !state.memory_clusters.is_empty() {
        build_survival_curves(store, &state);
    }

    // Step 3: M5 — Tier boundaries + cold_archive migration
    if mem_count >= config.adaptive.tier_cold_start as u64 {
        run_tiering(store, &mut state, config);
    }

    // Step 3a: v0.26 Cap C — bulk cold-tier archival summary worker. Walks
    // rows flagged by `run_tiering` (`needs_archival_summary = 1`), claims +
    // generates summaries via the LLM, applies the 3-invariant lossless
    // contract, persists with 5-way CAS. Skipped cleanly when
    // `ars.cold_archive_enabled = false`. No-op when nothing was flagged.
    //
    // Pipeline ordering note: `run_tiering` above also runs the M5 strip
    // pass which replaces `memory.content` with `memory.summary` for
    // archived rows. Cap C's `attempt_one` reads `cold_archive.content` as
    // a fallback when the row is in the archive table, so the worker
    // always sees the original content even when strip ran first
    // (v0.26.0 patch: Option C — defended inside the worker rather than
    // by reordering pipeline steps).
    {
        let cold_config = crate::ops::cold_archive_summary::ColdArchiveConfig::from_ars(&config.ars);
        match crate::ops::cold_archive_summary::run_cold_archive_summary(store, config, &cold_config) {
            Ok(report) => {
                if report.considered > 0 {
                    tracing::info!(
                        considered = report.considered,
                        generated = report.generated,
                        skipped_short = report.skipped_short,
                        strikes = report.strikes,
                        errors = report.errors,
                        exhausted = report.exhausted,
                        "Cap C: cold-archive pass complete"
                    );
                }
            }
            Err(e) => tracing::warn!("Cap C: cold-archive worker failed: {e}"),
        }
    }

    // Step 4: M2 — Counterfactual alpha optimization (peek events, learn alphas)
    if let Some(batch) = run_alpha_learning(store, &mut state, config) {
        pending_offset_batches.push(batch);
    }

    // Step 4a: Reranker weight learning from agent feedback. Self-contained
    // peek+commit (writes to `weights` table, not `adaptive_state`), so it
    // commits its own offsets in-function and does NOT contribute to the
    // post-save batch list.
    run_reranker_weight_learning(store);

    // Step 4b: M6 — Consume threshold exploration data + co-recall signal → update dedup thresholds
    if let Some(batch) = run_m6_threshold_learning(store, &mut state) {
        pending_offset_batches.push(batch);
    }

    // Step 5: Embedding-based dedup for memories marked needs_vec_dedup
    run_vec_dedup(store, config);

    // Step 6: Persist snapshot + emit param_update event
    state.version += 1;
    let snapshot_saved = match state.save_snapshot(store.conn()) {
        Ok(()) => {
            tracing::debug!("adaptive state v{} saved", state.version);
            true
        }
        Err(e) => {
            tracing::warn!("failed to save adaptive state: {e}");
            false
        }
    };

    // Step 6b: Post-save offset commits. Honor the module invariant —
    // never advance a consumer's cursor unless the derived state change
    // is durable. If save_snapshot failed, all pending batches are
    // discarded; the next pipeline pass will re-peek and replay.
    if snapshot_saved {
        for batch in &pending_offset_batches {
            let pairs: Vec<(&str, i64)> =
                batch.iter().map(|(c, id)| (*c, *id)).collect();
            if let Err(e) = crate::store::adaptive::commit_offset(store.conn(), &pairs) {
                tracing::warn!(
                    consumers = ?batch.iter().map(|(c, _)| *c).collect::<Vec<_>>(),
                    error = %e,
                    "post-save commit_offset failed; events will be re-peeked next pass"
                );
            }
        }
    } else if !pending_offset_batches.is_empty() {
        tracing::warn!(
            batches = pending_offset_batches.len(),
            "snapshot save failed; deferring {} pending offset batches for replay",
            pending_offset_batches.len()
        );
    }

    // Convergence health summary
    {
        let max_alpha_delta = state
            .learned_alpha
            .iter()
            .map(|(k, entry)| {
                let prev_val = prev_state
                    .learned_alpha
                    .get(k)
                    .map(|e| e.value)
                    .unwrap_or(entry.value);
                (entry.value - prev_val).abs()
            })
            .fold(0.0_f64, f64::max);

        let alpha_warning = max_alpha_delta > 0.10;

        let survival_curves_active: u64 = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM metadata WHERE key LIKE 'survival_curve:%'",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0);

        let tiers_monotonic = state.hot_threshold >= state.cold_threshold;

        tracing::info!(
            max_alpha_delta = %format!("{max_alpha_delta:.4}"),
            alpha_warning,
            survival_curves_active,
            tiers_monotonic,
            hot_threshold = %format!("{:.4}", state.hot_threshold),
            cold_threshold = %format!("{:.4}", state.cold_threshold),
            "adaptive convergence summary"
        );
    }

    // Step 7: Cleanup expired events
    let cleaned = crate::store::adaptive::cleanup_expired_events(
        store.conn(),
        config.adaptive.event_retention_days,
    )
    .unwrap_or(0);
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
        "SELECT vm.id, vm.embedding FROM vec_memories vm
             JOIN memories m ON m.id = vm.id
             LIMIT ?1",
    ) {
        Ok(mut stmt) => stmt
            .query_map(rusqlite::params![load_limit as i64], |row| {
                let id: String = row.get(0)?;
                let blob: Vec<u8> = row.get(1)?;
                let floats: Vec<f32> = blob
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                Ok((id, floats))
            })
            .ok()
            .map(|rows| rows.filter_map(|r| r.ok()).collect())
            .unwrap_or_default(),
        Err(_) => return,
    };

    if embeddings.len() < 50 {
        return;
    }

    let min_cluster_size = 5.max(embeddings.len() / 50); // adaptive: ~2% of dataset
    let result = crate::store::hdbscan::hdbscan(&embeddings, min_cluster_size);

    let clustered_ids: std::collections::HashSet<&str> =
        embeddings.iter().map(|(id, _)| id.as_str()).collect();
    let mut sampled_clusters: std::collections::HashMap<String, u32> =
        std::collections::HashMap::new();
    for (i, label) in result.labels.iter().enumerate() {
        let mem_id = &embeddings[i].0;
        if let Some(cluster_id) = label {
            sampled_clusters.insert(mem_id.clone(), *cluster_id);
        }
    }

    // Compute and persist cluster centroids BEFORE reassigning non-sampled memories,
    // so we can use nearest-centroid instead of keeping stale cluster_id labels.
    let dim = embeddings.first().map(|(_, v)| v.len()).unwrap_or(0);
    let centroids: std::collections::HashMap<u32, Vec<f32>> = if dim > 0 {
        let mut cluster_points: std::collections::HashMap<u32, Vec<usize>> =
            std::collections::HashMap::new();
        for (i, label) in result.labels.iter().enumerate() {
            if let Some(cluster_id) = label {
                cluster_points.entry(*cluster_id).or_default().push(i);
            }
        }
        let mut c: std::collections::HashMap<u32, Vec<f32>> = std::collections::HashMap::new();
        for (cluster_id, indices) in &cluster_points {
            let centroid = compute_cluster_centroid(indices, &embeddings, dim);
            c.insert(*cluster_id, centroid);
        }
        c
    } else {
        std::collections::HashMap::new()
    };

    // Reassign non-sampled memories via nearest centroid instead of keeping stale
    // cluster_id labels from the previous HDBSCAN run.  Cluster IDs are local
    // labels — reusing old IDs from a different run would silently corrupt
    // per-cluster adaptive state (M2 alpha, M3 survival, A1 dedup thresholds).
    //
    // Use the in-memory `clustered_ids` set (derived from the actual HDBSCAN input)
    // to identify non-sampled memories — avoids a second LIMIT query whose row set
    // could differ from the first due to non-deterministic ordering.
    let persist_result =
        (|| -> crate::types::ReinResult<(std::collections::HashMap<String, u32>, u32)> {
            let mut new_clusters = sampled_clusters.clone();
            let mut reassigned = 0u32;
            store.conn().execute_batch("SAVEPOINT hdbscan_recluster")?;
            store.conn().execute_batch(
                "CREATE TEMP TABLE IF NOT EXISTS _hdbscan_sampled (id TEXT PRIMARY KEY)",
            )?;
            store.conn().execute("DELETE FROM _hdbscan_sampled", [])?;
            {
                let mut ins = store
                    .conn()
                    .prepare_cached("INSERT OR IGNORE INTO _hdbscan_sampled (id) VALUES (?1)")?;
                for id in &clustered_ids {
                    ins.execute(rusqlite::params![id])?;
                }
            }

            for id in &clustered_ids {
                store.conn().execute(
                    "UPDATE memories SET cluster_id = NULL WHERE id = ?1",
                    rusqlite::params![id],
                )?;
            }
            for (mem_id, cluster_id) in &sampled_clusters {
                store.conn().execute(
                    "UPDATE memories SET cluster_id = ?1 WHERE id = ?2",
                    rusqlite::params![cluster_id, mem_id],
                )?;
            }

            store
                .conn()
                .execute("DELETE FROM metadata WHERE key LIKE 'survival_curve:%'", [])?;
            crate::store::adaptive::save_cluster_centroids(
                store.conn(),
                &centroids,
                state.cluster_version + 1,
                dim,
            )?;

            if !centroids.is_empty() {
                store.conn().execute(
                    "UPDATE memories SET cluster_id = NULL
                 WHERE cluster_id IS NOT NULL
                   AND id NOT IN (SELECT id FROM _hdbscan_sampled)",
                    [],
                )?;

                let reassign_batch_size = 2000i64;
                let mut cursor = String::new();
                loop {
                    let batch: Vec<(String, Vec<f32>)> = {
                        let mut stmt = store.conn().prepare(
                            "SELECT vm.id, vm.embedding FROM vec_memories vm
                         WHERE vm.id NOT IN (SELECT id FROM _hdbscan_sampled)
                           AND vm.id > ?1
                         ORDER BY vm.id
                         LIMIT ?2",
                        )?;
                        let rows = stmt.query_map(
                            rusqlite::params![&cursor, reassign_batch_size],
                            |row| {
                                let id: String = row.get(0)?;
                                let blob: Vec<u8> = row.get(1)?;
                                let floats: Vec<f32> = blob
                                    .chunks_exact(4)
                                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                                    .collect();
                                Ok((id, floats))
                            },
                        )?;
                        rows.collect::<Result<Vec<_>, _>>()?
                    };
                    if batch.is_empty() {
                        break;
                    }
                    let batch_len = batch.len() as i64;
                    if let Some((last_id, _)) = batch.last() {
                        cursor = last_id.clone();
                    }
                    for (mem_id, emb) in &batch {
                        if let Some(cid) =
                            crate::store::adaptive::assign_to_nearest_centroid(&centroids, emb)
                        {
                            store.conn().execute(
                                "UPDATE memories SET cluster_id = ?1 WHERE id = ?2",
                                rusqlite::params![cid, mem_id],
                            )?;
                            new_clusters.insert(mem_id.clone(), cid);
                            reassigned += 1;
                        }
                    }
                    if batch_len < reassign_batch_size {
                        break;
                    }
                }
            }

            store
                .conn()
                .execute("DROP TABLE IF EXISTS _hdbscan_sampled", [])?;
            store.conn().execute_batch("RELEASE hdbscan_recluster")?;
            Ok((new_clusters, reassigned))
        })();
    let (new_clusters, reassigned) = match persist_result {
        Ok(result) => result,
        Err(e) => {
            let _ = store.conn().execute_batch("ROLLBACK TO hdbscan_recluster");
            let _ = store.conn().execute_batch("RELEASE hdbscan_recluster");
            tracing::error!("M4: failed to persist recluster atomically: {e}");
            return;
        }
    };
    state.memory_clusters = new_clusters;
    state.dedup_thresholds.clear();
    state.learned_alpha.retain(|k, _| !k.contains(':'));
    if reassigned > 0 {
        tracing::info!("M4: reassigned {reassigned} non-sampled memories via nearest centroid");
    }

    state.cluster_version += 1;
    state.centroid_version = state.cluster_version;
    tracing::info!(
        "M4: {} clusters, {} noise points, {} assigned (v{})",
        result.clusters.len(),
        result.noise_indices.len(),
        state.memory_clusters.len(),
        state.cluster_version,
    );
}

/// Element-wise mean of the given embedding indices.
fn compute_cluster_centroid(
    indices: &[usize],
    embeddings: &[(String, Vec<f32>)],
    dim: usize,
) -> Vec<f32> {
    let mut centroid = vec![0.0f64; dim];
    let count = indices.len();
    for &idx in indices {
        if idx < embeddings.len() {
            for (j, &v) in embeddings[idx].1.iter().enumerate().take(dim) {
                centroid[j] += v as f64;
            }
        }
    }
    centroid
        .into_iter()
        .map(|c| (c / count.max(1) as f64) as f32)
        .collect()
}

// ===========================================================================
// M3: Build per-cluster survival curves from access timestamps
// ===========================================================================

fn build_survival_curves(store: &SqliteStore, state: &crate::store::adaptive::AdaptiveState) {
    use std::collections::HashMap;

    // Group memories by cluster, collect access timestamps
    let mut cluster_intervals: HashMap<u32, Vec<crate::search::survival::SurvivalInterval>> =
        HashMap::new();
    let now = chrono::Utc::now();

    for (mem_id, &cluster_id) in &state.memory_clusters {
        // Get created_at, last_accessed, access_count for this memory
        let row: Option<(String, String, u32)> = store
            .conn()
            .query_row(
                "SELECT created_at, last_accessed, access_count FROM memories WHERE id = ?1",
                rusqlite::params![mem_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .ok();

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
            if access_count == 1 {
                // Single access: one observed event at total_days since creation.
                // Skip near-zero durations (created_at ≈ last_accessed) to avoid
                // flooding the KM estimator with zero-duration events.
                if total_days > 0.01 {
                    intervals.push(crate::search::survival::SurvivalInterval {
                        duration_days: total_days,
                        is_event: true,
                    });
                }
            } else if total_days > 0.0 {
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
        if intervals.len() < 10 {
            continue;
        } // Need minimum data for meaningful curve

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

    // M3 cold-start: build global prior from all cluster observations combined
    let all_intervals: Vec<crate::search::survival::SurvivalInterval> =
        cluster_intervals.values().flatten().cloned().collect();
    if all_intervals.len() >= 20 {
        if let Some(mut global_curve) = crate::search::survival::kaplan_meier(&all_intervals) {
            // Cap total_count to keep the global prior in the cold-start blend zone
            // (between cold_start_min=20 and cold_start_max=50). This ensures
            // adaptive_strength() still blends with Ebbinghaus rather than trusting
            // the global curve 100%, preserving its role as a prior, not an oracle.
            global_curve.total_count = global_curve.total_count.min(35);
            let key = "survival_curve:global";
            if let Ok(json) = serde_json::to_string(&global_curve) {
                let _ = store.conn().execute(
                    "INSERT INTO metadata (key, value) VALUES (?1, ?2)
                     ON CONFLICT(key) DO UPDATE SET value = ?2",
                    rusqlite::params![key, json],
                );
                tracing::info!(
                    "M3: built global prior survival curve ({} observations)",
                    global_curve.total_count
                );
            }
        }
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
    if let Ok(mut stmt) = store
        .conn()
        .prepare("SELECT access_count, created_at FROM memories WHERE status = 'active'")
    {
        let rates: Vec<f64> = stmt
            .query_map([], |row| {
                let ac: u32 = row.get(0)?;
                let created_str: String = row.get(1)?;
                let created = chrono::DateTime::parse_from_rfc3339(&created_str)
                    .map(|dt| dt.with_timezone(&chrono::Utc))
                    .unwrap_or_else(|_| chrono::Utc::now());
                Ok(crate::store::tiering::compute_access_rate(ac, created))
            })
            .ok()
            .map(|rows| rows.filter_map(|r| r.ok()).collect())
            .unwrap_or_default();

        if !rates.is_empty() {
            boundaries.update(&rates);
            state.hot_threshold = boundaries.hot_threshold;
            state.cold_threshold = boundaries.cold_threshold;
        }
    }

    // Update tier labels on memories
    // NOTE: SQL formula must stay in sync with crate::store::tiering::compute_access_rate
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

        // Cap C (v0.26): flag freshly-demoted cold memories for archival-summary
        // generation by the slow-channel worker (`ops/cold_archive_summary.rs`).
        // Re-flag covers both cases: (a) row never summarized yet (NULL summary),
        // OR (b) summary present but written under a previous
        // `ARCHIVAL_SUMMARY_VERSION` (recall suppresses stale-version summaries
        // anyway, so they need regeneration). `needs_archival_summary = 0` skip
        // avoids redundant worker wake-ups; on exhaustion the flag is set to 2
        // (terminal) and stays invisible until the next cold transition rebumps it.
        let _ = store.conn().execute(
            "UPDATE memories SET needs_archival_summary = 1
             WHERE status = 'active'
               AND tier = 'cold'
               AND needs_archival_summary = 0
               AND (
                   archival_summary IS NULL
                   OR archival_summary_version IS NULL
                   OR archival_summary_version != ?1
               )",
            rusqlite::params![crate::ops::cold_archive_summary::ARCHIVAL_SUMMARY_VERSION as i64],
        );
    }

    // Migrate cold memories to cold_archive (content → summary, original in archive).
    // The strip happens here regardless of Cap C state. Cap C's generator reads
    // `cold_archive.content` as a fallback when the row is in the archive table,
    // so the order of strip vs Cap C does NOT affect what content the LLM sees
    // (v0.26.0 patch: Option C — Cap C self-defends via cold_archive fallback
    // in `attempt_one`, instead of relying on pipeline step ordering).
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
        let archived_ids: Vec<String> = store
            .conn()
            .prepare(
                "SELECT memory_id FROM cold_archive WHERE memory_id IN (
                SELECT id FROM memories WHERE tier = 'cold' AND strength < 0.3 AND access_count = 0
            )",
            )
            .ok()
            .and_then(|mut stmt| {
                stmt.query_map([], |row| row.get(0))
                    .ok()
                    .map(|rows| rows.filter_map(|r| r.ok()).collect())
            })
            .unwrap_or_default();

        // Batch cold archive stripping in a transaction
        let _ = store.conn().execute_batch("BEGIN");
        for aid in &archived_ids {
            if let Ok(mut mem) = store.get(aid) {
                mem.content = mem.summary.clone();
                let _ = store.update(&mem); // Triggers Tantivy + FTS update
            }
        }
        let _ = store.conn().execute_batch("COMMIT");
        tracing::info!(
            "M5: migrated {migrated} cold memories to archive ({} stripped), hot={:.4} cold={:.4}",
            archived_ids.len(),
            state.hot_threshold,
            state.cold_threshold
        );
    } else {
        tracing::debug!(
            "M5: hot={:.4}, cold={:.4}, no migrations needed",
            state.hot_threshold,
            state.cold_threshold
        );
    }
}

// ===========================================================================
// M2: Counterfactual alpha optimization — consume events, learn alphas
// ===========================================================================

/// Fetch pending recall_complete events from feedback_events table
/// without consuming them (offset is not advanced yet).
fn peek_recall_events(conn: &rusqlite::Connection) -> Vec<crate::store::adaptive::StoredEvent> {
    let last_offset: i64 = conn
        .query_row(
            "SELECT last_event_id FROM consumer_offsets WHERE consumer = 'alpha_optimizer'",
            [],
            |r| r.get(0),
        )
        .unwrap_or(0);

    match conn.prepare(
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
        Err(_) => Vec::new(),
    }
}

/// Parse candidate score logs from a recall_complete event payload.
/// Returns a list of CandidateLog structs extracted from the JSON payload.
fn parse_candidates_from_event(
    event: &crate::store::adaptive::StoredEvent,
    access_events: &[crate::store::adaptive::StoredEvent],
) -> Option<crate::search::alpha_optimizer::RecallEvent> {
    let request_id = event.request_id.as_ref()?.clone();
    let payload = event.payload.as_ref()?;

    let payload_obj: serde_json::Value = serde_json::from_str(payload).unwrap_or_default();
    let candidates_json: Vec<serde_json::Value> = payload_obj
        .get("candidates")
        .and_then(|c| c.as_array())
        .cloned()
        .unwrap_or_default();
    if candidates_json.is_empty() {
        return None;
    }

    let candidates: Vec<crate::search::alpha_optimizer::CandidateLog> = candidates_json
        .iter()
        .filter_map(|c| {
            Some(crate::search::alpha_optimizer::CandidateLog {
                memory_id: c.get("id")?.as_str()?.to_string(),
                bm25_norm: c.get("bm25_norm")?.as_f64()? as f32,
                vec_norm: c.get("vec_norm")?.as_f64()? as f32,
                kg_norm: c.get("kg_norm").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                episode_norm: c
                    .get("episode_norm")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as f32,
                support_count: c.get("support_count").and_then(|v| v.as_u64()).unwrap_or(1) as u32,
                source_diversity: c
                    .get("source_diversity")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(1.0) as f32,
            })
        })
        .collect();

    if candidates.is_empty() {
        return None;
    }

    let ts = chrono::DateTime::parse_from_rfc3339(&event.ts)
        .map(|dt| dt.with_timezone(&chrono::Utc))
        .unwrap_or_else(|_| chrono::Utc::now());

    // Find which candidate memories were actually accessed (injected by hook_prompt).
    // Match by: (a) memory_id appears in this recall's candidate set, AND
    // (b) access event timestamp is within 10 minutes of the recall event.
    // The time window reduces false attribution when the same memory appears
    // in multiple unrelated recalls.
    let candidate_ids: std::collections::HashSet<&str> =
        candidates.iter().map(|c| c.memory_id.as_str()).collect();
    let accessed_ids: Vec<String> = access_events
        .iter()
        .filter(|a| {
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
        .into_iter()
        .collect();

    Some(crate::search::alpha_optimizer::RecallEvent {
        request_id,
        candidates,
        accessed_ids,
        timestamp: ts,
    })
}

/// Compute optimal alpha values via counterfactual replay over candidate sets.
/// Updates both global and per-query-type alphas in the AdaptiveState.
fn compute_counterfactual_alphas(
    events_with_access: &[crate::search::alpha_optimizer::RecallEvent],
    stored_events: &[crate::store::adaptive::StoredEvent],
    state: &mut crate::store::adaptive::AdaptiveState,
    config: &ReinConfig,
) {
    let decay_lambda = 0.06; // ~11 day half-life for event weighting

    // Compute global alpha
    if let Some(learned) =
        crate::search::alpha_optimizer::optimize_alpha(events_with_access, decay_lambda)
    {
        let key = "global".to_string();
        let current = state
            .learned_alpha
            .get(&key)
            .map(|e| e.value)
            .unwrap_or(0.5);
        let stepped = crate::search::alpha_optimizer::apply_max_step(
            current,
            learned.value,
            config.adaptive.alpha_max_step,
        );
        let shrunk = crate::search::alpha_optimizer::bayesian_shrinkage(
            stepped,
            0.5,
            learned.sample_count,
            config.adaptive.shrinkage_prior,
        );

        state.learned_alpha.insert(
            key,
            crate::store::adaptive::LearnedAlphaEntry {
                value: shrunk,
                sample_count: learned.sample_count,
                last_updated: chrono::Utc::now().to_rfc3339(),
            },
        );

        tracing::info!(
            "M2: learned global alpha = {shrunk:.3} (from {} events, raw={:.3})",
            learned.sample_count,
            learned.value
        );
    }

    // Per-query-type alphas
    // Build request_id → query_type map once to avoid O(n²) lookups
    let qt_map: std::collections::HashMap<&str, &str> = stored_events
        .iter()
        .filter_map(|se| se.request_id.as_deref().zip(se.query_type.as_deref()))
        .collect();

    for qt in &[
        "episodic",
        "temporal",
        "preference",
        "exact",
        "semantic",
        "exploratory",
    ] {
        let qt_events: Vec<_> = events_with_access
            .iter()
            .filter(|e| qt_map.get(e.request_id.as_str()).copied() == Some(qt))
            .cloned()
            .collect();

        if qt_events.len() < config.adaptive.min_samples_alpha {
            continue;
        }

        if let Some(learned) =
            crate::search::alpha_optimizer::optimize_alpha(&qt_events, decay_lambda)
        {
            let global_alpha = state
                .learned_alpha
                .get("global")
                .map(|e| e.value)
                .unwrap_or(0.5);
            let shrunk = crate::search::alpha_optimizer::bayesian_shrinkage(
                learned.value,
                global_alpha,
                learned.sample_count,
                config.adaptive.shrinkage_prior,
            );

            state.learned_alpha.insert(
                qt.to_string(),
                crate::store::adaptive::LearnedAlphaEntry {
                    value: shrunk,
                    sample_count: learned.sample_count,
                    last_updated: chrono::Utc::now().to_rfc3339(),
                },
            );

            tracing::info!(
                "M2: learned {qt} alpha = {shrunk:.3} ({} events)",
                learned.sample_count
            );
        }
    }

    // M2 extension: per-cluster per-query-type alpha learning
    // Bucket events by (query_type, dominant_cluster_id)
    let mut cluster_buckets: std::collections::HashMap<
        (String, u32),
        Vec<crate::search::alpha_optimizer::RecallEvent>,
    > = std::collections::HashMap::new();
    for re in events_with_access {
        let qt = qt_map
            .get(re.request_id.as_str())
            .copied()
            .unwrap_or("semantic");
        // Vote for dominant cluster among accessed memories
        let mut votes: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        for id in &re.accessed_ids {
            if let Some(&cid) = state.memory_clusters.get(id) {
                *votes.entry(cid).or_default() += 1;
            }
        }
        if let Some((&cid, _)) = votes.iter().max_by_key(|(_, &c)| c) {
            cluster_buckets
                .entry((qt.to_string(), cid))
                .or_default()
                .push(re.clone());
        }
    }

    for ((qt, cluster_id), events) in &cluster_buckets {
        if events.len() < config.adaptive.min_samples_alpha {
            continue;
        }
        if let Some(learned) = crate::search::alpha_optimizer::optimize_alpha(events, decay_lambda)
        {
            // Shrink toward the query-type level alpha (or global as fallback)
            let parent_alpha = state
                .learned_alpha
                .get(qt.as_str())
                .or_else(|| state.learned_alpha.get("global"))
                .map(|e| e.value)
                .unwrap_or(0.5);
            // Dampen step size to prevent volatile swings with small sample counts
            let current = state
                .learned_alpha
                .get(&crate::store::adaptive::AdaptiveState::bucket_key(
                    qt,
                    Some(*cluster_id),
                ))
                .map(|e| e.value)
                .unwrap_or(parent_alpha);
            let stepped = crate::search::alpha_optimizer::apply_max_step(
                current,
                learned.value,
                config.adaptive.alpha_max_step,
            );
            let shrunk = crate::search::alpha_optimizer::bayesian_shrinkage(
                stepped,
                parent_alpha,
                learned.sample_count,
                config.adaptive.shrinkage_prior,
            );
            let key = crate::store::adaptive::AdaptiveState::bucket_key(qt, Some(*cluster_id));
            state.learned_alpha.insert(
                key,
                crate::store::adaptive::LearnedAlphaEntry {
                    value: shrunk,
                    sample_count: learned.sample_count,
                    last_updated: chrono::Utc::now().to_rfc3339(),
                },
            );
            tracing::info!(
                "M2: learned {qt}:{cluster_id} alpha = {shrunk:.3} ({} events)",
                learned.sample_count
            );
        }
    }
}

/// Main orchestrator for M2 alpha learning.
///
/// Peeks at recall events, parses candidates, advances both consumer offsets
/// (`alpha_optimizer` over recall_complete, `alpha_optimizer_access` over
/// recall_access) atomically, then computes optimal alphas.
///
/// **Atomicity invariant:** both offsets are advanced inside one
/// `BEGIN IMMEDIATE` / `COMMIT` block. The previous implementation advanced
/// `alpha_optimizer_access` mid-function via `consume_events` and
/// `alpha_optimizer` later via a separate `INSERT`, with no enclosing
/// transaction. A crash between those writes marked access events consumed
/// while the matching recall_complete events were re-peeked — producing
/// ghost events (no access signal), silently dropped by
/// `events_with_access`, and the alpha for that window went unlearned.
fn run_alpha_learning(
    store: &SqliteStore,
    state: &mut crate::store::adaptive::AdaptiveState,
    config: &ReinConfig,
) -> Option<Vec<(&'static str, i64)>> {
    let conn = store.conn();

    let events = peek_recall_events(conn);
    if events.is_empty() {
        return None;
    }

    // Peek recall_access events WITHOUT advancing their offset. Advancement
    // happens below, inside the same transaction that moves the
    // alpha_optimizer cursor, so the two offsets cannot disagree across a
    // process crash.
    let access_events = peek_access_events(conn, 500);

    // Replay-safety (Codex Tier-B+C round-2 HIGH): if a prior pass
    // applied alpha shrinkage and the post-save `commit_offset` failed,
    // replay must NOT double-apply. Filter by the durable watermark.
    //
    // Codex Tier-B+C round-3 HIGH: the watermark is bumped LATER (after
    // the prefix-safe walk computes `advance_to`/`access_advance_to`) —
    // NOT to the raw peek-max — so late-arriving access events for
    // already-peeked recall ids are never silently discarded.
    let prior_alpha_water = state.alpha_optimizer_last_id;
    let prior_access_water = state.alpha_optimizer_access_last_id;
    let raw_events_was_nonempty = !events.is_empty();
    let events: Vec<_> = events
        .into_iter()
        .filter(|e| e.id > prior_alpha_water)
        .collect();
    let access_events: Vec<_> = access_events
        .into_iter()
        .filter(|e| e.id > prior_access_water)
        .collect();
    if events.is_empty() {
        // Codex Tier-B+C round-3 HIGH (livelock fix): pure replay — all
        // peeked events were already applied in a prior pass whose
        // `commit_offset` failed. The stale offset would loop forever
        // unless we drain it now. Returning the prior watermarks lets
        // the orchestrator's `commit_offset` advance the consumer
        // cursor past the already-applied prefix.
        if raw_events_was_nonempty {
            let mut pending: Vec<(&'static str, i64)> = Vec::new();
            if prior_alpha_water > 0 {
                pending.push(("alpha_optimizer", prior_alpha_water));
            }
            if prior_access_water > 0 {
                pending.push(("alpha_optimizer_access", prior_access_water));
            }
            return if pending.is_empty() { None } else { Some(pending) };
        }
        return None;
    }

    // Build RecallEvent structs from stored events
    let recall_events: Vec<crate::search::alpha_optimizer::RecallEvent> = events
        .iter()
        .filter_map(|event| parse_candidates_from_event(event, &access_events))
        .collect();

    // Only learn from events that have actual access data
    let events_with_access: Vec<_> = recall_events
        .iter()
        .filter(|e| !e.accessed_ids.is_empty())
        .cloned()
        .collect();

    // Advance offset through contiguous prefix of matched or expired events.
    // Stop at the first live unmatched event (its access signal may arrive later).
    // 24h expiry prevents a single stale event from permanently blocking the pipeline.
    let cutoff = chrono::Utc::now() - chrono::Duration::hours(24);
    let matched_request_ids: std::collections::HashSet<&str> = recall_events
        .iter()
        .filter(|re| !re.accessed_ids.is_empty())
        .map(|re| re.request_id.as_str())
        .collect();

    let mut advance_to: Option<i64> = None;
    let mut rids_we_advanced_through: std::collections::HashSet<String> =
        std::collections::HashSet::new();
    for event in &events {
        let rid = event.request_id.as_deref().unwrap_or("");
        let is_matched = matched_request_ids.contains(rid);
        let is_expired = chrono::DateTime::parse_from_rfc3339(&event.ts)
            .map(|dt| dt.with_timezone(&chrono::Utc) < cutoff)
            .unwrap_or(false);

        if is_matched || is_expired {
            advance_to = Some(event.id);
            rids_we_advanced_through.insert(rid.to_string());
        } else {
            break;
        }
    }

    // Advance the access cursor only through the contiguous id-order prefix
    // of access events whose `request_id` correlates to a recall_complete we
    // also advanced past in this pass. Two constraints force the
    // prefix-safe walk:
    //
    // 1. `peek_access_events` loads up to 500 rows while `peek_recall_events`
    //    caps at 100. If we naively advanced to the MAX correlated id, an
    //    interleaved access event for a not-yet-peeked recall sitting
    //    between two "advanced-through" access events would get silently
    //    consumed — on a later pass when that recall is peeked, its access
    //    signal is already past the cursor and it ages out unlearned.
    //
    // 2. The consumer_offsets row for `alpha_optimizer_access` is a single
    //    monotonic id. We cannot mark id=100 and id=102 consumed while
    //    leaving id=101 unconsumed. The only safe advance is the highest X
    //    such that *every* access event in (prev_offset, X] is in
    //    `rids_we_advanced_through`.
    //
    // Worst case: an orphan access event forever gates the prefix until its
    // recall arrives (or `cleanup_expired_events` prunes it). That is the
    // intended trade-off — correctness over liveness.
    let mut access_advance_to: Option<i64> = None;
    for ae in &access_events {
        let rid = ae.request_id.as_deref().unwrap_or("");
        if rids_we_advanced_through.contains(rid) {
            access_advance_to = Some(ae.id);
        } else {
            break;
        }
    }

    // v0.24 peek+commit migration: instead of advancing both offsets here
    // mid-function, return the pending pair to the orchestrator. The pair
    // is committed atomically (single `commit_offset` call wraps both in
    // BEGIN IMMEDIATE) only after `state.save_snapshot()` succeeds. The
    // intra-pair atomicity invariant from the prior B5 fix is preserved.
    //
    // Codex Tier-B+C round-3 HIGH: bump the in-memory replay watermarks
    // ONLY for the prefix we actually applied (advance_to /
    // access_advance_to), not the raw peek-max. Otherwise a late-
    // arriving access event for a recall we walked past as "orphan"
    // would be filtered out on its next peek and never contribute to
    // alpha learning.
    if let Some(off) = advance_to {
        state.alpha_optimizer_last_id = state.alpha_optimizer_last_id.max(off);
    }
    if let Some(off) = access_advance_to {
        state.alpha_optimizer_access_last_id =
            state.alpha_optimizer_access_last_id.max(off);
    }

    let mut pending: Vec<(&'static str, i64)> = Vec::new();
    if let Some(off) = advance_to {
        pending.push(("alpha_optimizer", off));
    }
    if let Some(off) = access_advance_to {
        pending.push(("alpha_optimizer_access", off));
    }

    if events_with_access.is_empty() {
        tracing::debug!(
            "M2: peeked {} events but none had access data yet (will retry)",
            events.len()
        );
        // Even with no learnable signal yet we still return any pending
        // offset advances — `expired-by-cutoff` events should not loop.
        return if pending.is_empty() { None } else { Some(pending) };
    }

    compute_counterfactual_alphas(&events_with_access, &events, state, config);
    if pending.is_empty() {
        None
    } else {
        Some(pending)
    }
}

/// Non-advancing peek for `recall_access` events. Mirrors `peek_recall_events`
/// but reads against the `alpha_optimizer_access` offset. Separated from
/// `consume_events` so the caller can defer offset advancement until after
/// all side-effects have been committed atomically alongside the
/// `alpha_optimizer` offset (B5 M2 atomicity fix).
fn peek_access_events(
    conn: &rusqlite::Connection,
    limit: usize,
) -> Vec<crate::store::adaptive::StoredEvent> {
    let last_offset: i64 = conn
        .query_row(
            "SELECT last_event_id FROM consumer_offsets WHERE consumer = 'alpha_optimizer_access'",
            [],
            |r| r.get(0),
        )
        .unwrap_or(0);

    match conn.prepare(
        "SELECT id, ts, event_type, request_id, memory_id, concept_id, query, query_type, topic, payload
         FROM feedback_events WHERE id > ?1 AND event_type = 'recall_access'
         ORDER BY id ASC LIMIT ?2",
    ) {
        Ok(mut stmt) => stmt
            .query_map(rusqlite::params![last_offset, limit as i64], |row| {
                Ok(crate::store::adaptive::StoredEvent {
                    id: row.get(0)?,
                    ts: row.get(1)?,
                    event_type: row.get(2)?,
                    request_id: row.get(3)?,
                    memory_id: row.get(4)?,
                    concept_id: row.get(5)?,
                    query: row.get(6)?,
                    query_type: row.get(7)?,
                    topic: row.get(8)?,
                    payload: row.get(9)?,
                })
            })
            .ok()
            .map(|rows| rows.filter_map(|r| r.ok()).collect())
            .unwrap_or_default(),
        Err(_) => Vec::new(),
    }
}

// ===========================================================================
// Reranker weight learning from agent feedback
// ===========================================================================

fn run_reranker_weight_learning(store: &SqliteStore) {
    // v0.25.2 (lifted v0.24.x backlog item): `save_weights` and
    // `commit_offset` are still two separate writes — but the
    // weights row now carries `last_access_event_id` +
    // `last_recall_event_id` watermarks persisted atomically with the
    // weights themselves. If `commit_offset` fails after `save_weights`
    // succeeds, the next pass re-peeks the same events and the
    // watermark filter (`id > weights.last_*_event_id`) drops them →
    // no double-application. The weights row is now the source-of-
    // truth watermark; `consumer_offsets` is a redundant best-effort
    // cache (kept for compatibility / observability).
    //
    // Five invariants for event-sourced state (see
    // `feedback_event_sourced_state_invariant` memory):
    //   1. watermark filter — `events.into_iter().filter(|e| e.id >
    //      prior_*_water)` below.
    //   2. applied-prefix bump — `weights.last_*_event_id =
    //      max(prior, peek_max)` is set *only when `updates > 0`*. Zero-
    //      update passes do NOT bump the weights watermark; `commit_offset`
    //      still bumps the redundant offset cache to drain the FIFO so the
    //      next peek doesn't re-surface the same window forever.
    //   3. replay-drain on startup — implicit. `peek_events` already
    //      filters by `consumer_offsets`; the additional
    //      `id > prior_*_water` filter composes to `id > max(offsets,
    //      weights)`, exactly the spec.
    //   4. CAS merge — `save_weights_cas` predicates the UPDATE on
    //      `(observed_access_id, observed_recall_id)` matching the row's
    //      current values. Concurrent writer wins → we log + skip.
    //   5. per-consumer offset record — kept as best-effort cache via
    //      the existing `commit_offset` calls; weights row dominates.
    let conn = store.conn();

    // Load weights up front so we can capture the prior watermarks
    // BEFORE peeking (invariant #1 + CAS-prep). The weights serve as
    // the source-of-truth replay watermark.
    let mut weights = crate::search::rerank::load_weights(conn);
    let prior_access_water = weights.last_access_event_id;
    let prior_recall_water = weights.last_recall_event_id;

    // Peek feedback events (RecallAccess with source=agent_feedback). Self-
    // contained peek+commit: this helper writes to the `metadata`
    // `rerank_weights` row (NOT `adaptive_state`), so it commits its
    // own consumer offsets.
    // Codex Tier-B+C round-1 MEDIUM: commit each peek's offset *after
    // evaluation* (not gated on `updates > 0`). Filtered-out events have
    // been evaluated → safe to mark consumed; otherwise they stuck
    // behind the 200-row peek limit forever when no agent-feedback rows
    // arrived.
    let raw_access_events = match crate::store::adaptive::peek_events(
        conn,
        "reranker_weights",
        &["recall_access"],
        200,
    ) {
        Ok(evts) => evts,
        Err(e) => {
            tracing::warn!("reranker weight learning: failed to peek events: {e}");
            return;
        }
    };
    // Codex R3 G9 fix: peek BOTH streams up front so the early-return
    // paths below (no agent-feedback rows / no used_ids) can also
    // replay-drain the recall consumer offset, not just access. Without
    // this, a prior successful weights-save with a failed recall
    // commit_offset would leave the recall consumer cursor stale
    // forever; eventually new recall_complete events fall behind the
    // peek window and future feedback can't find their candidate
    // features. `unwrap_or_default()` is best-effort — peek errors
    // simply leave the recall queue undrained until next cycle.
    let raw_recall_events = crate::store::adaptive::peek_events(
        conn,
        "reranker_weights_recall",
        &["recall_complete"],
        100,
    )
    .unwrap_or_default();

    // Capture the RAW peek max BEFORE the watermark filter (Codex R1
    // G1 fix). When `commit_offset` was lost last cycle but the weights
    // row's watermark was already saved, the next peek surfaces those
    // already-applied events. The watermark filter drops them all, so
    // the post-filter `weights_max_id` is None — but we still must
    // advance the consumer offset past them, otherwise the same 200
    // events pin the cursor forever and starve later feedback events
    // behind the peek-window limit.
    let raw_access_max = raw_access_events.last().map(|e| e.id);
    let raw_recall_max = raw_recall_events.last().map(|e| e.id);
    // Replay-only drain target for the recall stream: the highest event
    // id that's STRICTLY ≤ the durable watermark in the weights row
    // (i.e., already applied to a prior cycle's gradient). Early-return
    // paths advance to this id only — they MUST NOT drop post-water
    // events that future feedback events may legitimately need.
    let recall_replay_drain_id = raw_recall_events
        .iter()
        .filter(|e| e.id <= prior_recall_water)
        .map(|e| e.id)
        .max();
    let recall_replay_pair: Option<(&str, i64)> =
        recall_replay_drain_id.map(|id| ("reranker_weights_recall", id));

    // Replay-safety filter (invariant #1): drop events whose gradient
    // effect is already durable in the weights row (commit_offset
    // failed last cycle, those events re-surface on this peek).
    let events: Vec<_> = raw_access_events
        .into_iter()
        .filter(|e| e.id > prior_access_water)
        .collect();
    let weights_max_id = events.last().map(|e| e.id);

    // Filter to agent_feedback source only
    let feedback_events: Vec<_> = events
        .iter()
        .filter(|e| {
            e.payload
                .as_deref()
                .and_then(|p| serde_json::from_str::<serde_json::Value>(p).ok())
                .and_then(|v| {
                    v.get("source")
                        .and_then(|s| s.as_str())
                        .map(|s| s == "agent_feedback")
                })
                .unwrap_or(false)
        })
        .collect();

    // `access_offset_id` falls back to the raw peek max so already-applied
    // events drained by the watermark filter still advance the consumer
    // offset (replay-drain invariant — Codex R1 G1).
    let access_offset_id = weights_max_id.or(raw_access_max);
    let access_offset_pair: Option<(&str, i64)> =
        access_offset_id.map(|id| ("reranker_weights", id));

    // Helper for the early-return paths: commit access advance plus the
    // recall replay-drain in a single atomic call (Codex R3 G9). The
    // recall portion only includes pure replays — post-water recall
    // events stay in queue for future feedback processing.
    let early_return_commit = |conn: &rusqlite::Connection| {
        let mut pairs: Vec<(&str, i64)> = Vec::with_capacity(2);
        if let Some(pair) = access_offset_pair {
            pairs.push(pair);
        }
        if let Some(pair) = recall_replay_pair {
            pairs.push(pair);
        }
        if !pairs.is_empty() {
            let _ = crate::store::adaptive::commit_offset(conn, &pairs);
        }
    };

    if feedback_events.is_empty() {
        // Evaluated this batch (no agent-feedback rows applied) →
        // commit the offset so the same window isn't re-peeked forever.
        // Weights row is unchanged so no watermark bump is needed.
        // (If `commit_offset` fails here, the next pass re-peeks the
        // same empty-filter window and re-evaluates to the same
        // empty-applied outcome → wasted work, no double-apply.)
        early_return_commit(conn);
        return;
    }
    if feedback_events.len() < 10 {
        tracing::debug!(
            events = feedback_events.len(),
            "reranker weight learning: few feedback events, learning with small batch"
        );
    }

    // Collect confirmed-used memory IDs
    let used_ids: std::collections::HashSet<String> = feedback_events
        .iter()
        .filter_map(|e| e.memory_id.clone())
        .collect();

    if used_ids.is_empty() {
        early_return_commit(conn);
        return;
    }

    // Same raw-vs-filtered split as the access stream — `recall_max_id`
    // drives the watermark bump (only events whose gradient was applied
    // count); `recall_offset_id` drives the consumer offset (must
    // advance past already-applied replays too). Recall events were
    // peeked up front for the early-return drain logic above; here we
    // consume them for the actual gradient computation.
    let recall_events: Vec<_> = raw_recall_events
        .into_iter()
        .filter(|e| e.id > prior_recall_water)
        .collect();
    let recall_max_id = recall_events.last().map(|e| e.id);
    let recall_offset_id = recall_max_id.or(raw_recall_max);

    // Build training pairs: for each recall, which candidates were used (positive) vs not (negative)
    // (`weights` already loaded above to capture prior watermarks)
    let lr: f32 = 0.005;
    let mut updates = 0;

    for event in &recall_events {
        let payload = match event
            .payload
            .as_deref()
            .and_then(|p| serde_json::from_str::<serde_json::Value>(p).ok())
        {
            Some(p) => p,
            None => continue,
        };

        let candidates = match payload.get("candidates").and_then(|c| c.as_array()) {
            Some(c) => c,
            None => continue,
        };

        for candidate in candidates {
            let id = match candidate.get("id").and_then(|v| v.as_str()) {
                Some(id) => id,
                None => continue,
            };
            let bm25 = candidate
                .get("bm25_norm")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0) as f32;
            let vec = candidate
                .get("vec_norm")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0) as f32;
            let kg = candidate
                .get("kg_norm")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0) as f32;
            let episode = candidate
                .get("episode_norm")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0) as f32;
            let canonical_support = candidate
                .get("support_count")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32 / (v as f32 + 1.0))
                .unwrap_or(0.0);
            let source_diversity = candidate
                .get("source_diversity")
                .and_then(|v| v.as_f64())
                .map(|v| {
                    let value = v as f32;
                    value / (value + 1.0)
                })
                .unwrap_or(0.0);

            // Target: 1.0 if used by agent, 0.0 if not
            let target = if used_ids.contains(id) { 1.0_f32 } else { 0.0 };
            let predicted = weights.w_fts * bm25
                + weights.w_vec * vec
                + weights.w_kg * kg
                + weights.w_episode * episode
                + weights.w_canonical_support * canonical_support
                + weights.w_source_diversity * source_diversity;
            let error = target - predicted;

            // Capture pre-update learned subtotal BEFORE gradient update.
            let pre_subtotal = weights.w_fts
                + weights.w_vec
                + weights.w_kg
                + weights.w_episode
                + weights.w_canonical_support
                + weights.w_source_diversity;

            // Gradient update for features present in recall_complete candidate payloads.
            weights.w_fts += lr * error * bm25;
            weights.w_vec += lr * error * vec;
            weights.w_kg += lr * error * kg;
            weights.w_episode += lr * error * episode;
            weights.w_canonical_support += lr * error * canonical_support;
            weights.w_source_diversity += lr * error * source_diversity;

            // Renormalize touched weights back to their pre-update subtotal
            // so untouched weights are not affected by the learning step.
            let post_subtotal = weights.w_fts
                + weights.w_vec
                + weights.w_kg
                + weights.w_episode
                + weights.w_canonical_support
                + weights.w_source_diversity;
            if post_subtotal > 0.0 && pre_subtotal > 0.0 {
                let scale = pre_subtotal / post_subtotal;
                weights.w_fts *= scale;
                weights.w_vec *= scale;
                weights.w_kg *= scale;
                weights.w_episode *= scale;
                weights.w_canonical_support *= scale;
                weights.w_source_diversity *= scale;
            }

            updates += 1;
        }
    }

    // Default to "true" so the no-save path (updates == 0) advances the
    // consumer offset normally. Only flipped to false when the CAS
    // write inside `if updates > 0` is rejected by a concurrent writer.
    let mut cas_succeeded = true;

    if updates > 0 {
        // Now normalize ALL weights to sum to 1.0
        let sum = weights.w_fts
            + weights.w_vec
            + weights.w_kg
            + weights.w_recency
            + weights.w_access
            + weights.w_strength
            + weights.w_importance
            + weights.w_keyword
            + weights.w_topic_match
            + weights.w_brevity
            + weights.w_channel_coverage
            + weights.w_canonical_support
            + weights.w_source_diversity
            + weights.w_usage_recency
            + weights.w_connectivity
            + weights.w_concept_richness
            + weights.w_tier_score
            + weights.w_is_current
            + weights.w_episode;
        if sum > 0.0 {
            weights.w_fts /= sum;
            weights.w_vec /= sum;
            weights.w_kg /= sum;
            weights.w_recency /= sum;
            weights.w_access /= sum;
            weights.w_strength /= sum;
            weights.w_importance /= sum;
            weights.w_keyword /= sum;
            weights.w_topic_match /= sum;
            weights.w_brevity /= sum;
            weights.w_channel_coverage /= sum;
            weights.w_canonical_support /= sum;
            weights.w_source_diversity /= sum;
            weights.w_usage_recency /= sum;
            weights.w_connectivity /= sum;
            weights.w_concept_richness /= sum;
            weights.w_tier_score /= sum;
            weights.w_is_current /= sum;
            weights.w_episode /= sum;
        }

        // v0.25.2 invariant #2 — applied-prefix bump. Update the
        // weights row's watermarks to the highest event id whose
        // gradient effect is in the (about-to-be-saved) weights blob.
        // peek-max == applied-prefix here because every event that
        // survived the agent_feedback / used_ids filtering and produced
        // a non-zero gradient is unconditionally absorbed.
        if let Some(id) = weights_max_id {
            weights.last_access_event_id = weights.last_access_event_id.max(id);
        }
        if let Some(id) = recall_max_id {
            weights.last_recall_event_id = weights.last_recall_event_id.max(id);
        }

        // CAS write: predicate on the watermarks observed at load
        // time. If a concurrent worker bumped them already, the UPDATE
        // misses → returns false. Codex R1 G2: we MUST gate the
        // post-evaluation commit on this result, otherwise we'd advance
        // the consumer offsets past events whose gradient never made
        // it into the weights row, permanently dropping them.
        cas_succeeded = crate::search::rerank::save_weights_cas(
            conn,
            &weights,
            prior_access_water,
            prior_recall_water,
        );
        if cas_succeeded {
            tracing::info!(updates, "reranker weights updated from agent feedback");
        } else {
            tracing::warn!(
                updates,
                "reranker weights CAS missed (concurrent worker won the race); \
                 leaving consumer offsets in place so next cycle can replay"
            );
        }
    }

    // Post-evaluation commit. The weights row's watermarks
    // (`last_access_event_id` / `last_recall_event_id`) are the durable
    // source of truth; `consumer_offsets` is a best-effort cache that
    // accelerates future peeks by skipping already-processed events.
    //
    // Three cases:
    //   1. updates == 0 (no gradient applied): `cas_succeeded` stays
    //      `true` (default) → safe to advance, those events won't
    //      contribute on retry either.
    //   2. updates > 0 AND CAS hit: gradient durable → safe to advance.
    //   3. updates > 0 AND CAS missed: gradient NOT durable → must NOT
    //      advance, so next cycle re-peeks the same events and retries
    //      with fresh watermarks (Codex R1 G2 fix).
    //
    // Both offsets are gated together: even though the recall stream
    // doesn't have its own CAS predicate, retrying needs the same
    // recall events available to re-derive the candidate features that
    // drove the gradient.
    let mut pairs: Vec<(&str, i64)> = Vec::with_capacity(2);
    if cas_succeeded {
        if let Some(pair) = access_offset_pair {
            pairs.push(pair);
        }
        if let Some(id) = recall_offset_id {
            pairs.push(("reranker_weights_recall", id));
        }
    }
    if !pairs.is_empty() {
        if let Err(e) = crate::store::adaptive::commit_offset(conn, &pairs) {
            tracing::warn!(
                error = %e,
                "reranker weights: commit_offset failed; events will be re-peeked"
            );
        }
    }
}

// ===========================================================================
// M6: Threshold learning — consume exploration data + co-recall signal
// ===========================================================================

fn run_m6_threshold_learning(
    store: &SqliteStore,
    state: &mut crate::store::adaptive::AdaptiveState,
) -> Option<Vec<(&'static str, i64)>> {
    let conn = store.conn();
    let mut pending: Vec<(&'static str, i64)> = Vec::new();

    // --- Part 1: Peek threshold_exploration events (from randomized A/B test) ---
    // peek+commit: state mutation lands in `state.global_dedup_threshold`
    // which is only durable after `state.save_snapshot()` in
    // `run_adaptive_pipeline`. Caller commits the returned offsets only
    // on save success.
    //
    // v0.25.2 — gate-and-stay (M6 latent watermark-on-peek, flagged in
    // v0.24.0 ship notes). The previous implementation pushed the offset
    // and bumped `m6_threshold_last_id` UNCONDITIONALLY after peek, which
    // silently dropped events 1..N-1 whenever the cycle woke up below the
    // gate threshold (>=10 explore_events). Now both side effects are
    // gated on the same condition as the state mutation: events stay in
    // the queue until the gate fires, matching the v0.24 peek+commit
    // pattern used by `recompute_concept_refresh_stats` and
    // `run_alpha_learning`. Pre-existing watermark filter (`id >
    // prior_threshold_water`) and CAS merge in `save_snapshot` are kept
    // unchanged.
    const M6_THRESHOLD_PEEK_LIMIT: usize = 200;
    let raw_events = crate::store::adaptive::peek_events(
        conn,
        "m6_threshold",
        &["param_update"],
        M6_THRESHOLD_PEEK_LIMIT,
    )
    .unwrap_or_default();
    // Codex R3 G8: capture the saturated-window flag BEFORE the move.
    // Used by the below-gate fallback to break the
    // "explore-contiguous-with-watermark + window full of noise"
    // deadlock that the simple `min_explore_id - 1` rule can't reach.
    let raw_window_saturated = raw_events.len() >= M6_THRESHOLD_PEEK_LIMIT;
    let max_threshold_id = raw_events.last().map(|e| e.id);
    let prior_threshold_water = state.m6_threshold_last_id;
    let post_water_events: Vec<_> = raw_events
        .into_iter()
        .filter(|e| e.id > prior_threshold_water)
        .collect();

    // Filter to threshold_exploration events only. Other consumers also
    // emit `param_update` (vec-dedup cleanup in ops/dedup.rs writes
    // `param_update` rows with no query_type), so the dedicated
    // `m6_threshold` cursor would starve on a steady stream of noise if
    // we waited for the explore-gate to fire. Below we advance past
    // noise-only batches even when the explore gate doesn't fire.
    let explore_events: Vec<_> = post_water_events
        .iter()
        .filter(|e| e.query_type.as_deref() == Some("threshold_exploration"))
        .collect();

    if explore_events.len() >= 10 {
        // Causal inference: compare dedup rates at different thresholds
        // Group by whether threshold was raised or lowered
        let mut raised_dedup = 0u32; // threshold raised (harder to dedup) → was_dedup count
        let mut raised_total = 0u32;
        let mut lowered_dedup = 0u32; // threshold lowered (easier to dedup) → was_dedup count
        let mut lowered_total = 0u32;

        for event in &explore_events {
            let payload: serde_json::Value = event
                .payload
                .as_deref()
                .and_then(|p| serde_json::from_str(p).ok())
                .unwrap_or_default();

            let offset = payload
                .get("offset")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let was_dedup = payload
                .get("was_dedup")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            if offset > 0.01 {
                raised_total += 1;
                if was_dedup {
                    raised_dedup += 1;
                }
            } else if offset < -0.01 {
                lowered_total += 1;
                if was_dedup {
                    lowered_dedup += 1;
                }
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
                state.global_dedup_threshold =
                    (state.global_dedup_threshold + adjustment as f32).clamp(0.40, 0.90);
                tracing::info!(
                    "M6: lowered global threshold to {:.3} (lowered_rate={:.2}, raised_rate={:.2})",
                    state.global_dedup_threshold,
                    lowered_rate,
                    raised_rate
                );
            } else if raised_rate > lowered_rate + 0.15 {
                // Raising threshold still catches duplicates → threshold too low (too aggressive)
                let adjustment = 0.02;
                state.global_dedup_threshold =
                    (state.global_dedup_threshold + adjustment as f32).clamp(0.40, 0.90);
                tracing::info!(
                    "M6: raised global threshold to {:.3} (raised_rate={:.2}, lowered_rate={:.2})",
                    state.global_dedup_threshold,
                    raised_rate,
                    lowered_rate
                );
            } else {
                tracing::debug!(
                    "M6: threshold stable (lowered={:.2}, raised={:.2})",
                    lowered_rate,
                    raised_rate
                );
            }
        }

        // Gate fired → explore events have been *consumed* (the inner
        // raised/lowered branch may or may not have nudged the threshold;
        // either way, re-applying the same batch on the next cycle would
        // compound the count and re-nudge). Bump the watermark to the
        // peeked-max and ask the orchestrator to commit the cursor on
        // save success. Without this paired bump, a `commit_offset`
        // failure after `save_snapshot` would let the same batch nudge
        // the threshold again on the next pass — exactly the issue
        // round-1 HIGH closed for the always-bumped variant.
        if let Some(id) = max_threshold_id {
            state.m6_threshold_last_id = state.m6_threshold_last_id.max(id);
            pending.push(("m6_threshold", id));
        }
    } else if explore_events.is_empty() && max_threshold_id.is_some() {
        // Three sub-cases collapse here, all advancing the cursor with
        // no state effect:
        //
        //  1. Noise-only: this consumer is dedicated to
        //     `threshold_exploration`, so non-exploration `param_update`
        //     rows (e.g. `ops/dedup.rs` cleanup events) have nothing to
        //     do for it. Without this advance, a steady cleanup stream
        //     would push future explore events past the 200-event peek
        //     window forever.
        //
        //  2. Replay-drain (Codex round-1 HIGH analog): a prior pass's
        //     `save_snapshot` succeeded but `commit_offset` failed. The
        //     in-memory `m6_threshold_last_id` was bumped (to W); the
        //     durable cursor is still at 0. This pass re-peeks events
        //     1..W. The `id > prior_threshold_water` filter drops all of
        //     them, so `post_water_events` AND `explore_events` are both
        //     empty — but the durable cursor MUST still advance to W,
        //     otherwise the next pass repeats the same drain and the
        //     events sit behind the cursor forever (until retention
        //     sweeps them). Mirrors `recompute_concept_refresh_stats`,
        //     which always returns `max_id_this_pass` once raw peek is
        //     non-empty.
        //
        //  3. Pure noise + replay-drain mix: same outcome.
        //
        // The watermark bump is a no-op in case 2 (`m6_threshold_last_id`
        // already at W or higher); in cases 1/3 it does real work. The
        // `max_threshold_id.is_some()` guard means we only push when the
        // raw peek returned at least one event — empty raw peeks don't
        // need a commit.
        if let Some(id) = max_threshold_id {
            state.m6_threshold_last_id = state.m6_threshold_last_id.max(id);
            pending.push(("m6_threshold", id));
        }
    } else if !explore_events.is_empty() {
        // Codex R2 G3 (mixed-batch livelock fix): explore_events is
        // non-empty but below the >=10 gate. The 200-row peek window
        // may have filled mostly with noise (cleanup `param_update`
        // rows from `ops/dedup.rs`) plus a small explore tail.
        //
        // Without this branch, the cursor stays put → next peek returns
        // the same 200 rows → gate never fires (livelock under noise
        // dominance). Codex R2 caught the path that Agent 5's self-audit
        // had flagged as out-of-scope.
        //
        // Safe drain rule: advance cursor to `min(explore.id) - 1`. All
        // events strictly BEFORE the first explore in this peek are
        // noise we can permanently drop (this consumer doesn't care
        // about non-exploration `param_update`). The explore events
        // themselves stay in the queue to accumulate with future
        // arrivals on the next cycle. Noise interleaved BETWEEN explores
        // stays for now (the next cycle re-filters and re-drops it,
        // cheap).
        let min_explore_id = explore_events
            .iter()
            .map(|e| e.id)
            .min()
            .expect("explore_events non-empty by branch guard");
        let safe_advance = min_explore_id.saturating_sub(1);
        if safe_advance > prior_threshold_water {
            state.m6_threshold_last_id = state.m6_threshold_last_id.max(safe_advance);
            pending.push(("m6_threshold", safe_advance));
        } else if raw_window_saturated {
            // Codex R3 G8 break-glass: first explore is contiguous with
            // the watermark (`min_explore_id - 1 == prior_threshold_water`)
            // AND the peek window is saturated → noise behind the
            // explore can never accumulate to the >=10 gate, so the
            // standard min-1 rule deadlocks. Break the deadlock by
            // advancing past the whole window, sacrificing up to 9
            // explore signals. The next cycle gets a clean window with
            // new arrivals only. Logged at WARN so the loss is visible
            // in observability — if this fires often, the right fix is
            // a separate consumer key for noise sources, not a wider
            // peek window.
            if let Some(raw_max) = max_threshold_id {
                tracing::warn!(
                    explore_count = explore_events.len(),
                    noise_count = post_water_events.len() - explore_events.len(),
                    advanced_to = raw_max,
                    "M6 below-gate deadlock break: explore signals dropped to advance past saturated window"
                );
                state.m6_threshold_last_id = state.m6_threshold_last_id.max(raw_max);
                pending.push(("m6_threshold", raw_max));
            }
        }
    }
    // else: gate didn't fire AND we have a partial batch of explore
    // events queued (1..=9). Leave the cursor + watermark untouched so
    // the next pass peeks them again alongside future arrivals and
    // (eventually) crosses the threshold.

    // --- Part 2: Co-recall frequency signal ---
    // If two memories always appear together in recall results, they might be duplicates
    // that slipped through dedup (threshold was too high).
    //
    // v0.25.2 — same gate-and-stay fix as Part 1. The outer `>= 5` gate
    // is the one that matters here: when it fires, real side effects
    // happen (UPDATE memories SET needs_vec_dedup = 1, plus
    // global/per-cluster threshold tweaks). Below the gate, leave events
    // queued for the next cycle.
    let raw_recall_events =
        crate::store::adaptive::peek_events(conn, "m6_corecall", &["recall_complete"], 100)
            .unwrap_or_default();
    let max_corecall_id = raw_recall_events.last().map(|e| e.id);
    let prior_corecall_water = state.m6_corecall_last_id;
    let recall_events: Vec<_> = raw_recall_events
        .into_iter()
        .filter(|e| e.id > prior_corecall_water)
        .collect();

    if recall_events.len() >= 5 {
        // Count pair co-occurrences in recall results
        let mut pair_counts: std::collections::HashMap<(String, String), u32> =
            std::collections::HashMap::new();
        let mut mem_counts: std::collections::HashMap<String, u32> =
            std::collections::HashMap::new();

        for event in &recall_events {
            let payload: serde_json::Value = event
                .payload
                .as_deref()
                .and_then(|p| serde_json::from_str(p).ok())
                .unwrap_or_default();

            // Payload is {"candidates": [...], ...} — extract candidate IDs
            let ids: Vec<String> = payload
                .get("candidates")
                .and_then(|c| c.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|c| c.get("id").and_then(|v| v.as_str()).map(|s| s.to_string()))
                        .take(10) // Only top-10 results
                        .collect()
                })
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
        let mut cluster_suspicious: std::collections::HashMap<u32, u32> =
            std::collections::HashMap::new();
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
                            co_count,
                            min_count,
                            sim,
                        );
                        // Mark the newer one for vec dedup (it will be caught in the next sweep)
                        let newer_id = if mem_a.created_at > mem_b.created_at {
                            id_a
                        } else {
                            id_b
                        };
                        let _ = conn.execute(
                            "UPDATE memories SET needs_vec_dedup = 1 WHERE id = ?1",
                            rusqlite::params![newer_id],
                        );
                        suspicious_pairs += 1;

                        // Track per-cluster suspicious counts for threshold adjustment
                        if let Some(cid) = mem_a.cluster_id.or(mem_b.cluster_id) {
                            *cluster_suspicious.entry(cid).or_default() += 1;
                        }
                    }
                }
            }
        }

        // If many co-recall pairs found, threshold is probably too high
        if suspicious_pairs > 0 && event_count >= 10 {
            let pair_ratio = suspicious_pairs as f64 / event_count as f64;
            if pair_ratio > 0.2 {
                state.global_dedup_threshold =
                    (state.global_dedup_threshold - 0.02).clamp(0.40, 0.90);
                tracing::info!(
                    "M6: co-recall signal lowered threshold to {:.3} ({suspicious_pairs} suspicious pairs in {event_count} events)",
                    state.global_dedup_threshold
                );
            }
        }

        // Persist per-cluster threshold adjustments from co-recall signal
        for (cluster_id, count) in &cluster_suspicious {
            if *count >= 2 {
                let current = state
                    .dedup_thresholds
                    .get(cluster_id)
                    .copied()
                    .unwrap_or(state.global_dedup_threshold);
                let adjusted = (current - 0.02).clamp(0.40, 0.90);
                state.dedup_thresholds.insert(*cluster_id, adjusted);
                tracing::info!(
                    "M6: co-recall lowered cluster {cluster_id} threshold {current:.3} → {adjusted:.3} ({count} suspicious pairs)",
                );
            }
        }

        // Outer gate fired → DB writes (`needs_vec_dedup = 1`) plus
        // global / per-cluster threshold tweaks have happened. Bump the
        // watermark to peeked-max and ask the orchestrator to commit
        // the cursor on save success. Without this paired bump,
        // re-applying the same batch would double-count pair occurrences
        // and re-flag the same memories on the next pass.
        if let Some(id) = max_corecall_id {
            state.m6_corecall_last_id = state.m6_corecall_last_id.max(id);
            pending.push(("m6_corecall", id));
        }
    } else if recall_events.is_empty() && max_corecall_id.is_some() {
        // Replay-drain (Codex round-1 HIGH analog, mirrors Part 1):
        // a prior pass's `save_snapshot` succeeded but `commit_offset`
        // failed. The in-memory `m6_corecall_last_id` was bumped (to W);
        // the durable cursor is still at 0. This pass re-peeks events
        // 1..W, the `id > prior_corecall_water` filter drops them all,
        // so `recall_events` is empty BUT raw peek was non-empty. The
        // durable cursor MUST still advance to W, or the events sit
        // behind the cursor forever. The watermark bump is a no-op
        // (state already at W or higher); only the cursor advance has
        // an effect.
        if let Some(id) = max_corecall_id {
            state.m6_corecall_last_id = state.m6_corecall_last_id.max(id);
            pending.push(("m6_corecall", id));
        }
    }
    // else: outer gate didn't fire (1..=4 post-watermark events). Leave
    // cursor + watermark untouched so the next pass picks up the same
    // events alongside fresh arrivals.

    if pending.is_empty() {
        None
    } else {
        Some(pending)
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

    // Pre-fetch content for all sampled members in one batch query to avoid N+1 store.get() calls
    let all_sample_ids: Vec<String> = clusters
        .values()
        .filter(|ids| ids.len() >= 5)
        .flat_map(|ids| ids.iter().take(20).cloned())
        .collect();
    let content_map: std::collections::HashMap<String, String> = all_sample_ids
        .iter()
        .filter_map(|id| store.get(id).ok().map(|m| (id.clone(), m.content)))
        .collect();

    for (cluster_id, mem_ids) in &clusters {
        if mem_ids.len() < 5 {
            continue;
        }

        // Sample up to 20 members to keep computation bounded
        let sample: Vec<&str> = mem_ids.iter().take(20).map(|s| s.as_str()).collect();
        let mut sims: Vec<f32> = Vec::new();

        // Use pre-fetched content
        let contents: Vec<&String> = sample
            .iter()
            .filter_map(|id| content_map.get(*id))
            .collect();

        // Compute pairwise similarities
        for i in 0..contents.len() {
            for j in (i + 1)..contents.len() {
                let sim = crate::extract::similarity(contents[i], contents[j]);
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
            tracing::debug!(
                "A1: cluster {cluster_id} dedup threshold = {threshold:.3} (from {} pairs)",
                sims.len()
            );
        }
    }

    // Update global threshold from all-clusters distribution
    if all_sims.len() >= 10 {
        all_sims.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p90_idx = (all_sims.len() as f64 * 0.90).floor() as usize;
        let p90_idx = p90_idx.min(all_sims.len() - 1);
        let global = all_sims[p90_idx].clamp(0.40, 0.90);
        state.global_dedup_threshold = global;
        tracing::debug!(
            "A1: global dedup threshold = {global:.3} (from {} total pairs)",
            all_sims.len()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ReinConfig;
    use crate::store::adaptive::{self, AdaptiveState, EventType, FeedbackEvent};
    use crate::store::SqliteStore;
    use crate::types::traits::MemoryStore;
    use crate::types::*;
    use chrono::Utc;

    fn test_memory(topic: &str, summary: &str, access_count: u32) -> Memory {
        Memory {
            id: ulid::Ulid::new().to_string(),
            layer: MemoryLayer::LTM,
            topic: topic.to_string(),
            summary: summary.to_string(),
            content: format!("Content about {summary} with some unique words for differentiation"),
            keywords: vec![],
            importance: Importance::Medium,
            source: Source::Manual,
            strength: 1.0,
            decay_lambda: 0.06,
            access_count,
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
            archival_summary: None,
            archival_summary_at: None,
            archival_summary_version: None,
            created_at: Utc::now() - chrono::Duration::days(30),
            updated_at: Utc::now(),
            last_accessed: Utc::now(),
        }
    }

    fn emit(store: &SqliteStore, event: FeedbackEvent) -> i64 {
        adaptive::emit_event(store.conn(), event).unwrap()
    }

    // ── Test 1: run_tiering assigns tiers ────────────────────────────────────

    #[test]
    fn test_run_tiering_assigns_tiers() {
        let store = SqliteStore::in_memory().unwrap();
        let mut config = ReinConfig::default();
        config.adaptive.tier_cold_start = 5; // lower threshold for test

        // Store memories with widely varying access counts and creation dates.
        // Use non-zero low counts to ensure cold_threshold > 0 (the SQL UPDATE
        // is gated by cold_threshold > 0.0).
        let mut ids = Vec::new();
        for i in 0..12u32 {
            let (ac, days_ago) = match i {
                0..=3 => (100 + i * 20, 5i64), // very high access rate
                4..=7 => (5, 30),              // moderate
                _ => (1, 120),                 // very low rate: 1/120 ≈ 0.008
            };
            let mut mem = test_memory("test", &format!("memory {i}"), ac);
            mem.created_at = Utc::now() - chrono::Duration::days(days_ago);
            let id = mem.id.clone();
            store.store(mem).unwrap();
            ids.push(id);
        }

        let mut state = AdaptiveState::default();
        run_tiering(&store, &mut state, &config);

        // Tier boundaries should be set (positive)
        assert!(
            state.hot_threshold > 0.0,
            "hot_threshold should be positive, got {}",
            state.hot_threshold
        );
        assert!(
            state.cold_threshold >= 0.0,
            "cold_threshold should be non-negative, got {}",
            state.cold_threshold
        );
        assert!(
            state.hot_threshold >= state.cold_threshold,
            "hot >= cold: {} vs {}",
            state.hot_threshold,
            state.cold_threshold
        );

        // Verify at least some memories got tier updates in DB
        let hot_count: u32 = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE tier = 'hot'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        let cold_count: u32 = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE tier = 'cold'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        let warm_count: u32 = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE tier = 'warm'",
                [],
                |r| r.get(0),
            )
            .unwrap();

        // With widely varying access rates, tiering should produce at least two tiers
        let tiers_used = (if hot_count > 0 { 1 } else { 0 })
            + (if warm_count > 0 { 1 } else { 0 })
            + (if cold_count > 0 { 1 } else { 0 });
        assert!(
            tiers_used >= 2,
            "tiering should use at least 2 tiers (hot={hot_count}, warm={warm_count}, cold={cold_count})"
        );
    }

    // ── Test 2: build_survival_curves with access data ───────────────────────

    #[test]
    fn test_build_survival_curves_with_access_data() {
        let store = SqliteStore::in_memory().unwrap();

        // Store memories with cluster_id set and varied access patterns
        for i in 0..15 {
            let mut mem = test_memory("survival", &format!("mem {i}"), (i + 1) * 2);
            mem.cluster_id = Some(0); // all in cluster 0
            mem.created_at = Utc::now() - chrono::Duration::days(30 + i as i64);
            mem.last_accessed = Utc::now() - chrono::Duration::days(i as i64);
            let id = mem.id.clone();
            store.store(mem).unwrap();
            // Set cluster_id in DB
            store
                .conn()
                .execute(
                    "UPDATE memories SET cluster_id = 0 WHERE id = ?1",
                    rusqlite::params![&id],
                )
                .unwrap();
        }

        // Build state with cluster assignments
        let mut state = AdaptiveState::default();
        // Read back memory IDs and assign clusters in state
        let mut stmt = store.conn().prepare("SELECT id FROM memories").unwrap();
        let ids: Vec<String> = stmt
            .query_map([], |row| row.get(0))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();
        for id in &ids {
            state.memory_clusters.insert(id.clone(), 0);
        }

        // Should not panic
        build_survival_curves(&store, &state);

        // Check that a survival curve was written to metadata
        let curve_count: u32 = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM metadata WHERE key LIKE 'survival_curve:%'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!(
            curve_count > 0,
            "should have written at least one survival curve, got {curve_count}"
        );
    }

    // ── Test 3: HDBSCAN clustering assigns clusters ──────────────────────────

    #[test]
    fn test_run_hdbscan_clustering_assigns_clusters() {
        let store = SqliteStore::in_memory().unwrap();

        // The in-memory store uses 3072-dim vec_memories table.
        // Store 60 memories and populate with 3072-dim fake embeddings.
        let dim = 3072;
        for i in 0..60 {
            let mem = test_memory("cluster", &format!("cluster mem {i}"), 0);
            let id = mem.id.clone();
            store.store(mem).unwrap();

            // Create fake embedding: two clusters with different base vectors
            let mut embedding = vec![0.0f32; dim];
            if i < 30 {
                // Cluster A
                embedding[0] = 1.0 + (i as f32 * 0.01);
                embedding[1] = 0.5;
            } else {
                // Cluster B
                embedding[0] = -1.0 + ((i - 30) as f32 * 0.01);
                embedding[1] = -0.5;
            }

            crate::store::vec::insert_embedding(store.conn(), &id, &embedding).unwrap();
        }

        let mut state = AdaptiveState::default();
        // Should not panic
        run_hdbscan_clustering(&store, &mut state, 60);

        // HDBSCAN should complete and increment cluster_version
        assert!(
            state.cluster_version > 0,
            "cluster_version should have been incremented"
        );
    }

    // ── Test 4: run_alpha_learning consumes events ───────────────────────────

    #[test]
    fn test_run_alpha_learning_consumes_events() {
        let store = SqliteStore::in_memory().unwrap();
        let config = ReinConfig::default();

        // Emit recall_complete events with candidate payloads
        for i in 0..5 {
            let candidates = serde_json::json!({
                "candidates": [
                    {"id": format!("mem-{i}-a"), "bm25_norm": 0.8, "vec_norm": 0.3},
                    {"id": format!("mem-{i}-b"), "bm25_norm": 0.2, "vec_norm": 0.9},
                ]
            });
            emit(
                &store,
                FeedbackEvent {
                    event_type: EventType::RecallComplete,
                    request_id: Some(format!("req-{i}")),
                    memory_id: None,
                    concept_id: None,
                    query: Some("test query".into()),
                    query_type: Some("semantic".into()),
                    topic: None,
                    payload: Some(candidates),
                },
            );

            // Emit corresponding recall_access events
            emit(
                &store,
                FeedbackEvent {
                    event_type: EventType::RecallAccess,
                    request_id: Some(format!("req-{i}")),
                    memory_id: Some(format!("mem-{i}-a")),
                    concept_id: None,
                    query: None,
                    query_type: None,
                    topic: None,
                    payload: None,
                },
            );
        }

        let mut state = AdaptiveState::default();
        // v0.24 peek+commit: run_alpha_learning returns the pending
        // (consumer, max_id) batch; the caller is responsible for
        // committing after save_snapshot succeeds. The function no
        // longer advances the offset itself.
        let pending = run_alpha_learning(&store, &mut state, &config);
        let pending = pending.expect("alpha_optimizer should report pending offsets after learning");
        assert!(
            pending.iter().any(|(c, off)| *c == "alpha_optimizer" && *off > 0),
            "alpha_optimizer pending offset should be present and > 0, got {pending:?}"
        );

        // Pre-commit: DB offset still at 0.
        let offset_before: i64 = store
            .conn()
            .query_row(
                "SELECT last_event_id FROM consumer_offsets WHERE consumer = 'alpha_optimizer'",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0);
        assert_eq!(
            offset_before, 0,
            "peek+commit: offset must not advance until commit_offset runs"
        );

        // Simulate orchestrator commit after save_snapshot success.
        let pairs: Vec<(&str, i64)> = pending.iter().map(|(c, id)| (*c, *id)).collect();
        crate::store::adaptive::commit_offset(store.conn(), &pairs).unwrap();

        let offset_after: i64 = store
            .conn()
            .query_row(
                "SELECT last_event_id FROM consumer_offsets WHERE consumer = 'alpha_optimizer'",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0);
        assert!(
            offset_after > 0,
            "alpha_optimizer offset should have advanced after commit, got {offset_after}"
        );
    }

    #[test]
    fn test_reranker_weight_learning_uses_canonical_features() {
        let store = SqliteStore::in_memory().unwrap();
        let before = crate::search::rerank::load_weights(store.conn());

        for i in 0..12 {
            emit(
                &store,
                FeedbackEvent {
                    event_type: EventType::RecallComplete,
                    request_id: Some(format!("req-rerank-{i}")),
                    memory_id: None,
                    concept_id: None,
                    query: Some("docker".into()),
                    query_type: Some("semantic".into()),
                    topic: None,
                    payload: Some(serde_json::json!({
                        "candidates": [
                            {
                                "id": format!("used-{i}"),
                                "bm25_norm": 0.4,
                                "vec_norm": 0.4,
                                "kg_norm": 0.1,
                                "episode_norm": 0.1,
                                "support_count": 5,
                                "source_diversity": 3.0
                            },
                            {
                                "id": format!("unused-{i}"),
                                "bm25_norm": 0.4,
                                "vec_norm": 0.4,
                                "kg_norm": 0.1,
                                "episode_norm": 0.1,
                                "support_count": 1,
                                "source_diversity": 1.0
                            }
                        ]
                    })),
                },
            );
            emit(
                &store,
                FeedbackEvent {
                    event_type: EventType::RecallAccess,
                    request_id: Some(format!("req-rerank-{i}")),
                    memory_id: Some(format!("used-{i}")),
                    concept_id: None,
                    query: None,
                    query_type: None,
                    topic: None,
                    payload: Some(serde_json::json!({ "source": "agent_feedback" })),
                },
            );
        }

        run_reranker_weight_learning(&store);
        let after = crate::search::rerank::load_weights(store.conn());

        assert!(after.w_canonical_support > before.w_canonical_support);
        assert!(after.w_source_diversity > before.w_source_diversity);
    }

    /// v0.25.2 — replay-safety: even if `commit_offset` fails after
    /// `save_weights`, a second invocation of the consumer loop must
    /// NOT apply the same gradient step twice. The watermark fields
    /// on the weights row drop the re-peeked events on the second
    /// call.
    #[test]
    fn test_reranker_replay_safe_when_commit_offset_lost() {
        let store = SqliteStore::in_memory().unwrap();

        // Generate the same training corpus the canonical-features
        // test uses (12 RecallComplete + 12 matching RecallAccess).
        for i in 0..12 {
            emit(
                &store,
                FeedbackEvent {
                    event_type: EventType::RecallComplete,
                    request_id: Some(format!("req-replay-{i}")),
                    memory_id: None,
                    concept_id: None,
                    query: Some("docker".into()),
                    query_type: Some("semantic".into()),
                    topic: None,
                    payload: Some(serde_json::json!({
                        "candidates": [
                            {
                                "id": format!("used-{i}"),
                                "bm25_norm": 0.4,
                                "vec_norm": 0.4,
                                "kg_norm": 0.1,
                                "episode_norm": 0.1,
                                "support_count": 5,
                                "source_diversity": 3.0
                            },
                            {
                                "id": format!("unused-{i}"),
                                "bm25_norm": 0.4,
                                "vec_norm": 0.4,
                                "kg_norm": 0.1,
                                "episode_norm": 0.1,
                                "support_count": 1,
                                "source_diversity": 1.0
                            }
                        ]
                    })),
                },
            );
            emit(
                &store,
                FeedbackEvent {
                    event_type: EventType::RecallAccess,
                    request_id: Some(format!("req-replay-{i}")),
                    memory_id: Some(format!("used-{i}")),
                    concept_id: None,
                    query: None,
                    query_type: None,
                    topic: None,
                    payload: Some(serde_json::json!({ "source": "agent_feedback" })),
                },
            );
        }

        // First run: gradient applies, weights row's watermarks bump.
        run_reranker_weight_learning(&store);
        let after_first = crate::search::rerank::load_weights(store.conn());
        assert!(after_first.last_access_event_id > 0);
        assert!(after_first.last_recall_event_id > 0);

        // Simulate `commit_offset` failure: roll the consumer offsets
        // back to 0 so the next peek re-surfaces every event. Without
        // the watermark filter on the weights row this would cause
        // double-application; the watermark filter must drop them.
        store
            .conn()
            .execute(
                "DELETE FROM consumer_offsets
                  WHERE consumer IN ('reranker_weights', 'reranker_weights_recall')",
                [],
            )
            .unwrap();

        // Second run: with consumer offsets reset, peek_events
        // returns ALL events again. Watermark filter on weights row
        // must drop them — weights MUST stay byte-identical.
        run_reranker_weight_learning(&store);
        let after_second = crate::search::rerank::load_weights(store.conn());

        let tol = 1e-9_f32;
        assert!(
            (after_second.w_fts - after_first.w_fts).abs() < tol,
            "w_fts double-applied: first={}, second={}",
            after_first.w_fts,
            after_second.w_fts
        );
        assert!(
            (after_second.w_vec - after_first.w_vec).abs() < tol,
            "w_vec double-applied: first={}, second={}",
            after_first.w_vec,
            after_second.w_vec
        );
        assert!(
            (after_second.w_canonical_support - after_first.w_canonical_support).abs() < tol,
            "w_canonical_support double-applied: first={}, second={}",
            after_first.w_canonical_support,
            after_second.w_canonical_support
        );
        assert!(
            (after_second.w_source_diversity - after_first.w_source_diversity).abs() < tol,
            "w_source_diversity double-applied: first={}, second={}",
            after_first.w_source_diversity,
            after_second.w_source_diversity
        );
        assert_eq!(
            after_second.last_access_event_id, after_first.last_access_event_id,
            "access watermark must not regress on replay"
        );
        assert_eq!(
            after_second.last_recall_event_id, after_first.last_recall_event_id,
            "recall watermark must not regress on replay"
        );
    }

    // ── Test 5: peek_recall_events ───────────────────────────────────────────

    #[test]
    fn test_peek_recall_events() {
        let store = SqliteStore::in_memory().unwrap();

        // Emit 3 RecallComplete events
        for i in 0..3 {
            emit(
                &store,
                FeedbackEvent {
                    event_type: EventType::RecallComplete,
                    request_id: Some(format!("req-{i}")),
                    memory_id: None,
                    concept_id: None,
                    query: Some("test".into()),
                    query_type: Some("semantic".into()),
                    topic: None,
                    payload: Some(serde_json::json!({"candidates": []})),
                },
            );
        }

        // Also emit a non-recall event that should be excluded
        emit(
            &store,
            FeedbackEvent {
                event_type: EventType::Store,
                request_id: None,
                memory_id: Some("m1".into()),
                concept_id: None,
                query: None,
                query_type: None,
                topic: None,
                payload: None,
            },
        );

        // Peek should return 3 recall_complete events
        let events = peek_recall_events(store.conn());
        assert_eq!(events.len(), 3, "should peek 3 recall_complete events");
        for e in &events {
            assert_eq!(e.event_type, "recall_complete");
        }

        // Peek again — offset not advanced, so still returns 3
        let events2 = peek_recall_events(store.conn());
        assert_eq!(
            events2.len(),
            3,
            "peek is non-consuming, should still return 3"
        );

        // Manually advance offset to simulate consumption
        store
            .conn()
            .execute(
                "INSERT INTO consumer_offsets (consumer, last_event_id, updated_at)
                 VALUES ('alpha_optimizer', ?1, strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                 ON CONFLICT(consumer) DO UPDATE SET last_event_id = ?1",
                rusqlite::params![events[2].id],
            )
            .unwrap();

        // After advancing offset past all 3 events, peek returns 0
        let events3 = peek_recall_events(store.conn());
        assert_eq!(
            events3.len(),
            0,
            "after advancing offset, peek should return 0"
        );
    }

    // ── Test 6: compute_counterfactual_alphas ────────────────────────────────

    #[test]
    fn test_compute_counterfactual_alphas() {
        let config = ReinConfig::default();

        // Create synthetic RecallEvent data
        let mut recall_events = Vec::new();
        let mut stored_events = Vec::new();

        for i in 0..15 {
            let mem_a = format!("mem-{i}-a");
            let mem_b = format!("mem-{i}-b");

            let recall_event = crate::search::alpha_optimizer::RecallEvent {
                request_id: format!("req-{i}"),
                candidates: vec![
                    crate::search::alpha_optimizer::CandidateLog {
                        memory_id: mem_a.clone(),
                        bm25_norm: 0.9,
                        vec_norm: 0.2,
                        kg_norm: 0.0,
                        episode_norm: 0.0,
                        support_count: 1,
                        source_diversity: 1.0,
                    },
                    crate::search::alpha_optimizer::CandidateLog {
                        memory_id: mem_b.clone(),
                        bm25_norm: 0.1,
                        vec_norm: 0.95,
                        kg_norm: 0.0,
                        episode_norm: 0.0,
                        support_count: 1,
                        source_diversity: 1.0,
                    },
                ],
                // Agent accessed the BM25-dominant candidate
                accessed_ids: vec![mem_a],
                timestamp: Utc::now() - chrono::Duration::hours(i as i64),
            };
            recall_events.push(recall_event);

            stored_events.push(adaptive::StoredEvent {
                id: (i + 1) as i64,
                ts: (Utc::now() - chrono::Duration::hours(i as i64)).to_rfc3339(),
                event_type: "recall_complete".into(),
                request_id: Some(format!("req-{i}")),
                memory_id: None,
                concept_id: None,
                query: Some("test".into()),
                query_type: Some("semantic".into()),
                topic: None,
                payload: None,
            });
        }

        let mut state = AdaptiveState::default();
        compute_counterfactual_alphas(&recall_events, &stored_events, &mut state, &config);

        // Should have learned a global alpha
        assert!(
            state.learned_alpha.contains_key("global"),
            "should have a global alpha entry"
        );
        let global = &state.learned_alpha["global"];
        assert!(
            (0.0..=1.0).contains(&global.value),
            "global alpha should be in [0, 1], got {}",
            global.value
        );
        assert!(global.sample_count > 0, "sample_count should be positive");
    }

    // ── Test 7: run_m6_threshold_learning ────────────────────────────────────

    #[test]
    fn test_run_m6_threshold_learning() {
        let store = SqliteStore::in_memory().unwrap();

        // Emit threshold_exploration param_update events
        for i in 0..15 {
            let offset = if i % 2 == 0 { 0.05 } else { -0.05 };
            let was_dedup = i % 3 == 0; // some are dedup hits
            emit(
                &store,
                FeedbackEvent {
                    event_type: EventType::ParamUpdate,
                    request_id: None,
                    memory_id: None,
                    concept_id: None,
                    query: None,
                    query_type: Some("threshold_exploration".into()),
                    topic: None,
                    payload: Some(serde_json::json!({
                        "offset": offset,
                        "was_dedup": was_dedup,
                    })),
                },
            );
        }

        // Emit recall_complete events for co-recall signal
        for i in 0..10 {
            emit(
                &store,
                FeedbackEvent {
                    event_type: EventType::RecallComplete,
                    request_id: Some(format!("corecall-{i}")),
                    memory_id: None,
                    concept_id: None,
                    query: Some("test query".into()),
                    query_type: None,
                    topic: None,
                    payload: Some(serde_json::json!({
                        "candidates": [
                            {"id": "shared-a", "bm25_norm": 0.5, "vec_norm": 0.5},
                            {"id": "shared-b", "bm25_norm": 0.4, "vec_norm": 0.6},
                            {"id": format!("unique-{i}"), "bm25_norm": 0.3, "vec_norm": 0.3},
                        ]
                    })),
                },
            );
        }

        let mut state = AdaptiveState {
            global_dedup_threshold: 0.70, // mirror serde default
            ..AdaptiveState::default()
        };

        // Should not panic
        run_m6_threshold_learning(&store, &mut state);

        // The function ran to completion — threshold may or may not have changed
        // depending on the distribution, but it should still be in valid range
        assert!(
            (0.40..=0.90).contains(&state.global_dedup_threshold),
            "global_dedup_threshold should be in [0.40, 0.90], got {}",
            state.global_dedup_threshold
        );
    }

    // ── v0.25.2 M6 gate-and-stay regression suite ───────────────────────────
    // The pre-v0.25.2 implementation pushed both `m6_threshold` /
    // `m6_corecall` offsets and bumped both watermarks UNCONDITIONALLY
    // after `peek_events`. When the gate (`>=10` for explore, `>=5` for
    // co-recall) didn't fire, the cursor still advanced after
    // `save_snapshot`, silently dropping events that should have stayed
    // queued for the next cycle. These tests pin the new behavior:
    //  - below the gate → no offset returned, no watermark bump, events
    //    stay re-peekable on the next call;
    //  - above the gate → offset returned, watermark bumped to peeked-max;
    //  - noise-only batch → offset advanced past the noise (so the
    //    dedicated `m6_threshold` cursor isn't starved by `param_update`
    //    rows from `ops/dedup.rs` cleanup events).

    /// Helper: read the cursor stored in `consumer_offsets` for a consumer.
    /// Returns 0 when the consumer has never been written.
    fn read_offset(store: &SqliteStore, consumer: &str) -> i64 {
        store
            .conn()
            .query_row(
                "SELECT last_event_id FROM consumer_offsets WHERE consumer = ?1",
                rusqlite::params![consumer],
                |row| row.get(0),
            )
            .unwrap_or(0)
    }

    /// Helper: emit N threshold-exploration events and return the last
    /// inserted row id.
    fn emit_explore_events(store: &SqliteStore, count: u32) -> i64 {
        let mut last_id = 0;
        for i in 0..count {
            let offset = if i % 2 == 0 { 0.05 } else { -0.05 };
            last_id = emit(
                store,
                FeedbackEvent {
                    event_type: EventType::ParamUpdate,
                    request_id: None,
                    memory_id: None,
                    concept_id: None,
                    query: None,
                    query_type: Some("threshold_exploration".into()),
                    topic: None,
                    payload: Some(serde_json::json!({
                        "offset": offset,
                        "was_dedup": i % 3 == 0,
                    })),
                },
            );
        }
        last_id
    }

    #[test]
    fn m6_below_threshold_does_not_commit() {
        let store = SqliteStore::in_memory().unwrap();
        emit_explore_events(&store, 5); // < 10 → gate must NOT fire
        let mut state = AdaptiveState {
            global_dedup_threshold: 0.70,
            ..AdaptiveState::default()
        };

        let result = run_m6_threshold_learning(&store, &mut state);
        // Watermark must remain at the default (0).
        assert_eq!(
            state.m6_threshold_last_id, 0,
            "watermark must not bump when explore gate didn't fire"
        );
        // No `m6_threshold` consumer pair returned for the orchestrator
        // to commit.
        let has_threshold_pair = result
            .as_deref()
            .map(|p| p.iter().any(|(c, _)| *c == "m6_threshold"))
            .unwrap_or(false);
        assert!(
            !has_threshold_pair,
            "m6_threshold cursor must not be in the pending batch when gate didn't fire"
        );

        // Belt-and-braces: simulate the orchestrator committing whatever
        // we returned. The cursor row must still be absent / at 0,
        // proving the events stay re-peekable.
        if let Some(batch) = result {
            let pairs: Vec<(&str, i64)> = batch.iter().map(|(c, id)| (*c, *id)).collect();
            crate::store::adaptive::commit_offset(store.conn(), &pairs).unwrap();
        }
        assert_eq!(
            read_offset(&store, "m6_threshold"),
            0,
            "consumer cursor must stay at 0 — events remain re-peekable next pass"
        );
    }

    #[test]
    fn m6_above_threshold_commits_after_save() {
        let store = SqliteStore::in_memory().unwrap();
        let last_id = emit_explore_events(&store, 12); // >= 10 → gate fires
        let mut state = AdaptiveState {
            global_dedup_threshold: 0.70,
            ..AdaptiveState::default()
        };

        let result = run_m6_threshold_learning(&store, &mut state);
        // Watermark bumped to peeked-max.
        assert_eq!(
            state.m6_threshold_last_id, last_id,
            "watermark must bump to peeked-max when gate fires"
        );

        // Pending batch carries the m6_threshold cursor at peeked-max.
        let batch = result.expect("gate fired → pending batch must be Some");
        let pair = batch
            .iter()
            .find(|(c, _)| *c == "m6_threshold")
            .expect("m6_threshold cursor must be in the pending batch");
        assert_eq!(pair.1, last_id, "pending offset must be the peeked-max id");

        // Simulate the orchestrator committing on save success →
        // cursor advances and a re-peek returns no new events.
        let pairs: Vec<(&str, i64)> = batch.iter().map(|(c, id)| (*c, *id)).collect();
        crate::store::adaptive::commit_offset(store.conn(), &pairs).unwrap();
        assert_eq!(read_offset(&store, "m6_threshold"), last_id);
        let re_peek = crate::store::adaptive::peek_events(
            store.conn(),
            "m6_threshold",
            &["param_update"],
            200,
        )
        .unwrap();
        assert!(re_peek.is_empty(), "no events should remain after commit");
    }

    #[test]
    fn m6_save_failure_leaves_offset_alone() {
        // Simulates the orchestrator path where `save_snapshot` fails
        // after `run_m6_threshold_learning` returns: the orchestrator
        // discards the pending batch (no `commit_offset` call), so the
        // cursor stays put and the next pass re-peeks the same events
        // and re-applies the watermark filter. This must not advance
        // the durable `consumer_offsets` row even if the in-memory
        // `state.m6_threshold_last_id` was bumped.
        let store = SqliteStore::in_memory().unwrap();
        let last_id = emit_explore_events(&store, 12);
        let mut state = AdaptiveState {
            global_dedup_threshold: 0.70,
            ..AdaptiveState::default()
        };

        // Run the helper but do NOT propagate the pending batch to
        // `commit_offset` (the orchestrator's save-failure branch).
        let _result = run_m6_threshold_learning(&store, &mut state);

        // Durable cursor is untouched — the event log and the cursor
        // row are the source of truth, the in-memory watermark bump is
        // only durable once `save_snapshot` succeeds.
        assert_eq!(
            read_offset(&store, "m6_threshold"),
            0,
            "consumer cursor must stay at 0 when save_snapshot would have failed"
        );

        // Re-peek (as the next pipeline pass would): events are still
        // returned, and the prior in-memory watermark of 0 means they
        // pass the `id > prior_threshold_water` filter. The next pass
        // can re-apply the gate cleanly.
        let re_peek = crate::store::adaptive::peek_events(
            store.conn(),
            "m6_threshold",
            &["param_update"],
            200,
        )
        .unwrap();
        assert_eq!(re_peek.len(), 12);
        assert_eq!(re_peek.last().unwrap().id, last_id);
    }

    #[test]
    fn m6_noise_only_advances_past_cleanup_events() {
        // ops/dedup.rs emits `param_update` rows with no query_type as
        // part of vec-dedup cleanup. M6's dedicated `m6_threshold`
        // cursor must advance past those even when the explore gate
        // doesn't fire, otherwise a steady cleanup-event stream pushes
        // future explore events past the 200-event peek window forever.
        let store = SqliteStore::in_memory().unwrap();
        let mut last_noise_id = 0;
        for i in 0..15 {
            last_noise_id = emit(
                &store,
                FeedbackEvent {
                    event_type: EventType::ParamUpdate,
                    request_id: None,
                    memory_id: None,
                    concept_id: None,
                    query: None,
                    query_type: None, // ← NOT "threshold_exploration"
                    topic: None,
                    payload: Some(serde_json::json!({"source": "dedup", "duplicates_merged": i})),
                },
            );
        }
        let mut state = AdaptiveState {
            global_dedup_threshold: 0.70,
            ..AdaptiveState::default()
        };

        let result = run_m6_threshold_learning(&store, &mut state);
        assert_eq!(
            state.m6_threshold_last_id, last_noise_id,
            "watermark must advance past noise-only batch"
        );
        let batch = result.expect("noise-only batch must still return a pending offset");
        let pair = batch
            .iter()
            .find(|(c, _)| *c == "m6_threshold")
            .expect("m6_threshold cursor must be in the pending batch (noise advance)");
        assert_eq!(pair.1, last_noise_id);
        // global_dedup_threshold stays at the seed value — no explore
        // events means no nudge.
        assert!((state.global_dedup_threshold - 0.70).abs() < f32::EPSILON);
    }

    #[test]
    fn m6_corecall_below_gate_does_not_commit() {
        let store = SqliteStore::in_memory().unwrap();
        // Emit 3 recall_complete events — below the outer `>= 5` gate.
        for i in 0..3 {
            emit(
                &store,
                FeedbackEvent {
                    event_type: EventType::RecallComplete,
                    request_id: Some(format!("rid-{i}")),
                    memory_id: None,
                    concept_id: None,
                    query: Some("q".into()),
                    query_type: None,
                    topic: None,
                    payload: Some(serde_json::json!({
                        "candidates": [
                            {"id": "a"}, {"id": "b"},
                        ]
                    })),
                },
            );
        }
        let mut state = AdaptiveState {
            global_dedup_threshold: 0.70,
            ..AdaptiveState::default()
        };

        let result = run_m6_threshold_learning(&store, &mut state);
        assert_eq!(
            state.m6_corecall_last_id, 0,
            "corecall watermark must not bump below the >=5 gate"
        );
        let has_corecall_pair = result
            .as_deref()
            .map(|p| p.iter().any(|(c, _)| *c == "m6_corecall"))
            .unwrap_or(false);
        assert!(
            !has_corecall_pair,
            "m6_corecall cursor must not be in the pending batch below the >=5 gate"
        );

        // Simulate the orchestrator committing whatever we returned.
        if let Some(batch) = result {
            let pairs: Vec<(&str, i64)> = batch.iter().map(|(c, id)| (*c, *id)).collect();
            crate::store::adaptive::commit_offset(store.conn(), &pairs).unwrap();
        }
        assert_eq!(read_offset(&store, "m6_corecall"), 0);
    }

    #[test]
    fn m6_replay_empty_drains_cursor() {
        // Simulates the Codex round-1 HIGH scenario for M6: a prior
        // pass's `save_snapshot` succeeded (in-memory watermark bumped)
        // but `commit_offset` failed (durable cursor still at 0). The
        // next pass re-peeks the same events, the watermark filter
        // drops them all, and a naive "events.is_empty() → return
        // None" branch would livelock the cursor at 0 forever (until
        // event retention sweeps the rows). The helper must still
        // surface the peeked-max id in `pending` so the orchestrator
        // advances the durable cursor on save success.
        let store = SqliteStore::in_memory().unwrap();
        let last_id = emit_explore_events(&store, 12);
        // Pre-load both M6 watermarks as if a prior pass had bumped them
        // and the orchestrator's commit_offset had crashed before
        // landing the cursor write.
        let mut state = AdaptiveState {
            global_dedup_threshold: 0.70,
            m6_threshold_last_id: last_id,
            ..AdaptiveState::default()
        };

        let result = run_m6_threshold_learning(&store, &mut state);
        // Watermark unchanged (already at last_id; max is a no-op).
        assert_eq!(state.m6_threshold_last_id, last_id);
        // Pending batch carries the peeked-max id so the orchestrator
        // can drain the stale cursor.
        let batch = result.expect("replay-drain → pending must be Some");
        let pair = batch
            .iter()
            .find(|(c, _)| *c == "m6_threshold")
            .expect("m6_threshold cursor must be in the pending batch (replay-drain)");
        assert_eq!(pair.1, last_id);

        // Simulate the orchestrator committing → durable cursor now
        // matches the in-memory watermark and a re-peek returns 0.
        let pairs: Vec<(&str, i64)> = batch.iter().map(|(c, id)| (*c, *id)).collect();
        crate::store::adaptive::commit_offset(store.conn(), &pairs).unwrap();
        let re_peek = crate::store::adaptive::peek_events(
            store.conn(),
            "m6_threshold",
            &["param_update"],
            200,
        )
        .unwrap();
        assert!(re_peek.is_empty());
    }

    #[test]
    fn m6_corecall_replay_empty_drains_cursor() {
        // Same Codex round-1 HIGH analog for the corecall consumer.
        let store = SqliteStore::in_memory().unwrap();
        let mut last_id = 0;
        for i in 0..6 {
            last_id = emit(
                &store,
                FeedbackEvent {
                    event_type: EventType::RecallComplete,
                    request_id: Some(format!("rid-{i}")),
                    memory_id: None,
                    concept_id: None,
                    query: Some("q".into()),
                    query_type: None,
                    topic: None,
                    payload: Some(serde_json::json!({
                        "candidates": [{"id": "a"}, {"id": "b"}]
                    })),
                },
            );
        }
        let mut state = AdaptiveState {
            global_dedup_threshold: 0.70,
            m6_corecall_last_id: last_id,
            ..AdaptiveState::default()
        };

        let result = run_m6_threshold_learning(&store, &mut state);
        assert_eq!(state.m6_corecall_last_id, last_id);
        let batch = result.expect("replay-drain → pending must be Some");
        let pair = batch
            .iter()
            .find(|(c, _)| *c == "m6_corecall")
            .expect("m6_corecall cursor must be in the pending batch (replay-drain)");
        assert_eq!(pair.1, last_id);

        let pairs: Vec<(&str, i64)> = batch.iter().map(|(c, id)| (*c, *id)).collect();
        crate::store::adaptive::commit_offset(store.conn(), &pairs).unwrap();
        let re_peek = crate::store::adaptive::peek_events(
            store.conn(),
            "m6_corecall",
            &["recall_complete"],
            100,
        )
        .unwrap();
        assert!(re_peek.is_empty());
    }

    #[test]
    fn m6_corecall_above_gate_commits_after_save() {
        let store = SqliteStore::in_memory().unwrap();
        // Emit 6 recall_complete events — fires the outer `>= 5` gate
        // (real DB writes for `needs_vec_dedup` may or may not happen
        // depending on co-recall ratios; we only assert on the cursor
        // bookkeeping here).
        let mut last_id = 0;
        for i in 0..6 {
            last_id = emit(
                &store,
                FeedbackEvent {
                    event_type: EventType::RecallComplete,
                    request_id: Some(format!("rid-{i}")),
                    memory_id: None,
                    concept_id: None,
                    query: Some("q".into()),
                    query_type: None,
                    topic: None,
                    payload: Some(serde_json::json!({
                        "candidates": [
                            {"id": "a"}, {"id": "b"},
                        ]
                    })),
                },
            );
        }
        let mut state = AdaptiveState {
            global_dedup_threshold: 0.70,
            ..AdaptiveState::default()
        };

        let result = run_m6_threshold_learning(&store, &mut state);
        assert_eq!(
            state.m6_corecall_last_id, last_id,
            "corecall watermark must bump to peeked-max when outer gate fires"
        );
        let batch = result.expect("outer gate fired → pending batch must be Some");
        let pair = batch
            .iter()
            .find(|(c, _)| *c == "m6_corecall")
            .expect("m6_corecall cursor must be in the pending batch");
        assert_eq!(pair.1, last_id);

        let pairs: Vec<(&str, i64)> = batch.iter().map(|(c, id)| (*c, *id)).collect();
        crate::store::adaptive::commit_offset(store.conn(), &pairs).unwrap();
        assert_eq!(read_offset(&store, "m6_corecall"), last_id);
    }

    /// Regression: M2 alpha learning must (a) advance both offsets in one
    /// transaction AND (b) advance the access cursor only through the
    /// contiguous id-order prefix of access events whose recall_complete is
    /// also in this pass's advance prefix — NOT just `max(correlated_id)`.
    ///
    /// Fixture (exercises both the trailing-orphan and interleaved-orphan
    /// hazards Codex flagged):
    /// - rid-A: RecallComplete at id=1, RecallAccess at id=2. Learned.
    /// - rid-B: RecallAccess at id=3 with NO matching RecallComplete yet.
    /// - rid-C: RecallComplete at id=4, RecallAccess at id=5. Learned.
    ///
    /// A naive `max()` over correlated rids would advance access cursor to
    /// id=5, silently consuming rid-B's access event at id=3. The
    /// prefix-safe walk must stop at id=2 (A's access), because id=3 (B's)
    /// is the first non-advanced rid in the sequence.
    #[test]
    fn run_alpha_learning_advances_cursors_correlated_not_past_unseen_recalls() {
        let store = SqliteStore::in_memory().unwrap();
        let mut config = ReinConfig::default();
        config.adaptive.enabled = true;

        // --- rid-A: full pair (will be matched + learned) ---
        let rid_a = "req-atomic-A".to_string();
        emit(
            &store,
            FeedbackEvent {
                event_type: EventType::RecallComplete,
                request_id: Some(rid_a.clone()),
                memory_id: None,
                concept_id: None,
                query: Some("q".into()),
                query_type: Some("Semantic".into()),
                topic: None,
                payload: Some(serde_json::json!({
                    "candidates": [{
                        "id": "mem-1",
                        "bm25_norm": 0.8,
                        "vec_norm": 0.6,
                        "kg_norm": 0.0,
                        "episode_norm": 0.0,
                        "support_count": 1,
                        "source_diversity": 1.0,
                    }],
                    "cc_alpha": 0.5,
                })),
            },
        );
        let access_a_id = emit(
            &store,
            FeedbackEvent {
                event_type: EventType::RecallAccess,
                request_id: Some(rid_a.clone()),
                memory_id: Some("mem-1".into()),
                concept_id: None,
                query: None,
                query_type: None,
                topic: None,
                payload: None,
            },
        );

        // --- rid-B: orphan access event BETWEEN A's and C's events ---
        // This is the exact case a `max(correlated_id)` strategy would miss:
        // B's rid is not in advance_through, but a naive max would skip B
        // entirely and land on C's access event id, silently consuming B's
        // access signal before B's recall_complete ever arrives.
        emit(
            &store,
            FeedbackEvent {
                event_type: EventType::RecallAccess,
                request_id: Some("req-atomic-B".into()),
                memory_id: Some("mem-2".into()),
                concept_id: None,
                query: None,
                query_type: None,
                topic: None,
                payload: None,
            },
        );

        // --- rid-C: full pair arriving AFTER B's orphan access ---
        let rid_c = "req-atomic-C".to_string();
        emit(
            &store,
            FeedbackEvent {
                event_type: EventType::RecallComplete,
                request_id: Some(rid_c.clone()),
                memory_id: None,
                concept_id: None,
                query: Some("q2".into()),
                query_type: Some("Semantic".into()),
                topic: None,
                payload: Some(serde_json::json!({
                    "candidates": [{
                        "id": "mem-3",
                        "bm25_norm": 0.7,
                        "vec_norm": 0.5,
                        "kg_norm": 0.0,
                        "episode_norm": 0.0,
                        "support_count": 1,
                        "source_diversity": 1.0,
                    }],
                    "cc_alpha": 0.5,
                })),
            },
        );
        let _access_c_id = emit(
            &store,
            FeedbackEvent {
                event_type: EventType::RecallAccess,
                request_id: Some(rid_c.clone()),
                memory_id: Some("mem-3".into()),
                concept_id: None,
                query: None,
                query_type: None,
                topic: None,
                payload: None,
            },
        );

        let mut state = AdaptiveState::default();
        // v0.24 peek+commit: pending offsets returned, must commit
        // to land them in DB.
        let pending = run_alpha_learning(&store, &mut state, &config)
            .expect("pending offsets expected when alpha_optimizer has work to do");
        let pairs: Vec<(&str, i64)> = pending.iter().map(|(c, id)| (*c, *id)).collect();
        crate::store::adaptive::commit_offset(store.conn(), &pairs).unwrap();

        let alpha_off: i64 = store
            .conn()
            .query_row(
                "SELECT last_event_id FROM consumer_offsets WHERE consumer = 'alpha_optimizer'",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0);
        let access_off: i64 = store
            .conn()
            .query_row(
                "SELECT last_event_id FROM consumer_offsets \
                  WHERE consumer = 'alpha_optimizer_access'",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0);

        assert!(alpha_off > 0, "alpha_optimizer offset must advance");
        // Access cursor must stop at rid-A's access event.
        //
        // The prefix-safe contract: walk access events in id-order,
        // advance through the contiguous prefix where every rid is in
        // `rids_we_advanced_through`. In this fixture the sequence is
        // [A@access_a_id, B@..., C@...]. A is advanced through, B is not
        // (no recall_complete yet), so the walk stops at A.
        //
        // A buggy `max(correlated_id)` advance would land on C's access
        // event id, silently swallowing B's access event at position 2.
        // On a later pass when B's recall_complete finally arrives, B's
        // access signal would already be past the cursor and B would age
        // out as "no access data", never contributing to alpha learning.
        assert_eq!(
            access_off, access_a_id,
            "alpha_optimizer_access must stop at rid-A's access (prefix-safe), \
             not advance to rid-C's access past rid-B's orphan. \
             A `max(correlated_id)` strategy would fail this assertion."
        );
    }

}
