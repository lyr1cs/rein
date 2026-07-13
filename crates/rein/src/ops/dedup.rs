//! Dedup operations: merge, provenance-preserving dedup, embedding-based dedup,
//! and cleanup event emission.

use crate::config::ReinConfig;
use crate::store::SqliteStore;
use crate::types::*;

use super::adaptive::run_adaptive_pipeline;
use super::consolidation::{load_group_memories, TopicGroup};
use super::stronger_tier;

const VEC_DEDUP_PENDING_LIMIT: usize = 50;

type VecDedupItem = (String, String, String, String);

pub(crate) fn emit_cleanup_event(
    store: &SqliteStore,
    event_type: crate::store::adaptive::EventType,
    memory_id: Option<String>,
    topic: Option<String>,
    payload: serde_json::Value,
) {
    let _ = crate::store::adaptive::emit_event(
        store.conn(),
        crate::store::adaptive::FeedbackEvent {
            event_type,
            request_id: None,
            memory_id,
            concept_id: None,
            query: None,
            query_type: Some("cleanup".to_string()),
            topic,
            payload: Some(payload),
        },
    );
}

fn record_dedup_artifacts(
    store: &SqliteStore,
    winner_id: &str,
    loser: &Memory,
    relation: DedupRelation,
    lexical_score: Option<f32>,
    embedding_score: Option<f32>,
    reason: &str,
    payload: Option<serde_json::Value>,
) {
    let canonical_id = store
        .canonical_id_for(winner_id)
        .unwrap_or_else(|_| winner_id.to_string());
    if let Err(e) = store.snapshot_memory_as_evidence(&canonical_id, loser) {
        tracing::warn!("dedup: failed to snapshot evidence for {}: {e}", loser.id);
    }
    let merged_summary = store.get(winner_id).ok().map(|memory| memory.summary);
    if let Err(e) = store.record_dedup_decision(DedupDecision {
        id: String::new(),
        winner_id: Some(winner_id.to_string()),
        loser_id: Some(loser.id.clone()),
        canonical_id: Some(canonical_id),
        lexical_score,
        embedding_score,
        relation,
        confidence: lexical_score
            .or(embedding_score)
            .unwrap_or(0.8)
            .clamp(0.0, 1.0),
        reason: reason.to_string(),
        operator: "auto".to_string(),
        reversible: true,
        merged_summary,
        novel_facts: vec![],
        conflict_detected: matches!(relation, DedupRelation::Update),
        payload,
        created_at: chrono::Utc::now(),
    }) {
        tracing::warn!(
            "dedup: failed to record decision for loser {}: {e}",
            loser.id
        );
    }
}

fn merge_memory_into_winner(
    store: &SqliteStore,
    config: Option<&ReinConfig>,
    winner_id: &str,
    loser: &Memory,
    lexical_score: Option<f32>,
    embedding_score: Option<f32>,
    reason: &str,
    payload: Option<serde_json::Value>,
) -> ReinResult<()> {
    // Wrap in SAVEPOINT for atomicity: if any step fails (update, mark_superseded,
    // evidence snapshot), the entire merge is rolled back to prevent partial state
    // (e.g. winner updated but loser provenance lost).
    store.conn().execute_batch("SAVEPOINT merge_winner")?;
    let result = (|| -> ReinResult<()> {
        let mut winner = match store.get(winner_id) {
            Ok(w) => w,
            Err(e) => {
                tracing::warn!("merge: winner {winner_id} not found: {e}");
                return Err(e);
            }
        };
        // Prefer LLM-computed novel_facts over mechanical unique-lines extraction.
        // novel_facts are produced by llm_dedup_verdict and stored in payload JSON.
        let llm_novel: Option<Vec<String>> = payload
            .as_ref()
            .and_then(|p| p.get("novel_facts"))
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            })
            .filter(|facts: &Vec<String>| !facts.is_empty());

        if let Some(facts) = llm_novel {
            winner.content.push_str(&format!(
                "\n\n[merged from {} on {}]\n{}",
                loser.id,
                loser.created_at.format("%Y-%m-%d"),
                facts.join("\n"),
            ));
        } else {
            let unique = extract_unique_lines(&loser.content, &winner.content);
            if !unique.is_empty() {
                winner.content.push_str(&format!(
                    "\n\n[merged from {} on {}]\n{}",
                    loser.id,
                    loser.created_at.format("%Y-%m-%d"),
                    unique,
                ));
            }
        }

        // Use LLM merged_summary for the winner's summary if provided.
        let llm_summary = payload
            .as_ref()
            .and_then(|p| p.get("merged_summary"))
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        for kw in &loser.keywords {
            if !winner.keywords.contains(kw) {
                winner.keywords.push(kw.clone());
            }
        }
        winner.access_count = winner
            .access_count
            .saturating_add(loser.access_count)
            .saturating_add(1);
        winner.strength = (winner.strength + 0.1).min(1.0);
        winner.importance = winner.importance.max(loser.importance);
        winner.layer = winner.importance.auto_layer();
        winner.decay_lambda = winner.decay_lambda.min(loser.decay_lambda);
        winner.tier = stronger_tier(winner.tier, loser.tier);
        winner.last_accessed = winner.last_accessed.max(loser.last_accessed);
        winner.updated_at = chrono::Utc::now();
        winner.summary = llm_summary.unwrap_or_else(|| {
            winner
                .content
                .chars()
                .take(crate::types::SUMMARY_MAX_CHARS)
                .collect()
        });
        store.update(&winner)?;
        store.mark_superseded(&loser.id, winner_id)?;
        record_dedup_artifacts(
            store,
            winner_id,
            loser,
            DedupRelation::Duplicate,
            lexical_score,
            embedding_score,
            reason,
            payload,
        );
        Ok(())
    })();
    match result {
        Ok(()) => {
            store.conn().execute_batch("RELEASE merge_winner")?;
            // Queue async LLM synthesis pass to collapse merged blocks into coherent prose.
            if let Some(cfg) = config {
                crate::extract::hooks::queue::queue_merge_refinement_job(
                    cfg,
                    winner_id.to_string(),
                );
            }
            Ok(())
        }
        Err(e) => {
            let _ = store.conn().execute_batch("ROLLBACK TO merge_winner");
            let _ = store.conn().execute_batch("RELEASE merge_winner");
            Err(e)
        }
    }
}

pub async fn resolve_dedup_job_async(
    store: &SqliteStore,
    config: &ReinConfig,
    existing_id: &str,
    new_id: &str,
    lexical_score: Option<f32>,
    reason: &str,
) -> ReinResult<DedupRelation> {
    let existing = match store.get(existing_id) {
        Ok(memory) => memory,
        Err(_) => return Ok(DedupRelation::Distinct),
    };
    let new_memory = match store.get(new_id) {
        Ok(memory) => memory,
        Err(_) => return Ok(DedupRelation::Distinct),
    };

    if existing.superseded_by.is_some() || new_memory.superseded_by.is_some() {
        return Ok(DedupRelation::Distinct);
    }

    let existing_canonical = store.canonical_id_for(existing_id)?;
    let new_canonical = store.canonical_id_for(new_id)?;
    if existing_canonical == new_canonical {
        return Ok(DedupRelation::Duplicate);
    }

    let verdict = match crate::extract::llm::llm_dedup_verdict(
        config,
        &existing.content,
        &new_memory.content,
    )
    .await
    {
        Ok(Some(v)) => v,
        Ok(None) => {
            tracing::debug!("LLM dedup verdict returned None, falling back to Distinct");
            crate::extract::llm::DedupVerdict {
                relation: DedupRelation::Distinct,
                confidence: 0.5,
                merged_summary: String::new(),
                novel_facts: vec![],
                conflict_detected: false,
                suggested_topic: None,
            }
        }
        Err(e) => {
            tracing::warn!("LLM dedup verdict failed, falling back to Distinct: {e}");
            crate::extract::llm::DedupVerdict {
                relation: DedupRelation::Distinct,
                confidence: 0.5,
                merged_summary: String::new(),
                novel_facts: vec![],
                conflict_detected: false,
                suggested_topic: None,
            }
        }
    };
    let payload = Some(serde_json::json!({
        "confidence": verdict.confidence,
        "merged_summary": verdict.merged_summary,
        "novel_facts": verdict.novel_facts,
        "suggested_topic": verdict.suggested_topic,
        "conflict_detected": verdict.conflict_detected,
        "reason": reason,
    }));

    match verdict.relation {
        DedupRelation::Duplicate => {
            merge_memory_into_winner(
                store,
                Some(config),
                existing_id,
                &new_memory,
                lexical_score,
                None,
                reason,
                payload,
            )?;
            Ok(DedupRelation::Duplicate)
        }
        DedupRelation::Update => {
            // Wrap in SAVEPOINT for atomicity (same as Duplicate branch)
            store.conn().execute_batch("SAVEPOINT dedup_update")?;
            let update_result = (|| -> ReinResult<()> {
                if let Ok(mut winner) = store.get(new_id) {
                    for kw in &existing.keywords {
                        if !winner.keywords.contains(kw) {
                            winner.keywords.push(kw.clone());
                        }
                    }
                    winner.importance = winner.importance.max(existing.importance);
                    winner.layer = winner.importance.auto_layer();
                    winner.decay_lambda = winner.decay_lambda.min(existing.decay_lambda);
                    winner.strength = (winner.strength + 0.05).min(1.0);
                    if !verdict.merged_summary.trim().is_empty() {
                        winner.summary = verdict
                            .merged_summary
                            .chars()
                            .take(crate::types::SUMMARY_MAX_CHARS)
                            .collect();
                    }
                    winner.updated_at = chrono::Utc::now();
                    store.update(&winner)?;
                }
                store.mark_superseded(existing_id, new_id)?;
                record_dedup_artifacts(
                    store,
                    new_id,
                    &existing,
                    DedupRelation::Update,
                    lexical_score,
                    None,
                    reason,
                    payload,
                );
                Ok(())
            })();
            match update_result {
                Ok(()) => {
                    store.conn().execute_batch("RELEASE dedup_update")?;
                    Ok(DedupRelation::Update)
                }
                Err(e) => {
                    let _ = store.conn().execute_batch("ROLLBACK TO dedup_update");
                    let _ = store.conn().execute_batch("RELEASE dedup_update");
                    Err(e)
                }
            }
        }
        DedupRelation::Related => {
            if let Ok(mut left) = store.get(existing_id) {
                if !left.related_ids.contains(&new_id.to_string()) {
                    left.related_ids.push(new_id.to_string());
                    if let Err(e) = store.update(&left) {
                        tracing::warn!(
                            "dedup: failed to update related link on {existing_id}: {e}"
                        );
                    }
                }
            }
            if let Ok(mut right) = store.get(new_id) {
                if !right.related_ids.contains(&existing_id.to_string()) {
                    right.related_ids.push(existing_id.to_string());
                    if let Err(e) = store.update(&right) {
                        tracing::warn!("dedup: failed to update related link on {new_id}: {e}");
                    }
                }
            }
            if let Err(e) = store.record_dedup_decision(DedupDecision {
                id: String::new(),
                winner_id: None,
                loser_id: Some(new_id.to_string()),
                canonical_id: None,
                lexical_score,
                embedding_score: None,
                relation: DedupRelation::Related,
                confidence: verdict.confidence as f32,
                reason: reason.to_string(),
                operator: "auto".to_string(),
                reversible: true,
                merged_summary: (!verdict.merged_summary.trim().is_empty())
                    .then_some(verdict.merged_summary),
                novel_facts: verdict.novel_facts,
                conflict_detected: verdict.conflict_detected,
                payload,
                created_at: chrono::Utc::now(),
            }) {
                tracing::warn!("dedup: failed to record Related decision: {e}");
            }
            Ok(DedupRelation::Related)
        }
        DedupRelation::Distinct => {
            if let Err(e) = store.record_dedup_decision(DedupDecision {
                id: String::new(),
                winner_id: None,
                loser_id: Some(new_id.to_string()),
                canonical_id: None,
                lexical_score,
                embedding_score: None,
                relation: DedupRelation::Distinct,
                confidence: verdict.confidence as f32,
                reason: reason.to_string(),
                operator: "auto".to_string(),
                reversible: true,
                merged_summary: (!verdict.merged_summary.trim().is_empty())
                    .then_some(verdict.merged_summary),
                novel_facts: verdict.novel_facts,
                conflict_detected: verdict.conflict_detected,
                payload,
                created_at: chrono::Utc::now(),
            }) {
                tracing::warn!("dedup: failed to record Distinct decision: {e}");
            }
            Ok(DedupRelation::Distinct)
        }
    }
}

fn vec_dedup_run_limit(config: &ReinConfig) -> usize {
    config.async_memory.batch_size.max(1).saturating_mul(2)
}

fn vec_dedup_embed_batch_size(config: &ReinConfig) -> usize {
    config.async_memory.batch_size.max(1)
}

fn vec_dedup_llm_budget(config: &ReinConfig) -> usize {
    config.cleanup.llm_budget.max(1)
}

fn vec_dedup_strong_threshold(config: &ReinConfig) -> f64 {
    config.cleanup.vec_dedup_strong_threshold
}

fn vec_dedup_weak_threshold(config: &ReinConfig) -> f64 {
    config.cleanup.vec_dedup_weak_threshold
}

fn vec_dedup_pending_limit(config: &ReinConfig) -> usize {
    config
        .async_memory
        .max_jobs_per_run
        .clamp(1, VEC_DEDUP_PENDING_LIMIT)
}

fn take_vec_dedup_window(
    mut pending: Vec<VecDedupItem>,
    config: &ReinConfig,
) -> (Vec<VecDedupItem>, usize) {
    let run_limit = vec_dedup_run_limit(config);
    let skipped = pending.len().saturating_sub(run_limit);
    if pending.len() > run_limit {
        pending.truncate(run_limit);
    }
    (pending, skipped)
}

fn embed_vec_dedup_batch(
    embedder: &crate::embed::EmbedderKind,
    texts: &[String],
) -> Option<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Some(Vec::new());
    }

    let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                use crate::types::traits::Embedder;
                embedder.embed_batch(&text_refs).await
            })
        })
    }));

    result.ok().and_then(|value| value.ok())
}

fn none_bucket_ann_threshold(config: &ReinConfig) -> usize {
    config.async_memory.batch_size.max(1) * 4
}

fn none_bucket_neighbor_limit(config: &ReinConfig) -> usize {
    config.async_memory.batch_size.max(4)
}

fn bucket_embeddings(
    store: &SqliteStore,
    config: &ReinConfig,
    mems: &[Memory],
    indices: &[usize],
) -> std::collections::HashMap<usize, Vec<f32>> {
    let conn = store.conn();
    let model = config.embedding_model();
    let mut embeddings = std::collections::HashMap::new();
    let mut uncached: Vec<(usize, String)> = Vec::new();

    for &idx in indices {
        if let Some(emb) = crate::store::vec::get_embedding(conn, &mems[idx].id)
            .ok()
            .flatten()
        {
            embeddings.insert(idx, emb);
            continue;
        }

        let enriched = crate::embed::prepend_metadata(
            &mems[idx].topic,
            &mems[idx].summary,
            &mems[idx].content,
        );
        if let Ok(Some(emb)) = crate::embed::EmbedCache::get(conn, &enriched, &model) {
            let _ = crate::store::vec::insert_embedding(conn, &mems[idx].id, &emb);
            embeddings.insert(idx, emb);
            continue;
        }
        uncached.push((idx, enriched));
    }

    if uncached.is_empty() {
        return embeddings;
    }

    let Some(embedder) = crate::embed::create_embedder(config) else {
        return embeddings;
    };

    for chunk in uncached.chunks(vec_dedup_embed_batch_size(config)) {
        let texts: Vec<String> = chunk.iter().map(|(_, text)| text.clone()).collect();
        let Some(batch) = embed_vec_dedup_batch(&embedder, &texts) else {
            continue;
        };
        if batch.len() != chunk.len() {
            continue;
        }
        for ((idx, enriched), emb) in chunk.iter().zip(batch) {
            let _ = crate::embed::EmbedCache::put(conn, enriched, &model, &emb);
            let _ = crate::store::vec::insert_embedding(conn, &mems[*idx].id, &emb);
            embeddings.insert(*idx, emb);
        }
    }

    embeddings
}

fn build_none_bucket_ann_candidates(
    store: &SqliteStore,
    config: &ReinConfig,
    mems: &[Memory],
    indices: &[usize],
) -> std::collections::HashMap<usize, Vec<usize>> {
    if indices.len() <= none_bucket_ann_threshold(config) {
        return std::collections::HashMap::new();
    }

    let embeddings = bucket_embeddings(store, config, mems, indices);
    if embeddings.is_empty() {
        return std::collections::HashMap::new();
    }

    let index_by_id: std::collections::HashMap<&str, usize> = indices
        .iter()
        .map(|&idx| (mems[idx].id.as_str(), idx))
        .collect();
    let mut neighbors: std::collections::HashMap<usize, std::collections::BTreeSet<usize>> =
        std::collections::HashMap::new();

    for (&idx, embedding) in &embeddings {
        let Ok(results) = crate::store::vec::search_vec(
            store.conn(),
            embedding,
            None,
            none_bucket_neighbor_limit(config) + 1,
        ) else {
            continue;
        };
        for (neighbor_id, _) in results {
            let Some(&j) = index_by_id.get(neighbor_id.as_str()) else {
                continue;
            };
            if j == idx {
                continue;
            }
            neighbors.entry(idx).or_default().insert(j);
            neighbors.entry(j).or_default().insert(idx);
        }
    }

    neighbors
        .into_iter()
        .map(|(idx, set)| (idx, set.into_iter().collect()))
        .collect()
}

/// Embedding-based dedup sweep for memories marked `needs_vec_dedup`.
/// Computes embeddings (if missing), searches vec_memories for near-duplicates,
/// and merges/supersedes matches. Runs in the GC slow channel (zero hot-path cost).
pub(crate) fn run_vec_dedup(store: &SqliteStore, config: &ReinConfig) {
    run_vec_dedup_inner(store, config, None);
}

/// Test-only entry point that injects a caller-provided `EmbedderKind`
/// (typically `MockEmbedder`) instead of calling `create_embedder` so
/// integration tests can drive this sweep end-to-end without real API
/// credentials. Available under `test-support`.
#[cfg(feature = "test-support")]
pub fn run_vec_dedup_with_embedder(
    store: &SqliteStore,
    config: &ReinConfig,
    embedder: crate::embed::EmbedderKind,
) {
    run_vec_dedup_inner(store, config, Some(embedder));
}

fn run_vec_dedup_inner(
    store: &SqliteStore,
    config: &ReinConfig,
    embedder_override: Option<crate::embed::EmbedderKind>,
) {
    let conn = store.conn();
    // Load adaptive shadow suggestions once; destructive branches resolve
    // them through the hard getter before merging.
    let adaptive = crate::store::adaptive::AdaptiveState::restore_snapshot(conn);
    let pending_limit = vec_dedup_pending_limit(config);
    // `status IN ('active', 'updated')`: round-5 H-1. Merged canonicals
    // are promoted from `active` to `updated` by the merge trigger; both
    // states represent live canonicals and `needs_vec_dedup = 1` must be
    // swept regardless of which state the row is in or a resummerize +
    // subsequent merge can strand `needs_vec_dedup = 1` forever.
    let pending: Vec<VecDedupItem> = match conn.prepare(
        "SELECT id, topic, summary, content FROM memories
         WHERE needs_vec_dedup = 1
           AND status IN ('active', 'updated')
           AND superseded_by IS NULL
         LIMIT ?1",
    ) {
        Ok(mut stmt) => stmt
            .query_map(rusqlite::params![pending_limit as i64], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
            })
            .ok()
            .map(|rows| rows.filter_map(|r| r.ok()).collect())
            .unwrap_or_default(),
        Err(_) => return,
    };

    if pending.is_empty() {
        return;
    }

    let (pending, deferred) = take_vec_dedup_window(pending, config);
    tracing::debug!(
        "vec_dedup: processing {} memories ({} deferred)",
        pending.len(),
        deferred
    );

    let embedder = match embedder_override.or_else(|| crate::embed::create_embedder(config)) {
        Some(e) => e,
        None => {
            tracing::debug!(
                "vec_dedup: no embedder configured, skipping (flags preserved for later)"
            );
            return;
        }
    };

    let model_name = config.embedding_model();
    let mut merged = 0u32;
    let mut llm_verdicts_used = 0usize;
    let mut embeddings_by_id: std::collections::HashMap<String, Vec<f32>> =
        std::collections::HashMap::new();
    let mut pending_embeddings: Vec<(String, String)> = Vec::new();

    for (id, topic, summary, content) in &pending {
        let enriched = crate::embed::prepend_metadata(topic, summary, content);
        match crate::embed::EmbedCache::get(conn, &enriched, &model_name) {
            Ok(Some(cached)) => {
                let _ = crate::store::vec::insert_embedding(conn, id, &cached);
                embeddings_by_id.insert(id.clone(), cached);
            }
            _ => pending_embeddings.push((id.clone(), enriched)),
        }
    }

    for chunk in pending_embeddings.chunks(vec_dedup_embed_batch_size(config)) {
        let chunk_texts: Vec<String> = chunk.iter().map(|(_, text)| text.clone()).collect();
        let Some(batch_embeddings) = embed_vec_dedup_batch(&embedder, &chunk_texts) else {
            tracing::debug!(
                "vec_dedup: failed to batch-embed {} pending memories",
                chunk.len()
            );
            continue;
        };

        if batch_embeddings.len() != chunk.len() {
            tracing::debug!(
                "vec_dedup: batch embedding returned {} embeddings for {} inputs",
                batch_embeddings.len(),
                chunk.len()
            );
            continue;
        }

        for ((id, enriched), emb) in chunk.iter().zip(batch_embeddings) {
            let _ = crate::embed::EmbedCache::put(conn, enriched, &model_name, &emb);
            let _ = crate::store::vec::insert_embedding(conn, id, &emb);
            embeddings_by_id.insert(id.clone(), emb);
        }
    }

    for (id, _topic, _summary, content) in &pending {
        let Some(embedding) = embeddings_by_id.get(id).cloned() else {
            // Codex round-5 H-2: preserve `needs_vec_dedup = 1` on embed
            // failure so a transient network blip doesn't permanently
            // strip the row's vector recall. The next slow-channel pass
            // will retry.
            tracing::debug!(
                "vec_dedup: failed to compute embedding for {id}; preserving flag for retry"
            );
            continue;
        };

        // Codex round-5 H-3: `apply_resummerize` evicts the stale HNSW
        // entry so recall doesn't serve pre-rewrite semantics; after
        // re-embed, re-insert into HNSW explicitly. `update_hnsw` is a
        // no-op for `:memory:` stores and self-heals via the dirty-flag
        // mechanism on lock/save failure.
        store.update_hnsw_for_vec_dedup(id, &embedding);

        // Track whether the end-of-loop `needs_vec_dedup = 0` clear is
        // safe to apply for this row (Codex round-6 MEDIUM, companion to
        // round-5 H-2). Set to `false` by any write-failure path that
        // wants to preserve the flag for retry.
        let mut flag_clear_ok = true;

        let vec_results = match crate::store::vec::search_vec(conn, &embedding, None, 10) {
            Ok(r) => r,
            Err(_) => {
                // Codex round-5 H-2: keep the flag set so next sweep retries.
                tracing::debug!("vec_dedup: search_vec failed for {id}; preserving flag for retry");
                continue;
            }
        };

        // A1: non-destructive candidate floor = minimum shadow suggestion minus
        // margin (never below 0.40). A low suggestion can widen review coverage,
        // but cannot authorize the strong merge below.
        let weak_floor = adaptive
            .as_ref()
            .map(|s| {
                let global = s.get_dedup_shadow_threshold(None);
                let min_threshold = s.dedup_thresholds.values().copied().fold(global, f32::min);
                (min_threshold as f64 - 0.10).max(0.40)
            })
            .unwrap_or_else(|| vec_dedup_weak_threshold(config));

        for (candidate_id, distance) in &vec_results {
            if candidate_id == id {
                continue;
            }

            let sim = 1.0 - (*distance as f64);
            if sim < weak_floor {
                break;
            }

            let candidate = match store.get(candidate_id) {
                Ok(m) => m,
                Err(_) => continue,
            };
            // Codex round-6 HIGH: accept both `Active` and `Updated` as
            // live canonical states. The pending selector widened to
            // `status IN ('active', 'updated')` in round-5 H-1, but this
            // candidate gate still required `Active`. A canonical that
            // was legitimately merged (status promoted to `Updated`)
            // could never be chosen as a dedup target, and the
            // end-of-loop flag clear would then prematurely drop
            // `needs_vec_dedup` for rows where the dedup pass was
            // effectively a no-op.
            if candidate.superseded_by.is_some()
                || !matches!(
                    candidate.status,
                    crate::types::MemoryStatus::Active | crate::types::MemoryStatus::Updated
                )
            {
                continue;
            }

            // Cosine similarity and lexical similarity are different score
            // spaces. Lexical shadow learning may widen the candidate floor
            // above, but the destructive vector boundary comes only from the
            // vector-specific operator config.
            let strong_threshold = vec_dedup_strong_threshold(config);

            if sim > strong_threshold {
                let (keep_id, discard_id, discard_content, discard_created) =
                    if candidate.access_count >= 1
                        || candidate.created_at < chrono::Utc::now() - chrono::Duration::hours(1)
                    {
                        (
                            &candidate.id,
                            id,
                            content.to_string(),
                            chrono::Utc::now().format("%Y-%m-%d").to_string(),
                        )
                    } else {
                        (
                            id,
                            candidate_id,
                            candidate.content.clone(),
                            candidate.created_at.format("%Y-%m-%d").to_string(),
                        )
                    };

                // Any strong-match write failure must preserve
                // `needs_vec_dedup = 1` for retry; otherwise a rolled-back
                // merge can still strand the row unprocessed.
                if let Err(err) = store.conn().execute_batch("SAVEPOINT vec_strong") {
                    tracing::warn!(
                        error = %err,
                        source_id = %id,
                        candidate_id = %candidate.id,
                        "vec_dedup: failed to open strong-match savepoint; preserving needs_vec_dedup flag for retry"
                    );
                    flag_clear_ok = false;
                    break;
                }
                let merge_ok = (|| -> ReinResult<()> {
                    let discard_mem = store.get(discard_id)?;
                    // v1.2 audit F20: mark_superseded runs FIRST — it is pure
                    // SQL (fully rollback-safe), while update() below fires
                    // non-transactional Tantivy/HNSW writes. The old order
                    // (update then mark_superseded) meant a mark_superseded
                    // failure rolled back the SQL but left Tantivy serving
                    // the never-committed merged content and the winner
                    // evicted from HNSW with no dirty marker. Same
                    // do-fallible-DB-work-first rule as the v0.27 R7 P2
                    // MergeIntoMany fix.
                    store.mark_superseded(discard_id, keep_id)?;
                    if let Ok(mut kept) = store.get(keep_id) {
                        let unique = extract_unique_lines(&discard_content, &kept.content);
                        if !unique.is_empty() {
                            kept.content.push_str(&format!(
                                "\n\n[merged from {discard_id} on {discard_created}]\n{unique}"
                            ));
                        }
                        // Merge metadata (match merge_memory_into_winner behavior)
                        for kw in &discard_mem.keywords {
                            if !kept.keywords.contains(kw) {
                                kept.keywords.push(kw.clone());
                            }
                        }
                        kept.access_count = kept
                            .access_count
                            .saturating_add(discard_mem.access_count)
                            .saturating_add(1);
                        kept.importance = kept.importance.max(discard_mem.importance);
                        kept.layer = kept.importance.auto_layer();
                        kept.decay_lambda = kept.decay_lambda.min(discard_mem.decay_lambda);
                        kept.tier = stronger_tier(kept.tier, discard_mem.tier);
                        kept.last_accessed = kept.last_accessed.max(discard_mem.last_accessed);
                        kept.summary = kept
                            .content
                            .chars()
                            .take(crate::types::SUMMARY_MAX_CHARS)
                            .collect();
                        kept.strength = (kept.strength + 0.2).min(1.0);
                        kept.updated_at = chrono::Utc::now();
                        store.update(&kept)?;
                        // v1.2 audit F16 (sibling): update() ran with
                        // embedding=None on enriched text (guaranteed
                        // EmbedCache miss) and evicted the winner from the
                        // vector channel — re-flag it for the next sweep.
                        let _ = store.conn().execute(
                            "UPDATE memories SET needs_vec_dedup = 1 WHERE id = ?1",
                            rusqlite::params![keep_id],
                        );
                    }
                    if let Ok(discard_memory) = store.get(discard_id) {
                        record_dedup_artifacts(
                            store,
                            keep_id,
                            &discard_memory,
                            DedupRelation::Duplicate,
                            None,
                            Some(sim as f32),
                            "vec_dedup_strong",
                            Some(serde_json::json!({ "cosine_similarity": sim })),
                        );
                    }
                    Ok(())
                })();
                match merge_ok {
                    Ok(()) => {
                        // v1.2 audit F21: RELEASE is the commit on this
                        // connection — a swallowed failure left the savepoint
                        // open (every later pair nested into the doomed txn)
                        // while the merge was reported done. Unwind + retry
                        // flag on failure, same pattern as
                        // persist_triples_replacing (codex v1.2 R2 P2).
                        if let Err(e) = store.conn().execute_batch("RELEASE vec_strong") {
                            tracing::warn!(
                                "vec_dedup: RELEASE vec_strong failed ({e}); rolling back"
                            );
                            let _ = store.conn().execute_batch("ROLLBACK TO vec_strong");
                            let _ = store.conn().execute_batch("RELEASE vec_strong");
                            flag_clear_ok = false;
                            break;
                        }
                    }
                    Err(e) => {
                        tracing::warn!("vec_dedup: strong-match merge failed, rolling back: {e}");
                        let _ = store.conn().execute_batch("ROLLBACK TO vec_strong");
                        let _ = store.conn().execute_batch("RELEASE vec_strong");
                        flag_clear_ok = false;
                        break;
                    }
                }

                tracing::info!(
                    "vec_dedup: merged {discard_id} into {keep_id} (cosine_sim={sim:.3})"
                );
                // codex remediation R3 P2: when the SOURCE row is the merge
                // winner, the end-of-loop `needs_vec_dedup = 0` clear would
                // immediately undo the re-flag set inside the merge —
                // leaving the enriched winner absent from the vector channel.
                // Keep the flag set so the next sweep re-embeds it.
                if keep_id == id {
                    flag_clear_ok = false;
                }
                merged += 1;
                break;
            }

            if llm_verdicts_used >= vec_dedup_llm_budget(config) {
                tracing::debug!(
                    "vec_dedup: LLM budget exhausted for {id}, skipping gray-zone verdict"
                );
                continue;
            }

            let relation = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async {
                        crate::extract::llm::llm_dedup_verdict(config, content, &candidate.content)
                            .await
                            .ok()
                            .flatten()
                            .map(|verdict| verdict.relation)
                            .unwrap_or(DedupRelation::Distinct)
                    })
                })
            }))
            .unwrap_or(DedupRelation::Distinct);
            llm_verdicts_used += 1;

            if matches!(relation, DedupRelation::Duplicate | DedupRelation::Update) {
                // Codex round-6 MEDIUM: treat `mark_superseded` failure as
                // a write-failure and preserve `needs_vec_dedup` for retry
                // rather than silently counting it as a successful merge.
                // Before this, a supersede error logged success, incremented
                // `merged`, broke out of the loop, and the end-of-loop
                // clear dropped the flag — the row ended up both
                // un-superseded AND un-flagged.
                if let Err(err) = store.mark_superseded(id, &candidate.id) {
                    tracing::warn!(
                        error = %err,
                        source_id = %id,
                        winner_id = %candidate.id,
                        "vec_dedup: mark_superseded failed; preserving needs_vec_dedup flag for retry"
                    );
                    // Skip the end-of-loop flag clear by continuing the
                    // OUTER `for (id, ...)` loop via `continue 'pending`.
                    // We don't have a labeled loop yet, so the simplest
                    // correct behavior: jump past the flag-clearing
                    // UPDATE for this row. Handled by the `flag_clear_ok`
                    // local below.
                    flag_clear_ok = false;
                    break;
                }
                if let Ok(discard_memory) = store.get(id) {
                    record_dedup_artifacts(
                        store,
                        &candidate.id,
                        &discard_memory,
                        relation,
                        None,
                        Some(sim as f32),
                        "vec_dedup_llm",
                        Some(serde_json::json!({ "cosine_similarity": sim })),
                    );
                }
                tracing::info!(
                    "vec_dedup: LLM verdict {} superseded {id} by {} (cosine_sim={sim:.3})",
                    relation,
                    candidate.id
                );
                merged += 1;
                break;
            }
        }

        if flag_clear_ok {
            let _ = conn.execute(
                "UPDATE memories SET needs_vec_dedup = 0 WHERE id = ?1",
                rusqlite::params![id],
            );
        }
    }

    if merged > 0 {
        tracing::info!("vec_dedup: merged {merged} semantic duplicates");
    }
}

/// Run dedup scan across the provided topic groups with provenance-preserving merge.
///
/// Instead of hard-deleting duplicates (which loses temporal anchors and unique
/// details), this extracts unique lines from the "loser" and appends them to the
/// "winner" with a provenance marker. The loser is then superseded, not deleted.
///
/// Returns (duplicates_found, duplicates_merged).
pub fn run_dedup_scoped(
    store: &SqliteStore,
    config: &ReinConfig,
    groups: &[TopicGroup],
    threshold: f32,
    dry_run: bool,
    merge_variants: bool,
) -> ReinResult<(u32, u32)> {
    let mut dups_found = 0u32;
    let mut dups_merged = 0u32;
    let mut changed = false;
    let llm_budget = vec_dedup_llm_budget(config);
    let mut llm_calls_used = 0usize;
    // Load adaptive suggestions once; each destructive comparison below
    // resolves a hard-effective threshold against static config.
    let adaptive_state = if config.adaptive.enabled {
        crate::store::adaptive::AdaptiveState::restore_snapshot(store.conn())
    } else {
        None
    };
    for group in groups {
        let mems: Vec<_> = load_group_memories(store, group)?
            .into_iter()
            .filter(|m| m.superseded_by.is_none())
            .collect();
        let mut processed: std::collections::HashSet<String> = std::collections::HashSet::new();

        // Group memories by cluster_id for O(n * max_cluster_size) comparison
        // instead of O(n^2). Unassigned memories (cluster_id = None) form their own group.
        let mut cluster_groups: std::collections::HashMap<Option<u32>, Vec<usize>> =
            std::collections::HashMap::new();
        for (idx, mem) in mems.iter().enumerate() {
            cluster_groups.entry(mem.cluster_id).or_default().push(idx);
        }

        let none_bucket_ann = cluster_groups
            .get(&None)
            .map(|indices| build_none_bucket_ann_candidates(store, config, &mems, indices))
            .unwrap_or_default();

        // Compare within each cluster group (much smaller than full pairwise)
        for (cluster_id, indices) in &cluster_groups {
            // Unlabeled per-cluster suggestions are shadow-only; the hard
            // getter keeps this destructive lexical merge on static config.
            let cluster_threshold = adaptive_state
                .as_ref()
                .map(|s| {
                    s.get_hard_dedup_threshold(*cluster_id, config.search.dedup_similarity as f32)
                })
                .unwrap_or(threshold);
            for ii in 0..indices.len() {
                let i = indices[ii];
                if processed.contains(&mems[i].id) {
                    continue;
                }
                let candidate_js: Vec<usize> =
                    if cluster_id.is_none() && !none_bucket_ann.is_empty() {
                        none_bucket_ann
                            .get(&i)
                            .cloned()
                            .filter(|neighbors| !neighbors.is_empty())
                            .unwrap_or_else(|| indices[(ii + 1)..].to_vec())
                    } else {
                        indices[(ii + 1)..].to_vec()
                    };
                for j in candidate_js {
                    if j <= i {
                        continue;
                    }
                    if processed.contains(&mems[j].id) {
                        continue;
                    }
                    let sim = crate::extract::similarity(&mems[i].content, &mems[j].content);
                    let gray_zone_floor =
                        (cluster_threshold - 0.15).max(0.50).min(cluster_threshold);
                    let relation = if sim >= cluster_threshold {
                        DedupRelation::Duplicate
                    } else if !dry_run && sim >= gray_zone_floor && llm_calls_used < llm_budget {
                        llm_calls_used += 1;
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            tokio::task::block_in_place(|| {
                                tokio::runtime::Handle::current().block_on(async {
                                    crate::extract::llm::llm_dedup_verdict(
                                        config,
                                        &mems[i].content,
                                        &mems[j].content,
                                    )
                                    .await
                                    .ok()
                                    .flatten()
                                    .map(|verdict| verdict.relation)
                                    .unwrap_or(DedupRelation::Distinct)
                                })
                            })
                        }))
                        .unwrap_or(DedupRelation::Distinct)
                    } else {
                        DedupRelation::Distinct
                    };

                    if matches!(relation, DedupRelation::Duplicate | DedupRelation::Update) {
                        dups_found += 1;
                        // Determine winner (longer/newer) and loser
                        let (winner_idx, loser_idx) =
                            if mems[j].content.len() >= mems[i].content.len() {
                                (j, i)
                            } else {
                                (i, j)
                            };
                        if dry_run {
                            tracing::debug!(
                                "dup: '{}' ~ '{}'",
                                &mems[loser_idx].summary.chars().take(40).collect::<String>(),
                                &mems[winner_idx]
                                    .summary
                                    .chars()
                                    .take(40)
                                    .collect::<String>()
                            );
                        } else {
                            // Provenance-preserving merge wrapped in SAVEPOINT for atomicity
                            let sp_name = format!("dedup_{dups_found}");
                            if let Err(e) =
                                store.conn().execute_batch(&format!("SAVEPOINT {sp_name}"))
                            {
                                tracing::warn!("dedup: failed to create savepoint: {e}");
                                continue;
                            }

                            let unique = extract_unique_lines(
                                &mems[loser_idx].content,
                                &mems[winner_idx].content,
                            );
                            let canonical_id = store
                                .canonical_id_for(&mems[winner_idx].id)
                                .unwrap_or_else(|_| mems[winner_idx].id.clone());

                            let merge_result = (|| -> ReinResult<()> {
                                let mut winner = store.get(&mems[winner_idx].id)?;
                                if !unique.is_empty() {
                                    let provenance = format!(
                                        "\n\n[merged from {} on {}]\n{}",
                                        mems[loser_idx].id,
                                        mems[loser_idx].created_at.format("%Y-%m-%d"),
                                        unique,
                                    );
                                    winner.content.push_str(&provenance);
                                }
                                for kw in &mems[loser_idx].keywords {
                                    if !winner.keywords.contains(kw) {
                                        winner.keywords.push(kw.clone());
                                    }
                                }
                                winner.access_count = winner
                                    .access_count
                                    .saturating_add(mems[loser_idx].access_count)
                                    .saturating_add(1);
                                winner.strength = (winner.strength + 0.1).min(1.0);
                                winner.importance =
                                    winner.importance.max(mems[loser_idx].importance);
                                winner.layer = winner.importance.auto_layer();
                                winner.decay_lambda =
                                    winner.decay_lambda.min(mems[loser_idx].decay_lambda);
                                winner.tier = stronger_tier(winner.tier, mems[loser_idx].tier);
                                winner.last_accessed = chrono::Utc::now();
                                winner.updated_at = chrono::Utc::now();
                                store.update(&winner)?;
                                store.mark_superseded(&mems[loser_idx].id, &mems[winner_idx].id)?;
                                store
                                    .snapshot_memory_as_evidence(&canonical_id, &mems[loser_idx])?;
                                store.record_dedup_decision(DedupDecision {
                                    id: String::new(),
                                    winner_id: Some(mems[winner_idx].id.clone()),
                                    loser_id: Some(mems[loser_idx].id.clone()),
                                    canonical_id: Some(canonical_id.clone()),
                                    lexical_score: Some(sim),
                                    embedding_score: None,
                                    relation,
                                    confidence: sim,
                                    reason: "batch_dedup".to_string(),
                                    operator: "manual".to_string(),
                                    reversible: true,
                                    merged_summary: Some(mems[winner_idx].summary.clone()),
                                    novel_facts: unique
                                        .lines()
                                        .map(|line| line.trim().to_string())
                                        .filter(|line| !line.is_empty())
                                        .collect(),
                                    conflict_detected: matches!(relation, DedupRelation::Update),
                                    payload: Some(serde_json::json!({
                                        "merge_variants": merge_variants,
                                    })),
                                    created_at: chrono::Utc::now(),
                                })?;
                                Ok(())
                            })();

                            match merge_result {
                                Ok(()) => {
                                    // v1.2 audit F21: RELEASE is the commit
                                    // on this connection. A swallowed failure
                                    // left the savepoint open — every later
                                    // pair nested into the doomed transaction
                                    // and the op reported "merged N" for work
                                    // that rolled back when the connection
                                    // dropped. Unwind and propagate instead.
                                    if let Err(e) =
                                        store.conn().execute_batch(&format!("RELEASE {sp_name}"))
                                    {
                                        let _ = store
                                            .conn()
                                            .execute_batch(&format!("ROLLBACK TO {sp_name}"));
                                        let _ = store
                                            .conn()
                                            .execute_batch(&format!("RELEASE {sp_name}"));
                                        return Err(crate::types::ReinError::Config(format!(
                                            "dedup: RELEASE {sp_name} (commit) failed: {e}"
                                        )));
                                    }
                                    emit_cleanup_event(
                                        store,
                                        crate::store::adaptive::EventType::Forget,
                                        Some(mems[loser_idx].id.clone()),
                                        Some(mems[loser_idx].topic.clone()),
                                        serde_json::json!({
                                            "source": "dedup",
                                            "replacement_id": mems[winner_idx].id,
                                            "winner_topic": mems[winner_idx].topic,
                                            "similarity": sim,
                                            "merge_variants": merge_variants,
                                        }),
                                    );
                                    dups_merged += 1;
                                    changed = true;
                                }
                                Err(e) => {
                                    tracing::warn!("dedup merge failed, rolling back: {e}");
                                    let _ = store
                                        .conn()
                                        .execute_batch(&format!("ROLLBACK TO {sp_name}"));
                                    let _ =
                                        store.conn().execute_batch(&format!("RELEASE {sp_name}"));
                                }
                            }
                        }
                        // Mark only the loser as processed
                        processed.insert(mems[loser_idx].id.clone());
                        // If mems[i] was the loser, stop scanning (it's been superseded)
                        // If mems[i] was the winner, continue scanning for more duplicates
                        if loser_idx == i {
                            break;
                        }
                    }
                }
            }
        } // end for (_cluster_id, indices)
    }
    if changed {
        emit_cleanup_event(
            store,
            crate::store::adaptive::EventType::ParamUpdate,
            None,
            None,
            serde_json::json!({
                "source": "dedup",
                "duplicates_merged": dups_merged,
                "merge_variants": merge_variants,
            }),
        );
        run_adaptive_pipeline(store, config);
    }
    Ok((dups_found, dups_merged))
}

pub fn run_dedup(
    store: &SqliteStore,
    config: &ReinConfig,
    threshold: f32,
    dry_run: bool,
    merge_variants: bool,
) -> ReinResult<(u32, u32)> {
    let groups = super::resolve_topic_groups(store, None, &[], None, true, merge_variants)?;
    run_dedup_scoped(store, config, &groups, threshold, dry_run, merge_variants)
}

/// Extract lines from `source` that are not present in `target`.
/// Used for provenance-preserving merge: keeps unique temporal anchors and details.
///
/// Comparison is done at the LINE level (not via substring containment) so a
/// line like "Stop" or "a" doesn't get dropped just because its lowercased
/// form appears inside some unrelated word in target — previously the
/// substring-based `target_lower.contains(...)` form could silently discard
/// short but genuinely-unique source lines during merge (B6 #31).
pub fn extract_unique_lines(source: &str, target: &str) -> String {
    let target_lines: std::collections::HashSet<String> = target
        .lines()
        .map(|line| line.trim().to_lowercase())
        .collect();
    let unique: Vec<&str> = source
        .lines()
        .filter(|line| {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                return false;
            }
            // Skip merge markers from previous merges (prevent marker accumulation)
            if trimmed.starts_with("[merged from ") || trimmed.starts_with("[merged on ") {
                return false;
            }
            !target_lines.contains(&trimmed.to_lowercase())
        })
        .collect();
    unique.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ReinConfig;
    use crate::store::SqliteStore;
    use crate::types::traits::MemoryStore;
    use chrono::Utc;

    fn test_memory(topic: &str, content: &str) -> Memory {
        Memory {
            id: ulid::Ulid::new().to_string(),
            layer: MemoryLayer::LTM,
            topic: topic.to_string(),
            summary: content.chars().take(50).collect(),
            content: content.to_string(),
            keywords: vec![],
            importance: Importance::Medium,
            source: Source::Manual,
            strength: 0.5,
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
            archival_summary: None,
            archival_summary_at: None,
            archival_summary_version: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_accessed: Utc::now(),
        }
    }

    fn test_memory_with_cluster(topic: &str, content: &str, cluster: u32) -> Memory {
        let mut m = test_memory(topic, content);
        m.cluster_id = Some(cluster);
        m
    }

    #[test]
    fn test_run_dedup_finds_and_merges_duplicates() {
        let store = SqliteStore::in_memory().unwrap();
        let config = ReinConfig::default();

        // Two memories with same topic and nearly identical content (>0.70 similarity)
        let id1 = store
            .store(test_memory(
                "docker",
                "deploy the application using docker compose up with the production config",
            ))
            .unwrap();
        let id2 = store
            .store(test_memory(
                "docker",
                "deploy the application using docker compose up with the production configuration",
            ))
            .unwrap();

        let (found, merged) = run_dedup(&store, &config, 0.70, false, false).unwrap();
        assert_eq!(found, 1, "should find 1 duplicate pair");
        assert_eq!(merged, 1, "should merge 1 duplicate pair");

        // One of them should be superseded
        let m1 = store.get(&id1).ok();
        let m2 = store.get(&id2).ok();
        let superseded_count = [&m1, &m2]
            .iter()
            .filter(|m| m.as_ref().is_some_and(|m| m.superseded_by.is_some()))
            .count();
        assert_eq!(
            superseded_count, 1,
            "exactly one memory should be superseded"
        );
    }

    #[test]
    fn test_run_dedup_dry_run_does_not_modify() {
        let store = SqliteStore::in_memory().unwrap();
        let config = ReinConfig::default();

        let id1 = store
            .store(test_memory(
                "docker",
                "deploy the application using docker compose up with the production config",
            ))
            .unwrap();
        let id2 = store
            .store(test_memory(
                "docker",
                "deploy the application using docker compose up with the production configuration",
            ))
            .unwrap();

        let (found, merged) = run_dedup(&store, &config, 0.70, true, false).unwrap();
        assert_eq!(found, 1, "should find 1 duplicate pair in dry run");
        assert_eq!(merged, 0, "should merge 0 in dry run");

        // Neither should be superseded
        let m1 = store.get(&id1).unwrap();
        let _m2 = store.get(&id2).unwrap();
        assert!(m1.superseded_by.is_none(), "m1 should not be superseded");
        assert!(_m2.superseded_by.is_none(), "m2 should not be superseded");
    }

    #[test]
    fn test_run_dedup_cluster_grouping() {
        let store = SqliteStore::in_memory().unwrap();
        let config = ReinConfig::default();

        // Cluster 1: two similar memories about docker
        store
            .store(test_memory_with_cluster(
                "infra",
                "deploy the application using docker compose up with the production config",
                1,
            ))
            .unwrap();
        store
            .store(test_memory_with_cluster(
                "infra",
                "deploy the application using docker compose up with the production configuration",
                1,
            ))
            .unwrap();

        // Cluster 2: two similar memories about kubernetes
        store
            .store(test_memory_with_cluster(
                "infra",
                "scale the kubernetes pods using kubectl scale deployment app replicas three",
                2,
            ))
            .unwrap();
        store
            .store(test_memory_with_cluster(
                "infra",
                "scale the kubernetes pods using kubectl scale deployment app replicas to three",
                2,
            ))
            .unwrap();

        let (found, merged) = run_dedup(&store, &config, 0.70, false, false).unwrap();
        // Should find duplicates only within each cluster, not cross-cluster
        assert_eq!(found, 2, "should find 2 within-cluster duplicate pairs");
        assert_eq!(merged, 2, "should merge 2 within-cluster duplicate pairs");
    }

    #[test]
    fn batch_lexical_dedup_ignores_shadow_below_static() {
        let store = SqliteStore::in_memory().unwrap();
        let config = ReinConfig::default();
        let mut state = crate::store::adaptive::AdaptiveState {
            global_dedup_threshold: 0.40,
            version: 1,
            ..Default::default()
        };
        state.dedup_thresholds.insert(7, 0.45);
        state.save_snapshot(store.conn()).unwrap();

        let left = "alpha bravo charlie delta echo";
        let right = "alpha bravo charlie foxtrot golf";
        let similarity = crate::extract::similarity(left, right);
        assert!(
            similarity > 0.45 && similarity < config.search.dedup_similarity as f32,
            "test fixture must sit between shadow and static thresholds: {similarity}"
        );

        store
            .store(test_memory_with_cluster("threshold-floor", left, 7))
            .unwrap();
        store
            .store(test_memory_with_cluster("threshold-floor", right, 7))
            .unwrap();

        let (found, merged) = run_dedup(
            &store,
            &config,
            config.search.dedup_similarity as f32,
            true,
            false,
        )
        .unwrap();
        assert_eq!(found, 0, "shadow threshold must not affect the hard policy");
        assert_eq!(merged, 0);
    }

    #[test]
    fn test_take_vec_dedup_window_caps_processing() {
        let config = ReinConfig::default();
        let pending: Vec<VecDedupItem> = (0..20)
            .map(|i| {
                (
                    format!("id-{i}"),
                    "topic".to_string(),
                    format!("summary-{i}"),
                    format!("content-{i}"),
                )
            })
            .collect();

        let (window, skipped) = take_vec_dedup_window(pending, &config);
        assert_eq!(window.len(), 16);
        assert_eq!(skipped, 4);
    }

    #[test]
    fn test_run_dedup_scoped_limits_to_selected_group() {
        let store = SqliteStore::in_memory().unwrap();
        let config = ReinConfig::default();

        let left_1 = store
            .store(test_memory(
                "docker",
                "deploy the application using docker compose up with the production config",
            ))
            .unwrap();
        let left_2 = store
            .store(test_memory(
                "docker",
                "deploy the application using docker compose up with the production configuration",
            ))
            .unwrap();
        let right_1 = store
            .store(test_memory(
                "kubernetes",
                "scale the kubernetes deployment to three replicas",
            ))
            .unwrap();
        let right_2 = store
            .store(test_memory(
                "kubernetes",
                "scale the kubernetes deployment to 3 replicas",
            ))
            .unwrap();

        let groups = vec![TopicGroup {
            canonical_topic: "docker".to_string(),
            topics: vec!["docker".to_string()],
        }];

        let (found, merged) =
            run_dedup_scoped(&store, &config, &groups, 0.70, false, false).unwrap();
        assert_eq!(found, 1);
        assert_eq!(merged, 1);

        let left_superseded = [store.get(&left_1).unwrap(), store.get(&left_2).unwrap()]
            .into_iter()
            .filter(|memory| memory.superseded_by.is_some())
            .count();
        let right_superseded = [store.get(&right_1).unwrap(), store.get(&right_2).unwrap()]
            .into_iter()
            .filter(|memory| memory.superseded_by.is_some())
            .count();
        assert_eq!(left_superseded, 1, "selected group should be deduplicated");
        assert_eq!(
            right_superseded, 0,
            "non-selected group must remain untouched"
        );
    }

    #[test]
    fn test_build_none_bucket_ann_candidates_uses_vector_neighbors() {
        let store = SqliteStore::in_memory().unwrap();
        let config = ReinConfig::default();
        let mut mems = Vec::new();

        for i in 0..40 {
            let id = store
                .store(test_memory("docker", &format!("docker note {i}")))
                .unwrap();
            let memory = store.get(&id).unwrap();
            mems.push(memory);

            let mut embedding = vec![0.0f32; 3072];
            if i < 2 {
                embedding[0] = 1.0;
                embedding[1] = 1.0;
            } else {
                embedding[(i % 32) + 2] = 10.0 + i as f32;
            }
            crate::store::vec::insert_embedding(store.conn(), &id, &embedding).unwrap();
        }

        let indices: Vec<usize> = (0..mems.len()).collect();
        let neighbors = build_none_bucket_ann_candidates(&store, &config, &mems, &indices);

        assert!(
            neighbors
                .get(&0)
                .map(|ids| ids.contains(&1))
                .unwrap_or(false),
            "ANN candidate graph should connect close unclustered memories"
        );
    }

    #[test]
    fn test_run_dedup_savepoint_atomicity() {
        let store = SqliteStore::in_memory().unwrap();
        let config = ReinConfig::default();

        let id1 = store
            .store(test_memory(
                "docker",
                "deploy the application using docker compose up with the production config",
            ))
            .unwrap();
        let id2 = store
            .store(test_memory(
                "docker",
                "deploy the application using docker compose up with the production configuration",
            ))
            .unwrap();

        let (_found, merged) = run_dedup(&store, &config, 0.70, false, false).unwrap();
        assert_eq!(merged, 1);

        // Determine winner and loser
        let m1 = store.get(&id1).unwrap();
        let (winner_id, _loser_id) = if m1.superseded_by.is_some() {
            (&id2, &id1)
        } else {
            (&id1, &id2)
        };

        // Winner should have merged content (or at least be the longer one)
        let winner = store.get(winner_id).unwrap();
        // The loser should be superseded
        let loser_id = if winner_id == &id1 { &id2 } else { &id1 };
        let loser = store.get(loser_id).unwrap();
        assert!(loser.superseded_by.is_some(), "loser should be superseded");
        assert_eq!(
            loser.superseded_by.as_deref(),
            Some(winner_id.as_str()),
            "loser should point to winner"
        );

        // dedup_decisions table should have a record
        let decisions = store.list_dedup_decisions(None, 10).unwrap();
        assert!(
            !decisions.is_empty(),
            "dedup_decisions should have at least one record"
        );
        assert_eq!(decisions[0].winner_id.as_deref(), Some(winner.id.as_str()));
    }

    #[test]
    fn test_merge_memory_into_winner_basic() {
        let store = SqliteStore::in_memory().unwrap();

        let mut winner_mem = test_memory("docker", "deploy with docker compose up");
        winner_mem.keywords = vec!["docker".to_string(), "deploy".to_string()];
        let winner_id = store.store(winner_mem).unwrap();

        let mut loser_mem = test_memory(
            "docker",
            "deploy with docker compose up and custom network settings",
        );
        loser_mem.keywords = vec!["docker".to_string(), "network".to_string()];
        let loser_id = store.store(loser_mem).unwrap();

        let loser = store.get(&loser_id).unwrap();

        merge_memory_into_winner(
            &store,
            None,
            &winner_id,
            &loser,
            Some(0.85),
            None,
            "test_merge",
            None,
        )
        .unwrap();

        let winner = store.get(&winner_id).unwrap();

        // Winner content should include provenance marker
        assert!(
            winner.content.contains("[merged from"),
            "winner should contain provenance marker, got: {}",
            winner.content
        );

        // Winner keywords should include loser's keywords
        assert!(
            winner.keywords.contains(&"network".to_string()),
            "winner should have loser's keyword 'network': {:?}",
            winner.keywords
        );

        // Loser should be marked superseded
        let loser_after = store.get(&loser_id).unwrap();
        assert!(
            loser_after.superseded_by.is_some(),
            "loser should be superseded"
        );
        assert_eq!(
            loser_after.superseded_by.as_deref(),
            Some(winner_id.as_str())
        );
    }

    #[test]
    fn test_merge_memory_into_winner_rollback_on_missing_winner() {
        let store = SqliteStore::in_memory().unwrap();

        // Store only the loser
        let loser_mem = test_memory("docker", "some content for the loser memory");
        let loser_id = store.store(loser_mem).unwrap();
        let loser = store.get(&loser_id).unwrap();

        // Attempt to merge with a non-existent winner
        let result = merge_memory_into_winner(
            &store,
            None,
            "nonexistent-winner-id",
            &loser,
            Some(0.85),
            None,
            "test_rollback",
            None,
        );

        assert!(result.is_err(), "should return Err for missing winner");

        // Loser should NOT be marked superseded (SAVEPOINT rollback)
        let loser_after = store.get(&loser_id).unwrap();
        assert!(
            loser_after.superseded_by.is_none(),
            "loser should not be superseded after rollback"
        );
    }

    #[test]
    fn test_extract_unique_lines_basic() {
        let source = "line A\nline B\nline C\nline D";
        let target = "line A\nline C\nline E";

        let unique = extract_unique_lines(source, target);
        assert!(unique.contains("line B"), "should keep unique line B");
        assert!(unique.contains("line D"), "should keep unique line D");
        assert!(!unique.contains("line A"), "should exclude shared line A");
        assert!(!unique.contains("line C"), "should exclude shared line C");
    }

    #[test]
    fn extract_unique_lines_does_not_drop_substring_matches() {
        // Before B6 #31, source lines that were substrings of target text were
        // silently discarded — "Stop" would disappear if target happened to
        // contain "Stopwatch" somewhere. The line-level match now preserves
        // these genuinely unique lines.
        let source = "Stop\nA short note\nDistinct fact";
        let target = "Stopwatch measurements were logged.\nUnrelated text that mentions a short-note in prose.\n";
        let unique = extract_unique_lines(source, target);
        assert!(
            unique.contains("Stop"),
            "short line 'Stop' must be preserved"
        );
        assert!(
            unique.contains("A short note"),
            "distinct multi-word line must be preserved even if its hyphenated form appears inside another word"
        );
        assert!(unique.contains("Distinct fact"));
    }

    #[test]
    fn extract_unique_lines_drops_exact_duplicate_lines() {
        let source = "line A\nline B\n";
        let target = "line A\nother\n";
        let unique = extract_unique_lines(source, target);
        assert!(!unique.contains("line A"));
        assert!(unique.contains("line B"));
    }
}
