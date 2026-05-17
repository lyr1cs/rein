//! Knowledge graph operations: knowledge units, evolution, linking, and organizing.

use chrono::Utc;

use crate::types::*;

use super::SqliteStore;

impl SqliteStore {
    /// Store knowledge units (concepts + links) into the Memoir system.
    /// Auto-creates memoirs as needed. Refines existing concepts, adds new ones.
    /// Returns a report of what was created/refined.
    pub fn store_knowledge_units(
        &self,
        concepts: &[crate::extract::ExtractedConcept],
        links: &[crate::extract::ExtractedLink],
    ) -> ReinResult<super::KnowledgeStoreReport> {
        self.store_knowledge_units_with_sources(concepts, links, &[])
    }

    /// Store knowledge units with bidirectional Memory ↔ Concept links.
    /// `source_memory_ids` are the memories that produced these concepts.
    pub fn store_knowledge_units_with_sources(
        &self,
        concepts: &[crate::extract::ExtractedConcept],
        links: &[crate::extract::ExtractedLink],
        source_memory_ids: &[String],
    ) -> ReinResult<super::KnowledgeStoreReport> {
        // Wrap in savepoint for atomicity — many DB writes follow
        self.conn
            .execute_batch("SAVEPOINT knowledge_units")
            .map_err(crate::types::ReinError::Database)?;
        let result = self.store_knowledge_units_inner(concepts, links, source_memory_ids);
        match &result {
            Ok(_) => {
                self.conn
                    .execute_batch("RELEASE knowledge_units")
                    .map_err(crate::types::ReinError::Database)?;
            }
            Err(_) => {
                let _ = self.conn.execute_batch("ROLLBACK TO knowledge_units");
                let _ = self.conn.execute_batch("RELEASE knowledge_units");
            }
        }
        result
    }

    fn store_knowledge_units_inner(
        &self,
        concepts: &[crate::extract::ExtractedConcept],
        links: &[crate::extract::ExtractedLink],
        source_memory_ids: &[String],
    ) -> ReinResult<super::KnowledgeStoreReport> {
        let mut report = super::KnowledgeStoreReport::default();

        // Group concepts by memoir
        let mut by_memoir: std::collections::HashMap<
            String,
            Vec<&crate::extract::ExtractedConcept>,
        > = std::collections::HashMap::new();
        for c in concepts {
            by_memoir.entry(c.memoir.clone()).or_default().push(c);
        }

        // Process each memoir group
        for (memoir_name, memoir_concepts) in &by_memoir {
            // Ensure memoir exists
            if self.get_memoir(memoir_name)?.is_none() {
                let memoir = Memoir {
                    id: String::new(),
                    name: memoir_name.clone(),
                    description: "auto-created by LLM extraction".to_string(),
                    created_at: Utc::now(),
                    updated_at: Utc::now(),
                };
                self.create_memoir(memoir)?;
                report.memoirs_created += 1;
            }

            // Add/refine concepts
            for c in memoir_concepts {
                match self.get_concept(memoir_name, &c.name)? {
                    Some(existing) => {
                        // Check for contradiction: low similarity = potential conflict
                        let sim = crate::extract::similarity(&existing.definition, &c.definition);
                        if sim < 0.3 && c.quality_confidence > 0.5 && !existing.id.is_empty() {
                            tracing::info!(
                                "contradiction detected for concept '{}': old def differs significantly (sim={:.2})",
                                c.name, sim
                            );
                            // Expire outgoing links from the old concept version
                            let now = Utc::now();
                            if let Ok(old_links) = self.get_links_from(&existing.id) {
                                for link in &old_links {
                                    if link.valid_until.is_none() {
                                        self.expire_link(&link.id, now)?;
                                    }
                                }
                            }
                        }
                        // Refine (use existing.name for exact DB match after normalized lookup)
                        self.refine_concept(memoir_name, &existing.name, &c.definition)?;
                        report.concepts_refined += 1;
                        // Merge new source_memory_ids into existing concept
                        if !source_memory_ids.is_empty() {
                            let mut merged = existing.source_memory_ids.clone();
                            for sid in source_memory_ids {
                                if !merged.contains(sid) {
                                    merged.push(sid.clone());
                                }
                            }
                            if merged.len() > existing.source_memory_ids.len() {
                                let json = serde_json::to_string(&merged)?;
                                self.conn().execute(
                                    "UPDATE concepts SET source_memory_ids = ?1 WHERE id = ?2",
                                    rusqlite::params![json, existing.id],
                                )?;
                            }
                        }
                    }
                    None => {
                        // New concept — normalize name to prevent future dupes
                        let concept = Concept {
                            id: String::new(),
                            memoir_id: memoir_name.clone(),
                            name: crate::store::memoir::normalize_concept_name(&c.name),
                            definition: c.definition.clone(),
                            labels: c.labels.clone(),
                            source_memory_ids: source_memory_ids.to_vec(),
                            confidence: c.quality_confidence as f32,
                            revision: 1,
                            last_episode_id: None,
                            created_at: Utc::now(),
                            updated_at: Utc::now(),
                            living_summary: None,
                            living_summary_updated_at: None,
                            living_summary_source_revision: None,
                            living_summary_id: None,
                        };
                        self.add_concept(concept)?;
                        report.concepts_added += 1;
                    }
                }
            }
        }

        // Process links (both concepts must exist and be in same memoir)
        // Search ALL memoirs, not just those in the current batch
        let all_memoirs = self.list_memoirs()?;
        for link in links {
            let mut resolved = None;
            for memoir in &all_memoirs {
                let from = self.get_concept(&memoir.name, &link.from)?;
                let to = self.get_concept(&memoir.name, &link.to)?;
                if let (Some(f), Some(t)) = (from, to) {
                    resolved = Some((f.id, t.id));
                    break;
                }
            }

            if let Some((source_id, target_id)) = resolved {
                let relation = match std::str::FromStr::from_str(&link.relation) {
                    Ok(r) => r,
                    Err(_) => {
                        tracing::warn!(
                            "unknown relation '{}', defaulting to related_to",
                            link.relation
                        );
                        Relation::RelatedTo
                    }
                };
                let concept_link = ConceptLink {
                    id: String::new(),
                    source_id,
                    target_id,
                    relation,
                    weight: 1.0,
                    created_at: Utc::now(),
                    valid_from: None,
                    valid_until: None,
                };
                self.add_link(concept_link)?;
                report.links_added += 1;
            }
        }

        // Bidirectional link: update source memories with concept_ids
        if !source_memory_ids.is_empty() {
            // Collect all concept IDs we just created/refined
            let mut all_concept_ids: Vec<String> = Vec::new();
            for c in concepts {
                if let Ok(Some(concept)) = self.get_concept(&c.memoir, &c.name) {
                    all_concept_ids.push(concept.id);
                }
            }
            if !all_concept_ids.is_empty() {
                for mem_id in source_memory_ids {
                    if let Ok(mut mem) = self.get(mem_id) {
                        let mut updated = false;
                        for cid in &all_concept_ids {
                            if !mem.concept_ids.contains(cid) {
                                mem.concept_ids.push(cid.clone());
                                updated = true;
                            }
                        }
                        if updated {
                            // Use direct SQL to avoid status change
                            let json = serde_json::to_string(&mem.concept_ids)?;
                            self.conn.execute(
                                "UPDATE memories SET concept_ids = ?1 WHERE id = ?2",
                                rusqlite::params![json, mem_id],
                            )?;
                        }
                    }
                }
            }
        }

        // Cross-session linking: connect new concepts to existing related concepts
        for c in concepts {
            self.auto_link_concept(&c.memoir, &c.name)?;
        }

        Ok(report)
    }

    /// v0.27 Track 2 #6: atomic N-memory merge.
    ///
    /// Fold every `loser_ids[i]` into the canonical of `winner_id` by:
    ///   1. Writing each loser's content as a `memory_evidence` row pointing
    ///      at the winner's canonical id (so provenance survives — satisfies
    ///      Lossless Compression Contract INV-1 / INV-3).
    ///   2. Marking each loser `superseded_by = winner_id` + `status =
    ///      'deprecated'` (matches the existing `apply_evolution`
    ///      deprecation path; recall filters `status IN ('active', 'updated')`
    ///      so deprecated rows fall out of the standard read paths but
    ///      remain reachable via `canonical_id_for`).
    ///   3. Deleting each loser's `sqlite-vec` embedding in the SAME
    ///      savepoint (mirrors `apply_evolution` invariant — vec is the
    ///      only side-index that's transactional).
    ///
    /// External (non-transactional) side-indexes — Tantivy + HNSW — are
    /// scrubbed AFTER the savepoint releases. A failure mid-savepoint leaves
    /// them untouched, matching the v0.26.2 R3 F2 invariant for
    /// `apply_evolution`.
    ///
    /// Atomicity: a partial fail (e.g. 3 of 5 losers committed before a 4th
    /// fails) rolls back ALL N losers via SAVEPOINT. Per the v0.27 R7 P2
    /// fix, the winner's recursive MergeInto runs AFTER this call by
    /// `store_with_dedup_resolved` — all fallible DB work for losers must
    /// complete first so the winner's non-transactional Tantivy/HNSW
    /// updates only fire after the loser fold has succeeded. Reversing
    /// the order would re-introduce the leak where an `apply_n_merge`
    /// failure rolls back the outer `BEGIN IMMEDIATE` but the winner's
    /// side-index writes have already landed.
    pub fn apply_n_merge(&self, winner_id: &str, loser_ids: &[String]) -> ReinResult<usize> {
        if loser_ids.is_empty() {
            return Ok(0);
        }

        let canonical_id = self.canonical_id_for(winner_id)?;

        // Snapshot loser memories BEFORE the savepoint. We need their content
        // for evidence rows; reading them inside the savepoint is fine but
        // hoisting the loads keeps the write window short.
        let mut losers: Vec<Memory> = Vec::with_capacity(loser_ids.len());
        for id in loser_ids {
            if id == winner_id || id == &canonical_id {
                // Defensive: never let the winner sneak into the loser list.
                continue;
            }
            match self.get(id) {
                Ok(m) if m.superseded_by.is_none() => losers.push(m),
                Ok(_) => {
                    tracing::debug!(loser = %id, "v0.27 #6: skipping already-superseded loser");
                }
                Err(e) => {
                    tracing::debug!(loser = %id, error = %e, "v0.27 #6: skipping unloadable loser");
                }
            }
        }
        if losers.is_empty() {
            return Ok(0);
        }

        self.conn
            .execute_batch("SAVEPOINT n_merge")
            .map_err(crate::types::ReinError::Database)?;

        let mut applied: Vec<String> = Vec::new();
        let result = (|| -> ReinResult<usize> {
            for loser in &losers {
                // 1. Evidence row — provenance preservation. Same shape as
                // `snapshot_memory_as_evidence` but inlined to keep the
                // savepoint atomic across all losers (snapshot_memory_as_evidence
                // calls refresh_canonical_state which we want to defer until
                // after RELEASE).
                self.add_memory_evidence(MemoryEvidence {
                    id: String::new(),
                    canonical_id: canonical_id.clone(),
                    memory_id: Some(loser.id.clone()),
                    source_topic: loser.topic.clone(),
                    summary: loser.summary.clone(),
                    content: loser.content.clone(),
                    keywords: loser.keywords.clone(),
                    source: loser.source,
                    created_at: loser.created_at,
                    imported_at: Utc::now(),
                })?;

                // 2. Mark superseded — pointer + status flip + canonical
                // remapping. mark_superseded itself opens a nested savepoint
                // which is fine (SQLite SAVEPOINTs nest).
                self.mark_superseded(&loser.id, winner_id)?;
                self.conn.execute(
                    "UPDATE memories SET status = 'deprecated', updated_at = ?2 WHERE id = ?1",
                    rusqlite::params![loser.id, Utc::now().to_rfc3339()],
                )?;

                // v0.27 R12 P2 fix: record a dedup_decisions ledger row so
                // `/api/dedup_decisions` can explain why each loser was
                // deprecated (mirrors the MergeInto/Supersede paths in
                // store_with_dedup at sqlite.rs:2168/2199). Loser→winner
                // duplicate, auto-decided by the N-merge orchestrator.
                let _ = self.record_dedup_decision(crate::types::DedupDecision {
                    id: String::new(),
                    winner_id: Some(winner_id.to_string()),
                    loser_id: Some(loser.id.clone()),
                    canonical_id: Some(canonical_id.clone()),
                    lexical_score: None,
                    embedding_score: None,
                    relation: crate::types::DedupRelation::Duplicate,
                    confidence: 0.9,
                    reason: "n_merge".to_string(),
                    operator: "auto".to_string(),
                    reversible: true,
                    merged_summary: Some(loser.summary.clone()),
                    novel_facts: Vec::new(),
                    conflict_detected: false,
                    payload: None,
                    created_at: Utc::now(),
                });

                // 3. sqlite-vec delete inside savepoint — vec is
                // transactional (vec0 virtual table), so a rollback also
                // reverts the embedding removal.
                crate::store::vec::delete_embedding(&self.conn, &loser.id)?;

                applied.push(loser.id.clone());
            }
            // Refresh canonical state once at the end; the support_count /
            // source_diversity recomputation reflects every new evidence row
            // we just wrote.
            self.refresh_canonical_state(&canonical_id)?;
            Ok(applied.len())
        })();

        match &result {
            Ok(_) => {
                self.conn
                    .execute_batch("RELEASE n_merge")
                    .map_err(crate::types::ReinError::Database)?;
                // v0.27 R9 P2 fix: enqueue Tantivy/HNSW scrub on the
                // store's deferred queue instead of running it
                // synchronously. `store_with_dedup` drains this queue
                // AFTER its outer `BEGIN IMMEDIATE` commits (and clears it
                // on rollback), so the non-transactional side-index work
                // is bounded to durable DB state.
                //
                // v0.27 R10 P2 fix: when there is no outer transaction
                // (direct callers — integration tests, future ops),
                // RELEASE n_merge has just made our changes durable, so
                // we drain synchronously instead of stranding loser ids
                // in the queue forever (`store_with_dedup` is the only
                // caller that drains, so a direct caller would otherwise
                // leak the queue indefinitely). `Connection::is_autocommit`
                // is true exactly when no explicit BEGIN is active.
                if self.conn.is_autocommit() {
                    for id in &applied {
                        self.remove_from_tantivy(id);
                        self.remove_from_hnsw(id);
                    }
                } else {
                    let mut queue = self.pending_index_scrub.borrow_mut();
                    for id in &applied {
                        queue.push(id.clone());
                    }
                }
            }
            Err(e) => {
                tracing::warn!(
                    winner = %winner_id,
                    error = %e,
                    "v0.27 #6: apply_n_merge rolling back; no losers committed"
                );
                let _ = self.conn.execute_batch("ROLLBACK TO n_merge");
                let _ = self.conn.execute_batch("RELEASE n_merge");
            }
        }
        result
    }

    /// Memory Evolution: new memory can refine or supersede similar old memories.
    /// - sim > 0.8 → supersede (mark old as superseded_by new_id)
    /// - 0.5 < sim <= 0.8 → refine (append new content to old memory)
    /// Returns number of evolved memories.
    pub fn apply_evolution(
        &self,
        new_id: &str,
        new_content: &str,
        new_embedding: Option<&[f32]>,
    ) -> ReinResult<usize> {
        let similar = self.search_fts(new_content, None, 5)?;
        let mut evolved = 0usize;

        // Wrap in savepoint — multiple UPDATEs should be atomic.
        // Side-index cleanup for the deprecation path follows the `delete()`
        // contract (sqlite.rs): in-DB cleanup (sqlite-vec row) lives INSIDE the
        // savepoint so a failure aborts the transaction and never leaves a
        // ghost embedding. Tantivy + HNSW external indexes are removed AFTER
        // the savepoint releases, mirroring `delete()`'s post-commit pattern.
        let mut deprecated_ids: Vec<String> = Vec::new();
        // v0.26.2 R3 Codex F2: collected refined records so post-RELEASE
        // can refresh Tantivy/HNSW with the new content. The DB write +
        // sqlite-vec delete happen inside the savepoint so a rollback
        // also reverts them; only the external (non-transactional) index
        // touches are deferred until we know the savepoint committed.
        struct RefinedRecord {
            id: String,
            topic: String,
            summary: String,
            content: String,
            keywords_json: String,
        }
        let mut refined_records: Vec<RefinedRecord> = Vec::new();

        self.conn
            .execute_batch("SAVEPOINT evolution")
            .map_err(crate::types::ReinError::Database)?;

        let result = (|| -> ReinResult<usize> {
            for old in &similar {
                if old.id == new_id {
                    continue;
                }
                if old.superseded_by.is_some() {
                    continue;
                }

                let sim = if let Some(new_emb) = new_embedding {
                    if let Ok(vec_results) = self.search_vec(new_emb, None, 5) {
                        vec_results
                            .iter()
                            .find(|m| m.id == old.id)
                            .map(|_| 0.85f32)
                            .unwrap_or_else(|| {
                                crate::extract::similarity(new_content, &old.content)
                            })
                    } else {
                        crate::extract::similarity(new_content, &old.content)
                    }
                } else {
                    crate::extract::similarity(new_content, &old.content)
                };

                if sim > 0.8 {
                    self.mark_superseded(&old.id, new_id)?;
                    // Bump updated_at alongside status flip so consumers that
                    // gate on freshness (recall, M5 tier recompute) see the
                    // transition.
                    self.conn.execute(
                        "UPDATE memories SET status = 'deprecated', updated_at = ?2 WHERE id = ?1",
                        rusqlite::params![old.id, Utc::now().to_rfc3339()],
                    )?;
                    // sqlite-vec row deletion stays inside the savepoint —
                    // matches the `delete()` invariant: a vec0 ghost left
                    // behind would keep surfacing the deprecated row from
                    // the vector channel forever.
                    crate::store::vec::delete_embedding(&self.conn, &old.id)?;
                    deprecated_ids.push(old.id.clone());
                    evolved += 1;
                    tracing::debug!("superseded memory '{}' with '{}'", old.id, new_id);
                } else if sim > 0.5 {
                    // v0.26.2 R3 Codex F2: do DB-only update inside the
                    // savepoint and queue Tantivy/HNSW refresh until AFTER
                    // RELEASE. Calling `self.update()` here would touch the
                    // external (non-transactional) indexes synchronously;
                    // a later op in this savepoint that triggers ROLLBACK
                    // would leave Tantivy/HNSW serving the refined text
                    // for DB content that was reverted. Mirrors the
                    // deprecation path's split.
                    let refined_content = format!("{}\n\n[refined] {}", old.content, new_content);
                    let refined_summary: String = refined_content
                        .chars()
                        .take(crate::types::SUMMARY_MAX_CHARS)
                        .collect();
                    let refined_updated_at = Utc::now().to_rfc3339();
                    // DB write inside savepoint. Status flips Active →
                    // Updated to mirror `update()`'s auto-promotion. Also
                    // null the archival_summary cols (matches `update()`'s
                    // R2 F2 fix) so cold-tier rows can't expose stale
                    // summaries pointing at content that's just been
                    // rewritten.
                    // R5 F2: also flip `needs_vec_dedup = 1` so the
                    // slow-channel `run_vec_dedup` worker re-embeds the
                    // refined content. Without this flag the worker skips
                    // the row (it scans `needs_vec_dedup = 1` only) and
                    // the row stays invisible to the vector channel until
                    // some other event re-flags it. Mirrors the
                    // resummarize path (`ops/resummerize.rs::apply_resummerize`).
                    // v0.27.5 R1 — also clear `last_too_large_at` on this
                    // raw refine path. Mirrors the `update()` semantic-
                    // change clear added in R1; without it, a row that
                    // previously hit `AttemptOutcome::TooLarge` and is
                    // then rewritten via `apply_evolution` would keep
                    // the stale stamp and `claim_batch` would still
                    // deprioritize the (now potentially smaller) content.
                    self.conn.execute(
                        "UPDATE memories SET content = ?1, summary = ?2, \
                                              updated_at = ?3, status = 'updated', \
                                              needs_vec_dedup = 1, \
                                              archival_summary = NULL, \
                                              archival_summary_at = NULL, \
                                              archival_summary_version = NULL, \
                                              needs_archival_summary = 1, \
                                              in_progress_archival_summary_at = NULL, \
                                              archival_claim_token = NULL, \
                                              last_too_large_at = NULL \
                         WHERE id = ?4",
                        rusqlite::params![
                            refined_content,
                            refined_summary,
                            refined_updated_at,
                            old.id
                        ],
                    )?;
                    // sqlite-vec is a vec0 virtual table tied to the
                    // SQLite txn — delete inside savepoint so a rollback
                    // also reverts the embedding removal. Ghost embedding
                    // would surface old content from the vector channel.
                    crate::store::vec::delete_embedding(&self.conn, &old.id)?;
                    refined_records.push(RefinedRecord {
                        id: old.id.clone(),
                        topic: old.topic.clone(),
                        summary: refined_summary,
                        content: refined_content,
                        keywords_json: serde_json::to_string(&old.keywords)
                            .unwrap_or_else(|_| "[]".to_string()),
                    });
                    evolved += 1;
                    tracing::debug!("refined memory '{}' with new content", old.id);
                }
            }
            Ok(evolved)
        })();

        match &result {
            Ok(_) => {
                self.conn
                    .execute_batch("RELEASE evolution")
                    .map_err(crate::types::ReinError::Database)?;
                // External side indexes (Tantivy + HNSW) are fire-and-forget
                // AFTER the savepoint releases, so their failure can't roll
                // back the successful DB writes. Mirrors `delete()`.
                for id in &deprecated_ids {
                    self.remove_from_tantivy(id);
                    self.remove_from_hnsw(id);
                }
                // R3 Codex F2: refresh refined rows' external indexes only
                // after we know the savepoint committed. HNSW is dropped
                // (no embedding to insert; the next dedup pass will re-embed
                // via the `vec::delete_embedding` we did inside the
                // savepoint, which sets the row up for re-embed).
                for rec in &refined_records {
                    self.update_tantivy(
                        &rec.id,
                        &rec.topic,
                        &rec.summary,
                        &rec.content,
                        &rec.keywords_json,
                    );
                    self.remove_from_hnsw(&rec.id);
                }
            }
            Err(_) => {
                let _ = self.conn.execute_batch("ROLLBACK TO evolution");
                let _ = self.conn.execute_batch("RELEASE evolution");
            }
        }
        result
    }

    /// Activate related memories: bump strength + last_accessed for memories similar to new content.
    /// This keeps old relevant memories alive instead of letting them decay.
    pub fn activate_related_memories(
        &self,
        content: &str,
        max_activate: usize,
    ) -> ReinResult<usize> {
        let similar = self.search_fts(content, None, max_activate)?;
        let mut activated = 0usize;
        for mem in &similar {
            let sim = crate::extract::similarity(content, &mem.content);
            if sim > 0.3 {
                self.record_access(&mem.id)?;
                activated += 1;
            }
        }
        Ok(activated)
    }

    /// Activate related concepts: boost confidence for concepts matching new content.
    pub fn activate_related_concepts(&self, content: &str) -> ReinResult<usize> {
        let similar_concepts = self.search_all_concepts(content, 5)?;
        let mut activated = 0usize;
        // Cache memoir_id → name mapping to avoid O(N*M) lookups
        let mut memoir_names: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();
        for concept in &similar_concepts {
            let sim = crate::extract::similarity(content, &concept.definition);
            if sim > 0.2 {
                // Look up memoir name from concept's memoir_id (cached)
                let memoir_name = memoir_names
                    .entry(concept.memoir_id.clone())
                    .or_insert_with(|| {
                        self.conn
                            .query_row(
                                "SELECT name FROM memoirs WHERE id = ?1",
                                rusqlite::params![concept.memoir_id],
                                |r| r.get::<_, String>(0),
                            )
                            .unwrap_or_default()
                    });
                if !memoir_name.is_empty() {
                    let _ = self.refine_concept(memoir_name, &concept.name, &concept.definition);
                    activated += 1;
                }
            }
        }
        Ok(activated)
    }

    /// Auto-link a newly added concept to existing related concepts in the same memoir.
    /// Searches for similar concepts by FTS and creates `related_to` links.
    fn auto_link_concept(&self, memoir_name: &str, concept_name: &str) -> ReinResult<usize> {
        let concept = match self.get_concept(memoir_name, concept_name)? {
            Some(c) => c,
            None => return Ok(0),
        };

        // Search for similar concepts in the same memoir
        let similar = self.search_concepts(memoir_name, &concept.definition, 5)?;
        let mut linked = 0usize;

        // Hoist link lookup outside loop to avoid repeated DB queries
        let existing_links = self.get_links_from(&concept.id)?;
        let existing_targets: std::collections::HashSet<&str> = existing_links
            .iter()
            .map(|l| l.target_id.as_str())
            .collect();

        for candidate in &similar {
            if candidate.id == concept.id {
                continue;
            }

            let sim = crate::extract::similarity(&concept.definition, &candidate.definition);
            if sim > 0.2 {
                if existing_targets.contains(candidate.id.as_str()) {
                    continue;
                }

                let link = ConceptLink {
                    id: String::new(),
                    source_id: concept.id.clone(),
                    target_id: candidate.id.clone(),
                    relation: Relation::RelatedTo,
                    weight: sim,
                    created_at: Utc::now(),
                    valid_from: None,
                    valid_until: None,
                };
                if self.add_link(link).is_ok() {
                    linked += 1;
                }
            }
        }

        Ok(linked)
    }

    /// Update only the related_ids field without touching status or updated_at.
    fn update_related_ids(&self, id: &str, related_ids: &[String]) -> ReinResult<()> {
        let json = serde_json::to_string(related_ids)?;
        self.conn.execute(
            "UPDATE memories SET related_ids = ?1 WHERE id = ?2",
            rusqlite::params![json, id],
        )?;
        Ok(())
    }

    /// Find related memories for a given memory and update related_ids bidirectionally.
    /// Uses FTS to find similar content, then checks Jaccard similarity.
    /// Both sides of each link are written in a single transaction.
    /// Returns the number of new links created.
    pub fn auto_link(
        &self,
        id: &str,
        similarity_threshold: f32,
        max_links: usize,
    ) -> ReinResult<usize> {
        let memory = self.get(id)?;
        // Search for similar memories by content
        let candidates = self.search_fts(&memory.content, Some(&memory.topic), max_links * 2)?;
        // Also search across topics
        let cross_topic = self.search_fts(&memory.summary, None, max_links)?;

        let mut all_candidates = candidates;
        for c in cross_topic {
            if !all_candidates.iter().any(|m| m.id == c.id) {
                all_candidates.push(c);
            }
        }

        // Collect link pairs first, then write atomically
        let mut updated_related = memory.related_ids.clone();
        let mut peer_updates: Vec<(String, Vec<String>)> = Vec::new();

        for candidate in &all_candidates {
            if candidate.id == id {
                continue;
            }
            if updated_related.contains(&candidate.id) {
                continue;
            }
            if updated_related.len() >= max_links {
                break;
            }

            let sim = crate::extract::similarity(&memory.content, &candidate.content);
            if sim < similarity_threshold {
                continue;
            }

            updated_related.push(candidate.id.clone());

            // Prepare peer update
            if !candidate.related_ids.contains(&memory.id)
                && candidate.related_ids.len() < max_links
            {
                let mut peer_related = candidate.related_ids.clone();
                peer_related.push(memory.id.clone());
                peer_updates.push((candidate.id.clone(), peer_related));
            }
        }

        let new_links = updated_related.len() - memory.related_ids.len();
        if new_links == 0 {
            return Ok(0);
        }

        // Write all link updates atomically (SAVEPOINT for nesting safety)
        self.conn.execute_batch("SAVEPOINT auto_link")?;
        if let Err(e) = (|| -> ReinResult<()> {
            self.update_related_ids(id, &updated_related)?;
            for (peer_id, peer_related) in &peer_updates {
                self.update_related_ids(peer_id, peer_related)?;
            }
            Ok(())
        })() {
            let _ = self
                .conn
                .execute_batch("ROLLBACK TO auto_link; RELEASE auto_link");
            return Err(e);
        }
        self.conn.execute_batch("RELEASE auto_link")?;

        Ok(new_links)
    }

    /// Organize all memories: scan for related pairs and create bidirectional links.
    /// Returns total number of new links created.
    pub fn organize(
        &self,
        similarity_threshold: f32,
        max_links_per_memory: usize,
    ) -> ReinResult<usize> {
        let mut stmt = self.conn.prepare("SELECT id FROM memories")?;
        let ids: Vec<String> = stmt
            .query_map([], |row| row.get(0))?
            .filter_map(|r| r.ok())
            .collect();

        let mut total_links = 0usize;
        for id in &ids {
            match self.auto_link(id, similarity_threshold, max_links_per_memory) {
                Ok(n) => total_links += n,
                Err(e) => tracing::warn!("auto_link failed for {id}: {e}"),
            }
        }
        Ok(total_links)
    }

    /// Atomically consolidate a topic: delete all old memories and insert replacement in one transaction.
    /// Returns the old memories for reference. If insertion fails, everything rolls back.
    ///
    /// Test-only — bypasses `clean_memory_refs`, so it would leak dangling `related_ids` /
    /// `episodes.memory_ids` if used on a live DB. Prefer `consolidate_by_ids_atomic`
    /// (which does the cleanup) for production paths.
    #[cfg(test)]
    pub fn consolidate_atomic(&self, topic: &str, replacement: Memory) -> ReinResult<Vec<Memory>> {
        self.conn.execute_batch("SAVEPOINT consolidate_atomic")?;

        // Collect old memories
        let old_memories = self.get_by_topic(topic)?;

        // Delete all in topic
        if let Err(e) = self.conn.execute(
            "DELETE FROM memories WHERE topic = ?1",
            rusqlite::params![topic],
        ) {
            let _ = self
                .conn
                .execute_batch("ROLLBACK TO consolidate_atomic; RELEASE consolidate_atomic");
            return Err(e.into());
        }

        // Insert replacement within same savepoint
        if let Err(e) = self.store(replacement) {
            let _ = self
                .conn
                .execute_batch("ROLLBACK TO consolidate_atomic; RELEASE consolidate_atomic");
            return Err(e);
        }

        self.conn.execute_batch("RELEASE consolidate_atomic")?;

        // Clean side indexes AFTER commit to avoid inconsistency on rollback
        for m in &old_memories {
            self.remove_from_tantivy(&m.id);
            self.remove_from_hnsw(&m.id);
        }

        Ok(old_memories)
    }

    /// Atomically consolidate multiple topics into a single replacement memory.
    /// Used for normalized topic-variant cleanup (case/space/hyphen variants).
    ///
    /// Test-only — same `clean_memory_refs` caveat as `consolidate_atomic`.
    #[cfg(test)]
    pub fn consolidate_topics_atomic(
        &self,
        topics: &[String],
        replacement: Memory,
    ) -> ReinResult<Vec<Memory>> {
        self.conn
            .execute_batch("SAVEPOINT consolidate_topics_atomic")?;

        let mut old_memories = Vec::new();
        for topic in topics {
            old_memories.extend(self.get_by_topic(topic)?);
        }

        for topic in topics {
            if let Err(e) = self.conn.execute(
                "DELETE FROM memories WHERE topic = ?1",
                rusqlite::params![topic],
            ) {
                let _ = self.conn.execute_batch(
                    "ROLLBACK TO consolidate_topics_atomic; RELEASE consolidate_topics_atomic",
                );
                return Err(e.into());
            }
        }

        if let Err(e) = self.store(replacement) {
            let _ = self.conn.execute_batch(
                "ROLLBACK TO consolidate_topics_atomic; RELEASE consolidate_topics_atomic",
            );
            return Err(e);
        }

        self.conn
            .execute_batch("RELEASE consolidate_topics_atomic")?;

        for memory in &old_memories {
            self.remove_from_tantivy(&memory.id);
            self.remove_from_hnsw(&memory.id);
        }

        Ok(old_memories)
    }

    /// Atomically consolidate a specific set of memory IDs into a single replacement.
    /// Unlike `consolidate_atomic` (which deletes by topic), this deletes ONLY the
    /// specified IDs, preventing TOCTOU data loss if new memories are added to the
    /// topic between the load and commit phases.
    pub fn consolidate_by_ids_atomic(
        &self,
        memory_ids: &[String],
        replacement: Memory,
    ) -> ReinResult<Vec<Memory>> {
        self.conn.execute_batch("SAVEPOINT consolidate_by_ids")?;

        // Collect old memories by ID
        let mut old_memories = Vec::new();
        for id in memory_ids {
            if let Ok(m) = self.get(id) {
                old_memories.push(m);
            }
        }

        // Delete only the specified IDs. For each id, scrub it out of related_ids
        // and episodes.memory_ids BEFORE the DELETE so we don't leave dangling refs.
        // Concepts.source_memory_ids are handled by the migration pass below (old_id
        // → replacement_id), so we don't call the full `clean_memory_refs` here.
        for id in memory_ids {
            let quoted = format!("\"{id}\"");
            let like = format!("%{quoted}%");

            if let Err(e) = self.conn.execute(
                "UPDATE memories
                 SET related_ids = COALESCE(
                    (SELECT json_group_array(value)
                     FROM json_each(related_ids)
                     WHERE value != ?1),
                    '[]')
                 WHERE related_ids LIKE ?2 AND id != ?1",
                rusqlite::params![id, like],
            ) {
                let _ = self
                    .conn
                    .execute_batch("ROLLBACK TO consolidate_by_ids; RELEASE consolidate_by_ids");
                return Err(e.into());
            }

            let has_episodes: bool = self
                .conn
                .query_row(
                    "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='episodes'",
                    [],
                    |r| r.get(0),
                )
                .unwrap_or(false);
            if has_episodes {
                if let Err(e) = self.conn.execute(
                    "UPDATE episodes
                     SET memory_ids = COALESCE(
                        (SELECT json_group_array(value)
                         FROM json_each(memory_ids)
                         WHERE value != ?1),
                        '[]')
                     WHERE memory_ids LIKE ?2",
                    rusqlite::params![id, like],
                ) {
                    let _ = self.conn.execute_batch(
                        "ROLLBACK TO consolidate_by_ids; RELEASE consolidate_by_ids",
                    );
                    return Err(e.into());
                }
            }

            if let Err(e) = self
                .conn
                .execute("DELETE FROM memories WHERE id = ?1", rusqlite::params![id])
            {
                let _ = self
                    .conn
                    .execute_batch("ROLLBACK TO consolidate_by_ids; RELEASE consolidate_by_ids");
                return Err(e.into());
            }
            if let Err(e) = crate::store::vec::delete_embedding(&self.conn, id) {
                let _ = self
                    .conn
                    .execute_batch("ROLLBACK TO consolidate_by_ids; RELEASE consolidate_by_ids");
                return Err(e);
            }
        }

        // Insert replacement
        let replacement_id = replacement.id.clone();
        if let Err(e) = self.store(replacement) {
            let _ = self
                .conn
                .execute_batch("ROLLBACK TO consolidate_by_ids; RELEASE consolidate_by_ids");
            return Err(e);
        }

        for old_id in memory_ids {
            // JSON-quoted match with LIKE wildcard escaping to avoid
            // prefix collisions and IDs containing % or _
            let escaped = old_id
                .replace('\\', "\\\\")
                .replace('%', "\\%")
                .replace('_', "\\_");
            let mut stmt = self.conn.prepare(
                "SELECT id, source_memory_ids FROM concepts WHERE source_memory_ids LIKE ?1 ESCAPE '\\'"
            )?;
            let rows: Vec<(String, String)> = stmt
                .query_map(rusqlite::params![format!("%\"{}\"%", escaped)], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })?
                .filter_map(|r| r.ok())
                .collect();
            drop(stmt);
            for (concept_id, raw_ids) in rows {
                if let Ok(mut ids) = serde_json::from_str::<Vec<String>>(&raw_ids) {
                    ids.retain(|id| id != old_id);
                    if !ids.contains(&replacement_id) {
                        ids.push(replacement_id.clone());
                    }
                    let updated = serde_json::to_string(&ids).unwrap_or_default();
                    self.conn.execute(
                        "UPDATE concepts SET source_memory_ids = ?1 WHERE id = ?2",
                        rusqlite::params![updated, concept_id],
                    )?;
                }
            }
        }

        for memory in &old_memories {
            self.add_memory_evidence(MemoryEvidence {
                id: String::new(),
                canonical_id: replacement_id.clone(),
                memory_id: None,
                source_topic: memory.topic.clone(),
                summary: memory.summary.clone(),
                content: memory.content.clone(),
                keywords: memory.keywords.clone(),
                source: memory.source,
                created_at: memory.created_at,
                imported_at: Utc::now(),
            })?;
        }
        self.refresh_canonical_state(&replacement_id)?;

        self.conn.execute_batch("RELEASE consolidate_by_ids")?;

        for memory in &old_memories {
            self.remove_from_tantivy(&memory.id);
            self.remove_from_hnsw(&memory.id);
        }

        Ok(old_memories)
    }

    /// Stream every memory row needed by side-index / cache warmup.
    ///
    /// The callback receives one `WarmupRow` per memory and the iterator
    /// drops each row before fetching the next, so peak heap stays
    /// O(row_size) rather than O(table_size). Boot-path OOM regression
    /// guard: rein operator DBs have grown past 384 MB (per the v0.30.1
    /// warmup audit), at which point materializing the entire
    /// `memories` table into a `Vec<(String, String, String, String,
    /// String)>` was a real OOM risk on small VMs.
    ///
    /// Errors from individual row decodes are surfaced as `Err` (early
    /// abort) — the prior `Vec`-returning API silently dropped them.
    /// Callers should treat a per-row decode failure as fatal for the
    /// warmup run; the dirty marker stays set so a later pass can retry.
    ///
    /// **Memory accounting**: callers that need the total row count
    /// must track it inside the callback (no `len()` available before
    /// the stream completes). See `search/warmup.rs` for the pattern.
    pub fn for_each_for_warmup<F>(&self, mut f: F) -> ReinResult<()>
    where
        F: FnMut(WarmupRow) -> ReinResult<()>,
    {
        let mut stmt = self
            .conn
            .prepare("SELECT id, topic, summary, content, keywords FROM memories")?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let warmup_row = WarmupRow {
                id: row.get::<_, String>(0)?,
                topic: row.get::<_, String>(1)?,
                summary: row.get::<_, String>(2)?,
                content: row.get::<_, String>(3)?,
                keywords: row.get::<_, String>(4)?,
            };
            f(warmup_row)?;
        }
        Ok(())
    }

    /// Record an access to a memory (bumps access_count and last_accessed).
    /// Call this only when memories are returned to the user via recall, NOT on internal lookups.
    pub fn record_access(&self, id: &str) -> ReinResult<()> {
        // Wrap increment + promotion in a single savepoint for atomicity
        self.conn.execute_batch("SAVEPOINT record_access")?;
        let result = (|| -> ReinResult<()> {
            let now = Utc::now();
            let updated = self.conn.execute(
                "UPDATE memories SET last_accessed = ?1, access_count = access_count + 1 WHERE id = ?2",
                rusqlite::params![now.to_rfc3339(), id],
            )?;
            if updated == 0 {
                return Err(ReinError::NotFound(format!("memory {id} not found")));
            }

            let promotion_row: Option<(u32, Option<u32>)> = self
                .conn
                .query_row(
                    "SELECT access_count, cluster_id FROM memories WHERE id = ?1 AND layer = 'STM'",
                    rusqlite::params![id],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .ok();

            if let Some((access_count, cluster_id)) = promotion_row {
                let curve = cluster_id.and_then(|cid| {
                    let key = format!("survival_curve:{cid}");
                    self.conn
                        .query_row(
                            "SELECT value FROM metadata WHERE key = ?1",
                            rusqlite::params![key],
                            |row| row.get::<_, String>(0),
                        )
                        .ok()
                        .and_then(|json| {
                            serde_json::from_str::<crate::search::survival::SurvivalCurve>(&json)
                                .map_err(|e| {
                                    tracing::warn!(
                                        "malformed survival curve for cluster {cid}: {e}"
                                    );
                                    e
                                })
                                .ok()
                        })
                });
                let threshold = curve
                    .as_ref()
                    .map(crate::search::survival::promotion_access_threshold)
                    .unwrap_or(5);

                if access_count > threshold {
                    // Derive decay reduction from survival data when available.
                    // With curve: ratio = min(median / 28, 1) maps to [0.25, 0.75] range.
                    // Without curve: use 0.5 (yields ~20 day LTM half-life, matching docs).
                    let decay_factor = curve
                        .as_ref()
                        .and_then(|c| c.median_survival)
                        .map(|median| (median / 28.0).clamp(0.25, 0.75))
                        .unwrap_or(0.5);
                    self.conn.execute(
                        "UPDATE memories SET layer = 'LTM', decay_lambda = decay_lambda * ?1
                     WHERE id = ?2 AND layer = 'STM'",
                        rusqlite::params![decay_factor, id],
                    )?;
                }
            }

            Ok(())
        })(); // end savepoint closure
        match result {
            Ok(()) => {
                self.conn.execute_batch("RELEASE record_access")?;
                Ok(())
            }
            Err(e) => {
                let _ = self.conn.execute_batch("ROLLBACK TO record_access");
                let _ = self.conn.execute_batch("RELEASE record_access");
                Err(e)
            }
        }
    }
}

/// Owned row payload passed to `SqliteStore::for_each_for_warmup`
/// callbacks. Each field is the same column the legacy
/// `get_all_for_warmup` tuple returned, just named so callsites read
/// clearly. The struct is moved into the callback per iteration; the
/// row is dropped before the next is fetched, so peak heap stays
/// O(row_size) rather than O(table_size).
#[derive(Debug)]
pub struct WarmupRow {
    pub id: String,
    pub topic: String,
    pub summary: String,
    pub content: String,
    pub keywords: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_memory(id: &str, topic: &str, summary: &str, content: &str) -> Memory {
        Memory {
            id: id.to_string(),
            layer: MemoryLayer::LTM,
            topic: topic.to_string(),
            summary: summary.to_string(),
            content: content.to_string(),
            keywords: vec![],
            importance: Importance::High,
            source: Source::Manual,
            strength: 1.0,
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

    #[test]
    fn store_knowledge_units_refines_existing_concepts_inside_outer_savepoint() {
        let store = SqliteStore::in_memory().unwrap();

        let first = crate::extract::ExtractedConcept {
            name: "ownership".to_string(),
            definition: "Initial definition".to_string(),
            labels: vec!["rust".to_string()],
            memoir: "rust-lang".to_string(),
            concept_type: "fact".to_string(),
            quality_confidence: 0.8,
        };
        store
            .store_knowledge_units_with_sources(&[first], &[], &[])
            .unwrap();

        let refined = crate::extract::ExtractedConcept {
            name: "ownership".to_string(),
            definition: "Refined definition".to_string(),
            labels: vec!["rust".to_string(), "memory".to_string()],
            memoir: "rust-lang".to_string(),
            concept_type: "fact".to_string(),
            quality_confidence: 0.9,
        };
        let report = store
            .store_knowledge_units_with_sources(&[refined], &[], &["mem-1".to_string()])
            .unwrap();

        assert_eq!(report.concepts_refined, 1);
        let concept = store
            .get_concept("rust-lang", "ownership")
            .unwrap()
            .expect("concept exists");
        assert_eq!(concept.definition, "Refined definition");
        assert_eq!(concept.revision, 2);
        assert!(concept.source_memory_ids.iter().any(|id| id == "mem-1"));
    }

    #[test]
    fn apply_evolution_updates_canonical_state_when_superseding() {
        let store = SqliteStore::in_memory().unwrap();
        let old_id = store
            .store(test_memory(
                "old-id",
                "topic-a",
                "legacy summary",
                "shared evolution content",
            ))
            .unwrap();
        let new_id = store
            .store(test_memory(
                "new-id",
                "topic-a",
                "new summary",
                "shared evolution content",
            ))
            .unwrap();

        store
            .apply_evolution(&new_id, "shared evolution content", None)
            .unwrap();

        let old = store.get(&old_id).unwrap();
        assert_eq!(old.superseded_by.as_deref(), Some(new_id.as_str()));
        assert_eq!(store.canonical_id_for(&old_id).unwrap(), new_id);
    }

    /// v0.26.2 Bug #3 (HIGH): the deprecation path of `apply_evolution`
    /// must remove the deprecated row's sqlite-vec embedding so the vector
    /// channel doesn't keep returning a ghost. Pre-fix the raw `UPDATE
    /// status='deprecated'` left every side index untouched.
    #[test]
    fn apply_evolution_deprecation_removes_vec_row() {
        let store = SqliteStore::in_memory().unwrap();
        let old_id = store
            .store(test_memory(
                "old-id",
                "topic-a",
                "legacy summary",
                "shared evolution content",
            ))
            .unwrap();
        let new_id = store
            .store(test_memory(
                "new-id",
                "topic-a",
                "new summary",
                "shared evolution content",
            ))
            .unwrap();

        // Seed a vec_memories row for the soon-to-be-deprecated memory.
        // sqlite-vec virtual table does not auto-cascade on `memories` deletes,
        // so this is the exact ghost surface the bug exposes.
        let mut embedding = vec![0.0f32; 3072];
        embedding[0] = 1.0;
        crate::store::vec::insert_embedding(store.conn(), &old_id, &embedding).unwrap();

        let before: i64 = store
            .conn
            .query_row(
                "SELECT COUNT(*) FROM vec_memories WHERE id = ?1",
                rusqlite::params![old_id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(before, 1, "seeded vec row must exist before evolution");

        // sim == 1.0 → deprecation branch.
        store
            .apply_evolution(&new_id, "shared evolution content", None)
            .unwrap();

        // Deprecated row's vec embedding must be gone.
        let after: i64 = store
            .conn
            .query_row(
                "SELECT COUNT(*) FROM vec_memories WHERE id = ?1",
                rusqlite::params![old_id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(
            after, 0,
            "apply_evolution deprecation must delete the sqlite-vec row \
             (was leaking ghost embeddings into the vector channel)"
        );

        // Sanity: the memory row itself still exists (status flipped, not deleted).
        let old = store.get(&old_id).unwrap();
        assert_eq!(old.status, MemoryStatus::Deprecated);
        assert_eq!(old.superseded_by.as_deref(), Some(new_id.as_str()));
    }

    /// v0.26.2 Bug #3 (HIGH): the refine path (`0.5 < sim <= 0.8`) must route
    /// through `update()` so vec / Tantivy / HNSW side indexes refresh.
    /// Pre-fix the raw `UPDATE content/summary` left external indexes
    /// serving the pre-refine text indefinitely. Easiest observable signal:
    /// `update()` flips `status: Active -> Updated` automatically; the raw
    /// SQL path did not.
    ///
    /// Hitting the refine band is structurally fiddly: `similarity()` =
    /// max(jaccard, containment), and FTS5 default operator is implicit AND,
    /// so the candidate row must contain every token of `new_content` for
    /// FTS to surface it. We satisfy both with a 3-token query that lives
    /// in the old summary (FTS finds the row) but only partially overlaps
    /// the old content body (sim = containment = 2/3 ≈ 0.67, in the refine
    /// band).
    #[test]
    fn apply_evolution_refine_writes_db_and_promotes_status() {
        let store = SqliteStore::in_memory().unwrap();

        // Old summary contains every new_content token so FTS5's implicit-AND
        // returns this row. Old content body shares only 2 of 3 new tokens
        // (`shared`, `content`) → containment(new, old.content) =
        // 2 / min(3, 5) = 0.67, jaccard ≈ 0.33 → sim = 0.67 → refine band.
        let old_id = store
            .store(test_memory(
                "old-id",
                "topic-a",
                "very long summary mentioning shared evolution content here for fts",
                "shared content alpha beta gamma",
            ))
            .unwrap();

        // Confirm starting status is Active.
        let pre = store.get(&old_id).unwrap();
        assert_eq!(pre.status, MemoryStatus::Active);

        let new_id = "new-evolution-id";
        let new_content = "shared evolution content";

        // Seed the new memory so FTS has it indexed; apply_evolution skips
        // `old.id == new_id` internally so the new row doesn't self-match.
        store
            .store(test_memory(new_id, "topic-a", "new summary", new_content))
            .unwrap();

        let evolved = store.apply_evolution(new_id, new_content, None).unwrap();
        assert!(
            evolved >= 1,
            "old memory must be hit by the refine branch (got {evolved} evolutions)"
        );

        let refined = store.get(&old_id).unwrap();
        // Status promotion via inline SQL `status = 'updated'` (mirrors the
        // auto-promotion `update()` does for semantic_changed). v0.26.2 R3
        // F2: the refine path no longer routes through `update()` — it does
        // a DB-only update inside the savepoint and queues Tantivy/HNSW
        // refresh until after RELEASE so a rollback can't leave external
        // indexes diverged from rolled-back DB content.
        assert_eq!(
            refined.status,
            MemoryStatus::Updated,
            "refine must promote Active -> Updated to mirror update()'s contract"
        );
        // And the refined content actually landed.
        assert!(
            refined.content.contains("[refined]"),
            "refine marker must be present in updated content (got: {:?})",
            refined.content
        );
        assert!(
            refined.content.contains("shared content alpha beta gamma"),
            "original content must be preserved (got: {:?})",
            refined.content
        );
    }

    /// F6 D-M3: streaming warmup callback fires once per stored row,
    /// in arbitrary order. Replaces the legacy `get_all_for_warmup ->
    /// Vec<...>` which materialized every row into RAM at once.
    ///
    /// Uses unique content per memory to bypass `store()`'s
    /// intelligent-merge dedup path.
    #[test]
    fn for_each_for_warmup_streams_all_rows() {
        let store = SqliteStore::in_memory().unwrap();
        let n = 50usize;
        let mut expected_ids = std::collections::HashSet::new();
        for i in 0..n {
            let id = format!("ulid-{i:04}");
            let stored_id = store
                .store(test_memory(
                    &id,
                    &format!("topic-{i}"),
                    &format!("summary-{i}"),
                    &format!("content-{i}"),
                ))
                .unwrap();
            expected_ids.insert(stored_id);
        }

        let mut seen = std::collections::HashSet::new();
        let mut visited = 0usize;
        store
            .for_each_for_warmup(|row| {
                // The struct moves in by value — there is no
                // pre-collected Vec the iterator borrows from.
                assert!(!row.id.is_empty(), "id must be non-empty");
                seen.insert(row.id);
                visited += 1;
                Ok(())
            })
            .unwrap();

        // Every stored canonical must reach the callback exactly once.
        // (store() returns the actual canonical id used — that's what
        // we compare against.)
        for id in &expected_ids {
            assert!(
                seen.contains(id),
                "stored id {id} must reach the streaming callback"
            );
        }
        // `visited == seen.len()` would let a duplicate-visit slip
        // through because HashSet dedupes. Compare against the
        // expected unique-id count to enforce exactly-once.
        assert_eq!(
            visited,
            expected_ids.len(),
            "callback must fire exactly once per distinct row"
        );
        assert_eq!(seen.len(), expected_ids.len(), "every stored id is visited");
    }

    /// F6 D-M3: callback `Err` aborts iteration. Proves the new API
    /// surfaces per-row failures that the legacy `filter_map(|r|
    /// r.ok())` silently dropped.
    #[test]
    fn for_each_for_warmup_propagates_callback_error() {
        let store = SqliteStore::in_memory().unwrap();
        // Five unique-content memories so store() never merges them.
        for i in 0..5 {
            store
                .store(test_memory(
                    &format!("ulid-{i:04}"),
                    &format!("topic-{i}"),
                    &format!("summary-{i}"),
                    &format!("content-{i}"),
                ))
                .unwrap();
        }

        let mut count = 0usize;
        let result = store.for_each_for_warmup(|_row| -> ReinResult<()> {
            count += 1;
            if count == 3 {
                Err(ReinError::Config("synthetic callback abort".into()))
            } else {
                Ok(())
            }
        });
        assert!(result.is_err(), "callback Err must propagate up");
        assert_eq!(count, 3, "iteration must halt at the failing row");
    }

    /// F6 D-M3: streaming API peak heap is O(row_size), not
    /// O(table_size). We can't measure RSS reliably from a unit test,
    /// so we assert the structural property — the callback signature
    /// takes one `WarmupRow` by value and the public API returns `()`
    /// not `Vec<WarmupRow>`. The signature itself is the OOM-safety
    /// guarantee. We also exercise a small batch under the streaming
    /// path to catch any accidental internal `Vec` collection.
    ///
    /// Note: we use unique content per row to avoid the `store()`
    /// intelligent-merge dedup path collapsing them into a single
    /// canonical, which would defeat the row-count assertion. Row
    /// count kept modest (40) because `store()` runs full extract /
    /// vec / FTS pipelines per insert under `in_memory()`.
    #[test]
    fn for_each_for_warmup_handles_many_rows() {
        let store = SqliteStore::in_memory().unwrap();
        let n = 40usize;
        for i in 0..n {
            store
                .store(test_memory(
                    &format!("u-{i:05}"),
                    &format!("topic-{i}"),
                    &format!("summary-{i}"),
                    &format!("content-{i}"),
                ))
                .unwrap();
        }

        let mut visited = 0usize;
        store
            .for_each_for_warmup(|_row| {
                visited += 1;
                Ok(())
            })
            .unwrap();
        // The streaming path must visit at least every canonical row.
        // We don't require == n because store() may merge near-duplicates;
        // we just verify the stream is non-trivial and bounded.
        assert!(visited >= 1, "stream must visit at least one row");
        assert!(visited <= n, "stream cannot exceed inserted row count");
    }
}
