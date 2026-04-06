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
                                        let _ = self.expire_link(&link.id, now);
                                    }
                                }
                            }
                        }
                        // Refine (use existing.name for exact DB match after normalized lookup)
                        if let Err(e) =
                            self.refine_concept(memoir_name, &existing.name, &c.definition)
                        {
                            tracing::warn!("failed to refine concept '{}': {e}", c.name);
                        } else {
                            report.concepts_refined += 1;
                        }
                        // Merge new source_memory_ids into existing concept
                        if !source_memory_ids.is_empty() {
                            let mut merged = existing.source_memory_ids.clone();
                            for sid in source_memory_ids {
                                if !merged.contains(sid) {
                                    merged.push(sid.clone());
                                }
                            }
                            if merged.len() > existing.source_memory_ids.len() {
                                let json = serde_json::to_string(&merged).unwrap_or_default();
                                let _ = self.conn().execute(
                                    "UPDATE concepts SET source_memory_ids = ?1 WHERE id = ?2",
                                    rusqlite::params![json, existing.id],
                                );
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
                        };
                        if let Err(e) = self.add_concept(concept) {
                            tracing::warn!("failed to add concept '{}': {e}", c.name);
                        } else {
                            report.concepts_added += 1;
                        }
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
                if let Err(e) = self.add_link(concept_link) {
                    tracing::warn!("failed to add link {} -> {}: {e}", link.from, link.to);
                } else {
                    report.links_added += 1;
                }
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
                            let json = serde_json::to_string(&mem.concept_ids).unwrap_or_default();
                            let _ = self.conn.execute(
                                "UPDATE memories SET concept_ids = ?1 WHERE id = ?2",
                                rusqlite::params![json, mem_id],
                            );
                        }
                    }
                }
            }
        }

        // Cross-session linking: connect new concepts to existing related concepts
        for c in concepts {
            if let Err(e) = self.auto_link_concept(&c.memoir, &c.name) {
                tracing::debug!("auto_link_concept failed for '{}': {e}", c.name);
            }
        }

        Ok(report)
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

        // Wrap in savepoint — multiple UPDATEs should be atomic
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
                    self.conn.execute(
                        "UPDATE memories SET superseded_by = ?1, status = 'deprecated' WHERE id = ?2",
                        rusqlite::params![new_id, old.id],
                    )?;
                    evolved += 1;
                    tracing::debug!("superseded memory '{}' with '{}'", old.id, new_id);
                } else if sim > 0.5 {
                    let refined_content = format!("{}\n\n[refined] {}", old.content, new_content);
                    let refined_summary: String = refined_content.chars().take(100).collect();
                    self.conn.execute(
                        "UPDATE memories SET content = ?1, summary = ?2, updated_at = ?3 WHERE id = ?4",
                        rusqlite::params![refined_content, refined_summary, Utc::now().to_rfc3339(), old.id],
                    )?;
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
        self.conn
            .execute_batch("SAVEPOINT consolidate_by_ids")?;

        // Collect old memories by ID
        let mut old_memories = Vec::new();
        for id in memory_ids {
            if let Ok(m) = self.get(id) {
                old_memories.push(m);
            }
        }

        // Delete only the specified IDs
        for id in memory_ids {
            if let Err(e) = self.conn.execute(
                "DELETE FROM memories WHERE id = ?1",
                rusqlite::params![id],
            ) {
                let _ = self.conn.execute_batch(
                    "ROLLBACK TO consolidate_by_ids; RELEASE consolidate_by_ids",
                );
                return Err(e.into());
            }
        }

        // Insert replacement
        if let Err(e) = self.store(replacement) {
            let _ = self.conn.execute_batch(
                "ROLLBACK TO consolidate_by_ids; RELEASE consolidate_by_ids",
            );
            return Err(e);
        }

        self.conn
            .execute_batch("RELEASE consolidate_by_ids")?;

        for memory in &old_memories {
            self.remove_from_tantivy(&memory.id);
            self.remove_from_hnsw(&memory.id);
        }

        Ok(old_memories)
    }

    /// Get all memory (id, topic, summary, content, keywords) tuples for cache warmup.
    pub fn get_all_for_warmup(&self) -> ReinResult<Vec<(String, String, String, String, String)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, topic, summary, content, keywords FROM memories")?;
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
            ))
        })?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    /// Record an access to a memory (bumps access_count and last_accessed).
    /// Call this only when memories are returned to the user via recall, NOT on internal lookups.
    pub fn record_access(&self, id: &str) -> ReinResult<()> {
        let now = Utc::now();
        self.conn.execute(
            "UPDATE memories SET last_accessed = ?1, access_count = access_count + 1 WHERE id = ?2",
            rusqlite::params![now.to_rfc3339(), id],
        )?;

        let promotion_row: Option<(u32, Option<u32>)> = self
            .conn
            .query_row(
                "SELECT access_count, cluster_id FROM memories WHERE id = ?1 AND layer = 'STM'",
                rusqlite::params![id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .ok();

        if let Some((access_count, cluster_id)) = promotion_row {
            let threshold = cluster_id
                .and_then(|cid| {
                    let key = format!("survival_curve:{cid}");
                    self.conn
                        .query_row(
                            "SELECT value FROM metadata WHERE key = ?1",
                            rusqlite::params![key],
                            |row| row.get::<_, String>(0),
                        )
                        .ok()
                        .and_then(|json| serde_json::from_str::<crate::search::survival::SurvivalCurve>(&json).ok())
                })
                .map(|curve| crate::search::survival::promotion_access_threshold(&curve))
                .unwrap_or(5);

            if access_count > threshold {
                self.conn.execute(
                    "UPDATE memories SET layer = 'LTM', decay_lambda = decay_lambda * 0.33
                     WHERE id = ?1 AND layer = 'STM'",
                    rusqlite::params![id],
                )?;
            }
        }

        Ok(())
    }
}
