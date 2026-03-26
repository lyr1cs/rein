//! Quality scoring, weight learning, recall tracking, and concept pruning.

use crate::types::*;

use super::SqliteStore;

impl SqliteStore {
    /// Compute quality score for a concept based on multiple features.
    /// Uses self-learned weights when enough data is available, falls back to llm_confidence.
    pub fn concept_quality_score(&self, concept: &Concept) -> f64 {
        // Feature 1: LLM confidence (from extraction)
        let llm_conf = concept.confidence as f64;

        // Feature 2: Utility — source memories' access/recall ratio
        let utility = {
            let mut total_access = 0u64;
            let mut total_recall = 0u64;
            for mem_id in &concept.source_memory_ids {
                if let Ok(mem) = self.get(mem_id) {
                    total_access += mem.access_count as u64;
                }
                // Get recall count from metadata
                let recall: u64 = self.conn
                    .query_row(
                        "SELECT COALESCE(CAST(value AS INTEGER), 0) FROM metadata WHERE key = ?1",
                        rusqlite::params![format!("recall_hit:{}", mem_id)],
                        |r| r.get(0),
                    ).unwrap_or(0);
                total_recall += recall;
            }
            if total_recall > 0 { total_access as f64 / total_recall as f64 } else { 0.5 }
        };

        // Feature 3: Connectivity — link count as graph structure signal
        let link_count = self.get_links_from(&concept.id).map(|l| l.len()).unwrap_or(0)
            + self.get_links_to(&concept.id).map(|l| l.len()).unwrap_or(0);
        let connectivity = (link_count as f64 / 3.0).min(1.0);

        // Feature 4: Recency — boost for recent concepts
        let hours = (chrono::Utc::now() - concept.created_at).num_hours() as f64;
        let recency = if hours <= 24.0 { 1.0 }
            else if hours <= 168.0 { 0.5 + 0.5 * (1.0 - (hours - 24.0) / 144.0) }
            else { 0.5 };

        // Check if we have enough data for learned weights
        let good_count: u64 = self.conn
            .query_row("SELECT COALESCE(CAST(value AS INTEGER), 0) FROM metadata WHERE key = 'quality:good_count'",
                [], |r| r.get(0)).unwrap_or(0);
        let bad_count: u64 = self.conn
            .query_row("SELECT COALESCE(CAST(value AS INTEGER), 0) FROM metadata WHERE key = 'quality:bad_count'",
                [], |r| r.get(0)).unwrap_or(0);

        if good_count + bad_count < 50 {
            // Cold start: use LLM confidence directly
            return llm_conf;
        }

        // Learned weights from data (parameterized queries to prevent SQL injection)
        let get_weight = |good_key: &str, bad_key: &str| -> f64 {
            let good_sum: f64 = self.conn
                .query_row("SELECT COALESCE(CAST(value AS REAL), 0) FROM metadata WHERE key = ?1",
                    rusqlite::params![good_key], |r| r.get(0)).unwrap_or(0.0);
            let bad_sum: f64 = self.conn
                .query_row("SELECT COALESCE(CAST(value AS REAL), 0) FROM metadata WHERE key = ?1",
                    rusqlite::params![bad_key], |r| r.get(0)).unwrap_or(0.0);
            let avg_good = if good_count > 0 { good_sum / good_count as f64 } else { 0.5 };
            let avg_bad = if bad_count > 0 { bad_sum / bad_count as f64 } else { 0.5 };
            avg_good / (avg_good + avg_bad + 0.001)
        };

        let w_llm = get_weight("quality:good_llm_sum", "quality:bad_llm_sum");
        let w_utility = get_weight("quality:good_utility_sum", "quality:bad_utility_sum");
        let w_connectivity = get_weight("quality:good_connectivity_sum", "quality:bad_connectivity_sum");
        let w_recency = get_weight("quality:good_recency_sum", "quality:bad_recency_sum");

        // Normalize weights
        let total_w = w_llm + w_utility + w_connectivity + w_recency + 0.001;
        (w_llm * llm_conf + w_utility * utility + w_connectivity * connectivity + w_recency * recency) / total_w
    }

    /// Update quality weight tracker based on observed good/bad memories.
    /// Call periodically (e.g., at gc or hook_stop).
    pub fn update_quality_weights(&self) {
        // Find "good" memories: recall_count > 0 AND access_count > 0
        let good_mems: Vec<(String, u32)> = self.conn.prepare(
            "SELECT m.id, m.access_count FROM memories m
             INNER JOIN metadata md ON md.key = 'recall_hit:' || m.id
             WHERE CAST(md.value AS INTEGER) > 0 AND m.access_count > 0"
        ).and_then(|mut stmt| {
            stmt.query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, u32>(1)?)))
                .map(|rows| rows.filter_map(|r| r.ok()).collect())
        }).unwrap_or_default();

        // Find "bad" memories: recall_count >= 3 AND access_count == 0
        let bad_mems: Vec<String> = self.conn.prepare(
            "SELECT m.id FROM memories m
             INNER JOIN metadata md ON md.key = 'recall_hit:' || m.id
             WHERE CAST(md.value AS INTEGER) >= 3 AND m.access_count = 0"
        ).and_then(|mut stmt| {
            stmt.query_map([], |row| row.get::<_, String>(0))
                .map(|rows| rows.filter_map(|r| r.ok()).collect())
        }).unwrap_or_default();

        if good_mems.is_empty() && bad_mems.is_empty() { return; }

        let upsert = |key: &str, val: f64| {
            let _ = self.conn.execute(
                "INSERT INTO metadata (key, value) VALUES (?1, ?2)
                 ON CONFLICT(key) DO UPDATE SET value = ?2",
                rusqlite::params![key, val.to_string()],
            );
        };

        let good_count = good_mems.len() as f64;
        let bad_count = bad_mems.len() as f64;
        upsert("quality:good_count", good_count);
        upsert("quality:bad_count", bad_count);

        // Compute averages for ALL 4 features for good memories
        let (mut sum_llm, mut sum_util, mut sum_conn, mut sum_rec) = (0.0, 0.0, 0.0, 0.0);
        for (id, ac) in &good_mems {
            sum_util += *ac as f64;
            if let Ok(mem) = self.get(id) {
                sum_llm += mem.strength;
                let hours = (chrono::Utc::now() - mem.created_at).num_hours() as f64;
                sum_rec += if hours <= 24.0 { 1.0 } else if hours <= 168.0 { 0.5 } else { 0.2 };
                sum_conn += (mem.concept_ids.len() as f64 / 3.0).min(1.0);
            }
        }

        // Same for bad memories
        let (mut bad_llm, mut bad_util, mut bad_conn, mut bad_rec) = (0.0, 0.0, 0.0, 0.0);
        for id in &bad_mems {
            bad_util += 0.0; // access_count == 0 by definition
            if let Ok(mem) = self.get(id) {
                bad_llm += mem.strength;
                let hours = (chrono::Utc::now() - mem.created_at).num_hours() as f64;
                bad_rec += if hours <= 24.0 { 1.0 } else if hours <= 168.0 { 0.5 } else { 0.2 };
                bad_conn += (mem.concept_ids.len() as f64 / 3.0).min(1.0);
            }
        }

        // Store averages for all features
        if good_count > 0.0 {
            upsert("quality:good_llm_sum", sum_llm / good_count);
            upsert("quality:good_utility_sum", sum_util / good_count);
            upsert("quality:good_connectivity_sum", sum_conn / good_count);
            upsert("quality:good_recency_sum", sum_rec / good_count);
        }
        if bad_count > 0.0 {
            upsert("quality:bad_llm_sum", bad_llm / bad_count);
            upsert("quality:bad_utility_sum", bad_util / bad_count);
            upsert("quality:bad_connectivity_sum", bad_conn / bad_count);
            upsert("quality:bad_recency_sum", bad_rec / bad_count);
        }
    }

    /// Record that memories were returned in a recall result.
    /// Increments a recall_count in metadata for tracking quality.
    pub fn record_recall_hit(&self, ids: &[String]) {
        if ids.is_empty() { return; }
        let _ = self.conn.execute_batch("SAVEPOINT recall_hit");
        for id in ids {
            let _ = self.conn.execute(
                "INSERT INTO metadata (key, value) VALUES (?1, '1')
                 ON CONFLICT(key) DO UPDATE SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT)",
                rusqlite::params![format!("recall_hit:{}", id)],
            );
        }
        let _ = self.conn.execute_batch("RELEASE recall_hit");
    }

    /// Compute quality score for the memory store.
    /// Returns (avg_access_per_recall, total_recalls, total_accesses).
    /// High access/recall ratio = memories are useful. Low = noisy.
    pub fn quality_metrics(&self) -> ReinResult<(f64, u64, u64)> {
        let total_accesses: u64 = self.conn
            .query_row("SELECT COALESCE(SUM(access_count), 0) FROM memories", [], |r| r.get(0))?;
        let total_recalls: u64 = self.conn
            .query_row(
                "SELECT COALESCE(SUM(CAST(value AS INTEGER)), 0) FROM metadata WHERE key LIKE 'recall_hit:%'",
                [], |r| r.get(0),
            ).unwrap_or(0);
        let ratio = if total_recalls > 0 {
            total_accesses as f64 / total_recalls as f64
        } else {
            1.0
        };
        Ok((ratio, total_recalls, total_accesses))
    }

    /// Prune low-quality concepts and cascade-clean orphaned memories.
    /// Called during gc. Removes concepts with quality < 0.2 whose source memories
    /// were recalled 5+ times but never accessed.
    pub fn prune_low_quality_concepts(&self) -> ReinResult<u64> {
        let all_memoirs = self.list_memoirs()?;
        let mut pruned = 0u64;

        for memoir in &all_memoirs {
            // Use memoir.id directly — already have full Memoir from list_memoirs()
            let mut stmt = self.conn.prepare("SELECT * FROM concepts WHERE memoir_id = ?1")?;
            let concepts: Vec<Concept> = stmt.query_map(rusqlite::params![memoir.id], |row| {
                super::memoir::row_to_concept(row).map_err(|e| {
                    rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e))
                })
            })?.filter_map(|r| r.ok()).collect();
            drop(stmt);

            for concept in &concepts {
                let score = self.concept_quality_score(concept);

                // Require majority of source memories to be weak (not just any one)
                let weak_count = concept.source_memory_ids.iter().filter(|mid| {
                        let recall: u64 = self.conn
                            .query_row(
                                "SELECT COALESCE(CAST(value AS INTEGER), 0) FROM metadata WHERE key = ?1",
                                rusqlite::params![format!("recall_hit:{}", mid)],
                                |r| r.get(0),
                            ).unwrap_or(0);
                        let access: u32 = self.get(mid).map(|m| m.access_count).unwrap_or(0);
                        recall >= 5 && access == 0
                    }).count();
                let majority_weak = !concept.source_memory_ids.is_empty()
                    && weak_count > concept.source_memory_ids.len() / 2;

                if score < 0.2 && majority_weak {
                    let _ = self.conn.execute(
                        "DELETE FROM concept_links WHERE source_id = ?1 OR target_id = ?1",
                        rusqlite::params![concept.id],
                    );
                    let _ = self.conn.execute(
                        "DELETE FROM concepts WHERE id = ?1",
                        rusqlite::params![concept.id],
                    );
                    pruned += 1;

                    for mid in &concept.source_memory_ids {
                        if let Ok(mut mem) = self.get(mid) {
                            mem.concept_ids.retain(|cid| cid != &concept.id);
                            let json = serde_json::to_string(&mem.concept_ids).unwrap_or_default();
                            let _ = self.conn.execute(
                                "UPDATE memories SET concept_ids = ?1 WHERE id = ?2",
                                rusqlite::params![json, mid],
                            );
                        }
                    }
                }
            }
        }
        Ok(pruned)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Importance, Source};

    fn test_memory(topic: &str, summary: &str, importance: Importance) -> Memory {
        Memory {
            id: ulid::Ulid::new().to_string(),
            layer: importance.auto_layer(),
            topic: topic.to_string(),
            summary: summary.to_string(),
            content: format!("Content for {summary}"),
            keywords: vec![],
            importance,
            source: Source::Manual,
            strength: 1.0,
            decay_lambda: 0.06 * importance.decay_factor(),
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

    #[test]
    fn test_concept_quality_score_cold_start() {
        // Cold start: no learned weights, should return LLM confidence directly
        let store = SqliteStore::in_memory().unwrap();

        let concept = Concept {
            id: "c1".to_string(),
            memoir_id: "test".to_string(),
            name: "test concept".to_string(),
            definition: "A test concept".to_string(),
            labels: vec![],
            source_memory_ids: vec![],
            confidence: 0.75,
            revision: 1,
            last_episode_id: None,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };

        let score = store.concept_quality_score(&concept);
        // Cold start returns llm_conf directly
        assert!((score - 0.75).abs() < 0.01, "Cold start should return LLM confidence, got {score}");
    }

    #[test]
    fn test_concept_quality_score_with_learned_weights() {
        let store = SqliteStore::in_memory().unwrap();

        // Seed enough good/bad data to trigger learned weights
        let upsert = |key: &str, val: &str| {
            store.conn().execute(
                "INSERT INTO metadata (key, value) VALUES (?1, ?2)
                 ON CONFLICT(key) DO UPDATE SET value = ?2",
                rusqlite::params![key, val],
            ).unwrap();
        };

        upsert("quality:good_count", "30");
        upsert("quality:bad_count", "25");
        // Good memories: high LLM, high utility
        upsert("quality:good_llm_sum", "0.8");
        upsert("quality:good_utility_sum", "3.0");
        upsert("quality:good_connectivity_sum", "0.6");
        upsert("quality:good_recency_sum", "0.7");
        // Bad memories: low LLM, zero utility
        upsert("quality:bad_llm_sum", "0.3");
        upsert("quality:bad_utility_sum", "0.0");
        upsert("quality:bad_connectivity_sum", "0.1");
        upsert("quality:bad_recency_sum", "0.5");

        let concept = Concept {
            id: "c1".to_string(),
            memoir_id: "test".to_string(),
            name: "good concept".to_string(),
            definition: "A well-connected concept".to_string(),
            labels: vec![],
            source_memory_ids: vec![],
            confidence: 0.9,
            revision: 1,
            last_episode_id: None,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };

        let score = store.concept_quality_score(&concept);
        // With learned weights and high confidence, score should be reasonable
        assert!(score > 0.4, "Learned weights should produce reasonable score, got {score}");
        assert!(score <= 1.0, "Score should not exceed 1.0, got {score}");
    }

    #[test]
    fn test_record_recall_hit() {
        let store = SqliteStore::in_memory().unwrap();
        let mem = test_memory("test", "recall test", Importance::Medium);
        let id = store.store(mem).unwrap();

        // Record 3 recalls
        store.record_recall_hit(&[id.clone()]);
        store.record_recall_hit(&[id.clone()]);
        store.record_recall_hit(&[id.clone()]);

        let count: u64 = store.conn().query_row(
            "SELECT COALESCE(CAST(value AS INTEGER), 0) FROM metadata WHERE key = ?1",
            rusqlite::params![format!("recall_hit:{}", id)],
            |r| r.get(0),
        ).unwrap();
        assert_eq!(count, 3);
    }

    #[test]
    fn test_quality_metrics() {
        let store = SqliteStore::in_memory().unwrap();

        // Empty store
        let (ratio, recalls, accesses) = store.quality_metrics().unwrap();
        assert_eq!(recalls, 0);
        assert_eq!(accesses, 0);
        assert!((ratio - 1.0).abs() < 0.01); // default ratio is 1.0

        // Store and access a memory
        let mem = test_memory("test", "metrics test", Importance::Medium);
        let id = store.store(mem).unwrap();
        store.record_access(&id).unwrap();
        store.record_access(&id).unwrap();
        store.record_recall_hit(&[id.clone()]);

        let (ratio, recalls, accesses) = store.quality_metrics().unwrap();
        assert_eq!(recalls, 1);
        assert_eq!(accesses, 2);
        assert!((ratio - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_update_quality_weights() {
        let store = SqliteStore::in_memory().unwrap();

        // Create a "good" memory: recalled AND accessed
        let mem = test_memory("test", "useful memory", Importance::High);
        let id = store.store(mem).unwrap();
        store.record_recall_hit(&[id.clone()]);
        store.record_access(&id).unwrap();

        // Create a "bad" memory: recalled 3+ times but never accessed
        let mem2 = test_memory("test", "noisy memory", Importance::Low);
        let id2 = store.store(mem2).unwrap();
        for _ in 0..3 {
            store.record_recall_hit(&[id2.clone()]);
        }

        store.update_quality_weights();

        // Verify weights were persisted
        let good_count: f64 = store.conn().query_row(
            "SELECT COALESCE(CAST(value AS REAL), 0) FROM metadata WHERE key = 'quality:good_count'",
            [], |r| r.get(0),
        ).unwrap_or(0.0);
        let bad_count: f64 = store.conn().query_row(
            "SELECT COALESCE(CAST(value AS REAL), 0) FROM metadata WHERE key = 'quality:bad_count'",
            [], |r| r.get(0),
        ).unwrap_or(0.0);

        assert_eq!(good_count, 1.0);
        assert_eq!(bad_count, 1.0);

        // Verify all 4 feature averages were written
        let good_llm: f64 = store.conn().query_row(
            "SELECT COALESCE(CAST(value AS REAL), -1) FROM metadata WHERE key = 'quality:good_llm_sum'",
            [], |r| r.get(0),
        ).unwrap_or(-1.0);
        assert!(good_llm >= 0.0, "good_llm_sum should be written");

        let good_util: f64 = store.conn().query_row(
            "SELECT COALESCE(CAST(value AS REAL), -1) FROM metadata WHERE key = 'quality:good_utility_sum'",
            [], |r| r.get(0),
        ).unwrap_or(-1.0);
        assert!(good_util >= 0.0, "good_utility_sum should be written");
    }

    #[test]
    fn test_prune_low_quality_concepts() {
        let store = SqliteStore::in_memory().unwrap();

        // Create a memoir and concept
        let memoir = Memoir {
            id: String::new(),
            name: "test-memoir".to_string(),
            description: "test".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };
        store.create_memoir(memoir).unwrap();

        // Create a source memory that is "bad" (recalled 5+ times, never accessed)
        let mem = test_memory("test", "bad source", Importance::Low);
        let mem_id = store.store(mem).unwrap();
        for _ in 0..6 {
            store.record_recall_hit(&[mem_id.clone()]);
        }

        // Add concept with very low confidence tied to that memory
        let concept = Concept {
            id: String::new(),
            memoir_id: "test-memoir".to_string(),
            name: "low-quality-concept".to_string(),
            definition: "This is a low quality concept".to_string(),
            labels: vec![],
            source_memory_ids: vec![mem_id.clone()],
            confidence: 0.1, // Below 0.2 threshold
            revision: 1,
            last_episode_id: None,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };
        store.add_concept(concept).unwrap();

        // Verify concept exists
        let before = store.get_concept("test-memoir", "low-quality-concept").unwrap();
        assert!(before.is_some());

        // Prune
        let pruned = store.prune_low_quality_concepts().unwrap();
        assert_eq!(pruned, 1, "Should prune 1 low-quality concept");

        // Verify concept is gone
        let after = store.get_concept("test-memoir", "low-quality-concept").unwrap();
        assert!(after.is_none(), "Low-quality concept should be deleted");
    }
}
