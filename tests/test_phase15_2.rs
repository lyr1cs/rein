//! Phase 15.2 regression tests — KG search, BFS temporal filtering, reranker,
//! classifier, and extraction postprocess.

use chrono::{Duration, Utc};
use rein::extract::llm::ExtractedMemory;
use rein::extract::postprocess::postprocess;
use rein::search::classify::{classify, QueryType};
use rein::search::kg_search::{bfs_expand_memories, search_concepts_ranked};
use rein::search::recall::recall_fast;
use rein::search::rerank::{default_weights, rerank_score, RerankFeatures};
use rein::store::SqliteStore;
use rein::types::*;

/// Helper to create test memories (matches integration.rs pattern).
fn make_memory(topic: &str, summary: &str, content: &str, importance: Importance) -> Memory {
    Memory {
        id: ulid::Ulid::new().to_string(),
        layer: importance.auto_layer(),
        topic: topic.to_string(),
        summary: summary.to_string(),
        content: content.to_string(),
        keywords: vec![],
        importance,
        source: Source::Manual,
        strength: 1.0,
        decay_lambda: 0.06 * importance.decay_factor(),
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
        created_at: Utc::now(),
        updated_at: Utc::now(),
        last_accessed: Utc::now(),
    }
}

/// Helper to create a memoir + concept linked to a given memory ID.
fn setup_memoir_with_concept(
    store: &SqliteStore,
    memoir_name: &str,
    concept_name: &str,
    concept_definition: &str,
    source_memory_ids: Vec<String>,
) -> String {
    let memoir = Memoir {
        id: String::new(),
        name: memoir_name.to_string(),
        description: "test memoir".to_string(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
    };
    store.create_memoir(memoir).unwrap();

    let concept = Concept {
        id: String::new(),
        memoir_id: memoir_name.to_string(),
        name: concept_name.to_string(),
        definition: concept_definition.to_string(),
        labels: vec!["test".to_string()],
        source_memory_ids,
        confidence: 0.9,
        revision: 1,
        last_episode_id: None,
        created_at: Utc::now(),
        updated_at: Utc::now(),
    };
    store.add_concept(concept).unwrap()
}

// ---------------------------------------------------------------------------
// 1. KG search finds memory via concept
// ---------------------------------------------------------------------------

#[test]
fn test_kg_search_finds_memory_via_concept() {
    let store = SqliteStore::in_memory().unwrap();

    // Store a memory
    let m = make_memory(
        "architecture",
        "Search pipeline design",
        "The search pipeline uses BM25 full-text search as the first stage",
        Importance::High,
    );
    let mem_id = store.store(m).unwrap();

    // Create a memoir + concept linked to that memory
    setup_memoir_with_concept(
        &store,
        "arch-memoir",
        "search pipeline",
        "Multi-stage search pipeline using BM25 and vector search",
        vec![mem_id.clone()],
    );

    // Search concepts ranked should find the memory via the concept
    let results = search_concepts_ranked(&store, "search pipeline", 10);
    assert!(!results.is_empty(), "KG search should return results");
    assert!(
        results.iter().any(|(id, _)| id == &mem_id),
        "KG search should return the linked memory ID"
    );
}

// ---------------------------------------------------------------------------
// 2. BFS expand skips expired links
// ---------------------------------------------------------------------------

#[test]
fn test_bfs_expand_skips_expired_links() {
    let store = SqliteStore::in_memory().unwrap();

    // Store two memories
    let m1 = make_memory(
        "topic-a",
        "Source concept memory",
        "Source data",
        Importance::Medium,
    );
    let m2 = make_memory(
        "topic-b",
        "Target concept memory",
        "Target data",
        Importance::Medium,
    );
    let mem_id1 = store.store(m1).unwrap();
    let mem_id2 = store.store(m2).unwrap();

    // Create memoir with two concepts
    let memoir = Memoir {
        id: String::new(),
        name: "bfs-test-expired".to_string(),
        description: "test".to_string(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
    };
    store.create_memoir(memoir).unwrap();

    let c1 = Concept {
        id: String::new(),
        memoir_id: "bfs-test-expired".to_string(),
        name: "source concept alpha".to_string(),
        definition: "The source concept alpha for BFS test".to_string(),
        labels: vec![],
        source_memory_ids: vec![mem_id1.clone()],
        confidence: 0.9,
        revision: 1,
        last_episode_id: None,
        created_at: Utc::now(),
        updated_at: Utc::now(),
    };
    let c1_id = store.add_concept(c1).unwrap();

    let c2 = Concept {
        id: String::new(),
        memoir_id: "bfs-test-expired".to_string(),
        name: "target concept beta".to_string(),
        definition: "The target concept beta for BFS test".to_string(),
        labels: vec![],
        source_memory_ids: vec![mem_id2.clone()],
        confidence: 0.8,
        revision: 1,
        last_episode_id: None,
        created_at: Utc::now(),
        updated_at: Utc::now(),
    };
    let c2_id = store.add_concept(c2).unwrap();

    // Link c1 -> c2 with valid_until in the past (expired)
    let link = ConceptLink {
        id: String::new(),
        source_id: c1_id.clone(),
        target_id: c2_id.clone(),
        relation: Relation::RelatedTo,
        weight: 1.0,
        created_at: Utc::now(),
        valid_from: Some(Utc::now() - Duration::days(30)),
        valid_until: Some(Utc::now() - Duration::days(1)), // expired yesterday
    };
    store.add_link(link).unwrap();

    // BFS from source concept — should NOT traverse expired link
    let results = bfs_expand_memories(&store, &["source concept alpha".to_string()], 2, 10);

    // Should find mem_id1 (seed concept) but NOT mem_id2 (expired link target)
    assert!(
        results.iter().any(|(id, _)| id == &mem_id1),
        "BFS should include seed concept's memory"
    );
    assert!(
        !results.iter().any(|(id, _)| id == &mem_id2),
        "BFS should NOT traverse expired link to target memory"
    );
}

// ---------------------------------------------------------------------------
// 3. BFS expand follows valid links
// ---------------------------------------------------------------------------

#[test]
fn test_bfs_expand_follows_valid_links() {
    let store = SqliteStore::in_memory().unwrap();

    // Store two memories
    let m1 = make_memory(
        "topic-a",
        "Source memory",
        "Source content",
        Importance::Medium,
    );
    let m2 = make_memory(
        "topic-b",
        "Target memory",
        "Target content",
        Importance::Medium,
    );
    let mem_id1 = store.store(m1).unwrap();
    let mem_id2 = store.store(m2).unwrap();

    // Create memoir with two concepts
    let memoir = Memoir {
        id: String::new(),
        name: "bfs-test-valid".to_string(),
        description: "test".to_string(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
    };
    store.create_memoir(memoir).unwrap();

    let c1 = Concept {
        id: String::new(),
        memoir_id: "bfs-test-valid".to_string(),
        name: "source concept gamma".to_string(),
        definition: "Source concept gamma for BFS valid link test".to_string(),
        labels: vec![],
        source_memory_ids: vec![mem_id1.clone()],
        confidence: 0.9,
        revision: 1,
        last_episode_id: None,
        created_at: Utc::now(),
        updated_at: Utc::now(),
    };
    let c1_id = store.add_concept(c1).unwrap();

    let c2 = Concept {
        id: String::new(),
        memoir_id: "bfs-test-valid".to_string(),
        name: "target concept delta".to_string(),
        definition: "Target concept delta for BFS valid link test".to_string(),
        labels: vec![],
        source_memory_ids: vec![mem_id2.clone()],
        confidence: 0.8,
        revision: 1,
        last_episode_id: None,
        created_at: Utc::now(),
        updated_at: Utc::now(),
    };
    let c2_id = store.add_concept(c2).unwrap();

    // Link c1 -> c2 with valid_until in the future (active)
    let link = ConceptLink {
        id: String::new(),
        source_id: c1_id.clone(),
        target_id: c2_id.clone(),
        relation: Relation::RelatedTo,
        weight: 1.0,
        created_at: Utc::now(),
        valid_from: Some(Utc::now() - Duration::days(10)),
        valid_until: Some(Utc::now() + Duration::days(30)), // still valid
    };
    store.add_link(link).unwrap();

    // BFS from source concept — should traverse valid link
    let results = bfs_expand_memories(&store, &["source concept gamma".to_string()], 2, 10);

    // Should find both memories
    assert!(
        results.iter().any(|(id, _)| id == &mem_id1),
        "BFS should include seed concept's memory"
    );
    assert!(
        results.iter().any(|(id, _)| id == &mem_id2),
        "BFS should traverse valid link and include target memory"
    );
}

// ---------------------------------------------------------------------------
// 4. Episode search ranks matching sessions
// ---------------------------------------------------------------------------

#[test]
fn test_episode_search_ranks_matching_sessions() {
    let store = SqliteStore::in_memory().unwrap();

    let mem = make_memory(
        "workflow",
        "Migration plan",
        "Generic plan note",
        Importance::Medium,
    );
    let mem_id = store.store(mem).unwrap();

    let episode = Episode {
        id: String::new(),
        title: "Billing migration planning session".to_string(),
        outcome: "Selected PostgreSQL rollout plan".to_string(),
        decisions: vec!["Use PostgreSQL for billing migration".to_string()],
        primary_topics: vec!["workflow".to_string(), "billing".to_string()],
        tags: vec!["main-agent".to_string()],
        involved_agents: vec!["test-agent".to_string()],
        important_paths: vec!["src/billing/mod.rs".to_string()],
        temporal_keywords: vec!["date:2026-04-04".to_string()],
        source_session_id: Some("session-billing-1".to_string()),
        concept_ids: vec![],
        memory_ids: vec![mem_id],
        created_at: Utc::now(),
    };
    store.create_episode(episode).unwrap();

    let ranked = store
        .search_episodes_ranked("what did we decide about billing migration", 10, None, None)
        .unwrap();
    assert!(
        !ranked.is_empty(),
        "episode search should find matching sessions"
    );
    assert!(
        ranked[0].0.title.contains("Billing migration"),
        "top-ranked episode should match billing migration session"
    );
}

// ---------------------------------------------------------------------------
// 5. recall_fast uses episode signal for episodic queries
// ---------------------------------------------------------------------------

#[test]
fn test_recall_fast_uses_episode_signal_for_episodic_queries() {
    let store = SqliteStore::in_memory().unwrap();
    let config = rein::config::ReinConfig::default();

    let mem = make_memory(
        "workflow",
        "Chosen plan",
        "Final implementation note recorded after the meeting.",
        Importance::High,
    );
    let mem_id = store.store(mem).unwrap();

    let episode = Episode {
        id: String::new(),
        title: "Billing migration meeting".to_string(),
        outcome: "Agreed on the new billing migration approach".to_string(),
        decisions: vec!["Use PostgreSQL for billing migration".to_string()],
        primary_topics: vec!["workflow".to_string(), "billing".to_string()],
        tags: vec!["main-agent".to_string()],
        involved_agents: vec!["test-agent".to_string()],
        important_paths: vec!["src/billing/mod.rs".to_string()],
        temporal_keywords: vec!["date:2026-04-04".to_string()],
        source_session_id: Some("session-billing-2".to_string()),
        concept_ids: vec![],
        memory_ids: vec![mem_id.clone()],
        created_at: Utc::now(),
    };
    store.create_episode(episode).unwrap();

    let results = recall_fast(
        &store,
        &config,
        "what did we decide about billing migration",
        None,
        None,
        10,
    )
    .unwrap();

    assert!(
        results.iter().any(|r| r.memory.id == mem_id),
        "episodic query should recall memory linked through matching episode"
    );
}

// ---------------------------------------------------------------------------
// 4. Rerank changes ordering
// ---------------------------------------------------------------------------

#[test]
fn test_rerank_changes_ordering() {
    let w = default_weights();

    // Candidate A: high FTS score but old and low access
    let a = RerankFeatures {
        fts_score: 0.95,
        vec_score: 0.1,
        kg_score: 0.0,
        episode_score: 0.0,
        recency_days: 60.0,
        access_count: 0,
        strength: 0.4,
        importance_weight: 0.4,
        keyword_overlap: 0.2,
        topic_match: 0.0,
        brevity: 0.5,
        channel_coverage: 0.33,
        canonical_support: 0.0,
        source_diversity: 0.0,
        usage_recency: 30.0,
        connectivity: 0.0,
        concept_richness: 0.0,
        tier_score: 0.0,
        is_current: 0.0,
    };

    // Candidate B: low FTS but very recent, high access, high strength
    let b = RerankFeatures {
        fts_score: 0.3,
        vec_score: 0.7,
        kg_score: 0.5,
        episode_score: 0.7,
        recency_days: 0.5,
        access_count: 15,
        strength: 1.0,
        importance_weight: 1.0,
        keyword_overlap: 0.8,
        topic_match: 1.0,
        brevity: 0.8,
        channel_coverage: 1.0,
        canonical_support: 0.7,
        source_diversity: 0.6,
        usage_recency: 1.0,
        connectivity: 0.5,
        concept_richness: 0.8,
        tier_score: 1.0,
        is_current: 1.0,
    };

    let score_a = rerank_score(&a, &w);
    let score_b = rerank_score(&b, &w);

    // By FTS-only ordering: A > B (0.95 vs 0.3)
    assert!(a.fts_score > b.fts_score, "A has higher raw FTS score");

    // After multi-feature reranking: B > A (recency+access+strength+vec+kg dominate)
    assert!(
        score_b > score_a,
        "Reranker should change ordering: B ({score_b}) should beat A ({score_a})"
    );
}

// ---------------------------------------------------------------------------
// 5. Classify: episodic before temporal
// ---------------------------------------------------------------------------

#[test]
fn test_classify_episodic_before_temporal() {
    // "what happened in our last meeting" should be Episodic, not Temporal
    let strategy = classify("what happened in our last meeting", false, false);
    assert_eq!(
        strategy.query_type,
        QueryType::Episodic,
        "Meeting reference should classify as Episodic"
    );

    // "what happened last week" should still be Temporal
    let strategy = classify("what happened last week", false, false);
    assert_eq!(
        strategy.query_type,
        QueryType::Temporal,
        "Time reference without session/meeting should classify as Temporal"
    );
}

// ---------------------------------------------------------------------------
// 6. Postprocess adds date keyword
// ---------------------------------------------------------------------------

#[test]
fn test_postprocess_adds_date_keyword() {
    let mut mem = ExtractedMemory {
        topic: "deployment".to_string(),
        summary: "Deployed v2.0".to_string(),
        content: "Deployed version 2.0 to production on 2026-03-15".to_string(),
        keywords: vec![],
        importance: "medium".to_string(),
        should_store: true,
        quality_confidence: 0.8,
    };

    postprocess(&mut mem);

    assert!(
        mem.keywords.contains(&"date:2026-03-15".to_string()),
        "Postprocess should extract ISO date as keyword, got: {:?}",
        mem.keywords
    );
}

// ---------------------------------------------------------------------------
// 7. Postprocess tags preference
// ---------------------------------------------------------------------------

#[test]
fn test_postprocess_tags_preference() {
    let mut mem = ExtractedMemory {
        topic: "settings".to_string(),
        summary: "User prefers dark mode".to_string(),
        content: "I prefer dark mode for all my editors and terminals".to_string(),
        keywords: vec![],
        importance: "medium".to_string(),
        should_store: true,
        quality_confidence: 0.7,
    };

    postprocess(&mut mem);

    assert_eq!(
        mem.topic, "user_preference",
        "Postprocess should change topic to user_preference"
    );
    assert!(
        mem.keywords.contains(&"preference".to_string()),
        "Postprocess should add 'preference' keyword, got: {:?}",
        mem.keywords
    );
}
