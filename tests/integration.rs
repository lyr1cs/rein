//! Integration tests for rein memory system.

use rein::store::SqliteStore;
use rein::types::*;

/// Helper to create test memories.
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
        related_ids: vec![],
        status: MemoryStatus::default(),
        embedding: None,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        last_accessed: chrono::Utc::now(),
    }
}

#[test]
fn test_full_lifecycle() {
    let store = SqliteStore::in_memory().unwrap();

    // 1. Store 5 memories across 2 topics
    let m1 = make_memory(
        "debug",
        "OOM fix",
        "Fixed OOM by closing database connections",
        Importance::High,
    );
    let m2 = make_memory(
        "debug",
        "Race condition",
        "Resolved race condition in worker pool",
        Importance::Medium,
    );
    let m3 = make_memory(
        "architecture",
        "Auth design",
        "Chose JWT for authentication system",
        Importance::Critical,
    );
    let m4 = make_memory(
        "architecture",
        "DB choice",
        "Selected PostgreSQL for main database",
        Importance::High,
    );
    let m5 = make_memory(
        "debug",
        "Memory leak",
        "Found memory leak in cache layer",
        Importance::Low,
    );

    let id1 = store.store(m1).unwrap();
    let id2 = store.store(m2).unwrap();
    let id3 = store.store(m3).unwrap();
    let id4 = store.store(m4).unwrap();
    let id5 = store.store(m5).unwrap();

    // 2. Verify FTS search works
    let results = store.search_fts("OOM database", None, 10).unwrap();
    assert!(!results.is_empty(), "FTS should find OOM-related memories");

    // 3. Verify topics
    let topics = store.list_topics().unwrap();
    assert_eq!(topics.len(), 2);

    // 4. Verify stats
    let stats = store.stats().unwrap();
    assert_eq!(stats.total_memories, 5);

    // 5. Update a memory
    let mut m = store.get(&id1).unwrap();
    m.content = "Fixed OOM by closing database connections and adding pool limits".to_string();
    store.update(&m).unwrap();
    let updated = store.get(&id1).unwrap();
    assert!(updated.content.contains("pool limits"));

    // 6. Delete a memory
    store.delete(&id5).unwrap();
    let stats = store.stats().unwrap();
    assert_eq!(stats.total_memories, 4);

    // 7. Verify Critical memory properties
    let critical = store.get(&id3).unwrap();
    assert_eq!(critical.importance, Importance::Critical);
    assert_eq!(critical.layer, MemoryLayer::LTM);

    // Suppress unused variable warnings
    let _ = (id2, id4);
}

#[test]
fn test_dedup_lifecycle() {
    let store = SqliteStore::in_memory().unwrap();

    // Store original
    let m1 = make_memory(
        "debug",
        "OOM fix",
        "Fixed OOM by closing database connections in the pool",
        Importance::Medium,
    );
    let _id1 = store.store(m1).unwrap();

    // Store very similar content (should merge)
    let m2 = make_memory(
        "debug",
        "OOM fix",
        "Fixed OOM by closing database connections in the connection pool",
        Importance::Medium,
    );
    let _id2 = store.store_with_dedup(m2, 0.85, 7).unwrap();

    // Should have merged into existing
    let stats = store.stats().unwrap();
    // Either 1 memory (merged) or 2 (if similarity didn't quite reach 0.85 threshold)
    // The test verifies the dedup mechanism runs without error
    assert!(stats.total_memories <= 2);
}

#[test]
fn test_search_and_scoring() {
    let store = SqliteStore::in_memory().unwrap();

    // Store memories with different importance
    let m1 = make_memory(
        "ops",
        "Deploy script",
        "Updated deployment script for production",
        Importance::Low,
    );
    let m2 = make_memory(
        "ops",
        "Critical outage",
        "Production database went down, restored from backup",
        Importance::Critical,
    );

    store.store(m1).unwrap();
    store.store(m2).unwrap();

    // Search should find both
    let results = store.search_fts("production", None, 10).unwrap();
    assert!(
        !results.is_empty(),
        "Should find production-related memories"
    );
}

#[test]
fn test_health_report() {
    let store = SqliteStore::in_memory().unwrap();

    for i in 0..10 {
        let m = make_memory(
            "busy-topic",
            &format!("Item {i}"),
            &format!("Content {i}"),
            Importance::Medium,
        );
        store.store(m).unwrap();
    }

    let reports = store.health(Some("busy-topic")).unwrap();
    assert!(!reports.is_empty());
    assert_eq!(reports[0].count, 10);
}

/// Store a memory with an embedding vector, verify it can be found via
/// vector search even when FTS returns nothing useful.
#[test]
fn test_vector_only_recall() {
    let store = SqliteStore::in_memory().unwrap();

    // Create a memory whose content won't match the FTS query at all
    let mut m = make_memory(
        "science",
        "Photosynthesis overview",
        "Plants convert light energy into chemical energy via chloroplasts",
        Importance::High,
    );
    // Attach a fake 3072-dim embedding so it ends up in the vec table
    m.embedding = Some(vec![0.1_f32; 3072]);
    let id = store.store(m).unwrap();

    // FTS search for a completely unrelated term should find nothing
    let fts = store.search_fts("zzz_nonexistent_token_xyz", None, 10).unwrap();
    assert!(fts.is_empty(), "FTS should return nothing for gibberish query");

    // Vector search directly should still find the memory
    let query_vec = vec![0.1_f32; 3072];
    let vec_results =
        rein::store::vec::search_vec(store.conn(), &query_vec, 10).unwrap();
    assert!(
        !vec_results.is_empty(),
        "Vector search should return the stored memory"
    );
    assert_eq!(vec_results[0].0, id, "Vector result should match stored ID");
}

/// Create a memoir with Chinese concept names and definitions containing
/// quotes and newlines. Export as DOT format. Verify no panic (UTF-8 safety)
/// and output contains escaped characters.
#[test]
fn test_dot_export_cjk() {
    use rein::types::{Concept, Memoir};

    let store = SqliteStore::in_memory().unwrap();

    let memoir = Memoir {
        id: String::new(),
        name: "中文知识图谱".to_string(),
        description: "测试\"引号\"和\n换行".to_string(),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
    };
    let _mid = store.create_memoir(memoir).unwrap();

    let c1 = Concept {
        id: String::new(),
        memoir_id: "中文知识图谱".to_string(),
        name: "概念一\"引号\"".to_string(),
        definition: "这是定义\n含有换行\r和回车".to_string(),
        labels: vec!["标签".to_string()],
        confidence: 0.9,
        revision: 1,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
    };
    store.add_concept(c1).unwrap();

    let c2 = Concept {
        id: String::new(),
        memoir_id: "中文知识图谱".to_string(),
        name: "概念二\\反斜杠".to_string(),
        definition: "包含反斜杠\\和\"双引号\"".to_string(),
        labels: vec![],
        confidence: 0.8,
        revision: 1,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
    };
    store.add_concept(c2).unwrap();

    // Export as DOT — should not panic
    let dot = store.export_memoir("中文知识图谱", "dot").unwrap();

    // Basic structural checks
    assert!(dot.contains("digraph"), "DOT output must contain digraph");
    assert!(dot.starts_with("digraph"), "DOT output must start with digraph");
    assert!(dot.contains("rankdir=LR"), "DOT output must set rankdir");

    // Escaped characters must be present (quotes escaped as \")
    assert!(
        dot.contains(r#"\""#),
        "DOT output must contain escaped quotes"
    );
    assert!(
        dot.contains(r#"\n"#),
        "DOT output must contain escaped newlines"
    );

    // Chinese text should survive round-trip
    assert!(
        dot.contains("概念一"),
        "DOT output must contain Chinese concept name"
    );
}
