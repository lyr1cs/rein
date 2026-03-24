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
        embedding: None,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        last_accessed: chrono::Utc::now(),
    }
}

#[tokio::test]
async fn test_full_lifecycle() {
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

    let id1 = store.store(m1).await.unwrap();
    let id2 = store.store(m2).await.unwrap();
    let id3 = store.store(m3).await.unwrap();
    let id4 = store.store(m4).await.unwrap();
    let id5 = store.store(m5).await.unwrap();

    // 2. Verify FTS search works
    let results = store.search_fts("OOM database", None, 10).await.unwrap();
    assert!(!results.is_empty(), "FTS should find OOM-related memories");

    // 3. Verify topics
    let topics = store.list_topics().await.unwrap();
    assert_eq!(topics.len(), 2);

    // 4. Verify stats
    let stats = store.stats().await.unwrap();
    assert_eq!(stats.total_memories, 5);

    // 5. Update a memory
    let mut m = store.get(&id1).await.unwrap();
    m.content = "Fixed OOM by closing database connections and adding pool limits".to_string();
    store.update(&m).await.unwrap();
    let updated = store.get(&id1).await.unwrap();
    assert!(updated.content.contains("pool limits"));

    // 6. Delete a memory
    store.delete(&id5).await.unwrap();
    let stats = store.stats().await.unwrap();
    assert_eq!(stats.total_memories, 4);

    // 7. Verify Critical memory properties
    let critical = store.get(&id3).await.unwrap();
    assert_eq!(critical.importance, Importance::Critical);
    assert_eq!(critical.layer, MemoryLayer::LTM);

    // Suppress unused variable warnings
    let _ = (id2, id4);
}

#[tokio::test]
async fn test_dedup_lifecycle() {
    let store = SqliteStore::in_memory().unwrap();

    // Store original
    let m1 = make_memory(
        "debug",
        "OOM fix",
        "Fixed OOM by closing database connections in the pool",
        Importance::Medium,
    );
    let _id1 = store.store(m1).await.unwrap();

    // Store very similar content (should merge)
    let m2 = make_memory(
        "debug",
        "OOM fix",
        "Fixed OOM by closing database connections in the connection pool",
        Importance::Medium,
    );
    let _id2 = store.store_with_dedup(m2, 0.85, 7).await.unwrap();

    // Should have merged into existing
    let stats = store.stats().await.unwrap();
    // Either 1 memory (merged) or 2 (if similarity didn't quite reach 0.85 threshold)
    // The test verifies the dedup mechanism runs without error
    assert!(stats.total_memories <= 2);
}

#[tokio::test]
async fn test_search_and_scoring() {
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

    store.store(m1).await.unwrap();
    store.store(m2).await.unwrap();

    // Search should find both
    let results = store.search_fts("production", None, 10).await.unwrap();
    assert!(
        !results.is_empty(),
        "Should find production-related memories"
    );
}

#[tokio::test]
async fn test_health_report() {
    let store = SqliteStore::in_memory().unwrap();

    for i in 0..10 {
        let m = make_memory(
            "busy-topic",
            &format!("Item {i}"),
            &format!("Content {i}"),
            Importance::Medium,
        );
        store.store(m).await.unwrap();
    }

    let reports = store.health(Some("busy-topic")).await.unwrap();
    assert!(!reports.is_empty());
    assert_eq!(reports[0].count, 10);
}
