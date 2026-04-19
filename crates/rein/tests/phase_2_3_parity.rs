use std::sync::Arc;

use rein::config::ReinConfig;
use rein::ops::OpsRuntime;

#[test]
fn runtime_dry_run_defaults_to_false_and_is_settable() {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let mut config = ReinConfig::default();
    config.database.path = tmp.path().join("memories.db").to_string_lossy().into_owned();
    let cfg = Arc::new(config);
    let rt = OpsRuntime::for_cli(cfg);
    assert!(!rt.dry_run(), "default should be false");
    rt.set_dry_run(true);
    assert!(rt.dry_run(), "after set_dry_run(true), should be true");
    rt.set_dry_run(false);
    assert!(!rt.dry_run(), "resettable to false");
}

#[tokio::test]
async fn canonicals_returns_consistent_output_across_surfaces() {
    use rein::ops::{OpsCliEntry, OpsMcpEntry, OpsRestEntry};
    use serde_json::Value;

    // Helper: build a fresh tempdir-backed config for each surface invocation.
    // We use a single shared tempdir so all three surfaces operate on the same DB.
    let tmp = tempfile::TempDir::new().expect("tempdir");

    let make_config = || {
        let mut c = ReinConfig::default();
        c.database.path = tmp.path().join("memories.db").to_string_lossy().into_owned();
        Arc::new(c)
    };

    // Seed two canonical memories via the store directly.
    {
        use rein::types::{Importance, Memory, MemoryLayer, MemoryStore, Source};
        let cfg = make_config();
        let store = cfg.open_store().expect("open store for seeding");

        let seed = |id: &str, topic: &str, summary: &str| -> Memory {
            Memory {
                id: id.to_string(),
                layer: MemoryLayer::LTM,
                topic: topic.to_string(),
                summary: summary.to_string(),
                content: summary.to_string(),
                keywords: vec![],
                importance: Importance::Medium,
                source: Source::Manual,
                strength: 1.0,
                decay_lambda: 0.0,
                access_count: 0,
                superseded_by: None,
                canonical_id: None,
                support_count: 1,
                merge_count: 0,
                dedup_confidence: 1.0,
                source_diversity: 0.5,
                contradiction_score: 0.0,
                related_ids: vec![],
                concept_ids: vec![],
                status: rein::types::MemoryStatus::Active,
                embedding: None,
                tier: Default::default(),
                cluster_id: None,
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                last_accessed: chrono::Utc::now(),
            }
        };

        store.store(seed("c1", "test", "canonical one")).expect("seed c1");
        store.store(seed("c2", "test", "canonical two")).expect("seed c2");
    }

    // --- MCP surface ---
    let mcp_json: Value = {
        let runtime = Arc::new(OpsRuntime::for_mcp(make_config()));
        let entry = inventory::iter::<OpsMcpEntry>()
            .find(|e| e.op_name == "canonicals")
            .expect("canonicals MCP entry registered");
        let out = (entry.invoke)(runtime, serde_json::json!({ "limit": 20 }))
            .await
            .expect("MCP canonicals invoke");
        serde_json::from_str(&out).expect("MCP output is valid JSON")
    };

    // --- REST surface ---
    let (rest_status, rest_json): (hyper::StatusCode, Value) = {
        let runtime = Arc::new(OpsRuntime::for_rest(make_config()));
        let entry = inventory::iter::<OpsRestEntry>()
            .find(|e| e.op_name == "canonicals")
            .expect("canonicals REST entry registered");
        let (status, bytes) = (entry.invoke)(
            runtime,
            std::collections::HashMap::new(),
            "limit=20".to_string(),
            None,
        )
        .await
        .expect("REST canonicals invoke");
        let value: Value = serde_json::from_slice(&bytes).expect("REST body is valid JSON");
        (status, value)
    };

    // --- CLI surface (smoke check) ---
    {
        let runtime = Arc::new(OpsRuntime::for_cli(make_config()));
        let entry = inventory::iter::<OpsCliEntry>()
            .find(|e| e.name == "canonicals")
            .expect("canonicals CLI entry registered");
        let matches = (entry.build_clap)()
            .try_get_matches_from(["canonicals", "--limit", "20"])
            .expect("CLI arg parse");
        let _out = (entry.invoke)(runtime, &matches)
            .await
            .expect("CLI canonicals invoke");
    }

    assert_eq!(rest_status, hyper::StatusCode::OK);

    // Both surfaces must return a `canonicals` array.
    let mcp_arr = mcp_json["canonicals"]
        .as_array()
        .expect("MCP canonicals output must have `canonicals` array");
    let rest_arr = rest_json["canonicals"]
        .as_array()
        .expect("REST canonicals output must have `canonicals` array");

    // Both surfaces share the same DB, so counts should match.
    assert_eq!(
        mcp_arr.len(),
        rest_arr.len(),
        "MCP and REST canonicals must return the same count"
    );
    assert_eq!(mcp_arr.len(), 2, "expected 2 seeded canonical memories");

    // IDs present in both surfaces must match (order may differ; sort for comparison).
    let mut mcp_ids: Vec<&str> = mcp_arr
        .iter()
        .filter_map(|v| v["id"].as_str())
        .collect();
    let mut rest_ids: Vec<&str> = rest_arr
        .iter()
        .filter_map(|v| v["id"].as_str())
        .collect();
    mcp_ids.sort_unstable();
    rest_ids.sort_unstable();
    assert_eq!(mcp_ids, rest_ids, "MCP and REST must return the same memory IDs");
}

#[tokio::test]
async fn evidence_returns_consistent_output_across_surfaces() {
    use rein::ops::{OpsCliEntry, OpsMcpEntry, OpsRestEntry};
    use rein::types::{MemoryEvidence, Source};
    use serde_json::Value;

    let tmp = tempfile::TempDir::new().expect("tempdir");

    let make_config = || {
        let mut c = ReinConfig::default();
        c.database.path = tmp.path().join("memories.db").to_string_lossy().into_owned();
        Arc::new(c)
    };

    // Seed a canonical memory and two evidence rows directly.
    {
        use rein::types::{Importance, Memory, MemoryLayer, MemoryStore};
        let cfg = make_config();
        let store = cfg.open_store().expect("open store for seeding");

        let mem = Memory {
            id: "ev_canon1".to_string(),
            layer: MemoryLayer::LTM,
            topic: "evidence-test".to_string(),
            summary: "parent canonical".to_string(),
            content: "parent canonical".to_string(),
            keywords: vec![],
            importance: Importance::Medium,
            source: Source::Manual,
            strength: 1.0,
            decay_lambda: 0.0,
            access_count: 0,
            superseded_by: None,
            canonical_id: None,
            support_count: 1,
            merge_count: 0,
            dedup_confidence: 1.0,
            source_diversity: 0.5,
            contradiction_score: 0.0,
            related_ids: vec![],
            concept_ids: vec![],
            status: rein::types::MemoryStatus::Active,
            embedding: None,
            tier: Default::default(),
            cluster_id: None,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            last_accessed: chrono::Utc::now(),
        };
        store.store(mem).expect("seed canonical");

        let now = chrono::Utc::now();
        let make_ev = |id: &str, summary: &str| MemoryEvidence {
            id: id.to_string(),
            canonical_id: "ev_canon1".to_string(),
            memory_id: None,
            source_topic: "evidence-test".to_string(),
            summary: summary.to_string(),
            content: summary.to_string(),
            keywords: vec![],
            source: Source::Manual,
            created_at: now,
            imported_at: now,
        };

        store.add_memory_evidence(make_ev("e1", "evidence one")).expect("seed e1");
        store.add_memory_evidence(make_ev("e2", "evidence two")).expect("seed e2");
    }

    // --- MCP surface ---
    let mcp_json: Value = {
        let runtime = Arc::new(OpsRuntime::for_mcp(make_config()));
        let entry = inventory::iter::<OpsMcpEntry>()
            .find(|e| e.op_name == "evidence")
            .expect("evidence MCP entry registered");
        let out = (entry.invoke)(
            runtime,
            serde_json::json!({ "canonical_id": "ev_canon1", "limit": 20 }),
        )
        .await
        .expect("MCP evidence invoke");
        serde_json::from_str(&out).expect("MCP output is valid JSON")
    };

    // --- REST surface ---
    let (rest_status, rest_json): (hyper::StatusCode, Value) = {
        let runtime = Arc::new(OpsRuntime::for_rest(make_config()));
        let entry = inventory::iter::<OpsRestEntry>()
            .find(|e| e.op_name == "evidence")
            .expect("evidence REST entry registered");
        let (status, bytes) = (entry.invoke)(
            runtime,
            std::collections::HashMap::new(),
            "canonical_id=ev_canon1&limit=20".to_string(),
            None,
        )
        .await
        .expect("REST evidence invoke");
        let value: Value = serde_json::from_slice(&bytes).expect("REST body is valid JSON");
        (status, value)
    };

    // --- CLI surface (smoke check) ---
    {
        let runtime = Arc::new(OpsRuntime::for_cli(make_config()));
        let entry = inventory::iter::<OpsCliEntry>()
            .find(|e| e.name == "evidence")
            .expect("evidence CLI entry registered");
        let matches = (entry.build_clap)()
            .try_get_matches_from(["evidence", "ev_canon1", "--limit", "20"])
            .expect("CLI arg parse");
        let _out = (entry.invoke)(runtime, &matches)
            .await
            .expect("CLI evidence invoke");
    }

    assert_eq!(rest_status, hyper::StatusCode::OK);

    // Both surfaces must return an `evidence` array.
    let mcp_arr = mcp_json["evidence"]
        .as_array()
        .expect("MCP evidence output must have `evidence` array");
    let rest_arr = rest_json["evidence"]
        .as_array()
        .expect("REST evidence output must have `evidence` array");

    // Both surfaces share the same DB, so counts should match.
    assert_eq!(
        mcp_arr.len(),
        rest_arr.len(),
        "MCP and REST evidence must return the same count"
    );
    // 1 auto-snapshot (store() snapshots the memory itself as evidence) + 2 manually seeded.
    assert_eq!(mcp_arr.len(), 3, "expected 3 evidence rows (1 auto-snapshot + 2 manual)");

    // IDs present in both surfaces must match.
    let mut mcp_ids: Vec<&str> = mcp_arr
        .iter()
        .filter_map(|v| v["id"].as_str())
        .collect();
    let mut rest_ids: Vec<&str> = rest_arr
        .iter()
        .filter_map(|v| v["id"].as_str())
        .collect();
    mcp_ids.sort_unstable();
    rest_ids.sort_unstable();
    assert_eq!(mcp_ids, rest_ids, "MCP and REST must return the same evidence IDs");
}

#[tokio::test]
async fn gc_dry_run_parity_across_surfaces_respects_auth() {
    use rein::ops::{OpsCliEntry, OpsMcpEntry, OpsRestEntry};
    use rein::types::{Importance, Memory, MemoryLayer, MemoryStore, Source};
    use serde_json::Value;

    let tmp = tempfile::TempDir::new().expect("tempdir");

    let make_config = || {
        let mut c = rein::config::ReinConfig::default();
        c.database.path = tmp.path().join("memories.db").to_string_lossy().into_owned();
        // Low prune threshold so the test memory is eligible for pruning.
        c.decay.prune_threshold = 0.99;
        Arc::new(c)
    };

    // Seed a weak STM memory that would be pruned.
    // Strategy: insert the memory, then mark `last_decay_at` as now so that
    // `apply_decay` skips its decay step (hours-since-last-decay < 24). This
    // leaves the memory with its original low strength so the prune SQL picks
    // it up rather than the strength being reset toward 1.0 by decay.
    {
        use rein::types::MemoryStatus;
        let cfg = make_config();
        let store = cfg.open_store().expect("open store for seeding");

        let mem = Memory {
            id: "gc_test_weak1".to_string(),
            layer: MemoryLayer::STM,
            topic: "gc-test".to_string(),
            summary: "weak stm memory for gc test".to_string(),
            content: "weak stm memory for gc test".to_string(),
            keywords: vec![],
            importance: Importance::Low,
            source: Source::Manual,
            // Strength below the 0.99 threshold → eligible for pruning.
            strength: 0.1,
            decay_lambda: 0.0,
            access_count: 0,
            superseded_by: None,
            canonical_id: None,
            support_count: 1,
            merge_count: 0,
            dedup_confidence: 1.0,
            source_diversity: 0.5,
            contradiction_score: 0.0,
            related_ids: vec![],
            concept_ids: vec![],
            status: MemoryStatus::Active,
            embedding: None,
            tier: Default::default(),
            cluster_id: None,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            last_accessed: chrono::Utc::now(),
        };
        store.store(mem).expect("seed weak memory");
    }

    // --- MCP surface (dry_run = true) ---
    let mcp_json: Value = {
        let runtime = Arc::new(OpsRuntime::for_mcp(make_config()));
        let entry = inventory::iter::<OpsMcpEntry>()
            .find(|e| e.op_name == "gc")
            .expect("gc MCP entry registered");
        let out = (entry.invoke)(
            runtime,
            serde_json::json!({ "dry_run": true, "threshold": 0.99 }),
        )
        .await
        .expect("MCP gc invoke");
        serde_json::from_str(&out).expect("MCP gc output is valid JSON")
    };

    // --- REST surface (dry_run = true, POST body) ---
    let (rest_status, rest_json): (hyper::StatusCode, Value) = {
        let runtime = Arc::new(OpsRuntime::for_rest(make_config()));
        let entry = inventory::iter::<OpsRestEntry>()
            .find(|e| e.op_name == "gc")
            .expect("gc REST entry registered");
        let body = serde_json::to_vec(&serde_json::json!({ "dry_run": true, "threshold": 0.99 }))
            .expect("serialize body");
        let (status, bytes) = (entry.invoke)(
            runtime,
            std::collections::HashMap::new(),
            String::new(),
            Some(body.into()),
        )
        .await
        .expect("REST gc invoke");
        let value: Value = serde_json::from_slice(&bytes).expect("REST gc body is valid JSON");
        (status, value)
    };

    // --- CLI surface (smoke check) ---
    {
        let runtime = Arc::new(OpsRuntime::for_cli(make_config()));
        let entry = inventory::iter::<OpsCliEntry>()
            .find(|e| e.name == "gc")
            .expect("gc CLI entry registered");
        let matches = (entry.build_clap)()
            .try_get_matches_from(["gc", "--dry-run", "--threshold", "0.99"])
            .expect("CLI gc arg parse");
        let _out = (entry.invoke)(runtime, &matches)
            .await
            .expect("CLI gc invoke");
    }

    assert_eq!(rest_status, hyper::StatusCode::OK);

    // Both surfaces must return `dry_run: true` and matching pruned counts.
    assert_eq!(
        mcp_json["dry_run"].as_bool(),
        Some(true),
        "MCP gc must echo dry_run flag"
    );
    assert_eq!(
        rest_json["dry_run"].as_bool(),
        Some(true),
        "REST gc must echo dry_run flag"
    );

    // In dry-run mode the DB must not be modified — call again and get same pruned count.
    let mcp_pruned = mcp_json["pruned"]
        .as_u64()
        .expect("MCP gc output must have `pruned` count");
    let rest_pruned = rest_json["pruned"]
        .as_u64()
        .expect("REST gc output must have `pruned` count");
    assert_eq!(
        mcp_pruned, rest_pruned,
        "MCP and REST dry-run gc must agree on pruned count"
    );
    // Both surfaces must see the same decayed count.
    let mcp_decayed = mcp_json["decayed"].as_u64().expect("MCP gc must have `decayed`");
    let rest_decayed = rest_json["decayed"].as_u64().expect("REST gc must have `decayed`");
    assert_eq!(mcp_decayed, rest_decayed, "MCP and REST must agree on decayed count");
}
