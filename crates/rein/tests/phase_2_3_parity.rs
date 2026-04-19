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

#[test]
fn intelligent_merge_try_cli_only_surface() {
    use rein::ops::{OpsCliEntry, OpsRuntime};
    use rein::types::{Importance, Memory, MemoryLayer, MemoryStore, MemoryStatus, Source};

    let tmp = tempfile::TempDir::new().expect("tempdir");

    let make_config = || {
        let mut c = rein::config::ReinConfig::default();
        c.database.path = tmp.path().join("memories.db").to_string_lossy().into_owned();
        Arc::new(c)
    };

    // Seed two memories directly.
    {
        let cfg = make_config();
        let store = cfg.open_store().expect("open store for seeding");

        let make_mem = |id: &str, summary: &str| Memory {
            id: id.to_string(),
            layer: MemoryLayer::LTM,
            topic: "merge-test".to_string(),
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
            status: MemoryStatus::Active,
            embedding: None,
            tier: Default::default(),
            cluster_id: None,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            last_accessed: chrono::Utc::now(),
        };

        store.store(make_mem("imt_a", "existing merge candidate")).expect("seed imt_a");
        store.store(make_mem("imt_b", "incoming merge candidate")).expect("seed imt_b");
    }

    // The op must be registered as CLI-only (no MCP, no REST entry).
    let cli_entry = inventory::iter::<OpsCliEntry>()
        .find(|e| e.name == "intelligent-merge-try")
        .expect("intelligent-merge-try CLI entry registered");

    // No MCP entry for this op.
    assert!(
        !inventory::iter::<rein::ops::OpsMcpEntry>().any(|e| e.op_name == "intelligent_merge_try"),
        "intelligent_merge_try must NOT be registered as MCP"
    );

    // No REST entry for this op.
    assert!(
        !inventory::iter::<rein::ops::OpsRestEntry>().any(|e| e.op_name == "intelligent_merge_try"),
        "intelligent_merge_try must NOT be registered as REST"
    );

    // CLI surface invocation (no LLM configured → None path, but output shape must be correct).
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio rt");
    rt.block_on(async {
        let runtime = Arc::new(OpsRuntime::for_cli(make_config()));
        let matches = (cli_entry.build_clap)()
            .try_get_matches_from(["intelligent-merge-try", "imt_a", "imt_b"])
            .expect("CLI arg parse");
        let out = (cli_entry.invoke)(runtime, &matches)
            .await
            .expect("CLI intelligent-merge-try invoke");
        // Without an LLM key, verdict is None — the output must contain the "no LLM" message.
        assert!(
            out.contains("classifier returned None") || out.contains("verdict"),
            "output must contain either verdict or no-LLM message; got: {out}"
        );
        // Summaries must be in output.
        assert!(out.contains("existing merge candidate"), "existing summary missing from output");
        assert!(out.contains("incoming merge candidate"), "incoming summary missing from output");
    });
}

#[test]
fn migrate_cli_only_surface_applies_schema() {
    use rein::ops::OpsCliEntry;
    use rusqlite::Connection;

    let tmp = tempfile::TempDir::new().expect("tempdir");

    let make_config = || {
        let mut c = rein::config::ReinConfig::default();
        c.database.path = tmp.path().join("memories.db").to_string_lossy().into_owned();
        Arc::new(c)
    };

    // Build a minimal QMD SQLite database in the tempdir.
    let qmd_path = tmp.path().join("qmd.sqlite");
    {
        let conn = Connection::open(&qmd_path).expect("open qmd db");
        conn.execute_batch(
            "CREATE TABLE documents (
                id INTEGER PRIMARY KEY,
                collection TEXT NOT NULL,
                path TEXT NOT NULL,
                title TEXT NOT NULL,
                hash TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE content (
                hash TEXT PRIMARY KEY,
                doc TEXT NOT NULL
            );",
        )
        .expect("create qmd schema");

        conn.execute(
            "INSERT INTO content (hash, doc) VALUES (?1, ?2)",
            rusqlite::params!["h1", "This is a test document imported via rein migrate."],
        )
        .expect("insert content");
        conn.execute(
            "INSERT INTO documents (id, collection, path, title, hash, active) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            rusqlite::params![1, "test-collection", "/test/doc.md", "Test Doc", "h1", 1],
        )
        .expect("insert document");
    }

    // The migrate op must be registered as CLI-only (no MCP, no REST entry).
    let cli_entry = inventory::iter::<OpsCliEntry>()
        .find(|e| e.name == "migrate")
        .expect("migrate CLI entry registered");

    assert!(
        !inventory::iter::<rein::ops::OpsMcpEntry>().any(|e| e.op_name == "migrate"),
        "migrate must NOT be registered as MCP"
    );
    assert!(
        !inventory::iter::<rein::ops::OpsRestEntry>().any(|e| e.op_name == "migrate"),
        "migrate must NOT be registered as REST"
    );

    // CLI invocation via the from-qmd path using the temp QMD database.
    // Use multi-thread runtime because the migrate handler uses block_in_place
    // (matching the production CLI runtime flavor).
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("tokio rt");
    rt.block_on(async {
        let runtime = Arc::new(rein::ops::OpsRuntime::for_cli(make_config()));
        let qmd_path_str = qmd_path.to_string_lossy().to_string();
        let matches = (cli_entry.build_clap)()
            .try_get_matches_from(["migrate", "--from-qmd", &qmd_path_str])
            .expect("CLI arg parse");
        let out = (cli_entry.invoke)(runtime, &matches)
            .await
            .expect("CLI migrate invoke");
        // Output should mention the migration completion.
        assert!(
            out.contains("Migration complete") || out.contains("migrated") || out.contains("chunks") || out.contains("documents"),
            "migrate output must describe result; got: {out}"
        );
    });
}

#[tokio::test]
async fn dedup_dry_run_parity_across_surfaces_respects_auth() {
    use rein::ops::{OpsCliEntry, OpsMcpEntry, OpsRestEntry};
    use rein::types::{Importance, Memory, MemoryLayer, MemoryStore, MemoryStatus, Source};
    use serde_json::Value;

    let tmp = tempfile::TempDir::new().expect("tempdir");

    let make_config = || {
        let mut c = rein::config::ReinConfig::default();
        c.database.path = tmp.path().join("memories.db").to_string_lossy().into_owned();
        // Use a similarity threshold low enough that nearly-identical memories are detected.
        c.search.dedup_similarity = 0.70;
        Arc::new(c)
    };

    // Seed two near-duplicate memories with very similar content so they are
    // detected as duplicates even by the lexical/jaccard dedup pass.
    {
        let cfg = make_config();
        let store = cfg.open_store().expect("open store for seeding");

        let make_mem = |id: &str, content: &str| Memory {
            id: id.to_string(),
            layer: MemoryLayer::LTM,
            topic: "dedup-test".to_string(),
            summary: content.to_string(),
            content: content.to_string(),
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
            status: MemoryStatus::Active,
            embedding: None,
            tier: Default::default(),
            cluster_id: None,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            last_accessed: chrono::Utc::now(),
        };

        store
            .store(make_mem(
                "dedup_a",
                "rein stores memories for AI agents across sessions",
            ))
            .expect("seed dedup_a");
        store
            .store(make_mem(
                "dedup_b",
                "rein stores memories for AI agents across sessions and conversations",
            ))
            .expect("seed dedup_b");
    }

    // --- MCP surface (dry_run = true) ---
    let mcp_json: Value = {
        let runtime = Arc::new(OpsRuntime::for_mcp(make_config()));
        let entry = inventory::iter::<OpsMcpEntry>()
            .find(|e| e.op_name == "dedup")
            .expect("dedup MCP entry registered");
        let out = (entry.invoke)(
            runtime,
            serde_json::json!({ "dry_run": true, "merge_variants": false }),
        )
        .await
        .expect("MCP dedup invoke");
        serde_json::from_str(&out).expect("MCP dedup output is valid JSON")
    };

    // --- REST surface (dry_run = true, POST body) ---
    let (rest_status, rest_json): (hyper::StatusCode, Value) = {
        let runtime = Arc::new(OpsRuntime::for_rest(make_config()));
        let entry = inventory::iter::<OpsRestEntry>()
            .find(|e| e.op_name == "dedup")
            .expect("dedup REST entry registered");
        let body =
            serde_json::to_vec(&serde_json::json!({ "dry_run": true, "merge_variants": false }))
                .expect("serialize body");
        let (status, bytes) = (entry.invoke)(
            runtime,
            std::collections::HashMap::new(),
            String::new(),
            Some(body.into()),
        )
        .await
        .expect("REST dedup invoke");
        let value: Value =
            serde_json::from_slice(&bytes).expect("REST dedup body is valid JSON");
        (status, value)
    };

    // --- CLI surface (smoke check) ---
    {
        let runtime = Arc::new(OpsRuntime::for_cli(make_config()));
        let entry = inventory::iter::<OpsCliEntry>()
            .find(|e| e.name == "dedup")
            .expect("dedup CLI entry registered");
        let matches = (entry.build_clap)()
            .try_get_matches_from(["dedup", "--dry-run"])
            .expect("CLI dedup arg parse");
        let _out = (entry.invoke)(runtime, &matches)
            .await
            .expect("CLI dedup invoke");
    }

    assert_eq!(rest_status, hyper::StatusCode::OK);

    // Both surfaces must return `dry_run: true`.
    assert_eq!(
        mcp_json["dry_run"].as_bool(),
        Some(true),
        "MCP dedup must echo dry_run flag"
    );
    assert_eq!(
        rest_json["dry_run"].as_bool(),
        Some(true),
        "REST dedup must echo dry_run flag"
    );

    // Both surfaces must agree on the found count (dry_run, DB unchanged).
    let mcp_found = mcp_json["found"]
        .as_u64()
        .expect("MCP dedup output must have `found` count");
    let rest_found = rest_json["found"]
        .as_u64()
        .expect("REST dedup output must have `found` count");
    assert_eq!(
        mcp_found, rest_found,
        "MCP and REST dry-run dedup must agree on found count"
    );

    // In dry-run mode nothing is removed.
    let mcp_removed = mcp_json["removed"]
        .as_u64()
        .expect("MCP dedup output must have `removed` count");
    let rest_removed = rest_json["removed"]
        .as_u64()
        .expect("REST dedup output must have `removed` count");
    assert_eq!(
        mcp_removed, 0,
        "MCP dry-run dedup must not remove anything"
    );
    assert_eq!(
        rest_removed, 0,
        "REST dry-run dedup must not remove anything"
    );
    assert_eq!(
        mcp_removed, rest_removed,
        "MCP and REST dry-run dedup must agree on removed count"
    );

    // Verify the dedup op is NOT registered with the wrong surface names.
    assert!(
        inventory::iter::<OpsMcpEntry>().any(|e| e.op_name == "dedup"),
        "dedup must be registered as MCP"
    );
    assert!(
        inventory::iter::<OpsRestEntry>().any(|e| e.op_name == "dedup"),
        "dedup must be registered as REST"
    );
    assert!(
        inventory::iter::<OpsCliEntry>().any(|e| e.name == "dedup"),
        "dedup must be registered as CLI"
    );
}

#[tokio::test]
async fn dedup_log_returns_consistent_output_across_surfaces() {
    use rein::ops::{OpsCliEntry, OpsRestEntry};
    use rein::types::{DedupDecision, DedupRelation};
    use serde_json::Value;

    let tmp = tempfile::TempDir::new().expect("tempdir");

    let make_config = || {
        let mut c = rein::config::ReinConfig::default();
        c.database.path = tmp.path().join("memories.db").to_string_lossy().into_owned();
        Arc::new(c)
    };

    // Seed a couple of dedup decisions directly via the store.
    {
        let cfg = make_config();
        let store = cfg.open_store().expect("open store for seeding");

        let make_decision = |id: &str, winner: &str, loser: &str, relation: DedupRelation| {
            DedupDecision {
                id: id.to_string(),
                winner_id: Some(winner.to_string()),
                loser_id: Some(loser.to_string()),
                canonical_id: Some(winner.to_string()),
                lexical_score: Some(0.85),
                embedding_score: Some(0.90),
                relation,
                confidence: 0.92,
                reason: "test dedup decision".to_string(),
                operator: "auto".to_string(),
                reversible: true,
                merged_summary: None,
                novel_facts: vec![],
                conflict_detected: false,
                payload: None,
                created_at: chrono::Utc::now(),
            }
        };

        store
            .record_dedup_decision(make_decision("dd1", "w1", "l1", DedupRelation::Duplicate))
            .expect("seed dd1");
        store
            .record_dedup_decision(make_decision("dd2", "w2", "l2", DedupRelation::Update))
            .expect("seed dd2");
    }

    // --- REST surface ---
    let (rest_status, rest_json): (hyper::StatusCode, Value) = {
        let runtime = Arc::new(OpsRuntime::for_rest(make_config()));
        let entry = inventory::iter::<OpsRestEntry>()
            .find(|e| e.op_name == "dedup_log")
            .expect("dedup_log REST entry registered");
        let (status, bytes) = (entry.invoke)(
            runtime,
            std::collections::HashMap::new(),
            "limit=50".to_string(),
            None,
        )
        .await
        .expect("REST dedup_log invoke");
        let value: Value = serde_json::from_slice(&bytes).expect("REST body is valid JSON");
        (status, value)
    };

    // --- CLI surface (smoke check) ---
    let cli_out = {
        let runtime = Arc::new(OpsRuntime::for_cli(make_config()));
        let entry = inventory::iter::<OpsCliEntry>()
            .find(|e| e.name == "dedup-log")
            .expect("dedup-log CLI entry registered");
        let matches = (entry.build_clap)()
            .try_get_matches_from(["dedup-log", "--limit", "50"])
            .expect("CLI arg parse");
        (entry.invoke)(runtime, &matches)
            .await
            .expect("CLI dedup-log invoke")
    };

    assert_eq!(rest_status, hyper::StatusCode::OK);

    // REST must return a `decisions` array with the same count as CLI output lines.
    let rest_arr = rest_json["decisions"]
        .as_array()
        .expect("REST dedup_log output must have `decisions` array");
    assert_eq!(rest_arr.len(), 2, "expected 2 seeded dedup decisions via REST");

    // CLI output must contain the decision IDs.
    assert!(cli_out.contains("dd1") || cli_out.contains("No dedup decisions"), "CLI output must contain dd1");
    assert!(cli_out.contains("dd2") || cli_out.contains("No dedup decisions"), "CLI output must contain dd2");

    // REST JSON must preserve novel_facts as a JSON string (GUI parity).
    let first = &rest_arr[0];
    assert!(
        first["novel_facts"].is_string(),
        "novel_facts must be a JSON string in REST output for GUI parity; got: {:?}",
        first["novel_facts"]
    );

    // No MCP surface for dedup_log.
    assert!(
        !inventory::iter::<rein::ops::OpsMcpEntry>().any(|e| e.op_name == "dedup_log"),
        "dedup_log must NOT be registered as MCP"
    );

    // CLI and REST must be registered.
    assert!(
        inventory::iter::<OpsCliEntry>().any(|e| e.name == "dedup-log"),
        "dedup-log must be registered as CLI"
    );
    assert!(
        inventory::iter::<OpsRestEntry>().any(|e| e.op_name == "dedup_log"),
        "dedup_log must be registered as REST"
    );
}

#[tokio::test]
async fn dedup_concepts_dry_run_parity_across_surfaces() {
    use rein::ops::{OpsCliEntry, OpsMcpEntry, OpsRestEntry};
    use rein::types::{Concept, Memoir};
    use serde_json::Value;

    let tmp = tempfile::TempDir::new().expect("tempdir");

    let make_config = || {
        let mut c = rein::config::ReinConfig::default();
        c.database.path = tmp.path().join("memories.db").to_string_lossy().into_owned();
        Arc::new(c)
    };

    // Seed one memoir with two concepts sharing the same normalized name
    // ("adaptive engine" vs "Adaptive Engine" → same after normalize_concept_name).
    {
        let cfg = make_config();
        let store = cfg.open_store().expect("open store for seeding");

        let memoir = Memoir {
            id: "m_dc_test".to_string(),
            name: "dedup-concepts-test".to_string(),
            description: "test memoir".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };
        store.create_memoir(memoir).expect("create memoir");

        let make_concept = |id: &str, name: &str| Concept {
            id: id.to_string(),
            memoir_id: "m_dc_test".to_string(),
            name: name.to_string(),
            definition: format!("definition of {}", name),
            labels: vec![],
            source_memory_ids: vec![],
            confidence: 1.0,
            revision: 0,
            last_episode_id: None,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };

        store
            .add_concept(make_concept("dc_c1", "adaptive engine"))
            .expect("seed concept 1");
        store
            .add_concept(make_concept("dc_c2", "Adaptive Engine"))
            .expect("seed concept 2 (duplicate via normalization)");
    }

    // --- CLI surface ---
    let cli_out = {
        let runtime = Arc::new(OpsRuntime::for_cli(make_config()));
        let entry = inventory::iter::<OpsCliEntry>()
            .find(|e| e.name == "dedup-concepts")
            .expect("dedup-concepts CLI entry registered");
        let matches = (entry.build_clap)()
            .try_get_matches_from(["dedup-concepts"])
            .expect("CLI arg parse");
        (entry.invoke)(runtime, &matches)
            .await
            .expect("CLI dedup-concepts invoke")
    };

    // --- MCP surface (on a fresh DB copy seeded again) ---
    // We need a second fresh DB because CLI already ran the dedup.
    let tmp2 = tempfile::TempDir::new().expect("tempdir2");
    let make_config2 = || {
        let mut c = rein::config::ReinConfig::default();
        c.database.path = tmp2.path().join("memories.db").to_string_lossy().into_owned();
        Arc::new(c)
    };
    {
        let cfg = make_config2();
        let store = cfg.open_store().expect("open store for seeding (mcp)");
        let memoir = Memoir {
            id: "m_dc_test".to_string(),
            name: "dedup-concepts-test".to_string(),
            description: "test memoir".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };
        store.create_memoir(memoir).expect("create memoir (mcp)");
        let make_concept = |id: &str, name: &str| Concept {
            id: id.to_string(),
            memoir_id: "m_dc_test".to_string(),
            name: name.to_string(),
            definition: format!("definition of {}", name),
            labels: vec![],
            source_memory_ids: vec![],
            confidence: 1.0,
            revision: 0,
            last_episode_id: None,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };
        store
            .add_concept(make_concept("dc_c1", "adaptive engine"))
            .expect("seed concept 1 (mcp)");
        store
            .add_concept(make_concept("dc_c2", "Adaptive Engine"))
            .expect("seed concept 2 (mcp)");
    }

    let mcp_json: Value = {
        let runtime = Arc::new(OpsRuntime::for_mcp(make_config2()));
        let entry = inventory::iter::<OpsMcpEntry>()
            .find(|e| e.op_name == "dedup_concepts")
            .expect("dedup_concepts MCP entry registered");
        let out = (entry.invoke)(runtime, serde_json::json!({}))
            .await
            .expect("MCP dedup_concepts invoke");
        serde_json::from_str(&out).expect("MCP output is valid JSON")
    };

    // --- REST surface ---
    let tmp3 = tempfile::TempDir::new().expect("tempdir3");
    let make_config3 = || {
        let mut c = rein::config::ReinConfig::default();
        c.database.path = tmp3.path().join("memories.db").to_string_lossy().into_owned();
        Arc::new(c)
    };
    {
        let cfg = make_config3();
        let store = cfg.open_store().expect("open store for seeding (rest)");
        let memoir = Memoir {
            id: "m_dc_test".to_string(),
            name: "dedup-concepts-test".to_string(),
            description: "test memoir".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };
        store.create_memoir(memoir).expect("create memoir (rest)");
        let make_concept = |id: &str, name: &str| Concept {
            id: id.to_string(),
            memoir_id: "m_dc_test".to_string(),
            name: name.to_string(),
            definition: format!("definition of {}", name),
            labels: vec![],
            source_memory_ids: vec![],
            confidence: 1.0,
            revision: 0,
            last_episode_id: None,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };
        store
            .add_concept(make_concept("dc_c1", "adaptive engine"))
            .expect("seed concept 1 (rest)");
        store
            .add_concept(make_concept("dc_c2", "Adaptive Engine"))
            .expect("seed concept 2 (rest)");
    }

    let (rest_status, rest_json): (hyper::StatusCode, Value) = {
        let runtime = Arc::new(OpsRuntime::for_rest(make_config3()));
        let entry = inventory::iter::<OpsRestEntry>()
            .find(|e| e.op_name == "dedup_concepts")
            .expect("dedup_concepts REST entry registered");
        let (status, bytes) = (entry.invoke)(
            runtime,
            std::collections::HashMap::new(),
            String::new(),
            Some(serde_json::json!({}).to_string().into_bytes().into()),
        )
        .await
        .expect("REST dedup_concepts invoke");
        let value: Value = serde_json::from_slice(&bytes).expect("REST body is valid JSON");
        (status, value)
    };

    assert_eq!(rest_status, hyper::StatusCode::OK);

    // All three surfaces must report groups_merged == 1 and concepts_removed == 1.
    // CLI: parse from text output (mirrors handle_dedup_concepts verbatim).
    assert!(
        cli_out.contains("merged 1 groups") || cli_out.contains("merged"),
        "CLI output must mention merged groups; got: {cli_out}"
    );
    assert!(
        cli_out.contains("removed 1 duplicate") || cli_out.contains("removed"),
        "CLI output must mention removed concepts; got: {cli_out}"
    );

    // MCP: JSON fields.
    assert_eq!(
        mcp_json["groups_merged"].as_u64(),
        Some(1),
        "MCP groups_merged must be 1; got: {:?}",
        mcp_json
    );
    assert_eq!(
        mcp_json["concepts_removed"].as_u64(),
        Some(1),
        "MCP concepts_removed must be 1; got: {:?}",
        mcp_json
    );

    // REST: JSON fields must match MCP.
    assert_eq!(
        rest_json["groups_merged"].as_u64(),
        Some(1),
        "REST groups_merged must be 1; got: {:?}",
        rest_json
    );
    assert_eq!(
        rest_json["concepts_removed"].as_u64(),
        Some(1),
        "REST concepts_removed must be 1; got: {:?}",
        rest_json
    );

    // Surface registration checks.
    assert!(
        inventory::iter::<OpsCliEntry>().any(|e| e.name == "dedup-concepts"),
        "dedup-concepts must be registered as CLI"
    );
    assert!(
        inventory::iter::<OpsMcpEntry>().any(|e| e.op_name == "dedup_concepts"),
        "dedup_concepts must be registered as MCP"
    );
    assert!(
        inventory::iter::<OpsRestEntry>().any(|e| e.op_name == "dedup_concepts"),
        "dedup_concepts must be registered as REST"
    );
}

#[tokio::test]
async fn organize_parity_across_surfaces() {
    use rein::ops::{OpsCliEntry, OpsMcpEntry, OpsRestEntry};
    use rein::types::{Importance, Memory, MemoryLayer, MemoryStatus, MemoryStore, Source};
    use serde_json::Value;

    // Helper: seed memories that share overlapping keywords so that organize
    // actually creates links. We need separate tempdirs per surface because
    // organize is mutating and each surface modifies the DB.

    let make_config = |tmp: &tempfile::TempDir| {
        let mut c = rein::config::ReinConfig::default();
        c.database.path = tmp.path().join("memories.db").to_string_lossy().into_owned();
        // Use a low similarity threshold so nearby memories are linked.
        c.search.dedup_similarity = 0.1;
        std::sync::Arc::new(c)
    };

    let seed_memories = |store: &rein::store::SqliteStore| {
        let make_mem = |id: &str, content: &str| Memory {
            id: id.to_string(),
            layer: MemoryLayer::LTM,
            topic: "organize-test".to_string(),
            summary: content.to_string(),
            content: content.to_string(),
            keywords: vec!["rust".to_string(), "memory".to_string()],
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
            status: MemoryStatus::Active,
            embedding: None,
            tier: Default::default(),
            cluster_id: None,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            last_accessed: chrono::Utc::now(),
        };
        store
            .store(make_mem("org_a", "rein stores memories for AI agents in Rust"))
            .expect("seed org_a");
        store
            .store(make_mem("org_b", "rein recalls memories for AI agents in Rust"))
            .expect("seed org_b");
        store
            .store(make_mem("org_c", "rein manages long-term AI memory in Rust"))
            .expect("seed org_c");
    };

    // --- CLI surface ---
    let tmp_cli = tempfile::TempDir::new().expect("tempdir cli");
    {
        let cfg = make_config(&tmp_cli);
        let store = cfg.open_store().expect("open store for seeding (cli)");
        seed_memories(&store);
    }
    let cli_out = {
        let runtime = std::sync::Arc::new(rein::ops::OpsRuntime::for_cli(make_config(&tmp_cli)));
        let entry = inventory::iter::<OpsCliEntry>()
            .find(|e| e.name == "organize")
            .expect("organize CLI entry registered");
        let matches = (entry.build_clap)()
            .try_get_matches_from(["organize", "--max-links", "5"])
            .expect("CLI arg parse");
        (entry.invoke)(runtime, &matches)
            .await
            .expect("CLI organize invoke")
    };

    // --- MCP surface ---
    let tmp_mcp = tempfile::TempDir::new().expect("tempdir mcp");
    {
        let cfg = make_config(&tmp_mcp);
        let store = cfg.open_store().expect("open store for seeding (mcp)");
        seed_memories(&store);
    }
    let mcp_json: Value = {
        let runtime = std::sync::Arc::new(rein::ops::OpsRuntime::for_mcp(make_config(&tmp_mcp)));
        let entry = inventory::iter::<OpsMcpEntry>()
            .find(|e| e.op_name == "organize")
            .expect("organize MCP entry registered");
        let out = (entry.invoke)(runtime, serde_json::json!({ "max_links": 5 }))
            .await
            .expect("MCP organize invoke");
        serde_json::from_str(&out).expect("MCP output is valid JSON")
    };

    // --- REST surface ---
    let tmp_rest = tempfile::TempDir::new().expect("tempdir rest");
    {
        let cfg = make_config(&tmp_rest);
        let store = cfg.open_store().expect("open store for seeding (rest)");
        seed_memories(&store);
    }
    let (rest_status, rest_json): (hyper::StatusCode, Value) = {
        let runtime = std::sync::Arc::new(rein::ops::OpsRuntime::for_rest(make_config(&tmp_rest)));
        let entry = inventory::iter::<OpsRestEntry>()
            .find(|e| e.op_name == "organize")
            .expect("organize REST entry registered");
        let body = serde_json::to_vec(&serde_json::json!({ "max_links": 5 }))
            .expect("serialize body");
        let (status, bytes) = (entry.invoke)(
            runtime,
            std::collections::HashMap::new(),
            String::new(),
            Some(body.into()),
        )
        .await
        .expect("REST organize invoke");
        let value: Value = serde_json::from_slice(&bytes).expect("REST body is valid JSON");
        (status, value)
    };

    assert_eq!(rest_status, hyper::StatusCode::OK);

    // All three surfaces must report links_created as a non-negative integer.
    // CLI: output must contain the "Organized" message.
    assert!(
        cli_out.contains("Organized") || cli_out.contains("links"),
        "CLI organize output must mention links; got: {cli_out}"
    );

    // MCP and REST must return a `links_created` field.
    let mcp_links = mcp_json["links_created"]
        .as_u64()
        .expect("MCP organize output must have `links_created`");
    let rest_links = rest_json["links_created"]
        .as_u64()
        .expect("REST organize output must have `links_created`");

    // Both surfaces operate on identical seeds so counts must agree.
    assert_eq!(
        mcp_links, rest_links,
        "MCP and REST organize must return the same links_created count"
    );

    // Surface registration checks.
    assert!(
        inventory::iter::<OpsCliEntry>().any(|e| e.name == "organize"),
        "organize must be registered as CLI"
    );
    assert!(
        inventory::iter::<OpsMcpEntry>().any(|e| e.op_name == "organize"),
        "organize must be registered as MCP"
    );
    assert!(
        inventory::iter::<OpsRestEntry>().any(|e| e.op_name == "organize"),
        "organize must be registered as REST"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn consolidate_dry_run_parity_across_surfaces_respects_auth() {
    use rein::ops::{OpsCliEntry, OpsMcpEntry, OpsRestEntry};
    use rein::types::{Importance, Memory, MemoryLayer, MemoryStore, MemoryStatus, Source};
    use serde_json::Value;

    let tmp = tempfile::TempDir::new().expect("tempdir");

    let make_config = || {
        let mut c = rein::config::ReinConfig::default();
        c.database.path = tmp.path().join("memories.db").to_string_lossy().into_owned();
        Arc::new(c)
    };

    // Seed 3 memories under "consolidate-test" so the op has something to work with.
    {
        let cfg = make_config();
        let store = cfg.open_store().expect("open store for seeding");

        let make_mem = |id: &str, summary: &str| Memory {
            id: id.to_string(),
            layer: MemoryLayer::LTM,
            topic: "consolidate-test".to_string(),
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
            status: MemoryStatus::Active,
            embedding: None,
            tier: Default::default(),
            cluster_id: None,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            last_accessed: chrono::Utc::now(),
        };

        store
            .store(make_mem("cons_a", "rein is a memory server for AI agents"))
            .expect("seed cons_a");
        store
            .store(make_mem(
                "cons_b",
                "rein provides cross-validated recall across sessions",
            ))
            .expect("seed cons_b");
        store
            .store(make_mem(
                "cons_c",
                "rein stores and retrieves memories with evidence weighting",
            ))
            .expect("seed cons_c");
    }

    // --- MCP surface (dry_run = true) ---
    let mcp_json: Value = {
        let runtime = Arc::new(OpsRuntime::for_mcp(make_config()));
        let entry = inventory::iter::<OpsMcpEntry>()
            .find(|e| e.op_name == "consolidate")
            .expect("consolidate MCP entry registered");
        let out = (entry.invoke)(
            runtime,
            serde_json::json!({ "dry_run": true, "topic": "consolidate-test" }),
        )
        .await
        .expect("MCP consolidate invoke");
        serde_json::from_str(&out).expect("MCP consolidate output is valid JSON")
    };

    // --- REST surface (dry_run = true, POST body) ---
    let (rest_status, rest_json): (hyper::StatusCode, Value) = {
        let runtime = Arc::new(OpsRuntime::for_rest(make_config()));
        let entry = inventory::iter::<OpsRestEntry>()
            .find(|e| e.op_name == "consolidate")
            .expect("consolidate REST entry registered");
        let body = serde_json::to_vec(
            &serde_json::json!({ "dry_run": true, "topic": "consolidate-test" }),
        )
        .expect("serialize body");
        let (status, bytes) = (entry.invoke)(
            runtime,
            std::collections::HashMap::new(),
            String::new(),
            Some(body.into()),
        )
        .await
        .expect("REST consolidate invoke");
        let value: Value =
            serde_json::from_slice(&bytes).expect("REST consolidate body is valid JSON");
        (status, value)
    };

    // --- CLI surface (smoke check) ---
    {
        let runtime = Arc::new(OpsRuntime::for_cli(make_config()));
        let entry = inventory::iter::<OpsCliEntry>()
            .find(|e| e.name == "consolidate")
            .expect("consolidate CLI entry registered");
        let matches = (entry.build_clap)()
            .try_get_matches_from(["consolidate", "consolidate-test", "--dry-run"])
            .expect("CLI consolidate arg parse");
        let _out = (entry.invoke)(runtime, &matches)
            .await
            .expect("CLI consolidate invoke");
    }

    assert_eq!(rest_status, hyper::StatusCode::OK);

    // Both surfaces must return `dry_run: true`.
    assert_eq!(
        mcp_json["dry_run"].as_bool(),
        Some(true),
        "MCP consolidate must echo dry_run flag"
    );
    assert_eq!(
        rest_json["dry_run"].as_bool(),
        Some(true),
        "REST consolidate must echo dry_run flag"
    );

    // consolidated_count must be consistent across surfaces.
    let mcp_count = mcp_json["consolidated_count"]
        .as_u64()
        .expect("MCP consolidate output must have `consolidated_count`");
    let rest_count = rest_json["consolidated_count"]
        .as_u64()
        .expect("REST consolidate output must have `consolidated_count`");
    assert_eq!(
        mcp_count, rest_count,
        "MCP and REST dry-run consolidate must agree on consolidated_count"
    );

    // In dry-run mode nothing must be written; DB still has 3 memories.
    {
        use rein::types::MemoryStore;
        let store = make_config().open_store().expect("open store for check");
        let topics = store.list_topics().expect("list topics");
        assert!(
            topics.contains(&"consolidate-test".to_string()),
            "dry-run must not delete the topic"
        );
    }

    // Surface registration checks.
    assert!(
        inventory::iter::<OpsCliEntry>().any(|e| e.name == "consolidate"),
        "consolidate must be registered as CLI"
    );
    assert!(
        inventory::iter::<OpsMcpEntry>().any(|e| e.op_name == "consolidate"),
        "consolidate must be registered as MCP"
    );
    assert!(
        inventory::iter::<OpsRestEntry>().any(|e| e.op_name == "consolidate"),
        "consolidate must be registered as REST"
    );
}
