//! Phase 2.5 memory-op parity tests — first consumer: `forget`.
//!
//! Verifies that the `#[op]`-migrated `forget` produces consistent results
//! across all three surfaces (MCP, REST, CLI) and that the underlying store
//! record is actually removed.

use std::sync::Arc;

use bytes::Bytes;
use hyper::{Method, Request, StatusCode};
use rein::config::ReinConfig;
use rein::mcp::rest::handle_rest_request_with_body;
use rein::ops::{OpsCliEntry, OpsMcpEntry, OpsRestEntry, OpsRuntime};
use rein::types::{Importance, Memory, MemoryLayer, MemoryStatus, MemoryStore, Source};

// ─── helpers ────────────────────────────────────────────────────────────────

fn make_config(tmp: &tempfile::TempDir) -> Arc<ReinConfig> {
    let mut c = ReinConfig::default();
    c.database.path = tmp
        .path()
        .join("memories.db")
        .to_string_lossy()
        .into_owned();
    Arc::new(c)
}

fn seed_memory(config: &ReinConfig, id: &str) {
    let store = config.open_store().expect("open store for seeding");
    let mem = Memory {
        id: id.to_string(),
        layer: MemoryLayer::LTM,
        topic: "test".to_string(),
        summary: format!("memory {id}"),
        content: format!("content for {id}"),
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
        archival_summary: None,
        archival_summary_at: None,
        archival_summary_version: None,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        last_accessed: chrono::Utc::now(),
    };
    store.store(mem).expect("seed memory");
}

fn assert_not_in_store(config: &ReinConfig, id: &str) {
    let store = config.open_store().expect("open store for check");
    let result = store.get(id);
    assert!(
        result.is_err(),
        "memory {id} should be gone from store, but get() returned Ok"
    );
}

fn delete_req_with_mutation(path: &str) -> Request<http_body_util::Full<Bytes>> {
    Request::builder()
        .method(Method::DELETE)
        .uri(format!("http://localhost{path}"))
        .header("x-rein-action", "1")
        .body(http_body_util::Full::new(Bytes::new()))
        .unwrap()
}

async fn rest_dispatch(
    req: &Request<http_body_util::Full<Bytes>>,
    config: &ReinConfig,
) -> (StatusCode, serde_json::Value) {
    let resp = handle_rest_request_with_body(req, None, config)
        .await
        .expect("handle_rest_request_with_body returned None");
    let status = resp.status();
    let body_bytes = {
        use http_body_util::BodyExt;
        resp.into_body().collect().await.unwrap().to_bytes()
    };
    let json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap_or_default();
    (status, json)
}

// ─── forget: inventory registration ─────────────────────────────────────────

#[test]
fn forget_registered_in_all_three_inventory_surfaces() {
    let mcp = inventory::iter::<OpsMcpEntry>()
        .find(|e| e.mcp_name == "rein_forget")
        .is_some();
    assert!(mcp, "rein_forget must be registered as an OpsMcpEntry");

    let rest = inventory::iter::<OpsRestEntry>()
        .find(|e| e.op_name == "forget")
        .is_some();
    assert!(rest, "forget must be registered as an OpsRestEntry");

    let cli = inventory::iter::<OpsCliEntry>()
        .find(|e| e.name == "forget")
        .is_some();
    assert!(cli, "forget must be registered as an OpsCliEntry");
}

#[test]
fn forget_rest_entry_has_correct_path_segments() {
    use rein::ops::PathSegment;
    let entry = inventory::iter::<OpsRestEntry>()
        .find(|e| e.op_name == "forget")
        .expect("forget OpsRestEntry must exist");

    assert_eq!(entry.method, hyper::Method::DELETE);
    assert_eq!(entry.path_template, "/api/memories/{id}");

    let segs = entry.path_segments;
    assert_eq!(
        segs.len(),
        3,
        "expected 3 segments for /api/memories/{{id}}"
    );
    assert_eq!(segs[0], PathSegment::Literal("api"));
    assert_eq!(segs[1], PathSegment::Literal("memories"));
    assert_eq!(segs[2], PathSegment::Param("id"));
}

#[test]
fn forget_rest_entry_requires_mutation_marker() {
    use rein::ops::AuthPolicy;
    let entry = inventory::iter::<OpsRestEntry>()
        .find(|e| e.op_name == "forget")
        .expect("forget OpsRestEntry must exist");
    assert_eq!(
        entry.auth_policy,
        AuthPolicy::MutationMarker,
        "forget must declare MutationMarker auth"
    );
}

// ─── forget: REST surface ────────────────────────────────────────────────────

#[tokio::test]
async fn forget_rest_deletes_memory_and_returns_200() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg = make_config(&tmp);
    seed_memory(&cfg, "mem-rest-1");

    let req = delete_req_with_mutation("/api/memories/mem-rest-1");
    let (status, body) = rest_dispatch(&req, &cfg).await;

    assert_eq!(status, StatusCode::OK, "should return 200, body={body}");
    assert_eq!(
        body["id"].as_str(),
        Some("mem-rest-1"),
        "response id must match"
    );
    assert_eq!(
        body["deleted"].as_bool(),
        Some(true),
        "deleted must be true"
    );

    assert_not_in_store(&cfg, "mem-rest-1");
}

#[tokio::test]
async fn forget_rest_returns_403_without_mutation_marker() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg = make_config(&tmp);
    seed_memory(&cfg, "mem-rest-auth");

    // Build DELETE request without x-rein-action header.
    let req = Request::builder()
        .method(Method::DELETE)
        .uri("http://localhost/api/memories/mem-rest-auth")
        .body(http_body_util::Full::new(Bytes::new()))
        .unwrap();

    let resp = handle_rest_request_with_body(&req, None, &cfg)
        .await
        .expect("should get a response");
    assert_eq!(resp.status(), StatusCode::FORBIDDEN, "missing marker → 403");
}

#[tokio::test]
async fn forget_rest_returns_error_for_missing_id() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg = make_config(&tmp);

    let req = delete_req_with_mutation("/api/memories/nonexistent-xyz");
    let (status, _body) = rest_dispatch(&req, &cfg).await;

    // H4 flow: ReinError::NotFound → 404.
    assert_eq!(
        status,
        StatusCode::NOT_FOUND,
        "missing memory should return 404"
    );
}

// ─── forget: MCP surface ────────────────────────────────────────────────────

#[tokio::test]
async fn forget_mcp_deletes_memory_and_returns_json() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg = make_config(&tmp);
    seed_memory(&cfg, "mem-mcp-1");

    let runtime = Arc::new(OpsRuntime::for_mcp(cfg.clone()));
    let entry = inventory::iter::<OpsMcpEntry>()
        .find(|e| e.mcp_name == "rein_forget")
        .expect("rein_forget MCP entry");

    let out = (entry.invoke)(runtime, serde_json::json!({ "id": "mem-mcp-1" }))
        .await
        .expect("MCP forget invoke");

    // Non-compact → serialized JSON.
    let json: serde_json::Value = serde_json::from_str(&out).expect("MCP output must be JSON");
    assert_eq!(json["id"].as_str(), Some("mem-mcp-1"));
    assert_eq!(json["deleted"].as_bool(), Some(true));

    assert_not_in_store(&cfg, "mem-mcp-1");
}

#[tokio::test]
async fn forget_mcp_compact_returns_ok_prefix() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg = make_config(&tmp);
    seed_memory(&cfg, "mem-mcp-compact");

    let runtime = Arc::new(OpsRuntime::for_mcp(cfg.clone()));
    runtime.set_compact(true);
    let entry = inventory::iter::<OpsMcpEntry>()
        .find(|e| e.mcp_name == "rein_forget")
        .expect("rein_forget MCP entry");

    let out = (entry.invoke)(runtime, serde_json::json!({ "id": "mem-mcp-compact" }))
        .await
        .expect("MCP compact forget invoke");

    // Compact mode → IntoMarkdown::to_markdown → "ok:{id}".
    assert_eq!(
        out, "ok:mem-mcp-compact",
        "compact MCP output must match legacy 'ok:{{id}}'"
    );
}

// ─── forget: CLI surface ────────────────────────────────────────────────────

#[tokio::test]
async fn forget_cli_deletes_memory_and_prints_deleted_message() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg = make_config(&tmp);
    seed_memory(&cfg, "mem-cli-1");

    let runtime = Arc::new(OpsRuntime::for_cli(cfg.clone()));
    let entry = inventory::iter::<OpsCliEntry>()
        .find(|e| e.name == "forget")
        .expect("forget CLI entry");

    let matches = (entry.build_clap)()
        .try_get_matches_from(["forget", "mem-cli-1"])
        .expect("CLI arg parse");
    let out = (entry.invoke)(runtime, &matches)
        .await
        .expect("CLI forget invoke");

    // CLI output must match the verbatim legacy handle_forget output.
    assert_eq!(
        out, "Deleted memory: mem-cli-1",
        "CLI output must match legacy 'Deleted memory: {{id}}'"
    );

    assert_not_in_store(&cfg, "mem-cli-1");
}
