//! Verifies that #[op]-registered ops actually appear in the inventory AND
//! that the fn pointers the macro emits actually execute without panicking.
//!
//! Phase 1.2 acceptance: `inventory::submit!` wiring.
//! Phase 1.6 addition: fn-pointer exercise (prevents "tests pass but the macro
//! emits a broken __op_*_invoke that panics on first real call" hazard).

use std::sync::Arc;

use rein::config::ReinConfig;
use rein::ops::{OpsCliEntry, OpsMcpEntry, OpsMetadata, OpsRestEntry, OpsRuntime};

#[test]
fn stats_registers_on_all_three_surfaces() {
    let cli: Vec<&OpsCliEntry> = inventory::iter::<OpsCliEntry>()
        .filter(|e| e.op_name == "stats")
        .collect();
    assert_eq!(cli.len(), 1, "stats should register one CLI entry");
    assert_eq!(cli[0].name, "stats");

    let mcp: Vec<&OpsMcpEntry> = inventory::iter::<OpsMcpEntry>()
        .filter(|e| e.op_name == "stats")
        .collect();
    assert_eq!(mcp.len(), 1, "stats should register one MCP entry");
    assert_eq!(mcp[0].mcp_name, "rein_stats");

    let rest: Vec<&OpsRestEntry> = inventory::iter::<OpsRestEntry>()
        .filter(|e| e.op_name == "stats")
        .collect();
    assert_eq!(rest.len(), 1, "stats should register one REST entry");
    assert_eq!(rest[0].path_template, "/api/stats");
    assert_eq!(rest[0].method, hyper::Method::GET);

    let meta: Vec<&OpsMetadata> = inventory::iter::<OpsMetadata>()
        .filter(|e| e.name == "stats")
        .collect();
    assert_eq!(meta.len(), 1, "stats should register one metadata entry");
    assert_eq!(meta[0].category, "memory");
    assert!(meta[0].cli_visible);
    assert!(meta[0].mcp_visible);
    assert!(meta[0].rest_visible);
    assert_eq!(meta[0].mcp_name, Some("rein_stats"));
}

#[test]
fn stats_schema_is_empty_object() {
    let meta = inventory::iter::<OpsMetadata>()
        .find(|e| e.name == "stats")
        .expect("stats metadata registered");
    let schema = (meta.params_schema)();
    let value: serde_json::Value = serde_json::to_value(schema).expect("schema to json");
    assert_eq!(value["type"], "object");
    assert!(
        value["properties"].as_object().map(|o| o.is_empty()).unwrap_or(false),
        "no-params op should have empty properties object"
    );
}

#[test]
fn health_registers_with_params_schema() {
    let cli = inventory::iter::<OpsCliEntry>()
        .find(|e| e.op_name == "health")
        .expect("health CLI registered");
    let cmd = (cli.build_clap)();
    assert!(
        cmd.get_arguments().any(|a| a.get_id() == "topic"),
        "health CLI should expose positional `topic` arg"
    );

    let mcp = inventory::iter::<OpsMcpEntry>()
        .find(|e| e.op_name == "health")
        .expect("health MCP registered");
    let schema = (mcp.input_schema)();
    let value: serde_json::Value = serde_json::to_value(schema).expect("schema to json");
    // schemars generates an object schema whose `properties` contains `topic`.
    assert!(
        value["properties"]["topic"].is_object()
            || value["$defs"].is_object()
            || value["definitions"].is_object(),
        "health MCP schema should describe the topic property (got {value})"
    );
}

fn runtime_for_test() -> (Arc<OpsRuntime>, tempfile::TempDir) {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let mut config = ReinConfig::default();
    config.database.path = tmp.path().join("memories.db").to_string_lossy().into_owned();
    let runtime = Arc::new(OpsRuntime::for_cli(Arc::new(config)));
    (runtime, tmp)
}

/// Exercise each migrated op's CLI fn pointer. Without this, a broken
/// `__op_cli_invoke` emission would silently pass the "does the entry exist?"
/// tests above — Codex flagged this gap during the 1.5 audit.
#[tokio::test]
async fn stats_cli_invoke_fn_pointer_returns_rendered_output() {
    let (runtime, _tmp) = runtime_for_test();
    let entry = inventory::iter::<OpsCliEntry>()
        .find(|e| e.op_name == "stats")
        .expect("stats CLI registered");
    let matches = (entry.build_clap)()
        .try_get_matches_from(["stats"])
        .expect("empty stats args parse");
    let out = (entry.invoke)(runtime, &matches)
        .await
        .expect("stats invoke");
    assert!(
        out.contains("Memory stats") && out.contains("total:"),
        "CLI output should match IntoCliText rendering, got: {out}"
    );
}

#[tokio::test]
async fn health_cli_invoke_fn_pointer_handles_no_topic() {
    let (runtime, _tmp) = runtime_for_test();
    let entry = inventory::iter::<OpsCliEntry>()
        .find(|e| e.op_name == "health")
        .expect("health CLI registered");
    let matches = (entry.build_clap)()
        .try_get_matches_from(["health"])
        .expect("health with no topic parses");
    let out = (entry.invoke)(runtime, &matches)
        .await
        .expect("health invoke");
    assert!(
        out.contains("System"),
        "health output should include system status, got: {out}"
    );
}

#[tokio::test]
async fn stats_rest_invoke_fn_pointer_returns_json() {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let mut config = ReinConfig::default();
    config.database.path = tmp.path().join("memories.db").to_string_lossy().into_owned();
    let runtime = Arc::new(OpsRuntime::for_rest(Arc::new(config)));

    let entry = inventory::iter::<OpsRestEntry>()
        .find(|e| e.op_name == "stats")
        .expect("stats REST registered");
    let (status, body) = (entry.invoke)(
        runtime,
        std::collections::HashMap::new(),
        String::new(),
        None,
    )
    .await
    .expect("stats REST invoke");
    assert_eq!(status, hyper::StatusCode::OK);
    let v: serde_json::Value = serde_json::from_slice(&body).expect("valid JSON body");
    assert!(v["total_memories"].is_number(), "got: {v}");
    assert!(v["ltm_count"].is_number());
}

#[test]
fn doctor_registers_on_cli_and_rest_but_not_mcp() {
    let cli: Vec<&OpsCliEntry> = inventory::iter::<OpsCliEntry>()
        .filter(|e| e.op_name == "doctor")
        .collect();
    assert_eq!(cli.len(), 1);
    assert_eq!(cli[0].name, "doctor");

    // doctor intentionally has no MCP surface — agents don't need to trigger
    // --fix mutations remotely, and the read-only diagnostics overlap with
    // `rein health` for agent-side monitoring.
    let mcp: Vec<&OpsMcpEntry> = inventory::iter::<OpsMcpEntry>()
        .filter(|e| e.op_name == "doctor")
        .collect();
    assert!(mcp.is_empty(), "doctor should not expose MCP surface");

    let rest: Vec<&OpsRestEntry> = inventory::iter::<OpsRestEntry>()
        .filter(|e| e.op_name == "doctor")
        .collect();
    assert_eq!(rest.len(), 1);
    assert_eq!(rest[0].path_template, "/api/doctor");
    assert_eq!(rest[0].method, hyper::Method::GET);

    let meta: Vec<&OpsMetadata> = inventory::iter::<OpsMetadata>()
        .filter(|e| e.name == "doctor")
        .collect();
    assert_eq!(meta.len(), 1);
    assert_eq!(meta[0].category, "diagnostics");
    assert!(meta[0].cli_visible);
    assert!(!meta[0].mcp_visible);
    assert!(meta[0].rest_visible);
}

#[test]
fn adaptive_status_registers_on_all_three_surfaces() {
    let cli: Vec<&OpsCliEntry> = inventory::iter::<OpsCliEntry>()
        .filter(|e| e.op_name == "adaptive_status")
        .collect();
    assert_eq!(cli.len(), 1);
    assert_eq!(cli[0].name, "adaptive-status");

    let mcp: Vec<&OpsMcpEntry> = inventory::iter::<OpsMcpEntry>()
        .filter(|e| e.op_name == "adaptive_status")
        .collect();
    assert_eq!(mcp.len(), 1);
    assert_eq!(mcp[0].mcp_name, "rein_adaptive_status");

    let rest: Vec<&OpsRestEntry> = inventory::iter::<OpsRestEntry>()
        .filter(|e| e.op_name == "adaptive_status")
        .collect();
    assert_eq!(rest.len(), 1);
    assert_eq!(rest[0].path_template, "/api/adaptive");
    assert_eq!(rest[0].method, hyper::Method::GET);

    let meta: Vec<&OpsMetadata> = inventory::iter::<OpsMetadata>()
        .filter(|e| e.name == "adaptive_status")
        .collect();
    assert_eq!(meta.len(), 1);
    assert_eq!(meta[0].category, "adaptive");
    assert!(meta[0].cli_visible);
    assert!(meta[0].mcp_visible);
    assert!(meta[0].rest_visible);
}

#[tokio::test]
async fn health_rest_preserves_legacy_top_level_health_key() {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let mut config = ReinConfig::default();
    config.database.path = tmp.path().join("memories.db").to_string_lossy().into_owned();
    let runtime = Arc::new(OpsRuntime::for_rest(Arc::new(config)));

    let entry = inventory::iter::<OpsRestEntry>()
        .find(|e| e.op_name == "health")
        .expect("health REST registered");
    let (status, body) = (entry.invoke)(
        runtime,
        std::collections::HashMap::new(),
        String::new(),
        None,
    )
    .await
    .expect("health REST invoke");
    assert_eq!(status, hyper::StatusCode::OK);
    let v: serde_json::Value = serde_json::from_slice(&body).expect("valid JSON body");
    // GUI reads response.health — preserving the pre-A1 key name matters.
    assert!(
        v["health"].is_array(),
        "response must keep the top-level `health` array (GUI contract), got: {v}"
    );
    assert!(v["indexes"]["hnsw"].is_object());
    assert!(v["status"].is_object());
}
