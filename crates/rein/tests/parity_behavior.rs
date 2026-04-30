//! Cross-surface parity tests for #[op]-migrated operations.
//!
//! Phase 1 exit criterion (Task 1.8): the same op, invoked through the CLI
//! OpsCliEntry, MCP OpsMcpEntry, and REST OpsRestEntry fn pointers, must
//! produce structurally-equivalent data. These tests dispatch each surface
//! independently and diff the parsed JSON outputs.
//!
//! This closes the gap Codex flagged: inventory_registration.rs exercises
//! individual fn pointers but doesn't prove cross-surface equivalence.

use std::sync::Arc;

use rein::config::ReinConfig;
use rein::ops::{OpsCliEntry, OpsMcpEntry, OpsRestEntry, OpsRuntime};
use serde_json::Value;

fn config_for_test() -> (Arc<ReinConfig>, tempfile::TempDir) {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let mut config = ReinConfig::default();
    config.database.path = tmp
        .path()
        .join("memories.db")
        .to_string_lossy()
        .into_owned();
    config.proxy.allow_unauthenticated_loopback = true;
    (Arc::new(config), tmp)
}

async fn invoke_cli(op_name: &str, args: &[&str]) -> String {
    let (config, _tmp) = config_for_test();
    let runtime = Arc::new(OpsRuntime::for_cli(config));
    let entry = inventory::iter::<OpsCliEntry>()
        .find(|e| e.op_name == op_name)
        .expect("CLI entry registered");
    let matches = (entry.build_clap)()
        .try_get_matches_from(std::iter::once(op_name).chain(args.iter().copied()))
        .expect("parse CLI args");
    (entry.invoke)(runtime, &matches).await.expect("CLI invoke")
}

async fn invoke_mcp(op_name: &str, args: Value) -> Value {
    let (config, _tmp) = config_for_test();
    let runtime = Arc::new(OpsRuntime::for_mcp(config));
    let entry = inventory::iter::<OpsMcpEntry>()
        .find(|e| e.op_name == op_name)
        .expect("MCP entry registered");
    let out = (entry.invoke)(runtime, args).await.expect("MCP invoke");
    serde_json::from_str(&out).expect("MCP output is valid JSON")
}

async fn invoke_rest(op_name: &str, query: &str) -> (hyper::StatusCode, Value) {
    let (config, _tmp) = config_for_test();
    let runtime = Arc::new(OpsRuntime::for_rest(config));
    let entry = inventory::iter::<OpsRestEntry>()
        .find(|e| e.op_name == op_name)
        .expect("REST entry registered");
    let (status, bytes, _content_type) = (entry.invoke)(
        runtime,
        std::collections::HashMap::new(),
        query.to_string(),
        None,
    )
    .await
    .expect("REST invoke");
    let value: Value = serde_json::from_slice(&bytes).expect("REST body is valid JSON");
    (status, value)
}

#[tokio::test]
async fn stats_parity_mcp_and_rest_emit_identical_fields() {
    // MCP returns a serialized StatsOutput JSON string; REST returns the same
    // struct as JSON bytes. Both must have identical top-level shape.
    let mcp = invoke_mcp("stats", Value::Object(Default::default())).await;
    let (rest_status, rest) = invoke_rest("stats", "").await;

    assert_eq!(rest_status, hyper::StatusCode::OK);

    // Each instantiation opens a fresh temp db, so counts differ (both empty).
    // Instead of comparing values, compare the set of keys and each value's
    // JSON type — that's what "same structured contract" means here.
    let mcp_obj = mcp.as_object().expect("stats MCP is an object");
    let rest_obj = rest.as_object().expect("stats REST is an object");

    let mcp_keys: std::collections::BTreeSet<_> = mcp_obj.keys().collect();
    let rest_keys: std::collections::BTreeSet<_> = rest_obj.keys().collect();
    assert_eq!(
        mcp_keys, rest_keys,
        "MCP and REST stats payloads must expose the same keys"
    );

    for key in mcp_keys {
        let m = &mcp_obj[key];
        let r = &rest_obj[key];
        assert_eq!(
            json_type_tag(m),
            json_type_tag(r),
            "stats field `{key}` has different JSON types across surfaces (MCP: {m:?}, REST: {r:?})"
        );
    }
}

#[tokio::test]
async fn stats_parity_cli_dispatch_runs_to_completion() {
    let out = invoke_cli("stats", &[]).await;
    assert!(
        out.contains("Memory stats") && out.contains("total:"),
        "stats CLI output drifted: {out}"
    );
}

#[tokio::test]
async fn health_parity_mcp_and_rest_share_shape() {
    // MCP inventory input: {} → all topics.
    // REST inventory input: empty query string → same.
    let mcp = invoke_mcp("health", Value::Object(Default::default())).await;
    let (rest_status, rest) = invoke_rest("health", "").await;

    assert_eq!(rest_status, hyper::StatusCode::OK);

    let mcp_obj = mcp.as_object().expect("health MCP is an object");
    let rest_obj = rest.as_object().expect("health REST is an object");

    let mcp_keys: std::collections::BTreeSet<_> = mcp_obj.keys().collect();
    let rest_keys: std::collections::BTreeSet<_> = rest_obj.keys().collect();
    assert_eq!(
        mcp_keys, rest_keys,
        "MCP and REST health payloads must expose the same top-level keys"
    );

    // Critical contract: the GUI reads response.health[]. Phase 1.6 preserves
    // this key via #[serde(rename = "health")]; a regression in either surface
    // would break the UI silently.
    assert!(
        mcp_obj.contains_key("health"),
        "MCP health output must include the `health` array key"
    );
    assert!(
        rest_obj.contains_key("health"),
        "REST health output must include the `health` array key"
    );
    assert!(mcp_obj["health"].is_array());
    assert!(rest_obj["health"].is_array());

    // system_health fields must be present on both.
    for required in ["indexes", "queues", "grayzone", "status"] {
        assert!(mcp_obj.contains_key(required), "MCP missing `{required}`");
        assert!(rest_obj.contains_key(required), "REST missing `{required}`");
    }
}

#[tokio::test]
async fn health_parity_topic_filter_applies_on_mcp() {
    let mcp = invoke_mcp(
        "health",
        serde_json::json!({ "topic": "nonexistent-topic-xxx" }),
    )
    .await;
    let mcp_obj = mcp.as_object().expect("object");
    assert!(
        mcp_obj["health"]
            .as_array()
            .map(|a| a.is_empty())
            .unwrap_or(false),
        "topic filter should return empty health array for nonexistent topic"
    );
}

#[tokio::test]
async fn health_parity_rest_query_filter_matches_mcp_args() {
    // REST query `?topic=X` should filter the same way as MCP args `{topic: X}`.
    let (rest_status, rest) = invoke_rest("health", "topic=nonexistent-topic-xxx").await;
    assert_eq!(rest_status, hyper::StatusCode::OK);
    assert!(
        rest["health"]
            .as_array()
            .map(|a| a.is_empty())
            .unwrap_or(false),
        "REST topic filter via query string should match MCP topic filter via args"
    );
}

#[tokio::test]
async fn adaptive_status_parity_mcp_and_rest_share_shape() {
    let mcp = invoke_mcp("adaptive_status", Value::Object(Default::default())).await;
    let (rest_status, rest) = invoke_rest("adaptive_status", "").await;

    assert_eq!(rest_status, hyper::StatusCode::OK);

    let mcp_obj = mcp.as_object().expect("adaptive_status MCP is an object");
    let rest_obj = rest.as_object().expect("adaptive_status REST is an object");

    let mcp_keys: std::collections::BTreeSet<_> = mcp_obj.keys().collect();
    let rest_keys: std::collections::BTreeSet<_> = rest_obj.keys().collect();
    assert_eq!(
        mcp_keys, rest_keys,
        "MCP and REST adaptive_status payloads must expose the same top-level keys"
    );

    // Contract fields the Adaptive GUI page reads. If any of these disappear,
    // the Neural Wiki dashboard panels go blank.
    for required in [
        "learned_alphas",
        "reranker_weights",
        "cluster_info",
        "tier_boundaries",
        "event_counts",
        "survival_curves",
        "dedup_thresholds",
        "cluster_profiles",
    ] {
        assert!(
            mcp_obj.contains_key(required),
            "MCP adaptive_status missing `{required}`"
        );
        assert!(
            rest_obj.contains_key(required),
            "REST adaptive_status missing `{required}`"
        );
    }
}

#[tokio::test]
async fn adaptive_status_parity_cli_dispatch_runs_to_completion() {
    let out = invoke_cli("adaptive_status", &[]).await;
    assert!(
        out.contains("learned_alphas") || out.contains("cluster_info"),
        "adaptive_status CLI output drifted: {out}"
    );
}

#[tokio::test]
async fn doctor_parity_rest_returns_checks_and_status() {
    // GET /api/doctor (no ?fix) should dispatch via inventory to the #[op]
    // doctor handler and return the same top-level shape the GUI expects.
    let (rest_status, rest) = invoke_rest("doctor", "").await;
    assert_eq!(rest_status, hyper::StatusCode::OK);
    let obj = rest.as_object().expect("doctor REST is an object");
    assert!(
        obj.contains_key("status") && obj.contains_key("checks"),
        "doctor output must expose `status` + `checks` fields, got: {rest:?}"
    );
    assert!(obj["checks"].is_array());
}

#[tokio::test]
async fn doctor_parity_cli_forces_fail_and_propagates_exit_code_1() {
    // Force a FAIL by pointing `database.path` at an un-creatable path.
    // `rein::doctor::run` pushes a Storage-category FAIL when `open_store`
    // errors, which bumps `DoctorReport::exit_code()` to 1. This proves the
    // end-to-end chain: op -> self.set_exit_code -> runtime channel -> CLI
    // dispatcher's std::process::exit call. The framework-level one-shot
    // contract is covered by `ops_runtime_exit_code_channel_round_trips`
    // in inventory_registration.rs.
    use std::sync::Arc;
    let mut config = rein::config::ReinConfig::default();
    config.database.path = "/nonexistent-parent-xxx-phase21/memories.db".to_string();
    let runtime = Arc::new(rein::ops::OpsRuntime::for_cli(Arc::new(config)));

    let entry = inventory::iter::<rein::ops::OpsCliEntry>()
        .find(|e| e.op_name == "doctor")
        .expect("doctor CLI registered");
    let matches = (entry.build_clap)()
        .try_get_matches_from(["doctor"])
        .expect("doctor args parse");
    let _out = (entry.invoke)(runtime.clone(), &matches)
        .await
        .expect("doctor invoke");

    assert_eq!(
        runtime.take_exit_code(),
        Some(1),
        "unreachable database path must produce FAIL -> exit_code 1"
    );
    assert!(
        runtime.take_exit_code().is_none(),
        "take_exit_code is one-shot"
    );
}

#[tokio::test]
async fn doctor_parity_cli_clean_run_leaves_exit_code_unset() {
    // Mirror case: a provisioned tempdir with no FAIL checks should not
    // touch the exit_code slot, so take_exit_code returns None and the
    // CLI dispatcher falls through to a normal exit 0.
    use std::sync::Arc;
    let (config, _tmp) = config_for_test();
    let runtime = Arc::new(rein::ops::OpsRuntime::for_cli(config));
    let entry = inventory::iter::<rein::ops::OpsCliEntry>()
        .find(|e| e.op_name == "doctor")
        .expect("doctor CLI registered");
    let matches = (entry.build_clap)()
        .try_get_matches_from(["doctor"])
        .expect("doctor args parse");
    let _out = (entry.invoke)(runtime.clone(), &matches)
        .await
        .expect("doctor invoke");
    assert_eq!(
        runtime.take_exit_code(),
        None,
        "healthy doctor run must leave the exit_code channel untouched"
    );
}

fn json_type_tag(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}
