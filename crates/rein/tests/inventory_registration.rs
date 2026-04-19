//! Verifies that #[op]-registered ops actually appear in the inventory.
//!
//! This is Phase 1.2 acceptance: the macro emission produces entries that the
//! `inventory` crate collects at startup. Without this test passing, the CLI /
//! MCP / REST adapters would iterate an empty registry.

use rein::ops::{OpsCliEntry, OpsMcpEntry, OpsMetadata, OpsRestEntry};

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
