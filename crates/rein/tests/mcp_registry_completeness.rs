//! Guards the MCP_OPERATIONS name set against drift from the actual tool surface.
//!
//! Phase 2.3 discovered that MCP_OPERATIONS could drift from the real tool surface
//! silently — the numeric drift check in doctor.rs asserts count equality only, not
//! set equality. Phantom-vs-missing pairs cancel arithmetically so the registry could
//! stay numerically balanced while silently diverging from real tools.
//!
//! H1 root cause: rein_feedback was unregistered in MCP_OPERATIONS while rein_upgrade
//! (a phantom with no backing handler) was registered. Net count stayed at 31 so the
//! count-only assertion in registry_counts_match_expected_values passed.
//!
//! This test exercises set equality, not count equality, for both directions:
//!   1. Every inventory OpsMcpEntry name is in MCP_OPERATIONS.
//!   2. Every legacy #[tool] name in mcp/server.rs is in MCP_OPERATIONS.
//!   3. Every MCP_OPERATIONS entry is backed by (1) or (2) — catches phantoms.

use std::collections::HashSet;

#[test]
fn mcp_operations_name_set_matches_actual_tool_surface() {
    let registry_names: HashSet<&'static str> = rein::ops::mcp_operations()
        .iter()
        .map(|op| op.name)
        .collect();

    // 1. Every inventory OpsMcpEntry name must appear in MCP_OPERATIONS.
    let inventory_names: HashSet<&'static str> = inventory::iter::<rein::ops::OpsMcpEntry>()
        .map(|e| e.mcp_name)
        .collect();
    let mut inventory_missing: Vec<&str> = inventory_names
        .difference(&registry_names)
        .copied()
        .collect();
    inventory_missing.sort_unstable();
    assert!(
        inventory_missing.is_empty(),
        "inventory MCP ops missing from MCP_OPERATIONS registry: {inventory_missing:?}"
    );

    // 2. Every legacy #[tool(name = "rein_...")] handler in mcp/server.rs must
    //    appear in MCP_OPERATIONS.
    //
    //    The scanner only runs on server.rs (not ops/handlers/) so there is no
    //    risk of double-counting #[op] mcp(name=...) attributes from other files.
    let server_rs = include_str!("../src/mcp/server.rs");
    let derived_names = extract_tool_names_from_server_rs(server_rs);
    let mut derived_missing: Vec<&&str> = derived_names.difference(&registry_names).collect();
    derived_missing.sort_unstable();
    assert!(
        derived_missing.is_empty(),
        "legacy #[tool] handlers in mcp/server.rs missing from MCP_OPERATIONS registry: {derived_missing:?}"
    );

    // 3. Every registry entry must be either an inventory op OR a legacy #[tool].
    //    Catches phantom registry entries — names registered but never actually served.
    let all_real: HashSet<&str> = inventory_names.union(&derived_names).copied().collect();
    let mut registry_phantoms: Vec<&str> = registry_names
        .difference(&all_real)
        .copied()
        .collect();
    registry_phantoms.sort_unstable();
    assert!(
        registry_phantoms.is_empty(),
        "MCP_OPERATIONS entries with no backing handler (phantoms): {registry_phantoms:?}"
    );
}

/// Naive scanner: extract all `name = "rein_..."` values from tool-router
/// `#[tool(...)]` attribute blocks in `mcp/server.rs`.
///
/// The pattern `name = "rein_` is distinctive enough to be safe in server.rs —
/// there are no `#[op]` mcp(name=...) attrs there, and no other use of the
/// pattern that could produce false positives.
///
/// Example match: `name = "rein_recall",` → yields `"rein_recall"`.
fn extract_tool_names_from_server_rs(src: &str) -> HashSet<&str> {
    // needle ends just before the opening quote of the value
    let prefix = "name = \"";
    let mut names = HashSet::new();
    for line in src.lines() {
        if let Some(pos) = line.find(prefix) {
            // after is the string starting at the first char of the value (after the `"`)
            let after = &line[pos + prefix.len()..];
            // only pick up rein_ tool names
            if after.starts_with("rein_") {
                if let Some(end) = after.find('"') {
                    names.insert(&after[..end]);
                }
            }
        }
    }
    names
}
