//! Guards the MCP tool surface against duplicate or missing names.
//!
//! Phase 2.3 H1 originally added this file to detect drift between a
//! hand-maintained `MCP_OPERATIONS` list and the real tool surface.
//! Phase 3 removed that registry — inventory is now authoritative — so the
//! drift mode it guarded against no longer exists. The test was rewritten
//! to catch the Phase 2.3 M4 concern instead: duplicate MCP tool names
//! between inventory entries and (historically) the legacy `#[tool]` set.
//! Today the legacy set is empty, so the check reduces to "inventory
//! MCP names are unique", which is still a useful invariant.

use std::collections::HashSet;

#[test]
fn mcp_tool_names_are_unique_across_inventory() {
    let mut names: Vec<&'static str> = inventory::iter::<rein::ops::OpsMcpEntry>()
        .map(|e| e.mcp_name)
        .collect();
    let unique: HashSet<&'static str> = names.iter().copied().collect();
    if names.len() != unique.len() {
        names.sort_unstable();
        let mut dups: Vec<&'static str> = Vec::new();
        for window in names.windows(2) {
            if window[0] == window[1] && !dups.contains(&window[0]) {
                dups.push(window[0]);
            }
        }
        panic!("duplicate MCP tool names across inventory: {dups:?}");
    }
}

#[test]
fn legacy_tool_handlers_are_fully_migrated() {
    // Post-Phase-2.6, no `#[tool(...)]` handlers should remain in
    // mcp/server.rs — every MCP tool is now served through inventory.
    // If a new `#[tool(name = "rein_...")]` is added, it should be
    // authored as `#[op(... mcp(name = "..."))]` instead.
    let server_rs = include_str!("../src/mcp/server.rs");
    let derived: HashSet<&str> = extract_tool_names_from_server_rs(server_rs);
    assert!(
        derived.is_empty(),
        "expected zero legacy #[tool] handlers in mcp/server.rs post-Phase-2.6, found: {derived:?}"
    );
}

/// Naive scanner: extract all `name = "rein_..."` values from tool-router
/// `#[tool(...)]` attribute blocks in `mcp/server.rs`.
fn extract_tool_names_from_server_rs(src: &str) -> HashSet<&str> {
    let prefix = "name = \"";
    let mut names = HashSet::new();
    for line in src.lines() {
        if let Some(pos) = line.find(prefix) {
            let after = &line[pos + prefix.len()..];
            if after.starts_with("rein_") {
                if let Some(end) = after.find('"') {
                    names.insert(&after[..end]);
                }
            }
        }
    }
    names
}
