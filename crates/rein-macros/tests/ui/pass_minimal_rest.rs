//! Minimal valid #[op]: REST-only, sync, no params, scalar output.
//! Phase 0b: macro emits the original method as no-op passthrough.
//! Phase 1: actual codegen wires CLI/MCP/REST adapters.

use rein_macros::op;

struct OpsRuntime;

impl OpsRuntime {
    #[op(
        name = "version",
        category = "metrics",
        description = "Show version",
        rest(method = "GET", path = "/api/version"),
    )]
    pub fn version(&self) -> &'static str {
        env!("CARGO_PKG_VERSION")
    }
}

fn main() {}
