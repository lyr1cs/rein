//! Three surfaces declared. Validates parser handles cli + mcp + rest blocks.

use rein_macros::op;

struct OpsRuntime;

impl OpsRuntime {
    #[op(
        name = "stats",
        category = "memory",
        description = "Show store statistics",
        cli(name = "stats"),
        mcp(name = "rein_stats"),
        rest(method = "GET", path = "/api/stats"),
    )]
    pub fn stats(&self) -> &'static str { "" }
}

fn main() {}
