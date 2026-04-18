use rein_macros::op;

struct OpsRuntime;

impl OpsRuntime {
    #[op(
        name = "x",
        category = "memory",
        description = "mcp.name must start with rein_",
        mcp(name = "x"),
    )]
    pub fn x(&self) -> &'static str { "" }
}

fn main() {}
