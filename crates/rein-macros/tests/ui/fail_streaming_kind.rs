use rein_macros::op;

struct OpsRuntime;

impl OpsRuntime {
    #[op(
        name = "x",
        category = "memory",
        description = "kind=stream reserved but not implemented in v0.21",
        kind = "stream",
        rest(method = "GET", path = "/api/x"),
    )]
    pub fn x(&self) -> &'static str { "" }
}

fn main() {}
