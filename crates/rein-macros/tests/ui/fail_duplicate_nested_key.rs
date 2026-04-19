use rein_macros::op;

struct OpsRuntime;

impl OpsRuntime {
    #[op(
        name = "x",
        category = "diagnostics",
        description = "duplicate rest.path inside the same block",
        rest(method = "GET", path = "/api/x", path = "/api/y"),
    )]
    pub fn x(&self) -> &'static str { "" }
}

fn main() {}
