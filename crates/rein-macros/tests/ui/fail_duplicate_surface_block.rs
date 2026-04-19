use rein_macros::op;

struct OpsRuntime;

impl OpsRuntime {
    #[op(
        name = "x",
        category = "diagnostics",
        description = "duplicate rest block",
        rest(method = "GET", path = "/api/x"),
        rest(method = "POST", path = "/api/x"),
    )]
    pub fn x(&self) -> &'static str { "" }
}

fn main() {}
