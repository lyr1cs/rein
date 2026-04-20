use rein_macros::op;

struct OpsRuntime;

impl OpsRuntime {
    #[op(
        name = "x",
        category = "diagnostics",
        description = "unbalanced brace in path template",
        rest(method = "GET", path = "/api/foo/{id"),
    )]
    pub fn x(&self) -> &'static str { "" }
}

fn main() {}
