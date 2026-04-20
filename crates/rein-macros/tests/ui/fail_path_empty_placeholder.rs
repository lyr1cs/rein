use rein_macros::op;

struct OpsRuntime;

impl OpsRuntime {
    #[op(
        name = "x",
        category = "diagnostics",
        description = "empty placeholder {} is not allowed",
        rest(method = "GET", path = "/api/foo/{}"),
    )]
    pub fn x(&self) -> &'static str { "" }
}

fn main() {}
