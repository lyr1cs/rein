use rein_macros::op;

struct OpsRuntime;

impl OpsRuntime {
    #[op(
        name = "x",
        category = "diagnostics",
        description = "placeholder must occupy entire segment — prefix{id} not allowed",
        rest(method = "GET", path = "/api/foo/prefix{id}"),
    )]
    pub fn x(&self) -> &'static str { "" }
}

fn main() {}
