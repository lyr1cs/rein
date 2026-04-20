use rein_macros::op;

struct OpsRuntime;

impl OpsRuntime {
    #[op(
        name = "x",
        category = "diagnostics",
        description = "trailing slash in path template is not allowed",
        rest(method = "GET", path = "/api/foo/{id}/"),
    )]
    pub fn x(&self) -> &'static str { "" }
}

fn main() {}
