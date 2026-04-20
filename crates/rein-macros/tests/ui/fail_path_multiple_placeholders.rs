use rein_macros::op;

struct OpsRuntime;

impl OpsRuntime {
    #[op(
        name = "x",
        category = "diagnostics",
        description = "path has two placeholders — Phase 2.5 single-seg MVP rejects this",
        rest(method = "GET", path = "/api/foo/{a}/{b}"),
    )]
    pub fn x(&self) -> &'static str { "" }
}

fn main() {}
