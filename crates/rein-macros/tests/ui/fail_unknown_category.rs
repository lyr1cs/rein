use rein_macros::op;

struct OpsRuntime;

impl OpsRuntime {
    #[op(
        name = "x",
        category = "frobnicate",
        description = "category must be in allowlist",
        rest(method = "GET", path = "/api/x"),
    )]
    pub fn x(&self) -> &'static str { "" }
}

fn main() {}
