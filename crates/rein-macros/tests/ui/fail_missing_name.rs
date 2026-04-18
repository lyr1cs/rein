use rein_macros::op;

struct OpsRuntime;

impl OpsRuntime {
    #[op(
        category = "metrics",
        description = "no name",
        rest(method = "GET", path = "/api/x"),
    )]
    pub fn x(&self) -> &'static str { "" }
}

fn main() {}
