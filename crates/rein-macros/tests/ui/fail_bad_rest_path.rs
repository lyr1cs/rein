use rein_macros::op;

struct OpsRuntime;

impl OpsRuntime {
    #[op(
        name = "x",
        category = "memory",
        description = "rest.path must start with /api/",
        rest(method = "GET", path = "/wrong/x"),
    )]
    pub fn x(&self) -> &'static str { "" }
}

fn main() {}
