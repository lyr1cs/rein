use rein_macros::op;

struct OpsRuntime;

impl OpsRuntime {
    #[op(
        name = "x",
        category = "diagnostics",
        description = "bad auth value",
        auth = "admin_only",
        cli(name = "x"),
    )]
    pub fn x(&self) -> &'static str { "" }
}

fn main() {}
