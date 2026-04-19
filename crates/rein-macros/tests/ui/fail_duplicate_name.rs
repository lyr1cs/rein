use rein_macros::op;

struct OpsRuntime;

impl OpsRuntime {
    #[op(
        name = "first",
        name = "second",
        category = "diagnostics",
        description = "duplicate name key",
        cli(name = "x"),
    )]
    pub fn x(&self) -> &'static str { "" }
}

fn main() {}
