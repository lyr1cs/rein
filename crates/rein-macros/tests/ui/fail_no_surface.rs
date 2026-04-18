use rein_macros::op;

struct OpsRuntime;

impl OpsRuntime {
    #[op(
        name = "x",
        category = "metrics",
        description = "no surface declared",
    )]
    pub fn x(&self) -> &'static str { "" }
}

fn main() {}
