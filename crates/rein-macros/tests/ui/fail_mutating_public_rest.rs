use rein_macros::op;

struct OpsRuntime;

impl OpsRuntime {
    // Regression: mutating = true with a REST write method + default auth = "public"
    // must fail to compile. Forgetting `auth = "mutation_marker"` previously
    // registered an unauthenticated mutating endpoint silently.
    #[op(
        name = "wipe_memories",
        category = "maintenance",
        description = "destructive op missing mutation_marker",
        mutating = true,
        cli(name = "wipe"),
        rest(method = "POST", path = "/api/wipe"),
    )]
    pub fn wipe(&self) -> &'static str { "" }
}

fn main() {}
