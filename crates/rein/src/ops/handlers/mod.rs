//! Per-category op handler modules. Populated incrementally in Phase 1/2.
//!
//! Each submodule contains `impl OpsRuntime { #[op] ... }` blocks for ops
//! in that category. Phase 1 adds `diagnostics` (stats + health). Phase 2
//! migrates the remaining categories one file at a time.

pub mod adaptive;
pub mod diagnostics;
pub mod maintenance;
pub mod session;

// Test-only path-template op (T5 Phase 2.5). Registered unconditionally so
// integration tests in tests/phase_2_5_path_template.rs can find it in
// inventory via `inventory::iter`. Prefixed `__test_` to signal test-only.
pub mod test_path_template;
