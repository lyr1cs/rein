//! Per-category op handler modules. Populated incrementally in Phase 1/2.
//!
//! Each submodule contains `impl OpsRuntime { #[op] ... }` blocks for ops
//! in that category. Phase 1 adds `diagnostics` (stats + health). Phase 2
//! migrates the remaining categories one file at a time.

pub mod adaptive;
pub mod diagnostics;
pub mod maintenance;
pub mod session;
