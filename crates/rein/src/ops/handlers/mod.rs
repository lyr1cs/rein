//! Per-category op handler modules.
//!
//! Each submodule contains `impl OpsRuntime { #[op] ... }` blocks for ops
//! in that category.

pub mod adaptive;
pub mod cold_archive;
pub mod concept_summary_feedback;
pub mod diagnostics;
pub mod judge;
pub mod knowledge;
pub mod maintenance;
pub mod memory;
pub mod session;

/// Test-only path-template ops. Gated behind the `test-support` feature so
/// they are absent from production builds. Integration tests activate this
/// feature via dev-dependencies in Cargo.toml.
#[cfg(feature = "test-support")]
pub mod test_path_template;
