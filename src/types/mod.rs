pub mod error;
pub mod memory;
pub mod traits;

pub use error::*;
pub use memory::*;
pub use traits::*;

/// Maximum character length for memory summaries.
/// Used across all entry points (store, dedup merge, consolidation, sync).
pub const SUMMARY_MAX_CHARS: usize = 240;
