// Allow clippy lints that are intentional design choices or low-value noise
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::assigning_clones)]
#![allow(clippy::doc_lazy_continuation)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::missing_transmute_annotations)]
#![allow(clippy::manual_strip)]
#![allow(clippy::manual_flatten)]
#![allow(clippy::redundant_slicing)]
#![allow(clippy::collapsible_match)]
#![allow(clippy::cloned_ref_to_slice_refs)]

// Allow `#[op]`-generated code to use `::rein::ops::*` paths uniformly —
// both inside this crate and in external trybuild tests that path-dep on rein.
extern crate self as rein;

/// Public URL of the canonical rein source repository.
///
/// Surfaced via `x-source-code` HTTP header on every REST response, in the
/// MCP `get_info()` instructions, and in the Neural Wiki GUI footer to
/// satisfy AGPL-3.0 §13 (network-use clause): users interacting with rein
/// over a network MUST have a clear path to the corresponding source.
///
/// Operators running modified forks SHOULD update this constant to point
/// at THEIR public source URL (the modified Corresponding Source) rather
/// than the upstream repo.
pub const SOURCE_URL: &str = "https://github.com/lyr1cs/rein";

/// SPDX identifier of the license rein is distributed under. Surfaced
/// alongside `SOURCE_URL` so clients see the legal terms without parsing
/// the LICENSE file.
pub const LICENSE_SPDX: &str = "AGPL-3.0-or-later";

pub mod compression;
pub mod config;
pub mod doctor;
pub mod embed;
pub mod eval;
pub mod extract;
pub mod init;
pub mod judge;
pub mod mcp;
pub mod ops;
pub mod proxy;
pub mod search;
pub mod service;
pub mod store;
pub mod sync;
pub mod types;
