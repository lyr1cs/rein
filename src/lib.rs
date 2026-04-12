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

pub mod config;
pub mod doctor;
pub mod embed;
pub mod extract;
pub mod init;
pub mod mcp;
pub mod ops;
pub mod proxy;
pub mod search;
pub mod service;
pub mod store;
pub mod sync;
pub mod types;
