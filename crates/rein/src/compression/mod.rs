//! Lossless compression contract for resummerize (v0.23).
//!
//! This module provides the seven invariants that any proposed resummerize
//! output must satisfy before the caller is allowed to persist it. All
//! invariants are pure functions over `(ContractInput, &str)` and never make
//! LLM calls or mutate state — they act as a guardrail, not a summarizer.
//!
//! See `contract` for the public API and documented limitations.

pub mod contract;

pub use contract::{check_all, check_each, ContractInput, EvidenceEntry, Violation};
