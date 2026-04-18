//! Procedural macros for rein's operation registry.
//!
//! See `docs/superpowers/specs/2026-04-18-A1-operation-registry-codegen-design.md`
//! for the full design.
//!
//! Phase 0b status: `#[op]` parses + validates the attribute, but emits the
//! original method as a no-op passthrough. Full codegen (CLI/MCP/REST inventory
//! entries + rmcp tool wrapper + metadata) is implemented in Phase 1 Task 1.2.

use proc_macro::TokenStream;

mod op_attr;
mod validation;

/// `#[op(...)]` attribute macro applied to methods on `OpsRuntime`.
///
/// Phase 0b: validates the attribute syntax and emits the method unchanged.
/// Phase 1+: emits CLI/MCP/REST adapter code via inventory + rmcp tool wrapper.
#[proc_macro_attribute]
pub fn op(attr: TokenStream, item: TokenStream) -> TokenStream {
    op_attr::expand(attr.into(), item.into())
        .unwrap_or_else(|err| err.to_compile_error())
        .into()
}
