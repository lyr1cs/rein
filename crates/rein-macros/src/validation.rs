//! Compile-time validation for parsed `#[op]` attributes.
//!
//! Spec §4.3 lists the rules:
//! - At least one surface (cli/mcp/rest) declared
//! - `category` must be in the allowlist
//! - `mcp.name` must start with `rein_`
//! - `rest.path` must start with `/api/`
//! - `kind = "stream"` is reserved but not implemented in v0.21

use proc_macro2::Span;

use crate::op_attr::OpAttr;

const ALLOWED_CATEGORIES: &[&str] = &[
    "server",
    "memory",
    "ingest",
    "diagnostics",
    "knowledge",
    "maintenance",
    "integration",
    "index",
    "adaptive",
    "worker",
    "hooks",
    "service",
    "session",
    "metrics",
    "health",
    "artifacts",
    "timeline",
];

pub fn validate(attr: &OpAttr) -> syn::Result<()> {
    if attr.cli.is_none() && attr.mcp.is_none() && attr.rest.is_none() {
        return Err(syn::Error::new(
            Span::call_site(),
            "#[op] must declare at least one surface (cli, mcp, or rest)",
        ));
    }

    if !ALLOWED_CATEGORIES.contains(&attr.category.as_str()) {
        return Err(syn::Error::new(
            Span::call_site(),
            format!(
                "#[op] category '{}' not in allowlist; allowed: {:?}",
                attr.category, ALLOWED_CATEGORIES
            ),
        ));
    }

    if let Some(mcp) = &attr.mcp {
        if !mcp.name.starts_with("rein_") {
            return Err(syn::Error::new(
                Span::call_site(),
                format!("mcp.name must start with 'rein_', got '{}'", mcp.name),
            ));
        }
    }

    if let Some(rest) = &attr.rest {
        if !rest.path.starts_with("/api/") {
            return Err(syn::Error::new(
                Span::call_site(),
                format!("rest.path must start with '/api/', got '{}'", rest.path),
            ));
        }
    }

    if attr.kind != "unary" {
        return Err(syn::Error::new(
            Span::call_site(),
            format!(
                "#[op] kind '{}' not supported in v0.21 — only 'unary' implemented; \
                 see backlog #C2/#C4 for streaming op design",
                attr.kind
            ),
        ));
    }

    if !matches!(
        attr.auth.as_str(),
        "public" | "mutation_marker" | "read_token"
    ) {
        return Err(syn::Error::new(
            Span::call_site(),
            format!(
                "#[op] auth '{}' not supported — must be one of: public, mutation_marker, read_token",
                attr.auth
            ),
        ));
    }

    Ok(())
}
