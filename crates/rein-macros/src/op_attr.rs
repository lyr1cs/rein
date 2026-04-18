//! `#[op(...)]` attribute parsing.
//!
//! Spec §4.1, §4.3, §4.4 define the attribute syntax. This module parses the
//! attribute into an `OpAttr` struct, runs `validation::validate`, then (in
//! Phase 0b) emits the original method unchanged.

use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::{
    parse::{Parse, ParseStream},
    parse2,
    punctuated::Punctuated,
    spanned::Spanned,
    Expr, ExprLit, ImplItemFn, Lit, Meta, Token,
};

use crate::validation;

/// Parsed `#[op(...)]` attribute. Fields populated in Phase 0b but consumed
/// only when Phase 1 fills in the codegen — silenced until then.
#[allow(dead_code)]
#[derive(Debug)]
pub struct OpAttr {
    pub name: String,
    pub category: String,
    pub description: String,
    pub kind: String,
    pub mutating: bool,
    pub cli: Option<CliBlock>,
    pub mcp: Option<McpBlock>,
    pub rest: Option<RestBlock>,
}

#[allow(dead_code)]
#[derive(Debug, Default)]
pub struct CliBlock {
    pub name: Option<String>,
    pub positional: Vec<String>,
    pub aliases: Vec<String>,
    pub hidden: bool,
    pub parent: Option<String>,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct McpBlock {
    pub name: String,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct RestBlock {
    pub method: String,
    pub path: String,
    pub path_params: Vec<String>,
}

/// Wrapper struct so we can use `Punctuated::parse_terminated` via `parse2`.
struct AttrInput {
    metas: Punctuated<Meta, Token![,]>,
}

impl Parse for AttrInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(Self {
            metas: Punctuated::parse_terminated(input)?,
        })
    }
}

/// Top-level expansion entry point — called from `#[op]` proc macro.
pub fn expand(attr: TokenStream, item: TokenStream) -> syn::Result<TokenStream> {
    let parsed_attr = parse_op_attr(attr)?;
    let parsed_fn: ImplItemFn = parse2(item)?;

    validation::validate(&parsed_attr)?;

    // Phase 0b: emit the original method unchanged. Full expansion in Phase 1.
    Ok(quote! { #parsed_fn })
}

fn parse_op_attr(attr: TokenStream) -> syn::Result<OpAttr> {
    let input: AttrInput = parse2(attr)?;

    let mut name: Option<String> = None;
    let mut category: Option<String> = None;
    let mut description: Option<String> = None;
    let mut kind = "unary".to_string();
    let mut mutating = false;
    let mut cli: Option<CliBlock> = None;
    let mut mcp: Option<McpBlock> = None;
    let mut rest: Option<RestBlock> = None;

    for meta in input.metas {
        match meta {
            Meta::NameValue(nv) => {
                let key = ident_string(&nv.path)?;
                match key.as_str() {
                    "name" => name = Some(extract_string_lit(&nv.value, "name")?),
                    "category" => category = Some(extract_string_lit(&nv.value, "category")?),
                    "description" => {
                        description = Some(extract_string_lit(&nv.value, "description")?)
                    }
                    "kind" => kind = extract_string_lit(&nv.value, "kind")?,
                    "mutating" => mutating = extract_bool_lit(&nv.value, "mutating")?,
                    other => {
                        return Err(syn::Error::new(
                            nv.path.span(),
                            format!("unknown #[op] key: '{other}'"),
                        ))
                    }
                }
            }
            Meta::List(list) => {
                let key = ident_string(&list.path)?;
                let inner = list.tokens.clone();
                match key.as_str() {
                    "cli" => cli = Some(parse_cli_block(inner)?),
                    "mcp" => mcp = Some(parse_mcp_block(inner)?),
                    "rest" => rest = Some(parse_rest_block(inner)?),
                    other => {
                        return Err(syn::Error::new(
                            list.path.span(),
                            format!("unknown #[op] block: '{other}'"),
                        ))
                    }
                }
            }
            other => {
                return Err(syn::Error::new(
                    other.span(),
                    "unsupported #[op] attribute form (use `key = value` or `block { ... }`)",
                ))
            }
        }
    }

    Ok(OpAttr {
        name: name
            .ok_or_else(|| syn::Error::new(Span::call_site(), "missing required #[op] key 'name'"))?,
        category: category.ok_or_else(|| {
            syn::Error::new(Span::call_site(), "missing required #[op] key 'category'")
        })?,
        description: description.ok_or_else(|| {
            syn::Error::new(Span::call_site(), "missing required #[op] key 'description'")
        })?,
        kind,
        mutating,
        cli,
        mcp,
        rest,
    })
}

fn parse_cli_block(tokens: TokenStream) -> syn::Result<CliBlock> {
    let input: AttrInput = parse2(tokens)?;
    let mut block = CliBlock::default();
    for meta in input.metas {
        match meta {
            Meta::NameValue(nv) => {
                let key = ident_string(&nv.path)?;
                match key.as_str() {
                    "name" => block.name = Some(extract_string_lit(&nv.value, "cli.name")?),
                    "hidden" => block.hidden = extract_bool_lit(&nv.value, "cli.hidden")?,
                    "parent" => block.parent = Some(extract_string_lit(&nv.value, "cli.parent")?),
                    "positional" => {
                        block.positional = extract_string_array(&nv.value, "cli.positional")?
                    }
                    "aliases" => {
                        block.aliases = extract_string_array(&nv.value, "cli.aliases")?
                    }
                    other => {
                        return Err(syn::Error::new(
                            nv.path.span(),
                            format!("unknown cli block key: '{other}'"),
                        ))
                    }
                }
            }
            other => {
                return Err(syn::Error::new(
                    other.span(),
                    "cli block expects `key = value` entries",
                ))
            }
        }
    }
    Ok(block)
}

fn parse_mcp_block(tokens: TokenStream) -> syn::Result<McpBlock> {
    let input: AttrInput = parse2(tokens)?;
    let mut name: Option<String> = None;
    for meta in input.metas {
        match meta {
            Meta::NameValue(nv) => {
                let key = ident_string(&nv.path)?;
                match key.as_str() {
                    "name" => name = Some(extract_string_lit(&nv.value, "mcp.name")?),
                    other => {
                        return Err(syn::Error::new(
                            nv.path.span(),
                            format!("unknown mcp block key: '{other}'"),
                        ))
                    }
                }
            }
            other => {
                return Err(syn::Error::new(
                    other.span(),
                    "mcp block expects `key = value` entries",
                ))
            }
        }
    }
    Ok(McpBlock {
        name: name.ok_or_else(|| {
            syn::Error::new(Span::call_site(), "mcp block missing required key 'name'")
        })?,
    })
}

fn parse_rest_block(tokens: TokenStream) -> syn::Result<RestBlock> {
    let input: AttrInput = parse2(tokens)?;
    let mut method: Option<String> = None;
    let mut path: Option<String> = None;
    let mut path_params: Vec<String> = Vec::new();
    for meta in input.metas {
        match meta {
            Meta::NameValue(nv) => {
                let key = ident_string(&nv.path)?;
                match key.as_str() {
                    "method" => method = Some(extract_string_lit(&nv.value, "rest.method")?),
                    "path" => path = Some(extract_string_lit(&nv.value, "rest.path")?),
                    "path_params" => {
                        path_params = extract_string_array(&nv.value, "rest.path_params")?
                    }
                    other => {
                        return Err(syn::Error::new(
                            nv.path.span(),
                            format!("unknown rest block key: '{other}'"),
                        ))
                    }
                }
            }
            other => {
                return Err(syn::Error::new(
                    other.span(),
                    "rest block expects `key = value` entries",
                ))
            }
        }
    }
    Ok(RestBlock {
        method: method.ok_or_else(|| {
            syn::Error::new(Span::call_site(), "rest block missing required key 'method'")
        })?,
        path: path.ok_or_else(|| {
            syn::Error::new(Span::call_site(), "rest block missing required key 'path'")
        })?,
        path_params,
    })
}

// --- helpers ---

fn ident_string(path: &syn::Path) -> syn::Result<String> {
    path.get_ident()
        .map(|i| i.to_string())
        .ok_or_else(|| syn::Error::new(path.span(), "expected simple identifier"))
}

fn extract_string_lit(expr: &Expr, key: &str) -> syn::Result<String> {
    if let Expr::Lit(ExprLit {
        lit: Lit::Str(s), ..
    }) = expr
    {
        Ok(s.value())
    } else {
        Err(syn::Error::new(
            expr.span(),
            format!("'{key}' expects a string literal"),
        ))
    }
}

fn extract_bool_lit(expr: &Expr, key: &str) -> syn::Result<bool> {
    if let Expr::Lit(ExprLit {
        lit: Lit::Bool(b), ..
    }) = expr
    {
        Ok(b.value())
    } else {
        Err(syn::Error::new(
            expr.span(),
            format!("'{key}' expects a bool literal"),
        ))
    }
}

fn extract_string_array(expr: &Expr, key: &str) -> syn::Result<Vec<String>> {
    if let Expr::Array(arr) = expr {
        arr.elems
            .iter()
            .map(|e| extract_string_lit(e, key))
            .collect()
    } else {
        Err(syn::Error::new(
            expr.span(),
            format!("'{key}' expects an array of string literals like [\"a\", \"b\"]"),
        ))
    }
}
