//! `#[op(...)]` attribute: parse + emit surface-aware inventory registrations.
//!
//! Phase 0b parsed the attribute; Phase 1.2 adds full emission. For each `#[op]`
//! we emit the original method unchanged, plus a hidden associated const
//! `__OP_INV_<name>: () = { ... }` whose block body declares per-surface invoke
//! helpers and `inventory::submit!` entries for CLI/MCP/REST + metadata.
//!
//! Emitted paths all go through `::rein::ops::*`. `crates/rein/src/lib.rs`
//! declares `extern crate self as rein;` so the paths resolve both inside the
//! rein crate and in external test crates that path-dep on rein.

use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use syn::{
    parse::{Parse, ParseStream},
    parse2,
    punctuated::Punctuated,
    spanned::Spanned,
    Expr, ExprLit, FnArg, ImplItemFn, Lit, Meta, Token, Type,
};

use crate::validation;

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

#[derive(Debug, Default)]
pub struct CliBlock {
    pub name: Option<String>,
    pub positional: Vec<String>,
    pub aliases: Vec<String>,
    pub hidden: bool,
    pub parent: Option<String>,
}

#[derive(Debug)]
pub struct McpBlock {
    pub name: String,
}

#[derive(Debug)]
pub struct RestBlock {
    pub method: String,
    pub path: String,
    pub path_params: Vec<String>,
}

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

/// Analyzed method signature: what the macro needs for emission.
struct FnInfo {
    name: syn::Ident,
    is_async: bool,
    params_ty: Option<Type>,
}

pub fn expand(attr: TokenStream, item: TokenStream) -> syn::Result<TokenStream> {
    let parsed_attr = parse_op_attr(attr)?;
    let parsed_fn: ImplItemFn = parse2(item)?;
    validation::validate(&parsed_attr)?;

    let fn_info = analyze_fn(&parsed_fn.sig)?;
    let emission = emit_inventory_block(&parsed_attr, &fn_info)?;

    Ok(quote! {
        #parsed_fn
        #emission
    })
}

fn analyze_fn(sig: &syn::Signature) -> syn::Result<FnInfo> {
    let name = sig.ident.clone();
    let is_async = sig.asyncness.is_some();

    let user_args: Vec<&FnArg> = sig
        .inputs
        .iter()
        .filter(|arg| matches!(arg, FnArg::Typed(_)))
        .collect();

    let params_ty = match user_args.len() {
        0 => None,
        1 => {
            if let FnArg::Typed(pt) = user_args[0] {
                Some((*pt.ty).clone())
            } else {
                None
            }
        }
        _ => {
            return Err(syn::Error::new(
                sig.span(),
                "#[op] methods must take 0 or 1 user argument (receiver + optional params struct)",
            ));
        }
    };

    Ok(FnInfo {
        name,
        is_async,
        params_ty,
    })
}

fn emit_inventory_block(attr: &OpAttr, fi: &FnInfo) -> syn::Result<TokenStream> {
    let fn_name = &fi.name;
    let const_name = format_ident!("__OP_INV_{}", fn_name);
    let op_name = &attr.name;
    let category = &attr.category;
    let description = &attr.description;
    let mutating = attr.mutating;

    let params_schema_fn = emit_params_schema_fn(fi.params_ty.as_ref());

    let cli_block = attr
        .cli
        .as_ref()
        .map(|cli| emit_cli_block(cli, op_name, fn_name, fi));
    let mcp_block = attr
        .mcp
        .as_ref()
        .map(|mcp| emit_mcp_block(mcp, op_name, description, fn_name, fi));
    let rest_block = attr
        .rest
        .as_ref()
        .map(|rest| emit_rest_block(rest, op_name, fn_name, fi));

    let cli_visible = attr.cli.is_some();
    let mcp_visible = attr.mcp.is_some();
    let rest_visible = attr.rest.is_some();

    let mcp_name_tokens = match &attr.mcp {
        Some(m) => {
            let n = &m.name;
            quote! { ::std::option::Option::Some(#n) }
        }
        None => quote! { ::std::option::Option::None },
    };
    let rest_method_tokens = match &attr.rest {
        Some(r) => {
            let method_ident = method_ident(&r.method)?;
            quote! { ::std::option::Option::Some(::hyper::Method::#method_ident) }
        }
        None => quote! { ::std::option::Option::None },
    };
    let rest_path_tokens = match &attr.rest {
        Some(r) => {
            let p = &r.path;
            quote! { ::std::option::Option::Some(#p) }
        }
        None => quote! { ::std::option::Option::None },
    };

    Ok(quote! {
        #[doc(hidden)]
        #[allow(non_upper_case_globals, dead_code)]
        const #const_name: () = {
            #params_schema_fn
            #cli_block
            #mcp_block
            #rest_block

            ::inventory::submit! {
                ::rein::ops::OpsMetadata {
                    name: #op_name,
                    category: #category,
                    description: #description,
                    kind: ::rein::ops::OpKind::Unary,
                    mutating: #mutating,
                    cli_visible: #cli_visible,
                    mcp_visible: #mcp_visible,
                    rest_visible: #rest_visible,
                    mcp_name: #mcp_name_tokens,
                    rest_method: #rest_method_tokens,
                    rest_path: #rest_path_tokens,
                    params_schema: __op_params_schema,
                }
            }
        };
    })
}

fn emit_params_schema_fn(params_ty: Option<&Type>) -> TokenStream {
    match params_ty {
        Some(ty) => quote! {
            fn __op_params_schema() -> ::schemars::Schema {
                ::schemars::schema_for!(#ty)
            }
        },
        None => quote! {
            fn __op_params_schema() -> ::schemars::Schema {
                let mut __map = ::serde_json::Map::new();
                __map.insert(
                    ::std::string::String::from("type"),
                    ::serde_json::Value::String(::std::string::String::from("object")),
                );
                __map.insert(
                    ::std::string::String::from("properties"),
                    ::serde_json::Value::Object(::serde_json::Map::new()),
                );
                ::schemars::Schema::from(__map)
            }
        },
    }
}

/// `OpsRuntime::foo(&runtime)` or `OpsRuntime::foo(&runtime, params)` plus `.await` if async.
fn emit_call(fn_name: &syn::Ident, has_params: bool, is_async: bool) -> TokenStream {
    let args = if has_params {
        quote! { &runtime, params }
    } else {
        quote! { &runtime }
    };
    let call = quote! { ::rein::ops::OpsRuntime::#fn_name(#args) };
    if is_async {
        quote! { #call.await }
    } else {
        quote! { #call }
    }
}

fn emit_cli_block(
    cli: &CliBlock,
    op_name: &str,
    fn_name: &syn::Ident,
    fi: &FnInfo,
) -> TokenStream {
    let cli_name = cli.name.clone().unwrap_or_else(|| op_name.to_string());
    let aliases = &cli.aliases;
    let hidden = cli.hidden;
    let parent_tokens = match &cli.parent {
        Some(p) => quote! { ::std::option::Option::Some(#p) },
        None => quote! { ::std::option::Option::None },
    };

    let (build_body, pre_extract) = match &fi.params_ty {
        Some(ty) => (
            quote! {
                let cmd = ::clap::Command::new(#cli_name);
                <#ty as ::clap::Args>::augment_args(cmd)
            },
            // Extract params synchronously (before async block) so `_matches`
            // borrow doesn't leak into the returned `'static` future.
            quote! {
                let params_result = <#ty as ::clap::FromArgMatches>::from_arg_matches(_matches);
            },
        ),
        None => (quote! { ::clap::Command::new(#cli_name) }, quote! {}),
    };

    let async_prep = match &fi.params_ty {
        Some(_) => quote! {
            let params = params_result.map_err(|e| {
                ::rein::types::ReinError::Config(format!("cli arg parse error: {e}"))
            })?;
        },
        None => quote! {},
    };

    let call_expr = emit_call(fn_name, fi.params_ty.is_some(), fi.is_async);

    quote! {
        fn __op_cli_build() -> ::clap::Command {
            #build_body
        }

        fn __op_cli_invoke(
            runtime: ::std::sync::Arc<::rein::ops::OpsRuntime>,
            _matches: &::clap::ArgMatches,
        ) -> ::std::pin::Pin<
            ::std::boxed::Box<
                dyn ::std::future::Future<
                    Output = ::rein::types::ReinResult<::std::string::String>,
                > + ::std::marker::Send,
            >,
        > {
            #pre_extract
            ::std::boxed::Box::pin(async move {
                #async_prep
                let out = #call_expr?;
                ::std::result::Result::Ok(
                    <_ as ::rein::ops::IntoCliText>::to_cli_text(&out),
                )
            })
        }

        ::inventory::submit! {
            ::rein::ops::OpsCliEntry {
                name: #cli_name,
                op_name: #op_name,
                parent: #parent_tokens,
                aliases: &[ #( #aliases ),* ],
                hidden: #hidden,
                build_clap: __op_cli_build,
                invoke: __op_cli_invoke,
            }
        }
    }
}

fn emit_mcp_block(
    mcp: &McpBlock,
    op_name: &str,
    description: &str,
    fn_name: &syn::Ident,
    fi: &FnInfo,
) -> TokenStream {
    let mcp_name = &mcp.name;
    let call_expr = emit_call(fn_name, fi.params_ty.is_some(), fi.is_async);

    let prep = match &fi.params_ty {
        Some(ty) => quote! {
            let params: #ty = ::serde_json::from_value(_params_json)?;
        },
        None => quote! {},
    };

    let schema_fn = match &fi.params_ty {
        Some(ty) => quote! {
            fn __op_mcp_schema() -> ::schemars::Schema {
                ::schemars::schema_for!(#ty)
            }
        },
        None => quote! {
            fn __op_mcp_schema() -> ::schemars::Schema {
                let mut __map = ::serde_json::Map::new();
                __map.insert(
                    ::std::string::String::from("type"),
                    ::serde_json::Value::String(::std::string::String::from("object")),
                );
                __map.insert(
                    ::std::string::String::from("properties"),
                    ::serde_json::Value::Object(::serde_json::Map::new()),
                );
                ::schemars::Schema::from(__map)
            }
        },
    };

    quote! {
        #schema_fn

        fn __op_mcp_invoke(
            runtime: ::std::sync::Arc<::rein::ops::OpsRuntime>,
            _params_json: ::serde_json::Value,
        ) -> ::std::pin::Pin<
            ::std::boxed::Box<
                dyn ::std::future::Future<
                    Output = ::rein::types::ReinResult<::std::string::String>,
                > + ::std::marker::Send,
            >,
        > {
            ::std::boxed::Box::pin(async move {
                #prep
                let out = #call_expr?;
                let json = ::serde_json::to_string(&out)?;
                ::std::result::Result::Ok(json)
            })
        }

        ::inventory::submit! {
            ::rein::ops::OpsMcpEntry {
                op_name: #op_name,
                mcp_name: #mcp_name,
                description: #description,
                input_schema: __op_mcp_schema,
                invoke: __op_mcp_invoke,
            }
        }
    }
}

fn emit_rest_block(
    rest: &RestBlock,
    op_name: &str,
    fn_name: &syn::Ident,
    fi: &FnInfo,
) -> TokenStream {
    let method_ident = match method_ident(&rest.method) {
        Ok(id) => id,
        Err(e) => return e.to_compile_error(),
    };
    let path = &rest.path;
    let path_params = &rest.path_params;
    let call_expr = emit_call(fn_name, fi.params_ty.is_some(), fi.is_async);

    let prep = match &fi.params_ty {
        Some(ty) => quote! {
            let params: #ty = ::serde_urlencoded::from_str(&_query)
                .map_err(|e| ::rein::types::ReinError::Config(
                    format!("query parse error: {e}")
                ))?;
        },
        None => quote! {},
    };

    quote! {
        fn __op_rest_invoke(
            runtime: ::std::sync::Arc<::rein::ops::OpsRuntime>,
            _path_values: ::std::collections::HashMap<&'static str, ::std::string::String>,
            _query: ::std::string::String,
            _body: ::std::option::Option<::bytes::Bytes>,
        ) -> ::std::pin::Pin<
            ::std::boxed::Box<
                dyn ::std::future::Future<
                    Output = ::rein::types::ReinResult<(::hyper::StatusCode, ::bytes::Bytes)>,
                > + ::std::marker::Send,
            >,
        > {
            ::std::boxed::Box::pin(async move {
                #prep
                let out = #call_expr?;
                let bytes = ::serde_json::to_vec(&out)?;
                ::std::result::Result::Ok((
                    ::hyper::StatusCode::OK,
                    ::bytes::Bytes::from(bytes),
                ))
            })
        }

        ::inventory::submit! {
            ::rein::ops::OpsRestEntry {
                method: ::hyper::Method::#method_ident,
                path_template: #path,
                path_params: &[ #( #path_params ),* ],
                op_name: #op_name,
                invoke: __op_rest_invoke,
            }
        }
    }
}

fn method_ident(method: &str) -> syn::Result<syn::Ident> {
    match method.to_ascii_uppercase().as_str() {
        "GET" => Ok(syn::Ident::new("GET", Span::call_site())),
        "POST" => Ok(syn::Ident::new("POST", Span::call_site())),
        "PUT" => Ok(syn::Ident::new("PUT", Span::call_site())),
        "DELETE" => Ok(syn::Ident::new("DELETE", Span::call_site())),
        "PATCH" => Ok(syn::Ident::new("PATCH", Span::call_site())),
        other => Err(syn::Error::new(
            Span::call_site(),
            format!("unsupported REST method '{other}' (use GET/POST/PUT/DELETE/PATCH)"),
        )),
    }
}

// ===== Parser (unchanged from Phase 0b) =====

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
