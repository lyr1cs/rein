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
    /// Declared auth policy. Parsed from `auth = "..."`. Stored as string
    /// so validation.rs can emit a precise error for unknown values; the
    /// emit_* functions lower this to `::rein::ops::AuthPolicy::Variant`.
    pub auth: String,
    pub cli: Option<CliBlock>,
    pub mcp: Option<McpBlock>,
    pub rest: Option<RestBlock>,
}

#[derive(Debug, Default)]
pub struct CliBlock {
    pub name: Option<String>,
    /// Parsed from `cli(positional = [...])` but NOT emitted into clap yet —
    /// Phase 3 cleanup will either wire it into `build_clap` or drop the key.
    /// Codex 2026-04-19 audit L2 flagged this as misleading surface area;
    /// keep the field so the parser doesn't reject `positional` in existing
    /// call sites.
    #[allow(dead_code)]
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
    let auth_variant = match attr.auth.as_str() {
        "public" => quote! { ::rein::ops::AuthPolicy::Public },
        "mutation_marker" => quote! { ::rein::ops::AuthPolicy::MutationMarker },
        "read_token" => quote! { ::rein::ops::AuthPolicy::ReadToken },
        // validation.rs rejects unknown values before we reach this path,
        // but keep a fallback that errors loudly rather than silently
        // defaulting — belt-and-braces for future additions.
        other => {
            return Err(syn::Error::new(
                Span::call_site(),
                format!("BUG: validation let through unknown auth '{other}'"),
            ))
        }
    };

    let params_schema_fn = emit_params_schema_fn(fi.params_ty.as_ref());

    let cli_block = attr
        .cli
        .as_ref()
        .map(|cli| emit_cli_block(cli, op_name, description, fn_name, fi));
    let mcp_block = attr
        .mcp
        .as_ref()
        .map(|mcp| emit_mcp_block(mcp, op_name, description, fn_name, fi, mutating));
    let rest_block = match attr.rest.as_ref() {
        Some(rest) => Some(emit_rest_block(rest, op_name, fn_name, fi, &auth_variant)?),
        None => None,
    };

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
                    auth_policy: #auth_variant,
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
    description: &str,
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

    let apply_aliases = if aliases.is_empty() {
        quote! {}
    } else {
        quote! { .visible_aliases([ #( #aliases ),* ]) }
    };
    let apply_hidden = if hidden {
        quote! { .hide(true) }
    } else {
        quote! {}
    };

    let (build_body, pre_extract) = match &fi.params_ty {
        Some(ty) => (
            quote! {
                let cmd = ::clap::Command::new(#cli_name)
                    .about(#description)
                    #apply_aliases
                    #apply_hidden;
                <#ty as ::clap::Args>::augment_args(cmd)
            },
            // Extract params synchronously (before async block) so `_matches`
            // borrow doesn't leak into the returned `'static` future.
            quote! {
                let params_result = <#ty as ::clap::FromArgMatches>::from_arg_matches(_matches);
            },
        ),
        None => (
            quote! {
                ::clap::Command::new(#cli_name)
                    .about(#description)
                    #apply_aliases
                    #apply_hidden
            },
            quote! {},
        ),
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
    mutating: bool,
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
                // M1: when compact mode is active, return human-readable markdown
                // (honouring the pre-A1 compact rendering contract). Non-compact
                // callers still receive serialized JSON.
                let body = if runtime.compact() {
                    <_ as ::rein::ops::IntoMarkdown>::to_markdown(&out)
                } else {
                    let json_value = <_ as ::rein::ops::IntoJson>::to_json(&out);
                    ::serde_json::to_string(&json_value)?
                };
                ::std::result::Result::Ok(body)
            })
        }

        ::inventory::submit! {
            ::rein::ops::OpsMcpEntry {
                op_name: #op_name,
                mcp_name: #mcp_name,
                description: #description,
                mutating: #mutating,
                input_schema: __op_mcp_schema,
                invoke: __op_mcp_invoke,
            }
        }
    }
}

/// Parse a `rest(path = "...")` literal at macro expansion time.
/// Returns `Ok(Vec<TokenStream>)` where each element is one `PathSegment` expr.
/// Validates the single-seg MVP constraint and malformed-brace rules.
///
/// Segments are split on `/`; the leading empty segment from a leading `/` is
/// filtered. Trailing empties are kept so `/api/foo/` becomes 4 tokens and
/// cannot match a 3-token template.
fn parse_path_segments(path: &str, path_span: proc_macro2::Span) -> syn::Result<Vec<TokenStream>> {
    // Strip the leading `/` to avoid a spurious empty first segment, but do NOT
    // strip trailing empties — this lets `/api/foo/` (4 segments) differ from
    // `/api/foo` (3 segments) in the template-match pass, giving trailing-slash 404.
    let stripped = path.trim_start_matches('/');
    let segments: Vec<&str> = if stripped.is_empty() {
        vec![]
    } else {
        stripped.split('/').collect()
    };

    let mut param_count = 0usize;
    let mut result = Vec::with_capacity(segments.len());

    for seg in &segments {
        // Reject empty literal segments: these arise from consecutive slashes
        // (/api//foo) or a trailing slash (/api/foo/). Both indicate a
        // malformed path template in the source code.
        if seg.is_empty() {
            return Err(syn::Error::new(
                path_span,
                "path template must not contain empty segments (no consecutive // or trailing /)",
            ));
        }

        // Count braces to detect unbalanced / mixed cases.
        let open = seg.chars().filter(|&c| c == '{').count();
        let close = seg.chars().filter(|&c| c == '}').count();

        if open == 0 && close == 0 {
            // Pure literal segment.
            let lit = proc_macro2::Literal::string(seg);
            result.push(quote! { ::rein::ops::PathSegment::Literal(#lit) });
            continue;
        }

        // Any brace mismatch → unbalanced.
        if open != close {
            return Err(syn::Error::new(
                path_span,
                format!("unbalanced brace in path template: segment '{seg}'"),
            ));
        }

        // Exactly one pair of braces: must span the entire segment, i.e. starts
        // with `{` and ends with `}` with no surrounding literal text.
        if !(seg.starts_with('{') && seg.ends_with('}')) {
            return Err(syn::Error::new(
                path_span,
                format!(
                    "path placeholders must occupy an entire segment: '{seg}' \
                     — use '{{name}}' alone, not 'prefix{{name}}' or '{{name}}suffix'"
                ),
            ));
        }

        // Multiple pairs in one segment — reject.
        if open > 1 {
            return Err(syn::Error::new(
                path_span,
                "path templates support at most one {param} placeholder per segment in Phase 2.5",
            ));
        }

        // Extract param name — strip `{` and `}`.
        let name = &seg[1..seg.len() - 1];

        // Empty placeholder `{}`.
        if name.is_empty() {
            return Err(syn::Error::new(
                path_span,
                "empty path placeholder {} is not allowed",
            ));
        }

        // Single-seg MVP: at most one Param across the whole path.
        param_count += 1;
        if param_count > 1 {
            return Err(syn::Error::new(
                path_span,
                "path templates support at most one {param} placeholder in Phase 2.5 \
                 (multiple placeholders like {a}/{b} are not yet supported)",
            ));
        }

        let name_lit = proc_macro2::Literal::string(name);
        result.push(quote! { ::rein::ops::PathSegment::Param(#name_lit) });
    }

    Ok(result)
}

fn emit_rest_block(
    rest: &RestBlock,
    op_name: &str,
    fn_name: &syn::Ident,
    fi: &FnInfo,
    auth_variant: &TokenStream,
) -> syn::Result<TokenStream> {
    let method_ident = method_ident(&rest.method)?;

    // T2: parse path segments at macro expansion time, emitting PathSegment literals.
    // Static paths with no placeholders keep path_segments: &[] (leading-only split,
    // never populate segments list unless a {seg} is present).
    let path = &rest.path;
    let path_span = Span::call_site();

    // Check whether the path contains any placeholders at all before parsing.
    let has_placeholder = path.contains('{');
    let path_segments_tokens = if has_placeholder {
        let segs = parse_path_segments(path, path_span)?;
        quote! { &[ #( #segs ),* ] }
    } else {
        // Literal-only path — no segments needed; exact-match pass handles it.
        quote! { &[] }
    };

    let call_expr = emit_call(fn_name, fi.params_ty.is_some(), fi.is_async);

    // A1 H5 (audit 2026-04-19 follow-up): GET/HEAD read from query string,
    // other methods read from JSON body. The macro cannot know the runtime
    // method at callsite, so we branch on the declared `rest.method` at
    // compile time. Both paths map parse failures to BadRequest so REST
    // clients see 400; the H4 plumbing carries the kind to the dispatcher.
    let is_body_method = matches!(
        rest.method.to_ascii_uppercase().as_str(),
        "POST" | "PUT" | "PATCH" | "DELETE"
    );

    // T4: path_values merge strategy.
    // GET/HEAD: we use query-string append so serde_urlencoded coercion still
    // works for bool/u64 query fields. Path values (always String per spec) are
    // appended after stripping any conflicting query key, so path wins.
    // POST/PUT/PATCH/DELETE: body is a JSON Value; we merge path_values into the
    // object before deserialization — path fields go in as JSON strings.
    let prep = match (&fi.params_ty, is_body_method) {
        (Some(ty), false) => quote! {
            // GET merge: strip any existing occurrence of path param keys from
            // the query string, then append path values so they win on conflict.
            // Path values are inserted structurally (as map entries) and the
            // whole set is re-serialized with serde_urlencoded::to_string, which
            // percent-encodes all values. This prevents a decoded path value that
            // contains '&' or '=' from forging extra query parameters.
            let _effective_query = if _path_values.is_empty() {
                _query.clone()
            } else {
                // Decode the existing query into (k, v) pairs, dropping any key
                // that would be overridden by a path value.
                let mut _q_pairs: Vec<(::std::string::String, ::std::string::String)> =
                    ::serde_urlencoded::from_str::<Vec<(::std::string::String, ::std::string::String)>>(&_query)
                        .unwrap_or_default()
                        .into_iter()
                        .filter(|(k, _)| !_path_values.contains_key(k.as_str()))
                        .collect();
                // Append path values — they always win over query-string values.
                for (pk, pv) in &_path_values {
                    _q_pairs.push((pk.to_string(), pv.clone()));
                }
                // Serialize back with proper percent-encoding.
                ::serde_urlencoded::to_string(&_q_pairs)
                    .unwrap_or_default()
            };
            let params: #ty = ::serde_urlencoded::from_str(&_effective_query)
                .map_err(|e| {
                    ::rein::types::ReinError::Config(
                        format!("query parse error: {e}")
                    )
                    .with_kind(::rein::types::OpsErrorKind::BadRequest)
                })?;
        },
        (Some(ty), true) => quote! {
            // POST/PUT/PATCH/DELETE merge: decode body to JSON Value first,
            // then overlay path_values (path always wins).
            // Empty body is treated as {} (omitting all optional fields).
            // Non-empty, non-object body (null, array, scalar, bool) is rejected
            // with 400 BadRequest — required-field validation must not be bypassed
            // by having path values silently satisfy fields the client omitted.
            let body_bytes = _body.unwrap_or_default();
            let mut _params_json: ::serde_json::Value = if body_bytes.is_empty() {
                ::serde_json::Value::Object(::serde_json::Map::new())
            } else {
                let _parsed: ::serde_json::Value = ::serde_json::from_slice(&body_bytes)
                    .map_err(|e| {
                        ::rein::types::ReinError::Config(
                            format!("JSON body parse error: {e}")
                        )
                        .with_kind(::rein::types::OpsErrorKind::BadRequest)
                    })?;
                if !_parsed.is_object() {
                    return ::std::result::Result::Err(
                        ::rein::types::ReinError::Config(
                            "request body must be a JSON object".into()
                        )
                        .with_kind(::rein::types::OpsErrorKind::BadRequest)
                    );
                }
                _parsed
            };
            // Merge path values — path wins over body.
            if let ::serde_json::Value::Object(ref mut obj) = _params_json {
                for (pk, pv) in &_path_values {
                    obj.insert(pk.to_string(), ::serde_json::Value::String(pv.clone()));
                }
            }
            let params: #ty = ::serde_json::from_value(_params_json)
                .map_err(|e| {
                    ::rein::types::ReinError::Config(
                        format!("params deserialize error: {e}")
                    )
                    .with_kind(::rein::types::OpsErrorKind::BadRequest)
                })?;
        },
        (None, _) => quote! {},
    };

    Ok(quote! {
        fn __op_rest_invoke(
            runtime: ::std::sync::Arc<::rein::ops::OpsRuntime>,
            _path_values: ::std::collections::HashMap<&'static str, ::std::string::String>,
            _query: ::std::string::String,
            _body: ::std::option::Option<::bytes::Bytes>,
        ) -> ::std::pin::Pin<
            ::std::boxed::Box<
                dyn ::std::future::Future<
                    Output = ::rein::types::ReinResult<(
                        ::hyper::StatusCode,
                        ::bytes::Bytes,
                        &'static str,
                    )>,
                > + ::std::marker::Send,
            >,
        > {
            ::std::boxed::Box::pin(async move {
                #prep
                let out = #call_expr?;
                // Phase 3: ops can opt into a raw (non-JSON) body + custom
                // content-type by overriding `IntoJson::to_raw_response`.
                // Default impl returns `None`, preserving the pre-Phase-3
                // JSON contract for every other op.
                let (bytes, content_type) =
                    match ::rein::ops::IntoJson::to_raw_response(&out) {
                        ::std::option::Option::Some((ct, raw)) => {
                            (::bytes::Bytes::from(raw), ct)
                        }
                        ::std::option::Option::None => {
                            let json_value = ::rein::ops::IntoJson::to_json(&out);
                            let json_bytes = ::serde_json::to_vec(&json_value)?;
                            (::bytes::Bytes::from(json_bytes), "application/json")
                        }
                    };
                ::std::result::Result::Ok((
                    ::hyper::StatusCode::OK,
                    bytes,
                    content_type,
                ))
            })
        }

        ::inventory::submit! {
            ::rein::ops::OpsRestEntry {
                method: ::hyper::Method::#method_ident,
                path_template: #path,
                // T2: segments populated for paths with {seg} placeholders;
                // literal-only paths keep &[] for exact-match pass efficiency.
                path_segments: #path_segments_tokens,
                op_name: #op_name,
                auth_policy: #auth_variant,
                invoke: __op_rest_invoke,
            }
        }
    })
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

/// Report "duplicate key/block 'X'" pointing at the offending span. Shared
/// between the top-level attribute parser and the nested cli/mcp/rest block
/// parsers so every duplicate declaration fails at compile time instead of
/// silently overwriting an earlier entry. Codex audit 2026-04-19 (H1) flagged
/// the silent last-wins behavior.
fn dup_err<S: quote::ToTokens>(span_src: S, key: &str) -> syn::Error {
    syn::Error::new_spanned(span_src, format!("duplicate #[op] key/block: '{key}'"))
}

fn parse_op_attr(attr: TokenStream) -> syn::Result<OpAttr> {
    let input: AttrInput = parse2(attr)?;

    let mut name: Option<String> = None;
    let mut category: Option<String> = None;
    let mut description: Option<String> = None;
    let mut kind: Option<String> = None;
    let mut mutating: Option<bool> = None;
    let mut auth: Option<String> = None;
    let mut cli: Option<CliBlock> = None;
    let mut mcp: Option<McpBlock> = None;
    let mut rest: Option<RestBlock> = None;

    for meta in input.metas {
        match meta {
            Meta::NameValue(nv) => {
                let key = ident_string(&nv.path)?;
                match key.as_str() {
                    "name" => {
                        if name.is_some() {
                            return Err(dup_err(&nv.path, "name"));
                        }
                        name = Some(extract_string_lit(&nv.value, "name")?);
                    }
                    "category" => {
                        if category.is_some() {
                            return Err(dup_err(&nv.path, "category"));
                        }
                        category = Some(extract_string_lit(&nv.value, "category")?);
                    }
                    "description" => {
                        if description.is_some() {
                            return Err(dup_err(&nv.path, "description"));
                        }
                        description = Some(extract_string_lit(&nv.value, "description")?);
                    }
                    "kind" => {
                        if kind.is_some() {
                            return Err(dup_err(&nv.path, "kind"));
                        }
                        kind = Some(extract_string_lit(&nv.value, "kind")?);
                    }
                    "mutating" => {
                        if mutating.is_some() {
                            return Err(dup_err(&nv.path, "mutating"));
                        }
                        mutating = Some(extract_bool_lit(&nv.value, "mutating")?);
                    }
                    "auth" => {
                        if auth.is_some() {
                            return Err(dup_err(&nv.path, "auth"));
                        }
                        auth = Some(extract_string_lit(&nv.value, "auth")?);
                    }
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
                    "cli" => {
                        if cli.is_some() {
                            return Err(dup_err(&list.path, "cli"));
                        }
                        cli = Some(parse_cli_block(inner)?);
                    }
                    "mcp" => {
                        if mcp.is_some() {
                            return Err(dup_err(&list.path, "mcp"));
                        }
                        mcp = Some(parse_mcp_block(inner)?);
                    }
                    "rest" => {
                        if rest.is_some() {
                            return Err(dup_err(&list.path, "rest"));
                        }
                        rest = Some(parse_rest_block(inner)?);
                    }
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

    let kind = kind.unwrap_or_else(|| "unary".to_string());
    let mutating = mutating.unwrap_or(false);
    // Default to "public" — read-only endpoints dominate the migration and
    // explicit opt-in on mutations keeps the surface area small.
    let auth = auth.unwrap_or_else(|| "public".to_string());

    Ok(OpAttr {
        name: name.ok_or_else(|| {
            syn::Error::new(Span::call_site(), "missing required #[op] key 'name'")
        })?,
        category: category.ok_or_else(|| {
            syn::Error::new(Span::call_site(), "missing required #[op] key 'category'")
        })?,
        description: description.ok_or_else(|| {
            syn::Error::new(
                Span::call_site(),
                "missing required #[op] key 'description'",
            )
        })?,
        kind,
        mutating,
        auth,
        cli,
        mcp,
        rest,
    })
}

fn parse_cli_block(tokens: TokenStream) -> syn::Result<CliBlock> {
    let input: AttrInput = parse2(tokens)?;
    let mut name: Option<String> = None;
    let mut hidden: Option<bool> = None;
    let mut parent: Option<String> = None;
    let mut positional: Option<Vec<String>> = None;
    let mut aliases: Option<Vec<String>> = None;
    for meta in input.metas {
        match meta {
            Meta::NameValue(nv) => {
                let key = ident_string(&nv.path)?;
                match key.as_str() {
                    "name" => {
                        if name.is_some() {
                            return Err(dup_err(&nv.path, "cli.name"));
                        }
                        name = Some(extract_string_lit(&nv.value, "cli.name")?);
                    }
                    "hidden" => {
                        if hidden.is_some() {
                            return Err(dup_err(&nv.path, "cli.hidden"));
                        }
                        hidden = Some(extract_bool_lit(&nv.value, "cli.hidden")?);
                    }
                    "parent" => {
                        if parent.is_some() {
                            return Err(dup_err(&nv.path, "cli.parent"));
                        }
                        parent = Some(extract_string_lit(&nv.value, "cli.parent")?);
                    }
                    "positional" => {
                        if positional.is_some() {
                            return Err(dup_err(&nv.path, "cli.positional"));
                        }
                        positional = Some(extract_string_array(&nv.value, "cli.positional")?);
                    }
                    "aliases" => {
                        if aliases.is_some() {
                            return Err(dup_err(&nv.path, "cli.aliases"));
                        }
                        aliases = Some(extract_string_array(&nv.value, "cli.aliases")?);
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
    Ok(CliBlock {
        name,
        hidden: hidden.unwrap_or(false),
        parent,
        positional: positional.unwrap_or_default(),
        aliases: aliases.unwrap_or_default(),
    })
}

fn parse_mcp_block(tokens: TokenStream) -> syn::Result<McpBlock> {
    let input: AttrInput = parse2(tokens)?;
    let mut name: Option<String> = None;
    for meta in input.metas {
        match meta {
            Meta::NameValue(nv) => {
                let key = ident_string(&nv.path)?;
                match key.as_str() {
                    "name" => {
                        if name.is_some() {
                            return Err(dup_err(&nv.path, "mcp.name"));
                        }
                        name = Some(extract_string_lit(&nv.value, "mcp.name")?);
                    }
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
    for meta in input.metas {
        match meta {
            Meta::NameValue(nv) => {
                let key = ident_string(&nv.path)?;
                match key.as_str() {
                    "method" => {
                        if method.is_some() {
                            return Err(dup_err(&nv.path, "rest.method"));
                        }
                        method = Some(extract_string_lit(&nv.value, "rest.method")?);
                    }
                    "path" => {
                        if path.is_some() {
                            return Err(dup_err(&nv.path, "rest.path"));
                        }
                        path = Some(extract_string_lit(&nv.value, "rest.path")?);
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
            syn::Error::new(
                Span::call_site(),
                "rest block missing required key 'method'",
            )
        })?,
        path: path.ok_or_else(|| {
            syn::Error::new(Span::call_site(), "rest block missing required key 'path'")
        })?,
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
