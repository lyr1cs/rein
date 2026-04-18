//! Inventory entry definitions collected by `inventory::collect!`.
//!
//! `#[op]` emits one `OpsMetadata` per op, plus up to one `OpsCliEntry`,
//! `OpsMcpEntry`, `OpsRestEntry` based on declared surfaces. Adapters iterate
//! `inventory::iter::<Entry>` at startup to build surface registries.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use bytes::Bytes;

use crate::ops::runtime::OpsRuntime;
use crate::types::ReinResult;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpKind {
    Unary,
}

pub struct OpsCliEntry {
    pub name: &'static str,
    pub op_name: &'static str,
    pub parent: Option<&'static str>,
    pub aliases: &'static [&'static str],
    pub hidden: bool,
    pub build_clap: fn() -> clap::Command,
    pub invoke: fn(
        runtime: Arc<OpsRuntime>,
        matches: &clap::ArgMatches,
    ) -> Pin<Box<dyn Future<Output = ReinResult<String>> + Send>>,
}

pub struct OpsMcpEntry {
    pub op_name: &'static str,
    pub mcp_name: &'static str,
    pub description: &'static str,
    pub input_schema: fn() -> schemars::Schema,
    pub invoke: fn(
        runtime: Arc<OpsRuntime>,
        params_json: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = ReinResult<String>> + Send>>,
}

pub struct OpsRestEntry {
    pub method: hyper::Method,
    pub path_template: &'static str,
    pub path_params: &'static [&'static str],
    pub op_name: &'static str,
    pub invoke: fn(
        runtime: Arc<OpsRuntime>,
        path_values: HashMap<&'static str, String>,
        query: String,
        body: Option<Bytes>,
    ) -> Pin<Box<dyn Future<Output = ReinResult<(hyper::StatusCode, Bytes)>> + Send>>,
}

pub struct OpsMetadata {
    pub name: &'static str,
    pub category: &'static str,
    pub description: &'static str,
    pub kind: OpKind,
    pub mutating: bool,
    pub cli_visible: bool,
    pub mcp_visible: bool,
    pub rest_visible: bool,
    pub mcp_name: Option<&'static str>,
    pub rest_method: Option<hyper::Method>,
    pub rest_path: Option<&'static str>,
    pub params_schema: fn() -> schemars::Schema,
}

inventory::collect!(OpsCliEntry);
inventory::collect!(OpsMcpEntry);
inventory::collect!(OpsRestEntry);
inventory::collect!(OpsMetadata);
