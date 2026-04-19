use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::time::Duration;

use serde::Serialize;

use crate::config::{Provider, ReinConfig};
use crate::embed;
use crate::extract::hooks::buffer;
use crate::extract::hooks::queue::{collect_queue_diagnostics, QueueGroupDiagnostics};
use crate::ops::registry;
use crate::search::warmup;
use crate::store::hnsw::HnswIndex;
use crate::store::sqlite::SqliteStore;
use crate::store::tantivy_fts::TantivyFts;
use crate::types::traits::{Embedder as _, MemoryStore};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum CheckStatus {
    Ok,
    Warn,
    Fail,
}

impl CheckStatus {
    fn label(self) -> &'static str {
        match self {
            Self::Ok => "OK",
            Self::Warn => "WARN",
            Self::Fail => "FAIL",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DoctorCategory {
    Configuration,
    Runtime,
    Storage,
    Index,
    Queue,
    Network,
    Architecture,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum DoctorSeverity {
    Info,
    Warning,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ReportStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

#[derive(Debug, Clone, Serialize)]
pub struct DoctorCheck {
    pub name: &'static str,
    pub category: DoctorCategory,
    pub severity: DoctorSeverity,
    pub status: CheckStatus,
    pub fixable: bool,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repair_hint: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DoctorReport {
    pub status: ReportStatus,
    pub checks: Vec<DoctorCheck>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub fixes_applied: Vec<String>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct DoctorOptions {
    pub network: bool,
    pub fix: bool,
}

#[derive(Debug)]
struct StoreSnapshot {
    total_memories: usize,
    active_memories: usize,
    embed_cache_rows: usize,
    artifact_rows: usize,
}

pub async fn run(config: &ReinConfig, options: DoctorOptions) -> DoctorReport {
    let mut checks = Vec::new();
    let mut fixes_applied = Vec::new();

    checks.push(check_embedding_provider(config));
    checks.push(check_extract_provider(config));
    checks.push(check_query_expansion_provider(config));
    checks.push(check_reranker_provider(config));
    checks.push(check_supermemory(config));
    checks.push(check_http_auth(config));
    checks.push(check_proxy_auth(config));
    checks.push(check_gui_runtime(config));
    checks.push(check_proxy_runtime(config));
    checks.push(check_overview_version());
    checks.push(check_cli_registry());
    checks.push(check_mcp_registry());
    checks.push(check_rest_registry());
    if options.fix {
        fixes_applied.extend(apply_queue_fixes(config));
    }

    match config.open_store() {
        Ok(store) => {
            let stats = store.stats();
            match stats {
                Ok(stats) => {
                    checks.push(ok_in(
                        DoctorCategory::Storage,
                        "database",
                        format!(
                            "{} memories, {} topics, {} memoirs at {}",
                            stats.total_memories,
                            stats.topic_count,
                            stats.memoir_count,
                            store.db_path().display()
                        ),
                    ));

                    match collect_store_snapshot(&store) {
                        Ok(snapshot) => {
                            if options.fix {
                                fixes_applied.extend(apply_local_fixes(config, &store));
                            }
                            let (hnsw_check, indexed_vectors) =
                                inspect_hnsw(&store, snapshot.total_memories);
                            checks.push(check_vector_coverage(config, &snapshot, indexed_vectors));
                            checks.push(check_tantivy(&store, snapshot.active_memories));
                            checks.push(hnsw_check);
                        }
                        Err(e) => checks.push(fail_in(
                            DoctorCategory::Storage,
                            "database_snapshot",
                            e.to_string(),
                        )),
                    }
                }
                Err(e) => checks.push(fail_in(DoctorCategory::Storage, "database", e.to_string())),
            }
        }
        Err(e) => checks.push(fail_in(DoctorCategory::Storage, "database", e.to_string())),
    }

    let queue_diag = collect_queue_diagnostics(config);
    checks.push(check_queues(&queue_diag));

    if options.network {
        checks.push(check_embedding_network(config).await);
    }

    DoctorReport {
        status: overall_status(&checks),
        checks,
        fixes_applied,
    }
}

pub fn format_human(report: &DoctorReport) -> String {
    let mut lines = vec!["rein doctor".to_string(), "===========".to_string()];
    for check in &report.checks {
        lines.push(format!(
            "[{}] {}: {}",
            check.status.label(),
            check.name,
            check.message
        ));
        if let Some(repair_hint) = &check.repair_hint {
            lines.push(format!("  repair: {repair_hint}"));
        }
    }
    if !report.fixes_applied.is_empty() {
        lines.push(String::new());
        lines.push("Fixes Applied:".to_string());
        for fix in &report.fixes_applied {
            lines.push(format!("- {fix}"));
        }
    }
    lines.push(String::new());
    lines.push(format!("Overall: {}", overall_label(report.status)));
    lines.join("\n")
}

impl DoctorReport {
    pub fn has_failures(&self) -> bool {
        self.checks
            .iter()
            .any(|check| matches!(check.status, CheckStatus::Fail))
    }

    pub fn exit_code(&self) -> i32 {
        if self.has_failures() {
            1
        } else {
            0
        }
    }
}

fn overall_status(checks: &[DoctorCheck]) -> ReportStatus {
    if checks.iter().any(|c| matches!(c.status, CheckStatus::Fail)) {
        ReportStatus::Unhealthy
    } else if checks.iter().any(|c| matches!(c.status, CheckStatus::Warn)) {
        ReportStatus::Degraded
    } else {
        ReportStatus::Healthy
    }
}

fn overall_label(status: ReportStatus) -> &'static str {
    match status {
        ReportStatus::Healthy => "healthy",
        ReportStatus::Degraded => "degraded",
        ReportStatus::Unhealthy => "unhealthy",
    }
}

fn ok_in(category: DoctorCategory, name: &'static str, message: impl Into<String>) -> DoctorCheck {
    DoctorCheck {
        name,
        category,
        severity: DoctorSeverity::Info,
        status: CheckStatus::Ok,
        fixable: false,
        message: message.into(),
        repair_hint: None,
    }
}

fn warn_in(
    category: DoctorCategory,
    name: &'static str,
    message: impl Into<String>,
) -> DoctorCheck {
    DoctorCheck {
        name,
        category,
        severity: DoctorSeverity::Warning,
        status: CheckStatus::Warn,
        fixable: false,
        message: message.into(),
        repair_hint: None,
    }
}

fn fail_in(
    category: DoctorCategory,
    name: &'static str,
    message: impl Into<String>,
) -> DoctorCheck {
    DoctorCheck {
        name,
        category,
        severity: DoctorSeverity::Error,
        status: CheckStatus::Fail,
        fixable: false,
        message: message.into(),
        repair_hint: None,
    }
}

fn warn_with_hint(
    category: DoctorCategory,
    name: &'static str,
    message: impl Into<String>,
    repair_hint: impl Into<String>,
) -> DoctorCheck {
    DoctorCheck {
        name,
        category,
        severity: DoctorSeverity::Warning,
        status: CheckStatus::Warn,
        fixable: true,
        message: message.into(),
        repair_hint: Some(repair_hint.into()),
    }
}

fn fail_with_hint(
    category: DoctorCategory,
    name: &'static str,
    message: impl Into<String>,
    repair_hint: impl Into<String>,
) -> DoctorCheck {
    DoctorCheck {
        name,
        category,
        severity: DoctorSeverity::Error,
        status: CheckStatus::Fail,
        fixable: false,
        message: message.into(),
        repair_hint: Some(repair_hint.into()),
    }
}

fn check_overview_version() -> DoctorCheck {
    let cargo_version = env!("CARGO_PKG_VERSION");
    match parse_agents_overview_version(include_str!("../../../AGENTS.md")) {
        Some(version) if version == cargo_version => ok_in(
            DoctorCategory::Architecture,
            "overview_version",
            format!("AGENTS overview version matches Cargo v{cargo_version}"),
        ),
        Some(version) => warn_in(
            DoctorCategory::Architecture,
            "overview_version",
            format!("AGENTS overview says v{version} but Cargo.toml is v{cargo_version}"),
        ),
        None => warn_in(
            DoctorCategory::Architecture,
            "overview_version",
            "could not parse AGENTS overview version",
        ),
    }
}

fn check_cli_registry() -> DoctorCheck {
    let registry_count = registry::cli_operations().len();
    // A1: migrated ops no longer appear as `Some(Commands::*)` arms in main.rs —
    // they register through `OpsCliEntry` inventory. Sum both sources so the
    // drift check stays honest during the incremental migration.
    let inventory_count = inventory::iter::<crate::ops::OpsCliEntry>().count();
    let derived_count = count_cli_operations_in_source(include_str!("main.rs"));
    let source_count = derived_count + inventory_count;
    if registry_count != source_count {
        return fail_in(
            DoctorCategory::Architecture,
            "cli_registry",
            format!(
                "registry has {registry_count} CLI operations but main.rs exposes {derived_count} derived + {inventory_count} inventory = {source_count}"
            ),
        );
    }
    ok_in(
        DoctorCategory::Architecture,
        "cli_registry",
        format!(
            "{registry_count} CLI operations match source ({derived_count} derived + {inventory_count} inventory)"
        ),
    )
}

fn check_mcp_registry() -> DoctorCheck {
    let registry_count = registry::mcp_operations().len();
    let source_count = count_mcp_tools_in_source(include_str!("mcp/server.rs"));
    if registry_count != source_count {
        return fail_in(
            DoctorCategory::Architecture,
            "mcp_registry",
            format!(
                "registry has {registry_count} MCP tools but src/mcp/server.rs exposes {source_count}"
            ),
        );
    }

    let doc_counts = documented_mcp_tool_counts();
    if !doc_counts.is_empty() && doc_counts.iter().any(|(_, count)| *count != registry_count) {
        let doc_summary = doc_counts
            .iter()
            .map(|(name, count)| format!("{name}={count}"))
            .collect::<Vec<_>>()
            .join(", ");
        return warn_in(
            DoctorCategory::Architecture,
            "mcp_registry",
            format!("{registry_count} MCP tools match source, but docs still say {doc_summary}"),
        );
    }

    ok_in(
        DoctorCategory::Architecture,
        "mcp_registry",
        format!("{registry_count} MCP tools match source"),
    )
}

fn check_rest_registry() -> DoctorCheck {
    let registry_count = registry::rest_operations().len();
    // A1: migrated ops appear via OpsRestEntry inventory; remaining legacy
    // routes still live in src/mcp/rest.rs as `(&Method::*, "/api/...")` arms.
    let inventory_count = inventory::iter::<crate::ops::OpsRestEntry>().count();
    let derived_count = count_rest_operations_in_source(include_str!("mcp/rest.rs"));
    let source_count = derived_count + inventory_count;
    if registry_count != source_count {
        return fail_in(
            DoctorCategory::Architecture,
            "rest_registry",
            format!(
                "registry has {registry_count} REST operations but src/mcp/rest.rs exposes {derived_count} derived + {inventory_count} inventory = {source_count}"
            ),
        );
    }
    ok_in(
        DoctorCategory::Architecture,
        "rest_registry",
        format!(
            "{registry_count} REST operations match source ({derived_count} derived + {inventory_count} inventory)"
        ),
    )
}

fn parse_agents_overview_version(text: &str) -> Option<String> {
    text.lines().find_map(|line| {
        let line = line.trim();
        let version = line.strip_prefix("rein v")?;
        let version = version.split_whitespace().next()?;
        Some(version.to_string())
    })
}

fn parse_documented_mcp_tool_count(text: &str) -> Option<usize> {
    text.lines().find_map(parse_documented_mcp_tool_count_line)
}

fn parse_documented_mcp_tool_count_line(line: &str) -> Option<usize> {
    let idx = line.find("MCP tools")?;
    let prefix = line[..idx].trim_end();
    let digits: String = prefix
        .chars()
        .rev()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    digits.parse().ok()
}

fn documented_mcp_tool_counts() -> Vec<(&'static str, usize)> {
    let mut counts = Vec::new();
    for (name, text) in [
        ("AGENTS.md", include_str!("../../../AGENTS.md")),
        ("README.md", include_str!("../../../README.md")),
    ] {
        if let Some(count) = parse_documented_mcp_tool_count(text) {
            counts.push((name, count));
        }
    }
    counts
}

fn count_cli_operations_in_source(source: &str) -> usize {
    let source = source.split("#[cfg(test)]").next().unwrap_or(source);
    source
        .lines()
        .filter(|line| line.contains("Some(Commands::"))
        .count()
}

fn count_mcp_tools_in_source(source: &str) -> usize {
    source
        .lines()
        .filter(|line| line.trim_start().starts_with("#[tool("))
        .count()
}

fn count_rest_operations_in_source(source: &str) -> usize {
    // Only count lines that open a route arm — i.e. the `(&Method::` appears at
    // the start of the (trimmed) line. This excludes comments, string literals,
    // and code that merely references the pattern in a helper.
    // Also strip #[cfg(test)] and later so test fixtures don't inflate the count.
    let source = source.split("#[cfg(test)]").next().unwrap_or(source);
    source
        .lines()
        .filter(|line| line.trim_start().starts_with("(&Method::"))
        .count()
}

fn check_embedding_provider(config: &ReinConfig) -> DoctorCheck {
    match config.embedding_provider() {
        Provider::Google => match config.embedding.google.api_key.as_ref() {
            Some(_) => ok_in(
                DoctorCategory::Configuration,
                "embedding_provider",
                format!(
                    "google:{} configured ({}d)",
                    config.embedding.google.model, config.embedding.dimensions
                ),
            ),
            None => warn_in(
                DoctorCategory::Configuration,
                "embedding_provider",
                "google configured but GEMINI_API_KEY is missing; vector embedding is disabled",
            ),
        },
        Provider::Omlx => ok_in(
            DoctorCategory::Configuration,
            "embedding_provider",
            format!(
                "omlx:{} at {} ({}d)",
                config.embedding.omlx.model,
                config.embedding.omlx.endpoint,
                config.embedding.dimensions
            ),
        ),
        Provider::None => ok_in(
            DoctorCategory::Configuration,
            "embedding_provider",
            "disabled",
        ),
    }
}

fn check_extract_provider(config: &ReinConfig) -> DoctorCheck {
    match config.extract_provider() {
        Provider::Google => match config.extract.google.api_key.as_ref() {
            Some(_) => ok_in(
                DoctorCategory::Configuration,
                "extract_provider",
                format!("google:{} configured", config.extract.google.model),
            ),
            None => warn_in(
                DoctorCategory::Configuration,
                "extract_provider",
                "google configured but GEMINI_API_KEY is missing; LLM extraction is disabled",
            ),
        },
        Provider::Omlx => ok_in(
            DoctorCategory::Configuration,
            "extract_provider",
            format!(
                "omlx:{} at {}",
                config.extract.omlx.model, config.extract.omlx.endpoint
            ),
        ),
        Provider::None => ok_in(
            DoctorCategory::Configuration,
            "extract_provider",
            "disabled",
        ),
    }
}

fn check_query_expansion_provider(config: &ReinConfig) -> DoctorCheck {
    match config.expand_provider() {
        Provider::Google => match config.query_expansion.google.api_key.as_ref() {
            Some(_) => ok_in(
                DoctorCategory::Configuration,
                "query_expansion",
                format!("google:{} configured", config.query_expansion.google.model),
            ),
            None => warn_in(
                DoctorCategory::Configuration,
                "query_expansion",
                "google configured but GEMINI_API_KEY is missing; expansion is disabled",
            ),
        },
        Provider::Omlx => ok_in(
            DoctorCategory::Configuration,
            "query_expansion",
            format!(
                "omlx:{} at {}",
                config.query_expansion.omlx.model, config.query_expansion.omlx.endpoint
            ),
        ),
        Provider::None => ok_in(DoctorCategory::Configuration, "query_expansion", "disabled"),
    }
}

fn check_reranker_provider(config: &ReinConfig) -> DoctorCheck {
    match config.reranker_provider() {
        Provider::Google => match config.query_expansion.google.api_key.as_ref() {
            Some(_) => ok_in(
                DoctorCategory::Configuration,
                "llm_reranker",
                format!(
                    "google:{} configured (top_n={})",
                    config.query_expansion.google.model, config.search.llm_reranker_top_n
                ),
            ),
            None => warn_in(
                DoctorCategory::Configuration,
                "llm_reranker",
                "google reranker configured but GEMINI_API_KEY is missing; reranker will be skipped",
            ),
        },
        Provider::Omlx => ok_in(
            DoctorCategory::Configuration,
            "llm_reranker",
            format!(
                "omlx:{} at {} (top_n={})",
                config.query_expansion.omlx.model,
                config.query_expansion.omlx.endpoint,
                config.search.llm_reranker_top_n
            ),
        ),
        Provider::None => ok_in(DoctorCategory::Configuration, "llm_reranker", "disabled"),
    }
}

fn check_supermemory(config: &ReinConfig) -> DoctorCheck {
    if !config.sync.supermemory_enabled {
        return ok_in(DoctorCategory::Configuration, "supermemory", "disabled");
    }
    match config.sync.api_key.as_ref() {
        Some(_) => ok_in(
            DoctorCategory::Configuration,
            "supermemory",
            format!("enabled via {}", config.sync.endpoint),
        ),
        None => warn_in(
            DoctorCategory::Configuration,
            "supermemory",
            "enabled but SUPERMEMORY_CC_API_KEY is missing; cross-validation will be partial",
        ),
    }
}

fn check_http_auth(config: &ReinConfig) -> DoctorCheck {
    if !config.server.sse_enabled && !config.server.gui_enabled {
        return ok_in(
            DoctorCategory::Configuration,
            "http_auth",
            "HTTP/SSE disabled",
        );
    }

    let token_present = std::env::var("REIN_HTTP_TOKEN")
        .ok()
        .is_some_and(|token| !token.trim().is_empty());
    let is_loopback = is_loopback_bind(&config.server.sse_bind);
    let allow_unauth = config.server.allow_unauthenticated_loopback && is_loopback;

    if token_present {
        ok_in(
            DoctorCategory::Configuration,
            "http_auth",
            format!(
                "token configured for {}:{}",
                config.server.sse_bind, config.server.sse_port
            ),
        )
    } else if allow_unauth {
        ok_in(
            DoctorCategory::Configuration,
            "http_auth",
            format!(
                "loopback-only unauthenticated access allowed for {}:{}",
                config.server.sse_bind, config.server.sse_port
            ),
        )
    } else {
        fail_with_hint(
            DoctorCategory::Configuration,
            "http_auth",
            format!(
                "HTTP/SSE is enabled on {}:{} without REIN_HTTP_TOKEN",
                config.server.sse_bind, config.server.sse_port
            ),
            "set REIN_HTTP_TOKEN=<secret> or enable [server].allow_unauthenticated_loopback for loopback-only access",
        )
    }
}

fn check_proxy_auth(config: &ReinConfig) -> DoctorCheck {
    let token_present = std::env::var("REIN_PROXY_TOKEN")
        .ok()
        .filter(|token| !token.trim().is_empty())
        .or_else(|| {
            std::env::var("REIN_HTTP_TOKEN")
                .ok()
                .filter(|token| !token.trim().is_empty())
        })
        .is_some();
    let is_loopback = is_loopback_bind(&config.proxy.bind);
    let allow_unauth = config.proxy.allow_unauthenticated_loopback && is_loopback;

    if token_present {
        ok_in(
            DoctorCategory::Configuration,
            "proxy_auth",
            format!(
                "token configured for {}:{}",
                config.proxy.bind, config.proxy.port
            ),
        )
    } else if allow_unauth {
        ok_in(
            DoctorCategory::Configuration,
            "proxy_auth",
            format!(
                "loopback-only unauthenticated access allowed for {}:{}",
                config.proxy.bind, config.proxy.port
            ),
        )
    } else {
        fail_with_hint(
            DoctorCategory::Configuration,
            "proxy_auth",
            format!(
                "proxy cannot start on {}:{} without REIN_PROXY_TOKEN or REIN_HTTP_TOKEN",
                config.proxy.bind, config.proxy.port
            ),
            "set REIN_PROXY_TOKEN=<secret> or enable [proxy].allow_unauthenticated_loopback for loopback-only access",
        )
    }
}

#[derive(Debug, Clone)]
enum RuntimeProbe {
    None,
    Ok(String),
    Warn(String),
}

fn runtime_status_check(
    name: &'static str,
    pid: Option<u32>,
    port: u16,
    port_open: bool,
    probe: RuntimeProbe,
) -> DoctorCheck {
    let probe_suffix = match &probe {
        RuntimeProbe::None => None,
        RuntimeProbe::Ok(extra) | RuntimeProbe::Warn(extra) => Some(extra.as_str()),
    };
    match (pid, port_open) {
        (Some(pid), true) => match probe {
            RuntimeProbe::None => ok_in(
                DoctorCategory::Runtime,
                name,
                format!("running on :{port} with PID {pid}"),
            ),
            RuntimeProbe::Ok(extra) => ok_in(
                DoctorCategory::Runtime,
                name,
                format!("running on :{port} with PID {pid}; {extra}"),
            ),
            RuntimeProbe::Warn(extra) => warn_in(
                DoctorCategory::Runtime,
                name,
                format!("running on :{port} with PID {pid}; {extra}"),
            ),
        },
        (None, true) => {
            let mut msg =
                format!("port :{port} is open but no PID file is present; external listener?");
            if let Some(extra) = probe_suffix {
                msg.push_str("; ");
                msg.push_str(extra);
            }
            warn_in(DoctorCategory::Runtime, name, msg)
        }
        (Some(pid), false) => {
            let mut msg = format!("PID {pid} recorded for :{port}, but the port is closed");
            if let Some(extra) = probe_suffix {
                msg.push_str("; ");
                msg.push_str(extra);
            }
            warn_in(DoctorCategory::Runtime, name, msg)
        }
        (None, false) => ok_in(DoctorCategory::Runtime, name, format!("stopped on :{port}")),
    }
}

fn check_gui_runtime(config: &ReinConfig) -> DoctorCheck {
    let port = config.server.sse_port;
    let pid = crate::service::is_running("gui");
    let port_open = localhost_port_open(port);
    let probe = if port_open {
        match probe_gui_health(port) {
            Ok(200) => RuntimeProbe::Ok("/api/health returned 200".to_string()),
            Ok(401) => RuntimeProbe::Warn("/api/health returned 401 (token mismatch?)".to_string()),
            Ok(status) => RuntimeProbe::Warn(format!("/api/health returned {status}")),
            Err(e) => RuntimeProbe::Warn(format!("/api/health probe failed: {e}")),
        }
    } else {
        RuntimeProbe::None
    };
    runtime_status_check("gui_runtime", pid, port, port_open, probe)
}

fn check_proxy_runtime(config: &ReinConfig) -> DoctorCheck {
    let port = config.proxy.port;
    let pid = crate::service::is_running("proxy");
    let port_open = localhost_port_open(port);
    let probe = if port_open {
        match probe_proxy_metrics(port) {
            Ok(Some((requests, errors, extractions))) => RuntimeProbe::Ok(format!(
                "metrics requests={requests} errors={errors} extractions={extractions}"
            )),
            Ok(None) => RuntimeProbe::Warn("/rein/metrics returned malformed JSON".to_string()),
            Err(e) => RuntimeProbe::Warn(format!("/rein/metrics probe failed: {e}")),
        }
    } else {
        RuntimeProbe::None
    };
    runtime_status_check("proxy_runtime", pid, port, port_open, probe)
}

fn localhost_port_open(port: u16) -> bool {
    TcpStream::connect_timeout(
        &SocketAddr::from(([127, 0, 0, 1], port)),
        Duration::from_millis(200),
    )
    .is_ok()
}

fn probe_gui_health(port: u16) -> anyhow::Result<u16> {
    let mut headers = Vec::new();
    if let Ok(token) = std::env::var("REIN_HTTP_TOKEN") {
        headers.push(("Authorization".to_string(), format!("Bearer {token}")));
    }
    let (status, _) = http_get_local(port, "/api/health?limit=1", &headers)?;
    Ok(status)
}

fn probe_proxy_metrics(port: u16) -> anyhow::Result<Option<(u64, u64, u64)>> {
    let mut headers = Vec::new();
    if let Ok(token) =
        std::env::var("REIN_PROXY_TOKEN").or_else(|_| std::env::var("REIN_HTTP_TOKEN"))
    {
        headers.push(("x-rein-token".to_string(), token));
    }
    let (status, body) = http_get_local(port, "/rein/metrics", &headers)?;
    if status == 401 {
        anyhow::bail!("401 unauthorized");
    }
    if status != 200 {
        anyhow::bail!("HTTP {status}");
    }
    let json: serde_json::Value = serde_json::from_str(&body)?;
    let requests = json.get("request_count").and_then(|v| v.as_u64());
    let errors = json.get("error_count").and_then(|v| v.as_u64());
    let extractions = json.get("extraction_count").and_then(|v| v.as_u64());
    Ok(requests
        .zip(errors)
        .zip(extractions)
        .map(|((r, e), x)| (r, e, x)))
}

fn http_get_local(
    port: u16,
    path: &str,
    headers: &[(String, String)],
) -> anyhow::Result<(u16, String)> {
    let mut stream = TcpStream::connect_timeout(
        &SocketAddr::from(([127, 0, 0, 1], port)),
        Duration::from_millis(300),
    )?;
    stream.set_read_timeout(Some(Duration::from_millis(500)))?;
    stream.set_write_timeout(Some(Duration::from_millis(500)))?;

    let mut request =
        format!("GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n");
    for (name, value) in headers {
        request.push_str(name);
        request.push_str(": ");
        request.push_str(value);
        request.push_str("\r\n");
    }
    request.push_str("\r\n");

    stream.write_all(request.as_bytes())?;
    let mut response = String::new();
    stream.read_to_string(&mut response)?;

    let mut lines = response.lines();
    let status_line = lines
        .next()
        .ok_or_else(|| anyhow::anyhow!("empty response"))?;
    let status = status_line
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("missing HTTP status"))?
        .parse::<u16>()?;
    let body = response
        .split("\r\n\r\n")
        .nth(1)
        .unwrap_or_default()
        .to_string();
    Ok((status, body))
}

fn collect_store_snapshot(store: &SqliteStore) -> anyhow::Result<StoreSnapshot> {
    let total_memories = count_sql(store, "SELECT COUNT(*) FROM memories")?;
    let active_memories = count_sql(
        store,
        "SELECT COUNT(*) FROM memories WHERE superseded_by IS NULL AND status = 'active'",
    )?;
    let embed_cache_rows = count_sql(store, "SELECT COUNT(*) FROM embed_cache")?;
    let artifact_rows = count_sql(store, "SELECT COUNT(*) FROM session_artifacts")?;
    Ok(StoreSnapshot {
        total_memories,
        active_memories,
        embed_cache_rows,
        artifact_rows,
    })
}

fn count_sql(store: &SqliteStore, sql: &str) -> anyhow::Result<usize> {
    Ok(store.conn().query_row(sql, [], |row| row.get(0))?)
}

fn check_vector_coverage(
    config: &ReinConfig,
    snapshot: &StoreSnapshot,
    indexed_vectors: Option<usize>,
) -> DoctorCheck {
    if snapshot.total_memories == 0 {
        return ok_in(DoctorCategory::Index, "vector_store", "0 memories");
    }

    let Some(indexed_vectors) = indexed_vectors else {
        let hint = match config.embedding_provider() {
            Provider::None => "embedding provider is disabled",
            Provider::Google if config.embedding.google.api_key.is_none() => {
                "embedding API key is missing"
            }
            _ => "run `rein warmup` or `rein migrate --reindex`",
        };
        return warn_with_hint(
            DoctorCategory::Index,
            "vector_store",
            format!(
                "vector index unavailable for {} memories ({} active, cache={}, artifacts={}); {}",
                snapshot.total_memories,
                snapshot.active_memories,
                snapshot.embed_cache_rows,
                snapshot.artifact_rows,
                hint
            ),
            "run `rein doctor --fix` for local side-index rebuilds, or `rein warmup` / `rein migrate --reindex` if embeddings are missing",
        );
    };

    let coverage = indexed_vectors as f64 / snapshot.total_memories as f64;
    let message = format!(
        "{} indexed vectors for {} memories ({} active, {:.0}% coverage, cache={}, artifacts={})",
        indexed_vectors,
        snapshot.total_memories,
        snapshot.active_memories,
        coverage * 100.0,
        snapshot.embed_cache_rows,
        snapshot.artifact_rows
    );
    if coverage >= 0.9 {
        ok_in(DoctorCategory::Index, "vector_store", message)
    } else {
        warn_with_hint(
            DoctorCategory::Index,
            "vector_store",
            message,
            "run `rein warmup` to fill missing cached embeddings",
        )
    }
}

fn check_tantivy(store: &SqliteStore, active_memories: usize) -> DoctorCheck {
    let db_path = store.db_path();
    let index_path = db_path.with_extension("tantivy");
    let dirty = warmup::tantivy_dirty_path(db_path).exists();

    if active_memories == 0 && !index_path.exists() {
        return ok_in(
            DoctorCategory::Index,
            "tantivy",
            "not built yet (0 active memories)",
        );
    }
    if !index_path.exists() {
        return warn_with_hint(
            DoctorCategory::Index,
            "tantivy",
            format!("index directory missing at {}", index_path.display()),
            "run `rein doctor --fix` or `rein warmup`",
        );
    }
    match TantivyFts::open(&index_path) {
        Ok(_) if dirty => warn_with_hint(
            DoctorCategory::Index,
            "tantivy",
            format!(
                "index opened at {} but dirty marker is present",
                index_path.display()
            ),
            "run `rein doctor --fix` or `rein warmup`",
        ),
        Ok(_) => ok_in(
            DoctorCategory::Index,
            "tantivy",
            format!("index opened at {}", index_path.display()),
        ),
        Err(e) => fail_in(
            DoctorCategory::Index,
            "tantivy",
            format!("failed to open {}: {e}", index_path.display()),
        ),
    }
}

fn inspect_hnsw(store: &SqliteStore, total_memories: usize) -> (DoctorCheck, Option<usize>) {
    let base_path = store.db_path().with_extension("");
    let index_path = base_path.with_extension("usearch");
    let meta_path = base_path.with_extension("usearch.meta");
    let dirty = HnswIndex::is_dirty(&base_path);

    if total_memories == 0 && !index_path.exists() {
        return (
            ok_in(DoctorCategory::Index, "hnsw", "not built yet (0 memories)"),
            Some(0),
        );
    }
    if !index_path.exists() {
        return (
            warn_with_hint(
                DoctorCategory::Index,
                "hnsw",
                format!("index file missing at {}", index_path.display()),
                "run `rein doctor --fix` or `rein warmup`",
            ),
            None,
        );
    }
    if !meta_path.exists() {
        return (
            warn_with_hint(
                DoctorCategory::Index,
                "hnsw",
                format!(
                    "index file exists at {} but metadata is missing at {}",
                    index_path.display(),
                    meta_path.display()
                ),
                "run `rein doctor --fix` or `rein warmup`",
            ),
            None,
        );
    }

    match HnswIndex::open(&base_path, store.dims) {
        Ok(index) => {
            let message = format!(
                "{} vectors indexed at {}",
                index.len(),
                index_path.display()
            );
            if dirty {
                (
                    warn_with_hint(
                        DoctorCategory::Index,
                        "hnsw",
                        format!("{message}; dirty marker is present"),
                        "run `rein doctor --fix` or `rein warmup`",
                    ),
                    Some(index.len()),
                )
            } else {
                (
                    ok_in(DoctorCategory::Index, "hnsw", message),
                    Some(index.len()),
                )
            }
        }
        Err(e) => (
            fail_in(
                DoctorCategory::Index,
                "hnsw",
                format!("failed to open {}: {e}", index_path.display()),
            ),
            None,
        ),
    }
}

fn check_queues(diag: &QueueGroupDiagnostics) -> DoctorCheck {
    let pending = diag.memory.pending + diag.cleanup.pending + diag.dedup.pending;
    let inflight = diag.memory.inflight + diag.cleanup.inflight + diag.dedup.inflight;
    let dead = diag.memory.dead_letters + diag.cleanup.dead_letters + diag.dedup.dead_letters;
    let issues = diag.memory.issues.len() + diag.cleanup.issues.len() + diag.dedup.issues.len();
    let message = format!(
        "memory p{} i{} d{} | cleanup p{} i{} d{} | dedup p{} i{} d{}",
        diag.memory.pending,
        diag.memory.inflight,
        diag.memory.dead_letters,
        diag.cleanup.pending,
        diag.cleanup.inflight,
        diag.cleanup.dead_letters,
        diag.dedup.pending,
        diag.dedup.inflight,
        diag.dedup.dead_letters
    );

    if issues > 0 {
        let first_issue = diag
            .memory
            .issues
            .iter()
            .chain(diag.cleanup.issues.iter())
            .chain(diag.dedup.issues.iter())
            .next()
            .cloned()
            .unwrap_or_else(|| "queue diagnostics failed".to_string());
        warn_with_hint(
            DoctorCategory::Queue,
            "queues",
            format!("{message}; {first_issue}"),
            "inspect and recover the affected queue files before draining workers",
        )
    } else if dead > 0 {
        warn_with_hint(
            DoctorCategory::Queue,
            "queues",
            format!("{message}; dead letters present"),
            "inspect dead-letter files before retrying jobs",
        )
    } else if inflight > 0 {
        warn_with_hint(
            DoctorCategory::Queue,
            "queues",
            format!("{message}; inflight jobs need a worker to finish"),
            "run `rein doctor --fix` or the relevant `rein worker ...` command",
        )
    } else if pending > 0 {
        warn_with_hint(
            DoctorCategory::Queue,
            "queues",
            format!("{message}; pending jobs are waiting to be drained"),
            "run the corresponding `rein worker ...` command",
        )
    } else {
        ok_in(DoctorCategory::Queue, "queues", message)
    }
}

async fn check_embedding_network(config: &ReinConfig) -> DoctorCheck {
    let Some(embedder) = embed::create_embedder(config) else {
        return ok_in(
            DoctorCategory::Network,
            "embedding_network",
            "skipped (embedding provider unavailable)",
        );
    };

    match embedder.embed("rein doctor ping").await {
        Ok(vector) => ok_in(
            DoctorCategory::Network,
            "embedding_network",
            format!(
                "{} responded with {} dimensions",
                embedder.model_name(),
                vector.len()
            ),
        ),
        Err(e) => fail_in(DoctorCategory::Network, "embedding_network", e.to_string()),
    }
}

fn is_loopback_bind(bind: &str) -> bool {
    matches!(bind, "127.0.0.1" | "::1" | "localhost")
}

fn apply_local_fixes(config: &ReinConfig, store: &SqliteStore) -> Vec<String> {
    let mut fixes = Vec::new();

    let tantivy_path = store.db_path().with_extension("tantivy");
    let tantivy_dirty = warmup::tantivy_dirty_path(store.db_path());
    if !tantivy_path.exists() || tantivy_dirty.exists() {
        warmup::populate_tantivy(store);
        fixes.push(format!(
            "triggered Tantivy rebuild at {}",
            tantivy_path.display()
        ));
    }

    let hnsw_base = store.db_path().with_extension("");
    let hnsw_path = hnsw_base.with_extension("usearch");
    let hnsw_meta = hnsw_base.with_extension("usearch.meta");
    if !hnsw_path.exists() || !hnsw_meta.exists() || HnswIndex::is_dirty(&hnsw_base) {
        warmup::populate_hnsw(store, config);
        fixes.push(format!("triggered HNSW rebuild at {}", hnsw_path.display()));
    }

    fixes
}

fn apply_queue_fixes(config: &ReinConfig) -> Vec<String> {
    let mut fixes = Vec::new();
    for (queue_name, prefix) in [
        ("memory", "memory_queue"),
        ("cleanup", "cleanup_queue"),
        ("dedup", "dedup_queue"),
    ] {
        let queue = queue_scoped_path(config, prefix);
        let inflight = queue_scoped_path(config, &format!("{prefix}_inflight"));
        if let Ok(Some(recovered)) = recover_inflight_file(&queue, &inflight) {
            fixes.push(format!("recovered {recovered} {queue_name} inflight jobs"));
        }
    }
    fixes
}

fn queue_scoped_path(config: &ReinConfig, prefix: &str) -> std::path::PathBuf {
    let base = buffer::resolve_buffer_dir(config);
    let db_tag = {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        config.resolve_db_path().hash(&mut h);
        format!("{:016x}", h.finish())
    };
    let queue_dir = base.join("queue").join(db_tag);
    let _ = std::fs::create_dir_all(&queue_dir);
    queue_dir.join(format!("{prefix}.jsonl"))
}

fn recover_inflight_file(
    queue_path: &std::path::Path,
    inflight_path: &std::path::Path,
) -> anyhow::Result<Option<usize>> {
    let content = match std::fs::read_to_string(inflight_path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e.into()),
    };

    let recovered = content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .count();
    if recovered > 0 {
        if let Some(parent) = queue_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(queue_path)?;
        file.write_all(content.as_bytes())?;
        if !content.ends_with('\n') {
            file.write_all(b"\n")?;
        }
    }
    let _ = std::fs::remove_file(inflight_path);
    Ok(Some(recovered))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::path::{Path, PathBuf};
    use std::sync::OnceLock;
    use std::thread;

    use crate::extract::hooks::buffer;
    use crate::types::{Importance, Memory, MemoryLayer, MemoryStatus, Source};

    fn test_memory(topic: &str, summary: &str, content: &str) -> Memory {
        Memory {
            id: ulid::Ulid::new().to_string(),
            layer: MemoryLayer::STM,
            topic: topic.to_string(),
            summary: summary.to_string(),
            content: content.to_string(),
            keywords: vec![],
            importance: Importance::Medium,
            source: Source::Manual,
            strength: 1.0,
            decay_lambda: 0.06,
            access_count: 0,
            superseded_by: None,
            canonical_id: None,
            support_count: 1,
            merge_count: 0,
            dedup_confidence: 1.0,
            source_diversity: 1.0,
            contradiction_score: 0.0,
            related_ids: vec![],
            concept_ids: vec![],
            status: MemoryStatus::Active,
            embedding: None,
            tier: crate::types::MemoryTier::Warm,
            cluster_id: None,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            last_accessed: chrono::Utc::now(),
        }
    }

    fn temp_config(tempdir: &tempfile::TempDir) -> ReinConfig {
        let toml = format!(
            r#"
[database]
path = "{}"

[embedding]
provider = "none"
dimensions = 3072
[embedding.google]
model = "text-embedding-004"
[embedding.omlx]
endpoint = "http://127.0.0.1:11434/v1/embeddings"

[search]
rrf_k = 60.0
rrf_fts_weight = 1.0
rrf_vec_weight = 1.0
dedup_similarity = 0.7
dedup_time_window_days = 7

[chunking]
max_tokens = 400
overlap_percent = 15
metadata_prefix = true

[sync]
supermemory_enabled = false
auto_memory_enabled = false
auto_memory_glob = "~/.claude/projects/**/*.md"

[decay]
base_lambda = 0.06
prune_threshold = 0.3

[server]
sse_enabled = true
sse_bind = "0.0.0.0"
sse_port = 8765
compact = false
gui_enabled = false
allow_unauthenticated_loopback = false

[hooks]
buffer_dir = "{}"

[extract]
provider = "none"
[extract.google]
model = "gemini-2.5-flash-lite"
[extract.omlx]
endpoint = "http://127.0.0.1:11434/v1/chat/completions"

[adaptive]
enabled = true

[query_expansion]
provider = "none"
[query_expansion.google]
model = "gemini-2.5-flash-lite"
[query_expansion.omlx]
endpoint = "http://127.0.0.1:11434/v1/chat/completions"

[proxy]
bind = "0.0.0.0"
port = 8777
allow_unauthenticated_loopback = false

[async_memory]
provider = "inherit"

[cleanup]
"#,
            tempdir.path().join("doctor.db").display(),
            tempdir.path().display()
        );
        ReinConfig::load_from_str(&toml).unwrap()
    }

    fn queue_file(config: &ReinConfig, prefix: &str) -> PathBuf {
        let base = buffer::resolve_buffer_dir(config);
        let db_tag = {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            config.resolve_db_path().hash(&mut h);
            format!("{:016x}", h.finish())
        };
        base.join("queue")
            .join(db_tag)
            .join(format!("{prefix}.jsonl"))
    }

    fn write_lines(path: &Path, count: usize) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let body = (0..count).map(|_| "{}").collect::<Vec<_>>().join("\n");
        std::fs::write(path, format!("{body}\n")).unwrap();
    }

    fn spawn_http_server(response: String) -> u16 {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut buf = [0u8; 1024];
                let _ = stream.read(&mut buf);
                let _ = stream.write_all(response.as_bytes());
            }
        });
        port
    }

    /// Async mutex so env-var tests can await safely without tripping
    /// `clippy::await_holding_lock`. All three doctor tests are `#[tokio::test]`.
    fn env_lock() -> &'static tokio::sync::Mutex<()> {
        static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
    }

    struct EnvRestore {
        key: &'static str,
        value: Option<String>,
    }

    impl Drop for EnvRestore {
        fn drop(&mut self) {
            match &self.value {
                Some(value) => std::env::set_var(self.key, value),
                None => std::env::remove_var(self.key),
            }
        }
    }

    #[test]
    fn test_format_human_reports_overall_status() {
        let report = DoctorReport {
            status: ReportStatus::Degraded,
            checks: vec![
                ok_in(DoctorCategory::Storage, "database", "connected"),
                warn_with_hint(
                    DoctorCategory::Queue,
                    "queues",
                    "pending jobs",
                    "run `rein worker memory`",
                ),
            ],
            fixes_applied: vec![],
        };
        let text = format_human(&report);
        assert!(text.contains("[OK] database: connected"));
        assert!(text.contains("[WARN] queues: pending jobs"));
        assert!(text.contains("repair: run `rein worker memory`"));
        assert!(text.contains("Overall: degraded"));
    }

    #[test]
    fn test_json_serializes_repair_hint_only_when_present() {
        let report = DoctorReport {
            status: ReportStatus::Degraded,
            checks: vec![
                ok_in(DoctorCategory::Storage, "database", "connected"),
                warn_with_hint(
                    DoctorCategory::Queue,
                    "queues",
                    "pending jobs",
                    "run `rein worker memory`",
                ),
            ],
            fixes_applied: vec![],
        };
        let json = serde_json::to_value(&report).unwrap();
        let checks = json.get("checks").and_then(|v| v.as_array()).unwrap();
        assert!(checks[0].get("repair_hint").is_none());
        assert_eq!(
            checks[0].get("category").and_then(|v| v.as_str()),
            Some("storage")
        );
        assert_eq!(
            checks[0].get("severity").and_then(|v| v.as_str()),
            Some("info")
        );
        assert_eq!(
            checks[0].get("fixable").and_then(|v| v.as_bool()),
            Some(false)
        );
        assert_eq!(
            checks[1].get("category").and_then(|v| v.as_str()),
            Some("queue")
        );
        assert_eq!(
            checks[1].get("severity").and_then(|v| v.as_str()),
            Some("warning")
        );
        assert_eq!(
            checks[1].get("fixable").and_then(|v| v.as_bool()),
            Some(true)
        );
        assert_eq!(
            checks[1].get("repair_hint").and_then(|v| v.as_str()),
            Some("run `rein worker memory`")
        );
    }

    #[test]
    fn test_runtime_status_check_surfaces_probe_details_for_external_listener() {
        let check = runtime_status_check(
            "gui_runtime",
            None,
            8765,
            true,
            RuntimeProbe::Warn("/api/health returned 401".to_string()),
        );
        assert_eq!(check.status, CheckStatus::Warn);
        assert!(check.message.contains("external listener"));
        assert!(check.message.contains("401"));
    }

    #[test]
    fn test_probe_gui_health_reads_http_status() {
        let port = spawn_http_server(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}"
                .to_string(),
        );
        let status = probe_gui_health(port).unwrap();
        assert_eq!(status, 200);
    }

    #[test]
    fn test_probe_proxy_metrics_parses_counts() {
        let port = spawn_http_server(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 58\r\n\r\n{\"request_count\":12,\"error_count\":1,\"extraction_count\":5}"
                .to_string(),
        );
        let metrics = probe_proxy_metrics(port).unwrap();
        assert_eq!(metrics, Some((12, 1, 5)));
    }

    #[test]
    fn test_recover_inflight_file_moves_jobs_back_to_queue() {
        let dir = tempfile::tempdir().unwrap();
        let queue = dir.path().join("memory_queue.jsonl");
        let inflight = dir.path().join("memory_queue_inflight.jsonl");
        std::fs::write(&inflight, "{\"job\":1}\n{\"job\":2}\n").unwrap();

        let recovered = recover_inflight_file(&queue, &inflight).unwrap();
        assert_eq!(recovered, Some(2));
        assert!(!inflight.exists());
        let queue_text = std::fs::read_to_string(&queue).unwrap();
        assert!(queue_text.contains("{\"job\":1}"));
        assert!(queue_text.contains("{\"job\":2}"));
    }

    #[tokio::test]
    #[serial_test::serial(global_state)]
    async fn test_doctor_fix_reports_applied_repairs() {
        let _guard = env_lock().lock().await;
        let _restore_http = EnvRestore {
            key: "REIN_HTTP_TOKEN",
            value: std::env::var("REIN_HTTP_TOKEN").ok(),
        };
        let _restore_proxy = EnvRestore {
            key: "REIN_PROXY_TOKEN",
            value: std::env::var("REIN_PROXY_TOKEN").ok(),
        };
        std::env::set_var("REIN_HTTP_TOKEN", "doctor-test-token");
        std::env::set_var("REIN_PROXY_TOKEN", "doctor-test-token");
        let tempdir = tempfile::tempdir().unwrap();
        let config = temp_config(&tempdir);
        let store = config.open_store().unwrap();
        let memory = test_memory("doctor", "doctor memory", "doctor content");
        let enriched =
            crate::embed::prepend_metadata(&memory.topic, &memory.summary, &memory.content);
        let model = config.embedding_model();
        let vector = vec![0.1f32; config.embedding.dimensions];
        crate::embed::EmbedCache::put(store.conn(), &enriched, &model, &vector).unwrap();
        store.store(memory).unwrap();

        let tantivy_dirty = warmup::tantivy_dirty_path(store.db_path());
        std::fs::create_dir_all(tantivy_dirty.parent().unwrap()).unwrap();
        std::fs::write(&tantivy_dirty, b"dirty").unwrap();
        let hnsw_dirty = store
            .db_path()
            .with_extension("")
            .with_extension("usearch.dirty");
        std::fs::write(&hnsw_dirty, b"dirty").unwrap();
        let inflight = queue_file(&config, "memory_queue_inflight");
        std::fs::create_dir_all(inflight.parent().unwrap()).unwrap();
        std::fs::write(&inflight, "{\"job\":1}\n").unwrap();

        let report = run(
            &config,
            DoctorOptions {
                network: false,
                fix: true,
            },
        )
        .await;

        assert!(report
            .fixes_applied
            .iter()
            .any(|item| item.contains("Tantivy")));
        assert!(report
            .fixes_applied
            .iter()
            .any(|item| item.contains("HNSW")));
        assert!(report
            .fixes_applied
            .iter()
            .any(|item| item.contains("memory inflight")));
        if report.has_failures() {
            // v0.19 root-fix: dump the actual failing check instead of a bare
            // `!has_failures()` assertion. This turned an opaque ~20% parallel
            // flake into an actionable diagnostic.
            let failures: Vec<String> = report
                .checks
                .iter()
                .filter(|c| matches!(c.status, CheckStatus::Fail))
                .map(|c| format!("{}: {}", c.name, c.message))
                .collect();
            panic!(
                "doctor test unexpectedly reported {} failures under parallel load: {:#?}",
                failures.len(),
                failures
            );
        }
        assert_eq!(report.status, ReportStatus::Degraded);
    }

    #[tokio::test]
    #[serial_test::serial(global_state)]
    async fn test_doctor_flags_auth_and_queue_warnings() {
        let _guard = env_lock().lock().await;
        let _restore_http = EnvRestore {
            key: "REIN_HTTP_TOKEN",
            value: std::env::var("REIN_HTTP_TOKEN").ok(),
        };
        let _restore_proxy = EnvRestore {
            key: "REIN_PROXY_TOKEN",
            value: std::env::var("REIN_PROXY_TOKEN").ok(),
        };
        std::env::remove_var("REIN_HTTP_TOKEN");
        std::env::remove_var("REIN_PROXY_TOKEN");

        let tempdir = tempfile::tempdir().unwrap();
        let config = temp_config(&tempdir);
        let store = config.open_store().unwrap();
        store
            .store(test_memory("doctor", "doctor memory", "doctor content"))
            .unwrap();

        let tantivy_dirty = warmup::tantivy_dirty_path(store.db_path());
        std::fs::create_dir_all(tantivy_dirty.parent().unwrap()).unwrap();
        std::fs::write(&tantivy_dirty, b"dirty").unwrap();
        let hnsw_dirty = store
            .db_path()
            .with_extension("")
            .with_extension("usearch.dirty");
        std::fs::write(&hnsw_dirty, b"dirty").unwrap();

        write_lines(&queue_file(&config, "memory_queue"), 2);
        write_lines(&queue_file(&config, "memory_queue_dead"), 1);

        let report = run(&config, DoctorOptions::default()).await;
        assert!(report.has_failures());

        let http = report
            .checks
            .iter()
            .find(|c| c.name == "http_auth")
            .unwrap();
        assert_eq!(http.status, CheckStatus::Fail);

        let proxy = report
            .checks
            .iter()
            .find(|c| c.name == "proxy_auth")
            .unwrap();
        assert_eq!(proxy.status, CheckStatus::Fail);

        let queues = report.checks.iter().find(|c| c.name == "queues").unwrap();
        assert_eq!(queues.status, CheckStatus::Warn);
        assert!(queues.message.contains("memory p2"));

        let tantivy = report.checks.iter().find(|c| c.name == "tantivy").unwrap();
        assert_eq!(tantivy.status, CheckStatus::Warn);
    }

    #[tokio::test]
    #[serial_test::serial(global_state)]
    async fn test_doctor_treats_empty_tokens_as_missing() {
        let _guard = env_lock().lock().await;
        let _restore_http = EnvRestore {
            key: "REIN_HTTP_TOKEN",
            value: std::env::var("REIN_HTTP_TOKEN").ok(),
        };
        let _restore_proxy = EnvRestore {
            key: "REIN_PROXY_TOKEN",
            value: std::env::var("REIN_PROXY_TOKEN").ok(),
        };
        std::env::set_var("REIN_HTTP_TOKEN", "");
        std::env::set_var("REIN_PROXY_TOKEN", "   ");

        let tempdir = tempfile::tempdir().unwrap();
        let config = temp_config(&tempdir);
        let report = run(&config, DoctorOptions::default()).await;

        let http = report
            .checks
            .iter()
            .find(|check| check.name == "http_auth")
            .unwrap();
        let proxy = report
            .checks
            .iter()
            .find(|check| check.name == "proxy_auth")
            .unwrap();
        assert_eq!(http.status, CheckStatus::Fail);
        assert_eq!(proxy.status, CheckStatus::Fail);
    }

    #[test]
    fn test_architecture_drift_helpers_match_source_counts() {
        let derived = count_cli_operations_in_source(include_str!("main.rs"));
        let inventory_count = inventory::iter::<crate::ops::OpsCliEntry>().count();
        assert_eq!(derived + inventory_count, registry::cli_operations().len());
        assert_eq!(
            count_mcp_tools_in_source(include_str!("mcp/server.rs")),
            registry::mcp_operations().len()
        );
        let derived_rest = count_rest_operations_in_source(include_str!("mcp/rest.rs"));
        let inventory_rest = inventory::iter::<crate::ops::OpsRestEntry>().count();
        assert_eq!(derived_rest + inventory_rest, registry::rest_operations().len());
        assert_eq!(
            parse_agents_overview_version("rein v1.2.3 — demo release"),
            Some("1.2.3".to_string())
        );
        assert_eq!(
            parse_documented_mcp_tool_count_line("| **28 MCP tools** | 13 core"),
            Some(28)
        );
    }
}
