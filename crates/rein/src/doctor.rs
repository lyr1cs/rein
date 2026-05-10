use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Serialize;

use crate::config::{Provider, ReinConfig};
use crate::embed;
use crate::extract::hooks::buffer;
use crate::extract::hooks::queue::{collect_queue_diagnostics, QueueGroupDiagnostics};
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
    checks.push(check_auth_policy_consistency(config));
    checks.push(check_oauth_provider(config));
    checks.push(check_proxy_auth(config));
    checks.push(check_codex_hooks());
    checks.push(check_codex_mcp_server(config));
    checks.push(check_gui_runtime(config));
    checks.push(check_proxy_runtime(config));
    checks.push(check_overview_version());
    checks.push(check_release_metadata_versions());
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
                            checks.push(check_resummerize(&store));
                            checks.push(check_pool_saturation(config));
                            checks.push(check_ars_parameter_policy(&store, config));
                            // v0.27.x judge checks
                            checks.push(check_judge_calibration(&store, config));
                            checks.push(check_judge_call_ledger(&store, config));
                            checks.push(check_judge_cache_size(config));
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

fn check_release_metadata_versions() -> DoctorCheck {
    let cargo_version = env!("CARGO_PKG_VERSION");
    let mismatches = release_metadata_versions()
        .into_iter()
        .filter_map(|(name, version)| (version != cargo_version).then_some((name, version)))
        .collect::<Vec<_>>();

    if mismatches.is_empty() {
        return ok_in(
            DoctorCategory::Architecture,
            "release_metadata_versions",
            format!("release metadata versions match Cargo v{cargo_version}"),
        );
    }

    let summary = mismatches
        .iter()
        .map(|(name, version)| format!("{name}={version}"))
        .collect::<Vec<_>>()
        .join(", ");
    warn_with_hint(
        DoctorCategory::Architecture,
        "release_metadata_versions",
        format!("Cargo.toml is v{cargo_version}, but release metadata says {summary}"),
        "update DXT and Claude plugin manifest versions before publishing",
    )
}

fn release_metadata_versions() -> Vec<(&'static str, String)> {
    let mut versions = Vec::new();
    collect_json_version(
        &mut versions,
        "dxt/manifest.json",
        include_str!("../../../dxt/manifest.json"),
        &["version"],
    );
    collect_json_version(
        &mut versions,
        ".claude-plugin/marketplace.json",
        include_str!("../../../.claude-plugin/marketplace.json"),
        &["version"],
    );
    collect_json_version(
        &mut versions,
        ".claude-plugin/marketplace.json plugins[0]",
        include_str!("../../../.claude-plugin/marketplace.json"),
        &["plugins", "0", "version"],
    );
    collect_json_version(
        &mut versions,
        "plugins/rein/.claude-plugin/plugin.json",
        include_str!("../../../plugins/rein/.claude-plugin/plugin.json"),
        &["version"],
    );
    versions
}

fn collect_json_version(
    versions: &mut Vec<(&'static str, String)>,
    name: &'static str,
    text: &str,
    path: &[&str],
) {
    let Ok(root) = serde_json::from_str::<serde_json::Value>(text) else {
        versions.push((name, "<invalid-json>".to_string()));
        return;
    };
    let mut value = &root;
    for key in path {
        value = if let Ok(idx) = key.parse::<usize>() {
            value
                .as_array()
                .and_then(|items| items.get(idx))
                .unwrap_or(&serde_json::Value::Null)
        } else {
            value.get(*key).unwrap_or(&serde_json::Value::Null)
        };
    }
    versions.push((
        name,
        value.as_str().unwrap_or("<missing-version>").to_string(),
    ));
}

// Phase 3 dropped the hand-maintained `ops::registry::*_OPERATIONS` arrays
// that used to sit between inventory and the drift check. Now the checks
// compare inventory counts (authoritative) against source-scanned derived
// counts directly. The MCP check retains the doc-drift warning against
// README / AGENTS.md MCP tool counts.

fn check_cli_registry() -> DoctorCheck {
    let duplicates = crate::ops::inventory::duplicate_report();
    if !duplicates.cli_names.is_empty() {
        return fail_in(
            DoctorCategory::Architecture,
            "cli_registry",
            format!(
                "CLI inventory duplicates detected: {}",
                duplicates.cli_names.join(", ")
            ),
        );
    }
    let inventory_count = inventory::iter::<crate::ops::OpsCliEntry>().count();
    let derived_count = count_cli_operations_in_source(include_str!("main.rs"));
    let source_count = derived_count + inventory_count;
    ok_in(
        DoctorCategory::Architecture,
        "cli_registry",
        format!(
            "{source_count} CLI operations in source ({derived_count} derived + {inventory_count} inventory)"
        ),
    )
}

fn check_mcp_registry() -> DoctorCheck {
    let duplicates = crate::ops::inventory::duplicate_report();
    if !duplicates.mcp_names.is_empty() {
        return fail_in(
            DoctorCategory::Architecture,
            "mcp_registry",
            format!(
                "MCP inventory duplicates detected: {}",
                duplicates.mcp_names.join(", ")
            ),
        );
    }
    let inventory_count = inventory::iter::<crate::ops::OpsMcpEntry>().count();
    let derived_count = count_mcp_tools_in_source(include_str!("mcp/server.rs"));
    let source_count = derived_count + inventory_count;

    let doc_counts = documented_mcp_tool_counts();
    if !doc_counts.is_empty() && doc_counts.iter().any(|(_, count)| *count != source_count) {
        let doc_summary = doc_counts
            .iter()
            .map(|(name, count)| format!("{name}={count}"))
            .collect::<Vec<_>>()
            .join(", ");
        return warn_in(
            DoctorCategory::Architecture,
            "mcp_registry",
            format!(
                "{source_count} MCP tools in source ({derived_count} derived + {inventory_count} inventory), uniqueness clean, but docs still say {doc_summary}"
            ),
        );
    }

    ok_in(
        DoctorCategory::Architecture,
        "mcp_registry",
        format!(
            "{source_count} MCP tools in source ({derived_count} derived + {inventory_count} inventory), uniqueness clean"
        ),
    )
}

fn check_rest_registry() -> DoctorCheck {
    let duplicates = crate::ops::inventory::duplicate_report();
    if !duplicates.rest_routes.is_empty() {
        return fail_in(
            DoctorCategory::Architecture,
            "rest_registry",
            format!(
                "REST inventory duplicates detected: {}",
                duplicates.rest_routes.join(", ")
            ),
        );
    }
    // Exclude test-support ops (op_name starts with "__test_") from the count
    // so the check is stable whether or not the test-support feature is active.
    let inventory_count = inventory::iter::<crate::ops::OpsRestEntry>()
        .filter(|e| !e.op_name.starts_with("__test_"))
        .count();
    let derived_count = count_rest_operations_in_source(include_str!("mcp/rest.rs"));
    let source_count = derived_count + inventory_count;
    ok_in(
        DoctorCategory::Architecture,
        "rest_registry",
        format!(
            "{source_count} REST operations in source ({derived_count} derived + {inventory_count} inventory), uniqueness clean"
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

fn check_codex_hooks() -> DoctorCheck {
    let Some(codex_dir) = std::env::var_os("HOME")
        .map(PathBuf::from)
        .map(|p| p.join(".codex"))
    else {
        return ok_in(
            DoctorCategory::Configuration,
            "codex_hooks",
            "HOME not set; skipping Codex hook checks",
        );
    };
    check_codex_hooks_at(&codex_dir)
}

fn check_codex_mcp_server(config: &ReinConfig) -> DoctorCheck {
    let Some(codex_dir) = std::env::var_os("HOME")
        .map(PathBuf::from)
        .map(|p| p.join(".codex"))
    else {
        return ok_in(
            DoctorCategory::Configuration,
            "codex_mcp",
            "HOME not set; skipping Codex MCP checks",
        );
    };
    check_codex_mcp_server_at(&codex_dir, &config.resolve_db_path())
}

fn check_codex_mcp_server_at(codex_dir: &Path, local_db_path: &Path) -> DoctorCheck {
    let config_path = codex_dir.join("config.toml");
    let content = match std::fs::read_to_string(&config_path) {
        Ok(content) => content,
        Err(_) => {
            return ok_in(
                DoctorCategory::Configuration,
                "codex_mcp",
                "Codex config not found; skipping Codex MCP checks",
            );
        }
    };
    let parsed: toml::Value = match toml::from_str(&content) {
        Ok(value) => value,
        Err(_) => {
            return warn_with_hint(
                DoctorCategory::Configuration,
                "codex_mcp",
                "Codex config.toml is not valid TOML",
                "fix ~/.codex/config.toml or rerun rein init",
            );
        }
    };

    let mut entries = Vec::new();
    if let Some(entry) = parsed
        .get("mcp_servers")
        .and_then(|v| v.get("rein"))
        .and_then(|v| v.as_table())
    {
        entries.push(("[mcp_servers.rein]", entry));
    }
    if let Some(entry) = parsed
        .get("mcp")
        .and_then(|v| v.get("rein"))
        .and_then(|v| v.as_table())
    {
        entries.push(("[mcp.rein]", entry));
    }

    if entries.is_empty() {
        return warn_with_hint(
            DoctorCategory::Configuration,
            "codex_mcp",
            "Codex rein MCP server is not configured",
            "run rein init",
        );
    }

    let mut ok_messages = Vec::new();
    for (label, entry) in entries {
        match check_codex_mcp_entry(label, entry, local_db_path) {
            Ok(message) => ok_messages.push(message),
            Err(check) => return check,
        }
    }

    ok_in(
        DoctorCategory::Configuration,
        "codex_mcp",
        format!("Codex rein MCP entries healthy: {}", ok_messages.join("; ")),
    )
}

fn check_codex_mcp_entry(
    label: &str,
    entry: &toml::map::Map<String, toml::Value>,
    local_db_path: &Path,
) -> Result<String, DoctorCheck> {
    if let Some(url) = entry.get("url").and_then(|v| v.as_str()) {
        return check_codex_mcp_url(label, url);
    }

    let Some(command) = entry.get("command").and_then(|v| v.as_str()) else {
        return Err(warn_with_hint(
            DoctorCategory::Configuration,
            "codex_mcp",
            format!("Codex {label} entry has neither url nor command"),
            "run rein init or edit ~/.codex/config.toml",
        ));
    };
    let command = command.trim();
    if command.is_empty() {
        return Err(warn_with_hint(
            DoctorCategory::Configuration,
            "codex_mcp",
            format!("Codex {label} command is empty"),
            "run rein init or set command = \"rein\"",
        ));
    }

    validate_codex_mcp_stdio_args(label, entry)?;
    if let Some(check) = check_codex_mcp_process_context(label, entry, local_db_path) {
        return Err(check);
    }
    if let Some(rein_db) = codex_mcp_rein_db_override(entry) {
        if let Some(check) = check_codex_mcp_rein_db_override(label, rein_db, local_db_path) {
            return Err(check);
        }
    }

    Ok(format!("{label} stdio command `{command}`"))
}

fn validate_codex_mcp_stdio_args(
    label: &str,
    entry: &toml::map::Map<String, toml::Value>,
) -> Result<(), DoctorCheck> {
    let Some(args) = entry.get("args").and_then(|v| v.as_array()) else {
        return Err(warn_with_hint(
            DoctorCategory::Configuration,
            "codex_mcp",
            format!("Codex {label} does not set args = [\"serve\"]"),
            "run rein init or set args = [\"serve\"] for stdio MCP",
        ));
    };
    let mut arg_strings = Vec::with_capacity(args.len());
    for arg in args {
        let Some(arg) = arg.as_str() else {
            return Err(warn_with_hint(
                DoctorCategory::Configuration,
                "codex_mcp",
                format!("Codex {label} args must be strings"),
                "run rein init or set args = [\"serve\"]",
            ));
        };
        arg_strings.push(arg);
    }

    if arg_strings.first().copied() != Some("serve") {
        return Err(warn_with_hint(
            DoctorCategory::Configuration,
            "codex_mcp",
            format!("Codex {label} does not start rein with `serve`"),
            "run rein init or set args = [\"serve\"]",
        ));
    }

    if let Some(flag) = arg_strings
        .iter()
        .skip(1)
        .find(|arg| matches!(**arg, "--sse" | "--gui" | "--proxy"))
    {
        return Err(warn_with_hint(
            DoctorCategory::Configuration,
            "codex_mcp",
            format!("Codex {label} uses `rein serve {flag}`, which is not stdio MCP"),
            "remove --sse/--gui/--proxy from the Codex MCP args; remote HTTP MCP belongs in a url entry",
        ));
    }

    Ok(())
}

fn codex_mcp_rein_db_override(entry: &toml::map::Map<String, toml::Value>) -> Option<&str> {
    codex_mcp_env_value(entry, "REIN_DB")
}

fn codex_mcp_env_value<'a>(
    entry: &'a toml::map::Map<String, toml::Value>,
    key: &str,
) -> Option<&'a str> {
    entry
        .get("env")
        .and_then(|v| v.as_table())
        .and_then(|env| env.get(key))
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

fn check_codex_mcp_process_context(
    label: &str,
    entry: &toml::map::Map<String, toml::Value>,
    local_db_path: &Path,
) -> Option<DoctorCheck> {
    if let Some(rein_config) = codex_mcp_env_value(entry, "REIN_CONFIG") {
        return Some(warn_with_hint(
            DoctorCategory::Configuration,
            "codex_mcp",
            format!(
                "Codex {label} sets REIN_CONFIG={rein_config}; it may load a different config/database than local `rein` CLI ({})",
                local_db_path.display()
            ),
            "remove the Codex MCP REIN_CONFIG override, or verify it resolves to the same database as `rein config`",
        ));
    }

    if let Some(home) = codex_mcp_env_value(entry, "HOME") {
        return Some(warn_with_hint(
            DoctorCategory::Configuration,
            "codex_mcp",
            format!(
                "Codex {label} sets HOME={home}; `database.path = \"auto\"` may resolve to a different database than local `rein` CLI ({})",
                local_db_path.display()
            ),
            "remove the Codex MCP HOME override, or set an explicit absolute REIN_DB that matches `rein config`",
        ));
    }

    if local_db_path.is_relative() {
        if let Some(cwd) = entry.get("cwd").and_then(|v| v.as_str()) {
            return Some(warn_with_hint(
                DoctorCategory::Configuration,
                "codex_mcp",
                format!(
                    "local rein database path is relative ({}) and Codex {label} sets cwd={cwd}; CLI and MCP recall may use different databases",
                    local_db_path.display()
                ),
                "set [database].path or [mcp_servers.rein.env].REIN_DB to an absolute path",
            ));
        }
    }

    None
}

fn check_codex_mcp_rein_db_override(
    label: &str,
    rein_db: &str,
    local_db_path: &Path,
) -> Option<DoctorCheck> {
    let override_path = PathBuf::from(rein_db);
    if override_path.is_relative() {
        return Some(warn_with_hint(
            DoctorCategory::Configuration,
            "codex_mcp",
            format!(
                "Codex {label} sets relative REIN_DB={rein_db}; recall may use a different database than local `rein` CLI ({}) depending on Codex cwd",
                local_db_path.display()
            ),
            "use an absolute REIN_DB path that matches `rein config`, or remove the Codex MCP env override",
        ));
    }

    if !paths_equivalent(&override_path, local_db_path) {
        return Some(warn_with_hint(
            DoctorCategory::Configuration,
            "codex_mcp",
            format!(
                "Codex {label} sets REIN_DB={} but local `rein` CLI uses {}; recall may use a different database",
                override_path.display(),
                local_db_path.display()
            ),
            "update ~/.codex/config.toml so [mcp_servers.rein.env].REIN_DB matches `rein config`, or remove the override",
        ));
    }

    None
}

fn paths_equivalent(left: &Path, right: &Path) -> bool {
    match (left.canonicalize(), right.canonicalize()) {
        (Ok(left), Ok(right)) => left == right,
        _ => left == right,
    }
}

fn check_codex_mcp_url(label: &str, url: &str) -> Result<String, DoctorCheck> {
    let uri: hyper::Uri = match url.parse() {
        Ok(uri) => uri,
        Err(_) => {
            return Err(warn_with_hint(
                DoctorCategory::Configuration,
                "codex_mcp",
                format!("Codex {label} url is not a valid URI: {url}"),
                "fix ~/.codex/config.toml or rerun rein init",
            ));
        }
    };
    let host = uri.host().unwrap_or_default();
    if is_loopback_http_host(host) {
        Ok(format!("{label} loopback HTTP endpoint {url}"))
    } else {
        Err(warn_with_hint(
            DoctorCategory::Configuration,
            "codex_mcp",
            format!(
                "Codex {label} points at non-loopback HTTP endpoint {url}; \
                 recall may use a different machine/database than local `rein` CLI"
            ),
            "use stdio (`command = \"rein\", args = [\"serve\"]`) for local memories, or verify the remote endpoint is intentional",
        ))
    }
}

fn is_loopback_http_host(host: &str) -> bool {
    let normalized = host
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .to_ascii_lowercase();
    if normalized == "localhost" {
        return true;
    }
    normalized
        .parse::<std::net::IpAddr>()
        .is_ok_and(|ip| ip.is_loopback())
}

fn check_codex_hooks_at(codex_dir: &Path) -> DoctorCheck {
    let config_path = codex_dir.join("config.toml");
    if !config_path.exists() {
        return ok_in(
            DoctorCategory::Configuration,
            "codex_hooks",
            "Codex config not found; skipping Codex hook checks",
        );
    }

    // Codex 0.129+ uses `[features].hooks`. Older releases used `codex_hooks`.
    // Accept either key as the enabling signal so this check works across the
    // rename window. `rein init` writes the new key going forward.
    let feature_enabled = std::fs::read_to_string(&config_path)
        .ok()
        .and_then(|content| toml::from_str::<toml::Value>(&content).ok())
        .and_then(|root| {
            root.get("features").and_then(|features| {
                features
                    .get("hooks")
                    .and_then(|enabled| enabled.as_bool())
                    .or_else(|| {
                        features
                            .get("codex_hooks")
                            .and_then(|enabled| enabled.as_bool())
                    })
            })
        })
        == Some(true);
    if !feature_enabled {
        return warn_with_hint(
            DoctorCategory::Configuration,
            "codex_hooks",
            "Codex [features].hooks = true is not configured",
            "run rein init",
        );
    }

    let hooks_path = codex_dir.join("hooks.json");
    let root = match std::fs::read_to_string(&hooks_path) {
        Ok(content) => match serde_json::from_str::<serde_json::Value>(&content) {
            Ok(value) => value,
            Err(_) => {
                return warn_with_hint(
                    DoctorCategory::Configuration,
                    "codex_hooks",
                    "Codex hooks.json is not valid JSON",
                    "run rein init",
                );
            }
        },
        Err(_) => {
            return warn_with_hint(
                DoctorCategory::Configuration,
                "codex_hooks",
                "Codex hooks.json not found",
                "run rein init",
            );
        }
    };

    let Some(root_obj) = root.as_object() else {
        return warn_with_hint(
            DoctorCategory::Configuration,
            "codex_hooks",
            "Codex hooks.json is not a JSON object",
            "run rein init",
        );
    };
    let hooks = root_obj
        .get("hooks")
        .and_then(|hooks| hooks.as_object())
        .unwrap_or(root_obj);

    let missing = expected_codex_hook_commands()
        .iter()
        .filter_map(|(event, command)| {
            if codex_event_has_command(hooks.get(*event), command) {
                None
            } else {
                Some(*event)
            }
        })
        .collect::<Vec<_>>();

    if missing.is_empty() {
        ok_in(
            DoctorCategory::Configuration,
            "codex_hooks",
            "six hooks configured",
        )
    } else {
        warn_with_hint(
            DoctorCategory::Configuration,
            "codex_hooks",
            format!("missing Codex Rein hook events: {}", missing.join(", ")),
            "run rein init",
        )
    }
}

fn expected_codex_hook_commands() -> &'static [(&'static str, &'static str)] {
    &[
        (
            "SessionStart",
            "REIN_AGENT_LABEL=codex rein hook session-start",
        ),
        ("PreToolUse", "REIN_AGENT_LABEL=codex rein hook pre"),
        (
            "PermissionRequest",
            "REIN_AGENT_LABEL=codex rein hook permission",
        ),
        ("PostToolUse", "REIN_AGENT_LABEL=codex rein hook post"),
        (
            "UserPromptSubmit",
            "REIN_AGENT_LABEL=codex rein hook prompt",
        ),
        ("Stop", "REIN_AGENT_LABEL=codex rein hook stop"),
    ]
}

fn codex_event_has_command(event_hooks: Option<&serde_json::Value>, command: &str) -> bool {
    event_hooks
        .and_then(|value| value.as_array())
        .into_iter()
        .flatten()
        .filter_map(|group| group.get("hooks").and_then(|hooks| hooks.as_array()))
        .flatten()
        .any(|handler| handler.get("command").and_then(|value| value.as_str()) == Some(command))
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

/// Codex R5 P3 helper — return true if the Google API key is reachable
/// via either the resolver's `api_key_env` (preferred path used by
/// migrated constructors) OR the legacy `config.extract.google.api_key`
/// field (populated from `GEMINI_API_KEY` at config-load time).
/// Mirrors the resolution order of `extract::llm::create_extractor`
/// post-R2 fix.
fn google_api_key_resolved(
    config: &ReinConfig,
    resolved: Option<&crate::config::ResolvedLlmConfig>,
) -> bool {
    // Resolved api_key_env path takes precedence — reflects how the
    // constructors actually decide whether the key exists at runtime.
    let env_present = resolved
        .and_then(|r| r.api_key_env.as_deref())
        .and_then(|env_name| std::env::var(env_name).ok())
        .is_some();
    env_present || config.extract.google.api_key.is_some()
}

fn check_extract_provider(config: &ReinConfig) -> DoctorCheck {
    // v0.27.1 B2 (spec §15 R9-K6): route through `resolve_llm_for("extract")`
    // so doctor reports the same provider/model production reads. Reading
    // legacy `[extract]` directly would mislead operators using `[llm]`
    // inheritance into thinking the resolved model differs from production.
    let resolved = config.resolve_llm_for("extract").ok();
    let provider = resolved
        .as_ref()
        .map(|r| r.provider)
        .unwrap_or(Provider::None);
    let model = resolved
        .as_ref()
        .map(|r| r.model.clone())
        .unwrap_or_default();
    let endpoint = resolved
        .as_ref()
        .map(|r| r.endpoint.clone())
        .unwrap_or_default();
    // Codex R5 P3 fix — honor the resolver's api_key_env. v0 doctor
    // hardcoded `GEMINI_API_KEY`, so an operator on a custom env name
    // (e.g. `[llm.google].api_key_env = "MY_KEY"`) saw a false WARN
    // even though the migrated constructors successfully read MY_KEY.
    let api_key_present_for_extract = google_api_key_resolved(config, resolved.as_ref());
    match provider {
        Provider::Google if api_key_present_for_extract => ok_in(
            DoctorCategory::Configuration,
            "extract_provider",
            format!("google:{model} configured"),
        ),
        Provider::Google => warn_in(
            DoctorCategory::Configuration,
            "extract_provider",
            "google configured but no API key found at the resolved api_key_env \
             (default GEMINI_API_KEY); LLM extraction is disabled",
        ),
        Provider::Omlx => ok_in(
            DoctorCategory::Configuration,
            "extract_provider",
            format!("omlx:{model} at {endpoint}"),
        ),
        Provider::None => ok_in(
            DoctorCategory::Configuration,
            "extract_provider",
            "disabled",
        ),
    }
}

fn check_query_expansion_provider(config: &ReinConfig) -> DoctorCheck {
    // v0.27.1 B2 (spec §15 R9-K6): route through
    // `resolve_llm_for("query_expansion")`.
    let resolved = config.resolve_llm_for("query_expansion").ok();
    let provider = resolved
        .as_ref()
        .map(|r| r.provider)
        .unwrap_or(Provider::None);
    let model = resolved
        .as_ref()
        .map(|r| r.model.clone())
        .unwrap_or_default();
    let endpoint = resolved
        .as_ref()
        .map(|r| r.endpoint.clone())
        .unwrap_or_default();
    // Codex R5 P3 — honor resolved api_key_env (mirror of extract path).
    let key_present = google_api_key_resolved_for_qe(config, resolved.as_ref());
    match provider {
        Provider::Google if key_present => ok_in(
            DoctorCategory::Configuration,
            "query_expansion",
            format!("google:{model} configured"),
        ),
        Provider::Google => warn_in(
            DoctorCategory::Configuration,
            "query_expansion",
            "google configured but no API key found at the resolved api_key_env \
             (default GEMINI_API_KEY); expansion is disabled",
        ),
        Provider::Omlx => ok_in(
            DoctorCategory::Configuration,
            "query_expansion",
            format!("omlx:{model} at {endpoint}"),
        ),
        Provider::None => ok_in(DoctorCategory::Configuration, "query_expansion", "disabled"),
    }
}

fn google_api_key_resolved_for_qe(
    config: &ReinConfig,
    resolved: Option<&crate::config::ResolvedLlmConfig>,
) -> bool {
    let env_present = resolved
        .and_then(|r| r.api_key_env.as_deref())
        .and_then(|env_name| std::env::var(env_name).ok())
        .is_some();
    env_present || config.query_expansion.google.api_key.is_some()
}

fn check_reranker_provider(config: &ReinConfig) -> DoctorCheck {
    // v0.27.1 B2 (spec §15 R9-K6): route through
    // `resolve_llm_for("search.llm_reranker")`.
    let resolved = config.resolve_llm_for("search.llm_reranker").ok();
    let provider = resolved
        .as_ref()
        .map(|r| r.provider)
        .unwrap_or(Provider::None);
    let model = resolved
        .as_ref()
        .map(|r| r.model.clone())
        .unwrap_or_default();
    let endpoint = resolved
        .as_ref()
        .map(|r| r.endpoint.clone())
        .unwrap_or_default();
    let top_n = config.search.llm_reranker_top_n;
    // Codex R5 P3 — honor resolved api_key_env (mirror of extract path).
    // Reranker shares `[query_expansion.google]` for the api_key field.
    let key_present = google_api_key_resolved_for_qe(config, resolved.as_ref());
    match provider {
        Provider::Google if key_present => ok_in(
            DoctorCategory::Configuration,
            "llm_reranker",
            format!("google:{model} configured (top_n={top_n})"),
        ),
        Provider::Google => warn_in(
            DoctorCategory::Configuration,
            "llm_reranker",
            "google reranker configured but no API key found at the resolved \
             api_key_env (default GEMINI_API_KEY); reranker will be skipped",
        ),
        Provider::Omlx => ok_in(
            DoctorCategory::Configuration,
            "llm_reranker",
            format!("omlx:{model} at {endpoint} (top_n={top_n})"),
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
    if let Some(policy) = config.server.auth {
        let policy = crate::auth::AuthPolicy::from(policy);
        return match policy {
            crate::auth::AuthPolicy::BearerRequired if token_present => ok_in(
                DoctorCategory::Configuration,
                "http_auth",
                "explicit auth policy bearer_required with REIN_HTTP_TOKEN configured",
            ),
            crate::auth::AuthPolicy::BearerRequired => fail_with_hint(
                DoctorCategory::Configuration,
                "http_auth",
                "explicit auth policy bearer_required requires REIN_HTTP_TOKEN",
                "set REIN_HTTP_TOKEN=<secret> or choose [server].auth = \"loopback_only\" / \"oauth\" / \"public\"",
            ),
            crate::auth::AuthPolicy::OAuth if token_present => ok_in(
                DoctorCategory::Configuration,
                "http_auth",
                "explicit auth policy oauth with owner approval token configured",
            ),
            crate::auth::AuthPolicy::OAuth => fail_with_hint(
                DoctorCategory::Configuration,
                "http_auth",
                "explicit auth policy oauth requires REIN_HTTP_TOKEN for owner approval",
                "set REIN_HTTP_TOKEN=<secret>; OAuth clients will not need this token, but the owner approval page will",
            ),
            crate::auth::AuthPolicy::LoopbackOnly => ok_in(
                DoctorCategory::Configuration,
                "http_auth",
                "explicit auth policy loopback_only",
            ),
            crate::auth::AuthPolicy::Public => ok_in(
                DoctorCategory::Configuration,
                "http_auth",
                "explicit auth policy public",
            ),
        };
    }
    let allow_unauth = config.server.loopback_unauth_requested();

    if token_present {
        if allow_unauth {
            warn_with_hint(
                DoctorCategory::Configuration,
                "http_auth",
                format!(
                    "REIN_HTTP_TOKEN is set for {}:{} and \
                     [server].allow_unauthenticated_loopback=true; token auth wins, \
                     so loopback unauth is disabled at runtime",
                    config.server.sse_bind, config.server.sse_port
                ),
                "unset REIN_HTTP_TOKEN for public/loopback-unauth testing, or set allow_unauthenticated_loopback=false when bearer auth is intended",
            )
        } else if config.server.allow_unauthenticated_loopback {
            warn_with_hint(
                DoctorCategory::Configuration,
                "http_auth",
                format!(
                    "REIN_HTTP_TOKEN is set for {}:{} and \
                     [server].allow_unauthenticated_loopback=true cannot take effect because the bind host is not loopback; \
                     bearer auth remains required",
                    config.server.sse_bind, config.server.sse_port
                ),
                "keep REIN_HTTP_TOKEN for this bind, or bind to 127.0.0.1, ::1, or localhost for loopback-unauth testing",
            )
        } else {
            ok_in(
                DoctorCategory::Configuration,
                "http_auth",
                format!(
                    "token configured for {}:{}",
                    config.server.sse_bind, config.server.sse_port
                ),
            )
        }
    } else if allow_unauth {
        ok_in(
            DoctorCategory::Configuration,
            "http_auth",
            format!(
                "loopback-only unauthenticated access allowed for {}:{}",
                config.server.sse_bind, config.server.sse_port
            ),
        )
    } else if config.server.allow_unauthenticated_loopback {
        fail_with_hint(
            DoctorCategory::Configuration,
            "http_auth",
            format!(
                "HTTP/SSE is enabled on {}:{} without REIN_HTTP_TOKEN; \
                 [server].allow_unauthenticated_loopback=true cannot take effect because the bind host is not loopback",
                config.server.sse_bind, config.server.sse_port
            ),
            "bind to 127.0.0.1, ::1, or localhost for loopback-unauth testing, or set REIN_HTTP_TOKEN=<secret>",
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

fn check_auth_policy_consistency(config: &ReinConfig) -> DoctorCheck {
    if !config.server.sse_enabled && !config.server.gui_enabled {
        return ok_in(
            DoctorCategory::Configuration,
            "auth_policy",
            "HTTP auth policy inactive because HTTP/SSE and GUI are disabled",
        );
    }

    let token_present = std::env::var("REIN_HTTP_TOKEN")
        .ok()
        .is_some_and(|token| !token.trim().is_empty());
    let policy = match config.resolve_auth_policy() {
        Ok(policy) => policy,
        Err(err) => {
            return fail_with_hint(
                DoctorCategory::Configuration,
                "auth_policy",
                format!("auth policy could not be resolved: {err}"),
                "set [server].auth explicitly, set REIN_HTTP_TOKEN, or use [server].auth = \"loopback_only\" for loopback-only local HTTP",
            );
        }
    };

    if crate::mcp::server::wildcard_bind_requires_allowlist(
        &config.server.sse_bind,
        config.server.allowed_hosts.as_deref(),
    ) && crate::mcp::server::auth_policy_requires_wildcard_allowlist(policy)
    {
        return fail_with_hint(
            DoctorCategory::Configuration,
            "auth_policy",
            format!(
                "auth policy {} on wildcard bind {} requires [server].allowed_hosts; rein serve will refuse to start",
                policy.as_str(),
                config.server.sse_bind
            ),
            "set [server].allowed_hosts = [\"your-public-host\"] or use auth = \"bearer_required\" with REIN_HTTP_TOKEN",
        );
    }

    if config.server.auth.is_some()
        && token_present
        && matches!(
            policy,
            crate::auth::AuthPolicy::LoopbackOnly | crate::auth::AuthPolicy::Public
        )
    {
        return warn_with_hint(
            DoctorCategory::Configuration,
            "auth_policy",
            format!(
                "REIN_HTTP_TOKEN is set but auth policy is {}; the token has no effect",
                policy.as_str()
            ),
            "unset REIN_HTTP_TOKEN or change [server].auth = \"bearer_required\" if bearer auth is intended",
        );
    }

    if config.server.auth.is_none() && config.server.allow_unauthenticated_loopback {
        return ok_in(
            DoctorCategory::Configuration,
            "auth_policy",
            "[server].allow_unauthenticated_loopback is deprecated; use [server].auth = \"public\" for the legacy remote read-only tunnel mode, or [server].auth = \"loopback_only\" for local-only access. Will be removed in v0.31.",
        );
    }

    ok_in(
        DoctorCategory::Configuration,
        "auth_policy",
        format!("auth policy resolved to {}", policy.as_str()),
    )
}

fn check_oauth_provider(config: &ReinConfig) -> DoctorCheck {
    let policy = config
        .resolve_auth_policy()
        .map(|policy| policy.as_str().to_string())
        .unwrap_or_else(|_| "unresolved".to_string());
    let store = match config.open_store() {
        Ok(store) => store,
        Err(err) => {
            return warn_with_hint(
                DoctorCategory::Configuration,
                "oauth_provider",
                format!("auth_policy={policy}; OAuth provider store check failed: {err}"),
                "run rein doctor after the database path is available",
            );
        }
    };
    let conn = store.conn();
    let client_count: i64 = conn
        .query_row("SELECT COUNT(*) FROM oauth_clients", [], |row| row.get(0))
        .unwrap_or(0);
    let now = chrono::Utc::now().timestamp();
    let active_grants: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM oauth_grants
             WHERE revoked_at IS NULL AND refresh_expires_at > ?1",
            [now],
            |row| row.get(0),
        )
        .unwrap_or(0);
    let expired_codes: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM oauth_auth_codes WHERE expires_at < ?1",
            [now - 86_400],
            |row| row.get(0),
        )
        .unwrap_or(0);
    let expired_grants: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM oauth_grants
             WHERE access_expires_at < ?1 AND refresh_expires_at < ?2",
            [now - 86_400 * 30, now - 86_400 * 30],
            |row| row.get(0),
        )
        .unwrap_or(0);
    // Post-upgrade grant-loss detection: when the explicit OAuth policy is
    // active and at least one DCR client is registered but every grant is
    // revoked or never issued, the connector is silently broken until the
    // operator disconnects + re-adds it on claude.ai. This is the signature
    // of the v0.30.0 `refresh_token_fingerprint` schema migration — the
    // backfill stamps `revoked_at` on every pre-release grant without a
    // fingerprint, and Anthropic's broker doesn't always fall back from a
    // failed refresh-token to a fresh authorization-code flow.
    //
    // Surface this as a single actionable WARN line so the operator can
    // recover with a single UI action instead of chasing a generic
    // "unknown error" message in the connector dialog. The same condition
    // also catches less common cases (operator manually revoked all grants,
    // every grant expired past TTL without a refresh) where the fix is the
    // same: re-authorize on claude.ai.
    if policy == "oauth" && client_count > 0 && active_grants == 0 {
        return warn_with_hint(
            DoctorCategory::Configuration,
            "oauth_provider",
            format!(
                "auth_policy={policy}; oauth_clients={client_count}; active_grants=0; expired_oauth_records={} — DCR clients exist but every grant is revoked",
                expired_codes + expired_grants
            ),
            "in claude.ai → Settings → Connectors, remove the rein connector and re-add it (same URL); this re-runs DCR + authorization-code and issues a fresh grant",
        );
    }

    if expired_codes + expired_grants > 1000 {
        warn_with_hint(
            DoctorCategory::Storage,
            "oauth_provider",
            format!(
                "auth_policy={policy}; oauth_clients={client_count}; active_grants={active_grants}; expired_oauth_records={}",
                expired_codes + expired_grants
            ),
            "run OAuth GC or restart rein so scheduled maintenance can prune expired OAuth rows",
        )
    } else {
        ok_in(
            DoctorCategory::Storage,
            "oauth_provider",
            format!(
                "auth_policy={policy}; oauth_clients={client_count}; active_grants={active_grants}; expired_oauth_records={}",
                expired_codes + expired_grants
            ),
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
    // F5/C1 follow-up: proxy is a separately-managed service (`rein proxy on`)
    // and is NOT auto-started. When the bind is loopback AND the service
    // is offline AND no token is configured, surface OK with a hint string
    // — a fresh install with default-deny defaults shouldn't bump the
    // exit code on a fail-check the operator can't act on yet.
    //
    // Wildcard binds (0.0.0.0/::/empty) MUST still FAIL without a token,
    // because they would expose the proxy to the LAN as soon as it
    // started. Same exposure rationale as the HTTP/SSE check: an
    // operator picking a wildcard bind is signalling intent to serve
    // beyond loopback, so the missing-token check is acted on now.
    if !token_present
        && !allow_unauth
        && is_loopback
        && crate::service::is_running("proxy").is_none()
    {
        return ok_in(
            DoctorCategory::Configuration,
            "proxy_auth",
            format!(
                "proxy not running on {}:{} (set REIN_PROXY_TOKEN or [proxy].allow_unauthenticated_loopback before `rein proxy on`)",
                config.proxy.bind, config.proxy.port
            ),
        );
    }

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
    let rebuild_state = warmup::tantivy_rebuild_state(db_path);

    if matches!(rebuild_state, warmup::TantivyRebuildState::Running) {
        return warn_with_hint(
            DoctorCategory::Index,
            "tantivy",
            format!(
                "index rebuild in progress at {}{}",
                index_path.display(),
                if dirty {
                    "; dirty marker is present"
                } else {
                    ""
                }
            ),
            "wait for the rebuild owner to finish, then rerun `rein doctor`",
        );
    }

    if matches!(rebuild_state, warmup::TantivyRebuildState::StaleMarker) {
        return warn_with_hint(
            DoctorCategory::Index,
            "tantivy",
            format!(
                "stale rebuild marker is present at {}",
                warmup::tantivy_rebuilding_path(db_path).display()
            ),
            "run `rein doctor --fix` or `rein warmup` to refresh the index",
        );
    }

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

/// Surface ConnPool saturation count from `PoolMetrics::try_get_saturated_count`.
/// `try_get` is the non-blocking acquire used by `search/recall.rs`'s 3-channel
/// fanout; on saturation it falls back to a fresh `SqliteStore::new`, which
/// preserves correctness but degrades into per-channel connection churn rather
/// than clean backpressure. Sustained nonzero growth here is the operator's
/// signal that pool size is undersized for the recall workload. Agent D Q10
/// (post-v0.23.0 architecture audit). The check is informational-only —
/// saturation isn't an error, but operators who never look at the count don't
/// know to investigate. Threshold (≥ 1000 events) is structural ("more than a
/// rare burst"), not a tunable.
fn check_pool_saturation(config: &ReinConfig) -> DoctorCheck {
    let db_path = std::path::PathBuf::from(&config.database.path);
    let metrics = match crate::config::pool_metrics_for_path(&db_path) {
        Ok(m) => m,
        Err(_) => {
            return ok_in(
                DoctorCategory::Storage,
                "pool_saturation",
                "pool not initialized for this path (no recall traffic yet)".to_string(),
            );
        }
    };
    // Post-fix audit L-2: warn only when saturation is BOTH frequent
    // (lifetime count over threshold) AND recent (event in the last hour).
    // A bursty load test that crossed 1000 events two hours ago should
    // NOT keep warning — the degraded regime has passed and the lifetime
    // counter alone can't tell operators when to care.
    //
    // Audit round-2 LOW 9: handle the NTP rollback edge case. If the
    // wall clock stepped backward (NTP correction, manual set, DST
    // oddity on systems that ignore UTC), `last_saturation_at` can end
    // up AFTER `now_s` — `saturating_sub` would then return 0 and the
    // recency check would falsely fire. Treat future timestamps as
    // "unknown" and skip the recency gate (same behavior as the
    // never-saturated path).
    let now_s = chrono::Utc::now().timestamp();
    let seconds_since_last_saturation =
        if metrics.last_saturation_at <= 0 || metrics.last_saturation_at > now_s {
            i64::MAX
        } else {
            now_s - metrics.last_saturation_at
        };
    let message = format!(
        "size={} idle={} in_use={} permits={} shrunk={} saturated={} recent={}",
        metrics.size,
        metrics.idle,
        metrics.in_use,
        metrics.available_permits,
        metrics.shrunk_count,
        metrics.try_get_saturated_count,
        if seconds_since_last_saturation == i64::MAX {
            "never".to_string()
        } else {
            format!("{}s ago", seconds_since_last_saturation)
        }
    );
    const RECENT_WINDOW_SECS: i64 = 3600;
    if metrics.try_get_saturated_count >= 1000
        && seconds_since_last_saturation <= RECENT_WINDOW_SECS
    {
        warn_with_hint(
            DoctorCategory::Storage,
            "pool_saturation",
            message,
            "many recall fanout calls have fallen back from the pool to fresh \
             SqliteStore::new in the last hour — consider increasing pool size \
             in config",
        )
    } else {
        ok_in(DoctorCategory::Storage, "pool_saturation", message)
    }
}

fn check_resummerize(store: &SqliteStore) -> DoctorCheck {
    // v0.23: surface resummerize backlog + recent failure rate. Backlog
    // alone is not an error — it just means the LLM hasn't yet processed
    // rows flagged by MergeInto cap hits. A high failure rate indicates
    // a broken prompt / model / contract tuning and should be triaged.
    //
    // Post-audit round-2 MED-2: ALSO surface `claim_lost` rate as a
    // separate signal. `recent_failure_rate` (quality metric) filters
    // claim_lost out so a contention spike doesn't page an operator
    // about "failing LLM quality" when the real issue is concurrent
    // workers. `recent_claim_lost_rate` (contention metric) is the
    // missing signal: high value = workers racing + burning LLM budget
    // without progress. Warn on its own threshold independent of the
    // quality gate.
    let backlog = crate::ops::resummerize::backlog_count(store).unwrap_or(0);
    let failure_rate = crate::store::resummerize_audit::recent_failure_rate(
        store.conn(),
        chrono::Duration::hours(24),
    )
    .unwrap_or(0.0);
    let claim_lost_rate = crate::store::resummerize_audit::recent_claim_lost_rate(
        store.conn(),
        chrono::Duration::hours(24),
    )
    .unwrap_or(0.0);
    let total_recent = crate::store::resummerize_audit::recent_run_count(
        store.conn(),
        chrono::Duration::hours(24),
    )
    .unwrap_or(0);

    let message = format!(
        "backlog={backlog} needing resummerize; last 24h: {total_recent} runs, \
         failure_rate={:.1}%, claim_lost_rate={:.1}%",
        failure_rate * 100.0,
        claim_lost_rate * 100.0,
    );

    // Thresholds are structural, not tuned:
    //   * failure rate ≥ 50% over ≥ 5 runs ⇒ contract/prompt/model is failing
    //   * claim_lost rate ≥ 30% over ≥ 10 total (quality + contention) ⇒
    //     workers are racing on the same canonicals; reduce batch
    //     overlap or serialize further. 30% is a looser gate than the
    //     failure threshold because some claim_lost under concurrent
    //     MergeInto load is normal; 30% over 10 runs is "unusually
    //     contentious", not "broken".
    //
    // `recent_run_count` excludes claim_lost (M-3), so the quality gate
    // uses `total_recent`, and the contention gate uses the full 24h
    // total (claim_lost + quality classes).
    let full_24h_total: i64 = store
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM resummerize_runs \
               WHERE finished_at IS NOT NULL AND finished_at >= ?1",
            rusqlite::params![(chrono::Utc::now() - chrono::Duration::hours(24)).to_rfc3339()],
            |row| row.get::<_, i64>(0),
        )
        .unwrap_or(0);

    let quality_alert = total_recent >= 5 && failure_rate >= 0.5;
    let contention_alert = full_24h_total >= 10 && claim_lost_rate >= 0.3;

    if quality_alert {
        warn_with_hint(
            DoctorCategory::Queue,
            "resummerize",
            message,
            "check [resummerize] config + recent resummarize_runs rows; a \
             systematic failure usually means the LLM prompt or contract \
             needs adjustment",
        )
    } else if contention_alert {
        warn_with_hint(
            DoctorCategory::Queue,
            "resummerize",
            message,
            "many resummerize runs lost their claim to peer workers in the \
             last 24h — LLM tokens are being spent without progress. \
             Reduce worker batch size overlap or serialize resummerize \
             further (lower concurrency).",
        )
    } else {
        ok_in(DoctorCategory::Queue, "resummerize", message)
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
    let tantivy_rebuild_state = warmup::tantivy_rebuild_state(store.db_path());
    if !tantivy_path.exists()
        || tantivy_dirty.exists()
        || matches!(
            tantivy_rebuild_state,
            warmup::TantivyRebuildState::StaleMarker
        )
    {
        match warmup::try_populate_tantivy(store) {
            warmup::TantivyRebuildOutcome::Rebuilt { indexed, errors } => {
                fixes.push(format!(
                    "rebuilt Tantivy index at {} ({indexed} indexed, {errors} errors)",
                    tantivy_path.display()
                ));
            }
            warmup::TantivyRebuildOutcome::AlreadyRunning { lock_path } => {
                fixes.push(format!(
                    "Tantivy rebuild already in progress at {}",
                    lock_path.display()
                ));
            }
            warmup::TantivyRebuildOutcome::Failed { reason } => {
                fixes.push(format!(
                    "Tantivy rebuild failed at {}: {reason}",
                    tantivy_path.display()
                ));
            }
            warmup::TantivyRebuildOutcome::SkippedInMemory => {}
        }
    }

    let hnsw_base = store.db_path().with_extension("");
    let hnsw_path = hnsw_base.with_extension("usearch");
    let hnsw_meta = hnsw_base.with_extension("usearch.meta");
    if !hnsw_path.exists() || !hnsw_meta.exists() || HnswIndex::is_dirty(&hnsw_base) {
        warmup::populate_hnsw(store, config);
        fixes.push(format!("triggered HNSW rebuild at {}", hnsw_path.display()));
    }

    // v0.28.7+ audit L5 — corrupt policy row recovery. When
    // `load_parameter_policy` reports `Corrupt`, every subsequent
    // `save_parameter_policy_cas` UPDATE matches 0 rows
    // (`json_valid(value)` fails) and the existence check then
    // returns `Ok(false)` — the caller treats this as a CAS miss and
    // never inserts a fresh row, so the policy refresh stalls
    // permanently. Delete the row so the next
    // `refresh_ars_parameter_policy` tick can `INSERT` cleanly.
    // Read-only doctor (`fix=false`) still surfaces the corruption
    // via `check_ars_parameter_policy`'s warn.
    //
    // **NOT** wired for `StorageError` (R4 P2 #2 audit catch
    // 2026-05-04): a transient SQLite busy/locked read returns
    // `StorageError`, but the underlying row may still be valid.
    // Unconditional deletion in that case would reset a healthy
    // canary policy as collateral damage from a recoverable read
    // error. Restrict destructive recovery to confirmed `Corrupt`
    // (the row was read AND failed `serde_json::from_str` or the
    // schema-version check); leave `StorageError` to be retried on
    // the next doctor pass when the lock has cleared.
    //
    // R10 P3 (2026-05-04): use the atomic `repair_corrupt_parameter_policy`
    // helper, which wraps the status re-check + DELETE in a single
    // `BEGIN IMMEDIATE` transaction. Pre-fix this path did
    // `load_parameter_policy` (read-only, no lock) followed by an
    // unconditional `delete_parameter_policy` — a peer
    // `refresh_ars_parameter_policy` tick or a concurrent
    // `doctor --fix` could rewrite the row to a healthy canary in the
    // gap between those two ops, and then the unconditional DELETE
    // would destroy that newly-valid state. The new helper observes
    // status under the write lock and refuses to delete when the row
    // is no longer Corrupt, surfacing the observed status in the fix
    // report so the operator can see what happened.
    match crate::store::ars_parameter_policy::repair_corrupt_parameter_policy(store.conn()) {
        Ok(outcome) if outcome.deleted > 0 => fixes.push(format!(
            "deleted corrupt ars_parameter_policy row ({} row{}): {}",
            outcome.deleted,
            if outcome.deleted == 1 { "" } else { "s" },
            outcome
                .error_at_delete
                .unwrap_or_else(|| "unknown error".to_string())
        )),
        Ok(_) => {
            // No-op: either the row was never Corrupt at recovery
            // time (peer repaired or transitioned to a different
            // unhealthy status under the write lock) or the row is
            // healthy. `fixes_applied` lists ACTIONS taken, not
            // non-actions — the doctor's `check_ars_parameter_policy`
            // already surfaces non-Corrupt unhealthy states via warn,
            // so silence here keeps the fix-list clean.
        }
        Err(e) => fixes.push(format!(
            "failed to delete corrupt ars_parameter_policy row: {e}"
        )),
    }

    // v0.28.7 H2 — drift-triggered rollback. If the parameter policy is in
    // Canary mode while `judge_calibration_state.judge_drift_alert*` is
    // positive, force a refresh which (per the demote-on-drift logic in
    // `ops/adaptive::refresh_ars_parameter_policy`) flips the policy back
    // to Shadow and zeroes runtime_adoption_weight on the next pipeline
    // tick. The refresh itself is idempotent and cheap.
    if let Some(state) = crate::store::adaptive::AdaptiveState::restore_snapshot(store.conn()) {
        let drift_active = state
            .judge_calibration_state
            .as_ref()
            .map(|cal| {
                cal.judge_drift_alert > 0
                    || cal.judge_drift_alert_synthesis > 0
                    || cal.judge_drift_alert_concept > 0
            })
            .unwrap_or(false);
        if drift_active {
            let loaded = crate::store::ars_parameter_policy::load_parameter_policy(store.conn());
            if matches!(
                loaded.status,
                crate::store::ars_parameter_policy::ArsParameterPolicyLoadStatus::Loaded
            ) && matches!(
                loaded.policy.mode,
                crate::store::ars_parameter_policy::ArsParameterPolicyMode::Canary
            ) {
                crate::ops::adaptive::refresh_ars_parameter_policy_for_doctor(
                    store.conn(),
                    config,
                    &state,
                );
                fixes.push(
                    "demoted ARS parameter policy from Canary to Shadow due to active judge drift alert"
                        .to_string(),
                );
            }
        }
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

// ── v0.27.x judge checks ───────────────────────────────────────────────────

/// v0.28.x — report the ARS dynamic-parameter activation policy separately
/// from the large AdaptiveState snapshot so rollback/corruption is visible.
fn check_ars_parameter_policy(
    store: &crate::store::SqliteStore,
    config: &ReinConfig,
) -> DoctorCheck {
    use crate::store::ars_parameter_policy::{
        load_parameter_policy, ArsParameterPolicyLoadStatus, ArsParameterPolicyMode,
    };

    let loaded = load_parameter_policy(store.conn());
    match loaded.status {
        ArsParameterPolicyLoadStatus::Missing => ok_in(
            DoctorCategory::Configuration,
            "ars_parameter_policy",
            "missing policy row; dynamic ARS parameters disabled".to_string(),
        ),
        ArsParameterPolicyLoadStatus::Corrupt
        | ArsParameterPolicyLoadStatus::UnsupportedSchema
        | ArsParameterPolicyLoadStatus::StorageError => {
            // All three statuses mean dynamic ARS parameters are
            // disabled; doctor surfaces them as a warn so an operator
            // can investigate. The status-specific recovery (delete
            // for Corrupt, leave for UnsupportedSchema/StorageError)
            // happens in `apply_local_fixes`.
            warn_in(
                DoctorCategory::Storage,
                "ars_parameter_policy",
                format!(
                    "policy row unhealthy; dynamic ARS parameters disabled ({})",
                    loaded.error.unwrap_or_else(|| "unknown error".to_string())
                ),
            )
        }
        ArsParameterPolicyLoadStatus::Loaded => {
            let policy = loaded.policy;
            let state = crate::store::adaptive::AdaptiveState::restore_snapshot(store.conn())
                .unwrap_or_default();
            let runtime_adoption_weight = if config.adaptive.enabled
                && config.ars.acceleration.enabled
                && !config.ars.acceleration.shadow_only
            {
                policy.runtime_adoption_weight(state.version)
            } else {
                0.0
            };
            let live_allowed = runtime_adoption_weight > f64::EPSILON;
            let message = format!(
                "mode={:?} revision={} source_adaptive_version={} current_adaptive_version={} live_allowed={} runtime_adoption_weight={:.3}",
                policy.mode,
                policy.revision,
                policy.source_adaptive_version,
                state.version,
                live_allowed,
                runtime_adoption_weight,
            );
            if matches!(policy.mode, ArsParameterPolicyMode::Canary) && !live_allowed {
                warn_in(
                    DoctorCategory::Configuration,
                    "ars_parameter_policy",
                    message,
                )
            } else {
                ok_in(
                    DoctorCategory::Configuration,
                    "ars_parameter_policy",
                    message,
                )
            }
        }
    }
}

/// v0.27.1 — surface `JudgeCalibrationState` stats so operators know
/// the runtime LLM judge is producing usable signal. Reports κ values,
/// pair counts, drift alert count, and last-computed timestamp.
fn check_judge_calibration(store: &crate::store::SqliteStore, config: &ReinConfig) -> DoctorCheck {
    if !config.ars.llm_judge.enabled {
        return ok_in(
            DoctorCategory::Configuration,
            "judge_calibration",
            "[ars.llm_judge].enabled = false (disabled by config)".to_string(),
        );
    }
    let state = match crate::store::adaptive::AdaptiveState::restore_snapshot(store.conn()) {
        Some(s) => s,
        None => {
            return ok_in(
                DoctorCategory::Configuration,
                "judge_calibration",
                "no AdaptiveState snapshot yet (no recall traffic)".to_string(),
            );
        }
    };
    let cal = match state.judge_calibration_state {
        Some(c) => c,
        None => {
            return ok_in(
                DoctorCategory::Configuration,
                "judge_calibration",
                "no calibration state yet (no judge events processed)".to_string(),
            );
        }
    };
    let synth_pairs = cal.recent_pairs_synthesis.len();
    let concept_pairs = cal.recent_pairs_concept.len();
    let total_drift_alerts = cal
        .judge_drift_alert
        .saturating_add(cal.judge_drift_alert_synthesis)
        .saturating_add(cal.judge_drift_alert_concept);
    let summary = format!(
        "kappa={:.2} runtime_vs_offline_kappa={:.2} synth_runtime_kappa={:.2} \
         concept_runtime_kappa={:.2} drift_alerts={} synth_drift_alerts={} \
         concept_drift_alerts={} synth_pairs={} concept_pairs={} total_offline={}",
        cal.kappa,
        cal.runtime_vs_offline_kappa,
        cal.runtime_vs_offline_kappa_synthesis,
        cal.runtime_vs_offline_kappa_concept,
        cal.judge_drift_alert,
        cal.judge_drift_alert_synthesis,
        cal.judge_drift_alert_concept,
        synth_pairs,
        concept_pairs,
        cal.total_offline_cron_events,
    );
    if total_drift_alerts > 0 {
        warn_in(
            DoctorCategory::Configuration,
            "judge_calibration",
            format!("drift alerts present: {summary}"),
        )
    } else {
        ok_in(DoctorCategory::Configuration, "judge_calibration", summary)
    }
}

/// v0.27.1 — rolling 24h judge_call_ledger usage vs daily_call_cap.
fn check_judge_call_ledger(store: &crate::store::SqliteStore, config: &ReinConfig) -> DoctorCheck {
    if !config.ars.llm_judge.enabled {
        return ok_in(
            DoctorCategory::Configuration,
            "judge_call_ledger",
            "[ars.llm_judge].enabled = false (disabled by config)".to_string(),
        );
    }
    let cap = config.ars.llm_judge.daily_call_cap;
    // Codex C234 P3 fix — exclude `reserved` rows older than
    // LLM_JUDGE_STALE_CLAIM_SECS from the active count. A worker
    // crash leaves these rows orphaned until next dispatch reaps
    // them; counting them as in_flight makes doctor falsely report
    // cap-reached for up to 24h post-crash.
    let stale_secs: i64 = crate::judge::contract::LLM_JUDGE_STALE_CLAIM_SECS;
    let result: Result<(i64, i64, i64), rusqlite::Error> = store.conn().query_row(
        "SELECT \
            COALESCE(SUM(CASE WHEN \
                status='done' OR status='failed' \
                OR (status='reserved' AND ts >= strftime('%s','now') - ?1) \
                THEN 1 ELSE 0 END), 0), \
            COALESCE(SUM(CASE WHEN status='reserved' AND ts >= strftime('%s','now') - ?1 \
                THEN 1 ELSE 0 END), 0), \
            COALESCE(SUM(CASE WHEN status='stale' OR \
                (status='reserved' AND ts < strftime('%s','now') - ?1) \
                THEN 1 ELSE 0 END), 0) \
         FROM judge_call_ledger \
         WHERE ts >= strftime('%s','now') - 86400",
        [stale_secs],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
    );
    let (active, in_flight, stale) = match result {
        Ok(t) => t,
        Err(rusqlite::Error::SqliteFailure(_, ref msg))
            if msg.as_deref().is_some_and(|m| m.contains("no such table")) =>
        {
            return ok_in(
                DoctorCategory::Configuration,
                "judge_call_ledger",
                "ledger not initialized (no judge calls yet)".to_string(),
            );
        }
        Err(e) => {
            return warn_in(
                DoctorCategory::Configuration,
                "judge_call_ledger",
                format!("ledger query failed: {e}"),
            );
        }
    };
    let summary = format!("24h: {active}/{cap} (in_flight={in_flight} stale={stale})");
    let active_u = active as u64;
    // v0.28.7 M-9 — saturation warn at 100% (existing) PLUS near-saturation
    // attention at >= 90% so operators see the cap pressure before the
    // worker starts dropping jobs into `dropped_cap`. Without this, the
    // judge surface goes from "all good" to "cap exhausted" with no
    // intermediate signal.
    if active_u >= cap {
        warn_in(
            DoctorCategory::Configuration,
            "judge_call_ledger",
            format!("daily cap exhausted — {summary}"),
        )
    } else if cap > 0 && active_u.saturating_mul(10) >= cap.saturating_mul(9) {
        warn_in(
            DoctorCategory::Configuration,
            "judge_call_ledger",
            format!("near saturation (>=90%) — {summary}"),
        )
    } else {
        ok_in(DoctorCategory::Configuration, "judge_call_ledger", summary)
    }
}

/// v0.27.2 — synthesis + concept-summary cache file size. Catches
/// reaper failures (cache should stay bounded around TTL × peak rate).
fn check_judge_cache_size(config: &ReinConfig) -> DoctorCheck {
    if !config.ars.llm_judge.enabled {
        return ok_in(
            DoctorCategory::Storage,
            "judge_cache_size",
            "[ars.llm_judge].enabled = false (disabled by config)".to_string(),
        );
    }
    let synth = crate::ops::handlers::judge::synthesis_cache_path_for_config(config);
    let concept = crate::ops::handlers::judge::concept_summary_cache_path_for_config(config);
    let size_of =
        |p: &std::path::Path| -> u64 { std::fs::metadata(p).map(|m| m.len()).unwrap_or(0) };
    let synth_bytes = size_of(&synth);
    let concept_bytes = size_of(&concept);
    let total_mb = (synth_bytes + concept_bytes) as f64 / 1_048_576.0;
    let summary = format!(
        "synthesis={}KB concept_summary={}KB total={:.1}MB",
        synth_bytes / 1024,
        concept_bytes / 1024,
        total_mb,
    );
    // 100MB threshold — well above expected steady-state for default
    // ttl (10 min) and daily_call_cap (10000 calls × ~8KB each = 80MB
    // worst case). If we cross this, the reaper is likely broken.
    if total_mb > 100.0 {
        warn_in(
            DoctorCategory::Storage,
            "judge_cache_size",
            format!("cache larger than expected: {summary} (reaper may be misfiring)"),
        )
    } else {
        ok_in(DoctorCategory::Storage, "judge_cache_size", summary)
    }
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
            archival_summary: None,
            archival_summary_at: None,
            archival_summary_version: None,
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

    #[cfg(unix)]
    fn hold_file_lock(path: &Path) -> std::fs::File {
        use std::os::unix::io::AsRawFd;

        let file = std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(path)
            .unwrap();
        let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        assert_eq!(rc, 0, "test failed to acquire advisory lock");
        file
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

    fn write_codex_config(codex_dir: &Path, enabled: bool) {
        // Codex 0.129+ format. `check_codex_hooks_at` should also accept the
        // legacy `codex_hooks` key — see `write_codex_config_legacy`.
        std::fs::create_dir_all(codex_dir).unwrap();
        std::fs::write(
            codex_dir.join("config.toml"),
            format!("[features]\nhooks = {enabled}\n"),
        )
        .unwrap();
    }

    fn write_codex_config_legacy(codex_dir: &Path, enabled: bool) {
        // Pre-0.129 layout. Verifies graceful read-side compat for users still
        // on the old codex CLI.
        std::fs::create_dir_all(codex_dir).unwrap();
        std::fs::write(
            codex_dir.join("config.toml"),
            format!("[features]\ncodex_hooks = {enabled}\n"),
        )
        .unwrap();
    }

    fn write_codex_config_raw(codex_dir: &Path, content: &str) {
        std::fs::create_dir_all(codex_dir).unwrap();
        std::fs::write(codex_dir.join("config.toml"), content).unwrap();
    }

    fn write_codex_hooks(codex_dir: &Path, omit_event: Option<&str>) {
        let mut hooks = serde_json::Map::new();
        for (event, command) in expected_codex_hook_commands() {
            if omit_event == Some(event) {
                continue;
            }
            hooks.insert(
                event.to_string(),
                serde_json::json!([
                    {
                        "hooks": [
                            {
                                "type": "command",
                                "command": command
                            }
                        ]
                    }
                ]),
            );
        }
        let root = serde_json::json!({ "hooks": hooks });
        std::fs::write(
            codex_dir.join("hooks.json"),
            serde_json::to_string_pretty(&root).unwrap(),
        )
        .unwrap();
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
    fn test_doctor_reports_codex_hooks_healthy() {
        let dir = tempfile::tempdir().unwrap();
        write_codex_config(dir.path(), true);
        write_codex_hooks(dir.path(), None);

        let check = check_codex_hooks_at(dir.path());

        assert_eq!(check.name, "codex_hooks");
        assert_eq!(check.category, DoctorCategory::Configuration);
        assert_eq!(check.status, CheckStatus::Ok);
        assert!(check.message.contains("six hooks configured"));
        assert_eq!(check.repair_hint, None);
    }

    #[test]
    fn test_doctor_reports_codex_hooks_feature_disabled() {
        let dir = tempfile::tempdir().unwrap();
        write_codex_config(dir.path(), false);
        write_codex_hooks(dir.path(), None);

        let check = check_codex_hooks_at(dir.path());

        assert_eq!(check.name, "codex_hooks");
        assert_eq!(check.status, CheckStatus::Warn);
        assert_eq!(check.repair_hint.as_deref(), Some("run rein init"));
        assert!(check.message.contains("hooks"));
    }

    #[test]
    fn test_doctor_accepts_legacy_codex_hooks_key() {
        // Older codex (<0.129) used `[features].codex_hooks = true`. Rein
        // doctor must keep recognising that until those users upgrade.
        let dir = tempfile::tempdir().unwrap();
        write_codex_config_legacy(dir.path(), true);
        write_codex_hooks(dir.path(), None);

        let check = check_codex_hooks_at(dir.path());

        assert_eq!(check.name, "codex_hooks");
        assert_eq!(check.status, CheckStatus::Ok);
        assert!(check.message.contains("six hooks configured"));
    }

    #[test]
    fn test_doctor_reports_codex_hooks_missing_hooks_file() {
        let dir = tempfile::tempdir().unwrap();
        write_codex_config(dir.path(), true);

        let check = check_codex_hooks_at(dir.path());

        assert_eq!(check.name, "codex_hooks");
        assert_eq!(check.status, CheckStatus::Warn);
        assert_eq!(check.repair_hint.as_deref(), Some("run rein init"));
        assert!(check.message.contains("hooks.json"));
    }

    #[test]
    fn test_doctor_reports_codex_hooks_missing_one_event() {
        let dir = tempfile::tempdir().unwrap();
        write_codex_config(dir.path(), true);
        write_codex_hooks(dir.path(), Some("Stop"));

        let check = check_codex_hooks_at(dir.path());

        assert_eq!(check.name, "codex_hooks");
        assert_eq!(check.status, CheckStatus::Warn);
        assert_eq!(check.repair_hint.as_deref(), Some("run rein init"));
        assert!(check.message.contains("Stop"));
    }

    #[test]
    fn test_doctor_reports_codex_mcp_stdio_healthy() {
        let dir = tempfile::tempdir().unwrap();
        write_codex_config_raw(
            dir.path(),
            "[mcp_servers.rein]\ncommand = \"rein\"\nargs = [\"serve\"]\n",
        );

        let check = check_codex_mcp_server_at(dir.path(), Path::new("/tmp/rein-local.db"));

        assert_eq!(check.name, "codex_mcp");
        assert_eq!(check.status, CheckStatus::Ok);
        assert!(check.message.contains("stdio"));
    }

    #[test]
    fn test_doctor_reports_codex_mcp_loopback_url_healthy() {
        let dir = tempfile::tempdir().unwrap();
        write_codex_config_raw(
            dir.path(),
            "[mcp_servers.rein]\nurl = \"http://127.0.0.1:8680/mcp\"\n",
        );

        let check = check_codex_mcp_server_at(dir.path(), Path::new("/tmp/rein-local.db"));

        assert_eq!(check.name, "codex_mcp");
        assert_eq!(check.status, CheckStatus::Ok);
        assert!(check.message.contains("loopback"));
    }

    #[test]
    fn test_doctor_reports_codex_mcp_loopback_ip_url_healthy() {
        let dir = tempfile::tempdir().unwrap();
        for url in [
            "http://127.0.0.2:8680/mcp",
            "http://[0:0:0:0:0:0:0:1]:8680/mcp",
        ] {
            write_codex_config_raw(
                dir.path(),
                &format!("[mcp_servers.rein]\nurl = \"{url}\"\n"),
            );

            let check = check_codex_mcp_server_at(dir.path(), Path::new("/tmp/rein-local.db"));

            assert_eq!(check.name, "codex_mcp");
            assert_eq!(check.status, CheckStatus::Ok);
            assert!(check.message.contains("loopback"));
        }
    }

    #[test]
    fn test_doctor_warns_on_codex_mcp_non_loopback_url() {
        let dir = tempfile::tempdir().unwrap();
        write_codex_config_raw(
            dir.path(),
            "[mcp_servers.rein]\nurl = \"http://100.64.0.10:8680/mcp\"\n",
        );

        let check = check_codex_mcp_server_at(dir.path(), Path::new("/tmp/rein-local.db"));

        assert_eq!(check.name, "codex_mcp");
        assert_eq!(check.status, CheckStatus::Warn);
        assert!(check.message.contains("non-loopback"));
        assert!(check.message.contains("different machine/database"));
        assert!(check.repair_hint.as_deref().unwrap().contains("stdio"));
    }

    #[test]
    fn test_doctor_warns_on_legacy_codex_mcp_non_loopback_url() {
        let dir = tempfile::tempdir().unwrap();
        write_codex_config_raw(
            dir.path(),
            "[mcp.rein]\nurl = \"http://100.64.0.10:8680/mcp\"\n",
        );

        let check = check_codex_mcp_server_at(dir.path(), Path::new("/tmp/rein-local.db"));

        assert_eq!(check.name, "codex_mcp");
        assert_eq!(check.status, CheckStatus::Warn);
        assert!(check.message.contains("[mcp.rein]"));
        assert!(check.message.contains("non-loopback"));
    }

    #[test]
    fn test_doctor_warns_when_any_codex_mcp_entry_uses_non_loopback_url() {
        let dir = tempfile::tempdir().unwrap();
        write_codex_config_raw(
            dir.path(),
            "[mcp_servers.rein]\ncommand = \"rein\"\nargs = [\"serve\"]\n\n[mcp.rein]\nurl = \"http://100.64.0.10:8680/mcp\"\n",
        );

        let check = check_codex_mcp_server_at(dir.path(), Path::new("/tmp/rein-local.db"));

        assert_eq!(check.name, "codex_mcp");
        assert_eq!(check.status, CheckStatus::Warn);
        assert!(check.message.contains("[mcp.rein]"));
        assert!(check.message.contains("different machine/database"));

        write_codex_config_raw(
            dir.path(),
            "[mcp_servers.rein]\nurl = \"http://100.64.0.10:8680/mcp\"\n\n[mcp.rein]\ncommand = \"rein\"\nargs = [\"serve\"]\n",
        );

        let check = check_codex_mcp_server_at(dir.path(), Path::new("/tmp/rein-local.db"));

        assert_eq!(check.status, CheckStatus::Warn);
        assert!(check.message.contains("[mcp_servers.rein]"));
        assert!(check.message.contains("different machine/database"));
    }

    #[test]
    fn test_doctor_warns_on_codex_mcp_non_stdio_args() {
        let dir = tempfile::tempdir().unwrap();
        write_codex_config_raw(
            dir.path(),
            "[mcp_servers.rein]\ncommand = \"rein\"\nargs = [\"serve\", \"--sse\"]\n",
        );

        let check = check_codex_mcp_server_at(dir.path(), Path::new("/tmp/rein-local.db"));

        assert_eq!(check.name, "codex_mcp");
        assert_eq!(check.status, CheckStatus::Warn);
        assert!(check.message.contains("not stdio MCP"));
    }

    #[test]
    fn test_doctor_warns_on_codex_mcp_wrong_args() {
        let dir = tempfile::tempdir().unwrap();
        write_codex_config_raw(
            dir.path(),
            "[mcp_servers.rein]\ncommand = \"rein\"\nargs = [\"doctor\"]\n",
        );

        let check = check_codex_mcp_server_at(dir.path(), Path::new("/tmp/rein-local.db"));

        assert_eq!(check.name, "codex_mcp");
        assert_eq!(check.status, CheckStatus::Warn);
        assert!(check.message.contains("does not start rein with `serve`"));
    }

    #[test]
    fn test_doctor_warns_on_codex_mcp_stdio_rein_db_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        write_codex_config_raw(
            dir.path(),
            "[mcp_servers.rein]\ncommand = \"rein\"\nargs = [\"serve\"]\n\n[mcp_servers.rein.env]\nREIN_DB = \"/tmp/remote-memories.db\"\n",
        );

        let check = check_codex_mcp_server_at(dir.path(), Path::new("/tmp/local-memories.db"));

        assert_eq!(check.name, "codex_mcp");
        assert_eq!(check.status, CheckStatus::Warn);
        assert!(check.message.contains("REIN_DB"));
        assert!(check.message.contains("different database"));
    }

    #[test]
    fn test_doctor_warns_on_codex_mcp_stdio_rein_config_override() {
        let dir = tempfile::tempdir().unwrap();
        write_codex_config_raw(
            dir.path(),
            "[mcp_servers.rein]\ncommand = \"rein\"\nargs = [\"serve\"]\n\n[mcp_servers.rein.env]\nREIN_CONFIG = \"/tmp/other-config.toml\"\n",
        );

        let check = check_codex_mcp_server_at(dir.path(), Path::new("/tmp/local-memories.db"));

        assert_eq!(check.name, "codex_mcp");
        assert_eq!(check.status, CheckStatus::Warn);
        assert!(check.message.contains("REIN_CONFIG"));
        assert!(check.message.contains("different config/database"));
    }

    #[test]
    fn test_doctor_warns_on_codex_mcp_stdio_home_override() {
        let dir = tempfile::tempdir().unwrap();
        write_codex_config_raw(
            dir.path(),
            "[mcp_servers.rein]\ncommand = \"rein\"\nargs = [\"serve\"]\n\n[mcp_servers.rein.env]\nHOME = \"/tmp/other-home\"\n",
        );

        let check = check_codex_mcp_server_at(dir.path(), Path::new("/tmp/local-memories.db"));

        assert_eq!(check.name, "codex_mcp");
        assert_eq!(check.status, CheckStatus::Warn);
        assert!(check.message.contains("HOME"));
        assert!(check.message.contains("different database"));
    }

    #[test]
    fn test_doctor_warns_on_codex_mcp_stdio_relative_rein_db() {
        let dir = tempfile::tempdir().unwrap();
        write_codex_config_raw(
            dir.path(),
            "[mcp_servers.rein]\ncommand = \"rein\"\nargs = [\"serve\"]\n\n[mcp_servers.rein.env]\nREIN_DB = \"memories.db\"\n",
        );

        let check = check_codex_mcp_server_at(dir.path(), Path::new("/tmp/local-memories.db"));

        assert_eq!(check.name, "codex_mcp");
        assert_eq!(check.status, CheckStatus::Warn);
        assert!(check.message.contains("relative REIN_DB"));
    }

    #[test]
    fn test_doctor_reports_codex_mcp_stdio_matching_rein_db_healthy() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("memories.db");
        write_codex_config_raw(
            dir.path(),
            &format!(
                "[mcp_servers.rein]\ncommand = \"rein\"\nargs = [\"serve\"]\n\n[mcp_servers.rein.env]\nREIN_DB = \"{}\"\n",
                db_path.display()
            ),
        );

        let check = check_codex_mcp_server_at(dir.path(), &db_path);

        assert_eq!(check.name, "codex_mcp");
        assert_eq!(check.status, CheckStatus::Ok);
        assert!(check.message.contains("stdio"));
    }

    #[test]
    #[serial_test::serial(global_state)]
    fn test_doctor_warns_when_loopback_unauth_cannot_apply_to_non_loopback_bind() {
        let _guard = EnvRestore {
            key: "REIN_HTTP_TOKEN",
            value: std::env::var("REIN_HTTP_TOKEN").ok(),
        };
        std::env::remove_var("REIN_HTTP_TOKEN");
        let mut config = ReinConfig::default();
        config.server.sse_enabled = true;
        config.server.sse_bind = "0.0.0.0".to_string();
        config.server.allow_unauthenticated_loopback = true;

        let check = check_http_auth(&config);

        assert_eq!(check.name, "http_auth");
        assert_eq!(check.status, CheckStatus::Fail);
        assert!(check.message.contains("cannot take effect"));
        assert!(check.repair_hint.as_deref().unwrap().contains("127.0.0.1"));
    }

    #[test]
    #[serial_test::serial(global_state)]
    fn test_doctor_warns_when_http_token_overrides_loopback_unauth() {
        let _guard = EnvRestore {
            key: "REIN_HTTP_TOKEN",
            value: std::env::var("REIN_HTTP_TOKEN").ok(),
        };
        std::env::set_var("REIN_HTTP_TOKEN", "doctor-test-token");
        let mut config = ReinConfig::default();
        config.server.sse_enabled = true;
        config.server.sse_bind = "127.0.0.1".to_string();
        config.server.allow_unauthenticated_loopback = true;

        let check = check_http_auth(&config);

        assert_eq!(check.name, "http_auth");
        assert_eq!(check.status, CheckStatus::Warn);
        assert!(check.message.contains("token auth wins"));
        assert!(check
            .repair_hint
            .as_deref()
            .unwrap()
            .contains("unset REIN_HTTP_TOKEN"));

        config.server.sse_bind = "0.0.0.0".to_string();
        let check = check_http_auth(&config);
        assert_eq!(check.status, CheckStatus::Warn);
        assert!(check.message.contains("cannot take effect"));
        assert!(check.message.contains("bearer auth remains required"));
    }

    #[test]
    #[serial_test::serial(global_state)]
    fn test_doctor_warns_when_token_has_no_effect_under_explicit_loopback_policy() {
        let _guard = EnvRestore {
            key: "REIN_HTTP_TOKEN",
            value: std::env::var("REIN_HTTP_TOKEN").ok(),
        };
        std::env::set_var("REIN_HTTP_TOKEN", "doctor-test-token");
        let mut config = ReinConfig::default();
        config.server.sse_enabled = true;
        config.server.auth = Some(crate::config::AuthPolicyConfig::LoopbackOnly);

        let check = check_auth_policy_consistency(&config);

        assert_eq!(check.name, "auth_policy");
        assert_eq!(check.status, CheckStatus::Warn);
        assert!(check.message.contains("token has no effect"));
        assert!(check
            .repair_hint
            .as_deref()
            .unwrap()
            .contains("bearer_required"));
    }

    #[test]
    #[serial_test::serial(global_state)]
    fn test_doctor_fails_wildcard_public_policy_without_allowed_hosts() {
        let _guard = EnvRestore {
            key: "REIN_HTTP_TOKEN",
            value: std::env::var("REIN_HTTP_TOKEN").ok(),
        };
        std::env::remove_var("REIN_HTTP_TOKEN");
        let mut config = ReinConfig::default();
        config.server.sse_enabled = true;
        config.server.sse_bind = "0.0.0.0".to_string();
        config.server.auth = Some(crate::config::AuthPolicyConfig::Public);

        let check = check_auth_policy_consistency(&config);

        assert_eq!(check.name, "auth_policy");
        assert_eq!(check.status, CheckStatus::Fail);
        assert!(check.message.contains("requires [server].allowed_hosts"));
        assert!(check.message.contains("refuse to start"));
    }

    #[test]
    #[serial_test::serial(global_state)]
    fn test_doctor_fails_wildcard_oauth_policy_without_allowed_hosts() {
        let _guard = EnvRestore {
            key: "REIN_HTTP_TOKEN",
            value: std::env::var("REIN_HTTP_TOKEN").ok(),
        };
        std::env::set_var("REIN_HTTP_TOKEN", "doctor-test-token");
        let mut config = ReinConfig::default();
        config.server.sse_enabled = true;
        config.server.sse_bind = "0.0.0.0".to_string();
        config.server.auth = Some(crate::config::AuthPolicyConfig::OAuth);

        let check = check_auth_policy_consistency(&config);

        assert_eq!(check.name, "auth_policy");
        assert_eq!(check.status, CheckStatus::Fail);
        assert!(check.message.contains("auth policy oauth"));
        assert!(check.message.contains("requires [server].allowed_hosts"));
    }

    #[test]
    #[serial_test::serial(global_state)]
    fn test_doctor_reports_legacy_loopback_flag_as_deprecated() {
        let _guard = EnvRestore {
            key: "REIN_HTTP_TOKEN",
            value: std::env::var("REIN_HTTP_TOKEN").ok(),
        };
        std::env::remove_var("REIN_HTTP_TOKEN");
        let mut config = ReinConfig::default();
        config.server.sse_enabled = true;
        config.server.allow_unauthenticated_loopback = true;

        let check = check_auth_policy_consistency(&config);

        assert_eq!(check.name, "auth_policy");
        assert_eq!(check.status, CheckStatus::Ok);
        assert_eq!(check.severity, DoctorSeverity::Info);
        assert!(check.message.contains("deprecated"));
        assert!(check.message.contains("auth = \"public\""));
        assert!(check.message.contains("auth = \"loopback_only\""));
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
    fn release_metadata_versions_match_cargo() {
        let check = check_release_metadata_versions();

        assert_eq!(check.name, "release_metadata_versions");
        assert_eq!(check.status, CheckStatus::Ok);
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
    #[cfg(unix)]
    #[serial_test::serial(global_state)]
    async fn test_doctor_fix_reports_tantivy_rebuild_in_progress() {
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
        store
            .store(test_memory("doctor", "doctor memory", "doctor content"))
            .unwrap();
        let tantivy_dirty = warmup::tantivy_dirty_path(store.db_path());
        std::fs::create_dir_all(tantivy_dirty.parent().unwrap()).unwrap();
        std::fs::write(&tantivy_dirty, b"dirty").unwrap();
        let lock_path = warmup::tantivy_rebuild_lock_path(store.db_path());
        let _lock = hold_file_lock(&lock_path);

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
            .any(|item| item.contains("Tantivy rebuild already in progress")));
        assert!(!report
            .fixes_applied
            .iter()
            .any(|item| item.contains("triggered Tantivy rebuild")));
        let tantivy = report.checks.iter().find(|c| c.name == "tantivy").unwrap();
        assert_eq!(tantivy.status, CheckStatus::Warn);
        assert!(tantivy.message.contains("rebuild in progress"));
        assert!(tantivy_dirty.exists());
    }

    #[tokio::test]
    #[serial_test::serial(global_state)]
    async fn test_doctor_fix_repairs_stale_tantivy_rebuild_marker() {
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
        TantivyFts::open(&store.db_path().with_extension("tantivy")).unwrap();
        let rebuilding = warmup::tantivy_rebuilding_path(store.db_path());
        std::fs::write(&rebuilding, b"rebuilding").unwrap();

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
            .any(|item| item.contains("rebuilt Tantivy index")));
        assert!(
            !rebuilding.exists(),
            "doctor --fix should clear stale Tantivy rebuild marker"
        );
        let tantivy = report.checks.iter().find(|c| c.name == "tantivy").unwrap();
        assert_ne!(tantivy.status, CheckStatus::Warn);
        assert!(!tantivy.message.contains("stale rebuild marker"));
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
        // Phase 3 dropped `ops::registry::*_OPERATIONS`; the helpers now
        // report counts directly from source + inventory. The test asserts
        // the helpers return sensible non-zero values and that the docs
        // counter parsers still work.
        let derived = count_cli_operations_in_source(include_str!("main.rs"));
        let inventory_count = inventory::iter::<crate::ops::OpsCliEntry>().count();
        assert!(
            derived + inventory_count > 0,
            "CLI source scan + inventory should report at least one op"
        );
        let derived_mcp = count_mcp_tools_in_source(include_str!("mcp/server.rs"));
        let inventory_mcp = inventory::iter::<crate::ops::OpsMcpEntry>().count();
        assert!(
            derived_mcp + inventory_mcp > 0,
            "MCP source scan + inventory should report at least one tool"
        );
        let derived_rest = count_rest_operations_in_source(include_str!("mcp/rest.rs"));
        // Exclude test-support ops so this assertion holds regardless of whether
        // the test-support feature is active.
        let inventory_rest = inventory::iter::<crate::ops::OpsRestEntry>()
            .filter(|e| !e.op_name.starts_with("__test_"))
            .count();
        assert!(
            derived_rest + inventory_rest > 0,
            "REST source scan + inventory should report at least one route"
        );
        assert_eq!(
            parse_agents_overview_version("rein v1.2.3 — demo release"),
            Some("1.2.3".to_string())
        );
        assert_eq!(
            parse_documented_mcp_tool_count_line("| **28 MCP tools** | 13 core"),
            Some(28)
        );
    }

    /// v0.28.7+ audit L5 — `doctor --fix` must delete a corrupt
    /// `ars_parameter_policy` metadata row so subsequent
    /// `save_parameter_policy_cas` writes can `INSERT` cleanly. Pre-fix
    /// the row was only surfaced as a warning; the silent stall was
    /// indefinite because every CAS attempt matched 0 rows on
    /// `json_valid(value)` and then short-circuited on the existence
    /// check.
    #[tokio::test]
    async fn doctor_fix_deletes_corrupt_ars_parameter_policy_row() {
        use crate::store::ars_parameter_policy::{
            load_parameter_policy, ArsParameterPolicyLoadStatus,
        };

        let tempdir = tempfile::tempdir().unwrap();
        let config = temp_config(&tempdir);
        let store = config.open_store().unwrap();

        // Plant a corrupt JSON row at the policy key (mirrors the
        // failure mode that `parameter_policy_missing_or_corrupt_loads_disabled`
        // covers in the store layer).
        store
            .conn()
            .execute(
                "INSERT INTO metadata (key, value) VALUES (?1, ?2)",
                rusqlite::params!["ars_parameter_policy", "{not json"],
            )
            .unwrap();
        assert_eq!(
            load_parameter_policy(store.conn()).status,
            ArsParameterPolicyLoadStatus::Corrupt,
            "precondition: planted row must read as Corrupt"
        );

        // Drop the in-test handle so `apply_local_fixes` opens its own
        // connection without contention. (The doctor `run` re-opens via
        // `config.open_store()` internally.)
        drop(store);

        let report = run(
            &config,
            DoctorOptions {
                network: false,
                fix: true,
            },
        )
        .await;

        assert!(
            report
                .fixes_applied
                .iter()
                .any(|line| line.contains("ars_parameter_policy")),
            "fixes_applied must mention ars_parameter_policy deletion (got {:?})",
            report.fixes_applied
        );

        // Re-open and confirm the row is gone.
        let store2 = config.open_store().unwrap();
        assert_eq!(
            load_parameter_policy(store2.conn()).status,
            ArsParameterPolicyLoadStatus::Missing,
            "row must be deleted after --fix"
        );

        // Idempotency: a second --fix run must not re-emit the deletion
        // line (no row to delete) and must not error.
        drop(store2);
        let report2 = run(
            &config,
            DoctorOptions {
                network: false,
                fix: true,
            },
        )
        .await;
        assert!(
            !report2
                .fixes_applied
                .iter()
                .any(|line| line.contains("ars_parameter_policy")),
            "second --fix must be a no-op for the corrupt-policy path (got {:?})",
            report2.fixes_applied
        );
    }

    /// v0.28.7+ audit L5 R4 P2 #2 — `doctor --fix` MUST NOT delete a
    /// healthy `ars_parameter_policy` row when the load encountered a
    /// transient `StorageError` (busy/locked SQLite read). Pre-R4 fix
    /// the recovery branch matched both `Corrupt` AND `StorageError`,
    /// so a transient read failure could destroy a valid canary
    /// policy as collateral damage. Post-fix only `Corrupt` triggers
    /// deletion; `StorageError` is left for the next doctor pass to
    /// retry against the now-unlocked row.
    ///
    /// We can't easily synthesize a real SQLite busy/locked read in a
    /// unit test (the lock would have to span the
    /// `apply_local_fixes`'s `load_parameter_policy` call), so the
    /// test instead asserts the invariant directly via the load
    /// helper: a hand-planted VALID JSON row that successfully loads
    /// must NOT be deleted by `doctor --fix`. This pins the
    /// "delete-only-on-Corrupt" behavior — any future regression that
    /// re-broadens the match arm to non-Corrupt statuses would also
    /// delete this valid row and the test would fail.
    #[tokio::test]
    async fn doctor_fix_preserves_valid_ars_parameter_policy_row() {
        use crate::store::ars_parameter_policy::{
            load_parameter_policy, ArsParameterPolicyLoadStatus,
        };

        let tempdir = tempfile::tempdir().unwrap();
        let config = temp_config(&tempdir);
        let store = config.open_store().unwrap();

        // Plant a syntactically valid policy row at the current schema
        // version; load_parameter_policy will return Loaded.
        let valid_policy = serde_json::json!({
            "schema_version": 1,
            "revision": 5,
            "mode": "canary",
            "source_adaptive_version": 0,
            "runtime_adoption_weight": 0.25,
            "adoption_weights": {},
            "last_event_id": 100,
            "last_updated": "2026-05-04T00:00:00Z",
        });
        store
            .conn()
            .execute(
                "INSERT INTO metadata (key, value) VALUES (?1, ?2)",
                rusqlite::params!["ars_parameter_policy", valid_policy.to_string()],
            )
            .unwrap();
        assert_eq!(
            load_parameter_policy(store.conn()).status,
            ArsParameterPolicyLoadStatus::Loaded,
            "precondition: planted row must read as Loaded"
        );

        drop(store);

        let report = run(
            &config,
            DoctorOptions {
                network: false,
                fix: true,
            },
        )
        .await;

        assert!(
            !report
                .fixes_applied
                .iter()
                .any(|line| line.contains("ars_parameter_policy")),
            "doctor --fix MUST NOT mention ars_parameter_policy when the row \
             is valid; only Corrupt rows trigger deletion. Got fixes_applied: \
             {:?}",
            report.fixes_applied
        );

        // Re-open and confirm the row STILL exists with the original
        // revision — the policy was preserved across --fix.
        let store2 = config.open_store().unwrap();
        let loaded = load_parameter_policy(store2.conn());
        assert_eq!(
            loaded.status,
            ArsParameterPolicyLoadStatus::Loaded,
            "valid policy row must survive doctor --fix"
        );
        assert_eq!(
            loaded.policy.revision, 5,
            "policy revision must be preserved exactly (was 5)"
        );
    }

    /// v0.28.7+ audit L5 R5 P2 — `doctor --fix` MUST NOT delete a row
    /// whose `schema_version` is FUTURE relative to this binary
    /// (downgrade scenario: a newer rein version wrote a payload that
    /// the older binary can't interpret, but the data is valid for
    /// the newer binary). `load_parameter_policy` distinguishes
    /// future-schema (`UnsupportedSchema`) from genuinely-malformed
    /// JSON (`Corrupt`); only the latter triggers deletion. Pre-R5
    /// fix both arms collapsed to `Corrupt` and the recovery branch
    /// would destroy valid future-schema data on every doctor pass.
    #[tokio::test]
    async fn doctor_fix_preserves_future_schema_ars_parameter_policy_row() {
        use crate::store::ars_parameter_policy::{
            load_parameter_policy, ArsParameterPolicyLoadStatus,
        };

        let tempdir = tempfile::tempdir().unwrap();
        let config = temp_config(&tempdir);
        let store = config.open_store().unwrap();

        // Plant a row whose schema_version is far in the future. The
        // JSON parses cleanly, but the schema-version check rejects
        // it, so load_parameter_policy returns `UnsupportedSchema`.
        let future_policy = serde_json::json!({
            "schema_version": 9999,
            "revision": 7,
            "mode": "canary",
            "source_adaptive_version": 0,
            "runtime_adoption_weight": 0.5,
            "adoption_weights": {},
            "last_event_id": 200,
            "last_updated": "2030-01-01T00:00:00Z",
            "future_field_unknown_to_us": "neat",
        });
        let original_value = future_policy.to_string();
        store
            .conn()
            .execute(
                "INSERT INTO metadata (key, value) VALUES (?1, ?2)",
                rusqlite::params!["ars_parameter_policy", &original_value],
            )
            .unwrap();
        let pre_loaded = load_parameter_policy(store.conn());
        assert_eq!(
            pre_loaded.status,
            ArsParameterPolicyLoadStatus::UnsupportedSchema,
            "precondition: future-schema row must read as UnsupportedSchema, \
             not Corrupt; collapsing the two would re-open the deletion bug"
        );

        drop(store);

        let report = run(
            &config,
            DoctorOptions {
                network: false,
                fix: true,
            },
        )
        .await;

        assert!(
            !report
                .fixes_applied
                .iter()
                .any(|line| line.contains("ars_parameter_policy")),
            "doctor --fix MUST NOT delete a future-schema row; it belongs to \
             a newer binary in a downgrade scenario. Got fixes_applied: {:?}",
            report.fixes_applied
        );

        // Re-open and confirm the row STILL exists with the original
        // raw value bit-for-bit (no rewrite, no clamp, no anything).
        let store2 = config.open_store().unwrap();
        let raw_now: String = store2
            .conn()
            .query_row(
                "SELECT value FROM metadata WHERE key = 'ars_parameter_policy'",
                [],
                |row| row.get(0),
            )
            .expect("future-schema row must survive doctor --fix");
        assert_eq!(
            raw_now, original_value,
            "future-schema row's raw bytes must be preserved exactly"
        );
        // And re-loading still produces UnsupportedSchema, not Corrupt.
        assert_eq!(
            load_parameter_policy(store2.conn()).status,
            ArsParameterPolicyLoadStatus::UnsupportedSchema
        );
    }
}
