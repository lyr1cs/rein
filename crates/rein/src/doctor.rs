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
    checks.push(check_retired_llm_models(config));
    checks.push(check_judge_llm_provider(config));
    checks.push(check_supermemory(config));
    checks.push(check_http_auth(config));
    checks.push(check_auth_policy_consistency(config));
    checks.push(check_oauth_provider(config));
    checks.push(check_proxy_auth(config));
    checks.push(check_codex_hooks());
    checks.push(check_claude_hooks());
    checks.push(check_codex_mcp_server(config));
    checks.push(check_gui_runtime(config));
    checks.push(check_proxy_runtime(config));
    checks.push(check_overview_version());
    checks.push(check_release_metadata_versions());
    checks.push(check_eval_gates());
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
                            checks.push(check_dedup_threshold_observability(&store, config));
                            checks.push(check_recall_fusion_calibration(&store, config));
                            checks.push(check_a12_input_epoch(&store));
                            checks.push(check_adaptive_pipeline_last_run(&store));
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
    fail_with_hint(
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

/// v0.32 T&M Phase 2 — surfaces eval-gate freshness + ship/bail status.
///
/// Reads `docs/eval-baselines/{name}.json` (committed) and
/// `target/eval-gates/{name}-run.json` (gitignored, per-build) anchored at
/// `env::current_dir()` — operators are expected to invoke `rein doctor`
/// from the source-repo root.  In a deployed binary far from source
/// repos the scorecards won't be found and every gate degrades to
/// `no_baseline`; the check then surfaces that as informational rather
/// than failing the doctor sequence.
///
/// Aggregation rules:
/// - any non-stub gate in `Bail` → WARN
/// - any non-stub gate with a baseline > 30 days old → WARN
/// - all non-stub gates `Ship` or `no_run` or stubs → OK
/// - no baselines exist for any non-stub gate → OK (with hint to run
///   `rein-eval gate baseline --gate all`)
fn check_eval_gates() -> DoctorCheck {
    use crate::eval::gates::{self, ScorecardStatus};
    let repo_root = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    let target_dir = repo_root.join("target");
    let now = chrono::Utc::now().timestamp();
    const STALE_DAYS: i64 = 30;

    let mut concerns: Vec<String> = Vec::new();
    let mut ship_count = 0usize;
    let mut no_baseline_count = 0usize;
    let mut stub_count = 0usize;
    let mut total_non_stub = 0usize;

    for gate in gates::all_gates() {
        let name = gate.name();
        if gate.is_stub() {
            stub_count += 1;
            continue;
        }
        total_non_stub += 1;
        let baseline_path = gates::baseline_path(&repo_root, name);
        let run_path = gates::run_path(&target_dir, name);
        // v0.32 R4 P2-#2: distinguish "file does not exist" from "file
        // exists but is unparseable".  Earlier `.ok()` swallowed both as
        // `None` and the doctor reported no_baseline / OK for a corrupt
        // committed JSON file.  Now corrupt artifacts are surfaced as
        // explicit concerns.
        let baseline = match gates::load_scorecard(&baseline_path) {
            gates::ScorecardLoad::Loaded(sc) => Some(sc),
            gates::ScorecardLoad::Missing => None,
            gates::ScorecardLoad::Corrupt(msg) => {
                concerns.push(format!("{name} baseline scorecard corrupt: {msg}"));
                None
            }
        };
        let current = match gates::load_scorecard(&run_path) {
            gates::ScorecardLoad::Loaded(sc) => Some(sc),
            gates::ScorecardLoad::Missing => None,
            gates::ScorecardLoad::Corrupt(msg) => {
                concerns.push(format!("{name} run scorecard corrupt: {msg}"));
                None
            }
        };

        let Some(b) = baseline.as_ref() else {
            no_baseline_count += 1;
            continue;
        };

        // Stale baseline check.
        let age_days = (now - b.created_at).max(0) / 86_400;
        if age_days > STALE_DAYS {
            concerns.push(format!(
                "{name} baseline is {age_days}d old (>{STALE_DAYS}d)"
            ));
        }

        // Bail check (requires both sides).
        if current.is_some() {
            let cmp = gates::compare_scorecards(
                name,
                baseline.as_ref(),
                current.as_ref(),
                gates::DEFAULT_NOISE_FLOOR,
            );
            match cmp.status {
                ScorecardStatus::Ship => ship_count += 1,
                ScorecardStatus::Bail => {
                    concerns.push(format!("{name} run is in Bail vs baseline"));
                }
                ScorecardStatus::NoData => {
                    // v0.32 R6 P2: distinguish "comparison ran but
                    // McNemar CI is too wide to call" (mcnemar.is_some
                    // — informational, more samples would help) from
                    // "comparison rejected before McNemar could run
                    // due to mismatched gate_name / kind / schema /
                    // fixture-id sets" (mcnemar.is_none — the
                    // committed/generated scorecards are miswired and
                    // need operator intervention to repair).  Earlier
                    // we dropped both cases as informational, so a
                    // stale or wrong-gate scorecard would silently
                    // pass `rein doctor` even though the comparison
                    // layer caught it.
                    if cmp.mcnemar.is_none() {
                        concerns.push(format!("{name} comparison invalid: {}", cmp.reason));
                    }
                    // mcnemar.is_some() with NoData = legitimately
                    // underpowered (CI straddles the noise floor).
                    // Informational — doctor surfaces the count via
                    // the existing total message rather than as a
                    // per-gate concern.
                }
            }
        }
    }

    let message = format!(
        "{} gate(s): {ship_count} ship, {no_baseline_count} no_baseline, {stub_count} stub",
        total_non_stub + stub_count,
    );

    if !concerns.is_empty() {
        warn_in(
            DoctorCategory::Architecture,
            "eval_gates",
            format!("{message}; concerns: {}", concerns.join("; ")),
        )
    } else if no_baseline_count == total_non_stub && total_non_stub > 0 {
        let mut check = ok_in(
            DoctorCategory::Architecture,
            "eval_gates",
            format!("{message} (no baselines yet — run `rein-eval gate baseline --gate all`)"),
        );
        check.repair_hint =
            Some("cargo run -p rein --bin rein-eval -- gate baseline --gate all".to_string());
        check
    } else {
        ok_in(DoctorCategory::Architecture, "eval_gates", message)
    }
}

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

/// Claude Code hook wiring in `~/.claude/settings.json`: which `rein hook`
/// commands are installed. Hooks are what feed Claude Code sessions into the
/// memory database (PostToolUse / PreCompact / Stop) and, with
/// `[hooks.claude].inject_prompt_context`, inject prompt-time context
/// (UserPromptSubmit).
fn check_claude_hooks() -> DoctorCheck {
    let Some(home) = std::env::var_os("HOME").map(PathBuf::from) else {
        return ok_in(
            DoctorCategory::Configuration,
            "claude_hooks",
            "HOME not set; skipping Claude Code hook checks",
        );
    };
    check_claude_hooks_at(&home.join(".claude").join("settings.json"))
}

const CLAUDE_HOOK_EVENTS: [(&str, &str); 4] = [
    ("PostToolUse", "rein hook post"),
    ("PreCompact", "rein hook compact"),
    ("Stop", "rein hook stop"),
    ("UserPromptSubmit", "rein hook prompt"),
];

fn check_claude_hooks_at(settings_path: &Path) -> DoctorCheck {
    let raw = match std::fs::read_to_string(settings_path) {
        Ok(raw) => raw,
        Err(_) => {
            return ok_in(
                DoctorCategory::Configuration,
                "claude_hooks",
                "Claude Code settings.json not found; skipping Claude Code hook checks",
            );
        }
    };
    let Ok(root) = serde_json::from_str::<serde_json::Value>(&raw) else {
        return warn_with_hint(
            DoctorCategory::Configuration,
            "claude_hooks",
            "Claude Code settings.json is not valid JSON",
            "fix ~/.claude/settings.json, then add the rein hook entries from the README",
        );
    };
    let hooks = root.get("hooks").and_then(|value| value.as_object());
    let installed: Vec<&str> = CLAUDE_HOOK_EVENTS
        .iter()
        .filter(|(event, command)| {
            hooks
                .and_then(|hooks| hooks.get(*event))
                .and_then(|entries| entries.as_array())
                .is_some_and(|entries| {
                    entries.iter().any(|entry| {
                        entry
                            .get("hooks")
                            .and_then(|inner| inner.as_array())
                            .is_some_and(|inner| {
                                inner.iter().any(|hook| {
                                    hook.get("command")
                                        .and_then(|value| value.as_str())
                                        .is_some_and(|value| value.contains(command))
                                })
                            })
                    })
                })
        })
        .map(|(event, _)| *event)
        .collect();
    let missing: Vec<&str> = CLAUDE_HOOK_EVENTS
        .iter()
        .map(|(event, _)| *event)
        .filter(|event| !installed.contains(event))
        .collect();
    if installed.is_empty() {
        return warn_with_hint(
            DoctorCategory::Configuration,
            "claude_hooks",
            "no rein hooks in Claude Code settings.json; Claude Code sessions are not feeding the memory database",
            "add the PostToolUse / PreCompact / Stop (and optional UserPromptSubmit) `rein hook` entries from the README \"Hook Setup for Claude Code\" section",
        );
    }
    let mut message = format!("Claude Code hooks installed: {}", installed.join(", "));
    if !missing.is_empty() {
        message.push_str(&format!("; not installed: {}", missing.join(", ")));
    }
    ok_in(DoctorCategory::Configuration, "claude_hooks", message)
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

const RETIRED_GEMINI_FLASH_LITE_PREVIEW_MODEL: &str = "gemini-3.1-flash-lite-preview";
const STABLE_GEMINI_FLASH_LITE_MODEL: &str = "gemini-3.1-flash-lite";

fn judge_runtime_active(config: &ReinConfig) -> bool {
    config.ars.llm_judge.enabled
        && (config.ars.llm_judge.synthesis_enabled || config.ars.llm_judge.concept_summary_enabled)
}

fn judge_nightly_active(config: &ReinConfig) -> bool {
    config.ars.llm_judge.enabled && config.ars.llm_judge.nightly_cron.enabled
}

/// Warn when an enabled LLM consumer resolves to the retired preview model.
/// Resolution is authoritative: inherited sections are reported under the
/// consumer that will actually call the model, while disabled opt-in sections
/// are ignored. This check is deliberately read-only.
fn check_retired_llm_models(config: &ReinConfig) -> DoctorCheck {
    let raw_provider_is_active =
        |provider: &str| !matches!(provider.trim().to_ascii_lowercase().as_str(), "" | "none");
    let sections = [
        ("extract", true),
        (
            "extract.async_memory",
            raw_provider_is_active(&config.async_memory.provider),
        ),
        (
            "extract.intelligent_merge",
            config.intelligent_merge.enabled,
        ),
        ("extract.dedup", true),
        ("query_expansion", true),
        (
            "search.llm_reranker",
            raw_provider_is_active(&config.search.llm_reranker),
        ),
        ("ars.recall_synthesis", config.ars.recall_synthesis_enabled),
        ("ars.concept_summary", config.ars.concept_summary_enabled),
        ("ars.cold_archive", config.ars.cold_archive_enabled),
        ("resummerize", config.resummerize.enabled),
        ("ars.llm_judge", judge_runtime_active(config)),
        ("ars.llm_judge.nightly_cron", judge_nightly_active(config)),
    ];
    let mut retired_sections = sections
        .into_iter()
        .filter(|(_, active)| *active)
        .filter_map(|(section, _)| {
            let resolved = config.resolve_llm_for(section).ok()?;
            (resolved.provider == Provider::Google
                && resolved.model == RETIRED_GEMINI_FLASH_LITE_PREVIEW_MODEL)
                .then_some(section)
        })
        .collect::<Vec<_>>();
    retired_sections.sort_unstable();
    retired_sections.dedup();

    if retired_sections.is_empty() {
        return ok_in(
            DoctorCategory::Configuration,
            "llm_model_lifecycle",
            "no active resolved LLM section uses a retired model id".to_string(),
        );
    }

    DoctorCheck {
        name: "llm_model_lifecycle",
        category: DoctorCategory::Configuration,
        severity: DoctorSeverity::Warning,
        status: CheckStatus::Warn,
        fixable: false,
        message: format!(
            "retired model {} is active in resolved LLM sections: {}",
            RETIRED_GEMINI_FLASH_LITE_PREVIEW_MODEL,
            retired_sections.join(", ")
        ),
        repair_hint: Some(format!(
            "Replace {} with the stable model id {} in the listed operator config sections. This diagnostic does not rewrite operator config automatically.",
            RETIRED_GEMINI_FLASH_LITE_PREVIEW_MODEL, STABLE_GEMINI_FLASH_LITE_MODEL
        )),
    }
}

/// Active judge consumers must resolve an actual provider. `enabled = true`
/// with `Provider::None` otherwise looks configured while both workers skip
/// their LLM calls. Runtime and nightly are independent production consumers.
fn check_judge_llm_provider(config: &ReinConfig) -> DoctorCheck {
    let consumers = [
        (
            "ars.llm_judge runtime",
            "ars.llm_judge",
            judge_runtime_active(config),
        ),
        (
            "ars.llm_judge.nightly_cron",
            "ars.llm_judge.nightly_cron",
            judge_nightly_active(config),
        ),
    ];
    let missing = consumers
        .into_iter()
        .filter(|(_, _, active)| *active)
        .filter_map(|(label, section, _)| {
            config
                .resolve_llm_for(section)
                .ok()
                .filter(|resolved| resolved.provider == Provider::None)
                .map(|_| label)
        })
        .collect::<Vec<_>>();

    if missing.is_empty() {
        return ok_in(
            DoctorCategory::Configuration,
            "judge_llm_provider",
            "all active judge consumers resolve an LLM provider".to_string(),
        );
    }

    DoctorCheck {
        name: "judge_llm_provider",
        category: DoctorCategory::Configuration,
        severity: DoctorSeverity::Warning,
        status: CheckStatus::Warn,
        fixable: false,
        message: format!(
            "active judge consumers resolve to Provider::None and will not make LLM calls: {}",
            missing.join(", ")
        ),
        repair_hint: Some(
            "Set [llm] provider = \"google\" or \"omlx\" with a model, or disable the listed judge consumer flags. This diagnostic is read-only."
                .to_string(),
        ),
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
    // v0.35 Phase 3: the legacy `allow_unauthenticated_loopback` bool was
    // removed; the only remaining branches are "explicit policy set" (handled
    // above) and "no explicit policy set, no token" — which is now an outright
    // FAIL because the implicit fallback path is gone. Configs that previously
    // relied on the bool are transformed at TOML load time by
    // `config::migrate_legacy_server_auth` into an explicit policy, so by the
    // time this check runs the only way to land in the no-policy branch is a
    // fresh-install config that never set anything.
    if token_present {
        ok_in(
            DoctorCategory::Configuration,
            "http_auth",
            format!(
                "token configured for {}:{}",
                config.server.sse_bind, config.server.sse_port
            ),
        )
    } else {
        fail_with_hint(
            DoctorCategory::Configuration,
            "http_auth",
            format!(
                "HTTP/SSE is enabled on {}:{} without REIN_HTTP_TOKEN and no explicit [server].auth",
                config.server.sse_bind, config.server.sse_port
            ),
            "set [server].auth = \"public\" / \"loopback_only\" / \"bearer_required\" / \"oauth\", or set REIN_HTTP_TOKEN=<secret> for bearer auth",
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

    // v0.35 Phase 3: the legacy `auth.is_none() && allow_unauthenticated_loopback`
    // branch is gone — the field was removed from ServerConfig and configs
    // that still carry the bool are transformed at TOML load time
    // (`config::migrate_legacy_server_auth`). By the time this check runs,
    // either `[server].auth` is set explicitly or `resolve_auth_policy`
    // would have errored above.

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
    // v1.2 Phase 3: explicit `[proxy].auth = "public"` (loopback bind only)
    // replaces the legacy `allow_unauthenticated_loopback` bool.
    let allow_unauth =
        config.proxy.auth == Some(crate::config::ProxyAuthPolicyConfig::Public) && is_loopback;
    // codex v1.2 R2 P2: an EXPLICIT `bearer_required` without a token is an
    // actionable misconfiguration (`rein proxy on` will refuse immediately),
    // so it must never take the fresh-install OK shortcut below — that
    // shortcut exists for the implicit default-deny posture only.
    let explicit_bearer_without_token = !token_present
        && config.proxy.auth == Some(crate::config::ProxyAuthPolicyConfig::BearerRequired);
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
        && !explicit_bearer_without_token
        && is_loopback
        && crate::service::is_running("proxy").is_none()
    {
        return ok_in(
            DoctorCategory::Configuration,
            "proxy_auth",
            format!(
                "proxy not running on {}:{} (set REIN_PROXY_TOKEN or [proxy].auth = \"public\" before `rein proxy on`)",
                config.proxy.bind, config.proxy.port
            ),
        );
    }

    // codex v1.2 R3 P2: explicit public must report BEFORE token presence —
    // the runtime honors `[proxy].auth = "public"` on loopback even when a
    // token is still exported (`resolve_proxy_startup_auth` returns None),
    // so "token configured" would describe a token the proxy never checks
    // and hide the actual unauthenticated posture.
    if allow_unauth {
        ok_in(
            DoctorCategory::Configuration,
            "proxy_auth",
            format!(
                "[proxy].auth = \"public\": loopback-only unauthenticated access for {}:{}{}",
                config.proxy.bind,
                config.proxy.port,
                if token_present {
                    " (exported proxy token is NOT enforced under explicit public)"
                } else {
                    ""
                }
            ),
        )
    } else if token_present {
        ok_in(
            DoctorCategory::Configuration,
            "proxy_auth",
            format!(
                "token configured for {}:{}",
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
            "set REIN_PROXY_TOKEN=<secret> or set [proxy].auth = \"public\" for loopback-only unauthenticated access",
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

    // v0.30.2 B2/B4: clean orphan side-index staging dirs/files from any
    // prior interrupted rebuild before deciding what to repair. Sequence
    // matters: stale `.tantivy.new` would otherwise confuse the rebuild
    // path's own `remove_dir_all` of the staging target.
    warmup::cleanup_tantivy_staging(store.db_path());
    warmup::cleanup_hnsw_staging(store.db_path());

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

    // v0.30.2 B5: TTL reset for stranded HNSW `.rebuilding` markers. A
    // panicked rebuild thread used to leave the marker in place, which
    // forced every later recall to drop to sqlite-vec O(n) brute force
    // forever. The spawned closure in `search/recall.rs` now has a
    // `catch_unwind` guard that restores `.dirty`, but operators upgrading
    // mid-incident may already have stale markers on disk; the 1-hour
    // TTL covers that recovery path.
    if let Some(reset_marker) = warmup::reset_stale_hnsw_rebuilding(
        store.db_path(),
        std::time::Duration::from_secs(60 * 60),
    ) {
        fixes.push(format!(
            "reset stale HNSW .rebuilding marker {} (>1h old) — next request triggers retry",
            reset_marker.display()
        ));
    }

    // v0.30.4 D4: symmetric Tantivy TTL reset.  See
    // `warmup::reset_stale_tantivy_rebuilding` for rationale — same 1h
    // TTL as HNSW for cross-index consistency.  Pre-D4, an interrupted
    // tantivy rebuild that left a `.rebuilding` marker without `.dirty`
    // would drop recall to FTS5 forever; the recall-path `StaleMarker`
    // re-trigger added in R23 couldn't always recover because a zombie
    // worker holding the flock blocked the spawn-time lock-acquire.
    if let Some(reset_marker) = warmup::reset_stale_tantivy_rebuilding(
        store.db_path(),
        std::time::Duration::from_secs(60 * 60),
    ) {
        fixes.push(format!(
            "reset stale Tantivy .rebuilding marker {} (>1h old) — next request triggers retry",
            reset_marker.display()
        ));
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

    // P2-1 epoch recovery. A missing `a12_input_epoch` row is re-created at
    // zero (same idempotent heal the schema bring-up applies on every open);
    // a malformed row is re-baselined ONLY together with invalidating the
    // active A12 calibration, so no sealed run keeps serving against an
    // untrusted counter. Both are no-ops on a healthy row.
    match store.conn().execute(
        "INSERT OR IGNORE INTO metadata(key, value) VALUES ('a12_input_epoch', '0')",
        [],
    ) {
        Ok(inserted) if inserted > 0 => fixes.push(
            "re-created missing a12_input_epoch row at zero; sealed A12 runs recalibrate fail-closed"
                .to_string(),
        ),
        Ok(_) => {}
        Err(e) => fixes.push(format!("failed to re-create a12_input_epoch row: {e}")),
    }
    match crate::store::a12_calibration::repair_malformed_a12_input_epoch(store.conn()) {
        Ok(outcome) if outcome.reset => fixes.push(format!(
            "re-baselined malformed a12_input_epoch (was {:?}) and invalidated the active A12 calibration ({})",
            outcome.observed_value.unwrap_or_default(),
            if outcome.calibration_invalidated {
                "active pointer cleared"
            } else {
                "no active pointer present"
            }
        )),
        Ok(_) => {}
        Err(e) => fixes.push(format!("failed to re-baseline a12_input_epoch: {e}")),
    }

    // P2-2: schema-2 (or otherwise corrupt) A12 active-pointer recovery.
    // Mirrors the parameter-policy repair above: the helper re-checks status
    // under `BEGIN IMMEDIATE` and deletes only a confirmed-Corrupt active
    // row — future-schema rows and immutable revision rows are preserved —
    // so the next `refresh_a12_calibration` tick reseals fresh instead of
    // early-returning Unhealthy forever.
    match crate::store::a12_calibration::repair_corrupt_a12_calibration(store.conn()) {
        Ok(outcome) if outcome.deleted > 0 => fixes.push(format!(
            "deleted corrupt a12_calibration active pointer ({} row{}): {}",
            outcome.deleted,
            if outcome.deleted == 1 { "" } else { "s" },
            outcome
                .error_at_delete
                .unwrap_or_else(|| "unknown error".to_string())
        )),
        Ok(_) => {}
        Err(e) => fixes.push(format!(
            "failed to delete corrupt a12_calibration active pointer: {e}"
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

fn check_dedup_threshold_observability(
    store: &crate::store::SqliteStore,
    config: &ReinConfig,
) -> DoctorCheck {
    let observability = crate::ops::dedup_threshold_observability(store, config);
    project_dedup_threshold_doctor_check(&observability)
}

fn project_dedup_threshold_doctor_check(
    observability: &crate::ops::DedupThresholdObservability,
) -> DoctorCheck {
    use crate::store::dedup_calibration::{DedupCalibrationLoadStatus, DedupCalibrationStatus};

    let calibration = &observability.calibration;
    let policy = &calibration.policy;
    let slices = policy
        .required_slices
        .iter()
        .map(|slice| {
            format!(
                "{}:+{}/-{}:{:?}",
                slice.name, slice.positive_count, slice.negative_count, slice.status
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let counts = &calibration.counterfactual_counts;
    let message = format!(
        "static={:.3} shadow={:.3} hard_effective={:.3} source={} hard_source={} \
         adaptive_enabled={} evidence_verified={} applied={} reason={} \
         policy_schema={} seal_schema={:?} policy_revision={} policy_status={:?} load_status={:?} \
         seal_status={:?} runtime_status={:?} scale={:?} provenance={:?} \
         train=+{}/-{} sealed=+{}/-{} supplemental_exact={} supplemental_structural={} \
         fp_count={} fp_ucb95={:?} utility_status={:?} utility_n={} \
         utility_baseline_only={} utility_candidate_only={} utility_miss_ucb95={:?} \
         confusion=tp{}/fp{}/tn{}/fn{} slices=[{}] \
         train_fingerprint={} holdout_fingerprint={} corpus_fingerprint={} \
         counterfactual=total{}/evaluable{}/probed_shadow{}/hard_effective{}",
        observability.static_threshold,
        observability.shadow_threshold,
        observability.hard_effective_threshold,
        observability.source,
        observability.hard_effective_source,
        calibration.adaptive_enabled,
        calibration.evidence_verified,
        calibration.applied,
        calibration.reason,
        calibration.policy_schema_version,
        calibration.seal_schema_version,
        policy.revision,
        policy.status,
        calibration.load_status,
        calibration.seal_status,
        calibration.runtime_status,
        policy.scale,
        policy.provenance,
        policy.train_positive_count,
        policy.train_negative_count,
        policy.sealed_positive_count,
        policy.sealed_negative_count,
        policy.sealed_exact_positive_count,
        policy.sealed_structural_negative_count,
        policy.false_positive_count,
        policy.false_positive_upper_95,
        policy.utility.status,
        policy.utility.n,
        policy.utility.baseline_only_hits,
        policy.utility.candidate_only_hits,
        policy.utility.miss_rate_upper_95,
        policy.holdout_confusion.true_positives,
        policy.holdout_confusion.false_positives,
        policy.holdout_confusion.true_negatives,
        policy.holdout_confusion.false_negatives,
        slices,
        policy.train_fingerprint,
        policy.holdout_fingerprint,
        policy.corpus_fingerprint,
        counts.total_events,
        counts.evaluable_events,
        counts.would_merge_at_probed_shadow,
        counts.would_merge_at_hard_effective,
    );

    let static_is_valid = observability.static_threshold.is_finite()
        && (0.0..=1.0).contains(&observability.static_threshold);
    let hard_is_valid = observability.hard_effective_threshold.is_finite()
        && (0.0..=1.0).contains(&observability.hard_effective_threshold);
    let hard_below_static = static_is_valid
        && hard_is_valid
        && observability.hard_effective_threshold + f64::EPSILON < observability.static_threshold;
    if !static_is_valid || !hard_is_valid || hard_below_static {
        return fail_with_hint(
            DoctorCategory::Configuration,
            "dedup_threshold_observability",
            format!("destructive dedup resolver violated the static safety floor; {message}"),
            "stop destructive dedup, reset the calibration policy and independent seal atomically, then run a fresh sealed recalibration; this diagnostic never mutates calibration state",
        );
    }

    let unhealthy_load = !matches!(
        calibration.runtime_status,
        DedupCalibrationLoadStatus::Missing | DedupCalibrationLoadStatus::Loaded
    ) || !matches!(
        calibration.seal_status,
        DedupCalibrationLoadStatus::Missing | DedupCalibrationLoadStatus::Loaded
    ) || (calibration.load_status == DedupCalibrationLoadStatus::Missing
        && calibration.seal_status == DedupCalibrationLoadStatus::Loaded)
        || (calibration.load_status == DedupCalibrationLoadStatus::Loaded
            && calibration.seal_status == DedupCalibrationLoadStatus::Missing);
    let terminal_needs_attention = calibration.evidence_verified
        && matches!(
            policy.status,
            DedupCalibrationStatus::Bail | DedupCalibrationStatus::NoData
        );
    if unhealthy_load || terminal_needs_attention {
        return DoctorCheck {
            name: "dedup_threshold_observability",
            category: DoctorCategory::Configuration,
            severity: DoctorSeverity::Warning,
            status: CheckStatus::Warn,
            fixable: false,
            message,
            repair_hint: observability.repair_advice.first().cloned(),
        };
    }

    DoctorCheck {
        name: "dedup_threshold_observability",
        category: DoctorCategory::Configuration,
        severity: DoctorSeverity::Info,
        status: CheckStatus::Ok,
        fixable: false,
        message,
        repair_hint: observability.repair_advice.first().cloned(),
    }
}

/// Format the shared A12 activation projection. The projection reads recall
/// scorecards through the shared artifact-root resolver (read-only) to attest
/// the current eval gate; it never persists those paths and never changes the
/// active pointer or policy. Recovery is always an operator/producer action;
/// doctor never repairs calibration evidence.
fn check_recall_fusion_calibration(
    store: &crate::store::SqliteStore,
    config: &ReinConfig,
) -> DoctorCheck {
    let report = crate::ops::a12_activation::collect_recall_fusion_activation_report(
        store,
        config,
        chrono::Utc::now().timestamp_millis(),
    );
    let scope_summary = if report.scopes.is_empty() {
        "scopes=[]".to_string()
    } else {
        report
            .scopes
            .iter()
            .map(|scope| {
                format!(
                    "{}[basis={:?} code={:?} active={} adoption={:.3} human_ess={} train_ess={} \
                     holdout_ess={} verdict={:?} gate={:?} generation={:?} revision={:?} \
                     valid_now={} calibrated_at={:?} evaluated_at={:?} valid_until={:?} reason={}]",
                    scope.scope,
                    scope.basis,
                    scope.health_code,
                    scope.active,
                    scope.adoption_weight,
                    scope.human_ess,
                    scope.train_family_ess,
                    scope.holdout_family_ess,
                    scope.verdict,
                    scope.recall_gate_status,
                    scope.a12_generation,
                    scope.a12_revision,
                    scope.valid_now,
                    scope.calibrated_at,
                    scope.evaluated_at,
                    scope.valid_until_exclusive,
                    scope.reason,
                )
            })
            .collect::<Vec<_>>()
            .join(" ")
    };
    let summary = format!(
        "activation_status={} health_status={} active={} policy_load_status={} policy_mode={} \
         a12_load_status={} current_adaptive_version={} source_adaptive_version={} \
         a12_generation={:?} a12_revision={:?} reason={} {}",
        report.activation_status,
        report.health_status,
        report.active,
        report.policy_load_status,
        report.policy_mode,
        report.a12_load_status,
        report.current_adaptive_version,
        report.source_adaptive_version,
        report.a12_generation,
        report.a12_revision,
        report.reason,
        scope_summary,
    );
    // Keyed on typed codes, never on reason prose. Benign absence — fresh
    // install, disabled/shadow-only config, or a policy without recall-fusion
    // evidence — reports Ok like the sibling ars_parameter_policy and dedup
    // observability checks; only corrupt-class loads, Bail/NoData verdicts,
    // and genuinely degraded scopes warrant operator attention.
    let unhealthy_scope = report.scopes.iter().any(|scope| {
        matches!(
            scope.verdict,
            Some(
                crate::store::a12_calibration::A12CalibrationVerdict::Bail
                    | crate::store::a12_calibration::A12CalibrationVerdict::NoData
            )
        ) || scope.health_code.is_degraded()
    });
    let unhealthy_load =
        |status: &str| matches!(status, "corrupt" | "unsupported_schema" | "storage_error");
    let attention = unhealthy_scope
        || unhealthy_load(&report.a12_load_status)
        || unhealthy_load(&report.policy_load_status)
        || matches!(
            report.health_status.as_str(),
            "bail" | "no_data" | "degraded"
        );
    if !attention {
        return ok_in(
            DoctorCategory::Configuration,
            "recall_fusion_calibration",
            summary,
        );
    }

    let advice = match report.health_status.as_str() {
        "missing" | "static" | "no_data" | "policy_missing" | "a12_missing" => {
            "Collect both training and permanent-holdout family evidence, produce a current Ship recall eval-gate attestation, and let the adaptive pipeline seal a new policy revision. Installed daemons outside a checkout must set REIN_EVAL_GATE_ROOT to an absolute artifact root; development runs may discover a checkout ancestor from the working directory. This check is read-only."
        }
        "bail" => {
            "Inspect the paired holdout regression, then calibrate a new immutable A12 generation after correcting the producer or retrieval behavior. Bail evidence is preserved and never overridden by doctor."
        }
        "policy_unsupported_schema" | "a12_unsupported_schema" => {
            "Upgrade rein to a binary that understands this policy/A12 schema. Preserve the active pointer and immutable revision bytes unchanged; this check never rewrites them."
        }
        "policy_corrupt" | "a12_corrupt" => {
            "Preserve the policy/A12 evidence for diagnosis, then run `rein doctor --fix`: it deletes only a confirmed-corrupt active row (immutable A12 revisions and future-schema bytes are preserved) so the next calibration/refresh tick reseals fresh. This read-only check itself never repairs."
        }
        "policy_storage_error" | "a12_storage_error" => {
            "Storage errors are usually transient (busy/locked reads); retry doctor once the lock clears. No destructive recovery runs for storage errors — the underlying row may still be healthy."
        }
        _ => {
            "Refresh the stale, expired, fingerprint-mismatched, or tampered A12 generation and seal a matching policy revision. A scope serving only its sealed human fallback recovers the same way: recalibrate so the automatic candidate is current again. This check reports evidence only and performs no repair."
        }
    };
    let mut check = warn_in(
        DoctorCategory::Configuration,
        "recall_fusion_calibration",
        summary,
    );
    check.repair_hint = Some(advice.to_string());
    check
}

/// The v4 `a12_input_epoch` row is the O(1) invalidation token for online
/// A12. When it is missing, every epoch-guarded write aborts until the next
/// database open self-heals the row at zero; when it is malformed, writes
/// stay aborted until an explicit repair. Read-only here — `rein doctor
/// --fix` re-baselines a malformed row only together with invalidating the
/// active A12 calibration so no sealed run can serve against an untrusted
/// counter.
/// Surface the last `run_adaptive_pipeline` pass recorded by
/// [`crate::ops::pipeline_run`]: outcome, age, total time and the slowest
/// stages. Warns when the pipeline never ran, failed, or has reported
/// `running` for longer than six hours (a killed pass).
fn check_adaptive_pipeline_last_run(store: &crate::store::SqliteStore) -> DoctorCheck {
    check_adaptive_pipeline_last_run_at(store, chrono::Utc::now().timestamp_millis())
}

fn check_adaptive_pipeline_last_run_at(
    store: &crate::store::SqliteStore,
    now_unix_ms: i64,
) -> DoctorCheck {
    use crate::ops::pipeline_run::{
        load_last_run, PipelineRunOutcome, PIPELINE_RUN_STALE_RUNNING_MS,
    };

    let Some(record) = load_last_run(store.conn()) else {
        let mut check = warn_in(
            DoctorCategory::Storage,
            "adaptive_pipeline_last_run",
            "adaptive pipeline has never completed a recorded pass on this database",
        );
        check.repair_hint = Some(
            "Run `rein gc --threshold 0` (or any consolidate/dedup pass) to run the adaptive pipeline without pruning memories."
                .to_string(),
        );
        return check;
    };
    let age_secs = now_unix_ms.saturating_sub(record.started_at_unix_ms).max(0) / 1000;
    let summary = format!(
        "{} trigger={} started {}s ago",
        record.summary_line(now_unix_ms),
        record.trigger,
        age_secs
    );
    match record.outcome {
        PipelineRunOutcome::Completed | PipelineRunOutcome::SkippedDisabled => ok_in(
            DoctorCategory::Storage,
            "adaptive_pipeline_last_run",
            summary,
        ),
        PipelineRunOutcome::Failed => {
            let mut check = warn_in(
                DoctorCategory::Storage,
                "adaptive_pipeline_last_run",
                summary,
            );
            check.repair_hint = Some(
                "Inspect the failed stage detail above, fix the cause, and re-run `rein gc --threshold 0`."
                    .to_string(),
            );
            check
        }
        PipelineRunOutcome::Running => {
            if now_unix_ms.saturating_sub(record.started_at_unix_ms) > PIPELINE_RUN_STALE_RUNNING_MS
            {
                let mut check = warn_in(
                    DoctorCategory::Storage,
                    "adaptive_pipeline_last_run",
                    format!("{summary}; still `running` after more than six hours, the process was probably killed"),
                );
                check.repair_hint = Some(
                    "Confirm no `rein gc` process is alive, then re-run `rein gc --threshold 0`; the single-flight lock releases with the process."
                        .to_string(),
                );
                check
            } else {
                ok_in(
                    DoctorCategory::Storage,
                    "adaptive_pipeline_last_run",
                    summary,
                )
            }
        }
    }
}

fn check_a12_input_epoch(store: &crate::store::SqliteStore) -> DoctorCheck {
    use rusqlite::OptionalExtension;

    let raw = store
        .conn()
        .query_row(
            "SELECT value FROM metadata WHERE key = ?1",
            rusqlite::params![crate::store::a12_calibration::A12_INPUT_EPOCH_METADATA_KEY],
            |row| row.get::<_, String>(0),
        )
        .optional();
    match raw {
        Err(error) => {
            let mut check = warn_in(
                DoctorCategory::Storage,
                "a12_input_epoch",
                format!("storage error reading a12_input_epoch: {error}"),
            );
            check.repair_hint = Some(
                "Transient storage errors clear on retry; re-run doctor once the database lock is released.".to_string(),
            );
            check
        }
        Ok(None) => {
            let mut check = warn_in(
                DoctorCategory::Storage,
                "a12_input_epoch",
                "a12_input_epoch row is missing; every memory/graph write is rejected until it is re-created".to_string(),
            );
            check.repair_hint = Some(
                "Restart or reopen the database (schema bring-up re-creates the row at zero) or run `rein doctor --fix`. Sealed A12 runs then mismatch the reset counter and recalibrate fail-closed.".to_string(),
            );
            check
        }
        Ok(Some(raw)) => {
            let healthy = raw
                .parse::<u64>()
                .is_ok_and(|epoch| epoch <= i64::MAX as u64);
            if healthy {
                ok_in(
                    DoctorCategory::Storage,
                    "a12_input_epoch",
                    format!("a12_input_epoch={raw}"),
                )
            } else {
                let mut check = warn_in(
                    DoctorCategory::Storage,
                    "a12_input_epoch",
                    format!(
                        "a12_input_epoch is malformed ({raw:?}); every memory/graph write is rejected until it is repaired"
                    ),
                );
                check.repair_hint = Some(
                    "Run `rein doctor --fix`: it re-baselines the counter to zero AND invalidates the active A12 calibration in the same transaction, so the next calibration run reseals fresh instead of serving against an untrusted counter.".to_string(),
                );
                check
            }
        }
    }
}

/// Surface human agreement, runtime-vs-nightly drift, and deterministic
/// structural health as three distinct evidence classes. The shared Trust
/// projection owns the semantics; doctor only formats it and remains read-only.
fn check_judge_calibration(store: &crate::store::SqliteStore, config: &ReinConfig) -> DoctorCheck {
    if !config.ars.llm_judge.enabled {
        return ok_in(
            DoctorCategory::Configuration,
            "judge_calibration",
            "[ars.llm_judge].enabled = false (disabled by config)".to_string(),
        );
    }
    let report = crate::ops::trust_measurement::collect_judge_calibration_observability(
        store,
        config,
        chrono::Utc::now().timestamp(),
    );
    let synthesis = format_judge_calibration_surface("synthesis", &report.synthesis);
    let concept = format_judge_calibration_surface("concept_summary", &report.concept_summary);
    let summary = format!(
        "{synthesis} {concept} human_runtime_watermark={} structural_watermark={} \
         last_runtime_computed_at={} total_runtime_nightly_events={} \
         global_drift_alerts={}",
        report.human_runtime_watermark,
        report.structural_watermark,
        report.last_runtime_computed_at,
        report.total_runtime_nightly_events,
        report.global_drift_alert_count,
    );
    let surface_drift_alerts = report
        .synthesis
        .runtime_vs_nightly
        .drift_alert_count
        .saturating_add(report.concept_summary.runtime_vs_nightly.drift_alert_count);
    let total_drift_alerts = report
        .global_drift_alert_count
        .saturating_add(surface_drift_alerts);
    let mut repair_advice = std::collections::BTreeSet::new();
    for advice in report
        .synthesis
        .structural
        .repair_advice
        .iter()
        .chain(report.concept_summary.structural.repair_advice.iter())
    {
        repair_advice.insert(advice.clone());
    }
    let attention = total_drift_alerts > 0
        || report.synthesis.structural.baseline_blocks_release
        || report.concept_summary.structural.baseline_blocks_release
        || report
            .synthesis
            .structural
            .recall_fusion_promotion
            .release_blocked
        || report
            .concept_summary
            .structural
            .surface_scope_promotion
            .release_blocked
        || !repair_advice.is_empty();
    if attention {
        let mut check = warn_in(DoctorCategory::Configuration, "judge_calibration", summary);
        if !repair_advice.is_empty() {
            check.repair_hint = Some(repair_advice.into_iter().collect::<Vec<_>>().join(" "));
        }
        check
    } else {
        ok_in(DoctorCategory::Configuration, "judge_calibration", summary)
    }
}

fn format_judge_calibration_surface(
    name: &str,
    report: &crate::ops::trust_measurement::JudgeSurfaceCalibrationReport,
) -> String {
    let optional_kappa = |value: Option<f64>| {
        value
            .map(|kappa| format!("{kappa:.2}"))
            .unwrap_or_else(|| "undefined".to_string())
    };
    let optional_text = |value: Option<&str>| value.unwrap_or("undefined").to_string();
    let optional_timestamp = |value: Option<i64>| {
        value
            .map(|timestamp| timestamp.to_string())
            .unwrap_or_else(|| "undefined".to_string())
    };
    format!(
        "{name}[mode={} load_status={} human_pairs={} human_kappa={} \
         runtime_nightly_pairs={} runtime_nightly_kappa={} drift_alerts={} \
         structural_status={} requested_action={} basis={} reason={} \
         baseline_allowed={} baseline_scale={:.2} baseline_blocks_release={} \
         recall_fusion_action={} recall_fusion_allowed={} \
         recall_fusion_blocks_release={} recall_fusion_release_blocked={} \
         surface_scope_action={} surface_scope_allowed={} \
         surface_scope_blocks_release={} surface_scope_release_blocked={} \
         expected_model_fingerprint={} observed_model_fingerprint={} \
         expected_rubric_fingerprint={} observed_rubric_fingerprint={} \
         expected_probe_set={} observed_probe_set={} last_probe_at={} \
         completed_at={} fresh_until={} fresh={}]",
        report.structural.mode,
        report.structural.load_status,
        report.human.pair_count,
        optional_kappa(report.human.kappa),
        report.runtime_vs_nightly.pair_count,
        optional_kappa(report.runtime_vs_nightly.kappa),
        report.runtime_vs_nightly.drift_alert_count,
        report.structural.status,
        report.structural.requested_action,
        report.structural.basis,
        report.structural.reason,
        report.structural.baseline_allowed,
        report.structural.configured_baseline_scale,
        report.structural.baseline_blocks_release,
        report.structural.recall_fusion_promotion.requested_action,
        report.structural.recall_fusion_promotion.promotion_allowed,
        report
            .structural
            .recall_fusion_promotion
            .promotion_blocks_release,
        report.structural.recall_fusion_promotion.release_blocked,
        report.structural.surface_scope_promotion.requested_action,
        report.structural.surface_scope_promotion.promotion_allowed,
        report
            .structural
            .surface_scope_promotion
            .promotion_blocks_release,
        report.structural.surface_scope_promotion.release_blocked,
        optional_text(report.structural.expected_model_fingerprint.as_deref()),
        optional_text(report.structural.observed_model_fingerprint.as_deref()),
        optional_text(report.structural.expected_rubric_fingerprint.as_deref()),
        optional_text(report.structural.observed_rubric_fingerprint.as_deref()),
        report.structural.expected_probe_set_version,
        optional_text(report.structural.observed_probe_set_version.as_deref()),
        optional_timestamp(report.structural.last_probe_at),
        optional_timestamp(report.structural.completed_at),
        optional_timestamp(report.structural.fresh_until),
        report.structural.is_fresh,
    )
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

[async_memory]
provider = "inherit"

[cleanup]
"#,
            tempdir.path().join("doctor.db").display(),
            tempdir.path().display()
        );
        ReinConfig::load_from_str(&toml).unwrap()
    }

    #[test]
    fn dedup_threshold_observability_reports_legacy_shadow_as_fail_closed() {
        let store = SqliteStore::in_memory().unwrap();
        let mut config = ReinConfig::default();
        config.search.dedup_similarity = 0.70;
        let state = crate::store::adaptive::AdaptiveState {
            global_dedup_threshold: 0.40,
            version: 1,
            ..Default::default()
        };
        state.save_snapshot(store.conn()).unwrap();

        let check = check_dedup_threshold_observability(&store, &config);
        assert_eq!(check.status, CheckStatus::Ok);
        assert!(check.message.contains("static=0.700"), "{}", check.message);
        assert!(check.message.contains("shadow=0.400"), "{}", check.message);
        assert!(
            check.message.contains("hard_effective=0.700"),
            "{}",
            check.message
        );
        assert!(
            check.message.contains("source=legacy_unlabeled_shadow"),
            "{}",
            check.message
        );
        assert!(!check.fixable);
        let advice = check.repair_hint.as_deref().unwrap_or_default();
        assert!(advice.contains("recalibrat"), "advice={advice}");
        assert!(!advice.contains("doctor --fix"), "advice={advice}");
    }

    #[test]
    fn dedup_threshold_observability_fails_if_destructive_resolver_drops_below_static() {
        let store = SqliteStore::in_memory().unwrap();
        let mut config = ReinConfig::default();
        config.search.dedup_similarity = 0.70;
        let mut observability = crate::ops::dedup_threshold_observability(&store, &config);
        observability.hard_effective_threshold = 0.40;

        let check = project_dedup_threshold_doctor_check(&observability);
        assert_eq!(check.status, CheckStatus::Fail);
        assert_eq!(check.severity, DoctorSeverity::Error);
        assert!(!check.fixable);
        let advice = check.repair_hint.as_deref().unwrap_or_default();
        assert!(advice.contains("recalibrat"), "advice={advice}");
        assert!(advice.contains("atomically"), "advice={advice}");
        assert!(!advice.contains("doctor --fix"), "advice={advice}");
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

    // v0.35 Phase 3: the two tests
    // `test_doctor_warns_when_loopback_unauth_cannot_apply_to_non_loopback_bind`
    // and `test_doctor_warns_when_http_token_overrides_loopback_unauth` were
    // removed alongside the `[server].allow_unauthenticated_loopback` field
    // they exercised. The remaining failure paths (no token + no explicit
    // policy) are covered by the new fresh-install assertion below and the
    // migration tests in `config.rs::migrate_legacy_server_auth`.

    #[test]
    #[serial_test::serial(global_state)]
    fn test_doctor_fails_when_no_token_and_no_explicit_policy() {
        let _guard = EnvRestore {
            key: "REIN_HTTP_TOKEN",
            value: std::env::var("REIN_HTTP_TOKEN").ok(),
        };
        std::env::remove_var("REIN_HTTP_TOKEN");
        let mut config = ReinConfig::default();
        config.server.sse_enabled = true;
        config.server.sse_bind = "127.0.0.1".to_string();
        // Default ServerConfig has auth=None and no legacy bool anymore.

        let check = check_http_auth(&config);

        assert_eq!(check.name, "http_auth");
        assert_eq!(check.status, CheckStatus::Fail);
        assert!(check.message.contains("without REIN_HTTP_TOKEN"));
        assert!(check.message.contains("no explicit [server].auth"));
        let hint = check.repair_hint.as_deref().unwrap_or("");
        // The hint enumerates all four policies inline so operators see the
        // migration matrix without leaving the doctor output.
        assert!(hint.contains("\"public\""));
        assert!(hint.contains("\"loopback_only\""));
        assert!(hint.contains("\"bearer_required\""));
        assert!(hint.contains("\"oauth\""));
        assert!(hint.contains("REIN_HTTP_TOKEN"));
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

    // v0.35 Phase 3: `test_doctor_reports_legacy_loopback_flag_as_deprecated`
    // was removed alongside the bool itself. The deprecation surface is now
    // a load-time WARN via `tracing::warn!` in
    // `config::migrate_legacy_server_auth`, not a doctor check, because the
    // bool is stripped from the merged TOML before deserialize. The
    // migration is exercised by the `legacy_server_auth_migrates_*` tests
    // in `crates/rein/src/config.rs`.

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

    /// codex v1.2 R2 P2: explicit `[proxy].auth = "bearer_required"` without
    /// a token must FAIL even on a loopback bind with the service offline —
    /// the fresh-install OK shortcut is for the implicit default-deny posture
    /// only (`rein proxy on` would refuse immediately here, so doctor must
    /// not report OK).
    #[tokio::test]
    #[serial_test::serial(global_state)]
    async fn proxy_auth_explicit_bearer_without_token_fails() {
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
        let mut config = temp_config(&tempdir);
        config.proxy.bind = "127.0.0.1".to_string();
        config.proxy.auth = Some(crate::config::ProxyAuthPolicyConfig::BearerRequired);

        let check = check_proxy_auth(&config);
        assert_eq!(
            check.status,
            CheckStatus::Fail,
            "explicit bearer_required without a token must FAIL, got: {}",
            check.message
        );
    }

    /// codex v1.2 R3 P2: explicit `[proxy].auth = "public"` on loopback must
    /// report the unauthenticated posture even when a token is still
    /// exported — the runtime does not enforce that token under explicit
    /// public, so "token configured" would be a false security posture.
    #[tokio::test]
    #[serial_test::serial(global_state)]
    async fn proxy_auth_explicit_public_reports_unauthenticated_over_token() {
        let _guard = env_lock().lock().await;
        let _restore_http = EnvRestore {
            key: "REIN_HTTP_TOKEN",
            value: std::env::var("REIN_HTTP_TOKEN").ok(),
        };
        let _restore_proxy = EnvRestore {
            key: "REIN_PROXY_TOKEN",
            value: std::env::var("REIN_PROXY_TOKEN").ok(),
        };
        std::env::set_var("REIN_PROXY_TOKEN", "exported-but-not-enforced");
        std::env::remove_var("REIN_HTTP_TOKEN");

        let tempdir = tempfile::tempdir().unwrap();
        let mut config = temp_config(&tempdir);
        config.proxy.bind = "127.0.0.1".to_string();
        config.proxy.auth = Some(crate::config::ProxyAuthPolicyConfig::Public);

        let check = check_proxy_auth(&config);
        assert_eq!(check.status, CheckStatus::Ok);
        assert!(
            check.message.contains("unauthenticated"),
            "explicit public must surface the unauthenticated posture, got: {}",
            check.message
        );
        assert!(
            check.message.contains("NOT enforced"),
            "exported-token caveat must be visible, got: {}",
            check.message
        );
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

    #[tokio::test]
    async fn doctor_fix_preserves_future_schema_dedup_policy_and_seal_bytes() {
        use crate::store::dedup_calibration::{
            DEDUP_CALIBRATION_METADATA_KEY, DEDUP_CALIBRATION_SEAL_METADATA_KEY,
        };

        let tempdir = tempfile::tempdir().unwrap();
        let config = temp_config(&tempdir);
        let store = config.open_store().unwrap();
        let policy_raw = r#"{"schema_version":9999,"revision":41,"future_policy":"keep"}"#;
        let seal_raw = r#"{"schema_version":9999,"revision":41,"future_seal":"keep"}"#;
        store
            .conn()
            .execute(
                "INSERT INTO metadata(key, value) VALUES (?1, ?2)",
                rusqlite::params![DEDUP_CALIBRATION_METADATA_KEY, policy_raw],
            )
            .unwrap();
        store
            .conn()
            .execute(
                "INSERT INTO metadata(key, value) VALUES (?1, ?2)",
                rusqlite::params![DEDUP_CALIBRATION_SEAL_METADATA_KEY, seal_raw],
            )
            .unwrap();
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
                .any(|line| line.contains("dedup_calibration")),
            "doctor must not mutate dedup calibration state: {:?}",
            report.fixes_applied
        );
        let check = report
            .checks
            .iter()
            .find(|check| check.name == "dedup_threshold_observability")
            .expect("dedup observability check");
        assert_eq!(check.status, CheckStatus::Warn);
        assert!(!check.fixable);
        assert!(
            check.message.contains("policy_schema=9999"),
            "message={}",
            check.message
        );
        let hint = check.repair_hint.as_deref().unwrap_or_default();
        assert!(hint.contains("upgrade"), "hint={hint}");
        assert!(hint.contains("preserve"), "hint={hint}");
        assert!(!hint.contains("reset"), "hint={hint}");
        assert!(!hint.contains("doctor --fix"), "hint={hint}");

        let store = config.open_store().unwrap();
        for (key, expected) in [
            (DEDUP_CALIBRATION_METADATA_KEY, policy_raw),
            (DEDUP_CALIBRATION_SEAL_METADATA_KEY, seal_raw),
        ] {
            let actual: String = store
                .conn()
                .query_row(
                    "SELECT value FROM metadata WHERE key = ?1",
                    rusqlite::params![key],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(actual, expected, "future-schema row {key} changed");
        }
    }

    #[test]
    fn recall_fusion_doctor_is_read_only_and_matches_adaptive_projection() {
        let store = crate::store::SqliteStore::in_memory().unwrap();
        let config = ReinConfig::default();
        let before: i64 = store
            .conn()
            .query_row("SELECT COUNT(*) FROM metadata", [], |row| row.get(0))
            .unwrap();
        let shared = crate::ops::a12_activation::collect_recall_fusion_activation_report(
            &store,
            &config,
            chrono::Utc::now().timestamp_millis(),
        );

        let check = check_recall_fusion_calibration(&store, &config);
        let adaptive = crate::ops::adaptive_status_with_config(&store, &config);
        let after: i64 = store
            .conn()
            .query_row("SELECT COUNT(*) FROM metadata", [], |row| row.get(0))
            .unwrap();

        assert_eq!(check.name, "recall_fusion_calibration");
        // Benign absence (fresh store, no policy row) is Ok like the sibling
        // ars_parameter_policy / dedup observability checks — the summary
        // still carries the full projection for operators.
        assert_eq!(check.status, CheckStatus::Ok);
        assert_eq!(shared.health_status, "policy_missing");
        assert!(!check.fixable);
        assert!(check
            .message
            .contains(&format!("activation_status={}", shared.activation_status)));
        assert!(check
            .message
            .contains(&format!("health_status={}", shared.health_status)));
        assert_eq!(
            adaptive["recall_fusion_calibration"],
            serde_json::to_value(shared).unwrap()
        );
        assert_eq!(before, after, "doctor check must not mutate metadata");
        assert!(!check.message.contains("/Users/"));
    }

    /// P1-1: disabled or shadow-only configuration is a benign state, not a
    /// permanently degraded doctor report.
    #[test]
    fn recall_fusion_doctor_reports_ok_when_activation_is_disabled_by_config() {
        let store = crate::store::SqliteStore::in_memory().unwrap();
        let mut config = ReinConfig::default();
        config.ars.acceleration.shadow_only = true;

        let check = check_recall_fusion_calibration(&store, &config);

        assert_eq!(check.status, CheckStatus::Ok);
        assert!(check.message.contains("activation_status=disabled"));
        assert!(check.repair_hint.is_none());
    }

    #[test]
    fn recall_fusion_doctor_preserves_future_schema_evidence() {
        let store = crate::store::SqliteStore::in_memory().unwrap();
        let config = ReinConfig::default();
        let raw = r#"{"schema_version":9999,"future_pointer":"keep"}"#;
        store
            .conn()
            .execute(
                "INSERT INTO metadata(key, value) VALUES (?1, ?2)",
                rusqlite::params![
                    crate::store::a12_calibration::A12_CALIBRATION_METADATA_KEY,
                    raw
                ],
            )
            .unwrap();

        let check = check_recall_fusion_calibration(&store, &config);
        let stored: String = store
            .conn()
            .query_row(
                "SELECT value FROM metadata WHERE key = ?1",
                rusqlite::params![crate::store::a12_calibration::A12_CALIBRATION_METADATA_KEY],
                |row| row.get(0),
            )
            .unwrap();

        assert_eq!(check.status, CheckStatus::Warn);
        assert!(!check.fixable);
        assert!(check.message.contains("unsupported_schema"));
        assert_eq!(stored, raw);
        assert!(!check
            .repair_hint
            .as_deref()
            .unwrap_or_default()
            .contains("--fix"));
    }

    #[test]
    fn adaptive_pipeline_last_run_never_run_warns() {
        let store = crate::store::SqliteStore::in_memory().unwrap();
        let check = check_adaptive_pipeline_last_run(&store);
        assert_eq!(check.status, CheckStatus::Warn);
        assert!(check.message.contains("never completed"));
        assert!(check
            .repair_hint
            .as_deref()
            .unwrap_or_default()
            .contains("rein gc --threshold 0"));
    }

    #[test]
    fn adaptive_pipeline_last_run_completed_is_ok() {
        use crate::ops::pipeline_run::{PipelineRunOutcome, PipelineRunRecorder};
        let store = crate::store::SqliteStore::in_memory().unwrap();
        let recorder = PipelineRunRecorder::start(&store, "gc");
        recorder.stage("m4_cluster", || ());
        recorder.finish(PipelineRunOutcome::Completed, None);
        let check = check_adaptive_pipeline_last_run(&store);
        assert_eq!(check.status, CheckStatus::Ok);
        assert!(check.message.contains("completed"));
        assert!(check.message.contains("trigger=gc"));
        assert!(check.message.contains("m4_cluster"));
    }

    #[test]
    fn adaptive_pipeline_last_run_stale_running_warns() {
        use crate::ops::pipeline_run::{PipelineRunRecorder, PIPELINE_RUN_STALE_RUNNING_MS};
        let store = crate::store::SqliteStore::in_memory().unwrap();
        let recorder = PipelineRunRecorder::start(&store, "gc");
        recorder.stage("m4_cluster", || ());
        let started = recorder.record().started_at_unix_ms;
        let fresh = check_adaptive_pipeline_last_run_at(&store, started + 1_000);
        assert_eq!(
            fresh.status,
            CheckStatus::Ok,
            "a young running pass is fine"
        );
        let stale = check_adaptive_pipeline_last_run_at(
            &store,
            started + PIPELINE_RUN_STALE_RUNNING_MS + 1,
        );
        assert_eq!(stale.status, CheckStatus::Warn);
        assert!(stale.message.contains("probably killed"));
    }

    #[test]
    fn adaptive_pipeline_last_run_failed_warns_with_error() {
        use crate::ops::pipeline_run::{PipelineRunOutcome, PipelineRunRecorder};
        let store = crate::store::SqliteStore::in_memory().unwrap();
        let recorder = PipelineRunRecorder::start(&store, "dedup");
        recorder.finish(
            PipelineRunOutcome::Failed,
            Some("snapshot save failed".into()),
        );
        let check = check_adaptive_pipeline_last_run(&store);
        assert_eq!(check.status, CheckStatus::Warn);
        assert!(check.message.contains("error=snapshot save failed"));
    }

    #[test]
    fn claude_hooks_missing_file_is_ok_and_missing_entries_warn() {
        let dir = tempfile::tempdir().unwrap();
        let absent = check_claude_hooks_at(&dir.path().join("settings.json"));
        assert_eq!(absent.status, CheckStatus::Ok);
        assert!(absent.message.contains("not found"));

        let path = dir.path().join("settings.json");
        std::fs::write(&path, r#"{"hooks":{"PreToolUse":[{"matcher":"Bash","hooks":[{"type":"command","command":"rtk hook claude"}]}]}}"#).unwrap();
        let none = check_claude_hooks_at(&path);
        assert_eq!(none.status, CheckStatus::Warn);
        assert!(none.message.contains("no rein hooks"));
        assert!(none
            .repair_hint
            .as_deref()
            .unwrap_or_default()
            .contains("README"));

        std::fs::write(&path, "{not json").unwrap();
        let bad = check_claude_hooks_at(&path);
        assert_eq!(bad.status, CheckStatus::Warn);
        assert!(bad.message.contains("not valid JSON"));
    }

    #[test]
    fn claude_hooks_present_reports_installed_and_missing_events() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        std::fs::write(
            &path,
            r#"{"hooks":{
                "PostToolUse":[{"matcher":"","hooks":[{"type":"command","command":"rein hook post","timeout":10}]}],
                "Stop":[{"matcher":"","hooks":[{"type":"command","command":"rein hook stop","timeout":30}]}]
            }}"#,
        )
        .unwrap();
        let check = check_claude_hooks_at(&path);
        assert_eq!(check.status, CheckStatus::Ok);
        assert!(
            check.message.contains("installed: PostToolUse, Stop"),
            "{}",
            check.message
        );
        assert!(
            check
                .message
                .contains("not installed: PreCompact, UserPromptSubmit"),
            "{}",
            check.message
        );
    }

    #[test]
    fn a12_input_epoch_doctor_reports_healthy_counter() {
        let store = crate::store::SqliteStore::in_memory().unwrap();

        let check = check_a12_input_epoch(&store);

        assert_eq!(check.status, CheckStatus::Ok);
        assert!(check.message.contains("a12_input_epoch="));
    }

    #[test]
    fn a12_input_epoch_doctor_warns_when_row_is_missing() {
        let store = crate::store::SqliteStore::in_memory().unwrap();
        store
            .conn()
            .execute(
                "DELETE FROM metadata WHERE key = ?1",
                rusqlite::params![crate::store::a12_calibration::A12_INPUT_EPOCH_METADATA_KEY],
            )
            .unwrap();

        let check = check_a12_input_epoch(&store);

        assert_eq!(check.status, CheckStatus::Warn);
        assert!(check.message.contains("missing"));
        assert!(check
            .repair_hint
            .as_deref()
            .unwrap_or_default()
            .contains("--fix"));
    }

    #[test]
    fn a12_input_epoch_doctor_warns_on_malformed_counter() {
        let store = crate::store::SqliteStore::in_memory().unwrap();
        store
            .conn()
            .execute(
                "UPDATE metadata SET value = 'garbage' WHERE key = ?1",
                rusqlite::params![crate::store::a12_calibration::A12_INPUT_EPOCH_METADATA_KEY],
            )
            .unwrap();

        let check = check_a12_input_epoch(&store);

        assert_eq!(check.status, CheckStatus::Warn);
        assert!(check.message.contains("malformed"));
        assert!(check
            .repair_hint
            .as_deref()
            .unwrap_or_default()
            .contains("--fix"));
    }

    /// P2-1: a fail-closed tampered scope must raise doctor attention even
    /// when another scope is active and its reason prose carries none of the
    /// old stale/expired/mismatch keywords — attention keys on the typed
    /// health code, not on reason strings.
    #[test]
    fn recall_fusion_doctor_flags_tampered_scope_despite_another_active_scope() {
        use crate::store::a12_calibration::{
            A12CalibrationPhase, A12CalibrationRunMetadata, A12CalibrationScope,
            A12CalibrationState, A12CalibrationVerdict, A12FusionSimplex, A12PairedTop3Stats,
            A12ProvenanceCounts, A12ScopeEntry, A12_CALIBRATION_SCHEMA_VERSION,
        };
        use crate::store::adaptive::{AdaptiveState, LearnedShadowFusionEntry};

        let store = crate::store::SqliteStore::in_memory().unwrap();
        let mut config = ReinConfig::default();
        config.adaptive.min_samples_alpha = 10;

        let human_entry = |sample_count: usize| LearnedShadowFusionEntry {
            weights: crate::store::adaptive::ShadowFusionWeightEntry {
                bm25: 0.20,
                vec: 0.25,
                kg: 0.20,
                episode: 0.15,
                support: 0.10,
                diversity: 0.10,
            },
            sample_count,
            last_updated: "2026-07-14T00:00:00Z".to_string(),
        };
        let mut adaptive = AdaptiveState {
            version: 9,
            ..AdaptiveState::default()
        };
        adaptive
            .learned_shadow_fusion
            .insert("global".to_string(), human_entry(12));
        adaptive
            .learned_shadow_fusion
            .insert("semantic".to_string(), human_entry(12));

        let mcnemar = crate::eval::mcnemar::mcnemar_from_counts(12, 0, 0, 0).unwrap();
        let entry = A12ScopeEntry {
            scope: A12CalibrationScope::Global,
            canonical_generation: 2,
            generation_fingerprint: "generation-fingerprint".to_string(),
            source_snapshot_fingerprint: "source-snapshot-fingerprint".to_string(),
            snapshot_cutoff: 1_700_000_000,
            corpus_fingerprint: "corpus-fingerprint".to_string(),
            // Below the eligibility floor so the seal records a Human basis
            // with an automatic candidate boundary.
            train_family_ess: 5,
            train_case_count: 5,
            holdout_family_ess: 12,
            simplex: A12FusionSimplex {
                bm25: 0.10,
                vector: 0.20,
                kg: 0.30,
                episode: 0.15,
                support: 0.15,
                diversity: 0.10,
            },
            verdict: A12CalibrationVerdict::Ship,
            noise_floor: 0.02,
            paired_top3: A12PairedTop3Stats {
                n: u64::from(mcnemar.n),
                both_hit: u64::from(mcnemar.a),
                baseline_only: u64::from(mcnemar.b),
                treatment_only: u64::from(mcnemar.c),
                neither_hit: u64::from(mcnemar.d),
                chi_squared: mcnemar.chi_squared,
                p_value: mcnemar.p_value,
                diff_point: mcnemar.diff_point,
                ci_lower: mcnemar.ci_lower,
                ci_upper: mcnemar.ci_upper,
                used_exact: mcnemar.used_exact,
            },
            provenance: A12ProvenanceCounts {
                canonical_loo: 5,
                concept_loo: 0,
                episode_loo: 0,
            },
            provenance_holdout: None,
            training_fingerprint: "training-fingerprint".to_string(),
            holdout_fingerprint: "holdout-fingerprint".to_string(),
            optimizer_fingerprint: "optimizer-fingerprint".to_string(),
            evaluation_fingerprint: "evaluation-fingerprint".to_string(),
            holdout_reason: "holdout evaluated".to_string(),
            calibrated_at: 1_700_000_000,
            evaluated_at: 1_700_000_050,
            valid_until_exclusive: None,
            cluster_generation: None,
            invalidation: None,
        };
        let a12_state = A12CalibrationState {
            schema_version: A12_CALIBRATION_SCHEMA_VERSION,
            revision: 3,
            generation: 2,
            generation_fingerprint: "generation-fingerprint".to_string(),
            snapshot_cutoff: 1_700_000_000,
            corpus_fingerprint: "corpus-fingerprint".to_string(),
            cluster_generation: 4,
            scopes: std::collections::BTreeMap::from([("global".to_string(), entry)]),
            created_at: 1_700_000_000,
            updated_at: 1_700_000_050,
            run: Some(A12CalibrationRunMetadata {
                phase: A12CalibrationPhase::Complete,
                source_input_epoch: 0,
                source_snapshot_fingerprint: "source-snapshot-fingerprint".to_string(),
                behavior_config_fingerprint: "behavior-config-fingerprint".to_string(),
            }),
        };
        assert!(
            crate::store::a12_calibration::compare_and_swap_a12_calibration(
                store.conn(),
                &a12_state,
                0
            )
            .unwrap()
        );
        let a12 = crate::store::a12_calibration::load_a12_calibration(store.conn());
        assert_eq!(
            a12.status,
            crate::store::a12_calibration::A12CalibrationLoadStatus::Loaded
        );

        // Seal policy evidence against the untampered live human state.
        let gate = crate::ops::a12_activation::RecallEvalGateAttestation {
            status: crate::store::ars_parameter_policy::ArsRecallGateStatus::Ship,
            reason_code: crate::ops::a12_activation::RecallEvalGateReasonCode::Compared,
            build_fingerprint: Some(env!("REIN_BUILD_FINGERPRINT").to_string()),
            fixture_fingerprint: Some("recall-fixtures".to_string()),
            evaluated_at: Some(1_700_000_100),
            reason: "paired recall gate shipped".to_string(),
        };
        let mut evidence = crate::ops::a12_activation::resolve_recall_fusion_evidence(
            &adaptive,
            &a12,
            10,
            0.02,
            1_700_000_060_000,
            &gate,
        );
        let sealed_global = evidence.get_mut("recall_fusion:global").unwrap();
        assert_eq!(
            sealed_global.basis,
            crate::store::ars_parameter_policy::ArsRecallFusionEvidenceBasis::Human
        );
        assert!(sealed_global.automatic_candidate_present);
        sealed_global.human_runtime_adoption_weight = Some(0.25);
        let policy = crate::store::ars_parameter_policy::ArsParameterPolicy {
            schema_version: crate::store::ars_parameter_policy::ARS_PARAMETER_POLICY_SCHEMA_VERSION,
            revision: 1,
            mode: crate::store::ars_parameter_policy::ArsParameterPolicyMode::Canary,
            disabled_reason: None,
            source_adaptive_version: 9,
            runtime_adoption_weight: 0.0,
            adoption_weights: std::collections::HashMap::from([
                ("recall_fusion:global".to_string(), 0.40),
                ("recall_fusion:semantic".to_string(), 0.40),
            ]),
            recall_fusion_evidence: evidence.into_iter().collect(),
            last_event_id: 0,
            last_updated: "2026-07-14T00:00:00Z".to_string(),
        };
        assert!(
            crate::store::ars_parameter_policy::save_parameter_policy_cas(store.conn(), &policy, 0)
                .unwrap()
        );

        // Tamper: the live human entry behind the sealed `global` fallback
        // changes after sealing. Its fail-closed reason ("live human fallback
        // does not match sealed simplex or ESS") carries no legacy keyword.
        adaptive
            .learned_shadow_fusion
            .get_mut("global")
            .unwrap()
            .sample_count = 13;
        adaptive.save_snapshot(store.conn()).unwrap();

        let report = crate::ops::a12_activation::collect_recall_fusion_activation_report(
            &store,
            &config,
            chrono::Utc::now().timestamp_millis(),
        );
        let check = check_recall_fusion_calibration(&store, &config);

        assert!(report.active, "the untampered human scope stays active");
        let tampered = report
            .scopes
            .iter()
            .find(|scope| scope.scope == "global")
            .unwrap();
        assert!(!tampered.active);
        assert_eq!(
            tampered.health_code,
            crate::ops::a12_activation::RecallFusionScopeHealthCode::Tampered
        );
        assert_eq!(report.health_status, "degraded");
        assert_eq!(check.status, CheckStatus::Warn);
        assert!(check.repair_hint.is_some());
    }

    #[test]
    fn judge_calibration_doctor_reports_modes_and_missing_state_side_by_side() {
        use crate::config::JudgeStructuralAnchorMode;

        let configured = |mode| {
            let mut config = ReinConfig::default();
            config.ars.llm_judge.enabled = true;
            config.ars.llm_judge.synthesis_enabled = true;
            config.ars.llm_judge.concept_summary_enabled = true;
            config.ars.llm_judge.structural_anchors.mode = mode;
            config.ars.llm_judge.structural_anchors.interval_secs = 100;
            config.llm.provider = "omlx".to_string();
            config.llm.omlx.model = Some("doctor-judge-model".to_string());
            config
        };

        for (mode, expected_status, expected_scale, expected_check_status) in [
            (
                JudgeStructuralAnchorMode::Off,
                "disabled",
                "baseline_scale=1.00",
                CheckStatus::Warn,
            ),
            (
                JudgeStructuralAnchorMode::Monitor,
                "unknown",
                "baseline_scale=1.00",
                CheckStatus::Warn,
            ),
            (
                JudgeStructuralAnchorMode::Enforce,
                "unknown",
                "baseline_scale=0.00",
                CheckStatus::Warn,
            ),
        ] {
            let store = crate::store::SqliteStore::in_memory().unwrap();
            let config = configured(mode);
            let check = check_judge_calibration(&store, &config);
            assert_eq!(check.status, expected_check_status, "mode={mode:?}");
            assert!(!check.fixable);
            assert!(
                check.message.contains(&format!(
                    "mode={} load_status=missing",
                    match mode {
                        JudgeStructuralAnchorMode::Off => "off",
                        JudgeStructuralAnchorMode::Monitor => "monitor",
                        JudgeStructuralAnchorMode::Enforce => "enforce",
                    }
                )),
                "message={}",
                check.message
            );
            assert!(
                check
                    .message
                    .contains("human_pairs=0 human_kappa=undefined"),
                "message={}",
                check.message
            );
            assert!(
                check
                    .message
                    .contains("runtime_nightly_pairs=0 runtime_nightly_kappa=undefined"),
                "message={}",
                check.message
            );
            assert!(
                check
                    .message
                    .contains(&format!("structural_status={expected_status}")),
                "message={}",
                check.message
            );
            assert!(
                check.message.contains(expected_scale),
                "message={}",
                check.message
            );
            assert!(
                check
                    .message
                    .contains("requested_action=keep_configured_baseline"),
                "message={}",
                check.message
            );
            assert!(
                check.message.contains("recall_fusion_release_blocked=true"),
                "message={}",
                check.message
            );
            if mode == JudgeStructuralAnchorMode::Off {
                assert!(check.repair_hint.is_none());
            } else {
                let hint = check.repair_hint.as_deref().unwrap_or_default();
                assert!(hint.contains("read-only"), "hint={hint}");
                assert!(!hint.contains("doctor --fix"), "hint={hint}");
            }
        }
    }

    #[test]
    fn judge_calibration_doctor_redacts_and_preserves_unhealthy_structural_rows() {
        use crate::config::JudgeStructuralAnchorMode;
        use crate::store::judge_structural_calibration::JUDGE_STRUCTURAL_CALIBRATION_METADATA_KEY;

        let mut config = ReinConfig::default();
        config.ars.llm_judge.enabled = true;
        config.ars.llm_judge.synthesis_enabled = true;
        config.ars.llm_judge.structural_anchors.mode = JudgeStructuralAnchorMode::Enforce;
        config.llm.provider = "omlx".to_string();
        config.llm.omlx.model = Some("doctor-judge-model".to_string());

        for (raw, expected_load, advice_word) in [
            (
                "{doctor-operator-secret".to_string(),
                "load_status=corrupt",
                "Preserve",
            ),
            (
                serde_json::json!({
                    "schema_version": 9999,
                    "revision": 4,
                    "doctor_operator_secret": "must-not-leak"
                })
                .to_string(),
                "load_status=unsupported_schema",
                "Upgrade",
            ),
        ] {
            let store = crate::store::SqliteStore::in_memory().unwrap();
            store
                .conn()
                .execute(
                    "INSERT INTO metadata(key, value) VALUES (?1, ?2)",
                    rusqlite::params![JUDGE_STRUCTURAL_CALIBRATION_METADATA_KEY, &raw],
                )
                .unwrap();
            let check = check_judge_calibration(&store, &config);
            assert_eq!(check.status, CheckStatus::Warn);
            assert!(!check.fixable);
            assert!(
                check.message.contains(expected_load),
                "message={}",
                check.message
            );
            assert!(!check.message.contains("operator-secret"));
            assert!(!check.message.contains("must-not-leak"));
            let hint = check.repair_hint.as_deref().unwrap_or_default();
            assert!(hint.contains(advice_word), "hint={hint}");
            assert!(!hint.contains("doctor --fix"), "hint={hint}");
            let preserved: String = store
                .conn()
                .query_row(
                    "SELECT value FROM metadata WHERE key = ?1",
                    [JUDGE_STRUCTURAL_CALIBRATION_METADATA_KEY],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(preserved, raw);
        }
    }

    #[test]
    fn judge_calibration_doctor_reports_stale_and_fingerprint_mismatch_with_freshness() {
        use crate::config::JudgeStructuralAnchorMode;
        use crate::store::adaptive::{JudgeStructuralProbeKind, JudgeSurface};
        use crate::store::judge_structural_calibration::{
            compare_and_swap_judge_structural_calibration, JudgeStructuralCalibrationState,
            JudgeStructuralSurfaceState,
        };

        let mut config = ReinConfig::default();
        config.ars.llm_judge.enabled = true;
        config.ars.llm_judge.synthesis_enabled = true;
        config.ars.llm_judge.structural_anchors.mode = JudgeStructuralAnchorMode::Enforce;
        config.ars.llm_judge.structural_anchors.interval_secs = 100;
        config.llm.provider = "omlx".to_string();
        config.llm.omlx.model = Some("doctor-model-a".to_string());
        let (model_fingerprint, rubric_fingerprint) =
            crate::ops::llm_judge_worker::structural_fingerprints_for_config(
                &config,
                JudgeSurface::Synthesis,
            )
            .unwrap();
        let completed_at = chrono::Utc::now().timestamp() - 201;
        let state = JudgeStructuralCalibrationState {
            revision: 1,
            synthesis: JudgeStructuralSurfaceState {
                run_id: Some("doctor-run-not-exposed".to_string()),
                probe_set_version: crate::ops::llm_judge_worker::JUDGE_STRUCTURAL_PROBE_SET_VERSION
                    .to_string(),
                model_fingerprint,
                rubric_fingerprint,
                run_token_hashes: JudgeStructuralProbeKind::ALL
                    .into_iter()
                    .map(|kind| (kind, "u".repeat(64)))
                    .collect(),
                seen_kinds: JudgeStructuralProbeKind::ALL.into_iter().collect(),
                run_started_at: completed_at - 10,
                last_probe_at: completed_at,
                completed_at: Some(completed_at),
                status: crate::judge::contract::JudgeStructuralStatus::Ready,
                ..JudgeStructuralSurfaceState::default()
            },
            updated_at: completed_at,
            ..JudgeStructuralCalibrationState::default()
        };
        let store = crate::store::SqliteStore::in_memory().unwrap();
        assert!(compare_and_swap_judge_structural_calibration(store.conn(), &state, 0).unwrap());

        let stale = check_judge_calibration(&store, &config);
        assert_eq!(stale.status, CheckStatus::Warn);
        assert!(stale.message.contains("structural_status=stale"));
        assert!(stale.message.contains("fresh_until="));
        assert!(stale
            .repair_hint
            .as_deref()
            .unwrap_or_default()
            .contains("refresh"));

        config.llm.omlx.model = Some("doctor-model-b".to_string());
        let mismatch = check_judge_calibration(&store, &config);
        assert_eq!(mismatch.status, CheckStatus::Warn);
        assert!(mismatch
            .message
            .contains("structural_status=fingerprint_mismatch"));
        assert!(mismatch.message.contains("expected_model_fingerprint="));
        assert!(mismatch.message.contains("observed_model_fingerprint="));
        assert!(mismatch
            .repair_hint
            .as_deref()
            .unwrap_or_default()
            .contains("current model"));
        assert!(!mismatch.message.contains("doctor-run-not-exposed"));
        assert!(!mismatch.message.contains(&"u".repeat(64)));
    }

    #[test]
    fn retired_gemini_preview_warning_covers_all_active_resolved_llm_sections() {
        let retired = "gemini-3.1-flash-lite-preview";
        let mut config = ReinConfig::default();
        config.llm.provider = "google".to_string();
        config.llm.google.model = Some(retired.to_string());
        config.extract.provider = "google".to_string();
        config.extract.google.model = retired.to_string();
        config.query_expansion.provider = "google".to_string();
        config.query_expansion.google.model = retired.to_string();
        config.async_memory.provider = "inherit".to_string();
        config.intelligent_merge.enabled = true;
        config.intelligent_merge.provider = "google".to_string();
        config.intelligent_merge.google.model = retired.to_string();
        config.search.llm_reranker = "google".to_string();
        config.ars.recall_synthesis_enabled = true;
        config.ars.concept_summary_enabled = true;
        config.ars.cold_archive_enabled = true;
        config.ars.llm_backend = "google".to_string();
        config.resummerize.enabled = true;
        config.resummerize.llm_backend = "google".to_string();
        config.ars.llm_judge.enabled = true;
        config.ars.llm_judge.synthesis_enabled = true;
        config.ars.llm_judge.nightly_cron.enabled = true;

        let original_extract = config.extract.google.model.clone();
        let original_expansion = config.query_expansion.google.model.clone();
        let check = check_retired_llm_models(&config);
        assert_eq!(check.status, CheckStatus::Warn);
        assert_eq!(check.severity, DoctorSeverity::Warning);
        assert!(!check.fixable);
        for section in [
            "extract",
            "extract.async_memory",
            "extract.intelligent_merge",
            "extract.dedup",
            "query_expansion",
            "search.llm_reranker",
            "ars.recall_synthesis",
            "ars.concept_summary",
            "ars.cold_archive",
            "resummerize",
            "ars.llm_judge",
            "ars.llm_judge.nightly_cron",
        ] {
            assert!(
                check.message.contains(section),
                "missing {section}: {}",
                check.message
            );
        }
        assert!(check.message.contains(retired));
        let hint = check.repair_hint.as_deref().unwrap_or_default();
        assert!(hint.contains("gemini-3.1-flash-lite"), "hint={hint}");
        assert!(hint.contains("does not rewrite"), "hint={hint}");
        assert!(!hint.contains("doctor --fix"), "hint={hint}");
        assert_eq!(config.extract.google.model, original_extract);
        assert_eq!(config.query_expansion.google.model, original_expansion);
    }

    #[test]
    fn retired_gemini_preview_warning_treats_nightly_as_independent_production_consumer() {
        let mut config = ReinConfig::default();
        config.llm.provider = "google".to_string();
        config.llm.google.model = Some(RETIRED_GEMINI_FLASH_LITE_PREVIEW_MODEL.to_string());
        config.extract.provider = "google".to_string();
        config.extract.google.model = STABLE_GEMINI_FLASH_LITE_MODEL.to_string();
        config.query_expansion.provider = "google".to_string();
        config.query_expansion.google.model = STABLE_GEMINI_FLASH_LITE_MODEL.to_string();
        config.async_memory.provider = "none".to_string();
        config.ars.llm_judge.enabled = true;
        config.ars.llm_judge.synthesis_enabled = false;
        config.ars.llm_judge.concept_summary_enabled = false;
        config.ars.llm_judge.nightly_cron.enabled = true;

        let check = check_retired_llm_models(&config);
        assert_eq!(check.status, CheckStatus::Warn);
        let listed = check
            .message
            .split("sections: ")
            .nth(1)
            .expect("lifecycle warning lists active sections");
        assert_eq!(listed, "ars.llm_judge.nightly_cron, extract.dedup");
    }

    #[test]
    fn judge_llm_provider_warning_reports_active_runtime_with_none_provider() {
        let mut config = ReinConfig::default();
        config.llm.provider = "none".to_string();
        config.ars.llm_judge.enabled = true;
        config.ars.llm_judge.synthesis_enabled = true;
        config.ars.llm_judge.concept_summary_enabled = false;
        config.ars.llm_judge.nightly_cron.enabled = false;
        assert_eq!(
            config.resolve_llm_for("ars.llm_judge").unwrap().provider,
            Provider::None
        );

        let check = check_judge_llm_provider(&config);
        assert_eq!(check.status, CheckStatus::Warn);
        assert_eq!(check.severity, DoctorSeverity::Warning);
        assert!(!check.fixable);
        assert!(check.message.contains("ars.llm_judge runtime"));
        assert!(!check.message.contains("nightly_cron"));
        let hint = check.repair_hint.as_deref().unwrap_or_default();
        assert!(hint.contains("[llm] provider"), "hint={hint}");
        assert!(!check.message.contains("endpoint"));
        assert!(!check.message.contains("api_key"));
    }

    #[test]
    fn judge_llm_provider_warning_reports_active_nightly_with_runtime_surfaces_off() {
        let mut config = ReinConfig::default();
        config.llm.provider = "none".to_string();
        config.ars.llm_judge.enabled = true;
        config.ars.llm_judge.synthesis_enabled = false;
        config.ars.llm_judge.concept_summary_enabled = false;
        config.ars.llm_judge.nightly_cron.enabled = true;
        assert_eq!(
            config
                .resolve_llm_for("ars.llm_judge.nightly_cron")
                .unwrap()
                .provider,
            Provider::None
        );

        let check = check_judge_llm_provider(&config);
        assert_eq!(check.status, CheckStatus::Warn);
        assert_eq!(check.severity, DoctorSeverity::Warning);
        assert!(!check.fixable);
        assert!(check.message.contains("ars.llm_judge.nightly_cron"));
        assert!(!check.message.contains("ars.llm_judge runtime"));
        let hint = check.repair_hint.as_deref().unwrap_or_default();
        assert!(hint.contains("[llm] provider"), "hint={hint}");
        assert!(!check.message.contains("endpoint"));
        assert!(!check.message.contains("api_key"));
    }

    #[test]
    fn retired_gemini_preview_warning_ignores_inactive_or_non_google_sections() {
        let retired = "gemini-3.1-flash-lite-preview";
        let stable = "gemini-3.1-flash-lite";
        let mut config = ReinConfig::default();
        config.llm.provider = "google".to_string();
        config.llm.google.model = Some(stable.to_string());
        config.extract.provider = "google".to_string();
        config.extract.google.model = stable.to_string();
        config.query_expansion.provider = "google".to_string();
        config.query_expansion.google.model = stable.to_string();
        config.intelligent_merge.enabled = false;
        config.intelligent_merge.provider = "google".to_string();
        config.intelligent_merge.google.model = retired.to_string();
        config.ars.recall_synthesis_enabled = false;
        config.ars.concept_summary_enabled = false;
        config.ars.cold_archive_enabled = false;
        config.resummerize.enabled = false;
        config.ars.llm_judge.enabled = false;

        let check = check_retired_llm_models(&config);
        assert_eq!(check.status, CheckStatus::Ok);
        assert!(check.repair_hint.is_none());

        config.extract.provider = "omlx".to_string();
        config.extract.omlx.model = retired.to_string();
        let non_google = check_retired_llm_models(&config);
        assert_eq!(non_google.status, CheckStatus::Ok);
    }
}
