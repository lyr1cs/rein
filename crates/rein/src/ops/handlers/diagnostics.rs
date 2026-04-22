//! Diagnostics-category op handlers (Phase 1: stats, health; Phase 2.1: doctor).

use clap::Args;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use rein_macros::op;

use crate::doctor::{self, DoctorOptions, DoctorReport};
use crate::ops::system_health::{
    self, GrayzoneSnapshot, IndexesSnapshot, QueuesSnapshot, SystemStatus,
};
use crate::ops::SurfaceKind;
use crate::ops::{IntoCliText, IntoJson, IntoMarkdown, OpsRuntime};
use crate::types::{HealthReport, MemoryStore, ReinResult, StoreStats};

#[derive(Serialize, Clone, Debug)]
pub struct StatsOutput {
    pub total_memories: usize,
    pub ltm_count: usize,
    pub stm_count: usize,
    pub topic_count: usize,
    pub avg_strength: f64,
    pub memoir_count: usize,
    pub concept_count: usize,
    pub link_count: usize,
    pub hot_count: usize,
    pub warm_count: usize,
    pub cold_count: usize,
}

impl From<StoreStats> for StatsOutput {
    fn from(s: StoreStats) -> Self {
        Self {
            total_memories: s.total_memories,
            ltm_count: s.ltm_count,
            stm_count: s.stm_count,
            topic_count: s.topic_count,
            avg_strength: s.avg_strength,
            memoir_count: s.memoir_count,
            concept_count: s.concept_count,
            link_count: s.link_count,
            hot_count: s.hot_count,
            warm_count: s.warm_count,
            cold_count: s.cold_count,
        }
    }
}

impl IntoJson for StatsOutput {
    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

impl IntoMarkdown for StatsOutput {
    fn to_markdown(&self) -> String {
        format!(
            "**Memory stats**\n\
             - total: {}\n\
             - LTM: {} / STM: {}\n\
             - topics: {}\n\
             - avg strength: {:.3}\n\
             - tiers: hot={} warm={} cold={}\n\
             - memoirs: {} concepts: {} links: {}",
            self.total_memories,
            self.ltm_count,
            self.stm_count,
            self.topic_count,
            self.avg_strength,
            self.hot_count,
            self.warm_count,
            self.cold_count,
            self.memoir_count,
            self.concept_count,
            self.link_count,
        )
    }
}

impl IntoCliText for StatsOutput {
    fn to_cli_text(&self) -> String {
        self.to_markdown()
    }
}

impl OpsRuntime {
    #[op(
        name = "stats",
        category = "memory",
        description = "Show store statistics — counts, layers, tiers",
        cli(name = "stats"),
        mcp(name = "rein_stats"),
        rest(method = "GET", path = "/api/stats")
    )]
    pub fn stats(&self) -> ReinResult<StatsOutput> {
        let stats = self.with_store(|s| s.stats())?;
        Ok(stats.into())
    }
}

#[derive(Args, Deserialize, JsonSchema, Debug, Clone, Default)]
pub struct HealthParams {
    /// Filter reports to a single topic (positional; pre-A1 CLI compat).
    #[serde(default)]
    pub topic: Option<String>,
}

#[derive(Serialize, Clone, Debug)]
pub struct HealthReportItem {
    pub topic: String,
    pub count: usize,
    pub avg_strength: f64,
    pub stale_count: usize,
    pub needs_consolidation: bool,
}

impl From<HealthReport> for HealthReportItem {
    fn from(r: HealthReport) -> Self {
        Self {
            topic: r.topic,
            count: r.count,
            avg_strength: r.avg_strength,
            stale_count: r.stale_count,
            needs_consolidation: r.needs_consolidation,
        }
    }
}

#[derive(Serialize, Clone, Debug)]
pub struct HealthOutput {
    // Serde-named "health" to preserve the pre-A1 `/api/health` response shape
    // that the Neural Wiki GUI reads from.
    #[serde(rename = "health")]
    pub reports: Vec<HealthReportItem>,
    pub indexes: IndexesSnapshot,
    pub queues: QueuesSnapshot,
    pub grayzone: GrayzoneSnapshot,
    pub status: SystemStatus,
}

impl IntoJson for HealthOutput {
    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

impl IntoMarkdown for HealthOutput {
    fn to_markdown(&self) -> String {
        let mut out = String::new();
        if self.reports.is_empty() {
            out.push_str("(no topic reports)\n");
        } else {
            out.push_str("**Topic health**\n");
            for r in &self.reports {
                out.push_str(&format!(
                    "- {}: count={} avg_strength={:.3} stale={} consolidate={}\n",
                    r.topic, r.count, r.avg_strength, r.stale_count, r.needs_consolidation
                ));
            }
        }
        out.push_str(&format!(
            "\n**System**: {}",
            if self.status.ok { "OK" } else { "DEGRADED" }
        ));
        if !self.status.issues.is_empty() {
            out.push_str("\nIssues:");
            for issue in &self.status.issues {
                out.push_str(&format!("\n- {}", issue));
            }
        }
        out.push_str(&format!(
            "\n**Queues**: memory p={} i={} d={} | cleanup p={} i={} d={} | dedup p={} i={} d={} | merge_refine p={} i={} d={}",
            self.queues.memory.pending, self.queues.memory.inflight, self.queues.memory.dead_letters,
            self.queues.cleanup.pending, self.queues.cleanup.inflight, self.queues.cleanup.dead_letters,
            self.queues.dedup.pending, self.queues.dedup.inflight, self.queues.dedup.dead_letters,
            self.queues.merge_refinement.pending, self.queues.merge_refinement.inflight, self.queues.merge_refinement.dead_letters,
        ));
        out.push_str(&format!(
            "\n**Grayzone pending**: {}",
            self.grayzone.pending
        ));
        out
    }
}

impl IntoCliText for HealthOutput {
    fn to_cli_text(&self) -> String {
        self.to_markdown()
    }
}

impl OpsRuntime {
    #[op(
        name = "health",
        category = "health",
        description = "Per-topic memory health + system index/queue lag",
        cli(name = "health"),
        mcp(name = "rein_health"),
        rest(method = "GET", path = "/api/health")
    )]
    pub fn health(&self, params: HealthParams) -> ReinResult<HealthOutput> {
        let topic = params.topic.as_deref();
        let store = self.config.open_store()?;
        let reports = store.health(topic)?;
        let system = system_health::collect(&store, &self.config);

        let reports = reports.into_iter().map(HealthReportItem::from).collect();

        Ok(HealthOutput {
            reports,
            indexes: system.indexes,
            queues: system.queues,
            grayzone: system.grayzone,
            status: system.status,
        })
    }
}

#[derive(Args, Deserialize, JsonSchema, Debug, Clone, Default)]
pub struct DoctorParams {
    /// Emit machine-readable JSON on CLI (REST always returns JSON).
    #[serde(default)]
    #[arg(long)]
    pub json: bool,
    /// Probe the embedding backend with a real network request.
    #[serde(default)]
    #[arg(long)]
    pub network: bool,
    /// Apply safe local fixes (side-index rebuilds, queue repair).
    #[serde(default)]
    #[arg(long)]
    pub fix: bool,
}

/// CLI and REST wrapper around `DoctorReport`. `json_cli` is a render hint set
/// from the `--json` flag; it's skipped from JSON output so the REST body stays
/// a plain `DoctorReport` (identical to the pre-A1 `/api/doctor` shape).
#[derive(Serialize, Clone, Debug)]
#[serde(transparent)]
pub struct DoctorOutput {
    pub report: DoctorReport,
    #[serde(skip)]
    pub json_cli: bool,
}

impl IntoJson for DoctorOutput {
    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(&self.report).unwrap_or(serde_json::Value::Null)
    }
}

impl IntoMarkdown for DoctorOutput {
    fn to_markdown(&self) -> String {
        doctor::format_human(&self.report)
    }
}

impl IntoCliText for DoctorOutput {
    fn to_cli_text(&self) -> String {
        if self.json_cli {
            serde_json::to_string_pretty(&self.report)
                .unwrap_or_else(|e| format!("Error serializing report: {e}"))
        } else {
            doctor::format_human(&self.report)
        }
    }
}

impl OpsRuntime {
    #[op(
        name = "doctor",
        category = "diagnostics",
        description = "Run system diagnostics: database, indexes, queues, provider readiness. CLI supports --json/--network/--fix; REST GET is read-only (ignores fix); POST /api/doctor applies fixes — see `doctor_fix` op.",
        cli(name = "doctor"),
        rest(method = "GET", path = "/api/doctor")
    )]
    pub async fn doctor(&self, params: DoctorParams) -> ReinResult<DoctorOutput> {
        let report = doctor::run(
            self.config.as_ref(),
            DoctorOptions {
                network: params.network,
                fix: params.fix,
            },
        )
        .await;
        // Preserve the pre-A1 CLI contract: exit 1 on any FAIL check so CI
        // scripts that grep the exit status keep working. MCP/REST surfaces
        // don't surface exit codes; the adapter ignores non-CLI exits.
        if matches!(self.surface(), SurfaceKind::Cli) {
            self.set_exit_code(report.exit_code());
        }
        Ok(DoctorOutput {
            report,
            json_cli: params.json,
        })
    }
}

/// Body params accepted by `POST /api/doctor`. `fix` defaults to `true`
/// because the whole reason to POST (vs GET) is to apply fixes, but
/// callers can explicitly set `fix: false` to run a mutation-authorized
/// read-only probe — that preserves the pre-migration query-string
/// behavior of `?fix=false` (the GUI does not use this knob today, but
/// external admin scripts might rely on it to assert-without-repair).
/// Extra fields in the JSON body are ignored so client evolution doesn't
/// wedge the server.
#[derive(Deserialize, JsonSchema, Debug, Clone)]
pub struct DoctorFixParams {
    /// Probe the embedding backend during the diagnostic.
    #[serde(default)]
    pub network: bool,
    /// Apply repairs (default true for POST). Set false for a dry-run
    /// diagnostic that still requires the mutation marker.
    #[serde(default = "default_doctor_fix_flag")]
    pub fix: bool,
}

fn default_doctor_fix_flag() -> bool {
    true
}

impl Default for DoctorFixParams {
    fn default() -> Self {
        Self {
            network: false,
            fix: true,
        }
    }
}

impl OpsRuntime {
    #[op(
        name = "doctor_fix",
        category = "diagnostics",
        description = "Run the doctor diagnostic with mutation authorization; defaults to applying fixes. Pass `fix: false` for an authed dry run. REST-only POST counterpart to the read-only GET /api/doctor.",
        mutating = true,
        rest(method = "POST", path = "/api/doctor"),
        auth = "mutation_marker"
    )]
    pub async fn doctor_fix(&self, params: DoctorFixParams) -> ReinResult<DoctorOutput> {
        // The H3 auth framework enforces `x-rein-action: 1` on this route
        // (see mcp/rest.rs enforce_auth_policy + AuthPolicy::MutationMarker);
        // the op body runs only after that gate passes.
        let report = doctor::run(
            self.config.as_ref(),
            DoctorOptions {
                network: params.network,
                fix: params.fix,
            },
        )
        .await;
        Ok(DoctorOutput {
            report,
            json_cli: false,
        })
    }
}

/// Summary of non-secret configuration values surfaced via `rein config`.
///
/// Scope is deliberately narrow: paths + provider/model names + numeric
/// knobs that shape behavior. API keys, tokens, and anything else loaded
/// from environment variables stay out. The CLI-only surface reinforces
/// that — if `rein config` is later promoted to REST/MCP, this struct is
/// already safe to serialize.
#[derive(Serialize, Clone, Debug)]
pub struct ConfigSnapshot {
    pub database_path: String,
    pub embedding_provider: String,
    pub embedding_dimensions: usize,
    pub extract_provider: String,
    pub extract_model: String,
    pub compact_mode: bool,
    pub sse_enabled: bool,
    pub decay_base_lambda: f64,
    pub dedup_similarity: f64,
}

impl IntoJson for ConfigSnapshot {
    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

impl IntoMarkdown for ConfigSnapshot {
    fn to_markdown(&self) -> String {
        format!(
            "**Configuration**\n\
             - Database path: {}\n\
             - Embedding provider: {}\n\
             - Embedding dimensions: {}\n\
             - Extract provider: {}\n\
             - Extract model: {}\n\
             - Compact mode: {}\n\
             - SSE enabled: {}\n\
             - Decay base_lambda: {}\n\
             - Dedup similarity: {}",
            self.database_path,
            self.embedding_provider,
            self.embedding_dimensions,
            self.extract_provider,
            self.extract_model,
            self.compact_mode,
            self.sse_enabled,
            self.decay_base_lambda,
            self.dedup_similarity,
        )
    }
}

impl IntoCliText for ConfigSnapshot {
    fn to_cli_text(&self) -> String {
        // Mirror the pre-A1 `rein config` line-by-line output for terminal
        // users that may grep it in scripts.
        format!(
            "Database path: {}\n\
             Embedding provider: {}\n\
             Embedding dimensions: {}\n\
             Extract provider: {}\n\
             Extract model: {}\n\
             Compact mode: {}\n\
             SSE enabled: {}\n\
             Decay base_lambda: {}\n\
             Dedup similarity: {}",
            self.database_path,
            self.embedding_provider,
            self.embedding_dimensions,
            self.extract_provider,
            self.extract_model,
            self.compact_mode,
            self.sse_enabled,
            self.decay_base_lambda,
            self.dedup_similarity,
        )
    }
}

impl OpsRuntime {
    #[op(
        name = "config",
        category = "diagnostics",
        description = "Show non-secret configuration: database path, providers, models, and tunable knobs. CLI-only to avoid accidental exposure over the network.",
        cli(name = "config")
    )]
    pub fn config_snapshot(&self) -> ReinResult<ConfigSnapshot> {
        let cfg = self.config.as_ref();
        let extract_model = match cfg.extract_provider() {
            crate::config::Provider::Omlx => cfg.extract.omlx.model.clone(),
            _ => cfg.extract.google.model.clone(),
        };
        Ok(ConfigSnapshot {
            database_path: cfg.resolve_db_path().display().to_string(),
            embedding_provider: cfg.embedding.provider.clone(),
            embedding_dimensions: cfg.embedding.dimensions,
            extract_provider: cfg.extract.provider.clone(),
            extract_model,
            compact_mode: cfg.server.compact,
            sse_enabled: cfg.server.sse_enabled,
            decay_base_lambda: cfg.decay.base_lambda,
            dedup_similarity: cfg.search.dedup_similarity,
        })
    }
}
