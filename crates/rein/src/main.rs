use std::sync::Arc;

use clap::{CommandFactory, FromArgMatches, Parser, Subcommand};

use rein::config;
use rein::ops::{OpsCliEntry, OpsRuntime};

mod commands;

#[derive(Parser)]
#[command(
    name = "rein",
    version,
    about = "Multi-source cross-validated memory for AI agents"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start MCP server (stdio transport)
    Serve {
        /// Enable compact output mode
        #[arg(long)]
        compact: bool,
        /// Enable SSE transport (HTTP)
        #[arg(long)]
        sse: bool,
        /// Start transparent proxy for LLM API recording
        #[arg(long)]
        proxy: bool,
        /// Enable web GUI (implies --sse)
        #[arg(long)]
        gui: bool,
    },
    /// Ingest a full session/transcript through the extraction pipeline
    Ingest {
        #[arg(short, long, conflicts_with = "file")]
        content: Option<String>,
        #[arg(short, long, conflicts_with = "content")]
        file: Option<String>,
        #[arg(long, conflicts_with_all = ["content", "file"])]
        json_file: Option<String>,
        #[arg(long)]
        asynchronous: bool,
        #[arg(long)]
        agent_label: Option<String>,
        #[arg(long)]
        subagent: bool,
    },
    // Stats + Health migrated to #[op] inventory (see ops/handlers/diagnostics.rs).
    // main() intercepts them before clap::Parser so they're invoked via OpsCliEntry.
    // Doctor migrated to #[op] inventory (see ops/handlers/diagnostics.rs).
    // main() intercepts it via OpsCliEntry before the derived Parser path, and
    // the inventory CLI dispatcher honors the op's `set_exit_code(1)` call so
    // `rein doctor` still exits 1 on any FAIL check for CI scripts.
    // Consolidate migrated to #[op] inventory (see ops/handlers/maintenance.rs).
    // main() intercepts "consolidate" via OpsCliEntry before the derived-enum path.
    // MCP: rein_consolidate. REST: POST /api/consolidate. auth = "mutation_marker".
    // Dedup migrated to #[op] inventory (see ops/handlers/maintenance.rs).
    // main() intercepts "dedup" via OpsCliEntry before the derived-enum path.
    // MCP: rein_dedup. REST: POST /api/dedup. auth = "mutation_marker".
    // IntelligentMergeTry migrated to #[op] inventory (see ops/handlers/maintenance.rs).
    // main() intercepts "intelligent-merge-try" via OpsCliEntry before the derived-enum path.
    // CLI-only surface — no MCP, no REST, no auth attribute.
    // Cleanup migrated to #[op] inventory (see ops/handlers/maintenance.rs).
    // main() intercepts "cleanup" via OpsCliEntry before the derived-enum path.
    // MCP: rein_cleanup (derived→inventory, net zero). REST: POST /api/cleanup. auth = "mutation_marker".
    // NOTE: the legacy --asynchronous flag (queue_cleanup_job) is not carried forward;
    // use `rein worker cleanup <args>` for background worker invocation.
    /// Auto-configure MCP clients
    Init {
        #[arg(long)]
        dry_run: bool,
        /// Configure shell aliases for proxy (rein-proxy, claudep, codexp)
        #[arg(long)]
        proxy: bool,
    },
    // Canonicals migrated to #[op] inventory (see ops/handlers/maintenance.rs).
    // main() intercepts "canonicals" via OpsCliEntry before the derived-enum path.
    // Evidence migrated to #[op] inventory (see ops/handlers/maintenance.rs).
    // main() intercepts "evidence" via OpsCliEntry before the derived-enum path.
    // Gc migrated to #[op] inventory (see ops/handlers/maintenance.rs).
    // main() intercepts "gc" via OpsCliEntry before the derived-enum path.
    // DedupLog migrated to #[op] inventory (see ops/handlers/maintenance.rs).
    // main() intercepts "dedup-log" via OpsCliEntry before the derived-enum path.
    // DedupConcepts migrated to #[op] inventory (see ops/handlers/maintenance.rs).
    // main() intercepts "dedup-concepts" via OpsCliEntry before the derived-enum path.
    // Organize migrated to #[op] inventory (see ops/handlers/maintenance.rs).
    // main() intercepts "organize" via OpsCliEntry before the derived-enum path.
    /// Export memories to file
    Export {
        /// Output format: md, json, or csv (default json)
        #[arg(short, long, default_value = "json")]
        format: String,
        /// Only export memories in this topic
        #[arg(short, long)]
        topic: Option<String>,
        /// Output file (default stdout)
        #[arg(short, long)]
        output: Option<String>,
    },
    /// Upgrade old memories into knowledge graph (concepts + links)
    Upgrade {
        /// Only process memories in this topic
        #[arg(short, long)]
        topic: Option<String>,
        /// Preview what would be extracted without storing
        #[arg(long)]
        dry_run: bool,
    },
    /// Pre-compute embeddings for uncached memories
    Warmup,
    // Config migrated to #[op] inventory (see ops/handlers/diagnostics.rs).
    // CLI-only surface — producing a typed ConfigSnapshot keeps the subset
    // of non-secret fields explicit in case the op is later exposed to
    // REST/MCP.
    // AdaptiveStatus migrated to #[op] inventory (see ops/handlers/adaptive.rs).
    // main() intercepts it via OpsCliEntry before the derived Parser path.
    /// Background worker commands
    Worker {
        #[command(subcommand)]
        action: WorkerAction,
    },
    /// Hook commands
    Hook {
        #[command(subcommand)]
        action: HookAction,
    },
    /// Show service status dashboard
    Dashboard,
    /// Manage GUI server
    Gui {
        #[arg(value_enum)]
        action: ServiceAction,
    },
    /// Manage proxy server
    Proxy {
        #[arg(value_enum)]
        action: ServiceAction,
    },
    /// v0.27.1 E direction Layer 2: run the nightly stricter offline LLM
    /// judge calibration cron. Reads the 24h cron-archive jsonl, joins
    /// against runtime judge verdicts in feedback_events by synthesis_id,
    /// re-judges via the `[ars.llm_judge.nightly_cron]`-resolved LLM, and
    /// emits SynthesisLlmJudgeOfflineCron / ConceptSummaryLlmJudgeOfflineCron
    /// events. The `judge_calibration` M1 consumer absorbs them on the
    /// next adaptive-pipeline pass and recomputes `runtime_vs_offline_kappa`
    /// + bumps `judge_drift_alert` when κ falls below threshold.
    ///
    /// Default-off in v0.27.1 (`[ars.llm_judge.nightly_cron].enabled = false`).
    #[command(name = "judge-calibrate-cron")]
    JudgeCalibrateCron {
        /// Print verbose per-entry processing logs. Default false (only the
        /// final summary report is printed).
        #[arg(long)]
        verbose: bool,
    },
}

#[derive(Clone, clap::ValueEnum)]
enum ServiceAction {
    On,
    Off,
}

#[derive(Subcommand)]
enum HookAction {
    /// Extract facts from tool output (PostToolUse)
    Post,
    /// Extract context before compaction (PreCompact)
    Compact,
    /// UserPromptSubmit compatibility hook (currently a no-op)
    Prompt,
    /// Save session summary on conversation end (Stop)
    Stop,
}

#[derive(Subcommand)]
enum WorkerAction {
    /// Drain the async memory queue for the current project
    Memory,
    /// Drain queued store-time dedup jobs for the current project
    DedupQueue,
    /// Run a detached cleanup pass in the current project
    Cleanup {
        topic: Option<String>,
        #[arg(long, value_delimiter = ',')]
        topics: Option<Vec<String>>,
        #[arg(long)]
        pattern: Option<String>,
        #[arg(long)]
        all: bool,
        #[arg(long)]
        exact_topics: bool,
        #[arg(long)]
        dry_run: bool,
    },
    /// Drain queued cleanup jobs for the current project
    CleanupQueue,
    /// Drain queued post-merge LLM synthesis jobs for the current project
    MergeRefinementQueue,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("REIN_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    // A1: build the clap command and augment it with inventory-registered
    // subcommands so `--help` lists them and clap can parse their args. Then
    // dispatch: if an inventory subcommand matched, invoke its fn pointer;
    // otherwise fall through to the derived-enum path.
    let augmented = augment_with_inventory(Cli::command());
    let matches = augmented.get_matches();

    if let Some((sub_name, sub_matches)) = matches.subcommand() {
        if let Some(entry) = inventory::iter::<OpsCliEntry>().find(|e| e.name == sub_name) {
            let config = config::ReinConfig::load()?;
            let runtime = Arc::new(OpsRuntime::for_cli(Arc::new(config)));
            let out = (entry.invoke)(runtime.clone(), sub_matches).await?;
            println!("{out}");
            if let Some(code) = runtime.take_exit_code() {
                std::process::exit(code);
            }
            return Ok(());
        }
    }

    let cli = Cli::from_arg_matches(&matches)
        .map_err(|e| anyhow::anyhow!("cli arg matches → struct: {e}"))?;
    let config = config::ReinConfig::load()?;

    match cli.command {
        Some(Commands::Serve {
            compact,
            sse,
            proxy,
            gui,
        }) => commands::handle_serve(config, compact, sse, proxy, gui).await?,
        Some(Commands::Ingest {
            content,
            file,
            json_file,
            asynchronous,
            agent_label,
            subagent,
        }) => {
            commands::handle_ingest(
                &config,
                content,
                file,
                json_file,
                asynchronous,
                agent_label,
                subagent,
            )
            .await?
        }
        // Commands::Organize migrated to #[op] inventory — intercepted via OpsCliEntry above.
        Some(Commands::Export {
            format,
            topic,
            output,
        }) => commands::handle_export(&config, format, topic, output)?,
        Some(Commands::Upgrade { topic, dry_run }) => {
            commands::handle_upgrade(&config, topic, dry_run).await?
        }
        Some(Commands::Warmup) => commands::handle_warmup(&config).await?,
        Some(Commands::Worker { action }) => match action {
            WorkerAction::Memory => commands::handle_worker_memory(&config).await?,
            WorkerAction::DedupQueue => commands::handle_worker_dedup_queue(&config).await?,
            WorkerAction::CleanupQueue => commands::handle_worker_cleanup_queue(&config).await?,
            WorkerAction::MergeRefinementQueue => {
                commands::handle_worker_merge_refinement_queue(&config).await?
            }
            WorkerAction::Cleanup {
                topic,
                topics,
                pattern,
                all,
                exact_topics,
                dry_run,
            } => {
                commands::handle_worker_cleanup(
                    &config,
                    topic,
                    topics,
                    pattern,
                    all,
                    exact_topics,
                    dry_run,
                )
                .await?
            }
        },
        None => {
            println!("rein v{}", env!("CARGO_PKG_VERSION"));
            println!("Run 'rein --help' for usage");
        }
        Some(Commands::Hook { action }) => {
            let action_str = match action {
                HookAction::Post => "post",
                HookAction::Compact => "compact",
                HookAction::Prompt => "prompt",
                HookAction::Stop => "stop",
            };
            commands::handle_hook(&config, action_str).await?
        }
        Some(Commands::Init { dry_run, proxy }) => commands::handle_init(dry_run, proxy)?,
        // Commands::Consolidate removed — intercepted above via OpsCliEntry.
        // Commands::Cleanup removed — intercepted above via OpsCliEntry.
        // DedupLog migrated to #[op] inventory (dedup_log op); intercepted above via OpsCliEntry.
        Some(Commands::Dashboard) => commands::handle_dashboard(&config),
        Some(Commands::Gui { action }) => match action {
            ServiceAction::On => commands::handle_gui_on()?,
            ServiceAction::Off => commands::handle_gui_off()?,
        },
        Some(Commands::Proxy { action }) => match action {
            ServiceAction::On => commands::handle_proxy_on()?,
            ServiceAction::Off => commands::handle_proxy_off()?,
        },
        Some(Commands::JudgeCalibrateCron { verbose }) => {
            // v0.27.1 E direction Layer 2 — emit-only cron. Re-judges
            // sampled syntheses with stricter LLM, writes OfflineCron
            // events. The `judge_calibration` consumer absorbs them on
            // the next slow-channel pass and updates κ + drift alert.
            let store = config.open_store()?;
            let report = rein::ops::judge_calibration::run_judge_calibration_cron(&store, &config)?;
            if verbose {
                println!("considered:                  {}", report.considered);
                println!("emitted (OfflineCron events): {}", report.emitted);
                println!(
                    "skipped (no runtime verdict): {}",
                    report.skipped_no_runtime_verdict
                );
                println!("dropped (errors):             {}", report.dropped);
                println!(
                    "dropped (cap reservation):    {} [Wave-1.5: R9-K1 reserve_call wiring pending]",
                    report.dropped_cap
                );
            } else {
                println!(
                    "judge-calibrate-cron: considered={} emitted={} skipped={} dropped={}",
                    report.considered,
                    report.emitted,
                    report.skipped_no_runtime_verdict,
                    report.dropped,
                );
            }
        }
    }
    Ok(())
}

/// Inject each `#[op]`-registered CLI subcommand into the top-level clap
/// Command so `--help` and argument parsing work uniformly for migrated and
/// legacy ops alike. Aborts with a clear message if two entries share a name —
/// the same check exists at test time via `inventory_registration.rs` but
/// catching it at startup prevents link-order-dependent silent shadowing.
fn augment_with_inventory(mut cmd: clap::Command) -> clap::Command {
    rein::ops::inventory::ensure_unique_registrations();
    let mut seen: std::collections::HashSet<&'static str> = std::collections::HashSet::new();
    for entry in inventory::iter::<OpsCliEntry>() {
        if !seen.insert(entry.name) {
            panic!(
                "duplicate OpsCliEntry name '{}': two #[op]s registered the same CLI subcommand",
                entry.name
            );
        }
        cmd = cmd.subcommand((entry.build_clap)());
    }
    cmd
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_parses_doctor_flags() {
        // doctor migrated to #[op]; parse via the inventory-built clap command
        // so regressions in the macro's CLI arg emission surface here.
        let entry = inventory::iter::<OpsCliEntry>()
            .find(|e| e.name == "doctor")
            .expect("doctor CLI entry registered");
        let matches = (entry.build_clap)()
            .try_get_matches_from(["doctor", "--json", "--network", "--fix"])
            .expect("doctor flags parse");
        assert!(matches.get_flag("json"));
        assert!(matches.get_flag("network"));
        assert!(matches.get_flag("fix"));
    }
}
