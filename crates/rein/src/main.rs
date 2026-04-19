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
    /// Store a memory
    Store {
        #[arg(short, long)]
        topic: String,
        #[arg(short, long)]
        content: String,
        #[arg(short = 'I', long, default_value = "medium")]
        importance: String,
        #[arg(short, long, value_delimiter = ',')]
        keywords: Option<Vec<String>>,
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
    /// Search memories
    Recall {
        query: String,
        #[arg(short, long)]
        topic: Option<String>,
        #[arg(short, long)]
        keyword: Option<String>,
        #[arg(short, long, default_value = "10")]
        limit: usize,
    },
    /// Delete a memory
    Forget { id: String },
    /// Update a memory
    Update {
        id: String,
        #[arg(short, long)]
        content: String,
        #[arg(short = 'I', long)]
        importance: Option<String>,
    },
    /// List all topics
    Topics,
    // Stats + Health migrated to #[op] inventory (see ops/handlers/diagnostics.rs).
    // main() intercepts them before clap::Parser so they're invoked via OpsCliEntry.
    // Doctor migrated to #[op] inventory (see ops/handlers/diagnostics.rs).
    // main() intercepts it via OpsCliEntry before the derived Parser path, and
    // the inventory CLI dispatcher honors the op's `set_exit_code(1)` call so
    // `rein doctor` still exits 1 on any FAIL check for CI scripts.
    /// Consolidate a topic into a single memory
    Consolidate {
        /// Single topic to consolidate (legacy mode)
        topic: Option<String>,
        #[arg(short, long)]
        summary: Option<String>,
        /// Comma-separated topic list
        #[arg(long, value_delimiter = ',')]
        topics: Option<Vec<String>>,
        /// Glob pattern for topics, e.g. "rmcp*"
        #[arg(long)]
        pattern: Option<String>,
        /// Consolidate all topics
        #[arg(long)]
        all: bool,
        /// Group case/space/hyphen variants before consolidating
        #[arg(long)]
        merge_variants: bool,
        /// Preview matched groups without writing changes
        #[arg(long)]
        dry_run: bool,
    },
    /// Scan for duplicates
    Dedup {
        #[arg(long)]
        dry_run: bool,
        /// Deduplicate across normalized topic variants instead of only exact topics
        #[arg(long)]
        merge_variants: bool,
    },
    /// Run intelligent merge classifier on two memory IDs (POC — read-only, prints verdict)
    IntelligentMergeTry {
        /// Existing memory ID
        existing: String,
        /// Incoming memory ID to compare against
        incoming: String,
    },
    /// One-click cleanup: consolidate fragmented topics, deduplicate, refresh adaptive state
    Cleanup {
        /// Optional single topic to clean
        topic: Option<String>,
        /// Optional comma-separated topic list to clean
        #[arg(long, value_delimiter = ',')]
        topics: Option<Vec<String>>,
        /// Optional glob pattern for matching topics
        #[arg(long)]
        pattern: Option<String>,
        /// Force processing all topics (default when no selector is provided)
        #[arg(long)]
        all: bool,
        /// Disable topic-variant grouping; use exact topic boundaries only
        #[arg(long)]
        exact_topics: bool,
        /// Preview matched groups without writing changes
        #[arg(long)]
        dry_run: bool,
        /// Spawn cleanup in a detached background worker process
        #[arg(long)]
        asynchronous: bool,
    },
    /// Migrate from QMD
    Migrate {
        #[arg(long)]
        from_qmd: Option<String>,
        /// Re-embed all memories with the current embedding model
        #[arg(long)]
        reindex: bool,
    },
    /// Auto-configure MCP clients
    Init {
        #[arg(long)]
        dry_run: bool,
        /// Configure shell aliases for proxy (rein-proxy, claudep, codexp)
        #[arg(long)]
        proxy: bool,
    },
    /// Show most recently created memories
    #[command(visible_alias = "list")]
    Recent {
        #[arg(short, long, default_value = "10")]
        limit: usize,
    },
    /// Show canonical memories (one row per canonical)
    Canonicals {
        #[arg(short, long, default_value = "20")]
        limit: usize,
    },
    /// Show evidence snapshots for a canonical memory
    Evidence {
        canonical_id: String,
        #[arg(short, long, default_value = "20")]
        limit: usize,
    },
    /// Show recent dedup decisions
    DedupLog {
        #[arg(long)]
        canonical: Option<String>,
        #[arg(short, long, default_value = "20")]
        limit: usize,
    },
    /// Garbage collect weak/stale STM memories below strength threshold
    Gc {
        #[arg(long)]
        dry_run: bool,
    },
    /// Auto-link related memories based on content similarity
    Organize,
    /// Deduplicate concepts with same normalized name (case/separator variants)
    DedupConcepts,
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
        Some(Commands::Store {
            topic,
            content,
            importance,
            keywords,
        }) => commands::handle_store(&config, topic, content, importance, keywords)?,
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
        Some(Commands::Recall {
            query,
            topic,
            keyword,
            limit,
        }) => commands::handle_recall(&config, query, topic, keyword, limit)?,
        Some(Commands::Topics) => commands::handle_topics(&config)?,
        Some(Commands::Forget { id }) => commands::handle_forget(&config, id)?,
        Some(Commands::Update {
            id,
            content,
            importance,
        }) => commands::handle_update(&config, id, content, importance)?,
        Some(Commands::Recent { limit }) => commands::handle_recent(&config, limit)?,
        Some(Commands::Gc { dry_run }) => commands::handle_gc(&config, dry_run)?,
        Some(Commands::Organize) => commands::handle_organize(&config)?,
        Some(Commands::DedupConcepts) => commands::handle_dedup_concepts(&config)?,
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
        Some(Commands::Migrate { from_qmd, reindex }) => {
            commands::handle_migrate(&config, from_qmd, reindex).await?
        }
        Some(Commands::Init { dry_run, proxy }) => commands::handle_init(dry_run, proxy)?,
        Some(Commands::Consolidate {
            topic,
            summary,
            topics,
            pattern,
            all,
            merge_variants,
            dry_run,
        }) => {
            commands::handle_consolidate(
                &config,
                topic,
                summary,
                topics,
                pattern,
                all,
                merge_variants,
                dry_run,
            )
            .await?
        }
        Some(Commands::Dedup {
            dry_run,
            merge_variants,
        }) => commands::handle_dedup(&config, dry_run, merge_variants)?,
        Some(Commands::IntelligentMergeTry { existing, incoming }) => {
            commands::handle_intelligent_merge_try(&config, &existing, &incoming)?
        }
        Some(Commands::Cleanup {
            topic,
            topics,
            pattern,
            all,
            exact_topics,
            dry_run,
            asynchronous,
        }) => {
            commands::handle_cleanup(
                &config,
                topic,
                topics,
                pattern,
                all,
                exact_topics,
                dry_run,
                asynchronous,
            )
            .await?
        }
        Some(Commands::Canonicals { limit }) => commands::handle_canonicals(&config, limit)?,
        Some(Commands::Evidence {
            canonical_id,
            limit,
        }) => commands::handle_evidence(&config, canonical_id, limit)?,
        Some(Commands::DedupLog { canonical, limit }) => {
            commands::handle_dedup_log(&config, canonical, limit)?
        }
        Some(Commands::Dashboard) => commands::handle_dashboard(&config),
        Some(Commands::Gui { action }) => match action {
            ServiceAction::On => commands::handle_gui_on()?,
            ServiceAction::Off => commands::handle_gui_off()?,
        },
        Some(Commands::Proxy { action }) => match action {
            ServiceAction::On => commands::handle_proxy_on()?,
            ServiceAction::Off => commands::handle_proxy_off()?,
        },
    }
    Ok(())
}

/// Inject each `#[op]`-registered CLI subcommand into the top-level clap
/// Command so `--help` and argument parsing work uniformly for migrated and
/// legacy ops alike. Aborts with a clear message if two entries share a name —
/// the same check exists at test time via `inventory_registration.rs` but
/// catching it at startup prevents link-order-dependent silent shadowing.
fn augment_with_inventory(mut cmd: clap::Command) -> clap::Command {
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
