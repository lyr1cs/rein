use clap::{Parser, Subcommand};

use rein::config;
use rein::embed;
use rein::extract;
use rein::mcp;
use rein::ops;
use rein::search;
use rein::store;
use rein::types;
use rein::types::MemoryStore;

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
    /// Show statistics
    Stats,
    /// Health check
    Health { topic: Option<String> },
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
    /// Show configuration
    Config,
    /// Show adaptive engine status (learned parameters, convergence info)
    AdaptiveStatus,
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

    let cli = Cli::parse();
    let config = config::ReinConfig::load()?;

    match cli.command {
        Some(Commands::Serve {
            compact,
            sse,
            proxy,
            gui,
        }) => {
            let mut config = config;
            if compact {
                config.server.compact = true;
            }
            if gui {
                config.server.gui_enabled = true;
                config.server.sse_enabled = true; // GUI implies SSE mode
            }
            if proxy {
                // Set before entering async proxy to avoid set_var in multi-threaded context.
                std::env::set_var("REIN_PROXY_ACTIVE", "1");
                rein::proxy::run_proxy(config).await?;
            } else if sse || gui {
                config.server.sse_enabled = true;
                mcp::server::run_http(config).await?;
            } else {
                mcp::server::run_stdio(config).await?;
            }
        }
        Some(Commands::Store {
            topic,
            content,
            importance,
            keywords,
        }) => {
            let store = config.open_store()?;
            let imp: types::Importance =
                importance.parse().map_err(|e: String| anyhow::anyhow!(e))?;
            let memory = rein::ops::build_memory(
                &config,
                topic,
                content.clone(),
                imp,
                keywords.unwrap_or_default(),
                types::Source::Manual,
            );
            let id = rein::ops::store_memory(&store, &config, memory)?;
            println!(
                "{}",
                mcp::compact::format_store_result(&id, config.server.compact)
            );
        }
        Some(Commands::Ingest {
            content,
            file,
            json_file,
            asynchronous,
            agent_label,
            subagent,
        }) => {
            let report = match (content, file, json_file) {
                (Some(text), None, None) => {
                    if asynchronous {
                        rein::ops::queue_ingest_session_text(
                            &config,
                            &text,
                            agent_label.as_deref(),
                            subagent,
                        )?
                    } else {
                        rein::ops::ingest_session_text_report(
                            &config,
                            &text,
                            agent_label.as_deref(),
                            subagent,
                        )
                        .await?
                    }
                }
                (None, Some(path), None) => {
                    let text = std::fs::read_to_string(path)?;
                    if asynchronous {
                        rein::ops::queue_ingest_session_text(
                            &config,
                            &text,
                            agent_label.as_deref(),
                            subagent,
                        )?
                    } else {
                        rein::ops::ingest_session_text_report(
                            &config,
                            &text,
                            agent_label.as_deref(),
                            subagent,
                        )
                        .await?
                    }
                }
                (None, None, Some(path)) => {
                    let raw = std::fs::read_to_string(path)?;
                    let session: types::SessionIngest = serde_json::from_str(&raw)?;
                    if asynchronous {
                        rein::ops::queue_ingest_session(
                            &config,
                            &session,
                            agent_label.as_deref(),
                            subagent,
                        )?
                    } else {
                        rein::ops::ingest_session_report(
                            &config,
                            &session,
                            agent_label.as_deref(),
                            subagent,
                        )
                        .await?
                    }
                }
                _ => {
                    return Err(anyhow::anyhow!(
                        "provide exactly one of --content, --file, or --json-file"
                    ));
                }
            };
            if config.server.compact {
                println!(
                    "ok queued:{} memories:{} concepts:{} links:{} artifact:{} episode:{}",
                    report.queued,
                    report.memory_count,
                    report.concept_count,
                    report.link_count,
                    report.artifact_id.as_deref().unwrap_or("-"),
                    report.episode_id.as_deref().unwrap_or("-"),
                );
            } else {
                println!(
                    "Ingested session: queued={} artifact={} episode={} memories={} concepts={} links={}",
                    report.queued,
                    report.artifact_id.as_deref().unwrap_or("-"),
                    report.episode_id.as_deref().unwrap_or("-"),
                    report.memory_count,
                    report.concept_count,
                    report.link_count
                );
            }
        }
        Some(Commands::Recall {
            query,
            topic,
            keyword,
            limit,
        }) => {
            let store = config.open_store()?;
            let results = search::recall::recall(
                &store,
                &config,
                &query,
                topic.as_deref(),
                keyword.as_deref(),
                limit,
            )?;
            let scored: Vec<(types::Memory, f32)> =
                results.into_iter().map(|r| (r.memory, r.score)).collect();
            println!(
                "{}",
                mcp::compact::format_recall_results(&scored, config.server.compact)
            );
        }
        Some(Commands::Topics) => {
            let store = config.open_store()?;
            let topics = store.list_topics()?;
            println!(
                "{}",
                mcp::compact::format_topics(&topics, config.server.compact)
            );
        }
        Some(Commands::Stats) => {
            let store = config.open_store()?;
            let stats = store.stats()?;
            println!(
                "{}",
                mcp::compact::format_stats(&stats, config.server.compact)
            );
        }
        Some(Commands::Health { topic }) => {
            let store = config.open_store()?;
            let reports = store.health(topic.as_deref())?;
            println!(
                "{}",
                mcp::compact::format_health(&reports, config.server.compact)
            );
        }
        Some(Commands::Forget { id }) => {
            let store = config.open_store()?;
            store.delete(&id)?;
            println!("Deleted memory: {id}");
        }
        Some(Commands::Update {
            id,
            content,
            importance,
        }) => {
            let store = config.open_store()?;
            let mut mem = store.get(&id)?;
            mem.content = content.clone();
            mem.summary = content.chars().take(100).collect();
            if let Some(imp_str) = importance {
                let imp: types::Importance =
                    imp_str.parse().map_err(|e: String| anyhow::anyhow!(e))?;
                mem.importance = imp;
                mem.layer = imp.auto_layer();
                mem.decay_lambda = config.decay.base_lambda * imp.decay_factor();
            }
            mem.updated_at = chrono::Utc::now();
            store.update(&mem)?;
            println!("Updated memory: {id}");
        }
        Some(Commands::Config) => {
            println!("Database path: {}", config.resolve_db_path().display());
            println!("Embedding provider: {}", config.embedding.provider);
            println!("Embedding dimensions: {}", config.embedding.dimensions);
            println!("Extract provider: {}", config.extract.provider);
            println!(
                "Extract model: {}",
                match config.extract_provider() {
                    config::Provider::Omlx => &config.extract.omlx.model,
                    _ => &config.extract.google.model,
                }
            );
            println!("Compact mode: {}", config.server.compact);
            println!("SSE enabled: {}", config.server.sse_enabled);
            println!("Decay base_lambda: {}", config.decay.base_lambda);
            println!("Dedup similarity: {}", config.search.dedup_similarity);
        }
        Some(Commands::AdaptiveStatus) => {
            let store = config.open_store()?;
            let status = ops::adaptive_status(&store);
            println!("{}", serde_json::to_string_pretty(&status)?);
        }
        Some(Commands::Recent { limit }) => {
            let store = config.open_store()?;
            let memories = store.recent(limit)?;
            if memories.is_empty() {
                println!("No memories found.");
            } else {
                for m in &memories {
                    let age = chrono::Utc::now().signed_duration_since(m.created_at);
                    let age_str = if age.num_days() > 0 {
                        format!("{}d ago", age.num_days())
                    } else if age.num_hours() > 0 {
                        format!("{}h ago", age.num_hours())
                    } else {
                        format!("{}m ago", age.num_minutes())
                    };
                    println!(
                        "[{}] {} ({}, {})",
                        m.topic, m.summary, m.importance, age_str
                    );
                }
            }
        }
        Some(Commands::Gc { dry_run }) => {
            let store = config.open_store()?;
            let threshold = config.decay.prune_threshold;
            let (decayed, pruned, concepts) =
                ops::run_gc_adaptive(&store, &config, threshold, dry_run)?;
            if dry_run {
                let mut msg = format!("Would decay {decayed} and prune {pruned} weak STM memories (threshold: {threshold})");
                if concepts > 0 {
                    msg.push_str(&format!(", {concepts} low-quality concepts"));
                }
                println!("{msg}");
            } else {
                let mut msg = format!("Decayed {decayed} memories, pruned {pruned} weak STM memories (threshold: {threshold})");
                if concepts > 0 {
                    msg.push_str(&format!(", {concepts} low-quality concepts"));
                }
                println!("{msg}");
            }
        }
        Some(Commands::Organize) => {
            let store = config.open_store()?;
            let threshold = config.search.dedup_similarity as f32;
            let links = store.organize(threshold, 5)?;
            println!("Organized: created {links} new links between related memories");
        }
        Some(Commands::DedupConcepts) => {
            let store = config.open_store()?;
            let (groups, removed) = store.dedup_concepts()?;
            println!("Concept dedup: merged {groups} groups, removed {removed} duplicate concepts");
        }
        Some(Commands::Export {
            format,
            topic,
            output,
        }) => {
            let store = config.open_store()?;
            let topics = if let Some(ref t) = topic {
                vec![t.clone()]
            } else {
                store.list_topics()?
            };

            let mut all_memories: Vec<types::Memory> = Vec::new();
            for t in &topics {
                all_memories.extend(store.get_by_topic(t)?);
            }
            all_memories = store.collapse_to_canonicals(all_memories, usize::MAX)?;
            all_memories.sort_by(|a, b| b.created_at.cmp(&a.created_at));

            let content = match format.as_str() {
                "json" => serde_json::to_string_pretty(&all_memories)?,
                "csv" => {
                    let mut lines = vec![
                        "id,topic,summary,content,importance,keywords,strength,created_at"
                            .to_string(),
                    ];
                    for m in &all_memories {
                        let kw = m.keywords.join(";");
                        // Escape CSV fields
                        let esc = |s: &str| format!("\"{}\"", s.replace('"', "\"\""));
                        lines.push(format!(
                            "{},{},{},{},{},{},{:.3},{}",
                            m.id,
                            esc(&m.topic),
                            esc(&m.summary),
                            esc(&m.content),
                            m.importance,
                            esc(&kw),
                            m.strength,
                            m.created_at.to_rfc3339(),
                        ));
                    }
                    lines.join("\n")
                }
                "md" => {
                    let mut parts = vec![format!(
                        "# rein Memory Export\n\n{} memories, {} topics\n",
                        all_memories.len(),
                        topics.len()
                    )];
                    let mut current_topic = String::new();
                    // Group by topic
                    all_memories.sort_by(|a, b| {
                        a.topic.cmp(&b.topic).then(b.created_at.cmp(&a.created_at))
                    });
                    for m in &all_memories {
                        if m.topic != current_topic {
                            current_topic = m.topic.clone();
                            parts.push(format!("\n## {}\n", current_topic));
                        }
                        parts.push(format!(
                            "### {} ({})\n{}\n",
                            m.summary, m.importance, m.content
                        ));
                    }
                    parts.join("\n")
                }
                _ => {
                    eprintln!("Unknown format '{}', use json, csv, or md", format);
                    return Ok(());
                }
            };

            if let Some(ref path) = output {
                std::fs::write(path, &content)?;
                println!("Exported {} memories to {}", all_memories.len(), path);
            } else {
                println!("{content}");
            }
        }
        Some(Commands::Upgrade { topic, dry_run }) => {
            if extract::llm::create_extractor(&config).is_none() {
                eprintln!("rein: WARNING — no LLM configured. Upgrade will use local rules only.");
            }
            let store = config.open_store()?;
            let report = ops::run_upgrade(&store, &config, topic.as_deref(), dry_run).await?;
            for line in &report.preview_lines {
                println!("{line}");
            }
            if dry_run {
                println!("\nDry run: would enrich {} memories, create {} concepts, {} links across {} topics",
                    report.enriched, report.concepts, report.links, report.topics_processed);
            } else {
                if report.deprecated > 0 {
                    println!("Deprecated {} low-quality memories", report.deprecated);
                }
                println!("Upgrade complete: {} memories enriched, {} memoirs created, {} concepts, {} links",
                    report.enriched, report.memoirs, report.concepts, report.links);
            }
        }
        Some(Commands::Warmup) => {
            let store = config.open_store()?;
            let (cached, errors) = search::warmup::warmup(&store, &config).await;
            println!("Warmup complete: {cached} embeddings cached, {errors} errors");
        }
        Some(Commands::Worker { action }) => match action {
            WorkerAction::Memory => {
                let processed = extract::hooks::queue::drain_memory_queue(&config).await?;
                if processed > 0 {
                    eprintln!("rein worker: processed {processed} memory jobs");
                }
            }
            WorkerAction::DedupQueue => {
                let processed = extract::hooks::queue::drain_dedup_queue(&config).await?;
                if processed > 0 {
                    eprintln!("rein worker: processed {processed} dedup jobs");
                }
            }
            WorkerAction::CleanupQueue => {
                let processed = extract::hooks::queue::drain_cleanup_queue(&config).await?;
                if processed > 0 {
                    eprintln!("rein worker: processed {processed} cleanup jobs");
                }
            }
            WorkerAction::Cleanup {
                topic,
                topics,
                pattern,
                all,
                exact_topics,
                dry_run,
            } => {
                let selected_topics = topics.unwrap_or_default();
                let scope_all =
                    all || (topic.is_none() && selected_topics.is_empty() && pattern.is_none());
                let store = config.open_store()?;
                let merge_variants = !exact_topics;
                let groups = ops::resolve_topic_groups(
                    &store,
                    topic.as_deref(),
                    &selected_topics,
                    pattern.as_deref(),
                    scope_all,
                    merge_variants,
                )?;
                if groups.is_empty() {
                    eprintln!("rein worker: no topics matched the selected scope");
                } else {
                    let report =
                        ops::run_cleanup_async(&store, &config, &groups, merge_variants, dry_run)
                            .await?;
                    eprintln!(
                        "rein worker: cleanup finished; groups={}, memories={}, dedup_removed={}/{}",
                        report.consolidation.groups_processed,
                        report.consolidation.memories_replaced,
                        report.duplicates_merged,
                        report.duplicates_found
                    );
                }
            }
        },
        None => {
            println!("rein v{}", env!("CARGO_PKG_VERSION"));
            println!("Run 'rein --help' for usage");
        }
        Some(Commands::Hook { action }) => match action {
            HookAction::Post => extract::hooks::hook_post(&config).await?,
            HookAction::Compact => extract::hooks::hook_compact(&config).await?,
            HookAction::Prompt => extract::hooks::hook_prompt(&config).await?,
            HookAction::Stop => extract::hooks::hook_stop(&config).await?,
        },
        Some(Commands::Migrate { from_qmd, reindex }) => {
            if reindex {
                let store = config.open_store()?;
                let report = store::migrate::reindex(&store, &config).await?;
                println!("{report}");
            } else {
                let qmd_path = from_qmd.map(std::path::PathBuf::from).unwrap_or_else(|| {
                    let home = std::env::var("HOME").unwrap_or_default();
                    std::path::PathBuf::from(home).join(".cache/qmd/index.sqlite")
                });
                let store = config.open_store()?;
                let embedder = embed::create_embedder(&config);
                let report =
                    store::migrate::migrate_from_qmd(&qmd_path, &store, &config, embedder.as_ref())
                        .await?;
                println!("{report}");
            }
        }
        Some(Commands::Init { dry_run, proxy }) => {
            rein::init::auto_configure(dry_run)?;
            if proxy {
                rein::init::proxy_configure(dry_run)?;
            }
        }
        Some(Commands::Consolidate {
            topic,
            summary,
            topics,
            pattern,
            all,
            merge_variants,
            dry_run,
        }) => {
            let store = config.open_store()?;
            let selected_topics = topics.unwrap_or_default();
            let groups = ops::resolve_topic_groups(
                &store,
                topic.as_deref(),
                &selected_topics,
                pattern.as_deref(),
                all,
                merge_variants,
            )?;

            if groups.is_empty() {
                if let Some(topic) = topic {
                    println!("No memories found in topic '{topic}'");
                } else if let Some(pattern) = pattern {
                    println!("No topics matched pattern '{pattern}'");
                } else {
                    println!("No topics matched the selected scope");
                }
            } else {
                let report = ops::run_consolidation_async(
                    &store,
                    &config,
                    &groups,
                    summary.as_deref(),
                    dry_run,
                )
                .await?;

                if dry_run {
                    println!(
                        "Dry run: {} groups, {} memories would be consolidated",
                        report.groups_processed, report.memories_replaced
                    );
                } else {
                    println!(
                        "Consolidated {} groups ({} memories)",
                        report.groups_processed, report.memories_replaced
                    );
                }

                for group in report.groups.iter().filter(|group| group.memory_count > 0) {
                    let sources = if group.source_topics.len() > 1 {
                        format!(" <= {}", group.source_topics.join(", "))
                    } else {
                        String::new()
                    };
                    if dry_run {
                        println!(
                            "- {}{} [{} memories]",
                            group.canonical_topic, sources, group.memory_count
                        );
                    } else if let Some(created_id) = &group.created_id {
                        println!(
                            "- {}{} [{} memories] -> {}",
                            group.canonical_topic, sources, group.memory_count, created_id
                        );
                    }
                }
            }
        }
        Some(Commands::Dedup {
            dry_run,
            merge_variants,
        }) => {
            let store = config.open_store()?;
            let threshold = config.search.dedup_similarity as f32;
            let (dups_found, dups_removed) =
                ops::run_dedup(&store, &config, threshold, dry_run, merge_variants)?;
            if dry_run {
                println!("Found {dups_found} duplicates (dry-run, nothing removed)");
            } else {
                println!("Removed {dups_removed} of {dups_found} duplicates");
            }
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
            let selected_topics = topics.unwrap_or_default();
            let scope_all =
                all || (topic.is_none() && selected_topics.is_empty() && pattern.is_none());
            if asynchronous {
                let job_id = extract::hooks::queue::queue_cleanup_job(
                    &config,
                    topic.clone(),
                    selected_topics,
                    pattern.clone(),
                    scope_all,
                    exact_topics,
                    dry_run,
                )?;
                extract::hooks::queue::spawn_cleanup_worker(&config);
                println!("Queued cleanup job {job_id}");
            } else {
                let store = config.open_store()?;
                let merge_variants = !exact_topics;
                let groups = ops::resolve_topic_groups(
                    &store,
                    topic.as_deref(),
                    &selected_topics,
                    pattern.as_deref(),
                    scope_all,
                    merge_variants,
                )?;
                if groups.is_empty() {
                    if let Some(topic) = topic {
                        println!("No memories found in topic '{topic}'");
                    } else if let Some(pattern) = pattern {
                        println!("No topics matched pattern '{pattern}'");
                    } else {
                        println!("No topics matched the selected scope");
                    }
                } else {
                    let report =
                        ops::run_cleanup_async(&store, &config, &groups, merge_variants, dry_run)
                            .await?;
                    if dry_run {
                        println!(
                            "Dry run: {} groups ({} memories) would be consolidated; found {} duplicates",
                            report.consolidation.groups_processed,
                            report.consolidation.memories_replaced,
                            report.duplicates_found
                        );
                    } else {
                        println!(
                            "Cleanup finished: {} groups consolidated ({} memories), removed {} of {} duplicates",
                            report.consolidation.groups_processed,
                            report.consolidation.memories_replaced,
                            report.duplicates_merged,
                            report.duplicates_found
                        );
                    }
                }
            }
        }
        Some(Commands::Canonicals { limit }) => {
            let store = config.open_store()?;
            let canonicals = store.list_canonical_memories(limit)?;
            if canonicals.is_empty() {
                println!("No canonical memories found");
            } else {
                for memory in canonicals {
                    println!(
                        "- {} [{}] support={} merges={} diversity={:.2} dedup_conf={:.2}",
                        memory.id,
                        memory.summary,
                        memory.support_count,
                        memory.merge_count,
                        memory.source_diversity,
                        memory.dedup_confidence,
                    );
                }
            }
        }
        Some(Commands::Evidence {
            canonical_id,
            limit,
        }) => {
            let store = config.open_store()?;
            let evidence = store.list_memory_evidence(&canonical_id, limit)?;
            if evidence.is_empty() {
                println!("No evidence found for canonical '{canonical_id}'");
            } else {
                for item in evidence {
                    println!(
                        "- {} [{}] {}\n{}",
                        item.id, item.source_topic, item.summary, item.content
                    );
                }
            }
        }
        Some(Commands::DedupLog { canonical, limit }) => {
            let store = config.open_store()?;
            let decisions = store.list_dedup_decisions(canonical.as_deref(), limit)?;
            if decisions.is_empty() {
                println!("No dedup decisions found");
            } else {
                for decision in decisions {
                    println!(
                        "- {} relation={} confidence={:.2} winner={:?} loser={:?} reason={}",
                        decision.id,
                        decision.relation,
                        decision.confidence,
                        decision.winner_id,
                        decision.loser_id,
                        decision.reason
                    );
                }
            }
        }
        Some(Commands::Dashboard) => {
            rein::service::print_dashboard(&config);
        }
        Some(Commands::Gui { action }) => match action {
            ServiceAction::On => {
                rein::service::start_service("gui", &["serve", "--gui"])?;
            }
            ServiceAction::Off => {
                rein::service::stop_service("gui")?;
            }
        },
        Some(Commands::Proxy { action }) => match action {
            ServiceAction::On => {
                rein::service::start_service("proxy", &["serve", "--proxy"])?;
            }
            ServiceAction::Off => {
                rein::service::stop_service("proxy")?;
            }
        },
    }
    Ok(())
}
