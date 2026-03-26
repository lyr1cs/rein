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
#[command(name = "rein", version, about = "Multi-source cross-validated memory for AI agents")]
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
        /// Enable SSE transport (not yet implemented)
        #[arg(long)]
        sse: bool,
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
    Forget {
        id: String,
    },
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
    Health {
        topic: Option<String>,
    },
    /// Consolidate a topic into a single memory
    Consolidate {
        topic: String,
        #[arg(short, long)]
        summary: String,
    },
    /// Scan for duplicates
    Dedup {
        #[arg(long)]
        dry_run: bool,
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
    },
    /// Show most recently created memories
    Recent {
        #[arg(short, long, default_value = "10")]
        limit: usize,
    },
    /// Garbage collect weak/stale STM memories below strength threshold
    Gc {
        #[arg(long)]
        dry_run: bool,
    },
    /// Auto-link related memories based on content similarity
    Organize,
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
    /// Hook commands
    Hook {
        #[command(subcommand)]
        action: HookAction,
    },
}

#[derive(Subcommand)]
enum HookAction {
    /// Extract facts from tool output (PostToolUse)
    Post,
    /// Extract context before compaction (PreCompact)
    Compact,
    /// Inject recalled memories into prompt (UserPromptSubmit)
    Prompt,
    /// Save session summary on conversation end (Stop)
    Stop,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("REIN_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    let cli = Cli::parse();
    let config = config::ReinConfig::load()?;
    config.validate();

    match cli.command {
        Some(Commands::Serve { compact, sse }) => {
            let mut config = config;
            if compact {
                config.server.compact = true;
            }
            if sse {
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
            let imp: types::Importance = importance
                .parse()
                .map_err(|e: String| anyhow::anyhow!(e))?;
            let memory = rein::ops::build_memory(
                &config,
                topic,
                content.clone(),
                imp,
                keywords.unwrap_or_default(),
                types::Source::Manual,
            );
            let id = store
                .store_with_dedup(
                    memory,
                    config.search.dedup_similarity as f32,
                    config.search.dedup_time_window_days,
                )?;
            println!(
                "{}",
                mcp::compact::format_store_result(&id, config.server.compact)
            );
        }
        Some(Commands::Recall {
            query,
            topic,
            keyword,
            limit,
        }) => {
            let store = config.open_store()?;
            let results = search::recall::recall(
                &store, &config, &query,
                topic.as_deref(), keyword.as_deref(), limit,
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
                let imp: types::Importance = imp_str
                    .parse()
                    .map_err(|e: String| anyhow::anyhow!(e))?;
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
            println!("Extract model: {}", match config.extract_provider() {
                config::Provider::Omlx => &config.extract.omlx.model,
                _ => &config.extract.google.model,
            });
            println!("Compact mode: {}", config.server.compact);
            println!("SSE enabled: {}", config.server.sse_enabled);
            println!("Decay base_lambda: {}", config.decay.base_lambda);
            println!("Dedup similarity: {}", config.search.dedup_similarity);
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
                    println!("[{}] {} ({}, {})", m.topic, m.summary, m.importance, age_str);
                }
            }
        }
        Some(Commands::Gc { dry_run }) => {
            let store = config.open_store()?;
            let threshold = config.decay.prune_threshold;
            let (decayed, pruned, concepts) = ops::run_gc_adaptive(&store, &config, threshold, dry_run)?;
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
        Some(Commands::Export { format, topic, output }) => {
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
            all_memories.sort_by(|a, b| b.created_at.cmp(&a.created_at));

            let content = match format.as_str() {
                "json" => serde_json::to_string_pretty(&all_memories)?,
                "csv" => {
                    let mut lines = vec!["id,topic,summary,content,importance,keywords,strength,created_at".to_string()];
                    for m in &all_memories {
                        let kw = m.keywords.join(";");
                        // Escape CSV fields
                        let esc = |s: &str| format!("\"{}\"", s.replace('"', "\"\""));
                        lines.push(format!("{},{},{},{},{},{},{:.3},{}",
                            m.id, esc(&m.topic), esc(&m.summary), esc(&m.content),
                            m.importance, esc(&kw), m.strength, m.created_at.to_rfc3339(),
                        ));
                    }
                    lines.join("\n")
                }
                "md" => {
                    let mut parts = vec![format!("# rein Memory Export\n\n{} memories, {} topics\n", all_memories.len(), topics.len())];
                    let mut current_topic = String::new();
                    // Group by topic
                    all_memories.sort_by(|a, b| a.topic.cmp(&b.topic).then(b.created_at.cmp(&a.created_at)));
                    for m in &all_memories {
                        if m.topic != current_topic {
                            current_topic = m.topic.clone();
                            parts.push(format!("\n## {}\n", current_topic));
                        }
                        parts.push(format!("### {} ({})\n{}\n", m.summary, m.importance, m.content));
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
        None => {
            println!("rein v{}", env!("CARGO_PKG_VERSION"));
            println!("Run 'rein --help' for usage");
        }
        Some(Commands::Hook { action }) => {
            match action {
                HookAction::Post => extract::hooks::hook_post(&config).await?,
                HookAction::Compact => extract::hooks::hook_compact(&config).await?,
                HookAction::Prompt => extract::hooks::hook_prompt(&config).await?,
                HookAction::Stop => extract::hooks::hook_stop(&config).await?,
            }
        }
        Some(Commands::Migrate { from_qmd, reindex }) => {
            if reindex {
                let store = config.open_store()?;
                let report = store::migrate::reindex(&store, &config).await?;
                println!("{report}");
            } else {
                let qmd_path = from_qmd
                    .map(std::path::PathBuf::from)
                    .unwrap_or_else(|| {
                        let home = std::env::var("HOME").unwrap_or_default();
                        std::path::PathBuf::from(home).join(".cache/qmd/index.sqlite")
                    });
                let store = config.open_store()?;
                let embedder = embed::create_embedder(&config);
                let report = store::migrate::migrate_from_qmd(
                    &qmd_path,
                    &store,
                    &config,
                    embedder.as_ref(),
                )
                .await?;
                println!("{report}");
            }
        }
        Some(Commands::Init { dry_run }) => {
            rein::init::auto_configure(dry_run)?;
        }
        Some(Commands::Consolidate { topic, summary }) => {
            let store = config.open_store()?;
            let consolidated = ops::build_consolidated(&config, topic.clone(), summary, vec![]);
            let old = store.consolidate_atomic(&topic, consolidated)?;
            if old.is_empty() {
                println!("No memories found in topic '{topic}'");
            } else {
                println!("Consolidated {} memories in topic '{topic}'", old.len());
            }
        }
        Some(Commands::Dedup { dry_run }) => {
            let store = config.open_store()?;
            let threshold = config.search.dedup_similarity as f32;
            let (dups_found, dups_removed) = ops::run_dedup(&store, threshold, dry_run)?;
            if dry_run {
                println!("Found {dups_found} duplicates (dry-run, nothing removed)");
            } else {
                println!("Removed {dups_removed} of {dups_found} duplicates");
            }
        }
    }
    Ok(())
}

