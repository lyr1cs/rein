mod config;
mod embed;
mod extract;
mod mcp;
mod search;
mod store;
mod types;

use clap::{Parser, Subcommand};
use types::MemoryStore;

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
    },
    /// Auto-configure MCP clients
    Init {
        #[arg(long)]
        dry_run: bool,
    },
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
    /// Post-conversation hook
    Post,
    /// Compact output hook
    Compact,
    /// Prompt generation hook
    Prompt,
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

    match cli.command {
        Some(Commands::Serve { compact, sse }) => {
            let mut config = config;
            if compact {
                config.server.compact = true;
            }
            if sse {
                config.server.sse_enabled = true;
            }
            mcp::server::run_stdio(config).await?;
        }
        Some(Commands::Store {
            topic,
            content,
            importance,
            keywords,
        }) => {
            let store = store::SqliteStore::new(&config.resolve_db_path())?;
            let imp: types::Importance = importance
                .parse()
                .map_err(|e: String| anyhow::anyhow!(e))?;
            let memory = types::Memory {
                id: ulid::Ulid::new().to_string(),
                layer: imp.auto_layer(),
                topic,
                summary: content.chars().take(100).collect(),
                content: content.clone(),
                keywords: keywords.unwrap_or_default(),
                importance: imp,
                source: types::Source::Manual,
                strength: 1.0,
                decay_lambda: config.decay.base_lambda * imp.decay_factor(),
                access_count: 0,
                superseded_by: None,
                related_ids: vec![],
                embedding: None,
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                last_accessed: chrono::Utc::now(),
            };
            let id = store
                .store_with_dedup(
                    memory,
                    config.search.dedup_similarity as f32,
                    config.search.dedup_time_window_days,
                )
                .await?;
            println!(
                "{}",
                mcp::compact::format_store_result(&id, config.server.compact)
            );
        }
        Some(Commands::Recall {
            query,
            topic,
            keyword: _,
            limit,
        }) => {
            let store = store::SqliteStore::new(&config.resolve_db_path())?;
            let results = store.search_fts(&query, topic.as_deref(), limit).await?;
            let scored: Vec<(types::Memory, f32)> =
                results.into_iter().map(|m| (m, 1.0)).collect();
            println!(
                "{}",
                mcp::compact::format_recall_results(&scored, config.server.compact)
            );
        }
        Some(Commands::Topics) => {
            let store = store::SqliteStore::new(&config.resolve_db_path())?;
            let topics = store.list_topics().await?;
            println!(
                "{}",
                mcp::compact::format_topics(&topics, config.server.compact)
            );
        }
        Some(Commands::Stats) => {
            let store = store::SqliteStore::new(&config.resolve_db_path())?;
            let stats = store.stats().await?;
            println!(
                "{}",
                mcp::compact::format_stats(&stats, config.server.compact)
            );
        }
        Some(Commands::Health { topic }) => {
            let store = store::SqliteStore::new(&config.resolve_db_path())?;
            let reports = store.health(topic.as_deref()).await?;
            println!(
                "{}",
                mcp::compact::format_health(&reports, config.server.compact)
            );
        }
        Some(Commands::Forget { id }) => {
            let store = store::SqliteStore::new(&config.resolve_db_path())?;
            store.delete(&id).await?;
            println!("Deleted memory: {id}");
        }
        Some(Commands::Update {
            id,
            content,
            importance,
        }) => {
            let store = store::SqliteStore::new(&config.resolve_db_path())?;
            let mut mem = store.get(&id).await?;
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
            store.update(&mem).await?;
            println!("Updated memory: {id}");
        }
        Some(Commands::Config) => {
            println!("Database path: {}", config.resolve_db_path().display());
            println!("Embedding provider: {}", config.embedding.provider);
            println!("Embedding dimensions: {}", config.embedding.dimensions);
            println!("Compact mode: {}", config.server.compact);
            println!("SSE enabled: {}", config.server.sse_enabled);
            println!("Decay base_lambda: {}", config.decay.base_lambda);
            println!("Dedup similarity: {}", config.search.dedup_similarity);
        }
        None => {
            println!("rein v{}", env!("CARGO_PKG_VERSION"));
            println!("Run 'rein --help' for usage");
        }
        _ => {
            println!("Not yet implemented");
        }
    }
    Ok(())
}
