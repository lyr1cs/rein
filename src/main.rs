use clap::{Parser, Subcommand};

use rein::config;
use rein::embed;
use rein::extract;
use rein::mcp;
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
            println!("Compact mode: {}", config.server.compact);
            println!("SSE enabled: {}", config.server.sse_enabled);
            println!("Decay base_lambda: {}", config.decay.base_lambda);
            println!("Dedup similarity: {}", config.search.dedup_similarity);
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
            auto_configure(dry_run)?;
        }
        Some(Commands::Consolidate { topic, summary }) => {
            let store = config.open_store()?;
            let imp = types::Importance::High;
            let consolidated = types::Memory {
                id: ulid::Ulid::new().to_string(),
                layer: imp.auto_layer(),
                topic: topic.clone(),
                summary: summary.chars().take(100).collect(),
                content: summary,
                keywords: vec![],
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
            let topics = store.list_topics()?;
            let mut dups_found = 0u32;
            let mut dups_removed = 0u32;
            for topic in &topics {
                let mems = store.get_by_topic(topic)?;
                // Build set of IDs to delete (losers). Keep the last one (newest) as winner.
                let mut to_delete: std::collections::HashSet<String> = std::collections::HashSet::new();
                for i in 0..mems.len() {
                    if to_delete.contains(&mems[i].id) { continue; }
                    for j in (i + 1)..mems.len() {
                        if to_delete.contains(&mems[j].id) { continue; }
                        let sim = extract::similarity(&mems[i].content, &mems[j].content);
                        if sim >= threshold {
                            // Keep the newer one (higher index = later ULID = newer)
                            to_delete.insert(mems[i].id.clone());
                            dups_found += 1;
                            if dry_run {
                                println!("  dup: '{}' ~ '{}' (sim={sim:.2})", &mems[i].summary.chars().take(40).collect::<String>(), &mems[j].summary.chars().take(40).collect::<String>());
                            }
                            break; // mems[i] is already marked, move to next i
                        }
                    }
                }
                if !dry_run {
                    for id in &to_delete {
                        store.delete(id)?;
                        dups_removed += 1;
                    }
                }
            }
            if dry_run {
                println!("Found {dups_found} duplicates (dry-run, nothing removed)");
            } else {
                println!("Removed {dups_removed} of {dups_found} duplicates");
            }
        }
    }
    Ok(())
}

fn auto_configure(dry_run: bool) -> anyhow::Result<()> {
    let home = std::env::var("HOME").unwrap_or_default();

    let clients: Vec<(&str, String, &str)> = vec![
        ("Claude Code", format!("{home}/.claude.json"), "json"),
        (
            "Claude Desktop",
            format!("{home}/Library/Application Support/Claude/claude_desktop_config.json"),
            "json",
        ),
        ("Cursor", format!("{home}/.cursor/mcp.json"), "json"),
        (
            "Windsurf",
            format!("{home}/.codeium/windsurf/mcp_config.json"),
            "json",
        ),
        (
            "VS Code",
            format!("{home}/Library/Application Support/Code/User/mcp.json"),
            "json",
        ),
        ("Gemini", format!("{home}/.gemini/settings.json"), "json"),
        ("Codex", format!("{home}/.codex/config.toml"), "toml"),
        (
            "OpenCode",
            format!("{home}/.config/opencode/opencode.json"),
            "json",
        ),
    ];

    for (name, path, format) in &clients {
        let path = std::path::Path::new(path);
        if path.exists() {
            if dry_run {
                println!("[dry-run] Would configure {name} at {}", path.display());
            } else {
                match configure_client(path, format) {
                    Ok(()) => println!("Configured {name}"),
                    Err(e) => println!("Failed to configure {name}: {e}"),
                }
            }
        } else {
            println!("- {name}: not found");
        }
    }
    Ok(())
}

fn configure_client(path: &std::path::Path, format: &str) -> anyhow::Result<()> {
    match format {
        "json" => configure_json_client(path),
        "toml" => configure_toml_client(path),
        _ => anyhow::bail!("unsupported config format: {format}"),
    }
}

fn configure_json_client(path: &std::path::Path) -> anyhow::Result<()> {
    let content = std::fs::read_to_string(path).unwrap_or_else(|_| "{}".to_string());
    // Create a backup before modifying
    let backup = path.with_extension("json.bak");
    std::fs::copy(path, &backup).ok();
    let mut root: serde_json::Value = serde_json::from_str(&content)?;

    let servers = root
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("config is not a JSON object"))?
        .entry("mcpServers")
        .or_insert_with(|| serde_json::json!({}));

    let servers_obj = servers
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("mcpServers is not a JSON object"))?;

    if servers_obj.contains_key("rein") {
        println!("  (rein already configured, skipping)");
        return Ok(());
    }

    servers_obj.insert(
        "rein".to_string(),
        serde_json::json!({
            "command": "rein",
            "args": ["serve"]
        }),
    );

    let formatted = serde_json::to_string_pretty(&root)?;
    std::fs::write(path, formatted)?;
    Ok(())
}

fn configure_toml_client(path: &std::path::Path) -> anyhow::Result<()> {
    let content = std::fs::read_to_string(path).unwrap_or_else(|_| String::new());
    let mut root: toml::Value = if content.is_empty() {
        toml::Value::Table(toml::map::Map::new())
    } else {
        toml::from_str(&content)?
    };

    let root_tbl = root
        .as_table_mut()
        .ok_or_else(|| anyhow::anyhow!("config is not a TOML table"))?;

    // Ensure [mcp] section exists
    let mcp = root_tbl
        .entry("mcp")
        .or_insert_with(|| toml::Value::Table(toml::map::Map::new()));
    let mcp_tbl = mcp
        .as_table_mut()
        .ok_or_else(|| anyhow::anyhow!("[mcp] is not a table"))?;

    if mcp_tbl.contains_key("rein") {
        println!("  (rein already configured, skipping)");
        return Ok(());
    }

    let mut rein_tbl = toml::map::Map::new();
    rein_tbl.insert(
        "command".to_string(),
        toml::Value::String("rein".to_string()),
    );
    rein_tbl.insert(
        "args".to_string(),
        toml::Value::Array(vec![toml::Value::String("serve".to_string())]),
    );
    mcp_tbl.insert("rein".to_string(), toml::Value::Table(rein_tbl));

    let formatted = toml::to_string_pretty(&root)?;
    std::fs::write(path, formatted)?;
    Ok(())
}
