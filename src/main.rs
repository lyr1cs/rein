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
                concept_ids: vec![],
                status: types::MemoryStatus::default(),
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
            println!("Extract provider: {}", config.extract.provider);
            println!("Extract model: {}", match config.extract.provider.as_str() {
                "omlx" => &config.extract.omlx.model,
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
            if dry_run {
                // Simulate: apply decay inside a savepoint, count what would be pruned, then rollback
                store.conn().execute_batch("SAVEPOINT gc_preview")?;
                let decayed = store.apply_decay()?;
                let mut stmt = store.conn().prepare(
                    "SELECT COUNT(*) FROM memories WHERE layer = 'STM' AND strength < ?1
                     AND importance NOT IN ('critical', 'high')"
                )?;
                let count: u64 = stmt.query_row(rusqlite::params![threshold], |row| row.get(0))?;
                drop(stmt);
                store.conn().execute_batch("ROLLBACK TO gc_preview")?;
                store.conn().execute_batch("RELEASE gc_preview")?;
                println!("Would decay {decayed} and prune {count} weak STM memories (threshold: {threshold})");
            } else {
                // First apply decay to update strengths, then prune
                let decayed = store.apply_decay()?;
                let pruned = store.prune(threshold)?;
                println!("Decayed {decayed} memories, pruned {pruned} weak STM memories (threshold: {threshold})");
            }
        }
        Some(Commands::Organize) => {
            let store = config.open_store()?;
            let threshold = config.search.dedup_similarity as f32;
            let links = store.organize(threshold, 5)?;
            println!("Organized: created {links} new links between related memories");
        }
        Some(Commands::Upgrade { topic, dry_run }) => {
            let has_llm = extract::llm::create_extractor(&config).is_some();
            if !has_llm {
                eprintln!("rein: WARNING — no LLM configured. Upgrade will use local rules only.");
                eprintln!("  Topic classification and keyword extraction will be basic.");
                eprintln!("  Knowledge graph (concepts/links) requires LLM — set GEMINI_API_KEY for full upgrade.");
            }

            let store = config.open_store()?;

            // Get memories to process
            let topics = if let Some(ref t) = topic {
                vec![t.clone()]
            } else {
                store.list_topics()?
            };

            let mut total_concepts = 0usize;
            let mut total_links = 0usize;
            let mut total_memoirs = 0usize;
            let mut total_enriched = 0usize;

            for topic_name in &topics {
                let memories = store.get_by_topic(topic_name)?;
                if memories.is_empty() { continue; }

                // Combine all memories in this topic into one text block
                let combined: String = memories.iter()
                    .map(|m| format!("[{}] {}\n{}", m.topic, m.summary, m.content))
                    .collect::<Vec<_>>()
                    .join("\n---\n");

                println!("Processing topic '{}' ({} memories)...", topic_name, memories.len());

                // Run full LLM extraction
                let result = extract::llm::extract_full_with_fallback(&config, &combined).await;

                if has_llm {
                    // === LLM path: full enrichment + knowledge graph ===
                    if dry_run {
                        let enrichable = result.memories.len().min(memories.len());
                        println!("  → would enrich {} memories, create {} concepts, {} links",
                                 enrichable, result.concepts.len(), result.links.len());
                        for c in &result.concepts {
                            println!("    concept: [{}] {} ({})", c.memoir, c.name, c.concept_type);
                        }
                        for l in &result.links {
                            println!("    link: {} --{}-> {}", l.from, l.relation, l.to);
                        }
                        total_concepts += result.concepts.len();
                        total_links += result.links.len();
                        total_enriched += enrichable;
                    } else {
                        // Enrich old memories with LLM-generated metadata
                        for new_mem in &result.memories {
                            let best_match = memories.iter()
                                .max_by(|a, b| {
                                    let sim_a = extract::similarity(&a.content, &new_mem.content);
                                    let sim_b = extract::similarity(&b.content, &new_mem.content);
                                    sim_a.partial_cmp(&sim_b).unwrap_or(std::cmp::Ordering::Equal)
                                });
                            if let Some(old) = best_match {
                                let sim = extract::similarity(&old.content, &new_mem.content);
                                if sim > 0.3 {
                                    let mut enriched = old.clone();
                                    enriched.topic = new_mem.topic.clone();
                                    enriched.summary = new_mem.summary.clone();
                                    enriched.keywords = new_mem.keywords.clone();
                                    if let Ok(imp) = new_mem.importance.parse::<types::Importance>() {
                                        enriched.importance = imp;
                                        enriched.layer = imp.auto_layer();
                                        enriched.decay_lambda = config.decay.base_lambda * imp.decay_factor();
                                    }
                                    if store.update(&enriched).is_ok() {
                                        total_enriched += 1;
                                    }
                                }
                            }
                        }

                        // Store concepts + links in knowledge graph
                        let (mut tc, mut tl) = (0usize, 0usize);
                        if !result.concepts.is_empty() || !result.links.is_empty() {
                            match store.store_knowledge_units(&result.concepts, &result.links) {
                                Ok(report) => {
                                    total_memoirs += report.memoirs_created;
                                    tc = report.concepts_added + report.concepts_refined;
                                    tl = report.links_added;
                                    total_concepts += tc;
                                    total_links += tl;
                                }
                                Err(e) => println!("  → error: {e}"),
                            }
                        }
                        println!("  → {tc} concepts, {tl} links");
                    }
                } else {
                    // === No-LLM path: local rule-based enrichment ===
                    // Can't produce concepts/links, but can fix topic + extract basic keywords
                    for old in &memories {
                        if old.topic != "auto-extracted" { continue; } // already enriched

                        let lower = old.content.to_lowercase();
                        // Classify topic by keywords
                        let new_topic = if ["architecture", "design", "component", "架构", "设计"].iter().any(|k| lower.contains(k)) {
                            "architecture"
                        } else if ["decided", "chose", "选型", "决策", "tradeoff"].iter().any(|k| lower.contains(k)) {
                            "decision"
                        } else if ["bug", "fix", "error", "crash", "修复", "解决"].iter().any(|k| lower.contains(k)) {
                            "debug"
                        } else if ["deploy", "install", "config", "migrate", "部署", "安装", "迁移"].iter().any(|k| lower.contains(k)) {
                            "workflow"
                        } else {
                            "general"
                        };

                        // Extract basic keywords from content (top scoring words)
                        let keywords: Vec<String> = old.content
                            .split_whitespace()
                            .filter(|w| w.len() > 3)
                            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
                            .filter(|w| !w.is_empty() && !["the", "this", "that", "with", "from", "have", "been", "into", "will"].contains(&w.as_str()))
                            .take(5)
                            .collect();

                        if dry_run {
                            if new_topic != "auto-extracted" || !keywords.is_empty() {
                                println!("  → would reclassify '{}' → topic='{}', keywords={:?}",
                                         old.summary.chars().take(40).collect::<String>(), new_topic, keywords);
                                total_enriched += 1;
                            }
                        } else {
                            let mut enriched = old.clone();
                            enriched.topic = new_topic.to_string();
                            if !keywords.is_empty() {
                                enriched.keywords = keywords;
                            }
                            // Score-based importance upgrade
                            let score = extract::score_sentence(&old.content);
                            if score >= 4 {
                                enriched.importance = types::Importance::High;
                                enriched.layer = enriched.importance.auto_layer();
                                enriched.decay_lambda = config.decay.base_lambda * enriched.importance.decay_factor();
                            }
                            if store.update(&enriched).is_ok() {
                                total_enriched += 1;
                            }
                        }
                    }
                    if !dry_run {
                        println!("  → enriched {} memories (local rules, no concepts/links)", total_enriched);
                    }
                }
            }

            if dry_run {
                println!("\nDry run: would enrich {total_enriched} memories, create {total_concepts} concepts, {total_links} links across {} topics", topics.len());
            } else {
                println!("\nUpgrade complete: {total_enriched} memories enriched, {total_memoirs} memoirs created, {total_concepts} concepts, {total_links} links");
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
                concept_ids: vec![],
                status: types::MemoryStatus::default(),
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

