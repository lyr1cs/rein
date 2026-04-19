//! CLI command handlers extracted from main.rs.
//!
//! Each `handle_*` function corresponds to a `Commands` variant in main.rs.
//! The `Commands` enum and argument parsing remain in main.rs; only the handler
//! bodies live here.

use rein::config::ReinConfig;
use rein::embed;
use rein::extract;
use rein::mcp;
use rein::ops;
use rein::search;
use rein::store;
use rein::types;
use rein::types::MemoryStore;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Resolve the set of topic groups for cleanup / consolidate / worker-cleanup.
///
/// This deduplicates the pattern that appeared three times in the old main.rs
/// match block.
pub fn resolve_cleanup_scope(
    store: &store::SqliteStore,
    topic: Option<String>,
    topics: &[String],
    pattern: Option<&str>,
    all: bool,
    exact_topics: bool,
) -> anyhow::Result<Vec<ops::TopicGroup>> {
    let merge_variants = !exact_topics;
    let scope_all = all || (topic.is_none() && topics.is_empty() && pattern.is_none());
    let groups = ops::resolve_topic_groups(
        store,
        topic.as_deref(),
        topics,
        pattern,
        scope_all,
        merge_variants,
    )?;
    Ok(groups)
}

/// Print a human-readable message after a consolidation run.
pub fn print_consolidation_report(report: &ops::ConsolidateReport, dry_run: bool) {
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

    for group in report.groups.iter().filter(|g| g.memory_count > 0) {
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

/// Print a human-readable message after a cleanup run.
pub fn print_cleanup_report(report: &ops::CleanupReport, dry_run: bool) {
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

// ---------------------------------------------------------------------------
// Command handlers
// ---------------------------------------------------------------------------

pub async fn handle_serve(
    config: ReinConfig,
    compact: bool,
    sse: bool,
    proxy: bool,
    gui: bool,
) -> anyhow::Result<()> {
    let mut config = config;
    if compact {
        config.server.compact = true;
    }
    if gui {
        config.server.gui_enabled = true;
        config.server.sse_enabled = true;
    }
    if proxy {
        // Note: REIN_PROXY_ACTIVE env var is set externally by shell aliases (claudep, codexp),
        // not here. Setting it here would be UB since tokio workers are already running.
        rein::proxy::run_proxy(config).await?;
    } else if sse || gui {
        config.server.sse_enabled = true;
        mcp::server::run_http(config).await?;
    } else {
        mcp::server::run_stdio(config).await?;
    }
    Ok(())
}

pub fn handle_store(
    config: &ReinConfig,
    topic: String,
    content: String,
    importance: String,
    keywords: Option<Vec<String>>,
) -> anyhow::Result<()> {
    let store = config.open_store()?;
    let imp: types::Importance = importance.parse().map_err(|e: String| anyhow::anyhow!(e))?;
    let memory = ops::build_memory(
        config,
        topic,
        content.clone(),
        imp,
        keywords.unwrap_or_default(),
        types::Source::Manual,
    );
    let id = ops::store_memory(&store, config, memory)?;
    println!(
        "{}",
        mcp::compact::format_store_result(&id, config.server.compact)
    );
    Ok(())
}

pub async fn handle_ingest(
    config: &ReinConfig,
    content: Option<String>,
    file: Option<String>,
    json_file: Option<String>,
    asynchronous: bool,
    agent_label: Option<String>,
    subagent: bool,
) -> anyhow::Result<()> {
    let report = match (content, file, json_file) {
        (Some(text), None, None) => {
            if asynchronous {
                ops::queue_ingest_session_text(config, &text, agent_label.as_deref(), subagent)?
            } else {
                ops::ingest_session_text_report(config, &text, agent_label.as_deref(), subagent)
                    .await?
            }
        }
        (None, Some(path), None) => {
            let text = std::fs::read_to_string(path)?;
            if asynchronous {
                ops::queue_ingest_session_text(config, &text, agent_label.as_deref(), subagent)?
            } else {
                ops::ingest_session_text_report(config, &text, agent_label.as_deref(), subagent)
                    .await?
            }
        }
        (None, None, Some(path)) => {
            let raw = std::fs::read_to_string(path)?;
            let session: types::SessionIngest = serde_json::from_str(&raw)?;
            if asynchronous {
                ops::queue_ingest_session(config, &session, agent_label.as_deref(), subagent)?
            } else {
                ops::ingest_session_report(config, &session, agent_label.as_deref(), subagent)
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
    Ok(())
}

pub fn handle_recall(
    config: &ReinConfig,
    query: String,
    topic: Option<String>,
    keyword: Option<String>,
    limit: usize,
) -> anyhow::Result<()> {
    let store = config.open_store()?;
    let results = search::recall::recall(
        &store,
        config,
        &query,
        topic.as_deref(),
        keyword.as_deref(),
        limit,
    )?;
    println!(
        "{}",
        mcp::compact::format_recall_results(&results, config.server.compact)
    );
    Ok(())
}

pub fn handle_topics(config: &ReinConfig) -> anyhow::Result<()> {
    let store = config.open_store()?;
    let topics = store.list_topics()?;
    println!(
        "{}",
        mcp::compact::format_topics(&topics, config.server.compact)
    );
    Ok(())
}

// handle_stats / handle_health migrated to #[op] (see ops/handlers/diagnostics.rs).
// main.rs intercepts those subcommands before Cli::parse() and dispatches through
// OpsCliEntry inventory.

pub fn handle_forget(config: &ReinConfig, id: String) -> anyhow::Result<()> {
    let store = config.open_store()?;
    store.delete(&id)?;
    println!("Deleted memory: {id}");
    Ok(())
}

pub fn handle_update(
    config: &ReinConfig,
    id: String,
    content: String,
    importance: Option<String>,
) -> anyhow::Result<()> {
    let store = config.open_store()?;
    let mut mem = store.get(&id)?;
    mem.content = content.clone();
    mem.summary = content
        .chars()
        .take(rein::types::SUMMARY_MAX_CHARS)
        .collect();
    if let Some(imp_str) = importance {
        let imp: types::Importance = imp_str.parse().map_err(|e: String| anyhow::anyhow!(e))?;
        mem.importance = imp;
        mem.layer = imp.auto_layer();
        mem.decay_lambda = config.decay.base_lambda * imp.decay_factor();
    }
    mem.updated_at = chrono::Utc::now();
    store.update(&mem)?;
    println!("Updated memory: {id}");
    Ok(())
}

pub fn handle_recent(config: &ReinConfig, limit: usize) -> anyhow::Result<()> {
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
    Ok(())
}

pub fn handle_organize(config: &ReinConfig) -> anyhow::Result<()> {
    let store = config.open_store()?;
    // A1: prefer adaptive global threshold over static config default.
    let threshold = ops::effective_dedup_threshold(&store, config);
    let links = store.organize(threshold, 5)?;
    println!("Organized: created {links} new links between related memories");
    Ok(())
}

pub fn handle_dedup_concepts(config: &ReinConfig) -> anyhow::Result<()> {
    let store = config.open_store()?;
    let (groups, removed) = store.dedup_concepts()?;
    println!("Concept dedup: merged {groups} groups, removed {removed} duplicate concepts");
    Ok(())
}

pub fn handle_export(
    config: &ReinConfig,
    format: String,
    topic: Option<String>,
    output: Option<String>,
) -> anyhow::Result<()> {
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
    all_memories.sort_by_key(|m| std::cmp::Reverse(m.created_at));

    let content = match format.as_str() {
        "json" => serde_json::to_string_pretty(&all_memories)?,
        "csv" => {
            let mut lines = vec![
                "id,topic,summary,content,importance,keywords,strength,created_at".to_string(),
            ];
            for m in &all_memories {
                let kw = m.keywords.join(";");
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
            all_memories
                .sort_by(|a, b| a.topic.cmp(&b.topic).then(b.created_at.cmp(&a.created_at)));
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
    Ok(())
}

pub async fn handle_upgrade(
    config: &ReinConfig,
    topic: Option<String>,
    dry_run: bool,
) -> anyhow::Result<()> {
    if extract::llm::create_extractor(config).is_none() {
        eprintln!("rein: WARNING \u{2014} no LLM configured. Upgrade will use local rules only.");
    }
    let store = config.open_store()?;
    let report = ops::run_upgrade(&store, config, topic.as_deref(), dry_run).await?;
    for line in &report.preview_lines {
        println!("{line}");
    }
    if dry_run {
        println!(
            "\nDry run: would enrich {} memories, create {} concepts, {} links across {} topics",
            report.enriched, report.concepts, report.links, report.topics_processed
        );
    } else {
        if report.deprecated > 0 {
            println!("Deprecated {} low-quality memories", report.deprecated);
        }
        println!(
            "Upgrade complete: {} memories enriched, {} memoirs created, {} concepts, {} links",
            report.enriched, report.memoirs, report.concepts, report.links
        );
    }
    Ok(())
}

pub async fn handle_warmup(config: &ReinConfig) -> anyhow::Result<()> {
    let store = config.open_store()?;
    let (cached, errors) = search::warmup::warmup(&store, config).await;
    println!("Warmup complete: {cached} embeddings cached, {errors} errors");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub async fn handle_consolidate(
    config: &ReinConfig,
    topic: Option<String>,
    summary: Option<String>,
    topics: Option<Vec<String>>,
    pattern: Option<String>,
    all: bool,
    merge_variants: bool,
    dry_run: bool,
) -> anyhow::Result<()> {
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
        let report =
            ops::run_consolidation_async(&store, config, &groups, summary.as_deref(), dry_run)
                .await?;
        print_consolidation_report(&report, dry_run);
    }
    Ok(())
}

pub fn handle_dedup(
    config: &ReinConfig,
    dry_run: bool,
    merge_variants: bool,
) -> anyhow::Result<()> {
    let store = config.open_store()?;
    // A1: adaptive threshold drives the dedup run; config remains last-resort fallback.
    let threshold = ops::effective_dedup_threshold(&store, config);
    let (dups_found, dups_removed) =
        ops::run_dedup(&store, config, threshold, dry_run, merge_variants)?;
    if dry_run {
        println!("Found {dups_found} duplicates (dry-run, nothing removed)");
    } else {
        println!("Removed {dups_removed} of {dups_found} duplicates");
    }
    Ok(())
}


#[allow(clippy::too_many_arguments)]
pub async fn handle_cleanup(
    config: &ReinConfig,
    topic: Option<String>,
    topics: Option<Vec<String>>,
    pattern: Option<String>,
    all: bool,
    exact_topics: bool,
    dry_run: bool,
    asynchronous: bool,
) -> anyhow::Result<()> {
    let selected_topics = topics.unwrap_or_default();
    let scope_all = all || (topic.is_none() && selected_topics.is_empty() && pattern.is_none());
    if asynchronous {
        let job_id = extract::hooks::queue::queue_cleanup_job(
            config,
            topic.clone(),
            selected_topics,
            pattern.clone(),
            scope_all,
            exact_topics,
            dry_run,
        )?;
        extract::hooks::queue::spawn_cleanup_worker(config);
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
                ops::run_cleanup_async(&store, config, &groups, merge_variants, dry_run).await?;
            print_cleanup_report(&report, dry_run);
        }
    }
    Ok(())
}

pub fn handle_dedup_log(
    config: &ReinConfig,
    canonical: Option<String>,
    limit: usize,
) -> anyhow::Result<()> {
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
    Ok(())
}

pub async fn handle_migrate(
    config: &ReinConfig,
    from_qmd: Option<String>,
    reindex: bool,
) -> anyhow::Result<()> {
    if reindex {
        let store = config.open_store()?;
        let report = store::migrate::reindex(&store, config).await?;
        println!("{report}");
    } else {
        let qmd_path = from_qmd.map(std::path::PathBuf::from).unwrap_or_else(|| {
            let home = std::env::var("HOME").unwrap_or_default();
            std::path::PathBuf::from(home).join(".cache/qmd/index.sqlite")
        });
        let store = config.open_store()?;
        let embedder = embed::create_embedder(config);
        let report =
            store::migrate::migrate_from_qmd(&qmd_path, &store, config, embedder.as_ref()).await?;
        println!("{report}");
    }
    Ok(())
}

pub fn handle_init(dry_run: bool, proxy: bool) -> anyhow::Result<()> {
    rein::init::auto_configure(dry_run)?;
    if proxy {
        rein::init::proxy_configure(dry_run)?;
    }
    Ok(())
}

pub async fn handle_worker_memory(config: &ReinConfig) -> anyhow::Result<()> {
    let processed = extract::hooks::queue::drain_memory_queue(config).await?;
    if processed > 0 {
        eprintln!("rein worker: processed {processed} memory jobs");
    }
    Ok(())
}

pub async fn handle_worker_dedup_queue(config: &ReinConfig) -> anyhow::Result<()> {
    let processed = extract::hooks::queue::drain_dedup_queue(config).await?;
    if processed > 0 {
        eprintln!("rein worker: processed {processed} dedup jobs");
    }
    Ok(())
}

pub async fn handle_worker_cleanup_queue(config: &ReinConfig) -> anyhow::Result<()> {
    let processed = extract::hooks::queue::drain_cleanup_queue(config).await?;
    if processed > 0 {
        eprintln!("rein worker: processed {processed} cleanup jobs");
    }
    Ok(())
}

pub async fn handle_worker_merge_refinement_queue(config: &ReinConfig) -> anyhow::Result<()> {
    let processed = extract::hooks::queue::drain_merge_refinement_queue(config).await?;
    if processed > 0 {
        eprintln!("rein worker: processed {processed} merge-refinement jobs");
    }
    Ok(())
}

pub async fn handle_worker_cleanup(
    config: &ReinConfig,
    topic: Option<String>,
    topics: Option<Vec<String>>,
    pattern: Option<String>,
    all: bool,
    exact_topics: bool,
    dry_run: bool,
) -> anyhow::Result<()> {
    let selected_topics = topics.unwrap_or_default();
    let merge_variants = !exact_topics;
    let store = config.open_store()?;
    let groups = resolve_cleanup_scope(
        &store,
        topic,
        &selected_topics,
        pattern.as_deref(),
        all,
        exact_topics,
    )?;
    if groups.is_empty() {
        eprintln!("rein worker: no topics matched the selected scope");
    } else {
        let report =
            ops::run_cleanup_async(&store, config, &groups, merge_variants, dry_run).await?;
        eprintln!(
            "rein worker: cleanup finished; groups={}, memories={}, dedup_removed={}/{}",
            report.consolidation.groups_processed,
            report.consolidation.memories_replaced,
            report.duplicates_merged,
            report.duplicates_found
        );
    }
    Ok(())
}

pub async fn handle_hook(config: &ReinConfig, action: &str) -> anyhow::Result<()> {
    match action {
        "post" => extract::hooks::hook_post(config).await?,
        "compact" => extract::hooks::hook_compact(config).await?,
        "prompt" => extract::hooks::hook_prompt(config).await?,
        "stop" => extract::hooks::hook_stop(config).await?,
        _ => unreachable!(),
    }
    Ok(())
}

pub fn handle_dashboard(config: &ReinConfig) {
    rein::service::print_dashboard(config);
}

pub fn handle_gui_on() -> anyhow::Result<()> {
    rein::service::start_service("gui", &["serve", "--gui"])?;
    Ok(())
}

pub fn handle_gui_off() -> anyhow::Result<()> {
    rein::service::stop_service("gui")?;
    Ok(())
}

pub fn handle_proxy_on() -> anyhow::Result<()> {
    rein::service::start_service("proxy", &["serve", "--proxy"])?;
    Ok(())
}

pub fn handle_proxy_off() -> anyhow::Result<()> {
    rein::service::stop_service("proxy")?;
    Ok(())
}
