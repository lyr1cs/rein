//! CLI command handlers extracted from main.rs.
//!
//! Each `handle_*` function corresponds to a `Commands` variant in main.rs.
//! The `Commands` enum and argument parsing remain in main.rs; only the handler
//! bodies live here.

use rein::config::ReinConfig;
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

// print_consolidation_report removed — rein consolidate migrated to #[op] inventory.
// The formatting logic now lives in ConsolidateOutput::to_cli_text() (ops/handlers/maintenance.rs).

// print_cleanup_report removed — rein cleanup migrated to #[op] inventory.
// The formatting logic now lives in CleanupOutput::to_cli_text() (ops/handlers/maintenance.rs).

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

// handle_stats / handle_health migrated to #[op] (see ops/handlers/diagnostics.rs).
// main.rs intercepts those subcommands before Cli::parse() and dispatches through
// OpsCliEntry inventory.

// handle_organize removed — rein organize migrated to #[op] inventory.
// handle_dedup_concepts removed — rein dedup-concepts migrated to #[op] inventory.

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
    let report = search::warmup::warmup(&store, config).await;
    let mut line = format!(
        "Warmup complete: {} memories embedded, {} vector rows restored from cache, {} errors",
        report.embedded, report.backfilled_from_cache, report.errors
    );
    if report.skipped_no_provider > 0 {
        line.push_str(&format!(
            ", {} skipped (no embedding provider configured)",
            report.skipped_no_provider
        ));
    }
    println!("{line}");
    Ok(())
}

// handle_consolidate removed — rein consolidate migrated to #[op] inventory.
// See ops/handlers/maintenance.rs for the new implementation.

// handle_dedup removed — rein dedup migrated to #[op] inventory.
// See ops/handlers/maintenance.rs for the new implementation.

// handle_cleanup removed — rein cleanup migrated to #[op] inventory.
// The handler now lives in OpsRuntime::cleanup() (ops/handlers/maintenance.rs).

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
        "session-start" => extract::hooks::hook_session_start(config).await?,
        "pre" => extract::hooks::hook_pre_tool_use(config).await?,
        "permission" => extract::hooks::hook_permission_request(config).await?,
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
