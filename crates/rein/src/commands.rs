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

pub async fn handle_warmup(
    config: &ReinConfig,
    options: search::warmup::WarmupOptions,
) -> anyhow::Result<()> {
    let store = config.open_store()?;
    let report = search::warmup::warmup_with_options(&store, config, options).await;
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
    if report.revalidation_skipped > 0 {
        line.push_str(&format!(
            ", {} skipped at publish (changed during migration; next warmup fills them)",
            report.revalidation_skipped
        ));
    }
    println!("{line}");
    if report.provenance_unknown {
        println!(
            "Existing vector rows have no model provenance. If they were produced by the configured \
             embedding model, re-run `rein warmup --trust-existing-vectors`; if the model changed, run \
             `rein warmup --reembed-all` or `rein migrate --reindex`."
        );
    }
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

pub async fn handle_providers_status(
    config: &ReinConfig,
    probe: bool,
    json: bool,
) -> anyhow::Result<()> {
    let report = rein::ops::handlers::providers::status_report(config, probe).await;
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print!("{}", rein::ops::handlers::providers::render_human(&report));
    }
    Ok(())
}

pub async fn handle_keys_list(config: &ReinConfig, json: bool) -> anyhow::Result<()> {
    let report = rein::ops::handlers::providers::status_report(config, false).await;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "credentials": report.credentials,
                "secrets_file": rein::secrets::default_path(),
            }))?
        );
        return Ok(());
    }
    println!("secrets file: {}", rein::secrets::default_path().display());
    if report.credentials.is_empty() {
        println!("  (no surface needs a key with the current configuration)");
    }
    for c in &report.credentials {
        let name = c.id.split_once(':').map(|(_, n)| n).unwrap_or(&c.id);
        let value = c.masked.clone().unwrap_or_else(|| "missing".into());
        println!(
            "  {name:<24} {value:<16} {:<13} {}",
            c.kind,
            c.used_by.join(", ")
        );
        if let Some(w) = &c.warning {
            println!("    ! {w}");
        }
    }
    Ok(())
}

pub fn handle_keys_set(name: &str, value: Option<String>, stdin: bool) -> anyhow::Result<()> {
    use std::io::{IsTerminal, Read};
    let value = match (value, stdin) {
        (Some(v), _) => v,
        (None, true) => {
            let mut s = String::new();
            std::io::stdin().read_to_string(&mut s)?;
            s.lines().next().unwrap_or("").to_string()
        }
        (None, false) if std::io::stdin().is_terminal() => {
            rpassword::prompt_password(format!("value for {name} (hidden): "))?
        }
        (None, false) => anyhow::bail!(
            "no value given and stdin is not a terminal: pass --stdin (recommended) or --value"
        ),
    };
    let path = rein::secrets::default_path();
    rein::secrets::set_key(&path, name, &value).map_err(|e| anyhow::anyhow!(e))?;
    let masked = rein::ops::handlers::providers::mask_secret(value.trim());
    println!("stored {name} = {masked} in {}", path.display());
    if std::env::var(name)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .is_some()
    {
        println!("note: {name} is also set in this process environment, which takes precedence");
    }
    // Reporting which surfaces use the key needs the config; tolerate a
    // config that does not load (the key is already stored).
    match ReinConfig::load() {
        Ok(config) => {
            let report = rein::ops::handlers::providers::collect(&config, None);
            let used: Vec<String> = report
                .credentials
                .iter()
                .find(|c| c.id == format!("env:{name}"))
                .map(|c| c.used_by.clone())
                .unwrap_or_default();
            if used.is_empty() {
                println!(
                    "no surface currently uses {name} (configure one with `rein providers set …`)"
                );
            } else {
                println!("used by: {}", used.join(", "));
                println!("verify with: rein providers status --probe");
            }
        }
        Err(e) => println!("note: config did not load ({e}); run `rein doctor`"),
    }
    Ok(())
}

pub fn handle_keys_unset(name: &str) -> anyhow::Result<()> {
    let path = rein::secrets::default_path();
    match rein::secrets::unset_key(&path, name).map_err(|e| anyhow::anyhow!(e))? {
        true => println!("removed {name} from {}", path.display()),
        false => println!("{name} was not set in {}", path.display()),
    }
    Ok(())
}

pub async fn handle_keys_probe(
    config: &ReinConfig,
    name: Option<&str>,
    json: bool,
) -> anyhow::Result<()> {
    let mut report = rein::ops::handlers::providers::status_report(config, true).await;
    if let Some(n) = name {
        let id = format!("env:{n}");
        report.credentials.retain(|c| c.id == id);
        if report.credentials.is_empty() {
            anyhow::bail!("no surface uses {n}");
        }
    }
    if json {
        println!("{}", serde_json::to_string_pretty(&report.credentials)?);
    } else {
        print!("{}", rein::ops::handlers::providers::render_human(&report));
    }
    Ok(())
}

pub fn handle_providers_set(
    req: rein::config_edit::SetProviderRequest,
    dry_run: bool,
    json: bool,
) -> anyhow::Result<()> {
    let path = rein::config::config_file_path();
    let current = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => anyhow::bail!("{}: {e}", path.display()),
    };
    let outcome =
        rein::config_edit::apply_set_provider(&current, &req).map_err(|e| anyhow::anyhow!(e))?;
    if dry_run {
        if json {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "path": path,
                    "diff": outcome.diff,
                    "hints": outcome.hints,
                    "toml": outcome.text,
                }))?
            );
        } else {
            println!("would write {}:\n\n{}", path.display(), outcome.text);
            for h in &outcome.hints {
                println!("next: {h}");
            }
        }
        return Ok(());
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    rein::init::atomic_write_string(&path, &outcome.text)?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "path": path,
                "diff": outcome.diff,
                "hints": outcome.hints,
            }))?
        );
    } else {
        println!("updated {}", path.display());
        for d in &outcome.diff {
            println!(
                "  {}: {} → {}",
                d.key,
                d.old.as_deref().unwrap_or("(unset)"),
                d.new
            );
        }
        if outcome.diff.is_empty() {
            println!("  (no changes)");
        }
        for h in &outcome.hints {
            println!("next: {h}");
        }
    }
    Ok(())
}

pub async fn handle_providers_show(
    config: &ReinConfig,
    surface: &str,
    json: bool,
) -> anyhow::Result<()> {
    let name = match surface {
        "embedding" | "extract" | "query_expansion" | "resummerize" | "supermemory" => {
            surface.to_string()
        }
        "async_memory" => "extract.async_memory".into(),
        "intelligent_merge" => "extract.intelligent_merge".into(),
        "reranker" => "search.llm_reranker".into(),
        "judge" => "ars.llm_judge".into(),
        "nightly" => "ars.llm_judge.nightly_cron".into(),
        "synthesis" => "ars.recall_synthesis".into(),
        "concept_summary" => "ars.concept_summary".into(),
        "cold_archive" => "ars.cold_archive".into(),
        other
            if rein::ops::handlers::providers::SURFACES
                .iter()
                .any(|s| s.name == other) =>
        {
            other.to_string()
        }
        other => anyhow::bail!("unknown surface '{other}'"),
    };
    let report = rein::ops::handlers::providers::status_report(config, false).await;
    let Some(s) = report.surfaces.iter().find(|s| s.surface == name) else {
        anyhow::bail!("unknown surface '{surface}'");
    };
    let cred = s
        .credential
        .as_ref()
        .and_then(|id| report.credentials.iter().find(|c| &c.id == id));
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({ "surface": s, "credential": cred }))?
        );
        return Ok(());
    }
    let pm = if s.model.is_empty() {
        s.provider.clone()
    } else {
        format!("{}:{}", s.provider, s.model)
    };
    println!("{:<12} {}", "surface", s.surface);
    println!("{:<12} {}", "role", s.role);
    println!("{:<12} {}", "status", s.status);
    println!("{:<12} {pm}", "provider");
    println!("{:<12} {}", "endpoint", s.endpoint);
    println!("{:<12} {} ({})", "source", s.source_table, s.source_level);
    match cred {
        Some(c) => println!(
            "{:<12} {} = {} ({})",
            "key",
            c.id.trim_start_matches("env:"),
            c.masked.clone().unwrap_or_else(|| "missing".into()),
            c.kind
        ),
        None => println!("{:<12} none needed", "key"),
    }
    Ok(())
}

pub fn handle_providers_vendors(json: bool) -> anyhow::Result<()> {
    let rows: Vec<serde_json::Value> = rein::vendors::VENDOR_PRESETS
        .iter()
        .map(|p| {
            serde_json::json!({
                "name": p.name,
                "chat_endpoint": p.chat_endpoint,
                "embeddings_endpoint": p.embeddings_endpoint,
                "api_key_env": p.api_key_env,
                "auth_header": p.auth_header.config_name(),
                "chat": p.chat,
                "embeddings": p.embeddings,
                "encoding": p.encoding.config_name(),
                "max_batch": p.max_batch,
            })
        })
        .collect();
    if json {
        println!("{}", serde_json::to_string_pretty(&rows)?);
        return Ok(());
    }
    println!("vendor       chat emb  key env                default endpoint");
    for p in rein::vendors::VENDOR_PRESETS {
        println!(
            "{:<12} {:<4} {:<4} {:<22} {}",
            p.name,
            if p.chat { "yes" } else { "-" },
            if p.embeddings { "yes" } else { "-" },
            p.api_key_env.unwrap_or("(none)"),
            p.chat_endpoint
                .or(p.embeddings_endpoint)
                .unwrap_or("(set --endpoint)")
        );
    }
    println!("\nuse: rein providers set <surface> --provider <vendor> --model <model>   then   rein keys set <KEY_ENV>");
    Ok(())
}
