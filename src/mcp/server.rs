use std::sync::{
    atomic::{AtomicU32, Ordering},
    Mutex,
};

use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{ServerCapabilities, ServerInfo};
use rmcp::{ServerHandler, ServiceExt, tool, tool_handler, tool_router};

use crate::config::ReinConfig;
use crate::mcp::compact;
use crate::mcp::tools::*;
use crate::store::SqliteStore;
use crate::types::*;

/// MCP server for rein memory system.
///
/// Uses a std::sync::Mutex for the store because all SqliteStore operations
/// are actually synchronous (the async trait methods don't contain real await
/// points), so we never hold the mutex guard across an actual await.
pub struct ReinServer {
    store: Mutex<SqliteStore>,
    config: ReinConfig,
    non_store_count: AtomicU32,
    tool_router: ToolRouter<Self>,
}

// Safety: SqliteStore uses SQLITE_OPEN_FULL_MUTEX (serialized mode), which makes
// the SQLite connection thread-safe. The Mutex<SqliteStore> provides exclusive access,
// and all MemoryStore trait methods are synchronous (no real await points).
unsafe impl Send for ReinServer {}
unsafe impl Sync for ReinServer {}

impl std::fmt::Debug for ReinServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReinServer")
            .field("compact", &self.config.server.compact)
            .finish_non_exhaustive()
    }
}

impl ReinServer {
    /// Create a new ReinServer.
    pub fn new(store: SqliteStore, config: ReinConfig) -> Self {
        Self {
            store: Mutex::new(store),
            config,
            non_store_count: AtomicU32::new(0),
            tool_router: Self::tool_router(),
        }
    }

    /// Append a store nudge if the non-store counter exceeds the threshold.
    fn maybe_nudge(&self, text: &mut String) {
        let count = self.non_store_count.load(Ordering::Relaxed);
        if count >= 10 {
            text.push_str(&format!(
                "\n\n[rein: {} tool calls since last store. Consider saving important context.]",
                count
            ));
        }
    }

    fn compact(&self) -> bool {
        self.config.server.compact
    }

    /// Lock the store, run a synchronous closure, return the result.
    fn with_store<F, R>(&self, f: F) -> Result<R, String>
    where
        F: FnOnce(&SqliteStore) -> Result<R, ReinError>,
    {
        let store = self.store.lock().map_err(|e| format!("Lock poisoned: {e}"))?;
        f(&store).map_err(|e| format!("{e}"))
    }
}

#[tool_router]
impl ReinServer {
    /// Search memories by query, topic, or keyword.
    /// Uses full pipeline: FTS5 → cached vectors → Google API → RRF fusion → Ebbinghaus weighting → cross-validation.
    #[tool(name = "rein_recall", description = "Search and recall memories by semantic query. Uses three-level waterfall search with cross-validation. Supports optional topic and keyword filters.")]
    fn rein_recall(&self, Parameters(params): Parameters<RecallParams>) -> String {
        self.non_store_count.fetch_add(1, Ordering::Relaxed);
        let limit = params.limit.unwrap_or(10);

        let result = self.with_store(|store| {
            crate::search::recall::recall(
                store,
                &self.config,
                &params.query,
                params.topic.as_deref(),
                params.keyword.as_deref(),
                limit,
            ).map_err(|e| ReinError::Config(format!("{e}")))
        });

        match result {
            Ok(results) => {
                let scored: Vec<(Memory, f32)> = results.into_iter().map(|r| (r.memory, r.score)).collect();
                let mut text = compact::format_recall_results(&scored, self.compact());
                if text.is_empty() {
                    text = if self.compact() {
                        "none".to_string()
                    } else {
                        "No memories found.".to_string()
                    };
                }
                self.maybe_nudge(&mut text);
                text
            }
            Err(e) => {
                let mut text = format!("Error searching: {e}");
                self.maybe_nudge(&mut text);
                text
            }
        }
    }

    /// Store a new memory with topic, content, importance, and keywords.
    #[tool(name = "rein_store", description = "Store a new memory. Automatically deduplicates against existing memories.")]
    fn rein_store(&self, Parameters(params): Parameters<StoreParams>) -> String {
        self.non_store_count.store(0, Ordering::Relaxed);

        if params.content.len() > 100_000 {
            return "Error: content too large (max 100KB)".to_string();
        }

        let importance: Importance = params
            .importance
            .as_deref()
            .unwrap_or("medium")
            .parse()
            .unwrap_or(Importance::Medium);

        let keywords: Vec<String> = params
            .keywords
            .map(|k| {
                k.split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            })
            .unwrap_or_default();

        let content = params.content.clone();
        let memory = Memory {
            id: ulid::Ulid::new().to_string(),
            layer: importance.auto_layer(),
            topic: params.topic,
            summary: content.chars().take(100).collect(),
            content,
            keywords,
            importance,
            source: Source::Manual,
            strength: 1.0,
            decay_lambda: self.config.decay.base_lambda * importance.decay_factor(),
            access_count: 0,
            superseded_by: None,
            related_ids: vec![],
            embedding: None,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            last_accessed: chrono::Utc::now(),
        };

        let dedup_sim = self.config.search.dedup_similarity as f32;
        let dedup_days = self.config.search.dedup_time_window_days;

        let result = self.with_store(|store| {
            store.store_with_dedup(memory, dedup_sim, dedup_days)
        });

        match result {
            Ok(id) => compact::format_store_result(&id, self.compact()),
            Err(e) => format!("Error storing: {e}"),
        }
    }

    /// Update an existing memory by ID.
    #[tool(name = "rein_update", description = "Update the content of an existing memory by its ID.")]
    fn rein_update(&self, Parameters(params): Parameters<UpdateParams>) -> String {
        // Don't count update as non-store (it's a mutation, not a read)
        self.non_store_count.store(0, Ordering::Relaxed);

        let base_lambda = self.config.decay.base_lambda;
        let compact = self.compact();
        let id = params.id.clone();

        let result = self.with_store(|store| {
            let mut memory = store.get(&params.id)?;
            memory.content = params.content.clone();
            memory.summary = params.content.chars().take(100).collect();
            if let Some(ref imp_str) = params.importance {
                if let Ok(imp) = imp_str.parse::<Importance>() {
                    memory.importance = imp;
                    memory.layer = imp.auto_layer();
                    memory.decay_lambda = base_lambda * imp.decay_factor();
                }
            }
            memory.updated_at = chrono::Utc::now();
            store.update(&memory)
        });

        match result {
            Ok(()) => {
                let mut text = if compact {
                    format!("ok:{id}")
                } else {
                    format!("Updated memory: {id}")
                };
                self.maybe_nudge(&mut text);
                text
            }
            Err(e) => {
                let mut text = format!("Error updating: {e}");
                self.maybe_nudge(&mut text);
                text
            }
        }
    }

    /// Delete a memory by ID.
    #[tool(name = "rein_forget", description = "Delete a memory by its ID.")]
    fn rein_forget(&self, Parameters(params): Parameters<ForgetParams>) -> String {
        // Don't count forget as non-store (it's a mutation, not a read)
        self.non_store_count.store(0, Ordering::Relaxed);
        let compact = self.compact();
        let id = params.id.clone();

        let result = self.with_store(|store| store.delete(&params.id));

        match result {
            Ok(()) => {
                let mut text = if compact {
                    format!("ok:{id}")
                } else {
                    format!("Deleted memory: {id}")
                };
                self.maybe_nudge(&mut text);
                text
            }
            Err(e) => {
                let mut text = format!("Error: {e}");
                self.maybe_nudge(&mut text);
                text
            }
        }
    }

    /// List all topics in the memory store.
    #[tool(name = "rein_list_topics", description = "List all memory topics.")]
    fn rein_list_topics(&self) -> String {
        self.non_store_count.fetch_add(1, Ordering::Relaxed);

        let result = self.with_store(|store| store.list_topics());

        match result {
            Ok(topics) => {
                let mut text = compact::format_topics(&topics, self.compact());
                self.maybe_nudge(&mut text);
                text
            }
            Err(e) => {
                let mut text = format!("Error: {e}");
                self.maybe_nudge(&mut text);
                text
            }
        }
    }

    /// Show memory store statistics.
    #[tool(name = "rein_stats", description = "Show memory store statistics (total, LTM/STM counts, avg strength).")]
    fn rein_stats(&self) -> String {
        self.non_store_count.fetch_add(1, Ordering::Relaxed);

        let result = self.with_store(|store| store.stats());

        match result {
            Ok(stats) => {
                let mut text = compact::format_stats(&stats, self.compact());
                self.maybe_nudge(&mut text);
                text
            }
            Err(e) => {
                let mut text = format!("Error: {e}");
                self.maybe_nudge(&mut text);
                text
            }
        }
    }

    /// Check health of memory topics.
    #[tool(name = "rein_health", description = "Check health of memory topics. Shows stale count, avg strength, consolidation needs.")]
    fn rein_health(&self, Parameters(params): Parameters<HealthParams>) -> String {
        self.non_store_count.fetch_add(1, Ordering::Relaxed);

        let result = self.with_store(|store| {
            store.health(params.topic.as_deref())
        });

        match result {
            Ok(reports) => {
                let mut text = compact::format_health(&reports, self.compact());
                self.maybe_nudge(&mut text);
                text
            }
            Err(e) => {
                let mut text = format!("Error: {e}");
                self.maybe_nudge(&mut text);
                text
            }
        }
    }

    /// Consolidate all memories in a topic into a single summary.
    #[tool(name = "rein_consolidate", description = "Consolidate all memories in a topic into a single summary memory, removing the originals.")]
    fn rein_consolidate(&self, Parameters(params): Parameters<ConsolidateParams>) -> String {
        self.non_store_count.fetch_add(1, Ordering::Relaxed);
        let base_lambda = self.config.decay.base_lambda;
        let compact = self.compact();

        let result = self.with_store(|store| {
            // Check if topic has any memories first
            let existing = store.get_by_topic(&params.topic)?;
            if existing.is_empty() {
                return Ok((0, String::new()));
            }

            let new_id = ulid::Ulid::new().to_string();
            let consolidated = Memory {
                id: new_id.clone(),
                layer: MemoryLayer::LTM,
                topic: params.topic.clone(),
                summary: params.summary.chars().take(100).collect(),
                content: params.summary.clone(),
                keywords: vec![],
                importance: Importance::High,
                source: Source::Manual,
                strength: 1.0,
                decay_lambda: base_lambda * Importance::High.decay_factor(),
                access_count: 0,
                superseded_by: None,
                related_ids: existing.iter().map(|m| m.id.clone()).collect(),
                embedding: None,
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                last_accessed: chrono::Utc::now(),
            };

            let old = store.consolidate_atomic(&params.topic, consolidated)?;
            Ok((old.len(), new_id))
        });

        match result {
            Ok((0, _)) => {
                let mut text = if compact {
                    "err:no_memories".to_string()
                } else {
                    format!("No memories found for topic: {}", params.topic)
                };
                self.maybe_nudge(&mut text);
                text
            }
            Ok((count, id)) => {
                let mut text = if compact {
                    format!("ok:{id} consolidated:{count}")
                } else {
                    format!("Consolidated {count} memories into new memory: {id}")
                };
                self.maybe_nudge(&mut text);
                text
            }
            Err(e) => {
                let mut text = format!("Error: {e}");
                self.maybe_nudge(&mut text);
                text
            }
        }
    }

    /// Scan for and optionally remove duplicate memories.
    #[tool(name = "rein_dedup", description = "Scan for duplicate memories using content similarity. Use dry_run=true to preview without deleting.")]
    fn rein_dedup(&self, Parameters(params): Parameters<DedupParams>) -> String {
        self.non_store_count.fetch_add(1, Ordering::Relaxed);
        let dry_run = params.dry_run.unwrap_or(false);
        let threshold = self.config.search.dedup_similarity as f32;
        let compact = self.compact();

        let result = self.with_store(|store| {
            let topics = store.list_topics()?;
            let mut dups_found = 0u32;
            let mut dups_removed = 0u32;

            for topic in &topics {
                // List ALL memories in this topic using direct SQL (not FTS with topic as query)
                let mems: Vec<Memory> = match store.get_by_topic(topic) {
                    Ok(m) => m,
                    Err(_) => continue,
                };

                let mut to_delete: std::collections::HashSet<String> = std::collections::HashSet::new();
                for i in 0..mems.len() {
                    if to_delete.contains(&mems[i].id) { continue; }
                    for j in (i + 1)..mems.len() {
                        if to_delete.contains(&mems[j].id) { continue; }
                        let sim = crate::extract::similarity(
                            &mems[i].content,
                            &mems[j].content,
                        );
                        if sim >= threshold {
                            to_delete.insert(mems[i].id.clone());
                            dups_found += 1;
                            break;
                        }
                    }
                }
                if !dry_run {
                    for id in &to_delete {
                        if store.delete(id).is_ok() {
                            dups_removed += 1;
                        }
                    }
                }
            }
            Ok((dups_found, dups_removed))
        });

        match result {
            Ok((dups_found, dups_removed)) => {
                let mut text = if compact {
                    format!("found:{dups_found} removed:{dups_removed}")
                } else if dry_run {
                    format!(
                        "Dedup scan: found {} potential duplicates (dry run, none removed)",
                        dups_found
                    )
                } else {
                    format!(
                        "Dedup scan: found {} duplicates, removed {}",
                        dups_found, dups_removed
                    )
                };
                self.maybe_nudge(&mut text);
                text
            }
            Err(e) => {
                let mut text = format!("Error: {e}");
                self.maybe_nudge(&mut text);
                text
            }
        }
    }
}


#[tool_handler]
impl ServerHandler for ReinServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions("rein: Multi-source cross-validated memory for AI agents. Use rein_store to save important context and rein_recall to search memories.")
    }
}

/// Start the MCP server over stdio.
pub async fn run_stdio(config: ReinConfig) -> anyhow::Result<()> {
    let store = config.open_store()?;
    let server = ReinServer::new(store, config);

    let transport = rmcp::transport::io::stdio();
    let service = server
        .serve(transport)
        .await
        .map_err(|e| anyhow::anyhow!("MCP server init error: {e}"))?;
    service
        .waiting()
        .await
        .map_err(|e| anyhow::anyhow!("MCP server error: {e}"))?;
    Ok(())
}

/// Start the MCP server over HTTP (Streamable HTTP / SSE).
/// Accessible via Tailscale or LAN for remote memory queries.
pub async fn run_http(config: ReinConfig) -> anyhow::Result<()> {
    use std::sync::Arc;
    use rmcp::transport::streamable_http_server::{
        StreamableHttpService, StreamableHttpServerConfig,
        session::local::LocalSessionManager,
    };
    use tokio_util::sync::CancellationToken;

    let bind = format!("{}:{}", config.server.sse_bind, config.server.sse_port);
    let db_path = config.resolve_db_path();
    let config_clone = config.clone();
    let cancel = CancellationToken::new();

    let session_manager = Arc::new(LocalSessionManager::default());
    let http_config = StreamableHttpServerConfig {
        stateful_mode: true,
        cancellation_token: cancel.clone(),
        ..Default::default()
    };

    let service = StreamableHttpService::new(
        move || {
            let store = config_clone.open_store()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
            let server = ReinServer::new(store, config_clone.clone());
            Ok(server)
        },
        session_manager,
        http_config,
    );

    let listener = tokio::net::TcpListener::bind(&bind).await?;
    tracing::info!("rein HTTP server listening on {bind}");
    eprintln!("rein HTTP server listening on http://{bind}/mcp");

    let service = hyper::service::service_fn(move |req| {
        let svc = service.clone();
        async move { Ok::<_, std::convert::Infallible>(svc.handle(req).await) }
    });

    loop {
        let (stream, addr) = listener.accept().await?;
        tracing::debug!("connection from {addr}");
        let svc = service.clone();
        tokio::spawn(async move {
            if let Err(e) = hyper_util::server::conn::auto::Builder::new(
                hyper_util::rt::TokioExecutor::new(),
            )
            .serve_connection(hyper_util::rt::TokioIo::new(stream), svc)
            .await
            {
                tracing::warn!("connection error: {e}");
            }
        });
    }
}
