use std::sync::atomic::{AtomicU32, Ordering};

use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{ServerCapabilities, ServerInfo};
use rmcp::{tool, tool_handler, tool_router, ServerHandler, ServiceExt};

use crate::config::ReinConfig;
use crate::mcp::compact;
use crate::mcp::tools::*;
use crate::store::SqliteStore;
use crate::types::*;

/// MCP server for rein memory system.
///
/// Creates a new SqliteStore (SQLite connection) per request instead of sharing
/// a single Mutex<SqliteStore>. This eliminates the Mutex bottleneck and enables
/// concurrent read operations. SQLite with WAL mode + FULL_MUTEX handles write
/// serialization internally.
pub struct ReinServer {
    config: ReinConfig,
    non_store_count: AtomicU32,
    tool_router: ToolRouter<Self>,
}

impl std::fmt::Debug for ReinServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReinServer")
            .field("compact", &self.config.server.compact)
            .finish_non_exhaustive()
    }
}

impl ReinServer {
    /// Create a new ReinServer.
    ///
    /// Stores the config so each request can open its own connection via
    /// `config.open_store()`, eliminating the Mutex bottleneck.
    pub fn new(config: ReinConfig) -> Self {
        Self {
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

    /// Open a per-request SqliteStore, run a synchronous closure, return the result.
    /// Each call gets its own SQLite connection — no Mutex needed.
    /// SQLite WAL mode handles concurrent readers and serializes writers.
    fn with_store<F, R>(&self, f: F) -> Result<R, String>
    where
        F: FnOnce(&SqliteStore) -> Result<R, ReinError>,
    {
        let store = self.config.open_store().map_err(|e| format!("{e}"))?;
        f(&store).map_err(|e| format!("{e}"))
    }
}

const HTTP_SESSION_COOKIE: &str = "rein_http_token";

fn constant_time_eq(left: &str, right: &str) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.bytes()
        .zip(right.bytes())
        .fold(0u8, |acc, (x, y)| acc | (x ^ y))
        == 0
}

fn cookie_value(headers: &hyper::HeaderMap, name: &str) -> Option<String> {
    let cookies = headers.get("cookie")?.to_str().ok()?;
    cookies.split(';').find_map(|part| {
        let (key, value) = part.trim().split_once('=')?;
        if key.trim() == name {
            Some(value.trim().to_string())
        } else {
            None
        }
    })
}

fn request_has_valid_http_auth(headers: &hyper::HeaderMap, expected: &str) -> bool {
    let auth_header = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let expected_str = format!("Bearer {expected}");
    if constant_time_eq(auth_header, &expected_str) {
        return true;
    }
    cookie_value(headers, HTTP_SESSION_COOKIE)
        .map(|value| constant_time_eq(&value, expected))
        .unwrap_or(false)
}

#[tool_router]
impl ReinServer {
    /// Search memories by query, topic, or keyword.
    /// Uses full pipeline: FTS5 → cached vectors → Google API → RRF fusion → Ebbinghaus weighting → cross-validation.
    #[tool(
        name = "rein_recall",
        description = "Search and recall memories by semantic query. Uses three-level waterfall search with cross-validation. Supports optional topic and keyword filters."
    )]
    fn rein_recall(&self, Parameters(params): Parameters<RecallParams>) -> String {
        self.non_store_count.fetch_add(1, Ordering::Relaxed);
        let limit = params.limit.unwrap_or(10).min(200);

        // Parse optional temporal filters
        let time_from = params.from.as_deref().and_then(parse_datetime);
        let time_to = params.to.as_deref().and_then(parse_datetime_end);
        let request_id = ulid::Ulid::new().to_string();

        let result = self.with_store(|store| {
            crate::search::recall::recall_temporal_with_request_id(
                store,
                &self.config,
                &params.query,
                params.topic.as_deref(),
                params.keyword.as_deref(),
                limit,
                time_from,
                time_to,
                params.expand,
                false, // MCP uses full pipeline
                Some(request_id.clone()),
            )
            .map_err(|e| ReinError::Config(format!("{e}")))
        });

        // Re-classify for transparency (sub-microsecond, no overhead)
        let route = crate::search::classify::classify(
            &params.query,
            time_from.is_some(),
            time_to.is_some(),
        );

        match result {
            Ok(results) => {
                let mut text = compact::format_recall_results_mcp(&results, self.compact());
                if text.is_empty() {
                    text = if self.compact() {
                        "none".to_string()
                    } else {
                        "No memories found.".to_string()
                    };
                }
                if !self.compact() && !results.is_empty() {
                    text = format!(
                        "[route: {} | request_id: {}] {}",
                        route.query_type, request_id, text
                    );
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
    #[tool(
        name = "rein_store",
        description = "Store a new memory. Automatically deduplicates against existing memories."
    )]
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

        let memory = crate::ops::build_memory(
            &self.config,
            params.topic,
            params.content.clone(),
            importance,
            keywords,
            Source::Manual,
        );

        let config = self.config.clone();
        let result = self.with_store(|store| crate::ops::store_memory(store, &config, memory));

        match result {
            Ok(id) => compact::format_store_result(&id, self.compact()),
            Err(e) => format!("Error storing: {e}"),
        }
    }

    /// Ingest a full session/transcript through the full extraction pipeline.
    #[tool(
        name = "rein_ingest_session",
        description = "Ingest a full session transcript into memories, concepts, links, and an episode using the full extraction pipeline."
    )]
    fn rein_ingest_session(&self, Parameters(params): Parameters<IngestSessionParams>) -> String {
        self.non_store_count.store(0, Ordering::Relaxed);

        let config = self.config.clone();
        let result = if let Some(turns) = params.turns {
            // Enforce size cap on turns path (same 500KB limit as content path)
            let total_chars: usize = turns.iter().map(|t| t.role.len() + t.content.len()).sum();
            if total_chars > 500_000 {
                return "Error: turns too large (max 500KB aggregate)".to_string();
            }
            if turns.len() > 1000 {
                return "Error: too many turns (max 1000)".to_string();
            }
            let started_at = params
                .started_at
                .as_deref()
                .and_then(parse_datetime)
                .or_else(|| params.started_at.as_deref().and_then(parse_datetime_end));
            let session = crate::types::SessionIngest {
                schema_version: 1,
                artifact_kind: "session".to_string(),
                session_id: params.session_id,
                title: params.title,
                started_at,
                ended_at: None,
                summary: params.summary,
                source_agent: params.agent_label.clone(),
                source_label: Some("mcp".to_string()),
                compact_summary: params.compact_summary,
                tool_outputs: params.tool_outputs.unwrap_or_default(),
                turns: turns
                    .into_iter()
                    .map(|turn| crate::types::SessionTurn {
                        role: turn.role,
                        content: turn.content,
                    })
                    .collect(),
            };
            if params.async_mode.unwrap_or(false) {
                crate::ops::queue_ingest_session(
                    &config,
                    &session,
                    params.agent_label.as_deref(),
                    params.is_subagent.unwrap_or(false),
                )
            } else {
                crate::ops::ingest_session_sync_report(
                    &config,
                    &session,
                    params.agent_label.as_deref(),
                    params.is_subagent.unwrap_or(false),
                )
            }
        } else if let Some(content) = params.content {
            if content.len() > 500_000 {
                return "Error: content too large (max 500KB)".to_string();
            }
            if params.async_mode.unwrap_or(false) {
                crate::ops::queue_ingest_session_text(
                    &config,
                    &content,
                    params.agent_label.as_deref(),
                    params.is_subagent.unwrap_or(false),
                )
            } else {
                crate::ops::ingest_session_text_sync_report(
                    &config,
                    &content,
                    params.agent_label.as_deref(),
                    params.is_subagent.unwrap_or(false),
                )
            }
        } else {
            Err(crate::types::ReinError::Config(
                "ingest_session requires either content or turns".to_string(),
            ))
        };

        match result {
            Ok(report) => {
                if self.compact() {
                    format!(
                        "ok queued:{} memories:{} concepts:{} links:{} artifact:{} episode:{}",
                        report.queued,
                        report.memory_count,
                        report.concept_count,
                        report.link_count,
                        report.artifact_id.as_deref().unwrap_or("-"),
                        report.episode_id.as_deref().unwrap_or("-"),
                    )
                } else {
                    format!(
                        "Ingested session: queued={} artifact={} episode={} memories={} concepts={} links={}",
                        report.queued,
                        report.artifact_id.as_deref().unwrap_or("-"),
                        report.episode_id.as_deref().unwrap_or("-"),
                        report.memory_count,
                        report.concept_count,
                        report.link_count
                    )
                }
            }
            Err(e) => format!("Error ingesting session: {e}"),
        }
    }

    /// Update an existing memory by ID.
    #[tool(
        name = "rein_update",
        description = "Update the content of an existing memory by its ID."
    )]
    fn rein_update(&self, Parameters(params): Parameters<UpdateParams>) -> String {
        // Don't count update as non-store (it's a mutation, not a read)
        self.non_store_count.store(0, Ordering::Relaxed);

        let base_lambda = self.config.decay.base_lambda;
        let compact = self.compact();
        let id = params.id.clone();

        let result = self.with_store(|store| {
            let mut memory = store.get(&params.id)?;
            memory.content = params.content.clone();
            memory.summary = params
                .content
                .chars()
                .take(crate::types::SUMMARY_MAX_CHARS)
                .collect();
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

    /// Report which recalled memories were actually used by the agent.
    /// This feedback improves future recall quality through adaptive weight learning.
    #[tool(
        name = "rein_feedback",
        description = "Report which recalled memories were actually used. Improves future recall quality. Call after using rein_recall results."
    )]
    fn rein_feedback(&self, Parameters(params): Parameters<FeedbackParams>) -> String {
        if params.memory_ids.is_empty() {
            return "No memory IDs provided.".to_string();
        }

        let result = self.with_store(|store| {
            let conn = store.conn();
            let mut emitted = 0;

            // Emit RecallAccess for each confirmed-used memory
            for mem_id in &params.memory_ids {
                let _ = crate::store::adaptive::emit_event(conn, crate::store::adaptive::FeedbackEvent {
                    event_type: crate::store::adaptive::EventType::RecallAccess,
                    request_id: params.request_id.clone(),
                    memory_id: Some(mem_id.clone()),
                    concept_id: None,
                    query: params.query.clone(),
                    query_type: None,
                    topic: None,
                    payload: Some(serde_json::json!({
                        "source": "agent_feedback",
                        "helpful": params.helpful,
                    })),
                });
                // Increment access_count on the memory
                let _ = conn.execute(
                    "UPDATE memories SET access_count = access_count + 1, last_accessed = ?1 WHERE id = ?2",
                    rusqlite::params![chrono::Utc::now().to_rfc3339(), mem_id],
                );
                emitted += 1;
            }
            Ok(emitted)
        });

        match result {
            Ok(count) => {
                let mut text = if self.compact() {
                    format!("ok:{count}")
                } else {
                    format!(
                        "Feedback recorded for {} memories. This improves future recall quality.",
                        count
                    )
                };
                self.maybe_nudge(&mut text);
                text
            }
            Err(e) => format!("Error: {e}"),
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
    #[tool(
        name = "rein_stats",
        description = "Show memory store statistics (total, LTM/STM counts, avg strength)."
    )]
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
    #[tool(
        name = "rein_health",
        description = "Check health of memory topics. Shows stale count, avg strength, consolidation needs."
    )]
    fn rein_health(&self, Parameters(params): Parameters<HealthParams>) -> String {
        self.non_store_count.fetch_add(1, Ordering::Relaxed);

        let result = self.with_store(|store| store.health(params.topic.as_deref()));

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
    #[tool(
        name = "rein_consolidate",
        description = "Consolidate all memories in a topic into a single summary memory, removing the originals."
    )]
    fn rein_consolidate(&self, Parameters(params): Parameters<ConsolidateParams>) -> String {
        self.non_store_count.store(0, Ordering::Relaxed); // consolidate is a write operation
        let compact = self.compact();
        let dry_run = params.dry_run.unwrap_or(false);
        let merge_variants = params.merge_variants.unwrap_or(false);
        let topic = params.topic.clone();
        let topics = params.topics.clone().unwrap_or_default();
        let pattern = params.pattern.clone();
        let summary = params.summary.clone();
        let all = params.all.unwrap_or(false);

        let result = self.with_store(|store| {
            let groups = crate::ops::resolve_topic_groups(
                store,
                topic.as_deref(),
                &topics,
                pattern.as_deref(),
                all,
                merge_variants,
            )?;
            if groups.is_empty() {
                return Ok(None);
            }
            let report = crate::ops::run_consolidation_sync(
                store,
                &self.config,
                &groups,
                summary.as_deref(),
                dry_run,
            )?;
            Ok(Some(report))
        });

        match result {
            Ok(None) => {
                let mut text = if compact {
                    "none".to_string()
                } else if let Some(topic) = topic {
                    format!("No memories found for topic: {topic}")
                } else if let Some(pattern) = pattern {
                    format!("No topics matched pattern: {pattern}")
                } else {
                    "No topics matched the selected scope.".to_string()
                };
                self.maybe_nudge(&mut text);
                text
            }
            Ok(Some(report)) => {
                let mut text = if compact {
                    if dry_run {
                        format!(
                            "dry:groups={} memories={}",
                            report.groups_processed, report.memories_replaced
                        )
                    } else {
                        format!(
                            "ok:groups={} memories={}",
                            report.groups_processed, report.memories_replaced
                        )
                    }
                } else if dry_run {
                    format!(
                        "Dry run: {} groups ({} memories) would be consolidated",
                        report.groups_processed, report.memories_replaced
                    )
                } else {
                    format!(
                        "Consolidated {} groups ({} memories)",
                        report.groups_processed, report.memories_replaced
                    )
                };

                if !compact {
                    let visible_groups: Vec<_> = report
                        .groups
                        .iter()
                        .filter(|group| group.memory_count > 0)
                        .take(8)
                        .collect();
                    for group in &visible_groups {
                        if group.source_topics.len() > 1 {
                            text.push_str(&format!(
                                "\n- {} <= {} [{} memories]",
                                group.canonical_topic,
                                group.source_topics.join(", "),
                                group.memory_count
                            ));
                        } else {
                            text.push_str(&format!(
                                "\n- {} [{} memories]",
                                group.canonical_topic, group.memory_count
                            ));
                        }
                    }
                    let total_non_empty = report
                        .groups
                        .iter()
                        .filter(|group| group.memory_count > 0)
                        .count();
                    if total_non_empty > visible_groups.len() {
                        text.push_str(&format!(
                            "\n... {} more groups",
                            total_non_empty - visible_groups.len()
                        ));
                    }
                }

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

    // ===== Knowledge Graph (Memoir/Concept/Link) tools =====

    /// Create a new memoir (knowledge graph container).
    #[tool(
        name = "rein_memoir_create",
        description = "Create a new memoir (named knowledge graph). Use to organize concepts and their relationships."
    )]
    fn rein_memoir_create(&self, Parameters(params): Parameters<MemoirCreateParams>) -> String {
        self.non_store_count.store(0, Ordering::Relaxed);
        let compact = self.compact();

        let memoir = crate::types::Memoir {
            id: String::new(),
            name: params.name.clone(),
            description: params.description.unwrap_or_default(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };

        let result = self.with_store(|store| store.create_memoir(memoir));

        match result {
            Ok(id) => {
                if compact {
                    format!("ok:{id}")
                } else {
                    format!("Created memoir '{}': {id}", params.name)
                }
            }
            Err(e) => format!("Error: {e}"),
        }
    }

    /// List all memoirs.
    #[tool(
        name = "rein_memoir_list",
        description = "List all memoirs (knowledge graphs)."
    )]
    fn rein_memoir_list(&self) -> String {
        self.non_store_count.fetch_add(1, Ordering::Relaxed);
        let compact = self.compact();

        let result = self.with_store(|store| store.list_memoirs());

        match result {
            Ok(memoirs) => {
                if memoirs.is_empty() {
                    return if compact {
                        "none".to_string()
                    } else {
                        "No memoirs found.".to_string()
                    };
                }
                let mut text = String::new();
                for m in &memoirs {
                    if compact {
                        text.push_str(&format!("{}:{}\n", m.name, m.description));
                    } else {
                        text.push_str(&format!("- {} — {}\n", m.name, m.description));
                    }
                }
                text
            }
            Err(e) => format!("Error: {e}"),
        }
    }

    /// Show a memoir and all its concepts.
    #[tool(
        name = "rein_memoir_show",
        description = "Show memoir details and list all concepts within it."
    )]
    fn rein_memoir_show(&self, Parameters(params): Parameters<MemoirShowParams>) -> String {
        self.non_store_count.fetch_add(1, Ordering::Relaxed);
        let compact = self.compact();

        let result = self.with_store(|store| {
            let memoir = store.get_memoir(&params.name)?.ok_or_else(|| {
                ReinError::NotFound(format!("memoir '{}' not found", params.name))
            })?;
            let export = store.export_memoir(&params.name, "ascii")?;
            Ok((memoir, export))
        });

        match result {
            Ok((memoir, export)) => {
                if compact {
                    export
                } else {
                    format!(
                        "Memoir: {} — {}\n\n{}",
                        memoir.name, memoir.description, export
                    )
                }
            }
            Err(e) => format!("Error: {e}"),
        }
    }

    /// Add a concept to a memoir.
    #[tool(
        name = "rein_memoir_add_concept",
        description = "Add a concept (knowledge node) to a memoir with name, definition, and optional labels."
    )]
    fn rein_memoir_add_concept(&self, Parameters(params): Parameters<ConceptAddParams>) -> String {
        self.non_store_count.store(0, Ordering::Relaxed);
        let compact = self.compact();

        let labels: Vec<String> = params
            .labels
            .map(|l| {
                l.split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            })
            .unwrap_or_default();

        let concept = crate::types::Concept {
            id: String::new(),
            memoir_id: params.memoir.clone(),
            name: params.name.clone(),
            definition: params.definition,
            labels,
            source_memory_ids: vec![],
            confidence: 0.5,
            revision: 1,
            last_episode_id: None,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };

        let result = self.with_store(|store| store.add_concept(concept));

        match result {
            Ok(id) => {
                if compact {
                    format!("ok:{id}")
                } else {
                    format!(
                        "Added concept '{}' to memoir '{}': {id}",
                        params.name, params.memoir
                    )
                }
            }
            Err(e) => format!("Error: {e}"),
        }
    }

    /// Refine a concept's definition.
    #[tool(
        name = "rein_memoir_refine",
        description = "Refine a concept: update definition, increment revision, boost confidence."
    )]
    fn rein_memoir_refine(&self, Parameters(params): Parameters<ConceptRefineParams>) -> String {
        self.non_store_count.store(0, Ordering::Relaxed);
        let compact = self.compact();

        let result = self.with_store(|store| {
            store.refine_concept(&params.memoir, &params.name, &params.definition)
        });

        match result {
            Ok(()) => {
                if compact {
                    format!("ok:{}", params.name)
                } else {
                    format!(
                        "Refined concept '{}' in memoir '{}'",
                        params.name, params.memoir
                    )
                }
            }
            Err(e) => format!("Error: {e}"),
        }
    }

    /// Search concepts within a memoir.
    #[tool(
        name = "rein_memoir_search",
        description = "Full-text search for concepts within a memoir."
    )]
    fn rein_memoir_search(&self, Parameters(params): Parameters<ConceptSearchParams>) -> String {
        self.non_store_count.fetch_add(1, Ordering::Relaxed);
        let compact = self.compact();
        let limit = params.limit.unwrap_or(10).min(100);

        let result =
            self.with_store(|store| store.search_concepts(&params.memoir, &params.query, limit));

        match result {
            Ok(concepts) => {
                if concepts.is_empty() {
                    return if compact {
                        "none".to_string()
                    } else {
                        "No concepts found.".to_string()
                    };
                }
                let mut text = String::new();
                for c in &concepts {
                    if compact {
                        text.push_str(&format!(
                            "{}:{}:r{}:c{:.1}\n",
                            c.name, c.definition, c.revision, c.confidence
                        ));
                    } else {
                        text.push_str(&format!(
                            "- {} (rev:{}, conf:{:.1}) — {}\n",
                            c.name, c.revision, c.confidence, c.definition
                        ));
                    }
                }
                text
            }
            Err(e) => format!("Error: {e}"),
        }
    }

    /// Search concepts across all memoirs.
    #[tool(
        name = "rein_memoir_search_all",
        description = "Full-text search for concepts across all memoirs."
    )]
    fn rein_memoir_search_all(
        &self,
        Parameters(params): Parameters<ConceptSearchAllParams>,
    ) -> String {
        self.non_store_count.fetch_add(1, Ordering::Relaxed);
        let compact = self.compact();
        let limit = params.limit.unwrap_or(10).min(100);

        let result = self.with_store(|store| store.search_all_concepts(&params.query, limit));

        match result {
            Ok(concepts) => {
                if concepts.is_empty() {
                    return if compact {
                        "none".to_string()
                    } else {
                        "No concepts found.".to_string()
                    };
                }
                let mut text = String::new();
                for c in &concepts {
                    if compact {
                        text.push_str(&format!(
                            "{}:{}:r{}:c{:.1}\n",
                            c.name, c.definition, c.revision, c.confidence
                        ));
                    } else {
                        text.push_str(&format!(
                            "- {} (rev:{}, conf:{:.1}) — {}\n",
                            c.name, c.revision, c.confidence, c.definition
                        ));
                    }
                }
                text
            }
            Err(e) => format!("Error: {e}"),
        }
    }

    /// Create a typed relation between two concepts.
    #[tool(
        name = "rein_memoir_link",
        description = "Create a typed relation (edge) between two concepts in a memoir. Relations: part_of, depends_on, related_to, contradicts, refines, alternative_to, caused_by, instance_of, superseded_by."
    )]
    fn rein_memoir_link(&self, Parameters(params): Parameters<LinkParams>) -> String {
        self.non_store_count.store(0, Ordering::Relaxed);
        let compact = self.compact();

        let relation = match params.relation.parse::<crate::types::Relation>() {
            Ok(r) => r,
            Err(e) => return format!("Error: {e}"),
        };

        let result = self.with_store(|store| {
            let from = store
                .get_concept(&params.memoir, &params.from)?
                .ok_or_else(|| {
                    ReinError::NotFound(format!(
                        "concept '{}' not found in memoir '{}'",
                        params.from, params.memoir
                    ))
                })?;
            let to = store
                .get_concept(&params.memoir, &params.to)?
                .ok_or_else(|| {
                    ReinError::NotFound(format!(
                        "concept '{}' not found in memoir '{}'",
                        params.to, params.memoir
                    ))
                })?;

            let link = crate::types::ConceptLink {
                id: String::new(),
                source_id: from.id,
                target_id: to.id,
                relation,
                weight: 1.0,
                created_at: chrono::Utc::now(),
                valid_from: None,
                valid_until: None,
            };
            store.add_link(link)
        });

        match result {
            Ok(id) => {
                if compact {
                    format!("ok:{id}")
                } else {
                    format!(
                        "Linked '{}' --{}-> '{}': {id}",
                        params.from, params.relation, params.to
                    )
                }
            }
            Err(e) => format!("Error: {e}"),
        }
    }

    /// Inspect a concept's neighborhood via BFS.
    #[tool(
        name = "rein_memoir_inspect",
        description = "Inspect a concept's neighborhood via BFS traversal. Returns the concept, its neighbors, and connecting links up to the specified depth."
    )]
    fn rein_memoir_inspect(&self, Parameters(params): Parameters<InspectParams>) -> String {
        self.non_store_count.fetch_add(1, Ordering::Relaxed);
        let compact = self.compact();
        let depth = params.depth.unwrap_or(1).min(5);

        let result =
            self.with_store(|store| store.inspect_concept(&params.memoir, &params.name, depth));

        match result {
            Ok((center, neighbors, links)) => {
                let mut text = String::new();
                if compact {
                    text.push_str(&format!(
                        "center:{}:c{:.1}:r{}\n",
                        center.name, center.confidence, center.revision
                    ));
                    for n in &neighbors {
                        text.push_str(&format!(
                            "neighbor:{}:c{:.1}:r{}\n",
                            n.name, n.confidence, n.revision
                        ));
                    }
                    for l in &links {
                        text.push_str(&format!(
                            "link:{}->{}:{}\n",
                            l.source_id, l.target_id, l.relation
                        ));
                    }
                } else {
                    text.push_str(&format!(
                        "Center: {} (conf:{:.1}, rev:{})\n  {}\n\n",
                        center.name, center.confidence, center.revision, center.definition
                    ));
                    if !neighbors.is_empty() {
                        text.push_str("Neighbors:\n");
                        for n in &neighbors {
                            text.push_str(&format!(
                                "  - {} (conf:{:.1}, rev:{}) — {}\n",
                                n.name, n.confidence, n.revision, n.definition
                            ));
                        }
                    }
                    if !links.is_empty() {
                        text.push_str("\nLinks:\n");
                        for l in &links {
                            text.push_str(&format!(
                                "  {} --{}-> {}\n",
                                l.source_id, l.relation, l.target_id
                            ));
                        }
                    }
                }
                text
            }
            Err(e) => format!("Error: {e}"),
        }
    }

    /// Export a memoir graph.
    #[tool(
        name = "rein_memoir_export",
        description = "Export a memoir's knowledge graph. Formats: json (structured), ascii (human-readable), dot (Graphviz)."
    )]
    fn rein_memoir_export(&self, Parameters(params): Parameters<ExportParams>) -> String {
        self.non_store_count.fetch_add(1, Ordering::Relaxed);
        let format = params.format.as_deref().unwrap_or("json");

        let result = self.with_store(|store| store.export_memoir(&params.memoir, format));

        match result {
            Ok(output) => output,
            Err(e) => format!("Error: {e}"),
        }
    }

    /// Scan for and optionally remove duplicate memories.
    #[tool(
        name = "rein_dedup",
        description = "Scan for duplicate memories using content similarity. Use dry_run=true to preview without deleting."
    )]
    fn rein_dedup(&self, Parameters(params): Parameters<DedupParams>) -> String {
        self.non_store_count.fetch_add(1, Ordering::Relaxed);
        let dry_run = params.dry_run.unwrap_or(false);
        let merge_variants = params.merge_variants.unwrap_or(false);
        let compact = self.compact();

        let result = self.with_store(|store| {
            // A1: adaptive threshold with config fallback.
            let threshold = crate::ops::effective_dedup_threshold(store, &self.config);
            crate::ops::run_dedup(store, &self.config, threshold, dry_run, merge_variants)
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

    /// One-click cleanup: consolidate fragmented topics, then deduplicate and refresh adaptive state.
    #[tool(
        name = "rein_cleanup",
        description = "One-click cleanup for memories: consolidate fragmented topics, deduplicate, and refresh adaptive state. Supports dry_run preview."
    )]
    fn rein_cleanup(&self, Parameters(params): Parameters<CleanupParams>) -> String {
        self.non_store_count.store(0, Ordering::Relaxed);
        let compact = self.compact();
        let dry_run = params.dry_run.unwrap_or(false);
        let topic = params.topic.clone();
        let topics = params.topics.clone().unwrap_or_default();
        let pattern = params.pattern.clone();
        let exact_topics = params.exact_topics.unwrap_or(false);
        let merge_variants = !exact_topics;
        let all = params.all.unwrap_or(false)
            || (topic.is_none() && topics.is_empty() && pattern.is_none());

        let result = self.with_store(|store| {
            let groups = crate::ops::resolve_topic_groups(
                store,
                topic.as_deref(),
                &topics,
                pattern.as_deref(),
                all,
                merge_variants,
            )?;
            if groups.is_empty() {
                return Ok(None);
            }
            let report = crate::ops::run_cleanup_sync(
                store,
                &self.config,
                &groups,
                merge_variants,
                dry_run,
            )?;
            Ok(Some(report))
        });

        match result {
            Ok(None) => {
                let mut text = if compact {
                    "none".to_string()
                } else if let Some(topic) = topic {
                    format!("No memories found for topic: {topic}")
                } else if let Some(pattern) = pattern {
                    format!("No topics matched pattern: {pattern}")
                } else {
                    "No topics matched the selected scope.".to_string()
                };
                self.maybe_nudge(&mut text);
                text
            }
            Ok(Some(report)) => {
                let mut text = if compact {
                    if dry_run {
                        format!(
                            "dry:groups={} memories={} found={}",
                            report.consolidation.groups_processed,
                            report.consolidation.memories_replaced,
                            report.duplicates_found
                        )
                    } else {
                        format!(
                            "ok:groups={} memories={} removed={}/{}",
                            report.consolidation.groups_processed,
                            report.consolidation.memories_replaced,
                            report.duplicates_merged,
                            report.duplicates_found
                        )
                    }
                } else if dry_run {
                    format!(
                        "Dry run: {} groups ({} memories) would be consolidated; found {} duplicates",
                        report.consolidation.groups_processed,
                        report.consolidation.memories_replaced,
                        report.duplicates_found
                    )
                } else {
                    format!(
                        "Cleanup finished: {} groups consolidated ({} memories), removed {} of {} duplicates",
                        report.consolidation.groups_processed,
                        report.consolidation.memories_replaced,
                        report.duplicates_merged,
                        report.duplicates_found
                    )
                };

                if !compact {
                    let visible_groups: Vec<_> = report
                        .consolidation
                        .groups
                        .iter()
                        .filter(|group| group.memory_count > 0)
                        .take(8)
                        .collect();
                    for group in &visible_groups {
                        if group.source_topics.len() > 1 {
                            text.push_str(&format!(
                                "\n- {} <= {} [{} memories]",
                                group.canonical_topic,
                                group.source_topics.join(", "),
                                group.memory_count
                            ));
                        } else {
                            text.push_str(&format!(
                                "\n- {} [{} memories]",
                                group.canonical_topic, group.memory_count
                            ));
                        }
                    }
                    let total_non_empty = report
                        .consolidation
                        .groups
                        .iter()
                        .filter(|group| group.memory_count > 0)
                        .count();
                    if total_non_empty > visible_groups.len() {
                        text.push_str(&format!(
                            "\n... {} more groups",
                            total_non_empty - visible_groups.len()
                        ));
                    }
                }

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

    /// Show the most recently created memories.
    #[tool(
        name = "rein_recent",
        description = "List the most recently created memories, ordered by creation time."
    )]
    fn rein_recent(&self, Parameters(params): Parameters<RecentParams>) -> String {
        self.non_store_count.fetch_add(1, Ordering::Relaxed);
        let limit = params.limit.unwrap_or(10).min(100);
        let compact = self.compact();

        let result = self.with_store(|store| store.recent(limit));

        match result {
            Ok(memories) => {
                if memories.is_empty() {
                    return "No memories found.".to_string();
                }
                let mut text = if compact {
                    memories
                        .iter()
                        .map(|m| format!("[{}] {}", m.topic, m.summary))
                        .collect::<Vec<_>>()
                        .join("\n")
                } else {
                    memories
                        .iter()
                        .map(|m| {
                            let age = chrono::Utc::now().signed_duration_since(m.created_at);
                            let age_str = if age.num_days() > 0 {
                                format!("{}d ago", age.num_days())
                            } else if age.num_hours() > 0 {
                                format!("{}h ago", age.num_hours())
                            } else {
                                format!("{}m ago", age.num_minutes())
                            };
                            format!(
                                "[{}] {} ({}, {}, str:{:.2})\n  id: {}",
                                m.topic, m.summary, m.importance, age_str, m.strength, m.id
                            )
                        })
                        .collect::<Vec<_>>()
                        .join("\n")
                };
                self.maybe_nudge(&mut text);
                text
            }
            Err(e) => format!("Error: {e}"),
        }
    }

    /// Garbage collect weak STM memories below the configured strength threshold.
    #[tool(
        name = "rein_gc",
        description = "Run garbage collection: apply decay to all memories, then prune weak STM memories below the configured strength threshold. Use dry_run=true to preview."
    )]
    fn rein_gc(&self, Parameters(params): Parameters<GcParams>) -> String {
        self.non_store_count.fetch_add(1, Ordering::Relaxed);
        let dry_run = params.dry_run.unwrap_or(false);
        let threshold = self.config.decay.prune_threshold;
        let compact = self.compact();

        let config = self.config.clone();
        let result = self
            .with_store(|store| crate::ops::run_gc_adaptive(store, &config, threshold, dry_run));

        match result {
            Ok((decayed, pruned, concepts)) => {
                let mut text = if compact {
                    if dry_run {
                        let mut s = format!("would_prune:{pruned}");
                        if concepts > 0 {
                            s.push_str(&format!(" concepts:{concepts}"));
                        }
                        s
                    } else {
                        format!("decayed:{decayed} pruned:{pruned}")
                    }
                } else if dry_run {
                    let mut s = format!("GC dry run: {pruned} weak STM memories would be pruned (threshold: {threshold})");
                    if concepts > 0 {
                        s.push_str(&format!(", {concepts} low-quality concepts"));
                    }
                    s
                } else {
                    let mut s = format!("GC complete: decayed {decayed} memories, pruned {pruned} weak STM memories (threshold: {threshold})");
                    if concepts > 0 {
                        s.push_str(&format!(", {concepts} low-quality concepts"));
                    }
                    s
                };
                self.maybe_nudge(&mut text);
                text
            }
            Err(e) => format!("Error: {e}"),
        }
    }

    /// Auto-link all memories based on content similarity. Creates bidirectional related_ids links.
    #[tool(
        name = "rein_organize",
        description = "Scan all memories and create bidirectional links between related ones based on content similarity. Returns the number of new links created."
    )]
    fn rein_organize(&self, Parameters(params): Parameters<OrganizeParams>) -> String {
        self.non_store_count.fetch_add(1, Ordering::Relaxed);
        let max_links = params.max_links.unwrap_or(5);
        let compact = self.compact();

        let result = self.with_store(|store| {
            // A1: prefer adaptive threshold over static config.
            let threshold = crate::ops::effective_dedup_threshold(store, &self.config);
            store.organize(threshold, max_links)
        });

        match result {
            Ok(links) => {
                let mut text = if compact {
                    format!("links:{links}")
                } else {
                    format!("Organized: created {links} new links between related memories")
                };
                self.maybe_nudge(&mut text);
                text
            }
            Err(e) => format!("Error: {e}"),
        }
    }

    /// View a timeline of events (episodes, concept changes, memory creation) in a date range.
    #[tool(
        name = "rein_timeline",
        description = "Show a chronological timeline of knowledge events: episodes, concept revisions, and memory creation. Supports date range filtering."
    )]
    fn rein_timeline(&self, Parameters(params): Parameters<TimelineParams>) -> String {
        let limit = params.limit.unwrap_or(20).min(200);
        let from = params.from.as_deref().and_then(parse_datetime);
        let to = params.to.as_deref().and_then(parse_datetime_end);

        // Warn on malformed dates (return error instead of silently ignoring)
        if params.from.is_some() && from.is_none() {
            return format!("Error: invalid 'from' date format: {:?}", params.from);
        }
        if params.to.is_some() && to.is_none() {
            return format!("Error: invalid 'to' date format: {:?}", params.to);
        }

        let result = self.with_store(|store| {
            let mut events: Vec<(chrono::DateTime<chrono::Utc>, String)> = Vec::new();

            // Collect episodes (support one-sided ranges via sentinel dates)
            let episodes = match (from, to) {
                (Some(f), Some(t)) => store.get_episodes_in_range(f, t)?,
                (Some(f), None) => store.get_episodes_in_range(f, chrono::Utc::now() + chrono::Duration::days(1))?,
                (None, Some(t)) => store.get_episodes_in_range(
                    chrono::DateTime::parse_from_rfc3339("2000-01-01T00:00:00Z").unwrap().with_timezone(&chrono::Utc), t
                )?,
                (None, None) => store.list_episodes(limit)?,
            };
            for ep in &episodes {
                let decisions = if ep.decisions.is_empty() { String::new() }
                    else { format!(" | decisions: {}", ep.decisions.join(", ")) };
                events.push((ep.created_at, format!(
                    "[episode] {} — {} concepts, {} memories{}",
                    ep.title, ep.concept_ids.len(), ep.memory_ids.len(), decisions
                )));
            }

            // Collect concept revisions in the window (parameterized queries)
            {
                let mut where_clauses = Vec::new();
                let mut param_values: Vec<String> = Vec::new();
                if let Some(f) = from {
                    param_values.push(f.to_rfc3339());
                    where_clauses.push(format!("r.created_at >= ?{}", param_values.len()));
                }
                if let Some(t) = to {
                    param_values.push(t.to_rfc3339());
                    where_clauses.push(format!("r.created_at <= ?{}", param_values.len()));
                }
                let where_str = if where_clauses.is_empty() {
                    String::new()
                } else {
                    format!(" WHERE {}", where_clauses.join(" AND "))
                };
                let rev_sql = format!(
                    "SELECT r.revision, r.definition, r.created_at, c.name, c.memoir_id \
                     FROM concept_revisions r JOIN concepts c ON r.concept_id = c.id{} \
                     ORDER BY r.created_at DESC LIMIT {}",
                    where_str, limit
                );
                match store.conn().prepare(&rev_sql) {
                    Ok(mut stmt) => {
                        let extract_row = |row: &rusqlite::Row| -> rusqlite::Result<(u32, String, String, String)> {
                            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
                        };
                        let collected: Vec<(u32, String, String, String)> = match param_values.len() {
                            0 => stmt.query_map([], extract_row)
                                .map(|rows| rows.filter_map(|r| r.ok()).collect())
                                .unwrap_or_default(),
                            1 => stmt.query_map(rusqlite::params![param_values[0]], extract_row)
                                .map(|rows| rows.filter_map(|r| r.ok()).collect())
                                .unwrap_or_default(),
                            _ => stmt.query_map(rusqlite::params![param_values[0], param_values[1]], extract_row)
                                .map(|rows| rows.filter_map(|r| r.ok()).collect())
                                .unwrap_or_default(),
                        };
                        for (rev, def, created_str, name) in collected {
                            if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(&created_str) {
                                let short_def: String = def.chars().take(80).collect();
                                events.push((dt.with_timezone(&chrono::Utc), format!(
                                    "[revision] {} r{}: {}", name, rev, short_def
                                )));
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!("timeline: concept revision query failed: {}", e);
                    }
                }
            }

            // Collect memories in the requested window via SQL (supports one-sided ranges)
            let mem_sql = if from.is_some() || to.is_some() {
                let mut where_parts = Vec::new();
                let mut param_values: Vec<String> = Vec::new();
                if let Some(f) = from {
                    where_parts.push(format!("created_at >= ?{}", param_values.len() + 1));
                    param_values.push(f.to_rfc3339());
                }
                if let Some(t) = to {
                    where_parts.push(format!("created_at <= ?{}", param_values.len() + 1));
                    param_values.push(t.to_rfc3339());
                }
                let sql = format!(
                    "SELECT id, topic, summary, created_at FROM memories WHERE {} ORDER BY created_at DESC LIMIT {}",
                    where_parts.join(" AND "), limit
                );
                let mut stmt = store.conn().prepare(&sql).map_err(ReinError::Database)?;
                let rows: Vec<_> = match param_values.len() {
                    1 => stmt.query_map(rusqlite::params![param_values[0]], |row| {
                        Ok((row.get::<_, String>(1)?, row.get::<_, String>(2)?, row.get::<_, String>(3)?))
                    }).map_err(ReinError::Database)?.filter_map(|r| r.ok()).collect(),
                    2 => stmt.query_map(rusqlite::params![param_values[0], param_values[1]], |row| {
                        Ok((row.get::<_, String>(1)?, row.get::<_, String>(2)?, row.get::<_, String>(3)?))
                    }).map_err(ReinError::Database)?.filter_map(|r| r.ok()).collect(),
                    _ => Vec::new(),
                };
                rows
            } else {
                store.recent(limit)?.iter().map(|m| (m.topic.clone(), m.summary.clone(), m.created_at.to_rfc3339())).collect()
            };
            for (topic, summary, created_str) in &mem_sql {
                if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(created_str) {
                    events.push((dt.with_timezone(&chrono::Utc), format!("[memory] [{}] {}", topic, summary)));
                }
            }

            events.sort_by(|a, b| b.0.cmp(&a.0));
            events.truncate(limit);
            Ok(events)
        });

        match result {
            Ok(events) => {
                if events.is_empty() {
                    "No events found in the specified range.".to_string()
                } else {
                    events
                        .iter()
                        .map(|(dt, desc)| format!("{} {}", dt.format("%Y-%m-%d %H:%M"), desc))
                        .collect::<Vec<_>>()
                        .join("\n")
                }
            }
            Err(e) => format!("Error: {e}"),
        }
    }

    /// Show revision history of a concept — when and how its definition changed over time.
    #[tool(
        name = "rein_concept_history",
        description = "Show the revision history of a concept: when it changed, what the old definitions were, and which episode triggered each change."
    )]
    fn rein_concept_history(&self, Parameters(params): Parameters<ConceptHistoryParams>) -> String {
        let limit = params.limit.unwrap_or(10).min(100);

        let result = self.with_store(|store| {
            let current = store
                .get_concept(&params.memoir, &params.name)?
                .ok_or_else(|| {
                    ReinError::NotFound(format!("concept '{}' not found", params.name))
                })?;
            let history = store.get_concept_history(&params.memoir, &params.name, limit)?;
            Ok((current, history))
        });

        match result {
            Ok((current, history)) => {
                let mut lines = vec![format!(
                    "## {} (current: r{}, confidence: {:.2})\n{}\n",
                    current.name, current.revision, current.confidence, current.definition
                )];
                if history.is_empty() {
                    lines.push(
                        "No revision history (concept has not been refined yet).".to_string(),
                    );
                } else {
                    lines.push(format!(
                        "### Revision History ({} entries)\n",
                        history.len()
                    ));
                    for rev in &history {
                        let ep = rev.episode_id.as_deref().unwrap_or("none");
                        lines.push(format!(
                            "- **r{}** ({}) [episode: {}]\n  {}\n",
                            rev.revision,
                            rev.created_at.format("%Y-%m-%d %H:%M"),
                            ep,
                            rev.definition.chars().take(200).collect::<String>()
                        ));
                    }
                }
                lines.join("\n")
            }
            Err(e) => format!("Error: {e}"),
        }
    }

    /// Show adaptive engine status: learned parameters, weights, cluster info.
    #[tool(
        name = "rein_adaptive_status",
        description = "Show adaptive engine status: learned alphas, reranker weights, cluster info, tier boundaries, event counts, survival curve summaries."
    )]
    fn rein_adaptive_status(&self) -> String {
        self.non_store_count.fetch_add(1, Ordering::Relaxed);
        let result = self.with_store(|store| Ok(crate::ops::adaptive_status(store)));
        match result {
            Ok(status) => {
                serde_json::to_string_pretty(&status).unwrap_or_else(|e| format!("Error: {e}"))
            }
            Err(e) => format!("Error: {e}"),
        }
    }
}

/// Parse a date string as start-of-day (for `from` bounds).
fn parse_datetime(s: &str) -> Option<chrono::DateTime<chrono::Utc>> {
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(s) {
        return Some(dt.with_timezone(&chrono::Utc));
    }
    if let Ok(date) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        let dt = date.and_hms_opt(0, 0, 0)?;
        return Some(chrono::DateTime::<chrono::Utc>::from_naive_utc_and_offset(
            dt,
            chrono::Utc,
        ));
    }
    None
}

/// Parse a date string as end-of-day (for `to` bounds).
/// YYYY-MM-DD becomes 23:59:59 so the entire day is included.
fn parse_datetime_end(s: &str) -> Option<chrono::DateTime<chrono::Utc>> {
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(s) {
        return Some(dt.with_timezone(&chrono::Utc));
    }
    if let Ok(date) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        let dt = date.and_hms_opt(23, 59, 59)?;
        return Some(chrono::DateTime::<chrono::Utc>::from_naive_utc_and_offset(
            dt,
            chrono::Utc,
        ));
    }
    None
}

#[tool_handler]
impl ServerHandler for ReinServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions("rein: Multi-source cross-validated memory for AI agents. Use rein_store to save important context and rein_recall to search memories.")
    }
}

/// Spawn background warmup task for embedding cache pre-computation.
fn spawn_background_warmup(config: &ReinConfig) {
    let warmup_config = config.clone();
    // SqliteStore is not Send, so we must run the entire warmup (including async embed)
    // inside spawn_blocking. Use a dedicated current-thread runtime for the async parts.
    tokio::task::spawn_blocking(move || {
        if let Ok(store) = warmup_config.open_store() {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build();
            if let Ok(rt) = rt {
                rt.block_on(crate::search::warmup::warmup(&store, &warmup_config));
            }
        }
    });
}

/// Start the MCP server over stdio.
pub async fn run_stdio(config: ReinConfig) -> anyhow::Result<()> {
    spawn_background_warmup(&config);

    let server = ReinServer::new(config);

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
    spawn_background_warmup(&config);

    use http_body_util::BodyExt;
    use rmcp::transport::streamable_http_server::{
        session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
    };
    use std::sync::Arc;
    use tokio_util::sync::CancellationToken;

    let bind = format!("{}:{}", config.server.sse_bind, config.server.sse_port);
    let config_clone = config.clone();

    // Bearer token authentication
    let auth_token = std::env::var("REIN_HTTP_TOKEN")
        .ok()
        .filter(|token| !token.trim().is_empty());
    let allow_loopback_unauth = config.server.allow_unauthenticated_loopback
        && (config.server.sse_bind == "127.0.0.1"
            || config.server.sse_bind == "::1"
            || config.server.sse_bind == "localhost");
    if auth_token.is_none() && !allow_loopback_unauth {
        return Err(anyhow::anyhow!(
            "REIN_HTTP_TOKEN must be set for HTTP/SSE access on '{}'. \
             Set REIN_HTTP_TOKEN=<secret> or explicitly opt into unauthenticated loopback with [server].allow_unauthenticated_loopback=true",
            config.server.sse_bind
        ));
    }

    let cancel = CancellationToken::new();

    let session_manager = Arc::new(LocalSessionManager::default());
    let http_config = StreamableHttpServerConfig::default()
        .with_stateful_mode(true)
        .with_cancellation_token(cancel.clone());

    // NOTE: Each HTTP session creates its own ReinServer with a separate SqliteStore.
    // The Mutex only serializes within a single session (MCP handles one request at a time).
    // Cross-session concurrency is handled by having independent connections.
    let service = StreamableHttpService::new(
        move || {
            let server = ReinServer::new(config_clone.clone());
            Ok(server)
        },
        session_manager,
        http_config,
    );

    let listener = match tokio::net::TcpListener::bind(&bind).await {
        Ok(l) => l,
        Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => {
            eprintln!("Error: port {} is already in use.", config.server.sse_port);
            eprintln!("Another rein instance may be running. Check with:");
            eprintln!("  lsof -i :{}", config.server.sse_port);
            eprintln!("Kill it first, or use a different port via REIN_SSE_PORT.");
            return Err(anyhow::anyhow!(
                "port {} already in use",
                config.server.sse_port
            ));
        }
        Err(e) => return Err(e.into()),
    };
    tracing::info!("rein HTTP server listening on {bind}");
    eprintln!("rein HTTP server listening on http://{bind}/mcp");
    if config.server.gui_enabled {
        eprintln!("Neural Wiki GUI available at http://{bind}/");
    }
    // Write PID file for service management (rein gui on/off, rein dashboard).
    let _ = crate::service::write_pid("gui");

    let rest_config = config.clone();
    let service = hyper::service::service_fn(move |req: hyper::Request<_>| {
        let svc = service.clone();
        let token = auth_token.clone();
        let cfg = rest_config.clone();
        async move {
            // Check bearer token for API/MCP paths only (GUI static assets are public
            // so the SPA can bootstrap and show a token input dialog).
            //
            // DELETE /api/session is exempted so a browser holding a stale/invalid
            // cookie can clear it without being 401-blocked — otherwise the SPA
            // would be stuck resending the bad cookie until manual browser-data wipe.
            let path = req.uri().path();
            let method = req.method();
            let is_clear_session = method == hyper::Method::DELETE && path == "/api/session";
            let needs_auth =
                !is_clear_session && (path.starts_with("/api/") || path.starts_with("/mcp"));
            if needs_auth {
                if let Some(ref expected) = token {
                    if !request_has_valid_http_auth(req.headers(), expected) {
                        return Ok::<_, std::convert::Infallible>(
                            hyper::Response::builder()
                                .status(401)
                                .body(
                                    http_body_util::Full::new(bytes::Bytes::from("Unauthorized"))
                                        .map_err(|never: std::convert::Infallible| match never {})
                                        .boxed(),
                                )
                                .unwrap(),
                        );
                    }
                }
            } // needs_auth
              // Try REST API / GUI routes before MCP dispatch
            if let Some(response) = crate::mcp::rest::handle_rest_request(&req, &cfg).await {
                return Ok::<_, std::convert::Infallible>(response);
            }

            Ok::<_, std::convert::Infallible>(svc.handle(req).await)
        }
    });

    loop {
        let (stream, addr) = listener.accept().await?;
        tracing::debug!("connection from {addr}");
        let svc = service.clone();
        tokio::spawn(async move {
            if let Err(e) =
                hyper_util::server::conn::auto::Builder::new(hyper_util::rt::TokioExecutor::new())
                    .serve_connection(hyper_util::rt::TokioIo::new(stream), svc)
                    .await
            {
                tracing::warn!("connection error: {e}");
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_auth_accepts_matching_cookie() {
        let mut headers = hyper::HeaderMap::new();
        headers.insert(
            "cookie",
            hyper::header::HeaderValue::from_static("rein_http_token=secret-token"),
        );
        assert!(request_has_valid_http_auth(&headers, "secret-token"));
    }

    #[test]
    fn request_auth_rejects_empty_cookie_value() {
        let mut headers = hyper::HeaderMap::new();
        headers.insert(
            "cookie",
            hyper::header::HeaderValue::from_static("rein_http_token="),
        );
        assert!(!request_has_valid_http_auth(&headers, "secret-token"));
    }
}
