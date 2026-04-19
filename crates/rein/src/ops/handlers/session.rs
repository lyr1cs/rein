//! Session-category op handlers (Phase 2.2: ingest_session).
//!
//! `rein_ingest_session` was the first MCP handler scheduled for A1 migration
//! in the session batch. It's the first real consumer of both the H3 auth
//! framework (POST with `auth = "mutation_marker"`) and the H5 JSON-body
//! prep path added in Phase 2.2 (macro branches on method to decode body
//! instead of query string).
//!
//! CLI surface stays on the legacy `rein ingest-session` command for now —
//! that subcommand reads files (`--file`, `--json-file`) which is shell-side
//! concern the op layer shouldn't touch. The op accepts content/turns
//! directly; CLI file reading happens before handing off.

use rein_macros::op;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::ops::{IntoCliText, IntoJson, IntoMarkdown, OpsRuntime};
use crate::types::{IngestReport, ReinError, ReinResult, SessionIngest, SessionTurn};

/// Maximum aggregate payload accepted by `ingest_session`. Mirrors the
/// pre-migration MCP handler's 500 KB ceiling — ingest artifacts above this
/// size are bugs upstream, not legitimate workloads.
pub const INGEST_SESSION_MAX_CHARS: usize = 500_000;
/// Maximum number of structured turns per ingest call.
pub const INGEST_SESSION_MAX_TURNS: usize = 1000;

#[derive(Debug, Deserialize, JsonSchema)]
pub struct IngestSessionInput {
    /// The full session/transcript text to ingest.
    #[serde(default)]
    pub content: Option<String>,
    /// Structured role/content turn list (alternative to `content`).
    #[serde(default)]
    pub turns: Option<Vec<IngestSessionTurnInput>>,
    /// Optional session title.
    #[serde(default)]
    pub title: Option<String>,
    /// Stable session identifier if the caller wants to pin one.
    #[serde(default)]
    pub session_id: Option<String>,
    /// RFC3339 session start time (or YYYY-MM-DD).
    #[serde(default)]
    pub started_at: Option<String>,
    /// Session summary / upstream synopsis.
    #[serde(default)]
    pub summary: Option<String>,
    /// Compact summary produced upstream (e.g. /compact output).
    #[serde(default)]
    pub compact_summary: Option<String>,
    /// Tool outputs associated with the session.
    #[serde(default)]
    pub tool_outputs: Option<Vec<String>>,
    /// Label for the originating agent/runtime.
    #[serde(default)]
    pub agent_label: Option<String>,
    /// Whether this session came from a subagent.
    #[serde(default)]
    pub is_subagent: Option<bool>,
    /// If true, persist the raw artifact and queue full processing in the
    /// background. Returns immediately with queued=true and counts=0.
    #[serde(default)]
    pub async_mode: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct IngestSessionTurnInput {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct IngestSessionOutput {
    pub queued: bool,
    pub artifact_id: Option<String>,
    pub session_id: Option<String>,
    pub episode_id: Option<String>,
    pub memory_count: u32,
    pub concept_count: u32,
    pub link_count: u32,
    pub turn_count: u32,
    pub filtered_count: u32,
    pub secret_filtered_count: u32,
    pub created_count: u32,
    pub merged_count: u32,
    pub superseded_count: u32,
    pub primary_topics: Vec<String>,
    pub stored_memory_ids: Vec<String>,
}

impl From<IngestReport> for IngestSessionOutput {
    fn from(r: IngestReport) -> Self {
        Self {
            queued: r.queued,
            artifact_id: r.artifact_id,
            session_id: r.session_id,
            episode_id: r.episode_id,
            memory_count: r.memory_count,
            concept_count: r.concept_count,
            link_count: r.link_count,
            turn_count: r.turn_count,
            filtered_count: r.filtered_count,
            secret_filtered_count: r.secret_filtered_count,
            created_count: r.created_count,
            merged_count: r.merged_count,
            superseded_count: r.superseded_count,
            primary_topics: r.primary_topics,
            stored_memory_ids: r.stored_memory_ids,
        }
    }
}

impl IntoJson for IngestSessionOutput {
    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

impl IntoMarkdown for IngestSessionOutput {
    fn to_markdown(&self) -> String {
        format!(
            "**Ingested session**\n\n- queued: {}\n- artifact: {}\n- episode: {}\n- memories: {}\n- concepts: {}\n- links: {}\n- turns: {}\n- filtered: {} ({} secret)\n",
            self.queued,
            self.artifact_id.as_deref().unwrap_or("-"),
            self.episode_id.as_deref().unwrap_or("-"),
            self.memory_count,
            self.concept_count,
            self.link_count,
            self.turn_count,
            self.filtered_count,
            self.secret_filtered_count,
        )
    }
}

impl IntoCliText for IngestSessionOutput {
    fn to_cli_text(&self) -> String {
        // Compact one-liner matches the pre-migration MCP text output so
        // `_ingest_ok`-style consumers that only inspect the prefix stay
        // unaffected. Full structured data is available via JSON.
        format!(
            "ok queued:{} memories:{} concepts:{} links:{} artifact:{} episode:{}",
            self.queued,
            self.memory_count,
            self.concept_count,
            self.link_count,
            self.artifact_id.as_deref().unwrap_or("-"),
            self.episode_id.as_deref().unwrap_or("-"),
        )
    }
}

impl OpsRuntime {
    #[op(
        name = "ingest_session",
        category = "session",
        description = "Ingest a full session transcript into memories, concepts, links, and an episode using the full extraction pipeline. Accepts `content` (raw text) or `turns` (structured role/content list).",
        mutating = true,
        mcp(name = "rein_ingest_session"),
        rest(method = "POST", path = "/api/ingest_session"),
        auth = "mutation_marker",
    )]
    pub async fn ingest_session(
        &self,
        params: IngestSessionInput,
    ) -> ReinResult<IngestSessionOutput> {
        let agent_label = params.agent_label.clone();
        let is_subagent = params.is_subagent.unwrap_or(false);
        let async_mode = params.async_mode.unwrap_or(false);

        let report = match (params.turns, params.content) {
            (Some(turns), _) => {
                let total_chars: usize =
                    turns.iter().map(|t| t.role.len() + t.content.len()).sum();
                if total_chars > INGEST_SESSION_MAX_CHARS {
                    return Err(ReinError::Config(format!(
                        "turns too large (max {} bytes aggregate)",
                        INGEST_SESSION_MAX_CHARS
                    ))
                    .with_kind(crate::types::OpsErrorKind::BadRequest));
                }
                if turns.len() > INGEST_SESSION_MAX_TURNS {
                    return Err(ReinError::Config(format!(
                        "too many turns (max {})",
                        INGEST_SESSION_MAX_TURNS
                    ))
                    .with_kind(crate::types::OpsErrorKind::BadRequest));
                }
                let started_at = params
                    .started_at
                    .as_deref()
                    .and_then(parse_datetime)
                    .or_else(|| params.started_at.as_deref().and_then(parse_datetime_end));
                let session = SessionIngest {
                    schema_version: 1,
                    artifact_kind: "session".to_string(),
                    session_id: params.session_id,
                    title: params.title,
                    started_at,
                    ended_at: None,
                    summary: params.summary,
                    source_agent: agent_label.clone(),
                    source_label: Some(self.surface_source_label().to_string()),
                    compact_summary: params.compact_summary,
                    tool_outputs: params.tool_outputs.unwrap_or_default(),
                    turns: turns
                        .into_iter()
                        .map(|t| SessionTurn {
                            role: t.role,
                            content: t.content,
                        })
                        .collect(),
                };
                let cfg = self.config.as_ref().clone();
                if async_mode {
                    crate::ops::queue_ingest_session(
                        &cfg,
                        &session,
                        agent_label.as_deref(),
                        is_subagent,
                    )?
                } else {
                    crate::ops::ingest_session_report(
                        &cfg,
                        &session,
                        agent_label.as_deref(),
                        is_subagent,
                    )
                    .await?
                }
            }
            (None, Some(content)) => {
                if content.len() > INGEST_SESSION_MAX_CHARS {
                    return Err(ReinError::Config(format!(
                        "content too large (max {} bytes)",
                        INGEST_SESSION_MAX_CHARS
                    ))
                    .with_kind(crate::types::OpsErrorKind::BadRequest));
                }
                let cfg = self.config.as_ref().clone();
                if async_mode {
                    crate::ops::queue_ingest_session_text(
                        &cfg,
                        &content,
                        agent_label.as_deref(),
                        is_subagent,
                    )?
                } else {
                    crate::ops::ingest_session_text_report(
                        &cfg,
                        &content,
                        agent_label.as_deref(),
                        is_subagent,
                    )
                    .await?
                }
            }
            (None, None) => {
                return Err(ReinError::Config(
                    "ingest_session requires either `content` or `turns`".to_string(),
                )
                .with_kind(crate::types::OpsErrorKind::BadRequest));
            }
        };

        Ok(report.into())
    }

    /// Tag artifacts with the surface they came through. Pre-migration the
    /// MCP handler hardcoded "mcp"; REST callers should be distinguishable
    /// because the raw artifact metadata is used later for dedup/audit.
    fn surface_source_label(&self) -> &'static str {
        match self.surface() {
            crate::ops::SurfaceKind::Mcp => "mcp",
            crate::ops::SurfaceKind::Rest => "rest",
            crate::ops::SurfaceKind::Cli => "cli",
        }
    }
}

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
