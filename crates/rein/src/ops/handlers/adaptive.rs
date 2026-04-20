//! Adaptive-category op handlers (Phase 2.1: adaptive_status; Phase 2.4: feedback).

use rein_macros::op;
use serde::{Deserialize, Serialize};

use crate::ops::render::render_value_as_markdown;
use crate::ops::{IntoCliText, IntoJson, IntoMarkdown, OpsRuntime};
use crate::types::{OpsErrorKind, ReinError, ReinResult};

/// Wrapper around the untyped `ops::adaptive_status` JSON value so the three
/// render traits can be implemented without restructuring the existing pipeline.
/// Preserving the raw `Value` keeps the GUI `/api/adaptive` contract identical
/// to the pre-migration response shape.
#[derive(Serialize, Clone, Debug)]
#[serde(transparent)]
pub struct AdaptiveStatusOutput(pub serde_json::Value);

impl IntoJson for AdaptiveStatusOutput {
    fn to_json(&self) -> serde_json::Value {
        self.0.clone()
    }
}

impl IntoMarkdown for AdaptiveStatusOutput {
    fn to_markdown(&self) -> String {
        render_value_as_markdown(&self.0, 0)
    }
}

impl IntoCliText for AdaptiveStatusOutput {
    fn to_cli_text(&self) -> String {
        serde_json::to_string_pretty(&self.0).unwrap_or_else(|e| format!("Error: {e}"))
    }
}

// ── feedback ─────────────────────────────────────────────────────────────────

/// Parameters for the `rein_feedback` / `POST /api/feedback` operation.
#[derive(Deserialize, schemars::JsonSchema, Debug, Clone)]
pub struct FeedbackParams {
    /// Memory IDs that were actually used by the agent.
    pub memory_ids: Vec<String>,
    /// The request_id from the recall result (for attribution).
    pub request_id: Option<String>,
    /// Optional: the query that produced these results.
    pub query: Option<String>,
    /// Optional: whether the recall was helpful overall.
    pub helpful: Option<bool>,
}

/// Output shape for the feedback op.
#[derive(Serialize, Clone, Debug)]
pub struct FeedbackOutput {
    /// Number of feedback events emitted.
    pub emitted: u32,
}

impl IntoJson for FeedbackOutput {
    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

impl IntoMarkdown for FeedbackOutput {
    fn to_markdown(&self) -> String {
        // M1 compact contract: matches the pre-A1 legacy MCP compact branch
        // verbatim (`format!("ok:{count}")`) so MCP callers that parse this
        // string continue to work.
        format!("ok:{}", self.emitted)
    }
}

impl IntoCliText for FeedbackOutput {
    fn to_cli_text(&self) -> String {
        format!(
            "Feedback recorded for {} {}. This improves future recall quality.",
            self.emitted,
            if self.emitted == 1 {
                "memory"
            } else {
                "memories"
            }
        )
    }
}

impl OpsRuntime {
    #[op(
        name = "feedback",
        category = "adaptive",
        description = "Record user feedback on a memory (thumbs up/down, click, relevance score) — drives the self-learning adaptive engine.",
        mutating = true,
        mcp(name = "rein_feedback"),
        rest(method = "POST", path = "/api/feedback"),
        auth = "mutation_marker"
    )]
    pub fn feedback(&self, params: FeedbackParams) -> ReinResult<FeedbackOutput> {
        if params.memory_ids.is_empty() {
            return Err(ReinError::Config("memory_ids cannot be empty".into())
                .with_kind(OpsErrorKind::BadRequest));
        }

        self.with_store(|store| {
            let conn = store.conn();
            let mut emitted: u32 = 0;

            // F2: wrap the entire per-id batch in BEGIN IMMEDIATE so
            // concurrent consumers never see partial state. Errors propagate
            // via `?` and the ROLLBACK branch prevents a leaked open tx.
            conn.execute_batch("BEGIN IMMEDIATE")?;
            let result = (|| -> crate::types::ReinResult<u32> {
                for mem_id in &params.memory_ids {
                    store.record_access(mem_id)?;
                    crate::store::adaptive::emit_event(
                        conn,
                        crate::store::adaptive::FeedbackEvent {
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
                        },
                    )?;
                    emitted += 1;
                }
                Ok(emitted)
            })();
            match result {
                Ok(n) => {
                    conn.execute_batch("COMMIT")?;
                    Ok(FeedbackOutput { emitted: n })
                }
                Err(e) => {
                    let _ = conn.execute_batch("ROLLBACK");
                    Err(e)
                }
            }
        })
    }
}

impl OpsRuntime {
    #[op(
        name = "adaptive_status",
        category = "adaptive",
        description = "Show adaptive engine status: learned alphas, reranker weights, cluster info, tier boundaries, event counts, survival curve summaries.",
        cli(name = "adaptive-status"),
        mcp(name = "rein_adaptive_status"),
        rest(method = "GET", path = "/api/adaptive")
    )]
    pub fn adaptive_status(&self) -> ReinResult<AdaptiveStatusOutput> {
        let value = self.with_store(|s| Ok(crate::ops::adaptive_status(s)))?;
        Ok(AdaptiveStatusOutput(value))
    }
}
