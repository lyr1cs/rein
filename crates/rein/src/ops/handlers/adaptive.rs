//! Adaptive-category op handlers (Phase 2.1: adaptive_status).

use rein_macros::op;
use serde::Serialize;

use crate::ops::render::render_value_as_markdown;
use crate::ops::{IntoCliText, IntoJson, IntoMarkdown, OpsRuntime};
use crate::types::ReinResult;

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

impl OpsRuntime {
    #[op(
        name = "adaptive_status",
        category = "adaptive",
        description = "Show adaptive engine status: learned alphas, reranker weights, cluster info, tier boundaries, event counts, survival curve summaries.",
        cli(name = "adaptive-status"),
        mcp(name = "rein_adaptive_status"),
        rest(method = "GET", path = "/api/adaptive"),
    )]
    pub fn adaptive_status(&self) -> ReinResult<AdaptiveStatusOutput> {
        let value = self.with_store(|s| Ok(crate::ops::adaptive_status(s)))?;
        Ok(AdaptiveStatusOutput(value))
    }
}
