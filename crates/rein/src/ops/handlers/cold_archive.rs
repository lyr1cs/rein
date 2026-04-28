//! Cap C v0.26: manual archival-summary refresh op handler.
//!
//! Wraps [`crate::ops::cold_archive_summary::refresh_one_for_handler`] in a
//! `#[op]`-registered surface so operators can regenerate the archival
//! summary for a single cold-tier memory via CLI / MCP / REST. The
//! background slow-channel worker
//! ([`crate::ops::cold_archive_summary::run_cold_archive_summary`]) handles
//! the bulk path; this handler exists for targeted retry after a contract
//! diagnosis or for one-off operator inspection.
//!
//! ## Wave 3 wiring (per spec §9):
//!
//! 1. `crates/rein/src/ops/handlers/mod.rs` — add `pub mod cold_archive;`
//!    so this module is compiled in.
//! 2. `crates/rein/src/ops/mod.rs` — add `pub mod cold_archive_summary;`
//!    so `inventory::iter` picks up our `#[op]` registration.
//!
//! Both are intentionally deferred to Wave 3's sequential editor per spec
//! §7.1 conflicts #6 + #7.

use clap::Args;
use rein_macros::op;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::ops::cold_archive_summary::{ColdArchiveConfig, ManualRefreshOutcome};
use crate::ops::{IntoCliText, IntoJson, IntoMarkdown, OpsRuntime};
use crate::types::ReinResult;

/// Parameters for the manual `archive_summary_refresh` op.
#[derive(Args, Deserialize, JsonSchema, Debug, Clone, Default)]
pub struct ArchiveSummaryRefreshParams {
    /// Memory id whose archival summary to regenerate. The memory must be
    /// in cold tier; warm/hot rows are skipped with `generated = false`
    /// and a `skipped_reason` carrying the rejected tier.
    pub memory_id: String,
    /// When true, regenerate even if `archival_summary_version` already
    /// matches the current `ARCHIVAL_SUMMARY_VERSION`. Default false so
    /// repeated invocations are idempotent and cheap.
    #[serde(default)]
    #[arg(long)]
    pub force: bool,
}

/// Output of the manual `archive_summary_refresh` op. Wraps the worker's
/// `ManualRefreshOutcome` to attach surface-specific render impls without
/// colliding with any future impls on the bare outcome type (mirrors the
/// `ConceptSummaryRefreshOutput` pattern in `handlers/knowledge.rs`).
#[derive(Serialize, Clone, Debug)]
pub struct ArchiveSummaryRefreshOutput {
    pub memory_id: String,
    pub generated: bool,
    pub version: u32,
    pub summary_chars: usize,
    pub skipped_reason: Option<String>,
}

impl ArchiveSummaryRefreshOutput {
    fn from_outcome(memory_id: String, outcome: ManualRefreshOutcome) -> Self {
        Self {
            memory_id,
            generated: outcome.generated,
            version: outcome.version,
            summary_chars: outcome.summary_chars,
            skipped_reason: outcome.skipped_reason,
        }
    }
}

impl IntoJson for ArchiveSummaryRefreshOutput {
    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

impl IntoMarkdown for ArchiveSummaryRefreshOutput {
    fn to_markdown(&self) -> String {
        if let Some(reason) = &self.skipped_reason {
            return format!(
                "Cap C archive_summary_refresh skipped for {}: {}",
                self.memory_id, reason
            );
        }
        format!(
            "Cap C archive_summary_refresh: memory {} generated v{} ({} chars)",
            self.memory_id, self.version, self.summary_chars
        )
    }
}

impl IntoCliText for ArchiveSummaryRefreshOutput {
    fn to_cli_text(&self) -> String {
        self.to_markdown()
    }
}

impl OpsRuntime {
    #[op(
        name = "archive_summary_refresh",
        category = "maintenance",
        description = "Cap C v0.26: manually regenerate the archival summary for a single cold-tier memory. Requires `[ars].cold_archive_enabled = true`. Use force=true to regenerate even when the existing summary is at the current version.",
        mutating = true,
        cli(name = "archive-summary-refresh"),
        mcp(name = "rein_archive_summary_refresh"),
        rest(method = "POST", path = "/api/cold_archive/refresh"),
        auth = "mutation_marker"
    )]
    pub fn archive_summary_refresh(
        &self,
        params: ArchiveSummaryRefreshParams,
    ) -> ReinResult<ArchiveSummaryRefreshOutput> {
        let memory_id = params.memory_id.clone();
        let force = params.force;
        let config = self.config.clone();
        let cold_config = ColdArchiveConfig::from_ars(&config.ars);

        self.with_store(|store| {
            let outcome = crate::ops::cold_archive_summary::refresh_one_for_handler(
                store,
                &config,
                &cold_config,
                &memory_id,
                force,
            )?;
            Ok(ArchiveSummaryRefreshOutput::from_outcome(
                memory_id.clone(),
                outcome,
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tempfile::TempDir;

    /// Build a runtime + temp DB directory. Cap C is disabled by default
    /// (matches `[ars].cold_archive_enabled = false` in `ArsConfig::default`).
    fn runtime_with_empty_db() -> (Arc<OpsRuntime>, TempDir) {
        let tmp = TempDir::new().expect("tempdir");
        let mut config = crate::config::ReinConfig::default();
        config.database.path = tmp
            .path()
            .join("memories.db")
            .to_string_lossy()
            .into_owned();
        let config = Arc::new(config);
        // Touch the store to ensure schema migrations run (`init_schema`
        // owns the new columns; A_SCHEMA's branch is required for this
        // path to compile against the cold_archive_summary columns).
        let _ = config.open_store().expect("open store applies migrations");
        (Arc::new(OpsRuntime::for_rest(config)), tmp)
    }

    #[test]
    fn refresh_skips_when_cap_c_disabled() {
        let (runtime, _tmp) = runtime_with_empty_db();
        // Default ReinConfig has `cold_archive_enabled = false`.
        let out = runtime
            .archive_summary_refresh(ArchiveSummaryRefreshParams {
                memory_id: "doesnotexist".to_string(),
                force: false,
            })
            .expect("call must succeed (skipped, not errored)");
        assert!(!out.generated);
        let reason = out.skipped_reason.expect("must surface skipped_reason");
        assert!(
            reason.contains("cold_archive_enabled = false"),
            "expected disabled-flag reason, got: {reason}"
        );
        assert_eq!(out.memory_id, "doesnotexist");
        assert_eq!(
            out.version,
            crate::ops::cold_archive_summary::ARCHIVAL_SUMMARY_VERSION
        );
    }

    #[test]
    fn refresh_output_render_impls_round_trip() {
        // Tests the IntoJson / IntoMarkdown / IntoCliText impls without
        // hitting the live LLM path.
        let out = ArchiveSummaryRefreshOutput {
            memory_id: "m1".to_string(),
            generated: true,
            version: 100,
            summary_chars: 580,
            skipped_reason: None,
        };
        let json = out.to_json();
        assert_eq!(json["generated"], true);
        assert_eq!(json["version"], 100);
        assert_eq!(json["summary_chars"], 580);
        let md = out.to_markdown();
        assert!(md.contains("memory m1 generated v100"));
        assert!(md.contains("580 chars"));

        let skipped = ArchiveSummaryRefreshOutput {
            memory_id: "m2".to_string(),
            generated: false,
            version: 100,
            summary_chars: 0,
            skipped_reason: Some("not cold tier".to_string()),
        };
        let md = skipped.to_markdown();
        assert!(md.contains("skipped"));
        assert!(md.contains("not cold tier"));
    }

    #[test]
    fn params_deserializes_with_default_force_false() {
        let params: ArchiveSummaryRefreshParams =
            serde_json::from_value(serde_json::json!({"memory_id": "m1"})).expect("default force");
        assert_eq!(params.memory_id, "m1");
        assert!(!params.force);

        let params: ArchiveSummaryRefreshParams =
            serde_json::from_value(serde_json::json!({"memory_id": "m2", "force": true}))
                .expect("explicit force");
        assert!(params.force);
    }
}
