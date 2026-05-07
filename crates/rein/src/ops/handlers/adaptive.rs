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

/// Read-only release/eval gate report for ARS acceleration rollout decisions.
#[derive(Serialize, Clone, Debug)]
#[serde(transparent)]
pub struct ArsAccelerationReleaseGateOutput(
    pub crate::ops::ars_release_gate::ArsAccelerationReleaseGateReport,
);

impl IntoJson for ArsAccelerationReleaseGateOutput {
    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(&self.0).unwrap_or(serde_json::Value::Null)
    }
}

impl IntoMarkdown for ArsAccelerationReleaseGateOutput {
    fn to_markdown(&self) -> String {
        let report = &self.0;
        let mut lines = vec![
            "ARS acceleration release gate".to_string(),
            format!(
                "canary: {}",
                if report.canary.allowed {
                    "allowed"
                } else {
                    "blocked"
                }
            ),
            format!(
                "default-on: {}",
                if report.default_on.allowed {
                    "allowed"
                } else {
                    "blocked"
                }
            ),
        ];
        if !report.canary.blockers.is_empty() {
            lines.push(format!(
                "canary blockers: {}",
                report.canary.blockers.join(", ")
            ));
        }
        if !report.default_on.blockers.is_empty() {
            lines.push(format!(
                "default-on blockers: {}",
                report.default_on.blockers.join(", ")
            ));
        }
        if !report.canary.warnings.is_empty() {
            lines.push(format!("warnings: {}", report.canary.warnings.join(", ")));
        }
        lines.push(format!(
            "policy: status={} mode={} live_allowed={}",
            report.signals.policy_status,
            report.signals.policy_mode.as_deref().unwrap_or("unknown"),
            report.signals.policy_allows_runtime,
        ));
        lines.join("\n")
    }
}

impl IntoCliText for ArsAccelerationReleaseGateOutput {
    fn to_cli_text(&self) -> String {
        IntoMarkdown::to_markdown(self)
    }
}

// ── feedback ─────────────────────────────────────────────────────────────────

/// Pre-v0.26 access-feedback shape. Existing callers POST this without a `kind`
/// field; the manual `Deserialize` impl on `FeedbackParams` (below) defaults
/// missing `kind` to `"access"` so the back-compat surface is preserved.
#[derive(Deserialize, Serialize, schemars::JsonSchema, Debug, Clone)]
pub struct AccessFeedbackParams {
    /// Memory IDs that were actually used by the agent.
    pub memory_ids: Vec<String>,
    /// The request_id from the recall result (for attribution).
    #[serde(default)]
    pub request_id: Option<String>,
    /// Optional: the query that produced these results.
    #[serde(default)]
    pub query: Option<String>,
    /// Optional: whether the recall was helpful overall.
    #[serde(default)]
    pub helpful: Option<bool>,
}

/// v0.26 D direction: user interacted with a Cap B synthesis prose surface.
/// Emitted by GUI hooks (dwell timer / source clicks / explicit thumb / immediate
/// requery) and drained by the `synthesis_feedback` consumer (B_FEEDBACK_EVENT,
/// `store/adaptive.rs::recompute_synthesis_feedback_stats`).
#[derive(Deserialize, Serialize, schemars::JsonSchema, Debug, Clone)]
pub struct SynthesisInteractionFeedbackParams {
    /// ULID of the synthesis output (echoes `RecallSynthesisOutcome.synthesis_id`).
    pub synthesis_id: String,
    /// ULID of the originating `rein_recall` call (echoes
    /// `RecallMemoryOutput.request_id`). Used to correlate interaction events
    /// with the recall that produced them.
    pub recall_id: String,
    /// Discriminated kind tagged via `interaction.kind`. See
    /// `crate::store::adaptive::SynthesisInteractionKind` for the variant set.
    pub interaction: crate::store::adaptive::SynthesisInteractionKind,
    /// Optional metadata used for cluster bucketing in the consumer
    /// (query_type / cluster_id / source_count / synthesis_chars).
    #[serde(default)]
    pub metadata: Option<crate::store::adaptive::SynthesisMetadata>,
}

/// Parameters for the `rein_feedback` / `POST /api/feedback` operation.
///
/// **Tagged-union dispatch** (v0.26): the variant is selected by the optional
/// `kind` field on the JSON payload. Defaults to `"access"` for back-compat —
/// every pre-v0.26 caller (Neural Wiki GUI, MCP clients) posts the access
/// shape without a `kind` field and MUST keep working.
///
/// `Deserialize` is implemented manually because `#[serde(tag, untagged)]`
/// cannot model "tag is optional, defaults to a specific variant". `JsonSchema`
/// is also manual to match the dispatch semantics — the auto-derive would
/// declare `kind` as required and reject existing pre-v0.26 payloads in the
/// generated schema.
#[derive(Debug, Clone)]
pub enum FeedbackParams {
    /// Pre-v0.26 access feedback. Selected by `kind == "access"` OR by absence
    /// of the `kind` field entirely.
    Access(AccessFeedbackParams),
    /// v0.26 synthesis interaction. Selected by `kind == "synthesis_interaction"`.
    SynthesisInteraction(SynthesisInteractionFeedbackParams),
}

impl<'de> Deserialize<'de> for FeedbackParams {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Read once into a `Value` so we can both peek `kind` and deserialize
        // the matching inner shape. `serde_json::Value::deserialize` accepts
        // any deserializer (not just JSON), so this works in MCP (which feeds
        // a `serde_json::Value` through `serde_json::from_value`) and in REST
        // (which feeds raw JSON bytes through `serde_json::from_slice`).
        let value = serde_json::Value::deserialize(deserializer)?;
        // Distinguish missing-kind (back-compat default to "access") from
        // present-but-non-string-kind (must reject — silently coercing
        // `{"kind": 123}` to access masks client bugs).
        let kind: &str = match value.get("kind") {
            None => "access",
            Some(serde_json::Value::String(s)) => s,
            Some(other) => {
                return Err(serde::de::Error::custom(format!(
                    "feedback `kind` must be a string (got {})",
                    type_name_of_value(other)
                )));
            }
        };
        match kind {
            "access" => {
                let inner: AccessFeedbackParams =
                    serde_json::from_value(value).map_err(serde::de::Error::custom)?;
                Ok(FeedbackParams::Access(inner))
            }
            "synthesis_interaction" => {
                let inner: SynthesisInteractionFeedbackParams =
                    serde_json::from_value(value).map_err(serde::de::Error::custom)?;
                Ok(FeedbackParams::SynthesisInteraction(inner))
            }
            other => Err(serde::de::Error::custom(format!(
                "unknown feedback kind: {other:?} (expected \"access\" or \"synthesis_interaction\")"
            ))),
        }
    }
}

/// Render a JSON Value's variant name for diagnostic error messages.
fn type_name_of_value(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "bool",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

impl schemars::JsonSchema for FeedbackParams {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("FeedbackParams")
    }

    fn schema_id() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("rein::ops::handlers::adaptive::FeedbackParams")
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        // oneOf union: each branch references the inner type's schema. The
        // access branch's `kind` is documented as optional with default
        // `"access"` so existing callers (which omit `kind`) validate against
        // the schema without a phantom required-field rejection.
        let access_inner = generator.subschema_for::<AccessFeedbackParams>();
        let synthesis_inner = generator.subschema_for::<SynthesisInteractionFeedbackParams>();

        schemars::json_schema!({
            "type": "object",
            "title": "FeedbackParams",
            "description": "Tagged union dispatched on optional `kind` field. Missing `kind` defaults to `\"access\"` for pre-v0.26 back-compat.",
            "oneOf": [
                {
                    "title": "Access",
                    "description": "Pre-v0.26 access feedback (default when `kind` is absent).",
                    "allOf": [access_inner],
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": ["access"],
                            "default": "access",
                            "description": "Optional — omit to use the default access shape."
                        }
                    }
                },
                {
                    "title": "SynthesisInteraction",
                    "description": "v0.26 D-direction synthesis interaction event (dwell / click / thumb / immediate-requery).",
                    "allOf": [synthesis_inner],
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": ["synthesis_interaction"]
                        }
                    },
                    "required": ["kind"]
                }
            ]
        })
    }
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
        description = "Record user feedback on a memory (thumbs up/down, click, relevance score) — drives the self-learning adaptive engine. v0.26 also accepts `kind = \"synthesis_interaction\"` to record GUI dwell / click / thumb / immediate-requery events on Cap B synthesis prose.",
        mutating = true,
        mcp(name = "rein_feedback"),
        rest(method = "POST", path = "/api/feedback"),
        auth = "mutation_marker"
    )]
    pub fn feedback(&self, params: FeedbackParams) -> ReinResult<FeedbackOutput> {
        match params {
            FeedbackParams::Access(p) => self.feedback_access(p),
            FeedbackParams::SynthesisInteraction(p) => self.feedback_synthesis_interaction(p),
        }
    }
}

impl OpsRuntime {
    fn feedback_access(&self, params: AccessFeedbackParams) -> ReinResult<FeedbackOutput> {
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

    fn feedback_synthesis_interaction(
        &self,
        params: SynthesisInteractionFeedbackParams,
    ) -> ReinResult<FeedbackOutput> {
        // Emit a single SynthesisInteraction event. The consumer
        // (`recompute_synthesis_feedback_stats`) drains and aggregates per
        // (cluster_id, query_type) — both pulled from `metadata` — into
        // `AdaptiveState.synthesis_feedback_stats`. No store-level state
        // mutation here; the event itself IS the durable record.
        //
        // `query_type` is also lifted onto the FeedbackEvent column so the
        // consumer can route via the indexed column without parsing the
        // payload JSON for every event. Mirrors how `concept_summary_refresh`
        // sets `concept_id` on the event row alongside payload.
        let query_type = params.metadata.as_ref().and_then(|m| m.query_type.clone());

        let payload = crate::store::adaptive::SynthesisInteractionPayload {
            synthesis_id: params.synthesis_id.clone(),
            recall_id: params.recall_id.clone(),
            interaction: params.interaction.clone(),
            metadata: params.metadata.clone(),
        };
        let payload_value = serde_json::to_value(&payload).map_err(|e| {
            ReinError::Config(format!("synthesis interaction payload serialize: {e}"))
                .with_kind(OpsErrorKind::Internal)
        })?;

        self.with_store(|store| {
            let conn = store.conn();
            // BEGIN IMMEDIATE for symmetry with the access path. The single
            // emit_event INSERT does not strictly need a transaction, but the
            // wrap keeps the failure semantics identical (rollback on err)
            // and prevents a partial offset advance if a future change adds
            // a second statement to this branch.
            conn.execute_batch("BEGIN IMMEDIATE")?;
            // Per contract §4.2: `request_id = None`, `memory_id = None`,
            // `concept_id = None`, `query_type = metadata.query_type`. The
            // recall_id and synthesis_id are carried via the payload only;
            // hoisting recall_id onto the request_id column would diverge
            // from the contract's explicit field-by-field spec and risk
            // double-counting in any downstream consumer that joins on
            // request_id (e.g. M2 alpha-optimizer's recall-attribution).
            let result = crate::store::adaptive::emit_event(
                conn,
                crate::store::adaptive::FeedbackEvent {
                    event_type: crate::store::adaptive::EventType::SynthesisInteraction,
                    request_id: None,
                    memory_id: None,
                    concept_id: None,
                    query: None,
                    query_type,
                    topic: None,
                    payload: Some(payload_value),
                },
            );
            match result {
                Ok(_) => {
                    conn.execute_batch("COMMIT")?;
                    Ok(FeedbackOutput { emitted: 1 })
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
        let config = self.config.clone();
        let value = self.with_store(|s| Ok(crate::ops::adaptive_status_with_config(s, &config)))?;
        Ok(AdaptiveStatusOutput(value))
    }

    #[op(
        name = "ars_acceleration_release_gate",
        category = "adaptive",
        description = "Read-only release/eval gate report for ARS acceleration canary and default-on decisions. Uses existing config, adaptive snapshot, parameter-policy, and adaptive-status signals; never flips defaults or writes policy.",
        cli(name = "ars-acceleration-gate"),
        mcp(name = "rein_ars_acceleration_gate"),
        rest(method = "GET", path = "/api/ars-acceleration-gate")
    )]
    pub fn ars_acceleration_release_gate(&self) -> ReinResult<ArsAccelerationReleaseGateOutput> {
        let config = self.config.clone();
        let report = self.with_store(|s| {
            Ok(crate::ops::ars_release_gate::ars_acceleration_release_gate_report(s, &config))
        })?;
        Ok(ArsAccelerationReleaseGateOutput(report))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── FeedbackParams Deserialize: back-compat + new variant dispatch ──────

    /// Test #1: explicit `kind = "synthesis_interaction"` → dispatches to the
    /// SynthesisInteraction variant with a populated `interaction` body.
    #[test]
    fn feedback_params_deserialize_synthesis_interaction_variant() {
        let json = serde_json::json!({
            "kind": "synthesis_interaction",
            "synthesis_id": "01H9X8Y7Z6W5V4U3T2S1R0Q",
            "recall_id": "01H9X8Y7Z6W5V4U3T2S1R01",
            "interaction": {
                "kind": "viewed",
                "dwell_ms": 4200_u64
            },
            "metadata": {
                "query_type": "Semantic",
                "cluster_id": 42_i64,
                "source_count": 3_u32
            }
        });
        let parsed: FeedbackParams =
            serde_json::from_value(json).expect("synthesis_interaction kind must parse");
        match parsed {
            FeedbackParams::SynthesisInteraction(inner) => {
                assert_eq!(inner.synthesis_id, "01H9X8Y7Z6W5V4U3T2S1R0Q");
                assert_eq!(inner.recall_id, "01H9X8Y7Z6W5V4U3T2S1R01");
                assert!(inner.metadata.is_some());
                let md = inner.metadata.unwrap();
                assert_eq!(md.query_type.as_deref(), Some("Semantic"));
                assert_eq!(md.cluster_id, Some(42));
            }
            FeedbackParams::Access(_) => panic!("expected SynthesisInteraction, got Access"),
        }
    }

    /// Test #2: pre-v0.26 payload (no `kind` field) → defaults to Access
    /// variant. This is the back-compat invariant — any GUI / MCP client
    /// shipped before v0.26 MUST keep working unchanged.
    #[test]
    fn feedback_params_deserialize_back_compat_no_kind_defaults_to_access() {
        // Verbatim the shape every pre-v0.26 caller posts.
        let json = serde_json::json!({
            "memory_ids": ["mem_abc", "mem_xyz"],
            "request_id": "req_123",
            "query": "what did we decide last week",
            "helpful": true
        });
        let parsed: FeedbackParams =
            serde_json::from_value(json).expect("missing kind must default to Access");
        match parsed {
            FeedbackParams::Access(inner) => {
                assert_eq!(inner.memory_ids, vec!["mem_abc", "mem_xyz"]);
                assert_eq!(inner.request_id.as_deref(), Some("req_123"));
                assert_eq!(inner.query.as_deref(), Some("what did we decide last week"));
                assert_eq!(inner.helpful, Some(true));
            }
            FeedbackParams::SynthesisInteraction(_) => {
                panic!("missing kind must default to Access, not SynthesisInteraction")
            }
        }
    }

    /// Test #3: explicit `kind = "access"` → produces Access variant. Forward
    /// compat: callers that prefer to be explicit about the variant can pass
    /// `kind` and it round-trips.
    #[test]
    fn feedback_params_deserialize_explicit_access_kind() {
        let json = serde_json::json!({
            "kind": "access",
            "memory_ids": ["mem_only"],
            "request_id": null,
            "query": null,
            "helpful": false
        });
        let parsed: FeedbackParams =
            serde_json::from_value(json).expect("explicit access kind must parse");
        match parsed {
            FeedbackParams::Access(inner) => {
                assert_eq!(inner.memory_ids, vec!["mem_only"]);
                assert_eq!(inner.request_id, None);
                assert_eq!(inner.query, None);
                assert_eq!(inner.helpful, Some(false));
            }
            FeedbackParams::SynthesisInteraction(_) => {
                panic!("expected Access, got SynthesisInteraction")
            }
        }
    }

    /// Test #4 part A: empty memory_ids on Access still rejects with the
    /// pre-v0.26 BadRequest semantics. Catching this in unit tests prevents a
    /// silent contract drift if the manual Deserialize ever swallows the
    /// validation the access handler does.
    #[test]
    fn feedback_params_deserialize_empty_memory_ids_still_parses() {
        // Deserialize succeeds — validation happens in feedback_access.
        let json = serde_json::json!({
            "kind": "access",
            "memory_ids": []
        });
        let parsed: FeedbackParams =
            serde_json::from_value(json).expect("empty memory_ids parses (validated in handler)");
        match parsed {
            FeedbackParams::Access(inner) => assert!(inner.memory_ids.is_empty()),
            FeedbackParams::SynthesisInteraction(_) => panic!("wrong variant"),
        }
    }

    /// Test #4 part B: unknown kind returns a meaningful error (not a panic
    /// or silent default). Future variants will add explicit arms; until then
    /// we surface "unknown" so misconfigured clients fail loud.
    #[test]
    fn feedback_params_deserialize_unknown_kind_errors() {
        let json = serde_json::json!({
            "kind": "not_a_real_kind",
            "memory_ids": ["mem_x"]
        });
        let result: Result<FeedbackParams, _> = serde_json::from_value(json);
        assert!(
            result.is_err(),
            "unknown kind must error, not silently default"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("unknown feedback kind"),
            "error message should mention unknown feedback kind: {err_msg}"
        );
    }

    /// Test #4 part C: JsonSchema serialization renders the oneOf without
    /// panicking and includes both branches. Specifically asserts that the
    /// access branch documents `kind` as optional with the `"access"` default
    /// — without this, a strict JSON-Schema-validating client would reject
    /// every pre-v0.26 caller.
    #[test]
    fn feedback_params_jsonschema_renders_oneof_with_optional_kind_default() {
        let schema = schemars::schema_for!(FeedbackParams);
        let value = serde_json::to_value(&schema).expect("schema serializes to JSON");

        // The root schema (or a referenced subschema) must contain a `oneOf`
        // with two branches. We allow either inline `oneOf` at the root or
        // inside the schema's definitions/$defs (schemars 1.x emits the
        // top-level type that way for refs).
        let oneof = value
            .pointer("/oneOf")
            .or_else(|| value.pointer("/definitions/FeedbackParams/oneOf"))
            .or_else(|| value.pointer("/$defs/FeedbackParams/oneOf"))
            .expect("schema must expose a oneOf for the FeedbackParams variants");
        let branches = oneof.as_array().expect("oneOf must be an array");
        assert_eq!(
            branches.len(),
            2,
            "FeedbackParams schema must declare two variants (Access + SynthesisInteraction)"
        );

        // Locate the Access branch and confirm `kind` is documented optional
        // with default `"access"`. The branch order is stable per the manual
        // impl above (Access first), so we check by index.
        let access_branch = &branches[0];
        let kind_default = access_branch
            .pointer("/properties/kind/default")
            .expect("Access branch must document kind default");
        assert_eq!(kind_default, &serde_json::Value::String("access".into()));
        // Access branch must NOT mark `kind` required (back-compat: pre-v0.26
        // payloads omit it).
        let access_required = access_branch
            .pointer("/required")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        assert!(
            !access_required.iter().any(|v| v.as_str() == Some("kind")),
            "Access branch MUST NOT require `kind` (back-compat invariant)"
        );

        // SynthesisInteraction branch documents kind as required.
        let synth_branch = &branches[1];
        let synth_required = synth_branch
            .pointer("/required")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        assert!(
            synth_required.iter().any(|v| v.as_str() == Some("kind")),
            "SynthesisInteraction branch must require kind"
        );
    }
}
