//! v0.27 ARS Cap A feedback loop: `rein_feedback_concept_summary` op.
//!
//! Mirrors the v0.26 D direction `rein_feedback` synthesis-interaction branch
//! (in `ops/handlers/adaptive.rs::feedback_synthesis_interaction`) for the
//! Cap A surface. Records a single
//! [`crate::store::adaptive::EventType::ConceptSummaryInteraction`] event,
//! which is later drained by the `concept_summary_feedback` consumer
//! (`store::adaptive::recompute_concept_summary_feedback_stats`) and folded
//! into [`crate::store::adaptive::ConceptSummaryFeedbackState`].
//!
//! Distinct MCP tool from `rein_feedback` (operator-defined boundary in the
//! Track 1 brief: do NOT extend the existing tagged-union dispatch). The
//! Cap A and Cap B feedback streams are deliberately separate at the API
//! surface so they can be enabled / observed independently.

use rein_macros::op;
use serde::{Deserialize, Serialize};

use crate::ops::handlers::adaptive::FeedbackOutput;
use crate::ops::OpsRuntime;
use crate::types::{OpsErrorKind, ReinError, ReinResult};

/// Parameters for the `rein_feedback_concept_summary` /
/// `POST /api/feedback/concept_summary` op.
///
/// Emitted by GUI hooks (dwell timer / source clicks / explicit thumb /
/// immediate requery) on the concept living-summary surface.
#[derive(Deserialize, Serialize, schemars::JsonSchema, Debug, Clone)]
pub struct ConceptSummaryFeedbackParams {
    /// Concept's persistent id (NOT a per-call ULID — concepts span sessions).
    pub concept_id: String,
    /// Per-refresh concept-summary ULID when the client knows it. Older
    /// clients omit this and the consumer falls back to `concept_id`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub concept_summary_id: Option<String>,
    /// Back-compat alias used by concept-state clients. Folded into
    /// `concept_summary_id` by the handler when the canonical field is absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub living_summary_id: Option<String>,
    /// ULID echoing `RecallMemoryOutput.request_id` so the back-end can join
    /// downstream recall traces with concept-summary interactions when the
    /// surface was reached via recall.
    pub recall_id: String,
    /// Discriminated kind tagged via `interaction.kind`. See
    /// `crate::store::adaptive::ConceptSummaryInteractionKind` for the
    /// variant set.
    pub interaction: crate::store::adaptive::ConceptSummaryInteractionKind,
    /// Optional metadata used for cluster bucketing in the consumer
    /// (query_type / cluster_id / concept_chars / revision_version).
    #[serde(default)]
    pub metadata: Option<crate::store::adaptive::ConceptSummaryMetadata>,
}

impl OpsRuntime {
    #[op(
        name = "feedback_concept_summary",
        category = "adaptive",
        description = "Record a user interaction on a Cap A concept living-summary surface (dwell / click / thumb / immediate-requery). Drains into the `concept_summary_feedback` consumer that powers the v0.27 per-query Cap-A gate.",
        mutating = true,
        mcp(name = "rein_feedback_concept_summary"),
        rest(method = "POST", path = "/api/feedback/concept_summary"),
        auth = "mutation_marker"
    )]
    pub fn feedback_concept_summary(
        &self,
        params: ConceptSummaryFeedbackParams,
    ) -> ReinResult<FeedbackOutput> {
        // codex R1 P2 (D1/D2 follow-up): rewrite the interaction's
        // `(cluster_id, query_type)` to the SAME synthetic per-concept
        // bucket the writer (`enqueue_judge_for_concept_summary`) and
        // gate reader (`decide_concept_summary_quality` from
        // `concept_state`) use. GUI clients still POST with their real
        // recall-context metadata (e.g. `query_type = "Exploratory"`,
        // `cluster_id = 7`), but Cap A pools all auto-judge events under
        // `(synthetic_hash(concept_id), "concept_refresh")`. Without
        // this override, human thumbs/dwell from GUI traffic accumulate
        // in `(7, "Exploratory")` and never influence the gate that
        // reads from the synthetic bucket — the per-concept feedback
        // loop is dead code for current clients. Per spec §15 R7-#1,
        // proper recall-context routing is v0.28+ work; until then,
        // unifying both writers under the synthetic key keeps the loop
        // closed.
        let synthetic_cid =
            crate::ops::concept_summary::synthetic_cluster_id_for_concept(&params.concept_id);
        let aligned_metadata = match &params.metadata {
            Some(meta) => {
                let mut m = meta.clone();
                m.cluster_id = Some(synthetic_cid);
                m.query_type = Some(
                    crate::ops::concept_summary::CONCEPT_SUMMARY_QUERY_TYPE_REFRESH.to_string(),
                );
                Some(m)
            }
            None => Some(crate::store::adaptive::ConceptSummaryMetadata {
                query_type: Some(
                    crate::ops::concept_summary::CONCEPT_SUMMARY_QUERY_TYPE_REFRESH.to_string(),
                ),
                cluster_id: Some(synthetic_cid),
                ..crate::store::adaptive::ConceptSummaryMetadata::default()
            }),
        };
        // Mirror the synthesis-interaction handler shape: lift query_type onto
        // the FeedbackEvent column so consumers can filter via the indexed
        // column without parsing the payload JSON for every event.
        let query_type = aligned_metadata.as_ref().and_then(|m| m.query_type.clone());
        let concept_summary_id = params
            .concept_summary_id
            .clone()
            .or_else(|| params.living_summary_id.clone())
            .or_else(|| Some(params.concept_id.clone()));

        let payload = crate::store::adaptive::ConceptSummaryInteractionPayload {
            concept_id: params.concept_id.clone(),
            concept_summary_id,
            recall_id: params.recall_id.clone(),
            interaction: params.interaction.clone(),
            metadata: aligned_metadata,
        };
        let payload_value = serde_json::to_value(&payload).map_err(|e| {
            ReinError::Config(format!(
                "concept_summary interaction payload serialize: {e}"
            ))
            .with_kind(OpsErrorKind::Internal)
        })?;

        self.with_store(|store| {
            let conn = store.conn();
            // BEGIN IMMEDIATE for symmetry with the access + synthesis-interaction
            // paths. The single emit_event INSERT does not strictly need a
            // transaction, but the wrap keeps failure semantics identical
            // (rollback on err) and prevents a partial offset advance if a
            // future change adds a second statement to this branch.
            conn.execute_batch("BEGIN IMMEDIATE")?;
            // `concept_id` is hoisted onto the FeedbackEvent column (mirrors
            // how `concept_summary_refreshed` does) so future per-concept
            // queries can index the column directly. `request_id`,
            // `memory_id`, `query`, `topic` left None per contract — the
            // recall_id and concept_id are carried via the payload only.
            let result = crate::store::adaptive::emit_event(
                conn,
                crate::store::adaptive::FeedbackEvent {
                    event_type: crate::store::adaptive::EventType::ConceptSummaryInteraction,
                    request_id: None,
                    memory_id: None,
                    concept_id: Some(params.concept_id.clone()),
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip the canonical params shape through JSON. Catches any future
    /// serde rename / tag drift on the params struct itself (the underlying
    /// kind enum is round-tripped in `store/adaptive.rs` tests).
    #[test]
    fn concept_summary_feedback_params_round_trip_serde() {
        let p = ConceptSummaryFeedbackParams {
            concept_id: "con-1".into(),
            concept_summary_id: Some("cs-1".into()),
            living_summary_id: None,
            recall_id: "rec-1".into(),
            interaction: crate::store::adaptive::ConceptSummaryInteractionKind::Viewed {
                dwell_ms: 4200,
            },
            metadata: Some(crate::store::adaptive::ConceptSummaryMetadata {
                query_type: Some("Semantic".into()),
                cluster_id: Some(42),
                concept_chars: Some(800),
                revision_version: Some(3),
            }),
        };
        let json = serde_json::to_string(&p).unwrap();
        let back: ConceptSummaryFeedbackParams = serde_json::from_str(&json).unwrap();
        assert_eq!(back.concept_id, p.concept_id);
        assert_eq!(back.concept_summary_id, p.concept_summary_id);
        assert_eq!(back.recall_id, p.recall_id);
        assert_eq!(back.metadata.unwrap().cluster_id, Some(42));
    }

    #[test]
    fn concept_summary_feedback_params_accept_living_summary_alias() {
        let json = serde_json::json!({
            "concept_id": "con-x",
            "living_summary_id": "cs-x",
            "recall_id": "rec-x",
            "interaction": {"kind": "explicit_thumb", "up": true}
        });
        let parsed: ConceptSummaryFeedbackParams =
            serde_json::from_value(json).expect("living_summary_id alias should parse");
        assert_eq!(parsed.concept_summary_id, None);
        assert_eq!(parsed.living_summary_id.as_deref(), Some("cs-x"));
    }

    /// JsonSchema derive sanity check: the schema must not panic to render.
    #[test]
    fn concept_summary_feedback_params_jsonschema_renders() {
        let schema = schemars::schema_for!(ConceptSummaryFeedbackParams);
        let value = serde_json::to_value(&schema).expect("schema serializes to JSON");
        // Confirm the params object lists at minimum `concept_id`,
        // `recall_id`, and `interaction` as required.
        let required = value
            .pointer("/required")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let names: std::collections::HashSet<&str> =
            required.iter().filter_map(|v| v.as_str()).collect();
        for must in &["concept_id", "recall_id", "interaction"] {
            assert!(
                names.contains(must),
                "expected required field {must} in schema, got {names:?}"
            );
        }
    }

    /// Back-compat: `metadata` is optional, so a payload without it
    /// deserializes to `None`.
    #[test]
    fn concept_summary_feedback_params_metadata_optional() {
        let json = serde_json::json!({
            "concept_id": "con-x",
            "recall_id": "rec-x",
            "interaction": {"kind": "viewed", "dwell_ms": 1000_u64}
        });
        let parsed: ConceptSummaryFeedbackParams =
            serde_json::from_value(json).expect("missing metadata must parse to None");
        assert_eq!(parsed.concept_id, "con-x");
        assert!(parsed.metadata.is_none());
    }
}
