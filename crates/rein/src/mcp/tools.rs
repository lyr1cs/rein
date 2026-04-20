use rmcp::schemars;
use serde::de::Deserializer;
use serde::Deserialize;

// deserialize_option_usize_from_string removed with the last legacy memoir
// Params struct in Phase 2.6. Op-side Params types use standard serde via
// `#[serde(default)] Option<usize>` on clap-derived fields.

fn normalize_string_list(mut values: Vec<String>) -> Option<Vec<String>> {
    values = values
        .drain(..)
        .flat_map(|value| {
            value
                .split(',')
                .map(str::trim)
                .filter(|item| !item.is_empty())
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .collect();

    if values.is_empty() {
        None
    } else {
        Some(values)
    }
}

pub(crate) fn deserialize_option_string_list<'de, D>(deserializer: D) -> Result<Option<Vec<String>>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StringListValue {
        Many(Vec<String>),
        One(String),
    }

    match Option::<StringListValue>::deserialize(deserializer)? {
        None => Ok(None),
        Some(StringListValue::Many(values)) => Ok(normalize_string_list(values)),
        Some(StringListValue::One(value)) => Ok(normalize_string_list(vec![value])),
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SessionTurnParams {
    /// Role for the turn, e.g. user/assistant/tool/system.
    pub role: String,
    /// Text content of the turn.
    pub content: String,
}

// RecallParams removed — rein_recall migrated to #[op] inventory.
// See ops/handlers/memory.rs for the new RecallMemoryParams.

/// Parameters for rein_ingest_session tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct IngestSessionParams {
    /// The full session/transcript text to ingest.
    pub content: Option<String>,
    /// Optional structured turn list for richer ingestion.
    pub turns: Option<Vec<SessionTurnParams>>,
    /// Optional session title.
    pub title: Option<String>,
    /// Optional stable session identifier.
    pub session_id: Option<String>,
    /// Optional RFC3339 session start time.
    pub started_at: Option<String>,
    /// Optional session summary or compacted synopsis.
    pub summary: Option<String>,
    /// Optional compact summary produced upstream.
    pub compact_summary: Option<String>,
    /// Optional tool outputs associated with the session.
    pub tool_outputs: Option<Vec<String>>,
    /// Optional label for the originating agent/runtime.
    pub agent_label: Option<String>,
    /// Whether this session came from a subagent.
    pub is_subagent: Option<bool>,
    /// If true, persist the raw artifact and queue full processing in the background.
    pub async_mode: Option<bool>,
}

// FeedbackParams removed — rein_feedback migrated to #[op] inventory (Phase 2.4).
// See ops/handlers/adaptive.rs for the new FeedbackParams + FeedbackOutput structs.

// HealthParams removed in A1 Phase 1.7 — rein_health migrated to #[op] and
// now uses ops::handlers::diagnostics::HealthParams (derives clap::Args too).

// ConsolidateParams removed — rein_consolidate migrated to #[op] inventory.
// See ops/handlers/maintenance.rs for the new ConsolidateParams + ConsolidateOutput structs.

// DedupParams removed — rein_dedup migrated to #[op] inventory.
// See ops/handlers/maintenance.rs for the new DedupParams + DedupOutput structs.

// CleanupParams removed — rein_cleanup migrated to #[op] inventory.
// See ops/handlers/maintenance.rs for the new CleanupParams + CleanupOutput structs.

// OrganizeParams removed — rein_organize migrated to #[op] inventory (see ops/handlers/maintenance.rs).

// Phase 2.6 complete: all memoir Params structs live alongside their #[op]
// handlers in ops/handlers/knowledge.rs. The legacy MemoirCreateParams /
// MemoirShowParams / ConceptAddParams / ConceptRefineParams /
// ConceptSearchParams / ConceptSearchAllParams / LinkParams / InspectParams /
// ExportParams types used by the pre-A1 #[tool] handlers were removed when
// those handlers migrated to #[op] inventory.

// TimelineParams + ConceptHistoryParams removed — migrated to #[op] inventory.
// See ops/handlers/memory.rs for the new definitions.

#[cfg(test)]
mod tests {
    use serde_json::json;

    #[test]
    fn consolidate_params_accept_comma_separated_topics() {
        use crate::ops::handlers::maintenance::ConsolidateParams;
        let params: ConsolidateParams = serde_json::from_value(json!({
            "topics": "rein-release, rein-architecture , rein-devlog"
        }))
        .unwrap();

        assert_eq!(
            params.topics,
            Some(vec![
                "rein-release".to_string(),
                "rein-architecture".to_string(),
                "rein-devlog".to_string()
            ])
        );
    }

    #[test]
    fn cleanup_params_accept_topic_array_with_commas() {
        // CleanupParams migrated to ops/handlers/maintenance.rs — test re-targeted.
        let params: rein::ops::handlers::maintenance::CleanupParams =
            serde_json::from_value(json!({
                "topics": ["alpha, beta", "gamma", " "]
            }))
            .unwrap();

        assert_eq!(
            params.topics,
            Some(vec![
                "alpha".to_string(),
                "beta".to_string(),
                "gamma".to_string()
            ])
        );
    }
}
