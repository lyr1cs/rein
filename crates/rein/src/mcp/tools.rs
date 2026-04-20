use rmcp::schemars;
use serde::de::{self, Deserializer};
use serde::Deserialize;

fn deserialize_option_usize_from_string<'de, D>(deserializer: D) -> Result<Option<usize>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum UsizeValue {
        Number(u64),
        String(String),
    }

    match Option::<UsizeValue>::deserialize(deserializer)? {
        None => Ok(None),
        Some(UsizeValue::Number(value)) => usize::try_from(value)
            .map(Some)
            .map_err(|_| de::Error::custom(format!("value {value} exceeds usize range"))),
        Some(UsizeValue::String(value)) => {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                return Ok(None);
            }
            trimmed
                .parse::<usize>()
                .map(Some)
                .map_err(de::Error::custom)
        }
    }
}

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

/// Parameters for rein_recall tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct RecallParams {
    /// The search query string.
    pub query: String,
    /// Optional topic filter.
    pub topic: Option<String>,
    /// Optional keyword filter.
    pub keyword: Option<String>,
    /// Maximum number of results to return (default 10).
    #[serde(default, deserialize_with = "deserialize_option_usize_from_string")]
    pub limit: Option<usize>,
    /// Filter memories created after this date (YYYY-MM-DD or RFC3339).
    pub from: Option<String>,
    /// Filter memories created before this date (YYYY-MM-DD or RFC3339).
    pub to: Option<String>,
    /// Override query expansion: true=force, false=disable, null=use config default.
    pub expand: Option<bool>,
}

/// Parameters for rein_store tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct StoreParams {
    /// Topic/category for the memory.
    pub topic: String,
    /// The content to store.
    pub content: String,
    /// Importance level: low, medium, high, or critical.
    pub importance: Option<String>,
    /// Comma-separated keywords for the memory.
    pub keywords: Option<String>,
}

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

/// Parameters for rein_update tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct UpdateParams {
    /// The memory ID to update.
    pub id: String,
    /// New content for the memory.
    pub content: String,
    /// New importance level (optional).
    pub importance: Option<String>,
}

// FeedbackParams removed — rein_feedback migrated to #[op] inventory (Phase 2.4).
// See ops/handlers/adaptive.rs for the new FeedbackParams + FeedbackOutput structs.

/// Parameters for rein_forget tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ForgetParams {
    /// The memory ID to delete.
    pub id: String,
}

// HealthParams removed in A1 Phase 1.7 — rein_health migrated to #[op] and
// now uses ops::handlers::diagnostics::HealthParams (derives clap::Args too).

// ConsolidateParams removed — rein_consolidate migrated to #[op] inventory.
// See ops/handlers/maintenance.rs for the new ConsolidateParams + ConsolidateOutput structs.

// DedupParams removed — rein_dedup migrated to #[op] inventory.
// See ops/handlers/maintenance.rs for the new DedupParams + DedupOutput structs.

// CleanupParams removed — rein_cleanup migrated to #[op] inventory.
// See ops/handlers/maintenance.rs for the new CleanupParams + CleanupOutput structs.

/// Parameters for rein_recent tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct RecentParams {
    /// Maximum number of recent memories to return (default 10).
    #[serde(default, deserialize_with = "deserialize_option_usize_from_string")]
    pub limit: Option<usize>,
}

// OrganizeParams removed — rein_organize migrated to #[op] inventory (see ops/handlers/maintenance.rs).

/// Parameters for rein_memoir_create tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct MemoirCreateParams {
    /// Name for the memoir (must be unique).
    pub name: String,
    /// Optional description of the memoir.
    pub description: Option<String>,
}

/// Parameters for rein_memoir_show tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct MemoirShowParams {
    /// Name of the memoir to show.
    pub name: String,
}

/// Parameters for rein_memoir_add_concept tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ConceptAddParams {
    /// Name of the memoir to add the concept to.
    pub memoir: String,
    /// Name of the concept.
    pub name: String,
    /// Definition of the concept.
    pub definition: String,
    /// Optional comma-separated labels for the concept.
    pub labels: Option<String>,
}

/// Parameters for rein_memoir_refine tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ConceptRefineParams {
    /// Name of the memoir containing the concept.
    pub memoir: String,
    /// Name of the concept to refine.
    pub name: String,
    /// New definition for the concept.
    pub definition: String,
}

/// Parameters for rein_memoir_search tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ConceptSearchParams {
    /// Name of the memoir to search in.
    pub memoir: String,
    /// Search query string.
    pub query: String,
    /// Maximum number of results (default 10).
    #[serde(default, deserialize_with = "deserialize_option_usize_from_string")]
    pub limit: Option<usize>,
}

/// Parameters for rein_memoir_search_all tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ConceptSearchAllParams {
    /// Search query string.
    pub query: String,
    /// Maximum number of results (default 10).
    #[serde(default, deserialize_with = "deserialize_option_usize_from_string")]
    pub limit: Option<usize>,
}

/// Parameters for rein_memoir_link tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct LinkParams {
    /// Name of the memoir containing both concepts.
    pub memoir: String,
    /// Name of the source concept.
    pub from: String,
    /// Name of the target concept.
    pub to: String,
    /// Relation type: part_of, depends_on, related_to, contradicts, refines, alternative_to, caused_by, instance_of, superseded_by.
    pub relation: String,
}

/// Parameters for rein_memoir_inspect tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct InspectParams {
    /// Name of the memoir.
    pub memoir: String,
    /// Name of the concept to inspect.
    pub name: String,
    /// BFS depth (default 1).
    #[serde(default, deserialize_with = "deserialize_option_usize_from_string")]
    pub depth: Option<usize>,
}

/// Parameters for rein_memoir_export tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ExportParams {
    /// Name of the memoir to export.
    pub memoir: String,
    /// Export format: json, ascii, or dot (default json).
    pub format: Option<String>,
}

/// Parameters for rein_timeline tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct TimelineParams {
    /// Start date (YYYY-MM-DD or RFC3339).
    pub from: Option<String>,
    /// End date (YYYY-MM-DD or RFC3339).
    pub to: Option<String>,
    /// Maximum entries (default 20).
    #[serde(default, deserialize_with = "deserialize_option_usize_from_string")]
    pub limit: Option<usize>,
}

/// Parameters for rein_concept_history tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ConceptHistoryParams {
    /// Memoir containing the concept.
    pub memoir: String,
    /// Name of the concept.
    pub name: String,
    /// Maximum revisions to return (default 10).
    #[serde(default, deserialize_with = "deserialize_option_usize_from_string")]
    pub limit: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn recall_params_accept_numeric_string_limit() {
        let params: RecallParams = serde_json::from_value(json!({
            "query": "release history",
            "limit": "100"
        }))
        .unwrap();

        assert_eq!(params.limit, Some(100));
    }

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
