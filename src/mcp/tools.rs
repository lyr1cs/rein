use rmcp::schemars;
use serde::Deserialize;

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

/// Parameters for rein_feedback tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct FeedbackParams {
    /// Memory IDs that were actually used by the agent.
    pub memory_ids: Vec<String>,
    /// The request_id from the recall result (for attribution).
    pub request_id: Option<String>,
    /// Optional: the query that produced these results.
    pub query: Option<String>,
    /// Optional: whether the recall was helpful overall (true/false).
    pub helpful: Option<bool>,
}

/// Parameters for rein_forget tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ForgetParams {
    /// The memory ID to delete.
    pub id: String,
}

/// Parameters for rein_health tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct HealthParams {
    /// Optional topic to check health for.
    pub topic: Option<String>,
}

/// Parameters for rein_consolidate tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ConsolidateParams {
    /// The topic to consolidate.
    pub topic: String,
    /// Summary for the consolidated memory.
    pub summary: String,
}

/// Parameters for rein_dedup tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct DedupParams {
    /// If true, only report duplicates without removing them.
    pub dry_run: Option<bool>,
}

/// Parameters for rein_recent tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct RecentParams {
    /// Maximum number of recent memories to return (default 10).
    pub limit: Option<usize>,
}

/// Parameters for rein_gc tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GcParams {
    /// If true, only report how many would be pruned without removing them.
    pub dry_run: Option<bool>,
}

/// Parameters for rein_organize tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct OrganizeParams {
    /// Maximum links per memory (default 5).
    pub max_links: Option<usize>,
}

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
    pub limit: Option<usize>,
}

/// Parameters for rein_memoir_search_all tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ConceptSearchAllParams {
    /// Search query string.
    pub query: String,
    /// Maximum number of results (default 10).
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
    pub limit: Option<usize>,
}
