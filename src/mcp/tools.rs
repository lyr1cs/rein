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
