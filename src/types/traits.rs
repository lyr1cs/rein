use super::error::ReinResult;
use super::memory::Memory;

/// Aggregate statistics for the memory store.
#[derive(Debug, Clone)]
pub struct StoreStats {
    pub total_memories: usize,
    pub ltm_count: usize,
    pub stm_count: usize,
    pub topic_count: usize,
    pub avg_strength: f64,
    pub memoir_count: usize,
    pub concept_count: usize,
    pub link_count: usize,
}

/// Health report for a single topic.
#[derive(Debug, Clone)]
pub struct HealthReport {
    pub topic: String,
    pub count: usize,
    pub avg_strength: f64,
    pub stale_count: usize,
    pub needs_consolidation: bool,
}

/// Core storage backend for memories.
pub trait MemoryStore {
    fn store(&self, memory: Memory) -> ReinResult<String>;
    fn get(&self, id: &str) -> ReinResult<Memory>;
    fn update(&self, memory: &Memory) -> ReinResult<()>;
    fn delete(&self, id: &str) -> ReinResult<()>;
    fn search_fts(&self, query: &str, topic: Option<&str>, limit: usize)
        -> ReinResult<Vec<Memory>>;
    fn search_vec(
        &self,
        embedding: &[f32],
        topic: Option<&str>,
        limit: usize,
    ) -> ReinResult<Vec<Memory>>;
    fn apply_decay(&self) -> ReinResult<u64>;
    fn prune(&self, threshold: f64) -> ReinResult<u64>;
    fn list_topics(&self) -> ReinResult<Vec<String>>;
    fn consolidate(&self, topic: &str) -> ReinResult<Vec<Memory>>;
    fn stats(&self) -> ReinResult<StoreStats>;
    fn health(&self, topic: Option<&str>) -> ReinResult<Vec<HealthReport>>;
}

/// Embedding provider for vector search.
/// Implementors must provide a stable model_name() for cache keying and model-change detection.
#[allow(async_fn_in_trait)]
pub trait Embedder {
    /// Unique identifier for this model (e.g., "gemini-embedding-001", "text-embedding-3-large").
    /// Used for cache keying and model-change detection.
    fn model_name(&self) -> &str;
    fn dimensions(&self) -> usize;
    async fn embed(&self, text: &str) -> ReinResult<Vec<f32>>;
    async fn embed_batch(&self, texts: &[&str]) -> ReinResult<Vec<Vec<f32>>>;
}

/// Cloud synchronization for cross-validation.
#[allow(async_fn_in_trait)]
pub trait CloudSync {
    async fn search(&self, query: &str) -> ReinResult<Vec<Memory>>;
    async fn store(&self, memory: &Memory) -> ReinResult<()>;
}
