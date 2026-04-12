use serde::Serialize;

use crate::config::{Provider, ReinConfig};
use crate::embed;
use crate::extract::hooks::queue::{collect_queue_diagnostics, QueueGroupDiagnostics};
use crate::search::warmup;
use crate::store::hnsw::HnswIndex;
use crate::store::sqlite::SqliteStore;
use crate::store::tantivy_fts::TantivyFts;
use crate::types::traits::{Embedder as _, MemoryStore};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum CheckStatus {
    Ok,
    Warn,
    Fail,
}

impl CheckStatus {
    fn label(self) -> &'static str {
        match self {
            Self::Ok => "OK",
            Self::Warn => "WARN",
            Self::Fail => "FAIL",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ReportStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

#[derive(Debug, Clone, Serialize)]
pub struct DoctorCheck {
    pub name: &'static str,
    pub status: CheckStatus,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct DoctorReport {
    pub status: ReportStatus,
    pub checks: Vec<DoctorCheck>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct DoctorOptions {
    pub network: bool,
}

#[derive(Debug)]
struct StoreSnapshot {
    total_memories: usize,
    active_memories: usize,
    embed_cache_rows: usize,
    artifact_rows: usize,
}

pub async fn run(config: &ReinConfig, options: DoctorOptions) -> DoctorReport {
    let mut checks = Vec::new();

    checks.push(check_embedding_provider(config));
    checks.push(check_extract_provider(config));
    checks.push(check_query_expansion_provider(config));
    checks.push(check_reranker_provider(config));
    checks.push(check_supermemory(config));
    checks.push(check_http_auth(config));
    checks.push(check_proxy_auth(config));

    match config.open_store() {
        Ok(store) => {
            let stats = store.stats();
            match stats {
                Ok(stats) => {
                    checks.push(ok(
                        "database",
                        format!(
                            "{} memories, {} topics, {} memoirs at {}",
                            stats.total_memories,
                            stats.topic_count,
                            stats.memoir_count,
                            store.db_path().display()
                        ),
                    ));

                    match collect_store_snapshot(&store) {
                        Ok(snapshot) => {
                            let (hnsw_check, indexed_vectors) =
                                inspect_hnsw(&store, snapshot.total_memories);
                            checks.push(check_vector_coverage(config, &snapshot, indexed_vectors));
                            checks.push(check_tantivy(&store, snapshot.active_memories));
                            checks.push(hnsw_check);
                        }
                        Err(e) => checks.push(fail("database_snapshot", e.to_string())),
                    }
                }
                Err(e) => checks.push(fail("database", e.to_string())),
            }
        }
        Err(e) => checks.push(fail("database", e.to_string())),
    }

    let queue_diag = collect_queue_diagnostics(config);
    checks.push(check_queues(&queue_diag));

    if options.network {
        checks.push(check_embedding_network(config).await);
    }

    DoctorReport {
        status: overall_status(&checks),
        checks,
    }
}

pub fn format_human(report: &DoctorReport) -> String {
    let mut lines = vec!["rein doctor".to_string(), "===========".to_string()];
    for check in &report.checks {
        lines.push(format!(
            "[{}] {}: {}",
            check.status.label(),
            check.name,
            check.message
        ));
    }
    lines.push(String::new());
    lines.push(format!("Overall: {}", overall_label(report.status)));
    lines.join("\n")
}

impl DoctorReport {
    pub fn has_failures(&self) -> bool {
        self.checks
            .iter()
            .any(|check| matches!(check.status, CheckStatus::Fail))
    }

    pub fn exit_code(&self) -> i32 {
        if self.has_failures() {
            1
        } else {
            0
        }
    }
}

fn overall_status(checks: &[DoctorCheck]) -> ReportStatus {
    if checks.iter().any(|c| matches!(c.status, CheckStatus::Fail)) {
        ReportStatus::Unhealthy
    } else if checks.iter().any(|c| matches!(c.status, CheckStatus::Warn)) {
        ReportStatus::Degraded
    } else {
        ReportStatus::Healthy
    }
}

fn overall_label(status: ReportStatus) -> &'static str {
    match status {
        ReportStatus::Healthy => "healthy",
        ReportStatus::Degraded => "degraded",
        ReportStatus::Unhealthy => "unhealthy",
    }
}

fn ok(name: &'static str, message: impl Into<String>) -> DoctorCheck {
    DoctorCheck {
        name,
        status: CheckStatus::Ok,
        message: message.into(),
    }
}

fn warn(name: &'static str, message: impl Into<String>) -> DoctorCheck {
    DoctorCheck {
        name,
        status: CheckStatus::Warn,
        message: message.into(),
    }
}

fn fail(name: &'static str, message: impl Into<String>) -> DoctorCheck {
    DoctorCheck {
        name,
        status: CheckStatus::Fail,
        message: message.into(),
    }
}

fn check_embedding_provider(config: &ReinConfig) -> DoctorCheck {
    match config.embedding_provider() {
        Provider::Google => match config.embedding.google.api_key.as_ref() {
            Some(_) => ok(
                "embedding_provider",
                format!(
                    "google:{} configured ({}d)",
                    config.embedding.google.model, config.embedding.dimensions
                ),
            ),
            None => warn(
                "embedding_provider",
                "google configured but GEMINI_API_KEY is missing; vector embedding is disabled",
            ),
        },
        Provider::Omlx => ok(
            "embedding_provider",
            format!(
                "omlx:{} at {} ({}d)",
                config.embedding.omlx.model,
                config.embedding.omlx.endpoint,
                config.embedding.dimensions
            ),
        ),
        Provider::None => ok("embedding_provider", "disabled"),
    }
}

fn check_extract_provider(config: &ReinConfig) -> DoctorCheck {
    match config.extract_provider() {
        Provider::Google => match config.extract.google.api_key.as_ref() {
            Some(_) => ok(
                "extract_provider",
                format!("google:{} configured", config.extract.google.model),
            ),
            None => warn(
                "extract_provider",
                "google configured but GEMINI_API_KEY is missing; LLM extraction is disabled",
            ),
        },
        Provider::Omlx => ok(
            "extract_provider",
            format!(
                "omlx:{} at {}",
                config.extract.omlx.model, config.extract.omlx.endpoint
            ),
        ),
        Provider::None => ok("extract_provider", "disabled"),
    }
}

fn check_query_expansion_provider(config: &ReinConfig) -> DoctorCheck {
    match config.expand_provider() {
        Provider::Google => match config.query_expansion.google.api_key.as_ref() {
            Some(_) => ok(
                "query_expansion",
                format!("google:{} configured", config.query_expansion.google.model),
            ),
            None => warn(
                "query_expansion",
                "google configured but GEMINI_API_KEY is missing; expansion is disabled",
            ),
        },
        Provider::Omlx => ok(
            "query_expansion",
            format!(
                "omlx:{} at {}",
                config.query_expansion.omlx.model, config.query_expansion.omlx.endpoint
            ),
        ),
        Provider::None => ok("query_expansion", "disabled"),
    }
}

fn check_reranker_provider(config: &ReinConfig) -> DoctorCheck {
    match config.reranker_provider() {
        Provider::Google => match config.query_expansion.google.api_key.as_ref() {
            Some(_) => ok(
                "llm_reranker",
                format!(
                    "google:{} configured (top_n={})",
                    config.query_expansion.google.model, config.search.llm_reranker_top_n
                ),
            ),
            None => warn(
                "llm_reranker",
                "google reranker configured but GEMINI_API_KEY is missing; reranker will be skipped",
            ),
        },
        Provider::Omlx => ok(
            "llm_reranker",
            format!(
                "omlx:{} at {} (top_n={})",
                config.query_expansion.omlx.model,
                config.query_expansion.omlx.endpoint,
                config.search.llm_reranker_top_n
            ),
        ),
        Provider::None => ok("llm_reranker", "disabled"),
    }
}

fn check_supermemory(config: &ReinConfig) -> DoctorCheck {
    if !config.sync.supermemory_enabled {
        return ok("supermemory", "disabled");
    }
    match config.sync.api_key.as_ref() {
        Some(_) => ok(
            "supermemory",
            format!("enabled via {}", config.sync.endpoint),
        ),
        None => warn(
            "supermemory",
            "enabled but SUPERMEMORY_CC_API_KEY is missing; cross-validation will be partial",
        ),
    }
}

fn check_http_auth(config: &ReinConfig) -> DoctorCheck {
    if !config.server.sse_enabled && !config.server.gui_enabled {
        return ok("http_auth", "HTTP/SSE disabled");
    }

    let token_present = std::env::var("REIN_HTTP_TOKEN").ok().is_some();
    let is_loopback = is_loopback_bind(&config.server.sse_bind);
    let allow_unauth = config.server.allow_unauthenticated_loopback && is_loopback;

    if token_present {
        ok(
            "http_auth",
            format!(
                "token configured for {}:{}",
                config.server.sse_bind, config.server.sse_port
            ),
        )
    } else if allow_unauth {
        ok(
            "http_auth",
            format!(
                "loopback-only unauthenticated access allowed for {}:{}",
                config.server.sse_bind, config.server.sse_port
            ),
        )
    } else {
        fail(
            "http_auth",
            format!(
                "HTTP/SSE is enabled on {}:{} without REIN_HTTP_TOKEN",
                config.server.sse_bind, config.server.sse_port
            ),
        )
    }
}

fn check_proxy_auth(config: &ReinConfig) -> DoctorCheck {
    let token_present = std::env::var("REIN_PROXY_TOKEN")
        .ok()
        .or_else(|| std::env::var("REIN_HTTP_TOKEN").ok())
        .is_some();
    let is_loopback = is_loopback_bind(&config.proxy.bind);
    let allow_unauth = config.proxy.allow_unauthenticated_loopback && is_loopback;

    if token_present {
        ok(
            "proxy_auth",
            format!(
                "token configured for {}:{}",
                config.proxy.bind, config.proxy.port
            ),
        )
    } else if allow_unauth {
        ok(
            "proxy_auth",
            format!(
                "loopback-only unauthenticated access allowed for {}:{}",
                config.proxy.bind, config.proxy.port
            ),
        )
    } else {
        fail(
            "proxy_auth",
            format!(
                "proxy cannot start on {}:{} without REIN_PROXY_TOKEN or REIN_HTTP_TOKEN",
                config.proxy.bind, config.proxy.port
            ),
        )
    }
}

fn collect_store_snapshot(store: &SqliteStore) -> anyhow::Result<StoreSnapshot> {
    let total_memories = count_sql(store, "SELECT COUNT(*) FROM memories")?;
    let active_memories = count_sql(
        store,
        "SELECT COUNT(*) FROM memories WHERE superseded_by IS NULL AND status = 'active'",
    )?;
    let embed_cache_rows = count_sql(store, "SELECT COUNT(*) FROM embed_cache")?;
    let artifact_rows = count_sql(store, "SELECT COUNT(*) FROM session_artifacts")?;
    Ok(StoreSnapshot {
        total_memories,
        active_memories,
        embed_cache_rows,
        artifact_rows,
    })
}

fn count_sql(store: &SqliteStore, sql: &str) -> anyhow::Result<usize> {
    Ok(store.conn().query_row(sql, [], |row| row.get(0))?)
}

fn check_vector_coverage(
    config: &ReinConfig,
    snapshot: &StoreSnapshot,
    indexed_vectors: Option<usize>,
) -> DoctorCheck {
    if snapshot.total_memories == 0 {
        return ok("vector_store", "0 memories");
    }

    let Some(indexed_vectors) = indexed_vectors else {
        let hint = match config.embedding_provider() {
            Provider::None => "embedding provider is disabled",
            Provider::Google if config.embedding.google.api_key.is_none() => {
                "embedding API key is missing"
            }
            _ => "run `rein warmup` or `rein migrate --reindex`",
        };
        return warn(
            "vector_store",
            format!(
                "vector index unavailable for {} memories ({} active, cache={}, artifacts={}); {}",
                snapshot.total_memories,
                snapshot.active_memories,
                snapshot.embed_cache_rows,
                snapshot.artifact_rows,
                hint
            ),
        );
    };

    let coverage = indexed_vectors as f64 / snapshot.total_memories as f64;
    let message = format!(
        "{} indexed vectors for {} memories ({} active, {:.0}% coverage, cache={}, artifacts={})",
        indexed_vectors,
        snapshot.total_memories,
        snapshot.active_memories,
        coverage * 100.0,
        snapshot.embed_cache_rows,
        snapshot.artifact_rows
    );
    if coverage >= 0.9 {
        ok("vector_store", message)
    } else {
        warn("vector_store", format!("{message}; consider `rein warmup`"))
    }
}

fn check_tantivy(store: &SqliteStore, active_memories: usize) -> DoctorCheck {
    let db_path = store.db_path();
    let index_path = db_path.with_extension("tantivy");
    let dirty = warmup::tantivy_dirty_path(db_path).exists();

    if active_memories == 0 && !index_path.exists() {
        return ok("tantivy", "not built yet (0 active memories)");
    }
    if !index_path.exists() {
        return warn(
            "tantivy",
            format!(
                "index directory missing at {}; next recall/warmup will rebuild it",
                index_path.display()
            ),
        );
    }
    match TantivyFts::open(&index_path) {
        Ok(_) if dirty => warn(
            "tantivy",
            format!(
                "index opened at {} but dirty marker is present; next recall/warmup should rebuild",
                index_path.display()
            ),
        ),
        Ok(_) => ok(
            "tantivy",
            format!("index opened at {}", index_path.display()),
        ),
        Err(e) => fail(
            "tantivy",
            format!("failed to open {}: {e}", index_path.display()),
        ),
    }
}

fn inspect_hnsw(store: &SqliteStore, total_memories: usize) -> (DoctorCheck, Option<usize>) {
    let base_path = store.db_path().with_extension("");
    let index_path = base_path.with_extension("usearch");
    let meta_path = base_path.with_extension("usearch.meta");
    let dirty = HnswIndex::is_dirty(&base_path);

    if total_memories == 0 && !index_path.exists() {
        return (ok("hnsw", "not built yet (0 memories)"), Some(0));
    }
    if !index_path.exists() {
        return (
            warn(
                "hnsw",
                format!(
                    "index file missing at {}; next vector recall/warmup will rebuild it",
                    index_path.display()
                ),
            ),
            None,
        );
    }
    if !meta_path.exists() {
        return (
            warn(
                "hnsw",
                format!(
                    "index file exists at {} but metadata is missing at {}",
                    index_path.display(),
                    meta_path.display()
                ),
            ),
            None,
        );
    }

    match HnswIndex::open(&base_path, store.dims) {
        Ok(index) => {
            let message = format!(
                "{} vectors indexed at {}",
                index.len(),
                index_path.display()
            );
            if dirty {
                (
                    warn(
                        "hnsw",
                        format!(
                            "{message}; dirty marker is present and next vector recall/warmup should rebuild"
                        ),
                    ),
                    Some(index.len()),
                )
            } else {
                (ok("hnsw", message), Some(index.len()))
            }
        }
        Err(e) => (
            fail(
                "hnsw",
                format!("failed to open {}: {e}", index_path.display()),
            ),
            None,
        ),
    }
}

fn check_queues(diag: &QueueGroupDiagnostics) -> DoctorCheck {
    let pending = diag.memory.pending + diag.cleanup.pending + diag.dedup.pending;
    let inflight = diag.memory.inflight + diag.cleanup.inflight + diag.dedup.inflight;
    let dead = diag.memory.dead_letters + diag.cleanup.dead_letters + diag.dedup.dead_letters;
    let issues = diag.memory.issues.len() + diag.cleanup.issues.len() + diag.dedup.issues.len();
    let message = format!(
        "memory p{} i{} d{} | cleanup p{} i{} d{} | dedup p{} i{} d{}",
        diag.memory.pending,
        diag.memory.inflight,
        diag.memory.dead_letters,
        diag.cleanup.pending,
        diag.cleanup.inflight,
        diag.cleanup.dead_letters,
        diag.dedup.pending,
        diag.dedup.inflight,
        diag.dedup.dead_letters
    );

    if issues > 0 {
        let first_issue = diag
            .memory
            .issues
            .iter()
            .chain(diag.cleanup.issues.iter())
            .chain(diag.dedup.issues.iter())
            .next()
            .cloned()
            .unwrap_or_else(|| "queue diagnostics failed".to_string());
        warn("queues", format!("{message}; {first_issue}"))
    } else if dead > 0 {
        warn("queues", format!("{message}; dead letters present"))
    } else if inflight > 0 {
        warn(
            "queues",
            format!("{message}; inflight jobs need a worker to finish"),
        )
    } else if pending > 0 {
        warn(
            "queues",
            format!("{message}; pending jobs are waiting to be drained"),
        )
    } else {
        ok("queues", message)
    }
}

async fn check_embedding_network(config: &ReinConfig) -> DoctorCheck {
    let Some(embedder) = embed::create_embedder(config) else {
        return ok(
            "embedding_network",
            "skipped (embedding provider unavailable)",
        );
    };

    match embedder.embed("rein doctor ping").await {
        Ok(vector) => ok(
            "embedding_network",
            format!(
                "{} responded with {} dimensions",
                embedder.model_name(),
                vector.len()
            ),
        ),
        Err(e) => fail("embedding_network", e.to_string()),
    }
}

fn is_loopback_bind(bind: &str) -> bool {
    matches!(bind, "127.0.0.1" | "::1" | "localhost")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};
    use std::sync::{Mutex, OnceLock};

    use crate::extract::hooks::buffer;
    use crate::types::{Importance, Memory, MemoryLayer, MemoryStatus, Source};

    fn test_memory(topic: &str, summary: &str, content: &str) -> Memory {
        Memory {
            id: ulid::Ulid::new().to_string(),
            layer: MemoryLayer::STM,
            topic: topic.to_string(),
            summary: summary.to_string(),
            content: content.to_string(),
            keywords: vec![],
            importance: Importance::Medium,
            source: Source::Manual,
            strength: 1.0,
            decay_lambda: 0.06,
            access_count: 0,
            superseded_by: None,
            canonical_id: None,
            support_count: 1,
            merge_count: 0,
            dedup_confidence: 1.0,
            source_diversity: 1.0,
            contradiction_score: 0.0,
            related_ids: vec![],
            concept_ids: vec![],
            status: MemoryStatus::Active,
            embedding: None,
            tier: crate::types::MemoryTier::Warm,
            cluster_id: None,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            last_accessed: chrono::Utc::now(),
        }
    }

    fn temp_config(tempdir: &tempfile::TempDir) -> ReinConfig {
        let toml = format!(
            r#"
[database]
path = "{}"

[embedding]
provider = "none"
dimensions = 3072
[embedding.google]
model = "text-embedding-004"
[embedding.omlx]
endpoint = "http://127.0.0.1:11434/v1/embeddings"

[search]
rrf_k = 60.0
rrf_fts_weight = 1.0
rrf_vec_weight = 1.0
dedup_similarity = 0.7
dedup_time_window_days = 7

[chunking]
max_tokens = 400
overlap_percent = 15
metadata_prefix = true

[sync]
supermemory_enabled = false
auto_memory_enabled = false
auto_memory_glob = "~/.claude/projects/**/*.md"

[decay]
base_lambda = 0.06
prune_threshold = 0.3

[server]
sse_enabled = true
sse_bind = "0.0.0.0"
sse_port = 8765
compact = false
gui_enabled = false
allow_unauthenticated_loopback = false

[hooks]
buffer_dir = "{}"

[extract]
provider = "none"
[extract.google]
model = "gemini-2.5-flash-lite"
[extract.omlx]
endpoint = "http://127.0.0.1:11434/v1/chat/completions"

[adaptive]
enabled = true

[query_expansion]
provider = "none"
[query_expansion.google]
model = "gemini-2.5-flash-lite"
[query_expansion.omlx]
endpoint = "http://127.0.0.1:11434/v1/chat/completions"

[proxy]
bind = "0.0.0.0"
port = 8777
allow_unauthenticated_loopback = false

[async_memory]
provider = "inherit"

[cleanup]
"#,
            tempdir.path().join("doctor.db").display(),
            tempdir.path().display()
        );
        ReinConfig::load_from_str(&toml).unwrap()
    }

    fn queue_file(config: &ReinConfig, prefix: &str) -> PathBuf {
        let base = buffer::resolve_buffer_dir(config);
        let db_tag = {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            config.resolve_db_path().hash(&mut h);
            format!("{:016x}", h.finish())
        };
        base.join("queue")
            .join(db_tag)
            .join(format!("{prefix}.jsonl"))
    }

    fn write_lines(path: &Path, count: usize) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let body = (0..count).map(|_| "{}").collect::<Vec<_>>().join("\n");
        std::fs::write(path, format!("{body}\n")).unwrap();
    }

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    struct EnvRestore {
        key: &'static str,
        value: Option<String>,
    }

    impl Drop for EnvRestore {
        fn drop(&mut self) {
            match &self.value {
                Some(value) => std::env::set_var(self.key, value),
                None => std::env::remove_var(self.key),
            }
        }
    }

    #[test]
    fn test_format_human_reports_overall_status() {
        let report = DoctorReport {
            status: ReportStatus::Degraded,
            checks: vec![ok("database", "connected"), warn("queues", "pending jobs")],
        };
        let text = format_human(&report);
        assert!(text.contains("[OK] database: connected"));
        assert!(text.contains("[WARN] queues: pending jobs"));
        assert!(text.contains("Overall: degraded"));
    }

    #[tokio::test]
    async fn test_doctor_flags_auth_and_queue_warnings() {
        let _guard = env_lock().lock().unwrap();
        let _restore_http = EnvRestore {
            key: "REIN_HTTP_TOKEN",
            value: std::env::var("REIN_HTTP_TOKEN").ok(),
        };
        let _restore_proxy = EnvRestore {
            key: "REIN_PROXY_TOKEN",
            value: std::env::var("REIN_PROXY_TOKEN").ok(),
        };
        std::env::remove_var("REIN_HTTP_TOKEN");
        std::env::remove_var("REIN_PROXY_TOKEN");

        let tempdir = tempfile::tempdir().unwrap();
        let config = temp_config(&tempdir);
        let store = config.open_store().unwrap();
        store
            .store(test_memory("doctor", "doctor memory", "doctor content"))
            .unwrap();

        let tantivy_dirty = warmup::tantivy_dirty_path(store.db_path());
        std::fs::create_dir_all(tantivy_dirty.parent().unwrap()).unwrap();
        std::fs::write(&tantivy_dirty, b"dirty").unwrap();
        let hnsw_dirty = store
            .db_path()
            .with_extension("")
            .with_extension("usearch.dirty");
        std::fs::write(&hnsw_dirty, b"dirty").unwrap();

        write_lines(&queue_file(&config, "memory_queue"), 2);
        write_lines(&queue_file(&config, "memory_queue_dead"), 1);

        let report = run(&config, DoctorOptions::default()).await;
        assert!(report.has_failures());

        let http = report
            .checks
            .iter()
            .find(|c| c.name == "http_auth")
            .unwrap();
        assert_eq!(http.status, CheckStatus::Fail);

        let proxy = report
            .checks
            .iter()
            .find(|c| c.name == "proxy_auth")
            .unwrap();
        assert_eq!(proxy.status, CheckStatus::Fail);

        let queues = report.checks.iter().find(|c| c.name == "queues").unwrap();
        assert_eq!(queues.status, CheckStatus::Warn);
        assert!(queues.message.contains("memory p2"));

        let tantivy = report.checks.iter().find(|c| c.name == "tantivy").unwrap();
        assert_eq!(tantivy.status, CheckStatus::Warn);
    }
}
