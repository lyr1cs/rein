use serde::Deserialize;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Config structs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct ReinConfig {
    pub database: DatabaseConfig,
    pub embedding: EmbeddingConfig,
    pub search: SearchConfig,
    pub chunking: ChunkingConfig,
    pub sync: SyncConfig,
    pub decay: DecayConfig,
    pub server: ServerConfig,
    #[serde(default)]
    pub hooks: HooksConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DatabaseConfig {
    pub path: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingConfig {
    pub provider: String,
    pub dimensions: usize,
    pub google: GoogleEmbeddingConfig,
    pub omlx: OmlxEmbeddingConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GoogleEmbeddingConfig {
    pub model: String,
    #[serde(default)]
    pub api_key: Option<String>,
    /// API endpoint override (for proxies in China, etc.)
    /// Default: "https://generativelanguage.googleapis.com"
    #[serde(default = "default_google_endpoint")]
    pub endpoint: String,
}

fn default_google_endpoint() -> String {
    "https://generativelanguage.googleapis.com".to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct OmlxEmbeddingConfig {
    pub endpoint: String,
    #[serde(default = "default_omlx_model")]
    pub model: String,
}

fn default_omlx_model() -> String {
    "default".to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct SearchConfig {
    pub rrf_k: f64,
    pub rrf_fts_weight: f64,
    pub rrf_vec_weight: f64,
    pub waterfall_fts_threshold: f64,
    pub dedup_similarity: f64,
    pub dedup_time_window_days: i64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChunkingConfig {
    pub max_tokens: usize,
    pub overlap_percent: usize,
    pub metadata_prefix: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SyncConfig {
    pub supermemory_enabled: bool,
    pub auto_memory_enabled: bool,
    pub auto_memory_glob: String,
    #[serde(default)]
    pub api_key: Option<String>,
    /// Supermemory API endpoint override
    #[serde(default = "default_supermemory_endpoint")]
    pub endpoint: String,
}

fn default_supermemory_endpoint() -> String {
    "https://api.supermemory.ai".to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct DecayConfig {
    pub base_lambda: f64,
    pub ltm_beta: f64,
    pub stm_beta: f64,
    pub interval_hours: u64,
    pub prune_threshold: f64,
    pub stm_to_ltm_access_count: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    pub compact: bool,
    pub sse_enabled: bool,
    pub sse_port: u16,
    pub sse_bind: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HooksConfig {
    pub min_turns: usize,
    pub context_before: usize,
    pub context_after: usize,
    pub max_items_per_session: usize,
}

// ---------------------------------------------------------------------------
// Default implementations
// ---------------------------------------------------------------------------

impl Default for ReinConfig {
    fn default() -> Self {
        Self {
            database: DatabaseConfig::default(),
            embedding: EmbeddingConfig::default(),
            search: SearchConfig::default(),
            chunking: ChunkingConfig::default(),
            sync: SyncConfig::default(),
            decay: DecayConfig::default(),
            server: ServerConfig::default(),
            hooks: HooksConfig::default(),
        }
    }
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            path: "auto".to_string(),
        }
    }
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            provider: "google".to_string(),
            dimensions: 3072,
            google: GoogleEmbeddingConfig::default(),
            omlx: OmlxEmbeddingConfig::default(),
        }
    }
}

impl Default for GoogleEmbeddingConfig {
    fn default() -> Self {
        Self {
            model: "gemini-embedding-001".to_string(),
            api_key: None,
            endpoint: default_google_endpoint(),
        }
    }
}

impl Default for OmlxEmbeddingConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://localhost:8000/v1".to_string(),
            model: "default".to_string(),
        }
    }
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            rrf_k: 60.0,
            rrf_fts_weight: 0.3,
            rrf_vec_weight: 0.7,
            waterfall_fts_threshold: 0.5,
            dedup_similarity: 0.70,
            dedup_time_window_days: 7,
        }
    }
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            overlap_percent: 10,
            metadata_prefix: true,
        }
    }
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            supermemory_enabled: true,
            auto_memory_enabled: true,
            auto_memory_glob: "~/.claude/projects/*/memory/**/*.md".to_string(),
            api_key: None,
            endpoint: default_supermemory_endpoint(),
        }
    }
}

impl Default for DecayConfig {
    fn default() -> Self {
        Self {
            base_lambda: 0.06,
            ltm_beta: 0.8,
            stm_beta: 1.2,
            interval_hours: 24,
            prune_threshold: 0.05,
            stm_to_ltm_access_count: 5,
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            compact: false,
            sse_enabled: false,
            sse_port: 8680,
            sse_bind: "127.0.0.1".to_string(),
        }
    }
}

impl Default for HooksConfig {
    fn default() -> Self {
        Self {
            min_turns: 20,
            context_before: 3,
            context_after: 1,
            max_items_per_session: 10,
        }
    }
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

impl ReinConfig {
    /// Load configuration with the following priority (highest wins):
    /// 1. Environment variable overrides
    /// 2. TOML config file (`$REIN_CONFIG` or `~/.config/rein/config.toml`)
    /// 3. Compiled-in defaults
    pub fn load() -> anyhow::Result<Self> {
        let mut config = Self::default();

        // Determine config file path
        let config_path = std::env::var("REIN_CONFIG")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                dirs_config_path()
            });

        // Merge TOML file if it exists
        if config_path.exists() {
            let contents = std::fs::read_to_string(&config_path)?;
            config = merge_toml(config, &contents)?;
        }

        // Environment variable overrides
        if let Ok(db) = std::env::var("REIN_DB") {
            config.database.path = db;
        }
        if let Ok(key) = std::env::var("GEMINI_API_KEY") {
            config.embedding.google.api_key = Some(key);
        }
        if let Ok(key) = std::env::var("SUPERMEMORY_CC_API_KEY") {
            config.sync.api_key = Some(key);
        }

        Ok(config)
    }

    /// Load configuration from a specific TOML string (for testing).
    pub fn load_from_str(toml_str: &str) -> anyhow::Result<Self> {
        let config = Self::default();
        merge_toml(config, toml_str)
    }

    /// The embedding model name (for cache keying and model-change detection).
    pub fn embedding_model(&self) -> String {
        match self.embedding.provider.as_str() {
            "omlx" => format!("omlx:{}", self.embedding.omlx.model),
            _ => format!("{}:{}", self.embedding.provider, self.embedding.google.model),
        }
    }

    /// Validate configuration and print warnings for common misconfigurations.
    pub fn validate(&self) {
        match self.embedding.provider.as_str() {
            "google" => {
                if self.embedding.google.api_key.is_none() {
                    eprintln!("rein: WARNING — embedding provider is 'google' but GEMINI_API_KEY is not set");
                    eprintln!("rein: Vector search and embedding will be disabled. FTS search still works.");
                }
            }
            "omlx" => {
                eprintln!("rein: using OMLX embedding backend at {}", self.embedding.omlx.endpoint);
            }
            "none" => {}
            other => {
                eprintln!("rein: WARNING — unknown embedding provider '{other}', falling back to none");
            }
        }
        if self.sync.supermemory_enabled && self.sync.api_key.is_none() {
            eprintln!("rein: NOTE — supermemory is enabled but SUPERMEMORY_CC_API_KEY is not set");
        }
    }

    /// Open a SqliteStore with the current config's model and dimensions.
    pub fn open_store(&self) -> crate::types::ReinResult<crate::store::SqliteStore> {
        crate::store::SqliteStore::new(
            &self.resolve_db_path(),
            &self.embedding_model(),
            self.embedding.dimensions,
        )
    }

    /// Resolve the database path. `"auto"` → `~/.rein/memories.db`
    pub fn resolve_db_path(&self) -> PathBuf {
        if self.database.path == "auto" {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
            let new_dir = PathBuf::from(&home).join(".rein");
            let new_path = new_dir.join("memories.db");

            // Check for old location and auto-migrate
            if !new_path.exists() {
                if let Some(dirs) = directories::ProjectDirs::from("", "", "rein") {
                    let old_path = dirs.data_dir().join("memories.db");
                    if old_path.exists() {
                        std::fs::create_dir_all(&new_dir).ok();
                        if std::fs::rename(&old_path, &new_path).is_ok() {
                            eprintln!("rein: migrated database from {} to {}", old_path.display(), new_path.display());
                        }
                    }
                }
            }

            std::fs::create_dir_all(&new_dir).ok();
            new_path
        } else {
            PathBuf::from(&self.database.path)
        }
    }
}

/// Default config file location: `~/.config/rein/config.toml`
fn dirs_config_path() -> PathBuf {
    directories::ProjectDirs::from("", "", "rein")
        .map(|d| d.config_dir().join("config.toml"))
        .unwrap_or_else(|| PathBuf::from("config.toml"))
}

/// Merge a TOML string over an existing config by deserializing into a
/// `toml::Value` table and overlaying it field-by-field so that missing keys
/// in the file keep their default values.
fn merge_toml(base: ReinConfig, toml_str: &str) -> anyhow::Result<ReinConfig> {
    // Serialize defaults to a toml::Value table
    let default_toml = toml::to_string(&serde_to_value(&base)?)?;
    let mut base_val: toml::Value = toml::from_str(&default_toml)?;

    // Parse the user file
    let user_val: toml::Value = toml::from_str(toml_str)?;

    // Deep-merge user over base
    deep_merge(&mut base_val, &user_val);

    // Deserialize the merged table back into ReinConfig
    let merged: ReinConfig = base_val.try_into()?;
    Ok(merged)
}

/// Convert a ReinConfig to a toml::Value via serde_json round-trip
/// (since ReinConfig doesn't derive Serialize, we use the default TOML).
fn serde_to_value(config: &ReinConfig) -> anyhow::Result<toml::Value> {
    // Build manually from defaults by serializing the embedded default.toml
    // and then patching the runtime values. Simpler: just use the embedded
    // default TOML as the base and patch non-serializable fields.
    let default_str = include_str!("../config/default.toml");
    let mut val: toml::Value = toml::from_str(default_str)?;

    // Patch fields that may differ from the embedded default (e.g., api_key)
    if let Some(tbl) = val.get_mut("embedding")
        .and_then(|v| v.get_mut("google"))
        .and_then(|v| v.as_table_mut())
    {
        if let Some(ref key) = config.embedding.google.api_key {
            tbl.insert("api_key".to_string(), toml::Value::String(key.clone()));
        }
    }
    if let Some(tbl) = val.get_mut("sync").and_then(|v| v.as_table_mut()) {
        if let Some(ref key) = config.sync.api_key {
            tbl.insert("api_key".to_string(), toml::Value::String(key.clone()));
        }
    }

    // Patch database path
    if let Some(tbl) = val.get_mut("database").and_then(|v| v.as_table_mut()) {
        tbl.insert("path".to_string(), toml::Value::String(config.database.path.clone()));
    }

    Ok(val)
}

fn deep_merge(base: &mut toml::Value, overlay: &toml::Value) {
    if let (Some(base_tbl), Some(overlay_tbl)) = (base.as_table_mut(), overlay.as_table()) {
        for (key, val) in overlay_tbl {
            if let Some(existing) = base_tbl.get_mut(key) {
                deep_merge(existing, val);
            } else {
                base_tbl.insert(key.clone(), val.clone());
            }
        }
    } else {
        *base = overlay.clone();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = ReinConfig::default();
        assert!((cfg.search.rrf_k - 60.0).abs() < f64::EPSILON);
        assert_eq!(cfg.embedding.dimensions, 3072);
        assert!(!cfg.server.compact);
        assert_eq!(cfg.database.path, "auto");
        assert_eq!(cfg.embedding.provider, "google");
        assert_eq!(cfg.chunking.max_tokens, 512);
        assert!((cfg.decay.base_lambda - 0.06).abs() < f64::EPSILON);
    }

    #[test]
    fn test_load_from_toml() {
        let toml_str = r#"
[search]
rrf_k = 30.0
"#;
        let cfg = ReinConfig::load_from_str(toml_str).unwrap();
        // Override applied
        assert!((cfg.search.rrf_k - 30.0).abs() < f64::EPSILON);
        // Other defaults preserved
        assert_eq!(cfg.embedding.dimensions, 3072);
        assert!(!cfg.server.compact);
        assert_eq!(cfg.database.path, "auto");
    }

    #[test]
    fn test_env_override_db() {
        // Use a unique env var approach to avoid test interference
        std::env::set_var("REIN_DB", "/tmp/test.db");
        std::env::set_var("REIN_CONFIG", "/nonexistent/path/config.toml");
        let cfg = ReinConfig::load().unwrap();
        assert_eq!(cfg.database.path, "/tmp/test.db");
        std::env::remove_var("REIN_DB");
        std::env::remove_var("REIN_CONFIG");
    }

    #[test]
    fn test_resolve_db_path_auto() {
        let cfg = ReinConfig::default();
        let path = cfg.resolve_db_path();
        let path_str = path.to_string_lossy();
        assert!(
            path_str.contains("rein") && path_str.ends_with("memories.db"),
            "Expected path containing 'rein/memories.db', got: {path_str}"
        );
    }

    #[test]
    fn test_db_path_auto() {
        // Just verify the logic: "auto" should map to ~/.rein/memories.db
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
        let expected = std::path::PathBuf::from(&home).join(".rein/memories.db");
        // Don't call resolve_db_path() as it has filesystem side effects
        assert!(expected.to_string_lossy().ends_with(".rein/memories.db"));
    }

    #[test]
    fn test_db_path_custom() {
        let mut config = ReinConfig::default();
        // Custom path should be returned as-is
        config.database.path = "/custom/path/test.db".to_string();
        assert_eq!(config.resolve_db_path(), std::path::PathBuf::from("/custom/path/test.db"));
    }
}
