use thiserror::Error;

#[derive(Debug, Error)]
pub enum ReinError {
    #[error("not found: {0}")]
    NotFound(String),

    #[error("database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("config error: {0}")]
    Config(String),

    #[error("embedding error: {0}")]
    Embedding(String),

    #[error("extraction error: {0}")]
    Extract(String),

    #[error("network error: {0}")]
    Network(#[from] reqwest::Error),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub type ReinResult<T> = Result<T, ReinError>;

/// Truncate a string for error messages (avoids dumping huge API responses into logs).
pub fn truncate_for_error(s: &str, max_chars: usize) -> String {
    s.chars().take(max_chars).collect()
}
