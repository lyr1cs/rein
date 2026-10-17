use thiserror::Error;

/// Semantic error category that survives propagation through `?`. REST
/// adapters map `OpsErrorKind` to HTTP status codes via
/// `OpsErrorKind::status_code` (defined in `ops/error.rs` so this crate
/// doesn't force hyper into every `types` consumer).
///
/// Handlers attach a kind via `ReinError::with_kind`. When unset, `kind()`
/// derives a best-effort classification from the variant (`NotFound` →
/// `NotFound`, everything else → `Internal`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OpsErrorKind {
    BadRequest,
    Unauthorized,
    Forbidden,
    NotFound,
    Conflict,
    Internal,
}

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

    /// A vector write was refused because the vector table is stamped for
    /// another embedding model, or holds rows of unknown provenance. The
    /// store/update paths degrade to "memory without a vector" on this
    /// variant instead of failing the memory write.
    #[error("vector provenance: {0}")]
    VectorProvenance(String),

    #[error("embedding error: {0}")]
    Embedding(String),

    #[error("extraction error: {0}")]
    Extract(String),

    #[error("network error: {0}")]
    Network(#[from] reqwest::Error),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// Wrapper carrying an explicit `OpsErrorKind` override. Created via
    /// `ReinError::with_kind` — handlers use this to tag a BadRequest /
    /// Forbidden / NotFound failure even when the underlying cause is
    /// one of the non-classifying variants above. Display delegates to
    /// the inner error so callers see the human-readable message; the
    /// kind is only consulted by `ReinError::kind()` for REST dispatch.
    #[error("{inner}")]
    Kinded {
        kind: OpsErrorKind,
        #[source]
        inner: Box<ReinError>,
    },
}

impl ReinError {
    /// Classify the error for the REST adapter. An explicit `Kinded` wrap
    /// always wins; otherwise fall back to variant-based heuristics.
    pub fn kind(&self) -> OpsErrorKind {
        match self {
            ReinError::Kinded { kind, .. } => *kind,
            ReinError::NotFound(_) => OpsErrorKind::NotFound,
            _ => OpsErrorKind::Internal,
        }
    }

    /// Tag the error with an explicit kind, preserving the original cause.
    /// Idempotent: calling twice replaces the outer kind without nesting.
    pub fn with_kind(self, kind: OpsErrorKind) -> Self {
        match self {
            ReinError::Kinded { inner, .. } => ReinError::Kinded { kind, inner },
            other => ReinError::Kinded {
                kind,
                inner: Box::new(other),
            },
        }
    }
}

pub type ReinResult<T> = Result<T, ReinError>;

/// Truncate a string for error messages (avoids dumping huge API responses into logs).
pub fn truncate_for_error(s: &str, max_chars: usize) -> String {
    s.chars().take(max_chars).collect()
}

/// Content-free summary of a non-success provider response, for error
/// messages that reach WARN logs and MCP clients at the default level.
///
/// Providers and proxies routinely echo the rejected request in the error
/// body, so embedding even a truncated body copies prompt / memory text into
/// every client's log on each provider hiccup. The message keeps the
/// operation, HTTP status and body length; the body preview is emitted at
/// DEBUG only (`REIN_LOG=debug`), which operators opt into deliberately.
pub fn provider_status_error(op: &str, status: reqwest::StatusCode, body: &str) -> String {
    let body_chars = body.chars().count();
    tracing::debug!(
        op,
        status = status.as_u16(),
        body_chars,
        body_preview = %truncate_for_error(body, 500),
        "provider returned a non-success status"
    );
    format!("{op} returned {status} (body {body_chars} chars; preview at debug level)")
}

/// Content-free summary of LLM output that failed to parse. Same contract as
/// [`provider_status_error`]: generated text can quote the prompt (and thus
/// memory content) verbatim, so it is emitted at DEBUG only.
pub fn unparseable_output_error(what: &str, output: &str) -> String {
    let output_chars = output.chars().count();
    tracing::debug!(
        what,
        output_chars,
        output_preview = %truncate_for_error(output, 200),
        "LLM output failed to parse"
    );
    format!(
        "failed to parse {what}: unparseable LLM output ({output_chars} chars; preview at debug level)"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_status_error_keeps_body_out_of_message() {
        let msg = provider_status_error(
            "Test API",
            reqwest::StatusCode::BAD_REQUEST,
            "echoed secret prompt",
        );
        assert!(!msg.contains("echoed secret prompt"), "{msg}");
        assert!(msg.starts_with("Test API returned 400"), "{msg}");
        assert!(msg.contains("body 20 chars"), "{msg}");
    }

    #[test]
    fn unparseable_output_error_keeps_output_out_of_message() {
        let msg = unparseable_output_error("dedup verdict", "密码是 hunter2");
        assert!(!msg.contains("hunter2"), "{msg}");
        assert!(msg.starts_with("failed to parse dedup verdict"), "{msg}");
        assert!(msg.contains("11 chars"), "{msg}");
    }

    #[test]
    fn kind_falls_back_to_not_found_variant() {
        let e = ReinError::NotFound("x".into());
        assert_eq!(e.kind(), OpsErrorKind::NotFound);
    }

    #[test]
    fn kind_defaults_to_internal_for_unclassified_variants() {
        let e = ReinError::Config("x".into());
        assert_eq!(e.kind(), OpsErrorKind::Internal);
    }

    #[test]
    fn with_kind_overrides_default_classification() {
        let e = ReinError::Config("x".into()).with_kind(OpsErrorKind::BadRequest);
        assert_eq!(e.kind(), OpsErrorKind::BadRequest);
        // Display passes through to inner
        assert_eq!(e.to_string(), "config error: x");
    }

    #[test]
    fn with_kind_replaces_instead_of_nesting() {
        let e = ReinError::Config("x".into())
            .with_kind(OpsErrorKind::BadRequest)
            .with_kind(OpsErrorKind::Conflict);
        assert_eq!(e.kind(), OpsErrorKind::Conflict);
        // Double-wrap would nest a second `Kinded` inside the first;
        // idempotent replacement keeps Display chain short.
        assert_eq!(e.to_string(), "config error: x");
    }

    #[test]
    fn with_kind_survives_question_mark_propagation() {
        // Proves the typed kind flows through `?` when callers use ReinResult.
        fn inner() -> ReinResult<()> {
            Err(ReinError::Config("bad input".into()).with_kind(OpsErrorKind::BadRequest))
        }
        fn outer() -> ReinResult<()> {
            inner()?;
            Ok(())
        }
        let e = outer().unwrap_err();
        assert_eq!(e.kind(), OpsErrorKind::BadRequest);
    }
}
