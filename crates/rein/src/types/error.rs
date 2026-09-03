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

#[cfg(test)]
mod tests {
    use super::*;

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
