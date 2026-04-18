//! Op error classification surface.
//!
//! `OpsErrorKind` maps semantic failure categories to HTTP status codes for
//! the REST adapter. Phase 1 stores kind alongside the original error via
//! a best-effort classifier; Phase 2 ops (memory NotFound, auth Unauthorized,
//! etc.) will plumb kinds through explicit `with_kind` calls — see TODO below.

use hyper::StatusCode;

use crate::types::ReinError;

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

impl OpsErrorKind {
    pub fn status_code(&self) -> StatusCode {
        match self {
            OpsErrorKind::BadRequest => StatusCode::BAD_REQUEST,
            OpsErrorKind::Unauthorized => StatusCode::UNAUTHORIZED,
            OpsErrorKind::Forbidden => StatusCode::FORBIDDEN,
            OpsErrorKind::NotFound => StatusCode::NOT_FOUND,
            OpsErrorKind::Conflict => StatusCode::CONFLICT,
            OpsErrorKind::Internal => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl ReinError {
    /// Default classification derived from the enum variant.
    /// Ops can override with `with_kind` when they have domain context
    /// (e.g. "parse failure on user input" → `BadRequest`).
    pub fn kind(&self) -> OpsErrorKind {
        match self {
            ReinError::NotFound(_) => OpsErrorKind::NotFound,
            _ => OpsErrorKind::Internal,
        }
    }

    // TODO(A1 Phase 2): plumb an optional kind override through ReinError so
    // handlers can tag auth/bad-input/conflict cases without introducing a
    // new error type. For Phase 1 stats/health both return Internal on
    // failure, so kind() is sufficient.
    pub fn with_kind(self, _kind: OpsErrorKind) -> Self {
        self
    }
}
