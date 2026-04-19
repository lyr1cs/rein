//! HTTP status-code mapping for `OpsErrorKind`.
//!
//! The enum itself lives in `types/error.rs` so `ReinError` can carry a
//! kind without the types module taking a dependency on hyper. This file
//! adds the REST-adapter-facing `status_code()` mapping and re-exports
//! the enum so existing `use crate::ops::OpsErrorKind` imports keep
//! working.

use hyper::StatusCode;

pub use crate::types::OpsErrorKind;

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
