//! Test-only op for Phase 2.5 path-template framework validation.
//!
//! Provides a minimal `#[op]` that uses `rest(path = "/api/test_path_template/{id}")`
//! so integration tests in `tests/phase_2_5_path_template.rs` can exercise
//! T2 (segment emission), T3 (dispatcher template match), and T4 (param merge)
//! without requiring a real migrated op.
//!
//! Registered unconditionally (not gated on `#[cfg(test)]`) because integration
//! tests link to the library crate and cannot see `#[cfg(test)]` modules from it.
//! The op name is prefixed `__test_` to signal test-only status to reviewers.
//!
//! Phase 3 cleanup: delete this file once real path-template ops are migrated and
//! the framework is validated by production ops themselves.

use rein_macros::op;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::ops::{IntoCliText, IntoMarkdown, OpsRuntime};
use crate::types::ReinResult;

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct TestPathTemplateParams {
    pub id: String,
}

#[derive(Debug, Serialize)]
pub struct TestPathTemplateOutput {
    pub echoed_id: String,
}

impl IntoMarkdown for TestPathTemplateOutput {
    fn to_markdown(&self) -> String {
        format!("echoed_id: {}", self.echoed_id)
    }
}

impl IntoCliText for TestPathTemplateOutput {
    fn to_cli_text(&self) -> String {
        self.echoed_id.clone()
    }
}

impl OpsRuntime {
    #[op(
        name = "__test_path_template",
        category = "diagnostics",
        description = "Test-only op for path-template framework (T5). Echoes the {id} path param.",
        rest(method = "GET", path = "/api/test_path_template/{id}"),
    )]
    pub fn __test_path_template(
        &self,
        params: TestPathTemplateParams,
    ) -> ReinResult<TestPathTemplateOutput> {
        Ok(TestPathTemplateOutput {
            echoed_id: params.id,
        })
    }
}
