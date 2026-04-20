//! Test-only op handlers for path-template framework validation.
//!
//! Compiled only when the `test-support` feature is active. Integration tests
//! in `tests/phase_2_5_path_template.rs` declare `rein` with `features =
//! ["test-support"]` (via `dev-dependencies`) so inventory entries are visible.
//!
//! Two ops are provided:
//!   - `__test_path_template`  — GET, Public auth, echoes the {id} param.
//!   - `__test_path_template_mut` — POST, MutationMarker auth, echoes {id}.
//!     Used by the H3 regression test (templated route auth checked pre-body).

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
        description = "Test-only op: echoes the {id} path param.",
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

    #[op(
        name = "__test_path_template_mut",
        category = "diagnostics",
        description = "Test-only op: mutation-auth templated route for H3 regression tests.",
        rest(method = "POST", path = "/api/test_path_template_mut/{id}"),
        auth = "mutation_marker",
    )]
    pub fn __test_path_template_mut(
        &self,
        params: TestPathTemplateParams,
    ) -> ReinResult<TestPathTemplateOutput> {
        Ok(TestPathTemplateOutput {
            echoed_id: params.id,
        })
    }
}
