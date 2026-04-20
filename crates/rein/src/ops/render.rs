//! Render traits for op outputs.
//!
//! Each op output type impls these via `#[derive(OpsRender)]` (added in Phase 1.2),
//! or manually for special cases. Phase 1.1 just defines the traits + a shared
//! markdown walker; the PoC types (stats, health) will manually impl `IntoJson`
//! and `IntoCliText` in Phase 1.3 / 1.4.

use serde_json::Value;

pub trait IntoJson {
    fn to_json(&self) -> Value;

    /// Override to emit a raw body with a specific content-type instead of
    /// serializing `to_json()` as `application/json`. The REST dispatcher
    /// checks this hook first; returning `None` (the default) keeps the
    /// usual JSON contract. Phase 3 added this so ops like `memoir_export`
    /// can serve `text/plain` ascii/dot graph bodies from inventory
    /// without a dispatcher-side op-name guard.
    fn to_raw_response(&self) -> Option<(&'static str, Vec<u8>)> {
        None
    }
}

pub trait IntoMarkdown {
    fn to_markdown(&self) -> String;
}

pub trait IntoCliText {
    fn to_cli_text(&self) -> String;
}

/// Walk a `serde_json::Value` and produce structured markdown.
/// Default `IntoMarkdown` impl emitted by `#[derive(OpsRender)]` delegates here.
pub fn render_value_as_markdown(v: &Value, depth: usize) -> String {
    let indent = "  ".repeat(depth);
    match v {
        Value::Null => format!("{}null", indent),
        Value::Bool(b) => format!("{}{}", indent, b),
        Value::Number(n) => format!("{}{}", indent, n),
        Value::String(s) => format!("{}{}", indent, s),
        Value::Array(arr) => {
            if arr.is_empty() {
                return format!("{}(none)", indent);
            }
            arr.iter()
                .map(|v| {
                    format!(
                        "{}- {}",
                        indent,
                        render_value_as_markdown(v, depth + 1).trim_start()
                    )
                })
                .collect::<Vec<_>>()
                .join("\n")
        }
        Value::Object(obj) => obj
            .iter()
            .map(|(k, v)| {
                format!(
                    "{}**{}**: {}",
                    indent,
                    k,
                    render_value_as_markdown(v, depth + 1).trim_start()
                )
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}
