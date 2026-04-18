//! Render traits for op outputs.
//!
//! Each op output type impls these via `#[derive(OpsRender)]` (added in Phase 1.2),
//! or manually for special cases. Phase 1.1 just defines the traits + a shared
//! markdown walker; the PoC types (stats, health) will manually impl `IntoJson`
//! and `IntoCliText` in Phase 1.3 / 1.4.

use serde_json::Value;

pub trait IntoJson {
    fn to_json(&self) -> Value;
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
