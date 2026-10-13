//! Portable MCP input schemas + schema-guided argument coercion.
//!
//! rein serves five MCP clients (Claude Code, Claude Desktop, Cursor, Gemini,
//! Codex). Each one re-encodes a tool's `inputSchema` into its own LLM
//! function-calling format, and the intersection of what they all accept is
//! much smaller than JSON Schema 2020-12:
//!
//! - no `$ref` / `$defs` (Gemini rejects them; Claude Code does not resolve
//!   them when flattening)
//! - no top-level `oneOf` / `anyOf` / `allOf` (Claude Code flattens the root
//!   to the first branch's direct `properties`, drops the rest, and sets
//!   `additionalProperties: true`; undeclared params are then stringified)
//! - no `type: ["x", "null"]` arrays, no `const`, no `format` (Gemini)
//!
//! [`mcp_schema_for`] is the single generator every `#[op]` uses for its MCP
//! surface; [`make_portable`] is the pure rewrite it applies. [`coerce_args`]
//! is the receive-side counterpart: when a client still ships a declared
//! array / object / boolean / number as a JSON string (or an integer as a
//! whole float), it is decoded before deserialization so one misbehaving
//! client cannot take a tool offline.

use serde_json::Value;

/// Generate the MCP-facing input schema for a params type.
///
/// Inlines every subschema and rewrites the result with [`make_portable`].
/// The root is always an object schema with a `properties` map, even for
/// ops without parameters, so every client sees the same top-level shape.
pub fn mcp_schema_for<T: ?Sized + schemars::JsonSchema>() -> schemars::Schema {
    let mut settings = schemars::generate::SchemaSettings::draft2020_12();
    settings.inline_subschemas = true;
    let schema = settings.into_generator().into_root_schema_for::<T>();
    let mut value: Value = schema.into();
    make_portable(&mut value);
    let mut root = match value {
        Value::Object(map) => map,
        _ => serde_json::Map::new(),
    };
    root.entry("type")
        .or_insert_with(|| Value::String("object".to_string()));
    if root.get("type") == Some(&Value::String("object".to_string())) {
        root.entry("properties")
            .or_insert_with(|| Value::Object(serde_json::Map::new()));
    }
    schemars::Schema::from(root)
}

/// Keys whose values are maps of subschemas.
const SCHEMA_MAP_KEYS: &[&str] = &["properties", "patternProperties", "$defs", "definitions"];
/// Keys whose values are a single subschema.
const SCHEMA_KEYS: &[&str] = &[
    "items",
    "additionalProperties",
    "not",
    "propertyNames",
    "contains",
];
/// Keys whose values are arrays of subschemas.
const SCHEMA_LIST_KEYS: &[&str] = &["anyOf", "oneOf", "allOf", "prefixItems"];
/// Keys no portable client understands (or that carry nothing after
/// inlining). `format` goes because Gemini rejects values such as `uint64`;
/// `minimum` stays and carries the same hint.
const STRIPPED_KEYS: &[&str] = &["$schema", "$id", "$defs", "definitions", "format"];

/// Rewrite a JSON Schema value in place into the client-portable subset
/// described in the module docs. Only schema-bearing keys are traversed;
/// `enum` / `default` / `examples` members are data and are left alone.
pub fn make_portable(value: &mut Value) {
    let Some(obj) = value.as_object_mut() else {
        return;
    };

    // Children first, so a member merged into its parent below is already
    // in portable form.
    for key in SCHEMA_MAP_KEYS {
        if let Some(Value::Object(map)) = obj.get_mut(*key) {
            for child in map.values_mut() {
                make_portable(child);
            }
        }
    }
    for key in SCHEMA_KEYS {
        if let Some(child) = obj.get_mut(*key) {
            if child.is_object() {
                make_portable(child);
            }
        }
    }
    for key in SCHEMA_LIST_KEYS {
        if let Some(Value::Array(items)) = obj.get_mut(*key) {
            for child in items.iter_mut() {
                make_portable(child);
            }
        }
    }

    for key in STRIPPED_KEYS {
        obj.remove(*key);
    }

    // `type: ["x", "null"]` → `type: "x"`; a genuine multi-type union
    // becomes `anyOf` of single-type branches.
    if let Some(Value::Array(types)) = obj.get("type").cloned() {
        let kept: Vec<Value> = types
            .into_iter()
            .filter(|t| t.as_str() != Some("null"))
            .collect();
        obj.remove("type");
        match kept.len() {
            0 => {}
            1 => {
                obj.insert(
                    "type".to_string(),
                    kept.into_iter().next().unwrap_or(Value::Null),
                );
            }
            _ => {
                let branches = kept
                    .into_iter()
                    .map(|t| serde_json::json!({ "type": t }))
                    .collect();
                push_any_of(obj, branches);
            }
        }
    }

    if obj.get("default").map(Value::is_null).unwrap_or(false) {
        obj.remove("default");
    }

    if let Some(constant) = obj.remove("const") {
        obj.insert("enum".to_string(), Value::Array(vec![constant]));
    }

    if let Some(Value::Array(branches)) = obj.remove("oneOf") {
        push_any_of(obj, branches);
    }

    // `anyOf: [X, {type: null}]` is how an inlined `Option<Struct>` renders;
    // drop the null branch and, if exactly one branch is left, merge it into
    // the parent so the property has a direct `type`.
    if let Some(Value::Array(branches)) = obj.remove("anyOf") {
        let kept: Vec<Value> = branches
            .into_iter()
            .filter(|b| !is_null_schema(b))
            .collect();
        if kept.len() == 1 {
            if let Some(Value::Object(member)) = kept.into_iter().next() {
                merge_schema_into(obj, member);
            }
        } else if !kept.is_empty() {
            obj.insert("anyOf".to_string(), Value::Array(kept));
        }
    }

    // `allOf: [{object schema}]` → fields hoisted into the parent.
    if let Some(Value::Array(members)) = obj.remove("allOf") {
        let mut leftover = Vec::new();
        for member in members {
            match member {
                Value::Object(map) if is_object_schema(&map) => merge_schema_into(obj, map),
                other => leftover.push(other),
            }
        }
        if !leftover.is_empty() {
            obj.insert("allOf".to_string(), Value::Array(leftover));
        }
    }

    if obj.contains_key("properties") && !obj.contains_key("type") {
        obj.insert("type".to_string(), Value::String("object".to_string()));
    }
}

fn push_any_of(obj: &mut serde_json::Map<String, Value>, mut branches: Vec<Value>) {
    match obj.get_mut("anyOf") {
        Some(Value::Array(existing)) => existing.append(&mut branches),
        _ => {
            obj.insert("anyOf".to_string(), Value::Array(branches));
        }
    }
}

fn is_null_schema(schema: &Value) -> bool {
    schema.get("type").and_then(Value::as_str) == Some("null")
}

fn is_object_schema(schema: &serde_json::Map<String, Value>) -> bool {
    schema.contains_key("properties")
        || schema.contains_key("required")
        || schema.get("type").and_then(Value::as_str) == Some("object")
}

/// Merge `member` into `parent`. Parent keys win on conflict, except that
/// `properties` maps are unioned (parent entry wins per property) and
/// `required` lists are unioned.
fn merge_schema_into(
    parent: &mut serde_json::Map<String, Value>,
    member: serde_json::Map<String, Value>,
) {
    for (key, value) in member {
        match key.as_str() {
            "properties" => {
                let Value::Object(incoming) = value else {
                    continue;
                };
                let slot = parent
                    .entry("properties")
                    .or_insert_with(|| Value::Object(serde_json::Map::new()));
                if let Value::Object(existing) = slot {
                    for (name, prop) in incoming {
                        existing.entry(name).or_insert(prop);
                    }
                }
            }
            "required" => {
                let Value::Array(incoming) = value else {
                    continue;
                };
                let slot = parent
                    .entry("required")
                    .or_insert_with(|| Value::Array(Vec::new()));
                if let Value::Array(existing) = slot {
                    for name in incoming {
                        if !existing.contains(&name) {
                            existing.push(name);
                        }
                    }
                }
            }
            _ => {
                parent.entry(key).or_insert(value);
            }
        }
    }
}

/// Decode string-encoded (or float-encoded) arguments whose declared type in
/// `schema` is array / object / boolean / integer / number. Returns the names
/// of the parameters that were rewritten so the caller can log them.
///
/// Only a string that parses as JSON *of the declared type* is substituted;
/// anything else is left for the op's own deserializer to reject with its
/// normal error.
pub fn coerce_args(schema: &Value, args: &mut Value) -> Vec<String> {
    let mut coerced = Vec::new();
    let Some(props) = schema.get("properties").and_then(Value::as_object) else {
        return coerced;
    };
    let Some(map) = args.as_object_mut() else {
        return coerced;
    };
    for (name, value) in map.iter_mut() {
        let Some(declared) = props
            .get(name)
            .and_then(|p| p.get("type"))
            .and_then(Value::as_str)
        else {
            continue;
        };
        if let Some(fixed) = coerce_value(declared, value) {
            *value = fixed;
            coerced.push(name.clone());
        }
    }
    coerced
}

fn coerce_value(declared: &str, value: &Value) -> Option<Value> {
    match value {
        Value::String(text) => {
            if !matches!(
                declared,
                "array" | "object" | "boolean" | "integer" | "number"
            ) {
                return None;
            }
            let parsed: Value = serde_json::from_str(text).ok()?;
            match (declared, &parsed) {
                ("array", Value::Array(_))
                | ("object", Value::Object(_))
                | ("boolean", Value::Bool(_))
                | ("number", Value::Number(_)) => Some(parsed),
                ("integer", Value::Number(n)) => {
                    if n.is_i64() || n.is_u64() {
                        Some(parsed)
                    } else {
                        whole_float_to_integer(n)
                    }
                }
                _ => None,
            }
        }
        Value::Number(n) if declared == "integer" && !(n.is_i64() || n.is_u64()) => {
            whole_float_to_integer(n)
        }
        _ => None,
    }
}

/// `5.0` → `5`; anything with a fractional part (or outside the exactly
/// representable range) is left alone.
fn whole_float_to_integer(n: &serde_json::Number) -> Option<Value> {
    let f = n.as_f64()?;
    const EXACT: f64 = 9_007_199_254_740_992.0; // 2^53
    if !f.is_finite() || f.fract() != 0.0 || f.abs() >= EXACT {
        return None;
    }
    Some(Value::from(f as i64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn portable(v: Value) -> Value {
        let mut v = v;
        make_portable(&mut v);
        v
    }

    // ── make_portable ────────────────────────────────────────────────────

    #[test]
    fn portable_collapses_nullable_type_array_and_drops_null_default() {
        let out = portable(json!({
            "type": "object",
            "properties": {
                "helpful": {"type": ["boolean", "null"], "default": null, "description": "d"}
            }
        }));
        assert_eq!(out["properties"]["helpful"]["type"], json!("boolean"));
        assert!(out["properties"]["helpful"].get("default").is_none());
        assert_eq!(out["properties"]["helpful"]["description"], json!("d"));
    }

    #[test]
    fn portable_merges_single_object_anyof_with_null_into_parent() {
        let out = portable(json!({
            "type": "object",
            "properties": {
                "metadata": {
                    "anyOf": [
                        {"type": "object", "properties": {"cluster_id": {"type": ["integer", "null"]}}},
                        {"type": "null"}
                    ],
                    "default": null,
                    "description": "meta"
                }
            }
        }));
        let md = &out["properties"]["metadata"];
        assert!(md.get("anyOf").is_none(), "{md}");
        assert_eq!(md["type"], json!("object"));
        assert_eq!(md["properties"]["cluster_id"]["type"], json!("integer"));
        assert_eq!(md["description"], json!("meta"));
    }

    #[test]
    fn portable_merges_allof_object_members_into_parent() {
        let out = portable(json!({
            "type": "object",
            "allOf": [
                {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
            ],
            "properties": {"kind": {"type": "string", "enum": ["x"]}},
            "required": ["kind"]
        }));
        assert!(out.get("allOf").is_none(), "{out}");
        assert_eq!(out["properties"]["a"]["type"], json!("string"));
        assert_eq!(out["properties"]["kind"]["enum"], json!(["x"]));
        let mut req: Vec<&str> = out["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        req.sort();
        assert_eq!(req, vec!["a", "kind"]);
    }

    #[test]
    fn portable_rewrites_oneof_to_anyof_and_const_to_enum() {
        let out = portable(json!({
            "oneOf": [
                {"type": "object", "properties": {"kind": {"type": "string", "const": "viewed"}}},
                {"type": "object", "properties": {"kind": {"type": "string", "const": "clicked"}}}
            ]
        }));
        assert!(out.get("oneOf").is_none(), "{out}");
        let any = out["anyOf"].as_array().expect("anyOf");
        assert_eq!(any.len(), 2);
        assert_eq!(any[0]["properties"]["kind"]["enum"], json!(["viewed"]));
        assert!(any[0]["properties"]["kind"].get("const").is_none());
    }

    #[test]
    fn portable_strips_schema_format_and_defs_keys() {
        let out = portable(json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$defs": {"Unused": {"type": "string"}},
            "type": "object",
            "properties": {
                "dwell_ms": {"type": "integer", "format": "uint64", "minimum": 0}
            }
        }));
        assert!(out.get("$schema").is_none());
        assert!(out.get("$defs").is_none());
        assert!(out["properties"]["dwell_ms"].get("format").is_none());
        assert_eq!(out["properties"]["dwell_ms"]["minimum"], json!(0));
    }

    #[test]
    fn portable_adds_object_type_when_properties_present_without_type() {
        let out = portable(json!({"properties": {"a": {"type": "string"}}}));
        assert_eq!(out["type"], json!("object"));
    }

    #[test]
    fn portable_recurses_into_array_items() {
        let out = portable(json!({
            "type": "object",
            "properties": {
                "turns": {
                    "type": "array",
                    "items": {"$schema": "x", "properties": {"role": {"type": ["string", "null"]}}}
                }
            }
        }));
        let items = &out["properties"]["turns"]["items"];
        assert_eq!(items["type"], json!("object"));
        assert_eq!(items["properties"]["role"]["type"], json!("string"));
        assert!(items.get("$schema").is_none());
    }

    #[test]
    fn portable_leaves_enum_values_untouched() {
        // `enum` members are data, not schemas: a member that happens to look
        // like a schema object must not be rewritten.
        let out = portable(json!({
            "type": "object",
            "properties": {"fmt": {"type": "string", "enum": ["md", "json"]}},
            "default": {"fmt": "md"}
        }));
        assert_eq!(out["properties"]["fmt"]["enum"], json!(["md", "json"]));
        assert_eq!(out["default"], json!({"fmt": "md"}));
    }

    // ── mcp_schema_for ───────────────────────────────────────────────────

    #[derive(schemars::JsonSchema)]
    #[allow(dead_code)]
    struct Inner {
        n: Option<u32>,
    }

    #[derive(schemars::JsonSchema)]
    #[allow(dead_code)]
    struct Outer {
        ids: Vec<String>,
        flag: Option<bool>,
        inner: Option<Inner>,
    }

    #[test]
    fn mcp_schema_for_inlines_refs_and_is_portable() {
        let v: Value = mcp_schema_for::<Outer>().into();
        assert_eq!(v["type"], json!("object"));
        assert!(v.get("$defs").is_none(), "{v}");
        assert!(v.get("$schema").is_none(), "{v}");
        assert_eq!(v["properties"]["ids"]["type"], json!("array"));
        assert_eq!(v["properties"]["flag"]["type"], json!("boolean"));
        assert_eq!(v["properties"]["inner"]["type"], json!("object"));
        assert_eq!(
            v["properties"]["inner"]["properties"]["n"]["type"],
            json!("integer")
        );
        assert!(v["properties"]["inner"].get("anyOf").is_none(), "{v}");
        assert_eq!(v["required"], json!(["ids"]));
    }

    // ── coerce_args ──────────────────────────────────────────────────────

    fn feedback_like_schema() -> Value {
        json!({
            "type": "object",
            "properties": {
                "memory_ids": {"type": "array", "items": {"type": "string"}},
                "helpful": {"type": "boolean"},
                "query": {"type": "string"},
                "limit": {"type": "integer"},
                "score": {"type": "number"},
                "metadata": {"type": "object"}
            }
        })
    }

    #[test]
    fn coerce_decodes_stringified_array_and_boolean() {
        // Exact payload shape Claude Code shipped on 2026-10-13.
        let mut args = json!({
            "kind": "access",
            "memory_ids": "[\"01KR0S0GGBJ18AF0138VKPWEJ9\",\"01KR15H6TKSQ4J0E26ET1ED2ZJ\"]",
            "helpful": "true",
            "query": "plain string stays"
        });
        let mut coerced = coerce_args(&feedback_like_schema(), &mut args);
        coerced.sort();
        assert_eq!(coerced, vec!["helpful", "memory_ids"]);
        assert_eq!(
            args["memory_ids"],
            json!(["01KR0S0GGBJ18AF0138VKPWEJ9", "01KR15H6TKSQ4J0E26ET1ED2ZJ"])
        );
        assert_eq!(args["helpful"], json!(true));
        assert_eq!(args["query"], json!("plain string stays"));
        assert_eq!(args["kind"], json!("access"));
    }

    #[test]
    fn coerce_decodes_stringified_integer_number_and_object() {
        let mut args = json!({
            "limit": "5",
            "score": "0.25",
            "metadata": "{\"cluster_id\": 3}"
        });
        let mut coerced = coerce_args(&feedback_like_schema(), &mut args);
        coerced.sort();
        assert_eq!(coerced, vec!["limit", "metadata", "score"]);
        assert_eq!(args["limit"], json!(5));
        assert_eq!(args["score"], json!(0.25));
        assert_eq!(args["metadata"], json!({"cluster_id": 3}));
    }

    #[test]
    fn coerce_turns_whole_float_into_integer_for_integer_params() {
        let mut args = json!({"limit": 5.0, "score": 2.5});
        let coerced = coerce_args(&feedback_like_schema(), &mut args);
        assert_eq!(coerced, vec!["limit"]);
        assert!(
            args["limit"].is_i64() || args["limit"].is_u64(),
            "{}",
            args["limit"]
        );
        assert_eq!(args["limit"], json!(5));
        assert_eq!(args["score"], json!(2.5));
    }

    #[test]
    fn coerce_leaves_mismatched_and_undeclared_values_alone() {
        let mut args = json!({
            "memory_ids": "not json at all",
            "helpful": "yes",
            "limit": "1.5",
            "unknown": "[1,2]",
            "query": 42
        });
        let before = args.clone();
        let coerced = coerce_args(&feedback_like_schema(), &mut args);
        assert!(coerced.is_empty(), "{coerced:?}");
        assert_eq!(args, before);
    }

    #[test]
    fn coerce_is_noop_for_well_typed_payload() {
        let mut args = json!({"memory_ids": ["a"], "helpful": false, "limit": 3});
        let before = args.clone();
        assert!(coerce_args(&feedback_like_schema(), &mut args).is_empty());
        assert_eq!(args, before);
    }

    #[test]
    fn coerce_ignores_non_object_args_and_schemas() {
        let mut args = json!("nope");
        assert!(coerce_args(&feedback_like_schema(), &mut args).is_empty());
        let mut args = json!({"helpful": "true"});
        assert!(coerce_args(&json!(null), &mut args).is_empty());
        assert_eq!(args["helpful"], json!("true"));
    }
}
