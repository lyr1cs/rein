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
//! is the receive-side counterpart: when an op rejects a payload at
//! deserialization and a client shipped a declared array / object /
//! boolean / number as a JSON string (or an integer as a whole float), the
//! values are decoded and the call retried once, so one misbehaving client
//! cannot take a tool offline. Payloads an op accepts as sent are never
//! rewritten.

use serde_json::Value;

/// Generate the MCP-facing input schema for a params type.
///
/// Inlines every subschema and rewrites the result with [`make_portable`].
/// The root is always an object schema with a `properties` map, even for
/// ops without parameters, so every client sees the same top-level shape.
///
/// Not covered: recursive params types (schemars keeps a `$ref` for those;
/// the portability lint rejects them) and OpenAI *strict* function calling
/// (which needs `additionalProperties: false` and every field required —
/// a different contract that no supported client applies to MCP tools).
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
///
/// Nullability is kept: `type: [x, "null"]` is the one type-array form
/// every client handles (Gemini CLI rewrites it to `nullable: true`), so an
/// `Option` field still admits an explicit null. An inlined `Option<Struct>`
/// (`anyOf: [{object}, {type: null}]`) becomes the object with
/// `type: ["object", "null"]`.
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

    // Type arrays: `[x, "null"]` in that order; a genuine multi-type union
    // becomes `anyOf` of single-type branches (plus a null branch).
    if let Some(Value::Array(types)) = obj.get("type").cloned() {
        let nullable = types.iter().any(|t| t.as_str() == Some("null"));
        let kept: Vec<Value> = types
            .into_iter()
            .filter(|t| t.as_str() != Some("null"))
            .collect();
        obj.remove("type");
        match kept.len() {
            0 => {
                if nullable {
                    obj.insert("type".to_string(), Value::String("null".to_string()));
                }
            }
            1 => {
                let only = kept.into_iter().next().unwrap_or(Value::Null);
                set_type(obj, only, nullable);
            }
            _ => {
                let mut branches: Vec<Value> = kept
                    .into_iter()
                    .map(|t| serde_json::json!({ "type": t }))
                    .collect();
                if nullable {
                    branches.push(serde_json::json!({ "type": "null" }));
                }
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
    // if exactly one non-null branch is left, merge it into the parent so
    // the property has a direct `type`, and keep the nullability on it.
    if let Some(Value::Array(branches)) = obj.remove("anyOf") {
        let nullable = branches.iter().any(is_null_schema);
        let mut kept: Vec<Value> = branches
            .into_iter()
            .filter(|b| !is_null_schema(b))
            .collect();
        if kept.len() == 1 {
            if let Some(Value::Object(member)) = kept.into_iter().next() {
                merge_schema_into(obj, member);
                if nullable {
                    match obj.get("type").cloned() {
                        Some(Value::String(t)) => set_type(obj, Value::String(t), true),
                        Some(Value::Array(_)) => {}
                        _ => {
                            if obj.contains_key("properties") {
                                set_type(obj, Value::String("object".to_string()), true);
                            }
                        }
                    }
                }
            }
        } else if !kept.is_empty() {
            if nullable {
                kept.push(serde_json::json!({ "type": "null" }));
            }
            obj.insert("anyOf".to_string(), Value::Array(kept));
        }
    }

    // `allOf: [{object schema}]` → fields hoisted into the parent, but only
    // when nothing is defined on both sides: a conflicting member stays
    // under `allOf` (and fails the portability lint) rather than silently
    // losing one of its constraints.
    if let Some(Value::Array(members)) = obj.remove("allOf") {
        let mut leftover = Vec::new();
        for member in members {
            match member {
                Value::Object(map) if is_object_schema(&map) && !conflicts_with(obj, &map) => {
                    merge_schema_into(obj, map)
                }
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

fn set_type(obj: &mut serde_json::Map<String, Value>, t: Value, nullable: bool) {
    let value = if nullable {
        Value::Array(vec![t, Value::String("null".to_string())])
    } else {
        t
    };
    obj.insert("type".to_string(), value);
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

/// True when `member` defines a property or `type` that `parent` already
/// defines differently, so hoisting would drop a constraint.
fn conflicts_with(
    parent: &serde_json::Map<String, Value>,
    member: &serde_json::Map<String, Value>,
) -> bool {
    if let (Some(Value::Object(parent_props)), Some(Value::Object(member_props))) =
        (parent.get("properties"), member.get("properties"))
    {
        for (name, schema) in member_props {
            if parent_props
                .get(name)
                .is_some_and(|existing| existing != schema)
            {
                return true;
            }
        }
    }
    matches!(
        (parent.get("type"), member.get("type")),
        (Some(parent_type), Some(member_type)) if parent_type != member_type
    )
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
/// `schema` is array / object / boolean / integer / number, recursing into
/// object properties (including those of `anyOf` branches) and array items.
/// Returns the dotted paths of the values that were rewritten so the caller
/// can log them.
///
/// Only a string that parses as JSON *of the declared type* is substituted;
/// anything else is left for the op's own deserializer to reject with its
/// normal error. Callers must apply this only after the op has rejected the
/// payload as sent (see `mcp::server`), so an op that legitimately accepts
/// a string for an array-typed field never has that string rewritten.
pub fn coerce_args(schema: &Value, args: &mut Value) -> Vec<String> {
    let mut coerced = Vec::new();
    if schema
        .get("properties")
        .and_then(Value::as_object)
        .is_none()
        || !args.is_object()
    {
        return coerced;
    }
    coerce_object_fields(schema, args, "", &mut coerced);
    coerced
}

fn coerce_node(node: &Value, value: &mut Value, path: &str, coerced: &mut Vec<String>) {
    if let Some(declared) = declared_type(node) {
        if let Some(fixed) = coerce_scalar(&declared, value) {
            *value = fixed;
            coerced.push(path.to_string());
        }
    }
    if value.is_object() {
        coerce_object_fields(node, value, path, coerced);
    } else if let Value::Array(items) = value {
        if let Some(item_schema) = node.get("items").filter(|s| s.is_object()) {
            for (i, item) in items.iter_mut().enumerate() {
                coerce_node(item_schema, item, &format!("{path}[{i}]"), coerced);
            }
        }
    }
}

fn coerce_object_fields(node: &Value, value: &mut Value, path: &str, coerced: &mut Vec<String>) {
    let props = child_properties(node);
    if props.is_empty() {
        return;
    }
    let Some(map) = value.as_object_mut() else {
        return;
    };
    for (name, child) in map.iter_mut() {
        let Some(child_schema) = props.get(name) else {
            continue;
        };
        let child_path = if path.is_empty() {
            name.clone()
        } else {
            format!("{path}.{name}")
        };
        coerce_node(child_schema, child, &child_path, coerced);
    }
}

/// Declared non-null type of a schema node: a `type` string, the non-null
/// half of a `[x, "null"]` pair, or the type shared by every non-null
/// `anyOf` branch. `None` when the node is untyped or a genuine union.
fn declared_type(node: &Value) -> Option<String> {
    match node.get("type") {
        Some(Value::String(t)) if t != "null" => return Some(t.clone()),
        Some(Value::String(_)) => return None,
        Some(Value::Array(types)) => {
            let non_null: Vec<&str> = types
                .iter()
                .filter_map(Value::as_str)
                .filter(|t| *t != "null")
                .collect();
            return (non_null.len() == 1).then(|| non_null[0].to_string());
        }
        _ => {}
    }
    let branches = node.get("anyOf")?.as_array()?;
    let mut common: Option<String> = None;
    for branch in branches.iter().filter(|b| !is_null_schema(b)) {
        let t = declared_type(branch)?;
        match &common {
            None => common = Some(t),
            Some(c) if *c == t => {}
            Some(_) => return None,
        }
    }
    common
}

/// Property schemas visible at `node`: its own `properties` plus those of
/// every `anyOf` branch (first definition wins).
fn child_properties(node: &Value) -> serde_json::Map<String, Value> {
    let mut out = serde_json::Map::new();
    let mut absorb = |source: &Value| {
        if let Some(props) = source.get("properties").and_then(Value::as_object) {
            for (name, schema) in props {
                out.entry(name.clone()).or_insert_with(|| schema.clone());
            }
        }
    };
    absorb(node);
    if let Some(branches) = node.get("anyOf").and_then(Value::as_array) {
        for branch in branches {
            absorb(branch);
        }
    }
    out
}

fn coerce_scalar(declared: &str, value: &Value) -> Option<Value> {
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
                    } else if lexically_integral(text) {
                        whole_float_to_integer(n)
                    } else {
                        None
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

/// True when a JSON number's spelling has no non-zero fractional digit
/// (`"7.000"`, `"1e2"`), so `f64` rounding cannot manufacture integrality
/// out of a value such as `"1.0000000000000001"`.
fn lexically_integral(text: &str) -> bool {
    let mantissa = text
        .trim()
        .split(['e', 'E'])
        .next()
        .unwrap_or("");
    match mantissa.split_once('.') {
        Some((_, fraction)) => fraction.chars().all(|c| c == '0'),
        None => true,
    }
}

/// `5.0` → `5`; anything with a fractional part (or outside the exactly
/// representable range) is left alone. Native floats have already lost
/// their spelling at parse time, so this is the documented compatibility
/// policy for clients that encode integers as floats.
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
    fn portable_keeps_nullable_type_array_and_drops_null_default() {
        // `[x, "null"]` is the one type-array form every client handles
        // (Gemini CLI rewrites it to `nullable: true`), so nullability is
        // kept: a client may still send an explicit null for an Option.
        let out = portable(json!({
            "type": "object",
            "properties": {
                "helpful": {"type": ["boolean", "null"], "default": null, "description": "d"}
            }
        }));
        assert_eq!(
            out["properties"]["helpful"]["type"],
            json!(["boolean", "null"])
        );
        assert!(out["properties"]["helpful"].get("default").is_none());
        assert_eq!(out["properties"]["helpful"]["description"], json!("d"));
    }

    #[test]
    fn portable_normalises_null_first_type_arrays() {
        let out = portable(json!({"type": ["null", "string"]}));
        assert_eq!(out["type"], json!(["string", "null"]));
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
        assert_eq!(md["type"], json!(["object", "null"]));
        assert_eq!(
            md["properties"]["cluster_id"]["type"],
            json!(["integer", "null"])
        );
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
    fn portable_keeps_conflicting_allof_members_unmerged() {
        // Two definitions of `a` cannot be hoisted without losing one; the
        // member stays under `allOf` so the portability lint fails loudly
        // instead of the schema silently weakening.
        let out = portable(json!({
            "type": "object",
            "allOf": [
                {"type": "object", "properties": {"a": {"type": "string"}}},
                {"type": "object", "properties": {"b": {"type": "boolean"}}}
            ],
            "properties": {"a": {"type": "integer"}}
        }));
        assert_eq!(out["properties"]["a"]["type"], json!("integer"));
        assert_eq!(out["properties"]["b"]["type"], json!("boolean"));
        let leftover = out["allOf"].as_array().expect("conflicting member kept");
        assert_eq!(leftover.len(), 1);
        assert_eq!(leftover[0]["properties"]["a"]["type"], json!("string"));
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
        assert_eq!(
            items["properties"]["role"]["type"],
            json!(["string", "null"])
        );
        assert!(items.get("$schema").is_none());
    }

    #[test]
    fn portable_leaves_enum_default_and_example_data_untouched() {
        // Members of `enum` / `default` / `examples` are data, not schemas:
        // an object member that happens to carry schema keywords must not
        // be rewritten.
        let looks_like_schema = json!({"type": ["x", "null"], "format": "zzz", "$ref": "#"});
        let out = portable(json!({
            "type": "object",
            "properties": {
                "fmt": {"type": "string", "enum": ["md", looks_like_schema.clone()]}
            },
            "default": {"fmt": looks_like_schema.clone()},
            "examples": [looks_like_schema.clone()]
        }));
        assert_eq!(out["properties"]["fmt"]["enum"][1], looks_like_schema);
        assert_eq!(out["default"]["fmt"], looks_like_schema);
        assert_eq!(out["examples"][0], looks_like_schema);
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
        assert_eq!(v["properties"]["flag"]["type"], json!(["boolean", "null"]));
        assert_eq!(v["properties"]["inner"]["type"], json!(["object", "null"]));
        assert_eq!(
            v["properties"]["inner"]["properties"]["n"]["type"],
            json!(["integer", "null"])
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
                "counts": {"type": ["array", "null"], "items": {"type": "integer"}},
                "helpful": {"type": ["boolean", "null"]},
                "query": {"type": ["string", "null"]},
                "limit": {"type": ["integer", "null"]},
                "score": {"type": "number"},
                "metadata": {
                    "type": ["object", "null"],
                    "properties": {"cluster_id": {"type": ["integer", "null"]}}
                },
                "interaction": {
                    "anyOf": [
                        {"type": "object", "properties": {
                            "kind": {"type": "string", "enum": ["viewed"]},
                            "dwell_ms": {"type": "integer"}}},
                        {"type": "object", "properties": {
                            "kind": {"type": "string", "enum": ["explicit_thumb"]},
                            "up": {"type": "boolean"}}}
                    ]
                }
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
    fn coerce_decodes_union_objects_and_nested_fields() {
        // `interaction` is advertised as `anyOf` of object branches: the
        // common type is object, so a string-encoded object decodes, and
        // the branch properties guide nested decoding.
        let mut args = json!({
            "interaction": "{\"kind\":\"explicit_thumb\",\"up\":\"true\"}",
            "metadata": {"cluster_id": "3"},
            "counts": ["1", 2.0, 3]
        });
        let mut coerced = coerce_args(&feedback_like_schema(), &mut args);
        coerced.sort();
        assert_eq!(
            coerced,
            vec![
                "counts[0]",
                "counts[1]",
                "interaction",
                "interaction.up",
                "metadata.cluster_id"
            ]
        );
        assert_eq!(
            args["interaction"],
            json!({"kind": "explicit_thumb", "up": true})
        );
        assert_eq!(args["metadata"], json!({"cluster_id": 3}));
        assert_eq!(args["counts"], json!([1, 2, 3]));
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
    fn coerce_rejects_string_floats_that_only_round_to_integers() {
        // "1.0000000000000001" is not an integer even though it rounds to
        // 1.0 in f64; the lexical form decides for strings.
        let mut args = json!({"limit": "1.0000000000000001"});
        assert!(coerce_args(&feedback_like_schema(), &mut args).is_empty());
        assert_eq!(args["limit"], json!("1.0000000000000001"));

        let mut args = json!({"limit": "7.000"});
        assert_eq!(
            coerce_args(&feedback_like_schema(), &mut args),
            vec!["limit"]
        );
        assert_eq!(args["limit"], json!(7));

        let mut args = json!({"limit": "1e2"});
        assert_eq!(
            coerce_args(&feedback_like_schema(), &mut args),
            vec!["limit"]
        );
        assert_eq!(args["limit"], json!(100));

        let mut args = json!({"limit": "1.5"});
        assert!(coerce_args(&feedback_like_schema(), &mut args).is_empty());
    }

    #[test]
    fn coerce_leaves_mismatched_and_undeclared_values_alone() {
        let mut args = json!({
            "memory_ids": "not json at all",
            "helpful": "yes",
            "unknown": "[1,2]",
            "query": 42,
            "metadata": {"unknown_nested": "[1]"}
        });
        let before = args.clone();
        let coerced = coerce_args(&feedback_like_schema(), &mut args);
        assert!(coerced.is_empty(), "{coerced:?}");
        assert_eq!(args, before);
    }

    #[test]
    fn coerce_is_noop_for_well_typed_payload() {
        let mut args = json!({
            "memory_ids": ["a"],
            "helpful": false,
            "limit": 3,
            "metadata": {"cluster_id": 1},
            "interaction": {"kind": "viewed", "dwell_ms": 10},
            "query": null
        });
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
