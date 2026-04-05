//! REST API layer for the rein web GUI.
//! Routes `/api/*` requests to store/ops functions, returning JSON.
//! Also serves the embedded SPA when the `gui` feature is enabled.

use bytes::Bytes;
use http_body_util::combinators::BoxBody;
use http_body_util::{BodyExt, Full};
use hyper::{Method, Request, Response, StatusCode};
use serde_json::json;

use crate::config::ReinConfig;
use crate::types::MemoryStore; // for store.delete()

type BoxedResponse = Response<BoxBody<Bytes, std::convert::Infallible>>;

fn json_response(status: StatusCode, body: serde_json::Value) -> BoxedResponse {
    let json_bytes = serde_json::to_vec(&body).unwrap_or_default();
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(
            Full::new(Bytes::from(json_bytes))
                .map_err(|never: std::convert::Infallible| match never {})
                .boxed(),
        )
        .unwrap_or_else(|_| {
            Response::new(
                Full::new(Bytes::from(r#"{"error":"internal"}"#))
                    .map_err(|never: std::convert::Infallible| match never {})
                    .boxed(),
            )
        })
}

fn error_response(status: StatusCode, msg: &str) -> BoxedResponse {
    json_response(status, json!({ "error": msg }))
}

fn parse_bounded_usize(
    query: &std::collections::HashMap<String, String>,
    key: &str,
    default: usize,
    min: usize,
    max: usize,
) -> Result<usize, String> {
    match query.get(key) {
        Some(raw) => {
            let value = raw
                .parse::<usize>()
                .map_err(|_| format!("invalid '{key}' parameter"))?;
            Ok(value.clamp(min, max))
        }
        None => Ok(default.clamp(min, max)),
    }
}

fn parse_bounded_i64(
    query: &std::collections::HashMap<String, String>,
    key: &str,
    default: i64,
    min: i64,
    max: i64,
) -> Result<i64, String> {
    match query.get(key) {
        Some(raw) => {
            let value = raw
                .parse::<i64>()
                .map_err(|_| format!("invalid '{key}' parameter"))?;
            Ok(value.clamp(min, max))
        }
        None => Ok(default.clamp(min, max)),
    }
}

/// Parse query string into key-value pairs with percent-decoding.
fn parse_query(uri: &hyper::Uri) -> std::collections::HashMap<String, String> {
    uri.query()
        .map(|q| {
            q.split('&')
                .filter_map(|pair| {
                    let mut parts = pair.splitn(2, '=');
                    let key = parts.next()?;
                    let value = parts.next().unwrap_or("");
                    Some((percent_decode(key), percent_decode(value)))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Simple percent-decoding: handles %XX and + (space).
fn percent_decode(s: &str) -> String {
    let s = s.replace('+', " ");
    let mut out = Vec::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(byte) =
                u8::from_str_radix(std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or(""), 16)
            {
                out.push(byte);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8(out).unwrap_or_else(|_| s.to_string())
}

/// Try to handle an API or GUI request. Returns `Some(response)` if matched, `None` to fall through to MCP.
pub async fn handle_rest_request<B>(
    req: &Request<B>,
    config: &ReinConfig,
) -> Option<BoxedResponse> {
    let path = req.uri().path();
    let method = req.method();

    // Only intercept /api/* and GUI asset paths
    if path.starts_with("/api/") {
        Some(handle_api(method, path, req.uri(), config).await)
    } else if config.server.gui_enabled && !path.starts_with("/mcp") {
        Some(serve_gui(path))
    } else {
        None
    }
}

async fn handle_api(
    method: &Method,
    path: &str,
    uri: &hyper::Uri,
    config: &ReinConfig,
) -> BoxedResponse {
    let query = parse_query(uri);

    match (method, path) {
        // --- Read endpoints ---
        (&Method::GET, "/api/stats") => api_stats(config),
        (&Method::GET, "/api/activity") => api_activity(config, &query),
        (&Method::GET, "/api/topics") => api_topics(config),
        (&Method::GET, "/api/recent") => api_recent(config, &query),
        (&Method::GET, "/api/adaptive") => api_adaptive(config),
        (&Method::GET, "/api/health") => api_health(config, &query),
        (&Method::GET, p)
            if p.starts_with("/api/memories") && !p.contains('/') || p == "/api/memories" =>
        {
            api_recall(config, &query)
        }
        (&Method::GET, p) if p.starts_with("/api/memories/") => {
            let id = &p["/api/memories/".len()..];
            api_get_memory(config, id)
        }
        (&Method::GET, "/api/memoirs") => api_memoirs(config),
        (&Method::GET, p) if p.starts_with("/api/memoirs/") => {
            handle_memoir_path(config, p, &query)
        }
        (&Method::GET, "/api/timeline") => api_timeline(config, &query),
        (&Method::GET, "/api/episodes") => api_episodes(config, &query),
        (&Method::GET, "/api/artifacts") => api_artifacts(config, &query),
        (&Method::GET, p) if p.starts_with("/api/artifacts/") => {
            let id = &p["/api/artifacts/".len()..];
            api_artifact_detail(config, id, &query)
        }

        // --- Mutation endpoints (placeholder for Phase 2) ---
        (&Method::DELETE, p) if p.starts_with("/api/memories/") => {
            let id = &p["/api/memories/".len()..];
            api_forget(config, id)
        }

        _ => error_response(StatusCode::NOT_FOUND, "unknown API endpoint"),
    }
}

fn handle_memoir_path(
    config: &ReinConfig,
    path: &str,
    query: &std::collections::HashMap<String, String>,
) -> BoxedResponse {
    let rest = &path["/api/memoirs/".len()..];
    if let Some(slash) = rest.find('/') {
        let name = &percent_decode(&rest[..slash]);
        let sub = &rest[slash + 1..];
        if sub == "export" {
            let format = query.get("format").map(|s| s.as_str()).unwrap_or("json");
            return api_memoir_export(config, name, format);
        }
        if sub.starts_with("inspect/") {
            let concept = percent_decode(&sub["inspect/".len()..]);
            let depth = match parse_bounded_usize(query, "depth", 1, 1, 8) {
                Ok(depth) => depth,
                Err(msg) => return error_response(StatusCode::BAD_REQUEST, &msg),
            };
            return api_memoir_inspect(config, name, &concept, depth);
        }
        api_memoir_show(config, name)
    } else {
        let decoded = percent_decode(rest);
        api_memoir_show(config, &decoded)
    }
}

// ===========================================================================
// API handlers
// ===========================================================================

fn api_stats(config: &ReinConfig) -> BoxedResponse {
    let store = match config.open_store() {
        Ok(s) => s,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    };
    match store.stats() {
        Ok(stats) => json_response(
            StatusCode::OK,
            json!({
                "total_memories": stats.total_memories,
                "ltm_count": stats.ltm_count,
                "stm_count": stats.stm_count,
                "topic_count": stats.topic_count,
                "avg_strength": stats.avg_strength,
                "memoir_count": stats.memoir_count,
                "concept_count": stats.concept_count,
                "link_count": stats.link_count,
            }),
        ),
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

fn api_activity(
    config: &ReinConfig,
    query: &std::collections::HashMap<String, String>,
) -> BoxedResponse {
    let days = match parse_bounded_i64(query, "days", 14, 1, 90) {
        Ok(days) => days,
        Err(msg) => return error_response(StatusCode::BAD_REQUEST, &msg),
    };
    let granularity = match query.get("granularity").map(|s| s.as_str()) {
        Some("hour") => "hour",
        Some("day") | None => "day",
        Some(_) => {
            return error_response(StatusCode::BAD_REQUEST, "invalid 'granularity' parameter")
        }
    };
    let store = match config.open_store() {
        Ok(s) => s,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    };
    let conn = store.conn();
    let offset = format!("-{days} days");

    // SQL truncation expression: hour → "YYYY-MM-DD HH:00", day → "YYYY-MM-DD"
    let (trunc_expr, group_alias) = if granularity == "hour" {
        ("strftime('%Y-%m-%d %H:00', {})", "bucket")
    } else {
        ("date({})", "bucket")
    };

    let recall_sql = format!(
        "SELECT {} as {group_alias}, COUNT(*) FROM feedback_events
         WHERE event_type = 'recall_complete' AND ts >= date('now', ?1)
         GROUP BY {group_alias} ORDER BY {group_alias}",
        trunc_expr.replace("{}", "ts")
    );
    let store_sql = format!(
        "SELECT {} as {group_alias}, COUNT(*) FROM memories
         WHERE created_at >= date('now', ?1)
         GROUP BY {group_alias} ORDER BY {group_alias}",
        trunc_expr.replace("{}", "created_at")
    );

    let mut recall_stmt = match conn.prepare(&recall_sql) {
        Ok(s) => s,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    };
    let recall_rows: Vec<(String, i64)> = recall_stmt
        .query_map(rusqlite::params![offset], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .ok()
        .map(|r| r.filter_map(|x| x.ok()).collect())
        .unwrap_or_default();

    let mut store_stmt = match conn.prepare(&store_sql) {
        Ok(s) => s,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    };
    let store_rows: Vec<(String, i64)> = store_stmt
        .query_map(rusqlite::params![offset], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .ok()
        .map(|r| r.filter_map(|x| x.ok()).collect())
        .unwrap_or_default();

    // Merge into unified series
    let mut bucket_map: std::collections::BTreeMap<String, (i64, i64)> =
        std::collections::BTreeMap::new();
    for (bucket, count) in &recall_rows {
        bucket_map.entry(bucket.clone()).or_default().0 = *count;
    }
    for (bucket, count) in &store_rows {
        bucket_map.entry(bucket.clone()).or_default().1 = *count;
    }

    let activity: Vec<serde_json::Value> = bucket_map.into_iter().map(|(date, (recalls, stores))| {
        json!({ "date": date, "recalls": recalls, "stores": stores })
    }).collect();

    json_response(
        StatusCode::OK,
        json!({ "activity": activity, "granularity": granularity }),
    )
}

fn api_topics(config: &ReinConfig) -> BoxedResponse {
    let store = match config.open_store() {
        Ok(s) => s,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    };
    match store.list_topics() {
        Ok(topics) => json_response(StatusCode::OK, json!({ "topics": topics })),
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

fn api_recent(
    config: &ReinConfig,
    query: &std::collections::HashMap<String, String>,
) -> BoxedResponse {
    let limit = match parse_bounded_usize(query, "limit", 20, 1, 100) {
        Ok(limit) => limit,
        Err(msg) => return error_response(StatusCode::BAD_REQUEST, &msg),
    };
    let store = match config.open_store() {
        Ok(s) => s,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    };
    match store.recent(limit) {
        Ok(memories) => {
            let items: Vec<serde_json::Value> = memories.iter().map(memory_to_json).collect();
            json_response(StatusCode::OK, json!({ "memories": items }))
        }
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

fn api_adaptive(config: &ReinConfig) -> BoxedResponse {
    let store = match config.open_store() {
        Ok(s) => s,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    };
    json_response(StatusCode::OK, crate::ops::adaptive_status(&store))
}

fn api_health(
    config: &ReinConfig,
    query: &std::collections::HashMap<String, String>,
) -> BoxedResponse {
    let topic = query.get("topic").map(|s| s.as_str());
    let store = match config.open_store() {
        Ok(s) => s,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    };
    match store.health(topic) {
        Ok(report) => {
            // HealthReport may not implement Serialize, convert via Debug
            json_response(StatusCode::OK, json!({ "health": format!("{:?}", report) }))
        }
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

fn api_recall(
    config: &ReinConfig,
    query: &std::collections::HashMap<String, String>,
) -> BoxedResponse {
    let q = match query.get("q") {
        Some(q) if !q.is_empty() => q.clone(),
        _ => return error_response(StatusCode::BAD_REQUEST, "missing 'q' query parameter"),
    };
    let topic = query.get("topic").map(|s| s.as_str());
    let keyword = query.get("keyword").map(|s| s.as_str());
    let limit = match parse_bounded_usize(query, "limit", 20, 1, 100) {
        Ok(limit) => limit,
        Err(msg) => return error_response(StatusCode::BAD_REQUEST, &msg),
    };
    let from = query.get("from").and_then(|s| parse_datetime(s));
    let to = query.get("to").and_then(|s| parse_datetime_end(s));

    let store = match config.open_store() {
        Ok(s) => s,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    };

    match crate::search::recall::recall_temporal(
        &store, config, &q, topic, keyword, limit, from, to, None, false,
    ) {
        Ok(results) => {
            let items: Vec<serde_json::Value> = results
                .iter()
                .map(|r| {
                    let mut m = memory_to_json(&r.memory);
                    if let Some(obj) = m.as_object_mut() {
                        obj.insert("score".to_string(), json!(r.score));
                        obj.insert("confidence".to_string(), json!(r.confidence));
                        obj.insert("sources_hit".to_string(), json!(r.sources_hit));
                    }
                    m
                })
                .collect();
            json_response(
                StatusCode::OK,
                json!({ "results": items, "count": items.len() }),
            )
        }
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

fn api_get_memory(config: &ReinConfig, id: &str) -> BoxedResponse {
    let store = match config.open_store() {
        Ok(s) => s,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    };
    match store.get(id) {
        Ok(m) => json_response(StatusCode::OK, memory_to_json(&m)),
        Err(e) => error_response(StatusCode::NOT_FOUND, &format!("memory not found: {e}")),
    }
}

fn api_forget(config: &ReinConfig, id: &str) -> BoxedResponse {
    let store = match config.open_store() {
        Ok(s) => s,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    };
    match store.delete(id) {
        Ok(_) => json_response(StatusCode::OK, json!({ "deleted": id })),
        Err(e) => error_response(StatusCode::NOT_FOUND, &e.to_string()),
    }
}

fn api_memoirs(config: &ReinConfig) -> BoxedResponse {
    let store = match config.open_store() {
        Ok(s) => s,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    };
    match store.list_memoirs() {
        Ok(memoirs) => json_response(StatusCode::OK, json!({ "memoirs": memoirs })),
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

fn api_memoir_show(config: &ReinConfig, name: &str) -> BoxedResponse {
    let store = match config.open_store() {
        Ok(s) => s,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    };
    // Use JSON export which includes memoir + concepts + links
    match store.export_memoir(name, "json") {
        Ok(output) => match serde_json::from_str::<serde_json::Value>(&output) {
            Ok(v) => json_response(StatusCode::OK, v),
            Err(_) => json_response(StatusCode::OK, json!({ "raw": output })),
        },
        Err(e) => {
            let msg = e.to_string();
            if msg.contains("not found") {
                error_response(StatusCode::NOT_FOUND, &msg)
            } else {
                error_response(StatusCode::INTERNAL_SERVER_ERROR, &msg)
            }
        }
    }
}

fn api_memoir_export(config: &ReinConfig, name: &str, format: &str) -> BoxedResponse {
    let store = match config.open_store() {
        Ok(s) => s,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    };
    match store.export_memoir(name, format) {
        Ok(output) => {
            if format == "json" {
                // Parse JSON string into Value so we return proper JSON
                match serde_json::from_str::<serde_json::Value>(&output) {
                    Ok(v) => json_response(StatusCode::OK, v),
                    Err(_) => json_response(StatusCode::OK, json!({ "raw": output })),
                }
            } else {
                // DOT/ASCII: return as text
                let body = Full::new(Bytes::from(output))
                    .map_err(|never: std::convert::Infallible| match never {})
                    .boxed();
                Response::builder()
                    .status(200)
                    .header("content-type", "text/plain")
                    .body(body)
                    .unwrap_or_else(|_| {
                        Response::new(
                            Full::new(Bytes::new())
                                .map_err(|never: std::convert::Infallible| match never {})
                                .boxed(),
                        )
                    })
            }
        }
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

fn api_memoir_inspect(
    config: &ReinConfig,
    memoir: &str,
    concept: &str,
    depth: usize,
) -> BoxedResponse {
    let store = match config.open_store() {
        Ok(s) => s,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    };
    match store.inspect_concept(memoir, concept, depth) {
        Ok((center, neighbors, links)) => json_response(
            StatusCode::OK,
            json!({
                "center": center,
                "neighbors": neighbors,
                "links": links,
            }),
        ),
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

fn api_timeline(
    config: &ReinConfig,
    query: &std::collections::HashMap<String, String>,
) -> BoxedResponse {
    let limit = match parse_bounded_usize(query, "limit", 50, 1, 200) {
        Ok(limit) => limit,
        Err(msg) => return error_response(StatusCode::BAD_REQUEST, &msg),
    };
    let from = query.get("from").and_then(|s| parse_datetime(s));
    let to = query.get("to").and_then(|s| parse_datetime_end(s));

    let store = match config.open_store() {
        Ok(s) => s,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    };

    // Collect episodes in window
    let episodes = match (from, to) {
        (Some(f), Some(t)) => store.get_episodes_in_range(f, t).unwrap_or_default(),
        (Some(f), None) => store
            .get_episodes_in_range(f, chrono::Utc::now() + chrono::Duration::days(1))
            .unwrap_or_default(),
        (None, Some(t)) => store
            .get_episodes_in_range(
                chrono::DateTime::parse_from_rfc3339("2000-01-01T00:00:00Z")
                    .unwrap()
                    .with_timezone(&chrono::Utc),
                t,
            )
            .unwrap_or_default(),
        (None, None) => store.list_episodes(limit).unwrap_or_default(),
    };

    // Collect memories in the same window (respect from/to filters)
    let memories = if from.is_some() || to.is_some() {
        let from_bound = from.unwrap_or_else(|| {
            chrono::DateTime::parse_from_rfc3339("2000-01-01T00:00:00Z")
                .unwrap()
                .with_timezone(&chrono::Utc)
        });
        let to_bound = to.unwrap_or_else(|| chrono::Utc::now() + chrono::Duration::days(1));
        let mut stmt = match store.conn().prepare(
            "SELECT * FROM memories WHERE created_at >= ?1 AND created_at <= ?2 ORDER BY created_at DESC LIMIT ?3",
        ) {
            Ok(stmt) => stmt,
            Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
        };
        let rows = match stmt.query_map(
            rusqlite::params![from_bound.to_rfc3339(), to_bound.to_rfc3339(), limit],
            |row| {
                crate::store::sqlite::row_to_memory(row).map_err(|e| {
                    rusqlite::Error::FromSqlConversionFailure(
                        0,
                        rusqlite::types::Type::Text,
                        Box::new(e),
                    )
                })
            },
        ) {
            Ok(rows) => rows,
            Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
        };
        rows.filter_map(|r| r.ok()).collect()
    } else {
        store.recent(limit).unwrap_or_default()
    };

    // Flatten into unified events array sorted by date descending
    let mut events: Vec<serde_json::Value> = Vec::new();
    for ep in &episodes {
        events.push(json!({
            "type": "episode",
            "id": ep.id,
            "title": ep.title,
            "outcome": ep.outcome,
            "decisions": ep.decisions,
            "created_at": ep.created_at.to_rfc3339(),
        }));
    }
    for m in &memories {
        let mut ev = memory_to_json(m);
        if let Some(obj) = ev.as_object_mut() {
            obj.insert("type".to_string(), json!("memory"));
        }
        events.push(ev);
    }
    // Sort by created_at descending
    events.sort_by(|a, b| {
        let da = a.get("created_at").and_then(|v| v.as_str()).unwrap_or("");
        let db = b.get("created_at").and_then(|v| v.as_str()).unwrap_or("");
        db.cmp(da)
    });
    events.truncate(limit);

    json_response(StatusCode::OK, json!({ "events": events }))
}

fn api_episodes(
    config: &ReinConfig,
    query: &std::collections::HashMap<String, String>,
) -> BoxedResponse {
    let limit = match parse_bounded_usize(query, "limit", 20, 1, 100) {
        Ok(limit) => limit,
        Err(msg) => return error_response(StatusCode::BAD_REQUEST, &msg),
    };
    let store = match config.open_store() {
        Ok(s) => s,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    };
    match store.list_episodes(limit) {
        Ok(episodes) => json_response(StatusCode::OK, json!({ "episodes": episodes })),
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

fn api_artifacts(
    config: &ReinConfig,
    query: &std::collections::HashMap<String, String>,
) -> BoxedResponse {
    let limit = match parse_bounded_usize(query, "limit", 20, 1, 100) {
        Ok(limit) => limit,
        Err(msg) => return error_response(StatusCode::BAD_REQUEST, &msg),
    };
    let offset = match parse_bounded_usize(query, "offset", 0, 0, 10000) {
        Ok(offset) => offset,
        Err(msg) => return error_response(StatusCode::BAD_REQUEST, &msg),
    };
    let store = match config.open_store() {
        Ok(s) => s,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    };

    // Query session_artifacts table directly
    let sql = "SELECT id, artifact_kind, title, summary, source_agent, source_label, \
               turn_count, episode_id, created_at FROM session_artifacts \
               ORDER BY created_at DESC LIMIT ?1 OFFSET ?2";
    let result = store.conn().prepare(sql).and_then(|mut stmt| {
        let rows = stmt.query_map(rusqlite::params![limit, offset], |row| {
            Ok(json!({
                "id": row.get::<_, String>(0)?,
                "artifact_kind": row.get::<_, String>(1)?,
                "title": row.get::<_, Option<String>>(2)?,
                "summary": row.get::<_, Option<String>>(3)?,
                "source_agent": row.get::<_, Option<String>>(4)?,
                "source_label": row.get::<_, Option<String>>(5)?,
                "turn_count": row.get::<_, u32>(6)?,
                "episode_id": row.get::<_, Option<String>>(7)?,
                "created_at": row.get::<_, String>(8)?,
            }))
        })?;
        Ok(rows.filter_map(|r| r.ok()).collect::<Vec<_>>())
    });

    match result {
        Ok(artifacts) => json_response(StatusCode::OK, json!({ "artifacts": artifacts })),
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

fn api_artifact_detail(
    config: &ReinConfig,
    id: &str,
    query: &std::collections::HashMap<String, String>,
) -> BoxedResponse {
    let store = match config.open_store() {
        Ok(s) => s,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    };
    let include_transcript = matches!(
        query.get("include_transcript").map(|v| v.as_str()),
        Some("true")
    );
    let sql = "SELECT id, schema_version, artifact_kind, session_id, title, summary, \
               source_agent, source_label, is_subagent, started_at, ended_at, \
               turn_count, transcript_text, transcript_json, episode_id, created_at \
               FROM session_artifacts WHERE id = ?1";
    let result = store.conn().query_row(sql, rusqlite::params![id], |row| {
        let transcript_text: String = row.get(12)?;
        Ok(json!({
            "id": row.get::<_, String>(0)?,
            "schema_version": row.get::<_, i32>(1)?,
            "artifact_kind": row.get::<_, String>(2)?,
            "session_id": row.get::<_, Option<String>>(3)?,
            "title": row.get::<_, Option<String>>(4)?,
            "summary": row.get::<_, Option<String>>(5)?,
            "source_agent": row.get::<_, Option<String>>(6)?,
            "source_label": row.get::<_, Option<String>>(7)?,
            "is_subagent": row.get::<_, bool>(8)?,
            "started_at": row.get::<_, Option<String>>(9)?,
            "ended_at": row.get::<_, Option<String>>(10)?,
            "turn_count": row.get::<_, u32>(11)?,
            "transcript_text": if include_transcript {
                Some(crate::extract::hooks::parsing::redact_secrets(&transcript_text))
            } else {
                None::<String>
            },
            "transcript_available": !transcript_text.trim().is_empty(),
            "transcript_json": row.get::<_, Option<String>>(13)?
                .map(|j| crate::extract::hooks::parsing::redact_secrets(&j)),
            "episode_id": row.get::<_, Option<String>>(14)?,
            "created_at": row.get::<_, String>(15)?,
        }))
    });

    match result {
        Ok(artifact) => json_response(StatusCode::OK, artifact),
        Err(_) => error_response(StatusCode::NOT_FOUND, "artifact not found"),
    }
}

// ===========================================================================
// Helpers
// ===========================================================================

fn memory_to_json(m: &crate::types::Memory) -> serde_json::Value {
    json!({
        "id": m.id,
        "layer": format!("{}", m.layer),
        "topic": m.topic,
        "summary": m.summary,
        "content": m.content,
        "keywords": m.keywords,
        "importance": format!("{}", m.importance),
        "source": format!("{}", m.source),
        "strength": m.strength,
        "tier": format!("{}", m.tier),
        "cluster_id": m.cluster_id,
        "access_count": m.access_count,
        "status": format!("{}", m.status),
        "related_ids": m.related_ids,
        "concept_ids": m.concept_ids,
        "superseded_by": m.superseded_by,
        "created_at": m.created_at.to_rfc3339(),
        "updated_at": m.updated_at.to_rfc3339(),
        "last_accessed": m.last_accessed.to_rfc3339(),
    })
}

fn parse_datetime(s: &str) -> Option<chrono::DateTime<chrono::Utc>> {
    chrono::DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|dt| dt.with_timezone(&chrono::Utc))
        .or_else(|| {
            chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d")
                .ok()
                .and_then(|d| d.and_hms_opt(0, 0, 0))
                .map(|dt| dt.and_utc())
        })
}

fn parse_datetime_end(s: &str) -> Option<chrono::DateTime<chrono::Utc>> {
    chrono::DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|dt| dt.with_timezone(&chrono::Utc))
        .or_else(|| {
            chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d")
                .ok()
                .and_then(|d| d.and_hms_opt(23, 59, 59))
                .map(|dt| dt.and_utc())
        })
}

// ===========================================================================
// GUI static file serving
// ===========================================================================

#[cfg(feature = "gui")]
#[derive(rust_embed::Embed)]
#[folder = "gui/dist"]
struct GuiAssets;

fn serve_gui(#[allow(unused)] path: &str) -> BoxedResponse {
    #[cfg(feature = "gui")]
    {
        let file_path = if path == "/" {
            "index.html"
        } else {
            &path[1..]
        };
        if let Some(content) = GuiAssets::get(file_path) {
            let mime = mime_from_path(file_path);
            return Response::builder()
                .status(200)
                .header("content-type", mime)
                .body(
                    Full::new(Bytes::from(content.data.to_vec()))
                        .map_err(|never: std::convert::Infallible| match never {})
                        .boxed(),
                )
                .unwrap_or_else(|_| {
                    json_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        json!({"error": "build response failed"}),
                    )
                });
        }
        // SPA fallback: serve index.html for client-side routing
        if let Some(index) = GuiAssets::get("index.html") {
            return Response::builder()
                .status(200)
                .header("content-type", "text/html")
                .body(
                    Full::new(Bytes::from(index.data.to_vec()))
                        .map_err(|never: std::convert::Infallible| match never {})
                        .boxed(),
                )
                .unwrap_or_else(|_| {
                    json_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        json!({"error": "build response failed"}),
                    )
                });
        }
    }
    error_response(
        StatusCode::NOT_FOUND,
        "GUI not available (build with --features gui)",
    )
}

#[cfg(feature = "gui")]
fn mime_from_path(path: &str) -> &'static str {
    if path.ends_with(".html") {
        "text/html"
    } else if path.ends_with(".js") {
        "application/javascript"
    } else if path.ends_with(".css") {
        "text/css"
    } else if path.ends_with(".json") {
        "application/json"
    } else if path.ends_with(".svg") {
        "image/svg+xml"
    } else if path.ends_with(".png") {
        "image/png"
    } else if path.ends_with(".ico") {
        "image/x-icon"
    } else if path.ends_with(".woff2") {
        "font/woff2"
    } else if path.ends_with(".woff") {
        "font/woff"
    } else {
        "application/octet-stream"
    }
}
