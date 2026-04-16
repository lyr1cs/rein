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
const HTTP_SESSION_COOKIE: &str = "rein_http_token";

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

fn session_cookie_value(token: &str) -> String {
    format!("{HTTP_SESSION_COOKIE}={token}; HttpOnly; SameSite=Strict; Path=/api/")
}

fn clear_session_cookie_value() -> String {
    format!("{HTTP_SESSION_COOKIE}=; HttpOnly; SameSite=Strict; Path=/api/; Max-Age=0")
}

fn json_response_with_cookie(
    status: StatusCode,
    body: serde_json::Value,
    cookie: &str,
) -> BoxedResponse {
    let json_bytes = serde_json::to_vec(&body).unwrap_or_default();
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .header("set-cookie", cookie)
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
        Some(handle_api(req, method, path, req.uri(), config).await)
    } else if config.server.gui_enabled && !path.starts_with("/mcp") {
        Some(serve_gui(path))
    } else {
        None
    }
}

#[allow(clippy::result_large_err)] // BoxedResponse is already a boxed body; boxing again would force every caller to dereference.
fn require_mutation_marker<B>(req: &Request<B>) -> Result<(), BoxedResponse> {
    let marker = req
        .headers()
        .get("x-rein-action")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("");
    if marker == "1" {
        Ok(())
    } else {
        Err(error_response(
            StatusCode::FORBIDDEN,
            "mutation requests require x-rein-action: 1",
        ))
    }
}

/// Require an `x-rein-token` header matching `$REIN_HTTP_TOKEN` for sensitive reads.
///
/// Used for endpoints that return raw upstream transcripts (e.g. `/api/artifacts`).
/// When `REIN_HTTP_TOKEN` is unset, the gate is permissive — this preserves the
/// localhost-only dev convenience. When it IS set, the token must match exactly.
#[allow(clippy::result_large_err)] // BoxedResponse is already a boxed body.
fn require_read_token<B>(req: &Request<B>) -> Result<(), BoxedResponse> {
    let expected = std::env::var("REIN_HTTP_TOKEN")
        .ok()
        .filter(|t| !t.trim().is_empty());
    let Some(expected) = expected else {
        return Ok(());
    };
    let presented = req
        .headers()
        .get("x-rein-token")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if presented == expected.trim() {
        Ok(())
    } else {
        Err(error_response(
            StatusCode::UNAUTHORIZED,
            "missing or invalid x-rein-token for protected read",
        ))
    }
}

async fn handle_api<B>(
    req: &Request<B>,
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
        (&Method::GET, "/api/dedup_decisions") => api_dedup_decisions(config, &query),
        (&Method::GET, "/api/intelligent_merge_metrics") => api_intelligent_merge_metrics(),
        (&Method::GET, "/api/health") => api_health(config, &query),
        (&Method::GET, "/api/doctor") => {
            if query
                .get("fix")
                .map(|v| matches!(v.as_str(), "1" | "true" | "yes"))
                .unwrap_or(false)
            {
                error_response(
                    StatusCode::METHOD_NOT_ALLOWED,
                    "doctor fixes require POST /api/doctor?fix=true",
                )
            } else {
                api_doctor(config, &query)
            }
        }
        (&Method::POST, "/api/doctor") => match require_mutation_marker(req) {
            Ok(()) => api_doctor(config, &query),
            Err(response) => response,
        },
        (&Method::POST, "/api/session") => match require_mutation_marker(req) {
            Ok(()) => api_create_session(),
            Err(response) => response,
        },
        (&Method::DELETE, "/api/session") => match require_mutation_marker(req) {
            Ok(()) => api_clear_session(),
            Err(response) => response,
        },
        (&Method::GET, "/api/recall_stream") => api_recall_stream(config, &query),
        (&Method::GET, p)
            if p.starts_with("/api/memories") && !p.contains('/') || p == "/api/memories" =>
        {
            api_recall(config, &query)
        }
        (&Method::GET, p) if p.starts_with("/api/memories/") => {
            let id = percent_decode(&p["/api/memories/".len()..]);
            api_get_memory(config, &id)
        }
        (&Method::GET, "/api/memoirs") => api_memoirs(config),
        (&Method::GET, p) if p.starts_with("/api/memoirs/") => {
            handle_memoir_path(config, p, &query)
        }
        (&Method::GET, "/api/timeline") => api_timeline(config, &query),
        (&Method::GET, "/api/episodes") => api_episodes(config, &query),
        (&Method::GET, "/api/artifacts") => match require_read_token(req) {
            Ok(()) => api_artifacts(config, &query),
            Err(response) => response,
        },
        (&Method::GET, p) if p.starts_with("/api/artifacts/") => {
            match require_read_token(req) {
                Ok(()) => {
                    let id = percent_decode(&p["/api/artifacts/".len()..]);
                    api_artifact_detail(config, &id, &query)
                }
                Err(response) => response,
            }
        }

        // --- Mutation endpoints (placeholder for Phase 2) ---
        (&Method::DELETE, p) if p.starts_with("/api/memories/") => {
            match require_mutation_marker(req) {
                Ok(()) => {
                    let id = percent_decode(&p["/api/memories/".len()..]);
                    api_forget(config, &id)
                }
                Err(response) => response,
            }
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
        error_response(StatusCode::NOT_FOUND, "unknown memoir API endpoint")
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
                "hot_count": stats.hot_count,
                "warm_count": stats.warm_count,
                "cold_count": stats.cold_count,
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

/// Return recent dedup_decisions rows so the GUI / MCP clients can explain
/// why a canonical exists in its current shape. Supports `?limit=N` and
/// `?operator=llm_verdict` to filter intelligent_merge decisions.
fn api_dedup_decisions(
    config: &ReinConfig,
    query: &std::collections::HashMap<String, String>,
) -> BoxedResponse {
    let store = match config.open_store() {
        Ok(s) => s,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    };
    let limit: i64 = query
        .get("limit")
        .and_then(|s| s.parse().ok())
        .unwrap_or(50)
        .clamp(1, 500);
    let operator_filter = query.get("operator").cloned();

    let rows_result: rusqlite::Result<Vec<serde_json::Value>> = (|| {
        let mut stmt = store.conn().prepare(
            "SELECT id, winner_id, loser_id, canonical_id, lexical_score, embedding_score,
                    relation, confidence, reason, operator, reversible, merged_summary,
                    novel_facts, conflict_detected, created_at
             FROM dedup_decisions
             WHERE (?1 IS NULL OR operator = ?1)
             ORDER BY created_at DESC
             LIMIT ?2",
        )?;
        let rows = stmt.query_map(rusqlite::params![operator_filter, limit], |r| {
            Ok(json!({
                "id": r.get::<_, String>(0)?,
                "winner_id": r.get::<_, Option<String>>(1)?,
                "loser_id": r.get::<_, Option<String>>(2)?,
                "canonical_id": r.get::<_, Option<String>>(3)?,
                "lexical_score": r.get::<_, Option<f64>>(4)?,
                "embedding_score": r.get::<_, Option<f64>>(5)?,
                "relation": r.get::<_, String>(6)?,
                "confidence": r.get::<_, f64>(7)?,
                "reason": r.get::<_, String>(8)?,
                "operator": r.get::<_, String>(9)?,
                "reversible": r.get::<_, i64>(10)? != 0,
                "merged_summary": r.get::<_, Option<String>>(11)?,
                "novel_facts": r.get::<_, String>(12)?,
                "conflict_detected": r.get::<_, i64>(13)? != 0,
                "created_at": r.get::<_, String>(14)?,
            }))
        })?;
        rows.collect()
    })();

    match rows_result {
        Ok(items) => json_response(StatusCode::OK, json!({ "decisions": items })),
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

/// Return process-wide intelligent_merge classifier counters for monitoring.
fn api_intelligent_merge_metrics() -> BoxedResponse {
    let (attempted, success, parse_err, http_err, stale_race) =
        crate::extract::intelligent_merge::metrics_snapshot();
    json_response(
        StatusCode::OK,
        json!({
            "attempted": attempted,
            "success": success,
            "parse_errors": parse_err,
            "http_errors": http_err,
            "stale_races": stale_race,
        }),
    )
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
        Ok(reports) => {
            let items: Vec<serde_json::Value> = reports
                .iter()
                .map(|r| {
                    json!({
                        "topic": r.topic,
                        "count": r.count,
                        "avg_strength": r.avg_strength,
                        "stale_count": r.stale_count,
                        "needs_consolidation": r.needs_consolidation,
                    })
                })
                .collect();
            json_response(StatusCode::OK, json!({ "health": items }))
        }
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

fn api_doctor(
    config: &ReinConfig,
    query: &std::collections::HashMap<String, String>,
) -> BoxedResponse {
    let network = query
        .get("network")
        .map(|v| matches!(v.as_str(), "1" | "true" | "yes"))
        .unwrap_or(false);
    let fix = query
        .get("fix")
        .map(|v| matches!(v.as_str(), "1" | "true" | "yes"))
        .unwrap_or(false);

    let report = match tokio::runtime::Handle::try_current() {
        Ok(handle) => tokio::task::block_in_place(|| {
            handle.block_on(crate::doctor::run(
                config,
                crate::doctor::DoctorOptions { network, fix },
            ))
        }),
        Err(_) => {
            let rt = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(rt) => rt,
                Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
            };
            rt.block_on(crate::doctor::run(
                config,
                crate::doctor::DoctorOptions { network, fix },
            ))
        }
    };
    json_response(
        StatusCode::OK,
        serde_json::to_value(report).unwrap_or_else(|_| json!({})),
    )
}

fn api_create_session() -> BoxedResponse {
    let token = std::env::var("REIN_HTTP_TOKEN")
        .ok()
        .filter(|token| !token.trim().is_empty());
    match token {
        Some(token) => json_response_with_cookie(
            StatusCode::OK,
            json!({ "authenticated": true }),
            &session_cookie_value(&token),
        ),
        None => error_response(
            StatusCode::BAD_REQUEST,
            "REIN_HTTP_TOKEN is not configured on this server",
        ),
    }
}

fn api_clear_session() -> BoxedResponse {
    json_response_with_cookie(
        StatusCode::OK,
        json!({ "authenticated": false }),
        &clear_session_cookie_value(),
    )
}

fn api_recall(
    config: &ReinConfig,
    query: &std::collections::HashMap<String, String>,
) -> BoxedResponse {
    match run_recall_query(config, query, None) {
        Ok(results) => recall_results_response(results, 0, None),
        Err(response) => response,
    }
}

fn api_recall_stream(
    config: &ReinConfig,
    query: &std::collections::HashMap<String, String>,
) -> BoxedResponse {
    let offset = match parse_bounded_usize(query, "offset", 0, 0, 10_000) {
        Ok(offset) => offset,
        Err(msg) => return error_response(StatusCode::BAD_REQUEST, &msg),
    };
    let page_limit = match parse_bounded_usize(query, "limit", 20, 1, 100) {
        Ok(limit) => limit,
        Err(msg) => return error_response(StatusCode::BAD_REQUEST, &msg),
    };

    let fetch_limit = offset.saturating_add(page_limit).saturating_add(1);
    match run_recall_query(config, query, Some(fetch_limit)) {
        Ok(results) => recall_results_response(results, offset, Some(page_limit)),
        Err(response) => response,
    }
}

#[allow(clippy::result_large_err)] // BoxedResponse is already a boxed body.
fn run_recall_query(
    config: &ReinConfig,
    query: &std::collections::HashMap<String, String>,
    limit_override: Option<usize>,
) -> Result<Vec<crate::search::recall::RecallResult>, BoxedResponse> {
    let q = match query.get("q") {
        Some(q) if !q.is_empty() => q.clone(),
        _ => {
            return Err(error_response(
                StatusCode::BAD_REQUEST,
                "missing 'q' query parameter",
            ))
        }
    };
    let topic = query.get("topic").map(|s| s.as_str());
    let keyword = query.get("keyword").map(|s| s.as_str());
    let limit = match limit_override {
        Some(limit) => limit,
        None => match parse_bounded_usize(query, "limit", 20, 1, 100) {
            Ok(limit) => limit,
            Err(msg) => return Err(error_response(StatusCode::BAD_REQUEST, &msg)),
        },
    };
    let from = query.get("from").and_then(|s| parse_datetime(s));
    let to = query.get("to").and_then(|s| parse_datetime_end(s));

    let store = match config.open_store() {
        Ok(s) => s,
        Err(e) => {
            return Err(error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &e.to_string(),
            ))
        }
    };

    crate::search::recall::recall_temporal(
        &store, config, &q, topic, keyword, limit, from, to, None, false,
    )
    .map_err(|e| error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()))
}

fn recall_result_to_json(r: &crate::search::recall::RecallResult) -> serde_json::Value {
    let mut memory = memory_to_json(&r.memory);
    if let Some(obj) = memory.as_object_mut() {
        obj.insert("score".to_string(), json!(r.score));
        obj.insert("confidence".to_string(), json!(r.confidence));
        obj.insert("sources_hit".to_string(), json!(r.sources_hit));
        obj.insert("evidence_count".to_string(), json!(r.evidence_count));
        obj.insert("evidence_preview".to_string(), json!(r.evidence_preview));
    }
    memory
}

fn recall_results_response(
    results: Vec<crate::search::recall::RecallResult>,
    offset: usize,
    page_limit: Option<usize>,
) -> BoxedResponse {
    let page = match page_limit {
        Some(limit) => {
            let end = offset.saturating_add(limit).min(results.len());
            if offset >= results.len() {
                vec![]
            } else {
                results[offset..end]
                    .iter()
                    .map(recall_result_to_json)
                    .collect::<Vec<_>>()
            }
        }
        None => results
            .iter()
            .map(recall_result_to_json)
            .collect::<Vec<_>>(),
    };

    if page_limit.is_some() {
        let end = offset.saturating_add(page.len());
        let has_more = results.len() > end;
        let next_offset = if has_more { Some(end) } else { None };
        json_response(
            StatusCode::OK,
            json!({
                "results": page,
                "count": page.len(),
                "offset": offset,
                "limit": page_limit,
                "next_offset": next_offset,
                "has_more": has_more,
            }),
        )
    } else {
        json_response(
            StatusCode::OK,
            json!({ "results": page, "count": page.len() }),
        )
    }
}

fn api_get_memory(config: &ReinConfig, id: &str) -> BoxedResponse {
    let store = match config.open_store() {
        Ok(s) => s,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    };
    match store.get(id) {
        Ok(m) => {
            let canonical_id = m.canonical_id.clone().unwrap_or_else(|| m.id.clone());
            let mut body = memory_to_json(&m);
            let evidence = store
                .list_memory_evidence(&canonical_id, 12)
                .unwrap_or_default()
                .into_iter()
                .filter(|item| item.memory_id.as_deref() != Some(canonical_id.as_str()))
                .map(|item| {
                    json!({
                        "id": item.id,
                        "canonical_id": item.canonical_id,
                        "memory_id": item.memory_id,
                        "source_topic": item.source_topic,
                        "summary": item.summary,
                        "content": item.content,
                        "keywords": item.keywords,
                        "source": format!("{}", item.source),
                        "created_at": item.created_at.to_rfc3339(),
                        "imported_at": item.imported_at.to_rfc3339(),
                    })
                })
                .collect::<Vec<_>>();
            if let Some(obj) = body.as_object_mut() {
                obj.insert("memory".to_string(), memory_to_json(&m));
                obj.insert("evidence".to_string(), json!(evidence));
            }
            json_response(StatusCode::OK, body)
        }
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
    if !matches!(format, "json" | "ascii" | "dot") {
        return error_response(StatusCode::BAD_REQUEST, "invalid export format");
    }
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
        let memories: Vec<crate::types::Memory> = rows.filter_map(|r| r.ok()).collect();
        match store.collapse_to_canonicals(memories, limit) {
            Ok(memories) => memories,
            Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
        }
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

/// Hard cap on the size of each transcript field returned by the detail endpoint.
/// Prevents unbounded memory allocation during redaction and keeps JSON responses
/// small enough for GUI consumption. Oversized bodies are truncated at a char
/// boundary and flagged via `transcript_truncated`.
const MAX_ARTIFACT_RESPONSE_BYTES: usize = 2 * 1024 * 1024;

fn cap_for_redaction(s: &str) -> (String, bool) {
    if s.len() <= MAX_ARTIFACT_RESPONSE_BYTES {
        return (s.to_string(), false);
    }
    // Truncate at a char boundary to preserve UTF-8 validity.
    let mut end = MAX_ARTIFACT_RESPONSE_BYTES;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    (s[..end].to_string(), true)
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
        let transcript_json_raw: Option<String> = row.get(13)?;

        // Only materialize + redact transcripts when the caller asked for them.
        // Otherwise omit both fields entirely to avoid leaking raw bodies.
        let (transcript_text_out, transcript_json_out, transcript_truncated) = if include_transcript
        {
            let (capped_text, text_truncated) = cap_for_redaction(&transcript_text);
            let redacted_text = crate::extract::hooks::parsing::redact_secrets(&capped_text);
            let (redacted_json, json_truncated) = match transcript_json_raw {
                Some(raw) => {
                    let (capped, t) = cap_for_redaction(&raw);
                    (
                        Some(crate::extract::hooks::parsing::redact_secrets(&capped)),
                        t,
                    )
                }
                None => (None, false),
            };
            (
                Some(redacted_text),
                redacted_json,
                text_truncated || json_truncated,
            )
        } else {
            (None, None, false)
        };

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
            "transcript_text": transcript_text_out,
            "transcript_available": !transcript_text.trim().is_empty(),
            "transcript_json": transcript_json_out,
            "transcript_truncated": transcript_truncated,
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
    let summary_short: String = m.summary.chars().take(110).collect();
    json!({
        "id": m.id,
        "layer": format!("{}", m.layer),
        "topic": m.topic,
        "summary": m.summary,
        "summary_short": summary_short,
        "content": m.content,
        "keywords": m.keywords,
        "importance": format!("{}", m.importance),
        "source": format!("{}", m.source),
        "strength": m.strength,
        "tier": format!("{}", m.tier),
        "cluster_id": m.cluster_id,
        "access_count": m.access_count,
        "canonical_id": m.canonical_id,
        "support_count": m.support_count,
        "merge_count": m.merge_count,
        "dedup_confidence": m.dedup_confidence,
        "source_diversity": m.source_diversity,
        "contradiction_score": m.contradiction_score,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::SqliteStore;
    use crate::types::{
        Importance, Memory, MemoryLayer, MemoryStatus, MemoryTier, SessionArtifact, Source,
    };
    use chrono::Utc;
    use hyper::Request;
    use tempfile::tempdir;

    fn test_memory(id: &str, summary: &str, content: &str) -> Memory {
        Memory {
            id: id.to_string(),
            layer: MemoryLayer::LTM,
            topic: "streaming".to_string(),
            summary: summary.to_string(),
            content: content.to_string(),
            keywords: vec!["recall".to_string()],
            importance: Importance::High,
            source: Source::Manual,
            strength: 1.0,
            decay_lambda: 0.06,
            access_count: 0,
            superseded_by: None,
            canonical_id: None,
            support_count: 1,
            merge_count: 0,
            dedup_confidence: 1.0,
            source_diversity: 1.0,
            contradiction_score: 0.0,
            related_ids: vec![],
            concept_ids: vec![],
            status: MemoryStatus::Active,
            embedding: None,
            tier: MemoryTier::Warm,
            cluster_id: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_accessed: Utc::now(),
        }
    }

    fn test_config(db_path: &std::path::Path) -> ReinConfig {
        let mut config = ReinConfig::default();
        config.database.path = db_path.to_string_lossy().to_string();
        config.embedding.provider = "none".to_string();
        config.query_expansion.provider = "none".to_string();
        config.sync.supermemory_enabled = false;
        config.sync.auto_memory_enabled = false;
        config
    }

    async fn read_json(response: BoxedResponse) -> serde_json::Value {
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn recall_stream_pages_results_without_changing_legacy_endpoint() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("memories.db");
        let config = test_config(&db_path);
        let store = SqliteStore::new(&db_path, &config.embedding_model(), 3072).unwrap();
        let ids: Vec<String> = [
            ("alpha", "streaming alpha"),
            ("beta", "streaming beta"),
            ("gamma", "streaming gamma"),
        ]
        .into_iter()
        .map(|(summary, content)| store.store(test_memory(summary, summary, content)).unwrap())
        .collect();
        assert_eq!(ids.len(), 3);

        let page1 = Request::builder()
            .method("GET")
            .uri("/api/recall_stream?q=streaming&limit=2&offset=0")
            .body(())
            .unwrap();
        let response = handle_rest_request(&page1, &config).await.unwrap();
        let json = read_json(response).await;
        assert_eq!(json["count"].as_u64(), Some(2));
        assert_eq!(json["offset"].as_u64(), Some(0));
        assert_eq!(json["limit"].as_u64(), Some(2));
        assert_eq!(json["next_offset"].as_u64(), Some(2));
        assert_eq!(json["has_more"].as_bool(), Some(true));
        assert_eq!(json["results"].as_array().unwrap().len(), 2);

        let page2 = Request::builder()
            .method("GET")
            .uri("/api/recall_stream?q=streaming&limit=2&offset=2")
            .body(())
            .unwrap();
        let response = handle_rest_request(&page2, &config).await.unwrap();
        let json = read_json(response).await;
        assert_eq!(json["count"].as_u64(), Some(1));
        assert_eq!(json["offset"].as_u64(), Some(2));
        assert_eq!(json["limit"].as_u64(), Some(2));
        assert!(json["next_offset"].is_null());
        assert_eq!(json["has_more"].as_bool(), Some(false));
        assert_eq!(json["results"].as_array().unwrap().len(), 1);

        let legacy = Request::builder()
            .method("GET")
            .uri("/api/memories?q=streaming&limit=2")
            .body(())
            .unwrap();
        let response = handle_rest_request(&legacy, &config).await.unwrap();
        let json = read_json(response).await;
        assert_eq!(json["count"].as_u64(), Some(2));
        assert!(json.get("next_offset").is_none());
        assert_eq!(json["results"].as_array().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn artifacts_endpoints_expose_proxy_artifacts_with_transcript() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("artifacts.db");
        let config = test_config(&db_path);
        let store = SqliteStore::new(&db_path, &config.embedding_model(), 3072).unwrap();

        let artifact_id = store
            .store_session_artifact(SessionArtifact {
                id: String::new(),
                schema_version: 1,
                artifact_kind: "proxy_first_party_ws".to_string(),
                session_id: Some("thread_123".to_string()),
                title: Some("WS GET /responses".to_string()),
                summary: Some("codex-first-party websocket mirror".to_string()),
                source_agent: Some("proxy".to_string()),
                source_label: Some("proxy:codex-first-party-ws".to_string()),
                is_subagent: false,
                started_at: None,
                ended_at: None,
                turn_count: 3,
                transcript_text: "authorization: <redacted>\nchatgpt-account-id: <redacted>\nresponse.output_text.delta\nhello ws".to_string(),
                transcript_json: Some(
                    r#"{"authorization":"<redacted>","chatgpt_account_id":"<redacted>","event":"response.output_text.delta"}"#.to_string(),
                ),
                episode_id: None,
                created_at: Utc::now(),
            })
            .unwrap();

        let list = Request::builder()
            .method("GET")
            .uri("/api/artifacts?limit=10&offset=0")
            .body(())
            .unwrap();
        let response = handle_rest_request(&list, &config).await.unwrap();
        let json = read_json(response).await;
        let artifacts = json["artifacts"].as_array().unwrap();
        assert_eq!(artifacts.len(), 1);
        assert_eq!(artifacts[0]["id"].as_str(), Some(artifact_id.as_str()));
        assert_eq!(
            artifacts[0]["artifact_kind"].as_str(),
            Some("proxy_first_party_ws")
        );
        assert_eq!(
            artifacts[0]["source_label"].as_str(),
            Some("proxy:codex-first-party-ws")
        );

        let detail = Request::builder()
            .method("GET")
            .uri(format!(
                "/api/artifacts/{}?include_transcript=true",
                artifact_id
            ))
            .body(())
            .unwrap();
        let response = handle_rest_request(&detail, &config).await.unwrap();
        let json = read_json(response).await;
        assert_eq!(json["id"].as_str(), Some(artifact_id.as_str()));
        assert_eq!(json["artifact_kind"].as_str(), Some("proxy_first_party_ws"));
        assert_eq!(json["session_id"].as_str(), Some("thread_123"));
        assert!(json["episode_id"].is_null());
        assert_eq!(json["transcript_available"].as_bool(), Some(true));
        assert!(json["transcript_text"]
            .as_str()
            .unwrap()
            .contains("response.output_text.delta"));
        assert!(json["transcript_text"]
            .as_str()
            .unwrap()
            .contains("<redacted>"));
        assert!(!json["transcript_text"]
            .as_str()
            .unwrap()
            .contains("Bearer "));
        assert!(json["transcript_json"]
            .as_str()
            .unwrap()
            .contains("<redacted>"));
    }

    // --- New tests: Stream B fixes (C3 auth, R2 raw-token redaction, H6 size cap, M4 gating) ---

    fn insert_artifact(store: &SqliteStore, text: &str, json_raw: Option<String>) -> String {
        store
            .store_session_artifact(SessionArtifact {
                id: String::new(),
                schema_version: 1,
                artifact_kind: "proxy_first_party_ws".to_string(),
                session_id: Some("thread_test".to_string()),
                title: None,
                summary: None,
                source_agent: Some("proxy".to_string()),
                source_label: Some("proxy:test".to_string()),
                is_subagent: false,
                started_at: None,
                ended_at: None,
                turn_count: 0,
                transcript_text: text.to_string(),
                transcript_json: json_raw,
                episode_id: None,
                created_at: Utc::now(),
            })
            .unwrap()
    }

    // Serialize env-var tests so parallel cases don't race on REIN_HTTP_TOKEN.
    // Async mutex so the guard is Send across `.await` (no `await_holding_lock`).
    fn env_lock() -> &'static tokio::sync::Mutex<()> {
        static LOCK: std::sync::OnceLock<tokio::sync::Mutex<()>> = std::sync::OnceLock::new();
        LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
    }

    #[tokio::test]
    #[serial_test::serial(global_state)]
    async fn api_artifacts_auth_gate_matrix() {
        let _guard = env_lock().lock().await;

        // Case 1: REIN_HTTP_TOKEN unset ⇒ permissive (dev convenience).
        std::env::remove_var("REIN_HTTP_TOKEN");
        let dir1 = tempdir().unwrap();
        let db_path1 = dir1.path().join("a1.db");
        let config1 = test_config(&db_path1);
        let _s1 = SqliteStore::new(&db_path1, &config1.embedding_model(), 3072).unwrap();
        let req = Request::builder()
            .method("GET")
            .uri("/api/artifacts")
            .body(())
            .unwrap();
        assert_eq!(
            handle_rest_request(&req, &config1).await.unwrap().status(),
            StatusCode::OK
        );

        // Case 2: token set, request without header → 401.
        std::env::set_var("REIN_HTTP_TOKEN", "secret-token-xyz");
        let dir2 = tempdir().unwrap();
        let db_path2 = dir2.path().join("a2.db");
        let config2 = test_config(&db_path2);
        let _s2 = SqliteStore::new(&db_path2, &config2.embedding_model(), 3072).unwrap();
        let req_no_token = Request::builder()
            .method("GET")
            .uri("/api/artifacts")
            .body(())
            .unwrap();
        assert_eq!(
            handle_rest_request(&req_no_token, &config2)
                .await
                .unwrap()
                .status(),
            StatusCode::UNAUTHORIZED
        );

        // Case 3: wrong token → 401.
        let req_wrong = Request::builder()
            .method("GET")
            .uri("/api/artifacts")
            .header("x-rein-token", "wrong")
            .body(())
            .unwrap();
        assert_eq!(
            handle_rest_request(&req_wrong, &config2)
                .await
                .unwrap()
                .status(),
            StatusCode::UNAUTHORIZED
        );

        // Case 4: correct token → 200.
        let req_ok = Request::builder()
            .method("GET")
            .uri("/api/artifacts")
            .header("x-rein-token", "secret-token-xyz")
            .body(())
            .unwrap();
        assert_eq!(
            handle_rest_request(&req_ok, &config2)
                .await
                .unwrap()
                .status(),
            StatusCode::OK
        );

        std::env::remove_var("REIN_HTTP_TOKEN");
    }

    #[tokio::test]
    #[serial_test::serial(global_state)]
    async fn api_artifact_detail_redacts_raw_bearer_tokens() {
        let _guard = env_lock().lock().await;
        std::env::remove_var("REIN_HTTP_TOKEN");
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("artifacts.db");
        let config = test_config(&db_path);
        let store = SqliteStore::new(&db_path, &config.embedding_model(), 3072).unwrap();

        // Raw unredacted Bearer token in transcript — the endpoint MUST redact it.
        let raw =
            "POST /responses\nAuthorization: Bearer sk-secret-raw-abc123xyz456\nContent-Type: application/json";
        let id = insert_artifact(&store, raw, None);

        let req = Request::builder()
            .method("GET")
            .uri(format!("/api/artifacts/{}?include_transcript=true", id))
            .body(())
            .unwrap();
        let response = handle_rest_request(&req, &config).await.unwrap();
        let json = read_json(response).await;
        let text = json["transcript_text"].as_str().unwrap();
        assert!(
            !text.contains("sk-secret-raw-abc123xyz456"),
            "raw bearer token must be redacted"
        );
    }

    #[tokio::test]
    #[serial_test::serial(global_state)]
    async fn api_artifact_detail_omits_transcripts_when_include_false() {
        let _guard = env_lock().lock().await;
        std::env::remove_var("REIN_HTTP_TOKEN");
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("artifacts.db");
        let config = test_config(&db_path);
        let store = SqliteStore::new(&db_path, &config.embedding_model(), 3072).unwrap();

        let id = insert_artifact(
            &store,
            "Authorization: Bearer sk-test-token-abcdef123456\nbody",
            Some(r#"{"secret":"sk-test-token-abcdef123456"}"#.to_string()),
        );

        // Default (no include_transcript=true) → both transcript fields omitted.
        let req = Request::builder()
            .method("GET")
            .uri(format!("/api/artifacts/{}", id))
            .body(())
            .unwrap();
        let response = handle_rest_request(&req, &config).await.unwrap();
        let json = read_json(response).await;
        assert!(
            json["transcript_text"].is_null(),
            "transcript_text must be null when include_transcript=false"
        );
        assert!(
            json["transcript_json"].is_null(),
            "transcript_json must be null when include_transcript=false"
        );
        // transcript_available is still reported so the GUI can show a button.
        assert_eq!(json["transcript_available"].as_bool(), Some(true));
    }

    #[tokio::test]
    #[serial_test::serial(global_state)]
    async fn api_artifact_detail_truncates_oversize_body() {
        let _guard = env_lock().lock().await;
        std::env::remove_var("REIN_HTTP_TOKEN");
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("artifacts.db");
        let config = test_config(&db_path);
        let store = SqliteStore::new(&db_path, &config.embedding_model(), 3072).unwrap();

        // 3 MiB body exceeds the 2 MiB cap.
        let big = "x".repeat(MAX_ARTIFACT_RESPONSE_BYTES + 1024 * 1024);
        let id = insert_artifact(&store, &big, None);

        let req = Request::builder()
            .method("GET")
            .uri(format!("/api/artifacts/{}?include_transcript=true", id))
            .body(())
            .unwrap();
        let response = handle_rest_request(&req, &config).await.unwrap();
        let json = read_json(response).await;
        assert_eq!(json["transcript_truncated"].as_bool(), Some(true));
        let returned = json["transcript_text"].as_str().unwrap();
        assert!(
            returned.len() <= MAX_ARTIFACT_RESPONSE_BYTES,
            "returned body must be <= cap, got {}",
            returned.len()
        );
    }
}
