//! REST API layer for the rein web GUI.
//! Routes `/api/*` requests to store/ops functions, returning JSON.
//! Also serves the embedded SPA when the `gui` feature is enabled.

use bytes::Bytes;
use http_body_util::combinators::BoxBody;
use http_body_util::{BodyExt, Full};
use hyper::{Method, Request, Response, StatusCode};
use serde_json::json;

use crate::config::ReinConfig;

type BoxedResponse = Response<BoxBody<Bytes, std::convert::Infallible>>;
const HTTP_SESSION_COOKIE: &str = "rein_http_token";

/// Default cap on REST request body size. Overridable via
/// `REIN_REST_MAX_BODY_BYTES`. 1 MiB is generous for JSON params on any
/// current op; ingest_session enforces its own 500 KB ceiling after decode.
const DEFAULT_MAX_BODY_BYTES: usize = 1024 * 1024;

/// Read the full body into a `Bytes` buffer, rejecting anything beyond the
/// configured cap. Cap is enforced **progressively** — chunks are checked
/// as they arrive so a chunked body that reports `upper = None` (no
/// advertised upper bound) cannot buffer gigabytes before the cap fires.
/// The `size_hint().upper()` pre-check still runs for well-behaved bodies
/// so we short-circuit before reading anything when possible.
///
/// Returns `Err(413-response)` when the cap is exceeded and
/// `Err(400-response)` on transport errors reading the body.
pub async fn collect_body_capped<B>(body: B) -> Result<Bytes, BoxedResponse>
where
    B: hyper::body::Body<Data = Bytes> + Unpin,
    B::Error: std::fmt::Display,
{
    let max = std::env::var("REIN_REST_MAX_BODY_BYTES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_MAX_BODY_BYTES);

    // Size-hint pre-check: if the body advertises an upper bound above the
    // cap, reject immediately instead of streaming until we blow the limit.
    // Not authoritative (size_hint is advisory) — the progressive check
    // below is the real guard.
    let hint = body.size_hint();
    if let Some(upper) = hint.upper() {
        if upper as usize > max {
            return Err(error_response(
                StatusCode::PAYLOAD_TOO_LARGE,
                &format!("request body exceeds {max} byte cap"),
            ));
        }
    }

    let mut body = body;
    let mut buf = bytes::BytesMut::with_capacity(hint.lower().min(max as u64) as usize);
    loop {
        let next = std::future::poll_fn(|cx| std::pin::Pin::new(&mut body).poll_frame(cx)).await;
        let frame = match next {
            Some(Ok(frame)) => frame,
            Some(Err(e)) => {
                return Err(error_response(
                    StatusCode::BAD_REQUEST,
                    &format!("body read error: {e}"),
                ));
            }
            None => break,
        };
        // Only data frames contribute to the body size cap. Trailer frames
        // (if any) are ignored for cap accounting but still consumed.
        if let Ok(data) = frame.into_data() {
            if buf.len().saturating_add(data.len()) > max {
                return Err(error_response(
                    StatusCode::PAYLOAD_TOO_LARGE,
                    &format!("request body exceeds {max} byte cap"),
                ));
            }
            buf.extend_from_slice(&data);
        }
    }
    Ok(buf.freeze())
}

fn json_response(status: StatusCode, body: serde_json::Value) -> BoxedResponse {
    let json_bytes = serde_json::to_vec(&body).unwrap_or_default();
    // AGPL §13: every network response carries a pointer to the source.
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .header("x-source-code", crate::SOURCE_URL)
        .header("x-license", crate::LICENSE_SPDX)
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
    // AGPL §13: every network response carries a pointer to the source.
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .header("set-cookie", cookie)
        .header("x-source-code", crate::SOURCE_URL)
        .header("x-license", crate::LICENSE_SPDX)
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
                    // Skip pairs with invalid UTF-8 in key or value.
                    Some((
                        percent_decode_component(key, true)?,
                        percent_decode_component(value, true)?,
                    ))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Percent-decode a URI component. Query-string components treat `+` as a
/// space; path segments must preserve literal `+`.
fn percent_decode_component(s: &str, plus_as_space: bool) -> Option<String> {
    let s = if plus_as_space {
        s.replace('+', " ")
    } else {
        s.to_string()
    };
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
    String::from_utf8(out).ok()
}

/// Percent-decode with lossy fallback for legacy route handlers that extract
/// IDs from path strings directly. Returns the raw string on UTF-8 failure so
/// downstream handlers can handle the invalid ID (typically a 404 from the DB
/// query). Use `percent_decode` (returns `Option`) for template matching.
fn percent_decode_lossy(s: &str) -> String {
    percent_decode_component(s, false).unwrap_or_else(|| s.to_string())
}

/// Try to handle an API or GUI request. Returns `Some(response)` if matched, `None` to fall through to MCP.
///
/// This body-less variant is the backwards-compatible entry point used by
/// tests that build requests with `.body(())`. Production callers should use
/// [`handle_rest_request_with_body`] which threads the collected body bytes
/// through to JSON-body ops (H5). A None body at an inventory POST/PUT/PATCH/
/// DELETE with a params struct yields a 400 from the macro-emitted prep block.
pub async fn handle_rest_request<B>(
    req: &Request<B>,
    config: &ReinConfig,
) -> Option<BoxedResponse> {
    handle_rest_request_with_body(req, None, config).await
}

/// Resolve the inventory `OpsRestEntry` for a given method + path without
/// building path values. Tries exact match first, then template match.
///
/// Used in two phases: (1) pre-body auth gate in `handle_api_request`, and
/// (2) inside `try_dispatch_inventory_rest` where path values are extracted.
/// Doing both with the same resolver guarantees they agree on which entry wins.
pub(crate) fn resolve_route(
    method: &Method,
    path: &str,
) -> Option<&'static crate::ops::OpsRestEntry> {
    // First: O(n) exact match on path_template string — cheapest case.
    if let Some(entry) = inventory::iter::<crate::ops::OpsRestEntry>()
        .find(|e| e.method == *method && e.path_template == path)
    {
        return Some(entry);
    }
    // Second: template match — only entries with non-empty path_segments.
    let req_segs = split_path_segments(path);
    inventory::iter::<crate::ops::OpsRestEntry>()
        .filter(|e| e.method == *method && !e.path_segments.is_empty())
        .find(|e| match_path_template(e.path_segments, &req_segs).is_some())
}

/// Production entry point for `/api/*` requests. Consumes the `Request<B>`
/// by value so body bytes can be collected for POST/PUT/PATCH/DELETE ops,
/// then rebuilds a body-less probe (`Request<()>`) that downstream REST
/// handlers use for header and URI access.
///
/// Always returns a response — `/api/*` paths either match a route or
/// receive a 404. Callers do not need to handle fall-through for this
/// entry point.
///
/// **Auth ordering invariant**: for inventory routes, the declared
/// `AuthPolicy` is enforced *before* body collection. An unauthenticated
/// POST must hit 403 without paying the 1 MiB body read — otherwise an
/// anonymous client can force the server to drain a large body and use
/// the 403/413 status difference to probe the body cap. The resolve_route
/// helper runs both exact and template match so templated routes are
/// protected at the same point as literal routes.
pub async fn handle_api_request<B>(req: Request<B>, config: &ReinConfig) -> BoxedResponse
where
    B: hyper::body::Body<Data = Bytes> + Unpin,
    B::Error: std::fmt::Display,
{
    // Pre-body auth gate: resolve the inventory entry (exact or template) and
    // enforce its declared AuthPolicy before any body bytes are consumed.
    // Routes not in inventory (legacy match arms in handle_api) do their own
    // auth checks inline, so no ordering inversion there.
    if let Some(entry) = resolve_route(req.method(), req.uri().path()) {
        if !matches!(entry.auth_policy, crate::ops::AuthPolicy::Public) {
            if let Err(resp) = enforce_auth_policy(&req, entry.auth_policy) {
                return resp;
            }
        }
    }

    let is_body_method = matches!(
        *req.method(),
        Method::POST | Method::PUT | Method::PATCH | Method::DELETE,
    );
    let (parts, body) = req.into_parts();
    let body_bytes = if is_body_method {
        match collect_body_capped(body).await {
            Ok(bytes) => Some(bytes),
            Err(resp) => return resp,
        }
    } else {
        None
    };
    let probe: Request<()> = Request::from_parts(parts, ());
    handle_rest_request_with_body(&probe, body_bytes, config)
        .await
        .unwrap_or_else(|| error_response(StatusCode::NOT_FOUND, "unknown API endpoint"))
}

/// Body-aware variant. Production service_fn collects the request body into
/// `Bytes` (with a max-size cap) and passes `Some(bytes)` here so POST/PUT/
/// PATCH/DELETE inventory ops can decode their JSON body. Tests that need to
/// exercise the body-JSON path construct Bytes directly and call this.
pub async fn handle_rest_request_with_body<B>(
    req: &Request<B>,
    body: Option<Bytes>,
    config: &ReinConfig,
) -> Option<BoxedResponse> {
    crate::ops::inventory::ensure_unique_registrations();
    let path = req.uri().path();
    let method = req.method();

    // Only intercept /api/* and GUI asset paths
    if path.starts_with("/api/") {
        Some(handle_api(req, method, path, req.uri(), body, config).await)
    } else if config.server.gui_enabled && !path.starts_with("/mcp") {
        Some(serve_gui(path))
    } else {
        None
    }
}

/// Dispatch the declared `AuthPolicy` to the existing per-kind gate helpers.
/// Returns `Err(response)` when the request fails the gate; the caller
/// (inventory REST dispatcher) short-circuits with that response.
#[allow(clippy::result_large_err)]
fn enforce_auth_policy<B>(
    req: &Request<B>,
    policy: crate::ops::AuthPolicy,
) -> Result<(), BoxedResponse> {
    match policy {
        crate::ops::AuthPolicy::Public => Ok(()),
        crate::ops::AuthPolicy::MutationMarker => require_mutation_marker(req),
        crate::ops::AuthPolicy::ReadToken => require_read_token(req),
    }
}

fn constant_time_eq(left: &str, right: &str) -> bool {
    use sha2::{Digest, Sha256};

    let left_hash = Sha256::digest(left.as_bytes());
    let right_hash = Sha256::digest(right.as_bytes());
    left_hash
        .iter()
        .zip(right_hash.iter())
        .fold(0u8, |acc, (x, y)| acc | (x ^ y))
        == 0
}

fn cookie_value(headers: &hyper::HeaderMap, name: &str) -> Option<String> {
    let cookies = headers.get("cookie")?.to_str().ok()?;
    cookies.split(';').find_map(|part| {
        let (key, value) = part.trim().split_once('=')?;
        if key.trim() == name {
            Some(value.trim().to_string())
        } else {
            None
        }
    })
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
    let auth_header = req
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let expected_bearer = format!("Bearer {}", expected.trim());
    let session_cookie = cookie_value(req.headers(), HTTP_SESSION_COOKIE).unwrap_or_default();
    if constant_time_eq(presented, expected.trim())
        || constant_time_eq(auth_header, &expected_bearer)
        || constant_time_eq(&session_cookie, expected.trim())
    {
        Ok(())
    } else {
        Err(error_response(
            StatusCode::UNAUTHORIZED,
            "missing or invalid protected-read credential",
        ))
    }
}

async fn handle_api<B>(
    req: &Request<B>,
    method: &Method,
    path: &str,
    uri: &hyper::Uri,
    body: Option<Bytes>,
    config: &ReinConfig,
) -> BoxedResponse {
    let query = parse_query(uri);

    // Pre-inventory guard: GET /api/doctor with fix=true is disallowed — fixes
    // require POST with the mutation marker. Without this check the inventory
    // doctor op would happily run with fix=true on a read-method request.
    if method == Method::GET
        && path == "/api/doctor"
        && query
            .get("fix")
            .map(|v| matches!(v.as_str(), "1" | "true" | "yes"))
            .unwrap_or(false)
    {
        return error_response(
            StatusCode::METHOD_NOT_ALLOWED,
            "doctor fixes require POST /api/doctor?fix=true",
        );
    }

    // A1: migrated ops get first crack via OpsRestEntry inventory. Falls through
    // to the legacy match for routes that aren't yet migrated. The `req` ref is
    // threaded in so the H3 auth gate can inspect headers before dispatch.
    // H5 (Phase 2.2): body bytes are threaded in for POST/PUT/PATCH/DELETE ops
    // that declare a params struct; GET ops still read from query string.
    if let Some(resp) = try_dispatch_inventory_rest(req, method, path, uri, body, config).await {
        return resp;
    }

    match (method, path) {
        // --- Read endpoints ---
        (&Method::GET, "/api/activity") => api_activity(config, &query),
        // GET /api/dedup_decisions migrated to #[op] inventory (dedup_log op in
        // ops/handlers/maintenance.rs); served via try_dispatch_inventory_rest above.
        (&Method::GET, "/api/intelligent_merge_metrics") => api_intelligent_merge_metrics(),
        // v0.27.1 E direction (spec §7 Layer 2): drift κ + Layer 1 J3 κ +
        // alert counts. Pure read; no auth. Cold-start (no AdaptiveState
        // row, no judge_calibration_state field) returns the zero-value
        // shape so the GUI never sees `undefined`.
        (&Method::GET, "/api/judge/calibration") => api_judge_calibration(config),
        (&Method::GET, "/api/version") => json_response(
            StatusCode::OK,
            json!({ "version": env!("CARGO_PKG_VERSION") }),
        ),
        // Both GET and POST /api/doctor are served via OpsRestEntry
        // inventory (see ops/handlers/diagnostics.rs). POST migrated in
        // Phase 2.2 as the first `auth = "mutation_marker"` real consumer;
        // H5 JSON body carries the `network` flag. GUI/curl clients must
        // send body instead of `?fix=true` query string.
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
            match require_read_token(req) {
                Ok(()) => api_recall(config, &query),
                Err(response) => response,
            }
        }
        // GET /api/memoirs migrated to #[op] inventory (see ops/handlers/knowledge.rs).
        // The inventory dispatcher at try_dispatch_inventory_rest intercepts this path
        // before the match arm below, leaving only the sub-path handler in place.
        (&Method::GET, p) if p.starts_with("/api/memoirs/") => {
            handle_memoir_path(config, p, &query)
        }
        (&Method::GET, "/api/timeline") => api_timeline(config, &query),
        (&Method::GET, "/api/episodes") => api_episodes(config, &query),
        (&Method::GET, "/api/artifacts") => match require_read_token(req) {
            Ok(()) => api_artifacts(config, &query),
            Err(response) => response,
        },
        (&Method::GET, p) if p.starts_with("/api/artifacts/") => match require_read_token(req) {
            Ok(()) => {
                let id = percent_decode_lossy(&p["/api/artifacts/".len()..]);
                api_artifact_detail(config, &id, &query)
            }
            Err(response) => response,
        },

        _ => error_response(StatusCode::NOT_FOUND, "unknown API endpoint"),
    }
}

fn handle_memoir_path(
    config: &ReinConfig,
    path: &str,
    query: &std::collections::HashMap<String, String>,
) -> BoxedResponse {
    // After Phase 3, the inventory dispatcher serves:
    //   GET /api/memoirs               → memoir_list
    //   GET /api/memoirs/{name}        → memoir_show
    //   GET /api/memoirs/{name}/export → memoir_export (all formats,
    //     with IntoJson::to_raw_response picking text/plain for ascii/dot
    //     and application/json for json).
    //
    // The only path that still needs this legacy helper is
    // /api/memoirs/{name}/inspect/{concept}, which has two path parameters
    // and therefore can't bind to the single-seg path-template framework
    // (spec §Q2). v0.22+ double-seg support will fold this in.
    let rest = &path["/api/memoirs/".len()..];
    let Some(slash) = rest.find('/') else {
        return error_response(StatusCode::NOT_FOUND, "unknown memoir API endpoint");
    };
    let name = &percent_decode_lossy(&rest[..slash]);
    let sub = &rest[slash + 1..];
    if let Some(concept_rest) = sub.strip_prefix("inspect/") {
        let concept = percent_decode_lossy(concept_rest);
        let depth = match parse_bounded_usize(query, "depth", 1, 1, 8) {
            Ok(depth) => depth,
            Err(msg) => return error_response(StatusCode::BAD_REQUEST, &msg),
        };
        return api_memoir_inspect(config, name, &concept, depth);
    }
    error_response(StatusCode::NOT_FOUND, "unknown memoir API endpoint")
}

// ===========================================================================
// API handlers
// ===========================================================================

// api_stats / api_health migrated to #[op] (see ops/handlers/diagnostics.rs).
// `try_dispatch_inventory_rest` intercepts /api/stats + /api/health before the
// legacy match in `handle_api`.

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

/// v0.27.1 E direction (spec §7) — surface `JudgeCalibrationState` to the GUI
/// + doctor + operators. Cold-start (consumer hasn't run yet) returns the
/// zero-value shape so the GUI never sees `null` / `undefined` for the
/// numeric fields.
fn api_judge_calibration(config: &ReinConfig) -> BoxedResponse {
    // Codex R2 P2 fix — `database.path` defaults to `"auto"` (sentinel for
    // `~/.rein/memories.db`). Passing the literal opens a stray DB named
    // `auto` instead of the resolved store. Use `resolve_db_path()`
    // which expands the sentinel + honors `REIN_DB` env override.
    let store = match crate::store::SqliteStore::new(
        &config.resolve_db_path(),
        &config.embedding_model(),
        config.embedding.dimensions,
    ) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(target: "rein.judge", "api_judge_calibration: store open failed: {e}");
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "judge calibration store unavailable",
            );
        }
    };
    let state =
        crate::store::adaptive::AdaptiveState::restore_snapshot(store.conn()).unwrap_or_default();
    let cal = state.judge_calibration_state.unwrap_or_default();
    let body = json!({
        "kappa": cal.kappa,
        "runtime_vs_offline_kappa": cal.runtime_vs_offline_kappa,
        "judge_drift_alert": cal.judge_drift_alert,
        "total_offline_cron_events": cal.total_offline_cron_events,
        "last_consumed_event_id_calibration": cal.last_consumed_event_id_calibration,
        "last_computed_at": cal.last_computed_at,
        "recent_pairs_synthesis_count": cal.recent_pairs_synthesis.len(),
        "recent_pairs_concept_count": cal.recent_pairs_concept.len(),
        "recent_pairs_runtime_vs_offline_count": cal.recent_pairs_runtime_vs_offline.len(),
    });
    json_response(StatusCode::OK, body)
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

/// Split a request path into segments using leading-only empty filtering.
/// The leading `/` produces an empty first token which is dropped; trailing
/// slashes are kept (they generate a trailing empty segment that causes a
/// length mismatch against templates, yielding a 404 per spec §Q3).
fn split_path_segments(path: &str) -> Vec<&str> {
    let stripped = path.trim_start_matches('/');
    if stripped.is_empty() {
        return vec![];
    }
    stripped.split('/').collect()
}

/// Attempt to match `req_segs` against `template_segs`. On full match returns
/// `Some(HashMap)` of `{ param_name → decoded_value }`. Returns `None` on any
/// mismatch (length difference, literal mismatch, or UTF-8 decode failure).
///
/// Percent-decoding happens AFTER segment split (spec §5 — decoding before
/// split would let `%2F` corrupt segment boundaries).
fn match_path_template(
    template_segs: &[crate::ops::PathSegment],
    req_segs: &[&str],
) -> Option<std::collections::HashMap<&'static str, String>> {
    if template_segs.len() != req_segs.len() {
        return None;
    }
    let mut values = std::collections::HashMap::new();
    for (tmpl, actual) in template_segs.iter().zip(req_segs.iter()) {
        match tmpl {
            crate::ops::PathSegment::Literal(lit) => {
                if *actual != *lit {
                    return None;
                }
            }
            crate::ops::PathSegment::Param(name) => {
                // Reject empty segments — an empty path segment cannot bind a
                // meaningful param value (spec: trailing slash or double slash).
                if actual.is_empty() {
                    return None;
                }
                // Percent-decode after split — spec §5 (decode before split
                // would let %2F corrupt segment boundaries).
                let decoded = percent_decode_component(actual, false)?;
                values.insert(*name, decoded);
            }
        }
    }
    Some(values)
}

async fn try_dispatch_inventory_rest<B>(
    req: &Request<B>,
    method: &Method,
    path: &str,
    uri: &hyper::Uri,
    body: Option<Bytes>,
    config: &ReinConfig,
) -> Option<BoxedResponse> {
    // Two-pass dispatch: exact match first, then template match on miss.

    // First pass: exact match — unchanged hot path.
    let exact_entry = inventory::iter::<crate::ops::OpsRestEntry>()
        .find(|e| e.method == *method && e.path_template == path);

    let (entry, path_values) = if let Some(e) = exact_entry {
        (e, std::collections::HashMap::new())
    } else {
        // Second pass: template match against entries with non-empty path_segments.
        let req_segs = split_path_segments(path);
        let template_hit = inventory::iter::<crate::ops::OpsRestEntry>()
            .filter(|e| e.method == *method && !e.path_segments.is_empty())
            .find_map(|e| match_path_template(e.path_segments, &req_segs).map(|vals| (e, vals)));
        template_hit?
    };

    // Enforce the entry's declared AuthPolicy. For requests arriving through
    // handle_api_request, auth was already enforced pre-body via resolve_route.
    // This secondary check covers requests that arrive through the body-less
    // handle_rest_request_with_body path (e.g. tests, legacy callers).
    if let Err(resp) = enforce_auth_policy(req, entry.auth_policy) {
        return Some(resp);
    }

    let query = uri.query().unwrap_or("").to_string();
    let runtime = std::sync::Arc::new(crate::ops::OpsRuntime::for_rest(std::sync::Arc::new(
        config.clone(),
    )));

    match (entry.invoke)(runtime, path_values, query, body).await {
        // Phase 3: ops return (status, body, content_type); inventory
        // respects whatever content-type the op emits (JSON by default via
        // the macro, or a raw content-type if the op implements
        // IntoJson::to_raw_response).
        Ok((status, body, content_type)) => Some(
            Response::builder()
                .status(status)
                .header("content-type", content_type)
                .header("x-source-code", crate::SOURCE_URL)
                .header("x-license", crate::LICENSE_SPDX)
                .body(
                    Full::new(body)
                        .map_err(|never: std::convert::Infallible| match never {})
                        .boxed(),
                )
                .unwrap_or_else(|_| {
                    error_response(StatusCode::INTERNAL_SERVER_ERROR, "build response")
                }),
        ),
        Err(e) => {
            // A1 H4 (audit 2026-04-19): honor ReinError::kind() so handlers
            // that tag BadRequest / NotFound / Forbidden / Conflict surface
            // the right HTTP status to REST clients. Pre-H4 this was a
            // hardcoded 500 regardless of error kind.
            let status = e.kind().status_code();
            Some(error_response(status, &e.to_string()))
        }
    }
}

// api_doctor migrated to #[op(rest = POST /api/doctor)] in Phase 2.2
// (see ops/handlers/diagnostics.rs::doctor_fix). Legacy callers that sent
// query-string flags now send a JSON body.

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

fn recall_synthesis_adaptive_state(
    config: &ReinConfig,
) -> Option<crate::store::adaptive::AdaptiveState> {
    config
        .open_store()
        .ok()
        .and_then(|s| crate::store::adaptive::AdaptiveState::restore_snapshot(s.conn()))
}

fn api_recall(
    config: &ReinConfig,
    query: &std::collections::HashMap<String, String>,
) -> BoxedResponse {
    match run_recall_query(config, query, None) {
        Ok((results, request_id, synthesize_query)) => {
            // Cap B (v0.25): when ?synthesize=true is set on /api/memories,
            // run recall-time synthesis. The LLM call sits OUTSIDE any store
            // open block (results already collected) so block_in_place never
            // nests inside any store transaction guard.
            //
            // v0.26 D direction: load `AdaptiveState` so per-query gate has
            // parity with the MCP/CLI path (`ops/handlers/memory.rs:710`).
            // Without this, REST callers always fall to the global flag and
            // the per-cluster `useful_rate` signal is invisible to REST
            // recalls — the very deployments most likely to feed back
            // `synthesis_interaction` events via the GUI. Codex round 1 F-6.
            let adaptive_state = recall_synthesis_adaptive_state(config);
            // v0.26.1: classify the original query so the synthesis gate
            // reads the matching per-cluster bucket (parity with the
            // MCP/CLI path in `ops/handlers/memory.rs:673` which already
            // classifies for routing). The classifier is rule-based and
            // pure — no LLM cost — so calling it twice across the recall
            // pipeline is fine.
            let route =
                crate::search::classify::classify(&synthesize_query.original_query, false, false);
            let synthesis = crate::ops::recall_synthesis::run_recall_synthesis(
                &results,
                &synthesize_query.original_query,
                config,
                synthesize_query.synthesize,
                route.query_type.synthesis_bucket_label(),
                adaptive_state.as_ref(),
                None,
            );
            recall_results_response(results, 0, None, &request_id, synthesis)
        }
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
        Ok((results, request_id, _)) => {
            // Synthesis intentionally NOT wired into recall_stream: paginated
            // streams are stateless across pages, and synthesizing on each
            // page would duplicate LLM cost. Callers wanting synthesis use
            // /api/memories (single-page) instead.
            recall_results_response(results, offset, Some(page_limit), &request_id, None)
        }
        Err(response) => response,
    }
}

/// Inputs the synthesis path needs that aren't already in the results tuple:
/// the original query string (for the synthesis prompt + `query` field on
/// `RecallSynthesisOutcome`) and the parsed `synthesize` flag.
struct RecallSynthesisInputs {
    original_query: String,
    synthesize: Option<bool>,
}

#[allow(clippy::result_large_err)] // BoxedResponse is already a boxed body.
fn run_recall_query(
    config: &ReinConfig,
    query: &std::collections::HashMap<String, String>,
    limit_override: Option<usize>,
) -> Result<
    (
        Vec<crate::search::recall::RecallResult>,
        String,
        RecallSynthesisInputs,
    ),
    BoxedResponse,
> {
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
    // Cap B (v0.25): parse ?synthesize=true|false. Anything else (or absent)
    // is treated as "not requested" — `run_recall_synthesis` returns None,
    // so the response shape stays bit-identical to pre-Cap-B for legacy
    // callers that never pass this param.
    let synthesize = query.get("synthesize").and_then(|s| match s.as_str() {
        "true" | "1" | "yes" => Some(true),
        "false" | "0" | "no" => Some(false),
        _ => None,
    });

    let store = match config.open_store() {
        Ok(s) => s,
        Err(e) => {
            return Err(error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &e.to_string(),
            ))
        }
    };

    // Generate a request_id here so feedback submitted against this recall
    // (via `POST /api/feedback` with request_id=...) can be correlated by M1.
    // Clients observe it in the `request_id` field of the recall response.
    let request_id = ulid::Ulid::new().to_string();
    let results = crate::search::recall::recall_temporal_with_request_id(
        &store,
        config,
        &q,
        topic,
        keyword,
        limit,
        from,
        to,
        None,
        false,
        Some(request_id.clone()),
    )
    .map_err(|e| error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()))?;
    Ok((
        results,
        request_id,
        RecallSynthesisInputs {
            original_query: q,
            synthesize,
        },
    ))
}

fn recall_result_to_json(r: &crate::search::recall::RecallResult) -> serde_json::Value {
    let mut memory = memory_to_json(&r.memory);
    if let Some(obj) = memory.as_object_mut() {
        obj.insert("score".to_string(), json!(r.score));
        obj.insert("confidence".to_string(), json!(r.confidence));
        obj.insert("sources_hit".to_string(), json!(r.sources_hit));
        obj.insert("evidence_count".to_string(), json!(r.evidence_count));
        obj.insert("evidence_preview".to_string(), json!(r.evidence_preview));
        // v0.26.2 (Bug #O3) + R4 F2: hand-serialize `archival_summary`
        // honoring the omit-when-None contract. `RecallResult` declares
        // `#[serde(skip_serializing_if = "Option::is_none")]` and the TS
        // type is `archival_summary?: string` — emitting literal `null`
        // for every Hot/Warm response (or when Cap C is disabled) would
        // change the wire shape for every recall and trip clients that
        // branch on property presence vs `undefined`. Only insert the
        // key when there's a real summary to surface.
        if let Some(s) = &r.archival_summary {
            obj.insert("archival_summary".to_string(), json!(s));
        }
    }
    memory
}

fn recall_results_response(
    results: Vec<crate::search::recall::RecallResult>,
    offset: usize,
    page_limit: Option<usize>,
    request_id: &str,
    synthesis: Option<crate::ops::recall_synthesis::RecallSynthesisOutcome>,
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

    // Cap B: serialize the synthesis outcome only if present. Pre-Cap-B
    // callers that never pass ?synthesize=true get None here, so the
    // response shape stays bit-identical (no `synthesis` field emitted).
    let synthesis_value = synthesis
        .as_ref()
        .map(|s| serde_json::to_value(s).unwrap_or(serde_json::Value::Null));

    if page_limit.is_some() {
        let end = offset.saturating_add(page.len());
        let has_more = results.len() > end;
        let next_offset = if has_more { Some(end) } else { None };
        let mut body = json!({
            "results": page,
            "count": page.len(),
            "offset": offset,
            "limit": page_limit,
            "next_offset": next_offset,
            "has_more": has_more,
            "request_id": request_id,
        });
        if let Some(s) = synthesis_value {
            if let Some(obj) = body.as_object_mut() {
                obj.insert("synthesis".to_string(), s);
            }
        }
        json_response(StatusCode::OK, body)
    } else {
        let mut body = json!({
            "results": page,
            "count": page.len(),
            "request_id": request_id,
        });
        if let Some(s) = synthesis_value {
            if let Some(obj) = body.as_object_mut() {
                obj.insert("synthesis".to_string(), s);
            }
        }
        json_response(StatusCode::OK, body)
    }
}

// api_memoirs migrated to #[op] (see ops/handlers/knowledge.rs::memoir_list).

// api_memoir_show migrated to #[op] (see ops/handlers/knowledge.rs::memoir_show).

// api_memoir_export fully migrated to #[op] (see ops/handlers/knowledge.rs::memoir_export).
// Phase 3 folded the ascii/dot text/plain branch into the inventory op via
// IntoJson::to_raw_response, so no helper remains here.

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
        Err(e) => error_response(e.kind().status_code(), &e.to_string()),
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
    let page_limit = match parse_bounded_usize(query, "limit", 20, 1, 100) {
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

    // Pagination response shape:
    //   { artifacts, count, offset, limit, next_offset, has_more }
    // `artifacts` remains the legacy GUI field; new callers should use
    // `has_more` instead of inferring from `artifacts.length == limit`.
    let fetch_limit = page_limit.saturating_add(1);

    // Query session_artifacts table directly
    let sql = "SELECT id, artifact_kind, title, summary, source_agent, source_label, \
               turn_count, episode_id, created_at FROM session_artifacts \
               ORDER BY created_at DESC LIMIT ?1 OFFSET ?2";
    let result = store.conn().prepare(sql).and_then(|mut stmt| {
        let rows = stmt.query_map(rusqlite::params![fetch_limit, offset], |row| {
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
        Ok(mut artifacts) => {
            let has_more = artifacts.len() > page_limit;
            if has_more {
                artifacts.truncate(page_limit);
            }
            let next_offset = if has_more {
                Some(offset.saturating_add(artifacts.len()))
            } else {
                None
            };
            json_response(
                StatusCode::OK,
                json!({
                    "artifacts": artifacts,
                    "count": artifacts.len(),
                    "offset": offset,
                    "limit": page_limit,
                    "next_offset": next_offset,
                    "has_more": has_more,
                }),
            )
        }
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

pub(crate) fn memory_to_json(m: &crate::types::Memory) -> serde_json::Value {
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
        "decay_lambda": m.decay_lambda,
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
            return gui_response_builder(mime)
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
            return gui_response_builder("text/html")
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

/// Builder for SPA responses with hardened security headers. Blocks
/// clickjacking (X-Frame-Options), MIME sniffing, and third-party script/
/// resource inclusion via a tight CSP suitable for the embedded SPA.
#[cfg(feature = "gui")]
fn gui_response_builder(mime: &'static str) -> hyper::http::response::Builder {
    Response::builder()
        .status(200)
        .header("content-type", mime)
        .header(
            "content-security-policy",
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; \
             img-src 'self' data:; font-src 'self' data:; connect-src 'self'; \
             frame-ancestors 'none'; base-uri 'self'; form-action 'self'",
        )
        .header("x-frame-options", "DENY")
        .header("x-content-type-options", "nosniff")
        .header("referrer-policy", "no-referrer")
        // AGPL §13: even SPA static-asset responses carry the source pointer
        // so a curl user pulling the GUI from a network deployment sees it.
        .header("x-source-code", crate::SOURCE_URL)
        .header("x-license", crate::LICENSE_SPDX)
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
    use crate::types::MemoryStore;
    use crate::types::{
        Importance, Memory, MemoryLayer, MemoryStatus, MemoryTier, SessionArtifact, Source,
    };
    use chrono::Utc;
    use hyper::Request;
    use tempfile::tempdir;

    // ---- H3 auth policy dispatch tests ----

    fn req_with<B: Default>(header: Option<(&str, &str)>) -> Request<B> {
        let mut builder = Request::builder().uri("/api/x");
        if let Some((k, v)) = header {
            builder = builder.header(k, v);
        }
        builder.body(B::default()).unwrap()
    }

    #[test]
    fn constant_time_eq_matches_proxy_digest_style() {
        assert!(constant_time_eq("secret-token", "secret-token"));
        assert!(!constant_time_eq("secret-token", "secret"));
        assert!(!constant_time_eq("secret-token", "secret-token-longer"));
    }

    #[test]
    fn enforce_auth_public_always_passes() {
        let req: Request<String> = req_with(None);
        assert!(enforce_auth_policy(&req, crate::ops::AuthPolicy::Public).is_ok());
    }

    #[test]
    fn enforce_auth_mutation_marker_rejects_missing_header() {
        let req: Request<String> = req_with(None);
        let result = enforce_auth_policy(&req, crate::ops::AuthPolicy::MutationMarker);
        assert!(
            result.is_err(),
            "missing x-rein-action: 1 header must be rejected"
        );
    }

    #[test]
    fn enforce_auth_mutation_marker_accepts_correct_header() {
        let req: Request<String> = req_with(Some(("x-rein-action", "1")));
        let result = enforce_auth_policy(&req, crate::ops::AuthPolicy::MutationMarker);
        assert!(result.is_ok(), "x-rein-action: 1 must pass the gate");
    }

    #[test]
    #[serial_test::serial(rein_http_token_env)]
    fn enforce_auth_read_token_is_permissive_when_env_unset() {
        // Dev-mode behavior: unset REIN_HTTP_TOKEN → read_token is effectively
        // public. The #[serial] attribute serializes against other tests that
        // touch the same env var (Codex 2026-04-19 audit flagged this as a
        // flake risk without explicit serialization).
        let original = std::env::var("REIN_HTTP_TOKEN").ok();
        // SAFETY: test-only temporary manipulation, serialized via
        // #[serial(rein_http_token_env)] — no parallel test can observe.
        unsafe {
            std::env::remove_var("REIN_HTTP_TOKEN");
        }

        let req: Request<String> = req_with(None);
        let result = enforce_auth_policy(&req, crate::ops::AuthPolicy::ReadToken);

        // Restore before asserting so a failing assert still leaves env clean.
        if let Some(v) = original {
            unsafe {
                std::env::set_var("REIN_HTTP_TOKEN", v);
            }
        }

        assert!(
            result.is_ok(),
            "read_token must be permissive when REIN_HTTP_TOKEN is unset"
        );
    }

    #[test]
    #[serial_test::serial(rein_http_token_env)]
    fn enforce_auth_read_token_accepts_bearer_and_cookie_auth() {
        let original = std::env::var("REIN_HTTP_TOKEN").ok();
        unsafe {
            std::env::set_var("REIN_HTTP_TOKEN", "secret-token");
        }

        let bearer_req: Request<String> = req_with(Some(("authorization", "Bearer secret-token")));
        assert!(
            enforce_auth_policy(&bearer_req, crate::ops::AuthPolicy::ReadToken).is_ok(),
            "Bearer auth should satisfy read_token policy"
        );

        let cookie_req = Request::builder()
            .uri("/api/x")
            .header("cookie", "rein_http_token=secret-token")
            .body(String::new())
            .unwrap();
        assert!(
            enforce_auth_policy(&cookie_req, crate::ops::AuthPolicy::ReadToken).is_ok(),
            "session cookie should satisfy read_token policy"
        );

        match original {
            Some(v) => unsafe {
                std::env::set_var("REIN_HTTP_TOKEN", v);
            },
            None => unsafe {
                std::env::remove_var("REIN_HTTP_TOKEN");
            },
        }
    }

    #[test]
    #[serial_test::serial(rein_http_token_env)]
    fn enforce_auth_read_token_rejects_wrong_length_values() {
        let original = std::env::var("REIN_HTTP_TOKEN").ok();
        unsafe {
            std::env::set_var("REIN_HTTP_TOKEN", "secret-token");
        }

        let short_req: Request<String> = req_with(Some(("x-rein-token", "secret")));
        assert!(
            enforce_auth_policy(&short_req, crate::ops::AuthPolicy::ReadToken).is_err(),
            "short x-rein-token must be rejected"
        );

        let long_bearer: Request<String> =
            req_with(Some(("authorization", "Bearer secret-token-longer")));
        assert!(
            enforce_auth_policy(&long_bearer, crate::ops::AuthPolicy::ReadToken).is_err(),
            "wrong-length bearer token must be rejected"
        );

        match original {
            Some(v) => unsafe {
                std::env::set_var("REIN_HTTP_TOKEN", v);
            },
            None => unsafe {
                std::env::remove_var("REIN_HTTP_TOKEN");
            },
        }
    }

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
            archival_summary: None,
            archival_summary_at: None,
            archival_summary_version: None,
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
    #[serial_test::serial(global_state)]
    async fn recall_stream_pages_results_without_changing_legacy_endpoint() {
        // Joins `global_state` so REIN_HTTP_TOKEN mutations from racing
        // auth tests can't make `require_read_token` reject these GETs
        // (which produces a 401 with no `count` field and panics the
        // assertion at "count == Some(2)").
        let _guard = env_lock().lock().await;
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

    /// v0.26 D direction (B_REST_MCP): `/api/adaptive` JSON response MUST
    /// include the `synthesis` key on a fresh DB (cold-start state). The GUI
    /// Adaptive page conditions its empty-state branch on the presence of
    /// this key — a regression here would silently break the synthesis
    /// observability surface even though the rest of `/api/adaptive` looks
    /// healthy.
    ///
    /// Cold-start contract per implementation-contract §4.3:
    ///     synthesis = { "by_cluster": [], "global": null }
    #[tokio::test]
    #[serial_test::serial(global_state)]
    async fn api_adaptive_response_includes_synthesis_cold_start_shape() {
        let _guard = env_lock().lock().await;
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("adaptive.db");
        let config = test_config(&db_path);
        // Touch the store so the schema migrations run before /api/adaptive
        // queries it. Without this, the inventory dispatcher would still
        // succeed (adaptive_status uses unwrap_or_default on missing rows)
        // but the test would also exercise the lazy-init path — keeping it
        // explicit makes the cold-start contract unambiguous.
        let _store = SqliteStore::new(&db_path, &config.embedding_model(), 3072).unwrap();

        let req = Request::builder()
            .method("GET")
            .uri("/api/adaptive")
            .body(())
            .unwrap();
        let response = handle_rest_request(&req, &config).await.unwrap();
        let json = read_json(response).await;

        // Existing keys preserved (smoke-check that we didn't accidentally
        // restructure the response body) — at least one well-known field
        // from the pre-v0.26 shape MUST still be present.
        assert!(
            json.get("learned_alphas").is_some(),
            "/api/adaptive must keep its pre-v0.26 keys (learned_alphas)"
        );
        assert!(
            json.get("cluster_info").is_some(),
            "/api/adaptive must keep its pre-v0.26 keys (cluster_info)"
        );

        // v0.26 surface: `synthesis` key with cold-start contract shape.
        let synthesis = json
            .get("synthesis")
            .expect("/api/adaptive response must include `synthesis` key");
        assert_eq!(
            synthesis,
            &serde_json::json!({
                "by_cluster": [],
                "global": null,
            }),
            "cold-start synthesis projection must match contract §4.3 exactly"
        );
    }

    #[tokio::test]
    #[serial_test::serial(global_state)]
    async fn artifacts_endpoints_expose_proxy_artifacts_with_transcript() {
        // Joins the global_state serial group so it can't race tests that mutate
        // REIN_HTTP_TOKEN (api_artifacts_auth_gate_matrix and friends).  When
        // REIN_HTTP_TOKEN was set by a racing test, require_read_token would
        // reject this GET with 401 and the `artifacts` field would be missing,
        // producing a bare `Option::unwrap()` panic at line 1473.
        let _guard = env_lock().lock().await;
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
    async fn api_artifacts_reports_page_metadata_for_offset_limit_clients() {
        let _guard = env_lock().lock().await;
        std::env::remove_var("REIN_HTTP_TOKEN");
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("artifacts.db");
        let config = test_config(&db_path);
        let store = SqliteStore::new(&db_path, &config.embedding_model(), 3072).unwrap();
        insert_artifact(&store, "artifact one", None);
        insert_artifact(&store, "artifact two", None);
        insert_artifact(&store, "artifact three", None);
        drop(store);

        let page1 = Request::builder()
            .method("GET")
            .uri("/api/artifacts?limit=2&offset=0")
            .body(())
            .unwrap();
        let response = handle_rest_request(&page1, &config).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json = read_json(response).await;
        assert_eq!(json["artifacts"].as_array().unwrap().len(), 2);
        assert_eq!(json["count"].as_u64(), Some(2));
        assert_eq!(json["offset"].as_u64(), Some(0));
        assert_eq!(json["limit"].as_u64(), Some(2));
        assert_eq!(json["next_offset"].as_u64(), Some(2));
        assert_eq!(json["has_more"].as_bool(), Some(true));

        let page2 = Request::builder()
            .method("GET")
            .uri("/api/artifacts?limit=2&offset=2")
            .body(())
            .unwrap();
        let response = handle_rest_request(&page2, &config).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json = read_json(response).await;
        assert_eq!(json["artifacts"].as_array().unwrap().len(), 1);
        assert_eq!(json["count"].as_u64(), Some(1));
        assert_eq!(json["offset"].as_u64(), Some(2));
        assert_eq!(json["limit"].as_u64(), Some(2));
        assert!(json["next_offset"].is_null());
        assert_eq!(json["has_more"].as_bool(), Some(false));
    }

    struct EnvVarGuard {
        key: &'static str,
        original: Option<std::ffi::OsString>,
    }

    impl EnvVarGuard {
        fn set_path(key: &'static str, value: &std::path::Path) -> Self {
            let original = std::env::var_os(key);
            unsafe {
                std::env::set_var(key, value);
            }
            Self { key, original }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.original {
                Some(value) => unsafe {
                    std::env::set_var(self.key, value);
                },
                None => unsafe {
                    std::env::remove_var(self.key);
                },
            }
        }
    }

    struct CurrentDirGuard {
        original: std::path::PathBuf,
    }

    impl CurrentDirGuard {
        fn change_to(path: &std::path::Path) -> Self {
            let original = std::env::current_dir().unwrap();
            std::env::set_current_dir(path).unwrap();
            Self { original }
        }
    }

    impl Drop for CurrentDirGuard {
        fn drop(&mut self) {
            std::env::set_current_dir(&self.original).unwrap();
        }
    }

    #[test]
    #[serial_test::serial(global_state)]
    fn recall_synthesis_adaptive_state_uses_resolved_auto_path() {
        let dir = tempdir().unwrap();
        let _home = EnvVarGuard::set_path("HOME", dir.path());

        let mut config = ReinConfig::default();
        config.database.path = "auto".to_string();
        config.embedding.provider = "none".to_string();
        config.query_expansion.provider = "none".to_string();
        config.sync.supermemory_enabled = false;
        config.sync.auto_memory_enabled = false;

        let resolved_db = config.resolve_db_path();
        assert_ne!(
            resolved_db,
            std::path::PathBuf::from("auto"),
            "test must exercise the auto sentinel rather than a literal path"
        );
        let store = SqliteStore::new(&resolved_db, &config.embedding_model(), 3072).unwrap();

        let query = "connection pool";
        let query_type = crate::search::classify::classify(query, false, false)
            .query_type
            .synthesis_bucket_label()
            .to_string();
        let mut by_cluster = std::collections::HashMap::new();
        by_cluster.insert(
            crate::store::adaptive::synthesis_bucket_key(Some(42), &query_type),
            crate::store::adaptive::ClusterSynthesisStats {
                viewed_count: 1,
                useful_rate: 0.0,
                ..Default::default()
            },
        );
        let state = crate::store::adaptive::AdaptiveState {
            synthesis_feedback_stats: Some(crate::store::adaptive::SynthesisFeedbackState {
                by_cluster,
                ..Default::default()
            }),
            ..Default::default()
        };
        state.save_snapshot(store.conn()).unwrap();
        drop(store);

        let loaded = {
            let _cwd = CurrentDirGuard::change_to(dir.path());
            recall_synthesis_adaptive_state(&config)
        }
        .expect("adaptive state must load from the resolved auto DB");
        let loaded_synthesis = loaded
            .synthesis_feedback_stats
            .expect("saved synthesis feedback state");
        assert!(
            loaded_synthesis.by_cluster.contains_key(
                &crate::store::adaptive::synthesis_bucket_key(Some(42), &query_type)
            ),
            "REST synthesis adaptive helper must not open a literal `auto` DB"
        );
        assert!(
            !dir.path().join("auto").exists(),
            "resolved auto helper must not create/read a literal `auto` DB"
        );
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

        let req_bearer = Request::builder()
            .method("GET")
            .uri("/api/artifacts")
            .header("authorization", "Bearer secret-token-xyz")
            .body(())
            .unwrap();
        assert_eq!(
            handle_rest_request(&req_bearer, &config2)
                .await
                .unwrap()
                .status(),
            StatusCode::OK
        );

        let req_cookie = Request::builder()
            .method("GET")
            .uri("/api/artifacts")
            .header("cookie", "rein_http_token=secret-token-xyz")
            .body(())
            .unwrap();
        assert_eq!(
            handle_rest_request(&req_cookie, &config2)
                .await
                .unwrap()
                .status(),
            StatusCode::OK
        );

        std::env::remove_var("REIN_HTTP_TOKEN");
    }

    #[tokio::test]
    async fn memoir_inspect_missing_concept_returns_404() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("memoir-inspect.db");
        let config = test_config(&db_path);
        let _store = SqliteStore::new(&db_path, &config.embedding_model(), 3072).unwrap();

        let req = Request::builder()
            .method("GET")
            .uri("/api/memoirs/missing/inspect/concept")
            .body(())
            .unwrap();
        let response = handle_rest_request(&req, &config).await.unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
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

    // ---- Phase 2.2 H3/H5 real-consumer parity matrix ----
    //
    // Exercises the full auth + body-JSON decode path for the two ops
    // migrated in the session batch (rein_ingest_session and doctor_fix).
    // Earlier phases proved the framework via unit tests on
    // `enforce_auth_policy`; these are the first E2E checks asserting the
    // declared `auth = "mutation_marker"` actually rejects forbidden
    // requests at the inventory dispatch layer and the H5 JSON body is
    // decoded (not the empty query string) for POSTs.

    fn build_post(uri: &str, with_action: bool, body: &[u8]) -> Request<()> {
        let mut b = Request::builder()
            .method(Method::POST)
            .uri(uri)
            .header("content-type", "application/json");
        if with_action {
            b = b.header("x-rein-action", "1");
        }
        // The handle_rest_request_with_body contract ignores req body type
        // (body bytes come through the `body` argument), so `()` is fine.
        let _ = body; // bytes are passed separately; silence warning.
        b.body(()).unwrap()
    }

    #[tokio::test]
    async fn ingest_session_rejects_missing_mutation_marker() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("memories.db");
        let config = test_config(&db_path);
        let body = Bytes::from(r#"{"content":"hello"}"#);
        let req = build_post("/api/ingest_session", false, &body);
        let resp = handle_rest_request_with_body(&req, Some(body), &config)
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn ingest_session_rejects_malformed_json_body() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("memories.db");
        let config = test_config(&db_path);
        // Missing closing brace → serde_json reports parse error → 400
        // via the macro's `.with_kind(OpsErrorKind::BadRequest)` mapping.
        let body = Bytes::from(r#"{"content":"hello"#);
        let req = build_post("/api/ingest_session", true, &body);
        let resp = handle_rest_request_with_body(&req, Some(body), &config)
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn ingest_session_rejects_empty_payload_params() {
        // Payload has neither `content` nor `turns` — the op itself tags
        // this BadRequest, going through the same H4 kind-plumbing path as
        // macro-emitted parse errors. Distinct code path from JSON malform.
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("memories.db");
        let config = test_config(&db_path);
        let body = Bytes::from(r#"{}"#);
        let req = build_post("/api/ingest_session", true, &body);
        let resp = handle_rest_request_with_body(&req, Some(body), &config)
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json = read_json(resp).await;
        assert!(
            json["error"].as_str().unwrap_or("").contains("content"),
            "error should mention missing content/turns: {}",
            json,
        );
    }

    #[tokio::test]
    #[serial_test::serial(global_state)]
    async fn ingest_session_accepts_valid_content_body() {
        // Lightweight success: a 20-char payload is well under the 500 KB
        // ceiling. Joins `global_state` since extraction may touch shared
        // caches in parallel with other artifact tests.
        let _guard = env_lock().lock().await;
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("memories.db");
        let config = test_config(&db_path);
        let body = Bytes::from(r#"{"content":"hello world from h5 parity test"}"#);
        let req = build_post("/api/ingest_session", true, &body);
        let resp = handle_rest_request_with_body(&req, Some(body), &config)
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "valid ingest body must return 200"
        );
        let json = read_json(resp).await;
        // Successful ingest always emits a queued flag; artifact_id and
        // episode_id may be null when extraction skips (e.g. short input).
        assert!(json.get("queued").is_some(), "missing queued: {}", json);
    }

    #[tokio::test]
    async fn doctor_fix_rejects_missing_mutation_marker() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("memories.db");
        let config = test_config(&db_path);
        let body = Bytes::from(r#"{}"#);
        let req = build_post("/api/doctor", false, &body);
        let resp = handle_rest_request_with_body(&req, Some(body), &config)
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn doctor_fix_rejects_malformed_body() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("memories.db");
        let config = test_config(&db_path);
        let body = Bytes::from(r#"{"network":"not-a-bool""#); // unterminated
        let req = build_post("/api/doctor", true, &body);
        let resp = handle_rest_request_with_body(&req, Some(body), &config)
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    #[serial_test::serial(global_state)]
    async fn doctor_fix_accepts_empty_object_body() {
        // Empty {} is the GUI's canonical POST body — all DoctorFixParams
        // fields default to false, fix is implied true in the op. Success
        // path returns a DoctorReport-shaped JSON.
        let _guard = env_lock().lock().await;
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("memories.db");
        let config = test_config(&db_path);
        let body = Bytes::from(r#"{}"#);
        let req = build_post("/api/doctor", true, &body);
        let resp = handle_rest_request_with_body(&req, Some(body), &config)
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = read_json(resp).await;
        // DoctorReport has a `checks` array; presence is enough to confirm
        // shape — we don't want to assert pass/fail because doctor runs
        // real diagnostics against the tempdir store.
        assert!(json.get("checks").is_some(), "missing checks: {}", json);
    }

    #[tokio::test]
    async fn handle_api_request_collects_body_and_dispatches() {
        // Full-path smoke: construct a real `Request<Full<Bytes>>` (the same
        // concrete body shape the hyper server receives) and route through
        // the body-aware entry point. Closes the gap between the ref-based
        // `handle_rest_request_with_body` tests (which pre-collect body) and
        // the production glue in server.rs that owns the body-read step.
        use http_body_util::Full;
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("memories.db");
        let config = test_config(&db_path);
        let body_bytes = Bytes::from(r#"{}"#);
        let req = Request::builder()
            .method(Method::POST)
            .uri("/api/doctor")
            .header("content-type", "application/json")
            .header("x-rein-action", "1")
            .body(Full::new(body_bytes))
            .unwrap();
        let resp = handle_api_request(req, &config).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn handle_api_request_rejects_missing_action_header() {
        // Same as above but without `x-rein-action: 1` — full path still
        // rejects at the H3 enforcement layer, proving body isn't read before
        // auth (a subtle race would expose auth info via body-size checks).
        use http_body_util::Full;
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("memories.db");
        let config = test_config(&db_path);
        let req = Request::builder()
            .method(Method::POST)
            .uri("/api/doctor")
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(r#"{}"#)))
            .unwrap();
        let resp = handle_api_request(req, &config).await;
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn handle_api_request_auth_rejects_before_body_cap() {
        use http_body_util::Full;
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("memories.db");
        let config = test_config(&db_path);
        let req = Request::builder()
            .method(Method::POST)
            .uri("/api/doctor")
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(vec![b'a'; 2 * 1024 * 1024])))
            .unwrap();
        let resp = handle_api_request(req, &config).await;
        assert_eq!(
            resp.status(),
            StatusCode::FORBIDDEN,
            "auth rejection should happen before the body-size gate on protected POST routes",
        );
    }

    #[tokio::test]
    async fn handle_api_request_returns_404_for_unknown_route() {
        use http_body_util::Full;
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("memories.db");
        let config = test_config(&db_path);
        let req = Request::builder()
            .method(Method::GET)
            .uri("/api/does_not_exist")
            .body(Full::new(Bytes::new()))
            .unwrap();
        let resp = handle_api_request(req, &config).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn collect_body_capped_rejects_oversize() {
        // Sanity-check the body-cap guard independently of any specific op.
        // Default cap is 1 MiB; construct a 2 MiB Full<Bytes> body and
        // confirm the capped reader rejects it with 413.
        use http_body_util::Full;
        let big = Full::new(Bytes::from(vec![b'a'; 2 * 1024 * 1024]));
        let result = collect_body_capped(big).await;
        let resp = result.expect_err("2 MiB body must exceed 1 MiB cap");
        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    /// Streaming body with no advertised upper bound — simulates a chunked
    /// transfer where the client lies about (or doesn't declare) size.
    /// `poll_frame` yields one 200 KiB chunk at a time until aborted.
    struct StreamingBody {
        remaining_chunks: usize,
    }

    impl hyper::body::Body for StreamingBody {
        type Data = Bytes;
        type Error = std::convert::Infallible;

        fn poll_frame(
            mut self: std::pin::Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Option<Result<hyper::body::Frame<Bytes>, Self::Error>>> {
            if self.remaining_chunks == 0 {
                std::task::Poll::Ready(None)
            } else {
                self.remaining_chunks -= 1;
                std::task::Poll::Ready(Some(Ok(hyper::body::Frame::data(Bytes::from(
                    vec![b'x'; 200 * 1024],
                )))))
            }
        }

        fn size_hint(&self) -> hyper::body::SizeHint {
            // Deliberately claim no upper bound — this is the pre-fix
            // pathological case where collect_body_capped's old
            // `.collect().await + post-len-check` would buffer everything
            // before tripping the cap.
            hyper::body::SizeHint::new()
        }
    }

    #[tokio::test]
    async fn collect_body_capped_progressive_on_streaming_body() {
        // 6 chunks of 200 KiB = 1200 KiB, above the 1 MiB default cap.
        // With the progressive frame-loop, cap fires mid-stream; with the
        // old implementation this test was the missing regression signal.
        let body = StreamingBody {
            remaining_chunks: 6,
        };
        let result = collect_body_capped(body).await;
        let resp = result.expect_err("streaming body past cap must 413");
        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn collect_body_capped_streaming_under_cap() {
        // 4 chunks of 200 KiB = 800 KiB, well under the 1 MiB cap.
        // Confirms the progressive loop drains the full stream and returns
        // the assembled buffer — not just the cap-rejection path.
        let body = StreamingBody {
            remaining_chunks: 4,
        };
        let result = collect_body_capped(body).await;
        let bytes = result.expect("streaming body under cap must succeed");
        assert_eq!(bytes.len(), 4 * 200 * 1024);
    }

    #[tokio::test]
    async fn doctor_fix_honors_explicit_fix_false() {
        // Pre-migration `POST /api/doctor?fix=false&network=1` ran the
        // diagnostic without applying repairs. Post-migration this lives
        // in the DoctorFixParams JSON body. Without the Option<bool>-ish
        // default, every POST would force fix=true and silently repair.
        // We can't easily assert "didn't repair" at this layer (the
        // doctor report doesn't tag fixes-applied), but we can confirm
        // the request goes through (200) with fix explicitly false.
        use http_body_util::Full;
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("memories.db");
        let config = test_config(&db_path);
        let body = Bytes::from(r#"{"fix":false,"network":false}"#);
        let req = Request::builder()
            .method(Method::POST)
            .uri("/api/doctor")
            .header("content-type", "application/json")
            .header("x-rein-action", "1")
            .body(Full::new(body))
            .unwrap();
        let resp = handle_api_request(req, &config).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    // ---- Phase 2.3 M3: auth/body-cap regression matrix for new POST routes ----
    //
    // Phase 2.3 added six new POST REST routes to the inventory dispatcher.
    // All declare `auth = "mutation_marker"` and are gated by the shared
    // body-cap middleware. The two tests below form a table-driven matrix that
    // asserts both invariants hold for every route, so a future refactor that
    // accidentally bypasses H3 auth or the body-cap for a specific inventory
    // route is caught immediately.

    /// POST routes migrated to the #[op] inventory and declared with
    /// `auth = "mutation_marker"`. Phase 2.3 added the first six; Phase 2.4
    /// adds `/api/feedback` so the auth + body-cap matrix covers it too.
    const NEW_POST_ROUTES: &[&str] = &[
        "/api/gc",
        "/api/dedup",
        "/api/dedup_concepts",
        "/api/organize",
        "/api/consolidate",
        "/api/cleanup",
        "/api/feedback",
    ];

    #[tokio::test]
    async fn new_post_routes_reject_missing_mutation_marker() {
        // Mirrors `ingest_session_rejects_missing_mutation_marker`: build_post
        // with `with_action = false`, call handle_rest_request_with_body, expect 403.
        // Auth runs before body collection so body content is irrelevant here.
        for route in NEW_POST_ROUTES {
            let dir = tempdir().unwrap();
            let db_path = dir.path().join("memories.db");
            let config = test_config(&db_path);
            let body = Bytes::from(r#"{}"#);
            let req = build_post(route, false, &body);
            let resp = handle_rest_request_with_body(&req, Some(body), &config)
                .await
                .unwrap();
            assert_eq!(
                resp.status(),
                StatusCode::FORBIDDEN,
                "{route} should 403 without x-rein-action: 1"
            );
        }
    }

    #[tokio::test]
    async fn new_post_routes_reject_oversized_body() {
        // Mirror image of `handle_api_request_auth_rejects_before_body_cap`
        // (which proves missing marker → 403 even with a 2 MiB body). Here
        // the marker IS present so auth passes; the body-cap gate fires next
        // and must return 413. Uses handle_api_request (Full<Bytes>) so that
        // collect_body_capped is exercised — handle_rest_request_with_body
        // takes pre-collected bytes and would bypass the cap entirely.
        use http_body_util::Full;
        let oversized = vec![b'a'; 2 * 1024 * 1024];
        for route in NEW_POST_ROUTES {
            let dir = tempdir().unwrap();
            let db_path = dir.path().join("memories.db");
            let config = test_config(&db_path);
            let req = Request::builder()
                .method(Method::POST)
                .uri(*route)
                .header("content-type", "application/json")
                .header("x-rein-action", "1")
                .body(Full::new(Bytes::from(oversized.clone())))
                .unwrap();
            let resp = handle_api_request(req, &config).await;
            assert_eq!(
                resp.status(),
                StatusCode::PAYLOAD_TOO_LARGE,
                "{route} should 413 for oversized body"
            );
        }
    }

    // ---- Phase 2.4 F1: feedback-specific E2E REST tests ----
    //
    // The Phase 2.3 M3 matrix above now covers /api/feedback for auth-marker
    // and body-cap. This test exercises the semantic validation path
    // (empty memory_ids → OpsErrorKind::BadRequest → 400) through the full
    // handle_rest_request_with_body → inventory dispatch path so the
    // BadRequest mapping is verified end-to-end for this op.

    #[tokio::test]
    async fn feedback_empty_memory_ids_returns_400() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("memories.db");
        let config = test_config(&db_path);
        let body = Bytes::from(r#"{"memory_ids":[]}"#);
        let req = build_post("/api/feedback", true, &body);
        let resp = handle_rest_request_with_body(&req, Some(body), &config)
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "/api/feedback with empty memory_ids should 400"
        );
    }

    /// Regression: `/api/memories` and `/api/recall_stream` must expose the
    /// server-generated `request_id` so clients can correlate later
    /// `POST /api/feedback` calls back to the originating recall. Previously
    /// the REST endpoint called `recall_temporal` (no request_id) which
    /// broke the M1 feedback→replay chain for every non-MCP caller.
    #[tokio::test]
    #[serial_test::serial(global_state)]
    async fn recall_response_includes_request_id_and_matches_feedback_event() {
        let _guard = env_lock().lock().await;
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("memories.db");
        let config = test_config(&db_path);
        let store = SqliteStore::new(&db_path, &config.embedding_model(), 3072).unwrap();
        store
            .store(test_memory(
                "debug",
                "pool fix",
                "connection pool saturation",
            ))
            .unwrap();

        let req = Request::builder()
            .method("GET")
            .uri("/api/memories?q=connection+pool&limit=5")
            .body(())
            .unwrap();
        let resp = handle_rest_request(&req, &config).await.unwrap();
        let json = read_json(resp).await;
        let request_id = json["request_id"]
            .as_str()
            .expect("recall response must include request_id")
            .to_string();
        assert!(!request_id.is_empty());

        // Feedback event must have been written with the same request_id.
        let store = SqliteStore::new(&db_path, &config.embedding_model(), 3072).unwrap();
        let logged: String = store
            .conn()
            .query_row(
                "SELECT request_id FROM feedback_events \
                  WHERE event_type = 'recall_complete' \
                  ORDER BY id DESC LIMIT 1",
                [],
                |r| r.get(0),
            )
            .expect("recall_complete event");
        assert_eq!(logged, request_id, "REST must propagate its request_id");

        // Streaming endpoint has the same contract.
        let req = Request::builder()
            .method("GET")
            .uri("/api/recall_stream?q=connection+pool&limit=5&offset=0")
            .body(())
            .unwrap();
        let resp = handle_rest_request(&req, &config).await.unwrap();
        let json = read_json(resp).await;
        assert!(
            json["request_id"].as_str().is_some(),
            "stream response must include request_id: {}",
            json
        );
    }

    /// v0.26.2 fix (Bug #O3): `recall_result_to_json` previously dropped
    /// `archival_summary` because the function hand-serializes sibling
    /// fields, bypassing the struct's `#[serde(skip_serializing_if)]` for
    /// the field. The REST recall path is the surface the Neural Wiki GUI
    /// consumes; without this insert, cold-tier memories never surfaced
    /// their condensed archival summary even when Cap C populated the
    /// field upstream.
    #[test]
    fn recall_result_to_json_surfaces_archival_summary() {
        let mut memory = test_memory("m1", "topic", "content");
        memory.tier = MemoryTier::Cold;
        memory.archival_summary = Some("condensed summary".to_string());
        let result = crate::search::recall::RecallResult {
            memory,
            score: 0.42,
            confidence: 0.7,
            sources_hit: 1,
            evidence_count: 0,
            evidence_preview: vec![],
            archival_summary: Some("condensed summary".to_string()),
        };
        let json = recall_result_to_json(&result);
        assert_eq!(
            json.get("archival_summary").and_then(|v| v.as_str()),
            Some("condensed summary"),
            "archival_summary MUST appear on the wire so GUI Cap C surfaces work; got {json}"
        );
    }

    /// v0.26.2 R4 F2: `recall_result_to_json` OMITS the key when None,
    /// matching the `RecallResult` struct's
    /// `#[serde(skip_serializing_if = "Option::is_none")]` and the TS
    /// type `archival_summary?: string`. Earlier R1 fix emitted JSON
    /// null which would break clients that branch on property presence.
    #[test]
    fn recall_result_to_json_omits_archival_summary_when_none() {
        let result = crate::search::recall::RecallResult {
            memory: test_memory("m1", "topic", "content"),
            score: 0.42,
            confidence: 0.7,
            sources_hit: 1,
            evidence_count: 0,
            evidence_preview: vec![],
            archival_summary: None,
        };
        let json = recall_result_to_json(&result);
        assert!(
            json.get("archival_summary").is_none(),
            "archival_summary key MUST be omitted when None to match the \
             optional TS type contract; got {json}"
        );
    }
}
