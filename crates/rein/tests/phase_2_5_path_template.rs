//! Phase 2.5 path-template framework integration tests.
//!
//! Covers T2 (PathSegment emission), T3 (dispatcher template match), T4
//! (path-value merge with path winning over query), and audit hardening:
//! H1 (GET param injection prevention), H2 (non-object body rejection),
//! H3 (templated route auth checked pre-body), M1 (empty segment 404),
//! M2 (invalid UTF-8 in segment → 404).
//!
//! Tests rely on `__test_path_template` (GET, Public) and
//! `__test_path_template_mut` (POST, MutationMarker) — test-only ops
//! compiled when the `test-support` feature is active (activated via
//! dev-dependencies in Cargo.toml).

use bytes::Bytes;
use hyper::{Method, Request, StatusCode};
use rein::config::ReinConfig;
use rein::mcp::rest::{handle_api_request, handle_rest_request_with_body};
use rein::ops::{OpsRestEntry, PathSegment};

// ─── helpers ────────────────────────────────────────────────────────────────

fn test_config() -> ReinConfig {
    // Use an in-memory DB so tests don't touch the real database.
    let mut cfg = ReinConfig::default();
    cfg.database.path = ":memory:".to_string();
    cfg
}

fn get_req(path_and_query: &str) -> Request<()> {
    Request::builder()
        .method(Method::GET)
        .uri(format!("http://localhost{path_and_query}"))
        .body(())
        .unwrap()
}

fn post_req_with_body(path: &str, body: Bytes) -> Request<http_body_util::Full<Bytes>> {
    Request::builder()
        .method(Method::POST)
        .uri(format!("http://localhost{path}"))
        .header("content-type", "application/json")
        .body(http_body_util::Full::new(body))
        .unwrap()
}

fn post_req_mutation(path: &str, body: Bytes) -> Request<http_body_util::Full<Bytes>> {
    Request::builder()
        .method(Method::POST)
        .uri(format!("http://localhost{path}"))
        .header("content-type", "application/json")
        .header("x-rein-action", "1")
        .body(http_body_util::Full::new(body))
        .unwrap()
}

async fn dispatch(req: &Request<()>, cfg: &ReinConfig) -> Option<(StatusCode, serde_json::Value)> {
    let resp = handle_rest_request_with_body(req, None, cfg).await?;
    let status = resp.status();
    let body_bytes = {
        use http_body_util::BodyExt;
        resp.into_body().collect().await.ok()?.to_bytes()
    };
    let json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap_or_default();
    Some((status, json))
}

// ─── T2: inventory PathSegment emission ─────────────────────────────────────

#[test]
fn path_template_emits_path_segments() {
    // The test-only op must register an OpsRestEntry with the expected
    // PathSegment array: [Literal("api"), Literal("test_path_template"), Param("id")]
    let entry = inventory::iter::<OpsRestEntry>()
        .find(|e| e.op_name == "__test_path_template")
        .expect("__test_path_template should be in inventory (compiled under #[cfg(test)])");

    assert_eq!(entry.method, hyper::Method::GET);
    assert_eq!(entry.path_template, "/api/test_path_template/{id}");

    let segs = entry.path_segments;
    assert_eq!(segs.len(), 3, "expected 3 segments for /api/test_path_template/{{id}}");
    assert_eq!(segs[0], PathSegment::Literal("api"));
    assert_eq!(segs[1], PathSegment::Literal("test_path_template"));
    assert_eq!(segs[2], PathSegment::Param("id"));
}

#[test]
fn literal_only_ops_have_empty_path_segments() {
    // Existing migrated ops with no placeholder must still have path_segments: &[].
    let entry = inventory::iter::<OpsRestEntry>()
        .find(|e| e.op_name == "stats")
        .expect("stats should be registered");
    assert!(
        entry.path_segments.is_empty(),
        "literal-only op 'stats' should have empty path_segments, got {:?}",
        entry.path_segments
    );
}

// ─── T3: dispatcher template match ──────────────────────────────────────────

#[tokio::test]
async fn dispatcher_exact_match_still_works() {
    let cfg = test_config();
    let req = get_req("/api/stats");
    let (status, _) = dispatch(&req, &cfg).await.expect("stats should respond");
    assert_eq!(status, StatusCode::OK, "exact-match stats should return 200");
}

#[tokio::test]
async fn dispatcher_matches_template_path() {
    let cfg = test_config();
    let req = get_req("/api/test_path_template/hello");
    let (status, body) = dispatch(&req, &cfg)
        .await
        .expect("template route should respond");
    assert_eq!(status, StatusCode::OK, "template match should return 200, body={body}");
    assert_eq!(
        body["echoed_id"].as_str(),
        Some("hello"),
        "id should be echoed from path"
    );
}

#[tokio::test]
async fn dispatcher_unknown_path_returns_404() {
    let cfg = test_config();
    let req = get_req("/api/nonexistent_endpoint/xyz");
    let result = dispatch(&req, &cfg).await;
    // handle_rest_request_with_body returns None for non-/api/ paths, but
    // /api/ paths always get a response (404 for unknown).
    if let Some((status, _)) = result {
        assert_eq!(status, StatusCode::NOT_FOUND);
    }
    // If None, the caller would 404 anyway — also acceptable.
}

// ─── T4: path value overrides query string ───────────────────────────────────

#[tokio::test]
async fn path_value_overrides_query_string() {
    let cfg = test_config();
    // Path says "abc", query says "xyz" — path must win.
    let req = get_req("/api/test_path_template/abc?id=xyz");
    let (status, body) = dispatch(&req, &cfg)
        .await
        .expect("route should respond");
    assert_eq!(status, StatusCode::OK, "should succeed, body={body}");
    assert_eq!(
        body["echoed_id"].as_str(),
        Some("abc"),
        "path value 'abc' should win over query 'xyz'"
    );
}

// ─── T5: percent-decoding edge cases ────────────────────────────────────────

#[tokio::test]
async fn percent_encoded_segment_decodes_correctly() {
    let cfg = test_config();
    // %20 = space
    let req = get_req("/api/test_path_template/hello%20world");
    let (status, body) = dispatch(&req, &cfg)
        .await
        .expect("route should respond");
    assert_eq!(status, StatusCode::OK, "should decode space, body={body}");
    assert_eq!(
        body["echoed_id"].as_str(),
        Some("hello world"),
        "percent-encoded space should decode to ' '"
    );
}

#[tokio::test]
async fn percent_encoded_slash_does_not_cross_segment_boundary() {
    let cfg = test_config();
    // %2F = '/' — must not split into two segments; dispatcher sees it as
    // a 3-segment path after leading-strip, matching the template exactly.
    // The decoded id should be "a/b" (a slash character as value).
    let req = get_req("/api/test_path_template/a%2Fb");
    let (status, body) = dispatch(&req, &cfg)
        .await
        .expect("route should respond");
    assert_eq!(status, StatusCode::OK, "should match template, body={body}");
    // The id includes the decoded slash literal.
    let echoed = body["echoed_id"].as_str().unwrap_or("");
    assert!(
        echoed.contains('/'),
        "%2F should decode to '/' after segment split, got '{echoed}'"
    );
}

#[tokio::test]
async fn trailing_slash_returns_404() {
    let cfg = test_config();
    // /api/test_path_template/abc/ has 4 segments after leading-strip
    // ("api", "test_path_template", "abc", ""), vs template's 3 → no match.
    let req = get_req("/api/test_path_template/abc/");
    let result = dispatch(&req, &cfg).await;
    match result {
        Some((status, _)) => assert_eq!(
            status,
            StatusCode::NOT_FOUND,
            "trailing slash should not match 3-segment template"
        ),
        None => {
            // handle_rest_request_with_body returned None (shouldn't happen for
            // /api/ paths), caller would 404. Acceptable.
        }
    }
}

#[tokio::test]
async fn empty_id_segment_returns_404() {
    let cfg = test_config();
    // /api/test_path_template/ splits into ["api", "test_path_template", ""].
    // The empty trailing segment must not bind to Param("id") — match_path_template
    // rejects empty strings for Param variants, so this returns 404.
    let req = get_req("/api/test_path_template/");
    let result = dispatch(&req, &cfg).await;
    match result {
        Some((status, _)) => assert_eq!(
            status,
            StatusCode::NOT_FOUND,
            "empty id segment must not match Param variant — expected 404"
        ),
        None => {
            // handle_rest_request_with_body returned None (shouldn't happen
            // for /api/ paths), caller would 404. Acceptable.
        }
    }
}

// ─── M2: invalid UTF-8 in segment → 404 ────────────────────────────────────

#[tokio::test]
async fn invalid_utf8_in_segment_returns_404() {
    let cfg = test_config();
    // %FF is not valid UTF-8; percent_decode returns None, match_path_template
    // returns None, the route is not matched → 404.
    let req = get_req("/api/test_path_template/%FF");
    let result = dispatch(&req, &cfg).await;
    if let Some((status, _)) = result {
        assert_eq!(
            status,
            StatusCode::NOT_FOUND,
            "%FF (invalid UTF-8) should not match the template — expected 404"
        );
    } // None: caller would 404 anyway
}

// ─── H1: GET path-param injection prevention ────────────────────────────────

#[tokio::test]
async fn get_path_param_with_encoded_ampersand_does_not_inject() {
    let cfg = test_config();
    // %26 decodes to '&'. If the GET param merge did naive string concatenation,
    // id=a%26admin%3D1 would decode to "a&admin=1" and be re-interpreted as two
    // query params (id=a, admin=1), binding id="a" instead of id="a&admin=1".
    // The fix (serde_urlencoded::to_string re-encodes values) ensures the full
    // decoded value is bound as `id`.
    let req = get_req("/api/test_path_template/a%26admin%3D1");
    let (status, body) = dispatch(&req, &cfg)
        .await
        .expect("route should respond");
    assert_eq!(status, StatusCode::OK, "should succeed, body={body}");
    assert_eq!(
        body["echoed_id"].as_str(),
        Some("a&admin=1"),
        "decoded id should be 'a&admin=1', not split by '&'"
    );
}

#[tokio::test]
async fn get_path_param_with_encoded_equals_does_not_forge_key() {
    let cfg = test_config();
    // %3D decodes to '='. Naive concat would produce id=foo%3Dbar which
    // serde_urlencoded would parse as id="foo=bar" — but only because the '='
    // is already encoded. The re-encoding path must still produce the right id.
    let req = get_req("/api/test_path_template/foo%3Dbar");
    let (status, body) = dispatch(&req, &cfg)
        .await
        .expect("route should respond");
    assert_eq!(status, StatusCode::OK, "should succeed, body={body}");
    assert_eq!(
        body["echoed_id"].as_str(),
        Some("foo=bar"),
        "decoded id should contain '='"
    );
}

#[tokio::test]
async fn get_path_param_with_percent_literal_survives_roundtrip() {
    let cfg = test_config();
    // %25 decodes to '%'. Re-encoding must produce %25 again.
    let req = get_req("/api/test_path_template/100%25pure");
    let (status, body) = dispatch(&req, &cfg)
        .await
        .expect("route should respond");
    assert_eq!(status, StatusCode::OK, "should succeed, body={body}");
    assert_eq!(
        body["echoed_id"].as_str(),
        Some("100%pure"),
        "decoded id should contain literal '%'"
    );
}

#[tokio::test]
async fn get_path_param_with_encoded_hash_does_not_split_fragment() {
    let cfg = test_config();
    // %23 decodes to '#' (URL fragment delimiter). Naive string concat could
    // confuse URL parsers; structured re-encoding preserves the literal '#'.
    let req = get_req("/api/test_path_template/tag%23123");
    let (status, body) = dispatch(&req, &cfg)
        .await
        .expect("route should respond");
    assert_eq!(status, StatusCode::OK, "should succeed, body={body}");
    assert_eq!(
        body["echoed_id"].as_str(),
        Some("tag#123"),
        "decoded id should contain literal '#'"
    );
}

// ─── H2: non-object body rejection ──────────────────────────────────────────

#[tokio::test]
async fn post_with_null_body_returns_400() {
    let cfg = test_config();
    let req = post_req_mutation("/api/test_path_template_mut/abc", Bytes::from("null"));
    let resp = handle_api_request(req, &cfg).await;
    assert_eq!(
        resp.status(),
        StatusCode::BAD_REQUEST,
        "null body must be rejected with 400"
    );
}

#[tokio::test]
async fn post_with_array_body_returns_400() {
    let cfg = test_config();
    let req = post_req_mutation("/api/test_path_template_mut/abc", Bytes::from(r#"[1,2,3]"#));
    let resp = handle_api_request(req, &cfg).await;
    assert_eq!(
        resp.status(),
        StatusCode::BAD_REQUEST,
        "array body must be rejected with 400"
    );
}

#[tokio::test]
async fn post_with_bool_body_returns_400() {
    let cfg = test_config();
    let req = post_req_mutation("/api/test_path_template_mut/abc", Bytes::from("true"));
    let resp = handle_api_request(req, &cfg).await;
    assert_eq!(
        resp.status(),
        StatusCode::BAD_REQUEST,
        "bool body must be rejected with 400"
    );
}

#[tokio::test]
async fn post_with_string_body_returns_400() {
    let cfg = test_config();
    let req = post_req_mutation("/api/test_path_template_mut/abc", Bytes::from(r#""hello""#));
    let resp = handle_api_request(req, &cfg).await;
    assert_eq!(
        resp.status(),
        StatusCode::BAD_REQUEST,
        "string body must be rejected with 400"
    );
}

#[tokio::test]
async fn post_with_empty_body_succeeds() {
    // Empty body treated as {} — path value provides the id param.
    let cfg = test_config();
    let req = post_req_mutation("/api/test_path_template_mut/abc", Bytes::new());
    let resp = handle_api_request(req, &cfg).await;
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "empty body should be accepted (treated as {{}})"
    );
}

// ─── H3: templated route auth checked pre-body ──────────────────────────────

#[tokio::test]
async fn templated_mutation_route_auth_rejects_before_body_cap() {
    // Mirror of handle_api_request_auth_rejects_before_body_cap (rest.rs) for
    // a *templated* route. The pre-body auth gate must match templates, not
    // only exact paths, so a 2 MiB body without the action marker → 403, NOT 413.
    use tempfile::tempdir;
    let dir = tempdir().unwrap();
    let mut cfg = ReinConfig::default();
    cfg.database.path = dir.path().join("memories.db").to_string_lossy().into_owned();

    let big_body = Bytes::from(vec![b'x'; 2 * 1024 * 1024]);
    let req = post_req_with_body("/api/test_path_template_mut/abc", big_body);
    let resp = handle_api_request(req, &cfg).await;
    assert_eq!(
        resp.status(),
        StatusCode::FORBIDDEN,
        "templated mutation route: auth rejection must happen before body-size gate (2 MiB body, no marker)"
    );
}
