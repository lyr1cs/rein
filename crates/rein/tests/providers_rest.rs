use bytes::Bytes;
use hyper::{Method, Request, StatusCode};
use rein::config::ReinConfig;
use rein::mcp::rest::{handle_api_request, BoxedResponse};
use serial_test::serial;

fn cfg() -> ReinConfig {
    let mut cfg = ReinConfig::default();
    cfg.database.path = ":memory:".into();
    cfg.secrets.insert(
        "GEMINI_API_KEY".into(),
        "AIzaSyTESTKEY00000000003f2a".into(),
    );
    cfg
}

fn post(
    path: &str,
    body: &str,
    marker: bool,
    owner: Option<&str>,
) -> Request<http_body_util::Full<Bytes>> {
    let mut b = Request::builder()
        .method(Method::POST)
        .uri(format!("http://localhost{path}"))
        .header("content-type", "application/json");
    if marker {
        b = b.header("x-rein-action", "1");
    }
    if let Some(t) = owner {
        b = b.header("x-rein-token", t);
    }
    b.body(http_body_util::Full::new(Bytes::from(body.to_string())))
        .unwrap()
}

async fn body_json(resp: BoxedResponse) -> (StatusCode, serde_json::Value, Option<String>) {
    use http_body_util::BodyExt;
    let status = resp.status();
    let cache = resp
        .headers()
        .get("cache-control")
        .map(|v| v.to_str().unwrap().to_string());
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or_default(),
        cache,
    )
}

#[tokio::test]
#[serial]
async fn providers_get_never_contains_plaintext() {
    std::env::remove_var("REIN_HTTP_TOKEN");
    std::env::remove_var("GEMINI_API_KEY");
    let req = Request::builder()
        .method(Method::GET)
        .uri("http://localhost/api/providers")
        .body(http_body_util::Full::new(Bytes::new()))
        .unwrap();
    let (status, json, _) = body_json(handle_api_request(req, &cfg()).await).await;
    assert_eq!(status, StatusCode::OK);
    let text = json.to_string();
    assert!(text.contains("AIza••••3f2a"), "{text}");
    assert!(!text.contains("AIzaSyTESTKEY"));
}

#[tokio::test]
#[serial]
async fn reveal_requires_marker_and_owner_token() {
    std::env::set_var("REIN_HTTP_TOKEN", "owner-tok");
    std::env::remove_var("GEMINI_API_KEY");
    let c = cfg();
    let body = r#"{"credential":"env:GEMINI_API_KEY"}"#;
    let (s, _, _) = body_json(
        handle_api_request(
            post("/api/providers/reveal", body, false, Some("owner-tok")),
            &c,
        )
        .await,
    )
    .await;
    assert_eq!(s, StatusCode::FORBIDDEN);
    let (s, _, _) =
        body_json(handle_api_request(post("/api/providers/reveal", body, true, None), &c).await)
            .await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);
    let (s, _, _) = body_json(
        handle_api_request(post("/api/providers/reveal", body, true, Some("wrong")), &c).await,
    )
    .await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);
    let (s, json, cache) = body_json(
        handle_api_request(
            post("/api/providers/reveal", body, true, Some("owner-tok")),
            &c,
        )
        .await,
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(json["id"], "env:GEMINI_API_KEY");
    assert_eq!(json["value"], "AIzaSyTESTKEY00000000003f2a");
    assert_eq!(cache.as_deref(), Some("no-store"));
    std::env::remove_var("REIN_HTTP_TOKEN");
}

#[tokio::test]
#[serial]
async fn reveal_unknown_or_endpoint_ids_are_404_and_public_policy_is_401() {
    std::env::set_var("REIN_HTTP_TOKEN", "owner-tok");
    let c = cfg();
    for body in [
        r#"{"credential":"env:NOPE"}"#,
        r#"{"credential":"endpoint:http://x"}"#,
        r#"{}"#,
    ] {
        let (s, _, _) = body_json(
            handle_api_request(
                post("/api/providers/reveal", body, true, Some("owner-tok")),
                &c,
            )
            .await,
        )
        .await;
        assert!(
            s == StatusCode::NOT_FOUND || s == StatusCode::BAD_REQUEST,
            "{body}: {s}"
        );
    }
    std::env::remove_var("REIN_HTTP_TOKEN");
    let (s, _, _) = body_json(
        handle_api_request(
            post(
                "/api/providers/reveal",
                r#"{"credential":"env:GEMINI_API_KEY"}"#,
                true,
                None,
            ),
            &c,
        )
        .await,
    )
    .await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);
}
