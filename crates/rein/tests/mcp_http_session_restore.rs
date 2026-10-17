//! Streamable HTTP session restore (GitHub issue #4).
//!
//! Drives the production `StreamableHttpService` (built by
//! `build_mcp_http_service`) directly through rmcp's public `handle`, so
//! no port is bound and the tests run in parallel with everything else.
//! The control test uses a raw rmcp service WITHOUT a session store to
//! prove the expiry path really produces a 404 when nothing restores it.

use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use http_body_util::combinators::BoxBody;
use http_body_util::{BodyExt, Full};
use hyper::{Request, Response, StatusCode};
use rein::config::ReinConfig;
use rein::mcp::server::{build_mcp_http_service, ReinServer};
use rein::mcp::session_store::PersistentSessionStore;
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::streamable_http_server::{
    SessionId, SessionManager, StreamableHttpServerConfig, StreamableHttpService,
};
use tokio_util::sync::CancellationToken;

type Service = StreamableHttpService<ReinServer, LocalSessionManager>;
type RmcpResponse = Response<BoxBody<Bytes, std::convert::Infallible>>;

const INIT_2025_06_18: &str = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"restore-test","version":"0"}}}"#;
const INIT_2025_03_26: &str = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"restore-test-old","version":"0"}}}"#;
const INITIALIZED: &str = r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#;
const TOOLS_LIST: &str = r#"{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}"#;
const TOOLS_LIST_AGAIN: &str = r#"{"jsonrpc":"2.0","id":3,"method":"tools/list","params":{}}"#;

fn test_config(tmp: &tempfile::TempDir) -> ReinConfig {
    let mut config = ReinConfig::default();
    config.database.path = tmp
        .path()
        .join("memories.db")
        .to_string_lossy()
        .into_owned();
    config.server.sse_bind = "127.0.0.1".to_string();
    config
}

/// Sessions expire after 100 ms instead of rmcp's 300 s default.
fn fast_expiry_manager() -> Arc<LocalSessionManager> {
    let mut manager = LocalSessionManager::default();
    manager.session_config.keep_alive = Some(Duration::from_millis(100));
    Arc::new(manager)
}

fn service_with_store(
    config: &ReinConfig,
    manager: Arc<LocalSessionManager>,
    store: Arc<PersistentSessionStore>,
) -> Service {
    build_mcp_http_service(config.clone(), CancellationToken::new(), manager, store)
}

/// Raw rmcp construction, no session store: the control group.
fn service_without_store(config: &ReinConfig, manager: Arc<LocalSessionManager>) -> Service {
    let config = config.clone();
    StreamableHttpService::new(
        move || Ok(ReinServer::new(config.clone())),
        manager,
        StreamableHttpServerConfig::default()
            .with_stateful_mode(true)
            .disable_allowed_hosts(),
    )
}

fn post(body: &'static str, session_id: Option<&str>) -> Request<Full<Bytes>> {
    let mut builder = Request::post("http://127.0.0.1:8680/mcp")
        .header("host", "127.0.0.1:8680")
        .header("accept", "application/json, text/event-stream")
        .header("content-type", "application/json");
    if let Some(id) = session_id {
        builder = builder.header("mcp-session-id", id);
    }
    builder
        .body(Full::new(Bytes::from_static(body.as_bytes())))
        .expect("request builds")
}

fn delete(session_id: &str) -> Request<Full<Bytes>> {
    Request::delete("http://127.0.0.1:8680/mcp")
        .header("host", "127.0.0.1:8680")
        .header("mcp-session-id", session_id)
        .body(Full::new(Bytes::new()))
        .expect("request builds")
}

fn session_id_of(response: &RmcpResponse) -> String {
    response
        .headers()
        .get("mcp-session-id")
        .expect("initialize response carries mcp-session-id")
        .to_str()
        .unwrap()
        .to_owned()
}

/// Read SSE frames until one carries a JSON-RPC result or the stream
/// ends, bounded by a timeout so a stream that never closes fails the
/// test instead of hanging it.
async fn first_result_frame(response: RmcpResponse) -> String {
    let mut body = response.into_body();
    let mut collected = String::new();
    tokio::time::timeout(Duration::from_secs(5), async {
        while let Some(frame) = body.frame().await {
            let frame = match frame {
                Ok(frame) => frame,
                Err(never) => match never {},
            };
            if let Some(data) = frame.data_ref() {
                collected.push_str(&String::from_utf8_lossy(data));
                if collected.contains("\"result\"") {
                    break;
                }
            }
        }
    })
    .await
    .expect("response stream produced a result frame within 5s");
    collected
}

/// initialize + notifications/initialized, returning the session id.
async fn initialize(service: &Service, init: &'static str) -> String {
    let response = service.handle(post(init, None)).await;
    assert_eq!(response.status(), StatusCode::OK, "initialize");
    let id = session_id_of(&response);
    let frame = first_result_frame(response).await;
    assert!(
        frame.contains("protocolVersion"),
        "initialize result frame: {frame}"
    );
    let ack = service.handle(post(INITIALIZED, Some(id.as_str()))).await;
    assert_eq!(
        ack.status(),
        StatusCode::ACCEPTED,
        "notifications/initialized"
    );
    id
}

/// Poll until rmcp's idle reaper has removed the session. A fixed sleep
/// is not enough: the worker unwinds through `svc.waiting()` and
/// `close_session` after the timer fires.
async fn wait_until_expired(manager: &LocalSessionManager, id: &str) {
    let deadline = Instant::now() + Duration::from_secs(2);
    let session: SessionId = Arc::from(id);
    while manager.has_session(&session).await.expect("has_session") {
        assert!(
            Instant::now() < deadline,
            "session {id} did not expire within 2s"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

async fn is_live(manager: &LocalSessionManager, id: &str) -> bool {
    let session: SessionId = Arc::from(id);
    manager.has_session(&session).await.expect("has_session")
}

#[tokio::test]
async fn expired_session_is_restored_transparently() {
    let tmp = tempfile::tempdir().unwrap();
    let config = test_config(&tmp);
    let manager = fast_expiry_manager();
    let store = Arc::new(PersistentSessionStore::in_memory());
    let service = service_with_store(&config, manager.clone(), store.clone());

    let id = initialize(&service, INIT_2025_06_18).await;
    assert!(store.contains(&id), "initialize persists the session");
    wait_until_expired(&manager, &id).await;

    let response = service.handle(post(TOOLS_LIST, Some(id.as_str()))).await;
    assert_eq!(
        response.status(),
        StatusCode::OK,
        "stale session id must be restored, not answered 404"
    );
    let frame = first_result_frame(response).await;
    assert!(
        frame.contains("rein_recall"),
        "tools/list after restore: {frame}"
    );
    assert!(
        is_live(&manager, &id).await,
        "restored session is live again"
    );
}

#[tokio::test]
async fn without_store_expired_session_is_404() {
    let tmp = tempfile::tempdir().unwrap();
    let config = test_config(&tmp);
    let manager = fast_expiry_manager();
    let service = service_without_store(&config, manager.clone());

    let id = initialize(&service, INIT_2025_06_18).await;
    wait_until_expired(&manager, &id).await;

    let response = service.handle(post(TOOLS_LIST, Some(id.as_str()))).await;
    assert_eq!(
        response.status(),
        StatusCode::NOT_FOUND,
        "control: without a store the expiry path must 404, otherwise the restore test proves nothing"
    );
}

#[tokio::test]
async fn client_delete_terminates_for_good() {
    let tmp = tempfile::tempdir().unwrap();
    let config = test_config(&tmp);
    let manager = Arc::new(LocalSessionManager::default());
    let store = Arc::new(PersistentSessionStore::in_memory());
    let service = service_with_store(&config, manager.clone(), store.clone());

    let id = initialize(&service, INIT_2025_06_18).await;
    let bye = service.handle(delete(&id)).await;
    assert_eq!(bye.status(), StatusCode::ACCEPTED, "DELETE /mcp");
    assert!(!store.contains(&id), "DELETE removes the store entry");
    assert!(!is_live(&manager, &id).await);

    let response = service.handle(post(TOOLS_LIST, Some(id.as_str()))).await;
    assert_eq!(
        response.status(),
        StatusCode::NOT_FOUND,
        "an explicitly terminated session must not be resurrected (MCP spec)"
    );
}

#[tokio::test]
async fn sessions_survive_process_restart() {
    let tmp = tempfile::tempdir().unwrap();
    let config = test_config(&tmp);
    let path = tmp.path().join("mcp-sessions-8680.json");

    let id = {
        let store_a = Arc::new(PersistentSessionStore::open(path.clone()));
        let service_a =
            service_with_store(&config, Arc::new(LocalSessionManager::default()), store_a);
        initialize(&service_a, INIT_2025_06_18).await
        // instance A dropped here: no in-memory state survives
    };

    let store_b = Arc::new(PersistentSessionStore::open(path));
    assert!(
        store_b.contains(&id),
        "sidecar carried the session across the restart"
    );
    let service_b = service_with_store(&config, Arc::new(LocalSessionManager::default()), store_b);

    let response = service_b.handle(post(TOOLS_LIST, Some(id.as_str()))).await;
    assert_eq!(
        response.status(),
        StatusCode::OK,
        "pre-restart session id is restored"
    );
    let frame = first_result_frame(response).await;
    assert!(
        frame.contains("rein_recall"),
        "tools/list after restart: {frame}"
    );
}

#[tokio::test]
async fn older_protocol_version_still_initializes() {
    let tmp = tempfile::tempdir().unwrap();
    let config = test_config(&tmp);
    let manager = Arc::new(LocalSessionManager::default());
    let store = Arc::new(PersistentSessionStore::in_memory());
    let service = service_with_store(&config, manager, store);

    let id = initialize(&service, INIT_2025_03_26).await;
    let response = service.handle(post(TOOLS_LIST, Some(id.as_str()))).await;
    assert_eq!(
        response.status(),
        StatusCode::OK,
        "rmcp 1.8 negotiation keeps 2025-03-26 clients working"
    );
}

#[tokio::test]
async fn sse_resume_on_restored_session_does_not_500() {
    let tmp = tempfile::tempdir().unwrap();
    let config = test_config(&tmp);
    let manager = fast_expiry_manager();
    let store = Arc::new(PersistentSessionStore::in_memory());
    let service = service_with_store(&config, manager.clone(), store);

    let id = initialize(&service, INIT_2025_06_18).await;
    wait_until_expired(&manager, &id).await;
    let restored = service.handle(post(TOOLS_LIST, Some(id.as_str()))).await;
    assert_eq!(restored.status(), StatusCode::OK);
    let _ = first_result_frame(restored).await;

    // A client that held a standalone GET stream before the expiry will
    // try to resume it. The restored worker has no event history; rmcp
    // may refuse, but it must not 500 and the session must keep working.
    let resume = Request::get("http://127.0.0.1:8680/mcp")
        .header("host", "127.0.0.1:8680")
        .header("accept", "text/event-stream")
        .header("mcp-session-id", &id)
        .header("last-event-id", "0/1")
        .body(Full::new(Bytes::new()))
        .unwrap();
    let response = service.handle(resume).await;
    let resume_status = response.status();
    drop(response);
    eprintln!("resume status on restored session: {resume_status}");
    assert_ne!(
        resume_status,
        StatusCode::INTERNAL_SERVER_ERROR,
        "resume against a restored session must not 500"
    );

    let again = service
        .handle(post(TOOLS_LIST_AGAIN, Some(id.as_str())))
        .await;
    assert_eq!(
        again.status(),
        StatusCode::OK,
        "session still serves after the resume attempt (resume answered {resume_status})"
    );
}
