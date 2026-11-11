use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{HashMap, VecDeque};
use std::sync::{LazyLock, Mutex};

use crate::auth::oauth::store::{self, RegisterClientInput};
use crate::auth::oauth::{oauth_error, OAuthResponse};
use crate::config::ReinConfig;

const DCR_LIMIT_PER_HOUR: usize = 10;
const DCR_RATE_LIMIT_MAX_KEYS: usize = 4096;
static DCR_RATE_LIMIT: LazyLock<Mutex<HashMap<String, VecDeque<i64>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Registrations accepted per hour from all callers together: half the
/// client table.
///
/// The per-caller bucket is keyed on an address that, behind a proxy which
/// passes forwarding headers through unchanged, the caller can choose, so on
/// its own it does not bound a flood. The table evicts its oldest unused
/// client to make room for a new one, and a client that has registered but
/// not yet been approved by the owner counts as unused. With at most half a
/// table of registrations per hour, such a client is not the oldest unused
/// one for at least an hour after it registers, whatever the callers send.
/// Ordinary use is a handful of registrations per connector.
const DCR_GLOBAL_LIMIT_PER_HOUR: usize = (store::MAX_OAUTH_CLIENTS / 2) as usize;
// Below the size of the client table, so a flood cannot cycle it within the
// hour, and above what one caller may send.
const _: () = assert!(
    (DCR_GLOBAL_LIMIT_PER_HOUR as i64) < store::MAX_OAUTH_CLIENTS
        && DCR_GLOBAL_LIMIT_PER_HOUR > DCR_LIMIT_PER_HOUR
);
static DCR_GLOBAL_RATE_LIMIT: LazyLock<Mutex<VecDeque<i64>>> =
    LazyLock::new(|| Mutex::new(VecDeque::new()));

fn check_global_rate_limit(now: i64) -> bool {
    let Ok(mut accepted) = DCR_GLOBAL_RATE_LIMIT.lock() else {
        return false;
    };
    check_global_rate_limit_in(&mut accepted, now, DCR_GLOBAL_LIMIT_PER_HOUR)
}

fn check_global_rate_limit_in(accepted: &mut VecDeque<i64>, now: i64, limit: usize) -> bool {
    let cutoff = now - 3600;
    while accepted.front().is_some_and(|seen| *seen <= cutoff) {
        accepted.pop_front();
    }
    if accepted.len() >= limit {
        return false;
    }
    accepted.push_back(now);
    true
}

#[derive(Debug, Deserialize)]
struct RegistrationRequest {
    client_name: Option<String>,
    redirect_uris: Vec<String>,
    #[serde(default = "default_grant_types")]
    grant_types: Vec<String>,
    #[serde(default = "default_token_endpoint_auth_method")]
    token_endpoint_auth_method: String,
}

#[derive(Debug, Serialize)]
struct RegistrationResponse {
    client_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    client_secret: Option<String>,
    client_id_issued_at: i64,
    client_secret_expires_at: i64,
    redirect_uris: Vec<String>,
    grant_types: Vec<String>,
    token_endpoint_auth_method: String,
}

fn default_grant_types() -> Vec<String> {
    vec![
        "authorization_code".to_string(),
        "refresh_token".to_string(),
    ]
}

fn default_token_endpoint_auth_method() -> String {
    "none".to_string()
}

fn client_name_or_default(name: Option<String>) -> String {
    name.as_deref()
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .unwrap_or("OAuth client")
        .to_string()
}

fn check_rate_limit(key: &str, now: i64) -> bool {
    let Ok(mut buckets) = DCR_RATE_LIMIT.lock() else {
        return false;
    };
    check_rate_limit_in_buckets(&mut buckets, key, now, DCR_RATE_LIMIT_MAX_KEYS)
}

fn check_rate_limit_in_buckets(
    buckets: &mut HashMap<String, VecDeque<i64>>,
    key: &str,
    now: i64,
    max_keys: usize,
) -> bool {
    if max_keys == 0 {
        return false;
    }
    let cutoff = now - 3600;
    prune_stale_rate_limit_buckets(buckets, cutoff);
    if !buckets.contains_key(key) && buckets.len() >= max_keys {
        evict_oldest_rate_limit_bucket(buckets);
    }
    let bucket = buckets.entry(key.to_string()).or_default();
    if bucket.len() >= DCR_LIMIT_PER_HOUR {
        return false;
    }
    bucket.push_back(now);
    true
}

fn prune_stale_rate_limit_buckets(buckets: &mut HashMap<String, VecDeque<i64>>, cutoff: i64) {
    for bucket in buckets.values_mut() {
        while bucket.front().is_some_and(|seen| *seen <= cutoff) {
            bucket.pop_front();
        }
    }
    buckets.retain(|_, bucket| !bucket.is_empty());
}

fn evict_oldest_rate_limit_bucket(buckets: &mut HashMap<String, VecDeque<i64>>) {
    let Some(oldest_key) = buckets
        .iter()
        .min_by_key(|(_, bucket)| bucket.back().copied().unwrap_or(i64::MIN))
        .map(|(key, _)| key.clone())
    else {
        return;
    };
    buckets.remove(&oldest_key);
}

/// Maps a `register_client` failure to its response.
///
/// Only a rejection of the client's own metadata is a 400 carrying the
/// reason. A full client table is a 503 with its operator-facing sentence:
/// the request was valid and succeeds once a slot is freed. Anything else
/// (SQLite, random generation, secret hashing) is logged and answered with
/// a fixed description, because that text describes server internals.
fn registration_failure(err: &anyhow::Error) -> OAuthResponse {
    if let Some(invalid) = err.downcast_ref::<store::InvalidClientMetadata>() {
        return oauth_error(
            hyper::StatusCode::BAD_REQUEST,
            "invalid_client_metadata",
            &invalid.to_string(),
        );
    }
    if let Some(full) = err.downcast_ref::<store::ClientLimitReached>() {
        return oauth_error(
            hyper::StatusCode::SERVICE_UNAVAILABLE,
            "temporarily_unavailable",
            &full.to_string(),
        );
    }
    tracing::warn!(error = %err, "oauth client registration failed inside the server");
    oauth_error(
        hyper::StatusCode::INTERNAL_SERVER_ERROR,
        "server_error",
        "client registration could not be stored",
    )
}

pub fn handle_register(body: &[u8], config: &ReinConfig, rate_limit_key: &str) -> OAuthResponse {
    let req = match serde_json::from_slice::<RegistrationRequest>(body) {
        Ok(req) => req,
        Err(err) => {
            return oauth_error(
                hyper::StatusCode::BAD_REQUEST,
                "invalid_request",
                &format!("invalid client metadata: {err}"),
            );
        }
    };
    // Only a request that parses as client metadata counts against the
    // caller's bucket; a malformed body is rejected above at no cost.
    let now = chrono::Utc::now().timestamp();
    if !check_rate_limit(rate_limit_key, now) || !check_global_rate_limit(now) {
        return oauth_error(
            hyper::StatusCode::TOO_MANY_REQUESTS,
            "invalid_request",
            "dynamic client registration rate limit exceeded",
        );
    }
    let store = match config.open_store() {
        Ok(store) => store,
        Err(err) => {
            tracing::warn!(error = %err, "oauth client registration could not open the store");
            return oauth_error(
                hyper::StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                "client registration could not be stored",
            );
        }
    };
    let registered = match store::register_client(
        store.conn(),
        RegisterClientInput {
            client_name: client_name_or_default(req.client_name),
            redirect_uris: req.redirect_uris,
            grant_types: req.grant_types,
            token_endpoint_auth_method: req.token_endpoint_auth_method,
        },
    ) {
        Ok(registered) => registered,
        Err(err) => return registration_failure(&err),
    };
    OAuthResponse::json(
        hyper::StatusCode::CREATED,
        json!(RegistrationResponse {
            client_id: registered.client_id,
            client_secret: registered.client_secret,
            client_id_issued_at: registered.client_id_issued_at,
            client_secret_expires_at: registered.client_secret_expires_at,
            redirect_uris: registered.redirect_uris,
            grant_types: registered.grant_types,
            token_endpoint_auth_method: registered.token_endpoint_auth_method,
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dcr_rate_limit_allows_ten_per_hour_per_key() {
        // A private bucket map: the process-global one is shared with the
        // registration tests, whose real-clock calls pruned this test's
        // `now = 1000` entries mid-run when the suite ran in parallel.
        let mut buckets = HashMap::new();
        let key = format!("test-{}", ulid::Ulid::new());
        for offset in 0..DCR_LIMIT_PER_HOUR {
            assert!(check_rate_limit_in_buckets(
                &mut buckets,
                &key,
                1000 + offset as i64,
                DCR_RATE_LIMIT_MAX_KEYS
            ));
        }
        assert!(!check_rate_limit_in_buckets(
            &mut buckets,
            &key,
            1100,
            DCR_RATE_LIMIT_MAX_KEYS
        ));
        assert!(check_rate_limit_in_buckets(
            &mut buckets,
            &key,
            1000 + 3601,
            DCR_RATE_LIMIT_MAX_KEYS
        ));
    }

    #[test]
    fn dcr_rate_limit_prunes_stale_keys_and_caps_bucket_count() {
        let mut buckets = HashMap::new();
        assert!(check_rate_limit_in_buckets(&mut buckets, "k1", 1000, 2));
        assert!(check_rate_limit_in_buckets(&mut buckets, "k2", 1001, 2));
        assert!(check_rate_limit_in_buckets(&mut buckets, "k3", 1002, 2));
        assert_eq!(buckets.len(), 2);
        assert!(!buckets.contains_key("k1"));

        assert!(check_rate_limit_in_buckets(&mut buckets, "fresh", 4603, 2));
        assert_eq!(buckets.len(), 1);
        assert!(buckets.contains_key("fresh"));
    }

    #[test]
    fn dcr_accepts_omitted_client_name() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut config = ReinConfig::default();
        config.database.path = dir.path().join("memories.db").display().to_string();
        let body = serde_json::json!({
            "redirect_uris": ["https://claude.ai/callback"],
            "grant_types": ["authorization_code"],
            "token_endpoint_auth_method": "none",
        });

        let response = handle_register(
            serde_json::to_vec(&body).expect("json").as_slice(),
            &config,
            &format!("test-{}", ulid::Ulid::new()),
        );

        assert_eq!(response.status, hyper::StatusCode::CREATED);
        let store = config.open_store().expect("open store");
        let clients = store::list_clients(store.conn()).expect("list clients");
        assert_eq!(clients[0].client_name, "OAuth client");
    }

    // The limit across all callers holds however many distinct keys the
    // registrations arrive under, and frees up as the hour passes.
    #[test]
    fn dcr_global_rate_limit_bounds_registrations_across_all_callers() {
        let mut accepted = VecDeque::new();
        for i in 0..5 {
            assert!(check_global_rate_limit_in(&mut accepted, 1000 + i, 5));
        }
        assert!(!check_global_rate_limit_in(&mut accepted, 1010, 5));
        // The first acceptance is an hour old: one slot is free again.
        assert!(check_global_rate_limit_in(&mut accepted, 1000 + 3600, 5));
        assert!(!check_global_rate_limit_in(&mut accepted, 1000 + 3600, 5));
    }

    #[test]
    fn dcr_malformed_body_does_not_consume_the_rate_limit_bucket() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut config = ReinConfig::default();
        config.database.path = dir.path().join("memories.db").display().to_string();
        let key = format!("test-{}", ulid::Ulid::new());

        for _ in 0..=DCR_LIMIT_PER_HOUR {
            let response = handle_register(b"{\"redirect_uris\": 7", &config, &key);
            assert_eq!(response.status, hyper::StatusCode::BAD_REQUEST);
        }

        let body = serde_json::json!({
            "redirect_uris": ["https://claude.ai/callback"],
        });
        let response = handle_register(
            serde_json::to_vec(&body).expect("json").as_slice(),
            &config,
            &key,
        );
        assert_eq!(response.status, hyper::StatusCode::CREATED);
    }

    fn register_named(config: &ReinConfig, name: &str) -> OAuthResponse {
        let body = serde_json::json!({
            "client_name": name,
            "redirect_uris": ["https://claude.ai/api/mcp/auth_callback"],
            "token_endpoint_auth_method": "client_secret_basic",
        });
        handle_register(
            serde_json::to_vec(&body).expect("json").as_slice(),
            config,
            &format!("test-{}", ulid::Ulid::new()),
        )
    }

    // GitHub #6: the client name is a label, so the same name registers
    // any number of times — including on a database whose `oauth_clients`
    // still carries the legacy UNIQUE on `client_name` and already holds a
    // client (with a grant) under that name.
    #[test]
    fn dcr_accepts_the_same_client_name_twice() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut config = ReinConfig::default();
        config.database.path = dir.path().join("memories.db").display().to_string();
        drop(config.open_store().expect("create database"));
        {
            let conn = rusqlite::Connection::open(&config.database.path).expect("raw open");
            conn.execute_batch(
                "PRAGMA foreign_keys = OFF;
                 DROP TABLE oauth_clients;
                 CREATE TABLE oauth_clients (
                    client_id TEXT PRIMARY KEY,
                    client_secret_hash TEXT,
                    client_name TEXT NOT NULL UNIQUE,
                    redirect_uris TEXT NOT NULL,
                    grant_types TEXT NOT NULL,
                    token_endpoint_auth_method TEXT NOT NULL CHECK(token_endpoint_auth_method IN ('none', 'client_secret_basic')),
                    registered_at INTEGER NOT NULL,
                    last_used_at INTEGER,
                    revoked_at INTEGER
                 );
                 INSERT INTO oauth_clients VALUES
                    ('old-client', NULL, 'Claude', '[]', '[]', 'none', 1, NULL, NULL);
                 INSERT INTO oauth_grants (
                    grant_id, client_id, access_token_jti, access_expires_at,
                    refresh_token_hash, refresh_token_fingerprint,
                    refresh_expires_at, issued_at, revoked_at
                 ) VALUES ('g1', 'old-client', 'jti1', 999999, 'rh', 'fp', 999999, 1, NULL);
                 UPDATE schema_migrations SET version = 1 WHERE name = 'oauth';",
            )
            .expect("seed legacy oauth_clients");
        }

        assert_eq!(
            register_named(&config, "Claude").status,
            hyper::StatusCode::CREATED
        );
        assert_eq!(
            register_named(&config, "Claude").status,
            hyper::StatusCode::CREATED
        );
        let store = config.open_store().expect("open store");
        assert_eq!(store::list_clients(store.conn()).expect("list").len(), 3);
        let grants: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM oauth_grants WHERE client_id = 'old-client'",
                [],
                |r| r.get(0),
            )
            .expect("count grants");
        assert_eq!(grants, 1);
    }

    #[test]
    fn dcr_only_metadata_rejections_are_reported_as_the_callers_fault() {
        // Failures that are neither SQLite errors nor metadata rejections,
        // such as random generation or secret hashing.
        for internal in ["secure random generation failed", "argon2 hash failed: x"] {
            let response = registration_failure(&anyhow::anyhow!(internal));
            assert_eq!(response.status, hyper::StatusCode::INTERNAL_SERVER_ERROR);
            let body = String::from_utf8(response.body).expect("utf8 body");
            assert!(body.contains("server_error"), "{body}");
            assert!(!body.contains(internal), "{body}");
        }

        let response = registration_failure(&anyhow::Error::new(store::ClientLimitReached));
        assert_eq!(response.status, hyper::StatusCode::SERVICE_UNAVAILABLE);
        let body = String::from_utf8(response.body).expect("utf8 body");
        assert!(body.contains("registration limit reached"), "{body}");

        // A real metadata rejection keeps its 400 and its reason.
        let dir = tempfile::tempdir().expect("tempdir");
        let mut config = ReinConfig::default();
        config.database.path = dir.path().join("memories.db").display().to_string();
        let body = serde_json::json!({
            "redirect_uris": ["https://claude.ai/callback"],
            "grant_types": ["password"],
        });
        let response = handle_register(
            serde_json::to_vec(&body).expect("json").as_slice(),
            &config,
            &format!("test-{}", ulid::Ulid::new()),
        );
        assert_eq!(response.status, hyper::StatusCode::BAD_REQUEST);
        let body = String::from_utf8(response.body).expect("utf8 body");
        assert!(body.contains("invalid_client_metadata"), "{body}");
        assert!(body.contains("grant_types must contain"), "{body}");
    }

    #[test]
    fn dcr_storage_failure_is_a_generic_server_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut config = ReinConfig::default();
        config.database.path = dir.path().join("memories.db").display().to_string();
        {
            let store = config.open_store().expect("open store");
            store
                .conn()
                .execute_batch(
                    "CREATE TRIGGER oauth_clients_reject BEFORE INSERT ON oauth_clients
                     BEGIN SELECT RAISE(ABORT, 'sqlite detail oauth_clients.client_name'); END;",
                )
                .expect("install failing trigger");
        }

        let response = register_named(&config, "Claude");

        assert_eq!(response.status, hyper::StatusCode::INTERNAL_SERVER_ERROR);
        let body = String::from_utf8(response.body).expect("utf8 body");
        assert!(body.contains("server_error"), "{body}");
        assert!(!body.contains("sqlite detail"), "{body}");
        assert!(!body.contains("oauth_clients"), "{body}");
    }
}
