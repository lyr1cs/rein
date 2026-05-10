use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{HashMap, VecDeque};
use std::sync::{LazyLock, Mutex};

use crate::auth::oauth::store::{self, RegisterClientInput};
use crate::auth::oauth::{oauth_error, OAuthResponse};
use crate::config::ReinConfig;

const DCR_LIMIT_PER_HOUR: usize = 10;
static DCR_RATE_LIMIT: LazyLock<Mutex<HashMap<String, VecDeque<i64>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

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
    let cutoff = now - 3600;
    let bucket = buckets.entry(key.to_string()).or_default();
    while bucket.front().is_some_and(|seen| *seen <= cutoff) {
        bucket.pop_front();
    }
    if bucket.len() >= DCR_LIMIT_PER_HOUR {
        return false;
    }
    bucket.push_back(now);
    true
}

pub fn handle_register(body: &[u8], config: &ReinConfig, rate_limit_key: &str) -> OAuthResponse {
    if !check_rate_limit(rate_limit_key, chrono::Utc::now().timestamp()) {
        return oauth_error(
            hyper::StatusCode::TOO_MANY_REQUESTS,
            "invalid_request",
            "dynamic client registration rate limit exceeded",
        );
    }
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
    let store = match config.open_store() {
        Ok(store) => store,
        Err(err) => {
            return oauth_error(
                hyper::StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                &err.to_string(),
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
        Err(err) => {
            return oauth_error(
                hyper::StatusCode::BAD_REQUEST,
                "invalid_client_metadata",
                &err.to_string(),
            );
        }
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
        let key = format!("test-{}", ulid::Ulid::new());
        for offset in 0..DCR_LIMIT_PER_HOUR {
            assert!(check_rate_limit(&key, 1000 + offset as i64));
        }
        assert!(!check_rate_limit(&key, 1100));
        assert!(check_rate_limit(&key, 1000 + 3601));
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
}
