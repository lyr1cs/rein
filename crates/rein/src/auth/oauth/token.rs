use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use serde::Deserialize;
use serde_json::json;

use crate::auth::oauth::jwt;
use crate::auth::oauth::pkce;
use crate::auth::oauth::store::{self, InsertGrantInput};
use crate::auth::oauth::{oauth_error, OAuthResponse};
use crate::config::ReinConfig;

#[derive(Debug, Deserialize)]
struct TokenRequest {
    grant_type: String,
    code: Option<String>,
    redirect_uri: Option<String>,
    client_id: Option<String>,
    client_secret: Option<String>,
    code_verifier: Option<String>,
    refresh_token: Option<String>,
}

fn basic_client_secret(headers: &hyper::HeaderMap) -> Option<(String, String)> {
    let auth = headers
        .get(hyper::header::AUTHORIZATION)?
        .to_str()
        .ok()?
        .strip_prefix("Basic ")?;
    let decoded = STANDARD.decode(auth).ok()?;
    let raw = String::from_utf8(decoded).ok()?;
    let (client_id, secret) = raw.split_once(':')?;
    Some((client_id.to_string(), secret.to_string()))
}

pub(crate) fn request_client_id(
    headers: &hyper::HeaderMap,
    form_client_id: Option<&str>,
) -> Option<String> {
    form_client_id
        .map(str::to_string)
        .or_else(|| basic_client_secret(headers).map(|(client_id, _)| client_id))
}

pub(crate) fn authenticate_client(
    client: &store::OAuthClient,
    headers: &hyper::HeaderMap,
    _form_secret: Option<&str>,
) -> anyhow::Result<()> {
    match client.token_endpoint_auth_method.as_str() {
        "none" => Ok(()),
        "client_secret_basic" => {
            let (presented_client_id, presented_secret) = basic_client_secret(headers)
                .ok_or_else(|| anyhow::anyhow!("missing client_secret_basic"))?;
            if presented_client_id != client.client_id {
                anyhow::bail!("invalid client_id");
            }
            let Some(hash) = client.client_secret_hash.as_deref() else {
                anyhow::bail!("client has no secret");
            };
            if !store::verify_secret(&presented_secret, hash) {
                anyhow::bail!("invalid client_secret");
            }
            Ok(())
        }
        other => anyhow::bail!("unsupported token_endpoint_auth_method: {other}"),
    }
}

fn client_allows_grant(client: &store::OAuthClient, grant_type: &str) -> bool {
    client.grant_types.iter().any(|grant| grant == grant_type)
}

fn issue_token_pair(
    conn: &rusqlite::Connection,
    client_id: &str,
    issue_refresh: bool,
) -> anyhow::Result<(String, Option<String>, i64, store::GrantRecord)> {
    let (access_token, refresh_token, grant_input) =
        prepare_token_material(conn, client_id, issue_refresh)?;
    let access_expires_at = grant_input.access_expires_at;
    let grant = store::insert_grant(conn, grant_input)?;
    let refresh_token = issue_refresh.then_some(refresh_token);
    Ok((access_token, refresh_token, access_expires_at, grant))
}

fn prepare_token_material(
    conn: &rusqlite::Connection,
    client_id: &str,
    issue_refresh: bool,
) -> anyhow::Result<(String, String, InsertGrantInput)> {
    let key = store::current_signing_key(conn)?;
    let now = chrono::Utc::now().timestamp();
    let jti = ulid::Ulid::new().to_string();
    let access_expires_at = now + 3600;
    let refresh_expires_at = if issue_refresh {
        now + 86_400 * 30
    } else {
        access_expires_at
    };
    let access_token =
        jwt::sign_access_token(&key.kid, &key.secret_hex, client_id, &jti, now, 3600)?;
    let refresh_token = store::random_token(32)?;
    Ok((
        access_token,
        refresh_token.clone(),
        InsertGrantInput {
            client_id: client_id.to_string(),
            access_token_jti: jti,
            access_expires_at,
            refresh_token,
            refresh_expires_at,
        },
    ))
}

fn token_response(access_token: String, refresh_token: Option<String>) -> OAuthResponse {
    let mut body = serde_json::Map::new();
    body.insert("access_token".to_string(), json!(access_token));
    body.insert("token_type".to_string(), json!("Bearer"));
    body.insert("expires_in".to_string(), json!(3600));
    body.insert("scope".to_string(), json!(""));
    if let Some(refresh_token) = refresh_token {
        body.insert("refresh_token".to_string(), json!(refresh_token));
    }
    OAuthResponse::json(hyper::StatusCode::OK, serde_json::Value::Object(body))
}

fn rollback_code_exchange(conn: &rusqlite::Connection) {
    let _ = conn.execute_batch("ROLLBACK TO oauth_code_exchange; RELEASE oauth_code_exchange");
}

fn release_code_exchange(conn: &rusqlite::Connection) -> anyhow::Result<()> {
    conn.execute_batch("RELEASE oauth_code_exchange")?;
    Ok(())
}

pub fn handle_token(headers: &hyper::HeaderMap, body: &[u8], config: &ReinConfig) -> OAuthResponse {
    let req = match serde_urlencoded::from_bytes::<TokenRequest>(body) {
        Ok(req) => req,
        Err(err) => {
            return oauth_error(
                hyper::StatusCode::BAD_REQUEST,
                "invalid_request",
                &err.to_string(),
            )
        }
    };
    let store = match config.open_store() {
        Ok(store) => store,
        Err(err) => {
            return oauth_error(
                hyper::StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                &err.to_string(),
            )
        }
    };
    match req.grant_type.as_str() {
        "authorization_code" => handle_authorization_code(headers, req, store.conn()),
        "refresh_token" => handle_refresh(headers, req, store.conn()),
        _ => oauth_error(
            hyper::StatusCode::BAD_REQUEST,
            "unsupported_grant_type",
            "grant_type must be authorization_code or refresh_token",
        ),
    }
}

fn handle_authorization_code(
    headers: &hyper::HeaderMap,
    req: TokenRequest,
    conn: &rusqlite::Connection,
) -> OAuthResponse {
    let Some(code) = req.code.as_deref() else {
        return oauth_error(
            hyper::StatusCode::BAD_REQUEST,
            "invalid_request",
            "code is required",
        );
    };
    let Some(client_id) = request_client_id(headers, req.client_id.as_deref()) else {
        return oauth_error(
            hyper::StatusCode::BAD_REQUEST,
            "invalid_request",
            "client_id is required",
        );
    };
    let Some(redirect_uri) = req.redirect_uri.as_deref() else {
        return oauth_error(
            hyper::StatusCode::BAD_REQUEST,
            "invalid_request",
            "redirect_uri is required",
        );
    };
    let Some(verifier) = req.code_verifier.as_deref() else {
        return oauth_error(
            hyper::StatusCode::BAD_REQUEST,
            "invalid_request",
            "code_verifier is required",
        );
    };
    let client = match store::get_client(conn, &client_id) {
        Ok(Some(client)) => client,
        Ok(None) => {
            return oauth_error(
                hyper::StatusCode::UNAUTHORIZED,
                "invalid_client",
                "unknown client",
            )
        }
        Err(err) => {
            return oauth_error(
                hyper::StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                &err.to_string(),
            )
        }
    };
    if let Err(err) = authenticate_client(&client, headers, req.client_secret.as_deref()) {
        return oauth_error(
            hyper::StatusCode::UNAUTHORIZED,
            "invalid_client",
            &err.to_string(),
        );
    }
    let issue_refresh = client_allows_grant(&client, "refresh_token");
    if let Err(err) = conn.execute_batch("SAVEPOINT oauth_code_exchange") {
        return oauth_error(
            hyper::StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            &err.to_string(),
        );
    }
    let consumed = match store::consume_auth_code(
        conn,
        code,
        &client_id,
        redirect_uri,
        chrono::Utc::now().timestamp(),
    ) {
        Ok(code) => code,
        Err(err) => {
            rollback_code_exchange(conn);
            return oauth_error(
                hyper::StatusCode::BAD_REQUEST,
                "invalid_grant",
                &err.to_string(),
            );
        }
    };
    if !pkce::verify_s256(verifier, &consumed.code_challenge) {
        if let Err(err) = release_code_exchange(conn) {
            return oauth_error(
                hyper::StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                &err.to_string(),
            );
        }
        return oauth_error(
            hyper::StatusCode::BAD_REQUEST,
            "invalid_grant",
            "PKCE verifier mismatch",
        );
    }
    let (access_token, refresh_token, _, _) =
        match issue_token_pair(conn, &client_id, issue_refresh) {
            Ok(pair) => pair,
            Err(err) => {
                rollback_code_exchange(conn);
                return oauth_error(
                    hyper::StatusCode::INTERNAL_SERVER_ERROR,
                    "server_error",
                    &err.to_string(),
                );
            }
        };
    if let Err(err) = release_code_exchange(conn) {
        return oauth_error(
            hyper::StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            &err.to_string(),
        );
    }
    token_response(access_token, refresh_token)
}

fn handle_refresh(
    headers: &hyper::HeaderMap,
    req: TokenRequest,
    conn: &rusqlite::Connection,
) -> OAuthResponse {
    let Some(client_id) = request_client_id(headers, req.client_id.as_deref()) else {
        return oauth_error(
            hyper::StatusCode::BAD_REQUEST,
            "invalid_request",
            "client_id is required",
        );
    };
    let Some(refresh_token) = req.refresh_token.as_deref() else {
        return oauth_error(
            hyper::StatusCode::BAD_REQUEST,
            "invalid_request",
            "refresh_token is required",
        );
    };
    let client = match store::get_client(conn, &client_id) {
        Ok(Some(client)) => client,
        Ok(None) => {
            return oauth_error(
                hyper::StatusCode::UNAUTHORIZED,
                "invalid_client",
                "unknown client",
            )
        }
        Err(err) => {
            return oauth_error(
                hyper::StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                &err.to_string(),
            )
        }
    };
    if let Err(err) = authenticate_client(&client, headers, req.client_secret.as_deref()) {
        return oauth_error(
            hyper::StatusCode::UNAUTHORIZED,
            "invalid_client",
            &err.to_string(),
        );
    }
    if !client_allows_grant(&client, "refresh_token") {
        return oauth_error(
            hyper::StatusCode::BAD_REQUEST,
            "unauthorized_client",
            "client is not registered for refresh_token grant",
        );
    }
    let (access_token, new_refresh_token, new_grant) =
        match prepare_token_material(conn, &client_id, true) {
            Ok(material) => material,
            Err(err) => {
                return oauth_error(
                    hyper::StatusCode::INTERNAL_SERVER_ERROR,
                    "server_error",
                    &err.to_string(),
                )
            }
        };
    match store::rotate_refresh_grant(
        conn,
        &client_id,
        refresh_token,
        new_grant,
        chrono::Utc::now().timestamp(),
    ) {
        Ok(store::RefreshGrantLookup::Active(_grant)) => {}
        Ok(store::RefreshGrantLookup::ReusedOrExpired) => {
            let _ = store::revoke_grants_for_client(conn, &client_id);
            return oauth_error(
                hyper::StatusCode::BAD_REQUEST,
                "invalid_grant",
                "refresh_token replay detected",
            );
        }
        Ok(store::RefreshGrantLookup::NotFound) => {
            return oauth_error(
                hyper::StatusCode::BAD_REQUEST,
                "invalid_grant",
                "invalid refresh_token",
            );
        }
        Err(err) => {
            return oauth_error(
                hyper::StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                &err.to_string(),
            )
        }
    };
    token_response(access_token, Some(new_refresh_token))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_config() -> (tempfile::TempDir, ReinConfig) {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut config = ReinConfig::default();
        config.database.path = dir.path().join("memories.db").display().to_string();
        (dir, config)
    }

    fn register_client_with_method(
        config: &ReinConfig,
        grant_types: Vec<String>,
        token_endpoint_auth_method: &str,
    ) -> store::RegisteredClient {
        let store = config.open_store().expect("open store");
        store::register_client(
            store.conn(),
            store::RegisterClientInput {
                client_name: format!("client-{}", ulid::Ulid::new()),
                redirect_uris: vec!["https://claude.ai/callback".to_string()],
                grant_types,
                token_endpoint_auth_method: token_endpoint_auth_method.to_string(),
            },
        )
        .expect("register client")
    }

    fn register_client_with_grants(
        config: &ReinConfig,
        grant_types: Vec<String>,
    ) -> store::RegisteredClient {
        register_client_with_method(config, grant_types, "none")
    }

    fn basic_headers(client_id: &str, client_secret: &str) -> hyper::HeaderMap {
        let mut headers = hyper::HeaderMap::new();
        headers.insert(
            hyper::header::AUTHORIZATION,
            hyper::header::HeaderValue::from_str(&format!(
                "Basic {}",
                STANDARD.encode(format!("{client_id}:{client_secret}"))
            ))
            .expect("basic header"),
        );
        headers
    }

    fn insert_code(config: &ReinConfig, client_id: &str, challenge: &str) -> String {
        let store = config.open_store().expect("open store");
        store::insert_auth_code(
            store.conn(),
            store::InsertAuthCodeInput {
                client_id: client_id.to_string(),
                redirect_uri: "https://claude.ai/callback".to_string(),
                code_challenge: challenge.to_string(),
                expires_at: chrono::Utc::now().timestamp() + 600,
            },
        )
        .expect("insert auth code")
    }

    #[test]
    fn authorization_code_does_not_issue_refresh_for_auth_code_only_client() {
        let (_dir, config) = temp_config();
        let client = register_client_with_grants(&config, vec!["authorization_code".to_string()]);
        let verifier = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~";
        let challenge = pkce::s256_challenge(verifier).expect("challenge");
        let code = insert_code(&config, &client.client_id, &challenge);
        let body = serde_urlencoded::to_string([
            ("grant_type", "authorization_code"),
            ("client_id", client.client_id.as_str()),
            ("redirect_uri", "https://claude.ai/callback"),
            ("code", code.as_str()),
            ("code_verifier", verifier),
        ])
        .expect("form");

        let response = handle_token(&hyper::HeaderMap::new(), body.as_bytes(), &config);
        assert_eq!(response.status, hyper::StatusCode::OK);
        let body: serde_json::Value =
            serde_json::from_slice(&response.body).expect("token response json");
        assert!(body.get("access_token").is_some());
        assert!(body.get("refresh_token").is_none());
    }

    #[test]
    fn authorization_code_is_not_burned_when_token_issuance_fails() {
        let (_dir, config) = temp_config();
        let client = register_client_with_grants(
            &config,
            vec![
                "authorization_code".to_string(),
                "refresh_token".to_string(),
            ],
        );
        let verifier = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~";
        let challenge = pkce::s256_challenge(verifier).expect("challenge");
        let code = insert_code(&config, &client.client_id, &challenge);
        let store = config.open_store().expect("open store");
        store
            .conn()
            .execute("DELETE FROM oauth_signing_keys", [])
            .expect("delete signing keys");
        let response = handle_authorization_code(
            &hyper::HeaderMap::new(),
            TokenRequest {
                grant_type: "authorization_code".to_string(),
                code: Some(code.clone()),
                redirect_uri: Some("https://claude.ai/callback".to_string()),
                client_id: Some(client.client_id.clone()),
                client_secret: None,
                code_verifier: Some(verifier.to_string()),
                refresh_token: None,
            },
            store.conn(),
        );

        assert_eq!(response.status, hyper::StatusCode::INTERNAL_SERVER_ERROR);
        let consumed = store::consume_auth_code(
            store.conn(),
            &code,
            &client.client_id,
            "https://claude.ai/callback",
            chrono::Utc::now().timestamp(),
        )
        .expect("code should remain usable after issuance failure");
        assert_eq!(consumed.code, code);
    }

    #[test]
    fn authorization_code_accepts_basic_client_auth_without_form_client_id() {
        let (_dir, config) = temp_config();
        let client = register_client_with_method(
            &config,
            vec![
                "authorization_code".to_string(),
                "refresh_token".to_string(),
            ],
            "client_secret_basic",
        );
        let secret = client.client_secret.as_deref().expect("client secret");
        let verifier = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~";
        let challenge = pkce::s256_challenge(verifier).expect("challenge");
        let code = insert_code(&config, &client.client_id, &challenge);
        let body = serde_urlencoded::to_string([
            ("grant_type", "authorization_code"),
            ("redirect_uri", "https://claude.ai/callback"),
            ("code", code.as_str()),
            ("code_verifier", verifier),
        ])
        .expect("form");

        let response = handle_token(
            &basic_headers(&client.client_id, secret),
            body.as_bytes(),
            &config,
        );
        assert_eq!(response.status, hyper::StatusCode::OK);
        let body: serde_json::Value =
            serde_json::from_slice(&response.body).expect("token response json");
        assert!(body.get("access_token").is_some());
        assert!(body.get("refresh_token").is_some());
    }

    #[test]
    fn authorization_code_rejects_client_secret_post_for_basic_client() {
        let (_dir, config) = temp_config();
        let client = register_client_with_method(
            &config,
            vec![
                "authorization_code".to_string(),
                "refresh_token".to_string(),
            ],
            "client_secret_basic",
        );
        let secret = client.client_secret.as_deref().expect("client secret");
        let verifier = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~";
        let challenge = pkce::s256_challenge(verifier).expect("challenge");
        let code = insert_code(&config, &client.client_id, &challenge);
        let body = serde_urlencoded::to_string([
            ("grant_type", "authorization_code"),
            ("client_id", client.client_id.as_str()),
            ("client_secret", secret),
            ("redirect_uri", "https://claude.ai/callback"),
            ("code", code.as_str()),
            ("code_verifier", verifier),
        ])
        .expect("form");

        let response = handle_token(&hyper::HeaderMap::new(), body.as_bytes(), &config);

        assert_eq!(response.status, hyper::StatusCode::UNAUTHORIZED);
    }

    #[test]
    fn refresh_accepts_basic_client_auth_without_form_client_id() {
        let (_dir, config) = temp_config();
        let client = register_client_with_method(
            &config,
            vec![
                "authorization_code".to_string(),
                "refresh_token".to_string(),
            ],
            "client_secret_basic",
        );
        let secret = client.client_secret.as_deref().expect("client secret");
        let store = config.open_store().expect("open store");
        store::insert_grant(
            store.conn(),
            store::InsertGrantInput {
                client_id: client.client_id.clone(),
                access_token_jti: "jti-basic-refresh".to_string(),
                access_expires_at: chrono::Utc::now().timestamp() + 3600,
                refresh_token: "refresh-basic-token".to_string(),
                refresh_expires_at: chrono::Utc::now().timestamp() + 86_400,
            },
        )
        .expect("insert grant");
        drop(store);

        let body = serde_urlencoded::to_string([
            ("grant_type", "refresh_token"),
            ("refresh_token", "refresh-basic-token"),
        ])
        .expect("form");
        let response = handle_token(
            &basic_headers(&client.client_id, secret),
            body.as_bytes(),
            &config,
        );
        assert_eq!(response.status, hyper::StatusCode::OK);
        let body: serde_json::Value =
            serde_json::from_slice(&response.body).expect("token response json");
        assert!(body.get("access_token").is_some());
        assert!(body.get("refresh_token").is_some());
    }

    #[test]
    fn random_invalid_refresh_token_does_not_revoke_client_family() {
        let (_dir, config) = temp_config();
        let client = register_client_with_grants(
            &config,
            vec![
                "authorization_code".to_string(),
                "refresh_token".to_string(),
            ],
        );
        let store = config.open_store().expect("open store");
        store::insert_grant(
            store.conn(),
            store::InsertGrantInput {
                client_id: client.client_id.clone(),
                access_token_jti: "jti-one".to_string(),
                access_expires_at: chrono::Utc::now().timestamp() + 3600,
                refresh_token: "real-refresh-token".to_string(),
                refresh_expires_at: chrono::Utc::now().timestamp() + 86_400,
            },
        )
        .expect("insert grant");
        drop(store);

        let body = serde_urlencoded::to_string([
            ("grant_type", "refresh_token"),
            ("client_id", client.client_id.as_str()),
            ("refresh_token", "bogus-refresh-token"),
        ])
        .expect("form");
        let response = handle_token(&hyper::HeaderMap::new(), body.as_bytes(), &config);
        assert_eq!(response.status, hyper::StatusCode::BAD_REQUEST);

        let store = config.open_store().expect("open store");
        let active =
            store::active_grant_count_for_client(store.conn(), &client.client_id).expect("count");
        assert_eq!(active, 1);
    }
}
