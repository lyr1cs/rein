use serde::Deserialize;

use crate::auth::oauth::jwt;
use crate::auth::oauth::store;
use crate::auth::oauth::OAuthResponse;
use crate::config::ReinConfig;

#[derive(Debug, Deserialize)]
struct RevokeRequest {
    token: String,
    token_type_hint: Option<String>,
    client_id: Option<String>,
    client_secret: Option<String>,
}

fn revoke_access_token_if_owned(conn: &rusqlite::Connection, client_id: &str, token: &str) -> bool {
    let Ok(keys) = store::signing_keys_for_verification(conn) else {
        return false;
    };
    let refs = keys
        .iter()
        .map(|key| jwt::SigningKeyRef {
            kid: key.kid.as_str(),
            secret_hex: key.secret_hex.as_str(),
        })
        .collect::<Vec<_>>();
    let Ok(claims) = jwt::verify_access_token(token, &refs, chrono::Utc::now().timestamp()) else {
        return false;
    };
    if claims.aud != client_id {
        return false;
    }
    let _ = store::revoke_grant_by_access_jti(conn, &claims.jti);
    true
}

fn revoke_refresh_token_if_owned(
    conn: &rusqlite::Connection,
    client_id: &str,
    token: &str,
) -> bool {
    match store::find_active_grant_by_refresh(
        conn,
        client_id,
        token,
        chrono::Utc::now().timestamp(),
    ) {
        Ok(Some(grant)) => {
            let _ = store::revoke_grant(conn, &grant.grant_id);
            true
        }
        _ => false,
    }
}

pub fn handle_revoke(
    headers: &hyper::HeaderMap,
    body: &[u8],
    config: &ReinConfig,
) -> OAuthResponse {
    let req = match serde_urlencoded::from_bytes::<RevokeRequest>(body) {
        Ok(req) => req,
        Err(_) => return OAuthResponse::json(hyper::StatusCode::OK, serde_json::json!({})),
    };
    let store = match config.open_store() {
        Ok(store) => store,
        Err(_) => return OAuthResponse::json(hyper::StatusCode::OK, serde_json::json!({})),
    };
    let Some(client_id) =
        crate::auth::oauth::token::request_client_id(headers, req.client_id.as_deref())
    else {
        return OAuthResponse::json(hyper::StatusCode::OK, serde_json::json!({}));
    };
    let Ok(Some(client)) = store::get_client(store.conn(), &client_id) else {
        return OAuthResponse::json(hyper::StatusCode::OK, serde_json::json!({}));
    };
    // RFC 7009 hides token existence, but client authentication must still be enforced.
    if let Err(err) = crate::auth::oauth::token::authenticate_client(
        &client,
        headers,
        req.client_secret.as_deref(),
    ) {
        return crate::auth::oauth::oauth_error(
            hyper::StatusCode::UNAUTHORIZED,
            "invalid_client",
            &err.to_string(),
        );
    }

    let access_first = req.token_type_hint.as_deref() == Some("access_token")
        || (req.token_type_hint.as_deref() != Some("refresh_token") && req.token.contains('.'));
    let revoked = if access_first {
        if revoke_access_token_if_owned(store.conn(), &client_id, &req.token) {
            true
        } else {
            revoke_refresh_token_if_owned(store.conn(), &client_id, &req.token)
        }
    } else {
        if revoke_refresh_token_if_owned(store.conn(), &client_id, &req.token) {
            true
        } else {
            revoke_access_token_if_owned(store.conn(), &client_id, &req.token)
        }
    };
    if !revoked {
        tracing::debug!("OAuth revoke request did not match an active token for client");
    }

    OAuthResponse::json(hyper::StatusCode::OK, serde_json::json!({}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::engine::general_purpose::STANDARD;
    use base64::Engine;

    fn temp_config() -> (tempfile::TempDir, ReinConfig) {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut config = ReinConfig::default();
        config.database.path = dir.path().join("memories.db").display().to_string();
        (dir, config)
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

    #[test]
    fn revoke_accepts_basic_client_auth_without_form_client_id() {
        let (_dir, config) = temp_config();
        let store = config.open_store().expect("open store");
        let client = store::register_client(
            store.conn(),
            store::RegisterClientInput {
                client_name: "claude.ai".to_string(),
                redirect_uris: vec!["https://claude.ai/callback".to_string()],
                grant_types: vec![
                    "authorization_code".to_string(),
                    "refresh_token".to_string(),
                ],
                token_endpoint_auth_method: "client_secret_basic".to_string(),
            },
        )
        .expect("register client");
        let secret = client.client_secret.as_deref().expect("client secret");
        store::insert_grant(
            store.conn(),
            store::InsertGrantInput {
                client_id: client.client_id.clone(),
                access_token_jti: "jti-revoke-basic".to_string(),
                access_expires_at: chrono::Utc::now().timestamp() + 3600,
                refresh_token: "refresh-to-revoke".to_string(),
                refresh_expires_at: chrono::Utc::now().timestamp() + 86_400,
            },
        )
        .expect("insert grant");
        drop(store);

        let body = serde_urlencoded::to_string([("token", "refresh-to-revoke")]).expect("form");
        let response = handle_revoke(
            &basic_headers(&client.client_id, secret),
            body.as_bytes(),
            &config,
        );
        assert_eq!(response.status, hyper::StatusCode::OK);

        let store = config.open_store().expect("open store");
        let active =
            store::active_grant_count_for_client(store.conn(), &client.client_id).expect("count");
        assert_eq!(active, 0);
    }

    #[test]
    fn access_token_revoke_is_scoped_to_authenticated_client() {
        let (_dir, config) = temp_config();
        let store = config.open_store().expect("open store");
        let client_a = store::register_client(
            store.conn(),
            store::RegisterClientInput {
                client_name: "connector-a".to_string(),
                redirect_uris: vec!["https://claude.ai/a".to_string()],
                grant_types: vec![
                    "authorization_code".to_string(),
                    "refresh_token".to_string(),
                ],
                token_endpoint_auth_method: "client_secret_basic".to_string(),
            },
        )
        .expect("register client a");
        let client_b = store::register_client(
            store.conn(),
            store::RegisterClientInput {
                client_name: "connector-b".to_string(),
                redirect_uris: vec!["https://claude.ai/b".to_string()],
                grant_types: vec![
                    "authorization_code".to_string(),
                    "refresh_token".to_string(),
                ],
                token_endpoint_auth_method: "client_secret_basic".to_string(),
            },
        )
        .expect("register client b");
        let key = store::current_signing_key(store.conn()).expect("signing key");
        let token = jwt::sign_access_token(
            &key.kid,
            &key.secret_hex,
            &client_a.client_id,
            "jti-client-a",
            chrono::Utc::now().timestamp(),
            3600,
        )
        .expect("access token");
        store::insert_grant(
            store.conn(),
            store::InsertGrantInput {
                client_id: client_a.client_id.clone(),
                access_token_jti: "jti-client-a".to_string(),
                access_expires_at: chrono::Utc::now().timestamp() + 3600,
                refresh_token: "refresh-client-a".to_string(),
                refresh_expires_at: chrono::Utc::now().timestamp() + 86_400,
            },
        )
        .expect("insert grant");
        drop(store);

        let body = serde_urlencoded::to_string([
            ("token", token.as_str()),
            ("token_type_hint", "access_token"),
        ])
        .expect("form");
        let response = handle_revoke(
            &basic_headers(
                &client_b.client_id,
                client_b.client_secret.as_deref().expect("client b secret"),
            ),
            body.as_bytes(),
            &config,
        );
        assert_eq!(response.status, hyper::StatusCode::OK);

        let store = config.open_store().expect("open store");
        let active =
            store::active_grant_count_for_client(store.conn(), &client_a.client_id).expect("count");
        assert_eq!(active, 1, "client B must not revoke client A's grant");
        drop(store);

        let response = handle_revoke(
            &basic_headers(
                &client_a.client_id,
                client_a.client_secret.as_deref().expect("client a secret"),
            ),
            body.as_bytes(),
            &config,
        );
        assert_eq!(response.status, hyper::StatusCode::OK);

        let store = config.open_store().expect("open store");
        let active =
            store::active_grant_count_for_client(store.conn(), &client_a.client_id).expect("count");
        assert_eq!(active, 0, "client A should revoke its own grant");
    }

    #[test]
    fn revoke_treats_token_type_hint_as_advisory() {
        let (_dir, config) = temp_config();
        let store = config.open_store().expect("open store");
        let client = store::register_client(
            store.conn(),
            store::RegisterClientInput {
                client_name: "connector".to_string(),
                redirect_uris: vec!["https://claude.ai/callback".to_string()],
                grant_types: vec![
                    "authorization_code".to_string(),
                    "refresh_token".to_string(),
                ],
                token_endpoint_auth_method: "client_secret_basic".to_string(),
            },
        )
        .expect("register client");
        let secret = client.client_secret.as_deref().expect("client secret");
        store::insert_grant(
            store.conn(),
            store::InsertGrantInput {
                client_id: client.client_id.clone(),
                access_token_jti: "jti-wrong-hint".to_string(),
                access_expires_at: chrono::Utc::now().timestamp() + 3600,
                refresh_token: "refresh-hinted-as-access".to_string(),
                refresh_expires_at: chrono::Utc::now().timestamp() + 86_400,
            },
        )
        .expect("insert grant");
        drop(store);

        let body = serde_urlencoded::to_string([
            ("token", "refresh-hinted-as-access"),
            ("token_type_hint", "access_token"),
        ])
        .expect("form");
        let response = handle_revoke(
            &basic_headers(&client.client_id, secret),
            body.as_bytes(),
            &config,
        );
        assert_eq!(response.status, hyper::StatusCode::OK);

        let store = config.open_store().expect("open store");
        let active =
            store::active_grant_count_for_client(store.conn(), &client.client_id).expect("count");
        assert_eq!(
            active, 0,
            "refresh token must be revoked despite wrong hint"
        );
    }
}
