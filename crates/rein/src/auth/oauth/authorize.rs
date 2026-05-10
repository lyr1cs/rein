use ring::hmac;
use serde::Deserialize;

use crate::auth::oauth::store::{self, InsertAuthCodeInput};
use crate::auth::oauth::{
    cookie_value, oauth_error, redirect_with_params, OAuthResponse, OAUTH_OWNER_COOKIE,
};
use crate::auth::policy::request_has_valid_static_bearer;
use crate::config::ReinConfig;

const CSRF_COOKIE: &str = "rein_oauth_csrf";

#[derive(Debug, Deserialize)]
struct AuthorizeQuery {
    client_id: String,
    redirect_uri: String,
    response_type: String,
    code_challenge: String,
    code_challenge_method: String,
    state: String,
}

#[derive(Debug, Deserialize)]
struct AuthorizeForm {
    client_id: String,
    redirect_uri: String,
    code_challenge: String,
    state: String,
    csrf_token: String,
    action: String,
}

fn html_escape(input: &str) -> String {
    input
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

fn csrf_mac(
    secret_hex: &str,
    nonce: &str,
    client_id: &str,
    redirect_uri: &str,
    state: &str,
    challenge: &str,
) -> anyhow::Result<String> {
    let key_bytes = hex_to_bytes(secret_hex)?;
    let key = hmac::Key::new(hmac::HMAC_SHA256, &key_bytes);
    let payload = format!("{nonce}\n{client_id}\n{redirect_uri}\n{state}\n{challenge}");
    let tag = hmac::sign(&key, payload.as_bytes());
    Ok(base64::Engine::encode(
        &base64::engine::general_purpose::URL_SAFE_NO_PAD,
        tag.as_ref(),
    ))
}

fn hex_to_bytes(secret_hex: &str) -> anyhow::Result<Vec<u8>> {
    if !secret_hex.len().is_multiple_of(2) {
        anyhow::bail!("invalid hex");
    }
    secret_hex
        .as_bytes()
        .chunks_exact(2)
        .map(|chunk| {
            let raw = std::str::from_utf8(chunk)?;
            Ok(u8::from_str_radix(raw, 16)?)
        })
        .collect()
}

fn make_csrf(
    secret_hex: &str,
    client_id: &str,
    redirect_uri: &str,
    state: &str,
    challenge: &str,
) -> anyhow::Result<(String, String)> {
    let nonce = store::random_token(16)?;
    let mac = csrf_mac(
        secret_hex,
        &nonce,
        client_id,
        redirect_uri,
        state,
        challenge,
    )?;
    Ok((format!("{nonce}.{mac}"), nonce))
}

fn verify_csrf(
    secret_hex: &str,
    cookie_nonce: &str,
    token: &str,
    client_id: &str,
    redirect_uri: &str,
    state: &str,
    challenge: &str,
) -> bool {
    let Some((nonce, mac)) = token.split_once('.') else {
        return false;
    };
    if nonce != cookie_nonce {
        return false;
    }
    csrf_mac(secret_hex, nonce, client_id, redirect_uri, state, challenge)
        .map(|expected| subtle::ConstantTimeEq::ct_eq(expected.as_bytes(), mac.as_bytes()).into())
        .unwrap_or(false)
}

fn validate_authorize_request(
    config: &ReinConfig,
    query: &AuthorizeQuery,
) -> anyhow::Result<(store::OAuthClient, store::SigningKey)> {
    if query.response_type != "code" {
        anyhow::bail!("response_type must be code");
    }
    if query.code_challenge_method != "S256" {
        anyhow::bail!("code_challenge_method must be S256");
    }
    if query.state.is_empty() {
        anyhow::bail!("state is required");
    }
    let store = config.open_store()?;
    let client = store::get_client(store.conn(), &query.client_id)?
        .ok_or_else(|| anyhow::anyhow!("unknown client_id"))?;
    if !client
        .redirect_uris
        .iter()
        .any(|uri| uri == &query.redirect_uri)
    {
        anyhow::bail!("redirect_uri mismatch");
    }
    let key = store::current_signing_key(store.conn())?;
    Ok((client, key))
}

fn registered_redirect_uri(config: &ReinConfig, query: &AuthorizeQuery) -> bool {
    let Ok(store) = config.open_store() else {
        return false;
    };
    let Ok(Some(client)) = store::get_client(store.conn(), &query.client_id) else {
        return false;
    };
    client
        .redirect_uris
        .iter()
        .any(|uri| uri == &query.redirect_uri)
}

fn redirect_or_bad_request(
    config: &ReinConfig,
    query: &AuthorizeQuery,
    description: &str,
) -> OAuthResponse {
    if registered_redirect_uri(config, query) {
        OAuthResponse::redirect(redirect_with_params(
            &query.redirect_uri,
            &[
                ("error", "invalid_request"),
                ("error_description", description),
                ("state", &query.state),
            ],
        ))
    } else {
        oauth_error(
            hyper::StatusCode::BAD_REQUEST,
            "invalid_request",
            description,
        )
    }
}

fn owner_authorized(headers: &hyper::HeaderMap) -> bool {
    let Some(token) = std::env::var("REIN_HTTP_TOKEN")
        .ok()
        .map(|token| token.trim().to_string())
        .filter(|token| !token.is_empty())
    else {
        return false;
    };
    request_has_valid_static_bearer(headers, &token)
        || cookie_value(headers, OAUTH_OWNER_COOKIE).is_some_and(|value| {
            subtle::ConstantTimeEq::ct_eq(value.as_bytes(), token.as_bytes()).into()
        })
}

fn require_owner_authorization(headers: &hyper::HeaderMap) -> Option<OAuthResponse> {
    if owner_authorized(headers) {
        None
    } else {
        Some(oauth_error(
            hyper::StatusCode::UNAUTHORIZED,
            "access_denied",
            "owner authentication is required before approving OAuth access",
        ))
    }
}

fn csrf_cookie_secure_suffix(config: &ReinConfig) -> &'static str {
    if config
        .server
        .public_url
        .as_deref()
        .map(str::trim)
        .is_some_and(|url| {
            url.get(..8)
                .is_some_and(|scheme| scheme.eq_ignore_ascii_case("https://"))
        })
    {
        "; Secure"
    } else {
        ""
    }
}

pub fn handle_authorize_get(
    headers: &hyper::HeaderMap,
    query: &str,
    config: &ReinConfig,
) -> OAuthResponse {
    let parsed = match serde_urlencoded::from_str::<AuthorizeQuery>(query) {
        Ok(parsed) => parsed,
        Err(err) => {
            return oauth_error(
                hyper::StatusCode::BAD_REQUEST,
                "invalid_request",
                &err.to_string(),
            );
        }
    };
    let (client, key) = match validate_authorize_request(config, &parsed) {
        Ok(value) => value,
        Err(err) => return redirect_or_bad_request(config, &parsed, &err.to_string()),
    };
    if let Some(response) = require_owner_authorization(headers) {
        return response;
    }
    let (csrf_token, nonce) = match make_csrf(
        &key.secret_hex,
        &parsed.client_id,
        &parsed.redirect_uri,
        &parsed.state,
        &parsed.code_challenge,
    ) {
        Ok(value) => value,
        Err(err) => {
            return oauth_error(
                hyper::StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                &err.to_string(),
            );
        }
    };
    let html = format!(
        r#"<!doctype html><html><head><meta charset="utf-8"><title>Rein OAuth Authorization</title></head>
<body><main><h1>Rein - Authorize external application</h1>
<p>{client_name} requests access to this rein memory server.</p>
<dl><dt>Client ID</dt><dd>{client_id}</dd><dt>Redirect</dt><dd>{redirect}</dd></dl>
<form method="post" action="/oauth/authorize">
<input type="hidden" name="client_id" value="{client_id}">
<input type="hidden" name="redirect_uri" value="{redirect}">
<input type="hidden" name="code_challenge" value="{challenge}">
<input type="hidden" name="state" value="{state}">
<input type="hidden" name="csrf_token" value="{csrf}">
<button type="submit" name="action" value="allow">Allow</button>
<button type="submit" name="action" value="deny">Deny</button>
</form></main></body></html>"#,
        client_name = html_escape(&client.client_name),
        client_id = html_escape(&parsed.client_id),
        redirect = html_escape(&parsed.redirect_uri),
        challenge = html_escape(&parsed.code_challenge),
        state = html_escape(&parsed.state),
        csrf = html_escape(&csrf_token),
    );
    OAuthResponse::html(
        hyper::StatusCode::OK,
        html,
        vec![(
            hyper::header::SET_COOKIE.as_str(),
            format!(
                "{CSRF_COOKIE}={nonce}; HttpOnly; SameSite=Lax; Path=/oauth/authorize; Max-Age=600{}",
                csrf_cookie_secure_suffix(config)
            ),
        )],
    )
}

pub fn handle_authorize_post(
    headers: &hyper::HeaderMap,
    body: &[u8],
    config: &ReinConfig,
) -> OAuthResponse {
    let form = match serde_urlencoded::from_bytes::<AuthorizeForm>(body) {
        Ok(form) => form,
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
    let client = match store::get_client(store.conn(), &form.client_id) {
        Ok(Some(client)) => client,
        Ok(None) => {
            return oauth_error(
                hyper::StatusCode::BAD_REQUEST,
                "invalid_request",
                "unknown client_id",
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
    if !client
        .redirect_uris
        .iter()
        .any(|uri| uri == &form.redirect_uri)
    {
        return oauth_error(
            hyper::StatusCode::BAD_REQUEST,
            "invalid_request",
            "redirect_uri mismatch",
        );
    }
    if let Some(response) = require_owner_authorization(headers) {
        return response;
    }
    let key = match store::current_signing_key(store.conn()) {
        Ok(key) => key,
        Err(err) => {
            return oauth_error(
                hyper::StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                &err.to_string(),
            )
        }
    };
    let Some(nonce) = cookie_value(headers, CSRF_COOKIE) else {
        return oauth_error(
            hyper::StatusCode::FORBIDDEN,
            "invalid_request",
            "missing CSRF cookie",
        );
    };
    if !verify_csrf(
        &key.secret_hex,
        &nonce,
        &form.csrf_token,
        &form.client_id,
        &form.redirect_uri,
        &form.state,
        &form.code_challenge,
    ) {
        return oauth_error(
            hyper::StatusCode::FORBIDDEN,
            "invalid_request",
            "invalid CSRF token",
        );
    }
    if form.action == "deny" {
        return OAuthResponse::redirect(redirect_with_params(
            &form.redirect_uri,
            &[("error", "access_denied"), ("state", &form.state)],
        ));
    }
    if form.action != "allow" {
        return oauth_error(
            hyper::StatusCode::BAD_REQUEST,
            "invalid_request",
            "action must be allow or deny",
        );
    }
    let code = match store::insert_auth_code(
        store.conn(),
        InsertAuthCodeInput {
            client_id: form.client_id,
            redirect_uri: form.redirect_uri.clone(),
            code_challenge: form.code_challenge,
            expires_at: chrono::Utc::now().timestamp() + 600,
        },
    ) {
        Ok(code) => code,
        Err(err) => {
            return oauth_error(
                hyper::StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                &err.to_string(),
            )
        }
    };
    OAuthResponse::redirect(redirect_with_params(
        &form.redirect_uri,
        &[("code", &code), ("state", &form.state)],
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EnvGuard {
        key: &'static str,
        value: Option<String>,
    }

    impl EnvGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let guard = Self {
                key,
                value: std::env::var(key).ok(),
            };
            std::env::set_var(key, value);
            guard
        }

        fn remove(key: &'static str) -> Self {
            let guard = Self {
                key,
                value: std::env::var(key).ok(),
            };
            std::env::remove_var(key);
            guard
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.value {
                Some(value) => std::env::set_var(self.key, value),
                None => std::env::remove_var(self.key),
            }
        }
    }

    fn temp_config() -> (tempfile::TempDir, ReinConfig) {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut config = ReinConfig::default();
        config.database.path = dir.path().join("memories.db").display().to_string();
        (dir, config)
    }

    fn register_client(config: &ReinConfig) -> store::RegisteredClient {
        let store = config.open_store().expect("open store");
        store::register_client(
            store.conn(),
            store::RegisterClientInput {
                client_name: format!("client-{}", ulid::Ulid::new()),
                redirect_uris: vec!["https://claude.ai/callback".to_string()],
                grant_types: vec![
                    "authorization_code".to_string(),
                    "refresh_token".to_string(),
                ],
                token_endpoint_auth_method: "none".to_string(),
            },
        )
        .expect("register client")
    }

    fn authorize_query(client_id: &str, redirect_uri: &str) -> String {
        serde_urlencoded::to_string([
            ("client_id", client_id),
            ("redirect_uri", redirect_uri),
            ("response_type", "code"),
            (
                "code_challenge",
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~",
            ),
            ("code_challenge_method", "S256"),
            ("state", "state-123"),
        ])
        .expect("authorize query")
    }

    fn html_input_value(body: &[u8], name: &str) -> String {
        let html = String::from_utf8_lossy(body);
        let needle = format!("name=\"{name}\" value=\"");
        let start = html.find(&needle).expect("input name") + needle.len();
        let rest = &html[start..];
        rest.split('"').next().expect("input value").to_string()
    }

    #[test]
    fn authorize_get_does_not_redirect_unknown_or_mismatched_redirect_uri() {
        let (_dir, config) = temp_config();
        let client = register_client(&config);
        let query = authorize_query(&client.client_id, "https://evil.example/callback");

        let response = handle_authorize_get(&hyper::HeaderMap::new(), &query, &config);

        assert_eq!(response.status, hyper::StatusCode::BAD_REQUEST);
        assert!(response
            .headers
            .iter()
            .all(|(name, _)| *name != hyper::header::LOCATION.as_str()));
    }

    #[test]
    #[serial_test::serial(global_state)]
    fn authorize_get_requires_owner_session_before_rendering_approval_page() {
        let _token = EnvGuard::set("REIN_HTTP_TOKEN", "owner-secret");
        let (_dir, config) = temp_config();
        let client = register_client(&config);
        let query = authorize_query(&client.client_id, "https://claude.ai/callback");

        let response = handle_authorize_get(&hyper::HeaderMap::new(), &query, &config);
        assert_eq!(response.status, hyper::StatusCode::UNAUTHORIZED);

        let mut headers = hyper::HeaderMap::new();
        headers.insert(
            hyper::header::COOKIE,
            hyper::header::HeaderValue::from_static("rein_oauth_owner=owner-secret"),
        );
        let response = handle_authorize_get(&headers, &query, &config);
        assert_eq!(response.status, hyper::StatusCode::OK);
    }

    #[test]
    #[serial_test::serial(global_state)]
    fn authorize_get_marks_csrf_cookie_secure_for_https_public_url() {
        let _token = EnvGuard::set("REIN_HTTP_TOKEN", "owner-secret");
        let (_dir, mut config) = temp_config();
        config.server.public_url = Some("https://rein.example.com".to_string());
        let client = register_client(&config);
        let query = authorize_query(&client.client_id, "https://claude.ai/callback");
        let mut headers = hyper::HeaderMap::new();
        headers.insert(
            hyper::header::COOKIE,
            hyper::header::HeaderValue::from_static("rein_oauth_owner=owner-secret"),
        );

        let response = handle_authorize_get(&headers, &query, &config);

        assert_eq!(response.status, hyper::StatusCode::OK);
        assert!(
            response.headers.iter().any(|(name, value)| *name
                == hyper::header::SET_COOKIE.as_str()
                && value.contains("rein_oauth_csrf=")
                && value.contains("Path=/oauth/authorize")
                && value.contains("Secure")),
            "HTTPS OAuth public_url must emit Secure CSRF cookie: {:?}",
            response.headers
        );
    }

    #[test]
    #[serial_test::serial(global_state)]
    fn authorize_get_rejects_when_owner_token_is_not_configured() {
        let _token = EnvGuard::remove("REIN_HTTP_TOKEN");
        let (_dir, config) = temp_config();
        let client = register_client(&config);
        let query = authorize_query(&client.client_id, "https://claude.ai/callback");

        let response = handle_authorize_get(&hyper::HeaderMap::new(), &query, &config);
        assert_eq!(response.status, hyper::StatusCode::UNAUTHORIZED);
    }

    #[test]
    #[serial_test::serial(global_state)]
    fn authorize_post_rejects_unknown_action_without_issuing_code() {
        let _token = EnvGuard::set("REIN_HTTP_TOKEN", "owner-secret");
        let (_dir, config) = temp_config();
        let client = register_client(&config);
        let query = authorize_query(&client.client_id, "https://claude.ai/callback");
        let mut headers = hyper::HeaderMap::new();
        headers.insert(
            hyper::header::COOKIE,
            hyper::header::HeaderValue::from_static("rein_oauth_owner=owner-secret"),
        );
        let get_response = handle_authorize_get(&headers, &query, &config);
        assert_eq!(get_response.status, hyper::StatusCode::OK);
        let csrf_cookie = get_response
            .headers
            .iter()
            .find(|(name, _)| *name == hyper::header::SET_COOKIE.as_str())
            .map(|(_, value)| value.split(';').next().unwrap_or("").to_string())
            .expect("csrf cookie");
        let csrf_token = html_input_value(&get_response.body, "csrf_token");
        let form = serde_urlencoded::to_string([
            ("client_id", client.client_id.as_str()),
            ("redirect_uri", "https://claude.ai/callback"),
            (
                "code_challenge",
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~",
            ),
            ("state", "state-123"),
            ("csrf_token", csrf_token.as_str()),
            ("action", "unexpected"),
        ])
        .expect("form");
        let cookie_header = format!("rein_oauth_owner=owner-secret; {csrf_cookie}");
        headers.insert(
            hyper::header::COOKIE,
            hyper::header::HeaderValue::from_str(&cookie_header).expect("cookie header"),
        );

        let response = handle_authorize_post(&headers, form.as_bytes(), &config);

        assert_eq!(response.status, hyper::StatusCode::BAD_REQUEST);
        let store = config.open_store().expect("open store");
        let code_count: i64 = store
            .conn()
            .query_row("SELECT COUNT(*) FROM oauth_auth_codes", [], |row| {
                row.get(0)
            })
            .expect("count auth codes");
        assert_eq!(code_count, 0);
    }
}
