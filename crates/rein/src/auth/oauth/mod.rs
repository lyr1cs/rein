pub mod authorize;
pub mod jwt;
pub mod metadata;
pub mod pkce;
pub mod register;
pub mod revoke;
pub mod store;
pub mod token;

pub const OAUTH_OWNER_COOKIE: &str = "rein_oauth_owner";

#[derive(Debug, Clone)]
pub struct OAuthResponse {
    pub status: hyper::StatusCode,
    pub content_type: &'static str,
    pub headers: Vec<(&'static str, String)>,
    pub body: Vec<u8>,
}

impl OAuthResponse {
    pub fn json(status: hyper::StatusCode, value: serde_json::Value) -> Self {
        Self {
            status,
            content_type: "application/json",
            headers: vec![
                (
                    hyper::header::CACHE_CONTROL.as_str(),
                    "no-store".to_string(),
                ),
                (hyper::header::PRAGMA.as_str(), "no-cache".to_string()),
            ],
            body: serde_json::to_vec(&value)
                .unwrap_or_else(|_| b"{\"error\":\"internal\"}".to_vec()),
        }
    }

    pub fn html(
        status: hyper::StatusCode,
        html: String,
        headers: Vec<(&'static str, String)>,
    ) -> Self {
        Self {
            status,
            content_type: "text/html; charset=utf-8",
            headers,
            body: html.into_bytes(),
        }
    }

    pub fn redirect(location: String) -> Self {
        Self {
            status: hyper::StatusCode::FOUND,
            content_type: "text/plain; charset=utf-8",
            headers: vec![(hyper::header::LOCATION.as_str(), location)],
            body: Vec::new(),
        }
    }
}

pub fn oauth_error(
    status: hyper::StatusCode,
    error: &'static str,
    description: &str,
) -> OAuthResponse {
    OAuthResponse::json(
        status,
        serde_json::json!({
            "error": error,
            "error_description": description,
        }),
    )
}

pub fn percent_encode(input: &str) -> String {
    input
        .bytes()
        .flat_map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                vec![b as char]
            }
            _ => format!("%{b:02X}").chars().collect(),
        })
        .collect()
}

pub fn redirect_with_params(base: &str, params: &[(&str, &str)]) -> String {
    let sep = if base.contains('?') { '&' } else { '?' };
    let query = params
        .iter()
        .map(|(k, v)| format!("{}={}", percent_encode(k), percent_encode(v)))
        .collect::<Vec<_>>()
        .join("&");
    format!("{base}{sep}{query}")
}

pub fn cookie_value(headers: &hyper::HeaderMap, name: &str) -> Option<String> {
    let cookies = headers.get(hyper::header::COOKIE)?.to_str().ok()?;
    cookies.split(';').find_map(|part| {
        let (key, value) = part.trim().split_once('=')?;
        (key.trim() == name).then(|| value.trim().to_string())
    })
}

pub fn verify_bearer(config: &crate::config::ReinConfig, headers: &hyper::HeaderMap) -> bool {
    let Some(token) = headers
        .get(hyper::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
    else {
        return false;
    };
    let store = match config.open_store() {
        Ok(store) => store,
        Err(err) => {
            tracing::warn!("OAuth bearer verification could not open store: {err}");
            return false;
        }
    };
    let keys = match store::signing_keys_for_verification(store.conn()) {
        Ok(keys) => keys,
        Err(err) => {
            tracing::warn!("OAuth bearer verification could not load signing keys: {err}");
            return false;
        }
    };
    let key_refs = keys
        .iter()
        .map(|key| jwt::SigningKeyRef {
            kid: key.kid.as_str(),
            secret_hex: key.secret_hex.as_str(),
        })
        .collect::<Vec<_>>();
    let now = chrono::Utc::now().timestamp();
    let claims = match jwt::verify_access_token(token, &key_refs, now) {
        Ok(claims) => claims,
        Err(_) => return false,
    };
    let grant = match store::active_grant_by_jti(store.conn(), &claims.jti, now) {
        Ok(Some(grant)) => grant,
        _ => return false,
    };
    if grant.client_id != claims.aud {
        return false;
    }
    let _ = store::mark_client_used(store.conn(), &claims.aud);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_oauth_responses_are_non_cacheable() {
        let response = OAuthResponse::json(hyper::StatusCode::OK, serde_json::json!({}));

        assert!(response
            .headers
            .iter()
            .any(|(name, value)| *name == "cache-control" && value == "no-store"));
        assert!(response
            .headers
            .iter()
            .any(|(name, value)| *name == "pragma" && value == "no-cache"));
    }
}
