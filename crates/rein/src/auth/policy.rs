#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthPolicy {
    LoopbackOnly,
    BearerRequired,
    OAuth,
    Public,
}

impl AuthPolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LoopbackOnly => "loopback_only",
            Self::BearerRequired => "bearer_required",
            Self::OAuth => "oauth",
            Self::Public => "public",
        }
    }

    pub fn evaluate_http(&self, ctx: &HttpAuthContext<'_>) -> AuthDecision {
        if matches!(self, Self::OAuth)
            && (ctx.path.starts_with("/oauth/")
                || (ctx.method == hyper::Method::GET && is_oauth_metadata_endpoint(ctx.path)))
        {
            return AuthDecision::OAuthEndpoint;
        }
        if ctx.method == hyper::Method::DELETE && ctx.path == "/api/session" {
            return AuthDecision::Allow;
        }
        if ctx.gui_enabled && !ctx.path.starts_with("/api/") && !ctx.path.starts_with("/mcp") {
            return AuthDecision::Allow;
        }
        if matches!(self, Self::OAuth) && is_oauth_owner_api_endpoint(ctx.method, ctx.path) {
            return AuthDecision::Allow;
        }

        match self {
            Self::Public => AuthDecision::Allow,
            Self::LoopbackOnly => {
                if ctx.request_host_is_loopback {
                    AuthDecision::Allow
                } else {
                    AuthDecision::Deny(hyper::StatusCode::UNAUTHORIZED)
                }
            }
            Self::BearerRequired => {
                if ctx
                    .rein_http_token
                    .is_some_and(|expected| request_has_valid_static_bearer(ctx.headers, expected))
                {
                    AuthDecision::Allow
                } else {
                    AuthDecision::Deny(hyper::StatusCode::UNAUTHORIZED)
                }
            }
            Self::OAuth => {
                if ctx.oauth_bearer_valid
                    || (ctx.path.starts_with("/api/")
                        && ctx.rein_http_token.is_some_and(|expected| {
                            request_has_valid_static_bearer(ctx.headers, expected)
                        }))
                {
                    AuthDecision::Allow
                } else {
                    AuthDecision::Deny(hyper::StatusCode::UNAUTHORIZED)
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthDecision {
    Allow,
    Deny(hyper::StatusCode),
    OAuthEndpoint,
}

pub struct HttpAuthContext<'a> {
    pub method: &'a hyper::Method,
    pub path: &'a str,
    pub gui_enabled: bool,
    pub headers: &'a hyper::HeaderMap,
    pub request_host_is_loopback: bool,
    pub rein_http_token: Option<&'a str>,
    pub oauth_bearer_valid: bool,
}

pub fn is_oauth_endpoint(path: &str) -> bool {
    path.starts_with("/oauth/") || is_oauth_metadata_endpoint(path)
}

fn is_oauth_owner_api_endpoint(method: &hyper::Method, path: &str) -> bool {
    (*method == hyper::Method::POST && path == "/api/session")
        || (*method == hyper::Method::GET && path == "/api/oauth/clients")
        || (*method == hyper::Method::POST
            && path.starts_with("/api/oauth/clients/")
            && path.ends_with("/revoke"))
}

pub fn is_oauth_metadata_endpoint(path: &str) -> bool {
    path == "/.well-known/oauth-authorization-server"
        || path == "/.well-known/oauth-protected-resource"
        || path.starts_with("/.well-known/oauth-protected-resource/")
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

fn cookie_values(headers: &hyper::HeaderMap, name: &str) -> Vec<String> {
    let mut values = Vec::new();
    for header in headers.get_all(hyper::header::COOKIE) {
        let Ok(cookies) = header.to_str() else {
            continue;
        };
        for part in cookies.split(';') {
            let Some((key, value)) = part.trim().split_once('=') else {
                continue;
            };
            if key.trim() == name {
                values.push(value.trim().to_string());
            }
        }
    }
    values
}

pub fn request_has_valid_static_bearer(headers: &hyper::HeaderMap, expected: &str) -> bool {
    let auth_header = headers
        .get(hyper::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let token_header = headers
        .get("x-rein-token")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let expected_str = format!("Bearer {expected}");
    constant_time_eq(auth_header, &expected_str)
        || constant_time_eq(token_header, expected)
        || cookie_values(headers, "rein_http_token")
            .iter()
            .any(|value| constant_time_eq(value, expected))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers_with_bearer(token: &str) -> hyper::HeaderMap {
        let mut headers = hyper::HeaderMap::new();
        headers.insert(
            hyper::header::AUTHORIZATION,
            hyper::header::HeaderValue::from_str(&format!("Bearer {token}")).unwrap(),
        );
        headers
    }

    #[test]
    fn loopback_only_allows_loopback_without_bearer_even_when_token_exists() {
        let headers = hyper::HeaderMap::new();
        let ctx = HttpAuthContext {
            method: &hyper::Method::POST,
            path: "/mcp",
            gui_enabled: false,
            headers: &headers,
            request_host_is_loopback: true,
            rein_http_token: Some("secret"),
            oauth_bearer_valid: false,
        };

        assert_eq!(
            AuthPolicy::LoopbackOnly.evaluate_http(&ctx),
            AuthDecision::Allow
        );
    }

    #[test]
    fn loopback_only_denies_non_loopback_even_with_bearer_token() {
        let headers = headers_with_bearer("secret");
        let ctx = HttpAuthContext {
            method: &hyper::Method::POST,
            path: "/mcp",
            gui_enabled: false,
            headers: &headers,
            request_host_is_loopback: false,
            rein_http_token: Some("secret"),
            oauth_bearer_valid: false,
        };

        assert_eq!(
            AuthPolicy::LoopbackOnly.evaluate_http(&ctx),
            AuthDecision::Deny(hyper::StatusCode::UNAUTHORIZED)
        );
    }

    #[test]
    fn bearer_required_accepts_matching_rein_http_token() {
        let headers = headers_with_bearer("secret");
        let ctx = HttpAuthContext {
            method: &hyper::Method::POST,
            path: "/mcp",
            gui_enabled: false,
            headers: &headers,
            request_host_is_loopback: false,
            rein_http_token: Some("secret"),
            oauth_bearer_valid: false,
        };

        assert_eq!(
            AuthPolicy::BearerRequired.evaluate_http(&ctx),
            AuthDecision::Allow
        );
    }

    #[test]
    fn static_bearer_accepts_matching_cookie_when_stale_duplicate_precedes_it() {
        let mut headers = hyper::HeaderMap::new();
        headers.insert(
            hyper::header::COOKIE,
            hyper::header::HeaderValue::from_static(
                "rein_http_token=stale; theme=dark; rein_http_token=secret",
            ),
        );

        assert!(request_has_valid_static_bearer(&headers, "secret"));
    }

    #[test]
    fn public_policy_allows_http_requests() {
        let headers = hyper::HeaderMap::new();
        let ctx = HttpAuthContext {
            method: &hyper::Method::POST,
            path: "/mcp",
            gui_enabled: false,
            headers: &headers,
            request_host_is_loopback: false,
            rein_http_token: None,
            oauth_bearer_valid: false,
        };

        assert_eq!(AuthPolicy::Public.evaluate_http(&ctx), AuthDecision::Allow);
    }

    #[test]
    fn oauth_policy_allows_metadata_and_oauth_endpoints_without_bearer() {
        for path in [
            "/.well-known/oauth-authorization-server",
            "/.well-known/oauth-protected-resource",
            "/.well-known/oauth-protected-resource/mcp",
            "/oauth/register",
            "/oauth/authorize",
        ] {
            let headers = hyper::HeaderMap::new();
            let ctx = HttpAuthContext {
                method: &hyper::Method::GET,
                path,
                gui_enabled: false,
                headers: &headers,
                request_host_is_loopback: false,
                rein_http_token: None,
                oauth_bearer_valid: false,
            };

            assert_eq!(
                AuthPolicy::OAuth.evaluate_http(&ctx),
                AuthDecision::OAuthEndpoint
            );
        }
    }

    #[test]
    fn oauth_policy_rejects_static_owner_token_for_protected_mcp() {
        let headers = headers_with_bearer("owner-secret");
        let ctx = HttpAuthContext {
            method: &hyper::Method::POST,
            path: "/mcp",
            gui_enabled: false,
            headers: &headers,
            request_host_is_loopback: false,
            rein_http_token: Some("owner-secret"),
            oauth_bearer_valid: false,
        };

        assert_eq!(
            AuthPolicy::OAuth.evaluate_http(&ctx),
            AuthDecision::Deny(hyper::StatusCode::UNAUTHORIZED)
        );
    }

    #[test]
    fn oauth_policy_allows_static_owner_token_for_gui_api_routes() {
        let headers = headers_with_bearer("owner-secret");
        let ctx = HttpAuthContext {
            method: &hyper::Method::GET,
            path: "/api/memories",
            gui_enabled: true,
            headers: &headers,
            request_host_is_loopback: false,
            rein_http_token: Some("owner-secret"),
            oauth_bearer_valid: false,
        };

        assert_eq!(AuthPolicy::OAuth.evaluate_http(&ctx), AuthDecision::Allow);
    }

    #[test]
    fn oauth_policy_allows_owner_api_routes_to_reach_route_level_auth() {
        for (method, path) in [
            (hyper::Method::POST, "/api/session"),
            (hyper::Method::GET, "/api/oauth/clients"),
            (hyper::Method::POST, "/api/oauth/clients/client-1/revoke"),
        ] {
            let headers = hyper::HeaderMap::new();
            let ctx = HttpAuthContext {
                method: &method,
                path,
                gui_enabled: false,
                headers: &headers,
                request_host_is_loopback: false,
                rein_http_token: Some("owner-secret"),
                oauth_bearer_valid: false,
            };

            assert_eq!(AuthPolicy::OAuth.evaluate_http(&ctx), AuthDecision::Allow);
        }
    }

    #[test]
    fn non_oauth_policies_do_not_bypass_oauth_paths() {
        let headers = hyper::HeaderMap::new();
        let ctx = HttpAuthContext {
            method: &hyper::Method::POST,
            path: "/oauth/register",
            gui_enabled: false,
            headers: &headers,
            request_host_is_loopback: false,
            rein_http_token: None,
            oauth_bearer_valid: false,
        };

        assert_eq!(
            AuthPolicy::BearerRequired.evaluate_http(&ctx),
            AuthDecision::Deny(hyper::StatusCode::UNAUTHORIZED)
        );
    }

    #[test]
    fn oauth_policy_does_not_bypass_unknown_or_wrong_method_well_known_paths() {
        let headers = hyper::HeaderMap::new();
        let unknown = HttpAuthContext {
            method: &hyper::Method::POST,
            path: "/.well-known/oauth-anything",
            gui_enabled: false,
            headers: &headers,
            request_host_is_loopback: false,
            rein_http_token: None,
            oauth_bearer_valid: false,
        };

        assert_eq!(
            AuthPolicy::OAuth.evaluate_http(&unknown),
            AuthDecision::Deny(hyper::StatusCode::UNAUTHORIZED)
        );

        let wrong_method = HttpAuthContext {
            method: &hyper::Method::POST,
            path: "/.well-known/oauth-authorization-server",
            gui_enabled: false,
            headers: &headers,
            request_host_is_loopback: false,
            rein_http_token: None,
            oauth_bearer_valid: false,
        };

        assert_eq!(
            AuthPolicy::OAuth.evaluate_http(&wrong_method),
            AuthDecision::Deny(hyper::StatusCode::UNAUTHORIZED)
        );
    }
}
