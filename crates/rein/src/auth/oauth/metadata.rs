use serde_json::{json, Value};

use crate::config::ReinConfig;

fn trim_trailing_slash(value: &str) -> String {
    value.trim().trim_end_matches('/').to_string()
}

pub fn issuer_from_request(headers: &hyper::HeaderMap, config: &ReinConfig) -> String {
    if let Some(public_url) = config
        .server
        .public_url
        .as_deref()
        .map(trim_trailing_slash)
        .filter(|value| !value.is_empty())
    {
        return public_url;
    }

    if let Some(host) = headers
        .get(hyper::header::HOST)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        let scheme = if host_header_is_loopback(host) {
            "http"
        } else {
            "https"
        };
        return format!("{scheme}://{host}");
    }

    format!("http://127.0.0.1:{}", config.server.sse_port)
}

fn host_header_is_loopback(host: &str) -> bool {
    let Ok(authority) = hyper::http::uri::Authority::try_from(host) else {
        return false;
    };
    let host = authority
        .host()
        .trim_start_matches('[')
        .trim_end_matches(']');
    host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback())
}

fn normalize_resource_path(path: &str) -> String {
    if path.is_empty() || path == "/" {
        String::new()
    } else if path.starts_with('/') {
        path.to_string()
    } else {
        format!("/{path}")
    }
}

pub fn authorization_server_metadata(headers: &hyper::HeaderMap, config: &ReinConfig) -> Value {
    let issuer = issuer_from_request(headers, config);
    json!({
        "issuer": issuer,
        "authorization_endpoint": format!("{issuer}/oauth/authorize"),
        "token_endpoint": format!("{issuer}/oauth/token"),
        "registration_endpoint": format!("{issuer}/oauth/register"),
        "revocation_endpoint": format!("{issuer}/oauth/revoke"),
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["none", "client_secret_basic"],
        "scopes_supported": [],
    })
}

pub fn protected_resource_metadata(
    headers: &hyper::HeaderMap,
    config: &ReinConfig,
    resource_path: &str,
) -> Value {
    let issuer = issuer_from_request(headers, config);
    let resource = format!("{issuer}{}", normalize_resource_path(resource_path));
    json!({
        "resource": resource,
        "authorization_servers": [issuer],
        "bearer_methods_supported": ["header"],
        "resource_documentation": "https://github.com/lyr1cs/rein",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_uses_configured_public_url_as_issuer() {
        let mut config = crate::config::ReinConfig::default();
        config.server.public_url = Some("https://rein.example.com/".to_string());
        let headers = hyper::HeaderMap::new();

        let value = authorization_server_metadata(&headers, &config);

        assert_eq!(value["issuer"], "https://rein.example.com");
        assert_eq!(
            value["authorization_endpoint"],
            "https://rein.example.com/oauth/authorize"
        );
        assert_eq!(value["code_challenge_methods_supported"][0], "S256");
    }

    #[test]
    fn metadata_falls_back_to_https_host_header() {
        let config = crate::config::ReinConfig::default();
        let mut headers = hyper::HeaderMap::new();
        headers.insert(
            hyper::header::HOST,
            hyper::header::HeaderValue::from_static("rein.ts.net"),
        );

        let value = authorization_server_metadata(&headers, &config);

        assert_eq!(value["issuer"], "https://rein.ts.net");
        assert_eq!(
            value["registration_endpoint"],
            "https://rein.ts.net/oauth/register"
        );
    }

    #[test]
    fn metadata_falls_back_to_http_for_loopback_host_header() {
        let config = crate::config::ReinConfig::default();
        let mut headers = hyper::HeaderMap::new();
        headers.insert(
            hyper::header::HOST,
            hyper::header::HeaderValue::from_static("127.0.0.1:8680"),
        );

        let value = authorization_server_metadata(&headers, &config);

        assert_eq!(value["issuer"], "http://127.0.0.1:8680");
        assert_eq!(
            value["authorization_endpoint"],
            "http://127.0.0.1:8680/oauth/authorize"
        );
    }

    #[test]
    fn metadata_falls_back_to_http_for_ipv6_loopback_host_header() {
        let config = crate::config::ReinConfig::default();
        let mut headers = hyper::HeaderMap::new();
        headers.insert(
            hyper::header::HOST,
            hyper::header::HeaderValue::from_static("[::1]:8680"),
        );

        let value = authorization_server_metadata(&headers, &config);

        assert_eq!(value["issuer"], "http://[::1]:8680");
        assert_eq!(
            value["authorization_endpoint"],
            "http://[::1]:8680/oauth/authorize"
        );
    }

    #[test]
    fn protected_resource_metadata_points_to_authorization_server() {
        let mut config = crate::config::ReinConfig::default();
        config.server.public_url = Some("https://rein.example.com".to_string());
        let headers = hyper::HeaderMap::new();

        let value = protected_resource_metadata(&headers, &config, "/mcp");

        assert_eq!(value["resource"], "https://rein.example.com/mcp");
        assert_eq!(
            value["authorization_servers"][0],
            "https://rein.example.com"
        );
    }
}
