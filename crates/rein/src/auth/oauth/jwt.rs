use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use ring::hmac;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Clone, Copy)]
pub struct SigningKeyRef<'a> {
    pub kid: &'a str,
    pub secret_hex: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccessTokenClaims {
    pub sub: String,
    pub jti: String,
    pub iat: i64,
    pub exp: i64,
    pub aud: String,
}

fn decode_hex_32(secret_hex: &str) -> anyhow::Result<[u8; 32]> {
    if secret_hex.len() != 64 {
        anyhow::bail!("OAuth signing key must be 32 bytes hex");
    }
    let mut out = [0u8; 32];
    for (i, chunk) in secret_hex.as_bytes().chunks_exact(2).enumerate() {
        let raw = std::str::from_utf8(chunk)?;
        out[i] = u8::from_str_radix(raw, 16)?;
    }
    Ok(out)
}

fn signing_input(header: &Value, claims: &AccessTokenClaims) -> anyhow::Result<String> {
    let header = URL_SAFE_NO_PAD.encode(serde_json::to_vec(header)?);
    let payload = URL_SAFE_NO_PAD.encode(serde_json::to_vec(claims)?);
    Ok(format!("{header}.{payload}"))
}

pub fn sign_access_token(
    kid: &str,
    secret_hex: &str,
    client_id: &str,
    jti: &str,
    issued_at: i64,
    ttl_seconds: i64,
) -> anyhow::Result<String> {
    let claims = AccessTokenClaims {
        sub: "rein-user".to_string(),
        jti: jti.to_string(),
        iat: issued_at,
        exp: issued_at + ttl_seconds,
        aud: client_id.to_string(),
    };
    let header = json!({ "alg": "HS256", "typ": "JWT", "kid": kid });
    let input = signing_input(&header, &claims)?;
    let key = hmac::Key::new(hmac::HMAC_SHA256, &decode_hex_32(secret_hex)?);
    let sig = hmac::sign(&key, input.as_bytes());
    Ok(format!("{input}.{}", URL_SAFE_NO_PAD.encode(sig.as_ref())))
}

pub fn decode_payload_unverified(token: &str) -> anyhow::Result<Value> {
    let payload = token
        .split('.')
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("malformed JWT"))?;
    let decoded = URL_SAFE_NO_PAD.decode(payload)?;
    Ok(serde_json::from_slice(&decoded)?)
}

pub fn verify_access_token(
    token: &str,
    keys: &[SigningKeyRef<'_>],
    now: i64,
) -> anyhow::Result<AccessTokenClaims> {
    let parts = token.split('.').collect::<Vec<_>>();
    if parts.len() != 3 {
        anyhow::bail!("malformed JWT");
    }
    let header_json: Value = serde_json::from_slice(&URL_SAFE_NO_PAD.decode(parts[0])?)?;
    if header_json.get("alg").and_then(Value::as_str) != Some("HS256") {
        anyhow::bail!("unsupported JWT alg");
    }
    let kid = header_json.get("kid").and_then(Value::as_str).unwrap_or("");
    let input = format!("{}.{}", parts[0], parts[1]);
    let sig = URL_SAFE_NO_PAD.decode(parts[2])?;

    let mut verified = false;
    for key_ref in keys.iter().filter(|key| key.kid == kid).chain(keys.iter()) {
        let key = hmac::Key::new(hmac::HMAC_SHA256, &decode_hex_32(key_ref.secret_hex)?);
        if hmac::verify(&key, input.as_bytes(), &sig).is_ok() {
            verified = true;
            break;
        }
    }
    if !verified {
        anyhow::bail!("invalid JWT signature");
    }

    let claims: AccessTokenClaims = serde_json::from_value(decode_payload_unverified(token)?)?;
    if claims.exp <= now {
        anyhow::bail!("expired JWT");
    }
    Ok(claims)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn access_token_round_trips_and_exposes_only_oauth_claims() {
        let secret_hex = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let token = sign_access_token(
            "kid-1",
            secret_hex,
            "client-1",
            "jti-1",
            1_700_000_000,
            3600,
        )
        .expect("sign token");

        let claims = verify_access_token(
            &token,
            &[SigningKeyRef {
                kid: "kid-1",
                secret_hex,
            }],
            1_700_000_100,
        )
        .expect("verify token");

        assert_eq!(claims.sub, "rein-user");
        assert_eq!(claims.aud, "client-1");
        assert_eq!(claims.jti, "jti-1");
        assert_eq!(claims.exp, 1_700_003_600);
        let payload = decode_payload_unverified(&token).expect("decode payload");
        assert!(payload.get("path").is_none());
        assert!(payload.get("memory_count").is_none());
    }

    #[test]
    fn access_token_rejects_expired_or_wrong_secret() {
        let secret_hex = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        let token = sign_access_token("kid-1", secret_hex, "client-1", "jti-1", 100, 10)
            .expect("sign token");

        assert!(verify_access_token(
            &token,
            &[SigningKeyRef {
                kid: "kid-1",
                secret_hex
            }],
            111,
        )
        .is_err());
        assert!(verify_access_token(
            &token,
            &[SigningKeyRef {
                kid: "kid-1",
                secret_hex: "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
            }],
            105,
        )
        .is_err());
    }
}
