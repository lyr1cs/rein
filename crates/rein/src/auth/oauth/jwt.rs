// ============================================================================
// v0.31 candidate — pending Codex audit before commit (Agent F4 / A-H1).
// Audit finding: `verify_access_token` previously fell through `.chain(keys.iter())`
// unconditionally after the `kid`-matching filter, allowing a holder of any
// active signing key (e.g. a retired-but-still-active rotation key) to forge
// tokens claiming arbitrary `kid` values.  This patch restricts verification
// to keys whose `kid` matches the JWT header; tokens without `kid` or with an
// unknown `kid` are rejected outright.  See `reviews/fix-20260511-F4-oauth.md`.
// ============================================================================
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
    let kid = header_json
        .get("kid")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("JWT missing kid header"))?;
    if kid.is_empty() {
        anyhow::bail!("JWT kid header is empty");
    }
    let input = format!("{}.{}", parts[0], parts[1]);
    let sig = URL_SAFE_NO_PAD.decode(parts[2])?;

    // v0.31 candidate (Agent F4 / A-H1): iterate ONLY keys whose kid matches.
    // The previous `.chain(keys.iter())` fallback let a compromised retired-
    // but-still-active key forge tokens claiming arbitrary kids.
    let mut verified = false;
    let mut kid_known = false;
    for key_ref in keys.iter().filter(|key| key.kid == kid) {
        kid_known = true;
        let key = hmac::Key::new(hmac::HMAC_SHA256, &decode_hex_32(key_ref.secret_hex)?);
        if hmac::verify(&key, input.as_bytes(), &sig).is_ok() {
            verified = true;
            break;
        }
    }
    if !kid_known {
        anyhow::bail!("unknown JWT kid");
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

    // v0.31 candidate (Agent F4 / A-H1) — regression: forge attempt during
    // signing-key rotation must fail.  Setup: two active keys A and B in the
    // rotation set.  Attacker holds A's secret.  Attacker signs a token with
    // A's secret but stamps `kid="future-attacker"` in the JWT header,
    // hoping the fallback chain finds *some* key that verifies the MAC.
    // With the fix, the kid is unknown to the keyring and verification fails
    // before any MAC is checked.
    #[test]
    fn verify_rejects_forged_kid_during_rotation() {
        let key_a_secret = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let key_b_secret = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        // Sign a token with key A but lie about the kid.
        let forged = sign_access_token(
            "future-attacker",
            key_a_secret,
            "client-1",
            "jti-forge",
            1_700_000_000,
            3600,
        )
        .expect("sign forged token");

        // Rotation set has both A and B active but NO "future-attacker".
        let keys = [
            SigningKeyRef {
                kid: "key-A",
                secret_hex: key_a_secret,
            },
            SigningKeyRef {
                kid: "key-B",
                secret_hex: key_b_secret,
            },
        ];
        let err = verify_access_token(&forged, &keys, 1_700_000_100)
            .expect_err("forged-kid token must be rejected");
        assert!(
            err.to_string().contains("unknown JWT kid"),
            "expected unknown-kid rejection, got: {err}"
        );
    }

    // v0.31 candidate (Agent F4 / A-H1) — sanity: tokens with missing or
    // empty `kid` header are rejected outright.
    #[test]
    fn verify_rejects_kidless_token() {
        // Hand-build a JWT with no kid header.
        let secret_hex = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
        let header = serde_json::json!({ "alg": "HS256", "typ": "JWT" });
        let claims = AccessTokenClaims {
            sub: "rein-user".to_string(),
            jti: "jti-x".to_string(),
            iat: 1_700_000_000,
            exp: 1_700_003_600,
            aud: "client-1".to_string(),
        };
        let header_b64 = URL_SAFE_NO_PAD.encode(serde_json::to_vec(&header).unwrap());
        let payload_b64 = URL_SAFE_NO_PAD.encode(serde_json::to_vec(&claims).unwrap());
        let input = format!("{header_b64}.{payload_b64}");
        let key = hmac::Key::new(hmac::HMAC_SHA256, &decode_hex_32(secret_hex).unwrap());
        let sig = hmac::sign(&key, input.as_bytes());
        let token = format!("{input}.{}", URL_SAFE_NO_PAD.encode(sig.as_ref()));

        let keys = [SigningKeyRef {
            kid: "key-A",
            secret_hex,
        }];
        let err = verify_access_token(&token, &keys, 1_700_000_100)
            .expect_err("kidless token must be rejected");
        assert!(
            err.to_string().contains("missing kid"),
            "expected missing-kid rejection, got: {err}"
        );
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
