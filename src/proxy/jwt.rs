//! JWT decode helpers for bearer-token diagnostics and routing hints.
//!
//! The proxy does NOT verify JWT signatures — upstream does. These helpers
//! extract a small set of claims for LOCAL routing decisions (is this a
//! ChatGPT login? does the token carry the expected `scp` scope?) and for
//! redacted diagnostic logging when a scope check fails.
//!
//! ## Defensive contract
//!
//! * [`bearer_jwt_info`] ignores tokens whose `exp` is in the past (30 s
//!   clock skew), so routing never trusts expired claims.
//! * [`redact_jwt_payload`] keeps only a small allowlist of claims
//!   (`iss`, `aud`, `sub`, `exp`, `iat`, `nbf`, `scp`, `chatgpt_account_id`,
//!   and the nested OpenAI `auth.chatgpt_account_id`). Everything else is
//!   dropped before any value reaches tracing.

use base64::engine::general_purpose::URL_SAFE_NO_PAD as BASE64_URL_SAFE_NO_PAD;
use base64::Engine;
use serde_json::Value;

/// Decode the JWT payload without validation. Returns None on any parse
/// failure (missing bearer, malformed base64, malformed JSON).  This is a
/// DIAGNOSTIC path — the returned Value MUST be passed through
/// [`redact_jwt_payload`] before being formatted into a log line.
pub(super) fn decode_jwt_payload_for_logging(headers: &hyper::HeaderMap) -> Option<Value> {
    let auth = headers.get("authorization")?.to_str().ok()?;
    let token = auth.strip_prefix("Bearer ")?;
    let payload = token.split('.').nth(1)?;
    let decoded = BASE64_URL_SAFE_NO_PAD.decode(payload).ok()?;
    serde_json::from_slice(&decoded).ok()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct BearerJwtInfo {
    pub(super) scopes: Vec<String>,
    pub(super) has_public_responses_scope: bool,
    pub(super) is_chatgpt_login: bool,
}

/// Keep only non-identifying claims that help debug a scope mismatch. The
/// allowlist is deliberately tight: anything that stably identifies the user
/// or tenant (`sub`, `chatgpt_account_id`, nested `auth.chatgpt_account_id`)
/// is DROPPED here and must not reach tracing output verbatim.
///
/// If a future diagnostic genuinely needs a tenant identifier, it should go
/// through [`hashed_account_fingerprint`] instead, so the log carries a stable
/// but non-invertible value.
pub(super) fn redact_jwt_payload(payload: &Value) -> Value {
    let mut out = serde_json::Map::new();
    // Safe claims: issuer, audience, timestamps, scope list. None of these
    // pin a specific user/tenant on their own.
    for k in ["iss", "aud", "exp", "iat", "nbf", "scp"] {
        if let Some(v) = payload.get(k) {
            out.insert(k.to_string(), v.clone());
        }
    }
    // Expose the presence of the nested OpenAI auth claim as a boolean —
    // enough to distinguish "ChatGPT-login JWT" from "API-key JWT" without
    // leaking the account id.
    let has_chatgpt_login = payload
        .get("https://api.openai.com/auth")
        .and_then(|v| v.get("chatgpt_account_id"))
        .is_some()
        || payload.get("chatgpt_account_id").is_some();
    out.insert(
        "has_chatgpt_login".to_string(),
        Value::Bool(has_chatgpt_login),
    );
    Value::Object(out)
}

/// Short-lived, in-version diagnostic tag for correlating scope-mismatch log
/// lines from the same caller within a single running process.
///
/// **Contract is deliberately narrow** (corrected in v0.19.1 after Codex review):
///
/// * **NOT stable across Rust releases.** `std::collections::hash_map::DefaultHasher`
///   is documented as "the internal algorithm is not specified, and so it and
///   its hashes should not be relied upon over releases" — so two rein binaries
///   built with different Rust toolchains may emit different tags for the same
///   account id. Do not use this as a persistent identifier.
/// * **NOT cryptographically non-invertible.** Truncating a non-keyed
///   `DefaultHasher` output to 32 bits is enough to frustrate casual log
///   readers, but a determined attacker with a dictionary of likely account
///   ids can trivially recover the original. Treat the tag as an obfuscation
///   hint, not a privacy primitive.
/// * **Collision-acceptable for small populations.** 32-bit fingerprint,
///   birthday bound ≈ 1.16×10⁻⁴ at n=1000 accounts. Good enough for
///   best-effort log grouping, insufficient for anything that needs to
///   uniquely identify a tenant.
///
/// If any caller starts relying on stability across builds, cryptographic
/// non-invertibility, or uniqueness for populations >10k, swap this for
/// a keyed BLAKE3 / SHA-256 truncated to ≥64 bits.
pub(super) fn hashed_account_fingerprint(raw: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    raw.hash(&mut h);
    format!("{:08x}", (h.finish() as u32))
}

pub(super) fn current_unix_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Extract the bearer JWT claims relevant to proxy routing. Returns None when
/// the header is missing, malformed, or carries an expired `exp` claim.
pub(super) fn bearer_jwt_info(headers: &hyper::HeaderMap) -> Option<BearerJwtInfo> {
    let auth = headers.get("authorization")?.to_str().ok()?;
    let token = auth.strip_prefix("Bearer ")?;
    let payload = token.split('.').nth(1)?;
    let decoded = BASE64_URL_SAFE_NO_PAD.decode(payload).ok()?;
    let json: Value = serde_json::from_slice(&decoded).ok()?;

    // Expiry check: if `exp` is present and in the past, treat the bearer as
    // absent so routing logic does not trust claims from an expired token.
    // Upstream still validates signatures; this is an additional local
    // sanity check with a 30 s clock-skew allowance.
    if let Some(exp) = json.get("exp").and_then(|v| v.as_i64()) {
        let now = current_unix_timestamp();
        if exp + 30 < now {
            tracing::debug!("bearer token is expired, ignoring JWT claims for routing");
            return None;
        }
    }

    let scopes = json
        .get("scp")
        .and_then(|value| value.as_array())
        .map(|scopes| {
            scopes
                .iter()
                .filter_map(|scope| scope.as_str().map(|s| s.to_string()))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let has_public_responses_scope = scopes
        .iter()
        .any(|scope| scope == "api.responses.read" || scope == "api.responses.write");
    let is_chatgpt_login = json
        .get("https://api.openai.com/auth")
        .and_then(|auth| auth.get("chatgpt_account_id"))
        .and_then(|account| account.as_str())
        .is_some()
        || json
            .get("chatgpt_account_id")
            .and_then(|account| account.as_str())
            .is_some();
    Some(BearerJwtInfo {
        scopes,
        has_public_responses_scope,
        is_chatgpt_login,
    })
}
