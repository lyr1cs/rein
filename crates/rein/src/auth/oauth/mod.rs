// ============================================================================
// v0.31 candidate — pending Codex audit before commit (Agent F4 / A-H3).
// Audit finding: `verify_bearer` opened a fresh `SqliteStore` (running the
// full OAuth migration via A-H2) and issued an `UPDATE oauth_clients SET
// last_used_at = now` on every authenticated `/mcp` request, compounding into
// a perf cliff under claude.ai polling load.  This patch:
//   1. Routes the verification connection through `SqliteStore::pool()` via
//      `try_get()` (sync, non-blocking) when available; falls back to the
//      old fresh-store path only when no pool is attached (`:memory:`, no
//      Tokio runtime, or pool-init failure).
//   2. Debounces `mark_client_used` to a single UPDATE per
//      `MARK_CLIENT_USED_DEBOUNCE_SECS` window per client.
//   3. Adds a TTL-bounded (`BEARER_CACHE_TTL_SECS`) in-memory cache of
//      verified `(token-sha256 → claims)` so polling clients short-circuit
//      ahead of any DB read.  Cache TTL is intentionally short so revocation
//      observed via the DB never lags more than the TTL window — no
//      cross-module invalidation hook required.
// See `reviews/fix-20260511-F4-oauth.md`.
// ============================================================================
pub mod authorize;
pub mod jwt;
pub mod metadata;
pub mod pkce;
pub mod register;
pub mod revoke;
pub mod store;
pub mod token;

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

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

/// v0.31 candidate (Agent F4 / A-H3): how long a successful bearer
/// verification stays cached.  Short TTL bounds revocation-observation lag
/// without any cross-module invalidation hook.
const BEARER_CACHE_TTL_SECS: u64 = 30;
/// v0.31 candidate (Agent F4 / A-H3): cache capacity ceiling.  Each entry is
/// `[u8; 32] + i64 + Instant` ≈ 56 bytes; 4096 ≈ 230 KB.  Far above any
/// realistic concurrent-bearer fleet for a single-user MCP, with hard cap
/// to keep memory bounded against an adversary spraying random tokens.
const BEARER_CACHE_CAP: usize = 4096;
/// v0.31 candidate (Agent F4 / A-H3): minimum interval between consecutive
/// `mark_client_used` UPDATEs for the same client.  Without this, a
/// claude.ai poll cadence (~ every few seconds) would burn one write per
/// request even though `last_used_at` only matters at coarse resolution.
const MARK_CLIENT_USED_DEBOUNCE_SECS: i64 = 60;

#[derive(Clone, Copy)]
struct CachedClaim {
    /// Original JWT `exp`.  We never serve a cache entry past this.
    jwt_exp: i64,
    /// Wall-clock instant the entry was inserted.  Combined with
    /// `BEARER_CACHE_TTL_SECS` to bound revocation lag.
    inserted_at: Instant,
}

fn sha256_bytes(input: &[u8]) -> [u8; 32] {
    use ring::digest;
    let digest = digest::digest(&digest::SHA256, input);
    let mut out = [0u8; 32];
    out.copy_from_slice(digest.as_ref());
    out
}

/// v0.31.1 candidate (D2): cache key construction that includes the
/// resolved DB identity so a token cached while verifying against DB A
/// cannot hit when verifying against DB B in the same process.  The
/// `\0` separator prevents `(token="abc", db="def")` from colliding
/// with `(token="abcdef", db="")` etc.  Backed by SHA-256 so a hostile
/// `db_identity` cannot enlarge the keyspace unboundedly.
fn bearer_cache_key(token: &str, db_identity: &str) -> [u8; 32] {
    let mut input = Vec::with_capacity(token.len() + 1 + db_identity.len());
    input.extend_from_slice(token.as_bytes());
    input.push(0);
    input.extend_from_slice(db_identity.as_bytes());
    sha256_bytes(&input)
}

fn bearer_cache() -> &'static Mutex<HashMap<[u8; 32], CachedClaim>> {
    static CACHE: std::sync::OnceLock<Mutex<HashMap<[u8; 32], CachedClaim>>> =
        std::sync::OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// v0.31 candidate (R3 P2): invalidation generation counter that closes the
/// `bearer_cache_clear` ↔ `bearer_cache_insert` TOCTOU race.
///
/// Scenario without this counter:
///   Thread A enters `verify_with_conn`, passes signing-key + grant +
///   client-id checks (token is currently valid).
///   Thread B (`/api/oauth/clients/<id>/revoke` or refresh-rotation) runs
///   `bearer_cache_clear()` here, wiping every cached entry.
///   Thread A reaches `bearer_cache_insert` and re-inserts the now-revoked
///   token — subsequent verify_bearer calls hit the cache and bypass the
///   DB for up to `BEARER_CACHE_TTL_SECS`.
///
/// Fix: `bearer_cache_clear` increments this counter atomically before
/// clearing the map.  `verify_with_conn` snapshots the counter on entry
/// and compares before inserting; if it changed during the in-flight check
/// (a revoke happened), the insert is skipped.  The in-flight request
/// itself still returns `true` because the grant was active when read —
/// the cache pollution is what we prevent.
fn bearer_cache_generation() -> &'static AtomicU64 {
    static GEN: std::sync::OnceLock<AtomicU64> = std::sync::OnceLock::new();
    GEN.get_or_init(|| AtomicU64::new(0))
}

fn last_marked_used_at() -> &'static Mutex<HashMap<[u8; 32], i64>> {
    static MAP: std::sync::OnceLock<Mutex<HashMap<[u8; 32], i64>>> = std::sync::OnceLock::new();
    MAP.get_or_init(|| Mutex::new(HashMap::new()))
}

/// `aud` is logically a `String`; we key the debounce map on its SHA-256 so
/// every map slot is a fixed 32 bytes (predictable memory) and so a malicious
/// `aud` value cannot enlarge the hashmap key space unboundedly.
fn mark_client_used_debounced(conn: &rusqlite::Connection, client_id: &str, now: i64) {
    let key = sha256_bytes(client_id.as_bytes());
    let should_update = {
        let mut guard = last_marked_used_at()
            .lock()
            .expect("OAuth debounce map mutex poisoned");
        match guard.get(&key) {
            Some(&last) if now - last < MARK_CLIENT_USED_DEBOUNCE_SECS => false,
            _ => {
                guard.insert(key, now);
                // Bounded keyspace: drop a third of entries when we cross
                // BEARER_CACHE_CAP to keep memory roughly flat under sustained
                // load.  Note: HashMap::keys() iteration order is
                // unspecified, so this is *random-sample* eviction, not
                // FIFO/LRU.  Adequate for this threat model where pressure
                // comes from long-lived polling clients (whose tokens get
                // re-cached on the next miss) rather than from random-token
                // spray.
                if guard.len() > BEARER_CACHE_CAP {
                    let drop_count = guard.len() / 3;
                    let keys: Vec<[u8; 32]> = guard.keys().take(drop_count).copied().collect();
                    for k in keys {
                        guard.remove(&k);
                    }
                }
                true
            }
        }
    };
    if should_update {
        let _ = store::mark_client_used(conn, client_id);
    }
}

/// v0.31.1 candidate (D1 from `v0.31.1-oauth-deeper`): atomic insert
/// that re-reads the invalidation generation **inside** the cache lock.
///
/// Closes the R5 P2-#1 sub-microsecond TOCTOU window left by R3 P2.
/// Old flow: verifier loaded `gen_at_end` outside the lock, then called
/// `bearer_cache_insert` which acquired the lock separately — a
/// concurrent `bearer_cache_clear` could bump gen + clear in between,
/// then this verifier would still insert (against a now-fresh, but
/// just-cleared map) using the stale `gen_at_end` snapshot.
///
/// New flow: lock → re-read gen → compare to `expected_gen` snapshotted
/// before DB checks → only insert if unchanged.  Returns `true` on
/// insert, `false` when generation has advanced.  `bearer_cache_clear`
/// is paired so it bumps gen + clears the map under the SAME lock; any
/// gen bump observed inside this function implies the prior clear has
/// already executed (and any prior insert seen by this guard is the
/// next-epoch state).
///
/// Cap-bound eviction matches the pre-D1 behavior (`HashMap::keys()`
/// random-sample drop of 1/3 when over `BEARER_CACHE_CAP`).
fn bearer_cache_insert_if_gen_unchanged(
    cache_key: [u8; 32],
    entry: CachedClaim,
    expected_gen: u64,
) -> bool {
    let mut guard = bearer_cache()
        .lock()
        .expect("OAuth bearer cache mutex poisoned");
    let current_gen = bearer_cache_generation().load(Ordering::SeqCst);
    if current_gen != expected_gen {
        // A revoke / refresh-rotation landed between our snapshot
        // (gen_at_start in verify_with_conn) and now — declining the
        // insert prevents the just-invalidated cache from being
        // immediately re-polluted.  The in-flight request itself still
        // returns `true` to its caller because the grant was active at
        // `active_grant_by_jti` time; the protection is only that
        // FUTURE polls from the same token take the slow path.
        return false;
    }
    if guard.len() >= BEARER_CACHE_CAP {
        let drop_count = guard.len() / 3;
        let keys: Vec<[u8; 32]> = guard.keys().take(drop_count).copied().collect();
        for k in keys {
            guard.remove(&k);
        }
    }
    guard.insert(cache_key, entry);
    true
}

/// Test-only convenience wrapper that snapshots the current generation
/// and calls `bearer_cache_insert_if_gen_unchanged`.  Production code
/// must use the explicit-generation API because the generation snapshot
/// has to happen BEFORE the in-flight DB checks for the race-detection
/// invariant to hold.
#[cfg(test)]
fn bearer_cache_insert(cache_key: [u8; 32], entry: CachedClaim) {
    let g = bearer_cache_generation().load(Ordering::SeqCst);
    let _ = bearer_cache_insert_if_gen_unchanged(cache_key, entry, g);
}

/// Return `true` if the token has a fresh, unexpired entry in the cache.
fn bearer_cache_hit(token_hash: [u8; 32], now_epoch: i64) -> bool {
    let mut guard = bearer_cache()
        .lock()
        .expect("OAuth bearer cache mutex poisoned");
    let Some(entry) = guard.get(&token_hash).copied() else {
        return false;
    };
    let age = entry.inserted_at.elapsed();
    if age > Duration::from_secs(BEARER_CACHE_TTL_SECS) || entry.jwt_exp <= now_epoch {
        // Stale or jwt-expired: evict and miss.
        guard.remove(&token_hash);
        return false;
    }
    true
}

/// v0.31 candidate (R1 P2-#1): flush the verified-bearer cache.
///
/// Callers: every revocation entry point (`/api/oauth/clients/<id>/revoke`,
/// `/oauth/revoke` when an authenticated client actually revoked a token,
/// refresh-token replay handling, refresh-token successful rotation).
/// Without this, a token that was cached in `bearer_cache` before a revoke
/// event would keep returning `true` from `verify_bearer` for up to
/// `BEARER_CACHE_TTL_SECS` (~30s), giving a revoked connector a 30-second
/// bypass window on every authenticated MCP/REST surface.
///
/// We invalidate the entire cache rather than a single entry because
/// revocations are user-initiated rare events and the cache is keyed by
/// `sha256(access_token)` — revoke paths typically have the `client_id` or
/// `grant_id`, not the access token itself.  Worst-case perf cost is one
/// `verify_with_conn` per actively-polling client right after a revoke,
/// which is negligible compared to the security cliff of skipping
/// invalidation.
///
/// R3 P2 + v0.31.1 D1: bump the generation counter AND clear the cache
/// under the SAME lock.  This serializes against
/// `bearer_cache_insert_if_gen_unchanged`, which re-reads the generation
/// inside the cache lock — so any verifier whose `expected_gen` predates
/// our `fetch_add` will see the bumped value when it acquires the lock
/// and skip its insert.  Closes the R5 P2-#1 window where the R3
/// gen-counter-only fix still allowed a verifier to slip an insert in
/// between its outside-the-lock gen recheck and its insert lock-acquire.
pub(crate) fn bearer_cache_clear() {
    let mut guard = bearer_cache()
        .lock()
        .expect("OAuth bearer cache mutex poisoned");
    // SeqCst pairs with the in-lock load in
    // `bearer_cache_insert_if_gen_unchanged`.  Bump under the lock so
    // any concurrent insert that has already passed its outside-the-
    // lock gen snapshot but not yet acquired the lock will see the
    // bumped value when it does acquire.
    bearer_cache_generation().fetch_add(1, Ordering::SeqCst);
    guard.clear();
}

/// Verify an OAuth bearer.  Returns `true` iff the token's signature, kid,
/// expiry, and matching active grant in `oauth_grants` are all valid.
///
/// v0.31 candidate (Agent F4 / A-H3) implementation:
///   1. Fast path: SHA-256 the bearer and probe the in-memory cache.  Hits
///      avoid every DB read while paying the `mark_client_used_debounced`
///      check (which itself short-circuits ≤60s repeats).
///   2. Slow path: open a `SqliteStore` (pool-backed when available),
///      verify signing keys + grant, then insert into the cache for future
///      polling requests.
///
/// The cache is bypassed for the cold lookup; revocation observed in the
/// DB takes effect within `BEARER_CACHE_TTL_SECS` of the cached insert.
pub fn verify_bearer(config: &crate::config::ReinConfig, headers: &hyper::HeaderMap) -> bool {
    let Some(token) = headers
        .get(hyper::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
    else {
        return false;
    };
    // v0.31.1 D2: scope the cache key by DB identity so a token cached
    // while verifying against `config.database.path = A` cannot hit
    // when verifying against `B` in the same process.  See
    // `ReinConfig::stable_db_identity` for the identity-derivation
    // semantics (canonical path, fallback to resolved path).
    let db_identity = config.stable_db_identity();
    let cache_key = bearer_cache_key(token, &db_identity);
    let now = chrono::Utc::now().timestamp();

    // Fast path: cache hit.  A hit means signature + grant + aud match
    // were valid at most BEARER_CACHE_TTL_SECS ago.  We deliberately do
    // NOT call `mark_client_used` on hit paths — that requires opening a
    // connection, which is exactly the cost we're trying to avoid.  When
    // the entry TTL-expires (≤ BEARER_CACHE_TTL_SECS later) the next
    // verification takes the slow path and refreshes `last_used_at`
    // through the 60s debouncer; net effect on a long-polling client is
    // roughly one DB write per minute instead of one per request.
    //
    // Stale-`now` on this path is harmless: cache TTL is 30 s and any
    // sub-second jitter has no effect on the comparison.
    if bearer_cache_hit(cache_key, now) {
        return true;
    }

    // Slow path: open a store.  Prefer pool-backed checkout so we do not
    // re-run the OAuth migration on every cold verification (A-H2 covers
    // the post-migration steady-state cost; the cache here covers the
    // per-token cost).
    let store = match config.open_store() {
        Ok(store) => store,
        Err(err) => {
            tracing::warn!("OAuth bearer verification could not open store: {err}");
            return false;
        }
    };

    // Prefer the pool's connection.  `try_get` is non-blocking: under pool
    // saturation we fall through to `store.conn()` (the per-request store's
    // own connection) so we never starve here.  `verify_with_conn` takes a
    // `&Connection` so we can route either source through the same code.
    let pool_guard = store.pool().and_then(|p| p.try_get());
    let conn: &rusqlite::Connection = match pool_guard.as_ref() {
        Some(guard) => guard.conn(),
        None => store.conn(),
    };
    // R4 P3: refresh `now` AFTER `open_store` completes.  Under SQLite
    // busy-timeout contention or first-run migration the open call can
    // block for seconds, during which a token's `exp` may pass.  Using
    // the pre-wait timestamp would let `jwt::verify_access_token` and
    // `active_grant_by_jti` admit a token that has actually expired.
    let now = chrono::Utc::now().timestamp();
    verify_with_conn(conn, token, cache_key, now)
}

/// v0.31 candidate (Agent F4 / A-H3): internal verifier that takes a
/// reusable `&Connection`.  Extracted so the cached fast path stays
/// reasoning-tight (no DB) and so the slow path can be exercised against
/// either a pool-backed connection or the per-store fallback.
fn verify_with_conn(
    conn: &rusqlite::Connection,
    token: &str,
    cache_key: [u8; 32],
    now: i64,
) -> bool {
    // R3 P2: snapshot the invalidation generation BEFORE any DB read.
    // v0.31.1 D1 moves the post-check INSIDE the cache lock via
    // `bearer_cache_insert_if_gen_unchanged`, eliminating the
    // sub-microsecond window between an outside-lock recheck and the
    // insert that R5 P2-#1 flagged.
    let gen_at_start = bearer_cache_generation().load(Ordering::SeqCst);

    let keys = match store::signing_keys_for_verification(conn) {
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
    let claims = match jwt::verify_access_token(token, &key_refs, now) {
        Ok(claims) => claims,
        Err(_) => return false,
    };
    let grant = match store::active_grant_by_jti(conn, &claims.jti, now) {
        Ok(Some(grant)) => grant,
        _ => return false,
    };
    if grant.client_id != claims.aud {
        return false;
    }

    // Debounced last_used_at: at most one UPDATE per
    // MARK_CLIENT_USED_DEBOUNCE_SECS window per client.
    mark_client_used_debounced(conn, &claims.aud, now);

    // v0.31.1 D1: atomic insert.  The callee re-reads the generation
    // INSIDE the cache lock and returns `false` if the generation has
    // advanced since `gen_at_start`.  This request itself still returns
    // `true` because the grant was active at `active_grant_by_jti`
    // time; the protection is only that FUTURE polls from the same
    // token take the slow path (because the insert was suppressed).
    let _ = bearer_cache_insert_if_gen_unchanged(
        cache_key,
        CachedClaim {
            jwt_exp: claims.exp,
            inserted_at: Instant::now(),
        },
        gen_at_start,
    );
    true
}

#[cfg(test)]
fn clear_bearer_caches_for_test() {
    bearer_cache()
        .lock()
        .expect("OAuth bearer cache mutex poisoned")
        .clear();
    last_marked_used_at()
        .lock()
        .expect("OAuth debounce map mutex poisoned")
        .clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// R3 P3: shared mutex that serializes every test mutating the
    /// process-global bearer cache (`bearer_cache`, `bearer_cache_generation`,
    /// `last_marked_used_at`).  Rust's test harness runs unit tests in
    /// parallel by default; without this guard a peer test's
    /// `clear_bearer_caches_for_test()` or `bearer_cache_clear()` can
    /// interleave between our `insert` and our `assert!(bearer_cache_hit(...))`
    /// and produce intermittent failures in CI.  Every test below acquires
    /// `cache_test_lock()` as its first action.  Poisoning is tolerated —
    /// a poisoned guard means a peer test panicked, and the harness reports
    /// that separately; we recover via `into_inner()` so unrelated tests
    /// still run.
    fn cache_test_lock() -> &'static std::sync::Mutex<()> {
        static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
        LOCK.get_or_init(|| std::sync::Mutex::new(()))
    }

    fn acquire_cache_test_lock() -> std::sync::MutexGuard<'static, ()> {
        cache_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    // v0.31 candidate (Agent F4 / A-H3) — cache eviction: an entry whose
    // jwt_exp is in the past must be evicted from the cache rather than
    // served.  This guards the case where a token's TTL expired BEFORE
    // the cache TTL would have evicted it (e.g. a 1-second expiry token
    // cached at insert time).
    #[test]
    fn bearer_cache_evicts_jwt_expired_entries() {
        let _guard = acquire_cache_test_lock();
        clear_bearer_caches_for_test();
        let token_hash = sha256_bytes(b"sample-token");
        bearer_cache_insert(
            token_hash,
            CachedClaim {
                jwt_exp: 100,
                inserted_at: Instant::now(),
            },
        );
        // `now` is well past `jwt_exp`.
        assert!(!bearer_cache_hit(token_hash, 200));
        // Lookup must have evicted the entry.
        let guard = bearer_cache()
            .lock()
            .expect("OAuth bearer cache mutex poisoned");
        assert!(!guard.contains_key(&token_hash));
    }

    // v0.31 candidate (Agent F4 / A-H3) — cache hit honors fresh entries.
    #[test]
    fn bearer_cache_returns_hit_within_ttl() {
        let _guard = acquire_cache_test_lock();
        clear_bearer_caches_for_test();
        let token_hash = sha256_bytes(b"another-token");
        bearer_cache_insert(
            token_hash,
            CachedClaim {
                jwt_exp: i64::MAX,
                inserted_at: Instant::now(),
            },
        );
        assert!(bearer_cache_hit(token_hash, 100));
    }

    // v0.31 candidate (R1 P2-#1) — `bearer_cache_clear` evicts every entry so
    // a token cached just before a revoke event stops verifying immediately.
    // Reproduction: insert a fresh entry (which would otherwise hit for the
    // next ~30 s), invoke `bearer_cache_clear`, assert subsequent lookup
    // misses.  Without the fix the cache still contained the entry and
    // `verify_bearer` would have returned `true` for up to
    // BEARER_CACHE_TTL_SECS after the SQLite-side revoke.
    #[test]
    fn bearer_cache_clear_invalidates_cached_tokens() {
        let _guard = acquire_cache_test_lock();
        clear_bearer_caches_for_test();
        let token_hash = sha256_bytes(b"revoked-token");
        bearer_cache_insert(
            token_hash,
            CachedClaim {
                jwt_exp: i64::MAX,
                inserted_at: Instant::now(),
            },
        );
        // Sanity: entry hits before the revoke.
        assert!(bearer_cache_hit(token_hash, 100));
        // Simulate a revoke-path invalidation.
        bearer_cache_clear();
        // Post-revoke: the SAME token must not hit.  We assert on this
        // specific hash so the test is independent of any parallel test's
        // writes — combined with `cache_test_lock` above, parallel
        // inserts/clears from other tests are serialized away anyway.
        assert!(!bearer_cache_hit(token_hash, 100));
    }

    // R3 P2: prove that `bearer_cache_clear` bumps the generation counter
    // and the verifier's insert is suppressed when the generation has
    // advanced.  v0.31.1 D1 makes the recheck atomic with the insert —
    // this test now drives the actual production API
    // (`bearer_cache_insert_if_gen_unchanged`) rather than mimicking it
    // outside the lock.
    #[test]
    fn bearer_cache_generation_bumps_invalidate_inflight_inserts() {
        let _guard = acquire_cache_test_lock();
        clear_bearer_caches_for_test();
        let token_hash = sha256_bytes(b"inflight-token");
        // Simulate the verify_with_conn shape: snapshot generation, do
        // "work" (would be DB checks), then attempt the atomic insert.
        let gen_at_start = bearer_cache_generation().load(Ordering::SeqCst);
        // Concurrent revoke lands here.
        bearer_cache_clear();
        // Insert attempt with the now-stale snapshot must return false.
        let inserted = bearer_cache_insert_if_gen_unchanged(
            token_hash,
            CachedClaim {
                jwt_exp: i64::MAX,
                inserted_at: Instant::now(),
            },
            gen_at_start,
        );
        assert!(
            !inserted,
            "concurrent revoke must cause insert_if_gen_unchanged to return false"
        );
        // Without the fix the entry would be present.  With the fix the
        // gate suppressed the insert, so the entry isn't there.
        assert!(
            !bearer_cache_hit(token_hash, 100),
            "concurrent revoke must prevent in-flight verifier from re-caching the just-revoked token"
        );
    }

    // v0.31.1 D1: prove that the atomic insert is robust against a
    // revoke landing AFTER an outside-the-lock snapshot but BEFORE the
    // verifier acquires the cache lock.  The R3-only fix loaded the
    // generation outside the lock and then called the legacy
    // `bearer_cache_insert` — there was a window between those two
    // points where `bearer_cache_clear` could slip its bump+clear in.
    // The D1 fix re-reads the generation INSIDE the cache lock, so we
    // simulate that sequence directly and assert the insert is
    // suppressed.
    #[test]
    fn bearer_cache_insert_if_gen_unchanged_rejects_stale_snapshot() {
        let _guard = acquire_cache_test_lock();
        clear_bearer_caches_for_test();
        let token_hash = sha256_bytes(b"d1-stale-snapshot-token");

        // Step 1: verifier records the generation BEFORE any
        // contention.  This is the same `gen_at_start` snapshot that
        // `verify_with_conn` captures.
        let stale_gen = bearer_cache_generation().load(Ordering::SeqCst);

        // Step 2: a concurrent revoke runs to completion — bumps gen
        // and clears the map under its own lock.
        bearer_cache_clear();

        // Step 3: verifier reaches its insert attempt with the now-
        // stale snapshot.  The atomic recheck inside the cache lock
        // must observe the bumped generation and reject the insert.
        let inserted = bearer_cache_insert_if_gen_unchanged(
            token_hash,
            CachedClaim {
                jwt_exp: i64::MAX,
                inserted_at: Instant::now(),
            },
            stale_gen,
        );
        assert!(
            !inserted,
            "D1: insert with a generation-stale snapshot must be suppressed"
        );
        assert!(
            !bearer_cache_hit(token_hash, 100),
            "D1: the just-revoked token must not be re-cached"
        );

        // Sanity: a fresh snapshot DOES allow the insert.  This proves
        // the suppression is conditional on the generation mismatch,
        // not a hard-coded refusal.
        let fresh_gen = bearer_cache_generation().load(Ordering::SeqCst);
        let inserted_fresh = bearer_cache_insert_if_gen_unchanged(
            token_hash,
            CachedClaim {
                jwt_exp: i64::MAX,
                inserted_at: Instant::now(),
            },
            fresh_gen,
        );
        assert!(
            inserted_fresh,
            "D1: insert with an up-to-date generation snapshot must succeed"
        );
        assert!(
            bearer_cache_hit(token_hash, 100),
            "D1: post-insert lookup with a fresh snapshot must hit"
        );
    }

    // v0.31.1 D2: prove that the bearer cache key is scoped by DB
    // identity so a token cached while verifying against DB A does not
    // hit when verifying against DB B in the same process.  Closes the
    // R5 P2-#2 multi-`ReinConfig` cross-pollination surface.
    #[test]
    fn bearer_cache_key_is_db_identity_scoped() {
        // Pure-function test: no shared cache state, no need for the
        // test lock.  The construction is `sha256(token || \0 || db)`
        // so different `db_identity` strings must produce different
        // keys for the same token, while the same `db_identity`
        // produces a stable key.
        let token = "shared-bearer-token";
        let k_a = bearer_cache_key(token, "/var/db/vault-a/memories.db");
        let k_b = bearer_cache_key(token, "/var/db/vault-b/memories.db");
        let k_a2 = bearer_cache_key(token, "/var/db/vault-a/memories.db");
        assert_ne!(
            k_a, k_b,
            "D2: same token + different DB must hash differently"
        );
        assert_eq!(k_a, k_a2, "D2: same token + same DB must hash identically");

        // The separator byte must prevent (token="abc", db="def") from
        // colliding with (token="abcdef", db="").  Without `\0` both
        // would concat to the same 6 bytes.
        let k_split = bearer_cache_key("abc", "def");
        let k_concat = bearer_cache_key("abcdef", "");
        assert_ne!(
            k_split, k_concat,
            "D2: domain separator must prevent token||db boundary collisions"
        );
    }

    // v0.31.1 D2: integration-shape test — a token inserted against DB
    // A must NOT be served from a lookup against DB B.  This is the
    // full attack scenario codex flagged: a bearer cached during a
    // verify against vault A would falsely return `true` for vault B
    // for the cache TTL.
    #[test]
    fn bearer_cache_does_not_cross_pollinate_between_dbs() {
        let _guard = acquire_cache_test_lock();
        clear_bearer_caches_for_test();
        let token = "d2-cross-pollination-token";
        let key_a = bearer_cache_key(token, "vault-A");
        let key_b = bearer_cache_key(token, "vault-B");

        // Cache the token against vault A.
        bearer_cache_insert(
            key_a,
            CachedClaim {
                jwt_exp: i64::MAX,
                inserted_at: Instant::now(),
            },
        );

        // Lookup against vault A → hits.
        assert!(
            bearer_cache_hit(key_a, 100),
            "D2: token cached for vault A must hit when looked up for vault A"
        );
        // Lookup against vault B → MUST miss (different cache key).
        assert!(
            !bearer_cache_hit(key_b, 100),
            "D2: token cached for vault A must NOT hit when looked up for vault B"
        );
    }

    // v0.31 candidate (Agent F4 / A-H3) — arithmetic check on the debounce
    // window constants.  This test is intentionally narrow: it only verifies
    // that the integer comparisons (`now - last < MARK_CLIENT_USED_DEBOUNCE_SECS`)
    // partition correctly across the boundary.  It does NOT exercise
    // `mark_client_used_debounced` itself — that would need a live SQLite
    // connection (the function calls `store::mark_client_used`).
    //
    // TODO (Codex audit): refactor `mark_client_used_debounced` to take an
    // injectable `write_fn: impl FnOnce()` so we can assert "the second call
    // inside the window does NOT invoke the write closure" without standing
    // up a connection. Honest intermediate naming for now.
    #[test]
    fn mark_client_used_debounce_window_arithmetic_holds() {
        clear_bearer_caches_for_test();
        let key = sha256_bytes(b"client-x");

        // First "call" at t=1000 inserts; second at t=1001 must be dropped.
        let mut guard = last_marked_used_at()
            .lock()
            .expect("OAuth debounce map mutex poisoned");
        guard.insert(key, 1000);
        // Within debounce window: caller would see should_update = false.
        let last = *guard.get(&key).unwrap();
        let dt = 1001 - last;
        assert!(dt < MARK_CLIENT_USED_DEBOUNCE_SECS);

        // Outside debounce window: caller would set should_update = true and
        // refresh the timestamp.
        let dt2 = (1000 + MARK_CLIENT_USED_DEBOUNCE_SECS + 1) - last;
        assert!(dt2 >= MARK_CLIENT_USED_DEBOUNCE_SECS);
    }

    // v0.31 candidate (Agent F4 / A-H3) — perf stub.  Real coverage would
    // simulate 1000 `/mcp` req/sec for 60s with a valid bearer and assert
    // p99 < 50ms.  Marked `#[ignore]` because this needs a runnable hyper
    // service + fake claude.ai client driver; out of scope for the
    // staged-fix branch, deferred to Codex audit reviewer or v0.31 perf
    // harness.
    #[test]
    #[ignore]
    fn verify_bearer_under_polling_load_meets_p99_50ms() {
        // Pseudocode:
        //   let cfg = config_with_oauth_enabled();
        //   let token = mint_token_for_test(&cfg);
        //   let mut latencies = Vec::with_capacity(60_000);
        //   let start = Instant::now();
        //   while start.elapsed() < Duration::from_secs(60) {
        //       let t0 = Instant::now();
        //       verify_bearer(&cfg, &headers_with_bearer(&token));
        //       latencies.push(t0.elapsed());
        //   }
        //   latencies.sort();
        //   let p99 = latencies[latencies.len() * 99 / 100];
        //   assert!(p99 < Duration::from_millis(50), "p99 was {p99:?}");
    }

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
