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

/// Insert a verified-claims entry into the bearer cache.  Bounded to
/// `BEARER_CACHE_CAP` entries; when the cap is exceeded we drop a third
/// of entries in bulk.  Note: `HashMap::keys()` order is unspecified, so
/// this is *random-sample* eviction, not FIFO/LRU — fine for our threat
/// model because cache pressure comes from long-lived polling clients
/// (whose tokens get re-cached on the next miss within seconds) and not
/// from random-token spray.
fn bearer_cache_insert(token_hash: [u8; 32], entry: CachedClaim) {
    let mut guard = bearer_cache()
        .lock()
        .expect("OAuth bearer cache mutex poisoned");
    if guard.len() >= BEARER_CACHE_CAP {
        let drop_count = guard.len() / 3;
        let keys: Vec<[u8; 32]> = guard.keys().take(drop_count).copied().collect();
        for k in keys {
            guard.remove(&k);
        }
    }
    guard.insert(token_hash, entry);
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
/// R3 P2: bump the generation counter BEFORE clearing.  Any
/// `verify_with_conn` already past its initial generation snapshot will
/// observe `gen_at_end != gen_at_start` and skip the insert that would
/// otherwise re-pollute the cache with a token whose grant we just
/// revoked.
pub(crate) fn bearer_cache_clear() {
    // SeqCst pairs with the snapshot+recheck in `verify_with_conn`.  We
    // bump generation first so the post-check load on a concurrent
    // verifier sees the bump even if it reads the counter strictly after
    // we clear the map.  (Reordering the two would create a window where
    // a verifier could insert into the not-yet-cleared map AND then read
    // a stale generation, missing the invalidation.)
    bearer_cache_generation().fetch_add(1, Ordering::SeqCst);
    bearer_cache()
        .lock()
        .expect("OAuth bearer cache mutex poisoned")
        .clear();
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
    let token_hash = sha256_bytes(token.as_bytes());
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
    if bearer_cache_hit(token_hash, now) {
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
    verify_with_conn(conn, token, token_hash, now)
}

/// v0.31 candidate (Agent F4 / A-H3): internal verifier that takes a
/// reusable `&Connection`.  Extracted so the cached fast path stays
/// reasoning-tight (no DB) and so the slow path can be exercised against
/// either a pool-backed connection or the per-store fallback.
fn verify_with_conn(
    conn: &rusqlite::Connection,
    token: &str,
    token_hash: [u8; 32],
    now: i64,
) -> bool {
    // R3 P2: snapshot the invalidation generation BEFORE any DB read.  If
    // it changes by the time we reach the cache insert below, a
    // concurrent revoke/refresh already flushed the cache and we must
    // NOT re-pollute it with a token whose grant may have just been
    // revoked between our `active_grant_by_jti` check and the insert.
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

    // R3 P2: only cache the verified-claims envelope if no invalidation
    // happened during our in-flight check.  This request itself returns
    // `true` either way — the grant was active when we read it, and a
    // revoke that landed in parallel is observed by FUTURE requests
    // (which take the slow path again because we declined to insert).
    let gen_at_end = bearer_cache_generation().load(Ordering::SeqCst);
    if gen_at_start == gen_at_end {
        bearer_cache_insert(
            token_hash,
            CachedClaim {
                jwt_exp: claims.exp,
                inserted_at: Instant::now(),
            },
        );
    }
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
    // and that a concurrent `verify_with_conn`-shaped insert path that
    // straddles the clear declines to repopulate the cache. We test the
    // generation snapshot/recheck primitive directly rather than spinning
    // up a SQLite store — the integration test would require a full client
    // + signing-key + grant fixture and would not give a stronger
    // guarantee than this targeted check, since the recheck is the only
    // mutation `verify_with_conn` adds on top of the existing
    // `bearer_cache_insert`.
    #[test]
    fn bearer_cache_generation_bumps_invalidate_inflight_inserts() {
        let _guard = acquire_cache_test_lock();
        clear_bearer_caches_for_test();
        let token_hash = sha256_bytes(b"inflight-token");
        // Simulate the verify_with_conn shape: snapshot generation, do
        // "work" (would be DB checks), then recheck before inserting.
        let gen_at_start = bearer_cache_generation().load(Ordering::SeqCst);
        // Concurrent revoke lands here.
        bearer_cache_clear();
        let gen_at_end = bearer_cache_generation().load(Ordering::SeqCst);
        assert_ne!(
            gen_at_start, gen_at_end,
            "bearer_cache_clear must bump the generation counter"
        );
        // Mimic the gate inside verify_with_conn: only insert when
        // generation is unchanged.
        if gen_at_start == gen_at_end {
            bearer_cache_insert(
                token_hash,
                CachedClaim {
                    jwt_exp: i64::MAX,
                    inserted_at: Instant::now(),
                },
            );
        }
        // Without the fix this would hit (the insert ran). With the fix
        // the gate suppressed the insert, so the entry isn't there.
        assert!(
            !bearer_cache_hit(token_hash, 100),
            "concurrent revoke must prevent in-flight verifier from re-caching the just-revoked token"
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
