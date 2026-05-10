# OAuth Provider Completion Audit

Date: 2026-05-10
Code/review audit scope: `006d984..ceeed5b`
Audit note: the follow-up documentation commit records this evidence; it does not change the reviewed Rust, GUI, script, or deployment behavior.

## Objective

Implement the goals in `docs/backlog/oauth-provider-l5.md` and run a complete code review cycle.

## Evidence Summary

Implemented and committed across the audit scope:

- Explicit HTTP auth policy: `loopback_only`, `bearer_required`, `oauth`, `public`.
- OAuth provider endpoints: metadata, protected-resource metadata, DCR, authorize, token, refresh, revoke.
- SQLite OAuth tables and signing key migration.
- OAuth bearer integration for `/mcp` and REST read-token routes.
- GUI owner approval and Connectors management page.
- OAuth GC and `rein doctor` integration.
- Remote deployment Recipe E and ADRs.
- End-to-end local OAuth script: `scripts/oauth-e2e-test.sh`.

Verification run on the reviewed implementation through `ceeed5b`:

- `cargo test --workspace --all-features` -> `1568 passed, 3 ignored`.
- `cargo clippy --workspace --all-targets -- -D warnings` -> clean.
- `cargo fmt --all -- --check` -> clean.
- `cargo audit` -> no vulnerabilities reported.
- `npm install && npm run lint && npm run build` in `crates/rein/gui` -> pass.
- `cargo build -p rein --release --features gui` -> pass.
- `./target/release/rein doctor` -> `Overall: healthy`.
- `./scripts/oauth-e2e-test.sh` -> `oauth e2e ok`.
- `REIN_DB=<temp-real-db> REIN_OAUTH_E2E_PORT=18731 ./scripts/oauth-e2e-test.sh && test ! -e <temp-real-db>` -> `oauth e2e ok` and no external DB file created.
- `scripts/oauth-live-readiness.sh http://127.0.0.1:8680` -> fails as expected on the current old runtime because the endpoint is not OAuth-ready.
- `REIN_EVAL_JUDGE=llm rein-eval synthesis baseline/run/compare` -> `SHIP (NonInferior)`.
- `codex review --uncommitted --title "v0.30 OAuth provider full audit after P2 fixes"` -> no blocking correctness, security, or maintainability issues found.
- `codex review --base 006d984402bfb9492a525075bb2fefac0a5b04eb --title "v0.30 OAuth provider final audit after session discovery hardening"` -> no discrete introduced correctness, security, or maintainability issues found; full Rust tests and OAuth e2e passed on this tree.
- `codex review --base 006d984402bfb9492a525075bb2fefac0a5b04eb --title "v0.30 OAuth provider final completion audit after review fixes"` -> found four P2 OAuth edge cases; fixed in `ceeed5b`.
- `codex review --base 006d984402bfb9492a525075bb2fefac0a5b04eb --title "v0.30 OAuth provider final completion audit after edge-case fixes"` -> no discrete correctness, security, or maintainability issues found; OAuth unit tests, cargo check, GUI build, and local OAuth e2e passed on this tree.
- Post-audit fresh check on HEAD `77c4551`: `./scripts/oauth-e2e-test.sh` -> `oauth e2e ok`.
- Live claude.ai/Cowork validation: switched the live service to `auth = "oauth"` behind Cloudflare Quick Tunnel `https://<your-tunnel-id>.trycloudflare.com`, `./scripts/oauth-live-readiness.sh` passed, Claude custom connector completed DCR + OAuth callback, and a fresh Claude conversation used the `rein` integration to answer the memory count as `5664`.

## Requirement Checklist

| Handoff requirement | Evidence | Status |
|---|---|---|
| Phase A config fields and auth policy derive | `crates/rein/src/config.rs`, `crates/rein/src/auth/policy.rs` | Done |
| Env token no silent override of explicit auth | `resolve_auth_policy`, doctor auth-policy checks, tests | Done |
| Deprecated `allow_unauthenticated_loopback` compatibility | config and doctor tests | Done |
| OAuth SQLite tables and signing key | `crates/rein/src/store/schema.rs` migration and tests | Done |
| OAuth authorization-server metadata | `crates/rein/src/auth/oauth/metadata.rs`, server routing | Done |
| Protected resource metadata | `metadata.rs`, `/mcp` challenge test/e2e | Done |
| Dynamic Client Registration | `register.rs`, `store.rs`, DCR tests and e2e | Done |
| Authorization endpoint and owner approval | `authorize.rs`, owner session/CSRF tests and e2e | Done |
| Token endpoint with PKCE S256 | `token.rs`, `pkce.rs`, RFC vector and negative tests | Done |
| Refresh rotation and replay detection | `store.rs`, `token.rs` tests | Done |
| OAuth bearer auth gate | `auth/policy.rs`, `mcp/server.rs`, `mcp/rest.rs`, e2e | Done |
| GUI Connectors page | `crates/rein/gui/src/pages/Connectors.tsx`, routes/nav | Done |
| Revocation endpoint | `revoke.rs`, e2e revoke and access-token 401 check | Done |
| DCR rate limit and durable client cap | `register.rs`, `store.rs` tests | Done |
| Security hardening: exact redirect URI, no fragments, bounded metadata, no client-secret logging, no-cache JSON, owner-token scoping before body reads, Secure cookies for HTTPS public URLs, advisory revocation hints, loopback metadata scheme handling, bounded DCR rate-limit map, hermetic e2e environment | OAuth module tests, e2e tests, and final code review | Done |
| OAuth GC | `auth/oauth/store.rs`, `ops/mod.rs`, doctor expired record count | Done |
| Doctor OAuth summary | `crates/rein/src/doctor.rs` | Done |
| Recipe E docs | `docs/manual/02b-remote-mcp-deployment.md` | Done |
| Auth-policy ADR | `docs/decisions/auth-policy-explicit.md` | Done |
| OAuth-provider ADR | `docs/decisions/oauth-provider.md` | Done |
| Local e2e script | `scripts/oauth-e2e-test.sh`, executable bit committed | Done |
| Live readiness check script | `scripts/oauth-live-readiness.sh`, executable bit committed | Done |
| Codex review convergence | Latest final HEAD review reported no blocking findings | Done |
| Implementation commits | `998804d feat(v0.30): add built-in OAuth provider`, `1fe2e83 test(v0.30): add OAuth live readiness check`, `4050f28 harden(v0.30): close OAuth public surface review gaps`, `f745115 test(v0.30): isolate OAuth e2e environment`, `dd8c1dd harden(v0.30): fix OAuth session and discovery edges`, `a5ff0d8 harden(v0.30): secure OAuth review response paths`, `ceeed5b harden(v0.30): close OAuth auth edge cases` | Done |
| Live claude.ai / Cowork connector validation | Cloudflare Quick Tunnel URL, Claude connector state `Configure`, and fresh Claude chat `<conversation-id>` answer the configured memory count after loading/using the rein integration | Done |
| Release tag (`v0.30.0`) | Operator explicitly said to push and release after final checks; phases were implemented together in the v0.30 audit scope | Ready for tag |
| Release notes | README / AGENTS / GitHub release notes document the v0.30 OAuth provider scope | Ready for release |
| Push / GitHub release | Operator explicitly said to push and release after final checks | Ready for release |

## Live Gate State

Current machine state after live validation:

- `tailscale funnel status` shows `https://<your-machine>.<your-tailnet>.ts.net` proxies `/` to `http://127.0.0.1:8680`.
- Tailscale Funnel was not used for validation because `curl -vkI https://<your-machine>.<your-tailnet>.ts.net/.well-known/oauth-authorization-server`
  from the current execution environment reached the local CGNAT-remapped address but failed TLS handshake with `SSL_ERROR_SYSCALL`, and an external reader could not resolve the `.ts.net` hostname.
- Cloudflare Quick Tunnel with HTTP/2 fallback exposed `http://127.0.0.1:8680`
  at `https://<your-tunnel-id>.trycloudflare.com`.
- `~/.rein/config.toml` was switched to `auth = "oauth"` and
  `public_url = "https://<your-tunnel-id>.trycloudflare.com"`.
- `GET http://127.0.0.1:8680/.well-known/oauth-authorization-server`
  returns OAuth metadata with the Cloudflare issuer.
- `./scripts/oauth-live-readiness.sh https://<your-tunnel-id>.trycloudflare.com`
  returns `oauth live readiness ok`.

The live claude.ai/Cowork gate was completed with Cloudflare Quick Tunnel because
the Tailscale Funnel hostname was not externally resolvable from the validation
path. The Cloudflare Quick Tunnel URL is ephemeral; a durable deployment should
use a named Cloudflare tunnel or a repaired public Funnel hostname before ship.

## Live Validation Runbook

This runbook was executed with a Cloudflare Quick Tunnel instead of the
Tailscale Funnel hostname, because the Tailscale hostname was not externally
reachable from the validation path.

1. Back up the active config:
   `cp ~/.rein/config.toml ~/.rein/config.toml.oauth-preflight-$(date +%Y%m%d%H%M%S).bak`.
2. Ensure an owner approval token is available in the service environment:
   `REIN_HTTP_TOKEN` must be set and non-empty.
3. Switch `[server]` to OAuth posture for the public hostname:
   `auth = "oauth"`, `public_url = "https://<public-hostname>"`,
   `sse_bind = "127.0.0.1"`, `sse_port = 8680`, and
   `allowed_hosts` containing the public hostname.
4. Stop the current listener on `127.0.0.1:8680` and start the OAuth-enabled
   build with GUI enabled.
5. Verify metadata before opening Claude:
   `curl -H 'Host: <your-machine>.<your-tailnet>.ts.net' http://127.0.0.1:8680/.well-known/oauth-authorization-server`
   must return JSON, and `./scripts/oauth-live-readiness.sh <public-url>` should
   pass from a network path that can validate the public TLS endpoint.
6. In claude.ai / Claude Connectors, add or reconnect the custom connector with
   MCP URL `https://<public-hostname>/mcp` and blank OAuth
   Client ID / Secret fields so Anthropic uses DCR.
7. Approve the browser `/oauth/authorize` page through the same public hostname.
8. Confirm the connector shows connected and a fresh conversation can ask rein
   to count memories.
9. If validation fails, restore the backed-up config and restart the previous
   service command.
