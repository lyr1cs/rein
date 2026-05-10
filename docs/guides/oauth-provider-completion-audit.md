# OAuth Provider Completion Audit

Date: 2026-05-10
Commit under audit: `998804d feat(v0.30): add built-in OAuth provider`

## Objective

Implement the goals in `docs/backlog/oauth-provider-l5.md` and run a complete code review cycle.

## Evidence Summary

Implemented and committed:

- Explicit HTTP auth policy: `loopback_only`, `bearer_required`, `oauth`, `public`.
- OAuth provider endpoints: metadata, protected-resource metadata, DCR, authorize, token, refresh, revoke.
- SQLite OAuth tables and signing key migration.
- OAuth bearer integration for `/mcp` and REST read-token routes.
- GUI owner approval and Connectors management page.
- OAuth GC and `rein doctor` integration.
- Remote deployment Recipe E and ADRs.
- End-to-end local OAuth script: `scripts/oauth-e2e-test.sh`.

Verification already run on the audited tree before commit:

- `cargo test --workspace --all-features` -> `1562 passed, 3 ignored`.
- `cargo clippy --workspace --all-targets -- -D warnings` -> clean.
- `cargo fmt --all -- --check` -> clean.
- `cargo audit` -> no vulnerabilities reported.
- `npm install && npm run lint && npm run build` in `crates/rein/gui` -> pass.
- `cargo build -p rein --release --features gui` -> pass.
- `./target/release/rein doctor` -> `Overall: healthy`.
- `./scripts/oauth-e2e-test.sh` -> `oauth e2e ok`.
- `REIN_EVAL_JUDGE=llm rein-eval synthesis baseline/run/compare` -> `SHIP (NonInferior)`.
- `codex review --uncommitted --title "v0.30 OAuth provider full audit after P2 fixes"` -> no blocking correctness, security, or maintainability issues found.

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
| Security hardening: exact redirect URI, no fragments, bounded metadata, no client-secret logging, no-cache JSON, owner-token scoping | OAuth module tests and code review | Done |
| OAuth GC | `auth/oauth/store.rs`, `ops/mod.rs`, doctor expired record count | Done |
| Doctor OAuth summary | `crates/rein/src/doctor.rs` | Done |
| Recipe E docs | `docs/manual/02b-remote-mcp-deployment.md` | Done |
| Auth-policy ADR | `docs/decisions/auth-policy-explicit.md` | Done |
| OAuth-provider ADR | `docs/decisions/oauth-provider.md` | Done |
| Local e2e script | `scripts/oauth-e2e-test.sh`, executable bit committed | Done |
| Codex review convergence | Latest review reported no blocking findings | Done |
| Implementation commit | `998804d feat(v0.30): add built-in OAuth provider` | Done |
| Live claude.ai / Cowork connector validation | Requires restarting the current public `:8680` service and operating the user's Claude connector | Blocked |
| Tag/push/release | Handoff section 11 says do not tag or push until the operator explicitly says ship | Intentionally not done |

## Live Gate State

Current machine state inspected after commit:

- `tailscale funnel status` shows `https://<your-machine>.<your-tailnet>.ts.net` proxies `/` to `http://127.0.0.1:8680`.
- `:8680` is currently served by PID `92942`: `/Users/<author>/.cargo/bin/rein serve --gui`.
- Current `~/.rein/config.toml` still uses the legacy public tunnel posture: `allow_unauthenticated_loopback = true` plus `allowed_hosts`.
- `GET http://127.0.0.1:8680/.well-known/oauth-authorization-server` currently returns GUI HTML, not OAuth metadata.

Therefore the live claude.ai/Cowork gate is not yet satisfied. Completing it requires operator approval to restart or replace the current public GUI process with the OAuth-enabled build and then use claude.ai/Claude Connectors to run the actual DCR + authorize + `/mcp` flow.

