# Explicit HTTP Auth Policy

## Status

Accepted for v0.29/v0.30.

## Context

The previous HTTP auth model mixed two controls:

- `REIN_HTTP_TOKEN`
- `[server].allow_unauthenticated_loopback`

When both were present, the environment token silently won. That made the
effective runtime posture differ from the explicit config and caused remote MCP
connector failures that were hard to diagnose.

## Decision

`[server].auth` is now the single policy source:

- `loopback_only`
- `bearer_required`
- `oauth`
- `public`

`allow_unauthenticated_loopback` remains readable during the deprecation window
and is used only to derive a policy when `[server].auth` is absent. If an
explicit policy conflicts with `REIN_HTTP_TOKEN`, `rein doctor` warns, but the
environment variable does not change the policy. `auth = "oauth"` requires
`REIN_HTTP_TOKEN` only as the owner approval secret for the local authorization
page; OAuth clients do not send it to `/mcp`.

For compatibility with the pre-policy remote tunnel recipe,
`allow_unauthenticated_loopback = true` plus a loopback bind and no
`REIN_HTTP_TOKEN` derives the legacy public-read posture. Remote MCP reads are
allowed through an allowed host, while public mutation guards continue to block
state-changing REST and MCP calls. New configs should use `auth = "public"` for
that legacy posture or `auth = "loopback_only"` for strict local-only access.

`loopback_only` means direct local clients, not "anything that arrived through
a local reverse proxy". A request is treated as direct loopback only when the
TCP peer and Host are loopback, no forwarding headers are present, and the
server is not configured with an external `allowed_hosts` or `public_url`
surface. Public tunnels and reverse proxies should use `oauth`,
`bearer_required`, or an explicitly public read-only posture instead.

## Consequences

Operators can inspect one setting to know the auth posture. Recommended remote
MCP postures depend on deployment shape:

- **Private single-user tunnel** (non-discoverable Funnel/Tunnel hostname,
  read-only memory access from claude.ai, all writes happen locally over
  stdio): use `auth = "public"`. The v0.28.18 mutation gate continues to block
  `rein_store` / `rein_forget` / `rein_update` on non-loopback Hosts, so the
  residual threat model is read-only leak only, and there is no per-restart
  re-authorization cost. This is the documented posture in
  `docs/manual/02b-remote-mcp-deployment.md` for the operator's own
  deployment.
- **Multi-user, shared deployment, or any setup that needs claude.ai to call
  mutating tools**: use `auth = "oauth"`. The owner browser must hold the
  `rein_oauth_owner` cookie (10-minute window per authorize flow), and the
  v0.30.0 grant-table migration revokes pre-existing grants on first restart
  after upgrade — both are documented in 02b.
- **Strict local-only**: use `auth = "loopback_only"`.

`REIN_HTTP_TOKEN` remains supported for `bearer_required` and as the
OAuth-mode owner credential for GUI `/api/*` routes and approval pages, but it
does not authenticate `/mcp` when `auth = "oauth"` and it no longer silently
overrides explicit config.
