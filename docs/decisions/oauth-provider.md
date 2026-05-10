# OAuth Provider For Remote MCP

## Status

Accepted for v0.30.

## Context

Anthropic remote custom connectors support public unauthenticated MCP endpoints
or OAuth. They do not provide a UI field for arbitrary static bearer tokens.
rein therefore needs a local OAuth 2.0 Authorization Server to support a secure
Cowork / claude.ai / mobile connector flow without relying on public no-auth
deployment.

## Decision

rein implements the OAuth pieces needed by MCP remote connectors:

- Authorization Server metadata
- Dynamic Client Registration
- Authorization Code with PKCE S256
- Refresh token rotation
- Revocation
- SQLite-backed clients, auth codes, grants, and HS256 signing keys
- Owner approval gated by the existing GUI/session token
  (`REIN_HTTP_TOKEN`) so public clients cannot self-approve access
- Public Dynamic Client Registration is durably capped; the store prunes the
  oldest clients that never received codes or grants before accepting more
  registrations.

The implementation remains single-user. Access tokens are root access to the
local rein server; no scopes, tenants, OIDC ID tokens, token introspection, or
JWKS endpoint are added.

## Alternatives

`oxide-auth` was not adopted because rein needs a small, auditable single-user
surface and custom SQLite grant semantics. HS256 is used instead of RS256
because tokens are issued and verified by the same local process, so publishing
public keys is unnecessary.

## Consequences

Remote MCP clients can register dynamically, open the local authorization page,
exchange an owner-approved authorization code for tokens, and call `/mcp` with
OAuth bearer tokens. Refresh tokens are one-time use; replay of a recognized
revoked or expired refresh token revokes the client's grant family.
