# Security

Rein is designed as a local-first memory system. It can expose HTTP, GUI, and
proxy surfaces when operators choose to run them, but the storage and security
model assume a single-user local deployment by default.

## Local-First Boundary

The primary data boundary is the local SQLite store and its side indexes. Rein
does not require a hosted Rein service, multi-tenant cloud account, or sync
backend to operate.

Remote providers are optional and feature-specific:

- Google Gemini can be used for embeddings and LLM tasks when `GEMINI_API_KEY`
  is configured.
- OMLX-compatible local endpoints can replace remote LLM or embedding calls for
  supported paths.
- Supermemory is only used when configured with `SUPERMEMORY_CC_API_KEY`.

If no remote provider is configured for a feature, Rein either falls back to
local rules where implemented or reports that the feature is unavailable.

## Auth And Sessions

Stdio MCP runs over a local process transport and does not use bearer tokens.
HTTP/SSE, GUI API routes, and Docker deployments use `REIN_HTTP_TOKEN`.

HTTP clients authenticate with:

```http
Authorization: Bearer <token>
```

The GUI can serve its static shell so a browser can bootstrap and show a token
input flow. API routes remain token-protected. The session clear route is
narrowly exempted so a browser can recover from a stale token cookie.

Proxy clients authenticate with:

```http
x-rein-token: <token>
```

Use `REIN_PROXY_TOKEN` for proxy clients. Proxy startup also accepts
`REIN_HTTP_TOKEN` as a fallback, but separate tokens are easier to rotate and
audit.

## Default-Deny Loopback

Unauthenticated loopback access is disabled by default for both the HTTP server
and proxy. To run without tokens on loopback, an operator must explicitly set
one of these local-only options:

```toml
[server]
auth = "loopback_only"   # strict local-only HTTP/SSE
# OR auth = "public"     # if non-discoverable tunnel hostname will route
                         # claude.ai connector reads through the bind

[proxy]
auth = "public"          # unauthenticated proxy; honored on loopback binds only
```

The `[server].auth` policy values are `"loopback_only"`, `"public"`,
`"bearer_required"`, and `"oauth"`. **v0.35.0 removed the legacy
`[server].allow_unauthenticated_loopback` bool.** Configs that still carry
it are handled by a load-time migration:

- bool true + loopback bind + no token → translated to `auth = "public"`.
- bool true + any bind + `REIN_HTTP_TOKEN` set → translated to
  `auth = "bearer_required"` (token auth always won over the bool at
  runtime; the migration preserves that).
- bool true + non-loopback bind + no token → the bool is stripped and a
  deprecation WARN is logged; `[server].auth` stays unset so stdio / CLI /
  `rein doctor` keep working. HTTP/SSE startup (`rein serve --sse`) then
  refuses with an explicit error pointing at `[server].auth` so the
  operator picks a policy.
- bool false (the old default) → silently stripped.

In every migration branch a one-time WARN is logged so the operator removes
the legacy key from their config.

**v1.2.0 applied the same treatment to the proxy surface**: the legacy
`[proxy].allow_unauthenticated_loopback` bool was removed in favor of an
explicit `[proxy].auth` policy (`"bearer_required"` or `"public"`). The
load-time migration mirrors the server mapping (token set →
`"bearer_required"`; loopback bind + no token → `"public"`; non-loopback +
no token → stripped with a WARN, and `rein proxy on` refuses to start until
a token or policy is set). `"public"` is honored only on loopback binds —
the proxy fronts provider credentials, so an unauthenticated non-loopback
listener is never inferred.

This opt-in only applies to loopback binds such as `127.0.0.1`, `localhost`, or
`::1`. Wildcard HTTP binds require a token or explicit host allowlist. Docker's
default command refuses to start without `REIN_HTTP_TOKEN`.

## Host And Origin Guard

The HTTP server and proxy validate Host and browser mutation context to reduce
DNS-rebinding and cross-origin mutation risk.

For HTTP/SSE:

- Specific binds derive their allowed host from the bind address.
- Wildcard binds require `REIN_HTTP_TOKEN` or `[server].allowed_hosts`.
- Browser-originating mutation requests are checked against Host/Origin rules.

For the proxy:

- Token-protected listeners accept authenticated clients.
- Unauthenticated loopback mode relies on Host/Origin guarding as the local
  DNS-rebinding defense.
- Path traversal segments are rejected before provider routing.

Tokens should be high entropy, kept out of shell history where practical, and
rotated when shared with another process or host.

## Record-Only Proxy

The proxy is a transparent recording surface. It forwards LLM API traffic to
the configured upstream and records eligible artifacts or text for later memory
extraction. It does not rewrite prompts, inject recalled memories, alter model
selection, or change upstream responses.

Extraction is policy-gated by route and configuration. First-party helper or
artifact routes can be mirrored without structured extraction. Queued extraction
work is handled by the same local worker and store model as hook ingestion.

## LLM Opt-In And Prompt Caps

LLM-backed features are explicit configuration choices. Examples include query
expansion, LLM reranking, intelligent merge verdicts, recall synthesis, concept
summary refresh, cold archival summaries, runtime LLM judge, and nightly judge
calibration.

Important defaults and controls:

- ARS synthesis and summary capabilities are gated by `[ars]` flags.
- Runtime judge behavior is gated by `[ars.llm_judge].enabled` and per-surface
  flags.
- Intelligent merge is gated by `[intelligent_merge].enabled`.
- Existing-memory injection into extraction prompts is opt-in through
  `[extract].inject_existing_context`.
- LLM consumers resolve provider, model, endpoint, API-key env var, and prompt
  cap through their own section, inherited `[llm]` defaults, or legacy section
  defaults.
- Prompt builders apply `max_input_chars` caps and route-specific truncation so
  large stores or long queries do not become unbounded LLM inputs.

Set providers to `none`, omit API keys, or keep feature flags disabled when a
deployment must avoid remote LLM calls.

## AGPL Network-Use Notice

Rein is licensed under AGPL-3.0-or-later. If you modify Rein and let users
interact with the modified program over a network, the AGPL network-use clause
requires you to offer those users the corresponding source code for your
modified version. This is a license compliance note, not legal advice.
