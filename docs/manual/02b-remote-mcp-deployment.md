# Remote MCP deployment (Claude Desktop Cowork, claude.ai, Mobile)

This chapter covers exposing rein as a **remote MCP server** so it can be
used by Claude clients that route MCP traffic through Anthropic's cloud
rather than local stdio. These clients include:

- **Claude Cowork** (the agentic-work tab inside Claude Desktop)
- **claude.ai** (web)
- **Claude mobile** (iOS / Android)

These clients are **not** reachable through the install paths in
[02-installation.md](02-installation.md) (DXT, plugin marketplace,
`claude_desktop_config.json` stdio entry). Per Anthropic's help
documentation:

> "Local MCP servers configured in Claude Desktop via
> `claude_desktop_config.json` are a separate mechanism and do use your
> local network, but those aren't available in Cowork or claude.ai."
> Remote connectors, by contrast, "work across every Claude client" and
> "the connection originates from Anthropic's servers, not from your
> machine's network interface."

So Cowork / claude.ai / mobile need a **public HTTPS endpoint** that
Anthropic's cloud can reach. This guide walks through three deployment
recipes plus the Claude UI configuration step.

If you only use Claude Desktop's Chat tab or Claude Code CLI, this
chapter is optional — those clients already work via the local-stdio
paths in 02-installation.md.

## Architecture

```
[Claude Cowork UI / claude.ai / mobile]
            │
            │ HTTPS (Anthropic cloud → your endpoint)
            ▼
[Public HTTPS endpoint]    ← Cloudflare Tunnel / Tailscale Funnel /
            │                Caddy + Let's Encrypt
            │ HTTP (loopback)
            ▼
[rein serve --sse]         ← actually serves Streamable HTTP at /mcp
            │
            ▼
[~/.rein/memories.db]      ← same DB shared with all local rein clients
```

rein's HTTP server (started via `rein serve --sse`) listens on
`127.0.0.1:8680/mcp` by default and implements the Streamable HTTP
transport (current MCP standard since spec version 2025-06-18). The
deployment work is to terminate HTTPS in front of it and route
authenticated traffic through.

### One rein memory, two Claude transports

The important mental model is **not** "install rein twice." It is one
private rein state directory, normally `~/.rein/`, exposed through two
different MCP transports:

- **Local Claude paths** — Claude Desktop Chat, Claude Code, and local
  plugin/DXT installs start `rein serve` over stdio. Traffic stays on
  the same machine. No public URL is involved.
- **Remote Claude paths** — Claude Cowork, claude.ai, and mobile do not
  run local stdio servers. Their connector traffic originates from
  Anthropic's cloud, so they need a public HTTPS URL that forwards back
  to the local rein HTTP listener.

Both paths read and write the same SQLite database, side indexes, queue,
and config under the same `HOME` / `REIN_DB` / `REIN_CONFIG`. The
transport differs; the memory source of truth does not.

```mermaid
flowchart LR
  subgraph Local["Your Mac / always-on host"]
    DB[("~/.rein/memories.db")]
    Queue[("~/.rein/queue + side indexes")]
    Stdio["rein serve\nstdio MCP"]
    HTTP["rein serve --gui\nHTTP MCP + OAuth\n127.0.0.1:8680"]
    Tunnel["Public HTTPS tunnel\nTailscale Funnel / Cloudflare / Caddy"]
  end

  ClaudeLocal["Claude Desktop Chat\nClaude Code\nlocal plugin / DXT"]
  ClaudeRemote["Claude Cowork\nclaude.ai\nmobile"]
  Anthropic["Anthropic cloud connector"]

  ClaudeLocal -->|"local stdio MCP"| Stdio
  Stdio --> DB
  Stdio --> Queue

  ClaudeRemote -->|"connector request"| Anthropic
  Anthropic -->|"HTTPS https://public-host/mcp"| Tunnel
  Tunnel -->|"loopback HTTP http://127.0.0.1:8680/mcp"| HTTP
  HTTP --> DB
  HTTP --> Queue
```

This split is deliberate:

- **Consistency** — a memory stored from Claude Code is immediately
  visible to Cowork, and a memory created from claude.ai is visible to
  Desktop Chat, because both surfaces hit the same database.
- **No database sync problem** — rein stays single-writer /
  single-owner. You do not need to rsync `memories.db` between "Claude"
  and "Cowork" machines.
- **Correct network boundary** — only the tunnel is public. rein itself
  still listens on `127.0.0.1`, so random LAN hosts cannot connect
  directly to the HTTP server.
- **Claude compatibility** — local Claude clients can use stdio because
  they run on your machine; Cowork / claude.ai cannot, because their MCP
  connector runs in Anthropic's cloud.

### Why `127.0.0.1` and `public_url` are both required

`127.0.0.1` is the **listen/upstream address**. It is where rein accepts
traffic from the tunnel process running on the same machine.

`public_url` is the **external identity**. It is what Claude sees, what
rein advertises in OAuth metadata, and what must appear in the
connector URL. These two values must not be collapsed into one setting:

```mermaid
flowchart LR
  Claude["Claude.ai / Cowork"]
  Public["public_url\nhttps://<your-machine>.<your-tailnet>.ts.net"]
  Funnel["Tailscale Funnel\nTLS + public routing"]
  Loopback["sse_bind = 127.0.0.1\nsse_port = 8680"]
  Rein["rein HTTP server\n/mcp /oauth/* /.well-known/*"]

  Claude -->|"uses https://.../mcp"| Public
  Public --> Funnel
  Funnel -->|"proxy http://127.0.0.1:8680"| Loopback
  Loopback --> Rein
```

If you put `127.0.0.1` into Claude's connector URL, Anthropic's cloud
tries to connect to **Anthropic's own localhost**, not your Mac. That
cannot work. `127.0.0.1` belongs only on the local side of the tunnel.

For v0.30+ OAuth deployments the minimal shape is:

```toml
[server]
auth = "oauth"
gui_enabled = true
sse_enabled = true
sse_bind = "127.0.0.1"
sse_port = 8680
public_url = "https://<your-machine>.<your-tailnet>.ts.net"
allowed_hosts = ["<your-machine>.<your-tailnet>.ts.net"]
```

`allowed_hosts` is not cosmetic. rein validates the incoming `Host:`
header before routing requests. Tailscale Funnel preserves the public
FQDN as the Host header, so that FQDN must be listed or rein returns
`403 Host header is not allowed`.

### OAuth flow when Cowork shares the same rein

In v0.30+ the recommended remote path is built-in OAuth. Claude does
not receive your static `REIN_HTTP_TOKEN`. Instead, `REIN_HTTP_TOKEN`
is the owner approval credential for the GUI, while Claude receives its
own OAuth access and refresh tokens after you approve the connector.

```mermaid
sequenceDiagram
  autonumber
  participant C as Claude Cowork / claude.ai
  participant A as Anthropic connector service
  participant T as Public HTTPS tunnel
  participant R as rein HTTP/OAuth server
  participant O as Owner browser
  participant D as ~/.rein/memories.db

  C->>A: Add connector: https://public-host/mcp
  A->>T: GET /.well-known/oauth-authorization-server
  T->>R: Forward to 127.0.0.1:8680
  R-->>A: issuer, authorize/token/register endpoints
  A->>R: POST /oauth/register
  R-->>A: client_id (+ optional client_secret)
  A->>O: Open /oauth/authorize in browser
  O->>R: Owner login with REIN_HTTP_TOKEN if needed
  O->>R: Approve connector access
  R-->>A: authorization code
  A->>R: POST /oauth/token with PKCE verifier
  R-->>A: access_token + refresh_token
  A->>R: POST /mcp with Authorization: Bearer access_token
  R->>D: read/write same memory store as local stdio clients
  R-->>A: rein_* tool results
  A-->>C: Tool result in Cowork / claude.ai
```

The same machine can therefore serve both surfaces at once:

- Local Claude continues to use stdio for low-latency local workflows.
- Cowork / claude.ai uses the public `/mcp` URL plus OAuth for remote
  access.
- Both surfaces converge on the same `~/.rein/memories.db`.

### Why this design is preferable to the alternatives

| Alternative | Why it looks attractive | Why rein avoids it |
| --- | --- | --- |
| Run a separate rein for Cowork | Keeps remote setup isolated | Creates two memory databases that drift immediately |
| Put `127.0.0.1` in Claude.ai | Matches local curl tests | Claude.ai runs in Anthropic's cloud; its localhost is not your Mac |
| Expose rein directly on `0.0.0.0` | Removes the tunnel process | Increases LAN/public attack surface and makes Host guard mistakes costly |
| Use Cloudflare/Quick Tunnel random URLs forever | Fast demo | URL changes or dies, forcing connector re-registration |
| Reuse `REIN_HTTP_TOKEN` as Claude bearer | Simple mental model | Anthropic's connector UI has no arbitrary bearer header field |

The durable pattern is: **one rein state directory, local loopback
listener, stable public HTTPS URL, OAuth for remote Claude clients**.

## What you need before starting

- rein v0.30.0 or newer for the built-in OAuth provider (`rein
  --version`). Older v0.28.9+ builds can expose remote MCP, but only
  with the legacy URL-only / proxy-auth recipes.
- `GEMINI_API_KEY` configured in `~/.rein/config.toml` or shell env
- A **Pro / Max / Team / Enterprise** Claude account (custom connectors
  are not available on Free)
- A way to expose `127.0.0.1:8680` to the public internet — see the
  decision tree below to pick the cheapest one for your situation

## Choosing your auth posture

`[server].auth` controls how rein gates incoming HTTP / `/mcp` requests.
Pick before choosing a recipe — the recipe just exposes the listener;
auth is what the broker negotiates against.

| Posture | When it fits | Anthropic UI auth | Recovery cost on rein restart |
| --- | --- | --- | --- |
| `public` | Single-user, private tunnel hostname, read-only memory access from claude.ai is the goal (writes happen locally over stdio anyway). v0.28.18's mutation gate still blocks `rein_store` / `rein_forget` / `rein_update` on non-loopback Hosts, so the threat model is read-only leak only. | None | None — claude.ai re-uses the same connector forever |
| `oauth` | Multi-user, shared deployment, or a setup where you want claude.ai to also call mutating tools. Requires the operator's browser to set the `rein_oauth_owner` cookie via `POST /api/session` before each authorize flow (10-minute window). | OAuth 2.0 (PKCE + DCR) | Connector usually re-authorizes silently; the v0.30.0 release-day migration revokes existing grants and forces a manual disconnect+reconnect on claude.ai |
| `bearer_required` | stdio-friendly token auth for tooling that can inject `Authorization: Bearer <REIN_HTTP_TOKEN>`. Anthropic's UI has no field for this — useful for direct API consumers (Codex CLI, custom scripts), not the claude.ai connector. | n/a | None |
| `loopback_only` | Local-only deployment; rein refuses any non-loopback request. Choose this when the only consumers are stdio MCP and same-machine REST. | n/a | None |

**For most personal Tailscale Funnel deployments**, `public` is the
simplest path: one config line, no cookie windows, no migration tax,
and the mutation gate keeps writes locked. Set it explicitly so doctor
isn't ambiguous:

```toml
[server]
auth = "public"
```

**For shared deployments or full read+write claude.ai integration**,
go through Recipe E (built-in OAuth) below. Be aware the connector
breaks once during a v0.30.x upgrade because the
`refresh_token_fingerprint` schema migration revokes pre-upgrade
grants — recover by removing and re-adding the connector on claude.ai
once. `rein doctor` flags this state with a one-line WARN and an
actionable hint.

## Choosing a recipe

You don't need a Tailscale account, a Cloudflare account, a domain, or a
VPS to use rein in Cowork. You just need **some** way to give your
local rein a public HTTPS URL. The recipes below are ranked by friction
and trade-offs; pick the first one whose prerequisites you already
meet.

| You have… | Recipe | Setup time | URL stability | Auth options |
| --- | --- | --- | --- | --- |
| **Nothing** (no account, no domain) | **A.0: Cloudflare Quick Tunnel** | ~1 min | Ephemeral (changes per restart) | URL obscurity only |
| ngrok account (free) | **D: ngrok** | ~3 min | Ephemeral on free tier; reserved domains on paid | URL obscurity only |
| Tailscale account (free) | **B: Tailscale Funnel** | ~5 min | Permanent (`*.ts.net`) | URL obscurity, or rein OAuth via Recipe E |
| Cloudflare account + own domain | **A: Cloudflare Tunnel** | ~15 min | Permanent under your domain | Cloudflare Access OIDC, or rein OAuth via Recipe E |
| VPS + own domain | **C: Caddy / nginx + Let's Encrypt** | ~30 min | Permanent | Anything (basic_auth, OIDC, mTLS), or rein OAuth via Recipe E |
| rein v0.30+ | **E: Built-in OAuth** | ~5 min after HTTPS exposure | Depends on tunnel | OAuth Authorization Code + PKCE |

**Network gotcha noted across recipes:** several routers and corporate
proxies intercept and remap `*.trycloudflare.com`, `*.ts.net`,
`*.ngrok-free.app` traffic — typically to private CGNAT space like
`198.18.0.0/15`. Some home LANs do this. The
operational symptoms differ per recipe (Quick Tunnel: cloudflared can't
establish QUIC tunnel; Tailscale: local curl returns `[HTTP 000]` but
the public path still works; ngrok: similar). Each recipe below
documents how to detect and work around its specific manifestation.

## Recipe A.0: Cloudflare Quick Tunnel (no account, no domain)

The lowest-friction option. `cloudflared tunnel --url` provisions a
random `*.trycloudflare.com` URL on Cloudflare's free quick-tunnel
service. No Cloudflare account needed; no domain needed. You install
`cloudflared`, run one command, get a URL.

### Trade-offs

- **No auth layer** — `*.trycloudflare.com` URL is the only secret.
  Anyone who learns it has rein access. Pair with rein's
  `auth = "public"` only if your memory
  database doesn't contain anything sensitive.
- **URL changes every time `cloudflared` restarts.** You'll re-edit
  `[server].allowed_hosts` in `~/.rein/config.toml` and re-add the
  custom connector in Claude Desktop on every reboot. Painful for
  daily use; fine for "I want to try Cowork once before signing up
  for anything."
- **Cloudflare reserves the right to throttle / cut Quick Tunnels**
  at any time — they're explicitly not for production. Quote from
  Cloudflare's own banner: "these account-less Tunnels have no
  uptime guarantee."
- **Requires QUIC outbound to Cloudflare's edge** — many corporate
  networks and some consumer routers block QUIC (UDP/443). When this
  happens, `cloudflared` logs `Failed to dial a quic connection
  error="failed to dial to edge with quic: timeout: no recent network
  activity"` repeatedly. Add `--protocol http2` to fall back to TCP;
  if HTTP/2 also fails, your network is too restrictive for Quick
  Tunnel — switch to Tailscale Funnel (Recipe B) which uses
  WireGuard and tends to work where QUIC is blocked.

### Steps

```bash
# 1. Install cloudflared (no Cloudflare account needed for Quick Tunnel)
brew install cloudflared        # macOS
# Linux: see https://pkg.cloudflare.com/index.html

# 2. Make sure rein is up with auth = "public".
#    See Recipe B Step 2-3 for the config rationale. Use `env -u
#    REIN_HTTP_TOKEN` (NOT just `unset`) so the inheritance is blocked
#    even when this command runs from a shell that has the token
#    exported — `unset` is shell-local and a forked nohup process
#    can still inherit a token set in a parent shell that hadn't
#    seen the unset.
REIN_LOG=info env -u REIN_HTTP_TOKEN \
    nohup rein serve --sse > /tmp/rein-sse.log 2>&1 &

# 3. Start the Quick Tunnel. --protocol http2 falls back to TCP if
#    QUIC is blocked on your network.
nohup cloudflared tunnel --protocol http2 --url http://localhost:8680 \
    > /tmp/cloudflared.log 2>&1 &

# 4. Wait ~10s, then read the assigned URL from the log:
grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' /tmp/cloudflared.log | head -1
# → e.g., https://<your-tunnel-id>.trycloudflare.com

# 5. Add that hostname to ~/.rein/config.toml:
#      [server]
#      auth = "public"
#      allowed_hosts = ["<your-tunnel-id>.trycloudflare.com"]
#    Restart rein after editing (kill the rein serve --sse process and
#    relaunch it; config is read at startup, not hot-reloaded).

# 6. In Claude Desktop: Customize → Connectors → "+" → "Add custom
#    connector" → URL = "https://<the-trycloudflare-hostname>/mcp",
#    OAuth fields blank, Add.
```

When you're done, kill `cloudflared` (`pkill -f 'cloudflared tunnel'`).

### When Quick Tunnel fails on a restrictive network

Symptoms in `/tmp/cloudflared.log`:

```
ERR Failed to dial a quic connection error="failed to dial to edge with quic: timeout: no recent network activity"
INF Retrying connection in up to 2s
ERR Failed to dial a quic connection ...
```

This means cloudflared can't reach Cloudflare's edge. Likely causes
(in order of frequency):

1. **Network blocks QUIC outbound on UDP/443** (corporate firewalls,
   some consumer routers, school networks). Mitigation: add
   `--protocol http2` to the cloudflared command. Falls back to TCP.
2. **Network intercepts Cloudflare IPs and remaps to private CGNAT
   space** (many home LANs do this — `198.18.x.x`
   resolution for Cloudflare hostnames). Even HTTP/2 fallback won't
   help; cloudflared traffic gets black-holed at the local proxy.
   Mitigation: switch to Tailscale Funnel (Recipe B). Tailscale uses
   WireGuard which tends to work where Cloudflare protocols are
   blocked or intercepted.
3. **Genuine Cloudflare outage** (rare; check
   <https://www.cloudflarestatus.com>).

If `--protocol http2` doesn't fix it within 30 seconds, give up on
Quick Tunnel and use Recipe B (Tailscale Funnel) — it has a free tier
with permanent URLs and is more network-tolerant.



## Recipe A: Cloudflare Tunnel with your own domain (production / OIDC)

This is Recipe A.0 with a permanent URL under a domain you own and the
option to put Cloudflare Access OIDC in front for real edge auth.
Requires a Cloudflare account and a domain on Cloudflare.

Unlike Quick Tunnel above, the URL is permanent (you choose it), and
you can attach Cloudflare Access policies for identity-based auth — the
only OAuth path that maps cleanly to Anthropic's connector OAuth
fields.

### Step 1 — Start rein in HTTP mode

rein refuses to start an HTTP listener with no auth by default. There are
two ways to satisfy that constraint, depending on whether you want
defense-in-depth bearer auth at the rein layer or whether you accept
that the tunnel-only-reachable loopback bind is enough.

**Option 1a (simpler, recommended for personal use):** opt into
unauthenticated loopback. rein binds to `127.0.0.1` only — the only
remote process that can reach it is `cloudflared` running on the same
machine and forwarding from your tunnel. Edit `~/.rein/config.toml`:

```toml
[server]
auth = "public"
# Required: rein validates the incoming Host header. Cloudflare Tunnel
# forwards the public hostname (e.g. rein.your-domain.com), which is
# NOT in the default localhost-only allowlist. Add every hostname that
# clients will use to reach rein.
allowed_hosts = ["rein.your-domain.com"]
```

If `allowed_hosts` is omitted, rein returns `403 Forbidden — Host
header is not allowed` for any request whose `Host:` is not localhost
/ 127.0.0.1 / ::1, and your tunnel will appear broken.

Then start rein:

```bash
unset REIN_HTTP_TOKEN
rein serve --sse &
```

**Option 1b (defense-in-depth):** keep bearer auth on, and inject the
bearer header at the proxy layer (Caddy / Cloudflare Worker / nginx)
so requests arriving at rein already carry it. The `allowed_hosts`
constraint above still applies — set it in `~/.rein/config.toml`
even when using bearer auth.

```bash
export REIN_HTTP_TOKEN="$(openssl rand -hex 32)"   # any strong secret
rein serve --sse &
```

Then in your tunnel config (Recipe C below covers Caddy specifically),
make the proxy add `Authorization: Bearer $REIN_HTTP_TOKEN` to every
forwarded request. Anthropic's connector UI does not have an arbitrary-
bearer field, so this header **must** be added by the proxy, not by
the connector itself.

Most users following Cloudflare Tunnel below should pick **Option 1a**.
Pick Option 1b only if you have specific operational requirements for
end-to-end bearer auth.

Verify rein is up:

```bash
curl -fsS http://127.0.0.1:8680/mcp \
    -H 'Accept: application/json, text/event-stream' \
    -H 'Content-Type: application/json' \
    -d '{"jsonrpc":"2.0","id":1,"method":"initialize",
         "params":{"protocolVersion":"2025-06-18","capabilities":{},
                   "clientInfo":{"name":"curl","version":"1.0"}}}' | head -5
```

You should get back a JSON or SSE response with `"result":` and a
`Mcp-Session-Id` header. If the request is rejected with 401, see
"Authentication" below.

### Step 2 — Install cloudflared and create the tunnel

```bash
brew install cloudflared
cloudflared tunnel login                       # opens browser for auth
cloudflared tunnel create rein                 # creates tunnel, saves credentials
```

This prints a tunnel UUID and credential JSON path. Note the UUID.

### Step 3 — Configure the tunnel

Create `~/.cloudflared/config.yml`:

```yaml
tunnel: <UUID-from-step-2>
credentials-file: /Users/<you>/.cloudflared/<UUID>.json

ingress:
  - hostname: rein.<your-domain>.com
    service: http://127.0.0.1:8680
  - service: http_status:404
```

Add a CNAME in your Cloudflare DNS pointing `rein.<your-domain>.com` to
the tunnel:

```bash
cloudflared tunnel route dns rein rein.<your-domain>.com
```

### Step 4 — Run the tunnel

Foreground (testing):

```bash
cloudflared tunnel run rein
```

Background (macOS launchd):

```bash
sudo cloudflared service install      # registers launchd job
```

Verify from anywhere on the internet:

```bash
curl -fsS https://rein.<your-domain>.com/mcp \
    -H 'Accept: application/json, text/event-stream' \
    -H 'Content-Type: application/json' \
    -d '{"jsonrpc":"2.0","id":1,"method":"initialize",
         "params":{"protocolVersion":"2025-06-18","capabilities":{},
                   "clientInfo":{"name":"curl","version":"1.0"}}}'
```

### Step 5 (optional) — Edge auth tradeoffs

Without an extra auth layer, anyone who learns the tunnel URL can call
rein. There is no clean Anthropic-connector-compatible way to put
bearer-only auth in front of the tunnel today, because the connector
UI has no arbitrary-bearer field — it only takes a URL and (optionally)
**OAuth Client ID + Secret** that perform a standard OAuth flow.

Three practical postures:

**Posture 1 — URL obscurity (simplest, what most personal users do):**
Use a Cloudflare Tunnel hostname under your own domain that is **not**
listed in any public DNS index. Tunnel hostnames you create via
`cloudflared tunnel route dns` go in your zone, but obscurity is not
zero security — anyone who learns the URL has access. Pair with
loopback-bound rein (Option 1a above) so the only externally reachable
attack surface is the tunnel URL itself. Acceptable for personal use
when the memory database doesn't contain secrets.

**Posture 2 — Cloudflare Access with an OIDC identity provider:**
Cloudflare Access supports OIDC identity providers (Google, GitHub,
Microsoft, etc., free tier). Configured this way, Access intercepts
Anthropic's request, runs an OIDC flow against your IdP, and only
forwards once authenticated. Anthropic's connector OAuth Client ID /
Secret fields can be used here, with the catch that Cloudflare Access
must be configured to accept Anthropic's OAuth flow as a valid
inbound identity. This is the cleanest "real auth" option but
requires careful Access policy configuration. See Cloudflare's
"Self-hosted application via OIDC" docs.

> **Do NOT use Cloudflare Access service tokens for this.** Service
> tokens (`CF-Access-Client-Id` + `CF-Access-Client-Secret` headers)
> are *not* OAuth credentials. Anthropic's connector field labeled
> "OAuth Client ID / Secret" performs a standard OAuth handshake and
> won't send the Cloudflare-specific headers, so a service-token
> setup will fail at the connector handshake. This is a real
> compatibility gap; if you've configured Cloudflare Access with a
> service token expecting it to work like OAuth, switch to an OIDC
> IdP-based Access policy instead.

**Posture 3 — OAuth-aware MCP gateway in front of rein:**
Run a separate process (e.g., Cloudflare Workers OAuth Provider, or
the npm `mcp-remote` proxy) that implements full OAuth 2.0 / OIDC and
proxies authenticated requests to local rein. More moving parts, but
gives you a proper OAuth flow that matches what Anthropic's connector
does. Out of scope for rein itself; rein has no built-in OAuth
provider.

For most personal users, **Posture 1** is the pragmatic default.
Switch to Posture 2 or 3 if your memory database holds anything
sensitive and you need authenticated access enforced.

### Cloudflare Tunnel pros / cons

- **Pro**: free, no public IP, no port forwarding, automatic HTTPS,
  optional identity-based auth.
- **Con**: tunnel goes down if your laptop sleeps or loses network. For
  always-on rein, run `cloudflared` on a VPS or always-on machine
  pointing at a remote rein.

## Recipe B: Tailscale Funnel

Simpler than Cloudflare for users who already use Tailscale.

```bash
brew install tailscale
sudo tailscale up
sudo tailscale funnel 8680                 # exposes :8680 publicly via *.ts.net
```

Tailscale prints a URL like `https://<machine>.<tailnet>.ts.net`. The
MCP endpoint becomes `https://<machine>.<tailnet>.ts.net/mcp`. HTTPS is
automatic; no DNS / cert work.

**Limitation — Tailscale Funnel does not provide an auth layer by
itself.** Tailscale gives you a stable public HTTPS route; rein still
has to decide how the remote caller authenticates. On v0.30+ the
recommended answer is **Recipe E: built-in OAuth** on top of the
Funnel URL. In that posture, Funnel only transports HTTPS traffic and
rein's OAuth provider protects `/mcp`.

For pre-v0.30 builds, or when you intentionally choose not to use
built-in OAuth, Anthropic's connector still cannot pass arbitrary
bearer tokens (no UI field), and rein's `REIN_HTTP_TOKEN` won't reach
rein through the connector. The legacy options with Tailscale Funnel
are:

1. **Accept unauthenticated public exposure** — set
   `auth = "public"` in `~/.rein/config.toml`
   (loopback bind only), AND add the funnel hostname to
   `allowed_hosts` (e.g.,
   `allowed_hosts = ["<machine>.<tailnet>.ts.net"]` — without this
   rein will 403 every Funnel request, see "Host header" caveat in
   Recipe A Step 1), and rely on the `*.ts.net` URL not being
   listed in any public index. Anyone who learns the URL has full
   access to your memory database. Acceptable only for memory
   stores that don't contain sensitive data.
2. **Run an auth/injection proxy in front of rein** — e.g., a local
   Caddy listening on 127.0.0.1:8443 that injects an upstream bearer
   header (see Recipe C Caddy pattern) and have Tailscale Funnel
   point at the Caddy port instead of `:8680`. The Caddy itself then
   needs an OIDC plugin / `basic_auth` / etc. for incoming auth.

If neither legacy option fits and you cannot upgrade to v0.30+, switch
to Recipe A (Cloudflare Tunnel + an OIDC IdP via Cloudflare Access)
instead — that path has a real edge-auth story outside rein.

### Verified end-to-end walkthrough (macOS Apple Silicon, 2026-05-06)

This is the exact command sequence that brought up a working Cowork
connector during v0.28.11 development. Every step was executed; every
output was observed. Use it as a copy-paste recipe; substitute your
own FQDN where shown.

**Step 1 — Confirm Tailscale is logged in and discover your FQDN.**

```bash
# Quick inventory — should show your Mac in the tailnet, status "active"
# or "-" (idle).
tailscale status | head -5

# Discover the exact FQDN Funnel will use. The Self.DNSName field is
# the source of truth (trailing dot is just FQDN canonical form;
# Funnel and Anthropic both accept it without the trailing dot).
tailscale status --json \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['Self']['DNSName'].rstrip('.'))"
# → e.g., <your-machine>.<your-tailnet>.ts.net
```

If the second command prints empty, you haven't run `sudo tailscale up`
yet (or your tailnet name hasn't been generated). Run `sudo tailscale
up` and retry.

**Step 2 — Edit `~/.rein/config.toml`.**

Add or amend the `[server]` section so it has BOTH the explicit
`auth = "public"` policy AND a matching `allowed_hosts`:

```toml
[server]
auth = "public"
allowed_hosts = ["<your-machine>.<your-tailnet>.ts.net"]   # ← your FQDN
```

The `allowed_hosts` entry is mandatory. Without it rein returns
`403 Host header is not allowed` for every Funnel request, even with
`auth = "public"` set. The default Host allowlist is localhost-only;
the Funnel forwards `Host:` set to your public FQDN, which doesn't
match.

**Step 3 — Start `rein serve --sse` with verbose logging.**

```bash
# With explicit `auth = "public"`, an inherited REIN_HTTP_TOKEN does
# NOT switch the server to bearer-required — `resolve_auth_policy`
# honors the explicit policy and emits a startup warning that the
# token has no effect. So unlike the legacy bool-based posture, you
# don't strictly need `env -u REIN_HTTP_TOKEN` here. It's still good
# hygiene if you don't want a stray bearer log line, and `env -u` is
# left in the command below for that reason; if you skip it, claude.ai
# requests will still work.
#
# REIN_LOG=info is set so the server emits per-request lines for the
# initialize / tools / call traffic that Anthropic's connector
# generates — without it, rein defaults to `warn` and successful
# requests are silent in the log, making `tail -F /tmp/rein-sse.log`
# useless as a "did Anthropic reach me?" signal during connector setup.
REIN_LOG=info env -u REIN_HTTP_TOKEN \
    nohup rein serve --sse > /tmp/rein-sse.log 2>&1 &
```

Verify it bound to loopback:

```bash
cat /tmp/rein-sse.log
# Expect: "rein HTTP server listening on http://127.0.0.1:8680/mcp"

curl -s -X POST http://127.0.0.1:8680/mcp \
    -H 'Accept: application/json, text/event-stream' \
    -H 'Content-Type: application/json' \
    -d '{"jsonrpc":"2.0","id":1,"method":"initialize",
         "params":{"protocolVersion":"2025-06-18","capabilities":{},
                   "clientInfo":{"name":"loopback-test","version":"1"}}}' \
    | head -5
# Expect: SSE response with "result":{"protocolVersion":...,"serverInfo":...}
# If you see "Unauthorized", REIN_HTTP_TOKEN was inherited despite step 3 —
# unset it in your current shell with `unset REIN_HTTP_TOKEN`, kill the
# rein process, and retry.
```

**Step 4 — Enable Tailscale Funnel.**

Modern Tailscale (>= 1.50) syntax — `--bg` runs cloudflared-style
background, `--https=443` is explicit:

```bash
tailscale funnel --bg --https=443 http://127.0.0.1:8680
# Expect:
#   Available on the internet:
#   https://<your-machine>.<your-tailnet>.ts.net/
#   |-- proxy http://127.0.0.1:8680
#   Funnel started and running in the background.

tailscale funnel status
# Expect: "Funnel on" with `/ proxy http://127.0.0.1:8680`
```

The first time you run Funnel under a given hostname, Tailscale's
edge provisions a Let's Encrypt cert. This typically takes 30 seconds
to 2 minutes; subsequent restarts reuse the cert.

**Step 5 — Verify externally (or skip this step).**

The local-network curl test below is **unreliable on many home/office
LANs** — corporate proxies, transparent CGNAT setups, and some
consumer routers intercept `*.ts.net` DNS resolution and remap it to
private space (e.g., `198.18.x.x`), so a curl from the same Mac
fails to reach the public Tailscale edge. Some networks do
this; the curl returned `[HTTP 000]` even though Funnel was healthy.

If you can curl from a phone on cellular data, that bypasses your
LAN's interception. Otherwise just skip to Step 6 — the actual proof
is whether Anthropic's cloud can reach the URL when you Add the
custom connector, which is what we're trying to verify anyway.

```bash
# May or may not work depending on your LAN.
curl -s -w "\n[HTTP %{http_code}]\n" --max-time 10 \
    https://<your-machine>.<your-tailnet>.ts.net/mcp \
    -X POST -H 'Accept: application/json, text/event-stream' \
    -H 'Content-Type: application/json' \
    -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"public-test","version":"1"}}}'
```

**Step 6 — Add the custom connector in Claude Desktop.**

Before clicking Add, open a second terminal:

```bash
tail -F /tmp/rein-sse.log
```

This shows incoming HTTP request lines in real time **only because
Step 3 started rein with `REIN_LOG=info`**, AND only for requests
that pass the pre-rmcp guards (Host allowlist + auth). The default
`REIN_LOG=warn` emits startup/shutdown but stays silent on successful
initialize / tool calls. **More importantly, requests rejected by
the Host guard (403) or auth guard (401) are also silent at
`REIN_LOG=info`** — they short-circuit before any per-request log
line is written. So a silent log can mean any of:

1. The handshake is **succeeding silently** at INFO (this is fine —
   check the Connectors list in Claude Desktop for the green
   "connected" dot).
2. Requests are **being rejected at the Host guard** (typo in
   `allowed_hosts`, missing FQDN entry).
3. Requests are **being rejected at the auth guard** (inherited
   `REIN_HTTP_TOKEN` putting rein into bearer-required mode despite
   the loopback opt-in).
4. Requests **never arrived at all** (tunnel down, DNS issue,
   Anthropic-side fault).

Don't infer from log silence alone. Use the Connectors list state in
Claude Desktop as the primary "did Anthropic reach me successfully"
signal. If it shows red / "failed to connect", check `allowed_hosts`
and the env-var inheritance trap before assuming the tunnel itself is
broken. For decisive proof you can also raise the verbosity to
`REIN_LOG=debug` which logs the guard rejections explicitly.

Now the UI:

1. Open **Claude Desktop**.
2. Click your avatar / username (top-left) → **"Customize"**, OR press
   `⌘,` to open Settings.
3. Top-level navigation: click **"Connectors"** (NOT the "Connectors"
   sub-tab nested under any plugin).
4. Click **"+"** → **"Add custom connector"**. The dialog title
   should say `Add custom connector  BETA`.
5. Fill in:
   - **Name:** `rein`
   - **Remote MCP server URL:** `https://<your-machine>.<your-tailnet>.ts.net/mcp`
   - **Advanced settings → OAuth Client ID / Secret:** **leave both blank**
6. Click **"Add"**.

Expected result: dialog closes, Connectors list shows `rein` with the
toggle on, and `tail -F /tmp/rein-sse.log` shows an `initialize`
request landing.

**Step 7 — Verify in Cowork.**

Switch to the **Cowork** tab in Claude Desktop, open a new
conversation, click the connectors menu in the chat, confirm `rein`
is listed and enabled. Try:

```text
Use rein to count my memories.
```

Expected: Claude calls `rein_stats` and returns the count from your
local `~/.rein/memories.db`.

### Tailscale Funnel — common pitfalls (from the v0.28.11 walkthrough)

- **`REIN_HTTP_TOKEN` env-var inheritance is the most common
  failure.** Loopback curl returns 401, you assume rein is broken,
  you spend 20 minutes restarting things — but your shell exports
  the token and rein just enabled bearer mode. Always start the
  Funnel-facing rein with `env -u REIN_HTTP_TOKEN nohup rein serve
  --sse …`. The Step 3 verification command above is the fast
  diagnostic.
- **The plugin marketplace `rein` plugin and the custom Cowork
  connector are two different things, and you may want both.** The
  `Personal plugins → Rein` entry (installed via the v0.28.9
  marketplace path) gives Claude Desktop's **Chat tab** and
  Claude Code stdio access; it does NOT make rein visible in
  Cowork. The custom connector you Add here gives Cowork /
  claude.ai / mobile access via the public Funnel URL. Both can be
  enabled simultaneously — Chat tab uses the faster stdio path
  automatically, Cowork uses the cloud-routed path.
- **LAN-side `*.ts.net` interception** — many home/office networks
  map `*.ts.net` resolution to private space, so local curl tests
  fail or time out. This does not mean Funnel is broken; the public
  edge still works, and Anthropic's cloud reaches it directly. Use
  `tail -F /tmp/rein-sse.log` as ground truth instead.
- **Cert provisioning delay** — first Funnel under a new hostname
  takes 30 seconds to 2 minutes for Let's Encrypt. If `Add custom
  connector` fails immediately, retry in 60 seconds. You can force
  cert provisioning with `tailscale cert <fqdn>` (writes
  `<fqdn>.crt` and `<fqdn>.key` into cwd — these are private keys;
  delete them when done, and the project's `.gitignore` already
  blocks `*.crt` / `*.key` since v0.28.11 as defense-in-depth).
- **Funnel hostname in `allowed_hosts`** — every additional client
  hostname you accept must be added; the default localhost-only
  allowlist is restrictive on purpose. Wildcards are not supported;
  list each FQDN explicitly.

## Recipe C: Self-hosted with Caddy + Let's Encrypt

For users with their own VPS and domain. Most flexible, most work.

### Caddyfile

The right Caddyfile shape depends on whether you have any inbound auth
gate at Caddy. **Bearer injection alone is not auth** — if Caddy
forwards every incoming request after stamping a bearer onto it, then
anyone who guesses the URL is authenticated to rein.

**Caddyfile A — Option 1a (loopback unauth, URL-only):**

```caddy
rein.<your-domain>.com {
    reverse_proxy 127.0.0.1:8680
}
```

Use only when you accept that anyone who learns the URL has rein
access. rein must be running with `auth = "public"` (Option 1a) and
an `allowed_hosts = ["rein.<your-domain>.com"]` entry in
`~/.rein/config.toml`.

**Caddyfile B — Option 1b (real inbound auth + upstream bearer
injection):**

```caddy
rein.<your-domain>.com {
    # Inbound auth — pick ONE of:
    #   basicauth { user $hashed_password }
    #   forward_auth http://your-oidc-proxy:8443 { ... }   # OIDC plugin
    # Without an inbound gate, this Caddyfile is no safer than A.
    basic_auth {                      # `basicauth` on Caddy < v2.8.0
        # Generate the hash with: caddy hash-password
        # Then store the literal hash here, not the cleartext password.
        you JDJhJDE0JC4uLg
    }

    # Once the caller is authenticated at Caddy, inject the bearer rein
    # expects so rein-side auth is also satisfied.
    reverse_proxy 127.0.0.1:8680 {
        header_up Authorization "Bearer {$REIN_HTTP_TOKEN}"
    }
}
```

Run Caddy with `REIN_HTTP_TOKEN` exported to the same value rein sees.
The defense-in-depth posture: edge-side `basicauth` gates inbound
callers, upstream `header_up` keeps rein-side bearer enforced. Failure
of either layer alone still leaves rein protected.

Anthropic's connector cannot supply HTTP Basic credentials directly —
this pattern only works if you front the connector with a more capable
gateway (Posture 3 in "Authentication"). Most users wanting Cowork +
real auth should reach for **Recipe A + Cloudflare Access OIDC**
instead, which Anthropic's connector OAuth fields can drive.

Caddy auto-provisions a Let's Encrypt cert on first request. Run as a
service:

```bash
sudo systemctl enable --now caddy
```

Make sure DNS A/AAAA for `rein.<your-domain>.com` points at the VPS,
and `rein serve --sse` is running on the same VPS bound to localhost.

### nginx alternative

The nginx equivalents follow the same posture split as the Caddy
examples above.

**nginx A — Option 1a (loopback unauth, URL-only):**

```nginx
server {
    listen 443 ssl http2;
    server_name rein.your-domain.com;

    ssl_certificate     /etc/letsencrypt/live/rein.your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/rein.your-domain.com/privkey.pem;

    # Streamable HTTP needs both POST (client→server) and GET (SSE).
    location /mcp {
        proxy_pass http://127.0.0.1:8680/mcp;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Connection "";
        # SSE buffering off so the stream is delivered promptly.
        proxy_buffering off;
        proxy_cache off;
        # Long timeout for SSE streams.
        proxy_read_timeout 1d;
    }
}
```

Same caveats as Caddyfile A: rein needs
`auth = "public"` plus
`allowed_hosts = ["rein.your-domain.com"]` in `~/.rein/config.toml`,
and you accept that anyone who knows the URL is authenticated.

**nginx B — Option 1b (basic auth + upstream bearer injection):**

```nginx
server {
    listen 443 ssl http2;
    server_name rein.your-domain.com;

    ssl_certificate     /etc/letsencrypt/live/rein.your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/rein.your-domain.com/privkey.pem;

    # Inbound auth — generate /etc/nginx/htpasswd with `htpasswd -c`.
    auth_basic           "rein";
    auth_basic_user_file /etc/nginx/htpasswd;

    location /mcp {
        # Replace inbound Authorization (basic) with rein's bearer.
        # nginx does not expand $REIN_HTTP_TOKEN from the environment in
        # `proxy_set_header` — render the literal value into the config
        # at deploy time (templating tool, sed, Ansible, etc.) and store
        # the rendered nginx config with appropriate file permissions.
        # The placeholder below MUST be replaced with the actual secret.
        proxy_set_header Authorization "Bearer REPLACE_WITH_REIN_HTTP_TOKEN";
        proxy_pass http://127.0.0.1:8680/mcp;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Connection "";
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 1d;
    }
}
```

> **nginx env-var caveat:** Stock nginx does not interpolate process
> environment variables into directive arguments. The Caddy example
> above can use `{$REIN_HTTP_TOKEN}` because Caddy expands env vars
> natively; nginx cannot. If you need the secret to live outside the
> on-disk config, options are: (a) the third-party `ngx_http_perl_module`
> + `perl_set` to read env, (b) `lua_nginx_module` (OpenResty) +
> `os.getenv`, or (c) deploy-time templating (sed / Ansible / Jinja).
> The recipe above uses (c) — easiest and avoids module dependencies.

Same Anthropic-connector compatibility note as Caddyfile B: the
connector won't supply HTTP Basic credentials directly. Use this
pattern only behind an OAuth/OIDC gateway in front, OR for callers
other than Cowork (e.g., your own scripts that can pass basic auth).

For nginx, preserving `Host` is not optional. If nginx rewrites the
upstream Host to `127.0.0.1`, rein cannot identify the public hostname
for `allowed_hosts`, OAuth metadata, or loopback-only auth decisions.

Get the cert with `certbot --nginx -d rein.your-domain.com`. After
rendering the bearer into the nginx config, restrict file permissions
(`chmod 600 /etc/nginx/conf.d/rein.conf`) since the secret is now
inline.

## Recipe D: ngrok

ngrok is the closest alternative to Recipe A.0 (Cloudflare Quick
Tunnel) for users whose network blocks Cloudflare's QUIC/HTTP/2 paths
or intercepts `*.trycloudflare.com` resolution. It needs a free
account but no domain. ngrok uses standard HTTPS over TCP/443 so it
tends to work where Cloudflare Quick Tunnel doesn't.

```bash
brew install ngrok
ngrok config add-authtoken <your-token>     # one-time, get from ngrok.com
ngrok http 8680
```

ngrok prints a temporary URL like `https://abcd-1234.ngrok-free.app`.
On the **free tier**, this URL changes every time `ngrok` restarts;
on **paid tiers** you can reserve a domain and the URL stays
permanent.

### When to choose ngrok

- You don't have a Cloudflare account or domain.
- Cloudflare Quick Tunnel (Recipe A.0) failed because your network
  blocks QUIC and HTTP/2 fallback also didn't help.
- You don't want to install Tailscale.
- You're OK signing up for one more SaaS account.

> **Host header caveat:** rein's `[server].allowed_hosts` must include
> the ngrok hostname (`abcd-1234.ngrok-free.app`) for each new tunnel,
> or every request returns `403 Host header is not allowed`. Since
> the ngrok URL changes per invocation on the free tier, you'll be
> editing `~/.rein/config.toml` and restarting `rein serve --sse`
> each session. ngrok paid tiers offer reserved domains that make
> this less painful and are worth the upgrade if you're using ngrok
> as your daily Cowork-rein tunnel.

## Recipe E: Built-in OAuth provider (recommended for v0.30+)

Use this when rein is reachable through any public HTTPS tunnel and you want
Claude's custom connector to complete OAuth itself.

**Prerequisite: use a GUI-enabled rein binary.** The standard
`cargo install --git https://github.com/lyr1cs/rein --tag v0.37.0 --locked rein`
path builds the default non-GUI binary (`default = []`), so `rein serve --gui`
cannot serve the Settings page needed to create the owner approval session.
Use one of these instead:

```bash
# Clone build with embedded GUI assets.
git clone --branch v0.37.0 https://github.com/lyr1cs/rein
cd rein
./scripts/install.sh

# Or, from an existing checkout after building the GUI assets:
cargo build -p rein --release --features gui
install -m 0755 target/release/rein ~/.cargo/bin/rein
```

The Claude Desktop `.mcpb` release artifact also contains the GUI-enabled
server binary.

`~/.rein/config.toml`:

```toml
[server]
sse_enabled = true
gui_enabled = true
sse_bind = "127.0.0.1"
sse_port = 8680
auth = "oauth"
public_url = "https://rein.example.com"
allowed_hosts = ["rein.example.com"]
```

Start rein:

```bash
export REIN_HTTP_TOKEN="$(openssl rand -hex 32)"
rein serve --gui
```

In OAuth mode this token is **not** required from Claude and does not
turn `/mcp` back into static-bearer auth. It is the owner approval
secret: sign into the rein GUI through the same public HTTPS hostname
used by the connector, not through `http://127.0.0.1:8680`, before
approving `/oauth/authorize`. The session cookie is host-only, so local
login does not authenticate the public approval page.

Expose `127.0.0.1:8680` with one of the HTTPS recipes above, then configure
Claude's custom connector with:

- MCP URL: `https://rein.example.com/mcp`
- OAuth fields: leave blank if the client supports Dynamic Client Registration

Expected connector flow:

1. Claude reads `/.well-known/oauth-authorization-server`.
2. Claude registers through `/oauth/register`.
3. Browser opens `/oauth/authorize`.
4. Owner signs into the rein GUI with `REIN_HTTP_TOKEN` if not already
   signed in, then approves access.
5. Claude exchanges the code at `/oauth/token`.
6. Claude calls `/mcp` with `Authorization: Bearer <access_token>`.

Verify locally before trying Claude:

```bash
./scripts/oauth-e2e-test.sh
```

## Authentication

This section consolidates the auth tradeoffs scattered across the
recipes above. Read it before deciding which posture to deploy.

Anthropic's custom-connector UI accepts:

1. **A URL only** — connector hits the URL with no auth and expects the
   server (or whatever sits in front of it) to either be open or fail
   the request.
2. **A URL + OAuth Client ID + Secret** in "Advanced settings" — the
   connector runs a standard OAuth 2.0 flow against the URL, expecting
   it to behave like an OAuth-protected resource server.

There is **no field for an arbitrary bearer token, API key header, or
custom auth scheme** in the connector UI. So if you want any auth in
front of the tunnel, it has to be either:

- A real OAuth/OIDC provider that the URL implements or proxies to, OR
- The proxy itself adding/checking auth (e.g., a custom-header check at
  Caddy, transparently adding bearer to upstream requests)

rein's own bearer auth (`REIN_HTTP_TOKEN`) is enforced **at the rein
HTTP listener**. For the remote-MCP path it works only if you can also
get the bearer header onto the request — which means the proxy must
add it (the connector won't), or you must use Option 1a above
(`auth = "public"`) and rely on edge-level auth
or URL obscurity instead.

Quick reference of the practical combinations:

| Posture | Edge auth | rein auth | Anthropic connector UI |
| --- | --- | --- | --- |
| Loopback-only (URL obscurity) | none | `auth = "public"` | URL only |
| Cloudflare Access OIDC | OIDC IdP | `auth = "public"` | URL + OAuth Client ID/Secret |
| Caddy bearer injection | Caddy header check | `REIN_HTTP_TOKEN` enforced | URL only (Caddy still gates) |
| OAuth gateway in front | gateway OAuth | either | URL + OAuth Client ID/Secret |
| Built-in OAuth | rein OAuth | `auth = "oauth"` + owner `REIN_HTTP_TOKEN` | URL only / DCR |

The known *broken* combination — Cloudflare Access **service tokens**
+ Anthropic OAuth fields — is documented in Recipe A Step 5.

## Configure the connector in Claude

### Pro / Max plan

1. Open Claude Desktop or claude.ai.
2. Click your avatar → **"Customize"** → **"Connectors"**.
3. Click **"+"** then **"Add custom connector"**.
4. Enter the **remote MCP server URL** — for the recipes above:
   - Cloudflare Tunnel: `https://rein.<your-domain>.com/mcp`
   - Tailscale Funnel: `https://<machine>.<tailnet>.ts.net/mcp`
   - Self-hosted: `https://rein.<your-domain>.com/mcp`
   - ngrok: `https://<random>.ngrok-free.app/mcp`
5. (If you've configured an OAuth/OIDC layer in front of the tunnel —
   see "Posture 2" in Step 5 of Recipe A) click **"Advanced settings"**
   → paste the OAuth Client ID and Secret your IdP issued. Do **not**
   paste Cloudflare Access service-token credentials here — they are
   not OAuth and the connector handshake will fail (see the boxed
   warning in Recipe A Step 5).
6. (If using URL-obscurity / Posture 1) leave the OAuth fields empty.
7. Click **"Add"**.

### Team / Enterprise plan (Owners only)

1. **"Organization settings"** → **"Connectors"** → **"Add"**.
2. Hover **"Custom"** → select **"Web"**.
3. Same URL + OAuth fields as above.
4. **"Add"**.
5. Tell members to **"Customize"** → **"Connectors"** → click
   **"Connect"** to authenticate.

### Verifying the connector

After "Add" succeeds, rein appears in the connectors list with a
toggle. Open a new conversation in **Cowork** or **claude.ai**, click
the connectors button, and confirm rein's tools (40 of them, prefixed
`rein_`) appear.

In your first conversation:

- "Recall what I know about <topic>" should call `rein_recall` and
  return matches.
- "Remember that <fact>" should call `rein_store`.

If the connector shows "failed to connect", check:

- The URL responds to `curl -fsS <URL>` from outside your network.
- `~/Library/Logs/Claude/mcp-*.log` for client-side errors.
- For Cloudflare: `cloudflared tunnel info rein` shows tunnel up.
- For Caddy: `journalctl -u caddy -f` for proxy errors.
- For self-host: confirm `127.0.0.1:8680/mcp` works locally first
  before adding the proxy layer.

### `unknown error` in claude.ai connector dialog after rein upgrade (OAuth posture only)

Symptom: after upgrading rein on the server, the existing claude.ai
Custom Connector starts surfacing `An unknown error occurred connecting
to the MCP server` (`ofid_*`). Calls were working before the upgrade.

Cause: the v0.30.0 `refresh_token_fingerprint` schema migration backfills
the new column on every grant created before v0.30.0 and revokes them at
the same time. Anthropic's connector broker does not always fall back from
a revoked refresh-token to a fresh authorization-code flow, so the
connector stays broken until the operator manually re-authorizes.

`rein doctor` surfaces this explicitly when `[server].auth = "oauth"`:

```
WARN  oauth_provider  auth_policy=oauth; oauth_clients=N; active_grants=0;
                       expired_oauth_records=M — DCR clients exist but every
                       grant is revoked
      → in claude.ai → Settings → Connectors, remove the rein connector
        and re-add it (same URL); this re-runs DCR + authorization-code
        and issues a fresh grant
```

Recovery: follow the repair hint. The Custom Connector configuration
(URL, OAuth Client ID/Secret) does **not** need to change — claude.ai
re-runs Dynamic Client Registration when you re-add the connector with
the same URL, and the broker walks the full authorize → token flow on
the next reconnect. This typically resolves within 1–2 minutes.

If `oauth_clients=0` is also reported, your install never completed an
authorize flow against this rein database (fresh install, new database,
or a `~/.rein/memories.db` swap). In that case the recovery is the
same — add the connector for the first time.

### `An unknown error occurred` mid-flow with `auth = "oauth"` (cookie expired)

Symptom: an existing connector that's been working stops working a few
minutes into a session; refreshing the connector dialog or restarting
the conversation makes it work again briefly.

Cause: the owner-approval cookie that lets the operator's browser
approve claude.ai's authorization request has a 10-minute `Max-Age` and
was originally only set by `POST /api/session`. If the browser tab
holding the cookie was idle for more than 10 minutes, the next
`/oauth/authorize` redirect from claude.ai found no valid cookie and
bounced back as `access_denied`, which the broker surfaced as a
generic `ofid_*` error.

Recovery (rein v0.35+): each successful `GET /oauth/authorize` now
re-emits the cookie with a fresh 600-second `Max-Age`, so the window
slides as long as the operator visits the authorize URL at least once
every 10 minutes. **Pre-v0.35**: `POST /api/session` again with the
bearer token to re-stamp the cookie, then retry the connector.

`rein --version` to verify which behavior applies.

### `403 Forbidden — Host header is not allowed`

The most common failure mode after a fresh deployment: rein returns
403 for requests with `Host: rein.your-domain.com` because the
hostname is not in `[server].allowed_hosts`. Fix by editing
`~/.rein/config.toml`:

```toml
[server]
allowed_hosts = ["rein.your-domain.com"]
```

then restart `rein serve --sse`. List **every** hostname the tunnel
might present — for Tailscale Funnel it's the `*.ts.net` URL, for
ngrok it's the ephemeral `*.ngrok-free.app` URL (which changes per
run, so for ongoing ngrok use you'd need to re-edit the config each
time — another reason ngrok is dev-only).

Verify with curl:

```bash
curl -fsS https://rein.your-domain.com/mcp \
    -H 'Accept: application/json, text/event-stream' \
    -H 'Content-Type: application/json' \
    -H 'Host: rein.your-domain.com' \
    -d '{"jsonrpc":"2.0","id":1,"method":"initialize",
         "params":{"protocolVersion":"2025-06-18","capabilities":{},
                   "clientInfo":{"name":"curl","version":"1.0"}}}'
```

If the curl returns a JSON / SSE response (not 403), the connector
will too.

## Operating considerations

### Always-on requirements

A connector pointing at a tunnel that goes down (laptop asleep,
network hiccup, tunnel process crashed) shows as "failed to connect"
and Claude's `rein_*` tools become unusable. For reliability:

- Run `cloudflared` / `tailscale` / `caddy` as a system service
  (launchd / systemd).
- Run `rein` as a service: see
  [02-installation.md § HTTP, SSE, And GUI](02-installation.md#http-sse-and-gui)
  for `rein serve --sse` background management. Or wrap with launchd
  manually.
- Consider deploying rein on a VPS so it's not tied to your laptop's
  uptime.

### Memory database location on a remote host

If rein runs on a different machine from where you originally seeded
its memories (e.g., laptop has the local database, VPS hosts the
remote rein), you have two options:

1. **Sync the SQLite database** — `rsync ~/.rein/ user@vps:~/.rein/`
   one-time, then accept that the two diverge.
2. **Run rein only on the VPS** — local Claude Desktop / Claude Code
   point to the same remote endpoint via custom connector (or stdio
   that calls `rein` over SSH, more complex). Single source of truth.

There is no built-in distributed mode. The architectural answer for
multi-machine memory sharing is "one rein instance accessed remotely
by all clients", not "multiple rein instances syncing".

### Observability

`rein doctor` from the same machine `rein serve --sse` runs on shows
all health metrics. Logs go to stderr (captured by launchd /
systemd / `cloudflared` parent process). Set `REIN_LOG=info` or
`REIN_LOG=debug` for more verbosity.

## Limitations

The current rein remote-MCP path has these known limitations:

- **Built-in OAuth is single-user** — v0.30's provider is designed for
  one owner approving clients against one private `~/.rein/` database.
  It is not a multi-tenant identity provider and does not implement
  JWKS, token introspection, or federated OIDC login. If you need
  organization-wide SSO policy, put Cloudflare Access / another OIDC
  gateway in front of rein.
- **No mobile-specific testing** — Anthropic says custom connectors
  work on mobile, but the rein team has not yet validated the
  end-to-end Cowork-mobile path. Open an issue if you find issues.
- **Single-tenant by design** — rein is meant to be private to one
  user (or organization). A public connector exposing `~/.rein/` is
  the same thing as exposing your personal memory database, so always
  pair with auth at some layer.

## Other platforms

The Streamable HTTP transport rein serves is platform-independent —
the rein binary runs on macOS Apple Silicon (currently the only
released platform), but the proxy / tunnel / Caddy layer can run on
any OS that can reach rein over loopback. Linux x64 and Windows x64
binaries are not currently produced; see
[docs/decisions/distribution-channels.md](../decisions/distribution-channels.md)
for the rationale and re-evaluation triggers.

## Related

- `02-installation.md` — local-stdio install paths (Claude Desktop
  Chat tab, Claude Code CLI)
- `docs/decisions/distribution-channels.md` — ADR on why DXT + plugin
  marketplace + remote MCP coexist as separate paths
- `docs/guides/dxt-build.md` — maintainer build guide for the local
  Claude Desktop DXT artifact
