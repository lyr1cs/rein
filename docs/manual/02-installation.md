# Installation

This chapter covers installing Rein from source, running it in Docker, enabling
the GUI, configuring MCP clients, and validating a new deployment. Commands are
intended to run from the repository root.

## Prerequisites

- Rust and Cargo for building the `rein` binary.
- Node.js and npm when building the embedded GUI.
- Optional: `GEMINI_API_KEY` for Google embeddings, extraction, query expansion,
  synthesis, and other LLM-backed features.
- Optional: `SUPERMEMORY_CC_API_KEY` when using the Supermemory integration.
- Optional: Docker and Docker Compose for container deployment.

Rein is local-first. By default it stores data in a local SQLite database; set
`REIN_DB` when you want an explicit database path.

## Source Install

Install the CLI and stdio MCP server:

```bash
cargo install --path crates/rein --locked
```

Build and embed the GUI assets before installing a GUI-enabled binary:

```bash
cd crates/rein/gui
npm ci
npm run build
cd ../../..
cargo install --path crates/rein --locked --features gui
```

The `--features gui` build embeds the Vite output into the Rust binary. Without
that feature, the CLI and MCP server still work, but the GUI service has no
embedded assets to serve.

## Install Script

The install script checks for Cargo, builds GUI assets when npm is available,
and installs the binary with locked Cargo dependencies:

```bash
./scripts/install.sh
```

For a CLI-only install through the script:

```bash
REIN_INSTALL_GUI=0 ./scripts/install.sh
```

When `REIN_INSTALL_GUI=1` or unset, npm controls whether the script attempts the
GUI build. If npm is missing, the script falls back to a CLI-only install.

## Docker

Build the image:

```bash
docker build -t rein .
```

Run the container with an HTTP bearer token:

```bash
export REIN_HTTP_TOKEN="change-this-token"
docker run --rm -p 8680:8680 \
  -e REIN_HTTP_TOKEN \
  -e GEMINI_API_KEY \
  -v rein-data:/data \
  rein
```

The image sets `REIN_DB=/data/memories.db`, listens on `0.0.0.0:8680`, and
refuses to start unless `REIN_HTTP_TOKEN` is set.

## Docker Compose

The Compose file builds the local image, publishes port `8680`, stores data in
the `rein-data` volume, and requires `REIN_HTTP_TOKEN`:

```bash
export REIN_HTTP_TOKEN="change-this-token"
export GEMINI_API_KEY="optional-gemini-key"
docker compose up -d
```

Check the service:

```bash
docker compose logs -f rein
```

## MCP Init

Run a dry run first to see which known clients are present:

```bash
rein init --dry-run
```

Then write client config entries:

```bash
rein init
```

Rein scans existing config files for Claude Code, Claude Desktop, Cursor,
Windsurf, VS Code, Gemini, Codex, and OpenCode. JSON clients receive an
`mcpServers.rein` entry that runs `rein serve`. Codex uses TOML; the official
Codex MCP server table shape is:

```toml
[mcp_servers.rein]
command = "rein"
args = ["serve"]
```

See the OpenAI Codex MCP documentation entry in
`docs/reference/bibliography.md` for the upstream config format reference.

## HTTP, SSE, And GUI

Start stdio MCP for local MCP clients:

```bash
rein serve
```

Start HTTP/SSE MCP:

```bash
export REIN_HTTP_TOKEN="change-this-token"
rein serve --sse
```

Start the GUI in the foreground:

```bash
export REIN_HTTP_TOKEN="change-this-token"
rein serve --gui
```

Manage the GUI as a background service:

```bash
export REIN_HTTP_TOKEN="change-this-token"
rein gui on
rein dashboard
rein gui off
```

HTTP/SSE and GUI API routes require `REIN_HTTP_TOKEN` unless you explicitly set
`[server].allow_unauthenticated_loopback = true` on a loopback bind. Wildcard
binds also require either a bearer token or explicit `allowed_hosts`.

## Proxy

The proxy transparently forwards LLM traffic and records eligible responses; it
does not inject memories into requests.

Start it in the foreground:

```bash
export REIN_PROXY_TOKEN="change-this-token"
rein serve --proxy
```

Or manage it as a background service:

```bash
export REIN_PROXY_TOKEN="change-this-token"
rein proxy on
rein dashboard
rein proxy off
```

Proxy clients should send `x-rein-token` from `REIN_PROXY_TOKEN`. Proxy startup
also accepts `REIN_HTTP_TOKEN` as a fallback, but setting a distinct proxy token
keeps HTTP and proxy credentials easier to rotate independently.

`rein init --proxy` can generate shell helpers and Codex provider overrides for
common proxy flows. Generated Codex proxy examples use `env_http_headers` so the
token is read from `REIN_PROXY_TOKEN` at invocation time.

## Environment Variables

| Variable | Purpose |
| --- | --- |
| `REIN_DB` | Override the SQLite database path. |
| `GEMINI_API_KEY` | Google embedding and LLM API key. |
| `SUPERMEMORY_CC_API_KEY` | Supermemory API key. |
| `REIN_HTTP_TOKEN` | Bearer token for HTTP/SSE, GUI APIs, and Docker. |
| `REIN_SSE_BIND` | Override HTTP/SSE bind address. |
| `REIN_SSE_PORT` | Override HTTP/SSE port. |
| `REIN_PROXY_TOKEN` | Proxy bearer token for `x-rein-token`. |
| `REIN_PROXY_BIND` | Override proxy bind address. |
| `REIN_PROXY_PORT` | Override proxy port. |
| `REIN_ASYNC_MEMORY_PROVIDER` | Override async memory worker provider. |
| `REIN_INSTALL_GUI` | Set to `0` to skip GUI build in `scripts/install.sh`. |
| `REIN_LOG` | Set Rust tracing level, such as `info` or `debug`. |

## Validation Smoke Tests

Validate client auto-configuration without writing:

```bash
rein init --dry-run
```

Run diagnostics:

```bash
rein doctor
```

Check services and store metrics:

```bash
rein dashboard
```

Store and recall a memory:

```bash
rein store --topic install-smoke --content "Rein install smoke test" --importance low --keywords smoke,install
rein recall "install smoke" --topic install-smoke --limit 5
```

For HTTP/SSE or GUI deployments, include the bearer token in API clients:

```bash
curl -H "Authorization: Bearer $REIN_HTTP_TOKEN" http://127.0.0.1:8680/api/version
```

For proxy deployments, smoke the metrics endpoint after startup:

```bash
curl -H "x-rein-token: $REIN_PROXY_TOKEN" http://127.0.0.1:8690/rein/metrics
```

## Auth Notes

Stdio MCP does not use HTTP bearer auth because it communicates over the local
process transport. HTTP/SSE, GUI API routes, Docker, and proxy deployments are
token-protected by default. Unauthenticated loopback is an explicit local-only
configuration choice, not the default.
