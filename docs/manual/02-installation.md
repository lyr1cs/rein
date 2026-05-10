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

## Remote Install (Pinned Tag)

When you want to install a specific tagged release directly from GitHub
without cloning, use `cargo install --git` with **both** `--tag` **and**
`--locked`:

```bash
# Install a specific release (binary only, no GUI)
cargo install --git https://github.com/lyr1cs/rein --tag v0.30.0 --locked rein
```

To upgrade an existing install in place, add `--force`:

```bash
cargo install --git https://github.com/lyr1cs/rein --tag v0.30.0 --locked rein --force
```

> ⚠️ **`--locked` is mandatory when using `--tag`.** Without `--locked`,
> `cargo install --git` ignores the `Cargo.lock` committed at that tag and
> re-resolves every transitive dependency to the latest semver-compatible
> version on crates.io. A newer dependency may pull in C/SIMD code that
> requires a newer host toolchain than the one on the target machine
> (e.g. `usearch 2.25` introduced `numkong 7.6.0`, whose SIMD probes need
> GCC 13+ or clang 17+ on aarch64 Linux). With `--locked`, you build the
> exact dependency graph that was tested at release time.

If you need the embedded GUI when installing remotely, you must clone the
source instead — `--git` alone does not run `npm` for the GUI bundle:

```bash
git clone --branch v0.30.0 https://github.com/lyr1cs/rein
cd rein
./scripts/install.sh        # builds GUI then runs cargo install --locked --features gui
```

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

Plain stdio MCP startup does not run background warmup by default. For Codex
agent-team workflows this avoids one side-index rebuild per subagent; run
`rein warmup` explicitly after bulk memory changes or keep the HTTP/GUI service
enabled for service-level startup warmup.

See the OpenAI Codex MCP documentation entry in
`docs/reference/bibliography.md` for the upstream config format reference.

## Claude Desktop (One-click via DXT)

Rein ships as a Claude Desktop Extension (DXT, file extension `.mcpb`) for
macOS Apple Silicon. This path requires no Rust toolchain and no manual
`mcpServers` editing.

### What gets installed

A `.mcpb` is a zip archive containing:

- `manifest.json` — declares the binary, env vars, and the user-config
  fields Claude Desktop should prompt for at install time.
- `server/rein-darwin-arm64` — precompiled rein binary (~16 MB with GUI).

Claude Desktop unpacks the archive into its extension directory, prompts
the user for the fields declared in `user_config`, injects them as env
vars, and spawns `rein serve` over stdio.

### Quick install (5 steps)

1. **Download** `rein-v<version>.mcpb` from
   [GitHub Releases](https://github.com/lyr1cs/rein/releases/latest).
2. **Clear macOS quarantine** (one-time — the build is unsigned):

   ```bash
   xattr -d com.apple.quarantine ~/Downloads/rein-v*.mcpb
   ```

   Without this, double-click triggers Gatekeeper's "cannot be opened
   because the developer cannot be verified" dialog. Alternative: open
   `Settings → Privacy & Security` after the rejection and click `Open
   Anyway`.

3. **Double-click** the `.mcpb`. Claude Desktop displays an install dialog.
4. **Fill in user config**:

   | Field                  | Required | Notes |
   | ---------------------- | :------: | ----- |
   | `Gemini API Key`       | yes      | Stored encrypted by Claude Desktop. Injected as `GEMINI_API_KEY`. Get a key at <https://aistudio.google.com/apikey> (1500 req/day free tier). |
   | `Memory database path` | no       | Defaults to `~/.rein/memories.db`. Override only if you maintain multiple memory stores. |
   | `Supermemory API Key`  | no       | Optional cross-validation source. Leave blank to skip Supermemory hybrid search. |

5. **Click Install.** Claude Desktop will spawn `rein serve` on the next
   conversation. ~40 `rein_*` tools appear in the tool list.

### Verifying the install

- **Tool count**: in a new chat, ask Claude to list rein tools. Expect
  `rein_recall`, `rein_store`, `rein_stats`, plus 37 others.
- **Process check**: `ps aux | grep "rein serve"` should show one process.
- **Logs**: `~/Library/Logs/Claude/mcp-server-rein.log`. Tail it with
  `tail -F` while exercising the extension.

### Upgrading

Download the new `.mcpb`, clear quarantine, double-click. Claude Desktop
replaces the existing extension in place; your `user_config` values are
preserved unless you reset them in `Settings → Extensions → rein`.

### Uninstalling

`Settings → Extensions → rein → Disable` (keeps config) or `Remove`
(deletes config). The SQLite database at `~/.rein/memories.db` is **not**
deleted — remove it manually if you want a clean slate:

```bash
rm -rf ~/.rein/
```

### Troubleshooting

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| "rein cannot be opened because the developer cannot be verified" | Build is unsigned | `xattr -d com.apple.quarantine` (step 2) or `Settings → Privacy & Security → Open Anyway` |
| Install dialog never appears after double-click | Claude Desktop < 1.0 | Update Claude Desktop. `Help → About Claude` shows the version. |
| Tools not appearing in chat | `GEMINI_API_KEY` unset or wrong | Check `mcp-server-rein.log` for `gemini api key not set` or HTTP 401. |
| `unable to open database` on first run | Sandbox can't write `~/.rein/` | Set `Memory database path` explicitly in `user_config` to a path inside Claude Desktop's data directory. |
| High memory or CPU after months of use | Memories DB grew | Run `rein gc` from a terminal install (or via the MCP `rein_gc` tool). |
| `cargo install` fails with `inlining failed in call to 'always_inline' 'vdotq_s32'` (or similar SIMD intrinsic name) on aarch64 Linux | `cargo install --git --tag` re-resolved dependencies and pulled a newer `usearch`/`numkong` requiring GCC 13+ / clang 17+ | Add `--locked` to the install command (see [Remote Install (Pinned Tag)](#remote-install-pinned-tag)). |

### Cowork, claude.ai, and mobile (remote MCP)

This DXT path enables rein in Claude Desktop's **Chat** tab only. It does
**not** make rein available in **Cowork** (the agentic-work tab in Claude
Desktop), **claude.ai** web, or the **Claude mobile** apps. Those clients
route MCP traffic through Anthropic's cloud rather than local stdio, so
they need a public HTTPS endpoint they can reach. See
[02b-remote-mcp-deployment.md](02b-remote-mcp-deployment.md) for the full
remote-MCP deployment guide (Cloudflare Tunnel / Tailscale Funnel / Caddy +
Let's Encrypt / ngrok recipes plus the Claude UI configuration flow).

### Other platforms

The current DXT ships only macOS Apple Silicon binaries. Intel Mac, Linux,
and Windows users have two alternatives:

1. **Claude Code plugin marketplace** — see the project README's
   `Install via Claude Code plugin marketplace` section. Requires
   `cargo install --git https://github.com/lyr1cs/rein --tag v0.30.0 --locked rein` separately.
2. **Manual MCP entry** — edit
   `~/Library/Application Support/Claude/claude_desktop_config.json`:

   ```json
   {
     "mcpServers": {
       "rein": {
         "command": "/Users/<you>/.cargo/bin/rein",
         "args": ["serve"],
         "env": { "GEMINI_API_KEY": "..." }
       }
     }
   }
   ```

   Use the **absolute path** because Claude Desktop does not inherit shell
   `PATH`.

For maintainers building the `.mcpb` artifact, see
[docs/guides/dxt-build.md](../guides/dxt-build.md).

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
