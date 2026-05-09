# Building the rein Claude Desktop Extension

This guide covers how the `.mcpb` artifact attached to GitHub Releases is
produced, and how to add new platforms or bump versions. End-user install
instructions live in
[docs/manual/02-installation.md](../manual/02-installation.md#claude-desktop-one-click-via-dxt) —
do not duplicate them here.

## Overview

A Claude Desktop Extension (DXT) is a zip archive (`.mcpb`) containing a
`manifest.json` and one precompiled binary per supported platform. Claude
Desktop reads the manifest, prompts the user for `user_config` fields, injects
them as env vars, and spawns the binary over stdio.

The current rein DXT targets **macOS Apple Silicon only**. The manifest, the
`scripts/build-dxt.sh` packaging script, and the GitHub Release flow all
assume a single-platform artifact. See [Adding new platforms](#adding-new-platforms)
at the end of this guide for what changes if/when more platforms are added.

## Prerequisites

- Rust toolchain (the version `crates/rein/Cargo.toml` requires)
- `npm` + `node` — the build script runs `npm ci && npm run build` in
  `crates/rein/gui/` before the cargo build, because `gui/dist/` is
  `.gitignore`d and rust-embed needs it to compile `--features gui`
- `zip` (preinstalled on macOS)
- `jq` (`brew install jq`)
- Apple Silicon Mac — the build script does not cross-compile
- Repo cloned at the workspace root

## Repository layout

```
source/rein/
├── dxt/
│   ├── manifest.json   # source of truth for the DXT manifest
│   └── README.md       # short summary; full guide is this file
├── scripts/
│   └── build-dxt.sh    # single command to produce target/rein-v<version>.mcpb
└── target/             # cargo output; the .mcpb lands at target/rein-v<version>.mcpb
```

The `dxt/` directory is **versioned** in git — the manifest is part of the
repo. The output `.mcpb` is **not** versioned (it is built per-release and
uploaded to the GitHub Release page).

## Build script

`scripts/build-dxt.sh` runs end-to-end:

1. Validates that `cargo`, `npm`, `jq`, and `zip` are on `PATH`. Bails early
   if any is missing.
2. Reads the version from `crates/rein/Cargo.toml` (single source of truth).
3. Runs `npm ci && npm run build` in `crates/rein/gui/` to produce
   `gui/dist/`. This step is required because the rust-embed derive in
   `crates/rein/src/mcp/rest.rs` reads `gui/dist/` at compile time when
   `--features gui` is on, and `gui/dist/` is `.gitignore`d so a clean
   checkout has nothing to embed.
4. Runs `cargo build -p rein --release --locked --features gui`.
5. Stages a temporary directory:
   - copies `dxt/manifest.json` (and `dxt/icon.png` if present)
   - copies `target/release/rein` to `server/rein-darwin-arm64`
   - patches the manifest's `version` field with `jq` so manifest and
     `Cargo.toml` never drift
6. Runs `jq -e .` on the staged manifest to validate JSON.
7. Zips the staging directory with `zip -r -X -D` (`-X` strips macOS
   extended attributes; `-D` omits directory entries that can confuse the
   MCPB unpacker).
8. Outputs `target/rein-v<version>.mcpb`.

Run:

```bash
chmod +x scripts/build-dxt.sh
./scripts/build-dxt.sh
```

Expected output: `target/rein-v<version>.mcpb`, ~10-16 MB.

## Manifest field reference

`dxt/manifest.json` follows the
[DXT spec v0.3](https://github.com/anthropics/dxt/blob/main/MANIFEST.md).
Fields rein uses:

| Field                                | Purpose                                | Notes |
| ------------------------------------ | -------------------------------------- | ----- |
| `manifest_version`                   | DXT spec version                       | Pin to `"0.3"` until we explicitly migrate. |
| `name` / `display_name`              | Identifiers                            | `name` is machine-readable, `display_name` shows in Claude Desktop UI. |
| `version`                            | Semver                                 | Auto-patched by `build-dxt.sh` from `Cargo.toml`. **Do not hand-edit.** |
| `description` / `long_description`   | Shown in install dialog                | `long_description` supports markdown. |
| `author` / `homepage` / `repository` / `license` / `keywords` | Metadata | Standard. |
| `icon` (optional)                    | Path to PNG inside archive             | 256×256 recommended. Currently absent — add when an official rein logo exists. |
| `server.type`                        | Always `"binary"` for rein             | rein has no Node/Python wrapper. |
| `server.entry_point`                 | Path inside the archive                | `server/rein-darwin-arm64`. |
| `server.mcp_config.command`          | Same as `entry_point` (with `${__dirname}` prefix) | Claude Desktop spawns this. |
| `server.mcp_config.args`             | `["serve"]`                            | Tells the rein binary to start its stdio MCP server. |
| `server.mcp_config.env`              | `${user_config.*}` substitutions       | Claude Desktop expands these from user-provided values. |
| `user_config.<key>`                  | One per env var the user must / may set | See `manifest.json` for the three fields rein uses. `sensitive: true` triggers encrypted storage in Claude Desktop's settings. |
| `tools_generated`                    | `true`                                 | rein registers tools dynamically — do **not** list 40 tools statically. |
| `compatibility.claude_desktop`       | `">=1.0.0"`                            | DXT support landed in Claude Desktop 1.0. |
| `compatibility.platforms`            | `["darwin"]`                           | rein currently macOS-only DXT. |

If you change `dxt/manifest.json`, run `jq -e . dxt/manifest.json > /dev/null`
to confirm it parses, then re-run `build-dxt.sh` end-to-end.

## Local testing

After `build-dxt.sh` produces `target/rein-v<version>.mcpb`:

1. Open Finder, navigate to `target/`.
2. (Once for unsigned builds) clear macOS quarantine:
   ```bash
   xattr -d com.apple.quarantine target/rein-v<version>.mcpb
   ```
3. Double-click the `.mcpb`. Claude Desktop opens an install dialog.
4. Fill in `Gemini API Key` (any string for tool registration testing — rein
   will fail at first API call but tool registration succeeds).
5. Click Install.
6. Open a new chat in Claude Desktop, ask Claude to list rein tools. Expect
   ~40 `rein_*` tools.
7. Tail the log to watch the spawn:
   ```bash
   tail -F ~/Library/Logs/Claude/mcp-server-rein.log
   ```
8. `Settings → Extensions → rein → Remove` before rebuilding to avoid stale
   state.

## Release flow

The DXT artifact is part of every GitHub Release tagged `v0.28.9` and later.
Steps:

1. Bump `crates/rein/Cargo.toml`'s `version`. Run `cargo build -p rein` to
   refresh the lockfile.
2. Update `dxt/manifest.json`'s `description` / `long_description` if the
   headline features changed (e.g., tool count). The `version` field is
   auto-patched by the build script — leave it as the previous version or
   the next, doesn't matter.
3. Run `./scripts/build-dxt.sh`.
4. Sanity-check the output `.mcpb` size and run the local-install test
   above.
5. `git commit` + tag: `git tag v<version> && git push --tags` (replace
   `<version>` with the value bumped in step 1).
6. Create the GitHub Release. Attach two artifacts:
   - the existing GUI binary (`rein` from `target/release/`), as in
     v0.27.4–v0.28.8 releases
   - `target/rein-v<version>.mcpb`
7. Edit the release notes to point install instructions at the new `.mcpb`
   filename.

## Adding new platforms

The current build is macOS Apple Silicon only. To add another platform
(e.g., Windows x64):

1. Build the binary on a runner of that platform — `cargo build -p rein
   --release --locked` on a Windows GitHub Actions runner produces
   `target/release/rein.exe`.
2. Copy the binary into the staging directory at
   `server/rein-<platform>-<arch>` (e.g., `server/rein-windows-x64.exe`).
3. Add `platform_overrides` to the manifest:
   ```json
   "mcp_config": {
     "command": "${__dirname}/server/rein-darwin-arm64",
     "args": ["serve"],
     "env": { "...": "..." },
     "platform_overrides": {
       "win32": { "command": "${__dirname}/server/rein-windows-x64.exe" }
     }
   }
   ```
4. Update `compatibility.platforms` to `["darwin", "win32"]`.
5. Verify on the target platform before tagging.

If you set up a GitHub Actions matrix that builds all platforms in one job
and assembles a multi-platform `.mcpb`, **codesign and notarization** become
the next concern (see next section).

## Codesign and notarization

The current single-platform build is **unsigned**. Users who download the
`.mcpb` must clear quarantine manually
(`xattr -d com.apple.quarantine`) — this is documented in the user manual.

Why we don't sign today:

- Apple Developer Program membership ($99/year) plus CI integration cost.
- Windows code-signing certificate (~$200/year) is needed too if Windows
  binaries are added.
- The current user base is small enough that the documented `xattr`
  workaround is acceptable.

When to revisit:

- After Windows or Intel Mac users start filing install issues.
- When DXT spec reaches v1.0 stable (a likely re-audit point anyway).

To enable signing later:

- macOS: install the Developer ID Application certificate, run `codesign`
  on the staged binary, then `notarytool submit` and `stapler staple` on the
  final `.mcpb`.
- Windows: use `signtool sign /fd SHA256 /tr http://timestamp.digicert.com
  /td SHA256` on the `.exe` before zipping.

## DXT spec compatibility

This guide is written against DXT spec **v0.3**. The spec is pre-1.0 and
may break compatibility. When upgrading:

1. Read the spec changelog at <https://github.com/anthropics/dxt>.
2. Update `manifest.json`'s `manifest_version` field.
3. Audit any field shape changes (`server.mcp_config`, `user_config`,
   `platform_overrides`, `${__dirname}` substitution semantics).
4. Re-run local testing on a fresh Claude Desktop install.
5. Bump rein's minor version (DXT spec change → minor bump, not patch).

## Common pitfalls

- **`zip -X`** strips macOS extended attributes. Without it the staged
  binary inherits HFS resource forks that can confuse Claude Desktop's
  unpacker. The build script already does this; if you zip manually,
  remember the flag.
- **`zip -D`** omits directory entries from the archive. The official
  `@anthropic-ai/mcpb unpack` tool (Anthropic's MCPB validator) fails
  with `ENOENT .../server/` when a directory entry is present, because
  it tries to open the directory as a file. Always include `-D` when
  packaging an `.mcpb`. Codex R3 audit on v0.28.9 caught this — the
  initial pack with bare `zip -r -X` produced an archive that the
  official unpacker rejected.
- **Forgetting `chmod +x`** on the staged binary makes Claude Desktop's
  spawn fail silently. The build script handles this; if you stage by
  hand, remember it.
- **`user_config.<key>.sensitive: true`** is required for API keys.
  Without it, Claude Desktop may write the value to plain-text settings.
- **`tools_generated: true`** is critical for rein. Listing all 40 tools
  statically would be a maintenance burden and a bug on every release that
  adds or removes tools.
- **`server.entry_point` must agree with `mcp_config.command`** (modulo the
  `${__dirname}` prefix Claude Desktop expects in the latter). Drift here
  causes silent failure — Claude Desktop reads `entry_point` for some
  validation paths and `command` for spawn.
- **Staged manifest version must match `Cargo.toml`** — the build script
  enforces this with `jq`. If you skip the script and zip manually, the
  manifest's `version` will be stale.
- **macOS Gatekeeper** is the #1 user-reported install issue. The user
  manual covers it; consider pinning a workaround note in the GitHub
  Release notes too.
- **Claude Desktop logs** live at `~/Library/Logs/Claude/mcp-server-rein.log`.
  When debugging spawn failures, tail this first before assuming the binary
  is at fault.
