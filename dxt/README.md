# dxt/

Source for the rein Claude Desktop Extension (`.mcpb`). The packaged
artifact is uploaded to each GitHub Release as `rein-v<version>.mcpb`.

## Files

- `manifest.json` — DXT spec v0.3 manifest. Declares the binary, env vars,
  and `user_config` fields Claude Desktop prompts for.
- `README.md` — this file.

The build script at `scripts/build-dxt.sh` stages this directory plus the
freshly compiled `target/release/rein` binary, patches the manifest version
from `Cargo.toml` via `jq`, and produces `target/rein-v<version>.mcpb`.

## End-user install

See [docs/manual/02-installation.md](../docs/manual/02-installation.md#claude-desktop-one-click-via-dxt).

## Build instructions for maintainers

See [docs/guides/dxt-build.md](../docs/guides/dxt-build.md).
