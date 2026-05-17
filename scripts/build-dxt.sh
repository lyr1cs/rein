#!/usr/bin/env bash
# Build the rein Claude Desktop Extension (.mcpb) for macOS Apple Silicon.
#
# Output: target/rein-v<version>.mcpb in repo root.
#
# Prerequisites:
#   - cargo (Rust toolchain matching crates/rein/Cargo.toml's MSRV)
#   - npm + node (for the embedded Neural Wiki GUI; `gui/dist/` is .gitignore'd
#     and rust-embed fails the cargo build if it isn't built first)
#   - jq (brew install jq)
#   - zip (preinstalled on macOS)
#   - Apple Silicon Mac (the script does not cross-compile)
#
# Run from the repo root:  ./scripts/build-dxt.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Build hygiene: normalize the absolute filesystem paths Rust's
# `file!()` macro expands into release binaries (panic locations,
# `tracing::event!` callsites) so the binary doesn't embed
# build-environment-specific path strings.
#
# `CARGO_ENCODED_RUSTFLAGS` (not plain `RUSTFLAGS`) is used so the
# replacement string can contain whitespace if desired — the encoded
# form delimits flags with ASCII Unit Separator (\x1f) instead of
# whitespace.
#
# Override the default replacement via env var:
#   HOME_REMAP=your-handle ./scripts/build-dxt.sh
#
# If either `CARGO_ENCODED_RUSTFLAGS` or `RUSTFLAGS` is already set in
# the environment, leave it alone — the caller has explicit control.
# Setting both is a cargo error, so a default is only applied when
# neither is present.
: "${HOME_REMAP:=user}"
if [ -z "${CARGO_ENCODED_RUSTFLAGS:-}" ] && [ -z "${RUSTFLAGS:-}" ]; then
  CARGO_HOME_PATH="${CARGO_HOME:-$HOME/.cargo}"
  # Each `--remap-path-prefix=…` is one flag, joined by `\x1f`.
  export CARGO_ENCODED_RUSTFLAGS=$'--remap-path-prefix='"$HOME"'='"$HOME_REMAP"$'\x1f''--remap-path-prefix='"$CARGO_HOME_PATH"'='"$HOME_REMAP"'/.cargo'
fi

# Sanity: are we on Apple Silicon?
if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
  echo "error: build-dxt.sh only supports macOS Apple Silicon (uname=$(uname -sm))." >&2
  echo "       For other platforms, see docs/guides/dxt-build.md → Adding new platforms." >&2
  exit 1
fi

# Sanity: required tools.
for tool in cargo npm jq zip; do
  if ! command -v "$tool" > /dev/null 2>&1; then
    echo "error: required tool '$tool' not found on PATH." >&2
    echo "       Prereqs: cargo (rustup), npm (node), jq (brew install jq), zip (preinstalled)." >&2
    exit 1
  fi
done

# Read version from crates/rein/Cargo.toml — the single source of truth.
VERSION="$(grep -m1 '^version' crates/rein/Cargo.toml | sed -E 's/version[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/')"
if [ -z "$VERSION" ]; then
  echo "error: could not extract version from crates/rein/Cargo.toml" >&2
  exit 1
fi

OUTPUT="$REPO_ROOT/target/rein-v${VERSION}.mcpb"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

# Build GUI assets first. crates/rein/gui/dist/ is .gitignore'd, and
# rust-embed's #[derive(Embed)] for `GuiAssets` reads `gui/dist/` at compile
# time — without these assets, `cargo build --features gui` fails on a
# clean checkout. Re-running is idempotent (Vite handles incremental
# rebuilds and `npm ci` is a no-op when the lockfile matches).
echo "==> Building Neural Wiki GUI assets (npm ci + npm run build)"
( cd crates/rein/gui && npm ci && npm run build )

echo "==> Building rein v$VERSION (release, --features gui)"
cargo build -p rein --release --locked --features gui

if [ ! -f target/release/rein ]; then
  echo "error: target/release/rein not produced by cargo build" >&2
  exit 1
fi

echo "==> Staging .mcpb contents in $STAGE"
cp dxt/manifest.json "$STAGE/manifest.json"
if [ -f dxt/icon.png ]; then
  cp dxt/icon.png "$STAGE/icon.png"
fi
mkdir -p "$STAGE/server"
cp target/release/rein "$STAGE/server/rein-darwin-arm64"
chmod +x "$STAGE/server/rein-darwin-arm64"

# Sync the manifest version with Cargo.toml (single source of truth).
jq --arg v "$VERSION" '.version = $v' "$STAGE/manifest.json" > "$STAGE/manifest.json.tmp"
mv "$STAGE/manifest.json.tmp" "$STAGE/manifest.json"

# Validate the manifest one more time before zipping.
if ! jq -e . "$STAGE/manifest.json" > /dev/null; then
  echo "error: staged manifest.json failed jq validation" >&2
  exit 1
fi

mkdir -p "$REPO_ROOT/target"
rm -f "$OUTPUT"

echo "==> Packaging $OUTPUT"
# zip flags:
#   -X  strip macOS extended attributes; without this Claude Desktop's
#       unpacker can misbehave on HFS resource forks.
#   -D  omit directory entries from the zip. The official
#       @anthropic-ai/mcpb unpack tool fails with `ENOENT .../server/`
#       when a directory entry like `server/` is present (it tries to
#       open the directory as a file). Codex R3 verified that rebuilding
#       with -D produces an archive that unpack accepts.
( cd "$STAGE" && zip -r -X -D "$OUTPUT" . )

SIZE="$(du -h "$OUTPUT" | cut -f1)"
echo
echo "==> Done."
echo "    Artifact: $OUTPUT ($SIZE)"
echo
echo "Local install test:"
echo "    xattr -d com.apple.quarantine \"$OUTPUT\" 2>/dev/null || true"
echo "    open \"$OUTPUT\""
