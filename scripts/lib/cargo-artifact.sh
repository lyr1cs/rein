#!/usr/bin/env bash
# Shared helper for packaging scripts: locate the executable Cargo actually
# built instead of guessing `target/release/<name>`.
#
# `cargo build --message-format=json-render-diagnostics` reports every
# compiled artifact, fresh or rebuilt, including the executable's real path.
# That path moves with `CARGO_TARGET_DIR`, `CARGO_BUILD_TARGET`, `--target`
# and `.cargo/config.toml` overrides; a fixed `target/release/<name>` can then
# point at a stale binary from an earlier, differently configured build,
# which a packaging step would ship under the current version number.
#
# Usage:
#   . scripts/lib/cargo-artifact.sh
#   BIN="$(cargo build ... --message-format=json-render-diagnostics | cargo_executable rein)"
#
# Prints the executable path of the named binary target, or nothing when
# the build output contains no such artifact. Requires `jq`.

cargo_executable() {
  local name="$1"
  jq -r --arg name "$name" '
    select(.reason == "compiler-artifact")
    | select(.target.name == $name)
    | select(.target.kind | index("bin"))
    | select(.executable != null)
    | .executable
  ' | tail -n 1
}
