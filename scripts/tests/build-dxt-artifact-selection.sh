#!/usr/bin/env bash
# Regression test for scripts/lib/cargo-artifact.sh (audit U22-01).
#
# Scenario: an old `target/release/rein` sits in the checkout while Cargo,
# honouring a target-directory or build-target override, builds the current
# source somewhere else. Packaging must stage the executable Cargo reports,
# never the stale default path.
#
# Run from the repo root:  bash scripts/tests/build-dxt-artifact-selection.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../lib/cargo-artifact.sh
. "$REPO_ROOT/scripts/lib/cargo-artifact.sh"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

mkdir -p "$WORK/target/release" "$WORK/override/aarch64-apple-darwin/release"
printf 'old build\n' > "$WORK/target/release/rein"
printf 'current build\n' > "$WORK/override/aarch64-apple-darwin/release/rein"

# What `cargo build --message-format=json-render-diagnostics` emits: a
# dependency's library artifact, a build script, the `rein` library target,
# a `rein` binary from another crate, then the real `rein` binary.
stream() {
  cat <<EOF
{"reason":"compiler-artifact","target":{"name":"libc","kind":["lib"]},"executable":null}
{"reason":"build-script-executed","package_id":"rein 1.5.0"}
{"reason":"compiler-artifact","target":{"name":"rein","kind":["lib"]},"executable":null}
{"reason":"compiler-artifact","target":{"name":"rein-eval","kind":["bin"]},"executable":"$WORK/override/aarch64-apple-darwin/release/rein-eval"}
{"reason":"compiler-artifact","target":{"name":"rein","kind":["bin"]},"executable":"$WORK/override/aarch64-apple-darwin/release/rein","fresh":true}
{"reason":"build-finished","success":true}
EOF
}

selected="$(stream | cargo_executable rein)"
expected="$WORK/override/aarch64-apple-darwin/release/rein"
if [ "$selected" != "$expected" ]; then
  echo "FAIL: selected '$selected', expected '$expected'" >&2
  exit 1
fi
if [ "$(cat "$selected")" != "current build" ]; then
  echo "FAIL: selected executable is not the current build" >&2
  exit 1
fi

# No `rein` binary in the report (a failed or partial build): nothing is
# selected, so the packaging script stops instead of falling back to a path.
none="$(printf '%s\n' '{"reason":"compiler-artifact","target":{"name":"rein","kind":["lib"]},"executable":null}' | cargo_executable rein)"
if [ -n "$none" ]; then
  echo "FAIL: selected '$none' from a report without a rein binary" >&2
  exit 1
fi

echo "ok: packaging selects the executable Cargo reports"
