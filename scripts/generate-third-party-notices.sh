#!/usr/bin/env bash
# Regenerate THIRD_PARTY_NOTICES.md from the current Cargo.lock dependency tree.
#
# Output: THIRD_PARTY_NOTICES.md (tracked) + THIRD_PARTY_NOTICES.yaml (gitignored).
#
# Run this as part of release prep, or whenever Cargo.lock changes
# substantially. Requires:
#   - cargo install cargo-bundle-licenses
#   - python3 with PyYAML (`pip3 install pyyaml`)

set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v cargo-bundle-licenses >/dev/null 2>&1; then
  echo "error: cargo-bundle-licenses not found. Install with:"
  echo "  cargo install cargo-bundle-licenses"
  exit 1
fi

if ! python3 -c "import yaml" 2>/dev/null; then
  echo "error: PyYAML not available. Install with:"
  echo "  pip3 install pyyaml"
  exit 1
fi

echo "==> Regenerating THIRD_PARTY_NOTICES.yaml ..."
cargo bundle-licenses --format yaml --output THIRD_PARTY_NOTICES.yaml

echo "==> Converting to THIRD_PARTY_NOTICES.md ..."
python3 scripts/yaml-to-third-party-md.py

echo "==> Done. Verify with:"
echo "    head -20 THIRD_PARTY_NOTICES.md"
