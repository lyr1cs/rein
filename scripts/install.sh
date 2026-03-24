#!/usr/bin/env bash
set -euo pipefail

echo "Installing rein..."

# Check for Rust
if ! command -v cargo &>/dev/null; then
    echo "Error: cargo not found. Install Rust: https://rustup.rs"
    exit 1
fi

# Build from source
cargo install --path . --locked

echo "rein installed successfully!"
echo "Run 'rein init' to configure your MCP clients."
