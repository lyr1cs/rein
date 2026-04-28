#!/usr/bin/env bash
set -euo pipefail

echo "Installing rein..."

# Check for Rust
if ! command -v cargo &>/dev/null; then
    echo "Error: cargo not found. Install Rust: https://rustup.rs"
    exit 1
fi

build_gui="${REIN_INSTALL_GUI:-1}"

if [[ "$build_gui" == "1" ]]; then
    if command -v npm &>/dev/null; then
        echo "Building GUI assets..."
        (cd crates/rein/gui && npm ci && npm run build)
        cargo install --path crates/rein --locked --features gui
    else
        echo "npm not found; installing CLI-only binary."
        echo "Install Node.js and rerun with REIN_INSTALL_GUI=1 for embedded GUI support."
        cargo install --path crates/rein --locked
    fi
else
    cargo install --path crates/rein --locked
fi

echo "rein installed successfully!"
echo "Run 'rein init' to configure your MCP clients."
