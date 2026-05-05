# rein (Claude Code plugin)

This is the Claude Code plugin manifest for the [rein](https://github.com/lyr1cs/rein)
memory MCP server. The plugin registers the rein MCP server entry; it does
NOT install the rein binary.

## Install

```text
/plugin marketplace add lyr1cs/rein
/plugin install rein@rein
```

## Prerequisites

The plugin assumes `rein` is on your `PATH`. Install it first:

```bash
cargo install --git https://github.com/lyr1cs/rein --locked rein
# or download a release binary from https://github.com/lyr1cs/rein/releases
```

Set `GEMINI_API_KEY` in your shell environment or `~/.rein/config.toml`.

## Alternative: Claude Desktop one-click

If you use Claude Desktop on macOS Apple Silicon, prefer the DXT (`.mcpb`)
artifact attached to each GitHub Release — it bundles the rein binary and
prompts for `GEMINI_API_KEY` at install time, no `cargo install` needed.
See [the project README](https://github.com/lyr1cs/rein#install-on-claude-desktop-dxt--macos-apple-silicon).

## License

AGPL-3.0-or-later. See [LICENSE](https://github.com/lyr1cs/rein/blob/master/LICENSE).
