# ADR — Distribution channels for rein

**Status:** Adopted, v0.28.9
**Date:** 2026-05-06

## Context

Through v0.28.8, rein was distributed only via `cargo install`. This works
for the developer audience that uses Claude Code / Codex CLI, but excludes
Claude Desktop users who do not have a Rust toolchain. Two new channels
were evaluated for v0.28.9.

### Candidates considered

1. **Claude Code plugin marketplace** — `.claude-plugin/marketplace.json` at
   the repo root, plus per-plugin `plugin.json` and `.mcp.json`. Surfaces
   rein in `/plugin marketplace add lyr1cs/rein`. The plugin registers the
   MCP server entry; the user still needs the `rein` binary on `PATH`.
2. **Claude Desktop Extension (DXT, `.mcpb`)** — zip archive containing a
   manifest and a precompiled binary. Anthropic's official one-click MCP
   install format for Claude Desktop. No user-side toolchain required.
3. **Homebrew formula** — adds a `brew install rein` step before any MCP
   setup. Provides packaging but no MCP integration.
4. **nix flake** — same packaging story, smaller audience overlap with
   rein's user base.
5. **Multi-platform DXT (macOS arm64/x64 + Linux x64/arm64 + Windows x64)** —
   one `.mcpb` covering everything via `platform_overrides`.
6. **Codesigned and notarized DXT** — eliminates the `xattr -d
   com.apple.quarantine` step on macOS.

### Considerations

- **Claude Desktop platform reality**: Anthropic ships official Claude
  Desktop on macOS and Windows. There is no official Linux Claude Desktop,
  which deprioritizes Linux DXT.
- **rein user base**: heavy macOS Apple Silicon overlap (Claude Code /
  Codex / OpenCode developer tooling). Intel Mac and Windows are minority
  platforms today.
- **Cross-platform compilation**: rein has native dependencies (Tantivy,
  usearch, sqlite-vec). Cross-compilation across Linux / Windows is
  unverified and likely needs CI work.
- **Codesigning cost**: Apple Developer Program $99/year plus CI
  integration; Windows code-signing certificate ~$200/year. Combined with
  notarization tooling, this is multi-day setup work.

## Decision

For v0.28.9, ship two channels:

1. **Claude Code plugin marketplace** — three-file manifest addition. Users
   with Claude Code can `/plugin marketplace add lyr1cs/rein` and
   `/plugin install rein@rein`. They still need `cargo install rein` (or a
   release binary on `PATH`).
2. **Claude Desktop DXT, macOS Apple Silicon only** — single-platform
   `.mcpb` attached to GitHub Releases. Unsigned; documented `xattr`
   workaround.

Rejected for v0.28.9:

- Homebrew / nix — provide no advantage over `cargo install` for the
  current user base.
- Multi-platform DXT — Linux is moot (no official client), and Windows /
  Intel Mac demand is not yet evidenced. Cross-compile validation is real
  work that should follow demand, not precede it.
- Codesign / notarization — cost not justified by current user volume.
  Documented manual workaround is sufficient.

## Consequences

- macOS Apple Silicon Claude Desktop users get a one-click experience.
- Other platforms are explicitly directed to the Claude Code plugin
  marketplace + `cargo install` path. Both README and the manual document
  this fallback.
- Adding new platforms in the future is a manifest + CI change, not a
  breaking release.
- The `xattr -d com.apple.quarantine` requirement is documented in the
  user manual's troubleshooting section. Expected to generate user issues;
  the response template is "this is by design until codesign is enabled".

## Re-evaluation triggers

Revisit this decision if any of the following hold:

- A Windows Claude Desktop user files an install issue → consider adding
  Windows DXT (still without codesign).
- Cumulative Intel Mac install attempts (visible via GitHub Release
  download stats) cross 5% of total `.mcpb` downloads → consider Intel Mac
  DXT.
- DXT spec reaches v1.0 stable → audit manifest for breaking changes,
  re-evaluate codesign cost in the context of any new spec features.
- Anthropic ships an official Linux Claude Desktop → consider Linux DXT.
- Any single quarter sees >5 distinct user issues blamed on the unsigned
  binary → fast-track codesign work even ahead of the v1.0 spec.
