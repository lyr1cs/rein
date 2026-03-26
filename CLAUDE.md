# CLAUDE.md

> See [AGENTS.md](AGENTS.md) for full directory structure, invariants, and pitfalls.

## Build & Test

```bash
cargo build
cargo test            # 102+ tests, all must pass
cargo install --path . # Install to ~/.cargo/bin/rein
```

## Architecture

rein is a multi-source cross-validated memory MCP server (22 tools). Key modules:

- `extract/llm.rs` — LLM extraction (Gemini 3.1 Flash Lite / OMLX local models), fallback to rule-based
- `extract/hooks.rs` — 4 hooks: post (PostToolUse), compact (PreCompact), prompt (UserPromptSubmit), stop (Stop)
- `search/recall.rs` — 3-level waterfall (Tantivy BM25 → HNSW → Gemini API) + RRF fusion + Ebbinghaus decay
- `store/sqlite.rs` — Per-request connection model, auto_link, organize, recent, gc

## Environment Variables

GEMINI_API_KEY, SUPERMEMORY_CC_API_KEY, REIN_HTTP_TOKEN, REIN_DB, REIN_LOG
