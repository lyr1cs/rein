# AGENTS.md

## Overview

rein — Multi-source cross-validated memory MCP server for AI agents. Rust single binary.

## Build & Test

```bash
cargo build           # Debug build
cargo test            # All tests must pass (70+)
cargo build --release # Optimized binary (~7MB)
cargo install --path . # Install to ~/.cargo/bin/rein
```

## Directory Structure

```
src/
├── main.rs          # CLI entry point (clap subcommands)
├── lib.rs           # Public API re-exports
├── config.rs        # Configuration loading (TOML + env)
├── init.rs          # Auto-configure MCP clients (JSON + TOML)
├── types/           # Memory, Importance, Embedder trait, errors
├── store/
│   ├── sqlite.rs    # Core CRUD, FTS, vector search, decay
│   ├── memoir.rs    # Knowledge graph CRUD, traversal, export
│   ├── schema.rs    # DDL, migrations, model-change detection
│   ├── migrate.rs   # QMD import, reindex
│   ├── fts.rs       # FTS5 search with sanitization
│   └── vec.rs       # sqlite-vec operations
├── embed/
│   ├── gemini.rs    # Google Gemini embedding API
│   ├── omlx.rs      # OMLX local embedding (OpenAI-compatible)
│   └── cache.rs     # Embedding cache with TTL
├── search/
│   ├── recall.rs    # Unified recall pipeline (waterfall + cross-validation)
│   ├── rrf.rs       # Reciprocal Rank Fusion
│   ├── scoring.rs   # Ebbinghaus decay formula
│   ├── waterfall.rs # Waterfall search strategy
│   ├── warmup.rs    # Background embedding cache warmup
│   └── chunker.rs   # Semantic text chunking
├── extract/
│   ├── patterns.rs  # Rule-based keyword scoring
│   ├── hooks.rs     # Three-layer hook commands
│   └── dedup.rs     # Similarity (Jaccard + containment)
├── sync/
│   ├── supermemory.rs # Supermemory v4 API client
│   ├── auto_memory.rs # ~/.claude/ file scanner
│   └── validate.rs    # Cross-source validation
└── mcp/
    ├── server.rs    # MCP server (19 tools, stdio + HTTP)
    ├── tools.rs     # Tool parameter structs
    └── compact.rs   # Output formatters
```

## Key Invariants

- All SQL uses parameterized queries (except vec table DDL which uses usize)
- FTS5 queries sanitized via `sanitize_fts_query()`
- LIKE queries escape `%` and `_`
- HTTP server requires REIN_HTTP_TOKEN for non-localhost bind
- Dedup threshold: 0.70 using max(jaccard, containment)
- Vector dimensions: configurable (default 3072)
- FTS5 tokenizer: unicode61 (CJK support)
- `unsafe impl Send/Sync` justified by SQLITE_OPEN_FULL_MUTEX

## Common Pitfalls

- Don't add async to MemoryStore trait methods (they're intentionally sync)
- Don't use reqwest::blocking inside tokio — use tokio::task::block_in_place
- String slicing: always use .chars() for CJK-safe truncation, never byte indexing
- DOT export: use escape_dot() for all user-provided strings
- Cross-memoir links are forbidden (validated in add_link)

## Environment Variables

GEMINI_API_KEY, SUPERMEMORY_CC_API_KEY, REIN_HTTP_TOKEN, REIN_DB, REIN_LOG
