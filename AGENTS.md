# AGENTS.md

## Overview

rein — Multi-source cross-validated memory MCP server for AI agents. Rust single binary. 22 MCP tools.

## Build & Test

```bash
cargo build           # Debug build
cargo test            # All tests must pass (118+)
cargo build --release # Optimized binary (~7MB)
cargo install --path . # Install to ~/.cargo/bin/rein
```

## Directory Structure

```
src/
├── main.rs          # CLI entry point (clap subcommands, 20 commands)
├── lib.rs           # Public API re-exports
├── config.rs        # Configuration loading (TOML + env), includes [extract] section
├── init.rs          # Auto-configure MCP clients (JSON + TOML)
├── types/           # Memory, Importance, Embedder trait, errors (incl. Extract variant)
├── store/
│   ├── sqlite.rs    # Core CRUD, FTS, vector search, decay, auto_link, organize, recent
│   ├── memoir.rs    # Knowledge graph CRUD, traversal, export
│   ├── schema.rs    # DDL, migrations, model-change detection
│   ├── migrate.rs   # QMD import, reindex
│   ├── fts.rs       # FTS5 search with sanitization
│   ├── vec.rs       # sqlite-vec operations
│   ├── hnsw.rs      # HNSW approximate nearest neighbor (usearch)
│   └── tantivy_fts.rs # Tantivy BM25 full-text search (BooleanQuery topic filter)
├── embed/
│   ├── gemini.rs    # Google Gemini embedding API
│   ├── omlx.rs      # OMLX local embedding (OpenAI-compatible)
│   └── cache.rs     # Embedding cache with TTL
├── search/
│   ├── recall.rs    # Unified recall pipeline (waterfall + cross-validation)
│   ├── rrf.rs       # Reciprocal Rank Fusion
│   ├── scoring.rs   # Ebbinghaus decay formula
│   ├── waterfall.rs # Waterfall search strategy
│   ├── warmup.rs    # Background warmup: embeddings + HNSW/Tantivy rebuild
│   └── chunker.rs   # Semantic text chunking
├── extract/
│   ├── llm.rs       # LLM extraction (Gemini + OMLX/Ollama), fallback to patterns
│   ├── patterns.rs  # Rule-based keyword scoring (fallback when LLM unavailable)
│   ├── hooks/       # Four-layer hook commands (parsing, buffer, scoring, mod)
│   └── dedup.rs     # Similarity (Jaccard + containment)
├── sync/
│   ├── supermemory.rs # Supermemory v4 API client
│   ├── auto_memory.rs # ~/.claude/ file scanner
│   └── validate.rs    # Cross-source validation
└── mcp/
    ├── server.rs    # MCP server (22 tools, stdio + HTTP)
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
- Per-request connection model: each MCP request opens its own `SqliteStore` with `SQLITE_OPEN_FULL_MUTEX`
- `store_with_dedup` uses `BEGIN IMMEDIATE` to prevent concurrent dedup races
- HNSW and Tantivy side indexes are updated on every store/update/delete (fire-and-forget)
- Warmup always rebuilds HNSW and Tantivy indexes before processing new embeddings
- LLM extraction falls back to rule-based patterns when provider is unavailable
- Tantivy topic filtering uses BooleanQuery at index level (not post-filter)
- Auto-link creates bidirectional related_ids on store
- `max_input_chars=0` only allowed for known 1M-token Gemini models (safety fallback to 16K)
- Beta values read from `MemoryLayer::beta()`, not hardcoded

## Common Pitfalls

- Don't add async to MemoryStore trait methods (they're intentionally sync)
- Don't use reqwest::blocking inside tokio — use tokio::task::block_in_place
- String slicing: always use .chars() for CJK-safe truncation, never byte indexing
- DOT export: use escape_dot() for all user-provided strings
- Cross-memoir links are forbidden (validated in add_link)
- LLM JSON output may be wrapped in code fences — use `strip_code_fences()` before parsing
- Some local models don't support `response_format: {"type": "json_object"}`

## Environment Variables

GEMINI_API_KEY, SUPERMEMORY_CC_API_KEY, REIN_HTTP_TOKEN, REIN_DB, REIN_LOG
