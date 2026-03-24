# CLAUDE.md

## Project

rein — Multi-source cross-validated memory MCP server for AI agents. Single Rust binary.

## Build & Test

```bash
cargo build
cargo test              # 60+ tests, all must pass
cargo build --release   # 6.5MB optimized binary
```

## Architecture

Single crate, module-based: `types/`, `store/`, `embed/`, `search/`, `extract/`, `sync/`, `mcp/`, `config.rs`. 19 MCP tools total (9 core + 10 knowledge graph).

- `store/sqlite.rs` — Core CRUD, uses `SQLITE_OPEN_FULL_MUTEX` for thread safety
- `mcp/server.rs` — MCP tools over stdio, wraps SqliteStore in `Mutex`
- `search/rrf.rs` — Reciprocal Rank Fusion, pure function
- `search/scoring.rs` — Ebbinghaus decay formula
- `extract/dedup.rs` — `similarity()` = max(jaccard, containment) for dedup; `jaccard_similarity()` only for cross-source validation
- `store/memoir.rs` — Knowledge graph CRUD, traversal, export
- Memoir tools: 10 additional MCP tools for graph operations (create, list, show, add_concept, refine, search, search_all, link, inspect, export)

## Key Invariants

- All SQL queries use parameterized statements (never string interpolation)
- FTS5 queries sanitized via `sanitize_fts_query()` (quote every token)
- LIKE queries escape `%` and `_` with `ESCAPE '\'`
- `unsafe impl Send/Sync` on `ReinServer` is justified by `SQLITE_OPEN_FULL_MUTEX`
- Dedup threshold default: 0.70 (using combined similarity, not pure Jaccard)
- Vector dimensions: 3072 (gemini-embedding-001)

## Environment Variables

- `GEMINI_API_KEY` — Required for embedding
- `SUPERMEMORY_CC_API_KEY` — Optional, for Supermemory sync
- `REIN_DB` — Override database path
- `REIN_LOG` — Log level (debug/info/warn/error)
