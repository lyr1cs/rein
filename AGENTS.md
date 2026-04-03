# AGENTS.md

## Overview

rein v0.9.0 — Multi-source cross-validated memory MCP server for AI agents. Rust single binary. 25 MCP tools. Self-adaptive engine (M1-M6). 3-channel retrieval (FTS + Vector + KG) with query expansion, LLM reranking, and parallel pipeline.

## Build & Test

```bash
cargo build           # Debug build
cargo test            # All tests must pass (242+)
cargo build --release # Optimized binary (~7MB)
cargo install --path . # Install to ~/.cargo/bin/rein
docker build -t rein . # Docker image (~165MB)
```

## Directory Structure

```
src/
├── main.rs          # CLI entry point (clap subcommands, 20 commands)
├── lib.rs           # Public API re-exports
├── config.rs        # Configuration loading (TOML + env), includes [extract] section
├── ops.rs           # Shared business logic (build_memory, store_memory, gc, upgrade)
├── init.rs          # Auto-configure MCP clients (JSON + TOML)
├── types/           # Memory, Importance, Embedder trait, errors (incl. Extract variant)
├── store/
│   ├── sqlite.rs    # Core CRUD, FTS, vector search, decay, auto_link, organize, recent
│   ├── memoir.rs    # Knowledge graph CRUD, traversal, export
│   ├── knowledge.rs # Knowledge units, evolution, linking, organizing
│   ├── quality.rs   # Self-learning quality scoring, pruning, recall tracking
│   ├── schema.rs    # DDL, migrations, model-change detection
│   ├── migrate.rs   # QMD import, reindex
│   ├── fts.rs       # FTS5 search with sanitization
│   ├── vec.rs       # sqlite-vec operations
│   ├── hnsw.rs      # HNSW approximate nearest neighbor (usearch)
│   ├── tantivy_fts.rs # Tantivy BM25 full-text search (BooleanQuery topic filter)
│   ├── adaptive.rs  # Feedback event sourcing, AdaptiveState cache, per-consumer offsets
│   ├── hdbscan.rs   # Pure Rust HDBSCAN clustering (dendrogram → condensed tree → EOMBST)
│   └── tiering.rs   # Three-tier memory (Hot/Warm/Cold) with streaming quantile estimator
├── embed/
│   ├── gemini.rs    # Google Gemini embedding API
│   ├── omlx.rs      # OMLX local embedding (OpenAI-compatible)
│   └── cache.rs     # Embedding cache with TTL
├── search/
│   ├── recall.rs    # 3-channel recall pipeline (FTS + Vector + KG) + RRF/CC fusion + R2 rerank
│   ├── classify.rs  # Query routing (Episodic/Temporal/Preference/ExactKeyword/Semantic/Exploratory)
│   ├── kg_search.rs # KG retrieval: concept FTS + BFS "land and expand" with temporal filtering
│   ├── rerank.rs    # Multi-feature reranker (8 features, learned weights from M1/M2)
│   ├── rrf.rs       # Reciprocal Rank Fusion + Convex Combination
│   ├── scoring.rs   # Ebbinghaus decay + KM survival curve scoring
│   ├── warmup.rs    # Background warmup: embeddings + HNSW/Tantivy rebuild
│   ├── chunker.rs   # Semantic text chunking
│   ├── alpha_optimizer.rs # Counterfactual offline alpha optimization for CC fusion
│   └── survival.rs  # Kaplan-Meier non-parametric survival analysis for adaptive decay
├── extract/
│   ├── llm.rs       # LLM extraction (Gemini + OMLX/Ollama), fallback to patterns
│   ├── postprocess.rs # Rule-based post-processing (date keywords, preference tagging, knowledge update)
│   ├── patterns.rs  # Rule-based keyword scoring (fallback when LLM unavailable)
│   ├── hooks/       # Four-layer hook commands (parsing, buffer, scoring, mod)
│   └── dedup.rs     # Similarity (Jaccard + containment)
├── sync/
│   ├── supermemory.rs # Supermemory v4 API client
│   ├── auto_memory.rs # ~/.claude/ file scanner
│   └── validate.rs    # Cross-source validation
└── mcp/
    ├── server.rs    # MCP server (24 tools, stdio + HTTP/SSE)
    ├── tools.rs     # Tool parameter structs
    └── compact.rs   # Output formatters (recall now includes content)

bench/               # LongMemEval benchmark adapters
├── longmemeval_fast.py  # Fast parallel adapter (per-question temp DB)
└── longmemeval_adapter.py # MCP HTTP adapter

Dockerfile           # Multi-stage build (rust:latest → debian:trixie-slim)
docker-compose.yml   # One-command deployment
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
- hook_prompt is kept as a compatibility no-op; rein no longer auto-injects prompt memory
- Post-store processing (auto_link, evolve) only runs on newly created memories, not merges
- postprocess enriches keywords only — caller-supplied topic/importance are authoritative

## Common Pitfalls

- Don't add async to MemoryStore trait methods (they're intentionally sync)
- Don't use reqwest::blocking inside tokio — use tokio::task::block_in_place
- String slicing: always use .chars() for CJK-safe truncation, never byte indexing
- DOT export: use escape_dot() for all user-provided strings
- Cross-memoir links are forbidden (validated in add_link)
- LLM JSON output may be wrapped in code fences — use `strip_code_fences()` before parsing
- Some local models don't support `response_format: {"type": "json_object"}`
- BFS traversal must filter expired links (valid_from/valid_until)
- KG search seeds by concept ID, not name (avoids cross-memoir collision)

## Environment Variables

GEMINI_API_KEY, SUPERMEMORY_CC_API_KEY, REIN_HTTP_TOKEN, REIN_DB, REIN_LOG, REIN_SSE_BIND, REIN_SSE_PORT
