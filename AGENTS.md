# AGENTS.md

## Overview

rein v0.18.2 — Multi-source cross-validated memory MCP server for AI agents. Rust single binary. 28 MCP tools. Self-adaptive engine (M1-M6). 3-channel retrieval (FTS + Vector + KG) with query expansion, LLM reranking, and parallel pipeline. Transparent LLM proxy (record-only) including **Codex subscription loopback proxy** with first-party WebSocket mirror, `permessage-deflate` decoding, ChatGPT backend helper routing (`/wham/*`, `/connectors/*`, `/authenticate_app_v2`, `/codex/safety/arc`), and `ArtifactMirrorOnly` recording gate. Async memory pipeline with file-based queue and background worker. Unified dedup architecture (canonical/evidence/ledger). Canonical-first read model, evidence-aware recall, hybrid CJK tokenization (`jieba-rs` + character bigrams) across Tantivy/FTS/dedup/classify, cluster-aware admission, embedding cross-topic dedup, ANN fallback for large unclustered dedup buckets, survival-driven STM promotion, session chunking for long-text extraction, and context-aware extraction with existing-memory injection. Service management (dashboard, gui on/off, proxy on/off). Neural Wiki GUI (React + Tailwind, embedded via rust-embed) with `/api/artifacts` readback panel.

## v0.18.0 highlights (2026-04-17)

- **Codex subscription proxy Phase C/D**: `codexsubp` (recommended loopback) + `codexsubpws` (experimental WS-first) entrypoints via `rein init --proxy`. `chatgpt_base_url` pinned to `http://127.0.0.1:PORT/backend-api/codex` for Codex CLI `PathStyle::from_base_url` compatibility. `/api/codex/*` path family also accepted.
- **Security hardening (28 fixes, 29 regression tests)**: WS deflate-bomb cap (1 MiB), JWT `exp` validation + redact helper, `/api/artifacts` `require_read_token` gate, `expand.rs` prompt-injection defense, rerank strict validation, KM degenerate-curve early-return, HDBSCAN single-point guard, deterministic tiering reservoir, dedup `Option<f32>` API for both-empty sets, adaptive cache TTL.
- **New config**: `config.search.strong_signal_{ratio,single}`, `config.adaptive.cache_ttl_secs`.

## Build & Test

```bash
cargo build           # Debug build
cargo test            # All tests must pass (666+)
cargo build --release # Optimized binary (~7MB)
cargo install --path . # Install to ~/.cargo/bin/rein
docker build -t rein . # Docker image (~165MB)
```

## Directory Structure

```
src/
├── main.rs          # CLI entry point (clap subcommands, 20+ commands)
├── commands.rs      # CLI command handler bodies (extracted from main.rs)
├── lib.rs           # Public API re-exports
├── config.rs        # Configuration loading (TOML + env), includes [extract] section
├── ops/             # Shared business logic (modularized)
│   ├── mod.rs       # Ingestion, GC, topic utilities, re-exports
│   ├── adaptive.rs  # M2-M6 adaptive pipeline, alpha learning, clustering, tiering, cluster profiles for UI
│   ├── dedup.rs     # Dedup strategies, merge, batch dedup (cluster-grouped, ANN fallback for None bucket)
│   └── consolidation.rs # Topic consolidation, cleanup orchestration, summary normalization
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
│   ├── jieba_tokenizer.rs # Custom Tantivy tokenizer: jieba-rs word segmentation + CJK bigrams
│   ├── adaptive.rs  # Feedback event sourcing, AdaptiveState cache, per-consumer offsets
│   ├── hdbscan.rs   # Pure Rust HDBSCAN clustering (dendrogram → condensed tree → EOMBST)
│   └── tiering.rs   # Three-tier memory (Hot/Warm/Cold) with streaming quantile estimator
├── embed/
│   ├── gemini.rs    # Google Gemini embedding API
│   ├── omlx.rs      # OMLX local embedding (OpenAI-compatible)
│   └── cache.rs     # Embedding cache with TTL
├── search/
│   ├── recall.rs    # 3-channel recall pipeline (FTS + Vector + KG) + evidence-aware rerank + RRF/CC fusion + R2 rerank
│   ├── classify.rs  # Query routing (Episodic/Temporal/Preference/ExactKeyword/Semantic/Exploratory)
│   ├── kg_search.rs # KG retrieval: concept FTS + BFS "land and expand" with temporal filtering
│   ├── rerank.rs    # Multi-feature reranker (8 features, learned weights from M1/M2)
│   ├── rrf.rs       # Reciprocal Rank Fusion + Convex Combination
│   ├── scoring.rs   # Ebbinghaus decay + KM survival curve scoring
│   ├── warmup.rs    # Background warmup: embeddings + HNSW/Tantivy rebuild
│   ├── chunker.rs   # Semantic text chunking
│   ├── alpha_optimizer.rs # Counterfactual offline alpha optimization for CC fusion (now includes KG/episode/support/diversity signals)
│   ├── expand.rs    # Query expansion (Gemini Flash Lite / OMLX dual backend) → 2-3 query variants
│   ├── rerank_llm.rs # LLM reranker (Gemini / OMLX) + strong-signal bypass
│   ├── mmr.rs       # Maximal Marginal Relevance re-ranking for result diversity
│   └── survival.rs  # Kaplan-Meier non-parametric survival analysis for adaptive decay
├── extract/
│   ├── llm.rs       # LLM extraction (Gemini + OMLX/Ollama), fallback to patterns
│   ├── postprocess.rs # Rule-based post-processing (date keywords, preference tagging, knowledge update)
│   ├── patterns.rs  # Rule-based keyword scoring (fallback when LLM unavailable)
│   ├── hooks/       # Hook commands + async pipeline
│   │   ├── mod.rs   # Hook orchestration (post, compact, prompt no-op, stop)
│   │   ├── queue.rs # Async memory queue (file-based, flock-protected, crash-safe)
│   │   ├── working_set.rs # Project-scoped memory surfaces (working set + always-on index)
│   │   ├── persist.rs # Memory persistence + working-set updates
│   │   ├── parsing.rs # JSON payload extraction, agent detection
│   │   ├── buffer.rs  # Session buffer I/O
│   │   └── scoring.rs # Signal scoring and filtering
│   ├── dedup.rs     # Similarity (hybrid CJK tokenization: jieba-rs + bigrams, Jaccard + containment, hot-path cluster-aware hints)
│   └── intelligent_merge.rs # LLM-driven semantic verdict classifier (opt-in: ignore/update/merge/create_new for gray-zone cases)
├── sync/
│   ├── supermemory.rs # Supermemory v4 API client
│   ├── auto_memory.rs # ~/.claude/ file scanner
│   └── validate.rs    # Cross-source validation
├── proxy/
│   ├── mod.rs       # Transparent proxy server (record-only, dedicated store thread)
│   ├── provider.rs  # Provider detection (Anthropic / OpenAI, extensible for Gemini)
│   ├── anthropic.rs # Anthropic /v1/messages format handling
│   ├── openai.rs    # OpenAI /v1/chat/completions format handling
│   ├── policy.rs    # Extraction policy decisions
│   └── extract.rs   # Async response extraction + queue integration
└── mcp/
    ├── server.rs    # MCP server (28 tools, stdio + HTTP/SSE + GUI)
    ├── rest.rs      # REST API layer (21 JSON endpoints for GUI)
    ├── tools.rs     # Tool parameter structs
    └── compact.rs   # Output formatters

gui/                 # Neural Wiki GUI (React 18 + TypeScript + Tailwind + Vite)
├── src/
│   ├── App.tsx      # Router + QueryClientProvider
│   ├── api/         # Fetch wrapper, TypeScript types
│   ├── hooks/       # TanStack Query hooks with configurable polling
│   ├── components/  # Layout (icon sidebar + vitals header)
│   └── pages/       # 8 pages: Dashboard, Brain, Memories, Adaptive, Graph, Timeline, Artifacts, Settings
└── vite.config.ts   # Dev proxy + manual chunks for react/charts/graph vendors

Dockerfile           # Multi-stage build (rust:latest → debian:trixie-slim)
docker-compose.yml   # One-command deployment
```

## Key Invariants

- All SQL uses parameterized queries (except vec table DDL which uses usize)
- FTS5 queries sanitized via `sanitize_fts_query()`
- LIKE queries escape `%` and `_`
- HTTP server requires REIN_HTTP_TOKEN for non-localhost bind
- Dedup threshold: per-cluster adaptive via A1 (P90 intra-cluster similarity); 0.70 global fallback. All paths (store, batch, vec dedup, CLI, MCP) use `get_dedup_threshold(cluster_id)` via `ops::effective_dedup_threshold()`
- Intelligent merge (opt-in via `[intelligent_merge] enabled = true`): LLM pre-flight classification of gray-zone (0.50–0.85 sim) pairs chooses ignore/update/merge/create_new. Pre-flight runs OUTSIDE `BEGIN IMMEDIATE` to avoid holding the write lock. Every verdict is logged to `dedup_decisions` with `operator='llm_verdict'`; Update/Merge snapshot the pre-merge existing memory into `memory_evidence` so the prior version is recoverable.
- Single-memory deletion (`SqliteStore::delete`, `rein_forget`, CLI delete) wraps row delete + JSON-array ref cleanup (`concept.source_memory_ids`, `memory.related_ids`, `episodes.memory_ids`) in one `BEGIN IMMEDIATE` so partial failure rolls back atomically
- CJK lexical dedup uses `jieba-rs` word segmentation plus character bigrams
- Vector dimensions: configurable (default 3072)
- FTS5 tokenizer: unicode61 (CJK support)
- Per-request connection model: each MCP request opens its own `SqliteStore` with `SQLITE_OPEN_FULL_MUTEX`
- `store_with_dedup` uses `BEGIN IMMEDIATE` to prevent concurrent dedup races
- `store_with_dedup` may infer cluster hints from local embedding cache, but must never trigger remote embedding calls on cache miss
- HNSW and Tantivy side indexes are updated on every store/update/delete (fire-and-forget)
- Warmup always rebuilds HNSW and Tantivy indexes before processing new embeddings
- LLM extraction falls back to rule-based patterns when provider is unavailable
- Tantivy writer lock failure → graceful skip (side index, not critical)
- Tantivy/HNSW rebuild uses flock to prevent concurrent corruption
- Proxy is record-only (no request modification); extraction via async queue
- Tantivy topic filtering uses BooleanQuery at index level (not post-filter)
- Auto-link creates bidirectional related_ids on store
- `max_input_chars=0` only allowed for known 1M-token Gemini models (safety fallback to 16K)
- Beta values read from `MemoryLayer::beta()`, not hardcoded
- hook_prompt is kept as a compatibility no-op; rein no longer auto-injects prompt memory
- Post-store processing (auto_link, evolve) only runs on newly created memories, not merges
- postprocess enriches keywords only — caller-supplied topic/importance are authoritative
- `/api/memories/:id` is backward-compatible: top-level memory fields remain, plus nested `memory` and `evidence`
- Recall is canonical-first: default result objects are canonicals, while evidence is previewed in recall and expanded on demand
- STM→LTM promotion uses survival-curve-derived thresholds when cluster curves exist, with a fixed fallback otherwise
- Large `cluster_id=None` dedup buckets use ANN candidate generation before pairwise comparison
- Summary display is layered: canonical summaries may be longer, while APIs/UI can expose `summary_short` for list views
- Adaptive status now exposes cluster-level dedup/admission/promotion decisions for the GUI
- Adaptive status now exposes `cluster_profiles` for per-cluster dedup/admission/promotion inspection
- M2 per-cluster alpha keys use format `"<query_type>:<cluster_id>"` (e.g. `"semantic:5"`); these are cleared on recluster via `retain(|k, _| !k.contains(':'))`
- M3 global prior survival curve stored as `"survival_curve:global"` in adaptive snapshot; `recall.rs` must skip `parse::<u32>()` for this key and branch on string `"global"` first
- AdaptiveState `save_snapshot` uses CAS retry (max 3 attempts): read-merge-write with version predicate; fails with `ReinError::Config` after exhaustion — do not swallow this error

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

GEMINI_API_KEY, SUPERMEMORY_CC_API_KEY, REIN_HTTP_TOKEN, REIN_DB, REIN_LOG, REIN_SSE_BIND, REIN_SSE_PORT, REIN_PROXY_BIND, REIN_PROXY_PORT, REIN_PROXY_TOKEN, REIN_PROXY_ACTIVE
