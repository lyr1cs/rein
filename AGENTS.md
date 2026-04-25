# AGENTS.md

## Overview

rein v0.26.1 — Multi-source cross-validated memory MCP server for AI agents. Rust single binary. 35 MCP tools (v0.26 adds `rein_archive_summary_refresh` for ARS Capability C cold-tier archival summary; v0.25 extends `rein_recall` with opt-in `synthesize=true` for ARS Capability B; v0.24 added `rein_concept_state` + `rein_concept_summary_refresh` for ARS Capability A). Unified operation registry (CLI / MCP / REST authored once via `#[op]` macro — Phase 2.6 closed the last legacy handler, Phase 3 deleted the hand-maintained registry middleman). Self-adaptive engine (M1-M6). 3-channel retrieval (FTS + Vector + KG) with query expansion, LLM reranking, and parallel pipeline. Transparent LLM proxy (record-only) including **Codex subscription loopback proxy** with first-party WebSocket mirror, `permessage-deflate` decoding, ChatGPT backend helper routing (`/wham/*`, `/connectors/*`, `/authenticate_app_v2`, `/codex/safety/arc`), and `ArtifactMirrorOnly` recording gate. Async memory pipeline with file-based queue and background worker. Unified dedup architecture (canonical/evidence/ledger). Canonical-first read model, evidence-aware recall, hybrid CJK tokenization (`jieba-rs` + character bigrams) across Tantivy/FTS/dedup/classify, cluster-aware admission, embedding cross-topic dedup, ANN fallback for large unclustered dedup buckets, survival-driven STM promotion, session chunking for long-text extraction, and context-aware extraction with existing-memory injection. Service management (dashboard, gui on/off, proxy on/off). Neural Wiki GUI (React + Tailwind, embedded via rust-embed) with `/api/artifacts` readback panel.

## v0.26.1 highlights (2026-04-25)

- **D direction query_type wiring fix**: v0.26.0 hardcoded `query_type = "Semantic"` inside `run_recall_synthesis`, which silently routed every non-Semantic query event into one bucket while the per-query gate read another — turning the per-cluster `useful_rate` gate into dead code for Episodic / Temporal / Preference / ExactKeyword / Exploratory queries. v0.26.1 introduces `QueryType::synthesis_bucket_label()` returning the canonical capitalised label and threads it from the recall handler (MCP/CLI + REST) into `decide_synthesize`.
- **Configurable cold-start threshold**: new `[ars].synthesis_cold_start_n` config knob (default 10, matches `SYNTHESIS_COLD_START_N` const). Operators on a fresh canary may lower to 3-5 to let the per-cluster gate fire against the partial event stream a soak collects without waiting for the bootstrap default.
- **`rein-eval cold_archive` subcommand**: parallel to `concept-summary` and `synthesis`. Baseline scores the post-M5-strip surface (`topic + summary` — `memory.content` is replaced by `memory.summary` per `ops/adaptive.rs:750`); `Run` invokes `ColdArchiveSummaryGenerator::generate` over fixture content and scores `topic + summary + archival_summary`; `Compare` runs paired McNemar under the additive `DecideShipKind::Synthesis` rule. 7 fixtures bundled across 4 categories (technical_decisions × 3, narrative_logs × 2, cjk_mixed × 1, multi_topic × 1). `print_summary` now flags `n < 12` corpora as power-limited.
- **Engineering**: 967 tests / 0 clippy / +5 unit tests for cold_archive + +3 for D direction wiring (synthesis_bucket_label correctness, Episodic-query bucket-lookup wiring, configurable cold_start_n).

## v0.26.0 highlights (2026-04-25)

- **ARS Capability C (cold-tier archival summary)**: opt-in `[ars].cold_archive_enabled` flag. Slow-channel worker in `ops/cold_archive_summary.rs` claims cold-tier rows via 5-way CAS (id + per-row ULID `archival_claim_token` + status-live + tier-still-cold + snapshot updated_at + needs-still-1 + superseded-NULL), runs a 3-invariant lossless contract (bounded length INV-3 + script preservation INV-5 catches LLM auto-translation + trigram coverage INV-1 catches fabrication), persists archival_summary on the row. 3-strike per-pass exhaustion fuse, 5-min stale-claim takeover, 180-second batch wall-time budget. New manual refresh op `rein_archive_summary_refresh`. Step 3a in `run_adaptive_pipeline`; the M5 strip stays inside `run_tiering`. v0.26.0 patch (commit `4f51c52`) added a `cold_archive.content` fallback inside `attempt_one` so Cap C reads the original content even when strip ran first.
- **D direction (synthesis interaction events + M1 consumer + per-query adaptive decision)**: new `EventType::SynthesisInteraction` variant with payload (Viewed / ClickedSource / ImmediateRequery / ExplicitThumb + metadata). M1 consumer `synthesis_feedback` strict 5-invariant pattern mirroring `recompute_concept_refresh_stats`. `decide_synthesize` per-query gate cold-start fallback to global flag at events < 10. `RecallSynthesisOutcome.synthesis_id` (ULID) + `skipped_adaptive_decision` flag. `/api/adaptive` extends with `synthesis: { by_cluster, global }` projection; `rein_feedback` MCP tool accepts `kind: "synthesis_interaction"`. GUI: SynthesisCard hooks (dwell + click + thumb), Adaptive page Synthesis Quality panel. Useful_rate formula 9 bootstrap constants; `by_cluster` hard cap 4096 + query_type whitelist normalize. Records-only on first install; gated on Cap B default-on (v0.25.4) + 2-4 weeks canary traffic before adaptive decision actually kicks in.
- **GUI 12-finding cleanup** from v0.25.1 audit (M1-M4 + L1-L7 + L8 unspecified).
- **Engineering**: 958+ tests / 0 clippy / 8-agent parallel fan-out via implementation contract / 2 Codex audit rounds (R1: 4 P1 + 5 P2 / R2: 0 P1 + 3 P2, HIGH/P1 saturation reached).

## v0.25.0 highlights (2026-04-24)

- **ARS Capability B (recall-time synthesis)**: opt-in `synthesize=true` param on `rein_recall` (MCP) + `?synthesize=true` on `/api/memories` (REST) + `--synthesize` on `rein recall` (CLI). When `[ars].recall_synthesis_enabled = true` (default `false`) AND results.len() ≥ `[ars].recall_synthesis_min_results` (default 3) AND an LLM provider is configured, the LLM produces a 3-6 sentence narrative over the top-N results and returns it as `RecallSynthesisOutcome` alongside the normal results list. `/api/recall_stream` intentionally NOT wired (paginated, would duplicate LLM cost).
- **Prompt-size safety net**: `build_synthesis_prompt` honours `extract::llm::resolve_max_input_for_kind(config, &extractor)` budget. Priority-aware truncation: top-ranked memories preserved whole, first overflow truncated mid-content + `[…remaining memories truncated]` notice + remaining dropped. Query itself capped to `max(max_chars / 4, 256)` chars. Final defensive `take(max_chars)` safety net guarantees prompt length ≤ cap regardless of edge cases.
- **Hallucination guardrail**: synthesis SYSTEM_PROMPT explicitly says "synthesize from the provided memories only; do not invent facts" and notes contradictions when memories disagree.
- **No new MCP tool**: Cap B extends existing `rein_recall`, MCP tool count stays at 34. JSON shape stays bit-identical for callers that don't pass `synthesize=true` (synthesis field absent, not null).
- **Tests**: 12 unit tests in `ops/recall_synthesis.rs` (5 outcome states + 7 prompt-cap edge cases including long-query, huge-query, extreme-tight-cap, no-cap path).
- **4 Codex audit rounds**: Round 1 caught REST + CLI not wired (P2x2 fixed), Round 2 caught unbounded prompt size (P2 fixed), Round 3 caught long-query bypass of size cap (P2 fixed), Round 4 clean.

## v0.24.0 highlights (2026-04-24)

- **ARS Capability A (Concept Living Summary)**: `living_summary` field on Concept nodes refreshed via `should_refresh_living_summary` trigger; new MCP tools `rein_concept_state` + `rein_concept_summary_refresh`; cluster-aware refresh-interval percentiles via `ConceptSummaryRefreshed` feedback events.
- **L4 concurrent CAS protection**: `write_living_summary_if_revision_unchanged` predicates on both `revision` AND `living_summary_source_revision IS prior` so two concurrent first-refreshes can't both commit.
- **Cross-cutting peek+commit refactor**: 5 consumers (M2/M3/A1/concept-summary/etc) migrated to peek-then-commit watermark pattern (Codex hammered this 5 HIGHs across 4 rounds before clean).

## v0.20.0 highlights (2026-04-17)

- **Full-stack audit sweep** (6 Explore agents + codex rescue): 45 findings total (1 CRITICAL + 15 HIGH + 17 MEDIUM + 12 LOW/NIT), 509 tests green.
- **CRITICAL data-integrity fix**: `consolidate_by_ids_atomic` now scrubs `memories.related_ids` and `episodes.memory_ids` before deleting — closes the dangling-ref corruption that silently accumulated on every `rein consolidate` / `rein cleanup --all` run.
- **Concurrency**: `apply_decay` uses a `WHERE last_accessed = ?` CAS predicate so a racing `record_access`/`intelligent_merge` no longer gets clobbered by a stale snapshot. New `pending_grayzone_jobs` SQLite table persists gray-zone dedup pairs inside the store transaction; drained at startup so the post-COMMIT enqueue window is no longer silent data loss. `session_artifacts.episode_id` backfilled on startup to heal stop-hook orphan episodes.
- **Algorithms (principled, data-driven)**: alpha optimizer returns None on zero-variance events (preserves prior instead of biasing to 0.0); CC fusion skips tied channels instead of awarding max to all; survival falls back to Ebbinghaus when `event_count == 0`; MMR handles all-negative relevance scores correctly; expanded queries deduped inter-variant as well as against original; expansion thread is cancellation-aware.
- **Security**: REST + proxy signal handling (Ctrl-C + SIGTERM graceful shutdown), GUI responses carry CSP / X-Frame-Options / nosniff / Referrer-Policy, Gemini endpoint scheme validation + retry with exponential backoff + Retry-After, `/api/memories` read gate aligned with `/api/artifacts`, proxy `token_eq` hashes both sides to close length leak, max_input_chars default 0 resolves to 1M-char cap on large-context models instead of unlimited.
- **Storage**: vec_memories DELETE fatal inside transaction (no more ghost embeddings), symmetric HNSW remove / update mark_dirty, session buffer file capped at 16 MiB, `feedback_events` prune ignores stale consumers.
- **GUI/REST polish**: polling interval is now reactive to Settings slider, Run Fix requires confirmation, Graph relation colors match the real server enum, server version shown dynamically via new `/api/version`, Memories delete errors surfaced in a toast, Brain page memoir exports parallelized via `Promise.all`.
- **Cleanup**: WebSocket mirror `assistant_text` capped at 200 KB matching stream_response; path-traversal guard rejects `..`/`//` segments before routing; `consolidate_atomic` / `consolidate_topics_atomic` gated behind `#[cfg(test)]` to close the latent trap; dedup containment direction preserved so new-contains-old supersedes old instead of merging into it.

## v0.18.0 highlights (2026-04-17)

- **Codex subscription proxy Phase C/D**: `codexsubp` (recommended loopback) + `codexsubpws` (experimental WS-first) entrypoints via `rein init --proxy`. `chatgpt_base_url` set to `http://127.0.0.1:PORT/backend-api` (v0.20.1 fix: removed `/codex` suffix — Codex hard-codes `/codex/` in analytics URL and uses `contains("/backend-api")` for `wham/apps`, so trailing `/codex` causes double-prefix 404s). `/api/codex/*` path family also accepted.
- **Security hardening (28 fixes, 29 regression tests)**: WS deflate-bomb cap (1 MiB), JWT `exp` validation + redact helper, `/api/artifacts` `require_read_token` gate, `expand.rs` prompt-injection defense, rerank strict validation, KM degenerate-curve early-return, HDBSCAN single-point guard, deterministic tiering reservoir, dedup `Option<f32>` API for both-empty sets, adaptive cache TTL.
- **New config**: `config.search.strong_signal_{ratio,single}`, `config.adaptive.cache_ttl_secs`.

## Build & Test

This is a 2-crate Cargo workspace since v0.21 A1: `crates/rein` (main) + `crates/rein-macros` (proc-macro for `#[op]`).

```bash
cargo build -p rein               # Debug build of main crate
cargo test --workspace            # All tests across both crates must pass
cargo build -p rein --release     # Optimized binary (~7MB)
cargo install --path crates/rein  # Install to ~/.cargo/bin/rein
docker build -t rein .            # Docker image (~165MB), build context = workspace root
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
    ├── server.rs    # MCP server (31 tools, stdio + HTTP/SSE + GUI)
    ├── rest.rs      # REST API layer (33 inventory routes + 2 legacy derived)
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
