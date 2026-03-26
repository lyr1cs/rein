# CLAUDE.md

> See [AGENTS.md](AGENTS.md) for full directory structure, invariants, and pitfalls.

## Build & Test

```bash
cargo build
cargo test            # 118+ tests, all must pass
cargo install --path . # Install to ~/.cargo/bin/rein
```

## Architecture

rein is a multi-source cross-validated memory MCP server (24 tools). Key modules:

- `extract/llm.rs` — LLM extraction (Gemini 3.1 Flash Lite), fallback to rule-based
- `extract/hooks/` — 4 hooks: post (PostToolUse), compact (PreCompact), prompt (UserPromptSubmit), stop (Stop)
- `search/recall.rs` — 3-level waterfall (Tantivy BM25 → HNSW → Gemini API) + RRF/CC fusion + Ebbinghaus decay
- `search/classify.rs` — Autonomous retrieval routing (Temporal/ExactKeyword/Semantic/Exploratory)
- `store/sqlite.rs` — Per-request connection model, Tantivy singleton cache
- `store/knowledge.rs` — Knowledge units, evolution, linking, organizing
- `store/quality.rs` — Self-learning quality scoring, pruning, recall tracking
- `store/memoir.rs` — Knowledge graph CRUD, temporal revisions, episodes, BFS traversal

## Temporal Knowledge Graph (v0.4.0)

- `concept_revisions` table — revision history auto-snapshotted on refine
- `episodes` table — session nodes with concept_ids + memory_ids
- ConceptLink has `valid_from`/`valid_until` temporal windows
- BFS traversal skips expired links
- MCP tools: `rein_timeline`, `rein_concept_history`, `rein_recall` with `from`/`to` params

## Autonomous Retrieval Routing (v0.4.0)

Query classifier routes to optimal search strategy:
- **Temporal** ("when did X change?") → BM25 bias (alpha=0.7), auto-inject time bounds
- **ExactKeyword** ("SqliteStore") → heavy BM25 (alpha=0.85)
- **Semantic** ("memory management strategies") → vector dominant (alpha=0.3)
- **Exploratory** ("what do I know about...") → balanced (alpha=0.5), 2x limit

MCP response includes `[route: type]` prefix for transparency.

## Environment Variables

GEMINI_API_KEY, SUPERMEMORY_CC_API_KEY, REIN_HTTP_TOKEN, REIN_DB, REIN_LOG
