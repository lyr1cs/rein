# CLAUDE.md

> See [AGENTS.md](AGENTS.md) for full directory structure, invariants, and pitfalls.

## Build & Test

```bash
cargo build
cargo test            # 189+ tests, all must pass
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
- `store/adaptive.rs` — Feedback event sourcing, AdaptiveState cache, per-consumer offsets
- `store/hdbscan.rs` — Pure Rust HDBSCAN clustering (dendrogram → condensed tree → EOMBST)
- `store/tiering.rs` — Three-tier memory (Hot/Warm/Cold) with streaming quantile estimator
- `search/alpha_optimizer.rs` — Counterfactual offline alpha optimization for CC fusion
- `search/survival.rs` — Kaplan-Meier non-parametric survival analysis for adaptive decay

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

## Adaptive Engine (v0.5.0)

Self-learning engine that replaces fixed parameters with data-driven adaptation:

- **M1 Event Sourcing**: `feedback_events` table captures all recall/store/click signals; AdaptiveState cache with per-consumer offsets enables replay
- **M2 Counterfactual Alpha Optimization**: Learns optimal `cc_alpha` from candidate logs — replays past searches with alternative weights to find the alpha that would have maximized relevance
- **M3 Per-Cluster Kaplan-Meier Decay**: Replaces fixed Ebbinghaus half-lives with non-parametric survival curves estimated per semantic cluster; activates when sufficient data is available
- **M4 HDBSCAN Clustering**: Groups memories into semantic neighborhoods (dendrogram → condensed tree → EOMBST extraction) for per-cluster learning in M2/M3
- **M5 Hot/Warm/Cold Tiering**: Three-tier memory with t-digest streaming quantile estimator for adaptive tier boundaries; hot memories get priority in search, cold memories are candidates for archival

## Environment Variables

GEMINI_API_KEY, SUPERMEMORY_CC_API_KEY, REIN_HTTP_TOKEN, REIN_DB, REIN_LOG
