# CLAUDE.md

> See [AGENTS.md](AGENTS.md) for full directory structure, invariants, and pitfalls.

## Build & Test

```bash
cargo build
cargo test            # 666+ tests, all must pass
cargo install --path . # Install to ~/.cargo/bin/rein
```

## Architecture

rein is a multi-source cross-validated memory MCP server (28 tools, evidence-aware canonical-first memory flow). Key modules:

- `extract/llm.rs` — LLM extraction (Gemini 3.1 Flash Lite), fallback to rule-based
- `extract/hooks/` — 4 hooks: post (PostToolUse), compact (PreCompact), prompt (UserPromptSubmit compatibility no-op), stop (Stop); admission and memory surfaces are cluster/canonical-aware
- `search/recall.rs` — 3-level waterfall (Tantivy BM25 → HNSW → Gemini API) + evidence-aware rerank + RRF/CC fusion + canonical-first collapse
- `search/expand.rs` — Query expansion (Gemini Flash Lite / OMLX dual backend)
- `search/rerank_llm.rs` — LLM reranker (Gemini / OMLX) + strong signal bypass
- `search/classify.rs` — Autonomous retrieval routing (Temporal/ExactKeyword/Semantic/Exploratory)
- `store/sqlite.rs` — Per-request connection model, Tantivy singleton cache
- `store/knowledge.rs` — Knowledge units, evolution, linking, organizing
- `store/quality.rs` — Self-learning quality scoring, pruning, recall tracking
- `store/memoir.rs` — Knowledge graph CRUD, temporal revisions, episodes, BFS traversal
- `store/adaptive.rs` — Feedback event sourcing, AdaptiveState cache, per-consumer offsets
- `store/hdbscan.rs` — Pure Rust HDBSCAN clustering (dendrogram → condensed tree → EOMBST)
- `store/tiering.rs` — Three-tier memory (Hot/Warm/Cold) with streaming quantile estimator
- `search/alpha_optimizer.rs` — Counterfactual offline alpha optimization with KG/episode/support/diversity-aware scoring
- `search/survival.rs` — Kaplan-Meier non-parametric survival analysis for adaptive decay
- `extract/dedup.rs` — Hybrid lexical dedup: whitespace tokens + `jieba-rs` + CJK bigrams
- `ops/mod.rs` — adaptive_status now exposes cluster profiles used by the Adaptive GUI page

## Temporal Knowledge Graph (v0.4.0)

- `concept_revisions` table — revision history auto-snapshotted on refine
- `episodes` table — session nodes with concept_ids + memory_ids
- ConceptLink has `valid_from`/`valid_until` temporal windows
- BFS traversal skips expired links
- MCP tools: `rein_timeline`, `rein_concept_history`, `rein_recall` with `from`/`to` params

## Autonomous Retrieval Routing (v0.4.0, extended v0.8.0)

Query classifier routes to optimal search strategy (6 types):
- **Episodic** ("what happened in our meeting") → balanced (alpha=0.5), 1.5x limit
- **Temporal** ("when did X change?") → BM25 bias (alpha=0.7), auto-inject time bounds
- **Preference** ("what do I prefer") → slight vector bias (alpha=0.4), 2x limit
- **ExactKeyword** ("SqliteStore") → heavy BM25 (alpha=0.85)
- **Semantic** ("memory management strategies") → vector dominant (alpha=0.3)
- **Exploratory** ("what do I know about...") → balanced (alpha=0.5), 2x limit

MCP response includes `[route: type]` prefix for transparency.

## Search Pipeline (v0.8.0 → v0.9.0)

3-channel retrieval: FTS (Tantivy BM25) + Vector (HNSW) + KG (concept FTS + BFS land-and-expand).
**Query expansion** (v0.9.0): LLM rewrites query into 2-3 variants (Gemini Flash Lite or OMLX), multi-query search with score merging.
Post-fusion multi-feature reranking (canonical support/diversity included in learned features).
**LLM reranker** (v0.9.0): Optional LLM-based rescoring of top N candidates, with strong-signal bypass.
**M5 tier filtering** (v0.9.0): Cold-tier memories excluded from non-Exploratory queries.
**Parallel pipeline** (v0.9.0): Supermemory + expansion launched concurrently with original query search.
Extraction postprocess: algorithmic date/preference/knowledge-update detection + LLM prompt rules.
Automatic prompt injection is disabled. `hook_prompt` remains as a compatibility no-op while memory production flows through record-only proxy + async worker + layered memory surfaces.

## Adaptive Engine (v0.5.0)

Self-learning engine that replaces fixed parameters with data-driven adaptation:

- **M1 Event Sourcing**: `feedback_events` table captures all recall/store/click signals; AdaptiveState cache with per-consumer offsets enables replay
- **M2 Counterfactual Alpha Optimization**: Learns optimal `cc_alpha` from candidate logs — replays past searches with alternative weights to find the alpha that would have maximized relevance, now with KG/episode/support/diversity-aware candidate scoring
- **M3 Per-Cluster Kaplan-Meier Decay**: Replaces fixed Ebbinghaus half-lives with non-parametric survival curves estimated per semantic cluster; activates when sufficient data is available
- **M4 HDBSCAN Clustering**: Groups memories into semantic neighborhoods (dendrogram → condensed tree → EOMBST extraction) for per-cluster learning in M2/M3
- **M5 Hot/Warm/Cold Tiering**: Three-tier memory with t-digest streaming quantile estimator for adaptive tier boundaries; hot memories get priority in search, cold memories are candidates for archival
- **Per-cluster admission + promotion**: admission and STM→LTM promotion now use cluster-aware strength/survival signals when available

## Environment Variables

GEMINI_API_KEY, SUPERMEMORY_CC_API_KEY, REIN_HTTP_TOKEN, REIN_DB, REIN_LOG, REIN_CONFIG, REIN_SSE_BIND, REIN_SSE_PORT
