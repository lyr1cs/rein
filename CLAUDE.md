# CLAUDE.md

> See [AGENTS.md](AGENTS.md) for full directory structure, invariants, and pitfalls.

## Build & Test

This is a 2-crate Cargo workspace (since v0.21 A1):
- `crates/rein/` — main crate (binary + lib)
- `crates/rein-macros/` — proc-macro crate (`#[op]`, `#[derive(OpsRender)]`)

```bash
cargo build -p rein                  # main crate
cargo build -p rein --features gui   # with embedded Neural Wiki GUI
cargo test --workspace               # all crates
cargo clippy --workspace --all-targets -- -D warnings
cargo install --path crates/rein     # install to ~/.cargo/bin/rein
```

## Architecture

rein is a multi-source cross-validated memory MCP server (**29 MCP tools** as of v0.23.0-rc1, evidence-aware canonical-first memory flow). Key modules:

- `extract/llm.rs` — LLM extraction (Gemini 3.1 Flash Lite), fallback to rule-based; `MockExtractor` under `test-support` feature
- `extract/hooks/` — 4 hooks: post (PostToolUse), compact (PreCompact), prompt (UserPromptSubmit compatibility no-op), stop (Stop); admission and memory surfaces are cluster/canonical-aware
- `search/recall.rs` — 3-level waterfall (Tantivy BM25 → HNSW → Gemini API) + evidence-aware rerank + RRF/CC fusion + canonical-first collapse
- `search/expand.rs` — Query expansion (Gemini Flash Lite / OMLX dual backend)
- `search/rerank_llm.rs` — LLM reranker (Gemini / OMLX) + strong signal bypass
- `search/classify.rs` — Autonomous retrieval routing (Temporal/ExactKeyword/Semantic/Exploratory)
- `store/sqlite.rs` — Per-request connection model, Tantivy singleton cache, `MERGE_CONTENT_CAP` pub const shared with resummerize path, `refresh_indexes_after_canonical_rewrite` for the v0.23 direct-SQL rewrite path
- `store/knowledge.rs` — Knowledge units, evolution, linking, organizing
- `store/quality.rs` — Self-learning quality scoring, pruning, recall tracking
- `store/memoir.rs` — Knowledge graph CRUD, temporal revisions, episodes, BFS traversal
- `store/adaptive.rs` — Feedback event sourcing, AdaptiveState cache, per-consumer offsets; v0.23 adds `CanonicalLengthStats` + `recompute_canonical_length_stats` for adaptive resummerize target_bytes
- `store/hdbscan.rs` — Pure Rust HDBSCAN clustering (dendrogram → condensed tree → EOMBST)
- `store/tiering.rs` — Three-tier memory (Hot/Warm/Cold) with streaming quantile estimator
- `store/hnsw.rs` — usearch HNSW wrapper; v0.23 fixed dirty-marker lifecycle (only full rebuilds clear `.dirty`)
- `store/resummerize_audit.rs` — v0.23 audit table row models + helpers; `status IN ('active', 'updated')` filters match `ops/resummerize.rs` eligibility
- `search/alpha_optimizer.rs` — Counterfactual offline alpha optimization with KG/episode/support/diversity-aware scoring
- `search/survival.rs` — Kaplan-Meier non-parametric survival analysis for adaptive decay
- `extract/dedup.rs` — Hybrid lexical dedup: whitespace tokens + `jieba-rs` + CJK bigrams
- `ops/mod.rs` — adaptive_status exposes cluster profiles used by the Adaptive GUI page
- `ops/dedup.rs` — `run_vec_dedup` slow channel; v0.23 preserves `needs_vec_dedup` on every write-failure path + re-inserts into HNSW via `update_hnsw_for_vec_dedup`
- `ops/resummerize.rs` — **v0.23** slow-channel resummerize op: atomic `apply_resummerize` with 5-way CAS (claim token + snapshot updated_at + live status); `BEGIN DEFERRED` snapshot block for read-consistent (canonical, updated_at_raw, evidence); 3-strike exhaustion fuse; 5-minute stale-claim timeout
- `compression/contract.rs` — **v0.23** Lossless Compression Contract: 7 invariants as pure `fn(&ContractInput, &str) -> Result<(), Violation>`
- `eval/mcnemar.rs` / `eval/scorecard.rs` — **v0.23** paired McNemar + ship/bail-out decision used by `rein-eval`
- `bin/rein_eval.rs` — **v0.23** standalone eval binary; `baseline` + `compare` fully implemented, `run` errors cleanly unless a live LLM provider is configured
- `embed/mock.rs` — `MockEmbedder` under `test-support` feature (scripted response queue for `run_vec_dedup_with_embedder` integration tests)

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

## Resummerize (v0.23.0-rc1)

LLM-driven canonical recompression replacing v0.21's keep-tail truncation at the 10KB `MergeInto` cap, gated by the 7-invariant Lossless Compression Contract.

- **Feature flag**: `[resummerize].enabled = false` by default; operator must opt in
- **Trigger**: `MergeInto` cap hit → `needs_resummerize = 1` flag set alongside keep-tail stopgap
- **Slow-channel op**: `ops/resummerize.rs::run_resummerize` picks flagged rows, atomically claims via `in_progress_resummerize_at` timestamp, reads a consistent snapshot of `(canonical, raw updated_at, evidence)` under `BEGIN DEFERRED`, calls the configured LLM, validates against the Lossless Contract, and applies only if the contract passes AND a 5-way CAS still matches (claim token + snapshot updated_at + live status). On any failure keep-tail remains the effective state
- **Side-index discipline**: Tantivy refresh + sqlite-vec delete (in-transaction) + `needs_vec_dedup = 1` (triggers adaptive pipeline's re-embed path which also re-inserts into HNSW)
- **Audit**: `resummarize_runs` table with per-run row (status + violations JSON + output_hash + finished_at)
- **Safety fuses**: 3-strike exhaustion clears flag after 3 consecutive failures; 5-minute stale-claim timeout for worker-crash recovery
- **Eval harness**: `cargo run --bin rein-eval -- resummerize {baseline,run,compare}` — McNemar paired non-inferiority test; `baseline` + `compare` work without LLM, `run` errors cleanly when no provider is configured

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
