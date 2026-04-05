# Dedup Architecture

This document describes the current dedup architecture after the canonical/evidence and cleanup improvements.

## Goals

- Unify dedup around a single candidate -> score -> resolve mental model
- Preserve original observations as evidence instead of only mutating the winning memory
- Keep hot-path latency bounded while still allowing richer async cleanup and LLM resolution
- Emit durable dedup decisions that can later be audited, reverted, and learned from

## Three Lanes

### Hot Path

Hot-path dedup runs during `store_with_dedup` and stays cheap:

- lexical candidate generation via FTS
- topic-variant expansion via normalized topic keys
- adaptive threshold override from `AdaptiveState`
- immediate resolution for strong matches
- gray-zone pairs are queued for async dedup resolution instead of blocking the hot path

The hot path is optimized for correctness first and latency second. It should avoid remote calls entirely unless an operator explicitly chooses a synchronous batch path.

### Warm Path

Warm-path dedup is async/batch-oriented:

- `rein cleanup`
- queued `rein cleanup --async`
- queued hot-path `dedup-queue` jobs
- `rein worker cleanup-queue`
- `rein worker dedup-queue`
- vector dedup sweeps
- LLM consolidation/summarization fan-out

This path can afford parallel planning and richer LLM reasoning because writes are committed after planning is complete.

### Cold Path

Cold-path dedup is adaptive maintenance:

- per-cluster dedup thresholds
- HDBSCAN reclustering
- survival/tiering refresh
- large-scale cleanup and topic consolidation

This path should run per batch, not per memory.

## Canonical Model

Canonical state is tracked separately from the `memories` row itself:

- `memory_canonical_state`
  - maps every memory row to its current canonical memory
  - stores support_count, merge_count, dedup_confidence, source_diversity, contradiction_score
- `memory_evidence`
  - immutable snapshots of absorbed/superseded observations
- `dedup_decisions`
  - append-only dedup ledger containing winner/loser, scores, relation, confidence, reason, operator, reversibility, and payload

This allows the winning memory to stay searchable while evidence and audit history remain queryable.

## Resolution Semantics

Current relations:

- `duplicate`
- `update`
- `related`
- `distinct`

Recommended semantics:

- `duplicate`: absorb loser into canonical winner
- `update`: canonical winner changes because the new observation supersedes prior state
- `related`: link or keep nearby, but do not merge
- `distinct`: create a new memory

## Parameter Optimization

Dedup should update memory parameters explicitly:

- `strength`: reinforce the kept memory after an absorb/merge
- `access_count`: aggregate supporting evidence
- `importance`: keep the strongest importance
- `decay_lambda`: preserve the slower decay when a stronger memory wins
- `tier`: keep the stronger tier when merging
- `last_accessed`: move forward on kept memories

Canonical-state parameters should be derived from evidence:

- `support_count`
- `merge_count`
- `source_diversity`
- `dedup_confidence`
- `contradiction_score`

## Async and Parallel Strategy

Parallelize:

- FTS candidate lookup
- ANN lookup
- topic-group planning
- LLM consolidation / verdict generation

Serialize:

- SQLite writes
- canonical-state refresh per winner
- final adaptive refresh per batch

The rule of thumb is: reads and inference may fan out; commits must stay deterministic.

## CLI Surface

- `rein consolidate ...`
- `rein dedup ...`
- `rein cleanup ...`
- `rein cleanup --async`

`rein cleanup` is the operator-friendly batch entrypoint for terminal usage.

## Next Steps

The next architectural steps after this baseline are:

1. cross-topic candidate generation using cluster blocking and ANN-first retrieval
2. richer structured LLM merge outputs for every warm-path resolution
3. MCP cleanup endpoints mirroring the CLI cleanup flow
4. canonical-first read models that use evidence as the primary compression source
