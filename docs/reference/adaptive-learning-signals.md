# Adaptive Learning Signals

Updated for Rein v0.28.8 (2026-05-04). This note summarizes which signals rein
currently feeds into adaptive learning.

## Reranker learning

The reranker now learns from:

- `fts_score`
- `vec_score`
- `kg_score`
- `episode_score`
- `recency_days`
- `access_count`
- `strength`
- `importance_weight`
- `keyword_overlap`
- `topic_match`
- `brevity`
- `channel_coverage`
- `canonical_support`
- `source_diversity`
- `usage_recency`
- `connectivity`
- `concept_richness`
- `tier_score`
- `is_current`

## Alpha optimization

Counterfactual alpha learning still optimizes a single CC alpha, but candidate scoring now sees:

- `bm25_norm`
- `vec_norm`
- `kg_norm`
- `episode_norm`
- `support_count`
- `source_diversity`

High-evidence accessed events also receive higher event weight.

Shadow ARS acceleration also replays six-dimensional fusion weights over
BM25/vector/KG/episode/support/diversity signals. v0.28.4 evaluates one-hot
dimensions, deterministic pairwise simplex blends, accessed centroids,
accessed-vs-other feature gaps, and bounded GP+EI-style proposals, then
averages tied winners before applying the normal parent-prior shrinkage.

### Cluster-bucket alignment (v0.28.8 M-8)

Both `compute_counterfactual_alphas` (alpha learning) and
`compute_shadow_fusion_weight_replay` (six-dimensional fusion) bucket events
through `top_vec_hit_cluster`, which prefers the recall-time top-vec memory
id (`query_top_vec_memory_id_at_recall`, persisted in the
`recall_complete` event payload) and looks it up against the CURRENT
`memory_clusters` map at learn-time. This is correct regardless of how
many M4 reclusters fired between recall and learn-time — closing the
M4-then-M2 normal pipeline-order bug where the legacy
`cluster_version_at_recall` version-match guard invalidated every event in
the common path.

### `learned_shadow_fusion` cap eviction (v0.28.8 L6)

LRU eviction at `LEARNED_SHADOW_FUSION_CAP = 4096` is restricted to
**cluster-scoped** keys (`{query_type}:{cluster_id}` shape, where the
suffix parses as `u32`). Fallback keys (`global` and per-query-type) are
never evicted, preserving the fallback chain in
`get_shadow_fusion_weights` for queries without surviving cluster-scoped
buckets.

## ARS scalar policy

When `[ars.acceleration]` is enabled and `ars_parameter_policy` is healthy in
canary mode, synthesis/concept cold-start thresholds, useful-rate thresholds,
LLM judge sample rates, and LLM judge weight decay can move from static config
toward calibrated feedback. v0.28.6 keeps the global
`runtime_adoption_weight` and adds scoped `adoption_weights` for recall
fusion/query/cluster and scalar surfaces. Missing policy, zero scoped adoption
weight, drift alerts, or insufficient calibration keep the static values.

SignalHint/bootstrap priors also feed the production useful-rate formulas under
the same canary gate. The last effective scalar values are persisted in the
adaptive snapshot so each slow-channel pass applies bounded max-step smoothing
instead of jumping directly from static defaults.

### Per-surface judge sample rates (v0.28.7 M-1 input-side + v0.28.8 M-1 persistence-side)

Judge calibration state is split per `JudgeSurface` (Synthesis vs
ConceptSummary). v0.28.7 threaded the surface tag through input-side helpers
and handlers; v0.28.8 added per-surface persistence keys to the snapshot
blob:

- `judge_sample_rate_cold_start_synthesis`
- `judge_sample_rate_cold_start_concept_summary`
- `judge_sample_rate_warm_synthesis`
- `judge_sample_rate_warm_concept_summary`

`compute_and_persist_judge_sample_rate(surface, …)` writes the per-surface
key once per surface; the legacy cluster-shared keys
(`judge_sample_rate_cold_start`, `judge_sample_rate_warm`) keep being
written with the synthesis-surface variant for downgrade-rollback compat.
Reader sites consult per-surface first via
`ars_effective_scalar_with_legacy_fallback`, falling back to the legacy
key only when the per-surface key is absent (one-time
first-tick-after-upgrade source).

Effect: synthesis-surface drift (`judge_drift_alert_synthesis > 0`) zeros
the synthesis scalars under the fail-closed early-return path WITHOUT
cross-contaminating the concept-summary scalars, and vice versa.

### Drift-triggered Canary→Shadow rollback (v0.28.7 H2)

When `judge_calibration_state.judge_drift_alert*` is positive while
`ars_parameter_policy` is in Canary mode, `apply_local_fixes` (called by
`rein doctor --fix` and the adaptive slow-channel) refreshes the policy.
The next `refresh_ars_parameter_policy` tick demotes the policy back to
Shadow with `runtime_adoption_weight = 0`, so drift cannot be merely
logged — runtime trust gates the static priors back in within one
adaptive cycle.

### Shadow vs production buckets (v0.28.7 H3)

Shadow buckets derived from `route_context` live in a separate
`CONCEPT_SUMMARY_BY_CLUSTER_SHADOW_CAP = 4096` LRU. The production
4096-bucket cap is sealed against shadow flooding; recall via the shadow
path cannot evict production cache entries.

## ARS release/eval gate

`rein ars-acceleration-gate`, `rein_ars_acceleration_gate`, and
`GET /api/ars-acceleration-gate` expose a pure report over the existing ARS
signals. The report answers whether the current explicit canary is allowed and
why default-on remains blocked. It is observational only: it does not refresh
`ars_parameter_policy`, set `shadow_only = false`, change defaults, or commit
adaptive offsets.

`rein trust-measurement`, `rein_trust_measurement`, and
`GET /api/trust-measurement` add the broader measurement layer: release gate,
recall/dedup/admission/latency eval gates, index consistency counts,
background queue/grayzone observability, and active-learning status.

## Admission and promotion

- admission can shift by cluster strength and cluster novelty context
- STM→LTM promotion can derive its threshold from cluster survival curves

## Cross-validation

Cross-validation still uses cross-source agreement first, but local canonicals get a small evidence bonus from:

- `support_count`
- `source_diversity`
