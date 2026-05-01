# Adaptive Learning Signals

This note summarizes which signals rein currently feeds into adaptive learning.

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

## ARS scalar policy

When `[ars.acceleration]` is explicitly in canary mode and
`ars_parameter_policy` is healthy, synthesis/concept cold-start thresholds,
useful-rate thresholds, LLM judge sample rates, and LLM judge weight decay can
move from static config toward calibrated feedback. v0.28.5 adds
`runtime_adoption_weight` to the policy row; every dynamic trust calculation is
multiplied by that weight so rollout slides from static priors toward learned
values. Missing policy, zero adoption weight, drift alerts, or insufficient
calibration keep the static values.

SignalHint/bootstrap priors also feed the production useful-rate formulas under
the same canary gate. The last effective scalar values are persisted in the
adaptive snapshot so each slow-channel pass applies bounded max-step smoothing
instead of jumping directly from static defaults.

## ARS release/eval gate

`rein ars-acceleration-gate`, `rein_ars_acceleration_gate`, and
`GET /api/ars-acceleration-gate` expose a pure report over the existing ARS
signals. The report answers whether the current explicit canary is allowed and
why default-on remains blocked. It is observational only: it does not refresh
`ars_parameter_policy`, set `shadow_only = false`, change defaults, or commit
adaptive offsets.

## Admission and promotion

- admission can shift by cluster strength and cluster novelty context
- STM→LTM promotion can derive its threshold from cluster survival curves

## Cross-validation

Cross-validation still uses cross-source agreement first, but local canonicals get a small evidence bonus from:

- `support_count`
- `source_diversity`
