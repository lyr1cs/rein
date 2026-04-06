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

## Admission and promotion

- admission can shift by cluster strength and cluster novelty context
- STM→LTM promotion can derive its threshold from cluster survival curves

## Cross-validation

Cross-validation still uses cross-source agreement first, but local canonicals get a small evidence bonus from:

- `support_count`
- `source_diversity`
