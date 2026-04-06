# Evidence-Aware Recall

rein recall now has two evidence layers:

1. canonical-first retrieval and ranking
2. evidence-aware correction and expansion

## Ranking path

- retrieve candidates from FTS / vector / KG / episode channels
- collapse them to canonicals
- rank with support/diversity-aware reranker features
- for lower-confidence or weakly corroborated results, apply evidence second-stage rerank

## Output path

- recall text includes `evidence_count` and `evidence_preview`
- REST recall includes `evidence_count` and `evidence_preview`
- `/api/memories/:id` expands the full evidence list

## Why evidence rerank is conservative

Evidence should improve weak recalls, not destabilize strong ones.

Current policy:

- skip second-stage evidence rerank for strongly corroborated results
- use evidence `summary` and `content` rematch for lower-confidence results
- keep canonical as the main object, evidence as supporting context
