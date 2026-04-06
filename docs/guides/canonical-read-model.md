# Canonical Read Model

rein treats canonical memories as the default read surface.

## Why

Raw memories are useful for provenance, but not for stable retrieval. Canonicals provide:

- one stable memory per durable fact
- aggregated evidence counters (`support_count`, `merge_count`, `source_diversity`)
- cleaner recall ranking and feedback signals

## Current behavior

- store-time gray-zone dedup prefers existing canonicals when evidence already exists
- read paths join `memory_canonical_state` so canonical stats are visible
- recall collapses raw hits to canonicals before final ranking
- GUI and REST views expand evidence only on demand

## Practical consequences

- list views use `summary_short`
- detail views use the full canonical summary/content
- evidence is a secondary expansion surface, not the default result object
