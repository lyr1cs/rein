# MCP Tools

Updated for Rein v0.28.6. Rein exposes 40 production MCP tools through the
operation inventory. Test-only tools and `/api/test_path_template/*` routes are
not part of this reference.

## Core Memory

| # | Tool | Purpose |
|---:|---|---|
| 1 | `rein_store` | Store a memory with topic, content, optional importance, and keywords. |
| 2 | `rein_recall` | Recall memories by query, topic, keyword, time range, optional expansion, and optional synthesis. |
| 3 | `rein_update` | Replace an existing memory's content and optionally change importance. |
| 4 | `rein_forget` | Permanently delete one memory and its related indexes/links. |
| 5 | `rein_recent` | Return recently created memories. |
| 6 | `rein_list_topics` | List unique memory topics. |
| 7 | `rein_canonicals` | List canonical memories with support and merge metadata. |
| 8 | `rein_evidence` | List evidence snapshots for a canonical memory. |
| 9 | `rein_stats` | Return store counts, layer counts, and tier counts. |
| 10 | `rein_health` | Return per-topic and system health, including index and queue lag. |

## Maintenance

| # | Tool | Purpose |
|---:|---|---|
| 11 | `rein_gc` | Apply decay and prune weak STM memories; supports dry-run. |
| 12 | `rein_dedup` | Scan and merge duplicate memories; supports dry-run and topic-variant merging. |
| 13 | `rein_dedup_concepts` | Merge duplicate concepts within memoirs. |
| 14 | `rein_organize` | Create related-memory links based on similarity. |
| 15 | `rein_consolidate` | Consolidate selected topics into summary memories; supports dry-run. |
| 16 | `rein_cleanup` | Run scoped consolidation, deduplication, and adaptive refresh. |
| 17 | `rein_resummerize` | Recompress flagged canonicals through the lossless-compression contract. |
| 18 | `rein_archive_summary_refresh` | Regenerate a cold-tier archival summary for one memory. |

## Knowledge Graph And Temporal

| # | Tool | Purpose |
|---:|---|---|
| 19 | `rein_timeline` | Return chronological memory, episode, and concept-revision events. |
| 20 | `rein_concept_history` | Show revision history for a concept in a memoir. |
| 21 | `rein_memoir_list` | List named knowledge graphs. |
| 22 | `rein_memoir_show` | Show a memoir, its concepts, and links. |
| 23 | `rein_memoir_create` | Create a named knowledge graph. |
| 24 | `rein_memoir_add_concept` | Add a concept with definition and optional labels. |
| 25 | `rein_memoir_refine` | Refine a concept definition and record a revision. |
| 26 | `rein_memoir_link` | Add a typed edge between concepts in one memoir. |
| 27 | `rein_memoir_search` | Search concepts within one memoir. |
| 28 | `rein_memoir_search_all` | Search concepts across all memoirs. |
| 29 | `rein_memoir_inspect` | Traverse a concept neighborhood by depth. |
| 30 | `rein_memoir_export` | Export a memoir as JSON, ASCII, or DOT. |
| 31 | `rein_concept_state` | Fetch a concept's current state and living summary metadata. |
| 32 | `rein_concept_summary_refresh` | Regenerate concept living summaries for one concept or a batch. |

## Adaptive, Session, ARS, And Judge

| # | Tool | Purpose |
|---:|---|---|
| 33 | `rein_feedback` | Record memory access feedback or synthesis interaction feedback. |
| 34 | `rein_feedback_concept_summary` | Record concept-summary interaction feedback. |
| 35 | `rein_adaptive_status` | Return adaptive alphas, reranker weights, clusters, tiers, and event counts. |
| 36 | `rein_ingest_session` | Ingest a transcript into memories, concepts, links, and an episode. |
| 37 | `rein_judge_synthesis` | Manually run the LLM judge for a prior synthesis output. |
| 38 | `rein_judge_concept_summary` | Manually run the LLM judge for a prior concept-summary output. |
| 39 | `rein_ars_acceleration_gate` | Inspect read-only ARS acceleration canary/default-on release-gate state. |
| 40 | `rein_trust_measurement` | Inspect release gate, eval gates, index consistency, background observability, and active-learning status. |
