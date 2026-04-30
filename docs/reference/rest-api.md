# REST API Reference

The REST API is served under `/api/*` when HTTP/SSE or the GUI server is
enabled. Network access is guarded by the HTTP server token policy; many
mutating routes also require `x-rein-action: 1`. Protected reads accept
`x-rein-token`, `Authorization: Bearer ...`, or the GUI session cookie when
`REIN_HTTP_TOKEN` is configured.

## Auth Classes

| Class | Requirement |
|---|---|
| Server auth | HTTP/SSE startup requires `REIN_HTTP_TOKEN` unless explicitly configured for unauthenticated loopback. |
| Public read | Route-specific inventory auth is public; still subject to server-level HTTP auth. |
| Protected read | Requires `REIN_HTTP_TOKEN` on sensitive reads when configured. |
| Mutation marker | Requires `x-rein-action: 1`; body size is capped by `REIN_REST_MAX_BODY_BYTES`. |

## Routes

| Method | Route family | Auth notes | Response notes |
|---|---|---|---|
| `GET` | `/api/memories?q=...` | Protected read | Recall results, count, request id, optional `synthesis` when requested. |
| `GET` | `/api/recall_stream?q=...` | Server auth | Paged recall results with `offset`, `limit`, `next_offset`, and `has_more`; no synthesis. |
| `GET` | `/api/memories/{id}` | Protected read | Flattened memory fields plus nested `memory`, `evidence`, and `evidence_total`. |
| `DELETE` | `/api/memories/{id}` | Mutation marker | `{ id, deleted }`. |
| `GET` | `/api/recent` | Public read | Recent memory list. |
| `GET` | `/api/topics` | Public read | Topic list. |
| `GET` | `/api/canonicals` | Public read | Canonical memories with support and merge metadata. |
| `GET` | `/api/evidence?canonical_id=...` | Public read | Evidence rows for one canonical memory. |
| `GET` | `/api/timeline` | Server auth | Chronological timeline events. |
| `GET` | `/api/episodes` | Server auth | Episode records. |
| `GET` | `/api/activity` | Server auth | Recent activity summary for the GUI. |
| `GET` | `/api/stats` | Public read | Store counts, layers, and tiers. |
| `GET` | `/api/health` | Public read | Topic health, index lag, queue lag, and status. |
| `GET` | `/api/doctor` | Public read | Diagnostic report; `fix=true` is rejected on GET. |
| `POST` | `/api/doctor` | Mutation marker | Diagnostic report; JSON body controls `network` and `fix`. |
| `GET` | `/api/version` | Protected read | `{ version }`. |
| `POST` | `/api/session` | Mutation marker | Sets GUI session cookie from `REIN_HTTP_TOKEN`. |
| `DELETE` | `/api/session` | Mutation marker | Clears GUI session cookie. |
| `POST` | `/api/feedback` | Mutation marker | Emits memory-access or synthesis-interaction feedback events. |
| `GET` | `/api/adaptive` | Public read | Adaptive alphas, reranker weights, clusters, tiers, event counts, and synthesis stats. |
| `GET` | `/api/dedup_decisions` | Public read | Recent dedup decisions, optionally filtered. |
| `GET` | `/api/intelligent_merge_metrics` | Server auth | Intelligent-merge metrics summary. |
| `GET` | `/api/judge/calibration` | Server auth | Runtime-vs-offline judge calibration state. |
| `POST` | `/api/gc` | Mutation marker | Decay/prune report. |
| `POST` | `/api/dedup` | Mutation marker | Duplicate scan and merge counts. |
| `POST` | `/api/dedup_concepts` | Mutation marker | Concept duplicate groups merged and rows removed. |
| `POST` | `/api/organize` | Mutation marker | Number of related-memory links created. |
| `POST` | `/api/consolidate` | Mutation marker | Topic consolidation report and per-group detail. |
| `POST` | `/api/cleanup` | Mutation marker | Cleanup consolidation and dedup report. |
| `POST` | `/api/resummerize` | Mutation marker | Canonical recompression counters and skip flags. |
| `POST` | `/api/cold_archive/refresh` | Mutation marker | One-memory archival summary refresh outcome. |
| `POST` | `/api/ingest_session` | Mutation marker | Session ingestion counts, artifact id, and episode id. |
| `GET` | `/api/memoirs` | Public read | Named knowledge graph list. |
| `GET` | `/api/memoirs/{name}` | Public read | Memoir metadata plus concept/link export data. |
| `GET` | `/api/memoirs/{name}/export?format=...` | Public read | Memoir export as JSON, ASCII, or DOT text. |
| `GET` | `/api/memoirs/{name}/inspect/{concept}` | Server auth | Legacy concept-neighborhood inspection route. |
| `GET` | `/api/concepts/{concept_id}/state` | Public read | Concept state plus living-summary metadata and gate state. |
| `POST` | `/api/concepts/summary_refresh` | Mutation marker | Concept living-summary refresh report. |
| `POST` | `/api/feedback/concept_summary` | Mutation marker | Concept-summary feedback event count. |
| `GET` | `/api/artifacts` | Protected read | Artifact list, optionally paged. |
| `GET` | `/api/artifacts/{id}` | Protected read | Artifact detail; `include_transcript=true` can return raw transcript data. |

Test-support-only route families are intentionally excluded.
