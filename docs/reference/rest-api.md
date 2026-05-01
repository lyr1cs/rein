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
| `GET` | `/api/health` | Public read | Topic health, index lag, queue lag, and status; index snapshots include `dirty`, `rebuilding`, `stale_rebuild_marker`, and `index_exists`. |
| `GET` | `/api/doctor` | Public read | Diagnostic report; `fix=true` is rejected on GET. |
| `POST` | `/api/doctor` | Mutation marker | Diagnostic report; JSON body controls `network` and `fix`. |
| `GET` | `/api/version` | Protected read | `{ version }`. |
| `POST` | `/api/session` | Mutation marker | Sets GUI session cookie from `REIN_HTTP_TOKEN`. |
| `DELETE` | `/api/session` | Mutation marker | Clears GUI session cookie. |
| `POST` | `/api/feedback` | Mutation marker | Emits memory-access or synthesis-interaction feedback events. |
| `GET` | `/api/adaptive` | Public read | Adaptive alphas, reranker weights, clusters, tiers, event counts, and synthesis stats. |
| `GET` | `/api/ars-acceleration-gate` | Public read | Read-only ARS acceleration canary/default-on gate report. |
| `GET` | `/api/trust-measurement` | Public read | Unified release gate, eval gate, index consistency, background observability, and active-learning report. |
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

## `/api/adaptive` ARS Acceleration Shape

v0.28 exposes ARS acceleration status, replay-learned fusion weights, and the
dynamic-parameter activation policy. v0.28.6 makes acceleration default-on, but
runtime adoption still fails closed unless `ars_parameter_policy` is loaded,
in canary mode, and has positive scoped adoption weights. The same policy gate
covers recall fusion, synthesis/concept scalar parameters, judge sample-rate
adaptation, LLM feedback decay, SignalHint-derived useful-rate priors, and
release-gate reporting.

`GET /api/adaptive` includes:

```json
{
  "ars_acceleration": {
    "enabled": true,
    "shadow_only": false,
    "parameter_policy": {
      "policy": {
        "schema_version": 1,
        "revision": 0,
        "mode": "disabled",
        "disabled_reason": "missing policy row",
        "source_adaptive_version": 0,
        "runtime_adoption_weight": 0.0,
        "adoption_weights": {},
        "last_event_id": 0,
        "last_updated": ""
      },
      "status": "missing"
    },
    "shadow_fusion_replay": {
      "enabled": true,
      "shadow_only": false,
      "status": "insufficient_samples",
      "replay_limit": 500,
      "eligible_samples": 0,
      "min_samples": 10,
      "global": null,
      "by_query_type": [],
      "by_cluster": []
    }
  }
}
```

`shadow_fusion_replay.status` is bounded to the current v0.28 states:
`disabled`, `insufficient_samples`, `ready`, or `no_learnable_signal`.
The default install reports `insufficient_samples` with
`eligible_samples: 0`, `global: null`, and empty `by_query_type` /
`by_cluster` arrays.

When `[ars.acceleration].enabled = true`, `ready` responses include learned
replay weights:

- `global`: `null` or `{ sample_count, last_updated, weights }`
- `by_query_type`: array of `{ query_type, sample_count, last_updated, weights }`
- `by_cluster`: array of `{ query_type, cluster_id, sample_count, last_updated, weights }`

`weights` is a normalized object with `bm25`, `vec`, `kg`, `episode`,
`support`, and `diversity` numbers. Runtime recall canary mode still requires
`parameter_policy.status = "loaded"`, `policy.mode = "canary"`, and a
compatible `source_adaptive_version` with positive global or scoped adoption
weight before consuming eligible persisted weights from
`AdaptiveState.learned_shadow_fusion`.

## `/api/ars-acceleration-gate`

`GET /api/ars-acceleration-gate` is a read-only release/eval report for ARS
acceleration rollout decisions. It evaluates the same existing signals that are
already visible through config, `/api/adaptive`, and `rein doctor`: adaptive
enablement, `[ars.acceleration]`, `ars_parameter_policy`, learned shadow-fusion
evidence, replay status, adaptive version, and judge drift status.

The report does not refresh the policy row, flip defaults, or enable canary
mode. `canary.allowed` means the current explicit canary configuration is
healthy enough for runtime adoption. `default_on.allowed` is intentionally
`false` in this skeleton and includes
`default_on_requires_release_evaluation` until a separate release evaluation
defines ship criteria.

```json
{
  "schema_version": 1,
  "purpose": "read_only_release_eval_gate_for_ars_acceleration",
  "signals": {
    "adaptive_enabled": true,
    "ars_acceleration_enabled": true,
    "ars_acceleration_shadow_only": false,
    "policy_status": "missing",
    "policy_mode": "disabled",
    "policy_allows_runtime": false,
    "runtime_adoption_weight": 0.0,
    "runtime_adoption_weights": {},
    "shadow_fusion_status": "insufficient_samples",
    "judge_drift_alert": false
  },
  "canary": {
    "allowed": false,
    "blockers": ["ars_parameter_policy_missing"],
    "warnings": ["shadow_fusion_replay_not_ready:insufficient_samples"]
  },
  "default_on": {
    "allowed": false,
    "blockers": ["default_on_requires_release_evaluation", "canary_not_allowed"],
    "warnings": ["default_on_gate_is_report_only_and_does_not_change_runtime_defaults"]
  }
}
```

## `/api/trust-measurement`

`GET /api/trust-measurement` returns the unified Trust & Measurement snapshot
also exposed as `rein trust-measurement` and `rein_trust_measurement`. It
combines the ARS release gate, recall/dedup/admission/latency eval gate
inventory, SQLite/vector index consistency counts, background queue/grayzone
observability, and LLM active-learning status.
