# Operations

This chapter is the day-to-day operator guide for a Rein store. Use
`rein --help` for exact flag spelling on the installed version.

## Memory Workflows

Store a memory:

```bash
rein store --topic debugging --content "Connection pool exhaustion was fixed by lowering worker fanout." --importance high --keywords database,pool
```

Recall memories:

```bash
rein recall "connection pool" --topic debugging --limit 10
```

Useful recall filters include `--keyword`, `--from`, `--to`, `--expand`, and
`--synthesize`. Synthesis is opt-in per request and also requires
`[ars].recall_synthesis_enabled = true` plus a configured LLM provider.

Inspect recent and canonical memories:

```bash
rein recent --limit 20
rein canonicals --limit 20
rein evidence <canonical_id> --limit 20
```

Update a memory by ID:

```bash
rein update <memory_id> --content "Updated content" --importance critical
```

Forget a memory by ID:

```bash
rein forget <memory_id>
```

`forget` is destructive. It removes the memory row and related index or graph
references in one operation.

## Cleanup And Maintenance

Preview broad cleanup before changing the store:

```bash
rein cleanup --all --dry-run
```

Run cleanup on an explicit scope:

```bash
rein cleanup debugging
rein cleanup --topics debugging,operations
rein cleanup --pattern "debug*"
rein cleanup --all
```

Destructive cleanup requires an explicit scope such as `--all`, a topic, a topic
list, or a pattern. Cleanup resolves topic groups, consolidates fragmented topic
variants, deduplicates memories, and refreshes adaptive state.

Consolidate topics without the full cleanup pipeline:

```bash
rein consolidate debugging --dry-run
rein consolidate debugging
rein consolidate --pattern "debug*" --merge-variants
rein consolidate --all --merge-variants
```

Run dedup directly:

```bash
rein dedup --dry-run
rein dedup
rein dedup --merge-variants
```

Inspect dedup decisions:

```bash
rein dedup-log --limit 50
rein dedup-log --canonical <canonical_id>
rein dedup-log --operator llm_verdict
```

Run garbage collection:

```bash
rein gc --dry-run
rein gc
rein gc --threshold 0.05
```

`gc` applies decay and prunes weak STM memories below the configured or supplied
threshold.

Organize related-memory links:

```bash
rein organize --max-links 5
```

Maintain knowledge graph concept names:

```bash
rein dedup-concepts
```

Refresh optional slow-channel summaries only when the matching feature is
enabled:

```bash
rein resummerize --dry-run
rein resummerize --canonical-id <canonical_id>
rein archive-summary-refresh <memory_id>
```

Capability A concept summaries, Capability B recall synthesis, Capability C
cold archival summaries, intelligent merge, and runtime LLM judge features are
configuration-gated and remain inactive unless enabled.

Concept living-summary refresh is exposed through the `rein_concept_summary_refresh`
MCP tool and `POST /api/concepts/summary_refresh` REST route rather than a CLI
subcommand.

## Workers And Queues

Rein has file-backed queues for work that should not block the hot path.

Drain queued memory extraction jobs:

```bash
rein worker memory
```

Drain store-time gray-zone dedup jobs:

```bash
rein worker dedup-queue
```

Run cleanup inline through the worker entrypoint:

```bash
rein worker cleanup --all --dry-run
rein worker cleanup debugging
rein worker cleanup --pattern "debug*"
```

Drain queued cleanup and merge-refinement work:

```bash
rein worker cleanup-queue
rein worker merge-refinement-queue
```

Use `rein doctor` to inspect queue health, dead letters, provider readiness,
database state, and side-index status.

## Agent Hooks

`rein init` configures supported MCP clients. For Codex CLI it also enables the
Codex hook engine and installs `~/.codex/hooks.json` entries for:

- `SessionStart` -> `REIN_AGENT_LABEL=codex rein hook session-start`
- `PreToolUse` -> `REIN_AGENT_LABEL=codex rein hook pre`
- `PermissionRequest` -> `REIN_AGENT_LABEL=codex rein hook permission`
- `PostToolUse` -> `REIN_AGENT_LABEL=codex rein hook post`
- `UserPromptSubmit` -> `REIN_AGENT_LABEL=codex rein hook prompt`
- `Stop` -> `REIN_AGENT_LABEL=codex rein hook stop`

Codex hook payloads are not identical to Claude Code payloads. Rein maps Codex
fields such as `hook_event_name`, `tool_input`, `tool_response`, `prompt`,
`last_assistant_message`, and `transcript_path`. `PostToolUse` and `Stop` feed
the same extraction queue used by Claude Code hooks. `PreToolUse` and
`PermissionRequest` are deny-only guardrail hooks for obviously destructive
shell commands.

Codex context injection is explicit opt-in:

```toml
[hooks.codex]
inject_prompt_context = true
inject_session_context = true
max_additional_context_chars = 1200
```

When enabled, `SessionStart` emits bounded project context and
`UserPromptSubmit` emits bounded relevant memory context using Codex's official
`hookSpecificOutput.additionalContext` JSON shape. Hook diagnostics are written
to stderr; stdout is kept empty unless Rein emits an official Codex hook JSON
object.

## GUI Service Management

The GUI requires a binary installed with `--features gui` and an HTTP token
unless unauthenticated loopback was explicitly configured.

Start and stop the background GUI service:

```bash
export REIN_HTTP_TOKEN="change-this-token"
rein gui on
rein dashboard
rein gui off
```

Run the GUI in the foreground for logs:

```bash
export REIN_HTTP_TOKEN="change-this-token"
rein serve --gui
```

The GUI serves static assets at the HTTP root and uses token-protected API
routes under `/api/*`. If the browser shows a token prompt, use the same token
as `REIN_HTTP_TOKEN`.

## Proxy Service Management

Start and stop the record-only proxy:

```bash
export REIN_PROXY_TOKEN="change-this-token"
rein proxy on
rein dashboard
rein proxy off
```

Run it in the foreground while validating client routing:

```bash
export REIN_PROXY_TOKEN="change-this-token"
rein serve --proxy
```

Set client base URLs to the local proxy and send `x-rein-token` from
`REIN_PROXY_TOKEN`. For OpenAI-compatible clients, the default proxy port is
`8690`; for Anthropic-compatible clients, set the provider base URL to the same
origin. Codex API-key proxy examples use a custom provider with
`env_http_headers = { "x-rein-token" = "REIN_PROXY_TOKEN" }`.

The proxy is record-only. It forwards upstream requests and responses, mirrors
eligible artifacts, and queues extraction work when enabled; it does not modify
the prompt or inject recalled memories.

## Diagnostics

Run a normal diagnostic:

```bash
rein doctor
```

Include network/provider probes:

```bash
rein doctor --network
```

Apply supported repairs:

```bash
rein doctor --fix
```

`rein doctor --fix` covers (non-exhaustive list):

- **Side-index rebuilds** — Tantivy and HNSW are rebuildable accelerators;
  doctor refreshes them when they drift from SQLite.
- **`ars_parameter_policy` corruption recovery (v0.28.8)** — when
  `load_parameter_policy` returns `Corrupt`, doctor calls
  `repair_corrupt_parameter_policy`, which re-checks the row's status under
  `BEGIN IMMEDIATE` and DELETEs only if it is still `Corrupt`. This closes
  a TOCTOU window where a peer `refresh_ars_parameter_policy` tick could
  rewrite the row to a healthy canary in the gap between the doctor's
  earlier read and the recovery call. `UnsupportedSchema` (future-schema)
  rows and `StorageError` (transient busy/lock) reads are LEFT IN PLACE
  rather than deleted.
- **Drift-triggered Canary→Shadow rollback (v0.28.7 H2)** — if
  `judge_calibration_state.judge_drift_alert*` is positive while the policy
  is in Canary mode, doctor refreshes `ars_parameter_policy` so the next
  `refresh_ars_parameter_policy` tick demotes the policy back to Shadow
  with `runtime_adoption_weight = 0`.

`rein doctor` (read-only) also surfaces:

- **`judge_call_ledger` saturation (v0.28.7 M-9)** — warns when the rolling
  24h LLM-judge call ledger is approaching `daily_call_cap`. The ledger
  is shared between the runtime judge and the nightly cron, so saturation
  on one path affects the other.
- **`learned_shadow_fusion` cap pressure** — warns when the LRU is at or
  near `LEARNED_SHADOW_FUSION_CAP = 4096`; eviction is restricted to
  cluster-scoped buckets (`{query_type}:{cluster_id}` shape) so the
  `global` and per-query-type fallback chain is preserved.
- **`policy.adoption_weights` cap pressure** — warns when the map exceeds
  `ADOPTION_WEIGHTS_CAP = 4128`; this is warn-only with no eviction (every
  `adoption_weights` key maps to a runtime trust gate scope identifier;
  silently dropping one would mute a canary scope).

Get machine-readable output:

```bash
rein doctor --json
```

Inspect the local configuration snapshot without secrets:

```bash
rein config
```

Check service status, ports, proxy metrics, and memory statistics:

```bash
rein dashboard
```

Increase logs while debugging:

```bash
REIN_LOG=debug rein serve --sse
REIN_LOG=debug rein serve --proxy
```

Common repair flow:

1. Run `rein doctor`.
2. If the issue is provider-specific, rerun with `--network`.
3. If the issue is an index or queue repair supported by doctor, run
   `rein doctor --fix`.
4. Recheck `rein dashboard`.
5. Drain relevant workers if queues remain pending.
