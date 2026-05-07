# Changelog

All notable changes to **rein** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The full release notes (with audit-round breakdowns and operator-visible
schema changes) live on the [GitHub Releases page](https://github.com/lyr1cs/rein/releases).
This file is a condensed index intended for quick scanning.

## [Unreleased]

_Nothing yet._

## [0.28.12] — 2026-05-07

Single-line hotfix on top of v0.28.11. Restores Claude Code MCP client
compatibility for the `rein_feedback` tool.

### Fixed

- **`rein_feedback` MCP `inputSchema` validation (hotfix)** — the manual
  `JsonSchema` impl on `FeedbackParams` (untagged enum, custom
  `Deserialize` for optional `kind` back-compat) emitted a schema with
  `oneOf` but no top-level `"type": "object"`. JSON Schema accepts this,
  but Claude Code's MCP client uses Zod strict-validation that requires
  `type` on every tool input schema, so `tools/list` rejected the entire
  rein tool list with `path: ["tools", 12, "inputSchema", "type"]`. Added
  `"type": "object"` to the hand-written schema. Runtime deserialization
  path unchanged. Codex review clean.

## [0.28.8] — 2026-05-04

Second-pass audit hardening on v0.28.7. **17 codex review rounds (R1–R17)**
saturated at 2-consecutive-clean. **15 P2 + 1 P3** findings closed; **0 P1**
throughout. **1462 tests / 0 fail / 3 ignored / 0 clippy / 0 fmt.**
Default-OFF behavior bit-identical to v0.28.7.

### Fixed

- **M-8 cluster-bucket alignment (R13, structural)** — learn-time
  `top_vec_hit_cluster` now prefers memory-id remap against current
  `memory_clusters`. Closes the M4-then-M2 normal pipeline-order bug where
  `cluster_version_at_recall` was invalidated for every event in the common
  path.
- **L6 fallback bucket preservation (R12)** — `learned_shadow_fusion` LRU
  eviction restricted to cluster-scoped buckets via the new
  `is_cluster_scoped_bucket` predicate; the `global` and per-query-type
  fallback chain stays intact.
- **`ars_parameter_policy` schema robustness** — schema_version peek before
  typed deserialize (R8); CAS predicate uses schema-aware COALESCE default
  (R8); `>` rather than `!=` for future-schema preservation (R15);
  `repair_corrupt_parameter_policy` wraps load+delete in `BEGIN IMMEDIATE`
  (R10).
- **R10 P2 SQL-fallback cluster id atomicity** — split
  `query_cluster_id_from_snapshot` (event payload) from `query_cluster_id`
  (read-time alpha selection).
- **M-5 / M-6 rollback / outer-blend** — anchors `static_threshold` on
  config default when `runtime_adoption_weight ≈ 0`; outer-blends ARS
  simplex against `legacy_score` by adoption weight.
- **L1 / L4 / L5 / L7** — `sanitize_bootstrap_priors` cap; auth-policy
  regression locks for `/api/trust-measurement` + `/api/ars-acceleration-gate`;
  doctor recovery covers Corrupt policy rows; release-gate test coverage.

### Added

- New `RecallEvent.query_top_vec_memory_id_at_recall: Option<String>`
  field for memory-id-remap bucket resolution.
- New 4 per-surface `ars_effective_scalars` keys
  (`judge_sample_rate_{cold_start,warm}_{synthesis,concept_summary}`)
  with `ars_effective_scalar_with_legacy_fallback` helper.
- New `repair_corrupt_parameter_policy(conn) -> RepairCorruptOutcome`
  public helper.
- New `is_cluster_scoped_bucket(key)` predicate.

### Schema

- Snapshot `ars_effective_scalars` blob gains 4 new per-surface keys.
- `learned_shadow_fusion` LRU enforces 4096-entry cap (cluster-scoped only).
- `policy.adoption_weights` warn-cap at 4128 (warn-only, no eviction).

## [0.28.7] — 2026-05-02

Audit-driven hardening on v0.28.6. Closes 4 HIGH + 4 MED items from the
2026-05-02 v0.28 audit. 4 codex review rounds saturated to 0 P1 + 1 deferred
P2; default-OFF behavior bit-identical to v0.28.6. 1419 tests / 0 fail / 3
ignored.

### Fixed

- **H0** — `[ars.llm_judge].enabled` and `[ars.llm_judge.nightly_cron].enabled`
  defaults reverted from `true` (v0.28.6) back to `false` in code AND
  embedded `default.toml` per the v0.28 charter Non-Goal "Do not make LLM
  judge default-on". Routine `cargo install` upgrade no longer triggers
  implicit LLM API spend.
- **H1** — `bootstrap_priors_from_replay` consumer guarded against
  placeholder `signal_hint` producer until v0.29 producer lands.
- **H2** — `apply_local_fixes` performs drift-triggered Canary→Shadow
  rollback via `refresh_ars_parameter_policy` when
  `judge_calibration_state.judge_drift_alert*` is positive.
- **H3** — `route_context` shadow buckets isolated in separate
  `CONCEPT_SUMMARY_BY_CLUSTER_SHADOW_CAP = 4096` LRU.
- M-1 input-side `JudgeSurface` threading; M-2 watermark cutoff; M-9
  `DrainStats` per-reason counters + ledger saturation doctor check.

## [0.28.6] — 2026-05-02

ARS default-on + Trust & Measurement.

### Added

- `[ars.acceleration]`, runtime LLM judge, and nightly calibration
  default-on; runtime adoption fail-closed behind `ars_parameter_policy`.
- Scoped adoption weights for recall fusion/query/cluster and scalar
  surfaces.
- New `rein_trust_measurement` MCP tool, `rein trust-measurement` CLI,
  and `GET /api/trust-measurement` REST route — unified release-gate +
  eval-gates + index-consistency + active-learning report.

## [0.28.5] — 2026-05-01

Gradual ARS runtime adoption.

### Added

- `runtime_adoption_weight` field in `ars_parameter_policy`.
- Recall fusion, synthesis/concept gates, judge sample rates, LLM
  feedback decay, and SignalHint-derived useful-rate priors all gate
  through `runtime_adoption_weight`.

### Changed

- Adoption weight moves by at most `0.05` per durable adaptive snapshot.

## [0.28.4] — 2026-05-01

ARS acceleration full pass + new `rein_ars_acceleration_gate` MCP tool
(39 → 40 tools at v0.28.6).

## [0.28.3] — 2026-05-01

ARS dynamic scalar expansion: policy-gated dynamic adoption extended from
recall fusion to synthesis/concept cold-start and useful-rate thresholds.

## [0.28.2] — 2026-05-01

ARS dynamic parameter policy: `ars_parameter_policy` metadata activation,
trust-weighted static-to-learned fusion adoption, κ/drift-gated LLM judge
`weight_decay_rate`, `/api/adaptive` policy status, `rein doctor` policy
health checks.

## [0.28.1] — 2026-04-30

ARS recall canary activation: persists replay-learned global/query-type/
cluster six-dimensional fusion weights in
`AdaptiveState.learned_shadow_fusion`.

## [0.28.0] — 2026-04-30

ARS acceleration groundwork: default-off, shadow-first acceleration
controller. `/api/adaptive` exposes `ars_acceleration.shadow_fusion_replay`
preview fields.

## [0.27.6] — 2026-04-30

Codex hook parity + deployment hardening: 6 Codex hook events configured
by `rein init`, validated by `rein doctor`. Conservative deny-only shell
guardrails.

## [0.27.5] — 2026-04-29

R10-residual cleanup: cold archive too-large backoff, Cap A 4096-bucket
LRU eviction, cron `cron_claims` pre-LLM dedup.

## [0.27.4] — 2026-04-29

Audit-team remediation: 5-agent fan-out closed 1 CRIT + 8 HIGH + 9 MED +
5 LOW from a v0.27.3 audit. 10 codex rounds drove P1 to 0. 1265 tests.

## [0.27.3] — 2026-04-28

Full-audit remediation. Released to GitHub.

## [0.27.2] — 2026-04-27

Judge ledger / cache reaper: `judge_call_ledger` daily-cap reservation
shared across runtime + cron; judge cache reaper.

## [0.27.1] — 2026-04-27

E direction — runtime LLM judge. Opt-in via `[ars.llm_judge].enabled = false`.
Hooks at synthesis (Cap B) and concept-summary (Cap A) mint time. **7-invariant
judge contract J1-J7**. New MCP tools `rein_judge_synthesis` +
`rein_judge_concept_summary`. `[llm]` config inheritance.

## [0.27.0] — 2026-04-26

Cap A mirror feedback + fact-layer dedup: `rein_feedback_concept_summary`
mirrors Cap B's loop onto concept living-summary. Triple extraction +
N-memory merge + temporal supersede direction.

## [0.26.2] — 2026-04-26

32-bug security + correctness hotfix: 8 HIGH + 8 MEDIUM original audit +
16 audit-cycle additions across 11 follow-up codex rounds. Auth default-deny
via `http_request_needs_auth(method, path, gui_enabled)`. Recall correctness
with status-aware SQL filters. 1002 tests.

## [0.26.1] — 2026-04-25

D direction wiring fix + cold_archive eval. v0.26.0 hardcoded `query_type =
"Semantic"` made the per-cluster gate dead code for 5 of 6 query types;
fixed.

## [0.26.0] — 2026-04-25

ARS Cap C + D direction full vertical. Cap C cold-tier archival summary
(`rein_archive_summary_refresh`). D direction event-sourced loop:
`SynthesisInteraction` → `synthesis_feedback` consumer → per-query gate.

## [0.25.x] — 2026-04-24/25

ARS Cap B + Synthesis Lab. Opt-in recall-time LLM narrative synthesis
(`rein_recall` extended with `synthesize=true`). Synthesis Lab GUI page.
Hybrid hit-checker (Snowball Porter2 stem + Gemini cosine fallback).
LLM-judged hit checker (`REIN_EVAL_JUDGE=llm`).

## [0.24.0] — 2026-04-24

ARS Cap A — concept living-summary. Per-concept rolling LLM summary
refreshed via L3 adaptive policy + L4 concurrent CAS. Cross-cutting
peek+commit refactor across 5 consumer offsets. New MCP tools
`rein_concept_state` + `rein_concept_summary_refresh`. 819 tests.

## [0.23.0] — 2026-04-23

Resummerize + 7-invariant Lossless Compression Contract. LLM-driven
canonical recompression at the 10 KB `MergeInto` cap. Atomic
`apply_resummerize` with 5-way CAS + 3-strike exhaustion fuse + 5-minute
stale-claim takeover. Paired `rein-eval` McNemar non-inferiority test. 750
tests.

## [0.22.0] — 2026-04-22

KG pool + service wiring + try_get fast-path. 675 tests / 7 codex audit
rounds.

## [0.21.0] — 2026-04-20

A1 Operation Registry. `#[op]` proc-macro: each operation authored **once**
in source, dispatched via `inventory` to thin CLI / MCP / REST adapters.
Eliminated three parallel hand-maintained registries. 625 tests.

---

For pre-v0.21 history (v0.4 → v0.20), see git log and the GitHub Releases
page. The `v0.21 → v0.28` arc rebuilt rein around three axes: a unified
operation registry, an adaptive read-side synthesis (ARS) stack with
feedback-driven gates, and end-to-end audit-cycle hardening of every
adaptive surface.

[Unreleased]: https://github.com/lyr1cs/rein/compare/v0.28.8...HEAD
[0.28.8]: https://github.com/lyr1cs/rein/releases/tag/v0.28.8
[0.28.7]: https://github.com/lyr1cs/rein/releases/tag/v0.28.7
[0.28.6]: https://github.com/lyr1cs/rein/releases/tag/v0.28.6
[0.28.5]: https://github.com/lyr1cs/rein/releases/tag/v0.28.5
[0.28.4]: https://github.com/lyr1cs/rein/releases/tag/v0.28.4
[0.28.3]: https://github.com/lyr1cs/rein/releases/tag/v0.28.3
[0.28.2]: https://github.com/lyr1cs/rein/releases/tag/v0.28.2
[0.28.1]: https://github.com/lyr1cs/rein/releases/tag/v0.28.1
[0.28.0]: https://github.com/lyr1cs/rein/releases/tag/v0.28.0
[0.27.6]: https://github.com/lyr1cs/rein/releases/tag/v0.27.6
[0.27.5]: https://github.com/lyr1cs/rein/releases/tag/v0.27.5
[0.27.4]: https://github.com/lyr1cs/rein/releases/tag/v0.27.4
[0.27.3]: https://github.com/lyr1cs/rein/releases/tag/v0.27.3
[0.27.2]: https://github.com/lyr1cs/rein/releases/tag/v0.27.2
[0.27.1]: https://github.com/lyr1cs/rein/releases/tag/v0.27.1
[0.27.0]: https://github.com/lyr1cs/rein/releases/tag/v0.27.0
[0.26.2]: https://github.com/lyr1cs/rein/releases/tag/v0.26.2
[0.26.1]: https://github.com/lyr1cs/rein/releases/tag/v0.26.1
[0.26.0]: https://github.com/lyr1cs/rein/releases/tag/v0.26.0
[0.25.x]: https://github.com/lyr1cs/rein/releases?q=v0.25
[0.24.0]: https://github.com/lyr1cs/rein/releases/tag/v0.24.0
[0.23.0]: https://github.com/lyr1cs/rein/releases/tag/v0.23.0
[0.22.0]: https://github.com/lyr1cs/rein/releases/tag/v0.22.0
[0.21.0]: https://github.com/lyr1cs/rein/releases/tag/v0.21.0
