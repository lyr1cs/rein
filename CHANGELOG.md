# Changelog

All notable changes to **rein** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The full release notes (with audit-round breakdowns and operator-visible
schema changes) live on the [GitHub Releases page](https://github.com/lyr1cs/rein/releases).
This file is a condensed index intended for quick scanning.

## [1.3.0] — 2026-07-15

Self-supervised activation minor: the #A12 recall-fusion activation
infrastructure lands end-to-end (default fail-closed, shadow until promoted),
the LLM judge gains deterministic structural calibration, and destructive
dedup thresholds get a statistical safety floor. Config schema version bumps
to 3; database schema version bumps to 4 (both forward-migrated
automatically). 1819 lib tests / 0 fail; 27 workspace suites green; 7
adversarial review rounds.

### Added

- **#A12 self-supervised recall-fusion calibration** — a
  leave-one-evidence-out corpus derived from structural families (canonical
  evidence / concept / episode); SHA-256 family-disjoint 5-fold split with a
  permanent activation holdout fold that is never passed to the optimizer;
  side-effect-free `recall_loo_trace` replay (no recall-access / recall-hit /
  tiering writes, dynamic weights disabled); family-equal optimizer + paired
  top-3 McNemar holdout gate; sealed versioned A12 calibration state
  (metadata CAS, immutable revision history); one shared evidence resolver
  feeds policy refresh, release gate, and runtime. LOO exclusion is
  family-level, so supersede-chain relatives cannot leak held-out answers.
- **`a12_input_epoch` coherence** — a schema v4 migration installs 21
  triggers maintaining a durable input-epoch counter over every
  replay-relevant write. Calibration runs bind to the epoch and the final
  seal re-checks it inside `BEGIN IMMEDIATE`; the runtime resolver
  re-verifies epoch, behavior fingerprint, and judge trust on every recall. A
  missing epoch row self-heals at open; a malformed row is repaired only by
  `rein doctor --fix`, which atomically re-baselines the counter and
  invalidates any active A12 calibration.
- **Judge structural calibration** — four deterministic probe kinds
  (`SupportedExactSingle` / `SupportedExactMulti` / `UnsupportedNonce` /
  `QueryMismatch`) with per-surface versioned state; probe payloads cannot
  self-report labels; Ready requires all four kinds in one run; a model or
  rubric change invalidates the run. New `[ars.llm_judge.structural_anchors]`
  config (`off` | `monitor` | `enforce`, default `off`). Zero human pairs
  never fabricates a human κ; anchors hold the configured baseline only and
  never raise judge weight or sample rate; runtime-vs-nightly κ is a
  drift-only signal.
- **#C2 dedup calibration safety** — destructive lexical merges floored at
  the static threshold (the legacy learned `0.40` demoted to a shadow
  suggestion); vector cleanup back on its own cosine threshold; sealed
  versioned dedup calibration policy (train/holdout split by pair-hash +
  canonical family; `Ship` / `Bail` / `NoData`; never auto-lowers the static
  floor). Promotion requires an exact one-sided Clopper-Pearson false-merge
  bound — zero-failure 2% @ 95% needs ≥ 149 sealed negatives; the current
  n = 10 cohort reports `NoData`. Doctor, trust measurement, and the GUI
  expose static vs shadow vs hard-effective thresholds.
- **Observability** — typed `RecallFusionScopeHealthCode` (11 variants)
  drives doctor attention states (no prose matching); benign absence (fresh
  install / shadow-only) is doctor-Ok; per-provenance holdout agreement
  diagnostics with a `provenance_direction_conflict` flag. New doctor checks
  `a12_input_epoch` and `recall_fusion_calibration`.
- **`REIN_EVAL_GATE_ROOT`** environment variable — absolute eval-gate
  artifact root for installed daemons outside a source checkout; relative
  paths fail closed.

### Changed

- **`ars_parameter_policy` payload schema 2 → 3** — schema-2 rows load
  fail-closed and are CAS-upgraded; future-schema rows stay byte-preserved.
  Automatic (non-human) evidence is restricted to `recall_fusion:*` adoption
  keys, steps at most `0.05` per refresh, and counts only while the judge
  structural trust gate holds; judge pair evidence enforces a 7-day TTL at
  read. Tampered calibration state zeroes adoption (never falls back to
  human evidence); stale-specific state degrades to sealed pure-human
  evidence only.
- **Config schema version 2 → 3** — v2 configs migrate automatically at
  load; older binaries refuse newer-stamped configs (existing downgrade
  guard).
- **Compiled default Gemini model id** migrated from the retired
  `gemini-3.1-flash-lite-preview` (retired 2026-05-25) to the stable
  `gemini-3.1-flash-lite`; `rein doctor` warns when an operator config still
  pins the retired id.

## [1.2.0] — 2026-06-12

Algorithm + hardening minor: the #A5 fact-layer reader lands, the hooks
extraction pipeline gets a full-sweep audit remediation (28 confirmed
findings fixed, two of them High), and the proxy moves to an explicit auth
policy. Config schema version bumps to 2; database schema version bumps
to 3 (both additive, forward-migrated automatically).

### Added

- **#A5 dedup-reuse reader** — the dedup gray-zone now reuses persisted
  triples when a new `memory_triple_meta` stamp validates (content hash +
  extractor version; schema v3). Any doubt recomputes and self-heals; a
  corpus equivalence gate pins reuse == recompute.
- **hook_stop content-level dedup** — mid-session flushes record a
  flushed-content hash ledger; stop-time extraction strips only
  byte-provably already-extracted tool output and keeps every
  conversation fact.
- **`[proxy].auth` policy** (`"bearer_required"` / `"public"`) replaces the
  removed `allow_unauthenticated_loopback` bool, with automatic load-time
  migration (config schema v2). Explicit `public` is honored on loopback
  binds only.
- **ARS canary quality blocker** — `shadow_fusion_replay_not_ready` is now
  a canary blocker, so a pure volume ramp can no longer promote the #A12
  simplex without a replay verdict.

### Fixed

- **User turns were dropped from session extraction** (High) — the
  transcript parser accepted only the legacy `"human"` entry type; Claude
  Code emits `"user"`. User-stated facts, preferences, and decisions now
  reach the extractor.
- **Active sessions stopped ingesting after the first Stop capture**
  (High) — prefix-fingerprint suppression treated every longer transcript
  snapshot as a duplicate while re-arming its own window. Growing
  transcripts now replace their pending job in place.
- 26 further audit findings across the store, hooks queue, recall, the
  adaptive engine, and the proxy — including M5 cold-strip CAS, supersede
  chain splicing on delete, HNSW rebuilds sourcing from `vec_memories`
  (fixes >10k-memory truncation and the `migrate --reindex` dirty loop),
  non-idempotent POST retries, and a cross-process single-flight lock for
  the adaptive pipeline. Full details on the GitHub Release.

## [1.0.1] — 2026-06-02

Correctness patch. No schema change, no config change — drop-in over 1.0.0.

### Fixed

- **Transitive canonical resolution in supersede chains** — a recall query
  matching the oldest revision's text in a depth-≥2 supersede chain (A→B→C)
  could surface the stale middle revision B instead of the live tip C.
  `canonical_id_for` now resolves transitively to the live tip (visited-set
  bounded, cycle-safe), so every reader — recall collapse, canonical lookups,
  dedup — lands on the live successor on any database, including ones written
  before this fix. The strong-signal recall fast-path and the canonical
  collapse now share one liveness predicate so the channel-skip optimization
  can never over-count survivors. No data loss in the prior behavior
  (provenance was preserved); the live tip was always independently recallable.

## [1.0.0] — 2026-05-31

The 1.0 freeze. rein commits to a stable surface and ships its durable
fact-layer foundation. (Releases 0.31–0.38 between this and the last
CHANGELOG entry are detailed on the GitHub Releases page.)

### Added

- **#A5 durable triple persistence** — opt-in `[dedup].persist_triples`
  (default off) persists extracted `(subject, predicate, object)` triples
  into a new `memory_triples` table (the schema-versioning framework's first
  forward migration, schema version 2). Default recall/dedup behaviour is
  unchanged when the flag is off.
- **`config_version`** config key + a load-time forward-compatibility guard:
  a config written by a newer rein is refused rather than misread.
- **REST `/v1/*` versioned alias** for `/api/*` (query preserved). `/v1`
  authenticates with a header token (`Authorization: Bearer …` /
  `x-rein-token`); the browser GUI session cookie stays scoped to `/api`.
- Documentation for 96 previously-undocumented configuration fields.

### Changed

- **Minimum Supported Rust Version pinned to 1.86** (declared in `Cargo.toml`,
  enforced by CI). Replaced three `is_multiple_of` (stabilized in Rust 1.87)
  uses with `%` to honour the floor.
- Baseline database schema and the MCP tool argument surface are now frozen
  behind regression tests; all future schema changes go through the
  forward-migration framework.

## [0.30.1] — 2026-05-11

Operator-driven follow-up to v0.30.0 closing two debugging-tax issues
that surfaced during the first end-to-end claude.ai integration cycle.

### Fixed

- Every `/mcp` 4xx response now ships as a JSON-RPC 2.0 error envelope
  (`{"jsonrpc":"2.0","id":<echoed>,"error":{"code":<code>,"message":"..."}}`)
  with `Content-Type: application/json`. Anthropic's MCP client and other
  spec-compliant clients can now surface the actual rejection reason
  (host guard / browser-origin guard / auth policy / public-mutation
  block / OAuth challenge) instead of "An unknown error occurred
  connecting to the MCP server". OAuth 401 still carries the
  `WWW-Authenticate: Bearer resource_metadata="..."` header per RFC 6750
  and RFC 9728. `/api/` REST rejections retain plain-text bodies.
- `rein doctor` `oauth_provider` check emits a `Configuration` WARN when
  `auth_policy = "oauth"` AND `oauth_clients > 0` AND `active_grants = 0`.
  Catches the v0.30.0 release-day `refresh_token_fingerprint` migration
  revoke (and any other state where every grant is revoked) and gives
  the operator a one-line actionable hint to remove + re-add the rein
  connector on claude.ai instead of chasing a generic error message.

### Documented

- `docs/manual/02b-remote-mcp-deployment.md` adds a
  `## Choosing your auth posture` decision tree covering
  `public` / `oauth` / `bearer_required` / `loopback_only`. Calls out
  `auth = "public"` as the simplest path for single-user private
  Tailscale Funnel deployments (mutation gate keeps writes locked) and
  documents the v0.30.0 OAuth grant-revoke trap so anyone upgrading
  along the OAuth path knows the recovery action.

### Security

- Redacts operator-specific Tailscale Funnel FQDN, Quick Tunnel URLs,
  Tailscale CGNAT IPs, claude.ai conversation IDs, internal vault path
  references, and author-perspective phrasings from public docs and
  test fixtures. Doc and fixture string substitutions only — no
  runtime / binary / behavior changes (replacement Tailscale CGNAT
  addresses stay in `100.64.0.0/10` and remain non-loopback for the
  same test assertions).

## [0.28.18] — 2026-05-09

Agent-team audit follow-up to v0.28.17's Cowork/auth diagnostics.

### Fixed

- Public unauthenticated HTTP now blocks mutating MCP/REST calls from
  non-loopback Hosts while preserving read-only Cowork recall.
- HTTPS same-origin browser mutation requests are accepted behind
  TLS-terminating reverse proxies.
- `rein doctor` validates both new and legacy Codex MCP tables, rejects
  non-stdio `rein serve --sse/--gui/--proxy` MCP entries, and warns on
  `REIN_DB`, `REIN_CONFIG`, and `HOME` database-split risks.
- DXT and Claude plugin manifest versions are checked against Cargo.

### Documented

- Install snippets and DXT artifact names now point at the current release.
- Codex config examples prefer `[mcp_servers.rein]` and `[features].hooks`.

## [0.28.17] — 2026-05-09

Cowork/auth diagnostics patch.

### Fixed

- `rein serve --gui` warns when `REIN_HTTP_TOKEN` is set alongside
  `[server].allow_unauthenticated_loopback=true`.
- `rein doctor` reports the same token-vs-loopback auth conflict.
- `rein doctor` warns when Codex's Rein MCP entry points at a non-loopback
  HTTP URL whose database may differ from local CLI recall.

## [0.28.16] — 2026-05-09

Codex 0.129 `[mcp.<name>]` → `[mcp_servers.<name>]` compatibility. `rein init`
writes the new table and preserves legacy customizations.

## [0.28.15] — 2026-05-09

Codex 0.129 `[features].codex_hooks` → `[features].hooks` compatibility.

## [0.28.14] — 2026-05-07

Docs-only polish on top of v0.28.13. Closes the
`cargo install --git --tag` footgun discovered while deploying v0.28.13 to
an aarch64 Linux host.

### Documented

- **`docs/manual/02-installation.md`** — new "Remote Install (Pinned Tag)"
  section that shows the correct `cargo install --git ... --tag X --locked`
  invocation, with a callout explaining why `--locked` is mandatory: without
  it, `cargo install --git` ignores the committed `Cargo.lock` and
  re-resolves transitive deps to the latest semver-compatible versions on
  crates.io, which can pull in newer C/SIMD code (e.g. `usearch 2.25` →
  `numkong 7.6.0`) requiring a host toolchain newer than what the target
  ships (GCC 13+ / clang 17+ on aarch64 Linux). Plus a Troubleshooting row
  pointing at this fix when users hit the
  `inlining failed in call to 'always_inline' 'vdotq_s32'` symptom.
- **`README.md`** — one-line install snippet now shows both the latest-master
  and pinned-tag forms, with a short callout linking to the manual section.
- **GitHub Release notes for v0.28.12 + v0.28.13** retroactively patched to
  add `--locked` to the upgrade command.

### Fixed

- **`AGENTS.md` overview line** — version pin updated from `v0.28.11` to
  `v0.28.14` so `rein doctor`'s `overview_version` check stops warning
  about a stale overview.

### No runtime behavior changes from v0.28.13

Binary is bit-identical with the version field bumped. Same dependency
graph, same `Cargo.lock`, same 1462 / 1462 tests.

## [0.28.13] — 2026-05-07

Second hotfix today. Restores remote MCP access via Tailscale Funnel
(and any other reverse-proxy fronting rein) by bridging rein's
`[server].allowed_hosts` config into rmcp 1.6's own streamable-HTTP host
guard.

### Fixed

- **rmcp host-guard bridge (hotfix)** — rmcp 1.6 added its own
  DNS-rebinding host check inside `StreamableHttpServerConfig` (default
  `["localhost", "127.0.0.1", "::1"]`) which runs **ahead** of rein's
  `validate_http_request_host`. rein already had `[server].allowed_hosts`
  in config and a working guard, but the rmcp config was built with
  `default()` and never told about the operator-supplied allowlist —
  silently rejecting non-loopback Hosts (e.g. Tailscale Funnel hostnames)
  before the request reached rein's middleware. Now `run_http` calls
  `.with_allowed_hosts(...)` or `.disable_allowed_hosts()` based on bind
  shape, preserving documented loopback / specific-bind / token-protected
  wildcard deployment modes. Codex review clean.

### Changed

- `rmcp = "1.2"` → `"1.6"` in `crates/rein/Cargo.toml`. The
  compatible-update was already resolving to 1.3.0 in lockfile; 1.6.0
  adds the `with_allowed_hosts` / `disable_allowed_hosts` builder methods
  without other API breakage.

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

[Unreleased]: https://github.com/lyr1cs/rein/compare/v0.30.1...HEAD
[0.30.1]: https://github.com/lyr1cs/rein/releases/tag/v0.30.1
[0.30.0]: https://github.com/lyr1cs/rein/releases/tag/v0.30.0
[0.28.18]: https://github.com/lyr1cs/rein/releases/tag/v0.28.18
[0.28.17]: https://github.com/lyr1cs/rein/releases/tag/v0.28.17
[0.28.16]: https://github.com/lyr1cs/rein/releases/tag/v0.28.16
[0.28.15]: https://github.com/lyr1cs/rein/releases/tag/v0.28.15
[0.28.14]: https://github.com/lyr1cs/rein/releases/tag/v0.28.14
[0.28.13]: https://github.com/lyr1cs/rein/releases/tag/v0.28.13
[0.28.12]: https://github.com/lyr1cs/rein/releases/tag/v0.28.12
[0.28.11]: https://github.com/lyr1cs/rein/releases/tag/v0.28.11
[0.28.10]: https://github.com/lyr1cs/rein/releases/tag/v0.28.10
[0.28.9]: https://github.com/lyr1cs/rein/releases/tag/v0.28.9
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
