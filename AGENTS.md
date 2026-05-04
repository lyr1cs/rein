# AGENTS.md

## Overview

rein v0.28.8 — Multi-source cross-validated memory MCP server for AI agents. Rust single binary. 40 MCP tools (v0.28.6 added `rein_trust_measurement`; v0.28.4 added `rein_ars_acceleration_gate`; v0.28.7 + v0.28.8 hardened existing surfaces without adding tools). v0.28.8 is a second-pass audit hardening on v0.28.7: 17 codex review rounds, 15 P2 + 1 P3 closed, 0 P1 throughout, 2-consecutive-clean saturation at R16/R17. Default-OFF runtime behavior is bit-identical to v0.28.7 — only `[ars.acceleration]` defaults `true`; `[ars.llm_judge]` + `[ars.llm_judge.nightly_cron]` stay default-off per the v0.28 charter Non-Goal. Live learned parameters still fail closed behind a healthy canary `ars_parameter_policy` row plus positive scoped adoption weights; runtime adoption remains scoped per parameter, surface, query type, and cluster; SignalHint feedback remains active outside shadow mode; release-gate output exposes scoped weights; and Trust & Measurement reports release gate, eval gates, index consistency, background observability, and active-learning status. Unified operation registry (CLI / MCP / REST authored once via `#[op]` macro). Self-adaptive engine (M1-M6), 3-channel retrieval (FTS + Vector + KG), transparent LLM proxy, async memory pipeline, unified dedup architecture, canonical-first read model, evidence-aware recall, hybrid CJK tokenization, cluster-aware admission, service management, and Neural Wiki GUI remain core surfaces.

## v0.28.8 release (2026-05-04)

Second-pass audit hardening on v0.28.7. **17 codex review rounds** (R1–R17) saturated at 2-consecutive-clean; **15 P2 + 1 P3** findings closed; **0 P1** throughout. **1462 tests / 0 fail / 3 ignored / 0 clippy / 0 fmt.** Default-OFF behavior bit-identical to v0.28.7.

- **M-8 (R13 — cluster-bucket alignment, structural)**: learn-time `top_vec_hit_cluster` now PREFERS looking up the recall-time top-vec memory id in the CURRENT `memory_clusters` map, returning the post-recluster cluster id a fresh read would also see. The legacy R3 `cluster_version_at_recall` version-match path stays as a backward-compat hook for pre-R13 events. Closes the M4-then-M2 normal pipeline-order bug where `cluster_version_at_recall` was invalidated for every event in the common path, silently dropping scoped learning whenever the read-time top-vec hit was filtered or canonical-collapsed between recall and event emission. New event field: `query_top_vec_memory_id_at_recall: Option<String>`.
- **L6 (R12 — LRU fallback preservation)**: `evict_learned_shadow_fusion_lru_if_at_cap` and `shrink_learned_shadow_fusion_to_cap` restrict eviction targets to cluster-scoped buckets (`{query_type}:{cluster_id}` shape, identified by `is_cluster_scoped_bucket` predicate). The `global` + per-query-type fallback chain that `get_shadow_fusion_weights` depends on stays intact under high cluster cardinality; if the cluster-scoped victim pool is exhausted, the cap is allowed to overshoot rather than corrupt the fallback chain (degenerate state — only ~7 fallback keys exist).
- **`ars_parameter_policy` schema robustness**: peek `schema_version` from raw JSON Value before typed deserialize (R8 — future-schema rows with unknown enum variants no longer fall into `Corrupt` and get deleted by `doctor --fix`); CAS UPDATE predicate uses schema-aware COALESCE default `?4` rather than `0` (R8 — rows missing the `schema_version` field accept refresh); peek comparison uses `>` rather than `!=` (R15 — older `schema_version=0` rows recover via `Corrupt` arm rather than stalling refresh forever); `repair_corrupt_parameter_policy` wraps load+delete in `BEGIN IMMEDIATE` (R10 — peer-race window between earlier load and unconditional delete closed).
- **M-1 persistence-side per-surface scalar split**: 4 new `ars_effective_scalars` keys — `judge_sample_rate_{cold_start,warm}_{synthesis,concept_summary}` — with `ars_effective_scalar_with_legacy_fallback` consulting the legacy cluster-shared key as a one-time first-tick-after-upgrade source. Per-surface drift visibility is now end-to-end. Schema-additive — pre-v0.28.8 snapshots remain readable.
- **M-5 / M-6**: M-5 anchors `static_threshold` on the config default when `runtime_adoption_weight ≈ 0` (closes the rollback-window hole in v0.28.7's H2 fix-closed promise); M-6 outer-blends ARS simplex score against route-aware `legacy_score` by `runtime_adoption_weight` (preserves ExactKeyword's alpha=0.85 BM25-heavy signal during partial-adoption canaries).
- **R10 P2 — SQL-fallback cluster id**: split `query_cluster_id_from_snapshot` (used for event payload, atomic with snapshot's cluster_version) from `query_cluster_id` (used for read-time alpha selection, falls back to SQL for live serving). The recorded field falls to `None` when SQL fallback fired, forcing learn-time to re-derive from candidates.
- **R10 P3 — doctor delete race**: `repair_corrupt_parameter_policy` (new public helper in `store/ars_parameter_policy.rs`) wraps load + DELETE in `BEGIN IMMEDIATE`, observes status under the write lock, and DELETEs only if still `Corrupt`.
- **L1 / L4 / L5 / L7**: `sanitize_bootstrap_priors` bounds weights and `prior_confidence` by `1e6`; auth-policy regression locks for `/api/trust-measurement` + `/api/ars-acceleration-gate` (currently `Public`); doctor recovery covers Corrupt policy rows; release-gate test coverage extended (5 new tests in `tests/ars_release_gate.rs`).
- **Schema additions** (snapshot blob): 4 per-surface scalar keys land alongside the existing legacy keys (legacy keys still updated for downgrade-rollback compat); `policy.adoption_weights` warn-cap at 4128 (warn-only, no eviction); `learned_shadow_fusion` LRU at 4096 (cluster-scoped only).

## v0.28.7 release (2026-05-02)

- **H0 — `[ars.llm_judge]` defaults reverted to `false`**: v0.28.6 defaulted `[ars.llm_judge].enabled` and `[ars.llm_judge.nightly_cron].enabled` to `true`, which would have triggered implicit LLM spend on a routine `cargo install` upgrade. v0.28.7 reverts both defaults to `false` in code AND embedded `default.toml` per the v0.28 charter Non-Goal "Do not make LLM judge default-on" — runtime LLM judge stays opt-in until v0.29 surface-policy gating lands. `[ars.acceleration]` stays `true` (the canary path is unchanged). Operators who already opted in via TOML config see no behavior change.
- **H1 — bootstrap-priors replay consumer held inactive**: the `bootstrap_priors_from_replay` consumer is wired but the v0.29 producer hasn't landed; the consumer is guarded against the placeholder `signal_hint` producer so it never advances against an empty source.
- **H2 — drift-triggered canary→shadow rollback**: `apply_local_fixes` now refreshes `ars_parameter_policy` when `judge_calibration_state.judge_drift_alert*` is positive while the policy is in Canary. The refresh demotes the policy back to Shadow with `runtime_adoption_weight = 0` on the next `refresh_ars_parameter_policy` tick — drift cannot be "merely logged".
- **H3 — `route_context` shadow buckets isolated from production cap**: shadow `route_context` buckets get a separate `CONCEPT_SUMMARY_BY_CLUSTER_SHADOW_CAP = 4096` LRU; recall via the shadow path cannot evict production cache entries.
- **M-1 input-side**: `JudgeSurface` is threaded through 5 helpers + handlers so per-surface drift visibility (Synthesis vs ConceptSummary) is preserved end-to-end. **M-2**: `bootstrap_priors_from_replay` watermark cutoff now uses state watermark (D3 replay-idempotence). **M-9**: `DrainStats` per-reason counters + `tracing::warn` on dropped cap + doctor `judge_call_ledger` saturation check. **M-4**: docs.
- **Verification**: 1419 tests / 0 fail / 3 ignored, 3 `codex review --uncommitted` rounds. Default-OFF behavior bit-identical to v0.28.6. M-1 persistence-side residual + LOW/NIT items deferred to v0.29.

## v0.28.6 release (2026-05-02)

- **Default-on, fail-closed ARS acceleration**: `[ars.acceleration]`, `[ars.llm_judge]`, and `[ars.llm_judge.nightly_cron]` now default on, but runtime adoption still requires a healthy canary parameter policy and positive scoped weights.
- **Scoped dynamic adoption**: `ars_parameter_policy.adoption_weights` covers recall fusion global/query/cluster keys plus `synthesis_gate`, `concept_summary_gate`, `judge_sample_rate`, `llm_feedback_decay`, and `signal_hint_priors`.
- **Auto promote / rollback**: adaptive refresh promotes eligible evidence to Canary with gradual scoped weights and rolls back to Shadow with zero weights when evidence, config, or policy health stops allowing runtime adoption.
- **Trust & Measurement**: `rein_trust_measurement` / `rein trust-measurement` / `/api/trust-measurement` reports release-gate state, eval gates, index consistency, background observability, and active-learning status.

## v0.28.5 release (2026-05-01)

- **Gradual ARS runtime adoption**: `ars_parameter_policy` now stores `runtime_adoption_weight` in `[0, 1]`; adaptive refresh moves it by at most 0.05 per durable snapshot toward the evidence-backed target and resets it to 0 outside live canary mode.
- **Weight-gated dynamic tuning**: recall fusion, synthesis/concept gates, judge sample rates, LLM feedback decay, and SignalHint-derived useful-rate priors multiply dynamic trust by `runtime_adoption_weight`, so static configured values remain the anchor throughout rollout.
- **Observability**: `rein doctor` and `rein ars-acceleration-gate` report the adoption weight alongside `live_allowed`, making stale rows, missing rows, and partial canaries visible without changing runtime defaults.

## v0.28.4 release (2026-05-01)

- **ARS acceleration full pass**: SignalHint/bootstrap priors now feed useful-rate formulas, dynamic scalar values persist for smoothing, and judge drift calibration tracks synthesis vs concept-summary separately.
- **Cap A recall-context production threading**: concept-summary GUI feedback dual-folds into the synthetic judge-aligned bucket and the real recall route bucket; `concept_state` prefers warmed real route buckets before falling back to synthetic.
- **Release/eval and optimizer groundwork**: `rein_ars_acceleration_gate` reports canary/default-on readiness without writing config or policy, judge input caps are configurable, recall-ranking judge jobs are safely recognized and dropped while default-off, and shadow fusion replay includes deterministic GP+EI proposals.

## v0.28.3 release (2026-05-01)

- **Dynamic ARS scalar gates**: synthesis and concept-summary cold-start/useful-rate gates now resolve effective values from static config plus calibrated adaptive evidence. Runtime adoption is fail-closed behind `ars_parameter_policy`, canary mode, calibration, and drift checks.
- **LLM acceleration hints**: synthesis and concept-summary judge jobs preserve shadow-only `signal_hint` payloads through cache enqueue and manual rehydrate, giving the worker bounded deterministic evidence without new LLM calls.
- **Shadow optimizer expansion**: `search/alpha_optimizer.rs` evaluates deterministic simplex candidates, accessed centroids, and accessed-vs-other gaps for six-dimensional fusion replay, so blended BM25/vector/KG/episode/support/diversity weights can be learned in shadow mode.

## v0.28.2 release (2026-05-01)

- **ARS parameter policy**: live ARS acceleration now requires a healthy metadata-backed `ars_parameter_policy` activation row. Missing/corrupt policy rows fail closed to disabled, `rein doctor` reports policy health, and `/api/adaptive` exposes the policy status.
- **Dynamic adoption weights**: learned fusion weights are blended from static priors through a trust function instead of replacing static values abruptly. Trust is driven by evidence count, calibration, drift state, prior strength, and canary mode.
- **LLM feedback acceleration**: the LLM judge `weight_decay_rate` can move from the static config value toward calibrated κ reliability only under policy-gated canary mode. Drift alerts zero the LLM contribution.

## v0.28.1 release (2026-04-30)

- **ARS recall canary activation**: `shadow_only=false` is now accepted as an explicit canary mode. Recall reads eligible six-dimensional fusion weights from `AdaptiveState.learned_shadow_fusion` and applies them only after live-row filtering, before cold-tier filtering and take/limit.
- **Snapshot-backed dynamic weights**: shadow replay now persists global, query-type, and query-type+cluster weights through the normal adaptive snapshot save path. Default `enabled=false` and shadow mode still leave production ranking unchanged.
- **Unchanged**: synthesis gates, concept summary behavior, and default deployments remain on the existing shipped path.

## v0.28.0 release (2026-04-30)

- **ARS acceleration groundwork**: `[ars.acceleration].enabled = false` by default and `shadow_only = true` was the safe default. v0.28.0 production recall fusion, synthesis gates, and summary behavior remained on the existing shipped path.
- **Shadow fusion replay observability**: `/api/adaptive` now includes `ars_acceleration.shadow_fusion_replay` with bounded preview fields: `enabled`, `shadow_only`, `status`, `replay_limit`, `eligible_samples`, `min_samples`, `global`, `by_query_type`, and `by_cluster`. Disabled/default installs report `status: "disabled"`, zero eligible samples, `global: null`, and empty bucket arrays.
- **R7-#1 shipped as shadow-first groundwork**: recall event replay can preview global, query-type, and cluster-scoped fusion weights from committed recall/access signals without committing adaptive offsets or altering production scoring.
- **Deferred at v0.28.0**: production activation of accelerated fusion, non-shadow mode, and any default-on ARS acceleration remained out of scope until the v0.28.1 canary slice.

## v0.27.6 release (2026-04-30)

- **Codex hook parity**: `rein hook` now covers `session-start`, `pre`, `permission`, `post`, `compact`, `prompt`, and `stop`; `SessionStart` / `UserPromptSubmit` can emit official `hookSpecificOutput.additionalContext`; `PreToolUse` / `PermissionRequest` apply conservative deny-only guardrails.
- **Install + doctor coverage**: `rein init` configures the six Codex hook events and enables `[features].codex_hooks = true` in Codex config; `rein doctor` reports the hook set as healthy only when all expected events are present.
- **Deployment**: local and Mac mini binaries updated; Mac mini launchd plists run `rein serve --gui` and `rein serve --proxy` through `/bin/zsh -l -c`, and Homebrew Rust (`cargo` / `rustc`) is installed for source builds.

## v0.27.5 release (2026-04-29)

- **Released and deployed**: `Cargo.toml` carries `0.27.5`; tag `v0.27.5` pushed; GitHub release published with 18.2MB GUI binary; `~/.cargo/bin/rein` replaced v0.27.4 → v0.27.5; GUI + proxy services restarted; doctor reports operational.
- **R10-residual cleanup** (closes 3 P2s deferred from v0.27.4): **R1** `memories.last_too_large_at` column + `claim_batch` ORDER BY backoff (NULL first → oldest stamp → FIFO) so oversized cold rows don't starve newer ones; stamp/clear discipline across success / manual refresh / `update()` semantic_changed / `apply_evolution` refine. **R2** `ClusterConceptSummaryStats.last_event_id` + `evict_concept_summary_lru_if_at_cap` helper replaces v0.27.4 drop-new-bucket so vaults with > 4096 distinct feedback-receiving concepts learn fresh signal. **R3** new `cron_claims` table + atomic `try_claim_cron` (INSERT OR IGNORE) primitive arbitrates concurrent crons before `reserve_call`/LLM (no double cap-burn); `claim_token` ownership-safe release + 5min stale-claim takeover + post-claim TOCTOU re-check + post-emit-crash reaper.
- **Codex saturation**: 10 serial `codex review --uncommitted` rounds — R6 + R10 fully clean (0 P1 / 0 P2 / 0 P3). Full ledger: `reviews/review-20260429-v0.27.5-remediation-report.md`.
- **Engineering**: 1035 lib tests / 0 clippy / 0 fmt. 16 files / +969 / -167.
- **Out of scope (still v0.28+)**: R7-#1 architectural threading (recall→enqueue cross-layer `(query_type, cluster_id)`); `[ars.llm_judge].enabled = false` default unchanged.

## v0.27.4 release (2026-04-29)

- **Released and deployed**: `Cargo.toml`/`Cargo.lock` carry `0.27.4`; tag `v0.27.4` pushed; GitHub release published with 18.1MB GUI binary; `~/.cargo/bin/rein` replaced v0.26.2 → v0.27.4; GUI + proxy services restarted; doctor reports operational. Vault release notes: `docs/devlog/v0.27.4-release-notes.md`.
- **Audit-team remediation**: 5-agent disjoint-slice fan-out closed 1 CRITICAL + 8 HIGH + 9 MEDIUM + 5 LOW from the 2026-04-28 v0.27.3 audit (vault doc `reviews/review-20260428-agent-team-v0.27.3-audit.md`). 10 serial `codex review --uncommitted` rounds saturated at 0 P1 across R7-R10 (full ledger: `reviews/review-20260429-v0.27.4-remediation-report.md`).
- **Headline fixes**: **C1** `[server,proxy].allow_unauthenticated_loopback` default flipped to `false` in code AND embedded `default.toml` (was bypassed by TOML merge); **E1** `refine_concept` conditional `living_summary*` clear; **E2** M5 strip `BEGIN IMMEDIATE` + post-COMMIT side-index discipline; **B1** Cap C input cap with wrapper-overhead reserve; **C2** proxy Host/Origin guard parity with server; **C3** wildcard bind requires token OR allowed_hosts; **C4** `/api/version` + `/rein/metrics` auth-gated; **D1+D2** Cap A bucket alignment via SHA-256-prefix synthetic `cluster_id` (closes R7-#1 deferred from v0.27.x); **A1+A2** judge handler null-`concept_id` reject + SQL hardening; **A3** OfflineCron UNIQUE migration with `json_valid` guard; **A5** ledger 7-day prune independent of cache TTL; **B2** dispatch ceiling pre-`reserve_call`; **D3** `*LlmJudge` replay-idempotence tests.
- **Engineering**: 1265 tests / 0 fails / 0 clippy / 0 fmt / 25 files / +1977 / -191. ~23.2 GB build artifacts cleaned post-deploy.
- **Known v0.27.5+ residuals** (documented in code comments at affected sites): cold_archive oversized backoff via `last_too_large_at`, Cap A 4096-bucket LRU eviction, cron pre-LLM `cron_claims` row.

## v0.27.3 highlights (2026-04-28)

- Full audit remediation patch (commit `6a3574e`); served as the base for the v0.27.4 audit-team work.
- Tracked the full-vault/source audit ledger in vault doc `docs/backlog/v0.27.3-full-audit-remediation.md`.
- v0.27.0 added Cap A mirror feedback + fact-layer dedup; v0.27.1/v0.27.2 added runtime LLM judge, `[llm]` config inheritance, cache reaper, `judge_model_override` extractor swap, judge ledger/doctor checks.

## v0.26.2 highlights (2026-04-26)

- **32-bug security + correctness hotfix**: 8 HIGH + 8 MEDIUM from a user-driven Codex (gpt-5.4) audit on v0.26.1, plus 16 audit-cycle additions discovered across 11 follow-up `codex review --uncommitted` rounds (3 P1 + 13 P2). Closed via 5-agent parallel fan-out (auth+proxy / recall+search / store integrity / adaptive ops / synthesis full-stack); R11 reported zero new findings.
- **Auth default-deny**: `mcp/server.rs::http_request_needs_auth(method, path, gui_enabled)` — non-`/api/`, non-`/mcp` paths now require bearer auth unless `gui_enabled` (closes the `POST /not-mcp` MCP-service bypass).
- **Recall correctness**: SQL `status IN ('active','updated')` filter on every FTS+vec path; `recall.rs` adds defensive `memory_map.retain` for Tantivy/KG/episode channels that bypass SQL; superseded rows preserved so `collapse_to_canonicals` maps them to live canonical successors.
- **`apply_evolution` side-index discipline**: deprecation uses `mark_superseded` + in-savepoint `vec::delete_embedding` + post-RELEASE Tantivy/HNSW removal; refine path does DB-only update inside savepoint + queues side-index refresh until after RELEASE.
- **Synthesis bucket round-trip**: `RecallSynthesisOutcome.{query_type, cluster_id}` carried backend → REST → GUI metadata (`SynthesisCard` and `SynthesisLab.immediate_requery` POST). cluster_id calc hoisted so `decide_synthesize` and `outcome.cluster_id` use the same source — D-direction adaptive gate now lives on GUI traffic.
- **`update()` archival lifecycle**: clears all 6 archival_summary cols (incl. `archival_summary_at`) on semantic content change; drops `cold_archive` row only when `content` itself changes (not topic/summary alone).
- **M5 strip bypasses `update()`**: raw SQL + direct `update_tantivy` + vec/HNSW invalidation, so the strip can't trigger update()'s cold_archive delete.
- **Engineering**: 1002 tests / 0 clippy / 0 GUI lint / 19 files / +2318 / -92.

## v0.26.1 highlights (2026-04-25)

- **D direction query_type wiring fix**: v0.26.0 hardcoded `query_type = "Semantic"` inside `run_recall_synthesis`, which silently routed every non-Semantic query event into one bucket while the per-query gate read another — turning the per-cluster `useful_rate` gate into dead code for Episodic / Temporal / Preference / ExactKeyword / Exploratory queries. v0.26.1 introduces `QueryType::synthesis_bucket_label()` returning the canonical capitalised label and threads it from the recall handler (MCP/CLI + REST) into `decide_synthesize`.
- **Configurable cold-start threshold**: new `[ars].synthesis_cold_start_n` config knob (default 10, matches `SYNTHESIS_COLD_START_N` const). Operators on a fresh canary may lower to 3-5 to let the per-cluster gate fire against the partial event stream a soak collects without waiting for the bootstrap default.
- **`rein-eval cold_archive` subcommand**: parallel to `concept-summary` and `synthesis`. Baseline scores the post-M5-strip surface (`topic + summary` — `memory.content` is replaced by `memory.summary` per `ops/adaptive.rs:750`); `Run` invokes `ColdArchiveSummaryGenerator::generate` over fixture content and scores `topic + summary + archival_summary`; `Compare` runs paired McNemar under the additive `DecideShipKind::Synthesis` rule. 7 fixtures bundled across 4 categories (technical_decisions × 3, narrative_logs × 2, cjk_mixed × 1, multi_topic × 1). `print_summary` now flags `n < 12` corpora as power-limited.
- **Engineering**: 967 tests / 0 clippy / +5 unit tests for cold_archive + +3 for D direction wiring (synthesis_bucket_label correctness, Episodic-query bucket-lookup wiring, configurable cold_start_n).

## v0.26.0 highlights (2026-04-25)

- **ARS Capability C (cold-tier archival summary)**: opt-in `[ars].cold_archive_enabled` flag. Slow-channel worker in `ops/cold_archive_summary.rs` claims cold-tier rows via 5-way CAS (id + per-row ULID `archival_claim_token` + status-live + tier-still-cold + snapshot updated_at + needs-still-1 + superseded-NULL), runs a 3-invariant lossless contract (bounded length INV-3 + script preservation INV-5 catches LLM auto-translation + trigram coverage INV-1 catches fabrication), persists archival_summary on the row. 3-strike per-pass exhaustion fuse, 5-min stale-claim takeover, 180-second batch wall-time budget. New manual refresh op `rein_archive_summary_refresh`. Step 3a in `run_adaptive_pipeline`; the M5 strip stays inside `run_tiering`. v0.26.0 patch (commit `4f51c52`) added a `cold_archive.content` fallback inside `attempt_one` so Cap C reads the original content even when strip ran first.
- **D direction (synthesis interaction events + M1 consumer + per-query adaptive decision)**: new `EventType::SynthesisInteraction` variant with payload (Viewed / ClickedSource / ImmediateRequery / ExplicitThumb + metadata). M1 consumer `synthesis_feedback` strict 5-invariant pattern mirroring `recompute_concept_refresh_stats`. `decide_synthesize` per-query gate cold-start fallback to global flag at events < 10. `RecallSynthesisOutcome.synthesis_id` (ULID) + `skipped_adaptive_decision` flag. `/api/adaptive` extends with `synthesis: { by_cluster, global }` projection; `rein_feedback` MCP tool accepts `kind: "synthesis_interaction"`. GUI: SynthesisCard hooks (dwell + click + thumb), Adaptive page Synthesis Quality panel. Useful_rate formula 9 bootstrap constants; `by_cluster` hard cap 4096 + query_type whitelist normalize. Records-only on first install; gated on Cap B default-on (v0.25.4) + 2-4 weeks canary traffic before adaptive decision actually kicks in.
- **GUI 12-finding cleanup** from v0.25.1 audit (M1-M4 + L1-L7 + L8 unspecified).
- **Engineering**: 958+ tests / 0 clippy / 8-agent parallel fan-out via implementation contract / 2 Codex audit rounds (R1: 4 P1 + 5 P2 / R2: 0 P1 + 3 P2, HIGH/P1 saturation reached).

## v0.25.0 highlights (2026-04-24)

- **ARS Capability B (recall-time synthesis)**: opt-in `synthesize=true` param on `rein_recall` (MCP) + `?synthesize=true` on `/api/memories` (REST) + `--synthesize` on `rein recall` (CLI). When `[ars].recall_synthesis_enabled = true` (default `false`) AND results.len() ≥ `[ars].recall_synthesis_min_results` (default 3) AND an LLM provider is configured, the LLM produces a 3-6 sentence narrative over the top-N results and returns it as `RecallSynthesisOutcome` alongside the normal results list. `/api/recall_stream` intentionally NOT wired (paginated, would duplicate LLM cost).
- **Prompt-size safety net**: `build_synthesis_prompt` honours `extract::llm::resolve_max_input_for_kind(config, &extractor)` budget. Priority-aware truncation: top-ranked memories preserved whole, first overflow truncated mid-content + `[…remaining memories truncated]` notice + remaining dropped. Query itself capped to `max(max_chars / 4, 256)` chars. Final defensive `take(max_chars)` safety net guarantees prompt length ≤ cap regardless of edge cases.
- **Hallucination guardrail**: synthesis SYSTEM_PROMPT explicitly says "synthesize from the provided memories only; do not invent facts" and notes contradictions when memories disagree.
- **No new MCP tool**: Cap B extends existing `rein_recall`, MCP tool count stays at 34. JSON shape stays bit-identical for callers that don't pass `synthesize=true` (synthesis field absent, not null).
- **Tests**: 12 unit tests in `ops/recall_synthesis.rs` (5 outcome states + 7 prompt-cap edge cases including long-query, huge-query, extreme-tight-cap, no-cap path).
- **4 Codex audit rounds**: Round 1 caught REST + CLI not wired (P2x2 fixed), Round 2 caught unbounded prompt size (P2 fixed), Round 3 caught long-query bypass of size cap (P2 fixed), Round 4 clean.

## v0.24.0 highlights (2026-04-24)

- **ARS Capability A (Concept Living Summary)**: `living_summary` field on Concept nodes refreshed via `should_refresh_living_summary` trigger; new MCP tools `rein_concept_state` + `rein_concept_summary_refresh`; cluster-aware refresh-interval percentiles via `ConceptSummaryRefreshed` feedback events.
- **L4 concurrent CAS protection**: `write_living_summary_if_revision_unchanged` predicates on both `revision` AND `living_summary_source_revision IS prior` so two concurrent first-refreshes can't both commit.
- **Cross-cutting peek+commit refactor**: 5 consumers (M2/M3/A1/concept-summary/etc) migrated to peek-then-commit watermark pattern (Codex hammered this 5 HIGHs across 4 rounds before clean).

## v0.20.0 highlights (2026-04-17)

- **Full-stack audit sweep** (6 Explore agents + codex rescue): 45 findings total (1 CRITICAL + 15 HIGH + 17 MEDIUM + 12 LOW/NIT), 509 tests green.
- **CRITICAL data-integrity fix**: `consolidate_by_ids_atomic` now scrubs `memories.related_ids` and `episodes.memory_ids` before deleting — closes the dangling-ref corruption that silently accumulated on every `rein consolidate` / `rein cleanup --all` run.
- **Concurrency**: `apply_decay` uses a `WHERE last_accessed = ?` CAS predicate so a racing `record_access`/`intelligent_merge` no longer gets clobbered by a stale snapshot. New `pending_grayzone_jobs` SQLite table persists gray-zone dedup pairs inside the store transaction; drained at startup so the post-COMMIT enqueue window is no longer silent data loss. `session_artifacts.episode_id` backfilled on startup to heal stop-hook orphan episodes.
- **Algorithms (principled, data-driven)**: alpha optimizer returns None on zero-variance events (preserves prior instead of biasing to 0.0); CC fusion skips tied channels instead of awarding max to all; survival falls back to Ebbinghaus when `event_count == 0`; MMR handles all-negative relevance scores correctly; expanded queries deduped inter-variant as well as against original; expansion thread is cancellation-aware.
- **Security**: REST + proxy signal handling (Ctrl-C + SIGTERM graceful shutdown), GUI responses carry CSP / X-Frame-Options / nosniff / Referrer-Policy, Gemini endpoint scheme validation + retry with exponential backoff + Retry-After, `/api/memories` read gate aligned with `/api/artifacts`, proxy `token_eq` hashes both sides to close length leak, max_input_chars default 0 resolves to 1M-char cap on large-context models instead of unlimited.
- **Storage**: vec_memories DELETE fatal inside transaction (no more ghost embeddings), symmetric HNSW remove / update mark_dirty, session buffer file capped at 16 MiB, `feedback_events` prune ignores stale consumers.
- **GUI/REST polish**: polling interval is now reactive to Settings slider, Run Fix requires confirmation, Graph relation colors match the real server enum, server version shown dynamically via new `/api/version`, Memories delete errors surfaced in a toast, Brain page memoir exports parallelized via `Promise.all`.
- **Cleanup**: WebSocket mirror `assistant_text` capped at 200 KB matching stream_response; path-traversal guard rejects `..`/`//` segments before routing; `consolidate_atomic` / `consolidate_topics_atomic` gated behind `#[cfg(test)]` to close the latent trap; dedup containment direction preserved so new-contains-old supersedes old instead of merging into it.

## v0.18.0 highlights (2026-04-17)

- **Codex subscription proxy Phase C/D**: `codexsubp` (recommended loopback) + `codexsubpws` (experimental WS-first) entrypoints via `rein init --proxy`. `chatgpt_base_url` set to `http://127.0.0.1:PORT/backend-api` (v0.20.1 fix: removed `/codex` suffix — Codex hard-codes `/codex/` in analytics URL and uses `contains("/backend-api")` for `wham/apps`, so trailing `/codex` causes double-prefix 404s). `/api/codex/*` path family also accepted.
- **Security hardening (28 fixes, 29 regression tests)**: WS deflate-bomb cap (1 MiB), JWT `exp` validation + redact helper, `/api/artifacts` `require_read_token` gate, `expand.rs` prompt-injection defense, rerank strict validation, KM degenerate-curve early-return, HDBSCAN single-point guard, deterministic tiering reservoir, dedup `Option<f32>` API for both-empty sets, adaptive cache TTL.
- **New config**: `config.search.strong_signal_{ratio,single}`, `config.adaptive.cache_ttl_secs`.

## Build & Test

This is a 2-crate Cargo workspace since v0.21 A1: `crates/rein` (main) + `crates/rein-macros` (proc-macro for `#[op]`).

```bash
cargo build -p rein               # Debug build of main crate
cargo test --workspace            # All tests across both crates must pass
cargo build -p rein --release     # Optimized binary (~7MB)
cargo install --path crates/rein  # Install to ~/.cargo/bin/rein
docker build -t rein .            # Docker image (~165MB), build context = workspace root
```

## Directory Structure

```
src/
├── main.rs          # CLI entry point (clap subcommands, 20+ commands)
├── commands.rs      # CLI command handler bodies (extracted from main.rs)
├── lib.rs           # Public API re-exports
├── config.rs        # Configuration loading (TOML + env), includes [extract] section
├── ops/             # Shared business logic (modularized)
│   ├── mod.rs       # Ingestion, GC, topic utilities, re-exports
│   ├── adaptive.rs  # M2-M6 adaptive pipeline, alpha learning, clustering, tiering, cluster profiles for UI
│   ├── dedup.rs     # Dedup strategies, merge, batch dedup (cluster-grouped, ANN fallback for None bucket)
│   └── consolidation.rs # Topic consolidation, cleanup orchestration, summary normalization
├── init.rs          # Auto-configure MCP clients (JSON + TOML)
├── types/           # Memory, Importance, Embedder trait, errors (incl. Extract variant)
├── store/
│   ├── sqlite.rs    # Core CRUD, FTS, vector search, decay, auto_link, organize, recent
│   ├── memoir.rs    # Knowledge graph CRUD, traversal, export
│   ├── knowledge.rs # Knowledge units, evolution, linking, organizing
│   ├── quality.rs   # Self-learning quality scoring, pruning, recall tracking
│   ├── schema.rs    # DDL, migrations, model-change detection
│   ├── migrate.rs   # QMD import, reindex
│   ├── fts.rs       # FTS5 search with sanitization
│   ├── vec.rs       # sqlite-vec operations
│   ├── hnsw.rs      # HNSW approximate nearest neighbor (usearch)
│   ├── tantivy_fts.rs # Tantivy BM25 full-text search (BooleanQuery topic filter)
│   ├── jieba_tokenizer.rs # Custom Tantivy tokenizer: jieba-rs word segmentation + CJK bigrams
│   ├── adaptive.rs  # Feedback event sourcing, AdaptiveState cache, per-consumer offsets
│   ├── hdbscan.rs   # Pure Rust HDBSCAN clustering (dendrogram → condensed tree → EOMBST)
│   └── tiering.rs   # Three-tier memory (Hot/Warm/Cold) with streaming quantile estimator
├── embed/
│   ├── gemini.rs    # Google Gemini embedding API
│   ├── omlx.rs      # OMLX local embedding (OpenAI-compatible)
│   └── cache.rs     # Embedding cache with TTL
├── search/
│   ├── recall.rs    # 3-channel recall pipeline (FTS + Vector + KG) + evidence-aware rerank + RRF/CC fusion + R2 rerank
│   ├── classify.rs  # Query routing (Episodic/Temporal/Preference/ExactKeyword/Semantic/Exploratory)
│   ├── kg_search.rs # KG retrieval: concept FTS + BFS "land and expand" with temporal filtering
│   ├── rerank.rs    # Multi-feature reranker (8 features, learned weights from M1/M2)
│   ├── rrf.rs       # Reciprocal Rank Fusion + Convex Combination
│   ├── scoring.rs   # Ebbinghaus decay + KM survival curve scoring
│   ├── warmup.rs    # Background warmup: embeddings + HNSW/Tantivy rebuild
│   ├── chunker.rs   # Semantic text chunking
│   ├── alpha_optimizer.rs # Counterfactual offline alpha optimization for CC fusion (now includes KG/episode/support/diversity signals)
│   ├── expand.rs    # Query expansion (Gemini Flash Lite / OMLX dual backend) → 2-3 query variants
│   ├── rerank_llm.rs # LLM reranker (Gemini / OMLX) + strong-signal bypass
│   ├── mmr.rs       # Maximal Marginal Relevance re-ranking for result diversity
│   └── survival.rs  # Kaplan-Meier non-parametric survival analysis for adaptive decay
├── extract/
│   ├── llm.rs       # LLM extraction (Gemini + OMLX/Ollama), fallback to patterns
│   ├── postprocess.rs # Rule-based post-processing (date keywords, preference tagging, knowledge update)
│   ├── patterns.rs  # Rule-based keyword scoring (fallback when LLM unavailable)
│   ├── hooks/       # Hook commands + async pipeline
│   │   ├── mod.rs   # Hook orchestration (post, compact, prompt no-op, stop)
│   │   ├── queue.rs # Async memory queue (file-based, flock-protected, crash-safe)
│   │   ├── working_set.rs # Project-scoped memory surfaces (working set + always-on index)
│   │   ├── persist.rs # Memory persistence + working-set updates
│   │   ├── parsing.rs # JSON payload extraction, agent detection
│   │   ├── buffer.rs  # Session buffer I/O
│   │   └── scoring.rs # Signal scoring and filtering
│   ├── dedup.rs     # Similarity (hybrid CJK tokenization: jieba-rs + bigrams, Jaccard + containment, hot-path cluster-aware hints)
│   └── intelligent_merge.rs # LLM-driven semantic verdict classifier (opt-in: ignore/update/merge/create_new for gray-zone cases)
├── sync/
│   ├── supermemory.rs # Supermemory v4 API client
│   ├── auto_memory.rs # ~/.claude/ file scanner
│   └── validate.rs    # Cross-source validation
├── proxy/
│   ├── mod.rs       # Transparent proxy server (record-only, dedicated store thread)
│   ├── provider.rs  # Provider detection (Anthropic / OpenAI, extensible for Gemini)
│   ├── anthropic.rs # Anthropic /v1/messages format handling
│   ├── openai.rs    # OpenAI /v1/chat/completions format handling
│   ├── policy.rs    # Extraction policy decisions
│   └── extract.rs   # Async response extraction + queue integration
└── mcp/
    ├── server.rs    # MCP server (31 tools, stdio + HTTP/SSE + GUI)
    ├── rest.rs      # REST API layer (33 inventory routes + 2 legacy derived)
    ├── tools.rs     # Tool parameter structs
    └── compact.rs   # Output formatters

gui/                 # Neural Wiki GUI (React 18 + TypeScript + Tailwind + Vite)
├── src/
│   ├── App.tsx      # Router + QueryClientProvider
│   ├── api/         # Fetch wrapper, TypeScript types
│   ├── hooks/       # TanStack Query hooks with configurable polling
│   ├── components/  # Layout (icon sidebar + vitals header)
│   └── pages/       # 8 pages: Dashboard, Brain, Memories, Adaptive, Graph, Timeline, Artifacts, Settings
└── vite.config.ts   # Dev proxy + manual chunks for react/charts/graph vendors

Dockerfile           # Multi-stage build (rust:latest → debian:trixie-slim)
docker-compose.yml   # One-command deployment
```

## Key Invariants

- All SQL uses parameterized queries (except vec table DDL which uses usize)
- FTS5 queries sanitized via `sanitize_fts_query()`
- LIKE queries escape `%` and `_`
- HTTP server requires REIN_HTTP_TOKEN for non-localhost bind
- Dedup threshold: per-cluster adaptive via A1 (P90 intra-cluster similarity); 0.70 global fallback. All paths (store, batch, vec dedup, CLI, MCP) use `get_dedup_threshold(cluster_id)` via `ops::effective_dedup_threshold()`
- Intelligent merge (opt-in via `[intelligent_merge] enabled = true`): LLM pre-flight classification of gray-zone (0.50–0.85 sim) pairs chooses ignore/update/merge/create_new. Pre-flight runs OUTSIDE `BEGIN IMMEDIATE` to avoid holding the write lock. Every verdict is logged to `dedup_decisions` with `operator='llm_verdict'`; Update/Merge snapshot the pre-merge existing memory into `memory_evidence` so the prior version is recoverable.
- Single-memory deletion (`SqliteStore::delete`, `rein_forget`, CLI delete) wraps row delete + JSON-array ref cleanup (`concept.source_memory_ids`, `memory.related_ids`, `episodes.memory_ids`) in one `BEGIN IMMEDIATE` so partial failure rolls back atomically
- CJK lexical dedup uses `jieba-rs` word segmentation plus character bigrams
- Vector dimensions: configurable (default 3072)
- FTS5 tokenizer: unicode61 (CJK support)
- Per-request connection model: each MCP request opens its own `SqliteStore` with `SQLITE_OPEN_FULL_MUTEX`
- `store_with_dedup` uses `BEGIN IMMEDIATE` to prevent concurrent dedup races
- `store_with_dedup` may infer cluster hints from local embedding cache, but must never trigger remote embedding calls on cache miss
- HNSW and Tantivy side indexes are updated on every store/update/delete (fire-and-forget)
- Warmup always rebuilds HNSW and Tantivy indexes before processing new embeddings
- LLM extraction falls back to rule-based patterns when provider is unavailable
- Tantivy writer lock failure → graceful skip (side index, not critical)
- Tantivy/HNSW rebuild uses flock to prevent concurrent corruption
- Proxy is record-only (no request modification); extraction via async queue
- Tantivy topic filtering uses BooleanQuery at index level (not post-filter)
- Auto-link creates bidirectional related_ids on store
- `max_input_chars=0` only allowed for known 1M-token Gemini models (safety fallback to 16K)
- Beta values read from `MemoryLayer::beta()`, not hardcoded
- Codex hooks use official stdout JSON only: SessionStart/UserPromptSubmit context injection is opt-in via `[hooks.codex]`, PreToolUse/PermissionRequest guardrails are deny-only, and diagnostic logs must go to stderr.
- Post-store processing (auto_link, evolve) only runs on newly created memories, not merges
- postprocess enriches keywords only — caller-supplied topic/importance are authoritative
- `/api/memories/:id` is backward-compatible: top-level memory fields remain, plus nested `memory` and `evidence`
- Recall is canonical-first: default result objects are canonicals, while evidence is previewed in recall and expanded on demand
- STM→LTM promotion uses survival-curve-derived thresholds when cluster curves exist, with a fixed fallback otherwise
- Large `cluster_id=None` dedup buckets use ANN candidate generation before pairwise comparison
- Summary display is layered: canonical summaries may be longer, while APIs/UI can expose `summary_short` for list views
- Adaptive status now exposes cluster-level dedup/admission/promotion decisions for the GUI
- Adaptive status now exposes `cluster_profiles` for per-cluster dedup/admission/promotion inspection
- M2 per-cluster alpha keys use format `"<query_type>:<cluster_id>"` (e.g. `"semantic:5"`); these are cleared on recluster via `retain(|k, _| !k.contains(':'))`
- M3 global prior survival curve stored as `"survival_curve:global"` in adaptive snapshot; `recall.rs` must skip `parse::<u32>()` for this key and branch on string `"global"` first
- AdaptiveState `save_snapshot` uses CAS retry (max 3 attempts): read-merge-write with version predicate; fails with `ReinError::Config` after exhaustion — do not swallow this error

## Common Pitfalls

- Don't add async to MemoryStore trait methods (they're intentionally sync)
- Don't use reqwest::blocking inside tokio — use tokio::task::block_in_place
- String slicing: always use .chars() for CJK-safe truncation, never byte indexing
- DOT export: use escape_dot() for all user-provided strings
- Cross-memoir links are forbidden (validated in add_link)
- LLM JSON output may be wrapped in code fences — use `strip_code_fences()` before parsing
- Some local models don't support `response_format: {"type": "json_object"}`
- BFS traversal must filter expired links (valid_from/valid_until)
- KG search seeds by concept ID, not name (avoids cross-memoir collision)

## Environment Variables

GEMINI_API_KEY, SUPERMEMORY_CC_API_KEY, REIN_HTTP_TOKEN, REIN_DB, REIN_LOG, REIN_SSE_BIND, REIN_SSE_PORT, REIN_PROXY_BIND, REIN_PROXY_PORT, REIN_PROXY_TOKEN, REIN_PROXY_ACTIVE
