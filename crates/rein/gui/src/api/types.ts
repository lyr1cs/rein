export interface Memory {
  id: string;
  layer: string;
  topic: string;
  summary: string;
  summary_short?: string;
  content: string;
  keywords: string[];
  importance: string;
  source: string;
  strength: number;
  tier: 'hot' | 'warm' | 'cold';
  cluster_id: number | null;
  access_count: number;
  canonical_id: string | null;
  support_count: number;
  merge_count: number;
  dedup_confidence: number;
  source_diversity: number;
  contradiction_score: number;
  status: string;
  related_ids: string[];
  concept_ids: string[];
  superseded_by: string | null;
  created_at: string;
  updated_at: string;
  last_accessed: string;
}

export interface RecallResult extends Memory {
  score: number;
  confidence: number;
  sources_hit: number;
  evidence_count: number;
  evidence_preview: string[];
  /**
   * v0.26 Cap C — archival summary surfaced when the memory is in Cold tier
   * AND `[ars].cold_archive_enabled = true` AND the row has a non-NULL
   * `archival_summary` at the current `ARCHIVAL_SUMMARY_VERSION`. `undefined`
   * for Hot/Warm or when the feature is off — UI/MCP must continue to render
   * `memory.content` in that case. Server omits the field via
   * `skip_serializing_if`, so the wire JSON stays compact for older clients.
   */
  archival_summary?: string;
}

export interface RecallPageResponse {
  results: RecallResult[];
  count: number;
  offset: number;
  limit: number;
  next_offset: number | null;
  has_more: boolean;
}

/**
 * A single inline citation extracted from synthesized prose (v0.25.2).
 *
 * The Rust backend emits **char offsets** (not byte offsets) so the JS
 * frontend can slice the prose with `Array.from(s)` / spread-then-join
 * without round-tripping through TextEncoder or worrying about UTF-16
 * surrogate pairs. CJK content is the canonical case where byte vs char
 * would silently desync the two stacks.
 */
export interface Citation {
  /** 1-based rank of the source memory in the results list. */
  rank: number;
  /** Char offset in the cleaned prose where this cited claim ends. */
  span_end: number;
}

/**
 * Outcome of an opt-in recall-time synthesis pass (v0.25 ARS Capability B).
 *
 * The server attaches this on `RecallMemoryOutput` whenever the request
 * carried `synthesize=true`. The optional `skipped_*` flags tell the UI why
 * no narrative was produced — the synthesis pipeline emits exactly one
 * branch (synthesis text OR a single skipped flag) so the GUI can decide
 * between rendering the prose, a muted reason note, or a generic fallback.
 *
 * `citations` (v0.25.2 ARS Cap B inline-citation feature) is a list of
 * `[#k]` markers parsed out of the LLM's output; markers are stripped
 * from `synthesis` and surfaced here as structured spans the UI can
 * render as click-to-scroll badges. Empty when the LLM emitted no
 * markers (older models / non-compliant outputs) or synthesis was
 * skipped — older backends without the feature simply omit the field.
 */
export interface RecallSynthesisOutcome {
  synthesis?: string;
  query: string;
  source_count: number;
  model_used?: string;
  skipped_disabled?: boolean;
  skipped_no_llm?: boolean;
  skipped_too_few_results?: boolean;
  /**
   * v0.26 D direction — set when the per-query adaptive decision (driven by
   * the synthesis_feedback consumer's `useful_rate` + cold-start fallback)
   * voted to skip synthesis for this cluster/query_type pair, *distinct from*
   * the operator-level `skipped_disabled` (which means `[ars]` config is
   * off entirely). Server may temporarily reuse `skipped_disabled` per
   * v0.26 contract §3.3 — UI tolerates either branch.
   */
  skipped_adaptive_decision?: boolean;
  citations?: Citation[];
  /**
   * v0.26 D direction — ULID stamped on each successful synthesis output
   * (only set when prose is non-empty, never on skipped paths). Clients
   * echo this in `SynthesisInteraction` feedback events posted to
   * `/api/feedback`. When `synthesis_id` is `undefined`, callers MUST NOT
   * post interaction events (per §8 invariant 9).
   */
  synthesis_id?: string;
  /**
   * v0.26.2 hotfix — capitalised query-type label
   * (`Episodic`/`Temporal`/`Preference`/`ExactKeyword`/`Semantic`/`Exploratory`)
   * the gate used when consulting the per-cluster synthesis bucket.
   * Surfaced so the GUI can echo it back through SynthesisInteraction
   * `metadata.query_type` and keep the M1 consumer's bucket key in
   * lockstep with the gate's lookup. Pre-v0.26.2 this round-trip was
   * missing, so every GUI feedback event landed in the consumer's
   * `(-1, "unknown")` bucket while the gate read from the real per-cluster
   * bucket — making the per-query adaptive gate dead code on GUI traffic.
   * Older backends omit the field; treat as missing (do NOT invent a
   * client-side default).
   */
  query_type?: string;
  /**
   * v0.26.2 hotfix — dominant cluster id for this recall (top-ranked
   * result's `cluster_id`, matching the source `decide_synthesize` reads).
   * Surfaced for the same metadata round-trip rationale as `query_type`
   * above. `undefined` when the result set is empty or the top result
   * carries no cluster assignment, which matches the cold-start gate
   * fallback. Older backends omit the field.
   */
  cluster_id?: number;
}

/**
 * Shape of `GET /api/memories?...` once Cap B lands. The legacy 24.x
 * response (`{ results, count }`) is a structural subset — the new
 * `synthesis` / `request_id` fields are optional so older backends still
 * type-check. (`route` was documented but never emitted by the legacy REST
 * handler at `mcp/rest.rs:965-1029`; only the inventory `recall` op
 * populates it on CLI/MCP — kept off the type to avoid doc drift.)
 */
export interface RecallMemoryOutput {
  results: RecallResult[];
  count?: number;
  request_id?: string;
  synthesis?: RecallSynthesisOutcome;
}

export interface MemoryEvidence {
  id: string;
  canonical_id: string;
  memory_id: string | null;
  source_topic: string;
  summary: string;
  content: string;
  keywords: string[];
  source: string;
  created_at: string;
  imported_at: string;
}

export interface MemoryDetailResponse {
  memory: Memory;
  // Preview-capped at 200 rows by the server so we don't ship megabyte
  // payloads for canonicals with huge evidence histories. Use
  // `evidence_total` for honest count labels — `evidence.length` is the
  // preview size, not the true total. `evidence_total` is optional so the
  // legacy servers (pre-v0.25.1) that only return `evidence` still type-check;
  // call sites should fall back to `evidence.length`.
  evidence: MemoryEvidence[];
  evidence_total?: number;
}

export interface StoreStats {
  total_memories: number;
  ltm_count: number;
  stm_count: number;
  topic_count: number;
  avg_strength: number;
  memoir_count: number;
  concept_count: number;
  link_count: number;
  hot_count: number;
  warm_count: number;
  cold_count: number;
}

/**
 * v0.26 D direction — synthesis interaction projection embedded inside
 * `AdaptiveStatus.synthesis`. The shape mirrors the JSON the server emits
 * from `synthesis_feedback_stats` (per `compute_useful_rate` +
 * `recompute_synthesis_feedback_stats`).
 *
 * `by_cluster` is a flat list keyed by `(cluster_id, query_type)` — the
 * server projects the HashMap into a deterministic-order array. `global`
 * is the rolled-up across-all-clusters scalar; `null` while the consumer
 * has yet to absorb any events (cold-start state — UI shows the "awaiting
 * traffic" hint when this branch hits).
 */
export interface AdaptiveStatusSynthesisCluster {
  cluster_id: number;
  query_type: string;
  useful_rate: number;
  viewed_count: number;
  viewed_dwell_p50_ms: number | null;
  clicked_source_rate: number;
  immediate_requery_rate: number;
  explicit_thumb_up_rate: number;
  /**
   * v0.27.1 E direction — runtime LLM judge counters per bucket. Older
   * backends (pre-v0.27.1) won't emit these keys; UI treats `undefined` as
   * cold-start ("no judge events folded yet"). `llm_judge_hit_rate` is
   * `null` when `llm_judge_count = 0` (zero division) — UI renders "n/a"
   * for that case rather than "0%".
   */
  llm_judge_count?: number;
  llm_judge_hit_count?: number;
  llm_judge_hit_rate?: number | null;
}

export interface AdaptiveStatusSynthesisGlobal {
  useful_rate: number | null;
  total_events: number;
  last_consumed_event_id: number;
}

export interface AdaptiveStatusSynthesis {
  by_cluster: AdaptiveStatusSynthesisCluster[];
  global: AdaptiveStatusSynthesisGlobal | null;
}

export interface AdaptiveStatus {
  learned_alphas: Record<string, { value: number; sample_count: number; last_updated: string }>;
  reranker_weights: Record<string, number>;
  cluster_info: { cluster_version: number; unique_clusters: number; assigned_memories: number };
  tier_boundaries: { hot_threshold: number; cold_threshold: number };
  event_counts: Record<string, number>;
  survival_curves: Array<{ cluster_id: string; median_survival: number | null; steps?: number[][] }>;
  dedup_thresholds: {
    /** Legacy wire field; now carries unlabeled per-cluster shadow suggestions. */
    per_cluster: Record<string, number>;
    /** Legacy wire field; now carries the unlabeled global shadow suggestion. */
    global: number;
    dedup_threshold_static?: number;
    dedup_threshold_shadow?: number;
    dedup_threshold_hard_effective?: number;
    source?: string;
    hard_effective_source?: string;
    calibration?: {
      adaptive_enabled: boolean;
      evidence_verified: boolean;
      applied: boolean;
      reason: string;
      counterfactual_counts?: {
        total_events: number;
        evaluable_events: number;
        would_merge_at_probed_shadow: number;
        would_merge_at_hard_effective: number;
      };
    };
    repair_advice?: string[];
  };
  cluster_profiles: Array<{
    cluster_id: number;
    memory_count: number;
    avg_strength: number;
    dedup_threshold: number;
    dedup_threshold_shadow?: number;
    dedup_threshold_hard_effective?: number;
    admission_threshold: number;
    promotion_threshold: number;
    median_survival: number | null;
  }>;
  /**
   * v0.26 D direction — synthesis-quality observability projection. `undefined`
   * on legacy backends that haven't shipped Cap C yet so callers can
   * type-check against either; the Adaptive page renders the awaiting-traffic
   * banner when this is undefined OR when `by_cluster` is empty AND `global`
   * is null.
   */
  synthesis?: AdaptiveStatusSynthesis;
}

/* ── Feedback payloads (v0.26 D direction) ──────────────────────────── */

/**
 * v0.26 D direction — typed body for `synthesis_interaction` POST. The
 * `kind` discriminator is serialized lower-case (`viewed`, `clicked_source`,
 * `immediate_requery`, `explicit_thumb`) to match the Rust
 * `SynthesisInteractionKind` `#[serde(rename_all = "snake_case", tag = "kind")]`.
 *
 * `source_index` is **1-based** to match the `[#k]` UI marker convention
 * (extract_citations rank, recall_synthesis.rs:60). Out-of-range indices are
 * accepted by the consumer and silently dropped.
 */
export type SynthesisInteractionKind =
  | { kind: 'viewed'; dwell_ms: number }
  | { kind: 'clicked_source'; source_index: number }
  | { kind: 'immediate_requery'; gap_ms: number }
  | { kind: 'explicit_thumb'; up: boolean };

/**
 * Optional per-event diagnostics. Server stores verbatim; consumer keys
 * stats by `(cluster_id, query_type)`. Fields C_RECALL/B_REST_MCP haven't
 * surfaced in v0.26.0 stay `undefined` on the wire and the consumer routes
 * them to the global bucket — keep this best-effort, do not invent
 * client-side guesses.
 */
export interface SynthesisMetadata {
  query_type?: string;
  cluster_id?: number;
  source_count?: number;
  synthesis_chars?: number;
}

/* ── Concept summary feedback (v0.27 ARS Cap A mirror of D-direction) ── */

/**
 * v0.27 — interaction kinds emitted by `ConceptSummaryCard` (Cap A mirror of
 * v0.26 D-direction synthesis instrumentation). Variant set + serde shape
 * (`#[serde(rename_all = "snake_case", tag = "kind")]`) match the backend's
 * `ConceptSummaryInteractionKind`.
 *
 * `source_index` is **1-based** to match any inline `[#k]` rendering convention
 * the GUI may add later; out-of-range indices are accepted by the consumer
 * and silently dropped (parity with `SynthesisInteractionKind`).
 */
export type ConceptSummaryInteractionKind =
  | { kind: 'viewed'; dwell_ms: number }
  | { kind: 'clicked_source'; source_index: number }
  | { kind: 'immediate_requery'; gap_ms: number }
  | { kind: 'explicit_thumb'; up: boolean };

/**
 * Optional per-event diagnostics carried verbatim to the consumer. Field
 * names mirror the backend `ConceptSummaryMetadata` struct; missing fields
 * route to fallback buckets server-side.
 */
export interface ConceptSummaryMetadata {
  query_type?: string;
  cluster_id?: number;
  concept_chars?: number;
  revision_version?: number;
}

/**
 * Wire body for `POST /api/feedback/concept-summary`. Unlike the v0.26
 * `/api/feedback` route which uses a `kind`-discriminated union, the
 * concept-summary endpoint accepts the bare event shape.
 *
 * `recall_id` is a per-view correlation id minted by the GUI when a
 * concept is selected — for Brain.tsx this is a `crypto.randomUUID()`
 * generated on selection rather than echoed from a recall response (Cap A
 * is not a recall surface). The backend treats it opaquely.
 */
export interface ConceptSummaryInteractionEvent {
  concept_id: string;
  recall_id: string;
  /**
   * v0.27.3 — per-refresh summary identity. Backends may expose this as
   * `concept_summary_id` directly or as `living_summary_id` on concept-state;
   * callers send whichever value is available so judge feedback can join to
   * the exact summary instance the user saw.
   */
  concept_summary_id?: string;
  living_summary_id?: string;
  interaction: ConceptSummaryInteractionKind;
  metadata?: ConceptSummaryMetadata;
}

/**
 * Discriminated union — `kind: 'access'` mirrors the legacy
 * `{memory_ids, request_id, query, helpful}` body so existing call sites
 * keep working through the back-compat shim on the server. The server
 * accepts both old (no `kind`) and new shapes; this typed view always
 * carries the discriminator so future TypeScript callers stay explicit.
 */
export type FeedbackPayload =
  | {
      kind: 'access';
      memory_ids: string[];
      request_id?: string;
      query?: string;
      helpful?: boolean;
    }
  | {
      kind: 'synthesis_interaction';
      synthesis_id: string;
      recall_id: string;
      interaction: SynthesisInteractionKind;
      metadata?: SynthesisMetadata;
    };

export interface DoctorCheck {
  name: string;
  category: 'configuration' | 'runtime' | 'storage' | 'index' | 'queue' | 'network';
  severity: 'info' | 'warning' | 'error';
  status: 'ok' | 'warn' | 'fail';
  fixable: boolean;
  message: string;
  repair_hint?: string;
}

export interface DoctorReport {
  status: 'healthy' | 'degraded' | 'unhealthy';
  checks: DoctorCheck[];
  fixes_applied?: string[];
}

export interface Episode {
  id: string;
  title: string;
  outcome: string;
  decisions: string[];
  primary_topics: string[];
  tags: string[];
  involved_agents: string[];
  important_paths: string[];
  temporal_keywords: string[];
  source_session_id: string | null;
  concept_ids: string[];
  memory_ids: string[];
  created_at: string;
}

export interface Artifact {
  id: string;
  artifact_kind: string;
  title: string | null;
  summary: string | null;
  source_agent: string | null;
  source_label: string | null;
  turn_count: number;
  episode_id: string | null;
  created_at: string;
}

export interface Concept {
  id: string;
  memoir_id: string;
  name: string;
  definition: string;
  labels: string[];
  source_memory_ids: string[];
  confidence: number;
  revision: number;
  last_episode_id: string | null;
  // v0.24 ARS Capability A. Mirrored from the Rust `Concept` for completeness
  // — Graph.tsx fetches the live snapshot via `getConceptState` separately,
  // but external clients hitting `/api/memoirs/{name}/export` get these
  // fields and may want to consume them without a second round trip.
  living_summary?: string | null;
  living_summary_updated_at?: string | null;
  living_summary_source_revision?: number | null;
  created_at: string;
  updated_at: string;
}

export interface ConceptLink {
  id: string;
  source_id: string;
  target_id: string;
  relation: string;
  weight: number;
  created_at: string;
  valid_from: string | null;
  valid_until: string | null;
}

/**
 * Memoir summary returned from `GET /api/memoirs`. Mirrors the server
 * `MemoirSummary` row shape — kept in `types.ts` (not inline in Brain.tsx /
 * Graph.tsx) so the new react-query hooks can share a single declaration.
 */
export interface Memoir {
  id: string;
  name: string;
  description: string;
}

/**
 * Full export of a memoir's concepts + concept-links via
 * `GET /api/memoirs/{name}/export?format=json`. Used by Brain.tsx (bounded
 * fan-out via `useQueries`) and Graph.tsx (single-memoir view).
 */
export interface MemoirExport {
  memoir: Memoir;
  concepts: Concept[];
  links: ConceptLink[];
}

/**
 * Snapshot of a concept's current state including the auto-refreshed
 * `living_summary` (v0.24 ARS Capability A). Fetched from
 * `GET /api/concepts/{id}/state`.
 *
 * `living_summary` and its companion fields are nullable — null means the
 * living summary has not been generated yet, in which case the summary
 * section should be hidden entirely (no placeholder).
 */
export interface ConceptState {
  id: string;
  memoir_id: string;
  name: string;
  definition: string;
  revision: number;
  last_episode_id: string | null;
  living_summary: string | null;
  living_summary_id?: string | null;
  concept_summary_id?: string | null;
  living_summary_updated_at: string | null;
  living_summary_source_revision: number | null;
  created_at: string;
  updated_at: string;
  /** v0.27 R2 P2: true when the Cap A adaptive gate suppressed `living_summary`. */
  living_summary_suppressed?: boolean;
  /** v0.27 R5 P2: representative cluster_id (mode of source memories' cluster_ids).
   * Used by the GUI to route concept-summary feedback events into the right
   * adaptive bucket. May be null when the concept's source memories aren't
   * clustered. */
  cluster_id?: number | null;
}

/* ── Page-local response shapes (relocated from individual pages) ───── */

/**
 * Detail row returned by `GET /api/artifacts/{id}?include_transcript=true`.
 * Extends the base `Artifact` summary with the full transcript text.
 */
export interface ArtifactDetail extends Artifact {
  transcript_text: string | null;
  transcript_available?: boolean;
}

/**
 * Single dedup verdict row returned by `GET /api/dedup_decisions`.
 */
export interface DedupDecision {
  id: string;
  winner_id: string | null;
  loser_id: string | null;
  canonical_id: string | null;
  lexical_score: number | null;
  embedding_score: number | null;
  relation: string;
  confidence: number;
  reason: string;
  operator: string;
  reversible: boolean;
  merged_summary: string | null;
  novel_facts: string;
  conflict_detected: boolean;
  created_at: string;
}

export interface DedupResponse {
  decisions: DedupDecision[];
}

/**
 * Counters from `GET /api/intelligent_merge_metrics` summarising recent
 * intelligent-merge attempts.
 */
export interface MergeMetrics {
  attempted: number;
  success: number;
  parse_errors: number;
  http_errors: number;
  stale_races: number;
}

/**
 * One row of the unified episode/memory feed returned by
 * `GET /api/timeline?from=...&to=...`.
 */
export interface TimelineEvent {
  type: 'episode' | 'memory';
  created_at: string;
  // Episode fields (when type=episode)
  id?: string;
  title?: string;
  outcome?: string;
  decisions?: string[];
  // Memory fields (when type=memory)
  summary?: string;
  topic?: string;
  tier?: 'hot' | 'warm' | 'cold';
  strength?: number;
}

export interface TimelineResponse {
  events: TimelineEvent[];
}

/**
 * One parsed conversational turn from an artifact transcript. Roles are
 * normalised to `user` / `assistant` / `transcript` (the catch-all when the
 * transcript text contains no role prefixes).
 */
export interface Turn {
  role: string;
  text: string;
}
