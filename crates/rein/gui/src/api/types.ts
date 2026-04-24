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
}

export interface RecallPageResponse {
  results: RecallResult[];
  count: number;
  offset: number;
  limit: number;
  next_offset: number | null;
  has_more: boolean;
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
  evidence: MemoryEvidence[];
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

export interface AdaptiveStatus {
  learned_alphas: Record<string, { value: number; sample_count: number; last_updated: string }>;
  reranker_weights: Record<string, number>;
  cluster_info: { cluster_version: number; unique_clusters: number; assigned_memories: number };
  tier_boundaries: { hot_threshold: number; cold_threshold: number };
  event_counts: Record<string, number>;
  survival_curves: Array<{ cluster_id: string; median_survival: number | null; steps?: number[][] }>;
  dedup_thresholds: { per_cluster: Record<string, number>; global: number };
  cluster_profiles: Array<{
    cluster_id: number;
    memory_count: number;
    avg_strength: number;
    dedup_threshold: number;
    admission_threshold: number;
    promotion_threshold: number;
    median_survival: number | null;
  }>;
}

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
  living_summary_updated_at: string | null;
  living_summary_source_revision: number | null;
  created_at: string;
  updated_at: string;
}
