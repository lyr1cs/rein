export interface Memory {
  id: string;
  layer: string;
  topic: string;
  summary: string;
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
}

export interface AdaptiveStatus {
  learned_alphas: Record<string, { value: number; sample_count: number; last_updated: string }>;
  reranker_weights: Record<string, number>;
  cluster_info: { cluster_version: number; unique_clusters: number; assigned_memories: number };
  tier_boundaries: { hot_threshold: number; cold_threshold: number };
  event_counts: Record<string, number>;
  survival_curves: Array<{ cluster_id: string; median_survival: number | null; steps?: number[][] }>;
  dedup_thresholds: { per_cluster: Record<string, number>; global: number };
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
