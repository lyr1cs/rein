use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

pub use crate::store::tiering::MemoryTier;

/// A memory record with all metadata for decay, search, and cross-validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: String,
    pub layer: MemoryLayer,
    pub topic: String,
    pub summary: String,
    pub content: String,
    pub keywords: Vec<String>,
    pub importance: Importance,
    pub source: Source,
    pub strength: f64,
    pub decay_lambda: f64,
    pub access_count: u32,
    pub superseded_by: Option<String>,
    #[serde(default)]
    pub canonical_id: Option<String>,
    #[serde(default = "default_support_count")]
    pub support_count: u32,
    #[serde(default)]
    pub merge_count: u32,
    #[serde(default = "default_dedup_confidence")]
    pub dedup_confidence: f32,
    #[serde(default = "default_source_diversity")]
    pub source_diversity: f32,
    #[serde(default)]
    pub contradiction_score: f32,
    pub related_ids: Vec<String>,
    /// IDs of concepts derived from this memory (Memory → Concept link)
    #[serde(default)]
    pub concept_ids: Vec<String>,
    #[serde(default)]
    pub status: MemoryStatus,
    #[serde(skip)]
    pub embedding: Option<Vec<f32>>,
    /// Memory tier: hot, warm, cold (M5 adaptive tiering)
    #[serde(default)]
    pub tier: MemoryTier,
    /// Cluster assignment from HDBSCAN (M4)
    #[serde(default)]
    pub cluster_id: Option<u32>,
    /// v0.26 Cap C: LLM-generated condensed summary written by the
    /// cold-tier archival worker. `None` until the worker fires for this
    /// memory at least once (see `run_cold_archive_summary`). Worker-only
    /// write surface; recall reads it through the standard SELECT path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub archival_summary: Option<String>,
    /// v0.26 Cap C: Unix epoch seconds when `archival_summary` was last
    /// regenerated. `None` iff `archival_summary` is `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub archival_summary_at: Option<i64>,
    /// v0.26 Cap C: snapshot of `ARCHIVAL_SUMMARY_VERSION` at the time
    /// `archival_summary` was generated. Used to invalidate stale summaries
    /// when the prompt or contract bumps. `None` iff `archival_summary` is
    /// `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub archival_summary_version: Option<u32>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub last_accessed: DateTime<Utc>,
}

fn default_support_count() -> u32 {
    1
}

fn default_dedup_confidence() -> f32 {
    1.0
}

fn default_source_diversity() -> f32 {
    1.0
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DedupRelation {
    Duplicate,
    Update,
    Related,
    #[default]
    Distinct,
}

impl DedupRelation {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Duplicate => "duplicate",
            Self::Update => "update",
            Self::Related => "related",
            Self::Distinct => "distinct",
        }
    }
}

impl fmt::Display for DedupRelation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl FromStr for DedupRelation {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "duplicate" => Ok(Self::Duplicate),
            "update" => Ok(Self::Update),
            "related" => Ok(Self::Related),
            "distinct" => Ok(Self::Distinct),
            other => Err(format!("unknown dedup relation: {other}")),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEvidence {
    pub id: String,
    pub canonical_id: String,
    pub memory_id: Option<String>,
    pub source_topic: String,
    pub summary: String,
    pub content: String,
    pub keywords: Vec<String>,
    pub source: Source,
    pub created_at: DateTime<Utc>,
    pub imported_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DedupDecision {
    pub id: String,
    pub winner_id: Option<String>,
    pub loser_id: Option<String>,
    pub canonical_id: Option<String>,
    pub lexical_score: Option<f32>,
    pub embedding_score: Option<f32>,
    pub relation: DedupRelation,
    pub confidence: f32,
    pub reason: String,
    pub operator: String,
    pub reversible: bool,
    pub merged_summary: Option<String>,
    #[serde(default)]
    pub novel_facts: Vec<String>,
    #[serde(default)]
    pub conflict_detected: bool,
    #[serde(default)]
    pub payload: Option<serde_json::Value>,
    pub created_at: DateTime<Utc>,
}

/// Lifecycle status of a memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum MemoryStatus {
    #[default]
    Active,
    Updated,
    Deprecated,
}

impl fmt::Display for MemoryStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Active => write!(f, "active"),
            Self::Updated => write!(f, "updated"),
            Self::Deprecated => write!(f, "deprecated"),
        }
    }
}

impl FromStr for MemoryStatus {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "active" => Ok(Self::Active),
            "updated" => Ok(Self::Updated),
            "deprecated" => Ok(Self::Deprecated),
            other => Err(format!("unknown MemoryStatus: '{other}'")),
        }
    }
}

/// Dual-layer memory system inspired by Ebbinghaus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemoryLayer {
    LTM,
    STM,
}

impl MemoryLayer {
    pub fn as_str(&self) -> &'static str {
        match self {
            MemoryLayer::LTM => "ltm",
            MemoryLayer::STM => "stm",
        }
    }

    /// Decay shape parameter from Ebbinghaus model.
    /// LTM: β=0.8 (sub-linear, gradual decay, half-life ~11 days)
    /// STM: β=1.2 (super-linear, rapid decay, half-life ~5 days)
    pub fn beta(&self) -> f64 {
        match self {
            MemoryLayer::LTM => 0.8,
            MemoryLayer::STM => 1.2,
        }
    }
}

impl fmt::Display for MemoryLayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl FromStr for MemoryLayer {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "ltm" => Ok(MemoryLayer::LTM),
            "stm" => Ok(MemoryLayer::STM),
            _ => Err(format!("unknown memory layer: {s}")),
        }
    }
}

/// Importance level controlling decay rate and auto-layer assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Importance {
    Low = 0,
    Medium = 1,
    High = 2,
    Critical = 3,
}

impl Importance {
    pub fn as_str(&self) -> &'static str {
        match self {
            Importance::Low => "low",
            Importance::Medium => "medium",
            Importance::High => "high",
            Importance::Critical => "critical",
        }
    }

    /// Multiplier applied to the base decay lambda.
    pub fn decay_factor(&self) -> f64 {
        match self {
            Importance::Low => 1.0,
            Importance::Medium => 0.7,
            Importance::High => 0.4,
            Importance::Critical => 0.1,
        }
    }

    /// Memories with importance >= High are automatically assigned to LTM.
    pub fn auto_layer(&self) -> MemoryLayer {
        if *self >= Importance::High {
            MemoryLayer::LTM
        } else {
            MemoryLayer::STM
        }
    }
}

impl fmt::Display for Importance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl FromStr for Importance {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "low" => Ok(Importance::Low),
            "medium" => Ok(Importance::Medium),
            "high" => Ok(Importance::High),
            "critical" => Ok(Importance::Critical),
            _ => Err(format!("unknown importance: {s}")),
        }
    }
}

/// How the memory was created.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Source {
    Manual,
    Hook,
    Migration,
    Supermemory,
    Proxy,
}

impl Source {
    pub fn as_str(&self) -> &'static str {
        match self {
            Source::Manual => "manual",
            Source::Hook => "hook",
            Source::Migration => "migration",
            Source::Supermemory => "supermemory",
            Source::Proxy => "proxy",
        }
    }
}

impl fmt::Display for Source {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl FromStr for Source {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "manual" => Ok(Source::Manual),
            "hook" => Ok(Source::Hook),
            "migration" => Ok(Source::Migration),
            "supermemory" => Ok(Source::Supermemory),
            "proxy" => Ok(Source::Proxy),
            _ => Err(format!("unknown source: {s}")),
        }
    }
}

/// A named knowledge container (like a structured topic).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memoir {
    pub id: String,
    pub name: String,
    pub description: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// A knowledge node within a memoir.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Concept {
    pub id: String,
    pub memoir_id: String,
    pub name: String,
    pub definition: String,
    pub labels: Vec<String>,
    /// IDs of memories that contributed to this concept (Concept → Memory link)
    #[serde(default)]
    pub source_memory_ids: Vec<String>,
    pub confidence: f32,
    pub revision: u32,
    /// Last episode that modified this concept.
    #[serde(default)]
    pub last_episode_id: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    /// v0.24 ARS: rolling 3-5 sentence "current state" synthesis generated
    /// by LLM from recent revisions. `None` until the refresh trigger fires
    /// for the first time (see `should_refresh_living_summary`).
    #[serde(default)]
    pub living_summary: Option<String>,
    /// v0.24 ARS: when `living_summary` was last generated. `None` iff
    /// `living_summary` is `None`.
    #[serde(default)]
    pub living_summary_updated_at: Option<DateTime<Utc>>,
    /// v0.24 ARS: the `revision` value at the time `living_summary` was
    /// last generated. Used to compute revisions-since-last-summary in
    /// the refresh-decision gate.
    #[serde(default)]
    pub living_summary_source_revision: Option<u32>,
}

/// A typed relation between two concepts, with optional temporal validity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptLink {
    pub id: String,
    pub source_id: String,
    pub target_id: String,
    pub relation: Relation,
    pub weight: f32,
    pub created_at: DateTime<Utc>,
    /// When this relationship became valid (None = since creation).
    #[serde(default)]
    pub valid_from: Option<DateTime<Utc>>,
    /// When this relationship was invalidated (None = still active).
    #[serde(default)]
    pub valid_until: Option<DateTime<Utc>>,
}

/// A historical snapshot of a concept's state at a given revision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptRevision {
    pub id: String,
    pub concept_id: String,
    pub revision: u32,
    pub definition: String,
    pub confidence: f32,
    pub labels: Vec<String>,
    pub source_memory_ids: Vec<String>,
    pub episode_id: Option<String>,
    pub created_at: DateTime<Utc>,
}

/// A session episode node in the temporal knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    pub id: String,
    pub title: String,
    pub outcome: String,
    pub decisions: Vec<String>,
    #[serde(default)]
    pub primary_topics: Vec<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub involved_agents: Vec<String>,
    #[serde(default)]
    pub important_paths: Vec<String>,
    #[serde(default)]
    pub temporal_keywords: Vec<String>,
    #[serde(default)]
    pub source_session_id: Option<String>,
    /// Concept IDs touched in this session.
    pub concept_ids: Vec<String>,
    /// Memory IDs created in this session.
    pub memory_ids: Vec<String>,
    pub created_at: DateTime<Utc>,
}

/// A single turn in a structured session transcript.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionTurn {
    pub role: String,
    pub content: String,
}

/// Structured ingestion payload for a session/transcript.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionIngest {
    #[serde(default = "default_session_schema_version")]
    pub schema_version: u32,
    #[serde(default = "default_session_artifact_kind")]
    pub artifact_kind: String,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub started_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub ended_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub summary: Option<String>,
    #[serde(default)]
    pub source_agent: Option<String>,
    #[serde(default)]
    pub source_label: Option<String>,
    #[serde(default)]
    pub compact_summary: Option<String>,
    #[serde(default)]
    pub tool_outputs: Vec<String>,
    #[serde(default)]
    pub turns: Vec<SessionTurn>,
}

fn default_session_schema_version() -> u32 {
    1
}

fn default_session_artifact_kind() -> String {
    "session".to_string()
}

/// Persisted raw session artifact, separate from derived memories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionArtifact {
    pub id: String,
    pub schema_version: u32,
    pub artifact_kind: String,
    pub session_id: Option<String>,
    pub title: Option<String>,
    pub summary: Option<String>,
    pub source_agent: Option<String>,
    pub source_label: Option<String>,
    pub is_subagent: bool,
    pub started_at: Option<DateTime<Utc>>,
    pub ended_at: Option<DateTime<Utc>>,
    pub turn_count: u32,
    pub transcript_text: String,
    pub transcript_json: Option<String>,
    pub episode_id: Option<String>,
    pub created_at: DateTime<Utc>,
}

/// Richer report for explicit session ingestion.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IngestReport {
    pub queued: bool,
    pub artifact_id: Option<String>,
    pub session_id: Option<String>,
    pub episode_id: Option<String>,
    pub memory_count: u32,
    pub concept_count: u32,
    pub link_count: u32,
    pub turn_count: u32,
    pub filtered_count: u32,
    pub secret_filtered_count: u32,
    pub created_count: u32,
    pub merged_count: u32,
    pub superseded_count: u32,
    pub stored_memory_ids: Vec<String>,
    pub primary_topics: Vec<String>,
}

/// Relation types between concepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Relation {
    PartOf,
    DependsOn,
    RelatedTo,
    Contradicts,
    Refines,
    AlternativeTo,
    CausedBy,
    InstanceOf,
    SupersededBy,
}

impl fmt::Display for Relation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PartOf => write!(f, "part_of"),
            Self::DependsOn => write!(f, "depends_on"),
            Self::RelatedTo => write!(f, "related_to"),
            Self::Contradicts => write!(f, "contradicts"),
            Self::Refines => write!(f, "refines"),
            Self::AlternativeTo => write!(f, "alternative_to"),
            Self::CausedBy => write!(f, "caused_by"),
            Self::InstanceOf => write!(f, "instance_of"),
            Self::SupersededBy => write!(f, "superseded_by"),
        }
    }
}

impl FromStr for Relation {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().replace('-', "_").as_str() {
            "part_of" | "partof" => Ok(Self::PartOf),
            "depends_on" | "dependson" => Ok(Self::DependsOn),
            "related_to" | "relatedto" => Ok(Self::RelatedTo),
            "contradicts" => Ok(Self::Contradicts),
            "refines" => Ok(Self::Refines),
            "alternative_to" | "alternativeto" => Ok(Self::AlternativeTo),
            "caused_by" | "causedby" => Ok(Self::CausedBy),
            "instance_of" | "instanceof" => Ok(Self::InstanceOf),
            "superseded_by" | "supersededby" => Ok(Self::SupersededBy),
            _ => Err(format!("unknown relation: {s}")),
        }
    }
}
