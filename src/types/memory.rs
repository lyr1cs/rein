use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

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
    pub related_ids: Vec<String>,
    /// IDs of concepts derived from this memory (Memory → Concept link)
    #[serde(default)]
    pub concept_ids: Vec<String>,
    #[serde(default)]
    pub status: MemoryStatus,
    #[serde(skip)]
    pub embedding: Option<Vec<f32>>,
    /// Memory tier: hot, warm, cold (M5 adaptive tiering)
    #[serde(default = "default_tier")]
    pub tier: String,
    /// Cluster assignment from HDBSCAN (M4)
    #[serde(default)]
    pub cluster_id: Option<u32>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub last_accessed: DateTime<Utc>,
}

fn default_tier() -> String { "warm".to_string() }

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
            other => {
                tracing::warn!("unknown MemoryStatus '{}', defaulting to Active", other);
                Ok(Self::Active)
            }
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
    /// Concept IDs touched in this session.
    pub concept_ids: Vec<String>,
    /// Memory IDs created in this session.
    pub memory_ids: Vec<String>,
    pub created_at: DateTime<Utc>,
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
