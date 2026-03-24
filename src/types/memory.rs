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
    #[serde(skip)]
    pub embedding: Option<Vec<f32>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub last_accessed: DateTime<Utc>,
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
}

impl Source {
    pub fn as_str(&self) -> &'static str {
        match self {
            Source::Manual => "manual",
            Source::Hook => "hook",
            Source::Migration => "migration",
            Source::Supermemory => "supermemory",
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
