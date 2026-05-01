//! ARS parameter-policy activation storage.

use rusqlite::{params, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const ARS_PARAMETER_POLICY_METADATA_KEY: &str = "ars_parameter_policy";
const ARS_PARAMETER_POLICY_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ArsParameterPolicyMode {
    #[default]
    Disabled,
    Shadow,
    Canary,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ArsParameterPolicy {
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    #[serde(default)]
    pub revision: u64,
    #[serde(default)]
    pub mode: ArsParameterPolicyMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disabled_reason: Option<String>,
    #[serde(default)]
    pub source_adaptive_version: u64,
    /// Runtime adoption cap in `[0, 1]`. This is the rollout-side weight
    /// multiplied into dynamic trust, so canary activation slides gradually
    /// from static priors toward learned values instead of acting as a binary
    /// switch.
    #[serde(default)]
    pub runtime_adoption_weight: f64,
    /// Optional scoped rollout weights. Keys are stable policy surfaces such as
    /// `recall_fusion:semantic`, `recall_fusion:semantic:7`, or
    /// `judge_sample_rate`. Missing scopes fall back to
    /// `runtime_adoption_weight`.
    #[serde(default)]
    pub adoption_weights: HashMap<String, f64>,
    #[serde(default)]
    pub last_event_id: i64,
    #[serde(default)]
    pub last_updated: String,
}

impl Default for ArsParameterPolicy {
    fn default() -> Self {
        Self {
            schema_version: ARS_PARAMETER_POLICY_SCHEMA_VERSION,
            revision: 0,
            mode: ArsParameterPolicyMode::Disabled,
            disabled_reason: Some("missing policy row".to_string()),
            source_adaptive_version: 0,
            runtime_adoption_weight: 0.0,
            adoption_weights: HashMap::new(),
            last_event_id: 0,
            last_updated: String::new(),
        }
    }
}

impl ArsParameterPolicy {
    pub fn disabled(reason: impl Into<String>) -> Self {
        Self {
            disabled_reason: Some(reason.into()),
            ..Self::default()
        }
    }

    pub fn allows_runtime_adoption(&self, adaptive_version: u64) -> bool {
        self.runtime_adoption_weight(adaptive_version) > f64::EPSILON
    }

    pub fn runtime_adoption_weight(&self, adaptive_version: u64) -> f64 {
        if !self.can_adopt_runtime(adaptive_version) {
            return 0.0;
        }
        clamp01(self.runtime_adoption_weight)
    }

    pub fn runtime_adoption_weight_for(&self, adaptive_version: u64, key: &str) -> f64 {
        if !self.can_adopt_runtime(adaptive_version) {
            return 0.0;
        }
        self.adoption_weights
            .get(key)
            .copied()
            .map(clamp01)
            .unwrap_or_else(|| self.runtime_adoption_weight(adaptive_version))
    }

    fn can_adopt_runtime(&self, adaptive_version: u64) -> bool {
        self.schema_version == ARS_PARAMETER_POLICY_SCHEMA_VERSION
            && matches!(self.mode, ArsParameterPolicyMode::Canary)
            && self.source_adaptive_version <= adaptive_version
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ArsParameterPolicyLoadStatus {
    Missing,
    Loaded,
    Corrupt,
    StorageError,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ArsParameterPolicyLoad {
    pub policy: ArsParameterPolicy,
    pub status: ArsParameterPolicyLoadStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

pub fn load_parameter_policy(conn: &rusqlite::Connection) -> ArsParameterPolicyLoad {
    let raw = match conn
        .query_row(
            "SELECT value FROM metadata WHERE key = ?1",
            params![ARS_PARAMETER_POLICY_METADATA_KEY],
            |row| row.get::<_, String>(0),
        )
        .optional()
    {
        Ok(raw) => raw,
        Err(e) => {
            return ArsParameterPolicyLoad {
                policy: ArsParameterPolicy::disabled("policy storage error"),
                status: ArsParameterPolicyLoadStatus::StorageError,
                error: Some(e.to_string()),
            };
        }
    };

    let Some(raw) = raw else {
        return ArsParameterPolicyLoad {
            policy: ArsParameterPolicy::default(),
            status: ArsParameterPolicyLoadStatus::Missing,
            error: None,
        };
    };

    match serde_json::from_str::<ArsParameterPolicy>(&raw) {
        Ok(mut policy) if policy.schema_version == ARS_PARAMETER_POLICY_SCHEMA_VERSION => {
            clamp_policy_weights(&mut policy);
            ArsParameterPolicyLoad {
                policy,
                status: ArsParameterPolicyLoadStatus::Loaded,
                error: None,
            }
        }
        Ok(_) => ArsParameterPolicyLoad {
            policy: ArsParameterPolicy::disabled("unsupported policy schema version"),
            status: ArsParameterPolicyLoadStatus::Corrupt,
            error: Some("unsupported policy schema version".to_string()),
        },
        Err(e) => ArsParameterPolicyLoad {
            policy: ArsParameterPolicy::disabled("corrupt policy row"),
            status: ArsParameterPolicyLoadStatus::Corrupt,
            error: Some(e.to_string()),
        },
    }
}

#[must_use = "callers must not assume a policy update landed after a CAS miss"]
pub fn save_parameter_policy_cas(
    conn: &rusqlite::Connection,
    policy: &ArsParameterPolicy,
    expected_revision: u64,
) -> rusqlite::Result<bool> {
    let mut policy = policy.clone();
    clamp_policy_weights(&mut policy);
    let json = serde_json::to_string(&policy)
        .expect("ArsParameterPolicy serialization cannot fail for finite fields");

    let updated = conn.execute(
        "UPDATE metadata
            SET value = ?1
          WHERE key = ?2
            AND json_valid(value)
            AND COALESCE(json_extract(value, '$.revision'), 0) = ?3",
        params![json, ARS_PARAMETER_POLICY_METADATA_KEY, expected_revision],
    )?;
    if updated == 1 {
        return Ok(true);
    }

    let exists: bool = conn.query_row(
        "SELECT COUNT(*) > 0 FROM metadata WHERE key = ?1",
        params![ARS_PARAMETER_POLICY_METADATA_KEY],
        |row| row.get(0),
    )?;
    if exists || expected_revision != 0 {
        return Ok(false);
    }

    match conn.execute(
        "INSERT INTO metadata (key, value) VALUES (?1, ?2)",
        params![ARS_PARAMETER_POLICY_METADATA_KEY, json],
    ) {
        Ok(1) => Ok(true),
        Ok(_) => Ok(false),
        Err(rusqlite::Error::SqliteFailure(err, _))
            if err.code == rusqlite::ErrorCode::ConstraintViolation =>
        {
            Ok(false)
        }
        Err(e) => Err(e),
    }
}

pub fn delete_parameter_policy(conn: &rusqlite::Connection) -> rusqlite::Result<usize> {
    conn.execute(
        "DELETE FROM metadata WHERE key = ?1",
        params![ARS_PARAMETER_POLICY_METADATA_KEY],
    )
}

fn default_schema_version() -> u32 {
    ARS_PARAMETER_POLICY_SCHEMA_VERSION
}

fn clamp01(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn clamp_policy_weights(policy: &mut ArsParameterPolicy) {
    policy.runtime_adoption_weight = clamp01(policy.runtime_adoption_weight);
    for value in policy.adoption_weights.values_mut() {
        *value = clamp01(*value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::alpha_optimizer::ShadowFusionWeights;
    use crate::store::adaptive::{
        AdaptiveState, LearnedShadowFusionEntry, ShadowFusionWeightEntry,
    };

    fn conn() -> rusqlite::Connection {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute_batch("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
            .unwrap();
        conn
    }

    fn canary_policy(revision: u64) -> ArsParameterPolicy {
        ArsParameterPolicy {
            schema_version: ARS_PARAMETER_POLICY_SCHEMA_VERSION,
            revision,
            mode: ArsParameterPolicyMode::Canary,
            disabled_reason: None,
            source_adaptive_version: 7,
            runtime_adoption_weight: 1.0,
            adoption_weights: HashMap::new(),
            last_event_id: 99,
            last_updated: "2026-05-01T00:00:00Z".to_string(),
        }
    }

    #[test]
    fn parameter_policy_missing_or_corrupt_loads_disabled() {
        let conn = conn();
        let missing = load_parameter_policy(&conn);
        assert_eq!(missing.status, ArsParameterPolicyLoadStatus::Missing);
        assert_eq!(missing.policy.mode, ArsParameterPolicyMode::Disabled);

        conn.execute(
            "INSERT INTO metadata (key, value) VALUES (?1, ?2)",
            params![ARS_PARAMETER_POLICY_METADATA_KEY, "{not json"],
        )
        .unwrap();

        let corrupt = load_parameter_policy(&conn);
        assert_eq!(corrupt.status, ArsParameterPolicyLoadStatus::Corrupt);
        assert_eq!(corrupt.policy.mode, ArsParameterPolicyMode::Disabled);
        assert!(corrupt.error.is_some());
    }

    #[test]
    fn parameter_policy_save_cas_rejects_stale_revision() {
        let conn = conn();
        assert!(save_parameter_policy_cas(&conn, &canary_policy(1), 0).unwrap());

        let stale = canary_policy(2);
        assert!(!save_parameter_policy_cas(&conn, &stale, 0).unwrap());

        let loaded = load_parameter_policy(&conn);
        assert_eq!(loaded.policy.revision, 1);
        assert!(save_parameter_policy_cas(&conn, &stale, 1).unwrap());
        assert_eq!(load_parameter_policy(&conn).policy.revision, 2);
    }

    #[test]
    fn canary_policy_requires_positive_runtime_adoption_weight() {
        let mut policy = canary_policy(1);
        policy.runtime_adoption_weight = 0.0;
        assert_eq!(policy.runtime_adoption_weight(7), 0.0);
        assert!(!policy.allows_runtime_adoption(7));

        policy.runtime_adoption_weight = 0.25;
        assert_eq!(policy.runtime_adoption_weight(7), 0.25);
        assert!(policy.allows_runtime_adoption(7));
    }

    #[test]
    fn canary_policy_uses_scoped_runtime_adoption_weights_with_global_fallback() {
        let mut policy = canary_policy(1);
        policy.runtime_adoption_weight = 0.25;
        policy
            .adoption_weights
            .insert("recall_fusion:semantic".to_string(), 0.40);
        policy
            .adoption_weights
            .insert("recall_fusion:semantic:7".to_string(), 0.65);

        assert_eq!(
            policy.runtime_adoption_weight_for(7, "recall_fusion:semantic"),
            0.40
        );
        assert_eq!(
            policy.runtime_adoption_weight_for(7, "recall_fusion:semantic:7"),
            0.65
        );
        assert_eq!(
            policy.runtime_adoption_weight_for(7, "concept_summary_gate"),
            0.25
        );
        assert_eq!(
            policy.runtime_adoption_weight_for(6, "recall_fusion:semantic"),
            0.0
        );
    }

    #[test]
    fn parameter_policy_load_clamps_runtime_adoption_weight() {
        let conn = conn();
        let mut policy = canary_policy(1);
        policy.runtime_adoption_weight = 2.5;
        assert!(save_parameter_policy_cas(&conn, &policy, 0).unwrap());

        let loaded = load_parameter_policy(&conn);
        assert_eq!(loaded.policy.runtime_adoption_weight, 1.0);

        let mut policy = loaded.policy;
        policy.revision += 1;
        policy.runtime_adoption_weight = f64::NAN;
        policy
            .adoption_weights
            .insert("judge_sample_rate".to_string(), 2.0);
        policy
            .adoption_weights
            .insert("llm_feedback_decay".to_string(), f64::NAN);
        assert!(save_parameter_policy_cas(&conn, &policy, 1).unwrap());
        let loaded = load_parameter_policy(&conn).policy;
        assert_eq!(loaded.runtime_adoption_weight, 0.0);
        assert_eq!(loaded.adoption_weights["judge_sample_rate"], 1.0);
        assert_eq!(loaded.adoption_weights["llm_feedback_decay"], 0.0);
    }

    #[test]
    fn policy_rollback_disables_canary_without_erasing_learned_shadow_fusion() {
        let conn = conn();
        let mut state = AdaptiveState {
            version: 7,
            ..AdaptiveState::default()
        };
        let weights = ShadowFusionWeights::default();
        state.learned_shadow_fusion.insert(
            "global".to_string(),
            LearnedShadowFusionEntry {
                weights: ShadowFusionWeightEntry {
                    bm25: weights.bm25,
                    vec: weights.vec,
                    kg: weights.kg,
                    episode: weights.episode,
                    support: weights.support,
                    diversity: weights.diversity,
                },
                sample_count: 20,
                last_updated: "2026-05-01T00:00:00Z".to_string(),
            },
        );
        state.save_snapshot(&conn).unwrap();
        assert!(save_parameter_policy_cas(&conn, &canary_policy(1), 0).unwrap());

        delete_parameter_policy(&conn).unwrap();

        let policy = load_parameter_policy(&conn);
        let restored = AdaptiveState::restore_snapshot(&conn).unwrap();
        assert_eq!(policy.policy.mode, ArsParameterPolicyMode::Disabled);
        assert!(restored.learned_shadow_fusion.contains_key("global"));
    }
}
