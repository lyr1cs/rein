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
    /// The row exists but its bytes failed JSON parse — a genuinely
    /// malformed value, no recoverable interpretation. Safe for
    /// `doctor --fix` to delete (R5 P2 audit catch 2026-05-04).
    Corrupt,
    /// The row exists and parses as JSON but its `schema_version`
    /// field disagrees with this binary's `ARS_PARAMETER_POLICY_SCHEMA_VERSION`.
    /// In a downgrade scenario this is the older binary reading a
    /// row written by a newer one — the data is valid, the older
    /// binary just can't interpret it. `doctor --fix` MUST NOT
    /// delete this; failing closed (`Disabled` semantics until the
    /// newer binary is restored or an operator hand-edits the row)
    /// is the correct behavior.
    UnsupportedSchema,
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

    // R8 P2 #1 (2026-05-04): peek `schema_version` BEFORE the typed
    // deserialize. Pre-fix, a future-schema row whose payload added an
    // unknown `mode` enum variant (or any other field this binary
    // cannot deserialize) failed the typed parse outright and fell
    // into the `Corrupt` arm; `doctor --fix` would then delete valid
    // future canary state on a downgrade. Inspecting the raw JSON
    // first makes `UnsupportedSchema` win for every future-vs-current
    // mismatch, regardless of whether the additive change is purely
    // field-additive or breaks the older binary's enum coverage.
    //
    // R15 P2 (2026-05-04): the comparison MUST be `>`, not `!=`.
    // An older row with `schema_version=0` (hand-edited, manually
    // restored from a backup, or otherwise corrupt) is NOT future
    // schema and must NOT receive the downgrade-preservation
    // treatment. Pre-R15 the `!=` check classified `0` as
    // `UnsupportedSchema`, `refresh_ars_parameter_policy` skipped
    // it as unhealthy, and `doctor --fix` refused to delete it —
    // policy refresh stalled permanently.  Only schemas STRICTLY
    // GREATER than the binary's current schema get preserved;
    // older or zero schemas fall through to typed deserialize and
    // are classified `Corrupt` so the recovery path can unblock
    // refresh.
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(&raw) {
        if let Some(schema) = value.get("schema_version").and_then(|v| v.as_u64()) {
            if schema > u64::from(ARS_PARAMETER_POLICY_SCHEMA_VERSION) {
                return ArsParameterPolicyLoad {
                    policy: ArsParameterPolicy::disabled("unsupported policy schema version"),
                    status: ArsParameterPolicyLoadStatus::UnsupportedSchema,
                    error: Some(format!(
                        "policy schema_version={} is newer than binary \
                         schema_version={}; treat as future-schema row and \
                         preserve until the newer binary is restored",
                        schema, ARS_PARAMETER_POLICY_SCHEMA_VERSION
                    )),
                };
            }
        }
    }

    match serde_json::from_str::<ArsParameterPolicy>(&raw) {
        Ok(mut policy) if policy.schema_version == ARS_PARAMETER_POLICY_SCHEMA_VERSION => {
            clamp_policy_weights(&mut policy);
            ArsParameterPolicyLoad {
                policy,
                status: ArsParameterPolicyLoadStatus::Loaded,
                error: None,
            }
        }
        Ok(policy) => ArsParameterPolicyLoad {
            policy: ArsParameterPolicy::disabled("older or invalid policy schema version"),
            // R15 P2 (2026-05-04): the peek above already filtered
            // out FUTURE schemas (`> current`). If the typed
            // deserialize lands here with a non-current
            // `schema_version`, the value is necessarily older or
            // hand-edited (e.g., literal `0`).  Pre-R15 this was
            // `UnsupportedSchema`, which `doctor --fix` refused to
            // delete and `refresh_ars_parameter_policy` skipped as
            // unhealthy — leaving an older-schema row in a
            // refresh-stalled state forever. Treat it as recoverable
            // corruption so the doctor recovery path can delete and
            // refresh-as-fresh-INSERT can re-establish the row at
            // the current schema.
            status: ArsParameterPolicyLoadStatus::Corrupt,
            error: Some(format!(
                "policy schema_version={} is older or invalid (current binary schema={}); \
                 row will be recovered by doctor --fix",
                policy.schema_version, ARS_PARAMETER_POLICY_SCHEMA_VERSION
            )),
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

    // R6 P2 defense-in-depth (2026-05-04): the UPDATE predicate must
    // ALSO match this binary's `ARS_PARAMETER_POLICY_SCHEMA_VERSION`.
    // Pre-fix the predicate was revision-only, so if a downgraded
    // (older) binary's caller built a default policy at revision=0
    // and a future-schema row happened to have a missing or zero
    // `revision` field, the UPDATE would silently overwrite valid
    // future-schema bytes. Adding the schema-version guard makes the
    // CAS fail-closed against ANY caller — including hypothetical
    // future code paths that bypass the refresh-layer early-return —
    // and the row survives the downgrade window untouched.
    //
    // R8 P2 #2 (2026-05-04): the COALESCE default for the schema_version
    // guard MUST be the current binary's `ARS_PARAMETER_POLICY_SCHEMA_VERSION`,
    // not 0. A row that omits the field entirely (e.g., one written by
    // an older binary before the field was introduced) deserializes via
    // `#[serde(default)]` to schema=1 and `load_parameter_policy`
    // reports `Loaded`, so the refresh layer hands the policy back at
    // its current revision. With the old `COALESCE(..., 0) = ?4`
    // predicate, the missing-field row coalesced to 0 ≠ 1 and every
    // refresh silently missed; the existing-row check then prevented
    // INSERT and policy promotion or rollback stalled forever.
    // Defaulting the COALESCE to `?4` makes a missing field interpret
    // as the current schema (matches), an explicit `?4` match
    // (matches), and any future schema (e.g., 2) NOT match — the
    // future-row preservation property the R6 guard added is preserved.
    let updated = conn.execute(
        "UPDATE metadata
            SET value = ?1
          WHERE key = ?2
            AND json_valid(value)
            AND COALESCE(json_extract(value, '$.revision'), 0) = ?3
            AND COALESCE(json_extract(value, '$.schema_version'), ?4) = ?4",
        params![
            json,
            ARS_PARAMETER_POLICY_METADATA_KEY,
            expected_revision,
            ARS_PARAMETER_POLICY_SCHEMA_VERSION,
        ],
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

/// Result of a `repair_corrupt_parameter_policy` call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepairCorruptOutcome {
    /// Number of rows deleted (0 if the row was no longer Corrupt at
    /// recovery time — e.g., a peer wrote a healthy row in between).
    pub deleted: usize,
    /// The corrupt-row error message captured INSIDE the transaction
    /// at delete time. Only populated when `deleted > 0`.
    pub error_at_delete: Option<String>,
    /// The status observed inside the recovery transaction; useful for
    /// log messages distinguishing "deleted" from "skipped (peer
    /// already repaired)" or "skipped (now StorageError)".
    pub observed_status: ArsParameterPolicyLoadStatus,
}

/// R10 P3 (2026-05-04): atomically check the policy row's status and
/// DELETE only if it is currently `Corrupt`. Closes a TOCTOU race
/// where a peer `doctor --fix` or `refresh_ars_parameter_policy`
/// could rewrite the row to a healthy canary state between an
/// earlier `load_parameter_policy` call and an unconditional
/// `delete_parameter_policy` — destroying valid state that just
/// landed.
///
/// The recovery wraps the load + delete in a single
/// `BEGIN IMMEDIATE` transaction (matches the resummerize / judge
/// contract pattern in this crate). `BEGIN IMMEDIATE` acquires the
/// write lock immediately, so no peer can interleave between the
/// status check and the delete.
///
/// Returns `RepairCorruptOutcome::observed_status` so callers can
/// distinguish:
/// - `Corrupt` + `deleted > 0` → row was still corrupt, recovery applied.
/// - `Loaded` / `Missing` + `deleted == 0` → peer already repaired
///   the row, recovery declined to touch it.
/// - `UnsupportedSchema` / `StorageError` + `deleted == 0` → row
///   transitioned to a different unhealthy state under the lock;
///   recovery declined per the R4/R5 "destructive only on Corrupt"
///   discipline.
#[must_use = "callers must report whether recovery actually deleted the row"]
pub fn repair_corrupt_parameter_policy(
    conn: &rusqlite::Connection,
) -> rusqlite::Result<RepairCorruptOutcome> {
    conn.execute_batch("BEGIN IMMEDIATE")?;
    let recovery: rusqlite::Result<RepairCorruptOutcome> = (|| {
        let loaded = load_parameter_policy(conn);
        if !matches!(loaded.status, ArsParameterPolicyLoadStatus::Corrupt) {
            return Ok(RepairCorruptOutcome {
                deleted: 0,
                error_at_delete: None,
                observed_status: loaded.status,
            });
        }
        let deleted = conn.execute(
            "DELETE FROM metadata WHERE key = ?1",
            params![ARS_PARAMETER_POLICY_METADATA_KEY],
        )?;
        Ok(RepairCorruptOutcome {
            deleted,
            error_at_delete: loaded.error,
            observed_status: loaded.status,
        })
    })();
    match recovery {
        Ok(outcome) => {
            conn.execute_batch("COMMIT")?;
            Ok(outcome)
        }
        Err(e) => {
            // Best-effort rollback; surface the original error.
            let _ = conn.execute_batch("ROLLBACK");
            Err(e)
        }
    }
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

/// v0.28.7+ audit L6 — defense-in-depth cap on
/// `ArsParameterPolicy.adoption_weights`. The map is repopulated from
/// `state.learned_shadow_fusion` on every pipeline tick (see
/// `next_scoped_adoption_weights`), so its size is dominated by the
/// learned_shadow_fusion cap (4096) plus a small fixed set of global
/// keys (`synthesis_gate`, `concept_summary_gate`, `judge_sample_rate`,
/// `llm_feedback_decay`, `signal_hint_priors`). Setting the cap above
/// 4096 + 32 = 4128 leaves comfortable headroom for that arithmetic and
/// for future global keys while still bounding pathological growth from
/// an unknown insert path.
///
/// Per the audit's "defense-in-depth" framing — and per the user's
/// no-subjective-heuristics guidance — this is **warn-only**: a save
/// that exceeds the cap still lands so the operator never silently
/// loses a key the canary policy depends on. The warning is the
/// signal; truncation is left to a deliberate v0.29 schema migration if
/// the cap is ever actually approached.
pub const ADOPTION_WEIGHTS_CAP: usize = 4128;

fn clamp_policy_weights(policy: &mut ArsParameterPolicy) {
    policy.runtime_adoption_weight = clamp01(policy.runtime_adoption_weight);
    for value in policy.adoption_weights.values_mut() {
        *value = clamp01(*value);
    }
    if policy.adoption_weights.len() > ADOPTION_WEIGHTS_CAP {
        // Warn-only — see ADOPTION_WEIGHTS_CAP doc for rationale.
        // tracing::warn! emits at WARN level so the doctor / GUI can
        // surface this as an operator alert. We do NOT truncate
        // because every adoption_weights entry maps to a scope identifier
        // that the runtime trust gate consults; silently dropping one
        // would silently mute a canary scope.
        tracing::warn!(
            adoption_weights_len = policy.adoption_weights.len(),
            cap = ADOPTION_WEIGHTS_CAP,
            "ars_parameter_policy: adoption_weights size exceeds defense-in-depth cap; \
             a runaway clusterer or unknown insert path may be growing the map. \
             No keys dropped — investigate cardinality source"
        );
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

    /// v0.28.7+ audit L5 R6 P2 — `save_parameter_policy_cas` MUST
    /// refuse to overwrite a row whose `schema_version` is FUTURE
    /// relative to this binary's `ARS_PARAMETER_POLICY_SCHEMA_VERSION`.
    /// Pre-fix the CAS predicate was revision-only, so a future-schema
    /// row with a missing/zero `revision` field could be silently
    /// overwritten by a downgraded binary's default-policy save at
    /// `expected_revision=0`. The schema-version guard makes the CAS
    /// fail-closed regardless of revision arithmetic.
    #[test]
    fn parameter_policy_save_cas_refuses_to_overwrite_future_schema_row() {
        let conn = conn();
        // Plant a future-schema row whose `revision` is missing
        // (defaults to 0 via COALESCE), schema_version is 9999.
        let future_value = serde_json::json!({
            "schema_version": 9999,
            "mode": "canary",
            "source_adaptive_version": 0,
            "runtime_adoption_weight": 0.5,
            "adoption_weights": {},
            "last_event_id": 0,
            "last_updated": "2030-01-01T00:00:00Z",
            "future_field": "neat",
        })
        .to_string();
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES (?1, ?2)",
            params![ARS_PARAMETER_POLICY_METADATA_KEY, &future_value],
        )
        .unwrap();

        // A downgraded binary builds a default disabled policy and
        // calls CAS at expected_revision=0 (matches the COALESCE
        // arithmetic on the future row).
        let downgraded_default = ArsParameterPolicy::disabled("downgrade reset attempt");
        let cas_result = save_parameter_policy_cas(&conn, &downgraded_default, 0).unwrap();
        assert!(
            !cas_result,
            "CAS must reject the overwrite — the schema_version guard \
             makes the UPDATE predicate fail even though revision matches. \
             Pre-R6 fix this returned `true` and silently destroyed the \
             future-schema row's bytes."
        );

        // The original future-schema bytes must still be in the DB.
        let raw_now: String = conn
            .query_row(
                "SELECT value FROM metadata WHERE key = ?1",
                params![ARS_PARAMETER_POLICY_METADATA_KEY],
                |row| row.get(0),
            )
            .expect("future-schema row must survive the CAS call");
        assert_eq!(
            raw_now, future_value,
            "future-schema row's raw bytes must be preserved exactly across \
             a downgrade-binary CAS attempt"
        );

        // Sanity: load_parameter_policy still reports UnsupportedSchema
        // (the CAS didn't accidentally clamp/normalize the row either).
        let loaded = load_parameter_policy(&conn);
        assert_eq!(
            loaded.status,
            ArsParameterPolicyLoadStatus::UnsupportedSchema
        );
    }

    /// v0.28.7+ audit R8 P2 #1 — a future-schema row that ALSO contains
    /// fields this binary cannot deserialize (e.g., a new `mode` enum
    /// variant) MUST be classified as `UnsupportedSchema`, not
    /// `Corrupt`. Pre-R8 the typed `serde_json::from_str::<ArsParameterPolicy>`
    /// ran first and failed before the schema-version branch ever
    /// fired, so the row fell into the `Err` arm and `doctor --fix`
    /// would delete valid future canary state on a downgrade. The
    /// load path now peeks `schema_version` from the raw JSON Value
    /// before attempting the typed parse.
    #[test]
    fn parameter_policy_load_future_schema_with_unknown_mode_is_unsupported_schema_not_corrupt() {
        let conn = conn();
        let future_value = serde_json::json!({
            "schema_version": 9999,
            "revision": 5,
            "mode": "future_only_variant",
            "source_adaptive_version": 0,
            "runtime_adoption_weight": 0.5,
            "adoption_weights": {},
            "last_event_id": 0,
            "last_updated": "2030-01-01T00:00:00Z",
            "future_only_field": "neat",
        })
        .to_string();
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES (?1, ?2)",
            params![ARS_PARAMETER_POLICY_METADATA_KEY, &future_value],
        )
        .unwrap();
        let loaded = load_parameter_policy(&conn);
        assert_eq!(
            loaded.status,
            ArsParameterPolicyLoadStatus::UnsupportedSchema,
            "future-schema row with an unknown mode variant must be classified \
             UnsupportedSchema (not Corrupt) so doctor --fix preserves it on \
             downgrade. Pre-R8 fix the typed deserialize failed first and the \
             row was wrongly destroyed."
        );
        // The raw bytes must still be intact — the load path must not
        // mutate the row regardless of how it classifies it.
        let raw_now: String = conn
            .query_row(
                "SELECT value FROM metadata WHERE key = ?1",
                params![ARS_PARAMETER_POLICY_METADATA_KEY],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(raw_now, future_value);
    }

    /// v0.28.7+ audit R8 P2 #2 — a row whose JSON omits the
    /// `schema_version` field entirely (e.g., one written by an older
    /// binary before the field was introduced) must deserialize via
    /// `#[serde(default)]` to schema=1 and accept CAS UPDATE under
    /// the matching revision. Pre-R8 the CAS predicate compared
    /// `COALESCE(..., 0) = ?4=1`, treating the missing field as 0 and
    /// silently rejecting every refresh. Combined with the existing-row
    /// check that prevented INSERT, policy promotion or rollback
    /// would stall permanently for upgraded rows.
    #[test]
    fn parameter_policy_save_cas_accepts_row_with_missing_schema_version_field() {
        let conn = conn();
        // Plant a row that lacks `schema_version` entirely.
        let no_schema = serde_json::json!({
            "revision": 7,
            "mode": "canary",
            "source_adaptive_version": 5,
            "runtime_adoption_weight": 0.3,
            "adoption_weights": {},
            "last_event_id": 12,
            "last_updated": "2026-04-01T00:00:00Z",
        })
        .to_string();
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES (?1, ?2)",
            params![ARS_PARAMETER_POLICY_METADATA_KEY, &no_schema],
        )
        .unwrap();

        // Sanity: load reports Loaded with schema_version defaulted to
        // ARS_PARAMETER_POLICY_SCHEMA_VERSION via #[serde(default)].
        let loaded = load_parameter_policy(&conn);
        assert_eq!(loaded.status, ArsParameterPolicyLoadStatus::Loaded);
        assert_eq!(
            loaded.policy.schema_version,
            ARS_PARAMETER_POLICY_SCHEMA_VERSION
        );
        assert_eq!(loaded.policy.revision, 7);

        // Build a refresh against the loaded policy at revision 7 (matches
        // the row's stored revision). The CAS UPDATE predicate must accept
        // this even though the on-disk JSON omits schema_version entirely.
        let mut refreshed = loaded.policy.clone();
        refreshed.revision = 8;
        refreshed.runtime_adoption_weight = 0.6;

        assert!(
            save_parameter_policy_cas(&conn, &refreshed, 7).unwrap(),
            "CAS UPDATE must succeed against a row that omits schema_version \
             (treated as the current binary's schema via COALESCE default). \
             Pre-R8 fix this returned false and refresh stalled forever."
        );

        // Verify the new bytes landed and now include schema_version.
        let after = load_parameter_policy(&conn);
        assert_eq!(after.policy.revision, 8);
        assert!((after.policy.runtime_adoption_weight - 0.6).abs() < f64::EPSILON);
        assert_eq!(
            after.policy.schema_version,
            ARS_PARAMETER_POLICY_SCHEMA_VERSION
        );
    }

    /// R8 P2 #2 follow-up — the COALESCE-default change MUST NOT
    /// regress the R6 future-row preservation property.  A future
    /// `schema_version=9999` row must still reject the downgrade-binary
    /// CAS even when the predicate's COALESCE default is `?4`
    /// (= current schema = 1) instead of 0, because COALESCE only
    /// substitutes when the field is missing — an explicit 9999 value
    /// is preserved by the comparison.
    #[test]
    fn parameter_policy_save_cas_still_refuses_future_schema_after_coalesce_default_change() {
        let conn = conn();
        let future_value = serde_json::json!({
            "schema_version": 9999,
            "revision": 0,
            "mode": "canary",
            "source_adaptive_version": 0,
            "runtime_adoption_weight": 0.5,
            "adoption_weights": {},
            "last_event_id": 0,
            "last_updated": "2030-01-01T00:00:00Z",
        })
        .to_string();
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES (?1, ?2)",
            params![ARS_PARAMETER_POLICY_METADATA_KEY, &future_value],
        )
        .unwrap();

        let downgraded_default = ArsParameterPolicy::disabled("downgrade reset attempt");
        let cas_result = save_parameter_policy_cas(&conn, &downgraded_default, 0).unwrap();
        assert!(
            !cas_result,
            "future-schema row preservation must hold even after the \
             COALESCE default flipped from 0 → ?4. The explicit 9999 \
             value still cannot match ?4=1 in the comparison."
        );
        let raw_now: String = conn
            .query_row(
                "SELECT value FROM metadata WHERE key = ?1",
                params![ARS_PARAMETER_POLICY_METADATA_KEY],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(raw_now, future_value);
    }

    /// R15 P2 (2026-05-04) — older or hand-edited schemas (e.g.,
    /// `schema_version: 0`) MUST be classified `Corrupt`, NOT
    /// `UnsupportedSchema`. Pre-R15 the peek check used `!=`, so a
    /// schema=0 row was wrongly preserved as future-schema and the
    /// recovery path stalled permanently. After R15 the peek uses
    /// `>` (strictly greater) and the typed-deserialize `Ok(_)` arm
    /// classifies older schemas as `Corrupt` so doctor --fix can
    /// recover them.
    #[test]
    fn parameter_policy_load_older_schema_version_is_corrupt_not_unsupported() {
        let conn = conn();
        // Plant a row with schema_version=0 — older than this
        // binary's schema=1. The JSON deserializes cleanly; the
        // schema-version branch must classify as Corrupt.
        let older_value = serde_json::json!({
            "schema_version": 0,
            "revision": 3,
            "mode": "shadow",
            "source_adaptive_version": 5,
            "runtime_adoption_weight": 0.0,
            "adoption_weights": {},
            "last_event_id": 7,
            "last_updated": "2026-04-01T00:00:00Z",
        })
        .to_string();
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES (?1, ?2)",
            params![ARS_PARAMETER_POLICY_METADATA_KEY, &older_value],
        )
        .unwrap();
        let loaded = load_parameter_policy(&conn);
        assert_eq!(
            loaded.status,
            ArsParameterPolicyLoadStatus::Corrupt,
            "older schema_version (0 < current=1) must be classified \
             Corrupt so `doctor --fix` can recover it. Pre-R15 fix the \
             peek check used `!=`, classifying older schemas as \
             UnsupportedSchema, which doctor refused to delete and \
             refresh skipped — refresh stalled forever."
        );
        // Recovery via repair helper still deletes it (the row is
        // genuinely corrupt for our purposes — schema_version older
        // than what this binary writes).
        let outcome = repair_corrupt_parameter_policy(&conn).unwrap();
        assert_eq!(outcome.deleted, 1);
        assert_eq!(
            outcome.observed_status,
            ArsParameterPolicyLoadStatus::Corrupt
        );
        // Subsequent load reports Missing — refresh path can now INSERT
        // a fresh row at current schema.
        let after = load_parameter_policy(&conn);
        assert_eq!(after.status, ArsParameterPolicyLoadStatus::Missing);
    }

    /// R10 P3 (2026-05-04) — `repair_corrupt_parameter_policy` must
    /// re-check the row's status under the write lock and decline to
    /// delete a row that has been repaired between an earlier read
    /// and this recovery call. Without the in-transaction re-check,
    /// a peer `refresh_ars_parameter_policy` tick or a concurrent
    /// `doctor --fix` could rewrite the row to a healthy canary in
    /// the gap, and the doctor would then destroy that newly-valid
    /// state.
    #[test]
    fn repair_corrupt_parameter_policy_skips_when_peer_already_repaired() {
        let conn = conn();
        // Plant a healthy canary policy directly. The repair path must
        // NOT touch this row even though a hypothetical earlier
        // `apply_local_fixes` read might have observed it as Corrupt
        // before a peer fixed it.
        assert!(save_parameter_policy_cas(&conn, &canary_policy(3), 0).unwrap());

        let outcome = repair_corrupt_parameter_policy(&conn).unwrap();
        assert_eq!(outcome.deleted, 0);
        assert_eq!(
            outcome.observed_status,
            ArsParameterPolicyLoadStatus::Loaded
        );
        assert!(outcome.error_at_delete.is_none());

        // Sanity — the canary policy is intact.
        let loaded = load_parameter_policy(&conn);
        assert_eq!(loaded.status, ArsParameterPolicyLoadStatus::Loaded);
        assert_eq!(loaded.policy.revision, 3);
        assert_eq!(loaded.policy.mode, ArsParameterPolicyMode::Canary);
    }

    /// R10 P3 (2026-05-04) — when the row is genuinely Corrupt at
    /// recovery time, `repair_corrupt_parameter_policy` deletes it
    /// and reports the error message captured INSIDE the transaction.
    #[test]
    fn repair_corrupt_parameter_policy_deletes_when_still_corrupt() {
        let conn = conn();
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES (?1, ?2)",
            params![ARS_PARAMETER_POLICY_METADATA_KEY, "{not json"],
        )
        .unwrap();

        let outcome = repair_corrupt_parameter_policy(&conn).unwrap();
        assert_eq!(outcome.deleted, 1);
        assert_eq!(
            outcome.observed_status,
            ArsParameterPolicyLoadStatus::Corrupt
        );
        assert!(outcome.error_at_delete.is_some());

        let loaded = load_parameter_policy(&conn);
        assert_eq!(loaded.status, ArsParameterPolicyLoadStatus::Missing);
    }

    /// R10 P3 (2026-05-04) — `repair_corrupt_parameter_policy` must NOT
    /// destroy a future-schema row even when called during a window
    /// the doctor's earlier read might have classified differently.
    /// The R8 P2 #1 fix made future-schema rows surface as
    /// `UnsupportedSchema` (not `Corrupt`) at the load layer, but the
    /// doctor's apply_local_fixes still passes ALL eligible rows
    /// through this helper. Belt-and-suspenders: the helper itself
    /// must observe the status under the write lock and decline.
    #[test]
    fn repair_corrupt_parameter_policy_preserves_future_schema_row() {
        let conn = conn();
        let future_value = serde_json::json!({
            "schema_version": 9999,
            "revision": 4,
            "mode": "canary",
            "source_adaptive_version": 0,
            "runtime_adoption_weight": 0.5,
            "adoption_weights": {},
            "last_event_id": 0,
            "last_updated": "2030-01-01T00:00:00Z",
        })
        .to_string();
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES (?1, ?2)",
            params![ARS_PARAMETER_POLICY_METADATA_KEY, &future_value],
        )
        .unwrap();

        let outcome = repair_corrupt_parameter_policy(&conn).unwrap();
        assert_eq!(outcome.deleted, 0);
        assert_eq!(
            outcome.observed_status,
            ArsParameterPolicyLoadStatus::UnsupportedSchema,
            "future-schema row must surface as UnsupportedSchema and \
             the helper must decline to delete it"
        );

        let raw_now: String = conn
            .query_row(
                "SELECT value FROM metadata WHERE key = ?1",
                params![ARS_PARAMETER_POLICY_METADATA_KEY],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(raw_now, future_value);
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

    /// v0.28.7+ audit L6 — `clamp_policy_weights` warns when the
    /// `adoption_weights` map exceeds the defense-in-depth cap, but
    /// MUST NOT silently drop entries (per advisor guidance: a 0.0
    /// weight from a deliberate canary→shadow rollback is operationally
    /// valuable, so heuristic eviction was rejected). The realistic
    /// post-L6 ceiling is `LEARNED_SHADOW_FUSION_CAP + 5 globals = 4101`
    /// per pipeline tick; the cap is set at `4128` for headroom. This
    /// test forces the over-cap branch by hand-building a degenerate
    /// policy and asserts the entry count is unchanged after clamping.
    #[test]
    fn clamp_policy_weights_warns_above_cap_but_does_not_drop_entries() {
        let mut policy = canary_policy(1);
        // Fill above the defense-in-depth cap. Use a value that will
        // still pass `clamp01` so we test the cap path independently.
        for i in 0..(ADOPTION_WEIGHTS_CAP + 32) {
            policy.adoption_weights.insert(format!("scope:{i}"), 0.5);
        }
        let pre_len = policy.adoption_weights.len();
        assert!(pre_len > ADOPTION_WEIGHTS_CAP);

        clamp_policy_weights(&mut policy);

        // No silent drops — the warn is the signal, the data is preserved.
        assert_eq!(
            policy.adoption_weights.len(),
            pre_len,
            "clamp_policy_weights MUST NOT drop entries above cap; the warn \
             is the operator-visible signal but every adoption_weights key \
             maps to a scope identifier the runtime trust gate consults — \
             dropping would silently mute a canary scope"
        );
        // Sanity: clamp01 still applied to values.
        for value in policy.adoption_weights.values() {
            assert!((0.0..=1.0).contains(value));
        }
    }
}
