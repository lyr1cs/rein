//! Deterministic A12 leave-one-evidence-out corpus, read-only replay, and
//! family-disjoint six-dimensional calibration.
//!
//! Tasks 1-3 only read snapshot-backed evidence. They never emit feedback,
//! access, or recall-hit writes, and the permanent activation holdout is never
//! passed to the optimizer.

// The staged crate-local outputs are consumed by Tasks 4-6 on their branches.
// Keep this module warning-clean before those integrations land.
#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet, HashMap};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::extract::dedup::similarity;
use crate::store::SqliteStore;
use crate::types::{ReinError, ReinResult};

const A12_FAMILY_PREFIX: &str = "a12-family:";
const A12_SPLIT_MODULUS: u8 = 5;

/// Source of an independently supported LOO positive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
// These names are the approved persisted provenance vocabulary from the A12
// design; keeping the `Loo` suffix avoids ambiguous future state migrations.
#[allow(clippy::enum_variant_names)]
pub(crate) enum A12OutcomeProvenance {
    CanonicalLoo,
    ConceptLoo,
    EpisodeLoo,
}

/// Permanent family-disjoint side of the A12 split.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum A12Fold {
    ActivationHoldout,
    Training,
}

/// Stable canonical family identity plus its current live tip.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct A12CanonicalFamily {
    pub stable_family_id: String,
    pub stable_created_at: DateTime<Utc>,
    pub split_bucket: u8,
    pub fold: A12Fold,
    pub live_tip_id: Option<String>,
    pub member_ids: Vec<String>,
}

/// Everything the read-only recall trace must remove before normalization.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct A12LooExclusion {
    pub held_out_memory_ids: Vec<String>,
    pub held_out_evidence_ids: Vec<String>,
    pub content_hash: String,
    pub equal_content_memory_ids: Vec<String>,
    pub near_duplicate_memory_ids: Vec<String>,
}

/// One positive live tip, with all independent provenance paths retained.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct A12LooPositive {
    pub stable_family_id: String,
    pub live_tip_id: String,
    pub provenance: Vec<A12OutcomeProvenance>,
}

/// One held-out evidence view inside a family-level observation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct A12LooCase {
    pub held_out_evidence_id: String,
    pub original_memory_id: Option<String>,
    pub query_text: String,
    pub exclusion: A12LooExclusion,
    pub positives: Vec<A12LooPositive>,
}

/// Equal-weight optimizer input. Multiple evidence views stay nested here, so
/// a family contributes one observation regardless of its evidence count.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct A12FamilyObservation {
    pub stable_family_id: String,
    pub live_tip_id: String,
    pub split_bucket: u8,
    pub fold: A12Fold,
    pub family_weight: f64,
    pub cases: Vec<A12LooCase>,
}

/// Why a held-out view cannot produce a leakage-free positive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum A12AbstentionReason {
    NoEvidenceViews,
    NoLiveCanonicalTip,
    HeldOutMemory,
    EqualContentHash,
    NearDuplicateContent,
    CrossFoldAuxiliary,
    NoIndependentPositive,
}

/// Explicit fail-closed record for a skipped view or family.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct A12LooAbstention {
    pub stable_family_id: String,
    pub held_out_evidence_id: Option<String>,
    pub original_memory_id: Option<String>,
    pub exclusion: Option<A12LooExclusion>,
    pub reason: A12AbstentionReason,
}

/// Deterministically ordered Task-1 output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct A12LooCorpus {
    /// Identity of the complete local recall state atomically observed while
    /// this corpus was assembled. Replay refuses to pair it with another
    /// generation.
    pub source_snapshot_fingerprint: String,
    /// Exact destructive bound used while constructing every exclusion set.
    pub hard_dedup_bound: f32,
    pub observations: Vec<A12FamilyObservation>,
    pub abstentions: Vec<A12LooAbstention>,
}

/// One read-only recall trace plus the scope labels captured from that query.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct A12CaseRecallTrace {
    pub query_type: String,
    pub cluster_id: Option<u32>,
    /// Earliest wall-clock instant (Unix milliseconds) at which this fixed-time
    /// replay can cease to match production eligibility. `now >= boundary`
    /// must fail closed.
    pub valid_until_exclusive: Option<i64>,
    pub trace: crate::search::recall::A12RecallTrace,
    pub provenance: A12ProvenanceCounts,
}

/// All traceable evidence views for one immutable canonical family.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct A12FamilyRecallTrace {
    pub stable_family_id: String,
    pub cases: Vec<A12CaseRecallTrace>,
}

/// Family-level paired hit@3 contingency table.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct A12PairedTop3 {
    pub both_hit: usize,
    pub baseline_only: usize,
    pub treatment_only: usize,
    pub neither_hit: usize,
}

/// Auditable count of the independent label paths represented in a scope.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct A12ProvenanceCounts {
    pub canonical_loo: usize,
    pub concept_loo: usize,
    pub episode_loo: usize,
}

impl A12ProvenanceCounts {
    fn add_assign(&mut self, other: Self) {
        self.canonical_loo = self.canonical_loo.saturating_add(other.canonical_loo);
        self.concept_loo = self.concept_loo.saturating_add(other.concept_loo);
        self.episode_loo = self.episode_loo.saturating_add(other.episode_loo);
    }
}

/// Generation metadata supplied by the snapshot builder. Keeping it explicit
/// makes the optimizer output deterministic for a fixed context and lets the
/// shared resolver compare exactly the identities that produced the weights.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct A12CalibrationContext {
    pub snapshot_fingerprint: String,
    pub corpus_fingerprint: String,
    pub optimizer_fingerprint: String,
    pub evaluation_fingerprint: String,
    pub calibrated_at: DateTime<Utc>,
}

#[cfg(test)]
fn a12_calibration_context_from_bytes(
    snapshot_bytes: &[u8],
    corpus_bytes: &[u8],
    calibrated_at: DateTime<Utc>,
) -> A12CalibrationContext {
    let snapshot_fingerprint = domain_separated_sha256(b"a12-snapshot-v1\0", snapshot_bytes);
    let corpus_fingerprint = domain_separated_sha256(b"a12-corpus-v1\0", corpus_bytes);
    let optimizer_fingerprint = fingerprint_framed(
        b"a12-family-equal-optimizer-v1\0",
        &[
            snapshot_fingerprint.as_bytes(),
            corpus_fingerprint.as_bytes(),
            env!("REIN_BUILD_FINGERPRINT").as_bytes(),
        ],
    );
    let evaluation_fingerprint = fingerprint_framed(
        b"a12-family-top3-evaluation-v1\0",
        &[
            optimizer_fingerprint.as_bytes(),
            env!("REIN_BUILD_FINGERPRINT").as_bytes(),
        ],
    );
    A12CalibrationContext {
        snapshot_fingerprint,
        corpus_fingerprint,
        optimizer_fingerprint,
        evaluation_fingerprint,
        calibrated_at,
    }
}

/// Internal, auditable projection of every operator config field that changes
/// the A12 LOO path. Remote providers are disabled in this mode, so credentials,
/// tokens, paths, and endpoints cannot enter either behavior or identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct A12RecallConfigProjection {
    schema_version: u8,
    lexical_backend: &'static str,
    temporal_policy: &'static str,
    fusion_method: String,
    rrf_k_bits: u64,
    rrf_fts_weight_bits: u64,
    rrf_vec_weight_bits: u64,
    cc_alpha_bits: u64,
    strong_signal_ratio_bits: u32,
    strong_signal_single_bits: u32,
    adaptive_enabled: bool,
    adaptive_min_samples_alpha: usize,
}

fn a12_recall_config_projection(config: &crate::config::ReinConfig) -> A12RecallConfigProjection {
    A12RecallConfigProjection {
        schema_version: 1,
        lexical_backend: "sqlite_fts5",
        temporal_policy: "fixed_evaluation_time_with_production_relative_and_kg_validity",
        fusion_method: config.search.fusion_method.clone(),
        rrf_k_bits: config.search.rrf_k.to_bits(),
        rrf_fts_weight_bits: config.search.rrf_fts_weight.to_bits(),
        rrf_vec_weight_bits: config.search.rrf_vec_weight.to_bits(),
        cc_alpha_bits: config.search.cc_alpha.to_bits(),
        strong_signal_ratio_bits: config.search.strong_signal_ratio.to_bits(),
        strong_signal_single_bits: config.search.strong_signal_single.to_bits(),
        adaptive_enabled: config.adaptive.enabled,
        adaptive_min_samples_alpha: config.adaptive.min_samples_alpha,
    }
}

/// Compute generation identities before the expensive recall replay. The
/// corpus must still match the complete local recall snapshot captured during
/// its own atomic build; otherwise callers must rebuild instead of mixing
/// generations.
pub(crate) fn a12_calibration_context_for_corpus(
    store: &SqliteStore,
    config: &crate::config::ReinConfig,
    corpus: &A12LooCorpus,
    trace_limit: usize,
    min_samples_alpha: usize,
    calibrated_at: DateTime<Utc>,
) -> ReinResult<A12CalibrationContext> {
    let snapshot_fingerprint = capture_a12_local_recall_snapshot_identity(store)?;
    if corpus.source_snapshot_fingerprint != snapshot_fingerprint {
        return Err(ReinError::Config(format!(
            "A12 corpus/local snapshot drift: corpus={} local={snapshot_fingerprint}",
            corpus.source_snapshot_fingerprint
        )));
    }
    let corpus_bytes = serde_json::to_vec(corpus)?;
    let corpus_fingerprint = domain_separated_sha256(b"a12-corpus-v1\0", &corpus_bytes);
    let config_bytes = serde_json::to_vec(&a12_recall_config_projection(config))?;
    let calibrated_at_bytes = calibrated_at.to_rfc3339_opts(chrono::SecondsFormat::Nanos, true);
    let optimizer_fingerprint = fingerprint_framed(
        b"a12-family-equal-optimizer-base-v2\0",
        &[
            snapshot_fingerprint.as_bytes(),
            corpus_fingerprint.as_bytes(),
            &u64::try_from(trace_limit).unwrap_or(u64::MAX).to_le_bytes(),
            &corpus.hard_dedup_bound.to_bits().to_le_bytes(),
            &config_bytes,
            calibrated_at_bytes.as_bytes(),
            env!("REIN_BUILD_FINGERPRINT").as_bytes(),
        ],
    );
    let required = min_samples_alpha.max(10);
    let evaluation_fingerprint = fingerprint_framed(
        b"a12-family-top3-evaluation-base-v2\0",
        &[
            optimizer_fingerprint.as_bytes(),
            &u64::try_from(required).unwrap_or(u64::MAX).to_le_bytes(),
            &3_u64.to_le_bytes(),
            &crate::eval::gates::DEFAULT_NOISE_FLOOR
                .to_bits()
                .to_le_bytes(),
            env!("REIN_BUILD_FINGERPRINT").as_bytes(),
        ],
    );
    Ok(A12CalibrationContext {
        snapshot_fingerprint,
        corpus_fingerprint,
        optimizer_fingerprint,
        evaluation_fingerprint,
        calibrated_at,
    })
}

/// Replay the Task-1 corpus through the Task-2 read-only trace and capture the
/// exact query classifier token used by that same recall path.
pub(crate) fn trace_a12_loo_corpus(
    store: &SqliteStore,
    config: &crate::config::ReinConfig,
    corpus: &A12LooCorpus,
    limit: usize,
) -> ReinResult<Vec<A12FamilyRecallTrace>> {
    let evaluation_at = Utc::now();
    trace_a12_loo_corpus_at(
        store,
        config,
        corpus,
        limit,
        evaluation_at,
        Some(&corpus.source_snapshot_fingerprint),
        |_store, _case_index| Ok(()),
    )
}

fn trace_a12_loo_corpus_at<F>(
    store: &SqliteStore,
    config: &crate::config::ReinConfig,
    corpus: &A12LooCorpus,
    limit: usize,
    evaluation_at: DateTime<Utc>,
    expected_snapshot: Option<&str>,
    mut after_case: F,
) -> ReinResult<Vec<A12FamilyRecallTrace>>
where
    F: FnMut(&SqliteStore, usize) -> ReinResult<()>,
{
    store.conn().execute_batch("BEGIN DEFERRED")?;
    let replay = (|| {
        let before = a12_local_recall_snapshot_identity(store)?;
        if expected_snapshot.is_some_and(|expected| expected != before)
            || corpus.source_snapshot_fingerprint != before
        {
            return Err(ReinError::Config(format!(
                "A12 replay snapshot mismatch: corpus={} expected={} local={before}",
                corpus.source_snapshot_fingerprint,
                expected_snapshot.unwrap_or("<none>")
            )));
        }
        let next_kg_boundary = a12_next_kg_validity_boundary_exclusive(store, evaluation_at)?;

        let mut case_index = 0_usize;
        let mut families = Vec::with_capacity(corpus.observations.len());
        for observation in &corpus.observations {
            if observation.fold != a12_family_fold(&observation.stable_family_id)
                || observation.split_bucket
                    != a12_family_split_bucket(&observation.stable_family_id)
            {
                return Err(ReinError::Config(format!(
                    "A12 corpus fold mismatch for stable family {}",
                    observation.stable_family_id
                )));
            }

            let mut cases = Vec::with_capacity(observation.cases.len());
            for case in &observation.cases {
                let trace = crate::search::recall::recall_loo_trace_at(
                    store,
                    config,
                    case,
                    limit.max(3),
                    evaluation_at,
                )?;
                let query_type = crate::search::classify::classify(&case.query_text, false, false)
                    .query_type
                    .to_string();
                let cluster_id = trace.event.query_cluster_id_at_recall;
                let next_relative_boundary = a12_next_relative_temporal_boundary_exclusive(
                    store,
                    &case.query_text,
                    evaluation_at,
                )?;
                cases.push(A12CaseRecallTrace {
                    query_type,
                    cluster_id,
                    valid_until_exclusive: min_optional_boundary(
                        next_kg_boundary,
                        next_relative_boundary,
                    ),
                    trace,
                    provenance: provenance_counts_for_case(case),
                });
                after_case(store, case_index)?;
                case_index = case_index.saturating_add(1);
            }
            families.push(A12FamilyRecallTrace {
                stable_family_id: observation.stable_family_id.clone(),
                cases,
            });
        }
        let after = a12_local_recall_snapshot_identity(store)?;
        if after != before {
            return Err(ReinError::Config(format!(
                "A12 local recall snapshot drifted during replay: before={before} after={after}"
            )));
        }
        Ok((families, before))
    })();

    let (families, snapshot) = match replay {
        Ok(value) => {
            store.conn().execute_batch("COMMIT")?;
            value
        }
        Err(error) => {
            let _ = store.conn().execute_batch("ROLLBACK");
            return Err(error);
        }
    };
    let live_after_commit = read_a12_local_recall_snapshot_identity(store)?;
    if live_after_commit != snapshot {
        return Err(ReinError::Config(format!(
            "A12 local recall snapshot advanced before generation finalization: replay={snapshot} live={live_after_commit}"
        )));
    }
    Ok(families)
}

fn min_optional_boundary(left: Option<i64>, right: Option<i64>) -> Option<i64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

fn a12_next_kg_validity_boundary_exclusive(
    store: &SqliteStore,
    evaluation_at: DateTime<Utc>,
) -> ReinResult<Option<i64>> {
    let mut statement = store
        .conn()
        .prepare("SELECT valid_from, valid_until FROM concept_links ORDER BY id")?;
    let rows = statement.query_map([], |row| {
        Ok((
            row.get::<_, Option<String>>(0)?,
            row.get::<_, Option<String>>(1)?,
        ))
    })?;
    let mut boundary = None;
    for row in rows {
        let (valid_from, valid_until) = row?;
        if let Some(valid_from) = valid_from {
            let valid_from = parse_timestamp(&valid_from, "concept_links.valid_from")?;
            if valid_from > evaluation_at {
                boundary = min_optional_boundary(boundary, Some(valid_from.timestamp_millis()));
            }
        }
        if let Some(valid_until) = valid_until {
            let valid_until = parse_timestamp(&valid_until, "concept_links.valid_until")?;
            // Production includes equality and changes immediately after it;
            // expiring at equality is the conservative exclusive contract.
            if valid_until >= evaluation_at {
                boundary = min_optional_boundary(boundary, Some(valid_until.timestamp_millis()));
            }
        }
    }
    Ok(boundary)
}

fn a12_next_relative_temporal_boundary_exclusive(
    store: &SqliteStore,
    query: &str,
    evaluation_at: DateTime<Utc>,
) -> ReinResult<Option<i64>> {
    let strategy = crate::search::classify::classify(query, false, false);
    if !strategy.force_temporal {
        return Ok(None);
    }
    let Some(days_back) = strategy.temporal_days_back else {
        return Ok(None);
    };
    let Some(window) = chrono::Duration::try_days(days_back) else {
        return Err(ReinError::Config(format!(
            "A12 temporal window days out of range: {days_back}"
        )));
    };

    let mut boundary = None;
    for sql in [
        "SELECT created_at FROM memories ORDER BY id",
        "SELECT created_at FROM episodes ORDER BY id",
    ] {
        let mut statement = store.conn().prepare(sql)?;
        let rows = statement.query_map([], |row| row.get::<_, String>(0))?;
        for row in rows {
            let created_at = parse_timestamp(&row?, "temporal.created_at")?;
            let next_change = if created_at > evaluation_at {
                // Future row enters the inclusive upper bound.
                Some(created_at)
            } else {
                // Existing row leaves once `now - days_back` passes it.
                created_at
                    .checked_add_signed(window)
                    .filter(|boundary| *boundary >= evaluation_at)
            };
            if let Some(next_change) = next_change {
                boundary = min_optional_boundary(boundary, Some(next_change.timestamp_millis()));
            }
        }
    }
    Ok(boundary)
}

/// End-to-end Task-3 entry point used by later persistence/policy tasks.
/// Fingerprints hash the exact deterministic JSON bytes of the observation
/// snapshot and complete corpus (including abstentions), respectively.
pub(crate) fn train_and_evaluate_a12_corpus(
    store: &SqliteStore,
    config: &crate::config::ReinConfig,
    corpus: &A12LooCorpus,
    limit: usize,
    min_samples_alpha: usize,
    calibrated_at: DateTime<Utc>,
) -> ReinResult<Vec<A12ScopeCalibration>> {
    // Resolve generation identity first so cadence callers can run the same
    // helper before deciding whether this expensive trace is necessary.
    let context = a12_calibration_context_for_corpus(
        store,
        config,
        corpus,
        limit,
        min_samples_alpha,
        calibrated_at,
    )?;
    let families = trace_a12_loo_corpus_at(
        store,
        config,
        corpus,
        limit,
        calibrated_at,
        Some(&context.snapshot_fingerprint),
        |_store, _case_index| Ok(()),
    )?;
    Ok(train_and_evaluate_a12_traces(
        &families,
        min_samples_alpha,
        &context,
    ))
}

fn provenance_counts_for_case(case: &A12LooCase) -> A12ProvenanceCounts {
    let mut counts = A12ProvenanceCounts::default();
    for provenance in case
        .positives
        .iter()
        .flat_map(|positive| positive.provenance.iter())
    {
        match provenance {
            A12OutcomeProvenance::CanonicalLoo => counts.canonical_loo += 1,
            A12OutcomeProvenance::ConceptLoo => counts.concept_loo += 1,
            A12OutcomeProvenance::EpisodeLoo => counts.episode_loo += 1,
        }
    }
    counts
}

fn domain_separated_sha256(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn fingerprint_framed(domain: &[u8], fields: &[&[u8]]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    for field in fields {
        hasher.update(u64::try_from(field.len()).unwrap_or(u64::MAX).to_le_bytes());
        hasher.update(field);
    }
    format!("{:x}", hasher.finalize())
}

/// Hash the exact SQLite-backed state read by A12 replay. External Tantivy and
/// HNSW are intentionally absent because A12 uses transaction-scoped FTS5 and
/// sqlite-vec only. Table rows are encoded by SQLite storage type and sorted by
/// their complete encoded bytes, so row order and query plans cannot perturb
/// the identity.
fn a12_local_recall_snapshot_identity(store: &SqliteStore) -> ReinResult<String> {
    const CORE_TABLES: &[&str] = &[
        "memories",
        "memory_canonical_state",
        "memory_evidence",
        "memories_fts",
        "vec_memories",
        "concepts",
        "concepts_fts",
        "concept_links",
        "episodes",
    ];

    let mut hasher = Sha256::new();
    hasher.update(b"a12-local-recall-snapshot-v2\0");
    let mut tables = CORE_TABLES
        .iter()
        .map(|table| (*table).to_string())
        .collect::<BTreeSet<_>>();
    // FTS5 and sqlite-vec search their version-dependent shadow tables. Hash
    // those too; hashing only the virtual table's stored content would miss a
    // stale/corrupt term or vector index whose visible rows still look intact.
    let mut statement = store.conn().prepare(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND (\
             name GLOB 'memories_fts_*' OR \
             name GLOB 'concepts_fts_*' OR \
             name GLOB 'vec_memories_*'\
         ) ORDER BY name",
    )?;
    for row in statement.query_map([], |row| row.get::<_, String>(0))? {
        tables.insert(row?);
    }
    for table in &tables {
        hash_sqlite_table(store.conn(), table, &mut hasher)?;
    }
    hash_a12_recall_metadata(store.conn(), &mut hasher)?;
    Ok(format!("{:x}", hasher.finalize()))
}

fn capture_a12_local_recall_snapshot_identity(store: &SqliteStore) -> ReinResult<String> {
    let snapshot = read_a12_local_recall_snapshot_identity(store)?;
    let live = read_a12_local_recall_snapshot_identity(store)?;
    if live != snapshot {
        return Err(ReinError::Config(format!(
            "A12 local recall snapshot advanced during identity capture: snapshot={snapshot} live={live}"
        )));
    }
    Ok(snapshot)
}

fn read_a12_local_recall_snapshot_identity(store: &SqliteStore) -> ReinResult<String> {
    store.conn().execute_batch("BEGIN DEFERRED")?;
    let snapshot = match a12_local_recall_snapshot_identity(store) {
        Ok(snapshot) => {
            if let Err(error) = store.conn().execute_batch("COMMIT") {
                let _ = store.conn().execute_batch("ROLLBACK");
                return Err(error.into());
            }
            snapshot
        }
        Err(error) => {
            let _ = store.conn().execute_batch("ROLLBACK");
            return Err(error);
        }
    };
    Ok(snapshot)
}

fn hash_sqlite_table(
    connection: &rusqlite::Connection,
    table: &str,
    hasher: &mut Sha256,
) -> ReinResult<()> {
    hash_field(hasher, table.as_bytes());
    let schema = connection
        .query_row(
            "SELECT sql FROM sqlite_master WHERE name = ?1 AND type IN ('table', 'view')",
            rusqlite::params![table],
            |row| row.get::<_, Option<String>>(0),
        )
        .ok()
        .flatten();
    let Some(schema) = schema else {
        hash_field(hasher, b"absent");
        return Ok(());
    };
    hash_field(hasher, schema.as_bytes());

    // Core names come from a compile-time allowlist and shadow names from
    // sqlite_master; still quote defensively before interpolation.
    let quoted_table = table.replace('"', "\"\"");
    let sql = format!("SELECT * FROM \"{quoted_table}\"");
    let mut statement = connection.prepare(&sql)?;
    let column_names = statement
        .column_names()
        .iter()
        .map(|name| (*name).to_string())
        .collect::<Vec<_>>();
    for column_name in &column_names {
        hash_field(hasher, column_name.as_bytes());
    }
    let column_count = statement.column_count();
    let mut rows = statement.query([])?;
    let mut encoded_rows = Vec::<Vec<u8>>::new();
    while let Some(row) = rows.next()? {
        let mut encoded = Vec::new();
        for index in 0..column_count {
            encode_sqlite_value(row.get_ref(index)?, &mut encoded);
        }
        encoded_rows.push(encoded);
    }
    encoded_rows.sort();
    hash_field(
        hasher,
        &u64::try_from(encoded_rows.len())
            .unwrap_or(u64::MAX)
            .to_le_bytes(),
    );
    for row in encoded_rows {
        hash_field(hasher, &row);
    }
    Ok(())
}

fn encode_sqlite_value(value: rusqlite::types::ValueRef<'_>, output: &mut Vec<u8>) {
    use rusqlite::types::ValueRef;
    match value {
        ValueRef::Null => output.push(0),
        ValueRef::Integer(value) => {
            output.push(1);
            output.extend_from_slice(&value.to_le_bytes());
        }
        ValueRef::Real(value) => {
            output.push(2);
            output.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        ValueRef::Text(value) => {
            output.push(3);
            append_framed(output, value);
        }
        ValueRef::Blob(value) => {
            output.push(4);
            append_framed(output, value);
        }
    }
}

fn append_framed(output: &mut Vec<u8>, bytes: &[u8]) {
    output.extend_from_slice(&u64::try_from(bytes.len()).unwrap_or(u64::MAX).to_le_bytes());
    output.extend_from_slice(bytes);
}

fn hash_field(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update(u64::try_from(bytes.len()).unwrap_or(u64::MAX).to_le_bytes());
    hasher.update(bytes);
}

fn hash_a12_recall_metadata(
    connection: &rusqlite::Connection,
    hasher: &mut Sha256,
) -> ReinResult<()> {
    let mut statement = connection.prepare(
        "SELECT key, value FROM metadata \
         WHERE key IN ('adaptive_state', 'rerank_weights') \
            OR key LIKE 'survival_curve:%' \
         ORDER BY key",
    )?;
    let rows = statement.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    })?;
    for row in rows {
        let (key, raw) = row?;
        hash_field(hasher, key.as_bytes());
        match serde_json::from_str::<serde_json::Value>(&raw) {
            Ok(value) => {
                hash_field(hasher, b"valid-json");
                if key == "adaptive_state" {
                    // `restore_snapshot` deserializes the complete struct before
                    // reading any individual field. A type error anywhere flips
                    // runtime behavior to the no-state fallback, so bind both
                    // that outcome and the full canonical value.
                    match serde_json::from_value::<crate::store::adaptive::AdaptiveState>(
                        value.clone(),
                    ) {
                        Ok(state) => {
                            hash_field(hasher, b"adaptive-restore-ok");
                            let projection = serde_json::json!({
                                "learned_alpha": state.learned_alpha,
                                "cluster_version": state.cluster_version,
                                "memory_clusters": state.memory_clusters,
                            });
                            hash_canonical_json(hasher, &projection);
                        }
                        Err(_) => {
                            // Any type error anywhere makes production's
                            // `restore_snapshot` return None. Bind the fallback
                            // outcome and complete value so distinct corrupt
                            // states cannot masquerade as current evidence.
                            hash_field(hasher, b"adaptive-restore-fallback");
                            hash_canonical_json(hasher, &value);
                            hash_field(hasher, raw.as_bytes());
                        }
                    }
                } else {
                    hash_canonical_json(hasher, &value);
                }
            }
            Err(_) => {
                // Malformed metadata triggers runtime fallback; retain the raw
                // bytes so a repair still advances the generation identity.
                hash_field(hasher, b"malformed-json");
                hash_field(hasher, raw.as_bytes());
            }
        }
    }
    Ok(())
}

fn hash_canonical_json(hasher: &mut Sha256, value: &serde_json::Value) {
    match value {
        serde_json::Value::Null => hash_field(hasher, b"json:null"),
        serde_json::Value::Bool(value) => {
            hash_field(hasher, b"json:bool");
            hash_field(hasher, if *value { b"1" } else { b"0" });
        }
        serde_json::Value::Number(value) => {
            hash_field(hasher, b"json:number");
            hash_field(hasher, value.to_string().as_bytes());
        }
        serde_json::Value::String(value) => {
            hash_field(hasher, b"json:string");
            hash_field(hasher, value.as_bytes());
        }
        serde_json::Value::Array(values) => {
            hash_field(hasher, b"json:array");
            hash_field(
                hasher,
                &u64::try_from(values.len())
                    .unwrap_or(u64::MAX)
                    .to_le_bytes(),
            );
            for value in values {
                hash_canonical_json(hasher, value);
            }
        }
        serde_json::Value::Object(values) => {
            hash_field(hasher, b"json:object");
            let mut keys = values.keys().collect::<Vec<_>>();
            keys.sort();
            for key in keys {
                hash_field(hasher, key.as_bytes());
                hash_canonical_json(hasher, &values[key]);
            }
        }
    }
}

fn normalized_scoped_trace_bytes(
    families: Option<&BTreeMap<String, Vec<&A12CaseRecallTrace>>>,
) -> Vec<u8> {
    let mut output = Vec::new();
    let Some(families) = families else {
        return output;
    };
    for (stable_family_id, cases) in families {
        append_framed(&mut output, stable_family_id.as_bytes());
        let mut encoded_cases = cases
            .iter()
            .map(|case| encode_a12_case_trace(case))
            .collect::<Vec<_>>();
        encoded_cases.sort();
        output.extend_from_slice(
            &u64::try_from(encoded_cases.len())
                .unwrap_or(u64::MAX)
                .to_le_bytes(),
        );
        for case in encoded_cases {
            append_framed(&mut output, &case);
        }
    }
    output
}

fn encode_a12_case_trace(case: &A12CaseRecallTrace) -> Vec<u8> {
    let mut output = Vec::new();
    append_framed(
        &mut output,
        case.query_type.trim().to_lowercase().as_bytes(),
    );
    encode_optional_u32(&mut output, case.cluster_id);
    match case.valid_until_exclusive {
        Some(boundary) => {
            output.push(1);
            output.extend_from_slice(&boundary.to_le_bytes());
        }
        None => output.push(0),
    }
    output.extend_from_slice(
        &u64::try_from(case.trace.legacy_order.len())
            .unwrap_or(u64::MAX)
            .to_le_bytes(),
    );
    for memory_id in &case.trace.legacy_order {
        append_framed(&mut output, memory_id.as_bytes());
    }

    let event = &case.trace.event;
    append_framed(&mut output, event.request_id.as_bytes());
    output.extend_from_slice(
        &u64::try_from(event.candidates.len())
            .unwrap_or(u64::MAX)
            .to_le_bytes(),
    );
    // Candidate order is intentionally retained: runtime treatment ties keep
    // the legacy order, so it is part of evaluation semantics.
    for candidate in &event.candidates {
        append_framed(&mut output, candidate.memory_id.as_bytes());
        output.extend_from_slice(&candidate.bm25_norm.to_bits().to_le_bytes());
        output.extend_from_slice(&candidate.vec_norm.to_bits().to_le_bytes());
        output.extend_from_slice(&candidate.kg_norm.to_bits().to_le_bytes());
        output.extend_from_slice(&candidate.episode_norm.to_bits().to_le_bytes());
        output.extend_from_slice(&candidate.support_count.to_le_bytes());
        output.extend_from_slice(&candidate.source_diversity.to_bits().to_le_bytes());
    }
    for labels in [&event.accessed_ids, &event.negative_ids] {
        let labels = labels.iter().collect::<BTreeSet<_>>();
        output.extend_from_slice(
            &u64::try_from(labels.len())
                .unwrap_or(u64::MAX)
                .to_le_bytes(),
        );
        for memory_id in labels {
            append_framed(&mut output, memory_id.as_bytes());
        }
    }
    append_framed(
        &mut output,
        event
            .timestamp
            .to_rfc3339_opts(chrono::SecondsFormat::Nanos, true)
            .as_bytes(),
    );
    encode_optional_u32(&mut output, event.query_cluster_id_at_recall);
    match event.cluster_version_at_recall {
        Some(value) => {
            output.push(1);
            output.extend_from_slice(&value.to_le_bytes());
        }
        None => output.push(0),
    }
    match &event.query_top_vec_memory_id_at_recall {
        Some(value) => {
            output.push(1);
            append_framed(&mut output, value.as_bytes());
        }
        None => output.push(0),
    }
    output.extend_from_slice(
        &u64::try_from(case.provenance.canonical_loo)
            .unwrap_or(u64::MAX)
            .to_le_bytes(),
    );
    output.extend_from_slice(
        &u64::try_from(case.provenance.concept_loo)
            .unwrap_or(u64::MAX)
            .to_le_bytes(),
    );
    output.extend_from_slice(
        &u64::try_from(case.provenance.episode_loo)
            .unwrap_or(u64::MAX)
            .to_le_bytes(),
    );
    output
}

fn encode_optional_u32(output: &mut Vec<u8>, value: Option<u32>) {
    match value {
        Some(value) => {
            output.push(1);
            output.extend_from_slice(&value.to_le_bytes());
        }
        None => output.push(0),
    }
}

fn encode_paired_outcomes(outcomes: &[crate::eval::mcnemar::PairedOutcome]) -> Vec<u8> {
    let mut encoded = outcomes
        .iter()
        .map(|outcome| {
            let mut bytes = Vec::new();
            append_framed(&mut bytes, outcome.case_id.as_bytes());
            bytes.push(u8::from(outcome.baseline_hit));
            bytes.push(u8::from(outcome.treatment_hit));
            bytes.extend_from_slice(
                &u64::try_from(outcome.baseline_length)
                    .unwrap_or(u64::MAX)
                    .to_le_bytes(),
            );
            bytes.extend_from_slice(
                &u64::try_from(outcome.treatment_length)
                    .unwrap_or(u64::MAX)
                    .to_le_bytes(),
            );
            bytes
        })
        .collect::<Vec<_>>();
    encoded.sort();
    let mut output = Vec::new();
    for outcome in encoded {
        append_framed(&mut output, &outcome);
    }
    output
}

fn encode_shadow_weights(
    weights: Option<crate::search::alpha_optimizer::ShadowFusionWeights>,
) -> Vec<u8> {
    let mut output = Vec::new();
    let Some(weights) = weights else {
        return output;
    };
    output.push(1);
    for value in [
        weights.bm25,
        weights.vec,
        weights.kg,
        weights.episode,
        weights.support,
        weights.diversity,
    ] {
        output.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    output
}

/// Pure Task-3 output for one existing shadow-fusion scope key.
#[derive(Debug, Clone)]
pub(crate) struct A12ScopeCalibration {
    pub scope: String,
    pub learned_weights: Option<crate::search::alpha_optimizer::ShadowFusionWeights>,
    pub train_family_ess: usize,
    pub train_case_count: usize,
    pub holdout_family_ess: usize,
    pub paired_top3: A12PairedTop3,
    pub mcnemar: crate::eval::mcnemar::McNemarResult,
    pub holdout_status: crate::eval::gates::ScorecardStatus,
    pub holdout_reason: String,
    pub provenance: A12ProvenanceCounts,
    /// Minimum next production-semantics boundary across every train/holdout
    /// case in this scope, expressed as Unix milliseconds.
    pub valid_until_exclusive: Option<i64>,
    pub snapshot_fingerprint: String,
    pub corpus_fingerprint: String,
    pub optimizer_fingerprint: String,
    pub evaluation_fingerprint: String,
    pub calibrated_at: DateTime<Utc>,
}

impl A12ScopeCalibration {
    pub(crate) fn is_current_at(&self, now: DateTime<Utc>) -> bool {
        self.valid_until_exclusive
            .is_none_or(|boundary| now.timestamp_millis() < boundary)
    }
}

#[derive(Debug, Clone)]
struct MemorySnapshot {
    id: String,
    content: String,
    created_at: DateTime<Utc>,
    status: String,
    superseded_by: Option<String>,
}

impl MemorySnapshot {
    fn is_live(&self) -> bool {
        self.superseded_by.is_none() && matches!(self.status.as_str(), "active" | "updated")
    }
}

#[derive(Debug, Clone)]
struct EvidenceView {
    id: String,
    original_memory_id: Option<String>,
    canonical_id: String,
    content: String,
    created_at: DateTime<Utc>,
    imported_at: DateTime<Utc>,
}

#[derive(Debug)]
struct FamilySnapshot {
    families: Vec<A12CanonicalFamily>,
    member_to_family: HashMap<String, String>,
    live_tip_content: HashMap<String, String>,
}

/// Compute the full 256-bit digest modulo five (rather than truncating the
/// digest to a machine integer).
pub(crate) fn a12_family_split_bucket(stable_family_id: &str) -> u8 {
    let digest = Sha256::digest(format!("{A12_FAMILY_PREFIX}{stable_family_id}").as_bytes());
    digest.iter().fold(0u8, |remainder, byte| {
        ((u16::from(remainder) * 256 + u16::from(*byte)) % u16::from(A12_SPLIT_MODULUS)) as u8
    })
}

pub(crate) fn a12_family_fold(stable_family_id: &str) -> A12Fold {
    if a12_family_split_bucket(stable_family_id) == 0 {
        A12Fold::ActivationHoldout
    } else {
        A12Fold::Training
    }
}

/// Learn and evaluate A12 weights for the existing shadow-fusion fallback
/// scopes (`global`, query type, and query type + cluster).
///
/// Fold membership is recomputed from the immutable stable family id here,
/// rather than trusting a caller-supplied flag. The permanent holdout can
/// therefore never enter optimizer input, even if an intermediate record is
/// malformed. Every scope counts distinct families, not trace rows.
pub(crate) fn train_and_evaluate_a12_traces(
    families: &[A12FamilyRecallTrace],
    min_samples_alpha: usize,
    context: &A12CalibrationContext,
) -> Vec<A12ScopeCalibration> {
    type ScopedCases<'a> = BTreeMap<String, BTreeMap<String, Vec<&'a A12CaseRecallTrace>>>;

    let mut training: ScopedCases<'_> = BTreeMap::new();
    let mut holdout: ScopedCases<'_> = BTreeMap::new();
    let mut provenance_by_scope = BTreeMap::<String, A12ProvenanceCounts>::new();
    let mut valid_until_by_scope = BTreeMap::<String, i64>::new();
    for family in families {
        let target = match a12_family_fold(&family.stable_family_id) {
            A12Fold::Training => &mut training,
            A12Fold::ActivationHoldout => &mut holdout,
        };
        for case in &family.cases {
            for scope in a12_scope_keys(case) {
                if let Some(boundary) = case.valid_until_exclusive {
                    valid_until_by_scope
                        .entry(scope.clone())
                        .and_modify(|current| *current = (*current).min(boundary))
                        .or_insert(boundary);
                }
                provenance_by_scope
                    .entry(scope.clone())
                    .or_default()
                    .add_assign(case.provenance);
                target
                    .entry(scope)
                    .or_default()
                    .entry(family.stable_family_id.clone())
                    .or_default()
                    .push(case);
            }
        }
    }

    let scopes = training
        .keys()
        .chain(holdout.keys())
        .cloned()
        .collect::<BTreeSet<_>>();
    let required = min_samples_alpha.max(10);
    let mut calibrations = Vec::with_capacity(scopes.len());

    for scope in scopes {
        let training_families = training.get(&scope);
        let optimizer_input = training_families
            .into_iter()
            .flat_map(|families| families.iter())
            .map(|(stable_family_id, cases)| {
                crate::search::alpha_optimizer::FamilyShadowObservation {
                    stable_family_id: stable_family_id.clone(),
                    events: cases.iter().map(|case| case.trace.event.clone()).collect(),
                }
            })
            .collect::<Vec<_>>();
        let learned =
            crate::search::alpha_optimizer::optimize_family_equal_shadow_weights(&optimizer_input);
        let learned_weights = learned.map(|value| value.weights);
        let train_family_ess = learned.map_or(0, |value| value.family_ess);
        let train_case_count = learned.map_or(0, |value| value.case_count);

        let holdout_families = holdout.get(&scope);
        let outcomes = learned_weights
            .into_iter()
            .flat_map(|weights| {
                holdout_families.into_iter().flat_map(move |families| {
                    families.iter().filter_map(move |(family_id, cases)| {
                        family_top3_outcome(family_id, cases, weights)
                    })
                })
            })
            .collect::<Vec<_>>();
        // Persistence validates family ESS against McNemar's paired n. A
        // family only enters ESS after both baseline and the learned treatment
        // have a defined aggregate outcome.
        let holdout_family_ess = outcomes.len();
        let paired_top3 = paired_top3_counts(&outcomes);
        let mut mcnemar = crate::eval::mcnemar::mcnemar(&outcomes);

        let (holdout_status, holdout_reason) = if learned_weights.is_none() {
            (
                crate::eval::gates::ScorecardStatus::NoData,
                "NoData: no informative training-family simplex".to_string(),
            )
        } else if train_family_ess < required || holdout_family_ess < required {
            (
                crate::eval::gates::ScorecardStatus::NoData,
                format!(
                    "NoData: family ESS below required={required} \
                     (train={train_family_ess}, holdout={holdout_family_ess})"
                ),
            )
        } else {
            let comparison = compare_a12_family_top3(&scope, &outcomes);
            if let Some(result) = comparison.mcnemar {
                mcnemar = result;
            }
            (comparison.status, comparison.reason)
        };

        let training_trace_bytes = normalized_scoped_trace_bytes(training_families);
        let learned_weight_bytes = encode_shadow_weights(learned_weights);
        let optimizer_fingerprint = fingerprint_framed(
            b"a12-family-equal-optimizer-scope-v2\0",
            &[
                context.optimizer_fingerprint.as_bytes(),
                context.corpus_fingerprint.as_bytes(),
                scope.as_bytes(),
                &training_trace_bytes,
                &learned_weight_bytes,
            ],
        );
        let holdout_trace_bytes = normalized_scoped_trace_bytes(holdout_families);
        let outcome_bytes = encode_paired_outcomes(&outcomes);
        let evaluation_fingerprint = fingerprint_framed(
            b"a12-family-top3-evaluation-scope-v2\0",
            &[
                context.evaluation_fingerprint.as_bytes(),
                context.corpus_fingerprint.as_bytes(),
                optimizer_fingerprint.as_bytes(),
                scope.as_bytes(),
                &holdout_trace_bytes,
                &outcome_bytes,
            ],
        );

        let provenance = provenance_by_scope.get(&scope).copied().unwrap_or_default();
        let valid_until_exclusive = valid_until_by_scope.get(&scope).copied();
        calibrations.push(A12ScopeCalibration {
            scope,
            learned_weights,
            train_family_ess,
            train_case_count,
            holdout_family_ess,
            paired_top3,
            mcnemar,
            holdout_status,
            holdout_reason,
            provenance,
            valid_until_exclusive,
            snapshot_fingerprint: context.snapshot_fingerprint.clone(),
            corpus_fingerprint: context.corpus_fingerprint.clone(),
            optimizer_fingerprint,
            evaluation_fingerprint,
            calibrated_at: context.calibrated_at,
        });
    }

    calibrations
}

fn a12_scope_keys(case: &A12CaseRecallTrace) -> Vec<String> {
    let query_type = case.query_type.trim().to_lowercase();
    let mut scopes = vec!["global".to_string()];
    if query_type.is_empty() {
        return scopes;
    }
    scopes.push(query_type.clone());
    let trace_cluster = case.trace.event.query_cluster_id_at_recall;
    if let Some(cluster_id) = case.cluster_id.filter(|id| Some(*id) == trace_cluster) {
        scopes.push(format!("{query_type}:{cluster_id}"));
    }
    scopes
}

fn family_top3_outcome(
    family_id: &str,
    cases: &[&A12CaseRecallTrace],
    weights: crate::search::alpha_optimizer::ShadowFusionWeights,
) -> Option<crate::eval::mcnemar::PairedOutcome> {
    let valid_cases = cases
        .iter()
        .filter(|case| !case.trace.event.accessed_ids.is_empty())
        .collect::<Vec<_>>();
    if valid_cases.is_empty() {
        return None;
    }

    let baseline_hits = valid_cases
        .iter()
        .filter(|case| legacy_top3_hit(&case.trace))
        .count();
    let treatment_hits = valid_cases
        .iter()
        .filter(|case| treatment_top3_hit(&case.trace.event, weights))
        .count();
    let family_hit = |hits: usize| hits.saturating_mul(2) >= valid_cases.len();

    Some(crate::eval::mcnemar::PairedOutcome {
        case_id: family_id.to_string(),
        baseline_hit: family_hit(baseline_hits),
        treatment_hit: family_hit(treatment_hits),
        baseline_length: 3.min(valid_cases[0].trace.legacy_order.len()),
        treatment_length: 3.min(valid_cases[0].trace.event.candidates.len()),
        treatment_summary: None,
    })
}

fn legacy_top3_hit(trace: &crate::search::recall::A12RecallTrace) -> bool {
    trace
        .legacy_order
        .iter()
        .take(3)
        .any(|memory_id| trace.event.accessed_ids.contains(memory_id))
}

fn treatment_top3_hit(
    event: &crate::search::alpha_optimizer::RecallEvent,
    weights: crate::search::alpha_optimizer::ShadowFusionWeights,
) -> bool {
    let mut scored = event
        .candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| {
            (
                crate::search::alpha_optimizer::score_candidate_with_shadow_weights(
                    candidate, weights,
                ),
                index,
                candidate.memory_id.as_str(),
            )
        })
        .collect::<Vec<_>>();
    scored.sort_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left.1.cmp(&right.1))
    });
    scored
        .iter()
        .take(3)
        .any(|(_score, _index, memory_id)| event.accessed_ids.iter().any(|id| id == memory_id))
}

fn paired_top3_counts(outcomes: &[crate::eval::mcnemar::PairedOutcome]) -> A12PairedTop3 {
    let mut counts = A12PairedTop3::default();
    for outcome in outcomes {
        match (outcome.baseline_hit, outcome.treatment_hit) {
            (true, true) => counts.both_hit += 1,
            (true, false) => counts.baseline_only += 1,
            (false, true) => counts.treatment_only += 1,
            (false, false) => counts.neither_hit += 1,
        }
    }
    counts
}

fn compare_a12_family_top3(
    scope: &str,
    outcomes: &[crate::eval::mcnemar::PairedOutcome],
) -> crate::eval::gates::GateComparison {
    use crate::eval::gates::{
        compare_scorecards, FixtureResult, GateScorecard, ScorecardKind, DEFAULT_NOISE_FLOOR,
        SCORECARD_SCHEMA_VERSION,
    };

    let gate_name = format!("a12_holdout:{scope}");
    let fixture_fingerprint = format!("a12-family-holdout:{scope}");
    let scorecard = |kind: ScorecardKind, treatment: bool| {
        let per_fixture = outcomes
            .iter()
            .map(|outcome| FixtureResult {
                fixture_id: outcome.case_id.clone(),
                hit: if treatment {
                    outcome.treatment_hit
                } else {
                    outcome.baseline_hit
                },
            })
            .collect::<Vec<_>>();
        let hits = per_fixture.iter().filter(|fixture| fixture.hit).count();
        GateScorecard {
            schema_version: SCORECARD_SCHEMA_VERSION,
            gate_name: gate_name.clone(),
            kind,
            created_at: 0,
            rein_version: env!("CARGO_PKG_VERSION").to_string(),
            build_fingerprint: env!("REIN_BUILD_FINGERPRINT").to_string(),
            fixture_fingerprint: fixture_fingerprint.clone(),
            fixture_count: per_fixture.len(),
            score: hits as f64 / per_fixture.len().max(1) as f64,
            per_fixture,
        }
    };
    let baseline = scorecard(ScorecardKind::Baseline, false);
    let treatment = scorecard(ScorecardKind::Run, true);
    compare_scorecards(
        &gate_name,
        Some(&baseline),
        Some(&treatment),
        DEFAULT_NOISE_FLOOR,
    )
}

/// Enumerate canonical/supersede families using the earliest
/// `(created_at, memory_id)` member as the stable identity.
pub(crate) fn enumerate_stable_root_families(
    store: &SqliteStore,
) -> ReinResult<Vec<A12CanonicalFamily>> {
    Ok(load_family_snapshot(store)?.families)
}

/// Build leakage-safe LOO cases without mutating the store.
pub(crate) fn build_a12_loo_corpus(
    store: &SqliteStore,
    hard_dedup_bound: f32,
) -> ReinResult<A12LooCorpus> {
    store.conn().execute_batch("BEGIN DEFERRED")?;
    let built = (|| {
        let before = a12_local_recall_snapshot_identity(store)?;
        let corpus = build_a12_loo_corpus_inner(store, hard_dedup_bound)?;
        let after = a12_local_recall_snapshot_identity(store)?;
        if after != before {
            return Err(ReinError::Config(format!(
                "A12 local recall snapshot drifted during corpus build: before={before} after={after}"
            )));
        }
        Ok((corpus, before))
    })();
    let (mut corpus, snapshot) = match built {
        Ok(value) => {
            store.conn().execute_batch("COMMIT")?;
            value
        }
        Err(error) => {
            let _ = store.conn().execute_batch("ROLLBACK");
            return Err(error);
        }
    };
    let live_after_commit = read_a12_local_recall_snapshot_identity(store)?;
    if live_after_commit != snapshot {
        return Err(ReinError::Config(format!(
            "A12 local recall snapshot advanced before corpus finalization: build={snapshot} live={live_after_commit}"
        )));
    }
    corpus.source_snapshot_fingerprint = snapshot;
    Ok(corpus)
}

fn build_a12_loo_corpus_inner(
    store: &SqliteStore,
    hard_dedup_bound: f32,
) -> ReinResult<A12LooCorpus> {
    if !hard_dedup_bound.is_finite() || !(0.0..=1.0).contains(&hard_dedup_bound) {
        return Err(ReinError::Config(format!(
            "A12 hard dedup bound must be finite and in [0, 1], got {hard_dedup_bound}"
        )));
    }

    let snapshot = load_family_snapshot(store)?;
    let family_by_id: HashMap<&str, &A12CanonicalFamily> = snapshot
        .families
        .iter()
        .map(|family| (family.stable_family_id.as_str(), family))
        .collect();
    let mut evidence_by_family = load_evidence_views(store, &snapshot.member_to_family)?;
    let auxiliary = load_auxiliary_family_links(store, &snapshot.member_to_family)?;

    let live_candidates: Vec<(&A12CanonicalFamily, &str, &str)> = snapshot
        .families
        .iter()
        .filter_map(|family| {
            let live_tip_id = family.live_tip_id.as_deref()?;
            let content = snapshot.live_tip_content.get(live_tip_id)?;
            Some((family, live_tip_id, content.as_str()))
        })
        .collect();

    let mut observations = Vec::new();
    let mut abstentions = Vec::new();

    for family in &snapshot.families {
        let views = evidence_by_family
            .remove(&family.stable_family_id)
            .unwrap_or_default();
        if views.is_empty() {
            abstentions.push(A12LooAbstention {
                stable_family_id: family.stable_family_id.clone(),
                held_out_evidence_id: None,
                original_memory_id: None,
                exclusion: None,
                reason: A12AbstentionReason::NoEvidenceViews,
            });
            continue;
        }

        let mut cases = Vec::new();
        for view in views {
            let exclusion = loo_exclusion(&view, &live_candidates, hard_dedup_bound);
            let Some(canonical_live_tip_id) = family.live_tip_id.as_deref() else {
                abstentions.push(A12LooAbstention {
                    stable_family_id: family.stable_family_id.clone(),
                    held_out_evidence_id: Some(view.id),
                    original_memory_id: view.original_memory_id,
                    exclusion: Some(exclusion),
                    reason: A12AbstentionReason::NoLiveCanonicalTip,
                });
                continue;
            };
            let mut positives: BTreeMap<(String, String), BTreeSet<A12OutcomeProvenance>> =
                BTreeMap::new();

            let canonical_reason = match exclusion.reason_for(canonical_live_tip_id) {
                Some(reason) => Some(reason),
                None => {
                    positives
                        .entry((
                            family.stable_family_id.clone(),
                            canonical_live_tip_id.to_string(),
                        ))
                        .or_default()
                        .insert(A12OutcomeProvenance::CanonicalLoo);
                    None
                }
            };

            let mut saw_cross_fold_auxiliary = false;
            if let Some(auxiliary_families) = auxiliary.get(&family.stable_family_id) {
                for (auxiliary_family_id, provenance) in auxiliary_families {
                    let Some(auxiliary_family) = family_by_id.get(auxiliary_family_id.as_str())
                    else {
                        continue;
                    };
                    if auxiliary_family.fold != family.fold {
                        saw_cross_fold_auxiliary = true;
                        continue;
                    }
                    let Some(live_tip_id) = auxiliary_family.live_tip_id.as_deref() else {
                        continue;
                    };
                    if exclusion.reason_for(live_tip_id).is_some() {
                        continue;
                    }
                    positives
                        .entry((auxiliary_family_id.clone(), live_tip_id.to_string()))
                        .or_default()
                        .extend(provenance.iter().copied());
                }
            }

            if positives.is_empty() {
                abstentions.push(A12LooAbstention {
                    stable_family_id: family.stable_family_id.clone(),
                    held_out_evidence_id: Some(view.id),
                    original_memory_id: view.original_memory_id,
                    exclusion: Some(exclusion),
                    reason: canonical_reason.unwrap_or(if saw_cross_fold_auxiliary {
                        A12AbstentionReason::CrossFoldAuxiliary
                    } else {
                        A12AbstentionReason::NoIndependentPositive
                    }),
                });
                continue;
            }

            let positives = positives
                .into_iter()
                .map(
                    |((stable_family_id, live_tip_id), provenance)| A12LooPositive {
                        stable_family_id,
                        live_tip_id,
                        provenance: provenance.into_iter().collect(),
                    },
                )
                .collect();
            cases.push(A12LooCase {
                held_out_evidence_id: view.id,
                original_memory_id: view.original_memory_id,
                query_text: view.content,
                exclusion,
                positives,
            });
        }

        if let Some(live_tip_id) = family.live_tip_id.as_ref().filter(|_| !cases.is_empty()) {
            observations.push(A12FamilyObservation {
                stable_family_id: family.stable_family_id.clone(),
                live_tip_id: live_tip_id.clone(),
                split_bucket: family.split_bucket,
                fold: family.fold,
                family_weight: 1.0,
                cases,
            });
        }
    }

    Ok(A12LooCorpus {
        source_snapshot_fingerprint: String::new(),
        hard_dedup_bound,
        observations,
        abstentions,
    })
}

impl A12LooExclusion {
    fn reason_for(&self, memory_id: &str) -> Option<A12AbstentionReason> {
        if self
            .held_out_memory_ids
            .binary_search_by(|candidate| candidate.as_str().cmp(memory_id))
            .is_ok()
        {
            return Some(A12AbstentionReason::HeldOutMemory);
        }
        if self
            .equal_content_memory_ids
            .binary_search_by(|candidate| candidate.as_str().cmp(memory_id))
            .is_ok()
        {
            return Some(A12AbstentionReason::EqualContentHash);
        }
        if self
            .near_duplicate_memory_ids
            .binary_search_by(|candidate| candidate.as_str().cmp(memory_id))
            .is_ok()
        {
            return Some(A12AbstentionReason::NearDuplicateContent);
        }
        None
    }
}

fn load_family_snapshot(store: &SqliteStore) -> ReinResult<FamilySnapshot> {
    let memories = load_memory_snapshots(store)?;
    let mut by_live_tip: BTreeMap<String, Vec<MemorySnapshot>> = BTreeMap::new();
    for memory in memories {
        let live_tip_id = store.canonical_id_for(&memory.id)?;
        by_live_tip.entry(live_tip_id).or_default().push(memory);
    }

    let mut assembled = Vec::with_capacity(by_live_tip.len());
    for (resolved_tip_id, mut members) in by_live_tip {
        members.sort_by(|left, right| {
            left.created_at
                .cmp(&right.created_at)
                .then_with(|| left.id.cmp(&right.id))
        });
        let root = members
            .first()
            .expect("a canonical family is created from at least one memory");
        let live_tip = members
            .iter()
            .find(|member| member.id == resolved_tip_id && member.is_live());
        let stable_family_id = root.id.clone();
        let split_bucket = a12_family_split_bucket(&stable_family_id);
        let family = A12CanonicalFamily {
            stable_family_id,
            stable_created_at: root.created_at,
            split_bucket,
            fold: if split_bucket == 0 {
                A12Fold::ActivationHoldout
            } else {
                A12Fold::Training
            },
            live_tip_id: live_tip.map(|tip| tip.id.clone()),
            member_ids: members.iter().map(|member| member.id.clone()).collect(),
        };
        assembled.push((family, live_tip.map(|tip| tip.content.clone())));
    }
    assembled.sort_by(|(left, _), (right, _)| {
        left.stable_created_at
            .cmp(&right.stable_created_at)
            .then_with(|| left.stable_family_id.cmp(&right.stable_family_id))
    });

    let mut member_to_family = HashMap::new();
    let mut live_tip_content = HashMap::new();
    let mut families = Vec::with_capacity(assembled.len());
    for (family, content) in assembled {
        for member_id in &family.member_ids {
            member_to_family.insert(member_id.clone(), family.stable_family_id.clone());
        }
        if let (Some(live_tip_id), Some(content)) = (&family.live_tip_id, content) {
            live_tip_content.insert(live_tip_id.clone(), content);
        }
        families.push(family);
    }

    Ok(FamilySnapshot {
        families,
        member_to_family,
        live_tip_content,
    })
}

fn load_memory_snapshots(store: &SqliteStore) -> ReinResult<Vec<MemorySnapshot>> {
    let mut statement = store.conn().prepare(
        "SELECT id, content, created_at, status, superseded_by \
         FROM memories ORDER BY created_at, id",
    )?;
    let rows = statement.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, Option<String>>(4)?,
        ))
    })?;

    let mut memories = Vec::new();
    for row in rows {
        let (id, content, created_at, status, superseded_by) = row?;
        memories.push(MemorySnapshot {
            id,
            content,
            created_at: parse_timestamp(&created_at, "memories.created_at")?,
            status,
            superseded_by,
        });
    }
    Ok(memories)
}

fn load_evidence_views(
    store: &SqliteStore,
    member_to_family: &HashMap<String, String>,
) -> ReinResult<BTreeMap<String, Vec<EvidenceView>>> {
    let mut statement = store.conn().prepare(
        "SELECT id, memory_id, canonical_id, content, created_at, imported_at \
         FROM memory_evidence ORDER BY created_at, imported_at, id",
    )?;
    let rows = statement.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, Option<String>>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, String>(4)?,
            row.get::<_, String>(5)?,
        ))
    })?;

    let mut by_family: BTreeMap<String, Vec<EvidenceView>> = BTreeMap::new();
    for row in rows {
        let (id, original_memory_id, canonical_id, content, created_at, imported_at) = row?;
        let stable_family_id = original_memory_id
            .as_ref()
            .and_then(|memory_id| member_to_family.get(memory_id))
            .or_else(|| member_to_family.get(&canonical_id));
        let Some(stable_family_id) = stable_family_id else {
            continue;
        };
        by_family
            .entry(stable_family_id.clone())
            .or_default()
            .push(EvidenceView {
                id,
                original_memory_id,
                canonical_id,
                content,
                created_at: parse_timestamp(&created_at, "memory_evidence.created_at")?,
                imported_at: parse_timestamp(&imported_at, "memory_evidence.imported_at")?,
            });
    }
    for views in by_family.values_mut() {
        views.sort_by(|left, right| {
            left.created_at
                .cmp(&right.created_at)
                .then_with(|| left.imported_at.cmp(&right.imported_at))
                .then_with(|| left.canonical_id.cmp(&right.canonical_id))
                .then_with(|| left.id.cmp(&right.id))
        });
    }
    Ok(by_family)
}

type AuxiliaryFamilyLinks = BTreeMap<String, BTreeMap<String, BTreeSet<A12OutcomeProvenance>>>;

fn load_auxiliary_family_links(
    store: &SqliteStore,
    member_to_family: &HashMap<String, String>,
) -> ReinResult<AuxiliaryFamilyLinks> {
    let mut links = BTreeMap::new();

    let mut concept_statement = store
        .conn()
        .prepare("SELECT source_memory_ids FROM concepts ORDER BY id")?;
    let concept_rows = concept_statement.query_map([], |row| row.get::<_, String>(0))?;
    for row in concept_rows {
        add_auxiliary_group(
            &mut links,
            &serde_json::from_str::<Vec<String>>(&row?)?,
            member_to_family,
            A12OutcomeProvenance::ConceptLoo,
        );
    }

    let mut episode_statement = store
        .conn()
        .prepare("SELECT memory_ids FROM episodes ORDER BY id")?;
    let episode_rows = episode_statement.query_map([], |row| row.get::<_, String>(0))?;
    for row in episode_rows {
        add_auxiliary_group(
            &mut links,
            &serde_json::from_str::<Vec<String>>(&row?)?,
            member_to_family,
            A12OutcomeProvenance::EpisodeLoo,
        );
    }

    Ok(links)
}

fn add_auxiliary_group(
    links: &mut AuxiliaryFamilyLinks,
    memory_ids: &[String],
    member_to_family: &HashMap<String, String>,
    provenance: A12OutcomeProvenance,
) {
    let family_ids: Vec<String> = memory_ids
        .iter()
        .filter_map(|memory_id| member_to_family.get(memory_id).cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    for source_family_id in &family_ids {
        for target_family_id in &family_ids {
            if source_family_id == target_family_id {
                continue;
            }
            links
                .entry(source_family_id.clone())
                .or_default()
                .entry(target_family_id.clone())
                .or_default()
                .insert(provenance);
        }
    }
}

fn loo_exclusion(
    view: &EvidenceView,
    live_candidates: &[(&A12CanonicalFamily, &str, &str)],
    hard_dedup_bound: f32,
) -> A12LooExclusion {
    let mut held_out_memory_ids = view.original_memory_id.iter().cloned().collect::<Vec<_>>();
    held_out_memory_ids.sort();
    let held_out_evidence_ids = vec![view.id.clone()];
    let content_hash = sha256_hex(&view.content);
    let mut equal_content_memory_ids = Vec::new();
    let mut near_duplicate_memory_ids = Vec::new();

    for (_, live_tip_id, content) in live_candidates {
        if sha256_hex(content) == content_hash {
            equal_content_memory_ids.push((*live_tip_id).to_string());
        }
        if similarity(&view.content, content) >= hard_dedup_bound {
            near_duplicate_memory_ids.push((*live_tip_id).to_string());
        }
    }
    equal_content_memory_ids.sort();
    equal_content_memory_ids.dedup();
    near_duplicate_memory_ids.sort();
    near_duplicate_memory_ids.dedup();

    A12LooExclusion {
        held_out_memory_ids,
        held_out_evidence_ids,
        content_hash,
        equal_content_memory_ids,
        near_duplicate_memory_ids,
    }
}

fn sha256_hex(content: &str) -> String {
    format!("{:x}", Sha256::digest(content.as_bytes()))
}

fn parse_timestamp(value: &str, field: &str) -> ReinResult<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(value)
        .map(|timestamp| timestamp.with_timezone(&Utc))
        .map_err(|error| ReinError::Config(format!("invalid {field} timestamp '{value}': {error}")))
}

#[cfg(test)]
mod tests {
    use chrono::{TimeZone, Utc};

    use super::*;
    use crate::store::SqliteStore;
    use crate::types::{
        Importance, Memory, MemoryEvidence, MemoryLayer, MemoryStatus, MemoryStore, MemoryTier,
        Source,
    };

    fn memory(id: &str, content: &str) -> Memory {
        Memory {
            id: id.to_string(),
            layer: MemoryLayer::LTM,
            topic: "a12-test".to_string(),
            summary: content.to_string(),
            content: content.to_string(),
            keywords: Vec::new(),
            importance: Importance::High,
            source: Source::Manual,
            strength: 1.0,
            decay_lambda: 0.01,
            access_count: 0,
            superseded_by: None,
            canonical_id: None,
            support_count: 1,
            merge_count: 0,
            dedup_confidence: 1.0,
            source_diversity: 1.0,
            contradiction_score: 0.0,
            related_ids: Vec::new(),
            concept_ids: Vec::new(),
            status: MemoryStatus::Active,
            embedding: None,
            tier: MemoryTier::Warm,
            cluster_id: None,
            archival_summary: None,
            archival_summary_at: None,
            archival_summary_version: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_accessed: Utc::now(),
        }
    }

    fn set_created_at(store: &SqliteStore, id: &str, timestamp: &str) {
        store
            .conn()
            .execute(
                "UPDATE memories SET created_at = ?2 WHERE id = ?1",
                rusqlite::params![id, timestamp],
            )
            .unwrap();
    }

    fn id_for_fold(prefix: &str, fold: A12Fold) -> String {
        (0..10_000)
            .map(|suffix| format!("{prefix}-{suffix}"))
            .find(|candidate| a12_family_fold(candidate) == fold)
            .expect("a mod-5 split must yield the requested side")
    }

    fn insert_auxiliary_group(store: &SqliteStore, memory_ids: &[&str]) {
        let now = Utc::now().to_rfc3339();
        let memory_ids = memory_ids
            .iter()
            .map(|id| (*id).to_string())
            .collect::<Vec<_>>();
        let source_json = serde_json::to_string(&memory_ids).unwrap();
        store
            .conn()
            .execute(
                "INSERT INTO memoirs (id, name, created_at, updated_at) \
                 VALUES ('a12-memoir', 'a12-memoir', ?1, ?1)",
                rusqlite::params![&now],
            )
            .unwrap();
        store
            .conn()
            .execute(
                "INSERT INTO concepts (id, memoir_id, name, definition, source_memory_ids, \
                                       created_at, updated_at) \
                 VALUES ('a12-concept', 'a12-memoir', 'A12', 'shared support', ?1, ?2, ?2)",
                rusqlite::params![&source_json, &now],
            )
            .unwrap();
        store
            .conn()
            .execute(
                "INSERT INTO episodes (id, title, memory_ids, created_at) \
                 VALUES ('a12-episode', 'A12', ?1, ?2)",
                rusqlite::params![&source_json, &now],
            )
            .unwrap();
    }

    #[test]
    fn canonical_loo_abstains_when_only_positive_leaks_heldout_content() {
        let store = SqliteStore::in_memory().unwrap();
        let root_id = "family-root";
        let live_tip_id = "family-live-tip";
        let leaked_content = "identical held-out content must never become its own label";

        store.store(memory(root_id, leaked_content)).unwrap();
        store.store(memory(live_tip_id, leaked_content)).unwrap();
        store.mark_superseded(root_id, live_tip_id).unwrap();

        let corpus = build_a12_loo_corpus(&store, 0.70).unwrap();
        assert!(corpus.observations.is_empty());
        let root_abstention = corpus
            .abstentions
            .iter()
            .find(|abstention| abstention.original_memory_id.as_deref() == Some(root_id))
            .expect("the held-out root evidence must abstain explicitly");
        assert_eq!(
            root_abstention.reason,
            A12AbstentionReason::EqualContentHash
        );
        let exclusion = root_abstention
            .exclusion
            .as_ref()
            .expect("abstentions must retain their auditable exclusion set");
        assert_eq!(exclusion.content_hash.len(), 64);
        assert_eq!(
            exclusion.equal_content_memory_ids,
            vec![live_tip_id.to_string()]
        );
        assert!(exclusion
            .near_duplicate_memory_ids
            .contains(&live_tip_id.to_string()));
        let live_tip_abstention = corpus
            .abstentions
            .iter()
            .find(|abstention| abstention.original_memory_id.as_deref() == Some(live_tip_id))
            .expect("the live-tip self view must also abstain");
        assert_eq!(
            live_tip_abstention.reason,
            A12AbstentionReason::HeldOutMemory
        );
    }

    #[test]
    fn family_split_uses_full_sha256_mod_five() {
        assert_eq!(a12_family_split_bucket("holdout-4"), 0);
        assert_eq!(a12_family_split_bucket("family-root"), 1);
        assert_eq!(a12_family_split_bucket("alpha"), 3);
        assert_eq!(a12_family_fold("holdout-4"), A12Fold::ActivationHoldout);
        assert_eq!(a12_family_fold("family-root"), A12Fold::Training);
    }

    #[test]
    fn stable_family_root_and_fold_survive_live_tip_changes() {
        let store = SqliteStore::in_memory().unwrap();
        let root_id = "stable-root";
        let middle_id = "middle-tip";
        let final_id = "final-tip";
        store.store(memory(root_id, "root evidence")).unwrap();
        store
            .store(memory(middle_id, "middle canonical text"))
            .unwrap();
        set_created_at(&store, root_id, "2026-01-01T00:00:00Z");
        set_created_at(&store, middle_id, "2026-02-01T00:00:00Z");
        store.mark_superseded(root_id, middle_id).unwrap();

        let before = enumerate_stable_root_families(&store).unwrap();
        assert_eq!(before.len(), 1);
        let family_before = &before[0];
        assert_eq!(family_before.stable_family_id, root_id);
        assert_eq!(family_before.live_tip_id.as_deref(), Some(middle_id));
        let original_bucket = family_before.split_bucket;
        let original_fold = family_before.fold;

        store
            .store(memory(final_id, "final canonical text"))
            .unwrap();
        set_created_at(&store, final_id, "2026-03-01T00:00:00Z");
        store.mark_superseded(middle_id, final_id).unwrap();

        let after = enumerate_stable_root_families(&store).unwrap();
        assert_eq!(after.len(), 1);
        let family_after = &after[0];
        assert_eq!(family_after.stable_family_id, root_id);
        assert_eq!(family_after.live_tip_id.as_deref(), Some(final_id));
        assert_eq!(family_after.split_bucket, original_bucket);
        assert_eq!(family_after.fold, original_fold);
        assert_eq!(family_after.member_ids, vec![root_id, middle_id, final_id]);
    }

    #[test]
    fn stable_family_root_tie_breaks_created_at_by_memory_id() {
        let store = SqliteStore::in_memory().unwrap();
        let lexically_later = "z-member";
        let lexically_earlier = "a-member";
        store
            .store(memory(lexically_later, "historical member"))
            .unwrap();
        store
            .store(memory(lexically_earlier, "current member"))
            .unwrap();
        let tied = "2026-01-01T00:00:00Z";
        set_created_at(&store, lexically_later, tied);
        set_created_at(&store, lexically_earlier, tied);
        store
            .mark_superseded(lexically_later, lexically_earlier)
            .unwrap();

        let families = enumerate_stable_root_families(&store).unwrap();
        assert_eq!(families.len(), 1);
        assert_eq!(families[0].stable_family_id, lexically_earlier);
        assert_eq!(
            families[0].member_ids,
            vec![lexically_earlier, lexically_later]
        );
    }

    #[test]
    fn concept_and_episode_auxiliaries_never_cross_train_holdout_split() {
        let store = SqliteStore::in_memory().unwrap();
        let root_id = id_for_fold("base-holdout", A12Fold::ActivationHoldout);
        let live_tip_id = "base-live-tip";
        let same_side_id = id_for_fold("same-holdout", A12Fold::ActivationHoldout);
        let opposite_side_id = id_for_fold("training-auxiliary", A12Fold::Training);

        store
            .store(memory(&root_id, "orchid greenhouse humidity notes"))
            .unwrap();
        store
            .store(memory(
                live_tip_id,
                "database transaction durability policy",
            ))
            .unwrap();
        store
            .store(memory(
                &same_side_id,
                "satellite orbital telemetry handbook",
            ))
            .unwrap();
        store
            .store(memory(
                &opposite_side_id,
                "culinary sourdough fermentation guide",
            ))
            .unwrap();
        store.mark_superseded(&root_id, live_tip_id).unwrap();
        insert_auxiliary_group(
            &store,
            &[&root_id, same_side_id.as_str(), opposite_side_id.as_str()],
        );

        let corpus = build_a12_loo_corpus(&store, 0.95).unwrap();
        let observation = corpus
            .observations
            .iter()
            .find(|observation| observation.stable_family_id == root_id)
            .expect("the base family has a leakage-free canonical case");
        let case = observation
            .cases
            .iter()
            .find(|case| case.original_memory_id.as_deref() == Some(root_id.as_str()))
            .expect("root evidence is a query view");
        let same_side = case
            .positives
            .iter()
            .find(|positive| positive.stable_family_id == same_side_id)
            .expect("same-side auxiliary support is retained");
        assert_eq!(
            same_side.provenance,
            vec![
                A12OutcomeProvenance::ConceptLoo,
                A12OutcomeProvenance::EpisodeLoo
            ]
        );
        assert!(case
            .positives
            .iter()
            .all(|positive| positive.stable_family_id != opposite_side_id));
    }

    #[test]
    fn near_duplicate_live_tip_abstains_without_equal_hash() {
        let store = SqliteStore::in_memory().unwrap();
        let root_id = "near-root";
        let live_tip_id = "near-live-tip";
        let held_out = "Rust borrow checker prevents dangling references";
        let live_tip = "Rust borrow checker prevents a dangling reference";
        assert_ne!(sha256_hex(held_out), sha256_hex(live_tip));
        let exact_bound = similarity(held_out, live_tip);
        assert!(exact_bound > 0.0 && exact_bound < 1.0);

        store.store(memory(root_id, held_out)).unwrap();
        store.store(memory(live_tip_id, live_tip)).unwrap();
        store.mark_superseded(root_id, live_tip_id).unwrap();

        let corpus = build_a12_loo_corpus(&store, exact_bound).unwrap();
        let abstention = corpus
            .abstentions
            .iter()
            .find(|abstention| abstention.original_memory_id.as_deref() == Some(root_id))
            .unwrap();
        assert_eq!(abstention.reason, A12AbstentionReason::NearDuplicateContent);
        let exclusion = abstention.exclusion.as_ref().unwrap();
        assert!(!exclusion
            .equal_content_memory_ids
            .contains(&live_tip_id.to_string()));
        assert!(exclusion
            .near_duplicate_memory_ids
            .contains(&live_tip_id.to_string()));
    }

    #[test]
    fn multiple_views_still_emit_one_equal_weight_family_observation() {
        let store = SqliteStore::in_memory().unwrap();
        let root_id = "multi-view-root";
        let live_tip_id = "multi-view-tip";
        store
            .store(memory(root_id, "rust ownership borrowing lifetimes"))
            .unwrap();
        store
            .store(memory(
                live_tip_id,
                "postgres transaction isolation durable commits",
            ))
            .unwrap();
        store.mark_superseded(root_id, live_tip_id).unwrap();
        store
            .add_memory_evidence(MemoryEvidence {
                id: "extra-evidence".to_string(),
                canonical_id: root_id.to_string(),
                memory_id: Some(root_id.to_string()),
                source_topic: "a12-test".to_string(),
                summary: "gardening evidence".to_string(),
                content: "tomato seedlings sunlight irrigation schedule".to_string(),
                keywords: Vec::new(),
                source: Source::Manual,
                created_at: Utc.with_ymd_and_hms(2025, 1, 1, 0, 0, 0).unwrap(),
                imported_at: Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap(),
            })
            .unwrap();

        let corpus = build_a12_loo_corpus(&store, 0.70).unwrap();
        assert_eq!(corpus.observations.len(), 1);
        let observation = &corpus.observations[0];
        assert_eq!(observation.stable_family_id, root_id);
        assert_eq!(observation.family_weight, 1.0);
        assert_eq!(observation.cases.len(), 2);
        assert!(observation
            .cases
            .iter()
            .any(|case| case.held_out_evidence_id == "extra-evidence"));
        assert!(observation.cases.iter().all(|case| {
            case.original_memory_id.as_deref() == Some(root_id)
                && case
                    .positives
                    .iter()
                    .any(|positive| positive.live_tip_id == live_tip_id)
        }));
    }

    #[test]
    fn family_and_corpus_ordering_are_deterministic() {
        let store = SqliteStore::in_memory().unwrap();
        store
            .store(memory("inserted-first", "first inserted content"))
            .unwrap();
        store
            .store(memory("chronologically-first", "second inserted content"))
            .unwrap();
        set_created_at(&store, "inserted-first", "2026-02-01T00:00:00Z");
        set_created_at(&store, "chronologically-first", "2026-01-01T00:00:00Z");

        let families_once = enumerate_stable_root_families(&store).unwrap();
        let families_twice = enumerate_stable_root_families(&store).unwrap();
        assert_eq!(families_once, families_twice);
        assert_eq!(
            families_once
                .iter()
                .map(|family| family.stable_family_id.as_str())
                .collect::<Vec<_>>(),
            vec!["chronologically-first", "inserted-first"]
        );
        assert_eq!(
            build_a12_loo_corpus(&store, 0.70).unwrap(),
            build_a12_loo_corpus(&store, 0.70).unwrap()
        );
    }

    #[test]
    fn deprecated_canonical_tip_abstains_explicitly() {
        let store = SqliteStore::in_memory().unwrap();
        let id = "deprecated-tip";
        let same_side_auxiliary = id_for_fold("live-auxiliary", a12_family_fold(id));
        store.store(memory(id, "deprecated content")).unwrap();
        store
            .store(memory(
                &same_side_auxiliary,
                "independent but ineligible auxiliary content",
            ))
            .unwrap();
        store
            .conn()
            .execute(
                "UPDATE memories SET status = 'deprecated' WHERE id = ?1",
                rusqlite::params![id],
            )
            .unwrap();
        insert_auxiliary_group(&store, &[id, same_side_auxiliary.as_str()]);

        let families = enumerate_stable_root_families(&store).unwrap();
        let deprecated_family = families
            .iter()
            .find(|family| family.stable_family_id == id)
            .unwrap();
        assert_eq!(deprecated_family.live_tip_id, None);
        let corpus = build_a12_loo_corpus(&store, 0.70).unwrap();
        assert!(corpus.observations.is_empty());
        let abstention = corpus
            .abstentions
            .iter()
            .find(|abstention| abstention.stable_family_id == id)
            .expect("a non-live canonical family must not disappear silently");
        assert_eq!(abstention.reason, A12AbstentionReason::NoLiveCanonicalTip);
    }

    fn optimizer_event(
        request_id: &str,
        target_dimension: &str,
    ) -> crate::search::alpha_optimizer::RecallEvent {
        use crate::search::alpha_optimizer::CandidateLog;

        let target_is_bm25 = target_dimension == "bm25";
        let candidate = |memory_id: &str, target: bool| CandidateLog {
            memory_id: memory_id.to_string(),
            bm25_norm: if target == target_is_bm25 { 1.0 } else { 0.0 },
            vec_norm: 0.0,
            kg_norm: if target != target_is_bm25 { 1.0 } else { 0.0 },
            episode_norm: 0.0,
            support_count: 1,
            source_diversity: 1.0,
        };
        crate::search::alpha_optimizer::RecallEvent {
            request_id: request_id.to_string(),
            candidates: vec![
                candidate("target", true),
                candidate("noise-a", false),
                candidate("noise-b", false),
                candidate("noise-c", false),
            ],
            accessed_ids: vec!["target".to_string()],
            negative_ids: Vec::new(),
            timestamp: Utc::now(),
            query_cluster_id_at_recall: Some(7),
            cluster_version_at_recall: Some(1),
            query_top_vec_memory_id_at_recall: None,
        }
    }

    fn optimizer_family(
        stable_family_id: String,
        target_dimension: &str,
        baseline_hit: bool,
    ) -> A12FamilyRecallTrace {
        let event = optimizer_event(&stable_family_id, target_dimension);
        let legacy_order = if baseline_hit {
            vec!["target", "noise-a", "noise-b", "noise-c"]
        } else {
            vec!["noise-a", "noise-b", "noise-c", "target"]
        }
        .into_iter()
        .map(str::to_string)
        .collect();
        A12FamilyRecallTrace {
            stable_family_id,
            cases: vec![A12CaseRecallTrace {
                query_type: "semantic".to_string(),
                cluster_id: Some(7),
                valid_until_exclusive: None,
                trace: crate::search::recall::A12RecallTrace {
                    legacy_order,
                    event,
                },
                provenance: A12ProvenanceCounts::default(),
            }],
        }
    }

    fn optimizer_families(
        prefix: &str,
        fold: A12Fold,
        count: usize,
        target_dimension: &str,
        baseline_hit: bool,
    ) -> Vec<A12FamilyRecallTrace> {
        (0..count)
            .map(|index| {
                optimizer_family(
                    id_for_fold(&format!("{prefix}-{index}"), fold),
                    target_dimension,
                    baseline_hit,
                )
            })
            .collect()
    }

    fn global_calibration(
        families: &[A12FamilyRecallTrace],
        min_samples_alpha: usize,
    ) -> A12ScopeCalibration {
        let context = a12_calibration_context_from_bytes(
            b"a12-test-snapshot",
            b"a12-test-corpus",
            DateTime::<Utc>::from_timestamp(0, 0).unwrap(),
        );
        train_and_evaluate_a12_traces(families, min_samples_alpha, &context)
            .into_iter()
            .find(|calibration| calibration.scope == "global")
            .expect("global scope is always reported when traces exist")
    }

    #[test]
    fn activation_holdout_families_never_enter_optimizer_training() {
        let training_id = id_for_fold("kg-training", A12Fold::Training);
        let training = optimizer_family(training_id, "kg", false);
        let expected = global_calibration(std::slice::from_ref(&training), 1).learned_weights;
        let mut families = vec![training];
        families.extend(optimizer_families(
            "bm25-holdout",
            A12Fold::ActivationHoldout,
            12,
            "bm25",
            false,
        ));

        let calibration = global_calibration(&families, 1);

        assert_eq!(calibration.train_family_ess, 1);
        assert_eq!(calibration.learned_weights, expected);
        assert_eq!(calibration.holdout_family_ess, 12);
    }

    #[test]
    fn paired_top3_improvement_propagates_ship() {
        let mut families = optimizer_families("kg-train", A12Fold::Training, 10, "kg", false);
        families.extend(optimizer_families(
            "kg-holdout",
            A12Fold::ActivationHoldout,
            10,
            "kg",
            false,
        ));

        let calibration = global_calibration(&families, 10);

        assert_eq!(
            calibration.holdout_status,
            crate::eval::gates::ScorecardStatus::Ship
        );
        assert_eq!(calibration.paired_top3.baseline_only, 0);
        assert_eq!(calibration.paired_top3.treatment_only, 10);
        assert_eq!(calibration.mcnemar.n, 10);
    }

    #[test]
    fn paired_top3_regression_propagates_bail() {
        let mut families = optimizer_families("bm25-train", A12Fold::Training, 10, "bm25", false);
        families.extend(optimizer_families(
            "kg-holdout-bail",
            A12Fold::ActivationHoldout,
            10,
            "kg",
            true,
        ));

        let calibration = global_calibration(&families, 10);

        assert_eq!(
            calibration.holdout_status,
            crate::eval::gates::ScorecardStatus::Bail
        );
        assert_eq!(calibration.paired_top3.baseline_only, 10);
        assert_eq!(calibration.paired_top3.treatment_only, 0);
        assert_eq!(calibration.mcnemar.n, 10);
    }

    #[test]
    fn paired_top3_requires_train_and_holdout_family_ess_floor() {
        let mut families = optimizer_families("kg-thin-train", A12Fold::Training, 10, "kg", false);
        families.extend(optimizer_families(
            "kg-thin-holdout",
            A12Fold::ActivationHoldout,
            10,
            "kg",
            false,
        ));

        let calibration = global_calibration(&families, 20);

        assert_eq!(calibration.train_family_ess, 10);
        assert_eq!(calibration.holdout_family_ess, 10);
        assert_eq!(calibration.mcnemar.n, 10);
        assert_eq!(
            calibration.holdout_status,
            crate::eval::gates::ScorecardStatus::NoData
        );
        assert!(calibration.holdout_reason.contains("required=20"));
    }

    #[test]
    fn treatment_top3_ties_preserve_legacy_candidate_order() {
        let mut event = optimizer_event("tie-order", "kg");
        for candidate in &mut event.candidates {
            candidate.bm25_norm = 0.0;
            candidate.vec_norm = 0.0;
            candidate.kg_norm = 0.0;
            candidate.episode_norm = 0.0;
            candidate.support_count = 1;
            candidate.source_diversity = 1.0;
        }
        assert!(treatment_top3_hit(
            &event,
            crate::search::alpha_optimizer::ShadowFusionWeights::default()
        ));

        event.candidates.rotate_left(1);
        assert!(!treatment_top3_hit(
            &event,
            crate::search::alpha_optimizer::ShadowFusionWeights::default()
        ));
    }

    #[test]
    fn scope_calibration_carries_resolver_metadata_and_provenance() {
        let mut families = vec![optimizer_family(
            id_for_fold("metadata-training", A12Fold::Training),
            "kg",
            false,
        )];
        families.extend(optimizer_families(
            "metadata-holdout",
            A12Fold::ActivationHoldout,
            10,
            "kg",
            false,
        ));
        families[0].cases[0].provenance = A12ProvenanceCounts {
            canonical_loo: 2,
            concept_loo: 1,
            episode_loo: 3,
        };
        let calibrated_at = Utc.with_ymd_and_hms(2026, 7, 13, 8, 30, 0).unwrap();
        let context = A12CalibrationContext {
            snapshot_fingerprint: "snapshot-fp".to_string(),
            corpus_fingerprint: "corpus-fp".to_string(),
            optimizer_fingerprint: "optimizer-fp".to_string(),
            evaluation_fingerprint: "evaluation-fp".to_string(),
            calibrated_at,
        };

        let calibration = train_and_evaluate_a12_traces(&families, 10, &context)
            .into_iter()
            .find(|entry| entry.scope == "global")
            .unwrap();

        assert_eq!(calibration.snapshot_fingerprint, "snapshot-fp");
        assert_eq!(calibration.corpus_fingerprint, "corpus-fp");
        assert_ne!(calibration.optimizer_fingerprint, "optimizer-fp");
        assert_ne!(calibration.evaluation_fingerprint, "evaluation-fp");
        assert_eq!(calibration.optimizer_fingerprint.len(), 64);
        assert_eq!(calibration.evaluation_fingerprint.len(), 64);
        assert_eq!(calibration.calibrated_at, calibrated_at);
        assert_eq!(
            calibration.provenance,
            A12ProvenanceCounts {
                canonical_loo: 2,
                concept_loo: 1,
                episode_loo: 3,
            }
        );
    }

    #[test]
    fn corpus_trace_uses_recall_classifier_scope_and_provenance() {
        let store = SqliteStore::in_memory().unwrap();
        let root_id = "scope-trace-root";
        let live_tip_id = "scope-trace-tip";
        store
            .store(memory(root_id, "what happened in the deployment session"))
            .unwrap();
        store
            .store(memory(
                live_tip_id,
                "transaction durability and recovery runbook",
            ))
            .unwrap();
        store.mark_superseded(root_id, live_tip_id).unwrap();
        let corpus = build_a12_loo_corpus(&store, 0.70).unwrap();
        let source_case = corpus.observations[0].cases[0].clone();

        let traces =
            trace_a12_loo_corpus(&store, &crate::config::ReinConfig::default(), &corpus, 10)
                .unwrap();

        let traced = &traces[0].cases[0];
        assert_eq!(
            traced.query_type,
            crate::search::classify::classify(&source_case.query_text, false, false)
                .query_type
                .to_string()
        );
        assert_eq!(
            traced.cluster_id,
            traced.trace.event.query_cluster_id_at_recall
        );
        assert_eq!(traced.provenance.canonical_loo, 1);
        assert_eq!(traced.provenance.concept_loo, 0);
        assert_eq!(traced.provenance.episode_loo, 0);
    }

    #[test]
    fn calibration_context_hashes_exact_snapshot_and_corpus_bytes() {
        let snapshot_bytes = b"snapshot\0with\nexact bytes";
        let corpus_bytes = b"corpus\0with\nother exact bytes";
        let calibrated_at = Utc.with_ymd_and_hms(2026, 7, 13, 9, 0, 0).unwrap();
        let context =
            a12_calibration_context_from_bytes(snapshot_bytes, corpus_bytes, calibrated_at);
        let expected = |domain: &[u8], bytes: &[u8]| {
            let mut hasher = Sha256::new();
            hasher.update(domain);
            hasher.update(bytes);
            hasher
                .finalize()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        };

        assert_eq!(
            context.snapshot_fingerprint,
            expected(b"a12-snapshot-v1\0", snapshot_bytes)
        );
        assert_eq!(
            context.corpus_fingerprint,
            expected(b"a12-corpus-v1\0", corpus_bytes)
        );
        assert_ne!(context.snapshot_fingerprint, context.corpus_fingerprint);
        assert_eq!(context.calibrated_at, calibrated_at);
    }

    #[test]
    fn corpus_context_is_pretrace_and_binds_optimizer_evaluation_semantics() {
        let store = SqliteStore::in_memory().unwrap();
        store
            .store(memory("context-root", "query evidence for calibration"))
            .unwrap();
        store
            .store(memory("context-tip", "independent canonical answer"))
            .unwrap();
        store
            .mark_superseded("context-root", "context-tip")
            .unwrap();
        let corpus = build_a12_loo_corpus(&store, 0.70).unwrap();
        let at = Utc.with_ymd_and_hms(2026, 7, 13, 9, 30, 0).unwrap();
        let config = crate::config::ReinConfig::default();
        let base =
            a12_calibration_context_for_corpus(&store, &config, &corpus, 10, 10, at).unwrap();
        let different_limit =
            a12_calibration_context_for_corpus(&store, &config, &corpus, 11, 10, at).unwrap();
        let different_minimum =
            a12_calibration_context_for_corpus(&store, &config, &corpus, 10, 20, at).unwrap();
        let mut changed_config = config.clone();
        changed_config.search.rrf_k += 1.0;
        let different_config =
            a12_calibration_context_for_corpus(&store, &changed_config, &corpus, 10, 10, at)
                .unwrap();
        let mut different_dedup = corpus.clone();
        different_dedup.hard_dedup_bound = 0.71;
        let different_dedup =
            a12_calibration_context_for_corpus(&store, &config, &different_dedup, 10, 10, at)
                .unwrap();

        assert_eq!(
            base.snapshot_fingerprint,
            different_limit.snapshot_fingerprint
        );
        assert_eq!(base.corpus_fingerprint, different_limit.corpus_fingerprint);
        assert_ne!(
            base.optimizer_fingerprint,
            different_limit.optimizer_fingerprint
        );
        assert_ne!(
            base.evaluation_fingerprint,
            different_minimum.evaluation_fingerprint
        );
        assert_ne!(
            base.optimizer_fingerprint,
            different_config.optimizer_fingerprint
        );
        assert_ne!(
            base.evaluation_fingerprint,
            different_config.evaluation_fingerprint
        );
        assert_ne!(base.corpus_fingerprint, different_dedup.corpus_fingerprint);
        assert_ne!(
            base.optimizer_fingerprint,
            different_dedup.optimizer_fingerprint
        );
        assert_eq!(base.optimizer_fingerprint.len(), 64);
        assert_eq!(base.evaluation_fingerprint.len(), 64);
    }

    #[test]
    fn recall_config_projection_ignores_secrets_and_binds_every_relevant_knob() {
        let base = crate::config::ReinConfig::default();
        let projection = a12_recall_config_projection(&base);

        let mut secrets = base.clone();
        secrets.sync.api_key = Some("secret-token".to_string());
        secrets.sync.endpoint = "https://secret-proxy.invalid".to_string();
        secrets.embedding.google.api_key = Some("embedding-secret".to_string());
        secrets.embedding.google.endpoint = "https://embedding-proxy.invalid".to_string();
        secrets.embedding.omlx.endpoint = "http://private-host.invalid/v1".to_string();
        assert_eq!(projection, a12_recall_config_projection(&secrets));

        let mutations: &[fn(&mut crate::config::ReinConfig)] = &[
            |config| config.search.fusion_method = "cc".to_string(),
            |config| config.search.rrf_k += 1.0,
            |config| config.search.rrf_fts_weight += 0.01,
            |config| config.search.rrf_vec_weight += 0.01,
            |config| config.search.cc_alpha += 0.01,
            |config| config.search.strong_signal_ratio += 0.01,
            |config| config.search.strong_signal_single += 0.01,
            |config| config.adaptive.enabled = !config.adaptive.enabled,
            |config| config.adaptive.min_samples_alpha += 1,
        ];
        for mutate in mutations {
            let mut changed = base.clone();
            mutate(&mut changed);
            assert_ne!(projection, a12_recall_config_projection(&changed));
        }
    }

    #[test]
    fn local_snapshot_identity_tracks_complete_adaptive_restore_semantics_and_json_types() {
        let store = SqliteStore::in_memory().unwrap();
        let state = crate::store::adaptive::AdaptiveState::default();
        let mut value = serde_json::to_value(&state).unwrap();
        let persist = |value: &serde_json::Value| {
            store
                .conn()
                .execute(
                    "INSERT OR REPLACE INTO metadata (key, value) VALUES ('adaptive_state', ?1)",
                    rusqlite::params![serde_json::to_string(value).unwrap()],
                )
                .unwrap();
        };
        persist(&value);
        assert!(crate::store::adaptive::AdaptiveState::restore_snapshot(store.conn()).is_some());
        let base = a12_local_recall_snapshot_identity(&store).unwrap();

        let mut irrelevant = state.clone();
        irrelevant.version = irrelevant.version.saturating_add(1);
        irrelevant.dedup_thresholds.insert(7, 0.83);
        irrelevant.judge_calibration_state =
            Some(crate::store::adaptive::JudgeCalibrationState::default());
        persist(&serde_json::to_value(&irrelevant).unwrap());
        assert!(crate::store::adaptive::AdaptiveState::restore_snapshot(store.conn()).is_some());
        assert_eq!(
            base,
            a12_local_recall_snapshot_identity(&store).unwrap(),
            "legal changes outside the three fields read by LOO must not churn the generation"
        );

        let mut relevant = state.clone();
        relevant.cluster_version = relevant.cluster_version.saturating_add(1);
        persist(&serde_json::to_value(&relevant).unwrap());
        assert_ne!(
            base,
            a12_local_recall_snapshot_identity(&store).unwrap(),
            "a legal change to a field LOO reads must advance the generation"
        );

        value["cluster_version"] = serde_json::Value::String("0".to_string());
        persist(&value);
        assert!(crate::store::adaptive::AdaptiveState::restore_snapshot(store.conn()).is_none());
        assert_ne!(
            base,
            a12_local_recall_snapshot_identity(&store).unwrap(),
            "number and same-looking string must have distinct fingerprints"
        );

        value = serde_json::to_value(&state).unwrap();
        value["hot_threshold"] = serde_json::Value::String("0".to_string());
        persist(&value);
        assert!(crate::store::adaptive::AdaptiveState::restore_snapshot(store.conn()).is_none());
        assert_ne!(
            base,
            a12_local_recall_snapshot_identity(&store).unwrap(),
            "a type error in any unprojected field changes restore success and must change identity"
        );
    }

    #[test]
    fn corpus_context_rejects_recall_state_drift_after_atomic_build() {
        let store = SqliteStore::in_memory().unwrap();
        store
            .store(memory("drift-root", "query evidence before drift"))
            .unwrap();
        store
            .store(memory("drift-tip", "independent positive before drift"))
            .unwrap();
        store.mark_superseded("drift-root", "drift-tip").unwrap();
        let corpus = build_a12_loo_corpus(&store, 0.70).unwrap();

        store
            .conn()
            .execute(
                "UPDATE memories SET access_count = access_count + 1 WHERE id = 'drift-tip'",
                [],
            )
            .unwrap();

        let error = a12_calibration_context_for_corpus(
            &store,
            &crate::config::ReinConfig::default(),
            &corpus,
            10,
            10,
            Utc.with_ymd_and_hms(2026, 7, 13, 10, 0, 0).unwrap(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("corpus/local snapshot drift"));
    }

    #[test]
    fn replay_aborts_and_rolls_back_mid_generation_state_drift() {
        let store = SqliteStore::in_memory().unwrap();
        store
            .store(memory("mid-drift-root", "what happened during deployment"))
            .unwrap();
        store
            .store(memory(
                "mid-drift-tip",
                "durable deployment recovery answer",
            ))
            .unwrap();
        store
            .mark_superseded("mid-drift-root", "mid-drift-tip")
            .unwrap();
        let corpus = build_a12_loo_corpus(&store, 0.70).unwrap();
        let evaluation_at = Utc.with_ymd_and_hms(2026, 7, 13, 10, 15, 0).unwrap();

        let error = trace_a12_loo_corpus_at(
            &store,
            &crate::config::ReinConfig::default(),
            &corpus,
            10,
            evaluation_at,
            Some(&corpus.source_snapshot_fingerprint),
            |store, case_index| {
                if case_index == 0 {
                    store.conn().execute(
                        "UPDATE memories SET access_count = access_count + 1 \
                         WHERE id = 'mid-drift-tip'",
                        [],
                    )?;
                }
                Ok(())
            },
        )
        .unwrap_err();

        assert!(error.to_string().contains("drifted during replay"));
        assert_eq!(
            store
                .conn()
                .query_row(
                    "SELECT access_count FROM memories WHERE id = 'mid-drift-tip'",
                    [],
                    |row| row.get::<_, u32>(0),
                )
                .unwrap(),
            0,
            "the failed generation must roll its test mutation back"
        );
    }

    #[test]
    fn final_scope_fingerprints_bind_train_holdout_traces_and_corpus() {
        let training_id = id_for_fold("fingerprint-train", A12Fold::Training);
        let holdout_id = id_for_fold("fingerprint-holdout", A12Fold::ActivationHoldout);
        let families = vec![
            optimizer_family(training_id, "kg", false),
            optimizer_family(holdout_id, "kg", false),
        ];
        let context = A12CalibrationContext {
            snapshot_fingerprint: "snapshot".to_string(),
            corpus_fingerprint: "corpus-a".to_string(),
            optimizer_fingerprint: "optimizer-base".to_string(),
            evaluation_fingerprint: "evaluation-base".to_string(),
            calibrated_at: Utc.with_ymd_and_hms(2026, 7, 13, 10, 30, 0).unwrap(),
        };
        let evaluate = |families: &[A12FamilyRecallTrace], context: &A12CalibrationContext| {
            train_and_evaluate_a12_traces(families, 1, context)
                .into_iter()
                .find(|entry| entry.scope == "global")
                .unwrap()
        };
        let base = evaluate(&families, &context);

        let mut train_changed = families.clone();
        train_changed[0].cases[0].trace.event.candidates[0].kg_norm = 0.875;
        let train_changed = evaluate(&train_changed, &context);
        assert_ne!(
            base.optimizer_fingerprint,
            train_changed.optimizer_fingerprint
        );
        assert_ne!(
            base.evaluation_fingerprint,
            train_changed.evaluation_fingerprint
        );

        let mut holdout_changed = families.clone();
        holdout_changed[1].cases[0]
            .trace
            .legacy_order
            .rotate_left(1);
        let holdout_changed = evaluate(&holdout_changed, &context);
        assert_eq!(
            base.optimizer_fingerprint,
            holdout_changed.optimizer_fingerprint
        );
        assert_ne!(
            base.evaluation_fingerprint,
            holdout_changed.evaluation_fingerprint
        );

        let mut other_corpus = context.clone();
        other_corpus.corpus_fingerprint = "corpus-b".to_string();
        let other_corpus = evaluate(&families, &other_corpus);
        assert_ne!(
            base.optimizer_fingerprint,
            other_corpus.optimizer_fingerprint
        );
        assert_ne!(
            base.evaluation_fingerprint,
            other_corpus.evaluation_fingerprint
        );
    }

    #[test]
    fn scope_uses_earliest_case_boundary_and_fails_closed_at_boundary() {
        let early = Utc.with_ymd_and_hms(2026, 7, 13, 12, 0, 0).unwrap();
        let late = early + chrono::Duration::hours(1);
        let mut families = vec![
            optimizer_family(
                id_for_fold("boundary-training", A12Fold::Training),
                "kg",
                false,
            ),
            optimizer_family(
                id_for_fold("boundary-holdout", A12Fold::ActivationHoldout),
                "kg",
                false,
            ),
        ];
        families[0].cases[0].valid_until_exclusive = Some(late.timestamp_millis());
        families[1].cases[0].valid_until_exclusive = Some(early.timestamp_millis());
        let calibration = global_calibration(&families, 1);

        assert_eq!(
            calibration.valid_until_exclusive,
            Some(early.timestamp_millis())
        );
        assert!(calibration.is_current_at(early - chrono::Duration::milliseconds(1)));
        assert!(!calibration.is_current_at(early));
        assert!(!calibration.is_current_at(early + chrono::Duration::milliseconds(1)));
    }

    #[test]
    fn relative_window_boundary_includes_future_entry_and_age_out() {
        let store = SqliteStore::in_memory().unwrap();
        let evaluation_at = Utc.with_ymd_and_hms(2026, 7, 13, 12, 0, 0).unwrap();
        store
            .store(memory("recent-row", "recent temporal candidate"))
            .unwrap();
        store
            .store(memory("future-row", "future temporal candidate"))
            .unwrap();
        set_created_at(
            &store,
            "recent-row",
            &(evaluation_at - chrono::Duration::days(3)).to_rfc3339(),
        );
        set_created_at(
            &store,
            "future-row",
            &(evaluation_at + chrono::Duration::days(2)).to_rfc3339(),
        );

        let boundary = a12_next_relative_temporal_boundary_exclusive(
            &store,
            "what happened last week",
            evaluation_at,
        )
        .unwrap();

        assert_eq!(
            boundary,
            Some((evaluation_at + chrono::Duration::days(2)).timestamp_millis())
        );
    }
}
