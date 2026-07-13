//! Deterministic leave-one-evidence-out corpus construction for A12.
//!
//! Task 1 deliberately stops before recall execution: this module only reads
//! the canonical graph and builds family-equal cases plus explicit
//! abstentions. No feedback, access, recall-hit, embedding, or index state is
//! written here.

// Task 1 intentionally publishes crate-local inputs for Tasks 2-6 before
// those consumers exist. Keep the staging module warning-clean in isolation.
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
    pub observations: Vec<A12FamilyObservation>,
    pub abstentions: Vec<A12LooAbstention>,
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
}
