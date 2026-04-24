use crate::config::{Provider, ReinConfig};
use crate::extract::llm::{strip_code_fences, ExtractorKind};
use crate::store::adaptive::{
    emit_event, AdaptiveState, EventType, FeedbackEvent, RefreshSample,
};
use crate::store::memoir::{row_to_concept, should_refresh_living_summary};
use crate::store::SqliteStore;
use crate::types::{Concept, ConceptRevision, OpsErrorKind, ReinError, ReinResult};
use chrono::{DateTime, Utc};

const SYSTEM_PROMPT: &str = "You are a concept-state synthesizer for a memory system. \
Given a concept's name, current definition, and revision history in chronological order, \
produce a 3-to-5-sentence summary of the concept's current state of understanding. \
Emphasize what changed in recent revisions. Output only the summary as plain prose, \
no preamble, no bullet points, no code fences.\n\n\
CRITICAL — preserve exact identifiers verbatim. Library names, acronyms, version \
strings, file names, config keys, API names, error class names, and specific metric \
names MUST appear unchanged in the output — do NOT paraphrase them into general \
terms like 'vector search library' or 'clustering algorithm'. When revision history \
shows a decision changing over time, preserve BOTH earlier AND later positions with \
their dates — do not collapse them into the final state.\n\n\
Example 1 (preserve identifiers):\n\
  Evidence: [2025-01-05] Uses tantivy for full-text + faiss for HNSW, nltk tokenizer.\n\
            [2025-01-20] Added fst as dictionary backend.\n\
  BAD:  \"The system combines full-text and vector search with text tokenization.\"\n\
  GOOD: \"Combines tantivy (full-text) with faiss (HNSW). Tokenization via nltk backed by fst (2025-01-20).\"\n\n\
Example 2 (preserve decision flip):\n\
  Evidence: [2025-02-10] Chose AWS Lambda for background jobs.\n\
            [2025-02-25] Switched to Fly.io Machines due to cold-start latency.\n\
  BAD:  \"Background jobs run on Fly.io Machines.\"\n\
  GOOD: \"2025-02-10 AWS Lambda chosen; 2025-02-25 switched to Fly.io Machines (cold-start latency).\"";

const REVISION_HISTORY_LIMIT: usize = 20;

/// Result payload of a `run_concept_summary` invocation.
///
/// Counter semantics (intentional asymmetry per design, confirmed by
/// Codex audit round 1 finding #4):
/// - **Single-target mode** (`concept_id = Some(id)`): operator explicitly
///   asked about ONE concept. If that concept fails the refresh trigger,
///   `skipped_not_eligible = 1` so the operator sees a clear signal in the
///   outcome ("you asked, here's why nothing happened").
/// - **Batch mode** (`concept_id = None`): selection is automatic over all
///   concepts. Ineligible concepts are silently filtered; `skipped_not_eligible`
///   stays 0 to avoid flooding telemetry with expected-filter noise. Only
///   eligible concepts that then reach the LLM path increment `attempted`.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct ConceptSummaryOutcome {
    pub attempted: u32,
    pub succeeded: u32,
    pub skipped_not_eligible: u32,
    pub llm_failed: u32,
    pub skipped_disabled: bool,
    pub skipped_no_llm: bool,
    pub dry_run: bool,
}

pub fn run_concept_summary(
    store: &SqliteStore,
    config: &ReinConfig,
    concept_id: Option<&str>,
    dry_run: bool,
) -> ReinResult<ConceptSummaryOutcome> {
    run_concept_summary_inner(store, config, concept_id, dry_run, None)
}

#[cfg(feature = "test-support")]
pub fn run_concept_summary_with_extractor(
    store: &SqliteStore,
    config: &ReinConfig,
    concept_id: Option<&str>,
    dry_run: bool,
    extractor: ExtractorKind,
) -> ReinResult<ConceptSummaryOutcome> {
    run_concept_summary_inner(store, config, concept_id, dry_run, Some(extractor))
}

fn run_concept_summary_inner(
    store: &SqliteStore,
    config: &ReinConfig,
    concept_id: Option<&str>,
    dry_run: bool,
    extractor_override: Option<ExtractorKind>,
) -> ReinResult<ConceptSummaryOutcome> {
    let mut outcome = ConceptSummaryOutcome {
        dry_run,
        ..Default::default()
    };

    if !config.ars.concept_summary_enabled {
        outcome.skipped_disabled = true;
        return Ok(outcome);
    }

    let state = AdaptiveState::restore_snapshot(store.conn()).unwrap_or_default();
    let now = Utc::now();

    let (eligible, not_eligible) = select_eligible(store, concept_id, &state, now)?;
    outcome.skipped_not_eligible = not_eligible;

    if dry_run {
        outcome.attempted = eligible.len() as u32;
        return Ok(outcome);
    }

    let extractor = match extractor_override {
        Some(e) => e,
        None => match create_concept_summary_extractor(config) {
            Some(e) => e,
            None => {
                outcome.skipped_no_llm = true;
                return Ok(outcome);
            }
        },
    };

    let batch_cap = config.ars.batch_size.max(1);
    for concept in eligible.into_iter().take(batch_cap) {
        outcome.attempted += 1;
        match summarize_one(store, &extractor, &concept) {
            Ok(()) => outcome.succeeded += 1,
            Err(SummaryError::Llm(e)) => {
                outcome.llm_failed += 1;
                tracing::warn!(
                    concept_id = %concept.id,
                    error = %e,
                    "concept_summary LLM call failed"
                );
            }
            Err(SummaryError::Store(e)) => {
                tracing::warn!(
                    concept_id = %concept.id,
                    error = %e,
                    "concept_summary write failed"
                );
            }
        }
    }

    Ok(outcome)
}

fn select_eligible(
    store: &SqliteStore,
    concept_id: Option<&str>,
    state: &AdaptiveState,
    now: DateTime<Utc>,
) -> ReinResult<(Vec<Concept>, u32)> {
    if let Some(id) = concept_id {
        match store.get_concept_by_id(id)? {
            Some(c) => {
                if should_refresh_living_summary(&c, state, now) {
                    Ok((vec![c], 0))
                } else {
                    Ok((Vec::new(), 1))
                }
            }
            None => Ok((Vec::new(), 1)),
        }
    } else {
        let mut stmt = store
            .conn()
            .prepare("SELECT * FROM concepts ORDER BY COALESCE(updated_at, created_at) ASC")?;
        let rows = stmt.query_map([], |row| {
            row_to_concept(row).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    0,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })
        })?;
        let mut eligible = Vec::new();
        for row in rows {
            match row {
                Ok(c) if should_refresh_living_summary(&c, state, now) => eligible.push(c),
                Ok(_) => {}
                Err(e) => return Err(ReinError::Database(e)),
            }
        }
        Ok((eligible, 0))
    }
}

enum SummaryError {
    Llm(ReinError),
    Store(ReinError),
}

fn summarize_one(
    store: &SqliteStore,
    extractor: &ExtractorKind,
    concept: &Concept,
) -> Result<(), SummaryError> {
    let revisions =
        load_revisions(store, &concept.id, REVISION_HISTORY_LIMIT).map_err(SummaryError::Store)?;
    let prompt = build_concept_summary_prompt(concept, &revisions);
    let source_revision = concept.revision;

    let llm_output = call_llm_sync(extractor, &prompt).map_err(SummaryError::Llm)?;
    let summary = strip_code_fences(&llm_output);
    let summary = summary.trim();
    if summary.is_empty() {
        return Err(SummaryError::Llm(ReinError::Extract(
            "concept_summary LLM returned empty output".to_string(),
        )));
    }

    let now = Utc::now();
    // L4 CAS: pass the prior `living_summary_source_revision` so a peer
    // refresh that committed first cannot be silently overwritten. Two
    // concurrent refreshes started against the same `concept.revision`
    // both see the same prior; whichever commits first wins, the loser
    // gets 0 rows and surfaces a Conflict (no LLM cost wasted on the
    // path back, just a stale write).
    let prior_source_revision = concept.living_summary_source_revision;
    let wrote = write_living_summary_if_revision_unchanged(
        store,
        &concept.id,
        source_revision,
        prior_source_revision,
        summary,
        now,
    )
    .map_err(SummaryError::Store)?;
    if !wrote {
        return Err(SummaryError::Store(
            ReinError::Config(format!(
                "concept '{}' changed while concept_summary was running; skipped stale living_summary write",
                concept.id
            ))
            .with_kind(OpsErrorKind::Conflict),
        ));
    }

    // L3 wiring: emit a `ConceptSummaryRefreshed` feedback event so the
    // adaptive slow-channel can learn refresh-interval percentiles.
    // Anchored to the prior summary if one exists; otherwise to concept
    // creation (see `RefreshSample` doc comment for rationale). Best-effort
    // — a failed event emit doesn't roll back the successful summary write.
    //
    // The `first_refresh` flag lets `recompute_concept_refresh_stats`
    // exclude the inception-anchored age from its percentile (Codex
    // round-2 MEDIUM) while still counting the sample for revision-side
    // bootstrap exit.
    let first_refresh = concept.living_summary_source_revision.is_none();
    let prior_revision = concept.living_summary_source_revision.unwrap_or(0);
    let revisions_since_last = source_revision.saturating_sub(prior_revision);
    let prior_anchor = concept
        .living_summary_updated_at
        .unwrap_or(concept.created_at);
    let age_secs_since_last = (now - prior_anchor).num_seconds().max(0);
    let sample = RefreshSample {
        revisions_since_last,
        age_secs_since_last,
        first_refresh,
    };
    if let Err(e) = emit_refresh_event(store, &concept.id, sample) {
        tracing::warn!(
            concept_id = %concept.id,
            error = %e,
            "concept_summary: failed to emit ConceptSummaryRefreshed event (non-fatal)"
        );
    }

    Ok(())
}

fn emit_refresh_event(
    store: &SqliteStore,
    concept_id: &str,
    sample: RefreshSample,
) -> ReinResult<()> {
    let payload = serde_json::to_value(sample).map_err(|e| {
        ReinError::Config(format!("failed to serialize RefreshSample: {e}"))
    })?;
    emit_event(
        store.conn(),
        FeedbackEvent {
            event_type: EventType::ConceptSummaryRefreshed,
            request_id: None,
            memory_id: None,
            concept_id: Some(concept_id.to_string()),
            query: None,
            query_type: None,
            topic: None,
            payload: Some(payload),
        },
    )?;
    Ok(())
}

/// L4 CAS write: succeeds only when the concept's `revision` AND its
/// `living_summary_source_revision` both match what we observed at the
/// start of the refresh.
///
/// - `revision` predicate (existing): a `refine_concept` racing with us
///   bumps revision and we abort — the new revision deserves a fresh
///   summary derived from the post-refine state, not our stale prompt.
/// - `living_summary_source_revision` predicate (L4): two concurrent
///   refreshes that both observed the same prior source_revision can
///   only have one winner. The first to commit advances the column;
///   the second's predicate fails and returns 0 rows. NULL is matched
///   via `IS` (SQLite's NULL-safe equality), so a first-ever refresh
///   races correctly against another first-ever refresh.
fn write_living_summary_if_revision_unchanged(
    store: &SqliteStore,
    concept_id: &str,
    source_revision: u32,
    prior_source_revision: Option<u32>,
    summary: &str,
    now: DateTime<Utc>,
) -> ReinResult<bool> {
    let now = now.to_rfc3339();
    let prior_param: Option<i64> = prior_source_revision.map(|v| v as i64);
    let rows = store.conn().execute(
        "UPDATE concepts \
         SET living_summary = ?1, \
             living_summary_updated_at = ?2, \
             living_summary_source_revision = ?3 \
         WHERE id = ?4 \
           AND revision = ?5 \
           AND living_summary_source_revision IS ?6",
        rusqlite::params![
            summary,
            &now,
            source_revision as i64,
            concept_id,
            source_revision as i64,
            prior_param,
        ],
    )?;
    Ok(rows > 0)
}

fn load_revisions(
    store: &SqliteStore,
    concept_id: &str,
    limit: usize,
) -> ReinResult<Vec<ConceptRevision>> {
    let mut stmt = store.conn().prepare(
        "SELECT * FROM ( \
             SELECT * FROM concept_revisions \
             WHERE concept_id = ?1 \
             ORDER BY revision DESC \
             LIMIT ?2 \
         ) \
         ORDER BY revision ASC",
    )?;
    let rows = stmt.query_map(rusqlite::params![concept_id, limit], |row| {
        let labels_json: String = row.get("labels")?;
        let source_json: String = row.get("source_memory_ids")?;
        let created_at_str: String = row.get("created_at")?;
        Ok(ConceptRevision {
            id: row.get("id")?,
            concept_id: row.get("concept_id")?,
            revision: row.get("revision")?,
            definition: row.get("definition")?,
            confidence: row.get("confidence")?,
            labels: serde_json::from_str(&labels_json).unwrap_or_default(),
            source_memory_ids: serde_json::from_str(&source_json).unwrap_or_default(),
            episode_id: row.get("episode_id").unwrap_or(None),
            created_at: DateTime::parse_from_rfc3339(&created_at_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
        })
    })?;
    // Codex audit finding #2: surface malformed rows via tracing::warn rather
    // than silent `filter_map(|r| r.ok())`. Preserves "continue on error"
    // batch semantics (one bad row doesn't abort summary) while giving
    // operators a signal to investigate corruption.
    let mut revisions = Vec::new();
    for row in rows {
        match row {
            Ok(r) => revisions.push(r),
            Err(e) => tracing::warn!(
                concept_id = %concept_id,
                error = %e,
                "load_revisions: dropped malformed concept_revisions row"
            ),
        }
    }
    Ok(revisions)
}

pub fn build_concept_summary_prompt(concept: &Concept, revisions: &[ConceptRevision]) -> String {
    let mut buf = String::with_capacity(
        concept.definition.len()
            + revisions.iter().map(|r| r.definition.len()).sum::<usize>()
            + 256,
    );
    buf.push_str("Concept: ");
    buf.push_str(&concept.name);
    buf.push_str("\n\nCurrent definition (revision ");
    buf.push_str(&concept.revision.to_string());
    buf.push_str("):\n");
    buf.push_str(&concept.definition);
    buf.push_str("\n\nRevision history (oldest first):\n");
    if revisions.is_empty() {
        buf.push_str("(no prior revisions recorded)\n");
    } else {
        for rev in revisions {
            buf.push_str(&format!(
                "--- Revision #{} (created {}) ---\n",
                rev.revision,
                rev.created_at.to_rfc3339()
            ));
            buf.push_str(&rev.definition);
            if !rev.definition.ends_with('\n') {
                buf.push('\n');
            }
        }
    }
    buf.push_str("\nNow produce the 3-to-5-sentence current-state summary.");
    buf
}

pub fn create_concept_summary_extractor(config: &ReinConfig) -> Option<ExtractorKind> {
    let extract_provider = config.extract_provider();
    match config.ars.resolved_provider(extract_provider) {
        Provider::None => None,
        Provider::Google => {
            let api_key = config.extract.google.api_key.as_ref()?.clone();
            Some(ExtractorKind::Gemini(
                crate::extract::llm::GeminiExtractor::new(
                    api_key,
                    config.extract.google.endpoint.clone(),
                    config.extract.google.model.clone(),
                ),
            ))
        }
        Provider::Omlx => Some(ExtractorKind::Omlx(
            crate::extract::llm::OmlxExtractor::new(
                config.extract.omlx.endpoint.clone(),
                config.extract.omlx.model.clone(),
                config.extract.omlx.disable_thinking,
            ),
        )),
    }
}

pub fn call_llm_sync(extractor: &ExtractorKind, prompt: &str) -> ReinResult<String> {
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        tokio::task::block_in_place(|| {
            handle.block_on(async { extractor.raw_text_with_prompt(SYSTEM_PROMPT, prompt).await })
        })
    } else {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| ReinError::Config(format!("failed to build tokio runtime: {e}")))?;
        rt.block_on(async { extractor.raw_text_with_prompt(SYSTEM_PROMPT, prompt).await })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Memoir;

    fn make_memoir(name: &str) -> Memoir {
        Memoir {
            id: String::new(),
            name: name.to_string(),
            description: "concept-summary unit test".to_string(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    fn make_concept(memoir_name: &str, name: &str) -> Concept {
        Concept {
            id: String::new(),
            memoir_id: memoir_name.to_string(),
            name: name.to_string(),
            definition: "initial definition".to_string(),
            labels: Vec::new(),
            source_memory_ids: Vec::new(),
            confidence: 0.8,
            revision: 1,
            last_episode_id: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            living_summary: None,
            living_summary_updated_at: None,
            living_summary_source_revision: None,
        }
    }

    #[test]
    fn stale_source_revision_does_not_write_living_summary() {
        let store = SqliteStore::in_memory().unwrap();
        store.create_memoir(make_memoir("test-memoir")).unwrap();
        let concept_id = store
            .add_concept(make_concept("test-memoir", "stale-concept"))
            .unwrap();
        store
            .refine_concept("test-memoir", "stale-concept", "updated definition")
            .unwrap();

        let wrote = write_living_summary_if_revision_unchanged(
            &store,
            &concept_id,
            1,
            None,
            "stale summary",
            Utc::now(),
        )
        .unwrap();

        assert!(!wrote, "stale source revision must not update the concept");
        let current = store
            .get_concept_by_id(&concept_id)
            .unwrap()
            .expect("concept exists");
        assert_eq!(current.revision, 2);
        assert!(current.living_summary.is_none());

        let wrote = write_living_summary_if_revision_unchanged(
            &store,
            &concept_id,
            2,
            None,
            "fresh summary",
            Utc::now(),
        )
        .unwrap();

        assert!(wrote, "matching source revision should update the concept");
        let current = store
            .get_concept_by_id(&concept_id)
            .unwrap()
            .expect("concept exists");
        assert_eq!(current.living_summary.as_deref(), Some("fresh summary"));
        assert_eq!(current.living_summary_source_revision, Some(2));
    }

    /// L4 CAS: two concurrent refreshes that both observed
    /// `living_summary_source_revision = None` simulate the first-refresh
    /// race. Whoever commits first wins; the loser's predicate fails and
    /// returns `Ok(false)` without overwriting the winner's summary.
    #[test]
    fn concurrent_first_refresh_loser_does_not_overwrite_winner() {
        let store = SqliteStore::in_memory().unwrap();
        store.create_memoir(make_memoir("test-memoir")).unwrap();
        let concept_id = store
            .add_concept(make_concept("test-memoir", "race-concept"))
            .unwrap();

        // Winner: passes prior=None (matches initial NULL) → commits.
        let winner_wrote = write_living_summary_if_revision_unchanged(
            &store,
            &concept_id,
            1,
            None,
            "winner summary",
            Utc::now(),
        )
        .unwrap();
        assert!(winner_wrote);

        // Loser: also passes prior=None (its observation predates the
        // winner's commit). Predicate `living_summary_source_revision IS
        // NULL` now fails because the column is `Some(1)` → 0 rows.
        let loser_wrote = write_living_summary_if_revision_unchanged(
            &store,
            &concept_id,
            1,
            None,
            "loser summary",
            Utc::now(),
        )
        .unwrap();
        assert!(!loser_wrote, "loser must not overwrite winner");

        let after = store
            .get_concept_by_id(&concept_id)
            .unwrap()
            .expect("concept exists");
        assert_eq!(after.living_summary.as_deref(), Some("winner summary"));
        assert_eq!(after.living_summary_source_revision, Some(1));
    }

    /// L4 CAS: same race in steady state — both refreshes observed
    /// `living_summary_source_revision = Some(prior)` from a prior summary.
    #[test]
    fn concurrent_steady_state_refresh_loser_does_not_overwrite_winner() {
        let store = SqliteStore::in_memory().unwrap();
        store.create_memoir(make_memoir("test-memoir")).unwrap();
        let concept_id = store
            .add_concept(make_concept("test-memoir", "race-concept-steady"))
            .unwrap();
        // Seed an initial summary so prior_source_revision = Some(1).
        write_living_summary_if_revision_unchanged(
            &store,
            &concept_id,
            1,
            None,
            "initial",
            Utc::now(),
        )
        .unwrap();
        // Bump to revision 5 (two refines).
        for _ in 0..4 {
            store
                .refine_concept("test-memoir", "race-concept-steady", "refined")
                .unwrap();
        }

        // Winner: prior = Some(1), source = 5 → commits, advances column to 5.
        let winner_wrote = write_living_summary_if_revision_unchanged(
            &store,
            &concept_id,
            5,
            Some(1),
            "winner",
            Utc::now(),
        )
        .unwrap();
        assert!(winner_wrote);

        // Loser: also observed prior = Some(1) before winner commit.
        // After winner: column is now Some(5) → loser predicate `IS Some(1)` fails.
        let loser_wrote = write_living_summary_if_revision_unchanged(
            &store,
            &concept_id,
            5,
            Some(1),
            "loser",
            Utc::now(),
        )
        .unwrap();
        assert!(!loser_wrote);

        let after = store
            .get_concept_by_id(&concept_id)
            .unwrap()
            .expect("concept exists");
        assert_eq!(after.living_summary.as_deref(), Some("winner"));
        assert_eq!(after.living_summary_source_revision, Some(5));
    }
}
