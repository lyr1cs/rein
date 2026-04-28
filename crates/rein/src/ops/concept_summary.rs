use crate::config::{Provider, ReinConfig};
use crate::extract::llm::{strip_code_fences, ExtractorKind};
use crate::store::adaptive::{
    concept_summary_bucket_key, emit_event, AdaptiveState, ClusterConceptSummaryStats, EventType,
    FeedbackEvent, RefreshSample, CONCEPT_SUMMARY_USEFUL_RATE_THRESHOLD,
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
    /// Codex R2 P2 fix — surface the per-refresh `living_summary_id` ULIDs
    /// minted during this run. Empty when `succeeded == 0`. Callers
    /// (MCP / CLI) feed these into `rein_judge_concept_summary` for
    /// manual A/B against the runtime LLM judge. Order matches the
    /// concept iteration order; index 0 is the first concept refreshed.
    /// Default-skipped on serialize when empty so existing callers see
    /// no JSON shape change.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub minted_summary_ids: Vec<String>,
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
        match summarize_one(store, &extractor, &concept, config, &state) {
            Ok(summary_id) => {
                outcome.succeeded += 1;
                if !summary_id.is_empty() {
                    outcome.minted_summary_ids.push(summary_id);
                }
            }
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
    config: &ReinConfig,
    adaptive_state: &AdaptiveState,
) -> Result<String, SummaryError> {
    // Codex R2 P2 fix — return the minted `living_summary_id` so
    // `run_concept_summary_inner` can surface it on `ConceptSummaryOutcome`.
    // Empty string when the L4 CAS path didn't write (concurrent refresh
    // won) — caller treats empty as "skipped, no id to surface".
    let revisions =
        load_revisions(store, &concept.id, REVISION_HISTORY_LIMIT).map_err(SummaryError::Store)?;
    let max_chars = crate::extract::llm::resolve_max_input_for_section_kind(
        config,
        "ars.concept_summary",
        extractor,
    );
    let prompt = build_concept_summary_prompt_with_cap(concept, &revisions, max_chars);
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
    // v0.27.1 E direction (spec §3.2 R8 P1): mint a per-instance ULID for
    // the new summary BEFORE the L4 CAS write. The same id is persisted
    // on `concepts.living_summary_id` (live row) AND on
    // `concept_summary_instances` (immutable retention row) so the
    // runtime LLM judge can link J5 back to the exact prose it scored
    // even after a subsequent refresh overwrites the live row.
    let summary_id = format!("cs_{}", ulid::Ulid::new());

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
        &summary_id,
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

    // v0.27.1 E direction: write the immutable retention row. Best-effort
    // — failure to record the instance doesn't roll back the successful
    // summary write (the live row carries the same id, so a subsequent
    // judge call falls back to the live row).
    if let Err(e) = insert_concept_summary_instance(store, &summary_id, &concept.id, summary, now) {
        tracing::warn!(
            concept_id = %concept.id,
            summary_id = %summary_id,
            error = %e,
            "concept_summary: failed to record concept_summary_instances retention row (non-fatal)"
        );
    }

    // v0.27.1 E direction (spec §6.5 + §7 + §9.2): runtime LLM judge wiring
    // — Cap A mirror of `recall_synthesis::enqueue_judge_for_synthesis`.
    // Cluster_id for concepts isn't surfaced by the type, so we use `None`
    // (routes to the global `-1` bucket for the sample-rate ladder, which
    // matches `should_refresh_living_summary`'s coarseness). Default-off
    // skips the cache + queue + cron-archive writes entirely.
    // Codex R2 P2: honor master + per-surface flag together.
    if config.ars.llm_judge.enabled && config.ars.llm_judge.concept_summary_enabled {
        enqueue_judge_for_concept_summary(
            config,
            adaptive_state,
            &summary_id,
            &concept.id,
            &prompt,
            summary,
        );
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
        summary_id: summary_id.clone(),
    };
    if let Err(e) = emit_refresh_event(store, &concept.id, sample) {
        tracing::warn!(
            concept_id = %concept.id,
            error = %e,
            "concept_summary: failed to emit ConceptSummaryRefreshed event (non-fatal)"
        );
    }

    Ok(summary_id)
}

fn insert_concept_summary_instance(
    store: &SqliteStore,
    summary_id: &str,
    concept_id: &str,
    summary: &str,
    now: DateTime<Utc>,
) -> ReinResult<()> {
    store.conn().execute(
        "INSERT OR IGNORE INTO concept_summary_instances \
         (summary_id, concept_id, summary_text, refreshed_at) \
         VALUES (?1, ?2, ?3, ?4)",
        rusqlite::params![summary_id, concept_id, summary, now.timestamp()],
    )?;
    Ok(())
}

fn emit_refresh_event(
    store: &SqliteStore,
    concept_id: &str,
    sample: RefreshSample,
) -> ReinResult<()> {
    let payload = serde_json::to_value(sample)
        .map_err(|e| ReinError::Config(format!("failed to serialize RefreshSample: {e}")))?;
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
    summary_id: &str,
    now: DateTime<Utc>,
) -> ReinResult<bool> {
    let now = now.to_rfc3339();
    let prior_param: Option<i64> = prior_source_revision.map(|v| v as i64);
    let rows = store.conn().execute(
        "UPDATE concepts \
         SET living_summary = ?1, \
             living_summary_updated_at = ?2, \
             living_summary_source_revision = ?3, \
             living_summary_id = ?7 \
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
            summary_id,
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
    build_concept_summary_prompt_with_cap(concept, revisions, 0)
}

pub fn build_concept_summary_prompt_with_cap(
    concept: &Concept,
    revisions: &[ConceptRevision],
    max_chars: usize,
) -> String {
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
    if max_chars > 0 && buf.chars().count() > max_chars {
        buf = cap_concept_summary_prompt(buf, max_chars);
    }
    buf
}

fn cap_concept_summary_prompt(mut prompt: String, max_chars: usize) -> String {
    const NOTICE: &str = "\n[...revision history truncated for prompt budget]\n";
    if max_chars == 0 || prompt.chars().count() <= max_chars {
        return prompt;
    }
    let notice_chars = NOTICE.chars().count();
    if max_chars <= notice_chars {
        return prompt.chars().take(max_chars).collect();
    }
    let take = max_chars.saturating_sub(notice_chars);
    prompt = prompt.chars().take(take).collect();
    prompt.push_str(NOTICE);
    if prompt.chars().count() > max_chars {
        prompt.chars().take(max_chars).collect()
    } else {
        prompt
    }
}

/// Build an `ExtractorKind` honoring the resolved LLM config for the
/// given consumer section. Shared by the Cap A (`ars.concept_summary`)
/// and Cap B (`ars.recall_synthesis`) call paths so each caller gets the
/// right `[llm]`-resolved provider/model/endpoint per spec §8.5.
///
/// API key + disable_thinking still live on `[extract.{provider}]` per
/// v0.26.x mapping.
pub fn create_ars_extractor(config: &ReinConfig, section: &str) -> Option<ExtractorKind> {
    let r = config.resolve_llm_for(section).ok()?;
    match r.provider {
        Provider::None => None,
        Provider::Google => {
            // Codex R1 P2 fix — honor the resolver's api_key_env. v0 wording
            // hardcoded `config.extract.google.api_key` which only reads
            // GEMINI_API_KEY at config-load. Operators setting
            // `[llm.google].api_key_env = "MY_KEY"` resolved to Google but
            // were silently disabled because this constructor never read MY_KEY.
            let api_key = r
                .api_key_env
                .as_deref()
                .and_then(|env_name| std::env::var(env_name).ok())
                .or_else(|| config.extract.google.api_key.clone())?;
            Some(ExtractorKind::Gemini(
                crate::extract::llm::GeminiExtractor::new(api_key, r.endpoint, r.model),
            ))
        }
        Provider::Omlx => Some(ExtractorKind::Omlx(
            crate::extract::llm::OmlxExtractor::new(
                r.endpoint,
                r.model,
                config.extract.omlx.disable_thinking,
            ),
        )),
    }
}

/// Cap A entry point — Concept Living Summary.
///
/// v0.27.1 B2: routes through `resolve_llm_for("ars.concept_summary")`
/// so `[llm]` inheritance applies. The resolver replicates v0.26.x's
/// `[ars].llm_backend` semantic ("inherit" → [extract].provider; named
/// provider → use that).
pub fn create_concept_summary_extractor(config: &ReinConfig) -> Option<ExtractorKind> {
    create_ars_extractor(config, "ars.concept_summary")
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

// ── v0.27 ARS Cap A feedback loop: per-query gate ───────────────────────────
//
// Mirrors `ops/recall_synthesis::decide_synthesize` for the Cap A surface.
// Decision logic is identical (operator override > cluster gate > cold-start
// fallback). Re-stated rather than parameterised because Cap A and Cap B are
// independent feature flags that may diverge over time
// (`feedback_no_subjective_params`).

/// Reason a per-query Cap-A gate skipped. Used inside
/// [`ConceptSummaryDecision::Skip`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConceptSummarySkipReason {
    /// `[ars].concept_summary_enabled = false` — operator opted out
    /// globally.
    OperatorDisabled,
    /// Per-query adaptive decision: cluster's `useful_rate` is below
    /// [`CONCEPT_SUMMARY_USEFUL_RATE_THRESHOLD`].
    AdaptiveDecision,
}

/// Per-query adaptive Cap-A decision. `Yes` flows into the existing
/// concept-summary surface; `Skip(reason)` short-circuits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConceptSummaryDecision {
    Yes,
    Skip(ConceptSummarySkipReason),
}

/// Per-query Cap-A gate. Decides whether to surface a concept living-summary
/// to the user based on the learned `(cluster_id, query_type)` `useful_rate`.
///
/// **Cold-start fallback**: when `global_enabled` is `true` but the adaptive
/// state cannot disambiguate (no `cluster_id`, no adaptive state snapshot,
/// no per-cluster bucket yet, or per-cluster events < `cold_start_n`), the
/// function returns [`ConceptSummaryDecision::Yes`] — i.e. "behave like the
/// pre-feedback v0.26.x default and let the surface render". The
/// global flag is binding only when an operator explicitly disabled Cap A.
///
/// When `global_enabled` is `false`, the function ALWAYS returns
/// `Skip(OperatorDisabled)` — operator override wins over any adaptive
/// signal.
///
/// Bucket key for `by_cluster.get(...)` is built via the canonical
/// [`concept_summary_bucket_key`] helper from `store::adaptive`, matching the
/// `"{cid}|{qtype}"` format documented on
/// [`crate::store::adaptive::ConceptSummaryFeedbackState::by_cluster`]. Both
/// sides reuse the same helper so they cannot drift; mismatch would produce a
/// silent dead-code path (the v0.26.0 D-direction bug v0.26.2 fixed for Cap B).
///
/// Pure function — no IO, no allocation beyond the cluster-key string.
pub fn decide_concept_summary_quality(
    global_enabled: bool,
    cluster_id: Option<i64>,
    query_type: &str,
    adaptive_state: Option<&AdaptiveState>,
    cold_start_n: u64,
    // Codex R9 P2 fix — mirror Cap B `decide_synthesize`'s zero-weight
    // gate (R6 #6). When operator sets `weight_decay_rate = 0.0` for
    // judge-telemetry-only mode, judge events MUST NOT advance the
    // cold-start counter; otherwise the bucket graduates and falls
    // through to a useful_rate of 0 (judge contributions zeroed),
    // suppressing summary refresh despite zero judge influence intent.
    judge_weight_decay_rate: f64,
) -> ConceptSummaryDecision {
    // Operator override wins. Even with rich adaptive data, if the operator
    // disabled the global flag, the Cap-A surface is off.
    if !global_enabled {
        return ConceptSummaryDecision::Skip(ConceptSummarySkipReason::OperatorDisabled);
    }

    // Cold-start ladder: each missing piece falls back to "Yes" (matches
    // pre-feedback Cap A behaviour). The gate must NEVER skip silently just
    // because the per-query data is missing.
    //
    // `cluster_id = None` short-circuits to Yes — the bucket helper supports
    // a `-1` "no cluster" key, but we deliberately do NOT route the gate
    // through it: the global `-1` bucket aggregates events across ALL queries
    // that lacked a cluster (different queries, different characteristics),
    // so its `useful_rate` is too noisy to gate individual recalls on. The
    // global bucket is preserved for the consumer-side `/api/adaptive`
    // rollup, not for runtime gating.
    let Some(cid) = cluster_id else {
        return ConceptSummaryDecision::Yes;
    };
    let Some(state) = adaptive_state else {
        return ConceptSummaryDecision::Yes;
    };
    let Some(cs_state) = state.concept_summary_feedback_stats.as_ref() else {
        return ConceptSummaryDecision::Yes;
    };
    // v0.27.2 R2 P2 revert — the R1 fallback to global `(None, "unknown")`
    // bucket conflated Cap A auto-judge signal with metadata-less
    // human interactions that also fold into the same bucket. A low
    // `useful_rate` from unrelated human events could mistakenly skip
    // summaries for new clustered concepts. Restore v0.27.1 behavior:
    // miss on the per-cluster bucket → cold-start `Yes`. The deeper
    // R7-#1 architectural mismatch (concepts have no natural cluster
    // at refresh time, so Cap A auto-judge can never warm a clustered
    // bucket without recall-surface routing) is documented in spec
    // §15 as v0.28+ work — needs either per-concept routing or a
    // dedicated judge-only bucket subspace.
    let key = concept_summary_bucket_key(Some(cid), query_type);
    let Some(cluster) = cs_state.by_cluster.get(&key) else {
        return ConceptSummaryDecision::Yes;
    };
    // v0.27.1 E direction (Codex R8 P1 fix mirror): cold-start `total_signal`
    // counts ALL signals including LLM judge events. Without this, an
    // MCP-only canary with zero `viewed_count` but a warm `llm_judge_count`
    // bucket would fall back to the global flag forever — defeating the
    // E direction premise on the Cap A surface.
    let llm_contribution = if judge_weight_decay_rate > 0.0 {
        cluster.llm_judge_count
    } else {
        0
    };
    let total_signal = cluster
        .viewed_count
        .saturating_add(cluster.explicit_up)
        .saturating_add(cluster.explicit_down)
        .saturating_add(llm_contribution);
    if total_signal < cold_start_n {
        return ConceptSummaryDecision::Yes;
    }

    // Per-cluster gate: skip if learned useful_rate is below the bootstrap
    // threshold (cluster has acquired enough events to disagree with the
    // global default).
    if cluster.useful_rate >= CONCEPT_SUMMARY_USEFUL_RATE_THRESHOLD {
        ConceptSummaryDecision::Yes
    } else {
        ConceptSummaryDecision::Skip(ConceptSummarySkipReason::AdaptiveDecision)
    }
}

// ─── v0.27.1 E direction — Cap A mirror of recall_synthesis judge wiring ────

/// Cap A mirror of [`crate::ops::recall_synthesis::current_sample_rate`] —
/// reads the per-bucket human-signal aggregate off
/// `ConceptSummaryFeedbackState`. Pure function, testable in isolation.
fn current_sample_rate_concept_summary(
    bucket: Option<&ClusterConceptSummaryStats>,
    cfg: &crate::config::ArsLlmJudgeConfig,
) -> f64 {
    let human_count = bucket
        .map(|s| {
            s.explicit_up
                .saturating_add(s.explicit_down)
                .saturating_add(s.viewed_count)
        })
        .unwrap_or(0);
    if human_count >= cfg.human_signal_threshold {
        cfg.sample_rate_warm
    } else {
        cfg.sample_rate_cold_start
    }
}

/// Bernoulli sample matching the `recall_synthesis` impl. Kept private to
/// each module so neither becomes the source-of-truth and accidentally
/// drifts.
fn bernoulli_fire_concept_summary(rate: f64, salt: &str) -> bool {
    if rate <= 0.0 {
        return false;
    }
    if rate >= 1.0 {
        return true;
    }
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let mut h = std::collections::hash_map::DefaultHasher::new();
    use std::hash::{Hash, Hasher};
    nanos.hash(&mut h);
    salt.hash(&mut h);
    let n = h.finish();
    let frac = (n as f64) / (u64::MAX as f64);
    frac < rate
}

/// Enqueue the runtime LLM judge artifacts for a freshly-minted
/// concept-summary refresh. Mirrors
/// `recall_synthesis::enqueue_judge_for_synthesis`. Cluster routing for
/// the sample-rate ladder uses `cluster_id = None` because Concept rows
/// don't carry a cluster id — the bucket lookup folds into the
/// `(None, "")` global bucket via `concept_summary_bucket_key`.
fn enqueue_judge_for_concept_summary(
    config: &ReinConfig,
    adaptive_state: &AdaptiveState,
    summary_id: &str,
    concept_id: &str,
    prompt: &str,
    candidate: &str,
) {
    use crate::ops::handlers::judge::{
        append_jsonl_line, concept_summary_cache_path_for_config, judge_queue_path_for_config,
    };
    use crate::ops::llm_judge_worker::JudgeJob;

    // Cap A judge has no query-text equivalent — synthesis is from concept
    // revisions, not a recall query. Use the empty string so the J7 stamp
    // hash stays deterministic across runtime + cron passes.
    let query = "";
    // Codex R7+R8 P2 fix — same combined-cap truncation as
    // recall_synthesis (see that function for the algorithm + rationale).
    use crate::ops::llm_judge_worker::JUDGE_MAX_INPUT_CHARS;
    const CANDIDATE_RESERVE_MAX: usize = 4_096;
    const PROMPT_FLOOR: usize = 1_024;
    let candidate_capped: String = candidate
        .chars()
        .take(CANDIDATE_RESERVE_MAX.min(JUDGE_MAX_INPUT_CHARS / 4))
        .collect();
    let joiner_overhead = "\n\nCandidate:\n".len();
    let prompt_budget = JUDGE_MAX_INPUT_CHARS
        .saturating_sub(candidate_capped.chars().count())
        .saturating_sub(joiner_overhead)
        .max(PROMPT_FLOOR);
    let prompt_truncated: String = prompt.chars().take(prompt_budget).collect();
    let prompt = prompt_truncated.as_str();
    let candidate = candidate_capped.as_str();

    let stamp_hash = JudgeJob::compute_stamp_hash(query, prompt, candidate);
    let cache_entry = serde_json::json!({
        "concept_summary_id": summary_id,
        "concept_id": concept_id,
        "query": query,
        "prompt": prompt,
        "candidate": candidate,
        "stamp_hash": stamp_hash,
        "query_type": serde_json::Value::Null,
        "cluster_id": serde_json::Value::Null,
        "source_count": 0u32,
        "stamped_at": chrono::Utc::now().to_rfc3339(),
    });

    // (1) Cache write — feeds manual MCP rehydration via
    // `rein_judge_concept_summary`.
    let cache_path = concept_summary_cache_path_for_config(config);
    if let Err(e) = append_jsonl_line(&cache_path, &cache_entry) {
        tracing::warn!(
            target: "rein.judge",
            concept_summary_id = %summary_id,
            "concept_summary: failed to write judge cache entry: {e}",
        );
    }

    // (2) Sample-rate Bernoulli → judge worker queue.
    // Codex R6 P2 fix — bucket key alignment. The consumer normalizes
    // empty/null query_type to "unknown" before storing (per
    // store::adaptive F-11 query_type clamp); the sampler must use
    // the same normalized key, otherwise it looks up empty-string
    // bucket while counts accumulate under "unknown" and the warm
    // ladder never fires for Cap A auto-sampled paths.
    let bucket = adaptive_state
        .concept_summary_feedback_stats
        .as_ref()
        .and_then(|sfs| {
            sfs.by_cluster
                .get(&concept_summary_bucket_key(None, "unknown"))
        });
    let rate = current_sample_rate_concept_summary(bucket, &config.ars.llm_judge);
    if bernoulli_fire_concept_summary(rate, summary_id) {
        let job = serde_json::json!({
            "kind": "concept_summary",
            "surface_id": summary_id,
            "concept_id": concept_id,
            "query": query,
            "prompt": prompt,
            "candidate": candidate,
            "stamp_hash": stamp_hash,
            "source": "AutoSampled",
            "query_type": serde_json::Value::Null,
            "cluster_id": serde_json::Value::Null,
            "source_count": 0u32,
        });
        let queue_path = judge_queue_path_for_config(config);
        if let Err(e) = append_jsonl_line(&queue_path, &job) {
            tracing::warn!(
                target: "rein.judge",
                concept_summary_id = %summary_id,
                "concept_summary: failed to enqueue judge job: {e}",
            );
        }
    }

    // (3) Cron-archive deterministic sample. Cap A sits in the same
    // archive as Cap B (one file per day per shard), so the cron consumer
    // can join on `synthesis_id` OR `concept_summary_id` per spec §7.
    //
    // Codex R1 P2 fix — entry MUST match `CronArchiveEntry` shape.
    // The v0 enqueue reused `cache_entry` shape and was malformed; cron
    // skipped every Cap A archive line.
    if config.ars.llm_judge.nightly_cron.enabled
        && crate::ops::judge_calibration::should_archive_for_cron(
            summary_id,
            config.ars.llm_judge.nightly_cron.sample_rate,
        )
    {
        let date = chrono::Utc::now().date_naive();
        let archive_path = crate::ops::judge_calibration::cron_archive_path(config, date, 0);
        let archive_entry = serde_json::json!({
            "surface": "ConceptSummary",
            "id": summary_id,
            "concept_id": concept_id,
            "stamp_hash": stamp_hash,
            "query": query,
            "sources": [prompt],
            "candidate": candidate,
            "metadata": {
                "query_type": serde_json::Value::Null,
                "cluster_id": serde_json::Value::Null,
                "source_count": 0u32,
                "judge_latency_ms": serde_json::Value::Null,
            },
            "minted_at": chrono::Utc::now().timestamp(),
        });
        if let Err(e) = append_jsonl_line(&archive_path, &archive_entry) {
            tracing::warn!(
                target: "rein.judge",
                concept_summary_id = %summary_id,
                "concept_summary: failed to write cron-archive entry: {e}",
            );
        }
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
            living_summary_id: None,
        }
    }

    fn make_revision(concept_id: &str, revision: u32, definition: &str) -> ConceptRevision {
        ConceptRevision {
            id: format!("rev-{revision}"),
            concept_id: concept_id.to_string(),
            revision,
            definition: definition.to_string(),
            confidence: 0.8,
            labels: Vec::new(),
            source_memory_ids: Vec::new(),
            episode_id: None,
            created_at: Utc::now(),
        }
    }

    #[test]
    fn build_concept_summary_prompt_with_cap_truncates_cjk_safely() {
        let mut concept = make_concept("test-memoir", "bounded-concept");
        concept.id = "concept-bounded".to_string();
        concept.definition = "当前定义".repeat(300);
        let revisions = vec![
            make_revision(&concept.id, 1, &"旧版本事实".repeat(300)),
            make_revision(&concept.id, 2, &"第二版事实".repeat(300)),
        ];

        let prompt = build_concept_summary_prompt_with_cap(&concept, &revisions, 512);

        assert!(
            prompt.chars().count() <= 512,
            "prompt must honor max_input_chars using char count"
        );
        assert!(prompt.contains("Concept: bounded-concept"));
        assert!(prompt.is_char_boundary(prompt.len()));
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
            "cs_test_stale",
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
            "cs_test_fresh",
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
            "cs_test_winner_first",
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
            "cs_test_loser_first",
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
            "cs_test_initial",
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
            "cs_test_winner_steady",
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
            "cs_test_loser_steady",
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

    // ── v0.27 Cap A feedback loop: decide_concept_summary_quality tests ──

    use crate::store::adaptive::{
        ClusterConceptSummaryStats, ConceptSummaryFeedbackState, CONCEPT_SUMMARY_COLD_START_N,
    };
    use std::collections::HashMap;

    fn cs_state_with_bucket(
        cluster_id: i64,
        query_type: &str,
        bucket: ClusterConceptSummaryStats,
    ) -> AdaptiveState {
        let mut by_cluster = HashMap::new();
        by_cluster.insert(
            concept_summary_bucket_key(Some(cluster_id), query_type),
            bucket,
        );
        AdaptiveState {
            concept_summary_feedback_stats: Some(ConceptSummaryFeedbackState {
                by_cluster,
                ..ConceptSummaryFeedbackState::default()
            }),
            ..AdaptiveState::default()
        }
    }

    #[test]
    fn decide_concept_summary_cold_start_no_state_returns_yes() {
        let decision = decide_concept_summary_quality(
            true,
            Some(42),
            "Semantic",
            None,
            CONCEPT_SUMMARY_COLD_START_N,
            0.3,
        );
        assert_eq!(decision, ConceptSummaryDecision::Yes);
    }

    #[test]
    fn decide_concept_summary_cold_start_no_feedback_state_returns_yes() {
        let state = AdaptiveState::default(); // concept_summary_feedback_stats: None
        let decision = decide_concept_summary_quality(
            true,
            Some(42),
            "Semantic",
            Some(&state),
            CONCEPT_SUMMARY_COLD_START_N,
            0.3,
        );
        assert_eq!(decision, ConceptSummaryDecision::Yes);
    }

    #[test]
    fn decide_concept_summary_cold_start_no_cluster_id_returns_yes() {
        let state = cs_state_with_bucket(
            42,
            "Semantic",
            ClusterConceptSummaryStats {
                viewed_count: 100,
                useful_rate: 0.0, // would skip if it reached the gate
                ..ClusterConceptSummaryStats::default()
            },
        );
        let decision = decide_concept_summary_quality(
            true,
            None, // no cluster_id → cold-start fallback
            "Semantic",
            Some(&state),
            CONCEPT_SUMMARY_COLD_START_N,
            0.3,
        );
        assert_eq!(decision, ConceptSummaryDecision::Yes);
    }

    #[test]
    fn decide_concept_summary_cold_start_insufficient_samples_returns_yes() {
        let state = cs_state_with_bucket(
            42,
            "Semantic",
            ClusterConceptSummaryStats {
                viewed_count: CONCEPT_SUMMARY_COLD_START_N - 1,
                useful_rate: 0.0, // would skip if it reached the gate
                ..ClusterConceptSummaryStats::default()
            },
        );
        let decision = decide_concept_summary_quality(
            true,
            Some(42),
            "Semantic",
            Some(&state),
            CONCEPT_SUMMARY_COLD_START_N,
            0.3,
        );
        assert_eq!(decision, ConceptSummaryDecision::Yes);
    }

    #[test]
    fn decide_concept_summary_warm_cluster_above_threshold_returns_yes() {
        let state = cs_state_with_bucket(
            42,
            "Semantic",
            ClusterConceptSummaryStats {
                viewed_count: 100,
                useful_rate: 0.7,
                ..ClusterConceptSummaryStats::default()
            },
        );
        let decision = decide_concept_summary_quality(
            true,
            Some(42),
            "Semantic",
            Some(&state),
            CONCEPT_SUMMARY_COLD_START_N,
            0.3,
        );
        assert_eq!(decision, ConceptSummaryDecision::Yes);
    }

    #[test]
    fn decide_concept_summary_warm_cluster_below_threshold_returns_skip_adaptive() {
        let state = cs_state_with_bucket(
            42,
            "Semantic",
            ClusterConceptSummaryStats {
                viewed_count: 100,
                useful_rate: 0.2,
                ..ClusterConceptSummaryStats::default()
            },
        );
        let decision = decide_concept_summary_quality(
            true,
            Some(42),
            "Semantic",
            Some(&state),
            CONCEPT_SUMMARY_COLD_START_N,
            0.3,
        );
        assert_eq!(
            decision,
            ConceptSummaryDecision::Skip(ConceptSummarySkipReason::AdaptiveDecision)
        );
    }

    #[test]
    fn decide_concept_summary_operator_disabled_overrides_adaptive() {
        // Even with rich adaptive data, operator-off wins.
        let state = cs_state_with_bucket(
            42,
            "Semantic",
            ClusterConceptSummaryStats {
                viewed_count: 100,
                useful_rate: 0.99, // would say Yes if it reached the gate
                ..ClusterConceptSummaryStats::default()
            },
        );
        let decision = decide_concept_summary_quality(
            false, // operator disabled
            Some(42),
            "Semantic",
            Some(&state),
            CONCEPT_SUMMARY_COLD_START_N,
            0.3,
        );
        assert_eq!(
            decision,
            ConceptSummaryDecision::Skip(ConceptSummarySkipReason::OperatorDisabled)
        );
    }

    #[test]
    fn decide_concept_summary_query_type_partition_isolates_buckets() {
        // Episodic bucket has bad rate, but query is Semantic — that bucket
        // doesn't exist yet → cold-start Yes. Confirms the per-query partition.
        let state = cs_state_with_bucket(
            42,
            "Episodic",
            ClusterConceptSummaryStats {
                viewed_count: 100,
                useful_rate: 0.1,
                ..ClusterConceptSummaryStats::default()
            },
        );
        let decision = decide_concept_summary_quality(
            true,
            Some(42),
            "Semantic", // different query_type
            Some(&state),
            CONCEPT_SUMMARY_COLD_START_N,
            0.3,
        );
        assert_eq!(
            decision,
            ConceptSummaryDecision::Yes,
            "different query_type must route to its own bucket (cold-start Yes)"
        );
    }

    #[test]
    fn decide_concept_summary_custom_cold_start_n() {
        // With cold_start_n = 2, viewed_count = 5 is past the threshold and
        // useful_rate = 0.1 should trigger AdaptiveDecision skip.
        let state = cs_state_with_bucket(
            42,
            "Semantic",
            ClusterConceptSummaryStats {
                viewed_count: 5,
                useful_rate: 0.1,
                ..ClusterConceptSummaryStats::default()
            },
        );
        let decision_default =
            decide_concept_summary_quality(true, Some(42), "Semantic", Some(&state), 10, 0.3);
        assert_eq!(
            decision_default,
            ConceptSummaryDecision::Yes,
            "cold_start_n=10 not met (only 5 views) → Yes"
        );
        let decision_canary =
            decide_concept_summary_quality(true, Some(42), "Semantic", Some(&state), 2, 0.3);
        assert_eq!(
            decision_canary,
            ConceptSummaryDecision::Skip(ConceptSummarySkipReason::AdaptiveDecision),
            "cold_start_n=2 met and rate below threshold → Skip"
        );
    }
}
