//! `rein-eval` — standalone evaluation harness for the v0.23 resummerize
//! feature.
//!
//! - `baseline` — scores the keep-tail canonical (the fixture's
//!   `current_canonical` IS the keep-tail output, captured when the
//!   fixture was authored) via a simple keyword-overlap hit checker.
//!   No LLM required.
//! - `run` — scores the LLM-generated resummerized canonical. Requires a
//!   configured LLM provider (`[extract]` / `[resummerize]` sections in
//!   `~/.rein/config.toml`) — errors cleanly otherwise rather than
//!   emitting misleading placeholder data.
//! - `compare` — loads two scorecards, joins them by `case_id`, runs
//!   paired McNemar, and prints the ship-or-bail decision.
//!
//! ## Binary name
//!
//! Cargo uses the explicit `[[bin]]` entry in `crates/rein/Cargo.toml`, so
//! invoke as `cargo run -p rein --bin rein-eval -- ...`.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use chrono::{DateTime, Utc};
use clap::{Parser, Subcommand};
use rein::compression::contract::{self, ContractInput, EvidenceEntry};
use rein::config::ReinConfig;
use rein::eval::concept_summary::score_concept_case;
use rein::eval::{
    decide_ship, mcnemar, CategoryStats, DecideShipKind, HitChecker, JudgeMode,
    KeywordOverlapHitChecker, LlmJudgeHitChecker, McNemarResult, PairedOutcome, Scorecard,
    ShipDecision, ShipReason, DEFAULT_SEMANTIC_THRESHOLD, LLM_JUDGE_VERSION,
};
use rein::extract::llm::{strip_code_fences, ExtractorKind};
// NOTE: `call_llm_sync` in ops::resummerize uses SYSTEM_PROMPT internally —
// the eval bin doesn't need to import it directly. Importing `build_prompt`
// and `call_llm_sync` is enough; the system prompt travels with the call.
// `create_resummerize_extractor` is used instead of `create_extractor` so
// the eval honors `[resummerize].llm_backend` the same way production
// does (post-fix audit M-1).
use rein::ops::concept_summary::{
    build_concept_summary_prompt, call_llm_sync as call_concept_summary_llm_sync,
    create_concept_summary_extractor,
};
use rein::ops::recall_synthesis::{
    build_synthesis_prompt_with_count, call_synthesis_llm_sync, extract_citations,
};
use rein::ops::resummerize::{
    build_prompt, call_llm_sync as call_resummerize_llm_sync, create_resummerize_extractor,
};
use rein::search::recall::RecallResult;
use rein::types::{
    Concept, ConceptRevision, Importance, Memory, MemoryLayer, MemoryStatus, MemoryTier, Source,
};
use serde::{Deserialize, Serialize};

#[derive(Parser)]
#[command(
    name = "rein-eval",
    about = "Evaluation harness for rein features (v0.23 resummerize)",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Evaluation routines for the resummerize feature.
    Resummerize {
        #[command(subcommand)]
        action: ResummerizeAction,
    },
    /// Evaluation routines for the v0.24 ARS Capability A concept living-summary
    /// feature. Parallel to `resummerize`: baseline scores the raw concept
    /// definition, `run` invokes the LLM to produce a living summary and
    /// scores `definition + " " + living_summary`, and `compare` runs paired
    /// McNemar + ship/bail-out.
    ConceptSummary {
        #[command(subcommand)]
        action: ConceptSummaryAction,
    },
    /// Evaluation routines for the v0.25 ARS Capability B recall-time synthesis
    /// feature (A3 harness, v0.25.1). Parallel to `concept-summary`: baseline
    /// scores the raw concatenated recall summary text (what the operator
    /// would see without synthesis); `run` invokes the production synthesis
    /// LLM bridge over fixture recall results and scores its prose output;
    /// `compare` runs paired McNemar + ship/bail-out under the additive
    /// `DecideShipKind::Synthesis` rule.
    Synthesis {
        #[command(subcommand)]
        action: SynthesisAction,
    },
}

#[derive(Subcommand)]
enum ResummerizeAction {
    /// Score the keep-tail canonical baseline over a directory of fixture cases.
    /// The fixture's `current_canonical` field already IS the keep-tail state;
    /// this command measures its recall via keyword-overlap against each evidence
    /// entry. No LLM required.
    Baseline {
        /// Directory containing fixture JSON files (one per case).
        #[arg(long)]
        fixtures: PathBuf,
        /// Number of iterations per case (currently informational only).
        #[arg(long, default_value_t = 1)]
        iterations: u32,
        /// Output path for the scorecard JSON.
        #[arg(long, default_value = "baseline_scorecard.json")]
        output: PathBuf,
    },
    /// Run the resummerize treatment over a directory of fixture cases using
    /// the configured LLM. Errors cleanly if no provider is set. Emits a
    /// scorecard that `compare` can pair with a baseline scorecard.
    Run {
        /// Directory containing fixture JSON files (one per case).
        #[arg(long)]
        fixtures: PathBuf,
        /// Number of LLM calls per case (currently informational only;
        /// scorecard records each fixture once with the last response).
        /// Mirrors the baseline flag for symmetry.
        #[arg(long, default_value_t = 1)]
        iterations: u32,
        /// Output path for the scorecard JSON.
        #[arg(long, default_value = "treatment_scorecard.json")]
        output: PathBuf,
        /// Print per-case contract-fail diagnostics (invariant names + a
        /// 200-char preview of the LLM output). Off by default — useful
        /// for debugging why a treatment scorecard has a low hit rate
        /// without re-running the costly LLM passes.
        #[arg(long, default_value_t = false)]
        verbose: bool,
    },
    /// Compare a baseline and a treatment scorecard via paired McNemar and
    /// apply the ship-or-bail policy. Fully implemented.
    Compare {
        /// Path to the baseline scorecard JSON.
        #[arg(long)]
        baseline: PathBuf,
        /// Path to the treatment scorecard JSON.
        #[arg(long)]
        treatment: PathBuf,
        /// Hit-rate difference tolerated as noise when calling
        /// non-inferiority. Typically derived from baseline variance runs.
        #[arg(long, default_value_t = 0.03)]
        noise_floor: f64,
    },
}

#[derive(Subcommand)]
enum ConceptSummaryAction {
    /// Score the raw concept `definition` (no living summary) against each
    /// fixture's `evidence_keywords`. No LLM required — the fixture's
    /// `definition` IS the baseline state.
    Baseline {
        /// Directory containing concept-summary fixture JSON files.
        #[arg(long)]
        fixtures: PathBuf,
        /// Output path for the scorecard JSON.
        #[arg(long, default_value = "concept_summary_baseline_scorecard.json")]
        output: PathBuf,
    },
    /// Run the living-summary treatment: construct a synthetic `Concept` +
    /// revision list from each fixture, call the configured LLM via
    /// `build_concept_summary_prompt` + `create_concept_summary_extractor`,
    /// and score `definition + " " + living_summary` against the fixture's
    /// `evidence_keywords`. Errors cleanly if no LLM provider is configured.
    Run {
        /// Directory containing concept-summary fixture JSON files.
        #[arg(long)]
        fixtures: PathBuf,
        /// Output path for the scorecard JSON.
        #[arg(long, default_value = "concept_summary_treatment_scorecard.json")]
        output: PathBuf,
        /// Print per-case LLM output previews (200-char snippet) on failure
        /// or empty response. Useful for diagnosing a low hit rate without
        /// re-running the costly LLM passes. Off by default.
        #[arg(long, default_value_t = false)]
        verbose: bool,
    },
    /// Compare a baseline and a treatment scorecard via paired McNemar and
    /// apply the ship-or-bail policy (reuses the resummerize decision path).
    Compare {
        #[arg(long)]
        baseline: PathBuf,
        #[arg(long)]
        treatment: PathBuf,
        /// Hit-rate difference tolerated as noise when calling
        /// non-inferiority. Tighter than resummerize's 0.03 default because
        /// ARS Capability A is expected to materially improve keyword
        /// recall rather than hold hit-rate flat.
        #[arg(long, default_value_t = 0.02)]
        noise_floor: f64,
    },
}

#[derive(Subcommand)]
enum SynthesisAction {
    /// Score the raw concatenated recall text (what the operator would see
    /// without synthesis) against each fixture's `evidence_keywords`. No
    /// LLM required — the fixture's `recall_results` IS the baseline state.
    /// Concatenated text is `summary + " " + evidence_preview.join(" ")` per
    /// result, joined across results — mirrors the `RecallResult` surface
    /// the GUI / MCP client renders pre-synthesis.
    Baseline {
        /// Directory containing recall-synthesis fixture JSON files.
        #[arg(long)]
        fixtures: PathBuf,
        /// Output path for the scorecard JSON.
        #[arg(long, default_value = "synthesis_baseline_scorecard.json")]
        output: PathBuf,
    },
    /// Run the recall-synthesis treatment: synthesize `Vec<RecallResult>` from
    /// each fixture, build the production prompt via `build_synthesis_prompt`,
    /// call the configured LLM via `call_synthesis_llm_sync` (the same
    /// `[ars].llm_backend` resolution path production uses through
    /// `create_concept_summary_extractor`), and score the LLM prose output
    /// against `evidence_keywords`. Errors cleanly if no LLM provider is
    /// configured.
    Run {
        /// Directory containing recall-synthesis fixture JSON files.
        #[arg(long)]
        fixtures: PathBuf,
        /// Output path for the scorecard JSON.
        #[arg(long, default_value = "synthesis_treatment_scorecard.json")]
        output: PathBuf,
        /// Print per-case LLM output previews (200-char snippet) on failure
        /// or empty response. Off by default.
        #[arg(long, default_value_t = false)]
        verbose: bool,
    },
    /// Compare a baseline and a treatment scorecard via paired McNemar and
    /// apply the additive synthesis ship-or-bail policy
    /// (`DecideShipKind::Synthesis` — non-inferior CI lower bound, length
    /// ignored because synthesis is by design longer than raw recall text).
    Compare {
        #[arg(long)]
        baseline: PathBuf,
        #[arg(long)]
        treatment: PathBuf,
        /// Hit-rate difference tolerated as noise when calling
        /// non-inferiority. Default 0.02 (matches concept-summary; ARS
        /// Capability B is also expected to be additive, not regressive).
        #[arg(long, default_value_t = 0.02)]
        noise_floor: f64,
    },
}

/// Fixture schema mirroring the JSON layout under
/// `crates/rein/tests/fixtures/resummerize/`. Every field except `case_id`
/// is optional so partially-populated fixtures still parse; the commands
/// that need a specific field surface a clear error when it's missing.
#[derive(Debug, Deserialize, Serialize)]
struct Fixture {
    case_id: String,
    #[serde(default)]
    category: Option<String>,
    /// The keep-tail canonical as captured when the fixture was authored.
    /// This IS the baseline state — no execution is needed to produce it.
    #[serde(default)]
    current_canonical: Option<String>,
    /// Merge-history entries whose content the resummerize output MUST be
    /// able to recall. The `content` of each entry is used as the query
    /// for the hit check.
    #[serde(default)]
    evidence: Vec<FixtureEvidenceEntry>,
    /// Target byte budget for the resummerize output (supplied to the LLM
    /// and checked by `length_bounded`). Optional because baseline doesn't
    /// need it; `run` errors cleanly when absent.
    #[serde(default)]
    target_bytes: Option<usize>,
    /// Legacy fields retained for forward compatibility with older fixtures.
    #[serde(default)]
    canonical: Option<String>,
    #[serde(default)]
    context: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
struct FixtureEvidenceEntry {
    content: String,
    #[serde(default)]
    merged_at: Option<String>,
}

impl Fixture {
    /// Resolve the keep-tail canonical, falling back through legacy
    /// field names.
    fn effective_canonical(&self) -> Option<&str> {
        self.current_canonical
            .as_deref()
            .or(self.canonical.as_deref())
            .or(self.context.as_deref())
    }
}

/// Fixture schema for the v0.24 ARS Capability A concept-summary eval.
///
/// `definition` is the current canonical definition of the concept (what
/// `Concept.definition` would hold in the DB). `revisions` carries the
/// historical snapshots (oldest first; the last entry's `definition` should
/// equal the top-level `definition` — mirrors the DB invariant that
/// `concept_revisions` includes a row for the current state). The harness
/// synthesizes a `Concept` + `Vec<ConceptRevision>` from these to feed
/// `build_concept_summary_prompt`.
///
/// `evidence_keywords` are the terms whose presence in `definition` +
/// optional `living_summary` determines a scored hit — see
/// [`rein::eval::concept_summary::score_concept_case`].
#[derive(Debug, Deserialize, Serialize)]
struct ConceptFixture {
    case_id: String,
    #[serde(default)]
    category: Option<String>,
    name: String,
    definition: String,
    #[serde(default)]
    revisions: Vec<ConceptFixtureRevision>,
    #[serde(default)]
    evidence_keywords: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize)]
struct ConceptFixtureRevision {
    revision: u32,
    definition: String,
    created_at: String,
}

impl ConceptFixture {
    /// Build a synthetic `Concept` for `build_concept_summary_prompt`. The
    /// `revision` field is taken from the last revision's number so a
    /// downstream "revisions since last summary" computation (if Agent 1
    /// uses it) sees a consistent value.
    fn to_concept(&self) -> Concept {
        let now = Utc::now();
        let last_rev = self.revisions.last().map(|r| r.revision).unwrap_or(1);
        Concept {
            id: format!("concept:{}", self.case_id),
            memoir_id: "eval_memoir".to_string(),
            name: self.name.clone(),
            definition: self.definition.clone(),
            labels: vec![],
            source_memory_ids: vec![],
            confidence: 1.0,
            revision: last_rev,
            last_episode_id: None,
            created_at: self
                .revisions
                .first()
                .and_then(|r| DateTime::parse_from_rfc3339(&r.created_at).ok())
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or(now),
            updated_at: self
                .revisions
                .last()
                .and_then(|r| DateTime::parse_from_rfc3339(&r.created_at).ok())
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or(now),
            living_summary: None,
            living_summary_updated_at: None,
            living_summary_source_revision: None,
        }
    }

    /// Build the synthetic `Vec<ConceptRevision>` for the prompt. Ordered
    /// oldest-first, which is how the DB returns revisions when queried
    /// via `list_concept_revisions`.
    fn to_revisions(&self) -> Vec<ConceptRevision> {
        let concept_id = format!("concept:{}", self.case_id);
        self.revisions
            .iter()
            .map(|r| {
                let created = DateTime::parse_from_rfc3339(&r.created_at)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now());
                ConceptRevision {
                    id: format!("{concept_id}:rev:{}", r.revision),
                    concept_id: concept_id.clone(),
                    revision: r.revision,
                    definition: r.definition.clone(),
                    confidence: 1.0,
                    labels: vec![],
                    source_memory_ids: vec![],
                    episode_id: None,
                    created_at: created,
                }
            })
            .collect()
    }
}

/// Fixture schema for the v0.25 ARS Capability B recall-time synthesis eval
/// (A3 harness, v0.25.1).
///
/// Each fixture captures a recall scenario: the user `query`, the
/// `recall_results` that were surfaced (synthetic `Memory`-shaped rows the
/// production prompt would receive), and the `evidence_keywords` whose
/// presence in the synthesized prose determines a hit. Baseline scoring
/// concatenates `summary + " " + evidence_preview.join(" ")` across all
/// results — this is the "what the operator would learn from raw recall
/// without synthesis" surface. Treatment scoring runs the LLM through the
/// production `build_synthesis_prompt` + `call_synthesis_llm_sync` path
/// and scores the prose output against the same keyword set.
///
/// The schema is deliberately decoupled from `Memory` field churn: the
/// harness builds a synthetic `Memory` from `id` + `summary` (used as
/// `Memory.content`) so future field additions don't break old fixtures.
#[derive(Debug, Deserialize, Serialize)]
struct SynthesisFixture {
    case_id: String,
    #[serde(default)]
    category: Option<String>,
    query: String,
    recall_results: Vec<SyntheticRecallResult>,
    evidence_keywords: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize)]
struct SyntheticRecallResult {
    id: String,
    summary: String,
    #[serde(default)]
    evidence_preview: Vec<String>,
    score: f32,
    confidence: f32,
    sources_hit: usize,
    evidence_count: usize,
}

impl SynthesisFixture {
    /// Build a `Vec<RecallResult>` mirroring what production would feed to
    /// `build_synthesis_prompt`. The synthetic `Memory` only needs the
    /// fields the prompt builder actually reads (`topic` + `content`); all
    /// other fields take safe defaults that match the prompt's null
    /// expectations. This keeps the fixture schema minimal — a new `Memory`
    /// field added later doesn't invalidate the corpus.
    fn to_recall_results(&self) -> Vec<RecallResult> {
        let now = Utc::now();
        self.recall_results
            .iter()
            .map(|r| {
                let memory = Memory {
                    id: r.id.clone(),
                    layer: MemoryLayer::LTM,
                    // Use the result id as the topic — `build_synthesis_prompt`
                    // emits `[N] Topic: {topic}` headers, so the id surfaces
                    // in the prompt and any LLM hallucination tracing back
                    // to a non-fixture id is detectable.
                    topic: r.id.clone(),
                    summary: r.summary.clone(),
                    // The prompt body uses `memory.content` — the fixture's
                    // `summary` field IS the content the LLM sees. We do
                    // NOT splice `evidence_preview` here: production never
                    // includes evidence_preview in the synthesis prompt
                    // (it's a UI-only surface), so the eval must match.
                    content: r.summary.clone(),
                    keywords: vec![],
                    importance: Importance::Medium,
                    source: Source::Manual,
                    strength: 1.0,
                    decay_lambda: 0.06,
                    access_count: 0,
                    superseded_by: None,
                    canonical_id: None,
                    support_count: 1,
                    merge_count: 0,
                    dedup_confidence: 1.0,
                    source_diversity: 1.0,
                    contradiction_score: 0.0,
                    related_ids: vec![],
                    concept_ids: vec![],
                    status: MemoryStatus::default(),
                    embedding: None,
                    tier: MemoryTier::Warm,
                    cluster_id: None,
                    archival_summary: None,
                    archival_summary_at: None,
                    archival_summary_version: None,
                    created_at: now,
                    updated_at: now,
                    last_accessed: now,
                };
                RecallResult {
                    memory,
                    score: r.score,
                    confidence: r.confidence,
                    sources_hit: r.sources_hit,
                    evidence_count: r.evidence_count,
                    evidence_preview: r.evidence_preview.clone(),
                    // v0.26 Cap C: synthetic eval rows synthesise from
                    // `treatment_response`, not the live recall pipeline,
                    // so they never carry an archival summary.
                    archival_summary: None,
                }
            })
            .collect()
    }

    /// Concatenate the user-visible recall surface for baseline scoring:
    /// per-result `summary` + `evidence_preview` joined. This is what an
    /// operator skimming a `rein_recall` response sees today (pre-Cap B);
    /// scoring it against `evidence_keywords` measures the "do the raw
    /// results already mention the answer?" floor. Treatment scoring then
    /// asks "does the synthesized prose preserve that?".
    fn baseline_text(&self) -> String {
        let mut buf =
            String::with_capacity(self.recall_results.iter().map(|r| r.summary.len() + 64).sum());
        for r in &self.recall_results {
            buf.push_str(&r.summary);
            buf.push(' ');
            for ev in &r.evidence_preview {
                buf.push_str(ev);
                buf.push(' ');
            }
        }
        buf
    }

    /// Per-result enriched source material for the LLM judge: each source's
    /// `summary` enriched with its `evidence_preview` items so the judge
    /// sees the same ground truth the baseline_text exposes and the
    /// synthesis LLM was given. Without this enrichment the judge can flag
    /// a perfectly-faithful candidate as "hallucinated" when the candidate
    /// (correctly) draws from evidence_preview but the source list passed
    /// to the judge contained only `summary` — mismatch caught on amb_002
    /// when judge said "no LLM call" was unsupported by source [#3] even
    /// though that string lived in [#3]'s evidence_preview.
    fn judge_source_summaries(&self) -> Vec<String> {
        self.recall_results
            .iter()
            .map(|r| {
                if r.evidence_preview.is_empty() {
                    r.summary.clone()
                } else {
                    format!("{} ({})", r.summary, r.evidence_preview.join("; "))
                }
            })
            .collect()
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Resummerize { action } => match action {
            ResummerizeAction::Baseline {
                fixtures,
                iterations,
                output,
            } => cmd_baseline(&fixtures, iterations, &output),
            ResummerizeAction::Run {
                fixtures,
                iterations,
                output,
                verbose,
            } => cmd_run(&fixtures, iterations, &output, verbose),
            ResummerizeAction::Compare {
                baseline,
                treatment,
                noise_floor,
            } => cmd_compare(
                &baseline,
                &treatment,
                noise_floor,
                DecideShipKind::Compression,
            ),
        },
        Commands::ConceptSummary { action } => match action {
            ConceptSummaryAction::Baseline { fixtures, output } => {
                cmd_concept_summary_baseline(&fixtures, &output)
            }
            ConceptSummaryAction::Run {
                fixtures,
                output,
                verbose,
            } => cmd_concept_summary_run(&fixtures, &output, verbose),
            ConceptSummaryAction::Compare {
                baseline,
                treatment,
                noise_floor,
            } => cmd_compare(
                &baseline,
                &treatment,
                noise_floor,
                DecideShipKind::Synthesis,
            ),
        },
        Commands::Synthesis { action } => match action {
            SynthesisAction::Baseline { fixtures, output } => {
                cmd_synthesis_baseline(&fixtures, &output)
            }
            SynthesisAction::Run {
                fixtures,
                output,
                verbose,
            } => cmd_synthesis_run(&fixtures, &output, verbose),
            SynthesisAction::Compare {
                baseline,
                treatment,
                noise_floor,
            } => cmd_compare(
                &baseline,
                &treatment,
                noise_floor,
                DecideShipKind::Synthesis,
            ),
        },
    }
}

// --- baseline / run -------------------------------------------------------

/// Build a [`KeywordOverlapHitChecker`]. When config exposes an embedder
/// (`[embedding].provider` resolved + API key present), attach it as the
/// v3 semantic fallback so morphologically-distant synonyms (e.g.
/// "Ebbinghaus" ≈ "decay") still score as hits. Threshold defaults to
/// [`DEFAULT_SEMANTIC_THRESHOLD`] but can be overridden via the
/// `REIN_EVAL_SEMANTIC_THRESHOLD` env var (clamped to `[0.0, 1.0]`;
/// out-of-range values fall back to the default with a warning).
///
/// Set `REIN_EVAL_DISABLE_SEMANTIC=1` (or any non-empty value) to force
/// stem-only mode even when an embedder is configured — used for the A/B
/// "did semantic actually help" comparison without having to unset
/// `GEMINI_API_KEY` (which the synthesis LLM still needs).
///
/// **Symmetry invariant**: the same call must be used for baseline AND
/// treatment of the same paired comparison so McNemar's per-case outcomes
/// are computed under one methodology — drift here would bias the test.
fn build_hybrid_checker(config: &ReinConfig) -> KeywordOverlapHitChecker {
    if std::env::var("REIN_EVAL_DISABLE_SEMANTIC")
        .map(|v| !v.is_empty())
        .unwrap_or(false)
    {
        eprintln!("[rein-eval] REIN_EVAL_DISABLE_SEMANTIC set — using stem-only checker");
        return KeywordOverlapHitChecker::stem_only();
    }
    let Some(embedder) = rein::embed::create_embedder(config) else {
        return KeywordOverlapHitChecker::stem_only();
    };
    let threshold = std::env::var("REIN_EVAL_SEMANTIC_THRESHOLD")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .filter(|t| (0.0..=1.0).contains(t))
        .unwrap_or(DEFAULT_SEMANTIC_THRESHOLD);
    eprintln!(
        "[rein-eval] hybrid hit checker enabled — semantic fallback threshold={:.3}",
        threshold
    );
    KeywordOverlapHitChecker::with_semantic(std::sync::Arc::new(embedder), threshold)
}

/// Build an LLM judge if `REIN_EVAL_JUDGE=llm` AND the configured
/// extractor is available. Returns None to fall back to keyword-overlap
/// path. Mode parameter selects synthesis vs concept-summary judging
/// shape (the prompts differ).
///
/// Symmetry invariant: same `mode` value must be used for the baseline +
/// treatment of one paired comparison so judgment shape stays consistent.
fn build_judge(config: &ReinConfig, mode: JudgeMode) -> Option<LlmJudgeHitChecker> {
    let enabled = std::env::var("REIN_EVAL_JUDGE")
        .map(|v| v.eq_ignore_ascii_case("llm"))
        .unwrap_or(false);
    if !enabled {
        return None;
    }
    // Mode picks the LLM backend the same way the production scoring
    // path picks it: synthesis judging shares Cap B's `[ars].llm_backend`,
    // concept-summary judging shares Cap A's `create_concept_summary_extractor`.
    let extractor = match mode {
        JudgeMode::SynthesisSourceCoverage => {
            rein::ops::concept_summary::create_concept_summary_extractor(config)
        }
        JudgeMode::ConceptSummaryFactCoverage => {
            rein::ops::concept_summary::create_concept_summary_extractor(config)
        }
    }?;
    eprintln!(
        "[rein-eval] LLM judge enabled (mode={:?}, version={})",
        mode, LLM_JUDGE_VERSION
    );
    Some(LlmJudgeHitChecker::new(
        std::sync::Arc::new(extractor),
        mode,
    ))
}

fn cmd_baseline(fixtures: &Path, iterations: u32, output: &Path) -> Result<()> {
    let fixtures_list = load_fixtures(fixtures)?;
    if fixtures_list.is_empty() {
        bail!("no fixtures found in {}", fixtures.display());
    }

    // Resummerize baseline uses `check_hit` (top-5 frequency overlap),
    // which doesn't dispatch the semantic fallback. Stem-only is the
    // honest configuration here — adding an embedder would be no-op cost.
    let checker = KeywordOverlapHitChecker::stem_only();
    let mut outcomes = Vec::with_capacity(fixtures_list.len());
    let mut skipped = 0usize;

    for fx in &fixtures_list {
        let Some(canonical) = fx.effective_canonical() else {
            eprintln!(
                "[rein-eval] baseline: skipping case {} (no current_canonical field)",
                fx.case_id
            );
            skipped += 1;
            continue;
        };
        if fx.evidence.is_empty() {
            eprintln!(
                "[rein-eval] baseline: skipping case {} (no evidence entries)",
                fx.case_id
            );
            skipped += 1;
            continue;
        }

        // Strict hit criterion: the canonical must recall EVERY evidence
        // entry's content to count as a baseline hit. Matches the
        // Lossless Compression Contract's "no facts dropped" spirit and
        // produces a clean binary outcome McNemar can consume.
        let all_recalled = fx
            .evidence
            .iter()
            .all(|e| checker.check_hit(&e.content, canonical));

        outcomes.push(PairedOutcome {
            case_id: fx.case_id.clone(),
            baseline_hit: all_recalled,
            // Treatment is measured by `cmd_run`; fill sentinel values here
            // so the `compare` path's merge-by-case_id logic has complete
            // rows when this baseline is joined with a later treatment run.
            treatment_hit: false,
            baseline_length: canonical.len(),
            treatment_length: 0,
            treatment_summary: None,
        });
    }

    if outcomes.is_empty() {
        bail!(
            "no fixtures in {} had both `current_canonical` and `evidence` fields — \
             baseline scoring requires both",
            fixtures.display()
        );
    }

    let category_map = build_category_map(&fixtures_list);
    let sc = Scorecard {
        fixtures_dir: fixtures.display().to_string(),
        iterations,
        timestamp: Utc::now(),
        outcomes,
        per_category: HashMap::new(),
        category_map,
        hit_checker_version: rein::eval::HIT_CHECKER_VERSION,
    };
    write_scorecard(output, &sc)?;

    eprintln!(
        "[rein-eval] baseline: wrote {} scored cases ({} skipped) to {}",
        sc.outcomes.len(),
        skipped,
        output.display()
    );
    Ok(())
}

fn cmd_run(fixtures: &Path, iterations: u32, output: &Path, verbose: bool) -> Result<()> {
    // Load config from $REIN_CONFIG / ~/.config/rein/config.toml; env vars
    // (e.g. GEMINI_API_KEY) override at the same priority used by `rein` and
    // hooks. If no provider is set we bail before parsing fixtures — a
    // missing API key is the most common failure mode here and surfaces
    // best at the top.
    let config = ReinConfig::load().context("loading rein config for cmd_run")?;
    // Use the production backend-selection path so eval honors
    // `[resummerize].llm_backend` (inherit / google / omlx / none) the same
    // way `ops::resummerize::run_resummerize` does. Without this, an
    // operator who configured a different resummerize backend would see
    // `compare` verdicts that don't reflect production behavior.
    let extractor = create_resummerize_extractor(&config).ok_or_else(|| {
        anyhow!(
            "no LLM extractor available — `[resummerize].llm_backend` resolved to None or \
             the configured provider is missing its API key. Set \
             `[resummerize].llm_backend = \"inherit\"` to follow `[extract].provider`, \
             or explicitly set `[resummerize].llm_backend = \"google\"` with \
             GEMINI_API_KEY, or `\"omlx\"` with a configured `[extract.omlx]` block."
        )
    })?;

    let fixtures_list = load_fixtures(fixtures)?;
    if fixtures_list.is_empty() {
        bail!("no fixtures found in {}", fixtures.display());
    }

    let extractor_tag = match &extractor {
        rein::extract::llm::ExtractorKind::Gemini(_) => "gemini",
        rein::extract::llm::ExtractorKind::Omlx(_) => "omlx",
        #[cfg(feature = "test-support")]
        rein::extract::llm::ExtractorKind::Mock(_) => "mock",
    };
    eprintln!(
        "[rein-eval] run: {} fixtures, extractor={}, iterations={}",
        fixtures_list.len(),
        extractor_tag,
        iterations,
    );

    run_treatment_with_extractor(
        &fixtures_list,
        &extractor,
        iterations,
        output,
        fixtures,
        verbose,
    )
}

/// Treatment loop extracted so unit tests can drive it with a `MockExtractor`
/// without hitting a live provider. Production callers go through `cmd_run`,
/// which loads config + builds the extractor once.
fn run_treatment_with_extractor(
    fixtures_list: &[Fixture],
    extractor: &rein::extract::llm::ExtractorKind,
    iterations: u32,
    output: &Path,
    fixtures_dir_for_meta: &Path,
    verbose: bool,
) -> Result<()> {
    // Resummerize treatment scoring also uses `check_hit` (no per-keyword
    // semantic dispatch), so stem-only matches the baseline path's
    // configuration.
    let checker = KeywordOverlapHitChecker::stem_only();
    let mut outcomes: Vec<PairedOutcome> = Vec::with_capacity(fixtures_list.len());
    let mut skipped = 0usize;
    let mut llm_failed = 0usize;
    let mut contract_failed = 0usize;

    for fx in fixtures_list {
        let Some(canonical_str) = fx.effective_canonical() else {
            eprintln!(
                "[rein-eval] run: skipping {} (no current_canonical)",
                fx.case_id
            );
            skipped += 1;
            continue;
        };
        if fx.evidence.is_empty() {
            eprintln!("[rein-eval] run: skipping {} (no evidence)", fx.case_id);
            skipped += 1;
            continue;
        }
        let Some(target_bytes) = fx.target_bytes else {
            // Bailing per-fixture (rather than defaulting) keeps the
            // harness honest — a fixture without target_bytes can't be
            // contract-gated the same way production canonicals are.
            eprintln!(
                "[rein-eval] run: skipping {} (no target_bytes — required for treatment)",
                fx.case_id
            );
            skipped += 1;
            continue;
        };

        // Build EvidenceEntry vec mirroring ops/resummerize.rs:513-519:
        // contract only uses `content`, but we still parse merged_at so
        // build_prompt's "merged at YYYY-MM-DD" line matches production.
        let evidence_entries: Vec<EvidenceEntry> = fx
            .evidence
            .iter()
            .map(|e| EvidenceEntry {
                content: e.content.clone(),
                merged_at: parse_merged_at(e.merged_at.as_deref()),
            })
            .collect();
        let input = ContractInput {
            evidence: &evidence_entries,
            current_canonical: canonical_str,
            target_bytes,
        };

        // SHARED prompt — `build_prompt` and `SYSTEM_PROMPT` come from
        // ops/resummerize.rs verbatim. Drift here would invalidate the
        // McNemar comparison.
        let prompt = build_prompt(&input);

        let mut last_output: Option<String> = None;
        let mut last_err: Option<String> = None;
        for _ in 0..iterations.max(1) {
            match call_resummerize_llm_sync(extractor, &prompt) {
                Ok(text) => {
                    last_output = Some(strip_code_fences(&text));
                    last_err = None;
                }
                Err(e) => {
                    last_err = Some(format!("{e}"));
                }
            }
        }

        // Production behavior on LLM error OR contract fail: keep-tail
        // stays in effect. The eval must reflect that — treatment is
        // effectively baseline in these cases, so `treatment_length` has
        // to equal `baseline_length` (otherwise `avg_length_ratio` gets
        // misattributed as "shorter!" for cases that were actually
        // rejected and never rewrote the canonical). Hit rate is also
        // the baseline's keyword-overlap hit — we don't know the
        // baseline's hit yet (filled in at `compare` time from the
        // baseline scorecard), so `treatment_hit` starts false and
        // `compare` will pair it correctly.
        let canonical_len = canonical_str.len();
        let Some(llm_output) = last_output else {
            eprintln!(
                "[rein-eval] run: {} LLM error (last: {})",
                fx.case_id,
                last_err.as_deref().unwrap_or("unknown")
            );
            llm_failed += 1;
            outcomes.push(PairedOutcome {
                case_id: fx.case_id.clone(),
                baseline_hit: false,
                treatment_hit: false,
                baseline_length: canonical_len,
                // Keep-tail stays in effect → effective treatment ==
                // baseline length.
                treatment_length: canonical_len,
                treatment_summary: None,
            });
            continue;
        };

        // Contract gate — production rewrites only on Ok(()); on
        // violation the canonical stays unchanged (= keep-tail). The eval
        // mirrors this: contract-failed output is NOT scored for hits,
        // AND its treatment_length reverts to baseline_length because
        // production never applied the LLM's shorter candidate.
        let contract_result = contract::check_all(&input, &llm_output);
        let contract_ok = contract_result.is_ok();
        if let Err(violations) = &contract_result {
            contract_failed += 1;
            if verbose {
                // Diagnostic: surface which invariants tripped + a 200-char
                // snippet of the LLM output so an operator can tell whether
                // failures are "LLM doing reasonable paraphrastic
                // compression the contract rejects" vs "LLM producing
                // garbage / refusal / JSON-wrapped output / wrong
                // language". One is a calibration issue, the other is a
                // setup bug. Off by default to keep batch runs quiet.
                let names: Vec<&str> = violations.iter().map(|v| v.invariant).collect();
                let snippet: String = llm_output.chars().take(200).collect();
                eprintln!(
                    "[rein-eval] run: {} contract fail: {} | llm_out[..200]={:?}",
                    fx.case_id,
                    names.join(","),
                    snippet,
                );
            }
        }
        let treatment_hit = if contract_ok {
            // Strict "every evidence must be recalled" — byte-identical
            // to baseline's hit predicate (cmd_baseline:207-210).
            evidence_entries
                .iter()
                .all(|e| checker.check_hit(&e.content, &llm_output))
        } else {
            false
        };
        let treatment_length = if contract_ok {
            llm_output.len()
        } else {
            canonical_len
        };

        outcomes.push(PairedOutcome {
            case_id: fx.case_id.clone(),
            baseline_hit: false, // filled in by `compare` join from the baseline scorecard.
            treatment_hit,
            baseline_length: canonical_len,
            treatment_length,
            treatment_summary: None,
        });
    }

    if outcomes.is_empty() {
        bail!(
            "no fixtures in {} produced a scorable treatment outcome (skipped={})",
            fixtures_dir_for_meta.display(),
            skipped,
        );
    }

    // Carry the case_id -> category map so `compare` can group joined
    // paired outcomes by category. We deliberately do NOT pre-compute
    // per_category here: McNemar over treatment-only outcomes (with
    // sentinel baseline_hit=false) would be nonsense, and `compare`'s
    // current code path trusts a non-empty `per_category` and skips
    // recomputation. Better to ship the raw map and let `compare` derive
    // per-category stats from the JOINED data once.
    let category_map = build_category_map(fixtures_list);

    let sc = Scorecard {
        fixtures_dir: fixtures_dir_for_meta.display().to_string(),
        iterations,
        timestamp: Utc::now(),
        outcomes,
        per_category: HashMap::new(),
        category_map,
        hit_checker_version: rein::eval::HIT_CHECKER_VERSION,
    };
    write_scorecard(output, &sc)?;

    eprintln!(
        "[rein-eval] run: wrote {} scored cases ({} skipped, {} llm_failed, {} contract_failed) to {}",
        sc.outcomes.len(),
        skipped,
        llm_failed,
        contract_failed,
        output.display()
    );
    Ok(())
}

/// Best-effort parse of fixture `merged_at`. Falls back to `Utc::now()` so
/// build_prompt's "merged at YYYY-MM-DD" line is still well-formed; the
/// contract checks don't depend on this value (they only look at
/// `content`), so the fallback only affects prompt formatting.
fn parse_merged_at(s: Option<&str>) -> DateTime<Utc> {
    s.and_then(|raw| DateTime::parse_from_rfc3339(raw).ok())
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or_else(Utc::now)
}

/// Build the `case_id -> category` map from fixture metadata. Used by
/// `baseline` and `run` to populate `Scorecard.category_map`; `compare`
/// uses this map (when present in either scorecard) to group joined
/// paired outcomes for per-category McNemar.
fn build_category_map(fixtures_list: &[Fixture]) -> HashMap<String, String> {
    fixtures_list
        .iter()
        .filter_map(|fx| {
            fx.category
                .as_ref()
                .map(|c| (fx.case_id.clone(), c.clone()))
        })
        .collect()
}

// --- compare (fully implemented) -------------------------------------------

fn cmd_compare(
    baseline: &Path,
    treatment: &Path,
    noise_floor: f64,
    kind: DecideShipKind,
) -> Result<()> {
    let base: Scorecard = load_scorecard(baseline)?;
    let treat: Scorecard = load_scorecard(treatment)?;

    // Post-fix audit M-2: refuse to pair scorecards produced under
    // different `HIT_CHECKER_VERSION`s. v1 was Latin-only
    // `is_alphanumeric` tokenize (broken on CJK — every Chinese sentence
    // collapsed into a single mega-token). v2 routes CJK through jieba.
    // Running McNemar across the two methodologies produces numbers that
    // look plausible but reflect two different scorers, not two runs of
    // the same scorer against baseline vs treatment pipelines.
    //
    // `hit_checker_version == 0` is the pre-version-tracking sentinel for
    // scorecards written before the field existed. Refuse to mix 0 with
    // any non-zero version — operators should re-run baseline with the
    // current binary before comparing.
    if base.hit_checker_version != treat.hit_checker_version {
        bail!(
            "scorecard `hit_checker_version` mismatch: baseline={} vs treatment={} — \
             the hit predicate is different, so pairing their outcomes via McNemar \
             would compare two scoring methodologies rather than two pipelines. \
             Re-run whichever scorecard is older with the current binary so both \
             sides share a version.",
            base.hit_checker_version,
            treat.hit_checker_version,
        );
    }

    // Merge by case_id. Take baseline_hit/baseline_length from the baseline
    // scorecard and treatment_hit/treatment_length from the treatment
    // scorecard. Cases that only appear in one file are counted and reported.
    let base_by_id: HashMap<&str, &PairedOutcome> = base
        .outcomes
        .iter()
        .map(|o| (o.case_id.as_str(), o))
        .collect();
    let treat_by_id: HashMap<&str, &PairedOutcome> = treat
        .outcomes
        .iter()
        .map(|o| (o.case_id.as_str(), o))
        .collect();

    let mut paired: Vec<PairedOutcome> = Vec::new();
    for (id, base_o) in &base_by_id {
        if let Some(treat_o) = treat_by_id.get(id) {
            paired.push(PairedOutcome {
                case_id: base_o.case_id.clone(),
                baseline_hit: base_o.baseline_hit,
                treatment_hit: treat_o.treatment_hit,
                baseline_length: base_o.baseline_length,
                treatment_length: treat_o.treatment_length,
                treatment_summary: None,
            });
        }
    }

    let only_in_baseline = base_by_id
        .keys()
        .filter(|k| !treat_by_id.contains_key(*k))
        .count();
    let only_in_treatment = treat_by_id
        .keys()
        .filter(|k| !base_by_id.contains_key(*k))
        .count();
    if only_in_baseline > 0 || only_in_treatment > 0 {
        eprintln!(
            "[rein-eval] case_id mismatch: {only_in_baseline} only in baseline, \
             {only_in_treatment} only in treatment (ignored)"
        );
    }

    if paired.is_empty() {
        bail!("no paired cases found between baseline and treatment scorecards");
    }

    // Overall McNemar.
    let overall = mcnemar(&paired);

    // Per-category McNemar where a category is provided on either side.
    // Prefer the baseline scorecard's category_stats; if empty, try to infer
    // from outcome.case_id prefix-before-colon (very lightweight convention).
    let per_category = compute_per_category(&paired, &base, &treat);

    // Average length ratio.
    let (mean_base_len, mean_treat_len) = mean_lengths(&paired);
    let ratio = if mean_base_len > 0.0 {
        mean_treat_len / mean_base_len
    } else {
        f64::NAN
    };

    let decision = decide_ship(&overall, &per_category, noise_floor, ratio, kind);

    print_summary(
        &paired,
        &overall,
        &per_category,
        mean_base_len,
        mean_treat_len,
        ratio,
    );
    print_decision(&decision, noise_floor);
    Ok(())
}

fn compute_per_category(
    paired: &[PairedOutcome],
    base: &Scorecard,
    treat: &Scorecard,
) -> HashMap<String, CategoryStats> {
    // First-choice path: a fixture-derived category_map ships with the
    // baseline / treatment scorecard. Prefer treatment's map (latest fixture
    // metadata wins), fall back to baseline's. McNemar is then computed
    // over the JOINED `paired` rows below — the per_category stats from
    // either side's scorecard alone would be degenerate (baseline has no
    // treatment column; run has sentinel baseline_hit=false).
    let category_lookup: HashMap<&str, &str> = if !treat.category_map.is_empty() {
        treat
            .category_map
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect()
    } else if !base.category_map.is_empty() {
        base.category_map
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect()
    } else {
        HashMap::new()
    };

    let mut groups: HashMap<String, Vec<PairedOutcome>> = HashMap::new();
    if !category_lookup.is_empty() {
        for o in paired {
            if let Some(cat) = category_lookup.get(o.case_id.as_str()) {
                groups
                    .entry((*cat).to_string())
                    .or_default()
                    .push(o.clone());
            }
        }
    } else {
        // Legacy fallback: derive categories from `case_id` prefix before
        // ':' (e.g. "single_session:case_3" -> "single_session"). If a
        // case_id has no colon, we skip it for per-category analysis.
        for o in paired {
            if let Some(cat) = o
                .case_id
                .split_once(':')
                .map(|(prefix, _)| prefix.to_string())
            {
                groups.entry(cat).or_default().push(o.clone());
            }
        }
    }

    let mut out = HashMap::new();
    for (cat, outcomes) in groups {
        let (mean_base_len, mean_treat_len) = mean_lengths(&outcomes);
        let hit_base = outcomes.iter().filter(|o| o.baseline_hit).count() as f64;
        let hit_treat = outcomes.iter().filter(|o| o.treatment_hit).count() as f64;
        let n_f = outcomes.len() as f64;
        let stats = CategoryStats {
            n: outcomes.len() as u32,
            baseline_hit_rate: if n_f > 0.0 { hit_base / n_f } else { 0.0 },
            treatment_hit_rate: if n_f > 0.0 { hit_treat / n_f } else { 0.0 },
            avg_baseline_length: mean_base_len,
            avg_treatment_length: mean_treat_len,
            mcnemar: mcnemar(&outcomes),
        };
        out.insert(cat, stats);
    }
    out
}

fn mean_lengths(outcomes: &[PairedOutcome]) -> (f64, f64) {
    if outcomes.is_empty() {
        return (0.0, 0.0);
    }
    let n = outcomes.len() as f64;
    let sum_b: usize = outcomes.iter().map(|o| o.baseline_length).sum();
    let sum_t: usize = outcomes.iter().map(|o| o.treatment_length).sum();
    (sum_b as f64 / n, sum_t as f64 / n)
}

fn print_summary(
    paired: &[PairedOutcome],
    overall: &McNemarResult,
    per_category: &HashMap<String, CategoryStats>,
    mean_base_len: f64,
    mean_treat_len: f64,
    ratio: f64,
) {
    // Header kept generic: `cmd_compare` is shared by the resummerize and
    // concept-summary subcommands, so the banner must not claim either one.
    println!("=== rein-eval: compare ===");
    println!("paired cases : {}", paired.len());
    println!(
        "avg length   : baseline={mean_base_len:.1}  treatment={mean_treat_len:.1}  \
         ratio={ratio:.3}"
    );
    println!();
    println!("overall McNemar:");
    println!(
        "  a={}  b={}  c={}  d={}  n={}",
        overall.a, overall.b, overall.c, overall.d, overall.n
    );
    println!(
        "  chi^2={:.4}  p={:.4}  used_exact={}",
        overall.chi_squared, overall.p_value, overall.used_exact
    );
    println!(
        "  diff_point={:.4}  95% CI=[{:.4}, {:.4}]",
        overall.diff_point, overall.ci_lower, overall.ci_upper
    );
    if !per_category.is_empty() {
        println!();
        println!("per-category McNemar:");
        let mut keys: Vec<&String> = per_category.keys().collect();
        keys.sort();
        for k in keys {
            let s = &per_category[k];
            println!(
                "  {k:30} n={:<4}  hit base/treat={:.3}/{:.3}  diff={:+.4}  p={:.4}",
                s.n,
                s.baseline_hit_rate,
                s.treatment_hit_rate,
                s.mcnemar.diff_point,
                s.mcnemar.p_value
            );
        }
    }
}

fn print_decision(d: &ShipDecision, noise_floor: f64) {
    println!();
    println!("=== ship decision (noise_floor={noise_floor:.3}) ===");
    match d {
        ShipDecision::Ship { reason, .. } => match reason {
            ShipReason::Superior { p_value } => {
                println!("SHIP (Superior): treatment wins with p={p_value:.4}");
            }
            ShipReason::NonInferiorAndShorter {
                avg_length_reduction_pct,
                ci_lower,
                noise_floor: nf,
            } => {
                println!(
                    "SHIP (NonInferiorAndShorter): \
                     avg length reduction {avg_length_reduction_pct:.1}%; \
                     CI lower {ci_lower:.4} > -{nf:.3}"
                );
            }
            ShipReason::NonInferior {
                ci_lower,
                noise_floor: nf,
            } => {
                println!(
                    "SHIP (NonInferior): CI lower {ci_lower:.4} > -{nf:.3} \
                     (length ignored — synthesis regime)"
                );
            }
        },
        ShipDecision::BailOut { reason, .. } => {
            println!("BAIL OUT: {reason}");
        }
    }
}

// --- concept-summary subcommand -------------------------------------------

/// Score each fixture's `definition` against its `evidence_keywords` with
/// no living-summary component. Produces a baseline scorecard comparable
/// against a treatment run via `cmd_compare` (reused from resummerize).
fn cmd_concept_summary_baseline(fixtures: &Path, output: &Path) -> Result<()> {
    let fixtures_list = load_concept_fixtures(fixtures)?;
    if fixtures_list.is_empty() {
        bail!(
            "no concept-summary fixtures found in {}",
            fixtures.display()
        );
    }

    // Hybrid checker keeps baseline + treatment scored under one
    // methodology so McNemar's per-case outcome remains an apples-to-apples
    // comparison. If config has no embedder, the helper degrades to
    // stem-only and the symmetry still holds.
    let config = ReinConfig::load()
        .context("loading rein config for concept-summary baseline (hybrid checker)")?;
    let judge = build_judge(&config, JudgeMode::ConceptSummaryFactCoverage);
    let checker = build_hybrid_checker(&config);
    let mut outcomes = Vec::with_capacity(fixtures_list.len());
    let mut skipped = 0usize;

    for fx in &fixtures_list {
        if fx.evidence_keywords.is_empty() {
            eprintln!(
                "[rein-eval] concept-summary baseline: skipping {} (no evidence_keywords)",
                fx.case_id
            );
            skipped += 1;
            continue;
        }
        // Baseline: score definition alone — no living summary. Mirrors
        // the asymmetry documented on `score_concept_case`:
        // baseline_length = definition.len().
        let hit = if let Some(j) = judge.as_ref() {
            match j.judge_concept_summary(&fx.definition, None, &fx.evidence_keywords) {
                Ok(outcome) => {
                    if !outcome.hit {
                        eprintln!(
                            "[rein-eval] concept-summary baseline {}: judge MISS — {}",
                            fx.case_id, outcome.reason
                        );
                    }
                    outcome.hit
                }
                Err(e) => {
                    eprintln!(
                        "[rein-eval] concept-summary baseline {}: judge error — treating as miss: {}",
                        fx.case_id, e
                    );
                    false
                }
            }
        } else {
            score_concept_case(&fx.definition, None, &fx.evidence_keywords, &checker)
        };
        outcomes.push(PairedOutcome {
            case_id: fx.case_id.clone(),
            baseline_hit: hit,
            treatment_hit: false,
            baseline_length: fx.definition.len(),
            treatment_length: 0,
            treatment_summary: None,
        });
    }

    if outcomes.is_empty() {
        bail!(
            "no fixtures in {} had evidence_keywords — baseline scoring requires keywords",
            fixtures.display()
        );
    }

    let category_map = build_concept_category_map(&fixtures_list);
    let sc = Scorecard {
        fixtures_dir: fixtures.display().to_string(),
        iterations: 1,
        timestamp: Utc::now(),
        outcomes,
        per_category: HashMap::new(),
        category_map,
        hit_checker_version: if judge.is_some() {
            LLM_JUDGE_VERSION
        } else {
            rein::eval::HIT_CHECKER_VERSION
        },
    };
    write_scorecard(output, &sc)?;

    eprintln!(
        "[rein-eval] concept-summary baseline: wrote {} scored cases ({} skipped) to {}",
        sc.outcomes.len(),
        skipped,
        output.display()
    );
    Ok(())
}

/// Drive the living-summary treatment: synthesize Concept+revisions from
/// each fixture, invoke `build_concept_summary_prompt` + the configured
/// LLM (`create_concept_summary_extractor`), then score
/// `definition + " " + living_summary` against `evidence_keywords`.
fn cmd_concept_summary_run(fixtures: &Path, output: &Path, verbose: bool) -> Result<()> {
    let config = ReinConfig::load().context("loading rein config for concept-summary run")?;
    let extractor = create_concept_summary_extractor(&config).ok_or_else(|| {
        anyhow!(
            "no LLM extractor available for concept-summary — configure \
             `[extract].provider` with a valid API key, or point Agent 1's \
             concept-summary backend-selection at a live provider."
        )
    })?;

    let fixtures_list = load_concept_fixtures(fixtures)?;
    if fixtures_list.is_empty() {
        bail!(
            "no concept-summary fixtures found in {}",
            fixtures.display()
        );
    }

    let extractor_tag = match &extractor {
        ExtractorKind::Gemini(_) => "gemini",
        ExtractorKind::Omlx(_) => "omlx",
        #[cfg(feature = "test-support")]
        ExtractorKind::Mock(_) => "mock",
    };
    eprintln!(
        "[rein-eval] concept-summary run: {} fixtures, extractor={}",
        fixtures_list.len(),
        extractor_tag,
    );

    // Hybrid checker mirrors `cmd_concept_summary_baseline`'s configuration
    // so the McNemar comparison stays methodology-symmetric. The same
    // `config` already loaded above feeds the embedder.
    let judge = build_judge(&config, JudgeMode::ConceptSummaryFactCoverage);
    let checker = build_hybrid_checker(&config);
    let mut outcomes: Vec<PairedOutcome> = Vec::with_capacity(fixtures_list.len());
    let mut llm_failed = 0usize;
    let mut empty_output = 0usize;
    let mut skipped = 0usize;

    for fx in &fixtures_list {
        if fx.evidence_keywords.is_empty() {
            eprintln!(
                "[rein-eval] concept-summary run: skipping {} (no evidence_keywords)",
                fx.case_id
            );
            skipped += 1;
            continue;
        }
        if fx.revisions.is_empty() {
            eprintln!(
                "[rein-eval] concept-summary run: skipping {} (no revisions to summarize)",
                fx.case_id
            );
            skipped += 1;
            continue;
        }

        let concept = fx.to_concept();
        let revisions = fx.to_revisions();
        let prompt = build_concept_summary_prompt(&concept, &revisions);

        // Use the production concept-summary LLM bridge so eval and
        // canary exercise the same system prompt and provider shape.
        let llm_result = call_concept_summary_llm_sync(&extractor, &prompt);
        let baseline_len = fx.definition.len();

        match llm_result {
            Ok(text) => {
                let living = strip_code_fences(&text);
                if living.trim().is_empty() {
                    if verbose {
                        eprintln!(
                            "[rein-eval] concept-summary run: {} empty LLM output",
                            fx.case_id
                        );
                    }
                    empty_output += 1;
                    outcomes.push(PairedOutcome {
                        case_id: fx.case_id.clone(),
                        baseline_hit: false,
                        treatment_hit: false,
                        baseline_length: baseline_len,
                        // Empty living_summary → treatment reads like
                        // baseline alone; length reflects that.
                        treatment_length: baseline_len,
                        treatment_summary: None,
                    });
                    continue;
                }
                let hit = if let Some(j) = judge.as_ref() {
                    match j.judge_concept_summary(
                        &fx.definition,
                        Some(&living),
                        &fx.evidence_keywords,
                    ) {
                        Ok(outcome) => {
                            // Always log judge MISS reasons — see Codex R1 P2
                            // for the symmetric-logging rationale (paired
                            // baseline/treatment must use the same policy).
                            if !outcome.hit {
                                eprintln!(
                                    "[rein-eval] concept-summary run {}: judge MISS — {}",
                                    fx.case_id, outcome.reason
                                );
                            }
                            outcome.hit
                        }
                        Err(e) => {
                            eprintln!(
                                "[rein-eval] concept-summary run {}: judge error — treating as miss: {}",
                                fx.case_id, e
                            );
                            false
                        }
                    }
                } else {
                    score_concept_case(
                        &fx.definition,
                        Some(&living),
                        &fx.evidence_keywords,
                        &checker,
                    )
                };
                // Treatment length: definition + " " + living_summary —
                // the exact text that was scored (see
                // `score_concept_case` docs on the asymmetry).
                let treatment_len = baseline_len + 1 + living.len();
                outcomes.push(PairedOutcome {
                    case_id: fx.case_id.clone(),
                    baseline_hit: false,
                    treatment_hit: hit,
                    baseline_length: baseline_len,
                    treatment_length: treatment_len,
                    treatment_summary: Some(living),
                });
            }
            Err(e) => {
                if verbose {
                    let snippet: String = format!("{e}").chars().take(200).collect();
                    eprintln!(
                        "[rein-eval] concept-summary run: {} LLM error: {}",
                        fx.case_id, snippet
                    );
                }
                llm_failed += 1;
                outcomes.push(PairedOutcome {
                    case_id: fx.case_id.clone(),
                    baseline_hit: false,
                    treatment_hit: false,
                    baseline_length: baseline_len,
                    // Match resummerize's LLM-error convention: treatment
                    // falls back to baseline-equivalent length.
                    treatment_length: baseline_len,
                    treatment_summary: None,
                });
            }
        }
    }

    if outcomes.is_empty() {
        bail!(
            "no fixtures in {} produced a scorable concept-summary treatment outcome (skipped={})",
            fixtures.display(),
            skipped,
        );
    }

    let category_map = build_concept_category_map(&fixtures_list);
    let sc = Scorecard {
        fixtures_dir: fixtures.display().to_string(),
        iterations: 1,
        timestamp: Utc::now(),
        outcomes,
        per_category: HashMap::new(),
        category_map,
        hit_checker_version: if judge.is_some() {
            LLM_JUDGE_VERSION
        } else {
            rein::eval::HIT_CHECKER_VERSION
        },
    };
    write_scorecard(output, &sc)?;

    eprintln!(
        "[rein-eval] concept-summary run: wrote {} scored cases ({} skipped, \
         {} llm_failed, {} empty_output) to {}",
        sc.outcomes.len(),
        skipped,
        llm_failed,
        empty_output,
        output.display()
    );
    Ok(())
}

fn build_concept_category_map(fixtures_list: &[ConceptFixture]) -> HashMap<String, String> {
    fixtures_list
        .iter()
        .filter_map(|fx| {
            fx.category
                .as_ref()
                .map(|c| (fx.case_id.clone(), c.clone()))
        })
        .collect()
}

// --- recall-synthesis subcommand (v0.25.1 A3) ------------------------------

/// Score the raw concatenated recall text (per-result `summary` +
/// `evidence_preview`) against each fixture's `evidence_keywords`. This is
/// the "what the operator sees pre-synthesis" floor — Cap B treatment must
/// at least match it (the additive non-inferiority bar).
fn cmd_synthesis_baseline(fixtures: &Path, output: &Path) -> Result<()> {
    let fixtures_list = load_synthesis_fixtures(fixtures)?;
    if fixtures_list.is_empty() {
        bail!("no recall-synthesis fixtures found in {}", fixtures.display());
    }

    // Symmetric scoring with `run_synthesis_treatment_with_extractor` —
    // both must use the same checker for McNemar to be honest. Loading
    // config here is cheap and lets baseline opt into the embedding-based
    // semantic fallback when one is configured.
    let config = ReinConfig::load()
        .context("loading rein config for synthesis baseline (hybrid checker)")?;
    let judge = build_judge(&config, JudgeMode::SynthesisSourceCoverage);
    let checker = build_hybrid_checker(&config);
    let mut outcomes = Vec::with_capacity(fixtures_list.len());
    let mut skipped = 0usize;

    for fx in &fixtures_list {
        if fx.evidence_keywords.is_empty() {
            eprintln!(
                "[rein-eval] synthesis baseline: skipping {} (no evidence_keywords)",
                fx.case_id
            );
            skipped += 1;
            continue;
        }
        if fx.recall_results.is_empty() {
            eprintln!(
                "[rein-eval] synthesis baseline: skipping {} (no recall_results)",
                fx.case_id
            );
            skipped += 1;
            continue;
        }

        let baseline_text = fx.baseline_text();
        // Reuse `score_concept_case` with `living_summary = None` — the
        // helper's "definition + (optional) summary" abstraction maps
        // cleanly onto baseline (raw text) vs treatment (LLM prose) here.
        let hit = if let Some(j) = judge.as_ref() {
            let summaries_owned = fx.judge_source_summaries();
            let summaries: Vec<&str> = summaries_owned.iter().map(|s| s.as_str()).collect();
            match j.judge_synthesis(&fx.query, &summaries, &baseline_text) {
                Ok(outcome) => {
                    if !outcome.hit {
                        eprintln!(
                            "[rein-eval] synthesis baseline {}: judge MISS — {}",
                            fx.case_id, outcome.reason
                        );
                    }
                    outcome.hit
                }
                Err(e) => {
                    eprintln!(
                        "[rein-eval] synthesis baseline {}: judge error — treating as miss: {}",
                        fx.case_id, e
                    );
                    false
                }
            }
        } else {
            score_concept_case(&baseline_text, None, &fx.evidence_keywords, &checker)
        };
        outcomes.push(PairedOutcome {
            case_id: fx.case_id.clone(),
            baseline_hit: hit,
            treatment_hit: false,
            baseline_length: baseline_text.len(),
            treatment_length: 0,
            treatment_summary: None,
        });
    }

    if outcomes.is_empty() {
        bail!(
            "no fixtures in {} had both `evidence_keywords` and `recall_results` — \
             baseline scoring requires both",
            fixtures.display()
        );
    }

    let category_map = build_synthesis_category_map(&fixtures_list);
    let sc = Scorecard {
        fixtures_dir: fixtures.display().to_string(),
        iterations: 1,
        timestamp: Utc::now(),
        outcomes,
        per_category: HashMap::new(),
        category_map,
        hit_checker_version: if judge.is_some() {
            LLM_JUDGE_VERSION
        } else {
            rein::eval::HIT_CHECKER_VERSION
        },
    };
    write_scorecard(output, &sc)?;

    eprintln!(
        "[rein-eval] synthesis baseline: wrote {} scored cases ({} skipped) to {}",
        sc.outcomes.len(),
        skipped,
        output.display()
    );
    Ok(())
}

/// Drive the recall-synthesis treatment: build the production prompt via
/// `build_synthesis_prompt`, call the same LLM bridge production uses
/// (`call_synthesis_llm_sync` + `create_concept_summary_extractor`), and
/// score the prose output against `evidence_keywords`.
fn cmd_synthesis_run(fixtures: &Path, output: &Path, verbose: bool) -> Result<()> {
    let config = ReinConfig::load().context("loading rein config for synthesis run")?;
    // Use the production extractor-selection path so eval honors
    // `[ars].llm_backend` (inherit / google / omlx) the same way
    // `run_recall_synthesis` does. Without this, an operator who
    // configured a different ARS backend would see `compare` verdicts
    // that don't reflect production behavior.
    let extractor = create_concept_summary_extractor(&config).ok_or_else(|| {
        anyhow!(
            "no LLM extractor available for recall synthesis — configure \
             `[extract].provider` (or `[ars].llm_backend = \"google\"`) with \
             a valid API key (GEMINI_API_KEY) or `[ars].llm_backend = \"omlx\"` \
             with a configured `[extract.omlx]` block."
        )
    })?;

    let fixtures_list = load_synthesis_fixtures(fixtures)?;
    if fixtures_list.is_empty() {
        bail!("no recall-synthesis fixtures found in {}", fixtures.display());
    }

    let extractor_tag = match &extractor {
        ExtractorKind::Gemini(_) => "gemini",
        ExtractorKind::Omlx(_) => "omlx",
        #[cfg(feature = "test-support")]
        ExtractorKind::Mock(_) => "mock",
    };
    eprintln!(
        "[rein-eval] synthesis run: {} fixtures, extractor={}",
        fixtures_list.len(),
        extractor_tag,
    );

    run_synthesis_treatment_with_extractor(&fixtures_list, &extractor, &config, output, fixtures, verbose)
}

/// Treatment loop extracted so unit tests can drive it with a `MockExtractor`
/// without hitting a live provider. Production callers go through
/// `cmd_synthesis_run`, which loads config + builds the extractor once.
fn run_synthesis_treatment_with_extractor(
    fixtures_list: &[SynthesisFixture],
    extractor: &ExtractorKind,
    config: &ReinConfig,
    output: &Path,
    fixtures_dir_for_meta: &Path,
    verbose: bool,
) -> Result<()> {
    // Hybrid checker — must match `cmd_synthesis_baseline`'s configuration
    // so the McNemar table reflects a consistent scoring methodology.
    let judge = build_judge(config, JudgeMode::SynthesisSourceCoverage);
    let checker = build_hybrid_checker(config);
    let mut outcomes: Vec<PairedOutcome> = Vec::with_capacity(fixtures_list.len());
    let mut llm_failed = 0usize;
    let mut empty_output = 0usize;
    let mut skipped = 0usize;

    // Mirror the production `max_input_chars` resolution so the eval
    // truncates the prompt the same way `run_recall_synthesis` would.
    // Drift here would silently change which evidence the LLM sees and
    // invalidate the McNemar comparison.
    let max_chars = rein::extract::llm::resolve_max_input_for_kind(config, extractor);

    for fx in fixtures_list {
        if fx.evidence_keywords.is_empty() {
            eprintln!(
                "[rein-eval] synthesis run: skipping {} (no evidence_keywords)",
                fx.case_id
            );
            skipped += 1;
            continue;
        }
        if fx.recall_results.is_empty() {
            eprintln!(
                "[rein-eval] synthesis run: skipping {} (no recall_results)",
                fx.case_id
            );
            skipped += 1;
            continue;
        }

        let results = fx.to_recall_results();
        // Codex R2 G4+G5: track `included_count` (memories the LLM
        // actually sees after truncation) and strip `[#k]` markers from
        // the LLM output the same way production does, so the eval's
        // `treatment_summary` / `treatment_length` / keyword scoring
        // reflect what users see — not the raw LLM response.
        let (prompt, included_count) =
            build_synthesis_prompt_with_count(&results, &fx.query, max_chars);
        let baseline_len = fx.baseline_text().len();

        match call_synthesis_llm_sync(extractor, &prompt) {
            Ok(raw) => {
                let raw_text = strip_code_fences(&raw).trim().to_string();
                // Mirror production: strip markers, keep clean prose for
                // scoring + recording. `included_count` matches the prod
                // contract; out-of-range markers are silently dropped.
                let (synthesis, _citations) =
                    extract_citations(&raw_text, included_count);
                if synthesis.is_empty() {
                    if verbose {
                        eprintln!(
                            "[rein-eval] synthesis run: {} empty LLM output",
                            fx.case_id
                        );
                    }
                    empty_output += 1;
                    outcomes.push(PairedOutcome {
                        case_id: fx.case_id.clone(),
                        baseline_hit: false,
                        treatment_hit: false,
                        baseline_length: baseline_len,
                        // Empty synthesis → operator effectively sees only
                        // the raw recall list; mirror that by reporting
                        // the baseline length.
                        treatment_length: baseline_len,
                        treatment_summary: None,
                    });
                    continue;
                }
                let hit = if let Some(j) = judge.as_ref() {
                    let summaries_owned = fx.judge_source_summaries();
                    let summaries: Vec<&str> = summaries_owned.iter().map(|s| s.as_str()).collect();
                    match j.judge_synthesis(&fx.query, &summaries, &synthesis) {
                        Ok(outcome) => {
                            // Always log judge MISS reasons — they are
                            // ship-decision-relevant and must mirror
                            // `cmd_synthesis_baseline`'s policy. Codex R1
                            // P2: dropping the verbose gate restores
                            // symmetric logging across paired runs.
                            if !outcome.hit {
                                eprintln!(
                                    "[rein-eval] synthesis run {}: judge MISS — {}",
                                    fx.case_id, outcome.reason
                                );
                            }
                            outcome.hit
                        }
                        Err(e) => {
                            eprintln!(
                                "[rein-eval] synthesis run {}: judge error — treating as miss: {}",
                                fx.case_id, e
                            );
                            false
                        }
                    }
                } else {
                    score_concept_case(&synthesis, None, &fx.evidence_keywords, &checker)
                };
                outcomes.push(PairedOutcome {
                    case_id: fx.case_id.clone(),
                    baseline_hit: false,
                    treatment_hit: hit,
                    baseline_length: baseline_len,
                    treatment_length: synthesis.len(),
                    treatment_summary: Some(synthesis),
                });
            }
            Err(e) => {
                if verbose {
                    let snippet: String = format!("{e}").chars().take(200).collect();
                    eprintln!(
                        "[rein-eval] synthesis run: {} LLM error: {}",
                        fx.case_id, snippet
                    );
                }
                llm_failed += 1;
                outcomes.push(PairedOutcome {
                    case_id: fx.case_id.clone(),
                    baseline_hit: false,
                    treatment_hit: false,
                    baseline_length: baseline_len,
                    // Match concept-summary's LLM-error convention: treatment
                    // falls back to baseline-equivalent length.
                    treatment_length: baseline_len,
                    treatment_summary: None,
                });
            }
        }
    }

    if outcomes.is_empty() {
        bail!(
            "no fixtures in {} produced a scorable synthesis treatment outcome (skipped={})",
            fixtures_dir_for_meta.display(),
            skipped,
        );
    }

    let category_map = build_synthesis_category_map(fixtures_list);
    let sc = Scorecard {
        fixtures_dir: fixtures_dir_for_meta.display().to_string(),
        iterations: 1,
        timestamp: Utc::now(),
        outcomes,
        per_category: HashMap::new(),
        category_map,
        hit_checker_version: if judge.is_some() {
            LLM_JUDGE_VERSION
        } else {
            rein::eval::HIT_CHECKER_VERSION
        },
    };
    write_scorecard(output, &sc)?;

    eprintln!(
        "[rein-eval] synthesis run: wrote {} scored cases ({} skipped, \
         {} llm_failed, {} empty_output) to {}",
        sc.outcomes.len(),
        skipped,
        llm_failed,
        empty_output,
        output.display()
    );
    Ok(())
}

fn build_synthesis_category_map(fixtures_list: &[SynthesisFixture]) -> HashMap<String, String> {
    fixtures_list
        .iter()
        .filter_map(|fx| {
            fx.category
                .as_ref()
                .map(|c| (fx.case_id.clone(), c.clone()))
        })
        .collect()
}

fn load_synthesis_fixtures(dir: &Path) -> Result<Vec<SynthesisFixture>> {
    if !dir.exists() {
        bail!("fixtures directory does not exist: {}", dir.display());
    }
    if !dir.is_dir() {
        bail!("fixtures path is not a directory: {}", dir.display());
    }
    let mut out = Vec::new();
    for entry in
        fs::read_dir(dir).with_context(|| format!("reading fixtures dir {}", dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let bytes =
            fs::read(&path).with_context(|| format!("reading fixture {}", path.display()))?;
        let cases: Vec<SynthesisFixture> = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing fixture {}", path.display()))?;
        out.extend(cases);
    }
    if out.is_empty() {
        return Err(anyhow!(
            "no .json recall-synthesis fixtures found in {}",
            dir.display()
        ));
    }
    out.sort_by(|a, b| a.case_id.cmp(&b.case_id));
    Ok(out)
}

fn load_concept_fixtures(dir: &Path) -> Result<Vec<ConceptFixture>> {
    if !dir.exists() {
        bail!("fixtures directory does not exist: {}", dir.display());
    }
    if !dir.is_dir() {
        bail!("fixtures path is not a directory: {}", dir.display());
    }
    let mut out = Vec::new();
    for entry in
        fs::read_dir(dir).with_context(|| format!("reading fixtures dir {}", dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let bytes =
            fs::read(&path).with_context(|| format!("reading fixture {}", path.display()))?;
        let cases: Vec<ConceptFixture> = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing fixture {}", path.display()))?;
        out.extend(cases);
    }
    if out.is_empty() {
        return Err(anyhow!(
            "no .json concept-summary fixtures found in {}",
            dir.display()
        ));
    }
    out.sort_by(|a, b| a.case_id.cmp(&b.case_id));
    Ok(out)
}

// --- I/O helpers -----------------------------------------------------------

fn load_fixtures(dir: &Path) -> Result<Vec<Fixture>> {
    if !dir.exists() {
        bail!("fixtures directory does not exist: {}", dir.display());
    }
    if !dir.is_dir() {
        bail!("fixtures path is not a directory: {}", dir.display());
    }
    let mut out = Vec::new();
    for entry in
        fs::read_dir(dir).with_context(|| format!("reading fixtures dir {}", dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let bytes =
            fs::read(&path).with_context(|| format!("reading fixture {}", path.display()))?;
        // Each seed fixture file is a JSON array of cases (per Agent B's
        // schema in `tests/fixtures/resummerize/*.json`). The previous
        // single-object parse failed on every shipped fixture; Codex
        // audit M9.
        let cases: Vec<Fixture> = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing fixture {}", path.display()))?;
        out.extend(cases);
    }
    if out.is_empty() {
        return Err(anyhow!("no .json fixtures found in {}", dir.display()));
    }
    // Sort for deterministic iteration order.
    out.sort_by(|a, b| a.case_id.cmp(&b.case_id));
    Ok(out)
}

fn load_scorecard(path: &Path) -> Result<Scorecard> {
    let bytes = fs::read(path).with_context(|| format!("reading scorecard {}", path.display()))?;
    let sc: Scorecard = serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing scorecard {}", path.display()))?;
    Ok(sc)
}

fn write_scorecard(path: &Path, sc: &Scorecard) -> Result<()> {
    let json = serde_json::to_vec_pretty(sc).context("serializing scorecard")?;
    fs::write(path, json).with_context(|| format!("writing scorecard {}", path.display()))?;
    Ok(())
}

#[cfg(all(test, feature = "test-support"))]
mod tests {
    use super::*;
    use rein::extract::llm::{ExtractorKind, MockExtractor};

    /// A tiny ASCII-only fixture whose evidence shares enough overlapping
    /// tokens that any reasonable summary will pass the keyword-overlap
    /// hit check, and whose target_bytes is large enough that almost any
    /// LLM output passes `length_bounded`.
    fn mini_fixture(case_id: &str, category: &str) -> Fixture {
        Fixture {
            case_id: case_id.to_string(),
            category: Some(category.to_string()),
            current_canonical: Some(
                "user prefers concise output. user prefers concise summaries. user wants \
                 brief replies."
                    .to_string(),
            ),
            evidence: vec![
                FixtureEvidenceEntry {
                    content: "user prefers concise output and brief explanations and concise \
                              summaries and brief replies"
                        .to_string(),
                    merged_at: Some("2026-04-01T10:00:00Z".to_string()),
                },
                FixtureEvidenceEntry {
                    content: "user wants concise replies and brief output and concise output \
                              and short replies"
                        .to_string(),
                    merged_at: Some("2026-04-02T10:00:00Z".to_string()),
                },
            ],
            target_bytes: Some(8000),
            canonical: None,
            context: None,
        }
    }

    fn tmp_path(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "rein_eval_test_{}_{}.json",
            name,
            std::process::id()
        ));
        p
    }

    #[test]
    fn cmd_run_with_mock_contract_pass_records_treatment_hit() {
        // Mock returns a response that contains every salient evidence
        // keyword AND fits the contract: short, no new facts, no temporal
        // anchors to drop, no CJK, no code blocks.
        // Vocab-restricted to maximize trigram overlap with evidence —
        // the `no_new_facts` invariant requires ≥90% of output trigrams to
        // appear in evidence + current_canonical. No punctuation
        // introduced (periods/commas would create unique trigrams).
        let mock_output = "user prefers concise output user wants brief concise replies and \
                           short summaries";
        let extractor = ExtractorKind::Mock(MockExtractor::with_responses(vec![
            Ok(mock_output.to_string()),
            Ok(mock_output.to_string()),
        ]));
        let fixtures = vec![
            mini_fixture("cjk_001", "cjk"),
            mini_fixture("cjk_002", "cjk"),
        ];
        let out_path = tmp_path("contract_pass");
        let res = run_treatment_with_extractor(
            &fixtures,
            &extractor,
            1,
            &out_path,
            Path::new("dummy"),
            /* verbose */ false,
        );
        assert!(res.is_ok(), "run_treatment_with_extractor failed: {res:?}");

        let sc: Scorecard = serde_json::from_slice(&fs::read(&out_path).unwrap()).unwrap();
        assert_eq!(sc.outcomes.len(), 2);
        for o in &sc.outcomes {
            assert!(
                o.treatment_hit,
                "expected treatment hit for case {}; output={mock_output}",
                o.case_id
            );
            assert_eq!(o.treatment_length, mock_output.len());
        }
        // category_map populated from Fixture.category.
        assert_eq!(
            sc.category_map.get("cjk_001").map(String::as_str),
            Some("cjk")
        );
        // per_category empty — `compare` derives it from joined data.
        assert!(sc.per_category.is_empty());
        let _ = fs::remove_file(&out_path);
    }

    #[test]
    fn cmd_run_with_mock_contract_violation_marks_treatment_miss() {
        // Mock returns a response that's far too long — guaranteed to
        // violate `length_bounded`. Uses target_bytes=200 so a 3KB output
        // blows past the +10% tolerance.
        let mut fx = mini_fixture("contradictions_001", "contradictions");
        fx.target_bytes = Some(200);
        let oversize_output = "x".repeat(3000) + " concise output brief replies user wants";
        let extractor =
            ExtractorKind::Mock(MockExtractor::with_responses(vec![Ok(oversize_output)]));
        let out_path = tmp_path("contract_violation");
        let res = run_treatment_with_extractor(
            &[fx],
            &extractor,
            1,
            &out_path,
            Path::new("dummy"),
            /* verbose */ false,
        );
        assert!(res.is_ok(), "run_treatment_with_extractor failed: {res:?}");

        let sc: Scorecard = serde_json::from_slice(&fs::read(&out_path).unwrap()).unwrap();
        assert_eq!(sc.outcomes.len(), 1);
        assert!(
            !sc.outcomes[0].treatment_hit,
            "contract-violating output must not score as a treatment hit"
        );
        // Contract fail → production keeps keep-tail → treatment_length
        // reverts to baseline_length in the scorecard.
        assert_eq!(
            sc.outcomes[0].treatment_length, sc.outcomes[0].baseline_length,
            "contract-failed cases must report keep-tail length, not the rejected LLM output length"
        );
        let _ = fs::remove_file(&out_path);
    }

    #[test]
    fn cmd_run_with_mock_llm_error_keeps_keep_tail_length() {
        // Production falls back to keep-tail on LLM error; the eval
        // mirrors that by reporting treatment_length == baseline_length
        // so `avg_length_ratio` doesn't falsely credit the failure with
        // a shorter output. Hit rate stays false (baseline scorecard
        // fills the baseline_hit side at compare time).
        let extractor =
            ExtractorKind::Mock(MockExtractor::with_persistent_error("simulated outage"));
        let fx = mini_fixture("code_blocks_001", "code_blocks");
        let canonical_len = fx.effective_canonical().unwrap().len();
        let out_path = tmp_path("llm_error");
        let res = run_treatment_with_extractor(
            &[fx],
            &extractor,
            1,
            &out_path,
            Path::new("dummy"),
            /* verbose */ false,
        );
        assert!(res.is_ok(), "run_treatment_with_extractor failed: {res:?}");

        let sc: Scorecard = serde_json::from_slice(&fs::read(&out_path).unwrap()).unwrap();
        assert_eq!(sc.outcomes.len(), 1);
        assert!(!sc.outcomes[0].treatment_hit);
        assert_eq!(sc.outcomes[0].treatment_length, canonical_len);
        assert_eq!(sc.outcomes[0].baseline_length, canonical_len);
        let _ = fs::remove_file(&out_path);
    }

    #[test]
    fn cmd_run_skips_fixture_missing_target_bytes() {
        let mut bad = mini_fixture("temporal_anchors_001", "temporal_anchors");
        bad.target_bytes = None;
        let good = mini_fixture("temporal_anchors_002", "temporal_anchors");
        let mock_output =
            "user prefers concise output user wants brief concise replies and short summaries";
        let extractor = ExtractorKind::Mock(MockExtractor::with_responses(vec![Ok(
            mock_output.to_string()
        )]));
        let out_path = tmp_path("skip_no_target");
        let res = run_treatment_with_extractor(
            &[bad, good],
            &extractor,
            1,
            &out_path,
            Path::new("dummy"),
            /* verbose */ false,
        );
        assert!(res.is_ok(), "run_treatment_with_extractor failed: {res:?}");

        let sc: Scorecard = serde_json::from_slice(&fs::read(&out_path).unwrap()).unwrap();
        // Only the good fixture is recorded; the missing-target one is skipped.
        assert_eq!(sc.outcomes.len(), 1);
        assert_eq!(sc.outcomes[0].case_id, "temporal_anchors_002");
        let _ = fs::remove_file(&out_path);
    }

    #[test]
    fn category_map_and_compare_per_category_use_joined_data() {
        // Verify compute_per_category prefers category_map + computes
        // McNemar over the JOINED paired data, not over either side's
        // standalone scorecard.
        let base = Scorecard {
            fixtures_dir: "test".into(),
            iterations: 1,
            timestamp: Utc::now(),
            outcomes: vec![
                PairedOutcome {
                    case_id: "cjk_001".into(),
                    baseline_hit: true,
                    treatment_hit: false,
                    baseline_length: 100,
                    treatment_length: 0,
                    treatment_summary: None,
                },
                PairedOutcome {
                    case_id: "cjk_002".into(),
                    baseline_hit: false,
                    treatment_hit: false,
                    baseline_length: 200,
                    treatment_length: 0,
                    treatment_summary: None,
                },
            ],
            per_category: HashMap::new(),
            category_map: [
                ("cjk_001".to_string(), "cjk".to_string()),
                ("cjk_002".to_string(), "cjk".to_string()),
            ]
            .into_iter()
            .collect(),
            hit_checker_version: rein::eval::HIT_CHECKER_VERSION,
        };
        let treat = Scorecard {
            fixtures_dir: "test".into(),
            iterations: 1,
            timestamp: Utc::now(),
            outcomes: vec![
                PairedOutcome {
                    case_id: "cjk_001".into(),
                    baseline_hit: false,
                    treatment_hit: true,
                    baseline_length: 100,
                    treatment_length: 50,
                    treatment_summary: None,
                },
                PairedOutcome {
                    case_id: "cjk_002".into(),
                    baseline_hit: false,
                    treatment_hit: true,
                    baseline_length: 200,
                    treatment_length: 80,
                    treatment_summary: None,
                },
            ],
            per_category: HashMap::new(),
            category_map: [
                ("cjk_001".to_string(), "cjk".to_string()),
                ("cjk_002".to_string(), "cjk".to_string()),
            ]
            .into_iter()
            .collect(),
            hit_checker_version: rein::eval::HIT_CHECKER_VERSION,
        };
        // Reproduce cmd_compare's join logic.
        let paired: Vec<PairedOutcome> = base
            .outcomes
            .iter()
            .filter_map(|b| {
                treat
                    .outcomes
                    .iter()
                    .find(|t| t.case_id == b.case_id)
                    .map(|t| PairedOutcome {
                        case_id: b.case_id.clone(),
                        baseline_hit: b.baseline_hit,
                        treatment_hit: t.treatment_hit,
                        baseline_length: b.baseline_length,
                        treatment_length: t.treatment_length,
                        treatment_summary: None,
                    })
            })
            .collect();

        let pc = compute_per_category(&paired, &base, &treat);
        assert_eq!(pc.len(), 1, "expected one category 'cjk'");
        let cjk = pc.get("cjk").unwrap();
        assert_eq!(cjk.n, 2);
        // baseline_hit_rate = 1/2 (cjk_001), treatment_hit_rate = 2/2.
        assert!((cjk.baseline_hit_rate - 0.5).abs() < 1e-9);
        assert!((cjk.treatment_hit_rate - 1.0).abs() < 1e-9);
        // McNemar 2x2: a=1 (both hit cjk_001? no, baseline_hit=true,
        // treatment_hit=true), b=0, c=1 (cjk_002), d=0.
        assert_eq!(cjk.mcnemar.a, 1);
        assert_eq!(cjk.mcnemar.b, 0);
        assert_eq!(cjk.mcnemar.c, 1);
        assert_eq!(cjk.mcnemar.d, 0);
    }

    #[test]
    fn concept_summary_eval_uses_production_system_prompt() {
        let (mock, probe) = MockExtractor::with_fixed_response_and_probe("summary");
        let extractor = ExtractorKind::Mock(mock);

        let output = call_concept_summary_llm_sync(&extractor, "concept prompt").unwrap();

        assert_eq!(output, "summary");
        let system = probe
            .last_system_prompt()
            .expect("mock should record the system prompt");
        assert!(
            system.contains("concept-state synthesizer"),
            "eval must use the production concept-summary system prompt; got: {system}"
        );
        assert!(
            system.contains("preserve exact identifiers"),
            "eval must preserve the identifier-preservation instructions; got: {system}"
        );
    }

    #[test]
    fn resummerize_phase2_fixtures_parse() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/resummerize-phase2");
        let fixtures = load_fixtures(&dir).expect("phase2 resummerize fixtures should parse");

        assert_eq!(fixtures.len(), 30);
        assert!(
            fixtures.iter().all(|f| f.target_bytes.is_some()),
            "phase2 fixtures must set target_bytes so treatment scoring is deterministic"
        );
        assert!(
            fixtures.iter().all(|f| !f.evidence.is_empty()),
            "phase2 fixtures must include evidence rows"
        );
    }

    #[test]
    fn concept_summary_fixtures_parse() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/concept_summary");
        let fixtures = load_concept_fixtures(&dir).expect("concept-summary fixtures should parse");

        assert_eq!(fixtures.len(), 6);
        assert!(
            fixtures.iter().all(|f| !f.revisions.is_empty()),
            "concept-summary fixtures must include revision history"
        );
        assert!(
            fixtures.iter().all(|f| !f.evidence_keywords.is_empty()),
            "concept-summary fixtures must include evidence keywords"
        );
    }

    // --- v0.25.1 A3 recall-synthesis harness tests --------------------------

    /// Verify all shipped synthesis fixture files load + each fixture
    /// has the minimum schema needed by the harness.
    #[test]
    fn recall_synthesis_fixtures_parse() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/recall_synthesis");
        let fixtures =
            load_synthesis_fixtures(&dir).expect("recall-synthesis fixtures should parse");

        // 6 fixture files × {2 (sq/mt/amb) or 6 (lt/cc/cf)} cases = 24 cases
        // total at v0.25.2 corpus expansion (3 original files + 3 hard-case
        // files: longtail / cross_cluster / conflicting).
        assert_eq!(fixtures.len(), 24);
        assert!(
            fixtures.iter().all(|f| !f.recall_results.is_empty()),
            "recall-synthesis fixtures must include at least one recall result"
        );
        assert!(
            fixtures.iter().all(|f| !f.evidence_keywords.is_empty()),
            "recall-synthesis fixtures must include evidence keywords"
        );
        // Strict-majority threshold for `score_concept_case` is
        // `hits * 2 > n`. With 5 keywords this means ≥3 hits, matching the
        // task spec. Anything other than 5 changes the bar; lock it down.
        assert!(
            fixtures.iter().all(|f| f.evidence_keywords.len() == 5),
            "recall-synthesis fixtures must have exactly 5 evidence_keywords \
             so the strict-majority threshold equals the spec's 'at least 3 of 5'"
        );
    }

    /// Baseline scoring is deterministic and produces a paired-outcome row
    /// per fixture with `baseline_length` matching the concatenated text.
    #[test]
    fn synthesis_baseline_is_deterministic() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/recall_synthesis");
        let out_path = tmp_path("synth_baseline_det");

        cmd_synthesis_baseline(&dir, &out_path)
            .expect("baseline command should succeed against the shipped fixtures");
        let sc1: Scorecard = serde_json::from_slice(&fs::read(&out_path).unwrap()).unwrap();

        // Run a second time and compare hit columns + lengths — output must
        // be deterministic (same fixtures + same hit checker version).
        cmd_synthesis_baseline(&dir, &out_path)
            .expect("baseline command should succeed on the second run");
        let sc2: Scorecard = serde_json::from_slice(&fs::read(&out_path).unwrap()).unwrap();

        assert_eq!(sc1.outcomes.len(), sc2.outcomes.len());
        assert_eq!(sc1.outcomes.len(), 24);
        for (a, b) in sc1.outcomes.iter().zip(sc2.outcomes.iter()) {
            assert_eq!(a.case_id, b.case_id);
            assert_eq!(a.baseline_hit, b.baseline_hit);
            assert_eq!(a.baseline_length, b.baseline_length);
        }
        assert_eq!(sc1.hit_checker_version, rein::eval::HIT_CHECKER_VERSION);
        let _ = fs::remove_file(&out_path);
    }

    /// Run scoring with a `MockExtractor` produces the expected scorecard:
    /// the mock returns prose containing every evidence keyword, so all
    /// cases score as treatment hits. `treatment_length` reflects the LLM
    /// output length, not the baseline.
    #[test]
    fn synthesis_run_with_mock_records_treatment_hits() {
        // Mock prose includes ≥3-of-5 evidence_keywords for every fixture in
        // the v0.25.2 expanded corpus (24 cases across 6 files) so the
        // strict-majority threshold is trivially satisfied for each. The
        // prose layers in synthesis-emergent meta-language (five, seven,
        // four, consensus, agreement, defense, semaphore, etc.) so the new
        // baseline-miss fixtures still treatment-hit under the mock without
        // per-fixture scripted responses.
        let mock_response = "The system uses proc-macro generated OpsRuntime methods registered \
             via inventory; CLI/MCP/REST adapters share dispatch with AuthPolicy middleware. \
             HDBSCAN builds a dendrogram from mutual reachability and extracts EOMBST clusters \
             which feed M3 Kaplan-Meier survival decay and AdaptiveState event-sourced tiering. \
             v0.23 introduced resummerize gated by a 7-invariant Compression contract; v0.24 \
             added concept living-summary; v0.25 added recall-time synthesis as Capability B. \
             STM/LTM use Ebbinghaus decay with provenance-preserving merge over a tiering \
             quantile estimator. Search is a Tantivy + HNSW + KG waterfall with rule-based \
             routing and parallel query expansion via Gemini fusion. \
             Five distinct consumers maintain different watermarks spanning the event log; \
             the list of consumer offsets is enumerated incrementally across four MCP-tool \
             additions in the v0.23-v0.25 lineage. Seven invariants together act as a complete \
             guardrail validation gate. MockEmbedder and MockExtractor sit behind the \
             test-support feature flag with a FIFO queue. The waterfall returns canonical \
             previews via Tantivy BM25, HNSW vector channel, KG BFS, sqlite-vec storage, and \
             a needs_vec_dedup audit marker. Eventual convergence is reached asynchronously; \
             loose consistency is tolerated. A layered defense uses multiple interlocking \
             complementary fuses. The semaphore bounds extract, expansion, rerank, and dedup \
             LLM calls; the alpha learner per cluster picks a tier per cluster ID. The \
             pattern is uniformly applied throughout the codebase with the same discipline. \
             The GUI surfaces 21 REST endpoints alongside MCP and CLI through OpsRuntime. \
             There is broad agreement consistent across versions that the proposal was \
             uniformly rejected; the stance held. Older designs were superseded and shifted \
             out, replaced through evolution and retired. A standing disagreement evolved \
             as the spec narrowed; some matters remain unresolved with tension. The \
             proc-macro plus inventory dispatch pattern is the chosen static-or-not \
             decision. HDBSCAN cluster stability is good despite reassignment, with \
             smoothing in M2 and M3. Five evidence_keywords with majority threshold \
             matches the spec.";

        // 24 fixtures × 1 mock response each → queue 24 copies.
        let extractor = ExtractorKind::Mock(MockExtractor::with_responses(
            (0..24).map(|_| Ok(mock_response.to_string())).collect(),
        ));
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/recall_synthesis");
        let fixtures =
            load_synthesis_fixtures(&dir).expect("recall-synthesis fixtures should parse");
        // Use a config with `recall_synthesis_enabled = true` so
        // `resolve_max_input_for_kind` resolves predictably; mock has no
        // real input cap so the value mostly matters for the prompt-shape
        // path, not the call.
        let config = ReinConfig::default();

        let out_path = tmp_path("synth_run_mock");
        run_synthesis_treatment_with_extractor(
            &fixtures,
            &extractor,
            &config,
            &out_path,
            &dir,
            /* verbose */ false,
        )
        .expect("treatment loop should succeed under mock");

        let sc: Scorecard = serde_json::from_slice(&fs::read(&out_path).unwrap()).unwrap();
        assert_eq!(sc.outcomes.len(), 24);
        for o in &sc.outcomes {
            assert!(
                o.treatment_hit,
                "expected treatment hit for {} given keyword-rich mock response",
                o.case_id
            );
            assert_eq!(
                o.treatment_length,
                mock_response.len(),
                "treatment_length must reflect the LLM output, not the baseline"
            );
            assert_eq!(o.treatment_summary.as_deref(), Some(mock_response));
        }
        assert_eq!(sc.category_map.len(), 24, "category_map populated per fixture");
        let _ = fs::remove_file(&out_path);
    }

    /// LLM error must NOT score as a treatment hit AND must report
    /// `treatment_length == baseline_length` so `avg_length_ratio` doesn't
    /// falsely credit the failure.
    #[test]
    fn synthesis_run_llm_error_does_not_score_as_hit() {
        let extractor =
            ExtractorKind::Mock(MockExtractor::with_persistent_error("simulated outage"));
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/recall_synthesis");
        let fixtures =
            load_synthesis_fixtures(&dir).expect("recall-synthesis fixtures should parse");
        let config = ReinConfig::default();

        let out_path = tmp_path("synth_run_err");
        run_synthesis_treatment_with_extractor(
            &fixtures,
            &extractor,
            &config,
            &out_path,
            &dir,
            /* verbose */ false,
        )
        .expect("treatment loop should still write a scorecard on LLM error");

        let sc: Scorecard = serde_json::from_slice(&fs::read(&out_path).unwrap()).unwrap();
        for o in &sc.outcomes {
            assert!(
                !o.treatment_hit,
                "LLM error must not produce a treatment hit for {}",
                o.case_id
            );
            assert_eq!(
                o.treatment_length, o.baseline_length,
                "LLM error → treatment_length collapses to baseline_length"
            );
            assert!(o.treatment_summary.is_none());
        }
        let _ = fs::remove_file(&out_path);
    }

    /// `cmd_compare` joins two synthesized scorecards and reaches a
    /// deterministic verdict under `DecideShipKind::Synthesis`. We
    /// construct two scorecards in-memory rather than going through
    /// `cmd_synthesis_run` to avoid coupling this test to a live LLM.
    #[test]
    fn synthesis_compare_with_synthetic_scorecards_ships_when_treatment_wins() {
        // Baseline: all misses; Treatment: all hits → strong superiority.
        let make_outcome =
            |case_id: &str, baseline_hit: bool, treatment_hit: bool| PairedOutcome {
                case_id: case_id.into(),
                baseline_hit,
                treatment_hit,
                baseline_length: 1000,
                treatment_length: 1500,
                treatment_summary: None,
            };

        let baseline_outcomes = (0..30)
            .map(|i| make_outcome(&format!("case_{i:03}"), false, false))
            .collect::<Vec<_>>();
        let treatment_outcomes = (0..30)
            .map(|i| make_outcome(&format!("case_{i:03}"), false, true))
            .collect::<Vec<_>>();

        let base = Scorecard {
            fixtures_dir: "test".into(),
            iterations: 1,
            timestamp: Utc::now(),
            outcomes: baseline_outcomes,
            per_category: HashMap::new(),
            category_map: HashMap::new(),
            hit_checker_version: rein::eval::HIT_CHECKER_VERSION,
        };
        let treat = Scorecard {
            fixtures_dir: "test".into(),
            iterations: 1,
            timestamp: Utc::now(),
            outcomes: treatment_outcomes,
            per_category: HashMap::new(),
            category_map: HashMap::new(),
            hit_checker_version: rein::eval::HIT_CHECKER_VERSION,
        };

        let baseline_path = tmp_path("synth_cmp_base");
        let treat_path = tmp_path("synth_cmp_treat");
        write_scorecard(&baseline_path, &base).unwrap();
        write_scorecard(&treat_path, &treat).unwrap();

        // Run the compare path through the production code, then load the
        // McNemar via direct calls to confirm the decision the user would
        // see. (cmd_compare prints to stdout; we re-derive the verdict.)
        let res = cmd_compare(
            &baseline_path,
            &treat_path,
            0.02,
            DecideShipKind::Synthesis,
        );
        assert!(res.is_ok(), "compare must succeed: {res:?}");

        // Independent re-derivation: McNemar over (baseline=0, treatment=1)
        // for 30 cases gives c=30 b=0 → exact binomial p ≈ 0 → Superior ship.
        let paired: Vec<PairedOutcome> = base
            .outcomes
            .iter()
            .zip(treat.outcomes.iter())
            .map(|(b, t)| PairedOutcome {
                case_id: b.case_id.clone(),
                baseline_hit: b.baseline_hit,
                treatment_hit: t.treatment_hit,
                baseline_length: b.baseline_length,
                treatment_length: t.treatment_length,
                treatment_summary: None,
            })
            .collect();
        let overall = mcnemar(&paired);
        let decision = decide_ship(
            &overall,
            &HashMap::new(),
            0.02,
            1.5,
            DecideShipKind::Synthesis,
        );
        match decision {
            ShipDecision::Ship {
                reason: ShipReason::Superior { .. },
                ..
            } => {}
            other => panic!(
                "expected Superior ship verdict for clean treatment win, got {other:?}"
            ),
        }

        let _ = fs::remove_file(&baseline_path);
        let _ = fs::remove_file(&treat_path);
    }

    /// `SynthesisFixture::baseline_text` concatenates summary +
    /// evidence_preview per result; verify the exact shape so future
    /// schema changes are explicit.
    #[test]
    fn synthesis_baseline_text_concatenates_summary_and_evidence_preview() {
        let fx = SynthesisFixture {
            case_id: "sample".into(),
            category: None,
            query: "q".into(),
            recall_results: vec![SyntheticRecallResult {
                id: "m1".into(),
                summary: "summary one".into(),
                evidence_preview: vec!["preview one".into(), "preview two".into()],
                score: 0.9,
                confidence: 0.9,
                sources_hit: 2,
                evidence_count: 2,
            }],
            evidence_keywords: vec!["one".into(), "two".into(), "three".into()],
        };
        let text = fx.baseline_text();
        assert!(text.contains("summary one"));
        assert!(text.contains("preview one"));
        assert!(text.contains("preview two"));
    }
}
