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
//! Cargo auto-discovers `src/bin/rein_eval.rs` as a binary target named
//! `rein_eval` (underscore). If the hyphenated name `rein-eval` is desired,
//! the main thread needs to add a `[[bin]]` entry in `crates/rein/Cargo.toml`:
//!
//! ```toml
//! [[bin]]
//! name = "rein-eval"
//! path = "src/bin/rein_eval.rs"
//! ```
//!
//! Until then, invoke as `cargo run -p rein --bin rein_eval -- ...`.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use chrono::{DateTime, Utc};
use clap::{Parser, Subcommand};
use rein::compression::contract::{self, ContractInput, EvidenceEntry};
use rein::config::ReinConfig;
use rein::eval::{
    decide_ship, mcnemar, CategoryStats, HitChecker, KeywordOverlapHitChecker, McNemarResult,
    PairedOutcome, Scorecard, ShipDecision, ShipReason,
};
use rein::extract::llm::strip_code_fences;
// NOTE: `call_llm_sync` in ops::resummerize uses SYSTEM_PROMPT internally —
// the eval bin doesn't need to import it directly. Importing `build_prompt`
// and `call_llm_sync` is enough; the system prompt travels with the call.
// `create_resummerize_extractor` is used instead of `create_extractor` so
// the eval honors `[resummerize].llm_backend` the same way production
// does (post-fix audit M-1).
use rein::ops::resummerize::{build_prompt, call_llm_sync, create_resummerize_extractor};
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
            } => cmd_compare(&baseline, &treatment, noise_floor),
        },
    }
}

// --- baseline / run -------------------------------------------------------

fn cmd_baseline(fixtures: &Path, iterations: u32, output: &Path) -> Result<()> {
    let fixtures_list = load_fixtures(fixtures)?;
    if fixtures_list.is_empty() {
        bail!("no fixtures found in {}", fixtures.display());
    }

    let checker = KeywordOverlapHitChecker;
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

    run_treatment_with_extractor(&fixtures_list, &extractor, iterations, output, fixtures, verbose)
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
    let checker = KeywordOverlapHitChecker;
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
            eprintln!(
                "[rein-eval] run: skipping {} (no evidence)",
                fx.case_id
            );
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
            match call_llm_sync(extractor, &prompt) {
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

fn cmd_compare(baseline: &Path, treatment: &Path, noise_floor: f64) -> Result<()> {
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
    let base_by_id: HashMap<&str, &PairedOutcome> =
        base.outcomes.iter().map(|o| (o.case_id.as_str(), o)).collect();
    let treat_by_id: HashMap<&str, &PairedOutcome> =
        treat.outcomes.iter().map(|o| (o.case_id.as_str(), o)).collect();

    let mut paired: Vec<PairedOutcome> = Vec::new();
    for (id, base_o) in &base_by_id {
        if let Some(treat_o) = treat_by_id.get(id) {
            paired.push(PairedOutcome {
                case_id: base_o.case_id.clone(),
                baseline_hit: base_o.baseline_hit,
                treatment_hit: treat_o.treatment_hit,
                baseline_length: base_o.baseline_length,
                treatment_length: treat_o.treatment_length,
            });
        }
    }

    let only_in_baseline = base_by_id.keys().filter(|k| !treat_by_id.contains_key(*k)).count();
    let only_in_treatment = treat_by_id.keys().filter(|k| !base_by_id.contains_key(*k)).count();
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

    let decision = decide_ship(&overall, &per_category, noise_floor, ratio);

    print_summary(&paired, &overall, &per_category, mean_base_len, mean_treat_len, ratio);
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
            if let Some(cat) = o.case_id.split_once(':').map(|(prefix, _)| prefix.to_string()) {
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
    println!("=== rein-eval: resummerize compare ===");
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
                s.n, s.baseline_hit_rate, s.treatment_hit_rate, s.mcnemar.diff_point, s.mcnemar.p_value
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
        },
        ShipDecision::BailOut { reason, .. } => {
            println!("BAIL OUT: {reason}");
        }
    }
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
    for entry in fs::read_dir(dir)
        .with_context(|| format!("reading fixtures dir {}", dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let bytes = fs::read(&path)
            .with_context(|| format!("reading fixture {}", path.display()))?;
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
    let bytes =
        fs::read(path).with_context(|| format!("reading scorecard {}", path.display()))?;
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
        p.push(format!("rein_eval_test_{}_{}.json", name, std::process::id()));
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
        assert_eq!(sc.category_map.get("cjk_001").map(String::as_str), Some("cjk"));
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
        let extractor = ExtractorKind::Mock(MockExtractor::with_responses(vec![Ok(
            oversize_output,
        )]));
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
            mock_output.to_string(),
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
                },
                PairedOutcome {
                    case_id: "cjk_002".into(),
                    baseline_hit: false,
                    treatment_hit: false,
                    baseline_length: 200,
                    treatment_length: 0,
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
                },
                PairedOutcome {
                    case_id: "cjk_002".into(),
                    baseline_hit: false,
                    treatment_hit: true,
                    baseline_length: 200,
                    treatment_length: 80,
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
}
