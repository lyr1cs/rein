//! `rein-eval` — standalone evaluation harness for the v0.23 resummerize
//! feature.
//!
//! Two of the three subcommands (`baseline`, `run`) are stubs at this stage:
//! they load fixtures and emit a placeholder scorecard so the main thread can
//! wire in real keep-tail / resummerize execution without changing the on-disk
//! shape.
//!
//! The `compare` subcommand is fully implemented: it loads two scorecards,
//! joins them by `case_id`, runs paired McNemar, and prints the ship-or-bail
//! decision.
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
use chrono::Utc;
use clap::{Parser, Subcommand};
use rein::eval::{
    decide_ship, mcnemar, CategoryStats, McNemarResult, PairedOutcome, Scorecard, ShipDecision,
    ShipReason,
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
}

#[derive(Subcommand)]
enum ResummerizeAction {
    /// Run the keep-tail baseline over a directory of fixture cases.
    ///
    /// STUB: currently emits a placeholder scorecard. Main thread should wire
    /// in real keep-tail execution via rein::store / rein::compression.
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
    /// Run the resummerize treatment over a directory of fixture cases.
    ///
    /// STUB: currently emits a placeholder scorecard with `treatment_hit =
    /// false`. Main thread should wire in the real resummerize pipeline.
    Run {
        /// Directory containing fixture JSON files (one per case).
        #[arg(long)]
        fixtures: PathBuf,
        /// Output path for the scorecard JSON.
        #[arg(long, default_value = "treatment_scorecard.json")]
        output: PathBuf,
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

/// Minimal fixture schema used by the stub `baseline`/`run` paths.
///
/// Production fixtures are expected to be JSON files with at minimum a
/// `case_id`; the rest is advisory/optional until the main thread wires real
/// execution. Unknown fields are ignored.
#[derive(Debug, Deserialize, Serialize)]
struct Fixture {
    case_id: String,
    #[serde(default)]
    category: Option<String>,
    #[serde(default)]
    canonical: Option<String>,
    #[serde(default)]
    context: Option<String>,
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
            ResummerizeAction::Run { fixtures, output } => cmd_run(&fixtures, &output),
            ResummerizeAction::Compare {
                baseline,
                treatment,
                noise_floor,
            } => cmd_compare(&baseline, &treatment, noise_floor),
        },
    }
}

// --- baseline / run stubs ---------------------------------------------------

fn cmd_baseline(fixtures: &Path, iterations: u32, output: &Path) -> Result<()> {
    let fixtures_list = load_fixtures(fixtures)?;
    let mut outcomes = Vec::with_capacity(fixtures_list.len());
    for fx in &fixtures_list {
        // TODO(main-thread): wire real keep-tail execution here. For now emit
        // a placeholder outcome so the scorecard JSON shape is exercised.
        outcomes.push(PairedOutcome {
            case_id: fx.case_id.clone(),
            baseline_hit: false,
            treatment_hit: false,
            baseline_length: 0,
            treatment_length: 0,
        });
    }

    let sc = Scorecard {
        fixtures_dir: fixtures.display().to_string(),
        iterations,
        timestamp: Utc::now(),
        outcomes,
        per_category: HashMap::new(),
    };
    write_scorecard(output, &sc)?;
    eprintln!(
        "[rein-eval] baseline (STUB): wrote {} cases to {}",
        fixtures_list.len(),
        output.display()
    );
    Ok(())
}

fn cmd_run(fixtures: &Path, output: &Path) -> Result<()> {
    let fixtures_list = load_fixtures(fixtures)?;
    let mut outcomes = Vec::with_capacity(fixtures_list.len());
    for fx in &fixtures_list {
        // TODO(main-thread): wire real resummerize + LLM execution here. For
        // now emit a placeholder outcome.
        outcomes.push(PairedOutcome {
            case_id: fx.case_id.clone(),
            baseline_hit: false,
            treatment_hit: false,
            baseline_length: 0,
            treatment_length: 0,
        });
    }

    let sc = Scorecard {
        fixtures_dir: fixtures.display().to_string(),
        iterations: 1,
        timestamp: Utc::now(),
        outcomes,
        per_category: HashMap::new(),
    };
    write_scorecard(output, &sc)?;
    eprintln!(
        "[rein-eval] run (STUB): wrote {} cases to {}",
        fixtures_list.len(),
        output.display()
    );
    Ok(())
}

// --- compare (fully implemented) -------------------------------------------

fn cmd_compare(baseline: &Path, treatment: &Path, noise_floor: f64) -> Result<()> {
    let base: Scorecard = load_scorecard(baseline)?;
    let treat: Scorecard = load_scorecard(treatment)?;

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
    // Fast path: if either scorecard already carries per_category, merge &
    // prefer the treatment stats (they reflect the latest run's categories).
    if !treat.per_category.is_empty() {
        return treat.per_category.clone();
    }
    if !base.per_category.is_empty() {
        return base.per_category.clone();
    }

    // Fallback: derive categories from `case_id` prefix before ':' (e.g.
    // "single_session:case_3" -> "single_session"). If a case_id has no
    // colon, we skip it for per-category analysis.
    let mut groups: HashMap<String, Vec<PairedOutcome>> = HashMap::new();
    for o in paired {
        if let Some(cat) = o.case_id.split_once(':').map(|(prefix, _)| prefix.to_string()) {
            groups.entry(cat).or_default().push(o.clone());
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
