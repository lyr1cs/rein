//! v0.27.1 Track 2 — `[llm]` config inheritance back-compat tests.
//!
//! Spec: [`docs/superpowers/specs/2026-04-27-v0.27.1-runtime-llm-judge-design.md`]
//! §8.3 + §11. The contract: every v0.26.x config fixture must resolve
//! identically before and after Track 2 lands. The resolver
//! (`ReinConfig::resolve_llm_for(section)`) is the single source of
//! truth that B2 will migrate consumer call-sites to in Wave 1.5.
//!
//! Each fixture in `tests/fixtures/config/v0.26x/` represents a
//! distinct production-shape pattern from the pre-Track-2 grep audit
//! per spec §8.5 (`/tmp/v0.27.1-llm-callsite-audit.txt`):
//!
//! 1. all-explicit-google — every consumer redundantly declares
//!    `provider = "google"` and its own `[*.google]` block
//! 2. partial-explicit-inherit — `[extract]` is base, slow-channel
//!    sections use `"inherit"`
//! 3. omlx-only-local — single-provider local-OMLX config
//! 4. mixed-provider-per-section — extract on Google, expansion on
//!    OMLX; tests per-section walk independence
//! 5. minimal-defaults-only — only `[database]` set; every other
//!    section gets compiled-in defaults
//!
//! These five cover every distinct pattern emitted by the §8.5 grep.
//!
//! When `[llm]` is absent (every v0.26.x fixture, by definition), the
//! resolver MUST skip levels 2-3 and fall through to level 1 (per-section
//! explicit) and level 4 (hardcoded baseline). Cross-provider walks
//! (Codex R5 P2: provider chosen at level 1 must read scoped fields from
//! that level's provider sub-table, not the OTHER provider) get separate
//! coverage in `walk_as_unit_*` tests below.

use rein::config::{PrecedenceSource, Provider, ReinConfig};

const FIXTURE_DIR: &str = "tests/fixtures/config/v0.26x";

fn load_fixture(name: &str) -> ReinConfig {
    let path = format!("{FIXTURE_DIR}/{name}");
    let toml_str =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    ReinConfig::load_from_str(&toml_str)
        .unwrap_or_else(|e| panic!("parse {path}: {e}"))
}

// ---------------------------------------------------------------------------
// Fixture 1 — all-explicit-google
// ---------------------------------------------------------------------------

#[test]
fn fixture_01_extract_resolves_to_google_flash_lite() {
    let cfg = load_fixture("01_all_explicit_google.toml");
    let r = cfg.resolve_llm_for("extract").unwrap();
    assert_eq!(r.provider, Provider::Google);
    assert_eq!(r.model, "gemini-3.1-flash-lite-preview");
    assert_eq!(r.api_key_env.as_deref(), Some("GEMINI_API_KEY"));
    assert_eq!(r.max_input_chars, 0);
    assert_eq!(r.source, PrecedenceSource::SectionExplicit);
}

#[test]
fn fixture_01_query_expansion_resolves_to_google() {
    let cfg = load_fixture("01_all_explicit_google.toml");
    let r = cfg.resolve_llm_for("query_expansion").unwrap();
    assert_eq!(r.provider, Provider::Google);
    assert_eq!(r.model, "gemini-3.1-flash-lite-preview");
    assert_eq!(r.source, PrecedenceSource::SectionExplicit);
}

#[test]
fn fixture_01_search_llm_reranker_resolves_to_google() {
    let cfg = load_fixture("01_all_explicit_google.toml");
    let r = cfg.resolve_llm_for("search.llm_reranker").unwrap();
    assert_eq!(r.provider, Provider::Google);
    // Reranker reads from [query_expansion.google] in v0.26.x.
    assert_eq!(r.model, "gemini-3.1-flash-lite-preview");
    assert_eq!(r.source, PrecedenceSource::SectionExplicit);
}

#[test]
fn fixture_01_ars_recall_synthesis_resolves_to_google() {
    let cfg = load_fixture("01_all_explicit_google.toml");
    let r = cfg.resolve_llm_for("ars.recall_synthesis").unwrap();
    assert_eq!(r.provider, Provider::Google);
    assert_eq!(r.model, "gemini-3.1-flash-lite-preview");
    assert_eq!(r.max_input_chars, 0);
}

#[test]
fn fixture_01_resummerize_resolves_to_google() {
    let cfg = load_fixture("01_all_explicit_google.toml");
    let r = cfg.resolve_llm_for("resummerize").unwrap();
    assert_eq!(r.provider, Provider::Google);
    assert_eq!(r.model, "gemini-3.1-flash-lite-preview");
}

// ---------------------------------------------------------------------------
// Fixture 2 — partial-explicit-inherit
// ---------------------------------------------------------------------------

#[test]
fn fixture_02_extract_explicit_google() {
    let cfg = load_fixture("02_partial_explicit_inherit.toml");
    let r = cfg.resolve_llm_for("extract").unwrap();
    assert_eq!(r.provider, Provider::Google);
    assert_eq!(r.model, "gemini-3.1-flash-lite-preview");
    assert_eq!(r.source, PrecedenceSource::SectionExplicit);
}

#[test]
fn fixture_02_ars_inherits_to_extract_google() {
    let cfg = load_fixture("02_partial_explicit_inherit.toml");
    // ars.llm_backend = "inherit" → walks back to [extract].provider = "google"
    let r = cfg.resolve_llm_for("ars.recall_synthesis").unwrap();
    assert_eq!(r.provider, Provider::Google);
    assert_eq!(r.model, "gemini-3.1-flash-lite-preview");
    // The "inherit" path is reported as SectionProvider precedence —
    // it's level 2 conceptually (section's `provider = "inherit"`
    // re-uses [extract]).
    assert_eq!(r.source, PrecedenceSource::SectionProvider);
}

#[test]
fn fixture_02_resummerize_inherits_to_extract() {
    let cfg = load_fixture("02_partial_explicit_inherit.toml");
    let r = cfg.resolve_llm_for("resummerize").unwrap();
    assert_eq!(r.provider, Provider::Google);
    assert_eq!(r.model, "gemini-3.1-flash-lite-preview");
}

#[test]
fn fixture_02_async_memory_inherits_to_extract() {
    let cfg = load_fixture("02_partial_explicit_inherit.toml");
    let r = cfg.resolve_llm_for("extract.async_memory").unwrap();
    assert_eq!(r.provider, Provider::Google);
    assert_eq!(r.model, "gemini-3.1-flash-lite-preview");
}

#[test]
fn fixture_02_search_llm_reranker_none() {
    let cfg = load_fixture("02_partial_explicit_inherit.toml");
    // [search].llm_reranker = "none" → resolver returns Provider::None
    let r = cfg.resolve_llm_for("search.llm_reranker").unwrap();
    assert_eq!(r.provider, Provider::None);
    assert_eq!(r.source, PrecedenceSource::SectionExplicit);
}

// ---------------------------------------------------------------------------
// Fixture 3 — omlx-only-local
// ---------------------------------------------------------------------------

#[test]
fn fixture_03_extract_resolves_omlx() {
    let cfg = load_fixture("03_omlx_only_local.toml");
    let r = cfg.resolve_llm_for("extract").unwrap();
    assert_eq!(r.provider, Provider::Omlx);
    assert_eq!(r.model, "qwen2.5:14b");
    assert_eq!(r.endpoint, "http://localhost:11434/v1");
    assert_eq!(r.max_input_chars, 32000);
    assert!(r.api_key_env.is_none(), "OMLX uses no api_key_env");
}

#[test]
fn fixture_03_query_expansion_omlx() {
    let cfg = load_fixture("03_omlx_only_local.toml");
    let r = cfg.resolve_llm_for("query_expansion").unwrap();
    assert_eq!(r.provider, Provider::Omlx);
    assert_eq!(r.model, "qwen2.5:7b");
    assert_eq!(r.endpoint, "http://localhost:8000/v1");
}

#[test]
fn fixture_03_search_reranker_omlx_uses_query_expansion_block() {
    let cfg = load_fixture("03_omlx_only_local.toml");
    let r = cfg.resolve_llm_for("search.llm_reranker").unwrap();
    assert_eq!(r.provider, Provider::Omlx);
    // Reranker reads model + endpoint from [query_expansion.omlx], not
    // [extract.omlx]. That's the v0.26.x semantic.
    assert_eq!(r.model, "qwen2.5:7b");
    assert_eq!(r.endpoint, "http://localhost:8000/v1");
}

#[test]
fn fixture_03_ars_omlx_explicit() {
    let cfg = load_fixture("03_omlx_only_local.toml");
    let r = cfg.resolve_llm_for("ars.concept_summary").unwrap();
    assert_eq!(r.provider, Provider::Omlx);
    // ars.llm_backend = "omlx" — reads from [extract.omlx]
    assert_eq!(r.model, "qwen2.5:14b");
}

// ---------------------------------------------------------------------------
// Fixture 4 — mixed-provider-per-section
// ---------------------------------------------------------------------------

#[test]
fn fixture_04_extract_google_query_expansion_omlx() {
    let cfg = load_fixture("04_mixed_provider_per_section.toml");
    let extract = cfg.resolve_llm_for("extract").unwrap();
    let qe = cfg.resolve_llm_for("query_expansion").unwrap();
    assert_eq!(extract.provider, Provider::Google);
    assert_eq!(extract.model, "gemini-3.1-flash-lite-preview");
    assert_eq!(qe.provider, Provider::Omlx);
    assert_eq!(qe.model, "qwen2.5:7b");
    // Per-section walk independence — these are independent walks.
}

#[test]
fn fixture_04_reranker_omlx_extract_google_dont_collide() {
    let cfg = load_fixture("04_mixed_provider_per_section.toml");
    let rr = cfg.resolve_llm_for("search.llm_reranker").unwrap();
    let ex = cfg.resolve_llm_for("extract").unwrap();
    assert_eq!(rr.provider, Provider::Omlx);
    assert_eq!(rr.model, "qwen2.5:7b");
    // Critical: extract is still Google despite reranker being OMLX.
    assert_eq!(ex.provider, Provider::Google);
    assert_ne!(rr.model, ex.model);
}

#[test]
fn fixture_04_ars_inherits_extract_google_not_reranker_omlx() {
    let cfg = load_fixture("04_mixed_provider_per_section.toml");
    // [ars].llm_backend = "inherit" follows [extract], NOT
    // [query_expansion]. Confirms walk-by-section.
    let r = cfg.resolve_llm_for("ars.recall_synthesis").unwrap();
    assert_eq!(r.provider, Provider::Google);
    assert_eq!(r.model, "gemini-3.1-flash-lite-preview");
}

#[test]
fn fixture_04_resummerize_explicit_google_overrides_inherit_path() {
    let cfg = load_fixture("04_mixed_provider_per_section.toml");
    let r = cfg.resolve_llm_for("resummerize").unwrap();
    assert_eq!(r.provider, Provider::Google);
    assert_eq!(r.source, PrecedenceSource::SectionExplicit);
}

// ---------------------------------------------------------------------------
// Fixture 5 — minimal-defaults-only
// ---------------------------------------------------------------------------

#[test]
fn fixture_05_extract_uses_compiled_default_google() {
    let cfg = load_fixture("05_minimal_defaults_only.toml");
    let r = cfg.resolve_llm_for("extract").unwrap();
    // ExtractConfig::default() sets provider = "google" + the
    // gemini-3.1-flash-lite-preview model. Resolver reads through that
    // level-1 explicit branch.
    assert_eq!(r.provider, Provider::Google);
    assert_eq!(r.model, "gemini-3.1-flash-lite-preview");
    assert_eq!(r.api_key_env.as_deref(), Some("GEMINI_API_KEY"));
}

#[test]
fn fixture_05_search_reranker_defaults_none() {
    let cfg = load_fixture("05_minimal_defaults_only.toml");
    let r = cfg.resolve_llm_for("search.llm_reranker").unwrap();
    // [search].llm_reranker default is "none"
    assert_eq!(r.provider, Provider::None);
}

#[test]
fn fixture_05_ars_defaults_inherit_to_extract() {
    let cfg = load_fixture("05_minimal_defaults_only.toml");
    let r = cfg.resolve_llm_for("ars.recall_synthesis").unwrap();
    // ArsConfig default llm_backend = "inherit" → walks to extract default = google
    assert_eq!(r.provider, Provider::Google);
    assert_eq!(r.model, "gemini-3.1-flash-lite-preview");
}

#[test]
fn fixture_05_ars_llm_judge_defaults_no_llm_block_returns_none() {
    let cfg = load_fixture("05_minimal_defaults_only.toml");
    // ars.llm_judge is a v0.27.1-new section with no level-1 explicit
    // shape. Without `[llm]` configured, it must fall through to
    // hardcoded baseline = Provider::None (no LLM call).
    let r = cfg.resolve_llm_for("ars.llm_judge").unwrap();
    assert_eq!(r.provider, Provider::None);
    assert_eq!(r.source, PrecedenceSource::HardcodedFallback);
}

// ---------------------------------------------------------------------------
// Cross-cutting walk-as-unit tests (Codex R5 P2 — most subtle invariant).
//
// These go beyond fixture replay: they assert the resolver does NOT do
// cross-provider walks. Provider chosen at one level must read scoped
// fields from THAT provider's sub-table at the same level (or the
// chain-back of the same provider), never from the other provider's
// sub-table.
// ---------------------------------------------------------------------------

#[test]
fn walk_as_unit_omlx_section_does_not_read_google_model() {
    // Operator pins extract = "omlx" and configures only [extract.omlx].
    // [extract.google].model still has its compiled default
    // ("gemini-3.1-flash-lite-preview"), but the resolver MUST NOT
    // cross-pollinate.
    let toml_str = r#"
[database]
path = "/tmp/walk-as-unit-1.db"

[extract]
provider = "omlx"

[extract.omlx]
endpoint = "http://localhost:11434/v1"
model = "llama3.1:70b"
max_input_chars = 64000
"#;
    let cfg = ReinConfig::load_from_str(toml_str).unwrap();
    let r = cfg.resolve_llm_for("extract").unwrap();
    assert_eq!(r.provider, Provider::Omlx);
    assert_eq!(r.model, "llama3.1:70b", "must read [extract.omlx].model");
    assert_ne!(
        r.model, "gemini-3.1-flash-lite-preview",
        "MUST NOT fall through to [extract.google].model — that would be cross-provider walk"
    );
}

#[test]
fn walk_as_unit_google_section_does_not_read_omlx_endpoint() {
    let toml_str = r#"
[database]
path = "/tmp/walk-as-unit-2.db"

[extract]
provider = "google"

[extract.google]
model = "gemini-3.1-pro"

[extract.omlx]
endpoint = "http://nope-this-is-omlx.invalid/v1"
model = "ignored"
"#;
    let cfg = ReinConfig::load_from_str(toml_str).unwrap();
    let r = cfg.resolve_llm_for("extract").unwrap();
    assert_eq!(r.provider, Provider::Google);
    assert_eq!(r.model, "gemini-3.1-pro");
    assert!(
        !r.endpoint.contains("nope-this-is-omlx"),
        "Google walk must NOT read [extract.omlx].endpoint, got: {}",
        r.endpoint
    );
}

#[test]
fn fail_fast_when_provider_chosen_but_no_model_anywhere() {
    // [llm].provider = "omlx" + [llm.omlx] missing → ars.llm_judge has
    // nowhere to read a model from. Codex R5 P2 fail-fast contract says
    // resolver MUST return Err(Config(...)).
    //
    // Note: we don't trip this on `extract` because ExtractConfig's
    // default already supplies a model, so the `[extract.omlx].model =
    // "default"` baseline saves us. But ars.llm_judge has no level-1
    // shape, so this is the canonical fail-fast check.
    let toml_str = r#"
[database]
path = "/tmp/fail-fast.db"

[llm]
provider = "omlx"
"#;
    let cfg = ReinConfig::load_from_str(toml_str).unwrap();
    let err = cfg.resolve_llm_for("ars.llm_judge").unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("model is configured"),
        "expected fail-fast 'model is configured' in error, got: {msg}"
    );
}

#[test]
fn unknown_section_returns_config_error() {
    let cfg = ReinConfig::default();
    let err = cfg.resolve_llm_for("not.a.real.section").unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("unknown section"),
        "expected 'unknown section' in error, got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// J6 invariant — ars_llm_judge weight_decay_rate validation
// ---------------------------------------------------------------------------

#[test]
fn j6_validate_rejects_negative_weight_decay_rate() {
    let toml_str = r#"
[database]
path = "/tmp/j6-neg.db"

[ars.llm_judge]
weight_decay_rate = -0.1
"#;
    let err = ReinConfig::load_from_str(toml_str).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("weight_decay_rate"),
        "expected weight_decay_rate complaint, got: {msg}"
    );
}

#[test]
fn j6_validate_rejects_above_one_weight_decay_rate() {
    let toml_str = r#"
[database]
path = "/tmp/j6-gt1.db"

[ars.llm_judge]
weight_decay_rate = 1.5
"#;
    let err = ReinConfig::load_from_str(toml_str).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("weight_decay_rate"));
}

#[test]
fn j6_validate_rejects_nan_weight_decay_rate() {
    let toml_str = r#"
[database]
path = "/tmp/j6-nan.db"

[ars.llm_judge]
weight_decay_rate = nan
"#;
    let err = ReinConfig::load_from_str(toml_str).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("weight_decay_rate"));
}

#[test]
fn j6_validate_accepts_zero_weight_decay_rate() {
    let toml_str = r#"
[database]
path = "/tmp/j6-zero.db"

[ars.llm_judge]
weight_decay_rate = 0.0
"#;
    let cfg = ReinConfig::load_from_str(toml_str).unwrap();
    assert_eq!(cfg.ars.llm_judge.weight_decay_rate, 0.0);
}

#[test]
fn j6_validate_accepts_one_weight_decay_rate() {
    let toml_str = r#"
[database]
path = "/tmp/j6-one.db"

[ars.llm_judge]
weight_decay_rate = 1.0
"#;
    let cfg = ReinConfig::load_from_str(toml_str).unwrap();
    assert_eq!(cfg.ars.llm_judge.weight_decay_rate, 1.0);
}

// ---------------------------------------------------------------------------
// `[llm]` parent block — when present, takes effect at level 3.
// ---------------------------------------------------------------------------

#[test]
fn llm_parent_block_overrides_compiled_defaults_for_new_sections() {
    // ars.llm_judge had no level-1 shape. With [llm] set, level 3 fires.
    let toml_str = r#"
[database]
path = "/tmp/llm-parent.db"

[llm]
provider = "google"

[llm.google]
model = "gemini-3.1-pro"
api_key_env = "MY_CUSTOM_KEY"
"#;
    let cfg = ReinConfig::load_from_str(toml_str).unwrap();
    let r = cfg.resolve_llm_for("ars.llm_judge").unwrap();
    assert_eq!(r.provider, Provider::Google);
    assert_eq!(r.model, "gemini-3.1-pro");
    assert_eq!(r.api_key_env.as_deref(), Some("MY_CUSTOM_KEY"));
    assert_eq!(r.source, PrecedenceSource::GlobalDefault);
}

#[test]
fn llm_parent_block_does_not_override_section_explicit() {
    // [extract] explicit must beat [llm] global default.
    let toml_str = r#"
[database]
path = "/tmp/llm-vs-extract.db"

[llm]
provider = "google"

[llm.google]
model = "gemini-3.1-pro"

[extract]
provider = "omlx"

[extract.omlx]
endpoint = "http://localhost:11434/v1"
model = "qwen2.5:14b"
"#;
    let cfg = ReinConfig::load_from_str(toml_str).unwrap();
    let r = cfg.resolve_llm_for("extract").unwrap();
    assert_eq!(r.provider, Provider::Omlx);
    assert_eq!(r.model, "qwen2.5:14b");
    assert_eq!(r.source, PrecedenceSource::SectionExplicit);
}

#[test]
fn nightly_cron_falls_back_to_judge_then_llm() {
    let toml_str = r#"
[database]
path = "/tmp/cron-fallback.db"

[llm]
provider = "google"

[llm.google]
model = "gemini-3.1-pro"
"#;
    let cfg = ReinConfig::load_from_str(toml_str).unwrap();
    // No [ars.llm_judge.nightly_cron] override → resolves through judge → [llm]
    let r = cfg.resolve_llm_for("ars.llm_judge.nightly_cron").unwrap();
    assert_eq!(r.provider, Provider::Google);
    assert_eq!(r.model, "gemini-3.1-pro");
}
