//! Provider wiring and credential status — one report consumed by
//! `rein providers`, `rein dashboard`, `rein config`, `GET /api/providers`,
//! the GUI AI Keys page and `rein doctor`. Never contains a plaintext key.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::config::{PrecedenceSource, Provider, ReinConfig, ResolvedLlmConfig};
use crate::store::SqliteStore;

/// Human-facing grouping used by the GUI and the CLI `role` column.
pub const ROLES: &[&str] = &["embed", "write", "search", "answer", "judge", "cloud"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceKind {
    /// Resolved through `ReinConfig::resolve_llm_for(section)`.
    Llm(&'static str),
    Embedding,
    Supermemory,
}

#[derive(Debug, Clone, Copy)]
pub struct SurfaceDef {
    pub name: &'static str,
    pub role: &'static str,
    pub kind: SurfaceKind,
}

/// Fixed order: the GUI flow diagram and the CLI table print in this order.
pub const SURFACES: &[SurfaceDef] = &[
    SurfaceDef {
        name: "embedding",
        role: "embed",
        kind: SurfaceKind::Embedding,
    },
    SurfaceDef {
        name: "extract",
        role: "write",
        kind: SurfaceKind::Llm("extract"),
    },
    SurfaceDef {
        name: "extract.async_memory",
        role: "write",
        kind: SurfaceKind::Llm("extract.async_memory"),
    },
    SurfaceDef {
        name: "extract.intelligent_merge",
        role: "write",
        kind: SurfaceKind::Llm("extract.intelligent_merge"),
    },
    SurfaceDef {
        name: "query_expansion",
        role: "search",
        kind: SurfaceKind::Llm("query_expansion"),
    },
    SurfaceDef {
        name: "search.llm_reranker",
        role: "search",
        kind: SurfaceKind::Llm("search.llm_reranker"),
    },
    SurfaceDef {
        name: "ars.recall_synthesis",
        role: "answer",
        kind: SurfaceKind::Llm("ars.recall_synthesis"),
    },
    SurfaceDef {
        name: "ars.concept_summary",
        role: "answer",
        kind: SurfaceKind::Llm("ars.concept_summary"),
    },
    SurfaceDef {
        name: "ars.cold_archive",
        role: "answer",
        kind: SurfaceKind::Llm("ars.cold_archive"),
    },
    SurfaceDef {
        name: "resummerize",
        role: "write",
        kind: SurfaceKind::Llm("resummerize"),
    },
    SurfaceDef {
        name: "ars.llm_judge",
        role: "judge",
        kind: SurfaceKind::Llm("ars.llm_judge"),
    },
    SurfaceDef {
        name: "ars.llm_judge.nightly_cron",
        role: "judge",
        kind: SurfaceKind::Llm("ars.llm_judge.nightly_cron"),
    },
    SurfaceDef {
        name: "supermemory",
        role: "cloud",
        kind: SurfaceKind::Supermemory,
    },
];

#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, PartialEq)]
pub struct SurfaceStatus {
    pub surface: String,
    pub role: String,
    pub enabled: bool,
    pub provider: String,
    pub model: String,
    pub endpoint: String,
    pub source_level: String,
    pub source_table: String,
    pub credential: Option<String>,
    pub status: String,
}

#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, PartialEq)]
pub struct CredentialStatus {
    pub id: String,
    pub kind: String,
    pub present: bool,
    pub masked: Option<String>,
    pub used_by: Vec<String>,
    pub warning: Option<String>,
    pub probe: Option<ProbeResult>,
}

#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, PartialEq)]
pub struct VectorIndexStatus {
    pub stamped: Option<String>,
    pub configured: String,
    pub matches: bool,
    pub rows: u64,
}

#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, PartialEq)]
pub struct ProbeResult {
    pub status: String,
    pub http_status: Option<u16>,
    pub latency_ms: u64,
    pub detail: Option<String>,
}

#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, PartialEq)]
pub struct ProviderStatusReport {
    pub surfaces: Vec<SurfaceStatus>,
    pub credentials: Vec<CredentialStatus>,
    pub vector_index: Option<VectorIndexStatus>,
    pub probed_at: Option<String>,
}

/// First 4 + `••••` + last 4 for ≥ 12 scalars, 2 + 2 for 8–11, dots only below.
pub fn mask_secret(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let (head, tail) = match chars.len() {
        n if n >= 12 => (4, 4),
        n if n >= 8 => (2, 2),
        _ => return "••••".to_string(),
    };
    let mut out: String = chars[..head].iter().collect();
    out.push_str("••••");
    out.extend(chars[chars.len() - tail..].iter());
    out
}

fn raw_provider_active(value: &str) -> bool {
    !matches!(value.trim().to_ascii_lowercase().as_str(), "" | "none")
}

fn surface_enabled(cfg: &ReinConfig, def: &SurfaceDef) -> bool {
    match def.name {
        "embedding" => cfg.embedding_provider() != Provider::None,
        "extract" => raw_provider_active(&cfg.extract.provider),
        "extract.async_memory" => raw_provider_active(&cfg.async_memory.provider),
        "extract.intelligent_merge" => cfg.intelligent_merge.enabled,
        "query_expansion" => raw_provider_active(&cfg.query_expansion.provider),
        "search.llm_reranker" => raw_provider_active(&cfg.search.llm_reranker),
        "ars.recall_synthesis" => cfg.ars.recall_synthesis_enabled,
        "ars.concept_summary" => cfg.ars.concept_summary_enabled,
        "ars.cold_archive" => cfg.ars.cold_archive_enabled,
        "resummerize" => cfg.resummerize.enabled,
        "ars.llm_judge" => cfg.judge_runtime_active(),
        "ars.llm_judge.nightly_cron" => cfg.judge_nightly_active(),
        "supermemory" => cfg.sync.supermemory_enabled,
        _ => false,
    }
}

fn source_level(r: &ResolvedLlmConfig) -> &'static str {
    match r.source {
        PrecedenceSource::SectionExplicit => "section_explicit",
        PrecedenceSource::SectionProvider => "section_provider",
        PrecedenceSource::GlobalDefault => "global_default",
        PrecedenceSource::HardcodedFallback => "hardcoded_fallback",
    }
}

/// The `[<section>.google].api_key` field that `load()` may have filled from
/// the environment or that the operator wrote into `config.toml`.
fn google_api_key_field<'a>(cfg: &'a ReinConfig, section: &str) -> (String, Option<&'a String>) {
    match section {
        "embedding" => (
            "embedding.google.api_key".to_string(),
            cfg.embedding.google.api_key.as_ref(),
        ),
        "query_expansion" | "search.llm_reranker" => (
            "query_expansion.google.api_key".to_string(),
            cfg.query_expansion.google.api_key.as_ref(),
        ),
        "extract.intelligent_merge" => (
            "intelligent_merge.google.api_key".to_string(),
            cfg.intelligent_merge.google.api_key.as_ref(),
        ),
        _ => (
            "extract.google.api_key".to_string(),
            cfg.extract.google.api_key.as_ref(),
        ),
    }
}

struct CredentialSeed {
    id: String,
    kind: String,
    present: bool,
    masked: Option<String>,
    warning: Option<String>,
}

/// Classify a key by env name: process env → secrets file → config.toml field.
fn credential_for(
    cfg: &ReinConfig,
    env_name: &str,
    config_field: Option<(&str, &String)>,
) -> CredentialSeed {
    let from_env = std::env::var(env_name)
        .ok()
        .filter(|v| !v.trim().is_empty());
    let from_file = cfg
        .secrets
        .get(env_name)
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty());
    if let Some(v) = from_env {
        return CredentialSeed {
            id: format!("env:{env_name}"),
            kind: "env".into(),
            present: true,
            masked: Some(mask_secret(v.trim())),
            warning: None,
        };
    }
    if let Some(v) = from_file {
        return CredentialSeed {
            id: format!("env:{env_name}"),
            kind: "secrets_file".into(),
            present: true,
            masked: Some(mask_secret(&v)),
            warning: None,
        };
    }
    if let Some((path, value)) = config_field {
        return CredentialSeed {
            id: format!("config:{path}"),
            kind: "config_file".into(),
            present: true,
            masked: Some(mask_secret(value.trim())),
            warning: Some(format!(
                "plaintext in config.toml; move it with `rein keys set {env_name}` and delete the api_key line"
            )),
        };
    }
    CredentialSeed {
        id: format!("env:{env_name}"),
        kind: "env".into(),
        present: false,
        masked: None,
        warning: None,
    }
}

struct Resolved {
    provider: String,
    model: String,
    endpoint: String,
    source_level: String,
    source_table: String,
    credential: Option<CredentialSeed>,
}

fn resolve_surface(cfg: &ReinConfig, def: &SurfaceDef) -> Resolved {
    match def.kind {
        SurfaceKind::Llm(section) => {
            let Ok(r) = cfg.resolve_llm_for(section) else {
                return Resolved {
                    provider: "none".into(),
                    model: String::new(),
                    endpoint: String::new(),
                    source_level: "hardcoded_fallback".into(),
                    source_table: "built-in".into(),
                    credential: None,
                };
            };
            let credential = match r.provider {
                Provider::None => None,
                Provider::Google => {
                    let env_name = r
                        .api_key_env
                        .clone()
                        .unwrap_or_else(|| "GEMINI_API_KEY".to_string());
                    let (path, field) = google_api_key_field(cfg, section);
                    Some(credential_for(
                        cfg,
                        &env_name,
                        field.map(|v| (path.as_str(), v)),
                    ))
                }
                Provider::Omlx => r
                    .api_key_env
                    .as_deref()
                    .map(|env_name| credential_for(cfg, env_name, None)),
            };
            Resolved {
                provider: r.provider_name.clone(),
                model: r.model.clone(),
                endpoint: r.endpoint.clone(),
                source_level: source_level(&r).into(),
                source_table: r.source_table.clone(),
                credential,
            }
        }
        SurfaceKind::Embedding => {
            let name = cfg.embedding_provider_name();
            match cfg.embedding_provider() {
                Provider::None => Resolved {
                    provider: "none".into(),
                    model: String::new(),
                    endpoint: String::new(),
                    source_level: "section".into(),
                    source_table: "[embedding]".into(),
                    credential: None,
                },
                Provider::Google => {
                    let env_name = cfg
                        .embedding
                        .google
                        .api_key_env
                        .clone()
                        .filter(|s| !s.trim().is_empty())
                        .unwrap_or_else(|| "GEMINI_API_KEY".to_string());
                    let field = cfg
                        .embedding
                        .google
                        .api_key
                        .as_ref()
                        .map(|v| ("embedding.google.api_key", v));
                    Resolved {
                        provider: "google".into(),
                        model: cfg.embedding.google.model.clone(),
                        endpoint: cfg.embedding.google.endpoint.clone(),
                        source_level: "section".into(),
                        source_table: "[embedding.google]".into(),
                        credential: Some(credential_for(cfg, &env_name, field)),
                    }
                }
                Provider::Omlx => {
                    let preset = crate::vendors::vendor_preset(&name);
                    let table = &cfg.embedding.omlx;
                    let endpoint =
                        crate::embed::resolve_embedding_endpoint(&name, &table.endpoint, preset)
                            .unwrap_or_default();
                    let env_name = table
                        .api_key_env
                        .clone()
                        .filter(|s| !s.trim().is_empty())
                        .or_else(|| preset.and_then(|p| p.api_key_env).map(|s| s.to_string()));
                    Resolved {
                        provider: name.clone(),
                        model: table.model.clone(),
                        endpoint,
                        source_level: "section".into(),
                        source_table: if name == "omlx" {
                            "[embedding.omlx]".into()
                        } else {
                            "[embedding.openai]".into()
                        },
                        credential: env_name.map(|e| credential_for(cfg, &e, None)),
                    }
                }
            }
        }
        SurfaceKind::Supermemory => Resolved {
            provider: "supermemory".into(),
            model: String::new(),
            endpoint: cfg.sync.endpoint.clone(),
            source_level: "section".into(),
            source_table: "[sync]".into(),
            credential: Some(credential_for(
                cfg,
                "SUPERMEMORY_CC_API_KEY",
                cfg.sync.api_key.as_ref().map(|v| ("sync.api_key", v)),
            )),
        },
    }
}

/// Build the report. `store` adds the vector-index stamp row.
pub fn collect(cfg: &ReinConfig, store: Option<&SqliteStore>) -> ProviderStatusReport {
    let mut surfaces = Vec::with_capacity(SURFACES.len());
    let mut credentials: Vec<CredentialStatus> = Vec::new();
    for def in SURFACES {
        let enabled = surface_enabled(cfg, def);
        let r = resolve_surface(cfg, def);
        let status = if !enabled || r.provider == "none" {
            "disabled"
        } else if r.credential.as_ref().is_some_and(|c| !c.present) {
            "missing_credential"
        } else {
            "ready"
        };
        let credential_id = r.credential.as_ref().map(|c| c.id.clone());
        if let Some(seed) = r.credential {
            match credentials.iter_mut().find(|c| c.id == seed.id) {
                Some(existing) => existing.used_by.push(def.name.to_string()),
                None => credentials.push(CredentialStatus {
                    id: seed.id,
                    kind: seed.kind,
                    present: seed.present,
                    masked: seed.masked,
                    used_by: vec![def.name.to_string()],
                    warning: seed.warning,
                    probe: None,
                }),
            }
        }
        surfaces.push(SurfaceStatus {
            surface: def.name.to_string(),
            role: def.role.to_string(),
            enabled,
            provider: r.provider,
            model: r.model,
            endpoint: r.endpoint,
            source_level: r.source_level,
            source_table: r.source_table,
            credential: credential_id,
            status: status.to_string(),
        });
    }
    let vector_index = store.map(|s| vector_index_status(cfg, s));
    ProviderStatusReport {
        surfaces,
        credentials,
        vector_index,
        probed_at: None,
    }
}

pub fn vector_index_status(cfg: &ReinConfig, store: &SqliteStore) -> VectorIndexStatus {
    use rusqlite::OptionalExtension;
    let configured = format!("{}:{}", cfg.embedding_model(), cfg.embedding.dimensions);
    let stamped: Option<String> = store
        .conn()
        .query_row(
            "SELECT value FROM metadata WHERE key = ?1",
            rusqlite::params![crate::store::schema::VEC_ROWS_PROVENANCE_KEY],
            |row| row.get(0),
        )
        .optional()
        .ok()
        .flatten();
    let rows: u64 = store
        .conn()
        .query_row("SELECT COUNT(*) FROM vec_memories", [], |row| {
            row.get::<_, i64>(0)
        })
        .map(|n| n.max(0) as u64)
        .unwrap_or(0);
    let matches = match &stamped {
        Some(s) => *s == configured,
        None => rows == 0,
    };
    VectorIndexStatus {
        stamped,
        configured,
        matches,
        rows,
    }
}

/// Plaintext of a `config:<table>.google.api_key` / `config:sync.api_key`
/// credential. Only the reveal route calls this.
pub fn config_file_key_value(cfg: &ReinConfig, id: &str) -> Option<String> {
    match id {
        "config:embedding.google.api_key" => cfg.embedding.google.api_key.clone(),
        "config:extract.google.api_key" => cfg.extract.google.api_key.clone(),
        "config:query_expansion.google.api_key" => cfg.query_expansion.google.api_key.clone(),
        "config:intelligent_merge.google.api_key" => cfg.intelligent_merge.google.api_key.clone(),
        "config:sync.api_key" => cfg.sync.api_key.clone(),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ProbeTarget {
    /// `GET {endpoint}/v1beta/models?pageSize=1` with `x-goog-api-key`.
    GoogleModels { endpoint: String, key: String },
    /// `GET {endpoint}/models` with the resolved auth header (if any).
    OpenAiModels {
        endpoint: String,
        key: Option<(crate::vendors::AuthHeader, String)>,
    },
    /// A real embedding call through `create_embedder`; checks dimensions.
    Embedding,
    /// `POST {endpoint}/v4/search` bearer.
    Supermemory { endpoint: String, key: String },
}

fn classify(status: reqwest::StatusCode) -> &'static str {
    match status.as_u16() {
        200..=299 => "ok",
        400 | 401 | 403 => "unauthorized",
        _ => "error",
    }
}

fn transport_result(err: &reqwest::Error) -> ProbeResult {
    let status = if err.is_connect() || err.is_timeout() {
        "unreachable"
    } else {
        "error"
    };
    ProbeResult {
        status: status.into(),
        http_status: None,
        latency_ms: 0,
        detail: None,
    }
}

async fn run_probe(cfg: &ReinConfig, target: &ProbeTarget) -> ProbeResult {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .expect("reqwest client");
    let started = std::time::Instant::now();
    let finish = |status: &str, http: Option<u16>, detail: Option<String>| ProbeResult {
        status: status.to_string(),
        http_status: http,
        latency_ms: started.elapsed().as_millis() as u64,
        detail,
    };
    match target {
        ProbeTarget::GoogleModels { endpoint, key } => {
            let url = format!("{endpoint}/v1beta/models?pageSize=1");
            match client.get(url).header("x-goog-api-key", key).send().await {
                Ok(resp) => finish(classify(resp.status()), Some(resp.status().as_u16()), None),
                Err(e) => transport_result(&e),
            }
        }
        ProbeTarget::OpenAiModels { endpoint, key } => {
            let mut rb = client.get(format!("{endpoint}/models"));
            if let Some((header, value)) = key {
                rb = rb.header(header.header_name(), header.header_value(value));
            }
            match rb.send().await {
                Ok(resp) => finish(classify(resp.status()), Some(resp.status().as_u16()), None),
                Err(e) => transport_result(&e),
            }
        }
        ProbeTarget::Supermemory { endpoint, key } => {
            let body = serde_json::json!({"q": "rein", "limit": 1});
            match client
                .post(format!("{endpoint}/v4/search"))
                .bearer_auth(key)
                .json(&body)
                .send()
                .await
            {
                Ok(resp) => finish(classify(resp.status()), Some(resp.status().as_u16()), None),
                Err(e) => transport_result(&e),
            }
        }
        ProbeTarget::Embedding => {
            let Some(embedder) = crate::embed::create_embedder(cfg) else {
                return finish("error", None, Some("embedder not configured".into()));
            };
            use crate::types::traits::Embedder;
            match embedder.embed("rein providers probe").await {
                Ok(v) if v.len() == cfg.embedding.dimensions => finish("ok", Some(200), None),
                Ok(v) => finish(
                    "dimension_mismatch",
                    Some(200),
                    Some(format!(
                        "expected {} dimensions, got {}",
                        cfg.embedding.dimensions,
                        v.len()
                    )),
                ),
                Err(e) => {
                    // Error strings are already content-free (provider_status_error).
                    let msg = e.to_string();
                    let http = msg
                        .split(|c: char| !c.is_ascii_digit())
                        .filter_map(|w| w.parse::<u16>().ok())
                        .find(|n| (100..600).contains(n));
                    let status = match http {
                        Some(400 | 401 | 403) => "unauthorized",
                        Some(_) => "error",
                        None if msg.contains("dimension mismatch") => "dimension_mismatch",
                        None => "unreachable",
                    };
                    let detail = msg
                        .split("dimension mismatch: ")
                        .nth(1)
                        .map(|s| s.to_string());
                    finish(status, http, detail)
                }
            }
        }
    }
}

/// Run one probe per distinct target and attach the result to every
/// credential (or synthetic endpoint row) that target belongs to.
pub async fn probe(report: &mut ProviderStatusReport, cfg: &ReinConfig) {
    use std::collections::HashMap;
    let mut plan: Vec<(String, ProbeTarget)> = Vec::new();
    for def in SURFACES {
        let Some(s) = report
            .surfaces
            .iter()
            .find(|s| s.surface == def.name)
            .cloned()
        else {
            continue;
        };
        if s.status == "disabled" || !s.enabled {
            continue;
        }
        let target = match def.kind {
            SurfaceKind::Embedding => ProbeTarget::Embedding,
            SurfaceKind::Supermemory => {
                let Some(key) = cfg
                    .secret("SUPERMEMORY_CC_API_KEY")
                    .or_else(|| cfg.sync.api_key.clone())
                else {
                    continue;
                };
                ProbeTarget::Supermemory {
                    endpoint: s.endpoint.clone(),
                    key,
                }
            }
            SurfaceKind::Llm(section) => {
                let Ok(r) = cfg.resolve_llm_for(section) else {
                    continue;
                };
                match r.provider {
                    Provider::None => continue,
                    Provider::Google => {
                        let key = crate::extract::llm::resolved_api_key(&r)
                            .or_else(|| google_api_key_field(cfg, section).1.cloned());
                        let Some(key) = key else { continue };
                        ProbeTarget::GoogleModels {
                            endpoint: r.endpoint.clone(),
                            key,
                        }
                    }
                    Provider::Omlx => ProbeTarget::OpenAiModels {
                        endpoint: r.endpoint.clone(),
                        key: crate::extract::llm::resolved_api_key(&r).map(|k| (r.auth_header, k)),
                    },
                }
            }
        };
        let id = s
            .credential
            .clone()
            .unwrap_or_else(|| format!("endpoint:{}", s.endpoint));
        if !plan.iter().any(|(pid, pt)| *pid == id && *pt == target) {
            plan.push((id, target));
        }
    }
    let mut results: HashMap<ProbeTarget, ProbeResult> = HashMap::new();
    for (id, target) in plan {
        let result = match results.get(&target) {
            Some(r) => r.clone(),
            None => {
                let r = run_probe(cfg, &target).await;
                results.insert(target.clone(), r.clone());
                r
            }
        };
        match report.credentials.iter_mut().find(|c| c.id == id) {
            Some(c) => {
                // Keep the worst outcome when one key serves several APIs.
                let already_bad = c.probe.as_ref().is_some_and(|p| p.status != "ok");
                if !already_bad {
                    c.probe = Some(result);
                }
            }
            None => report.credentials.push(CredentialStatus {
                id,
                kind: "endpoint".into(),
                present: true,
                masked: None,
                used_by: Vec::new(),
                warning: None,
                probe: Some(result),
            }),
        }
    }
    let surfaces = report.surfaces.clone();
    for c in report
        .credentials
        .iter_mut()
        .filter(|c| c.kind == "endpoint")
    {
        c.used_by = surfaces
            .iter()
            .filter(|s| s.credential.is_none() && format!("endpoint:{}", s.endpoint) == c.id)
            .map(|s| s.surface.clone())
            .collect();
    }
    report.probed_at = Some(chrono::Utc::now().to_rfc3339());
}

use crate::ops::{IntoCliText, IntoJson, IntoMarkdown, OpsRuntime};
use crate::types::ReinResult;
use clap::Args;
use rein_macros::op;

#[derive(Args, Deserialize, JsonSchema, Debug, Clone, Default)]
pub struct ProvidersParams {
    /// Call each configured provider once to verify the key (5 s timeout each).
    #[serde(default)]
    #[arg(long)]
    pub probe: bool,
}

#[derive(Serialize, Clone, Debug)]
pub struct ProvidersOutput {
    #[serde(flatten)]
    pub report: ProviderStatusReport,
    #[serde(skip)]
    pub json_cli: bool,
}

impl IntoJson for ProvidersOutput {
    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(&self.report).unwrap_or(serde_json::Value::Null)
    }
}

impl IntoMarkdown for ProvidersOutput {
    fn to_markdown(&self) -> String {
        crate::ops::render::render_value_as_markdown(&self.to_json(), 0)
    }
}

fn pad(s: &str, w: usize) -> String {
    let n = s.chars().count();
    if n >= w {
        s.to_string()
    } else {
        format!("{s}{}", " ".repeat(w - n))
    }
}

/// Human table for the CLI (`rein providers`).
pub fn render_human(report: &ProviderStatusReport) -> String {
    let mut out = String::new();
    out.push_str("Surfaces\n");
    out.push_str(&format!(
        "  {} {} {} {} {}\n",
        pad("surface", 28),
        pad("role", 7),
        pad("provider:model", 42),
        pad("source", 28),
        "status"
    ));
    for s in &report.surfaces {
        let pm = if s.provider == "none" {
            "—".to_string()
        } else if s.model.is_empty() {
            s.provider.clone()
        } else {
            format!("{}:{}", s.provider, s.model)
        };
        out.push_str(&format!(
            "  {} {} {} {} {}\n",
            pad(&s.surface, 28),
            pad(&s.role, 7),
            pad(&pm, 42),
            pad(&s.source_table, 28),
            s.status
        ));
    }
    out.push_str("\nKeys\n");
    out.push_str(&format!(
        "  {} {} {} {} {}\n",
        pad("credential", 36),
        pad("source", 13),
        pad("value", 16),
        pad("probe", 22),
        "used by"
    ));
    for c in &report.credentials {
        let name = c.id.split_once(':').map(|(_, n)| n).unwrap_or(&c.id);
        let value = c.masked.clone().unwrap_or_else(|| {
            if c.present {
                "set".into()
            } else {
                "missing".into()
            }
        });
        let probe = match &c.probe {
            Some(p) => match (&p.http_status, &p.detail) {
                (Some(h), Some(d)) => format!("{} ({h}) {d}", p.status),
                (Some(h), None) => format!("{} ({h}, {} ms)", p.status, p.latency_ms),
                (None, Some(d)) => format!("{} {d}", p.status),
                (None, None) => p.status.clone(),
            },
            None => "—".to_string(),
        };
        out.push_str(&format!(
            "  {} {} {} {} {}\n",
            pad(name, 36),
            pad(&c.kind, 13),
            pad(&value, 16),
            pad(&probe, 22),
            c.used_by.join(", ")
        ));
        if let Some(w) = &c.warning {
            out.push_str(&format!("    ! {w}\n"));
        }
    }
    if let Some(vi) = &report.vector_index {
        let mark = if vi.matches {
            "✓"
        } else {
            "✗ run `rein warmup --reembed-all`"
        };
        out.push_str(&format!(
            "\nvector index: stamped {} / configured {} ({} rows) {mark}\n",
            vi.stamped.as_deref().unwrap_or("none"),
            vi.configured,
            vi.rows
        ));
    }
    if let Some(t) = &report.probed_at {
        out.push_str(&format!("probed at {t}\n"));
    }
    out
}

impl IntoCliText for ProvidersOutput {
    fn to_cli_text(&self) -> String {
        if self.json_cli {
            serde_json::to_string_pretty(&self.to_json()).unwrap_or_else(|e| format!("Error: {e}"))
        } else {
            render_human(&self.report)
        }
    }
}

/// Shared by the CLI subcommand (`rein providers`) and `rein keys list`.
pub async fn status_report(config: &ReinConfig, do_probe: bool) -> ProviderStatusReport {
    let mut report = match config.open_store() {
        Ok(store) => collect(config, Some(&store)),
        Err(_) => collect(config, None),
    };
    if do_probe {
        probe(&mut report, config).await;
    }
    report
}

impl OpsRuntime {
    #[op(
        name = "providers",
        category = "diagnostics",
        description = "Provider wiring and credential status: which model each surface resolves to, where that comes from, which API key it needs (masked), optional live probe. Never returns plaintext secrets.",
        rest(method = "GET", path = "/api/providers"),
        auth = "read_token"
    )]
    pub async fn providers(&self, params: ProvidersParams) -> ReinResult<ProvidersOutput> {
        let config = self.config.clone();
        let report = status_report(&config, params.probe).await;
        Ok(ProvidersOutput {
            report,
            json_cli: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ReinConfig;

    /// Tests run in a process that may carry real keys; neutralise the ones
    /// the surfaces consult so assertions are deterministic.
    fn clear_env() {
        for k in [
            "GEMINI_API_KEY",
            "OPENAI_API_KEY",
            "OPENROUTER_API_KEY",
            "SUPERMEMORY_CC_API_KEY",
        ] {
            std::env::remove_var(k);
        }
    }

    #[test]
    fn mask_secret_three_tiers_and_cjk() {
        assert_eq!(mask_secret("AIzaSyA1234567890abcdef3f2a"), "AIza••••3f2a");
        assert_eq!(mask_secret("abcdefghij"), "ab••••ij");
        assert_eq!(mask_secret("abcdefgh"), "ab••••gh");
        assert_eq!(mask_secret("abcdefg"), "••••");
        assert_eq!(mask_secret(""), "••••");
        assert_eq!(
            mask_secret("密钥密钥密钥密钥密钥密钥密钥"),
            "密钥密钥••••密钥密钥"
        );
    }

    #[test]
    fn every_surface_has_a_role_and_the_table_is_complete() {
        assert_eq!(SURFACES.len(), 13);
        let names: Vec<&str> = SURFACES.iter().map(|s| s.name).collect();
        for expected in [
            "embedding",
            "extract",
            "extract.async_memory",
            "extract.intelligent_merge",
            "query_expansion",
            "search.llm_reranker",
            "ars.recall_synthesis",
            "ars.concept_summary",
            "ars.cold_archive",
            "resummerize",
            "ars.llm_judge",
            "ars.llm_judge.nightly_cron",
            "supermemory",
        ] {
            assert!(
                names.contains(&expected),
                "{expected} missing from SURFACES"
            );
        }
        for s in SURFACES {
            assert!(
                ROLES.contains(&s.role),
                "{} has unknown role {}",
                s.name,
                s.role
            );
        }
    }

    #[test]
    fn default_config_without_keys_reports_missing_gemini_credential() {
        clear_env();
        let cfg = ReinConfig::default();
        let report = collect(&cfg, None);
        let embedding = report
            .surfaces
            .iter()
            .find(|s| s.surface == "embedding")
            .unwrap();
        assert_eq!(embedding.role, "embed");
        assert_eq!(embedding.provider, "google");
        assert_eq!(embedding.model, "gemini-embedding-001");
        assert_eq!(embedding.credential.as_deref(), Some("env:GEMINI_API_KEY"));
        assert_eq!(embedding.status, "missing_credential");
        let gem = report
            .credentials
            .iter()
            .find(|c| c.id == "env:GEMINI_API_KEY")
            .unwrap();
        assert!(!gem.present);
        assert!(gem.masked.is_none());
        assert!(gem.used_by.contains(&"embedding".to_string()));
        assert!(gem.used_by.contains(&"extract".to_string()));
    }

    #[test]
    fn secrets_file_key_is_reported_as_secrets_file_and_masked() {
        clear_env();
        let mut cfg = ReinConfig::default();
        cfg.secrets.insert(
            "GEMINI_API_KEY".into(),
            "AIzaSyTESTKEY00000000003f2a".into(),
        );
        let report = collect(&cfg, None);
        let gem = report
            .credentials
            .iter()
            .find(|c| c.id == "env:GEMINI_API_KEY")
            .unwrap();
        assert!(gem.present);
        assert_eq!(gem.kind, "secrets_file");
        assert_eq!(gem.masked.as_deref(), Some("AIza••••3f2a"));
        assert!(gem.warning.is_none());
        let json = serde_json::to_string(&report).unwrap();
        assert!(!json.contains("AIzaSyTESTKEY"), "plaintext leaked: {json}");
    }

    #[test]
    fn config_file_key_is_flagged() {
        clear_env();
        let cfg = ReinConfig::load_from_str(
            "[extract]\nprovider = \"google\"\n[extract.google]\nmodel = \"gemini-3.5-flash-lite\"\napi_key = \"AIzaInFile0000000000000000zz\"\n",
        )
        .unwrap();
        let report = collect(&cfg, None);
        let c = report
            .credentials
            .iter()
            .find(|c| c.id == "config:extract.google.api_key")
            .unwrap();
        assert_eq!(c.kind, "config_file");
        assert!(c.present);
        assert!(c.warning.as_deref().unwrap().contains("rein keys set"));
        let ex = report
            .surfaces
            .iter()
            .find(|s| s.surface == "extract")
            .unwrap();
        assert_eq!(ex.status, "ready");
        assert_eq!(
            ex.credential.as_deref(),
            Some("config:extract.google.api_key")
        );
        assert_eq!(
            config_file_key_value(&cfg, "config:extract.google.api_key").as_deref(),
            Some("AIzaInFile0000000000000000zz")
        );
    }

    #[test]
    fn vendor_surfaces_resolve_their_own_keys_and_sources() {
        clear_env();
        let mut cfg = ReinConfig::load_from_str(
            "[llm]\nprovider = \"google\"\n[llm.google]\nmodel = \"gemini-3.6-flash\"\n[ars.llm_judge]\nenabled = true\nprovider = \"openrouter\"\n[ars.llm_judge.openai]\nmodel = \"anthropic/claude-sonnet-4.5\"\n[embedding]\nprovider = \"omlx\"\ndimensions = 768\n",
        )
        .unwrap();
        cfg.secrets
            .insert("OPENROUTER_API_KEY".into(), "or-key-1234567890".into());
        let report = collect(&cfg, None);
        let judge = report
            .surfaces
            .iter()
            .find(|s| s.surface == "ars.llm_judge")
            .unwrap();
        assert_eq!(judge.provider, "openrouter");
        assert_eq!(judge.source_level, "section_explicit");
        assert_eq!(judge.source_table, "[ars.llm_judge.openai]");
        assert_eq!(judge.credential.as_deref(), Some("env:OPENROUTER_API_KEY"));
        assert_eq!(judge.status, "ready");
        let synth = report
            .surfaces
            .iter()
            .find(|s| s.surface == "ars.recall_synthesis")
            .unwrap();
        assert_eq!(synth.source_table, "[llm.google]");
        // `[ars].llm_backend = "inherit"` walks to `[llm]` at the section-provider level.
        assert_eq!(synth.source_level, "section_provider");
        let emb = report
            .surfaces
            .iter()
            .find(|s| s.surface == "embedding")
            .unwrap();
        assert_eq!(emb.provider, "omlx");
        assert!(emb.credential.is_none());
        assert_eq!(emb.status, "ready");
        assert_eq!(emb.source_table, "[embedding.omlx]");
    }

    #[test]
    fn disabled_surfaces_are_disabled_not_missing() {
        clear_env();
        let cfg = ReinConfig::load_from_str(
            "[search]\nllm_reranker = \"none\"\n[sync]\nsupermemory_enabled = false\n",
        )
        .unwrap();
        let report = collect(&cfg, None);
        let rr = report
            .surfaces
            .iter()
            .find(|s| s.surface == "search.llm_reranker")
            .unwrap();
        assert_eq!(rr.status, "disabled");
        assert!(!rr.enabled);
        let sm = report
            .surfaces
            .iter()
            .find(|s| s.surface == "supermemory")
            .unwrap();
        assert_eq!(sm.status, "disabled");
    }

    #[test]
    fn vector_index_status_matches_doctor_formula() {
        let cfg = {
            let mut c = ReinConfig::default();
            c.database.path = ":memory:".into();
            c
        };
        let store = cfg.open_store().unwrap();
        let report = collect(&cfg, Some(&store));
        let vi = report.vector_index.expect("store given");
        assert_eq!(
            vi.configured,
            format!("{}:{}", cfg.embedding_model(), cfg.embedding.dimensions)
        );
        assert_eq!(vi.rows, 0);
        assert!(vi.matches, "no stamp + zero rows counts as matching");
    }

    #[tokio::test]
    async fn probe_classifies_ok_unauthorized_unreachable() {
        clear_env();
        let ok = crate::stub_http::StubServer::start(vec![(200, "{\"data\":[]}".into())]);
        let unauth =
            crate::stub_http::StubServer::start(vec![(401, "{\"error\":\"nope secret\"}".into())]);
        let mut cfg = ReinConfig::load_from_str(&format!(
            "[extract]\nprovider = \"openai\"\n[extract.openai]\nmodel = \"m\"\nendpoint = \"{}\"\napi_key_env = \"REIN_TEST_PROBE_A\"\n[query_expansion]\nprovider = \"openai\"\n[query_expansion.openai]\nmodel = \"m\"\nendpoint = \"{}\"\napi_key_env = \"REIN_TEST_PROBE_B\"\n[search]\nllm_reranker = \"none\"\n[embedding]\nprovider = \"none\"\n[sync]\nsupermemory_enabled = false\n[async_memory]\nprovider = \"none\"\n",
            ok.url, unauth.url
        ))
        .unwrap();
        cfg.secrets.insert("REIN_TEST_PROBE_A".into(), "ka".into());
        cfg.secrets.insert("REIN_TEST_PROBE_B".into(), "kb".into());
        let mut report = collect(&cfg, None);
        probe(&mut report, &cfg).await;
        assert!(report.probed_at.is_some());
        let a = report
            .credentials
            .iter()
            .find(|c| c.id == "env:REIN_TEST_PROBE_A")
            .unwrap();
        assert_eq!(a.probe.as_ref().unwrap().status, "ok");
        assert_eq!(a.probe.as_ref().unwrap().http_status, Some(200));
        let b = report
            .credentials
            .iter()
            .find(|c| c.id == "env:REIN_TEST_PROBE_B")
            .unwrap();
        assert_eq!(b.probe.as_ref().unwrap().status, "unauthorized");
        let detail = b.probe.as_ref().unwrap().detail.clone().unwrap_or_default();
        assert!(!detail.contains("secret"), "{detail}");
        assert_eq!(ok.requests()[0].path, "/models");
        assert_eq!(ok.requests()[0].header("authorization"), Some("Bearer ka"));
    }

    #[tokio::test]
    async fn probe_embedding_reports_dimension_mismatch() {
        clear_env();
        let body =
            serde_json::json!({"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]}).to_string();
        let stub = crate::stub_http::StubServer::start(vec![(200, body)]);
        let mut cfg = ReinConfig::load_from_str(&format!(
            "[embedding]\nprovider = \"openai\"\ndimensions = 2\n[embedding.openai]\nmodel = \"m\"\nendpoint = \"{}\"\n[extract]\nprovider = \"none\"\n[query_expansion]\nprovider = \"none\"\n[search]\nllm_reranker = \"none\"\n[sync]\nsupermemory_enabled = false\n[async_memory]\nprovider = \"none\"\n",
            stub.url
        ))
        .unwrap();
        cfg.secrets.insert("OPENAI_API_KEY".into(), "k".into());
        let mut report = collect(&cfg, None);
        probe(&mut report, &cfg).await;
        let c = report
            .credentials
            .iter()
            .find(|c| c.id == "env:OPENAI_API_KEY")
            .unwrap();
        let p = c.probe.as_ref().unwrap();
        assert_eq!(p.status, "dimension_mismatch");
        assert_eq!(p.detail.as_deref(), Some("expected 2 dimensions, got 3"));
    }

    #[tokio::test]
    async fn probe_unreachable_endpoint() {
        clear_env();
        let mut cfg = ReinConfig::load_from_str(
            "[extract]\nprovider = \"openai\"\n[extract.openai]\nmodel = \"m\"\nendpoint = \"http://127.0.0.1:9/v1\"\n[query_expansion]\nprovider = \"none\"\n[search]\nllm_reranker = \"none\"\n[embedding]\nprovider = \"none\"\n[sync]\nsupermemory_enabled = false\n[async_memory]\nprovider = \"none\"\n",
        )
        .unwrap();
        cfg.secrets.insert("OPENAI_API_KEY".into(), "k".into());
        let mut report = collect(&cfg, None);
        probe(&mut report, &cfg).await;
        let c = report
            .credentials
            .iter()
            .find(|c| c.id == "env:OPENAI_API_KEY")
            .unwrap();
        assert_eq!(c.probe.as_ref().unwrap().status, "unreachable");
    }

    #[test]
    fn cli_text_lists_surfaces_and_masked_keys_only() {
        clear_env();
        let mut cfg = ReinConfig::default();
        cfg.secrets.insert(
            "GEMINI_API_KEY".into(),
            "AIzaSyTESTKEY00000000003f2a".into(),
        );
        let out = ProvidersOutput {
            report: collect(&cfg, None),
            json_cli: false,
        };
        let text = out.to_cli_text();
        assert!(text.contains("embedding"));
        assert!(text.contains("google:gemini-embedding-001"));
        assert!(text.contains("GEMINI_API_KEY"));
        assert!(text.contains("AIza••••3f2a"));
        assert!(!text.contains("AIzaSyTESTKEY"));
        assert!(text.contains("role"));
        let json = ProvidersOutput {
            report: collect(&cfg, None),
            json_cli: true,
        }
        .to_cli_text();
        assert!(json.contains("\"surfaces\""));
        assert!(!json.contains("AIzaSyTESTKEY"));
    }
}
