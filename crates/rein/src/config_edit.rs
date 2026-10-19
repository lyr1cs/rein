//! In-place edits of `config.toml` for `rein providers set`. Works on text
//! with `toml_edit` so comments and formatting survive, and validates the
//! result with `ReinConfig::load_from_str` before anyone writes it.

use toml_edit::{DocumentMut, Item, Table};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SurfaceLocation {
    /// Dotted TOML section holding the provider key, e.g. `ars.llm_judge`.
    pub section: &'static str,
    /// Key inside that section: `provider` or `llm_backend` / `llm_reranker`.
    pub provider_key: &'static str,
    /// Whether `[section.google]` / `[section.openai]` tables exist for model etc.
    pub has_tables: bool,
}

pub const SURFACE_NAMES: &[&str] = &[
    "embedding",
    "extract",
    "async_memory",
    "intelligent_merge",
    "query_expansion",
    "reranker",
    "resummerize",
    "ars",
    "judge",
    "llm",
];

pub fn surface_location(surface: &str) -> Result<SurfaceLocation, String> {
    let loc = match surface {
        "embedding" => SurfaceLocation {
            section: "embedding",
            provider_key: "provider",
            has_tables: true,
        },
        "extract" => SurfaceLocation {
            section: "extract",
            provider_key: "provider",
            has_tables: true,
        },
        "async_memory" => SurfaceLocation {
            section: "async_memory",
            provider_key: "provider",
            has_tables: false,
        },
        "intelligent_merge" => SurfaceLocation {
            section: "intelligent_merge",
            provider_key: "provider",
            has_tables: true,
        },
        "query_expansion" => SurfaceLocation {
            section: "query_expansion",
            provider_key: "provider",
            has_tables: true,
        },
        "reranker" => SurfaceLocation {
            section: "search",
            provider_key: "llm_reranker",
            has_tables: false,
        },
        "resummerize" => SurfaceLocation {
            section: "resummerize",
            provider_key: "llm_backend",
            has_tables: false,
        },
        "ars" => SurfaceLocation {
            section: "ars",
            provider_key: "llm_backend",
            has_tables: false,
        },
        "judge" => SurfaceLocation {
            section: "ars.llm_judge",
            provider_key: "provider",
            has_tables: true,
        },
        "llm" => SurfaceLocation {
            section: "llm",
            provider_key: "provider",
            has_tables: true,
        },
        other => {
            return Err(format!(
                "unknown surface '{other}'; expected one of: {}",
                SURFACE_NAMES.join(", ")
            ))
        }
    };
    Ok(loc)
}

#[derive(Debug, Clone, Default)]
pub struct SetProviderRequest {
    pub surface: String,
    pub provider: String,
    pub model: Option<String>,
    pub endpoint: Option<String>,
    pub api_key_env: Option<String>,
    pub auth_header: Option<String>,
    pub disable_thinking: Option<bool>,
    pub dimensions: Option<usize>,
    pub send_dimensions: Option<bool>,
    pub encoding: Option<String>,
    pub query_input_type: Option<String>,
    pub document_input_type: Option<String>,
    pub max_batch: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct FieldChange {
    pub key: String,
    pub old: Option<String>,
    pub new: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SetProviderOutcome {
    pub text: String,
    pub diff: Vec<FieldChange>,
    pub hints: Vec<String>,
}

fn table_at<'a>(doc: &'a mut DocumentMut, dotted: &str) -> &'a mut Table {
    let mut cur: &mut Table = doc.as_table_mut();
    for part in dotted.split('.') {
        if !cur.contains_key(part) {
            let mut t = Table::new();
            t.set_implicit(true);
            cur.insert(part, Item::Table(t));
        }
        cur = cur
            .get_mut(part)
            .and_then(|i| i.as_table_mut())
            .expect("table just inserted");
    }
    cur
}

fn item_to_string(item: Option<&Item>) -> Option<String> {
    let v = item?.as_value()?;
    Some(match v {
        toml_edit::Value::String(s) => s.value().to_string(),
        toml_edit::Value::Integer(i) => i.value().to_string(),
        toml_edit::Value::Boolean(b) => b.value().to_string(),
        other => other.to_string().trim().to_string(),
    })
}

/// Replace a value while keeping the old item's surrounding whitespace and
/// trailing comment (`provider = "google"   # was google` keeps its comment).
fn replace_value(table: &mut Table, key: &str, mut new_value: toml_edit::Value) {
    if let Some(old) = table.get(key).and_then(|i| i.as_value()) {
        *new_value.decor_mut() = old.decor().clone();
    }
    table[key] = Item::Value(new_value);
}

fn set_str(table: &mut Table, key: &str, new: &str, path: &str, diff: &mut Vec<FieldChange>) {
    let old = item_to_string(table.get(key));
    if old.as_deref() != Some(new) {
        replace_value(table, key, toml_edit::Value::from(new));
        diff.push(FieldChange {
            key: path.to_string(),
            old,
            new: new.to_string(),
        });
    }
}

fn set_bool(table: &mut Table, key: &str, new: bool, path: &str, diff: &mut Vec<FieldChange>) {
    let old = item_to_string(table.get(key));
    if old.as_deref() != Some(&new.to_string()) {
        replace_value(table, key, toml_edit::Value::from(new));
        diff.push(FieldChange {
            key: path.to_string(),
            old,
            new: new.to_string(),
        });
    }
}

fn set_int(table: &mut Table, key: &str, new: usize, path: &str, diff: &mut Vec<FieldChange>) {
    let old = item_to_string(table.get(key));
    if old.as_deref() != Some(&new.to_string()) {
        replace_value(table, key, toml_edit::Value::from(new as i64));
        diff.push(FieldChange {
            key: path.to_string(),
            old,
            new: new.to_string(),
        });
    }
}

/// Which provider sub-table a vendor name writes to. Existing `omlx` tables
/// are reused so the file never carries both alias spellings.
fn provider_table_name(section: &Table, provider: &str) -> Result<&'static str, String> {
    match crate::config::Provider::from_str(provider) {
        crate::config::Provider::Google => Ok("google"),
        crate::config::Provider::Omlx => Ok(if section.get("omlx").is_some() {
            "omlx"
        } else {
            "openai"
        }),
        crate::config::Provider::None => Err(format!("invalid provider '{provider}'")),
    }
}

pub fn apply_set_provider(
    current: &str,
    req: &SetProviderRequest,
) -> Result<SetProviderOutcome, String> {
    let loc = surface_location(&req.surface)?;
    let provider = req.provider.trim().to_ascii_lowercase();
    if !matches!(provider.as_str(), "none" | "inherit")
        && crate::config::Provider::from_str(&provider) == crate::config::Provider::None
    {
        return Err(format!(
            "invalid provider '{}'; expected google, none, inherit or a vendor from `rein providers vendors`",
            req.provider
        ));
    }
    let table_fields_given = req.model.is_some()
        || req.endpoint.is_some()
        || req.api_key_env.is_some()
        || req.auth_header.is_some()
        || req.disable_thinking.is_some()
        || req.send_dimensions.is_some()
        || req.encoding.is_some()
        || req.query_input_type.is_some()
        || req.document_input_type.is_some()
        || req.max_batch.is_some();
    if !loc.has_tables && table_fields_given {
        return Err(format!(
            "surface '{}' only selects a provider ([{}].{}); set model/endpoint/key on the surface whose tables it inherits (extract, query_expansion, llm)",
            req.surface, loc.section, loc.provider_key
        ));
    }
    let embedding_only_given = req.dimensions.is_some()
        || req.send_dimensions.is_some()
        || req.encoding.is_some()
        || req.query_input_type.is_some()
        || req.document_input_type.is_some()
        || req.max_batch.is_some();
    if req.surface != "embedding" && embedding_only_given {
        return Err("--dimensions/--send-dimensions/--encoding/--query-input-type/--document-input-type/--max-batch apply to the embedding surface only".to_string());
    }
    if req.surface == "embedding" && req.disable_thinking.is_some() {
        return Err("--disable-thinking applies to LLM surfaces only".to_string());
    }

    let mut doc: DocumentMut = current
        .parse()
        .map_err(|e| format!("current config is not valid TOML: {e}"))?;
    let mut diff = Vec::new();
    let section_path = loc.section;
    {
        let section = table_at(&mut doc, section_path);
        section.set_implicit(false);
        set_str(
            section,
            loc.provider_key,
            &provider,
            &format!("{section_path}.{}", loc.provider_key),
            &mut diff,
        );
        if req.surface == "embedding" {
            if let Some(d) = req.dimensions {
                set_int(section, "dimensions", d, "embedding.dimensions", &mut diff);
            }
        }
        if loc.has_tables && !matches!(provider.as_str(), "none" | "inherit") {
            let sub = provider_table_name(section, &provider)?;
            let sub_path = format!("{section_path}.{sub}");
            if !section.contains_key(sub) {
                section.insert(sub, Item::Table(Table::new()));
            }
            let tbl = section
                .get_mut(sub)
                .and_then(|i| i.as_table_mut())
                .expect("provider table");
            tbl.set_implicit(false);
            if let Some(m) = &req.model {
                set_str(tbl, "model", m, &format!("{sub_path}.model"), &mut diff);
            }
            if let Some(e) = &req.endpoint {
                set_str(
                    tbl,
                    "endpoint",
                    e.trim_end_matches('/'),
                    &format!("{sub_path}.endpoint"),
                    &mut diff,
                );
            }
            if let Some(k) = &req.api_key_env {
                set_str(
                    tbl,
                    "api_key_env",
                    k,
                    &format!("{sub_path}.api_key_env"),
                    &mut diff,
                );
            }
            if let Some(h) = &req.auth_header {
                set_str(
                    tbl,
                    "auth_header",
                    h,
                    &format!("{sub_path}.auth_header"),
                    &mut diff,
                );
            }
            if let Some(b) = req.disable_thinking {
                set_bool(
                    tbl,
                    "disable_thinking",
                    b,
                    &format!("{sub_path}.disable_thinking"),
                    &mut diff,
                );
            }
            if let Some(b) = req.send_dimensions {
                set_bool(
                    tbl,
                    "send_dimensions",
                    b,
                    &format!("{sub_path}.send_dimensions"),
                    &mut diff,
                );
            }
            if let Some(e) = &req.encoding {
                set_str(
                    tbl,
                    "encoding",
                    e,
                    &format!("{sub_path}.encoding"),
                    &mut diff,
                );
            }
            if let Some(q) = &req.query_input_type {
                set_str(
                    tbl,
                    "query_input_type",
                    q,
                    &format!("{sub_path}.query_input_type"),
                    &mut diff,
                );
            }
            if let Some(d) = &req.document_input_type {
                set_str(
                    tbl,
                    "document_input_type",
                    d,
                    &format!("{sub_path}.document_input_type"),
                    &mut diff,
                );
            }
            if let Some(n) = req.max_batch {
                set_int(
                    tbl,
                    "max_batch",
                    n,
                    &format!("{sub_path}.max_batch"),
                    &mut diff,
                );
            }
        }
    }
    let text = doc.to_string();
    // Validate exactly what would be written; surfaces every config rule
    // (capability bits, unknown names, judge provider, …).
    let cfg = crate::config::ReinConfig::load_from_str(&text)
        .map_err(|e| format!("resulting config is invalid, nothing written: {e}"))?;

    let mut hints = Vec::new();
    let embedding_changed = diff.iter().any(|d| d.key.starts_with("embedding."));
    if embedding_changed {
        hints.push("embedding changed: run `rein warmup --reembed-all` (re-embeds every memory under the new model), then restart services".to_string());
    } else if !diff.is_empty() {
        hints.push("restart `com.rein.gui` (launchctl kickstart -k gui/$(id -u)/com.rein.gui) and any host-spawned `rein serve` to apply; the adaptive tick re-reads config each run".to_string());
    }
    if let Some(env_name) = key_env_for_surface(&cfg, &req.surface) {
        if cfg.secret(&env_name).is_none() {
            hints.push(format!(
                "no key found for {env_name}: run `rein keys set {env_name}`"
            ));
        }
    }
    Ok(SetProviderOutcome { text, diff, hints })
}

/// The env var the chosen provider will read for this surface, if any.
fn key_env_for_surface(cfg: &crate::config::ReinConfig, surface: &str) -> Option<String> {
    let report = crate::ops::handlers::providers::collect(cfg, None);
    let name = match surface {
        "embedding" => "embedding",
        "extract" => "extract",
        "async_memory" => "extract.async_memory",
        "intelligent_merge" => "extract.intelligent_merge",
        "query_expansion" => "query_expansion",
        "reranker" => "search.llm_reranker",
        "resummerize" => "resummerize",
        "ars" => "ars.recall_synthesis",
        "judge" => "ars.llm_judge",
        "llm" => "extract",
        _ => return None,
    };
    report
        .surfaces
        .iter()
        .find(|s| s.surface == name)
        .and_then(|s| s.credential.clone())
        .and_then(|id| id.strip_prefix("env:").map(|n| n.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(surface: &str, provider: &str) -> SetProviderRequest {
        SetProviderRequest {
            surface: surface.to_string(),
            provider: provider.to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn surface_names_map_to_toml_locations() {
        assert_eq!(
            surface_location("embedding").unwrap(),
            SurfaceLocation {
                section: "embedding",
                provider_key: "provider",
                has_tables: true
            }
        );
        assert_eq!(
            surface_location("judge").unwrap(),
            SurfaceLocation {
                section: "ars.llm_judge",
                provider_key: "provider",
                has_tables: true
            }
        );
        assert_eq!(
            surface_location("reranker").unwrap(),
            SurfaceLocation {
                section: "search",
                provider_key: "llm_reranker",
                has_tables: false
            }
        );
        assert_eq!(
            surface_location("ars").unwrap(),
            SurfaceLocation {
                section: "ars",
                provider_key: "llm_backend",
                has_tables: false
            }
        );
        assert_eq!(
            surface_location("resummerize").unwrap(),
            SurfaceLocation {
                section: "resummerize",
                provider_key: "llm_backend",
                has_tables: false
            }
        );
        assert_eq!(
            surface_location("async_memory").unwrap(),
            SurfaceLocation {
                section: "async_memory",
                provider_key: "provider",
                has_tables: false
            }
        );
        assert!(surface_location("nope").is_err());
    }

    #[test]
    fn sets_embedding_vendor_preserving_comments_and_other_sections() {
        std::env::remove_var("PERPLEXITY_API_KEY");
        let text = "# my config\n[embedding]\nprovider = \"google\"   # was google\ndimensions = 3072\n\n[embedding.google]\nmodel = \"gemini-embedding-001\"\n\n[server]\nsse_port = 8680\n";
        let mut r = req("embedding", "perplexity");
        r.model = Some("pplx-embed-v1-4b".into());
        r.dimensions = Some(2560);
        let out = apply_set_provider(text, &r).unwrap();
        assert!(out.text.contains("# my config"));
        assert!(out.text.contains("# was google"));
        assert!(out.text.contains("[server]\nsse_port = 8680"));
        let doc: DocumentMut = out.text.parse().unwrap();
        assert_eq!(doc["embedding"]["provider"].as_str(), Some("perplexity"));
        assert_eq!(doc["embedding"]["dimensions"].as_integer(), Some(2560));
        assert_eq!(
            doc["embedding"]["openai"]["model"].as_str(),
            Some("pplx-embed-v1-4b")
        );
        assert_eq!(
            doc["embedding"]["google"]["model"].as_str(),
            Some("gemini-embedding-001")
        );
        assert!(out.diff.iter().any(|d| d.key == "embedding.provider"
            && d.old.as_deref() == Some("google")
            && d.new == "perplexity"));
        assert!(out
            .hints
            .iter()
            .any(|h| h.contains("rein warmup --reembed-all")));
        assert!(out.hints.iter().any(|h| h.contains("PERPLEXITY_API_KEY")));
    }

    #[test]
    fn reuses_existing_omlx_table_name_instead_of_adding_openai() {
        let text = "[extract]\nprovider = \"omlx\"\n[extract.omlx]\nendpoint = \"http://localhost:11434/v1\"\nmodel = \"qwen3\"\n";
        let mut r = req("extract", "openai");
        r.model = Some("gpt-5-mini".into());
        let out = apply_set_provider(text, &r).unwrap();
        let doc: DocumentMut = out.text.parse().unwrap();
        assert_eq!(doc["extract"]["omlx"]["model"].as_str(), Some("gpt-5-mini"));
        assert!(doc["extract"].get("openai").is_none());
        crate::config::ReinConfig::load_from_str(&out.text).unwrap();
    }

    #[test]
    fn judge_and_global_llm_write_their_own_tables() {
        let mut r = req("judge", "openrouter");
        r.model = Some("anthropic/claude-sonnet-4.5".into());
        let out = apply_set_provider("", &r).unwrap();
        let doc: DocumentMut = out.text.parse().unwrap();
        assert_eq!(
            doc["ars"]["llm_judge"]["provider"].as_str(),
            Some("openrouter")
        );
        assert_eq!(
            doc["ars"]["llm_judge"]["openai"]["model"].as_str(),
            Some("anthropic/claude-sonnet-4.5")
        );
        let mut g = req("llm", "google");
        g.model = Some("gemini-3.6-flash".into());
        let out = apply_set_provider(&out.text, &g).unwrap();
        let doc: DocumentMut = out.text.parse().unwrap();
        assert_eq!(doc["llm"]["provider"].as_str(), Some("google"));
        assert_eq!(
            doc["llm"]["google"]["model"].as_str(),
            Some("gemini-3.6-flash")
        );
        assert!(!out.hints.iter().any(|h| h.contains("warmup")));
        assert!(out.hints.iter().any(|h| h.contains("restart")));
    }

    #[test]
    fn provider_only_sections_reject_table_fields() {
        let mut r = req("reranker", "omlx");
        r.model = Some("x".into());
        let err = apply_set_provider("", &r).unwrap_err();
        assert!(err.contains("reranker"), "{err}");
        let out = apply_set_provider("", &req("reranker", "omlx")).unwrap();
        let doc: DocumentMut = out.text.parse().unwrap();
        assert_eq!(doc["search"]["llm_reranker"].as_str(), Some("omlx"));
    }

    #[test]
    fn invalid_results_never_come_back_as_text() {
        let err = apply_set_provider("", &req("embedding", "deepseek")).unwrap_err();
        assert!(err.contains("no embeddings API"), "{err}");
        let err = apply_set_provider("", &req("extract", "voyage")).unwrap_err();
        assert!(err.contains("no chat API"), "{err}");
        let err = apply_set_provider("", &req("extract", "nope")).unwrap_err();
        assert!(err.contains("invalid"), "{err}");
    }

    #[test]
    fn embedding_options_land_in_the_table() {
        std::env::remove_var("VOYAGE_KEY_2");
        let mut r = req("embedding", "voyage");
        r.model = Some("voyage-4".into());
        r.query_input_type = Some("query".into());
        r.document_input_type = Some("document".into());
        r.send_dimensions = Some(true);
        r.max_batch = Some(100);
        r.encoding = Some("float".into());
        r.auth_header = Some("bearer".into());
        r.api_key_env = Some("VOYAGE_KEY_2".into());
        let out = apply_set_provider("", &r).unwrap();
        let t: DocumentMut = out.text.parse().unwrap();
        let tbl = &t["embedding"]["openai"];
        assert_eq!(tbl["query_input_type"].as_str(), Some("query"));
        assert_eq!(tbl["send_dimensions"].as_bool(), Some(true));
        assert_eq!(tbl["max_batch"].as_integer(), Some(100));
        assert_eq!(tbl["api_key_env"].as_str(), Some("VOYAGE_KEY_2"));
        assert!(out.hints.iter().any(|h| h.contains("VOYAGE_KEY_2")));
    }
}
