/// Auto-configuration helpers for MCP clients.
///
/// Scans well-known config paths (Claude Code, Claude Desktop, Cursor, Windsurf,
/// VS Code, Gemini, Codex, OpenCode) and injects a `rein` MCP server entry when
/// the config file already exists but rein is not yet configured.
use std::path::Path;

pub fn auto_configure(dry_run: bool) -> anyhow::Result<()> {
    let home = std::env::var("HOME").unwrap_or_default();
    if home.is_empty() {
        anyhow::bail!("HOME environment variable is not set");
    }

    let clients: Vec<(&str, String, &str)> = vec![
        ("Claude Code", format!("{home}/.claude.json"), "json"),
        (
            "Claude Desktop",
            format!("{home}/Library/Application Support/Claude/claude_desktop_config.json"),
            "json",
        ),
        ("Cursor", format!("{home}/.cursor/mcp.json"), "json"),
        (
            "Windsurf",
            format!("{home}/.codeium/windsurf/mcp_config.json"),
            "json",
        ),
        (
            "VS Code",
            format!("{home}/Library/Application Support/Code/User/mcp.json"),
            "json",
        ),
        ("Gemini", format!("{home}/.gemini/settings.json"), "json"),
        ("Codex", format!("{home}/.codex/config.toml"), "toml"),
        (
            "OpenCode",
            format!("{home}/.config/opencode/opencode.json"),
            "json",
        ),
    ];

    for (name, path, format) in &clients {
        let path = Path::new(path);
        if path.exists() {
            if dry_run {
                println!("[dry-run] Would configure {name} at {}", path.display());
            } else {
                match configure_client(path, format) {
                    Ok(()) => println!("Configured {name}"),
                    Err(e) => println!("Failed to configure {name}: {e}"),
                }
            }
        } else {
            println!("- {name}: not found");
        }
    }
    Ok(())
}

fn configure_client(path: &Path, format: &str) -> anyhow::Result<()> {
    match format {
        "json" => configure_json_client(path),
        "toml" => configure_toml_client(path),
        _ => anyhow::bail!("unsupported config format: {format}"),
    }
}

/// Strip JSONC extensions (// comments, /* */ block comments, trailing commas) to valid JSON.
fn strip_jsonc(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    let mut in_string = false;
    while let Some(c) = chars.next() {
        if in_string {
            out.push(c);
            if c == '\\' {
                if let Some(&next) = chars.peek() {
                    out.push(next);
                    chars.next();
                }
            } else if c == '"' {
                in_string = false;
            }
        } else if c == '"' {
            in_string = true;
            out.push(c);
        } else if c == '/' {
            match chars.peek() {
                Some('/') => {
                    chars.next();
                    for ch in chars.by_ref() {
                        if ch == '\n' {
                            out.push('\n');
                            break;
                        }
                    }
                }
                Some('*') => {
                    chars.next();
                    while let Some(ch) = chars.next() {
                        if ch == '*' && chars.peek() == Some(&'/') {
                            chars.next();
                            break;
                        }
                    }
                }
                _ => out.push(c),
            }
        } else {
            out.push(c);
        }
    }
    // Strip trailing commas before } or ]
    let re = regex::Regex::new(r",\s*([}\]])").unwrap();
    re.replace_all(&out, "$1").into_owned()
}

fn configure_json_client(path: &Path) -> anyhow::Result<()> {
    let content = std::fs::read_to_string(path)?;
    // Create a backup before modifying
    let backup = path.with_extension("json.bak");
    std::fs::copy(path, &backup).ok();
    // Support JSONC (comments + trailing commas) used by VS Code / Cursor
    let cleaned = strip_jsonc(&content);
    let mut root: serde_json::Value = serde_json::from_str(&cleaned)?;

    let servers = root
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("config is not a JSON object"))?
        .entry("mcpServers")
        .or_insert_with(|| serde_json::json!({}));

    let servers_obj = servers
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("mcpServers is not a JSON object"))?;

    if servers_obj.contains_key("rein") {
        println!("  (rein already configured, skipping)");
        return Ok(());
    }

    servers_obj.insert(
        "rein".to_string(),
        serde_json::json!({
            "command": "rein",
            "args": ["serve"]
        }),
    );

    let formatted = serde_json::to_string_pretty(&root)?;
    std::fs::write(path, formatted)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Proxy alias configuration
// ---------------------------------------------------------------------------

fn proxy_url_host(bind: &str) -> String {
    match bind {
        "0.0.0.0" | "::" => "127.0.0.1".to_string(),
        "::1" => "[::1]".to_string(),
        other if other.contains(':') && !other.starts_with('[') => format!("[{other}]"),
        other => other.to_string(),
    }
}

fn codexsub_provider_override(
    provider_key: &str,
    provider_name: &str,
    proxy_url: &str,
    supports_websockets: bool,
) -> String {
    include_str!("../scripts/codexsubp_provider.toml.tmpl")
        .trim()
        .replace("__PROVIDER_KEY__", provider_key)
        .replace("__PROVIDER_NAME__", provider_name)
        .replace("__PROXY_URL__", proxy_url)
        .replace(
            "__SUPPORTS_WEBSOCKETS__",
            if supports_websockets { "true" } else { "false" },
        )
}

fn codexsubp_provider_override(proxy_url: &str) -> String {
    codexsub_provider_override(
        "rein_sub_proxy",
        "Rein Subscription Proxy",
        proxy_url,
        false,
    )
}

fn codexsubpws_provider_override(proxy_url: &str) -> String {
    codexsub_provider_override(
        "rein_sub_proxy_ws",
        "Rein Subscription Proxy WS",
        proxy_url,
        true,
    )
}

fn proxy_aliases(bind: &str, port: u16) -> Vec<(String, String)> {
    let host = proxy_url_host(bind);
    let proxy_url = format!("http://{host}:{port}");
    vec![
        (
            "rein-proxy".to_string(),
            r#"alias rein-proxy="rein serve --proxy &""#.to_string(),
        ),
        (
            "claudep".to_string(),
            // Use a shell function (not alias) so $REIN_PROXY_TOKEN is expanded
            // at invocation time instead of at rc-sourcing time. Otherwise a
            // token set or rotated after the shell starts would be captured as
            // empty/stale in the alias definition, and the proxy would 401.
            format!(
                r#"claudep() {{ REIN_PROXY_ACTIVE=1 ANTHROPIC_BASE_URL=http://{host}:{port} ANTHROPIC_CUSTOM_HEADERS="x-rein-token: ${{REIN_PROXY_TOKEN:-}}" claude "$@"; }}"#
            ),
        ),
        (
            "codexp".to_string(),
            format!(
                r#"codexp() {{ REIN_PROXY_ACTIVE=1 codex -c 'model_providers.rein_proxy={{ name = "Rein Proxy", base_url = "http://{host}:{port}/v1", env_key = "OPENAI_API_KEY", wire_api = "responses", supports_websockets = false, env_http_headers = {{ "x-rein-token" = "REIN_PROXY_TOKEN" }} }}' -c 'model_provider="rein_proxy"' "$@"; }}"#
            ),
        ),
        // v0.20.1 fix: chatgpt_base_url must be `/backend-api` (NOT
        // `/backend-api/codex`). Codex hard-codes a `/codex/` prefix in front
        // of analytics-events (codex-rs/analytics/src/client.rs) and uses a
        // string-contains switch for `wham/apps` vs `api/codex/apps` in the
        // `codex_apps` MCP endpoint (codex-rs/codex-mcp/src/mcp/mod.rs). With
        // `/backend-api/codex` set as base we would double-prefix to
        // `/backend-api/codex/codex/analytics-events/events` → upstream 404,
        // and `/backend-api/codex/wham/apps` → `codex_apps` MCP initialize
        // failure (HTML error body breaks rmcp streamable-http decode).
        //
        // `PathStyle::from_base_url` detects `/backend-api` via a
        // `contains("/backend-api")` check, so dropping the `/codex` suffix
        // still keeps PathStyle::ChatGptApi, which is what cloud-tasks
        // expects. See investigation in docs/backlog/architecture.md #C3.
        (
            "codexsubp".to_string(),
            format!(
                r#"codexsubp() {{ REIN_PROXY_ACTIVE=1 codex -c '{}' -c 'model_provider="rein_sub_proxy"' -c 'chatgpt_base_url="{}/backend-api"' "$@"; }}"#,
                codexsubp_provider_override(&proxy_url),
                proxy_url,
            ),
        ),
        (
            "codexsubpws".to_string(),
            format!(
                r#"codexsubpws() {{ REIN_PROXY_ACTIVE=1 codex -c '{}' -c 'model_provider="rein_sub_proxy_ws"' -c 'chatgpt_base_url="{}/backend-api"' "$@"; }}"#,
                codexsubpws_provider_override(&proxy_url),
                proxy_url,
            ),
        ),
    ]
}

/// Configure shell aliases for rein proxy and clean up Codex config.
pub fn proxy_configure(dry_run: bool) -> anyhow::Result<()> {
    let proxy = crate::config::ReinConfig::load()
        .map(|config| config.proxy)
        .unwrap_or_default();
    let port = proxy.port;
    let bind = proxy.bind;
    let home = std::env::var("HOME").unwrap_or_default();

    println!("\n--- Proxy Configuration ---");

    // Step 1: Configure shell aliases in ~/.zshrc (or ~/.bashrc)
    let shell_rc = if Path::new(&format!("{home}/.zshrc")).exists() {
        format!("{home}/.zshrc")
    } else {
        format!("{home}/.bashrc")
    };
    let rc_path = Path::new(&shell_rc);

    if rc_path.exists() {
        configure_shell_aliases(rc_path, &bind, port, dry_run)?;
    } else {
        println!("  Shell rc not found ({shell_rc}), skipping aliases");
    }

    // Step 2: Clean up Codex config.toml — remove openai_base_url if pointing to rein proxy
    let codex_config = format!("{home}/.codex/config.toml");
    let codex_path = Path::new(&codex_config);
    if codex_path.exists() {
        clean_codex_proxy_config(codex_path, &bind, port, dry_run)?;
    }

    println!("\nProxy setup complete. Usage:");
    println!("  1. rein-proxy        # start proxy in background");
    println!("  2. claudep           # Claude Code via proxy");
    println!("  3. codexp            # Codex CLI via proxy");
    println!("  4. codexsubp         # Codex ChatGPT-login via proxy (loopback)");
    println!("  5. codexsubpws       # Codex ChatGPT-login via proxy (experimental WS-first)");
    println!("  (source your shell rc to apply: source {shell_rc})");

    Ok(())
}

/// Add/update/skip proxy aliases in a shell rc file.
fn configure_shell_aliases(
    rc_path: &Path,
    bind: &str,
    port: u16,
    dry_run: bool,
) -> anyhow::Result<()> {
    let content = std::fs::read_to_string(rc_path)?;
    let mut lines: Vec<String> = content.lines().map(|l| l.to_string()).collect();
    let mut modified = false;
    let aliases = proxy_aliases(bind, port);

    for (name, expected_line) in &aliases {
        let alias_prefix = format!("alias {name}=");
        let fn_prefix = format!("{name}()");

        // Find existing alias or function definition line (handles old alias format too).
        // Note: all rein proxy definitions are single-line (alias or function body on one line).
        // Multi-line function bodies are not supported by this updater; users with multi-line
        // definitions should edit their shell rc manually.
        let existing_idx = lines.iter().position(|l| {
            let trimmed = l.trim();
            !trimmed.starts_with('#')
                && (trimmed.starts_with(&alias_prefix) || trimmed.starts_with(&fn_prefix))
        });

        match existing_idx {
            Some(idx) => {
                if lines[idx].trim() == expected_line {
                    println!("  {name}: already configured");
                } else {
                    if dry_run {
                        println!("  [dry-run] Would update {name} alias");
                    } else {
                        lines[idx] = expected_line.to_string();
                        println!("  {name}: updated");
                    }
                    modified = true;
                }
            }
            None => {
                if dry_run {
                    println!("  [dry-run] Would add {name} alias");
                } else {
                    // Add comment header if this is the first alias we're adding
                    let has_header = lines.iter().any(|l| l.contains("rein proxy aliases"));
                    if !has_header {
                        lines.push(String::new());
                        lines.push("# rein proxy aliases".to_string());
                    }
                    lines.push(expected_line.to_string());
                    println!("  {name}: added");
                }
                modified = true;
            }
        }
    }

    if modified && !dry_run {
        // Backup before writing
        let backup = rc_path.with_extension("bak");
        std::fs::copy(rc_path, &backup).ok();
        // Ensure trailing newline
        let mut output = lines.join("\n");
        if !output.ends_with('\n') {
            output.push('\n');
        }
        std::fs::write(rc_path, output)?;
    }

    Ok(())
}

/// Remove openai_base_url from Codex config.toml if it points to rein proxy.
fn clean_codex_proxy_config(
    path: &Path,
    bind: &str,
    port: u16,
    dry_run: bool,
) -> anyhow::Result<()> {
    let content = std::fs::read_to_string(path)?;
    let host_port = format!("{}:{port}", proxy_url_host(bind));
    let loopback_port = format!("127.0.0.1:{port}");
    let localhost_port = format!("localhost:{port}");

    // Check if openai_base_url exists and points to rein proxy
    let has_proxy_url = content.lines().any(|line| {
        let trimmed = line.trim();
        trimmed.starts_with("openai_base_url")
            && (trimmed.contains(&host_port)
                || trimmed.contains(&loopback_port)
                || trimmed.contains(&localhost_port))
    });

    if !has_proxy_url {
        println!("  Codex config: clean (no proxy base_url)");
        return Ok(());
    }

    if dry_run {
        println!("  [dry-run] Would remove openai_base_url from Codex config");
        return Ok(());
    }

    // Remove the line
    let new_content: String = content
        .lines()
        .filter(|line| {
            let trimmed = line.trim();
            !(trimmed.starts_with("openai_base_url")
                && (trimmed.contains(&host_port)
                    || trimmed.contains(&loopback_port)
                    || trimmed.contains(&localhost_port)))
        })
        .collect::<Vec<_>>()
        .join("\n");

    // Backup and write
    let backup = path.with_extension("toml.proxy-bak");
    std::fs::copy(path, &backup).ok();
    let mut output = new_content;
    if !output.ends_with('\n') {
        output.push('\n');
    }
    std::fs::write(path, output)?;
    println!("  Codex config: removed openai_base_url (use codexp alias instead)");

    Ok(())
}

fn configure_toml_client(path: &Path) -> anyhow::Result<()> {
    let content = std::fs::read_to_string(path)?;
    // Create a backup before modifying
    let backup = path.with_extension("toml.bak");
    std::fs::copy(path, &backup).ok();
    let mut root: toml::Value = if content.is_empty() {
        toml::Value::Table(toml::map::Map::new())
    } else {
        toml::from_str(&content)?
    };

    let root_tbl = root
        .as_table_mut()
        .ok_or_else(|| anyhow::anyhow!("config is not a TOML table"))?;

    // Ensure [mcp] section exists
    let mcp = root_tbl
        .entry("mcp")
        .or_insert_with(|| toml::Value::Table(toml::map::Map::new()));
    let mcp_tbl = mcp
        .as_table_mut()
        .ok_or_else(|| anyhow::anyhow!("[mcp] is not a table"))?;

    if mcp_tbl.contains_key("rein") {
        println!("  (rein already configured, skipping)");
        return Ok(());
    }

    let mut rein_tbl = toml::map::Map::new();
    rein_tbl.insert(
        "command".to_string(),
        toml::Value::String("rein".to_string()),
    );
    rein_tbl.insert(
        "args".to_string(),
        toml::Value::Array(vec![toml::Value::String("serve".to_string())]),
    );
    mcp_tbl.insert("rein".to_string(), toml::Value::Table(rein_tbl));

    let formatted = toml::to_string_pretty(&root)?;
    std::fs::write(path, formatted)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proxy_aliases_include_subscription_login_path() {
        let aliases = proxy_aliases("127.0.0.1", 8690);
        let codexsubp = aliases
            .iter()
            .find(|(name, _)| name == "codexsubp")
            .map(|(_, line)| line)
            .expect("codexsubp alias should exist");

        assert!(codexsubp.contains(r#"model_provider="rein_sub_proxy""#));
        assert!(codexsubp.contains(r#"requires_openai_auth = true"#));
        assert!(codexsubp.contains(r#"supports_websockets = false"#));
        // v0.20.1: MUST be /backend-api (not /backend-api/codex) — see
        // template comment in proxy_aliases for the Codex source-level
        // analysis. Asserting the absence of `/backend-api/codex` here so
        // any regression (re-introducing the suffix) fails loudly.
        assert!(codexsubp.contains(r#"chatgpt_base_url="http://127.0.0.1:8690/backend-api""#));
        assert!(
            !codexsubp.contains("/backend-api/codex"),
            "chatgpt_base_url must NOT include /codex suffix (causes double-prefix 404 on \
             analytics and codex_apps MCP init failure)"
        );

        let codexsubpws = aliases
            .iter()
            .find(|(name, _)| name == "codexsubpws")
            .map(|(_, line)| line)
            .expect("codexsubpws alias should exist");
        assert!(codexsubpws.contains(r#"model_provider="rein_sub_proxy_ws""#));
        assert!(codexsubpws.contains(r#"requires_openai_auth = true"#));
        assert!(codexsubpws.contains(r#"supports_websockets = true"#));
        assert!(codexsubpws.contains(r#"chatgpt_base_url="http://127.0.0.1:8690/backend-api""#));
        assert!(
            !codexsubpws.contains("/backend-api/codex"),
            "chatgpt_base_url must NOT include /codex suffix (causes double-prefix 404 on \
             analytics and codex_apps MCP init failure)"
        );
    }

    #[test]
    fn codexsubp_provider_override_replaces_proxy_url() {
        let override_str = codexsubp_provider_override("http://127.0.0.1:8788");
        assert!(override_str.contains(r#"name = "Rein Subscription Proxy""#));
        assert!(override_str.contains(r#"rein_sub_proxy"#));
        assert!(override_str.contains(r#"base_url = "http://127.0.0.1:8788""#));
        assert!(override_str.contains(r#"requires_openai_auth = true"#));
        assert!(override_str.contains(r#"supports_websockets = false"#));
    }

    #[test]
    fn codexsubpws_provider_override_enables_websockets() {
        let override_str = codexsubpws_provider_override("http://127.0.0.1:8788");
        assert!(override_str.contains(r#"name = "Rein Subscription Proxy WS""#));
        assert!(override_str.contains(r#"rein_sub_proxy_ws"#));
        assert!(override_str.contains(r#"base_url = "http://127.0.0.1:8788""#));
        assert!(override_str.contains(r#"requires_openai_auth = true"#));
        assert!(override_str.contains(r#"supports_websockets = true"#));
    }

    #[test]
    fn smoke_script_uses_shared_codexsubp_template() {
        let script = include_str!("../scripts/smoke_codexsubp.sh");
        assert!(script.contains("codexsubp_provider.toml.tmpl"));
        assert!(script.contains("PROVIDER_OVERRIDE"));
    }

    #[test]
    fn websocket_smoke_script_uses_shared_codexsubp_template() {
        let script = include_str!("../scripts/smoke_codexsubp_ws.sh");
        assert!(script.contains("codexsubp_provider.toml.tmpl"));
        assert!(script.contains("PROVIDER_OVERRIDE"));
        assert!(script.contains("REIN_SUB_PROXY_WS"));
    }

    #[test]
    fn readme_documents_codexsubp_contract() {
        let readme = include_str!("../README.md");
        assert!(readme.contains("scripts/codexsubp_provider.toml.tmpl"));
        assert!(readme.contains("requires_openai_auth = true"));
        assert!(readme.contains("supports_websockets = false"));
        assert!(readme.contains("codexsubpws"));
        assert!(readme.contains("artifact-mirror-only"));
    }

    #[test]
    fn support_matrix_doc_mentions_current_proxy_contract() {
        let doc =
            include_str!("../../../docs/reference/codex-subscription-proxy-support-matrix.md");
        assert!(doc.contains("ArtifactMirrorOnly"));
        assert!(doc.contains("responses_scope_support_matrix"));
        assert!(doc.contains("route_resolution_support_matrix"));
        assert!(doc.contains("proxy_returns_426_when_codex_websocket_upstream_is_unavailable"));
        assert!(doc.contains("codexsubp"));
    }

}
