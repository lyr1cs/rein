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
                let result = if *name == "Codex" {
                    configure_codex_client(path)
                } else {
                    configure_client(path, format)
                };
                match result {
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
    // codex R7 P2: do NOT bake the proxy token into the generated
    // alias. The template uses Codex CLI's `env_http_headers = { ... =
    // "REIN_PROXY_TOKEN" }` form, which expands the env var at
    // invocation time. This means:
    //   - rotating REIN_PROXY_TOKEN immediately takes effect
    //   - secrets never land in the user's shell rc file
    //   - running `rein init --proxy` before setting the token still
    //     produces a working alias (just authenticate later by
    //     setting the env var).
    include_str!("../../../scripts/codexsubp_provider.toml.tmpl")
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
            // codex R9 P2: ensure REIN_PROXY_TOKEN is set before
            // invoking codex. The provider template's
            // `env_http_headers` reads REIN_PROXY_TOKEN exclusively,
            // but the proxy itself ALSO accepts REIN_HTTP_TOKEN as a
            // fallback (`run_proxy`, `check_proxy_auth`, dashboard
            // metrics). Without this fallback, an operator who set
            // only REIN_HTTP_TOKEN gets 401 from the alias even though
            // every other proxy surface works. The `:=` shell
            // assignment exports the resolved token to the codex
            // child process at invocation time, so token rotation
            // takes effect on the next call (no rc re-source needed).
            format!(
                r#"codexsubp() {{ REIN_PROXY_ACTIVE=1 REIN_PROXY_TOKEN="${{REIN_PROXY_TOKEN:-${{REIN_HTTP_TOKEN:-}}}}" codex -c '{}' -c 'model_provider="rein_sub_proxy"' -c 'chatgpt_base_url="{}/backend-api"' "$@"; }}"#,
                codexsubp_provider_override(&proxy_url),
                proxy_url,
            ),
        ),
        (
            "codexsubpws".to_string(),
            format!(
                r#"codexsubpws() {{ REIN_PROXY_ACTIVE=1 REIN_PROXY_TOKEN="${{REIN_PROXY_TOKEN:-${{REIN_HTTP_TOKEN:-}}}}" codex -c '{}' -c 'model_provider="rein_sub_proxy_ws"' -c 'chatgpt_base_url="{}/backend-api"' "$@"; }}"#,
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

    // Codex 0.129+ uses `[mcp_servers.<name>]` for stdio MCP servers; older
    // Codex used `[mcp.<name>]`. (Same `legacy-clean` pass that renamed
    // `[features].codex_hooks` → `[features].hooks`.) Strategy mirrors
    // `enable_codex_hooks_feature`:
    //
    //   - If `[mcp_servers.rein]` is already present, no-op (Codex 0.129+
    //     setup already done — leave the user's customisations alone).
    //   - Otherwise always write `[mcp_servers.rein]` so Codex 0.129+ picks
    //     up rein as an MCP server.
    //   - On fresh inits where `[mcp.rein]` was not previously present, also
    //     write `[mcp.rein]` for Codex <0.129 compat — `rein init` cannot
    //     detect the operator's Codex version, so a fresh install must work
    //     on both.
    //   - Don't touch an existing `[mcp.rein]` entry: if the user customised
    //     it (e.g. different command path or args), respect their choice.
    let has_new = root_tbl
        .get("mcp_servers")
        .and_then(|t| t.as_table())
        .map(|t| t.contains_key("rein"))
        .unwrap_or(false);
    if has_new {
        println!("  (rein already configured in [mcp_servers], skipping)");
        return Ok(());
    }

    // Snapshot any pre-existing `[mcp.rein]` entry so we can clone the
    // user's customisations (custom command path, env, args, cwd, …) into
    // the new `[mcp_servers.rein]` table. Codex 0.129+ ignores `[mcp]`,
    // so without this clone an operator who had customised the legacy
    // entry would silently fall back to rein's defaults after upgrading.
    let legacy_entry = root_tbl
        .get("mcp")
        .and_then(|t| t.as_table())
        .and_then(|t| t.get("rein"))
        .cloned();

    fn rein_entry() -> toml::Value {
        let mut t = toml::map::Map::new();
        t.insert(
            "command".to_string(),
            toml::Value::String("rein".to_string()),
        );
        t.insert(
            "args".to_string(),
            toml::Value::Array(vec![toml::Value::String("serve".to_string())]),
        );
        toml::Value::Table(t)
    }

    let new_entry = legacy_entry.clone().unwrap_or_else(rein_entry);
    let mcp_servers = root_tbl
        .entry("mcp_servers")
        .or_insert_with(|| toml::Value::Table(toml::map::Map::new()));
    let mcp_servers_tbl = mcp_servers
        .as_table_mut()
        .ok_or_else(|| anyhow::anyhow!("[mcp_servers] is not a table"))?;
    mcp_servers_tbl.insert("rein".to_string(), new_entry);

    if legacy_entry.is_none() {
        // Fresh init — also write `[mcp.rein]` (rein defaults) for Codex
        // <0.129 compat. `rein init` cannot detect the operator's Codex
        // version, so a fresh install must work on both.
        let mcp = root_tbl
            .entry("mcp")
            .or_insert_with(|| toml::Value::Table(toml::map::Map::new()));
        let mcp_tbl = mcp
            .as_table_mut()
            .ok_or_else(|| anyhow::anyhow!("[mcp] is not a table"))?;
        mcp_tbl.insert("rein".to_string(), rein_entry());
    }

    let formatted = toml::to_string_pretty(&root)?;
    std::fs::write(path, formatted)?;
    Ok(())
}

fn configure_codex_client(path: &Path) -> anyhow::Result<()> {
    configure_toml_client(path)?;
    enable_codex_hooks_feature(path)?;
    configure_codex_hooks_file(path)?;
    Ok(())
}

fn enable_codex_hooks_feature(path: &Path) -> anyhow::Result<()> {
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
    let features = root_tbl
        .entry("features")
        .or_insert_with(|| toml::Value::Table(toml::map::Map::new()));
    let features_tbl = features
        .as_table_mut()
        .ok_or_else(|| anyhow::anyhow!("[features] is not a table"))?;
    // Codex 0.129+ uses `[features].hooks`. Pre-0.129 used `codex_hooks`.
    // Strategy across the rename window:
    //
    //   - If `hooks = true` is already set, no-op (user is on Codex 0.129+
    //     and already configured).
    //   - Otherwise always set `hooks = true` so Codex 0.129+ picks up hooks.
    //   - On fresh inits where `codex_hooks` was not previously present, also
    //     set `codex_hooks = true` so users still on Codex <0.129 get hooks
    //     enabled — `rein doctor` (which accepts either key) cannot
    //     distinguish the two cases, so a fresh install must work on both.
    //   - Don't touch an existing `codex_hooks` entry: if the user explicitly
    //     set it (true or false), respect their choice.
    if features_tbl.get("hooks").and_then(|v| v.as_bool()) == Some(true) {
        return Ok(());
    }

    features_tbl.insert("hooks".to_string(), toml::Value::Boolean(true));
    if !features_tbl.contains_key("codex_hooks") {
        features_tbl.insert("codex_hooks".to_string(), toml::Value::Boolean(true));
    }
    let formatted = toml::to_string_pretty(&root)?;
    std::fs::write(path, formatted)?;
    Ok(())
}

fn configure_codex_hooks_file(config_path: &Path) -> anyhow::Result<()> {
    let Some(config_dir) = config_path.parent() else {
        anyhow::bail!("Codex config path has no parent directory");
    };
    let hooks_path = config_dir.join("hooks.json");
    let mut root = if hooks_path.exists() {
        let content = std::fs::read_to_string(&hooks_path)?;
        serde_json::from_str::<serde_json::Value>(&content)?
    } else {
        serde_json::json!({})
    };

    let root_obj = root
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("Codex hooks file is not a JSON object"))?;
    let hooks = root_obj
        .entry("hooks")
        .or_insert_with(|| serde_json::json!({}));
    let hooks_obj = hooks
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("Codex hooks section is not a JSON object"))?;

    let mut modified = false;
    modified |= ensure_codex_hook(
        hooks_obj,
        "SessionStart",
        Some("*"),
        "REIN_AGENT_LABEL=codex rein hook session-start",
        5,
        Some("Loading Rein project context"),
    );
    modified |= ensure_codex_hook(
        hooks_obj,
        "PreToolUse",
        Some("*"),
        "REIN_AGENT_LABEL=codex rein hook pre",
        5,
        None,
    );
    modified |= ensure_codex_hook(
        hooks_obj,
        "PermissionRequest",
        Some("*"),
        "REIN_AGENT_LABEL=codex rein hook permission",
        5,
        None,
    );
    modified |= ensure_codex_hook(
        hooks_obj,
        "PostToolUse",
        Some("*"),
        "REIN_AGENT_LABEL=codex rein hook post",
        10,
        Some("Recording tool output in Rein"),
    );
    modified |= ensure_codex_hook(
        hooks_obj,
        "UserPromptSubmit",
        None,
        "REIN_AGENT_LABEL=codex rein hook prompt",
        5,
        None,
    );
    modified |= ensure_codex_hook(
        hooks_obj,
        "Stop",
        None,
        "REIN_AGENT_LABEL=codex rein hook stop",
        30,
        Some("Summarizing Codex session in Rein"),
    );

    if modified {
        if hooks_path.exists() {
            let backup = hooks_path.with_extension("json.bak");
            std::fs::copy(&hooks_path, &backup).ok();
        }
        let formatted = serde_json::to_string_pretty(&root)?;
        std::fs::write(hooks_path, format!("{formatted}\n"))?;
    }
    Ok(())
}

fn ensure_codex_hook(
    hooks_obj: &mut serde_json::Map<String, serde_json::Value>,
    event: &str,
    matcher: Option<&str>,
    command: &str,
    timeout: u64,
    status_message: Option<&str>,
) -> bool {
    if event_has_command(hooks_obj.get(event), command) {
        return false;
    }

    let mut handler = serde_json::json!({
        "type": "command",
        "command": command,
        "timeout": timeout
    });
    if let Some(message) = status_message {
        handler["statusMessage"] = serde_json::Value::String(message.to_string());
    }

    let mut group = serde_json::json!({
        "hooks": [handler]
    });
    if let Some(matcher) = matcher {
        group["matcher"] = serde_json::Value::String(matcher.to_string());
    }

    let entry = hooks_obj
        .entry(event.to_string())
        .or_insert_with(|| serde_json::json!([]));
    if let Some(groups) = entry.as_array_mut() {
        groups.push(group);
    } else {
        *entry = serde_json::json!([group]);
    }
    true
}

fn event_has_command(event_hooks: Option<&serde_json::Value>, command: &str) -> bool {
    event_hooks
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
        .filter_map(|group| group.get("hooks").and_then(|v| v.as_array()))
        .flatten()
        .any(|handler| handler.get("command").and_then(|v| v.as_str()) == Some(command))
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
        let script = include_str!("../../../scripts/smoke_codexsubp.sh");
        assert!(script.contains("codexsubp_provider.toml.tmpl"));
        assert!(script.contains("PROVIDER_OVERRIDE"));
    }

    #[test]
    fn websocket_smoke_script_uses_shared_codexsubp_template() {
        let script = include_str!("../../../scripts/smoke_codexsubp_ws.sh");
        assert!(script.contains("codexsubp_provider.toml.tmpl"));
        assert!(script.contains("PROVIDER_OVERRIDE"));
        assert!(script.contains("REIN_SUB_PROXY_WS"));
    }

    #[test]
    fn readme_documents_codexsubp_contract() {
        let readme = include_str!("../../../README.md");
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

    #[test]
    fn generic_toml_configuration_only_installs_mcp() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.toml");
        std::fs::write(&config_path, "").unwrap();

        configure_toml_client(&config_path).unwrap();

        let config = std::fs::read_to_string(&config_path).unwrap();
        // Fresh init writes BOTH `[mcp_servers.rein]` (Codex 0.129+) and
        // `[mcp.rein]` (Codex <0.129) since `configure_toml_client` cannot
        // detect the operator's Codex version.
        assert!(
            config.contains("[mcp_servers.rein]"),
            "missing new mcp_servers entry: {config}"
        );
        assert!(
            config.contains("[mcp.rein]"),
            "missing legacy mcp entry: {config}"
        );
        // No hooks feature flag for the generic (non-Codex) path.
        assert!(!config.contains("codex_hooks"));
        assert!(!config.contains("hooks = true"));
        assert!(!dir.path().join("hooks.json").exists());
    }

    #[test]
    fn codex_init_writes_new_mcp_servers_and_preserves_legacy_mcp() {
        // Regression for the codex 0.129 [mcp.<name>] → [mcp_servers.<name>]
        // rename. When a user already has `[mcp.rein]` from an older codex
        // setup, `rein init` should add `[mcp_servers.rein]` (so codex 0.129+
        // actually discovers rein) without removing the legacy entry.
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.toml");
        std::fs::write(
            &config_path,
            "[mcp.rein]\ncommand = \"rein\"\nargs = [\"serve\"]\n",
        )
        .unwrap();

        configure_codex_client(&config_path).unwrap();

        let config = std::fs::read_to_string(&config_path).unwrap();
        assert!(
            config.contains("[mcp_servers.rein]"),
            "new mcp_servers entry missing: {config}"
        );
        assert!(
            config.contains("[mcp.rein]"),
            "legacy mcp entry was removed: {config}"
        );
    }

    #[test]
    fn codex_init_skips_when_mcp_servers_rein_already_present() {
        // If `[mcp_servers.rein]` is already there (Codex 0.129+ user is
        // already configured), init must not duplicate the entry and must
        // not retroactively add `[mcp.rein]`.
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.toml");
        std::fs::write(
            &config_path,
            "[mcp_servers.rein]\ncommand = \"rein\"\nargs = [\"serve\"]\n",
        )
        .unwrap();

        configure_codex_client(&config_path).unwrap();

        let config = std::fs::read_to_string(&config_path).unwrap();
        assert_eq!(
            config.matches("[mcp_servers.rein]").count(),
            1,
            "mcp_servers.rein duplicated: {config}"
        );
        assert!(
            !config.contains("[mcp.rein]"),
            "legacy mcp.rein retroactively added: {config}"
        );
    }

    #[test]
    fn codex_init_clones_user_customised_legacy_mcp_into_mcp_servers() {
        // If a user has customised the legacy `[mcp.rein]` entry (e.g. with
        // a non-default command path or extra args/env), `rein init` must
        // CLONE those customisations into the new `[mcp_servers.rein]`
        // entry, not silently fall back to rein's defaults. Codex 0.129+
        // ignores `[mcp]`, so a default `[mcp_servers.rein]` would replace
        // the user's working setup.
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.toml");
        std::fs::write(
            &config_path,
            "[mcp.rein]\ncommand = \"/opt/custom/rein\"\nargs = [\"serve\", \"--quiet\"]\ncwd = \"/opt/custom\"\n\n[mcp.rein.env]\nREIN_DB = \"/opt/custom/memories.db\"\nREIN_AGENT_LABEL = \"codex\"\n",
        )
        .unwrap();

        configure_codex_client(&config_path).unwrap();

        let config = std::fs::read_to_string(&config_path).unwrap();
        let parsed: toml::Value = toml::from_str(&config).unwrap();

        // Legacy `[mcp.rein]` preserved verbatim.
        assert_eq!(
            parsed["mcp"]["rein"]["command"].as_str(),
            Some("/opt/custom/rein"),
            "legacy command was overwritten: {config}"
        );

        // New `[mcp_servers.rein]` cloned the user's customisations.
        assert_eq!(
            parsed["mcp_servers"]["rein"]["command"].as_str(),
            Some("/opt/custom/rein"),
            "new entry did not clone user's command: {config}"
        );
        let new_args: Vec<&str> = parsed["mcp_servers"]["rein"]["args"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(
            new_args,
            vec!["serve", "--quiet"],
            "new entry did not clone user's args: {config}"
        );
        assert_eq!(
            parsed["mcp_servers"]["rein"]["cwd"].as_str(),
            Some("/opt/custom"),
            "new entry did not clone user's cwd: {config}"
        );
        assert_eq!(
            parsed["mcp_servers"]["rein"]["env"]["REIN_DB"].as_str(),
            Some("/opt/custom/memories.db"),
            "new entry did not clone user's REIN_DB env: {config}"
        );
        assert_eq!(
            parsed["mcp_servers"]["rein"]["env"]["REIN_AGENT_LABEL"].as_str(),
            Some("codex"),
            "new entry did not clone user's env table: {config}"
        );
    }

    #[test]
    fn codex_configuration_installs_rein_hooks() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.toml");
        std::fs::write(&config_path, "").unwrap();

        configure_codex_client(&config_path).unwrap();

        let config = std::fs::read_to_string(&config_path).unwrap();
        // Fresh init must enable hooks on BOTH Codex 0.129+ (`hooks`) and
        // pre-0.129 (`codex_hooks`) since `rein doctor` accepts either key
        // and would otherwise falsely report healthy on old Codex.
        assert!(config.contains("hooks = true"), "config: {config}");
        assert!(config.contains("codex_hooks = true"), "config: {config}");

        let hooks_path = dir.path().join("hooks.json");
        let hooks = std::fs::read_to_string(&hooks_path).unwrap();
        let root: serde_json::Value = serde_json::from_str(&hooks).unwrap();

        let expected = [
            (
                "SessionStart",
                "REIN_AGENT_LABEL=codex rein hook session-start",
            ),
            ("PreToolUse", "REIN_AGENT_LABEL=codex rein hook pre"),
            (
                "PermissionRequest",
                "REIN_AGENT_LABEL=codex rein hook permission",
            ),
            ("PostToolUse", "REIN_AGENT_LABEL=codex rein hook post"),
            (
                "UserPromptSubmit",
                "REIN_AGENT_LABEL=codex rein hook prompt",
            ),
            ("Stop", "REIN_AGENT_LABEL=codex rein hook stop"),
        ];
        for (event, command) in expected {
            assert_eq!(root["hooks"][event][0]["hooks"][0]["command"], command);
        }
    }

    #[test]
    fn codex_init_writes_new_hooks_key_and_preserves_legacy_codex_hooks() {
        // Regression for the codex 0.129 [features].codex_hooks → hooks rename.
        // When a user already has `codex_hooks = true` from an older codex
        // setup, `rein init` should add the new `hooks = true` (so codex 0.129+
        // actually picks up hooks) without removing the legacy key (in case
        // they downgrade or share the config across machines).
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.toml");
        std::fs::write(&config_path, "[features]\ncodex_hooks = true\n").unwrap();

        configure_codex_client(&config_path).unwrap();

        let config = std::fs::read_to_string(&config_path).unwrap();
        assert!(
            config.contains("hooks = true"),
            "new hooks key missing: {config}"
        );
        assert!(
            config.contains("codex_hooks = true"),
            "legacy codex_hooks key was removed: {config}"
        );
    }

    #[test]
    fn codex_init_is_idempotent_under_new_hooks_key() {
        // If `hooks = true` is already present, init should be a no-op for the
        // feature flag — including not adding the legacy `codex_hooks` key.
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.toml");
        std::fs::write(&config_path, "[features]\nhooks = true\n").unwrap();

        configure_codex_client(&config_path).unwrap();

        let config = std::fs::read_to_string(&config_path).unwrap();
        // Exactly one occurrence — we did not duplicate the line.
        assert_eq!(
            config.matches("hooks = true").count(),
            1,
            "hooks flag was duplicated: {config}"
        );
        assert!(!config.contains("codex_hooks"));
    }

    #[test]
    fn codex_init_respects_explicitly_disabled_legacy_key() {
        // If the user has explicitly set `codex_hooks = false`, init must
        // still write `hooks = true` (Codex 0.129+) without flipping the
        // explicit opt-out to true.
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.toml");
        std::fs::write(&config_path, "[features]\ncodex_hooks = false\n").unwrap();

        configure_codex_client(&config_path).unwrap();

        let config = std::fs::read_to_string(&config_path).unwrap();
        assert!(config.contains("hooks = true"), "config: {config}");
        // Explicit user opt-out preserved.
        assert!(
            config.contains("codex_hooks = false"),
            "explicit codex_hooks=false was overwritten: {config}"
        );
        assert!(
            !config.contains("codex_hooks = true"),
            "explicit codex_hooks=false was overwritten: {config}"
        );
    }
}
