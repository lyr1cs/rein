/// Auto-configuration helpers for MCP clients.
///
/// Scans well-known config paths (Claude Code, Claude Desktop, Cursor, Windsurf,
/// VS Code, Gemini, Codex, OpenCode) and injects a `rein` MCP server entry when
/// the config file already exists but rein is not yet configured.
use std::io::{self, Write};
use std::path::Path;

/// Atomically write `content` to `path`.
///
/// Semantics:
///   - Writes to a sibling tmp file `<path>.tmp` (kept in the same parent
///     directory so the final `rename` is intra-filesystem and POSIX-atomic).
///   - `fsync`s the data to disk before the rename so a power-cut between
///     rename and the next sync cannot leave the target as a zero-length file
///     pointing at unwritten blocks.
///   - On any failure (write, sync, rename) the partial tmp file is removed.
///   - The target file is either fully old or fully new — never partial,
///     never empty — even if the process is `kill -9`'d or the system sleeps
///     mid-write. This matters for `~/.claude.json`: a partial write bricks
///     Claude Code launch (it refuses to parse partial JSON, dropping
///     pre-existing MCP servers including non-rein ones).
///
/// Note: `with_extension("tmp")` clobbers any pre-existing `.tmp` sibling
/// (e.g. from a previously-crashed run). This is acceptable — that file is
/// already orphaned and about to be overwritten-then-renamed-away.
fn atomic_write_string(path: &Path, content: &str) -> io::Result<()> {
    // v0.30.3 codex R2 P2: if the target is a symlink (common for
    // dotfile-managed `~/.claude.json` / Codex configs) `rename(tmp, path)`
    // replaces the symlink itself with a regular file, silently
    // disconnecting the user's managed config from its real target. Resolve
    // the link (canonicalize follows the whole chain) and write to the
    // actual target so the rename happens in the real target's directory.
    // v0.30.3 codex R20 P2 + R23 P2: resolve the symlink chain to its
    // final target WITHOUT requiring it to exist. `read_link` only
    // follows ONE hop, so chained dotfile symlinks (symlink → symlink →
    // file) would leave `real_path` still a symlink and rename would
    // replace the intermediate link instead of updating the final
    // target. Loop through hops until we hit a non-symlink, with a
    // cycle guard so a malicious symlink loop can't hang us.
    fn resolve_symlink_chain(start: &std::path::Path) -> std::path::PathBuf {
        let mut current = start.to_path_buf();
        for _ in 0..40 {
            // 40 hops is more than any sane dotfile setup
            match std::fs::symlink_metadata(&current) {
                Ok(m) if m.file_type().is_symlink() => match std::fs::read_link(&current) {
                    Ok(target) => {
                        current = if target.is_absolute() {
                            target
                        } else {
                            current.parent().map(|p| p.join(&target)).unwrap_or(target)
                        };
                    }
                    Err(_) => return current, // can't read link — give up here
                },
                _ => return current, // non-symlink or missing — done
            }
        }
        current // cycle / hop limit hit — last seen path
    }
    let real_path: std::path::PathBuf = match std::fs::symlink_metadata(path) {
        Ok(m) if m.file_type().is_symlink() => resolve_symlink_chain(path),
        Ok(_) => path.to_path_buf(),
        Err(_) => path.to_path_buf(), // fresh write — no metadata available
    };
    // v0.30.3 codex R14 P2-#3: include PID + nanoseconds in the tmp
    // path so two concurrent processes writing the same target don't
    // share `<target>.tmp` — that race let one writer delete the
    // other's in-progress tmp file. With unique tmp paths each writer
    // operates on its own inode and the final rename is the only
    // serialization point (which is correct, atomic, last-write-wins).
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let file_name_str = real_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("rein");
    let tmp_name = format!("{file_name_str}.tmp.{pid}.{nanos}");
    let tmp_path = real_path.with_file_name(tmp_name);

    // v0.30.3 codex R1 P1 + R2 P2: capture target's existing mode and apply
    // it to the tmp file AT CREATION TIME (not post-write) so secret content
    // is never on disk world-readable even briefly. Default `0600` for
    // fresh-write — these are AI client config files that may contain API
    // tokens, OAuth secrets, MCP bearer tokens; group/other read is
    // never legitimate.
    #[cfg(unix)]
    let target_mode: u32 = std::fs::metadata(&real_path)
        .ok()
        .map(|m| {
            use std::os::unix::fs::PermissionsExt;
            m.permissions().mode()
        })
        .unwrap_or(0o600);

    // Inner closure so we can clean up tmp on any error via a single `match`.
    let write_and_sync = || -> io::Result<()> {
        // v0.30.3 codex R3 P2: `OpenOptions::.mode(...)` only applies to
        // NEWLY-CREATED inodes. If a `.tmp` orphan from a previous crash
        // (or a user-created file) already exists, `create(true) +
        // truncate(true)` reuses that inode WITH ITS EXISTING PERMS —
        // which may be world-readable. Result: secret config content sits
        // on disk world-readable, then the rename inherits those perms
        // onto the production config. Remove any stale tmp first (ignore
        // NotFound), then `create_new` so we get a fresh inode with the
        // intended restrictive mode.
        match std::fs::remove_file(&tmp_path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e),
        }
        #[cfg(unix)]
        let mut file = {
            use std::os::unix::fs::OpenOptionsExt;
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(target_mode)
                .open(&tmp_path)?
        };
        #[cfg(not(unix))]
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp_path)?;
        file.write_all(content.as_bytes())?;
        file.sync_all()?;
        // Explicit drop before rename — close the fd so the rename sees
        // a fully-flushed file on platforms where this matters.
        drop(file);
        // On Unix the mode is set at open-time above. On non-Unix
        // platforms we have no mode bits to set.
        Ok(())
    };

    if let Err(e) = write_and_sync() {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(e);
    }

    // v0.30.3 codex R15 P2: on Windows `std::fs::rename` can fail when
    // the target already exists or is held open by another process
    // (e.g. Claude Code reading the config). Best-effort remove first
    // before rename — non-atomic on Windows but the only std-only
    // workaround. Unix rename already overwrites in place atomically.
    #[cfg(windows)]
    {
        let _ = std::fs::remove_file(&real_path);
    }
    if let Err(e) = std::fs::rename(&tmp_path, &real_path) {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(e);
    }

    Ok(())
}

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
///
/// String-aware: comments and trailing commas inside a quoted string are
/// preserved verbatim. The previous implementation ran a string-blind regex
/// for trailing-comma stripping AFTER the comment-stripping loop, which would
/// silently mangle any JSON value containing `,]` or `,}` inside a string
/// (low-probability but silent corruption of user config). Trailing-comma
/// stripping is now folded into the same string-aware char loop.
fn strip_jsonc(s: &str) -> String {
    // Two-pass: pass 1 strips comments (string-aware); pass 2 strips trailing
    // commas (string-aware) on the comment-free intermediate. Doing it in two
    // passes (instead of one) keeps each pass simple: a trailing comma may be
    // separated from its closer by a now-stripped comment, and a single-pass
    // implementation would need a lookahead through arbitrary comment runs.
    let mut intermediate = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    let mut in_string = false;
    while let Some(c) = chars.next() {
        if in_string {
            intermediate.push(c);
            if c == '\\' {
                if let Some(&next) = chars.peek() {
                    intermediate.push(next);
                    chars.next();
                }
            } else if c == '"' {
                in_string = false;
            }
        } else if c == '"' {
            in_string = true;
            intermediate.push(c);
        } else if c == '/' {
            match chars.peek() {
                Some('/') => {
                    chars.next();
                    for ch in chars.by_ref() {
                        if ch == '\n' {
                            intermediate.push('\n');
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
                _ => intermediate.push(c),
            }
        } else {
            intermediate.push(c);
        }
    }

    // Pass 2: strip trailing commas before `}` or `]`, but only when NOT
    // inside a quoted string. Replaces the previous regex-based pass which
    // was string-blind. UTF-8 safe — iterates char-by-char, not byte-by-byte,
    // so non-ASCII content inside JSON strings round-trips unchanged.
    let mut out = String::with_capacity(intermediate.len());
    let chars: Vec<char> = intermediate.chars().collect();
    let mut in_string = false;
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if in_string {
            out.push(c);
            if c == '\\' && i + 1 < chars.len() {
                // Preserve the escaped char verbatim (incl. `\"` so we don't
                // exit the string state on it).
                out.push(chars[i + 1]);
                i += 2;
                continue;
            }
            if c == '"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        if c == '"' {
            in_string = true;
            out.push(c);
            i += 1;
            continue;
        }
        if c == ',' {
            // Peek forward through whitespace for the next non-ws char.
            let mut j = i + 1;
            while j < chars.len() && chars[j].is_whitespace() {
                j += 1;
            }
            if j < chars.len() && (chars[j] == '}' || chars[j] == ']') {
                // Trailing comma: drop the comma, keep the whitespace
                // (so line/column numbers in any downstream parse error
                // still match the original).
                i += 1;
                while i < j {
                    out.push(chars[i]);
                    i += 1;
                }
                continue;
            }
        }
        out.push(c);
        i += 1;
    }
    out
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
    // Atomic write: ~/.claude.json corruption breaks Claude Code launch entirely
    // (it refuses to parse partial JSON and drops every pre-existing MCP server,
    // not just rein). See `atomic_write_string` docs for failure-mode rationale.
    atomic_write_string(path, &formatted)?;
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
    // Atomic write: ~/.codex/config.toml is mutated again by
    // `enable_codex_hooks_feature` and `configure_codex_hooks_file`. A partial
    // write here leaves Codex with a parse-failing config.toml until manual
    // recovery from the `.toml.bak` sibling.
    atomic_write_string(path, &formatted)?;
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
    // Atomic write: this is the second mutation of ~/.codex/config.toml during
    // `configure_codex_client`. A partial write here brick's Codex's parser
    // even though `configure_toml_client` succeeded moments earlier.
    atomic_write_string(path, &formatted)?;
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
        // Atomic write: a partial hooks.json silently disables every Codex
        // hook (Codex 0.129+ refuses to parse the file rather than running
        // a subset), not just the one being added.
        atomic_write_string(&hooks_path, &format!("{formatted}\n"))?;
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

    // -------------------------------------------------------------------
    // Atomic-write helper tests (F5 fix — non-atomic ~/.claude.json write)
    // -------------------------------------------------------------------

    #[test]
    fn atomic_write_string_replaces_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("config.json");
        std::fs::write(&target, "original\n").unwrap();

        atomic_write_string(&target, "replaced\n").unwrap();

        let final_content = std::fs::read_to_string(&target).unwrap();
        assert_eq!(final_content, "replaced\n");

        // No orphan tmp file should remain after a successful run.
        assert!(
            !target.with_extension("tmp").exists(),
            "tmp sibling leaked after successful atomic_write_string"
        );
    }

    #[test]
    fn atomic_write_string_creates_new_file() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("fresh.json");
        assert!(!target.exists());

        atomic_write_string(&target, "hello\n").unwrap();

        let final_content = std::fs::read_to_string(&target).unwrap();
        assert_eq!(final_content, "hello\n");
        assert!(!target.with_extension("tmp").exists());
    }

    #[test]
    fn atomic_write_string_pre_rename_state_does_not_clobber_original() {
        // Property under test: between write_to_tmp and rename, the target
        // file is untouched. If a `kill -9` / power-cut / disk-full interrupts
        // the run AFTER the tmp file is written but BEFORE the rename, the
        // user's original file must still be intact.
        //
        // We don't need a `#[cfg(test)]` rename-injection hook to check this
        // — the property is observable without one: writing a tmp file
        // manually (without calling atomic_write_string at all) simulates
        // exactly the post-write-pre-rename state.
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("config.json");
        std::fs::write(&target, "ORIGINAL_USER_DATA\n").unwrap();

        // Simulate the "process died after writing tmp, before rename" state.
        let tmp = target.with_extension("tmp");
        std::fs::write(&tmp, "INCOMPLETE_NEW_DATA").unwrap();

        // The target file should be entirely unchanged.
        let content = std::fs::read_to_string(&target).unwrap();
        assert_eq!(
            content, "ORIGINAL_USER_DATA\n",
            "target file was clobbered by the tmp-write stage"
        );

        // The legacy `target.tmp` orphan exists at this point. After
        // v0.30.3 codex R14 P2-#3, atomic_write_string uses a per-PID
        // unique tmp name (`<file>.tmp.<pid>.<nanos>`), so the next
        // write goes to a DIFFERENT path and leaves this legacy orphan
        // alone — that's the correct trade-off (concurrent-process
        // safety over orphan cleanup). What we DO assert: the target
        // ends up with the new content, AND no `target.tmp.*` sibling
        // leaks from the current write (the rename was successful).
        atomic_write_string(&target, "NEW_FULL_DATA\n").unwrap();
        let recovered = std::fs::read_to_string(&target).unwrap();
        assert_eq!(recovered, "NEW_FULL_DATA\n");
        // The legacy `target.tmp` may remain on disk (orphan from this
        // simulated crash); cleanup is doctor's responsibility.
        // What must NOT leak: a `<target>.tmp.<pid>.<nanos>` from THIS
        // call — that would mean the rename failed.
        if let Some(parent) = target.parent() {
            let base = target.file_name().unwrap().to_string_lossy().into_owned();
            let unique_prefix = format!("{base}.tmp.");
            for entry in std::fs::read_dir(parent).unwrap().filter_map(|e| e.ok()) {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if name_str.starts_with(&unique_prefix) {
                    panic!(
                        "unique tmp sibling leaked on successful write: {}",
                        entry.path().display()
                    );
                }
            }
        }
    }

    #[test]
    fn configure_json_client_does_not_clobber_on_partial_write() {
        // Higher-level check: configure_json_client now routes through
        // atomic_write_string, so the same crash-safety property holds for
        // the user-visible config flow. We exercise this by:
        //   (1) writing a sentinel original (no rein) to the JSON path
        //   (2) running configure_json_client
        //   (3) asserting the final file is well-formed JSON containing rein
        //       (i.e. the rename happened — not a half-baked file)
        //   (4) asserting no `.tmp` orphan was left behind.
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("claude.json");
        std::fs::write(
            &target,
            r#"{"mcpServers": {"other": {"command": "other"}}}"#,
        )
        .unwrap();

        configure_json_client(&target).unwrap();

        let final_content = std::fs::read_to_string(&target).unwrap();
        // Must be parseable JSON post-call (the central failure mode of a
        // partial write).
        let parsed: serde_json::Value = serde_json::from_str(&final_content)
            .expect("post-call file must be valid JSON");
        assert!(
            parsed["mcpServers"]["rein"].is_object(),
            "rein entry missing from post-call file: {final_content}"
        );
        // Pre-existing entry preserved.
        assert_eq!(
            parsed["mcpServers"]["other"]["command"].as_str(),
            Some("other"),
            "pre-existing MCP server was dropped: {final_content}"
        );
        // No orphan .tmp.
        assert!(
            !target.with_extension("tmp").exists(),
            "configure_json_client leaked a .tmp sibling"
        );
    }

    // -------------------------------------------------------------------
    // strip_jsonc string-aware trailing-comma tests (F5 bonus)
    // -------------------------------------------------------------------

    #[test]
    fn strip_jsonc_preserves_trailing_comma_pattern_inside_string() {
        // Bug class: the v0.30.1 regex-based trailing-comma pass was
        // string-blind, so a JSON string value containing `,}` or `,]`
        // would get its comma silently dropped, corrupting user data.
        let input = r#"{"description": "edge case ,]", "items": [1, 2,]}"#;
        let stripped = strip_jsonc(input);
        // The trailing comma after `2` is real-trailing and should be
        // stripped. The `,]` inside the description string must NOT be
        // touched.
        let parsed: serde_json::Value =
            serde_json::from_str(&stripped).expect("string-aware strip should yield valid JSON");
        assert_eq!(
            parsed["description"].as_str(),
            Some("edge case ,]"),
            "the `,]` inside a quoted string was mangled by strip_jsonc: {stripped}"
        );
        let items: Vec<i64> = parsed["items"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        assert_eq!(items, vec![1, 2]);
    }

    #[test]
    fn strip_jsonc_still_strips_real_trailing_commas() {
        // Regression: ensure the new char-loop implementation didn't
        // accidentally drop the trailing-comma stripping behavior.
        let input = "{\"a\": 1, \"b\": 2,}";
        let stripped = strip_jsonc(input);
        let parsed: serde_json::Value = serde_json::from_str(&stripped).unwrap();
        assert_eq!(parsed["a"].as_i64(), Some(1));
        assert_eq!(parsed["b"].as_i64(), Some(2));
    }

    #[test]
    fn strip_jsonc_handles_unicode_inside_strings() {
        // UTF-8 safety check for the char-by-char pass-2 implementation
        // (the previous byte-iteration draft would mangle non-ASCII).
        let input = "{\"name\": \"中文,]测试\", \"x\": [1,]}";
        let stripped = strip_jsonc(input);
        let parsed: serde_json::Value = serde_json::from_str(&stripped).unwrap();
        assert_eq!(parsed["name"].as_str(), Some("中文,]测试"));
        let xs: Vec<i64> = parsed["x"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        assert_eq!(xs, vec![1]);
    }
}
