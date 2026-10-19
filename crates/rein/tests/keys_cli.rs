use std::process::Command;

fn env_for(dir: &std::path::Path) -> Vec<(String, String)> {
    vec![
        ("HOME".into(), dir.display().to_string()),
        (
            "REIN_CONFIG".into(),
            dir.join("config.toml").display().to_string(),
        ),
        (
            "REIN_SECRETS_FILE".into(),
            dir.join("secrets.env").display().to_string(),
        ),
        (
            "REIN_DB".into(),
            dir.join("memories.db").display().to_string(),
        ),
    ]
}

fn run(dir: &std::path::Path, args: &[&str], stdin: Option<&str>) -> (i32, String, String) {
    use std::io::Write;
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_rein"));
    cmd.args(args).env_clear();
    for (k, v) in env_for(dir) {
        cmd.env(k, v);
    }
    cmd.env("PATH", std::env::var("PATH").unwrap_or_default());
    cmd.stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    let mut child = cmd.spawn().unwrap();
    if let Some(s) = stdin {
        child.stdin.take().unwrap().write_all(s.as_bytes()).unwrap();
    } else {
        drop(child.stdin.take());
    }
    let out = child.wait_with_output().unwrap();
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).into(),
        String::from_utf8_lossy(&out.stderr).into(),
    )
}

#[test]
fn keys_set_list_unset_round_trip_without_plaintext_in_output() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("config.toml"),
        "[extract]\nprovider = \"openai\"\n[extract.openai]\nmodel = \"gpt-5-mini\"\n",
    )
    .unwrap();
    let (code, out, err) = run(
        dir.path(),
        &["keys", "set", "OPENAI_API_KEY", "--stdin"],
        Some("sk-test-1234567890abcd\n"),
    );
    assert_eq!(code, 0, "{err}");
    assert!(out.contains("OPENAI_API_KEY"), "{out}");
    assert!(out.contains("sk-t••••abcd"), "{out}");
    assert!(!out.contains("sk-test-1234567890abcd"));
    assert!(out.contains("used by: extract"), "{out}");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(dir.path().join("secrets.env"))
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o600);
    }
    let (code, out, err) = run(dir.path(), &["keys", "list", "--json"], None);
    assert_eq!(code, 0, "{err}");
    let json: serde_json::Value = serde_json::from_str(&out).unwrap();
    let creds = json["credentials"].as_array().unwrap();
    assert!(
        creds.iter().any(|c| c["id"] == "env:OPENAI_API_KEY"
            && c["kind"] == "secrets_file"
            && c["masked"] == "sk-t••••abcd"),
        "{out}"
    );
    assert!(!out.contains("sk-test-1234567890abcd"));
    let (code, out, _) = run(dir.path(), &["keys", "unset", "OPENAI_API_KEY"], None);
    assert_eq!(code, 0);
    assert!(out.contains("removed"));
    let (code, out, _) = run(dir.path(), &["keys", "unset", "OPENAI_API_KEY"], None);
    assert_eq!(code, 0);
    assert!(out.contains("not set"));
}

#[test]
fn keys_set_without_value_fails_when_stdin_is_not_a_tty() {
    let dir = tempfile::tempdir().unwrap();
    let (code, _, err) = run(dir.path(), &["keys", "set", "OPENAI_API_KEY"], None);
    assert_ne!(code, 0);
    assert!(err.contains("--value") || err.contains("--stdin"), "{err}");
}

#[test]
fn providers_set_writes_config_validates_and_hints() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("config.toml"),
        "# keep me\n[server]\nsse_port = 8680\n",
    )
    .unwrap();
    let (code, out, err) = run(
        dir.path(),
        &[
            "providers",
            "set",
            "embedding",
            "--provider",
            "perplexity",
            "--model",
            "pplx-embed-v1-4b",
            "--dimensions",
            "2560",
        ],
        None,
    );
    assert_eq!(code, 0, "{err}");
    assert!(out.contains("embedding.provider"), "{out}");
    assert!(out.contains("rein warmup --reembed-all"), "{out}");
    assert!(out.contains("rein keys set PERPLEXITY_API_KEY"), "{out}");
    let text = std::fs::read_to_string(dir.path().join("config.toml")).unwrap();
    assert!(text.contains("# keep me"));
    assert!(text.contains("provider = \"perplexity\""));
    // invalid change: nothing written
    let (code, _, err) = run(
        dir.path(),
        &["providers", "set", "embedding", "--provider", "deepseek"],
        None,
    );
    assert_ne!(code, 0);
    assert!(err.contains("no embeddings API"), "{err}");
    let after = std::fs::read_to_string(dir.path().join("config.toml")).unwrap();
    assert_eq!(after, text);
    // dry run: nothing written
    let (code, out, _) = run(
        dir.path(),
        &[
            "providers",
            "set",
            "judge",
            "--provider",
            "openrouter",
            "--model",
            "m",
            "--dry-run",
        ],
        None,
    );
    assert_eq!(code, 0);
    assert!(out.contains("[ars.llm_judge]"), "{out}");
    assert_eq!(
        std::fs::read_to_string(dir.path().join("config.toml")).unwrap(),
        text
    );
}

#[test]
fn providers_show_and_vendors() {
    let dir = tempfile::tempdir().unwrap();
    let (code, out, _) = run(dir.path(), &["providers", "vendors", "--json"], None);
    assert_eq!(code, 0);
    let json: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert!(json
        .as_array()
        .unwrap()
        .iter()
        .any(|v| v["name"] == "perplexity" && v["embeddings"] == true));
    let (code, out, err) = run(dir.path(), &["providers", "show", "embedding"], None);
    assert_eq!(code, 0, "{err}");
    assert!(out.contains("google:gemini-embedding-001"), "{out}");
    let (code, _, err) = run(dir.path(), &["providers", "show", "nope"], None);
    assert_ne!(code, 0);
    assert!(err.contains("unknown surface"), "{err}");
}
