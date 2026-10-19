//! rein's own key store: a dotenv-subset file (default `~/.rein/secrets.env`,
//! override `REIN_SECRETS_FILE`). Lines are `NAME=value`, `#` comments and
//! blanks; values may be single- or double-quoted; no interpolation. Loaded
//! into `ReinConfig.secrets`; process env always wins (see `ReinConfig::secret`).

use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};

pub fn default_path() -> PathBuf {
    if let Some(p) = std::env::var("REIN_SECRETS_FILE")
        .ok()
        .filter(|s| !s.trim().is_empty())
    {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(home).join(".rein").join("secrets.env")
}

pub fn valid_name(name: &str) -> bool {
    let mut chars = name.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_uppercase())
        && chars.all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
}

/// Missing file → empty map. Group/other permission bits → error (never load
/// a world-readable key file). Bad lines → error naming the line number only.
pub fn load_default() -> Result<BTreeMap<String, String>, String> {
    load_from(&default_path())
}

pub fn load_from(path: &Path) -> Result<BTreeMap<String, String>, String> {
    let meta = match std::fs::metadata(path) {
        Ok(m) => m,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(BTreeMap::new()),
        Err(e) => return Err(format!("{}: {e}", path.display())),
    };
    check_permissions(path, &meta)?;
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    parse(&text).map_err(|line| format!("{}: line {line}: expected NAME=value", path.display()))
}

#[cfg(unix)]
fn check_permissions(path: &Path, meta: &std::fs::Metadata) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    let mode = meta.permissions().mode() & 0o777;
    if mode & 0o077 != 0 {
        return Err(format!(
            "{} is mode {mode:03o}; refusing to load a key file readable by others (chmod 600)",
            path.display()
        ));
    }
    Ok(())
}

#[cfg(not(unix))]
fn check_permissions(_path: &Path, _meta: &std::fs::Metadata) -> Result<(), String> {
    Ok(())
}

/// Returns `Err(line_number)` on the first malformed line.
pub fn parse(text: &str) -> Result<BTreeMap<String, String>, usize> {
    let mut map = BTreeMap::new();
    for (idx, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((name, value)) = line.split_once('=') else {
            return Err(idx + 1);
        };
        let name = name.trim();
        if !valid_name(name) {
            return Err(idx + 1);
        }
        let value = value.trim();
        let value = if value.len() >= 2 && value.starts_with('"') && value.ends_with('"') {
            value[1..value.len() - 1]
                .replace("\\\"", "\"")
                .replace("\\\\", "\\")
        } else if value.len() >= 2 && value.starts_with('\'') && value.ends_with('\'') {
            value[1..value.len() - 1].to_string()
        } else {
            value.to_string()
        };
        map.insert(name.to_string(), value);
    }
    Ok(map)
}

pub fn render(map: &BTreeMap<String, String>) -> String {
    let mut out = String::from("# rein secrets — managed by `rein keys`; mode 600\n");
    for (k, v) in map {
        let needs_quotes = v
            .chars()
            .any(|c| c.is_whitespace() || c == '#' || c == '"' || c == '\\');
        if needs_quotes {
            let escaped = v.replace('\\', "\\\\").replace('"', "\\\"");
            out.push_str(&format!("{k}=\"{escaped}\"\n"));
        } else {
            out.push_str(&format!("{k}={v}\n"));
        }
    }
    out
}

/// Write the map atomically with mode 0600 (directory created 0700 if missing).
pub fn write_to(path: &Path, map: &BTreeMap<String, String>) -> io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700));
        }
    }
    let tmp = path.with_extension(format!("env.tmp-{}", std::process::id()));
    {
        let mut opts = std::fs::OpenOptions::new();
        opts.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        let mut f = opts.open(&tmp)?;
        use std::io::Write;
        f.write_all(render(map).as_bytes())?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_handles_comments_quotes_and_blank_lines() {
        let m = parse("# c\n\nA=1\nB_2=\"two words\"\nC='x#y'\n").unwrap();
        assert_eq!(m["A"], "1");
        assert_eq!(m["B_2"], "two words");
        assert_eq!(m["C"], "x#y");
    }

    #[test]
    fn parse_reports_line_number_without_content() {
        assert_eq!(parse("A=1\nnot a pair\n"), Err(2));
        assert_eq!(parse("lower=1\n"), Err(1));
    }

    #[test]
    fn render_round_trips_through_parse() {
        let mut m = BTreeMap::new();
        m.insert("K".to_string(), "plain".to_string());
        m.insert("Q".to_string(), "has space \"quoted\"".to_string());
        assert_eq!(parse(&render(&m)).unwrap(), m);
    }

    #[cfg(unix)]
    #[test]
    fn write_sets_0600_and_load_refuses_wider_modes() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("secrets.env");
        let mut m = BTreeMap::new();
        m.insert("A".to_string(), "1".to_string());
        write_to(&path, &m).unwrap();
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert_eq!(load_from(&path).unwrap(), m);
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        let err = load_from(&path).unwrap_err();
        assert!(err.contains("chmod 600"), "{err}");
    }

    #[test]
    fn missing_file_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        assert!(load_from(&dir.path().join("nope.env")).unwrap().is_empty());
    }
}
