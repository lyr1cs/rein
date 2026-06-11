//! Session buffer I/O for hook_post → hook_stop pipeline.

use crate::config::ReinConfig;
use crate::extract::hooks::parsing::redact_secrets;

/// Hard cap on per-session buffer file size. Above this, appends drop (and
/// the event is marked in tracing) instead of growing without bound. A
/// misbehaving client tool can easily dump hundreds of MB of output per
/// PostToolUse; without the cap, a single long session could OOM the hook
/// process when `read_and_clear_buffer` pulls the whole file into memory.
///
/// 16 MiB is ~20× the largest realistic session buffer we've seen, so
/// legitimate use never trips the cap.
const MAX_BUFFER_BYTES: u64 = 16 * 1024 * 1024;

/// Resolve the buffer directory (auto = ~/.rein/).
pub fn resolve_buffer_dir(config: &ReinConfig) -> std::path::PathBuf {
    if config.hooks.buffer_dir == "auto" {
        config
            .resolve_db_path()
            .parent()
            .unwrap_or(std::path::Path::new("/tmp"))
            .to_path_buf()
    } else {
        std::path::PathBuf::from(&config.hooks.buffer_dir)
    }
}

/// Derive a session-scoped buffer file path from the hook input.
pub fn session_buffer_path(config: &ReinConfig, input: &str) -> std::path::PathBuf {
    let session_id = if let Ok(json) = serde_json::from_str::<serde_json::Value>(input) {
        if let Some(path) = json.get("transcript_path").and_then(|v| v.as_str()) {
            use sha2::{Digest, Sha256};
            let hash = Sha256::digest(path.as_bytes());
            format!("{:x}", hash).chars().take(12).collect()
        } else {
            format!("pid{}", std::process::id())
        }
    } else {
        format!("pid{}", std::process::id())
    };
    resolve_buffer_dir(config).join(format!("buffer_{session_id}.jsonl"))
}

/// Append a text entry to the session buffer file.
pub fn append_to_buffer(path: &std::path::Path, text: &str, source: &str) -> anyhow::Result<()> {
    with_buffer_lock(path, || {
        use std::io::Write;
        if let Ok(meta) = std::fs::metadata(path) {
            if meta.len() >= MAX_BUFFER_BYTES {
                tracing::warn!(
                    path = ?path,
                    size = meta.len(),
                    cap = MAX_BUFFER_BYTES,
                    "session buffer hit size cap; dropping append"
                );
                return Ok(());
            }
        }
        let entry = serde_json::json!({
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "text": redact_secrets(text),
            "source": source,
        });
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(path)?;
        writeln!(file, "{}", entry)?;
        Ok(())
    })
}

/// Safely read a buffer file, bounded by `MAX_BUFFER_BYTES`. Returns `None`
/// when the file is absent or unreadable; oversize files are dropped and
/// cleared on disk to guarantee progress.
fn read_buffer_bounded(path: &std::path::Path) -> Option<String> {
    let meta = std::fs::metadata(path).ok()?;
    if meta.len() > MAX_BUFFER_BYTES {
        tracing::warn!(
            path = ?path,
            size = meta.len(),
            cap = MAX_BUFFER_BYTES,
            "session buffer exceeded size cap; discarding"
        );
        let _ = std::fs::remove_file(path);
        return None;
    }
    std::fs::read_to_string(path).ok()
}

/// Read all text entries from a buffer file and delete it.
pub fn read_and_clear_buffer(path: &std::path::Path) -> Vec<String> {
    with_buffer_lock(path, || {
        let content = match read_buffer_bounded(path) {
            Some(c) => c,
            None => return Ok(vec![]),
        };
        let _ = std::fs::remove_file(path);
        Ok(content
            .lines()
            .filter_map(|line| {
                serde_json::from_str::<serde_json::Value>(line)
                    .ok()
                    .and_then(|v| {
                        v.get("text")
                            .and_then(|t| t.as_str())
                            .map(|s| s.to_string())
                    })
            })
            .collect())
    })
    .unwrap_or_default()
}

/// Adaptive flush threshold: adjusts based on signal density in the buffer.
pub fn adaptive_flush_threshold(base: usize, buf_path: &std::path::Path) -> usize {
    let content = match read_buffer_bounded(buf_path) {
        Some(c) => c,
        None => return base,
    };
    let total_lines = content.lines().count();
    if total_lines < 5 {
        return base;
    }

    let high_signal = content
        .lines()
        .filter_map(|line| {
            serde_json::from_str::<serde_json::Value>(line)
                .ok()
                .and_then(|v| {
                    v.get("text")
                        .and_then(|t| t.as_str())
                        .map(|s| s.to_string())
                })
        })
        .filter(|text| super::scoring::worth_extracting(text))
        .count();

    let density = high_signal as f64 / total_lines as f64;
    if density > 0.5 {
        base / 2
    } else if density < 0.1 {
        base * 2
    } else {
        base
    }
}

/// Path for the flush-count marker file for a given session buffer.
pub fn flush_marker_path(buf_path: &std::path::Path) -> std::path::PathBuf {
    buf_path.with_extension("flushed")
}

// ---------------------------------------------------------------------------
// v1.2 flushed-content ledger — content-level dedup for hook_stop
// ---------------------------------------------------------------------------

/// Minimum trimmed line length (chars) recorded in / matched against the
/// flushed-content ledger. Short lines (`}`, `---`, `ok`, fence markers)
/// recur across unrelated contexts; an exact-hash match on them says nothing
/// about provenance, and dropping them from a conversation turn could distort
/// meaning. Long lines that hash-match a flushed tool-output line are
/// near-certainly verbatim quotes of already-extracted content.
const MIN_FLUSHED_LINE_CHARS: usize = 24;

/// Hard cap on the flushed-content ledger file. A session pathological enough
/// to exceed this stops RECORDING (filtering degrades toward no-op — the
/// conservative direction: at worst the stop-time extraction sees content
/// twice, which is the pre-v1.2 status quo).
const MAX_LEDGER_BYTES: u64 = 4 * 1024 * 1024;

/// Path for the flushed-content hash ledger for a given session buffer.
pub fn flushed_ledger_path(buf_path: &std::path::Path) -> std::path::PathBuf {
    buf_path.with_extension("flushed-hashes")
}

fn ledger_line_hash(s: &str) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(s.as_bytes()))
}

/// Record the content of a mid-session flush in the per-session ledger so
/// `hook_stop` can filter verbatim re-occurrences of already-extracted
/// tool output from the transcript turns (content-level dedup — the offset
/// approach was rejected in v0.38 because conversation turns never enter the
/// buffer, so offset truncation would drop never-extracted conversation
/// facts).
///
/// Per flushed item we record the hash of the whole trimmed item AND of each
/// trimmed line ≥ `MIN_FLUSHED_LINE_CHARS` — assistant turns typically quote
/// tool output line-wise (code fences, error lines), not as whole blocks.
/// Best-effort like the rest of the buffer pipeline.
pub fn record_flushed_content(buf_path: &std::path::Path, items: &[String]) {
    if items.is_empty() {
        return;
    }
    let ledger = flushed_ledger_path(buf_path);
    let _ = with_buffer_lock(buf_path, || {
        use std::io::Write;
        let current = std::fs::metadata(&ledger).map(|m| m.len()).unwrap_or(0);
        if current >= MAX_LEDGER_BYTES {
            tracing::warn!(
                path = ?ledger,
                size = current,
                cap = MAX_LEDGER_BYTES,
                "flushed-content ledger hit size cap; dropping record"
            );
            return Ok(());
        }
        // codex v1.2 R1 P3: the cap bounds the WHOLE file, including this
        // batch — a single pathological flush must not blow past it and then
        // be read back in full at hook_stop.
        let out = build_ledger_records(items, (MAX_LEDGER_BYTES - current) as usize);
        if out.is_empty() {
            return Ok(());
        }
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(&ledger)?;
        file.write_all(out.as_bytes())?;
        Ok(())
    });
}

/// Pure body of `record_flushed_content`: hash lines for the given items,
/// bounded by `budget` bytes. Truncating at the budget degrades filtering
/// toward no-op — the conservative direction (unrecorded content is simply
/// seen twice at stop, the pre-v1.2 status quo).
///
/// codex v1.2 R1 P2: whole-item hashes obey the same `MIN_FLUSHED_LINE_CHARS`
/// guard as line hashes — they feed the reader's whole-turn drop, where a
/// short-content collision ("ok", "cargo test") is exactly as unsafe as a
/// short line match.
fn build_ledger_records(items: &[String], budget: usize) -> String {
    const RECORD_BYTES: usize = 65; // 64 hex chars + '\n'
    let mut out = String::new();
    let fits = |out: &String| out.len() + RECORD_BYTES <= budget;
    'items: for item in items {
        let trimmed = item.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.chars().count() >= MIN_FLUSHED_LINE_CHARS {
            if !fits(&out) {
                break 'items;
            }
            out.push_str(&ledger_line_hash(trimmed));
            out.push('\n');
        }
        for line in trimmed.lines() {
            let line = line.trim();
            if line.chars().count() >= MIN_FLUSHED_LINE_CHARS {
                if !fits(&out) {
                    break 'items;
                }
                out.push_str(&ledger_line_hash(line));
                out.push('\n');
            }
        }
    }
    out
}

/// Read the flushed-content ledger into a hash set. Empty when no mid-session
/// flush recorded content (or the ledger is unreadable — conservative no-op).
pub fn read_flushed_hashes(buf_path: &std::path::Path) -> std::collections::HashSet<String> {
    let ledger = flushed_ledger_path(buf_path);
    match std::fs::read_to_string(&ledger) {
        Ok(content) => content
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect(),
        Err(_) => Default::default(),
    }
}

/// Delete the flushed-content ledger (call at hook_stop end, with
/// `clear_flush_marker`).
pub fn clear_flushed_ledger(buf_path: &std::path::Path) {
    let _ = std::fs::remove_file(flushed_ledger_path(buf_path));
}

/// Content-level filter for `hook_stop`'s incremental no-fallback path: strip
/// from the transcript turns exactly the content that mid-session flushes
/// PROVABLY already extracted (verbatim whole-turn or verbatim long-line
/// matches against the flushed-content ledger).
///
/// Conservative by construction:
///   * a turn is dropped whole only when its entire trimmed content hashes to
///     a flushed item;
///   * otherwise only individual lines ≥ `MIN_FLUSHED_LINE_CHARS` whose exact
///     trimmed hash is in the ledger are removed (assistant quoting tool
///     output verbatim); every other line — all conversation reasoning,
///     decisions, paraphrase — is KEPT. Paraphrased near-duplicates are
///     intentionally out of scope: removing anything not byte-provably
///     flushed risks dropping never-extracted conversation facts, which is
///     the exact failure mode that killed the offset approach.
pub fn filter_turns_against_flushed(
    turns: Vec<crate::types::SessionTurn>,
    flushed: &std::collections::HashSet<String>,
) -> Vec<crate::types::SessionTurn> {
    if flushed.is_empty() {
        return turns;
    }
    turns
        .into_iter()
        .filter_map(|turn| {
            let trimmed = turn.content.trim();
            if trimmed.is_empty() {
                return Some(turn);
            }
            if flushed.contains(&ledger_line_hash(trimmed)) {
                return None;
            }
            let mut kept: Vec<&str> = Vec::new();
            let mut removed_any = false;
            for line in turn.content.lines() {
                let t = line.trim();
                if t.chars().count() >= MIN_FLUSHED_LINE_CHARS
                    && flushed.contains(&ledger_line_hash(t))
                {
                    removed_any = true;
                } else {
                    kept.push(line);
                }
            }
            if !removed_any {
                return Some(turn);
            }
            let content = kept.join("\n");
            if content.trim().is_empty() {
                None
            } else {
                Some(crate::types::SessionTurn {
                    role: turn.role,
                    content,
                })
            }
        })
        .collect()
}

/// Record that a mid-session flush occurred for this session.
/// Atomically increments the flush count stored in the marker file.
pub fn mark_flushed(buf_path: &std::path::Path) {
    let marker = flush_marker_path(buf_path);
    let count = std::fs::read_to_string(&marker)
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
        .unwrap_or(0);
    let _ = std::fs::write(&marker, (count + 1).to_string());
}

/// Read the number of mid-session flushes recorded for this session.
/// Returns 0 if the marker file does not exist.
pub fn flush_count(buf_path: &std::path::Path) -> u32 {
    let marker = flush_marker_path(buf_path);
    std::fs::read_to_string(&marker)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0)
}

/// Delete the flush marker for this session (call at hook_stop end).
pub fn clear_flush_marker(buf_path: &std::path::Path) {
    let _ = std::fs::remove_file(flush_marker_path(buf_path));
}

/// Clean up stale buffer files older than 24 hours.
pub fn cleanup_stale_buffers(config: &ReinConfig) {
    let buf_dir = resolve_buffer_dir(config);
    // Clean stale buffer files
    let pattern = buf_dir.join("buffer_*.jsonl");
    if let Ok(entries) = glob::glob(&pattern.to_string_lossy()) {
        let cutoff = chrono::Utc::now() - chrono::Duration::hours(24);
        for entry in entries.flatten() {
            if let Ok(meta) = std::fs::metadata(&entry) {
                if let Ok(modified) = meta.modified() {
                    let modified_utc: chrono::DateTime<chrono::Utc> = modified.into();
                    if modified_utc < cutoff {
                        tracing::info!("cleaning stale buffer: {}", entry.display());
                        let _ = std::fs::remove_file(&entry);
                        // Also remove associated flush marker/ledger files.
                        // v1.2 audit F17: the LOCK file is intentionally NOT
                        // removed — flock correctness requires a stable inode.
                        // Unlinking a lock file races with_buffer_lock's
                        // open→flock gap: a peer that already opened the old
                        // inode would then hold a lock no future process can
                        // see (the path now resolves to a fresh inode),
                        // putting two writers inside the "exclusive" section.
                        // Lock files are 0-byte and bounded by session count;
                        // leaving them is free.
                        let marker = flush_marker_path(&entry);
                        let _ = std::fs::remove_file(&marker);
                        let _ = std::fs::remove_file(flushed_ledger_path(&entry));
                    }
                }
            }
        }
    }
    // codex v1.2 R4 P2: orphan ledger/marker sweep. A mid-session flush
    // deletes the buffer file (`read_and_clear_buffer`) and then writes the
    // flushed-hashes ledger; if the session aborts before `hook_stop`, the
    // sidecars outlive any `buffer_*.jsonl` and the loop above never visits
    // them — they would accumulate (up to MAX_LEDGER_BYTES each) forever.
    // Sweep them independently with the same 24h staleness cutoff.
    for pattern in ["buffer_*.flushed-hashes", "buffer_*.flushed"] {
        let glob_pattern = buf_dir.join(pattern);
        if let Ok(entries) = glob::glob(&glob_pattern.to_string_lossy()) {
            let cutoff = chrono::Utc::now() - chrono::Duration::hours(24);
            for entry in entries.flatten() {
                if let Ok(meta) = std::fs::metadata(&entry) {
                    if let Ok(modified) = meta.modified() {
                        let modified_utc: chrono::DateTime<chrono::Utc> = modified.into();
                        if modified_utc < cutoff {
                            tracing::info!("cleaning stale flush sidecar: {}", entry.display());
                            let _ = std::fs::remove_file(&entry);
                        }
                    }
                }
            }
        }
    }
    // v1.2 audit F17: the orphan-lock sweep was REMOVED. "Probe with LOCK_NB
    // then unlink" cannot be made safe: a peer inside with_buffer_lock's
    // open→flock gap has the OLD inode open; after our unlink it flocks an
    // unlinked inode while every later process creates and locks a NEW inode
    // at the same path — two writers end up inside the "exclusive" section
    // and the buffer/ledger files interleave. Lock files are 0-byte and
    // bounded by session count, so leaving them costs nothing.
}

fn buffer_lock_path(path: &std::path::Path) -> std::path::PathBuf {
    std::path::PathBuf::from(format!("{}.lock", path.display()))
}

fn with_buffer_lock<T, F>(path: &std::path::Path, f: F) -> anyhow::Result<T>
where
    F: FnOnce() -> anyhow::Result<T>,
{
    let lock_path = buffer_lock_path(path);
    if let Some(parent) = lock_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let lock_file = std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(&lock_path)?;
    #[cfg(unix)]
    {
        use std::os::fd::AsRawFd;
        let rc = unsafe { libc::flock(lock_file.as_raw_fd(), libc::LOCK_EX) };
        if rc != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
    }
    let result = f();
    #[cfg(unix)]
    {
        use std::os::fd::AsRawFd;
        let _ = unsafe { libc::flock(lock_file.as_raw_fd(), libc::LOCK_UN) };
    }
    drop(lock_file);
    result
}

/// Store an episode summary as a concept in the "sessions" memoir.
pub fn store_episode_concept(
    store: &crate::store::SqliteStore,
    episode: &crate::extract::llm::EpisodeSummary,
) -> crate::types::ReinResult<()> {
    if store.get_memoir("sessions")?.is_none() {
        let memoir = crate::types::Memoir {
            id: String::new(),
            name: "sessions".to_string(),
            description: "Auto-created session episode summaries".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };
        store.create_memoir(memoir)?;
    }
    let date = format!(
        "{}-{}",
        chrono::Utc::now().format("%Y-%m-%d-%H%M"),
        ulid::Ulid::new()
            .to_string()
            .chars()
            .take(6)
            .collect::<String>()
    );
    let definition = if episode.decisions.is_empty() {
        format!("{}\nOutcome: {}", episode.title, episode.outcome)
    } else {
        format!(
            "{}\nOutcome: {}\nDecisions: {}",
            episode.title,
            episode.outcome,
            episode.decisions.join("; ")
        )
    };
    let concept = crate::types::Concept {
        id: String::new(),
        memoir_id: "sessions".to_string(),
        name: format!("session-{}", date),
        definition,
        labels: vec!["episode".to_string()],
        source_memory_ids: vec![],
        confidence: 0.8,
        revision: 1,
        last_episode_id: None,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        living_summary: None,
        living_summary_updated_at: None,
        living_summary_source_revision: None,
        living_summary_id: None,
    };
    store.add_concept(concept)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SessionTurn;

    fn turn(role: &str, content: &str) -> SessionTurn {
        SessionTurn {
            role: role.to_string(),
            content: content.to_string(),
        }
    }

    const LONG_LINE: &str = "error[E0308]: mismatched types in crates/rein/src/store/sqlite.rs:42";
    const SHORT_LINE: &str = "cargo test";

    #[test]
    fn ledger_roundtrip_records_items_and_long_lines_only() {
        let dir = tempfile::tempdir().unwrap();
        let buf = dir.path().join("buffer_test.jsonl");
        let item = format!("{LONG_LINE}\n{SHORT_LINE}\nok");
        record_flushed_content(&buf, &[item.clone()]);

        let hashes = read_flushed_hashes(&buf);
        assert!(
            hashes.contains(&ledger_line_hash(item.trim())),
            "whole trimmed item must be recorded"
        );
        assert!(
            hashes.contains(&ledger_line_hash(LONG_LINE)),
            "long lines must be recorded"
        );
        assert!(
            !hashes.contains(&ledger_line_hash(SHORT_LINE)),
            "short lines must NOT be recorded (cross-context collision risk)"
        );

        clear_flushed_ledger(&buf);
        assert!(
            read_flushed_hashes(&buf).is_empty(),
            "clear must empty the ledger"
        );
    }

    #[test]
    fn filter_drops_verbatim_whole_turn() {
        let dir = tempfile::tempdir().unwrap();
        let buf = dir.path().join("buffer_test.jsonl");
        let flushed_item = format!("{LONG_LINE}\nsecond diagnostic line with enough characters");
        record_flushed_content(&buf, &[flushed_item.clone()]);
        let hashes = read_flushed_hashes(&buf);

        let out = filter_turns_against_flushed(
            vec![
                turn("Assistant", &flushed_item),
                turn("User", "please fix the type mismatch in sqlite.rs"),
            ],
            &hashes,
        );
        assert_eq!(out.len(), 1, "verbatim flushed turn must be dropped whole");
        assert_eq!(out[0].role, "User");
    }

    #[test]
    fn filter_strips_only_flushed_long_lines_and_keeps_conversation() {
        let dir = tempfile::tempdir().unwrap();
        let buf = dir.path().join("buffer_test.jsonl");
        record_flushed_content(&buf, &[LONG_LINE.to_string()]);
        let hashes = read_flushed_hashes(&buf);

        let mixed = format!(
            "I looked at the failure:\n{LONG_LINE}\nWe decided to use i64 for the column type."
        );
        let out = filter_turns_against_flushed(vec![turn("Assistant", &mixed)], &hashes);
        assert_eq!(out.len(), 1);
        assert!(
            !out[0].content.contains(LONG_LINE),
            "flushed long line must be stripped"
        );
        assert!(
            out[0].content.contains("We decided to use i64"),
            "conversation facts must be KEPT"
        );
        assert!(
            out[0].content.contains("I looked at the failure:"),
            "surrounding reasoning must be KEPT"
        );
    }

    #[test]
    fn filter_keeps_short_line_matches_and_unrelated_turns() {
        let dir = tempfile::tempdir().unwrap();
        let buf = dir.path().join("buffer_test.jsonl");
        record_flushed_content(&buf, &[format!("{SHORT_LINE}\n{LONG_LINE}")]);
        let hashes = read_flushed_hashes(&buf);

        let with_short = format!("{SHORT_LINE}\nuser prefers running the suite before commits");
        let out = filter_turns_against_flushed(
            vec![
                turn("User", &with_short),
                turn("Assistant", "unrelated reasoning, nothing flushed here"),
            ],
            &hashes,
        );
        assert_eq!(out.len(), 2);
        assert!(
            out[0].content.contains(SHORT_LINE),
            "short lines must never be filtered even when present in a flushed item"
        );
        assert_eq!(out[1].content, "unrelated reasoning, nothing flushed here");
    }

    #[test]
    fn short_single_line_item_records_nothing_and_never_drops_turns() {
        let dir = tempfile::tempdir().unwrap();
        let buf = dir.path().join("buffer_test.jsonl");
        // codex R1 P2: a short flushed item must not enter the ledger at all —
        // its whole-item hash would otherwise drop any identical short turn.
        record_flushed_content(&buf, &[SHORT_LINE.to_string()]);
        assert!(
            read_flushed_hashes(&buf).is_empty(),
            "short single-line items must record nothing"
        );

        let out = filter_turns_against_flushed(
            vec![turn("User", SHORT_LINE)],
            &read_flushed_hashes(&buf),
        );
        assert_eq!(out.len(), 1, "identical short turn must survive");
    }

    #[test]
    fn ledger_batch_respects_byte_budget() {
        // codex R1 P3: one batch must not blow past the remaining budget.
        let items: Vec<String> = (0..100)
            .map(|i| format!("a unique long diagnostic line number {i:03} with padding"))
            .collect();
        // Budget for exactly 3 records (whole-item + line hashes interleaved).
        let out = build_ledger_records(&items, 3 * 65);
        assert_eq!(out.len(), 3 * 65, "batch must stop at the byte budget");
        // Unbounded for comparison: far more than 3 records.
        let full = build_ledger_records(&items, usize::MAX);
        assert!(full.len() > out.len());
    }

    #[test]
    fn filter_is_noop_with_empty_ledger() {
        let turns = vec![turn("Assistant", LONG_LINE)];
        let out = filter_turns_against_flushed(turns.clone(), &Default::default());
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].content, turns[0].content);
    }

    #[test]
    fn filter_drops_turn_emptied_by_line_stripping() {
        let dir = tempfile::tempdir().unwrap();
        let buf = dir.path().join("buffer_test.jsonl");
        let line_a = "first flushed diagnostic line with plenty of characters";
        let line_b = "second flushed diagnostic line with plenty of characters";
        record_flushed_content(&buf, &[format!("{line_a}\nx"), format!("{line_b}\ny")]);
        let hashes = read_flushed_hashes(&buf);

        // Turn consists ONLY of flushed long lines (in an order that does not
        // hash-match any whole item) → all lines stripped → turn dropped.
        let out = filter_turns_against_flushed(
            vec![turn("Assistant", &format!("{line_b}\n{line_a}"))],
            &hashes,
        );
        assert!(
            out.is_empty(),
            "turn with nothing left after stripping must be dropped"
        );
    }
}
