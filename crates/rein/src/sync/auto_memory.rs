use crate::types::{Importance, Memory, MemoryLayer, MemoryStatus, MemoryTier, Source};
use std::collections::HashSet;
use std::io::Read;
use std::path::Path;

/// Bytes of each file that are read, matched against the query and returned.
/// An auto-memory file stands in for one memory, so it gets the size a merged
/// canonical is allowed; the head of the file is kept.
const MAX_FILE_BYTES: usize = crate::store::sqlite::MERGE_CONTENT_CAP;

pub struct AutoMemoryScanner {
    glob_pattern: String,
}

impl AutoMemoryScanner {
    pub fn new(glob_pattern: String) -> Self {
        Self { glob_pattern }
    }

    /// Scan all auto-memory files and return every file matching the query,
    /// best match first. The caller applies its own filters and result limit;
    /// each returned file is bounded to its first `MAX_FILE_BYTES` bytes.
    pub fn scan(&self, query: &str) -> Vec<Memory> {
        self.scan_limited(query, usize::MAX)
    }

    /// Scan all auto-memory files and return at most `limit` matches.
    ///
    /// Query and file are tokenized with the same stop-word-aware, CJK-aware
    /// tokenizer (`eval::tokenize`: jieba for CJK text, stemmed words
    /// otherwise). A file matches when it shares at least one such term with
    /// the query; matches are ranked by the number of shared terms, then by
    /// modification time (newest first), then by path.
    pub fn scan_limited(&self, query: &str, limit: usize) -> Vec<Memory> {
        let query_terms: HashSet<String> = crate::eval::tokenize(query).into_iter().collect();
        if query_terms.is_empty() || limit == 0 {
            return Vec::new();
        }

        // Expand ~ to home dir
        let pattern = self.glob_pattern.replace('~', &dirs_home());
        let Ok(paths) = glob::glob(&pattern) else {
            return Vec::new();
        };

        let mut matches: Vec<(usize, Memory)> = Vec::new();
        for path in paths.flatten() {
            let Some(head) = read_head(&path, MAX_FILE_BYTES) else {
                continue;
            };
            let overlap = term_overlap(&query_terms, &head);
            if overlap == 0 {
                continue;
            }

            let (title, body) = parse_frontmatter(&head);
            let timestamp = std::fs::metadata(&path)
                .ok()
                .and_then(|meta| meta.modified().ok())
                .map(chrono::DateTime::<chrono::Utc>::from)
                .unwrap_or_else(chrono::Utc::now);
            matches.push((
                overlap,
                Memory {
                    id: format!("auto:{}", path.display()),
                    layer: MemoryLayer::LTM,
                    topic: "auto-memory".to_string(),
                    summary: title.unwrap_or_else(|| {
                        path.file_stem()
                            .map(|s| s.to_string_lossy().to_string())
                            .unwrap_or_default()
                    }),
                    content: body,
                    keywords: vec![],
                    importance: Importance::Medium,
                    source: Source::Manual, // auto-memory files are human-written
                    strength: 1.0,
                    decay_lambda: 0.0,
                    access_count: 0,
                    superseded_by: None,
                    canonical_id: None,
                    support_count: 1,
                    merge_count: 0,
                    dedup_confidence: 1.0,
                    source_diversity: 1.0,
                    contradiction_score: 0.0,
                    related_ids: vec![],
                    concept_ids: vec![],
                    status: MemoryStatus::default(),
                    embedding: None,
                    tier: MemoryTier::Warm,
                    cluster_id: None,
                    archival_summary: None,
                    archival_summary_at: None,
                    archival_summary_version: None,
                    created_at: timestamp,
                    updated_at: timestamp,
                    last_accessed: timestamp,
                },
            ));
            // Keep the working set small on large directories: only the best
            // `limit` candidates can be returned.
            if matches.len() >= limit.saturating_mul(2) {
                keep_best(&mut matches, limit);
            }
        }
        keep_best(&mut matches, limit);
        matches.into_iter().map(|(_, memory)| memory).collect()
    }
}

/// Sort by shared-term count (descending), then newest first, then path, and
/// keep the first `limit` entries.
fn keep_best(matches: &mut Vec<(usize, Memory)>, limit: usize) {
    matches.sort_by(|a, b| {
        b.0.cmp(&a.0)
            .then_with(|| b.1.updated_at.cmp(&a.1.updated_at))
            .then_with(|| a.1.id.cmp(&b.1.id))
    });
    matches.truncate(limit);
}

/// Number of distinct query terms that occur in `text`.
///
/// A term counts when the tokenized text contains it. A CJK term also counts
/// when it occurs as a substring: jieba may segment the same characters
/// differently inside a longer sentence than in a short query.
fn term_overlap(query_terms: &HashSet<String>, text: &str) -> usize {
    let text_terms: HashSet<String> = crate::eval::tokenize(text).into_iter().collect();
    let lower = text.to_lowercase();
    query_terms
        .iter()
        .filter(|term| {
            text_terms.contains(*term)
                || (crate::extract::dedup::contains_cjk(term) && lower.contains(term.as_str()))
        })
        .count()
}

/// Read at most `max_bytes` from the start of a file as UTF-8.
///
/// A multi-byte character cut in half by the byte limit is dropped. Files
/// that are unreadable or not UTF-8 return `None`.
fn read_head(path: &Path, max_bytes: usize) -> Option<String> {
    let file = std::fs::File::open(path).ok()?;
    let mut bytes = Vec::new();
    file.take(max_bytes as u64).read_to_end(&mut bytes).ok()?;
    match String::from_utf8(bytes) {
        Ok(text) => Some(text),
        Err(err) => {
            let utf8_error = err.utf8_error();
            if utf8_error.error_len().is_some() {
                // Invalid bytes inside the head, not a cut at its end.
                return None;
            }
            let mut bytes = err.into_bytes();
            bytes.truncate(utf8_error.valid_up_to());
            String::from_utf8(bytes).ok()
        }
    }
}

fn dirs_home() -> String {
    std::env::var("HOME").unwrap_or_else(|_| "~".to_string())
}

/// Parse YAML frontmatter from markdown. Returns (title, body).
fn parse_frontmatter(content: &str) -> (Option<String>, String) {
    if content.starts_with("---\n") {
        if let Some(end) = content[4..].find("\n---") {
            let frontmatter = &content[4..4 + end];
            let body = content[4 + end + 4..].trim().to_string();
            // Extract name/title from frontmatter
            let title = frontmatter
                .lines()
                .find(|l| l.starts_with("name:") || l.starts_with("title:"))
                .map(|l| {
                    l.split_once(':')
                        .map(|x| x.1)
                        .unwrap_or("")
                        .trim()
                        .to_string()
                });
            return (title, body);
        }
    }
    (None, content.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_frontmatter() {
        let content = "---\ntitle: My Memory\ntags: test\n---\nSome body content here.";
        let (title, body) = parse_frontmatter(content);
        assert_eq!(title, Some("My Memory".to_string()));
        assert_eq!(body, "Some body content here.");
    }

    #[test]
    fn test_parse_frontmatter_with_name() {
        let content = "---\nname: Named Memory\n---\nBody text.";
        let (title, body) = parse_frontmatter(content);
        assert_eq!(title, Some("Named Memory".to_string()));
        assert_eq!(body, "Body text.");
    }

    #[test]
    fn test_parse_no_frontmatter() {
        let content = "Just plain markdown content without frontmatter.";
        let (title, body) = parse_frontmatter(content);
        assert_eq!(title, None);
        assert_eq!(body, content);
    }

    #[test]
    fn test_scan_with_tempdir() {
        let dir = tempfile::tempdir().unwrap();
        let file1 = dir.path().join("rust_tips.md");
        std::fs::write(&file1, "Rust Tips\nUse pattern matching for error handling").unwrap();
        let file2 = dir.path().join("python_notes.md");
        std::fs::write(
            &file2,
            "Python Notes\nUse list comprehensions for clean code",
        )
        .unwrap();

        let pattern = format!("{}/*.md", dir.path().display());
        let scanner = AutoMemoryScanner::new(pattern);

        // A stop word alone matches nothing ("for" appears in both files)
        let results = scanner.scan("for");
        assert_eq!(results.len(), 0);

        // Query matching only rust file
        let results = scanner.scan("Rust");
        assert_eq!(results.len(), 1);
        assert!(results[0].id.contains("rust_tips"));

        // Query matching nothing
        let results = scanner.scan("javascript");
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn scan_matches_cjk_ranks_by_overlap_and_bounds_results() {
        let dir = tempfile::tempdir().unwrap();
        let write = |name: &str, body: &str| std::fs::write(dir.path().join(name), body).unwrap();
        write(
            "db_migration.md",
            "数据库迁移记录\n我们决定把数据库从 MySQL 迁移到 PostgreSQL。",
        );
        write(
            "rust_errors.md",
            "Rust Tips\nUse pattern matching for error handling",
        );
        write("error_codes.md", "Generic notes about the error codes");
        // A file far larger than one memory, in multi-byte characters.
        write("huge.md", &"部署流水线".repeat(40_000));

        let scanner = AutoMemoryScanner::new(format!("{}/*.md", dir.path().display()));

        // A Chinese query has no spaces; it still matches by its words.
        let results = scanner.scan("数据库迁移");
        assert_eq!(results.len(), 1);
        assert!(results[0].id.contains("db_migration"));

        // Stop words in the query do not match files on their own.
        assert!(scanner.scan("the for and").is_empty());

        // More shared terms rank first, and `limit` bounds the result.
        let results = scanner.scan("rust error handling");
        assert_eq!(results.len(), 2);
        assert!(results[0].id.contains("rust_errors"));
        assert!(results[1].id.contains("error_codes"));
        let results = scanner.scan_limited("rust error handling", 1);
        assert_eq!(results.len(), 1);
        assert!(results[0].id.contains("rust_errors"));

        // Only the head of a large file is returned, cut on a char boundary.
        let results = scanner.scan("部署流水线");
        assert_eq!(results.len(), 1);
        assert!(results[0].id.contains("huge"));
        assert!(results[0].content.len() <= MAX_FILE_BYTES);
        assert!(results[0].content.starts_with("部署流水线"));

        // `scan` returns every match; `scan_limited` bounds the count.
        let many = tempfile::tempdir().unwrap();
        for i in 0..15 {
            std::fs::write(
                many.path().join(format!("note_{i}.md")),
                format!("kubernetes rollout note {i}"),
            )
            .unwrap();
        }
        let scanner = AutoMemoryScanner::new(format!("{}/*.md", many.path().display()));
        assert_eq!(scanner.scan("kubernetes").len(), 15);
        assert_eq!(scanner.scan_limited("kubernetes", 10).len(), 10);
    }
}
