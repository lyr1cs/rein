use crate::types::{Importance, Memory, MemoryLayer, MemoryStatus, Source};
use std::collections::HashSet;

pub struct AutoMemoryScanner {
    glob_pattern: String,
}

impl AutoMemoryScanner {
    pub fn new(glob_pattern: String) -> Self {
        Self { glob_pattern }
    }

    /// Scan all auto-memory files and return those matching the query.
    pub fn scan(&self, query: &str) -> Vec<Memory> {
        let query_lower = query.to_lowercase();
        let query_words: HashSet<&str> = query_lower.split_whitespace().collect();

        let mut results = Vec::new();

        // Expand ~ to home dir
        let pattern = self.glob_pattern.replace('~', &dirs_home());

        for entry in glob::glob(&pattern).unwrap_or_else(|_| glob::glob("/dev/null").unwrap()) {
            if let Ok(path) = entry {
                if let Ok(content) = std::fs::read_to_string(&path) {
                    // Check keyword overlap
                    let content_lower = content.to_lowercase();
                    let content_words: HashSet<&str> = content_lower.split_whitespace().collect();
                    let overlap = query_words.intersection(&content_words).count();

                    if overlap > 0 {
                        let (title, body) = parse_frontmatter(&content);
                        results.push(Memory {
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
                            related_ids: vec![],
                            status: MemoryStatus::default(),
                            embedding: None,
                            created_at: chrono::Utc::now(),
                            updated_at: chrono::Utc::now(),
                            last_accessed: chrono::Utc::now(),
                        });
                    }
                }
            }
        }
        results
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
                .map(|l| l.splitn(2, ':').nth(1).unwrap_or("").trim().to_string());
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
        std::fs::write(&file1, "Rust Tips\nUse pattern matching for error handling")
            .unwrap();
        let file2 = dir.path().join("python_notes.md");
        std::fs::write(&file2, "Python Notes\nUse list comprehensions for clean code")
            .unwrap();

        let pattern = format!("{}/*.md", dir.path().display());
        let scanner = AutoMemoryScanner::new(pattern);

        // Query matching both files (word "for" appears in both)
        let results = scanner.scan("for");
        assert_eq!(results.len(), 2);

        // Query matching only rust file
        let results = scanner.scan("Rust");
        assert_eq!(results.len(), 1);
        assert!(results[0].id.contains("rust_tips"));

        // Query matching nothing
        let results = scanner.scan("javascript");
        assert_eq!(results.len(), 0);
    }
}
