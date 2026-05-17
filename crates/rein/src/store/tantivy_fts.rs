use std::path::{Path, PathBuf};

use tantivy::collector::TopDocs;
use tantivy::query::{BooleanQuery, Occur, QueryParser, TermQuery};
use tantivy::schema::*;
use tantivy::{doc, Index, IndexReader, IndexWriter, TantivyDocument};

use crate::store::jieba_tokenizer::{JiebaTokenizer, TOKENIZER_NAME};

/// Tantivy-based full-text search with BM25 scoring.
/// Falls back gracefully — callers should handle errors and use FTS5 as backup.
pub struct TantivyFts {
    index: Index,
    reader: IndexReader,
    dirty_marker_path: PathBuf,
    id_field: Field,
    topic_field: Field,
    topic_exact_field: Field,
    summary_field: Field,
    content_field: Field,
    keywords_field: Field,
}

/// Marker file: presence means the index was built with the jieba tokenizer schema.
/// Absence triggers a full rebuild on next open.
const TOKENIZER_MARKER: &str = ".tokenizer_v2";

impl TantivyFts {
    /// Open or create a Tantivy index at the given directory.
    /// If an old index exists without the jieba tokenizer marker, it is deleted
    /// and recreated so searches use proper CJK segmentation. Data is repopulated
    /// by the warmup path (which reads from SQLite).
    pub fn open(path: &Path) -> Result<Self, tantivy::TantivyError> {
        let index_path = path.to_path_buf();
        std::fs::create_dir_all(&index_path)
            .map_err(|e| tantivy::TantivyError::SystemError(format!("mkdir: {e}")))?;

        // Migration: if old index exists without the tokenizer marker, wipe it so
        // it gets rebuilt with the jieba tokenizer schema.
        let marker_path = index_path.join(TOKENIZER_MARKER);
        let needs_rebuild = index_path.join("meta.json").exists() && !marker_path.exists();
        if needs_rebuild {
            tracing::info!(
                "tantivy index at {} missing jieba marker — rebuilding for CJK tokenizer",
                index_path.display()
            );
            // Remove old index files (best-effort; errors are logged but not fatal —
            // Tantivy's create_in_dir will fail if conflicting meta.json remains)
            if let Ok(entries) = std::fs::read_dir(&index_path) {
                for entry in entries.flatten() {
                    if let Err(e) = std::fs::remove_file(entry.path()) {
                        tracing::warn!(
                            "tantivy migration: could not remove {}: {e}",
                            entry.path().display()
                        );
                    }
                }
            }
        }

        let schema = Self::build_schema();

        let index = match Index::open_in_dir(&index_path) {
            Ok(idx) => idx,
            Err(_) => {
                // Attempt to create a fresh index.  If another process is racing the
                // same migration and holds the Tantivy writer lock, wait briefly and
                // retry the open (by then the winner will have created the index).
                match Index::create_in_dir(&index_path, schema) {
                    Ok(idx) => {
                        // Write marker only after confirmed successful creation
                        let _ = std::fs::write(&marker_path, b"jieba");
                        idx
                    }
                    Err(tantivy::TantivyError::LockFailure(..)) => {
                        std::thread::sleep(std::time::Duration::from_millis(150));
                        Index::open_in_dir(&index_path).map_err(|e| {
                            tantivy::TantivyError::SystemError(format!(
                                "tantivy open after concurrent migration: {e}"
                            ))
                        })?
                    }
                    Err(e) => return Err(e),
                }
            }
        };

        Self::from_opened_index(index, &index_path)
    }

    /// v0.30.4 D1 (deferred from v0.30.3 codex R13 P2-#2): non-creating
    /// variant of `open` for the recall read path.  Returns `Err` instead
    /// of recreating `<index_path>/` empty when the dir is missing.
    ///
    /// Why this matters: the warmup rebuild path renames the production
    /// tantivy dir to `.old` for a brief window during the staging→prod
    /// swap.  If `recall` calls `TantivyFts::open` in that window, the
    /// `std::fs::create_dir_all` at the top of `open` recreates an empty
    /// `<index_path>/`, which then makes the rebuild's
    /// `rename(staging → prod)` fail with EEXIST — promotion is lost,
    /// the backup is lost, and tantivy stays empty until the next
    /// rebuild trigger.  `open_existing` lets `recall` fall through to
    /// FTS5 in that window instead of corrupting the swap.
    ///
    /// Does NOT perform the jieba-tokenizer-marker migration (only the
    /// rebuild path needs that; reader path just opens what's there).
    pub fn open_existing(path: &Path) -> Result<Self, tantivy::TantivyError> {
        let index_path = path.to_path_buf();
        if !index_path.is_dir() {
            return Err(tantivy::TantivyError::SystemError(format!(
                "tantivy index dir does not exist: {}",
                index_path.display()
            )));
        }
        // `Index::open_in_dir` itself does NOT create the dir (only
        // `create_in_dir` does), so this is a pure read open.  Any
        // failure here (missing meta.json, corrupt segments) bubbles
        // up to the recall caller, which then falls back to FTS5.
        let index = Index::open_in_dir(&index_path)?;
        Self::from_opened_index(index, &index_path)
    }

    /// Shared post-open setup: jieba tokenizer registration, field
    /// handle derivation, dirty-marker sibling path computation.
    /// Used by both `open` (create-if-missing) and `open_existing`
    /// (read-only) so the field-derivation logic stays in one place.
    fn from_opened_index(index: Index, index_path: &Path) -> Result<Self, tantivy::TantivyError> {
        // Register the jieba tokenizer so the index (and QueryParser) can use it
        index.tokenizers().register(TOKENIZER_NAME, JiebaTokenizer);

        // Re-derive field handles from the opened index's schema
        let opened_schema = index.schema();
        let id_field = opened_schema
            .get_field("id")
            .map_err(|e| tantivy::TantivyError::SchemaError(format!("{e}")))?;
        let topic_field = opened_schema
            .get_field("topic")
            .map_err(|e| tantivy::TantivyError::SchemaError(format!("{e}")))?;
        let topic_exact_field = opened_schema
            .get_field("topic_exact")
            .unwrap_or(topic_field); // fallback for indexes without topic_exact
        let summary_field = opened_schema
            .get_field("summary")
            .map_err(|e| tantivy::TantivyError::SchemaError(format!("{e}")))?;
        let content_field = opened_schema
            .get_field("content")
            .map_err(|e| tantivy::TantivyError::SchemaError(format!("{e}")))?;
        let keywords_field = opened_schema
            .get_field("keywords")
            .map_err(|e| tantivy::TantivyError::SchemaError(format!("{e}")))?;

        let reader = index.reader()?;

        // v0.30.3 codex R21 P2: dirty marker MUST live at the SIBLING
        // path (`<index_path>.dirty`), NOT inside the directory. The
        // staging-and-swap rebuild path in `search/warmup.rs` renames
        // the entire index_path dir to `<index_path>.old` on success,
        // so a marker nested inside it gets swept away (signal lost)
        // or recreates the dir mid-swap (promotion fails EEXIST).
        // Compute the sibling: `memories.tantivy/` → `memories.tantivy.dirty`.
        let dirty_marker_path = match index_path.file_name().and_then(|s| s.to_str()) {
            Some(name) => index_path.with_file_name(format!("{name}.dirty")),
            None => index_path.join(".dirty"),
        };
        Ok(Self {
            index,
            reader,
            dirty_marker_path,
            id_field,
            topic_field,
            topic_exact_field,
            summary_field,
            content_field,
            keywords_field,
        })
    }

    /// Index a memory document. Replaces any existing doc with the same ID.
    /// CJK segmentation is handled natively by the jieba tokenizer — no pre-processing needed.
    pub fn insert(
        &self,
        id: &str,
        topic: &str,
        summary: &str,
        content: &str,
        keywords: &str,
    ) -> Result<(), tantivy::TantivyError> {
        self.insert_with_lock_policy(id, topic, summary, content, keywords, true)
    }

    /// Index a memory document during a full rebuild.
    ///
    /// Unlike hot-path `insert`, writer-lock contention is returned as an
    /// error so rebuild accounting does not count skipped documents as indexed.
    pub fn insert_strict(
        &self,
        id: &str,
        topic: &str,
        summary: &str,
        content: &str,
        keywords: &str,
    ) -> Result<(), tantivy::TantivyError> {
        self.insert_with_lock_policy(id, topic, summary, content, keywords, false)
    }

    fn insert_with_lock_policy(
        &self,
        id: &str,
        topic: &str,
        summary: &str,
        content: &str,
        keywords: &str,
        mark_dirty_on_lock: bool,
    ) -> Result<(), tantivy::TantivyError> {
        let mut writer: IndexWriter = match self.index.writer(15_000_000) {
            Ok(w) => w,
            Err(e @ tantivy::TantivyError::LockFailure(..)) => {
                if mark_dirty_on_lock {
                    self.mark_dirty();
                    tracing::debug!(
                        "tantivy writer locked by another process, marking index dirty"
                    );
                    return Ok(());
                }
                return Err(e);
            }
            Err(e) => return Err(e),
        };

        // Delete old doc with same ID first
        let id_term = tantivy::Term::from_field_text(self.id_field, id);
        writer.delete_term(id_term);
        writer.add_document(doc!(
            self.id_field => id,
            self.topic_field => topic,
            self.topic_exact_field => topic,
            self.summary_field => summary,
            self.content_field => content,
            self.keywords_field => keywords,
        ))?;
        writer.commit()?;
        Ok(())
    }

    /// Search for documents matching the query string.
    /// CJK queries are segmented by the jieba tokenizer registered on this index.
    /// When a topic filter is provided, uses BooleanQuery to filter at the index level.
    /// Returns pairs of (memory_id, BM25_score).
    pub fn search(
        &self,
        query_str: &str,
        topic: Option<&str>,
        limit: usize,
    ) -> Result<Vec<(String, f32)>, tantivy::TantivyError> {
        self.reader.reload()?;
        let searcher = self.reader.searcher();

        let query_parser = QueryParser::for_index(
            &self.index,
            vec![
                self.summary_field,
                self.content_field,
                self.keywords_field,
                self.topic_field,
            ],
        );

        // Try to parse; on failure, strip special chars and retry
        let text_query = match query_parser.parse_query(query_str) {
            Ok(q) => q,
            Err(_) => {
                let escaped = query_str
                    .chars()
                    .filter(|c| c.is_alphanumeric() || c.is_whitespace())
                    .collect::<String>();
                if escaped.trim().is_empty() {
                    return Ok(vec![]);
                }
                match query_parser.parse_query(&escaped) {
                    Ok(q) => q,
                    Err(_) => return Ok(vec![]),
                }
            }
        };

        // Combine text query with topic filter at index level
        let final_query: Box<dyn tantivy::query::Query> = if let Some(t) = topic {
            let topic_term = tantivy::Term::from_field_text(self.topic_exact_field, t);
            let topic_query = TermQuery::new(topic_term, IndexRecordOption::Basic);
            Box::new(BooleanQuery::new(vec![
                (Occur::Must, text_query),
                (Occur::Must, Box::new(topic_query)),
            ]))
        } else {
            text_query
        };

        let top_docs = searcher.search(&final_query, &TopDocs::with_limit(limit))?;

        let mut results = Vec::new();
        for (score, doc_address) in top_docs {
            let retrieved_doc: TantivyDocument = searcher.doc(doc_address)?;
            if let Some(id_val) = retrieved_doc.get_first(self.id_field) {
                if let OwnedValue::Str(ref id_str) = id_val {
                    results.push((id_str.clone(), score));
                }
            }
        }

        Ok(results)
    }

    /// Delete a document by memory ID.
    pub fn delete(&self, id: &str) -> Result<(), tantivy::TantivyError> {
        let mut writer: IndexWriter = match self.index.writer(15_000_000) {
            Ok(w) => w,
            Err(tantivy::TantivyError::LockFailure(..)) => {
                self.mark_dirty();
                tracing::debug!("tantivy writer locked by another process, marking index dirty");
                return Ok(());
            }
            Err(e) => return Err(e),
        };
        let term = tantivy::Term::from_field_text(self.id_field, id);
        writer.delete_term(term);
        writer.commit()?;
        Ok(())
    }

    fn mark_dirty(&self) {
        let _ = std::fs::write(&self.dirty_marker_path, b"dirty");
    }

    /// Build the schema using the jieba tokenizer for all TEXT fields.
    fn build_schema() -> Schema {
        let jieba_text = TextOptions::default().set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer(TOKENIZER_NAME)
                .set_index_option(IndexRecordOption::WithFreqsAndPositions),
        );

        let mut schema_builder = Schema::builder();
        schema_builder.add_text_field("id", STRING | STORED);
        schema_builder.add_text_field("topic", jieba_text.clone() | STORED);
        schema_builder.add_text_field("topic_exact", STRING);
        schema_builder.add_text_field("summary", jieba_text.clone());
        schema_builder.add_text_field("content", jieba_text.clone());
        schema_builder.add_text_field("keywords", jieba_text);
        schema_builder.build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tantivy_basic() {
        let dir = tempfile::tempdir().unwrap();
        let fts = TantivyFts::open(dir.path()).unwrap();

        fts.insert(
            "m1",
            "rust",
            "ownership rules",
            "Rust ownership and borrowing",
            "rust,memory",
        )
        .unwrap();
        fts.insert(
            "m2",
            "python",
            "decorators",
            "Python decorators for functions",
            "python,decorators",
        )
        .unwrap();

        let results = fts.search("ownership", None, 10).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].0, "m1");

        // Topic filter
        let results = fts.search("ownership", Some("python"), 10).unwrap();
        assert!(results.is_empty());

        // Delete
        fts.delete("m1").unwrap();
        let results = fts.search("ownership", None, 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_tantivy_reopen() {
        let dir = tempfile::tempdir().unwrap();
        {
            let fts = TantivyFts::open(dir.path()).unwrap();
            fts.insert("m1", "rust", "traits", "Rust traits and generics", "rust")
                .unwrap();
        }
        // Reopen — jieba tokenizer must be re-registered
        let fts = TantivyFts::open(dir.path()).unwrap();
        let results = fts.search("traits", None, 10).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_tantivy_special_chars() {
        let dir = tempfile::tempdir().unwrap();
        let fts = TantivyFts::open(dir.path()).unwrap();
        fts.insert("m1", "test", "test", "some content", "test")
            .unwrap();

        // Should not crash on special characters
        let results = fts.search("\" OR 1=1; DROP TABLE --", None, 10);
        assert!(results.is_ok());

        let results = fts.search("***^^^", None, 10);
        assert!(results.is_ok());
    }

    #[test]
    fn test_tantivy_cjk_search() {
        let dir = tempfile::tempdir().unwrap();
        let fts = TantivyFts::open(dir.path()).unwrap();

        fts.insert(
            "m1",
            "ai",
            "机器学习基础",
            "机器学习是人工智能的核心技术",
            "机器学习,人工智能",
        )
        .unwrap();
        fts.insert(
            "m2",
            "programming",
            "Rust programming",
            "Rust ownership model",
            "rust,ownership",
        )
        .unwrap();

        // CJK query should find the CJK document
        let results = fts.search("机器学习", None, 10).unwrap();
        assert!(!results.is_empty(), "CJK query should return results");
        assert_eq!(results[0].0, "m1");

        // ASCII query should not find the CJK document
        let results = fts.search("ownership", None, 10).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].0, "m2");
    }

    #[test]
    fn test_tantivy_migration_marker() {
        let dir = tempfile::tempdir().unwrap();
        // First open creates the marker
        let _fts = TantivyFts::open(dir.path()).unwrap();
        assert!(
            dir.path().join(TOKENIZER_MARKER).exists(),
            "marker file should be created on first open"
        );
    }

    /// v0.30.4 D1: `open_existing` MUST return `Err` when the index
    /// dir does not exist.  This is the whole point: the recall read
    /// path falls back to FTS5 instead of recreating an empty dir
    /// mid-swap (which would corrupt the warmup rebuild's promotion).
    #[test]
    fn open_existing_returns_err_when_dir_missing() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("never_created.tantivy");
        assert!(!missing.exists(), "test setup: path must not pre-exist");
        let result = TantivyFts::open_existing(&missing);
        assert!(
            result.is_err(),
            "D1: open_existing must reject a missing index dir, not silently create it"
        );
        // And critically, it must NOT have created the dir as a side
        // effect — that was the v0.30.3 bug.
        assert!(
            !missing.exists(),
            "D1: open_existing must not create the index dir as a side effect"
        );
    }

    /// v0.30.4 D1: `open_existing` MUST succeed against a directory
    /// previously initialized by `open`.  This proves the refactor
    /// (extracting `from_opened_index`) didn't break the happy path.
    #[test]
    fn open_existing_succeeds_against_initialized_index() {
        let dir = tempfile::tempdir().unwrap();
        let index_path = dir.path().join("inited.tantivy");
        // Use `open` to do the first-time setup (creates dir + writes
        // marker + creates fresh index).
        {
            let fts = TantivyFts::open(&index_path).unwrap();
            // Insert a doc so the index has segments
            fts.insert("d1", "Rust topic", "Summary text", "Content text", "k1,k2")
                .unwrap();
        }
        // Now `open_existing` must succeed and behave the same as
        // `open` would for the read path.
        let fts2 = TantivyFts::open_existing(&index_path)
            .expect("D1: open_existing must succeed against an initialized index");
        let results = fts2
            .search("Rust", None, 10)
            .expect("D1: search must work against an open_existing handle");
        assert!(
            results.iter().any(|(id, _)| id == "d1"),
            "D1: open_existing handle must see previously-inserted docs"
        );
    }
}
