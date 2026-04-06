use std::path::{Path, PathBuf};

use tantivy::collector::TopDocs;
use tantivy::query::{BooleanQuery, Occur, QueryParser, TermQuery};
use tantivy::schema::*;
use tantivy::{doc, Index, IndexReader, IndexWriter, TantivyDocument};

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

impl TantivyFts {
    /// Open or create a Tantivy index at the given directory.
    /// The path is used directly as the index directory (e.g. `~/.rein/memories.tantivy/`).
    pub fn open(path: &Path) -> Result<Self, tantivy::TantivyError> {
        let index_path = path.to_path_buf();
        std::fs::create_dir_all(&index_path)
            .map_err(|e| tantivy::TantivyError::SystemError(format!("mkdir: {e}")))?;

        let mut schema_builder = Schema::builder();
        schema_builder.add_text_field("id", STRING | STORED);
        schema_builder.add_text_field("topic", TEXT | STORED);
        schema_builder.add_text_field("topic_exact", STRING);
        schema_builder.add_text_field("summary", TEXT);
        schema_builder.add_text_field("content", TEXT);
        schema_builder.add_text_field("keywords", TEXT);
        let schema = schema_builder.build();

        // Try to open existing, otherwise create new
        let index = match Index::open_in_dir(&index_path) {
            Ok(idx) => idx,
            Err(_) => Index::create_in_dir(&index_path, schema)?,
        };

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
            .unwrap_or(topic_field); // fallback for old indexes without topic_exact
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

        Ok(Self {
            index,
            reader,
            dirty_marker_path: index_path.join(".dirty"),
            id_field,
            topic_field,
            topic_exact_field,
            summary_field,
            content_field,
            keywords_field,
        })
    }

    /// Index a memory document. Replaces any existing doc with the same ID.
    /// For CJK text, content is pre-tokenized with jieba so Tantivy's default
    /// tokenizer (which splits on whitespace) can index individual Chinese words.
    /// If another process holds the writer lock, mark the index dirty so the next
    /// recall/warmup can rebuild it.
    pub fn insert(
        &self,
        id: &str,
        topic: &str,
        summary: &str,
        content: &str,
        keywords: &str,
    ) -> Result<(), tantivy::TantivyError> {
        let mut writer: IndexWriter = match self.index.writer(15_000_000) {
            Ok(w) => w,
            Err(tantivy::TantivyError::LockFailure(..)) => {
                self.mark_dirty();
                tracing::debug!("tantivy writer locked by another process, marking index dirty");
                return Ok(());
            }
            Err(e) => return Err(e),
        };

        // Pre-tokenize CJK fields: append jieba tokens so Tantivy indexes them as
        // separate terms alongside the original text (which Tantivy will also tokenize
        // with its default whitespace/punctuation splitter).
        let enriched_summary = enrich_cjk(summary);
        let enriched_content = enrich_cjk(content);

        // Delete old doc with same ID first
        let id_term = tantivy::Term::from_field_text(self.id_field, id);
        writer.delete_term(id_term);
        writer.add_document(doc!(
            self.id_field => id,
            self.topic_field => topic,
            self.topic_exact_field => topic,
            self.summary_field => enriched_summary.as_str(),
            self.content_field => enriched_content.as_str(),
            self.keywords_field => keywords,
        ))?;
        writer.commit()?;
        Ok(())
    }

    /// Search for documents matching the query string.
    /// For CJK queries, enriches the query with jieba tokens for better BM25 matching.
    /// When a topic filter is provided, uses BooleanQuery to filter at the index level
    /// instead of post-filtering in memory.
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

        // Enrich CJK queries with jieba tokens for better BM25 matching
        let enriched_query = enrich_cjk(query_str);

        // Try to parse; on failure, escape special chars and retry
        let text_query = match query_parser.parse_query(&enriched_query) {
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
    /// If another process holds the writer lock, mark the index dirty so it can
    /// be rebuilt later.
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
}

/// If text contains CJK characters, append jieba-segmented tokens so Tantivy's
/// default tokenizer (whitespace-based) can index individual words.
/// For pure ASCII text, returns as-is (zero overhead).
fn enrich_cjk(text: &str) -> String {
    if !crate::extract::dedup::contains_cjk(text) {
        return text.to_string();
    }
    let extra_tokens = crate::extract::tokenize_for_fts(text);
    if extra_tokens.is_empty() {
        return text.to_string();
    }
    format!("{text} {extra_tokens}")
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
        // Reopen
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
}
