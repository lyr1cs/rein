use crate::types::{Memory, ReinResult};
use rusqlite::Connection;

use super::sqlite::{memory_select_base, row_to_memory, MEMORY_SELECT_COLUMNS};

/// Sanitize user input for FTS5 queries by quoting each token.
/// NOTE: FTS5 uses the unicode61 tokenizer which does NOT segment CJK text.
/// CJK content is stored as-is, so we pass raw whitespace-split tokens here.
/// Jieba-based tokenization is only used in the Tantivy search path.
pub fn sanitize_fts_query(query: &str) -> String {
    query
        .split_whitespace()
        .map(|t| {
            let clean: String = t.chars().filter(|c| !"-*:()\"^".contains(*c)).collect();
            format!("\"{}\"", clean)
        })
        .filter(|t| t != "\"\"")
        .collect::<Vec<_>>()
        .join(" ")
}

/// Search using FTS5 with BM25 ranking, falling back to LIKE if no results.
pub fn search_fts(
    conn: &Connection,
    query: &str,
    topic: Option<&str>,
    limit: usize,
) -> ReinResult<Vec<(Memory, f32)>> {
    let sanitized = sanitize_fts_query(query);
    if sanitized.is_empty() {
        return Ok(vec![]);
    }

    // Try FTS5 first
    let results = if let Some(topic) = topic {
        let mut stmt = conn.prepare(
            &format!(
            "SELECT {MEMORY_SELECT_COLUMNS}, bm25(memories_fts) as rank
             FROM memories_fts f
             JOIN memories m ON m.id = f.id
             LEFT JOIN memory_canonical_state cs ON cs.memory_id = m.id
             WHERE memories_fts MATCH ?1
             AND m.topic = ?2
             ORDER BY rank
             LIMIT ?3"
            ),
        )?;
        let rows = stmt.query_map(rusqlite::params![sanitized, topic, limit as i64], |row| {
            let memory = row_to_memory(row).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    0,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })?;
            let rank: f64 = row.get("rank")?;
            Ok((memory, rank as f32))
        })?;
        rows.collect::<Result<Vec<_>, _>>()?
    } else {
        let mut stmt = conn.prepare(
            &format!(
            "SELECT {MEMORY_SELECT_COLUMNS}, bm25(memories_fts) as rank
             FROM memories_fts f
             JOIN memories m ON m.id = f.id
             LEFT JOIN memory_canonical_state cs ON cs.memory_id = m.id
             WHERE memories_fts MATCH ?1
             ORDER BY rank
             LIMIT ?2"
            ),
        )?;
        let rows = stmt.query_map(rusqlite::params![sanitized, limit as i64], |row| {
            let memory = row_to_memory(row).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    0,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })?;
            let rank: f64 = row.get("rank")?;
            Ok((memory, rank as f32))
        })?;
        rows.collect::<Result<Vec<_>, _>>()?
    };

    if !results.is_empty() {
        return Ok(results);
    }

    // Fallback to LIKE search on topic and summary
    // Escape LIKE wildcards in user input to prevent wildcard injection
    let escaped = query.replace('%', "\\%").replace('_', "\\_");
    let like_pattern = format!("%{escaped}%");
    let fallback = if let Some(topic) = topic {
        let sql = format!(
            "{} WHERE (m.topic LIKE ?1 ESCAPE '\\' OR m.summary LIKE ?1 ESCAPE '\\') \
             AND m.topic = ?2 ORDER BY m.strength DESC LIMIT ?3",
            memory_select_base()
        );
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(
            rusqlite::params![like_pattern, topic, limit as i64],
            |row| {
                row_to_memory(row).map_err(|e| {
                    rusqlite::Error::FromSqlConversionFailure(
                        0,
                        rusqlite::types::Type::Text,
                        Box::new(e),
                    )
                })
            },
        )?;
        rows.collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .map(|m| (m, 0.0f32))
            .collect()
    } else {
        let sql = format!(
            "{} WHERE m.topic LIKE ?1 ESCAPE '\\' OR m.summary LIKE ?1 ESCAPE '\\' \
             ORDER BY m.strength DESC LIMIT ?2",
            memory_select_base()
        );
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(rusqlite::params![like_pattern, limit as i64], |row| {
            row_to_memory(row).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    0,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .map(|m| (m, 0.0f32))
            .collect()
    };

    Ok(fallback)
}
