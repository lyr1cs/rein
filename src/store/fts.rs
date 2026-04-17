use crate::types::{Memory, ReinResult};
use rusqlite::Connection;

use super::sqlite::{memory_select_base, row_to_memory, MEMORY_SELECT_COLUMNS};

/// Sanitize user input for FTS5 queries by quoting each token.
///
/// NOTE: FTS5 uses the unicode61 tokenizer which does NOT segment CJK text.
/// CJK content is stored as-is, so we pass raw whitespace-split tokens here.
/// Jieba-based tokenization is only used in the Tantivy search path.
///
/// Also strips fullwidth / Unicode smart-quote variants — a straight `"` is
/// already filtered, but `"`, `"`, `'`, `'`, `」`, `「` etc. would otherwise
/// slip into the phrase literal and produce malformed FTS5 MATCH syntax.
pub fn sanitize_fts_query(query: &str) -> String {
    const FTS_RESERVED: &str = "-*:()\"^";
    const UNICODE_QUOTES: &[char] = &[
        '\u{201C}', '\u{201D}', // “ ”
        '\u{2018}', '\u{2019}', // ‘ ’
        '\u{FF02}', '\u{FF07}', // ＂ ＇
        '\u{300C}', '\u{300D}', // 「 」
        '\u{300E}', '\u{300F}', // 『 』
        '\u{FF08}', '\u{FF09}', // （ ）
    ];
    query
        .split_whitespace()
        .map(|t| {
            let clean: String = t
                .chars()
                .filter(|c| !FTS_RESERVED.contains(*c) && !UNICODE_QUOTES.contains(c))
                .collect();
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
        let mut stmt = conn.prepare(&format!(
            "SELECT {MEMORY_SELECT_COLUMNS}, bm25(memories_fts) as rank
             FROM memories_fts f
             JOIN memories m ON m.id = f.id
             LEFT JOIN memory_canonical_state cs ON cs.memory_id = m.id
             WHERE memories_fts MATCH ?1
             AND m.topic = ?2
             ORDER BY rank
             LIMIT ?3"
        ))?;
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
        let mut stmt = conn.prepare(&format!(
            "SELECT {MEMORY_SELECT_COLUMNS}, bm25(memories_fts) as rank
             FROM memories_fts f
             JOIN memories m ON m.id = f.id
             LEFT JOIN memory_canonical_state cs ON cs.memory_id = m.id
             WHERE memories_fts MATCH ?1
             ORDER BY rank
             LIMIT ?2"
        ))?;
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

    // Fallback to LIKE search on topic and summary.
    // Escape order matters: we use `ESCAPE '\'` in the SQL, so the escape
    // character (`\`) itself MUST be escaped first. Otherwise a user query
    // containing a literal backslash would cause the next character to be
    // interpreted as escaped, silently changing the match behavior
    // (e.g. "a\\b" would become "a\\b" with the `\` consuming the `b`).
    let escaped = query
        .replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_");
    let like_pattern = format!("%{escaped}%");
    let fallback: Vec<(crate::types::Memory, f32)> = if let Some(topic) = topic {
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

    if !fallback.is_empty() {
        return Ok(fallback);
    }

    // Final chance: LIKE search on content field (not just topic/summary).
    // Catches cases where the query string appears verbatim in memory body
    // but the FTS tokenizer missed it (e.g. rare technical identifiers that
    // were filtered as stopwords, or CJK edge cases).
    //
    // Gating: This is an unindexed O(N) full-table scan over `memories.content`.
    // We only run it when:
    //   1. The store is small enough that a full scan is cheap (< 50_000 rows), OR
    //   2. The query is long enough (>= 4 chars) that index filters are likely
    //      to return fast even on big stores (BM25 wouldn't have reached this
    //      branch if either tokenizer had indexed it).
    // This keeps latency bounded on large rein installations without giving up
    // the "catch verbatim matches FTS missed" safety net on typical deployments.
    const CONTENT_FALLBACK_SCAN_THRESHOLD: i64 = 50_000;
    const CONTENT_FALLBACK_MIN_QUERY_CHARS: usize = 4;
    let memory_count: i64 = conn
        .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))
        .unwrap_or(CONTENT_FALLBACK_SCAN_THRESHOLD + 1);
    if memory_count > CONTENT_FALLBACK_SCAN_THRESHOLD
        && query.chars().count() < CONTENT_FALLBACK_MIN_QUERY_CHARS
    {
        tracing::debug!(
            memory_count,
            query_chars = query.chars().count(),
            "fts: skipping content-field LIKE fallback (store too large for short query)"
        );
        return Ok(Vec::new());
    }

    let content_fallback = if let Some(topic) = topic {
        let sql = format!(
            "{} WHERE m.content LIKE ?1 ESCAPE '\\' AND m.topic = ?2 \
             ORDER BY m.strength DESC LIMIT ?3",
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
            .collect::<Vec<_>>()
    } else {
        let sql = format!(
            "{} WHERE m.content LIKE ?1 ESCAPE '\\' \
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
            .collect::<Vec<_>>()
    };

    Ok(content_fallback)
}
