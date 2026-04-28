use crate::types::{Memory, ReinResult};
use rusqlite::Connection;

use super::sqlite::{memory_select_base, row_to_memory, MEMORY_SELECT_COLUMNS};

/// Canonical "live memory" SQL predicate used by every FTS path.
///
/// "Live" excludes `Deprecated` (terminal dead rows from `apply_evolution`).
/// Superseded rows (`superseded_by IS NOT NULL` with `status='active'`) are
/// NOT excluded — `collapse_results_to_canonicals` in `recall.rs` maps them
/// to their live canonical successor under the canonical-first read model.
/// Filtering them here would silently lose queries that match only the
/// old/evidence text. v0.26.2 R2 Codex F3 (corrects R1 over-broad guidance).
///
/// Mirrors `store::vec::search_vec` and `ops/adaptive.rs` — single source
/// of truth for the predicate.
const LIVE_MEMORY_FILTER: &str = "m.status IN ('active', 'updated')";

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
        // Bug #2: filter out deprecated / superseded rows so recall never
        // returns lifecycle-dead memories.
        let mut stmt = conn.prepare(&format!(
            "SELECT {MEMORY_SELECT_COLUMNS}, bm25(memories_fts) as rank
             FROM memories_fts f
             JOIN memories m ON m.id = f.id
             LEFT JOIN memory_canonical_state cs ON cs.memory_id = m.id
             WHERE memories_fts MATCH ?1
             AND m.topic = ?2
             AND {LIVE_MEMORY_FILTER}
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
        // Bug #2: same live-status filter as the topic-bound branch above.
        let mut stmt = conn.prepare(&format!(
            "SELECT {MEMORY_SELECT_COLUMNS}, bm25(memories_fts) as rank
             FROM memories_fts f
             JOIN memories m ON m.id = f.id
             LEFT JOIN memory_canonical_state cs ON cs.memory_id = m.id
             WHERE memories_fts MATCH ?1
             AND {LIVE_MEMORY_FILTER}
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
        // Bug #2: live-status filter applies to LIKE fallback too.
        let sql = format!(
            "{} WHERE (m.topic LIKE ?1 ESCAPE '\\' OR m.summary LIKE ?1 ESCAPE '\\') \
             AND m.topic = ?2 AND {LIVE_MEMORY_FILTER} \
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
            .collect()
    } else {
        // Bug #2: live-status filter — wrap the OR with parentheses so the
        // AND binds to both LIKE branches, not just the summary one.
        let sql = format!(
            "{} WHERE (m.topic LIKE ?1 ESCAPE '\\' OR m.summary LIKE ?1 ESCAPE '\\') \
             AND {LIVE_MEMORY_FILTER} \
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
        // Bug #2: live-status filter on content fallback too.
        let sql = format!(
            "{} WHERE m.content LIKE ?1 ESCAPE '\\' AND m.topic = ?2 \
             AND {LIVE_MEMORY_FILTER} \
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
        // Bug #2: live-status filter on the no-topic content fallback.
        let sql = format!(
            "{} WHERE m.content LIKE ?1 ESCAPE '\\' \
             AND {LIVE_MEMORY_FILTER} \
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::SqliteStore;
    use crate::types::{Importance, Memory, MemoryLayer, MemoryStatus, MemoryStore, Source};
    use chrono::Utc;

    /// Build a memory with an explicit status / superseded_by combo so the
    /// test can assert the live-status filter (Bug #2) excludes the right
    /// rows. All other fields are intentionally identical so the only
    /// signal driving the filter is `status` / `superseded_by`.
    fn fixture(
        topic: &str,
        content: &str,
        status: MemoryStatus,
        superseded_by: Option<&str>,
    ) -> Memory {
        Memory {
            id: ulid::Ulid::new().to_string(),
            layer: MemoryLayer::LTM,
            topic: topic.to_string(),
            summary: content.to_string(),
            content: content.to_string(),
            keywords: vec![],
            importance: Importance::Medium,
            source: Source::Manual,
            strength: 0.8,
            decay_lambda: 0.02,
            access_count: 0,
            superseded_by: superseded_by.map(|s| s.to_string()),
            canonical_id: None,
            support_count: 1,
            merge_count: 0,
            dedup_confidence: 1.0,
            source_diversity: 1.0,
            contradiction_score: 0.0,
            related_ids: vec![],
            concept_ids: vec![],
            status,
            embedding: None,
            tier: crate::store::tiering::MemoryTier::Warm,
            cluster_id: None,
            archival_summary: None,
            archival_summary_at: None,
            archival_summary_version: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_accessed: Utc::now(),
        }
    }

    /// Bug #2 (HIGH) + R2 Codex F3: the FTS5 + LIKE + content-fallback paths
    /// must filter out `Deprecated` rows (set by `apply_evolution`) but MUST
    /// surface `superseded_by IS NOT NULL` rows (set by `mark_superseded`,
    /// which leaves `status='active'`) — `collapse_results_to_canonicals`
    /// later maps them to the live canonical successor.
    #[test]
    fn search_fts_excludes_deprecated_keeps_superseded_for_canonical_collapse() {
        let store = SqliteStore::in_memory().unwrap();

        // Insert four rows with the same FTS-matching content. Use direct SQL
        // for `Updated` and `Deprecated` so we can place exact statuses
        // without going through the `update()` path.
        let active_id = store
            .store(fixture(
                "life",
                "alpha beta gamma",
                MemoryStatus::Active,
                None,
            ))
            .unwrap();
        let updated_id = store
            .store(fixture(
                "life",
                "alpha beta gamma",
                MemoryStatus::Active,
                None,
            ))
            .unwrap();
        let deprecated_id = store
            .store(fixture(
                "life",
                "alpha beta gamma",
                MemoryStatus::Active,
                None,
            ))
            .unwrap();
        let superseded_id = store
            .store(fixture(
                "life",
                "alpha beta gamma",
                MemoryStatus::Active,
                None,
            ))
            .unwrap();
        // Force the dead statuses post-insert so the FTS row exists.
        store
            .conn()
            .execute(
                "UPDATE memories SET status = 'updated' WHERE id = ?1",
                rusqlite::params![updated_id],
            )
            .unwrap();
        store
            .conn()
            .execute(
                "UPDATE memories SET status = 'deprecated' WHERE id = ?1",
                rusqlite::params![deprecated_id],
            )
            .unwrap();
        // Mirror `mark_superseded`: set superseded_by but leave status='active'
        // so this row tests the dual-predicate guard.
        store
            .conn()
            .execute(
                "UPDATE memories SET superseded_by = ?2 WHERE id = ?1",
                rusqlite::params![superseded_id, active_id],
            )
            .unwrap();

        // FTS5 path (no topic).
        let results = search_fts(store.conn(), "alpha", None, 10).unwrap();
        let ids: Vec<&str> = results.iter().map(|(m, _)| m.id.as_str()).collect();
        assert!(ids.contains(&active_id.as_str()), "Active must surface");
        assert!(ids.contains(&updated_id.as_str()), "Updated must surface");
        assert!(
            !ids.contains(&deprecated_id.as_str()),
            "Deprecated must be hidden"
        );
        // R2 F3: superseded rows surface so collapse can map them.
        assert!(
            ids.contains(&superseded_id.as_str()),
            "superseded row must surface for canonical-collapse mapping"
        );

        // FTS5 path with topic — same invariant.
        let results = search_fts(store.conn(), "alpha", Some("life"), 10).unwrap();
        let ids: Vec<&str> = results.iter().map(|(m, _)| m.id.as_str()).collect();
        assert!(ids.contains(&active_id.as_str()));
        assert!(ids.contains(&updated_id.as_str()));
        assert!(!ids.contains(&deprecated_id.as_str()));
        assert!(ids.contains(&superseded_id.as_str()));
    }
}
