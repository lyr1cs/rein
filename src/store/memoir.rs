use chrono::{DateTime, Utc};
use std::collections::{HashSet, VecDeque};
use std::str::FromStr;

use crate::store::fts::sanitize_fts_query;
use crate::types::*;

use super::sqlite::SqliteStore;

/// Escape a string for use in DOT (Graphviz) quoted strings.
fn escape_dot(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}

/// Map a rusqlite Row to a Memoir struct.
fn row_to_memoir(row: &rusqlite::Row) -> ReinResult<Memoir> {
    let id: String = row.get("id").map_err(ReinError::Database)?;
    let name: String = row.get("name").map_err(ReinError::Database)?;
    let description: String = row.get("description").map_err(ReinError::Database)?;
    let created_at_str: String = row.get("created_at").map_err(ReinError::Database)?;
    let updated_at_str: String = row.get("updated_at").map_err(ReinError::Database)?;

    let created_at = DateTime::parse_from_rfc3339(&created_at_str)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| ReinError::Config(format!("invalid created_at: {e}")))?;
    let updated_at = DateTime::parse_from_rfc3339(&updated_at_str)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| ReinError::Config(format!("invalid updated_at: {e}")))?;

    Ok(Memoir {
        id,
        name,
        description,
        created_at,
        updated_at,
    })
}

/// Map a rusqlite Row to a Concept struct.
pub(crate) fn row_to_concept(row: &rusqlite::Row) -> ReinResult<Concept> {
    let id: String = row.get("id").map_err(ReinError::Database)?;
    let memoir_id: String = row.get("memoir_id").map_err(ReinError::Database)?;
    let name: String = row.get("name").map_err(ReinError::Database)?;
    let definition: String = row.get("definition").map_err(ReinError::Database)?;
    let labels_json: String = row.get("labels").map_err(ReinError::Database)?;
    let confidence: f32 = row.get("confidence").map_err(ReinError::Database)?;
    let revision: u32 = row.get("revision").map_err(ReinError::Database)?;
    let created_at_str: String = row.get("created_at").map_err(ReinError::Database)?;
    let updated_at_str: String = row.get("updated_at").map_err(ReinError::Database)?;

    let labels: Vec<String> = serde_json::from_str(&labels_json)?;
    let source_memory_ids_json: String = row
        .get("source_memory_ids")
        .unwrap_or_else(|_| "[]".to_string());
    let source_memory_ids: Vec<String> =
        serde_json::from_str(&source_memory_ids_json).unwrap_or_default();

    let created_at = DateTime::parse_from_rfc3339(&created_at_str)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| ReinError::Config(format!("invalid created_at: {e}")))?;
    let updated_at = DateTime::parse_from_rfc3339(&updated_at_str)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| ReinError::Config(format!("invalid updated_at: {e}")))?;

    let last_episode_id: Option<String> = row.get("last_episode_id").unwrap_or(None);

    Ok(Concept {
        id,
        memoir_id,
        name,
        definition,
        labels,
        source_memory_ids,
        confidence,
        revision,
        last_episode_id,
        created_at,
        updated_at,
    })
}

/// Map a rusqlite Row to a ConceptLink struct.
fn row_to_link(row: &rusqlite::Row) -> ReinResult<ConceptLink> {
    let id: String = row.get("id").map_err(ReinError::Database)?;
    let source_id: String = row.get("source_id").map_err(ReinError::Database)?;
    let target_id: String = row.get("target_id").map_err(ReinError::Database)?;
    let relation_str: String = row.get("relation").map_err(ReinError::Database)?;
    let weight: f32 = row.get("weight").map_err(ReinError::Database)?;
    let created_at_str: String = row.get("created_at").map_err(ReinError::Database)?;

    let relation = Relation::from_str(&relation_str).map_err(ReinError::Config)?;

    let created_at = DateTime::parse_from_rfc3339(&created_at_str)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| ReinError::Config(format!("invalid created_at: {e}")))?;

    let valid_from: Option<DateTime<Utc>> = row
        .get::<_, Option<String>>("valid_from")
        .unwrap_or(None)
        .and_then(|s| {
            DateTime::parse_from_rfc3339(&s)
                .ok()
                .map(|dt| dt.with_timezone(&Utc))
        });
    let valid_until: Option<DateTime<Utc>> = row
        .get::<_, Option<String>>("valid_until")
        .unwrap_or(None)
        .and_then(|s| {
            DateTime::parse_from_rfc3339(&s)
                .ok()
                .map(|dt| dt.with_timezone(&Utc))
        });

    Ok(ConceptLink {
        id,
        source_id,
        target_id,
        relation,
        weight,
        created_at,
        valid_from,
        valid_until,
    })
}

/// Map a rusqlite Row to an Episode struct.
fn row_to_episode(row: &rusqlite::Row) -> ReinResult<Episode> {
    let decisions_json: String = row.get("decisions").map_err(ReinError::Database)?;
    let concept_ids_json: String = row.get("concept_ids").map_err(ReinError::Database)?;
    let memory_ids_json: String = row.get("memory_ids").map_err(ReinError::Database)?;
    let created_at_str: String = row.get("created_at").map_err(ReinError::Database)?;

    let created_at = DateTime::parse_from_rfc3339(&created_at_str)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| ReinError::Config(format!("invalid episode created_at: {e}")))?;

    Ok(Episode {
        id: row.get("id").map_err(ReinError::Database)?,
        title: row.get("title").map_err(ReinError::Database)?,
        outcome: row.get("outcome").map_err(ReinError::Database)?,
        decisions: serde_json::from_str(&decisions_json)?,
        concept_ids: serde_json::from_str(&concept_ids_json)?,
        memory_ids: serde_json::from_str(&memory_ids_json)?,
        created_at,
    })
}

impl SqliteStore {
    // --- Memoir CRUD ---

    /// Create a new memoir. Returns the generated ID.
    pub fn create_memoir(&self, memoir: Memoir) -> ReinResult<String> {
        let id = if memoir.id.is_empty() {
            ulid::Ulid::new().to_string()
        } else {
            memoir.id.clone()
        };
        let now = Utc::now();

        self.conn().execute(
            "INSERT INTO memoirs (id, name, description, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![
                id,
                memoir.name,
                memoir.description,
                now.to_rfc3339(),
                now.to_rfc3339(),
            ],
        )?;

        Ok(id)
    }

    /// Get a memoir by name.
    pub fn get_memoir(&self, name: &str) -> ReinResult<Option<Memoir>> {
        let mut stmt = self
            .conn()
            .prepare("SELECT * FROM memoirs WHERE name = ?1")?;
        let result = stmt.query_row(rusqlite::params![name], |row| {
            row_to_memoir(row).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    0,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })
        });

        match result {
            Ok(m) => Ok(Some(m)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(ReinError::Database(e)),
        }
    }

    /// List all memoirs.
    pub fn list_memoirs(&self) -> ReinResult<Vec<Memoir>> {
        let mut stmt = self
            .conn()
            .prepare("SELECT * FROM memoirs ORDER BY created_at DESC")?;
        let rows = stmt.query_map([], |row| {
            row_to_memoir(row).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    0,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })
        })?;
        Ok(rows
            .filter_map(|r| match r {
                Ok(m) => Some(m),
                Err(e) => {
                    tracing::warn!("failed to deserialize memoir row: {e}");
                    None
                }
            })
            .collect())
    }

    /// Delete a memoir by name. CASCADE deletes all concepts and links.
    pub fn delete_memoir(&self, name: &str) -> ReinResult<()> {
        let rows = self.conn().execute(
            "DELETE FROM memoirs WHERE name = ?1",
            rusqlite::params![name],
        )?;
        if rows == 0 {
            return Err(ReinError::NotFound(format!("memoir '{name}' not found")));
        }
        Ok(())
    }

    // --- Concept CRUD ---

    /// Add a concept to a memoir. Returns the generated concept ID.
    pub fn add_concept(&self, concept: Concept) -> ReinResult<String> {
        let id = if concept.id.is_empty() {
            ulid::Ulid::new().to_string()
        } else {
            concept.id.clone()
        };
        let now = Utc::now();
        let labels_json = serde_json::to_string(&concept.labels)?;
        let source_memory_ids_json = serde_json::to_string(&concept.source_memory_ids)?;

        // Look up memoir_id from memoir name if memoir_id looks like a name
        let memoir_id = self.resolve_memoir_id(&concept.memoir_id)?;

        self.conn().execute(
            "INSERT INTO concepts (id, memoir_id, name, definition, labels, source_memory_ids, confidence, revision, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            rusqlite::params![
                id,
                memoir_id,
                concept.name,
                concept.definition,
                labels_json,
                source_memory_ids_json,
                concept.confidence,
                concept.revision,
                now.to_rfc3339(),
                now.to_rfc3339(),
            ],
        )?;

        Ok(id)
    }

    /// Get a concept by memoir name and concept name.
    pub fn get_concept(
        &self,
        memoir_name: &str,
        concept_name: &str,
    ) -> ReinResult<Option<Concept>> {
        let memoir = self.get_memoir(memoir_name)?;
        let memoir = match memoir {
            Some(m) => m,
            None => return Ok(None),
        };

        let mut stmt = self
            .conn()
            .prepare("SELECT * FROM concepts WHERE memoir_id = ?1 AND name = ?2")?;
        let result = stmt.query_row(rusqlite::params![memoir.id, concept_name], |row| {
            row_to_concept(row).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    0,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })
        });

        match result {
            Ok(c) => Ok(Some(c)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(ReinError::Database(e)),
        }
    }

    /// Refine a concept: update definition, increment revision, boost confidence by 0.1 (max 1.0).
    pub fn refine_concept(
        &self,
        memoir_name: &str,
        concept_name: &str,
        new_definition: &str,
    ) -> ReinResult<()> {
        let memoir = self
            .get_memoir(memoir_name)?
            .ok_or_else(|| ReinError::NotFound(format!("memoir '{memoir_name}' not found")))?;

        // Snapshot current state as a revision before overwriting
        if let Some(old) = self.get_concept(memoir_name, concept_name)? {
            let rev_id = ulid::Ulid::new().to_string();
            let labels_json = serde_json::to_string(&old.labels).unwrap_or_default();
            let source_json = serde_json::to_string(&old.source_memory_ids).unwrap_or_default();
            if let Err(e) = self.conn().execute(
                "INSERT INTO concept_revisions (id, concept_id, revision, definition, confidence, labels, source_memory_ids, episode_id, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                rusqlite::params![
                    rev_id, old.id, old.revision, old.definition, old.confidence,
                    labels_json, source_json, old.last_episode_id,
                    old.updated_at.to_rfc3339(),
                ],
            ) {
                tracing::warn!("failed to snapshot concept revision for '{}': {e}", concept_name);
            }
        }

        let now = Utc::now();
        let rows = self.conn().execute(
            "UPDATE concepts
             SET definition = ?1,
                 revision = revision + 1,
                 confidence = MIN(confidence + 0.1, 1.0),
                 updated_at = ?2
             WHERE memoir_id = ?3 AND name = ?4",
            rusqlite::params![new_definition, now.to_rfc3339(), memoir.id, concept_name],
        )?;

        if rows == 0 {
            return Err(ReinError::NotFound(format!(
                "concept '{concept_name}' not found in memoir '{memoir_name}'"
            )));
        }

        Ok(())
    }

    /// FTS search for concepts within a memoir.
    pub fn search_concepts(
        &self,
        memoir_name: &str,
        query: &str,
        limit: usize,
    ) -> ReinResult<Vec<Concept>> {
        let memoir = self
            .get_memoir(memoir_name)?
            .ok_or_else(|| ReinError::NotFound(format!("memoir '{memoir_name}' not found")))?;

        let sanitized = sanitize_fts_query(query);
        if sanitized.is_empty() {
            return Ok(vec![]);
        }

        let mut stmt = self.conn().prepare(
            "SELECT c.*
             FROM concepts_fts f
             JOIN concepts c ON c.id = f.id
             WHERE concepts_fts MATCH ?1
             AND c.memoir_id = ?2
             ORDER BY bm25(concepts_fts)
             LIMIT ?3",
        )?;
        let rows = stmt.query_map(
            rusqlite::params![sanitized, memoir.id, limit as i64],
            |row| {
                row_to_concept(row).map_err(|e| {
                    rusqlite::Error::FromSqlConversionFailure(
                        0,
                        rusqlite::types::Type::Text,
                        Box::new(e),
                    )
                })
            },
        )?;

        Ok(rows
            .filter_map(|r| match r {
                Ok(c) => Some(c),
                Err(e) => {
                    tracing::warn!("failed to deserialize concept row: {e}");
                    None
                }
            })
            .collect())
    }

    /// Search concepts across all memoirs.
    pub fn search_all_concepts(&self, query: &str, limit: usize) -> ReinResult<Vec<Concept>> {
        let sanitized = sanitize_fts_query(query);
        if sanitized.is_empty() {
            return Ok(vec![]);
        }

        let mut stmt = self.conn().prepare(
            "SELECT c.*
             FROM concepts_fts f
             JOIN concepts c ON c.id = f.id
             WHERE concepts_fts MATCH ?1
             ORDER BY bm25(concepts_fts)
             LIMIT ?2",
        )?;
        let rows = stmt.query_map(rusqlite::params![sanitized, limit as i64], |row| {
            row_to_concept(row).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    0,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })
        })?;

        Ok(rows
            .filter_map(|r| match r {
                Ok(c) => Some(c),
                Err(e) => {
                    tracing::warn!("failed to deserialize concept row: {e}");
                    None
                }
            })
            .collect())
    }

    // --- Link CRUD ---

    /// Add a link between two concepts. Returns the generated link ID.
    pub fn add_link(&self, link: ConceptLink) -> ReinResult<String> {
        // Validate both concepts exist and belong to the same memoir
        let source_memoir: String = self
            .conn()
            .query_row(
                "SELECT memoir_id FROM concepts WHERE id = ?1",
                rusqlite::params![&link.source_id],
                |row| row.get(0),
            )
            .map_err(|_| {
                ReinError::NotFound(format!("source concept {} not found", link.source_id))
            })?;

        let target_memoir: String = self
            .conn()
            .query_row(
                "SELECT memoir_id FROM concepts WHERE id = ?1",
                rusqlite::params![&link.target_id],
                |row| row.get(0),
            )
            .map_err(|_| {
                ReinError::NotFound(format!("target concept {} not found", link.target_id))
            })?;

        if source_memoir != target_memoir {
            return Err(ReinError::Config(format!(
                "cross-memoir links not allowed: source in memoir {}, target in memoir {}",
                source_memoir, target_memoir
            )));
        }

        let id = if link.id.is_empty() {
            ulid::Ulid::new().to_string()
        } else {
            link.id.clone()
        };
        let now = Utc::now();

        let valid_from = link.valid_from.unwrap_or(now).to_rfc3339();
        let valid_until = link.valid_until.map(|dt| dt.to_rfc3339());

        let rows = self.conn().execute(
            "INSERT OR IGNORE INTO concept_links (id, source_id, target_id, relation, weight, created_at, valid_from, valid_until)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            rusqlite::params![
                id,
                link.source_id,
                link.target_id,
                link.relation.to_string(),
                link.weight,
                now.to_rfc3339(),
                valid_from,
                valid_until,
            ],
        )?;

        if rows == 0 {
            // Duplicate link — return existing link ID
            let existing_id: String = self.conn().query_row(
                "SELECT id FROM concept_links WHERE source_id = ?1 AND target_id = ?2 AND relation = ?3",
                rusqlite::params![link.source_id, link.target_id, link.relation.to_string()],
                |row| row.get(0),
            )?;
            return Ok(existing_id);
        }

        Ok(id)
    }

    /// Expire a link by setting its valid_until timestamp.
    pub fn expire_link(&self, link_id: &str, valid_until: DateTime<Utc>) -> ReinResult<()> {
        let rows = self.conn().execute(
            "UPDATE concept_links SET valid_until = ?1 WHERE id = ?2",
            rusqlite::params![valid_until.to_rfc3339(), link_id],
        )?;
        if rows == 0 {
            return Err(ReinError::NotFound(format!("link not found: {}", link_id)));
        }
        Ok(())
    }

    /// Get all links originating from a concept.
    pub fn get_links_from(&self, concept_id: &str) -> ReinResult<Vec<ConceptLink>> {
        let mut stmt = self
            .conn()
            .prepare("SELECT * FROM concept_links WHERE source_id = ?1")?;
        let rows = stmt.query_map(rusqlite::params![concept_id], |row| {
            row_to_link(row).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    0,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })
        })?;
        Ok(rows
            .filter_map(|r| match r {
                Ok(l) => Some(l),
                Err(e) => {
                    tracing::warn!("failed to deserialize link row: {e}");
                    None
                }
            })
            .collect())
    }

    /// Get all links pointing to a concept.
    pub fn get_links_to(&self, concept_id: &str) -> ReinResult<Vec<ConceptLink>> {
        let mut stmt = self
            .conn()
            .prepare("SELECT * FROM concept_links WHERE target_id = ?1")?;
        let rows = stmt.query_map(rusqlite::params![concept_id], |row| {
            row_to_link(row).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    0,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })
        })?;
        Ok(rows
            .filter_map(|r| match r {
                Ok(l) => Some(l),
                Err(e) => {
                    tracing::warn!("failed to deserialize link row: {e}");
                    None
                }
            })
            .collect())
    }

    // --- Graph traversal ---

    /// BFS neighborhood: returns (center concept, neighboring concepts, connecting links)
    /// up to `depth` hops from the center.
    pub fn inspect_concept(
        &self,
        memoir_name: &str,
        concept_name: &str,
        depth: usize,
    ) -> ReinResult<(Concept, Vec<Concept>, Vec<ConceptLink>)> {
        let center = self
            .get_concept(memoir_name, concept_name)?
            .ok_or_else(|| {
                ReinError::NotFound(format!(
                    "concept '{concept_name}' not found in memoir '{memoir_name}'"
                ))
            })?;

        let mut visited: HashSet<String> = HashSet::new();
        let mut seen_links: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<(String, usize)> = VecDeque::new();
        let mut neighbors: Vec<Concept> = Vec::new();
        let mut links: Vec<ConceptLink> = Vec::new();

        visited.insert(center.id.clone());
        queue.push_back((center.id.clone(), 0));

        while let Some((current_id, current_depth)) = queue.pop_front() {
            if current_depth >= depth {
                continue;
            }

            let now = Utc::now();

            // Get outgoing links (skip temporally invalid links)
            let outgoing = self.get_links_from(&current_id)?;
            for link in outgoing {
                if let Some(until) = link.valid_until {
                    if until < now {
                        continue;
                    } // expired link
                }
                if let Some(from) = link.valid_from {
                    if from > now {
                        continue;
                    } // not yet active
                }
                if seen_links.insert(link.id.clone()) {
                    links.push(link.clone());
                }
                if !visited.contains(&link.target_id) {
                    visited.insert(link.target_id.clone());
                    if let Some(concept) = self.get_concept_by_id(&link.target_id)? {
                        neighbors.push(concept);
                        queue.push_back((link.target_id, current_depth + 1));
                    }
                }
            }

            // Get incoming links (skip temporally invalid links)
            let incoming = self.get_links_to(&current_id)?;
            for link in incoming {
                if let Some(until) = link.valid_until {
                    if until < now {
                        continue;
                    }
                }
                if let Some(from) = link.valid_from {
                    if from > now {
                        continue;
                    }
                }
                if seen_links.insert(link.id.clone()) {
                    links.push(link.clone());
                }
                if !visited.contains(&link.source_id) {
                    visited.insert(link.source_id.clone());
                    if let Some(concept) = self.get_concept_by_id(&link.source_id)? {
                        neighbors.push(concept);
                        queue.push_back((link.source_id, current_depth + 1));
                    }
                }
            }
        }

        Ok((center, neighbors, links))
    }

    // --- Export ---

    /// Export a memoir in the given format: "json", "ascii", or "dot".
    pub fn export_memoir(&self, memoir_name: &str, format: &str) -> ReinResult<String> {
        let memoir = self
            .get_memoir(memoir_name)?
            .ok_or_else(|| ReinError::NotFound(format!("memoir '{memoir_name}' not found")))?;

        let concepts = self.get_concepts_by_memoir(&memoir.id)?;
        let links = self.get_links_by_memoir(&memoir.id)?;

        match format {
            "json" => {
                let export = serde_json::json!({
                    "memoir": {
                        "name": memoir.name,
                        "description": memoir.description,
                    },
                    "concepts": concepts.iter().map(|c| {
                        serde_json::json!({
                            "id": c.id,
                            "name": c.name,
                            "definition": c.definition,
                            "labels": c.labels,
                            "confidence": c.confidence,
                            "revision": c.revision,
                        })
                    }).collect::<Vec<_>>(),
                    "links": links.iter().map(|l| {
                        serde_json::json!({
                            "source_id": l.source_id,
                            "target_id": l.target_id,
                            "relation": l.relation.to_string(),
                            "weight": l.weight,
                        })
                    }).collect::<Vec<_>>(),
                });
                Ok(serde_json::to_string_pretty(&export)?)
            }
            "dot" => {
                let mut dot = String::new();
                dot.push_str(&format!("digraph \"{}\" {{\n", escape_dot(&memoir.name)));
                dot.push_str("  rankdir=LR;\n");
                dot.push_str("  node [shape=box, style=rounded];\n\n");

                for c in &concepts {
                    let escaped_def = escape_dot(&c.definition);
                    let label = if escaped_def.chars().count() > 60 {
                        format!("{}...", escaped_def.chars().take(60).collect::<String>())
                    } else {
                        escaped_def
                    };
                    let escaped_name = escape_dot(&c.name);
                    dot.push_str(&format!(
                        "  \"{}\" [label=\"{}\\n---\\n{}\"];\n",
                        c.id, escaped_name, label
                    ));
                }
                dot.push('\n');

                for l in &links {
                    let escaped_relation = escape_dot(&l.relation.to_string());
                    dot.push_str(&format!(
                        "  \"{}\" -> \"{}\" [label=\"{}\"];\n",
                        l.source_id, l.target_id, escaped_relation
                    ));
                }

                dot.push_str("}\n");
                Ok(dot)
            }
            "ascii" => {
                let mut out = String::new();
                let id_to_name: std::collections::HashMap<&str, &str> = concepts
                    .iter()
                    .map(|c| (c.id.as_str(), c.name.as_str()))
                    .collect();

                out.push_str(&format!("=== Memoir: {} ===\n\n", memoir.name));

                for c in &concepts {
                    out.push_str(&format!(
                        "+-- {} (confidence: {:.1}, rev: {})\n",
                        c.name, c.confidence, c.revision
                    ));
                    out.push_str(&format!("|   {}\n", c.definition));

                    // Show outgoing links from this concept
                    let outgoing: Vec<&ConceptLink> =
                        links.iter().filter(|l| l.source_id == c.id).collect();
                    for l in &outgoing {
                        let target_name = id_to_name.get(l.target_id.as_str()).unwrap_or(&"?");
                        out.push_str(&format!("|   --> {} --> {}\n", l.relation, target_name));
                    }

                    // Show incoming links to this concept
                    let incoming: Vec<&ConceptLink> =
                        links.iter().filter(|l| l.target_id == c.id).collect();
                    for l in &incoming {
                        let source_name = id_to_name.get(l.source_id.as_str()).unwrap_or(&"?");
                        out.push_str(&format!("|   <-- {} <-- {}\n", l.relation, source_name));
                    }

                    out.push_str("|\n");
                }

                Ok(out)
            }
            _ => Err(ReinError::Config(format!(
                "unknown export format: '{format}'. Use 'json', 'ascii', or 'dot'."
            ))),
        }
    }

    // --- Internal helpers ---

    /// Resolve a memoir name to its ID. If the input is already a valid memoir ID, return it.
    fn resolve_memoir_id(&self, name_or_id: &str) -> ReinResult<String> {
        // First try as name
        if let Some(m) = self.get_memoir(name_or_id)? {
            return Ok(m.id);
        }
        // Then check if it's a raw ID
        let exists: bool = self.conn().query_row(
            "SELECT COUNT(*) > 0 FROM memoirs WHERE id = ?1",
            rusqlite::params![name_or_id],
            |row| row.get(0),
        )?;
        if exists {
            Ok(name_or_id.to_string())
        } else {
            Err(ReinError::NotFound(format!(
                "memoir '{name_or_id}' not found"
            )))
        }
    }

    /// Get a concept by its ID (for BFS traversal).
    pub fn get_concept_by_id(&self, id: &str) -> ReinResult<Option<Concept>> {
        let mut stmt = self
            .conn()
            .prepare("SELECT * FROM concepts WHERE id = ?1")?;
        let result = stmt.query_row(rusqlite::params![id], |row| {
            row_to_concept(row).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    0,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })
        });

        match result {
            Ok(c) => Ok(Some(c)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(ReinError::Database(e)),
        }
    }

    /// Get all concepts in a memoir by memoir ID.
    fn get_concepts_by_memoir(&self, memoir_id: &str) -> ReinResult<Vec<Concept>> {
        let mut stmt = self
            .conn()
            .prepare("SELECT * FROM concepts WHERE memoir_id = ?1 ORDER BY name")?;
        let rows = stmt.query_map(rusqlite::params![memoir_id], |row| {
            row_to_concept(row).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    0,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })
        })?;
        Ok(rows
            .filter_map(|r| match r {
                Ok(c) => Some(c),
                Err(e) => {
                    tracing::warn!("failed to deserialize concept row: {e}");
                    None
                }
            })
            .collect())
    }

    /// Get all links where both endpoints belong to concepts in the given memoir.
    fn get_links_by_memoir(&self, memoir_id: &str) -> ReinResult<Vec<ConceptLink>> {
        let mut stmt = self.conn().prepare(
            "SELECT l.*
             FROM concept_links l
             JOIN concepts c1 ON l.source_id = c1.id
             WHERE c1.memoir_id = ?1",
        )?;
        let rows = stmt.query_map(rusqlite::params![memoir_id], |row| {
            row_to_link(row).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    0,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })
        })?;
        Ok(rows
            .filter_map(|r| match r {
                Ok(l) => Some(l),
                Err(e) => {
                    tracing::warn!("failed to deserialize link row: {e}");
                    None
                }
            })
            .collect())
    }
    // --- Episode CRUD ---

    /// Create a new episode node. Returns the generated ID.
    pub fn create_episode(&self, episode: Episode) -> ReinResult<String> {
        let id = if episode.id.is_empty() {
            ulid::Ulid::new().to_string()
        } else {
            episode.id
        };
        let decisions_json = serde_json::to_string(&episode.decisions).unwrap_or_default();
        let concept_ids_json = serde_json::to_string(&episode.concept_ids).unwrap_or_default();
        let memory_ids_json = serde_json::to_string(&episode.memory_ids).unwrap_or_default();
        let now = Utc::now();

        self.conn().execute(
            "INSERT INTO episodes (id, title, outcome, decisions, concept_ids, memory_ids, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            rusqlite::params![id, episode.title, episode.outcome, decisions_json, concept_ids_json, memory_ids_json, now.to_rfc3339()],
        )?;
        Ok(id)
    }

    /// Get an episode by ID.
    pub fn get_episode(&self, id: &str) -> ReinResult<Option<Episode>> {
        let result = self.conn().query_row(
            "SELECT * FROM episodes WHERE id = ?1",
            rusqlite::params![id],
            |row| {
                row_to_episode(row).map_err(|e| {
                    rusqlite::Error::FromSqlConversionFailure(
                        0,
                        rusqlite::types::Type::Text,
                        Box::new(e),
                    )
                })
            },
        );
        match result {
            Ok(ep) => Ok(Some(ep)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(ReinError::Database(e)),
        }
    }

    /// List recent episodes.
    pub fn list_episodes(&self, limit: usize) -> ReinResult<Vec<Episode>> {
        let mut stmt = self
            .conn()
            .prepare("SELECT * FROM episodes ORDER BY created_at DESC LIMIT ?1")?;
        let rows = stmt.query_map(rusqlite::params![limit], |row| {
            row_to_episode(row).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    0,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })
        })?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    /// Get episodes in a time range.
    pub fn get_episodes_in_range(
        &self,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
    ) -> ReinResult<Vec<Episode>> {
        let mut stmt = self.conn().prepare(
            "SELECT * FROM episodes WHERE created_at >= ?1 AND created_at <= ?2 ORDER BY created_at DESC"
        )?;
        let rows = stmt.query_map(
            rusqlite::params![from.to_rfc3339(), to.to_rfc3339()],
            |row| {
                row_to_episode(row).map_err(|e| {
                    rusqlite::Error::FromSqlConversionFailure(
                        0,
                        rusqlite::types::Type::Text,
                        Box::new(e),
                    )
                })
            },
        )?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    /// Search recent episodes by title/outcome/decisions using lightweight token matching.
    /// Returns ranked episodes with a heuristic score.
    pub fn search_episodes_ranked(
        &self,
        query: &str,
        limit: usize,
        from: Option<DateTime<Utc>>,
        to: Option<DateTime<Utc>>,
    ) -> ReinResult<Vec<(Episode, f32)>> {
        let scan_limit = limit.max(20) * 10;
        let episodes = match (from, to) {
            (Some(from), Some(to)) => self.get_episodes_in_range(from, to)?,
            (Some(from), None) => self.get_episodes_in_range(from, Utc::now() + chrono::Duration::days(1))?,
            (None, Some(to)) => self.get_episodes_in_range(
                chrono::DateTime::parse_from_rfc3339("2000-01-01T00:00:00Z")
                    .unwrap()
                    .with_timezone(&Utc),
                to,
            )?,
            (None, None) => self.list_episodes(scan_limit)?,
        };
        if episodes.is_empty() {
            return Ok(vec![]);
        }

        let query_lower = query.trim().to_lowercase();
        if query_lower.is_empty() {
            return Ok(vec![]);
        }

        let tokens: Vec<String> = query_lower
            .split_whitespace()
            .map(str::trim)
            .filter(|t| t.len() >= 2)
            .map(ToOwned::to_owned)
            .collect();

        let mut ranked = Vec::new();
        for episode in episodes {
            let haystack = format!(
                "{}\n{}\n{}",
                episode.title,
                episode.outcome,
                episode.decisions.join("\n")
            );
            let haystack_lower = haystack.to_lowercase();
            let mut score = 0.0_f32;

            if haystack_lower.contains(&query_lower) {
                score += 2.5;
            }

            for token in &tokens {
                if haystack_lower.contains(token) {
                    score += 0.6;
                }
            }

            if score <= 0.0 {
                continue;
            }

            let age_days = (Utc::now() - episode.created_at).num_hours() as f32 / 24.0;
            let recency = 1.0 / (1.0 + age_days / 30.0);
            ranked.push((episode, score + recency * 0.3));
        }

        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.0.created_at.cmp(&a.0.created_at))
        });
        ranked.truncate(limit);
        Ok(ranked)
    }

    // --- Concept Revision History ---

    /// Get revision history for a concept.
    pub fn get_concept_history(
        &self,
        memoir_name: &str,
        concept_name: &str,
        limit: usize,
    ) -> ReinResult<Vec<ConceptRevision>> {
        let concept = self
            .get_concept(memoir_name, concept_name)?
            .ok_or_else(|| ReinError::NotFound(format!("concept '{concept_name}' not found")))?;

        let mut stmt = self.conn().prepare(
            "SELECT * FROM concept_revisions WHERE concept_id = ?1 ORDER BY revision DESC LIMIT ?2",
        )?;
        let rows = stmt.query_map(rusqlite::params![concept.id, limit], |row| {
            let labels_json: String = row.get("labels")?;
            let source_json: String = row.get("source_memory_ids")?;
            let created_at_str: String = row.get("created_at")?;
            Ok(ConceptRevision {
                id: row.get("id")?,
                concept_id: row.get("concept_id")?,
                revision: row.get("revision")?,
                definition: row.get("definition")?,
                confidence: row.get("confidence")?,
                labels: serde_json::from_str(&labels_json).unwrap_or_default(),
                source_memory_ids: serde_json::from_str(&source_json).unwrap_or_default(),
                episode_id: row.get("episode_id").unwrap_or(None),
                created_at: DateTime::parse_from_rfc3339(&created_at_str)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now()),
            })
        })?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    /// Get the concept state at a specific point in time.
    pub fn get_concept_at(
        &self,
        memoir_name: &str,
        concept_name: &str,
        at: DateTime<Utc>,
    ) -> ReinResult<Option<ConceptRevision>> {
        let concept = self
            .get_concept(memoir_name, concept_name)?
            .ok_or_else(|| ReinError::NotFound(format!("concept '{concept_name}' not found")))?;

        let result = self.conn().query_row(
            "SELECT * FROM concept_revisions WHERE concept_id = ?1 AND created_at <= ?2 ORDER BY revision DESC LIMIT 1",
            rusqlite::params![concept.id, at.to_rfc3339()],
            |row| {
                let labels_json: String = row.get("labels")?;
                let source_json: String = row.get("source_memory_ids")?;
                let created_at_str: String = row.get("created_at")?;
                Ok(ConceptRevision {
                    id: row.get("id")?,
                    concept_id: row.get("concept_id")?,
                    revision: row.get("revision")?,
                    definition: row.get("definition")?,
                    confidence: row.get("confidence")?,
                    labels: serde_json::from_str(&labels_json).unwrap_or_default(),
                    source_memory_ids: serde_json::from_str(&source_json).unwrap_or_default(),
                    episode_id: row.get("episode_id").unwrap_or(None),
                    created_at: DateTime::parse_from_rfc3339(&created_at_str)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                })
            },
        );
        // Helper to synthesize a revision from the live concept row
        let live_revision = || ConceptRevision {
            id: format!("live-{}", concept.id),
            concept_id: concept.id.clone(),
            revision: concept.revision,
            definition: concept.definition.clone(),
            confidence: concept.confidence,
            labels: concept.labels.clone(),
            source_memory_ids: concept.source_memory_ids.clone(),
            episode_id: concept.last_episode_id.clone(),
            created_at: concept.updated_at,
        };

        match result {
            Ok(rev) => {
                // If the live concept has been updated after this revision AND the
                // requested time is after that update, return the live state instead.
                // This handles "after the latest refine" correctly.
                if concept.updated_at > rev.created_at && concept.updated_at <= at {
                    Ok(Some(live_revision()))
                } else {
                    Ok(Some(rev))
                }
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => {
                // No revision found — use live row if concept existed at requested time
                if concept.created_at <= at {
                    Ok(Some(live_revision()))
                } else {
                    Ok(None)
                }
            }
            Err(e) => Err(ReinError::Database(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_memoir(name: &str, desc: &str) -> Memoir {
        Memoir {
            id: String::new(),
            name: name.to_string(),
            description: desc.to_string(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    fn make_concept(memoir_id: &str, name: &str, definition: &str) -> Concept {
        Concept {
            id: String::new(),
            memoir_id: memoir_id.to_string(),
            name: name.to_string(),
            definition: definition.to_string(),
            labels: vec!["test".to_string()],
            source_memory_ids: vec![],
            confidence: 0.5,
            revision: 1,
            last_episode_id: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    #[test]
    fn test_create_and_list_memoirs() {
        let store = SqliteStore::in_memory().unwrap();
        store
            .create_memoir(make_memoir("rust-lang", "Rust language knowledge"))
            .unwrap();
        store
            .create_memoir(make_memoir("python", "Python knowledge"))
            .unwrap();

        let memoirs = store.list_memoirs().unwrap();
        assert_eq!(memoirs.len(), 2);

        let names: Vec<&str> = memoirs.iter().map(|m| m.name.as_str()).collect();
        assert!(names.contains(&"rust-lang"));
        assert!(names.contains(&"python"));
    }

    #[test]
    fn test_add_and_get_concept() {
        let store = SqliteStore::in_memory().unwrap();
        let memoir_id = store
            .create_memoir(make_memoir("rust-lang", "Rust"))
            .unwrap();

        let concept = make_concept(
            &memoir_id,
            "ownership",
            "Rust ownership model ensures memory safety without GC",
        );
        store.add_concept(concept).unwrap();

        let fetched = store.get_concept("rust-lang", "ownership").unwrap();
        assert!(fetched.is_some());
        let c = fetched.unwrap();
        assert_eq!(c.name, "ownership");
        assert_eq!(c.confidence, 0.5);
        assert_eq!(c.revision, 1);
    }

    #[test]
    fn test_refine_concept() {
        let store = SqliteStore::in_memory().unwrap();
        let memoir_id = store
            .create_memoir(make_memoir("rust-lang", "Rust"))
            .unwrap();

        let concept = make_concept(
            &memoir_id,
            "borrowing",
            "Borrowing allows references without ownership transfer",
        );
        store.add_concept(concept).unwrap();

        store
            .refine_concept(
                "rust-lang",
                "borrowing",
                "Borrowing: immutable and mutable references with lifetime rules",
            )
            .unwrap();

        let fetched = store
            .get_concept("rust-lang", "borrowing")
            .unwrap()
            .unwrap();
        assert_eq!(fetched.revision, 2);
        assert!((fetched.confidence - 0.6).abs() < 0.01);
        assert!(fetched.definition.contains("lifetime rules"));
    }

    #[test]
    fn test_add_link_and_traverse() {
        let store = SqliteStore::in_memory().unwrap();
        let memoir_id = store
            .create_memoir(make_memoir("rust-lang", "Rust"))
            .unwrap();

        let c1_id = store
            .add_concept(make_concept(&memoir_id, "ownership", "Ownership model"))
            .unwrap();
        let c2_id = store
            .add_concept(make_concept(&memoir_id, "borrowing", "Borrowing rules"))
            .unwrap();

        let link = ConceptLink {
            id: String::new(),
            source_id: c1_id.clone(),
            target_id: c2_id.clone(),
            relation: Relation::RelatedTo,
            weight: 1.0,
            created_at: Utc::now(),
            valid_from: None,
            valid_until: None,
        };
        store.add_link(link).unwrap();

        let from = store.get_links_from(&c1_id).unwrap();
        assert_eq!(from.len(), 1);
        assert_eq!(from[0].target_id, c2_id);

        let to = store.get_links_to(&c2_id).unwrap();
        assert_eq!(to.len(), 1);
        assert_eq!(to[0].source_id, c1_id);
    }

    #[test]
    fn test_inspect_neighborhood() {
        let store = SqliteStore::in_memory().unwrap();
        let memoir_id = store
            .create_memoir(make_memoir("rust-lang", "Rust"))
            .unwrap();

        let c1_id = store
            .add_concept(make_concept(&memoir_id, "ownership", "Ownership model"))
            .unwrap();
        let c2_id = store
            .add_concept(make_concept(&memoir_id, "borrowing", "Borrowing rules"))
            .unwrap();
        let c3_id = store
            .add_concept(make_concept(
                &memoir_id,
                "lifetimes",
                "Lifetime annotations",
            ))
            .unwrap();

        // ownership -> borrowing -> lifetimes
        store
            .add_link(ConceptLink {
                id: String::new(),
                source_id: c1_id.clone(),
                target_id: c2_id.clone(),
                relation: Relation::RelatedTo,
                weight: 1.0,
                created_at: Utc::now(),
                valid_from: None,
                valid_until: None,
            })
            .unwrap();
        store
            .add_link(ConceptLink {
                id: String::new(),
                source_id: c2_id.clone(),
                target_id: c3_id.clone(),
                relation: Relation::DependsOn,
                weight: 1.0,
                created_at: Utc::now(),
                valid_from: None,
                valid_until: None,
            })
            .unwrap();

        // depth=1 from ownership should find borrowing but not lifetimes
        let (center, neighbors, links) =
            store.inspect_concept("rust-lang", "ownership", 1).unwrap();
        assert_eq!(center.name, "ownership");
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].name, "borrowing");
        assert_eq!(links.len(), 1);

        // depth=2 from ownership should find both borrowing and lifetimes
        let (_, neighbors, links) = store.inspect_concept("rust-lang", "ownership", 2).unwrap();
        assert_eq!(neighbors.len(), 2);
        // 2 unique links: ownership->borrowing + borrowing->lifetimes (deduped by link ID)
        assert_eq!(links.len(), 2);
    }

    #[test]
    fn test_export_json() {
        let store = SqliteStore::in_memory().unwrap();
        let memoir_id = store
            .create_memoir(make_memoir("rust-lang", "Rust"))
            .unwrap();

        let c1_id = store
            .add_concept(make_concept(&memoir_id, "ownership", "Ownership model"))
            .unwrap();
        let c2_id = store
            .add_concept(make_concept(&memoir_id, "borrowing", "Borrowing rules"))
            .unwrap();

        store
            .add_link(ConceptLink {
                id: String::new(),
                source_id: c1_id,
                target_id: c2_id,
                relation: Relation::RelatedTo,
                weight: 1.0,
                created_at: Utc::now(),
                valid_from: None,
                valid_until: None,
            })
            .unwrap();

        let json = store.export_memoir("rust-lang", "json").unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["concepts"].as_array().unwrap().len(), 2);
        assert_eq!(parsed["links"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn test_delete_memoir_cascades() {
        let store = SqliteStore::in_memory().unwrap();
        let memoir_id = store
            .create_memoir(make_memoir("rust-lang", "Rust"))
            .unwrap();

        let c1_id = store
            .add_concept(make_concept(&memoir_id, "ownership", "Ownership model"))
            .unwrap();
        let c2_id = store
            .add_concept(make_concept(&memoir_id, "borrowing", "Borrowing rules"))
            .unwrap();

        store
            .add_link(ConceptLink {
                id: String::new(),
                source_id: c1_id.clone(),
                target_id: c2_id.clone(),
                relation: Relation::RelatedTo,
                weight: 1.0,
                created_at: Utc::now(),
                valid_from: None,
                valid_until: None,
            })
            .unwrap();

        // Delete memoir
        store.delete_memoir("rust-lang").unwrap();

        // Memoir should be gone
        assert!(store.get_memoir("rust-lang").unwrap().is_none());

        // Concepts should be cascade-deleted
        let concept_count: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM concepts WHERE memoir_id = ?1",
                rusqlite::params![memoir_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(concept_count, 0);

        // Links should be cascade-deleted
        let link_count: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM concept_links WHERE source_id = ?1 OR target_id = ?2",
                rusqlite::params![c1_id, c2_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(link_count, 0);
    }

    #[test]
    fn test_search_concepts_fts() {
        let store = SqliteStore::in_memory().unwrap();
        let memoir_id = store
            .create_memoir(make_memoir("rust-lang", "Rust"))
            .unwrap();

        store
            .add_concept(make_concept(
                &memoir_id,
                "ownership",
                "Rust ownership model ensures memory safety",
            ))
            .unwrap();
        store
            .add_concept(make_concept(
                &memoir_id,
                "borrowing",
                "Borrowing allows references without ownership transfer",
            ))
            .unwrap();
        store
            .add_concept(make_concept(
                &memoir_id,
                "lifetimes",
                "Lifetime annotations specify reference validity",
            ))
            .unwrap();

        let results = store.search_concepts("rust-lang", "ownership", 10).unwrap();
        assert!(!results.is_empty());
        // "ownership" appears in both concepts, but the one named "ownership" should be there
        assert!(results.iter().any(|c| c.name == "ownership"));
    }

    #[test]
    fn test_concept_revision_history() {
        let store = SqliteStore::in_memory().unwrap();
        let memoir_id = store.create_memoir(make_memoir("test", "Test")).unwrap();
        store
            .add_concept(make_concept(&memoir_id, "ownership", "Original definition"))
            .unwrap();

        // Refine twice — should create 2 revision snapshots
        store
            .refine_concept("test", "ownership", "Refined definition v2")
            .unwrap();
        store
            .refine_concept("test", "ownership", "Refined definition v3")
            .unwrap();

        let history = store.get_concept_history("test", "ownership", 10).unwrap();
        assert_eq!(history.len(), 2, "Should have 2 revisions");
        assert_eq!(history[0].revision, 2); // most recent first
        assert_eq!(history[1].revision, 1);
        assert!(history[1].definition.contains("Original"));

        // Current concept should be at r3
        let current = store.get_concept("test", "ownership").unwrap().unwrap();
        assert_eq!(current.revision, 3);
        assert!(current.definition.contains("v3"));
    }

    #[test]
    fn test_concept_at_point_in_time() {
        let store = SqliteStore::in_memory().unwrap();
        let memoir_id = store.create_memoir(make_memoir("test", "Test")).unwrap();
        store
            .add_concept(make_concept(&memoir_id, "api", "REST API v1"))
            .unwrap();
        store
            .refine_concept("test", "api", "GraphQL API v2")
            .unwrap();

        // get_concept_at with future time should return the live (current) definition
        let future = Utc::now() + chrono::Duration::hours(1);
        let rev = store.get_concept_at("test", "api", future).unwrap();
        assert!(rev.is_some());
        assert!(
            rev.unwrap().definition.contains("GraphQL"),
            "should return live definition after latest refine"
        );
    }

    #[test]
    fn test_episode_crud() {
        let store = SqliteStore::in_memory().unwrap();

        let episode = Episode {
            id: String::new(),
            title: "Test session".to_string(),
            outcome: "Built FT-2".to_string(),
            decisions: vec!["Use SAVEPOINT".to_string()],
            concept_ids: vec!["c1".to_string()],
            memory_ids: vec!["m1".to_string(), "m2".to_string()],
            created_at: Utc::now(),
        };
        let ep_id = store.create_episode(episode).unwrap();
        assert!(!ep_id.is_empty());

        // Get by ID
        let fetched = store.get_episode(&ep_id).unwrap().unwrap();
        assert_eq!(fetched.title, "Test session");
        assert_eq!(fetched.decisions.len(), 1);
        assert_eq!(fetched.memory_ids.len(), 2);

        // List
        let episodes = store.list_episodes(10).unwrap();
        assert_eq!(episodes.len(), 1);
    }

    #[test]
    fn test_episodes_in_range() {
        let store = SqliteStore::in_memory().unwrap();

        // Create episode
        let episode = Episode {
            id: String::new(),
            title: "Today's session".to_string(),
            outcome: String::new(),
            decisions: vec![],
            concept_ids: vec![],
            memory_ids: vec![],
            created_at: Utc::now(),
        };
        store.create_episode(episode).unwrap();

        // Range query: last 24 hours should find it
        let from = Utc::now() - chrono::Duration::hours(1);
        let to = Utc::now() + chrono::Duration::hours(1);
        let results = store.get_episodes_in_range(from, to).unwrap();
        assert_eq!(results.len(), 1);

        // Range query: yesterday should find nothing
        let old_from = Utc::now() - chrono::Duration::days(2);
        let old_to = Utc::now() - chrono::Duration::days(1);
        let results = store.get_episodes_in_range(old_from, old_to).unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_temporal_link_expiry() {
        let store = SqliteStore::in_memory().unwrap();
        let memoir_id = store.create_memoir(make_memoir("test", "Test")).unwrap();

        let c1_id = store
            .add_concept(make_concept(&memoir_id, "a", "Concept A"))
            .unwrap();
        let c2_id = store
            .add_concept(make_concept(&memoir_id, "b", "Concept B"))
            .unwrap();

        let link_id = store
            .add_link(ConceptLink {
                id: String::new(),
                source_id: c1_id.clone(),
                target_id: c2_id.clone(),
                relation: Relation::RelatedTo,
                weight: 1.0,
                created_at: Utc::now(),
                valid_from: None,
                valid_until: None,
            })
            .unwrap();

        // Expire the link
        store.expire_link(&link_id, Utc::now()).unwrap();

        // BFS should skip expired link
        let (_, neighbors, links) = store.inspect_concept("test", "a", 1).unwrap();
        assert!(
            neighbors.is_empty(),
            "Expired link should be skipped in BFS"
        );
        assert!(
            links.is_empty(),
            "Expired link should not appear in results"
        );
    }
}
