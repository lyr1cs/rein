use chrono::{DateTime, Utc};
use std::collections::{HashSet, VecDeque};
use std::str::FromStr;

use crate::store::fts::sanitize_fts_query;
use crate::types::*;

use super::sqlite::SqliteStore;

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
fn row_to_concept(row: &rusqlite::Row) -> ReinResult<Concept> {
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

    let created_at = DateTime::parse_from_rfc3339(&created_at_str)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| ReinError::Config(format!("invalid created_at: {e}")))?;
    let updated_at = DateTime::parse_from_rfc3339(&updated_at_str)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| ReinError::Config(format!("invalid updated_at: {e}")))?;

    Ok(Concept {
        id,
        memoir_id,
        name,
        definition,
        labels,
        confidence,
        revision,
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

    let relation = Relation::from_str(&relation_str)
        .map_err(|e| ReinError::Config(e))?;

    let created_at = DateTime::parse_from_rfc3339(&created_at_str)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| ReinError::Config(format!("invalid created_at: {e}")))?;

    Ok(ConceptLink {
        id,
        source_id,
        target_id,
        relation,
        weight,
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
        let result = stmt
            .query_row(rusqlite::params![name], |row| {
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

        // Look up memoir_id from memoir name if memoir_id looks like a name
        let memoir_id = self.resolve_memoir_id(&concept.memoir_id)?;

        self.conn().execute(
            "INSERT INTO concepts (id, memoir_id, name, definition, labels, confidence, revision, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            rusqlite::params![
                id,
                memoir_id,
                concept.name,
                concept.definition,
                labels_json,
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

        let mut stmt = self.conn().prepare(
            "SELECT * FROM concepts WHERE memoir_id = ?1 AND name = ?2",
        )?;
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
    pub fn search_all_concepts(
        &self,
        query: &str,
        limit: usize,
    ) -> ReinResult<Vec<Concept>> {
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
        let rows = stmt.query_map(
            rusqlite::params![sanitized, limit as i64],
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

    // --- Link CRUD ---

    /// Add a link between two concepts. Returns the generated link ID.
    pub fn add_link(&self, link: ConceptLink) -> ReinResult<String> {
        let id = if link.id.is_empty() {
            ulid::Ulid::new().to_string()
        } else {
            link.id.clone()
        };
        let now = Utc::now();

        self.conn().execute(
            "INSERT INTO concept_links (id, source_id, target_id, relation, weight, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            rusqlite::params![
                id,
                link.source_id,
                link.target_id,
                link.relation.to_string(),
                link.weight,
                now.to_rfc3339(),
            ],
        )?;

        Ok(id)
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
        let mut queue: VecDeque<(String, usize)> = VecDeque::new();
        let mut neighbors: Vec<Concept> = Vec::new();
        let mut links: Vec<ConceptLink> = Vec::new();

        visited.insert(center.id.clone());
        queue.push_back((center.id.clone(), 0));

        while let Some((current_id, current_depth)) = queue.pop_front() {
            if current_depth >= depth {
                continue;
            }

            // Get outgoing links
            let outgoing = self.get_links_from(&current_id)?;
            for link in outgoing {
                links.push(link.clone());
                if !visited.contains(&link.target_id) {
                    visited.insert(link.target_id.clone());
                    if let Some(concept) = self.get_concept_by_id(&link.target_id)? {
                        neighbors.push(concept);
                        queue.push_back((link.target_id, current_depth + 1));
                    }
                }
            }

            // Get incoming links
            let incoming = self.get_links_to(&current_id)?;
            for link in incoming {
                links.push(link.clone());
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
                dot.push_str(&format!("digraph \"{}\" {{\n", memoir.name));
                dot.push_str("  rankdir=LR;\n");
                dot.push_str("  node [shape=box, style=rounded];\n\n");

                // Map concept ID to name for labels
                let id_to_name: std::collections::HashMap<&str, &str> = concepts
                    .iter()
                    .map(|c| (c.id.as_str(), c.name.as_str()))
                    .collect();

                for c in &concepts {
                    let escaped_def = c.definition.replace('"', "\\\"");
                    let label = if escaped_def.len() > 60 {
                        format!("{}...", &escaped_def[..60])
                    } else {
                        escaped_def
                    };
                    dot.push_str(&format!(
                        "  \"{}\" [label=\"{}\\n---\\n{}\"];\n",
                        c.id, c.name, label
                    ));
                }
                dot.push('\n');

                for l in &links {
                    let src = id_to_name.get(l.source_id.as_str()).unwrap_or(&"?");
                    let tgt = id_to_name.get(l.target_id.as_str()).unwrap_or(&"?");
                    dot.push_str(&format!(
                        "  \"{}\" -> \"{}\" [label=\"{}\"];\n",
                        l.source_id, l.target_id, l.relation
                    ));
                    let _ = (src, tgt); // used for readable comments if needed
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

                out.push_str(&format!(
                    "=== Memoir: {} ===\n\n",
                    memoir.name
                ));

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
                        let target_name =
                            id_to_name.get(l.target_id.as_str()).unwrap_or(&"?");
                        out.push_str(&format!(
                            "|   --> {} --> {}\n",
                            l.relation, target_name
                        ));
                    }

                    // Show incoming links to this concept
                    let incoming: Vec<&ConceptLink> =
                        links.iter().filter(|l| l.target_id == c.id).collect();
                    for l in &incoming {
                        let source_name =
                            id_to_name.get(l.source_id.as_str()).unwrap_or(&"?");
                        out.push_str(&format!(
                            "|   <-- {} <-- {}\n",
                            l.relation, source_name
                        ));
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
        let exists: bool = self
            .conn()
            .query_row(
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
    fn get_concept_by_id(&self, id: &str) -> ReinResult<Option<Concept>> {
        let mut stmt = self.conn().prepare("SELECT * FROM concepts WHERE id = ?1")?;
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
            confidence: 0.5,
            revision: 1,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    #[test]
    fn test_create_and_list_memoirs() {
        let store = SqliteStore::in_memory().unwrap();
        store.create_memoir(make_memoir("rust-lang", "Rust language knowledge")).unwrap();
        store.create_memoir(make_memoir("python", "Python knowledge")).unwrap();

        let memoirs = store.list_memoirs().unwrap();
        assert_eq!(memoirs.len(), 2);

        let names: Vec<&str> = memoirs.iter().map(|m| m.name.as_str()).collect();
        assert!(names.contains(&"rust-lang"));
        assert!(names.contains(&"python"));
    }

    #[test]
    fn test_add_and_get_concept() {
        let store = SqliteStore::in_memory().unwrap();
        let memoir_id = store.create_memoir(make_memoir("rust-lang", "Rust")).unwrap();

        let concept = make_concept(&memoir_id, "ownership", "Rust ownership model ensures memory safety without GC");
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
        let memoir_id = store.create_memoir(make_memoir("rust-lang", "Rust")).unwrap();

        let concept = make_concept(&memoir_id, "borrowing", "Borrowing allows references without ownership transfer");
        store.add_concept(concept).unwrap();

        store.refine_concept("rust-lang", "borrowing", "Borrowing: immutable and mutable references with lifetime rules").unwrap();

        let fetched = store.get_concept("rust-lang", "borrowing").unwrap().unwrap();
        assert_eq!(fetched.revision, 2);
        assert!((fetched.confidence - 0.6).abs() < 0.01);
        assert!(fetched.definition.contains("lifetime rules"));
    }

    #[test]
    fn test_add_link_and_traverse() {
        let store = SqliteStore::in_memory().unwrap();
        let memoir_id = store.create_memoir(make_memoir("rust-lang", "Rust")).unwrap();

        let c1_id = store.add_concept(make_concept(&memoir_id, "ownership", "Ownership model")).unwrap();
        let c2_id = store.add_concept(make_concept(&memoir_id, "borrowing", "Borrowing rules")).unwrap();

        let link = ConceptLink {
            id: String::new(),
            source_id: c1_id.clone(),
            target_id: c2_id.clone(),
            relation: Relation::RelatedTo,
            weight: 1.0,
            created_at: Utc::now(),
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
        let memoir_id = store.create_memoir(make_memoir("rust-lang", "Rust")).unwrap();

        let c1_id = store.add_concept(make_concept(&memoir_id, "ownership", "Ownership model")).unwrap();
        let c2_id = store.add_concept(make_concept(&memoir_id, "borrowing", "Borrowing rules")).unwrap();
        let c3_id = store.add_concept(make_concept(&memoir_id, "lifetimes", "Lifetime annotations")).unwrap();

        // ownership -> borrowing -> lifetimes
        store.add_link(ConceptLink {
            id: String::new(),
            source_id: c1_id.clone(),
            target_id: c2_id.clone(),
            relation: Relation::RelatedTo,
            weight: 1.0,
            created_at: Utc::now(),
        }).unwrap();
        store.add_link(ConceptLink {
            id: String::new(),
            source_id: c2_id.clone(),
            target_id: c3_id.clone(),
            relation: Relation::DependsOn,
            weight: 1.0,
            created_at: Utc::now(),
        }).unwrap();

        // depth=1 from ownership should find borrowing but not lifetimes
        let (center, neighbors, links) = store.inspect_concept("rust-lang", "ownership", 1).unwrap();
        assert_eq!(center.name, "ownership");
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].name, "borrowing");
        assert_eq!(links.len(), 1);

        // depth=2 from ownership should find both borrowing and lifetimes
        let (_, neighbors, links) = store.inspect_concept("rust-lang", "ownership", 2).unwrap();
        assert_eq!(neighbors.len(), 2);
        // 3 links: ownership->borrowing (from depth-0), borrowing->lifetimes + ownership->borrowing (from depth-1)
        assert_eq!(links.len(), 3);
    }

    #[test]
    fn test_export_json() {
        let store = SqliteStore::in_memory().unwrap();
        let memoir_id = store.create_memoir(make_memoir("rust-lang", "Rust")).unwrap();

        let c1_id = store.add_concept(make_concept(&memoir_id, "ownership", "Ownership model")).unwrap();
        let c2_id = store.add_concept(make_concept(&memoir_id, "borrowing", "Borrowing rules")).unwrap();

        store.add_link(ConceptLink {
            id: String::new(),
            source_id: c1_id,
            target_id: c2_id,
            relation: Relation::RelatedTo,
            weight: 1.0,
            created_at: Utc::now(),
        }).unwrap();

        let json = store.export_memoir("rust-lang", "json").unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["concepts"].as_array().unwrap().len(), 2);
        assert_eq!(parsed["links"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn test_delete_memoir_cascades() {
        let store = SqliteStore::in_memory().unwrap();
        let memoir_id = store.create_memoir(make_memoir("rust-lang", "Rust")).unwrap();

        let c1_id = store.add_concept(make_concept(&memoir_id, "ownership", "Ownership model")).unwrap();
        let c2_id = store.add_concept(make_concept(&memoir_id, "borrowing", "Borrowing rules")).unwrap();

        store.add_link(ConceptLink {
            id: String::new(),
            source_id: c1_id.clone(),
            target_id: c2_id.clone(),
            relation: Relation::RelatedTo,
            weight: 1.0,
            created_at: Utc::now(),
        }).unwrap();

        // Delete memoir
        store.delete_memoir("rust-lang").unwrap();

        // Memoir should be gone
        assert!(store.get_memoir("rust-lang").unwrap().is_none());

        // Concepts should be cascade-deleted
        let concept_count: i64 = store.conn().query_row(
            "SELECT COUNT(*) FROM concepts WHERE memoir_id = ?1",
            rusqlite::params![memoir_id],
            |row| row.get(0),
        ).unwrap();
        assert_eq!(concept_count, 0);

        // Links should be cascade-deleted
        let link_count: i64 = store.conn().query_row(
            "SELECT COUNT(*) FROM concept_links WHERE source_id = ?1 OR target_id = ?2",
            rusqlite::params![c1_id, c2_id],
            |row| row.get(0),
        ).unwrap();
        assert_eq!(link_count, 0);
    }

    #[test]
    fn test_search_concepts_fts() {
        let store = SqliteStore::in_memory().unwrap();
        let memoir_id = store.create_memoir(make_memoir("rust-lang", "Rust")).unwrap();

        store.add_concept(make_concept(&memoir_id, "ownership", "Rust ownership model ensures memory safety")).unwrap();
        store.add_concept(make_concept(&memoir_id, "borrowing", "Borrowing allows references without ownership transfer")).unwrap();
        store.add_concept(make_concept(&memoir_id, "lifetimes", "Lifetime annotations specify reference validity")).unwrap();

        let results = store.search_concepts("rust-lang", "ownership", 10).unwrap();
        assert!(!results.is_empty());
        // "ownership" appears in both concepts, but the one named "ownership" should be there
        assert!(results.iter().any(|c| c.name == "ownership"));
    }
}
