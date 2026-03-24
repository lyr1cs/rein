---
name: rein
description: >
  Multi-source cross-validated memory system with knowledge graph. Use for storing, recalling,
  and managing persistent memories across sessions, plus building knowledge graphs via memoirs.
  Provides recall, store, update, forget, topics, stats, health, consolidate, dedup, and 10
  memoir commands. Triggers when user mentions memory, recall, remember, past sessions,
  previous work, knowledge graph, memoir, concepts, or wants to save important context.
---

# rein Memory System

Use rein to persist and retrieve knowledge across sessions. rein runs as an MCP server
(19 tools) or via CLI (15 commands). Memories have topics, importance levels (low/medium/high/critical),
and automatically decay over time using an Ebbinghaus forgetting curve. Memoirs provide a
knowledge graph layer with concepts, typed links, BFS traversal, and export.

## Available Commands

### Recall memories
```bash
rein recall "query" [-t topic] [-l limit]
```
Search memories by semantic query. Supports optional topic filter and result limit (default 10).

### Store a memory
```bash
rein store -t <topic> -c <content> -I <importance> [-k keyword1,keyword2]
```
Importance levels: `low`, `medium`, `high`, `critical`. Critical memories never decay.
Automatically deduplicates against existing memories (token similarity > 0.70, using max of Jaccard and containment).

### Update a memory
```bash
rein update <id> -c <new_content> [-I <importance>]
```

### Delete a memory
```bash
rein forget <id>
```

### List topics
```bash
rein topics
```

### Show statistics
```bash
rein stats
```
Returns total count, LTM/STM breakdown, average strength.

### Health check
```bash
rein health [topic]
```
Shows stale count, average strength, and consolidation recommendations.

### Consolidate a topic
```bash
rein consolidate <topic> -s "summary of all memories in this topic"
```
Merges all memories in a topic into a single summary memory.

### Deduplicate
```bash
rein dedup [--dry-run]
```
Scans for duplicate memories using content similarity. Use `--dry-run` to preview.

### Show configuration
```bash
rein config
```

### Auto-configure MCP clients
```bash
rein init [--dry-run]
```

### Migrate / Reindex
```bash
rein migrate [--from-qmd path] [--reindex]
```

### Hook commands (used by Claude Code hooks)
```bash
rein hook post      # Extract facts from tool output (stdin)
rein hook compact   # Extract context before compaction (stdin)
rein hook prompt    # Inject recalled memories into prompt (stdin/stdout)
```

### Create a memoir
```bash
rein_memoir_create name="my-project" description="Project knowledge"
```
Creates a knowledge graph container.

### List memoirs
```bash
rein_memoir_list
```

### Show memoir details
```bash
rein_memoir_show name="my-project"
```

### Add a concept
```bash
rein_memoir_add_concept memoir="my-project" name="caching" definition="In-memory LRU cache layer" labels="performance,infra"
```

### Refine a concept
```bash
rein_memoir_refine memoir="my-project" name="caching" definition="Updated definition with Redis backend"
```
Updates the concept definition and boosts its confidence.

### Search within a memoir
```bash
rein_memoir_search memoir="my-project" query="cache" limit=5
```

### Search across all memoirs
```bash
rein_memoir_search_all query="cache" limit=5
```

### Link two concepts
```bash
rein_memoir_link memoir="my-project" from="caching" to="redis" relation="depends_on"
```
Relation types: `part_of`, `depends_on`, `related_to`, `contradicts`, `refines`, `alternative_to`, `caused_by`, `instance_of`, `superseded_by`

### Inspect concept neighborhood
```bash
rein_memoir_inspect memoir="my-project" name="caching" depth=2
```
BFS traversal showing linked concepts up to the given depth.

### Export memoir graph
```bash
rein_memoir_export memoir="my-project" format="dot"
```
Formats: `json`, `ascii`, `dot`

## When to Use

- **Store**: After solving a complex bug, learning a project pattern, or discovering important configuration
- **Recall**: At the start of a session, when context seems relevant, or when the user asks about past work
- **Consolidate**: When a topic has many small memories that should be summarized
- **Health**: Periodically check for stale or redundant memories
- **Dedup**: When the store grows large, scan for duplicates

## MCP Tool Names

When rein runs as an MCP server, use these tool names:
- `rein_recall` - search memories
- `rein_store` - store a memory
- `rein_update` - update a memory
- `rein_forget` - delete a memory
- `rein_list_topics` - list topics
- `rein_stats` - show statistics
- `rein_health` - health check
- `rein_consolidate` - consolidate a topic
- `rein_dedup` - deduplicate memories
- `rein_memoir_create` - create a knowledge container
- `rein_memoir_list` - list all memoirs
- `rein_memoir_show` - show memoir details + concepts
- `rein_memoir_add_concept` - add knowledge node
- `rein_memoir_refine` - update concept, boost confidence
- `rein_memoir_search` - FTS search within memoir
- `rein_memoir_search_all` - search across all memoirs
- `rein_memoir_link` - link two concepts
- `rein_memoir_inspect` - BFS neighborhood traversal
- `rein_memoir_export` - export graph (json/ascii/dot)
