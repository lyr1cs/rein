---
name: rein
description: >
  Multi-source cross-validated memory system. Use for storing, recalling, and managing
  persistent memories across sessions. Provides recall, store, update, forget, topics,
  stats, health, consolidate, and dedup commands. Triggers when user mentions memory,
  recall, remember, past sessions, previous work, or wants to save important context.
---

# rein Memory System

Use rein to persist and retrieve knowledge across sessions. rein runs as an MCP server
(9 tools) or via CLI (14 commands). Memories have topics, importance levels (low/medium/high/critical),
and automatically decay over time using an Ebbinghaus forgetting curve.

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

### Migrate from QMD
```bash
rein migrate [--from-qmd path]
```

### Hook commands (used by Claude Code hooks)
```bash
rein hook post      # Extract facts from tool output (stdin)
rein hook compact   # Extract context before compaction (stdin)
rein hook prompt    # Inject recalled memories into prompt (stdin/stdout)
```

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
