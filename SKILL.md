---
name: rein
description: >
  Multi-source cross-validated memory system with adaptive engine, temporal knowledge graph,
  autonomous retrieval routing, and LLM extraction. 25 MCP tools, 20+ CLI commands. Features
  Neural Wiki GUI (React + Tailwind web dashboard), self-learning fusion weights (counterfactual
  alpha optimization), survival-based decay (per-cluster Kaplan-Meier replacing fixed Ebbinghaus),
  HDBSCAN semantic clustering, three-tier memory (Hot/Warm/Cold with t-digest boundaries),
  temporal knowledge graph (revision history, episodes, temporal links), autonomous query
  routing, memory evolution, canonical-first read model, evidence-aware recall,
  multi-factor admission control, concept dedup with name normalization, hybrid CJK lexical
  dedup (`jieba-rs` + bigrams), survival-driven STM promotion, and CC/RRF fusion.
  Triggers when user mentions memory, recall, remember, past sessions, knowledge graph,
  memoir, concepts, timeline, history, export, GUI, dashboard, or wants to save/search
  important context.
---

# rein Memory System (v0.13.x+)

Use rein to persist and retrieve knowledge across sessions. rein runs as an MCP server
(25 tools) or via CLI (20+ commands). Features Neural Wiki GUI (`rein serve --gui`),
LLM-powered extraction (Gemini 3.1 Flash Lite or local models), temporal knowledge graph,
autonomous retrieval routing, memory evolution, transparent LLM proxy (record-only),
concept name normalization and dedup, async memory pipeline with file-based queue and
background worker, project-scoped working set + always-on index for memory surfaces,
canonical-first recall with evidence previews, and adaptive learning loops that use
support_count/source_diversity in reranking and alpha optimization.

## CLI Commands

### Core Memory
```bash
rein recall "query" [-t topic] [-l limit]   # Search memories
rein store -t <topic> -c <content> -I <importance> [-k kw1,kw2]
rein update <id> -c <new_content> [-I <importance>]
rein forget <id>                             # Delete a memory
rein topics                                  # List all topics
rein stats                                   # Store statistics
rein health [topic]                          # Health check
rein recent [-l 20]                          # Most recent memories
```

### Maintenance
```bash
rein consolidate <topic> -s "summary"        # Merge topic into one memory
rein dedup [--dry-run]                       # Scan/remove duplicates
rein gc [--dry-run]                          # Garbage collect weak STM
rein organize                                # Auto-link related memories
rein dedup-concepts                          # Merge duplicate concepts (case/separator variants)
rein upgrade [--topic X] [--dry-run]         # Upgrade old memories to knowledge graph
rein export [--format json|md|csv] [--topic X] [--output file]
```

### System
```bash
rein serve [--compact] [--sse] [--proxy] [--gui]  # Start MCP server, proxy, or GUI
rein init [--dry-run]                        # Auto-configure MCP clients
rein config                                  # Show configuration
rein warmup                                  # Pre-compute embeddings
rein migrate [--from-qmd path] [--reindex]   # Import/reindex
rein worker memory                           # Drain async memory queue
```

### Hooks (used by Claude Code)
```bash
rein hook post      # Buffer tool output + pattern extraction (crash safety net)
rein hook compact   # Record compact context for async extraction
rein hook prompt    # Compatibility no-op (injection disabled)
rein hook stop      # Queue full knowledge extraction via async worker
```

### Proxy (record-only)
```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8690 claude   # Claude Code
OPENAI_BASE_URL=http://127.0.0.1:8690 codex       # Codex CLI
```

## MCP Tools (22)

### Core (12)
| Tool | Description |
|------|-------------|
| `rein_recall` | Semantic search with recency boost |
| `rein_store` | Store with auto-dedup + evolution + knowledge graph |
| `rein_update` | Update memory content/importance |
| `rein_forget` | Delete by ID |
| `rein_list_topics` | List all topics |
| `rein_stats` | Store statistics |
| `rein_health` | Topic health check |
| `rein_consolidate` | Merge topic memories |
| `rein_dedup` | Scan/remove duplicates |
| `rein_recent` | Most recent memories |
| `rein_gc` | Garbage collect weak STM |
| `rein_organize` | Auto-link related memories |

### Knowledge Graph (10)
| Tool | Description |
|------|-------------|
| `rein_memoir_create` | Create knowledge container |
| `rein_memoir_list` | List all memoirs |
| `rein_memoir_show` | Show memoir + concepts |
| `rein_memoir_add_concept` | Add knowledge node |
| `rein_memoir_refine` | Update concept, boost confidence |
| `rein_memoir_search` | FTS search within memoir |
| `rein_memoir_search_all` | Search across all memoirs |
| `rein_memoir_link` | Link concepts (9 relation types) |
| `rein_memoir_inspect` | BFS neighborhood traversal |
| `rein_memoir_export` | Export graph (json/ascii/dot) |

## Key Features

- **LLM Extraction**: Gemini 3.1 Flash Lite or local models (Ollama/LM Studio/vLLM)
- **Knowledge Graph**: Auto-creates concepts + typed links from session transcripts
- **Memory Evolution**: New memories automatically refine or supersede similar old ones
- **Canonical-First Read Model**: recall returns canonicals by default; evidence expands on demand
- **Evidence-Aware Recall**: low-confidence recall can use evidence previews and second-stage evidence rerank
- **Recency Boost**: 24h memories +50% search score, 7-day linear decay
- **Bidirectional Links**: Memory ↔ Concept, Memory ↔ Memory, Concept ↔ Concept
- **Buffer Architecture**: hook_post buffers, hook_stop does batch LLM extraction
- **Self-Learning Quality**: Concept-level scoring with auto-learned weights (PropMem/FActScore inspired)
- **Admission Control**: Adaptive threshold gates low-quality content from entering storage
- **Quality Pruning**: Auto-removes low-quality concepts that are recalled but never used
- **Adaptive Threshold**: Buffer flush adjusts by content signal density
- **Hybrid CJK Dedup**: Chinese/Japanese/Korean lexical dedup uses jieba-rs plus character bigrams
- **Cluster-Aware Admission**: admission threshold and novelty scoring incorporate cluster strength and cold-start blending
- **Survival-Driven Promotion**: STM→LTM promotion uses survival curves when cluster data exists
- **Layered Summaries**: canonical summaries stay detailed while API/UI expose `summary_short` for list views

## When to Use

- **Store**: After solving bugs, learning patterns, making architecture decisions
- **Recall**: Start of session, when context seems relevant, user asks about past work
- **Upgrade**: Convert old memories to knowledge graph format
- **Export**: Backup, migration, or sharing memory data
- **GC**: Periodically clean up weak memories
