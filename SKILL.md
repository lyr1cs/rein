---
name: rein
description: >
  Multi-source cross-validated memory system with adaptive engine, temporal knowledge graph,
  autonomous retrieval routing, query expansion, LLM reranking, MMR diversity, and LLM extraction.
  28 MCP tools, 20+ CLI commands. Features Neural Wiki GUI (React + Tailwind web dashboard),
  self-learning fusion weights (counterfactual alpha optimization — global, per-query-type, and
  per-cluster M2), survival-based decay (per-cluster Kaplan-Meier + global cold-start prior M3),
  HDBSCAN semantic clustering (M4), three-tier memory Hot/Warm/Cold (M5), per-cluster adaptive
  dedup thresholds A1 (full pipeline: store/batch/vec dedup), temporal knowledge graph (revision
  history, episodes, temporal links), autonomous query routing (6 strategies, rule-based, 0 LLM
  calls), memory evolution, canonical-first read model, evidence-aware recall, multi-factor
  admission control, hybrid CJK lexical dedup (jieba-rs + bigrams), survival-driven STM promotion,
  Maximal Marginal Relevance reranking, and CC/RRF fusion with learned alpha weights.
  Triggers when user mentions memory, recall, remember, past sessions, knowledge graph,
  memoir, concepts, timeline, history, export, GUI, dashboard, or wants to save/search
  important context.
---

# rein Memory System (v0.20.0)

Use rein to persist and retrieve knowledge across sessions. rein runs as an MCP server
(28 tools) or via CLI (20+ commands). Features Neural Wiki GUI (`rein serve --gui`),
LLM-powered extraction (Gemini 3.1 Flash Lite or local models), temporal knowledge graph,
autonomous retrieval routing (6 strategies), query expansion (2-3 variants), LLM reranker
with strong-signal bypass, MMR diversity pass, memory evolution, transparent LLM proxy
(record-only) including **Codex subscription loopback** (`codexsubp` / `codexsubpws`
via `rein init --proxy` — ChatGPT-login WebSocket mirror with permessage-deflate, first-party
routing, ArtifactMirrorOnly lock for non-`/responses` routes, `/api/artifacts` readback),
concept name normalization and dedup, async memory pipeline with file-based queue and
background worker, project-scoped working set + always-on index for memory surfaces,
canonical-first recall with evidence previews, and adaptive learning loops (M1-M6 + A1).

## CLI Commands

### Core Memory
```bash
rein recall "query" [-t topic] [-l limit] [--from DATE] [--to DATE]  # Search memories
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
rein cleanup --all [--dry-run] [--async]     # One-click consolidate + dedup + adaptive refresh
rein gc [--dry-run]                          # Garbage collect weak STM
rein organize                                # Auto-link related memories
rein dedup-concepts                          # Merge duplicate concepts (case/separator variants)
rein upgrade [--topic X] [--dry-run]         # Upgrade old memories to knowledge graph
```

### System
```bash
rein serve [--compact] [--sse] [--proxy] [--gui]  # Start MCP server, proxy, or GUI
rein init [--dry-run]                        # Auto-configure MCP clients
rein config                                  # Show configuration
rein warmup                                  # Pre-compute embeddings
rein migrate [--from-qmd path] [--reindex]   # Import/reindex
rein worker memory                           # Drain async memory queue
rein doctor [--fix]                          # Health diagnostics and auto-repair
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

## MCP Tools (28)

### Core (13)
| Tool | Description |
|------|-------------|
| `rein_recall` | Semantic search with recency boost + time range filtering (from/to) |
| `rein_store` | Store with auto-dedup + evolution + knowledge graph |
| `rein_update` | Update memory content/importance |
| `rein_forget` | Delete by ID |
| `rein_list_topics` | List all topics |
| `rein_stats` | Store statistics |
| `rein_health` | Topic health check |
| `rein_consolidate` | Merge topic memories (supports --all, --pattern, --merge-variants) |
| `rein_dedup` | Scan/remove duplicates |
| `rein_cleanup` | One-click: consolidate + dedup + adaptive refresh |
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

### Temporal (2)
| Tool | Description |
|------|-------------|
| `rein_timeline` | Chronological view of episodes, concept changes, memory events |
| `rein_concept_history` | Revision history of a concept over time |

### Adaptive & Session (3)
| Tool | Description |
|------|-------------|
| `rein_adaptive_status` | Inspect learned alpha weights, cluster profiles, dedup thresholds, survival curve stats |
| `rein_feedback` | Report which recalled memories were used (drives M1 event sourcing) |
| `rein_ingest_session` | Ingest full session transcript through extraction pipeline |

## Key Features

- **LLM Extraction**: Gemini 3.1 Flash Lite or local models (Ollama/LM Studio/vLLM)
- **Query Expansion**: LLM rewrites query into 2-3 variants; results merged before fusion
- **LLM Reranker**: Gemini/OMLX rescoring with strong-signal bypass
- **MMR Diversity**: Maximal Marginal Relevance post-rerank pass
- **Adaptive Engine M1-M6+A1**: All learning loops closed — data drives fusion weights, decay curves, dedup thresholds, tier boundaries
- **M2 Per-cluster Alpha**: CC alpha learned globally, per-query-type, and per-cluster with Bayesian shrinkage
- **M3 Global Prior**: Kaplan-Meier survival curves per-cluster; global cold-start prior bridges new clusters
- **A1 Adaptive Dedup**: Per-cluster P90 similarity thresholds applied across all dedup paths
- **Knowledge Graph**: Auto-creates concepts + typed links from session transcripts
- **Memory Evolution**: New memories automatically refine or supersede similar old ones
- **Canonical-First Read Model**: recall returns canonicals by default; evidence expands on demand
- **Evidence-Aware Recall**: low-confidence recall uses evidence previews and second-stage evidence rerank
- **Autonomous Routing**: 6 query strategies (Episodic/Temporal/Preference/ExactKeyword/Semantic/Exploratory)
- **Hybrid CJK Dedup**: Chinese/Japanese/Korean lexical dedup uses jieba-rs plus character bigrams
- **Cluster-Aware Admission**: admission threshold and novelty scoring incorporate cluster strength and cold-start blending
- **Survival-Driven Promotion**: STM→LTM promotion uses survival curves when cluster data exists
- **Provenance-Preserving Merge**: temporal anchors and unique details never lost on dedup

## When to Use

- **Store**: After solving bugs, learning patterns, making architecture decisions
- **Recall**: Start of session, when context seems relevant, user asks about past work
- **Feedback**: After using recalled memories, report which ones were helpful
- **Ingest Session**: When you want to persist a full conversation to the knowledge base
- **Upgrade**: Convert old memories to knowledge graph format
- **GC**: Periodically clean up weak memories
