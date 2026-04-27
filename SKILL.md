---
name: rein
description: |
  Multi-source cross-validated memory MCP server for AI agents. Provides 38 MCP tools and a CLI for persistent memory across sessions, with knowledge graph, temporal links, and adaptive learning loops (M1-M6 + A1 + ARS Cap A/B/C).

  Trigger this skill whenever the user (in any language, including English / 中文) wants to save, store, recall, search, or recover information across sessions, or mentions any of:
  memory / 记忆, recall / 召回 / 回忆, remember / 记住, save / 存储 / 保存, search past / 搜索历史,
  knowledge graph / 知识图谱 / 概念 / concepts, memoir / 知识库, timeline / 时间线 / 历史, episode / session / 会话,
  rein, rein_store, rein_recall, MCP memory, GUI / dashboard / Neural Wiki,
  past sessions / 之前的工作 / 上次说过, "what do I know about X" / "我对 X 了解什么".

  Use rein_store for new facts/decisions/preferences, rein_recall for retrieval,
  rein_memoir_* for the knowledge graph, rein_timeline / rein_concept_history for temporal queries,
  rein_feedback after acting on recall, and rein_adaptive_status to inspect learning state.
allowed-tools:
  - Bash
---

# rein Memory System (v0.27.1)

Use rein to persist and retrieve knowledge across sessions. rein runs as an MCP server (38 tools) or via CLI. Storage is `~/.rein/memories.db` (SQLite + FTS5 + sqlite-vec); the same DB is shared across every rein client (Claude Desktop, Claude Code, CLI, REST). Embeddings via Gemini 3.1 (or local Ollama / OMLX). Optional Neural Wiki GUI on `:8680`.

## When to use rein

| User intent | Tool |
|---|---|
| "记住这件事" / "save this" / "存到记忆里" | `rein_store` |
| "我们之前说过什么" / "recall X" / "search past" | `rein_recall` |
| "X 是什么时候改的" / temporal questions | `rein_recall` with `from`/`to`, or `rein_timeline` |
| "X 这个概念演化过吗" / concept history | `rein_concept_history` |
| Build / inspect knowledge graph | `rein_memoir_*` (10 tools) |
| Report which recalled memory was useful | `rein_feedback` |
| Inspect learning state | `rein_adaptive_status` |
| Refresh archival summaries (cold tier) | `rein_archive_summary_refresh` |

Default to `rein_recall` at the start of a session when the user references past work. Default to `rein_store` after solving bugs, making architecture decisions, capturing user preferences, or learning project facts.

## CLI Commands

### Core memory
```bash
rein recall "query" [-t topic] [-l limit] [--from DATE] [--to DATE]  # search
rein store -t <topic> -c <content> -I <importance> [-k kw1,kw2]
rein update <id> -c <new_content> [-I <importance>]
rein forget <id>
rein topics
rein recent [-l 20]
rein stats
rein health [topic]
```

### Maintenance
```bash
rein consolidate <topic> -s "summary"    # merge topic into one memory
rein dedup [--dry-run]                   # scan/remove duplicates
rein cleanup --all [--dry-run] [--async] # consolidate + dedup + adaptive refresh
rein gc [--dry-run]                      # garbage collect weak STM
rein organize                            # auto-link related memories
rein dedup-concepts                      # merge duplicate concepts
rein doctor [--fix]                      # health check + auto-repair
```

### System / runtime
```bash
rein serve [--sse] [--proxy] [--gui]    # MCP stdio / HTTP / proxy / Neural Wiki GUI
rein init [--dry-run]                   # auto-configure MCP clients
rein config                             # show config
rein warmup                             # pre-compute embeddings
rein worker memory                      # drain async memory queue
```

### Hooks (Claude Code only — wired in `~/.claude/settings.json`)
```bash
rein hook post     # PostToolUse  → LLM extraction → store (async)
rein hook compact  # PreCompact   → context snapshot
rein hook prompt   # UserPromptSubmit (compatibility no-op)
rein hook stop     # Stop         → session summary + Episode
```

### Proxy (record-only LLM mirror)
```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8690 claude
OPENAI_BASE_URL=http://127.0.0.1:8690 codex
```

## MCP Tools (38, as of v0.27.1)

### Core memory (13)
`rein_recall`, `rein_store`, `rein_update`, `rein_forget`, `rein_list_topics`, `rein_stats`, `rein_health`, `rein_consolidate`, `rein_dedup`, `rein_cleanup`, `rein_recent`, `rein_gc`, `rein_organize`

### Knowledge graph (10)
`rein_memoir_create`, `rein_memoir_list`, `rein_memoir_show`, `rein_memoir_add_concept`, `rein_memoir_refine`, `rein_memoir_search`, `rein_memoir_search_all`, `rein_memoir_link`, `rein_memoir_inspect`, `rein_memoir_export`

### Temporal (2)
`rein_timeline`, `rein_concept_history`

### Adaptive / session / ARS (13)
`rein_adaptive_status`, `rein_feedback`, `rein_ingest_session`, `rein_concept_state`, `rein_archive_summary_refresh`, `rein_feedback_concept_summary` (v0.27 ARS Cap A), `rein_judge_synthesis` + `rein_judge_concept_summary` (v0.27.1 — manually trigger an LLM judge call against a recent synthesis / concept-summary so MCP-only callers can prime ARS adaptive feedback without GUI dwell/click traffic), and 5 more registered via `#[op]`.

## Architecture highlights (v0.27.0)

- **Search pipeline**: 3-channel waterfall (Tantivy BM25 + usearch HNSW + Gemini embeddings) → RRF/CC fusion with adaptive `cc_alpha` → multi-feature reranker → optional LLM rerank → MMR diversity → Ebbinghaus / Kaplan-Meier decay
- **Autonomous routing**: 6 query strategies (Episodic / Temporal / Preference / ExactKeyword / Semantic / Exploratory), rule-based, 0 LLM calls
- **Adaptive engine M1-M6 + A1**: feedback event sourcing → counterfactual alpha optimization → per-cluster survival decay → HDBSCAN clustering → hot/warm/cold tiering → randomized exploration → per-cluster dedup thresholds
- **ARS Cap A/B/C**: concept living-summary, recall-time LLM synthesis (`synthesize=true`), cold-tier archival summary — all opt-in via `[ars]` config
- **v0.27 dedup**: triple extraction (S, P, O), N-memory merge (`MergeIntoMany`) with savepoint atomicity, temporal supersede with `2026年` CJK form support
- **Storage**: per-request SQLite connection model + WAL + FULL_MUTEX (multi-process safe)
- **Hybrid CJK**: jieba-rs + character bigrams (avoids the `is_alphanumeric` collapse-Chinese-to-mega-token trap)

## Tips for Claude

1. **Don't try to spawn rein from inside Cowork's VM** — the rein binary lives at `~/.cargo/bin/rein` on the host and the Cowork VM doesn't mount it. Use rein from the main Claude Desktop UI or Claude Code CLI on the host instead.
2. **All ARS feature flags default off**. If user says "concept summary 没出来", check `~/.rein/config.toml` for `[ars].concept_summary_enabled = true`.
3. **Memory is shared**: rein's SQLite DB at `~/.rein/memories.db` is a single source of truth across every client, so memory written from Claude Desktop is immediately visible from Claude Code and vice versa.
4. **Two-tier hooks for Claude Code only**: the 4 `rein hook *` commands write to `~/.rein/memories.db` automatically. Claude Desktop has no equivalent — there you must call `rein_store` explicitly.
5. **Auto-memory markdown** at `~/.claude/projects/.../memory/MEMORY.md` is independent from rein's DB unless `sync/auto_memory.rs` is wired up.

## Common pitfalls

- LLM JSON output may be wrapped in code fences — prefer `rein_recall`'s built-in handling
- `rein doctor` reports `Overall: degraded` mainly on doc-version drift — usually benign, check `mcp_registry` line
- Cold-tier memories excluded from non-Exploratory queries (M5 filter) — switch to `query_type=Exploratory` if needed
