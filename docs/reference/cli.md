# CLI Reference

`rein --help` and each subcommand's `--help` output are authoritative for exact
flag spelling. This table is a compact public index of the current legacy Clap
commands and inventory-backed operation commands.

## Legacy Commands

| Command | Key arguments | Purpose |
|---|---|---|
| `rein serve` | `--compact`, `--sse`, `--proxy`, `--gui` | Start MCP stdio, HTTP/SSE, proxy, or GUI surfaces. |
| `rein ingest` | `--content`, `--file`, `--json-file`, `--asynchronous`, `--agent-label`, `--subagent` | Ingest a session or transcript through extraction. |
| `rein init` | `--dry-run`, `--proxy` | Generate MCP client config and optional proxy shell helpers. |
| `rein export` | `--format`, `--topic`, `--output` | Export memories as JSON, Markdown, or CSV. |
| `rein upgrade` | `--topic`, `--dry-run` | Extract knowledge-graph concepts and links from old memories. |
| `rein warmup` | `--trust-existing-vectors`, `--reembed-all` | Make every live memory durable in the vector store: fill missing `vec_memories` rows from the embedding cache or the provider, then rebuild HNSW. Rows created before provenance stamping are untrusted until attested with `--trust-existing-vectors`; `--reembed-all` re-embeds every live row under the configured model and replaces the table atomically (each row revalidated at publish). |
| `rein worker memory` | none | Drain the async memory queue. |
| `rein worker dedup-queue` | none | Drain queued store-time dedup jobs. |
| `rein worker cleanup` | topic selectors, `--all`, `--exact-topics`, `--dry-run` | Run a detached cleanup pass. |
| `rein worker cleanup-queue` | none | Drain queued cleanup jobs. |
| `rein worker merge-refinement-queue` | none | Drain queued post-merge refinement jobs. |
| `rein hook session-start` | Codex `SessionStart` payload on stdin | Optionally inject bounded project memory context. |
| `rein hook pre` | Codex `PreToolUse` payload on stdin | Run deny-only conservative guardrails before tool execution. |
| `rein hook permission` | Codex `PermissionRequest` payload on stdin | Run deny-only conservative guardrails for permission prompts. |
| `rein hook post` | hook payload on stdin | Extract facts from tool output. |
| `rein hook compact` | hook payload on stdin | Extract context before compaction. |
| `rein hook prompt` | Codex `UserPromptSubmit` payload on stdin | Optionally inject bounded relevant memory context. |
| `rein hook stop` | hook payload on stdin | Persist session summary at conversation end. |
| `rein dashboard` | none | Print local service status. |
| `rein gui on/off` | `on`, `off` | Manage the GUI service. |
| `rein proxy on/off` | `on`, `off` | Manage the record-only proxy service. |
| `rein judge-calibrate-cron` | `--verbose` | Run the judge calibration cron over archived samples. |
| `rein trust-measurement` | none | Show ARS release gate, eval gates, index consistency, background observability, and active-learning status. |

## Inventory-Backed Commands

| Command | Key arguments | Purpose |
|---|---|---|
| `rein store` | topic, content, `--importance`, `--keywords` | Store a memory. |
| `rein recall` | query, `--topic`, `--keyword`, `--limit`, `--from`, `--to`, `--expand`, `--synthesize` | Recall memories through the multi-channel search pipeline. |
| `rein update` | id, content, `--importance` | Update memory content and optional importance. |
| `rein forget` | id | Delete a memory. |
| `rein topics` | none | List topics. |
| `rein recent` | `--limit` | Show recent memories. |
| `rein timeline` | `--from`, `--to`, `--limit` | Show timeline events. |
| `rein concept-history` | memoir, concept name, `--limit` | Show concept revision history. |
| `rein stats` | none | Show store statistics. |
| `rein health` | optional topic | Show topic and system health. |
| `rein doctor` | `--json`, `--network`, `--fix` | Run diagnostics and optional local repairs. |
| `rein config` | none | Print non-secret effective configuration. |
| `rein adaptive-status` | none | Show adaptive engine state. |
| `rein canonicals` | `--limit` | List canonical memories. |
| `rein evidence` | canonical id, `--limit` | List evidence for a canonical memory. |
| `rein gc` | `--dry-run`, `--threshold` | Apply decay and prune weak STM rows. |
| `rein dedup` | `--dry-run`, `--merge-variants` | Deduplicate memories. |
| `rein dedup-concepts` | none | Deduplicate concepts in memoirs. |
| `rein dedup-log` | `--canonical`, `--operator`, `--limit` | Show recent dedup decisions. |
| `rein organize` | `--max-links` | Create related-memory links. |
| `rein intelligent-merge-try` | existing id, incoming id | Dry-run an LLM gray-zone merge verdict. |
| `rein migrate` | `--from-qmd`, `--reindex` | Import QMD data or rebuild embeddings. |
| `rein consolidate` | topic selectors, `--all`, `--merge-variants`, `--summary`, `--dry-run` | Consolidate topic groups. |
| `rein cleanup` | topic selectors, `--all`, `--exact-topics`, `--dry-run` | Run scoped cleanup; the hidden deprecated async flag is replaced by `rein worker cleanup`. |
| `rein resummerize` | `--dry-run`, `--canonical-id` | Recompress flagged canonicals. |
| `rein archive-summary-refresh` | memory id, `--force` | Regenerate one cold-tier archival summary. |
