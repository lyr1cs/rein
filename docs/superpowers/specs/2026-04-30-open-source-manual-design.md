---
title: Open Source Manual Design
date: 2026-04-30
status: approved-for-planning
scope: docs
---

# Open Source Manual Design

## Goal

Rein needs a GitHub-ready public manual that explains how to install, operate,
and understand the system. The manual must cover the mechanism, code
architecture, algorithms, and references clearly enough for external users and
contributors. It must also be easy to extend because Rein is still under active
development.

The public manual will be written in English. Internal vault notes remain source
material only; public docs must not depend on Obsidian wikilinks, private vault
paths, or historical devlogs as user-facing navigation.

## Constraints

- Public docs live in the product repository: `source/rein/`.
- Git commands, validation, and release edits must be run from `source/rein/`.
- GitHub rendering is the target. Diagrams must use fenced Mermaid blocks.
- The manual should not become a single large file. It should be split by
  stable responsibilities so future changes touch small files.
- Bibliography entries must be verified against primary or authoritative
  sources and formatted in Nature-style numbered references.
- Unsupported or stale claims must be softened or removed rather than cited
  loosely.
- README should remain an entry point, not the full manual.

## Approaches Considered

### Option A: One large `docs/MANUAL.md`

This is simple to link and easy to publish on GitHub. It is also the easiest
format to let decay because installation, architecture, algorithms, and API
reference change at different rates.

Decision: rejected for maintainability.

### Option B: Split manual plus reference docs

Use `docs/manual/README.md` as the manual landing page and split long-lived
explanatory chapters from frequently changing reference tables.

Decision: accepted. This matches GitHub rendering, keeps diffs small, and lets
new features add chapters or reference rows without rewriting the narrative.

### Option C: Documentation site generator

Adopt mdBook, Docusaurus, or MkDocs immediately.

Decision: deferred. A static Markdown structure is enough for the first
open-source release and avoids adding a docs build pipeline. The proposed
layout can be migrated to a site generator later.

## Public Documentation Layout

```text
source/rein/
├── README.md
└── docs/
    ├── manual/
    │   ├── README.md
    │   ├── 01-overview.md
    │   ├── 02-installation.md
    │   ├── 03-mechanism.md
    │   ├── 04-architecture.md
    │   ├── 05-algorithms.md
    │   ├── 06-operations.md
    │   └── 07-security.md
    └── reference/
        ├── cli.md
        ├── mcp-tools.md
        ├── rest-api.md
        ├── config.md
        └── bibliography.md
```

`docs/manual/README.md` is the GitHub-rendered manual home. `README.md` at the
repo root should link to it near the top.

## Chapter Responsibilities

### `docs/manual/README.md`

Purpose: table of contents, version note, and reading paths.

Content:
- What Rein is in one paragraph.
- Current release baseline.
- "Start here" links for users, operators, contributors, and architecture
  readers.
- A note that reference tables are under `docs/reference/`.

### `01-overview.md`

Purpose: product framing.

Content:
- Local-first memory for AI agents.
- What Rein is not: not SaaS, not sync, not a multi-tenant cloud service.
- Public surfaces: CLI, MCP, REST, GUI, hooks, and record-only proxy.
- Core principles: explicit recall, canonical-first reads, provenance
  preservation, adaptive feedback.

### `02-installation.md`

Purpose: deployable instructions.

Content:
- Source install: CLI-only and GUI-enabled.
- Install script behavior.
- Docker and Docker Compose.
- Release binary section only if asset names and platform support are verified.
- Required and optional environment variables.
- Auth requirements for HTTP/SSE, GUI, and proxy.
- `rein init`, `rein doctor`, `rein dashboard`, and smoke-test commands.

Known corrections to include:
- Use `cargo install --path crates/rein --locked`.
- Docker requires `REIN_HTTP_TOKEN`.
- GUI/HTTP/proxy quick starts must mention token setup or explicit loopback
  unauth opt-in.

### `03-mechanism.md`

Purpose: explain how Rein works end to end.

Content:
- Ingestion lifecycle.
- Storage and canonical/evidence model.
- Recall lifecycle.
- Adaptive slow-channel lifecycle.
- ARS Cap A/B/C as opt-in projections, not source of truth.

Required diagrams:
- End-to-end flow.
- Recall pipeline.
- Feedback and adaptive loop.
- ARS synthesis and summary loop.

### `04-architecture.md`

Purpose: map code architecture to runtime behavior.

Content:
- Binary/library split.
- `rein-macros` and the `#[op]` registry.
- CLI/MCP/REST shared operation model.
- SQLite store, side indexes, and migrations.
- Search, extract, proxy, GUI, and adaptive modules.
- Major invariants that contributors must preserve.

### `05-algorithms.md`

Purpose: explain algorithmic choices with citations.

Content:
- Query classification.
- Tantivy BM25 and FTS5 fallback.
- HNSW vector search and sqlite-vec fallback.
- KG land-and-expand retrieval.
- RRF and convex combination fusion.
- MMR diversity reranking.
- Kaplan-Meier survival curves and Ebbinghaus fallback.
- HDBSCAN clustering.
- Dedup: lexical, CJK tokenization, embedding candidates, gray-zone LLM
  verdicts, evidence ledger.
- Adaptive learning: M1-M6, A1 thresholds, runtime judge, useful-rate gates.

Claims must be tied either to code modules or bibliography entries. Claims that
are only aspirational should be marked as roadmap or omitted.

### `06-operations.md`

Purpose: day-to-day operator guide.

Content:
- Store, recall, update, forget.
- Cleanup, dedup, consolidate, gc, organize.
- Workers and queues.
- GUI operations.
- Proxy operation.
- Diagnostics and repair workflow.

### `07-security.md`

Purpose: security and privacy model.

Content:
- Local-first storage boundary.
- Bearer token and session auth model.
- Default-deny unauthenticated loopback behavior.
- Host/Origin guard.
- Record-only proxy guarantee.
- LLM feature flags, prompt caps, and opt-in behavior.
- AGPL network-use notice.

## Reference Responsibilities

### `reference/cli.md`

Generated or manually audited CLI command table. Should separate legacy clap
commands from inventory-backed commands only if useful to contributors; normal
users should see one command list.

### `reference/mcp-tools.md`

A complete table of the 38 production MCP tools. Test-only tools must be
excluded.

### `reference/rest-api.md`

REST route inventory, auth class, method, parameters, and response notes.

### `reference/config.md`

Config sections, key environment variables, defaults, and security-sensitive
fields. Must document `[llm]` inheritance and the opt-in defaults for ARS and
judge features.

### `reference/bibliography.md`

Nature-style numbered bibliography. Split into:
- Research references.
- Software and official documentation references.
- Background or inspiration references.

Each entry must include DOI or stable URL where available. Examples of verified
core references include Kaplan-Meier survival estimation, RRF, HDBSCAN, HNSW,
BM25, MMR, SemDeDup, SQLite FTS5, and Gemini embedding documentation.

## Mermaid Diagram Rules

- Use GitHub-supported Mermaid fenced blocks only.
- Prefer `flowchart TD` for system pipelines.
- Use `sequenceDiagram` for request/response or CAS/claim lifecycles.
- Use `erDiagram` only for simplified public schema views.
- Keep node labels short enough to render on GitHub.
- Avoid Obsidian-only embeds or local image links.

Initial diagrams to add:
- System surface map: user/agent -> CLI/MCP/REST/GUI/proxy -> core -> store.
- Recall pipeline: classify -> expand -> FTS/vector/KG -> fusion -> rerank ->
  canonical/evidence output.
- Adaptive engine: feedback events -> M1-M6/A1 -> adaptive snapshot.
- Dedup decision tree: candidates -> lexical/vector score -> action -> evidence
  and ledger.
- Cold archive CAS: claim -> LLM summary -> invariants -> CAS persist.

## Citation Policy

Use numbered Nature-style references:

```text
1. Kaplan, E. L. & Meier, P. Nonparametric estimation from incomplete
   observations. J. Am. Stat. Assoc. 53, 457-481 (1958).
   https://doi.org/10.1080/01621459.1958.10501452
```

Rules:
- Prefer primary publisher pages, DOI records, arXiv, OpenReview, or official
  documentation.
- Do not cite unverified local claims as literature.
- If a source is inspiration rather than implemented behavior, label it as
  background.
- Software documentation belongs in the software/documentation section.
- If a claim changes over time, such as model ranking or API model availability,
  cite official current documentation or avoid ranking language.

## Stale Claims To Fix During Implementation

- README says both 38 and 32 MCP tools. Use 38 production tools.
- README architecture diagram still says 32 MCP tools and 8 GUI pages.
- README CLI table is incomplete and includes stale `cleanup --async` wording.
- README should not claim FTS5 unicode61 is the CJK segmentation solution;
  Tantivy plus `jieba-rs` and CJK bigrams are the CJK path, while FTS5 is a
  fallback.
- "Zero subjective parameters" should be softened to "reduces fixed parameters
  through adaptive feedback" because the code still has bootstrap constants,
  gates, clamps, and fallback thresholds.
- "Causal inference" for M6 should be softened unless the code implements a
  stronger causal method; current behavior is randomized threshold exploration
  and co-recall heuristics.
- A-MAC 2026, TA-Mem 2026, and MemR3 2025 require verified bibliography entries
  or softer wording.
- `gemini-embedding-001` dimensionality can be cited from official Google docs;
  leaderboard/ranking claims need an explicit current source or should be
  removed.

## Maintenance Model

The docs should support continuous development:

- Add new stable concepts as new manual chapters only when they affect the
  public mental model.
- Add new commands, tools, routes, and config keys to reference docs.
- Keep release history in release notes, not in the manual.
- Add a short "Updated for version" line to each manual chapter.
- When a chapter grows too large, split by responsibility rather than chronology.
- Avoid duplicating long tables in both README and reference files.
- Treat `docs/reference/bibliography.md` as the single source of citation
  details.

## First Implementation Scope

The first documentation pass should:

1. Create the manual and reference skeleton.
2. Move durable explanatory content out of README into manual chapters.
3. Fix README drift that would mislead GitHub users.
4. Add Mermaid diagrams for the core mechanism and architecture.
5. Add verified bibliography entries for implemented core algorithms.
6. Add complete MCP and CLI reference tables from the current source.
7. Leave REST and full config reference as accurate but compact initial tables
   if time is constrained.

## Non-Goals

- Do not introduce a docs site generator in the first pass.
- Do not expose private vault paths, internal agent notes, or archived devlogs
  as public navigation.
- Do not rewrite all historical release notes.
- Do not document test-only routes or private remediation workflow as public API.
- Do not present future federation or v0.28 work as shipped behavior.

## Approval

The user approved:
- English-only public manual.
- GitHub-oriented location under `source/rein/docs/`.
- Mermaid diagrams for mechanism explanations.
- Verified Nature-style bibliography using Exa/Perplexity or equivalent
  authoritative web sources.
- A modular format that is easy to update as Rein continues development.
