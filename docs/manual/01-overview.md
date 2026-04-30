# Overview

Rein is a local-first memory system for AI agents. It stores durable facts,
decisions, preferences, observations, and session context in a local SQLite
database, then exposes explicit recall through CLI, MCP, REST, GUI, hooks, and a
record-only proxy. Its goal is to help agents carry useful context across
sessions without turning memory into an opaque remote service.

Rein is also self-adaptive. Feedback events, access patterns, dedup decisions,
and recall outcomes feed a slow maintenance channel that can update retrieval
weights, survival curves, tier boundaries, clustering state, and dedup
thresholds. Bootstrap constants still exist for cold start and safety, but the
system is designed to reduce fixed parameters as real usage data accumulates.

## What Rein Is

- A single Rust binary for local memory storage, recall, maintenance, and
  service hosting.
- A memory MCP server for tools that can call `rein_store`, `rein_recall`, and
  related operations.
- A local SQLite-backed knowledge base with full-text, vector, graph, temporal,
  canonical, and evidence-aware read paths.
- A provenance-preserving dedup system that keeps original observations as
  evidence when memories are merged or superseded.
- An adaptive retrieval system that learns from events instead of relying only
  on static ranking parameters.

## What Rein Is Not

- Rein is not a hosted SaaS product.
- Rein is not a multi-tenant cloud memory service.
- Rein is not a general file sync system.
- Rein is not a replacement for source control, issue trackers, or project
  documentation.
- Rein is not designed to silently inject memory into every prompt. Recall is a
  deliberate operation exposed through public surfaces.

## Public Surfaces

### CLI

The `rein` command stores, recalls, updates, forgets, consolidates, deduplicates,
organizes, diagnoses, and serves memories. It is the operator-friendly surface
for local workflows and service management.

### MCP

`rein serve` exposes production MCP tools for AI clients. The MCP surface uses
the same operation registry as the CLI and REST layers, so tool behavior is not
maintained as a separate implementation.

### REST

The REST API powers the GUI and supports local HTTP integrations. Sensitive
routes are token-protected when HTTP access is configured for real service use.

### GUI

The embedded Neural Wiki GUI provides dashboard, memory, adaptive, graph,
timeline, artifact, and settings views when Rein is built with the GUI feature.

### Hooks

Hook commands support session and tool-output extraction flows. They are
intended for clients that can invoke post-action, compaction, prompt, or stop
hooks. The prompt hook is retained as a compatibility no-op; Rein no longer
auto-injects prompt memory.

### Record-Only Proxy

The proxy records request and response context for extraction without modifying
the upstream request. Proxy operation is explicit and authenticated separately
from normal HTTP service access.

## Design Principles

### Local-First Storage

The primary store is a local SQLite database. Optional external providers can be
used for embeddings, LLM extraction, reranking, synthesis, or validation, but
the durable memory boundary is local by default.

### Explicit Recall

Agents and users ask for memory through `rein_recall`, CLI recall, REST recall,
or GUI views. Rein avoids hidden prompt injection as a core behavior because
memory should be inspectable and attributable.

### Canonical-First Reads

Recall and list views prefer canonical memories: stable rows representing the
current durable fact or state. Raw observations remain available as evidence,
but canonical memories are the default read surface.

### Provenance Preservation

Dedup and consolidation preserve source observations in `memory_evidence` and
record merge decisions in an append-only ledger. A memory can be compressed,
merged, or superseded without losing the audit trail of how the canonical row
was formed.

### Multi-Channel Retrieval

Recall combines full-text search, vector search, and knowledge graph or episode
signals. Results can be fused, decay-weighted, reranked, diversified, collapsed
to canonicals, and enriched with evidence previews.

### Adaptive Feedback

Feedback events are not just telemetry. They are inputs for the slow channel
that updates retrieval weights, survival curves, tiering, cluster profiles, and
dedup thresholds.

### Opt-In LLM Work

LLM-dependent features are controlled through configuration and runtime flags.
Rein has local and rule-based fallbacks for many paths, and ARS synthesis,
concept summaries, archive summaries, and runtime judge features are opt-in
projections rather than the source of truth.

## Current Limits

- Rein is under active development, so command and API reference tables should
  be checked against the current binary before release notes or automation are
  generated.
- The default database is local and single-user oriented. Shared or multi-tenant
  deployments require external operational controls beyond Rein itself.
- Several advanced features depend on optional LLM or embedding providers. When
  those providers are unavailable, Rein degrades to configured fallbacks or skips
  the optional projection.
- Adaptive learning needs enough events to become meaningful. Cold-start
  behavior still uses conservative defaults and bootstrap thresholds.
- Side indexes such as Tantivy and HNSW are rebuildable accelerators. SQLite is
  the durable store; side-index failures should degrade search quality rather
  than corrupt memory state.
