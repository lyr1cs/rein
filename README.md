# rein

> Multi-source cross-validated memory for AI agents

<p align="center">
  <a href="#english">English</a> | <a href="#中文">中文</a>
</p>

---

## English

rein is a self-adaptive memory system for AI coding agents. It stores, recalls, and manages memories across sessions with embedding-based semantic dedup, data-driven decay (Kaplan-Meier survival curves), and a fully closed self-learning loop that replaces fixed parameters with learned values.

### Features

| Feature | Description |
|---------|-------------|
| **24 MCP tools** | 12 core memory tools + 10 knowledge graph tools + 2 temporal tools |
| **Self-adaptive engine** | M1-M6: all learning loops closed — data drives fusion weights, decay curves, dedup thresholds, and tier boundaries |
| **Counterfactual alpha learning** | Replays past recalls to find optimal CC fusion weights per query type (M2) |
| **Per-cluster survival decay** | Kaplan-Meier curves replace fixed Ebbinghaus when sufficient data exists (M3) |
| **HDBSCAN clustering** | Pure Rust semantic clustering with sampling for large datasets (M4) |
| **Hot/Warm/Cold tiering** | Streaming quantile estimator + cold_archive migration (M5) |
| **Adaptive dedup thresholds** | Per-cluster P90 similarity thresholds (SemDeDup-inspired, M6/A1) |
| **Provenance-preserving dedup** | Merges preserve temporal anchors and unique details instead of hard-deleting |
| **Embedding semantic dedup** | Catches paraphrases Jaccard misses, runs in GC slow channel (zero hot-path cost) |
| **Temporal knowledge graph** | Memoir / Concept / ConceptLink with 9 relation types, revision history, episode nodes, temporal validity windows, BFS traversal (skips expired links) |
| **Autonomous retrieval routing** | Query classifier routes to Temporal/ExactKeyword/Semantic/Exploratory strategy |
| **OMLX local embedding** | Optional local embedding backend via EmbedderKind enum dispatch (Google / OMLX) |
| **Dual-layer decay** | LTM / STM layers with KM survival curves (data-driven) or Ebbinghaus (cold-start) |
| **Dual-path search** | FTS (Tantivy BM25 → FTS5 fallback) + Vector (HNSW cache → API embed) → RRF/CC fusion |
| **Multi-source cross-validation** | 3 sources (local, hook-extracted, Supermemory) with confidence scoring |
| **RRF / CC fusion** | Reciprocal Rank Fusion or Convex Combination (Bruch 2023), with learned alpha weights |
| **Multi-factor admission** | A-MAC 2026 inspired: llm_conf + novelty + type_prior + recency scoring |
| **Semantic chunking** | Heading / paragraph / sentence splitting with metadata-prefixed embeddings |
| **FTS5 unicode61 tokenizer** | Full-text search with CJK support, sub-millisecond latency |
| **Supermemory v4 API** | Hybrid search via `api.supermemory.ai/v4/search` for cross-validation |
| **Zero local models** | No GPU required by default; optional OMLX local backend |
| **~2-5 MB footprint** | Single SQLite file with FTS5 + sqlite-vec |
| **gemini-embedding-001** | MTEB #1 model (68.32), 3072 dimensions |
| **20+ CLI commands** | Everything the MCP tools do, plus init, config, migrate, hooks, recent, gc, organize, upgrade |
| **Auto-configure** | `rein init` detects and configures 8 MCP clients automatically |
| **Remote access** | HTTP / SSE transport with bearer token authentication |

### Installation

#### From source

```bash
git clone https://github.com/lyr1cs/rein.git
cd rein
cargo install --path .
```

Or use the install script:

```bash
./scripts/install.sh
```

#### Prerequisites

- Rust toolchain (1.75+)
- A Gemini API key (free tier: 1500 req/day)

### Quick Start

```bash
# 1. Set your API key
export GEMINI_API_KEY="your-key-here"

# 2. Auto-configure all detected MCP clients
rein init

# 3. Start the MCP server (usually done by your client)
rein serve
```

### CLI Reference

| Command | Description | Example |
|---------|-------------|---------|
| `serve` | Start MCP server (stdio or SSE) | `rein serve [--compact] [--sse]` |
| `store` | Store a memory | `rein store -t debug -c "OOM fix" -I high -k oom,memory` |
| `recall` | Search memories | `rein recall "connection pool" -t debug -l 5` |
| `forget` | Delete a memory by ID | `rein forget 01J...` |
| `update` | Update memory content | `rein update 01J... -c "new content" -I critical` |
| `topics` | List all topics | `rein topics` |
| `stats` | Show store statistics | `rein stats` |
| `health` | Check topic health | `rein health [topic]` |
| `consolidate` | Merge topic into one memory | `rein consolidate debug -s "summary"` |
| `dedup` | Scan / remove duplicates | `rein dedup [--dry-run]` |
| `migrate` | Import from QMD / reindex | `rein migrate [--from-qmd path] [--reindex]` |
| `init` | Auto-configure MCP clients | `rein init [--dry-run]` |
| `config` | Show current configuration | `rein config` |
| `recent` | Show most recent memories | `rein recent [-l 20]` |
| `gc` | Garbage collect weak STM memories | `rein gc [--dry-run]` |
| `organize` | Auto-link related memories | `rein organize` |
| `upgrade` | Upgrade old memories to knowledge graph | `rein upgrade [--topic X] [--dry-run]` |
| `hook post` | Extract facts from tool output | `rein hook post` |
| `hook compact` | Save context before compaction | `rein hook compact` |
| `hook prompt` | Compatibility no-op for UserPromptSubmit | `rein hook prompt` |
| `hook stop` | Full knowledge extraction on session end | `rein hook stop` |

### MCP Tools

When running as an MCP server (`rein serve`), 24 tools are exposed.

#### Core Tools (12)

| Tool | Parameters | Description |
|------|-----------|-------------|
| `rein_recall` | `query`, `topic?`, `keyword?`, `limit?`, `from?`, `to?` | Semantic search with optional time range |
| `rein_store` | `topic`, `content`, `importance?`, `keywords?` | Store a new memory (auto-dedup) |
| `rein_update` | `id`, `content`, `importance?` | Update an existing memory |
| `rein_forget` | `id` | Delete a memory by ID |
| `rein_list_topics` | *(none)* | List all memory topics |
| `rein_stats` | *(none)* | Total count, LTM/STM breakdown, avg strength |
| `rein_health` | `topic?` | Stale count, avg strength, consolidation hints |
| `rein_consolidate` | `topic`, `summary` | Merge all memories in a topic into one |
| `rein_dedup` | `dry_run?` | Scan for and remove duplicate memories |
| `rein_recent` | `limit?` | List most recently created memories |
| `rein_gc` | `dry_run?` | Garbage collect weak STM memories |
| `rein_organize` | `max_links?` | Auto-link related memories |

#### Knowledge Graph Tools (10)

| Tool | Parameters | Description |
|------|-----------|-------------|
| `rein_memoir_create` | `name`, `description?` | Create a knowledge container |
| `rein_memoir_list` | *(none)* | List all memoirs |
| `rein_memoir_show` | `name` | Show memoir details + concepts |
| `rein_memoir_add_concept` | `memoir`, `name`, `definition`, `labels?` | Add a knowledge node |
| `rein_memoir_refine` | `memoir`, `name`, `definition` | Update concept, boost confidence |
| `rein_memoir_search` | `memoir`, `query`, `limit?` | FTS search within a memoir |
| `rein_memoir_search_all` | `query`, `limit?` | Search across all memoirs |
| `rein_memoir_link` | `memoir`, `from`, `to`, `relation` | Link two concepts |
| `rein_memoir_inspect` | `memoir`, `name`, `depth?` | BFS neighborhood traversal |
| `rein_memoir_export` | `memoir`, `format?` | Export graph (json / ascii / dot) |

#### Temporal Tools (2)

| Tool | Parameters | Description |
|------|-----------|-------------|
| `rein_timeline` | `from?`, `to?`, `limit?` | Chronological timeline of episodes, concept changes, and memory events |
| `rein_concept_history` | `memoir`, `name`, `limit?` | Revision history of a concept: when/how it changed over time |

#### Knowledge Graph Relation Types

`part_of`, `depends_on`, `related_to`, `contradicts`, `refines`, `alternative_to`, `caused_by`, `instance_of`, `superseded_by`

### LLM Extraction (v0.3)

rein uses LLM (Gemini 3.1 Flash Lite or local models via OMLX) for structured memory extraction. The hook system automatically builds a knowledge graph from coding sessions.

**Architecture:**
- `hook_post` — local pattern extraction (crash safety net) + buffer to session file
- `hook_compact` — record compact context for async extraction
- `hook_stop` — queue full session distillation: memories + concepts + links + episode summary
- `hook_prompt` — compatibility no-op (automatic prompt injection removed)

**Upgrade old memories:**
```bash
rein upgrade --dry-run    # preview
rein upgrade              # convert all old memories to knowledge graph
rein upgrade --topic debug  # convert specific topic only
```

**Configuration:**
```toml
[extract]
provider = "google"    # or "omlx" or "none"

[extract.google]
model = "gemini-3.1-flash-lite-preview"
max_input_chars = 0    # 0 = no truncation (1M token model)

[extract.omlx]
endpoint = "http://localhost:11434/v1"  # Ollama, LM Studio, vLLM, etc.
model = "default"
max_input_chars = 16000
```

### Self-Learning Quality System (v0.3.0)

rein automatically learns which memories are useful and which are noise, without human parameter tuning.

**How it works:**
1. LLM assigns `quality_confidence` (0-1) at extraction time — zero extra API cost
2. System tracks recall-then-access patterns to classify memories as "good" (used) or "bad" (recalled but unused)
3. Feature weights auto-adjust from data: utility, novelty, connectivity, recency
4. Adaptive admission threshold rises when recent quality is low, relaxes when high
5. GC prunes low-quality concepts whose source memories are recalled 5+ times but never accessed

**No manual tuning needed** — cold-starts with LLM judgment, data gradually takes over.

Based on: ICLR 2026 Admission Control, PropMem (Prosus), FActScore, MACLA Bayesian posteriors.

### Temporal Knowledge Graph (v0.4.0)

rein now tracks **when** knowledge changes, not just what the current state is. Inspired by Zep/Graphiti 2025.

**Capabilities:**
- **Concept revision history** — every `refine_concept` auto-snapshots the old state before overwriting
- **Episode nodes** — each session creates an Episode linking to concepts and memories touched
- **Temporal link validity** — ConceptLink has `valid_from`/`valid_until` windows; expired links are skipped in BFS
- **Contradiction detection** — when a new definition differs significantly (sim < 0.3), old outgoing links are expired
- **Temporal recall** — `rein_recall` supports `from`/`to` date params for time-range filtering
- **Timeline view** — `rein_timeline` shows chronological events (episodes, concept changes, memory creation)
- **Concept history** — `rein_concept_history` shows how a concept's definition evolved over time

**Example queries enabled:**
- "What changed last week?" → `rein_timeline --from 2026-03-19 --to 2026-03-26`
- "When did concept X change?" → `rein_concept_history --memoir rust --name ownership`
- "What did I know about Y before March?" → `rein_recall "Y" --to 2026-03-01`

### Autonomous Retrieval Routing (v0.4.0)

rein automatically classifies queries and routes them to the optimal search strategy — no configuration needed.

| Query Type | Example | Strategy |
|------------|---------|----------|
| **Temporal** | "when did the API change?" | BM25 bias (alpha=0.7), auto-inject time bounds |
| **ExactKeyword** | "SqliteStore", "fn recall" | Heavy BM25 (alpha=0.85) |
| **Semantic** | "memory management strategies" | Vector dominant (alpha=0.3) |
| **Exploratory** | "what do I know about rein?" | Balanced (alpha=0.5), 2x result limit |

Classification is rule-based (zero LLM calls, sub-microsecond). MCP responses include `[route: type]` prefix for transparency. Based on TA-Mem 2026 and MemR3 2025.

### Adaptive Engine (v0.6.0+)

rein's core philosophy: **zero subjective parameters** — all parameters are data-driven and self-adaptive. The adaptive engine runs during GC in a slow channel (zero recall latency impact).

**Pipeline: M4 → M3 → M5 → M2 → M6**

| Module | What it learns | How |
|--------|---------------|-----|
| **M1** Event Sourcing | *(foundation)* | Append-only feedback log + per-consumer offsets |
| **M2** Alpha Optimizer | CC fusion weights | Counterfactual replay of past recalls with coarse-fine grid search |
| **M3** Survival Analysis | Per-cluster decay curves | Kaplan-Meier estimator from access interval data |
| **M4** HDBSCAN Clustering | Semantic neighborhoods | Pure Rust HDBSCAN (dendrogram → condensed tree → EOMBST) |
| **M5** Tiering | Hot/Warm/Cold boundaries | Streaming quantile estimator (P25/P75) + cold_archive migration |
| **M6** Threshold Explorer | Dedup thresholds | Randomized A/B exploration + causal inference + co-recall signal |

**Also:**
- **A1** Per-cluster adaptive dedup thresholds from intra-cluster similarity P90
- **Embedding-based semantic dedup** in GC slow channel (catches paraphrases)
- **Provenance-preserving merge** — temporal anchors and unique details never lost

### Configuration

rein loads configuration with the following priority (highest wins):

1. Environment variables
2. TOML config file (`$REIN_CONFIG` or `~/.config/rein/config.toml`)
3. Compiled-in defaults

#### Environment Variables

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Google Gemini API key for embeddings |
| `SUPERMEMORY_CC_API_KEY` | Supermemory API key for cross-validation |
| `REIN_HTTP_TOKEN` | Bearer token for non-localhost HTTP/SSE access |
| `REIN_DB` | Override database path |
| `REIN_CONFIG` | Override config file path |
| `REIN_LOG` | Log level filter (e.g. `debug`, `info`, `warn`) |

#### config.toml

```toml
[database]
path = "auto"                          # "auto" = ~/.rein/memories.db

[embedding]
provider = "google"    # or "omlx" or "none"
dimensions = 3072

[embedding.google]
model = "gemini-embedding-001"

[embedding.omlx]
endpoint = "http://localhost:8000/v1"
model = "default"

[search]
rrf_k = 60.0
rrf_fts_weight = 0.3
rrf_vec_weight = 0.7
fusion_method = "rrf"      # or "cc" (Convex Combination, Bruch 2023)
cc_alpha = 0.5             # CC blend: alpha * sparse + (1-alpha) * dense

dedup_similarity = 0.70    # uses max(jaccard, containment) similarity
dedup_time_window_days = 7

[chunking]
max_tokens = 512
overlap_percent = 10
metadata_prefix = true

[sync]
supermemory_enabled = true
auto_memory_enabled = true
auto_memory_glob = "~/.claude/projects/*/memory/**/*.md"

[decay]
base_lambda = 0.06
ltm_beta = 0.8
stm_beta = 1.2
interval_hours = 24
prune_threshold = 0.05
stm_to_ltm_access_count = 5

[server]
compact = false
sse_enabled = false
sse_port = 8680
sse_bind = "127.0.0.1"
```

### Database

The database is stored at `~/.rein/memories.db` by default. rein auto-migrates from the old location if needed.

Override with the `REIN_DB` environment variable or the `[database] path` config key.

### Hook Setup for Claude Code

Add the following to your Claude Code `settings.json` to enable automatic memory extraction:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "",
        "hooks": [
          { "type": "command", "command": "rein hook post", "timeout": 10 }
        ]
      }
    ],
    "PreCompact": [
      {
        "matcher": "",
        "hooks": [
          { "type": "command", "command": "rein hook compact", "timeout": 10 }
        ]
      }
    ],
    "Stop": [
      {
        "matcher": "",
        "hooks": [
          { "type": "command", "command": "rein hook stop", "timeout": 30 }
        ]
      }
    ]
  }
}
```

**Hook behavior (3 active hooks + 1 compatibility hook):**

- `PostToolUse` -- local pattern extraction (crash safety net) + buffers for session-end batch processing
- `PreCompact` -- records compact context for the async memory pipeline
- `UserPromptSubmit` -- compatibility no-op; rein no longer auto-injects prompt context
- `Stop` -- queues full knowledge extraction: memories + concepts + links + episode summary via async worker

### Remote Access via HTTP/SSE

Start rein with SSE transport for remote or multi-client access:

```bash
rein serve --sse
```

By default, the server binds to `127.0.0.1:8680`.

To bind to a non-localhost address, you **must** set the `REIN_HTTP_TOKEN` environment variable for bearer token authentication:

```bash
export REIN_HTTP_TOKEN="your-secret-token"
```

Configure bind address and port in `config.toml`:

```toml
[server]
sse_enabled = true
sse_port = 8680
sse_bind = "0.0.0.0"    # requires REIN_HTTP_TOKEN
```

### Architecture

```
                      User / AI Agent
                            |
                   +--------+--------+
                   |                 |
               CLI (20 cmds)  MCP Server (24 tools)
                   |                 |
                   +--------+--------+
                            |
                     +------+------+
                     |             |
                  Recall        Hooks
                  Pipeline    (extract + stop)
                     |
              +------+------+
              |             |
          FTS Search    Vec Search
              |             |
              |        +----+----+
              |        |         |
           Tantivy  Cache hit  Embed API
           (BM25)   (HNSW)    (Google/OMLX)
          fallback:  fallback:     |
            FTS5    sqlite-vec    HNSW
              |        |          |
              +---+----+----+-----+
                  |
             RRF/CC Fusion (weighted)
                  |
           Ebbinghaus Scoring
                  |
           Cross-Validation
        (local + supermemory + auto-memory)
                  |
              Results

Storage (source of truth):
  SQLite memories.db
    ├── memories table (CRUD)
    ├── FTS5 (built-in text index)
    └── sqlite-vec (vector fallback)
Side Indexes (derived, auto-rebuilt):
    ├── memories.tantivy/ (BM25 FTS)
    └── memories.usearch (HNSW ANN)
```

#### Search Pipeline

Two independent search paths run in parallel, then merge:

**Text path:**
1. **Tantivy BM25** -- full-text search with BM25 ranking (falls back to FTS5 if Tantivy unavailable)

**Vector path:**
2. **Cache check** -- look up query embedding in local cache (keyed by model + query)
3. **HNSW search** -- O(log n) approximate nearest neighbor via usearch (falls back to sqlite-vec)
4. If cache miss: **Embed API** -- call Google gemini-embedding-001 or OMLX, cache result, then HNSW search

**Merge:**
5. **RRF/CC fusion** -- Reciprocal Rank Fusion or Convex Combination merges text + vector results (path quality gating excludes empty paths)
6. **Adaptive scoring** -- Per-cluster Kaplan-Meier survival curves (or Ebbinghaus cold-start fallback) weight final ranking + temporal filtering
7. **Cross-validation** -- compare with Supermemory + auto-memory results, assign confidence

#### Embedding Backends

rein uses an `EmbedderKind` enum dispatch to support multiple embedding backends:

- **Google** (`gemini-embedding-001`) -- default, 3072 dimensions, MTEB #1 (68.32)
- **OMLX** -- local embedding via OpenAI-compatible API endpoint

Set `[embedding] provider` to `"google"`, `"omlx"`, or `"none"` in config.

#### Proxy / Endpoint Override

For users in China or behind firewalls, all API endpoints are configurable:

**Direct proxy (Cloudflare Worker, Nginx reverse proxy):**
```toml
[embedding.google]
endpoint = "https://your-gemini-proxy.com"
# Requests: {endpoint}/v1beta/models/gemini-embedding-001:embedContent

[sync]
endpoint = "https://your-supermemory-proxy.com"
```

**OpenRouter or other OpenAI-compatible aggregators:**
```toml
[embedding]
provider = "omlx"

[embedding.omlx]
endpoint = "https://openrouter.ai/api/v1"
model = "google/gemini-embedding-001"
```

This works because the OMLX backend uses the OpenAI `/v1/embeddings` format, which is compatible with OpenRouter, LiteLLM, and similar services.

#### Memory Decay Model

- **Critical** memories never decay (strength = 1.0 forever)
- **STM** (Short-Term Memory): faster decay (beta = 1.2), auto-promoted to LTM after 5 accesses
- **LTM** (Long-Term Memory): slower decay (beta = 0.8), assigned to high / critical importance
- Access count slows decay: `lambda_eff = lambda / (1 + access_count * 0.2)`

### Supported Clients

`rein init` auto-detects and configures:

- Claude Code
- Claude Desktop
- Cursor
- Windsurf
- VS Code (Copilot)
- Gemini CLI
- Codex
- OpenCode

### Performance Targets

| Metric | Target |
|--------|--------|
| Tantivy BM25 search | < 1 ms |
| HNSW ANN search | < 1 ms |
| FTS5 fallback search | < 1 ms |
| Vector search (cached) | < 1 ms |
| Vector search (API) | < 300 ms |
| Store (with dedup) | < 5 ms |
| Memory footprint | 2-5 MB |
| Binary size (release) | < 10 MB |

### Cost Estimate

| Component | Free tier | Cost at scale |
|-----------|-----------|---------------|
| gemini-embedding-001 | 1500 req/day | ~$0.00 |
| Supermemory | Optional | Free tier available |
| SQLite storage | Local | $0.00 |
| **Total** | **$0.00/month** | **< $0.03/month** |

### License

[MIT](LICENSE)

---

<details>
<summary><h2 id="中文">中文</h2></summary>

### 项目简介

rein 是一个自适应记忆系统，专为 AI 编程智能体设计。它跨会话存储、检索和管理记忆，核心理念是**零主观参数** — 所有参数由数据驱动、自动学习，不需要人工调参。

### 核心特性

| 特性 | 说明 |
|------|------|
| **24 个 MCP 工具** | 12 个核心记忆工具 + 10 个知识图谱工具 + 2 个时序工具 |
| **自适应引擎** | M1-M6 六模块：事件溯源 → 反事实 alpha 学习 → KM 生存曲线 → HDBSCAN 聚类 → 三层分级 → 阈值探索 |
| **反事实 Alpha 优化** | 回放历史 recall，粗细网格搜索最优 CC 融合权重（M2） |
| **Per-cluster KM 衰减** | Kaplan-Meier 生存曲线替代固定遗忘曲线（数据足够时自动切换，M3） |
| **HDBSCAN 语义聚类** | 纯 Rust 实现，dendrogram → 凝聚树 → EOMBST，大数据自动采样（M4） |
| **Hot/Warm/Cold 分层** | 流式分位数估计器 + cold_archive 迁移（M5） |
| **自适应去重阈值** | 基于簇内相似度 P90 计算（SemDeDup 风格，M6/A1） |
| **保留来源的去重** | 合并时保留时间锚点和独特细节，不丢失信息 |
| **嵌入语义去重** | 向量相似度捕捉文本相似度遗漏的改写，GC 慢通道执行 |
| **时序知识图谱** | Memoir / Concept / ConceptLink，9 种关系类型，修订历史，Episode 节点，时间窗口 |
| **自主检索路由** | 查询自动分类（时序/精确/语义/探索），自适应融合权重（TA-Mem 2026） |
| **OMLX 本地嵌入** | 可选本地嵌入后端（Google / OMLX） |
| **双路搜索** | Tantivy BM25 + HNSW ANN → RRF/CC 融合（学到的权重） |
| **多源交叉验证** | 3 个来源（本地、Hook 提取、Supermemory）+ 置信度评分 |
| **多因子准入控制** | A-MAC 2026：llm_conf + novelty + type_prior + recency 评分 |
| **语义分块** | 按标题/段落/句子分割，嵌入时附加元数据前缀 |
| **FTS5 unicode61 分词器** | 全文搜索，支持 CJK，亚毫秒级延迟 |
| **Supermemory v4 API** | 通过 `api.supermemory.ai/v4/search` 进行混合搜索交叉验证 |
| **零本地模型** | 默认无需 GPU（可选 OMLX 本地后端） |
| **~2-5 MB 占用** | 单个 SQLite 文件 + FTS5 + sqlite-vec |
| **gemini-embedding-001** | MTEB 排名第一（68.32），3072 维 |
| **20+ CLI 命令** | MCP 工具的全部功能，另加 init、config、migrate、hooks、recent、gc、organize、upgrade |
| **自动配置** | `rein init` 自动检测并配置 8 个 MCP 客户端 |
| **远程访问** | HTTP / SSE 传输，支持 bearer token 认证 |

### 安装

#### 从源码安装

```bash
git clone https://github.com/lyr1cs/rein.git
cd rein
cargo install --path .
```

或使用安装脚本：

```bash
./scripts/install.sh
```

#### 前置条件

- Rust 工具链 (1.75+)
- Gemini API 密钥（免费额度：1500 请求/天）

### 快速开始

```bash
# 1. 设置 API 密钥
export GEMINI_API_KEY="your-key-here"

# 2. 自动配置所有检测到的 MCP 客户端
rein init

# 3. 启动 MCP 服务（通常由客户端自动启动）
rein serve
```

### CLI 命令参考

| 命令 | 说明 | 示例 |
|------|------|------|
| `serve` | 启动 MCP 服务（stdio 或 SSE） | `rein serve [--compact] [--sse]` |
| `store` | 存储一条记忆 | `rein store -t debug -c "OOM fix" -I high -k oom,memory` |
| `recall` | 搜索记忆 | `rein recall "connection pool" -t debug -l 5` |
| `forget` | 按 ID 删除记忆 | `rein forget 01J...` |
| `update` | 更新记忆内容 | `rein update 01J... -c "new content" -I critical` |
| `topics` | 列出所有主题 | `rein topics` |
| `stats` | 显示存储统计 | `rein stats` |
| `health` | 检查主题健康状态 | `rein health [topic]` |
| `consolidate` | 将主题合并为一条记忆 | `rein consolidate debug -s "summary"` |
| `dedup` | 扫描/移除重复项 | `rein dedup [--dry-run]` |
| `migrate` | 从 QMD 导入 / 重建索引 | `rein migrate [--from-qmd path] [--reindex]` |
| `init` | 自动配置 MCP 客户端 | `rein init [--dry-run]` |
| `config` | 显示当前配置 | `rein config` |
| `hook post` | 从工具输出提取事实 | `rein hook post` |
| `hook compact` | 压缩前保存上下文 | `rein hook compact` |
| `hook prompt` | UserPromptSubmit 兼容性空操作 | `rein hook prompt` |
| `recent` | 显示最近记忆 | `rein recent [-l 20]` |
| `gc` | 垃圾回收弱 STM 记忆 | `rein gc [--dry-run]` |
| `organize` | 自动关联记忆 | `rein organize` |
| `upgrade` | 将旧记忆升级为知识图谱 | `rein upgrade [--topic X] [--dry-run]` |
| `hook post` | 从工具输出提取事实 | `rein hook post` |
| `hook compact` | 压缩前保存上下文 | `rein hook compact` |
| `hook prompt` | 自动注入已取消，仅保留命令入口 | `rein hook prompt` |
| `hook stop` | 会话结束时完整知识提取 | `rein hook stop` |

### MCP 工具

以 MCP 服务运行时（`rein serve`），共暴露 22 个工具。

#### 核心工具（12 个）

| 工具 | 参数 | 说明 |
|------|------|------|
| `rein_recall` | `query`, `topic?`, `keyword?`, `limit?` | 语义搜索记忆 |
| `rein_store` | `topic`, `content`, `importance?`, `keywords?` | 存储新记忆（自动去重） |
| `rein_update` | `id`, `content`, `importance?` | 更新已有记忆 |
| `rein_forget` | `id` | 按 ID 删除记忆 |
| `rein_list_topics` | *(无)* | 列出所有记忆主题 |
| `rein_stats` | *(无)* | 总数、LTM/STM 分布、平均强度 |
| `rein_health` | `topic?` | 陈旧计数、平均强度、合并建议 |
| `rein_consolidate` | `topic`, `summary` | 将主题内所有记忆合并为一条 |
| `rein_dedup` | `dry_run?` | 扫描并移除重复记忆 |
| `rein_recent` | `limit?` | 查看最近创建的记忆 |
| `rein_gc` | `dry_run?` | 垃圾回收弱 STM 记忆 |
| `rein_organize` | `max_links?` | 自动关联记忆 |

#### 知识图谱工具（10 个）

| 工具 | 参数 | 说明 |
|------|------|------|
| `rein_memoir_create` | `name`, `description?` | 创建知识容器 |
| `rein_memoir_list` | *(无)* | 列出所有 Memoir |
| `rein_memoir_show` | `name` | 显示 Memoir 详情及概念 |
| `rein_memoir_add_concept` | `memoir`, `name`, `definition`, `labels?` | 添加知识节点 |
| `rein_memoir_refine` | `memoir`, `name`, `definition` | 更新概念，提升置信度 |
| `rein_memoir_search` | `memoir`, `query`, `limit?` | 在 Memoir 内全文搜索 |
| `rein_memoir_search_all` | `query`, `limit?` | 跨所有 Memoir 搜索 |
| `rein_memoir_link` | `memoir`, `from`, `to`, `relation` | 链接两个概念 |
| `rein_memoir_inspect` | `memoir`, `name`, `depth?` | BFS 邻域遍历 |
| `rein_memoir_export` | `memoir`, `format?` | 导出图谱（json / ascii / dot） |

#### 知识图谱关系类型

`part_of`, `depends_on`, `related_to`, `contradicts`, `refines`, `alternative_to`, `caused_by`, `instance_of`, `superseded_by`

### LLM 提取层 (v0.3)

rein 使用 LLM（Gemini 3.1 Flash Lite 或本地模型）进行结构化记忆提取，自动构建知识图谱。

**架构：**
- `hook_post` — 本地模式提取（崩溃安全网）+ 缓冲到 session 文件
- `hook_compact` — 记录 compact 上下文，交给异步 memory worker 提炼
- `hook_stop` — 完整知识提取：记忆 + 概念 + 关系 + 会话摘要（异步 worker）
- `hook_prompt` — 兼容性空操作（已取消自动注入）

**升级旧记忆：**
```bash
rein upgrade --dry-run    # 预览
rein upgrade              # 将旧记忆转为知识图谱
```

**配置：**
```toml
[extract]
provider = "google"    # 或 "omlx" 或 "none"

[extract.google]
model = "gemini-3.1-flash-lite-preview"
max_input_chars = 0    # 0 = 不截断（1M token 模型）

[extract.omlx]
endpoint = "http://localhost:11434/v1"  # Ollama, LM Studio, vLLM 等
model = "default"
max_input_chars = 16000
```

### 自学习质量系统 (v0.3.0)

rein 自动学习哪些记忆有用、哪些是噪声，无需人工调参。

**工作原理：**
1. LLM 在提取时给出 `quality_confidence` (0-1) — 零额外 API 成本
2. 系统追踪 recall → access 模式，分类"好记忆"（被使用）和"差记忆"（被召回但未使用）
3. 特征权重自动从数据学习：使用率、新颖度、连通度、时效性
4. 自适应入口阈值：近期质量低 → 收紧，高 → 放松
5. GC 清理质量低且被召回 5+ 次但从未使用的概念

**无需手动调参** — 冷启动用 LLM 判断，数据逐渐接管。

基于：ICLR 2026 Admission Control, PropMem (Prosus), FActScore, MACLA。

### 配置

rein 按以下优先级加载配置（高优先级覆盖低优先级）：

1. 环境变量
2. TOML 配置文件（`$REIN_CONFIG` 或 `~/.config/rein/config.toml`）
3. 编译时默认值

#### 环境变量

| 变量 | 说明 |
|------|------|
| `GEMINI_API_KEY` | Google Gemini API 密钥（用于嵌入） |
| `SUPERMEMORY_CC_API_KEY` | Supermemory API 密钥（用于交叉验证） |
| `REIN_HTTP_TOKEN` | 非 localhost HTTP/SSE 访问的 bearer token |
| `REIN_DB` | 覆盖数据库路径 |
| `REIN_CONFIG` | 覆盖配置文件路径 |
| `REIN_LOG` | 日志级别过滤（如 `debug`、`info`、`warn`） |

#### config.toml

```toml
[database]
path = "auto"                          # "auto" = ~/.rein/memories.db

[embedding]
provider = "google"    # 或 "omlx" 或 "none"
dimensions = 3072

[embedding.google]
model = "gemini-embedding-001"

[embedding.omlx]
endpoint = "http://localhost:8000/v1"
model = "default"

[search]
rrf_k = 60.0
rrf_fts_weight = 0.3
rrf_vec_weight = 0.7

dedup_similarity = 0.70    # 使用 max(jaccard, containment) 相似度
dedup_time_window_days = 7

[chunking]
max_tokens = 512
overlap_percent = 10
metadata_prefix = true

[sync]
supermemory_enabled = true
auto_memory_enabled = true
auto_memory_glob = "~/.claude/projects/*/memory/**/*.md"

[decay]
base_lambda = 0.06
ltm_beta = 0.8
stm_beta = 1.2
interval_hours = 24
prune_threshold = 0.05
stm_to_ltm_access_count = 5

[server]
compact = false
sse_enabled = false
sse_port = 8680
sse_bind = "127.0.0.1"
```

### 数据库

数据库默认存储在 `~/.rein/memories.db`。rein 会自动从旧位置迁移数据。

可通过 `REIN_DB` 环境变量或 `[database] path` 配置项覆盖路径。

### Claude Code Hook 设置

在 Claude Code 的 `settings.json` 中添加以下内容以启用自动记忆提取：

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "",
        "hooks": [
          { "type": "command", "command": "rein hook post", "timeout": 10 }
        ]
      }
    ],
    "PreCompact": [
      {
        "matcher": "",
        "hooks": [
          { "type": "command", "command": "rein hook compact", "timeout": 10 }
        ]
      }
    ],
    "Stop": [
      {
        "matcher": "",
        "hooks": [
          { "type": "command", "command": "rein hook stop", "timeout": 30 }
        ]
      }
    ]
  }
}
```

**Hook 行为说明（3 个活跃 Hook + 1 个兼容 Hook）：**

- `PostToolUse` -- 本地模式提取（崩溃安全网）+ 缓冲到 session 文件
- `PreCompact` -- 记录重要上下文并交给异步 memory worker
- `UserPromptSubmit` -- 兼容性空操作；不再自动注入提示词
- `Stop` -- 完整知识提取：记忆 + 概念 + 关系 + 会话摘要（通过异步 worker）

### 通过 HTTP/SSE 远程访问

启动 SSE 传输以支持远程或多客户端访问：

```bash
rein serve --sse
```

默认绑定地址为 `127.0.0.1:8680`。

若要绑定到非 localhost 地址，**必须**设置 `REIN_HTTP_TOKEN` 环境变量以启用 bearer token 认证：

```bash
export REIN_HTTP_TOKEN="your-secret-token"
```

在 `config.toml` 中配置绑定地址和端口：

```toml
[server]
sse_enabled = true
sse_port = 8680
sse_bind = "0.0.0.0"    # 需要设置 REIN_HTTP_TOKEN
```

### 架构

```
                      用户 / AI 智能体
                            |
                   +--------+--------+
                   |                 |
              CLI (20 命令)   MCP 服务 (22 工具)
                   |                 |
                   +--------+--------+
                            |
                     +------+------+
                     |             |
                   召回          Hooks
                   管线      (提取 + 会话保存)
                     |
              +------+------+
              |             |
           全文搜索      向量搜索
              |             |
              |        +----+----+
              |        |         |
           Tantivy   缓存命中  Embed API
           (BM25)    (HNSW)   (Google/OMLX)
          回退:      回退:        |
            FTS5   sqlite-vec   HNSW
              |        |         |
              +---+----+----+----+
                  |
             RRF 融合（加权）
                  |
             艾宾浩斯评分
                  |
             交叉验证
        (本地 + Supermemory + auto-memory)
                  |
               结果

存储（唯一真实来源）:
  SQLite memories.db
    ├── memories 表（CRUD）
    ├── FTS5（内置全文索引）
    └── sqlite-vec（向量回退）
旁路索引（派生，可自动重建）:
    ├── memories.tantivy/（BM25 全文搜索）
    └── memories.usearch（HNSW 近似最近邻）
```

#### 搜索管线

1. **Tantivy BM25** -- Tantivy 全文搜索（回退到 FTS5），亚毫秒级
2. **HNSW ANN** -- O(log n) 近似最近邻（usearch），回退到 sqlite-vec 暴力搜索
3. **缓存向量** -- sqlite-vec 中预计算的嵌入向量
4. **API 向量** -- 通过 gemini-embedding-001（或 OMLX 本地后端）按需嵌入
5. **RRF 融合** -- 加权倒数排名融合合并所有结果列表
6. **艾宾浩斯评分** -- `strength(t) = exp(-lambda_eff * days^beta)` 加权最终排序

#### 嵌入后端

rein 使用 `EmbedderKind` 枚举分发支持多种嵌入后端：

- **Google**（`gemini-embedding-001`）-- 默认，3072 维，MTEB 排名第一（68.32）
- **OMLX** -- 通过 OpenAI 兼容 API 端点进行本地嵌入

在配置中设置 `[embedding] provider` 为 `"google"`、`"omlx"` 或 `"none"`。

#### 代理 / Endpoint 覆盖

国内用户或防火墙环境，所有 API 端点均可配置：

**直接代理（Cloudflare Worker、Nginx 反代）：**
```toml
[embedding.google]
endpoint = "https://your-gemini-proxy.com"
# 请求路径: {endpoint}/v1beta/models/gemini-embedding-001:embedContent

[sync]
endpoint = "https://your-supermemory-proxy.com"
```

**OpenRouter 等 OpenAI 兼容聚合商：**
```toml
[embedding]
provider = "omlx"

[embedding.omlx]
endpoint = "https://openrouter.ai/api/v1"
model = "google/gemini-embedding-001"
```

OMLX 后端使用 OpenAI `/v1/embeddings` 格式，兼容 OpenRouter、LiteLLM 等服务。

#### 记忆衰减模型

- **Critical** 记忆永不衰减（强度始终为 1.0）
- **STM**（短期记忆）：衰减较快（beta = 1.2），5 次访问后自动晋升为 LTM
- **LTM**（长期记忆）：衰减较慢（beta = 0.8），分配给 high / critical 重要度
- 访问次数减缓衰减：`lambda_eff = lambda / (1 + access_count * 0.2)`

### 支持的客户端

`rein init` 自动检测并配置：

- Claude Code
- Claude Desktop
- Cursor
- Windsurf
- VS Code (Copilot)
- Gemini CLI
- Codex
- OpenCode

### 性能目标

| 指标 | 目标 |
|------|------|
| Tantivy BM25 搜索 | < 1 ms |
| HNSW ANN 搜索 | < 1 ms |
| FTS5 回退搜索 | < 1 ms |
| 向量搜索（缓存） | < 1 ms |
| 向量搜索（API） | < 300 ms |
| 存储（含去重） | < 5 ms |
| 内存占用 | 2-5 MB |
| 二进制大小（release） | < 10 MB |

### 成本估算

| 组件 | 免费额度 | 大规模使用成本 |
|------|----------|----------------|
| gemini-embedding-001 | 1500 请求/天 | ~$0.00 |
| Supermemory | 可选 | 有免费额度 |
| SQLite 存储 | 本地 | $0.00 |
| **合计** | **$0.00/月** | **< $0.03/月** |

### 许可证

[MIT](LICENSE)

</details>
