# rein

> Multi-source cross-validated memory for AI agents

<p align="center">
  <a href="#english">English</a> | <a href="#中文">中文</a>
</p>

---

## English

rein is a lightweight, persistent memory system designed for AI coding agents. It stores, recalls, and manages memories across sessions with automatic deduplication, Ebbinghaus-inspired decay, and multi-source cross-validation.

### Features

| Feature | Description |
|---------|-------------|
| **19 MCP tools** | 9 core memory tools + 10 knowledge graph tools |
| **Knowledge graph** | Memoir / Concept / ConceptLink with 9 relation types, BFS traversal, export (json / ascii / dot) |
| **OMLX local embedding** | Optional local embedding backend via EmbedderKind enum dispatch (Google / OMLX) |
| **Dual-layer Ebbinghaus decay** | LTM / STM layers with configurable lambda, beta, and access-boosted retention |
| **Dual-path search** | FTS (Tantivy BM25 → FTS5 fallback) + Vector (HNSW cache → API embed) → RRF fusion |
| **Multi-source cross-validation** | 3 sources (local, hook-extracted, Supermemory) with confidence scoring |
| **Weighted RRF fusion** | Reciprocal Rank Fusion with configurable per-source weights |
| **Semantic chunking** | Heading / paragraph / sentence splitting with metadata-prefixed embeddings |
| **FTS5 unicode61 tokenizer** | Full-text search with CJK support, sub-millisecond latency |
| **Supermemory v4 API** | Hybrid search via `api.supermemory.ai/v4/search` for cross-validation |
| **Zero local models** | No GPU required by default; optional OMLX local backend |
| **~2-5 MB footprint** | Single SQLite file with FTS5 + sqlite-vec |
| **gemini-embedding-001** | MTEB #1 model (68.32), 3072 dimensions |
| **15+ CLI commands** | Everything the MCP tools do, plus init, config, migrate, hooks |
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
| `hook post` | Extract facts from tool output | `rein hook post` |
| `hook compact` | Save context before compaction | `rein hook compact` |
| `hook prompt` | Inject recalled memories into prompt | `rein hook prompt` |
| `hook stop` | Save session summary on conversation end | `rein hook stop` |

### MCP Tools

When running as an MCP server (`rein serve`), 19 tools are exposed.

#### Core Tools (9)

| Tool | Parameters | Description |
|------|-----------|-------------|
| `rein_recall` | `query`, `topic?`, `keyword?`, `limit?` | Semantic search over memories |
| `rein_store` | `topic`, `content`, `importance?`, `keywords?` | Store a new memory (auto-dedup) |
| `rein_update` | `id`, `content`, `importance?` | Update an existing memory |
| `rein_forget` | `id` | Delete a memory by ID |
| `rein_list_topics` | *(none)* | List all memory topics |
| `rein_stats` | *(none)* | Total count, LTM/STM breakdown, avg strength |
| `rein_health` | `topic?` | Stale count, avg strength, consolidation hints |
| `rein_consolidate` | `topic`, `summary` | Merge all memories in a topic into one |
| `rein_dedup` | `dry_run?` | Scan for and remove duplicate memories |

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

#### Knowledge Graph Relation Types

The following 9 relation types are available for linking concepts:

`part_of`, `depends_on`, `related_to`, `contradicts`, `refines`, `alternative_to`, `caused_by`, `instance_of`, `superseded_by`

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
waterfall_fts_threshold = 0.5
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
    "UserPromptSubmit": [
      {
        "matcher": "",
        "hooks": [
          { "type": "command", "command": "rein hook prompt", "timeout": 10 }
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

**Hook behavior (4 hooks):**

- `PostToolUse` -- extracts facts from tool output, stores as auto-extracted memories
- `PreCompact` -- saves important context before context window compression
- `UserPromptSubmit` -- injects recalled memories into the prompt as `<rein-context>`
- `Stop` -- extracts session summary with signal-based context windows on conversation end

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
               CLI (16 cmds)  MCP Server (19 tools)
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
             RRF Fusion (weighted)
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
5. **RRF fusion** -- weighted Reciprocal Rank Fusion merges text + vector results
6. **Ebbinghaus scoring** -- `strength(t) = exp(-lambda_eff * days^beta)` weights final ranking
7. **Cross-validation** -- compare with Supermemory + auto-memory results, assign confidence

#### Embedding Backends

rein uses an `EmbedderKind` enum dispatch to support multiple embedding backends:

- **Google** (`gemini-embedding-001`) -- default, 3072 dimensions, MTEB #1 (68.32)
- **OMLX** -- local embedding via OpenAI-compatible API endpoint

Set `[embedding] provider` to `"google"`, `"omlx"`, or `"none"` in config.

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

rein 是一个轻量级的持久化记忆系统，专为 AI 编程智能体设计。它跨会话存储、检索和管理记忆，具备自动去重、基于艾宾浩斯遗忘曲线的衰减机制，以及多源交叉验证能力。

### 核心特性

| 特性 | 说明 |
|------|------|
| **19 个 MCP 工具** | 9 个核心记忆工具 + 10 个知识图谱工具 |
| **知识图谱** | Memoir / Concept / ConceptLink，9 种关系类型，BFS 遍历，导出（json / ascii / dot） |
| **OMLX 本地嵌入** | 可选本地嵌入后端，通过 EmbedderKind 枚举分发（Google / OMLX） |
| **双层艾宾浩斯衰减** | LTM / STM 层，可配置 lambda、beta，访问次数越多衰减越慢 |
| **四级瀑布搜索** | Tantivy BM25 → HNSW ANN → 缓存向量 → Google API |
| **多源交叉验证** | 3 个来源（本地、Hook 提取、Supermemory）+ 置信度评分 |
| **加权 RRF 融合** | 可配置权重的倒数排名融合 |
| **语义分块** | 按标题/段落/句子分割，嵌入时附加元数据前缀 |
| **FTS5 unicode61 分词器** | 全文搜索，支持 CJK，亚毫秒级延迟 |
| **Supermemory v4 API** | 通过 `api.supermemory.ai/v4/search` 进行混合搜索交叉验证 |
| **零本地模型** | 默认无需 GPU（可选 OMLX 本地后端） |
| **~2-5 MB 占用** | 单个 SQLite 文件 + FTS5 + sqlite-vec |
| **gemini-embedding-001** | MTEB 排名第一（68.32），3072 维 |
| **15+ CLI 命令** | MCP 工具的全部功能，另加 init、config、migrate、hooks |
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
| `hook prompt` | 将召回的记忆注入提示词 | `rein hook prompt` |
| `hook stop` | 会话结束时保存会话摘要 | `rein hook stop` |

### MCP 工具

以 MCP 服务运行时（`rein serve`），共暴露 19 个工具。

#### 核心工具（9 个）

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

可用于链接概念的 9 种关系类型：

`part_of`, `depends_on`, `related_to`, `contradicts`, `refines`, `alternative_to`, `caused_by`, `instance_of`, `superseded_by`

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
waterfall_fts_threshold = 0.5
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
    "UserPromptSubmit": [
      {
        "matcher": "",
        "hooks": [
          { "type": "command", "command": "rein hook prompt", "timeout": 10 }
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

**Hook 行为说明（4 个 Hook）：**

- `PostToolUse` -- 从工具输出中提取事实，作为自动提取的记忆存储
- `PreCompact` -- 在上下文窗口压缩前保存重要上下文
- `UserPromptSubmit` -- 将召回的记忆以 `<rein-context>` 形式注入提示词
- `Stop` -- 会话结束时通过信号关键词上下文窗口提取会话摘要

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
              CLI (16 命令)   MCP 服务 (19 工具)
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
