# rein

**Multi-source cross-validated memory for AI agents.**

rein is a lightweight, persistent memory system designed for AI coding agents. It stores, recalls, and manages memories across sessions with automatic deduplication, Ebbinghaus-inspired decay, and multi-source cross-validation.

---

## Features

| Feature | Description |
|---------|-------------|
| **Multi-source cross-validation** | 3 sources (local, hook-extracted, Supermemory) with confidence scoring |
| **Dual-layer Ebbinghaus decay** | LTM/STM layers with configurable lambda, beta, and access-boosted retention |
| **Three-level waterfall search** | FTS5 (<1ms) -> cached vectors (<1ms) -> Google API (~255ms) *(vector/cross-validation wiring in progress)* |
| **Weighted RRF fusion** | Reciprocal Rank Fusion with configurable per-source weights (not fixed) |
| **Semantic chunking** | Heading/paragraph/sentence splitting with metadata-prefixed embeddings |
| **Zero local models** | Zero local models by default (optional OMLX local backend) |
| **~2-5 MB memory footprint** | Single SQLite file with FTS5 + sqlite-vec |
| **gemini-embedding-001** | MTEB #1 model (68.32), 3072 dimensions |
| **Knowledge graph** | Memoir/Concept/Link with 9 relation types, BFS traversal, export (json/ascii/dot) |
| **19 MCP tools** | 9 core + 10 knowledge graph |
| **15 CLI commands** | Everything the MCP tools do, plus init, config, migrate, hooks |
| **Auto-configure** | `rein init` detects and configures 8 MCP clients automatically |

## Installation

### From source

```bash
git clone https://github.com/lyr1cs/rein.git
cd rein
cargo install --path .
```

Or use the install script:

```bash
./scripts/install.sh
```

### Prerequisites

- Rust toolchain (1.75+)
- A Gemini API key (free tier: 1500 req/day)

## Quick Start

```bash
# 1. Set your API key
export GEMINI_API_KEY="your-key-here"

# 2. Auto-configure all detected MCP clients
rein init

# 3. Start the MCP server (usually done by your client)
rein serve
```

## CLI Reference

| Command | Description | Example |
|---------|-------------|---------|
| `serve` | Start MCP server (stdio) | `rein serve [--compact] [--sse]` |
| `store` | Store a memory | `rein store -t debug -c "OOM fix" -I high -k oom,memory` |
| `recall` | Search memories | `rein recall "connection pool" -t debug -l 5` |
| `forget` | Delete a memory by ID | `rein forget 01J...` |
| `update` | Update memory content | `rein update 01J... -c "new content" -I critical` |
| `topics` | List all topics | `rein topics` |
| `stats` | Show store statistics | `rein stats` |
| `health` | Check topic health | `rein health [topic]` |
| `consolidate` | Merge topic into one memory | `rein consolidate debug -s "summary"` |
| `dedup` | Scan/remove duplicates | `rein dedup [--dry-run]` |
| `migrate` | Import from QMD / reindex | `rein migrate [--from-qmd path] [--reindex]` |
| `init` | Auto-configure MCP clients | `rein init [--dry-run]` |
| `config` | Show current configuration | `rein config` |
| `hook` | Run hook subcommands | `rein hook post\|compact\|prompt` |

## MCP Tools

When running as an MCP server (`rein serve`), the following 19 tools are exposed:

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
| `rein_memoir_create` | `name`, `description?` | Create a knowledge container |
| `rein_memoir_list` | *(none)* | List all memoirs |
| `rein_memoir_show` | `name` | Show memoir details + concepts |
| `rein_memoir_add_concept` | `memoir`, `name`, `definition`, `labels?` | Add knowledge node |
| `rein_memoir_refine` | `memoir`, `name`, `definition` | Update concept, boost confidence |
| `rein_memoir_search` | `memoir`, `query`, `limit?` | FTS search within memoir |
| `rein_memoir_search_all` | `query`, `limit?` | Search across all memoirs |
| `rein_memoir_link` | `memoir`, `from`, `to`, `relation` | Link two concepts |
| `rein_memoir_inspect` | `memoir`, `name`, `depth?` | BFS neighborhood traversal |
| `rein_memoir_export` | `memoir`, `format?` | Export graph (json/ascii/dot) |

### Knowledge Graph Relation Types

The following 9 relation types are available for linking concepts in memoirs:

`part_of`, `depends_on`, `related_to`, `contradicts`, `refines`, `alternative_to`, `caused_by`, `instance_of`, `superseded_by`

## Configuration

rein loads configuration with the following priority (highest wins):

1. Environment variables
2. TOML config file (`$REIN_CONFIG` or `~/.config/rein/config.toml`)
3. Compiled-in defaults

### Environment Variables

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Google Gemini API key for embeddings |
| `SUPERMEMORY_CC_API_KEY` | Supermemory API key for cross-validation |
| `REIN_DB` | Override database path |
| `REIN_CONFIG` | Override config file path |
| `REIN_LOG` | Log level filter (e.g. `debug`, `info`, `warn`) |

### config.toml

```toml
[database]
path = "auto"                          # "auto" = ~/.local/share/rein/memories.db

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
```

## Hook Setup for Claude Code

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
    ]
  }
}
```

**Hook behavior:**

- `PostToolUse` -- extracts facts from tool output, stores as auto-extracted memories
- `PreCompact` -- saves important context before context window compression
- `UserPromptSubmit` -- injects recalled memories into the prompt as `<supermemory-context>`

## Architecture

```
                        User / AI Agent
                             |
                    +--------+--------+
                    |                 |
                CLI (15 cmds)   MCP Server (19 tools)
                    |                 |
                    +--------+--------+
                             |
                      +------+------+
                      |             |
                  Waterfall      Hooks
                  Search       (extract)
                      |
            +---------+---------+
            |         |         |
          FTS5    Cached Vec  API Vec
         (<1ms)   (<1ms)     (~255ms)
            |         |         |
            +----+----+---------+
                 |
            RRF Fusion (weighted)
                 |
          Ebbinghaus Scoring
          (strength * access)
                 |
            +----+----+
            |         |
          SQLite   sqlite-vec
          (FTS5)   (embeddings)
            |         |
            +----+----+
                 |
           memories.db
            (~2-5 MB)
```

### Search Pipeline

1. **FTS5** -- full-text search with SQLite FTS5, sub-millisecond
2. **Cached vectors** -- pre-computed embeddings in sqlite-vec
3. **API vectors** -- on-demand embedding via gemini-embedding-001
4. **RRF fusion** -- weighted Reciprocal Rank Fusion merges all result lists
5. **Ebbinghaus scoring** -- `strength(t) = exp(-lambda_eff * days^beta)` weights final ranking

### Memory Decay Model

- **Critical** memories never decay (strength = 1.0 forever)
- **STM** (Short-Term Memory): faster decay (beta = 1.2), auto-promoted to LTM after 5 accesses
- **LTM** (Long-Term Memory): slower decay (beta = 0.8), assigned to high/critical importance
- Access count slows decay: `lambda_eff = lambda / (1 + access_count * 0.2)`

## Performance Targets

| Metric | Target |
|--------|--------|
| FTS5 search | < 1 ms |
| Vector search (cached) | < 1 ms |
| Vector search (API) | < 300 ms |
| Store (with dedup) | < 5 ms |
| Memory footprint | 2-5 MB |
| Binary size (release) | < 10 MB |

## Cost Estimate

| Component | Free tier | Cost at scale |
|-----------|-----------|---------------|
| gemini-embedding-001 | 1500 req/day | ~$0.00 |
| Supermemory | Optional | Free tier available |
| SQLite storage | Local | $0.00 |
| **Total** | **$0.00/month** | **< $0.03/month** |

## Supported Clients

`rein init` auto-detects and configures:

- Claude Code
- Claude Desktop
- Cursor
- Windsurf
- VS Code (Copilot)
- Gemini CLI
- Codex
- OpenCode

See `config/` for individual client configuration snippets.

## License

[MIT](LICENSE)

---

# rein (中文)

**面向 AI 智能体的多源交叉验证记忆系统。**

rein 是一个轻量级的持久化记忆系统,专为 AI 编程智能体设计。它跨会话存储、检索和管理记忆,具备自动去重、基于艾宾浩斯遗忘曲线的衰减机制,以及多源交叉验证能力。

## 核心特性

- **多源交叉验证** -- 3 个来源(本地、Hook 提取、Supermemory)+ 置信度评分
- **双层艾宾浩斯衰减** -- LTM/STM 层,可配置 lambda、beta,访问次数越多衰减越慢
- **三级瀑布搜索** -- FTS5 (<1ms) -> 缓存向量 (<1ms) -> Google API (~255ms)
- **加权 RRF 融合** -- 可配置权重的倒数排名融合(非固定权重)
- **语义分块** -- 按标题/段落/句子分割,嵌入时附加元数据前缀
- **零本地模型** -- 默认无需 GPU(可选 OMLX 本地后端)
- **~2-5 MB 内存占用** -- 单个 SQLite 文件 + FTS5 + sqlite-vec
- **gemini-embedding-001** -- MTEB 排名第一(68.32),3072 维
- **知识图谱** -- Memoir/Concept/Link,9 种关系类型,BFS 遍历,导出(json/ascii/dot)
- **19 个 MCP 工具 + 15 个 CLI 命令**

## 快速开始

```bash
# 安装
cargo install --path .

# 设置 API 密钥
export GEMINI_API_KEY="your-key"

# 自动配置所有检测到的 MCP 客户端
rein init

# 启动 MCP 服务
rein serve
```

## 配置

配置文件位于 `~/.config/rein/config.toml`,支持环境变量覆盖。详见上方英文文档。

## 许可证

[MIT](LICENSE)
