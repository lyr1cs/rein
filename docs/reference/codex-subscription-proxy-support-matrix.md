---
title: Codex 订阅登录态 Proxy Support Matrix
date: 2026-04-16
tags: [reference, codex, proxy, subscription, support-matrix, rein]
---

# Codex 订阅登录态 Proxy Support Matrix

基线来源：官方 `openai/codex` 仓库，HEAD `b0324f9`。

当前结论：`codexsubp` 是推荐的 ChatGPT-login loopback 入口。rein 已经把路由分层、`codex_upstream` / `chatgpt_upstream` 拆分、first-party `/responses` 记录链路，以及 helper/discovery 透明转发接通。现在剩余的主要是 hardening 和自动化，不是补齐基础功能。

补充：实验性的 websocket-first 入口现在也能通过本地 `426` fallback 跑通 smoke，但默认推荐入口仍然是 `codexsubp`。

## 1. 路由分层

Codex 的 ChatGPT 登录态不是单一路径，而是两层 base URL：

- **模型 API base**
  - `https://chatgpt.com/backend-api/codex`
  - 端点：`/responses`、`/responses/compact`、`/models`、`/memories/trace_summarize`
- **ChatGPT backend root**
  - `https://chatgpt.com/backend-api`
  - 端点：`/wham/*`、`/connectors/*`、`/v1/agent/register`、`/authenticate_app_v2`、`/codex/safety/arc`

这意味着 proxy 不能只识别 `/backend-api/codex/*`，还必须识别 `chatgpt_base_url` 那一层 helper/discovery 请求。

## 2. 当前支持矩阵

| 路径族 | 触发条件 | upstream | recording mode | 当前状态 / 测试 |
|---|---|---|---|---|
| `/responses` | API-key scope: `api.responses.read` + `api.responses.write` | `openai_upstream` (`/v1/responses`) | StructuredText | 已覆盖：`route_resolution_support_matrix`, `proxy_forwards_api_responses_route_to_openai_upstream_with_v1_prefix` |
| `/responses` | ChatGPT login token | `codex_upstream` (`/responses`) | StructuredText | 已覆盖：`route_resolution_support_matrix`, `proxy_forwards_chatgpt_login_responses_route_to_codex_upstream` |
| `/models` | ChatGPT login token | `codex_upstream` (`/models`) | ArtifactMirrorOnly | 已覆盖：`route_resolution_support_matrix`, `proxy_forwards_chatgpt_login_models_route_to_codex_upstream` |
| `/responses/compact` | ChatGPT login token | `codex_upstream` (`/responses/compact`) | ArtifactMirrorOnly | 已覆盖：`route_resolution_support_matrix`, `proxy_forwards_chatgpt_login_compact_route_to_codex_upstream` |
| `/memories/trace_summarize` | ChatGPT login token | `codex_upstream` (`/memories/trace_summarize`) | ArtifactMirrorOnly | 已覆盖：`route_resolution_support_matrix`, `proxy_forwards_chatgpt_login_memories_route_to_codex_upstream` |
| `/wham/*` | ChatGPT login token | `chatgpt_upstream` | ArtifactMirrorOnly | 已覆盖：`route_resolution_support_matrix`, `proxy_forwards_chatgpt_helper_paths_to_chatgpt_upstream`, `proxy_forwards_chatgpt_tasks_list_route_to_chatgpt_upstream`, `proxy_forwards_chatgpt_task_details_route_to_chatgpt_upstream`, `proxy_forwards_chatgpt_sibling_turns_route_to_chatgpt_upstream`, `proxy_forwards_chatgpt_requirements_route_to_chatgpt_upstream` |
| `/connectors/*` | ChatGPT login token | `chatgpt_upstream` | ArtifactMirrorOnly | 已覆盖：`route_resolution_support_matrix`, `proxy_forwards_chatgpt_connector_directory_route_to_chatgpt_upstream`, `proxy_forwards_chatgpt_workspace_connector_route_to_chatgpt_upstream` |
| `/v1/agent/register` | ChatGPT login token | `chatgpt_upstream` | ArtifactMirrorOnly | 已覆盖：`route_resolution_support_matrix`, `proxy_forwards_chatgpt_agent_register_route_to_chatgpt_upstream` |
| `/authenticate_app_v2` | ChatGPT login token | `chatgpt_upstream` | ArtifactMirrorOnly | 已覆盖：`route_resolution_support_matrix`, `proxy_forwards_chatgpt_authenticate_app_route_to_chatgpt_upstream` |
| `/codex/safety/arc` | ChatGPT login token | `chatgpt_upstream` | ArtifactMirrorOnly | 已覆盖：`route_resolution_support_matrix`, `proxy_forwards_chatgpt_arc_monitor_route_to_chatgpt_upstream` |
| `GET /responses` WebSocket upgrade | ChatGPT login token | `codex_upstream` (`/responses`) | StructuredText + `proxy_first_party_ws` artifact | 已覆盖：`proxy_forwards_chatgpt_login_websocket_upgrade_to_codex_upstream`, `proxy_returns_426_when_codex_websocket_upstream_is_unavailable`, `proxy_stores_redacted_first_party_websocket_artifact_for_chatgpt_login_responses`, `websocket_request_mirror_extracts_response_create_query`, `websocket_mirror_reassembles_fragmented_text_frames`, `websocket_mirror_decodes_compressed_text_frames` |

补充边界：

- 只有结构化文本的 `/responses` 路径允许 websocket upgrade；`ArtifactMirrorOnly` 的 first-party 路径（如 `/models`、`/responses/compact`、`/memories/trace_summarize`、`/wham/*`、`/connectors/*`）会保持 HTTP 透传。
- proxy 现在不会向客户端回显 `sec-websocket-extensions`，避免在本地 client-side 没有完整协商器时错误宣称 `permessage-deflate` 已经谈成。

## 3. Header / Auth 语义

订阅登录态下必须保留：

- `Authorization: Bearer ...`
- `ChatGPT-Account-ID: ...`
- `X-OpenAI-Fedramp: true`（仅特定 token）
- `x-client-request-id`
- `x-codex-turn-state`
- `x-codex-turn-metadata`
- `x-codex-window-id`
- `x-codex-parent-thread-id`
- `x-openai-subagent`

WebSocket 额外需要：

- `OpenAI-Beta: responses_websockets=2026-02-06`

## 4. 当前测试覆盖

- `route_resolution_support_matrix` + `responses_scope_support_matrix` 现在共用一份表驱动 case 数据，覆盖了 API-key `/responses`、ChatGPT 登录 `/responses`、`/models`、`/responses/compact`、`/memories/trace_summarize`、`/wham/usage`、`/connectors/directory/list`、`/backend-api/*` 路径，以及对 `/v1/models` 的拒绝。
- HTTP 透传测试覆盖了 `openai_upstream`、`codex_upstream`、`chatgpt_upstream` 三路分流。
- first-party artifact mirror 覆盖了 `/responses` 的脱敏 HTTP 归档、REST readback，以及 WebSocket 归档；非 `/responses` first-party 路径保持 `ArtifactMirrorOnly`。
- WebSocket 行为覆盖了 upgrade、`426` 本地 fallback、request-query mirror、fragmentation、compression，以及 assistant text / event mirror 的重组逻辑。
- 实机 smoke 已验证两条入口：
  - `codexsubp`：HTTP-first 推荐路径
  - `codexsubpws`：实验性 websocket-first 路径，依赖本地 `426` fallback
- websocket artifact 现在还覆盖了更接近真实客户端的 header/session 组合：
  - `x-codex-parent-thread-id` 可作为 session id 回退
  - `x-codex-window-id`
  - `x-openai-subagent`
  - `client_ws_events` 中的 `response.create`

## 5. Remaining Work

- 更复杂的 WebSocket query/header 组合与 permessage-deflate / fragmentation 组合
- `codexsubp` smoke / E2E fixture 自动化
- support matrix 自动生成和回归校验继续往更完整的单一数据源推进
- non-`/responses` first-party 路径是否要永远保持 `ArtifactMirrorOnly`
