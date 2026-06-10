# Pets Agent

基于可插拔 Agent SDK 的角色化知识库助手服务。

## 项目概述

Pets Agent 提供两种角色化助手：

- **文档助手（reviewer）**：基于知识库的问答助手，支持多轮对话，回答用户关于工作区内容的问题
- **开发助手（developer）**：代码修改助手，使用配置的 Agent SDK 读取、编辑、创建代码并运行验证

两种角色均通过 SSE 实时流式输出思维过程、文本和工具调用。

## 技术栈

- **运行时**: Node.js 22+ / TypeScript
- **框架**: Fastify
- **AI SDK**: 可配置 Claude Agent SDK、CodeBuddy Agent SDK 或 Pi Coding Agent SDK
- **消息通道**: 企业微信（可扩展）+ 浏览器开发页面
- **LLM 配置**: 支持 Anthropic 兼容 API（如 MiniMax）

## 快速开始

```bash
# 安装依赖
npm install --legacy-peer-deps

# 初始化本地测试环境
npm run harness -- --reset

# 启动开发服务
npm run dev
```

浏览器访问 http://127.0.0.1:3000/ ，选择角色后开始对话。

## 可用命令

| 命令                         | 说明                              |
| ---------------------------- | --------------------------------- |
| `npm run dev`                | 启动开发服务                      |
| `npm run harness -- --reset` | 创建/重置本地知识库沙箱           |
| `npm run smoke`              | 运行时回归测试（需要服务运行中）  |
| `npm run smoke:codebuddy`    | CodeBuddy 真实 SDK 回归测试       |
| `npm run smoke:loop`         | Loop 模块真实 LLM 冒烟测试       |
| `npm run db:backup`          | 备份 SQLite 状态库                |
| `npm run db:restore`         | 从备份恢复 SQLite 状态库          |
| `npm run db:verify`          | 校验 SQLite 状态库完整性          |
| `npm run check`              | 类型检查 + lint + 单元测试 + 构建 |
| `npm run check:coverage`     | 带覆盖率报告的完整检查            |
| `npm test`                   | 运行 Vitest 单元测试              |
| `npm run test:coverage`      | 生成 Vitest 覆盖率报告            |
| `npm run build`              | 编译生产版本                      |

## 架构

```
src/
├── index.ts              入口：组装依赖、启动服务
├── core/                 核心层（纯业务逻辑，无外部依赖）
│   ├── index.ts            系统契约
│   └── orchestrator.ts     编排器：路由消息到对应角色 runtime
├── agent/                Agent 适配层
│   ├── index.ts            Agent 模块公开契约
│   ├── claude/             Claude SDK runtime 适配器
│   ├── codebuddy/          CodeBuddy SDK runtime 适配器
│   ├── pi/                 Pi Coding Agent runtime 适配器
│   ├── intent/             意图识别 runtime 包装
│   ├── policy/             工具权限策略
│   └── shared/             SDK runtime 共享转换逻辑
├── server/               传输层（HTTP/SSE）
│   ├── createServer.ts     Fastify 路由
│   ├── dev-chat/           前端资源（html/css/js）
│   └── sseProgressBroker.ts   SSE 进度推送
├── auth/                 授权适配（角色与能力）
├── wechat/               企业微信适配
├── config/               配置加载
├── persistence/          持久化适配（SQLite 与文件存储）
├── workspace/            工作区解析
├── loop/                 Loop 持续执行控制面（Phase 0）
├── logging/              日志
└── harness/              开发沙箱工具
```

依赖方向：外层依赖内层，内层不依赖外层。`core/index.ts` 和各模块 `index.ts` 是公开契约入口。

## Runtime 配置

模型端点和当前启用的 Agent SDK 在 `config/runtime.json` 中配置：

```json
{
  "dbPath": ".harness/state/agent.db",
  "conversationStore": "sqlite",
  "conversationArchiveRetentionDays": 180,
  "conversationArchiveCleanupIntervalMs": 86400000,
  "cron": {
    "enabled": true,
    "jobStore": "sqlite",
    "jobStorePath": ".harness/state/cron-jobs.json"
  },
  "llm": {
    "baseUrl": "https://api.minimaxi.com/anthropic",
    "apiKeyEnv": "LOCAL_LLM_API_KEY",
    "modelId": "MiniMax-M3",
    "maxTokens": 8192
  },
  "agentSdkType": "claude",
  "agentSdks": {
    "claude": {
      "baseUrl": "https://api.minimaxi.com/anthropic",
      "apiKeyEnv": "LOCAL_LLM_API_KEY",
      "modelId": "MiniMax-M3",
      "api": "anthropic-messages",
      "provider": "anthropic",
      "contextWindow": 200000
    },
    "codebuddy": {
      "baseUrl": "https://copilot.tencent.com",
      "modelId": "glm-5.1",
      "endpointEnv": "CODEBUDDY_ENDPOINT",
      "environment": "internal"
    },
    "pi": {
      "baseUrl": "https://api.minimaxi.com/anthropic",
      "apiKeyEnv": "LOCAL_LLM_API_KEY",
      "modelId": "MiniMax-M3"
    }
  }
}
```

`agentSdkType` 必须指向 `agentSdks` 中已有的条目。API 密钥和企业端点通过环境变量提供，不写入配置文件。
`conversationStore` 默认为 `sqlite`，会把多轮对话 session 和 history 写入 `dbPath`；显式设为 `file` 时才使用 `sessionStorePath` 和 `historyStorePath`。
`conversationArchiveRetentionDays` 控制已归档历史的保留期，SQLite 模式启动后会定期清理过期归档并写入 system 日志。
`cron.jobStore` 默认为 `sqlite`，会把定时任务定义和 run state 写入 `dbPath`；显式设为 `file` 时才使用 `cron.jobStorePath`。

## 多轮对话

- SDK 通过 `sessionId` 管理会话上下文，自动支持多轮对话
- 发送 `/new` 可归档当前对话并开始新会话
- 默认对话 session/history 记录在 SQLite `dbPath` 中；轻量 file 模式记录在 `.harness/state/history.json` 中
- SQLite 模式默认保留 conversation history archives 180 天，过期归档由后台任务清理

## 日志

- `.harness/logs/conversation.jsonl` — 用户输入和最终输出
- `.harness/logs/llm-raw.jsonl` — LLM 请求/响应/错误事件
- `.harness/logs/system.jsonl` — 编排、权限、cron、WeCom 连接、`wechat.session_metrics` 周期指标，以及启用 `wechat.rejectWhenConnectionUnavailable` 后的断连期 `wechat.connection_unavailable_message_rejected` 事件；Loop 模块事件（`loop.started`、`loop.step.*`、`loop.verified`、`loop.completed` 等）也写入此日志，携带 `loopRunId`、`stepId`、`attempt` 执行上下文

请勿在日志中记录 API 密钥、授权头或令牌。

## 运维检查

- 启动时会打印 `pets-agent startup` 横幅，集中展示 server、agent sdk、intent LLM、WeCom WSS、cron、knowledge base、logs 和 state 路径，不包含密钥字段。
- `GET /healthz` — 进程存活检查
- `GET /readyz` — 就绪检查，返回 SQLite、企业微信 WSS、cron scheduler 状态；组件失败时返回 503
- cron 启用时默认使用 SQLite 保存 job/run state，并使用 `cron.leaderLeasePath` 文件租约做本地 leader guard；`/cron/status` 会返回当前实例的 `leader` 状态。
- Cron 多副本生产化方案见 `docs/cron-production-plan.md`；跨主机部署前应迁移到外部调度/队列或共享 leader election。
- 持久化数据生命周期见 `docs/persistence-lifecycle.md`，JSONL 只作为审计与排障日志，不作为业务恢复来源。
- SQLite 备份/恢复演练见 `docs/sqlite-backup-restore.md`；恢复前请停止服务，避免覆盖仍被写入的数据库。

## Troubleshooting

| 现象                         | 先看哪里                                                                                                           | 处理方向                                                                                         |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| 企业微信重连或断连期流式失败 | `GET /readyz`、`.harness/logs/system.jsonl` 中的 `wechat.session_metrics` / `wechat.stream.failure`                | 默认会继续处理已收到的消息，并在流式回复失败时尝试 HTTP fallback；若启用断连拒收，再检查拒收计数 |
| LLM 429 或模型调用失败       | `.harness/logs/llm-raw.jsonl` 的 `llm.error`、`.harness/logs/system.jsonl` 的 `runtime.selected` / `context.usage` | 检查对应 `apiKeyEnv`、模型端点、限流窗口和上下文用量；失败后先保留同一 `messageId` 的日志链路    |
| SQLite locked 或持久化异常   | `GET /readyz`、启动横幅中的 `db=` 路径、`.harness/logs/system.jsonl`                                               | 确认只有一个本地写入进程使用同一 SQLite 文件；生产备份用离线副本或受控窗口执行                   |
| cron 多实例或租约异常        | `/cron/status?userId=<admin>`、`cron.leader.*` system 事件、启动横幅中的 `cronLeader=` 路径                        | 本地文件租约只适合同机轻量 guard；跨主机多副本需迁移到共享调度/队列或共享 leader 后端            |

## CI Smoke

`.github/workflows/codebuddy-smoke.yml` 会在工作日定时和手动触发时运行 `npm run check` 与 `npm run smoke:codebuddy`。需要配置 `LOCAL_LLM_API_KEY`，以及 `CODEBUDDY_SMOKE_API_KEY` 或 `CODEBUDDY_AUTH_TOKEN`；可选配置 `CODEBUDDY_SMOKE_ENDPOINT`、`CODEBUDDY_SMOKE_ENVIRONMENT`、`CODEBUDDY_SMOKE_MODEL`。失败时会上传 `.harness/codebuddy-smoke` artifact，保留 7 天用于查看 `llm-raw.jsonl`、`system.jsonl` 和临时 runtime config。

## 已知限制

- 企业微信智能机器人依赖 WSS 长连接；断线重连期间收到的消息会被立即拒绝并提示稍后重试，不进入模型运行时或附件下载流程。
- 定时任务 job/run state 默认已进入 SQLite，但调度触发仍依赖进程内 tick 和本地文件租约；正式多副本部署前仍建议迁移到共享调度/队列和共享 leader election。
- 默认 SQLite conversation store 已避免 JSON 文件多进程同写并清理过期归档；file store 仅建议用于本地轻量模式。
- CodeBuddy 真实 SDK smoke 已接入受控 GitHub Actions；未配置 secrets 时 workflow 会在 preflight 阶段失败并提示缺失项。

## 验证

```bash
npm run check    # 类型检查、lint、单元测试、构建
npm run check:coverage # 同 check，但生成覆盖率报告，不强制阈值
npm run smoke    # 运行时回归测试（需先 npm run dev）
```
