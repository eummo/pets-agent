# Pets Agent

企业微信连接的 Claude 知识库代理服务。

本项目包含开发工作流和本地测试环境（harness），用于安全地构建代理应用。

## 项目概述

Pets Agent 是一个基于知识库的自然语言问答服务，它：
- 连接企业微信作为消息通道
- 使用 Claude 大语言模型提供智能问答
- 支持多知识库/多工作空间管理
- 提供本地开发测试环境

## 技术栈

- **运行时**: Node.js + TypeScript
- **框架**: Fastify
- **AI SDK**: Claude Code SDK
- **消息通道**: 企业微信（可扩展的 MessageChannel 接口）
- **LLM 配置**: 支持本地兼容 API（如 MiniMax）

## 快速开始

### 安装依赖

```bash
npm install
```

### 初始化本地测试环境

```bash
npm run harness -- --reset
```

### 启动开发服务

```bash
npm run dev
```

### 运行检查

```bash
npm run check
```

## 可用命令

| 命令 | 说明 |
|------|------|
| `npm run dev` | 启动 Fastify 开发服务 |
| `npm run harness -- --reset` | 创建本地知识库沙箱 |
| `npm run smoke` | 运行浏览器/运行时回归测试 |
| `npm run typecheck` | 运行 TypeScript 类型检查 |
| `npm run lint` | 运行 ESLint 代码检查 |
| `npm test` | 运行 Vitest 单元测试 |
| `npm run build` | 编译生产版本 |
| `npm run check` | 运行类型检查、lint、测试和构建 |

## 开发环境

### 测试页面

开发服务启动后，访问：

```
http://127.0.0.1:3000/
```

### Harness 目录结构

```
.harness/
  knowledge-base/
    CLAUDE.md          # 知识库说明
    docs/              # 文档目录
    requirements/      # 需求目录
    code/              # 代码仓库
      catalog-api/     # 目录 API 服务
      order-service/   # 订单服务
  repos.json
```

### 日志文件

- `.harness/logs/conversation.jsonl` - 用户输入和最终输出
- `.harness/logs/llm-raw.jsonl` - LLM 请求/响应/错误事件

**注意**: 请勿在日志中记录 API 密钥、密钥、授权头、访问令牌或刷新令牌。

## 架构设计

项目采用端口/适配器（Ports/Adapters）架构，确保provider特定代码与核心逻辑解耦：

- 消息通道实现统一的 `MessageChannel` 接口
- Claude Code SDK 实现第一个 `AgentRuntime`
- 未来 SDK 可添加新的 `AgentRuntime` 适配器而不改变编排逻辑
- GitHub PR 发布逻辑封装在 `ChangePublisher` 后面

详细设计请参考：
- [docs/architecture.md](docs/architecture.md) - 架构文档
- [docs/development-workflow.md](docs/development-workflow.md) - 开发工作流
- [docs/multi-agent-team-design.md](docs/multi-agent-team-design.md) - 多代理设计

## LLM 配置

本地兼容模型端点在 `config/llm.json` 中配置：

```json
{
  "baseUrl": "https://api.minimaxi.com/anthropic",
  "modelId": "MiniMax-M2",
  "apiKeyEnv": "LOCAL_LLM_API_KEY"
}
```

实际 API 密钥仅存储在环境变量中。

### 多轮对话

默认情况下，消息运行时在 `.harness/state/history.json` 中按渠道、用户和工作空间维护本地多轮历史记录。发送 `/new` 可归档当前历史并开始新的对话。

如需使用 SDK 托管的多轮会话，请在 `config/llm.json` 中设置 `runtime` 为 `managed-sessions`，并提供：
- `agentIdEnv`: 包含 Managed Agents `agent_...` ID 的环境变量
- `environmentIdEnv`: 包含 Managed Agents `env_...` ID 的环境变量

## 开发工作流

1. 阅读 `docs/development-workflow.md` 了解实现变更流程
2. 阅读 `docs/architecture.md` 了解端口/适配器边界
3. 保持 provider 特定代码在适配器后面，编排层不直接依赖企业微信、Claude Code、MiniMax、GitHub 等

### 验证步骤

完成变更后，运行以下命令验证：

```bash
npm run check
npm run smoke
```

## 项目文件

- `AGENTS.md` - Agent 项目指南
- `CLAUDE.md` - 项目级指令（用于 Codex 和编码代理）
- `docs/` - 架构和开发文档
- `config/` - 配置文件
- `src/` - 源代码