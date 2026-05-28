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
| `npm run check`              | 类型检查 + lint + 单元测试 + 构建 |
| `npm test`                   | 运行 Vitest 单元测试              |
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
├── logging/              日志
└── harness/              开发沙箱工具
```

依赖方向：外层依赖内层，内层不依赖外层。`core/index.ts` 和各模块 `index.ts` 是公开契约入口。

## LLM 配置

兼容模型端点在 `config/llm.json` 中配置：

```json
{
  "baseUrl": "https://api.minimaxi.com/anthropic",
  "modelId": "MiniMax-M2.7",
  "apiKeyEnv": "LOCAL_LLM_API_KEY"
}
```

API 密钥通过环境变量提供，不写入配置文件。

## 多轮对话

- SDK 通过 `sessionId` 管理会话上下文，自动支持多轮对话
- 发送 `/new` 可归档当前对话并开始新会话
- 对话历史记录在 `.harness/state/history.json` 中

## 日志

- `.harness/logs/conversation.jsonl` — 用户输入和最终输出
- `.harness/logs/llm-raw.jsonl` — LLM 请求/响应/错误事件

请勿在日志中记录 API 密钥、授权头或令牌。

## 验证

```bash
npm run check    # 类型检查、lint、单元测试、构建
npm run smoke    # 运行时回归测试（需先 npm run dev）
```
