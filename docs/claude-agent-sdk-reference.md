# Claude Agent SDK 中文文档

> **SDK 版本**：v0.3.160（2026-06-02 更新确认）
> 完整文档来源：https://code.claude.com/docs/en/agent-sdk/overview

---

## 目录

- [概览](#概览)
- [快速入门](#快速入门)
- [代理循环](#代理循环)
- [TypeScript API 参考](#typescript-api-参考)
  - [安装](#安装)
  - [函数](#函数)
  - [类型](#类型)
  - [消息类型](#消息类型)
  - [钩子类型](#钩子类型)
  - [工具输入/输出类型](#工具输入输出类型)
  - [权限类型](#权限类型)
  - [沙箱配置](#沙箱配置)
  - [其他类型](#其他类型)
- [指南](#指南)
  - [钩子](#钩子)
  - [子代理](#子代理)
  - [MCP 集成](#mcp-集成)
  - [权限配置](#权限配置)
  - [用户输入处理](#用户输入处理)
  - [会话管理](#会话管理)
  - [自定义工具](#自定义工具)
  - [流式输出](#流式输出)
  - [流式输入](#流式输入)
  - [结构化输出](#结构化输出)
  - [成本跟踪](#成本跟踪)
  - [托管部署](#托管部署)
  - [安全部署](#安全部署)

---

# 概览

构建能够自主读取文件、运行命令、搜索网络、编辑代码等的 AI 代理。Agent SDK 提供了与 Claude Code 相同的工具、代理循环和上下文管理能力，可通过 **Python** 和 **TypeScript** 编程控制。

### TypeScript 示例

```typescript
import { query } from "@anthropic-ai/claude-agent-sdk";

for await (const message of query({
  prompt: "Find and fix the bug in auth.ts",
  options: { allowedTools: ["Read", "Edit", "Bash"] }
})) {
  console.log(message);
}
```

## 核心能力

| 能力 | 说明 |
|:---|:---|
| **内置工具** | Read、Write、Edit、Bash、Monitor、Glob、Grep、WebSearch、WebFetch、AskUserQuestion |
| **钩子** | 在代理生命周期的关键点运行自定义代码，验证、记录、阻止或转换代理行为 |
| **子代理** | 生成专门的代理处理聚焦子任务，支持上下文隔离、并行执行和专业化指令 |
| **MCP** | 通过模型上下文协议连接数据库、浏览器、API 等外部系统 |
| **权限** | 控制代理可使用的工具，允许安全操作、阻止危险操作或要求审批 |
| **会话** | 跨多次交互维护上下文，支持恢复、分叉会话 |

## 与其他 Claude 工具的比较

### Agent SDK vs Client SDK

Client SDK 提供**直接 API 访问**：你发送提示并自己实现工具执行。Agent SDK 提供**内置工具执行**的 Claude。

### Agent SDK vs Claude Code CLI

| 用例 | 最佳选择 |
|:---|:---|
| 交互式开发 | CLI |
| CI/CD 管道 | SDK |
| 自定义应用 | SDK |
| 一次性任务 | CLI |
| 生产自动化 | SDK |

### Agent SDK vs Managed Agents

| | Agent SDK | Managed Agents |
|:---|:---|:---|
| **运行位置** | 你的进程、你的基础设施 | Anthropic 管理的基础设施 |
| **接口** | Python 或 TypeScript 库 | REST API |
| **代理工作对象** | 你基础设施上的文件 | 每会话一个托管沙箱 |
| **最佳用途** | 本地原型设计、直接操作文件系统和服务的代理 | 不需要运行沙箱或会话基础设施的生产代理 |

## 第三方 API 提供商

| 提供商 | 环境变量 |
|:---|:---|
| **Amazon Bedrock** | `CLAUDE_CODE_USE_BEDROCK=1` + AWS 凭证 |
| **Claude Platform on AWS** | `CLAUDE_CODE_USE_ANTHROPIC_AWS=1` + `ANTHROPIC_AWS_WORKSPACE_ID` + AWS 凭证 |
| **Google Vertex AI** | `CLAUDE_CODE_USE_VERTEX=1` + Google Cloud 凭证 |
| **Microsoft Azure** | `CLAUDE_CODE_USE_FOUNDRY=1` + Azure 凭证 |

---

# 快速入门

## 前提条件

- **Node.js 18+** 或 **Python 3.10+**
- Anthropic 账户

## 步骤

### 1. 安装 SDK

```bash
# TypeScript
npm install @anthropic-ai/claude-agent-sdk

# Python (uv)
uv init && uv add claude-agent-sdk

# Python (pip)
python3 -m venv .venv && source .venv/bin/activate
pip3 install claude-agent-sdk
```

### 2. 设置 API 密钥

```bash
export ANTHROPIC_API_KEY=your-api-key
```

### 3. 运行第一个代理

```typescript
import { query } from "@anthropic-ai/claude-agent-sdk";

for await (const message of query({
  prompt: "What files are in this directory?",
  options: { allowedTools: ["Bash", "Glob"] }
})) {
  if ("result" in message) console.log(message.result);
}
```

## 代码三要素

1. **`query`**：主入口，创建代理循环，返回异步迭代器流式传输消息
2. **`prompt`**：你希望 Claude 做什么
3. **`options`**：代理配置 — `allowedTools` 预批准工具，`permissionMode` 控制审批行为

## 关键概念

### 工具 — 代理能做什么

| 工具 | 代理能力 |
|:---|:---|
| `Read`、`Glob`、`Grep` | 只读分析 |
| `Read`、`Edit`、`Glob` | 分析和修改代码 |
| `Read`、`Edit`、`Bash`、`Glob`、`Grep` | 完全自动化 |

### 权限模式 — 人工监督程度

| 模式 | 行为 | 用例 |
|:---|:---|:---|
| `acceptEdits` | 自动批准文件编辑和常见文件系统命令 | 受信任的开发工作流 |
| `dontAsk` | 拒绝 `allowedTools` 之外的任何操作 | 锁定的无头代理 |
| `auto` | 模型分类器批准或拒绝每个工具调用 | 带安全护栏的自主代理 |
| `bypassPermissions` | 运行每个工具无需提示 | 沙箱 CI、完全信任的环境 |
| `default` | 需要 `canUseTool` 回调处理审批 | 自定义审批流程 |

---

# 代理循环

## 循环概览

每个代理会话遵循此循环：

1. **接收提示** — Claude 接收你的提示、系统提示、工具定义和对话历史
2. **评估和响应** — Claude 评估当前状态，可能响应文本、请求工具调用或两者兼有
3. **执行工具** — SDK 运行每个请求的工具并收集结果
4. **重复** — 步骤 2 和 3 循环执行。每个完整循环是一轮
5. **返回结果** — SDK 产出最终的 `ResultMessage`

## 轮次和消息

一轮是一次往返：Claude 产出带工具调用的输出 → SDK 执行工具 → 结果自动反馈给 Claude。轮次持续直到 Claude 产出没有工具调用的响应。

### 示例会话："修复 auth.ts 中的失败测试"

| 轮次 | 动作 | 产出的消息 |
|:-----|:-------|:------------|
| 初始化 | SDK 发送提示给 Claude | `SystemMessage`（会话元数据） |
| 第 1 轮 | Claude 调用 `Bash` 运行 `npm test` | `AssistantMessage` → `UserMessage`（3 个测试失败） |
| 第 2 轮 | Claude 调用 `Read` 读取文件 | `AssistantMessage` → 文件内容返回 |
| 第 3 轮 | Claude 调用 `Edit` 修改代码，然后 `Bash` 重跑测试 | `AssistantMessage` → 全部通过 |
| 最终轮 | Claude 产出文本响应 | `AssistantMessage` → `ResultMessage` |

## 限制

- `maxTurns` — 限制工具使用轮次
- `maxBudgetUsd` — 限制花费阈值

## 上下文窗口

上下文窗口是 Claude 在会话中可用的总信息量，不会在轮次间重置。

### 消耗上下文的来源

| 来源 | 加载时机 | 影响 |
|:---|:---|:---|
| 系统提示 | 每次请求 | 小的固定开销 |
| CLAUDE.md 文件 | 会话启动 | 每次请求中完整内容（提示缓存） |
| 工具定义 | 每次请求 | 内置模式每次加载；工具搜索延迟 MCP 模式 |
| 对话历史 | 随轮次累积 | 随轮次增长 |
| 技能描述 | 会话启动 | 短摘要；仅在调用时加载完整内容 |

### 自动压缩

当上下文窗口接近限制时，SDK 自动压缩对话：将较旧的历史摘要化以释放空间。SDK 发出 `compact_boundary` 消息。

**重要：** 压缩用摘要替换旧消息，因此早期对话中的特定指令可能不被保留。**持久规则应放在 CLAUDE.md 中**（每次请求重新注入），而非初始提示。

### 高效上下文策略

- 使用子代理处理子任务 — 每个从全新对话开始
- 选择性使用工具 — 每个工具定义占用上下文空间
- 为常规任务使用较低 effort — 减少令牌使用和成本

---

# TypeScript API 参考

`@anthropic-ai/claude-agent-sdk` 的完整 API 参考。

## 安装

```bash
npm install @anthropic-ai/claude-agent-sdk
```

SDK 将原生 Claude Code 二进制文件作为可选依赖打包（例如 `@anthropic-ai/claude-agent-sdk-darwin-arm64`），无需单独安装 Claude Code。如果包管理器跳过了可选依赖，SDK 会抛出 `Native CLI binary for <platform> not found` 错误；此时可通过 [`pathToClaudeCodeExecutable`](#options) 选项指向已安装的 `claude` 二进制文件。

### 编译为单一可执行文件

使用 `bun build --compile` 编译时，需使用 `extractFromBunfs()` 辅助函数将二进制文件提取到真实路径。

```typescript
import binPath from "@anthropic-ai/claude-agent-sdk-darwin-arm64/claude" with { type: "file" };
import { extractFromBunfs } from "@anthropic-ai/claude-agent-sdk/extract";
import { query } from "@anthropic-ai/claude-agent-sdk";

const cliPath = extractFromBunfs(binPath);

for await (const message of query({
  prompt: "Hello",
  options: { pathToClaudeCodeExecutable: cliPath },
})) {
  console.log(message);
}
```

交叉编译时需安装对应平台的包，例如 `npm install @anthropic-ai/claude-agent-sdk-linux-x64 --force`。Windows 上二进制子路径为 `claude.exe`。

## 函数

### `query()`

与 Claude Code 交互的主要函数。创建一个异步生成器，流式返回消息。

```typescript
function query({
  prompt,
  options
}: {
  prompt: string | AsyncIterable<SDKUserMessage>;
  options?: Options;
}): Query;
```

| 参数 | 类型 | 说明 |
|:---|:---|:---|
| `prompt` | `string \| AsyncIterable<SDKUserMessage>` | 输入提示，字符串或异步可迭代对象（流式模式） |
| `options` | `Options` | 可选配置对象 |

返回 `Query` 对象，扩展了 `AsyncGenerator<SDKMessage, void>` 并附加了其他方法。

### `startup()`

预热线 CLI 子进程，在提示可用之前完成初始化握手。

```typescript
function startup(params?: {
  options?: Options;
  initializeTimeoutMs?: number;
}): Promise<WarmQuery>;
```

```typescript
import { startup } from "@anthropic-ai/claude-agent-sdk";

const warm = await startup({ options: { maxTurns: 3 } });

for await (const message of warm.query("What files are here?")) {
  console.log(message);
}
```

### `tool()`

为 SDK MCP 服务器创建类型安全的 MCP 工具定义。

```typescript
function tool<Schema extends AnyZodRawShape>(
  name: string,
  description: string,
  inputSchema: Schema,
  handler: (args: InferShape<Schema>, extra: unknown) => Promise<CallToolResult>,
  extras?: { annotations?: ToolAnnotations }
): SdkMcpToolDefinition<Schema>;
```

**ToolAnnotations 字段：**

| 字段 | 类型 | 默认值 | 说明 |
|:---|:---|:---|:---|
| `title` | `string` | `undefined` | 人类可读的工具标题 |
| `readOnlyHint` | `boolean` | `false` | 为 `true` 时工具不修改环境 |
| `destructiveHint` | `boolean` | `true` | 为 `true` 时工具可能执行破坏性更新 |
| `idempotentHint` | `boolean` | `false` | 为 `true` 时相同参数的重复调用无额外效果 |
| `openWorldHint` | `boolean` | `true` | 为 `true` 时工具与外部实体交互 |

```typescript
import { tool } from "@anthropic-ai/claude-agent-sdk";
import { z } from "zod";

const searchTool = tool(
  "search",
  "Search the web",
  { query: z.string() },
  async ({ query }) => {
    return { content: [{ type: "text", text: `Results for: ${query}` }] };
  },
  { annotations: { readOnlyHint: true, openWorldHint: true } }
);
```

### `createSdkMcpServer()`

创建在与应用程序相同进程中运行的 MCP 服务器实例。

```typescript
function createSdkMcpServer(options: {
  name: string;
  version?: string;
  tools?: Array<SdkMcpToolDefinition<any>>;
}): McpSdkServerConfigWithInstance;
```

### `listSessions()`

发现并列出过去的会话及其轻量级元数据。

```typescript
function listSessions(options?: ListSessionsOptions): Promise<SDKSessionInfo[]>;
```

**SDKSessionInfo 返回类型：**

| 属性 | 类型 | 说明 |
|:---|:---|:---|
| `sessionId` | `string` | 唯一会话标识符 (UUID) |
| `summary` | `string` | 显示标题 |
| `lastModified` | `number` | 最后修改时间（毫秒） |
| `fileSize` | `number \| undefined` | 会话文件大小（字节） |
| `customTitle` | `string \| undefined` | 用户设置的会话标题 |
| `firstPrompt` | `string \| undefined` | 会话中第一个有意义的用户提示 |
| `gitBranch` | `string \| undefined` | 会话结束时的 Git 分支 |
| `cwd` | `string \| undefined` | 会话的工作目录 |
| `tag` | `string \| undefined` | 用户设置的会话标签 |
| `createdAt` | `number \| undefined` | 创建时间（毫秒） |

### `getSessionMessages()`

读取过去会话记录中的用户和助手消息。

### `getSessionInfo()`

通过 ID 读取单个会话的元数据，无需扫描整个项目目录。

### `renameSession()`

通过追加自定义标题条目来重命名会话。

### `tagSession()`

为会话添加标签。传入 `null` 清除标签。

### `resolveSettings()`

使用与 CLI 相同的合并引擎解析给定目录的有效 Claude Code 设置，无需启动 Claude CLI。

```typescript
const { effective, provenance } = await resolveSettings({
  cwd: "/path/to/project",
  settingSources: ["user", "project", "local"],
});
```

## 类型

### `Options`

`query()` 函数的配置对象。

| 属性 | 类型 | 默认值 | 说明 |
|:---|:---|:---|:---|
| `abortController` | `AbortController` | `new AbortController()` | 取消操作的控制器 |
| `additionalDirectories` | `string[]` | `[]` | Claude 可访问的附加目录 |
| `agent` | `string` | `undefined` | 主线程代理名称 |
| `agents` | `Record<string, AgentDefinition>` | `undefined` | 编程定义子代理 |
| `agentProgressSummaries` | `boolean` | `false` | 为子代理生成进度摘要 |
| `allowDangerouslySkipPermissions` | `boolean` | `false` | 启用绕过权限 |
| `allowedTools` | `string[]` | `[]` | 自动批准的工具 |
| `betas` | `SdkBeta[]` | `[]` | 启用 Beta 功能 |
| `canUseTool` | `CanUseTool` | `undefined` | 自定义工具权限函数 |
| `continue` | `boolean` | `false` | 继续最近的对话 |
| `cwd` | `string` | `process.cwd()` | 当前工作目录 |
| `debug` | `boolean` | `false` | 启用调试模式 |
| `debugFile` | `string` | `undefined` | 将调试日志写入指定文件 |
| `disallowedTools` | `string[]` | `[]` | 拒绝的工具 |
| `effort` | `'low'\|'medium'\|'high'\|'xhigh'\|'max'` | `'high'` | 控制 Claude 的努力程度 |
| `enableFileCheckpointing` | `boolean` | `false` | 启用文件更改跟踪 |
| `env` | `Record<string, string \| undefined>` | `process.env` | 环境变量 |
| `executable` | `'bun'\|'deno'\|'node'` | 自动检测 | JavaScript 运行时 |
| `fallbackModel` | `string` | `undefined` | 备选模型 |
| `forkSession` | `boolean` | `false` | 使用 `resume` 时分叉到新会话 |
| `forwardSubagentText` | `boolean` | `false` | 转发子代理文本和思考块 |
| `hooks` | `Partial<Record<HookEvent, HookCallbackMatcher[]>>` | `{}` | 事件钩子回调 |
| `includeHookEvents` | `boolean` | `false` | 在消息流中包含钩子事件 |
| `includePartialMessages` | `boolean` | `false` | 包含部分消息事件 |
| `maxBudgetUsd` | `number` | `undefined` | 达到此 USD 值时停止查询 |
| `maxTurns` | `number` | `undefined` | 最大代理轮次 |
| `mcpServers` | `Record<string, McpServerConfig>` | `{}` | MCP 服务器配置 |
| `model` | `string` | CLI 默认 | Claude 模型 |
| `onElicitation` | `(request, options) => Promise<ElicitationResult>` | `undefined` | MCP 请求用户输入的回调 |
| `outputFormat` | `{ type: 'json_schema', schema: JSONSchema }` | `undefined` | 定义代理结果的输出格式 |
| `pathToClaudeCodeExecutable` | `string` | 自动解析 | Claude Code 可执行文件路径 |
| `permissionMode` | `PermissionMode` | `'default'` | 权限模式 |
| `persistSession` | `boolean` | `true` | 为 `false` 时禁用会话持久化 |
| `plugins` | `SdkPluginConfig[]` | `[]` | 自定义插件 |
| `promptSuggestions` | `boolean` | `false` | 启用提示建议 |
| `resume` | `string` | `undefined` | 要恢复的会话 ID |
| `sandbox` | `SandboxSettings` | `undefined` | 沙箱行为配置 |
| `sessionId` | `string` | 自动生成 | 指定会话 UUID |
| `sessionStore` | `SessionStore` | `undefined` | 会话存储适配器 |
| `settings` | `string \| Settings` | `undefined` | 内联设置或设置文件路径 |
| `settingSources` | `SettingSource[]` | CLI 默认 | 控制加载哪些文件系统设置 |
| `skills` | `string[] \| 'all'` | `undefined` | 会话可用的技能 |
| `spawnClaudeCodeProcess` | `(options) => SpawnedProcess` | `undefined` | 自定义进程生成函数 |
| `stderr` | `(data: string) => void` | `undefined` | stderr 输出回调 |
| `strictMcpConfig` | `boolean` | `false` | 仅使用 `mcpServers` 中的服务器 |
| `systemPrompt` | `string \| { type: 'preset'; preset: 'claude_code'; append?: string; excludeDynamicSections?: boolean }` | `undefined` | 系统提示配置 |
| `thinking` | `ThinkingConfig` | `{ type: 'adaptive' }` | 控制 Claude 的思考/推理行为 |
| `title` | `string` | `undefined` | 会话的显示标题 |
| `toolAliases` | `Record<string, string>` | `undefined` | 将内置工具名映射到 MCP 工具名 |
| `toolConfig` | `ToolConfig` | `undefined` | 内置工具行为配置 |
| `tools` | `string[] \| { type: 'preset'; preset: 'claude_code' }` | `undefined` | 工具配置 |

#### 处理缓慢或停滞的 API 响应

```typescript
const result = query({
  prompt: "Analyze this code",
  options: {
    env: {
      ...process.env,
      API_TIMEOUT_MS: "120000",
      CLAUDE_CODE_MAX_RETRIES: "2",
      CLAUDE_ASYNC_AGENT_STALL_TIMEOUT_MS: "120000",
    },
  },
});
```

- `API_TIMEOUT_MS`：每次请求超时（毫秒），默认 `600000`
- `CLAUDE_CODE_MAX_RETRIES`：最大重试次数，默认 `10`
- `CLAUDE_ASYNC_AGENT_STALL_TIMEOUT_MS`：后台子代理停滞超时，默认 `600000`

### `Query` 对象

```typescript
interface Query extends AsyncGenerator<SDKMessage, void> {
  interrupt(): Promise<void>;
  rewindFiles(userMessageId: string, options?: { dryRun?: boolean }): Promise<RewindFilesResult>;
  setPermissionMode(mode: PermissionMode): Promise<void>;
  setModel(model?: string): Promise<void>;
  applyFlagSettings(settings: { [K in keyof Settings]?: Settings[K] | null }): Promise<void>;
  initializationResult(): Promise<SDKControlInitializeResponse>;
  supportedCommands(): Promise<SlashCommand[]>;
  supportedModels(): Promise<ModelInfo[]>;
  supportedAgents(): Promise<AgentInfo[]>;
  mcpServerStatus(): Promise<McpServerStatus[]>;
  accountInfo(): Promise<AccountInfo>;
  reconnectMcpServer(serverName: string): Promise<void>;
  toggleMcpServer(serverName: string, enabled: boolean): Promise<void>;
  setMcpServers(servers: Record<string, McpServerConfig>): Promise<McpSetServersResult>;
  streamInput(stream: AsyncIterable<SDKUserMessage>): Promise<void>;
  stopTask(taskId: string): Promise<void>;
  close(): void;
}
```

### `WarmQuery`

`startup()` 返回的句柄。支持 `await using` 自动清理。

```typescript
interface WarmQuery extends AsyncDisposable {
  query(prompt: string | AsyncIterable<SDKUserMessage>): Query;
  close(): void;
}
```

### `AgentDefinition`

```typescript
type AgentDefinition = {
  description: string;           // 必填：何时使用此代理
  prompt: string;               // 必填：代理的系统提示
  tools?: string[];              // 允许的工具，省略则继承父级
  disallowedTools?: string[];   // 禁止的工具
  model?: string;                // 模型别名或 ID
  mcpServers?: AgentMcpServerSpec[];
  skills?: string[];
  initialPrompt?: string;
  maxTurns?: number;
  background?: boolean;
  memory?: "user" | "project" | "local";
  effort?: "low" | "medium" | "high" | "xhigh" | "max" | number;
  permissionMode?: PermissionMode;
};
```

### `PermissionMode`

```typescript
type PermissionMode =
  | "default"            // 标准权限行为
  | "acceptEdits"        // 自动接受文件编辑
  | "bypassPermissions"  // 绕过所有权限检查
  | "plan"               // 计划模式 - 仅限只读工具
  | "dontAsk"            // 不提示权限，未预批准则拒绝
  | "auto";              // 使用模型分类器
```

### `CanUseTool` / `PermissionResult`

```typescript
type CanUseTool = (
  toolName: string,
  input: Record<string, unknown>,
  options: { signal: AbortSignal; suggestions?: PermissionUpdate[]; ... }
) => Promise<PermissionResult>;

type PermissionResult =
  | { behavior: "allow"; updatedInput?: Record<string, unknown>; updatedPermissions?: PermissionUpdate[] }
  | { behavior: "deny"; message: string; interrupt?: boolean };
```

### `McpServerConfig`

```typescript
type McpServerConfig =
  | McpStdioServerConfig    // { type?: "stdio"; command: string; args?: string[]; env?: Record<string, string> }
  | McpSSEServerConfig      // { type: "sse"; url: string; headers?: Record<string, string> }
  | McpHttpServerConfig     // { type: "http"; url: string; headers?: Record<string, string> }
  | McpSdkServerConfigWithInstance; // { type: "sdk"; name: string; instance: McpServer }
```

### `ThinkingConfig`

```typescript
type ThinkingConfig =
  | { type: "adaptive"; display?: "summarized" | "omitted" }   // 模型决定推理量
  | { type: "enabled"; budgetTokens?: number; display?: ... }  // 固定思考令牌预算
  | { type: "disabled" };  // 无扩展思考
```

### `SettingSource`

| 值 | 说明 | 位置 |
|:---|:---|:---|
| `'user'` | 全局用户设置 | `~/.claude/settings.json` |
| `'project'` | 共享项目设置 | `.claude/settings.json` |
| `'local'` | 本地项目设置 | `.claude/settings.local.json` |

## 消息类型

### `SDKMessage`

查询返回的所有可能消息的联合类型。

```typescript
type SDKMessage =
  | SDKAssistantMessage | SDKUserMessage | SDKUserMessageReplay
  | SDKResultMessage | SDKSystemMessage | SDKPartialAssistantMessage
  | SDKCompactBoundaryMessage | SDKStatusMessage | SDKLocalCommandOutputMessage
  | SDKHookStartedMessage | SDKHookProgressMessage | SDKHookResponseMessage
  | SDKPluginInstallMessage | SDKToolProgressMessage | SDKAuthStatusMessage
  | SDKTaskNotificationMessage | SDKTaskStartedMessage | SDKTaskProgressMessage
  | SDKTaskUpdatedMessage | SDKSessionStateChangedMessage | SDKNotificationMessage
  | SDKFilesPersistedEvent | SDKToolUseSummaryMessage | SDKMemoryRecallMessage
  | SDKRateLimitEvent | SDKElicitationCompleteMessage | SDKPermissionDeniedMessage
  | SDKPromptSuggestionMessage | SDKAPIRetryMessage | SDKMirrorErrorMessage;
```

### `SDKAssistantMessage`

```typescript
type SDKAssistantMessage = {
  type: "assistant";
  uuid: UUID;
  session_id: string;
  message: BetaMessage;
  parent_tool_use_id: string | null;
  error?: 'authentication_failed' | 'rate_limit' | 'model_not_found' | 'server_error' | ... ;
};
```

### `SDKResultMessage`

成功时包含 `result`、`total_cost_usd`、`usage`、`session_id` 等。错误时 `subtype` 为 `error_max_turns`、`error_during_execution`、`error_max_budget_usd` 或 `error_max_structured_output_retries`。

### `SDKMessageOrigin`

```typescript
type SDKMessageOrigin =
  | { kind: "human" }                    // 最终用户直接输入
  | { kind: "channel"; server: string }   // 来自通道
  | { kind: "peer"; from: string }        // 来自其他代理
  | { kind: "task-notification" }         // 后台任务完成
  | { kind: "coordinator" };              // 来自团队协调器
```

## 钩子类型

### `HookEvent`

```typescript
type HookEvent =
  | "PreToolUse" | "PostToolUse" | "PostToolUseFailure" | "PostToolBatch"
  | "Notification" | "UserPromptSubmit"
  | "SessionStart" | "SessionEnd" | "Stop"
  | "SubagentStart" | "SubagentStop"
  | "PreCompact" | "PermissionRequest"
  | "Setup" | "TeammateIdle" | "TaskCompleted"
  | "ConfigChange" | "WorktreeCreate" | "WorktreeRemove";
```

### `HookCallbackMatcher`

```typescript
interface HookCallbackMatcher {
  matcher?: string;       // 正则模式匹配工具名
  hooks: HookCallback[];  // 回调函数数组
  timeout?: number;       // 超时（秒）
}
```

### 钩子输出

**PreToolUse 钩子特定输出：**

- `permissionDecision`：`"allow" | "deny" | "ask" | "defer"`
- `updatedInput`：更新后的工具输入
- `additionalContext`：附加上下文

**优先级：** deny > defer > ask > allow

## 工具输入/输出类型

### 主要工具输入

| 工具 | 关键字段 |
|:---|:---|
| `Agent` | `description`, `prompt`, `subagent_type`, `model?`, `run_in_background?`, `max_turns?` |
| `Bash` | `command`, `timeout?`, `run_in_background?`, `dangerouslyDisableSandbox?` |
| `Read` | `file_path`, `offset?`, `limit?` |
| `Edit` | `file_path`, `old_string`, `new_string`, `replace_all?` |
| `Write` | `file_path`, `content` |
| `Glob` | `pattern`, `path?` |
| `Grep` | `pattern`, `path?`, `output_mode?`, `-i?`, `-C?` |
| `WebFetch` | `url`, `prompt` |
| `WebSearch` | `query`, `allowed_domains?`, `blocked_domains?` |
| `AskUserQuestion` | `questions` 数组，含 `question`、`header`、`options`、`multiSelect` |
| `TaskCreate` | `subject`, `description` |
| `TaskUpdate` | `taskId`, `status?`, `subject?`, `description?` |

### 主要工具输出

| 工具 | 输出说明 |
|:---|:---|
| `Agent` | `status: "completed"` / `"async_launched"` / `"sub_agent_entered"` |
| `Bash` | `stdout`, `stderr`, `interrupted`, `backgroundTaskId?` |
| `Read` | 按 `type` 区分：text / image / notebook / pdf / parts |
| `Edit` | `filePath`, `structuredPatch` |
| `Glob` | `filenames[]`, `numFiles`, `truncated` |
| `Grep` | `filenames[]`, `content?`, `numMatches?` |

## 权限类型

### `PermissionUpdate`

```typescript
type PermissionUpdate =
  | { type: "addRules"; rules: PermissionRuleValue[]; behavior: PermissionBehavior; destination: PermissionUpdateDestination }
  | { type: "replaceRules"; ... }
  | { type: "removeRules"; ... }
  | { type: "setMode"; mode: PermissionMode; destination: ... }
  | { type: "addDirectories"; directories: string[]; destination: ... }
  | { type: "removeDirectories"; directories: string[]; destination: ... };
```

### `PermissionUpdateDestination`

`"userSettings"` | `"projectSettings"` | `"localSettings"` | `"session"` | `"cliArg"`

## 沙箱配置

### `SandboxSettings`

| 属性 | 类型 | 默认值 | 说明 |
|:---|:---|:---|:---|
| `enabled` | `boolean` | `false` | 启用沙箱模式 |
| `autoAllowBashIfSandboxed` | `boolean` | `true` | 沙箱启用时自动批准 bash |
| `excludedCommands` | `string[]` | `[]` | 始终绕过沙箱的命令 |
| `allowUnsandboxedCommands` | `boolean` | `true` | 允许模型请求沙箱外执行 |
| `network` | `SandboxNetworkConfig` | `undefined` | 网络沙箱配置 |
| `filesystem` | `SandboxFilesystemConfig` | `undefined` | 文件系统沙箱配置 |

### `SandboxNetworkConfig`

`allowedDomains?`, `deniedDomains?`, `allowLocalBinding?`, `allowUnixSockets?`, `httpProxyPort?`, `socksProxyPort?`

### `SandboxFilesystemConfig`

`allowWrite?`, `denyWrite?`, `denyRead?` — 文件路径模式数组

## 其他类型

| 类型 | 说明 |
|:---|:---|
| `ApiKeySource` | `"user" \| "project" \| "org" \| "temporary" \| "oauth"` |
| `ModelInfo` | `value`, `displayName`, `description`, `supportsEffort?` |
| `AgentInfo` | `name`, `description`, `model?` |
| `McpServerStatus` | `name`, `status`, `serverInfo?`, `error?`, `tools?` |
| `AccountInfo` | `email?`, `organization?`, `subscriptionType?` |
| `ModelUsage` | `inputTokens`, `outputTokens`, `cacheReadInputTokens`, `costUSD` |
| `CallToolResult` | `content[]`, `structuredContent?`, `isError?` |
| `RewindFilesResult` | `canRewind`, `filesChanged?`, `insertions?`, `deletions?` |

---

# 指南

## 钩子

钩子是在代理事件触发时运行的回调函数，使你能在关键执行点拦截和自定义代理行为。

### 核心用例

- **阻止危险操作** — 在执行前阻止破坏性 shell 命令
- **记录和审计** — 为合规、调试或分析记录每个工具调用
- **转换输入和输出** — 清理数据、注入凭证或重定向文件路径
- **要求人工审批** — 对敏感操作如数据库写入或 API 调用
- **跟踪会话生命周期** — 管理状态、清理资源或发送通知

### 5 步生命周期

1. 事件触发
2. SDK 收集已注册的钩子
3. 匹配器过滤哪些钩子运行
4. 回调函数执行
5. 你的回调返回决策

### 可用钩子事件

| 钩子事件 | 触发时机 | 示例用例 |
|:---|:---|:---|
| `PreToolUse` | 工具调用请求（可阻止或修改） | 阻止危险命令 |
| `PostToolUse` | 工具执行结果 | 记录文件变更 |
| `PostToolUseFailure` | 工具执行失败 | 处理或记录工具错误 |
| `UserPromptSubmit` | 用户提示提交 | 向提示注入上下文 |
| `Stop` | 代理执行停止 | 保存会话状态 |
| `SubagentStart` / `SubagentStop` | 子代理初始化/完成 | 跟踪并行任务 |
| `PreCompact` | 对话压缩请求 | 压缩前归档完整记录 |
| `SessionStart` / `SessionEnd` | 会话开始/结束 | 初始化日志、清理资源 |
| `Notification` | 代理状态消息 | 发送到 Slack/PagerDuty |

### 匹配器

匹配器通过正则模式过滤工具名称。省略匹配器则为该事件类型的每次触发运行。

```typescript
const options = {
  hooks: {
    PreToolUse: [
      { matcher: "Write|Edit|Delete", hooks: [fileSecurityHook] },
      { matcher: "^mcp__", hooks: [mcpAuditHook] },
      { hooks: [globalLogger] }  // 无匹配器，匹配所有
    ]
  }
};
```

### 示例：阻止写入 .env 文件

```typescript
const protectEnvFiles: HookCallback = async (input, toolUseID, { signal }) => {
  const preInput = input as PreToolUseHookInput;
  const filePath = (preInput.tool_input as any)?.file_path ?? "";
  if (filePath.endsWith(".env")) {
    return {
      hookSpecificOutput: {
        hookEventName: preInput.hook_event_name,
        permissionDecision: "deny",
        permissionDecisionReason: "Cannot modify .env files",
      }
    };
  }
  return {};
};
```

### 示例：修改工具输入（重定向到沙箱）

```typescript
const redirectToSandbox: HookCallback = async (input) => {
  const preInput = input as PreToolUseHookInput;
  if (preInput.tool_name === "Write") {
    const toolInput = preInput.tool_input as Record<string, unknown>;
    return {
      hookSpecificOutput: {
        hookEventName: preInput.hook_event_name,
        permissionDecision: "allow",
        updatedInput: { ...toolInput, file_path: `/sandbox${toolInput.file_path}` }
      }
    };
  }
  return {};
};
```

### 异步输出

对于仅副作用的钩子（日志、webhook），返回异步输出让代理立即继续：

```typescript
return { async: true, asyncTimeout: 5000 };
```

---

## 子代理

子代理是你的主代理可以生成的独立代理实例，用于处理聚焦的子任务。

### 三种创建方式

1. **编程方式**（SDK 推荐）：使用 `agents` 参数
2. **文件系统**：在 `.claude/agents/` 目录中定义为 Markdown 文件
3. **内置通用型**：Claude 通过 Agent 工具调用内置 `general-purpose` 子代理

**注意：** 子代理不能再生成自己的子代理。不要在子代理的 `tools` 数组中包含 `Agent`。

### 优势

- **上下文隔离** — 每个子代理在独立的新对话中运行
- **并行化** — 多个子代理可并发运行
- **专业化指令** — 每个子代理可拥有定制的系统提示
- **工具限制** — 子代理可限制为特定工具

### 子代理继承规则

| 子代理接收 | 子代理不接收 |
|:---|:---|
| 自己的系统提示和 Agent 工具的提示 | 父级的对话历史或工具结果 |
| 项目 CLAUDE.md | 预加载的技能内容（除非列在 `skills` 中） |
| 工具定义（继承或子集） | 父级的系统提示 |

### 后台代理

设置 `background: true` 可将子代理作为非阻塞后台任务运行。

### 检测子代理调用

子代理通过 Agent 工具调用。检查 `tool_use` 块中 `name` 为 `"Agent"`（旧版为 `"Task"`）。子代理上下文中的消息包含 `parent_tool_use_id` 字段。

```typescript
for await (const message of query({
  prompt: "Use the code-reviewer agent to review this codebase",
  options: {
    allowedTools: ["Read", "Glob", "Grep", "Agent"],
    agents: {
      "code-reviewer": {
        description: "Expert code reviewer for quality and security reviews.",
        prompt: "Analyze code quality and suggest improvements.",
        tools: ["Read", "Glob", "Grep"]
      }
    }
  }
})) {
  if ("result" in message) console.log(message.result);
}
```

---

## MCP 集成

MCP 是连接 AI 代理与外部工具和数据源的开放标准。

### 添加 MCP 服务器

```typescript
for await (const message of query({
  prompt: "List files in my project",
  options: {
    mcpServers: {
      filesystem: {
        command: "npx",
        args: ["-y", "@modelcontextprotocol/server-filesystem", "/Users/me/projects"]
      }
    },
    allowedTools: ["mcp__filesystem__*"]
  }
})) {
  if (message.type === "result" && message.subtype === "success") {
    console.log(message.result);
  }
}
```

### 传输类型

| 传输 | 使用时机 | 关键配置 |
|:---|:---|:---|
| **stdio** | 文档给出要运行的命令 | `command` + `args` |
| **HTTP/SSE** | 文档给出 URL | `type: "http"` 或 `"sse"` + `url` |
| **SDK MCP 服务器** | 在代码中构建自己的工具 | 使用 `createSdkMcpServer()` |

### MCP 工具权限

MCP 工具命名模式：`mcp__<server-name>__<tool-name>`

```typescript
allowedTools: [
  "mcp__github__*",           // 所有 github 服务器工具
  "mcp__db__query",           // 仅 db 服务器的 query 工具
  "mcp__slack__send_message"  // 仅 slack 的 send_message
]
```

### 凭证传递

通过 `env` 字段传递 API 密钥：

```typescript
mcpServers: {
  github: {
    command: "npx",
    args: ["-y", "@modelcontextprotocol/server-github"],
    env: { GITHUB_TOKEN: process.env.GITHUB_TOKEN }
  }
}
```

HTTP 头传递认证：

```typescript
mcpServers: {
  "secure-api": {
    type: "sse",
    url: "https://api.example.com/mcp/sse",
    headers: { Authorization: `Bearer ${process.env.API_TOKEN}` }
  }
}
```

---

## 权限配置

### 权限评估流程

1. **钩子** — 最先运行，可拒绝或放行
2. **拒绝规则** — 检查 `disallowedTools` 和 `settings.json`
3. **权限模式** — 应用活动权限模式
4. **允许规则** — 检查 `allowedTools` 和 `settings.json`
5. **`canUseTool` 回调** — 未被以上解决的调用

### 允许和拒绝规则

- `allowedTools`：预批准列出的工具。未列出的工具仍然可用，只是走权限流程
- `disallowedTools`：裸名称（如 `"Bash"`）从上下文中完全移除；范围规则（如 `"Bash(rm *)"`）保留工具但拒绝匹配调用

### 锁定代理模式

```typescript
const options = {
  allowedTools: ["Read", "Glob", "Grep"],
  permissionMode: "dontAsk"
};
```

列出的工具被批准；其他一切被直接拒绝。

---

## 用户输入处理

Claude 在两种情况下请求用户输入：

1. **使用工具的权限** — 如删除文件或运行命令
2. **澄清问题** — 通过 `AskUserQuestion` 工具

### 响应模式

| 模式 | 说明 |
|:---|:---|
| **批准** | `{ behavior: "allow", updatedInput: input }` |
| **带修改批准** | 修改输入后批准 |
| **批准并记住** | 包含 `updatedPermissions` 持久化规则 |
| **拒绝** | `{ behavior: "deny", message: "..." }` |
| **建议替代** | 拒绝但提供指导 |
| **完全重定向** | 通过流式输入发送新指令 |

### 处理澄清问题

检测 `toolName === "AskUserQuestion"` 并路由到专用处理器。

```typescript
async function handleAskUserQuestion(input: any) {
  const answers: Record<string, string> = {};
  for (const q of input.questions) {
    // 展示问题给用户并收集答案
    answers[q.question] = "user's selection";
  }
  return { behavior: "allow", updatedInput: { questions: input.questions, answers } };
}
```

---

## 会话管理

### 选择正确的方式

| 构建内容 | 使用什么 |
|:---|:---|
| 一次性任务 | 一个 `query()` 调用 |
| 单进程多轮对话 | `continue: true` (TypeScript) |
| 进程重启后继续 | `continue: true` 恢复最近会话 |
| 恢复特定历史会话 | 捕获 `session_id` 并传给 `resume` |
| 尝试替代方案 | 使用 `forkSession: true` 分叉会话 |

### Continue vs Resume vs Fork

| 方式 | 说明 |
|:---|:---|
| **Continue** | 找到当前目录中最近的会话，无需跟踪 ID |
| **Resume** | 接受你跟踪的特定会话 ID，适用于多用户应用 |
| **Fork** | 创建新会话，以原始历史的副本开始但从此处分叉 |

### 捕获会话 ID

```typescript
let sessionId: string | undefined;

for await (const message of query({
  prompt: "Analyze the auth module",
  options: { allowedTools: ["Read", "Glob", "Grep"] }
})) {
  if (message.type === "result") {
    sessionId = message.session_id;
  }
}

// 稍后恢复
for await (const message of query({
  prompt: "Now refactor it",
  options: { resume: sessionId, allowedTools: ["Read", "Edit", "Write", "Glob", "Grep"] }
})) {
  if (message.type === "result" && message.subtype === "success") {
    console.log(message.result);
  }
}
```

### 跨主机恢复

会话文件是创建它们的机器的本地文件。要在不同主机上恢复，需移动会话文件或使用 `SessionStore` 适配器。

---

## 自定义工具

### 定义自定义工具的四个部分

| 部分 | 说明 |
|:---|:---|
| **名称** | Claude 调用工具的唯一标识符 |
| **描述** | 工具功能描述 — Claude 读取以决定何时调用 |
| **输入模式** | Claude 必须提供的参数（Zod 模式） |
| **处理器** | Claude 调用时运行的异步函数 |

### 处理器返回对象

- **`content`**（必填）：结果块数组
- **`structuredContent`**（可选）：机器可读的 JSON 对象
- **`isError`**（可选）：设为 `true` 表示工具失败

### 完整示例：天气工具

```typescript
import { tool, createSdkMcpServer } from "@anthropic-ai/claude-agent-sdk";
import { z } from "zod";

const getTemperature = tool(
  "get_temperature",
  "Get the current temperature at a location",
  {
    latitude: z.number().describe("Latitude coordinate"),
    longitude: z.number().describe("Longitude coordinate")
  },
  async (args) => {
    const response = await fetch(
      `https://api.open-meteo.com/v1/forecast?latitude=${args.latitude}&longitude=${args.longitude}&current=temperature_2m&temperature_unit=fahrenheit`
    );
    const data: any = await response.json();
    return {
      content: [{ type: "text", text: `Temperature: ${data.current.temperature_2m}°F` }]
    };
  }
);

const weatherServer = createSdkMcpServer({
  name: "weather",
  version: "1.0.0",
  tools: [getTemperature]
});

// 使用
for await (const message of query({
  prompt: "What's the temperature in San Francisco?",
  options: {
    mcpServers: { weather: weatherServer },
    allowedTools: ["mcp__weather__get_temperature"]
  }
})) {
  if (message.type === "result" && message.subtype === "success") {
    console.log(message.result);
  }
}
```

### 错误处理

| 情况 | 结果 |
|:---|:---|
| 处理器抛出未捕获异常 | 代理循环**停止** |
| 处理器捕获错误并返回 `isError: true` | 代理循环**继续**，Claude 可重试 |

### 返回图片和资源

`content` 数组接受 `text`、`image` 和 `resource` 块。图片使用 base64 编码。

---

## 流式输出

启用 `includePartialMessages: true` 可实时流式接收响应增量。

### 流式文本

```typescript
for await (const message of query({
  prompt: "Explain how databases work",
  options: { includePartialMessages: true }
})) {
  if (message.type === "stream_event") {
    const event = message.event;
    if (event.type === "content_block_delta" && event.delta.type === "text_delta") {
      process.stdout.write(event.delta.text);
    }
  }
}
```

### 流式工具调用

| 事件类型 | 用途 |
|:---|:---|
| `content_block_start` | 工具开始 — 检查 `content_block.type` 是否为 `tool_use` |
| `content_block_delta` | 输入 JSON 块通过 `delta.partial_json` 到达 |
| `content_block_stop` | 工具调用完成 |

### 消息流（启用部分消息）

```
StreamEvent (message_start)
StreamEvent (content_block_start/delta/stop) — 文本块
StreamEvent (content_block_start/delta/stop) — 工具使用块
AssistantMessage — 完整消息
... 工具执行 ...
ResultMessage — 最终结果
```

---

## 流式输入

### 流式输入模式（推荐）

通过 `AsyncGenerator` 随时间产出消息，实现自然的多轮对话。

```typescript
async function* generateMessages(): AsyncGenerator<SDKUserMessage> {
  yield {
    type: "user",
    message: { role: "user", content: "Analyze this codebase" },
    parent_tool_use_id: null
  };

  // 等待条件或用户输入
  await new Promise((resolve) => setTimeout(resolve, 2000));

  // 带图片的后续消息
  yield {
    type: "user",
    message: {
      role: "user",
      content: [
        { type: "text", text: "Review this diagram" },
        { type: "image", source: { type: "base64", media_type: "image/png", data: "..." } }
      ]
    },
    parent_tool_use_id: null
  };
}

for await (const message of query({
  prompt: generateMessages(),
  options: { maxTurns: 10, allowedTools: ["Read", "Grep"] }
})) {
  if (message.type === "result" && message.subtype === "success") {
    console.log(message.result);
  }
}
```

### 流式 vs 单消息

| 方面 | 流式输入 | 单消息输入 |
|:---|:---|:---|
| 多轮对话 | 原生支持 | 需 `continue` 标志 |
| 图片附件 | 支持 | 不支持 |
| 中断 | 支持 | 不支持 |
| 钩子 | 支持 | 不支持 |
| 最佳用途 | 交互式、长期运行的代理 | 无状态/一次性用例 |

---

## 结构化输出

定义你希望代理返回的确切数据形状。代理使用任何所需工具完成任务，你仍获得匹配模式的验证 JSON。

### 快速开始

```typescript
import { z } from "zod";

const FeaturePlan = z.object({
  feature_name: z.string(),
  steps: z.array(z.object({
    step_number: z.number(),
    description: z.string(),
    estimated_complexity: z.enum(["low", "medium", "high"])
  }))
});

const schema = z.toJSONSchema(FeaturePlan);

for await (const message of query({
  prompt: "Plan how to add dark mode support...",
  options: {
    outputFormat: { type: "json_schema", schema }
  }
})) {
  if (message.type === "result" && message.subtype === "success" && message.structured_output) {
    const parsed = FeaturePlan.safeParse(message.structured_output);
    if (parsed.success) {
      const plan = parsed.data; // 完全类型化
    }
  }
}
```

### 错误处理

| 结果子类型 | 含义 |
|:---|:---|
| `success` | 输出成功生成并验证 |
| `error_max_structured_output_retries` | 多次尝试后仍无法生成有效输出 |

### 避免错误的提示

1. **保持模式聚焦** — 深度嵌套的模式更难满足
2. **模式与任务匹配** — 信息可能不全的字段设为可选
3. **使用清晰的提示** — 模糊的提示使输出更难预测

---

## 成本跟踪

**关键提示：** `total_cost_usd` 和 `costUSD` 字段是**客户端估算**，不是权威计费数据。

### 获取查询总成本

```typescript
for await (const message of query({ prompt: "Summarize this project" })) {
  if (message.type === "result") {
    console.log(`Total cost: $${message.total_cost_usd}`);
  }
}
```

### 每步使用跟踪

当 Claude 在一轮中使用多个工具（并行工具调用）时，**同轮次的所有消息共享相同 ID 和使用数据**。必须按 ID 去重。

### 每模型使用明细

结果消息包含 `modelUsage` — 模型名称到令牌计数和成本的映射。

### 跨多次调用累积成本

SDK 不提供会话级总计。每次 `query()` 调用返回自己的 `total_cost_usd`。你必须自行累积。

### 提示缓存

SDK 自动使用提示缓存。通过设置 `ENABLE_PROMPT_CACHING_1H` 环境变量可将缓存 TTL 从 5 分钟延长到 1 小时（写入成本更高）。

---

## 托管部署

### 系统要求

| 类别 | 要求 |
|:---|:---|
| 运行时 | Python 3.10+ 或 Node.js 18+ |
| RAM | 推荐 1 GiB |
| 磁盘 | 推荐 5 GiB |
| CPU | 推荐 1 CPU |
| 网络 | 到 `api.anthropic.com` 的出站 HTTPS |

### 沙箱提供商

| 提供商 | 链接 |
|:---|:---|
| Modal Sandbox | [文档](https://modal.com/docs/guide/sandbox) |
| Cloudflare Sandboxes | [GitHub](https://github.com/cloudflare/sandbox-sdk) |
| Daytona | [daytona.io](https://www.daytona.io/) |
| E2B | [e2b.dev](https://e2b.dev/) |
| Fly Machines | [文档](https://fly.io/docs/machines/) |
| Vercel Sandbox | [文档](https://vercel.com/docs/functions/sandbox) |

### 生产部署模式

| 模式 | 说明 | 最佳用例 |
|:---|:---|:---|
| **临时会话** | 每个用户任务创建新容器，完成后销毁 | Bug 修复、发票处理 |
| **长期运行会话** | 维持持久容器实例 | 邮件代理、站点构建器 |
| **混合会话** | 临时容器，从历史/状态注入 | 深度研究、客户支持 |
| **单容器** | 一个全局容器中运行多个 SDK 进程 | 代理模拟 |

---

## 安全部署

### 威胁模型

核心威胁：代理可能因**提示注入**（嵌入在内容中的指令）或**模型错误**而采取意外行动。

### 安全原则

1. **安全边界** — 将敏感资源放在代理边界之外
2. **最小权限** — 仅限制代理完成特定任务所需的能力
3. **纵深防御** — 层叠多个控制

### 隔离技术

| 技术 | 隔离强度 | 性能开销 | 复杂度 |
|:---|:---|:---|:---|
| **Sandbox Runtime** | 良好 | 极低 | 低 |
| **容器 (Docker)** | 取决于配置 | 低 | 中 |
| **gVisor** | 优秀 | 中/高 | 中 |
| **VM (Firecracker, QEMU)** | 优秀 | 高 | 中/高 |

### Docker 加固配置

```bash
docker run \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=100m \
  --network none \
  --memory 2g \
  --cpus 2 \
  --pids-limit 100 \
  --user 1000:1000 \
  -v /path/to/code:/workspace:ro \
  agent-image
```

### 凭证管理（代理模式）

推荐：在代理安全边界之外运行代理，向出站请求注入凭证。代理发送不含凭证的请求；代理添加凭证并转发。

**好处：**
1. 代理永远看不到实际凭证
2. 代理可以强制执行允许列表
3. 代理可以记录所有请求
4. 凭证存储在一个安全位置

### 文件系统配置

避免挂载敏感目录如 `~/.ssh`、`~/.aws`、`~/.config`。使用只读挂载 `:ro` 或临时文件系统。

---

## 另见

- [SDK 概览](https://code.claude.com/docs/en/agent-sdk/overview)
- [TypeScript SDK 参考](https://code.claude.com/docs/en/agent-sdk/typescript)
- [Python SDK 参考](https://code.claude.com/docs/en/agent-sdk/python)
- [CLI 参考](https://code.claude.com/docs/en/cli-reference)
- [常见工作流](https://code.claude.com/docs/en/common-workflows)
- [示例代理](https://github.com/anthropics/claude-agent-sdk-demos)
- [TypeScript SDK 变更日志](https://github.com/anthropics/claude-agent-sdk-typescript/blob/main/CHANGELOG.md)
- [Python SDK 变更日志](https://github.com/anthropics/claude-agent-sdk-python/blob/main/CHANGELOG.md)
