# TypeScript SDK 参考

> **版本要求**：本文档针对 CodeBuddy Agent SDK v0.1.0 及以上版本。

本文档提供 TypeScript SDK 的完整 API 参考。有关快速入门和使用示例，请参阅 [SDK 概览](codebuddy-sdk-quickstart.md)。

## Requirements

| 依赖 | 版本要求 |
|------|----------|
| Node.js | \>= 18.0.0 |
| TypeScript | \>= 5.0.0（推荐） |

**运行时支持**：Node.js（推荐）、Bun、Deno

## Installation

```bash
npm install @tencent-ai/agent-sdk
# 或
yarn add @tencent-ai/agent-sdk
pnpm add @tencent-ai/agent-sdk
```

### 环境变量

| 变量名 | 说明 | 必需 |
|--------|------|------|
| `CODEBUDDY_CODE_PATH` | CodeBuddy CLI 可执行文件路径 | 可选 |

### 认证配置

SDK 支持使用已有登录凭据、API Key 或 OAuth Client Credentials 认证，详见 [SDK 概览 - 认证配置](codebuddy-sdk-quickstart.md#认证配置)。

## Functions

### query()

主要 API 入口，创建一个查询并返回消息流。

```typescript
function query(params: {
  prompt: string | AsyncIterable<UserMessage>;
  options?: Options;
}): Query;
```

**参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `prompt` | `string \| AsyncIterable<UserMessage>` | 查询提示词或用户消息流 |
| `options` | `Options` | 配置选项（可选） |

**返回值**：`Query` - 扩展了 `AsyncGenerator<Message, void>` 的接口

### Query 接口

```typescript
interface Query extends AsyncGenerator<Message, void> {
  // 中断当前执行
  interrupt(): Promise<void>;

  // 动态修改权限模式
  setPermissionMode(mode: PermissionMode): Promise<void>;

  // 动态修改模型
  setModel(model?: string): Promise<void>;

  // 设置最大思考 token 数
  setMaxThinkingTokens(tokens: number | null): Promise<void>;

  // 获取可用权限模式列表
  getAvailableModes(): Promise<ModeInfo[]>;

  // 获取可用模型列表
  getAvailableModels(): Promise<ModelInfo[]>;

  // 获取支持的斜杠命令
  supportedCommands(): Promise<SlashCommand[]>;

  // 获取支持的模型列表
  supportedModels(): Promise<ModelInfo[]>;

  // 获取 MCP 服务器状态
  mcpServerStatus(): Promise<McpServerStatus[]>;

  // 获取账户信息
  accountInfo(): Promise<AccountInfo>;

  // 流式输入用户消息
  streamInput(stream: AsyncIterable<UserMessage>): Promise<void>;
}
```

### Constants

```typescript
// 所有支持的 Hook 事件
const HOOK_EVENTS: readonly [
  'PreToolUse', 'PostToolUse', 'PostToolUseFailure',
  'Notification', 'UserPromptSubmit', 'SessionStart', 'SessionEnd',
  'Stop', 'SubagentStart', 'SubagentStop', 'PreCompact',
  'PermissionRequest', 'WorktreeCreate', 'WorktreeRemove'
];

// 所有退出原因
const EXIT_REASONS: readonly [
  'user_cancelled', 'tool_error', 'max_turns',
  'max_budget_usd', 'completed', 'interrupted', 'hook_blocked'
];
```

### Errors

```typescript
class AbortError extends Error {
  // 当操作被中止时抛出
}
```

## Unstable V2 API

> **警告**：以下 API 处于实验阶段，接口可能在未来版本中变更。

### unstable_v2_createSession()

创建新的交互式会话。

```typescript
function unstable_v2_createSession(options: SessionOptions): Session;
```

### unstable_v2_resumeSession()

恢复现有会话。

```typescript
function unstable_v2_resumeSession(
  sessionId: string,
  options: SessionOptions
): Session;
```

### unstable_v2_prompt()

单次查询便捷函数。

```typescript
function unstable_v2_prompt(
  message: string,
  options: SessionOptions
): Promise<Message[]>;
```

### unstable_v2_authenticate()

发起交互式登录流程，支持多环境认证。

```typescript
function unstable_v2_authenticate(options: AuthenticateOptions): Promise<AuthenticateResponse>;
```

**参数**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `onAuthUrl` | `(authState: AuthState) => Promise<void>` | 认证 URL 回调 |
| `environment` | `'external' \| 'internal' \| 'ioa' \| 'cloudhosted'` | 预定义环境 |
| `endpoint` | `string` | 自定义 endpoint URL（与 environment 二选一） |
| `methodId` | `string` | 认证方法 ID，默认 'external' |
| `timeout` | `number` | 超时时间（毫秒），默认 300000 |
| `pathToCodebuddyCode` | `string` | CLI 可执行文件路径（可选） |
| `env` | `Record<string, string>` | 环境变量（可选） |

**返回值**：`Promise<AuthenticateResponse>`

- `userinfo` - 用户信息对象，包含 userId、userName、userNickname、token 等字段

**示例**：

```typescript
import { unstable_v2_authenticate } from '@tencent-ai/agent-sdk';
import open from 'open';

// 海外版登录
const result = await unstable_v2_authenticate({
  environment: 'external',
  onAuthUrl: async (authState) => {
    console.log('请登录:', authState.authUrl);
    await open(authState.authUrl);
  }
});

console.log('登录成功:', result.userinfo.userName);
```

### unstable_v2_logout()

登出并清除缓存的认证 token。

```typescript
function unstable_v2_logout(options?: LogoutOptions): Promise<void>;
```

### Session 接口

```typescript
interface Session {
  readonly sessionId: string;
  send(message: string | UserMessage): Promise<void>;
  stream(): AsyncGenerator<Message, void>;
  close(): void;
  [Symbol.asyncDispose](): Promise<void>;
}
```

### SessionOptions

```typescript
type SessionOptions = {
  model: string;
  pathToCodebuddyCode?: string;
  executable?: 'node' | 'bun';
  executableArgs?: string[];
  env?: Record<string, string | undefined>;
  canUseTool?: CanUseTool;
};
```

## Types

### Options

完整配置选项：

| 字段 | 类型 | 说明 |
|------|------|------|
| `abortController` | `AbortController` | 用于取消请求 |
| `executable` | `'bun' \| 'deno' \| 'node'` | 运行时 |
| `executableArgs` | `string[]` | 运行时参数 |
| `pathToCodebuddyCode` | `string` | CLI 路径 |
| `cwd` | `string` | 工作目录 |
| `additionalDirectories` | `string[]` | 额外的目录 |
| `env` | `Record<string, string \| undefined>` | 环境变量 |
| `model` | `string` | 指定模型 |
| `fallbackModel` | `string` | 备用模型 |
| `thinking` | `ThinkingConfig` | 思考模式配置 |
| `effort` | `'low' \| 'medium' \| 'high' \| 'xhigh'` | 模型推理努力程度 |
| `allowedTools` | `string[]` | 允许的工具白名单 |
| `disallowedTools` | `string[]` | 禁止的工具黑名单 |
| `canUseTool` | `CanUseTool` | 权限回调函数 |
| `permissionMode` | `PermissionMode` | 权限模式 |
| `allowDangerouslySkipPermissions` | `boolean` | 允许跳过权限 |
| `permissionPromptToolName` | `string` | 权限提示工具名 |
| `continue` | `boolean` | 继续最近的会话 |
| `resume` | `string` | 要恢复的会话 ID |
| `resumeSessionAt` | `string` | 恢复到特定消息位置 |
| `persistSession` | `boolean` | 持久化会话 |
| `forkSession` | `boolean` | 分叉会话 |
| `agents` | `Record<string, AgentDefinition>` | 自定义 Agent |
| `hooks` | `Partial<Record<HookEvent, HookCallbackMatcher[]>>` | Hook 配置 |
| `outputFormat` | `OutputFormat` | 输出格式 |
| `systemPrompt` | `string \| { append: string }` | 系统提示词 |
| `includePartialMessages` | `boolean` | 包含部分消息 |
| `maxTurns` | `number` | 最大对话轮数 |
| `mcpServers` | `Record<string, McpServerConfig>` | MCP 服务器配置 |
| `strictMcpConfig` | `boolean` | 严格 MCP 配置 |
| `sandbox` | `SandboxSettings` | 沙箱设置 |
| `settingSources` | `SettingSource[]` | 配置源 |

### SettingSource

```typescript
type SettingSource = 'user' | 'project' | 'local';
```

| 值 | 说明 | 位置 |
|----|------|------|
| `'user'` | 全局用户设置 | `~/.codebuddy/settings.json` |
| `'project'` | 项目共享设置 | `.codebuddy/settings.json` |
| `'local'` | 项目本地设置 | `.codebuddy/settings.local.json` |

**默认行为**：当 `settingSources` 未指定时，SDK **不加载任何文件系统配置**。

### PermissionMode

```typescript
type PermissionMode =
  | 'default'           // 默认模式，所有操作需确认
  | 'acceptEdits'       // 自动批准文件编辑
  | 'bypassPermissions' // 跳过所有权限检查
  | 'plan'              // 规划模式，仅允许读取
```

### PermissionResult

```typescript
type PermissionResult =
  | {
      behavior: 'allow';
      updatedInput: Record<string, unknown>;
      updatedPermissions?: PermissionUpdate[];
      toolUseID?: string;
    }
  | {
      behavior: 'deny';
      message: string;
      interrupt?: boolean;
      toolUseID?: string;
    };
```

### CanUseTool

```typescript
type CanUseTool = (
  toolName: string,
  input: Record<string, unknown>,
  options: CanUseToolOptions
) => Promise<PermissionResult>;

type CanUseToolOptions = {
  signal: AbortSignal;
  suggestions?: PermissionUpdate[];
  blockedPath?: string;
  decisionReason?: string;
  toolUseID: string;
  agentID?: string;
};
```

### AgentDefinition

```typescript
type AgentDefinition = {
  description: string;          // Agent 描述
  prompt: string;               // 系统提示词
  tools?: string[];             // 允许的工具
  disallowedTools?: string[];   // 禁止的工具
  model?: string;               // 使用的模型
};
```

### McpServerConfig

```typescript
// Stdio 类型
type McpStdioServerConfig = {
  type?: 'stdio';
  command: string;
  args?: string[];
  env?: Record<string, string>;
};

// SSE 类型
type McpSSEServerConfig = {
  type: 'sse';
  url: string;
  headers?: Record<string, string>;
};

// HTTP 类型
type McpHttpServerConfig = {
  type: 'http';
  url: string;
  headers?: Record<string, string>;
};

type McpServerConfig =
  | McpStdioServerConfig
  | McpSSEServerConfig
  | McpHttpServerConfig;
```

### HookEvent

```typescript
type HookEvent =
  | 'PreToolUse'
  | 'PostToolUse'
  | 'PostToolUseFailure'
  | 'Notification'
  | 'UserPromptSubmit'
  | 'SessionStart'
  | 'SessionEnd'
  | 'Stop'
  | 'SubagentStart'
  | 'SubagentStop'
  | 'PreCompact'
  | 'PermissionRequest'
  | 'WorktreeCreate'
  | 'WorktreeRemove';
```

### HookCallback

```typescript
type HookCallback = (
  input: HookInput,
  toolUseID: string | undefined,
  options: { signal: AbortSignal }
) => Promise<HookJSONOutput>;

interface HookCallbackMatcher {
  matcher?: string;
  hooks: HookCallback[];
  timeout?: number;
}
```

### HookJSONOutput

```typescript
type SyncHookJSONOutput = {
  continue?: boolean;
  suppressOutput?: boolean;
  stopReason?: string;
  decision?: 'approve' | 'block';
  systemMessage?: string;
  reason?: string;
  hookSpecificOutput?: Record<string, unknown>;
};

type AsyncHookJSONOutput = {
  async: true;
  asyncTimeout?: number;
};

type HookJSONOutput = SyncHookJSONOutput | AsyncHookJSONOutput;
```

## Message Types

### Message

所有消息类型的联合：

```typescript
type Message =
  | SystemMessage
  | UserMessage
  | AssistantMessage
  | PartialAssistantMessage
  | ResultMessage
  | CompactBoundaryMessage
  | StatusMessage
  | ToolProgressMessage;
```

### SystemMessage

```typescript
type SystemMessage = {
  type: 'system';
  subtype: 'init';
  uuid: string;
  session_id: string;
  apiKeySource?: string;
  cwd?: string;
  tools: string[];
  mcp_servers?: Array<{ name: string; status: string }>;
  model: string;
  permissionMode: PermissionMode;
  slash_commands?: string[];
  codebuddy_code_version?: string;
  skills?: string[];
  plugins?: Array<{ name: string; path: string }>;
};
```

### UserMessage

```typescript
type UserMessage = {
  type: 'user';
  uuid?: string;
  session_id: string;
  message: {
    role: 'user';
    content: string | ContentBlock[];
  };
  parent_tool_use_id: string | null;
  isSynthetic?: boolean;
  tool_use_result?: unknown;
};
```

### AssistantMessage

```typescript
type AssistantMessage = {
  type: 'assistant';
  uuid: string;
  session_id: string;
  message: {
    id: string;
    type: 'message';
    role: 'assistant';
    model: string;
    content: ContentBlock[];
    stop_reason: StopReason | null;
    stop_sequence: string | null;
    usage: Usage;
  };
  parent_tool_use_id: string | null;
  error?: string;
};
```

### ResultMessage

```typescript
type ResultMessage =
  | {
      type: 'result';
      subtype: 'success';
      uuid: string;
      session_id: string;
      duration_ms: number;
      duration_api_ms: number;
      is_error: boolean;
      num_turns: number;
      result: string;
      total_cost_usd: number;
      usage: Usage;
      permission_denials: PermissionDenial[];
      structured_output?: unknown;
    }
  | {
      type: 'result';
      subtype: 'error_during_execution' | 'error_max_turns' | 'error_max_budget_usd';
      uuid: string;
      session_id: string;
      duration_ms: number;
      duration_api_ms: number;
      is_error: boolean;
      num_turns: number;
      total_cost_usd: number;
      usage: Usage;
      permission_denials: PermissionDenial[];
      errors?: string[];
    };
```

### ContentBlock

```typescript
interface TextContentBlock {
  type: 'text';
  text: string;
}

interface ToolUseContentBlock {
  type: 'tool_use';
  id: string;
  name: string;
  input: Record<string, unknown>;
}

interface ToolResultContentBlock {
  type: 'tool_result';
  tool_use_id: string;
  content?: string | ContentBlock[];
  is_error?: boolean;
}

type ContentBlock =
  | TextContentBlock
  | ToolUseContentBlock
  | ToolResultContentBlock;
```

### Usage

```typescript
interface Usage {
  input_tokens: number;
  output_tokens: number;
  cache_read_input_tokens?: number | null;
  cache_creation_input_tokens?: number | null;
}
```

## Input Types

### AskUserQuestionInput

```typescript
interface AskUserQuestionInput {
  questions: AskUserQuestionQuestion[];
  answers?: Record<string, string>;
}
```

### AskUserQuestionQuestion

```typescript
interface AskUserQuestionQuestion {
  question: string;
  header: string;
  options: AskUserQuestionOption[];
  multiSelect: boolean;
}
```

### AskUserQuestionOption

```typescript
interface AskUserQuestionOption {
  label: string;
  description: string;
}
```

## 相关文档

- [SDK 概览](codebuddy-sdk-quickstart.md) - 快速入门和使用示例
- [Python SDK 参考](codebuddy-sdk-python.md) - Python 版本 API
- [Hook 参考指南](codebuddy-sdk-hooks.md) - 详细的 Hook 配置说明
- [MCP 集成](codebuddy-sdk-mcp.md) - MCP 服务器配置指南
