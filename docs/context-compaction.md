# 上下文压缩处理

本文档描述 pets-agent 的上下文管理机制，包括已实现的 SDK 自动压缩集成、事件观测、历史同步、智能截断和 token 用量报告。

## 架构概览

上下文管理涉及两层历史系统的协同：

| 层 | 存储位置 | 用途 | 管理 |
|---|---|---|---|
| 应用层历史 | `ConversationHistoryStore` (`.harness/state/history.json`) | 意图检测（最近 4 条）、反馈上下文 | `maxMessages` 可配，压缩时通过 `compact()` 同步 |
| SDK 层历史 | Claude Agent SDK 内部 session | LLM 对话上下文 | SDK 自动压缩，通过 PostCompact hook 同步到应用层 |

**数据流**：

```text
用户消息
  → Orchestrator.handle()
    → historyStore.get() → 取最近 4 条 → 意图检测
    → runtime.run(AgentRequest)
      → AgentRequest.onCompact → PostCompact hook 回调
      → buildWorkspacePrompt(request, workspaceMaxChars)
      → Claude SDK query({ resume, settings: { autoCompactEnabled, autoCompactWindow }, hooks: { PostCompact } })
        → SDK 自行管理上下文，接近阈值时自动压缩
        → PostCompact hook → onCompact(summary) → historyStore.compact()
        → SDKCompactBoundaryMessage → forwardSystemMessageEvents() → compact_complete 流事件
      → 提取 result.usage → ContextUsageReport
    → historyStore.append([user, assistant])
    → logEvent("context.usage", contextUsage)
```

---

## 运行时配置

通过 `config/runtime.json` 的 `context` 段配置（`src/config/runtimeConfig.ts`）：

```json
{
  "context": {
    "autoCompactEnabled": true,
    "autoCompactWindow": 150000,
    "workspaceMaxChars": 8000,
    "historyMaxMessages": 20
  }
}
```

| 配置项 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `autoCompactEnabled` | `boolean` | `true` | 是否启用 SDK 自动压缩 |
| `autoCompactWindow` | `number` | `150000` | 触发压缩的 token 阈值（contextWindow 200k - 50k 余量） |
| `workspaceMaxChars` | `number` | `8000` | CLAUDE.md 最大字符数（按 markdown 章节边界截断） |
| `historyMaxMessages` | `number` | `20` | 应用层对话历史最大条数 |

所有字段均可省略，使用默认值。`ContextConfig` 类型定义在 `src/config/runtimeConfig.ts`。

---

## SDK 自动压缩

### 启用方式

`ClaudeSdkAgentRuntime` 在构建 `query()` 选项时传入（`src/agent/claudeSdkAgentRuntime.ts`）：

```typescript
if (this.contextConfig.autoCompactEnabled) {
  queryOptions["settings"] = {
    autoCompactEnabled: true,
    autoCompactWindow: this.contextConfig.autoCompactWindow,
  };
}
```

当 `autoCompactEnabled: false` 时不传 `settings`，SDK 不会自动压缩。

### 压缩触发流程

SDK 在对话 token 数接近 `autoCompactWindow` 时自动触发：

1. SDK 调用 LLM 对已有对话生成摘要（`compact_summary`）
2. 用摘要替换旧的对话消息，保留最近几轮
3. 触发 `PostCompact` hook → 应用层 `onCompact` 回调
4. 在消息流中发出 `SDKCompactBoundaryMessage` 和 `SDKStatusMessage`
5. 后续请求基于压缩后的上下文继续

---

## 压缩事件观测

### 流事件

`AgentStreamEvent` 扩展了两种压缩事件（`src/core/contracts.ts`）：

```typescript
| { type: "compact_start" }
| { type: "compact_complete"; preTokens: number; postTokens?: number; durationMs?: number }
```

事件来源（`src/agent/claudeSdkMessageMapper.ts`）：

| SDK 消息 | 映射 |
|---|---|
| `SDKStatusMessage`（`status: 'compacting'`） | → `compact_start` |
| `SDKCompactBoundaryMessage`（`subtype: 'compact_boundary'`） | → `compact_complete`（含 `preTokens`/`postTokens`/`durationMs`） |

前端通过 SSE 流可收到这些事件，展示"正在压缩上下文..."状态。

### 日志

压缩完成时写入 `llm-raw.jsonl`（`src/agent/claudeSdkAgentRuntime.ts`）：

```json
{
  "type": "llm.compact",
  "runtime": "claude-sdk-reviewer",
  "userId": "user-1",
  "workspacePath": "D:/kb",
  "sessionId": "session-abc",
  "trigger": "auto",
  "preTokens": 180000,
  "postTokens": 45000,
  "durationMs": 1200
}
```

---

## 应用层历史同步

### compact() 方法

`ConversationHistoryStore` 契约新增 `compact()` 方法（`src/core/contracts.ts`）：

```typescript
compact(key: ConversationSessionKey, summary: string): Promise<void>;
```

实现逻辑（`src/persistence/fileConversationHistoryStore.ts`）：

1. 在历史头部插入压缩摘要：`{ role: "assistant", content: "[Previous conversation summary]\n" + summary }`
2. 保留最近 2 条消息（最近一问一答）
3. 总条数不超过 `maxMessages`

### PostCompact hook 注册

`AgentRequest` 新增 `onCompact` 回调（`src/core/contracts.ts`）：

```typescript
readonly onCompact?: (summary: string) => Promise<void>;
```

`ClaudeSdkAgentRuntime` 在 `request.onCompact` 存在时注册 SDK hook（`src/agent/claudeSdkAgentRuntime.ts`）：

```typescript
if (request.onCompact !== undefined) {
  queryOptions["hooks"] = {
    PostCompact: [{
      hooks: [async (input: Record<string, unknown>) => {
        const summary = input["compact_summary"] as string;
        await request.onCompact?.(summary);
      }],
    }],
  };
}
```

### Orchestrator 绑定

`AgentOrchestrator.executeRuntime()` 中构建回调（`src/core/orchestrator.ts`）：

```typescript
const request: AgentRequest = {
  // ...existing fields
  onCompact: async (summary) => {
    await this.dependencies.historyStore?.compact(sessionKey, summary);
    await this.logEvent("context.compacted", message, {
      workspacePath,
      summaryLength: summary.length,
    });
  },
};
```

压缩后 `system.jsonl` 记录 `context.compacted` 事件。管理员审批反馈时能看到压缩摘要而非丢失上下文。

---

## 工作区上下文智能截断

CLAUDE.md 内容通过按 markdown 章节边界截断，替代硬字符截断（`src/agent/workspacePromptBuilder.ts`）。

### 截断算法

```typescript
export function truncateToBudget(content: string, maxChars: number): string {
  if (content.length <= maxChars) return content;
  const sections = splitAtHeadings(content);
  let result = "";
  for (const section of sections) {
    if (result.length + section.length > maxChars) break;
    result += section;
  }
  return result.length > 0 ? result : content.slice(0, maxChars);
}
```

- `splitAtHeadings()` 按 `#`/`##`/`###` 行分段，h4+ 不分段
- 按顺序累加完整 section，超出预算时停止
- 首个 section 超长时回退到硬截断
- 上限通过 `context.workspaceMaxChars` 配置，默认 8,000 字符

---

## Token 用量报告

### ContextUsageReport

每次 LLM 调用后提取 token 用量（`src/core/contracts.ts`）：

```typescript
export type ContextUsageReport = {
  readonly inputTokens: number;
  readonly outputTokens: number;
  readonly cacheReadTokens?: number;
  readonly cacheCreationTokens?: number;
  readonly contextWindow: number;       // 等于 autoCompactWindow
  readonly usagePercent: number;        // inputTokens / contextWindow * 100
};
```

### 提取逻辑

从 SDK `result` 消息的 `usage` 字段提取（`src/agent/claudeSdkAgentRuntime.ts`）：

```typescript
function extractContextUsage(usage: unknown, contextWindow: number): ContextUsageReport | undefined {
  if (usage === null || usage === undefined || typeof usage !== "object") return undefined;
  const u = usage as Record<string, unknown>;
  const inputTokens = u["input_tokens"];
  const outputTokens = u["output_tokens"];
  if (typeof inputTokens !== "number" || typeof outputTokens !== "number") return undefined;
  return {
    inputTokens, outputTokens,
    ...(typeof u["cache_read_input_tokens"] === "number" ? { cacheReadTokens: u["cache_read_input_tokens"] } : {}),
    ...(typeof u["cache_creation_input_tokens"] === "number" ? { cacheCreationTokens: u["cache_creation_input_tokens"] } : {}),
    contextWindow,
    usagePercent: contextWindow > 0 ? Math.round((inputTokens / contextWindow) * 100) : 0,
  };
}
```

当 SDK result 不含 `usage` 字段时，`contextUsage` 为 `undefined`。

### 日志

`AgentResponse.contextUsage` 通过 Orchestrator 写入 `system.jsonl`（`src/core/orchestrator.ts`）：

```json
{
  "type": "context.usage",
  "workspacePath": "D:/kb",
  "inputTokens": 120000,
  "outputTokens": 500,
  "cacheReadTokens": 80000,
  "cacheCreationTokens": 30000,
  "contextWindow": 150000,
  "usagePercent": 80
}
```

---

## 完整压缩场景数据流

```text
SDK 检测上下文接近 autoCompactWindow 阈值
  → SDK 触发 auto-compaction
    → LLM 生成 compact_summary
    → SDK 替换旧消息为摘要
    → PostCompact hook → request.onCompact(summary)
      → historyStore.compact(sessionKey, summary)
        → 保留最近 2 条消息 + 头部插入 "[Previous conversation summary]\n..."
      → logEvent("context.compacted", { workspacePath, summaryLength })
    → 消息流发出 SDKStatusMessage(status: 'compacting')
      → forwardSystemMessageEvents() → request.stream({ type: "compact_start" })
      → 前端展示"正在压缩上下文..."
    → 消息流发出 SDKCompactBoundaryMessage
      → forwardSystemMessageEvents() → request.stream({ type: "compact_complete", preTokens, postTokens, durationMs })
      → rawLogger.write({ type: "llm.compact", ... })
      → 前端展示压缩完成状态
    → 后续请求基于压缩后上下文继续
  → runtime.run() 返回 AgentResponse { text, sessionId, contextUsage }
    → logEvent("context.usage", { inputTokens, outputTokens, ..., usagePercent })
    → historyStore.append([user_msg, assistant_msg])
```

---

## 文件索引

| 文件 | 职责 |
|---|---|
| `src/config/runtimeConfig.ts` | `ContextConfig` 类型和 Zod schema，注入 `RuntimeConfig` |
| `src/agent/claudeSdkAgentRuntime.ts` | 传 settings/hooks 给 SDK query()，处理 system 消息，提取 usage，PostCompact hook |
| `src/agent/claudeSdkMessageMapper.ts` | `isSystemMessage`、`forwardSystemMessageEvents`（compact_boundary/status → 流事件） |
| `src/agent/workspacePromptBuilder.ts` | `splitAtHeadings`、`truncateToBudget` 智能截断 |
| `src/core/contracts.ts` | `AgentStreamEvent`（compact_start/complete）、`AgentRequest.onCompact`、`ConversationHistoryStore.compact`、`ContextUsageReport`、`AgentResponse.contextUsage` |
| `src/core/orchestrator.ts` | `onCompact` 回调绑定、`context.usage`/`context.compacted` 事件日志 |
| `src/core/streamProgressMapper.ts` | compact_start/complete → progress event stage 映射 |
| `src/persistence/fileConversationHistoryStore.ts` | `compact()` 实现 |
| `src/agent/createAgentRuntimes.ts` | `contextConfig` 透传 |
| `src/index.ts` | `config.context` 传入 setupAgentRuntimes 和 FileConversationHistoryStore |
