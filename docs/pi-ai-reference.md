# Pi AI SDK 中文文档

> **SDK 版本**：v0.78.0（2026-06-02 更新确认）
> 完整文档来源：https://github.com/earendil-works/pi（packages/ai）

---

## 目录

- [概览](#概览)
- [快速入门](#快速入门)
- [核心概念](#核心概念)
- [TypeScript API 参考](#typescript-api-参考)
  - [安装](#安装)
  - [函数](#函数)
  - [类型](#类型)
  - [事件类型](#事件类型)
  - [内容块类型](#内容块类型)
- [指南](#指南)
  - [模型与提供商配置](#模型与提供商配置)
  - [自定义模型](#自定义模型)
  - [工具（函数调用）](#工具函数调用)
  - [流式输出](#流式输出)
  - [思考/推理](#思考推理)
  - [图片输入](#图片输入)
  - [图片生成](#图片生成)
  - [跨提供商切换](#跨提供商切换)
  - [上下文序列化](#上下文序列化)
  - [错误处理与中止](#错误处理与中止)
  - [Faux Provider（测试）](#faux-provider测试)

---

# 概览

`@earendil-works/pi-ai` 是一个统一的 LLM API，提供跨多个 AI 提供商（OpenAI、Anthropic、Google 等）的一致接口。它自动处理模型发现、提供商配置、令牌/成本跟踪、上下文持久化和跨提供商切换。**仅包含支持工具调用的模型**（对代理工作流至关重要）。

### TypeScript 示例

```typescript
import { getModel, complete } from "@earendil-works/pi-ai";

const model = getModel("anthropic", "claude-sonnet-4-20250514");

const response = await complete(model, {
  systemPrompt: "You are a helpful assistant.",
  messages: [{ role: "user", content: "Hello!" }],
});

for (const block of response.content) {
  if (block.type === "text") {
    console.log(block.text);
  }
}
```

## 核心能力

| 能力 | 说明 |
|:---|:---|
| **统一 API** | 同一套 API 适配 OpenAI、Anthropic、Google、DeepSeek、xAI、Groq 等 25+ 提供商 |
| **自动模型发现** | 内置模型目录，IDE 自动补全，一行代码获取模型配置 |
| **自定义模型** | 支持 Ollama、vLLM、LM Studio 等任意 OpenAI 兼容 API |
| **工具调用** | TypeBox 模式定义工具参数，跨提供商统一的函数调用接口 |
| **流式/完整** | `stream()` 和 `complete()` 两种响应模式 |
| **思考/推理** | 支持 Claude、GPT、Gemini 的思考模式，跨提供商自动转换 |
| **成本跟踪** | 内置令牌和成本计算，每次响应附带用量数据 |
| **上下文持久化** | Context 对象可直接序列化为 JSON，支持会话恢复 |
| **跨提供商切换** | 同一对话中无缝切换模型，思考块自动转换 |

## 与其他 LLM 库的比较

| | pi-ai | 官方 Client SDK | claude-agent-sdk |
|:---|:---|:---|:---|
| **定位** | 统一 LLM API | 单提供商 API | 代理运行时 |
| **多提供商** | 25+ | 1 | 仅 Claude |
| **工具调用** | 内置 | 需自行处理 | 内置代理循环 |
| **模型发现** | 自动 | 手动 | 无 |
| **成本跟踪** | 内置 | 需自行计算 | 内置 |
| **最佳用途** | 跨提供商 LLM 调用 | 深度使用单一提供商 | 自主代理执行 |

---

# 快速入门

## 前提条件

- **Node.js 18+**
- 至少一个提供商的 API 密钥

## 步骤

### 1. 安装

```bash
npm install @earendil-works/pi-ai
```

包同时导出 TypeBox 的 `Type`、`Static`、`TSchema`。

### 2. 设置 API 密钥

```bash
export ANTHROPIC_API_KEY=your-api-key
# 或
export OPENAI_API_KEY=your-api-key
# 或
export GEMINI_API_KEY=your-api-key
```

### 3. 发送第一个请求

```typescript
import { getModel, complete } from "@earendil-works/pi-ai";

const model = getModel("anthropic", "claude-sonnet-4-20250514");

const response = await complete(model, {
  messages: [{ role: "user", content: "What is TypeScript?" }],
});

for (const block of response.content) {
  if (block.type === "text") {
    console.log(block.text);
  }
}
```

## 代码三要素

1. **`getModel()`**：获取模型配置，指定提供商和模型 ID
2. **`Context`**：对话上下文，包含系统提示、消息历史和工具定义
3. **`complete()` / `stream()`**：发送请求，获取完整或流式响应

---

# 核心概念

## 对话模型

pi-ai 的核心抽象是**模型无关的对话上下文**（`Context`）。同一个 `Context` 可以传给任何提供商的模型，API 自动处理格式转换。

```
用户消息 → Context → complete(model, context) → 响应消息
                                   ↓
                            提供商适配层
                          (OpenAI/Anthropic/Google/...)
```

## 响应模式

| 模式 | 函数 | 说明 |
|:---|:---|:---|
| 完整响应 | `complete()` | 等待完整响应后返回 |
| 流式响应 | `stream()` | 返回异步迭代器，实时产出事件 |

## 令牌与成本

每次响应自动附带用量数据：

```typescript
const response = await complete(model, context);
console.log(`输入令牌: ${response.usage.input}`);
console.log(`输出令牌: ${response.usage.output}`);
console.log(`成本: $${response.usage.cost.total.toFixed(4)}`);
```

---

# TypeScript API 参考

`@earendil-works/pi-ai` 的完整 API 参考。

## 安装

```bash
npm install @earendil-works/pi-ai
```

## 函数

### `complete()`

发送完整请求，等待模型返回完整响应。

```typescript
function complete(
  model: Model,
  context: Context,
  options?: CompleteOptions
): Promise<Message>;
```

| 参数 | 类型 | 说明 |
|:---|:---|:---|
| `model` | `Model` | 模型配置，通过 `getModel()` 或自定义构建 |
| `context` | `Context` | 对话上下文 |
| `options` | `CompleteOptions` | 可选配置（API 密钥、中止信号等） |

返回 `Message` 对象，包含 `content`、`usage`、`stopReason` 等字段。

### `stream()`

发送流式请求，返回异步迭代器实时产出事件。

```typescript
function stream(
  model: Model,
  context: Context,
  options?: StreamOptions
): Stream;
```

| 参数 | 类型 | 说明 |
|:---|:---|:---|
| `model` | `Model` | 模型配置 |
| `context` | `Context` | 对话上下文 |
| `options` | `StreamOptions` | 可选配置 |

返回 `Stream` 对象，既是 `AsyncIterable<StreamEvent>` 又有 `result()` 方法获取最终消息。

```typescript
const s = stream(model, context);

for await (const event of s) {
  if (event.type === "text_delta") {
    process.stdout.write(event.delta);
  }
}

const finalMessage = await s.result();
context.messages.push(finalMessage);
```

### `completeSimple()`

简化接口，自动处理思考/推理配置。

```typescript
function completeSimple(
  model: Model,
  context: Context,
  options?: { reasoning?: ReasoningLevel }
): Promise<SimpleMessage>;
```

### `streamSimple()`

简化流式接口。

```typescript
function streamSimple(
  model: Model,
  context: Context,
  options?: { reasoning?: ReasoningLevel; signal?: AbortSignal }
): AsyncIterable<SimpleStreamEvent>;
```

### `getModel()`

获取内置模型配置，支持 IDE 自动补全。

```typescript
function getModel(provider: string, modelId: string): Model;
```

```typescript
const model = getModel("anthropic", "claude-sonnet-4-20250514");
const model = getModel("openai", "gpt-4o-mini");
const model = getModel("google", "gemini-2.5-flash");
```

### `getModels()`

列出指定提供商的所有可用模型。

```typescript
function getModels(provider: string): ModelInfo[];
```

### `getProviders()`

列出所有可用提供商。

```typescript
function getProviders(): string[];
```

### `getEnvApiKey()`

检查提供商对应的环境变量 API 密钥。

```typescript
function getEnvApiKey(provider: string): string | undefined;
```

### `getImageModel()`

获取图片生成模型配置。

```typescript
function getImageModel(provider: string, modelId: string): ImageModel;
```

### `generateImages()`

使用图片生成模型生成图片。

```typescript
function generateImages(
  model: ImageModel,
  input: ImageInput,
  options?: { apiKey?: string }
): Promise<ImageResult>;
```

### `validateToolCall()`

验证工具调用参数是否符合定义的模式。

```typescript
function validateToolCall(
  tools: Tool[],
  toolCall: ToolCallBlock
): Record<string, unknown>;
```

### `registerFauxProvider()`

注册测试用的模拟提供商。详见 [Faux Provider](#faux-provider测试)。

## 类型

### `Model`

模型配置对象，描述提供商、API 类型、能力、成本等。

```typescript
type Model<TApi extends string = string> = {
  id: string;                                    // 模型 ID（如 "gpt-4o-mini"）
  name: string;                                  // 显示名称
  api: TApi;                                     // API 类型标识
  provider: string;                               // 提供商标识
  baseUrl?: string;                               // 自定义 API 端点
  reasoning: boolean;                             // 是否支持推理/思考
  input: ("text" | "image")[];                    // 支持的输入类型
  cost: {                                         // 每百万令牌成本（USD）
    input: number;
    output: number;
    cacheRead: number;
    cacheWrite: number;
  };
  contextWindow: number;                          // 上下文窗口大小
  maxTokens: number;                              // 最大输出令牌数
  headers?: Record<string, string>;               // 自定义请求头
  thinkingLevelMap?: Record<string, string | null>; // 思考级别映射
  compat?: {                                       // 兼容性设置
    supportsDeveloperRole?: boolean;
    supportsReasoningEffort?: boolean;
  };
};
```

### `Context`

对话上下文，包含系统提示、消息历史和工具定义。

```typescript
type Context = {
  systemPrompt?: string;            // 系统提示
  messages: Message[];               // 消息历史
  tools?: Tool[];                    // 可用工具定义
};
```

### `Message`

对话消息，包含内容块和用量数据。

```typescript
type Message = {
  role: "assistant";
  content: ContentBlock[];          // 内容块（文本、工具调用、思考等）
  usage: {
    input: number;                   // 输入令牌数
    output: number;                  // 输出令牌数
    cost: {
      input: number;                 // 输入成本
      output: number;                // 输出成本
      cacheRead: number;             // 缓存读取成本
      cacheWrite: number;            // 缓存写入成本
      total: number;                // 总成本
    };
  };
  stopReason: StopReason;           // 停止原因
  timestamp: number;                 // 时间戳
};
```

### `Tool`

工具定义，包含名称、描述和 TypeBox 参数模式。

```typescript
type Tool = {
  name: string;                      // 工具名称
  description: string;               // 工具描述
  parameters: TSchema;               // TypeBox 参数模式
};
```

### `CompleteOptions`

```typescript
type CompleteOptions = {
  apiKey?: string;                   // API 密钥（覆盖环境变量）
  signal?: AbortSignal;              // 中止信号
  thinkingEnabled?: boolean;         // 启用思考（Anthropic）
  thinkingBudgetTokens?: number;     // 思考令牌预算（Anthropic）
  reasoningEffort?: string;          // 推理努力程度（OpenAI）
  reasoningSummary?: string;         // 推理摘要级别（OpenAI）
  thinking?: { enabled: boolean; budgetTokens?: number }; // 思考配置（Google）
  onPayload?: (payload: unknown) => void; // 调试：查看发送给提供商的原始请求
};
```

### `StopReason`

| 值 | 说明 |
|:---|:---|
| `"stop"` | 正常完成 |
| `"length"` | 输出达到最大令牌限制 |
| `"toolUse"` | 模型请求工具调用 |
| `"error"` | 生成过程中出错 |
| `"aborted"` | 请求被中止 |

## 事件类型

### 流式事件

`stream()` 返回的异步迭代器产出以下事件：

| 事件类型 | 说明 | 关键属性 |
|:---|:---|:---|
| `start` | 流开始 | `partial`: 初始助手消息结构 |
| `text_start` | 文本块开始 | `contentIndex` |
| `text_delta` | 文本增量 | `delta`, `contentIndex` |
| `text_end` | 文本块完成 | `content`, `contentIndex` |
| `thinking_start` | 思考块开始 | `contentIndex` |
| `thinking_delta` | 思考增量 | `delta`, `contentIndex` |
| `thinking_end` | 思考块完成 | `content`, `contentIndex` |
| `toolcall_start` | 工具调用开始 | `contentIndex` |
| `toolcall_delta` | 工具参数流式增量 | `delta`, `partial`, `contentIndex` |
| `toolcall_end` | 工具调用完成 | `toolCall`: `{ id, name, arguments }` |
| `done` | 流完成 | `reason`, `message` |
| `error` | 出错 | `reason`（"error" 或 "aborted"）, `error` |

**注意：** 不同内容块的事件**不保证连续**。使用 `contentIndex` 关联事件与其所属块。

### 简化流式事件

`streamSimple()` 产出的事件：

| 事件类型 | 说明 |
|:---|:---|
| `thinking_start` | 思考开始 |
| `thinking_delta` | 思考增量 |
| `thinking_end` | 思考完成 |
| `text_delta` | 文本增量 |
| `toolcall_end` | 工具调用完成 |
| `done` | 流完成 |
| `error` | 出错 |

## 内容块类型

### `ContentBlock`

`Message.content` 数组中的每个元素。

```typescript
type ContentBlock =
  | TextBlock
  | ThinkingBlock
  | ToolCallBlock
  | ToolResultBlock;
```

### `TextBlock`

```typescript
type TextBlock = {
  type: "text";
  text: string;
};
```

### `ThinkingBlock`

```typescript
type ThinkingBlock = {
  type: "thinking";
  thinking: string;
};
```

### `ToolCallBlock`

```typescript
type ToolCallBlock = {
  type: "toolCall";
  id: string;
  name: string;
  arguments: Record<string, unknown>;
};
```

### `ToolResultBlock`

```typescript
type ToolResultBlock = {
  role: "toolResult";
  toolCallId: string;
  toolName: string;
  content: Array<TextContent | ImageContent>;
  isError: boolean;
  timestamp: number;
};
```

---

# 指南

## 模型与提供商配置

### 内置模型发现

```typescript
import { getProviders, getModels, getModel } from "@earendil-works/pi-ai";

// 列出所有提供商
const providers = getProviders();
// ['openai', 'anthropic', 'google', 'xai', 'groq', ...]

// 列出某提供商的所有模型
const anthropicModels = getModels("anthropic");
for (const model of anthropicModels) {
  console.log(`${model.id}: ${model.name}`);
  console.log(`  API: ${model.api}`);
  console.log(`  上下文: ${model.contextWindow} 令牌`);
  console.log(`  视觉: ${model.input.includes("image")}`);
  console.log(`  推理: ${model.reasoning}`);
}

// 获取特定模型（IDE 自动补全）
const model = getModel("openai", "gpt-4o-mini");
```

### 支持的提供商

| 提供商 | API 类型 |
|:---|:---|
| OpenAI | `openai-responses` |
| Azure OpenAI | `azure-openai-responses` |
| Anthropic | `anthropic-messages` |
| Google | `google-generative-ai` |
| Vertex AI | `google-vertex` |
| Mistral | `mistral-conversations` |
| DeepSeek | `openai-completions` |
| xAI | `openai-completions` |
| Groq | `openai-completions` |
| Cerebras | `openai-completions` |
| Cloudflare AI Gateway | `openai-completions` |
| Cloudflare Workers AI | `openai-completions` |
| OpenRouter | `openai-completions` |
| Vercel AI Gateway | `openai-completions` |
| MiniMax | `openai-completions` |
| Together AI | `openai-completions` |
| GitHub Copilot | `openai-completions`（OAuth） |
| Amazon Bedrock | `bedrock-converse-stream` |
| Fireworks | `anthropic-messages`（兼容） |
| Kimi For Coding | `anthropic-messages`（兼容） |
| 小米 MiMo | `anthropic-messages`（兼容） |
| 任意 OpenAI 兼容 API | `openai-completions` |

### 环境变量

| 提供商 | 环境变量 |
|:---|:---|
| OpenAI | `OPENAI_API_KEY` |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_BASE_URL` |
| Anthropic | `ANTHROPIC_API_KEY` 或 `ANTHROPIC_OAUTH_TOKEN` |
| DeepSeek | `DEEPSEEK_API_KEY` |
| Google | `GEMINI_API_KEY` |
| Vertex AI | `GOOGLE_CLOUD_API_KEY` 或 ADC |
| Mistral | `MISTRAL_API_KEY` |
| Groq | `GROQ_API_KEY` |
| Cerebras | `CEREBRAS_API_KEY` |
| xAI | `XAI_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |
| Together AI | `TOGETHER_API_KEY` |
| GitHub Copilot | `COPILOT_GITHUB_TOKEN` |
| Fireworks | `FIREWORKS_API_KEY` |
| MiniMax | `MINIMAX_API_KEY` |
| Cloudflare | `CLOUDFLARE_API_KEY` + `CLOUDFLARE_ACCOUNT_ID` |
| Vercel AI Gateway | `AI_GATEWAY_API_KEY` |

---

## 自定义模型

用于 Ollama、vLLM、LM Studio 等自托管或 OpenAI 兼容的 API。

### 基本自定义模型

```typescript
import { type Model, stream } from "@earendil-works/pi-ai";

const ollamaModel: Model<"openai-completions"> = {
  id: "llama-3.1-8b",
  name: "Llama 3.1 8B (Ollama)",
  api: "openai-completions",
  provider: "ollama",
  baseUrl: "http://localhost:11434/v1",
  reasoning: false,
  input: ["text"],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  contextWindow: 128000,
  maxTokens: 32000,
};

const response = await stream(ollamaModel, context, {
  apiKey: "dummy", // Ollama 不需要真实密钥
});
```

### 带兼容性设置的自定义模型

```typescript
const customModel: Model<"openai-completions"> = {
  id: "gpt-oss:20b",
  name: "GPT-OSS 20B",
  api: "openai-completions",
  provider: "ollama",
  baseUrl: "http://localhost:11434/v1",
  reasoning: true,
  input: ["text"],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  contextWindow: 131072,
  maxTokens: 32000,
  thinkingLevelMap: {
    minimal: null, low: null, medium: null, high: "high", xhigh: null,
  },
  compat: {
    supportsDeveloperRole: false,
    supportsReasoningEffort: false,
  },
};
```

### 带代理的自定义模型

```typescript
const proxyModel: Model<"anthropic-messages"> = {
  id: "claude-sonnet-4",
  name: "Claude Sonnet 4 (Proxied)",
  api: "anthropic-messages",
  provider: "custom-proxy",
  baseUrl: "https://proxy.example.com/v1",
  reasoning: true,
  input: ["text", "image"],
  cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
  contextWindow: 200000,
  maxTokens: 8192,
  headers: {
    "X-Custom-Auth": "bearer-token-here",
  },
};
```

---

## 工具（函数调用）

### 定义工具

使用 TypeBox 模式定义工具参数：

```typescript
import { Type, StringEnum, type Tool } from "@earendil-works/pi-ai";

const weatherTool: Tool = {
  name: "get_weather",
  description: "获取指定地点的当前天气",
  parameters: Type.Object({
    location: Type.String({ description: "城市名或坐标" }),
    units: StringEnum(["celsius", "fahrenheit"], { default: "celsius" }),
  }),
};
```

**注意：** 使用 `StringEnum`（而非 `Type.Enum`）以确保 Google API 兼容性。

### 使用工具发送请求

```typescript
const model = getModel("openai", "gpt-4o-mini");

const response = await complete(model, {
  systemPrompt: "You are a weather assistant.",
  messages: [{ role: "user", content: "What's the weather in Tokyo?" }],
  tools: [weatherTool],
});
```

### 处理工具调用与结果

```typescript
for (const block of response.content) {
  if (block.type === "toolCall") {
    const result = await executeWeatherApi(block.arguments);

    // 将工具结果推入上下文
    context.messages.push({
      role: "toolResult",
      toolCallId: block.id,
      toolName: block.name,
      content: [{ type: "text", text: JSON.stringify(result) }],
      isError: false,
      timestamp: Date.now(),
    });
  }
}

// 如果有工具调用，继续对话
if (response.content.some((b) => b.type === "toolCall")) {
  context.messages.push(response);
  const continuation = await complete(model, context);
  context.messages.push(continuation);
}
```

### 工具结果中的图片

```typescript
import { readFileSync } from "fs";

const imageBuffer = readFileSync("chart.png");
context.messages.push({
  role: "toolResult",
  toolCallId: "tool_xyz",
  toolName: "generate_chart",
  content: [
    { type: "text", text: "Generated chart showing temperature trends" },
    { type: "image", data: imageBuffer.toString("base64"), mimeType: "image/png" },
  ],
  isError: false,
  timestamp: Date.now(),
});
```

### 验证工具参数

```typescript
import { validateToolCall } from "@earendil-works/pi-ai";

for await (const event of stream(model, context)) {
  if (event.type === "toolcall_end") {
    try {
      const validatedArgs = validateToolCall(tools, event.toolCall);
      const result = await executeMyTool(event.toolCall.name, validatedArgs);
    } catch (error) {
      context.messages.push({
        role: "toolResult",
        toolCallId: event.toolCall.id,
        toolName: event.toolCall.name,
        content: [{ type: "text", text: error.message }],
        isError: true,
        timestamp: Date.now(),
      });
    }
  }
}
```

---

## 流式输出

### 基本流式

```typescript
const s = stream(model, context);

for await (const event of s) {
  switch (event.type) {
    case "text_delta":
      process.stdout.write(event.delta);
      break;
    case "toolcall_end":
      console.log(`工具调用: ${event.toolCall.name}`);
      console.log(`参数: ${JSON.stringify(event.toolCall.arguments)}`);
      break;
    case "done":
      console.log(`\n完成: ${event.reason}`);
      break;
    case "error":
      console.error(`错误: ${event.error}`);
      break;
  }
}

// 获取最终消息
const finalMessage = await s.result();
context.messages.push(finalMessage);
```

### 流式工具调用

```typescript
const s = stream(model, context);

for await (const event of s) {
  if (event.type === "toolcall_delta") {
    const toolCall = event.partial.content[event.contentIndex];
    if (toolCall.type === "toolCall" && toolCall.arguments) {
      // 注意：流式期间 arguments 可能不完整，需防御性处理
      if (toolCall.name === "write_file" && toolCall.arguments.path) {
        console.log(`写入: ${toolCall.arguments.path}`);
      }
    }
  }
  if (event.type === "toolcall_end") {
    // 完整参数在此可用
    console.log(`工具完成: ${event.toolCall.name}`, event.toolCall.arguments);
  }
}
```

**注意：** Google 提供商**不支持**函数调用流式——你会收到一个包含完整参数的单个 `toolcall_delta`。

### 调试提供商请求

```typescript
const response = await complete(model, context, {
  onPayload: (payload) => {
    console.log("提供商请求:", JSON.stringify(payload, null, 2));
  },
});
```

---

## 思考/推理

### 简化接口

```typescript
import { getModel, completeSimple } from "@earendil-works/pi-ai";

const model = getModel("anthropic", "claude-sonnet-4-20250514");

if (model.reasoning) {
  console.log("模型支持推理/思考");
}

const response = await completeSimple(model, {
  messages: [{ role: "user", content: "Solve: 2x + 5 = 13" }],
}, {
  reasoning: "medium", // 'minimal' | 'low' | 'medium' | 'high' | 'xhigh'
});

for (const block of response.content) {
  if (block.type === "thinking") console.log("思考:", block.thinking);
  else if (block.type === "text") console.log("回答:", block.text);
}
```

### 提供商特定选项

```typescript
// OpenAI（o1, o3, gpt-5）
await complete(openaiModel, context, {
  reasoningEffort: "medium",
  reasoningSummary: "detailed",
});

// Anthropic（Claude Sonnet 4）
await complete(anthropicModel, context, {
  thinkingEnabled: true,
  thinkingBudgetTokens: 8192,
});

// Google Gemini
await complete(googleModel, context, {
  thinking: { enabled: true, budgetTokens: 8192 }, // -1 = 动态, 0 = 禁用
});
```

### 流式思考内容

```typescript
const s = streamSimple(model, context, { reasoning: "high" });

for await (const event of s) {
  switch (event.type) {
    case "thinking_start":
      console.log("[模型开始思考]");
      break;
    case "thinking_delta":
      process.stdout.write(event.delta);
      break;
    case "thinking_end":
      console.log("\n[思考完成]");
      break;
  }
}
```

---

## 图片输入

```typescript
import { readFileSync } from "fs";
import { getModel, complete } from "@earendil-works/pi-ai";

const model = getModel("openai", "gpt-4o-mini");

if (model.input.includes("image")) {
  console.log("模型支持视觉");
}

const imageBuffer = readFileSync("image.png");
const base64Image = imageBuffer.toString("base64");

const response = await complete(model, {
  messages: [{
    role: "user",
    content: [
      { type: "text", text: "这张图片里有什么？" },
      { type: "image", data: base64Image, mimeType: "image/png" },
    ],
  }],
});
```

---

## 图片生成

```typescript
import { getImageModel, generateImages } from "@earendil-works/pi-ai";

const model = getImageModel("openrouter", "google/gemini-2.5-flash-image");

const result = await generateImages(model, {
  input: [{ type: "text", text: "生成一个白色背景上的红色圆形。" }],
}, {
  apiKey: process.env.OPENROUTER_API_KEY,
});

for (const block of result.output) {
  if (block.type === "text") console.log(block.text);
  else if (block.type === "image") console.log(block.mimeType, block.data.substring(0, 32));
}
```

**重要：** 使用 `getImageModel()` + `generateImages()` 进行图片生成。**不要**使用 `stream()` 或 `complete()`。

---

## 跨提供商切换

在同一对话中无缝切换模型——思考块自动转换为 `<thinking>` 标记文本。

```typescript
import { getModel, complete } from "@earendil-works/pi-ai";

const claude = getModel("anthropic", "claude-sonnet-4-20250514");
const context = { messages: [] };

context.messages.push({ role: "user", content: "25 * 18 等于多少？" });
const claudeResponse = await complete(claude, context, { thinkingEnabled: true });
context.messages.push(claudeResponse);

// 切换到 GPT-5——Claude 的思考以 <thinking> 标记文本形式呈现
const gpt5 = getModel("openai", "gpt-5-mini");
context.messages.push({ role: "user", content: "那个计算对吗？" });
const gptResponse = await complete(gpt5, context);
context.messages.push(gptResponse);

// 切换到 Gemini
const gemini = getModel("google", "gemini-2.5-flash");
context.messages.push({ role: "user", content: "原问题是什么？" });
const geminiResponse = await complete(gemini, context);
```

---

## 上下文序列化

`Context` 对象可直接序列化为 JSON 并恢复，支持会话持久化。

```typescript
const context = {
  systemPrompt: "You are a helpful assistant.",
  messages: [{ role: "user", content: "What is TypeScript?" }],
};

const model = getModel("openai", "gpt-4o-mini");
const response = await complete(model, context);
context.messages.push(response);

// 序列化
const serialized = JSON.stringify(context);
localStorage.setItem("conversation", serialized);

// 反序列化并继续
const restored = JSON.parse(localStorage.getItem("conversation")!);
restored.messages.push({ role: "user", content: "告诉我更多" });
const newModel = getModel("anthropic", "claude-3-5-haiku-20241022");
const continuation = await complete(newModel, restored);
```

---

## 错误处理与中止

```typescript
const controller = new AbortController();
setTimeout(() => controller.abort(), 2000);

const s = stream(model, {
  messages: [{ role: "user", content: "Write a long story" }],
}, { signal: controller.signal });

for await (const event of s) {
  if (event.type === "text_delta") {
    process.stdout.write(event.delta);
  } else if (event.type === "error") {
    console.log(`${event.reason === "aborted" ? "已中止" : "错误"}:`, event.error.errorMessage);
  }
}

const response = await s.result();
if (response.stopReason === "aborted") {
  console.log("部分内容:", response.content);
}
```

---

## Faux Provider（测试）

使用模拟提供商编写确定性测试，无需真实 API 调用。

### 注册与使用

```typescript
import {
  complete, stream,
  fauxAssistantMessage, fauxText, fauxThinking,
  fauxToolCall, registerFauxProvider,
} from "@earendil-works/pi-ai";

const registration = registerFauxProvider({ tokensPerSecond: 50 });
const model = registration.getModel();

registration.setResponses([
  fauxAssistantMessage([
    fauxThinking("需要先检查包元数据。"),
    fauxToolCall("echo", { text: "package.json" }),
  ], { stopReason: "toolUse" }),
]);

const result = await complete(model, context);

// 清理
registration.unregister();
```

### 可用的 Faux 构造器

| 函数 | 说明 |
|:---|:---|
| `fauxAssistantMessage(blocks, options?)` | 构造助手消息 |
| `fauxText(text)` | 构造文本块 |
| `fauxThinking(text)` | 构造思考块 |
| `fauxToolCall(name, arguments)` | 构造工具调用块 |

---

## 另见

- [Pi GitHub 仓库](https://github.com/earendil-works/pi)
- [Pi 官网](https://pi.dev)
- [Pi 文档](https://pi.dev/docs/latest)
- [Pi Discord](https://discord.com/invite/3cU7Bz4UPx)
