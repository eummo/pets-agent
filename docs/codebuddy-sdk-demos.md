# SDK 示例项目

本文档介绍 CodeBuddy Agent SDK 的官方示例项目，帮助你快速了解 SDK 的各种使用场景。

## 示例仓库

所有示例代码托管在官方仓库：

**仓库地址**：https://cnb.cool/codebuddy/agent-sdk-demos

```bash
git clone https://cnb.cool/codebuddy/agent-sdk-demos.git
cd agent-sdk-demos
```

## 示例概览

| 示例 | 语言 | 核心功能 | 适用场景 |
|---|---|---|---|
| quick-start | TypeScript | 基础 API、消息流、Hooks | SDK 入门 |
| multi-turn-session | TypeScript | 多轮对话、会话恢复 | 交互式应用 |
| research-assistant | Python | 多 Agent 协作 | 复杂任务分解 |
| profile-builder | TypeScript | 网络搜索、文档生成 | 信息收集 |
| chat-demo | TypeScript | WebSocket、流式响应 | Web 应用 |
| mail-assistant | TypeScript | MCP 协议、自定义工具 | 业务系统对接 |
| spreadsheet-assistant | TypeScript | Electron IPC | 桌面应用 |

## 环境准备

### 前置条件

- [Bun](https://bun.sh) 或 Node.js 18+
- Python 3.10+（Python 示例）
- 已完成 CodeBuddy CLI 登录认证

### 安装 SDK

**TypeScript**

```bash
npm install @tencent-ai/agent-sdk
```

**Python**

```bash
pip install codebuddy-agent-sdk
```

### 认证方式

SDK 支持多种认证方式：

1. **复用 CLI 登录态**：如果已通过 `codebuddy` 命令登录，SDK 自动使用现有凭据
2. **API Key 认证**：通过环境变量配置

详细的认证配置请参阅 [SDK 概览 - 认证配置](codebuddy-sdk-quickstart.md#认证配置)

## 基础示例

### quick-start：SDK 入门

演示 `query()` API 的基本用法，包括消息流处理和 Hooks 机制。

```typescript
import { query } from '@tencent-ai/agent-sdk';

const conversation = query({
  prompt: '你好！请介绍一下你能做什么。',
  options: {
    model: 'claude-4.5',
    maxTurns: 100,
    allowedTools: ['Read', 'Write', 'Bash', 'Glob', 'Grep'],
  },
});

for await (const message of conversation) {
  if (message.type === 'assistant') {
    const text = message.message.content.find(c => c.type === 'text');
    if (text) console.log(text.text);
  }
  if (message.type === 'result') {
    console.log(`完成，耗时 ${message.duration_ms}ms`);
  }
}
```

**运行示例**：

```bash
cd quick-start
npm install
npx tsx quick-start.ts
```

### multi-turn-session：多轮对话

演示 Session API 实现多轮对话和会话恢复。

```typescript
import { unstable_v2_createSession, unstable_v2_resumeSession } from '@tencent-ai/agent-sdk';

// 创建会话
await using session = unstable_v2_createSession({ model: 'claude-4.5' });

// 第一轮
await session.send('今年是哪一年？');
for await (const msg of session.stream()) { /* ... */ }

// 第二轮（保持上下文）
await session.send('再往后推 10 年是哪一年？');
for await (const msg of session.stream()) { /* ... */ }
```

**运行示例**：

```bash
cd multi-turn-session
npm install
npx tsx examples.ts basic        # 基础会话
npx tsx examples.ts multi-turn   # 多轮对话
npx tsx examples.ts resume       # 会话恢复
```

## 进阶示例

### research-assistant：多 Agent 协作

Python 示例，展示如何定义多个专业化子 Agent 协作完成复杂任务。

**工作流程**：

1. **主 Agent** 将研究请求拆分为子任务
2. **研究员** 使用 WebSearch 搜索信息，保存到 `files/research_notes/`
3. **数据分析师** 从研究笔记提取数据，生成图表到 `files/charts/`
4. **报告撰写者** 整合内容，生成 PDF 报告到 `files/reports/`

```python
from codebuddy_agent_sdk import CodeBuddySDKClient, CodeBuddyAgentOptions, AgentDefinition

agents = {
    "researcher": AgentDefinition(
        description="使用网络搜索收集研究信息",
        tools=["WebSearch", "Write"],
        model="claude-haiku-4.5"
    ),
    "data-analyst": AgentDefinition(
        description="从研究笔记提取数据并生成图表",
        tools=["Glob", "Read", "Bash", "Write"],
        model="claude-haiku-4.5"
    ),
    "report-writer": AgentDefinition(
        description="整合研究和数据生成 PDF 报告",
        tools=["Skill", "Glob", "Read", "Write", "Bash"],
        model="claude-haiku-4.5"
    )
}

options = CodeBuddyAgentOptions(
    allowed_tools=["Task"],  # 主 Agent 只能委托任务
    agents=agents,
    model="claude-haiku-4.5"
)

async with CodeBuddySDKClient(options=options) as client:
    await client.query("研究 2025 年量子计算发展")
    async for msg in client.receive_response():
        # 处理消息
```

**运行示例**：

```bash
cd research-assistant
uv sync
uv run python research_agent/agent.py
```

### profile-builder：信息收集与文档生成

演示 WebSearch 工具和文档生成能力。

```typescript
const q = query({
  prompt: `搜索 "${personName}" 的资料，创建一份专业简历`,
  options: {
    allowedTools: ['WebSearch', 'WebFetch', 'Bash', 'Write', 'Read'],
    systemPrompt: '你是简历撰写专家...',
  },
});
```

**运行示例**：

```bash
cd profile-builder
npm install
npm start "姓名"
# 输出：agent/custom_scripts/resume.docx
```

## Web 应用集成

### chat-demo：流式响应架构

演示如何将 SDK 集成到 Web 应用，通过 WebSocket 实现流式响应。

**架构**：

```
Browser (React) ←─ WebSocket ─→ Express Server ←─ SDK query()
```

**服务端封装**：

```typescript
import { query } from "@tencent-ai/agent-sdk";

export class Agent {
  async sendMessage(content: string) {
    this.stream = query({
      prompt: content,
      options: {
        maxTurns: 1,
        allowedTools: ['Bash', 'Read', 'Write', 'WebSearch'],
      },
    })[Symbol.asyncIterator]();
  }

  async *getOutputStream() {
    while (true) {
      const { value, done } = await this.stream.next();
      if (done) break;
      yield value;
    }
  }
}
```

**运行示例**：

```bash
cd chat-demo
npm install
npm run dev
# 后端：http://localhost:3001
# 前端：http://localhost:5173
```

### mail-assistant：MCP 工具扩展

演示通过 MCP 协议扩展 Agent 能力，实现邮件系统操作。

```typescript
const q = query({
  prompt: '查找本周未读的重要邮件',
  options: {
    allowedTools: [
      'Read', 'Write', 'Bash',
      'mcp__email__search_inbox',   // MCP 工具
      'mcp__email__read_emails'
    ],
    mcpServers: {
      "email": customEmailServer
    },
  },
});
```

**运行示例**：

```bash
cd mail-assistant
cp .env.example .env  # 配置 IMAP 凭据
bun install
bun run dev
# 访问 http://localhost:3000
```

## 桌面应用集成

### spreadsheet-assistant：Electron 集成

演示在 Electron 应用中通过 IPC 集成 SDK。

**主进程**：

```typescript
import { query } from '@tencent-ai/agent-sdk';

ipcMain.on('agent:query', async (event, data) => {
  for await (const message of query({ prompt: data.content, options })) {
    event.reply('agent:response', message);
  }
});
```

**渲染进程**：

```typescript
window.electron.ipcRenderer.on('agent:response', (message) => {
  // 更新 UI
});

window.electron.ipcRenderer.sendMessage('agent:query', { content: '创建销售报表' });
```

**运行示例**：

```bash
cd spreadsheet-assistant
npm install
npm start
```

## Hooks 安全控制

所有示例都支持通过 Hooks 实现安全控制：

```typescript
const q = query({
  prompt: '...',
  options: {
    hooks: {
      PreToolUse: [{
        matcher: 'Write|Edit',
        hooks: [async (input) => {
          const filePath = input.tool_input.file_path;
          // 限制写入目录
          if (!filePath.startsWith('/allowed/path/')) {
            return { decision: 'block', stopReason: '路径不允许' };
          }
          return { continue: true };
        }]
      }]
    }
  }
});
```

## 相关文档

- [SDK 概览](codebuddy-sdk-quickstart.md) - SDK 完整介绍
- [TypeScript SDK 参考](codebuddy-sdk-typescript.md) - TypeScript API 详细文档
- [Python SDK 参考](codebuddy-sdk-python.md) - Python API 详细文档
- [SDK MCP 集成](codebuddy-sdk-mcp.md) - MCP 服务器配置
- [SDK 自定义工具](codebuddy-sdk-custom-tools.md) - Custom Tools 指南
