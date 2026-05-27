# SDK 会话管理

> **版本要求**：本文档针对 CodeBuddy Agent SDK v0.1.0 及以上版本。

本文档介绍如何在 SDK 中管理会话，包括获取会话 ID、恢复会话、分叉会话和多轮对话。

## 概述

会话（Session）是 CodeBuddy 的核心概念，用于：

- **保持对话上下文**：多轮对话中 AI 能记住之前的内容
- **支持会话恢复**：可以从上次中断的地方继续
- **支持会话分叉**：从某个点创建分支，探索不同方向

每个会话都有一个唯一的 `session_id`，可以用于后续恢复。

## 获取会话 ID

会话开始时，SDK 会返回一个 `system` 类型的初始化消息,其中包含 `session_id`。

### 使用 query API

**TypeScript:**

```typescript
import { query } from '@tencent-ai/agent-sdk';

let sessionId: string | undefined;

const q = query({
  prompt: '帮我构建一个 Web 应用',
  options: {
    model: 'deepseek-v3.1'
  }
});

for await (const message of q) {
  if (message.type === 'system' && message.subtype === 'init') {
    sessionId = message.session_id;
    console.log(`会话 ID: ${sessionId}`);
  }

  console.log(message);
}
```

**Python:**

```python
import asyncio
from codebuddy_agent_sdk import query, CodeBuddyAgentOptions, SystemMessage

session_id = None

async def main():
    global session_id

    options = CodeBuddyAgentOptions(model="deepseek-v3.1")

    async for message in query(prompt="帮我构建一个 Web 应用", options=options):
        if isinstance(message, SystemMessage):
            session_id = message.data.get("session_id")
            print(f"会话 ID: {session_id}")

        print(message)

asyncio.run(main())
```

### 使用 v2 Session API (TypeScript)

```typescript
import { unstable_v2_createSession } from '@tencent-ai/agent-sdk';

const session = unstable_v2_createSession({
  model: 'deepseek-v3.1'
});

await session.send('帮我构建一个 Web 应用');

for await (const message of session.stream()) {
  if (message.type === 'system' && message.subtype === 'init') {
    console.log(`会话 ID: ${message.session_id}`);
  }
  console.log(message);
}

console.log(`会话 ID: ${session.sessionId}`);

session.close();
```

### 使用 Client API (Python)

```python
from codebuddy_agent_sdk import CodeBuddySDKClient, CodeBuddyAgentOptions

async def main():
    options = CodeBuddyAgentOptions(model="deepseek-v3.1")

    async with CodeBuddySDKClient(options=options) as client:
        await client.query("帮我构建一个 Web 应用")

        async for message in client.receive_response():
            if isinstance(message, SystemMessage):
                print(f"会话 ID: {message.data.get('session_id')}")
            print(message)
```

## 恢复会话

使用之前保存的 `session_id` 可以恢复会话，继续之前的对话。

### 使用 resume 选项

**TypeScript - query API:**

```typescript
import { query } from '@tencent-ai/agent-sdk';

const savedSessionId = 'abc123-xyz789';

const q = query({
  prompt: '继续我们之前的工作',
  options: {
    model: 'deepseek-v3.1',
    resume: savedSessionId
  }
});

for await (const message of q) {
  console.log(message);
}
```

**TypeScript - v2 API:**

```typescript
import { unstable_v2_resumeSession } from '@tencent-ai/agent-sdk';

const savedSessionId = 'abc123-xyz789';

const session = unstable_v2_resumeSession(savedSessionId, {
  model: 'deepseek-v3.1'
});

await session.send('继续我们之前的工作');

for await (const message of session.stream()) {
  console.log(message);
}

session.close();
```

**Python:**

```python
from codebuddy_agent_sdk import query, CodeBuddyAgentOptions

saved_session_id = "abc123-xyz789"

options = CodeBuddyAgentOptions(
    model="deepseek-v3.1",
    resume=saved_session_id
)

async for message in query(prompt="继续我们之前的工作", options=options):
    print(message)
```

### 继续最近的会话

使用 `continue` / `continue_conversation` 选项可以自动继续最近的会话：

**TypeScript:**

```typescript
const q = query({
  prompt: '继续',
  options: {
    model: 'deepseek-v3.1',
    continue: true
  }
});
```

**Python:**

```python
options = CodeBuddyAgentOptions(
    model="deepseek-v3.1",
    continue_conversation=True
)
```

## 多轮对话

多轮对话允许在同一个会话中进行多次交互，保持上下文连贯。

### TypeScript：使用 query API

每次新的 query 调用使用 `resume` 恢复会话：

```typescript
import { query } from '@tencent-ai/agent-sdk';

async function multiTurnWithQuery() {
  let sessionId: string;

  // 第一轮对话
  const q1 = query({
    prompt: '帮我创建一个 React 项目',
    options: { model: 'deepseek-v3.1' }
  });

  for await (const msg of q1) {
    if (msg.type === 'system' && msg.subtype === 'init') {
      sessionId = msg.session_id;
    }
    if (msg.type === 'result') {
      console.log('第一轮完成');
    }
  }

  // 第二轮对话（恢复会话）
  const q2 = query({
    prompt: '添加一个用户登录页面',
    options: {
      model: 'deepseek-v3.1',
      resume: sessionId
    }
  });

  for await (const msg of q2) {
    if (msg.type === 'result') {
      console.log('第二轮完成');
    }
  }
}
```

### TypeScript：使用 v2 Session API

v2 API 提供更简洁的多轮对话体验：

```typescript
import { unstable_v2_createSession } from '@tencent-ai/agent-sdk';

async function multiTurnWithSession() {
  const session = unstable_v2_createSession({
    model: 'deepseek-v3.1'
  });

  try {
    await session.send('帮我创建一个 React 项目');
    for await (const msg of session.stream()) {
      console.log(msg);
    }

    await session.send('添加一个用户登录页面');
    for await (const msg of session.stream()) {
      console.log(msg);
    }

    await session.send('添加表单验证');
    for await (const msg of session.stream()) {
      console.log(msg);
    }

  } finally {
    session.close();
  }
}
```

### Python：使用 CodeBuddySDKClient

```python
from codebuddy_agent_sdk import CodeBuddySDKClient, CodeBuddyAgentOptions

async def multi_turn_conversation():
    options = CodeBuddyAgentOptions(model="deepseek-v3.1")

    async with CodeBuddySDKClient(options=options) as client:
        # 第一轮对话
        await client.query("帮我创建一个 React 项目")
        async for msg in client.receive_response():
            print(msg)

        # 第二轮对话（自动保持上下文）
        await client.query("添加一个用户登录页面")
        async for msg in client.receive_response():
            print(msg)

        # 第三轮对话
        await client.query("添加表单验证")
        async for msg in client.receive_response():
            print(msg)
```

## 相关文档

- [SDK 概览](codebuddy-sdk-quickstart.md) - 快速入门和使用示例
- [SDK 权限控制](codebuddy-sdk-permissions.md) - 权限模式和 canUseTool
- [TypeScript SDK 参考](codebuddy-sdk-typescript.md) - 完整 API 参考
- [Python SDK 参考](codebuddy-sdk-python.md) - 完整 API 参考
