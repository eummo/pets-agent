# CodeBuddy Agent SDK 快速开始

> **版本要求**：本文档针对 CodeBuddy Agent SDK v0.1.0 及以上版本。
> **功能状态**：SDK 当前处于 **Preview** 阶段，接口和行为可能在未来版本中调整。

**重要：环境隔离**

SDK 默认**不加载任何文件系统配置**，包括 `settings.json`、`CODEBUDDY.md`、MCP 服务器、子代理、斜杠命令、Rules 和 Skills。这是与 CLI 直接使用的关键区别，确保 SDK 应用的行为完全由代码控制，具有可预测性和一致性。

如需加载这些配置，请使用 `settingSources` 选项显式指定。详见 [环境隔离](#环境隔离settingsources) 章节。

CodeBuddy Agent SDK 允许你在应用程序中以编程方式控制 CodeBuddy Agent。支持 TypeScript/JavaScript 和 Python，可实现自动化任务执行、自定义权限控制、构建 AI 驱动的开发工具等场景。

---

## 为什么使用 SDK

### 超越命令行的能力

- **程序化控制**：在你的应用程序中嵌入 AI 编程助手，实现自动化工作流
- **自定义交互**：构建符合你需求的用户界面和交互方式
- **批量处理**：对多个文件或项目执行批量 AI 操作
- **集成现有系统**：将 AI 能力无缝集成到 CI/CD、IDE 插件或其他开发工具中

### 精细化控制

- **权限管控**：通过 `canUseTool` 回调实现企业级权限策略
- **行为定制**：使用 Hook 系统拦截和修改 Agent 行为
- **资源限制**：控制 token 消耗、执行时间和费用预算
- **会话管理**：持久化和恢复对话上下文

### 扩展能力

- **自定义 Agent**：创建专门化的子 Agent 处理特定领域任务
- **MCP 集成**：接入自定义工具和服务
- **多模型支持**：灵活切换和配置不同的 AI 模型

---

## 你可以构建什么

### 开发工具增强

- **IDE 插件**：为 VS Code、JetBrains 等 IDE 构建智能编程助手
- **代码审查工具**：自动化代码质量检查和安全扫描
- **文档生成器**：自动生成 API 文档、README 和代码注释

### 自动化工作流

- **CI/CD 集成**：在流水线中执行智能代码分析和修复
- **测试生成**：自动生成单元测试和集成测试
- **重构助手**：批量执行代码重构和迁移任务

### 企业应用

- **内部开发平台**：构建企业级 AI 编程平台
- **知识库问答**：基于代码库的智能问答系统
- **培训工具**：交互式编程学习和代码评审系统

---

## 功能概览

- **消息流式传输**：实时接收系统消息、助手响应和工具调用结果
- **多轮对话**：支持跨多次推理调用的对话上下文保持
- **会话管理**：通过会话 ID 继续或恢复现有对话
- **权限控制**：细粒度的工具访问权限管理
- **Hook 系统**：在工具执行前后插入自定义逻辑
- **自定义 Agent**：定义专门化的子 Agent 处理特定任务
- **MCP 集成**：支持配置自定义 MCP 服务器扩展功能

---

## 安装

**TypeScript:**

```bash
npm install @tencent-ai/agent-sdk
# 或
yarn add @tencent-ai/agent-sdk
# 或
pnpm add @tencent-ai/agent-sdk
```

**Python:**

```bash
uv add codebuddy-agent-sdk
# 或
pip install codebuddy-agent-sdk
```

### 环境要求

| 语言 | 版本要求 |
|---|---|
| TypeScript/JavaScript | Node.js >= 18.20 |
| Python | Python >= 3.10 |

### 认证配置

#### 使用已有登录凭据

如果你已经在终端中通过 `codebuddy` 命令完成了交互式登录，SDK 会自动使用该认证信息，无需额外配置。

#### 使用 API Key

如果未登录或需要使用不同的凭据，可以通过 API Key 认证：

```bash
export CODEBUDDY_API_KEY="your-api-key"
```

**获取 API Key：**

| 版本 | 获取地址 |
|---|---|
| 海外版 | https://www.codebuddy.ai/profile/keys |
| 中国版 | https://copilot.tencent.com/profile/ |
| iOA 版 | https://tencent.sso.copilot.tencent.com/profile/keys |

> **注意**：使用 `CODEBUDDY_API_KEY` 时，必须根据版本正确配置 `CODEBUDDY_INTERNET_ENVIRONMENT` 环境变量：
>
> - 海外版：不设置（默认）
> - 中国版：`export CODEBUDDY_INTERNET_ENVIRONMENT=internal`
> - iOA 版：`export CODEBUDDY_INTERNET_ENVIRONMENT=ioa`

也可以在代码中通过 `env` 选项传递：

**TypeScript:**

```typescript
const q = query({
  prompt: '...',
  options: {
    env: {
      CODEBUDDY_API_KEY: process.env.MY_API_KEY,
      // 中国版用户需要设置：
      // CODEBUDDY_INTERNET_ENVIRONMENT: 'internal'
      // iOA 版用户需要设置：
      // CODEBUDDY_INTERNET_ENVIRONMENT: 'ioa'
    }
  }
});
```

**Python:**

```python
options = CodeBuddyAgentOptions(
    env={
        "CODEBUDDY_API_KEY": os.environ.get("MY_API_KEY"),
        # 中国版用户需要设置：
        # "CODEBUDDY_INTERNET_ENVIRONMENT": "internal"
        # iOA 版用户需要设置：
        # "CODEBUDDY_INTERNET_ENVIRONMENT": "ioa"
    }
)
```

#### 企业用户：OAuth Client Credentials

企业用户需要先通过 OAuth 2.0 Client Credentials 流程获取 access token，然后传入 SDK。

**第 1 步：创建应用获取凭据**

参考 [企业开发者快速入门](https://copilot.tencent.com/apiDocs/open-platform.html) 创建应用并获取 Client ID 和 Client Secret。

**第 2 步：获取 token 并调用 SDK**

**TypeScript:**

```typescript
async function getOAuthToken(clientId: string, clientSecret: string): Promise<string> {
  const response = await fetch('https://copilot.tencent.com/oauth2/token', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      grant_type: 'client_credentials',
      client_id: clientId,
      client_secret: clientSecret,
    }),
  });
  const data = await response.json();
  return data.access_token;
}

const token = await getOAuthToken('your-client-id', 'your-client-secret');

for await (const msg of query({
  prompt: 'Hello',
  options: {
    env: { CODEBUDDY_AUTH_TOKEN: token },
  },
})) {
  console.log(msg);
}
```

**Python:**

```python
import httpx
from codebuddy_agent_sdk import query, CodeBuddyAgentOptions

async def get_oauth_token(client_id: str, client_secret: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://copilot.tencent.com/oauth2/token",
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            },
        )
        return response.json()["access_token"]

token = await get_oauth_token("your-client-id", "your-client-secret")

options = CodeBuddyAgentOptions(
    env={"CODEBUDDY_AUTH_TOKEN": token}
)

async for msg in query(prompt="Hello", options=options):
    print(msg)
```

### 其他环境变量

| 变量名 | 说明 | 必需 |
|---|---|---|
| `CODEBUDDY_CODE_PATH` | CodeBuddy CLI 可执行文件路径 | 可选 |

如果未设置，SDK 会自动尝试查找 CLI。

---

## 基础用法

### 简单查询

最基础的用法是发送一个提示词并处理响应：

**TypeScript:**

```typescript
import { query } from '@tencent-ai/agent-sdk';

async function main() {
  const q = query({
    prompt: '请解释什么是递归函数',
    options: {
      permissionMode: 'bypassPermissions'
    }
  });

  for await (const message of q) {
    if (message.type === 'assistant') {
      for (const block of message.message.content) {
        if (block.type === 'text') {
          console.log(block.text);
        }
      }
    }
  }
}

main();
```

**Python:**

```python
import asyncio
from codebuddy_agent_sdk import query, CodeBuddyAgentOptions
from codebuddy_agent_sdk import AssistantMessage, TextBlock

async def main():
    options = CodeBuddyAgentOptions(
        permission_mode="bypassPermissions"
    )

    async for message in query(prompt="请解释什么是递归函数", options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(block.text)

asyncio.run(main())
```

### 提取结果

查询完成后，会收到一个 `result` 消息，包含执行统计信息：

**TypeScript:**

```typescript
for await (const message of q) {
  if (message.type === 'result') {
    if (message.subtype === 'success') {
      console.log('完成！耗时：', message.duration_ms, 'ms');
      console.log('费用：', message.total_cost_usd, 'USD');
    } else {
      console.log('执行出错');
    }
  }
}
```

**Python:**

```python
from codebuddy_agent_sdk import ResultMessage

async for message in query(prompt="...", options=options):
    if isinstance(message, ResultMessage):
        if message.subtype == "success":
            print(f"完成！耗时： {message.duration_ms} ms")
            print(f"费用： {message.total_cost_usd} USD")
        else:
            print("执行出错")
```

### 消息类型处理

SDK 返回多种类型的消息：

**TypeScript:**

```typescript
for await (const message of q) {
  switch (message.type) {
    case 'system':
      console.log('会话 ID:', message.session_id);
      console.log('可用工具：', message.tools);
      break;

    case 'assistant':
      for (const block of message.message.content) {
        if (block.type === 'text') {
          console.log('[文本]', block.text);
        } else if (block.type === 'tool_use') {
          console.log('[工具调用]', block.name, block.input);
        } else if (block.type === 'tool_result') {
          console.log('[工具结果]', block.content);
        }
      }
      break;

    case 'result':
      console.log('执行完成，耗时：', message.duration_ms, 'ms');
      break;
  }
}
```

**Python:**

```python
from codebuddy_agent_sdk import (
    SystemMessage, AssistantMessage, ResultMessage,
    TextBlock, ToolUseBlock, ToolResultBlock
)

async for message in query(prompt="...", options=options):
    if isinstance(message, SystemMessage):
        print(f"会话 ID: {message.data.get('session_id')}")
        print(f"可用工具： {message.data.get('tools')}")

    elif isinstance(message, AssistantMessage):
        for block in message.content:
            if isinstance(block, TextBlock):
                print(f"[文本] {block.text}")
            elif isinstance(block, ToolUseBlock):
                print(f"[工具调用] {block.name}: {block.input}")
            elif isinstance(block, ToolResultBlock):
                print(f"[工具结果] {block.content}")

    elif isinstance(message, ResultMessage):
        print(f"执行完成，耗时： {message.duration_ms} ms")
```

---

## 配置选项

### 权限模式

通过 `permissionMode` 控制工具调用的权限行为：

| 模式 | 说明 |
|---|---|
| `default` | 默认模式，所有操作需确认 |
| `acceptEdits` | 自动批准文件编辑，Bash 仍需确认 |
| `plan` | 规划模式，仅允许读取操作 |
| `bypassPermissions` | 跳过所有权限检查（谨慎使用） |

**TypeScript:**

```typescript
const q = query({
  prompt: '分析项目结构',
  options: {
    permissionMode: 'plan'  // 只读模式
  }
});
```

**Python:**

```python
options = CodeBuddyAgentOptions(
    permission_mode="plan"  # 只读模式
)
async for msg in query(prompt="分析项目结构", options=options):
    pass
```

### 工作目录

指定 Agent 的工作目录：

**TypeScript:**

```typescript
const q = query({
  prompt: '读取 package.json',
  options: {
    cwd: '/path/to/project'
  }
});
```

**Python:**

```python
options = CodeBuddyAgentOptions(
    cwd="/path/to/project"
)
```

### 模型选择

指定使用的 AI 模型：

**TypeScript:**

```typescript
const q = query({
  prompt: '...',
  options: {
    model: 'deepseek-v3.1',
    fallbackModel: 'deepseek-v3.1'
  }
});
```

**Python:**

```python
options = CodeBuddyAgentOptions(
    model="deepseek-v3.1",
    fallback_model="deepseek-v3.1"
)
```

### 资源限制

限制执行范围：

**TypeScript:**

```typescript
const q = query({
  prompt: '...',
  options: {
    maxTurns: 20         // 最大对话轮数
  }
});
```

**Python:**

```python
options = CodeBuddyAgentOptions(
    max_turns=20,        # 最大对话轮数
)
```

---

## 环境隔离（settingSources）

### 设计理念

SDK 默认**不加载任何文件系统配置**，提供完全干净的运行环境。这是与 CLI 直接使用的关键区别。

### 为什么这样设计？

1. **可预测性**：SDK 应用的行为完全由代码控制，不受用户或项目配置文件影响
2. **隔离性**：避免用户的个人偏好或项目设置干扰 SDK 应用的逻辑
3. **安全性**：敏感配置（如 hooks、权限规则）不会意外泄露到 SDK 环境
4. **一致性**：在不同机器上运行时，行为保持一致

### 默认行为对比

| 场景 | Settings | Memory | MCP | Subagent | Commands | Rules | Skills |
|---|---|---|---|---|---|---|---|
| SDK 调用（默认） | 不加载 | 不加载 | 不加载 | 不加载 | 不加载 | 不加载 | 不加载 |
| CLI 直接运行 | 加载全部 | 加载全部 | 加载全部 | 加载全部 | 加载全部 | 加载全部 | 加载全部 |

**配置文件位置参考**：

| 配置类型 | 用户级位置 | 项目级位置 | 说明 |
|---|---|---|---|
| Settings | `~/.codebuddy/settings.json` | `.codebuddy/settings.json` | 权限、hooks、环境变量等 |
| Memory | `~/.codebuddy/CODEBUDDY.md` | `CODEBUDDY.md` | 项目指令和上下文 |
| MCP | `~/.codebuddy/.mcp.json` | `.mcp.json` | MCP 服务器配置 |
| Subagent | `~/.codebuddy/agents/` | `.codebuddy/agents/` | 自定义子代理 |
| Commands | `~/.codebuddy/commands/` | `.codebuddy/commands/` | 自定义斜杠命令 |
| Rules | `~/.codebuddy/rules/` | `.codebuddy/rules/` | 模块化规则文件 |
| Skills | `~/.codebuddy/skills/` | `.codebuddy/skills/` | AI 自动调用的技能 |

### 显式加载配置

如需加载文件系统配置，使用 `settingSources` 显式指定：

**TypeScript:**

```typescript
const q = query({
  prompt: '...',
  options: {
    settingSources: ['project'],
    // 或加载全部配置
    // settingSources: ['user', 'project', 'local']
  }
});
```

**Python:**

```python
options = CodeBuddyAgentOptions(
    setting_sources=["project"],
    # 或加载全部配置
    # setting_sources=["user", "project", "local"]
)
```

### 配置源说明

| 值 | 说明 | 位置 |
|---|---|---|
| `'user'` | 全局用户设置 | `~/.codebuddy/settings.json`, `~/.codebuddy/CODEBUDDY.md` |
| `'project'` | 项目共享设置 | `.codebuddy/settings.json`, `CODEBUDDY.md` |
| `'local'` | 项目本地设置 | `.codebuddy/settings.local.json`, `CODEBUDDY.local.md` |

---

## 权限控制

### canUseTool 回调

通过 `canUseTool` 回调实现细粒度权限控制：

**TypeScript:**

```typescript
import { query } from '@tencent-ai/agent-sdk';

const q = query({
  prompt: '分析项目结构',
  options: {
    canUseTool: async (toolName, input, options) => {
      const readOnlyTools = ['Read', 'Glob', 'Grep'];

      if (readOnlyTools.includes(toolName)) {
        return {
          behavior: 'allow',
          updatedInput: input
        };
      }

      return {
        behavior: 'deny',
        message: `工具 ${toolName} 不允许使用`
      };
    }
  }
});
```

**Python:**

```python
from codebuddy_agent_sdk import (
    query, CodeBuddyAgentOptions,
    CanUseToolOptions, PermissionResultAllow, PermissionResultDeny
)

async def can_use_tool(
    tool_name: str,
    input_data: dict,
    options: CanUseToolOptions
):
    read_only_tools = ["Read", "Glob", "Grep"]

    if tool_name in read_only_tools:
        return PermissionResultAllow(updated_input=input_data)

    return PermissionResultDeny(
        message=f"工具 {tool_name} 不允许使用"
    )

options = CodeBuddyAgentOptions(can_use_tool=can_use_tool)
```

### 拦截危险操作

结合权限回调拦截危险命令：

**TypeScript:**

```typescript
const dangerousCommands = ['rm -rf', 'sudo', 'chmod 777'];

const q = query({
  prompt: '清理临时文件',
  options: {
    canUseTool: async (toolName, input) => {
      if (toolName === 'Bash') {
        const command = input.command as string;
        for (const dangerous of dangerousCommands) {
          if (command.includes(dangerous)) {
            return {
              behavior: 'deny',
              message: `危险命令被拦截: ${dangerous}`,
              interrupt: true  // 中断整个会话
            };
          }
        }
      }
      return { behavior: 'allow', updatedInput: input };
    }
  }
});
```

**Python:**

```python
dangerous_commands = ["rm -rf", "sudo", "chmod 777"]

async def can_use_tool(tool_name, input_data, options):
    if tool_name == "Bash":
        command = input_data.get("command", "")
        for dangerous in dangerous_commands:
            if dangerous in command:
                return PermissionResultDeny(
                    message=f"危险命令被拦截： {dangerous}",
                    interrupt=True
                )
    return PermissionResultAllow(updated_input=input_data)
```

---

## 多轮对话

### 使用 Session/Client API

对于需要多轮交互的场景，使用 Session（TypeScript）或 Client（Python）API：

**TypeScript:**

```typescript
import { unstable_v2_createSession } from '@tencent-ai/agent-sdk';

async function main() {
  const session = unstable_v2_createSession({
    model: 'deepseek-v3.1'
  });

  // 第一轮对话
  await session.send('分析这个项目的架构');
  for await (const message of session.stream()) {
    console.log(message);
  }

  // 第二轮对话（保持上下文）
  await session.send('请详细解释第三点');
  for await (const message of session.stream()) {
    console.log(message);
  }

  session.close();
}
```

**Python:**

```python
from codebuddy_agent_sdk import CodeBuddySDKClient, CodeBuddyAgentOptions

async def main():
    options = CodeBuddyAgentOptions(model="deepseek-v3.1")

    async with CodeBuddySDKClient(options=options) as client:
        await client.query("分析这个项目的架构")
        async for message in client.receive_response():
            print(message)

        await client.query("请详细解释第三点")
        async for message in client.receive_response():
            print(message)

asyncio.run(main())
```

### 中断执行

在运行过程中中断执行：

**TypeScript:**

```typescript
const q = query({ prompt: '执行长时间任务...' });

let count = 0;
for await (const message of q) {
  if (message.type === 'assistant') {
    for (const block of message.message.content) {
      if (block.type === 'tool_use') {
        count++;
        if (count >= 10) {
          await q.interrupt();
          break;
        }
      }
    }
  }
}
```

**Python:**

```python
async with CodeBuddySDKClient(options=options) as client:
    await client.query("执行长时间任务...")

    count = 0
    async for message in client.receive_messages():
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, ToolUseBlock):
                    count += 1
                    if count >= 10:
                        await client.interrupt()
                        break
```

---

## Hook 系统

Hook 允许在工具执行前后插入自定义逻辑。

### PreToolUse Hook

在工具执行前拦截和处理：

**TypeScript:**

```typescript
const q = query({
  prompt: '清理临时文件',
  options: {
    hooks: {
      PreToolUse: [{
        matcher: 'Bash',
        hooks: [
          async (input, toolUseId) => {
            console.log('即将执行命令：', input.command);

            if (input.command.includes('rm')) {
              return {
                decision: 'block',
                reason: '删除命令被阻止'
              };
            }

            return { continue: true };
          }
        ]
      }]
    }
  }
});
```

**Python:**

```python
from codebuddy_agent_sdk import HookMatcher, HookContext

async def pre_tool_hook(input_data, tool_use_id, context: HookContext):
    print(f"即将执行命令： {input_data.get('command')}")

    if "rm" in input_data.get("command", ""):
        return {"continue_": False, "reason": "删除命令被阻止"}

    return {"continue_": True}

options = CodeBuddyAgentOptions(
    hooks={
        "PreToolUse": [
            HookMatcher(matcher="Bash", hooks=[pre_tool_hook])
        ]
    }
)
```

### Hook 事件类型

| 事件 | 触发时机 |
|---|---|
| `PreToolUse` | 工具执行前 |
| `PostToolUse` | 工具执行成功后 |
| `PostToolUseFailure` | 工具执行失败后 |
| `UserPromptSubmit` | 用户提交提示词 |
| `SessionStart` | 会话开始 |
| `SessionEnd` | 会话结束 |
| `WorktreeCreate` | 创建隔离 worktree 时 |
| `WorktreeRemove` | 删除隔离 worktree 时 |

---

## 扩展能力

### 自定义 Agent

定义专门化的子 Agent：

**TypeScript:**

```typescript
const q = query({
  prompt: '使用 code-reviewer 审查代码',
  options: {
    agents: {
      'code-reviewer': {
        description: '专业代码审查助手',
        tools: ['Read', 'Glob', 'Grep'],
        disallowedTools: ['Bash', 'Write', 'Edit'],
        prompt: `你是代码审查专家，请检查：
1. 代码规范
2. 潜在 bug
3. 性能问题
4. 安全漏洞`,
        model: 'deepseek-v3.1'
      }
    }
  }
});
```

**Python:**

```python
from codebuddy_agent_sdk import AgentDefinition

options = CodeBuddyAgentOptions(
    agents={
        "code-reviewer": AgentDefinition(
            description="专业代码审查助手",
            tools=["Read", "Glob", "Grep"],
            disallowed_tools=["Bash", "Write", "Edit"],
            prompt="""你是代码审查专家，请检查：
1. 代码规范
2. 潜在 bug
3. 性能问题
4. 安全漏洞""",
            model="deepseek-v3.1"
        )
    }
)
```

### MCP 服务器配置

集成自定义 MCP 服务器：

**TypeScript:**

```typescript
const q = query({
  prompt: '查询数据库',
  options: {
    mcpServers: {
      'database': {
        type: 'stdio',
        command: 'node',
        args: ['./mcp-servers/db-server.js'],
        env: {
          DB_HOST: 'localhost',
          DB_PORT: '5432'
        }
      }
    }
  }
});
```

**Python:**

```python
options = CodeBuddyAgentOptions(
    mcp_servers={
        "database": {
            "type": "stdio",
            "command": "node",
            "args": ["./mcp-servers/db-server.js"],
            "env": {
                "DB_HOST": "localhost",
                "DB_PORT": "5432"
            }
        }
    }
)
```

### 处理 AskUserQuestion

AI 可能会通过 `AskUserQuestion` 工具向用户提问，可以在权限回调中处理：

**TypeScript:**

```typescript
const q = query({
  prompt: '配置数据库连接',
  options: {
    canUseTool: async (toolName, input) => {
      if (toolName === 'AskUserQuestion') {
        const questions = input.questions as any[];
        const answers: Record<string, string> = {};

        for (const q of questions) {
          console.log(`问题: ${q.question}`);
          answers[q.question] = q.options[0].label;
        }

        return {
          behavior: 'allow',
          updatedInput: { ...input, answers }
        };
      }
      return { behavior: 'allow', updatedInput: input };
    }
  }
});
```

**Python:**

```python
async def can_use_tool(tool_name, input_data, options):
    if tool_name == "AskUserQuestion":
        questions = input_data.get("questions", [])
        answers = {}

        for q in questions:
            print(f"问题： {q['question']}")
            answers[q["question"]] = q["options"][0]["label"]

        return PermissionResultAllow(
            updated_input={**input_data, "answers": answers}
        )

    return PermissionResultAllow(updated_input=input_data)
```

---

## 错误处理

**TypeScript:**

```typescript
import { query, AbortError } from '@tencent-ai/agent-sdk';

try {
  const q = query({ prompt: '...' });
  for await (const message of q) {
    // ...
  }
} catch (error) {
  if (error instanceof AbortError) {
    console.log('操作被中止');
  } else {
    console.error('发生错误：', error);
  }
}
```

**Python:**

```python
from codebuddy_agent_sdk import (
    query, CodeBuddySDKError,
    CLIConnectionError, CLINotFoundError
)

try:
    async for message in query(prompt="..."):
        pass
except CLINotFoundError as e:
    print(f"CLI 未找到： {e}")
except CLIConnectionError as e:
    print(f"连接失败： {e}")
except CodeBuddySDKError as e:
    print(f"SDK 错误： {e}")
```

---

## 最佳实践

1. **权限控制**：在生产环境中使用 `canUseTool` 实现细粒度权限，避免使用 `bypassPermissions`
2. **资源限制**：使用 `maxTurns` 限制执行范围，防止意外的资源消耗
3. **错误处理**：始终处理 `result` 消息中的错误状态
4. **Hook 超时**：为 Hook 设置合理的超时时间

---

## 相关文档

- [TypeScript SDK 参考](codebuddy-sdk-typescript.md)
- [Python SDK 参考](codebuddy-sdk-python.md)
- [SDK Hook 系统](codebuddy-sdk-hooks.md)
- [SDK 权限控制](codebuddy-sdk-permissions.md)
- [SDK 会话管理](codebuddy-sdk-sessions.md)
- [SDK 自定义工具](codebuddy-sdk-custom-tools.md)
- [SDK MCP 集成](codebuddy-sdk-mcp.md)
- [SDK 示例项目](codebuddy-sdk-demos.md)
