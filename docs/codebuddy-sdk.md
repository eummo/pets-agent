# CodeBuddy Agent SDK 快速开始

> **SDK 版本**：v0.3.157（2025-05-27）
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


---

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


---

# Python SDK 参考

> **版本要求**：本文档针对 CodeBuddy Agent SDK v0.1.0 及以上版本。

本文档提供 Python SDK 的完整 API 参考。有关快速入门和使用示例，请参阅 [SDK 概览](codebuddy-sdk-quickstart.md)。

## Requirements

| 依赖 | 版本要求 |
| --- | --- |
| Python | \>= 3.10 |
| CodeBuddy CLI | 已安装 |

**异步运行时**：SDK 基于 `asyncio`，所有 API 都是异步的。

## Installation

推荐使用 [uv](https://docs.astral.sh/uv/) 进行依赖管理：

```bash
uv add codebuddy-agent-sdk
```

或使用 pip：

```bash
pip install codebuddy-agent-sdk
```

### 环境变量

| 变量名 | 说明 | 必需 |
| --- | --- | --- |
| `CODEBUDDY_CODE_PATH` | CodeBuddy CLI 可执行文件路径 | 可选 |

如果未设置，SDK 会按以下顺序查找 CLI：

1. 环境变量 `CODEBUDDY_CODE_PATH`
2. SDK 包内置的二进制文件
3. 开发环境 monorepo 路径

### 认证配置

SDK 支持使用已有登录凭据、API Key 或 OAuth Client Credentials 认证，详见 [SDK 概览 - 认证配置](codebuddy-sdk-quickstart.md#认证配置)。

## Functions

### query()

主要 API 入口，创建一个查询并返回消息异步迭代器。

```python
async def query(
    *,
    prompt: str | AsyncIterable[dict[str, Any]],
    options: CodeBuddyAgentOptions | None = None,
    transport: Transport | None = None,
) -> AsyncIterator[Message]:
```

**参数**：

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `prompt` | `str \| AsyncIterable[dict]` | 查询提示词或用户消息流 |
| `options` | `CodeBuddyAgentOptions` | 配置选项（可选） |
| `transport` | `Transport` | 自定义传输层（可选） |

**返回值**：`AsyncIterator[Message]` - 消息异步迭代器

**示例**：

```python
from codebuddy_agent_sdk import query, AssistantMessage, TextBlock

async for message in query(prompt="What is 2+2?"):
    if isinstance(message, AssistantMessage):
        for block in message.content:
            if isinstance(block, TextBlock):
                print(block.text)
```

## Client Class

### CodeBuddySDKClient

用于双向交互式对话的客户端类。支持多轮对话、中断和动态控制。

```python
class CodeBuddySDKClient:
    def __init__(
        self,
        options: CodeBuddyAgentOptions | None = None,
        transport: Transport | None = None,
    ): ...
```

**方法**：

#### connect()

连接到 CodeBuddy。

```python
async def connect(
    self,
    prompt: str | AsyncIterable[dict[str, Any]] | None = None
) -> None:
```

#### query()

发送用户消息。

```python
async def query(
    self,
    prompt: str | AsyncIterable[dict[str, Any]],
    session_id: str = "default",
) -> None:
```

#### receive_response()

接收消息直到收到 ResultMessage。

```python
async def receive_response(self) -> AsyncIterator[Message]:
```

#### receive_messages()

接收所有消息（不会自动停止）。

```python
async def receive_messages(self) -> AsyncIterator[Message]:
```

#### disconnect()

断开连接。

```python
async def disconnect(self) -> None:
```

**上下文管理器支持**：

```python
async with CodeBuddySDKClient() as client:
    await client.query("Hello!")
    async for msg in client.receive_response():
        print(msg)
```

#### mcp_server_status()

获取 MCP 服务器连接状态。

```python
async def mcp_server_status(self) -> list[McpServerStatus]:
```

## Authentication

SDK 提供独立的认证 API，采用 **two-phase** 设计：先获取登录 URL，再等待用户完成认证。

### authenticate()

启动认证流程，返回 `AuthFlow` 对象。

```python
async def authenticate(
    *,
    method_id: str = "external",
    environment: str | None = None,
    endpoint: str | None = None,
    codebuddy_code_path: str | None = None,
    env: dict[str, str] | None = None,
    timeout: float = 300.0,
) -> AuthFlow:
```

**参数**：

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `method_id` | `str` | 认证方法标识（默认 `"external"`） |
| `environment` | `str \| None` | 预定义环境名 |
| `endpoint` | `str \| None` | 自定义端点 URL（与 environment 互斥） |
| `codebuddy_code_path` | `str \| None` | CLI 可执行文件路径 |
| `env` | `dict[str, str] \| None` | 额外环境变量 |
| `timeout` | `float` | 用户完成登录的超时时间（秒，默认 300） |

**返回值**：`AuthFlow` — 携带登录 URL 的可等待对象

**示例**：

```python
from codebuddy_agent_sdk import authenticate

# Two-phase: 获取 URL → 展示给用户 → 等待完成
auth = await authenticate()
if auth.auth_url:
    print(f"请访问: {auth.auth_url}")
result = await auth
print(f"欢迎, {result.userinfo.user_name}")

# 已登录时 auth.auth_url 为空，await 立即返回
auth = await authenticate()
result = await auth  # 已登录则立即返回

# 自定义超时
auth = await authenticate()
result = await auth.wait(timeout=60)
```

### AuthFlow

认证流程对象，由 `authenticate()` 返回。实现了 `__await__` 协议，可直接 `await`。

**属性**：

| 属性 | 类型 | 说明 |
| --- | --- | --- |
| `auth_url` | `str` | 登录 URL（已登录时为空字符串） |
| `method_id` | `str \| None` | 认证方法标识 |

**方法**：

#### wait()

等待用户完成认证。

```python
async def wait(self, timeout: float | None = None) -> AuthenticateResponse:
```

#### cancel()

取消认证流程并释放资源。

```python
async def cancel(self) -> None:
```

### logout()

登出并清除缓存的认证令牌。

```python
async def logout(
    *,
    environment: str | None = None,
    endpoint: str | None = None,
    codebuddy_code_path: str | None = None,
    env: dict[str, str] | None = None,
) -> None:
```

## Unstable API

> **警告**：以下 API 处于实验阶段，接口可能在未来版本中变更。

### interrupt()

发送中断信号。

```python
async def interrupt(self) -> None:
```

### set_permission_mode()

动态修改权限模式。

```python
async def set_permission_mode(self, mode: str) -> None:
```

### set_model()

动态修改模型。

```python
async def set_model(self, model: str | None) -> None:
```

## Types

### CodeBuddyAgentOptions

完整配置选项：

```python
@dataclass
class CodeBuddyAgentOptions:
    allowed_tools: list[str] = field(default_factory=list)
    disallowed_tools: list[str] = field(default_factory=list)
    system_prompt: str | AppendSystemPrompt | None = None
    mcp_servers: dict[str, McpServerConfig] | str | Path = field(default_factory=dict)
    permission_mode: PermissionMode | None = None
    continue_conversation: bool = False
    resume: str | None = None
    max_turns: int | None = None
    model: str | None = None
    fallback_model: str | None = None
    cwd: str | Path | None = None
    codebuddy_code_path: str | Path | None = None
    env: dict[str, str] = field(default_factory=dict)
    extra_args: dict[str, str | None] = field(default_factory=dict)
    stderr: Callable[[str], None] | None = None
    hooks: dict[HookEvent, list[HookMatcher]] | None = None
    include_partial_messages: bool = False
    fork_session: bool = False
    agents: dict[str, AgentDefinition] | None = None
    setting_sources: list[SettingSource] | None = None
    can_use_tool: CanUseTool | None = None
```

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `allowed_tools` | `list[str]` | 自动允许的工具白名单 |
| `disallowed_tools` | `list[str]` | 禁止使用的工具黑名单 |
| `system_prompt` | `str \| AppendSystemPrompt` | 系统提示词配置 |
| `mcp_servers` | `dict[str, McpServerConfig]` | MCP 服务器配置 |
| `permission_mode` | `PermissionMode` | 权限模式 |
| `continue_conversation` | `bool` | 继续最近的会话 |
| `resume` | `str` | 要恢复的会话 ID |
| `max_turns` | `int` | 最大对话轮数 |
| `model` | `str` | 指定模型 |
| `fallback_model` | `str` | 备用模型 |
| `cwd` | `str \| Path` | 工作目录 |
| `codebuddy_code_path` | `str \| Path` | CLI 可执行文件路径 |
| `env` | `dict[str, str]` | 环境变量 |
| `extra_args` | `dict[str, str \| None]` | 额外的 CLI 参数 |
| `stderr` | `Callable[[str], None]` | stderr 回调 |
| `hooks` | `dict[HookEvent, list[HookMatcher]]` | Hook 配置 |
| `include_partial_messages` | `bool` | 包含部分消息 |
| `fork_session` | `bool` | 分叉会话 |
| `agents` | `dict[str, AgentDefinition]` | 自定义 Agent |
| `setting_sources` | `list[SettingSource]` | 设置来源 |
| `can_use_tool` | `CanUseTool` | 权限回调函数 |
| `thinking` | `ThinkingConfig` | 思考模式配置 |
| `effort` | `'low' \| 'medium' \| 'high' \| 'xhigh'` | 模型推理努力程度 |

### PermissionMode

```python
PermissionMode = Literal["default", "acceptEdits", "plan", "bypassPermissions"]
```

| 值 | 说明 |
| --- | --- |
| `"default"` | 默认模式，所有操作需确认 |
| `"acceptEdits"` | 自动批准文件编辑 |
| `"plan"` | 规划模式，仅允许读取 |
| `"bypassPermissions"` | 跳过所有权限检查 |

### PermissionResult

```python
PermissionResult = PermissionResultAllow | PermissionResultDeny

@dataclass
class PermissionResultAllow:
    updated_input: dict[str, Any]
    behavior: Literal["allow"] = "allow"
    updated_permissions: list[dict[str, Any]] | None = None

@dataclass
class PermissionResultDeny:
    message: str
    behavior: Literal["deny"] = "deny"
    interrupt: bool = False
```

### CanUseTool

```python
CanUseTool = Callable[
    [str, dict[str, Any], CanUseToolOptions],
    Awaitable[PermissionResult],
]

@dataclass
class CanUseToolOptions:
    tool_use_id: str
    signal: Any | None = None
    agent_id: str | None = None
    suggestions: list[dict[str, Any]] | None = None
    blocked_path: str | None = None
    decision_reason: str | None = None
```

### AgentDefinition

```python
@dataclass
class AgentDefinition:
    description: str        # Agent 描述
    prompt: str             # 系统提示词
    tools: list[str] | None = None           # 允许的工具
    disallowed_tools: list[str] | None = None  # 禁止的工具
    model: str | None = None                 # 使用的模型
```

### McpServerConfig

```python
class McpStdioServerConfig(TypedDict):
    type: NotRequired[Literal["stdio"]]
    command: str
    args: NotRequired[list[str]]
    env: NotRequired[dict[str, str]]

McpServerConfig = McpStdioServerConfig
```

### HookEvent

```python
HookEvent = (
    Literal["PreToolUse"]
    | Literal["PostToolUse"]
    | Literal["UserPromptSubmit"]
    | Literal["Stop"]
    | Literal["SubagentStop"]
    | Literal["PreCompact"]
    | Literal["WorktreeCreate"]
    | Literal["WorktreeRemove"]
)
```

### HookMatcher

```python
@dataclass
class HookMatcher:
    matcher: str | None = None      # 匹配模式（支持正则）
    hooks: list[HookCallback] = field(default_factory=list)
    timeout: float | None = None    # 超时时间（秒）
```

### SettingSource

```python
SettingSource = Literal["user", "project", "local"]
```

| 值 | 说明 | 位置 |
| --- | --- | --- |
| `"user"` | 全局用户设置 | `~/.codebuddy/settings.json` |
| `"project"` | 项目共享设置 | `.codebuddy/settings.json` |
| `"local"` | 项目本地设置 | `.codebuddy/settings.local.json` |

**默认行为**：当 `setting_sources` 未指定时，SDK **不加载任何文件系统配置**。

## Message Types

### Message

所有消息类型的联合：

```python
Message = UserMessage | AssistantMessage | SystemMessage | ResultMessage | StreamEvent
```

### SystemMessage

```python
@dataclass
class SystemMessage:
    subtype: str
    data: dict[str, Any]
```

### UserMessage

```python
@dataclass
class UserMessage:
    content: str | list[ContentBlock]
    uuid: str | None = None
    parent_tool_use_id: str | None = None
```

### AssistantMessage

```python
@dataclass
class AssistantMessage:
    content: list[ContentBlock]
    model: str
    parent_tool_use_id: str | None = None
    error: str | None = None
```

### ResultMessage

```python
@dataclass
class ResultMessage:
    subtype: str
    duration_ms: int
    duration_api_ms: int
    is_error: bool
    num_turns: int
    session_id: str
    total_cost_usd: float | None = None
    usage: dict[str, Any] | None = None
    result: str | None = None
```

### StreamEvent

```python
@dataclass
class StreamEvent:
    uuid: str
    session_id: str
    event: dict[str, Any]
    parent_tool_use_id: str | None = None
```

### ContentBlock

```python
ContentBlock = TextBlock | ThinkingBlock | ToolUseBlock | ToolResultBlock

@dataclass
class TextBlock:
    text: str

@dataclass
class ThinkingBlock:
    thinking: str
    signature: str

@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]

@dataclass
class ToolResultBlock:
    tool_use_id: str
    content: str | list[dict[str, Any]] | None = None
    is_error: bool | None = None
```

## Errors

所有异常都继承自 `CodeBuddySDKError`。

### CodeBuddySDKError

```python
class CodeBuddySDKError(Exception):
    """Base exception for CodeBuddy SDK errors."""
    pass
```

### CLIConnectionError

当连接到 CLI 失败或未建立连接时抛出。

```python
class CLIConnectionError(CodeBuddySDKError):
    pass
```

### CLINotFoundError

当找不到 CLI 可执行文件时抛出。

```python
class CLINotFoundError(CodeBuddySDKError):
    def __init__(
        self,
        message: str,
        platform: str | None = None,
        arch: str | None = None,
    ): ...
```

### CLIJSONDecodeError

当 CLI 输出的 JSON 解码失败时抛出。

```python
class CLIJSONDecodeError(CodeBuddySDKError):
    pass
```

### ProcessError

当 CLI 进程遇到错误时抛出。

```python
class ProcessError(CodeBuddySDKError):
    pass
```

### CLIStartupError

当 CLI 进程在启动阶段崩溃或未产生任何输出时抛出。

```python
class CLIStartupError(CodeBuddySDKError):
    def __init__(
        self,
        message: str,
        stderr: str = "",
        exit_code: int | None = None,
    ): ...
```

### ExecutionError

当执行失败时抛出（如认证错误、API 错误）。

```python
class ExecutionError(CodeBuddySDKError):
    def __init__(self, errors: list[str], subtype: str): ...
```

### AuthenticationError

当认证失败时抛出。

```python
class AuthenticationError(CodeBuddySDKError):
    def __init__(self, error_type: str, message: str): ...
```

## Auth Types

### AuthenticateResponse

```python
@dataclass(slots=True)
class AuthenticateResponse:
    userinfo: UserInfo
```

### UserInfo

```python
@dataclass(slots=True)
class UserInfo:
    user_id: str
    user_name: str = ""
    user_nickname: str = ""
    token: str = ""
    enterprise_id: str | None = None
    enterprise: str | None = None
```

### McpServerStatus

```python
@dataclass(slots=True)
class McpServerStatus:
    name: str
    status: Literal["connected", "failed", "needs-auth", "pending"]
    server_info: dict[str, Any] | None = None
```

## 相关文档

- [SDK 概览](codebuddy-sdk-quickstart.md) - 快速入门和使用示例
- [TypeScript SDK 参考](codebuddy-sdk-typescript.md) - TypeScript 版本 API
- [Hook 参考指南](codebuddy-sdk-hooks.md) - 详细的 Hook 配置说明
- [MCP 集成](codebuddy-sdk-mcp.md) - MCP 服务器配置指南


---

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


---

# SDK 权限控制

> **版本要求**：本文档针对 CodeBuddy Agent SDK v0.1.0 及以上版本。

本文档介绍如何在 SDK 中实现权限控制，包括权限模式、canUseTool 回调和工具白名单/黑名单。

## 概述

CodeBuddy Agent SDK 提供多种权限控制机制：

| 机制 | 说明 | 适用场景 |
|------|------|----------|
| **权限模式** | 全局控制权限行为 | 快速设置整体策略 |
| **canUseTool 回调** | 运行时动态审批 | 交互式权限确认 |
| **工具白名单/黑名单** | 声明式工具过滤 | 静态策略配置 |

## 权限模式

通过 `permissionMode`（TypeScript）或 `permission_mode`（Python）设置全局权限行为。

### 可用模式

| 模式 | 说明 |
|------|------|
| `default` | 默认模式，所有工具操作需要确认 |
| `acceptEdits` | 自动批准文件编辑，其他操作仍需确认 |
| `plan` | 规划模式，仅允许只读工具 |
| `bypassPermissions` | 跳过所有权限检查（谨慎使用） |

### 初始配置

**TypeScript**

```typescript
import { query } from '@tencent-ai/agent-sdk';

const q = query({
  prompt: '帮我重构这段代码',
  options: {
    model: 'deepseek-v3.1',
    permissionMode: 'acceptEdits'
  }
});
```

**Python**

```python
import asyncio
from codebuddy_agent_sdk import query, CodeBuddyAgentOptions

async def main():
    options = CodeBuddyAgentOptions(
        model="deepseek-v3.1",
        permission_mode="acceptEdits"
    )

    async for message in query(prompt="帮我重构这段代码", options=options):
        print(message)

asyncio.run(main())
```

### 动态修改权限模式

使用 Session/Client API 可以在运行时动态修改权限模式：

**Python**

```python
from codebuddy_agent_sdk import CodeBuddySDKClient, CodeBuddyAgentOptions

async def main():
    options = CodeBuddyAgentOptions(model="deepseek-v3.1")

    async with CodeBuddySDKClient(options=options) as client:
        await client.query("分析这个项目")
        async for msg in client.receive_response():
            print(msg)

        # 动态切换权限模式
        await client.set_permission_mode("acceptEdits")

        await client.query("现在帮我修改代码")
        async for msg in client.receive_response():
            print(msg)
```

## canUseTool 回调

`canUseTool` 回调在工具需要权限确认时触发，允许你实现自定义的权限逻辑。

### 回调签名

**TypeScript**

```typescript
type CanUseTool = (
  toolName: string,
  input: Record<string, unknown>,
  options: CanUseToolOptions
) => Promise<PermissionResult>;

type CanUseToolOptions = {
  signal: AbortSignal;
  toolUseID: string;
  agentID?: string;
  suggestions?: PermissionUpdate[];
  blockedPath?: string;
  decisionReason?: string;
};

type PermissionResult =
  | { behavior: 'allow'; updatedInput: Record<string, unknown> }
  | { behavior: 'deny'; message: string; interrupt?: boolean };
```

**Python**

```python
CanUseTool = Callable[
    [str, dict[str, Any], CanUseToolOptions],
    Awaitable[PermissionResult],
]

@dataclass
class CanUseToolOptions:
    tool_use_id: str
    signal: Any | None = None
    agent_id: str | None = None
    suggestions: list[dict[str, Any]] | None = None
    blocked_path: str | None = None
    decision_reason: str | None = None

PermissionResult = PermissionResultAllow | PermissionResultDeny
```

### 完整示例：交互式审批

**TypeScript**

```typescript
import { query } from '@tencent-ai/agent-sdk';

const q = query({
  prompt: '帮我分析这个代码库',
  options: {
    model: 'deepseek-v3.1',
    canUseTool: async (toolName, input, options) => {
      console.log(`\n工具请求: ${toolName}`);
      console.log(`   参数:`, JSON.stringify(input, null, 2));

      // 只读工具自动允许
      const readOnlyTools = ['Read', 'Glob', 'Grep'];
      if (readOnlyTools.includes(toolName)) {
        return { behavior: 'allow', updatedInput: input };
      }

      // 危险命令拒绝
      if (toolName === 'Bash') {
        const command = input.command as string;
        if (command.includes('rm -rf') || command.includes('sudo')) {
          return {
            behavior: 'deny',
            message: '危险命令被拒绝',
            interrupt: true
          };
        }
      }

      // 其他情况：模拟用户确认
      const approved = await promptUser(`允许执行 ${toolName}?`);

      if (approved) {
        return { behavior: 'allow', updatedInput: input };
      } else {
        return { behavior: 'deny', message: '用户拒绝' };
      }
    }
  }
});
```

**Python**

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
    print(f"\n工具请求： {tool_name}")
    print(f"   参数： {input_data}")

    # 只读工具自动允许
    read_only_tools = ["Read", "Glob", "Grep"]
    if tool_name in read_only_tools:
        return PermissionResultAllow(updated_input=input_data)

    # 危险命令拒绝
    if tool_name == "Bash":
        command = input_data.get("command", "")
        if "rm -rf" in command or "sudo" in command:
            return PermissionResultDeny(
                message="危险命令被拒绝",
                interrupt=True
            )

    # 其他情况：模拟用户确认
    answer = input(f"允许执行 {tool_name}? (y/n): ")

    if answer.lower() == 'y':
        return PermissionResultAllow(updated_input=input_data)
    else:
        return PermissionResultDeny(message="用户拒绝")

async def main():
    options = CodeBuddyAgentOptions(
        model="deepseek-v3.1",
        can_use_tool=can_use_tool
    )

    async for message in query(prompt="帮我分析这个代码库", options=options):
        print(message)
```

### 修改工具输入

可以在 `canUseTool` 中修改工具的输入参数：

**TypeScript**

```typescript
canUseTool: async (toolName, input) => {
  if (toolName === 'Bash') {
    return {
      behavior: 'allow',
      updatedInput: {
        ...input,
        command: `set -e; ${input.command}`
      }
    };
  }
  return { behavior: 'allow', updatedInput: input };
}
```

**Python**

```python
async def can_use_tool(tool_name, input_data, options):
    if tool_name == "Bash":
        return PermissionResultAllow(
            updated_input={
                **input_data,
                "command": f"set -e; {input_data.get('command', '')}"
            }
        )
    return PermissionResultAllow(updated_input=input_data)
```

## 处理 AskUserQuestion

当 AI 需要向用户提问时，会调用 `AskUserQuestion` 工具。你需要在 `canUseTool` 中处理这个工具。

### 输入结构

```typescript
{
  questions: [
    {
      question: "使用哪个数据库？",
      header: "数据库",
      options: [
        { label: "PostgreSQL", description: "关系型数据库" },
        { label: "MongoDB", description: "文档数据库" }
      ],
      multiSelect: false
    }
  ]
}
```

### 返回答案

**TypeScript**

```typescript
canUseTool: async (toolName, input) => {
  if (toolName === 'AskUserQuestion') {
    const questions = input.questions as any[];
    const answers: Record<string, string> = {};

    for (const q of questions) {
      console.log(`问题: ${q.question}`);
      for (let i = 0; i < q.options.length; i++) {
        console.log(`  ${i + 1}. ${q.options[i].label}`);
      }

      const choice = await getUserChoice();
      answers[q.question] = q.options[choice].label;
    }

    return {
      behavior: 'allow',
      updatedInput: { ...input, answers }
    };
  }

  return { behavior: 'allow', updatedInput: input };
}
```

**Python**

```python
async def can_use_tool(tool_name, input_data, options):
    if tool_name == "AskUserQuestion":
        questions = input_data.get("questions", [])
        answers = {}

        for q in questions:
            print(f"问题： {q['question']}")
            for i, opt in enumerate(q["options"]):
                print(f"  {i + 1}. {opt['label']}")

            choice = int(input("选择 （1/2/...): ")) - 1
            answers[q["question"]] = q["options"][choice]["label"]

        return PermissionResultAllow(
            updated_input={**input_data, "answers": answers}
        )

    return PermissionResultAllow(updated_input=input_data)
```

## 工具白名单/黑名单

SDK 提供多种工具过滤机制：

| 选项 | 说明 | 优先级 |
|------|------|--------|
| `tools` | 内置工具白名单，从根本上限制可用工具集 | 最高 |
| `allowedTools` | 允许使用的工具（支持模式匹配） | 中 |
| `disallowedTools` | 禁止使用的工具（支持模式匹配） | 中 |

### tools：内置工具白名单

使用 `tools` 选项从根本上限制 CodeBuddy 可使用的内置工具集：

**TypeScript**

```typescript
const q = query({
  prompt: '分析项目结构',
  options: {
    model: 'deepseek-v3.1',
    tools: ['Read', 'Glob', 'Grep']
  }
});

// 禁用所有内置工具（仅使用 MCP 工具）
const q2 = query({
  prompt: '使用 MCP 工具完成任务',
  options: {
    tools: []
  }
});
```

**Python**

```python
options = CodeBuddyAgentOptions(
    model="deepseek-v3.1",
    tools=["Read", "Glob", "Grep"]
)

# 禁用所有内置工具（仅使用 MCP 工具）
options2 = CodeBuddyAgentOptions(
    model="deepseek-v3.1",
    tools=[]
)
```

### allowedTools/disallowedTools：工具过滤

**TypeScript**

```typescript
const q = query({
  prompt: '分析项目结构',
  options: {
    model: 'deepseek-v3.1',
    allowedTools: ['Read', 'Glob', 'Grep'],
    disallowedTools: ['Bash', 'Write', 'Edit']
  }
});
```

**Python**

```python
options = CodeBuddyAgentOptions(
    model="deepseek-v3.1",
    allowed_tools=["Read", "Glob", "Grep"],
    disallowed_tools=["Bash", "Write", "Edit"]
)
```

### 常用工具名称

| 工具名 | 功能 |
|--------|------|
| `Read` | 读取文件 |
| `Write` | 写入文件 |
| `Edit` | 编辑文件 |
| `Glob` | 文件模式匹配 |
| `Grep` | 内容搜索 |
| `Bash` | 执行 Shell 命令 |
| `Task` | 子 Agent 任务 |
| `WebFetch` | 获取网页内容 |
| `WebSearch` | 网络搜索 |
| `ToolSearch` | 搜索延迟加载的工具 |

## 最佳实践

1. **默认使用 `default` 模式**：提供最完整的权限控制

2. **只读任务使用 `plan` 模式**：

```typescript
permissionMode: 'plan'  // 只允许 Read、Glob、Grep
```

3. **结合白名单精确控制**：

```typescript
allowedTools: ['Read', 'Glob', 'Grep'],
permissionMode: 'bypassPermissions'  // 允许的工具自动执行
```

4. **危险命令使用 `interrupt`**：

```typescript
return {
  behavior: 'deny',
  message: '危险操作',
  interrupt: true  // 立即中断，不让 AI 继续尝试
};
```

5. **生产环境避免 `bypassPermissions`**：该模式会跳过所有权限检查

## 相关文档

- [SDK 概览](codebuddy-sdk-quickstart.md) - 快速入门和使用示例
- [SDK Hook 系统](codebuddy-sdk-hooks.md) - 更细粒度的工具控制
- [TypeScript SDK 参考](codebuddy-sdk-typescript.md) - 完整 API 参考
- [Python SDK 参考](codebuddy-sdk-python.md) - 完整 API 参考


---

# SDK Hook 系统

> **版本要求**：本文档针对 CodeBuddy Agent SDK v0.1.0 及以上版本。

本文档介绍如何在 SDK 中使用 Hook 系统，在工具执行前后插入自定义逻辑。

## 概述

Hook 允许你在 CodeBuddy 的会话生命周期内插入自定义逻辑，实现：

- 工具调用前的校验和拦截
- 工具执行后的日志记录
- 用户提交内容的审查
- 会话开始/结束时的初始化和清理
- worktree 创建与清理时的自定义流程

### 支持的事件

| 事件 | 触发时机 |
|---|---|
| `PreToolUse` | 工具执行前 |
| `PostToolUse` | 工具执行成功后 |
| `UserPromptSubmit` | 用户提交消息时 |
| `Stop` | 主 Agent 响应结束时 |
| `SubagentStop` | 子 Agent 结束时 |
| `PreCompact` | 上下文压缩前 |
| `WorktreeCreate` | 创建隔离 worktree 时 |
| `WorktreeRemove` | 删除隔离 worktree 时 |
| `unstable_Checkpoint` | 文件修改后自动创建检查点时 |

## Hook 配置

通过 `hooks` 选项配置 Hook。每个事件可以有多个 matcher，每个 matcher 可以有多个 hook 回调。

### 基本结构

**TypeScript**

```typescript
import { query } from '@tencent-ai/agent-sdk';

const q = query({
  prompt: '帮我分析代码',
  options: {
    model: 'deepseek-v3.1',
    hooks: {
      PreToolUse: [
        {
          matcher: 'Bash',
          hooks: [
            async (input, toolUseId, ctx) => {
              console.log('即将执行：', input);
              return { continue: true };
            }
          ],
          timeout: 5000
        }
      ]
    }
  }
});
```

**Python**

```python
from codebuddy_agent_sdk import query, CodeBuddyAgentOptions, HookMatcher

async def pre_tool_hook(input_data, tool_use_id, context):
    print(f"即将执行： {input_data}")
    return {"continue_": True}

options = CodeBuddyAgentOptions(
    model="deepseek-v3.1",
    hooks={
        "PreToolUse": [
            HookMatcher(
                matcher="Bash",
                hooks=[pre_tool_hook],
                timeout=5.0
            )
        ]
    }
)

async for msg in query(prompt="帮我分析代码", options=options):
    print(msg)
```

### HookMatcher 结构

| 字段 | 类型 | 说明 |
|---|---|---|
| `matcher` | `string` | 匹配模式,支持正则表达式。`*` 或空字符串匹配所有 |
| `hooks` | `HookCallback[]` | 回调函数数组 |
| `timeout` | `number` | 超时时间（TypeScript 毫秒，Python 秒） |

### Matcher 模式

- **精确匹配**：`"Bash"` 只匹配 Bash 工具
- **正则匹配**：`"Edit|Write"` 匹配 Edit 或 Write
- **通配符**：`"*"` 或 `""` 匹配所有工具
- **前缀匹配**：`"mcp__.*"` 匹配所有 MCP 工具

## 事件类型

### PreToolUse

工具执行前触发，可以阻止执行或修改输入。

**TypeScript**

```typescript
hooks: {
  PreToolUse: [{
    matcher: 'Bash',
    hooks: [
      async (input, toolUseId, ctx) => {
        const command = input.command as string;

        if (command.includes('rm -rf')) {
          return {
            decision: 'block',
            reason: '危险命令被阻止'
          };
        }

        return { continue: true };
      }
    ]
  }]
}
```

**Python**

```python
async def pre_bash_hook(input_data, tool_use_id, context):
    command = input_data.get("command", "")

    if "rm -rf" in command:
        return {
            "decision": "block",
            "reason": "危险命令被阻止"
        }

    return {"continue_": True}

hooks = {
    "PreToolUse": [
        HookMatcher(matcher="Bash", hooks=[pre_bash_hook])
    ]
}
```

### PostToolUse

工具执行成功后触发，可以添加额外上下文。

**TypeScript**

```typescript
hooks: {
  PostToolUse: [{
    matcher: 'Write|Edit',
    hooks: [
      async (input, toolUseId) => {
        console.log(`文件已修改: ${input.file_path}`);
        await logFileChange(input.file_path);
        return { continue: true };
      }
    ]
  }]
}
```

**Python**

```python
async def post_write_hook(input_data, tool_use_id, context):
    print(f"文件已修改： {input_data.get('file_path')}")
    await log_file_change(input_data.get("file_path"))
    return {"continue_": True}

hooks = {
    "PostToolUse": [
        HookMatcher(matcher="Write|Edit", hooks=[post_write_hook])
    ]
}
```

### UserPromptSubmit

用户提交消息时触发，可以添加上下文或阻止处理。

**TypeScript**

```typescript
hooks: {
  UserPromptSubmit: [{
    hooks: [
      async (input) => {
        const prompt = input.prompt as string;

        if (containsSensitiveWords(prompt)) {
          return {
            decision: 'block',
            reason: '消息包含敏感内容'
          };
        }

        return { continue: true };
      }
    ]
  }]
}
```

### Stop / SubagentStop

Agent 响应结束时触发，可以阻止停止并要求继续。

**TypeScript**

```typescript
hooks: {
  Stop: [{
    hooks: [
      async (input) => {
        if (!isTaskComplete()) {
          return {
            decision: 'block',
            reason: '任务未完成，请继续'
          };
        }
        return { continue: true };
      }
    ]
  }]
}
```

### unstable_Checkpoint（实验性）

文件修改后（Write/Edit/MultiEdit 工具执行成功）自动触发，提供文件快照和变更统计信息。

**TypeScript**

```typescript
import type { CheckpointHookInput } from '@tencent-ai/agent-sdk';

hooks: {
  unstable_Checkpoint: [{
    hooks: [
      async (input) => {
        const checkpointInput = input as CheckpointHookInput;
        const checkpoint = checkpointInput.checkpoint;

        console.log('文件变更检查点：', {
          id: checkpoint.id,
          label: checkpoint.label,
          files: checkpoint.fileChangeStats?.files,
          additions: checkpoint.fileChangeStats?.additions,
          deletions: checkpoint.fileChangeStats?.deletions
        });

        return { continue: true };
      }
    ]
  }]
}
```

**Checkpoint 数据结构**：

- `id`: 检查点唯一标识符
- `label`: 人类可读标签
- `createdAt`: 创建时间戳
- `fileSnapshots`: 文件路径到版本信息的映射
- `fileChangeStats`: 文件变更统计（files、additions、deletions）

## Hook 输入

Hook 回调接收的输入结构因事件类型而异。

### 公共字段

```json
{
  "session_id": "abc123",
  "cwd": "/path/to/project",
  "permission_mode": "default",
  "hook_event_name": "PreToolUse"
}
```

### PreToolUse / PostToolUse 输入

```json
{
  "tool_name": "Bash",
  "tool_input": {
    "command": "ls -la"
  }
}
```

### UserPromptSubmit 输入

```json
{
  "prompt": "帮我写一个函数"
}
```

### WorktreeCreate 输入

```json
{
  "hook_event_name": "WorktreeCreate",
  "session_id": "abc123",
  "cwd": "/path/to/project",
  "transcript_path": "/path/to/transcript.jsonl",
  "name": "feature-auth"
}
```

### WorktreeRemove 输入

```json
{
  "hook_event_name": "WorktreeRemove",
  "session_id": "abc123",
  "cwd": "/path/to/project",
  "transcript_path": "/path/to/transcript.jsonl",
  "worktree_path": "/tmp/codebuddy-worktrees/feature-auth"
}
```

## Hook 输出

Hook 回调返回的输出控制后续行为。

### 基本输出字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `continue` / `continue_` | `boolean` | 是否继续执行（默认 true） |
| `decision` | `'block'` | 设为 `'block'` 阻止操作 |
| `reason` | `string` | 阻止原因 |
| `stopReason` | `string` | 当 `continue` 为 false 时显示的停止消息 |
| `suppressOutput` | `boolean` | 隐藏输出 |

### PreToolUse 特殊输出

可以修改工具输入：

**TypeScript**

```typescript
return {
  continue: true,
  hookSpecificOutput: {
    hookEventName: 'PreToolUse',
    updatedInput: {
      command: `echo "安全检查通过" && ${input.command}`
    }
  }
};
```

**Python**

```python
return {
    "continue_": True,
    "hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "updatedInput": {
            "command": f'echo "安全检查通过" && {input_data["command"]}'
        }
    }
}
```

## 完整示例：Bash 命令审计

**TypeScript**

```typescript
import { query } from '@tencent-ai/agent-sdk';
import * as fs from 'fs';

const logFile = '/tmp/bash-audit.log';

const q = query({
  prompt: '帮我清理临时文件',
  options: {
    model: 'deepseek-v3.1',
    hooks: {
      PreToolUse: [{
        matcher: 'Bash',
        hooks: [
          async (input, toolUseId) => {
            const command = input.command as string;
            const timestamp = new Date().toISOString();

            fs.appendFileSync(logFile, `${timestamp} [PRE] ${command}\n`);

            const dangerous = ['rm -rf /', 'mkfs', ':(){:|:&};:'];
            for (const d of dangerous) {
              if (command.includes(d)) {
                return {
                  decision: 'block',
                  reason: `危险命令被阻止: ${d}`
                };
              }
            }

            return { continue: true };
          }
        ]
      }],
      PostToolUse: [{
        matcher: 'Bash',
        hooks: [
          async (input, toolUseId) => {
            const command = input.command as string;
            const timestamp = new Date().toISOString();

            fs.appendFileSync(logFile, `${timestamp} [POST] ${command} - 完成\n`);

            return { continue: true };
          }
        ]
      }]
    }
  }
});

for await (const message of q) {
  console.log(message);
}
```

**Python**

```python
import asyncio
from datetime import datetime
from codebuddy_agent_sdk import query, CodeBuddyAgentOptions, HookMatcher

log_file = "/tmp/bash-audit.log"

async def pre_bash_hook(input_data, tool_use_id, context):
    command = input_data.get("command", "")
    timestamp = datetime.now().isoformat()

    with open(log_file, "a") as f:
        f.write(f"{timestamp} [PRE] {command}\n")

    dangerous = ["rm -rf /", "mkfs", ":(){:|:&};:"]
    for d in dangerous:
        if d in command:
            return {
                "decision": "block",
                "reason": f"危险命令被阻止： {d}"
            }

    return {"continue_": True}

async def post_bash_hook(input_data, tool_use_id, context):
    command = input_data.get("command", "")
    timestamp = datetime.now().isoformat()

    with open(log_file, "a") as f:
        f.write(f"{timestamp} [POST] {command} - 完成\n")

    return {"continue_": True}

async def main():
    options = CodeBuddyAgentOptions(
        model="deepseek-v3.1",
        hooks={
            "PreToolUse": [
                HookMatcher(matcher="Bash", hooks=[pre_bash_hook])
            ],
            "PostToolUse": [
                HookMatcher(matcher="Bash", hooks=[post_bash_hook])
            ]
        }
    )

    async for message in query(prompt="帮我清理临时文件", options=options):
        print(message)

asyncio.run(main())
```

## 示例：限制文件修改范围

**TypeScript**

```typescript
hooks: {
  PreToolUse: [{
    matcher: 'Write|Edit',
    hooks: [
      async (input) => {
        const filePath = input.file_path as string;

        if (!filePath.startsWith('/path/to/project/src/')) {
          return {
            decision: 'block',
            reason: `不允许修改 src 目录外的文件: ${filePath}`
          };
        }

        if (filePath.endsWith('.env') || filePath.includes('.git/')) {
          return {
            decision: 'block',
            reason: '不允许修改敏感文件'
          };
        }

        return { continue: true };
      }
    ]
  }]
}
```

## 相关文档

- [SDK 概览](codebuddy-sdk-quickstart.md) - 快速入门和使用示例
- [SDK 权限控制](codebuddy-sdk-permissions.md) - canUseTool 回调
- [TypeScript SDK 参考](codebuddy-sdk-typescript.md) - 完整 API 参考
- [Python SDK 参考](codebuddy-sdk-python.md) - 完整 API 参考


---

# SDK Custom Tools Guide

> **版本要求**：本文档针对 CodeBuddy Agent SDK v0.1.24 及以上版本。
> **功能状态**：SDK Custom Tools 是 CodeBuddy Agent SDK 的一项 **Preview** 功能。

本文档介绍如何在 CodeBuddy Agent SDK 中创建和使用自定义工具。自定义工具允许你定义专属的功能，让 Agent 能够调用它们来完成特定任务。

## 概述

Custom Tools 是 CodeBuddy Agent SDK 提供的一种通过 MCP（Model Context Protocol）创建自定义工具的方式。与配置外部 MCP 服务器不同，Custom Tools 允许你直接在应用程序中定义工具，无需单独的进程或服务器。

### 核心优势

- **内进程执行**：工具在应用程序内执行，无需创建独立进程
- **类型安全**：支持 TypeScript 完整的类型检查和类型推断
- **简化部署**：无需单独部署 MCP 服务器，一切随应用部署
- **紧密集成**：与应用程序共享内存和状态
- **零额外依赖**：利用现有的 SDK 基础设施

## 快速开始

### TypeScript

创建一个简单的计算器工具：

```typescript
import { query, createSdkMcpServer, tool } from '@tencent-ai/agent-sdk';
import { z } from 'zod';

// 创建 MCP 服务器并定义工具
const calculatorServer = createSdkMcpServer('calculator', {
  tools: [
    tool({
      name: 'add',
      description: 'Add two numbers',
      schema: z.object({
        a: z.number().describe('First number'),
        b: z.number().describe('Second number'),
      }),
      handler: async ({ a, b }) => {
        return { result: a + b };
      },
    }),
    tool({
      name: 'multiply',
      description: 'Multiply two numbers',
      schema: z.object({
        a: z.number().describe('First number'),
        b: z.number().describe('Second number'),
      }),
      handler: async ({ a, b }) => {
        return { result: a * b };
      },
    }),
  ],
});

// 在 SDK 中使用自定义工具
const result = query({
  prompt: 'Calculate 15 + 27 and then multiply the result by 3',
  options: {
    mcpServers: {
      'calculator': calculatorServer,
    },
  },
});

for await (const message of result) {
  console.log(message);
}
```

### Python

Python SDK 使用装饰器模式定义工具：

```python
from codebuddy_agent_sdk import query, create_sdk_mcp_server, tool
from typing import Any

# 定义工具
@tool(
    "add",
    "Add two numbers",
    {"a": float, "b": float}
)
async def add(args: dict[str, Any]) -> dict[str, Any]:
    return {'result': args['a'] + args['b']}

@tool(
    "multiply",
    "Multiply two numbers",
    {"a": float, "b": float}
)
async def multiply(args: dict[str, Any]) -> dict[str, Any]:
    return {'result': args['a'] * args['b']}

# 创建 MCP 服务器并注册工具
calculator_server = create_sdk_mcp_server(
    name='calculator',
    tools=[add, multiply]
)

# 在 SDK 中使用自定义工具
async def calculate():
    result = query(
        prompt='Calculate 15 + 27 and then multiply the result by 3',
        options={
            'mcp_servers': {
                'calculator': calculator_server,
            },
        },
    )

    async for message in result:
        print(message)

import asyncio
asyncio.run(calculate())
```

## 创建自定义工具

### TypeScript - 基本工具定义

```typescript
import { createSdkMcpServer, tool } from '@tencent-ai/agent-sdk';
import { z } from 'zod';

const myServer = createSdkMcpServer('my-tools', {
  tools: [
    tool({
      name: 'my_tool',
      description: 'Description of what the tool does',
      schema: z.object({
        parameter1: z.string().describe('Description of parameter1'),
        parameter2: z.number().optional().describe('Optional parameter'),
      }),
      handler: async (input) => {
        return {
          result: 'Tool output',
          details: input,
        };
      },
    }),
  ],
});
```

### TypeScript - 完整示例：文件分析工具

```typescript
import { createSdkMcpServer, tool } from '@tencent-ai/agent-sdk';
import { z } from 'zod';
import * as fs from 'fs/promises';
import * as path from 'path';

const fileAnalysisServer = createSdkMcpServer('file-analysis', {
  tools: [
    tool({
      name: 'count_lines',
      description: 'Count lines in a file',
      schema: z.object({
        filePath: z.string().describe('Path to the file'),
      }),
      handler: async ({ filePath }) => {
        try {
          const content = await fs.readFile(filePath, 'utf-8');
          const lineCount = content.split('\n').length;
          return {
            success: true,
            filePath,
            lineCount,
          };
        } catch (error) {
          return {
            success: false,
            error: error instanceof Error ? error.message : 'Unknown error',
          };
        }
      },
    }),
    tool({
      name: 'list_files',
      description: 'List all files in a directory',
      schema: z.object({
        dirPath: z.string().describe('Path to the directory'),
        pattern: z.string().optional().describe('Optional glob pattern'),
      }),
      handler: async ({ dirPath, pattern }) => {
        try {
          const files = await fs.readdir(dirPath);

          let filtered = files;
          if (pattern) {
            const minimatch = require('minimatch').minimatch;
            filtered = files.filter(f => minimatch(f, pattern));
          }

          return {
            success: true,
            dirPath,
            files: filtered,
            count: filtered.length,
          };
        } catch (error) {
          return {
            success: false,
            error: error instanceof Error ? error.message : 'Unknown error',
          };
        }
      },
    }),
  ],
});

export default fileAnalysisServer;
```

### Python - 装饰器模式

```python
from codebuddy_agent_sdk import create_sdk_mcp_server, tool
from typing import Any
import os

@tool(
    "count_lines",
    "Count lines in a file",
    {"file_path": str}
)
async def count_lines(args: dict[str, Any]) -> dict[str, Any]:
    try:
        with open(args['file_path'], 'r') as f:
            line_count = len(f.readlines())
        return {
            'success': True,
            'file_path': args['file_path'],
            'line_count': line_count,
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
        }

# 创建 MCP 服务器并注册工具
file_analysis_server = create_sdk_mcp_server(
    name='file-analysis',
    tools=[count_lines, list_files, get_file_info]
)
```

## 多个工具管理

### TypeScript

```typescript
import { createSdkMcpServer, tool } from '@tencent-ai/agent-sdk';
import { z } from 'zod';

const multiToolServer = createSdkMcpServer('multi-tools', {
  tools: [
    tool({
      name: 'tool_one',
      description: 'First tool',
      schema: z.object({ input: z.string() }),
      handler: async ({ input }) => ({ result: `Tool 1: ${input}` }),
    }),
    tool({
      name: 'tool_two',
      description: 'Second tool',
      schema: z.object({ data: z.number() }),
      handler: async ({ data }) => ({ result: `Tool 2: ${data * 2}` }),
    }),
  ],
});

// 在 SDK 中使用
const result = query({
  prompt: 'Use all the available tools',
  options: {
    mcpServers: {
      'multi-tools': multiToolServer,
    },
  },
});
```

### Python

```python
from codebuddy_agent_sdk import create_sdk_mcp_server, tool
from typing import Any

@tool("tool_one", "First tool", {"input": str})
async def tool_one(args: dict[str, Any]) -> dict[str, Any]:
    return {'result': f"Tool 1: {args['input']}"}

@tool("tool_two", "Second tool", {"data": int})
async def tool_two(args: dict[str, Any]) -> dict[str, Any]:
    return {'result': f"Tool 2: {args['data'] * 2}"}

# 创建 MCP 服务器并注册工具
server = create_sdk_mcp_server(
    name='multi-tools',
    tools=[tool_one, tool_two]
)
```

## 类型安全

### TypeScript - 使用 Zod 模式

Zod 提供运行时类型验证和强大的类型推断：

```typescript
import { createSdkMcpServer, tool } from '@tencent-ai/agent-sdk';
import { z } from 'zod';

const dataProcessingServer = createSdkMcpServer('data-processing', {
  tools: [
    tool({
      name: 'process_user_data',
      description: 'Process and validate user data',
      schema: z.object({
        userId: z.number().int().positive().describe('User ID'),
        email: z.string().email().describe('User email'),
        tags: z.array(z.string()).describe('User tags'),
        preferences: z.object({
          notifications: z.boolean().default(true),
          theme: z.enum(['light', 'dark', 'auto']).default('auto'),
        }).optional(),
      }),
      handler: async (input) => {
        const result = {
          userId: input.userId,
          email: input.email,
          tagCount: input.tags.length,
          hasPreferences: !!input.preferences,
        };
        return result;
      },
    }),
  ],
});
```

### Python - 类型注解

Python SDK 支持简单类型映射或 JSON Schema：

```python
from codebuddy_agent_sdk import create_sdk_mcp_server, tool
from typing import Any

# 使用 JSON Schema 进行高级验证
@tool(
    "process_user_data",
    "Process and validate user data",
    {
        "type": "object",
        "properties": {
            "user_id": {"type": "integer", "minimum": 1},
            "email": {"type": "string", "format": "email"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "notifications": {"type": "boolean", "default": True},
            "theme": {"type": "string", "enum": ["light", "dark", "auto"], "default": "auto"}
        },
        "required": ["user_id", "email", "tags"]
    }
)
async def process_user_data(args: dict[str, Any]) -> dict[str, Any]:
    return {
        'user_id': args['user_id'],
        'email': args['email'],
        'tag_count': len(args['tags']),
        'theme': args.get('theme', 'auto'),
        'notifications': args.get('notifications', True),
    }

server = create_sdk_mcp_server(
    name='data-processing',
    tools=[process_user_data]
)
```

## 完整示例：数据库查询工具

### TypeScript

```typescript
import { createSdkMcpServer, tool } from '@tencent-ai/agent-sdk';
import { z } from 'zod';

interface Database {
  query(sql: string, params?: any[]): Promise<{ rows: Record<string, any>[]; rowCount: number }>;
}

const db: Database = new Database();

const databaseServer = createSdkMcpServer('database', {
  tools: [
    tool({
      name: 'execute_query',
      description: 'Execute a read-only SQL query',
      schema: z.object({
        sql: z.string().describe('SQL query to execute'),
        params: z.array(z.any()).optional().describe('Query parameters'),
      }),
      handler: async ({ sql, params }) => {
        try {
          const upperSql = sql.toUpperCase();
          if (
            upperSql.includes('DROP') ||
            upperSql.includes('DELETE') ||
            upperSql.includes('UPDATE') ||
            upperSql.includes('INSERT')
          ) {
            return {
              success: false,
              error: 'Only SELECT queries are allowed',
            };
          }

          const result = await db.query(sql, params);
          return {
            success: true,
            rows: result.rows,
            rowCount: result.rowCount,
          };
        } catch (error) {
          return {
            success: false,
            error: error instanceof Error ? error.message : 'Query execution failed',
          };
        }
      },
    }),
    tool({
      name: 'get_table_schema',
      description: 'Get the schema of a table',
      schema: z.object({
        tableName: z.string().describe('Name of the table'),
      }),
      handler: async ({ tableName }) => {
        try {
          const result = await db.query(
            `SELECT column_name, data_type FROM information_schema.columns WHERE table_name = $1`,
            [tableName]
          );
          return {
            success: true,
            tableName,
            columns: result.rows,
          };
        } catch (error) {
          return {
            success: false,
            error: error instanceof Error ? error.message : 'Schema retrieval failed',
          };
        }
      },
    }),
  ],
});
```

### Python

```python
from codebuddy_agent_sdk import create_sdk_mcp_server, tool
from typing import Any

class Database:
    async def query(self, sql: str, params: list[Any] = None) -> dict[str, Any]:
        pass

db = Database()

@tool(
    "execute_query",
    "Execute a read-only SQL query",
    {"sql": str, "params": list}
)
async def execute_query(args: dict[str, Any]) -> dict[str, Any]:
    try:
        sql = args['sql']
        params = args.get('params')

        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT']
        if any(keyword in sql.upper() for keyword in dangerous_keywords):
            return {
                'success': False,
                'error': 'Only SELECT queries are allowed',
            }

        result = await db.query(sql, params)
        return {
            'success': True,
            'rows': result.get('rows', []),
            'row_count': result.get('row_count', 0),
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
        }

server = create_sdk_mcp_server(
    name='database',
    tools=[execute_query, get_table_schema]
)
```

## 错误处理

### TypeScript - API 调用错误处理

```typescript
import { createSdkMcpServer, tool } from '@tencent-ai/agent-sdk';
import { z } from 'zod';

const apiServer = createSdkMcpServer('api-tools', {
  tools: [
    tool({
      name: 'fetch_data',
      description: 'Fetch data from an API',
      schema: z.object({
        endpoint: z.string().url().describe('API endpoint URL'),
      }),
      handler: async ({ endpoint }) => {
        try {
          const response = await fetch(endpoint);
          if (!response.ok) {
            return {
              content: [{
                type: 'text',
                text: `API error: ${response.status} ${response.statusText}`,
              }],
            };
          }
          const data = await response.json();
          return {
            content: [{
              type: 'text',
              text: JSON.stringify(data, null, 2),
            }],
          };
        } catch (error) {
          return {
            content: [{
              type: 'text',
              text: `Failed to fetch data: ${error instanceof Error ? error.message : String(error)}`,
            }],
          };
        }
      },
    }),
  ],
});
```

## 最佳实践

### 1. 使用明确的参数类型和描述

为工具参数提供清晰的类型和描述，帮助 Agent 理解如何调用工具：

```typescript
tool({
  name: 'process_data',
  schema: z.object({
    data: z.array(z.string()).describe('Data to process'),
    format: z.enum(['json', 'csv']).describe('Output format'),
  }),
  handler: async ({ data, format }) => {
    // 处理逻辑
  },
})
```

### 2. 提供有意义的错误反馈

始终返回明确的错误信息，以便 Agent 和用户理解发生了什么。

### 3. 验证输入参数

确保输入符合预期的格式和范围。

## 相关文档

- [SDK 概览](codebuddy-sdk-quickstart.md)
- [SDK MCP 集成](codebuddy-sdk-mcp.md)
- [TypeScript SDK 参考](codebuddy-sdk-typescript.md)
- [Python SDK 参考](codebuddy-sdk-python.md)
- [SDK 权限系统](codebuddy-sdk-permissions.md)

## 更多资源

- [MCP 官方文档](https://modelcontextprotocol.io/)
- [Zod 验证库](https://zod.dev/)


---

# SDK MCP Integration Guide

> **版本要求**：本文档针对 CodeBuddy Agent SDK v0.1.24 及以上版本。
> **功能状态**：SDK MCP 支持是 CodeBuddy Agent SDK 的一项 **Preview** 功能。

本文档介绍如何在 CodeBuddy Agent SDK 中集成和使用 MCP（Model Context Protocol）服务器，为你的应用程序扩展自定义工具和功能。

## 概述

MCP 服务器允许你将自定义工具、资源和提示集成到 CodeBuddy Agent SDK 中。通过 SDK，你可以以编程方式启用这些工具，从而为你的 AI 工作流添加特定领域的功能。

### 核心概念

* **MCP 服务器**：提供工具、资源和提示的独立进程或服务
* **工具**：MCP 服务器暴露的可执行函数
* **资源**：MCP 服务器提供的可读取的数据或文件
* **提示**：MCP 服务器提供的模板化提示词

## 支持的传输类型

CodeBuddy Agent SDK 支持三种 MCP 服务器通信方式：

| 传输类型 | 使用场景 | 说明 |
| --- | --- | --- |
| **STDIO** | 本地工具 | 通过标准输入输出与本地进程通信 |
| **HTTP/SSE** | 远程服务 | 通过 HTTP 流式传输或 Server-Sent Events 与远程服务通信 |
| **SDK MCP** | IDE/SDK 集成 | 由外部 SDK（如 IDE 扩展）注册的 MCP 服务器 |

## 配置 MCP 服务器

`mcpServers` 支持两种配置方式：

1. **对象格式**：直接在代码中定义服务器配置
2. **文件路径字符串**：指向一个 MCP 配置文件（JSON 格式）

### 对象格式

**TypeScript**：

```typescript
import { query } from '@tencent-ai/agent-sdk';

const result = query({
  prompt: 'Analyze my project structure',
  options: {
    mcpServers: {
      'my-tools': {
        type: 'stdio',
        command: 'node',
        args: ['./mcp-server.js'],
        env: {
          NODE_ENV: 'production'
        }
      },
      'api-server': {
        type: 'http',
        url: 'https://api.example.com/mcp',
        headers: {
          'Authorization': 'Bearer your-token'
        }
      },
      'sse-server': {
        type: 'sse',
        url: 'https://events.example.com/mcp/sse',
        headers: {
          'X-API-Key': 'your-api-key'
        }
      }
    }
  }
});
```

**Python**：

```python
from codebuddy_agent_sdk import query

result = query(
    prompt='Analyze my project structure',
    options={
        'mcp_servers': {
            'my-tools': {
                'type': 'stdio',
                'command': 'python',
                'args': ['-m', 'my_mcp_server'],
                'env': {
                    'PYTHONPATH': '/path/to/tools'
                }
            },
            'api-server': {
                'type': 'http',
                'url': 'https://api.example.com/mcp',
                'headers': {
                    'Authorization': 'Bearer your-token'
                }
            }
        }
    }
)
```

### 文件路径格式

也可以传入一个 MCP 配置文件路径。配置文件为 JSON 格式，结构与 CLI `--mcp-config` 参数一致。

**TypeScript**：

```typescript
const result = query({
  prompt: 'Analyze my project structure',
  options: {
    mcpServers: './mcp-config.json'
  }
});
```

**Python**：

```python
result = query(
    prompt='Analyze my project structure',
    options={
        'mcp_servers': './mcp-config.json'
    }
)
```

配置文件格式示例（`mcp-config.json`）：

```json
{
  "mcpServers": {
    "my-tools": {
      "type": "stdio",
      "command": "node",
      "args": ["./mcp-server.js"]
    }
  }
}
```

> **注意**：使用文件路径格式时，不支持 SDK MCP 类型（`type: 'sdk'`）的服务器配置。SDK MCP 服务器仅在对象格式中可用。

## 服务器配置详解

### STDIO 配置

STDIO 服务器通过标准输入输出与本地进程通信，适用于本地工具。

```typescript
{
  type: 'stdio',
  command: 'python',                    // 可执行文件或命令
  args: ['-m', 'my_mcp_server'],       // 命令行参数
  env: {                               // 环境变量
    PYTHONPATH: '/path/to/tools',
    DEBUG: 'true'
  }
}
```

**常见用例**：

```typescript
// Python MCP 服务器
{
  type: 'stdio',
  command: 'python',
  args: ['-m', 'fastmcp']
}

// Node.js MCP 服务器
{
  type: 'stdio',
  command: 'node',
  args: ['./server.js']
}

// 本地二进制文件
{
  type: 'stdio',
  command: '/usr/local/bin/my-tool',
  args: ['--config', '/etc/config.yaml']
}
```

### HTTP 配置

HTTP 服务器通过 HTTP 流式传输与远程服务通信。

```typescript
{
  type: 'http',
  url: 'https://mcp.example.com/api/v1',
  headers: {
    'Authorization': 'Bearer your-token',
    'Content-Type': 'application/json'
  }
}
```

### SSE 配置

SSE 服务器通过 Server-Sent Events 与远程服务通信。

```typescript
{
  type: 'sse',
  url: 'https://events.example.com/mcp/sse',
  headers: {
    'Authorization': 'Bearer your-token',
    'X-API-Version': 'v1'
  }
}
```

## 权限管理

MCP 工具支持精细化的权限控制。通过 `canUseTool` 回调，你可以决定哪些工具可以被使用。

### 特定工具的权限控制

```typescript
options: {
  canUseTool: (toolCall) => {
    // 阻止特定服务器的工具
    if (toolCall.name.startsWith('mcp__restricted')) {
      return false;
    }

    // 允许特定工具
    if (toolCall.name === 'mcp__github__list_issues') {
      return true;
    }

    // 默认询问
    return null;
  }
}
```

## 使用 MCP 工具

配置完 MCP 服务器后，Agent 会自动发现这些服务器提供的工具，并在需要时调用它们。

### 自动工具发现

```typescript
const result = query({
  prompt: `
    Using the available MCP tools, complete these tasks:
    1. Query the database for recent transactions
    2. Analyze the results
    3. Generate a summary report
  `,
  options: {
    mcpServers: {
      'database': {
        type: 'stdio',
        command: 'python',
        args: ['-m', 'db_mcp_server']
      }
    }
  }
});
```

## 实例：数据库查询 MCP 服务器

### 创建 MCP 服务器

**db_mcp_server.py**：

```python
from fastmcp import FastMCP

mcp = FastMCP('database')

@mcp.tool()
def query_database(sql: str) -> str:
    """Execute a SQL query and return results"""
    return f"Query results for: {sql}"

@mcp.tool()
def get_schema(table_name: str) -> str:
    """Get the schema of a specific table"""
    return f"Schema for: {table_name}"

if __name__ == '__main__':
    mcp.run()
```

### 在 SDK 中使用

**TypeScript**：

```typescript
import { query } from '@tencent-ai/agent-sdk';

async function analyzeData() {
  const result = query({
    prompt: `
      I need to analyze our user activity data.
      1. First, check the schema of the users table
      2. Query for users active in the last 7 days
      3. Provide insights based on the results
    `,
    options: {
      mcpServers: {
        'database': {
          type: 'stdio',
          command: 'python',
          args: ['-m', 'db_mcp_server'],
          env: {
            DATABASE_URL: process.env.DATABASE_URL
          }
        }
      },
      permissionMode: 'acceptEdits'
    }
  });

  for await (const message of result) {
    if (message.type === 'content') {
      console.log('Analysis:', message.text);
    }
  }
}
```

**Python**：

```python
from codebuddy_agent_sdk import query
import os

async def analyze_data():
    result = query(
        prompt="""
            I need to analyze our user activity data.
            1. First, check the schema of the users table
            2. Query for users active in the last 7 days
            3. Provide insights based on the results
        """,
        options={
            'mcp_servers': {
                'database': {
                    'type': 'stdio',
                    'command': 'python',
                    'args': ['-m', 'db_mcp_server'],
                    'env': {
                        'DATABASE_URL': os.environ.get('DATABASE_URL')
                    }
                }
            },
            'permission_mode': 'acceptEdits'
        }
    )

    async for message in result:
        if message.get('type') == 'content':
            print('Analysis:', message.get('text'))
```

## 实例：远程 SSE 服务器

SSE 服务器适用于需要实时数据流或事件推送的场景。

```typescript
import { query } from '@tencent-ai/agent-sdk';

async function monitorSystem() {
  const result = query({
    prompt: 'Monitor system metrics and alert if any threshold is exceeded',
    options: {
      mcpServers: {
        'monitoring': {
          type: 'sse',
          url: 'https://monitor.example.com/mcp/events',
          headers: {
            'Authorization': `Bearer ${process.env.MONITOR_TOKEN}`,
            'X-Client-ID': 'codebuddy-sdk'
          }
        }
      }
    }
  });

  for await (const message of result) {
    if (message.type === 'content') {
      console.log('Alert:', message.text);
    }
  }
}
```

## 错误处理

在处理 MCP 服务器连接时，监控初始化状态并识别失败的连接：

**TypeScript**：

```typescript
import { query } from '@tencent-ai/agent-sdk';

const result = query({
  prompt: 'Use my MCP tools',
  options: {
    mcpServers: {
      'my-tool': {
        type: 'stdio',
        command: 'python',
        args: ['-m', 'my_mcp_server']
      }
    }
  }
});

for await (const message of result) {
  // 在初始化时检查 MCP 服务器状态
  if (message.type === 'system' && message.subtype === 'init') {
    const failedServers = message.mcp_servers.filter(
      s => s.status !== 'connected'
    );
    if (failedServers.length > 0) {
      console.warn('Failed to connect:', failedServers);
    }
  }

  // 处理执行错误
  if (message.type === 'result' && message.subtype === 'error_during_execution') {
    console.error('Tool execution failed:', message);
  }
}
```

**Python**：

```python
from codebuddy_agent_sdk import query

result = query(
    prompt='Use my MCP tools',
    options={
        'mcp_servers': {
            'my-tool': {
                'type': 'stdio',
                'command': 'python',
                'args': ['-m', 'my_mcp_server']
            }
        }
    }
)

async for message in result:
    if message.get('type') == 'system' and message.get('subtype') == 'init':
        failed_servers = [
            s for s in message.get('mcp_servers', [])
            if s.get('status') != 'connected'
        ]
        if failed_servers:
            print(f'Failed to connect: {failed_servers}')

    if message.get('type') == 'result' and message.get('subtype') == 'error_during_execution':
        print('Tool execution failed:', message)
```

## 相关文档

* [SDK 概览](codebuddy-sdk-quickstart.md)
* [TypeScript SDK 参考](codebuddy-sdk-typescript.md)
* [Python SDK 参考](codebuddy-sdk-python.md)
* [SDK 权限系统](codebuddy-sdk-permissions.md)
* [SDK Custom Tools](codebuddy-sdk-custom-tools.md)

## 更多资源

* [MCP 官方文档](https://modelcontextprotocol.io/)
* [MCP Python SDK - FastMCP](https://github.com/modelcontextprotocol/python-sdk)
* [MCP TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk)


---

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


---

