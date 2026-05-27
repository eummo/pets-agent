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
