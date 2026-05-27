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
