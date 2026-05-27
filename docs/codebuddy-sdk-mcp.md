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
