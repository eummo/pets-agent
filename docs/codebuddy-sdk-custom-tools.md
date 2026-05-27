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
