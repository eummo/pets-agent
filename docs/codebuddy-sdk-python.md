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
