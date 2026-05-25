# Architecture

This service is a layered gateway for knowledge-base and workspace agents. Users may send messages
from TUI, HTTP APIs, browser pages, Enterprise WeChat, or future channels, but every channel must
normalize its input into the same core message contract and hand it to the gateway.

## Target Flow

```text
User channel
  -> channel adapter
  -> message gateway
  -> role and permission services
  -> intent gate
  -> agent runtime adapter
  -> selected workspace / knowledge base
```

The gateway is the only place that coordinates the business flow:

1. receive a normalized `InboundMessage`;
2. resolve the selected workspace or knowledge base;
3. identify the user role;
4. load role permissions from the configured store;
5. classify the user's intent;
6. deny and record feedback when the intent is not allowed;
7. call the configured `AgentRuntime` when the intent is allowed;
8. persist session, history, feedback, progress, and conversation logs.

## Layers

### 1. Channel Adapters

Channel adapters own transport details and nothing else. They parse inbound payloads, build
`InboundMessage`, call the gateway, and translate `OutboundMessage` back to the channel.

Current examples:

- HTTP and browser dev page routes in `src/server/createServer.ts`
- Enterprise WeChat smart bot adapter (WebSocket long connection) in `src/wechat/wechatSmartBotAdapter.ts`

Future examples:

- TUI adapter
- public HTTP API adapter
- Slack, Feishu, DingTalk, or other chat adapters

Channel adapters must not decide roles, permissions, intent, workspace access, or agent SDK
behavior.

### 2. Message Gateway

The gateway implements `MessageHandler`. It is the application orchestration layer and should stay
provider-neutral.

Current implementation:

- `src/core/orchestrator.ts`

Gateway responsibilities:

- command handling such as `/help` and `/new`;
- workspace resolution through `KnowledgeWorkspaceResolver`;
- role lookup and permission checks through `AuthorizationService`;
- intent classification through `IntentDetectionService`;
- feedback capture through `FeedbackStore`;
- runtime selection through `AgentRuntime` or `AgentRuntimeFactory`;
- session and history persistence;
- progress and final conversation logging.

The gateway must not import Enterprise WeChat, Claude SDK, MiniMax, GitHub, Codex SDK, pi-agent SDK,
or future provider SDK types directly.

### 3. Domain Services And Stores

Domain services answer business questions for the gateway:

- `AuthorizationService`: maps user -> role, and role + action -> allow/deny.
- `IntentDetectionService`: classifies the user's request as query, mutation, or knowledge-base update.
- `KnowledgeWorkspaceResolver`: selects the knowledge-base root or source repository workspace.
- `FeedbackStore`: records denied-but-useful requests for later review.
- `RoleConfigStore`: stores role prompts, capabilities, model selection, and allowed tools.
- session and history stores keep conversation continuity independent of the channel.

Database-backed implementations belong in `src/db`. Static or development implementations may live
in focused adapter folders such as `src/security` or `src/repos`.

### 4. Agent Runtime Adapters

Agent SDKs sit behind `AgentRuntime`. Adding a new SDK means adding a runtime adapter, not changing
channel adapters or gateway policy.

Current examples:

- `ClaudeSdkAgentRuntime` in `src/agent/claudeSdkAgentRuntime.ts`
- `EchoAgentRuntime` for local fallback/testing

Future examples:

- `PiAgentRuntime`
- `CodexSdkAgentRuntime`
- another Claude-compatible or custom runtime

Runtime adapters may know provider SDK details. They receive an already-authorized `AgentRequest`
with the selected workspace path, user text, role-derived configuration, session id, and stream
callbacks.

### 5. Infrastructure

Infrastructure modules provide storage, logging, configuration, and server lifecycle:

- `src/config`: model and endpoint configuration
- `src/db`: SQLite-backed role and feedback stores
- `src/logging`: JSONL logs
- `src/harness`: local workspace fixture and smoke harness
- `src/index.ts`: composition root that wires concrete adapters into ports

`src/index.ts` may import concrete adapters because it is the composition root. Core orchestration
code should depend only on ports.

## Ports

Stable contracts live in `src/core/ports.ts`.

Important ports:

- `MessageHandler`: the gateway entry point used by every channel adapter.
- `AgentRuntime`: provider-neutral execution interface for Claude, pi-agent, Codex, or future SDKs.
- `AgentRuntimeFactory`: creates a role-specific runtime from role configuration.
- `AuthorizationService`: role lookup and capability checks.
- `IntentDetectionService`: intent classification before runtime execution.
- `KnowledgeWorkspaceResolver`: workspace or knowledge-base selection.
- `FeedbackStore`: denied intent capture.
- `RoleConfigStore`: role, prompt, capability, tool, and model configuration.
- `ConversationSessionStore` and `ConversationHistoryStore`: channel-independent conversation state,
  keyed by `(channel, userId, workspacePath, chatId?)`. The optional `chatId` isolates group chats
  from each other and from single chats within the same channel.

## Adding A New Channel

1. Create a channel adapter folder, for example `src/tui` or `src/api`.
2. Convert the channel payload into `InboundMessage`.
3. Call the injected `MessageHandler`.
4. Convert `OutboundMessage` and stream/progress events back to the channel's response shape.
5. Add the adapter to `src/index.ts` or a small composition module.

No role, permission, intent, or SDK logic should be added to the channel adapter.

## Adding A New Agent SDK

1. Create an adapter under `src/agent`, for example `piAgentRuntime.ts` or `codexSdkAgentRuntime.ts`.
2. Implement `AgentRuntime`.
3. Convert `AgentRequest` into the provider SDK request shape.
4. Convert provider responses and streams into `AgentResponse` and `AgentStreamEvent`.
5. Register the runtime in the composition root or runtime factory.

Do not import provider SDK types into `src/core`.

## Adding A New Permission Or Tool

1. Add the smallest new `RoleCapability` in `src/core/ports.ts`.
2. Store it in role configuration through `RoleConfigStore`.
3. Map user intent or gateway actions to the capability in `AuthorizationService`.
4. Add focused tests for allow and deny behavior.

Permissions should be checked before calling `AgentRuntime`. When a user's intent is useful but not
allowed, the gateway should return a clear denial and save the request into feedback.

## Dependency Rule

Dependencies point inward:

```text
channel adapters  -> core ports
agent adapters    -> core ports
db adapters       -> core ports
composition root  -> all adapters
core gateway      -> core ports only
```

Provider-specific code stays behind adapters. The gateway decides business flow; adapters translate
between the outside world and stable ports.
