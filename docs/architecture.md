# Architecture

The codebase is organized around stable ports and replaceable adapters.

## Core Ports

- `MessageChannel` receives inbound messages and sends outbound replies.
- `AgentRuntime` executes a user request inside a selected workspace.
- `KnowledgeWorkspaceResolver` selects the knowledge-base root or source repository workspace.
- `AuthorizationService` answers whether a user may read, suggest, or mutate.
- `ChangePublisher` turns approved changes into commits, pushes, and pull requests.

## Initial Adapters

- Fastify hosts HTTP routes and owns no business workflow.
- Enterprise WeChat implements the first message-channel adapter.
- Claude Code SDK will implement the first real `AgentRuntime`; the current development service can use a test runtime.
- GitHub implements `ChangePublisher`.
- File-based JSON config implements repository, RBAC, and model configuration.

## Extension Rule

New channels or SDKs must be added as adapters behind these ports. The orchestration layer should not import
Enterprise WeChat, Claude Code, GitHub, or provider-specific SDK types directly.
