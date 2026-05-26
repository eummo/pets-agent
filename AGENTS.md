# Agent Project Guide

This project builds a knowledge-base agent service. The assistant should answer questions about the selected knowledge base or workspace, not about the agent implementation itself unless the user explicitly asks how the agent is built, tested, or integrated.

## Read First

- Follow `docs/development-workflow.md` for every implementation change.
- Follow `docs/typescript-development-guidelines.md` whenever writing or refactoring TypeScript. Treat it as the project coding rule for readable, extensible code.
- Follow `docs/architecture.md` for the contracts/adapters boundary.
- Keep provider-specific code behind adapters. The orchestration layer must not depend directly on Enterprise WeChat, Claude Code, MiniMax, GitHub, or future provider SDKs.

## Required Verification

Before calling work complete, run:

```bash
npm run check
npm run smoke
```

`npm run check` verifies TypeScript, lint, unit tests, and build.
`npm run smoke` calls the running local service, performs model-backed regression checks, and verifies logs.

## Runtime Logs

Runtime logs are the primary observability surface for customer behavior. Use them to reconstruct
what the customer asked, how the service classified and authorized the request, which model calls
were made, which tools were invoked, and what final answer was returned.

- `.harness/logs/conversation.jsonl` records user input and final output.
- `.harness/logs/system.jsonl` records orchestration events such as workspace resolution,
  role resolution, final intent classification, permission denial, runtime selection, context usage,
  and compaction.
- `.harness/logs/llm-raw.jsonl` records model and tool observability events:
  - `llm.request`, `llm.response`, and `llm.error` for agent-runtime model calls;
  - `llm.request`, `llm.response`, and `intent.result` for intent-detection model calls;
  - `llm.request`, `llm.response`, and `tool.permission_result` for Bash permission classification;
  - `agent.tool_call` and `agent.tool_result` for customer requests that cause the agent to invoke tools.

When investigating a customer report, start from the latest matching `conversation.turn`, then follow
the same `messageId`, `userId`, `workspacePath`, and nearby timestamps across `system.jsonl` and
`llm-raw.jsonl`. A denied request should still have intent-detection logs, but it should not have an
agent-runtime `llm.response`.

Do not log API keys, secrets, authorization headers, access tokens, or refresh tokens.

Log files are UTF-8 encoded and contain CJK characters. On Windows, always read logs with UTF-8 encoding:

```powershell
Get-Content .harness\logs\llm-raw.jsonl -Encoding utf8
Get-Content .harness\logs\system.jsonl -Encoding utf8
Get-Content .harness\logs\conversation.jsonl -Encoding utf8
```

## Development Harness

Create or reset the local workspace fixture with:

```bash
npm run harness -- --reset
```

Start the local service with:

```bash
npm run dev
```

Browser test page:

```text
http://127.0.0.1:3000/
```

## Regression Rule

When a bug is found from browser behavior or logs:

- add a deterministic unit test when possible;
- add or update `src/smoke/regressionSmoke.ts` when the bug is runtime/model behavior;
- keep the regression focused on observable behavior, not implementation details.

Current important regression: when users ask about the current project or architecture, answer from the selected workspace content only. Do not describe the agent runtime, message channels, model provider, browser test page, or Enterprise WeChat integration unless the user explicitly asks about those implementation details.
