# Agent Project Guide

This project builds a knowledge-base agent service. The assistant should answer questions about the selected knowledge base or workspace, not about the agent implementation itself unless the user explicitly asks how the agent is built, tested, or integrated.

## Read First

- Follow `docs/development-workflow.md` for every implementation change.
- Follow `docs/typescript-development-guidelines.md` whenever writing or refactoring TypeScript. Treat it as the project coding rule for readable, extensible code.
- Follow `docs/architecture.md` for the ports/adapters boundary.
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

Use these logs to verify actual behavior:

- `.harness/logs/conversation.jsonl` records user input and final output.
- `.harness/logs/llm-raw.jsonl` records LLM request/response/error events.

Do not log API keys, secrets, authorization headers, access tokens, or refresh tokens.

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
