# pets-agent

Enterprise WeChat connected Claude knowledge-base agent.

This repository starts with the development workflow and local harness needed to build the agent safely.

Project-level instructions for Codex and coding agents live in `AGENTS.md`.

## Development

```bash
npm install
npm run harness -- --reset
npm run dev
npm run check
```

Useful scripts:

- `npm run dev` starts the Fastify development service.
- `npm run harness -- --reset` creates a local `.harness/` knowledge-base sandbox.
- `npm run smoke` runs browser/runtime regression smoke tests against the running service.
- `npm run typecheck` runs strict TypeScript checks.
- `npm run lint` runs ESLint.
- `npm test` runs Vitest.
- `npm run build` compiles production output.
- `npm run check` runs typecheck, lint, tests, and build.

## Harness

The harness creates a local sandbox under `.harness/`:

```text
.harness/
  knowledge-base/
    CLAUDE.md
    docs/
    requirements/
    .claude/
    code/
      catalog-api/
      order-service/
  repos.json
```

Each repository fixture under `knowledge-base/code/` has its own source tree and, when Git is available,
its own Git repository. The harness is intentionally ignored by Git.

## Architecture Direction

The implementation should keep provider-specific integrations behind ports:

- message channels implement a common `MessageChannel` interface;
- Claude Code SDK implements the first `AgentRuntime`;
- future SDKs should add new `AgentRuntime` adapters without changing orchestration;
- GitHub PR publishing should live behind `ChangePublisher`.

See `docs/architecture.md`.
See `docs/development-workflow.md` for the fixed implementation and verification loop.

## LLM Configuration

The local Anthropic-compatible model endpoint is configured in `config/llm.json`.

Current defaults:

- `baseUrl`: `https://api.minimaxi.com/anthropic`
- `modelId`: `MiniMax-M2.7`
- `apiKeyEnv`: `LOCAL_LLM_API_KEY`

Keep the actual API key in the environment only.

By default, the messages runtime keeps local multi-turn history in `.harness/state/history.json`,
keyed by channel, user, and workspace. Send `/new` to archive the active local history and start a fresh current conversation; archived history remains stored but is not loaded into new turns.

To use SDK-managed multi-turn sessions, set `runtime` to `managed-sessions` in `config/llm.json` and provide:

- `agentIdEnv`: environment variable containing the Managed Agents `agent_...` id
- `environmentIdEnv`: environment variable containing the Managed Agents `env_...` id

The service stores local channel/user/workspace to SDK session mappings in `.harness/state/sessions.json` by default. Send `/new` to archive the current SDK session and start fresh.
