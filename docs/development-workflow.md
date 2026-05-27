# Development Workflow

Use this workflow for every feature or behavior change.

Before writing TypeScript, also follow `docs/typescript-development-guidelines.md` for readability,
type modeling, naming, errors, and test expectations.

## 1. Start From A Fresh Harness

```bash
npm run harness -- --reset
npm run dev
```

Open the browser test page:

```text
http://127.0.0.1:3000/
```

## 2. Implement Behind System Contracts

- Message channels must stay behind `MessageChannel` / HTTP adapters.
- LLM or coding SDKs must stay behind `AgentRuntime`.
- Repository selection must stay behind `KnowledgeWorkspaceResolver`.
- Permissions must stay behind `AuthorizationService`.
- PR or branch publication must stay behind `ChangePublisher`.

Do not import Enterprise WeChat, Claude Code, MiniMax, GitHub, or other provider SDKs into orchestration code.

## 3. Evaluate Test Coverage Needs

Before running checks, consider whether the current change requires new or updated test cases:

- **Unit test**: needed for new deterministic logic (parsing, routing, authorization, configuration, persistence, formatting).
- **Smoke test**: needed for runtime/model behavior changes (intent classification, role permissions, workspace grounding, stream events, tool permissions, new API endpoints).
- If no new test is needed (pure refactor, doc change, config tweak), note why.

This step is not optional — every change must explicitly consider test coverage, even if the conclusion is "no new test needed."

## 4. Run Static And Unit Checks

```bash
npm run check
```

This runs typecheck, lint, unit tests, and production build.

## 5. Run Browser/Runtime Smoke

With the service running:

```bash
npm run smoke
```

The smoke test sends real `/dev/chat` messages, checks key response expectations, and verifies both log files exist:

- `.harness/logs/conversation.jsonl`
- `.harness/logs/llm-raw.jsonl`

## 6. Read Logs Before Calling Work Done

Inspect the latest lines:

```powershell
Get-Content .harness\logs\conversation.jsonl | Select-Object -Last 5
Get-Content .harness\logs\llm-raw.jsonl | Select-Object -Last 10
```

Confirm:

- user input is encoded correctly;
- the selected workspace is correct;
- LLM request contains grounding context when needed;
- output matches the intended behavior;
- no API keys or secrets appear in logs.

## 7. Add Or Update Regression Cases

When a bug is found from logs or browser testing:

- add a focused unit test if it is deterministic;
- add or update a smoke case in `src/smoke/regressionSmoke.ts` if it is runtime behavior;
- keep the case small and named after the behavior, not the implementation.

Current fixed regression:

- asking what the current project does must answer "enterprise knowledge-base agent" and must not infer a pet business from the project name.
