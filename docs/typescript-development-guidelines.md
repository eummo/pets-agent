# TypeScript Development Guidelines

Use these guidelines when adding or changing TypeScript code in this project. They complement the
required workflow in `docs/development-workflow.md` and the ports/adapters boundary in
`docs/architecture.md`.

## Design Priorities

1. Keep behavior easy to read before making it clever.
2. Keep orchestration code provider-neutral.
3. Make invalid states difficult to represent with types.
4. Prefer small, named steps over deeply nested control flow.
5. Add tests around observable behavior, not private implementation details.

## Module Boundaries

- Put stable contracts in `src/core/ports.ts` or a nearby core module.
- Put provider-specific code in adapters such as `src/wechat`, `src/agent`, `src/server`, or future
  provider directories.
- Do not import Enterprise WeChat, Claude SDK, MiniMax, GitHub, or provider SDK types into core
  orchestration code.
- Let dependencies point inward: adapters depend on ports, but ports do not depend on adapters.
- Prefer constructor-injected dependencies for services that coordinate multiple ports.

## Function Shape

- Keep functions focused on one decision or one transformation.
- Use early returns for guard cases and permission failures.
- Extract named helpers when a block has a business meaning, not merely to reduce line count.
- Avoid long ternary chains when a map, switch, or named helper would explain intent better.
- Keep side effects near the orchestration edge. Pure parsing, formatting, and selection helpers
  should not write logs, mutate stores, or call providers.

## Types

- Use `type` aliases consistently, matching the ESLint rule.
- Prefer readonly object shapes and readonly arrays for contracts.
- Model domain choices with literal unions, for example `"read" | "suggest" | "mutate"`.
- Use discriminated unions for event streams and status-like values.
- Avoid `any`. Use `unknown` at boundaries, validate it, then narrow to a useful type.
- Avoid optional properties when a separate union member communicates the state more clearly.
- Keep DTOs and domain concepts separate when provider payloads are noisy or unstable.

## Naming

- Name ports by capability: `AgentRuntime`, `AuthorizationService`, `KnowledgeWorkspaceResolver`.
- Name adapters by implementation or transport: `ClaudeSdkAgentRuntime`, `StaticAuthorizationService`.
- Name booleans as predicates: `allowed`, `isStreaming`, `hasSession`.
- Name functions by the observable action: `resolve`, `can`, `append`, `archive`.
- Avoid vague buckets such as `utils`, `helpers`, or `manager` unless the file is very small and
  local to one module.

## Errors And Logging

- Return user-facing errors from orchestration; keep raw provider errors out of final responses.
- Preserve enough internal detail in logs to debug behavior, but never log API keys, authorization
  headers, access tokens, or refresh tokens.
- Treat `catch (error)` as `unknown` and format it through a single helper when possible.
- Include workspace, user, and message identifiers in runtime logs when they are already available.

## Tests

- Add deterministic unit tests for parsing, routing, authorization, persistence, and formatting.
- Add smoke regression cases for runtime/model behavior in `src/smoke/regressionSmoke.ts`.
- Test observable behavior through ports and public functions.
- Keep tests named after the behavior they protect.
- When a bug is discovered from logs or browser behavior, add the smallest regression that would
  have caught it.

## Review Checklist

- Does the change preserve the ports/adapters boundary?
- Can a reader understand the main path without jumping through several files?
- Are provider-specific details hidden behind an adapter?
- Are invalid states represented by types or validated at the boundary?
- Are errors safe for users and logs safe for secrets?
- Is there a focused unit or smoke regression for changed behavior?
- Do `npm run check` and, for runtime behavior, `npm run smoke` pass?
