import type { LoopExecutionContext } from "./loopTypes.js";

// ── Event Types ───────────────────────────────────────────────────────────────

export type LoopEventType =
  | "loop.started"
  | "loop.step.planned"
  | "loop.step.acted"
  | "loop.step.observed"
  | "loop.step.verified"
  | "loop.step.decided"
  | "loop.step.interrupted"
  | "loop.step.failed"
  | "loop.verified"
  | "loop.completed"
  | "loop.paused"
  | "loop.cancelled"
  | "loop.recovered";

// ── Event Factory ─────────────────────────────────────────────────────────────
// Ensures every loop event carries the full execution context for traceability.

export function createLoopEvent(
  type: LoopEventType,
  context: LoopExecutionContext,
  data?: Record<string, unknown>
): Record<string, unknown> {
  return {
    type,
    loopRunId: context.loopRunId,
    stepId: context.stepId,
    attempt: context.attempt,
    idempotencyKey: context.idempotencyKey,
    requestedBy: context.requestedBy,
    executionPrincipal: context.executionPrincipal,
    policyVersion: context.authorizedPolicyVersion,
    ...data
  };
}
