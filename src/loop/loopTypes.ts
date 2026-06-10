import { z } from "zod";

// ── Status Types ──────────────────────────────────────────────────────────────

export type LoopRunStatus =
  | "queued"
  | "running"
  | "paused"
  | "completed"
  | "blocked"
  | "failed"
  | "cancelled";

export type LoopStepStatus =
  | "queued"
  | "running"
  | "succeeded"
  | "failed"
  | "interrupted";

export type LoopStepPhase =
  | "plan"
  | "act"
  | "observe"
  | "verify"
  | "decide";

export type LoopTriggerType = "manual" | "cron" | "api";

// ── Decision ──────────────────────────────────────────────────────────────────

export const loopDecisionSchema = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("complete"), reason: z.string().min(1) }),
  z.object({ kind: z.literal("continue"), nextAction: z.string().min(1) }),
  z.object({ kind: z.literal("pause"), reason: z.string().min(1) }),
  z.object({ kind: z.literal("fail"), reason: z.string().min(1) }),
  z.object({
    kind: z.literal("retry"),
    reason: z.string().min(1),
    changedStrategy: z.boolean()
  })
]);

export type LoopDecision = z.infer<typeof loopDecisionSchema>;

// ── Execution Context ─────────────────────────────────────────────────────────
// Flows through gateway, runtime, tools, and logs for correlation and idempotency.

export type LoopExecutionContext = {
  readonly loopRunId: string;
  readonly stepId: string;
  readonly attempt: number;
  readonly idempotencyKey: string;
  readonly requestedBy: string;
  readonly executionPrincipal: string;
  readonly authorizedPolicyVersion: string;
};

// ── Definition ────────────────────────────────────────────────────────────────
// Template describing what a loop does, not a specific run.

export const loopDefinitionSchema = z.object({
  id: z.string().min(1),
  name: z.string().min(1),
  goal: z.string().min(1),
  workspacePath: z.string().min(1),
  role: z.string().min(1),
  maxIterations: z.number().int().positive(),
  timeoutMs: z.number().int().positive(),
  maxTokenBudget: z.number().int().positive(),
  triggerType: z.enum(["manual", "cron", "api"]),
  verificationStrategy: z.string().min(1),
  createdAt: z.string().min(1),
  updatedAt: z.string().min(1)
});

export type LoopDefinition = z.infer<typeof loopDefinitionSchema>;

// ── Run ───────────────────────────────────────────────────────────────────────
// A single execution instance of a LoopDefinition.

export const loopRunSchema = z.object({
  id: z.string().min(1),
  definitionId: z.string().min(1),
  status: z.enum([
    "queued",
    "running",
    "paused",
    "completed",
    "blocked",
    "failed",
    "cancelled"
  ]),
  requestedBy: z.string().min(1),
  executionPrincipal: z.string().min(1),
  authorizedPolicyVersion: z.string().min(1),
  currentIteration: z.number().int().nonnegative(),
  cumulativeTokenUsage: z.number().int().nonnegative(),
  startedAt: z.string().min(1),
  completedAt: z.string().nullable(),
  lastStepId: z.string().nullable(),
  checkpoint: z.string().nullable()
});

export type LoopRun = z.infer<typeof loopRunSchema>;

// ── Step ──────────────────────────────────────────────────────────────────────
// A single step within a run, tracking one pass through plan-act-observe-verify-decide.

export const loopStepSchema = z.object({
  id: z.string().min(1),
  runId: z.string().min(1),
  iteration: z.number().int().positive(),
  attempt: z.number().int().positive(),
  status: z.enum(["queued", "running", "succeeded", "failed", "interrupted"]),
  phase: z.enum(["plan", "act", "observe", "verify", "decide"]),
  idempotencyKey: z.string().min(1),
  claimOwner: z.string().nullable(),
  leaseExpiry: z.string().nullable(),
  actionDescription: z.string().nullable(),
  observation: z.string().nullable(),
  decision: loopDecisionSchema.nullable(),
  startedAt: z.string().min(1),
  completedAt: z.string().nullable()
});

export type LoopStep = z.infer<typeof loopStepSchema>;

// ── Action Executor Contract ──────────────────────────────────────────────────
// Abstract seam between LoopService and concrete execution (MessageGateway, etc.)

export type ActionResult = {
  readonly output: string;
  readonly tokenUsage: number;
  readonly evidence: string;
};

export type ActionExecutor = {
  execute(
    context: LoopExecutionContext,
    action: string,
    signal: AbortSignal
  ): Promise<ActionResult>;
};

// ── Store Contract ────────────────────────────────────────────────────────────

export type LoopStore = {
  // Definitions
  getDefinition(id: string): Promise<LoopDefinition | undefined>;
  createDefinition(
    definition: Omit<LoopDefinition, "id" | "createdAt" | "updatedAt">
  ): Promise<LoopDefinition>;
  deleteDefinition(id: string): Promise<boolean>;

  // Runs
  getRun(id: string): Promise<LoopRun | undefined>;
  getActiveRunForDefinition(
    definitionId: string
  ): Promise<LoopRun | undefined>;
  createRun(
    run: Omit<LoopRun, "id" | "startedAt">
  ): Promise<LoopRun>;
  updateRun(
    id: string,
    patch: Partial<Omit<LoopRun, "id" | "definitionId" | "startedAt">>
  ): Promise<LoopRun | undefined>;

  // Steps
  getStep(id: string): Promise<LoopStep | undefined>;
  getStepsByRun(runId: string): Promise<readonly LoopStep[]>;
  createStep(
    step: Omit<LoopStep, "id" | "startedAt">
  ): Promise<LoopStep>;
  updateStep(
    id: string,
    patch: Partial<Omit<LoopStep, "id" | "runId" | "startedAt">>
  ): Promise<LoopStep | undefined>;
  claimStep(
    id: string,
    owner: string,
    leaseExpiry: string
  ): Promise<LoopStep | undefined>;

  // Recovery
  getExpiredRunningSteps(olderThan: string): Promise<readonly LoopStep[]>;
};

// ── Event Logger Contract ─────────────────────────────────────────────────────

export type LoopEventLogger = {
  write(event: Record<string, unknown>): Promise<void>;
};
