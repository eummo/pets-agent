import { describe, expect, it } from "vitest";
import { InMemoryLoopStore } from "./loopStore.js";
import { LoopService } from "./loopService.js";
import type {
  ActionExecutor,
  ActionResult,
  AuthorizationService,
  LoopEventLogger,
  LoopExecutionContext,
  LoopDefinition
} from "./loopTypes.js";

// ── Test Doubles ──────────────────────────────────────────────────────────────

class RecordingActionExecutor implements ActionExecutor {
  public readonly calls: {
    context: LoopExecutionContext;
    action: string;
  }[] = [];
  private readonly responses: ActionResult[];
  private nextResponse = 0;

  public constructor(responses: ActionResult[]) {
    this.responses = responses;
  }

  public execute(
    context: LoopExecutionContext,
    action: string,
    signal: AbortSignal
  ): Promise<ActionResult> {
    // Signal is intentionally unused but required by ActionExecutor contract
    void signal;
    this.calls.push({ context, action });
    const response = this.responses[this.nextResponse];
    if (response === undefined) {
      return Promise.reject(new Error("No more responses"));
    }
    this.nextResponse++;
    return Promise.resolve(response);
  }
}

class RecordingEventLogger implements LoopEventLogger {
  public readonly events: Record<string, unknown>[] = [];

  public write(event: Record<string, unknown>): Promise<void> {
    this.events.push(event);
    return Promise.resolve();
  }
}

class SlowActionExecutor implements ActionExecutor {
  public constructor(
    private readonly delayMs: number,
    private readonly result: ActionResult
  ) {}

  public execute(
    context: LoopExecutionContext,
    action: string,
    signal: AbortSignal
  ): Promise<ActionResult> {
    return new Promise<ActionResult>((resolve, reject) => {
      const timer = setTimeout(() => {
        resolve(this.result);
      }, this.delayMs);

      signal.addEventListener(
        "abort",
        () => {
          clearTimeout(timer);
          reject(new DOMException("Aborted", "AbortError"));
        },
        { once: true }
      );
    });
  }
}

class StubAuthorizationService implements AuthorizationService {
  public constructor(private readonly allowed: boolean) {}
  public roleFor(): Promise<string> { return Promise.resolve("reviewer"); }
  public can(): Promise<{ allowed: boolean; reason?: string }> {
    return Promise.resolve({ allowed: this.allowed });
  }
  public hasCapability(): Promise<boolean> {
    return Promise.resolve(this.allowed);
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

async function setupWithDef(
  executorResponses: ActionResult[],
  definitionOverrides?: {
    name?: string;
    goal?: string;
    workspacePath?: string;
    role?: string;
    maxIterations?: number;
    timeoutMs?: number;
    maxTokenBudget?: number;
    triggerType?: LoopDefinition["triggerType"];
    verificationStrategy?: string;
    authorization?: AuthorizationService;
  }
): Promise<{
  service: LoopService;
  store: InMemoryLoopStore;
  executor: RecordingActionExecutor;
  logger: RecordingEventLogger;
  definitionId: string;
}> {
  const store = new InMemoryLoopStore();
  const executor = new RecordingActionExecutor(executorResponses);
  const logger = new RecordingEventLogger();

  const created = await store.createDefinition({
    name: definitionOverrides?.name ?? "test-loop",
    goal: definitionOverrides?.goal ?? "Check workspace health",
    workspacePath:
      definitionOverrides?.workspacePath ?? "/workspace/test",
    role: definitionOverrides?.role ?? "reviewer",
    maxIterations: definitionOverrides?.maxIterations ?? 3,
    timeoutMs: definitionOverrides?.timeoutMs ?? 60_000,
    maxTokenBudget: definitionOverrides?.maxTokenBudget ?? 100_000,
    triggerType: definitionOverrides?.triggerType ?? "manual",
    verificationStrategy:
      definitionOverrides?.verificationStrategy ?? "file-check"
  });

  const service = new LoopService({
    store,
    actionExecutor: executor,
    eventLogger: logger,
    authorization: definitionOverrides?.authorization,
    ownerId: "test-service"
  });

  return {
    service,
    store,
    executor,
    logger,
    definitionId: created.id
  };
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe("LoopService", () => {
  describe("startRun", () => {
    it("creates run and executes first step", async () => {
      const { service, store, definitionId } = await setupWithDef([
        { output: "Plan: check README", tokenUsage: 50, evidence: "plan-ok" },
        {
          output: "DONE: README is healthy",
          tokenUsage: 100,
          evidence: "readme-exists"
        }
      ]);

      const run = await service.startRun(definitionId, "user-1");

      // Allow async execution to complete
      await new Promise((resolve) => setTimeout(resolve, 50));

      const finalRun = await store.getRun(run.id);
      expect(finalRun).toBeDefined();
      if (finalRun === undefined) return;
      expect(finalRun.status).toBe("completed");

      const steps = await store.getStepsByRun(run.id);
      expect(steps.length).toBeGreaterThanOrEqual(1);
      const firstStep = steps[0];
      expect(firstStep).toBeDefined();
      if (firstStep === undefined) return;
      expect(firstStep.iteration).toBe(1);
      expect(firstStep.attempt).toBe(1);
    });

    it("respects maxIterations limit", async () => {
      const { service, store, definitionId } = await setupWithDef(
        [
          {
            output: "Continue checking",
            tokenUsage: 50,
            evidence: "more-work"
          },
          {
            output: "Continue checking",
            tokenUsage: 50,
            evidence: "more-work"
          },
          {
            output: "Continue checking",
            tokenUsage: 50,
            evidence: "more-work"
          },
          {
            output: "Continue checking",
            tokenUsage: 50,
            evidence: "more-work"
          },
          {
            output: "Continue checking",
            tokenUsage: 50,
            evidence: "more-work"
          },
          {
            output: "Continue checking",
            tokenUsage: 50,
            evidence: "more-work"
          }
        ],
        { maxIterations: 2 }
      );

      const run = await service.startRun(definitionId, "user-1");
      await new Promise((resolve) => setTimeout(resolve, 100));

      const finalRun = await store.getRun(run.id);
      expect(finalRun).toBeDefined();
      if (finalRun === undefined) return;
      expect(finalRun.status).toBe("failed");

      const steps = await store.getStepsByRun(run.id);
      expect(steps.length).toBeLessThanOrEqual(2);
    });

    it("throws for unknown definition", async () => {
      const store = new InMemoryLoopStore();
      const executor = new RecordingActionExecutor([]);
      const service = new LoopService({
        store,
        actionExecutor: executor,
        ownerId: "test-service"
      });

      await expect(
        service.startRun("nonexistent", "user-1")
      ).rejects.toThrow("Loop definition not found");
    });
  });

  describe("cancelRun", () => {
    it("cancels an active run", async () => {
      const store = new InMemoryLoopStore();
      const executor = new SlowActionExecutor(10_000, {
        output: "slow result",
        tokenUsage: 100,
        evidence: "slow"
      });
      const logger = new RecordingEventLogger();

      const created = await store.createDefinition({
        name: "cancel-test",
        goal: "Test cancellation",
        workspacePath: "/ws",
        role: "reviewer",
        maxIterations: 5,
        timeoutMs: 60_000,
        maxTokenBudget: 100_000,
        triggerType: "manual",
        verificationStrategy: "none"
      });

      const service = new LoopService({
        store,
        actionExecutor: executor,
        eventLogger: logger,
        ownerId: "test-service"
      });

      const run = await service.startRun(created.id, "user-1");

      // Cancel immediately
      await service.cancelRun(run.id, "user-1");

      const finalRun = await store.getRun(run.id);
      expect(finalRun).toBeDefined();
      if (finalRun === undefined) return;
      expect(finalRun.status).toBe("cancelled");
    });
  });

  describe("pauseRun", () => {
    it("pauses an active run", async () => {
      const store = new InMemoryLoopStore();
      const executor = new SlowActionExecutor(10_000, {
        output: "slow result",
        tokenUsage: 100,
        evidence: "slow"
      });
      const logger = new RecordingEventLogger();

      const created = await store.createDefinition({
        name: "pause-test",
        goal: "Test pause",
        workspacePath: "/ws",
        role: "reviewer",
        maxIterations: 5,
        timeoutMs: 60_000,
        maxTokenBudget: 100_000,
        triggerType: "manual",
        verificationStrategy: "none"
      });

      const service = new LoopService({
        store,
        actionExecutor: executor,
        eventLogger: logger,
        ownerId: "test-service"
      });

      const run = await service.startRun(created.id, "user-1");

      // Pause immediately
      await service.pauseRun(run.id, "user-1");

      const finalRun = await store.getRun(run.id);
      expect(finalRun).toBeDefined();
      if (finalRun === undefined) return;
      expect(finalRun.status).toBe("paused");
    });
  });

  describe("resumeRun", () => {
    it("resumes a paused run", async () => {
      const { service, store, definitionId } = await setupWithDef([
        // First iteration: outputs pause signal
        {
          output: "Plan: check status",
          tokenUsage: 50,
          evidence: "plan-ok"
        },
        {
          output: "PAUSE: need approval",
          tokenUsage: 100,
          evidence: "needs-approval"
        },
        // After resume: complete
        {
          output: "Plan: resumed",
          tokenUsage: 50,
          evidence: "resumed"
        },
        {
          output: "DONE: approved and complete",
          tokenUsage: 100,
          evidence: "complete"
        }
      ]);

      const run = await service.startRun(definitionId, "user-1");
      await new Promise((resolve) => setTimeout(resolve, 50));

      // Should be paused after first step
      const pausedRun = await store.getRun(run.id);
      expect(pausedRun).toBeDefined();
      if (pausedRun === undefined) return;
      expect(pausedRun.status).toBe("paused");

      // Resume
      await service.resumeRun(run.id);
      await new Promise((resolve) => setTimeout(resolve, 50));

      const finalRun = await store.getRun(run.id);
      expect(finalRun).toBeDefined();
      if (finalRun === undefined) return;
      expect(finalRun.status).toBe("completed");
    });

    it("throws for non-paused run", async () => {
      const { service, store, definitionId } = await setupWithDef([
        {
          output: "Plan: go",
          tokenUsage: 50,
          evidence: "ok"
        },
        {
          output: "DONE: complete",
          tokenUsage: 50,
          evidence: "done"
        }
      ]);

      const run = await service.startRun(definitionId, "user-1");
      await new Promise((resolve) => setTimeout(resolve, 50));

      // Run completed, can't resume
      const finalRun = await store.getRun(run.id);
      expect(finalRun).toBeDefined();
      if (finalRun === undefined) return;
      expect(finalRun.status).toBe("completed");

      await expect(service.resumeRun(run.id)).rejects.toThrow(
        "Cannot resume run in status: completed"
      );
    });
  });

  describe("step phases", () => {
    it("progresses through plan-act-observe-verify-decide", async () => {
      const { service, store, definitionId } = await setupWithDef([
        {
          output: "Plan: run health check",
          tokenUsage: 50,
          evidence: "plan-ok"
        },
        {
          output: "DONE: health check passed",
          tokenUsage: 100,
          evidence: "all-pass"
        }
      ]);

      const run = await service.startRun(definitionId, "user-1");
      await new Promise((resolve) => setTimeout(resolve, 50));

      const steps = await store.getStepsByRun(run.id);
      expect(steps.length).toBeGreaterThanOrEqual(1);

      // Last step should be in decide phase (completed)
      const lastStep = steps[steps.length - 1];
      expect(lastStep).toBeDefined();
      if (lastStep === undefined) return;
      expect(lastStep.phase).toBe("decide");
      expect(lastStep.status).toBe("succeeded");
    });
  });

  describe("execution context propagation", () => {
    it("propagates context to ActionExecutor", async () => {
      const { service, executor, definitionId } =
        await setupWithDef([
          {
            output: "Plan action",
            tokenUsage: 50,
            evidence: "plan"
          },
          {
            output: "DONE: complete",
            tokenUsage: 100,
            evidence: "done"
          }
        ]);

      const run = await service.startRun(definitionId, "user-1");
      await new Promise((resolve) => setTimeout(resolve, 50));

      // Executor should have been called with context
      expect(executor.calls.length).toBeGreaterThanOrEqual(2);

      const firstCall = executor.calls[0];
      expect(firstCall).toBeDefined();
      if (firstCall === undefined) return;
      expect(firstCall.context.requestedBy).toBe("user-1");
      expect(firstCall.context.executionPrincipal).toBe("test-service");
      expect(firstCall.context.loopRunId).toBe(run.id);
      expect(firstCall.context.attempt).toBe(1);
      expect(firstCall.context.idempotencyKey).toContain(
        ":iter:1:attempt:1"
      );

      // The second call (act phase) should also have context
      const secondCall = executor.calls[1];
      expect(secondCall).toBeDefined();
      if (secondCall === undefined) return;
      expect(secondCall.context.loopRunId).toBe(run.id);
      expect(secondCall.context.stepId).toBeTruthy();
    });
  });

  describe("event logging", () => {
    it("logs events with correct types and context", async () => {
      const { service, logger, definitionId } = await setupWithDef([
        {
          output: "Plan action",
          tokenUsage: 50,
          evidence: "plan"
        },
        {
          output: "DONE: complete",
          tokenUsage: 100,
          evidence: "done"
        }
      ]);

      const run = await service.startRun(definitionId, "user-1");
      await new Promise((resolve) => setTimeout(resolve, 50));

      // Should have logged started event
      const startedEvent = logger.events.find(
        (e) => e["type"] === "loop.started"
      );
      expect(startedEvent).toBeDefined();
      if (startedEvent === undefined) return;
      expect(startedEvent["definitionId"]).toBe(definitionId);
      expect(startedEvent["requestedBy"]).toBe("user-1");

      // Should have step phase events
      const stepEvents = logger.events.filter(
        (e) => {
          const t = e["type"];
          return typeof t === "string" && t.startsWith("loop.step.");
        }
      );
      expect(stepEvents.length).toBeGreaterThan(0);

      // Step events should carry execution context
      for (const event of stepEvents) {
        expect(event["loopRunId"]).toBe(run.id);
        expect(event["stepId"]).toBeTruthy();
        expect(event["attempt"]).toBeDefined();
      }

      // Should have completed event
      const completedEvent = logger.events.find(
        (e) => e["type"] === "loop.completed"
      );
      expect(completedEvent).toBeDefined();
    });

    it("run-level events include loopRunId", async () => {
      const { service, logger, definitionId } = await setupWithDef([
        { output: "Plan", tokenUsage: 50, evidence: "plan" },
        { output: "DONE: complete", tokenUsage: 100, evidence: "done" }
      ]);

      const run = await service.startRun(definitionId, "user-1");
      await new Promise((resolve) => setTimeout(resolve, 50));

      const runLevelEvents = logger.events.filter(
        (e) => e["type"] === "loop.started" || e["type"] === "loop.completed"
      );

      for (const event of runLevelEvents) {
        expect(event["loopRunId"]).toBe(run.id);
      }
    });
  });

  describe("retry behavior", () => {
    it("retries on verification failure", async () => {
      const { service, store, definitionId } = await setupWithDef([
        // First attempt: plan succeeds but act returns error
        {
          output: "Plan: check files",
          tokenUsage: 50,
          evidence: "plan"
        },
        {
          output: "ERROR: file not found",
          tokenUsage: 100,
          evidence: "error"
        },
        // Retry: succeeds
        {
          output: "Plan: retry with different path",
          tokenUsage: 50,
          evidence: "plan-retry"
        },
        {
          output: "DONE: found file",
          tokenUsage: 100,
          evidence: "found"
        }
      ]);

      const run = await service.startRun(definitionId, "user-1");
      await new Promise((resolve) => setTimeout(resolve, 100));

      const steps = await store.getStepsByRun(run.id);
      // Should have at least 2 steps (first failed, retry succeeded)
      expect(steps.length).toBeGreaterThanOrEqual(2);

      // First step should have error observation
      const firstStep = steps[0];
      expect(firstStep).toBeDefined();
      if (firstStep === undefined) return;
      expect(firstStep.observation).toContain("ERROR");

      const finalRun = await store.getRun(run.id);
      expect(finalRun).toBeDefined();
      if (finalRun === undefined) return;
      expect(finalRun.status).toBe("completed");
    });
  });

  describe("decision outcomes", () => {
    it("decision complete transitions run to completed", async () => {
      const { service, store, definitionId } = await setupWithDef([
        {
          output: "Plan: final check",
          tokenUsage: 50,
          evidence: "plan"
        },
        {
          output: "DONE: all checks passed",
          tokenUsage: 100,
          evidence: "all-pass"
        }
      ]);

      const run = await service.startRun(definitionId, "user-1");
      await new Promise((resolve) => setTimeout(resolve, 50));

      const finalRun = await store.getRun(run.id);
      expect(finalRun).toBeDefined();
      if (finalRun === undefined) return;
      expect(finalRun.status).toBe("completed");
      expect(finalRun.completedAt).toBeTruthy();
    });

    it("decision fail transitions run to failed", async () => {
      const { service, store, definitionId } = await setupWithDef([
        {
          output: "Plan: check",
          tokenUsage: 50,
          evidence: "plan"
        },
        {
          output: "FAIL: critical error",
          tokenUsage: 100,
          evidence: "critical"
        }
      ]);

      const run = await service.startRun(definitionId, "user-1");
      await new Promise((resolve) => setTimeout(resolve, 50));

      const finalRun = await store.getRun(run.id);
      expect(finalRun).toBeDefined();
      if (finalRun === undefined) return;
      expect(finalRun.status).toBe("failed");
      expect(finalRun.completedAt).toBeTruthy();
    });
  });

  describe("token budget enforcement", () => {
    it("tracks token usage from ActionResult", async () => {
      const { service, store, definitionId } = await setupWithDef([
        { output: "Plan", tokenUsage: 300, evidence: "plan" },
        { output: "DONE: complete", tokenUsage: 500, evidence: "done" }
      ]);

      const run = await service.startRun(definitionId, "user-1");
      await new Promise((resolve) => setTimeout(resolve, 50));

      const finalRun = await store.getRun(run.id);
      expect(finalRun).toBeDefined();
      if (finalRun === undefined) return;
      // 300 + 500 = 800 total from plan + act
      expect(finalRun.cumulativeTokenUsage).toBe(800);
    });

    it("fails when token budget is exceeded", async () => {
      const { service, store, definitionId } = await setupWithDef(
        [
          { output: "Continue", tokenUsage: 600, evidence: "ok" },
          { output: "Continue", tokenUsage: 600, evidence: "ok" },
          { output: "Continue", tokenUsage: 600, evidence: "ok" },
          { output: "Continue", tokenUsage: 600, evidence: "ok" }
        ],
        { maxTokenBudget: 1_000, maxIterations: 10 }
      );

      const run = await service.startRun(definitionId, "user-1");
      await new Promise((resolve) => setTimeout(resolve, 100));

      const finalRun = await store.getRun(run.id);
      expect(finalRun).toBeDefined();
      if (finalRun === undefined) return;
      expect(finalRun.status).toBe("failed");
    });
  });

  describe("timeout enforcement", () => {
    it("fails run when timeoutMs is exceeded", async () => {
      let clockNow = 0;
      const incrementingClock = (): number => {
        clockNow += 10;
        return clockNow;
      };

      const store = new InMemoryLoopStore();
      const executor: ActionExecutor = {
        execute(): Promise<ActionResult> {
          return Promise.resolve({
            output: "Continue working",
            tokenUsage: 50,
            evidence: "more"
          });
        }
      };
      const logger = new RecordingEventLogger();

      const created = await store.createDefinition({
        name: "timeout-test", goal: "test timeout",
        workspacePath: "/ws", role: "reviewer",
        maxIterations: 100, timeoutMs: 25, maxTokenBudget: 100_000,
        triggerType: "manual", verificationStrategy: "none"
      });

      const service = new LoopService({
        store, actionExecutor: executor, eventLogger: logger,
        ownerId: "test-service", clock: incrementingClock
      });

      const run = await service.startRun(created.id, "user-1");
      await new Promise((resolve) => setTimeout(resolve, 100));

      const finalRun = await store.getRun(run.id);
      expect(finalRun).toBeDefined();
      if (finalRun === undefined) return;
      expect(finalRun.status).toBe("failed");
    });
  });

  describe("pre-aborted signal", () => {
    it("immediately cancels run when signal is already aborted", async () => {
      const { service, store, definitionId } = await setupWithDef([
        { output: "Plan: check", tokenUsage: 50, evidence: "plan" },
        { output: "DONE: complete", tokenUsage: 50, evidence: "done" }
      ]);

      const controller = new AbortController();
      controller.abort("pre-cancelled");

      const run = await service.startRun(definitionId, "user-1", controller.signal);
      await new Promise((resolve) => setTimeout(resolve, 50));

      const finalRun = await store.getRun(run.id);
      expect(finalRun).toBeDefined();
      if (finalRun === undefined) return;
      expect(finalRun.status).toBe("cancelled");

      // No steps should have been created
      const steps = await store.getStepsByRun(run.id);
      expect(steps.length).toBe(0);
    });
  });

  describe("terminal state guard", () => {
    it("cancelRun status is not overwritten by async executeRun completion", async () => {
      const store = new InMemoryLoopStore();
      const executor = new SlowActionExecutor(10_000, {
        output: "slow", tokenUsage: 100, evidence: "slow"
      });
      const logger = new RecordingEventLogger();
      const created = await store.createDefinition({
        name: "race-test", goal: "race condition test",
        workspacePath: "/ws", role: "reviewer",
        maxIterations: 5, timeoutMs: 60_000, maxTokenBudget: 100_000,
        triggerType: "manual", verificationStrategy: "none"
      });
      const service = new LoopService({
        store, actionExecutor: executor, eventLogger: logger, ownerId: "test-service"
      });

      const run = await service.startRun(created.id, "user-1");
      // Cancel before async executeRun can complete
      await service.cancelRun(run.id, "user-1");

      // Wait and verify cancelled status sticks
      await new Promise((resolve) => setTimeout(resolve, 50));
      const finalRun = await store.getRun(run.id);
      expect(finalRun).toBeDefined();
      if (finalRun === undefined) return;
      expect(finalRun.status).toBe("cancelled");
    });
  });

  describe("authorization", () => {
    it("startRun throws when loop_manage capability is missing", async () => {
      const { service, definitionId } = await setupWithDef(
        [],
        { authorization: new StubAuthorizationService(false) }
      );
      await expect(service.startRun(definitionId, "user-1")).rejects.toThrow(
        "Insufficient permissions"
      );
    });

    it("cancelRun throws when loop_manage capability is missing", async () => {
      const store = new InMemoryLoopStore();
      const service = new LoopService({
        store,
        actionExecutor: new RecordingActionExecutor([]),
        authorization: new StubAuthorizationService(false),
        ownerId: "test-service"
      });
      await expect(service.cancelRun("run-1", "user-1")).rejects.toThrow(
        "Insufficient permissions"
      );
    });

    it("pauseRun throws when loop_manage capability is missing", async () => {
      const store = new InMemoryLoopStore();
      const service = new LoopService({
        store,
        actionExecutor: new RecordingActionExecutor([]),
        authorization: new StubAuthorizationService(false),
        ownerId: "test-service"
      });
      await expect(service.pauseRun("run-1", "user-1")).rejects.toThrow(
        "Insufficient permissions"
      );
    });

    it("startRun succeeds with loop_manage capability", async () => {
      const { service, store, definitionId } = await setupWithDef(
        [
          { output: "Plan", tokenUsage: 50, evidence: "plan" },
          { output: "DONE: ok", tokenUsage: 50, evidence: "done" }
        ],
        { authorization: new StubAuthorizationService(true) }
      );
      const run = await service.startRun(definitionId, "user-1");
      await new Promise((resolve) => setTimeout(resolve, 50));
      const finalRun = await store.getRun(run.id);
      expect(finalRun?.status).toBe("completed");
    });

    it("startRun succeeds without authorization configured (backward compat)", async () => {
      const { service, store, definitionId } = await setupWithDef([
        { output: "Plan", tokenUsage: 50, evidence: "plan" },
        { output: "DONE: ok", tokenUsage: 50, evidence: "done" }
      ]);
      // No authorization configured — should work as before
      const run = await service.startRun(definitionId, "user-1");
      await new Promise((resolve) => setTimeout(resolve, 50));
      const finalRun = await store.getRun(run.id);
      expect(finalRun?.status).toBe("completed");
    });
  });
});
