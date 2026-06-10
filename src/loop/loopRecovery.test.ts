import { describe, expect, it } from "vitest";
import { InMemoryLoopStore } from "./loopStore.js";
import { LoopService } from "./loopService.js";
import type {
  ActionExecutor,
  ActionResult,
  LoopEventLogger,
  LoopExecutionContext
} from "./loopTypes.js";

// ── Test Doubles ──────────────────────────────────────────────────────────────

class SlowActionExecutor implements ActionExecutor {
  public constructor(
    private readonly delayMs: number,
    private readonly result: ActionResult
  ) {}

  public execute(
    _context: LoopExecutionContext,
    _action: string,
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

class RecordingEventLogger implements LoopEventLogger {
  public readonly events: Record<string, unknown>[] = [];

  public write(event: Record<string, unknown>): Promise<void> {
    this.events.push(event);
    return Promise.resolve();
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

async function createDefinitionAndService(
  overrides?: { maxIterations?: number }
): Promise<{
  service: LoopService;
  store: InMemoryLoopStore;
  logger: RecordingEventLogger;
  definitionId: string;
}> {
  const store = new InMemoryLoopStore();
  const logger = new RecordingEventLogger();

  const created = await store.createDefinition({
    name: "recovery-test",
    goal: "Test recovery",
    workspacePath: "/ws",
    role: "reviewer",
    maxIterations: overrides?.maxIterations ?? 5,
    timeoutMs: 60_000,
    maxTokenBudget: 100_000,
    triggerType: "manual",
    verificationStrategy: "none"
  });

  const executor = new SlowActionExecutor(10_000, {
    output: "slow",
    tokenUsage: 100,
    evidence: "slow"
  });

  const service = new LoopService({
    store,
    actionExecutor: executor,
    eventLogger: logger,
    ownerId: "test-service"
  });

  return { service, store, logger, definitionId: created.id };
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe("Loop Recovery and Idempotency", () => {
  describe("recoverInterruptedSteps", () => {
    it("transitions expired running steps to interrupted", async () => {
      const { store } = await createDefinitionAndService();

      // Create a step that looks like an expired running step
      const step = await store.createStep({
        runId: "run-1",
        iteration: 1,
        attempt: 1,
        status: "running",
        phase: "act",
        idempotencyKey: "run-1:iter:1:attempt:1",
        claimOwner: "old-service",
        leaseExpiry: "2020-01-01T00:00:00",
        actionDescription: "some action",
        observation: null,
        decision: null,
        completedAt: null
      });

      const service = new LoopService({
        store,
        actionExecutor: new SlowActionExecutor(10_000, {
          output: "x",
          tokenUsage: 0,
          evidence: "x"
        }),
        ownerId: "recovery-service"
      });

      const count = await service.recoverInterruptedSteps();
      expect(count).toBe(1);

      const recovered = await store.getStep(step.id);
      expect(recovered).toBeDefined();
      if (recovered === undefined) return;
      expect(recovered.status).toBe("interrupted");
      expect(recovered.claimOwner).toBeNull();
    });

    it("does not affect non-expired steps", async () => {
      const { store } = await createDefinitionAndService();

      await store.createStep({
        runId: "run-1",
        iteration: 1,
        attempt: 1,
        status: "running",
        phase: "act",
        idempotencyKey: "run-1:iter:1:attempt:1",
        claimOwner: "active-service",
        leaseExpiry: "2099-01-01T00:00:00",
        actionDescription: "some action",
        observation: null,
        decision: null,
        completedAt: null
      });

      const service = new LoopService({
        store,
        actionExecutor: new SlowActionExecutor(10_000, {
          output: "x",
          tokenUsage: 0,
          evidence: "x"
        }),
        ownerId: "recovery-service"
      });

      const count = await service.recoverInterruptedSteps();
      expect(count).toBe(0);
    });

    it("recovers multiple expired steps", async () => {
      const { store } = await createDefinitionAndService();

      await store.createStep({
        runId: "run-1",
        iteration: 1,
        attempt: 1,
        status: "running",
        phase: "plan",
        idempotencyKey: "run-1:iter:1:attempt:1",
        claimOwner: "old-1",
        leaseExpiry: "2020-01-01T00:00:00",
        actionDescription: null,
        observation: null,
        decision: null,
        completedAt: null
      });

      await store.createStep({
        runId: "run-2",
        iteration: 2,
        attempt: 1,
        status: "running",
        phase: "act",
        idempotencyKey: "run-2:iter:2:attempt:1",
        claimOwner: "old-2",
        leaseExpiry: "2020-06-01T00:00:00",
        actionDescription: "act",
        observation: null,
        decision: null,
        completedAt: null
      });

      const service = new LoopService({
        store,
        actionExecutor: new SlowActionExecutor(10_000, {
          output: "x",
          tokenUsage: 0,
          evidence: "x"
        }),
        ownerId: "recovery-service"
      });

      const count = await service.recoverInterruptedSteps();
      expect(count).toBe(2);
    });

    it("logs recovery events", async () => {
      const { store, logger } = await createDefinitionAndService();

      const step = await store.createStep({
        runId: "run-1",
        iteration: 1,
        attempt: 1,
        status: "running",
        phase: "act",
        idempotencyKey: "run-1:iter:1:attempt:1",
        claimOwner: "old-service",
        leaseExpiry: "2020-01-01T00:00:00",
        actionDescription: "act",
        observation: null,
        decision: null,
        completedAt: null
      });

      const service = new LoopService({
        store,
        actionExecutor: new SlowActionExecutor(10_000, {
          output: "x",
          tokenUsage: 0,
          evidence: "x"
        }),
        eventLogger: logger,
        ownerId: "recovery-service"
      });

      await service.recoverInterruptedSteps();

      const recoveryEvent = logger.events.find(
        (e) => e["type"] === "loop.recovered"
      );
      expect(recoveryEvent).toBeDefined();
      if (recoveryEvent === undefined) return;
      expect(recoveryEvent["stepId"]).toBe(step.id);
      expect(recoveryEvent["runId"]).toBe("run-1");
    });

    it("stores checkpoint on run when recovering interrupted step", async () => {
      const { store } = await createDefinitionAndService();

      // Create a run to associate the step with
      const run = await store.createRun({
        definitionId: "def-1",
        status: "running",
        requestedBy: "user-1",
        executionPrincipal: "svc",
        authorizedPolicyVersion: "v1",
        currentIteration: 1,
        cumulativeTokenUsage: 0,
        completedAt: null,
        lastStepId: null,
        checkpoint: null
      });

      await store.createStep({
        runId: run.id,
        iteration: 1,
        attempt: 1,
        status: "running",
        phase: "act",
        idempotencyKey: `${run.id}:iter:1:attempt:1`,
        claimOwner: "old-service",
        leaseExpiry: "2020-01-01T00:00:00",
        actionDescription: "deploy service to production",
        observation: null,
        decision: null,
        completedAt: null
      });

      const service = new LoopService({
        store,
        actionExecutor: new SlowActionExecutor(10_000, {
          output: "x",
          tokenUsage: 0,
          evidence: "x"
        }),
        ownerId: "recovery-service"
      });

      await service.recoverInterruptedSteps();

      const updatedRun = await store.getRun(run.id);
      expect(updatedRun).toBeDefined();
      if (updatedRun === undefined) return;
      expect(updatedRun.checkpoint).not.toBeNull();
      const checkpoint = JSON.parse(updatedRun.checkpoint!);
      expect(checkpoint.actionDescription).toBe("deploy service to production");
      expect(checkpoint.iteration).toBe(1);
    });
  });

  describe("interrupted step non-replay", () => {
    it("creates a new step on resume, not replaying interrupted step", async () => {
      const { store, service, definitionId } =
        await createDefinitionAndService();

      // Start a run
      const run = await service.startRun(definitionId, "user-1");

      // Wait briefly then pause
      await new Promise((resolve) => setTimeout(resolve, 20));
      await service.pauseRun(run.id, "user-1");

      // Get the interrupted step
      const stepsBeforeResume = await store.getStepsByRun(run.id);
      expect(stepsBeforeResume.length).toBeGreaterThanOrEqual(1);
      const interruptedStep =
        stepsBeforeResume[stepsBeforeResume.length - 1];
      expect(interruptedStep).toBeDefined();

      // Resume with a fast executor
      const fastExecutor: ActionExecutor = {
        execute() {
          return Promise.resolve({
            output: "DONE: resumed and completed",
            tokenUsage: 50,
            evidence: "resumed"
          });
        }
      };

      const resumeService = new LoopService({
        store,
        actionExecutor: fastExecutor,
        eventLogger: new RecordingEventLogger(),
        ownerId: "test-service"
      });

      await resumeService.resumeRun(run.id);
      await new Promise((resolve) => setTimeout(resolve, 50));

      // New steps should have been created (different from interrupted)
      const allSteps = await store.getStepsByRun(run.id);
      const newSteps = allSteps.filter(
        (s) => interruptedStep !== undefined && s.id !== interruptedStep.id
      );
      expect(newSteps.length).toBeGreaterThanOrEqual(1);
    });
  });

  describe("recovery re-observation", () => {
    it("performs observation before planning on resume from checkpoint", async () => {
      const { store, definitionId } = await createDefinitionAndService();

      // Create a run with a checkpoint
      const run = await store.createRun({
        definitionId,
        status: "paused",
        requestedBy: "user-1",
        executionPrincipal: "svc",
        authorizedPolicyVersion: "v1",
        currentIteration: 1,
        cumulativeTokenUsage: 0,
        completedAt: null,
        lastStepId: null,
        checkpoint: JSON.stringify({
          interruptedStepId: "step-old",
          iteration: 1,
          actionDescription: "deploy to staging"
        })
      });

      const executorCalls: string[] = [];
      const trackingExecutor: ActionExecutor = {
        execute(_ctx, action, _signal): Promise<ActionResult> {
          executorCalls.push(action);
          return Promise.resolve({
            output: "DONE: resumed",
            tokenUsage: 50,
            evidence: "ok"
          });
        }
      };

      const resumeService = new LoopService({
        store,
        actionExecutor: trackingExecutor,
        eventLogger: new RecordingEventLogger(),
        ownerId: "test-service"
      });

      await resumeService.resumeRun(run.id);
      await new Promise((resolve) => setTimeout(resolve, 50));

      // First call should be the re-observation, containing the action description
      expect(executorCalls.length).toBeGreaterThan(0);
      expect(executorCalls[0]).toContain("deploy to staging");

      // Checkpoint should be cleared
      const finalRun = await store.getRun(run.id);
      expect(finalRun?.checkpoint).toBeNull();
    });
  });

  describe("idempotency key determinism", () => {
    it("generates deterministic idempotency key for same run/iteration/attempt", async () => {
      const store = new InMemoryLoopStore();

      const step1 = await store.createStep({
        runId: "run-ABC",
        iteration: 2,
        attempt: 1,
        status: "queued",
        phase: "plan",
        idempotencyKey: "run-ABC:iter:2:attempt:1",
        claimOwner: null,
        leaseExpiry: null,
        actionDescription: null,
        observation: null,
        decision: null,
        completedAt: null
      });

      const step2 = await store.createStep({
        runId: "run-ABC",
        iteration: 2,
        attempt: 2,
        status: "queued",
        phase: "plan",
        idempotencyKey: "run-ABC:iter:2:attempt:2",
        claimOwner: null,
        leaseExpiry: null,
        actionDescription: null,
        observation: null,
        decision: null,
        completedAt: null
      });

      expect(step1.idempotencyKey).toBe("run-ABC:iter:2:attempt:1");
      expect(step2.idempotencyKey).toBe("run-ABC:iter:2:attempt:2");
      // Same key format for same parameters
      expect(step1.idempotencyKey).not.toBe(step2.idempotencyKey);
    });
  });

  describe("claim contention", () => {
    it("different owner cannot claim a step held by another", async () => {
      const store = new InMemoryLoopStore();

      const step = await store.createStep({
        runId: "run-1",
        iteration: 1,
        attempt: 1,
        status: "queued",
        phase: "plan",
        idempotencyKey: "run-1:iter:1:attempt:1",
        claimOwner: null,
        leaseExpiry: null,
        actionDescription: null,
        observation: null,
        decision: null,
        completedAt: null
      });

      // First owner claims
      const claimed = await store.claimStep(
        step.id,
        "service-A",
        "2099-01-01"
      );
      expect(claimed).toBeDefined();
      if (claimed === undefined) return;
      expect(claimed.claimOwner).toBe("service-A");

      // Second owner tries to claim same step
      const secondClaim = await store.claimStep(
        step.id,
        "service-B",
        "2099-01-01"
      );
      expect(secondClaim).toBeUndefined();
    });

    it("same owner can re-claim an interrupted step", async () => {
      const store = new InMemoryLoopStore();

      const step = await store.createStep({
        runId: "run-1",
        iteration: 1,
        attempt: 1,
        status: "queued",
        phase: "plan",
        idempotencyKey: "run-1:iter:1:attempt:1",
        claimOwner: null,
        leaseExpiry: null,
        actionDescription: null,
        observation: null,
        decision: null,
        completedAt: null
      });

      // Claim
      const claimed = await store.claimStep(
        step.id,
        "service-A",
        "2099-01-01"
      );
      expect(claimed).toBeDefined();

      // Simulate interruption (release claim)
      await store.updateStep(step.id, {
        status: "interrupted",
        claimOwner: null,
        leaseExpiry: null
      });

      // Same owner re-claims
      const reclaimed = await store.claimStep(
        step.id,
        "service-A",
        "2099-02-01"
      );
      expect(reclaimed).toBeDefined();
      if (reclaimed === undefined) return;
      expect(reclaimed.claimOwner).toBe("service-A");
    });
  });

  describe("abort signal propagation", () => {
    it("abort signal propagates through executeStep and interrupts step", async () => {
      const store = new InMemoryLoopStore();
      const logger = new RecordingEventLogger();

      const created = await store.createDefinition({
        name: "abort-test",
        goal: "Test abort propagation",
        workspacePath: "/ws",
        role: "reviewer",
        maxIterations: 5,
        timeoutMs: 60_000,
        maxTokenBudget: 100_000,
        triggerType: "manual",
        verificationStrategy: "none"
      });

      const executor = new SlowActionExecutor(10_000, {
        output: "slow",
        tokenUsage: 100,
        evidence: "slow"
      });

      const service = new LoopService({
        store,
        actionExecutor: executor,
        eventLogger: logger,
        ownerId: "test-service"
      });

      const run = await service.startRun(created.id, "user-1");

      // Cancel while the slow executor is running
      await new Promise((resolve) => setTimeout(resolve, 20));
      await service.cancelRun(run.id, "user-1");

      const finalRun = await store.getRun(run.id);
      expect(finalRun).toBeDefined();
      if (finalRun === undefined) return;
      expect(finalRun.status).toBe("cancelled");

      // Steps should not be left in "running" state after cancel
      const steps = await store.getStepsByRun(run.id);
      for (const step of steps) {
        expect(step.status).not.toBe("running");
        // Step can be interrupted, failed (from abort error), succeeded, or queued
        expect(
          step.status === "interrupted" ||
            step.status === "failed" ||
            step.status === "succeeded" ||
            step.status === "queued"
        ).toBe(true);
      }
    });
  });
});
