import { describe, expect, it } from "vitest";
import { InMemoryLoopStore } from "./loopStore.js";
import type { LoopRunStatus, LoopStepStatus } from "./loopTypes.js";

// ── Helpers ───────────────────────────────────────────────────────────────────

function sampleDefinitionInput() {
  return {
    name: "test-loop",
    goal: "Check workspace health",
    workspacePath: "/workspace/test",
    role: "reviewer",
    maxIterations: 3,
    timeoutMs: 60_000,
    maxTokenBudget: 10_000,
    triggerType: "manual" as const,
    verificationStrategy: "file-check"
  };
}

function sampleRunInput(overrides?: {
  definitionId?: string;
  status?: LoopRunStatus;
}) {
  return {
    definitionId: overrides?.definitionId ?? "def-1",
    status: overrides?.status ?? "queued",
    requestedBy: "user-1",
    executionPrincipal: "loop-service",
    authorizedPolicyVersion: "v1",
    currentIteration: 0,
    cumulativeTokenUsage: 0,
    completedAt: null as string | null,
    lastStepId: null as string | null,
    checkpoint: null as string | null
  };
}

function sampleStepInput(overrides?: {
  runId?: string;
  iteration?: number;
  attempt?: number;
  status?: LoopStepStatus;
}) {
  return {
    runId: overrides?.runId ?? "run-1",
    iteration: overrides?.iteration ?? 1,
    attempt: overrides?.attempt ?? 1,
    status: overrides?.status ?? "queued",
    phase: "plan" as const,
    idempotencyKey: "run-1:iter:1:attempt:1",
    claimOwner: null as string | null,
    leaseExpiry: null as string | null,
    actionDescription: null as string | null,
    observation: null as string | null,
    decision: null as
      | {
          kind: "complete";
          reason: string;
        }
      | null,
    completedAt: null as string | null
  };
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe("InMemoryLoopStore", () => {
  describe("definitions", () => {
    it("creates definition with generated ID and timestamps", async () => {
      const store = new InMemoryLoopStore();
      const def = await store.createDefinition(sampleDefinitionInput());

      expect(def.id).toMatch(/^loop-def-test-loop-/);
      expect(def.createdAt).toBeTruthy();
      expect(def.updatedAt).toBeTruthy();
      expect(def.name).toBe("test-loop");
      expect(def.goal).toBe("Check workspace health");
    });

    it("returns created definition by id", async () => {
      const store = new InMemoryLoopStore();
      const created = await store.createDefinition(sampleDefinitionInput());

      const found = await store.getDefinition(created.id);
      expect(found).toEqual(created);
    });

    it("returns undefined for unknown definition id", async () => {
      const store = new InMemoryLoopStore();
      const found = await store.getDefinition("nonexistent");
      expect(found).toBeUndefined();
    });

    it("deletes a definition and returns true", async () => {
      const store = new InMemoryLoopStore();
      const created = await store.createDefinition(sampleDefinitionInput());

      const deleted = await store.deleteDefinition(created.id);
      expect(deleted).toBe(true);

      const found = await store.getDefinition(created.id);
      expect(found).toBeUndefined();
    });

    it("returns false when deleting unknown definition", async () => {
      const store = new InMemoryLoopStore();
      const deleted = await store.deleteDefinition("nonexistent");
      expect(deleted).toBe(false);
    });
  });

  describe("runs", () => {
    it("creates run with generated ID and startedAt", async () => {
      const store = new InMemoryLoopStore();
      const run = await store.createRun(sampleRunInput());

      expect(run.id).toMatch(/^run-/);
      expect(run.startedAt).toBeTruthy();
      expect(run.status).toBe("queued");
      expect(run.definitionId).toBe("def-1");
    });

    it("returns created run by id", async () => {
      const store = new InMemoryLoopStore();
      const created = await store.createRun(sampleRunInput());

      const found = await store.getRun(created.id);
      expect(found).toEqual(created);
    });

    it("returns undefined for unknown run id", async () => {
      const store = new InMemoryLoopStore();
      const found = await store.getRun("nonexistent");
      expect(found).toBeUndefined();
    });

    it("getActiveRunForDefinition returns running run", async () => {
      const store = new InMemoryLoopStore();
      const created = await store.createRun(
        sampleRunInput({ definitionId: "def-A", status: "running" })
      );

      const active = await store.getActiveRunForDefinition("def-A");
      expect(active).toEqual(created);
    });

    it("getActiveRunForDefinition returns queued run", async () => {
      const store = new InMemoryLoopStore();
      await store.createRun(
        sampleRunInput({ definitionId: "def-A", status: "queued" })
      );

      const active = await store.getActiveRunForDefinition("def-A");
      expect(active).toBeDefined();
    });

    it("getActiveRunForDefinition returns paused run", async () => {
      const store = new InMemoryLoopStore();
      await store.createRun(
        sampleRunInput({ definitionId: "def-A", status: "paused" })
      );

      const active = await store.getActiveRunForDefinition("def-A");
      expect(active).toBeDefined();
    });

    it("getActiveRunForDefinition returns blocked run", async () => {
      const store = new InMemoryLoopStore();
      await store.createRun(
        sampleRunInput({ definitionId: "def-A", status: "blocked" })
      );

      const active = await store.getActiveRunForDefinition("def-A");
      expect(active).toBeDefined();
    });

    it("getActiveRunForDefinition ignores completed run", async () => {
      const store = new InMemoryLoopStore();
      await store.createRun(
        sampleRunInput({ definitionId: "def-A", status: "completed" })
      );

      const active = await store.getActiveRunForDefinition("def-A");
      expect(active).toBeUndefined();
    });

    it("getActiveRunForDefinition ignores failed run", async () => {
      const store = new InMemoryLoopStore();
      await store.createRun(
        sampleRunInput({ definitionId: "def-A", status: "failed" })
      );

      const active = await store.getActiveRunForDefinition("def-A");
      expect(active).toBeUndefined();
    });

    it("getActiveRunForDefinition ignores cancelled run", async () => {
      const store = new InMemoryLoopStore();
      await store.createRun(
        sampleRunInput({ definitionId: "def-A", status: "cancelled" })
      );

      const active = await store.getActiveRunForDefinition("def-A");
      expect(active).toBeUndefined();
    });

    it("updateRun applies partial patch", async () => {
      const store = new InMemoryLoopStore();
      const created = await store.createRun(sampleRunInput());

      const updated = await store.updateRun(created.id, {
        status: "running",
        currentIteration: 1
      });

      expect(updated).toBeDefined();
      if (updated === undefined) return;
      expect(updated.status).toBe("running");
      expect(updated.currentIteration).toBe(1);
      expect(updated.id).toBe(created.id);
      expect(updated.definitionId).toBe(created.definitionId);
    });

    it("updateRun returns undefined for unknown run", async () => {
      const store = new InMemoryLoopStore();
      const result = await store.updateRun("nonexistent", { status: "running" });
      expect(result).toBeUndefined();
    });
  });

  describe("steps", () => {
    it("creates step with generated ID and startedAt", async () => {
      const store = new InMemoryLoopStore();
      const step = await store.createStep(sampleStepInput());

      expect(step.id).toMatch(/^step-/);
      expect(step.startedAt).toBeTruthy();
      expect(step.status).toBe("queued");
    });

    it("returns created step by id", async () => {
      const store = new InMemoryLoopStore();
      const created = await store.createStep(sampleStepInput());

      const found = await store.getStep(created.id);
      expect(found).toEqual(created);
    });

    it("getStepsByRun returns steps for given run", async () => {
      const store = new InMemoryLoopStore();
      await store.createStep(sampleStepInput({ runId: "run-A" }));
      await store.createStep(sampleStepInput({ runId: "run-A" }));
      await store.createStep(sampleStepInput({ runId: "run-B" }));

      const steps = await store.getStepsByRun("run-A");
      expect(steps).toHaveLength(2);
      expect(steps.every((s) => s.runId === "run-A")).toBe(true);
    });

    it("getStepsByRun returns empty for unknown run", async () => {
      const store = new InMemoryLoopStore();
      const steps = await store.getStepsByRun("nonexistent");
      expect(steps).toHaveLength(0);
    });

    it("updateStep applies partial patch", async () => {
      const store = new InMemoryLoopStore();
      const created = await store.createStep(sampleStepInput());

      const updated = await store.updateStep(created.id, {
        status: "running",
        phase: "act",
        actionDescription: "Run health check"
      });

      expect(updated).toBeDefined();
      if (updated === undefined) return;
      expect(updated.status).toBe("running");
      expect(updated.phase).toBe("act");
      expect(updated.actionDescription).toBe("Run health check");
      expect(updated.id).toBe(created.id);
    });

    it("updateStep returns undefined for unknown step", async () => {
      const store = new InMemoryLoopStore();
      const result = await store.updateStep("nonexistent", { status: "running" });
      expect(result).toBeUndefined();
    });
  });

  describe("claimStep", () => {
    it("claims a queued step", async () => {
      const store = new InMemoryLoopStore();
      const created = await store.createStep(
        sampleStepInput({ status: "queued" })
      );

      const claimed = await store.claimStep(created.id, "owner-A", "2099-01-01");
      expect(claimed).toBeDefined();
      if (claimed === undefined) return;
      expect(claimed.status).toBe("running");
      expect(claimed.claimOwner).toBe("owner-A");
      expect(claimed.leaseExpiry).toBe("2099-01-01");
    });

    it("claims an interrupted step", async () => {
      const store = new InMemoryLoopStore();
      const created = await store.createStep(
        sampleStepInput({ status: "interrupted" })
      );

      const claimed = await store.claimStep(created.id, "owner-A", "2099-01-01");
      expect(claimed).toBeDefined();
      if (claimed === undefined) return;
      expect(claimed.status).toBe("running");
      expect(claimed.claimOwner).toBe("owner-A");
    });

    it("rejects claim on running step with different owner", async () => {
      const store = new InMemoryLoopStore();
      const created = await store.createStep(
        sampleStepInput({ status: "queued" })
      );

      const first = await store.claimStep(created.id, "owner-A", "2099-01-01");
      expect(first).toBeDefined();

      const second = await store.claimStep(created.id, "owner-B", "2099-01-01");
      expect(second).toBeUndefined();
    });

    it("rejects claim on succeeded step", async () => {
      const store = new InMemoryLoopStore();
      const created = await store.createStep(
        sampleStepInput({ status: "succeeded" })
      );

      const claimed = await store.claimStep(created.id, "owner-A", "2099-01-01");
      expect(claimed).toBeUndefined();
    });

    it("rejects claim on failed step", async () => {
      const store = new InMemoryLoopStore();
      const created = await store.createStep(
        sampleStepInput({ status: "failed" })
      );

      const claimed = await store.claimStep(created.id, "owner-A", "2099-01-01");
      expect(claimed).toBeUndefined();
    });

    it("rejects claim on unknown step", async () => {
      const store = new InMemoryLoopStore();
      const claimed = await store.claimStep("nonexistent", "owner-A", "2099-01-01");
      expect(claimed).toBeUndefined();
    });

    it("allows same owner to re-claim interrupted step", async () => {
      const store = new InMemoryLoopStore();
      const created = await store.createStep(
        sampleStepInput({ status: "queued" })
      );

      const first = await store.claimStep(created.id, "owner-A", "2099-01-01");
      expect(first).toBeDefined();

      // Simulate interruption
      await store.updateStep(created.id, {
        status: "interrupted",
        claimOwner: null,
        leaseExpiry: null
      });

      const reclaimed = await store.claimStep(
        created.id,
        "owner-A",
        "2099-02-01"
      );
      expect(reclaimed).toBeDefined();
      if (reclaimed === undefined) return;
      expect(reclaimed.status).toBe("running");
      expect(reclaimed.claimOwner).toBe("owner-A");
    });
  });

  describe("getExpiredRunningSteps", () => {
    it("returns running steps with past leaseExpiry", async () => {
      const store = new InMemoryLoopStore();
      await store.createStep({
        ...sampleStepInput({ status: "running" }),
        claimOwner: "owner-A",
        leaseExpiry: "2020-01-01"
      });

      const expired = await store.getExpiredRunningSteps("2025-01-01");
      expect(expired).toHaveLength(1);
    });

    it("excludes steps with future leaseExpiry", async () => {
      const store = new InMemoryLoopStore();
      await store.createStep({
        ...sampleStepInput({ status: "running" }),
        claimOwner: "owner-A",
        leaseExpiry: "2099-01-01"
      });

      const expired = await store.getExpiredRunningSteps("2025-01-01");
      expect(expired).toHaveLength(0);
    });

    it("excludes non-running steps even with past leaseExpiry", async () => {
      const store = new InMemoryLoopStore();
      await store.createStep({
        ...sampleStepInput({ status: "queued" }),
        claimOwner: null,
        leaseExpiry: "2020-01-01"
      });

      const expired = await store.getExpiredRunningSteps("2025-01-01");
      expect(expired).toHaveLength(0);
    });

    it("excludes steps with null leaseExpiry", async () => {
      const store = new InMemoryLoopStore();
      await store.createStep({
        ...sampleStepInput({ status: "running" }),
        claimOwner: "owner-A",
        leaseExpiry: null
      });

      const expired = await store.getExpiredRunningSteps("2025-01-01");
      expect(expired).toHaveLength(0);
    });

    it("returns multiple expired steps", async () => {
      const store = new InMemoryLoopStore();
      await store.createStep({
        ...sampleStepInput({ status: "running", runId: "run-1" }),
        claimOwner: "owner-A",
        leaseExpiry: "2020-01-01"
      });
      await store.createStep({
        ...sampleStepInput({ status: "running", runId: "run-2" }),
        claimOwner: "owner-B",
        leaseExpiry: "2020-06-01"
      });

      const expired = await store.getExpiredRunningSteps("2025-01-01");
      expect(expired).toHaveLength(2);
    });
  });
});
