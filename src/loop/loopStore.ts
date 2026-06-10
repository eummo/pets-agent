import { toLocalIsoString } from "../logging/jsonlLogger.js";
import type {
  LoopDefinition,
  LoopRun,
  LoopRunStatus,
  LoopStep,
  LoopStepStatus,
  LoopStore
} from "./loopTypes.js";

// ── ID Generation ─────────────────────────────────────────────────────────────

function randomSuffix(): string {
  return Math.random().toString(36).slice(2, 10);
}

function definitionIdFromName(name: string): string {
  const slug = name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 32);
  return `loop-def-${slug}-${randomSuffix()}`;
}

// ── Active Statuses ───────────────────────────────────────────────────────────

const ACTIVE_RUN_STATUSES: ReadonlySet<LoopRunStatus> = new Set([
  "queued",
  "running",
  "paused",
  "blocked"
]);

const CLAIMABLE_STEP_STATUSES: ReadonlySet<LoopStepStatus> = new Set([
  "queued",
  "interrupted"
]);

// ── InMemoryLoopStore ─────────────────────────────────────────────────────────

export class InMemoryLoopStore implements LoopStore {
  private readonly definitions = new Map<string, LoopDefinition>();
  private readonly runs = new Map<string, LoopRun>();
  private readonly steps = new Map<string, LoopStep>();

  // ── Definitions ───────────────────────────────────────────────────────────

  public getDefinition(
    id: string
  ): Promise<LoopDefinition | undefined> {
    return Promise.resolve(this.definitions.get(id));
  }

  public createDefinition(
    input: Omit<LoopDefinition, "id" | "createdAt" | "updatedAt">
  ): Promise<LoopDefinition> {
    const now = toLocalIsoString(new Date());
    const definition: LoopDefinition = {
      ...input,
      id: definitionIdFromName(input.name),
      createdAt: now,
      updatedAt: now
    };
    this.definitions.set(definition.id, definition);
    return Promise.resolve(definition);
  }

  public deleteDefinition(id: string): Promise<boolean> {
    return Promise.resolve(this.definitions.delete(id));
  }

  // ── Runs ──────────────────────────────────────────────────────────────────

  public getRun(id: string): Promise<LoopRun | undefined> {
    return Promise.resolve(this.runs.get(id));
  }

  public getActiveRunForDefinition(
    definitionId: string
  ): Promise<LoopRun | undefined> {
    for (const run of this.runs.values()) {
      if (
        run.definitionId === definitionId &&
        ACTIVE_RUN_STATUSES.has(run.status)
      ) {
        return Promise.resolve(run);
      }
    }
    return Promise.resolve(undefined);
  }

  public createRun(
    input: Omit<LoopRun, "id" | "startedAt">
  ): Promise<LoopRun> {
    const run: LoopRun = {
      ...input,
      id: `run-${randomSuffix()}`,
      startedAt: toLocalIsoString(new Date()),
      completedAt: input.completedAt ?? null,
      lastStepId: input.lastStepId ?? null,
      checkpoint: input.checkpoint ?? null
    };
    this.runs.set(run.id, run);
    return Promise.resolve(run);
  }

  public updateRun(
    id: string,
    patch: Partial<
      Omit<LoopRun, "id" | "definitionId" | "startedAt">
    >
  ): Promise<LoopRun | undefined> {
    const existing = this.runs.get(id);
    if (existing === undefined) {
      return Promise.resolve(undefined);
    }
    const updated: LoopRun = { ...existing, ...patch };
    this.runs.set(id, updated);
    return Promise.resolve(updated);
  }

  // ── Steps ─────────────────────────────────────────────────────────────────

  public getStep(id: string): Promise<LoopStep | undefined> {
    return Promise.resolve(this.steps.get(id));
  }

  public getStepsByRun(runId: string): Promise<readonly LoopStep[]> {
    const result: LoopStep[] = [];
    for (const step of this.steps.values()) {
      if (step.runId === runId) {
        result.push(step);
      }
    }
    return Promise.resolve(result);
  }

  public createStep(
    input: Omit<LoopStep, "id" | "startedAt">
  ): Promise<LoopStep> {
    const step: LoopStep = {
      ...input,
      id: `step-${randomSuffix()}`,
      startedAt: toLocalIsoString(new Date()),
      claimOwner: input.claimOwner ?? null,
      leaseExpiry: input.leaseExpiry ?? null,
      actionDescription: input.actionDescription ?? null,
      observation: input.observation ?? null,
      decision: input.decision ?? null,
      completedAt: input.completedAt ?? null
    };
    this.steps.set(step.id, step);
    return Promise.resolve(step);
  }

  public updateStep(
    id: string,
    patch: Partial<
      Omit<LoopStep, "id" | "runId" | "startedAt">
    >
  ): Promise<LoopStep | undefined> {
    const existing = this.steps.get(id);
    if (existing === undefined) {
      return Promise.resolve(undefined);
    }
    const updated: LoopStep = { ...existing, ...patch };
    this.steps.set(id, updated);
    return Promise.resolve(updated);
  }

  public claimStep(
    id: string,
    owner: string,
    leaseExpiry: string
  ): Promise<LoopStep | undefined> {
    const step = this.steps.get(id);
    if (step === undefined) {
      return Promise.resolve(undefined);
    }
    if (!CLAIMABLE_STEP_STATUSES.has(step.status)) {
      return Promise.resolve(undefined);
    }
    if (step.claimOwner != null && step.claimOwner !== owner) {
      return Promise.resolve(undefined);
    }
    const updated: LoopStep = {
      ...step,
      status: "running",
      claimOwner: owner,
      leaseExpiry
    };
    this.steps.set(id, updated);
    return Promise.resolve(updated);
  }

  // ── Recovery ──────────────────────────────────────────────────────────────

  public getExpiredRunningSteps(
    olderThan: string
  ): Promise<readonly LoopStep[]> {
    const result: LoopStep[] = [];
    for (const step of this.steps.values()) {
      if (
        step.status === "running" &&
        step.leaseExpiry !== null &&
        step.leaseExpiry < olderThan
      ) {
        result.push(step);
      }
    }
    return Promise.resolve(result);
  }
}
