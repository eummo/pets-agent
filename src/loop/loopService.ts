import { toLocalIsoString } from "../logging/jsonlLogger.js";
import type { AuthorizationService } from "../auth/index.js";
import { createLoopEvent } from "./loopEventLogger.js";
import type {
  ActionExecutor,
  ActionResult,
  LoopDecision,
  LoopDefinition,
  LoopEventLogger,
  LoopExecutionContext,
  LoopRun,
  LoopRunStatus,
  LoopStep,
  LoopStepPhase,
  LoopStore
} from "./loopTypes.js";

// ── Dependencies ──────────────────────────────────────────────────────────────

export type LoopServiceDependencies = {
  readonly store: LoopStore;
  readonly actionExecutor: ActionExecutor;
  readonly eventLogger?: LoopEventLogger;
  readonly authorization?: AuthorizationService;
  readonly stepLeaseDurationMs?: number;
  readonly clock?: () => number;
  readonly ownerId: string;
};

// ── Constants ─────────────────────────────────────────────────────────────────

const DEFAULT_STEP_LEASE_DURATION_MS = 300_000; // 5 minutes
const MAX_RETRY_ATTEMPTS_PER_ITERATION = 3;

// ── Idempotency Key ───────────────────────────────────────────────────────────

function stepIdempotencyKey(
  runId: string,
  iteration: number,
  attempt: number
): string {
  return `${runId}:iter:${iteration}:attempt:${attempt}`;
}

// ── LoopService ───────────────────────────────────────────────────────────────

export class LoopService {
  private readonly store: LoopStore;
  private readonly actionExecutor: ActionExecutor;
  private readonly eventLogger: LoopEventLogger | undefined;
  private readonly authorization: AuthorizationService | undefined;
  private readonly stepLeaseDurationMs: number;
  private readonly clock: () => number;
  private readonly ownerId: string;
  private readonly activeRunControllers = new Map<string, AbortController>();

  public constructor(private readonly deps: LoopServiceDependencies) {
    this.store = deps.store;
    this.actionExecutor = deps.actionExecutor;
    this.eventLogger = deps.eventLogger;
    this.authorization = deps.authorization;
    this.stepLeaseDurationMs =
      deps.stepLeaseDurationMs ?? DEFAULT_STEP_LEASE_DURATION_MS;
    this.clock = deps.clock ?? Date.now;
    this.ownerId = deps.ownerId;
  }

  // ── Public API ────────────────────────────────────────────────────────────

  private async requireCapability(userId: string): Promise<void> {
    if (this.authorization === undefined) {
      return; // No authorization configured — backward compatible
    }
    const hasPermission = await this.authorization.hasCapability(
      { id: userId },
      "loop_manage"
    );
    if (!hasPermission) {
      throw new Error("Insufficient permissions: missing loop_manage capability.");
    }
  }

  public async startRun(
    definitionId: string,
    requestedBy: string,
    signal?: AbortSignal
  ): Promise<LoopRun> {
    await this.requireCapability(requestedBy);
    const definition = await this.store.getDefinition(definitionId);
    if (definition === undefined) {
      throw new Error(`Loop definition not found: ${definitionId}`);
    }

    const run = await this.store.createRun({
      definitionId,
      status: "queued",
      requestedBy,
      executionPrincipal: this.ownerId,
      authorizedPolicyVersion: "initial",
      currentIteration: 0,
      cumulativeTokenUsage: 0,
      completedAt: null,
      lastStepId: null,
      checkpoint: null
    });

    const controller = new AbortController();
    this.activeRunControllers.set(run.id, controller);

    // Forward external signal to internal controller
    if (signal !== undefined) {
      if (signal.aborted) {
        controller.abort(signal.reason);
      } else {
        signal.addEventListener(
          "abort",
          () => controller.abort(signal.reason),
          { once: true }
        );
      }
    }

    // If signal was already aborted before we start, go straight to cancelled
    if (signal?.aborted === true) {
      await this.transitionRunStatus(run.id, "cancelled");
      this.activeRunControllers.delete(run.id);
      const updatedRun = await this.store.getRun(run.id);
      return updatedRun ?? run;
    }

    await this.logEvent("loop.started", undefined, {
      definitionId,
      requestedBy
    }, run.id);

    await this.transitionRunStatus(run.id, "running");

    // Execute asynchronously — don't await to allow cancel/pause interaction
    void this.executeRun(run.id, definition, controller.signal).catch(
      (error: unknown) => {
        // Unhandled errors transition run to failed
        void this.handleExecutionError(run.id, error);
      }
    );

    // Return the run in its current state
    const updatedRun = await this.store.getRun(run.id);
    return updatedRun ?? run;
  }

  public async cancelRun(runId: string, requestedBy: string): Promise<void> {
    await this.requireCapability(requestedBy);
    const controller = this.activeRunControllers.get(runId);
    if (controller !== undefined) {
      controller.abort("cancelled");
    }
    await this.transitionRunStatus(runId, "cancelled");
    this.activeRunControllers.delete(runId);
  }

  public async pauseRun(runId: string, requestedBy: string): Promise<void> {
    await this.requireCapability(requestedBy);
    const controller = this.activeRunControllers.get(runId);
    if (controller !== undefined) {
      controller.abort("paused");
    }
    await this.transitionRunStatus(runId, "paused");
    this.activeRunControllers.delete(runId);
  }

  public async resumeRun(
    runId: string,
    requestedBy?: string,
    signal?: AbortSignal
  ): Promise<LoopRun> {
    const run = await this.store.getRun(runId);
    if (run === undefined) {
      throw new Error(`Loop run not found: ${runId}`);
    }
    if (run.status !== "paused") {
      throw new Error(`Cannot resume run in status: ${run.status}`);
    }

    const effectiveRequestedBy = requestedBy ?? run.requestedBy;
    await this.requireCapability(effectiveRequestedBy);

    const definition = await this.store.getDefinition(run.definitionId);
    if (definition === undefined) {
      throw new Error(
        `Loop definition not found: ${run.definitionId}`
      );
    }

    const controller = new AbortController();
    this.activeRunControllers.set(runId, controller);

    if (signal !== undefined) {
      if (signal.aborted) {
        controller.abort(signal.reason);
      } else {
        signal.addEventListener(
          "abort",
          () => controller.abort(signal.reason),
          { once: true }
        );
      }
    }

    await this.transitionRunStatus(runId, "running");

    void this.executeRun(runId, definition, controller.signal).catch(
      (error: unknown) => {
        void this.handleExecutionError(runId, error);
      }
    );

    const updatedRun = await this.store.getRun(runId);
    return updatedRun ?? run;
  }

  public async recoverInterruptedSteps(): Promise<number> {
    const now = toLocalIsoString(new Date());
    const expired =
      await this.store.getExpiredRunningSteps(now);

    let recovered = 0;
    for (const step of expired) {
      const updated = await this.store.updateStep(step.id, {
        status: "interrupted",
        claimOwner: null,
        leaseExpiry: null,
        completedAt: now
      });

      if (updated !== undefined) {
        recovered++;
        await this.logEvent("loop.recovered", undefined, {
          stepId: step.id,
          runId: step.runId,
          iteration: step.iteration,
          attempt: step.attempt
        }, step.runId);

        // Store checkpoint on the run for re-observation on resume
        await this.store.updateRun(step.runId, {
          checkpoint: JSON.stringify({
            interruptedStepId: step.id,
            iteration: step.iteration,
            actionDescription: step.actionDescription
          })
        });
      }
    }

    return recovered;
  }

  // ── Run Execution ─────────────────────────────────────────────────────────

  private async executeRun(
    runId: string,
    definition: LoopDefinition,
    signal: AbortSignal
  ): Promise<void> {
    let run = await this.store.getRun(runId);
    if (run === undefined) {
      return;
    }

    const startTime = this.clock();
    let iteration = run.currentIteration + 1;
    let attempt = 1;

    // Re-observe external state if resuming from an interrupted checkpoint
    if (run.checkpoint !== null) {
      try {
        const checkpoint = JSON.parse(run.checkpoint);
        if (typeof checkpoint.actionDescription === "string") {
          await this.actionExecutor.execute(
            {
              loopRunId: runId,
              stepId: "recovery-observe",
              attempt: 0,
              idempotencyKey: `${runId}:recovery-observe`,
              requestedBy: run.requestedBy,
              executionPrincipal: run.executionPrincipal,
              authorizedPolicyVersion: run.authorizedPolicyVersion
            },
            `observe: check results of previous action: ${checkpoint.actionDescription}`,
            signal
          );
        }
      } catch {
        // Non-fatal: observation failure should not prevent run execution
      }
      // Clear checkpoint after observation attempt
      await this.store.updateRun(runId, { checkpoint: null });
      run = (await this.store.getRun(runId)) ?? run;
    }

    while (run.status === "running") {
      // Check timeout
      if (this.clock() - startTime >= definition.timeoutMs) {
        await this.transitionRunStatus(runId, "failed");
        await this.logEvent("loop.completed", undefined, {
          reason: "Timeout exceeded"
        }, runId);
        break;
      }

      // Check iteration limit
      if (iteration > definition.maxIterations) {
        await this.transitionRunStatus(runId, "failed");
        await this.logEvent("loop.completed", undefined, {
          reason: "max iterations reached"
        }, runId);
        break;
      }

      // Check token budget
      const budgetDecision = this.checkBudget(run, definition);
      if (budgetDecision !== undefined) {
        await this.handleDecision(runId, budgetDecision);
        break;
      }

      // Check abort before creating step
      if (this.isAborted(signal)) {
        return; // Status already set by cancel/pause
      }

      const idempotencyKey = stepIdempotencyKey(runId, iteration, attempt);
      const step = await this.store.createStep({
        runId,
        iteration,
        attempt,
        status: "queued",
        phase: "plan",
        idempotencyKey,
        claimOwner: null,
        leaseExpiry: null,
        actionDescription: null,
        observation: null,
        decision: null,
        completedAt: null
      });

      // Claim the step
      const leaseExpiry = toLocalIsoString(
        new Date(this.clock() + this.stepLeaseDurationMs)
      );
      const claimed = await this.store.claimStep(
        step.id,
        this.ownerId,
        leaseExpiry
      );
      if (claimed === undefined) {
        // Could not claim — another instance took it
        attempt++;
        if (attempt > MAX_RETRY_ATTEMPTS_PER_ITERATION) {
          await this.transitionRunStatus(runId, "failed");
          break;
        }
        continue;
      }

      // Execute the step
      const { decision, tokenUsage: stepTokenUsage } = await this.executeStep(
        claimed,
        run,
        definition,
        signal
      );

      // Update run with step results
      run =
        (await this.store.updateRun(runId, {
          lastStepId: claimed.id,
          currentIteration: iteration,
          cumulativeTokenUsage: run.cumulativeTokenUsage + stepTokenUsage
        })) ?? run;

      // Handle the decision
      switch (decision.kind) {
        case "complete":
          await this.handleDecision(runId, decision);
          return;
        case "continue":
          iteration++;
          attempt = 1;
          break;
        case "pause":
          await this.handleDecision(runId, decision);
          return;
        case "fail":
          await this.handleDecision(runId, decision);
          return;
        case "retry":
          attempt++;
          if (attempt > MAX_RETRY_ATTEMPTS_PER_ITERATION) {
            await this.transitionRunStatus(runId, "failed");
            await this.logEvent("loop.completed", undefined, {
              reason: "max retry attempts exceeded"
            }, runId);
            return;
          }
          break;
      }
    }
  }

  // ── Step Execution ────────────────────────────────────────────────────────

  private isAborted(signal: AbortSignal): boolean {
    return signal.aborted;
  }

  private async executeStep(
    step: LoopStep,
    run: LoopRun,
    definition: LoopDefinition,
    signal: AbortSignal
  ): Promise<{ decision: LoopDecision; tokenUsage: number }> {
    const context = this.createExecutionContext(run, step);
    let stepTokenUsage = 0;

    try {
      // Phase: plan
      await this.transitionStepPhase(step.id, "plan");
      const planResult = await this.actionExecutor.execute(
        context,
        definition.goal,
        signal
      );
      stepTokenUsage += planResult.tokenUsage;
      await this.store.updateStep(step.id, {
        actionDescription: planResult.output
      });
      await this.logEvent("loop.step.planned", context, {
        action: planResult.output
      });

      if (this.isAborted(signal)) {
        const aborted = await this.abortStep(step.id);
        return { decision: aborted, tokenUsage: stepTokenUsage };
      }

      // Phase: act
      await this.transitionStepPhase(step.id, "act");
      const actResult = await this.actionExecutor.execute(
        context,
        planResult.output,
        signal
      );
      stepTokenUsage += actResult.tokenUsage;
      await this.store.updateStep(step.id, {
        observation: actResult.output
      });
      await this.logEvent("loop.step.acted", context, {
        evidence: actResult.evidence
      });

      if (this.isAborted(signal)) {
        const aborted = await this.abortStep(step.id);
        return { decision: aborted, tokenUsage: stepTokenUsage };
      }

      // Phase: observe
      await this.transitionStepPhase(step.id, "observe");
      await this.logEvent("loop.step.observed", context, {
        observation: actResult.output
      });

      if (this.isAborted(signal)) {
        const aborted = await this.abortStep(step.id);
        return { decision: aborted, tokenUsage: stepTokenUsage };
      }

      // Phase: verify
      await this.transitionStepPhase(step.id, "verify");
      const verificationPassed =
        actResult.output.length > 0 && !actResult.output.includes("ERROR:");
      await this.logEvent("loop.step.verified", context, {
        passed: verificationPassed,
        evidence: actResult.evidence
      });

      if (this.isAborted(signal)) {
        const aborted = await this.abortStep(step.id);
        return { decision: aborted, tokenUsage: stepTokenUsage };
      }

      // Phase: decide
      await this.transitionStepPhase(step.id, "decide");
      const decision = this.makeDecision(
        actResult,
        verificationPassed
      );
      await this.store.updateStep(step.id, {
        decision,
        status: "succeeded",
        completedAt: toLocalIsoString(new Date())
      });
      await this.logEvent("loop.step.decided", context, {
        decision: decision.kind,
        reason:
          decision.kind === "continue"
            ? decision.nextAction
            : decision.reason
      });

      return { decision, tokenUsage: stepTokenUsage };
    } catch (error: unknown) {
      // Step-level error -> mark step failed, return retry decision
      await this.store.updateStep(step.id, {
        status: "failed",
        completedAt: toLocalIsoString(new Date())
      });
      await this.logEvent("loop.step.failed", context, {
        error: error instanceof Error ? error.message : String(error)
      });

      return {
        decision: {
          kind: "retry",
          reason: error instanceof Error ? error.message : String(error),
          changedStrategy: false
        },
        tokenUsage: 0
      };
    }
  }

  // ── Decision Making ───────────────────────────────────────────────────────

  private makeDecision(
    actResult: ActionResult,
    verificationPassed: boolean
  ): LoopDecision {
    // If verification failed, retry
    if (!verificationPassed) {
      return {
        kind: "retry",
        reason: `Verification failed: ${actResult.evidence}`,
        changedStrategy: false
      };
    }

    // If output contains completion signal, complete
    if (
      actResult.output.includes("DONE:") ||
      actResult.output.includes("TASK COMPLETE")
    ) {
      return {
        kind: "complete",
        reason: actResult.output
      };
    }

    // If output contains pause signal, pause
    if (actResult.output.includes("PAUSE:")) {
      return {
        kind: "pause",
        reason: actResult.output
      };
    }

    // If output contains explicit fail signal, fail
    if (actResult.output.includes("FAIL:")) {
      return {
        kind: "fail",
        reason: actResult.output
      };
    }

    // Otherwise, continue to next iteration
    return {
      kind: "continue",
      nextAction: actResult.output
    };
  }

  private async handleDecision(
    runId: string,
    decision: LoopDecision
  ): Promise<void> {
    switch (decision.kind) {
      case "complete":
        await this.transitionRunStatus(runId, "completed");
        await this.logEvent("loop.completed", undefined, {
          reason: decision.reason
        }, runId);
        break;
      case "pause":
        await this.transitionRunStatus(runId, "paused");
        await this.logEvent("loop.paused", undefined, {
          reason: decision.reason
        }, runId);
        break;
      case "fail":
        await this.transitionRunStatus(runId, "failed");
        await this.logEvent("loop.completed", undefined, {
          reason: decision.reason
        }, runId);
        break;
      default:
        break;
    }
    this.activeRunControllers.delete(runId);
  }

  // ── Budget ────────────────────────────────────────────────────────────────

  private checkBudget(
    run: LoopRun,
    definition: LoopDefinition
  ): LoopDecision | undefined {
    if (run.cumulativeTokenUsage >= definition.maxTokenBudget) {
      return {
        kind: "fail",
        reason: `Token budget exceeded: ${run.cumulativeTokenUsage} >= ${definition.maxTokenBudget}`
      };
    }
    return undefined;
  }

  // ── State Transitions ─────────────────────────────────────────────────────

  private static isTerminalStatus(status: LoopRunStatus): boolean {
    return status === "completed" || status === "failed" || status === "cancelled";
  }

  private async transitionRunStatus(
    runId: string,
    newStatus: LoopRunStatus
  ): Promise<void> {
    const current = await this.store.getRun(runId);
    if (current !== undefined && LoopService.isTerminalStatus(current.status)) {
      return; // Already in a terminal state — do not overwrite
    }
    const patch: Partial<LoopRun> = { status: newStatus };
    if (LoopService.isTerminalStatus(newStatus)) {
      patch.completedAt = toLocalIsoString(new Date());
    }
    await this.store.updateRun(runId, patch);
  }

  private async transitionStepPhase(
    stepId: string,
    phase: LoopStepPhase
  ): Promise<void> {
    await this.store.updateStep(stepId, { phase });
  }

  private async abortStep(stepId: string): Promise<LoopDecision> {
    await this.store.updateStep(stepId, {
      status: "interrupted",
      completedAt: toLocalIsoString(new Date())
    });
    await this.logEvent("loop.step.interrupted", undefined, {
      stepId
    });
    return { kind: "pause", reason: "Aborted by signal" };
  }

  // ── Context ───────────────────────────────────────────────────────────────

  private createExecutionContext(
    run: LoopRun,
    step: LoopStep
  ): LoopExecutionContext {
    return {
      loopRunId: run.id,
      stepId: step.id,
      attempt: step.attempt,
      idempotencyKey: step.idempotencyKey,
      requestedBy: run.requestedBy,
      executionPrincipal: run.executionPrincipal,
      authorizedPolicyVersion: run.authorizedPolicyVersion
    };
  }

  // ── Logging ───────────────────────────────────────────────────────────────

  private async logEvent(
    type: string,
    context: LoopExecutionContext | undefined,
    data?: Record<string, unknown>,
    runId?: string
  ): Promise<void> {
    if (this.eventLogger === undefined) {
      return;
    }
    const event =
      context !== undefined
        ? createLoopEvent(
            type as "loop.started",
            context,
            data
          )
        : { type, ...(runId !== undefined ? { loopRunId: runId } : {}), ...data };
    await this.eventLogger.write(event);
  }

  // ── Error Handling ────────────────────────────────────────────────────────

  private async handleExecutionError(
    runId: string,
    error: unknown
  ): Promise<void> {
    await this.transitionRunStatus(runId, "failed");
    await this.logEvent("loop.completed", undefined, {
      reason: `Unhandled error: ${error instanceof Error ? error.message : String(error)}`
    }, runId);
    this.activeRunControllers.delete(runId);
  }
}
