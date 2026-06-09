import { CronExpressionParser } from "cron-parser";
import type { ConversationLogger, InboundMessage, MessageGateway } from "../core/index.js";
import type {
  CronJob,
  CronJobResult,
  CronJobStore,
  CronSchedule,
  DeliveryPayload
} from "./cronTypes.js";
import { CompositeDeliveryChannel } from "./delivery/compositeDelivery.js";
import type { CronLeaderLease } from "./cronLeaderLease.js";

export type CronSchedulerDependencies = {
  readonly jobStore: CronJobStore;
  readonly messageHandler: MessageGateway;
  readonly delivery: CompositeDeliveryChannel;
  readonly eventLogger?: ConversationLogger;
  readonly conversationLogger?: ConversationLogger;
  readonly tickIntervalMs?: number;
  readonly staleGraceMs?: number;
  readonly leaderLease?: CronLeaderLease;
  readonly leaderRenewIntervalMs?: number;
};

const DEFAULT_TICK_INTERVAL_MS = 60_000;
const DEFAULT_STALE_GRACE_MS = 300_000; // 5 min
const DEFAULT_TIMEOUT_MS = 120_000;
const DEFAULT_LEADER_RENEW_INTERVAL_MS = 30_000;

export class TickCronScheduler {
  private intervalHandle: ReturnType<typeof setInterval> | undefined;
  private readonly tickIntervalMs: number;
  private readonly staleGraceMs: number;
  private readonly leaderRenewIntervalMs: number;
  private readonly runningJobIds = new Set<string>();
  private readonly runningJobAbortControllers = new Map<string, AbortController>();
  private tickInFlight = false;
  private _isRunning = false;
  private _isLeader = false;
  private leaderRenewHandle: ReturnType<typeof setInterval> | undefined;

  public constructor(private readonly deps: CronSchedulerDependencies) {
    this.tickIntervalMs = deps.tickIntervalMs ?? DEFAULT_TICK_INTERVAL_MS;
    this.staleGraceMs = deps.staleGraceMs ?? DEFAULT_STALE_GRACE_MS;
    this.leaderRenewIntervalMs = deps.leaderRenewIntervalMs ?? DEFAULT_LEADER_RENEW_INTERVAL_MS;
  }

  public get isRunning(): boolean {
    return this._isRunning;
  }

  public get isLeader(): boolean {
    return this._isLeader;
  }

  public start(): void {
    if (this._isRunning) return;
    this._isRunning = true;

    // Run the first tick immediately
    void this.runTick();

    this.intervalHandle = setInterval(() => {
      void this.runTick();
    }, this.tickIntervalMs);

    void this.logEvent("cron.started", { tickIntervalMs: this.tickIntervalMs });
  }

  public stop(): void {
    if (!this._isRunning) return;
    this._isRunning = false;
    if (this.intervalHandle !== undefined) {
      clearInterval(this.intervalHandle);
      this.intervalHandle = undefined;
    }
    this.stopLeaderRenewal();
    void this.deps.leaderLease?.release();
    this._isLeader = false;
    void this.logEvent("cron.stopped", {});
  }

  public async triggerNow(jobId: string): Promise<CronJobResult> {
    const job = await this.deps.jobStore.getById(jobId);
    if (job === undefined) {
      throw new Error(`Cron job not found: ${jobId}`);
    }
    return this.executeJob(job);
  }

  private async runTick(): Promise<void> {
    if (this.tickInFlight) {
      void this.logEvent("cron.tick.skipped", { reason: "tick already in progress" });
      return;
    }

    this.tickInFlight = true;
    try {
      if (!(await this.ensureLeadership())) {
        return;
      }
      await this.tick();
    } finally {
      this.tickInFlight = false;
    }
  }

  private async ensureLeadership(): Promise<boolean> {
    const lease = this.deps.leaderLease;
    if (lease === undefined) {
      return true;
    }

    if (this._isLeader) {
      return true;
    }

    const acquired = await lease.acquire();
    if (!acquired) {
      void this.logEvent("cron.leader.skipped", {
        reason: "another scheduler owns the cron leader lease",
        leasePath: lease.leasePath
      });
      return false;
    }

    this._isLeader = true;
    this.startLeaderRenewal();
    void this.logEvent("cron.leader.acquired", {
      leasePath: lease.leasePath,
      ownerId: lease.ownerId
    });
    return true;
  }

  private startLeaderRenewal(): void {
    if (this.leaderRenewHandle !== undefined || this.deps.leaderLease === undefined) {
      return;
    }

    this.leaderRenewHandle = setInterval(() => {
      void this.renewLeadership();
    }, this.leaderRenewIntervalMs);
  }

  private stopLeaderRenewal(): void {
    if (this.leaderRenewHandle === undefined) {
      return;
    }
    clearInterval(this.leaderRenewHandle);
    this.leaderRenewHandle = undefined;
  }

  private async renewLeadership(): Promise<void> {
    const lease = this.deps.leaderLease;
    if (lease === undefined || !this._isLeader) {
      return;
    }

    const renewed = await lease.renew();
    if (renewed) {
      return;
    }

    this._isLeader = false;
    this.stopLeaderRenewal();
    this.abortRunningJobs("lost leadership");
    void this.logEvent("cron.leader.lost", {
      leasePath: lease.leasePath,
      ownerId: lease.ownerId
    });
  }

  private async tick(): Promise<void> {
    try {
      const jobs = await this.deps.jobStore.getAll();
      const enabledJobs = jobs.filter((j) => j.enabled);
      const now = new Date();

      for (const job of enabledJobs) {
        const nextRunAtStr = await this.deps.jobStore.getNextRunAt(job.id);

        // Initialize nextRunAt if not set
        if (nextRunAtStr === undefined) {
          const computed = this.computeNextRunAt(job.schedule, now);
          await this.deps.jobStore.setNextRunAt(job.id, computed.toISOString());
          continue;
        }

        const nextRunAt = new Date(nextRunAtStr);
        if (nextRunAt > now) continue;

        // Stale fast-forward: if the job is overdue beyond grace window,
        // skip to the next future occurrence instead of firing all missed runs.
        const overdueMs = now.getTime() - nextRunAt.getTime();
        if (overdueMs > this.staleGraceMs) {
          const computed = this.computeNextRunAt(job.schedule, now);
          await this.deps.jobStore.setNextRunAt(job.id, computed.toISOString());
          void this.logEvent("cron.job.skipped", {
            jobId: job.id,
            jobName: job.name,
            overdueMs,
            graceMs: this.staleGraceMs,
            nextRunAt: computed.toISOString()
          });
          continue;
        }

        // At-most-once: pre-advance nextRunAt BEFORE execution
        const nextAfter = this.computeNextRunAt(job.schedule, nextRunAt);
        await this.deps.jobStore.setNextRunAt(job.id, nextAfter.toISOString());

        // For one-shot schedules, disable after pre-advance
        if (job.schedule.type === "once") {
          await this.deps.jobStore.update(job.id, { enabled: false });
        }

        await this.executeJob(job);
      }
    } catch (error) {
      void this.logEvent("cron.tick.error", {
        error: error instanceof Error ? error.message : String(error)
      });
    }
  }

  private async executeJob(job: CronJob): Promise<CronJobResult> {
    const startedAt = new Date().toISOString();

    if (this.hasLostLeadership()) {
      const result = this.createSkippedJobResult(job.id, startedAt, "Lost cron leader lease");
      await this.deps.jobStore.setLastResult(job.id, result);
      void this.logEvent("cron.job.skipped", {
        jobId: job.id,
        jobName: job.name,
        reason: "lost leadership"
      });
      return result;
    }

    if (this.runningJobIds.has(job.id)) {
      const result = this.createSkippedJobResult(job.id, startedAt, "Job is already running");
      await this.deps.jobStore.setLastResult(job.id, result);
      void this.logEvent("cron.job.skipped", {
        jobId: job.id,
        jobName: job.name,
        reason: "job already running"
      });
      return result;
    }

    this.runningJobIds.add(job.id);
    const abortController = new AbortController();
    this.runningJobAbortControllers.set(job.id, abortController);
    void this.logEvent("cron.job.started", {
      jobId: job.id,
      jobName: job.name,
      workspacePath: job.workspacePath
    });

    let output = "";
    let status: CronJobResult["status"] = "success";
    let error: string | undefined;

    try {
      try {
        const inbound = this.createInboundMessage(job);
        const timeoutMs = job.timeoutMs ?? DEFAULT_TIMEOUT_MS;

        const result = await this.withTimeout(
          this.deps.messageHandler.handle(inbound),
          timeoutMs,
          abortController.signal
        );

        output = result.text;

        await this.deps.conversationLogger?.write({
          type: "conversation.turn",
          channel: "cron",
          messageId: inbound.id,
          userId: "cron-scheduler",
          input: job.prompt,
          output,
          workspacePath: job.workspacePath
        });
      } catch (err) {
        if (err instanceof TimeoutError) {
          status = "timeout";
          error = `Job timed out after ${job.timeoutMs ?? DEFAULT_TIMEOUT_MS}ms`;
        } else if (err instanceof LostLeadershipError) {
          status = "skipped";
          error = "Lost cron leader lease";
        } else {
          status = "error";
          error = err instanceof Error ? err.message : String(err);
        }

        void this.logEvent("cron.job.failed", {
          jobId: job.id,
          jobName: job.name,
          status,
          error
        });
      }

      const finishedAt = new Date().toISOString();
      const result: CronJobResult = {
        jobId: job.id,
        startedAt,
        finishedAt,
        status,
        output,
        ...(error !== undefined ? { error } : {})
      };

      await this.deps.jobStore.setLastResult(job.id, result);

      if (status === "skipped" && error === "Lost cron leader lease") {
        void this.logEvent("cron.job.skipped", {
          jobId: job.id,
          jobName: job.name,
          reason: "lost leadership"
        });
        return result;
      }

      void this.logEvent("cron.job.completed", {
        jobId: job.id,
        jobName: job.name,
        status,
        outputLength: output.length
      });

      // Deliver results (unless silent on empty output)
      if (job.silentOnEmpty === true && output.length === 0) {
        void this.logEvent("cron.delivery.skipped", {
          jobId: job.id,
          reason: "empty output"
        });
      } else {
        await this.deliverResult(job, result);
      }

      return result;
    } finally {
      this.runningJobIds.delete(job.id);
      this.runningJobAbortControllers.delete(job.id);
    }
  }

  private async deliverResult(job: CronJob, result: CronJobResult): Promise<void> {
    const payload: DeliveryPayload = {
      jobName: job.name,
      output: result.output,
      ...(result.error !== undefined ? { error: result.error } : {}),
      ...(job.delivery.template !== undefined ? { template: job.delivery.template } : {})
    };

    try {
      await this.deps.delivery.deliverAll(job.delivery.channels, payload);
      void this.logEvent("cron.delivery.sent", {
        jobId: job.id,
        channels: job.delivery.channels
      });
    } catch (error) {
      void this.logEvent("cron.delivery.failed", {
        jobId: job.id,
        channels: job.delivery.channels,
        error: error instanceof Error ? error.message : String(error)
      });
    }
  }

  private createInboundMessage(job: CronJob): InboundMessage {
    return {
      id: `cron-${job.id}-${Date.now()}`,
      channel: "cron",
      user: { id: "cron-scheduler" },
      text: job.prompt,
      receivedAt: new Date(),
      chatId: `job:${job.id}`,
      ...(job.role !== undefined ? { roleOverride: job.role } : {})
    };
  }

  private computeNextRunAt(schedule: CronSchedule, after: Date): Date {
    switch (schedule.type) {
      case "cron": {
        const interval = CronExpressionParser.parse(schedule.expression, { currentDate: after });
        const next = interval.next();
        return next.toDate();
      }
      case "interval": {
        return new Date(after.getTime() + schedule.milliseconds);
      }
      case "once": {
        return new Date(schedule.runAt);
      }
    }
  }

  private hasLostLeadership(): boolean {
    return this.deps.leaderLease !== undefined && !this._isLeader;
  }

  private abortRunningJobs(reason: string): void {
    for (const controller of this.runningJobAbortControllers.values()) {
      controller.abort(reason);
    }
  }

  private createSkippedJobResult(jobId: string, startedAt: string, error: string): CronJobResult {
    return {
      jobId,
      startedAt,
      finishedAt: new Date().toISOString(),
      status: "skipped",
      output: "",
      error
    };
  }

  private async withTimeout<T>(
    promise: Promise<T>,
    timeoutMs: number,
    abortSignal?: AbortSignal
  ): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      const cleanup = (): void => {
        clearTimeout(timer);
        abortSignal?.removeEventListener("abort", handleAbort);
      };
      const handleAbort = (): void => {
        cleanup();
        reject(new LostLeadershipError());
      };
      const timer = setTimeout(() => {
        cleanup();
        reject(new TimeoutError(timeoutMs));
      }, timeoutMs);
      if (abortSignal?.aborted === true) {
        handleAbort();
        return;
      }
      abortSignal?.addEventListener("abort", handleAbort, { once: true });
      promise.then(
        (value) => {
          cleanup();
          resolve(value);
        },
        (error: unknown) => {
          cleanup();
          reject(error instanceof Error ? error : new Error(String(error)));
        }
      );
    });
  }

  private async logEvent(type: string, data: Record<string, unknown>): Promise<void> {
    await this.deps.eventLogger?.write({
      type,
      ...data
    });
  }
}

class TimeoutError extends Error {
  public constructor(timeoutMs: number) {
    super(`Operation timed out after ${timeoutMs}ms`);
    this.name = "TimeoutError";
  }
}

class LostLeadershipError extends Error {
  public constructor() {
    super("Lost cron leader lease");
    this.name = "LostLeadershipError";
  }
}
