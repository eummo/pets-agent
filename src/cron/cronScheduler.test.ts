import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import type { CronJobStore, CronJob, CronJobResult } from "./cronTypes.js";
import { TickCronScheduler } from "./cronScheduler.js";
import type { MessageGateway, OutboundMessage } from "../core/index.js";
import type { CompositeDeliveryChannel } from "./delivery/compositeDelivery.js";
import type { CronLeaderLease } from "./cronLeaderLease.js";

function makeJob(overrides: Partial<CronJob> = {}): CronJob {
  return {
    id: "test-job",
    name: "Test Job",
    schedule: { type: "cron", expression: "0 9 * * *" },
    prompt: "Summarize changes",
    workspacePath: "/workspace/default",
    enabled: true,
    delivery: { channels: ["sse:admin"] },
    createdAt: "2026-05-28T10:00:00Z",
    updatedAt: "2026-05-28T10:00:00Z",
    ...overrides
  };
}

type MockFn = ReturnType<typeof vi.fn>;

function createMockStore(jobs: CronJob[] = []): CronJobStore & {
  getAllMock: MockFn;
  setLastResultMock: MockFn;
  setNextRunAtMock: MockFn;
  updateMock: MockFn;
} {
  const jobMap = new Map(jobs.map((j) => [j.id, { ...j }]));
  const runStateMap = new Map<string, { nextRunAt?: string; lastResult?: CronJobResult }>();

  for (const job of jobs) {
    runStateMap.set(job.id, {});
  }

  const setNextRunAtMock = vi.fn((id: string, nextRunAt: string) => {
    const state = runStateMap.get(id) ?? {};
    state.nextRunAt = nextRunAt;
    runStateMap.set(id, state);
    return Promise.resolve();
  });

  const updateMock = vi.fn((id: string, patch: Partial<CronJob>) => {
    const existing = jobMap.get(id);
    if (!existing) return Promise.resolve(undefined);
    const updated: CronJob = { ...existing, ...patch, id, updatedAt: new Date().toISOString() };
    jobMap.set(id, updated);
    return Promise.resolve(updated);
  });
  const getAllMock = vi.fn(() => Promise.resolve(Array.from(jobMap.values())));
  const setLastResultMock = vi.fn((id: string, result: CronJobResult) => {
    const state = runStateMap.get(id) ?? {};
    state.lastResult = result;
    runStateMap.set(id, state);
    return Promise.resolve();
  });

  return {
    getAll: getAllMock,
    getById: vi.fn((id: string) => Promise.resolve(jobMap.get(id))),
    create: vi.fn((data) => {
      const job = {
        ...data,
        id: "generated-id",
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      } as CronJob;
      jobMap.set(job.id, job);
      runStateMap.set(job.id, {});
      return Promise.resolve(job);
    }),
    update: updateMock,
    delete: vi.fn((id: string) => {
      const existed = jobMap.has(id);
      jobMap.delete(id);
      runStateMap.delete(id);
      return Promise.resolve(existed);
    }),
    getNextRunAt: vi.fn((id: string) => Promise.resolve(runStateMap.get(id)?.nextRunAt)),
    setNextRunAt: setNextRunAtMock,
    getLastResult: vi.fn((id: string) => Promise.resolve(runStateMap.get(id)?.lastResult)),
    setLastResult: setLastResultMock,
    getAllMock,
    setLastResultMock,
    setNextRunAtMock,
    updateMock
  };
}

function createMockGateway(
  response: OutboundMessage = { text: "Done" }
): MessageGateway & { handleMock: MockFn } {
  const handleMock = vi.fn(() => Promise.resolve(response));
  return { handle: handleMock, handleMock };
}

function createMockDelivery(): CompositeDeliveryChannel & { deliverAllMock: MockFn } {
  const deliverAllMock = vi.fn(() => Promise.resolve());
  return {
    prefix: "",
    deliver: vi.fn(() => Promise.resolve()),
    deliverAll: deliverAllMock,
    deliverAllMock
  } as unknown as CompositeDeliveryChannel & { deliverAllMock: MockFn };
}

function createMockLeaderLease(acquireResult: boolean): CronLeaderLease & {
  acquireMock: MockFn;
  renewMock: MockFn;
  releaseMock: MockFn;
} {
  const acquireMock = vi.fn(() => Promise.resolve(acquireResult));
  const renewMock = vi.fn(() => Promise.resolve(acquireResult));
  const releaseMock = vi.fn(() => Promise.resolve());

  return {
    leasePath: "/tmp/cron.leader",
    ownerId: "test-owner",
    acquire: acquireMock,
    renew: renewMock,
    release: releaseMock,
    acquireMock,
    renewMock,
    releaseMock
  };
}

describe("TickCronScheduler", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("starts and stops correctly", () => {
    const scheduler = new TickCronScheduler({
      jobStore: createMockStore(),
      messageHandler: createMockGateway(),
      delivery: createMockDelivery(),
      tickIntervalMs: 1000
    });

    expect(scheduler.isRunning).toBe(false);
    scheduler.start();
    expect(scheduler.isRunning).toBe(true);
    scheduler.stop();
    expect(scheduler.isRunning).toBe(false);
  });

  it("does not start twice", () => {
    const scheduler = new TickCronScheduler({
      jobStore: createMockStore(),
      messageHandler: createMockGateway(),
      delivery: createMockDelivery(),
      tickIntervalMs: 1000
    });

    scheduler.start();
    scheduler.start();
    expect(scheduler.isRunning).toBe(true);
    scheduler.stop();
  });

  it("triggers a job immediately via triggerNow", async () => {
    const job = makeJob();
    const store = createMockStore([job]);
    const gateway = createMockGateway({ text: "Summary result" });
    const delivery = createMockDelivery();

    const scheduler = new TickCronScheduler({
      jobStore: store,
      messageHandler: gateway,
      delivery
    });

    const result = await scheduler.triggerNow(job.id);

    expect(result.status).toBe("success");
    expect(result.output).toBe("Summary result");
    expect(result.jobId).toBe(job.id);
    expect(gateway.handleMock).toHaveBeenCalledOnce();
  });

  it("passes cron job role as a trusted role override", async () => {
    const job = makeJob({ role: "developer" });
    const store = createMockStore([job]);
    const gateway = createMockGateway({ text: "Developer result" });
    const delivery = createMockDelivery();

    const scheduler = new TickCronScheduler({
      jobStore: store,
      messageHandler: gateway,
      delivery
    });

    await scheduler.triggerNow(job.id);

    expect(gateway.handleMock).toHaveBeenCalledWith(
      expect.objectContaining({
        chatId: `job:${job.id}`,
        roleOverride: "developer"
      })
    );
  });

  it("throws when triggering non-existent job", async () => {
    const scheduler = new TickCronScheduler({
      jobStore: createMockStore(),
      messageHandler: createMockGateway(),
      delivery: createMockDelivery()
    });

    await expect(scheduler.triggerNow("nonexistent")).rejects.toThrow("Cron job not found");
  });

  it("records error when gateway fails", async () => {
    const job = makeJob();
    const store = createMockStore([job]);
    const errorFn = vi.fn(() => Promise.reject(new Error("Gateway error")));
    const gateway: MessageGateway = { handle: errorFn };
    const delivery = createMockDelivery();

    const scheduler = new TickCronScheduler({
      jobStore: store,
      messageHandler: gateway,
      delivery
    });

    const result = await scheduler.triggerNow(job.id);

    expect(result.status).toBe("error");
    expect(result.error).toBe("Gateway error");
  });

  it("records timeout when gateway takes too long", async () => {
    const job = makeJob({ timeoutMs: 100 });
    const store = createMockStore([job]);
    const slowFn = vi.fn(
      () =>
        new Promise<OutboundMessage>((resolve) => {
          setTimeout(() => resolve({ text: "Should not see this" }), 200);
        })
    );
    const gateway: MessageGateway = { handle: slowFn };
    const delivery = createMockDelivery();

    const scheduler = new TickCronScheduler({
      jobStore: store,
      messageHandler: gateway,
      delivery
    });

    // Start the trigger which will set up the timeout
    const triggerPromise = scheduler.triggerNow(job.id);

    // Advance fake timers past the timeout
    await vi.advanceTimersByTimeAsync(200);

    const result = await triggerPromise;
    expect(result.status).toBe("timeout");
  });

  it("skips a manual trigger while the same job is already running", async () => {
    const job = makeJob();
    const store = createMockStore([job]);
    let resolveFirst: ((response: OutboundMessage) => void) | undefined;
    const handleMock = vi.fn(
      () =>
        new Promise<OutboundMessage>((resolve) => {
          resolveFirst = resolve;
        })
    );
    const gateway: MessageGateway = { handle: handleMock };
    const delivery = createMockDelivery();

    const scheduler = new TickCronScheduler({
      jobStore: store,
      messageHandler: gateway,
      delivery
    });

    const firstRun = scheduler.triggerNow(job.id);
    await Promise.resolve();
    const secondRun = await scheduler.triggerNow(job.id);

    expect(secondRun.status).toBe("skipped");
    expect(handleMock).toHaveBeenCalledOnce();

    if (resolveFirst === undefined) throw new Error("Expected first run to be pending");
    resolveFirst({ text: "Done" });
    await firstRun;
  });

  it("skips delivery when silentOnEmpty is true and output is empty", async () => {
    const job = makeJob({ silentOnEmpty: true });
    const store = createMockStore([job]);
    const gateway = createMockGateway({ text: "" });
    const delivery = createMockDelivery();

    const scheduler = new TickCronScheduler({
      jobStore: store,
      messageHandler: gateway,
      delivery
    });

    await scheduler.triggerNow(job.id);
    expect(delivery.deliverAllMock).not.toHaveBeenCalled();
  });

  it("delivers result to configured channels", async () => {
    const job = makeJob({ delivery: { channels: ["sse:admin", "wecom:user:zhangsan"] } });
    const store = createMockStore([job]);
    const gateway = createMockGateway({ text: "Report" });
    const delivery = createMockDelivery();

    const scheduler = new TickCronScheduler({
      jobStore: store,
      messageHandler: gateway,
      delivery
    });

    await scheduler.triggerNow(job.id);
    expect(delivery.deliverAllMock).toHaveBeenCalledWith(
      job.delivery.channels,
      expect.objectContaining({ jobName: job.name, output: "Report" })
    );
  });

  it("initializes nextRunAt for jobs without run state", async () => {
    const now = new Date("2026-05-28T08:00:00Z");
    vi.setSystemTime(now);

    const job = makeJob({ schedule: { type: "cron", expression: "0 9 * * *" } });
    const store = createMockStore([job]);
    const delivery = createMockDelivery();

    const scheduler = new TickCronScheduler({
      jobStore: store,
      messageHandler: createMockGateway(),
      delivery,
      tickIntervalMs: 60_000
    });

    scheduler.start();
    await vi.advanceTimersByTimeAsync(100);
    expect(store.setNextRunAtMock).toHaveBeenCalled();

    const calls = store.setNextRunAtMock.mock.calls;
    expect(calls.length).toBeGreaterThan(0);
    const callArgs = calls[0];
    if (callArgs === undefined) throw new Error("Expected setNextRunAt call");
    const nextRunAt = new Date(callArgs[1] as string);
    expect(nextRunAt.getTime()).toBeGreaterThan(now.getTime());

    scheduler.stop();
  });

  it("skips stale overdue jobs beyond grace window", async () => {
    const now = new Date("2026-05-28T12:00:00Z");
    vi.setSystemTime(now);

    const job = makeJob();
    const store = createMockStore([job]);
    (store.getNextRunAt as MockFn).mockResolvedValue("2026-05-28T09:00:00Z");

    const gateway = createMockGateway();
    const delivery = createMockDelivery();

    const scheduler = new TickCronScheduler({
      jobStore: store,
      messageHandler: gateway,
      delivery,
      staleGraceMs: 300_000,
      tickIntervalMs: 60_000
    });

    scheduler.start();
    await vi.advanceTimersByTimeAsync(100);

    expect(gateway.handleMock).not.toHaveBeenCalled();
    scheduler.stop();
  });

  it("does not overlap ticks while a previous tick is still executing", async () => {
    const now = new Date("2026-05-28T12:00:00Z");
    vi.setSystemTime(now);

    const job = makeJob({ schedule: { type: "interval", milliseconds: 1_000 } });
    const store = createMockStore([job]);
    (store.getNextRunAt as MockFn).mockResolvedValue("2026-05-28T11:59:59Z");
    const slowFn = vi.fn(
      () =>
        new Promise<OutboundMessage>((resolve) => {
          setTimeout(() => resolve({ text: "Done" }), 5_000);
        })
    );
    const gateway: MessageGateway = { handle: slowFn };

    const scheduler = new TickCronScheduler({
      jobStore: store,
      messageHandler: gateway,
      delivery: createMockDelivery(),
      tickIntervalMs: 1_000
    });

    scheduler.start();
    await vi.advanceTimersByTimeAsync(3_000);

    expect(slowFn).toHaveBeenCalledOnce();
    scheduler.stop();
    await vi.advanceTimersByTimeAsync(5_000);
  });

  it("does not run scheduled ticks when another process owns the leader lease", async () => {
    const now = new Date("2026-05-28T12:00:00Z");
    vi.setSystemTime(now);

    const job = makeJob({ schedule: { type: "interval", milliseconds: 1_000 } });
    const store = createMockStore([job]);
    (store.getNextRunAt as MockFn).mockResolvedValue("2026-05-28T11:59:59Z");
    const gateway = createMockGateway();
    const leaderLease = createMockLeaderLease(false);

    const scheduler = new TickCronScheduler({
      jobStore: store,
      messageHandler: gateway,
      delivery: createMockDelivery(),
      leaderLease,
      tickIntervalMs: 1_000
    });

    scheduler.start();
    await vi.advanceTimersByTimeAsync(100);

    expect(leaderLease.acquireMock).toHaveBeenCalled();
    expect(scheduler.isLeader).toBe(false);
    expect(store.getAllMock).not.toHaveBeenCalled();
    expect(gateway.handleMock).not.toHaveBeenCalled();

    scheduler.stop();
  });

  it("renews the leader lease while running", async () => {
    const leaderLease = createMockLeaderLease(true);
    const scheduler = new TickCronScheduler({
      jobStore: createMockStore(),
      messageHandler: createMockGateway(),
      delivery: createMockDelivery(),
      leaderLease,
      tickIntervalMs: 1_000,
      leaderRenewIntervalMs: 1_000
    });

    scheduler.start();
    await vi.advanceTimersByTimeAsync(1_100);

    expect(scheduler.isLeader).toBe(true);
    expect(leaderLease.renewMock).toHaveBeenCalled();

    scheduler.stop();
    expect(leaderLease.releaseMock).toHaveBeenCalled();
  });

  it("skips delivery when leadership is lost while a job is running", async () => {
    const now = new Date("2026-05-28T12:00:00Z");
    vi.setSystemTime(now);

    const job = makeJob({ schedule: { type: "interval", milliseconds: 1_000 } });
    const store = createMockStore([job]);
    (store.getNextRunAt as MockFn).mockResolvedValue("2026-05-28T11:59:59Z");
    const delivery = createMockDelivery();
    let resolveGateway: ((message: OutboundMessage) => void) | undefined;
    const handleMock = vi.fn(
      () =>
        new Promise<OutboundMessage>((resolve) => {
          resolveGateway = resolve;
        })
    );
    const gateway: MessageGateway & { handleMock: MockFn } = {
      handle: handleMock,
      handleMock
    };
    const leaderLease: CronLeaderLease = {
      leasePath: "/tmp/cron.leader",
      ownerId: "test-owner",
      acquire: vi.fn(() => Promise.resolve(true)),
      renew: vi.fn(() => Promise.resolve(false)),
      release: vi.fn(() => Promise.resolve())
    };

    const scheduler = new TickCronScheduler({
      jobStore: store,
      messageHandler: gateway,
      delivery,
      leaderLease,
      tickIntervalMs: 1_000,
      leaderRenewIntervalMs: 1_000
    });

    scheduler.start();
    await vi.advanceTimersByTimeAsync(100);
    expect(gateway.handleMock).toHaveBeenCalledOnce();

    await vi.advanceTimersByTimeAsync(1_100);

    expect(store.setLastResultMock).toHaveBeenCalledWith(
      job.id,
      expect.objectContaining({
        status: "skipped",
        error: "Lost cron leader lease"
      })
    );
    expect(delivery.deliverAllMock).not.toHaveBeenCalled();

    resolveGateway?.({ text: "late result" });
    scheduler.stop();
  });

  it("executes one-shot jobs via triggerNow", async () => {
    const job = makeJob({ schedule: { type: "once", runAt: "2026-05-28T09:00:00Z" } });
    const store = createMockStore([job]);
    const delivery = createMockDelivery();

    const scheduler = new TickCronScheduler({
      jobStore: store,
      messageHandler: createMockGateway({ text: "Done" }),
      delivery
    });

    const result = await scheduler.triggerNow(job.id);
    expect(result.status).toBe("success");
    // Note: one-shot disable only happens during tick() pre-advance, not triggerNow
  });
});
