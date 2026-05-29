import Fastify from "fastify";
import { describe, expect, it, vi } from "vitest";
import type { AuthorizationAction, AuthorizationService, RoleCapability } from "../auth/index.js";
import type { ChannelUser } from "../core/index.js";
import type { KnowledgeWorkspace } from "../workspace/index.js";
import { registerCronRoutes } from "./cronRoutes.js";
import type { CronJob, CronJobResult, CronJobStore, CronScheduler } from "./cronTypes.js";

function makeJob(overrides: Partial<CronJob> = {}): CronJob {
  return {
    id: "daily-report",
    name: "Daily Report",
    schedule: { type: "cron", expression: "0 9 * * *" },
    prompt: "Summarize yesterday",
    workspacePath: ".harness/knowledge-base",
    enabled: true,
    delivery: { channels: ["sse:admin"] },
    createdAt: "2026-05-28T10:00:00.000Z",
    updatedAt: "2026-05-28T10:00:00.000Z",
    ...overrides,
  };
}

function makeStore(jobs: readonly CronJob[] = []): CronJobStore {
  const jobMap = new Map(jobs.map((job) => [job.id, job]));

  return {
    getAll() {
      return Promise.resolve([...jobMap.values()]);
    },
    getById(id) {
      return Promise.resolve(jobMap.get(id));
    },
    create(job) {
      const created: CronJob = {
        ...job,
        id: "created-job",
        createdAt: "2026-05-28T10:00:00.000Z",
        updatedAt: "2026-05-28T10:00:00.000Z",
      };
      jobMap.set(created.id, created);
      return Promise.resolve(created);
    },
    update(id, patch) {
      const existing = jobMap.get(id);
      if (existing === undefined) return Promise.resolve(undefined);
      const updated: CronJob = { ...existing, ...patch, id, updatedAt: "2026-05-28T10:00:00.000Z" };
      jobMap.set(id, updated);
      return Promise.resolve(updated);
    },
    delete(id) {
      return Promise.resolve(jobMap.delete(id));
    },
    getNextRunAt() {
      return Promise.resolve(undefined);
    },
    setNextRunAt() {
      return Promise.resolve();
    },
    getLastResult() {
      return Promise.resolve(undefined);
    },
    setLastResult() {
      return Promise.resolve();
    },
  };
}

function makeScheduler(): CronScheduler {
  const result: CronJobResult = {
    jobId: "daily-report",
    startedAt: "2026-05-28T10:00:00.000Z",
    finishedAt: "2026-05-28T10:00:01.000Z",
    status: "success",
    output: "Done",
  };

  return {
    start() {},
    stop() {},
    triggerNow: vi.fn(() => Promise.resolve(result)),
    isRunning: true,
  };
}

function makeAuthorization(capabilities: Record<string, readonly RoleCapability[]>): AuthorizationService {
  return {
    roleFor(user) {
      return Promise.resolve(user.id);
    },
    can(user: ChannelUser, action: AuthorizationAction, workspace: KnowledgeWorkspace) {
      void user;
      void action;
      void workspace;
      return Promise.resolve({ allowed: true });
    },
    hasCapability(user, capability) {
      return Promise.resolve(capabilities[user.id]?.includes(capability) ?? false);
    },
  };
}

describe("registerCronRoutes", () => {
  it("rejects cron management from non-local clients", async () => {
    const server = Fastify();
    registerCronRoutes(server, {
      jobStore: makeStore([makeJob()]),
      scheduler: makeScheduler(),
      authorization: makeAuthorization({ admin: ["cron_manage"] }),
    });

    const response = await server.inject({
      method: "GET",
      url: "/cron/jobs?userId=admin",
      remoteAddress: "10.0.0.5",
    });

    expect(response.statusCode).toBe(403);
    expect(response.json()).toEqual({ error: "Cron management is only available from localhost." });
  });

  it("lists jobs for a local user with cron_manage capability", async () => {
    const server = Fastify();
    registerCronRoutes(server, {
      jobStore: makeStore([makeJob()]),
      scheduler: makeScheduler(),
      authorization: makeAuthorization({ admin: ["cron_manage"] }),
    });

    const response = await server.inject({
      method: "GET",
      url: "/cron/jobs?userId=admin",
    });

    expect(response.statusCode).toBe(200);
    expect(response.json()).toEqual([
      expect.objectContaining({ id: "daily-report", name: "Daily Report" }),
    ]);
  });
});
