import { describe, expect, it } from "vitest";
import { createSqliteConnection } from "../persistence/sqliteConnection.js";
import { SqliteCronJobStore } from "./sqliteCronJobStore.js";
import type { CronJob } from "./cronTypes.js";

function makeJob(
  overrides: Partial<CronJob> = {}
): Omit<CronJob, "id" | "createdAt" | "updatedAt"> {
  return {
    name: "Test Job",
    schedule: { type: "cron", expression: "0 9 * * *" },
    prompt: "Summarize today's changes",
    workspacePath: "/workspace/default",
    enabled: true,
    delivery: { channels: ["sse:admin"] },
    ...overrides
  };
}

describe("SqliteCronJobStore", () => {
  it("creates and retrieves a job", async () => {
    const store = new SqliteCronJobStore(createSqliteConnection(":memory:"));

    const job = await store.create(makeJob());
    const retrieved = await store.getById(job.id);

    expect(retrieved).toMatchObject({
      id: job.id,
      name: "Test Job",
      schedule: { type: "cron", expression: "0 9 * * *" },
      delivery: { channels: ["sse:admin"] }
    });
  });

  it("lists jobs in creation order", async () => {
    const store = new SqliteCronJobStore(createSqliteConnection(":memory:"));

    await store.create(makeJob({ name: "Job A" }));
    await store.create(makeJob({ name: "Job B" }));

    const jobs = await store.getAll();
    expect(jobs.map((job) => job.name)).toEqual(["Job A", "Job B"]);
  });

  it("updates optional job fields", async () => {
    const store = new SqliteCronJobStore(createSqliteConnection(":memory:"));
    const job = await store.create(makeJob());

    const updated = await store.update(job.id, {
      enabled: false,
      role: "developer",
      timeoutMs: 30_000,
      silentOnEmpty: true
    });

    expect(updated).toMatchObject({
      enabled: false,
      role: "developer",
      timeoutMs: 30_000,
      silentOnEmpty: true
    });
    await expect(store.getById(job.id)).resolves.toMatchObject({
      enabled: false,
      role: "developer",
      timeoutMs: 30_000,
      silentOnEmpty: true
    });
  });

  it("persists run state", async () => {
    const store = new SqliteCronJobStore(createSqliteConnection(":memory:"));
    const job = await store.create(makeJob());
    const nextRunAt = "2026-06-08T09:00:00.000Z";
    const result = {
      jobId: job.id,
      startedAt: "2026-06-08T09:00:00.000Z",
      finishedAt: "2026-06-08T09:00:01.000Z",
      status: "success" as const,
      output: "Done"
    };

    await store.setNextRunAt(job.id, nextRunAt);
    await store.setLastResult(job.id, result);

    await expect(store.getNextRunAt(job.id)).resolves.toBe(nextRunAt);
    await expect(store.getLastResult(job.id)).resolves.toEqual(result);
  });

  it("deletes run state when deleting a job", async () => {
    const db = createSqliteConnection(":memory:");
    const store = new SqliteCronJobStore(db);
    const job = await store.create(makeJob());

    await store.setNextRunAt(job.id, "2026-06-08T09:00:00.000Z");
    await expect(store.delete(job.id)).resolves.toBe(true);

    await expect(store.getById(job.id)).resolves.toBeUndefined();
    const stateCount = db
      .prepare("SELECT COUNT(*) AS count FROM cron_run_state WHERE job_id = ?")
      .get(job.id) as { readonly count: number };
    expect(stateCount.count).toBe(0);
  });
});
