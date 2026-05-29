import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";
import { FileCronJobStore } from "./cronJobStore.js";
import type { CronJob } from "./cronTypes.js";

function makeJob(overrides: Partial<CronJob> = {}): Omit<CronJob, "id" | "createdAt" | "updatedAt"> {
  return {
    name: "Test Job",
    schedule: { type: "cron", expression: "0 9 * * *" },
    prompt: "Summarize today's changes",
    workspacePath: "/workspace/default",
    enabled: true,
    delivery: { channels: ["sse:admin"] },
    ...overrides,
  };
}

describe("FileCronJobStore", () => {
  it("creates and retrieves a job", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "cron-store-"));
    const filePath = path.join(root, "cron-jobs.json");
    const store = new FileCronJobStore(filePath);

    const job = await store.create(makeJob());
    const retrieved = await store.getById(job.id);

    expect(retrieved).toBeDefined();
    expect(retrieved?.name).toBe("Test Job");
    expect(retrieved?.schedule).toEqual({ type: "cron", expression: "0 9 * * *" });
    expect(retrieved?.createdAt).toBeTruthy();
    expect(retrieved?.updatedAt).toBeTruthy();
  });

  it("lists all jobs", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "cron-store-"));
    const filePath = path.join(root, "cron-jobs.json");
    const store = new FileCronJobStore(filePath);

    await store.create(makeJob({ name: "Job A" }));
    await store.create(makeJob({ name: "Job B" }));

    const jobs = await store.getAll();
    expect(jobs).toHaveLength(2);
    expect(jobs.map((j) => j.name).sort()).toEqual(["Job A", "Job B"]);
  });

  it("updates a job", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "cron-store-"));
    const filePath = path.join(root, "cron-jobs.json");
    const store = new FileCronJobStore(filePath);

    const job = await store.create(makeJob({ name: "Original" }));
    const updated = await store.update(job.id, { name: "Updated", enabled: false });

    expect(updated?.name).toBe("Updated");
    expect(updated?.enabled).toBe(false);

    const retrieved = await store.getById(job.id);
    expect(retrieved?.name).toBe("Updated");
  });

  it("returns undefined when updating non-existent job", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "cron-store-"));
    const filePath = path.join(root, "cron-jobs.json");
    const store = new FileCronJobStore(filePath);

    const result = await store.update("nonexistent", { name: "X" });
    expect(result).toBeUndefined();
  });

  it("deletes a job", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "cron-store-"));
    const filePath = path.join(root, "cron-jobs.json");
    const store = new FileCronJobStore(filePath);

    const job = await store.create(makeJob());
    const deleted = await store.delete(job.id);
    expect(deleted).toBe(true);

    const retrieved = await store.getById(job.id);
    expect(retrieved).toBeUndefined();
  });

  it("returns false when deleting non-existent job", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "cron-store-"));
    const filePath = path.join(root, "cron-jobs.json");
    const store = new FileCronJobStore(filePath);

    const deleted = await store.delete("nonexistent");
    expect(deleted).toBe(false);
  });

  it("persists run state (nextRunAt, lastResult)", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "cron-store-"));
    const filePath = path.join(root, "cron-jobs.json");
    const store = new FileCronJobStore(filePath);

    const job = await store.create(makeJob());

    await expect(store.getNextRunAt(job.id)).resolves.toBeUndefined();
    await expect(store.getLastResult(job.id)).resolves.toBeUndefined();

    const nextRun = new Date("2026-06-01T09:00:00Z").toISOString();
    await store.setNextRunAt(job.id, nextRun);
    await expect(store.getNextRunAt(job.id)).resolves.toBe(nextRun);

    const result = {
      jobId: job.id,
      startedAt: "2026-05-28T09:00:00Z",
      finishedAt: "2026-05-28T09:00:45Z",
      status: "success" as const,
      output: "Summary: 3 commits",
    };
    await store.setLastResult(job.id, result);
    await expect(store.getLastResult(job.id)).resolves.toEqual(result);
  });

  it("persists to file so new instances can read", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "cron-store-"));
    const filePath = path.join(root, "cron-jobs.json");

    const store1 = new FileCronJobStore(filePath);
    const job = await store1.create(makeJob({ name: "Persistent Job" }));

    const store2 = new FileCronJobStore(filePath);
    const retrieved = await store2.getById(job.id);
    expect(retrieved?.name).toBe("Persistent Job");
  });

  it("concurrent writes do not lose data", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "cron-store-"));
    const filePath = path.join(root, "cron-jobs.json");
    const store = new FileCronJobStore(filePath);

    await Promise.all(
      Array.from({ length: 10 }, (_, i) =>
        store.create(makeJob({ name: `Job ${i}` }))
      )
    );

    const all = await store.getAll();
    expect(all).toHaveLength(10);
    expect(new Set(all.map((j) => j.id)).size).toBe(10);
  });

  it("writes valid JSON to the file", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "cron-store-"));
    const filePath = path.join(root, "cron-jobs.json");
    const store = new FileCronJobStore(filePath);

    await store.create(makeJob());
    const raw = await readFile(filePath, "utf8");
    const parsed = JSON.parse(raw) as { jobs: unknown; runState: unknown };
    expect(parsed.jobs).toBeDefined();
    expect(parsed.runState).toBeDefined();
  });
});
