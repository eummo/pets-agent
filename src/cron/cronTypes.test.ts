import { describe, expect, it } from "vitest";
import { cronJobSchema, cronJobStoreFileSchema, cronScheduleSchema } from "./cronTypes.js";

describe("cron schemas", () => {
  it("parses supported schedules and persisted job records", () => {
    expect(cronScheduleSchema.parse({ type: "interval", milliseconds: 60_000 })).toEqual({
      type: "interval",
      milliseconds: 60_000
    });
    expect(() => cronScheduleSchema.parse({ type: "interval", milliseconds: "60000" })).toThrow();

    expect(
      cronJobStoreFileSchema.parse({
        jobs: {
          "job-1": {
            id: "job-1",
            name: "Daily summary",
            schedule: { type: "cron", expression: "0 9 * * *" },
            prompt: "Summarize changes",
            workspacePath: "D:/workspace",
            enabled: true,
            delivery: { channels: ["sse:admin"] },
            createdAt: "2026-06-09T00:00:00.000Z",
            updatedAt: "2026-06-09T00:00:00.000Z"
          }
        },
        runState: {
          "job-1": {
            nextRunAt: "2026-06-09T09:00:00.000Z",
            lastResult: {
              jobId: "job-1",
              startedAt: "2026-06-09T09:00:00.000Z",
              finishedAt: "2026-06-09T09:00:01.000Z",
              status: "success",
              output: "ok"
            }
          }
        }
      })
    ).toMatchObject({
      jobs: { "job-1": { schedule: { type: "cron" } } },
      runState: { "job-1": { lastResult: { status: "success" } } }
    });
  });

  it("rejects unsupported schedule types", () => {
    expect(() =>
      cronJobSchema.parse({
        id: "job-1",
        name: "Broken",
        schedule: { type: "unknown" },
        prompt: "Run",
        workspacePath: "D:/workspace",
        enabled: true,
        delivery: { channels: ["sse:admin"] },
        createdAt: "2026-06-09T00:00:00.000Z",
        updatedAt: "2026-06-09T00:00:00.000Z"
      })
    ).toThrow();
  });
});
