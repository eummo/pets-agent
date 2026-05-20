/**
 * TaskHistory — Unit Tests
 * Run: npx vitest run src/tests/task-history.test.ts --no-typecheck
 */

import { describe, it, expect, beforeEach } from "vitest";
import { taskHistory } from "../tasks/task-history.js";
import type { Task, TaskHistoryQuery } from "../tasks/task-history.js";

function makeTask(overrides: Partial<Task> = {}): Task {
  return {
    id: Math.random().toString(36).slice(2),
    name: "test-task",
    agentType: "claude-code",
    prompt: "do something",
    status: "done",
    createdAt: new Date(),
    progress: [],
    ...overrides,
  } as Task;
}

describe("TaskHistory", () => {
  // Snapshot original entries so we don't pollute real data
  const snapshot: string[] = [];

  beforeEach(() => {
    // Save existing entries
    for (const e of taskHistory.getAll()) {
      snapshot.push(e.id);
    }
    // Clear in-memory entries for test isolation
    (taskHistory as any).entries = [];
  });

  describe("add()", () => {
    it("adds a task and makes it queryable", () => {
      const task = makeTask({ name: "my-task", status: "done" });
      taskHistory.add(task);
      const results = taskHistory.query({});
      expect(results.some((e) => e.name === "my-task")).toBe(true);
    });

    it("extracts file count from progress lines", () => {
      const task = makeTask({ progress: ["Installing deps...", "Created 5 files"] });
      taskHistory.add(task);
      const results = taskHistory.query({});
      expect(results[0].fileCount).toBe(5);
    });
  });

  describe("query()", () => {
    beforeEach(() => {
      taskHistory.add(makeTask({ agentType: "claude-code", status: "done" }));
      taskHistory.add(makeTask({ agentType: "codex", status: "done" }));
      taskHistory.add(makeTask({ agentType: "claude-code", status: "failed" }));
    });

    it("returns all entries when no filter", () => {
      const results = taskHistory.query({});
      expect(results.length).toBeGreaterThanOrEqual(3);
    });

    it("filters by agentType", () => {
      const results = taskHistory.query({ agentType: "claude-code" });
      expect(results.every((e) => e.agentType === "claude-code")).toBe(true);
    });

    it("filters by status", () => {
      const results = taskHistory.query({ status: "failed" });
      expect(results.every((e) => e.status === "failed")).toBe(true);
    });

    it("filters by since date (epoch returns all)", () => {
      // epoch = all entries since the beginning of time
      const results = taskHistory.query({ since: "1970-01-01T00:00:00.000Z" });
      expect(results.length).toBeGreaterThanOrEqual(3);
    });

    it("respects limit", () => {
      const results = taskHistory.query({ limit: 2 });
      expect(results.length).toBeLessThanOrEqual(2);
    });

    it("filters by agentType AND status together", () => {
      const results = taskHistory.query({ agentType: "claude-code", status: "done" });
      expect(results.every((e) => e.agentType === "claude-code" && e.status === "done")).toBe(true);
    });
  });

  describe("writeLog() / readLog()", () => {
    it("writes and reads back log lines", () => {
      const taskId = "log-test-" + Math.random().toString(36).slice(2);
      taskHistory.writeLog(taskId, ["line 1", "line 2", "line 3"]);
      const lines = taskHistory.readLog(taskId);
      expect(lines).toContain("line 1");
      expect(lines).toContain("line 2");
      expect(lines).toContain("line 3");
    });

    it("readLog returns empty array for non-existent task", () => {
      const lines = taskHistory.readLog("non-existent-task-id-xyz");
      expect(lines).toHaveLength(0);
    });
  });
});
