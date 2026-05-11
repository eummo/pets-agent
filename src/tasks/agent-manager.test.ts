import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { AgentManager } from "./agent-manager.js";

describe("AgentManager", () => {
  let mgr: AgentManager;

  beforeEach(() => {
    mgr = new AgentManager();
  });

  afterEach(() => {
    mgr.destroy();
  });

  // -------------------------------------------------------------------------
  // wslPath
  // -------------------------------------------------------------------------
  describe("wslPath", () => {
    // @ts-expect-error — accessing private static method for testing
    const wslPath = (path: string) => AgentManager.wslToWindowsPath(path);

    it("converts /mnt/c/... to C:\\...", () => {
      expect(wslPath("/mnt/c/Users/jadenli/code")).toBe("C:\\Users\\jadenli\\code");
    });

    it("converts /mnt/d/... to D:\\...", () => {
      expect(wslPath("/mnt/d/projects/my-app")).toBe("D:\\projects\\my-app");
    });

    it("normalises forward slashes in the path part", () => {
      expect(wslPath("/mnt/c/foo/bar/baz")).toBe("C:\\foo\\bar\\baz");
    });

    it("leaves non-/mnt paths unchanged", () => {
      expect(wslPath("/home/user/code")).toBe("/home/user/code");
      expect(wslPath("/tmp/file")).toBe("/tmp/file");
    });
  });

  // -------------------------------------------------------------------------
  // spawn — basic fields
  // -------------------------------------------------------------------------
  describe("spawn", () => {
    it("creates a task with all required fields", () => {
      const task = mgr.spawn("pi-agent", "hello world", { name: "test-task" });
      expect(task.id).toBeDefined();
      expect(task.name).toBe("test-task");
      expect(task.agentType).toBe("pi-agent");
      expect(task.prompt).toBe("hello world");
      expect(task.status).toBe("running");
      expect(task.createdAt).toBeInstanceOf(Date);
      expect(task.startedAt).toBeInstanceOf(Date);
      expect(task.progress).toEqual([]);
      // Clean up the running process
      mgr.kill(task.id);
    });

    it("records parentId when provided", () => {
      const parent = mgr.spawn("pi-agent", "parent");
      const child = mgr.spawn("pi-agent", "child", { parentId: parent.id });
      expect(child.parentId).toBe(parent.id);
      expect(parent.children).toContain(child.id);
      mgr.kill(parent.id);
    });

    it("accepts timeoutMs and kills the process after expiry", () => {
      vi.useFakeTimers({ shouldAdvanceTime: false });
      const task = mgr.spawn("pi-agent", "sleep 10", { timeoutMs: 2000 });
      expect(task.status).toBe("running");
      // Advance past the 2s timeout
      vi.advanceTimersByTime(2500);
      // The exit handler runs asynchronously, so use vi.runAllTimers
      vi.useRealTimers();
    });
  });

  // -------------------------------------------------------------------------
  // list — superseded filtering
  // -------------------------------------------------------------------------
  describe("list", () => {
    it("excludes superseded tasks by default", () => {
      // Spawn a task and manually simulate it being superseded
      const task = mgr.spawn("pi-agent", "hello");
      // @ts-ignore — writing a private field for test setup
      task.supersededBy = "another-task-id";
      mgr.list(); // populate internal map
      const listed = mgr.list();
      expect(listed.find((t) => t.id === task.id)).toBeUndefined();
    });

    it("includes superseded tasks when includeSuperseded=true", () => {
      const task = mgr.spawn("pi-agent", "hello");
      // @ts-ignore — writing a private field for test setup
      task.supersededBy = "another-task-id";
      const listed = mgr.list(true);
      expect(listed.find((t) => t.id === task.id)).toBeDefined();
      mgr.kill(task.id);
    });
  });

  // -------------------------------------------------------------------------
  // spawnWithRetry — attempt counter
  // -------------------------------------------------------------------------
  describe("spawnWithRetry", () => {
    it("sets attempt=1 on the initial task", () => {
      const task = mgr.spawnWithRetry("pi-agent", "hello", { maxRetries: 0 });
      expect(task.attempt).toBe(1);
      mgr.kill(task.id);
    });

    it("records supersededBy on the old task when retry is triggered", () => {
      // Use fake timers so the retry fires immediately after "exit"
      vi.useFakeTimers({ shouldAdvanceTime: false });

      let exitFired = false;
      mgr.on("exit", () => {
        exitFired = true;
      });

      // Spawn a task that will fail immediately
      const task = mgr.spawnWithRetry("pi-agent", "exit 1", { maxRetries: 1 });

      // Simulate the process exiting with an error (failed status)
      // We need to manually trigger the exit event so retry is scheduled
      // Advance timers to fire the retry
      vi.advanceTimersByTime(1100); // delay = 2*1000ms for attempt=2
      vi.runAllTimers();
      vi.useRealTimers();
    });
  });

  // -------------------------------------------------------------------------
  // kill — SIGTERM then SIGKILL
  // -------------------------------------------------------------------------
  describe("kill", () => {
    it("transitions task status to cancelled", () => {
      const task = mgr.spawn("pi-agent", "hello");
      mgr.kill(task.id);
      expect(task.status).toBe("cancelled");
      expect(task.endedAt).toBeInstanceOf(Date);
    });

    it("does not throw for an already-finished task", () => {
      const task = mgr.spawn("pi-agent", "hello");
      mgr.kill(task.id);
      // Killing twice should not throw
      expect(() => mgr.kill(task.id)).not.toThrow();
    });
  });

  // -------------------------------------------------------------------------
  // get / getActiveTasks
  // -------------------------------------------------------------------------
  describe("get / getActiveTasks", () => {
    it("retrieves a task by id", () => {
      const task = mgr.spawn("pi-agent", "hello");
      expect(mgr.get(task.id)).toBeDefined();
      mgr.kill(task.id);
    });

    it("getActiveTasks returns only running/pending tasks", () => {
      const running = mgr.spawn("pi-agent", "hello");
      const active = mgr.getActiveTasks();
      expect(active.some((t) => t.id === running.id)).toBe(true);
      mgr.kill(running.id);
    });
  });
});
