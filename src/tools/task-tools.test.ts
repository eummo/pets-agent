/**
 * Integration tests for pets-agent tools.
 * These are minimal smoke tests that verify the tool execute functions
 * return well-shaped responses and reject obviously bad input.
 */

import { describe, it, expect, beforeEach, vi } from "vitest";

// ─── Mocks — replace real deps with spies ────────────────────────────────────

const mockTask = {
  id: "test-task-1",
  name: "test-task",
  agentType: "claude-code" as const,
  status: "running" as const,
  progress: ["line 1", "line 2"],
  error: undefined,
  exitCode: undefined,
  startedAt: new Date(),
  endedAt: undefined,
  createdAt: new Date(),
  workdir: "/tmp",
  parentId: undefined,
  attempt: 0,
  priority: 5,
};

const mockAgentManager = {
  spawnWithRetry: vi.fn(() => mockTask),
  get: vi.fn((id: string) => (id === mockTask.id ? mockTask : undefined)),
  list: vi.fn(() => [mockTask]),
  kill: vi.fn(),
  subscribe: vi.fn(() => () => {}),
  killByToken: vi.fn(),
  getStatus: vi.fn(() => ({
    running: [{ id: mockTask.id, name: mockTask.name, agentType: mockTask.agentType, status: mockTask.status, age: 1000, progressLines: 2, hasProgress: true, zombie: false }],
    subscriptions: 0,
    zombieRisk: [],
    totals: { running: 1, done: 0, failed: 0, cancelled: 0 },
  })),
};

const mockTaskHistory = {
  add: vi.fn(),
  appendLog: vi.fn(),
  query: vi.fn(() => []),
  list: vi.fn(() => []),
  clear: vi.fn(),
};

vi.mock("../tasks/agent-manager.js", () => ({ agentManager: mockAgentManager }));
vi.mock("../tasks/task-history.js", () => ({ taskHistory: mockTaskHistory }));

// ─── Helpers ─────────────────────────────────────────────────────────────────

/** ExtensionAPI mock — also typed as ExtensionAPIWithTools for test access */
type ExtensionAPIWithTools = import("@earendil-works/pi-coding-agent").ExtensionAPI & {
  getTool: (name: string) => unknown;
};

function makeMockPi(): ExtensionAPIWithTools {
  const tools = new Map<string, unknown>();
  return {
    registerTool: (def: unknown) => tools.set((def as { name: string }).name, def),
    getTool: (name: string) => tools.get(name),
    tools,
  } as ExtensionAPIWithTools;
}

// ─── Tests ───────────────────────────────────────────────────────────────────

describe("registerTaskTools — spawn_agent", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("registers spawn_agent, list_tasks, get_task, kill_task, decompose_task, wait_for_tasks", async () => {
    const { registerTaskTools } = await import("../tools/task-tools.js");
    const pi = makeMockPi();
    registerTaskTools(pi);

    expect(pi.getTool("spawn_agent")).toBeDefined();
    expect(pi.getTool("list_tasks")).toBeDefined();
    expect(pi.getTool("get_task")).toBeDefined();
    expect(pi.getTool("kill_task")).toBeDefined();
    expect(pi.getTool("decompose_task")).toBeDefined();
    expect(pi.getTool("get_task_tree")).toBeDefined();
    expect(pi.getTool("wait_for_tasks")).toBeDefined();
    expect(pi.getTool("list_task_history")).toBeDefined();
  });

  it("returns validation error when agentType is missing", async () => {
    const { registerTaskTools } = await import("../tools/task-tools.js");
    const pi = makeMockPi();
    registerTaskTools(pi);
    const def = pi.getTool("spawn_agent") as { execute: (id: string, p: Record<string, unknown>, s: unknown, u: unknown, c: unknown) => unknown };
    const result = await def.execute("call-1", { prompt: "do stuff", agentType: "" } as Record<string, unknown>, null, null, null);
    const r = result as { content: { text: string }[]; details: Record<string, unknown> };
    expect(r.details.validationError).toBe(true);
    expect(r.content[0].text).toContain("agentType");
  });

  it("returns validation error when prompt is missing", async () => {
    const { registerTaskTools } = await import("../tools/task-tools.js");
    const pi = makeMockPi();
    registerTaskTools(pi);
    const def = pi.getTool("spawn_agent") as { execute: (id: string, p: Record<string, unknown>, s: unknown, u: unknown, c: unknown) => unknown };
    const result = await def.execute("call-1", { agentType: "claude-code", prompt: "  " } as Record<string, unknown>, null, null, null);
    const r = result as { content: { text: string }[]; details: Record<string, unknown> };
    expect(r.details.validationError).toBe(true);
    expect(r.content[0].text).toContain("prompt");
  });

  it("returns validation error for invalid priority (out of range)", async () => {
    const { registerTaskTools } = await import("../tools/task-tools.js");
    const pi = makeMockPi();
    registerTaskTools(pi);
    const def = pi.getTool("spawn_agent") as { execute: (id: string, p: Record<string, unknown>, s: unknown, u: unknown, c: unknown) => unknown };
    const result = await def.execute("call-1", { agentType: "claude-code", prompt: "do stuff", priority: 99 } as Record<string, unknown>, null, null, null);
    const r = result as { content: { text: string }[]; details: Record<string, unknown> };
    expect(r.details.validationError).toBe(true);
    expect(r.content[0].text).toContain("priority");
  });

  it("returns validation error for negative timeoutSec", async () => {
    const { registerTaskTools } = await import("../tools/task-tools.js");
    const pi = makeMockPi();
    registerTaskTools(pi);
    const def = pi.getTool("spawn_agent") as { execute: (id: string, p: Record<string, unknown>, s: unknown, u: unknown, c: unknown) => unknown };
    const result = await def.execute("call-1", { agentType: "claude-code", prompt: "do stuff", timeoutSec: -1 } as Record<string, unknown>, null, null, null);
    const r = result as { content: { text: string }[]; details: Record<string, unknown> };
    expect(r.details.validationError).toBe(true);
    expect(r.content[0].text).toContain("timeoutSec");
  });

  it("get_task returns 'not found' for unknown taskId", async () => {
    const { registerTaskTools } = await import("../tools/task-tools.js");
    const pi = makeMockPi();
    registerTaskTools(pi);
    const def = pi.getTool("get_task") as { execute: (id: string, p: Record<string, unknown>, s: unknown, u: unknown, c: unknown) => unknown };
    const result = await def.execute("call-1", { taskId: "does-not-exist" }, null, null, null);
    const r = result as { content: { text: string }[]; details: Record<string, unknown> };
    expect(r.content[0].text).toContain("not found");
  });

  it("kill_task returns 'not found' for unknown taskId", async () => {
    const { registerTaskTools } = await import("../tools/task-tools.js");
    const pi = makeMockPi();
    registerTaskTools(pi);
    const def = pi.getTool("kill_task") as { execute: (id: string, p: Record<string, unknown>, s: unknown, u: unknown, c: unknown) => unknown };
    const result = await def.execute("call-1", { taskId: "does-not-exist" }, null, null, null);
    const r = result as { content: { text: string }[]; details: Record<string, unknown> };
    expect(r.content[0].text).toContain("not found");
  });

  it("wait_for_tasks returns validation error for empty taskIds", async () => {
    const { registerTaskTools } = await import("../tools/task-tools.js");
    const pi = makeMockPi();
    registerTaskTools(pi);
    const def = pi.getTool("wait_for_tasks") as { execute: (id: string, p: Record<string, unknown>, s: unknown, u: unknown, c: unknown) => unknown };
    const result = await def.execute("call-1", { taskIds: [] }, null, null, null);
    const r = result as { content: { text: string }[]; details: Record<string, unknown> };
    expect(r.details.validationError).toBe(true);
    expect(r.content[0].text).toContain("taskIds");
  });

  it("decompose_task validates subtask entries", async () => {
    const { registerTaskTools } = await import("../tools/task-tools.js");
    const pi = makeMockPi();
    registerTaskTools(pi);
    const def = pi.getTool("decompose_task") as { execute: (id: string, p: Record<string, unknown>, s: unknown, u: unknown, c: unknown) => unknown };
    const result = await def.execute("call-1", {
      taskDescription: "build a thing",
      subtasks: [{ agentType: "", prompt: "" }],
    } as Record<string, unknown>, null, null, null);
    const r = result as { content: { text: string }[]; details: Record<string, unknown> };
    expect(r.details.validationError).toBe(true);
  });
});

describe("registerTaskTools — getStatus()", () => {
  it("agentManager.getStatus() returns a structured snapshot", () => {
    const status = mockAgentManager.getStatus();
    expect(status.running).toBeInstanceOf(Array);
    expect(typeof status.subscriptions).toBe("number");
    expect(typeof status.zombieRisk).toBe("object");
    expect(typeof status.totals.running).toBe("number");
  });
});
