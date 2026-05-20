/**
 * Integration tests for team-tools.
 * Smoke tests verifying execute functions return well-shaped responses
 * and reject bad input via the new runtime validation guards.
 */

import { describe, it, expect, beforeEach, vi } from "vitest";

// ─── Mocks ───────────────────────────────────────────────────────────────────

const mockProject = {
  id: "proj-1",
  name: "Test Project",
  description: "A test project",
  phase: "idea" as const,
  status: "active" as const,
  target: "test target",
  successCriteria: "all tests pass",
  updatedAt: new Date(),
  createdAt: new Date(),
};

const mockProjectStore = {
  create: vi.fn(() => mockProject),
  get: vi.fn((id: string) => (id === mockProject.id ? mockProject : undefined)),
  list: vi.fn(() => [mockProject]),
  update: vi.fn(),
};

const mockPhaseController = {
  canAdvance: vi.fn(() => ({ ok: true })),
  advance: vi.fn(() => ({ ...mockProject, phase: "feasibility" as const })),
};

const mockMeetingManager = {
  createDecision: vi.fn(() => ({ id: "dec-1", projectId: mockProject.id })),
  createMeeting: vi.fn(() => ({ id: "meet-1", projectId: mockProject.id })),
};

const mockProjectManager = {
  planPhase: vi.fn(() => ({ ok: true, plan: "Phase plan content" })),
  makeDecision: vi.fn(() => ({ id: "dec-1" })),
};

const mockTeam = {
  runRole: vi.fn(() => Promise.resolve({ ok: true, result: { role: "developer", status: "done", summary: "done", artifacts: [], nextActions: [] } })),
  createArtifact: vi.fn(() => ({ id: "art-1" })),
  reviewArtifact: vi.fn(() => ({ id: "art-rev-1" })),
  formatTeamStatus: vi.fn(() => "Project: Test Project\nPhase: idea"),
};

vi.mock("../multi-agent-team/project-store.js", () => ({ projectStore: mockProjectStore }));
vi.mock("../multi-agent-team/project-manager.js", () => ({ projectManager: mockProjectManager }));
vi.mock("../multi-agent-team/phase-controller.js", () => ({ phaseController: mockPhaseController }));
vi.mock("../multi-agent-team/meeting.js", () => ({ meetingManager: mockMeetingManager }));
vi.mock("../multi-agent-team/team.js", () => ({
  runRole: mockTeam.runRole,
  createArtifact: mockTeam.createArtifact,
  reviewArtifact: mockTeam.reviewArtifact,
  formatTeamStatus: mockTeam.formatTeamStatus,
}));

// ─── Helpers ─────────────────────────────────────────────────────────────────

function makeMockPi() {
  const tools = new Map<string, unknown>();
  type ExtensionAPIWithTools = import("@earendil-works/pi-coding-agent").ExtensionAPI & {
    getTool: (name: string) => unknown;
  };
  return {
    registerTool: (def: unknown) => tools.set((def as { name: string }).name, def),
    getTool: (name: string) => tools.get(name),
    tools,
  } as ExtensionAPIWithTools;
}

// ─── Tests ───────────────────────────────────────────────────────────────────

describe("registerTeamTools — validation guards", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("registers all 11 team tools", async () => {
    const { registerTeamTools } = await import("../tools/team-tools.js");
    const pi = makeMockPi();
    registerTeamTools(pi);
    const names = ["create_project", "list_projects", "get_project", "plan_phase", "run_role",
      "create_artifact", "review_artifact", "advance_phase", "make_decision", "team_meeting", "generate_doc"];
    for (const name of names) {
      expect(pi.getTool(name)).toBeDefined();
    }
  });

  it("create_project rejects empty name", async () => {
    const { registerTeamTools } = await import("../tools/team-tools.js");
    const pi = makeMockPi();
    registerTeamTools(pi);
    const def = pi.getTool("create_project") as { execute: (id: string, p: Record<string, unknown>, s: unknown, u: unknown, c: unknown) => unknown };
    const result = await def.execute("c1", { name: "  ", description: "desc" }, null, null, null);
    const r = result as { content: { text: string }[]; details: Record<string, unknown> };
    expect(r.details.validationError).toBe(true);
  });

  it("create_project rejects empty description", async () => {
    const { registerTeamTools } = await import("../tools/team-tools.js");
    const pi = makeMockPi();
    registerTeamTools(pi);
    const def = pi.getTool("create_project") as { execute: (id: string, p: Record<string, unknown>, s: unknown, u: unknown, c: unknown) => unknown };
    const result = await def.execute("c1", { name: "proj", description: "" }, null, null, null);
    const r = result as { content: { text: string }[]; details: Record<string, unknown> };
    expect(r.details.validationError).toBe(true);
  });

  it("plan_phase rejects invalid phase enum", async () => {
    const { registerTeamTools } = await import("../tools/team-tools.js");
    const pi = makeMockPi();
    registerTeamTools(pi);
    const def = pi.getTool("plan_phase") as { execute: (id: string, p: Record<string, unknown>, s: unknown, u: unknown, c: unknown) => unknown };
    const result = await def.execute("c1", { projectId: "proj-1", phase: "not-a-phase" }, null, null, null);
    const r = result as { content: { text: string }[]; details: Record<string, unknown> };
    expect(r.details.validationError).toBe(true);
    expect(r.content[0].text).toContain("phase");
  });

  it("plan_phase rejects empty projectId", async () => {
    const { registerTeamTools } = await import("../tools/team-tools.js");
    const pi = makeMockPi();
    registerTeamTools(pi);
    const def = pi.getTool("plan_phase") as { execute: (id: string, p: Record<string, unknown>, s: unknown, u: unknown, c: unknown) => unknown };
    const result = await def.execute("c1", { projectId: "", phase: "idea" }, null, null, null);
    const r = result as { content: { text: string }[]; details: Record<string, unknown> };
    expect(r.details.validationError).toBe(true);
  });

  it("run_role rejects invalid role enum", async () => {
    const { registerTeamTools } = await import("../tools/team-tools.js");
    const pi = makeMockPi();
    registerTeamTools(pi);
    const def = pi.getTool("run_role") as { execute: (id: string, p: Record<string, unknown>, s: unknown, u: unknown, c: unknown) => unknown };
    const result = await def.execute("c1", { projectId: "proj-1", role: "notarole", phase: "idea" }, null, null, null);
    const r = result as { content: { text: string }[]; details: Record<string, unknown> };
    expect(r.details.validationError).toBe(true);
  });

  it("run_role rejects empty projectId", async () => {
    const { registerTeamTools } = await import("../tools/team-tools.js");
    const pi = makeMockPi();
    registerTeamTools(pi);
    const def = pi.getTool("run_role") as { execute: (id: string, p: Record<string, unknown>, s: unknown, u: unknown, c: unknown) => unknown };
    const result = await def.execute("c1", { projectId: "  ", role: "developer", phase: "idea" }, null, null, null);
    const r = result as { content: { text: string }[]; details: Record<string, unknown> };
    expect(r.details.validationError).toBe(true);
  });

  it("make_decision rejects selected index out of range", async () => {
    const { registerTeamTools } = await import("../tools/team-tools.js");
    const pi = makeMockPi();
    registerTeamTools(pi);
    const def = pi.getTool("make_decision") as { execute: (id: string, p: Record<string, unknown>, s: unknown, u: unknown, c: unknown) => unknown };
    const result = await def.execute("c1", {
      projectId: "proj-1", topic: "how to build", options: ["a", "b"], rationale: "because", selected: 99, madeBy: "pm",
    }, null, null, null);
    const r = result as { content: { text: string }[]; details: Record<string, unknown> };
    expect(r.details.validationError).toBe(true);
    expect(r.content[0].text).toContain("selected");
  });

  it("make_decision rejects fewer than 2 options", async () => {
    const { registerTeamTools } = await import("../tools/team-tools.js");
    const pi = makeMockPi();
    registerTeamTools(pi);
    const def = pi.getTool("make_decision") as { execute: (id: string, p: Record<string, unknown>, s: unknown, u: unknown, c: unknown) => unknown };
    const result = await def.execute("c1", {
      projectId: "proj-1", topic: "how to build", options: ["only-one"], rationale: "because", selected: 0, madeBy: "pm",
    }, null, null, null);
    const r = result as { content: { text: string }[]; details: Record<string, unknown> };
    expect(r.details.validationError).toBe(true);
    expect(r.content[0].text).toContain("options");
  });

  it("team_meeting rejects empty participants array", async () => {
    const { registerTeamTools } = await import("../tools/team-tools.js");
    const pi = makeMockPi();
    registerTeamTools(pi);
    const def = pi.getTool("team_meeting") as { execute: (id: string, p: Record<string, unknown>, s: unknown, u: unknown, c: unknown) => unknown };
    const result = await def.execute("c1", { projectId: "proj-1", topic: "standup", participants: [] }, null, null, null);
    const r = result as { content: { text: string }[]; details: Record<string, unknown> };
    expect(r.details.validationError).toBe(true);
  });
});
