/**
 * PhaseController — Unit Tests
 * Run: npx vitest run src/tests/phase-controller.test.ts --no-typecheck
 */

import { describe, it, expect } from "vitest";
import { PHASE_ORDER, PHASE_GATES, phaseController } from "../multi-agent-team/phase-controller.js";
import type { Project, ProjectPhase, TeamRole } from "../multi-agent-team/types.js";

function makeArtifact(type: string, status: "draft" | "approved" | "pending" = "draft") {
  return {
    id: `${type}-id`,
    projectId: "test-id",
    type: type as any,
    title: type,
    content: "test content",
    phase: "idea" as ProjectPhase,
    createdBy: "pm" as TeamRole,
    createdAt: new Date().toISOString(),
    version: 1,
    status: status as any,
    reviewers: [],
  };
}

function makeProject(phase: ProjectPhase, artifacts: ReturnType<typeof makeArtifact>[] = []): Project {
  return {
    id: "test-id",
    name: "Test Project",
    description: "Test",
    phase,
    status: "planning",
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    members: [],
    artifacts,
    decisions: [],
    currentBlocker: undefined,
  };
}

describe("PHASE_ORDER", () => {
  it("has 7 phases in correct order", () => {
    expect(PHASE_ORDER).toHaveLength(7);
    expect(PHASE_ORDER[0]).toBe("idea");
    expect(PHASE_ORDER[6]).toBe("evaluation");
  });
});

describe("PHASE_GATES", () => {
  it("idea requires idea_form", () => {
    expect(PHASE_GATES.idea.requiredArtifacts).toContain("idea_form");
  });

  it("requirements requires prd and user_story_map", () => {
    expect(PHASE_GATES.requirements.requiredArtifacts).toContain("prd");
    expect(PHASE_GATES.requirements.requiredArtifacts).toContain("user_story_map");
  });

  it("design requires design_spec and tech_spec", () => {
    expect(PHASE_GATES.design.requiredArtifacts).toContain("design_spec");
    expect(PHASE_GATES.design.requiredArtifacts).toContain("tech_spec");
  });

  it("testing requires test_plan and test_report", () => {
    expect(PHASE_GATES.testing.requiredArtifacts).toContain("test_plan");
    expect(PHASE_GATES.testing.requiredArtifacts).toContain("test_report");
  });

  it("all phases have noBlockers: true", () => {
    for (const phase of PHASE_ORDER) {
      expect(PHASE_GATES[phase].noBlockers).toBe(true);
    }
  });
});

describe("PhaseController.canAdvance()", () => {
  it("allows advance from idea when idea_form is approved", () => {
    const proj = makeProject("idea", [makeArtifact("idea_form", "approved")]);
    const result = phaseController.canAdvance(proj);
    expect(result.ok).toBe(true);
  });

  it("blocks advance when required artifact is missing", () => {
    const proj = makeProject("idea", []);
    const result = phaseController.canAdvance(proj);
    expect(result.ok).toBe(false);
    expect(result.missingArtifacts).toContain("idea_form");
  });

  it("blocks advance when required artifact is pending (not approved)", () => {
    const proj = makeProject("idea", [makeArtifact("idea_form", "pending")]);
    const result = phaseController.canAdvance(proj);
    expect(result.ok).toBe(false);
    expect(result.missingArtifacts).toContain("idea_form (pending approval)");
  });

  it("blocks advance when project has a currentBlocker", () => {
    const proj = makeProject("idea", [makeArtifact("idea_form", "approved")]);
    proj.currentBlocker = "Waiting for design assets";
    const result = phaseController.canAdvance(proj);
    expect(result.ok).toBe(false);
    expect(result.reason).toContain("Waiting for design assets");
  });

  it("returns ok=false at final phase", () => {
    const proj = makeProject("evaluation", [
      makeArtifact("assessment", "approved"),
      makeArtifact("retrospective", "approved"),
    ]);
    const result = phaseController.canAdvance(proj);
    expect(result.ok).toBe(false);
    expect(result.reason).toContain("final phase");
  });

  it("returns ok=false for unknown phase", () => {
    const proj = makeProject("idea", []);
    (proj as any).phase = "unknown" as ProjectPhase;
    const result = phaseController.canAdvance(proj);
    expect(result.ok).toBe(false);
  });
});

describe("PhaseController.advance()", () => {
  it("advances phase when gate is satisfied", () => {
    const proj = makeProject("idea", [makeArtifact("idea_form", "approved")]);
    const updated = phaseController.advance(proj);
    expect(updated.phase).toBe("feasibility");
  });

  it("advance() does NOT check gate — just moves phase forward", () => {
    // advance() is unconditional; canAdvance() is the gate keeper
    const proj = makeProject("idea", []); // no artifacts, gate not met
    const updated = phaseController.advance(proj);
    // advance() blindly moves to next phase regardless of gate
    expect(updated.phase).toBe("feasibility");
  });

  it("returns unchanged project at final phase", () => {
    const proj = makeProject("evaluation", [
      makeArtifact("assessment", "approved"),
      makeArtifact("retrospective", "approved"),
    ]);
    const updated = phaseController.advance(proj);
    expect(updated.phase).toBe("evaluation");
  });
});

describe("PhaseController.setBlocker() / clearBlocker()", () => {
  it("setBlocker adds blocker and sets status to blocked", () => {
    const proj = makeProject("idea");
    const updated = phaseController.setBlocker(proj, "Missing budget approval");
    expect(updated.currentBlocker).toBe("Missing budget approval");
    expect(updated.status).toBe("blocked");
  });

  it("clearBlocker removes blocker and sets status to active", () => {
    const proj = makeProject("idea");
    proj.currentBlocker = "old blocker";
    const updated = phaseController.clearBlocker(proj);
    expect(updated.currentBlocker).toBeUndefined();
    expect(updated.status).toBe("active");
  });
});

describe("PhaseController.progress()", () => {
  it("returns current, completed, pending phases and percentage", () => {
    const proj = makeProject("requirements");
    const prog = phaseController.progress(proj);
    expect(prog.current).toBe("requirements");
    expect(prog.completed).toContain("idea");
    expect(prog.completed).toContain("feasibility");
    expect(prog.pending).toContain("design");
    expect(prog.pct).toBeGreaterThan(0);
    expect(prog.pct).toBeLessThan(100);
  });

  it("idea phase shows 0%", () => {
    const proj = makeProject("idea");
    expect(phaseController.progress(proj).pct).toBe(0);
  });

  it("evaluation phase shows 100%", () => {
    const proj = makeProject("evaluation");
    expect(phaseController.progress(proj).pct).toBe(100);
  });
});

describe("PhaseController.review()", () => {
  it("creates a phase review record", () => {
    const proj = makeProject("idea", [makeArtifact("idea_form", "approved")]);
    const review = phaseController.review(proj, "pm", "pass", "Looks good");
    expect(review.phase).toBe("idea");
    expect(review.verdict).toBe("pass");
    expect(review.comments).toBe("Looks good");
    expect(review.reviewedBy).toBe("pm");
    expect(review.deliverables.length).toBeGreaterThan(0);
  });
});
