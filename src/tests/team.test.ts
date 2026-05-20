/**
 * Team utilities — Unit Tests
 * Run: npx vitest run src/tests/team.test.ts --no-typecheck
 */

import { describe, it, expect } from "vitest";
import { PHASE_ROLES, ROLE_LABELS, PHASE_LABELS } from "../multi-agent-team/types.js";
import type { TeamRole, ProjectPhase } from "../multi-agent-team/types.js";

describe("PHASE_ROLES", () => {
  it("every phase has at least one role", () => {
    const phases: ProjectPhase[] = ["idea", "feasibility", "requirements", "design", "implementation", "testing", "evaluation"];
    for (const phase of phases) {
      expect(PHASE_ROLES[phase].length).toBeGreaterThan(0);
    }
  });

  it("idea and feasibility include business role", () => {
    expect(PHASE_ROLES.idea).toContain("business");
    expect(PHASE_ROLES.feasibility).toContain("business");
  });

  it("requirements includes product role", () => {
    expect(PHASE_ROLES.requirements).toContain("product");
  });

  it("design includes designer and developer", () => {
    expect(PHASE_ROLES.design).toContain("designer");
    expect(PHASE_ROLES.design).toContain("developer");
  });

  it("implementation includes developer", () => {
    expect(PHASE_ROLES.implementation).toContain("developer");
  });

  it("testing includes qa", () => {
    expect(PHASE_ROLES.testing).toContain("qa");
  });

  it("evaluation includes all roles", () => {
    expect(PHASE_ROLES.evaluation).toContain("pm");
    expect(PHASE_ROLES.evaluation).toContain("product");
    expect(PHASE_ROLES.evaluation).toContain("designer");
    expect(PHASE_ROLES.evaluation).toContain("developer");
    expect(PHASE_ROLES.evaluation).toContain("qa");
    expect(PHASE_ROLES.evaluation).toContain("business");
  });
});

describe("ROLE_LABELS", () => {
  const roles: TeamRole[] = ["pm", "product", "designer", "developer", "qa", "business"];
  for (const role of roles) {
    it(`${role} has a label`, () => {
      expect(ROLE_LABELS[role]).toBeTruthy();
      expect(ROLE_LABELS[role].length).toBeGreaterThan(0);
    });
  }
});

describe("PHASE_LABELS", () => {
  const phases: ProjectPhase[] = ["idea", "feasibility", "requirements", "design", "implementation", "testing", "evaluation"];
  for (const phase of phases) {
    it(`${phase} has a label`, () => {
      expect(PHASE_LABELS[phase]).toBeTruthy();
    });
  }
});
