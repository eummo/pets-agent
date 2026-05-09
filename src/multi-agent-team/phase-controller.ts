/**
 * PhaseController — state machine for project phases.
 *
 * Manages phase transitions, validates gate criteria, tracks blockers.
 */

import type {
  ProjectPhase,
  Project,
  Artifact,
  PhaseReview,
  TeamRole,
} from "./types.js";
import { PHASE_LABELS } from "./types.js";

export const PHASE_ORDER: ProjectPhase[] = [
  "idea",
  "feasibility",
  "requirements",
  "design",
  "implementation",
  "testing",
  "evaluation",
];

export interface GateCriteria {
  requiredArtifacts: Artifact["type"][];
  minDecisions?: number;
  noBlockers?: boolean;
}

export const PHASE_GATES: Record<ProjectPhase, GateCriteria> = {
  idea: {
    requiredArtifacts: ["idea_form"],
    noBlockers: true,
  },
  feasibility: {
    requiredArtifacts: ["feasibility_report"],
    noBlockers: true,
  },
  requirements: {
    requiredArtifacts: ["prd", "user_story_map"],
    noBlockers: true,
  },
  design: {
    requiredArtifacts: ["design_spec", "tech_spec"],
    noBlockers: true,
  },
  implementation: {
    requiredArtifacts: ["code"],
    noBlockers: true,
  },
  testing: {
    requiredArtifacts: ["test_plan", "test_report"],
    noBlockers: true,
  },
  evaluation: {
    requiredArtifacts: ["assessment", "retrospective"],
    noBlockers: true,
  },
};

export class PhaseController {
  /**
   * Check if a project is ready to advance to the next phase.
   */
  canAdvance(project: Project): { ok: boolean; reason?: string; missingArtifacts?: string[] } {
    const currentIdx = PHASE_ORDER.indexOf(project.phase);
    if (currentIdx === -1 || currentIdx === PHASE_ORDER.length - 1) {
      return { ok: false, reason: "Already at final phase or unknown phase." };
    }

    const gate = PHASE_GATES[project.phase];
    const nextPhase = PHASE_ORDER[currentIdx + 1];

    // Check no blockers
    if (gate.noBlockers && project.currentBlocker) {
      return { ok: false, reason: `Current blocker: ${project.currentBlocker}` };
    }

    // Check required artifacts exist and are approved
    const existingTypes = new Set(project.artifacts.map((a) => a.type));
    const approved = new Set(
      project.artifacts
        .filter((a) => a.status === "approved")
        .map((a) => a.type)
    );

    const missing: string[] = [];
    for (const req of gate.requiredArtifacts) {
      if (!existingTypes.has(req)) {
        missing.push(req);
      } else if (!approved.has(req)) {
        missing.push(`${req} (pending approval)`);
      }
    }

    if (missing.length > 0) {
      return { ok: false, missingArtifacts: missing };
    }

    // Check minimum decisions
    if (gate.minDecisions !== undefined) {
      if (project.decisions.length < gate.minDecisions) {
        return {
          ok: false,
          reason: `Need at least ${gate.minDecisions} decisions, have ${project.decisions.length}`,
        };
      }
    }

    return { ok: true };
  }

  /**
   * Advance to next phase.
   */
  advance(project: Project): Project {
    const currentIdx = PHASE_ORDER.indexOf(project.phase);
    if (currentIdx === -1 || currentIdx === PHASE_ORDER.length - 1) {
      return project;
    }

    const nextPhase = PHASE_ORDER[currentIdx + 1];
    return {
      ...project,
      phase: nextPhase,
      updatedAt: new Date().toISOString(),
      currentBlocker: undefined,
    };
  }

  /**
   * Set a blocker on the current phase.
   */
  setBlocker(project: Project, blocker: string): Project {
    return {
      ...project,
      currentBlocker: blocker,
      status: "blocked",
      updatedAt: new Date().toISOString(),
    };
  }

  /**
   * Clear blocker and resume.
   */
  clearBlocker(project: Project): Project {
    return {
      ...project,
      currentBlocker: undefined,
      status: "active",
      updatedAt: new Date().toISOString(),
    };
  }

  /**
   * Generate phase progress summary.
   */
  progress(project: Project): {
    current: ProjectPhase;
    completed: ProjectPhase[];
    pending: ProjectPhase[];
    pct: number;
  } {
    const currentIdx = PHASE_ORDER.indexOf(project.phase);
    return {
      current: project.phase,
      completed: PHASE_ORDER.slice(0, currentIdx),
      pending: PHASE_ORDER.slice(currentIdx + 1),
      pct: Math.round((currentIdx / (PHASE_ORDER.length - 1)) * 100),
    };
  }

  /**
   * Create a phase review record.
   */
  review(
    project: Project,
    reviewedBy: TeamRole,
    verdict: PhaseReview["verdict"],
    comments: string
  ): PhaseReview {
    const gate = PHASE_GATES[project.phase];
    const allArtifacts = project.artifacts.filter((a) => a.phase === project.phase);
    const approvedArtifacts = allArtifacts.filter((a) => a.status === "approved");

    return {
      phase: project.phase,
      deliverables: gate.requiredArtifacts.map((t) => ({
        id: "",
        projectId: project.id,
        type: t,
        title: t,
        content: "",
        phase: project.phase,
        createdBy: "pm",
        createdAt: "",
        version: 0,
        status: "draft" as const,
        reviewers: [],
      })),
      actualDeliverables: allArtifacts,
      blockers: project.currentBlocker ? [project.currentBlocker] : [],
      verdict,
      comments,
      reviewedBy,
      reviewedAt: new Date().toISOString(),
    };
  }
}

export const phaseController = new PhaseController();
