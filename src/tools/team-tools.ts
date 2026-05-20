/**
 * Team & Project Tools — create_project, list_projects, get_project,
 * plan_phase, run_role, create_artifact, review_artifact, advance_phase,
 * make_decision, team_meeting, generate_doc
 */

import { Type } from "typebox";
import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { projectStore } from "../multi-agent-team/project-store.js";
import { projectManager } from "../multi-agent-team/project-manager.js";
import { phaseController } from "../multi-agent-team/phase-controller.js";
import { meetingManager } from "../multi-agent-team/meeting.js";
import {
  runRole,
  createArtifact,
  reviewArtifact,
  formatTeamStatus,
} from "../multi-agent-team/team.js";
import { productManager } from "../multi-agent-team/role-agents/product-agent.js";
import { developer } from "../multi-agent-team/role-agents/developer-agent.js";
import { qaTester } from "../multi-agent-team/role-agents/qa-agent.js";
import { businessAnalyst } from "../multi-agent-team/role-agents/business-agent.js";
import { designer } from "../multi-agent-team/role-agents/design-agent.js";
import type { TeamRole, ProjectPhase, ArtifactType } from "../multi-agent-team/types.js";

// ─── Runtime validation helpers ─────────────────────────────────────────────────

/** A validation error returned as a tool result */
function validationError(
  message: string,
  details: Record<string, unknown> = {}
): { content: { type: "text"; text: string }[]; details: Record<string, unknown> } {
  return {
    content: [{ type: "text", text: `Validation error: ${message}` }],
    details: { ...details, validationError: true },
  };
}

function requireString(
  value: unknown,
  name: string
): { content: { type: "text"; text: string }[]; details: Record<string, unknown> } | null {
  if (value == null || (typeof value === "string" && !value.trim())) {
    return validationError(`${name} is required and must be a non-empty string`, {
      param: name,
    });
  }
  return null;
}

function validateEnum(
  value: unknown,
  name: string,
  allowed: string[]
): { content: { type: "text"; text: string }[]; details: Record<string, unknown> } | null {
  if (value == null) return null;
  if (typeof value !== "string" || !allowed.includes(value)) {
    return validationError(`${name} must be one of: ${allowed.join(", ")}`, {
      param: name,
      value,
    });
  }
  return null;
}

function validatePositiveNumber(
  value: unknown,
  name: string,
  message: string
): { content: { type: "text"; text: string }[]; details: Record<string, unknown> } | null {
  if (value == null) return null;
  if (typeof value !== "number" || value < 0) {
    return validationError(`${name} must be ${message}`, { param: name, value });
  }
  return null;
}

export function registerTeamTools(pi: ExtensionAPI): void {
  const CreateProjectParams = Type.Object({
    name: Type.String({ description: "Project name" }),
    description: Type.String({ description: "Project description / idea" }),
    target: Type.Optional(Type.String({ description: "Success target (optional)" })),
    successCriteria: Type.Optional(Type.String({ description: "How to measure success (optional)" })),
  });

  const ListProjectsParams = Type.Object({
    status: Type.Optional(Type.String({ description: "Filter by status: active, planning, completed" })),
  });

  const GetProjectParams = Type.Object({
    projectId: Type.String({ description: "Project ID" }),
  });

  const PlanPhaseParams = Type.Object({
    projectId: Type.String({ description: "Project ID" }),
    phase: Type.String({
      description: "Phase to plan",
      enum: ["idea", "feasibility", "requirements", "design", "implementation", "testing", "evaluation"],
    }),
  });

  const RunRoleParams = Type.Object({
    projectId: Type.String({ description: "Project ID" }),
    role: Type.String({
      description: "Role to execute",
      enum: ["pm", "product", "designer", "developer", "qa", "business"],
    }),
    phase: Type.String({
      description: "Current phase",
      enum: ["idea", "feasibility", "requirements", "design", "implementation", "testing", "evaluation"],
    }),
    input: Type.Optional(Type.String({ description: "JSON-encoded input data for the role" })),
    workdir: Type.Optional(Type.String({ description: "Working directory (optional)" })),
  });

  const CreateArtifactParams = Type.Object({
    projectId: Type.String({ description: "Project ID" }),
    type: Type.String({
      description: "Artifact type",
      enum: ["idea_form", "feasibility_report", "prd", "user_story_map", "design_spec", "tech_spec", "code", "test_plan", "test_report", "defect_list", "assessment", "meeting_notes", "decision_record", "retrospective"],
    }),
    title: Type.String({ description: "Artifact title" }),
    content: Type.String({ description: "Artifact content (markdown)" }),
    phase: Type.String({
      description: "Phase",
      enum: ["idea", "feasibility", "requirements", "design", "implementation", "testing", "evaluation"],
    }),
    createdBy: Type.Optional(Type.String({
      description: "Role creating this artifact (default: pm). Should match the actual role performing the work.",
      enum: ["pm", "product", "designer", "developer", "qa", "business"],
    })),
    summary: Type.Optional(Type.String({ description: "Short summary (optional)" })),
  });

  const ReviewArtifactParams = Type.Object({
    projectId: Type.String({ description: "Project ID" }),
    artifactId: Type.String({ description: "Artifact ID" }),
    verdict: Type.String({ description: "Verdict", enum: ["approve", "reject"] }),
    comment: Type.Optional(Type.String({ description: "Review comment (optional)" })),
  });

  const AdvancePhaseParams = Type.Object({
    projectId: Type.String({ description: "Project ID" }),
  });

  const MakeDecisionParams = Type.Object({
    projectId: Type.String({ description: "Project ID" }),
    topic: Type.String({ description: "Decision topic" }),
    options: Type.Array(Type.String(), { description: "Options array" }),
    rationale: Type.String({ description: "Reason for the decision" }),
    selected: Type.Number({ description: "Index of selected option (0-based)" }),
    madeBy: Type.String({ description: "Role making the decision", enum: ["pm", "product", "designer", "developer", "qa", "business"] }),
  });

  const TeamMeetingParams = Type.Object({
    projectId: Type.String({ description: "Project ID" }),
    topic: Type.String({ description: "Meeting topic" }),
    participants: Type.Array(Type.String(), { description: "Roles to invite" }),
    notes: Type.Optional(Type.String({ description: "Meeting notes (optional)" })),
  });

  // ─── create_project ───────────────────────────────────────────────────────
  pi.registerTool(defineTool({
    name: "create_project",
    label: "Create Project",
    description: "Create a new project team with a multi-role agent group.",
    parameters: CreateProjectParams,

    async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
      const err1 = requireString(params.name, "name");
      if (err1) return err1;
      const err2 = requireString(params.description, "description");
      if (err2) return err2;

      const project = projectStore.create({
        name: params.name,
        description: params.description,
        target: params.target,
        successCriteria: params.successCriteria,
      });
      const lines = [
        `✓ Project created: ${project.name}`,
        `  ID: ${project.id}`,
        `  Phase: ${project.phase}`,
        `  Description: ${project.description}`,
        "",
        "Next: use plan_phase to start the idea phase.",
      ];
      return { content: [{ type: "text", text: lines.join("\n") }], details: { projectId: project.id } };
    },
  }));

  // ─── list_projects ────────────────────────────────────────────────────────
  pi.registerTool(defineTool({
    name: "list_projects",
    label: "List Projects",
    description: "List all projects. Optionally filter by status.",
    parameters: ListProjectsParams,

    async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
      let projects = projectStore.list();
      if (params.status) {
        projects = projects.filter((p) => p.status === params.status);
      }
      if (projects.length === 0) {
        return { content: [{ type: "text", text: "No projects found." }], details: {} };
      }
      const lines = projects.map((p) =>
        `[${p.status}] ${p.name} (${p.id}) — ${p.phase} | ${new Date(p.updatedAt).toLocaleDateString("zh-CN")}`
      );
      return { content: [{ type: "text", text: lines.join("\n") }], details: { projects } };
    },
  }));

  // ─── get_project ─────────────────────────────────────────────────────────
  pi.registerTool(defineTool({
    name: "get_project",
    label: "Get Project",
    description: "Get full project details and team status.",
    parameters: GetProjectParams,

    async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
      const text = formatTeamStatus(params.projectId);
      return { content: [{ type: "text", text }], details: {} };
    },
  }));

  // ─── plan_phase ───────────────────────────────────────────────────────────
  pi.registerTool(defineTool({
    name: "plan_phase",
    label: "Plan Phase",
    description: "Generate a detailed phase plan with roles, deliverables, and risks.",
    parameters: PlanPhaseParams,

    async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
      const err1 = requireString(params.projectId, "projectId");
      if (err1) return err1;
      const err2 = validateEnum(params.phase, "phase", [
        "idea", "feasibility", "requirements", "design",
        "implementation", "testing", "evaluation",
      ]);
      if (err2) return err2;

      const result = projectManager.planPhase(params.projectId, params.phase as ProjectPhase);
      if (!result.ok) {
        return { content: [{ type: "text", text: `Project not found: ${params.projectId}` }], details: {} };
      }
      return { content: [{ type: "text", text: result.plan }], details: {} };
    },
  }));

  // ─── run_role ─────────────────────────────────────────────────────────────
  pi.registerTool(defineTool({
    name: "run_role",
    label: "Run Role",
    description: "Execute a specific team role for the current phase.",
    parameters: RunRoleParams,

    async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
      const err1 = requireString(params.projectId, "projectId");
      if (err1) return err1;
      const err2 = validateEnum(params.role, "role", [
        "pm", "product", "designer", "developer", "qa", "business",
      ]);
      if (err2) return err2;
      const err3 = validateEnum(params.phase, "phase", [
        "idea", "feasibility", "requirements", "design",
        "implementation", "testing", "evaluation",
      ]);
      if (err3) return err3;

      const workdir = params.workdir ?? process.cwd();
      let input: unknown = {};
      if (params.input) {
        try { input = JSON.parse(params.input); } catch { /* ignore */ }
      }
      const result = await runRole(
        params.projectId,
        params.role as TeamRole,
        params.phase as ProjectPhase,
        input,
        workdir,
      );
      if (!result.ok || !result.result) {
        return { content: [{ type: "text", text: `Role ${params.role} failed to execute.` }], details: {} };
      }
      const r = result.result;
      const lines = [
        `=== ${r.role} @ ${params.phase} ===`,
        `Status: ${r.status}`,
        r.blockedReason ? `Blocked: ${r.blockedReason}` : "",
        "",
        `Summary: ${r.summary}`,
        "",
        r.artifacts.length > 0 ? `Artifacts: ${r.artifacts.length} created` : "",
        r.nextActions.length > 0 ? `Next: ${r.nextActions.join(" → ")}` : "",
      ].filter(Boolean);
      return { content: [{ type: "text", text: lines.join("\n") }], details: { result: r } };
    },
  }));

  // ─── create_artifact ─────────────────────────────────────────────────────
  pi.registerTool(defineTool({
    name: "create_artifact",
    label: "Create Artifact",
    description: "Create a project artifact (PRD, design spec, test plan, etc.)",
    parameters: CreateArtifactParams,

    async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
      const err1 = requireString(params.projectId, "projectId");
      if (err1) return err1;
      const err2 = requireString(params.title, "title");
      if (err2) return err2;
      const err3 = requireString(params.content, "content");
      if (err3) return err3;
      const err4 = validateEnum(params.phase, "phase", [
        "idea", "feasibility", "requirements", "design",
        "implementation", "testing", "evaluation",
      ]);
      if (err4) return err4;

      try {
        const artifact = createArtifact({
          projectId: params.projectId,
          type: params.type as ArtifactType,
          title: params.title,
          content: params.content,
          phase: params.phase as ProjectPhase,
          createdBy: (params.createdBy ?? "pm") as TeamRole,
          summary: params.summary,
        });
        return {
          content: [{ type: "text", text: `Artifact created: ${params.title} (${params.type})\nID: ${artifact.id}` }],
          details: { artifactId: artifact.id },
        };
      } catch (e) {
        return { content: [{ type: "text", text: `Error: ${e}` }], details: {} };
      }
    },
  }));

  // ─── review_artifact ─────────────────────────────────────────────────────
  pi.registerTool(defineTool({
    name: "review_artifact",
    label: "Review Artifact",
    description: "Approve or reject a project artifact.",
    parameters: ReviewArtifactParams,

    async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
      const artifact = reviewArtifact(
        params.projectId,
        params.artifactId,
        params.verdict as "approve" | "reject",
        params.comment ?? "",
        "pm",
      );
      if (!artifact) {
        return { content: [{ type: "text", text: "Artifact not found." }], details: {} };
      }
      return {
        content: [{ type: "text", text: `Artifact ${artifact.id}: ${params.verdict.toUpperCase()}\nStatus: ${artifact.status}` }],
        details: { artifact },
      };
    },
  }));

  // ─── advance_phase ────────────────────────────────────────────────────────
  pi.registerTool(defineTool({
    name: "advance_phase",
    label: "Advance Phase",
    description: "Advance the project to the next phase if all gate criteria are met.",
    parameters: AdvancePhaseParams,

    async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
      const err1 = requireString(params.projectId, "projectId");
      if (err1) return err1;

      const project = projectStore.get(params.projectId);
      if (!project) {
        return { content: [{ type: "text", text: "Project not found." }], details: {} };
      }
      const gate = phaseController.canAdvance(project);
      if (!gate.ok) {
        const missing = gate.missingArtifacts?.join(", ") ?? gate.reason ?? "unknown";
        return {
          content: [{ type: "text", text: `Cannot advance: ${missing}` }],
          details: { blocked: true, reason: missing },
        };
      }
      const updated = phaseController.advance(project);
      projectStore.update(updated);
      return {
        content: [{
          type: "text",
          text: `✓ Phase advanced: ${project.phase} → ${updated.phase}`,
        }],
        details: { newPhase: updated.phase },
      };
    },
  }));

  // ─── make_decision ────────────────────────────────────────────────────────
  pi.registerTool(defineTool({
    name: "make_decision",
    label: "Make Decision",
    description: "Record a project decision with options, rationale, and outcome.",
    parameters: MakeDecisionParams,

    async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
      const err1 = requireString(params.projectId, "projectId");
      if (err1) return err1;
      const err2 = requireString(params.topic, "topic");
      if (err2) return err2;
      if (!Array.isArray(params.options) || params.options.length < 2) {
        return validationError("options must be an array with at least 2 items", {
          param: "options",
        });
      }
      if (typeof params.selected !== "number" || params.selected < 0 || params.selected >= params.options.length) {
        return validationError(`selected must be a number between 0 and ${params.options.length - 1}`, {
          param: "selected",
          value: params.selected,
        });
      }
      const err3 = validateEnum(params.madeBy, "madeBy", [
        "pm", "product", "designer", "developer", "qa", "business",
      ]);
      if (err3) return err3;

      const project = projectStore.get(params.projectId);
      if (!project) {
        return { content: [{ type: "text", text: "Project not found." }], details: {} };
      }
      const decision = meetingManager.createDecision({
        projectId: params.projectId,
        topic: params.topic,
        options: params.options,
        rationale: params.rationale,
        selected: params.selected,
        madeBy: params.madeBy as TeamRole,
        phase: project.phase,
      });
      projectStore.addDecision(params.projectId, decision);
      const lines = [
        `✓ Decision recorded: ${params.topic}`,
        `Selected: ${params.options[params.selected]}`,
        `Rationale: ${params.rationale}`,
      ];
      return { content: [{ type: "text", text: lines.join("\n") }], details: { decisionId: decision.id } };
    },
  }));

  // ─── team_meeting ────────────────────────────────────────────────────────
  pi.registerTool(defineTool({
    name: "team_meeting",
    label: "Team Meeting",
    description: "Create a meeting record and collect outcomes.",
    parameters: TeamMeetingParams,

    async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
      const err1 = requireString(params.projectId, "projectId");
      if (err1) return err1;
      const err2 = requireString(params.topic, "topic");
      if (err2) return err2;
      if (!Array.isArray(params.participants) || params.participants.length === 0) {
        return validationError("participants must be a non-empty array of role strings", {
          param: "participants",
        });
      }

      const meeting = meetingManager.createMeeting({
        projectId: params.projectId,
        topic: params.topic,
        participants: params.participants as TeamRole[],
        notes: params.notes,
      });
      return {
        content: [{ type: "text", text: `Meeting created: ${params.topic}\nID: ${meeting.id}\nParticipants: ${params.participants.join(", ")}` }],
        details: { meetingId: meeting.id },
      };
    },
  }));

  // ─── generate_doc ────────────────────────────────────────────────────────
  pi.registerTool(defineTool({
    name: "generate_doc",
    label: "Generate Document",
    description: "Generate a document from a specific role agent (prd, tech_spec, test_plan, feasibility_report, design_spec).",
    parameters: Type.Object({
      type: Type.String({
        description: "Doc type",
        enum: ["prd", "tech_spec", "test_plan", "feasibility_report", "design_spec"],
      }),
      projectName: Type.String({ description: "Project name" }),
      input: Type.Optional(Type.String({ description: "JSON input for the generator" })),
    }),

    async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
      let content = "";
      let title = params.type.toUpperCase();

      switch (params.type) {
        case "prd": {
          const inp = params.input ? JSON.parse(params.input) : {};
          content = productManager.generatePRD({ productName: params.projectName, goal: inp.goal ?? "", targetUsers: inp.targetUsers ?? [], features: inp.features ?? [] });
          title = "PRD";
          break;
        }
        case "tech_spec": {
          const inp = params.input ? JSON.parse(params.input) : {};
          content = developer.generateTechSpec({ projectName: params.projectName, features: inp.features ?? [], techStack: inp.techStack });
          title = "Tech Spec";
          break;
        }
        case "test_plan": {
          const inp = params.input ? JSON.parse(params.input) : {};
          content = qaTester.generateTestPlan({ projectName: params.projectName, features: inp.features ?? [] });
          title = "Test Plan";
          break;
        }
        case "feasibility_report": {
          const inp = params.input ? JSON.parse(params.input) : {};
          content = businessAnalyst.generateFeasibilityReport({ idea: inp.idea ?? params.projectName, targetMarket: inp.targetMarket, competitors: inp.competitors });
          title = "Feasibility Report";
          break;
        }
        case "design_spec": {
          const inp = params.input ? JSON.parse(params.input) : {};
          content = designer.generateDesignSpec({ productName: params.projectName, pages: inp.pages ?? [], userFlows: inp.userFlows ?? [] });
          title = "Design Spec";
          break;
        }
      }

      return { content: [{ type: "text", text: content }], details: {} };
    },
  }));
}
