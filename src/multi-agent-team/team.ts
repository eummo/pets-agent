/**
 * Team — team initialization, orchestration, and role factory.
 *
 * Provides the unified interface for spawning and coordinating
 * all team roles within a project context.
 */

import type { Project, TeamRole, ProjectPhase, Artifact, ArtifactType } from "./types.js";
import { PHASE_ROLES, ROLE_LABELS } from "./types.js";
import { projectStore } from "./project-store.js";
import { projectManager } from "./project-manager.js";
import { businessAnalyst } from "./role-agents/business-agent.js";
import { productManager } from "./role-agents/product-agent.js";
import { designer } from "./role-agents/design-agent.js";
import { developer } from "./role-agents/developer-agent.js";
import { qaTester } from "./role-agents/qa-agent.js";
import { meetingManager } from "./meeting.js";
import { phaseController, PHASE_GATES } from "./phase-controller.js";
import { Role, type RoleContext } from "./role.js";
import { randomBytes } from "crypto";
import type { Artifact as ArtifactType2 } from "./types.js";

const ROLE_CLASSES: Record<TeamRole, Role> = {
  pm: projectManager,
  product: productManager,
  designer,
  developer,
  qa: qaTester,
  business: businessAnalyst,
};

/**
 * Spawn a specific role to execute work for a project.
 */
export async function runRole(
  projectId: string,
  role: TeamRole,
  phase: ProjectPhase,
  input: unknown,
  workdir: string,
  opts?: { signal?: AbortSignal; timeoutMs?: number }
): Promise<{ ok: boolean; result?: Awaited<ReturnType<Role["run"]>> }> {
  const ctx: RoleContext = {
    projectId,
    phase,
    input,
    workdir,
    signal: opts?.signal,
    timeoutMs: opts?.timeoutMs,
  };

  const roleInstance = ROLE_CLASSES[role];
  if (!roleInstance) {
    return { ok: false };
  }

  try {
    const result = await roleInstance.run(ctx);
    return { ok: true, result };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    console.error(`[Team] Role ${role} failed:`, message);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return { ok: false, result: { error: message } } as any;
  }
}

/**
 * Run all roles for a given phase in parallel.
 */
export async function runPhaseRoles(
  projectId: string,
  phase: ProjectPhase,
  input: unknown,
  workdir: string,
  opts?: { signal?: AbortSignal; timeoutMs?: number }
): Promise<Array<{ role: TeamRole; ok: boolean; result?: Awaited<ReturnType<Role["run"]>> }>> {
  const roles = PHASE_ROLES[phase];
  const runs = roles.map((role) => runRole(projectId, role, phase, input, workdir, opts));
  const results = await Promise.all(runs);
  return roles.map((role, i) => ({ role, ...results[i] }));
}

/**
 * Create an artifact and add it to the project.
 */
export function createArtifact(params: {
  projectId: string;
  type: ArtifactType;
  title: string;
  content: string;
  phase: ProjectPhase;
  createdBy: TeamRole;
  summary?: string;
}): Artifact {
  const project = projectStore.get(params.projectId);
  if (!project) throw new Error(`Project not found: ${params.projectId}`);

  const artifact: Artifact = {
    id: randomBytes(8).toString("hex"),
    projectId: params.projectId,
    type: params.type,
    title: params.title,
    content: params.content,
    phase: params.phase,
    createdBy: params.createdBy,
    createdAt: new Date().toISOString(),
    version: 1,
    status: "draft",
    reviewers: [],
    summary: params.summary,
  };

  projectStore.addArtifact(params.projectId, artifact);
  return artifact;
}

/**
 * Review an artifact and update its status.
 */
export function reviewArtifact(
  projectId: string,
  artifactId: string,
  verdict: "approve" | "reject",
  comment: string,
  reviewedBy: TeamRole
): Artifact | null {
  const project = projectStore.get(projectId);
  if (!project) return null;

  const artifact = project.artifacts.find((a) => a.id === artifactId);
  if (!artifact) return null;

  artifact.reviewers.push({ role: reviewedBy, verdict, comment });

  if (verdict === "reject") {
    // Check if any-rejection-veto gate applies
    const gate = PHASE_GATES[artifact.phase];
    if (gate.anyRejectionVeto) {
      artifact.status = "rejected";
    } else {
      // Softer rejection — still pending, just recorded
      artifact.status = artifact.status === "in_review" ? "in_review" : "draft";
    }
  } else {
    // Approve: check configurable minApprovals threshold
    const gate = PHASE_GATES[artifact.phase];
    const phaseRoles = PHASE_ROLES[artifact.phase];
    const approvals = artifact.reviewers.filter((r) => r.verdict === "approve").length;
    const minApprovals = gate.minApprovals ?? Math.ceil(phaseRoles.length / 2);
    if (approvals >= minApprovals) {
      artifact.status = "approved";
    } else {
      artifact.status = "in_review";
    }
  }

  projectStore.updateArtifact(projectId, artifact);
  return artifact;
}

/**
 * Get full team status for a project.
 */
export function getTeamStatus(projectId: string) {
  const project = projectStore.get(projectId);
  if (!project) return null;

  const progress = phaseController.progress(project);
  const pendingArtifacts = project.artifacts.filter(
    (a) => a.status === "draft" || a.status === "in_review"
  );
  const openDecisions = project.decisions;

  return {
    project,
    phaseProgress: progress,
    pendingArtifacts: pendingArtifacts.map((a) => ({
      id: a.id,
      type: a.type,
      title: a.title,
      status: a.status,
    })),
    openDecisions,
    blockers: project.currentBlocker ? [project.currentBlocker] : [],
    nextActions: getNextActions(project),
  };
}

function getNextActions(project: Project): string[] {
  const actions: string[] = [];
  const phase = project.phase;

  // Pre-filter once, use throughout
  const draftArtifacts = project.artifacts.filter((a) => a.status === "draft");
  const inReviewArtifacts = project.artifacts.filter((a) => a.status === "in_review");
  const draftTypes = new Set(draftArtifacts.map((a) => a.type));
  const inReviewTypes = new Set(inReviewArtifacts.map((a) => a.type));
  const allArtifactTypes = new Set(project.artifacts.map((a) => a.type));
  const memberRoles = new Set(project.members.map((m) => m.role));

  if (project.currentBlocker) {
    actions.push(`⚠️ 当前阻塞: ${project.currentBlocker}`);
  }

  if (phase === "idea") {
    if (!allArtifactTypes.has("idea_form")) actions.push("Business analyst: 评估想法可行性和商业价值");
    if (memberRoles.has("pm")) actions.push("PM: 主持想法评审会议");
    else actions.push("请分配 PM 角色");
  }

  if (phase === "feasibility") {
    if (draftTypes.has("feasibility_report")) {
      actions.push("Business analyst: 完成可行性报告并提交评审");
    }
    if (!memberRoles.has("developer")) {
      actions.push("请分配 Developer 角色参与技术可行性评审");
    } else {
      actions.push("Developer: 技术可行性评审");
    }
    if (inReviewArtifacts.length > 0) {
      actions.push("PM: 评审可行性报告，决定是否进入需求阶段");
    }
  }

  if (phase === "requirements") {
    if (draftTypes.has("prd")) {
      actions.push("Product manager: 编写PRD");
    }
    if (!memberRoles.has("designer")) {
      actions.push("请分配 Designer 角色参与需求评审");
    } else {
      actions.push("Designer: 参与用户体验需求评审");
    }
    if (inReviewTypes.has("prd")) {
      actions.push("PM: 冻结需求范围，准备进入设计阶段");
    }
  }

  if (phase === "design") {
    if (draftTypes.has("design_spec")) {
      actions.push("Designer: 完成UX设计规格");
    }
    if (!draftTypes.has("tech_spec")) {
      actions.push("Developer: 完成技术架构设计");
    }
    if (inReviewTypes.has("design_spec") || inReviewTypes.has("tech_spec")) {
      actions.push("全角色: 设计评审通过后进入开发阶段");
    }
  }

  if (phase === "implementation") {
    const codingArtifacts = draftArtifacts.filter((a) => a.type === "code");
    if (codingArtifacts.length > 0) {
      actions.push(`Developer: 实现 ${codingArtifacts.length} 个模块代码`);
    }
    if (!allArtifactTypes.has("test_plan")) {
      actions.push("QA: 同步编写测试用例");
    }
  }

  if (phase === "testing") {
    if (draftTypes.has("test_plan")) {
      actions.push("QA: 执行测试，报告缺陷");
    }
    if (allArtifactTypes.has("defect_list")) {
      actions.push("Developer: 修复 P0/P1 缺陷");
    }
    if (inReviewTypes.has("test_report")) {
      actions.push("PM: 评估测试报告，决定是否上线");
    }
  }

  if (phase === "evaluation") {
    actions.push("PM: 主持项目回顾会议");
    actions.push("全角色: 输出评估意见和改进建议");
  }

  if (project.decisions.length > 0) {
    for (const d of project.decisions) {
      const decided = d.selected !== undefined ? ` → ${d.options[d.selected]}` : " (待决策)";
      actions.push(`决策 [${d.topic}]: ${d.options.join(" vs ")}${decided}`);
    }
  }

  return actions;
}

/**
 * Format team status as readable text.
 */
export function formatTeamStatus(projectId: string): string {
  const status = getTeamStatus(projectId);
  if (!status) return "Project not found.";

  const { project, phaseProgress, pendingArtifacts, blockers } = status;

  const lines = [
    `═══════════════════════════════════════`,
    `  ${project.name}`,
    `═══════════════════════════════════════`,
    `Status: ${project.status}`,
    `Phase: ${phaseProgress.current} (${phaseProgress.pct}% complete)`,
    `Created: ${new Date(project.createdAt).toLocaleString("zh-CN")}`,
    "",
    `Completed phases: ${phaseProgress.completed.length > 0 ? phaseProgress.completed.join(", ") || "(none)" : "(none)"}`,
    `Pending phases: ${phaseProgress.pending.join(", ") || "(none)"}`,
    "",
    `Artifacts: ${project.artifacts.length} total`,
  ];

  if (pendingArtifacts.length > 0) {
    lines.push(`  Pending: ${pendingArtifacts.map((a) => `${a.type}(${a.status})`).join(", ")}`);
  }

  if (blockers.length > 0) {
    lines.push(`  Blockers: ${blockers.join("; ")}`);
  }

  lines.push("", "Next Actions:");
  const nextActions = getNextActions(project);
  if (nextActions.length > 0) {
    lines.push(...nextActions.map((a) => `  • ${a}`));
  } else {
    lines.push("  (none)");
  }

  lines.push("", `Members: ${project.members.length}`);
  for (const m of project.members) {
    lines.push(`  ${m.role}: joined ${new Date(m.joinedAt).toLocaleDateString("zh-CN")}`);
  }

  return lines.join("\n");
}
