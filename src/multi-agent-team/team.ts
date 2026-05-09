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
import { phaseController } from "./phase-controller.js";
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
  workdir: string
): Promise<{ ok: boolean; result?: Awaited<ReturnType<Role["run"]>> }> {
  const ctx: RoleContext = {
    projectId,
    phase,
    input,
    workdir,
  };

  const roleInstance = ROLE_CLASSES[role];
  if (!roleInstance) {
    return { ok: false };
  }

  try {
    const result = await roleInstance.run(ctx);
    return { ok: true, result };
  } catch (err) {
    console.error(`[Team] Role ${role} failed:`, err);
    return { ok: false };
  }
}

/**
 * Run all roles for a given phase in parallel.
 */
export async function runPhaseRoles(
  projectId: string,
  phase: ProjectPhase,
  input: unknown,
  workdir: string
): Promise<Array<{ role: TeamRole; ok: boolean; result?: Awaited<ReturnType<Role["run"]>> }>> {
  const roles = PHASE_ROLES[phase];
  const runs = roles.map((role) => runRole(projectId, role, phase, input, workdir));
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

  if (verdict === "approve") {
    // Check if all required reviewers have approved
    const phaseRoles = PHASE_ROLES[artifact.phase];
    const approvals = artifact.reviewers.filter((r) => r.verdict === "approve").length;
    if (approvals >= Math.ceil(phaseRoles.length / 2)) {
      artifact.status = "approved";
    }
  } else {
    artifact.status = "rejected";
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

  if (phase === "idea") {
    actions.push("Business analyst: 评估想法可行性和商业价值");
    actions.push("PM: 主持想法评审会议");
  }
  if (phase === "feasibility") {
    actions.push("Business analyst: 编写可行性报告");
    actions.push("Developer: 技术可行性评审");
    actions.push("PM: 组织评审，决定是否进入需求阶段");
  }
  if (phase === "requirements") {
    actions.push("Product manager: 编写PRD");
    actions.push("Designer: 参与评审用户体验");
    actions.push("PM: 冻结需求范围");
  }
  if (phase === "design") {
    actions.push("Designer: 完成UX设计");
    actions.push("Developer: 完成技术架构设计");
    actions.push("PM: 设计评审通过后进入开发");
  }
  if (phase === "implementation") {
    actions.push("Developer: 按模块实现代码");
    actions.push("QA: 同步编写测试用例");
  }
  if (phase === "testing") {
    actions.push("QA: 执行测试，报告缺陷");
    actions.push("Developer: 修复P0/P1缺陷");
    actions.push("PM: 评估测试报告，决定是否上线");
  }
  if (phase === "evaluation") {
    actions.push("PM: 主持回顾会议");
    actions.push("全角色: 输出评估意见");
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
