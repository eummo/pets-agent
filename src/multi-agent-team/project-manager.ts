/**
 * ProjectManager — the orchestrator role.
 *
 * Responsibilities:
 * - Create and initialize projects
 * - Decompose work into phase plans
 * - Monitor progress and track blockers
 * - Make or escalate decisions
 * - Advance phases when gates are cleared
 */

import { Role, type RoleContext } from "./role.js";
import type { TeamRole, Project, ProjectPhase, Artifact } from "./types.js";
import { PHASE_LABELS, PHASE_ROLES } from "./types.js";
import { phaseController, PHASE_ORDER } from "./phase-controller.js";
import { meetingManager } from "./meeting.js";
import { projectStore } from "./project-store.js";

export interface PlanPhaseInput {
  projectId: string;
  phase: ProjectPhase;
}

export interface ReviewDeliverableInput {
  artifactId: string;
  verdict: "approve" | "reject";
  comment?: string;
  reviewedBy: TeamRole;
}

export interface DecideInput {
  projectId: string;
  topic: string;
  options: string[];
  rationale: string;
  selected: number;
  madeBy: TeamRole;
}

export class ProjectManager extends Role {
  readonly name: TeamRole = "pm";
  readonly description = "统筹项目全局，负责任务分解、阶段推进、决策和风险管理。";

  protected async _prepare(ctx: RoleContext): Promise<{ ok: true; data: Project } | { ok: false; reason: string }> {
    const project = projectStore.get(ctx.projectId);
    if (!project) {
      return { ok: false, reason: `Project not found: ${ctx.projectId}` };
    }
    return { ok: true, data: project };
  }

  protected async _execute(ctx: RoleContext, prepData: unknown): Promise<{
    status: "ok" | "blocked";
    reason?: string;
    data?: { project: Project; nextActions: string[]; blockers: string[] };
    summary?: string;
  }> {
    const project = prepData as Project;
    const blockers: string[] = [];
    const nextActions: string[] = [];

    // Check current phase gate
    const gateCheck = phaseController.canAdvance(project);
    if (!gateCheck.ok) {
      blockers.push(gateCheck.reason ?? "Gate check failed");
    }

    // Determine next actions based on phase
    const phaseRoles = PHASE_ROLES[project.phase];
    nextActions.push(
      `当前阶段: ${PHASE_LABELS[project.phase]} (${project.phase})`,
      `参与角色: ${phaseRoles.join(", ")}`,
      `Gate检查: ${gateCheck.ok ? "通过" : "未通过 — " + (gateCheck.reason ?? "")}`
    );

    if (!gateCheck.ok && gateCheck.missingArtifacts) {
      nextActions.push(`缺失产物: ${gateCheck.missingArtifacts.join(", ")}`);
    }

    if (project.currentBlocker) {
      blockers.push(`阻塞: ${project.currentBlocker}`);
    }

    const summary = [
      `Phase: ${project.phase} | Status: ${project.status}`,
      blockers.length > 0 ? `Blockers: ${blockers.join("; ")}` : "No blockers",
      gateCheck.ok ? "✓ Ready to advance" : `✗ Cannot advance: ${gateCheck.reason ?? ""}`,
    ].join(" | ");

    return {
      status: blockers.length > 0 ? "blocked" : "ok",
      reason: blockers.length > 0 ? blockers[0] : undefined,
      data: { project, nextActions, blockers },
      summary,
    };
  }

  // PM-specific methods (called by tools, not through run())

  planPhase(projectId: string, phase: ProjectPhase): { ok: boolean; plan: string } {
    const project = projectStore.get(projectId);
    if (!project) return { ok: false, plan: "" };

    const phaseRoles = PHASE_ROLES[phase];
    const prevPhase = phase !== "idea" ? PHASE_ORDER[PHASE_ORDER.indexOf(phase) - 1] : null;

    const lines = [
      `# ${PHASE_LABELS[phase]} 阶段计划`,
      "",
      `**项目**: ${project.name}`,
      `**阶段**: ${phase}`,
      `**参与角色**: ${phaseRoles.join(", ")}`,
      "",
      "## 阶段目标",
      this.phaseGoals(phase),
      "",
      "## 参与者职责",
      ...phaseRoles.map((r) => `- ${r}: ${this.roleResponsibilities(r, phase)}`),
      "",
      "## 输入物",
      prevPhase
        ? `上一阶段产物需评审通过后，方可开始本阶段。`
        : "无（首个阶段）",
      "",
      "## 产出物",
      this.phaseDeliverables(phase),
      "",
      "## 评审标准",
      this.phaseReviewCriteria(phase),
      "",
      "## 风险点",
      ...this.phaseRisks(phase),
    ];

    return { ok: true, plan: lines.join("\n") };
  }

  private phaseGoals(phase: ProjectPhase): string {
    const goals: Record<ProjectPhase, string> = {
      idea: "明确项目想法、目标、初步范围，建立项目基本信息。",
      feasibility: "评估技术、市场、财务可行性，识别关键风险。",
      requirements: "定义完整的产品需求，建立用户故事地图。",
      design: "完成技术架构设计和用户体验设计。",
      implementation: "实现所有功能代码，保持代码质量。",
      testing: "完成功能测试、集成测试，修复所有P0/P1缺陷。",
      evaluation: "评估项目完成度，记录经验教训。",
    };
    return goals[phase];
  }

  private roleResponsibilities(role: TeamRole, phase: ProjectPhase): string {
    const responsibilities: Record<string, Record<ProjectPhase, string>> = {
      pm: {
        idea: "主持想法评审会议，记录决策",
        feasibility: "跟踪分析进度，组织技术评审",
        requirements: "主持PRD评审，冻结需求范围",
        design: "跟踪设计进度，组织设计评审",
        implementation: "跟踪开发进度，管理依赖",
        testing: "跟踪测试进度，评审测试报告",
        evaluation: "主持回顾会议，输出评估报告",
      },
      product: {
        idea: "撰写想法描述，确定目标用户",
        feasibility: "评估市场需求和商业价值",
        requirements: "编写PRD和用户故事",
        design: "评审UX设计，定义验收标准",
        implementation: "评审功能实现是否符合PRD",
        testing: "验证功能是否符合用户故事",
        evaluation: "评估产品是否达到目标",
      },
      designer: {
        idea: "评估用户体验初步方向",
        feasibility: "评估UX可行性",
        requirements: "参与用户故事评审",
        design: "负责全部设计工作",
        implementation: "提供设计支持和标注",
        testing: "评审测试用例的UX覆盖",
        evaluation: "评估用户体验完成度",
      },
      developer: {
        idea: "评估技术可行性和工作量",
        feasibility: "编写技术可行性分析",
        requirements: "评审技术依赖和约束",
        design: "负责技术架构设计",
        implementation: "负责全部编码工作",
        testing: "修复测试发现的缺陷",
        evaluation: "评估技术债务和可维护性",
      },
      qa: {
        idea: "评估测试风险和策略",
        feasibility: "参与风险评估",
        requirements: "评审验收标准的完整性",
        design: "评审测试覆盖策略",
        implementation: "编写测试计划和用例",
        testing: "执行测试，报告缺陷",
        evaluation: "评估测试质量和覆盖率",
      },
      business: {
        idea: "评估商业价值和战略意义",
        feasibility: "编写市场分析和ROI",
        requirements: "评审商业目标一致性",
        design: "评审商业模式和变现路径",
        implementation: "跟踪商业指标",
        testing: "验证商业指标可测量性",
        evaluation: "评估商业目标达成度",
      },
    };
    return responsibilities[role]?.[phase] ?? "";
  }

  private phaseDeliverables(phase: ProjectPhase): string {
    const deliverables: Record<ProjectPhase, string[]> = {
      idea: ["想法登记表 (idea_form)"],
      feasibility: ["可行性报告 (feasibility_report)"],
      requirements: ["PRD (prd)", "用户故事地图 (user_story_map)"],
      design: ["设计规范 (design_spec)", "技术规格 (tech_spec)"],
      implementation: ["源代码 (code)", "API文档"],
      testing: ["测试计划 (test_plan)", "测试报告 (test_report)", "缺陷列表 (defect_list)"],
      evaluation: ["评估报告 (assessment)", "回顾记录 (retrospective)"],
    };
    return deliverables[phase].map((d) => `- ${d}`).join("\n");
  }

  private phaseReviewCriteria(phase: ProjectPhase): string {
    const criteria: Record<ProjectPhase, string> = {
      idea: "想法清晰、目标明确、有潜在价值",
      feasibility: "可行性评分 ≥ 6/10，无不可克服风险",
      requirements: "所有用户故事有验收标准，范围已冻结",
      design: "技术方案和UX设计均已评审通过",
      implementation: "代码通过CI，单元测试覆盖率 ≥ 70%",
      testing: "所有P0/P1缺陷已修复，测试覆盖率 ≥ 80%",
      evaluation: "所有评估维度有量化数据，有改进建议",
    };
    return criteria[phase];
  }

  private phaseRisks(phase: ProjectPhase): string[] {
    const risks: Record<ProjectPhase, string[]> = {
      idea: ["想法过于模糊，无法评估", "与现有项目范围重叠"],
      feasibility: ["技术方案存在重大不确定性", "市场规模评估过于乐观"],
      requirements: ["需求范围蔓延", "验收标准不明确"],
      design: ["设计与实现脱节", "技术方案影响用户体验"],
      implementation: ["依赖外部服务不稳定", "代码质量债务累积"],
      testing: ["缺陷修复引入新缺陷", "测试环境与生产环境差异"],
      evaluation: ["缺乏量化数据支撑结论", "回顾流于形式"],
    };
    return risks[phase].map((r) => `- 风险: ${r}`);
  }
}

export const projectManager = new ProjectManager();
