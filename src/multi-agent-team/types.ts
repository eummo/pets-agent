/**
 * Multi-Agent Team Types — shared types for the project team system.
 */

// ============================================================================
// Roles
// ============================================================================

export type TeamRole =
  | "pm"           // Project Manager —统筹、决策、进度
  | "product"      // Product Manager —需求、PRD、用户故事
  | "designer"     // Designer — UX、信息架构、视觉
  | "developer"    // Developer —架构、编码、集成
  | "qa"           // QATester —测试策略、用例、缺陷
  | "business";    // Business Analyst —市场、竞品、ROI

export const ROLE_LABELS: Record<TeamRole, string> = {
  pm: "Project Manager",
  product: "Product Manager",
  designer: "Designer",
  developer: "Developer",
  qa: "QA Tester",
  business: "Business Analyst",
};

export const ROLE_DESCRIPTIONS: Record<TeamRole, string> = {
  pm: "统筹项目全局，负责任务分解、阶段推进、决策和风险管理。",
  product: "负责需求分析、PRD编写、用户故事、验收标准。",
  designer: "负责UX研究、交互设计、信息架构、视觉规范。",
  developer: "负责架构设计、编码实现、API设计、集成文档。",
  qa: "负责测试策略、用例设计、缺陷管理、回归测试。",
  business: "负责市场分析、竞品研究、ROI评估、可行性判断。",
};

// ============================================================================
// Phases
// ============================================================================

export type ProjectPhase =
  | "idea"         // 想法录入
  | "feasibility"  // 可行性分析
  | "requirements" // 需求定义
  | "design"       // 设计
  | "implementation" // 实现
  | "testing"       // 测试
  | "evaluation";  // 评估交付

export const PHASE_LABELS: Record<ProjectPhase, string> = {
  idea: "想法录入",
  feasibility: "可行性分析",
  requirements: "需求定义",
  design: "设计",
  implementation: "实现",
  testing: "测试",
  evaluation: "评估交付",
};

export const PHASE_ROLES: Record<ProjectPhase, TeamRole[]> = {
  idea: ["business", "pm"],
  feasibility: ["business", "developer"],
  requirements: ["product", "designer", "business"],
  design: ["designer", "developer", "product"],
  implementation: ["developer", "qa"],
  testing: ["qa", "developer"],
  evaluation: ["pm", "product", "designer", "developer", "qa", "business"],
};

// ============================================================================
// Artifacts
// ============================================================================

export type ArtifactType =
  | "idea_form"       // 想法登记表
  | "feasibility_report" // 可行性报告
  | "prd"             // 产品需求文档
  | "user_story_map"  // 用户故事地图
  | "design_spec"     // 设计规范
  | "tech_spec"       // 技术规格文档
  | "code"            // 源代码
  | "test_plan"       // 测试计划
  | "test_report"     // 测试报告
  | "defect_list"    // 缺陷列表
  | "assessment"      // 评估报告
  | "meeting_notes"   // 会议记录
  | "decision_record" // 决策记录
  | "retrospective";  // 回顾记录

export interface Artifact {
  id: string;
  projectId: string;
  type: ArtifactType;
  title: string;
  content: string;        // 主要内容（markdown）
  phase: ProjectPhase;
  createdBy: TeamRole;
  createdAt: string;
  version: number;
  status: "draft" | "in_review" | "approved" | "rejected" | "superseded";
  reviewers: Array<{ role: TeamRole; verdict: "approve" | "reject" | "abstain"; comment?: string }>;
  summary?: string;       // 简短摘要（用于展示）
}

// ============================================================================
// Projects
// ============================================================================

export interface ProjectMember {
  role: TeamRole;
  agentId: string;        // pets-agent task id
  joinedAt: string;
  contributions: string[]; // artifact ids
}

export type ProjectStatus =
  | "planning"   // 规划中
  | "active"     // 进行中
  | "blocked"    // 阻塞
  | "paused"     // 暂停
  | "completed"  // 完成
  | "cancelled"; // 取消

export interface Project {
  id: string;
  name: string;
  description: string;
  phase: ProjectPhase;
  status: ProjectStatus;
  createdAt: string;
  updatedAt: string;
  members: ProjectMember[];
  artifacts: Artifact[];
  decisions: Decision[];
  currentBlocker?: string;
  target?: string;         // 目标描述
  successCriteria?: string; // 成功标准
}

// ============================================================================
// Decisions
// ============================================================================

export interface Decision {
  id: string;
  projectId: string;
  topic: string;
  options: string[];
  selected: number;        // index of selected option
  rationale: string;
  madeBy: TeamRole;
  madeAt: string;
  phase: ProjectPhase;
}

// ============================================================================
// Meetings
// ============================================================================

export interface Meeting {
  id: string;
  projectId: string;
  topic: string;
  participants: TeamRole[];
  notes: string;
  outcomes: string[];
  createdAt: string;
}

// ============================================================================
// Phase Reviews
// ============================================================================

export interface PhaseReview {
  phase: ProjectPhase;
  deliverables: Artifact[];   // 预期产出
  actualDeliverables: Artifact[]; // 实际产出
  blockers: string[];
  verdict: "pass" | "fail" | "needs_revision";
  comments: string;
  reviewedBy: TeamRole;
  reviewedAt: string;
}

// ============================================================================
// Team Status
// ============================================================================

export interface TeamStatus {
  project: Project;
  phaseProgress: {
    current: ProjectPhase;
    completed: ProjectPhase[];
    pending: ProjectPhase[];
  };
  pendingArtifacts: Artifact[];
  openDecisions: Decision[];
  blockers: string[];
  nextActions: string[];
}
