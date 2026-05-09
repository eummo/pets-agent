/**
 * BusinessAnalyst agent — market analysis, feasibility, ROI, risk.
 */

import { Role, type RoleContext, type RoleResult } from "../role.js";
import type { TeamRole } from "../types.js";

export class BusinessAnalyst extends Role {
  readonly name: TeamRole = "business";
  readonly description = "负责市场分析、竞品研究、ROI评估、可行性判断。";

  protected async _execute(_ctx: RoleContext, _prepData: unknown): Promise<{
    status: "ok" | "blocked";
    reason?: string;
    data?: unknown;
    summary?: string;
    nextActions?: string[];
  }> {
    // Business analyst work is triggered with specific phase input
    // This is a template — actual analysis prompted per task
    return {
      status: "ok",
      data: {},
      summary: "Business analyst ready. Execute with specific analysis task.",
      nextActions: ["Define market research scope", "Identify competitors", "Assess ROI"],
    };
  }

  generateFeasibilityReport(input: {
    idea: string;
    targetMarket?: string;
    competitors?: string[];
  }): string {
    const { idea, targetMarket, competitors } = input;

    const sections = [
      "# 可行性分析报告",
      "",
      `**项目想法**: ${idea}`,
      "",
      "## 1. 市场机会",
      targetMarket
        ? `- 目标市场: ${targetMarket}\n- 市场规模: 待评估\n- 增长趋势: 待研究`
        : "- 目标市场: 待定义",
      "",
      "## 2. 竞品分析",
      competitors && competitors.length > 0
        ? competitors.map((c, i) => `${i + 1}. ${c}`).join("\n")
        : "- 竞品: 待识别",
      "",
      "## 3. 商业模式",
      "- 变现路径: 待设计\n- 收入模型: 待确定\n- 单位经济模型: 待构建",
      "",
      "## 4. 风险评估",
      "| 风险类型 | 概率 | 影响 | 应对 |",
      "|----------|------|------|------|",
      "| 技术风险 | 低/中/高 | 低/中/高 | |",
      "| 市场风险 | 低/中/高 | 低/中/高 | |",
      "| 竞争风险 | 低/中/高 | 低/中/高 | |",
      "| 监管风险 | 低/中/高 | 低/中/高 | |",
      "",
      "## 5. ROI 估算",
      "- 开发成本: 待估算\n- 运营成本: 待估算\n- 预期回报: 待计算\n- 盈亏平衡点: 待确定",
      "",
      "## 6. 可行性评分",
      "| 维度 | 评分(1-10) | 说明 |",
      "|------|-----------|------|",
      "| 技术可行性 | /10 | |",
      "| 市场可行性 | /10 | |",
      "| 财务可行性 | /10 | |",
      "| 团队可行性 | /10 | |",
      "",
      "**综合评分**: /40 → 阈值 ≥ 24/40 (60%) 可通过",
      "",
      "## 7. 结论与建议",
      "- 结论: [通过/有条件通过/不通过]\n- 主要优势: \n- 主要风险点: \n- 下一步行动: ",
    ];

    return sections.join("\n");
  }
}

export const businessAnalyst = new BusinessAnalyst();
