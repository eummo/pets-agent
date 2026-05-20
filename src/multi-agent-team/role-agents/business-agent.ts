/**
 * BusinessAnalyst agent — market analysis, feasibility, ROI, risk.
 * LLM-powered via BaseRoleLLM.
 */

import { BaseRoleLLM } from "../base-role-llm.js";
import { type RoleContext } from "../role.js";
import type { TeamRole, ProjectPhase } from "../types.js";
import { PHASE_LABELS } from "../types.js";

export class BusinessAnalyst extends BaseRoleLLM {
  readonly name: TeamRole = "business";
  readonly description = "负责市场分析、竞品研究、ROI评估、可行性判断。";

  protected role(): TeamRole {
    return "business";
  }

  protected override buildSystemPrompt(ctx: RoleContext): string {
    const phase = ctx.phase as ProjectPhase;
    return `你是资深商业分析师，负责市场分析、竞品研究、ROI评估、可行性判断。

当前阶段: ${PHASE_LABELS[phase]}
项目目录: ${ctx.workdir}

你的职责：
1. 分析目标市场和用户需求
2. 研究竞争对手和市场份额
3. 评估商业模式和变现路径
4. 进行风险评估和 ROI 测算
5. 输出可行性报告

**输出质量标准：**
- 市场分析需有数据支撑（市场规模、增长率、用户基数）
- 竞品分析需包含市场份额、功能对比、定价策略
- ROI 测算需包含固定成本、变动成本、收入预测、盈亏平衡点
- 风险评估需分维度（技术/市场/财务/运营）并给出概率评级

**示例输出片段：**

\`\`\`
## 3. 商业模式

### 3.1 变现路径
SaaS 订阅制，按月/年收费

### 3.2 单位经济模型
| 指标 | 数值 |
|------|------|
| 单用户 ARPU | ¥99/月 |
| 获客成本 CAC | ¥150 |
| 用户生命周期 LTV | 18 个月 |
| LTV/CAC 比 | 11.88 ✓ |
| 盈亏平衡客户数 | 1,667 |

## 4. 风险评估

| 风险类型 | 概率 | 影响 | 应对策略 |
|----------|------|------|----------|
| 技术：数据泄露 | 中 | 高 | ISO27001 合规 + 加密存储 |
| 市场：竞争加剧 | 高 | 中 | 差异化功能 + 用户粘性 |
| 财务：现金流断裂 | 低 | 高 | 提前 6 个月融资 |
\`\`\`
\`\`\`

输出格式（Markdown）：
\`\`\`
# 可行性分析报告

## 1. 市场机会
[市场规模/用户需求/增长趋势]

## 2. 竞品分析
[主要竞品/差异化/竞争优势]

## 3. 商业模式
[变现路径/收入模型/单位经济]

## 4. 风险评估
| 风险类型 | 概率 | 影响 | 应对策略 |

## 5. ROI 估算
[成本/收益/盈亏平衡/投资回报周期]

## 6. 可行性评分
| 维度 | 评分(1-10) | 说明 |

## 7. 结论
[通过/有条件通过/不通过 + 建议]
\`\`\``;
  }

  protected buildUserPrompt(ctx: RoleContext): string {
    const input = (ctx.input ?? {}) as {
      idea?: string;
      targetMarket?: string;
      competitors?: string[];
      estimatedBudget?: string;
    };

    return `## 项目信息
- 项目想法: ${input.idea ?? "待定义"}
- 目标市场: ${input.targetMarket ?? "待定义"}
- 主要竞品: ${(input.competitors ?? []).join("、") || "待识别"}
- 预算规模: ${input.estimatedBudget ?? "待评估"}
- 工作目录: ${ctx.workdir}

请生成完整可行性分析报告。`;
  }

  protected async _execute(
    ctx: RoleContext,
    _prepData: unknown
  ): Promise<{
    status: "ok" | "blocked";
    reason?: string;
    data?: unknown;
    summary?: string;
    nextActions?: string[];
  }> {
    const userPrompt = this.buildUserPrompt(ctx);

    try {
      const content = await this.callLLM(userPrompt, ctx);

      if (content === "[cancelled]") {
        return { status: "blocked", reason: "LLM 调用被取消", data: {} };
      }

      const lineCount = content.split("\n").length;

      return {
        status: "ok",
        data: { feasibilityReport: content },
        summary: `生成 ${lineCount} 行可行性分析报告`,
        nextActions: [
          "商业评审会议",
          "确认目标市场和定位",
          "细化 ROI 测算",
        ],
      };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      return {
        status: "blocked",
        reason: `LLM 调用失败: ${msg}`,
        data: { error: msg },
      };
    }
  }

  // Fallback template method
  generateFeasibilityReport(input: {
    idea: string;
    targetMarket?: string;
    competitors?: string[];
  }): string {
    const { idea, targetMarket, competitors } = input;

    return `# 可行性分析报告

**项目想法**: ${idea}
**日期**: ${new Date().toLocaleDateString("zh-CN")}

## 1. 市场机会
- 目标市场: ${targetMarket ?? "待定义"}
- 市场规模: 待评估

## 2. 竞品分析
${(competitors ?? []).map((c, i) => `${i + 1}. ${c}`).join("\n") || "- 竞品: 待识别"}

## 3. 可行性评分
| 维度 | 评分(1-10) | 说明 |
|------|-----------|------|
| 技术可行性 | /10 | |
| 市场可行性 | /10 | |
| 财务可行性 | /10 | |
| 团队可行性 | /10 | |

> 注意：此为 fallback 模板。优先使用 LLM 生成的完整可行性报告。`;
  }
}

export const businessAnalyst = new BusinessAnalyst();
