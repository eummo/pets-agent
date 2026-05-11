/**
 * Designer agent — UX research, interaction design, visual specs.
 * LLM-powered via BaseRoleLLM.
 */

import { BaseRoleLLM } from "../base-role-llm.js";
import { type RoleContext } from "../role.js";
import type { TeamRole, ProjectPhase } from "../types.js";
import { PHASE_LABELS } from "../types.js";

export class Designer extends BaseRoleLLM {
  readonly name: TeamRole = "designer";
  readonly description = "负责UX研究、交互设计、信息架构、视觉规范。";

  protected role(): TeamRole {
    return "designer";
  }

  protected override buildSystemPrompt(ctx: RoleContext): string {
    const phase = ctx.phase as ProjectPhase;
    return `你是资深 UX 设计师，负责用户体验设计、信息架构、交互设计、视觉规范制定。

当前阶段: ${PHASE_LABELS[phase]}
项目目录: ${ctx.workdir}

你的职责：
1. 研究用户行为和需求
2. 制定信息架构和页面结构
3. 设计用户流程和交互模式
4. 制定组件库和视觉规范
5. 输出设计规范文档

**输出质量标准：**
- 每个页面需标注组件状态（default / hover / active / disabled / error）
- 交互流程需包含异常路径和出错状态
- 视觉规范需包含间距系统（4px 基准网格）、色彩语义化命名
- 响应式断点需明确说明各端适配策略

**示例输出片段：**

\`\`\`
## 4. 组件规范

### Button 按钮

| 状态 | 背景色 | 文字色 | 边框 | 圆角 |
|------|--------|--------|------|------|
| default | #2563EB | #FFFFFF | none | 6px |
| hover | #1D4ED8 | #FFFFFF | none | 6px |
| active | #1E40AF | #FFFFFF | none | 6px |
| disabled | #D1D5DB | #9CA3AF | none | 6px |
| error | #DC2626 | #FFFFFF | none | 6px |

### 间距系统
- 基准单位：4px
- 间距档位：4 / 8 / 12 / 16 / 24 / 32 / 48 / 64px
- 页面边距：移动端 16px，平板 24px，桌面 32px

### 色彩语义
| token | 用途 |
|-------|------|
| primary-500 | 主按钮、链接 |
| success-500 | 成功状态 |
| warning-500 | 警告状态 |
| error-500 | 错误、危险操作 |
\`\`\`
\`\`\`

输出格式（Markdown）：
\`\`\`
# 设计规范

## 1. 设计原则
[清晰/一致/高效/可及]

## 2. 信息架构
[页面结构/导航体系]

## 3. 页面结构
[每个页面的布局/组件]

## 4. 用户流程
[主要用户旅程]

## 5. 组件规范
| 组件 | 状态 | 样式 | 交互 |

## 6. 视觉规范
[色彩/字体/间距/圆角/阴影]

## 7. 响应式策略
\`\`\``;
  }

  protected buildUserPrompt(ctx: RoleContext): string {
    const input = (ctx.input ?? {}) as {
      productName?: string;
      pages?: string[];
      userFlows?: string[];
    };

    return `## 设计信息
- 产品名称: ${input.productName ?? "待定义"}
- 页面列表: ${(input.pages ?? []).join("、") || "待定义"}
- 用户流程: ${(input.userFlows ?? []).join(" → ") || "待定义"}
- 工作目录: ${ctx.workdir}

请生成完整设计规范文档。`;
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
        data: { designSpec: content },
        summary: `生成 ${lineCount} 行设计规范`,
        nextActions: [
          "设计评审会议",
          "输出高保真原型",
          "建立设计组件库",
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
  generateDesignSpec(input: {
    productName: string;
    pages: string[];
    userFlows: string[];
  }): string {
    const { productName, pages, userFlows } = input;

    return `# 设计规范

**产品**: ${productName}
**版本**: v1.0
**日期**: ${new Date().toLocaleDateString("zh-CN")}

## 1. 设计原则
- 清晰: 界面信息层次分明
- 一致: 交互模式统一
- 高效: 减少用户操作步骤
- 可及: 考虑无障碍访问

## 2. 信息架构
${(pages ?? []).map((p) => `- ${p}`).join("\n")}

## 3. 用户流程
${(userFlows ?? []).map((f, i) => `${i + 1}. ${f}`).join("\n")}

## 4. 组件规范
| 组件 | 状态 | 样式 |
|------|------|------|
| Button | default/hover/active | |
| Input | default/focus/error | |

> 注意：此为 fallback 模板。优先使用 LLM 生成完整设计规范。`;
  }
}

export const designer = new Designer();
