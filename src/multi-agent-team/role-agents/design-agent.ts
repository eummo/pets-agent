/**
 * Designer agent — UX research, interaction design, visual specs.
 */

import { Role, type RoleContext } from "../role.js";
import type { TeamRole } from "../types.js";

export class Designer extends Role {
  readonly name: TeamRole = "designer";
  readonly description = "负责UX研究、交互设计、信息架构、视觉规范。";

  protected async _execute(_ctx: RoleContext, _prepData: unknown): Promise<{
    status: "ok" | "blocked";
    reason?: string;
    data?: unknown;
    summary?: string;
    nextActions?: string[];
  }> {
    return {
      status: "ok",
      data: {},
      summary: "Designer ready. Execute with specific design task.",
      nextActions: ["Define information architecture", "Create wireframes", "Define component library"],
    };
  }

  generateDesignSpec(input: {
    productName: string;
    pages: string[];
    userFlows: string[];
  }): string {
    const { productName, pages, userFlows } = input;

    const sections = [
      "# 设计规范",
      "",
      `**产品**: ${productName}`,
      `**版本**: v1.0`,
      `**日期**: ${new Date().toLocaleDateString("zh-CN")}`,
      "",
      "## 1. 设计原则",
      "- 清晰: 界面信息层次分明",
      "- 一致: 交互模式统一",
      "- 高效: 减少用户操作步骤",
      "- 可及: 考虑无障碍访问",
      "",
      "## 2. 信息架构",
      ...pages.map((p) => `- ${p}`),
      "",
      "## 3. 页面结构",
      ...pages.map((p) => [
        `### 3.x ${p}`,
        "",
        "**布局**:",
        "- 顶部导航: ",
        "- 主内容区: ",
        "- 底部/侧边: ",
        "",
        "**组件**:",
        "- 组件1",
        "- 组件2",
        "",
      ].join("\n")),
      "",
      "## 4. 用户流程",
      ...userFlows.map((flow, i) => `${i + 1}. ${flow}`),
      "",
      "## 5. 组件规范",
      "| 组件 | 状态 | 样式 | 交互 |",
      "|------|------|------|-------|",
      "| Button | default/hover/active/disabled | | |",
      "| Input | default/focus/error/disabled | | |",
      "| Card | | | |",
      "",
      "## 6. 视觉规范",
      "- 色彩: 主色 #XXX, 辅色 #XXX, 强调色 #XXX",
      "- 字体: 主字体, 备选字体",
      "- 间距: 基准间距 4px/8px/16px/24px/32px",
      "- 圆角: 4px/8px/12px",
      "- 阴影: ",
      "",
      "## 7. 响应式策略",
      "- Desktop: ≥ 1200px",
      "- Tablet: 768px - 1199px",
      "- Mobile: < 768px",
    ];

    return sections.join("\n");
  }
}

export const designer = new Designer();
