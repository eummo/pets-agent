/**
 * ProductManager agent — requirements, PRD, user stories.
 */

import { Role, type RoleContext } from "../role.js";
import type { TeamRole } from "../types.js";

export class ProductManager extends Role {
  readonly name: TeamRole = "product";
  readonly description = "负责需求分析、PRD编写、用户故事、验收标准。";

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
      summary: "Product manager ready. Execute with specific product task.",
      nextActions: ["Define feature scope", "Write user stories", "Set acceptance criteria"],
    };
  }

  generatePRD(input: {
    productName: string;
    goal: string;
    targetUsers: string[];
    features: string[];
  }): string {
    const { productName, goal, targetUsers, features } = input;

    const sections = [
      `# 产品需求文档 (PRD)`,
      "",
      `**产品名称**: ${productName}`,
      `**版本**: v0.1`,
      `**日期**: ${new Date().toLocaleDateString("zh-CN")}`,
      "",
      "## 1. 概述与目标",
      "",
      "### 1.1 产品目标",
      goal,
      "",
      "### 1.2 成功标准",
      "- [ ] 指标1: \n- [ ] 指标2: \n- [ ] 指标3: ",
      "",
      "## 2. 用户与用例",
      "",
      "### 2.1 目标用户",
      ...targetUsers.map((u) => `- ${u}`),
      "",
      "### 2.2 用户画像",
      "| 用户类型 | 痛点 | 需求 | 场景 |",
      "|----------|------|------|------|",
      ...targetUsers.map((u) => `| ${u} | | | |`),
      "",
      "## 3. 功能需求",
      ...features.map((f, i) => `${i + 1}. ${f}`),
      "",
      "## 4. 非功能需求",
      "- 性能: 响应时间 < Xms",
      "- 可用性: 99.9% uptime",
      "- 安全: 符合GDPR/等保",
      "- 可扩展: 支持X并发",
      "",
      "## 5. 用户故事地图",
      this.generateStoryMap(targetUsers, features),
      "",
      "## 6. 验收标准",
      "| 功能 | 验收标准 | 测试方法 |",
      "|------|----------|----------|",
      ...features.map((f) => `| ${f} | | |`),
      "",
      "## 7. 排除项 (Out of Scope)",
      "- 排除的功能点1\n- 排除的功能点2",
      "",
      "## 8. 依赖与风险",
      "- 依赖: \n- 风险: ",
    ];

    return sections.join("\n");
  }

  private generateStoryMap(users: string[], features: string[]): string {
    const rows = [
      "| 阶段 |",
      "|-----------|",
      ...["发现", "评估", "上手", "使用", "留存", "推荐"].map((stage) => `| ${stage} |`),
    ];

    if (users.length === 0 || features.length === 0) {
      return rows.join("\n");
    }

    // Generate simple story grid
    const grid = [
      "",
      "| 用户 | 发现 | 评估 | 上手 | 使用 | 留存 | 推荐 |",
      "|------|------|------|------|------|------|------|",
      ...users.flatMap((user) => [
        `| ${user} | ${features.join(" | ")} |`,
      ]),
    ];

    return grid.join("\n");
  }

  generateUserStories(features: string[]): string {
    return features
      .map((f, i) => [
        `### ${i + 1}. ${f}`,
        "",
        "**为** [用户角色]",
        "",
        "**我想要** [功能描述]",
        "",
        "**以便** [业务价值]",
        "",
        "**验收标准**:",
        "- [ ] 标准1\n- [ ] 标准2\n- [ ] 标准3",
        "",
        "**优先级**: P0/P1/P2/P3",
        "",
        "**工作量**: X人天",
        "",
        "---",
      ].join("\n"))
      .join("\n");
  }
}

export const productManager = new ProductManager();
