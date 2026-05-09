/**
 * Developer agent — architecture, coding, API design.
 */

import { Role, type RoleContext } from "../role.js";
import type { TeamRole } from "../types.js";

export class Developer extends Role {
  readonly name: TeamRole = "developer";
  readonly description = "负责架构设计、编码实现、API设计、集成文档。";

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
      summary: "Developer ready. Execute with specific development task.",
      nextActions: ["Design architecture", "Set up project structure", "Implement features"],
    };
  }

  generateTechSpec(input: {
    projectName: string;
    features: string[];
    techStack?: string[];
  }): string {
    const { projectName, features, techStack } = input;
    const stack = techStack ?? ["TypeScript", "Node.js", "PostgreSQL"];

    const sections = [
      "# 技术规格文档",
      "",
      `**项目**: ${projectName}`,
      `**版本**: v0.1`,
      `**日期**: ${new Date().toLocaleDateString("zh-CN")}`,
      "",
      "## 1. 技术栈",
      ...stack.map((t) => `- ${t}`),
      "",
      "## 2. 系统架构",
      "```",
      "[前端] → [API Gateway] → [服务层] → [数据层]",
      "```",
      "",
      "## 3. 项目结构",
      "```",
      "src/",
      "├── api/          # API路由",
      "├── services/     # 业务逻辑",
      "├── models/       # 数据模型",
      "├── utils/       # 工具函数",
      "└── index.ts     # 入口",
      "```",
      "",
      "## 4. API 设计",
      "### 4.1 RESTful 端点",
      "| 方法 | 路径 | 描述 | 请求体 | 响应 |",
      "|------|------|------|--------|------|",
      "| GET | /resources | 获取列表 | - | { data: [] } |",
      "| POST | /resources | 创建 | { ... } | { id, ... } |",
      "| GET | /resources/:id | 获取详情 | - | { data } |",
      "| PUT | /resources/:id | 更新 | { ... } | { data } |",
      "| DELETE | /resources/:id | 删除 | - | { success } |",
      "",
      "## 5. 数据模型",
      "### 5.1 Entity Name",
      "| 字段 | 类型 | 约束 | 说明 |",
      "|------|------|------|------|",
      "| id | UUID | PK | 主键 |",
      "| createdAt | timestamp | NOT NULL | 创建时间 |",
      "| updatedAt | timestamp | | 更新时间 |",
      "",
      "## 6. 功能模块",
      ...features.map((f, i) => `${i + 1}. ${f}`),
      "",
      "## 7. 质量标准",
      "- 单元测试覆盖率 ≥ 70%",
      "- API 文档覆盖率 100%",
      "- 依赖安全扫描通过",
      "- TypeScript 严格模式",
      "",
      "## 8. 部署架构",
      "- 环境: 开发/预发/生产",
      "- 容器化: Docker",
      "- CI/CD: GitHub Actions",
      "- 监控: 日志 + 指标",
    ];

    return sections.join("\n");
  }
}

export const developer = new Developer();
