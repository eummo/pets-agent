/**
 * Developer agent — architecture, coding, API design.
 * LLM-powered via BaseRoleLLM.
 */

import { BaseRoleLLM } from "../base-role-llm.js";
import { type RoleContext } from "../role.js";
import type { TeamRole, ProjectPhase } from "../types.js";
import { PHASE_LABELS, ROLE_DESCRIPTIONS } from "../types.js";

export class Developer extends BaseRoleLLM {
  readonly name: TeamRole = "developer";
  readonly description = "负责架构设计、编码实现、API设计、集成文档。";

  protected role(): TeamRole {
    return "developer";
  }

  protected override buildSystemPrompt(ctx: RoleContext): string {
    const phase = ctx.phase as ProjectPhase;
    return `你是资深全栈软件架构师，负责技术规格文档编写、架构设计、代码实现。

当前阶段: ${PHASE_LABELS[phase]}
项目目录: ${ctx.workdir}

你的职责：
1. 分析需求，评估技术可行性
2. 设计系统架构（前端/后端/数据层）
3. 编写技术规格文档（Tech Spec）
4. 提供代码实现指导

**输出质量标准：**
- 每个技术选型必须给出至少 2 个备选方案及选型理由
- API 设计需包含完整 Endpoint + Method + Request/Response Schema
- 数据模型需包含字段类型、约束、索引
- 风险点需包含概率评估（高/中/低）和缓解方案

**示例输出片段：**

\`\`\`
## 2. 技术栈选型

| 用途 | 推荐方案 | 备选方案 | 选型理由 |
|------|---------|---------|---------|
| 前端框架 | React 18 | Vue 3 | 生态丰富，TypeScript 支持完善 |
| 后端框架 | FastAPI | NestJS | 异步性能好，Swagger 自动生成 |

## 4. API 设计

\`\`\`
POST /api/v1/users
Request:  { "name": string, "email": string, "role": "admin"|"user" }
Response: { "id": string, "name": string, "createdAt": ISO8601 }
Errors:   400 Invalid input | 409 Email already exists
\`\`\`
\`\`\`

输出格式（Markdown）：
\`\`\`
# 技术规格文档

## 1. 技术栈选型
[表格：用途 | 推荐方案 | 备选方案 | 选型理由]

## 2. 系统架构
[架构图/模块划分]

## 3. 项目结构
[目录结构说明]

## 4. API 设计
[RESTful API 端点]

## 5. 数据模型
[核心实体/数据库表结构]

## 6. 部署方案
[环境/容器/CI-CD]

## 7. 风险点
[技术风险及缓解措施]
\`\`\`

要求：
- 语言：中文
- 技术选型需给出理由
- 架构图用文字形式描述`;
  }

  protected buildUserPrompt(ctx: RoleContext): string {
    const input = (ctx.input ?? {}) as {
      projectName?: string;
      features?: string[];
      techStack?: string[];
      constraints?: string;
    };

    const projectName = input.projectName ?? "未命名项目";
    const features = input.features ?? [];
    const techStack = input.techStack ?? [];
    const constraints = input.constraints ?? "无特殊约束";

    return `## 项目信息
- 项目名称: ${projectName}
- 核心功能: ${features.length > 0 ? features.join("、") : "待定义"}
- 已有技术栈: ${techStack.length > 0 ? techStack.join("、") : "未指定"}
- 约束条件: ${constraints}
- 工作目录: ${ctx.workdir}

请生成完整技术规格文档。`;
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
        data: { techSpec: content },
        summary: `生成 ${lineCount} 行技术规格文档`,
        nextActions: [
          "技术方案评审会议",
          "确认技术栈选型",
          "拆解开发任务",
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
  // Fallback template method (used by generate_doc tool as fallback)
  generateTechSpec(input: {
    projectName: string;
    features: string[];
    techStack?: string[];
  }): string {
    const { projectName, features, techStack } = input;
    const stack = techStack ?? ["待定"];

    return `# 技术规格文档

**项目**: ${projectName}
**版本**: v0.1
**日期**: ${new Date().toLocaleDateString("zh-CN")}

## 1. 技术栈

${stack.map((t) => `- ${t}`).join("\n")}

## 2. 功能列表

${features.map((f, i) => `${i + 1}. ${f}`).join("\n")}

> 注意：此为 fallback 模板。优先使用 LLM 生成的完整技术规格。`;
  }
}

export const developer = new Developer();
