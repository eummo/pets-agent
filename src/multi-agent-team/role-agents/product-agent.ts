/**
 * ProductManager agent — requirements, PRD, user stories.
 * LLM-powered via BaseRoleLLM.
 */

import { BaseRoleLLM } from "../base-role-llm.js";
import { type RoleContext } from "../role.js";
import type { TeamRole, ProjectPhase } from "../types.js";
import { PHASE_LABELS } from "../types.js";

export class ProductManager extends BaseRoleLLM {
  readonly name: TeamRole = "product";
  readonly description = "负责需求分析、PRD编写、用户故事、验收标准。";

  protected role(): TeamRole {
    return "product";
  }

  protected override buildSystemPrompt(ctx: RoleContext): string {
    const phase = ctx.phase as ProjectPhase;
    return `你是资深产品经理，负责产品需求文档编写、用户故事分析、验收标准制定。

当前阶段: ${PHASE_LABELS[phase]}
项目目录: ${ctx.workdir}

你的职责：
1. 理解用户需求和业务目标
2. 编写结构化 PRD（产品需求文档）
3. 制定用户故事地图
4. 定义验收标准和成功指标

**输出质量标准：**
- 每个功能必须有独立验收标准（可测试、可量化）
- 成功指标必须是数字化的 KPI，而非模糊描述
- 用户画像需包含场景、痛点、需求的完整链路
- 排除项必须明确标注，防止范围蔓延

**示例输出片段：**

\`\`\`
### 2.2 用户画像

| 角色 | 场景 | 痛点 | 需求 |
|------|------|------|------|
| 上班族 | 通勤时浏览 | 内容加载慢 | 离线缓存 |
| 学生 | 课后学习 | 找不到内容 | 智能推荐 |

### 3. 功能需求

#### FR-001: 离线缓存

**描述**：用户可将内容下载到本地，无网时阅读

**验收标准**：
- [ ] 下载完成后，断网状态下可完整浏览已缓存内容
- [ ] 缓存命中率 ≥ 80%（7日滚动计算）
- [ ] 单次下载最大超时 30s，超时自动重试 3 次
\`\`\`
\`\`\`

输出格式（Markdown）：
\`\`\`
# 产品需求文档

## 1. 概述与目标
### 1.1 产品目标
[清晰描述产品要解决的问题]

### 1.2 成功标准
- [ ] 量化指标1
- [ ] 量化指标2

## 2. 用户分析
### 2.1 目标用户
[用户群体描述]

### 2.2 用户画像
| 用户类型 | 痛点 | 需求 | 场景 |

## 3. 功能需求
[功能列表，每条有描述 + 验收标准]

## 4. 用户故事地图
[按用户旅程阶段：发现/评估/上手/使用/留存/推荐]

## 5. 验收标准
| 功能 | 验收标准 | 优先级 |

## 6. 非功能需求
[性能/安全/可用性要求]

## 7. 排除项

## 8. 依赖与风险
\`\`\`

要求：
- 语言：中文
- 每个功能需有验收标准
- 成功指标需可量化`;
  }

  protected buildUserPrompt(ctx: RoleContext): string {
    const input = (ctx.input ?? {}) as {
      productName?: string;
      goal?: string;
      targetUsers?: string[];
      features?: string[];
      constraints?: string;
    };

    return `## 产品信息
- 产品名称: ${input.productName ?? "待定义"}
- 产品目标: ${input.goal ?? "待定义"}
- 目标用户: ${(input.targetUsers ?? []).join("、") || "待定义"}
- 核心功能: ${(input.features ?? []).join("、") || "待定义"}
- 约束条件: ${input.constraints ?? "无"}
- 工作目录: ${ctx.workdir}

请生成完整产品需求文档（PRD）。`;
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
        data: { prd: content },
        summary: `生成 ${lineCount} 行产品需求文档`,
        nextActions: [
          "产品评审会议",
          "确认功能范围",
          "冻结需求范围",
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
  generatePRD(input: {
    productName: string;
    goal: string;
    targetUsers: string[];
    features: string[];
  }): string {
    const { productName, goal, targetUsers, features } = input;

    return `# 产品需求文档 (PRD)

**产品名称**: ${productName}
**版本**: v0.1
**日期**: ${new Date().toLocaleDateString("zh-CN")}

## 1. 概述与目标

### 1.1 产品目标
${goal}

### 1.2 成功标准
- [ ] 指标1: 
- [ ] 指标2: 

## 2. 用户与用例

### 2.1 目标用户
${targetUsers.map((u) => `- ${u}`).join("\n")}

## 3. 功能需求
${features.map((f, i) => `${i + 1}. ${f}`).join("\n")}

## 4. 验收标准
| 功能 | 验收标准 | 优先级 |
|------|----------|--------|
${features.map((f) => `| ${f} | | P0/P1 |`).join("\n")}

> 注意：此为 fallback 模板。优先使用 LLM 生成的完整 PRD。`;
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
        "- [ ] 标准1\n- [ ] 标准2",
        "",
        "**优先级**: P0/P1/P2/P3",
        "",
        "---",
      ].join("\n"))
      .join("\n");
  }
}

export const productManager = new ProductManager();
