/**
 * QATester agent — test strategy, cases, defect management.
 * LLM-powered via BaseRoleLLM.
 */

import { BaseRoleLLM } from "../base-role-llm.js";
import { type RoleContext } from "../role.js";
import type { TeamRole, ProjectPhase } from "../types.js";
import { PHASE_LABELS } from "../types.js";

export class QATester extends BaseRoleLLM {
  readonly name: TeamRole = "qa";
  readonly description = "负责测试策略、用例设计、缺陷管理、回归测试。";

  protected role(): TeamRole {
    return "qa";
  }

  protected override buildSystemPrompt(ctx: RoleContext): string {
    const phase = ctx.phase as ProjectPhase;
    return `你是资深 QA 测试工程师，负责测试计划、用例设计、缺陷分析。

当前阶段: ${PHASE_LABELS[phase]}
项目目录: ${ctx.workdir}

你的职责：
1. 制定测试策略（单元/集成/E2E/性能）
2. 设计测试用例，覆盖正向/逆向/边界
3. 执行测试并报告缺陷
4. 评估测试覆盖率和质量

**输出质量标准：**
- 每个功能模块必须覆盖正向用例、逆向用例、边界值测试
- 测试用例需包含：ID、模块名、前置条件、操作步骤、预期结果、优先级
- 缺陷等级必须包含 P0（阻断）、P1（严重）、P2（一般）、P3（建议）
- 测试环境需说明数据准备、依赖服务、初始化脚本

**示例输出片段：**

\`\`\`
## 4. 测试用例

### TC-001: 用户登录成功

| 字段 | 值 |
|------|-----|
| ID | TC-001 |
| 模块 | 认证模块 |
| 前置条件 | 用户已注册，邮箱 verified |
| 操作步骤 | 1. 输入正确邮箱 2. 输入正确密码 3. 点击登录 |
| 预期结果 | 登录成功，跳转至首页，显示用户头像 |
| 优先级 | P0 |
| 类型 | 正向用例 |

### TC-002: 登录密码错误

| 字段 | 值 |
|------|-----|
| ID | TC-002 |
| 模块 | 认证模块 |
| 前置条件 | 用户已注册 |
| 操作步骤 | 1. 输入正确邮箱 2. 输入错误密码 3. 点击登录 |
| 预期结果 | 登录失败，提示"邮箱或密码错误"，不跳转 |
| 优先级 | P0 |
| 类型 | 逆向用例 |

### TC-003: 边界——密码长度

| 字段 | 值 |
|------|-----|
| ID | TC-003 |
| 模块 | 认证模块 |
| 前置条件 | — |
| 操作步骤 | 密码输入 5 个字符（最小长度边界） |
| 预期结果 | 输入框下方提示"密码至少6位" |
| 优先级 | P2 |
| 类型 | 边界值测试 |
\`\`\`
\`\`\`

输出格式（Markdown）：
\`\`\`
# 测试计划

## 1. 测试范围
| 功能模块 | 测试类型 | 优先级 |

## 2. 测试策略
| 测试级别 | 目标 | 工具 |

## 3. 测试环境
[环境配置/测试数据]

## 4. 测试用例
| ID | 功能 | 前置条件 | 步骤 | 预期结果 | 优先级 |

## 5. 缺陷管理
[缺陷等级/提交规范/修复周期]

## 6. 回归测试策略
\`\`\``;
  }

  protected buildUserPrompt(ctx: RoleContext): string {
    const input = (ctx.input ?? {}) as {
      projectName?: string;
      features?: string[];
      testLevels?: string[];
    };

    return `## 测试信息
- 项目名称: ${input.projectName ?? "待定义"}
- 测试功能: ${(input.features ?? []).join("、") || "待定义"}
- 测试级别: ${(input.testLevels ?? ["单元测试", "集成测试", "E2E测试"]).join("、")}
- 工作目录: ${ctx.workdir}

请生成完整测试计划文档。`;
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
        data: { testPlan: content },
        summary: `生成 ${lineCount} 行测试计划`,
        nextActions: [
          "编写详细测试用例",
          "搭建测试环境",
          "执行冒烟测试",
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
  generateTestPlan(input: {
    projectName: string;
    features: string[];
    testLevels?: string[];
  }): string {
    const { projectName, features, testLevels } = input;
    const levels = testLevels ?? ["单元测试", "集成测试", "E2E测试"];

    return `# 测试计划

**项目**: ${projectName}
**版本**: v0.1
**日期**: ${new Date().toLocaleDateString("zh-CN")}

## 1. 测试范围
| 功能模块 | 测试类型 | 优先级 |
|----------|----------|--------|
${features.map((f) => `| ${f} | 功能测试 | P1 |`).join("\n")}

## 2. 测试策略
${levels.map((l) => `- ${l}`).join("\n")}

## 3. 缺陷等级
- P0: 阻断（立即修复）
- P1: 严重（24h修复）
- P2: 一般（72h修复）
- P3: 建议（下一版本）

> 注意：此为 fallback 模板。优先使用 LLM 生成的完整测试计划。`;
  }

  generateTestReport(input: {
    projectName: string;
    totalCases: number;
    passedCases: number;
    failedCases: number;
    blockers: string[];
  }): string {
    const { projectName, totalCases, passedCases, failedCases, blockers } = input;
    const passRate = totalCases > 0 ? Math.round((passedCases / totalCases) * 100) : 0;

    return `# 测试报告

**项目**: ${projectName}
**日期**: ${new Date().toLocaleDateString("zh-CN")}

## 1. 测试概览
| 指标 | 数值 |
|------|------|
| 总用例数 | ${totalCases} |
| 通过数 | ${passedCases} |
| 失败数 | ${failedCases} |
| 通过率 | ${passRate}% |

## 2. 结论
${passRate >= 80 ? "✓ 通过率 ≥ 80%，达到上线标准" : "✗ 通过率不足，未达到上线标准"}

## 3. 缺陷
${blockers.length > 0
  ? blockers.map((b, i) => `- DEF${String(i+1).padStart(3,"0")}: ${b}`).join("\n")
  : "- 无阻塞缺陷"}`;
  }
}

export const qaTester = new QATester();
