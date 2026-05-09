/**
 * QATester agent — test strategy, cases, defect management.
 */

import { Role, type RoleContext } from "../role.js";
import type { TeamRole } from "../types.js";

export class QATester extends Role {
  readonly name: TeamRole = "qa";
  readonly description = "负责测试策略、用例设计、缺陷管理、回归测试。";

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
      summary: "QA tester ready. Execute with specific testing task.",
      nextActions: ["Define test strategy", "Write test cases", "Execute tests", "Report defects"],
    };
  }

  generateTestPlan(input: {
    projectName: string;
    features: string[];
    testLevels?: string[];
  }): string {
    const { projectName, features, testLevels } = input;
    const levels = testLevels ?? ["单元测试", "集成测试", "端到端测试", "性能测试"];

    const sections = [
      "# 测试计划",
      "",
      `**项目**: ${projectName}`,
      `**版本**: v0.1`,
      `**日期**: ${new Date().toLocaleDateString("zh-CN")}`,
      "",
      "## 1. 测试范围",
      "| 功能模块 | 测试类型 | 优先级 | 状态 |",
      "|----------|----------|--------|------|",
      ...features.map((f) => `| ${f} | 功能测试 | P1 | 待测 |`),
      "",
      "## 2. 测试策略",
      "| 测试级别 | 目标 | 入口标准 | 出口标准 |",
      "|----------|------|----------|----------|",
      ...levels.map((l) => `| ${l} | | | |`),
      "",
      "## 3. 测试环境",
      "- 测试数据: ",
      "- 环境配置: ",
      "- 工具: Jest / Playwright / k6",
      "",
      "## 4. 测试用例示例",
      "| ID | 功能 | 前置条件 | 测试步骤 | 预期结果 | 优先级 |",
      "|----|------|----------|----------|----------|--------|",
      "| TC001 | | | | | P0 |",
      "| TC002 | | | | | P1 |",
      "",
      "## 5. 缺陷管理",
      "- 缺陷等级: P0(阻断) / P1(严重) / P2(一般) / P3(建议)",
      "- 提交规范: 复现步骤 + 预期/实际结果 + 截图",
      "- 修复周期: P0 立即 / P1 24h / P2 72h / P3 下一版本",
      "",
      "## 6. 回归测试",
      "- 触发条件: 每次代码变更",
      "- 执行方式: CI自动 + 人工确认",
      "- 覆盖范围: 全量回归",
      "",
      "## 7. 测试进度跟踪",
      "| 指标 | 目标 | 实际 |",
      "|------|------|------|",
      "| 用例执行率 | 100% | |",
      "| 缺陷发现数 | - | |",
      "| 缺陷修复率 | ≥ 95% | |",
      "| 测试覆盖率 | ≥ 80% | |",
    ];

    return sections.join("\n");
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

    const sections = [
      "# 测试报告",
      "",
      `**项目**: ${projectName}`,
      `**日期**: ${new Date().toLocaleDateString("zh-CN")}`,
      "",
      "## 1. 测试概览",
      "| 指标 | 数值 |",
      "|------|------|",
      `| 总用例数 | ${totalCases} |`,
      `| 通过数 | ${passedCases} |`,
      `| 失败数 | ${failedCases} |`,
      `| 通过率 | ${passRate}% |`,
      "",
      "## 2. 测试结果摘要",
      passRate >= 80
        ? `✓ 测试通过率 ${passRate}%，达到上线标准`
        : `✗ 测试通过率 ${passRate}%，未达到上线标准 (≥80%)`,
      "",
      "## 3. 缺陷汇总",
      blockers.length > 0
        ? [
            "| ID | 描述 | 严重度 | 状态 |",
            "|----|------|--------|------|",
            ...blockers.map((b, i) => `| DEF${String(i + 1).padStart(3, "0")} | ${b} | | 待修复 |`),
          ].join("\n")
        : "- 无阻塞缺陷",
      "",
      "## 4. 测试覆盖率",
      "- 代码覆盖率: %",
      "- 需求覆盖率: %",
      "- 场景覆盖率: %",
      "",
      "## 5. 结论与建议",
      `- 测试结论: ${passRate >= 80 ? "通过" : "不通过"}`,
      "- 主要风险: ",
      "- 下一步行动: ",
    ];

    return sections.join("\n");
  }
}

export const qaTester = new QATester();
