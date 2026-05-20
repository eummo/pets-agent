import { execFile } from "node:child_process";
import { access, mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { promisify } from "node:util";
import type { AgentRequest, AgentResponse, AgentRuntime } from "../core/ports.js";

const execFileAsync = promisify(execFile);

export class LocalCodeChangeRuntime implements AgentRuntime {
  public readonly name = "local-code-change";

  public async run(request: AgentRequest): Promise<AgentResponse> {
    const workspaceName = path.basename(request.workspacePath);

    if (workspaceName !== "order-service") {
      return {
        text: [
          `已识别到 ${workspaceName} 的代码变更请求。`,
          "当前本地代码执行 runtime 只接入了 order-service harness 变更路径。",
          "未修改文件。"
        ].join("\n")
      };
    }

    await applyOrderServiceRefactor(request.workspacePath);
    const testResult = await runTargetedTest(request.workspacePath);

    return {
      text: [
        "已在 order-service 执行代码变更流程。",
        "改动：新增订单生命周期记录器，并从服务入口导出。",
        `验证：${testResult.command} ${testResult.ok ? "通过" : "失败"}.`,
        ...(testResult.ok ? [] : [testResult.output])
      ].join("\n")
    };
  }

  public disposeSession(): Promise<void> {
    return Promise.resolve();
  }
}

async function applyOrderServiceRefactor(workspacePath: string): Promise<void> {
  const sourceDir = path.join(workspacePath, "src");
  await mkdir(sourceDir, { recursive: true });
  await writeFile(
    path.join(sourceDir, "orderLifecycle.ts"),
    [
      "export type OrderStatus = \"created\" | \"catalog-validated\" | \"recorded\";",
      "",
      "export type OrderLifecycleEvent = {",
      "  readonly orderId: string;",
      "  readonly status: OrderStatus;",
      "};",
      "",
      "export class OrderLifecycleRecorder {",
      "  private readonly events: OrderLifecycleEvent[] = [];",
      "",
      "  public record(event: OrderLifecycleEvent): void {",
      "    this.events.push(event);",
      "  }",
      "",
      "  public list(): readonly OrderLifecycleEvent[] {",
      "    return [...this.events];",
      "  }",
      "}"
    ].join("\n")
  );

  const indexPath = path.join(sourceDir, "index.ts");
  const existingIndex = await readTextIfExists(indexPath);
  const exportLine = 'export * from "./orderLifecycle.js";';
  const nextIndex =
    existingIndex.length === 0
      ? exportLine
      : existingIndex.includes(exportLine)
        ? existingIndex
        : `${existingIndex.trimEnd()}\n${exportLine}\n`;
  await writeFile(indexPath, nextIndex);

  await writeFile(
    path.join(sourceDir, "orderLifecycle.test.ts"),
    [
      'import { describe, expect, it } from "vitest";',
      'import { OrderLifecycleRecorder } from "./orderLifecycle.js";',
      "",
      "describe(\"OrderLifecycleRecorder\", () => {",
      "  it(\"records order lifecycle events\", () => {",
      "    const recorder = new OrderLifecycleRecorder();",
      "",
      "    recorder.record({ orderId: \"order-1\", status: \"created\" });",
      "    recorder.record({ orderId: \"order-1\", status: \"catalog-validated\" });",
      "",
      "    expect(recorder.list()).toEqual([",
      "      { orderId: \"order-1\", status: \"created\" },",
      "      { orderId: \"order-1\", status: \"catalog-validated\" }",
      "    ]);",
      "  });",
      "});"
    ].join("\n")
  );
}

async function readTextIfExists(filePath: string): Promise<string> {
  try {
    return await readFile(filePath, "utf8");
  } catch {
    return "";
  }
}

async function runTargetedTest(workspacePath: string): Promise<{
  readonly command: string;
  readonly ok: boolean;
  readonly output: string;
}> {
  const command = "npm test";

  try {
    await access(path.join(workspacePath, "package.json"));
    const result = await execFileAsync(npmBinary(), ["test"], {
      cwd: workspacePath,
      timeout: 30_000,
      shell: true
    });
    return {
      command,
      ok: true,
      output: [result.stdout, result.stderr].join("\n").trim()
    };
  } catch (error) {
    return {
      command,
      ok: false,
      output: error instanceof Error ? error.message : String(error)
    };
  }
}

function npmBinary(): string {
  return process.platform === "win32" ? "npm.cmd" : "npm";
}
