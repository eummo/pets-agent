import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { LocalCodeChangeRuntime } from "./localCodeChangeRuntime.js";

describe("LocalCodeChangeRuntime", () => {
  it("applies and verifies the order-service harness refactor", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "pets-agent-code-runtime-"));
    const workspacePath = path.join(root, "order-service");
    await mkdir(path.join(workspacePath, "src"), { recursive: true });
    await writeFile(path.join(workspacePath, "src", "index.ts"), 'export const serviceName = "order-service";\n');
    await writeFile(
      path.join(workspacePath, "package.json"),
      JSON.stringify({
        type: "module",
        scripts: {
          test: "node -e \"process.exit(0)\""
        }
      })
    );

    const response = await new LocalCodeChangeRuntime().run({
      user: { id: "developer-1" },
      text: "重构订单系统",
      workspacePath
    });

    await expect(readFile(path.join(workspacePath, "src", "orderLifecycle.ts"), "utf8")).resolves.toContain(
      "OrderLifecycleRecorder"
    );
    await expect(readFile(path.join(workspacePath, "src", "index.ts"), "utf8")).resolves.toContain(
      'export * from "./orderLifecycle.js";'
    );
    expect(response.text).toContain("order-service");
    expect(response.text).toContain("通过");
  });

  it("returns an early response for non-order-service workspaces", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "pets-agent-code-runtime-"));
    const workspacePath = path.join(root, "catalog-service");

    const response = await new LocalCodeChangeRuntime().run({
      user: { id: "developer-1" },
      text: "修改目录服务",
      workspacePath
    });

    expect(response.text).toContain("catalog-service");
    expect(response.text).toContain("未修改文件");
    expect(response.text).not.toContain("执行代码变更流程");
  });

  it("reports test failure when npm test fails", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "pets-agent-code-runtime-"));
    const workspacePath = path.join(root, "order-service");
    await mkdir(path.join(workspacePath, "src"), { recursive: true });
    await writeFile(path.join(workspacePath, "src", "index.ts"), 'export const serviceName = "order-service";\n');
    await writeFile(
      path.join(workspacePath, "package.json"),
      JSON.stringify({
        type: "module",
        scripts: {
          test: "node -e \"process.exit(1)\""
        }
      })
    );

    const response = await new LocalCodeChangeRuntime().run({
      user: { id: "developer-1" },
      text: "重构订单系统",
      workspacePath
    });

    expect(response.text).toContain("失败");
    expect(response.text).not.toContain("通过");
  });

  it("is idempotent when run twice on the same workspace", { timeout: 15_000 }, async () => {
    const root = await mkdtemp(path.join(tmpdir(), "pets-agent-code-runtime-"));
    const workspacePath = path.join(root, "order-service");
    await mkdir(path.join(workspacePath, "src"), { recursive: true });
    await writeFile(path.join(workspacePath, "src", "index.ts"), 'export const serviceName = "order-service";\n');
    await writeFile(
      path.join(workspacePath, "package.json"),
      JSON.stringify({
        type: "module",
        scripts: {
          test: "node -e \"process.exit(0)\""
        }
      })
    );

    const runtime = new LocalCodeChangeRuntime();
    await runtime.run({
      user: { id: "developer-1" },
      text: "重构订单系统",
      workspacePath
    });

    const secondResponse = await runtime.run({
      user: { id: "developer-1" },
      text: "重构订单系统",
      workspacePath
    });

    expect(secondResponse.text).toContain("通过");
    const indexContent = await readFile(path.join(workspacePath, "src", "index.ts"), "utf8");
    const exportOccurrences = indexContent.match(/export \* from "\.\/orderLifecycle\.js"/g);
    expect(exportOccurrences).toHaveLength(1);
  });

  it("resolves disposeSession without error", async () => {
    const runtime = new LocalCodeChangeRuntime();
    await expect(runtime.disposeSession()).resolves.toBeUndefined();
  });
});
