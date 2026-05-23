import { spawn } from "node:child_process";
import { mkdir, mkdtemp, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { createHarnessEnvironment } from "./createHarness.js";

describe("createHarnessEnvironment", () => {
  it("creates a knowledge-base fixture with multiple source repositories", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "pets-agent-harness-"));

    const environment = await createHarnessEnvironment({
      root,
      initializeGit: false
    });

    await expect(stat(path.join(environment.knowledgeBasePath, "CLAUDE.md"))).resolves.toBeTruthy();
    await expect(
      stat(path.join(environment.knowledgeBasePath, "docs", "business-processes", "order-flow.md"))
    ).resolves.toBeTruthy();
    await expect(
      stat(path.join(environment.knowledgeBasePath, "code", "catalog-api", "src", "index.ts"))
    ).resolves.toBeTruthy();
    await expect(
      stat(path.join(environment.knowledgeBasePath, "code", "order-service", "src", "index.ts"))
    ).resolves.toBeTruthy();
    expect(environment.repositories.map((repo) => repo.name)).toEqual([
      "catalog-api",
      "order-service"
    ]);
  });

  it("stops a running PID-backed service before reset", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "pets-agent-harness-"));
    const stateDir = path.join(root, "state");
    await mkdir(stateDir, { recursive: true });

    const child = spawn(process.execPath, ["-e", "setInterval(() => undefined, 1000);"], {
      stdio: "ignore",
    });
    if (child.pid === undefined) {
      throw new Error("Expected child process to have a pid.");
    }
    const childExit = new Promise<void>((resolve) => {
      child.once("exit", () => resolve());
    });

    await writeFile(path.join(stateDir, "server.pid"), `${child.pid}\n`, "utf8");

    try {
      await createHarnessEnvironment({
        root,
        reset: true,
        initializeGit: false
      });
      await childExit;
    } finally {
      if (isProcessAlive(child.pid)) {
        child.kill("SIGTERM");
      }
    }

    expect(isProcessAlive(child.pid)).toBe(false);
    await expect(stat(path.join(root, "knowledge-base", "CLAUDE.md"))).resolves.toBeTruthy();
  });

  it("ignores stale PID files during reset", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "pets-agent-harness-"));
    const stateDir = path.join(root, "state");
    await mkdir(stateDir, { recursive: true });
    await writeFile(path.join(stateDir, "server.pid"), "99999999\n", "utf8");

    const environment = await createHarnessEnvironment({
      root,
      reset: true,
      initializeGit: false
    });

    await expect(stat(path.join(environment.knowledgeBasePath, "CLAUDE.md"))).resolves.toBeTruthy();
    await expect(stat(path.join(root, "state", "server.pid"))).rejects.toMatchObject({ code: "ENOENT" });
  });

  it("does not stop a PID-backed process when reset is not requested", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "pets-agent-harness-"));
    const stateDir = path.join(root, "state");
    await mkdir(stateDir, { recursive: true });
    await writeFile(path.join(stateDir, "server.pid"), `${process.pid}\n`, "utf8");

    await createHarnessEnvironment({
      root,
      initializeGit: false
    });

    await expect(stat(path.join(root, "state", "server.pid"))).resolves.toBeTruthy();
  });
});

function isProcessAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}
