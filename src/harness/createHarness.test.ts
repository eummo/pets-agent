import { mkdtemp, stat } from "node:fs/promises";
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
});
