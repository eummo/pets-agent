import { mkdir, mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { buildWorkspaceContext } from "./workspaceContext.js";

describe("buildWorkspaceContext", () => {
  it("reads key knowledge-base files as model grounding context", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "pets-agent-context-"));
    await mkdir(path.join(root, "docs"), { recursive: true });
    await writeFile(path.join(root, "CLAUDE.md"), "Project assistant instructions.");
    await writeFile(path.join(root, "docs", "business.md"), "This project manages order flows.");

    const context = await buildWorkspaceContext({ workspacePath: root });

    expect(context).toContain("--- CLAUDE.md ---");
    expect(context).toContain("Project assistant instructions.");
    expect(context).toContain("--- docs");
    expect(context).toContain("This project manages order flows.");
  });

  it("only includes test entrypoint documents when the query asks about testing", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "pets-agent-context-"));
    await mkdir(path.join(root, "docs"), { recursive: true });
    await writeFile(path.join(root, "CLAUDE.md"), "Order domain workspace.");
    await writeFile(path.join(root, "docs", "test-entrypoints.md"), "The browser test page is available.");

    const architectureContext = await buildWorkspaceContext({
      workspacePath: root,
      query: "What is the current architecture?"
    });
    const testingContext = await buildWorkspaceContext({
      workspacePath: root,
      query: "Which test entrypoints are available?"
    });

    expect(architectureContext).not.toContain("browser test page");
    expect(testingContext).toContain("browser test page");
  });

  it("returns a not-found message for empty workspaces", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "pets-agent-context-"));

    const context = await buildWorkspaceContext({ workspacePath: root });

    expect(context).toBe("No readable workspace context files were found.");
  });

  it("truncates files exceeding maxBytesPerFile", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "pets-agent-context-"));
    const longContent = "A".repeat(5000);
    await writeFile(path.join(root, "README.md"), longContent);

    const context = await buildWorkspaceContext({
      workspacePath: root,
      maxBytesPerFile: 100
    });

    expect(context).toContain("[truncated]");
  });

  it("skips .git, node_modules, dist, and coverage directories", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "pets-agent-context-"));
    await mkdir(path.join(root, ".git"), { recursive: true });
    await mkdir(path.join(root, "node_modules"), { recursive: true });
    await mkdir(path.join(root, "dist"), { recursive: true });
    await mkdir(path.join(root, "coverage"), { recursive: true });
    await writeFile(path.join(root, ".git", "config.md"), "git config");
    await writeFile(path.join(root, "node_modules", "pkg.md"), "package docs");
    await writeFile(path.join(root, "dist", "output.md"), "build output");
    await writeFile(path.join(root, "coverage", "report.md"), "coverage report");
    await writeFile(path.join(root, "README.md"), "readme content");

    const context = await buildWorkspaceContext({ workspacePath: root });

    expect(context).toContain("readme content");
    expect(context).not.toContain("git config");
    expect(context).not.toContain("package docs");
    expect(context).not.toContain("build output");
    expect(context).not.toContain("coverage report");
  });

  it("respects the maxFiles limit", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "pets-agent-context-"));
    for (let i = 0; i < 5; i++) {
      await writeFile(path.join(root, `file-${i}.md`), `Content ${i}`);
    }

    const context = await buildWorkspaceContext({
      workspacePath: root,
      maxFiles: 2
    });

    const fileCount = (context.match(/--- .* ---/g) ?? []).length;
    expect(fileCount).toBeLessThanOrEqual(2);
  });
});
