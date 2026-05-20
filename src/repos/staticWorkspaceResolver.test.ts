import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { StaticWorkspaceResolver } from "./staticWorkspaceResolver.js";

describe("StaticWorkspaceResolver", () => {
  it("routes repository aliases to source repository workspaces", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "pets-agent-repos-"));
    const repositoriesConfigPath = path.join(root, "repos.json");
    await writeFile(
      repositoriesConfigPath,
      JSON.stringify({
        repositories: [
          {
            name: "order-service",
            aliases: ["orders", "order system", "订单系统", "订单服务"],
            relativePath: "knowledge-base/code/order-service"
          }
        ]
      })
    );

    const resolver = new StaticWorkspaceResolver({
      knowledgeBasePath: path.join(root, "knowledge-base"),
      repositoriesConfigPath
    });

    await expect(
      resolver.resolve({
        id: "1",
        channel: "test",
        user: { id: "user-1" },
        text: "重构订单系统",
        receivedAt: new Date()
      })
    ).resolves.toEqual([
      {
        kind: "source-repository",
        id: "order-service",
        path: path.join(root, "knowledge-base", "code", "order-service")
      }
    ]);
  });
});
