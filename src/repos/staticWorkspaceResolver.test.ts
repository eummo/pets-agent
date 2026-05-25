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

  it("reloads repository config when the file changes", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "pets-agent-repos-"));
    const repositoriesConfigPath = path.join(root, "repos.json");
    await writeRepositoriesConfig(repositoriesConfigPath, "order-service", ["订单系统"]);

    const resolver = new StaticWorkspaceResolver({
      knowledgeBasePath: path.join(root, "knowledge-base"),
      repositoriesConfigPath
    });

    await expect(resolveText(resolver, "订单系统")).resolves.toEqual([
      expect.objectContaining({ id: "order-service" })
    ]);

    await new Promise((resolve) => setTimeout(resolve, 5));
    await writeRepositoriesConfig(repositoriesConfigPath, "catalog-api", ["商品系统"]);

    await expect(resolveText(resolver, "商品系统")).resolves.toEqual([
      expect.objectContaining({ id: "catalog-api" })
    ]);
  });

  it("logs invalid repository config and falls back to the knowledge base", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "pets-agent-repos-"));
    const repositoriesConfigPath = path.join(root, "repos.json");
    const events: Record<string, unknown>[] = [];
    await writeFile(repositoriesConfigPath, "{ invalid json", "utf8");

    const resolver = new StaticWorkspaceResolver({
      knowledgeBasePath: path.join(root, "knowledge-base"),
      repositoriesConfigPath,
      logger: {
        write(event) {
          events.push(event);
          return Promise.resolve();
        }
      }
    });

    await expect(resolveText(resolver, "订单系统")).resolves.toEqual([
      {
        kind: "knowledge-base",
        id: "knowledge-base",
        path: path.join(root, "knowledge-base")
      }
    ]);
    expect(events).toEqual([
      expect.objectContaining({ type: "workspace.repositories_config_error", configPath: repositoriesConfigPath })
    ]);
  });
});

async function writeRepositoriesConfig(
  repositoriesConfigPath: string,
  name: string,
  aliases: readonly string[],
): Promise<void> {
  await writeFile(
    repositoriesConfigPath,
    JSON.stringify({
      repositories: [
        {
          name,
          aliases,
          relativePath: `knowledge-base/code/${name}`
        }
      ]
    }),
    "utf8"
  );
}

function resolveText(resolver: StaticWorkspaceResolver, text: string) {
  return resolver.resolve({
    id: "1",
    channel: "test",
    user: { id: "user-1" },
    text,
    receivedAt: new Date()
  });
}
