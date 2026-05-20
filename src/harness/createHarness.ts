import { execFileSync } from "node:child_process";
import { mkdir, rm, writeFile } from "node:fs/promises";
import path from "node:path";

export type HarnessRepository = {
  readonly name: string;
  readonly aliases: readonly string[];
  readonly relativePath: string;
  readonly defaultBranch: string;
  readonly testCommand: string;
};

export type HarnessEnvironment = {
  readonly root: string;
  readonly knowledgeBasePath: string;
  readonly repositories: readonly HarnessRepository[];
};

export type CreateHarnessOptions = {
  readonly root: string;
  readonly reset?: boolean;
  readonly initializeGit?: boolean;
};

const repositories = [
  {
    name: "catalog-api",
    aliases: ["catalog", "catalog system", "catalog service", "商品系统", "目录服务"],
    relativePath: "knowledge-base/code/catalog-api",
    defaultBranch: "main",
    testCommand: "npm test"
  },
  {
    name: "order-service",
    aliases: ["orders", "order system", "order service", "订单系统", "订单服务"],
    relativePath: "knowledge-base/code/order-service",
    defaultBranch: "main",
    testCommand: "npm test"
  }
] as const satisfies readonly HarnessRepository[];

export async function createHarnessEnvironment(
  options: CreateHarnessOptions
): Promise<HarnessEnvironment> {
  const root = path.resolve(options.root);
  const knowledgeBasePath = path.join(root, "knowledge-base");

  if (options.reset === true) {
    await rm(root, { recursive: true, force: true });
  }

  await mkdir(path.join(knowledgeBasePath, "docs", "business-processes"), { recursive: true });
  await mkdir(path.join(knowledgeBasePath, "requirements"), { recursive: true });
  await mkdir(path.join(knowledgeBasePath, ".claude", "skills"), { recursive: true });
  await mkdir(path.join(knowledgeBasePath, ".claude", "commands"), { recursive: true });

  await writeFile(
    path.join(knowledgeBasePath, "CLAUDE.md"),
    [
      "# Pets Agent Knowledge Base",
      "",
      "This workspace documents the order-domain system used for local knowledge-base testing.",
      "The documented business flow starts with customer order creation, validates item availability through the catalog system, and records the order lifecycle in the order service."
    ].join("\n")
  );
  await writeFile(
    path.join(knowledgeBasePath, "docs", "business-processes", "order-flow.md"),
    [
      "# Order Flow",
      "",
      "1. A customer creates an order.",
      "2. The catalog system validates item availability.",
      "3. The order service records the order lifecycle."
    ].join("\n")
  );
  await writeFile(
    path.join(knowledgeBasePath, "docs", "test-entrypoints.md"),
    [
      "# Test Entrypoints",
      "",
      "The local browser test page is available at http://127.0.0.1:3000/.",
      "The development chat API is POST /dev/chat.",
      "The Enterprise WeChat callback is /wechat/callback.",
      "The health check is /health."
    ].join("\n")
  );
  await writeFile(
    path.join(knowledgeBasePath, "requirements", "REQ-0001.md"),
    [
      "# REQ-0001 Order Flow Knowledge",
      "",
      "Users need to understand how order creation, catalog availability checks, and order lifecycle recording fit together.",
      "Code-related questions about catalog behavior should use the catalog-api repository.",
      "Code-related questions about order lifecycle behavior should use the order-service repository."
    ].join("\n")
  );

  for (const repository of repositories) {
    await createRepositoryFixture(root, repository, options.initializeGit !== false);
  }

  await writeFile(
    path.join(root, "repos.json"),
    `${JSON.stringify({ repositories }, null, 2)}\n`
  );

  return { root, knowledgeBasePath, repositories };
}

async function createRepositoryFixture(
  root: string,
  repository: HarnessRepository,
  initializeGit: boolean
): Promise<void> {
  const repoPath = path.join(root, repository.relativePath);
  await mkdir(path.join(repoPath, "src"), { recursive: true });
  await writeFile(
    path.join(repoPath, "CLAUDE.md"),
    `# ${repository.name}\n\nUse this repository workspace for ${repository.name} code tasks.\n`
  );
  await writeFile(
    path.join(repoPath, "src", "index.ts"),
    [
      `export const serviceName = "${repository.name}";`,
      "",
      "export function health(): string {",
      "  return `${serviceName}:ok`;",
      "}"
    ].join("\n")
  );
  await writeFile(
    path.join(repoPath, "package.json"),
    `${JSON.stringify(
      {
        name: repository.name,
        version: "0.0.0",
        private: true,
        type: "module",
        scripts: {
          test: "node -e \"process.exit(0)\""
        }
      },
      null,
      2
    )}\n`
  );

  if (initializeGit) {
    initializeRepository(repoPath, repository.defaultBranch);
  }
}

function initializeRepository(repoPath: string, branch: string): void {
  try {
    execFileSync("git", ["init", "-b", branch], { cwd: repoPath, stdio: "ignore" });
    execFileSync("git", ["add", "."], { cwd: repoPath, stdio: "ignore" });
    execFileSync("git", ["commit", "-m", "chore: initialize harness repository"], {
      cwd: repoPath,
      stdio: "ignore"
    });
  } catch {
    // Git may be unavailable or user.name/user.email may be missing. The file fixture is still useful.
  }
}
