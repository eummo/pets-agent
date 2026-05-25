import { readFile, stat } from "node:fs/promises";
import path from "node:path";
import { z } from "zod";
import type {
  InboundMessage,
  KnowledgeWorkspace,
  KnowledgeWorkspaceResolver
} from "../core/contracts.js";

export type ConfiguredWorkspaceResolverOptions = {
  readonly knowledgeBasePath: string;
  readonly repositoriesConfigPath?: string;
  readonly logger?: { write(event: Record<string, unknown>): Promise<void> };
};

const repositoriesConfigSchema = z.object({
  repositories: z.array(
    z.object({
      name: z.string(),
      aliases: z.array(z.string()),
      relativePath: z.string()
    })
  )
});

type RepositoryConfig = z.infer<typeof repositoriesConfigSchema>["repositories"][number];

export class ConfiguredWorkspaceResolver implements KnowledgeWorkspaceResolver {
  private cachedRepositories: readonly RepositoryConfig[] | undefined;
  private cachedMtimeMs: number | undefined;

  public constructor(private readonly options: ConfiguredWorkspaceResolverOptions) {}

  public async resolve(message: InboundMessage): Promise<readonly KnowledgeWorkspace[]> {
    const repository = await this.findMatchingRepository(message.text);

    if (repository !== undefined) {
      return [
        {
          kind: "source-repository",
          id: repository.name,
          path: path.resolve(path.dirname(this.repositoriesConfigPath()), repository.relativePath)
        }
      ];
    }

    return [
      {
        kind: "knowledge-base",
        id: "knowledge-base",
        path: path.resolve(this.options.knowledgeBasePath)
      }
    ];
  }

  private async findMatchingRepository(text: string): Promise<RepositoryConfig | undefined> {
    const repositories = await this.loadRepositories();
    const normalizedText = normalizeSearchText(text);

    return repositories.find((repository) =>
      [repository.name, ...repository.aliases].some((alias) =>
        normalizedText.includes(normalizeSearchText(alias))
      )
    );
  }

  private async loadRepositories(): Promise<readonly RepositoryConfig[]> {
    const configPath = this.repositoriesConfigPath();
    const fileStat = await stat(configPath).catch(() => undefined);
    if (fileStat === undefined) {
      this.cachedRepositories = [];
      this.cachedMtimeMs = undefined;
      return [];
    }

    if (this.cachedRepositories !== undefined && this.cachedMtimeMs === fileStat.mtimeMs) {
      return this.cachedRepositories;
    }

    try {
      const content = await readFile(configPath, "utf8");
      const repositories = repositoriesConfigSchema.parse(JSON.parse(content)).repositories;
      this.cachedRepositories = repositories;
      this.cachedMtimeMs = fileStat.mtimeMs;
      return repositories;
    } catch (error) {
      await this.options.logger?.write({
        type: "workspace.repositories_config_error",
        configPath,
        message: error instanceof Error ? error.message : String(error),
      });
      this.cachedRepositories = [];
      this.cachedMtimeMs = fileStat.mtimeMs;
      return [];
    }
  }

  private repositoriesConfigPath(): string {
    return path.resolve(this.options.repositoriesConfigPath ?? path.join(".harness", "repos.json"));
  }
}

function normalizeSearchText(value: string): string {
  return value.trim().toLowerCase().replace(/[\s_-]+/g, "");
}


