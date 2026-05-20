import { readFile } from "node:fs/promises";
import path from "node:path";
import { z } from "zod";
import type {
  InboundMessage,
  KnowledgeWorkspace,
  KnowledgeWorkspaceResolver
} from "../core/ports.js";

export type StaticWorkspaceResolverOptions = {
  readonly knowledgeBasePath: string;
  readonly repositoriesConfigPath?: string;
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

export class StaticWorkspaceResolver implements KnowledgeWorkspaceResolver {
  public constructor(private readonly options: StaticWorkspaceResolverOptions) {}

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
    try {
      const content = await readFile(this.repositoriesConfigPath(), "utf8");
      return repositoriesConfigSchema.parse(JSON.parse(content)).repositories;
    } catch {
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
