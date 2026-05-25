import { readFile } from "node:fs/promises";
import path from "node:path";
import { z } from "zod";
import type { ResolvedLlmConfig } from "./llmConfig.js";
import { resolveLlmConfig } from "./llmConfig.js";

const llmConfigSchema = z.object({
  baseUrl: z.url(),
  apiKeyEnv: z.string().min(1),
  modelId: z.string().min(1),
  maxTokens: z.number().int().positive().optional(),
});

const runtimeConfigSchema = z.object({
  port: z.number().int().positive().default(3000),
  host: z.string().min(1).default("0.0.0.0"),
  knowledgeBasePath: z.string().min(1).default(".harness/knowledge-base"),
  logDir: z.string().min(1).default(".harness/logs"),
  dbPath: z.string().min(1).default(".harness/state/agent.db"),
  sessionStorePath: z.string().min(1).default(".harness/state/sessions.json"),
  historyStorePath: z.string().min(1).default(".harness/state/history.json"),
  enableDevRoutes: z.boolean().default(true),
  wechat: z.object({
    token: z.string().min(1).default("dev-token"),
  }).default({ token: "dev-token" }),
  llm: llmConfigSchema,
});

export type RuntimeConfig = {
  readonly port: number;
  readonly host: string;
  readonly knowledgeBasePath: string;
  readonly logDir: string;
  readonly dbPath: string;
  readonly sessionStorePath: string;
  readonly historyStorePath: string;
  readonly enableDevRoutes: boolean;
  readonly wechat: {
    readonly token: string;
  };
  readonly llm: ResolvedLlmConfig;
};

export async function loadRuntimeConfig(
  configPath = process.env["CONFIG_PATH"] ?? path.resolve("config", "runtime.json"),
  env: NodeJS.ProcessEnv = process.env,
): Promise<RuntimeConfig> {
  let raw: string;
  try {
    raw = await readFile(configPath, "utf8");
  } catch (error) {
    throw new Error(`Failed to read config file at ${configPath}: ${error instanceof Error ? error.message : String(error)}`, { cause: error });
  }

  let json: unknown;
  try {
    json = JSON.parse(raw);
  } catch (error) {
    throw new Error(`Invalid JSON in config file at ${configPath}: ${error instanceof Error ? error.message : String(error)}`, { cause: error });
  }

  const parsed = runtimeConfigSchema.safeParse(json);
  if (!parsed.success) {
    const issues = parsed.error.issues.map((i) => `  - ${i.path.join(".")}: ${i.message}`).join("\n");
    throw new Error(`Invalid config in ${configPath}:\n${issues}`);
  }

  const resolvedLlm = resolveLlmConfig(parsed.data.llm, env);
  return { ...parsed.data, llm: resolvedLlm };
}
