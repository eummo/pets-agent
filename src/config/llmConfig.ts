import { readFile } from "node:fs/promises";
import { z } from "zod";

const llmConfigSchema = z.object({
  runtime: z.enum(["messages", "managed-sessions"]).optional(),
  baseUrl: z.string().url(),
  apiKeyEnv: z.string().min(1),
  modelId: z.string().min(1),
  agentIdEnv: z.string().min(1).optional(),
  environmentIdEnv: z.string().min(1).optional()
});

export type LlmConfig = z.infer<typeof llmConfigSchema>;

export type ResolvedLlmConfig = LlmConfig & {
  readonly apiKey: string;
  readonly agentId?: string;
  readonly environmentId?: string;
};

export async function loadLlmConfig(path: string): Promise<LlmConfig> {
  const raw = await readFile(path, "utf8");
  return llmConfigSchema.parse(JSON.parse(raw) as unknown);
}

export function resolveLlmConfig(config: LlmConfig, env: NodeJS.ProcessEnv = process.env): ResolvedLlmConfig {
  const apiKey = env[config.apiKeyEnv];

  if (apiKey === undefined || apiKey.trim().length === 0) {
    throw new Error(`Missing LLM API key environment variable: ${config.apiKeyEnv}`);
  }

  if (config.runtime !== "managed-sessions") {
    return {
      ...config,
      apiKey
    };
  }

  if (config.agentIdEnv === undefined) {
    throw new Error("Managed sessions runtime requires agentIdEnv in LLM config.");
  }

  if (config.environmentIdEnv === undefined) {
    throw new Error("Managed sessions runtime requires environmentIdEnv in LLM config.");
  }

  const agentId = env[config.agentIdEnv];
  const environmentId = env[config.environmentIdEnv];

  if (agentId === undefined || agentId.trim().length === 0) {
    throw new Error(`Missing managed agent environment variable: ${config.agentIdEnv}`);
  }

  if (environmentId === undefined || environmentId.trim().length === 0) {
    throw new Error(`Missing managed agent environment variable: ${config.environmentIdEnv}`);
  }

  return {
    ...config,
    apiKey,
    agentId,
    environmentId
  };
}

export function summarizeLlmConfig(config: LlmConfig): Pick<LlmConfig, "baseUrl" | "apiKeyEnv" | "modelId"> {
  return {
    baseUrl: config.baseUrl,
    apiKeyEnv: config.apiKeyEnv,
    modelId: config.modelId
  };
}
