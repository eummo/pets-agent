import { readFile } from "node:fs/promises";
import { z } from "zod";

const llmConfigSchema = z.object({
  baseUrl: z.string().url(),
  apiKeyEnv: z.string().min(1),
  modelId: z.string().min(1)
});

export type LlmConfig = z.infer<typeof llmConfigSchema>;

export type ResolvedLlmConfig = LlmConfig & {
  readonly apiKey: string;
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

  return {
    ...config,
    apiKey
  };
}

export function summarizeLlmConfig(config: LlmConfig): Pick<LlmConfig, "baseUrl" | "apiKeyEnv" | "modelId"> {
  return {
    baseUrl: config.baseUrl,
    apiKeyEnv: config.apiKeyEnv,
    modelId: config.modelId
  };
}
