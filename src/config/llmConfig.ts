import { readFile } from "node:fs/promises";
import type { Model } from "@earendil-works/pi-ai";
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

export function buildPiModel(config: ResolvedLlmConfig): Model<"anthropic-messages"> {
  return {
    id: config.modelId,
    name: config.modelId,
    api: "anthropic-messages",
    provider: "pets-agent",
    baseUrl: config.baseUrl.replace(/\/+$/, ""),
    reasoning: false,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 200000,
    maxTokens: 256,
  };
}
