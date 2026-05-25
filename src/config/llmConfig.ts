import type { Model } from "@earendil-works/pi-ai";

const DEFAULT_MAX_TOKENS = 256;

export type LlmConfig = {
  readonly baseUrl: string;
  readonly apiKeyEnv: string;
  readonly modelId: string;
  readonly maxTokens?: number | undefined;
};

export type ResolvedLlmConfig = LlmConfig & {
  readonly apiKey: string;
};

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
    maxTokens: config.maxTokens ?? DEFAULT_MAX_TOKENS,
  };
}
