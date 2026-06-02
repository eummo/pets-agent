import type { Api, Model } from "@earendil-works/pi-ai";
import type { ResolvedLlmConfig } from "../config/llmConfig.js";

const PI_MODEL_DEFAULTS = {
  api: "anthropic-messages",
  provider: "pets-agent",
  contextWindow: 200000,
  maxTokens: 8192,
  reasoning: false,
  input: ["text"] as readonly ("text" | "image")[]
} as const;

export function buildPiModel(config: ResolvedLlmConfig): Model<Api> {
  return {
    id: config.modelId,
    name: config.modelId,
    api: PI_MODEL_DEFAULTS.api,
    provider: PI_MODEL_DEFAULTS.provider,
    baseUrl: config.baseUrl.replace(/\/+$/, ""),
    reasoning: PI_MODEL_DEFAULTS.reasoning,
    input: [...PI_MODEL_DEFAULTS.input],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: PI_MODEL_DEFAULTS.contextWindow,
    maxTokens: config.maxTokens ?? PI_MODEL_DEFAULTS.maxTokens
  };
}
