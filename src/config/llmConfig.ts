import type { Api, Model } from "@earendil-works/pi-ai";

// ── Defaults for agent SDK model construction ────────────────────────────────

export const AGENT_SDK_DEFAULTS = {
  api: "anthropic-messages",
  provider: "pets-agent",
  contextWindow: 200000,
  maxTokens: 256,
  reasoning: false,
  input: ["text"] as readonly ("text" | "image")[],
} as const;

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

export function buildPiModel(config: ResolvedLlmConfig): Model<Api> {
  return {
    id: config.modelId,
    name: config.modelId,
    api: AGENT_SDK_DEFAULTS.api,
    provider: AGENT_SDK_DEFAULTS.provider,
    baseUrl: config.baseUrl.replace(/\/+$/, ""),
    reasoning: AGENT_SDK_DEFAULTS.reasoning,
    input: [...AGENT_SDK_DEFAULTS.input],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: AGENT_SDK_DEFAULTS.contextWindow,
    maxTokens: config.maxTokens ?? AGENT_SDK_DEFAULTS.maxTokens,
  };
}

// ── Agent SDK Configuration ─────────────────────────────────────────────────
// Selects which agent runtime SDK to use and provides its connection config.

export type AgentSdkType = "claude" | "codebuddy" | "pi";

export type AgentSdkConfig = {
  readonly type: AgentSdkType;
  readonly baseUrl: string;
  readonly apiKeyEnv: string;
  readonly modelId: string;
  readonly agentDir?: string | undefined;
  readonly provider?: string | undefined;
  readonly api?: string | undefined;
  readonly contextWindow?: number | undefined;
  readonly reasoning?: boolean | undefined;
  readonly input?: readonly ("text" | "image")[] | undefined;
};

export type ResolvedAgentSdkConfig = AgentSdkConfig & {
  readonly apiKey: string;
};

export function resolveAgentSdkConfig(config: AgentSdkConfig, env: NodeJS.ProcessEnv = process.env): ResolvedAgentSdkConfig {
  const apiKey = env[config.apiKeyEnv];

  if (apiKey === undefined || apiKey.trim().length === 0) {
    throw new Error(`Missing Agent SDK API key environment variable: ${config.apiKeyEnv}`);
  }

  return {
    ...config,
    apiKey
  };
}

export function summarizeAgentSdkConfig(config: AgentSdkConfig): { readonly type: AgentSdkType; readonly baseUrl: string; readonly apiKeyEnv: string; readonly modelId: string } {
  return {
    type: config.type,
    baseUrl: config.baseUrl,
    apiKeyEnv: config.apiKeyEnv,
    modelId: config.modelId
  };
}
