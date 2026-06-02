import type { Api, Model } from "@earendil-works/pi-ai";

// ── Defaults for agent SDK model construction ────────────────────────────────

export const AGENT_SDK_DEFAULTS = {
  api: "anthropic-messages",
  provider: "pets-agent",
  contextWindow: 200000,
  maxTokens: 8192,
  reasoning: false,
  input: ["text"] as readonly ("text" | "image")[]
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

// ── API Key Resolution ───────────────────────────────────────────────────────

function resolveApiKey(
  config: { readonly apiKeyEnv: string },
  errorLabel: string,
  env: NodeJS.ProcessEnv
): string {
  const apiKey = env[config.apiKeyEnv];

  if (apiKey === undefined || apiKey.trim().length === 0) {
    throw new Error(`Missing ${errorLabel} API key environment variable: ${config.apiKeyEnv}`);
  }

  return apiKey;
}

export function resolveLlmConfig(
  config: LlmConfig,
  env: NodeJS.ProcessEnv = process.env
): ResolvedLlmConfig {
  return { ...config, apiKey: resolveApiKey(config, "LLM", env) };
}

export function summarizeLlmConfig(
  config: LlmConfig
): Pick<LlmConfig, "baseUrl" | "apiKeyEnv" | "modelId"> {
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
    maxTokens: config.maxTokens ?? AGENT_SDK_DEFAULTS.maxTokens
  };
}

// ── Agent SDK Configuration ─────────────────────────────────────────────────
// agentSdks maps each SDK type to its connection config.
// agentSdkType selects which one is active.

export type AgentSdkType = "claude" | "codebuddy" | "pi";
export type CodebuddyAuthEnvironment = "external" | "internal" | "ioa" | "cloudhosted";

export type AgentSdkEntry = {
  readonly baseUrl: string;
  readonly apiKeyEnv?: string | undefined;
  readonly modelId: string;
  readonly endpoint?: string | undefined;
  readonly endpointEnv?: string | undefined;
  readonly environment?: CodebuddyAuthEnvironment | undefined;
  readonly agentDir?: string | undefined;
  readonly provider?: string | undefined;
  readonly api?: string | undefined;
  readonly contextWindow?: number | undefined;
  readonly reasoning?: boolean | undefined;
  readonly input?: readonly ("text" | "image")[] | undefined;
};

export type AgentSdksConfig = {
  readonly claude?: AgentSdkEntry | undefined;
  readonly codebuddy?: AgentSdkEntry | undefined;
  readonly pi?: AgentSdkEntry | undefined;
};

export type AgentSdkConfig = AgentSdkEntry & {
  readonly type: AgentSdkType;
};

export type ResolvedAgentSdkConfig = AgentSdkConfig & {
  readonly apiKey: string;
};

export function resolveActiveAgentSdk(
  agentSdkType: AgentSdkType,
  agentSdks: AgentSdksConfig,
  env: NodeJS.ProcessEnv = process.env
): ResolvedAgentSdkConfig {
  const entry = agentSdks[agentSdkType];
  if (entry === undefined) {
    throw new Error(
      `No agentSdk config found for type "${agentSdkType}". Available: ${Object.keys(agentSdks).join(", ") || "none"}`
    );
  }
  const endpoint = resolveOptionalEndpoint(entry, agentSdkType, env);
  const config: AgentSdkConfig = {
    ...entry,
    ...(endpoint !== undefined ? { endpoint } : {}),
    type: agentSdkType
  };
  const apiKey = resolveOptionalAgentSdkApiKey(config, env);
  return { ...config, apiKey };
}

function resolveOptionalEndpoint(
  config: AgentSdkEntry,
  type: AgentSdkType,
  env: NodeJS.ProcessEnv
): string | undefined {
  if (config.endpointEnv === undefined) return config.endpoint;

  const endpoint = env[config.endpointEnv];
  if (endpoint === undefined || endpoint.trim().length === 0) {
    if (type === "codebuddy") {
      return config.endpoint;
    }
    throw new Error(`Missing Agent SDK endpoint environment variable: ${config.endpointEnv}`);
  }

  return endpoint;
}

function resolveOptionalAgentSdkApiKey(config: AgentSdkConfig, env: NodeJS.ProcessEnv): string {
  if (config.apiKeyEnv !== undefined) {
    return resolveApiKey({ apiKeyEnv: config.apiKeyEnv }, `Agent SDK (${config.type})`, env);
  }

  if (config.type === "codebuddy") {
    return "";
  }

  throw new Error(`Missing Agent SDK (${config.type}) apiKeyEnv.`);
}

export function summarizeAgentSdkConfig(config: AgentSdkConfig): {
  readonly type: AgentSdkType;
  readonly baseUrl: string;
  readonly apiKeyEnv?: string | undefined;
  readonly modelId: string;
  readonly endpoint?: string | undefined;
  readonly endpointEnv?: string | undefined;
  readonly environment?: CodebuddyAuthEnvironment | undefined;
} {
  return {
    type: config.type,
    baseUrl: config.baseUrl,
    apiKeyEnv: config.apiKeyEnv,
    modelId: config.modelId,
    ...(config.endpoint !== undefined ? { endpoint: config.endpoint } : {}),
    ...(config.endpointEnv !== undefined ? { endpointEnv: config.endpointEnv } : {}),
    ...(config.environment !== undefined ? { environment: config.environment } : {})
  };
}
