import { readFile } from "node:fs/promises";
import path from "node:path";
import { z } from "zod";
import type { ResolvedLlmConfig, ResolvedAgentSdkConfig } from "./llmConfig.js";
import { resolveLlmConfig, resolveAgentSdkConfig } from "./llmConfig.js";

const llmConfigSchema = z.object({
  baseUrl: z.url(),
  apiKeyEnv: z.string().min(1),
  modelId: z.string().min(1),
  maxTokens: z.number().int().positive().optional(),
});

const agentSdkConfigSchema = z.object({
  type: z.enum(["claude", "codebuddy", "pi"]),
  baseUrl: z.url(),
  apiKeyEnv: z.string().min(1),
  modelId: z.string().min(1),
  agentDir: z.string().min(1).optional(),
  provider: z.string().min(1).optional(),
  api: z.string().min(1).optional(),
  contextWindow: z.number().int().positive().optional(),
  reasoning: z.boolean().optional(),
  input: z.array(z.enum(["text", "image"])).optional(),
});

const contextConfigSchema = z.object({
  autoCompactEnabled: z.boolean().default(true),
  autoCompactWindow: z.number().int().positive().default(150_000),
  workspaceMaxChars: z.number().int().positive().default(8_000),
  historyMaxMessages: z.number().int().positive().default(20),
}).default(() => ({
  autoCompactEnabled: true,
  autoCompactWindow: 150_000,
  workspaceMaxChars: 8_000,
  historyMaxMessages: 20,
}));

const runtimeConfigSchema = z.object({
  port: z.number().int().positive().default(3000),
  host: z.string().min(1).default("127.0.0.1"),
  knowledgeBasePath: z.string().min(1).default(".harness/knowledge-base"),
  logDir: z.string().min(1).default(".harness/logs"),
  dbPath: z.string().min(1).default(".harness/state/agent.db"),
  sessionStorePath: z.string().min(1).default(".harness/state/sessions.json"),
  historyStorePath: z.string().min(1).default(".harness/state/history.json"),
  enableDevRoutes: z.boolean().default(false),
  wechat: z.object({
    botId: z.string().min(1).default("dev-bot-id"),
    secret: z.string().min(1).default("dev-secret"),
    botIdEnv: z.string().min(1).optional(),
    secretEnv: z.string().min(1).optional(),
    wsUrl: z.string().min(1).optional(),
    reconnectInterval: z.number().int().positive().optional(),
    maxReconnectAttempts: z.number().int().optional(),
  }).default({ botId: "dev-bot-id", secret: "dev-secret" }),
  llm: llmConfigSchema,
  agentSdk: agentSdkConfigSchema.optional(),
  context: contextConfigSchema,
});

export type WechatConfig = {
  readonly botId: string;
  readonly secret: string;
  readonly wsUrl?: string;
  readonly reconnectInterval?: number;
  readonly maxReconnectAttempts?: number;
};

export type ContextConfig = {
  readonly autoCompactEnabled: boolean;
  readonly autoCompactWindow: number;
  readonly workspaceMaxChars: number;
  readonly historyMaxMessages: number;
};

export type RuntimeConfig = {
  readonly port: number;
  readonly host: string;
  readonly knowledgeBasePath: string;
  readonly logDir: string;
  readonly dbPath: string;
  readonly sessionStorePath: string;
  readonly historyStorePath: string;
  readonly enableDevRoutes: boolean;
  readonly wechat: WechatConfig;
  readonly llm: ResolvedLlmConfig;
  readonly agentSdk: ResolvedAgentSdkConfig;
  readonly context: ContextConfig;
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
  const resolvedAgentSdk = resolveAgentSdkConfig(
    parsed.data.agentSdk ?? { type: "claude", baseUrl: parsed.data.llm.baseUrl, apiKeyEnv: parsed.data.llm.apiKeyEnv, modelId: parsed.data.llm.modelId },
    env,
  );
  const wechat: WechatConfig = {
    botId: resolveEnvOrDirect(parsed.data.wechat.botId, parsed.data.wechat.botIdEnv, env),
    secret: resolveEnvOrDirect(parsed.data.wechat.secret, parsed.data.wechat.secretEnv, env),
    ...(parsed.data.wechat.wsUrl !== undefined ? { wsUrl: parsed.data.wechat.wsUrl } : {}),
    ...(parsed.data.wechat.reconnectInterval !== undefined ? { reconnectInterval: parsed.data.wechat.reconnectInterval } : {}),
    ...(parsed.data.wechat.maxReconnectAttempts !== undefined ? { maxReconnectAttempts: parsed.data.wechat.maxReconnectAttempts } : {}),
  };
  return { ...parsed.data, llm: resolvedLlm, agentSdk: resolvedAgentSdk, wechat };
}

function resolveEnvOrDirect(directValue: string, envName: string | undefined, env: NodeJS.ProcessEnv): string {
  if (envName !== undefined) {
    const envValue = env[envName];
    if (envValue !== undefined && envValue.length > 0) {
      return envValue;
    }
  }
  return directValue;
}
