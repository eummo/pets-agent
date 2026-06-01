import { readFile } from "node:fs/promises";
import path from "node:path";
import { z } from "zod";
import type { ResolvedLlmConfig, ResolvedAgentSdkConfig } from "./llmConfig.js";
import { resolveLlmConfig, resolveActiveAgentSdk } from "./llmConfig.js";

const llmConfigSchema = z.object({
  baseUrl: z.url(),
  apiKeyEnv: z.string().min(1),
  modelId: z.string().min(1),
  maxTokens: z.number().int().positive().optional()
});

const agentSdkEntrySchema = z.object({
  baseUrl: z.url(),
  apiKeyEnv: z.string().min(1).optional(),
  modelId: z.string().min(1),
  endpoint: z.url().optional(),
  endpointEnv: z.string().min(1).optional(),
  environment: z.enum(["external", "internal", "ioa", "cloudhosted"]).optional(),
  agentDir: z.string().min(1).optional(),
  provider: z.string().min(1).optional(),
  api: z.string().min(1).optional(),
  contextWindow: z.number().int().positive().optional(),
  reasoning: z.boolean().optional(),
  input: z.array(z.enum(["text", "image"])).optional()
});

const agentSdkTypeSchema = z.enum(["claude", "codebuddy", "pi"]);

const agentSdksSchema = z.object({
  claude: agentSdkEntrySchema.optional(),
  codebuddy: agentSdkEntrySchema.optional(),
  pi: agentSdkEntrySchema.optional()
});

const contextConfigSchema = z
  .object({
    autoCompactEnabled: z.boolean().default(true),
    autoCompactWindow: z.number().int().positive().default(150_000),
    workspaceMaxChars: z.number().int().positive().default(8_000),
    historyMaxMessages: z.number().int().positive().default(20)
  })
  .default(() => ({
    autoCompactEnabled: true,
    autoCompactWindow: 150_000,
    workspaceMaxChars: 8_000,
    historyMaxMessages: 20
  }));

const cronWecomConfigSchema = z.object({
  corpId: z.string().min(1),
  corpSecretEnv: z.string().min(1),
  agentId: z.string().min(1),
  tokenCacheMs: z.number().int().positive().default(7_200_000)
});

const cronConfigSchema = z
  .object({
    enabled: z.boolean().default(false),
    tickIntervalMs: z.number().int().positive().default(60_000),
    staleGraceMs: z.number().int().positive().default(300_000),
    jobStorePath: z.string().min(1).default(".harness/state/cron-jobs.json"),
    wecom: cronWecomConfigSchema.optional()
  })
  .default(() => ({
    enabled: false,
    tickIntervalMs: 60_000,
    staleGraceMs: 300_000,
    jobStorePath: ".harness/state/cron-jobs.json"
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
  wechat: z
    .object({
      botId: z.string().min(1).default("dev-bot-id"),
      secret: z.string().min(1).default("dev-secret"),
      botIdEnv: z.string().min(1).optional(),
      secretEnv: z.string().min(1).optional(),
      wsUrl: z.string().min(1).optional(),
      reconnectInterval: z.number().int().positive().optional(),
      maxReconnectAttempts: z.number().int().optional()
    })
    .default({ botId: "dev-bot-id", secret: "dev-secret" }),
  llm: llmConfigSchema,
  agentSdkType: agentSdkTypeSchema,
  agentSdks: agentSdksSchema,
  context: contextConfigSchema,
  cron: cronConfigSchema
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

export const DEFAULT_CONTEXT_CONFIG: ContextConfig = {
  autoCompactEnabled: true,
  autoCompactWindow: 150_000,
  workspaceMaxChars: 8_000,
  historyMaxMessages: 20
};

export type CronWecomConfig = {
  readonly corpId: string;
  readonly corpSecret: string;
  readonly agentId: string;
  readonly tokenCacheMs: number;
};

export type CronConfig = {
  readonly enabled: boolean;
  readonly tickIntervalMs: number;
  readonly staleGraceMs: number;
  readonly jobStorePath: string;
  readonly wecom?: CronWecomConfig;
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
  readonly cron: CronConfig;
};

export async function loadRuntimeConfig(
  configPath = process.env["CONFIG_PATH"] ?? path.resolve("config", "runtime.json"),
  env: NodeJS.ProcessEnv = process.env
): Promise<RuntimeConfig> {
  let raw: string;
  try {
    raw = await readFile(configPath, "utf8");
  } catch (error) {
    throw new Error(
      `Failed to read config file at ${configPath}: ${error instanceof Error ? error.message : String(error)}`,
      { cause: error }
    );
  }

  let json: unknown;
  try {
    json = JSON.parse(raw);
  } catch (error) {
    throw new Error(
      `Invalid JSON in config file at ${configPath}: ${error instanceof Error ? error.message : String(error)}`,
      { cause: error }
    );
  }

  const parsed = runtimeConfigSchema.safeParse(json);
  if (!parsed.success) {
    const issues = parsed.error.issues
      .map((i) => `  - ${i.path.join(".")}: ${i.message}`)
      .join("\n");
    throw new Error(`Invalid config in ${configPath}:\n${issues}`);
  }

  const resolvedLlm = resolveLlmConfig(parsed.data.llm, env);
  const resolvedAgentSdk = resolveActiveAgentSdk(
    parsed.data.agentSdkType,
    parsed.data.agentSdks,
    env
  );
  const wechat: WechatConfig = {
    botId: resolveEnvOrDirect(parsed.data.wechat.botId, parsed.data.wechat.botIdEnv, env),
    secret: resolveEnvOrDirect(parsed.data.wechat.secret, parsed.data.wechat.secretEnv, env),
    ...(parsed.data.wechat.wsUrl !== undefined ? { wsUrl: parsed.data.wechat.wsUrl } : {}),
    ...(parsed.data.wechat.reconnectInterval !== undefined
      ? { reconnectInterval: parsed.data.wechat.reconnectInterval }
      : {}),
    ...(parsed.data.wechat.maxReconnectAttempts !== undefined
      ? { maxReconnectAttempts: parsed.data.wechat.maxReconnectAttempts }
      : {})
  };
  const cron: CronConfig = {
    enabled: parsed.data.cron.enabled,
    tickIntervalMs: parsed.data.cron.tickIntervalMs,
    staleGraceMs: parsed.data.cron.staleGraceMs,
    jobStorePath: parsed.data.cron.jobStorePath,
    ...(parsed.data.cron.wecom !== undefined
      ? {
          wecom: {
            corpId: parsed.data.cron.wecom.corpId,
            corpSecret: resolveEnvOrDirect("", parsed.data.cron.wecom.corpSecretEnv, env),
            agentId: parsed.data.cron.wecom.agentId,
            tokenCacheMs: parsed.data.cron.wecom.tokenCacheMs
          }
        }
      : {})
  };
  return { ...parsed.data, llm: resolvedLlm, agentSdk: resolvedAgentSdk, wechat, cron };
}

function resolveEnvOrDirect(
  directValue: string,
  envName: string | undefined,
  env: NodeJS.ProcessEnv
): string {
  if (envName !== undefined) {
    const envValue = env[envName];
    if (envValue !== undefined && envValue.length > 0) {
      return envValue;
    }
  }
  return directValue;
}
