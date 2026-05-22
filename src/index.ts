import path from "node:path";
import "dotenv/config";
import { ClaudeSdkAgentRuntime, REVIEWER_CONFIG, DEVELOPER_CONFIG } from "./agent/claudeSdkAgentRuntime.js";
import { EchoAgentRuntime } from "./agent/echoAgentRuntime.js";
import { loadLlmConfig, resolveLlmConfig, summarizeLlmConfig } from "./config/llmConfig.js";
import { FileConversationHistoryStore } from "./core/fileConversationHistoryStore.js";
import { FileConversationSessionStore } from "./core/fileConversationSessionStore.js";
import { AgentOrchestrator } from "./core/orchestrator.js";
import type { AgentRuntime } from "./core/ports.js";
import { createSqliteConnection } from "./db/sqliteConnection.js";
import { SqliteFeedbackStore } from "./db/sqliteFeedbackStore.js";
import { SqliteRoleConfigStore } from "./db/sqliteRoleConfigStore.js";
import { seedDefaultRoles } from "./db/seedRoles.js";
import { LlmIntentDetectionService } from "./intent/llmIntentDetectionService.js";
import { createJsonlLogger, type JsonlLogger } from "./logging/jsonlLogger.js";
import { StaticWorkspaceResolver } from "./repos/staticWorkspaceResolver.js";
import { createServer } from "./server/createServer.js";
import { DevProgressBroker } from "./server/progressBroker.js";
import { createDevRoleStore } from "./security/devRoleStore.js";
import { StaticAuthorizationService } from "./security/staticAuthorizationService.js";

export async function main(): Promise<void> {
  const port = Number.parseInt(process.env["PORT"] ?? "3000", 10);
  const knowledgeBasePath = process.env["KNOWLEDGE_BASE_PATH"] ?? ".harness/knowledge-base";
  const wechatToken = process.env["WECHAT_TOKEN"] ?? "dev-token";
  const logDir = process.env["LOG_DIR"] ?? path.resolve(".harness", "logs");
  const sessionStorePath = process.env["SESSION_STORE_PATH"] ?? path.resolve(".harness", "state", "sessions.json");
  const historyStorePath = process.env["HISTORY_STORE_PATH"] ?? path.resolve(".harness", "state", "history.json");
  const dbPath = process.env["DB_PATH"] ?? path.resolve(".harness", "state", "agent.db");
  const conversationLogger = createJsonlLogger(path.join(logDir, "conversation.jsonl"));
  const llmRawLogger = createJsonlLogger(path.join(logDir, "llm-raw.jsonl"));
  const devRoleStore = createDevRoleStore();
  const progressBroker = new DevProgressBroker();

  // Initialize SQLite and stores
  const db = createSqliteConnection(dbPath);
  const roleConfigStore = new SqliteRoleConfigStore(db);
  const feedbackStore = new SqliteFeedbackStore(db);

  // Seed default roles if table is empty
  await seedDefaultRoles(roleConfigStore);

  await configureSdkEnvironment();

  const resolvedLlmConfig = await loadResolvedLlmConfig();
  const agentRuntimes = await createAgentRuntimes(llmRawLogger, roleConfigStore, resolvedLlmConfig);

  // Build intent detection service
  const intentDetection = resolvedLlmConfig !== undefined
    ? new LlmIntentDetectionService(resolvedLlmConfig)
    : undefined;

  const orchestrator = new AgentOrchestrator({
    workspaceResolver: new StaticWorkspaceResolver({ knowledgeBasePath }),
    authorization: new StaticAuthorizationService(devRoleStore, roleConfigStore),
    agentRuntimes,
    sessionStore: new FileConversationSessionStore(sessionStorePath),
    historyStore: new FileConversationHistoryStore(historyStorePath),
    conversationLogger,
    progressReporter: progressBroker,
    intentDetection,
    feedbackStore,
  });
  const server = createServer({
    messageHandler: orchestrator,
    wechatToken,
    devRoleStore,
    roleConfigStore,
    progressBroker,
    logger: true
  });

  await server.listen({ port, host: "0.0.0.0" });
  console.info(`pets-agent listening on http://localhost:${port}`);
  console.info(`reviewer runtime: ${agentRuntimes["reviewer"]?.name ?? "not configured"}`);
  console.info(`developer runtime: ${agentRuntimes["developer"]?.name ?? "not configured"}`);
  console.info(`conversation log: ${conversationLogger.filePath}`);
  console.info(`llm raw log: ${llmRawLogger.filePath}`);
  console.info(`database: ${dbPath}`);
}

async function configureSdkEnvironment(): Promise<void> {
  try {
    const llmConfigPath = process.env["LLM_CONFIG_PATH"] ?? path.resolve("config", "llm.json");
    const llmConfig = await loadLlmConfig(llmConfigPath);
    const resolved = resolveLlmConfig(llmConfig);

    process.env["ANTHROPIC_API_KEY"] ??= resolved.apiKey;
    process.env["ANTHROPIC_BASE_URL"] ??= resolved.baseUrl;

    const summary = summarizeLlmConfig(resolved);
    console.info(`SDK configured: ${summary.modelId} at ${summary.baseUrl}`);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.warn(`SDK env configuration skipped: ${message}`);
  }
}

async function loadResolvedLlmConfig(): Promise<ReturnType<typeof resolveLlmConfig> | undefined> {
  try {
    const llmConfigPath = process.env["LLM_CONFIG_PATH"] ?? path.resolve("config", "llm.json");
    const llmConfig = await loadLlmConfig(llmConfigPath);
    return resolveLlmConfig(llmConfig);
  } catch {
    return undefined;
  }
}

async function createAgentRuntimes(
  llmRawLogger: JsonlLogger,
  roleConfigStore: SqliteRoleConfigStore,
  resolvedLlmConfig: ReturnType<typeof resolveLlmConfig> | undefined,
): Promise<Record<string, AgentRuntime>> {
  // Read role configs from DB
  const roleConfigs = await roleConfigStore.getAll();

  if (roleConfigs.length > 0 && resolvedLlmConfig !== undefined) {
    const runtimes: Record<string, AgentRuntime> = {};
    for (const config of roleConfigs) {
      runtimes[config.name] = new ClaudeSdkAgentRuntime({
        roleConfig: config,
        rawLogger: llmRawLogger,
        model: config.model ?? resolvedLlmConfig.modelId,
      });
    }
    return runtimes;
  }

  // Fallback: use hardcoded defaults
  if (resolvedLlmConfig !== undefined) {
    return {
      reviewer: new ClaudeSdkAgentRuntime({
        roleConfig: REVIEWER_CONFIG,
        rawLogger: llmRawLogger,
        model: resolvedLlmConfig.modelId,
      }),
      developer: new ClaudeSdkAgentRuntime({
        roleConfig: DEVELOPER_CONFIG,
        rawLogger: llmRawLogger,
        model: resolvedLlmConfig.modelId,
      }),
    };
  }

  console.warn("using echo runtime because real LLM runtime is not configured");
  return {
    reviewer: new EchoAgentRuntime(),
    developer: new EchoAgentRuntime(),
  };
}

await main();
