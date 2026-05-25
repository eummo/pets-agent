/**
 * pets-agent - Main entry point for the pets-agent server.
 * Initializes and starts the agent orchestrator with all required services.
 *
 * Architecture overview:
 * - AgentOrchestrator: Coordinates agent interactions, workspace resolution, and authorization
 * - Agent runtimes: Execute agent logic (Claude SDK, Echo for dev, or custom implementations)
 * - Intent detection: Routes incoming requests to appropriate intents using LLM
 * - Stores: Persist sessions, conversation history, role configs, and feedback
 * - Server: HTTP/WebSocket server handling incoming messages and progress updates
 */
import path from "node:path";
import "dotenv/config";
import { createAgentRuntimeFactory, createAgentRuntimes } from "./agent/createAgentRuntimes.js";
import { buildPiModel, loadLlmConfig, resolveLlmConfig, summarizeLlmConfig, type ResolvedLlmConfig } from "./config/llmConfig.js";
import { loadRuntimeConfig } from "./config/runtimeConfig.js";
import { FileConversationHistoryStore } from "./db/fileConversationHistoryStore.js";
import { FileConversationSessionStore } from "./db/fileConversationSessionStore.js";
import { AgentOrchestrator } from "./core/orchestrator.js";
import { createSqliteConnection } from "./db/sqliteConnection.js";
import { SqliteFeedbackStore } from "./db/sqliteFeedbackStore.js";
import { SqliteRoleConfigStore } from "./db/sqliteRoleConfigStore.js";
import { seedDefaultRoles } from "./db/seedRoles.js";
import { LlmIntentDetectionService } from "./intent/llmIntentDetectionService.js";
import { createJsonlLogger } from "./logging/jsonlLogger.js";
import { StaticWorkspaceResolver } from "./repos/staticWorkspaceResolver.js";
import { createServer } from "./server/createServer.js";
import { DevProgressBroker } from "./server/progressBroker.js";
import { StaticAuthorizationService } from "./security/staticAuthorizationService.js";

export async function main(): Promise<void> {
  // Load runtime configuration, create loggers, and initialize all services
  const runtimeConfig = loadRuntimeConfig();
  const conversationLogger = createJsonlLogger(path.join(runtimeConfig.logDir, "conversation.jsonl"));
  const llmRawLogger = createJsonlLogger(path.join(runtimeConfig.logDir, "llm-raw.jsonl"));
  const systemLogger = createJsonlLogger(path.join(runtimeConfig.logDir, "system.jsonl"));
  const progressBroker = new DevProgressBroker();

  // Initialize SQLite and stores
  const db = createSqliteConnection(runtimeConfig.dbPath);
  const roleConfigStore = new SqliteRoleConfigStore(db);
  const feedbackStore = new SqliteFeedbackStore(db);

  // Seed default roles if table is empty
  await seedDefaultRoles(roleConfigStore);

  const resolvedLlmConfig = await loadAndApplyLlmConfig();
  const agentRuntimes = await createAgentRuntimes(llmRawLogger, roleConfigStore, resolvedLlmConfig);

  const intentDetection = new LlmIntentDetectionService(buildPiModel(resolvedLlmConfig), resolvedLlmConfig.apiKey);

  const authorization = new StaticAuthorizationService(roleConfigStore);

  const runtimeFactory = createAgentRuntimeFactory(llmRawLogger, roleConfigStore, resolvedLlmConfig);

  const orchestrator = new AgentOrchestrator({
    workspaceResolver: new StaticWorkspaceResolver({
      knowledgeBasePath: runtimeConfig.knowledgeBasePath,
      logger: systemLogger,
    }),
    authorization,
    agentRuntimes,
    runtimeFactory,
    sessionStore: new FileConversationSessionStore(runtimeConfig.sessionStorePath),
    historyStore: new FileConversationHistoryStore(runtimeConfig.historyStorePath),
    conversationLogger,
    eventLogger: systemLogger,
    progressReporter: progressBroker,
    intentDetection,
    feedbackStore,
  });
  const server = createServer({
    messageHandler: orchestrator,
    wechatToken: runtimeConfig.wechatToken,
    roleConfigStore,
    feedbackStore,
    authorization,
    progressBroker,
    enableDevRoutes: runtimeConfig.enableDevRoutes,
    logger: true
  });

  await server.listen({ port: runtimeConfig.port, host: "0.0.0.0" });
  console.info(`pets-agent listening on http://localhost:${runtimeConfig.port}`);
  console.info(`reviewer runtime: ${agentRuntimes["reviewer"]?.name ?? "not configured"}`);
  console.info(`developer runtime: ${agentRuntimes["developer"]?.name ?? "not configured"}`);
  console.info(`admin runtime: ${agentRuntimes["admin"]?.name ?? "not configured"}`);
  console.info(`conversation log: ${conversationLogger.filePath}`);
  console.info(`llm raw log: ${llmRawLogger.filePath}`);
  console.info(`database: ${runtimeConfig.dbPath}`);
}

async function loadAndApplyLlmConfig(): Promise<ResolvedLlmConfig> {
  const llmConfigPath = process.env["LLM_CONFIG_PATH"] ?? path.resolve("config", "llm.json");
  const llmConfig = await loadLlmConfig(llmConfigPath);
  const resolved = resolveLlmConfig(llmConfig);

  process.env["ANTHROPIC_API_KEY"] ??= resolved.apiKey;
  process.env["ANTHROPIC_BASE_URL"] ??= resolved.baseUrl;

  const summary = summarizeLlmConfig(resolved);
  console.info(`SDK configured: ${summary.modelId} at ${summary.baseUrl}`);

  return resolved;
}

await main();
