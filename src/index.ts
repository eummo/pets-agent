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
 * - WeChat adapter: WebSocket long connection to Enterprise WeChat smart bot
 */
import path from "node:path";
import "dotenv/config";
import { setupAgentRuntimes } from "./agent/createAgentRuntimes.js";
import { buildPiModel, summarizeLlmConfig } from "./config/llmConfig.js";
import { loadRuntimeConfig } from "./config/runtimeConfig.js";
import { FileConversationHistoryStore } from "./persistence/fileConversationHistoryStore.js";
import { FileConversationSessionStore } from "./persistence/fileConversationSessionStore.js";
import { AgentOrchestrator } from "./core/orchestrator.js";
import { createSqliteConnection } from "./persistence/sqliteConnection.js";
import { SqliteFeedbackStore } from "./persistence/sqliteFeedbackStore.js";
import { SqliteRoleConfigStore } from "./persistence/sqliteRoleConfigStore.js";
import { seedDefaultRoles } from "./persistence/seedRoles.js";
import { LlmIntentDetectionService } from "./intent/llmIntentDetectionService.js";
import { createJsonlLogger } from "./logging/jsonlLogger.js";
import { ConfiguredWorkspaceResolver } from "./workspace/configuredWorkspaceResolver.js";
import { createServer } from "./server/createServer.js";
import { SseProgressBroker } from "./server/sseProgressBroker.js";
import { InMemoryRoleAuthorizationService } from "./auth/inMemoryRoleAuthorizationService.js";
import { WechatSmartBotAdapter } from "./wechat/wechatSmartBotAdapter.js";

export async function main(): Promise<void> {
  const config = await loadRuntimeConfig();
  const conversationLogger = createJsonlLogger(path.join(config.logDir, "conversation.jsonl"));
  const llmRawLogger = createJsonlLogger(path.join(config.logDir, "llm-raw.jsonl"));
  const systemLogger = createJsonlLogger(path.join(config.logDir, "system.jsonl"));
  const progressBroker = new SseProgressBroker();

  // Initialize SQLite and stores
  const db = createSqliteConnection(config.dbPath);
  const roleConfigStore = new SqliteRoleConfigStore(db);
  const feedbackStore = new SqliteFeedbackStore(db);

  // Seed default roles if table is empty
  await seedDefaultRoles(roleConfigStore);

  // Configure Anthropic SDK env vars from resolved LLM config
  process.env["ANTHROPIC_API_KEY"] ??= config.llm.apiKey;
  process.env["ANTHROPIC_BASE_URL"] ??= config.llm.baseUrl;
  const summary = summarizeLlmConfig(config.llm);
  console.info(`SDK configured: ${summary.modelId} at ${summary.baseUrl}`);

  const { agentRuntimes, runtimeFactory } = await setupAgentRuntimes(llmRawLogger, roleConfigStore, config.llm);

  const intentDetection = new LlmIntentDetectionService(buildPiModel(config.llm), config.llm.apiKey);

  const authorization = new InMemoryRoleAuthorizationService(roleConfigStore);

  const orchestrator = new AgentOrchestrator({
    workspaceResolver: new ConfiguredWorkspaceResolver({
      knowledgeBasePath: config.knowledgeBasePath,
      logger: systemLogger,
    }),
    authorization,
    agentRuntimes,
    runtimeFactory,
    sessionStore: new FileConversationSessionStore(config.sessionStorePath),
    historyStore: new FileConversationHistoryStore(config.historyStorePath),
    conversationLogger,
    eventLogger: systemLogger,
    progressReporter: progressBroker,
    intentDetection,
    feedbackStore,
  });

  // Start HTTP server for dev browser and health checks
  const server = createServer({
    messageHandler: orchestrator,
    roleConfigStore,
    feedbackStore,
    authorization,
    progressBroker,
    enableDevRoutes: config.enableDevRoutes,
    logger: true
  });

  await server.listen({ port: config.port, host: config.host });
  console.info(`pets-agent listening on http://${config.host}:${config.port}`);
  console.info(`reviewer runtime: ${agentRuntimes["reviewer"]?.name ?? "not configured"}`);
  console.info(`developer runtime: ${agentRuntimes["developer"]?.name ?? "not configured"}`);
  console.info(`admin runtime: ${agentRuntimes["admin"]?.name ?? "not configured"}`);
  console.info(`conversation log: ${conversationLogger.filePath}`);
  console.info(`llm raw log: ${llmRawLogger.filePath}`);
  console.info(`database: ${config.dbPath}`);

  // Start WeChat smart bot adapter (WebSocket long connection)
  const wechatAdapter = new WechatSmartBotAdapter({
    botId: config.wechat.botId,
    secret: config.wechat.secret,
    messageHandler: orchestrator,
    conversationLogger,
    eventLogger: systemLogger,
    ...(config.wechat.wsUrl !== undefined ? { wsUrl: config.wechat.wsUrl } : {}),
    ...(config.wechat.reconnectInterval !== undefined ? { reconnectInterval: config.wechat.reconnectInterval } : {}),
    ...(config.wechat.maxReconnectAttempts !== undefined ? { maxReconnectAttempts: config.wechat.maxReconnectAttempts } : {}),
  });
  wechatAdapter.connect();
  console.info(`WeChat smart bot connected: botId=${config.wechat.botId}`);

  // Graceful shutdown
  const shutdown = (): void => {
    console.info("Shutting down...");
    wechatAdapter.disconnect();
    void server.close();
    process.exit(0);
  };
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}

await main();
