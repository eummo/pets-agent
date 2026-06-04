/**
 * pets-agent - Main entry point for the pets-agent server.
 * Initializes and starts the agent orchestrator with all required services.
 *
 * Architecture overview:
 * - AgentOrchestrator: Coordinates agent interactions, workspace resolution, and authorization
 * - Agent runtimes: Execute agent logic (Claude SDK, Echo for dev, or custom implementations)
 * - Intent detection: Classifies incoming requests before runtime execution
 * - Stores: Persist sessions, conversation history, role configs, and feedback
 * - Server: HTTP/WebSocket server handling incoming messages and progress updates
 * - WeChat adapter: WebSocket long connection to Enterprise WeChat smart bot
 */
import path from "node:path";
import "dotenv/config";
import { setupAgentRuntimes } from "./agent/createAgentRuntimes.js";
import { summarizeLlmConfig, summarizeAgentSdkConfig } from "./config/llmConfig.js";
import { loadRuntimeConfig } from "./config/runtimeConfig.js";
import { FileConversationHistoryStore } from "./persistence/fileConversationHistoryStore.js";
import { FileConversationSessionStore } from "./persistence/fileConversationSessionStore.js";
import { AgentOrchestrator } from "./core/orchestrator.js";
import { createSqliteConnection } from "./persistence/sqliteConnection.js";
import { SqliteFeedbackStore } from "./persistence/sqliteFeedbackStore.js";
import { SqliteRoleConfigStore } from "./persistence/sqliteRoleConfigStore.js";
import { seedDefaultRoles } from "./persistence/seedRoles.js";
import { createJsonlLogger } from "./logging/jsonlLogger.js";
import { ConfiguredWorkspaceResolver } from "./workspace/configuredWorkspaceResolver.js";
import { createServer } from "./server/createServer.js";
import { SseProgressBroker } from "./server/sseProgressBroker.js";
import { InMemoryRoleAuthorizationService } from "./auth/inMemoryRoleAuthorizationService.js";
import { WechatSmartBotAdapter } from "./wechat/wechatSmartBotAdapter.js";
import { FileCronJobStore } from "./cron/cronJobStore.js";
import { TickCronScheduler } from "./cron/cronScheduler.js";
import { CompositeDeliveryChannel } from "./cron/delivery/compositeDelivery.js";
import { SseDeliveryChannel } from "./cron/delivery/sseDelivery.js";
import { WecomAppMessageDeliveryChannel } from "./cron/delivery/wecomAppMessageDelivery.js";
import { WecomBotDeliveryChannel } from "./cron/delivery/wecomBotDelivery.js";
import { WebhookDeliveryChannel } from "./cron/delivery/webhookDelivery.js";
import { registerCronRoutes } from "./cron/cronRoutes.js";
import { LlmCronParseService } from "./cron/cronParseService.js";
import { LlmIntentDetectionService } from "./intent/llmIntentDetectionService.js";
import { buildPiModel } from "./intent/piModel.js";

// Bootstraps the pets-agent server: loads runtime config, wires up persistence,
// intent detection, agent runtimes, the HTTP/WebSocket server, the WeChat smart
// bot adapter, and the optional cron scheduler, then blocks on a graceful
// shutdown handler for SIGINT/SIGTERM.
//
// Note: `main` is the single entry point executed by `node dist/index.js`
// (or `tsx src/index.ts` in dev) and is also re-exported for tests.
export async function main(): Promise<void> {
  const config = await loadRuntimeConfig();
  const conversationLogger = createJsonlLogger(path.join(config.logDir, "conversation.jsonl"));
  const llmRawLogger = createJsonlLogger(path.join(config.logDir, "llm-raw.jsonl"));
  const systemLogger = createJsonlLogger(path.join(config.logDir, "system.jsonl"));
  const progressBroker = new SseProgressBroker();

  const db = createSqliteConnection(config.dbPath);
  const roleConfigStore = new SqliteRoleConfigStore(db);
  const feedbackStore = new SqliteFeedbackStore(db);

  await seedDefaultRoles(roleConfigStore);

  if (config.agentSdk.type === "claude") {
    process.env["ANTHROPIC_API_KEY"] ??= config.agentSdk.apiKey;
    process.env["ANTHROPIC_BASE_URL"] ??= config.agentSdk.baseUrl;
  }
  if (config.agentSdk.type === "codebuddy") {
    if (config.agentSdk.environment !== undefined) {
      process.env["CODEBUDDY_INTERNET_ENVIRONMENT"] ??= config.agentSdk.environment;
    }
  }
  const sdkSummary = summarizeAgentSdkConfig(config.agentSdk);
  const llmSummary = summarizeLlmConfig(config.llm);
  console.info(`Agent SDK: ${sdkSummary.type} ${sdkSummary.modelId} at ${sdkSummary.baseUrl}`);
  console.info(`Intent LLM: ${llmSummary.modelId} at ${llmSummary.baseUrl}`);

  const intentDetection = new LlmIntentDetectionService(
    buildPiModel(config.llm),
    config.llm.apiKey,
    llmRawLogger
  );
  const runtimeFactory = setupAgentRuntimes(
    llmRawLogger,
    roleConfigStore,
    config.llm,
    config.agentSdk,
    config.context,
    intentDetection.decideToolPermission
  );

  const agentRuntimes = await runtimeFactory.warmup();

  const authorization = new InMemoryRoleAuthorizationService(roleConfigStore);

  const orchestrator = new AgentOrchestrator({
    workspaceResolver: new ConfiguredWorkspaceResolver({
      knowledgeBasePath: config.knowledgeBasePath,
      logger: systemLogger
    }),
    authorization,
    runtimeFactory,
    initialRuntimes: agentRuntimes,
    sessionStore: new FileConversationSessionStore(config.sessionStorePath),
    historyStore: new FileConversationHistoryStore(config.historyStorePath, {
      maxMessages: config.context.historyMaxMessages
    }),
    conversationLogger,
    eventLogger: systemLogger,
    progressReporter: progressBroker,
    feedbackStore,
    intentDetection
  });

  const server = createServer({
    messageHandler: orchestrator,
    roleConfigStore,
    feedbackStore,
    authorization,
    progressBroker,
    enableDevRoutes: config.enableDevRoutes,
    logger: true
  });

  // ── WeChat Smart Bot Adapter ─────────────────────────────────────────────
  const wechatAdapter = new WechatSmartBotAdapter({
    botId: config.wechat.botId,
    secret: config.wechat.secret,
    messageHandler: orchestrator,
    conversationLogger,
    eventLogger: systemLogger,
    ...(config.wechat.wsUrl !== undefined ? { wsUrl: config.wechat.wsUrl } : {}),
    ...(config.wechat.reconnectInterval !== undefined
      ? { reconnectInterval: config.wechat.reconnectInterval }
      : {}),
    ...(config.wechat.maxReconnectAttempts !== undefined
      ? { maxReconnectAttempts: config.wechat.maxReconnectAttempts }
      : {}),
    ...(config.wechat.uploadRootPath !== undefined
      ? { uploadRootPath: config.wechat.uploadRootPath }
      : {})
  });

  // ── Cron Scheduler ─────────────────────────────────────────────────────────
  let cronScheduler: TickCronScheduler | undefined;
  if (config.cron.enabled) {
    const cronJobStore = new FileCronJobStore(config.cron.jobStorePath);

    const wecomDelivery =
      config.cron.wecom !== undefined
        ? new WecomAppMessageDeliveryChannel(config.cron.wecom)
        : new WecomBotDeliveryChannel(wechatAdapter);
    const deliveryChannels: import("./cron/cronTypes.js").DeliveryChannel[] = [
      new SseDeliveryChannel(progressBroker),
      wecomDelivery,
      new WebhookDeliveryChannel()
    ];

    const compositeDelivery = new CompositeDeliveryChannel(deliveryChannels);

    cronScheduler = new TickCronScheduler({
      jobStore: cronJobStore,
      messageHandler: orchestrator,
      delivery: compositeDelivery,
      eventLogger: systemLogger,
      conversationLogger,
      tickIntervalMs: config.cron.tickIntervalMs,
      staleGraceMs: config.cron.staleGraceMs
    });

    if (config.enableDevRoutes) {
      const cronParseService = new LlmCronParseService(
        buildPiModel(config.llm),
        config.llm.apiKey,
        llmRawLogger
      );
      registerCronRoutes(server, {
        jobStore: cronJobStore,
        scheduler: cronScheduler,
        authorization,
        cronParseService
      });
    }
  }

  await server.listen({ port: config.port, host: config.host });
  console.info(`pets-agent listening on http://${config.host}:${config.port}`);
  console.info(`reviewer runtime: ${agentRuntimes["reviewer"]?.name ?? "not configured"}`);
  console.info(`developer runtime: ${agentRuntimes["developer"]?.name ?? "not configured"}`);
  console.info(`admin runtime: ${agentRuntimes["admin"]?.name ?? "not configured"}`);
  console.info(`conversation log: ${conversationLogger.filePath}`);
  console.info(`llm raw log: ${llmRawLogger.filePath}`);
  console.info(`database: ${config.dbPath}`);

  if (cronScheduler !== undefined) {
    cronScheduler.start();
    console.info(
      `cron scheduler: enabled (tick=${config.cron.tickIntervalMs}ms, grace=${config.cron.staleGraceMs}ms)`
    );
  }

  wechatAdapter.connect();
  console.info(`WeChat smart bot connected: botId=${config.wechat.botId}`);
  const shutdown = (): void => {
    console.info("Shutting down...");
    cronScheduler?.stop();
    wechatAdapter.disconnect();
    void server.close();
    process.exit(0);
  };
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}

await main();
