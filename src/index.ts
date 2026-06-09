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
import type Database from "better-sqlite3";
import { setupAgentRuntimes } from "./agent/createAgentRuntimes.js";
import { summarizeLlmConfig, summarizeAgentSdkConfig } from "./config/llmConfig.js";
import { loadRuntimeConfig } from "./config/runtimeConfig.js";
import { FileConversationHistoryStore } from "./persistence/fileConversationHistoryStore.js";
import { FileConversationSessionStore } from "./persistence/fileConversationSessionStore.js";
import {
  SqliteConversationHistoryStore,
  SqliteConversationSessionStore
} from "./persistence/sqliteConversationStores.js";
import { startConversationArchiveRetention } from "./persistence/conversationArchiveRetention.js";
import type { ConversationHistoryStore, ConversationSessionStore } from "./persistence/index.js";
import { AgentOrchestrator } from "./core/orchestrator.js";
import { createSqliteConnection } from "./persistence/sqliteConnection.js";
import { SqliteFeedbackStore } from "./persistence/sqliteFeedbackStore.js";
import { SqliteRoleConfigStore } from "./persistence/sqliteRoleConfigStore.js";
import { seedDefaultRoles } from "./persistence/seedRoles.js";
import { createJsonlLogger } from "./logging/jsonlLogger.js";
import { ConfiguredWorkspaceResolver } from "./workspace/configuredWorkspaceResolver.js";
import { createServer, type HealthCheck } from "./server/createServer.js";
import { SseProgressBroker } from "./server/sseProgressBroker.js";
import { InMemoryRoleAuthorizationService } from "./auth/inMemoryRoleAuthorizationService.js";
import { WechatSmartBotAdapter } from "./wechat/wechatSmartBotAdapter.js";
import { startWechatSessionMetricsLogger } from "./wechat/wechatSessionMetrics.js";
import { FileCronJobStore } from "./cron/cronJobStore.js";
import { FileCronLeaderLease } from "./cron/cronLeaderLease.js";
import { SqliteCronJobStore } from "./cron/sqliteCronJobStore.js";
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
import { formatStartupBanner, type StartupCronSummary } from "./server/startupBanner.js";

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
  const conversationStores = createConversationStores({
    backend: config.conversationStore,
    db,
    sessionStorePath: config.sessionStorePath,
    historyStorePath: config.historyStorePath,
    historyMaxMessages: config.context.historyMaxMessages
  });
  const stopConversationArchiveRetention =
    config.conversationStore === "sqlite"
      ? createConversationArchiveRetentionStopper({
          db,
          retentionDays: config.conversationArchiveRetentionDays,
          cleanupIntervalMs: config.conversationArchiveCleanupIntervalMs,
          logger: systemLogger
        })
      : () => undefined;

  const orchestrator = new AgentOrchestrator({
    workspaceResolver: new ConfiguredWorkspaceResolver({
      knowledgeBasePath: config.knowledgeBasePath,
      logger: systemLogger
    }),
    authorization,
    runtimeFactory,
    initialRuntimes: agentRuntimes,
    sessionStore: conversationStores.sessionStore,
    historyStore: conversationStores.historyStore,
    conversationLogger,
    eventLogger: systemLogger,
    progressReporter: progressBroker,
    feedbackStore,
    intentDetection
  });

  // ── WeChat Smart Bot Adapter ─────────────────────────────────────────────
  const wechatAdapter = new WechatSmartBotAdapter({
    botId: config.wechat.botId,
    secret: config.wechat.secret,
    messageHandler: orchestrator,
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
      : {}),
    rejectWhenConnectionUnavailable: config.wechat.rejectWhenConnectionUnavailable
  });

  let cronScheduler: TickCronScheduler | undefined;

  const server = createServer({
    messageHandler: orchestrator,
    roleConfigStore,
    feedbackStore,
    authorization,
    progressBroker,
    enableDevRoutes: config.enableDevRoutes,
    logger: true,
    readinessChecks: buildReadinessChecks({
      db,
      cronEnabled: config.cron.enabled,
      getCronScheduler: () => cronScheduler,
      getWechatAdapter: () => wechatAdapter
    })
  });

  // ── Cron Scheduler ─────────────────────────────────────────────────────────
  if (config.cron.enabled) {
    const cronJobStore = createCronJobStore({
      backend: config.cron.jobStore,
      db,
      filePath: config.cron.jobStorePath
    });

    const wecomDelivery =
      config.cron.wecom !== undefined
        ? new WecomAppMessageDeliveryChannel(config.cron.wecom)
        : await createFallbackWecomBotDelivery(wechatAdapter, systemLogger);
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
      staleGraceMs: config.cron.staleGraceMs,
      leaderLease: new FileCronLeaderLease({
        leasePath: config.cron.leaderLeasePath,
        ttlMs: config.cron.leaderLeaseTtlMs
      }),
      leaderRenewIntervalMs: Math.max(1_000, Math.floor(config.cron.leaderLeaseTtlMs / 3))
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

  if (cronScheduler !== undefined) {
    cronScheduler.start();
  }

  console.info(
    formatStartupBanner({
      serverUrl: `http://${config.host}:${config.port}`,
      devRoutesEnabled: config.enableDevRoutes,
      agentSdk: {
        type: sdkSummary.type,
        modelId: sdkSummary.modelId,
        baseUrl: sdkSummary.baseUrl
      },
      intentLlm: {
        modelId: llmSummary.modelId,
        baseUrl: llmSummary.baseUrl
      },
      runtimes: [
        { role: "reviewer", runtimeName: agentRuntimes["reviewer"]?.name ?? "not configured" },
        { role: "developer", runtimeName: agentRuntimes["developer"]?.name ?? "not configured" },
        { role: "admin", runtimeName: agentRuntimes["admin"]?.name ?? "not configured" }
      ],
      wechat: {
        status: "connecting",
        wsUrl: config.wechat.wsUrl ?? "wss://openws.work.weixin.qq.com"
      },
      cron: buildStartupCronSummary(config.cron),
      paths: {
        knowledgeBasePath: config.knowledgeBasePath,
        conversationLogPath: conversationLogger.filePath,
        llmRawLogPath: llmRawLogger.filePath,
        systemLogPath: systemLogger.filePath,
        databasePath: config.dbPath,
        sessionStorePath:
          config.conversationStore === "sqlite"
            ? `${config.dbPath} (sqlite conversation_sessions)`
            : config.sessionStorePath,
        historyStorePath:
          config.conversationStore === "sqlite"
            ? `${config.dbPath} (sqlite conversation_histories)`
            : config.historyStorePath,
        ...(config.cron.enabled
          ? {
              cronJobStorePath:
                config.cron.jobStore === "sqlite"
                  ? `${config.dbPath} (sqlite cron_jobs)`
                  : config.cron.jobStorePath,
              cronLeaderLeasePath: config.cron.leaderLeasePath
            }
          : {})
      }
    })
  );

  wechatAdapter.connect();
  const stopWechatSessionMetricsLogger = startWechatSessionMetricsLogger({
    source: wechatAdapter,
    logger: systemLogger
  });
  console.info("WeChat smart bot connection requested.");
  let shuttingDown = false;
  const shutdown = (): void => {
    if (shuttingDown) {
      return;
    }
    shuttingDown = true;
    console.info("Shutting down...");
    void shutdownGracefully({
      cronScheduler,
      wechatAdapter,
      server,
      loggers: [conversationLogger, llmRawLogger, systemLogger],
      stopWechatSessionMetricsLogger,
      stopConversationArchiveRetention
    }).finally(() => {
      process.exit(0);
    });
  };
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}

await main();

function createConversationStores(options: {
  readonly backend: "sqlite" | "file";
  readonly db: Database.Database;
  readonly sessionStorePath: string;
  readonly historyStorePath: string;
  readonly historyMaxMessages: number;
}): {
  readonly sessionStore: ConversationSessionStore;
  readonly historyStore: ConversationHistoryStore;
} {
  if (options.backend === "sqlite") {
    return {
      sessionStore: new SqliteConversationSessionStore(options.db),
      historyStore: new SqliteConversationHistoryStore(options.db, {
        maxMessages: options.historyMaxMessages
      })
    };
  }

  return {
    sessionStore: new FileConversationSessionStore(options.sessionStorePath),
    historyStore: new FileConversationHistoryStore(options.historyStorePath, {
      maxMessages: options.historyMaxMessages
    })
  };
}

function createConversationArchiveRetentionStopper(options: {
  readonly db: Database.Database;
  readonly retentionDays: number;
  readonly cleanupIntervalMs: number;
  readonly logger: ReturnType<typeof createJsonlLogger>;
}): () => void {
  const handle = startConversationArchiveRetention(options);
  return () => {
    handle.stop();
  };
}

function createCronJobStore(options: {
  readonly backend: "sqlite" | "file";
  readonly db: Database.Database;
  readonly filePath: string;
}) {
  if (options.backend === "sqlite") {
    return new SqliteCronJobStore(options.db);
  }
  return new FileCronJobStore(options.filePath);
}

async function createFallbackWecomBotDelivery(
  adapter: WechatSmartBotAdapter,
  systemLogger: ReturnType<typeof createJsonlLogger>
): Promise<WecomBotDeliveryChannel> {
  await systemLogger.write({
    type: "cron.wecom_config_missing",
    deliveryMode: "smart_bot_fallback",
    message:
      "Cron WeCom app-message config is missing; using the smart bot WSS connection for wecom:* delivery targets."
  });
  console.warn("Cron WeCom app-message config is missing; using smart bot WSS delivery fallback.");
  return new WecomBotDeliveryChannel(adapter);
}

function buildReadinessChecks(options: {
  readonly db: Database.Database;
  readonly cronEnabled: boolean;
  getCronScheduler(): TickCronScheduler | undefined;
  getWechatAdapter(): WechatSmartBotAdapter | undefined;
}): readonly HealthCheck[] {
  return [
    {
      name: "sqlite",
      check: () => {
        options.db.prepare("SELECT 1").get();
        return { status: "ok" };
      }
    },
    {
      name: "wechat_ws",
      check: () =>
        options.getWechatAdapter()?.isConnected === true
          ? { status: "ok" }
          : { status: "fail", message: "disconnected" }
    },
    {
      name: "cron_scheduler",
      check: () => {
        if (!options.cronEnabled) {
          return { status: "warn", message: "disabled" };
        }
        return options.getCronScheduler()?.isRunning === true
          ? { status: "ok" }
          : { status: "fail", message: "not running" };
      }
    }
  ];
}

function buildStartupCronSummary(cron: {
  readonly enabled: boolean;
  readonly tickIntervalMs: number;
  readonly staleGraceMs: number;
  readonly leaderLeaseTtlMs: number;
  readonly wecom?: unknown;
}): StartupCronSummary {
  if (!cron.enabled) {
    return { enabled: false };
  }

  return {
    enabled: true,
    tickIntervalMs: cron.tickIntervalMs,
    staleGraceMs: cron.staleGraceMs,
    leaderLeaseTtlMs: cron.leaderLeaseTtlMs,
    deliveryMode: cron.wecom === undefined ? "smart-bot-fallback" : "app-message"
  };
}

async function shutdownGracefully(options: {
  readonly cronScheduler: TickCronScheduler | undefined;
  readonly wechatAdapter: WechatSmartBotAdapter;
  readonly server: ReturnType<typeof createServer>;
  readonly loggers: readonly ReturnType<typeof createJsonlLogger>[];
  stopWechatSessionMetricsLogger(): void;
  stopConversationArchiveRetention(): void;
}): Promise<void> {
  options.stopWechatSessionMetricsLogger();
  options.stopConversationArchiveRetention();
  options.cronScheduler?.stop();
  options.wechatAdapter.disconnect();
  await options.server.close();
  await Promise.all(
    options.loggers.map(async (logger) => {
      await logger.close?.();
    })
  );
}
