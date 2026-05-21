import path from "node:path";
import { pathToFileURL } from "node:url";
import "dotenv/config";
import { ClaudeSdkAgentRuntime, REVIEWER_CONFIG, DEVELOPER_CONFIG } from "./agent/claudeSdkAgentRuntime.js";
import { EchoAgentRuntime } from "./agent/echoAgentRuntime.js";
import { loadLlmConfig, resolveLlmConfig, summarizeLlmConfig } from "./config/llmConfig.js";
import { FileConversationHistoryStore } from "./core/fileConversationHistoryStore.js";
import { FileConversationSessionStore } from "./core/fileConversationSessionStore.js";
import { AgentOrchestrator } from "./core/orchestrator.js";
import type { AgentRuntime } from "./core/ports.js";
import { createJsonlLogger, type JsonlLogger } from "./logging/jsonlLogger.js";
import { StaticWorkspaceResolver } from "./repos/staticWorkspaceResolver.js";
import { createServer } from "./server/createServer.js";
import { DevProgressBroker } from "./server/progressBroker.js";
import { createDevRoleStore } from "./security/devRoleStore.js";
import { StaticAuthorizationService } from "./security/staticAuthorizationService.js";

// Pets Agent - Main entry point for the agent runtime service
export const serviceName = "pets-agent";

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const port = Number.parseInt(process.env["PORT"] ?? "3000", 10);
  const knowledgeBasePath = process.env["KNOWLEDGE_BASE_PATH"] ?? ".harness/knowledge-base";
  const wechatToken = process.env["WECHAT_TOKEN"] ?? "dev-token";
  const logDir = process.env["LOG_DIR"] ?? path.resolve(".harness", "logs");
  const sessionStorePath = process.env["SESSION_STORE_PATH"] ?? path.resolve(".harness", "state", "sessions.json");
  const historyStorePath = process.env["HISTORY_STORE_PATH"] ?? path.resolve(".harness", "state", "history.json");
  const conversationLogger = createJsonlLogger(path.join(logDir, "conversation.jsonl"));
  const llmRawLogger = createJsonlLogger(path.join(logDir, "llm-raw.jsonl"));
  const devRoleStore = createDevRoleStore();
  const progressBroker = new DevProgressBroker();

  // Configure SDK environment from LLM config
  await configureSdkEnvironment();

  const agentRuntimes = await createAgentRuntimes(llmRawLogger);
  const orchestrator = new AgentOrchestrator({
    workspaceResolver: new StaticWorkspaceResolver({ knowledgeBasePath }),
    authorization: new StaticAuthorizationService(devRoleStore),
    agentRuntimes,
    sessionStore: new FileConversationSessionStore(sessionStorePath),
    historyStore: new FileConversationHistoryStore(historyStorePath),
    conversationLogger,
    progressReporter: progressBroker
  });
  const server = createServer({
    messageHandler: orchestrator,
    wechatToken,
    devRoleStore,
    progressBroker,
    logger: true
  });

  await server.listen({ port, host: "0.0.0.0" });
  console.info(`${serviceName} listening on http://localhost:${port}`);
  console.info(`reviewer runtime: ${agentRuntimes["reviewer"]?.name ?? "not configured"}`);
  console.info(`developer runtime: ${agentRuntimes["developer"]?.name ?? "not configured"}`);
  console.info(`conversation log: ${conversationLogger.filePath}`);
  console.info(`llm raw log: ${llmRawLogger.filePath}`);
}

async function configureSdkEnvironment(): Promise<void> {
  try {
    const llmConfigPath = process.env["LLM_CONFIG_PATH"] ?? path.resolve("config", "llm.json");
    const llmConfig = await loadLlmConfig(llmConfigPath);
    const resolved = resolveLlmConfig(llmConfig);

    // The Claude Agent SDK uses ANTHROPIC_API_KEY and ANTHROPIC_BASE_URL
    // Map from the existing LLM config
    process.env["ANTHROPIC_API_KEY"] ??= resolved.apiKey;
    process.env["ANTHROPIC_BASE_URL"] ??= resolved.baseUrl;

    const summary = summarizeLlmConfig(resolved);
    console.info(`SDK configured: ${summary.modelId} at ${summary.baseUrl}`);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.warn(`SDK env configuration skipped: ${message}`);
  }
}

async function createAgentRuntimes(llmRawLogger: JsonlLogger): Promise<Record<string, AgentRuntime>> {
  try {
    const llmConfigPath = process.env["LLM_CONFIG_PATH"] ?? path.resolve("config", "llm.json");
    const llmConfig = await loadLlmConfig(llmConfigPath);
    const resolved = resolveLlmConfig(llmConfig);

    return {
      reviewer: new ClaudeSdkAgentRuntime({
        roleConfig: REVIEWER_CONFIG,
        rawLogger: llmRawLogger,
        model: resolved.modelId,
      }),
      developer: new ClaudeSdkAgentRuntime({
        roleConfig: DEVELOPER_CONFIG,
        rawLogger: llmRawLogger,
        model: resolved.modelId,
      }),
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.warn(`using echo runtime because real LLM runtime is not configured: ${message}`);
    return {
      reviewer: new EchoAgentRuntime(),
      developer: new EchoAgentRuntime(),
    };
  }
}
