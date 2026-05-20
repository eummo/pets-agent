import path from "node:path";
import { pathToFileURL } from "node:url";
import "dotenv/config";
import { AnthropicCodeChangeRuntime } from "./agent/anthropicCodeChangeRuntime.js";
import { AnthropicCompatibleAgentRuntime } from "./agent/anthropicCompatibleAgentRuntime.js";
import { AnthropicManagedSessionAgentRuntime } from "./agent/anthropicManagedSessionAgentRuntime.js";
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
  const runtimes = await createAgentRuntimes(llmRawLogger);
  const orchestrator = new AgentOrchestrator({
    workspaceResolver: new StaticWorkspaceResolver({ knowledgeBasePath }),
    authorization: new StaticAuthorizationService(devRoleStore),
    agentRuntime: runtimes.agentRuntime,
    ...(runtimes.codeChangeRuntime === undefined ? {} : { codeChangeRuntime: runtimes.codeChangeRuntime }),
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
  console.info(`agent runtime: ${runtimes.agentRuntime.name}`);
  console.info(`code change runtime: ${runtimes.codeChangeRuntime?.name ?? "not configured"}`);
  console.info(`conversation log: ${conversationLogger.filePath}`);
  console.info(`llm raw log: ${llmRawLogger.filePath}`);
}

async function createAgentRuntimes(llmRawLogger: JsonlLogger): Promise<{
  readonly agentRuntime: AgentRuntime;
  readonly codeChangeRuntime?: AgentRuntime;
}> {
  try {
    const llmConfigPath = process.env["LLM_CONFIG_PATH"] ?? path.resolve("config", "llm.json");
    const llmConfig = await loadLlmConfig(llmConfigPath);
    const resolved = resolveLlmConfig(llmConfig);
    const summary = summarizeLlmConfig(resolved);
    console.info(`loaded LLM config: ${summary.modelId} at ${summary.baseUrl}`);
    if (resolved.runtime === "managed-sessions") {
      return {
        agentRuntime: new AnthropicManagedSessionAgentRuntime({
          baseUrl: resolved.baseUrl,
          apiKey: resolved.apiKey,
          agentId: resolved.agentId ?? "",
          environmentId: resolved.environmentId ?? "",
          rawLogger: llmRawLogger
        }),
        codeChangeRuntime: new AnthropicCodeChangeRuntime({
          baseUrl: resolved.baseUrl,
          apiKey: resolved.apiKey,
          modelId: resolved.modelId,
          rawLogger: llmRawLogger
        })
      };
    }

    return {
      agentRuntime: new AnthropicCompatibleAgentRuntime({
        baseUrl: resolved.baseUrl,
        apiKey: resolved.apiKey,
        modelId: resolved.modelId,
        rawLogger: llmRawLogger
      }),
      codeChangeRuntime: new AnthropicCodeChangeRuntime({
        baseUrl: resolved.baseUrl,
        apiKey: resolved.apiKey,
        modelId: resolved.modelId,
        rawLogger: llmRawLogger
      })
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.warn(`using echo runtime because real LLM runtime is not configured: ${message}`);
    return {
      agentRuntime: new EchoAgentRuntime()
    };
  }
}
