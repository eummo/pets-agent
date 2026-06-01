/**
 * Pi SDK regression smoke tests.
 *
 * Usage: npm run smoke:pi
 *
 * This script creates a temporary config with agentSdkType="pi",
 * starts a dev server on an alternate port, runs base and Pi-specific
 * smoke cases, then stops the server.
 */
import { cp, writeFile, mkdir, rm } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { tmpdir } from "node:os";
import "dotenv/config";
import { loadRuntimeConfig } from "../config/runtimeConfig.js";
import {
  type SmokeConfig,
  readPositiveIntEnv,
  createChatHelpers,
  runBaseSmokeCases,
  startDevServer,
  stopDevServer,
  assertAgentSdkRuntimeMatchesConfig,
  assertStreamEvents,
  assertSessionPersistence,
  assertToolPermissionLogged,
  assertBashRestrictedForReviewer,
  assertRoleSwitchCarriesHistory,
  assertApiKeyNotInLogs
} from "./smokeHelpers.js";

const PI_PORT = 3002;
const PI_AGENT_SDK_TYPE = "pi";

async function main(): Promise<void> {
  // Load the base config and create a temp copy with agentSdkType="pi"
  const baseConfig = await loadRuntimeConfig();
  const tempDir = path.join(tmpdir(), `pets-agent-smoke-pi-${Date.now()}`);
  await mkdir(tempDir, { recursive: true });

  const tempConfigPath = path.join(tempDir, "runtime.json");
  const tempLogDir = path.join(tempDir, "logs");
  const tempDbPath = path.join(tempDir, "state", "agent.db");
  const tempSessionPath = path.join(tempDir, "state", "sessions.json");
  const tempHistoryPath = path.join(tempDir, "state", "history.json");
  const tempKbPath = path.join(tempDir, "knowledge-base");

  // Copy the harness knowledge base
  const harnessDir = path.resolve(".harness");
  const harnessKnowledgeBasePath = path.join(harnessDir, "knowledge-base");
  if (existsSync(harnessKnowledgeBasePath)) {
    await cp(harnessKnowledgeBasePath, tempKbPath, { recursive: true });
  }

  // Write temp config
  const tempConfig = {
    port: PI_PORT,
    host: "127.0.0.1",
    knowledgeBasePath: tempKbPath,
    logDir: tempLogDir,
    dbPath: tempDbPath,
    sessionStorePath: tempSessionPath,
    historyStorePath: tempHistoryPath,
    enableDevRoutes: true,
    wechat: { botId: "dev-bot-id", secret: "dev-secret" },
    llm: {
      baseUrl: baseConfig.llm.baseUrl,
      apiKeyEnv: baseConfig.llm.apiKeyEnv,
      modelId: baseConfig.llm.modelId,
      maxTokens: baseConfig.llm.maxTokens
    },
    agentSdkType: PI_AGENT_SDK_TYPE,
    agentSdks: {
      pi: {
        baseUrl: baseConfig.agentSdk.baseUrl,
        apiKeyEnv: "LOCAL_LLM_API_KEY",
        modelId: baseConfig.agentSdk.modelId
      }
    }
  };

  await mkdir(path.dirname(tempDbPath), { recursive: true });
  await mkdir(tempLogDir, { recursive: true });
  await writeFile(tempConfigPath, JSON.stringify(tempConfig, null, 2), "utf8");

  // The Pi SDK runtime uses AuthStorage.setRuntimeApiKey for auth.
  // The API key is resolved from the shared apiKeyEnv (LOCAL_LLM_API_KEY) in config.
  process.env["LOCAL_LLM_API_KEY"] = baseConfig.agentSdk.apiKey;

  // Start dev server with the pi config
  console.info(`Starting Pi dev server on port ${PI_PORT}...`);
  const serverProcess = await startDevServer(tempConfigPath, PI_PORT);

  try {
    const baseUrl = `http://127.0.0.1:${PI_PORT}`;
    const conversationLogPath = path.resolve(tempLogDir, "conversation.jsonl");
    const llmRawLogPath = path.resolve(tempLogDir, "llm-raw.jsonl");
    const systemLogPath = path.resolve(tempLogDir, "system.jsonl");

    const smokeConfig: SmokeConfig = {
      baseUrl,
      conversationLogPath,
      llmRawLogPath,
      systemLogPath,
      dbPath: tempDbPath,
      requestTimeoutMs: readPositiveIntEnv("SMOKE_REQUEST_TIMEOUT_MS", 30_000),
      chatTimeoutMs: readPositiveIntEnv("SMOKE_CHAT_TIMEOUT_MS", 240_000),
      piAiTimeoutMs: readPositiveIntEnv("SMOKE_PI_AI_TIMEOUT_MS", 60_000)
    };

    const helpers = createChatHelpers(smokeConfig);
    const { chat, resetChat, setRole } = helpers;

    // ── Base smoke cases ────────────────────────────────────────────────────
    await runBaseSmokeCases(smokeConfig, helpers);

    // ── Agent SDK runtime matching ──────────────────────────────────────────
    await assertAgentSdkRuntimeMatchesConfig({ llmRawLogPath, agentSdkType: PI_AGENT_SDK_TYPE });
    console.info("[pass] agent-sdk-runtime-matches-config");

    // ── Pi-specific smoke cases ─────────────────────────────────────────────

    // 1. Stream events: text_delta, tool_use_start, completed events are forwarded
    await assertStreamEvents({
      chat,
      resetChat,
      runtimePrefix: "pi-",
      textDeltaRequired: true
    });
    console.info("[pass] pi-stream-events");

    // 2. Session persistence: follow-up chat returns sessionId and session is reused
    await assertSessionPersistence({
      chat,
      resetChat,
      llmRawLogPath,
      runtimePrefix: "pi-",
      sessionIdRequired: true
    });
    console.info("[pass] pi-session-persistence");

    // 3. Tool permission logged: agent.tool_call entries contain permittedByRole field
    await assertToolPermissionLogged({
      chat,
      resetChat,
      setRole,
      llmRawLogPath,
      runtimePrefix: "pi-"
    });
    console.info("[pass] pi-tool-permission-logged");

    // 4. Bash restricted for reviewer: reviewer cannot invoke Bash tool
    await assertBashRestrictedForReviewer({
      chat,
      resetChat,
      llmRawLogPath,
      runtimePrefix: "pi-"
    });
    console.info("[pass] pi-bash-restricted-for-reviewer");

    // 5. Role switch carries history and gets new sessionId
    await assertRoleSwitchCarriesHistory({
      chat,
      resetChat,
      setRole,
      systemLogPath,
      userId: "smoke-pi-role-switch-user",
      checkRuntimeSelected: true,
      sessionIdMismatchIsError: true
    });
    console.info("[pass] pi-role-switch-carries-history");

    // 6. API key not leaked in logs
    const apiKey = process.env["LOCAL_LLM_API_KEY"] ?? baseConfig.agentSdk.apiKey;
    await assertApiKeyNotInLogs({ llmRawLogPath, apiKey });
    console.info("[pass] pi-api-key-not-in-logs");
  } finally {
    stopDevServer(serverProcess);
    // Clean up temp directory
    await rm(tempDir, { recursive: true, force: true }).catch(() => undefined);
  }
}

await main();
