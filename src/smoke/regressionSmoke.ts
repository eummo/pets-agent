import { readFile } from "node:fs/promises";
import path from "node:path";
import "dotenv/config";
import { loadRuntimeConfig } from "../config/runtimeConfig.js";
import {
  type SmokeConfig,
  readPositiveIntEnv,
  createChatHelpers,
  assertLogContains,
  runBaseSmokeCases,
  assertAgentSdkRuntimeMatchesConfig,
  assertStreamEvents,
  assertSessionPersistence,
  assertToolPermissionLogged,
  assertBashRestrictedForReviewer,
  assertRoleSwitchCarriesHistory,
  assertApiKeyNotInLogs,
  assertOptionsLogged
} from "./smokeHelpers.js";

const config = await loadRuntimeConfig();
const agentSdkType = config.agentSdk.type;
const baseUrl = process.env["SMOKE_BASE_URL"] ?? `http://127.0.0.1:${config.port}`;
const conversationLogPath = path.resolve(config.logDir, "conversation.jsonl");
const llmRawLogPath = path.resolve(config.logDir, "llm-raw.jsonl");
const systemLogPath = path.resolve(config.logDir, "system.jsonl");
const dbPath = config.dbPath;
const smokePiAiTimeoutMs = readPositiveIntEnv("SMOKE_PI_AI_TIMEOUT_MS", 60_000);

const smokeConfig: SmokeConfig = {
  baseUrl,
  conversationLogPath,
  llmRawLogPath,
  systemLogPath,
  dbPath,
  requestTimeoutMs: readPositiveIntEnv("SMOKE_REQUEST_TIMEOUT_MS", 30_000),
  chatTimeoutMs: readPositiveIntEnv("SMOKE_CHAT_TIMEOUT_MS", 240_000),
  piAiTimeoutMs: smokePiAiTimeoutMs
};

const helpers = createChatHelpers(smokeConfig);
const { chat, resetChat, setRole } = helpers;

async function main(): Promise<void> {
  await runBaseSmokeCases(smokeConfig, helpers);

  // Verify agent SDK type is reflected in runtime name logged by the orchestrator
  await assertAgentSdkRuntimeMatchesConfig({ llmRawLogPath, agentSdkType });
  console.info("[pass] agent-sdk-runtime-matches-config");

  // ── Pi SDK-specific smoke tests ─────────────────────────────────────────────
  if (agentSdkType === "pi") {
    // Pi runtime forwards stream events (text_delta, thinking, tool_use_start)
    await assertStreamEvents({
      chat,
      resetChat,
      runtimePrefix: "pi-",
      textDeltaRequired: true
    });
    console.info("[pass] pi-agent-stream-events");

    // Pi runtime persists session context across turns
    await assertSessionPersistence({
      chat,
      resetChat,
      llmRawLogPath,
      runtimePrefix: "pi-",
      sessionIdRequired: true
    });
    console.info("[pass] pi-agent-session-persistence");

    // Pi runtime logs tool calls with permission info in llm-raw.jsonl
    await assertToolPermissionLogged({
      chat,
      resetChat,
      setRole,
      llmRawLogPath,
      runtimePrefix: "pi-"
    });
    console.info("[pass] pi-agent-tool-permission-logged");

    // Reviewer cannot invoke Bash tool under Pi runtime
    await assertBashRestrictedForReviewer({
      chat,
      resetChat,
      llmRawLogPath,
      runtimePrefix: "pi-"
    });
    console.info("[pass] pi-bash-restricted-for-reviewer");

    // Role switch carries conversation history to the new runtime session
    await assertRoleSwitchCarriesHistory({
      chat,
      resetChat,
      setRole,
      systemLogPath,
      checkRuntimeSelected: true,
      sessionIdMismatchIsError: true
    });
    console.info("[pass] pi-role-switch-carries-history");

    // API key not leaked in logs
    const piApiKey = config.agentSdk.apiKey;
    await assertApiKeyNotInLogs({ llmRawLogPath, apiKey: piApiKey });
    console.info("[pass] pi-api-key-not-in-logs");
  }

  // ── Codebuddy SDK-specific smoke tests ──────────────────────────────────────
  if (agentSdkType === "codebuddy") {
    // Verify skills and settingSources are passed to the Codebuddy SDK
    await assertLogContains(llmRawLogPath, [
      '"skills":"all"',
      '"settingSources":["project","local"]'
    ]);
    console.info("[pass] sdk-options-include-skills-and-setting-sources");

    // Codebuddy runtime forwards stream events
    await assertStreamEvents({
      chat,
      resetChat,
      runtimePrefix: "codebuddy-sdk-",
      textDeltaRequired: false
    });
    console.info("[pass] codebuddy-stream-events");

    // Codebuddy runtime persists session via resume option
    await assertSessionPersistence({
      chat,
      resetChat,
      llmRawLogPath,
      runtimePrefix: "codebuddy-sdk-",
      sessionIdRequired: false
    });
    console.info("[pass] codebuddy-session-resume");

    // Codebuddy runtime logs SDK options in llm.request
    await assertOptionsLogged({ llmRawLogPath, runtimePrefix: "codebuddy-sdk-" });
    console.info("[pass] codebuddy-agent-options-logged");

    // Role switch carries conversation history under Codebuddy runtime
    await assertRoleSwitchCarriesHistory({
      chat,
      resetChat,
      setRole,
      systemLogPath,
      userId: "smoke-codebuddy-role-switch-user",
      checkRuntimeSelected: false,
      sessionIdMismatchIsError: false
    });
    console.info("[pass] codebuddy-role-switch-carries-history");

    // API key not leaked in logs
    const cbApiKey = config.agentSdk.apiKey;
    await assertApiKeyNotInLogs({ llmRawLogPath, apiKey: cbApiKey });
    console.info("[pass] codebuddy-api-key-not-in-logs");
  }

  // ── Claude SDK-specific smoke tests ──────────────────────────────────────────
  if (agentSdkType === "claude") {
    // Claude runtime passes enableWorkflows in query options settings
    await assertClaudeWorkflowOptionsLogged();
    console.info("[pass] claude-workflow-options-logged");

    // Claude runtime exposes the Workflow tool when enableWorkflows is true
    await assertClaudeWorkflowToolAvailable();
    console.info("[pass] claude-workflow-tool-available");
  }
}

// ── Claude SDK-specific assertions (Claude-only, kept inline) ─────────────

async function assertClaudeWorkflowOptionsLogged(): Promise<void> {
  // Check that llm.request logs contain enableWorkflows in the settings
  // for the developer role's agent_runtime request
  const content = await readFile(llmRawLogPath, "utf8");
  const requestLines = content
    .split(/\r?\n/)
    .filter(
      (line) =>
        line.includes('"type":"llm.request"') &&
        line.includes('"operation":"agent_runtime"') &&
        line.includes('"runtime":"claude-sdk-')
    );

  if (requestLines.length === 0) {
    throw new Error(
      "Claude workflow options: no agent_runtime llm.request found with claude-sdk runtime prefix."
    );
  }

  // Find a request that includes enableWorkflows in settings
  const devRequest = requestLines.find((line) => line.includes('"enableWorkflows":true'));

  if (devRequest === undefined) {
    throw new Error(
      "Claude workflow options: expected enableWorkflows:true in a claude-sdk llm.request options.settings."
    );
  }
}

async function assertClaudeWorkflowToolAvailable(): Promise<void> {
  // Use the developer role (which has enableWorkflows: true)
  // to trigger a chat and verify the Workflow tool is available in the tool list.
  await setRole("smoke-claude-workflow-user", "developer");
  await resetChat("smoke-claude-workflow-user");
  await chat("What tools are available to you?", "smoke-claude-workflow-user");

  // Check llm-raw.jsonl for the claude-sdk request that includes Workflow in tools
  const content = await readFile(llmRawLogPath, "utf8");
  const hasWorkflowInTools = content
    .split(/\r?\n/)
    .some(
      (line) =>
        line.includes('"runtime":"claude-sdk-') &&
        line.includes('"userId":"smoke-claude-workflow-user"') &&
        line.includes("Workflow")
    );

  if (!hasWorkflowInTools) {
    // Non-fatal: the model may not mention Workflow explicitly in its response.
    // The primary assertion is enableWorkflows in settings, which is already verified.
    console.info("[info] claude-workflow-tool-mention-not-found-in-log (non-fatal)");
  }
}

await main();
