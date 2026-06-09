/**
 * Codebuddy SDK regression smoke tests.
 *
 * Usage: npm run smoke:codebuddy
 *
 * This script creates a temporary config with agentSdkType="codebuddy",
 * starts a dev server on an alternate port, runs base and Codebuddy-specific
 * smoke cases, then stops the server.
 */
import { cp, readFile, writeFile, mkdir, rm } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { homedir, tmpdir } from "node:os";
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
  assertRoleSwitchCarriesHistory,
  assertApiKeyNotInLogs,
  assertOptionsLogged
} from "./smokeHelpers.js";

const CODEBUDDY_PORT = 3001;
const CODEBUDDY_AGENT_SDK_TYPE = "codebuddy";
const CODEBUDDY_API_KEY_ENV = "CODEBUDDY_SMOKE_API_KEY";
const CODEBUDDY_ENDPOINT_ENV = "CODEBUDDY_SMOKE_ENDPOINT";
const CODEBUDDY_AUTH_ENVIRONMENT_ENV = "CODEBUDDY_SMOKE_ENVIRONMENT";
const CODEBUDDY_MODEL_ENV = "CODEBUDDY_SMOKE_MODEL";

async function main(): Promise<void> {
  // Load the base config and create a temp copy with agentSdkType="codebuddy"
  const baseConfig = await loadRuntimeConfig();
  const tempDir = resolveCodebuddySmokeTempDir();
  await rm(tempDir, { recursive: true, force: true });
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

  // Resolve endpoint and environment from config or local settings.
  const codebuddyApiKey = process.env[CODEBUDDY_API_KEY_ENV];
  const existingAuthToken = process.env["CODEBUDDY_AUTH_TOKEN"];
  const codebuddyEndpoint =
    nonEmptyEnv(CODEBUDDY_ENDPOINT_ENV) ?? (await readLocalCodebuddyEnterpriseEndpoint());
  const codebuddyEnvironment =
    nonEmptyEnv(CODEBUDDY_AUTH_ENVIRONMENT_ENV) ?? inferEnvironmentFromEndpoint(codebuddyEndpoint);

  // Ensure CODEBUDDY_INTERNET_ENVIRONMENT is set in the process environment
  // so the dev server child process inherits it. The Codebuddy CLI subprocess
  // needs this to route to the correct authentication server.
  if (
    codebuddyEnvironment !== undefined &&
    process.env["CODEBUDDY_INTERNET_ENVIRONMENT"] === undefined
  ) {
    process.env["CODEBUDDY_INTERNET_ENVIRONMENT"] = codebuddyEnvironment;
  }

  const configuredCodebuddyEntry = await readCodebuddyEntry();
  const codebuddyModelId =
    nonEmptyEnv(CODEBUDDY_MODEL_ENV) ??
    readStringField(configuredCodebuddyEntry, "modelId") ??
    baseConfig.agentSdk.modelId;
  const codebuddyBaseUrl =
    readStringField(configuredCodebuddyEntry, "baseUrl") ?? baseConfig.agentSdk.baseUrl;
  const codebuddyEntry = {
    baseUrl: codebuddyBaseUrl,
    modelId: codebuddyModelId,
    ...(codebuddyEndpoint !== undefined ? { endpoint: codebuddyEndpoint } : {}),
    ...(codebuddyEnvironment !== undefined ? { environment: codebuddyEnvironment } : {}),
    ...(codebuddyApiKey !== undefined && codebuddyApiKey.trim().length > 0
      ? { apiKeyEnv: CODEBUDDY_API_KEY_ENV }
      : {})
  };
  const tempConfig = {
    port: CODEBUDDY_PORT,
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
    agentSdkType: CODEBUDDY_AGENT_SDK_TYPE,
    agentSdks: {
      codebuddy: codebuddyEntry
    }
  };

  await mkdir(path.dirname(tempDbPath), { recursive: true });
  await mkdir(tempLogDir, { recursive: true });
  await writeFile(tempConfigPath, JSON.stringify(tempConfig, null, 2), "utf8");

  if (codebuddyApiKey === undefined || codebuddyApiKey.trim().length === 0) {
    if (existingAuthToken !== undefined) {
      console.info("[info] Codebuddy smoke: using CODEBUDDY_AUTH_TOKEN from environment.");
    } else {
      console.info("[info] Codebuddy smoke: using CodeBuddy CLI cached login.");
    }
  }
  if (codebuddyEndpoint !== undefined) {
    console.info(`[info] Codebuddy smoke: using endpoint ${codebuddyEndpoint}.`);
  }
  if (codebuddyEnvironment !== undefined) {
    console.info(`[info] Codebuddy smoke: using environment ${codebuddyEnvironment}.`);
  }
  if (process.env[CODEBUDDY_MODEL_ENV] !== undefined) {
    console.info(`[info] Codebuddy smoke: using model ${codebuddyModelId}.`);
  }

  // Start dev server with the codebuddy config
  console.info(`Starting Codebuddy dev server on port ${CODEBUDDY_PORT}...`);
  const serverProcess = await startDevServer(tempConfigPath, CODEBUDDY_PORT);
  let succeeded = false;

  try {
    const baseUrl = `http://127.0.0.1:${CODEBUDDY_PORT}`;
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
    try {
      await runBaseSmokeCases(smokeConfig, helpers);
    } catch (error) {
      throw await enrichCodebuddySmokeError(error, llmRawLogPath);
    }

    // ── Agent SDK runtime matching ──────────────────────────────────────────
    await assertAgentSdkRuntimeMatchesConfig({
      llmRawLogPath,
      agentSdkType: CODEBUDDY_AGENT_SDK_TYPE
    });
    console.info("[pass] agent-sdk-runtime-matches-config");

    // ── Codebuddy-specific smoke cases ──────────────────────────────────────

    // 1. Stream events: text_delta and completed events are forwarded
    await assertStreamEvents({
      chat,
      resetChat,
      runtimePrefix: "codebuddy-sdk-",
      textDeltaRequired: true
    });
    console.info("[pass] codebuddy-stream-events");

    // 2. Session resume: follow-up chat reuses session via resume option
    await assertSessionPersistence({
      chat,
      resetChat,
      llmRawLogPath,
      runtimePrefix: "codebuddy-sdk-",
      sessionIdRequired: true
    });
    console.info("[pass] codebuddy-session-resume");

    // 3. Options logged: llm.request contains skills and settingSources
    await assertOptionsLogged({ llmRawLogPath, runtimePrefix: "codebuddy-sdk-" });
    console.info("[pass] codebuddy-agent-options-logged");

    // 4. API key / auth token not leaked in logs
    const secretToCheck = codebuddyApiKey ?? existingAuthToken;
    if (secretToCheck !== undefined && secretToCheck.trim().length > 0) {
      await assertApiKeyNotInLogs({ llmRawLogPath, apiKey: secretToCheck });
      console.info("[pass] codebuddy-secret-not-in-logs");
    }

    // 5. Role switch carries history
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
    succeeded = true;
  } finally {
    stopDevServer(serverProcess);
    if (shouldRemoveCodebuddySmokeTemp(succeeded)) {
      await rm(tempDir, { recursive: true, force: true }).catch(() => undefined);
    } else {
      console.info(`[info] Codebuddy smoke: temp directory kept at ${tempDir}`);
    }
  }
}

function resolveCodebuddySmokeTempDir(): string {
  const configured = nonEmptyEnv("CODEBUDDY_SMOKE_TEMP_DIR");
  if (configured !== undefined) {
    return path.resolve(configured);
  }
  return path.join(tmpdir(), `pets-agent-smoke-codebuddy-${Date.now()}`);
}

function shouldRemoveCodebuddySmokeTemp(succeeded: boolean): boolean {
  if (process.env["CODEBUDDY_SMOKE_KEEP_TEMP"] === "1") {
    return false;
  }
  return succeeded;
}

function nonEmptyEnv(name: string): string | undefined {
  const value = process.env[name];
  return value !== undefined && value.trim().length > 0 ? value : undefined;
}

function inferEnvironmentFromEndpoint(endpoint: string | undefined): string | undefined {
  if (endpoint === undefined) return undefined;
  if (endpoint.includes("copilot.qq.com") || endpoint.includes("tencent.com")) {
    return "internal";
  }
  return undefined;
}

async function readLocalCodebuddyEnterpriseEndpoint(): Promise<string | undefined> {
  const settingsPath = path.join(homedir(), ".codebuddy", "settings.json");
  try {
    const settings: unknown = JSON.parse(await readFile(settingsPath, "utf8"));
    const endpoint = isRecord(settings) ? settings["enterpriseEndpoint"] : undefined;
    return typeof endpoint === "string" && endpoint.trim().length > 0 ? endpoint : undefined;
  } catch {
    return undefined;
  }
}

async function readCodebuddyEntry(): Promise<Record<string, unknown> | undefined> {
  const configPath = path.resolve("config", "runtime.json");
  try {
    const content = await readFile(configPath, "utf8");
    const config: unknown = JSON.parse(content);
    if (
      !isRecord(config) ||
      !isRecord(config["agentSdks"]) ||
      !isRecord(config["agentSdks"]["codebuddy"])
    ) {
      return undefined;
    }
    return config["agentSdks"]["codebuddy"];
  } catch {
    return undefined;
  }
}

function readStringField(
  record: Record<string, unknown> | undefined,
  field: string
): string | undefined {
  if (record === undefined) return undefined;
  const value = record[field];
  return typeof value === "string" && value.trim().length > 0 ? value : undefined;
}

async function enrichCodebuddySmokeError(error: unknown, llmRawLogPath: string): Promise<Error> {
  const message = error instanceof Error ? error.message : String(error);
  const latestRuntimeError = await readLatestCodebuddyRuntimeError(llmRawLogPath);
  if (latestRuntimeError === undefined) {
    return error instanceof Error ? error : new Error(message);
  }
  return new Error(`${message}\nLatest CodeBuddy SDK error: ${latestRuntimeError}`, {
    cause: error
  });
}

async function readLatestCodebuddyRuntimeError(llmRawLogPath: string): Promise<string | undefined> {
  try {
    const content = await readFile(llmRawLogPath, "utf8");
    const lines = content.split(/\r?\n/).filter((line) => line.trim().length > 0);
    for (let i = lines.length - 1; i >= 0; i--) {
      const line = lines[i];
      if (line === undefined) continue;
      const event: unknown = JSON.parse(line);
      if (!isRecord(event)) continue;
      if (event["type"] !== "llm.error") continue;
      if (event["operation"] !== "agent_runtime") continue;
      const runtime = event["runtime"];
      if (typeof runtime !== "string" || !runtime.startsWith("codebuddy-sdk-")) continue;
      const eventError = event["error"];
      return typeof eventError === "string" && eventError.trim().length > 0
        ? eventError
        : undefined;
    }
    return undefined;
  } catch {
    return undefined;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

await main();
