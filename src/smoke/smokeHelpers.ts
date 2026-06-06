/**
 * Shared types, helpers, and constants used by all regression smoke test scripts.
 */
import { readFile } from "node:fs/promises";
import { spawn, type ChildProcess } from "node:child_process";
import path from "node:path";
import Database from "better-sqlite3";
import { isRecord, stringField } from "../core/unknownRecord.js";

// ── Types ─────────────────────────────────────────────────────────────────────

export type SseEvent = {
  readonly type: string;
  readonly text?: string;
  readonly toolName?: string;
  readonly sessionId?: string;
  readonly message?: string;
  readonly toolUseId?: string;
  readonly input?: unknown;
  readonly isError?: boolean;
  readonly preTokens?: number;
};

export type ChatResult = {
  readonly text: string;
  readonly sessionId?: string;
  readonly toolCalls: readonly string[];
  readonly events: readonly SseEvent[];
};

export type ChatAttachmentPayload = {
  readonly name: string;
  readonly mimeType: string;
  readonly contentBase64: string;
  readonly sizeBytes: number;
};

export type ProgressEvent = {
  readonly stage?: string;
  readonly message?: string;
};

export type FetchTimeoutOptions = {
  readonly label: string;
  readonly timeoutMs: number;
};

export type SmokeConfig = {
  readonly baseUrl: string;
  readonly conversationLogPath: string;
  readonly llmRawLogPath: string;
  readonly systemLogPath: string;
  readonly dbPath: string;
  readonly requestTimeoutMs: number;
  readonly chatTimeoutMs: number;
  readonly piAiTimeoutMs: number;
};

export type KnowledgeBaseSmokeCase = {
  readonly name: string;
  readonly text: string;
  readonly userId?: string;
  readonly expectedIncludes: readonly string[];
  readonly forbiddenIncludes: readonly string[];
};

// ── Constants ─────────────────────────────────────────────────────────────────

export const RUNTIME_PREFIX_FOR_SDK_TYPE: Record<string, string> = {
  claude: "claude-sdk",
  codebuddy: "codebuddy-sdk",
  pi: "pi"
};

export const KNOWLEDGE_BASE_SMOKE_CASES: readonly KnowledgeBaseSmokeCase[] = [
  {
    name: "project-purpose-grounding",
    text: "What is the current project for?",
    expectedIncludes: ["order"],
    forbiddenIncludes: [
      "pet business",
      "pet application",
      "not a pet",
      "WeChat",
      "message channel",
      "agent runtime"
    ]
  },
  {
    name: "architecture-focuses-on-workspace",
    text: "What is the current architecture?",
    expectedIncludes: ["catalog", "lifecycle"],
    forbiddenIncludes: ["WeChat", "browser", "agent runtime", "model provider"]
  },
  {
    name: "follow-up-reuses-session-context",
    text: "For the service you just mentioned, what is its main responsibility?",
    expectedIncludes: ["order"],
    forbiddenIncludes: ["WeChat", "browser", "agent runtime", "model provider"]
  },
  {
    name: "chinese-creation-question-is-query",
    text: "客户订单是怎么创建的",
    userId: "smoke-user-cn",
    expectedIncludes: ["订单", "创建"],
    forbiddenIncludes: ["修改请求", "不能修改", "反馈", "agent runtime", "model provider"]
  },
  {
    name: "feedback-query-stays-in-selected-workspace",
    text: "有哪些反馈信息",
    userId: "smoke-user-cn-feedback",
    expectedIncludes: ["反馈"],
    forbiddenIncludes: [
      "FeedbackEntry",
      "sqliteFeedbackStore",
      "/dev/feedback",
      "src/core",
      "src/agent",
      "src/server",
      "src/persistence",
      "contracts.ts",
      "代码分析"
    ]
  }
] as const;

// ── Environment helpers ────────────────────────────────────────────────────────

export function readPositiveIntEnv(name: string, fallback: number): number {
  const raw = process.env[name];
  if (raw === undefined || raw.trim().length === 0) {
    return fallback;
  }

  const parsed = Number(raw);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${name} must be a positive integer, got: ${raw}`);
  }
  return parsed;
}

// ── Network helpers ───────────────────────────────────────────────────────────

export function isAbortError(error: unknown): boolean {
  if (error instanceof DOMException) {
    return error.name === "AbortError";
  }
  if (error instanceof Error) {
    return error.name === "AbortError";
  }
  return false;
}

export async function fetchWithTimeout(
  url: string,
  init: RequestInit,
  options: FetchTimeoutOptions
): Promise<Response> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), options.timeoutMs);

  try {
    return await fetch(url, {
      ...init,
      signal: controller.signal
    });
  } catch (error) {
    if (isAbortError(error)) {
      throw new Error(`${options.label} timed out after ${options.timeoutMs}ms.`, { cause: error });
    }
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

// ── Chat helpers ──────────────────────────────────────────────────────────────

export function createChatHelpers(config: SmokeConfig) {
  async function chat(
    text: string,
    userId = "smoke-user",
    attachments: readonly ChatAttachmentPayload[] = []
  ): Promise<ChatResult> {
    const response = await fetchWithTimeout(
      `${config.baseUrl}/dev/chat`,
      {
        method: "POST",
        headers: {
          "content-type": "application/json; charset=utf-8"
        },
        body: JSON.stringify({
          userId,
          text,
          ...(attachments.length > 0 ? { attachments } : {})
        })
      },
      {
        label: `Chat request for ${userId}: ${text}`,
        timeoutMs: config.chatTimeoutMs
      }
    );

    if (!response.ok) {
      throw new Error(`Chat request failed: ${response.status} ${await response.text()}`);
    }

    const body = await response.text();
    let finalText = "";
    let sessionId: string | undefined;
    const toolCalls: string[] = [];
    const events: SseEvent[] = [];

    for (const line of body.split("\n")) {
      if (line.startsWith("data: ")) {
        try {
          const event = JSON.parse(line.slice(6)) as SseEvent;
          events.push(event);
          if (event.type === "completed") {
            finalText = event.text ?? finalText;
            sessionId = event.sessionId;
          } else if (event.type === "text_delta") {
            finalText += event.text ?? "";
          } else if (event.type === "tool_use_start") {
            toolCalls.push(event.toolName ?? "unknown");
          }
        } catch {
          // skip malformed lines
        }
      }
    }

    return {
      text: finalText,
      ...(sessionId !== undefined ? { sessionId } : {}),
      toolCalls,
      events
    };
  }

  async function resetChat(userId: string): Promise<void> {
    await chat("/new", userId);
  }

  async function setRole(userId: string, role: string): Promise<void> {
    const response = await fetchWithTimeout(
      `${config.baseUrl}/dev/role`,
      {
        method: "POST",
        headers: {
          "content-type": "application/json; charset=utf-8"
        },
        body: JSON.stringify({
          userId,
          role
        })
      },
      {
        label: `Role update for ${userId}`,
        timeoutMs: config.requestTimeoutMs
      }
    );

    if (!response.ok) {
      throw new Error(`Role request failed: ${response.status} ${await response.text()}`);
    }
  }

  async function updateFeedbackStatus(
    id: number,
    status: "reviewed" | "resolved",
    userId: string
  ): Promise<void> {
    const response = await fetchWithTimeout(
      `${config.baseUrl}/dev/feedback/${id}`,
      {
        method: "PATCH",
        headers: {
          "content-type": "application/json; charset=utf-8"
        },
        body: JSON.stringify({
          userId,
          status
        })
      },
      {
        label: `Feedback status update for ${id}`,
        timeoutMs: config.requestTimeoutMs
      }
    );

    if (!response.ok) {
      throw new Error(`Feedback status update failed: ${response.status} ${await response.text()}`);
    }
  }

  return { chat, resetChat, setRole, updateFeedbackStatus };
}

// ── Assertion helpers ──────────────────────────────────────────────────────────

export function assertIncludes(
  text: string,
  expectedValues: readonly string[],
  caseName: string
): void {
  const normalizedText = text.toLowerCase();
  for (const expected of expectedValues) {
    if (!normalizedText.includes(expected.toLowerCase())) {
      throw new Error(
        `Smoke case ${caseName} expected response to include ${expected}. Response: ${text}`
      );
    }
  }
}

export function assertForbidden(
  text: string,
  forbiddenValues: readonly string[],
  caseName: string
): void {
  const normalizedText = text.toLowerCase();
  for (const forbidden of forbiddenValues) {
    if (normalizedText.includes(forbidden.toLowerCase())) {
      throw new Error(
        `Smoke case ${caseName} response included forbidden text ${forbidden}. Response: ${text}`
      );
    }
  }
}

export async function assertLogContains(
  filePath: string,
  expectedValues: readonly string[]
): Promise<void> {
  const content = await readFile(filePath, "utf8");

  for (const expected of expectedValues) {
    if (!content.includes(expected)) {
      throw new Error(`Expected ${filePath} to include ${expected}.`);
    }
  }
}

export async function assertLogContainsAny(
  filePath: string,
  expectedValues: readonly string[]
): Promise<void> {
  const content = await readFile(filePath, "utf8");

  if (!expectedValues.some((expected) => content.includes(expected))) {
    throw new Error(`Expected ${filePath} to include one of: ${expectedValues.join(", ")}.`);
  }
}

// ── Base smoke cases (shared across all runtimes) ─────────────────────────────

export async function runBaseSmokeCases(
  config: SmokeConfig,
  helpers: ReturnType<typeof createChatHelpers>
): Promise<void> {
  const { chat, resetChat, setRole, updateFeedbackStatus } = helpers;

  // Health check
  const healthResponse = await fetchWithTimeout(
    `${config.baseUrl}/health`,
    {},
    { label: "Health check", timeoutMs: config.requestTimeoutMs }
  );
  if (!healthResponse.ok) {
    throw new Error(`Health check failed: ${healthResponse.status}`);
  }
  console.info("[pass] health-check");

  // Dev events stream connects
  const controller = new AbortController();
  const eventsTimeout = setTimeout(() => controller.abort(), 5000);
  const eventsResponse = await fetch(`${config.baseUrl}/dev/events?userId=smoke-events-user`, {
    signal: controller.signal
  });
  try {
    if (!eventsResponse.ok) {
      throw new Error(`Events stream failed: ${eventsResponse.status}`);
    }
    if (eventsResponse.body === null) {
      throw new Error("Events stream response had no body.");
    }
    const event = await readFirstSseData<ProgressEvent>(eventsResponse.body);
    if (event.stage !== "events.connected") {
      throw new Error(`Expected events.connected progress event, got: ${JSON.stringify(event)}`);
    }
  } finally {
    clearTimeout(eventsTimeout);
    controller.abort();
  }
  console.info("[pass] dev-events-stream-connects");

  // Path traversal denied
  const traversalResponse = await fetchWithTimeout(
    `${config.baseUrl}/dev/chat/..%2F..%2Fpackage.json`,
    {},
    { label: "Dev static path traversal check", timeoutMs: config.requestTimeoutMs }
  );
  if (traversalResponse.status !== 403) {
    throw new Error(
      `Expected dev static path traversal to return 403, got ${traversalResponse.status}`
    );
  }
  console.info("[pass] dev-chat-path-traversal-denied");

  // Reset session
  await chat("/new");

  // Knowledge-base grounding cases
  for (const smokeCase of KNOWLEDGE_BASE_SMOKE_CASES) {
    const userId = smokeCase.userId ?? "smoke-user";
    if (smokeCase.userId !== undefined) {
      await resetChat(userId);
    }
    const result = await chat(smokeCase.text, userId);
    assertIncludes(result.text, smokeCase.expectedIncludes, smokeCase.name);
    assertForbidden(result.text, smokeCase.forbiddenIncludes, smokeCase.name);
    console.info(`[pass] ${smokeCase.name}`);
  }

  await assertUploadedAttachmentAnswerFlow(config, helpers);
  console.info("[pass] uploaded-attachment-answer-flow");

  // Reviewer mutation denied and recorded as feedback
  const feedbackUserId = "smoke-feedback-user";
  await resetChat(feedbackUserId);
  const feedbackText = "Please update the documentation with the latest order lifecycle.";
  const feedbackResult = await chat(feedbackText, feedbackUserId);
  assertIncludes(feedbackResult.text, ["记录"], "reviewer-update-recorded-as-feedback");
  assertFeedbackRecorded(config.dbPath, feedbackUserId, feedbackText, "update_kb");
  assertFeedbackTimestampLocal(
    config.dbPath,
    feedbackUserId,
    "reviewer-update-recorded-as-feedback"
  );
  console.info("[pass] reviewer-update-recorded-as-feedback");

  // Chinese mutation denied
  const chineseMutationUserId = "smoke-reviewer-mutation-user";
  await resetChat(chineseMutationUserId);
  const chineseMutationText = "我想修改订单系统";
  const chineseMutationResult = await chat(chineseMutationText, chineseMutationUserId);
  assertIncludes(
    chineseMutationResult.text,
    ["修改请求", "记录"],
    "reviewer-chinese-mutation-denied"
  );
  assertForbidden(
    chineseMutationResult.text,
    ["计划", "审批", "Express", "TypeScript"],
    "reviewer-chinese-mutation-denied"
  );
  assertFeedbackRecorded(config.dbPath, chineseMutationUserId, chineseMutationText, "mutate");
  await assertNoAgentRuntimeLlmResponseForUser(config.llmRawLogPath, chineseMutationUserId);
  console.info("[pass] reviewer-chinese-mutation-denied");

  // Developer can act
  await setRole("smoke-developer", "developer");
  await resetChat("smoke-developer");
  const mutationResult = await chat("Add a comment to the main file", "smoke-developer");
  assertIncludes(mutationResult.text, [], "developer-can-act");
  console.info("[pass] developer-can-act");

  // Admin can manage feedback
  await setRole("smoke-admin", "admin");
  const feedbackListResponse = await fetchWithTimeout(
    `${config.baseUrl}/dev/feedback?userId=smoke-admin`,
    {},
    { label: "Admin feedback list", timeoutMs: config.requestTimeoutMs }
  );
  if (!feedbackListResponse.ok) {
    throw new Error(`Admin feedback list failed: ${feedbackListResponse.status}`);
  }
  const feedbackData = (await feedbackListResponse.json()) as { feedback: unknown[] };
  if (!Array.isArray(feedbackData.feedback)) {
    throw new Error("Admin feedback list did not return an array");
  }
  console.info("[pass] admin-can-view-feedback");

  // Get the feedback ID for status update
  const feedbackId = getLatestFeedbackId(config.dbPath, feedbackUserId);
  await updateFeedbackStatus(feedbackId, "reviewed", "smoke-admin");
  assertFeedbackStatus(config.dbPath, feedbackId, "reviewed", "admin-can-update-feedback-status");
  console.info("[pass] admin-can-update-feedback-status");

  // Reviewer cannot access feedback
  await setRole("smoke-reviewer-no-feedback", "reviewer");
  const deniedFeedbackResponse = await fetchWithTimeout(
    `${config.baseUrl}/dev/feedback?userId=smoke-reviewer-no-feedback`,
    {},
    { label: "Reviewer feedback denial check", timeoutMs: config.requestTimeoutMs }
  );
  if (deniedFeedbackResponse.status !== 403) {
    throw new Error(
      `Expected 403 for reviewer feedback access, got ${deniedFeedbackResponse.status}`
    );
  }
  console.info("[pass] reviewer-denied-feedback-access");

  // New conversation resets context
  const resetResult = await chat("/new");
  assertIncludes(resetResult.text, ["New conversation started"], "new-conversation-command");
  const postResetResult = await chat("What is the current project after reset?");
  assertIncludes(postResetResult.text, ["order"], "post-reset-starts-fresh-history");
  assertForbidden(
    postResetResult.text,
    ["pet business", "pet application", "pet orders"],
    "post-reset-starts-fresh-history"
  );
  console.info("[pass] new-conversation-resets-context");

  // Verify logs
  await assertLogContains(config.conversationLogPath, [
    "conversation.turn",
    "smoke-user",
    "What is the current project for?"
  ]);
  await assertLogContains(config.llmRawLogPath, [
    '"type":"llm.request"',
    '"type":"llm.response"',
    '"operation":"intent_detection"',
    '"operation":"agent_runtime"',
    '"type":"intent.result"'
  ]);
  await assertLogContainsAny(config.llmRawLogPath, [
    '"type":"agent.tool_call"',
    '"type":"llm.compact"'
  ]);
  await assertContextUsageLogged(config.systemLogPath);
  console.info("[pass] logs-written");

  // Workspace rules loaded
  await resetChat("smoke-rules-user");
  const ruleResult = await chat("What is the scope of this workspace?", "smoke-rules-user");
  assertIncludes(ruleResult.text.toLowerCase(), ["workspace"], "workspace-rules-loaded");
  assertForbidden(
    ruleResult.text,
    ["agent runtime", "model provider", "WeChat"],
    "workspace-rules-loaded"
  );
  console.info("[pass] workspace-rules-loaded");
}

// ── Dev server lifecycle ──────────────────────────────────────────────────────

async function assertUploadedAttachmentAnswerFlow(
  config: SmokeConfig,
  helpers: ReturnType<typeof createChatHelpers>
): Promise<void> {
  const documentUserId = "smoke-upload-document-user";
  const documentText = "Uploaded support token: GLACIER-742.";
  await helpers.resetChat(documentUserId);
  const documentResult = await helpers.chat(
    "Use the uploaded document only. What is the uploaded support token?",
    documentUserId,
    [
      {
        name: "support-note.md",
        mimeType: "text/markdown",
        contentBase64: Buffer.from(documentText, "utf8").toString("base64"),
        sizeBytes: Buffer.byteLength(documentText)
      }
    ]
  );
  assertIncludes(documentResult.text, ["GLACIER-742"], "uploaded-document-answer");

  const imageUserId = "smoke-upload-image-user";
  const imageBytes = Buffer.from(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII=",
    "base64"
  );
  await helpers.resetChat(imageUserId);
  const imageResult = await helpers.chat(
    "Acknowledge the uploaded image named smoke-diagram.png.",
    imageUserId,
    [
      {
        name: "smoke-diagram.png",
        mimeType: "application/octet-stream",
        contentBase64: imageBytes.toString("base64"),
        sizeBytes: imageBytes.length
      }
    ]
  );
  if (imageResult.text.length === 0) {
    throw new Error("uploaded-image-answer: expected a non-empty response.");
  }

  await assertLogContains(config.conversationLogPath, [
    '"userId":"smoke-upload-document-user"',
    '"attachmentCount":1',
    '"name":"support-note.md"',
    '"userId":"smoke-upload-image-user"',
    '"name":"smoke-diagram.png"',
    '"mimeType":"image/png"'
  ]);
  await assertLogContains(config.llmRawLogPath, [
    "Uploaded support token: GLACIER-742",
    "Image: smoke-diagram.png",
    "Media type: image/png"
  ]);
  await assertLogsDoNotExposeUploadStoragePath(config);
}

async function assertLogsDoNotExposeUploadStoragePath(config: SmokeConfig): Promise<void> {
  const [conversationContent, rawContent, systemContent] = await Promise.all([
    readFile(config.conversationLogPath, "utf8"),
    readFile(config.llmRawLogPath, "utf8"),
    readFile(config.systemLogPath, "utf8")
  ]);
  const combined = [conversationContent, rawContent, systemContent].join("\n");
  const forbiddenPaths = [".harness/uploads", ".harness\\uploads"];
  const leakedPath = forbiddenPaths.find((value) => combined.includes(value));
  if (leakedPath !== undefined) {
    throw new Error(`Upload logs exposed storage path segment: ${leakedPath}`);
  }
}

export async function startDevServer(configPath: string, port: number): Promise<ChildProcess> {
  const tsxCliPath = path.resolve("node_modules", "tsx", "dist", "cli.mjs");
  const env = {
    ...process.env,
    CONFIG_PATH: configPath,
    PORT: String(port)
  };

  const serverProcess = spawn(process.execPath, [tsxCliPath, "src/index.ts"], {
    cwd: process.cwd(),
    env,
    stdio: ["pipe", "pipe", "pipe"]
  });

  // Wait for the server to become healthy
  const maxWaitMs = 30_000;
  const startMs = Date.now();
  const baseUrl = `http://127.0.0.1:${port}`;

  while (Date.now() - startMs < maxWaitMs) {
    try {
      const response = await fetch(`${baseUrl}/health`);
      if (response.ok) {
        return serverProcess;
      }
    } catch {
      // Server not ready yet
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }

  // Server didn't start in time — kill it and throw
  serverProcess.kill();
  throw new Error(`Dev server did not become healthy within ${maxWaitMs}ms.`);
}

export function stopDevServer(serverProcess: ChildProcess): void {
  if (serverProcess.exitCode === null) {
    serverProcess.kill("SIGTERM");
  }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

async function readFirstSseData<T>(body: ReadableStream<Uint8Array>): Promise<T> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) {
        throw new Error("SSE stream ended before sending data.");
      }

      buffer += decoder.decode(value, { stream: true });
      const eventBoundary = buffer.indexOf("\n\n");
      if (eventBoundary === -1) {
        continue;
      }

      const eventText = buffer.slice(0, eventBoundary);
      const dataLine = eventText.split("\n").find((line) => line.startsWith("data: "));
      if (dataLine === undefined) {
        throw new Error(`SSE event did not include a data line: ${eventText}`);
      }

      return JSON.parse(dataLine.slice(6)) as T;
    }
  } finally {
    await reader.cancel().catch(() => undefined);
  }
}

async function assertContextUsageLogged(systemLogPath: string): Promise<void> {
  const content = await readFile(systemLogPath, "utf8");
  const usageEvents = content
    .split(/\r?\n/)
    .filter(
      (line) => line.includes('"type":"context.usage"') && line.includes('"userId":"smoke-user"')
    );

  if (usageEvents.length === 0) {
    throw new Error(`Expected ${systemLogPath} to include a context.usage event for smoke-user.`);
  }

  const latest = JSON.parse(usageEvents[usageEvents.length - 1] ?? "{}") as {
    readonly inputTokens?: unknown;
    readonly outputTokens?: unknown;
    readonly contextWindow?: unknown;
    readonly usagePercent?: unknown;
  };

  if (
    typeof latest.inputTokens !== "number" ||
    typeof latest.outputTokens !== "number" ||
    typeof latest.contextWindow !== "number" ||
    typeof latest.usagePercent !== "number"
  ) {
    throw new Error(
      `context.usage event is missing numeric token fields: ${JSON.stringify(latest)}.`
    );
  }
}

async function assertNoAgentRuntimeLlmResponseForUser(
  llmRawLogPath: string,
  userId: string
): Promise<void> {
  const content = await readFile(llmRawLogPath, "utf8");

  const hasAgentRuntimeResponse = content
    .split(/\r?\n/)
    .some(
      (line) =>
        line.includes(`"userId":"${userId}"`) &&
        line.includes('"type":"llm.response"') &&
        line.includes('"operation":"agent_runtime"')
    );

  if (hasAgentRuntimeResponse) {
    throw new Error(
      `Expected no agent runtime LLM response for denied reviewer mutation user ${userId}.`
    );
  }
}

function assertFeedbackRecorded(
  dbPath: string,
  userId: string,
  text: string,
  intentType: string
): number {
  const db = new Database(dbPath, { readonly: true });
  try {
    const row = db
      .prepare(
        `
      SELECT id, user_message, intent_type, status
      FROM feedback
      WHERE user_id = ?
      ORDER BY id DESC
      LIMIT 1
    `
      )
      .get(userId) as
      | {
          readonly id: number;
          readonly user_message: string;
          readonly intent_type: string;
          readonly status: string;
        }
      | undefined;

    if (row === undefined) {
      throw new Error(`Expected feedback row for ${userId}.`);
    }
    if (row.user_message !== text || row.intent_type !== intentType || row.status !== "pending") {
      throw new Error(`Unexpected feedback row: ${JSON.stringify(row)}.`);
    }
    return row.id;
  } finally {
    db.close();
  }
}

function getLatestFeedbackId(dbPath: string, userId: string): number {
  const db = new Database(dbPath, { readonly: true });
  try {
    const row = db
      .prepare(
        `
      SELECT id
      FROM feedback
      WHERE user_id = ?
      ORDER BY id DESC
      LIMIT 1
    `
      )
      .get(userId) as { readonly id: number } | undefined;

    if (row === undefined) {
      throw new Error(`Expected feedback row for ${userId}.`);
    }
    return row.id;
  } finally {
    db.close();
  }
}

function assertFeedbackStatus(
  dbPath: string,
  id: number,
  expectedStatus: string,
  caseName: string
): void {
  const db = new Database(dbPath, { readonly: true });
  try {
    const row = db
      .prepare(
        `
      SELECT status
      FROM feedback
      WHERE id = ?
    `
      )
      .get(id) as { readonly status: string } | undefined;

    if (row === undefined) {
      throw new Error(`[${caseName}] Expected feedback row ${id}.`);
    }
    if (row.status !== expectedStatus) {
      throw new Error(
        `[${caseName}] Expected feedback ${id} status ${expectedStatus}, got ${row.status}.`
      );
    }
  } finally {
    db.close();
  }
}

function assertFeedbackTimestampLocal(dbPath: string, userId: string, caseName: string): void {
  const db = new Database(dbPath, { readonly: true });
  try {
    const row = db
      .prepare(
        `
      SELECT created_at
      FROM feedback
      WHERE user_id = ?
      ORDER BY id DESC
      LIMIT 1
    `
      )
      .get(userId) as { readonly created_at: string } | undefined;

    if (row === undefined) {
      throw new Error(`[${caseName}] Expected feedback row for ${userId}.`);
    }

    const createdAt = row.created_at;
    const localNow = new Date();
    const parsed = new Date(createdAt.replace(" ", "T"));
    if (Number.isNaN(parsed.getTime())) {
      throw new Error(`[${caseName}] Invalid created_at format: ${createdAt}`);
    }

    const diffMs = Math.abs(localNow.getTime() - parsed.getTime());
    const toleranceMs = 5 * 60 * 1000;
    if (diffMs > toleranceMs) {
      throw new Error(
        `[${caseName}] created_at (${createdAt}) is more than 5 min from local time (${localNow.toISOString()}). Diff: ${diffMs}ms`
      );
    }
  } finally {
    db.close();
  }
}

// ── SDK-specific shared assertion functions ──────────────────────────────────

export type ChatFn = (
  text: string,
  userId?: string,
  attachments?: readonly ChatAttachmentPayload[]
) => Promise<ChatResult>;
export type ResetChatFn = (userId: string) => Promise<void>;
export type SetRoleFn = (userId: string, role: string) => Promise<void>;

// 1a. Agent SDK runtime matching
export async function assertAgentSdkRuntimeMatchesConfig(options: {
  readonly llmRawLogPath: string;
  readonly agentSdkType: string;
}): Promise<void> {
  const runtimePrefix = RUNTIME_PREFIX_FOR_SDK_TYPE[options.agentSdkType];
  if (runtimePrefix === undefined) {
    throw new Error(`Unknown agentSdk.type: ${options.agentSdkType}`);
  }

  const runtimeNames = await readAgentRuntimeNames(options.llmRawLogPath);
  if (runtimeNames.length === 0) {
    throw new Error(
      `Expected ${options.llmRawLogPath} to include at least one agent_runtime llm.request.`
    );
  }

  const expectedPrefix = `${runtimePrefix}-`;
  const unexpected = runtimeNames.filter((name) => !name.startsWith(expectedPrefix));
  if (unexpected.length > 0) {
    throw new Error(
      `Expected agentSdk.type=${options.agentSdkType} to use runtime prefix ${expectedPrefix}. Saw: ${[...new Set(runtimeNames)].join(", ")}.`
    );
  }
}

// 1b. Stream events (text_delta, completed, non-empty text)
export async function assertStreamEvents(options: {
  readonly chat: ChatFn;
  readonly resetChat: ResetChatFn;
  readonly userId?: string;
  readonly runtimePrefix: string;
  readonly textDeltaRequired: boolean;
}): Promise<void> {
  const userId = options.userId ?? `smoke-${options.runtimePrefix}stream-user`;
  await options.resetChat(userId);
  const result = await options.chat("What is the order catalog?", userId);

  const hasTextDelta = result.events.some((e) => e.type === "text_delta");
  if (options.textDeltaRequired && !hasTextDelta) {
    throw new Error(
      `Stream events: expected at least one text_delta event. Got types: ${result.events.map((e) => e.type).join(", ")}`
    );
  }

  const hasCompleted = result.events.some((e) => e.type === "completed");
  if (!hasCompleted) {
    throw new Error("Stream events: expected a completed event.");
  }

  if (result.text.length === 0) {
    throw new Error("Stream events: expected non-empty final text.");
  }
}

// 1c. Session persistence (sessionId, runtime log entries)
export async function assertSessionPersistence(options: {
  readonly chat: ChatFn;
  readonly resetChat: ResetChatFn;
  readonly llmRawLogPath: string;
  readonly runtimePrefix: string;
  readonly userId?: string;
  readonly sessionIdRequired: boolean;
}): Promise<void> {
  const userId = options.userId ?? `smoke-${options.runtimePrefix}session-user`;
  await options.resetChat(userId);

  const first = await options.chat("What is the order lifecycle?", userId);
  if (options.sessionIdRequired && first.sessionId === undefined) {
    throw new Error("Session persistence: first chat did not return a sessionId.");
  }

  const second = await options.chat("Tell me more about it.", userId);
  if (options.sessionIdRequired && second.sessionId === undefined) {
    throw new Error("Session persistence: second chat did not return a sessionId.");
  }

  // Verify the llm-raw log contains agent_runtime responses with the runtime prefix
  const content = await readFile(options.llmRawLogPath, "utf8");
  const runtimeLines = content
    .split(/\r?\n/)
    .filter(
      (line) =>
        line.includes(`"runtime":"${options.runtimePrefix}`) &&
        line.includes(`"userId":"${userId}"`)
    );

  if (runtimeLines.length === 0) {
    throw new Error(
      `Session persistence: no ${options.runtimePrefix} runtime log entries found for ${userId}.`
    );
  }
}

// 1d. Tool permission logged (agent.tool_call has permittedByRole)
export async function assertToolPermissionLogged(options: {
  readonly chat: ChatFn;
  readonly resetChat: ResetChatFn;
  readonly setRole: SetRoleFn;
  readonly llmRawLogPath: string;
  readonly runtimePrefix: string;
  readonly userId?: string;
}): Promise<void> {
  const userId = options.userId ?? `smoke-${options.runtimePrefix}tool-user`;
  await options.setRole(userId, "developer");
  await options.resetChat(userId);
  const previousContent = await readFile(options.llmRawLogPath, "utf8");
  await options.chat("Read the README file", userId);

  const content = await readFile(options.llmRawLogPath, "utf8");
  const newContent = content.startsWith(previousContent)
    ? content.slice(previousContent.length)
    : content;
  const toolCallLines = newContent
    .split(/\r?\n/)
    .filter(
      (line) =>
        line.includes('"type":"agent.tool_call"') &&
        line.includes(`"runtime":"${options.runtimePrefix}`) &&
        line.includes(`"userId":"${userId}"`)
    );

  // If the model made tool calls, they should be logged with permission info
  if (toolCallLines.length > 0) {
    const firstCall = JSON.parse(toolCallLines[0] ?? "{}") as {
      readonly permittedByRole?: unknown;
    };
    if (firstCall.permittedByRole === undefined) {
      throw new Error("Tool permission: agent.tool_call log missing permittedByRole field.");
    }
  }

  // Verify the runtime name prefix is correct in the agent_runtime llm.response
  const runtimeResponseLines = newContent
    .split(/\r?\n/)
    .filter(
      (line) =>
        line.includes('"type":"llm.response"') &&
        line.includes('"operation":"agent_runtime"') &&
        line.includes(`"runtime":"${options.runtimePrefix}`) &&
        line.includes(`"userId":"${userId}"`)
    );

  if (runtimeResponseLines.length === 0) {
    throw new Error(`Tool permission: no agent_runtime llm.response found for ${userId}.`);
  }
}

// 1e. Bash restricted for reviewer (reviewer cannot invoke Bash tool)
export async function assertBashRestrictedForReviewer(options: {
  readonly chat: ChatFn;
  readonly resetChat: ResetChatFn;
  readonly llmRawLogPath: string;
  readonly runtimePrefix: string;
  readonly userId?: string;
}): Promise<void> {
  const userId = options.userId ?? `smoke-${options.runtimePrefix}bash-reviewer-user`;
  await options.resetChat(userId);
  const previousContent = await readFile(options.llmRawLogPath, "utf8");

  await options.chat("List the files in this workspace", userId);

  const content = await readFile(options.llmRawLogPath, "utf8");
  const newContent = content.startsWith(previousContent)
    ? content.slice(previousContent.length)
    : content;
  const bashToolCallLines = newContent
    .split(/\r?\n/)
    .filter(
      (line) =>
        line.includes('"type":"agent.tool_call"') &&
        line.includes(`"runtime":"${options.runtimePrefix}`) &&
        line.includes(`"userId":"${userId}"`) &&
        line.includes('"toolName":"Bash"')
    );

  if (bashToolCallLines.length > 0) {
    throw new Error(
      "Bash restriction: reviewer should not be able to invoke Bash tool, but a Bash tool_call was logged."
    );
  }
}

// 1f. Role switch carries history (developer→admin, prior topic referenced)
export async function assertRoleSwitchCarriesHistory(options: {
  readonly chat: ChatFn;
  readonly resetChat: ResetChatFn;
  readonly setRole: SetRoleFn;
  readonly systemLogPath: string;
  readonly userId?: string;
  readonly checkRuntimeSelected: boolean;
  readonly sessionIdMismatchIsError: boolean;
}): Promise<void> {
  const switchUserId = options.userId ?? "smoke-role-switch-user";

  await options.setRole(switchUserId, "developer");
  await options.resetChat(switchUserId);

  const first = await options.chat("客户订单是怎么创建的", switchUserId);
  assertIncludes(first.text, ["订单"], "role-switch-first-turn");
  if (first.sessionId === undefined) {
    throw new Error("Role switch: first chat did not return a sessionId.");
  }
  const developerSessionId = first.sessionId;

  await options.setRole(switchUserId, "admin");

  const second = await options.chat("我的第一个问题是什么", switchUserId);
  if (second.sessionId === undefined) {
    throw new Error("Role switch: second chat did not return a sessionId.");
  }

  if (options.sessionIdMismatchIsError && second.sessionId === developerSessionId) {
    throw new Error(
      `Role switch: expected a new sessionId after role change, got the same: ${second.sessionId}.`
    );
  }

  const secondLower = second.text.toLowerCase();
  if (
    !secondLower.includes("订单") &&
    !secondLower.includes("创建") &&
    !secondLower.includes("order") &&
    !secondLower.includes("客户")
  ) {
    throw new Error(
      `Role switch: expected admin to reference the prior conversation about orders. Response: ${second.text}`
    );
  }

  // Verify system.jsonl shows role.resolved events with different roles
  const systemContent = await readFile(options.systemLogPath, "utf8");
  const switchUserEvents = systemContent
    .split(/\r?\n/)
    .filter((line) => line.includes(`"userId":"${switchUserId}"`));

  const roleResolvedEvents = switchUserEvents.filter((line) =>
    line.includes('"type":"role.resolved"')
  );
  if (roleResolvedEvents.length < 2) {
    throw new Error("Role switch: expected at least 2 role.resolved events for the switch user.");
  }

  const roles = roleResolvedEvents.map((line) => {
    const parsed = JSON.parse(line) as { readonly role?: string };
    return parsed.role;
  });
  if (!roles.includes("developer") || !roles.includes("admin")) {
    throw new Error(
      `Role switch: expected both developer and admin roles, got: ${roles.join(", ")}`
    );
  }

  if (options.checkRuntimeSelected) {
    const runtimeSelectedEvents = switchUserEvents.filter((line) =>
      line.includes('"type":"runtime.selected"')
    );
    if (runtimeSelectedEvents.length < 2) {
      throw new Error(
        "Role switch: expected at least 2 runtime.selected events for the switch user."
      );
    }
  }
}

// 1g. API key not leaked in logs
export async function assertApiKeyNotInLogs(options: {
  readonly llmRawLogPath: string;
  readonly apiKey: string;
}): Promise<void> {
  if (options.apiKey.trim().length === 0) {
    return;
  }

  const content = await readFile(options.llmRawLogPath, "utf8");
  if (content.includes(options.apiKey)) {
    throw new Error("API key safety: the API key value was found in llm-raw.jsonl logs.");
  }
}

// 1h. Options logged (Codebuddy skills/settingSources in llm.request)
export async function assertOptionsLogged(options: {
  readonly llmRawLogPath: string;
  readonly runtimePrefix: string;
}): Promise<void> {
  const content = await readFile(options.llmRawLogPath, "utf8");
  const requestLines = content
    .split(/\r?\n/)
    .filter(
      (line) =>
        line.includes('"type":"llm.request"') &&
        line.includes('"operation":"agent_runtime"') &&
        line.includes(`"runtime":"${options.runtimePrefix}`)
    );

  if (requestLines.length === 0) {
    throw new Error(
      `Options logged: no agent_runtime llm.request found with ${options.runtimePrefix} runtime prefix.`
    );
  }

  const firstRequest = JSON.parse(requestLines[0] ?? "{}") as {
    readonly options?: Record<string, unknown>;
  };

  if (firstRequest.options === undefined) {
    throw new Error("Options logged: llm.request missing options field.");
  }

  if (
    firstRequest.options["skills"] === undefined &&
    firstRequest.options["settingSources"] === undefined
  ) {
    throw new Error(
      `Options logged: expected skills or settingSources in options. Got: ${JSON.stringify(Object.keys(firstRequest.options))}`
    );
  }
}

// ── Log reading helpers ───────────────────────────────────────────────────────

export async function readAgentRuntimeNames(llmRawLogPath: string): Promise<readonly string[]> {
  const content = await readFile(llmRawLogPath, "utf8");
  const runtimeNames: string[] = [];

  for (const line of content.split(/\r?\n/)) {
    if (line.trim().length === 0) continue;

    const parsed = JSON.parse(line) as unknown;
    if (!isRecord(parsed)) continue;
    if (stringField(parsed, "type") !== "llm.request") continue;
    if (stringField(parsed, "operation") !== "agent_runtime") continue;

    const runtime = stringField(parsed, "runtime");
    if (runtime !== undefined) {
      runtimeNames.push(runtime);
    }
  }

  return runtimeNames;
}
