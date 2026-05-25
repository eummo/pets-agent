import { readFile } from "node:fs/promises";
import path from "node:path";
import Database from "better-sqlite3";
import "dotenv/config";
import { loadRuntimeConfig } from "../config/runtimeConfig.js";

type SseEvent = {
  readonly type: string;
  readonly text?: string;
  readonly toolName?: string;
  readonly sessionId?: string;
  readonly message?: string;
};

type ChatResult = {
  readonly text: string;
  readonly sessionId?: string;
  readonly toolCalls: readonly string[];
};

type ProgressEvent = {
  readonly stage?: string;
  readonly message?: string;
};

const config = await loadRuntimeConfig();
const baseUrl = process.env["SMOKE_BASE_URL"] ?? `http://127.0.0.1:${config.port}`;
const conversationLogPath = path.resolve(config.logDir, "conversation.jsonl");
const llmRawLogPath = path.resolve(config.logDir, "llm-raw.jsonl");
const systemLogPath = path.resolve(config.logDir, "system.jsonl");
const dbPath = config.dbPath;
const firstCaseText = "What is the current project for?";
const resetCaseText = "What is the current project after reset?";

const cases = [
  {
    name: "project-purpose-grounding",
    text: firstCaseText,
    expectedIncludes: ["order", "catalog"],
    forbiddenIncludes: ["pet business", "pet application", "not a pet", "WeChat", "message channel", "agent runtime"]
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
  }
] as const;

async function main(): Promise<void> {
  await assertHealthy();
  await assertDevEventsStreamConnects();
  await assertDevChatPathTraversalDenied();

  // Pi-ai smoke: verify complete() works with the configured Anthropic endpoint
  await assertPiAiComplete();

  for (const smokeCase of cases) {
    const result = await chat(smokeCase.text);
    assertIncludes(result.text, smokeCase.expectedIncludes, smokeCase.name);
    assertForbidden(result.text, smokeCase.forbiddenIncludes, smokeCase.name);
    console.info(`[pass] ${smokeCase.name}`);
  }

  // Reviewer mutation requests are denied and captured as feedback.
  const feedbackUserId = "smoke-feedback-user";
  const feedbackText = "Please update the documentation with the latest order lifecycle.";
  const feedbackResult = await chat(feedbackText, feedbackUserId);
  assertIncludes(feedbackResult.text, ["记录"], "reviewer-update-recorded-as-feedback");
  const feedbackId = assertFeedbackRecorded(feedbackUserId, feedbackText, "update_kb");
  assertFeedbackTimestampLocal(feedbackUserId, "reviewer-update-recorded-as-feedback");
  console.info("[pass] reviewer-update-recorded-as-feedback");

  const chineseMutationUserId = "smoke-reviewer-mutation-user";
  const chineseMutationText = "我想修改订单系统";
  const chineseMutationResult = await chat(chineseMutationText, chineseMutationUserId);
  assertIncludes(chineseMutationResult.text, ["修改请求", "记录"], "reviewer-chinese-mutation-denied");
  assertForbidden(chineseMutationResult.text, ["计划", "审批", "Express", "TypeScript"], "reviewer-chinese-mutation-denied");
  assertFeedbackRecorded(chineseMutationUserId, chineseMutationText, "mutate");
  await assertNoLlmResponseForUser(chineseMutationUserId);
  console.info("[pass] reviewer-chinese-mutation-denied");

  // Developer role can make code changes
  await setRole("smoke-developer", "developer");
  const mutationResult = await chat("Add a comment to the main file", "smoke-developer");
  assertIncludes(mutationResult.text, [], "developer-can-act");
  console.info("[pass] developer-can-act");

  // Admin role can manage feedback
  await setRole("smoke-admin", "admin");
  const feedbackListResponse = await fetch(`${baseUrl}/dev/feedback?userId=smoke-admin`);
  if (!feedbackListResponse.ok) {
    throw new Error(`Admin feedback list failed: ${feedbackListResponse.status}`);
  }
  const feedbackData = await feedbackListResponse.json() as { feedback: unknown[] };
  if (!Array.isArray(feedbackData.feedback)) {
    throw new Error("Admin feedback list did not return an array");
  }
  console.info("[pass] admin-can-view-feedback");

  await updateFeedbackStatus(feedbackId, "reviewed", "smoke-admin");
  assertFeedbackStatus(feedbackId, "reviewed", "admin-can-update-feedback-status");
  console.info("[pass] admin-can-update-feedback-status");

  // Reviewer cannot access feedback
  await setRole("smoke-reviewer-no-feedback", "reviewer");
  const deniedFeedbackResponse = await fetch(`${baseUrl}/dev/feedback?userId=smoke-reviewer-no-feedback`);
  if (deniedFeedbackResponse.status !== 403) {
    throw new Error(`Expected 403 for reviewer feedback access, got ${deniedFeedbackResponse.status}`);
  }
  console.info("[pass] reviewer-denied-feedback-access");

  // New conversation resets context
  const resetResult = await chat("/new");
  assertIncludes(resetResult.text, ["New conversation started"], "new-conversation-command");
  const postResetResult = await chat(resetCaseText);
  assertIncludes(postResetResult.text, ["order"], "post-reset-starts-fresh-history");

  // Verify logs
  await assertLogContains(conversationLogPath, ["conversation.turn", "smoke-user", firstCaseText]);
  await assertLogContainsAny(llmRawLogPath, ["llm.response"]);
  await assertContextUsageLogged();
  console.info("[pass] logs-written");
}

async function assertHealthy(): Promise<void> {
  const response = await fetch(`${baseUrl}/health`);
  if (!response.ok) {
    throw new Error(`Health check failed: ${response.status}`);
  }
}

async function assertDevEventsStreamConnects(): Promise<void> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 5000);
  const response = await fetch(`${baseUrl}/dev/events?userId=smoke-events-user`, {
    signal: controller.signal,
  });

  try {
    if (!response.ok) {
      throw new Error(`Events stream failed: ${response.status}`);
    }
    if (response.body === null) {
      throw new Error("Events stream response had no body.");
    }

    const event = await readFirstSseData<ProgressEvent>(response.body);
    if (event.stage !== "events.connected") {
      throw new Error(`Expected events.connected progress event, got: ${JSON.stringify(event)}`);
    }
  } finally {
    clearTimeout(timeout);
    controller.abort();
  }

  console.info("[pass] dev-events-stream-connects");
}

async function assertDevChatPathTraversalDenied(): Promise<void> {
  const response = await fetch(`${baseUrl}/dev/chat/..%2F..%2Fpackage.json`);
  if (response.status !== 403) {
    throw new Error(`Expected dev static path traversal to return 403, got ${response.status}`);
  }
  console.info("[pass] dev-chat-path-traversal-denied");
}

async function chat(text: string, userId = "smoke-user"): Promise<ChatResult> {
  const response = await fetch(`${baseUrl}/dev/chat`, {
    method: "POST",
    headers: {
      "content-type": "application/json; charset=utf-8"
    },
    body: JSON.stringify({
      userId,
      text
    })
  });

  if (!response.ok) {
    throw new Error(`Chat request failed: ${response.status} ${await response.text()}`);
  }

  // Parse SSE response
  const body = await response.text();
  let finalText = "";
  let sessionId: string | undefined;
  const toolCalls: string[] = [];

  for (const line of body.split("\n")) {
    if (line.startsWith("data: ")) {
      try {
        const event = JSON.parse(line.slice(6)) as SseEvent;
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

  return { text: finalText, ...(sessionId !== undefined ? { sessionId } : {}), toolCalls };
}

async function setRole(userId: string, role: string): Promise<void> {
  const response = await fetch(`${baseUrl}/dev/role`, {
    method: "POST",
    headers: {
      "content-type": "application/json; charset=utf-8"
    },
    body: JSON.stringify({
      userId,
      role
    })
  });

  if (!response.ok) {
    throw new Error(`Role request failed: ${response.status} ${await response.text()}`);
  }
}

async function updateFeedbackStatus(id: number, status: "reviewed" | "resolved", userId: string): Promise<void> {
  const response = await fetch(`${baseUrl}/dev/feedback/${id}`, {
    method: "PATCH",
    headers: {
      "content-type": "application/json; charset=utf-8"
    },
    body: JSON.stringify({
      userId,
      status
    })
  });

  if (!response.ok) {
    throw new Error(`Feedback status update failed: ${response.status} ${await response.text()}`);
  }
}

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

function assertIncludes(text: string, expectedValues: readonly string[], caseName: string): void {
  const normalizedText = text.toLowerCase();
  for (const expected of expectedValues) {
    if (!normalizedText.includes(expected.toLowerCase())) {
      throw new Error(`Smoke case ${caseName} expected response to include ${expected}. Response: ${text}`);
    }
  }
}

function assertForbidden(text: string, forbiddenValues: readonly string[], caseName: string): void {
  const normalizedText = text.toLowerCase();
  for (const forbidden of forbiddenValues) {
    if (normalizedText.includes(forbidden.toLowerCase())) {
      throw new Error(`Smoke case ${caseName} response included forbidden text ${forbidden}. Response: ${text}`);
    }
  }
}

async function assertLogContains(filePath: string, expectedValues: readonly string[]): Promise<void> {
  const content = await readFile(filePath, "utf8");

  for (const expected of expectedValues) {
    if (!content.includes(expected)) {
      throw new Error(`Expected ${filePath} to include ${expected}.`);
    }
  }
}

async function assertLogContainsAny(filePath: string, expectedValues: readonly string[]): Promise<void> {
  const content = await readFile(filePath, "utf8");

  if (!expectedValues.some((expected) => content.includes(expected))) {
    throw new Error(`Expected ${filePath} to include one of: ${expectedValues.join(", ")}.`);
  }
}

async function assertContextUsageLogged(): Promise<void> {
  const content = await readFile(systemLogPath, "utf8");
  const usageEvents = content
    .split(/\r?\n/)
    .filter((line) => line.includes('"type":"context.usage"') && line.includes('"userId":"smoke-user"'));

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
    typeof latest.inputTokens !== "number"
    || typeof latest.outputTokens !== "number"
    || typeof latest.contextWindow !== "number"
    || typeof latest.usagePercent !== "number"
  ) {
    throw new Error(`context.usage event is missing numeric token fields: ${JSON.stringify(latest)}.`);
  }
}

async function assertNoLlmResponseForUser(userId: string): Promise<void> {
  const content = await readFile(llmRawLogPath, "utf8");

  if (content.split(/\r?\n/).some((line) => line.includes(`"userId":"${userId}"`))) {
    throw new Error(`Expected no LLM response for denied reviewer mutation user ${userId}.`);
  }
}

function assertFeedbackRecorded(userId: string, text: string, intentType: string): number {
  const db = new Database(dbPath, { readonly: true });
  try {
    const row = db.prepare(`
      SELECT id, user_message, intent_type, status
      FROM feedback
      WHERE user_id = ?
      ORDER BY id DESC
      LIMIT 1
    `).get(userId) as { readonly id: number; readonly user_message: string; readonly intent_type: string; readonly status: string } | undefined;

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

function assertFeedbackStatus(id: number, expectedStatus: string, caseName: string): void {
  const db = new Database(dbPath, { readonly: true });
  try {
    const row = db.prepare(`
      SELECT status
      FROM feedback
      WHERE id = ?
    `).get(id) as { readonly status: string } | undefined;

    if (row === undefined) {
      throw new Error(`[${caseName}] Expected feedback row ${id}.`);
    }
    if (row.status !== expectedStatus) {
      throw new Error(`[${caseName}] Expected feedback ${id} status ${expectedStatus}, got ${row.status}.`);
    }
  } finally {
    db.close();
  }
}

async function assertPiAiComplete(): Promise<void> {
  const { complete } = await import("@earendil-works/pi-ai");
  const { buildPiModel } = await import("../config/llmConfig.js");

  const resolved = config.llm;
  const model = buildPiModel(resolved);

  const response = await complete(model, {
    systemPrompt: "Respond with exactly one word.",
    messages: [{ role: "user", content: "What color is the sky?", timestamp: Date.now() }],
  }, {
    apiKey: resolved.apiKey,
  });

  const text = response.content
    .filter((b): b is Extract<typeof b, { type: "text" }> => b.type === "text")
    .map((b) => b.text)
    .join("")
    .trim()
    .toLowerCase();

  if (!text.includes("blue")) {
    throw new Error(`pi-ai smoke: expected "blue" in response, got: ${text}`);
  }

  console.info("[pass] pi-ai-complete");
}

function assertFeedbackTimestampLocal(userId: string, caseName: string): void {
  const db = new Database(dbPath, { readonly: true });
  try {
    const row = db.prepare(`
      SELECT created_at
      FROM feedback
      WHERE user_id = ?
      ORDER BY id DESC
      LIMIT 1
    `).get(userId) as { readonly created_at: string } | undefined;

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
        `[${caseName}] created_at (${createdAt}) is more than 5 min from local time (${localNow.toISOString()}). Diff: ${diffMs}ms`,
      );
    }
  } finally {
    db.close();
  }
}

await main();
