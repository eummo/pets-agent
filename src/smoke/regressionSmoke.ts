import { readFile } from "node:fs/promises";
import path from "node:path";
import Database from "better-sqlite3";

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

const baseUrl = process.env["SMOKE_BASE_URL"] ?? "http://127.0.0.1:3000";
const conversationLogPath =
  process.env["CONVERSATION_LOG_PATH"] ?? path.resolve(".harness", "logs", "conversation.jsonl");
const llmRawLogPath = process.env["LLM_RAW_LOG_PATH"] ?? path.resolve(".harness", "logs", "llm-raw.jsonl");
const dbPath = process.env["DB_PATH"] ?? path.resolve(".harness", "state", "agent.db");
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
    expectedIncludes: ["catalog", "order lifecycle"],
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
  assertFeedbackRecorded(feedbackUserId, feedbackText, "update_kb");
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

  // New conversation resets context
  const resetResult = await chat("/new");
  assertIncludes(resetResult.text, ["New conversation started"], "new-conversation-command");
  const postResetResult = await chat(resetCaseText);
  assertIncludes(postResetResult.text, ["order"], "post-reset-starts-fresh-history");

  // Verify logs
  await assertLogContains(conversationLogPath, ["conversation.turn", "smoke-user", firstCaseText]);
  await assertLogContainsAny(llmRawLogPath, ["llm.response"]);
  console.info("[pass] logs-written");
}

async function assertHealthy(): Promise<void> {
  const response = await fetch(`${baseUrl}/health`);
  if (!response.ok) {
    throw new Error(`Health check failed: ${response.status}`);
  }
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

async function assertNoLlmResponseForUser(userId: string): Promise<void> {
  const content = await readFile(llmRawLogPath, "utf8");

  if (content.split(/\r?\n/).some((line) => line.includes(`"userId":"${userId}"`))) {
    throw new Error(`Expected no LLM response for denied reviewer mutation user ${userId}.`);
  }
}

function assertFeedbackRecorded(userId: string, text: string, intentType: string): void {
  const db = new Database(dbPath, { readonly: true });
  try {
    const row = db.prepare(`
      SELECT user_message, intent_type, status
      FROM feedback
      WHERE user_id = ?
      ORDER BY id DESC
      LIMIT 1
    `).get(userId) as { readonly user_message: string; readonly intent_type: string; readonly status: string } | undefined;

    if (row === undefined) {
      throw new Error(`Expected feedback row for ${userId}.`);
    }
    if (row.user_message !== text || row.intent_type !== intentType || row.status !== "pending") {
      throw new Error(`Unexpected feedback row: ${JSON.stringify(row)}.`);
    }
  } finally {
    db.close();
  }
}

await main();
