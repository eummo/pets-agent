import { readFile } from "node:fs/promises";
import path from "node:path";
import Database from "better-sqlite3";
import "dotenv/config";
import { loadRuntimeConfig } from "../config/runtimeConfig.js";
import { isRecord, stringField } from "../core/unknownRecord.js";

type SseEvent = {
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

type ChatResult = {
  readonly text: string;
  readonly sessionId?: string;
  readonly toolCalls: readonly string[];
  readonly events: readonly SseEvent[];
};

type ProgressEvent = {
  readonly stage?: string;
  readonly message?: string;
};

const config = await loadRuntimeConfig();
const agentSdkType = config.agentSdk.type;
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
    forbiddenIncludes: ["FeedbackEntry", "sqliteFeedbackStore", "/dev/feedback", "src/", "contracts.ts", "代码分析"]
  }
] as const;

async function main(): Promise<void> {
  await assertHealthy();
  await assertDevEventsStreamConnects();
  await assertDevChatPathTraversalDenied();

  // Pi-ai smoke: verify complete() works with the configured Anthropic endpoint
  await assertPiAiComplete();

  // Reset session to ensure clean context for smoke cases
  await chat("/new");

  for (const smokeCase of cases) {
    const userId = "userId" in smokeCase ? smokeCase.userId : "smoke-user";
    if (userId !== "smoke-user") {
      await resetChat(userId);
    }
    const result = await chat(smokeCase.text, userId);
    assertIncludes(result.text, smokeCase.expectedIncludes, smokeCase.name);
    assertForbidden(result.text, smokeCase.forbiddenIncludes, smokeCase.name);
    console.info(`[pass] ${smokeCase.name}`);
  }

  // Reviewer mutation requests are denied and captured as feedback.
  const feedbackUserId = "smoke-feedback-user";
  await resetChat(feedbackUserId);
  const feedbackText = "Please update the documentation with the latest order lifecycle.";
  const feedbackResult = await chat(feedbackText, feedbackUserId);
  assertIncludes(feedbackResult.text, ["记录"], "reviewer-update-recorded-as-feedback");
  const feedbackId = assertFeedbackRecorded(feedbackUserId, feedbackText, "update_kb");
  assertFeedbackTimestampLocal(feedbackUserId, "reviewer-update-recorded-as-feedback");
  console.info("[pass] reviewer-update-recorded-as-feedback");

  const chineseMutationUserId = "smoke-reviewer-mutation-user";
  await resetChat(chineseMutationUserId);
  const chineseMutationText = "我想修改订单系统";
  const chineseMutationResult = await chat(chineseMutationText, chineseMutationUserId);
  assertIncludes(chineseMutationResult.text, ["修改请求", "记录"], "reviewer-chinese-mutation-denied");
  assertForbidden(chineseMutationResult.text, ["计划", "审批", "Express", "TypeScript"], "reviewer-chinese-mutation-denied");
  assertFeedbackRecorded(chineseMutationUserId, chineseMutationText, "mutate");
  await assertNoAgentRuntimeLlmResponseForUser(chineseMutationUserId);
  console.info("[pass] reviewer-chinese-mutation-denied");

  // Developer role can make code changes
  await setRole("smoke-developer", "developer");
  await resetChat("smoke-developer");
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
  assertForbidden(postResetResult.text, ["pet business", "pet-related", "pet application", "pet orders"], "post-reset-starts-fresh-history");

  // Verify logs
  await assertLogContains(conversationLogPath, ["conversation.turn", "smoke-user", firstCaseText]);
  await assertLogContains(llmRawLogPath, [
    '"type":"llm.request"',
    '"type":"llm.response"',
    '"operation":"intent_detection"',
    '"operation":"agent_runtime"',
    '"type":"intent.result"',
  ]);
  await assertLogContainsAny(llmRawLogPath, ['"type":"agent.tool_call"', '"type":"llm.compact"']);
  await assertContextUsageLogged();
  console.info("[pass] logs-written");

  // Verify skills and settingSources are passed to the Codebuddy SDK
  if (agentSdkType === "codebuddy") {
    await assertLogContains(llmRawLogPath, ['"skills":"all"', '"settingSources":["project","local"]']);
    console.info("[pass] sdk-options-include-skills-and-setting-sources");
  }

  // Verify .claude/rules content is loaded: ask a question that the rule should steer
  await resetChat("smoke-rules-user");
  const ruleResult = await chat("What is the scope of this workspace?", "smoke-rules-user");
  assertIncludes(ruleResult.text.toLowerCase(), ["workspace"], "workspace-rules-loaded");
  assertForbidden(ruleResult.text, ["agent runtime", "model provider", "WeChat"], "workspace-rules-loaded");
  console.info("[pass] workspace-rules-loaded");

  // Verify agent SDK type is reflected in runtime name logged by the orchestrator
  await assertAgentSdkRuntimeMatchesConfig();
  console.info("[pass] agent-sdk-runtime-matches-config");

  // ── Pi SDK-specific smoke tests ─────────────────────────────────────────────
  if (agentSdkType === "pi") {
    // Pi runtime forwards stream events (text_delta, thinking, tool_use_start)
    await assertPiStreamEvents();
    console.info("[pass] pi-agent-stream-events");

    // Pi runtime persists session context across turns
    await assertPiSessionPersistence();
    console.info("[pass] pi-agent-session-persistence");

    // Pi runtime logs tool calls with permission info in llm-raw.jsonl
    await assertPiToolPermissionLogged();
    console.info("[pass] pi-agent-tool-permission-logged");
  }

  // ── Codebuddy SDK-specific smoke tests ──────────────────────────────────────
  if (agentSdkType === "codebuddy") {
    // Codebuddy runtime logs SDK options in llm.request
    await assertCodebuddyOptionsLogged();
    console.info("[pass] codebuddy-agent-options-logged");

    // Codebuddy runtime supports session resume via resume option
    await assertCodebuddySessionResume();
    console.info("[pass] codebuddy-agent-session-resume");
  }
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

  return { text: finalText, ...(sessionId !== undefined ? { sessionId } : {}), toolCalls, events };
}

async function resetChat(userId: string): Promise<void> {
  await chat("/new", userId);
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

async function assertNoAgentRuntimeLlmResponseForUser(userId: string): Promise<void> {
  const content = await readFile(llmRawLogPath, "utf8");

  const hasAgentRuntimeResponse = content
    .split(/\r?\n/)
    .some((line) => (
      line.includes(`"userId":"${userId}"`)
      && line.includes('"type":"llm.response"')
      && line.includes('"operation":"agent_runtime"')
    ));

  if (hasAgentRuntimeResponse) {
    throw new Error(`Expected no agent runtime LLM response for denied reviewer mutation user ${userId}.`);
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
  // Verify the model endpoint is reachable via pi-ai (used internally by pi-coding-agent)
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

  console.info("[pass] pi-model-reachable");
}

// ── Agent SDK runtime matching ────────────────────────────────────────────────

async function assertAgentSdkRuntimeMatchesConfig(): Promise<void> {
  const runtimePrefix = RUNTIME_PREFIX_FOR_SDK_TYPE[agentSdkType];
  if (runtimePrefix === undefined) {
    throw new Error(`Unknown agentSdk.type: ${agentSdkType}`);
  }

  const runtimeNames = await readAgentRuntimeNames();
  if (runtimeNames.length === 0) {
    throw new Error(`Expected ${llmRawLogPath} to include at least one agent_runtime llm.request.`);
  }

  const expectedPrefix = `${runtimePrefix}-`;
  const unexpected = runtimeNames.filter((runtimeName) => !runtimeName.startsWith(expectedPrefix));
  if (unexpected.length > 0) {
    throw new Error(
      `Expected agentSdk.type=${agentSdkType} to use runtime prefix ${expectedPrefix}. Saw: ${[...new Set(runtimeNames)].join(", ")}.`,
    );
  }
}

const RUNTIME_PREFIX_FOR_SDK_TYPE: Record<string, string> = {
  claude: "claude-sdk",
  codebuddy: "codebuddy-sdk",
  pi: "pi",
};

async function readAgentRuntimeNames(): Promise<readonly string[]> {
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

// ── Pi SDK-specific assertions ────────────────────────────────────────────────

async function assertPiStreamEvents(): Promise<void> {
  // Ask a question that likely triggers text_delta stream events
  await resetChat("smoke-pi-stream-user");
  const result = await chat("What is the order catalog?", "smoke-pi-stream-user");

  const hasTextDelta = result.events.some((e) => e.type === "text_delta");
  if (!hasTextDelta) {
    throw new Error(`Pi stream events: expected at least one text_delta event. Got types: ${result.events.map((e) => e.type).join(", ")}`);
  }

  const hasCompleted = result.events.some((e) => e.type === "completed");
  if (!hasCompleted) {
    throw new Error("Pi stream events: expected a completed event.");
  }

  if (result.text.length === 0) {
    throw new Error("Pi stream events: expected non-empty final text.");
  }
}

async function assertPiSessionPersistence(): Promise<void> {
  await resetChat("smoke-pi-session-user");

  // First message should return a sessionId
  const first = await chat("What is the order lifecycle?", "smoke-pi-session-user");
  if (first.sessionId === undefined) {
    throw new Error("Pi session persistence: first chat did not return a sessionId.");
  }

  // Follow-up with the same user should reuse the session
  const second = await chat("Tell me more about it.", "smoke-pi-session-user");
  if (second.sessionId === undefined) {
    throw new Error("Pi session persistence: second chat did not return a sessionId.");
  }

  // Verify the llm-raw log contains agent_runtime responses with the runtime name prefix
  const content = await readFile(llmRawLogPath, "utf8");
  const piRuntimeLines = content
    .split(/\r?\n/)
    .filter((line) => line.includes('"runtime":"pi-') && line.includes('"userId":"smoke-pi-session-user"'));

  if (piRuntimeLines.length === 0) {
    throw new Error("Pi session persistence: no pi runtime log entries found for smoke-pi-session-user.");
  }
}

async function assertPiToolPermissionLogged(): Promise<void> {
  // Use a developer role to trigger tool calls, then check logs
  await setRole("smoke-pi-tool-user", "developer");
  await resetChat("smoke-pi-tool-user");
  const previousContent = await readFile(llmRawLogPath, "utf8");
  await chat("Read the README file", "smoke-pi-tool-user");

  // Check llm-raw.jsonl for agent.tool_call entries with the pi runtime prefix
  const content = await readFile(llmRawLogPath, "utf8");
  const newContent = content.startsWith(previousContent) ? content.slice(previousContent.length) : content;
  const toolCallLines = newContent
    .split(/\r?\n/)
    .filter((line) =>
      line.includes('"type":"agent.tool_call"')
      && line.includes('"runtime":"pi-')
      && line.includes('"userId":"smoke-pi-tool-user"'),
    );

  // If the model made tool calls, they should be logged with permission info
  if (toolCallLines.length > 0) {
    const firstCall = JSON.parse(toolCallLines[0] ?? "{}") as { readonly permittedByRole?: unknown };
    if (firstCall.permittedByRole === undefined) {
      throw new Error("Pi tool permission: agent.tool_call log missing permittedByRole field.");
    }
  }

  // Also verify the runtime name prefix is correct in the agent_runtime llm.response
  const runtimeResponseLines = newContent
    .split(/\r?\n/)
    .filter((line) =>
      line.includes('"type":"llm.response"')
      && line.includes('"operation":"agent_runtime"')
      && line.includes('"runtime":"pi-')
      && line.includes('"userId":"smoke-pi-tool-user"'),
    );

  if (runtimeResponseLines.length === 0) {
    throw new Error("Pi tool permission: no agent_runtime llm.response found for smoke-pi-tool-user.");
  }
}

// ── Codebuddy SDK-specific assertions ─────────────────────────────────────────

async function assertCodebuddyOptionsLogged(): Promise<void> {
  // Check that llm.request logs contain codebuddy-specific fields
  const content = await readFile(llmRawLogPath, "utf8");
  const requestLines = content
    .split(/\r?\n/)
    .filter((line) =>
      line.includes('"type":"llm.request"')
      && line.includes('"operation":"agent_runtime"')
      && line.includes('"runtime":"codebuddy-sdk-'),
    );

  if (requestLines.length === 0) {
    throw new Error("Codebuddy options: no agent_runtime llm.request found with codebuddy-sdk runtime prefix.");
  }

  const firstRequest = JSON.parse(requestLines[0] ?? "{}") as {
    readonly options?: Record<string, unknown>;
  };

  if (firstRequest.options === undefined) {
    throw new Error("Codebuddy options: llm.request missing options field.");
  }

  // Codebuddy SDK should log skills and settingSources in its query options
  if (firstRequest.options["skills"] === undefined && firstRequest.options["settingSources"] === undefined) {
    throw new Error(`Codebuddy options: expected skills or settingSources in options. Got: ${JSON.stringify(Object.keys(firstRequest.options))}`);
  }
}

async function assertCodebuddySessionResume(): Promise<void> {
  await resetChat("smoke-codebuddy-session-user");

  // First chat returns a sessionId
  const first = await chat("What is the order catalog?", "smoke-codebuddy-session-user");
  if (first.sessionId === undefined) {
    throw new Error("Codebuddy session resume: first chat did not return a sessionId.");
  }

  // Follow-up should also succeed (session resume handled by the SDK)
  const second = await chat("Tell me more about the catalog.", "smoke-codebuddy-session-user");
  if (second.text.length === 0) {
    throw new Error("Codebuddy session resume: second chat returned empty text.");
  }

  // Verify llm-raw logs contain the codebuddy-sdk runtime prefix for this user
  const content = await readFile(llmRawLogPath, "utf8");
  const responseLines = content
    .split(/\r?\n/)
    .filter((line) =>
      line.includes('"type":"llm.response"')
      && line.includes('"operation":"agent_runtime"')
      && line.includes('"runtime":"codebuddy-sdk-')
      && line.includes('"userId":"smoke-codebuddy-session-user"'),
    );

  if (responseLines.length === 0) {
    throw new Error("Codebuddy session resume: no codebuddy-sdk agent_runtime responses found for smoke-codebuddy-session-user.");
  }
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
