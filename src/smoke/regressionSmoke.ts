import { readFile } from "node:fs/promises";
import path from "node:path";

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

async function setRole(userId: string, role: "developer" | "reviewer"): Promise<void> {
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

await main();
