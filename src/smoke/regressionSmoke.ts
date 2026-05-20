import { readFile } from "node:fs/promises";
import path from "node:path";

type ChatResponse = {
  readonly text: string;
};

type SmokeCase = {
  readonly name: string;
  readonly text: string;
  readonly expectedIncludes: readonly string[];
  readonly forbiddenIncludes?: readonly string[];
};

const baseUrl = process.env["SMOKE_BASE_URL"] ?? "http://127.0.0.1:3000";
const conversationLogPath =
  process.env["CONVERSATION_LOG_PATH"] ?? path.resolve(".harness", "logs", "conversation.jsonl");
const llmRawLogPath = process.env["LLM_RAW_LOG_PATH"] ?? path.resolve(".harness", "logs", "llm-raw.jsonl");
const firstCaseText = "What is the current project for?";
const resetCaseText = "What is the current project after reset?";

const cases: readonly SmokeCase[] = [
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
    name: "viewer-mutate-request-is-blocked",
    text: "Refactor the order system",
    expectedIncludes: ["修改请求", "不能直接修改文件"],
    forbiddenIncludes: ["I can help design", "would need access"]
  },
  {
    name: "follow-up-reuses-session-context",
    text: "For the service you just mentioned, what is its main responsibility?",
    expectedIncludes: ["order"],
    forbiddenIncludes: ["WeChat", "browser", "agent runtime", "model provider"]
  }
];

async function main(): Promise<void> {
  await assertHealthy();

  for (const smokeCase of cases) {
    const response = await chat(smokeCase.text);
    assertIncludes(response.text, smokeCase.expectedIncludes, smokeCase.name);
    assertForbidden(response.text, smokeCase.forbiddenIncludes ?? [], smokeCase.name);
    console.info(`[pass] ${smokeCase.name}`);
  }

  await setRole("smoke-developer", "developer");
  const mutationResponse = await chat("重构订单系统", "smoke-developer");
  assertIncludes(mutationResponse.text, ["Claude/Anthropic SDK", "代码变更流程", "通过"], "developer-mutate-runs-code-change");
  console.info("[pass] developer-mutate-runs-code-change");

  const resetResponse = await chat("/new");
  assertIncludes(resetResponse.text, ["New conversation started"], "new-conversation-command");
  const postResetResponse = await chat(resetCaseText);
  assertIncludes(postResetResponse.text, ["order"], "post-reset-starts-fresh-history");

  await assertLogContains(conversationLogPath, ["conversation.turn", "smoke-user", firstCaseText]);
  await assertLogContainsAny(llmRawLogPath, ["llm.request", "llm.session.request"]);
  await assertLogContainsAny(llmRawLogPath, ["llm.response", "llm.session.response"]);
  await assertLogContains(llmRawLogPath, ["code_change.request", "code_change.raw_response"]);
  await assertMessagesRequestContainsHistory(llmRawLogPath);
  await assertMessagesRequestAfterResetStartsFresh(llmRawLogPath);
  console.info("[pass] logs-written");
}

async function assertHealthy(): Promise<void> {
  const response = await fetch(`${baseUrl}/health`);
  if (!response.ok) {
    throw new Error(`Health check failed: ${response.status}`);
  }
}

async function chat(text: string, userId = "smoke-user"): Promise<ChatResponse> {
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

  return (await response.json()) as ChatResponse;
}

async function setRole(userId: string, role: "developer" | "viewer"): Promise<void> {
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

async function assertMessagesRequestContainsHistory(filePath: string): Promise<void> {
  const content = await readFile(filePath, "utf8");
  const requestEvents = content
    .trim()
    .split(/\r?\n/)
    .map((line) => JSON.parse(line) as { readonly type?: string; readonly request?: { readonly messages?: readonly unknown[] } })
    .filter((event) => event.type === "llm.request");

  if (
    requestEvents.length > 1 &&
    !requestEvents.some((event) => (event.request?.messages?.length ?? 0) > 1)
  ) {
    throw new Error("Expected at least one messages request to include prior conversation history.");
  }
}

async function assertMessagesRequestAfterResetStartsFresh(filePath: string): Promise<void> {
  const content = await readFile(filePath, "utf8");
  const requestEvents = content
    .trim()
    .split(/\r?\n/)
    .map((line) => JSON.parse(line) as { readonly type?: string; readonly request?: { readonly messages?: readonly { readonly content?: unknown }[] } })
    .filter((event) => event.type === "llm.request");
  const resetRequest = requestEvents.find((event) =>
    event.request?.messages?.some((message) => message.content === resetCaseText)
  );

  if (resetRequest !== undefined && resetRequest.request?.messages?.length !== 1) {
    throw new Error("Expected the first request after /new to start with a fresh local messages history.");
  }
}

await main();
