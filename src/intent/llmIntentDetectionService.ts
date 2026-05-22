import type { IntentDetectionService, UserIntent, UserRole } from "../core/ports.js";
import type { ResolvedLlmConfig } from "../config/llmConfig.js";

const INTENT_PROMPT = `You are an intent classifier for a knowledge-base assistant.
Given a user message and their current role, classify the intent into exactly one of:

- "query": The user is asking a question, searching for information, or requesting an explanation.
- "mutate": The user wants to modify, create, delete, or update code/files in the workspace.
- "update_kb": The user wants to update, add, or modify knowledge-base content (not code).

Respond with ONLY the intent label, nothing else.

User role: {role}
User message: {message}`;

const INTENT_TIMEOUT_MS = 5000;

const VALID_INTENTS = new Set<string>(["query", "mutate", "update_kb"]);

const UPDATE_KB_KEYWORDS = [
  "update the documentation",
  "update documentation",
  "update the knowledge base",
  "update knowledge base",
  "add to the knowledge base",
  "\u66f4\u65b0\u77e5\u8bc6\u5e93",
  "\u4fee\u6539\u77e5\u8bc6\u5e93",
  "\u8865\u5145\u77e5\u8bc6\u5e93",
  "\u66f4\u65b0\u6587\u6863",
  "\u4fee\u6539\u6587\u6863",
  "\u8865\u5145\u6587\u6863",
] as const;

const MUTATION_KEYWORDS = [
  "fix the bug",
  "modify the file",
  "edit the file",
  "write code",
  "change the code",
  "add a comment",
  "\u4fee\u6539\u4ee3\u7801",
  "\u4fee\u590d",
  "\u6539\u4e00\u4e0b",
  "\u5199\u4ee3\u7801",
  "\u91cd\u6784",
  "\u5b9e\u73b0",
  "\u5f00\u53d1",
  "\u65b0\u589e",
  "\u6dfb\u52a0",
  "\u589e\u52a0",
  "\u5220\u9664",
] as const;

const CHINESE_MUTATION_VERBS = /[\u4fee\u6539\u589e\u52a0\u6dfb\u5220\u91cd\u5b9e\u5f00]/u;

export class LlmIntentDetectionService implements IntentDetectionService {
  public constructor(private readonly config: ResolvedLlmConfig) {}

  public async detectIntent(userMessage: string, role: UserRole): Promise<UserIntent> {
    const deterministicIntent = detectDeterministicIntent(userMessage);
    if (deterministicIntent !== undefined) {
      return deterministicIntent;
    }

    try {
      const prompt = INTENT_PROMPT.replace("{role}", role).replace("{message}", userMessage);

      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), INTENT_TIMEOUT_MS);

      const baseUrl = this.config.baseUrl.replace(/\/+$/, "");
      const response = await fetch(`${baseUrl}/v1/messages`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "x-api-key": this.config.apiKey,
          "anthropic-version": "2023-06-01",
        },
        body: JSON.stringify({
          model: this.config.modelId,
          max_tokens: 10,
          messages: [{ role: "user", content: prompt }],
        }),
        signal: controller.signal,
      });

      clearTimeout(timeout);

      if (!response.ok) {
        return { type: "query" };
      }

      const data = await response.json() as Record<string, unknown>;
      const content = data["content"] as Record<string, unknown>[] | undefined;
      const text = content?.[0]?.["text"] as string | undefined;
      const label = text?.trim().toLowerCase() ?? "";

      if (VALID_INTENTS.has(label)) {
        return { type: label as UserIntent["type"] };
      }

      return { type: "query" };
    } catch {
      return { type: "query" };
    }
  }
}

function detectDeterministicIntent(userMessage: string): UserIntent | undefined {
  const normalized = userMessage.toLowerCase();

  if (UPDATE_KB_KEYWORDS.some((keyword) => normalized.includes(keyword))) {
    return { type: "update_kb" };
  }

  if (MUTATION_KEYWORDS.some((keyword) => normalized.includes(keyword))) {
    return { type: "mutate" };
  }

  if (CHINESE_MUTATION_VERBS.test(userMessage)) {
    return { type: "mutate" };
  }

  return undefined;
}
