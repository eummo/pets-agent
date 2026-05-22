import type { IntentDetectionService, UserIntent, UserRole } from "../core/ports.js";
import type { ResolvedLlmConfig } from "../config/llmConfig.js";

const INTENT_PROMPT = `You are an intent classifier for a knowledge-base assistant.
Given a user message and their current role, classify the intent into exactly one of:

- "query": The user is asking a question, searching for information, or requesting an explanation.
- "mutate": The user wants to modify, create, delete, or update code/files in the workspace.
- "update_kb": The user wants to update, add, or modify knowledge-base content (not code).

Important:
- Do not grant or deny permission. Only classify the user's intent.
- If the user asks to add, change, refactor, implement, delete, or continue implementation work, classify as "mutate".
- If the user asks to update documentation or knowledge-base content, classify as "update_kb".

Examples:
- "What is the current architecture?" -> query
- "Please update the documentation with the latest order lifecycle." -> update_kb
- "I want to modify the order system." -> mutate
- "我想修改订单系统" -> mutate
- "添加新的订单功能，增加下单" -> mutate
- "更新知识库里的订单流程" -> update_kb

Respond with ONLY the intent label, nothing else.

User role: {role}
User message: {message}`;

const INTENT_TIMEOUT_MS = 5000;
const INTENT_MAX_TOKENS = 256;

const VALID_INTENTS = new Set<string>(["query", "mutate", "update_kb"]);

export class LlmIntentDetectionService implements IntentDetectionService {
  public constructor(private readonly config: ResolvedLlmConfig) {}

  public async detectIntent(userMessage: string, role: UserRole): Promise<UserIntent> {
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
          max_tokens: INTENT_MAX_TOKENS,
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
      const text = content
        ?.map((block) => block["text"])
        .find((value): value is string => typeof value === "string");
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
