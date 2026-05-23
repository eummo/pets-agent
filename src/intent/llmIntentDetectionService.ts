import type { Model } from "@earendil-works/pi-ai";
import { complete } from "@earendil-works/pi-ai";
import type { IntentDetectionService, UserIntent, UserRole } from "../core/ports.js";

const INTENT_SYSTEM_PROMPT = `You are an intent classifier for a knowledge-base assistant.
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

Respond with ONLY the intent label, nothing else.`;

const INTENT_TIMEOUT_MS = 5000;

const VALID_INTENTS = new Set<string>(["query", "mutate", "update_kb"]);

export class LlmIntentDetectionService implements IntentDetectionService {
  public constructor(
    private readonly model: Model<"anthropic-messages">,
    private readonly apiKey: string,
  ) {}

  public async detectIntent(userMessage: string, role: UserRole): Promise<UserIntent> {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), INTENT_TIMEOUT_MS);

      const response = await complete(this.model, {
        systemPrompt: INTENT_SYSTEM_PROMPT,
        messages: [{
          role: "user",
          content: `User role: ${role}\nUser message: ${userMessage}`,
          timestamp: Date.now(),
        }],
      }, {
        apiKey: this.apiKey,
        signal: controller.signal,
      }).finally(() => clearTimeout(timeout));

      if (response.stopReason === "error") {
        return { type: "query" };
      }

      const text = response.content
        .filter((block): block is Extract<typeof block, { type: "text" }> => block.type === "text")
        .map((block) => block.text)
        .join("");
      const label = text.trim().toLowerCase();

      if (VALID_INTENTS.has(label)) {
        return { type: label as UserIntent["type"] };
      }

      return { type: "query" };
    } catch {
      return { type: "query" };
    }
  }
}
