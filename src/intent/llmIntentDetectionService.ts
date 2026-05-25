import type { Api, Model } from "@earendil-works/pi-ai";
import { complete } from "@earendil-works/pi-ai";
import { withRetry } from "../config/retry.js";
import { fallbackIntentFor } from "../core/intentHeuristics.js";
import type { AgentConversationMessage, IntentDetectionService, UserIntent, UserRole } from "../core/contracts.js";

const INTENT_SYSTEM_PROMPT = `You are an intent classifier for a knowledge-base assistant.
Given a user message, conversation history (if any), and their current role, classify the intent into exactly one of:

- "query": The user is asking a question, searching for information, or requesting an explanation.
- "mutate": The user wants to modify, create, delete, or update code/files in the workspace.
- "update_kb": The user wants to update, add, or modify knowledge-base content (not code).

Classification rules:
- Use the conversation history to resolve ambiguous short messages.
  If the assistant just suggested adding/updating content and the user says something like
  "补充一下", "好的", "go ahead", "please do", classify as the same intent as the suggestion.
- When the user asks about something that contains modification-related words in its name
  (e.g., "更新日志", "修改记录", "changelog", "update log"), but is clearly asking
  for information, classify as "query".
- Do not grant or deny permission. Only classify the user's intent.
- If in doubt, classify as "query".

Examples (without history):
- "What is the current architecture?" -> query
- "更新日志是什么" -> query
- "修改记录怎么查看" -> query
- "我想修改订单系统" -> mutate
- "更新知识库里的订单流程" -> update_kb

Examples (with history):
- [assistant: "需要补充参数文档"] "补充一下" -> update_kb
- [assistant: "是否需要修改代码？"] "好的" -> mutate
- [assistant: "订单流程分为三步"] "补充一下" -> update_kb
- [assistant: "这是修改记录的内容"] "修改记录是什么" -> query

Respond with ONLY the intent label, nothing else.`;

const INTENT_TIMEOUT_MS = 5000;

const VALID_INTENTS = new Set<string>(["query", "mutate", "update_kb"]);

export class LlmIntentDetectionService implements IntentDetectionService {
  public constructor(
    private readonly model: Model<Api>,
    private readonly apiKey: string,
  ) {}

  public async detectIntent(userMessage: string, role: UserRole, history?: readonly AgentConversationMessage[]): Promise<UserIntent> {
    try {
      const response = await withRetry(async () => {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), INTENT_TIMEOUT_MS);

        return complete(this.model, {
          systemPrompt: INTENT_SYSTEM_PROMPT,
          messages: [{
            role: "user",
            content: buildIntentUserContent(userMessage, role, history),
            timestamp: Date.now(),
          }],
        }, {
          apiKey: this.apiKey,
          signal: controller.signal,
        }).finally(() => clearTimeout(timeout));
      });

      if (response.stopReason === "error") {
        return fallbackIntentFor(userMessage);
      }

      const text = response.content
        .filter((block): block is Extract<typeof block, { type: "text" }> => block.type === "text")
        .map((block) => block.text)
        .join("");
      const label = text.trim().toLowerCase();

      if (VALID_INTENTS.has(label)) {
        return { type: label as UserIntent["type"] };
      }

      return fallbackIntentFor(userMessage);
    } catch {
      return fallbackIntentFor(userMessage);
    }
  }
}

function buildIntentUserContent(userMessage: string, role: UserRole, history?: readonly AgentConversationMessage[]): string {
  if (history === undefined || history.length === 0) {
    return `User role: ${role}\nUser message: ${userMessage}`;
  }

  return [
    `User role: ${role}`,
    "",
    "Conversation history:",
    ...history.map((m) => `${m.role}: ${m.content}`),
    "",
    "Current user message:",
    userMessage,
  ].join("\n");
}

