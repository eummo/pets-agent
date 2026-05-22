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
  const updateKnowledgeBaseKeywords = [
    "update the documentation",
    "update documentation",
    "update the knowledge base",
    "update knowledge base",
    "add to the knowledge base",
    "更新知识库",
    "修改知识库",
    "补充知识库",
    "更新文档",
    "修改文档",
    "补充文档",
  ];
  if (updateKnowledgeBaseKeywords.some((keyword) => normalized.includes(keyword))) {
    return { type: "update_kb" };
  }

  const mutationKeywords = [
    "fix the bug",
    "modify the file",
    "edit the file",
    "write code",
    "change the code",
    "add a comment",
    "修改代码",
    "修复",
    "改一下",
    "写代码",
  ];
  if (mutationKeywords.some((keyword) => normalized.includes(keyword))) {
    return { type: "mutate" };
  }

  return undefined;
}
