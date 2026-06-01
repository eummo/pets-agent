import type { AgentConversationMessage, UserRole } from "../core/index.js";

export type UserIntent =
  | { readonly type: "query" }
  | { readonly type: "suggest" }
  | { readonly type: "mutate" }
  | { readonly type: "update_kb" };

export type IntentDetectionService = {
  detectIntent(
    userMessage: string,
    role: UserRole,
    history?: readonly AgentConversationMessage[]
  ): Promise<UserIntent>;
};

const VALID_INTENT_LABELS = new Set<string>(["query", "mutate", "update_kb"]);

export function isValidIntentType(label: string): label is UserIntent["type"] {
  return VALID_INTENT_LABELS.has(label);
}
