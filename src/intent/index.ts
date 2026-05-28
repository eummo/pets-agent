export type UserIntent =
  | { readonly type: "query" }
  | { readonly type: "suggest" }
  | { readonly type: "mutate" }
  | { readonly type: "update_kb" };

const VALID_INTENT_LABELS = new Set<string>(["query", "mutate", "update_kb"]);

export function isValidIntentType(label: string): label is UserIntent["type"] {
  return VALID_INTENT_LABELS.has(label);
}
