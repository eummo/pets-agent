export type UserIntent =
  | { readonly type: "query" }
  | { readonly type: "suggest" }
  | { readonly type: "mutate" }
  | { readonly type: "update_kb" };