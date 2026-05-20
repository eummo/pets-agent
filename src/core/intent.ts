export type MessageIntent = "read" | "mutate";

const mutatePatterns = [
  /\b(refactor|rewrite|implement|modify|change|fix|update|delete|create|add|remove)\b/i,
  /重构|实现|修改|修复|更新|删除|创建|新增|调整|改造|优化/
];

export function classifyMessageIntent(text: string): MessageIntent {
  return mutatePatterns.some((pattern) => pattern.test(text)) ? "mutate" : "read";
}
