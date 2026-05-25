import type { UserIntent } from "./contracts.js";

const KNOWLEDGE_BASE_TERMS = [
  "documentation",
  "document",
  "docs",
  "knowledge base",
  "kb",
  "readme",
  "知识库",
  "文档",
  "说明",
] as const;

const UPDATE_TERMS = [
  "update",
  "change",
  "modify",
  "edit",
  "add",
  "create",
  "delete",
  "remove",
  "write",
  "implement",
  "refactor",
  "fix",
  "更新",
  "修改",
  "编辑",
  "添加",
  "增加",
  "创建",
  "删除",
  "移除",
  "写入",
  "实现",
  "重构",
  "修复",
] as const;

export function fallbackIntentFor(userMessage: string): UserIntent {
  const normalized = userMessage.trim().toLowerCase();

  if (!containsAnyTerm(normalized, UPDATE_TERMS)) {
    return { type: "query" };
  }

  if (containsAnyTerm(normalized, KNOWLEDGE_BASE_TERMS)) {
    return { type: "update_kb" };
  }

  return { type: "mutate" };
}

function containsAnyTerm(text: string, terms: readonly string[]): boolean {
  return terms.some((term) => text.includes(term));
}

