import type { UserIntent } from "../intent/index.js";

const KNOWLEDGE_BASE_TERMS = [
  "documentation",
  "document",
  "docs",
  "knowledge base",
  "kb",
  "readme",
  "知识库",
  "文档",
  "说明"
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
  "修复"
] as const;

const INFORMATION_QUESTION_TERMS = [
  "what",
  "how",
  "why",
  "where",
  "when",
  "which",
  "explain",
  "describe",
  "tell me",
  "是什么",
  "怎么",
  "如何",
  "为什么",
  "哪里",
  "哪个",
  "介绍",
  "说明"
] as const;

const ATTACHMENT_REFERENCE_TERMS = [
  "uploaded",
  "upload",
  "attachment",
  "image",
  "file",
  "document",
  "上传",
  "附件",
  "图片",
  "文件",
  "文档"
] as const;

const ATTACHMENT_QUERY_TERMS = [
  "acknowledge",
  "describe",
  "summarize",
  "read",
  "extract",
  "what",
  "确认",
  "描述",
  "总结",
  "读取",
  "提取",
  "是什么"
] as const;

export function fallbackIntentFor(userMessage: string): UserIntent {
  const normalized = userMessage.trim().toLowerCase();

  if (containsAnyTerm(normalized, INFORMATION_QUESTION_TERMS)) {
    return { type: "query" };
  }

  if (!containsAnyTerm(normalized, UPDATE_TERMS)) {
    return { type: "query" };
  }

  if (containsAnyTerm(normalized, KNOWLEDGE_BASE_TERMS)) {
    return { type: "update_kb" };
  }

  return { type: "mutate" };
}

export function reconcileIntentWithHeuristics(userMessage: string, intent: UserIntent): UserIntent {
  if (intent.type === "query") {
    return intent;
  }

  const normalized = userMessage.trim().toLowerCase();
  if (
    containsAnyTerm(normalized, ATTACHMENT_REFERENCE_TERMS) &&
    containsAnyTerm(normalized, ATTACHMENT_QUERY_TERMS)
  ) {
    return { type: "query" };
  }

  return intent;
}

function containsAnyTerm(text: string, terms: readonly string[]): boolean {
  return terms.some((term) => text.includes(term));
}
