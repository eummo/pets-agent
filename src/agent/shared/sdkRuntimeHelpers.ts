/**
 * @fileoverview Shared helpers for SDK-based agent runtimes.
 *
 * Claude SDK and Codebuddy SDK runtimes share identical serialization, error
 * formatting, and context-usage extraction logic. This module provides a single
 * source of truth so both adapters stay in sync.
 */

import { isRecord, stringField, formatUnknownError } from "../../core/unknownRecord.js";
import type { ContextUsageReport } from "../index.js";

export { formatUnknownError };

// ── Serialization Helpers ────────────────────────────────────────────────────

const SERIALIZABLE_QUERY_KEYS: readonly string[] = [
  "cwd",
  "tools",
  "allowedTools",
  "disallowedTools",
  "permissionMode",
  "allowDangerouslySkipPermissions",
  "systemPrompt",
  "includePartialMessages",
  "maxTurns",
  "model",
  "resume",
  "settings",
  "skills",
  "settingSources"
];

export function serializeQueryOptions(
  queryOptions: Record<string, unknown>
): Record<string, unknown> {
  return Object.fromEntries(
    SERIALIZABLE_QUERY_KEYS.filter((key) => queryOptions[key] !== undefined).map((key) => [
      key,
      queryOptions[key]
    ])
  );
}

export function serializeSdkResult(
  result: Record<string, unknown> | undefined
): Record<string, unknown> | undefined {
  if (result === undefined) return undefined;

  return {
    subtype: result["subtype"],
    sessionId: result["session_id"],
    result: result["result"],
    errors: result["errors"],
    usage: result["usage"]
  };
}

// ── Text Extraction ──────────────────────────────────────────────────────────

export function extractToolResultText(block: Record<string, unknown>): string {
  const content = block["content"];
  if (!Array.isArray(content)) return "";

  return content
    .map((part: unknown) => {
      if (typeof part === "string") return part;
      if (!isRecord(part)) return "";
      return stringField(part, "text") ?? "";
    })
    .join("");
}

// ── Context Usage ────────────────────────────────────────────────────────────

export function extractContextUsage(
  usage: unknown,
  contextWindow: number
): ContextUsageReport | undefined {
  if (!isRecord(usage)) return undefined;

  const inputTokens = usage["input_tokens"];
  const outputTokens = usage["output_tokens"];
  if (typeof inputTokens !== "number" || typeof outputTokens !== "number") return undefined;

  const cacheReadTokens = usage["cache_read_input_tokens"];
  const cacheCreationTokens = usage["cache_creation_input_tokens"];
  const usagePercent = contextWindow > 0 ? Math.round((inputTokens / contextWindow) * 100) : 0;

  return {
    inputTokens,
    outputTokens,
    ...(typeof cacheReadTokens === "number" ? { cacheReadTokens } : {}),
    ...(typeof cacheCreationTokens === "number" ? { cacheCreationTokens } : {}),
    contextWindow,
    usagePercent
  };
}
