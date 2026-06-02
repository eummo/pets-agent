/**
 * @fileoverview Shared helpers for SDK-based agent runtimes.
 *
 * Claude SDK and Codebuddy SDK runtimes share identical serialization, error
 * formatting, context-usage extraction, query option construction, result
 * parsing, and compact event logging. This module provides a single source of
 * truth so both adapters stay in sync.
 */

import {
  isRecord,
  stringArrayField,
  stringField,
  formatUnknownError
} from "../../core/unknownRecord.js";
import {
  autoAllowedToolsForRole,
  availableToolsForRole,
  disallowedToolsForRole
} from "../../auth/index.js";
import type { AgentRequest, ContextUsageReport } from "../index.js";
import type { StoredRoleConfig } from "../../auth/index.js";
import type { ContextConfig } from "../../config/runtimeConfig.js";
import type { JsonlLogger } from "../../logging/jsonlLogger.js";
import type { CompactBoundaryData } from "./sdkMessageMapper.js";

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
  "settingSources",
  "planModeInstructions",
  "environment"
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

// ── Query Options Construction ───────────────────────────────────────────────

export type SdkQueryOptionsInput = {
  readonly request: AgentRequest;
  readonly roleConfig: StoredRoleConfig;
  readonly contextConfig: ContextConfig;
  readonly model: string | undefined;
  readonly canUseTool: (toolName: string, input: Record<string, unknown>) => Promise<unknown>;
};

export function buildSdkQueryOptions(input: SdkQueryOptionsInput): Record<string, unknown> {
  const { request, roleConfig, contextConfig, model, canUseTool } = input;

  const queryOptions: Record<string, unknown> = {
    cwd: request.workspacePath,
    tools: availableToolsForRole(roleConfig),
    allowedTools: autoAllowedToolsForRole(roleConfig),
    disallowedTools: disallowedToolsForRole(roleConfig),
    permissionMode: roleConfig.permissionMode,
    allowDangerouslySkipPermissions: roleConfig.permissionMode === "bypassPermissions",
    systemPrompt: roleConfig.systemPrompt,
    includePartialMessages: true,
    canUseTool
  };

  if (roleConfig.maxTurns !== undefined) {
    queryOptions["maxTurns"] = roleConfig.maxTurns;
  }
  if (model !== undefined) {
    queryOptions["model"] = model;
  }
  if (request.sessionId !== undefined) {
    queryOptions["resume"] = request.sessionId;
  }
  if (contextConfig.autoCompactEnabled) {
    queryOptions["settings"] = {
      autoCompactEnabled: true,
      autoCompactWindow: contextConfig.autoCompactWindow
    };
  }
  if (roleConfig.enableWorkflows === true) {
    const existing = queryOptions["settings"] as Record<string, unknown> | undefined;
    queryOptions["settings"] = {
      ...(existing ?? {}),
      enableWorkflows: true
    };
  }
  queryOptions["settingSources"] = roleConfig.settingSources ?? ["user", "project", "local"];
  if (roleConfig.skills !== undefined) {
    queryOptions["skills"] = roleConfig.skills;
  }
  if (roleConfig.planModeInstructions !== undefined) {
    queryOptions["planModeInstructions"] = roleConfig.planModeInstructions;
  }
  if (request.onCompact !== undefined) {
    queryOptions["hooks"] = {
      PostCompact: [
        {
          hooks: [
            async (hookInput: Record<string, unknown>) => {
              const summary = stringField(hookInput, "compact_summary");
              if (summary !== undefined) {
                await request.onCompact?.(summary);
              }
            }
          ]
        }
      ]
    };
  }

  return queryOptions;
}

// ── Result Message Parsing ──────────────────────────────────────────────────

export type SdkResultOutcome = {
  readonly sdkResult: Record<string, unknown>;
  readonly sessionId: string | undefined;
  readonly finalText: string;
  readonly contextUsage: ContextUsageReport | undefined;
};

export function handleSdkResultMessage(message: unknown, contextWindow: number): SdkResultOutcome {
  const resultData: Record<string, unknown> = isRecord(message) ? message : {};
  const sessionId = stringField(resultData, "session_id");
  const subtype = stringField(resultData, "subtype");
  let finalText: string;
  let contextUsage: ContextUsageReport | undefined;

  if (subtype === "success") {
    finalText = stringField(resultData, "result") ?? "";
    contextUsage = extractContextUsage(resultData["usage"], contextWindow);
  } else {
    const errors = stringArrayField(resultData, "errors");
    finalText = `Agent error: ${errors?.[0] ?? "Unknown error"}`;
    contextUsage = undefined;
  }

  return { sdkResult: resultData, sessionId, finalText, contextUsage };
}

// ── Compact Event Logging ───────────────────────────────────────────────────

export async function logCompactEvent(
  rawLogger: JsonlLogger | undefined,
  compactData: CompactBoundaryData,
  sessionId: string | undefined,
  runtimeName: string,
  request: AgentRequest
): Promise<void> {
  await rawLogger?.write({
    type: "llm.compact",
    runtime: runtimeName,
    userId: request.user.id,
    workspacePath: request.workspacePath,
    sessionId: compactData.sessionId ?? sessionId,
    trigger: compactData.trigger,
    preTokens: compactData.preTokens,
    ...(compactData.postTokens !== undefined ? { postTokens: compactData.postTokens } : {}),
    ...(compactData.durationMs !== undefined ? { durationMs: compactData.durationMs } : {})
  });
}
