/**
 * @fileoverview Shared message mapping logic for SDK-based agent runtimes.
 *
 * Claude SDK and Codebuddy SDK produce structurally identical message types.
 * This module provides provider-agnostic mapping functions that accept a
 * content array, so each adapter only needs to extract the content from its
 * provider-specific message type.
 */

import type { AgentRequest, AgentStreamEvent } from "../index.js";
import type { StoredRoleConfig } from "../../auth/index.js";
import type { JsonlLogger } from "../../logging/jsonlLogger.js";
import {
  arrayField,
  booleanField,
  isRecord,
  numberField,
  recordField,
  stringField
} from "../../core/unknownRecord.js";
import { canUseConfiguredTool } from "../policy/toolPolicy.js";
import { extractToolResultText } from "./sdkRuntimeHelpers.js";

// ── Assistant Message Forwarding ────────────────────────────────────────────

export function forwardAssistantContentEvents(
  content: readonly unknown[],
  request: AgentRequest,
  roleConfig: StoredRoleConfig
): void {
  for (const rawBlock of content) {
    if (!isRecord(rawBlock)) continue;

    const block = rawBlock;
    const blockType = stringField(block, "type");

    if (blockType === "tool_use") {
      const toolName = stringField(block, "name");
      const toolUseId = stringField(block, "id");
      if (toolName === undefined || toolUseId === undefined) continue;

      if (!canUseConfiguredTool(roleConfig, toolName)) {
        const message = `Tool ${toolName} is not permitted for role ${roleConfig.name}.`;
        request.stream?.({ type: "error", message });
        void request.progress?.({
          stage: "agent.error",
          message,
          data: { toolName }
        });
        continue;
      }

      const input = recordField(block, "input") ?? {};
      const toolEvent: AgentStreamEvent = {
        type: "tool_use_start",
        toolName,
        toolUseId,
        input
      };
      request.stream?.(toolEvent);
      void request.progress?.({
        stage: "agent.tool_use_start",
        message: `${toolName}: ${JSON.stringify(input).slice(0, 100)}`,
        data: { toolUseId, toolName }
      });
    } else if (blockType === "tool_result") {
      const toolUseId = stringField(block, "tool_use_id");
      if (toolUseId === undefined) continue;

      const isError = booleanField(block, "is_error") ?? false;
      const resultText = (arrayField(block, "content") ?? [])
        .map((p: unknown) => {
          if (typeof p === "string") return p;
          if (!isRecord(p)) return "";
          return stringField(p, "text") ?? "";
        })
        .join("");
      request.stream?.({
        type: "tool_use_result",
        toolUseId,
        result: resultText,
        ...(isError ? { isError: true } : {})
      });
    }
  }
}

// ── Stream Event Forwarding ────────────────────────────────────────────────

export function forwardStreamEvent(event: Record<string, unknown>, request: AgentRequest): void {
  const e = recordField(event, "event");
  if (e === undefined) return;

  const eventType = stringField(e, "type");

  if (eventType === "content_block_delta") {
    const delta = recordField(e, "delta");
    if (delta === undefined) return;

    const deltaType = stringField(delta, "type");
    const text = stringField(delta, "text");
    const thinking = stringField(delta, "thinking");
    if (deltaType === "text_delta" && text !== undefined) {
      request.stream?.({ type: "text_delta", text });
    } else if (deltaType === "thinking_delta" && thinking !== undefined) {
      request.stream?.({ type: "thinking", text: thinking });
    }
  }
}

// ── System Message Forwarding ──────────────────────────────────────────────

export type CompactBoundaryData = {
  readonly sessionId?: string;
  readonly trigger: "manual" | "auto";
  readonly preTokens: number;
  readonly postTokens?: number;
  readonly durationMs?: number;
};

export function forwardSystemContentEvents(
  data: Record<string, unknown>,
  request: AgentRequest
): CompactBoundaryData | undefined {
  const subtype = stringField(data, "subtype");

  if (subtype === "status") {
    const status = stringField(data, "status");
    if (status === "compacting") {
      request.stream?.({ type: "compact_start" });
    }
    return undefined;
  }

  if (subtype === "compact_boundary") {
    const metadata = recordField(data, "compact_metadata");
    if (metadata === undefined) return undefined;

    const trigger = stringField(metadata, "trigger");
    const preTokens = numberField(metadata, "pre_tokens");
    if ((trigger !== "manual" && trigger !== "auto") || preTokens === undefined) {
      return undefined;
    }

    const sessionId = stringField(data, "session_id");
    const postTokens = numberField(metadata, "post_tokens");
    const durationMs = numberField(metadata, "duration_ms");
    const compactData: CompactBoundaryData = {
      ...(sessionId !== undefined ? { sessionId } : {}),
      trigger,
      preTokens,
      ...(postTokens !== undefined ? { postTokens } : {}),
      ...(durationMs !== undefined ? { durationMs } : {})
    };

    request.stream?.({
      type: "compact_complete",
      preTokens: compactData.preTokens,
      ...(compactData.postTokens !== undefined ? { postTokens: compactData.postTokens } : {}),
      ...(compactData.durationMs !== undefined ? { durationMs: compactData.durationMs } : {})
    });

    return compactData;
  }

  return undefined;
}

// ── Tool Event Logging ─────────────────────────────────────────────────────

export async function logToolEventsFromContent(
  content: readonly unknown[],
  runtimeName: string,
  request: AgentRequest,
  sessionId: string | undefined,
  roleConfig: StoredRoleConfig,
  rawLogger: JsonlLogger | undefined
): Promise<void> {
  for (const rawBlock of content) {
    if (!isRecord(rawBlock)) continue;

    const block = rawBlock;
    const blockType = stringField(block, "type");

    if (blockType === "tool_use") {
      const toolName = stringField(block, "name");
      const toolUseId = stringField(block, "id");
      const input = recordField(block, "input") ?? {};
      if (toolName === undefined) continue;

      await rawLogger?.write({
        type: "agent.tool_call",
        runtime: runtimeName,
        userId: request.user.id,
        workspacePath: request.workspacePath,
        sessionId,
        userInput: request.text,
        toolName,
        toolUseId,
        permittedByRole: canUseConfiguredTool(roleConfig, toolName),
        input
      });
    } else if (blockType === "tool_result") {
      const toolUseId = stringField(block, "tool_use_id");
      const isError = booleanField(block, "is_error") ?? false;
      await rawLogger?.write({
        type: "agent.tool_result",
        runtime: runtimeName,
        userId: request.user.id,
        workspacePath: request.workspacePath,
        sessionId,
        userInput: request.text,
        toolUseId,
        isError,
        result: extractToolResultText(block)
      });
    }
  }
}
