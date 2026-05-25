import type { SDKMessage } from "@anthropic-ai/claude-agent-sdk";
import type { AgentRequest, AgentStreamEvent, StoredRoleConfig } from "../core/contracts.js";
import { canUseConfiguredTool } from "./claudeToolPolicy.js";

export function isAssistantMessage(msg: SDKMessage): boolean {
  return msg.type === "assistant";
}

export function isResultMessage(msg: SDKMessage): boolean {
  return msg.type === "result";
}

export function forwardAssistantMessageEvents(
  msg: Extract<SDKMessage, { type: "assistant" }>,
  request: AgentRequest,
  roleConfig: StoredRoleConfig,
): void {
  const msgData = msg as unknown as { message?: { content?: unknown[] } };
  const content = msgData.message?.content ?? [];
  for (const rawBlock of content) {
    const block = rawBlock as Record<string, unknown>;
    const blockType = block["type"] as string;

    if (blockType === "tool_use") {
      const toolName = block["name"] as string;
      if (!canUseConfiguredTool(roleConfig, toolName)) {
        const message = `Tool ${toolName} is not permitted for role ${roleConfig.name}.`;
        request.stream?.({ type: "error", message });
        void request.progress?.({
          stage: "agent.error",
          message,
          data: { toolName },
        });
        continue;
      }

      const input = (block["input"] as Record<string, unknown> | null) ?? {};
      const toolEvent: AgentStreamEvent = {
        type: "tool_use_start",
        toolName,
        toolUseId: block["id"] as string,
        input,
      };
      request.stream?.(toolEvent);
      void request.progress?.({
        stage: "agent.tool_use_start",
        message: `${toolName}: ${JSON.stringify(block["input"]).slice(0, 100)}`,
        data: { toolUseId: block["id"], toolName },
      });
    } else if (blockType === "tool_result") {
      const isError = block["is_error"] === true;
      const contentParts = block["content"] as unknown[] | undefined;
      const resultText = (contentParts ?? [])
        .map((p: unknown) => {
          const part = p as Record<string, unknown>;
          return typeof part["text"] === "string" ? part["text"] : "";
        })
        .join("");
      request.stream?.({
        type: "tool_use_result",
        toolUseId: block["tool_use_id"] as string,
        result: resultText,
        ...(isError ? { isError: true } : {}),
      });
    }
  }
}

export function forwardStreamEvent(event: Record<string, unknown>, request: AgentRequest): void {
  const e = event["event"] as Record<string, unknown> | undefined;
  if (e === undefined) return;

  const eventType = e["type"] as string | undefined;

  if (eventType === "content_block_delta") {
    const delta = e["delta"] as Record<string, unknown> | undefined;
    const deltaType = delta?.["type"] as string | undefined;
    if (deltaType === "text_delta" && typeof delta?.["text"] === "string") {
      request.stream?.({ type: "text_delta", text: delta["text"] });
    } else if (deltaType === "thinking_delta" && typeof delta?.["thinking"] === "string") {
      request.stream?.({ type: "thinking", text: delta["thinking"] });
    }
  }
}
