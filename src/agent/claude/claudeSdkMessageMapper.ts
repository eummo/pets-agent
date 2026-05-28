import type { SDKMessage } from "@anthropic-ai/claude-agent-sdk";

export function isAssistantMessage(
  msg: SDKMessage
): msg is Extract<SDKMessage, { type: "assistant" }> {
  return msg.type === "assistant";
}

export function isResultMessage(msg: SDKMessage): msg is Extract<SDKMessage, { type: "result" }> {
  return msg.type === "result";
}

export function isSystemMessage(msg: SDKMessage): msg is Extract<SDKMessage, { type: "system" }> {
  return msg.type === "system";
}
