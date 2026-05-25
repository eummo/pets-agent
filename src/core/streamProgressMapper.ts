import type { AgentProgressEvent, AgentStreamEvent } from "./contracts.js";

export function progressEventForAgentStreamEvent(event: AgentStreamEvent): AgentProgressEvent {
  const stage = event.type === "text_delta" ? "agent.text_delta"
    : event.type === "tool_use_start" ? "agent.tool_use_start"
    : event.type === "tool_use_result" ? "agent.tool_use_result"
    : event.type === "thinking" ? "agent.thinking"
    : event.type === "compact_start" ? "agent.compact_start"
    : event.type === "compact_complete" ? "agent.compact_complete"
    : event.type === "completed" ? "agent.completed"
    : "agent.error";

  return {
    stage,
    message: stage,
    data: event,
  };
}
