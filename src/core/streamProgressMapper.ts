import type { AgentProgressEvent, AgentStreamEvent } from "../agent/index.js";

const EVENT_STAGE_MAP: Readonly<Record<AgentStreamEvent["type"], string>> = {
  text_delta: "agent.text_delta",
  tool_use_start: "agent.tool_use_start",
  tool_use_result: "agent.tool_use_result",
  thinking: "agent.thinking",
  compact_start: "agent.compact_start",
  compact_complete: "agent.compact_complete",
  completed: "agent.completed",
  error: "agent.error",
};

export function progressEventForAgentStreamEvent(event: AgentStreamEvent): AgentProgressEvent {
  const stage = EVENT_STAGE_MAP[event.type];
  return { stage, message: stage, data: event };
}
