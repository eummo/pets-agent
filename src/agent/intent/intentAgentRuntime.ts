import type { AgentRequest, AgentResponse, AgentRuntime } from "../index.js";
import type { UserIntent } from "../../intent/index.js";
import { isValidIntentType } from "../../intent/index.js";
import { LlmIntentDetectionService } from "../../intent/llmIntentDetectionService.js";
import { fallbackIntentFor } from "../../core/intentHeuristics.js";

export class IntentAgentRuntime implements AgentRuntime {
  public readonly name = "intent";
  private readonly detector: LlmIntentDetectionService;

  public constructor(detector: LlmIntentDetectionService) {
    this.detector = detector;
  }

  public async run(request: AgentRequest): Promise<AgentResponse> {
    const role = request.role ?? "unknown";
    const history = request.history?.slice(-4);
    const intent = await this.detector.detectIntent(request.text, role, history);
    return { text: intent.type };
  }

  public disposeSession(sessionId: string): Promise<void> {
    void sessionId;
    return Promise.resolve();
  }
}

export function parseIntentResponse(text: string): UserIntent {
  const label = text.trim().toLowerCase();
  if (isValidIntentType(label)) {
    return { type: label };
  }
  return fallbackIntentFor(text);
}
