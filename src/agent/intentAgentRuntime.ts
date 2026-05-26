import type { AgentRequest, AgentResponse, AgentRuntime, UserIntent } from "../core/contracts.js";
import { LlmIntentDetectionService } from "../intent/llmIntentDetectionService.js";
import { fallbackIntentFor } from "../core/intentHeuristics.js";

const VALID_INTENT_LABELS = new Set<string>(["query", "mutate", "update_kb"]);

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

  public disposeSession(): Promise<void> {
    return Promise.resolve();
  }
}

export function parseIntentResponse(text: string): UserIntent {
  const label = text.trim().toLowerCase();
  if (VALID_INTENT_LABELS.has(label)) {
    return { type: label as UserIntent["type"] };
  }
  return fallbackIntentFor(text);
}
