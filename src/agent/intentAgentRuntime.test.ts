import { describe, expect, it } from "vitest";
import type { AgentRequest } from "./index.js";
import { IntentAgentRuntime, parseIntentResponse } from "./intentAgentRuntime.js";
import { LlmIntentDetectionService } from "../intent/llmIntentDetectionService.js";
import {
  registerFauxProvider,
  fauxAssistantMessage,
  fauxText,
} from "@earendil-works/pi-ai";

describe("IntentAgentRuntime", () => {
  it("returns intent label from detector as response text", async () => {
    const registration = registerFauxProvider({ tokensPerSecond: 50 });
    try {
      const model = registration.getModel();
      registration.setResponses([
        fauxAssistantMessage([fauxText("mutate")]),
      ]);
      const detector = new LlmIntentDetectionService(model, "test-key");
      const runtime = new IntentAgentRuntime(detector);

      const request: AgentRequest = {
        user: { id: "user-1" },
        text: "修改代码",
        workspacePath: "/workspace",
        role: "reviewer",
      };
      const response = await runtime.run(request);

      expect(response.text).toBe("mutate");
    } finally {
      registration.unregister();
    }
  });

  it("has name 'intent'", () => {
    const registration = registerFauxProvider({ tokensPerSecond: 50 });
    try {
      const detector = new LlmIntentDetectionService(registration.getModel(), "test-key");
      const runtime = new IntentAgentRuntime(detector);
      expect(runtime.name).toBe("intent");
    } finally {
      registration.unregister();
    }
  });

  it("disposeSession does not throw", async () => {
    const registration = registerFauxProvider({ tokensPerSecond: 50 });
    try {
      const detector = new LlmIntentDetectionService(registration.getModel(), "test-key");
      const runtime = new IntentAgentRuntime(detector);
      await expect(runtime.disposeSession("test-session")).resolves.toBeUndefined();
    } finally {
      registration.unregister();
    }
  });
});

describe("parseIntentResponse", () => {
  it("parses valid intent labels", () => {
    expect(parseIntentResponse("query")).toEqual({ type: "query" });
    expect(parseIntentResponse("mutate")).toEqual({ type: "mutate" });
    expect(parseIntentResponse("update_kb")).toEqual({ type: "update_kb" });
  });

  it("handles whitespace and casing", () => {
    expect(parseIntentResponse("  QUERY  ")).toEqual({ type: "query" });
    expect(parseIntentResponse("\nMutate\n")).toEqual({ type: "mutate" });
  });

  it("falls back to heuristic for invalid labels", () => {
    const result = parseIntentResponse("unknown_intent");
    // fallbackIntentFor("unknown_intent") returns { type: "query" } by default
    expect(result.type).toBe("query");
  });

  it("falls back for mutation-like messages with invalid labels", () => {
    const result = parseIntentResponse("修改");
    // fallbackIntentFor("修改") returns { type: "mutate" }
    expect(result.type).toBe("mutate");
  });
});
