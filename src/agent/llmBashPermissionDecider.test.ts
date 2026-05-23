import { beforeEach, afterEach, describe, expect, it } from "vitest";
import type { Model } from "@earendil-works/pi-ai";
import {
  registerFauxProvider,
  fauxAssistantMessage,
  fauxText,
  fauxThinking,
} from "@earendil-works/pi-ai";
import type { RoleConfig } from "./claudeSdkAgentRuntime.js";
import { LlmBashPermissionDecider } from "./llmBashPermissionDecider.js";

const roleConfig: RoleConfig = {
  name: "reviewer",
  allowedTools: ["Read", "Bash"],
  permissionMode: "dontAsk",
  systemPrompt: "Read only.",
};

describe("LlmBashPermissionDecider", () => {
  let registration: ReturnType<typeof registerFauxProvider>;

  beforeEach(() => {
    registration = registerFauxProvider({ tokensPerSecond: 50 });
  });

  afterEach(() => {
    registration.unregister();
  });

  function createDecider(responses: ReturnType<typeof fauxAssistantMessage>[]): LlmBashPermissionDecider {
    registration.setResponses(responses);
    return new LlmBashPermissionDecider(registration.getModel() as Model<"anthropic-messages">, "test-key");
  }

  it("allows commands when the model classifies them as read-only", async () => {
    const decider = createDecider([
      fauxAssistantMessage([
        fauxThinking("This only lists files."),
        fauxText("allow"),
      ]),
    ]);
    const result = await decider.decide(roleConfig, "Bash", { command: "ls -la" });

    expect(result).toEqual({ behavior: "allow", decisionClassification: "user_temporary" });
  });

  it("denies commands when the model does not classify them as read-only", async () => {
    const decider = createDecider([
      fauxAssistantMessage([fauxText("deny")]),
    ]);
    const result = await decider.decide(roleConfig, "Bash", { command: "rm -rf dist" });

    expect(result.behavior).toBe("deny");
  });

  it("denies non-Bash tools immediately", async () => {
    const decider = createDecider([]);
    const result = await decider.decide(roleConfig, "Edit", { file_path: "a.ts" });

    expect(result.behavior).toBe("deny");
  });

  it("denies when command is missing", async () => {
    const decider = createDecider([]);
    const result = await decider.decide(roleConfig, "Bash", {});

    expect(result.behavior).toBe("deny");
  });

  it("denies on provider error", async () => {
    // No responses set — faux provider will throw
    const decider = new LlmBashPermissionDecider(registration.getModel() as Model<"anthropic-messages">, "test-key");
    const result = await decider.decide(roleConfig, "Bash", { command: "ls" });

    expect(result.behavior).toBe("deny");
  });

  it("denies when provider returns an error message", async () => {
    const decider = createDecider([
      fauxAssistantMessage([fauxText("allow")], {
        stopReason: "error",
        errorMessage: "provider rejected request",
      }),
    ]);
    const result = await decider.decide(roleConfig, "Bash", { command: "ls" });

    expect(result.behavior).toBe("deny");
  });
});
