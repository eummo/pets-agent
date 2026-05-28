import { beforeEach, afterEach, describe, expect, it } from "vitest";
import {
  registerFauxProvider,
  fauxAssistantMessage,
  fauxText,
  fauxThinking
} from "@earendil-works/pi-ai";
import type { StoredRoleConfig } from "../../auth/index.js";
import { LlmBashPermissionDecider } from "./llmBashPermissionDecider.js";

const roleConfig: StoredRoleConfig = {
  name: "reviewer",
  allowedTools: ["Read", "Bash"],
  permissionMode: "dontAsk",
  systemPrompt: "Read only."
};

describe("LlmBashPermissionDecider", () => {
  let registration: ReturnType<typeof registerFauxProvider>;

  beforeEach(() => {
    registration = registerFauxProvider({ tokensPerSecond: 50 });
  });

  afterEach(() => {
    registration.unregister();
  });

  function createDecider(
    responses: ReturnType<typeof fauxAssistantMessage>[]
  ): LlmBashPermissionDecider {
    registration.setResponses(responses);
    return new LlmBashPermissionDecider(registration.getModel(), "test-key");
  }

  it("allows commands when the model classifies them as read-only", async () => {
    const decider = createDecider([
      fauxAssistantMessage([fauxThinking("This only lists files."), fauxText("allow")])
    ]);
    const result = await decider.decide(roleConfig, "Bash", { command: "ls -la" });

    expect(result).toEqual({ behavior: "allow", decisionClassification: "user_temporary" });
  });

  it("logs bash permission model request, response, and decision", async () => {
    registration.setResponses([fauxAssistantMessage([fauxText("allow")])]);
    const rawEvents: Record<string, unknown>[] = [];
    const decider = new LlmBashPermissionDecider(registration.getModel(), "test-key", {
      filePath: "memory.jsonl",
      write(event) {
        rawEvents.push(event);
        return Promise.resolve();
      }
    });

    const result = await decider.decide(roleConfig, "Bash", { command: "ls -la" });

    expect(result.behavior).toBe("allow");
    expect(rawEvents.map((event) => event["type"])).toEqual([
      "llm.request",
      "llm.response",
      "tool.permission_result"
    ]);
    expect(rawEvents[0]).toMatchObject({
      type: "llm.request",
      operation: "bash_permission",
      role: "reviewer",
      command: "ls -la"
    });
    expect(rawEvents[1]).toMatchObject({
      type: "llm.response",
      operation: "bash_permission",
      role: "reviewer",
      command: "ls -la"
    });
    expect(asRecord(rawEvents[1]?.["response"])["stopReason"]).toBe("stop");
    expect(rawEvents[2]).toMatchObject({
      type: "tool.permission_result",
      operation: "bash_permission",
      role: "reviewer",
      command: "ls -la",
      behavior: "allow"
    });
  });

  it("denies commands when the model does not classify them as read-only", async () => {
    const decider = createDecider([fauxAssistantMessage([fauxText("deny")])]);
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
    const decider = new LlmBashPermissionDecider(registration.getModel(), "test-key");
    const result = await decider.decide(roleConfig, "Bash", { command: "ls" });

    expect(result.behavior).toBe("deny");
  });

  it("denies when provider returns an error message", async () => {
    const decider = createDecider([
      fauxAssistantMessage([fauxText("allow")], {
        stopReason: "error",
        errorMessage: "provider rejected request"
      })
    ]);
    const result = await decider.decide(roleConfig, "Bash", { command: "ls" });

    expect(result.behavior).toBe("deny");
  });
});

function asRecord(value: unknown): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`Expected object, got ${JSON.stringify(value)}.`);
  }
  return value as Record<string, unknown>;
}
