import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ResolvedLlmConfig } from "../config/llmConfig.js";
import type { RoleConfig } from "./claudeSdkAgentRuntime.js";
import { LlmBashPermissionDecider } from "./llmBashPermissionDecider.js";

const mockConfig: ResolvedLlmConfig = {
  baseUrl: "https://api.example.com",
  apiKeyEnv: "TEST_KEY",
  modelId: "test-model",
  apiKey: "test-api-key",
};

const roleConfig: RoleConfig = {
  name: "reviewer",
  allowedTools: ["Read", "Bash"],
  permissionMode: "dontAsk",
  systemPrompt: "Read only.",
};

describe("LlmBashPermissionDecider", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("allows commands when the model classifies them as read-only", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(JSON.stringify({
        content: [
          { type: "thinking", thinking: "This only lists files." },
          { type: "text", text: "allow" },
        ]
      }), { status: 200 })
    );

    const decider = new LlmBashPermissionDecider(mockConfig);
    const result = await decider.decide(roleConfig, "Bash", { command: "ls -la" });

    expect(result).toEqual({ behavior: "allow", decisionClassification: "user_temporary" });
    expect(fetchSpy).toHaveBeenCalledTimes(1);
  });

  it("denies commands when the model does not classify them as read-only", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(JSON.stringify({ content: [{ type: "text", text: "deny" }] }), { status: 200 })
    );

    const decider = new LlmBashPermissionDecider(mockConfig);
    const result = await decider.decide(roleConfig, "Bash", { command: "rm -rf dist" });

    expect(result.behavior).toBe("deny");
  });
});
