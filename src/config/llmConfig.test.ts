import { describe, expect, it } from "vitest";
import { resolveLlmConfig, summarizeLlmConfig } from "./llmConfig.js";

describe("llmConfig", () => {
  it("resolves api keys from the named environment variable without exposing it in summaries", () => {
    const config = {
      baseUrl: "https://api.minimaxi.com/anthropic",
      apiKeyEnv: "LOCAL_LLM_API_KEY",
      modelId: "MiniMax-M2.7"
    };

    const resolved = resolveLlmConfig(config, {
      LOCAL_LLM_API_KEY: "secret-key"
    });

    expect(resolved.apiKey).toBe("secret-key");
    expect(summarizeLlmConfig(resolved)).toEqual(config);
    expect(JSON.stringify(summarizeLlmConfig(resolved))).not.toContain("secret-key");
  });

  it("fails clearly when the api key environment variable is missing", () => {
    expect(() =>
      resolveLlmConfig({
        baseUrl: "https://api.minimaxi.com/anthropic",
        apiKeyEnv: "LOCAL_LLM_API_KEY",
        modelId: "MiniMax-M2.7"
      }, {})
    ).toThrow("Missing LLM API key environment variable: LOCAL_LLM_API_KEY");
  });
});
