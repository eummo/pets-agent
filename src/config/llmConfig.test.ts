import { describe, expect, it } from "vitest";
import {
  resolveLlmConfig,
  summarizeLlmConfig,
  resolveActiveAgentSdk,
  summarizeAgentSdkConfig
} from "./llmConfig.js";

describe("llmConfig", () => {
  it("resolves api keys from the named environment variable without exposing it in summaries", () => {
    const config = {
      baseUrl: "https://api.minimaxi.com/anthropic",
      apiKeyEnv: "LOCAL_LLM_API_KEY",
      modelId: "MiniMax-M3"
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
      resolveLlmConfig(
        {
          baseUrl: "https://api.minimaxi.com/anthropic",
          apiKeyEnv: "LOCAL_LLM_API_KEY",
          modelId: "MiniMax-M3"
        },
        {}
      )
    ).toThrow("Missing LLM API key environment variable: LOCAL_LLM_API_KEY");
  });
});

describe("resolveActiveAgentSdk", () => {
  const agentSdks = {
    claude: {
      baseUrl: "https://claude.example.com",
      apiKeyEnv: "CLAUDE_KEY",
      modelId: "claude-3"
    },
    codebuddy: {
      baseUrl: "https://codebuddy.example.com",
      apiKeyEnv: "CODEBUDDY_KEY",
      modelId: "cb-model"
    },
    pi: {
      baseUrl: "https://pi.example.com",
      apiKeyEnv: "PI_KEY",
      modelId: "pi-model"
    }
  };

  it("resolves the selected SDK type from agentSdks", () => {
    const resolved = resolveActiveAgentSdk("claude", agentSdks, { CLAUDE_KEY: "sk-123" });
    expect(resolved.type).toBe("claude");
    expect(resolved.apiKey).toBe("sk-123");
    expect(resolved.modelId).toBe("claude-3");
  });

  it("resolves codebuddy SDK when selected", () => {
    const resolved = resolveActiveAgentSdk("codebuddy", agentSdks, { CODEBUDDY_KEY: "cb-456" });
    expect(resolved.type).toBe("codebuddy");
    expect(resolved.apiKey).toBe("cb-456");
  });

  it("allows codebuddy SDK to use local authentication without an API key env", () => {
    const resolved = resolveActiveAgentSdk(
      "codebuddy",
      {
        codebuddy: {
          baseUrl: "https://codebuddy.example.com",
          modelId: "cb-model"
        }
      },
      {}
    );
    expect(resolved.type).toBe("codebuddy");
    expect(resolved.apiKey).toBe("");
    expect(resolved.apiKeyEnv).toBeUndefined();
  });

  it("resolves pi SDK when selected", () => {
    const resolved = resolveActiveAgentSdk("pi", agentSdks, { PI_KEY: "pi-789" });
    expect(resolved.type).toBe("pi");
    expect(resolved.apiKey).toBe("pi-789");
  });

  it("throws when the selected SDK type has no config in agentSdks", () => {
    expect(() =>
      resolveActiveAgentSdk("claude", { codebuddy: agentSdks.codebuddy }, { CODEBUDDY_KEY: "x" })
    ).toThrow('No agentSdk config found for type "claude"');
  });

  it("throws when the API key env variable is missing", () => {
    expect(() => resolveActiveAgentSdk("claude", agentSdks, {})).toThrow(
      "Missing Agent SDK (claude) API key environment variable: CLAUDE_KEY"
    );
  });

  it("throws when non-codebuddy SDK omits apiKeyEnv", () => {
    expect(() =>
      resolveActiveAgentSdk(
        "claude",
        {
          claude: {
            baseUrl: "https://claude.example.com",
            modelId: "claude-3"
          }
        },
        {}
      )
    ).toThrow("Missing Agent SDK (claude) apiKeyEnv.");
  });

  it("preserves optional fields from the SDK entry", () => {
    const resolved = resolveActiveAgentSdk(
      "claude",
      {
        claude: {
          baseUrl: "https://claude.example.com",
          apiKeyEnv: "CLAUDE_KEY",
          modelId: "claude-3",
          api: "anthropic-messages",
          provider: "anthropic",
          contextWindow: 200_000
        }
      },
      { CLAUDE_KEY: "sk-123" }
    );
    expect(resolved.api).toBe("anthropic-messages");
    expect(resolved.provider).toBe("anthropic");
    expect(resolved.contextWindow).toBe(200_000);
  });

  it("preserves codebuddy enterprise authentication routing fields", () => {
    const resolved = resolveActiveAgentSdk(
      "codebuddy",
      {
        codebuddy: {
          baseUrl: "https://codebuddy.example.com",
          modelId: "cb-model",
          endpoint: "https://enterprise.example.com/"
        }
      },
      {}
    );

    expect(resolved.endpoint).toBe("https://enterprise.example.com/");
    expect(resolved.environment).toBeUndefined();
    expect(resolved.apiKey).toBe("");
  });

  it("resolves codebuddy enterprise endpoint from env", () => {
    const resolved = resolveActiveAgentSdk(
      "codebuddy",
      {
        codebuddy: {
          baseUrl: "https://codebuddy.example.com",
          modelId: "cb-model",
          endpointEnv: "CODEBUDDY_ENDPOINT"
        }
      },
      { CODEBUDDY_ENDPOINT: "https://enterprise.example.com/" }
    );

    expect(resolved.endpoint).toBe("https://enterprise.example.com/");
    expect(resolved.endpointEnv).toBe("CODEBUDDY_ENDPOINT");
    expect(resolved.apiKey).toBe("");
  });

  it("allows codebuddy to fall back to local CLI endpoint settings when endpointEnv is missing", () => {
    const resolved = resolveActiveAgentSdk(
      "codebuddy",
      {
        codebuddy: {
          baseUrl: "https://codebuddy.example.com",
          modelId: "cb-model",
          endpointEnv: "CODEBUDDY_ENDPOINT"
        }
      },
      {}
    );

    expect(resolved.endpoint).toBeUndefined();
    expect(resolved.endpointEnv).toBe("CODEBUDDY_ENDPOINT");
    expect(resolved.apiKey).toBe("");
  });
});

describe("summarizeAgentSdkConfig", () => {
  it("includes type, baseUrl, apiKeyEnv, and modelId", () => {
    const summary = summarizeAgentSdkConfig({
      type: "codebuddy",
      baseUrl: "https://cb.example.com",
      apiKeyEnv: "CB_KEY",
      modelId: "cb-1",
      endpointEnv: "CODEBUDDY_ENDPOINT",
      contextWindow: 200_000
    });
    expect(summary).toEqual({
      type: "codebuddy",
      baseUrl: "https://cb.example.com",
      apiKeyEnv: "CB_KEY",
      modelId: "cb-1",
      endpointEnv: "CODEBUDDY_ENDPOINT"
    });
  });
});
