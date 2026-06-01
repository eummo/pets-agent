import { writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { loadRuntimeConfig } from "./runtimeConfig.js";

function validConfig(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    port: 3000,
    host: "0.0.0.0",
    knowledgeBasePath: ".harness/knowledge-base",
    logDir: ".harness/logs",
    dbPath: ".harness/state/agent.db",
    sessionStorePath: ".harness/state/sessions.json",
    historyStorePath: ".harness/state/history.json",
    enableDevRoutes: true,
    wechat: { botId: "dev-bot-id", secret: "dev-secret" },
    llm: {
      baseUrl: "https://api.example.com",
      apiKeyEnv: "TEST_API_KEY",
      modelId: "test-model"
    },
    agentSdkType: "claude",
    agentSdks: {
      claude: {
        baseUrl: "https://api.example.com",
        apiKeyEnv: "TEST_API_KEY",
        modelId: "test-model"
      }
    },
    ...overrides
  };
}

describe("loadRuntimeConfig", () => {
  it("loads a valid config and resolves the LLM API key from env", async () => {
    const filePath = path.join(tmpdir(), `runtime-${Date.now()}.json`);
    await writeFile(filePath, JSON.stringify(validConfig()));

    const config = await loadRuntimeConfig(filePath, { TEST_API_KEY: "secret-key" });

    expect(config.port).toBe(3000);
    expect(config.host).toBe("0.0.0.0");
    expect(config.llm.apiKey).toBe("secret-key");
    expect(config.wechat.botId).toBe("dev-bot-id");
    expect(config.wechat.secret).toBe("dev-secret");
  });

  it("applies context config defaults when not specified", async () => {
    const filePath = path.join(tmpdir(), `runtime-${Date.now()}.json`);
    await writeFile(filePath, JSON.stringify(validConfig()));

    const config = await loadRuntimeConfig(filePath, { TEST_API_KEY: "secret-key" });

    expect(config.context).toEqual({
      autoCompactEnabled: true,
      autoCompactWindow: 150_000,
      workspaceMaxChars: 8_000,
      historyMaxMessages: 20
    });
  });

  it("applies cron config defaults when not specified", async () => {
    const filePath = path.join(tmpdir(), `runtime-${Date.now()}.json`);
    await writeFile(filePath, JSON.stringify(validConfig()));

    const config = await loadRuntimeConfig(filePath, { TEST_API_KEY: "secret-key" });

    expect(config.cron).toEqual({
      enabled: false,
      tickIntervalMs: 60_000,
      staleGraceMs: 300_000,
      jobStorePath: ".harness/state/cron-jobs.json"
    });
  });

  it("resolves cron WeCom app-message credentials from env", async () => {
    const filePath = path.join(tmpdir(), `runtime-${Date.now()}.json`);
    await writeFile(
      filePath,
      JSON.stringify(
        validConfig({
          cron: {
            enabled: true,
            tickIntervalMs: 30_000,
            staleGraceMs: 120_000,
            jobStorePath: ".harness/state/custom-cron.json",
            wecom: {
              corpId: "corp-id",
              corpSecretEnv: "WECOM_CORP_SECRET",
              agentId: "1000002",
              tokenCacheMs: 3_600_000
            }
          }
        })
      )
    );

    const config = await loadRuntimeConfig(filePath, {
      TEST_API_KEY: "secret-key",
      WECOM_CORP_SECRET: "corp-secret"
    });

    expect(config.cron).toEqual({
      enabled: true,
      tickIntervalMs: 30_000,
      staleGraceMs: 120_000,
      jobStorePath: ".harness/state/custom-cron.json",
      wecom: {
        corpId: "corp-id",
        corpSecret: "corp-secret",
        agentId: "1000002",
        tokenCacheMs: 3_600_000
      }
    });
  });

  it("loads custom context config values", async () => {
    const filePath = path.join(tmpdir(), `runtime-${Date.now()}.json`);
    await writeFile(
      filePath,
      JSON.stringify(
        validConfig({
          context: {
            autoCompactEnabled: false,
            autoCompactWindow: 100_000,
            workspaceMaxChars: 12_000,
            historyMaxMessages: 50
          }
        })
      )
    );

    const config = await loadRuntimeConfig(filePath, { TEST_API_KEY: "secret-key" });

    expect(config.context).toEqual({
      autoCompactEnabled: false,
      autoCompactWindow: 100_000,
      workspaceMaxChars: 12_000,
      historyMaxMessages: 50
    });
  });

  it("applies defaults for optional fields", async () => {
    const filePath = path.join(tmpdir(), `runtime-${Date.now()}.json`);
    await writeFile(
      filePath,
      JSON.stringify({
        llm: {
          baseUrl: "https://api.example.com",
          apiKeyEnv: "TEST_API_KEY",
          modelId: "test-model"
        },
        agentSdkType: "claude",
        agentSdks: {
          claude: {
            baseUrl: "https://api.example.com",
            apiKeyEnv: "TEST_API_KEY",
            modelId: "test-model"
          }
        }
      })
    );

    const config = await loadRuntimeConfig(filePath, { TEST_API_KEY: "secret-key" });

    expect(config.port).toBe(3000);
    expect(config.host).toBe("127.0.0.1");
    expect(config.enableDevRoutes).toBe(false);
    expect(config.wechat.botId).toBe("dev-bot-id");
    expect(config.wechat.secret).toBe("dev-secret");
  });

  it("reports missing config file with path", async () => {
    await expect(loadRuntimeConfig("/nonexistent/runtime.json", {})).rejects.toThrow(
      "Failed to read config file at /nonexistent/runtime.json"
    );
  });

  it("reports invalid JSON with file path", async () => {
    const filePath = path.join(tmpdir(), `runtime-${Date.now()}.json`);
    await writeFile(filePath, "{ invalid json");

    await expect(loadRuntimeConfig(filePath, {})).rejects.toThrow(
      `Invalid JSON in config file at ${filePath}`
    );
  });

  it("reports schema validation errors with field paths", async () => {
    const filePath = path.join(tmpdir(), `runtime-${Date.now()}.json`);
    await writeFile(filePath, JSON.stringify({ port: "not-a-number" }));

    await expect(loadRuntimeConfig(filePath, {})).rejects.toThrow(`Invalid config in ${filePath}`);
  });

  it("includes field path in schema error for nested fields", async () => {
    const filePath = path.join(tmpdir(), `runtime-${Date.now()}.json`);
    await writeFile(filePath, JSON.stringify(validConfig({ llm: { baseUrl: "not-a-url" } })));

    let error: Error | undefined;
    try {
      await loadRuntimeConfig(filePath, { TEST_API_KEY: "key" });
    } catch (e) {
      error = e as Error;
    }
    expect(error).toBeDefined();
    expect(error?.message).toContain("llm.baseUrl");
  });

  it("rejects config with missing llm section", async () => {
    const filePath = path.join(tmpdir(), `runtime-${Date.now()}.json`);
    await writeFile(filePath, JSON.stringify({ port: 3000 }));

    await expect(loadRuntimeConfig(filePath, {})).rejects.toThrow();
  });

  it("rejects config with invalid port", async () => {
    const filePath = path.join(tmpdir(), `runtime-${Date.now()}.json`);
    await writeFile(filePath, JSON.stringify(validConfig({ port: -1 })));

    await expect(loadRuntimeConfig(filePath, { TEST_API_KEY: "key" })).rejects.toThrow();
  });

  it("throws clearly when the LLM API key is missing from env", async () => {
    const filePath = path.join(tmpdir(), `runtime-${Date.now()}.json`);
    await writeFile(filePath, JSON.stringify(validConfig()));

    await expect(loadRuntimeConfig(filePath, {})).rejects.toThrow(
      "Missing LLM API key environment variable: TEST_API_KEY"
    );
  });
});

describe("loadRuntimeConfig agentSdk resolution", () => {
  it("resolves the selected agentSdkType from agentSdks", async () => {
    const filePath = path.join(tmpdir(), `runtime-${Date.now()}.json`);
    await writeFile(
      filePath,
      JSON.stringify(
        validConfig({
          agentSdkType: "codebuddy",
          agentSdks: {
            claude: {
              baseUrl: "https://claude.example.com",
              apiKeyEnv: "CLAUDE_KEY",
              modelId: "claude-3"
            },
            codebuddy: {
              baseUrl: "https://codebuddy.example.com",
              apiKeyEnv: "CODEBUDDY_KEY",
              modelId: "cb-model"
            }
          }
        })
      )
    );

    const config = await loadRuntimeConfig(filePath, {
      TEST_API_KEY: "secret-key",
      CLAUDE_KEY: "sk-claude",
      CODEBUDDY_KEY: "sk-cb"
    });

    expect(config.agentSdk.type).toBe("codebuddy");
    expect(config.agentSdk.apiKey).toBe("sk-cb");
    expect(config.agentSdk.modelId).toBe("cb-model");
  });

  it("resolves pi SDK when selected", async () => {
    const filePath = path.join(tmpdir(), `runtime-${Date.now()}.json`);
    await writeFile(
      filePath,
      JSON.stringify(
        validConfig({
          agentSdkType: "pi",
          agentSdks: {
            pi: {
              baseUrl: "https://pi.example.com",
              apiKeyEnv: "PI_KEY",
              modelId: "pi-model"
            }
          }
        })
      )
    );

    const config = await loadRuntimeConfig(filePath, {
      TEST_API_KEY: "secret-key",
      PI_KEY: "sk-pi"
    });

    expect(config.agentSdk.type).toBe("pi");
    expect(config.agentSdk.apiKey).toBe("sk-pi");
  });

  it("allows codebuddy SDK to use local authentication when apiKeyEnv is omitted", async () => {
    const filePath = path.join(tmpdir(), `runtime-${Date.now()}.json`);
    await writeFile(
      filePath,
      JSON.stringify(
        validConfig({
          agentSdkType: "codebuddy",
          agentSdks: {
            codebuddy: {
              baseUrl: "https://codebuddy.example.com",
              modelId: "cb-model"
            }
          }
        })
      )
    );

    const config = await loadRuntimeConfig(filePath, {
      TEST_API_KEY: "secret-key"
    });

    expect(config.agentSdk.type).toBe("codebuddy");
    expect(config.agentSdk.apiKey).toBe("");
    expect(config.agentSdk.apiKeyEnv).toBeUndefined();
  });

  it("loads codebuddy enterprise endpoint when configured", async () => {
    const filePath = path.join(tmpdir(), `runtime-${Date.now()}.json`);
    await writeFile(
      filePath,
      JSON.stringify(
        validConfig({
          agentSdkType: "codebuddy",
          agentSdks: {
            codebuddy: {
              baseUrl: "https://codebuddy.example.com",
              modelId: "cb-model",
              endpoint: "https://enterprise.example.com/"
            }
          }
        })
      )
    );

    const config = await loadRuntimeConfig(filePath, {
      TEST_API_KEY: "secret-key"
    });

    expect(config.agentSdk.type).toBe("codebuddy");
    expect(config.agentSdk.endpoint).toBe("https://enterprise.example.com/");
  });

  it("resolves codebuddy enterprise endpoint from env", async () => {
    const filePath = path.join(tmpdir(), `runtime-${Date.now()}.json`);
    await writeFile(
      filePath,
      JSON.stringify(
        validConfig({
          agentSdkType: "codebuddy",
          agentSdks: {
            codebuddy: {
              baseUrl: "https://codebuddy.example.com",
              modelId: "cb-model",
              endpointEnv: "CODEBUDDY_ENDPOINT"
            }
          }
        })
      )
    );

    const config = await loadRuntimeConfig(filePath, {
      TEST_API_KEY: "secret-key",
      CODEBUDDY_ENDPOINT: "https://enterprise.example.com/"
    });

    expect(config.agentSdk.type).toBe("codebuddy");
    expect(config.agentSdk.endpoint).toBe("https://enterprise.example.com/");
    expect(config.agentSdk.endpointEnv).toBe("CODEBUDDY_ENDPOINT");
  });

  it("throws when the selected agentSdkType has no entry in agentSdks", async () => {
    const filePath = path.join(tmpdir(), `runtime-${Date.now()}.json`);
    await writeFile(
      filePath,
      JSON.stringify(
        validConfig({
          agentSdkType: "codebuddy",
          agentSdks: {
            claude: {
              baseUrl: "https://claude.example.com",
              apiKeyEnv: "CLAUDE_KEY",
              modelId: "claude-3"
            }
          }
        })
      )
    );

    await expect(
      loadRuntimeConfig(filePath, { TEST_API_KEY: "secret-key", CLAUDE_KEY: "sk-claude" })
    ).rejects.toThrow('No agentSdk config found for type "codebuddy"');
  });

  it("throws when agentSdkType or agentSdks is missing", async () => {
    const filePath = path.join(tmpdir(), `runtime-${Date.now()}.json`);
    await writeFile(
      filePath,
      JSON.stringify({
        llm: {
          baseUrl: "https://api.example.com",
          apiKeyEnv: "TEST_API_KEY",
          modelId: "test-model"
        }
      })
    );

    await expect(loadRuntimeConfig(filePath, { TEST_API_KEY: "key" })).rejects.toThrow();
  });

  it("preserves optional fields from the selected SDK entry", async () => {
    const filePath = path.join(tmpdir(), `runtime-${Date.now()}.json`);
    await writeFile(
      filePath,
      JSON.stringify(
        validConfig({
          agentSdkType: "claude",
          agentSdks: {
            claude: {
              baseUrl: "https://claude.example.com",
              apiKeyEnv: "CLAUDE_KEY",
              modelId: "claude-3",
              api: "anthropic-messages",
              provider: "anthropic",
              contextWindow: 200_000
            }
          }
        })
      )
    );

    const config = await loadRuntimeConfig(filePath, {
      TEST_API_KEY: "secret-key",
      CLAUDE_KEY: "sk-claude"
    });

    expect(config.agentSdk.api).toBe("anthropic-messages");
    expect(config.agentSdk.provider).toBe("anthropic");
    expect(config.agentSdk.contextWindow).toBe(200_000);
  });
});
