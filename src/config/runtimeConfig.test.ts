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
    wechat: { token: "dev-token" },
    llm: {
      baseUrl: "https://api.example.com",
      apiKeyEnv: "TEST_API_KEY",
      modelId: "test-model",
    },
    ...overrides,
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
    expect(config.wechat.token).toBe("dev-token");
  });

  it("applies defaults for optional fields", async () => {
    const filePath = path.join(tmpdir(), `runtime-${Date.now()}.json`);
    await writeFile(filePath, JSON.stringify({
      llm: {
        baseUrl: "https://api.example.com",
        apiKeyEnv: "TEST_API_KEY",
        modelId: "test-model",
      },
    }));

    const config = await loadRuntimeConfig(filePath, { TEST_API_KEY: "secret-key" });

    expect(config.port).toBe(3000);
    expect(config.host).toBe("0.0.0.0");
    expect(config.enableDevRoutes).toBe(true);
    expect(config.wechat.token).toBe("dev-token");
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

    await expect(loadRuntimeConfig(filePath, {})).rejects.toThrow(
      `Invalid config in ${filePath}`
    );
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
