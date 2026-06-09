import { writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, expectTypeOf, it, vi } from "vitest";
import type { RoleConfigStore, StoredRoleConfig } from "./auth/index.js";
import { resolveActiveAgentSdk } from "./config/llmConfig.js";
import { loadRuntimeConfig } from "./config/runtimeConfig.js";
import { cronJobStoreFileSchema, cronJobSchema, cronScheduleSchema } from "./cron/index.js";
import type { JsonlLogger } from "./logging/jsonlLogger.js";
import { createAgentRuntimeFactory } from "./agent/createAgentRuntimes.js";
import type { ResolvedLlmConfig, ResolvedAgentSdkConfig } from "./config/llmConfig.js";

const sdkMocks = vi.hoisted(() => ({
  claudeQuery: vi.fn()
}));

vi.mock("@anthropic-ai/claude-agent-sdk", () => ({
  query: sdkMocks.claudeQuery
}));

describe("dependency regression samples", () => {
  it("keeps Zod discriminated unions and record schemas stable for cron persistence", () => {
    expect(cronScheduleSchema.parse({ type: "interval", milliseconds: 60_000 })).toEqual({
      type: "interval",
      milliseconds: 60_000
    });
    expect(() => cronScheduleSchema.parse({ type: "interval", milliseconds: "60000" })).toThrow();

    expect(
      cronJobStoreFileSchema.parse({
        jobs: {
          "job-1": {
            id: "job-1",
            name: "Daily summary",
            schedule: { type: "cron", expression: "0 9 * * *" },
            prompt: "Summarize changes",
            workspacePath: "D:/workspace",
            enabled: true,
            delivery: { channels: ["sse:admin"] },
            createdAt: "2026-06-09T00:00:00.000Z",
            updatedAt: "2026-06-09T00:00:00.000Z"
          }
        },
        runState: {
          "job-1": {
            nextRunAt: "2026-06-09T09:00:00.000Z",
            lastResult: {
              jobId: "job-1",
              startedAt: "2026-06-09T09:00:00.000Z",
              finishedAt: "2026-06-09T09:00:01.000Z",
              status: "success",
              output: "ok"
            }
          }
        }
      })
    ).toMatchObject({
      jobs: { "job-1": { schedule: { type: "cron" } } },
      runState: { "job-1": { lastResult: { status: "success" } } }
    });

    expect(() =>
      cronJobSchema.parse({
        id: "job-1",
        name: "Broken",
        schedule: { type: "unknown" },
        prompt: "Run",
        workspacePath: "D:/workspace",
        enabled: true,
        delivery: { channels: ["sse:admin"] },
        createdAt: "2026-06-09T00:00:00.000Z",
        updatedAt: "2026-06-09T00:00:00.000Z"
      })
    ).toThrow();
  });

  it("keeps Zod default application stable for runtime config", async () => {
    const filePath = path.join(tmpdir(), `runtime-dependency-${Date.now()}.json`);
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

    expect(config.conversationStore).toBe("sqlite");
    expect(config.cron.jobStore).toBe("sqlite");
    expect(config.cron.leaderLeasePath).toBe(".harness/state/cron-jobs.json.leader");
    expectTypeOf(config.cron.jobStore).toEqualTypeOf<"sqlite" | "file">();
  });

  it("keeps TypeScript optional SDK auth mapping stable for CodeBuddy local auth", () => {
    const config = resolveActiveAgentSdk(
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

    expect(config.apiKey).toBe("");
    expect(config.endpoint).toBeUndefined();
    expect(config.endpointEnv).toBe("CODEBUDDY_ENDPOINT");
    expectTypeOf(config.apiKeyEnv).toEqualTypeOf<string | undefined>();
  });

  it("keeps runtime factory cache keys tied to role update timestamps", async () => {
    const store = makeRoleConfigStore([
      makeRoleConfig("reviewer", "2026-06-09T00:00:00.000Z"),
      makeRoleConfig("developer", "2026-06-09T00:01:00.000Z")
    ]);
    const factory = createAgentRuntimeFactory(
      rawLogger,
      store,
      resolvedLlmConfig,
      resolvedAgentSdkConfig
    );

    await expect(factory.cacheKeyForRole?.("reviewer")).resolves.toBe(
      "reviewer:2026-06-09T00:00:00.000Z"
    );
    await expect(factory.cacheKeyForRole?.("developer")).resolves.toBe(
      "developer:2026-06-09T00:01:00.000Z"
    );
    await expect(factory.cacheKeyForRole?.("missing")).resolves.toBeUndefined();
  });
});

const rawLogger: JsonlLogger = {
  filePath: "dependency-regression.jsonl",
  write: vi.fn().mockResolvedValue(undefined)
};

const resolvedLlmConfig: ResolvedLlmConfig = {
  baseUrl: "https://llm.example.com",
  apiKeyEnv: "LLM_API_KEY",
  modelId: "llm-model",
  apiKey: "llm-key"
};

const resolvedAgentSdkConfig: ResolvedAgentSdkConfig = {
  type: "claude",
  baseUrl: "https://sdk.example.com",
  apiKeyEnv: "SDK_API_KEY",
  modelId: "sdk-model",
  apiKey: "sdk-key"
};

function makeRoleConfig(name: string, updatedAt: string): StoredRoleConfig {
  return {
    name,
    allowedTools: ["Read"],
    permissionMode: "dontAsk",
    systemPrompt: `Role: ${name}`,
    updatedAt
  };
}

function makeRoleConfigStore(configs: readonly StoredRoleConfig[]): RoleConfigStore {
  return {
    getAll: vi.fn().mockResolvedValue(configs),
    getByName: vi.fn().mockImplementation((name: string) => {
      const config = configs.find((entry) => entry.name === name);
      return Promise.resolve(config);
    }),
    upsert: vi.fn().mockResolvedValue(undefined),
    deleteByName: vi.fn().mockResolvedValue(false)
  };
}
