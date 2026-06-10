import { beforeEach, describe, expect, it, vi } from "vitest";
import type { StoredRoleConfig } from "../auth/index.js";
import type { RoleConfigStore } from "../auth/index.js";
import type { JsonlLogger } from "../logging/jsonlLogger.js";
import type { ResolvedLlmConfig, ResolvedAgentSdkConfig } from "../config/llmConfig.js";
import { createAgentRuntimes, createAgentRuntimeFactory } from "./createAgentRuntimes.js";

// ── Mock SDKs ─────────────────────────────────────────────────────────────────

const claudeSdkMocks = vi.hoisted(() => ({ query: vi.fn() }));
vi.mock("@anthropic-ai/claude-agent-sdk", () => ({
  query: claudeSdkMocks.query
}));

const codebuddySdkMocks = vi.hoisted(() => ({ query: vi.fn() }));
vi.mock("@tencent-ai/agent-sdk", () => ({
  query: codebuddySdkMocks.query
}));

const piSdkMocks = vi.hoisted(() => {
  const listeners: ((event: unknown) => void)[] = [];
  const session = {
    sessionId: "pi-test-session",
    agent: {},
    subscribe: vi.fn((listener: (event: unknown) => void) => {
      listeners.push(listener);
      return () => {
        const idx = listeners.indexOf(listener);
        if (idx >= 0) listeners.splice(idx, 1);
      };
    }),
    prompt: vi.fn().mockImplementation(() => {
      for (const listener of listeners) {
        listener({ type: "agent_end", messages: [] });
      }
    }),
    dispose: vi.fn(),
    _listeners: listeners
  };
  return {
    mockSession: session,
    createAgentSession: vi.fn().mockResolvedValue({
      session,
      extensionsResult: { extensions: [], diagnostics: [] }
    }),
    DefaultResourceLoader: vi.fn().mockImplementation(function () {
      return {
        reload: vi.fn().mockResolvedValue(undefined),
        getSkills: () => ({ skills: [], diagnostics: [] }),
        getPrompts: () => ({ prompts: [], diagnostics: [] }),
        getSystemPrompt: () => undefined,
        getAppendSystemPrompt: () => [],
        getAgentsFiles: () => ({ agentsFiles: [] })
      };
    }),
    SessionManager: {
      inMemory: vi.fn().mockReturnValue({
        create: vi.fn(),
        load: vi.fn().mockResolvedValue(undefined),
        save: vi.fn()
      })
    },
    AuthStorage: {
      inMemory: vi.fn().mockReturnValue({
        setRuntimeApiKey: vi.fn(),
        removeRuntimeApiKey: vi.fn(),
        getApiKey: vi.fn().mockResolvedValue("test-key")
      })
    }
  };
});
vi.mock("@earendil-works/pi-coding-agent", () => ({
  createAgentSession: piSdkMocks.createAgentSession,
  DefaultResourceLoader: piSdkMocks.DefaultResourceLoader,
  SessionManager: piSdkMocks.SessionManager,
  AuthStorage: piSdkMocks.AuthStorage
}));

// ── Fixtures ──────────────────────────────────────────────────────────────────

const resolvedLlmConfig: ResolvedLlmConfig = {
  baseUrl: "https://llm.example.com",
  apiKeyEnv: "LLM_API_KEY",
  modelId: "llm-model",
  apiKey: "llm-key"
};

function makeAgentSdkConfig(type: "claude" | "codebuddy" | "pi"): ResolvedAgentSdkConfig {
  return {
    type,
    baseUrl: "https://sdk.example.com",
    apiKeyEnv: "SDK_API_KEY",
    modelId: "sdk-model",
    apiKey: "sdk-key"
  };
}

const rawLogger: JsonlLogger = {
  filePath: "test.jsonl",
  write: vi.fn().mockResolvedValue(undefined)
};

function makeRoleConfig(name: string, overrides?: Partial<StoredRoleConfig>): StoredRoleConfig {
  return {
    name,
    allowedTools: ["Read"],
    permissionMode: "dontAsk",
    systemPrompt: `Role: ${name}`,
    updatedAt: "2026-01-01",
    ...overrides
  };
}

function makeRoleConfigStore(configs: StoredRoleConfig[]): RoleConfigStore {
  return {
    getAll: vi.fn().mockResolvedValue(configs),
    getByName: vi.fn().mockImplementation((name: string) => {
      const config = configs.find((c) => c.name === name);
      return Promise.resolve(config);
    }),
    upsert: vi.fn().mockResolvedValue(undefined),
    deleteByName: vi.fn().mockResolvedValue(false)
  };
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe("createAgentRuntimes", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    claudeSdkMocks.query.mockReturnValue(
      streamMessages({ type: "result", subtype: "success", result: "done" })
    );
    codebuddySdkMocks.query.mockReturnValue(
      streamMessages({ type: "result", subtype: "success", result: "done" })
    );
  });

  it("creates ClaudeSdkAgentRuntime when agentSdkConfig.type is claude", async () => {
    const store = makeRoleConfigStore([makeRoleConfig("reviewer")]);
    const runtimes = await createAgentRuntimes(
      rawLogger,
      store,
      resolvedLlmConfig,
      makeAgentSdkConfig("claude")
    );

    const reviewer = runtimes["reviewer"];
    expect(reviewer).toBeDefined();
    if (reviewer === undefined) return;
    expect(reviewer.name).toBe("claude-sdk-reviewer");
  });

  it("creates CodebuddySdkAgentRuntime when agentSdkConfig.type is codebuddy", async () => {
    const store = makeRoleConfigStore([makeRoleConfig("reviewer")]);
    const runtimes = await createAgentRuntimes(
      rawLogger,
      store,
      resolvedLlmConfig,
      makeAgentSdkConfig("codebuddy")
    );

    const reviewer = runtimes["reviewer"];
    expect(reviewer).toBeDefined();
    if (reviewer === undefined) return;
    expect(reviewer.name).toBe("codebuddy-sdk-reviewer");
  });

  it("creates PiAgentRuntime when agentSdkConfig.type is pi", async () => {
    const store = makeRoleConfigStore([makeRoleConfig("reviewer")]);
    const runtimes = await createAgentRuntimes(
      rawLogger,
      store,
      resolvedLlmConfig,
      makeAgentSdkConfig("pi")
    );

    const reviewer = runtimes["reviewer"];
    expect(reviewer).toBeDefined();
    if (reviewer === undefined) return;
    expect(reviewer.name).toBe("pi-reviewer");
  });

  it("creates one runtime per role config from the store", async () => {
    const configs = [
      makeRoleConfig("reviewer"),
      makeRoleConfig("editor", { allowedTools: ["Read", "Edit"], permissionMode: "acceptEdits" })
    ];
    const store = makeRoleConfigStore(configs);
    const runtimes = await createAgentRuntimes(
      rawLogger,
      store,
      resolvedLlmConfig,
      makeAgentSdkConfig("claude")
    );

    expect(Object.keys(runtimes)).toEqual(["reviewer", "editor"]);
    const reviewer = runtimes["reviewer"];
    const editor = runtimes["editor"];
    if (reviewer === undefined || editor === undefined) return;
    expect(reviewer.name).toBe("claude-sdk-reviewer");
    expect(editor.name).toBe("claude-sdk-editor");
  });
});

describe("createAgentRuntimeFactory", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    claudeSdkMocks.query.mockReturnValue(
      streamMessages({ type: "result", subtype: "success", result: "done" })
    );
    codebuddySdkMocks.query.mockReturnValue(
      streamMessages({ type: "result", subtype: "success", result: "done" })
    );
  });

  it("warmup caches runtimes and does not re-fetch from the store", async () => {
    const configs = [makeRoleConfig("reviewer")];
    const getAll = vi.fn().mockResolvedValue(configs);
    const store: RoleConfigStore = {
      getAll,
      getByName: vi.fn().mockResolvedValue(undefined),
      upsert: vi.fn().mockResolvedValue(undefined),
      deleteByName: vi.fn().mockResolvedValue(false)
    };
    const factory = createAgentRuntimeFactory(
      rawLogger,
      store,
      resolvedLlmConfig,
      makeAgentSdkConfig("claude")
    );

    const first = await factory.warmup();
    const second = await factory.warmup();

    expect(first).toBe(second);
    expect(getAll).toHaveBeenCalledTimes(1);
  });

  it("createRuntime returns the correct runtime for an existing role", async () => {
    const store = makeRoleConfigStore([makeRoleConfig("reviewer")]);
    const factory = createAgentRuntimeFactory(
      rawLogger,
      store,
      resolvedLlmConfig,
      makeAgentSdkConfig("codebuddy")
    );

    const runtime = await factory.createRuntime("reviewer");

    expect(runtime).toBeDefined();
    if (runtime === undefined) return;
    expect(runtime.name).toBe("codebuddy-sdk-reviewer");
  });

  it("createRuntime returns undefined for an unknown role", async () => {
    const store = makeRoleConfigStore([makeRoleConfig("reviewer")]);
    const factory = createAgentRuntimeFactory(
      rawLogger,
      store,
      resolvedLlmConfig,
      makeAgentSdkConfig("claude")
    );

    const runtime = await factory.createRuntime("nonexistent");

    expect(runtime).toBeUndefined();
  });

  it("createRuntime does not create an agent runtime for intent classification", async () => {
    const store = makeRoleConfigStore([]);
    const factory = createAgentRuntimeFactory(
      rawLogger,
      store,
      resolvedLlmConfig,
      makeAgentSdkConfig("claude")
    );

    const runtime = await factory.createRuntime("intent");

    expect(runtime).toBeUndefined();
  });

  it("cacheKeyForRole returns undefined for unknown roles", async () => {
    const store = makeRoleConfigStore([makeRoleConfig("reviewer")]);
    const factory = createAgentRuntimeFactory(
      rawLogger,
      store,
      resolvedLlmConfig,
      makeAgentSdkConfig("claude")
    );

    const key = await factory.cacheKeyForRole?.("unknown");

    expect(key).toBeUndefined();
  });

  it("cacheKeyForRole changes when the role config update timestamp changes", async () => {
    const store = makeRoleConfigStore([
      makeRoleConfig("reviewer", { updatedAt: "2026-06-09T00:00:00.000Z" }),
      makeRoleConfig("developer", { updatedAt: "2026-06-09T00:01:00.000Z" })
    ]);
    const factory = createAgentRuntimeFactory(
      rawLogger,
      store,
      resolvedLlmConfig,
      makeAgentSdkConfig("claude")
    );

    await expect(factory.cacheKeyForRole?.("reviewer")).resolves.toBe(
      "reviewer:2026-06-09T00:00:00.000Z"
    );
    await expect(factory.cacheKeyForRole?.("developer")).resolves.toBe(
      "developer:2026-06-09T00:01:00.000Z"
    );
  });

  it("cacheKeyForRole returns undefined for the intent classifier", async () => {
    const store = makeRoleConfigStore([]);
    const factory = createAgentRuntimeFactory(
      rawLogger,
      store,
      resolvedLlmConfig,
      makeAgentSdkConfig("claude")
    );

    const key = await factory.cacheKeyForRole?.("intent");

    expect(key).toBeUndefined();
  });
});

// ── Helpers ───────────────────────────────────────────────────────────────────

function streamMessages(...messages: readonly unknown[]): AsyncIterable<unknown> {
  return {
    async *[Symbol.asyncIterator]() {
      for (const message of messages) {
        await Promise.resolve();
        yield message;
      }
    }
  };
}
