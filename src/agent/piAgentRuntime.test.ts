import { beforeEach, describe, expect, it, vi } from "vitest";
import type { AgentStreamEvent } from "./index.js";
import type { StoredRoleConfig } from "../auth/index.js";
import type { JsonlLogger } from "../logging/jsonlLogger.js";
import { PiAgentRuntime, _piToolsForRole } from "./piAgentRuntime.js";

// ── Mock pi-coding-agent ────────────────────────────────────────────────────

const { mockSession, mockCreateAgentSession, mockResourceLoaderReload, mockAuthStorage } = vi.hoisted(() => {
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
    prompt: vi.fn(),
    dispose: vi.fn(),
    _listeners: listeners,
    _emit(event: unknown) {
      for (const listener of listeners) {
        listener(event);
      }
    },
  };

  const createAgentSession = vi.fn().mockResolvedValue({
    session,
    extensionsResult: { extensions: [], diagnostics: [] },
  });

  const resourceLoaderReload = vi.fn().mockResolvedValue(undefined);

  const authStorage = {
    setRuntimeApiKey: vi.fn(),
    removeRuntimeApiKey: vi.fn(),
    getApiKey: vi.fn().mockResolvedValue("test-key"),
  };

  return {
    mockSession: session,
    mockCreateAgentSession: createAgentSession,
    mockResourceLoaderReload: resourceLoaderReload,
    mockAuthStorage: authStorage,
  };
});

vi.mock("@earendil-works/pi-coding-agent", () => ({
  createAgentSession: mockCreateAgentSession,
  DefaultResourceLoader: vi.fn().mockImplementation(function() {
    return {
      reload: mockResourceLoaderReload,
      getSkills: () => ({ skills: [], diagnostics: [] }),
      getPrompts: () => ({ prompts: [], diagnostics: [] }),
      getSystemPrompt: () => undefined,
      getAppendSystemPrompt: () => [],
      getAgentsFiles: () => ({ agentsFiles: [] }),
    };
  }),
  SessionManager: {
    inMemory: vi.fn().mockReturnValue({
      create: vi.fn(),
      load: vi.fn().mockResolvedValue(undefined),
      save: vi.fn(),
    }),
  },
  AuthStorage: {
    inMemory: vi.fn().mockReturnValue(mockAuthStorage),
  },
}));

// ── Test fixtures ───────────────────────────────────────────────────────────

const agentSdkConfig = {
  type: "pi" as const,
  baseUrl: "https://api.example.com",
  apiKeyEnv: "TEST_API_KEY",
  modelId: "test-model",
  apiKey: "test-key",
};

const roleConfig: StoredRoleConfig = {
  name: "reviewer",
  allowedTools: ["Read", "Grep"],
  permissionMode: "dontAsk",
  systemPrompt: "Answer from the workspace.",
};

// ── Tests ───────────────────────────────────────────────────────────────────

describe("PiAgentRuntime", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockSession._listeners.length = 0;
    mockSession.prompt.mockReset();
    mockSession.prompt.mockResolvedValue(undefined);
  });

  it("creates a session and sends a prompt", async () => {
    // Simulate agent_end after prompt
    mockSession.prompt.mockImplementation(() => {
      mockSession._emit({
        type: "message_update",
        assistantMessageEvent: { type: "text_delta", delta: "final answer" },
        message: { role: "assistant", content: [], timestamp: Date.now() },
      });
      mockSession._emit({
        type: "agent_end",
        messages: [],
      });
    });

    const runtime = new PiAgentRuntime({ roleConfig, agentSdkConfig });
    const response = await runtime.run({
      user: { id: "user-1" },
      text: "What is the architecture?",
      workspacePath: "D:/workspace",
    });

    expect(mockCreateAgentSession).toHaveBeenCalled();
    expect(mockSession.prompt).toHaveBeenCalled();
    expect(response.text).toContain("final answer");
    expect(response.sessionId).toBeDefined();
  });

  it("returns the final text from text_delta events", async () => {
    mockSession.prompt.mockImplementation(() => {
      mockSession._emit({
        type: "message_update",
        assistantMessageEvent: { type: "text_delta", delta: "The architecture is layered." },
        message: { role: "assistant", content: [], timestamp: Date.now() },
      });
      mockSession._emit({ type: "agent_end", messages: [] });
    });

    const runtime = new PiAgentRuntime({ roleConfig, agentSdkConfig });
    const response = await runtime.run({
      user: { id: "user-1" },
      text: "What is the architecture?",
      workspacePath: "D:/workspace",
    });

    expect(response.text).toBe("The architecture is layered.");
  });

  it("forwards text_delta stream events", async () => {
    mockSession.prompt.mockImplementation(() => {
      mockSession._emit({
        type: "message_update",
        assistantMessageEvent: { type: "text_delta", delta: "hel" },
        message: { role: "assistant", content: [], timestamp: Date.now() },
      });
      mockSession._emit({
        type: "message_update",
        assistantMessageEvent: { type: "text_delta", delta: "lo" },
        message: { role: "assistant", content: [], timestamp: Date.now() },
      });
      mockSession._emit({ type: "agent_end", messages: [] });
    });

    const streamEvents: AgentStreamEvent[] = [];
    const runtime = new PiAgentRuntime({ roleConfig, agentSdkConfig });

    await runtime.run({
      user: { id: "user-1" },
      text: "Test",
      workspacePath: "D:/workspace",
      stream: (event) => streamEvents.push(event),
    });

    expect(streamEvents).toContainEqual({ type: "text_delta", text: "hel" });
    expect(streamEvents).toContainEqual({ type: "text_delta", text: "lo" });
    const completedEvent = streamEvents.find((e) => e.type === "completed");
    expect(completedEvent).toBeDefined();
  });

  it("forwards thinking_delta stream events", async () => {
    mockSession.prompt.mockImplementation(() => {
      mockSession._emit({
        type: "message_update",
        assistantMessageEvent: { type: "thinking_delta", delta: "reasoning" },
        message: { role: "assistant", content: [], timestamp: Date.now() },
      });
      mockSession._emit({ type: "agent_end", messages: [] });
    });

    const streamEvents: AgentStreamEvent[] = [];
    const runtime = new PiAgentRuntime({ roleConfig, agentSdkConfig });

    await runtime.run({
      user: { id: "user-1" },
      text: "Test",
      workspacePath: "D:/workspace",
      stream: (event) => streamEvents.push(event),
    });

    expect(streamEvents).toContainEqual({ type: "thinking", text: "reasoning" });
  });

  it("reuses session on subsequent calls with same sessionId", async () => {
    mockSession.prompt.mockImplementation(() => {
      mockSession._emit({ type: "agent_end", messages: [] });
    });

    const runtime = new PiAgentRuntime({ roleConfig, agentSdkConfig });

    const firstResponse = await runtime.run({
      user: { id: "user-1" },
      text: "First question",
      workspacePath: "D:/workspace",
    });

    const sessionId = firstResponse.sessionId;
    expect(sessionId).toBeDefined();
    if (sessionId === undefined) return;

    // Second call with the same sessionId should reuse the session
    await runtime.run({
      user: { id: "user-1" },
      text: "Follow-up",
      workspacePath: "D:/workspace",
      sessionId,
    });

    // createAgentSession should only be called once (session reused)
    expect(mockCreateAgentSession).toHaveBeenCalledTimes(1);
    // prompt should be called twice
    expect(mockSession.prompt).toHaveBeenCalledTimes(2);
  });

  it("disposes session correctly", async () => {
    mockSession.prompt.mockImplementation(() => {
      mockSession._emit({ type: "agent_end", messages: [] });
    });

    const runtime = new PiAgentRuntime({ roleConfig, agentSdkConfig });

    const response = await runtime.run({
      user: { id: "user-1" },
      text: "Test",
      workspacePath: "D:/workspace",
    });

    const sid = response.sessionId;
    if (sid === undefined) return;

    await runtime.disposeSession(sid);
    expect(mockSession.dispose).toHaveBeenCalled();
  });

  it("writes raw logs for request and response", async () => {
    mockSession.prompt.mockImplementation(() => {
      mockSession._emit({
        type: "message_end",
        message: {
          role: "assistant",
          content: [{ type: "text", text: "answer" }],
          usage: { input: 100, output: 50, cacheRead: 0, cacheWrite: 0, totalTokens: 150 },
          timestamp: Date.now(),
        },
      });
      mockSession._emit({ type: "agent_end", messages: [] });
    });

    const rawEvents: Record<string, unknown>[] = [];
    const rawLogger: JsonlLogger = {
      filePath: "test.jsonl",
      write(event) {
        rawEvents.push(event);
        return Promise.resolve();
      },
    };
    const runtime = new PiAgentRuntime({ roleConfig, agentSdkConfig, rawLogger });

    await runtime.run({
      user: { id: "user-1" },
      text: "Test",
      workspacePath: "D:/workspace",
    });

    const types = rawEvents.map((e) => e["type"]);
    expect(types).toContain("llm.request");
    expect(rawEvents[0]).toMatchObject({
      type: "llm.request",
      operation: "agent_runtime",
      runtime: "pi-reviewer",
      userId: "user-1",
      workspacePath: "D:/workspace",
    });
  });

  it("writes error logs when prompt fails", async () => {
    mockSession.prompt.mockRejectedValue(new Error("Prompt failed"));

    const rawEvents: Record<string, unknown>[] = [];
    const rawLogger: JsonlLogger = {
      filePath: "test.jsonl",
      write(event) {
        rawEvents.push(event);
        return Promise.resolve();
      },
    };
    const runtime = new PiAgentRuntime({ roleConfig, agentSdkConfig, rawLogger });

    await expect(
      runtime.run({
        user: { id: "user-1" },
        text: "Test",
        workspacePath: "D:/workspace",
      })
    ).rejects.toThrow("Prompt failed");

    expect(rawEvents.map((e) => e["type"])).toContain("llm.error");
  });

  it("returns a default message when the response has no text", async () => {
    mockSession.prompt.mockImplementation(() => {
      mockSession._emit({ type: "agent_end", messages: [] });
    });

    const runtime = new PiAgentRuntime({ roleConfig, agentSdkConfig });

    const response = await runtime.run({
      user: { id: "user-1" },
      text: "Test",
      workspacePath: "D:/workspace",
    });

    expect(response.text).toBe("Agent completed without text output.");
  });

  it("sets API key via AuthStorage", async () => {
    mockSession.prompt.mockImplementation(() => {
      mockSession._emit({ type: "agent_end", messages: [] });
    });

    const runtime = new PiAgentRuntime({ roleConfig, agentSdkConfig });
    await runtime.run({
      user: { id: "user-1" },
      text: "Test",
      workspacePath: "D:/workspace",
    });

    expect(mockAuthStorage.setRuntimeApiKey).toHaveBeenCalledWith("pets-agent", "test-key");
  });

  it("does not enable Bash for non-mutating roles because Pi has no dynamic Bash permission decider", () => {
    expect(_piToolsForRole({
      name: "reviewer",
      allowedTools: ["Read", "Bash", "Grep"],
      permissionMode: "dontAsk",
      systemPrompt: "Read only.",
    })).toEqual(["Read", "Grep"]);
  });

  it("preserves Bash for roles that can mutate files", () => {
    expect(_piToolsForRole({
      name: "developer",
      allowedTools: ["Read", "Edit", "Bash"],
      permissionMode: "bypassPermissions",
      systemPrompt: "Can edit.",
    })).toEqual(["Read", "Edit", "Bash"]);
  });

  it("passes an explicit empty tools list when the role has no tools", async () => {
    mockSession.prompt.mockImplementation(() => {
      mockSession._emit({ type: "agent_end", messages: [] });
    });

    const runtime = new PiAgentRuntime({
      roleConfig: {
        name: "no-tools",
        allowedTools: [],
        permissionMode: "dontAsk",
        systemPrompt: "No tools.",
      },
      agentSdkConfig,
    });

    await runtime.run({
      user: { id: "user-1" },
      text: "Test",
      workspacePath: "D:/workspace",
    });

    const call = mockCreateAgentSession.mock.calls[0]?.[0] as { readonly tools?: readonly string[] } | undefined;
    expect(call?.tools).toEqual([]);
  });
});
