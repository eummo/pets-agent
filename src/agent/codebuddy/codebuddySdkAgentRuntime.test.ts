import { mkdtemp, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { AgentStreamEvent } from "../index.js";
import type { StoredRoleConfig } from "../../auth/index.js";
import type { JsonlLogger } from "../../logging/jsonlLogger.js";
import { CodebuddySdkAgentRuntime } from "./codebuddySdkAgentRuntime.js";

const sdkMocks = vi.hoisted(() => ({
  query: vi.fn()
}));

vi.mock("@tencent-ai/agent-sdk", () => ({
  query: sdkMocks.query
}));

const agentSdkConfig = {
  type: "codebuddy" as const,
  baseUrl: "https://api.example.com",
  apiKeyEnv: "LOCAL_LLM_API_KEY",
  modelId: "test-model",
  apiKey: "test-api-key"
};

const roleConfig: StoredRoleConfig = {
  name: "tester",
  allowedTools: ["Read", "Grep"],
  permissionMode: "dontAsk",
  systemPrompt: "Answer from the workspace.",
  maxTurns: 3
};

describe("CodebuddySdkAgentRuntime", () => {
  beforeEach(() => {
    sdkMocks.query.mockReset();
    delete process.env["CODEBUDDY_AUTH_TOKEN"];
  });

  it("builds SDK query options from the role config and request", async () => {
    sdkMocks.query.mockReturnValue(
      streamMessages({
        type: "result",
        subtype: "success",
        session_id: "session-2",
        result: "final answer"
      })
    );
    const runtime = new CodebuddySdkAgentRuntime({ roleConfig, agentSdkConfig, model: "model-1" });

    await runtime.run({
      user: { id: "user-1" },
      text: "What changed?",
      workspacePath: "D:/workspace",
      sessionId: "session-1"
    });

    expect(runtime.name).toBe("codebuddy-sdk-tester");
    const call = firstQueryCall();
    expect(call.prompt).toBe("What changed?");
    expect(call.options).toMatchObject({
      cwd: "D:/workspace",
      tools: ["Read", "Grep"],
      allowedTools: [],
      disallowedTools: [],
      permissionMode: "dontAsk",
      allowDangerouslySkipPermissions: false,
      systemPrompt: "Answer from the workspace.",
      includePartialMessages: true,
      maxTurns: 3,
      model: "model-1",
      resume: "session-1",
      env: {
        CODEBUDDY_API_KEY: "test-api-key"
      }
    });
  });

  it("passes the API key via env option", async () => {
    sdkMocks.query.mockReturnValue(
      streamMessages({ type: "result", subtype: "success", result: "done" })
    );
    const runtime = new CodebuddySdkAgentRuntime({ roleConfig, agentSdkConfig });

    await runtime.run({
      user: { id: "user-1" },
      text: "Test",
      workspacePath: "D:/workspace"
    });

    const call = firstQueryCall();
    expect(call.options).toBeDefined();
    expect(call.options["env"]).toEqual({
      CODEBUDDY_API_KEY: "test-api-key"
    });
  });

  it("passes enterprise endpoint to the CLI startup env", async () => {
    sdkMocks.query.mockReturnValue(
      streamMessages({ type: "result", subtype: "success", result: "done" })
    );
    const runtime = new CodebuddySdkAgentRuntime({
      roleConfig,
      agentSdkConfig: {
        ...agentSdkConfig,
        endpoint: "https://enterprise.example.com/",
        environment: "internal"
      }
    });

    await runtime.run({
      user: { id: "user-1" },
      text: "Test",
      workspacePath: "D:/workspace"
    });

    const call = firstQueryCall();
    expect(call.options["env"]).toMatchObject({
      ACC_PRODUCT_CONFIG_V3: JSON.stringify({
        endpoint: "https://enterprise.example.com",
        stagingEndpoint: "https://enterprise.example.com"
      }),
      CODEBUDDY_BASE_URL: "https://enterprise.example.com/v2",
      CODEBUDDY_INTERNET_ENVIRONMENT: "internal",
      CODEBUDDY_API_KEY: "test-api-key"
    });
    expect(call.options["endpoint"]).toBeUndefined();
    expect(call.options["environment"]).toBe("internal");
  });

  it("passes predefined environment when no enterprise endpoint is configured", async () => {
    sdkMocks.query.mockReturnValue(
      streamMessages({ type: "result", subtype: "success", result: "done" })
    );
    const runtime = new CodebuddySdkAgentRuntime({
      roleConfig,
      agentSdkConfig: {
        ...agentSdkConfig,
        environment: "internal"
      }
    });

    await runtime.run({
      user: { id: "user-1" },
      text: "Test",
      workspacePath: "D:/workspace"
    });

    const call = firstQueryCall();
    expect(call.options["environment"]).toBe("internal");
    expect(call.options["endpoint"]).toBeUndefined();
    expect(call.options["env"]).toMatchObject({
      CODEBUDDY_INTERNET_ENVIRONMENT: "internal",
      CODEBUDDY_API_KEY: "test-api-key"
    });
  });

  it("passes enterprise auth token when present", async () => {
    process.env["CODEBUDDY_AUTH_TOKEN"] = "enterprise-token";
    sdkMocks.query.mockReturnValue(
      streamMessages({ type: "result", subtype: "success", result: "done" })
    );
    const runtime = new CodebuddySdkAgentRuntime({ roleConfig, agentSdkConfig });

    await runtime.run({
      user: { id: "user-1" },
      text: "Test",
      workspacePath: "D:/workspace"
    });

    const call = firstQueryCall();
    // When CODEBUDDY_AUTH_TOKEN is set, it takes priority over CODEBUDDY_API_KEY.
    expect(call.options["env"]).toMatchObject({
      CODEBUDDY_AUTH_TOKEN: "enterprise-token"
    });
    // CODEBUDDY_API_KEY should NOT be present when auth token is set
    expect((call.options["env"] as Record<string, string>)["CODEBUDDY_API_KEY"]).toBeUndefined();
  });

  it("omits auth env when no credentials are configured", async () => {
    sdkMocks.query.mockReturnValue(
      streamMessages({ type: "result", subtype: "success", result: "done" })
    );
    const runtime = new CodebuddySdkAgentRuntime({
      roleConfig,
      agentSdkConfig: {
        ...agentSdkConfig,
        apiKeyEnv: undefined,
        apiKey: ""
      }
    });

    await runtime.run({
      user: { id: "user-1" },
      text: "Test",
      workspacePath: "D:/workspace"
    });

    const call = firstQueryCall();
    // When no auth is explicitly configured, no auth env vars are set
    // (the CLI subprocess will try its own cached credentials).
    expect(call.options["env"]).toBeUndefined();
  });

  it("checks read-only tool paths against the selected workspace", async () => {
    sdkMocks.query.mockReturnValue(
      streamMessages({ type: "result", subtype: "success", result: "done" })
    );
    const runtime = new CodebuddySdkAgentRuntime({
      roleConfig: {
        name: "reviewer",
        allowedTools: ["Read", "Grep"],
        permissionMode: "dontAsk",
        systemPrompt: "Read only."
      },
      agentSdkConfig
    });

    await runtime.run({
      user: { id: "user-1" },
      text: "Inspect files",
      workspacePath: "D:/code/pets-agent/.harness/knowledge-base"
    });

    const call = firstQueryCall();
    const canUseTool = call.options["canUseTool"];
    if (!isCanUseTool(canUseTool)) {
      throw new Error("Expected canUseTool callback.");
    }

    await expect(
      canUseTool("Read", {
        file_path: "D:/code/pets-agent/src/core/contracts.ts"
      })
    ).resolves.toMatchObject({
      behavior: "deny",
      message: expect.stringContaining("outside the selected workspace") as string
    });
    await expect(
      canUseTool("Grep", {
        path: "D:/code/pets-agent/.harness/knowledge-base/docs"
      })
    ).resolves.toMatchObject({
      behavior: "allow",
      updatedInput: {
        path: "D:/code/pets-agent/.harness/knowledge-base/docs"
      }
    });
  });

  it("returns the final result and forwards stream events", async () => {
    sdkMocks.query.mockReturnValue(
      streamMessages(
        {
          type: "assistant",
          session_id: "session-2",
          message: {
            content: [
              {
                type: "tool_use",
                id: "tool-1",
                name: "Read",
                input: { file_path: "README.md" }
              },
              {
                type: "tool_result",
                tool_use_id: "tool-1",
                content: [{ text: "file content" }]
              }
            ]
          }
        },
        {
          type: "stream_event",
          event: {
            type: "content_block_delta",
            delta: { type: "text_delta", text: "partial" }
          }
        },
        {
          type: "result",
          subtype: "success",
          session_id: "session-2",
          result: "final answer"
        }
      )
    );
    const streamEvents: AgentStreamEvent[] = [];
    const rawEvents: Record<string, unknown>[] = [];
    const rawLogger: JsonlLogger = {
      filePath: "memory.jsonl",
      write(event) {
        rawEvents.push(event);
        return Promise.resolve();
      }
    };
    const runtime = new CodebuddySdkAgentRuntime({ roleConfig, agentSdkConfig, rawLogger });

    const response = await runtime.run({
      user: { id: "user-1" },
      text: "Inspect files",
      workspacePath: "D:/workspace",
      stream: (event) => streamEvents.push(event)
    });

    expect(response).toEqual({ text: "final answer", sessionId: "session-2" });
    expect(streamEvents).toEqual([
      {
        type: "tool_use_start",
        toolName: "Read",
        toolUseId: "tool-1",
        input: { file_path: "README.md" }
      },
      {
        type: "tool_use_result",
        toolUseId: "tool-1",
        result: "file content"
      },
      { type: "text_delta", text: "partial" }
    ]);
    expect(rawEvents.map((event) => event["type"])).toEqual([
      "llm.request",
      "agent.tool_call",
      "agent.tool_result",
      "llm.response"
    ]);
  });

  it("returns a user-facing error when the SDK result is not successful", async () => {
    sdkMocks.query.mockReturnValue(
      streamMessages({
        type: "result",
        subtype: "error",
        errors: ["permission denied"]
      })
    );
    const runtime = new CodebuddySdkAgentRuntime({ roleConfig, agentSdkConfig });

    await expect(
      runtime.run({
        user: { id: "user-1" },
        text: "Edit files",
        workspacePath: "D:/workspace"
      })
    ).resolves.toEqual({ text: "Agent error: permission denied" });
  });

  it("resolves disposeSession without calling the SDK", async () => {
    const runtime = new CodebuddySdkAgentRuntime({ roleConfig, agentSdkConfig });

    await expect(runtime.disposeSession("test-session")).resolves.toBeUndefined();
  });

  it("reports an error for unauthorized mutating tool events", async () => {
    sdkMocks.query.mockReturnValue(
      streamMessages(
        {
          type: "assistant",
          session_id: "session-2",
          message: {
            content: [
              {
                type: "tool_use",
                id: "tool-1",
                name: "Edit",
                input: { file_path: "README.md" }
              }
            ]
          }
        },
        {
          type: "result",
          subtype: "success",
          session_id: "session-2",
          result: "final answer"
        }
      )
    );
    const streamEvents: AgentStreamEvent[] = [];
    const runtime = new CodebuddySdkAgentRuntime({
      roleConfig: {
        name: "reviewer",
        allowedTools: ["Read", "Edit"],
        permissionMode: "dontAsk",
        systemPrompt: "Read only."
      },
      agentSdkConfig
    });

    await runtime.run({
      user: { id: "user-1" },
      text: "Inspect files",
      workspacePath: "D:/workspace",
      stream: (event) => streamEvents.push(event)
    });

    expect(streamEvents).toEqual([
      {
        type: "error",
        message: "Tool Edit is not permitted for role reviewer."
      }
    ]);
  });

  it("forwards compact_start and compact_complete stream events", async () => {
    sdkMocks.query.mockReturnValue(
      streamMessages(
        {
          type: "system",
          subtype: "status",
          status: "compacting"
        },
        {
          type: "system",
          subtype: "compact_boundary",
          compact_metadata: {
            trigger: "auto",
            pre_tokens: 180_000,
            post_tokens: 45_000,
            duration_ms: 1200
          }
        },
        {
          type: "result",
          subtype: "success",
          session_id: "session-compact",
          result: "answer after compaction"
        }
      )
    );
    const streamEvents: AgentStreamEvent[] = [];
    const runtime = new CodebuddySdkAgentRuntime({ roleConfig, agentSdkConfig });

    const response = await runtime.run({
      user: { id: "user-1" },
      text: "Continue",
      workspacePath: "D:/workspace",
      stream: (event) => streamEvents.push(event)
    });

    expect(response.text).toBe("answer after compaction");
    expect(streamEvents).toEqual([
      { type: "compact_start" },
      {
        type: "compact_complete",
        preTokens: 180_000,
        postTokens: 45_000,
        durationMs: 1200
      }
    ]);
  });

  it("extracts context usage from the SDK result", async () => {
    sdkMocks.query.mockReturnValue(
      streamMessages({
        type: "result",
        subtype: "success",
        session_id: "session-usage",
        result: "answer",
        usage: {
          input_tokens: 120_000,
          output_tokens: 500,
          cache_creation_input_tokens: 30_000,
          cache_read_input_tokens: 80_000
        }
      })
    );
    const runtime = new CodebuddySdkAgentRuntime({
      roleConfig,
      agentSdkConfig,
      contextConfig: {
        autoCompactEnabled: true,
        autoCompactWindow: 150_000,
        workspaceMaxChars: 6_000,
        historyMaxMessages: 30
      }
    });

    const response = await runtime.run({
      user: { id: "user-1" },
      text: "Test",
      workspacePath: "D:/workspace"
    });

    expect(response.contextUsage).toEqual({
      inputTokens: 120_000,
      outputTokens: 500,
      cacheCreationTokens: 30_000,
      cacheReadTokens: 80_000,
      contextWindow: 150_000,
      usagePercent: 80
    });
  });

  it("omits context usage when the SDK result has no usage field", async () => {
    sdkMocks.query.mockReturnValue(
      streamMessages({
        type: "result",
        subtype: "success",
        result: "answer"
      })
    );
    const runtime = new CodebuddySdkAgentRuntime({ roleConfig, agentSdkConfig });

    const response = await runtime.run({
      user: { id: "user-1" },
      text: "Test",
      workspacePath: "D:/workspace"
    });

    expect(response.contextUsage).toBeUndefined();
  });

  it("logs compact boundary events to the raw logger", async () => {
    sdkMocks.query.mockReturnValue(
      streamMessages(
        {
          type: "system",
          subtype: "compact_boundary",
          session_id: "session-compact",
          compact_metadata: {
            trigger: "auto",
            pre_tokens: 160_000,
            duration_ms: 800
          }
        },
        {
          type: "result",
          subtype: "success",
          session_id: "session-2",
          result: "done"
        }
      )
    );
    const rawEvents: Record<string, unknown>[] = [];
    const rawLogger: JsonlLogger = {
      filePath: "test.jsonl",
      write(event) {
        rawEvents.push(event);
        return Promise.resolve();
      }
    };
    const runtime = new CodebuddySdkAgentRuntime({ roleConfig, agentSdkConfig, rawLogger });

    await runtime.run({
      user: { id: "user-1" },
      text: "Continue",
      workspacePath: "D:/workspace"
    });

    const compactLog = rawEvents.find((e) => e["type"] === "llm.compact");
    expect(compactLog).toMatchObject({
      type: "llm.compact",
      runtime: "codebuddy-sdk-tester",
      userId: "user-1",
      workspacePath: "D:/workspace",
      sessionId: "session-compact",
      trigger: "auto",
      preTokens: 160_000,
      durationMs: 800
    });
  });

  it("writes detailed llm.request and llm.response raw logs", async () => {
    sdkMocks.query.mockReturnValue(
      streamMessages({
        type: "result",
        subtype: "success",
        session_id: "session-2",
        result: "final answer"
      })
    );
    const rawEvents: Record<string, unknown>[] = [];
    const rawLogger: JsonlLogger = {
      filePath: "memory.jsonl",
      write(event) {
        rawEvents.push(event);
        return Promise.resolve();
      }
    };
    const runtime = new CodebuddySdkAgentRuntime({ roleConfig, agentSdkConfig, rawLogger });

    await runtime.run({
      user: { id: "user-1" },
      text: "Inspect files",
      workspacePath: "D:/workspace"
    });

    expect(rawEvents[0]).toMatchObject({
      type: "llm.request",
      operation: "agent_runtime",
      runtime: "codebuddy-sdk-tester",
      userId: "user-1",
      workspacePath: "D:/workspace"
    });
    expect(String(rawEvents[0]?.["prompt"])).toContain("Inspect files");
    const loggedOptions = asRecord(rawEvents[0]?.["options"]);
    expect(loggedOptions["cwd"]).toBe("D:/workspace");
    expect(loggedOptions["tools"]).toEqual(["Read", "Grep"]);

    const rawEvent = rawEvents[1];
    expect(rawEvent).toMatchObject({
      type: "llm.response",
      operation: "agent_runtime",
      runtime: "codebuddy-sdk-tester",
      userId: "user-1",
      workspacePath: "D:/workspace",
      sessionId: "session-2",
      extractedText: "final answer"
    });
    if (rawEvent !== undefined) {
      expect(typeof rawEvent["durationMs"]).toBe("number");
    }
  });

  it("logs llm.error when the SDK stream throws during iteration", async () => {
    sdkMocks.query.mockReturnValue(
      (async function* () {
        await Promise.resolve();
        yield {
          type: "stream_event",
          event: { type: "content_block_delta", delta: { type: "text_delta", text: "partial" } }
        };
        throw new Error("stream broke");
      })()
    );
    const rawEvents: Record<string, unknown>[] = [];
    const rawLogger: JsonlLogger = {
      filePath: "error.jsonl",
      write(event) {
        rawEvents.push(event);
        return Promise.resolve();
      }
    };
    const runtime = new CodebuddySdkAgentRuntime({ roleConfig, agentSdkConfig, rawLogger });

    await expect(
      runtime.run({
        user: { id: "user-1" },
        text: "Test",
        workspacePath: "D:/workspace"
      })
    ).rejects.toThrow("stream broke");

    expect(rawEvents.map((e) => e["type"])).toContain("llm.error");
    const errorLog = rawEvents.find((e) => e["type"] === "llm.error");
    expect(errorLog).toMatchObject({
      type: "llm.error",
      operation: "agent_runtime",
      runtime: "codebuddy-sdk-tester",
      userId: "user-1",
      workspacePath: "D:/workspace",
      error: "stream broke"
    });
  });

  it("passes skills when the role config specifies skills: 'all'", async () => {
    sdkMocks.query.mockReturnValue(
      streamMessages({ type: "result", subtype: "success", result: "done" })
    );
    const runtime = new CodebuddySdkAgentRuntime({
      roleConfig: {
        name: "skilled",
        allowedTools: ["Read"],
        permissionMode: "dontAsk",
        systemPrompt: "Read only.",
        skills: "all",
        settingSources: ["project"]
      },
      agentSdkConfig
    });

    await runtime.run({
      user: { id: "user-1" },
      text: "Test",
      workspacePath: "D:/workspace"
    });

    const call = firstQueryCall();
    expect(call.options["skills"]).toBe("all");
    expect(call.options["settingSources"]).toEqual(["project"]);
  });

  it("passes a filtered skill list when the role specifies specific skills", async () => {
    sdkMocks.query.mockReturnValue(
      streamMessages({ type: "result", subtype: "success", result: "done" })
    );
    const runtime = new CodebuddySdkAgentRuntime({
      roleConfig: {
        name: "custom",
        allowedTools: ["Read"],
        permissionMode: "dontAsk",
        systemPrompt: "Custom.",
        skills: ["order-check"],
        settingSources: ["project"]
      },
      agentSdkConfig
    });

    await runtime.run({
      user: { id: "user-1" },
      text: "Test",
      workspacePath: "D:/workspace"
    });

    const call = firstQueryCall();
    expect(call.options["skills"]).toEqual(["order-check"]);
    expect(call.options["settingSources"]).toEqual(["project"]);
  });

  it("passes enableWorkflows when the role config specifies it", async () => {
    sdkMocks.query.mockReturnValue(
      streamMessages({ type: "result", subtype: "success", result: "done" })
    );
    const runtime = new CodebuddySdkAgentRuntime({
      roleConfig: {
        name: "workflow-user",
        allowedTools: ["Read", "Edit"],
        permissionMode: "acceptEdits",
        systemPrompt: "Can edit.",
        enableWorkflows: true
      },
      agentSdkConfig
    });

    await runtime.run({
      user: { id: "user-1" },
      text: "Test",
      workspacePath: "D:/workspace"
    });

    const call = firstQueryCall();
    const settings = call.options["settings"] as Record<string, unknown> | undefined;
    expect(settings).toBeDefined();
    expect(settings?.["enableWorkflows"]).toBe(true);
  });

  it("passes planModeInstructions when the role config specifies it", async () => {
    sdkMocks.query.mockReturnValue(
      streamMessages({ type: "result", subtype: "success", result: "done" })
    );
    const runtime = new CodebuddySdkAgentRuntime({
      roleConfig: {
        name: "planner",
        allowedTools: ["Read"],
        permissionMode: "dontAsk",
        systemPrompt: "Plan only.",
        planModeInstructions: "Always plan before executing."
      },
      agentSdkConfig
    });

    await runtime.run({
      user: { id: "user-1" },
      text: "Test",
      workspacePath: "D:/workspace"
    });

    const call = firstQueryCall();
    expect(call.options["planModeInstructions"]).toBe("Always plan before executing.");
  });

  it("grounds the prompt with workspace context when CLAUDE.md exists", async () => {
    sdkMocks.query.mockReturnValue(
      streamMessages({
        type: "result",
        subtype: "success",
        result: "final answer"
      })
    );
    const workspacePath = await mkdtemp(path.join(os.tmpdir(), "pets-agent-codebuddy-"));
    await writeFile(
      path.join(workspacePath, "CLAUDE.md"),
      "This workspace documents catalog checks and order lifecycle recording.",
      "utf8"
    );
    const runtime = new CodebuddySdkAgentRuntime({ roleConfig, agentSdkConfig });

    try {
      await runtime.run({
        user: { id: "user-1" },
        text: "What is the current architecture?",
        workspacePath
      });
    } finally {
      await rm(workspacePath, { recursive: true, force: true });
    }

    const call = firstQueryCall();
    expect(call.prompt).toContain("Selected workspace context:");
    expect(call.prompt).toContain("catalog checks and order lifecycle recording");
    expect(call.prompt).toContain("User request:\nWhat is the current architecture?");
  });
});

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

function firstQueryCall(): { readonly prompt: string; readonly options: Record<string, unknown> } {
  const calls = sdkMocks.query.mock.calls as unknown as [
    { readonly prompt: string; readonly options: Record<string, unknown> }
  ][];
  const call = calls[0]?.[0];
  if (call === undefined) {
    throw new Error("Expected SDK query to be called.");
  }
  return call;
}

function isCanUseTool(
  value: unknown
): value is (toolName: string, input: Record<string, unknown>) => Promise<unknown> {
  return typeof value === "function";
}

function asRecord(value: unknown): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`Expected object, got ${JSON.stringify(value)}.`);
  }
  return value as Record<string, unknown>;
}
