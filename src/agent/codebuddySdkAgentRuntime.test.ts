import { beforeEach, describe, expect, it, vi } from "vitest";
import type { AgentStreamEvent } from "./index.js";
import type { StoredRoleConfig } from "../auth/index.js";
import type { JsonlLogger } from "../logging/jsonlLogger.js";
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
  apiKeyEnv: "CODEBUDDY_API_KEY",
  modelId: "test-model",
  apiKey: "test-api-key",
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
        CODEBUDDY_API_KEY: "test-api-key",
      },
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
      workspacePath: "D:/workspace",
    });

    const call = firstQueryCall();
    expect(call.options).toBeDefined();
    expect(call.options["env"]).toEqual({
      CODEBUDDY_API_KEY: "test-api-key",
    });
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
        systemPrompt: "Read only.",
      },
      agentSdkConfig,
    });

    await runtime.run({
      user: { id: "user-1" },
      text: "Inspect files",
      workspacePath: "D:/code/pets-agent/.harness/knowledge-base",
    });

    const call = firstQueryCall();
    const canUseTool = call.options["canUseTool"];
    if (!isCanUseTool(canUseTool)) {
      throw new Error("Expected canUseTool callback.");
    }

    await expect(canUseTool("Read", {
      file_path: "D:/code/pets-agent/src/core/contracts.ts",
    })).resolves.toMatchObject({
      behavior: "deny",
      message: expect.stringContaining("outside the selected workspace") as string,
    });
    await expect(canUseTool("Grep", {
      path: "D:/code/pets-agent/.harness/knowledge-base/docs",
    })).resolves.toMatchObject({
      behavior: "allow",
      updatedInput: {
        path: "D:/code/pets-agent/.harness/knowledge-base/docs",
      },
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
      stream: (event) => streamEvents.push(event),
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
      { type: "text_delta", text: "partial" },
    ]);
    expect(rawEvents.map((event) => event["type"])).toEqual([
      "llm.request",
      "agent.tool_call",
      "agent.tool_result",
      "llm.response",
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

    await expect(runtime.disposeSession()).resolves.toBeUndefined();
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
        systemPrompt: "Read only.",
      },
      agentSdkConfig,
    });

    await runtime.run({
      user: { id: "user-1" },
      text: "Inspect files",
      workspacePath: "D:/workspace",
      stream: (event) => streamEvents.push(event),
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
          status: "compacting",
        },
        {
          type: "system",
          subtype: "compact_boundary",
          compact_metadata: {
            trigger: "auto",
            pre_tokens: 180_000,
            post_tokens: 45_000,
            duration_ms: 1200,
          },
        },
        {
          type: "result",
          subtype: "success",
          session_id: "session-compact",
          result: "answer after compaction",
        }
      )
    );
    const streamEvents: AgentStreamEvent[] = [];
    const runtime = new CodebuddySdkAgentRuntime({ roleConfig, agentSdkConfig });

    const response = await runtime.run({
      user: { id: "user-1" },
      text: "Continue",
      workspacePath: "D:/workspace",
      stream: (event) => streamEvents.push(event),
    });

    expect(response.text).toBe("answer after compaction");
    expect(streamEvents).toEqual([
      { type: "compact_start" },
      {
        type: "compact_complete",
        preTokens: 180_000,
        postTokens: 45_000,
        durationMs: 1200,
      },
    ]);
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

function isCanUseTool(value: unknown): value is (toolName: string, input: Record<string, unknown>) => Promise<unknown> {
  return typeof value === "function";
}
