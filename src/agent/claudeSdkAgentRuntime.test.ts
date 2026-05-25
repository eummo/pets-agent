import { mkdtemp, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { AgentStreamEvent, StoredRoleConfig } from "../core/contracts.js";
import type { JsonlLogger } from "../logging/jsonlLogger.js";
import { ClaudeSdkAgentRuntime } from "./claudeSdkAgentRuntime.js";

const sdkMocks = vi.hoisted(() => ({
  query: vi.fn()
}));

vi.mock("@anthropic-ai/claude-agent-sdk", () => ({
  query: sdkMocks.query
}));

const roleConfig: StoredRoleConfig = {
  name: "tester",
  allowedTools: ["Read", "Grep"],
  permissionMode: "dontAsk",
  systemPrompt: "Answer from the workspace.",
  maxTurns: 3
};

describe("ClaudeSdkAgentRuntime", () => {
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
    const runtime = new ClaudeSdkAgentRuntime({ roleConfig, model: "model-1" });

    await runtime.run({
      user: { id: "user-1" },
      text: "What changed?",
      workspacePath: "D:/workspace",
      sessionId: "session-1"
    });

    const expectedOptions: Record<string, unknown> = {
      cwd: "D:/workspace",
      tools: ["Read", "Grep"],
      allowedTools: ["Read", "Grep"],
      disallowedTools: [],
      permissionMode: "dontAsk",
      allowDangerouslySkipPermissions: false,
      systemPrompt: "Answer from the workspace.",
      includePartialMessages: true,
      canUseTool: expect.any(Function),
      maxTurns: 3,
      model: "model-1",
      resume: "session-1",
      settings: {
        autoCompactEnabled: true,
        autoCompactWindow: 150_000,
      },
    };

    expect(runtime.name).toBe("claude-sdk-tester");
    expect(sdkMocks.query).toHaveBeenCalledWith({
      prompt: "What changed?",
      options: expectedOptions
    });
  });

  it("grounds the prompt with the selected workspace context when available", async () => {
    sdkMocks.query.mockReturnValue(
      streamMessages({
        type: "result",
        subtype: "success",
        result: "final answer"
      })
    );
    const workspacePath = await mkdtemp(path.join(os.tmpdir(), "pets-agent-runtime-"));
    await writeFile(
      path.join(workspacePath, "CLAUDE.md"),
      "This workspace documents catalog checks and order lifecycle recording.",
      "utf8"
    );
    const runtime = new ClaudeSdkAgentRuntime({ roleConfig });

    try {
      await runtime.run({
        user: { id: "user-1" },
        text: "What is the current architecture?",
        workspacePath,
      });
    } finally {
      await rm(workspacePath, { recursive: true, force: true });
    }

    const call = firstQueryCall();
    expect(call.prompt).toContain("Selected workspace context:");
    expect(call.prompt).toContain("catalog checks and order lifecycle recording");
    expect(call.prompt).toContain("User request:\nWhat is the current architecture?");
  });

  it("does not pass mutating tools to the SDK when the role lacks edit permission mode", async () => {
    sdkMocks.query.mockReturnValue(
      streamMessages({
        type: "result",
        subtype: "success",
        result: "final answer"
      })
    );
    const runtime = new ClaudeSdkAgentRuntime({
      roleConfig: {
        name: "misconfigured-reviewer",
        allowedTools: ["Read", "Bash", "Edit", "Write"],
        permissionMode: "dontAsk",
        systemPrompt: "Read only.",
      }
    });

    await runtime.run({
      user: { id: "user-1" },
      text: "Inspect files",
      workspacePath: "D:/workspace",
    });

    const call = firstQueryCall();
    expect(call.prompt).toBe("Inspect files");
    expect(call.options["allowedTools"]).toEqual(["Read"]);
    expect(call.options["tools"]).toEqual(["Read", "Bash"]);
    expect(call.options["disallowedTools"]).toEqual(["Edit", "Write"]);
    expect(call.options["permissionMode"]).toBe("dontAsk");
    expect(call.options["allowDangerouslySkipPermissions"]).toBe(false);
  });

  it("exposes Bash but does not auto-allow it for read-only roles", async () => {
    sdkMocks.query.mockReturnValue(
      streamMessages({
        type: "result",
        subtype: "success",
        result: "final answer"
      })
    );
    const runtime = new ClaudeSdkAgentRuntime({
      roleConfig: {
        name: "reviewer",
        allowedTools: ["Read", "Bash"],
        permissionMode: "dontAsk",
        systemPrompt: "Read only.",
      }
    });

    await runtime.run({
      user: { id: "user-1" },
      text: "List files",
      workspacePath: "D:/workspace",
    });

    const call = firstQueryCall();
    expect(call.options["tools"]).toEqual(["Read", "Bash"]);
    expect(call.options["allowedTools"]).toEqual(["Read"]);
    expect(call.options["permissionMode"]).toBe("dontAsk");
  });

  it("uses the permission decider for Bash on read-only roles", async () => {
    sdkMocks.query.mockReturnValue(
      streamMessages({
        type: "result",
        subtype: "success",
        result: "final answer"
      })
    );
    const decisions: Record<string, unknown>[] = [];
    const runtime = new ClaudeSdkAgentRuntime({
      roleConfig: {
        name: "reviewer",
        allowedTools: ["Read", "Bash"],
        permissionMode: "dontAsk",
        systemPrompt: "Read only.",
      },
      toolPermissionDecider(roleConfig, toolName, input) {
        decisions.push({ roleName: roleConfig.name, toolName, input });
        return Promise.resolve({ behavior: "allow" });
      }
    });

    await runtime.run({
      user: { id: "user-1" },
      text: "List files",
      workspacePath: "D:/workspace",
    });

    const call = firstQueryCall();
    const canUseTool = call.options["canUseTool"];
    if (!isCanUseTool(canUseTool)) {
      throw new Error("Expected canUseTool callback.");
    }
    await canUseTool("Bash", { command: "ls" });

    expect(decisions).toEqual([
      { roleName: "reviewer", toolName: "Bash", input: { command: "ls" } }
    ]);
  });

  it("passes mutating tools to the SDK when the role has edit permission mode", async () => {
    sdkMocks.query.mockReturnValue(
      streamMessages({
        type: "result",
        subtype: "success",
        result: "final answer"
      })
    );
    const runtime = new ClaudeSdkAgentRuntime({
      roleConfig: {
        name: "editor",
        allowedTools: ["Read", "Edit"],
        permissionMode: "acceptEdits",
        systemPrompt: "Edit when needed.",
      }
    });

    await runtime.run({
      user: { id: "user-1" },
      text: "Edit files",
      workspacePath: "D:/workspace",
    });

    const call = firstQueryCall();
    expect(call.prompt).toBe("Edit files");
    expect(call.options["tools"]).toEqual(["Read", "Edit"]);
    expect(call.options["allowedTools"]).toEqual(["Read", "Edit"]);
    expect(call.options["disallowedTools"]).toEqual([]);
  });

  it("returns the final result, forwards stream events, and writes raw logs", async () => {
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
          type: "stream_event",
          event: {
            type: "content_block_delta",
            delta: { type: "thinking_delta", thinking: "checking files" }
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
    const rawEvents: Record<string, unknown>[] = [];
    const rawLogger: JsonlLogger = {
      filePath: "memory.jsonl",
      write(event) {
        rawEvents.push(event);
        return Promise.resolve();
      }
    };
    const streamEvents: AgentStreamEvent[] = [];
    const progressEvents: Record<string, unknown>[] = [];
    const runtime = new ClaudeSdkAgentRuntime({ roleConfig, rawLogger });

    const response = await runtime.run({
      user: { id: "user-1" },
      text: "Inspect files",
      workspacePath: "D:/workspace",
      stream: (event) => streamEvents.push(event),
      progress: (event) => {
        progressEvents.push(event);
        return Promise.resolve();
      }
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
      { type: "thinking", text: "checking files" }
    ]);
    expect(progressEvents).toEqual([
      expect.objectContaining({
        stage: "agent.tool_use_start",
        data: { toolUseId: "tool-1", toolName: "Read" }
      })
    ]);
    expect(rawEvents).toHaveLength(1);
    const rawEvent = rawEvents[0];
    expect(rawEvent).toMatchObject({
      type: "llm.response",
      runtime: "claude-sdk-tester",
      userId: "user-1",
      workspacePath: "D:/workspace",
      sessionId: "session-2",
      extractedText: "final answer",
    });
    if (rawEvent !== undefined) {
      expect(typeof rawEvent["durationMs"]).toBe("number");
    }
  });

  it("reports an error instead of forwarding unauthorized mutating tool events", async () => {
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
    const runtime = new ClaudeSdkAgentRuntime({
      roleConfig: {
        name: "reviewer",
        allowedTools: ["Read", "Edit"],
        permissionMode: "dontAsk",
        systemPrompt: "Read only.",
      }
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

  it("returns a user-facing error when the SDK result is not successful", async () => {
    sdkMocks.query.mockReturnValue(
      streamMessages({
        type: "result",
        subtype: "error",
        errors: ["permission denied"]
      })
    );
    const runtime = new ClaudeSdkAgentRuntime({ roleConfig });

    await expect(
      runtime.run({
        user: { id: "user-1" },
        text: "Edit files",
        workspacePath: "D:/workspace"
      })
    ).resolves.toEqual({ text: "Agent error: permission denied" });
  });

  it("resolves disposeSession without calling the SDK", async () => {
    const runtime = new ClaudeSdkAgentRuntime({ roleConfig });

    await expect(runtime.disposeSession()).resolves.toBeUndefined();
  });

  it("passes auto-compaction settings to the SDK from context config", async () => {
    sdkMocks.query.mockReturnValue(
      streamMessages({
        type: "result",
        subtype: "success",
        result: "done"
      })
    );
    const runtime = new ClaudeSdkAgentRuntime({
      roleConfig,
      contextConfig: {
        autoCompactEnabled: true,
        autoCompactWindow: 100_000,
        workspaceMaxChars: 6_000,
        historyMaxMessages: 30,
      },
    });

    await runtime.run({
      user: { id: "user-1" },
      text: "Test",
      workspacePath: "D:/workspace",
    });

    const call = firstQueryCall();
    expect(call.options["settings"]).toEqual({
      autoCompactEnabled: true,
      autoCompactWindow: 100_000,
    });
  });

  it("omits auto-compaction settings when disabled", async () => {
    sdkMocks.query.mockReturnValue(
      streamMessages({
        type: "result",
        subtype: "success",
        result: "done"
      })
    );
    const runtime = new ClaudeSdkAgentRuntime({
      roleConfig,
      contextConfig: {
        autoCompactEnabled: false,
        autoCompactWindow: 100_000,
        workspaceMaxChars: 6_000,
        historyMaxMessages: 30,
      },
    });

    await runtime.run({
      user: { id: "user-1" },
      text: "Test",
      workspacePath: "D:/workspace",
    });

    const call = firstQueryCall();
    expect(call.options["settings"]).toBeUndefined();
  });

  it("forwards compact_start and compact_complete stream events from system messages", async () => {
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
    const runtime = new ClaudeSdkAgentRuntime({ roleConfig });

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
            duration_ms: 800,
          },
        },
        {
          type: "result",
          subtype: "success",
          session_id: "session-2",
          result: "done",
        }
      )
    );
    const rawEvents: Record<string, unknown>[] = [];
    const rawLogger: JsonlLogger = {
      filePath: "test.jsonl",
      write(event) {
        rawEvents.push(event);
        return Promise.resolve();
      },
    };
    const runtime = new ClaudeSdkAgentRuntime({ roleConfig, rawLogger });

    await runtime.run({
      user: { id: "user-1" },
      text: "Continue",
      workspacePath: "D:/workspace",
    });

    const compactLog = rawEvents.find((e) => e["type"] === "llm.compact");
    expect(compactLog).toMatchObject({
      type: "llm.compact",
      runtime: "claude-sdk-tester",
      userId: "user-1",
      workspacePath: "D:/workspace",
      sessionId: "session-compact",
      trigger: "auto",
      preTokens: 160_000,
      durationMs: 800,
    });
  });

  it("extracts context usage from the SDK result and returns it in the response", async () => {
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
          cache_read_input_tokens: 80_000,
        },
      })
    );
    const runtime = new ClaudeSdkAgentRuntime({ roleConfig });

    const response = await runtime.run({
      user: { id: "user-1" },
      text: "Test",
      workspacePath: "D:/workspace",
    });

    expect(response.contextUsage).toEqual({
      inputTokens: 120_000,
      outputTokens: 500,
      cacheCreationTokens: 30_000,
      cacheReadTokens: 80_000,
      contextWindow: 150_000,
      usagePercent: 80,
    });
  });

  it("omits context usage when the SDK result has no usage field", async () => {
    sdkMocks.query.mockReturnValue(
      streamMessages({
        type: "result",
        subtype: "success",
        result: "answer",
      })
    );
    const runtime = new ClaudeSdkAgentRuntime({ roleConfig });

    const response = await runtime.run({
      user: { id: "user-1" },
      text: "Test",
      workspacePath: "D:/workspace",
    });

    expect(response.contextUsage).toBeUndefined();
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
