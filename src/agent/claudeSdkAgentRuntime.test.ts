import { beforeEach, describe, expect, it, vi } from "vitest";
import type { AgentStreamEvent } from "../core/ports.js";
import type { JsonlLogger } from "../logging/jsonlLogger.js";
import { ClaudeSdkAgentRuntime, type RoleConfig } from "./claudeSdkAgentRuntime.js";

const sdkMocks = vi.hoisted(() => ({
  query: vi.fn()
}));

vi.mock("@anthropic-ai/claude-agent-sdk", () => ({
  query: sdkMocks.query
}));

const roleConfig: RoleConfig = {
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
      allowedTools: ["Read", "Grep"],
      permissionMode: "dontAsk",
      allowDangerouslySkipPermissions: false,
      systemPrompt: "Answer from the workspace.",
      includePartialMessages: true,
      maxTurns: 3,
      model: "model-1",
      resume: "session-1"
    };

    expect(runtime.name).toBe("claude-sdk-tester");
    expect(sdkMocks.query).toHaveBeenCalledWith({
      prompt: "What changed?",
      options: expectedOptions
    });
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
    expect(rawEvents).toEqual([
      {
        type: "llm.response",
        runtime: "claude-sdk-tester",
        userId: "user-1",
        workspacePath: "D:/workspace",
        sessionId: "session-2",
        extractedText: "final answer"
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
