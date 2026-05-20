import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import type { Message } from "@anthropic-ai/sdk/resources/messages";
import { describe, expect, it } from "vitest";
import { AnthropicCompatibleAgentRuntime } from "./anthropicCompatibleAgentRuntime.js";

describe("AnthropicCompatibleAgentRuntime", () => {
  it("sends prior conversation history before the current user message", async () => {
    const workspacePath = await createWorkspace();
    const requests: unknown[] = [];
    const runtime = new AnthropicCompatibleAgentRuntime({
      baseUrl: "https://example.test",
      apiKey: "secret",
      modelId: "test-model",
      client: {
        messages: {
          create(params) {
            requests.push(params);
            return Promise.resolve({
              id: "msg_1",
              type: "message",
              role: "assistant",
              model: "test-model",
              container: null,
              content: [{ type: "text", text: "second answer" }],
              stop_details: null,
              stop_reason: "end_turn",
              stop_sequence: null,
              usage: {
                input_tokens: 1,
                output_tokens: 1
              }
            } as Message);
          }
        }
      }
    });

    const response = await runtime.run({
      user: { id: "user-1" },
      text: "second question",
      workspacePath,
      history: [
        { role: "user", content: "first question" },
        { role: "assistant", content: "first answer" }
      ]
    });

    expect(response.text).toBe("second answer");
    expect(requests).toHaveLength(1);
    expect(requests[0]).toMatchObject({
      messages: [
        { role: "user", content: "first question" },
        { role: "assistant", content: "first answer" },
        { role: "user", content: "second question" }
      ]
    });
  });
});

async function createWorkspace(): Promise<string> {
  const root = await mkdtemp(path.join(tmpdir(), "anthropic-compatible-runtime-"));
  await writeFile(path.join(root, "CLAUDE.md"), "Project assistant instructions.", "utf8");
  return root;
}
