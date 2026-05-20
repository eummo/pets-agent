import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import type { Message } from "@anthropic-ai/sdk/resources/messages";
import { describe, expect, it } from "vitest";
import { AnthropicCodeChangeRuntime } from "./anthropicCodeChangeRuntime.js";

describe("AnthropicCodeChangeRuntime", () => {
  it("calls the SDK, applies returned file changes, and runs verification", async () => {
    const workspacePath = await mkdtemp(path.join(tmpdir(), "pets-agent-sdk-code-change-"));
    await writeFile(path.join(workspacePath, "package.json"), JSON.stringify({
      type: "module",
      scripts: {
        test: "node -e \"process.exit(0)\""
      }
    }));
    await writeFile(path.join(workspacePath, "index.ts"), "export const before = true;\n");
    const requests: unknown[] = [];
    const runtime = new AnthropicCodeChangeRuntime({
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
              content: [
                {
                  type: "text",
                  text: JSON.stringify({
                    summary: "Refactored the order lifecycle entry point.",
                    changes: [
                      {
                        path: "index.ts",
                        content: "export const after = true;\n"
                      }
                    ],
                    verificationCommand: "npm test"
                  })
                }
              ],
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
      user: { id: "developer-1" },
      text: "重构订单系统",
      workspacePath
    });

    expect(requests).toHaveLength(1);
    expect(await readFile(path.join(workspacePath, "index.ts"), "utf8")).toBe("export const after = true;\n");
    expect(response.text).toContain("Claude/Anthropic SDK");
    expect(response.text).toContain("通过");
  });

  it("rejects model changes outside the selected workspace", async () => {
    const workspacePath = await mkdtemp(path.join(tmpdir(), "pets-agent-sdk-code-change-"));
    await writeFile(path.join(workspacePath, "package.json"), JSON.stringify({ scripts: { test: "node -e \"process.exit(0)\"" } }));
    const runtime = new AnthropicCodeChangeRuntime({
      baseUrl: "https://example.test",
      apiKey: "secret",
      modelId: "test-model",
      client: {
        messages: {
          create() {
            return Promise.resolve({
              id: "msg_1",
              type: "message",
              role: "assistant",
              model: "test-model",
              container: null,
              content: [
                {
                  type: "text",
                  text: JSON.stringify({
                    summary: "bad",
                    changes: [{ path: "../outside.ts", content: "bad" }],
                    verificationCommand: "npm test"
                  })
                }
              ],
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

    await expect(
      runtime.run({
        user: { id: "developer-1" },
        text: "重构订单系统",
        workspacePath
      })
    ).rejects.toThrow("Refusing to write outside workspace");
  });
});
