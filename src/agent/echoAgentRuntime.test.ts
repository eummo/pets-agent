import { describe, expect, it } from "vitest";
import { EchoAgentRuntime } from "./echoAgentRuntime.js";

describe("EchoAgentRuntime", () => {
  it("returns a formatted echo response with workspace, user, and input", async () => {
    const runtime = new EchoAgentRuntime();
    const response = await runtime.run({
      user: { id: "dev-1" },
      text: "hello world",
      workspacePath: "/path/to/workspace"
    });

    expect(response.text).toContain("开发 harness 已连接");
    expect(response.text).toContain("/path/to/workspace");
    expect(response.text).toContain("dev-1");
    expect(response.text).toContain("hello world");
  });

  it("exposes the correct name", () => {
    const runtime = new EchoAgentRuntime();

    expect(runtime.name).toBe("echo");
  });

  it("resolves disposeSession without error", async () => {
    const runtime = new EchoAgentRuntime();

    await expect(runtime.disposeSession()).resolves.toBeUndefined();
  });
});
